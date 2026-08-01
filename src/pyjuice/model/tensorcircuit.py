from __future__ import annotations

import math
import torch
import torch.nn as nn
import time
import triton
import triton.language as tl
from tqdm import tqdm
from functools import partial
from typing import Optional, Sequence, Callable, Union, Tuple, Dict
from contextlib import contextmanager

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes, ExternalParamsSumNodes, \
                          foreach, summate, multiply
from pyjuice.layer import Layer, InputLayer, ProdLayer, SumLayer, ExternalParamsSumLayer, LayerGroup, \
                          StagedExternalParams, EXTERNAL_PARAMS_KWARG, EXTERNAL_PARAMS_GRAD_KWARG, \
                          EXTERNAL_PARAMS_BUFFER_KWARG, EXTERNAL_PARAMS_GRAD_BUFFER_KWARG
from pyjuice.layer.external_sum_layer import validate_external_tensors
from pyjuice.utils.grad_fns import ReverseGrad
from pyjuice.utils import BitSet

from .backend import compile_cum_par_flows_fn, compute_cum_par_flows, cum_par_flows_to_device, \
                     compile_par_update_fn, em_par_update, par_update_to_device, \
                     normalize_parameters, eval_top_down_probs


def _pc_model_backward_hook(grad, pc, inputs, record_cudagraph, apply_cudagraph, propagation_alg, **kwargs):
    grad = grad.permute(1, 0)
    pc.backward(
        inputs = inputs,
        ll_weights = grad / grad.sum() * grad.size(1),
        compute_param_flows = pc._optim_hyperparams["compute_param_flows"], 
        flows_memory = pc._optim_hyperparams["flows_memory"],
        record_cudagraph = record_cudagraph,
        apply_cudagraph = apply_cudagraph,
        propagation_alg = propagation_alg,
        **kwargs
    )

    return None


def layer_iterator(pc, reverse = False, ret_layer_groups = False, ignore_input_layers = False):
    if not reverse:
        if ret_layer_groups:
            if not ignore_input_layers:
                yield pc.input_layer_group

            for layer_group in pc.inner_layer_groups:
                yield layer_group

        else:
            if not ignore_input_layers:
                for layer in pc.input_layer_group:
                    yield layer

            for layer_group in pc.inner_layer_groups:
                for layer in layer_group:
                    yield layer
    else:
        if ret_layer_groups:
            for layer_group in pc.inner_layer_groups[::-1]:
                yield layer_group

            if not ignore_input_layers:
                yield pc.input_layer_group

        else:
            for layer_group in pc.inner_layer_groups[::-1]:
                for layer in layer_group:
                    yield layer

            if not ignore_input_layers:
                for layer in pc.input_layer_group:
                    yield layer


@contextmanager
def device_grad_controller(device, no_grad = True):
    device_type = device.type
    if device_type == "cpu":
        if no_grad:
            with torch.no_grad():
                yield
        else:
            yield
    else:
        # Pass the `torch.device` straight through: it handles an index-less CUDA device
        # (`torch.device("cuda")`, `.index is None`) by selecting the current device, whereas
        # `f"cuda:{device.index}"` would build the invalid string "cuda:None".
        with torch.cuda.device(device):
            if no_grad:
                with torch.no_grad():
                    yield
            else:
                yield


from pyjuice.nodes.external_params.external_params import ExternalSumParams


def _staged_copy(plain_d: list, plain_s: list, fast: list) -> None:
    """
    Issue the staging copies: a tiled transpose for the pairs that are one, a batched `copy_` for the
    rest.

    `Tensor.copy_` on a transposed view goes through TensorIterator, which handles arbitrary strides but
    does not tile, so one side of the transpose is uncoalesced -- 8.2 us against 2.1 us for the same
    4 MB. Staging is worth optimizing because it was 37-59% of the whole cost of applying a gate at
    batch 256.

    Which pairs qualify is decided by the CALLER from the parameterization's declared layout, not by
    inspecting strides here: this runs on every forward, and building a permuted view per tensor just to
    test it cost more Python than the kernel saved in GPU time.
    """
    if fast:
        from pyjuice.nodes.external_params.kernels.c import get_module
        mod = get_module()
        if mod is None:
            for dst, src in fast:                     # no extension: fall back, still correct
                plain_d.append(dst)
                plain_s.append(src.permute(*range(1, src.dim()), 0))
        else:
            for dst, src in fast:
                batch = src.size(0)
                mod.staging_transpose(dst, src, batch, src.numel() // max(batch, 1))

    if plain_d:
        torch._foreach_copy_(plain_d, plain_s)


class TensorCircuit(nn.Module):
    """
    A class for compiled PCs. It is a subclass of `torch.nn.Module`.

    :param root_ns: the root node of the PC's DAG
    :type root_ns: CircuitNodes

    :param layer_sparsity_tol: the maximum allowed fraction for added pseudo edges within every layer (better to set to a small number for sparse/block-sparse PCs)
    :type layer_sparsity_tol: float

    :param max_num_partitions: maximum number of partitions in a layer
    :type max_num_partitions: Optional[int]

    :param disable_gpu_compilation: force PyJuice to use CPU compilation
    :type disable_gpu_compilation: bool

    :param force_gpu_compilation: force PyJuice to use GPU compilation
    :type force_gpu_compilation: bool

    :param max_tied_ns_per_parflow_block: how many groups of tied parameters are allowed to share the same flow/gradient accumulator (higher values -> consumes less GPU memory; lower values -> potentially avoid stalls caused by atomic operations)

        :note: the default is high (32) because the memory/locality side of that trade dominates on
               current hardware -- splitting tied groups across accumulators multiplies the flow buffer,
               and the extra traffic costs more than the atomic contention it avoids. Measured on a
               homogeneous HMM (seq 32, 1024 latents, 126464 cats, top-k 1024 soft evidence), going from
               4 accumulator blocks to 1: `param_flows` 4.14 -> 1.04 GB, backward 16.8 -> 13.4 ms and
               `init_param_flows` 5.7 -> 1.4 ms (1.41x on the step), with parameters after an EM step
               agreeing to ~2e-5 relative -- flow-accumulation reassociation only. For scale, on the same
               GPU 268M `atomic_add`s all targeting ONE address cost 3.7 ms, versus 44 ms for the same
               count scattered over a 4 GB table: contention is the cheap side of the trade. Lower it
               only for a model where atomic stalls are demonstrably the bottleneck.

    :type max_tied_ns_per_parflow_block: int

    :param verbose: Whether to display the progress of the compilation
    :type verbose: bool
    """

    def __init__(self, root_ns: CircuitNodes, layer_sparsity_tol: float = 0.5, 
                 max_num_partitions: Optional[int] = None, disable_gpu_compilation: bool = False, 
                 force_gpu_compilation: bool = False,
                 max_tied_ns_per_parflow_block: int = 32,
                 device: Optional[Union[int,torch.device]] = None,
                 verbose: bool = True) -> None:

        super(TensorCircuit, self).__init__()

        assert isinstance(root_ns, CircuitNodes), "`root_ns` should be an instance of `CircuitNodes`."

        self.root_ns = root_ns
        self.device = torch.device("cpu")

        self.num_vars = self._get_num_vars(self.root_ns)

        self.node_mars = None
        self.element_mars = None
        self.node_flows = None
        self.element_flows = None
        self.param_flows = None
        self.node_mars_tempered = None

        # Staging buffers for externally supplied per-sample sum parameters, and for the per-sample
        # gradients returned for them. Like `node_mars`, they are flat and (re)allocated by
        # `_init_buffer` whenever the batch size changes.
        self.external_params = None
        self.external_params_grad = None
        
        self._init_layers(
            layer_sparsity_tol = layer_sparsity_tol, 
            max_num_partitions = max_num_partitions, 
            disable_gpu_compilation = disable_gpu_compilation, 
            force_gpu_compilation = force_gpu_compilation,
            max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block,
            device = device,
            verbose = verbose
        )
        
        # Hyperparameters for backward pass
        self._optim_hyperparams = {
            "compute_param_flows": True,
            "flows_memory": 1.0
        }

        # Partial evaluation
        self._fw_partial_eval_enabled = False
        self._bk_partial_eval_enabled = False

        # CudaGraph options
        self._recorded_cuda_graphs = dict()

        # Mode for forward and backward pass
        self.default_propagation_alg = "LL" # Could be "LL", "MPE", or "GeneralLL"
        self.propagation_alg_kwargs = dict()

        # Running parameters
        self._run_params = dict()

        # Cumulative flows
        self._cum_flow = 0.0

    def to(self, device):
        # Normalize to a `torch.device` so all downstream consumers (which rely on `self.device.type` /
        # `self.device.index`) work uniformly. Accept an int ordinal (`pc.to(0)`), a string
        # (`pc.to("cuda:0")` / `pc.to("cpu")`), or a `torch.device`.
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        else:
            device = torch.device(device)

        super(TensorCircuit, self).to(device)

        self.input_layer_group.to(device)

        self.device = device

        # For parameter flow accumulation
        self.parflow_fusing_kwargs = cum_par_flows_to_device(self.parflow_fusing_kwargs, device)
        
        # For parameter update
        self.par_update_kwargs = par_update_to_device(self.par_update_kwargs, device)

        return self

    def set_propagation_alg(self, propagation_alg: str, **kwargs):
        """
        Set the default propagation algorithm used by :func:`forward` and :func:`backward`.

        :param propagation_alg: the propagation algorithm; one of `"LL"` (log-likelihood / standard
            marginal inference), `"MPE"` (most-probable-explanation / max-product), or `"GeneralLL"`
            (an entropy-/temperature-style generalization that interpolates between `"LL"` and `"MPE"`)
        :type propagation_alg: str

        For `"GeneralLL"`, an `alpha` keyword argument must be provided. The algorithm can also be
        selected per-call by passing `propagation_alg=...` directly to :func:`forward`/:func:`backward`.
        """
        if propagation_alg == "LL":
            self.default_propagation_alg = "LL"
            self.propagation_alg_kwargs.clear()
        elif propagation_alg == "MPE":
            self.default_propagation_alg = "MPE"
            self.propagation_alg_kwargs.clear()
        elif propagation_alg == "GeneralLL":
            assert "alpha" in kwargs, "Argument `alpha` should be provided for the `GeneralLL` propagation algorithm."
            self.default_propagation_alg = "GeneralLL"
            self.propagation_alg_kwargs.clear()
            self.propagation_alg_kwargs["alpha"] = kwargs["alpha"]
        else:
            raise NotImplementedError(f"Unknown propagation algorithm {propagation_alg}.")
        
    def forward(self, inputs: torch.Tensor, input_layer_fn: Optional[Union[str,Callable]] = None,
                cache: Optional[dict] = None, return_cache: bool = False, record_cudagraph: bool = False, 
                apply_cudagraph: bool = True, force_use_bf16: bool = False, force_use_fp32: bool = False, 
                propagation_alg: Optional[Union[str,Sequence[str]]] = None, pflow_temperature: float = 1.0, 
                _inner_layers_only: bool = False, _no_buffer_reset: bool = False, **kwargs):
        """
        Forward evaluation of the PC.

        :param inputs: input tensor of size `[B, num_vars]`
        :type inputs: torch.Tensor

        :param input_layer_fn: Custom forward function for input layers; if it is a string, then try to call the corresponding member function of the input layers
        :type input_layer_fn: Optional[Union[str,Callable]]
        """

        with device_grad_controller(device = self.device, no_grad = True):
        
            B = inputs.size(0)

            origin_inputs = inputs
            if input_layer_fn is None:
                assert inputs.dim() == 2

                inputs = inputs.permute(1, 0)

            # Set propagation algorithm
            if propagation_alg is None:
                propagation_alg = self.default_propagation_alg
                kwargs.update(self.propagation_alg_kwargs)

            # Tempered param flow
            pflow_tempered_enabled = abs(pflow_temperature - 1.0) >= 1e-6

            self._check_external_params_kwargs(kwargs)
            self._stage_external_params(kwargs, B)

            ## Initialize buffers for forward pass ##

            if not _no_buffer_reset:
                self._init_buffer(name = "node_mars", shape = (self.num_nodes, B), set_value = 0.0)
                self._init_buffer(name = "element_mars", shape = (self.num_elements, B), set_value = -torch.inf)
                if pflow_tempered_enabled:
                    self._init_buffer(name = "node_mars_tempered", shape = (self.num_nodes, B), set_value = 0.0)
                    kwargs["node_mars_tempered"] = self.node_mars_tempered

            # Load cached node marginals
            if self._buffer_matches(name = "node_mars", cache = cache):
                self.node_mars[:,:] = cache["node_mars"]

            if pflow_tempered_enabled and self._buffer_matches(name = "node_mars_tempered", cache = cache):
                self.node_mars_tempered[:,:] = cache["node_mars_tempered"]
                kwargs["node_mars_tempered"] = self.node_mars_tempered

            ## Run forward pass ##

            # Input layers
            if not _inner_layers_only:
                for idx, layer in enumerate(self.input_layer_group):
                    if input_layer_fn is None:
                        layer(inputs, self.node_mars, **kwargs)

                    elif isinstance(input_layer_fn, str):
                        assert hasattr(layer, input_layer_fn), f"Custom input function `{input_layer_fn}` not found for layer type {type(layer)}."
                        getattr(layer, input_layer_fn)(inputs, self.node_mars, **kwargs)

                    elif isinstance(input_layer_fn, Callable):
                        ret = input_layer_fn(layer, inputs, self.node_mars, **kwargs)

                        # If the layer is not handled by `input_layer_fn`, we assume it will return `False`
                        if not ret and ret is not None:
                            layer(inputs, self.node_mars, **kwargs)

                    else:
                        raise ValueError(f"Custom input function should be either a `str` or a `Callable`. Found {type(input_layer_fn)} instead.")

            # Inner layers
            def _run_inner_layers():
                for layer_id, layer_group in enumerate(self.inner_layer_groups):
                    if layer_group.is_prod():
                        # Prod layer
                        layer_group(self.node_mars, self.element_mars)

                    elif layer_group.is_sum():
                        # Sum layer
                        layer_group(self.node_mars, self.element_mars, self.params, 
                                    force_use_bf16 = force_use_bf16,
                                    force_use_fp32 = force_use_fp32, 
                                    propagation_alg = propagation_alg if isinstance(propagation_alg, str) else propagation_alg[layer_id], 
                                    pflow_temperature = pflow_temperature,
                                    **kwargs)

                    else:
                        raise ValueError(f"Unknown layer type {type(layer)}.")

            # `external_params` is in the signature because the staging buffer is re-allocated when its
            # layout changes, and a captured graph holds the old pointer
            signature = (0, id(self.node_mars), id(self.element_mars), id(self.params), B,
                         id(self.external_params))
            if record_cudagraph and signature not in self._recorded_cuda_graphs:
                # Warmup
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _run_inner_layers()
                torch.cuda.current_stream().wait_stream(s)

                # Capture
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _run_inner_layers()

                # Save
                self._recorded_cuda_graphs[signature] = g

            if apply_cudagraph and signature in self._recorded_cuda_graphs:
                g = self._recorded_cuda_graphs[signature]
                g.replay()
            else:
                _run_inner_layers()
                
            lls = self.node_mars[self._root_node_range[0]:self._root_node_range[1],:]
            lls = lls.permute(1, 0)

        ## Create/Update cache if needed ##

        if return_cache:
            if cache is None:
                cache = dict()

            with torch.no_grad():
                cache["node_mars"] = self.node_mars.clone()

        ## Add gradient hook for backward pass ##

        if torch.is_grad_enabled():
            lls.requires_grad = True
            lls.register_hook(
                partial(
                    _pc_model_backward_hook, 
                    pc = self, 
                    inputs = origin_inputs, 
                    record_cudagraph = record_cudagraph, 
                    apply_cudagraph = apply_cudagraph,
                    propagation_alg = propagation_alg,
                    **kwargs
                )
            )

        if return_cache:
            return lls.clone(), cache
        else:
            return lls.clone()

    def backward(self, inputs: Optional[torch.Tensor] = None, 
                 ll_weights: Optional[torch.Tensor] = None,
                 compute_param_flows: bool = True, 
                 flows_memory: float = 1.0,
                 input_layer_fn: Optional[Union[str,Callable]] = None,
                 cache: Optional[dict] = None,
                 sum_layer_pre_backward_callback: Optional[Callable] = None,
                 sum_layer_post_backward_callback: Optional[Callable] = None,
                 return_cache: bool = False,
                 record_cudagraph: bool = False, 
                 apply_cudagraph: bool = True,
                 allow_modify_flows: bool = False,
                 propagation_alg: Union[str,Sequence[str]] = "LL",
                 logspace_flows: bool = True,
                 negate_pflows: bool = False,
                 _inner_layers_only: bool = False,
                 _disable_buffer_init: bool = False,
                 force_use_fp32: bool = False,
                 pflow_temperature: float = 1.0,
                 temper_eflow: bool = False,
                 compute_external_grads: bool = True,
                 **kwargs):
        """
        Backward evaluation of the PC that computes node flows as well as parameter flows.

        :param inputs: input tensor of size `[B, num_vars]`
        :type inputs: torch.Tensor

        :param ll_weights: weights of the log-likelihoods of size [B] or [num_roots, B]
        :type ll_weights: torch.Tensor
        
        :param input_layer_fn: Custom forward function for input layers; if it is a string, then try to call the corresponding member function of the input layers
        :type input_layer_fn: Optional[Union[str,Callable]]
        """

        self._run_params["allow_modify_flows"] = allow_modify_flows
        self._run_params["propagation_alg"] = propagation_alg
        self._run_params["logspace_flows"] = logspace_flows
        self._run_params["negate_pflows"] = negate_pflows
        self._run_params["force_use_fp32"] = force_use_fp32
        self._run_params["pflow_temperature"] = pflow_temperature
        self._run_params["temper_eflow"] = temper_eflow

        assert self.node_mars is not None and self.element_mars is not None, "Should run forward path first."
        if input_layer_fn is None:
            assert inputs.dim() == 2 and inputs.size(1) == self.num_vars
            inputs = inputs.permute(1, 0)

        with device_grad_controller(device = self.device, no_grad = True):

            B = self.node_mars.size(1)

            ## Initialize buffers for backward pass ##

            if not _disable_buffer_init:
                self._init_buffer(name = "node_flows", shape = (self.num_nodes, B), set_value = 0.0 if not logspace_flows else -float("inf"))
                self._init_buffer(name = "element_flows", shape = (self.num_elements, B), set_value = 0.0 if not logspace_flows else -float("inf"))

            # Tempered pflows
            if abs(pflow_temperature - 1.0) >= 1e-6:
                assert hasattr(self, "node_mars_tempered")
                kwargs["node_mars_tempered"] = self.node_mars_tempered

            # Set root node flows
            def _set_root_node_flows():
                nonlocal ll_weights
                nonlocal logspace_flows
                if ll_weights is None:
                    root_flows = 1.0 if not logspace_flows else 0.0
                    self.node_flows[self._root_node_range[0]:self._root_node_range[1],:] = root_flows
                else:
                    if ll_weights.dim() == 1:
                        ll_weights = ll_weights.unsqueeze(0)

                    assert ll_weights.size(0) == self.num_root_nodes

                    root_flows = ll_weights if not logspace_flows else ll_weights.log()
                    self.node_flows[self._root_node_range[0]:self._root_node_range[1],:] = root_flows

            _set_root_node_flows()

            # Accumulate the total amount of flows added to the PC
            if compute_param_flows:
                if ll_weights is None:
                    self._cum_flow += (self._root_node_range[1] - self._root_node_range[0]) * B
                else:
                    self._cum_flow += ll_weights.sum().item()

            # Load cached node flows
            if self._buffer_matches(name = "node_flows", cache = cache):
                self.node_flows[:,:] = cache["node_flows"]

            ## Initialize parameter flows ##
            if compute_param_flows:
                self.init_param_flows(flows_memory = flows_memory)

            ## External parameters: reuse what the forward staged; reset the gradient buffers ##
            self._check_external_params_kwargs(kwargs)
            self._resolve_backward_external_params(kwargs)
            self._init_external_params_grads(kwargs, B, compute_external_grads)

            ## Run backward pass ##

            # Inner layers
            def _run_inner_layers():

                # Backward pass for inner layers
                for layer_id in range(len(self.inner_layer_groups) - 1, -1, -1):
                    layer_group = self.inner_layer_groups[layer_id]

                    if layer_group.is_prod():
                        # Prod layer
                        layer_group.backward(self.node_flows, self.element_flows, logspace_flows = logspace_flows)

                    elif layer_group.is_sum():
                        # Sum layer

                        # First recompute the previous product layer
                        self.inner_layer_groups[layer_id-1].forward(self.node_mars, self.element_mars, _for_backward = True)

                        # Execute pre-backward callback
                        layer_group.callback(
                            sum_layer_pre_backward_callback, 
                            node_flows = self.node_flows,
                            element_flows = self.element_flows,
                            node_mars = self.node_mars,
                            element_mars = self.element_mars,
                            params = self.params,
                            param_flows = self.param_flows if hasattr(self, "param_flows") else None
                        )

                        # Backward sum layer
                        layer_group.backward(self.node_flows, self.element_flows, self.node_mars, self.element_mars, self.params, 
                                             param_flows = self.param_flows if compute_param_flows else None,
                                             allow_modify_flows = allow_modify_flows, 
                                             propagation_alg = propagation_alg if isinstance(propagation_alg, str) else propagation_alg[layer_id], 
                                             logspace_flows = logspace_flows, negate_pflows = negate_pflows, force_use_fp32 = force_use_fp32, 
                                             pflow_temperature = pflow_temperature, temper_eflow = temper_eflow, **kwargs)

                        # Execute post-backward callback
                        layer_group.callback(
                            sum_layer_post_backward_callback, 
                            node_flows = self.node_flows,
                            element_flows = self.element_flows,
                            node_mars = self.node_mars,
                            element_mars = self.element_mars,
                            params = self.params,
                            param_flows = self.param_flows if hasattr(self, "param_flows") else None
                        )

                    else:
                        raise ValueError(f"Unknown layer type {type(layer)}.")

            signature = (1, id(self.node_flows), id(self.element_flows), id(self.node_mars), id(self.element_mars), id(self.params), id(self.param_flows), B, 
                         allow_modify_flows, logspace_flows, ((abs(pflow_temperature) - 1.0) < 1e-6), temper_eflow,
                         id(self.external_params), id(self.external_params_grad))
            if record_cudagraph and signature not in self._recorded_cuda_graphs:
                # Warmup
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self.node_flows[:,:] = 0.0
                        _set_root_node_flows()
                        _run_inner_layers()
                torch.cuda.current_stream().wait_stream(s)

                # Capture
                self.node_flows[:,:] = 0.0
                _set_root_node_flows()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _run_inner_layers()

                # Save
                self._recorded_cuda_graphs[signature] = g

            if apply_cudagraph and signature in self._recorded_cuda_graphs:
                g = self._recorded_cuda_graphs[signature]
                g.replay()
            else:
                _run_inner_layers()

            # Compute backward pass for all input layers
            if not _inner_layers_only:
                for idx, layer in enumerate(self.input_layer_group):
                    if input_layer_fn is None:
                        layer.backward(inputs, self.node_flows, self.node_mars, logspace_flows = logspace_flows, **kwargs)

                    elif isinstance(input_layer_fn, str):
                        assert hasattr(layer, input_layer_fn), f"Custom input function `{input_layer_fn}` not found for layer type {type(layer)}."
                        getattr(layer, input_layer_fn)(inputs, self.node_flows, self.node_mars, logspace_flows = logspace_flows, **kwargs)

                    elif isinstance(input_layer_fn, Callable):
                        ret = input_layer_fn(layer, inputs, self.node_flows, self.node_mars, logspace_flows = logspace_flows, **kwargs)

                        # If the layer is not handled by `input_layer_fn`, we assume it will return `False`
                        if not ret and ret is not None:
                            layer.backward(inputs, self.node_flows, self.node_mars, logspace_flows = logspace_flows, **kwargs)

                    else:
                        raise ValueError(f"Custom input function should be either a `str` or a `Callable`. Found {type(input_layer_fn)} instead.")

        self._write_external_params_grads()

        if return_cache:
            if cache is None:
                cache = dict()

            with torch.no_grad():
                cache["node_flows"] = self.node_flows.clone()

            return cache
        else:
            return None

    def _write_external_params_grads(self) -> None:
        """
        Copy the gradients into the buffers the caller supplied to `sum_external_params_grad`.

        The kernels write the PC's internal buffer, which is in STORAGE layout (batch innermost); the
        caller's tensors are in the layout they supplied the gates in. `from_storage` is what maps
        between the two, so this is the exact inverse of what staging does on the forward, batched into
        one `_foreach_copy_` rather than one op per node.
        """
        dsts = getattr(self, "_external_grad_dsts", None)
        if not dsts:
            return None

        d, s = [], []
        for ns, tensors in dsts.items():
            src = ns.external_params.from_storage(ns, self._staged_external_params_grad[ns])
            if torch.is_tensor(tensors):
                tensors = (tensors,)
            for dst, one in zip(tensors, src):
                d.append(dst)
                s.append(one)

        if d:
            torch._foreach_copy_(d, s)

        self._external_grad_dsts = None

    def _external_params_layout(self, batch_size: int):
        """
        Slot layout of the external-parameter staging buffer at `batch_size`.

        Every node that takes external parameters gets its own contiguous slot for each tensor its
        parameterization declares, laid out in `external_params_nodes` order. The shapes come from the
        descriptor, so the layout is whatever the parameterization says it is; the buffer just gives
        every tensor a known, contiguous home that the kernels can address and that keeps a stable
        pointer across calls (which is what lets the caller pass freshly allocated tensors, and lets
        the layers run under CUDA-graph capture).

        Cached per batch size, since that is the only thing the layout depends on.

        :returns: `(total_numel, {ns: [(offset, shape), ...]})`
        """
        layout = self._external_params_layouts.get(batch_size, None)
        if layout is not None:
            return layout

        # Two passes so that a node may share another node's slots (see
        # `ExternalSumParams.storage_owner`): allocate for the owners first, then alias the rest onto
        # them. Aliased nodes still appear in the map, so the layers and kernels need no special case --
        # they look up their own `ns` and simply find the same memory.
        offset = 0
        ns2slots = dict()
        for ns in self.external_params_nodes:
            if ns.external_params.storage_owner(ns) is not ns:
                continue

            slots = []
            for shape in ns.external_params.storage_shapes(ns, batch_size):
                slots.append((offset, tuple(shape)))
                offset += math.prod(shape)

            ns2slots[ns] = slots

        for ns in self.external_params_nodes:
            owner = ns.external_params.storage_owner(ns)
            if owner is not ns:
                assert owner in ns2slots, \
                    f"`storage_owner` of {ns} is not itself an external-parameter node of this circuit."
                ns2slots[ns] = ns2slots[owner]

        layout = (offset, ns2slots)
        self._external_params_layouts[batch_size] = layout

        return layout

    def _external_params_views(self, name: str, batch_size: int, set_value: Optional[float] = None):
        """
        (Re)allocate the staging buffer `name` for `batch_size` and return one contiguous view per
        tensor slot, as `{ns: (view, ...)}`.

        The views are what the layers and kernels consume, so a parameterization never has to know
        that a shared buffer exists -- it just receives correctly shaped, contiguous, correctly placed
        tensors.
        """
        total_numel, ns2slots = self._external_params_layout(batch_size)

        self._init_buffer(name = name, shape = (total_numel,), set_value = set_value)

        buffer = self.__dict__[name]

        # Building these is pure Python -- one slice and one view per node -- and it runs on every
        # forward. On a 32-timestep gated HMM it was 67 us of the 111 us the whole staging cost, six
        # times what the copy itself takes, so it is cached. The views depend only on the layout (fixed
        # per batch size) and the buffer's identity, so the check is `is` against the tensor the
        # allocation just returned: a reallocation -- a new batch size, a `.to(device)` -- hands back a
        # different object and rebuilds. `_init_buffer` still runs every time, since it is also what
        # zeroes the gradient buffer.
        #
        # The returned dict is SHARED between calls. Both callers copy it into a `StagedExternalParams`
        # before deleting entries, so nothing mutates it; a new caller must do the same.
        key = (name, batch_size)
        cached = self._external_params_views_cache.get(key)
        if cached is not None and cached[0] is buffer:
            return cached[1]

        views = {ns: tuple(buffer[offset:offset + math.prod(shape)].view(shape)
                           for offset, shape in slots)
                 for ns, slots in ns2slots.items()}
        self._external_params_views_cache[key] = (buffer, views)
        return views

    def register_external_params_group(self, name: str, nodes: Sequence[CircuitNodes],
                                       dim: int = 1) -> None:
        """
        Address several nodes' external parameters with ONE concatenated tensor, under a name.

        Nodes that share a parameterization usually share a shape too -- an HMM's per-timestep
        transitions being the case this exists for -- and a head producing them naturally emits one
        tensor already. Without a group the caller has to slice that tensor apart before every call,
        once per node, only for the PC to copy each piece separately.

        Once registered, the name is usable wherever an `ns` key is:

        .. code-block:: python

            pc.register_external_params_group("router", [t0, t1, t2])
            pc(x, sum_external_params = {"router": phi})        # phi: [B, 3 * Nk, Ck]

        and may be mixed freely with per-node entries in the same dict.

        Members are split along `dim` by their OWN extents, so they need not have equal shapes there --
        only on every other axis. Nothing is copied: each member receives a view.

        :note: order is significant. `nodes[i]` receives the i-th slice, so the list is the contract.

        :note: for a parameterization with several tensors per node (low-rank factors, say) the value
               is a tuple of concatenated tensors, one per slot, each split the same way.

        :param name: the key to use in `sum_external_params`
        :type name: str

        :param nodes: the nodes this group addresses, in slice order
        :type nodes: Sequence[CircuitNodes]

        :param dim: axis of the concatenation. Defaults to 1, the first non-batch axis; `0` is the
                    batch and is rejected.
        :type dim: int
        """
        assert isinstance(name, str) and name != "", "An external-parameter group needs a non-empty name."
        assert name not in self.external_params_groups, \
            f"External-parameter group '{name}' is already registered."
        assert len(nodes) > 0, f"External-parameter group '{name}' is empty."
        assert dim != 0, \
            "`dim = 0` is the batch axis; a group concatenates along a PARAMETER axis (default 1)."

        seen = set()
        for ns in nodes:
            assert ns in self.external_params_nodes, \
                f"External-parameter group '{name}' contains a node this PC did not compile with an " \
                f"external parameterization. Construct it with `pyjuice.summate(..., external_params = ...)`."
            if id(ns) in seen:
                raise AssertionError(f"External-parameter group '{name}' lists the same node twice.")
            seen.add(id(ns))

        sigs = {ns.external_params.get_signature() for ns in nodes}
        assert len(sigs) == 1, \
            f"External-parameter group '{name}' mixes parameterizations {sigs}; one concatenated " \
            f"tensor cannot serve nodes whose layouts differ."

        # Members that SHARE storage (a tied node and its copies) are fed once, by the owner. Handing
        # such a pair two different slices is a contradiction, and silently letting one win would apply
        # the wrong gate to the other -- so it is rejected here rather than resolved arbitrarily.
        owners = [ns.external_params.storage_owner(ns) for ns in nodes]
        if len(set(id(o) for o in owners)) != len(owners):
            raise AssertionError(
                f"External-parameter group '{name}' contains nodes that share one external tensor "
                f"(`tie_external`). They are supplied once, through the storage owner, so they cannot "
                f"also take separate slices of a concatenated tensor."
            )

        # Shapes must agree everywhere except along `dim`. Checked once, at registration, on a nominal
        # batch of 1: the shapes are affine in the batch, so the comparison holds for every batch.
        shapes = [ns.external_params.tensor_shapes(ns, 1) for ns in nodes]
        n_slots = len(shapes[0])
        assert all(len(sh) == n_slots for sh in shapes), \
            f"External-parameter group '{name}' mixes nodes with different numbers of tensors."
        for slot in range(n_slots):
            ref = list(shapes[0][slot])
            nd = len(ref)
            assert -nd <= dim < nd, \
                f"`dim = {dim}` is out of range for slot {slot} of group '{name}', whose tensors " \
                f"have {nd} axes."
            axis = dim % nd
            for i, sh in enumerate(shapes[1:], start = 1):
                got = list(sh[slot])
                if [d for a, d in enumerate(got) if a != axis] != [d for a, d in enumerate(ref) if a != axis]:
                    raise AssertionError(
                        f"External-parameter group '{name}': node {i}'s slot-{slot} shape {tuple(got)} "
                        f"disagrees with node 0's {tuple(ref)} on an axis other than {axis}."
                    )

        self.external_params_groups[name] = (list(nodes), int(dim))

    def unregister_external_params_group(self, name: str) -> None:
        """Remove a group registered by :func:`register_external_params_group`."""
        assert name in self.external_params_groups, \
            f"No external-parameter group named '{name}'."
        del self.external_params_groups[name]

    def _group_fast_stage(self, name: str, tensors, views: dict, batch_size: int):
        """
        `(destination, source)` staging a WHOLE group in one transpose, or None if it cannot.

        Splitting a group into per-node views is correct but costs: each slice of a batch-first
        concatenated tensor is strided, so every member falls off the tiled-transpose path onto the
        generic copy, and the group ends up SLOWER to stage than the per-node form it replaces.

        It does not have to. When the members' slots happen to be adjacent in the staging buffer and
        in the group's own order -- which is what compiling an HMM's transitions in depth order
        produces -- the whole group is one contiguous destination, and `[B, sum Nk, Ck]` maps onto it
        by exactly the transpose the fast kernel already does. One copy instead of T.

        Every precondition is checked rather than assumed; anything unmet just returns None and the
        per-member path runs.
        """
        nodes, dim = self.external_params_groups[name]

        # `dim = 1` is what makes the concatenated axis the OUTERMOST storage axis, so that
        # concatenating along it is the same thing as laying the members out back to back.
        if dim != 1 or len(nodes) < 2:
            return None

        ep = nodes[0].external_params
        perm = ep.storage_perm()
        if not (perm is not None
                and type(ep).to_storage is ExternalSumParams.to_storage
                and tuple(perm) == tuple(range(1, len(perm))) + (0,)):
            return None

        if torch.is_tensor(tensors):
            tensors = (tensors,)
        if len(tensors) != 1:                       # one slot per node; the multi-slot case splits
            return None
        cat = tensors[0]
        if not (cat.is_contiguous() and cat.dtype == torch.float32 and cat.dim() >= 3):
            return None

        base = views[nodes[0]][0]
        buf = self.external_params
        elem = buf.element_size()
        total = 0
        for ns in nodes:
            d = views[ns][0]
            if d.data_ptr() != base.data_ptr() + total * elem:
                return None                          # not adjacent, or not in the group's order
            total += d.numel()

        if total != cat.numel():
            return None
        start = (base.data_ptr() - buf.data_ptr()) // elem
        return buf.narrow(0, start, total), cat

    def _expand_external_params_groups(self, ns2tensors: dict, batch_size: int) -> dict:
        """
        Replace every group-name key with one entry per member, holding a VIEW of the caller's tensor.

        Splitting rather than copying is the point: the per-node staging below already accepts
        arbitrary strides, so the views cost nothing here and the single batched copy absorbs them.
        """
        if not any(isinstance(k, str) for k in ns2tensors):
            return ns2tensors

        # Collected up front, because dict order is the caller's: a node supplied both directly and
        # through a group must be caught whichever key comes first.
        bare = {k for k in ns2tensors if not isinstance(k, str)}
        claimed = {}

        out = {}
        for key, tensors in ns2tensors.items():
            if not isinstance(key, str):
                out[key] = tensors
                continue

            assert key in self.external_params_groups, \
                f"`{EXTERNAL_PARAMS_KWARG}` names an unregistered external-parameter group: '{key}'. " \
                f"Register it with `pc.register_external_params_group('{key}', [...])`."
            nodes, dim = self.external_params_groups[key]

            if torch.is_tensor(tensors):
                tensors = (tensors,)
            n_slots = len(nodes[0].external_params.tensor_shapes(nodes[0], batch_size))
            assert len(tensors) == n_slots, \
                f"External-parameter group '{key}' expects {n_slots} concatenated tensor(s), got " \
                f"{len(tensors)}."

            splits = []
            for slot, cat in enumerate(tensors):
                assert torch.is_tensor(cat), \
                    f"External-parameter group '{key}', slot {slot}: expected a tensor, got {type(cat)}."
                sizes = [ns.external_params.tensor_shapes(ns, batch_size)[slot] for ns in nodes]
                axis = dim % len(sizes[0])
                want = list(sizes[0])
                want[axis] = sum(int(s[axis]) for s in sizes)
                assert tuple(cat.size()) == tuple(want), \
                    f"External-parameter group '{key}', slot {slot}: expected the concatenation of " \
                    f"{len(nodes)} tensors along axis {axis}, i.e. shape {tuple(want)}, got " \
                    f"{tuple(cat.size())}."
                splits.append(torch.split(cat, [int(s[axis]) for s in sizes], dim = axis))

            for i, ns in enumerate(nodes):
                assert ns not in bare, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node both directly and through group '{key}'."
                assert ns not in claimed, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node through two groups, "\
                    f"'{claimed[ns]}' and '{key}'."
                claimed[ns] = key
                out[ns] = tuple(sp[i] for sp in splits)

        return out

    def _stage_external_params(self, kwargs: dict, batch_size: int) -> None:
        """
        Copy the caller's external tensors into the staging buffer and replace the `sum_external_params`
        entry with views into it, so the layers only ever see the buffer.

        This is what decouples the caller's memory layout from the kernels'. The caller may hand over
        slices of whatever their own head produced -- strided, freshly allocated each step, in any
        arrangement -- and the layers still receive contiguous tensors at a stable address. The copy is
        issued as ONE batched op rather than one per tensor: a per-tensor `copy_` would cost a kernel
        launch per node, which for a model with many tied copies would dominate the correction itself.

        Called from the forward pass. The backward reuses what was staged here rather than asking for
        the tensors again, which also makes it impossible to run a backward whose external parameters
        disagree with the forward that produced `node_mars`.
        """
        ns2tensors = kwargs.get(EXTERNAL_PARAMS_KWARG, None)

        if ns2tensors is None or isinstance(ns2tensors, StagedExternalParams):
            # Nothing supplied, or already staged (e.g. re-entered through the autograd hook)
            return None

        views = StagedExternalParams(self._external_params_views(
            name = "external_params", batch_size = batch_size
        ))

        # A group whose members land on a contiguous run of the buffer stages as ONE transpose; the
        # rest resolve to per-node views and go through the loop below, which never learns that groups
        # exist. Tried before the expansion because it is the concatenated tensor that is contiguous,
        # not the slices of it.
        # Conflicts are checked HERE, on the caller's dict, before either staging path touches it: a
        # group taken wholesale is dropped before the expansion below, so the expansion's own check
        # would never see it.
        _bare = {k for k in ns2tensors if not isinstance(k, str)}
        _claimed = {}
        for key in ns2tensors:
            if not isinstance(key, str):
                continue
            assert key in self.external_params_groups, \
                f"`{EXTERNAL_PARAMS_KWARG}` names an unregistered external-parameter group: '{key}'. " \
                f"Register it with `pc.register_external_params_group('{key}', [...])`."
            for ns in self.external_params_groups[key][0]:
                assert ns not in _bare, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node both directly and through group '{key}'."
                assert ns not in _claimed, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node through two groups, " \
                    f"'{_claimed[ns]}' and '{key}'."
                _claimed[ns] = key

        group_fast, group_done, fast_keys = [], set(), set()
        for key, tensors in ns2tensors.items():
            if not isinstance(key, str):
                continue
            assert key in self.external_params_groups, \
                f"`{EXTERNAL_PARAMS_KWARG}` names an unregistered external-parameter group: '{key}'. " \
                f"Register it with `pc.register_external_params_group('{key}', [...])`."
            whole = self._group_fast_stage(key, tensors, views, batch_size)
            if whole is not None:
                group_fast.append(whole)
                group_done.update(self.external_params_groups[key][0])
                fast_keys.add(key)

        # A group staged wholesale is dropped BEFORE the expansion: splitting it into T views and
        # validating each, only for the loop below to skip them, was most of what was left of the cost.
        if fast_keys:
            ns2tensors = {k: v for k, v in ns2tensors.items() if k not in fast_keys}

        ns2tensors = self._expand_external_params_groups(ns2tensors, batch_size)

        # Validate against the buffer's own device rather than `self.device`, which may be index-less
        # (`torch.device("cuda")`) and so compare unequal to an otherwise identical `cuda:0`
        device = self.external_params.device

        dsts, srcs, fast = [], [], list(group_fast)
        for ns, tensors in ns2tensors.items():
            tensors = validate_external_tensors(
                ns, ns.external_params, tensors, batch_size, device, require_contiguous = False
            )

            # The staging buffer may hold the tensors in a different LAYOUT than the caller uses --
            # a different axis order (batch-innermost, so the kernels read them like `element_mars`),
            # or a different shape entirely, where the caller's layout is the one that reads naturally
            # for the model and storage keeps only the entries the kernels index. The copy absorbs
            # whichever it is, so the caller never has to know about it.
            #
            # When that layout change is exactly "rotate the batch axis to the end", the copy is a
            # transpose and gets the tiled kernel. Recognized from the DECLARED permutation -- a tuple
            # comparison -- rather than from the tensors, so the test costs nothing per forward.
            perm = ns.external_params.storage_perm()
            rotates_batch = (perm is not None
                             and type(ns.external_params).to_storage is ExternalSumParams.to_storage
                             and tuple(perm) == tuple(range(1, len(perm))) + (0,))

            if rotates_batch:
                for dst, tensor in zip(views[ns], tensors):
                    if tensor.is_contiguous() and tensor.dtype == torch.float32 and tensor.dim() >= 2:
                        fast.append((dst, tensor))
                    else:
                        dsts.append(dst)
                        srcs.append(tensor.permute(*range(1, tensor.dim()), 0))
                continue

            tensors = ns.external_params.to_storage(ns, tensors)

            dsts.extend(views[ns])
            srcs.extend(tensors)

        _staged_copy(dsts, srcs, fast)

        # Nodes the caller did NOT supply keep whatever the buffer held; drop them so a layer sees
        # exactly the set that was staged.
        #
        # Membership is by STORAGE GROUP, not by node: when several nodes share one set of tensors (see
        # `ExternalSumParams.storage_owner`) the caller supplies them once, and every node in that group
        # is staged. Comparing node-by-node would keep only the one that was named and silently turn the
        # rest into plain sum layers -- correct-looking output with the correction applied to one
        # timestep out of many.
        supplied_groups = {ns.external_params.storage_owner(ns)
                           for ns in (set(ns2tensors) | group_done)}
        for ns in list(views.keys()):
            if ns.external_params.storage_owner(ns) not in supplied_groups:
                del views[ns]

        kwargs[EXTERNAL_PARAMS_KWARG] = views
        kwargs[EXTERNAL_PARAMS_BUFFER_KWARG] = self.external_params
        self._staged_external_params = views

    def _resolve_backward_external_params(self, kwargs: dict) -> None:
        """
        Point the backward pass at the external parameters the forward actually staged.

        The forward leaves `node_mars` in a form that only the matching external backward interprets
        correctly, so running a backward against *different* external parameters -- or none -- would
        silently produce wrong flows. Taking them from the staging buffer makes that impossible: the
        values used are, by construction, the ones the forward used, and they are pyjuice's own
        snapshot, so nothing the caller does to their tensors in between can perturb them.

        A caller may still pass `sum_external_params` (the autograd hook forwards it), but it only has
        to name the same set of nodes; the staged values are what gets used.
        """
        staged = self._staged_external_params

        ns2tensors = kwargs.get(EXTERNAL_PARAMS_KWARG, None)
        if ns2tensors is not None and not isinstance(ns2tensors, StagedExternalParams):
            assert staged is not None, \
                f"`{EXTERNAL_PARAMS_KWARG}` was given to the backward pass, but the forward pass did " \
                f"not receive any external parameters."
            named = set()
            for k in ns2tensors:
                named.update(self.external_params_groups[k][0] if isinstance(k, str) else [k])
            assert named == set(staged.keys()), \
                f"`{EXTERNAL_PARAMS_KWARG}` names a different set of nodes than the forward pass did."

        if staged is not None:
            kwargs[EXTERNAL_PARAMS_KWARG] = staged
            kwargs[EXTERNAL_PARAMS_BUFFER_KWARG] = self.external_params

    def _check_external_params_kwargs(self, kwargs: dict) -> None:
        """
        Check that every `sum_external_params` / `sum_external_params_grad` entry names a node this PC
        actually compiled with an external parameterization.

        Without this, a stale or mistyped `ns` key is silently ignored -- the PC runs on its shared
        parameters and quietly returns a different answer than intended.
        """
        for kwarg_name in (EXTERNAL_PARAMS_KWARG,):
            ns2tensors = kwargs.get(kwarg_name, None)
            if ns2tensors is None:
                continue

            assert isinstance(ns2tensors, dict), \
                f"`{kwarg_name}` should be a dict mapping nodes to tensors, got {type(ns2tensors)}."

            for ns in ns2tensors:
                if isinstance(ns, str):
                    assert ns in self.external_params_groups, \
                        f"`{kwarg_name}` names an unregistered external-parameter group: '{ns}'. " \
                        f"Register it with `pc.register_external_params_group('{ns}', [...])`."
                    continue
                assert ns in self.external_params_nodes, \
                    f"`{kwarg_name}` contains a node that this PC did not compile with an external " \
                    f"parameterization: {ns}. Construct it with `pyjuice.summate(..., external_params = ...)`."

    def _init_external_params_grads(self, kwargs: dict, batch_size: int, compute_external_grads: bool) -> None:
        """
        Allocate and zero the gradient buffer for the staged external parameters, and hand the layers
        views into it.

        It mirrors the value buffer slot for slot, so the gradient of a node's tensor lives at the
        same offset as the tensor itself and :func:`get_external_params_grad` can hand back a view
        rather than copying anything out. Zeroing happens once, here, because the layers ACCUMULATE --
        which is what lets a node appear in several layers (a tied transition) and sum its
        contributions.
        """
        # The caller may supply DESTINATIONS -- `{ns: buffer}` or `{group: buffer}`, exactly the
        # forward's shape -- to have the gradients written into their own tensors. The kernels still
        # write the PC's internal buffer (they address it by compiled offset), so this records the
        # destinations and the copy-out happens once the backward is done; `get_external_params_grad`
        # keeps working either way.
        dsts = kwargs.pop(EXTERNAL_PARAMS_GRAD_KWARG, None)
        self._external_grad_dsts = None
        if dsts is not None:
            assert isinstance(dsts, dict), \
                f"`{EXTERNAL_PARAMS_GRAD_KWARG}` should be a dict mapping nodes (or group names) to " \
                f"the buffers to fill, got {type(dsts)}."
            assert self._staged_external_params is not None, \
                f"`{EXTERNAL_PARAMS_GRAD_KWARG}` was given, but the forward pass received no external " \
                f"parameters."
            dsts = self._expand_external_params_groups(dsts, batch_size)
            for ns, tensors in dsts.items():
                assert ns in self._staged_external_params, \
                    f"`{EXTERNAL_PARAMS_GRAD_KWARG}` names {ns}, which was not given external " \
                    f"parameters in the forward pass."
                validate_external_tensors(ns, ns.external_params, tensors, batch_size,
                                          self.external_params.device, require_contiguous = False)
            self._external_grad_dsts = dsts
            compute_external_grads = True          # asking for them is asking for them

        if self._staged_external_params is None or not compute_external_grads:
            self._staged_external_params_grad = None
            return None

        grad_views = StagedExternalParams(self._external_params_views(
            name = "external_params_grad", batch_size = batch_size, set_value = 0.0
        ))

        # Only the nodes that actually got external parameters this pass
        for ns in list(grad_views.keys()):
            if ns not in self._staged_external_params:
                del grad_views[ns]

        kwargs[EXTERNAL_PARAMS_GRAD_KWARG] = grad_views
        # The flat buffer too: a kernel that writes gradients addresses it with the same compiled
        # offsets as the forward uses for the tensors themselves (the two buffers share a layout).
        kwargs[EXTERNAL_PARAMS_GRAD_BUFFER_KWARG] = self.external_params_grad
        self._staged_external_params_grad = grad_views

    def get_external_params_grad(self, ns: CircuitNodes):
        """
        Per-sample gradients of the external parameters of `ns`, as computed by the last backward pass.

        Returned as views into the PC's gradient buffer, laid out exactly like the tensors that were
        supplied for `ns` -- so there is nothing to allocate and nothing to copy out. They are valid
        until the next backward pass overwrites them; clone if you need to keep them.

        :param ns: a node that was given external parameters in the last forward pass
        :type ns: CircuitNodes

        A group name registered with :func:`register_external_params_group` is also accepted, and
        returns the members' gradients concatenated along the group's axis -- laid out exactly like the
        tensor that was supplied. Unlike the forward, this one COPIES: the per-node gradients are views
        into a buffer in which a group's members are not generally adjacent.

        :returns: a tuple of gradient tensors, matching the tensors supplied for `ns`
        """
        if isinstance(ns, str):
            assert ns in self.external_params_groups, f"No external-parameter group named '{ns}'."
            nodes, dim = self.external_params_groups[ns]
            per_node = [self.get_external_params_grad(m) for m in nodes]
            n_slots = len(per_node[0])
            axis = dim % per_node[0][0].dim()
            return tuple(torch.cat([g[slot] for g in per_node], dim = axis) for slot in range(n_slots))

        assert self._staged_external_params_grad is not None, \
            "No external-parameter gradients are available. Run a forward pass with " \
            f"`{EXTERNAL_PARAMS_KWARG}` and then a backward pass with `compute_external_grads = True`."
        assert ns in self._staged_external_params_grad, \
            f"No external-parameter gradients for {ns}; it was not given external parameters in the " \
            f"last forward pass."

        # Refused HERE rather than in the backward, because the buffer is allocated and zeroed by
        # default (`compute_external_grads = True`): a parameterization that computes the element and
        # parameter flows but not its own gradient would otherwise either break every backward or hand
        # back a plausible-looking tensor of zeros.
        if not ns.external_params.computes_external_grads:
            raise NotImplementedError(
                f"`{ns.external_params.get_signature()}` does not compute gradients with respect to "
                f"its own parameters yet. The element and parameter flows it contributes ARE computed, "
                f"so `pc.backward()` and the EM optimizers work; only this read is unavailable."
            )

        grad_tensors = self._staged_external_params_grad[ns]

        # Stored in the kernels' layout; hand them back in the caller's, so they line up
        # element-for-element with the tensors that were supplied.
        return ns.external_params.from_storage(ns, grad_tensors)

    def forward_ll(self, *args, **kwargs):
        self.forward(*args, propagation_alg = "LL", **kwargs)

    def forward_mpe(self, *args, **kwargs):
        self.forward(*args, propagation_alg = "MPE", **kwargs)

    def forward_general_ll(self, *args, alpha: float = 1.0, **kwargs):
        self.forward(*args, propagation_alg = "GeneralLL", **kwargs)

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0, keep_zero_params: bool = False,
                      step_size_rescaling: bool = False, use_cudagraph: bool = False):
        """
        Perform an EM parameter update step using the accumulated parameter flows.

        :param step_size: Step size - updated_params <- (1-step_size) * params + step_size * new_params
        :type step_size: float

        :param pseudocount: a pseudo count added to the parameter flows
        :type pseudocount: float

        :param keep_zero_params: if set to `True`, do not add pseudocounts to zero parameters
        :type keep_zero_params: bool

        :param step_size_rescaling: whether to rescale the step size by flows
        :type step_size_rescaling: bool
        """
        assert not step_size_rescaling or self._cum_flow > 0.0, "Please perform a backward pass before calling `mini_batch_em`."
        assert 0.0 < step_size <= 1.0, "`step_size` should be between 0 and 1."

        with device_grad_controller(device = self.device, no_grad = True):

            # Apply step size rescaling according to the mini-batch EM objective derivation
            if step_size_rescaling:
                self.init_param_flows(flows_memory = step_size / self._cum_flow)

                eval_top_down_probs(self, update_pflow = True, scale = (1.0 - step_size), use_cudagraph = use_cudagraph)

                self._cum_flow = 0.0 # Zero out the cumulative flow value
                step_size = 1.0 # We have applied the step size within the parameter flows

            # Update input layers
            for layer in self.input_layer_group:
                layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount, keep_zero_params = keep_zero_params)

            # Accumulate parameter flows of tied nodes
            compute_cum_par_flows(self.param_flows, self.parflow_fusing_kwargs)

            # Normalize and update parameters
            em_par_update(self.params, self.param_flows, self.par_update_kwargs, 
                        step_size = step_size, pseudocount = pseudocount,
                        keep_zero_params = keep_zero_params)

    def cumulate_flows(self, inputs: torch.Tensor, params: Optional[torch.Tensor] = None):
        with torch.no_grad():
            self.forward(inputs, params)
            self.backward(inputs = inputs, compute_param_flows = True, flows_memory = 1.0)

    def init_param_flows(self, flows_memory: float = 1.0, batch_size: Optional[int] = None):
        """
        Initialize parameter flows.

        :param flows_memory: the number that the current parameter flows (if any) will be multiplied by; equivalent to zeroling the flows if set to 0
        :type flows_memory: float
        """

        assert 0.0 <= flows_memory <= 1.0, f"`flows_memory` should be in [0.0, 1.0]"

        if batch_size is None:
            pflow_shape = (self.num_param_flows,)
        else:
            pflow_shape = (self.num_param_flows, batch_size)
            
        self._init_buffer(name = "param_flows", shape = pflow_shape)

        if flows_memory != 1.0:
            self.param_flows[:] *= flows_memory

        # For input layers
        for layer in self.input_layer_group:
            layer.init_param_flows(flows_memory = flows_memory)

        return None

    def sync_param_flows(self, dtype: Optional[torch.dtype] = None, op = None, group = None,
                         sync_cum_flow: bool = True):
        """
        All-reduce the parameter flows across a ``torch.distributed`` process group, for DDP EM
        training. Sums ``self.param_flows`` (sum layers) and every input layer's ``param_flows`` so
        each rank ends up with the total flow over the global batch. Call after ``backward(...)`` and
        before the EM update (``mini_batch_em`` / ``full_batch_em``); it replaces the hand-rolled
        ``dist.all_reduce(pc.param_flows); for l in pc.input_layer_group: dist.all_reduce(l.param_flows)``.

        :param dtype: if given (e.g. ``torch.bfloat16``), reduce in this lower precision to halve the
            communication on bandwidth-bound interconnects (validated ~2x faster; the param-flow
            rounding is benign, ~1e-4 ΔLL). Each buffer is cast, reduced, and copied back in place;
            stored ``param_flows`` stay float32.
        :param op:    reduction op (default ``ReduceOp.SUM`` — flows are additive across data shards).
        :param group: process group (default: the default group).
        :param sync_cum_flow: also reduce the EM normalizer ``_cum_flow`` with the same op (default
            True), so the post-sync EM update is correct under load imbalance without a manual
            ``_cum_flow *= world_size``. It's a 1-element collective and its readback is the sync the
            EM update would force anyway -> no measurable overhead vs the param-flow reduce.

        No-op if ``torch.distributed`` is unavailable / uninitialized / ``world_size == 1``.
        """
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return None
        if op is None:
            op = dist.ReduceOp.SUM

        tensors = []
        if getattr(self, "param_flows", None) is not None:
            tensors.append(self.param_flows)
        for layer in self.input_layer_group:
            pf = getattr(layer, "param_flows", None)
            if pf is not None:
                tensors.append(pf)

        for pf in tensors:
            if dtype is not None and pf.dtype != dtype:
                buf = pf.to(dtype)
                dist.all_reduce(buf, op = op, group = group)
                pf.copy_(buf)
            else:
                dist.all_reduce(pf, op = op, group = group)

        # Reduce the scalar EM normalizer in the SAME op. Must be called by ALL ranks unconditionally
        # (collectives can't be value-gated, or they deadlock); _cum_flow always exists (init 0.0).
        if sync_cum_flow:
            ct = torch.tensor([float(self._cum_flow)], dtype = torch.float64, device = self.device)
            dist.all_reduce(ct, op = op, group = group)
            self._cum_flow = ct.item()

        return None

    def zero_param_flows(self):
        """
        Zero out parameter flows.
        """
        self.init_param_flows(flows_memory = 0.0)

    def update_parameters(self, clone: bool = True):
        """
        Copy parameters from this `TensorCircuit` to the original `CircuitNodes`.

        :param clone: whether to deepcopy parameters
        :type clone: bool
        """
        params = self.params.detach().cpu()

        for ns in self.root_ns:
            if ns.is_sum() and not ns.is_tied():
                ns.update_parameters(params, clone = clone)

        for layer in self.input_layer_group:
            layer.update_parameters()

        return None

    def update_param_flows(self, clone: bool = True, origin_ns_only: bool = True):
        """
        Copy parameter flows from this `TensorCircuit` to the original `CircuitNodes`.

        :param clone: whether to deepcopy parameters
        :type clone: bool
        """
        param_flows = self.param_flows.detach().cpu()

        for ns in self.root_ns:
            if ns.is_sum() and not ns.is_tied():
                ns.update_param_flows(param_flows, clone = clone, origin_ns_only = origin_ns_only)

    def print_statistics(self):
        """
        Print the statistics of the PC.
        """
        print(f"> Number of nodes: {self.num_nodes}")
        print(f"> Number of edges: {self.num_edges}")
        print(f"> Number of sum parameters: {self.num_sum_params}")

        num_input_parameters = 0
        for layer in self.input_layer_group:
            try:
                num_input_parameters += layer.params.numel()
            except AttributeError:
                pass
        
        print(f"> Number of input parameters: {num_input_parameters}")

    def get_node_mars(self, ns: CircuitNodes):
        """
        Retrieve the node values of `ns` from the previous forward pass.

        :params ns: the target nodes
        :type ns: CircuitNodes
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "node_mars") and self.node_mars is not None
        assert hasattr(self, "element_mars") and self.element_mars is not None

        nsid, neid = ns._output_ind_range

        if ns.is_sum() or ns.is_input():
            return self.node_mars[nsid:neid,:].detach()
        else:
            assert ns.is_prod()

            target_layer = None
            for layer_group in self.inner_layer_groups:
                for layer in layer_group:
                    if layer.is_prod() and ns in layer.nodes:
                        target_layer = layer
                        break

                if target_layer is not None:
                    break

            # Rerun the corresponding product layer to get the node values
            layer(self.node_mars, self.element_mars)

            return self.element_mars[nsid:neid,:].detach()

    def get_node_flows(self, ns: CircuitNodes, **kwargs):
        """
        Retrieve the node flows of `ns` from the previous backward pass.

        :params ns: the target nodes
        :type ns: CircuitNodes
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "node_flows") and self.node_flows is not None
        assert hasattr(self, "element_flows") and self.element_flows is not None

        nsid, neid = ns._output_ind_range

        if ns.is_sum() or ns.is_input():
            return self.node_flows[nsid:neid,:].detach()
        else:
            assert ns.is_prod()

            layer_id = None
            for idx, layer_group in enumerate(self.inner_layer_groups):
                for layer in layer_group:
                    if layer.is_prod() and ns in layer.nodes:
                        layer_id = idx
                        break

                if layer_id is not None:
                    break

            # Rerun the corresponding product layer to get the node values
            self.inner_layer_groups[layer_id].forward(self.node_mars, self.element_mars, _for_backward = True)
            self.inner_layer_groups[layer_id+1].backward(
                self.node_flows, self.element_flows, self.node_mars, self.element_mars, self.params, 
                param_flows = None, allow_modify_flows = self._run_params["allow_modify_flows"], 
                propagation_alg = self._run_params["propagation_alg"], 
                logspace_flows = self._run_params["logspace_flows"], 
                negate_pflows = self._run_params["negate_pflows"],
                force_use_fp32 = self._run_params["force_use_fp32"], **kwargs
            )

            return self.element_flows[nsid:neid,:].detach()

    def get_node_params(self, ns: CircuitNodes, clone: bool = True, **kwargs):
        """
        Retrieve the node parameters of `ns`.

        :params ns: the target nodes
        :type ns: CircuitNodes

        :params clone: whether to clone the parameters
        :type clone: bool
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "params") and self.params is not None

        if not ns.is_sum() or ns.is_tied():
            return None

        psid, peid = ns._param_range
        if clone:
            ns_params = self.params[psid:peid].detach().clone()
        else:
            ns_params = self.params[psid:peid]

        local_parids = (ns._param_ids - psid) // (ns.block_size * ns.ch_block_size)
        num_parblocks = local_parids.size(0)
        ns_params = ns_params.reshape(num_parblocks, ns.ch_block_size, ns.block_size)

        return ns_params[local_parids,:,:].permute(0, 2, 1)

    def get_node_param_flows(self, ns: CircuitNodes, clone: bool = True, **kwargs):
        """
        Retrieve the node parameter flows of `ns`.

        :params ns: the target nodes
        :type ns: CircuitNodes

        :params clone: whether to clone the parameter flows
        :type clone: bool
        """
        assert self.root_ns.contains(ns)
        assert hasattr(self, "param_flows") and self.param_flows is not None

        if not ns.is_sum() or ns.is_tied():
            return None

        pfsid, pfeid = ns._param_flow_range
        if clone:
            ns_param_flows = self.param_flows[pfsid:pfeid].detach().clone()
        else:
            ns_param_flows = self.param_flows[pfsid:pfeid]

        local_parfids = (ns._param_flow_ids - pfsid) // (ns.block_size * ns.ch_block_size)
        num_parfblocks = local_parfids.size(0)
        ns_param_flows = ns_param_flows.reshape(num_parfblocks, ns.ch_block_size, ns.block_size)
        return ns_param_flows[local_parfids,:,:].permute(0, 2, 1)

    def layers(self, reverse: bool = False, ret_layer_groups: bool = False, ignore_input_layers: bool = False):
        """
        Returns an iterator of all PC layers.

        :param ret_layer_groups: whether to return `LayerGroup`s instead of `Layer`s
        :type ret_layer_groups: bool
        """
        return layer_iterator(self, reverse = reverse, ret_layer_groups = ret_layer_groups, ignore_input_layers = ignore_input_layers)

    def enable_partial_evaluation(self, scopes: Union[Sequence[BitSet],Sequence[int]],
                                  forward: bool = False, backward: bool = False, overwrite: bool = False):
        """
        Restrict subsequent forward and/or backward passes to only the nodes whose scope is contained
        in `scopes`. This speeds up repeated queries that touch a fixed subset of variables (e.g.,
        evaluating or updating only part of the circuit). Call :func:`disable_partial_evaluation` to
        revert to evaluating the whole circuit.

        :param scopes: the scopes to evaluate, given either as a sequence of variable ids or :class:`~pyjuice.utils.BitSet` scopes
        :type scopes: Union[Sequence[BitSet], Sequence[int]]

        :param forward: whether to enable partial evaluation for the forward pass
        :type forward: bool

        :param backward: whether to enable partial evaluation for the backward pass
        :type backward: bool

        :param overwrite: whether to overwrite an already-enabled partial-evaluation configuration
        :type overwrite: bool
        """
        # Create scope2nid cache
        self._create_scope2nid_cache()

        if not overwrite and (forward and self._fw_partial_eval_enabled or backward and self._bk_partial_eval_enabled):
            raise RuntimeError("Partial evaluation already enabled, consider calling `disable_partial_evaluation` first.")

        if isinstance(scopes[0], int):
            scopes = [BitSet.from_array([var]) for var in scopes]

        fw_scopes = scopes if forward else None
        bk_scopes = scopes if backward else None

        # Input layers
        for layer in self.input_layer_group:
            layer.enable_partial_evaluation(fw_scopes = fw_scopes, bk_scopes = bk_scopes)

        # Inner layers
        for layer_group in self.inner_layer_groups:
            layer_group.enable_partial_evaluation(fw_scopes = fw_scopes, bk_scopes = bk_scopes)

        if forward:
            self._fw_partial_eval_enabled = True

        if backward:
            self._bk_partial_eval_enabled = True

    def disable_partial_evaluation(self, forward: bool = True, backward: bool = True):
        """
        Disable partial evaluation (see :func:`enable_partial_evaluation`), so that subsequent passes
        again evaluate the entire circuit.

        :param forward: whether to disable partial evaluation for the forward pass
        :type forward: bool

        :param backward: whether to disable partial evaluation for the backward pass
        :type backward: bool
        """
        # Input layers
        for layer in self.input_layer_group:
            layer.disable_partial_evaluation(forward = forward, backward = backward)

        # Inner layers
        for layer_group in self.inner_layer_groups:
            layer_group.disable_partial_evaluation(forward = forward, backward = backward)

        if forward:
            self._fw_partial_eval_enabled = False

        if backward:
            self._bk_partial_eval_enabled = False

    def _init_buffer(self, name: str, shape: Tuple, set_value: Optional[float] = None, check_device: bool = True):
        flag = False
        if not name in self.__dict__:
            flag = True

        tensor = self.__dict__[name]
        if not flag and not isinstance(tensor, torch.Tensor):
            flag = True

        if not flag and tensor.dim() != len(shape):
            flag = True

        for i, d in enumerate(shape):
            if not flag and tensor.size(i) != d:
                flag = True

        if not flag and check_device and self.device.index is not None and tensor.device != self.device:
            flag = True

        if flag:
            self.__dict__[name] = torch.zeros(shape, device = self.device)

        if set_value is not None:
            if len(shape) == 1:
                self.__dict__[name][:] = set_value
            elif len(shape) == 2:
                self.__dict__[name][:,:] = set_value
            elif len(shape) == 3:
                self.__dict__[name][:,:,:] = set_value
            elif len(shape) == 4:
                self.__dict__[name][:,:,:,:] = set_value
            elif len(shape) == 5:
                self.__dict__[name][:,:,:,:,:] = set_value
            else:
                raise ValueError(f"Too many dimensions ({len(shape)}).")

    def _buffer_matches(self, name: str, cache: Optional[dict], check_device: bool = True):
        if cache is None:
            return False

        assert name in self.__dict__

        tensor = self.__dict__[name]
        
        if name not in cache:
            return False

        if tensor.size() != cache[name].size():
            return False

        if check_device and tensor.device != cache[name].device:
            return False

        return True

    def _get_num_vars(self, ns: CircuitNodes):
        num_vars = 0
        for v in ns.scope:
            if (v + 1) > num_vars:
                num_vars = v + 1
        return num_vars

    def _init_layers(self, layer_sparsity_tol: Optional[float] = None, max_num_partitions: Optional[int] = None,
                     disable_gpu_compilation: bool = False, force_gpu_compilation: bool = False, 
                     max_tied_ns_per_parflow_block: int = 32, verbose: bool = True, device: Optional[Union[str,torch.device]] = None):

        if hasattr(self, "input_layer_group") or hasattr(self, "inner_layer_groups"):
            raise ValueError("Attempting to initialize a TensorCircuit for the second time. " + \
                "Please instead create a new TensorCircuit instance by calling `pc = TensorCircuit(root_ns)`.")

        # Clear hooks/pointers used by previous `TensorCircuit`s
        self.root_ns._clear_tensor_circuit_hooks()

        # TOtal number of variables
        pc_num_vars = len(self.root_ns.scope)

        # Create layers
        depth2nodes, num_layers, max_node_block_size, max_ele_block_size = self._create_node_layers()

        self.input_layer_group = None
        self.inner_layer_groups = []

        self.num_dummy_nodes = max_ele_block_size
        self.num_dummy_eles = max_node_block_size
        self.num_dummy_params = max_node_block_size * max_ele_block_size

        # Nodes include `max_ele_block_size` dummy nodes and all input/sum nodes in the PC
        num_nodes = self.num_dummy_nodes

        # Total number of edges
        num_edges = 0

        # Elements include `max_node_block_size` dummy elements and all product nodes in the PC
        num_elements = self.num_dummy_eles

        # Number of parameters
        num_parameters = self.num_dummy_params

        # Number of parameter flows
        num_param_flows = 0

        # Stores distributed parameter flows
        node2tiednodes = dict()

        # Every `ns` in this PC that takes external parameters, mapped to the layer that compiled it.
        # Insertion-ordered, and the order is what fixes each node's slot in the staging buffer, so it
        # must stay deterministic. Also used to reject `sum_external_params` entries keyed by a node
        # the PC does not have -- which would otherwise be a silent no-op.
        self.external_params_nodes = dict()

        # (batch size) -> (total numel, {ns: [(offset, shape), ...]}) for the staging buffer
        self._external_params_layouts = dict()

        # Views staged by the most recent forward pass; the backward reuses them, and writes the
        # matching per-sample gradients into views of the same shape
        self._staged_external_params = None
        self._staged_external_params_grad = None
        self._external_grad_dsts = None

        # name -> (nodes, dim): several `ns` addressed by ONE concatenated tensor. See
        # `register_external_params_group`.
        self.external_params_groups = dict()

        # (buffer name, batch size) -> (buffer, {ns: views}); see `_external_params_views`
        self._external_params_views_cache = dict()

        if verbose:
            print(f"Compiling {num_layers} TensorCircuit layers...")

        # Select device to use
        if device is None:
            device = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            device = device.index

        with torch.cuda.device(f"cuda:{device}"):
            layer_id = 0
            for depth in tqdm(range(num_layers), disable = not verbose):
                if depth == 0:
                    # Input layer
                    signature2nodes = self._categorize_input_nodes(depth2nodes[0]["input"])
                    input_layer_id = 0
                    input_layers = []
                    for signature, nodes in signature2nodes.items():
                        input_layer = InputLayer(
                            nodes = nodes, cum_nodes = num_nodes,
                            max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block,
                            pc_num_vars = pc_num_vars
                        )

                        # Special treatment for input layers with the `External` distribution
                        if input_layer.dist.requires_external_inputs():
                            scope = BitSet()
                            for ns in input_layer.nodes:
                                scope |= ns.scope
                            vars = torch.sort(torch.tensor(scope.to_list())).values
                            var_idmapping = torch.zeros([pc_num_vars], dtype = torch.long)
                            var_idmapping[vars] = torch.arange(0, vars.size(0))
                            input_layer.register_buffer("var_idmapping", var_idmapping)

                        input_layers.append(input_layer)
                        
                        input_layer_id += 1
                        num_nodes += input_layer.num_nodes

                    self.input_layer_group = LayerGroup(input_layers)

                else:
                    assert len(depth2nodes[depth]["prod"]) > 0 and len(depth2nodes[depth]["sum"]) > 0, \
                        "Depth {}: (# prod nodes: {}, # sum nodes: {})".format(depth, len(depth2nodes[depth]["prod"]), len(depth2nodes[depth]["sum"]))

                    # Product layer(s)
                    gsize2prod_nodes = dict()
                    for ns in depth2nodes[depth]["prod"]:
                        gsize = ns.block_size
                        if gsize not in gsize2prod_nodes:
                            gsize2prod_nodes[gsize] = []
                        gsize2prod_nodes[gsize].append(ns)
                    
                    layer_num_elements = max_node_block_size
                    prod_layers = []
                    for gsize, nodes in gsize2prod_nodes.items():
                        prod_layer = ProdLayer(
                            nodes = nodes, 
                            global_nid_start = layer_num_elements,
                            layer_sparsity_tol = layer_sparsity_tol,
                            max_num_partitions = max_num_partitions,
                            disable_gpu_compilation = disable_gpu_compilation,
                            force_gpu_compilation = force_gpu_compilation
                        )

                        layer_num_elements += prod_layer.num_nodes
                        num_edges += prod_layer.num_edges

                        prod_layers.append(prod_layer)
                    
                    prod_layer_group = LayerGroup(prod_layers)
                    self.inner_layer_groups.append(prod_layer_group)
                    self.add_module(f"prod_layer_{layer_id}", prod_layer_group)

                    if layer_num_elements > num_elements:
                        num_elements = layer_num_elements

                    # Sum layer(s)
                    # Nodes are grouped by (block size, external-parameter signature): one layer
                    # compiles one set of kernels, so nodes whose effective parameters are formed
                    # differently -- or not modified at all -- must not share a layer. This keeps the
                    # standard sum-layer kernels free of any per-call branch on the parameterization.
                    gsize2sum_nodes = dict()
                    for ns in depth2nodes[depth]["sum"]:
                        ext_signature = ns.get_external_signature() if isinstance(ns, ExternalParamsSumNodes) else None
                        gsize = (ns.block_size, ext_signature)
                        if gsize not in gsize2sum_nodes:
                            gsize2sum_nodes[gsize] = []
                        gsize2sum_nodes[gsize].append(ns)

                    sum_layers = []
                    for (gsize, ext_signature), nodes in gsize2sum_nodes.items():
                        layer_class = SumLayer if ext_signature is None else ExternalParamsSumLayer

                        sum_layer = layer_class(
                            nodes = nodes,
                            global_nid_start = num_nodes, 
                            global_pid_start = num_parameters,
                            global_pfid_start = num_param_flows,
                            node2tiednodes = node2tiednodes,
                            layer_sparsity_tol = layer_sparsity_tol,
                            max_num_partitions = max_num_partitions,
                            max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block,
                            disable_gpu_compilation = disable_gpu_compilation,
                            force_gpu_compilation = force_gpu_compilation
                        )

                        num_nodes += sum_layer.num_nodes
                        num_edges += sum_layer.num_edges
                        num_parameters += sum_layer.num_parameters
                        num_param_flows += sum_layer.num_param_flows

                        if ext_signature is not None:
                            for ns in sum_layer.nodes:
                                self.external_params_nodes[ns] = sum_layer

                        sum_layers.append(sum_layer)

                    sum_layer_group = LayerGroup(sum_layers)
                    self.inner_layer_groups.append(sum_layer_group)
                    self.add_module(f"sum_layer_{layer_id}", sum_layer_group)

                    layer_id += 1

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_elements = num_elements
        self.num_sum_params = num_parameters
        self.num_param_flows = num_param_flows

        # For parameter flow accumulation
        self.parflow_fusing_kwargs = compile_cum_par_flows_fn(node2tiednodes, MAX_NBLOCKS = 2048, BLOCK_SIZE = 2048)
        
        # For parameter update
        self.par_update_kwargs = compile_par_update_fn(self.root_ns, BLOCK_SIZE = 32)

        # Register root nodes
        self.num_root_nodes = self.root_ns.num_nodes
        self._root_node_range = (self.num_nodes - self.num_root_nodes, self.num_nodes)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self, perturbation: float = 4.0, pseudocount: float = 0.0):
        for ns in self.root_ns:
            if not ns.is_tied() and (ns.is_sum() or ns.is_input()) and not ns.has_params():
                ns.init_parameters(perturbation = perturbation, recursive = False)

        params = torch.exp(torch.rand([self.num_sum_params]) * -perturbation)
        params[:self.num_dummy_params] = 0.0

        # Copy initial parameters if provided
        for ns in self.root_ns:
            if ns.is_sum() and not ns.is_tied() and ns.has_params():
                ns.gather_parameters(params)

        self._normalize_parameters(params, pseudocount = pseudocount)
        self.params = nn.Parameter(params)

        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.params.requires_grad = False

        # Initialize parameters for input layers
        for idx, layer in enumerate(self.input_layer_group):
            layer._init_parameters(perturbation)

    def _normalize_parameters(self, params, pseudocount: float = 0.0):
        if params is not None:
            normalize_parameters(params, self.par_update_kwargs, pseudocount)

    def _create_node_layers(self):
        depth2nodes = dict()
        nodes2depth = dict()

        num_layers = 1
        max_node_block_size = 0
        max_ele_block_size = 0

        def dfs(ns: CircuitNodes):

            nonlocal num_layers
            nonlocal max_node_block_size
            nonlocal max_ele_block_size

            if ns in nodes2depth:
                return
            if ns.is_input():
                nodes2depth[ns] = 0
                if 0 not in depth2nodes:
                    depth2nodes[0] = {"input": []}
                depth2nodes[0]["input"].append(ns)
            else:
                for cs in ns.chs:
                    dfs(cs)

                depth = max(map(lambda ms: nodes2depth[ms], ns.chs)) + (1 if ns.is_prod() else 0)
                num_layers = max(depth + 1, num_layers)
                nodes2depth[ns] = depth

                if depth not in depth2nodes:
                    depth2nodes[depth] = {"sum": [], "prod": []} # lists for sum and product nodes
                
                if ns.is_sum():
                    for idx, cs in enumerate(ns.chs):
                        cs_depth = nodes2depth[cs]
                        if cs_depth < depth:
                            # TODO: Make the block size be 1
                            pass_sum_ns = summate(
                                cs, num_node_blocks = cs.num_node_blocks, block_size = cs.block_size,
                                edge_ids = torch.arange(0, cs.num_node_blocks)[None,:].repeat(2, 1),
                                params = torch.eye(cs.block_size)[None,:,:].repeat(cs.num_node_blocks, 1, 1)
                            )
                            pass_prod_ns = multiply(pass_sum_ns)
                            ns.chs[idx] = pass_prod_ns

                            depth2nodes[cs_depth]["sum"].append(pass_sum_ns)

                            nodes2depth[pass_sum_ns] = cs_depth
                            nodes2depth[pass_prod_ns] = depth

                    depth2nodes[depth]["sum"].append(ns)

                    if ns.block_size > max_node_block_size:
                        max_node_block_size = ns.block_size
                elif ns.is_prod():
                    if ns.block_size > max_ele_block_size:
                        max_ele_block_size = ns.block_size
                else:
                    raise NotImplementedError(f"Unsupported node type {type(n)}.")

        dfs(self.root_ns)

        pns2layer = dict()
        for layer in range(1, len(depth2nodes)):
            for ns in depth2nodes[layer]["sum"]:
                for cs in ns.chs:
                    if cs.is_prod():
                        if id(cs) in pns2layer:
                            assert pns2layer[id(cs)] == layer, "Disallowed circumstance: a product node requested by sum nodes at different layers."
                        else:
                            depth2nodes[layer]["prod"].append(cs)

                            pns2layer[id(cs)] = layer

        return depth2nodes, num_layers, max_node_block_size, max_ele_block_size

    def _categorize_input_nodes(self, nodes: Sequence[InputNodes]):
        signature2nodes = dict()
        for ns in nodes:
            signature = ns.dist.get_signature()
            if signature not in signature2nodes:
                signature2nodes[signature] = []
            signature2nodes[signature].append(ns)

        return signature2nodes

    def _create_scope2nid_cache(self):
        # Input layers
        for idx, layer in enumerate(self.input_layer_group):
            layer._prepare_scope2nids()

        # Inner layers
        prod_scope_eleids = None
        for layer_group in self.inner_layer_groups:
            if layer_group.is_prod():
                prod_scope_eleids = layer_group._prepare_scope2nids()
            else:
                assert layer_group.is_sum()

                layer_group._prepare_scope2nids(prod_scope_eleids)


def compile(ns: CircuitNodes, layer_sparsity_tol: float = 0.5, 
            max_num_partitions: Optional[int] = None, disable_gpu_compilation: bool = False, 
            force_gpu_compilation: bool = False,
            max_tied_ns_per_parflow_block: int = 32,
            device: Optional[Union[int,torch.device]] = None,
            verbose: bool = True) -> nn.Module:
    """
    Compile a PC represented by a DAG into an equivalent `torch.nn.Module`.

    :param ns: the root node of the PC's DAG
    :type ns: CircuitNodes

    :param layer_sparsity_tol: the maximum allowed fraction for added pseudo edges within every layer (better to set to a small number for sparse/block-sparse PCs)
    :type layer_sparsity_tol: float

    :param max_num_partitions: maximum number of partitions in a layer
    :type max_num_partitions: Optional[int]

    :param disable_gpu_compilation: force PyJuice to use CPU compilation
    :type disable_gpu_compilation: bool

    :param force_gpu_compilation: force PyJuice to use GPU compilation
    :type force_gpu_compilation: bool

    :param max_tied_ns_per_parflow_block: how many groups of tied parameters are allowed to share the same flow/gradient accumulator (higher values -> consumes less GPU memory; lower values -> potentially avoid stalls caused by atomic operations)
    :type max_tied_ns_per_parflow_block: int

    :param device: Which GPU do we use for compilation (the default is `torch.cuda.current_device`)
    :type device: Optional[Union[int,torch.device]]

    :param verbose: Whether to display the progress of the compilation
    :type verbose: bool

    :returns: the compiled PC with type `torch.nn.Module`
    """
    return TensorCircuit(ns, layer_sparsity_tol = layer_sparsity_tol, max_num_partitions = max_num_partitions,
                         disable_gpu_compilation = disable_gpu_compilation, force_gpu_compilation = force_gpu_compilation,
                         max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block, device = device, verbose = verbose)
