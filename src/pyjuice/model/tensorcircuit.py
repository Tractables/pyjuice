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

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes, foreach, summate, multiply
from pyjuice.layer import Layer, InputLayer, ProdLayer, SumLayer, LayerGroup
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
        with torch.cuda.device(f"cuda:{device.index}"):
            if no_grad:
                with torch.no_grad():
                    yield
            else:
                yield


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
    :type max_tied_ns_per_parflow_block: int

    :param verbose: Whether to display the progress of the compilation
    :type verbose: bool
    """

    def __init__(self, root_ns: CircuitNodes, layer_sparsity_tol: float = 0.5, 
                 max_num_partitions: Optional[int] = None, disable_gpu_compilation: bool = False, 
                 force_gpu_compilation: bool = False,
                 max_tied_ns_per_parflow_block: int = 8,
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
        super(TensorCircuit, self).to(device)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")

        self.input_layer_group.to(device)

        self.device = device

        # For parameter flow accumulation
        self.parflow_fusing_kwargs = cum_par_flows_to_device(self.parflow_fusing_kwargs, device)
        
        # For parameter update
        self.par_update_kwargs = par_update_to_device(self.par_update_kwargs, device)

        return self

    def set_propagation_alg(self, propagation_alg: str, **kwargs):
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
                propagation_alg: Optional[Union[str,Sequence[str]]] = None, _inner_layers_only: bool = False, 
                _no_buffer_reset: bool = False, **kwargs):
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
            
            ## Initialize buffers for forward pass ##

            if not _no_buffer_reset:
                self._init_buffer(name = "node_mars", shape = (self.num_nodes, B), set_value = 0.0)
                self._init_buffer(name = "element_mars", shape = (self.num_elements, B), set_value = -torch.inf)

            # Load cached node marginals
            if self._buffer_matches(name = "node_mars", cache = cache):
                self.node_mars[:,:] = cache["node_mars"]

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
                                    **kwargs)

                    else:
                        raise ValueError(f"Unknown layer type {type(layer)}.")

            signature = (0, id(self.node_mars), id(self.element_mars), id(self.params), B)
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
                 allow_modify_flows: bool = True,
                 propagation_alg: Union[str,Sequence[str]] = "LL",
                 logspace_flows: bool = False,
                 negate_pflows: bool = False,
                 _inner_layers_only: bool = False,
                 _disable_buffer_init: bool = False,
                 force_use_fp32: bool = False,
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
                                             logspace_flows = logspace_flows, negate_pflows = negate_pflows, force_use_fp32 = force_use_fp32, **kwargs)

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

            signature = (1, id(self.node_flows), id(self.element_flows), id(self.node_mars), id(self.element_mars), id(self.params), id(self.param_flows), B, allow_modify_flows, logspace_flows)
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

        if return_cache:
            if cache is None:
                cache = dict()

            with torch.no_grad():
                cache["node_flows"] = self.node_flows.clone()

            return cache
        else:
            return None

    def forward_ll(self, *args, **kwargs):
        self.forward(*args, propagation_alg = "LL", **kwargs)

    def forward_mpe(self, *args, **kwargs):
        self.forward(*args, propagation_alg = "MPE", **kwargs)

    def forward_general_ll(self, *args, alpha: float = 1.0, **kwargs):
        self.forward(*args, propagation_alg = "GeneralLL", **kwargs)

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0, keep_zero_params: bool = False,
                      step_size_rescaling: bool = False, use_cudagraph: bool = True):
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

    def layers(self, reverse: bool = False, ret_layer_groups: bool = False, ignore_input_layers: bool = False):
        """
        Returns an iterator of all PC layers.

        :param ret_layer_groups: whether to return `LayerGroup`s instead of `Layer`s
        :type ret_layer_groups: bool
        """
        return layer_iterator(self, reverse = reverse, ret_layer_groups = ret_layer_groups, ignore_input_layers = ignore_input_layers)

    def enable_partial_evaluation(self, scopes: Union[Sequence[BitSet],Sequence[int]], 
                                  forward: bool = False, backward: bool = False, overwrite: bool = False):
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
                     max_tied_ns_per_parflow_block: int = 8, verbose: bool = True, device: Optional[Union[str,torch.device]] = None):

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
                        if input_layer.get_dist().requires_external_inputs():
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
                    gsize2sum_nodes = dict()
                    for ns in depth2nodes[depth]["sum"]:
                        gsize = ns.block_size
                        if gsize not in gsize2sum_nodes:
                            gsize2sum_nodes[gsize] = []
                        gsize2sum_nodes[gsize].append(ns)
                    
                    sum_layers = []
                    for gsize, nodes in gsize2sum_nodes.items():
                        sum_layer = SumLayer(
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
            max_tied_ns_per_parflow_block: int = 8,
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
