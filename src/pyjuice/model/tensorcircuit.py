from __future__ import annotations

import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from typing import Optional, Sequence, Callable, Union

from pyjuice.nodes import CircuitNodes
from pyjuice.layer.sum_layer import PRECISIONS, _pop_removed_precision_flags
from pyjuice.nodes.methods import is_structured_decomposable

from .backend import cum_par_flows_to_device, par_update_to_device

from .build import _BuildMixin
from .common import _pc_model_backward_hook, device_grad_controller, layer_iterator
from .cuda_graphs import _CudaGraphsMixin
from .external_params import _ExternalParamsMixin
from .inspection import _InspectionMixin
from .learning import _LearningMixin


class TensorCircuit(_BuildMixin, _CudaGraphsMixin, _ExternalParamsMixin, _LearningMixin, _InspectionMixin,
                    nn.Module):
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

    :param cuda_graphs: OFF by default. When on, a forward or backward that runs a second time with the
        same buffers and options records its inner layers into a CUDA graph and later identical calls
        replay it, as if `record_cudagraph = True` were passed (an explicit `record_cudagraph` still wins).
        Worth it when the host bounds a step -- small batches. Anything that cannot be captured runs eagerly.
    :type cuda_graphs: bool
    """

    #: Whether forward / backward record CUDA graphs when the call does not say (`record_cudagraph = None`).
    #: Set by the `cuda_graphs` argument; OFF by default.
    cuda_graphs: bool = False

    #: The most CUDA graphs of the inner layers (forward and backward together) kept at once; the least
    #: recently replayed one is dropped beyond that. A graph holds no device memory of its own here (the
    #: passes write the circuit's persistent buffers), only the driver's copy of ~one node per kernel.
    max_cuda_graphs: int = 128

    #: How many graphs may be recorded for one buffer layout before recording stops for it. Distinct
    #: options legitimately give distinct graphs (an LL and an MPE forward, ...), but more than this
    #: means some option takes a new value on every call -- each call would pay a capture and replay
    #: nothing -- or that `max_cuda_graphs` is too small for the workload.
    max_cuda_graphs_per_layout: int = 8

    def __init__(self, root_ns: CircuitNodes, layer_sparsity_tol: float = 0.5, 
                 max_num_partitions: Optional[int] = None, disable_gpu_compilation: bool = False, 
                 force_gpu_compilation: bool = False,
                 max_tied_ns_per_parflow_block: int = 32,
                 device: Optional[Union[int,torch.device]] = None,
                 verbose: bool = True, cuda_graphs: bool = False, precision: str = "auto") -> None:

        super(TensorCircuit, self).__init__()

        self.cuda_graphs = bool(cuda_graphs)
        self.precision = precision

        assert isinstance(root_ns, CircuitNodes), "`root_ns` should be an instance of `CircuitNodes`."

        self.root_ns = root_ns
        self.device = torch.device("cpu")

        self.num_vars = self._get_num_vars(self.root_ns)

        # Whether every product node decomposes its scope the same way, i.e. whether the circuit
        # respects a single vtree. Computed once here because it is a property of the structure and
        # several passes want it: the top-down sampler, in particular, can precompute its whole index
        # plan when it holds, since the frontier's shape is then a function of the scopes alone and
        # never of which node a draw happened to select.
        self.is_structured_decomposable = is_structured_decomposable(self.root_ns)

        self.node_mars = None
        self.element_mars = None
        self.node_flows = None
        self.element_flows = None
        self.param_flows = None
        # Denominator flow buffer (see `ExternalSumParams.requests_denom_param_flows`): a second
        # accumulator, allocated only when some layer requests it. `None` on an ordinary PC, which then
        # pays nothing. Its LAYOUT belongs to the requesting parameterization -- it is NOT a
        # `param_flows` mirror and is not addressed by `pfids` -- so the PC only owns the flat tensor
        # and the per-layer slice offsets. Both are finalized at compile.
        self.denom_param_flows = None
        self._requests_denom_param_flows = False
        self.num_denom_flows = 0
        self.node_mars_tempered = None

        # Staging buffers for externally supplied per-sample sum parameters, and for the per-sample
        # gradients returned for them. Like `node_mars`, they are flat and (re)allocated by
        # `_init_buffer` whenever the batch size changes.
        self.external_params = None
        self.external_params_grad = None

        # Forward and backward passes run so far: a decoder session (`juice.decoder`) keeps its values in this
        # circuit's buffers between its calls and starts over when another pass has run in between
        self._num_runs = 0
        
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

        # Recorded CUDA graphs of the inner layers, least recently replayed first (see `_run_inner_pass`)
        self._recorded_cuda_graphs = OrderedDict()
        self._cuda_graph_records = dict()

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

        # The layers' tables moved, and recorded graphs point at the old ones
        self._drop_cuda_graphs()

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
        
    @property
    def precision(self) -> str:
        """
        Precision of the sum layers' products, in the forward and the backward pass:

        * ``"auto"`` (the default): the fastest per layer -- bf16 products in the block-sparse kernels from batch 16
          (about 4e-3 in a log-likelihood), TF32 on large dense layers, which run through cuBLAS;
        * ``"tf32"``: never bf16 -- round-to-nearest TF32 products (about 5e-4);
        * ``"fp32"``: fp32-level products in the forward and the backward (about 1e-6; ``tf32x3`` on tensor cores).

        Batches below 16 run fp32 kernels in every mode. Settable at any time; a recorded CUDA graph is keyed on it.
        """
        return self.__dict__.get("_precision", "auto")

    @precision.setter
    def precision(self, value: str):
        if value not in PRECISIONS:
            raise ValueError(f"`precision` must be one of {PRECISIONS}, got {value!r}.")
        self.__dict__["_precision"] = value

    def forward(self, inputs: torch.Tensor, input_layer_fn: Optional[Union[str,Callable]] = None,
                cache: Optional[dict] = None, return_cache: bool = False, record_cudagraph: Optional[bool] = None, 
                apply_cudagraph: bool = True, 
                propagation_alg: Optional[Union[str,Sequence[str]]] = None, pflow_temperature: float = 1.0, 
                _inner_layers_only: bool = False, _no_buffer_reset: bool = False, **kwargs):
        """
        Forward evaluation of the PC.

        :param inputs: input tensor of size `[B, num_vars]`
        :type inputs: torch.Tensor

        :param input_layer_fn: Custom forward function for input layers; if it is a string, then try to call the corresponding member function of the input layers
        :type input_layer_fn: Optional[Union[str,Callable]]

        :param record_cudagraph: True: capture the inner layers into a CUDA graph the first time this pass
            runs with these buffers and options (the call itself runs normally), so later identical calls
            replay it without the per-layer Python. Worth it when the host, not the GPU, bounds a step --
            small batches. Graphs are keyed on everything the inner layers are given (buffers by address,
            every option and keyword argument except the input layers' per-call evidence, see
            `Distribution.call_kwargs`), at most `max_cuda_graphs` are kept, `.to()` drops
            them, and a pass that cannot be captured runs eagerly. None (the default): the circuit's
            `cuda_graphs` setting, which is off unless it was compiled with it. False: do not record
        :type record_cudagraph: Optional[bool]

        :param apply_cudagraph: replay a graph recorded under exactly this call's signature, if there is one
        :type apply_cudagraph: bool
        """

        _pop_removed_precision_flags(kwargs)
        self._num_runs = getattr(self, "_num_runs", 0) + 1

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

            # Inner layers. Besides the buffers, everything they are given is in `sum_opts`, and a recorded
            # graph is keyed on that dict (`_graph_key_options`), so no option can reach a captured pass
            # without being part of its key.
            sum_opts = dict(kwargs, precision = self.precision, pflow_temperature = pflow_temperature)

            def _run_inner_layers():
                for layer_id, layer_group in enumerate(self.inner_layer_groups):
                    if layer_group.is_prod():
                        # Prod layer
                        layer_group(self.node_mars, self.element_mars)

                    elif layer_group.is_sum():
                        # Sum layer
                        layer_group(self.node_mars, self.element_mars, self.params,
                                    propagation_alg = propagation_alg if isinstance(propagation_alg, str) else propagation_alg[layer_id],
                                    **sum_opts)

                    else:
                        raise ValueError(f"Unknown layer type {type(layer)}.")

            # A `fast_inference` scope may turn graphs on for the whole block, so callers need not
            # thread `record_cudagraph` through every call. Its contract -- parameters, and the
            # buffers a capture bakes pointers to, do not move -- is what makes that sound, and the
            # scope frees what it captured on exit.
            from pyjuice.utils.fast_inference import cudagraphs_allowed, register_cudagraph_owner
            if record_cudagraph or cudagraphs_allowed():
                record, on_repeat = True, False
            elif record_cudagraph is None:
                record, on_repeat = self.cuda_graphs, True          # the circuit's default: OFF unless compiled with it
            else:
                record, on_repeat = False, False
            self._run_inner_pass(_run_inner_layers, 0, (self.node_mars, self.element_mars, self.params),
                                 self._graph_key_options(sum_opts, propagation_alg = propagation_alg),
                                 record = record, apply = apply_cudagraph, record_on_repeat = on_repeat)
            if record:
                register_cudagraph_owner(self)

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
                 record_cudagraph: Optional[bool] = None, 
                 apply_cudagraph: bool = True,
                 allow_modify_flows: bool = False,
                 propagation_alg: Union[str,Sequence[str]] = "LL",
                 logspace_flows: bool = True,
                 negate_pflows: bool = False,
                 _inner_layers_only: bool = False,
                 _disable_buffer_init: bool = False,
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

        :param record_cudagraph: as in :func:`forward`; a backward given layer callbacks always runs eagerly
        :type record_cudagraph: Optional[bool]

        :param apply_cudagraph: as in :func:`forward`
        :type apply_cudagraph: bool
        """

        self._num_runs = getattr(self, "_num_runs", 0) + 1
        self._run_params["allow_modify_flows"] = allow_modify_flows
        self._run_params["propagation_alg"] = propagation_alg
        self._run_params["logspace_flows"] = logspace_flows
        self._run_params["negate_pflows"] = negate_pflows
        self._run_params["precision"] = self.precision
        _pop_removed_precision_flags(kwargs)
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

            # Inner layers. As in `forward`, everything the sum layers are given besides the buffers is in
            # `sum_opts`, which is also what a recorded graph is keyed on. A query's `compute_param_flows =
            # False` used to replay a training step's graph, which writes parameter flows.
            sum_opts = dict(kwargs,
                            param_flows = self.param_flows if compute_param_flows else None,
                            denom_param_flows = self.denom_param_flows if compute_param_flows else None,
                            allow_modify_flows = allow_modify_flows, logspace_flows = logspace_flows,
                            negate_pflows = negate_pflows, precision = self.precision,
                            pflow_temperature = pflow_temperature, temper_eflow = temper_eflow)

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
                        self.inner_layer_groups[layer_id-1].forward(self.node_mars, self.element_mars)

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
                                             propagation_alg = propagation_alg if isinstance(propagation_alg, str) else propagation_alg[layer_id],
                                             **sum_opts)

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

            # Callbacks run inside the captured region and could do anything, so they make the pass eager.
            use_graphs = sum_layer_pre_backward_callback is None and sum_layer_post_backward_callback is None
            record, on_repeat = (self.cuda_graphs, True) if record_cudagraph is None else (record_cudagraph, False)
            self._run_inner_pass(_run_inner_layers, 1,
                                 (self.node_flows, self.element_flows, self.node_mars, self.element_mars, self.params),
                                 self._graph_key_options(sum_opts, propagation_alg = propagation_alg),
                                 record = use_graphs and record, apply = use_graphs and apply_cudagraph,
                                 record_on_repeat = on_repeat, reads_layer_states = True)

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

    def forward_ll(self, *args, **kwargs):
        self.forward(*args, propagation_alg = "LL", **kwargs)

    def forward_mpe(self, *args, **kwargs):
        self.forward(*args, propagation_alg = "MPE", **kwargs)

    def forward_general_ll(self, *args, alpha: float = 1.0, **kwargs):
        self.forward(*args, propagation_alg = "GeneralLL", **kwargs)

    def layers(self, reverse: bool = False, ret_layer_groups: bool = False, ignore_input_layers: bool = False):
        """
        Returns an iterator of all PC layers.

        :param ret_layer_groups: whether to return `LayerGroup`s instead of `Layer`s
        :type ret_layer_groups: bool
        """
        return layer_iterator(self, reverse = reverse, ret_layer_groups = ret_layer_groups, ignore_input_layers = ignore_input_layers)

    def _get_num_vars(self, ns: CircuitNodes):
        num_vars = 0
        for v in ns.scope:
            if (v + 1) > num_vars:
                num_vars = v + 1
        return num_vars


def compile(ns: CircuitNodes, layer_sparsity_tol: float = 0.5, 
            max_num_partitions: Optional[int] = None, disable_gpu_compilation: bool = False, 
            force_gpu_compilation: bool = False,
            max_tied_ns_per_parflow_block: int = 32,
            device: Optional[Union[int,torch.device]] = None,
            verbose: bool = True, cuda_graphs: bool = False, precision: str = "auto") -> nn.Module:
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

    :param cuda_graphs: record CUDA graphs of the inner layers by default (see :class:`TensorCircuit`); OFF
        by default
    :type cuda_graphs: bool

    :param precision: precision of the sum layers' products: "auto" (the default; the fastest per layer), "tf32"
        or "fp32" (see :attr:`TensorCircuit.precision`; settable after compiling)
    :type precision: str

    :returns: the compiled PC with type `torch.nn.Module`
    """
    return TensorCircuit(ns, layer_sparsity_tol = layer_sparsity_tol, max_num_partitions = max_num_partitions,
                         disable_gpu_compilation = disable_gpu_compilation, force_gpu_compilation = force_gpu_compilation,
                         max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block, device = device, verbose = verbose,
                         cuda_graphs = cuda_graphs, precision = precision)
