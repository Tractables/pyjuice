from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from typing import Optional, Sequence, Callable, Union

from pyjuice.nodes import CircuitNodes
from pyjuice.layer.sum_layer import PRECISIONS, _pop_removed_precision_flags

#: The dtypes the sum layers' parameters may be stored in (`TensorCircuit.param_dtype`)
PARAM_DTYPES = (torch.float32, torch.bfloat16)
from pyjuice.nodes.methods import is_structured_decomposable

from .backend import cum_par_flows_to_device, par_update_to_device

from .build import _BuildMixin
from .common import _pc_model_backward_hook, device_grad_controller, layer_iterator
from .cuda_graphs import _CudaGraphsMixin
from pyjuice.utils.deferred import deferred_scope
from .external_params import _ExternalParamsMixin
from .inspection import _InspectionMixin
from .learning import _LearningMixin

try:                                    # (torch's registries of the hooks every module call runs)
    from torch.nn.modules.module import (_global_forward_hooks, _global_forward_pre_hooks, _global_backward_hooks,
                                         _global_backward_pre_hooks)
    _GLOBAL_HOOKS = (_global_forward_hooks, _global_forward_pre_hooks, _global_backward_hooks, _global_backward_pre_hooks)
except ImportError:
    _GLOBAL_HOOKS = None


def _calls_its_forward(module: nn.Module) -> bool:
    """Whether calling ``module`` would just call its `forward`: no hooks of its own or global ones, no compiled call, not
    traced -- `nn.Module.__call__`'s own fast path, checked without the call. False where torch keeps the hooks elsewhere."""
    if _GLOBAL_HOOKS is None:
        return False
    return not (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks
                or any(_GLOBAL_HOOKS) or getattr(module, "_compiled_call_impl", None) is not None
                or torch._C._get_tracing_state())


# The work per layer from which the layers reading only shared fp32 parameters take their bf16 copy
# (`TensorCircuit._param_operand_groups`): the parameters such a layer reads a sample, times the batch. MEASURED
# (training steps of tied HMMs, the copy forced off and on): slower with it to 2^27 (HMM-1024 B 32 / 128 +2.4 / +2.2%,
# HMM-2048 B 16 / 32 +1.5 / +1.1%), faster from 2^28 (HMM-1024 B 256 -2.2%, HMM-2048 B 64 -3.0%, HMM-4096 B 16 -3.5%;
# 2^29: HMM-2048 B 128 -9.9%, HMM-4096 B 32 -7.1%; 2^31: HMM-4096 B 128 -38%).
_COPY_MIN_WORK = 1 << 28


def _operand_columns(readers, batch_size: int) -> int:
    """The columns a layer group's kernels on the bf16 copy run at, the least over its layers': the batch for a plain
    layer's, what its parameterization widens it to for another (`readers`: the layers' parameterizations, None for a
    plain layer; `SumParameterization.param_operand_columns`)."""
    return min(batch_size if p is None else p.param_operand_columns(batch_size) for p in readers)

# The forward options a fused product forward serves (`_stream_fusion_groups`): the plain regime's, and the external
# parameters' (a gated layer whose parameterization takes the fused forward, `fuses_stream_children`)
_STREAM_FUSION_FW_OPTS = frozenset(("precision", "pflow_temperature", "sum_external_params",
                                    "sum_external_params_buffer"))

# The backward options of the plain regime a fused product backward serves (`_stream_fusion_bk_groups`)
_STREAM_FUSION_BK_OPTS = frozenset(("param_flows", "denom_param_flows", "allow_modify_flows", "logspace_flows",
                                    "negate_pflows", "precision", "pflow_temperature", "temper_eflow",
                                    "defer_param_flows", "pf_store_first", "sum_external_params",
                                    "sum_external_params_buffer", "sum_external_params_grad",
                                    "sum_external_params_grad_buffer"))
# (a fused product backward's trials compute no gate gradients: the gradient buffers left out of them)
_STREAM_FUSION_BK_GRADS = ("sum_external_params_grad", "sum_external_params_grad_buffer")


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
               GPU 268M `atomic_add` calls all targeting ONE address cost 3.7 ms, versus 44 ms for the same
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
                 verbose: bool = True, cuda_graphs: bool = False, precision: str = "auto",
                 param_dtype: torch.dtype = torch.float32) -> None:

        super(TensorCircuit, self).__init__()

        self.cuda_graphs = bool(cuda_graphs)
        self.precision = precision
        self.param_dtype = param_dtype
        # The fp32 master copy of `params` while a circuit storing them below fp32 trains (`_params_for_update`)
        self.register_buffer("params_master", None, persistent = False)

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
        # Denominator flow buffer (see `SumParameterization.requests_denom_param_flows`): a second
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

        moved = getattr(self, "device", None) != device

        super(TensorCircuit, self).to(device)

        self.input_layer_group.to(device)

        # `nn.Module.to` moves the inner layers' tables but not the tensors their passes derived from them
        if moved:
            for layer_group in self.inner_layer_groups:
                layer_group.to(device)

        self.device = device

        # The layers' tables moved, and recorded graphs point at the old ones
        self._drop_cuda_graphs()

        # The product groups' tables the fused stream forward was handed (`_stream_fusion_groups`), and its decisions,
        # made anew on this device
        if moved:
            for name in ("_fused_child_tables", "_stream_fusion_", "_fused_child_parents", "_stream_fusion_bk_"):
                self.__dict__.pop(name, None)

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
          (about 4e-3 in a log-likelihood), TF32 on large dense layers, which run through cuBLAS -- unless several layers
          share their fp32 parameters (a tied HMM's transitions): those may read a bf16 copy made in every forward (its
          backward reads the same) in their kernels whose products are bf16 -- the CUDA kernels that round the
          parameters to bf16 anyway (the same bits, half the bytes), and the stream kernels, bf16 parameters only,
          offered on the copy in the forward and the element flows --, every other kernel reading the fp32 parameters;
          they do where the parameters such a layer reads a sample, times the batch, reach `_COPY_MIN_WORK` (below it a
          step is host-bound, and the copy's host time costs more than the GPU time it saves), and are at least what
          their parameterization takes it from (BlockScale's: 2^23);
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

    @property
    def param_dtype(self) -> torch.dtype:
        """
        The dtype the sum layers' parameters (``params``) are STORED in: ``torch.float32`` (the default) or
        ``torch.bfloat16``. Every kernel reads the parameters in that dtype and computes in fp32 (or in the bf16 / fp16
        operand it already used): at fp32 nothing changes; at bf16 the parameters cost half the memory and the memory
        traffic, which is what the large layers' time is -- and the forward's block-sparse kernels round the parameters
        to bf16 for their products anyway (MEASURED: HCLT-1024 log-likelihoods 0.0000 nats apart on average, 5e-4 at
        most). Parameter flows, node and element values stay fp32. Training keeps an fp32 master copy of the parameters
        (``params_master``, made by the first update, 4 bytes a parameter until :meth:`drop_params_master`): the updates
        run on it and store its rounding, because a bf16 value moves only by more than half its ulp (MEASURED: stored
        alone, a mini-batch EM step of size 0.01 or an SGD step left 99.9-100% of the parameters unchanged); with it,
        every optimizer follows its fp32 run (1-3e-3 relative). fp16 is not offered: its range turns small probabilities
        into exact zeros. Set at compilation; :meth:`set_param_dtype` converts a compiled circuit.
        """
        return self.__dict__.get("_param_dtype", torch.float32)

    @param_dtype.setter
    def param_dtype(self, value: torch.dtype):
        if value not in PARAM_DTYPES:
            raise ValueError(f"`param_dtype` must be one of {PARAM_DTYPES}, got {value!r}.")
        self.__dict__["_param_dtype"] = value

    def set_param_dtype(self, dtype: torch.dtype) -> "TensorCircuit":
        """Convert the sum layers' parameters in place to ``dtype`` (:attr:`param_dtype`); returns the circuit.

        Converts the fp32 master copy where a training run left one (the precision the stored values rounded away), and
        drops what was derived from the old tensor: the CUDA graphs recorded on it (never replayed -- keyed by it -- but
        held) and the layers' operand caches, rebuilt by the next pass as after a device move (BlockScale's plans, whose
        CUDA forks depend on the dtype, and its parameter sums, keyed by the tensor's version -- which a new tensor
        restarts)."""
        self.param_dtype = dtype
        if self.params.dtype != dtype:
            values = self._params_in_fp32()     # `params`, or the master copy in line with them
            self.params = nn.Parameter(values.detach().to(dtype), requires_grad = self.params.requires_grad)
            self._drop_cuda_graphs()
            for group in self.inner_layer_groups:
                for layer in group:
                    drop = getattr(layer, "_drop_device_caches", None)
                    if drop is not None:
                        drop()
        self.drop_params_master()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        # `params` loaded in their stored dtype; the master copy of a training run is not part of the state. Dropped here
        # rather than in `load_state_dict`, which a module holding the circuit does not call on it
        super()._load_from_state_dict(*args, **kwargs)
        self.drop_params_master()

    # ---------------------------------------------------------------------------------------------
    # The bf16 copy of shared fp32 parameters, for the sum layers' CUDA kernels
    # ---------------------------------------------------------------------------------------------

    def _param_operand_plan(self) -> tuple:
        """``(groups, ranges)``: the inner sum layer groups whose every layer reads only parameters that several sum
        nodes share (a tied node's source and its copies) and is a plain sum layer or an external-parameter one whose
        parameterization takes the copy (`SumParameterization.reads_param_operands`), at offsets the bf16 kernels' 16-byte
        loads can take, each such layer marked (`SumLayer._reads_param_operands`), and those parameters' ranges -- what
        `_param_operands` copies. Shared ranges only: there one copy serves every layer that reads them, while an unshared
        parameter would be copied (6 bytes) for one read (4). Built once, from the structure, with each group's size for
        `_param_operand_groups` (`_param_operand_sizes_`)."""
        plan = self.__dict__.get("_param_operand_plan_")
        if plan is not None:
            return plan
        from pyjuice.layer.sum_layer import SumLayer
        from pyjuice.layer.parameterized_sum_layer import ParameterizedSumLayer

        def source(ns):
            return ns.get_source_ns() if ns.is_tied() else ns

        def aligned(layer):
            # (a bf16 kernel loads 8 parameters as one 16-byte load: every parameter base -- a node block's first
            # parameter of a child, the forward's and the backward's -- and the block a multiple of 8, as for bf16-stored
            # parameters, `SumLayer._cuda_param_loads_aligned`; one sync a table, here once)
            return layer.block_size % 8 == 0 and all(bool((t % 8 == 0).all()) for t in (*layer.partitioned_pids,
                                                                                         *layer.partitioned_parpids))

        readers = {}
        for ns in self.root_ns:
            if ns.is_sum():
                readers[source(ns)] = readers.get(source(ns), 0) + 1
        groups, ranges, sizes = [], set(), {}
        for group in self.inner_layer_groups:
            if not group.is_sum():
                continue
            marked = []
            for layer in group:
                takes = type(layer) is SumLayer or (isinstance(layer, ParameterizedSumLayer)
                                                   and layer.parameterization.reads_param_operands)
                srcs = [source(ns) for ns in layer.nodes] if takes else None
                ok = srcs is not None and all(readers.get(s, 0) >= 2 for s in srcs) and aligned(layer)
                layer.__dict__["_reads_param_operands"] = ok
                marked.append(ok)
                if ok:
                    ranges.update(s._param_range for s in srcs)
            if all(marked):
                groups.append(group)
                # (the parameters its layers read a sample -- their mean -- and the least its parameterizations
                # take the copy from: `_param_operand_groups`)
                sizes[id(group)] = (sum(layer.num_edges for layer in group) / len(group),
                                    max(getattr(getattr(layer, "parameterization", None), "param_operands_min_edges", 0)
                                        for layer in group),
                                    tuple({id(p): p for p in (getattr(layer, "parameterization", None)
                                                              for layer in group)}.values()))
        self.__dict__["_param_operand_sizes_"] = sizes
        plan = self.__dict__["_param_operand_plan_"] = (frozenset(id(g) for g in groups), tuple(sorted(ranges)))
        return plan

    #: Overrides `_param_operand_groups`' rule: True gives the copy to every group it applies to, False to none; None:
    #: the rule.
    _param_operands_force: Optional[bool] = None

    def _param_operand_groups(self, batch_size: int) -> frozenset:
        """The groups (`_param_operand_plan`'s, by id) a pass at this batch size gives the bf16 copy (`_param_operands`,
        where it applies; none: no copy made): those whose layers read enough parameters for the batch to make them
        GPU-bound -- the parameters such a layer reads a sample (their mean over the group) times the columns its
        kernels run at at least `_COPY_MIN_WORK`, those columns 16 at least -- and at least what their parameterizations
        take it from (`SumParameterization.param_operands_min_edges`). The columns: the least over the group's layers --
        the batch for a plain layer, what a parameterization widens it to for another
        (`SumParameterization.param_operand_columns`: LowRank's virtual batch, `rank` columns a sample) --, so no layer
        is handed the copy below 16 columns of its own (a plain layer's small-batch kernels read it too). The kernels
        reading the copy are faster, but it costs host time (made by every forward, handed through the layers), which a
        host-bound step pays in full while the GPU's savings hide behind the host; a layer's GPU time grows with that
        work, its host time does not. A rule, not a measurement of the caller's steps: every pass at a batch size
        computes with the same kernels -- the stream kernels' bf16 products among them --, the first as the rest, as the
        kernels' own tuning keeps them (a measurement would switch them while it measures). `_param_operands_force`
        overrides it (the columns' least kept)."""
        force = self._param_operands_force
        if force is False:
            return frozenset()
        wide, taken = self._param_operand_rule(batch_size)
        return wide if force else taken

    def _param_operand_rule(self, batch_size: int) -> tuple:
        """``(wide, taken)`` at this batch size, groups by id: those of the plan whose kernels run at 16 columns at
        least (`_operand_columns`), and those of them the rule gives the copy (`_param_operand_groups`). Once a batch
        size."""
        cache = self.__dict__.setdefault("_param_operand_rule_", {})
        hit = cache.get(batch_size)
        if hit is None:
            self._param_operand_plan()
            cols = {g: _operand_columns(readers, batch_size)
                    for g, (_, _, readers) in self.__dict__["_param_operand_sizes_"].items()}
            wide = frozenset(g for g, c in cols.items() if c >= 16)
            hit = cache[batch_size] = (wide, frozenset(
                g for g, (edges, least, _) in self.__dict__["_param_operand_sizes_"].items()
                if g in wide and edges >= least and edges * cols[g] >= _COPY_MIN_WORK))
        return hit

    def _param_operands(self, batch_size: int, refresh: Optional[bool] = True) -> Optional[torch.Tensor]:
        """The bf16 copy of the fp32 parameters several sum layers share (`_param_operand_plan`), for those layers'
        kernels whose products are bf16 under "auto" (`SumLayer.forward`'s and `backward`'s `op_params`): brought in
        line with `params` by every forward that takes it (`_param_operand_groups`), inside its pass -- a graph recorded
        from it copies them too; never keyed on a version counter, which a write through `params.data` does not move --,
        and read by the backward that follows: the parameters its forward computed the node values with
        (`_param_operands_made`); a backward whose forward made none (another precision, fewer than 16 columns) makes it
        before its pass. None where it does not apply: another precision, no layer group of the plan whose kernels run
        at 16 columns at least at this batch size (fp32 kernels in every mode below them; a group's columns: the
        batch's, or what its parameterizations widen it to, `_param_operand_rule`), parameters stored below fp32 (the
        kernels read them as they are), no such layer. ``refresh = False``: the buffer made, nothing copied (before a
        pass: a capture allocates nothing); ``None``: copied unless the last forward made the copy in this buffer (the
        backward, before its pass)."""
        p = self.params
        if self.precision != "auto" or p.dtype != torch.float32 or not p.is_cuda:
            return None
        ranges = self._param_operand_plan()[1]
        if not self._param_operand_rule(batch_size)[0]:
            return None
        op = self.__dict__.get("_param_operands_")
        if op is None or op.device != p.device or op.numel() != p.numel():
            op = self.__dict__["_param_operands_"] = torch.zeros(p.numel(), dtype = torch.bfloat16, device = p.device)
            self.__dict__["_param_operands_made"] = False
        if refresh is None:
            refresh = not self.__dict__.get("_param_operands_made", False)
        if refresh:
            # The ranges' views, kept with the storage they slice rather than sliced again every call (one `copy_` a
            # range): made anew once `params` is another tensor or its `.data` another storage -- the kept views hold
            # the old storage, so no new one can start at its address
            views = self.__dict__.get("_param_operands_views")
            if views is None or views[0] is not p or views[1] != p.data_ptr() or views[2] is not op:
                src = p.detach()
                views = self.__dict__["_param_operands_views"] = (p, p.data_ptr(), op,
                                                                  [(op[a:b], src[a:b]) for a, b in ranges])
            for dst, src in views[3]:
                dst.copy_(src)                               # (round to nearest even, as the kernels round)
        return op

    def _stream_fusion_groups(self, batch_size: int, sum_opts: dict, propagation_alg, op_groups) -> frozenset:
        """The product layer groups (by index) whose forward the sum group above forms in its stream kernels' pre-pass
        in a forward at this batch size (`SumLayer.forward`'s `stream_children`), the product layer's forward not
        launched then: where every layer of that sum group is a plain `SumLayer` whose settled forward runs the stream
        kernel and takes the fused copy (`SumLayer._stream_fusion`), and that copy MEASURES the faster -- against the
        product group's forward and the sum group's (`autotune.pick`, keyed by the pair's shape; no margin: the same
        bits either way, so the faster is taken). One more option of the tuners: decided once per batch size, at its
        second plain forward (the first settles the layers' launches), and kept; none before, nor for a pass of another
        regime (not LL, tempered, keyword arguments). Asked outside the pass: the trials run on the live buffers, whose
        rows the pass then computes again."""
        pt = sum_opts.get("pflow_temperature", 1.0)
        if (propagation_alg != "LL" or pt != 1.0 or sum_opts.keys() - _STREAM_FUSION_FW_OPTS
                or not self.node_mars.is_cuda):
            return frozenset()
        state = self.__dict__.setdefault("_stream_fusion_", {})
        key = (batch_size, self.precision, self.params.dtype, op_groups, "sum_external_params" in sum_opts)
        seen = state.get(key)
        if isinstance(seen, frozenset):
            return seen
        if seen is None:                                     # (the first: the layers' launches settle in it)
            state[key] = 1
            return frozenset()
        from pyjuice.layer.sum_layer import SumLayer
        from pyjuice.layer.fused_children import element_children
        import pyjuice.layer.kernels.autotune as autotune
        tables = self.__dict__.get("_fused_child_tables")
        if tables is None or len(tables) != len(self.inner_layer_groups):
            tables = self.__dict__["_fused_child_tables"] = element_children(self)
        groups, fused = self.inner_layer_groups, []
        op = self._param_operands(batch_size, refresh = False) if op_groups else None
        for k in range(len(groups) - 1):
            prod, sumg = groups[k], groups[k + 1]
            if not (prod.is_prod() and sumg.is_sum() and all(
                    type(layer) is SumLayer or getattr(getattr(layer, "parameterization", None), "fuses_stream_children",
                                                       False) for layer in sumg)):
                continue
            opts = dict(sum_opts, op_params = op) if (op is not None and id(sumg) in op_groups) else sum_opts
            plans = [layer._stream_fusion((batch_size, self.params.dtype, self.precision,
                                           op is not None and id(sumg) in op_groups
                                           and layer.__dict__.get("_reads_param_operands", False)), tables[k])
                     for layer in sumg]
            if any(plan is None for plan in plans):
                continue
            shape = ("prod_stream_fusion", batch_size, str(self.params.dtype), self.precision, op is not None,
                     tuple((layer.num_nodes, layer.num_edges, layer.block_size, layer.num_fw_partitions) for layer in sumg),
                     tuple((layer.num_nodes, layer.block_size) for layer in prod))

            def bench(choice, prod = prod, sumg = sumg, opts = opts, table = tables[k]):
                if choice == "fused":
                    sumg.forward(self.node_mars, self.element_mars, self.params, propagation_alg = "LL",
                                 stream_children = table, **opts)
                else:
                    prod.forward(self.node_mars, self.element_mars)
                    sumg.forward(self.node_mars, self.element_mars, self.params, propagation_alg = "LL", **opts)
            if autotune.pick(shape, ["unfused", "fused"], bench, margin = 1.0) == "fused":
                fused.append(k)
        state[key] = frozenset(fused)
        return state[key]

    def _stream_fusion_bk_groups(self, batch_size: int, sum_opts: dict, propagation_alg, op_groups) -> frozenset:
        """The product layer groups (by index) whose backward this backward leaves to the sum group above: that group's
        element flows by the stream kernel with the product fused into its merge, which writes the product's children's
        node flows, and its parameter flows captured from the children (`SumLayer.backward`'s `stream_children`) -- so
        the product layer's forward (the recompute) and backward are not launched. Where every layer of the sum group is
        a plain `SumLayer` whose backward can take it (`SumLayer._stream_fusion_bk`: settled on the stream kernel,
        parameter flows deferred), the products have two children each, and each of those children has no other parent
        element in the circuit (its flow is the one element's: logaddexp(-inf, f) = f, what the product backward
        writes); and where that MEASURES the faster -- the product layer's forward, the sum group's backward and the
        product layer's backward against the sum group's fused backward, element flows only, into scratch flow buffers
        (`autotune.pick`, keyed by the pair's shape; margin 1.0: the same bits). One more option of the tuners: decided
        once per batch size at its second plain backward (the first settles the layers' kernels), and kept; none
        before, nor for a backward of another regime (not LL in log space, tempered, negated or modified flows, keyword
        arguments, layer callbacks)."""
        if not (propagation_alg == "LL" and sum_opts.get("logspace_flows", False)
                and not sum_opts.get("allow_modify_flows", False) and not sum_opts.get("negate_pflows", False)
                and sum_opts.get("pflow_temperature", 1.0) == 1.0 and not sum_opts.get("temper_eflow", False)
                and not (sum_opts.keys() - _STREAM_FUSION_BK_OPTS) and self.node_mars.is_cuda):
            return frozenset()
        pf = sum_opts.get("param_flows") is not None
        state = self.__dict__.setdefault("_stream_fusion_bk_", {})
        key = (batch_size, self.precision, self.params.dtype, op_groups, pf, "sum_external_params_buffer" in sum_opts)
        seen = state.get(key)
        if isinstance(seen, frozenset):
            return seen
        if seen is None:                                     # (the first: the layers' kernels settle in it)
            state[key] = 1
            return frozenset()
        from pyjuice.layer.sum_layer import SumLayer
        from pyjuice.layer.fused_children import element_children
        import pyjuice.layer.kernels.autotune as autotune
        tables = self.__dict__.get("_fused_child_tables")
        if tables is None or len(tables) != len(self.inner_layer_groups):
            tables = self.__dict__["_fused_child_tables"] = element_children(self)
        parents = self.__dict__.get("_fused_child_parents")
        if parents is None:                                  # (per node: its parent elements, over every product group)
            parents = torch.zeros(self.num_nodes, dtype = torch.int32, device = self.node_mars.device)
            for t in tables:
                if t is not None:
                    parents.index_add_(0, t[0].reshape(-1).long(), torch.ones_like(t[0].reshape(-1)))
            parents = self.__dict__["_fused_child_parents"] = parents
        groups, fused = self.inner_layer_groups, []
        op = self._param_operands(batch_size, refresh = False) if op_groups else None
        nf, ef = None, None
        for k in range(len(groups) - 1):
            prod, sumg = groups[k], groups[k + 1]
            if not (prod.is_prod() and sumg.is_sum() and all(
                    type(layer) is SumLayer or getattr(getattr(layer, "parameterization", None), "fuses_stream_children",
                                                       False) for layer in sumg)):
                continue
            table = tables[k]
            given = op is not None and id(sumg) in op_groups
            if not (table[0].size(1) == 2 and all(layer._stream_fusion_bk(
                    batch_size, self.params.dtype, given and layer.__dict__.get("_reads_param_operands", False), pf)
                    for layer in sumg)):
                continue
            if not bool((parents[table[0].reshape(-1).long()] == 1).all()):
                continue
            if nf is None:                                   # (scratch flows: the trials write children's rows; and
                # node values: a gated layer's backward shifts its own rows by log Z and back, not to the bit)
                nf, ef, nm = self.node_flows.clone(), self.element_flows.clone(), self.node_mars.clone()
            opts = {k: v for k, v in sum_opts.items() if k not in _STREAM_FUSION_BK_GRADS}
            opts.update(param_flows = None, denom_param_flows = None, defer_param_flows = False, pf_store_first = False,
                        **(dict(op_params = op) if given else {}))
            shape = ("prod_stream_fusion_bk", batch_size, str(self.params.dtype), self.precision, given,
                     tuple((layer.num_nodes, layer.num_edges, layer.block_size, layer.num_bk_partitions) for layer in sumg),
                     tuple((layer.num_nodes, layer.block_size) for layer in prod))

            def bench(choice, prod = prod, sumg = sumg, opts = opts, table = table):
                if choice == "fused":
                    sumg.backward(nf, ef, nm, self.element_mars, self.params, propagation_alg = "LL",
                                  stream_children = table, **opts)
                else:
                    prod.forward(nm, self.element_mars)
                    sumg.backward(nf, ef, nm, self.element_mars, self.params, propagation_alg = "LL", **opts)
                    prod.backward(nf, ef, logspace_flows = True)
            if autotune.pick(shape, ["unfused", "fused"], bench, margin = 1.0) == "fused":
                fused.append(k)
        state[key] = frozenset(fused)
        return state[key]

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
            # (the groups this forward and its backward give the bf16 copy of shared parameters, if any:
            # `_param_operand_groups`)
            op_groups = self._param_operand_groups(B)

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
                # The buffers and parameters looked up once, and a group without hooks called at its `forward` -- what
                # `nn.Module.__call__` does for it, less that call's 0.6 us (MEASURED, a step of an HMM: two groups)
                node_mars, element_mars, params = self.node_mars, self.element_mars, self.params
                per_layer_alg = not isinstance(propagation_alg, str)
                # (the bf16 copy of shared fp32 parameters, made here, inside the pass: `_param_operands`)
                op = self._param_operands(node_mars.size(1)) if op_groups else None
                given = op_groups if op is not None else ()
                for layer_id, layer_group in enumerate(self.inner_layer_groups):
                    run = layer_group.forward if _calls_its_forward(layer_group) else layer_group
                    if layer_group.is_prod():
                        # Prod layer (none where the sum group above forms its values: `_stream_fusion_groups`)
                        if layer_id not in fused:
                            run(node_mars, element_mars)

                    elif layer_group.is_sum():
                        # Sum layer
                        alg = propagation_alg[layer_id] if per_layer_alg else propagation_alg
                        extra = sum_opts if layer_id - 1 not in fused else \
                            dict(sum_opts, stream_children = self.__dict__["_fused_child_tables"][layer_id - 1])
                        if id(layer_group) in given:
                            run(node_mars, element_mars, params, propagation_alg = alg, op_params = op, **extra)
                        else:
                            run(node_mars, element_mars, params, propagation_alg = alg, **extra)

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
            # (the copy's buffer made outside a capture; whether this pass makes the copy -- eagerly or replayed from a
            # graph, which runs none of `_run_inner_layers` -- kept for the backward that follows)
            made = bool(op_groups) and self._param_operands(B, refresh = False) is not None
            # (the product layer groups this pass leaves to the stream kernels of the sum group above: decided before
            # the pass, outside a capture)
            fused = self._stream_fusion_groups(B, sum_opts, propagation_alg, op_groups)
            self._run_inner_pass(_run_inner_layers, 0, (self.node_mars, self.element_mars, self.params),
                                 self._graph_key_options(sum_opts, propagation_alg = propagation_alg,
                                                         param_operands = op_groups, stream_fusion = fused),
                                 record = record, apply = apply_cudagraph, record_on_repeat = on_repeat)
            self.__dict__["_param_operands_made"] = made
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

    def _plan_param_flow_stores(self, B: int, flows_memory: float, pflow_temperature: float, propagation_alg,
                                defer: bool, mode = None):
        """The pass's parameter-flow STORES. At `flows_memory = 0` every slot of `param_flows` is reset before the
        layers add their flows; instead, walking the sum layers in backward order, a partition whose kernel can store
        (`SumLayer.param_flow_slot_plan`) and whose slots no earlier writer of the pass touched STORES them -- the reset
        folded into its write --, and only the slots no storing writer covers are zeroed (`init_param_flows`'s
        `_zero_ranges`; among them every slot written by an accumulating kernel and every slot nobody writes). Sets each
        layer's `_pf_store_flags` (partition -> stores) for the pass -- read only by a backward given `pf_store_first`, the
        circuit's option for this pass, so a layer's backward called on its own accumulates; the plan is cached per (batch, options). At any
        other memory the flows are scaled as before and nothing stores. Returns the ranges to zero, or None for all.
        The input layers' flow buffers are separate and untouched."""
        groups = self.inner_layer_groups
        sums = self.__dict__.get("_pf_sum_layers")
        if sums is None:                                   # (the sum layers in backward order, once)
            sums = self.__dict__["_pf_sum_layers"] = [(lid, layer) for lid in range(len(groups) - 1, -1, -1)
                                                       if groups[lid].is_sum() for layer in groups[lid]]
        if flows_memory != 0.0 or not sums:
            if self.__dict__.get("_pf_applied_key") is not None:
                for _, layer in sums:
                    layer.__dict__["_pf_store_flags"] = {}
                    layer.__dict__["_pf_store_key"] = None
                self.__dict__["_pf_applied_key"] = None
            return None
        key = (B, float(pflow_temperature), propagation_alg if isinstance(propagation_alg, str) else tuple(propagation_alg),
               bool(defer), mode, self.num_param_flows)
        plans = self.__dict__.setdefault("_pf_store_plans", {})
        plan = plans.get(key)
        if plan is None:
            from pyjuice.layer.sum_layer import _merge_ranges, _ranges_overlap, _ranges_complement
            seen = np.zeros((0, 2), dtype = np.int64)
            stored, flags = [], []
            for lid, layer in sums:
                alg = propagation_alg if isinstance(propagation_alg, str) else propagation_alg[lid]
                f = {}
                for pid, (ranges, capable) in enumerate(layer.param_flow_slot_plan(B, pflow_temperature, alg, defer, mode)):
                    store = capable and not _ranges_overlap(seen, ranges)
                    f[pid] = store
                    if store:
                        stored.append(ranges)
                    if ranges.shape[0]:
                        seen = _merge_ranges(np.concatenate([seen, ranges]))
                flags.append(f)
            covered = _merge_ranges(np.concatenate(stored)) if stored else np.zeros((0, 2), dtype = np.int64)
            zero = _ranges_complement(covered, self.num_param_flows)
            if len(zero) > 256 or zero == [(0, self.num_param_flows)]:   # (too fragmented, or nothing stored: one zero pass)
                zero = None
            # (each layer's decisions as the tuple its launch-plan key holds, built once here)
            plan = plans[key] = (flags, [tuple(sorted(f.items())) for f in flags], zero)
            self.__dict__["_pf_applied_key"] = None        # (a new plan: the layers take its decisions below)
        flags, keys, zero = plan
        if self.__dict__.get("_pf_applied_key") != key:   # (the layers hold the last plan's decisions)
            for (_, layer), f, k in zip(sums, flags, keys):
                layer.__dict__["_pf_store_flags"] = f
                layer.__dict__["_pf_store_key"] = k
            self.__dict__["_pf_applied_key"] = key
        return zero

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
            # (`flows_memory = 0`: the pass's first writer of a slot stores it, the reset folded into its write --
            # `_plan_param_flow_stores`; only the slots no storing writer covers are zeroed here)
            pf_defer = sum_layer_pre_backward_callback is None and sum_layer_post_backward_callback is None
            if compute_param_flows:
                zero_ranges = self._plan_param_flow_stores(B, flows_memory, pflow_temperature, propagation_alg, pf_defer,
                                                           kwargs.get("mode"))
                self.init_param_flows(flows_memory = flows_memory, _zero_ranges = zero_ranges)
            else:
                self._plan_param_flow_stores(B, 1.0, pflow_temperature, propagation_alg, pf_defer)   # (flags cleared)

            ## External parameters: reuse what the forward staged; reset the gradient buffers ##
            self._check_external_params_kwargs(kwargs)
            self._resolve_backward_external_params(kwargs)
            self._init_external_params_grads(kwargs, B, compute_external_grads)

            ## Run backward pass ##

            # Inner layers. As in `forward`, everything the sum layers are given besides the buffers is in
            # `sum_opts`, which is also what a recorded graph is keyed on. A query's `compute_param_flows =
            # False` used to replay a training step's graph, which writes parameter flows.
            #
            # `defer_param_flows`: a layer may leave its parameter flows to the end of the pass (`deferred_scope`, e.g.
            # a tied group's, added by one contraction) -- not when a layer callback is given, which sees the flows
            # between layers.
            sum_opts = dict(kwargs,
                            param_flows = self.param_flows if compute_param_flows else None,
                            denom_param_flows = self.denom_param_flows if compute_param_flows else None,
                            allow_modify_flows = allow_modify_flows, logspace_flows = logspace_flows,
                            negate_pflows = negate_pflows, precision = self.precision,
                            pflow_temperature = pflow_temperature, temper_eflow = temper_eflow,
                            defer_param_flows = compute_param_flows and pf_defer,
                            pf_store_first = compute_param_flows and flows_memory == 0.0)

            def _inner_layers():

                # The bf16 copy of shared fp32 parameters, for the element kernels of the layers that read it
                # (`_param_operands`, `SumLayer.backward`'s `op_params`): the forward's, or made before this pass
                op = self._param_operands(B, refresh = False) if op_groups else None
                given = op_groups if op is not None else ()

                # Backward pass for inner layers
                for layer_id in range(len(self.inner_layer_groups) - 1, -1, -1):
                    layer_group = self.inner_layer_groups[layer_id]

                    if layer_group.is_prod():
                        # Prod layer (none where the sum group above wrote its children's flows: `_stream_fusion_bk_groups`)
                        if layer_id not in fused_bk:
                            layer_group.backward(self.node_flows, self.element_flows, logspace_flows = logspace_flows)

                    elif layer_group.is_sum():
                        # Sum layer

                        # First recompute the previous product layer (none where this group forms its values itself)
                        if layer_id - 1 not in fused_bk:
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
                        alg = propagation_alg if isinstance(propagation_alg, str) else propagation_alg[layer_id]
                        extra = sum_opts if layer_id - 1 not in fused_bk else \
                            dict(sum_opts, stream_children = self.__dict__["_fused_child_tables"][layer_id - 1])
                        if id(layer_group) in given:
                            layer_group.backward(self.node_flows, self.element_flows, self.node_mars, self.element_mars,
                                                 self.params, propagation_alg = alg, op_params = op, **extra)
                        else:
                            layer_group.backward(self.node_flows, self.element_flows, self.node_mars, self.element_mars,
                                                 self.params, propagation_alg = alg, **extra)

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

            def _run_inner_layers():
                # What the layers deferred to the end of the pass, run inside it: a graph recorded from the pass replays it
                with deferred_scope() as deferred:
                    _inner_layers()
                    deferred.run()

            # Callbacks run inside the captured region and could do anything, so they make the pass eager.
            use_graphs = sum_layer_pre_backward_callback is None and sum_layer_post_backward_callback is None
            record, on_repeat = (self.cuda_graphs, True) if record_cudagraph is None else (record_cudagraph, False)
            # The copy the forward made -- the parameters it computed the node values with -- else made here, before the
            # pass (outside a capture, so a replayed pass is preceded by it as an eager one is); for the groups that take
            # it at this batch size (`_param_operand_groups`), as its forward's did
            op_groups = self._param_operand_groups(B)
            if op_groups:
                self._param_operands(B, refresh = None)
            # (the product layer groups whose forward and backward the sum group above takes over: decided before the
            # pass, outside a capture; not where the flows did not start at -inf -- buffers kept or loaded from a cache)
            fused_bk = (self._stream_fusion_bk_groups(B, sum_opts, propagation_alg, op_groups)
                        if use_graphs and not _disable_buffer_init and cache is None else frozenset())
            self._run_inner_pass(_run_inner_layers, 1,
                                 (self.node_flows, self.element_flows, self.node_mars, self.element_mars, self.params),
                                 self._graph_key_options(sum_opts, propagation_alg = propagation_alg,
                                                         param_operands = op_groups, stream_fusion_bk = fused_bk),
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
            verbose: bool = True, cuda_graphs: bool = False, precision: str = "auto",
            param_dtype: torch.dtype = torch.float32) -> nn.Module:
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

    :param param_dtype: the dtype the sum layers' parameters are stored in, ``torch.float32`` (the default) or
        ``torch.bfloat16`` (see :attr:`TensorCircuit.param_dtype`)
    :type param_dtype: torch.dtype

    :returns: the compiled PC with type `torch.nn.Module`
    """
    return TensorCircuit(ns, layer_sparsity_tol = layer_sparsity_tol, max_num_partitions = max_num_partitions,
                         disable_gpu_compilation = disable_gpu_compilation, force_gpu_compilation = force_gpu_compilation,
                         max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block, device = device, verbose = verbose,
                         cuda_graphs = cuda_graphs, precision = precision, param_dtype = param_dtype)
