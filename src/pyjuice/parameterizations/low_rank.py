from __future__ import annotations

import torch
from typing import Optional

from .sum_parameterization import SumParameterization
from .lowrank import forward as lowrank_forward
from .lowrank import virtual as lowrank_virtual
from .lowrank.backward import _BackwardMixin
from .lowrank.denominators import _DenominatorsMixin
from .lowrank.forward import _ForwardMixin


#: The forms of :class:`LowRank`: how the two factors make a parameter's per-sample scale
LOWRANK_FORMS = ("exp_sum", "sum_exp")


class LowRank(_ForwardMixin, _BackwardMixin, _DenominatorsMixin, SumParameterization):
    """
    A per-sample **multiplicative low-rank scale** on the sum node's parameters, supplied externally at call time.

    A node vector of `M` nodes over `N` children takes two tensors per sample `b`, `A_b` of `[M, k]` and `B_b` of
    `[k, N]`; they make a scale `g_b[i, j]` of every (node, child) pair, and the effective parameters are

    .. code-block:: text

        theta_b[i, j] = theta_shared[i, j] * g_b[i, j] / Z_b[i]
        Z_b[i]        = sum_j theta_shared[i, j] * g_b[i, j]

    normalized over the children of node `i`, as :class:`~.block_scale.BlockScale`' are. `form` says how
    the factors make the scale:

    .. code-block:: text

        "exp_sum":  g_b[i, j] = exp( sum_r A_b[i, r] * B_b[r, j] )          # the log-scale is the rank-k product
        "sum_exp":  g_b[i, j] = sum_r exp( A_b[i, r] ) * exp( B_b[r, j] )   # a mixture of k rank-one scales

    `"exp_sum"` gives every pair an exponent of its own, so it costs one `exp` per (sample, node, child) and the
    samples share no matmul; `"sum_exp"` is `k` matmuls with the shared parameters. Both are defined on the pairs the
    node vector's `edge_ids` connect; a pair without parameters has none here either.

    **`top_k`** (the `"exp_sum"` form). The scale also has a CANDIDATE axis: an entry for each of the first `top_k`
    candidates of a `SoftEvidenceCategorical` variable (`soft_evidence_cat_ids`, in the caller's order) and one shared by
    all the others.

    * On a soft-evidence LEAF (`juice.inputs(..., parameterization = LowRank(rank, top_k = K))`) the candidates are the
      leaf's own values: `theta_b[i, c] = theta[i, c] q_b(c) exp(A_b[i] . B_b[:, s(c)]) / Z_b[i]`, normalized over the
      candidates, `s(c)` the candidate's place capped at `K`; `B` is `[B, K, top_k + 1]`.
    * On a SUM node, multiplied in its one parent product with such a leaf (node `i` with node `i`), the candidate is
      the token that leaf emits: `log theta_b[i, j | x] = log theta[i, j] + A_b[i] . B_b[:, j] + G_b[s(x), j] - log
      Z_b[i, s(x)]`, normalized over `j` for every (node, slot); a third tensor `G [B, top_k + 1, N]`. A missing token
      is summed out under the leaf's own probabilities -- the circuit with a state for every (node, slot), exactly.

    Which token is implied by the structure (the compiler checks that it is unambiguous); nothing names it.

    **Tensor layout.** Two `float32` tensors per node vector, in this order:

    * `A`: `[B, ns.num_nodes, k]`
    * `B`: `[B, k, ns.num_ch_nodes]`

    whatever blocks the node vector is stored in. A registered group (`TensorCircuit.register_external_params_group`)
    concatenates both along axis 1: `A` over the members' nodes, `B` over their ranks.

    **Training.** The backward writes the element and parameter flows itself, and `d LL / d A`, `d LL / d B` into
    the gradient buffers. With `apply_z_correction` it also accumulates the denominator flow's statistic, and the
    M-step is the conditional `theta <- normalize(theta * F+ / F-)`
    (:func:`~.base.SumParameterization.compute_em_correction`): `Z_b` depends on `theta`, so
    `normalize(F+)` alone is not EM here, as for BlockScale. What is kept is

    .. code-block:: text

        W[i, j] = sum_b f_b[i] * g_b[i, j] / Z_b[i]            F-[i, j] = theta[i, j] * W[i, j]

    in the parameter flows' layout (it has no smaller form here), FREE OF THE PARAMETERS as BlockScale's gate-space
    statistic is, and multiplied by them when `F-` is read. That is what an optimizer averaging the flows over steps
    needs (Anemone's momentum): the update is then `normalize(avg F+ / avg W)`, the M-step of the averaged
    statistics. Kept as `F-` itself, the average would carry the parameters of past steps against the current ones
    in `theta * F+ / F-` -- a parameter that grew is pushed further -- and the training is another: MEASURED, a
    low-rank scale built to equal a BlockScale gate, Anemone with momentum 0.9, parameters against BlockScale's
    after 16 steps 3e-2 apart that way, and the same to rounding this way and without momentum either way.

    :note: the passes: `"exp_sum"` through its fused kernels (`lowrank/kernels/c/lowrank_exp_sum.cu`: the scale
           made tile by tile on tensor cores and used where it is made, never stored) where a node vector's parameters
           lie block-major in one run and the rank is at most 64 (`lowrank.forward.exp_sum_fused`) -- under the default
           precision the factors' products in bf16 (`A B` from bf16 hi + lo, ~2^-16 of each product; the gradients'
           products one bf16 term each), under `precision = "fp32"` in three-term TF32 (~fp32) --, `"sum_exp"` through
           the plain sum-layer kernels on a virtual batch (`lowrank.virtual`) where it serves; elsewhere plain torch --
           dense per node vector, `exp_sum` over `[B, M, N]` in chunks of samples, `sum_exp` by float64 matmuls (a
           sample's child values spread over hundreds of nats, more than float32 holds beside one shift a sample).
           Correct on every structure.

    :param rank: the rank `k`
    :type rank: int

    :param form: `"exp_sum"` or `"sum_exp"` (above)
    :type form: str

    :param apply_z_correction: accumulate the denominator flow and take the conditional M-step (default); `False`
                               leaves the plain `normalize(F+)` and allocates nothing
    :type apply_z_correction: bool

    :param tie_external: share one pair of tensors across every copy of a tied node, instead of one per copy
    :type tie_external: bool

    :param flow_precision: `"bf16"` (default) or `"fp8"`, an OPT-IN for `"sum_exp"`: the pass's end's two products
                           -- the parameter flows' `theta * (Ud Vd^T)` and the denominators' `Un Vt^T`, over every
                           (copy, sample, rank) of the pass -- from e4m3 operands (a power-of-2 scale a row, fp32
                           accumulation) where they are bf16 by default (precision "auto", the products large enough:
                           `lowrank.virtual`); about twice the bf16 products' speed, the statistics' entries about
                           16x less accurate (MEASURED against float64 on an HMM of 1024 states, rank 16, batch 64:
                           the M-step's `F+ / F-` a median 3e-4 .. 1.4e-2 and at most 2e-3 .. 0.18 off as the factors'
                           spread goes from 0.3 to 4 nats, against bf16's 2e-5 .. 8e-4 and 1e-4 .. 1e-2; no entry
                           made 0). Under precision "fp32" the products stay fp32. The gradients are not affected.
    :type flow_precision: str
    """

    #: The node values are not the shared ones corrected: the scale differs for every pair, so they are made here
    replaces_shared_forward = True
    #: ... and so are both flows
    replaces_shared_backward = True
    computes_external_grads = True

    @property
    def honors_param_flows_mask(self) -> bool:
        """The `top_k` form's backward takes `param_flows_mask` (`lowrank/top_k.py`)."""
        return self.top_k is not None

    #: A forward may be handed the product layer below, its own forward not run (`stream_fusion_plan`): the mixture's
    #: virtual batch reads the products' values from their children, every other pass forms them first
    fuses_stream_children = True
    @property
    def reads_param_operands(self) -> bool:
        """The virtual batch's plain kernels take the circuit's bf16 copy of tied fp32 parameters
        (`TensorCircuit._param_operands`), as a plain layer's do: its forward `op_params`, its backward the layer's
        per-call copy (`_bk_op_params`), handed to the layer's twin (`lowrank.virtual`) -- the mixture form's, the
        `exp_sum` form's torch passes reading `params`."""
        return self.form == "sum_exp"

    def param_operand_columns(self, batch_size: int) -> int:
        """The mixture form's plain kernels run on `rank` columns a sample (the virtual batch)."""
        return batch_size * self.rank if self.form == "sum_exp" else batch_size

    @property
    def param_operands_min_work(self):
        """The mixture form takes the circuit's bf16 copy whatever its layers' work (`TensorCircuit._param_operand_groups`,
        16 columns at least): its one-launch kernels read their parameters from the copy alone, and without it its
        layers take the plain path. MEASURED (toast, HMM of 31 layers, step ms, the circuit's bound on the work -> the
        copy whatever the work): L128 rank 16 B256 4.96 -> 2.28, B64 4.41 -> 2.20; L256 rank 16 B64 4.51 -> 2.25; L128
        rank 32 B256 9.64 -> 2.47 (the plain path's ~440 launches a step against ~95). The `exp_sum` form: the
        circuit's bound."""
        return 0 if self.form == "sum_exp" else None

    def __init__(self, rank: int, form: str = "exp_sum", apply_z_correction: bool = True, tie_external: bool = False,
                 flow_precision: str = "bf16", top_k: Optional[int] = None):
        super(LowRank, self).__init__()

        assert isinstance(rank, int) and rank >= 1, f"`rank` must be a positive integer, got {rank}."
        assert form in LOWRANK_FORMS, f"`form` must be one of {LOWRANK_FORMS}, got {form!r}."
        assert flow_precision in ("bf16", "fp8"), f"`flow_precision` must be \"bf16\" or \"fp8\", got {flow_precision!r}."
        assert flow_precision == "bf16" or form == "sum_exp", \
            "`flow_precision = \"fp8\"` is the \"sum_exp\" form's (its pass's end products); \"exp_sum\" makes its " \
            "flows in its fused kernels."
        assert top_k is None or (isinstance(top_k, int) and top_k >= 1), f"`top_k` must be a positive integer or None, got {top_k}."
        assert top_k is None or form == "exp_sum", "`top_k` is a form of the \"exp_sum\" scale."

        self.rank = rank
        self.form = form
        self.apply_z_correction = bool(apply_z_correction)
        self.tie_external = bool(tie_external)
        self.flow_precision = flow_precision
        self.top_k = top_k

    # ------------------------------------------------------------------ identity

    def get_signature(self) -> str:
        return f"LowRank_{self.form}_r{self.rank}" + ("" if self.apply_z_correction else "_noz") + \
            ("_tied" if self.tie_external else "") + ("_f8" if self.flow_precision == "fp8" else "") + \
            ("" if self.top_k is None else f"_k{self.top_k}")

    def _get_constructor(self):
        return LowRank, {"rank": self.rank, "form": self.form, "apply_z_correction": self.apply_z_correction,
                         "tie_external": self.tie_external, "flow_precision": self.flow_precision, "top_k": self.top_k}

    def __reduce__(self):
        return (self.__class__, (self.rank, self.form, self.apply_z_correction, self.tie_external, self.flow_precision,
                                 self.top_k))

    def validate_ns(self, ns) -> None:
        """On input nodes: a `SoftEvidenceCategorical` leaf (its candidates are what `top_k` counts), with `top_k`."""
        if ns.is_input():
            from pyjuice.nodes.distributions import SoftEvidenceCategorical
            assert isinstance(ns.dist, SoftEvidenceCategorical), \
                f"`LowRank` on input nodes takes `SoftEvidenceCategorical` leaves, got {type(ns.dist).__name__}."
            assert self.top_k is not None, \
                "`LowRank` on a soft-evidence leaf needs `top_k`: its scale has a column for each of the first `top_k` " \
                "candidates and one shared by the others."

    @property
    def reads_sibling_leaf(self) -> bool:
        """With `top_k`, a sum node's parameters read the token of the leaf beside it."""
        return self.top_k is not None

    def validate_sibling_leaf(self, ns, leaf) -> None:
        from pyjuice.nodes.distributions import SoftEvidenceCategorical
        assert isinstance(leaf.dist, SoftEvidenceCategorical) and len(leaf.scope) == 1, \
            f"`LowRank(top_k)` reads the candidates of a univariate `SoftEvidenceCategorical` leaf beside its node; got " \
            f"{type(leaf.dist).__name__} over {len(leaf.scope)} variable(s)."

    def input_forward(self, layer, data, node_mars, kwargs) -> None:
        """An input layer's pass for its node vectors that were given tensors (`lowrank/top_k.py`); `data` [num_vars, B]
        and `kwargs` as the caller gave them."""
        from .lowrank import top_k as lowrank_top_k
        lowrank_top_k.leaf_forward(self, layer, data, node_mars, kwargs)

    def input_forward_replaces(self, layer, kwargs) -> bool:
        """Every node vector of the leaf layer given tensors: the values are all its (`lowrank/top_k.py`)."""
        from .lowrank import top_k as lowrank_top_k
        staged = lowrank_top_k._staged(kwargs)
        return bool(staged) and all(ns in staged for ns in layer.nodes)

    def input_backward(self, layer, data, node_flows, node_mars, kwargs, logspace_flows: bool = False,
                       missing_mask = None, batch_first: bool = True, compute_param_flows: bool = True) -> bool:
        """An input layer's backward where its node vectors were given tensors (`lowrank/top_k.py`): True, done; False
        (none given), the plain leaf's backward."""
        from .lowrank import top_k as lowrank_top_k
        return lowrank_top_k.leaf_backward(self, layer, data, node_flows, kwargs, logspace_flows, missing_mask, batch_first,
                                           compute_param_flows)

    def input_conditional(self, layer, node_flows, kwargs, rows: dict, num_rows: int, over_candidates: bool = False):
        """The conditional of the leaf's variables over their candidates (`lowrank/top_k_queries.py`)."""
        from .lowrank import top_k_queries
        return top_k_queries.conditional(self, layer, node_flows, kwargs, rows, num_rows, over_candidates)

    def input_sample(self, layer, samples, node_flows, kwargs, seed = None) -> bool:
        """The leaf's draw where its node vectors were given tensors: each selected node's candidate, within the slot the
        sum beside it drew (`lowrank/top_k_queries.py`); False (none given), the plain leaf's."""
        if not self.input_forward_replaces(layer, kwargs):
            return super().input_sample(layer, samples, node_flows, kwargs, seed = seed)
        from .lowrank import top_k_queries
        top_k_queries.sample_leaves(self, layer, samples, node_flows, kwargs)
        return True

    def sample_levels(self, pc, plan, layer_id: int, node_samples, seed_ptr, conditional: bool, kwargs: dict) -> int:
        """With `top_k`: a run of its sum levels walked by one kernel, a program a sample (`lowrank/top_k_queries.py`)."""
        if self.top_k is None:
            return 0
        from .lowrank import top_k_queries
        return top_k_queries.walk(pc, plan, layer_id, node_samples, seed_ptr, conditional, kwargs)

    def sample_layer(self, layer, ns_tensors, node_mars, element_mars, params, node_samples, element_samples, rows,
                     erows, seed_ptr, conditional: bool = False, **kwargs) -> None:
        """With `top_k`: each selected node's slot and child drawn together (`lowrank/top_k_queries.py`)."""
        if self.top_k is None:
            return super().sample_layer(layer, ns_tensors, node_mars, element_mars, params, node_samples,
                                        element_samples, rows, erows, seed_ptr, conditional = conditional, **kwargs)
        from .lowrank import top_k_queries
        top_k_queries.sample_sums(self, layer, ns_tensors, element_mars, params, node_samples, element_samples, rows,
                                  erows, seed_ptr, conditional, kwargs)

    def storage_owner(self, ns):
        """With `tie_external`, every copy of a tied node reads one shared pair of tensors -- the source's."""
        if self.tie_external and ns.is_tied():
            return ns.get_source_ns()

        return ns

    @property
    def requests_denom_param_flows(self) -> bool:
        return self.apply_z_correction

    # ------------------------------------------------------------------ layout

    def tensor_shapes(self, ns, batch_size: int):
        """What the caller hands over: `A [B, M, K]`, `B [B, K, N]`; with `top_k`, on a sum node also `G [B, top_k + 1, N]`
        (a row over the children for each of the first `top_k` candidates of the token emitted beside the node and one
        for the others), and on a soft-evidence leaf `B [B, K, top_k + 1]` (the candidates' columns)."""
        if ns.is_input():
            return ((batch_size, ns.num_nodes, self.rank), (batch_size, self.rank, self.top_k + 1))
        shapes = ((batch_size, ns.num_nodes, self.rank), (batch_size, self.rank, ns.num_ch_nodes))
        if self.top_k is not None:
            shapes = shapes + ((batch_size, self.top_k + 1, ns.num_ch_nodes),)
        return shapes

    def storage_shapes(self, ns, batch_size: int):
        """Stored as the caller gives them, `A [B, M, K]`, `B [B, K, N]`, either form, and the gradients in the same
        layout. `"exp_sum"`: read IN PLACE at rank 16 (`reads_in_place`: its fp16-rounded factors' kernels take a
        sample's block where it lies, `lowrank/kernels/c/lowrank_exp_sum_h.cu`), else the staging copy a plain copy
        (its other fused kernels take a sample's block whole). `"sum_exp"`: read IN PLACE by the forward
        (`reads_in_place`): the kernels that
        serve its one-launch forms take the caller's tensors as they are, the rest a batch-innermost copy -- `A [M, K,
        B]`, `B [K, N, B]`, as the circuit's own buffers are: their threads run over the samples -- made when first
        asked for in the pass (`lowrank/virtual.py`'s `_inner`), their gradients back the same way."""
        return self.tensor_shapes(ns, batch_size)

    def storage_perm(self):
        return None

    def takes_external_rows(self, ns) -> bool:
        """With `top_k` (sum nodes and leaves): the passes read each sample's rows through the map (`lowrank/top_k.py`)."""
        return self.top_k is not None

    def reads_in_place(self, ns) -> bool:
        """The forward's staging hands the layers the caller's tensors themselves (no copy): `"sum_exp"` where the
        virtual batch can serve it (the CUDA extension built: `lowrank/virtual.py`); `"exp_sum"` where its fused
        kernels can (`lowrank/forward.py`'s `exp_sum_fused`: a rank they hold, their extension built -- the
        fp16-rounded factors' kernels at the ranks they hold, `exp_sum_h`, the bf16 / TF32 ones at the others and at
        `precision = "fp32"`, all taking each sample's block where it lies and adding into the gradients where they
        lie), other ranks copied. With `top_k` (sums and leaves): read where they lie, every pass taking a node
        vector's views as they come (the fused forward's kernels a batch stride: `lowrank/top_k_fused.py`; the
        backward's, the leaves' and the torch passes any strides). Another leaf: copied."""
        if self.top_k is not None:
            return True
        if ns.is_input():
            return False
        if self.form == "exp_sum":
            if lowrank_forward.EXP_SUM_H and self.rank <= lowrank_forward.EXP_SUM_H_RANK_MOST and \
                    lowrank_forward.exp_sum_h_module() is not None:
                return True
            return lowrank_forward.EXP_SUM_FUSED and self.rank <= lowrank_forward.EXP_SUM_FUSED_RANK_MOST and \
                lowrank_forward.exp_sum_module() is not None
        return self.form == "sum_exp" and lowrank_virtual._module() is not None

    def external_dtypes(self, ns):
        """float32, and float16 / bfloat16 where the "exp_sum" form's fused kernels read the factors in place
        (`reads_in_place`): the fp16 kernels (`lowrank/forward.py`'s `exp_sum_h`) take fp16 factors of their padded
        ranks (16, 32, 64) as they lie -- the rounding they make of float32 ones anyway, `A' = log2 e A` rounded once
        more from fp16 A (`lowrank/kernels/c/lowrank_exp_sum_h.cu`'s `scale_h2`) -- with no copy, the others (bf16
        ones, an autocast head's; fp16 ones of other ranks) through their fp16 copy, as float32 ones; the bf16 / TF32
        kernels (the other ranks, `precision = "fp32"`) take any of the three as they lie, their values exactly
        (`lowrank_exp_sum.cu`'s prologues). float32 and float16 where the "sum_exp" form reads them in place: its
        one-launch forward and backward, their side CTAs and the operands' kernels take fp16 factors as they lie, read
        as float32 -- the conversion exact: the bits of float32 factors of those values --, its other paths (the plain
        kernels' batch-innermost copies, the e4m3 pass's end, the copies of a source of factors of both types) float32
        copies; the gradients float32 either way."""
        if self.form == "exp_sum" and self.reads_in_place(ns):
            return (torch.float32, torch.float16, torch.bfloat16)
        if self.form == "sum_exp" and self.reads_in_place(ns):
            return (torch.float32, torch.float16)
        return (torch.float32,)

    def initializes_grads(self, ns) -> bool:
        """Both forms: the backward claims each gradient destination it writes in place -- `"sum_exp"`'s one-launch
        form storing into all of it, `"exp_sum"`'s fp16 kernels zeroing a chunk's slice ahead of their adds (in the
        L2: no DRAM round trip of zeros), the torch passes copying, the other paths zeroing it before they add
        (`lowrank/virtual.py`, `backward.py`)."""
        return True

    def storage_offsets(self, ns):
        """Where an edge block's entries begin in each slot, in per-batch units (the layer's tables; the passes here
        address the slots by their own geometry): `A` at the node block's first node, `B` at the child block's first
        child."""
        nb, cb = ns.edge_ids[0].long(), ns.edge_ids[1].long()
        return (nb * (ns.block_size * self.rank), cb * ns.ch_block_size)

    def _as_given(self, tensors):
        """The staged factors as the caller's `(A [B, M, K], B [B, K, N])`: themselves (stored so, either form)."""
        return tensors[0], tensors[1]

    def _to_stored(self, dA, dB):
        """The caller-layout gradients `(dA [B, M, K], dB [B, K, N])` in the storage's layout: themselves."""
        return dA, dB

    def _geometry(self, layer, ns_info, device):
        """
        What ties a node vector's dense `[M, N]` grid to the circuit's buffers, built once per (layer, node vector):

        * `grid` -- `(off, Mb, Nb, cbs, bs)` where the node vector's parameters lie BLOCK-MAJOR in one run: every
          cell's edge block in its place, node block by node block, each in child order (however the edge blocks
          are listed), the run from `off` on in its source's range (`p0` in `params`) -- the grid is then a strided
          view of the run (`_dense_params`, `_scatter_add`), and the tables below are not made; else None, and
        * `flat`, `row`, `pid`, `pfid`, `local` -- for every stored parameter, its cell of the grid (`i * N + j`),
          its node `i`, its index in `params` and in `param_flows`, and its place in the node vector's own range of
          either. An edge block `e` holds `[ch_block_size, block_size]`, child then node, from `_param_ids[e]` on;
          a tied copy reads its source's parameters, and its flows lie in its own range as the parameters do in
          theirs;
        * `eid`, `has` -- every child's element id, and whether an edge reaches it at all; `cols`, `kids` -- the
          children an edge reaches (their columns of the grid, their element ids), `full` -- whether that is all;
          `rows` -- every child's element id, -1 where no edge reaches it (the kernels' form).
        """
        cache = layer.__dict__.setdefault("_lr_geometry", {})
        hit = cache.get(id(ns_info.ns))
        if hit is not None and hit["eid"].device == device:
            return hit

        ns = ns_info.ns
        src = ns.get_source_ns() if ns.is_tied() else ns
        bs, cbs = ns.block_size, ns.ch_block_size
        M, N = ns.num_nodes, ns.num_ch_nodes
        nb = ns.edge_ids[0].to(device = device, dtype = torch.long)
        cb = ns.edge_ids[1].to(device = device, dtype = torch.long)
        m = torch.arange(bs, device = device)
        c = torch.arange(cbs, device = device)

        pids0 = src._param_ids.to(device = device, dtype = torch.long)
        E, Mb, Nb = nb.numel(), M // bs, N // cbs
        grid = None
        if E == Mb * Nb and E > 0:
            # (each edge block at its cell's place from the run's first id: the ids are distinct, so then the cells
            # are, and every cell is reached)
            base = int(pids0.min())
            if bool(torch.equal(pids0, base + (nb * Nb + cb) * (cbs * bs))):
                grid = (base - src._param_range[0], Mb, Nb, cbs, bs)

        eid = torch.full((N,), -1, dtype = torch.long, device = device)
        ch_eids = ns_info.ch_eids.to(device = device, dtype = torch.long)
        eid[(cb[:, None] * cbs + c[None, :]).reshape(-1)] = (ch_eids[:, None] + c[None, :]).reshape(-1)

        has = eid >= 0
        reached = torch.nonzero(has).reshape(-1)
        geo = dict(M = M, N = N, grid = grid, p0 = None if grid is None else src._param_range[0] + grid[0],
                   eid = eid.clamp(min = 0), has = has, cols = reached, kids = eid[reached],
                   full = reached.numel() == N, rows = eid.contiguous())
        if grid is None:
            rows = (nb[:, None, None] * bs + m[None, None, :]).expand(-1, cbs, -1)   # [E, cbs, bs]
            cols = (cb[:, None, None] * cbs + c[None, :, None]).expand(-1, -1, bs)
            within = (c[:, None] * bs + m[None, :]).reshape(-1)                 # [cbs * bs]
            pid = (pids0[:, None] + within[None, :]).reshape(-1)
            local = pid - src._param_range[0]
            geo.update(flat = (rows * N + cols).reshape(-1), row = rows.reshape(-1), pid = pid, local = local,
                       pfid = ns._param_flow_range[0] + local)
        cache[id(ns)] = geo
        return geo

    @staticmethod
    def _info(layer, ns):
        for info in layer.external_node_infos:
            if info.ns is ns:
                return info
        raise KeyError(f"{ns} is not a node vector of this layer.")

    @staticmethod
    def _dense_params(geo, params, dtype = torch.float32) -> torch.Tensor:
        """The node vector's shared parameters on its `[M, N]` grid, in `dtype`, 0 where it has none: one strided copy
        of their run where they lie block-major (`grid`), converting as it copies, a gather and a scatter else.
        `params`: the circuit's, or its bf16 copy (`TensorCircuit._param_operands`), which is indexed alike."""
        M, N, grid = geo["M"], geo["N"], geo["grid"]
        if grid is not None:
            _, Mb, Nb, cbs, bs = grid
            theta = torch.empty(Mb, bs, Nb, cbs, dtype = dtype, device = params.device)
            theta.copy_(params[geo["p0"]:geo["p0"] + M * N].view(Mb, Nb, cbs, bs).permute(0, 3, 1, 2))
            return theta.view(M, N)
        theta = torch.zeros(M * N, dtype = dtype, device = params.device)
        theta[geo["flat"]] = params[geo["pid"]].to(dtype)
        return theta.view(M, N)

    @staticmethod
    def _scatter_add(geo, dst: torch.Tensor, base: int, values: torch.Tensor, params: Optional[torch.Tensor] = None):
        """`dst[base + local] += (values * theta)[flat]`: a grid's values (`[M, N]`; with `params`, times the node
        vector's parameters, read from them where they lie) onto the slots of its stored parameters in a range of `dst`
        from `base` on (its flows', its denominators'): one strided add where they lie block-major (`grid`), a gather
        and an `index_add_` else."""
        grid = geo["grid"]
        if grid is None:
            v = values.reshape(-1)[geo["flat"]]
            if params is not None:
                v = v * params[geo["pid"]]
            dst.index_add_(0, base + geo["local"], v.to(dst.dtype))
            return
        off, Mb, Nb, cbs, bs = grid
        out = dst[base + off:base + off + geo["M"] * geo["N"]].view(Mb, Nb, cbs, bs)
        perm = lambda t: t.view(Mb, bs, Nb, cbs).permute(0, 2, 3, 1)
        if params is None:
            out.add_(perm(values))
        else:
            out.addcmul_(perm(values), params[geo["p0"]:geo["p0"] + geo["M"] * geo["N"]].view(Mb, Nb, cbs, bs))

    @staticmethod
    def _child_values(geo, element_mars) -> torch.Tensor:
        """`[B, N]`: the children's log-values, -inf for a child no edge reaches."""
        lc = element_mars[geo["eid"]].t()
        if not geo["full"]:
            lc = lc.masked_fill(~geo["has"][None, :], -float("inf"))
        return lc

    def _require_full_supply(self, layer, ns_tensors) -> None:
        """The node values are made here, so a layer's node vectors are supplied all or none."""
        if len(ns_tensors) != len(layer.external_node_infos):
            raise NotImplementedError(
                f"`LowRank`: {len(layer.external_node_infos) - len(ns_tensors)} of "
                f"{len(layer.external_node_infos)} node vectors sharing this sum layer were given no external "
                f"parameters. Supply all of them (a registered group is the convenient way), or none."
            )

    def compile(self, layer) -> None:
        """What the virtual batch's twin of the layer is made from (`lowrank.virtual._twin`), noted while the layer is
        as compilation left it: which of its attributes are caches -- every empty container and every None then --
        and which are its compiled tables and settings."""
        lowrank_virtual.note_compiled(layer)

    def drop_device_caches(self, layer) -> None:
        super().drop_device_caches(layer)
        for name in ("_lr_geometry", "_lr_state", "_lr_denom_offsets", "_lr_kids_own", "_lr_succ", "_lrv_state",
                     "_lrv_disjoint", "_lrv_link", "_lrv_want", "_lrv_wasted", "_lrv_twin", "_lrv_stream",
                     "_lrv_stream_bk", "_lrv_kd", "_lrv_fw_fast", "_lrv_serves", "_lrv_bw_fast", "_lr_top_k_infos",
                     "_lr_top_k_settled", "_lr_top_k_pairs"):
            layer.__dict__.pop(name, None)
