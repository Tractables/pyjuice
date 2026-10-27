from __future__ import annotations

import warnings

import torch
from typing import Optional, Tuple

from .blockscale.backward import _BackwardMixin
from .blockscale.denominators import _DenominatorsMixin
from .blockscale.forward import _ForwardMixin
from .blockscale.sample import _SampleMixin
from .external_params import ExternalSumParams


class BlockScaleSumParams(_ForwardMixin, _BackwardMixin, _SampleMixin, _DenominatorsMixin, ExternalSumParams):
    """
    A per-sample **multiplicative gate** on each parameter block, supplied externally at call time.

    Every gate owns one scalar per sample. Writing `g(n,c)` for the gate covering the edge from node `n`
    to child `c`, the effective parameters for batch element `b` are

    .. code-block:: text

        theta_b[n, c] = phi[b, g(n,c)] * theta_shared[n, c] / Z_b[n]
        Z_b[n]        = sum_c phi[b, g(n,c)] * theta_shared[n, c]

    normalized over the children of `n`. Factoring out each gate's share of the node's mass,
    `sigma[g,n] = sum_{c in g} theta_shared[n,c]`, shows what the gate actually does:

    .. code-block:: text

        theta_b[n,c] = [ phi[b,g] sigma[g,n] / sum_{g'} phi[b,g'] sigma[g',n] ] * ( theta_shared[n,c] / sigma[g,n] )

    -- it reweights the **gate-level mixture weights** per sample and leaves each gate's internal
    conditional untouched. Two consequences follow directly:

    * it is an **exact no-op when a node has only one gate**, since the reweighting then cancels against
      the normalizer. Whether a node has several gates is entirely the caller's choice of gate sizes; a
      dense transition with one edge block and the default gate size does nothing at all.
    * a gate that does not depend on the parent node is equivalent to multiplying the child's value, i.e.
      to soft evidence on the children, which PyJuice already provides. The capability that is new here
      is a gate that depends on the parent group **and** the child group.

    **Why it is cheap.** Because `phi` is constant within a gate it pulls out of the block matmul, so in
    log space it folds into the child values as an elementwise add -- the dense per-sample parameter
    matrix is never materialized, and the tensor is `[B, num_gates]` rather than anything shaped like the
    parameters. For a 1024-state layer at `block_size = 32` that is 1 MB at batch 256.

    **Tensor layout.** One `float32` tensor of LOG gates:

    * `phi`: `[B, Nk, Ck]`, a dense grid with `Nk = ns.num_nodes / block_size` gates along the node axis
      and `Ck = ns.num_ch_nodes / ch_block_size` along the child axis. `phi[b, i, j]` scales the
      parameters from child gate-block `j` into node gate-block `i`, for sample `b`.

    The grid is indexed by the GATE's own block sizes and the layer's node counts -- it does not depend
    on `ns.block_size`, so asking for gates of a given size gives a grid of exactly that many whatever
    blocking pyjuice happens to use inside. Entries for (node block, child block) pairs the layer does
    not connect are ignored, and their gradients come back zero.

    Log space keeps the gate positive for free and makes the fold an add rather than a multiply.

    :note: the gate tile is deliberately NOT forced to equal the parameter block. Refining along the CHILD axis
           is free (the fold is elementwise in the child index), while refining along the NODE axis costs
           one reduction per subgroup, because a single staged tile can only serve nodes that share a
           gate. Shrinking `ns.block_size` to obtain a finer gate would instead tax the whole layer: the
           node tile of the standard kernels collapses with it, below the 16 rows an efficient MMA wants,
           and the compiled index tensors grow as `K^2 / block_size`.

    :param block_size: the GATE's block size -- how many nodes one gate spans, NOT the node's own
                       `block_size`. `None` means the node's, i.e. one gate per parameter block, and it
                       must divide the node's. Only the full block size is supported so far; splitting
                       the node axis needs a separate kernel.
    :type block_size: Optional[int]

    :param ch_block_size: the GATE's child block size -- how many children one gate spans. `None` means
                          the node's own. Any divisor is allowed and costs nothing, which is why this is
                          the axis to refine.
    :type ch_block_size: Optional[int]

    :param apply_z_correction: make EM exact under a live gate. The gate makes the normalizer
                               `Z = sum_c phi * theta` depend on `theta`, so pyjuice's plain M-step
                               `theta <- normalize(F+)` solves a stationarity condition that is missing
                               the term `sum_b f_b * theta_b`. With this set, the backward also
                               accumulates that term as a DENOMINATOR flow `F-` and the M-step becomes
                               the conditional `theta <- normalize(theta * F+ / F-)`, whose fixed point
                               is the true one (and whose `F+ - F-` is `d LL / d log theta`). Costs a
                               second flow accumulator -- kept in gate space, `ch_block_size` times
                               smaller than `param_flows` -- and ~1% of an EM step. On by default; with
                               `False` the layer allocates nothing and takes the plain M-step. Not
                               implemented for a gate finer than `ns.block_size` along the node axis (the
                               backward raises): pass `False` there. Supported by `FullBatchEM` /
                               `MiniBatchEM` / `Anemone`; the gradient optimizers warn, since their
                               descent direction is uncorrected.
    :type apply_z_correction: bool

    :param tie_external: share one gate tensor across every copy of a tied node, instead of one per copy.
    :type tie_external: bool
    """

    #: This parameterization reweights the per-edge-block partial sums, and those are gone once the
    #: standard kernel has summed them -- `sum_e phi_e M_e` cannot be recovered from `sum_e M_e`. So it
    #: computes the node values itself instead of correcting them afterwards.
    replaces_shared_forward = True

    #: `d LL / d log phi` IS computed, in two halves (see `post_backward_layer`): the `Ntilde` term is
    #: fused into the element-flow backward -- all three forks carry it, Triton, CuTe and small-batch,
    #: since they must agree and the launcher picks between them by speed alone -- where the
    #: per-(parent block, child, sample) partial it needs is already in registers. The `log Z` term is
    #: a small standalone kernel using the forward's cached `log Z` and a `sigma[g,n]` that is
    #: recomputed only when `params` changes.
    #:
    #: This flag is a STATIC claim -- "this parameterization implements gradients at all" -- and cannot
    #: express that support is shape-dependent: a shape no element fork can serve (`ptr_inc_step != 1`,
    #: say) RAISES during the backward rather than returning a silent zero. That is the same contract
    #: the flow kernels use, and it is why the flag stays a plain bool.
    computes_external_grads = True

    def __init__(self, block_size: Optional[int] = None, ch_block_size: Optional[int] = None,
                 apply_z_correction: bool = True, tie_external: bool = False):
        super(BlockScaleSumParams, self).__init__()

        for name, value in (("block_size", block_size), ("ch_block_size", ch_block_size)):
            assert value is None or (isinstance(value, int) and value >= 1), \
                f"`{name}` must be a positive integer or None, got {value}."

        # These are the GATE tile's dimensions, not the node's. Both exist in this class's scope, so
        # every other method goes through `gate_sizes(ns)` rather than reading these directly -- that
        # keeps the one place where `self.block_size` and `ns.block_size` meet down to a single method.
        self.block_size = block_size
        self.ch_block_size = ch_block_size

        # `Z = sum_c phi * theta` depends on `theta`, unlike the low-rank parameterization where the
        # shared parameters' contribution to the normalizer was the constant 1. So the M-step pyjuice
        # performs -- normalize the flows per node -- solves a stationarity condition missing the term
        # `sum_b f_b * theta_b`, and is therefore not exactly EM under a live gate. Setting this makes
        # the layer accumulate that term too (`requests_denom_param_flows` -> the denominator flow,
        # `blockscale/denominators.py`) and switches the M-step to the conditional
        # `theta <- normalize(theta * F+ / F-)`.
        self.apply_z_correction = bool(apply_z_correction)

        # Share one gate tensor across the copies of a tied node (see `storage_owner`).
        self.tie_external = bool(tie_external)

        self._auto_ok = None

    # ------------------------------------------------------------------ identity

    def get_signature(self) -> str:
        # Granularity changes the tensor layout and the kernel specialization, and `tie_external` changes
        # which nodes share storage, so nodes that disagree on any of them must not share a layer.
        node = "full" if self.block_size is None else str(self.block_size)
        child = "full" if self.ch_block_size is None else str(self.ch_block_size)

        return f"BlockScale_n{node}_c{child}" + ("_tied" if self.tie_external else "")

    def _get_constructor(self):
        return BlockScaleSumParams, {"block_size": self.block_size,
                                     "ch_block_size": self.ch_block_size,
                                     "apply_z_correction": self.apply_z_correction,
                                     "tie_external": self.tie_external}

    def __reduce__(self):
        return (self.__class__, (self.block_size, self.ch_block_size,
                                 self.apply_z_correction, self.tie_external))

    def storage_owner(self, ns):
        """With `tie_external`, every copy of a tied node reads one shared gate tensor -- the source's."""
        if self.tie_external and ns.is_tied():
            return ns.get_source_ns()

        return ns

    # ------------------------------------------------------------------ layout

    def gate_sizes(self, ns) -> Tuple[int, int]:
        """
        The gate tile's dimensions in nodes and children, resolved against `ns`.

        `None` means "the node's own block size", so the default is one gate per parameter block.
        """
        return (self.block_size if self.block_size is not None else ns.block_size,
                self.ch_block_size if self.ch_block_size is not None else ns.ch_block_size)

    def gate_counts(self, ns) -> Tuple[int, int]:
        """How many gates tile one parameter block, along the node and the child axis respectively."""
        gate_bs, gate_cbs = self.gate_sizes(ns)

        return ns.block_size // gate_bs, ns.ch_block_size // gate_cbs

    def validate_ns(self, ns) -> None:
        gate_bs, gate_cbs = self.gate_sizes(ns)

        assert ns.block_size % gate_bs == 0, \
            f"the gate's `block_size` ({gate_bs}) must divide the node's ({ns.block_size})."
        assert ns.ch_block_size % gate_cbs == 0, \
            f"the gate's `ch_block_size` ({gate_cbs}) must divide the node's ({ns.ch_block_size})."

        # A gate that varies across the nodes of a block cannot share one staged tile, so it needs one
        # reduction per subgroup. Rejected here rather than silently mis-computed in the kernel.
        # A gate FINER than the node block along the node axis is served by the portable Triton
        # forward, which keeps its M tile inside one node gate so `phi` still folds onto the child
        # operand. The CUDA forward forks and the whole backward do not carry a node-gate axis yet and
        # refuse at their own dispatch, so this is no longer decided here.

        if ns.edge_ids.size(1) == 1 and gate_cbs == ns.ch_block_size:
            warnings.warn(
                f"`BlockScaleSumParams` on a node with a single gate is an exact no-op: the gate cancels "
                f"against the normalizer. Use a smaller gate `ch_block_size` (or more edge blocks) for "
                f"it to have any effect.", RuntimeWarning)

    def tensor_shapes(self, ns, batch_size: int):
        """
        `[B, Nk, Ck]`: one gate per (node gate-block, child gate-block), for every sample.

        `Nk = ns.num_nodes // block_size` and `Ck = ns.num_ch_nodes // ch_block_size` are the gate
        counts along each axis -- a function of the GATE's block sizes and the layer's node counts, and
        deliberately NOT of `ns.block_size`. Whatever blocking pyjuice uses internally, a caller who
        asks for gates of a given size gets a grid of exactly that many.

        The grid is dense: it has an entry for every (node block, child block) pair, including pairs
        this layer's `edge_ids` do not connect. Those entries are ignored -- staging keeps only the
        ones the kernels index (see :func:`storage_shapes`), and their gradients come back zero.
        Paying for them costs a little memory and buys an indexing scheme that does not depend on the
        order or sparsity of `edge_ids`.
        """
        gate_bs, gate_cbs = self.gate_sizes(ns)

        return ((batch_size, ns.num_nodes // gate_bs, ns.num_ch_nodes // gate_cbs),)

    def storage_shapes(self, ns, batch_size: int):
        """
        Stored `[Nk, Ck, B]` -- the caller's own grid, batch innermost.

        The kernel reads a gate for a `(child, batch)` tile, so consecutive threads -- which differ in
        the batch index -- must read consecutive addresses. Nothing else about the layout changes,
        which is what makes staging a permuted copy and not a gather: the compiled table (see
        :func:`storage_offsets`) points each edge block at its own row of the grid, and the kernel's
        `(base + d) * batch + b` addressing then walks the child gates from there exactly as before.
        """
        gate_bs, gate_cbs = self.gate_sizes(ns)

        return ((ns.num_nodes // gate_bs, ns.num_ch_nodes // gate_cbs, batch_size),)

    def storage_perm(self):
        return (1, 2, 0)         # (B, Nk, Ck) -> (Nk, Ck, B)

    def storage_offsets(self, ns):
        """
        Edge block `e` connects node block `edge_ids[0,e]` to child block `edge_ids[1,e]`, so its gates
        start at row `nb * n_node_gates` and column `cb * n_child_gates` of the grid.

        Returning this is what lets the caller's layout be the dense grid at no cost: the indirection
        the kernel already performs -- one compiled base per (row, edge block) -- absorbs it.
        """
        n_node_gates, n_child_gates = self.gate_counts(ns)
        ck = ns.num_ch_nodes // self.gate_sizes(ns)[1]

        nb = ns.edge_ids[0].to(torch.long)
        cb = ns.edge_ids[1].to(torch.long)

        return ((nb * n_node_gates) * ck + cb * n_child_gates,)

    # ------------------------------------------------------------------ execution

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs) -> None:
        raise NotImplementedError(
            "`BlockScaleSumParams` has no per-node reference path; it is served by `forward_layer`."
        )

    @staticmethod
    def _ext_base(ns_tensors, external_params, batch_size: int) -> int:
        """
        Where this layer's gates start in the staging buffer, in per-batch rows.

        RELATIVE to the buffer, so it is a property of the layout and the batch and not of any address:
        reallocate the buffer and this is unchanged. That is what lets `forward_layer` key its plan
        cache on shape alone.
        """
        first_view = ns_tensors[0][1][0]
        return ((first_view.data_ptr() - external_params.data_ptr())
                // external_params.element_size()) // batch_size

    def _gate_nstride(self, layer, nids):
        """
        `Ck` -- the gate grid's row length -- for each compiled row of a partition.

        One node gate down the grid is `Ck` entries along, so this is what turns a node-gate INDEX into
        the row offset the kernel adds. Per row rather than a constexpr because two `ns` sharing a layer
        can have different child counts, and the kernel only reads it when a node block holds more than
        one gate.
        """
        # CACHED, and only ever called when there is more than one node gate. It builds tensors and
        # runs a sort plus a searchsorted, which is nothing per plan but was measured at 1.02-1.07x ->
        # 1.17-1.30x on the gated backward when it ran on every launch of the param kernel.
        # Its own dict, via `__dict__` -- the forward reaches this before `_bs_bw_gate_cache` exists,
        # and `nn.Module.__getattr__` raises rather than returning None for a missing attribute.
        #
        # Keyed by the rows' memory, and the entry keeps `nids` itself alive, so that memory cannot be
        # handed to another tensor while the key exists: a hit is the same compiled rows. The key used to
        # read `nids[0]` and `nids[-1]` to tell recycled addresses apart -- two device-to-host syncs on
        # every call, which also made the gated passes impossible to capture in a CUDA graph.
        cache = layer.__dict__.setdefault("_bs_nstride_cache", {})
        key = (nids.data_ptr(), tuple(nids.shape), tuple(nids.stride()), nids.dtype)
        entry = cache.get(key)
        if entry is not None:
            return entry[1]

        dev = nids.device
        starts = torch.tensor([i.ns._output_ind_range[0] for i in layer.external_node_infos],
                              device = dev, dtype = torch.long)
        cks = torch.tensor([i.ns.num_ch_nodes // self.gate_sizes(i.ns)[1]
                            for i in layer.external_node_infos], device = dev, dtype = torch.long)

        # Which `ns` owns each row: the last one starting at or before it.
        order = torch.argsort(starts)
        starts, cks = starts[order], cks[order]
        pos = (torch.searchsorted(starts, nids.to(dev).to(torch.long), right = True) - 1).clamp(min = 0)
        out = cks[pos].contiguous()
        cache[key] = (nids, out)
        return out

    @staticmethod
    def _require_full_supply(layer, ns_tensors) -> None:
        """
        Refuse a layer whose gated nodes were only PARTLY supplied. Must run before `_ext_base`.

        That base is measured from the FIRST SUPPLIED node, but the gate tables it is added to already
        carry `layer.ext_unit_bases[ns_idx]`, the cursor over ALL of the layer's gated nodes -- so the
        two only compose when the first supplied node IS the layer's first. Supply gates for a later
        node alone and every row shifts by the earlier nodes' slabs: the unsupplied node is gated with
        the supplied one's `phi`, the supplied one reads past the end of the buffer (finite,
        plausible, wrong), and the backward's gradient write faults. Refusing is the honest fix --
        correcting the base alone would not help, since the kernels still run over every row of the
        layer and would read whatever a previous call left staged for the nodes nobody supplied.
        """
        supplied = {id(t[0].ns) for t in ns_tensors}
        missing = [i.ns for i in layer.external_node_infos
                   if isinstance(i.ns.external_params, BlockScaleSumParams)
                   and id(i.ns) not in supplied]
        if missing:
            raise NotImplementedError(
                f"`BlockScaleSumParams`: {len(missing)} of "
                f"{len(missing) + len(ns_tensors)} gated nodes sharing this sum layer were given no "
                f"external parameters. Partial supply is not supported -- the layer's kernels run over "
                f"every node in it, so an unsupplied node has no gate to read. Pass one tensor per "
                f"gated node (a registered group is the convenient way), or give the nodes separate "
                f"layers."
            )

    # Its hooks' only host-side handoff is `_bs_bw_state` (below); the rest is shape-keyed caches
    cuda_graph_safe = True

    def cuda_graph_state(self, layer):
        # What the forward leaves for the backward and the sampler: its plan's `log Z` and geometry
        return getattr(layer, "_bs_bw_state", None)

    def restore_cuda_graph_state(self, layer, state) -> None:
        layer._bs_bw_state = state

    # What BlockScale keeps on a layer -- every one built when missing
    _LAYER_CACHES = ("_bs_nstride_cache", "_bs_bw_state", "_bs_fw_plans", "_bs_sample_nstride", "_bs_denom_geom",
                     "_bs_denom_td_gids", "_bs_sigma_cache", "_bs_bw_gate_cache", "_bs_grad_ext", "_bs_denom_plans",
                     "_bs_logz_plans", "_bs_pd_member", "_bs_post_plans")

    def drop_device_caches(self, layer) -> None:
        super().drop_device_caches(layer)
        for name in self._LAYER_CACHES:
            layer.__dict__.pop(name, None)
        self.__dict__.pop("_bs_fminus_scratch", None)
        self.__dict__.pop("_bs_post_scratch", None)
