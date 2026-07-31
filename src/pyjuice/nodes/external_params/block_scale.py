from __future__ import annotations

import warnings

import torch
from typing import Optional, Tuple

from .external_params import ExternalSumParams


_BUFFER_KWARG = None

# Packs (parent node id, child element id) into one int64 key for the backward gate lookup. Node and
# element ids are global row indices into `node_mars` / `element_mars`, so this holds for any circuit
# that fits in memory; asserted where the keys are built.
_KEY_STRIDE = 1 << 32


def _buffer_kwarg() -> str:
    """The staging-buffer kwarg name, resolved once (a per-call import shows up in profiles)."""
    global _BUFFER_KWARG
    if _BUFFER_KWARG is None:
        from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_BUFFER_KWARG
        _BUFFER_KWARG = EXTERNAL_PARAMS_BUFFER_KWARG
    return _BUFFER_KWARG


class BlockScaleSumParams(ExternalSumParams):
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

    :param apply_z_correction: whether the parameter flows include the term coming from `Z`'s own
                               dependence on `theta`. See :func:`post_backward_layer`.
    :type apply_z_correction: bool

    :param tie_external: share one gate tensor across every copy of a tied node, instead of one per copy.
    :type tie_external: bool
    """

    #: This parameterization reweights the per-edge-block partial sums, and those are gone once the
    #: standard kernel has summed them -- `sum_e phi_e M_e` cannot be recovered from `sum_e M_e`. So it
    #: computes the node values itself instead of correcting them afterwards.
    replaces_shared_forward = True

    #: The element and parameter flows ARE computed (by two forks of the standard backward kernels, so
    #: `replaces_shared_backward` stays False and the standard backward's own table derivation is
    #: reused -- see `pre_backward_layer`). `d LL / d log phi` is not: its second term carries
    #: `sigma[g,n] = sum_{c in g} theta[n,c]`, which the forward stopped materializing once the
    #: normalizer became a second contraction against the same operand, so it needs its own kernel.
    computes_external_grads = False

    def __init__(self, block_size: Optional[int] = None, ch_block_size: Optional[int] = None,
                 apply_z_correction: bool = False, tie_external: bool = False):
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
        # shared parameters' contribution to the normalizer was the constant 1. So the exact derivative
        # w.r.t. `log theta` carries a second term, `- theta * sum_b f phi / Z`. Off by default: the first
        # term alone still sums to `sum_b f` per node, so it drops straight into PyJuice's EM optimizers,
        # whereas the corrected quantity sums to zero and is a gradient, not an expected count.
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
        if gate_bs != ns.block_size:
            raise NotImplementedError(
                f"a gate `block_size` smaller than the node's is not supported yet (got {gate_bs} vs "
                f"{ns.block_size}). Refining the gate's `ch_block_size` is free and is usually what is "
                f"wanted; splitting the node axis needs a separate kernel."
            )

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

    def _build_plan(self, layer, ns_tensors, node_mars, element_mars, params, external_params):
        """
        Resolve every per-layer launch argument once, and check each kernel's assumptions.

        TWO kernels serve this type, and which applies is a property of the shape:

          * the CuTe/TMA fork, for batches it can tile (`batch % 64 == 0`) on sm_90+ with CUTLASS. It
            carries the normalizer as a gate-factored contraction against a precomputed `sigma`;
          * a plain-CUDA small-batch kernel, one warp per 32 nodes, which accumulates the normalizer
            inline and needs neither `batch % 64` nor `num_edges % 64` nor CUTLASS.

        Both are collected here and the choice is MEASURED, so where they overlap the faster one wins
        rather than the one that happened to be checked first. Where neither applies this raises: there
        is no Triton fallback for this parameterization.
        """
        from .kernels.c import get_cute_module, get_sb_module
        import pyjuice.layer.kernels.c as ck

        mod = get_cute_module()
        sb_mod = get_sb_module()
        if mod is None and sb_mod is None:
            raise NotImplementedError(
                "`BlockScaleSumParams` needs one of its CUDA extensions, and neither is available here "
                "(the CuTe forward needs nvcc, CUTLASS and sm_90+; the small-batch forward needs only "
                "nvcc). There is no Triton fallback -- see the compile warning above."
            )

        if getattr(layer, "ext_slots", None) is None:
            raise NotImplementedError(
                "`BlockScaleSumParams` needs the compiled edge-block tables, which require a "
                "batch-innermost storage layout."
            )

        ns0 = ns_tensors[0][0].ns
        batch_size = node_mars.size(1)
        block_size = layer.block_size

        n_node_gates, n_child_gates = self.gate_counts(ns0)
        _, gate_cbs = self.gate_sizes(ns0)
        node_cbs = ns0.ch_block_size

        if n_node_gates != 1:
            raise NotImplementedError(
                f"a gate spanning fewer nodes than the block cannot share the staged tile; got "
                f"{n_node_gates} node gates per block."
            )

        # The tiles that fit BOTH this layer's shape and this DEVICE. The kernel decides the second
        # part: its shared-memory need depends on the gate width, and the opt-in ceiling is a property
        # of the part (48 / 64 / 100 / 227 KB), not something to hardcode.
        cfgs = [tuple(c) for c in mod.configs()] if mod is not None else []
        valid = ([int(i) for i in mod.fitting_configs(block_size, batch_size, gate_cbs)]
                 if mod is not None else [])
        cfg = valid[0] if valid else 0    # provisional; measured once the launch args exist

        # The small-batch kernel needs only 32-node groups, but its children must be contiguous across
        # the WHOLE row rather than within each edge tile -- it walks the edge axis with a single base.
        sb_ok = (sb_mod is not None) and (block_size % 32 == 0)

        first_view = ns_tensors[0][1][0]
        ext_base = ((first_view.data_ptr() - external_params.data_ptr())
                    // external_params.element_size()) // batch_size

        dev = node_mars.device
        calls, sb_calls, shift_args = [], [], []
        for partition_id in range(layer.num_fw_partitions):
            nids = layer.partitioned_nids[partition_id]
            cids = layer.partitioned_cids[partition_id]
            pids = layer.partitioned_pids[partition_id]
            gate = layer.ext_slots[0][partition_id]

            num_edges = cids.size(1)
            if num_edges % node_cbs != 0:
                raise NotImplementedError(
                    f"the compiled edge count ({num_edges}) must be a whole number of child blocks "
                    f"({node_cbs})."
                )

            rows = nids.size(0)
            n_eblks = num_edges // node_cbs
            log_z = torch.empty([rows * block_size * batch_size], dtype = torch.float32, device = dev)

            # `log Z` is what the backward needs to turn the stored `log N - log Z` back into `log N`
            # (see `pre_backward_layer`). Collected here rather than dug out of `calls` afterwards,
            # because the two kernels order their arguments differently.
            shift_args.append((nids, log_z, rows))

            c2 = cids.to(torch.int64)
            p2 = pids.to(torch.int64)
            ar_e = torch.arange(num_edges, device = cids.device, dtype = torch.int64)

            # ---- the CuTe fork's layout: contiguous children and block_size-strided params PER TILE
            if valid and num_edges % 64 == 0:
                knt = num_edges // 64
                c3, p3 = c2.view(-1, knt, 64), p2.view(-1, knt, 64)
                ebase = c3[:, :, 0].contiguous()
                pbase = p3[:, :, 0].contiguous()
                ar = torch.arange(64, device = cids.device, dtype = torch.int64)

                if (torch.equal(c3, ebase.unsqueeze(-1) + ar.view(1, 1, -1))
                        and torch.equal(p3, pbase.unsqueeze(-1) + ar.view(1, 1, -1) * block_size)):
                    calls.append((nids, ebase, pbase, pids, gate, log_z,
                                  block_size, num_edges, node_cbs, gate_cbs, n_node_gates,
                                  ext_base, cfg))

            # ---- the small-batch kernel's layout: the same, but across the WHOLE row at once
            if sb_ok:
                eb_row = c2[:, 0].contiguous()
                pb_row = p2[:, 0].contiguous()
                if (torch.equal(c2, eb_row.unsqueeze(-1) + ar_e.view(1, -1))
                        and torch.equal(p2, pb_row.unsqueeze(-1) + ar_e.view(1, -1) * block_size)):
                    sb_calls.append((nids, eb_row, pb_row, gate, log_z,
                                     block_size, num_edges, node_cbs, gate_cbs, ext_base, 0))

        # Every partition must be served by the same kernel -- a plan that covers only some of them
        # would silently leave the rest of the layer unevaluated.
        if len(calls) != layer.num_fw_partitions:
            calls = []
        if len(sb_calls) != layer.num_fw_partitions:
            sb_calls = []

        if not calls and not sb_calls:
            raise NotImplementedError(
                f"no block-scale forward applies to this layer: block_size={block_size}, "
                f"batch={batch_size}, gate ch_block_size={gate_cbs}, CuTe tiles="
                f"{[(c[0], c[1]) for c in cfgs]}. The CuTe forward needs `block_size % BM == 0`, "
                f"`batch % BN == 0`, `num_edges % 64 == 0` and enough shared memory; the small-batch "
                f"forward needs `block_size % 32 == 0`. Both need each row's children contiguous and "
                f"its parameters strided by `block_size`."
            )

        # ---- PICK THE KERNEL AND ITS CONFIG BY MEASURING THEM ----
        #
        # No rule gets this right, for either choice. Between the two kernels: the CuTe fork wins
        # comfortably at large batch, but its cost is a serial walk over each row's edge tiles, so as
        # the batch shrinks towards its 64-wide tile it stops being obviously better than the
        # small-batch kernel, which tiles the node axis instead. Within the CuTe fork: a narrower tile
        # gives more blocks, which a latency-bound kernel wants, but the accumulator fragment is
        # `BM * BN / threads` registers and the fork carries three of them, so a tile with twice the
        # WARPS halves the fragment and doubles the warps resident per SM. Those pull opposite ways and
        # which wins flips with the batch size.
        #
        # Safe to run on the LIVE buffers because every write is an assignment: `node_mars` and
        # `log_z` are overwritten, never accumulated, so a repeated trial leaves exactly what one
        # run would. `forward_layer` then does the real launch.
        #
        # A read-accumulate-write kernel must NOT be tuned this way -- each trial would add its
        # contribution again. The backward is exactly that, so when it is autotuned it has to run into
        # a scratch clone, as `sum_layer`'s param-flow and element-flow tuners already do.
        def _with_cfg(argslist, i):
            return [tuple(a[:-1]) + (i,) for a in argslist]

        # Each trial launches the layer REPEATS times. The candidates differ by a few microseconds and
        # a single launch spends comparable time in Python and in the driver, so timing one launch
        # measures mostly that: with eight candidates the tuner mis-picked three of eight shapes,
        # costing 29% on the worst. Repeating inside the timed region amortizes the overhead away, and
        # is only safe because every launch here is idempotent (see the note above).
        REPEATS = 10

        def _runner(fn, argslist, i):
            tuned = _with_cfg(argslist, i)

            def run():
                for _ in range(REPEATS):
                    for args in tuned:
                        fn(node_mars, element_mars, params, external_params, *args)

            return run

        cands = []
        if calls:
            for i in valid:
                cands.append((("cute", i), _runner(mod.blockscale_forward, calls, i)))
        if sb_calls:
            for i in range(len(sb_mod.blockscale_sb_configs())):
                cands.append((("sb", i), _runner(sb_mod.blockscale_sb_forward, sb_calls, i)))

        best = ck.autotune(cands, warmup = 2, reps = 9) or cands[0][0]
        kind, best_cfg = best

        if kind == "cute":
            out_mod, fname, calls = mod, "blockscale_forward", _with_cfg(calls, best_cfg)
        else:
            out_mod, fname, calls = sb_mod, "blockscale_sb_forward", _with_cfg(sb_calls, best_cfg)

        # Everything the backward needs that only the forward knows: where this layer's gates start in
        # the staging buffer, the gate geometry the kernels specialize on, and the per-partition `log Z`.
        layer._bs_bw_state = {
            "block_size": block_size,
            "batch_size": batch_size,
            "ext_base": ext_base,
            "gate_cbs": gate_cbs,
            "node_cbs": node_cbs,
            "shift_args": shift_args,
        }

        return out_mod, fname, calls

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Compute this layer's node values under the gate, REPLACING the standard sum forward.

        Unlike a corrective parameterization this owns the whole computation: it reweights the
        per-edge-block partial sums, and those do not survive the standard kernel.
        """
        if len(ns_tensors) == 0:
            return None

        external_params = kwargs.get(_buffer_kwarg(), None)
        if external_params is None:
            return None

        ptrs = (node_mars.data_ptr(), element_mars.data_ptr(), params.data_ptr(),
                external_params.data_ptr(), node_mars.size(1))

        entry = getattr(layer, "_bs_fw_plan", None)
        if entry is None or entry[0] != ptrs:
            entry = (ptrs, self._build_plan(layer, ns_tensors, node_mars, element_mars, params,
                                            external_params))
            layer._bs_fw_plan = entry

        mod, fname, calls = entry[1]
        fn = getattr(mod, fname)
        for args in calls:
            fn(node_mars, element_mars, params, external_params, *args)

        return None

    # ------------------------------------------------------------------ backward

    def _gate_bw_table(self, layer, chids, ele_ebase):
        """
        `gate_bw[eb, kt]` -- the staging row of the gate on the edge from the parent block of tile `kt`
        into child block `eb`, in per-batch units, `-1` where the two are not connected.

        The element-flow kernel walks the TRANSPOSE of the forward's indexing: one row per CHILD block
        and one column per incident parent tile, where the forward has one row per node block. So the
        forward's `ext_slots` cannot be reused, but its *key* can -- an edge block is identified
        globally by (first node id of its parent block, first element id of its child block), which
        `par_nids` and `ch_eids` already hold, and `storage_offsets` gives the row it maps to.

        Built by lookup rather than by re-deriving the backward tables: `ele_ebase` has already been
        through partitioning, `parids` reconstruction and edge trimming, and a second derivation that
        had to track all three would be one more thing to keep in step with `SumLayer`.
        """
        dev = chids.device
        chids = chids.to(torch.long)

        keys, vals, ch_all = [], [], []
        for ns_info in layer.external_node_infos:
            ns = ns_info.ns
            par = ns_info.par_nids.to(dev).to(torch.long)
            ch = ns_info.ch_eids.to(dev).to(torch.long)
            off = ns.external_params.storage_offsets(ns)[0].to(dev).to(torch.long)

            keys.append(par * _KEY_STRIDE + ch)
            vals.append(off + layer.ext_unit_bases[ns_info.ns_idx][0])
            ch_all.append(ch)

        key = torch.cat(keys)
        val = torch.cat(vals)
        assert int(torch.cat(ch_all).max()) < _KEY_STRIDE, \
            "circuit too large for the packed (parent, child) backward gate key."
        order = torch.argsort(key)
        key, val = key[order], val[order]

        # The parent BLOCK a tile belongs to: tiles are `TILE_SIZE_K`-wide slices of a node block, so
        # the block is the greatest node-block start not exceeding the tile's first node. Found by
        # search rather than by arithmetic on `block_size`, which would assume every `ns` in the layer
        # starts on the same grid.
        nid_all = torch.cat([layer.partitioned_nids[p].to(dev).to(torch.long)
                             for p in range(layer.num_fw_partitions)])
        nid_sorted = torch.sort(nid_all)[0]
        flat = ele_ebase.reshape(-1).to(torch.long)
        blk = nid_sorted[(torch.searchsorted(nid_sorted, flat, right = True) - 1).clamp(min = 0)]

        q = blk * _KEY_STRIDE + chids.view(-1, 1).expand_as(ele_ebase).reshape(-1)

        pos = torch.searchsorted(key, q).clamp(max = key.numel() - 1)
        hit = key[pos] == q
        gate = torch.where(hit, val[pos], torch.full_like(q, -1))

        # Every edge block whose child this partition owns must have been found. A miss would not
        # crash -- the kernel reads `-1` as "no gate" and drops the parent's contribution entirely --
        # so without this check a mis-derived key would surface as quietly missing flows.
        n_expect = int(torch.isin(torch.cat(ch_all), chids).sum())
        n_found = int(torch.unique(q[hit]).numel()) if bool(hit.any()) else 0
        assert n_found == n_expect, \
            f"the block-scale backward gate table matched {n_found} of {n_expect} edge blocks; " \
            f"`par_nids` / `ch_eids` and the compiled backward tables disagree."

        return gate.view(ele_ebase.shape).contiguous()

    def _bw_state(self, layer, node_mars, kwargs):
        """The forward's leftovers, checked against the call the backward is running under."""
        state = getattr(layer, "_bs_bw_state", None)
        if state is None:
            raise RuntimeError(
                "`BlockScaleSumParams` backward ran without a matching forward: the normalizer "
                "`log Z` is produced by the forward kernel and is not recomputed here."
            )
        if state["batch_size"] != node_mars.size(1):
            raise RuntimeError(
                f"`BlockScaleSumParams` backward at batch {node_mars.size(1)} but the last forward "
                f"ran at batch {state['batch_size']}; the cached `log Z` does not apply."
            )
        if not kwargs.get("logspace_flows", False):
            raise NotImplementedError(
                "`BlockScaleSumParams` requires `logspace_flows = True`; the gate enters the element "
                "flows as a shift of a log-space running maximum, which the linear-space kernels have "
                "no place for."
            )
        return state

    def pre_backward_layer(self, layer, ns_tensors, node_flows, element_flows, node_mars,
                           element_mars, params, **kwargs) -> None:
        """
        Put `node_mars` back into the form the standard kernels expect, and hand them the gated kernels.

        The flow through an edge is `f[n,b] * theta_b[n,c] * em[c,b] / N_b[n]` with `theta_b = phi
        theta / Z`, and the forward stored `node_mars = log N - log Z`. The two normalizers cancel:

        .. code-block:: text

            f * (phi theta / Z) * e^em / (N / Z) = f * phi * theta * e^(em - log N)

        so the kernels need `log N`, i.e. `node_mars + log Z` -- one elementwise add over this layer's
        rows, undone in :func:`post_backward_layer`. What is left, `phi`, is the only thing the forked
        kernels do that the standard ones do not.
        """
        if len(ns_tensors) == 0:
            return None

        # Refused BEFORE anything is perturbed, so a caller asking for something unimplemented does not
        # get a circuit whose `node_mars` was left shifted.
        if self.apply_z_correction:
            raise NotImplementedError(
                "`apply_z_correction = True` is not implemented: the parameter flows this writes are "
                "the first term only, `sum_b f phi theta e^em / N`, which is the expected count "
                "PyJuice's EM optimizers consume."
            )

        external_params = kwargs.get(_buffer_kwarg(), None)
        if external_params is None:
            raise RuntimeError("the external-parameter staging buffer was not supplied to the backward.")

        state = self._bw_state(layer, node_mars, kwargs)

        from .kernels.c import get_module, get_ele_bw_module, get_par_bw_module

        plain, ele_mod, par_mod = get_module(), get_ele_bw_module(), get_par_bw_module()
        if plain is None or ele_mod is None or par_mod is None:
            raise NotImplementedError(
                "the block-scale backward needs its CUDA extensions (the two CuTe/TMA fork kernels and "
                "the plain one holding the normalizer shift), and at least one is unavailable here. "
                "There is no Triton fallback."
            )

        batch_size, block_size = state["batch_size"], state["block_size"]
        ext_base, gate_cbs, node_cbs = state["ext_base"], state["gate_cbs"], state["node_cbs"]

        for nids, log_z, rows in state["shift_args"]:
            plain.lowrank_shift_logz(node_mars, nids, log_z, block_size, 1.0)

        cache = getattr(layer, "_bs_bw_gate_cache", None)
        if cache is None:
            cache = layer._bs_bw_gate_cache = dict()

        def _ele_hook(element_flows, element_mars, node_flows, node_mars, params, chids, ele_ebase,
                      ele_pbase, batch, blk_size, cs_block_size, knt, partition_id):
            key = (partition_id, id(chids), int(knt))
            gate_bw = cache.get(key)
            if gate_bw is None:
                gate_bw = self._gate_bw_table(layer, chids, ele_ebase)
                cache[key] = gate_bw

            ele_mod.blockscale_ele_backward(
                element_flows, element_mars, node_flows, node_mars, params, external_params,
                chids, ele_ebase, ele_pbase, gate_bw,
                batch, blk_size, cs_block_size, knt, gate_cbs, ext_base)

        def _par_hook(param_flows, node_flows, node_mars, element_mars, params, nbase, cbase, pbase,
                      fbase, batch, blk_size, num_edges, partition_id):
            # The parameter flows are indexed exactly as the forward is -- one row per node block, one
            # column per incident edge block -- so the forward's own table serves them unchanged.
            par_mod.blockscale_par_backward(
                param_flows, node_flows, node_mars, element_mars, params, external_params,
                nbase, cbase, pbase, fbase, layer.ext_slots[0][partition_id],
                batch, blk_size, num_edges, node_cbs, gate_cbs, ext_base, 0)

        layer._ext_bw_ele_hook = _ele_hook
        layer._ext_bw_par_hook = _par_hook

        return None

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                            node_mars, element_mars, params, param_flows = None, **kwargs) -> None:
        """Undo the normalizer shift and take the kernels back off the standard backward."""
        if len(ns_tensors) == 0:
            return None

        layer._ext_bw_ele_hook = None
        layer._ext_bw_par_hook = None

        from .kernels.c import get_module

        state = layer._bs_bw_state
        for nids, log_z, rows in state["shift_args"]:
            get_module().lowrank_shift_logz(node_mars, nids, log_z, state["block_size"], -1.0)

        return None
