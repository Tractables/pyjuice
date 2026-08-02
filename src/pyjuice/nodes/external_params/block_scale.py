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
                               dependence on `theta`. **Not implemented, and measured not to be worth
                               implementing** -- see below.
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
        # shared parameters' contribution to the normalizer was the constant 1. So the M-step pyjuice
        # performs -- normalize the flows per node -- solves a stationarity condition missing the term
        # `sum_b f_b * theta_b`, and is therefore not exactly EM under a live gate.
        #
        # MEASURED, not assumed. A corrected M-step (exact normalization within each gate, plus an MM
        # update on the gate masses) was prototyped and compared against the shipped one:
        #
        #   * it is REAL: +0.09 to +0.13 train LL, at every data-to-parameter ratio over a 100x sweep
        #     (256 to 32768 samples against 16384 parameters). So the dropped term is not negligible.
        #   * it does NOT generalize: held-out LL was worse in 6 of 6 runs, by 0.04 to 0.10. It is a
        #     better optimizer of the training objective, and what that extra fit buys is the training
        #     set's particular gate configuration -- gates are per-sample, so it does not transfer.
        #   * for scale: the gate itself is worth +0.5 to +1.25 nats of held-out LL over an ungated
        #     model, with the UNCORRECTED M-step. The correction is an order of magnitude smaller and
        #     points the wrong way.
        #
        # Full-batch EM was also verified monotone under live gates (no decrease in 12 steps at gate
        # scales 0, 1 and 3, with `pseudocount = 0` so that exact EM would be provably monotone).
        #
        # So the uncorrected M-step is the better default, not merely the convenient one, and this flag
        # raises rather than silently doing nothing. The prototype is ~20 lines of PyTorch and the
        # comparison is cheap to re-run if a setting arises where it might differ -- in particular a
        # multi-timestep HMM, where gates are correlated ACROSS timesteps within a sample, which the
        # single-layer test that produced these numbers does not exercise.
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

        # PARTIAL SUPPLY IS REFUSED, and the check has to come before `ext_base`. That base is measured
        # from the FIRST SUPPLIED node, but the gate tables it is added to already carry
        # `layer.ext_unit_bases[ns_idx]`, the cursor over ALL of the layer's gated nodes -- so the two
        # only compose when the first supplied node IS the layer's first. Supply gates for a later node
        # alone and every row shifts by the earlier nodes' slabs: the unsupplied node is gated with the
        # supplied one's `phi`, the supplied one reads past the end of the buffer (finite, plausible,
        # wrong), and the backward's gradient write faults. Refusing is the honest fix -- correcting the
        # base alone would not help, since the kernels still run over every row of the layer and would
        # read whatever a previous call left staged for the nodes nobody supplied.
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

            # PADDED SLOTS ARE EXCUSED from the contiguity requirements below, because the kernels
            # already compute exactly nothing for them.
            #
            # A row with fewer edge blocks than the partition's width is padded out with `cids == 0`
            # and `pids == 0` -- the dummy element and the dummy parameter -- and that is an EXACT
            # test, not a heuristic: a real child is an element `>= num_dummy_eles > 0` and a real
            # parameter is `>= num_dummy_params > 0`.
            #
            # Requiring contiguity over such a slot is what refused every ragged topology. It is not
            # something the kernels need. `ext_slots` carries `-1` for a padded EDGE BLOCK, the
            # forward reads that as `log phi = -inf`, and the fold happens BEFORE the max-stabilizer
            # is taken -- so a padded lane is `-inf` in the staged `element_mars` and contributes
            # exactly zero to `N`, while its normalizer operand `exp(log phi - mz)` is exactly zero
            # and so it contributes nothing to `Z` either. Neither depends on what the lane's DERIVED
            # addresses (`ebase + j`, `pbase + j * block_size`) happen to read, which is what lets the
            # base+stride addressing survive a padded -- even a part-padded -- tile. The gate table is
            # indexed per edge block, which is FINER than the 64-wide tile, so this covers a tile that
            # is part real and part padding, not merely a wholly padded one.
            #
            # What must still be refused is genuine BLOCK-SPARSITY: a row whose REAL children are not
            # a contiguous run cannot be addressed from one base at all. Hence the mask rather than
            # dropping the check.
            pad = (c2 == 0) & (p2 == 0)
            assert bool((((c2 == 0) == (p2 == 0))).all()), \
                "`cids == 0` and `pids == 0` disagree on which compiled slots are padding; the " \
                "padding convention this relaxation depends on has changed."

            # ---- the CuTe fork's layout: contiguous children and block_size-strided params PER TILE
            if valid and num_edges % 64 == 0:
                knt = num_edges // 64
                c3, p3 = c2.view(-1, knt, 64), p2.view(-1, knt, 64)
                pad3 = pad.view(-1, knt, 64)
                # A wholly padded tile's base is the dummy (0), which is what the kernel wants: its
                # derived lanes then land inside the dummy element / dummy parameter regions.
                ebase = c3[:, :, 0].contiguous()
                pbase = p3[:, :, 0].contiguous()
                ar = torch.arange(64, device = cids.device, dtype = torch.int64)

                if (bool(((c3 == ebase.unsqueeze(-1) + ar.view(1, 1, -1)) | pad3).all())
                        and bool(((p3 == pbase.unsqueeze(-1) + ar.view(1, 1, -1) * block_size)
                                  | pad3).all())):
                    # A PART-PADDED tile keeps its REAL base, so its padded lanes derive
                    # `pbase + j * block_size` past the row's own parameters -- and past the end of
                    # `params` itself on the last row of the last gated layer. The kernel CLAMPS that
                    # read (it takes `params.numel()`); see the note at its parameter-staging loop for
                    # why the value does not matter and why `memcheck` cannot see the unclamped
                    # version. Nothing to check here.
                    calls.append((nids, ebase, pbase, pids, gate, log_z,
                                  block_size, num_edges, node_cbs, gate_cbs, n_node_gates,
                                  ext_base, cfg))

            # ---- the small-batch kernel's layout: the same, but across the WHOLE row at once
            if sb_ok:
                eb_row = c2[:, 0].contiguous()
                pb_row = p2[:, 0].contiguous()
                if (bool(((c2 == eb_row.unsqueeze(-1) + ar_e.view(1, -1)) | pad).all())
                        and bool(((p2 == pb_row.unsqueeze(-1) + ar_e.view(1, -1) * block_size)
                                  | pad).all())
                        # Same bound as the CuTe fork, but this one walks the WHOLE row from a single
                        # base, so a padded row's derived reach is the row's full width rather than
                        # one tile's -- correspondingly further past the end.
                        and int(pb_row.max()) + num_edges * block_size <= params.numel()):
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

    def _grad_store_ok(self, gate_tile, ctx, gate_cbs) -> bool:
        """
        May the Triton element fork emit the gradient with a plain STORE rather than an atomic?

        Only if no two programs can ever write the same row. Two things can make them:

        * an element tile NARROWER than a gate. Then `(tile_id * TILE_SIZE_M) // GATE_CBS` is constant
          over the `GATE_CBS // TILE_SIZE_M` tiles inside one gate, and those are distinct `pid_m`
          programs each holding a partial sum over its own children.
        * two edge blocks sharing a storage row -- what `tie_external` does across every layer holding
          a copy of a tied node, and even within one layer when two copies land in it.

        The second is tested rather than inferred: a repeated value in the gate table IS the collision,
        whatever produced it, so a future aliasing route cannot quietly slip past a `tie_external`
        check. Only the rows actually emitted are compared -- the table repeats a gate across the TPB
        k-tiles of one parent block BY DESIGN, and those are accumulated in-register before the write.

        THAT GENERALITY IS WHAT MAKES RAGGED LAYERS SAFE, and it is worth spelling out because the
        third aliasing route arrived later and by a completely different road. On a padded layer
        `_gate_bw_table` hands a wholly-padded k-tile a REAL gate row -- its `ele_ebase` is the dummy
        node 0, and the `searchsorted(...).clamp(min = 0)` below maps that onto the FIRST node block,
        which is usually connected. Call it a phantom. A phantom is inert for the flows and for an
        ATOMIC gradient (the tile's parents are dummy nodes, so `log_n_fdm_max` is -inf, the tile
        drops, and `contrib` is exactly 0), but a STORE of that 0 would overwrite whatever the real
        owner of that row had accumulated. MEASURED with the store forced on such a shape: the
        reference-free zero-sum invariant goes from 3e-5 to 1.2e-1, and the error exceeds the
        gradient's own magnitude.

        No special case is needed, because a phantom IS a duplicated emitted row and duplicates are
        exactly what this tests. Verified rather than assumed: on the padded shapes the FIRST
        condition is False (the tile is not narrower than a gate, so it is not what declines the
        store) and the duplicate count equals the phantom count. It is structural, not luck -- padding
        is by whole parent blocks, so a padded block's every k-tile is a phantom including the one
        that gets emitted. If `_gate_bw_table` is ever changed to return `-1` for padded tiles the
        phantoms disappear, this returns True on ragged layers, and that is CORRECT -- there would
        then be nothing to overwrite.
        """
        if ctx["TILE_SIZE_M"] < gate_cbs or self.tie_external:
            return False

        blk, tk = ctx["block_size"], ctx["TILE_SIZE_K"]
        tpb = blk // tk if blk > tk else 1
        emitted = gate_tile[:, tpb - 1::tpb]
        pos = emitted[emitted >= 0]
        return int(torch.unique(pos).numel()) == int(pos.numel())

    def _par_write_flags(self, layer, cids, pfids):
        """
        `(PADDED, PF_ATOMIC)` for `_bs_triton_par_kernel`, cached per compiled partition.

        This kernel is the sum layer's UNCONDITIONAL param fallback, so it runs precisely where
        `_par_flow_collision_free` failed and the guarded CuTe / small-batch forks were skipped. It
        therefore has to decide for itself what made the write unsafe:

        * `PADDED` -- the partition has padded slots (`cids == 0`, exact: a real child is an element
          `>= num_dummy_eles > 0`). Their `pfids` are all 0, which `param_flows` gives to a REAL edge
          block, so a padded lane's `+0.0` read-add-store can discard a real update. Masked out.
        * `PF_ATOMIC` -- the slots that are actually WRITTEN still collide, which masking cannot fix.
          Parameter tying does this: two members of one tie group in the same layer share a
          `_param_flow_range` and so compile to identical `pfids` rows.

        Judged on the real slots alone, which is the point: a ragged untied layer collides only
        through its padding, so masking restores collision-freedom and the fast non-atomic
        read-add-store survives. Atomics are then paid only where they are genuinely required.

        Cached because `torch.unique` on every backward would be a per-call cost on the hot path;
        keyed by tensor identity, like `SumLayer._par_flow_collision_free` itself.
        """
        cache = getattr(layer, "_bs_bw_gate_cache", None)
        if cache is None:
            cache = layer._bs_bw_gate_cache = dict()

        key = ("parwrite", id(cids), id(pfids))
        flags = cache.get(key)
        if flags is None:
            real = cids != 0
            padded = not bool(real.all())
            written = pfids[real] if padded else pfids
            flags = (1 if padded else 0,
                     0 if layer._par_flow_collision_free(written.contiguous()) else 1)
            cache[key] = flags

        return flags

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
                "`apply_z_correction = True` is not implemented, and was measured not to be worth "
                "implementing: a prototype of the corrected M-step gains +0.09 to +0.13 TRAIN LL at "
                "every data-to-parameter ratio, but was WORSE on held-out LL in 6 of 6 runs (by 0.04 "
                "to 0.10) -- it fits the training set's per-sample gate configuration, which does not "
                "transfer. For scale, the gate itself is worth +0.5 to +1.25 nats held-out with the "
                "uncorrected M-step. See `BlockScaleSumParams.__init__` for the full measurement."
            )

        external_params = kwargs.get(_buffer_kwarg(), None)
        if external_params is None:
            raise RuntimeError("the external-parameter staging buffer was not supplied to the backward.")

        from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_GRAD_KWARG, \
            EXTERNAL_PARAMS_GRAD_BUFFER_KWARG
        grads = kwargs.get(EXTERNAL_PARAMS_GRAD_KWARG, None)
        want_grad = grads is not None and any(ns_info.ns in grads for ns_info, _ in ns_tensors)
        layer._bs_grad_ext = kwargs.get(EXTERNAL_PARAMS_GRAD_BUFFER_KWARG, None) if want_grad else None
        if want_grad and layer._bs_grad_ext is None:
            raise RuntimeError("external-parameter gradients were requested but no gradient buffer "
                               "was supplied to the backward.")

        state = self._bw_state(layer, node_mars, kwargs)

        from .kernels.c import get_module, get_ele_bw_module, get_par_bw_module, get_sb_bw_module

        plain, ele_mod, par_mod = get_module(), get_ele_bw_module(), get_par_bw_module()
        sb_mod = get_sb_bw_module()
        if plain is None or (ele_mod is None and sb_mod is None):
            raise NotImplementedError(
                "the block-scale backward needs its CUDA extensions -- the plain one holding the "
                "normalizer shift, plus at least one of the CuTe/TMA forks (large batch) and the "
                "small-batch forks -- and they are unavailable here. There is no Triton fallback."
            )

        batch_size, block_size = state["batch_size"], state["block_size"]
        ext_base, gate_cbs, node_cbs = state["ext_base"], state["gate_cbs"], state["node_cbs"]

        for nids, log_z, rows in state["shift_args"]:
            plain.lowrank_shift_logz(node_mars, nids, log_z, block_size, 1.0)

        cache = getattr(layer, "_bs_bw_gate_cache", None)
        if cache is None:
            cache = layer._bs_bw_gate_cache = dict()

        def _gate_for(kind, chids, ele_ebase, blk_size, num_edges, partition_id, knt):
            key = (kind, partition_id, id(chids), int(knt))
            g = cache.get(key)
            if g is None:
                if kind == "sb":
                    nblk = num_edges // blk_size
                    ele_ebase = (ele_ebase[:, :1]
                                 + torch.arange(nblk, device = ele_ebase.device).view(1, -1) * blk_size)
                g = self._gate_bw_table(layer, chids, ele_ebase)
                cache[key] = g
            return g

        def _ele_hook(ctx):
            """
            Run this layer's gated element flows with whichever of the forks is fastest here.

            Three may apply and which wins is not deducible: the CuTe fork is strongest at very large
            batch, but the ungated layer's own autotuner prefers Triton to its CuTe kernel at many
            shapes, and below batch 16 the plain-CUDA fork beats both. So they are MEASURED.

            The whole decision -- eligibility, gate tables, winner -- is cached per
            (signature, batch, partition) and only the winning launch is built per call. An earlier
            version rebuilt the candidate list every time; its two `torch.equal` eligibility checks
            forced a device sync per backward and cost more at small batch than the gate itself.
            """
            key = ("eleplan", ctx["signature"], ctx["batch_size"], ctx["partition_id"])
            plan = cache.get(key)
            if plan is None:
                plan = _build_ele_plan(ctx)
                cache[key] = plan
            _launch_ele(ctx, plan, ctx["element_flows"])

        def _build_ele_plan(ctx):
            import pyjuice.layer.kernels.c as ck

            chids, ebase, pbase = ctx["chids"], ctx["ele_ebase"], ctx["ele_pbase"]
            batch, blk, cbs = ctx["batch_size"], ctx["block_size"], ctx["cs_block_size"]
            ne, knt, tk = ctx["num_edges"], ctx["K_NUM_TILES"], ctx["TILE_SIZE_K"]
            plan = {"gate_tile": None, "gate_sb": None, "sb": None}
            kinds = []

            if ctx["ptr_inc_step"] == 1:
                plan["gate_tile"] = self._gate_bw_table(layer, chids, ebase)

            if (ele_mod is not None and ctx["ele_cuda_ok"] and tk == 64
                    and cbs % 128 == 0 and batch % 64 == 0 and plan["gate_tile"] is not None):
                kinds.append("cute")

            if sb_mod is not None and batch < 16 and ne % blk == 0 and ctx["ele_cuda_ok"]:
                art = torch.arange(knt, device = ebase.device, dtype = torch.int64).view(1, -1)
                if (torch.equal(ebase, ebase[:, :1] + art * tk)
                        and torch.equal(pbase, pbase[:, :1] + art * tk)):
                    nblk = ne // blk
                    starts = (ebase[:, :1]
                              + torch.arange(nblk, device = ebase.device).view(1, -1) * blk)
                    plan["gate_sb"] = self._gate_bw_table(layer, chids, starts)
                    plan["sb"] = (ebase[:, 0].contiguous(), pbase[:, 0].contiguous())
                    kinds += [("sb", c) for c in range(len(sb_mod.blockscale_sb_ele_configs()))]

            if plan["gate_tile"] is not None:
                kinds.append("triton")
                plan["grad_store_ok"] = self._grad_store_ok(plan["gate_tile"], ctx, gate_cbs)

            if not kinds:
                raise NotImplementedError(
                    f"no external element-flow backward applies here: cs_block_size={cbs}, "
                    f"batch={batch}, TILE_SIZE_K={tk}, ptr_inc_step={ctx['ptr_inc_step']}. Every fork "
                    f"needs each k-tile's parents to lie in ONE node block (ptr_inc_step == 1), which "
                    f"is what lets the gate factor out of the contraction."
                )

            if layer._bs_grad_ext is not None:
                # Gradients no longer force one fork: any candidate that implements `d LL / d log phi`
                # stays in, so the choice is still made on speed. A fork that does NOT implement it is
                # dropped rather than silently returning no gradient -- and if that empties the list,
                # the guard above has already reported the shape.
                #
                # This block sits AFTER that guard on purpose: an unsupported shape has no gate table,
                # and returning a kind for it reached the launch with a null table and died with an
                # AttributeError instead of a clear NotImplementedError.
                # Every fork implements the gradient; the filter stays as the place to drop one that
                # does not.
                kinds = [k for k in kinds if k in ("triton", "cute") or isinstance(k, tuple)]
                if not kinds:
                    raise NotImplementedError(
                        "external-parameter gradients are implemented for the Triton and small-batch "
                        "element forks; no gradient-capable fork applies to this shape."
                    )

            if len(kinds) == 1:
                plan["kind"] = kinds[0]
                return plan

            # Measured into a SCRATCH clone. Every candidate STORES its output so a live trial would
            # be harmless today, but the choice is cached forever and the scratch keeps that true.
            scr = layer._bk_ele_scratch
            if scr is None or scr.shape != ctx["element_flows"].shape:
                scr = layer._bk_ele_scratch = torch.empty_like(ctx["element_flows"])
            # The gradient is OFF for the trials. Its writes are atomic ACCUMULATIONS into the real
            # gradient buffer, so measuring N candidates over warmup+reps would add the gradient tens
            # of times over -- which is exactly what it did: the value came out ~128x too large, and
            # the zero-sum invariant blew up. The flow output already goes to a scratch buffer for the
            # same reason; the gradient needed the same treatment and did not have it.
            #
            # Choosing on flow cost alone is a small inaccuracy (it ignores the gradient's share of
            # each candidate) and is the conservative trade against corrupting the answer.
            saved_grad = layer._bs_grad_ext
            layer._bs_grad_ext = None
            try:
                trials = [(k, (lambda k = k: _launch_ele(ctx, plan, scr, k))) for k in kinds]
                plan["kind"] = ck.autotune(trials) or kinds[0]
            finally:
                layer._bs_grad_ext = saved_grad
            return plan

        def _launch_ele(ctx, plan, tgt, kind = None):
            kind = kind if kind is not None else plan["kind"]
            chids = ctx["chids"]
            batch, blk, cbs = ctx["batch_size"], ctx["block_size"], ctx["cs_block_size"]
            ne, knt = ctx["num_edges"], ctx["K_NUM_TILES"]

            if kind == "cute":
                ele_mod.blockscale_ele_backward(
                    tgt, ctx["element_mars"], ctx["node_flows"], ctx["node_mars"], ctx["params"],
                    external_params, chids, ctx["ele_ebase"], ctx["ele_pbase"], plan["gate_tile"],
                    (layer._bs_grad_ext if layer._bs_grad_ext is not None
                     else external_params.new_empty(0)),
                    batch, blk, cbs, knt, gate_cbs, ext_base)

            elif isinstance(kind, tuple):                     # ("sb", cfg)
                e0, p0 = plan["sb"]
                sb_mod.blockscale_sb_ele_backward(
                    tgt, ctx["element_mars"], ctx["node_flows"], ctx["node_mars"], ctx["params"],
                    external_params, chids, e0, p0, plan["gate_sb"],
                    (layer._bs_grad_ext if layer._bs_grad_ext is not None
                     else external_params.new_empty(0)),
                    batch, blk, cbs, ne, gate_cbs, ext_base, kind[1])

            else:                                             # "triton"
                from .kernels.blockscale_backward import _bs_triton_ele_kernel
                gb, grid = plan["gate_tile"], ctx["grid"]
                for s0 in range(0, grid[1], 32768):
                    cg = (grid[0], min(s0 + 32768, grid[1]) - s0)
                    _bs_triton_ele_kernel[cg](
                        node_flows = ctx["node_flows"], element_flows = tgt,
                        node_mars = ctx["node_mars"], element_mars = ctx["element_mars"],
                        mparams = ctx["params"], ext = external_params, gate = gb, chids = chids,
                        parids_start = ctx["parids_start"], parids_increment = ctx["parids_increment"],
                        parpids_start = ctx["parpids_start"],
                        parpids_increment = ctx["parpids_increment"],
                        batch_size = batch, ptr_inc_step = ctx["ptr_inc_step"],
                        BLOCK_B = ctx["BLOCK_B"], TILE_SIZE_K = ctx["TILE_SIZE_K"],
                        K_NUM_TILES = knt, TILE_SIZE_M = ctx["TILE_SIZE_M"],
                        BLOCK_SIZE_M = cbs, BLOCK_SIZE_K = blk, TL_DOT = ctx["TL_DOT"],
                        GATE_CBS = gate_cbs, gate_stride = gb.size(1), ext_base = ext_base,
                        grad_ext = (layer._bs_grad_ext if layer._bs_grad_ext is not None
                                    else external_params),
                        WRITE_GRAD = (1 if layer._bs_grad_ext is not None else 0),
                        GRAD_ATOMIC = (0 if plan.get("grad_store_ok") else 1),
                        pid_m_offset = s0, num_stages = 1)

        def _par_hook(ctx):
            """
            The gated parameter flows, choosing between the CuTe fork and the Triton one by measuring.

            The forward's own table serves both -- one row per node block, one column per incident edge
            block -- so only the launch differs. Measured rather than assumed for the same reason as
            the element flows: the ungated layer's autotuner prefers Triton to its CuTe param kernel at
            some shapes, and forcing CuTe there cost 63 -> 123 us at batch 1024.
            """
            import pyjuice.layer.kernels.c as ck

            gb = layer.ext_slots[0][ctx["partition_id"]]
            batch, blk, ne = ctx["batch_size"], ctx["block_size"], ctx["num_edges"]

            # BOTH candidates below write `param_flows` non-atomically, and neither takes a padding
            # mask. That is sound only because the sum layer reaches this hook from behind `par_ok`,
            # which requires collision-free `pfids` AND edge-contiguous `cids`. Restated here as an
            # assertion rather than left as an assumption: the guard lives in another file, and if it
            # is ever loosened the failure mode is a silent lost update, not a crash.
            assert layer._par_flow_collision_free(ctx["pfids"]) \
                and bool((ctx["cids"] != 0).all()), \
                "`_par_hook`'s forks assume collision-free `pfids` and unpadded `cids`; the sum " \
                "layer's `par_ok` guard is supposed to have established both."

            def _cute(tgt):
                par_mod.blockscale_par_backward(
                    tgt, ctx["node_flows"], ctx["node_mars"], ctx["element_mars"], ctx["params"],
                    external_params, ctx["nbase"], ctx["cbase"], ctx["pbase"], ctx["fbase"], gb,
                    # trailing 0 = `use_atomic`: 0 read-add-store, 1 atomicAdd, 2 store-only. Safe at
                    # 0 only under the assertion above.
                    batch, blk, ne, node_cbs, gate_cbs, ext_base, 0)

            def _triton(tgt):
                from .kernels.blockscale_backward import _bs_triton_par_kernel
                grid = ctx["grid"]
                for s0 in range(0, grid[1], 32768):
                    cg = (grid[0], min(s0 + 32768, grid[1]) - s0)
                    _bs_triton_par_kernel[cg](
                        node_flows = ctx["node_flows"], node_mars = ctx["node_mars"],
                        element_mars = ctx["element_mars"], mparams = ctx["params"],
                        param_flows = tgt, ext = external_params, gate = gb,
                        nids = ctx["nids"], cids = ctx["cids"], pids = ctx["pids"],
                        pfids = ctx["pfids"], batch_size = batch, num_edges = ne,
                        TILE_SIZE_B = ctx["TILE_SIZE_B"], B_NUM_TILES = ctx["B_NUM_TILES"],
                        TILE_SIZE_K = ctx["TILE_SIZE_K"], TILE_SIZE_M = ctx["TILE_SIZE_M"],
                        BLOCK_SIZE_M = blk, TL_DOT = ctx["TL_DOT"], NODE_CBS = node_cbs,
                        GATE_CBS = gate_cbs, gate_stride = gb.size(1), ext_base = ext_base,
                        pid_m_offset = s0, num_stages = 1,
                        # Reached only from behind `par_ok` (sum_layer.py), which already requires
                        # `_par_flow_collision_free` AND edge-contiguous `cids` -- so neither hazard
                        # can apply here, and passing constants keeps this launch generating exactly
                        # the code it did before the flags existed. Asserted, not assumed: if that
                        # dispatch guard is ever loosened, this fires instead of losing updates.
                        PADDED = 0, PF_ATOMIC = 0)

            cands = ([("cute", _cute)] if (par_mod is not None and ctx["cute_ok"]) else []) \
                    + [("triton", _triton)]

            key = ("parplan", ctx["signature"], batch, ctx["partition_id"])
            pick = cache.get(key)
            if pick is None:
                if len(cands) == 1:
                    pick = cands[0][0]
                else:
                    # A SCRATCH clone: these kernels are read-accumulate-write, so trials on the live
                    # `param_flows` would each add their contribution again.
                    try:
                        scr = torch.empty_like(ctx["param_flows"])
                        # Each trial launches the kernel REPEATS times inside the timed region. These
                        # candidates differ by only 6-25%, and a single launch spends comparable time
                        # in Python and in the driver -- enough noise to mis-pick, which is exactly
                        # what happened at K=2048 (CuTe chosen where Triton is 6% faster). Safe to
                        # repeat because the target is a scratch buffer.
                        REPEATS = 5

                        def _rep(f):
                            def go():
                                for _ in range(REPEATS):
                                    f(scr)
                            return go

                        pick = ck.autotune([(n, _rep(f)) for n, f in cands]) or cands[0][0]
                        del scr
                    except torch.cuda.OutOfMemoryError:
                        pick = cands[0][0]
                cache[key] = pick

            dict(cands)[pick](ctx["param_flows"])

        def _par_sb_hook(param_flows, node_flows, node_mars, element_mars, params, nids, cids, pids,
                         pfids, batch, blk_size, num_edges, partition_id):
            # As in `_par_hook`: the small-batch fork's write is a plain read-add-store with no
            # padding mask, sound only because its dispatch conjunction already tested
            # `_par_flow_collision_free`. Asserted so a loosened guard fails loudly.
            assert layer._par_flow_collision_free(pfids) and bool((cids != 0).all()), \
                "`blockscale_sb_par_backward` assumes collision-free `pfids` and unpadded `cids`."
            sb_mod.blockscale_sb_par_backward(
                param_flows, node_flows, node_mars, element_mars, params, external_params,
                nids, cids, pids, pfids, layer.ext_slots[0][partition_id],
                batch, blk_size, num_edges, node_cbs, gate_cbs, ext_base, 0)

        def _par_triton_hook(ctx):
            """
            The gated param flows via Triton, for shapes no CUDA fork covers.

            This is the UNCONDITIONAL fallback, so unlike `_par_hook` and `_par_sb_hook` it cannot
            assume anything: it is reached exactly when `_par_flow_collision_free` failed and those
            two were skipped. `_par_write_flags` works out which hazard applies -- padded slots whose
            write must be masked, colliding written slots that need an atomic, or (the common case)
            neither.
            """
            from .kernels.blockscale_backward import _bs_triton_par_kernel
            gb, grid = layer.ext_slots[0][ctx["partition_id"]], ctx["grid"]
            padded, pf_atomic = self._par_write_flags(layer, ctx["cids"], ctx["pfids"])
            for s0 in range(0, grid[1], 32768):
                cg = (grid[0], min(s0 + 32768, grid[1]) - s0)
                _bs_triton_par_kernel[cg](
                    node_flows = ctx["node_flows"], node_mars = ctx["node_mars"],
                    element_mars = ctx["element_mars"], mparams = ctx["params"],
                    param_flows = ctx["param_flows"], ext = external_params, gate = gb,
                    nids = ctx["nids"], cids = ctx["cids"], pids = ctx["pids"], pfids = ctx["pfids"],
                    batch_size = ctx["batch_size"], num_edges = ctx["num_edges"],
                    TILE_SIZE_B = ctx["TILE_SIZE_B"], B_NUM_TILES = ctx["B_NUM_TILES"],
                    TILE_SIZE_K = ctx["TILE_SIZE_K"], TILE_SIZE_M = ctx["TILE_SIZE_M"],
                    BLOCK_SIZE_M = ctx["block_size"], TL_DOT = ctx["TL_DOT"],
                    NODE_CBS = node_cbs, GATE_CBS = gate_cbs, gate_stride = gb.size(1),
                    ext_base = ext_base, pid_m_offset = s0, num_stages = 1,
                    PADDED = padded, PF_ATOMIC = pf_atomic)

        layer._ext_bw_ele_hook = _ele_hook
        # Installed UNCONDITIONALLY, even without the CuTe param module: `_par_hook` falls back to its
        # own Triton candidate when `par_mod` is None. Making the hook itself conditional left a hole
        # -- the layer's CUDA param regime is gated on a DIFFERENT extension, so it can be available
        # while `par_mod` is not, and the backward would then run the ungated kernel and return
        # SHARED-parameter flows for a gated layer. Silently, with no error.
        layer._ext_bw_par_hook = _par_hook
        layer._ext_bw_par_sb_hook = _par_sb_hook if sb_mod is not None else None
        layer._ext_bw_par_triton_hook = _par_triton_hook

        return None

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                            node_mars, element_mars, params, param_flows = None, **kwargs) -> None:
        """Undo the normalizer shift and take the kernels back off the standard backward."""
        if len(ns_tensors) == 0:
            return None

        layer._ext_bw_ele_hook = None
        layer._ext_bw_par_hook = None
        layer._ext_bw_par_sb_hook = None
        layer._ext_bw_par_triton_hook = None

        from .kernels.c import get_module

        state = layer._bs_bw_state

        # The log-Z half of `d LL / d log phi`. Runs here, not in the element kernel, because it shares
        # no operand with it. Uses `log Z` straight from the forward's cache, so nothing is recomputed.
        grad_ext = getattr(layer, "_bs_grad_ext", None)
        if grad_ext is not None:
            import triton
            from .kernels.blockscale_backward import _bs_triton_phigrad_logz_kernel

            external_params = kwargs.get(_buffer_kwarg(), None)
            bs_, gate_cbs = state["block_size"], state["gate_cbs"]
            batch = state["batch_size"]

            for pid, (nids, log_z, rows) in enumerate(state["shift_args"]):
                sigma, n_gates = self._sigma(layer, params, pid, bs_, gate_cbs)
                gb = layer.ext_slots[0][pid]
                # WHY THE FIRST TERM STAYS FUSED. Fusing it costs the element kernel ~13 us -- far more
                # than the ~0.8 us of arithmetic it adds -- which looks like an occupancy penalty from
                # carrying its accumulator through that kernel's k loop. So a standalone kernel
                # computing BOTH terms was written and measured against it. It lost everywhere: at
                # K=1024 / block 128 / gate 8 it ran the gated backward at 1.38x the ungated one
                # against the fused path's 1.34x, and at batch 8 at 3.6x against ~1.2x, because the
                # separate kernel re-reads `node_flows`, `node_mars` and the parameters that the
                # element kernel already has in registers. Reading them twice costs more than the
                # occupancy does. Do not re-split this without re-measuring.
                #
                # `rows` is the y axis, which CUDA caps at 65535. The kernel reads `pid_nb` straight
                # from `program_id(1)` with no offset argument, so it cannot be chunked the way the
                # flow launches above are -- refuse rather than let the driver truncate the grid and
                # silently drop the gradient of every node block past the cap.
                assert rows <= 65535, \
                    f"partition {pid} has {rows} node blocks, past the 65535 CUDA grid-y limit for " \
                    f"the external-gradient kernel. Split the partition, or use a larger block size."

                def _go(gt, bb, tgt, nids = nids, log_z = log_z, rows = rows, sigma = sigma,
                        gb = gb, n_gates = n_gates):
                    _bs_triton_phigrad_logz_kernel[
                            (triton.cdiv(n_gates, gt), rows, triton.cdiv(batch, bb))](
                        node_flows = node_flows, log_z = log_z, sigma = sigma, ext = external_params,
                        gate = gb, grad_ext = tgt, nids = nids,
                        batch_size = batch, n_gates = n_gates,
                        N_CHILD_GATES = state["node_cbs"] // gate_cbs, BLOCK_SIZE_M = bs_,
                        BLOCK_B = bb, GATE_TILE = gt,
                        USE_DOT = (1 if (gt >= 16 and bs_ >= 16 and batch >= 64) else 0),
                        gate_stride = gb.size(1),
                        ext_base = state["ext_base"], num_stages = 1)

                gt, bb = self._logz_tile(layer, _go, grad_ext, batch, bs_, n_gates, rows)
                _go(gt, bb, grad_ext)

            layer._bs_grad_ext = None

        for nids, log_z, rows in state["shift_args"]:
            get_module().lowrank_shift_logz(node_mars, nids, log_z, state["block_size"], -1.0)

        return None

    def _logz_tile(self, layer, launch, grad_ext, batch, block_size, n_gates, rows):
        """
        `(GATE_TILE, BLOCK_B)` for the log-Z gradient kernel, chosen by MEASUREMENT and cached.

        The two axes trade against each other and neither dominates: a wider gate tile amortizes the
        [M, B] `node_flows` / `log Z` load over more gates, a wider batch tile amortizes the [M, G]
        `sigma` load over more samples, and their product sets how many programs there are. Picking
        them independently -- a gate tile sized to fill the grid, a batch tile sized to fit shared
        memory -- landed on (32, 128), and measuring the grid found (32, 16) and (64, 32) instead:
        5.4 us against 10.4 at K=1024, 11.1 against 22.1 at K=2048, 3.0 against 12.2 at gate width 32.
        Roughly 2x, and where the optimum sits moves with the shape, so it is measured rather than
        ruled.

        Below 16 the kernel loses `tl.dot` and falls back to a broadcast-sum with an [M, G, B]
        intermediate, which at a wide batch tile measured 200-1200 us -- two orders of magnitude off,
        and the reason a naive sweep must not simply take the smallest tile. Such tiles are therefore
        offered only against the narrowest batch tile. But `GATE_TILE = 1` has to stay on the menu: a
        single-node-block layer has one `rows` and, at small batch, one batch tile, so a 32-wide gate
        tile leaves 2 programs for 188 SMs. Dropping it cost 1.2x -> 2.4x there, which is what the
        [M, 1, B] intermediate buys back.

        Trials run into a SCRATCH buffer. The kernel accumulates into the real one with `atomic_add`,
        so timing candidates against it would add the log-Z term once per trial.
        """
        import triton
        import pyjuice.layer.kernels.c as ck

        key = ("logztile", batch, block_size, n_gates, rows)
        pick = layer._bs_bw_gate_cache.get(key)
        if pick is not None:
            return pick

        bcap = max(16, min(64, triton.next_power_of_2(batch)))
        cands = [(gt, bb)
                 for gt in (1, 8, 16, 32, 64) if gt <= max(1, triton.next_power_of_2(n_gates))
                 for bb in (16, 32, 64) if bb <= bcap
                 if gt >= 16 or bb == 16]

        # RANKED before measuring, and only the finalists are timed. Every candidate that gets timed
        # also gets COMPILED, and the constexprs differ per shape so nothing is reused between them --
        # timing all of them put 230s of compilation into the test suite alone. The model only has to
        # be good enough to keep the winner in the shortlist, which the measurement then confirms.
        def _est(c):
            gt, bb = c
            gtiles = (n_gates + gt - 1) // gt
            btiles = (batch + bb - 1) // bb
            progs = gtiles * rows * btiles
            # `node_flows` + `log Z` are [M, bb] per program and re-read once per gate tile; `sigma` is
            # [M, gt] per program and re-read once per batch tile. Under-occupancy is what the two
            # would otherwise be traded away for, so charge for it.
            traffic = gtiles * rows * (batch * block_size * 8 + btiles * block_size * gt * 4)
            starve = max(1.0, 376.0 / max(progs, 1))
            # Below 16 gates the kernel loses `tl.dot` for a broadcast-sum over an [M, gt, B]
            # intermediate, so its work grows with the tile rather than shrinking.
            return traffic * starve * (gt if gt < 16 else 1)

        # Ordered by FOOTPRINT for the fallback below: the narrowest tiles are the ones most likely to
        # fit when the wide ones do not.
        by_size = sorted(cands, key = lambda c: (c[1], c[0]))
        short = sorted(cands, key = _est)[:4]
        # The narrowest pair always stays on the shortlist. `_est` ranks by traffic, which favours WIDE
        # tiles, so at a large block size every shortlisted candidate can be one that does not fit in
        # shared memory -- measured at block_size 1024 / gate 4 / batch 1024, where all four needed
        # 136-264 KB against a 101 KB limit, leaving nothing to measure.
        if by_size and by_size[0] not in short:
            short.append(by_size[0])

        try:
            scr = torch.empty_like(grad_ext)
            # Shared memory is what rules a candidate out, and it is not worth modelling: Triton's own
            # liveness decides which tiles are alive at once. Ask it, and drop what it refuses.
            ok = []
            for gt, bb in short:
                try:
                    launch(gt, bb, scr)
                    ok.append((gt, bb))
                except Exception:
                    pass
            # Every shortlisted candidate was refused. Widen to the whole set, narrowest first, and
            # take the first that runs -- an unmeasured but WORKING tile beats the alternative, which
            # was to return one of the candidates that had just raised and let the real launch fail.
            if not ok:
                for c in by_size:
                    if c in short:
                        continue
                    try:
                        launch(c[0], c[1], scr)
                        ok.append(c)
                        break
                    except Exception:
                        pass
            torch.cuda.synchronize()
            if len(ok) > 1:
                trials = [(c, (lambda c = c: launch(c[0], c[1], scr))) for c in ok]
                pick = ck.autotune(trials) or ok[0]
            elif ok:
                pick = ok[0]
            del scr
        except torch.cuda.OutOfMemoryError:
            pick = None

        if pick is None:
            # Nothing fit, or the scratch buffer could not be allocated. Hand back the narrowest pair
            # and let the real launch report the failure against the real operands.
            pick = by_size[0] if by_size else (1, 16)
        layer._bs_bw_gate_cache[key] = pick
        return pick

    def _sigma(self, layer, params, pid, block_size, gate_cbs):
        """
        `sigma[node, flat gate] = sum_{c in that gate} theta[node, c]`, cached until `params` changes.

        Recomputing it every backward would be a full read of theta for a quantity that only moves when
        the parameters do -- i.e. once per EM step, not once per batch. `Tensor._version` bumps on the
        in-place parameter update, which is exactly the invalidation signal.
        """
        cache = getattr(layer, "_bs_sigma_cache", None)
        key = (pid, params._version, block_size, gate_cbs)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]

        pids = layer.partitioned_pids[pid].to(torch.int64)
        rows, num_edges = pids.shape
        ar = torch.arange(block_size, device = params.device, dtype = torch.int64)
        th = params[pids[:, :, None] + ar[None, None, :]]              # [rows, edges, block_size]
        n_gates = num_edges // gate_cbs
        sg = th.view(rows, n_gates, gate_cbs, block_size).sum(dim = 2)  # [rows, gates, block_size]
        sigma = sg.permute(0, 2, 1).reshape(rows * block_size, n_gates).contiguous()

        layer._bs_sigma_cache = (key, sigma, n_gates)
        return sigma, n_gates
