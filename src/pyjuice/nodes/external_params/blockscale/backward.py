"""
The backward of :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams` (:class:`_BackwardMixin`, one
of its bases): before the standard sum backward, `node_mars` put back into the form its kernels expect and the gated
kernels handed to them (the element and parameter flows, through the gate table of every child block); after it, the
normalizer shift undone and the gradient of the log gates' `log Z` term computed (its tiles chosen by measurement).
"""

from __future__ import annotations

import os

import torch

from .common import _buffer_kwarg

#: The log-Z half of `d LL / d log phi` contracts at fp32 precision; PYJUICE_BLOCKSCALE_LOGZ_TF32=1 takes TF32 dots
#: instead (the denominator flow's switch is PYJUICE_BLOCKSCALE_DENOM_TF32).
LOGZ_DOT_IEEE = 0 if os.environ.get("PYJUICE_BLOCKSCALE_LOGZ_TF32", "0") == "1" else 1

# Packs (parent node id, child element id) into one int64 key for the backward gate lookup. Node and
# element ids are global row indices into `node_mars` / `element_mars`, so this holds for any circuit
# that fits in memory; asserted where the keys are built.
_KEY_STRIDE = 1 << 32


class _BackwardMixin:
    """The backward of :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams`, one of its bases."""

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

        keys, vals, ch_all, cks = [], [], [], []
        for ns_info in layer.external_node_infos:
            ns = ns_info.ns
            par = ns_info.par_nids.to(dev).to(torch.long)
            ch = ns_info.ch_eids.to(dev).to(torch.long)
            off = ns.external_params.storage_offsets(ns)[0].to(dev).to(torch.long)

            keys.append(par * _KEY_STRIDE + ch)
            vals.append(off + layer.ext_unit_bases[ns_info.ns_idx][0])
            ch_all.append(ch)
            # One grid ROW per node gate, so a node-axis gate can offset by it below. Per `ns`, since
            # two sharing a layer may have different child counts.
            cks.append(torch.full_like(off, ns.num_ch_nodes // ns.external_params.gate_sizes(ns)[1]))

        key = torch.cat(keys)
        val = torch.cat(vals)
        ck = torch.cat(cks)
        assert int(torch.cat(ch_all).max()) < _KEY_STRIDE, \
            "circuit too large for the packed (parent, child) backward gate key."
        order = torch.argsort(key)
        key, val, ck = key[order], val[order], ck[order]

        # The parent BLOCK a tile belongs to: tiles are `TILE_SIZE_K`-wide slices of a node block, so
        # the block is the greatest node-block start not exceeding the tile's first node. Found by
        # search rather than by arithmetic on `block_size`, which would assume every `ns` in the layer
        # starts on the same grid.
        nid_all = torch.cat([layer.partitioned_nids[p].to(dev).to(torch.long)
                             for p in range(layer.num_fw_partitions)])
        nid_sorted = torch.sort(nid_all)[0]
        flat = ele_ebase.reshape(-1).to(torch.long)
        blk = nid_sorted[(torch.searchsorted(nid_sorted, flat, right = True) - 1).clamp(min = 0)]

        # `ele_ebase` may carry a trailing GROUP axis ([n_eleblocks, K_NUM_TILES, ptr_inc_step])
        # when a k-tile spans several parent blocks, so the child ids are broadcast against
        # whatever its trailing shape is rather than assuming it is 2-D.
        q = blk * _KEY_STRIDE + chids.view((-1,) + (1,) * (ele_ebase.dim() - 1)) \
                                     .expand_as(ele_ebase).reshape(-1)

        pos = torch.searchsorted(key, q).clamp(max = key.numel() - 1)
        hit = key[pos] == q
        # The NODE-AXIS row. `blk` is the parent block's first node and `flat` the tile's first
        # parent, so their difference in units of `gate_bs` is which node gate this tile's parents sit
        # in. The kernel is unchanged: a tile carries ONE row, which is exact as long as it does not
        # SPAN two node gates -- enforced where the fork is chosen.
        # Unconditional: `nrow` is identically zero when there is one gate per node block, so this is
        # the same table there. A branch on `layer.block_size` was tried and is WRONG -- that is the
        # layer's block size, not the gated `ns`'s, so node-axis shapes took the coarse arm and lost
        # the row (flows off by 0.8 relative).
        # Gated on the GATED `ns`'s block size, not `layer.block_size` and not unconditionally --
        # both were tried and both are wrong. Unconditional perturbs PADDED coarse tables, because
        # `blk` comes from a searchsorted that does not land cleanly on a padded entry, so `flat - blk`
        # is not zero there and rows that were duplicated stop being so. Keying on `layer.block_size`
        # takes this arm for node-axis shapes and drops the row entirely (flows off by 0.8).
        ns0 = layer.external_node_infos[0].ns
        gbs0 = self.gate_sizes(ns0)[0]
        if ns0.block_size // gbs0 > 1:
            nrow = (flat - blk) // gbs0
            gate = torch.where(hit, val[pos] + nrow * ck[pos], torch.full_like(q, -1))
        else:
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
                f"`BlockScaleSumParams` backward at batch {node_mars.size(1)}, but the last GATED "
                f"forward ran at batch {state['batch_size']}, so its cached `log Z` does not apply.\n\n"
                f"A forward given no `sum_external_params` runs this layer as a plain sum layer and "
                f"leaves the previous staging -- and this `log Z` -- in place rather than replacing "
                f"them, so the usual cause is an intervening ungated forward at another batch size:\n"
                f"    pc(x_a, sum_external_params = ...)   # gated, batch {state['batch_size']}\n"
                f"    pc(x_b)                              # no gate -> state left stale\n"
                f"    pc.backward(x_b)                     # <- here, at batch {node_mars.size(1)}\n\n"
                f"Pass `sum_external_params` on every forward of this circuit, or re-run the gated "
                f"forward immediately before the backward that consumes it."
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
        #
        gate_bs_ = self.gate_sizes(ns_tensors[0][0].ns)[0]
        n_ngates_ = self.gate_counts(ns_tensors[0][0].ns)[0]

        # `apply_z_correction` accumulates the denominator flow `F-` (see `_accumulate_denom` /
        # `post_backward_layer`); the M-step then normalizes `theta * F+ / F-`. `F-` sums over a whole
        # node block, which a node-axis gate splits -- so, exactly like `d LL / d log phi`, it is served
        # only at one node gate per block. Refused here, before `node_mars` is perturbed.
        if self.apply_z_correction and n_ngates_ > 1:
            raise NotImplementedError(
                f"`apply_z_correction = True` (the default) is not implemented for a gate finer than "
                f"`ns.block_size` along the NODE axis ({n_ngates_} node gates per block). Build the node at "
                f"the gate's block size for a served equivalent -- the gate tensor has the same shape -- or "
                f"pass `apply_z_correction = False` for the plain M-step."
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

        # The FLOWS now serve a node-axis gate; `d LL / d log phi` does not. Its log-Z term sums
        # `sigma[n, g]` over a whole node block, which a node gate splits, so that kernel needs a
        # structure change rather than the row offset the flows take.
        if n_ngates_ > 1 and layer._bs_grad_ext is not None:
            raise NotImplementedError(
                f"`d LL / d log phi` is not implemented for a gate finer than `ns.block_size` along "
                f"the NODE axis ({n_ngates_} node gates per block). The forward and both flows ARE "
                f"served, so EM and the circuit's own parameters train; only the gradient with respect "
                f"to the gate is missing. Build the node at the gate's block size for a fully served "
                f"equivalent -- the gate tensor has the same shape."
            )

        state = self._bw_state(layer, node_mars, kwargs)

        from ..kernels.c import get_ele_bw_module, get_par_bw_module, get_sb_bw_module
        from ..kernels.shift_logz import shift_logz

        # NO CUDA extension is required here. The CuTe and small-batch forks are accelerators, and
        # every kernel on this path also has a Triton fork: a gate table is always built, so
        # `"triton"` is always among the element candidates; `_par_triton_hook` is installed
        # unconditionally; and the normalizer shift -- the last piece that used to pin the whole
        # gated backward to nvcc -- is now `shift_logz`, which falls back to Triton on its own.
        # Every use of `ele_mod` / `sb_mod` / `par_mod` below checks for itself, so a missing one
        # costs speed, not correctness.
        ele_mod, par_mod = get_ele_bw_module(), get_par_bw_module()
        sb_mod = get_sb_bw_module()

        batch_size, block_size = state["batch_size"], state["block_size"]
        ext_base, gate_cbs, node_cbs = state["ext_base"], state["gate_cbs"], state["node_cbs"]

        for nids, log_z, rows in state["shift_args"]:
            shift_logz(node_mars, nids, log_z, block_size, 1.0)

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
            key = ("eleplan", ctx["signature"], ctx["batch_size"], ctx["partition_id"], ctx["params"].dtype)
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

            # ONE GATE ROW PER (k-tile, PARENT GROUP). `ele_ebase` from the layer is the group-0
            # slice, which is all the CuTe and small-batch forks can use; the Triton fork handles
            # several groups per tile, so its table is built from the FULL reconstruction. That is
            # the same telescoping cumsum `SumLayer` does, minus the `[:, :, 0]` -- `parids_increment`
            # is literally the first difference of `parids`, so this is exact for any `parids`.
            #
            # At `ptr_inc_step == 1` the result is `[n, K, 1]` and flattens to precisely the table
            # this used to build.
            pinc = ctx["ptr_inc_step"]
            _c = torch.cumsum(ctx["parids_increment"].to(torch.int64), 1)
            _sh = torch.zeros_like(_c)
            _sh[:, 1:] = _c[:, :-1]
            full_ebase = ctx["parids_start"][:, None, :].to(torch.int64) + _sh   # [n, K, pinc]
            plan["gate_tile"] = self._gate_bw_table(layer, chids, full_ebase) \
                                    .reshape(full_ebase.size(0), -1).contiguous()

            # The CuTe fork still reads ONE gate per k-tile, so it needs a tile to lie in a single
            # parent block. Only the Triton fork splits the contraction per group.
            # (`ele_loads_aligned`: its 8-parameter loads, aligned for the parameters' dtype -- always at fp32)
            if (ele_mod is not None and ctx["ele_cuda_ok"] and ctx.get("ele_loads_aligned", True) and tk == 64
                    and pinc == 1 and cbs % 128 == 0 and batch % 64 == 0 and plan["gate_tile"] is not None):
                kinds.append("cute")

            if (sb_mod is not None and batch < 16 and ne % blk == 0 and ctx["ele_cuda_ok"]
                    and pinc == 1):
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

            if n_ngates_ > 1:
                # The CuTe and small-batch element forks apply one gate per edge block to a whole node
                # block; only Triton takes the table's per-(k-tile, group) row. And a k-tile carries
                # ONE row, so it must not span two node gates.
                span = min(blk, tk)
                kinds = [k for k in kinds if k == "triton"] if span <= gate_bs_ else []
                if not kinds:
                    raise NotImplementedError(
                        f"a node-axis gate of {gate_bs_} needs each k-tile of parents inside one node "
                        f"gate, but this layer's tile spans {span} (block_size={blk}, "
                        f"TILE_SIZE_K={tk}). Build the node at the gate's block size instead."
                    )
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
                from ..kernels.blockscale_backward import _bs_triton_ele_kernel
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
            #
            # Asked of `_par_write_flags`, which CACHES per compiled tensor, rather than recomputing
            # the predicates here. Written the obvious way -- `bool((ctx["cids"] != 0).all())` inline
            # -- this ran a full device reduction AND forced a sync on every backward, and cost 16%
            # of the whole gated backward at batch 512. A correctness assert on a hot path has to be
            # free, or it stops being worth having.
            assert self._par_write_flags(layer, ctx["cids"], ctx["pfids"]) == (0, 0), \
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
                from ..kernels.blockscale_backward import _bs_triton_par_kernel
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
                        GATE_BS = gate_bs_, N_NODE_GATES = n_ngates_,
                        gate_nstride = (self._gate_nstride(layer, ctx["nids"])
                                        if n_ngates_ > 1 else None),
                        pid_m_offset = s0, num_stages = 1,
                        # Reached only from behind `par_ok` (sum_layer.py), which already requires
                        # `_par_flow_collision_free` AND edge-contiguous `cids` -- so neither hazard
                        # can apply here, and passing constants keeps this launch generating exactly
                        # the code it did before the flags existed. Asserted, not assumed: if that
                        # dispatch guard is ever loosened, this fires instead of losing updates.
                        PADDED = 0, PF_ATOMIC = 0)

            cands = ([("cute", _cute)] if (par_mod is not None and ctx["cute_ok"]) else []) \
                    + [("triton", _triton)]

            if n_ngates_ > 1:
                if ctx["TILE_SIZE_M"] > gate_bs_:
                    raise NotImplementedError(
                        f"a node-axis gate of {gate_bs_} needs the parameter kernel's node tile inside "
                        f"one node gate, but TILE_SIZE_M is {ctx['TILE_SIZE_M']}."
                    )
                cands = [c for c in cands if c[0] == "triton"]

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
            # The small-batch CUDA fork applies one gate per edge block to a whole node block.
            if n_ngates_ > 1:
                raise NotImplementedError(
                    f"a node-axis gate of {gate_bs_} is not served by the small-batch parameter fork. "
                    f"Build the node at the gate's block size instead."
                )
            # This fork is now offered whatever `pfids` look like -- the layer no longer decides on
            # the descriptor's behalf -- so it works out its own write mode, exactly as the Triton
            # fork does. `PADDED` masks padded lanes out (their contribution is zero, but their
            # read-add-store of `+0.0` would race a real one for pfid 0), and `PF_ATOMIC` covers slots
            # that still collide afterwards, which is what parameter tying produces.
            padded, pf_atomic = self._par_write_flags(layer, cids, pfids)
            sb_mod.blockscale_sb_par_backward(
                param_flows, node_flows, node_mars, element_mars, params, external_params,
                nids, cids, pids, pfids, layer.ext_slots[0][partition_id],
                batch, blk_size, num_edges, node_cbs, gate_cbs, ext_base,
                padded, pf_atomic, 0)

        def _par_triton_hook(ctx):
            """
            The gated param flows via Triton, for shapes no CUDA fork covers.

            This is the UNCONDITIONAL fallback, so unlike `_par_hook` and `_par_sb_hook` it cannot
            assume anything: it is reached exactly when `_par_flow_collision_free` failed and those
            two were skipped. `_par_write_flags` works out which hazard applies -- padded slots whose
            write must be masked, colliding written slots that need an atomic, or (the common case)
            neither.
            """
            from ..kernels.blockscale_backward import _bs_triton_par_kernel
            gb, grid = layer.ext_slots[0][ctx["partition_id"]], ctx["grid"]
            # The UNCONDITIONAL fallback, so it must check too: a shape `_par_hook` refuses arrives
            # here anyway, and would apply one gate row to a whole node block silently.
            if n_ngates_ > 1 and ctx["TILE_SIZE_M"] > gate_bs_:
                raise NotImplementedError(
                    f"a node-axis gate of {gate_bs_} needs the parameter kernel's node tile inside one "
                    f"node gate, but TILE_SIZE_M is {ctx['TILE_SIZE_M']}."
                )
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
                    GATE_BS = gate_bs_, N_NODE_GATES = n_ngates_,
                    gate_nstride = (self._gate_nstride(layer, ctx["nids"])
                                        if n_ngates_ > 1 else None),
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
                            node_mars, element_mars, params, param_flows = None,
                            denom_param_flows = None, **kwargs) -> None:
        """Undo the normalizer shift and take the kernels back off the standard backward."""
        if len(ns_tensors) == 0:
            return None

        layer._ext_bw_ele_hook = None
        layer._ext_bw_par_hook = None
        layer._ext_bw_par_sb_hook = None
        layer._ext_bw_par_triton_hook = None

        from ..kernels.shift_logz import shift_logz

        state = layer._bs_bw_state

        # The log-Z half of `d LL / d log phi`. Runs here, not in the element kernel, because it shares
        # no operand with it. Uses `log Z` straight from the forward's cache, so nothing is recomputed.
        grad_ext = getattr(layer, "_bs_grad_ext", None)
        if grad_ext is not None:
            import triton
            from ..kernels.blockscale_backward import _bs_triton_phigrad_logz_kernel

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

                def _go(gt, bb, tm, tgt, nids = nids, log_z = log_z, rows = rows, sigma = sigma,
                        gb = gb, n_gates = n_gates):
                    _bs_triton_phigrad_logz_kernel[
                            (triton.cdiv(n_gates, gt), rows * (bs_ // tm), triton.cdiv(batch, bb))](
                        node_flows = node_flows, log_z = log_z, sigma = sigma, ext = external_params,
                        gate = gb, grad_ext = tgt, nids = nids,
                        batch_size = batch, n_gates = n_gates,
                        N_CHILD_GATES = state["node_cbs"] // gate_cbs, BLOCK_SIZE_M = bs_,
                        BLOCK_B = bb, GATE_TILE = gt, TILE_SIZE_M = tm,
                        USE_DOT = (1 if (gt >= 16 and tm >= 16 and batch >= 64) else 0),
                        gate_stride = gb.size(1),
                        ext_base = state["ext_base"], num_stages = 1, DOT_IEEE = LOGZ_DOT_IEEE)

                gt, bb, tm = self._logz_tile(layer, _go, grad_ext, batch, bs_, n_gates, rows)
                _go(gt, bb, tm, grad_ext)

            layer._bs_grad_ext = None

        # The denominator flow `F-` for `apply_z_correction`. Present exactly when the PC allocated
        # `denom_param_flows` (i.e. a layer requested it): a model built with `apply_z_correction = False`
        # does nothing here. Uses the forward's cached `log Z`, so it is independent of the `node_mars`
        # shift about to be undone below.
        if denom_param_flows is not None:
            external_params = kwargs.get(_buffer_kwarg(), None)
            if external_params is None:
                raise RuntimeError("`apply_z_correction` needs the external-parameter staging buffer.")
            self._accumulate_denom(layer, node_flows, params, external_params, denom_param_flows)

        for nids, log_z, rows in state["shift_args"]:
            shift_logz(node_mars, nids, log_z, state["block_size"], -1.0)

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

        # PREFER A TILE THAT CAN USE `tl.dot`, on ACCURACY, wherever one exists.
        #
        # Below 16 the kernel loses `tl.dot` and falls back to a broadcast-sum over an [M, G, B]
        # intermediate, and that fallback is measurably less accurate: scored against the
        # reference-free zero-sum invariant, 3.99e-3 against 5.79e-4 on a 128-wide layer at gate 8,
        # and 4.70e-3 against 5.68e-4 at gate 4 -- 7x to 8x. `GATE_TILE` itself has NO effect on
        # accuracy (1, 8 and 16 all give 3.96e-3 with the fallback); it is purely whether `tl.dot` is
        # reachable, which needs an output tile of 16.
        #
        # The narrow tiles exist for OCCUPANCY -- a single-node-block layer at small batch leaves the
        # grid nearly empty with a wide gate tile -- so this used to look like a trade. MEASURED, it
        # is not one: forcing the dot-capable tile on exactly those shapes costs 1.001x and 0.999x
        # END TO END, because this kernel is a few microseconds of a step that is milliseconds. The
        # accuracy is worth having for a gate gradient that a router is trained on.
        #
        # Narrow tiles stay on the menu where `tl.dot` is unreachable anyway (`USE_DOT` also needs
        # `block_size >= 16` and `batch >= 64`), and where no wide tile fits the gate count -- so the
        # single-node-block case that motivated them is still served.
        if block_size >= 16 and batch >= 64:
            dot_capable = [c for c in cands if c[0] >= 16]
            if dot_capable:
                cands = dot_capable

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

        # NODE-TILE LADDER, widest first. `node_flows` and `log Z` are `[TILE_SIZE_M, bb]` per program,
        # so at a wide block they exhaust shared memory for EVERY `(GATE_TILE, BLOCK_B)` pair and the
        # search below has nothing to return -- measured at `block_size = 2048`, where the narrowest
        # candidate still wanted 139264 bytes against a 101376 limit and a 2048-state HMM therefore
        # could not run its gate gradient at all. Splitting the node axis is the only axis left.
        #
        # The full block is tried FIRST, so every shape that already fits keeps exactly the tile it
        # had; the ladder only descends when the whole `(gt, bb)` search comes back empty.
        tm_ladder = [block_size]
        t = block_size
        while t > 16:
            t //= 2
            tm_ladder.append(t)

        try:
            scr = torch.empty_like(grad_ext)
            ok = []
            for tm in tm_ladder:
                # Shared memory is what rules a candidate out, and it is not worth modelling: Triton's
                # own liveness decides which tiles are alive at once. Ask it, and drop what it refuses.
                for gt, bb in short:
                    try:
                        launch(gt, bb, tm, scr)
                        ok.append((gt, bb, tm))
                    except Exception:
                        pass
                # Every shortlisted candidate was refused. Widen to the whole set, narrowest first, and
                # take the first that runs -- an unmeasured but WORKING tile beats the alternative,
                # which was to return one that had just raised and let the real launch fail.
                if not ok:
                    for c in by_size:
                        if c in short:
                            continue
                        try:
                            launch(c[0], c[1], tm, scr)
                            ok.append((c[0], c[1], tm))
                            break
                        except Exception:
                            pass
                if ok:
                    break
            torch.cuda.synchronize()
            if len(ok) > 1:
                trials = [(c, (lambda c = c: launch(c[0], c[1], c[2], scr))) for c in ok]
                pick = ck.autotune(trials) or ok[0]
            elif ok:
                pick = ok[0]
            del scr
        except torch.cuda.OutOfMemoryError:
            pick = None

        if pick is None:
            # Nothing fit, or the scratch buffer could not be allocated. Hand back the narrowest pair
            # and let the real launch report the failure against the real operands.
            pick = (by_size[0][0], by_size[0][1], tm_ladder[-1]) if by_size else (1, 16, min(16, block_size))
        layer._bs_bw_gate_cache[key] = pick
        return pick
