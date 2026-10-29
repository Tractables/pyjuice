"""
The backward of :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams` (:class:`_BackwardMixin`, one
of its bases): before the standard sum backward, `node_mars` put back into the form its kernels expect and the gated
kernels handed to them (the element and parameter flows, through the gate table of every child block); after it, the
normalizer shift undone and the gradient of the log gates' `log Z` term computed (its tiles chosen by measurement).
"""

from __future__ import annotations

import os

import torch

from pyjuice.utils.launch_plans import LaunchPlans
from .common import _buffer_kwarg

def _set_attrs(layer, **attrs) -> None:
    """Plain attributes of a layer -- functions, tensors, None: never a parameter, buffer or module -- written past
    `nn.Module.__setattr__`, whose registration checks cost ~1.5 us a write (a gated layer's backward makes fifteen)."""
    layer.__dict__.update(attrs)


def _grad_kwargs():
    """`(EXTERNAL_PARAMS_GRAD_KWARG, EXTERNAL_PARAMS_GRAD_BUFFER_KWARG)`, imported once (a module-level import would
    be circular: `external_sum_layer` imports the parameterizations)."""
    global _GRAD_KWARGS
    if _GRAD_KWARGS is None:
        from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_GRAD_KWARG, EXTERNAL_PARAMS_GRAD_BUFFER_KWARG
        _GRAD_KWARGS = (EXTERNAL_PARAMS_GRAD_KWARG, EXTERNAL_PARAMS_GRAD_BUFFER_KWARG)
    return _GRAD_KWARGS


_GRAD_KWARGS = None


_MODS = {}


def _mod(name: str):
    """The module `..kernels.<name>`, imported on first use and then looked up (an import statement a call costs
    ~0.5 us; read attributes off the module, so a test replacing one still takes effect)."""
    m = _MODS.get(name)
    if m is None:
        import importlib
        m = _MODS[name] = importlib.import_module(f"pyjuice.nodes.external_params.kernels.{name}")
    return m


def _deferred_par():
    """`pyjuice.layer.deferred_par`, imported on first use (`pyjuice.layer` imports `pyjuice.nodes`, which is importing
    this module: a top-level import would be circular)."""
    m = _MODS.get("deferred_par")
    if m is None:
        import importlib
        m = _MODS["deferred_par"] = importlib.import_module("pyjuice.layer.deferred_par")
    return m


def _shift_logz():
    """`kernels.shift_logz.shift_logz` (read off its module each call)."""
    return _mod("shift_logz").shift_logz


_NO_GRAD = {}


def _no_grad(ext):
    """A 0-element tensor of `ext`'s dtype on its device: the CUDA forks' gradient argument when no gradient is wanted
    (made once per device and dtype: an allocation a launch was ~1 us)."""
    key = (ext.device, ext.dtype)
    t = _NO_GRAD.get(key)
    if t is None:
        t = _NO_GRAD[key] = ext.new_empty(0)
    return t


def _host_extra(runs):
    """
    Each of a plan's candidates' host time a call beyond the cheapest's, in seconds: the best of a few bursts of calls
    issued back to back without a sync (the calls' own issue cost, their GPU work queued behind it).

    A trial times one call between two CUDA events: max(its host time, its GPU time), so where a candidate's GPU work
    outlasts its host time the event time does not see the host time -- which a host-bound backward pays in full, the
    GPU waiting on it. MEASURED (the split element backward's three launches against the pre-pass kernel's two): 7.1
    against 5.1 us a call, and a training step 1-2% slower where a near tie on the GPU went its way (a tied gated
    HMM-2048 x4, 15 layers, B 64).
    """
    import time

    host = dict()
    for k, run in runs.items():
        try:
            run()                                   # (a first call compiles)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(2):
                t0 = time.perf_counter()
                for _ in range(3):
                    run()
                best = min(best, (time.perf_counter() - t0) / 3)
                torch.cuda.synchronize()
            host[k] = best
        except Exception:
            pass
    h0 = min(host.values(), default = 0.0)
    return {k: (host[k] - h0 if k in host else 0.0) for k in runs}


def _charged(run, extra):
    """`run` behind a host wait of `extra` seconds -- between a trial's first event and the launch, so the trial's event
    time counts the host time the candidate costs beyond the cheapest's (`_host_extra`)."""
    if extra <= 0:
        return run
    import time

    def go():
        end = time.perf_counter() + extra
        while time.perf_counter() < end:
            pass
        run()

    return go


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

    @staticmethod
    def _eles_tables(gate_tile, ebase, kt, split, pmax):
        """
        The split element backward's tables (kernels/c/blockscale_ele_backward_stream.cu) for k-steps of `kt` parents
        and `split` ranges, or None where an element block would have more than `pmax` pieces.

        An element block's gated k-steps -- its k-tiles with a gate, `64 // kt` steps each -- are cut into `split`
        equal ranges (the GEMM's third grid axis), and a range again wherever the gate changes or the k-steps stop
        being consecutive: a PIECE, one stabilizer's and one gate's share of a range. Per element block: int32
        `pc_k0` / `pc_k1` / `pc_gate` [n, pmax] (a piece's k-steps and gate; padding 0, 0, -1) and `sp_lo` / `sp_hi`
        [n, split] (a range's pieces); element blocks of equal parents and pieces share the pre-pass's exponentials,
        a GROUP: `ugrp` [n] (an element block's group) and `grep` (one element block of each group). `npc` is the
        most pieces an element block has. The kernel takes them as one int32 tensor, `tables` (one argument instead of
        seven); the named entries are views of it.

        Every element block at once (numpy, on the host): positions are an element block's gated k-steps counted
        from 0, a piece's bounds the union of its runs' first positions and its ranges' bounds. MEASURED: the 14
        tables a plan may build at 10^4 element blocks in 0.3 s (a loop over element blocks took 4 s).
        """
        import numpy as np

        sub = 64 // kt
        g = gate_tile.cpu().numpy()
        n = g.shape[0]
        on = g >= 0
        # a run starts at a gated k-tile whose predecessor is ungated, absent or under another gate
        cont = np.zeros_like(on)
        cont[:, 1:] = on[:, :-1] & on[:, 1:] & (g[:, :-1] == g[:, 1:])
        start = on & ~cont
        nruns = start.sum(1)
        R = max(1, int(nruns.max()))
        if R > pmax:
            return None
        ns = on.sum(1) * sub                                         # gated k-steps
        vr = np.arange(R)[None, :] < nruns[:, None]
        st_tile = np.argsort(~start, axis = 1, kind = "stable")[:, :R]   # the runs' first k-tiles, in order
        st_pos = np.take_along_axis(np.cumsum(on, axis = 1) - 1, st_tile, 1) * sub
        st_pos = np.where(vr, st_pos, ns[:, None])                   # (no such run: at the end, bounding nothing)
        qb = np.arange(split + 1)[None, :] * ns[:, None] // split    # the ranges' bounds
        bd = np.sort(np.concatenate([st_pos, qb], axis = 1), axis = 1)
        a, b = bd[:, :-1], bd[:, 1:]
        real = a < b
        npc_eb = real.sum(1)
        P = max(1, int(npc_eb.max()))
        if P > pmax:
            return None
        o = np.argsort(~real, axis = 1, kind = "stable")[:, :P]     # the pieces, in order
        pa, pb = np.take_along_axis(a, o, 1), np.take_along_axis(b, o, 1)
        vp = np.arange(P)[None, :] < npc_eb[:, None]
        ri = np.clip((st_pos[:, None, :] <= pa[:, :, None]).sum(-1) - 1, 0, R - 1)   # a piece's run
        rt, rp = np.take_along_axis(st_tile, ri, 1), np.take_along_axis(st_pos, ri, 1)
        k0 = np.zeros((n, pmax), np.int32)
        k1 = np.zeros((n, pmax), np.int32)
        pg = np.full((n, pmax), -1, np.int32)
        k0[:, :P] = np.where(vp, rt * sub + (pa - rp), 0)
        k1[:, :P] = np.where(vp, rt * sub + (pb - rp), 0)
        pg[:, :P] = np.where(vp, np.take_along_axis(g, rt, 1), -1)
        first = np.where(vp, pa, np.iinfo(np.int64).max)
        lo = (first[:, None, :] < qb[:, :-1, None]).sum(-1).astype(np.int32)
        hi = (first[:, None, :] < qb[:, 1:, None]).sum(-1).astype(np.int32)
        keys = np.concatenate([ebase.cpu().numpy().astype(np.int64), k0, k1], axis = 1)
        _, grep, ugrp = np.unique(keys, axis = 0, return_index = True, return_inverse = True)

        parts = [k0, k1, pg, lo, hi, ugrp.reshape(-1).astype(np.int32), grep.astype(np.int32)]
        tables = torch.from_numpy(np.concatenate([x.reshape(-1) for x in parts])).to(gate_tile.device)
        ends = np.cumsum([x.size for x in parts]).tolist()
        views = [tables[a:b].view(x.shape) for a, b, x in zip([0] + ends[:-1], ends, parts)]
        return dict(zip(("pc_k0", "pc_k1", "pc_gate", "sp_lo", "sp_hi", "ugrp", "grep"), views), tables = tables,
                    ngrp = int(grep.size), npc = P)

    def _eles_code(self, eles_mod, ctx, gate_tile, scr, external_params, ext_base, gate_cbs, fit):
        """
        The split element backward's `config id * 64 + split` for this layer's shape and its tables, or (None, None)
        where no split keeps within the piece limit. The rule's choice -- the first fitting tile at the split whose
        grid fills its last wave best, at one block an SM -- or another (tile, split) where it measures faster
        (`autotune.pick`, keyed by shape: equal layers share it), timed into the scratch flows without the gradient.
        """
        from pyjuice.layer.kernels import autotune
        from .denominators import _sm_count

        batch, blk, cbs, knt = ctx["batch_size"], ctx["block_size"], ctx["cs_block_size"], ctx["K_NUM_TILES"]
        neb = ctx["chids"].numel()
        cfgs = [tuple(int(v) for v in c) for c in eles_mod.configs()]
        pmax = int(eles_mod.max_pieces())
        sms = _sm_count(scr.device)
        gated = int((gate_tile >= 0).sum(1).max())
        tables, cands = dict(), []
        for i in fit:
            bm, bn, kt = cfgs[i]
            tiles = (batch // bn) * neb * (cbs // bm)
            splits = [s for s in (1, 2, 3, 4, 5, 6, 8) if s <= gated * (64 // kt) and s * tiles <= 8 * sms]
            for s in splits:
                if (kt, s) not in tables:
                    tables[(kt, s)] = self._eles_tables(gate_tile, ctx["ele_ebase"], kt, s, pmax)
            ok = [s for s in splits if tables[(kt, s)] is not None]
            if ok:
                best = max(ok, key = lambda s: s * tiles / (-(-s * tiles // sms) * sms))
                cands += [i * 64 + s for s in [best] + [s for s in ok if s != best]]
        if not cands:
            return None, None
        table = lambda code: tables[(cfgs[code // 64][2], code % 64)]
        if len(cands) == 1:
            return cands[0], table(cands[0])
        none = external_params.new_empty(0)

        def bench(code):
            tb = table(code)
            for _ in range(3):
                eles_mod.blockscale_ele_backward_stream(
                    scr, ctx["element_mars"], ctx["node_flows"], ctx["node_mars"], ctx["params"], external_params,
                    ctx["chids"], ctx["ele_ebase"], ctx["ele_pbase"], none, batch, blk, cbs, knt, gate_cbs, ext_base,
                    0, code, tb["tables"], tb["ngrp"], tb["npc"])

        shape = tuple(sorted((k, t["npc"], t["ngrp"]) for k, t in tables.items() if t is not None))
        key = ("bs_ele_stream", blk, cbs, batch, gate_cbs, knt, neb, shape, tuple(cands))
        code = autotune.pick(key, cands, bench)
        return code, table(code)

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

    def backward_plan(self, layer, ns_tensors, ns_grad_tensors, node_mars, param_flows, denom_param_flows, kwargs):
        """
        This layer's backward as a launch plan (`ExternalSumParams.backward_plan`): a launcher for the `+log Z` shift
        (`pre_backward_layer`), one a partition of element flows (`_ele_hook`) and of parameter flows (`_par_hook`,
        `_par_triton_hook` or `_par_sb_hook`), and one for the post-backward (`post_backward_layer`). Keyed by the
        gradient wanted, the deferral flag and the fixed half of the state the forward left -- its `log Z` buffers
        are read off it at each call. Not planned where the full path raises (no forward state or one of another
        batch, linear-space flows, a buffer missing), nor with the denominator flow's torch reference.
        """
        state = layer.__dict__.get("_bs_bw_state")
        if (state is None or state["batch_size"] != node_mars.size(1) or not kwargs.get("logspace_flows", False)
                or kwargs.get(_buffer_kwarg()) is None):
            return None
        grads_kwarg, grad_buffer_kwarg = _grad_kwargs()
        grads = kwargs.get(grads_kwarg, None)
        want_grad = grads is not None and any(ns_info.ns in grads for ns_info, _ in ns_tensors)
        if want_grad and kwargs.get(grad_buffer_kwarg) is None:
            return None
        if denom_param_flows is not None and os.environ.get("PYJUICE_BLOCKSCALE_DENOM_REF", "0") == "1":
            return None
        key = ("blockscale", want_grad, bool(kwargs.get("defer_param_flows", False)), state["block_size"],
               state["ext_base"], state["gate_cbs"], state["node_cbs"])
        return key, 2 + layer.num_bk_partitions + (layer.num_fw_partitions if param_flows is not None else 0)

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

        # `apply_z_correction` accumulates the denominator flow `F-` (see `_post_fused` /
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

        EXTERNAL_PARAMS_GRAD_KWARG, EXTERNAL_PARAMS_GRAD_BUFFER_KWARG = _grad_kwargs()
        grads = kwargs.get(EXTERNAL_PARAMS_GRAD_KWARG, None)
        want_grad = grads is not None and any(ns_info.ns in grads for ns_info, _ in ns_tensors)
        _set_attrs(layer, _bs_grad_ext = kwargs.get(EXTERNAL_PARAMS_GRAD_BUFFER_KWARG, None) if want_grad else None)
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

        # (the getters read off the module each call -- a test switches the extensions off by replacing them --, not
        # imported per call: three `from ... import` statements were ~1.5 us a layer, every backward)
        shift_logz = _shift_logz()

        # NO CUDA extension is required here. The CuTe and small-batch forks are accelerators, and
        # every kernel on this path also has a Triton fork: a gate table is always built, so
        # `"triton"` is always among the element candidates; `_par_triton_hook` is installed
        # unconditionally; and the normalizer shift -- the last piece that used to pin the whole
        # gated backward to nvcc -- is now `shift_logz`, which falls back to Triton on its own.
        # Every use of `ele_mod` / `sb_mod` / `par_mod` below checks for itself, so a missing one
        # costs speed, not correctness.
        ele_mod, par_mod = _mod("c").get_ele_bw_module(), _mod("c").get_par_bw_module()
        sb_mod = _mod("c").get_sb_bw_module()
        elep_mod = _mod("c").get_ele_prepass_module()
        eles_mod = _mod("c").get_ele_stream_module() if params.dtype == torch.bfloat16 else None
        # A tied group's parameter flows may be left to the end of the pass (`deferred_par`), if the backward allows it
        import pyjuice.layer.kernels.c as _ck
        pd_mod = _ck.get_par_deferred_module()
        _set_attrs(layer, _defer_param_flows = bool(kwargs.get("defer_param_flows", False)))

        batch_size, block_size = state["batch_size"], state["block_size"]
        ext_base, gate_cbs, node_cbs = state["ext_base"], state["gate_cbs"], state["node_cbs"]

        for nids, log_z, rows in state["shift_args"]:
            shift_logz(node_mars, nids, log_z, block_size, 1.0)

        # Recording a plan of this backward (`backward_plan`): each site below adds its launcher -- the call's tensors,
        # the staging buffers among them (`bk` / `gbk` in the backward's kwargs), its arguments.
        planning = LaunchPlans.recording(self.BACKWARD_PLAN_TAG)
        want_grad = layer._bs_grad_ext is not None
        bk, gbk = _buffer_kwarg(), _grad_kwargs()[1]
        if planning:
            defer = layer._defer_param_flows

            def _pre(node_flows, element_flows, node_mars, element_mars, params, param_flows, denom_param_flows,
                     kwargs):
                _set_attrs(layer, _defer_param_flows = defer)
                shift = _shift_logz()
                for nids, log_z, rows in layer.__dict__["_bs_bw_state"]["shift_args"]:
                    shift(node_mars, nids, log_z, block_size, 1.0)

            LaunchPlans.add(_pre, self.BACKWARD_PLAN_TAG)

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

            Four may apply and which wins is not deducible: the CuTe fork is strongest at very large
            batch, but the ungated layer's own autotuner prefers Triton to its CuTe kernel at many
            shapes, below batch 16 the plain-CUDA fork beats both, and the pre-pass fork
            takes the shapes it serves from Triton by 2-4x where measured. So they are MEASURED.

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
            run = _ele_fn(ctx, plan, plan["kind"])
            if planning:
                def _ele(node_flows, element_flows, node_mars, element_mars, params, param_flows, denom_param_flows,
                         kwargs):
                    run(element_flows, node_flows, node_mars, element_mars, params, kwargs[bk],
                        kwargs[gbk] if want_grad else None)

                LaunchPlans.add(_ele, self.BACKWARD_PLAN_TAG)
            run(ctx["element_flows"], ctx["node_flows"], ctx["node_mars"], ctx["element_mars"], ctx["params"],
                external_params, layer._bs_grad_ext)

        def _scratch(ctx):
            scr = layer._bk_ele_scratch
            if scr is None or scr.shape != ctx["element_flows"].shape:
                scr = layer._bk_ele_scratch = torch.empty_like(ctx["element_flows"])
            return scr

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

            # The pre-pass fork (kernels/c/blockscale_ele_backward_prepass.cu) on the CuTe fork's tables, under the same
            # contract (a k-tile of 64 parents in one parent block, the 8-parameter loads aligned); its own scope --
            # batch % 64, a power-of-two gate width >= 4, the element block a multiple of a config's rows, the gates in
            # shared memory -- is `fitting_configs`'.
            if (elep_mod is not None and ctx["ele_cuda_ok"] and ctx.get("ele_loads_aligned", True) and tk == 64
                    and pinc == 1 and plan["gate_tile"] is not None):
                kinds += [("elep", int(c)) for c in elep_mod.fitting_configs(cbs, batch, gate_cbs, knt)]

            # The split element backward (kernels/c/blockscale_ele_backward_stream.cu), bf16 parameters, on the
            # pre-pass fork's tables and contract: an element block's gated k-steps cut into ranges and pieces, each
            # piece under one stabilizer and one gate, so each is a plain bf16 GEMM (the exponentials as hi + lo
            # halves), merged after -- not the fork's bits (more accurate). One candidate, its (tile, split) and
            # tables measured first (`_eles_code`); its own scope is `fitting_configs`' and the piece limit.
            if (eles_mod is not None and ctx["params"].dtype == torch.bfloat16 and ctx["ele_cuda_ok"]
                    and ctx.get("ele_loads_aligned", True) and tk == 64 and pinc == 1 and n_ngates_ == 1
                    and plan["gate_tile"] is not None and external_params.data_ptr() % 16 == 0):
                fit = [int(c) for c in eles_mod.fitting_configs(cbs, batch, gate_cbs)]
                if fit:
                    code, tables = self._eles_code(eles_mod, ctx, plan["gate_tile"], _scratch(ctx), external_params,
                                                   ext_base, gate_cbs, fit)
                    if code is not None:
                        plan["eles"] = tables
                        kinds.append(("eles", code))

            # The small-batch fork takes the per-tile tables themselves (a tile's first parent and first parameter),
            # so it needs only `ele_cuda_ok` (one parameter-block group per tile) and whole parent blocks for its
            # gate rows. (It used to require the tiles to chain across the whole row, which no layer of more than one
            # parent block satisfies -- it never ran on a gated HMM-4096.)
            if (sb_mod is not None and batch < 16 and ne % blk == 0 and ctx["ele_cuda_ok"]
                    and pinc == 1 and ne == knt * tk):
                nblk = ne // blk
                jj = torch.arange(nblk, device = ebase.device, dtype = torch.int64) * blk
                starts = ebase[:, jj // tk] + (jj % tk).view(1, -1)              # each parent block's first parent
                plan["gate_sb"] = self._gate_bw_table(layer, chids, starts)
                plan["sb"] = (ebase.contiguous(), pbase.contiguous())
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
            scr = _scratch(ctx)
            # The trials write the gradient (where the backward wants one) into a SCRATCH buffer. Its writes
            # are atomic ACCUMULATIONS, so into the real buffer N candidates over warmup + reps would add it
            # tens of times over -- which is exactly what an earlier version did: the value came out ~128x
            # too large and the zero-sum invariant blew up. The flow output goes to a scratch buffer for the
            # same reason. Trials without the gradient compared the flows' cost alone, and that was not a
            # small inaccuracy: MEASURED (HMM-4096 x4, gate width 64, B 128) the pre-pass fork 49 us without
            # it, 58 with it; the split backward 49 and 42.
            saved_grad = layer._bs_grad_ext
            layer._bs_grad_ext = None if saved_grad is None else torch.zeros_like(saved_grad)
            try:
                runs = {k: (lambda k = k: _launch_ele(ctx, plan, scr, k)) for k in kinds}
                # Each trial carries the host time its candidate costs beyond the cheapest's: a fork wins by saving
                # more GPU time than it adds host time (`_host_extra`), so a host-bound backward does not lose to it.
                extra = _host_extra(runs)
                trials = [(k, _charged(runs[k], extra[k])) for k in kinds]
                plan["kind"] = ck.autotune(trials) or kinds[0]
            finally:
                layer._bs_grad_ext = saved_grad
            return plan

        def _launch_ele(ctx, plan, tgt, kind = None):
            _ele_fn(ctx, plan, plan["kind"] if kind is None else kind)(
                tgt, ctx["node_flows"], ctx["node_mars"], ctx["element_mars"], ctx["params"], external_params,
                layer._bs_grad_ext)

        def _ele_fn(ctx, plan, kind):
            """`kind`'s launch of this partition's element flows as a function of the call's tensors -- `(target,
            node_flows, node_mars, element_mars, params, ext, grad)`, `grad` None without the gradient --, everything
            else fixed by `ctx` and `plan`: what a plan replays (`backward_plan`)."""
            chids = ctx["chids"]
            batch, blk, cbs = ctx["batch_size"], ctx["block_size"], ctx["cs_block_size"]
            ne, knt = ctx["num_edges"], ctx["K_NUM_TILES"]
            ebase, pbase = ctx["ele_ebase"], ctx["ele_pbase"]
            no_grad = _no_grad(external_params)

            if kind == "cute":
                gt = plan["gate_tile"]

                def run(tgt, node_flows, node_mars, element_mars, params, ext, grad):
                    ele_mod.blockscale_ele_backward(
                        tgt, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gt,
                        no_grad if grad is None else grad, batch, blk, cbs, knt, gate_cbs, ext_base)

            elif isinstance(kind, tuple) and kind[0] == "elep":
                gt, cfg, tri = plan["gate_tile"], kind[1], _ele_fn(ctx, plan, "triton")

                def run(tgt, node_flows, node_mars, element_mars, params, ext, grad):
                    if ext.data_ptr() % 16:                    # (its log-gate rows are 16-byte copies)
                        return tri(tgt, node_flows, node_mars, element_mars, params, ext, grad)
                    elep_mod.blockscale_ele_backward_prepass(
                        tgt, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gt,
                        no_grad if grad is None else grad, batch, blk, cbs, knt, gate_cbs, ext_base,
                        0 if grad is None else 1, cfg)

            elif isinstance(kind, tuple) and kind[0] == "eles":
                tb, code, tri = plan["eles"], kind[1], _ele_fn(ctx, plan, "triton")

                def run(tgt, node_flows, node_mars, element_mars, params, ext, grad):
                    if ext.data_ptr() % 16:                    # (its log-gate rows are 16-byte loads)
                        return tri(tgt, node_flows, node_mars, element_mars, params, ext, grad)
                    eles_mod.blockscale_ele_backward_stream(
                        tgt, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase,
                        no_grad if grad is None else grad, batch, blk, cbs, knt, gate_cbs, ext_base,
                        0 if grad is None else 1, code, tb["tables"], tb["ngrp"], tb["npc"])

            elif isinstance(kind, tuple):                     # ("sb", cfg)
                (e0, p0), gsb, cfg = plan["sb"], plan["gate_sb"], kind[1]

                tk_sb = ctx["TILE_SIZE_K"]
                # The gradient's per-child shares ([element block, child, parent block, sample]): the kernel stores them
                # and sums a gate's children in order, so equal backwards give equal gradients (its atomic add across
                # the children did not). One buffer a plan, made at its first launch with a gradient.
                gp_shape = chids.numel() * cbs * (ne // blk) * batch

                def run(tgt, node_flows, node_mars, element_mars, params, ext, grad):
                    if grad is None:
                        gp = no_grad
                    else:
                        gp = plan.get("sb_gpart")
                        if gp is None or gp.numel() < gp_shape or gp.device != grad.device:
                            gp = plan["sb_gpart"] = torch.empty(gp_shape, device = grad.device)
                    sb_mod.blockscale_sb_ele_backward(
                        tgt, element_mars, node_flows, node_mars, params, ext, chids, e0, p0, gsb,
                        no_grad if grad is None else grad, gp, batch, blk, cbs, ne, tk_sb, gate_cbs, ext_base, cfg)

            else:                                             # "triton"
                from ..kernels.blockscale_backward import _bs_triton_ele_kernel
                gb, grid = plan["gate_tile"], ctx["grid"]
                ps, pi, pps, ppi = ctx["parids_start"], ctx["parids_increment"], ctx["parpids_start"], \
                    ctx["parpids_increment"]
                step, bb, tk, tm, tl_dot = ctx["ptr_inc_step"], ctx["BLOCK_B"], ctx["TILE_SIZE_K"], \
                    ctx["TILE_SIZE_M"], ctx["TL_DOT"]
                grad_atomic = 0 if plan.get("grad_store_ok") else 1
                chunks = [((grid[0], min(s0 + 32768, grid[1]) - s0), s0) for s0 in range(0, grid[1], 32768)]

                def run(tgt, node_flows, node_mars, element_mars, params, ext, grad):
                    for cg, s0 in chunks:
                        _bs_triton_ele_kernel[cg](
                            node_flows = node_flows, element_flows = tgt,
                            node_mars = node_mars, element_mars = element_mars,
                            mparams = params, ext = ext, gate = gb, chids = chids,
                            parids_start = ps, parids_increment = pi,
                            parpids_start = pps,
                            parpids_increment = ppi,
                            batch_size = batch, ptr_inc_step = step,
                            BLOCK_B = bb, TILE_SIZE_K = tk,
                            K_NUM_TILES = knt, TILE_SIZE_M = tm,
                            BLOCK_SIZE_M = cbs, BLOCK_SIZE_K = blk, TL_DOT = tl_dot,
                            GATE_CBS = gate_cbs, gate_stride = gb.size(1), ext_base = ext_base,
                            grad_ext = ext if grad is None else grad,
                            WRITE_GRAD = 0 if grad is None else 1,
                            GRAD_ATOMIC = grad_atomic,
                            pid_m_offset = s0, num_stages = 1)

            return run

        def _par_hook(ctx):
            """
            The gated parameter flows, choosing between the CuTe fork and the Triton one by measuring.

            The forward's own table serves both -- one row per node block, one column per incident edge
            block -- so only the launch differs. Measured rather than assumed for the same reason as
            the element flows: the ungated layer's autotuner prefers Triton to its CuTe param kernel at
            some shapes, and forcing CuTe there cost 63 -> 123 us at batch 1024.
            """
            gb = layer.ext_slots[0][ctx["partition_id"]]

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

            run = _par_fn(ctx, gb)
            if planning:
                LaunchPlans.add(_par_launcher(run), self.BACKWARD_PLAN_TAG)
            run(ctx["param_flows"], ctx["node_flows"], ctx["node_mars"], ctx["element_mars"], ctx["params"],
                external_params)

        def _par_fn(ctx, gb):
            """`_par_hook`'s launch as a function of the call's tensors -- `(param_flows, node_flows, node_mars,
            element_mars, params, ext)` --, everything else fixed by `ctx`: a tied step's flows into its group's slots
            where they defer (`_deferrer`), else the CuTe or the Triton fork, picked by measuring once."""
            from ..kernels.blockscale_backward import _bs_triton_par_kernel

            batch, blk, ne = ctx["batch_size"], ctx["block_size"], ctx["num_edges"]
            nbase, cbase, pbase, fbase = ctx["nbase"], ctx["cbase"], ctx["pbase"], ctx["fbase"]
            nids, cids, pids, pfids = ctx["nids"], ctx["cids"], ctx["pids"], ctx["pfids"]
            tsb, bnt, tk, tm, tl_dot = ctx["TILE_SIZE_B"], ctx["B_NUM_TILES"], ctx["TILE_SIZE_K"], \
                ctx["TILE_SIZE_M"], ctx["TL_DOT"]

            if n_ngates_ > 1 and tm > gate_bs_:
                raise NotImplementedError(
                    f"a node-axis gate of {gate_bs_} needs the parameter kernel's node tile inside "
                    f"one node gate, but TILE_SIZE_M is {tm}."
                )
            nstride = self._gate_nstride(layer, nids) if n_ngates_ > 1 else None
            grid = ctx["grid"]
            chunks = [((grid[0], min(s0 + 32768, grid[1]) - s0), s0) for s0 in range(0, grid[1], 32768)]

            def _cute(tgt, node_flows, node_mars, element_mars, params, ext):
                par_mod.blockscale_par_backward(
                    tgt, node_flows, node_mars, element_mars, params,
                    ext, nbase, cbase, pbase, fbase, gb,
                    # trailing 0 = `use_atomic`: 0 read-add-store, 1 atomicAdd, 2 store-only. Safe at
                    # 0 only under the assertion above.
                    batch, blk, ne, node_cbs, gate_cbs, ext_base, 0)

            def _triton(tgt, node_flows, node_mars, element_mars, params, ext):
                for cg, s0 in chunks:
                    _bs_triton_par_kernel[cg](
                        node_flows = node_flows, node_mars = node_mars,
                        element_mars = element_mars, mparams = params,
                        param_flows = tgt, ext = ext, gate = gb,
                        nids = nids, cids = cids, pids = pids,
                        pfids = pfids, batch_size = batch, num_edges = ne,
                        TILE_SIZE_B = tsb, B_NUM_TILES = bnt,
                        TILE_SIZE_K = tk, TILE_SIZE_M = tm,
                        BLOCK_SIZE_M = blk, TL_DOT = tl_dot, NODE_CBS = node_cbs,
                        GATE_CBS = gate_cbs, gate_stride = gb.size(1), ext_base = ext_base,
                        GATE_BS = gate_bs_, N_NODE_GATES = n_ngates_,
                        gate_nstride = nstride,
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
                cands = [c for c in cands if c[0] == "triton"]
            fns = dict(cands)
            key = ("parplan", ctx["signature"], batch, ctx["partition_id"])
            defer = _deferrer(ctx, gb)

            def run(param_flows, node_flows, node_mars, element_mars, params, ext):
                # A tied step's flows into its group's slots, added with the group's after the pass (`deferred_par`)
                if defer is not None and defer(param_flows, node_flows, node_mars, element_mars, params, ext):
                    return
                pick = cache.get(key)
                if pick is None:
                    pick = cache[key] = _par_pick(cands, param_flows, node_flows, node_mars, element_mars, params,
                                                  ext)
                fns[pick](param_flows, node_flows, node_mars, element_mars, params, ext)

            return run

        def _par_pick(cands, param_flows, *args):
            """The fastest of `_par_fn`'s candidates (their names and launches) at this call's tensors."""
            import pyjuice.layer.kernels.c as ck

            if len(cands) == 1:
                return cands[0][0]
            # A SCRATCH clone: these kernels are read-accumulate-write, so trials on the live
            # `param_flows` would each add their contribution again.
            try:
                scr = torch.empty_like(param_flows)
                # Each trial launches the kernel REPEATS times inside the timed region. These
                # candidates differ by only 6-25%, and a single launch spends comparable time
                # in Python and in the driver -- enough noise to mis-pick, which is exactly
                # what happened at K=2048 (CuTe chosen where Triton is 6% faster). Safe to
                # repeat because the target is a scratch buffer.
                REPEATS = 5

                def _rep(f):
                    def go():
                        for _ in range(REPEATS):
                            f(scr, *args)
                    return go

                pick = ck.autotune([(n, _rep(f)) for n, f in cands]) or cands[0][0]
                del scr
            except torch.cuda.OutOfMemoryError:
                pick = cands[0][0]
            return pick

        def _deferrer(ctx, gb):
            """`defer(param_flows, node_flows, node_mars, element_mars, params, ext)`: whether `deferred_par.try_defer`
            took this call's flows of the partition into its tied group's slots -- or None where its shape never
            defers (`deferred_par.may_defer`)."""
            if not _deferred_par().may_defer(pd_mod, ctx, n_ngates_):
                return None
            static = {k: v for k, v in ctx.items()
                      if k not in ("node_flows", "element_flows", "node_mars", "element_mars", "params", "param_flows")}

            def defer(param_flows, node_flows, node_mars, element_mars, params, ext):
                c = dict(static, param_flows = param_flows, node_flows = node_flows, node_mars = node_mars,
                         element_mars = element_mars, params = params)
                return _deferred_par().try_defer(pd_mod, layer, c, gb, ext, node_cbs, gate_cbs, ext_base, n_ngates_)

            return defer

        def _par_launcher(run):
            """A plan's launcher of a parameter-flow partition's `run(param_flows, node_flows, node_mars, element_mars,
            params, ext)`."""
            def _par(node_flows, element_flows, node_mars, element_mars, params, param_flows, denom_param_flows,
                     kwargs):
                run(param_flows, node_flows, node_mars, element_mars, params, kwargs[bk])

            return _par

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
            gt = layer.ext_slots[0][partition_id]

            def run(param_flows, node_flows, node_mars, element_mars, params, ext):
                sb_mod.blockscale_sb_par_backward(
                    param_flows, node_flows, node_mars, element_mars, params, ext,
                    nids, cids, pids, pfids, gt,
                    batch, blk_size, num_edges, node_cbs, gate_cbs, ext_base,
                    padded, pf_atomic, 0)

            if planning:
                LaunchPlans.add(_par_launcher(run), self.BACKWARD_PLAN_TAG)
            run(param_flows, node_flows, node_mars, element_mars, params, external_params)

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
            # A tied step deferred as from `_par_hook` -- only without either hazard: its flush writes every slot of
            # the block per node block, unmasked and non-atomically (two node blocks on one block would race)
            defer = _deferrer(ctx, gb) if (padded, pf_atomic) == (0, 0) else None
            nids, cids, pids, pfids = ctx["nids"], ctx["cids"], ctx["pids"], ctx["pfids"]
            batch, ne, blk = ctx["batch_size"], ctx["num_edges"], ctx["block_size"]
            tsb, bnt, tk, tm, tl_dot = ctx["TILE_SIZE_B"], ctx["B_NUM_TILES"], ctx["TILE_SIZE_K"], \
                ctx["TILE_SIZE_M"], ctx["TL_DOT"]
            nstride = self._gate_nstride(layer, nids) if n_ngates_ > 1 else None
            chunks = [((grid[0], min(s0 + 32768, grid[1]) - s0), s0) for s0 in range(0, grid[1], 32768)]

            def run(param_flows, node_flows, node_mars, element_mars, params, ext):
                if defer is not None and defer(param_flows, node_flows, node_mars, element_mars, params, ext):
                    return
                for cg, s0 in chunks:
                    _bs_triton_par_kernel[cg](
                        node_flows = node_flows, node_mars = node_mars,
                        element_mars = element_mars, mparams = params,
                        param_flows = param_flows, ext = ext, gate = gb,
                        nids = nids, cids = cids, pids = pids, pfids = pfids,
                        batch_size = batch, num_edges = ne,
                        TILE_SIZE_B = tsb, B_NUM_TILES = bnt,
                        TILE_SIZE_K = tk, TILE_SIZE_M = tm,
                        BLOCK_SIZE_M = blk, TL_DOT = tl_dot,
                        NODE_CBS = node_cbs, GATE_CBS = gate_cbs, gate_stride = gb.size(1),
                        GATE_BS = gate_bs_, N_NODE_GATES = n_ngates_,
                        gate_nstride = nstride,
                        ext_base = ext_base, pid_m_offset = s0, num_stages = 1,
                        PADDED = padded, PF_ATOMIC = pf_atomic)

            if planning:
                LaunchPlans.add(_par_launcher(run), self.BACKWARD_PLAN_TAG)
            run(ctx["param_flows"], ctx["node_flows"], ctx["node_mars"], ctx["element_mars"], ctx["params"],
                external_params)

        _set_attrs(layer, _ext_bw_ele_hook = _ele_hook)
        # Installed UNCONDITIONALLY, even without the CuTe param module: `_par_hook` falls back to its
        # own Triton candidate when `par_mod` is None. Making the hook itself conditional left a hole
        # -- the layer's CUDA param regime is gated on a DIFFERENT extension, so it can be available
        # while `par_mod` is not, and the backward would then run the ungated kernel and return
        # SHARED-parameter flows for a gated layer. Silently, with no error.
        _set_attrs(layer, _ext_bw_par_hook = _par_hook,
                   _ext_bw_par_sb_hook = _par_sb_hook if sb_mod is not None else None,
                   _ext_bw_par_triton_hook = _par_triton_hook)

        return None

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                            node_mars, element_mars, params, param_flows = None,
                            denom_param_flows = None, **kwargs) -> None:
        """Undo the normalizer shift and take the kernels back off the standard backward."""
        if len(ns_tensors) == 0:
            return None

        _set_attrs(layer, _ext_bw_ele_hook = None, _ext_bw_par_hook = None, _ext_bw_par_sb_hook = None,
                   _ext_bw_par_triton_hook = None)

        shift_logz = _shift_logz()

        state = layer._bs_bw_state
        grad_ext = getattr(layer, "_bs_grad_ext", None)
        external_params = kwargs.get(_buffer_kwarg(), None)
        if denom_param_flows is not None and external_params is None:
            raise RuntimeError("`apply_z_correction` needs the external-parameter staging buffer.")

        # The log-Z half of `d LL / d log phi` (with the gradient), the denominator flow `F-` of `apply_z_correction`
        # (with `denom_param_flows`, which the circuit allocates exactly when a layer asks for it) and the `-log Z`
        # shift in one launch per partition; the shift alone without either. The denominator flow's torch reference
        # (PYJUICE_BLOCKSCALE_DENOM_REF=1, what the kernel is validated against) runs on its own: it reads `node_flows`
        # and `log Z`, never `node_mars`.
        denom_ref = denom_param_flows is not None and os.environ.get("PYJUICE_BLOCKSCALE_DENOM_REF", "0") == "1"
        fused_w = denom_param_flows is not None and not denom_ref
        if grad_ext is not None or fused_w:
            self._post_fused(layer, state, node_flows, node_mars, params, external_params, grad_ext,
                             denom_param_flows if fused_w else None)
        else:
            for nids, log_z, rows in state["shift_args"]:
                shift_logz(node_mars, nids, log_z, state["block_size"], -1.0)
        if denom_ref:
            self._accumulate_denom_torch(layer, node_flows, params, external_params, denom_param_flows)

        # A plan's launcher of the above (`backward_plan`, which plans no call with the reference): the forward's
        # state read at each call, as the staging buffers
        if LaunchPlans.recording(self.BACKWARD_PLAN_TAG):
            want_grad, bk, gbk = grad_ext is not None, _buffer_kwarg(), _grad_kwargs()[1]

            def _post(node_flows, element_flows, node_mars, element_mars, params, param_flows, denom_param_flows,
                      kwargs):
                st = layer.__dict__["_bs_bw_state"]
                if want_grad or fused_w:
                    self._post_fused(layer, st, node_flows, node_mars, params, kwargs[bk],
                                     kwargs[gbk] if want_grad else None, denom_param_flows if fused_w else None)
                else:
                    shift = _shift_logz()
                    for nids, log_z, rows in st["shift_args"]:
                        shift(node_mars, nids, log_z, st["block_size"], -1.0)

            LaunchPlans.add(_post, self.BACKWARD_PLAN_TAG)
        _set_attrs(layer, _bs_grad_ext = None)

        return None

    def _post_fused(self, layer, state, node_flows, node_mars, params, external_params, grad_ext, denom_param_flows):
        """
        The log-Z gradient (with `grad_ext`), the denominator flow (with `denom_param_flows`) and the `-log Z` shift of
        each partition in one launch of `_bs_post_backward_kernel`, its plan built once per batch (`_post_plan`). Uses
        `log Z` straight from the forward's cache, so nothing is recomputed.

        WHY THE GRADIENT'S FIRST TERM STAYS IN THE ELEMENT KERNEL. Fusing it there costs that kernel ~13 us -- far more
        than the ~0.8 us of arithmetic it adds, an occupancy penalty from carrying its accumulator through the k loop --
        so a standalone kernel computing BOTH terms was written and measured against it. It lost everywhere: at K=1024 /
        block 128 / gate 8 the gated backward ran at 1.38x the ungated one against 1.34x, and at batch 8 at 3.6x against
        ~1.2x, re-reading the `node_flows`, `node_mars` and parameters the element kernel already holds in registers.
        The log-Z term shares no operand with that kernel, so it runs here. Do not re-split without re-measuring.
        """
        from ..kernels.blockscale_post_backward import _bs_post_backward_kernel

        bs_, gate_cbs, batch = state["block_size"], state["gate_cbs"], state["batch_size"]
        want_grad, want_w = grad_ext is not None, denom_param_flows is not None

        plans = layer.__dict__.setdefault("_bs_post_plans", {})
        for pid, (nids, log_z, rows) in enumerate(state["shift_args"]):
            if want_grad:
                sigma, n_gates = self._sigma(layer, params, pid, bs_, gate_cbs)
            else:
                sigma, n_gates = log_z, self.denom_gate_geometry(layer)[pid][2]     # (`sigma` unread)
            W = self._denom_slice(layer, denom_param_flows, pid) if want_w else log_z      # (unread without)
            gb = layer.ext_slots[0][pid]
            pkey = (batch, pid, bs_, gate_cbs, state["node_cbs"], state["ext_base"], want_grad, want_w)
            plan = plans.get(pkey)
            if plan is None:
                plan = plans[pkey] = self._post_plan(state, rows, n_gates, gb, nids, log_z, sigma, external_params,
                                                     grad_ext, W, want_grad, want_w)
            grid, consts, shares = plan
            if shares is not None:                 # the node tiles' gradient shares and their groups' tickets
                partial, counter = self._post_scratch(log_z.device, *shares)
            else:
                partial = counter = log_z          # (unread)
            _bs_post_backward_kernel[grid](
                node_flows = node_flows, node_mars = node_mars, log_z = log_z, sigma = sigma, ext = external_params,
                gate = gb, nids = nids, grad_ext = grad_ext if want_grad else log_z, W = W, partial = partial,
                counter = counter, **consts)

    def _post_scratch(self, device, n_shares, n_groups):
        """`(shares, counters)` for `_bs_post_backward_kernel`'s gradient sum: one pair per (device, size) for every
        layer -- they run one after another on the stream -- never reallocated, so a CUDA graph that captured one keeps
        a live address. The counters must read 0 before a launch, and every launch leaves them so; zeroed when made, or
        -- made inside a capture, whose zeroing runs only on replay -- by the first call outside one."""
        bufs = self.__dict__.setdefault("_bs_post_scratch", {})
        key = (device, n_shares, n_groups)
        entry = bufs.get(key)
        if entry is None:
            entry = bufs[key] = [torch.empty(n_shares, device = device),
                                 torch.zeros(n_groups, device = device, dtype = torch.int32),
                                 not torch.cuda.is_current_stream_capturing()]
        elif not entry[2] and not torch.cuda.is_current_stream_capturing():
            entry[1].zero_()
            entry[2] = True
        return entry[0], entry[1]

    def _post_plan(self, state, rows, n_gates, gb, nids, log_z, sigma, external_params, grad_ext, W, want_grad, want_w):
        """`(grid, constant arguments, shares)` of `_bs_post_backward_kernel` for one partition at this batch --
        `shares` the sizes `(gradient shares, groups)` of its scratch where a node block is several tiles, else None: a
        tile by rule, or one of a few around it where it measures faster (`autotune.pick`, keyed by shape: equal layers
        share it), timed into scratch buffers -- the gradient and `W` are accumulated, `node_mars` written -- read
        through scratch node rows (`nids` of their own)."""
        import triton
        from pyjuice.layer.kernels import autotune
        from ..kernels.blockscale_post_backward import _bs_post_backward_kernel
        from .denominators import _sm_count

        bs_, gate_cbs, batch = state["block_size"], state["gate_cbs"], state["batch_size"]
        nb_ = triton.next_power_of_2(batch)
        gmax = max(16, triton.next_power_of_2(n_gates))

        def fits(c):
            tm, tg, tb, bpp, _ = c
            return tm <= bs_ and bs_ % tm == 0 and tg <= gmax and tb <= max(16, nb_) and bpp * tb <= max(16, nb_)

        # The rule: 32 nodes x 32 gates x 32 samples, the batch tile and then the node tile halved while the grid is
        # under two programs an SM. MEASURED (graph-timed, tf32x3, against a sweep of 3^4 x 3 tiles): the best tile
        # or within 10% of it at HMM-4096 B 128 / 512, HMM-2048 B 64, HMM-1024 B 16 / 128 (gate width 64 and 8),
        # HMM-512 B 1024, HMM-256 B 8. Then the tiles around it, adopted where they measure faster.
        tg0 = min(gmax, 32)
        tm0, tb0 = min(32, bs_), 32

        def progs(tm, tb):
            return rows * (bs_ // tm) * triton.cdiv(n_gates, tg0) * triton.cdiv(batch, tb)

        if progs(tm0, tb0) < 2 * _sm_count(log_z.device):
            tb0 = 16
        if progs(tm0, tb0) < 2 * _sm_count(log_z.device):
            tm0 = min(16, bs_)
        # (a half node tile only while it keeps `tl.dot`: below 16 it is the broadcast-sum branch, never the faster)
        cands = [c for c in [(tm0, tg0, tb0, 1, 4), (tm0 * 2, tg0, tb0, 1, 4), (tm0 // 2, tg0, tb0, 1, 4),
                             (tm0, tg0, tb0, 1, 2)] if (c[0] >= 16 or c[0] == tm0) and fits(c)]
        cands = list(dict.fromkeys(cands)) or [(min(16, bs_), 16, 16, 1, 4)]

        def grid(c):
            tm, tg, tb, bpp, _ = c
            return (rows * (bs_ // tm), triton.cdiv(n_gates, tg), triton.cdiv(triton.cdiv(batch, tb), bpp))

        def consts(c):
            tm, tg, tb, bpp, warps = c
            return dict(batch_size = batch, n_gates = n_gates, N_CHILD_GATES = state["node_cbs"] // gate_cbs,
                        BLOCK_SIZE_M = bs_, gate_stride = gb.size(1), ext_base = state["ext_base"], TILE_M = tm,
                        TILE_G = tg, TILE_B = tb, B_TILES_PER_PROG = bpp, WANT_GRAD = int(want_grad),
                        WANT_W = int(want_w), W_ATOMIC = 1 if grid(c)[2] > 1 else 0,
                        USE_DOT = 1 if min(tm, tg, tb) >= 16 else 0,
                        GRAD_PARTIAL = 1 if (want_grad and bs_ // tm > 1) else 0, num_warps = warps, num_stages = 1)

        def shares(c):
            if not (want_grad and bs_ // c[0] > 1):
                return None
            g = grid(c)
            return (rows * (bs_ // c[0]) * n_gates * batch, rows * g[1] * g[2])

        key = ("bs_post", bs_, rows, n_gates, state["node_cbs"] // gate_cbs, gb.size(1), batch, want_grad, want_w,
               cands[0])
        cfg = autotune.cached(key)
        if cfg is None and autotune.should_tune(key, len(cands)):
            dev = log_z.device
            scr_rows = torch.zeros(rows * bs_, batch, device = dev)
            scr_nm = torch.zeros(rows * bs_, batch, device = dev)
            scr_nids = (torch.arange(rows, device = dev) * bs_).to(nids.dtype)
            scr_g = torch.zeros_like(grad_ext) if want_grad else log_z
            scr_w = torch.zeros_like(W) if want_w else log_z

            scr_s = {c: (torch.empty(shares(c)[0], device = dev),
                         torch.zeros(shares(c)[1], device = dev, dtype = torch.int32)) for c in cands
                     if shares(c) is not None}

            def bench(c):
                part, counter = scr_s.get(c, (log_z, log_z))
                _bs_post_backward_kernel[grid(c)](
                    node_flows = scr_rows, node_mars = scr_nm, log_z = log_z, sigma = sigma, ext = external_params,
                    gate = gb, nids = scr_nids, grad_ext = scr_g, W = scr_w, partial = part, counter = counter,
                    **consts(c))

            cfg = autotune.pick(key, cands, bench)
        cfg = cfg or cands[0]
        return grid(cfg), consts(cfg), shares(cfg)
