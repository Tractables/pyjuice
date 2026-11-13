"""
A tied group's parameter flows, deferred to one contraction (kernels/c/par_deferred_fp16.cu: fp16 operands with a
power-of-two scale per (row, slot); or its copy par_deferred_fp16_flush_ws.cu, the operand tiles loaded by a producer
warp: the same bits, the flush's tuner picks) -- a regular sum layer's, or a `BlockScale` layer's with its gates. A
step's capture takes kernels/c/par_deferred_fp16_capture.cu where its slots' layout allows (the same operands in two
chained launches), else that module's or par_deferred_fp16_fused.cu's -- and, where the group's GPU measured it faster,
a SIDE stream: what a capture with the product layer fused reads (its layer's flows and node values, its products'
children's node values) no later launch of the pass writes -- later layers write lower nodes' flows; a gated layer's
parameterization may change its own node values in place (BlockScale shifts them by log Z), and its backward
joins the side stream first (`join_side`, `ParameterizedSumLayer.backward`) -- so it runs beside the next layers'
chains, on the SMs their kernels leave idle, joined before the slots are flushed and as the backward's scope closes.

Tied copies of a sum node -- a homogeneous HMM's steps -- write their parameter flows into ONE flow block
(compilation gives up to `max_tied_ns_per_parflow_block` copies the same `_param_flow_range`). Step by step, each
copy's kernel read-adds-writes the whole block and reads the parameters again: at a 4096-state HMM that is 168 MB a
step, the step's kernel at the DRAM's rate. Deferred, a step only captures its two operands into its group's slots,
and after the backward's inner layers (`pyjuice.utils.deferred`) one contraction over (step, sample) adds the group's
flows once -- an accumulation like the per-step kernels', so flows already in the buffer (`flows_memory = 1`,
micro-batches) are kept.

A layer defers (`try_defer`) only when all of these hold, each checked, none assumed:
  * the backward allows it (`defer_param_flows`: no layer callback, which would see the flows between layers) and a
    scope is open (a layer's backward called on its own defers nothing);
  * the layer is tied, or its flow block already has a group this backward -- the group's operands and tables
    (`pfids`, `pids`) verified equal to the member's once (one device sync; never during a graph capture);
  * the step's kernel writes without either hazard (a regular layer's CUDA regime: contiguous tables, collision-free
    flows; a gated one's `_par_hook` guard), one gate per node block, `num_edges % 64 == 0` and node blocks in 64s --
    or in 32s where they make one operand group (below) --, and the group's step is enough work -- at least
    `MIN_FLOWS` flows, or `MIN_FLOW_SAMPLES` flows x samples (a smaller step's backward is host-bound, and there
    the captures' launches cost more than the per-step kernels they replace);
  * the slots fit their budget (`PAR_DEFER_MB`, at most `PAR_DEFER_SHARE` of the device's memory) with room for two
    steps (a full group flushes and goes on).
A group's slots belong to the scope that filled them: a new backward -- also after one that raised -- starts empty.
A slot is at most 512 of a step's columns (its rows' scales its own): a batch past 512 takes `cps` slots a step.

OPERAND GROUPS: a step's operands are, per node block, its nodes' A = exp(lf - M) and its children's Bop = exp(emars +
log phi + M), M the column maxima. Node blocks with the same children and no gates -- a regular HMM's, all of a layer's
-- take one M for the lot and one Bop (`GROUP_OPERANDS`; `par_deferred_fp16_capture.cu`'s groups): a slot then holds the
children's rows once, not once a node block (a tied HMM-8192 in blocks of 64: 140 MB a slot of 64 columns -> 2 MB), and
the flush is one product over the group's rows. MEASURED (a numerics emulation of the fp16 operands, tied HMM states
against float64, `scratchpad/nz/shared_bop_numerics.py`): the max relative error over flows >= 1e-6 of their block's max
1.1e-4 -> 1.15e-4 sharing one M over 128 node blocks, 7.4e-4 -> 7.9e-4 with peaked parameters (perturbation 16); no flow
lost. Node blocks of other children, or gated ones (their Bop holds their gates), or a step the chained capture cannot
take keep one operand a node block, as before (the same bits) -- and node blocks of 32 so do not defer (`may_defer`:
measured slower than the per-step kernels).
"""

from __future__ import annotations

import functools
import weakref

import torch

from pyjuice.utils.deferred import current_scope

#: The operand slots a group may hold (MB; device memory a deferring group keeps from its first backward on, up to 64
#: slots), and at most `PAR_DEFER_SHARE` of the device's memory (a smaller GPU's budget its share of it). A slot holds a
#: node block's children's operands for every node block -- a tied HMM's in blocks of 64 at 8192 states, 72 MB at 32
#: columns, 140 MB at 64 --, so the budget sets how often a long chain flushes. MEASURED (the backward alone, one GPU
#: no one else used, a tied HMM 32 x 8192, 31 steps, 512 -> 4096 MB): in blocks of 64, batch 64 16.80 -> 12.85 ms (3
#: slots -> 29: 11 flushes -> 2), batch 32 16.24 -> 14.93; in blocks of 128, batch 64 10.18 -> 7.91, batch 32 13.15 ->
#: 12.24; HMM-4096 in blocks of 64, batch 32 (18 MB slots: 2 flushes -> 1) 4.22 -> 4.07, batch 64 (35 MB: 3 -> 1) 3.16
#: -> 3.21.
#: Before (the training step, 128 -> 512 MB): HMM-8192 in 8 blocks, 7 steps, B 512 19.97 -> 13.00 ms (a 76 MB slot: at
#: 128 MB no room for two, the group did not defer); HMM-4096 in 4 blocks B 512 (a 21 MB slot) 7 steps 4.091 -> 3.989
#: (one flush, not two), 15 steps 8.092 -> 7.996.
PAR_DEFER_MB = 4096
PAR_DEFER_SHARE = 1 / 16

#: The smallest group (flows per step) that defers on its flows alone -- the per-step kernels' read-add-write of the
#: flow block, whatever the batch. The kernels win from ~1M flows (MEASURED with the TF32-operand
#: kernels the fp16 ones replaced in 4f535bfd -- the fp16 flush beats theirs 1.4-1.8x, not re-measured here --, gate
#: width 64, B 128, their time against the per-step Triton kernels': 262K flows (HMM-512 in 2 blocks, 15 steps)
#: 67 -> 81 us; 1M (HMM-1024, 7 steps) 74 -> 56, 31 steps 345 -> 219; 16.8M (HMM-4096 in 4 blocks, 7 steps)
#: 769 -> 364), but a host-bound backward hides the per-step kernels in the GPU's idle time and exposes the flush
#: after its last layer:
#: the whole tied backward, off -> on, HMM-4096 in 4 blocks 7 steps 1.87 -> 1.47 ms (B 512 5.01 -> 3.99, fp32
#: 2.11 -> 1.52) against HMM-2048 in 2 blocks (4M flows) 15 steps 2.26 -> 2.40 and HMM-1024 31 steps 4.64 -> 4.47.
MIN_FLOWS = 1 << 23

#: ... or on its flows x samples, the per-step kernels' contraction: a smaller group defers where the batch makes its
#: step's kernels the GPU's work. MEASURED (the fp16 kernels, the training step, deferring or not): HMM-2048 in 2 blocks
#: (4.2M flows), 15 steps, B 256 2.831 -> 2.32 ms, B 128 (2^29) 2.216 -> 2.07; in 4 blocks B 256 2.854 -> 2.41;
#: HMM-1024 (1M), 31 steps, B 512 (2^29) 4.18 -> 3.69 -- against, below the line, the host-bound HMM-2048 in 4 blocks
#: B 64 (2^28) 1.93 -> 2.04 and HMM-1024 B 128 (2^27) 3.42 -> 3.45: there a capture's 8 us of host (a memset, two
#: launches) is more than the per-step kernel it replaces.
MIN_FLOW_SAMPLES = 1 << 29

#: Backwards a group times in each place of its captures (the main stream, a side stream: the same bits) before it keeps
#: the side stream if it was at least `SIDE_MARGIN` faster -- a backward's time from its first capture to the NEXT
#: backward's (CUDA events: the whole step, the next forward's wait for a host still busy with this backward included --
#: timed only to its flush, the side stream was kept where it is 3% slower), the best of each place's, after `SIDE_WARM`
#: backwards untimed (the other tuners' first calls). MEASURED (the training step, bf16, the side stream against the main
#: one, both forced in one process): HMM-4096 in 4 blocks B128 -4.8%, B512 +1.7%; HMM-2048 in 2 blocks B128 -5.5% (+6.3%
#: with the fork in Python, ~12 us of host a capture) -- either way, so each group's GPU decides. Main stream without
#: autotuning (`autotune.ENABLED`) or in a capture.
SIDE_TRIALS, SIDE_WARM, SIDE_MARGIN = 3, 3, 0.01

#: Node blocks with the same children share one operand of them in a group's slots (see the module's docstring); False
#: keeps each node block its own (the layout before: for an A/B)
GROUP_OPERANDS = True

# device index -> [the side stream, whether a capture was left on it since the last join]
_SIDE = dict()

# id(cids table) -> (a weak reference to it, whether every node block's edges have the same children)
_SHARED = dict()


def _shared(cids, nb: int) -> bool:
    """Whether every node block of a partition's table has the same children (one device sync a table, cached; False
    where it is not known yet during a graph capture, which may not sync)."""
    entry = _SHARED.get(id(cids))
    if entry is None or entry[0]() is not cids:
        if torch.cuda.is_current_stream_capturing():
            return False
        c = _long(cids).contiguous().view(nb, -1)
        entry = _SHARED[id(cids)] = (weakref.ref(cids, lambda _r, k = id(cids): _SHARED.pop(k, None)),
                                     bool((c == c[:1]).all()))
    return entry[1]


def _slot_columns(batch: int):
    """`(cps, W)`: a step's slots and a slot's columns at this batch (a multiple of 32 up to 512)."""
    Bp = (batch + 31) // 32 * 32
    cps = (Bp + 511) // 512
    return cps, (Bp // 32 + cps - 1) // cps * 32


def _steps_fit(batch: int) -> bool:
    """Whether the chained capture (`par_deferred_fp16_capture.cu`: groups, node blocks in 32s) takes a step's slots."""
    import pyjuice.layer.kernels.c as cuda_kernels
    cmod = cuda_kernels.get_par_deferred_capture_module()
    return cmod is not None and bool(cmod.fits(_slot_columns(batch)[1], batch))


def operand_group(ctx, gated: bool = False) -> int:
    """The node blocks one operand group of this partition holds: all of them where they have the same children, the
    layer has no gates, their rows together are in 64s and the chained capture takes the step's slots; else 1."""
    blk, nb, batch = ctx["block_size"], ctx["nids"].numel(), ctx["batch_size"]
    if not GROUP_OPERANDS or gated or nb < 2 or (nb * blk) % 64 or ctx.get("cids") is None or not _steps_fit(batch):
        return 1
    return nb if _shared(ctx["cids"], nb) else 1


@functools.lru_cache(maxsize = None)
def _device_bytes(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).total_memory)


def slot_budget(device) -> int:
    """The bytes a group's slots may take on `device`: `PAR_DEFER_MB`, at most `PAR_DEFER_SHARE` of its memory."""
    return int(min(PAR_DEFER_MB * 2 ** 20, PAR_DEFER_SHARE * _device_bytes(torch.device(device).index)))


def join_side(device) -> None:
    """The current stream waits for every capture left on `device`'s side stream -- before anything writes what a
    capture reads (a gated layer's backward), before a group's slots are flushed, and as a backward's scope closes
    (raised or not: no later launch, nor a buffer freed, may overlap a capture of it)."""
    entry = _SIDE.get(torch.device(device).index)
    if entry is not None and entry[1]:
        torch.cuda.current_stream(device).wait_stream(entry[0])
        entry[1] = False


def _side_stream(device):
    entry = _SIDE.get(device.index)
    if entry is None:
        entry = _SIDE[device.index] = [torch.cuda.Stream(device = device), False]
    return entry

# id(param_flows buffer) -> (a weak reference to it, {group key: _Group}); by identity, not equality (a tensor's `==`
# is elementwise), the entry gone with the buffer
_GROUPS = dict()


def _groups_of(pf, create: bool):
    entry = _GROUPS.get(id(pf))
    if entry is None or entry[0]() is not pf:
        if not create:
            return {}
        entry = _GROUPS[id(pf)] = (weakref.ref(pf, lambda _r, k = id(pf): _GROUPS.pop(k, None)), dict())
    return entry[1]


class _Group:
    """The operand slots of one flow block at one batch size, and the tables every member shares."""

    def __init__(self, pfids, pids, nb, blk, ne, batch, pf, G = 1):
        self.cps, self.W = _slot_columns(batch)                # (slots a step, a slot's columns)
        self.G, ng = G, nb // G                                # (node blocks an operand group; the groups)
        per_slot = (nb * blk + ng * ne) * (self.W * 2 + 4)     # (fp16 values, a float scale a row)
        self.S = int(min(64, slot_budget(pf.device) // per_slot)) // self.cps * self.cps
        self.pfids, self.pids = pfids, pids
        self.bulk = _bulk_ok(pfids, pf)
        self.slots, self.scope, self.cfg = 0, None, None
        if self.S >= 2 * self.cps:
            dev = pf.device
            self.A = torch.zeros(nb, blk, self.S * self.W, device = dev, dtype = torch.half)
            self.Bop = torch.zeros(ng, ne, self.S * self.W, device = dev, dtype = torch.half)
            self.fA = torch.ones(nb, self.S, blk, device = dev)
            self.fB = torch.ones(ng, self.S, ne, device = dev)
            self.M = torch.empty(ng, self.cps * self.W, device = dev, dtype = torch.int32)
            # (`par_deferred_fp16_capture.cu`: a step's column maxima its own, zeroed as the slots fill from the first)
            self.Ms = torch.empty(self.S // self.cps, ng, self.cps * self.W, device = dev, dtype = torch.int32)
        self.steps_ok = None                                  # (whether that capture takes the group's slots)
        # where its captures run: `side` the chosen place (None until measured), `mode` this backward's, `trials` the
        # backwards timed [place, an event at its first capture], `warm` backwards left untimed
        self.side, self.mode, self.trials, self.warm = None, None, [], SIDE_WARM

    def flush(self, mod, pf, params):
        join_side(pf.device)
        if self.slots == 0:
            return
        if self.cfg is None:
            self.cfg = self._pick_cfg(mod, pf, params)
        if self.slots == 0:                                   # (nothing captured since: the trials above are no adds)
            return
        fmod, cfg = self.cfg
        fmod.flush(pf, self.A, self.Bop, self.fA, self.fB, self.pfids, self.pids, params, self.slots, cfg, self.bulk)
        self.slots = 0

    def on_side(self, eligible: bool) -> bool:
        """Whether this backward's captures run on the side stream: the place chosen, else the next one timed."""
        if self.mode is not None:
            return self.mode
        self.mode = False
        if not eligible:
            return False
        from pyjuice.layer.kernels import autotune
        if self.side is not None:
            self.mode = self.side
        elif autotune.ENABLED and not torch.cuda.is_current_stream_capturing():
            if self.warm > 0:
                self.warm -= 1
                return False
            t0 = torch.cuda.Event(enable_timing = True)
            t0.record()
            self.trials.append([None, t0])
            done = {False: [], True: []}                     # (a backward's place: its first capture to the next's)
            for (place, a), (_, b) in zip(self.trials, self.trials[1:]):
                if b.query():
                    done[place].append(a.elapsed_time(b))
            if min(len(done[False]), len(done[True])) >= SIDE_TRIALS:
                self.side = self.mode = min(done[True]) < (1 - SIDE_MARGIN) * min(done[False])
                self.trials = []
            else:
                self.mode = self.trials[-1][0] = len(self.trials) % 2 == 0     # (main, side, main, side, ...)
        return self.mode

    def _pick_cfg(self, mod, pf, params):
        """The flush's (module, config): `mod`'s configs and its producer-warp copy's (the same bits), measured once
        into a scratch copy of the flows (the flush read-adds-writes them)."""
        import pyjuice.layer.kernels.c as ck

        blk, ne = self.A.size(1), self.Bop.size(1)
        rows = self.G * blk                                    # (an operand group's rows)
        mods = [m for m in (mod, ck.get_par_deferred_flush_ws_module()) if m is not None]
        cands = [(m, i) for m in mods for i, (bm, bn, bk) in enumerate(m.flush_configs())
                 if rows % bm == 0 and (blk % bm == 0 if bm <= blk else bm % blk == 0 and blk & (blk - 1) == 0)
                 and ne % bn == 0 and self.W % bk == 0]
        if len(cands) == 1 or torch.cuda.is_current_stream_capturing():
            return cands[0]
        scr = pf.clone()
        trials = [(c, (lambda c = c: c[0].flush(scr, self.A, self.Bop, self.fA, self.fB, self.pfids, self.pids, params,
                                                 self.slots, c[1], self.bulk))) for c in cands]
        return ck.autotune(trials) or cands[0]


def _bulk_ok(pfids, pf) -> bool:
    """Whether the flush's adds can be the L2's bulk reductions: every edge's run of flows 16-byte aligned."""
    return bool((pfids % 4 == 0).all()) and pf.data_ptr() % 16 == 0


def _long(t):
    return t if t.dtype == torch.long else t.long()


def may_defer(mod, ctx, n_node_gates, gated: bool = False) -> bool:
    """Whether a partition of this shape (`ctx`: its tables and sizes) defers at all -- the conditions fixed by it,
    which `try_defer` checks first; the rest are a call's. Node blocks in 64s, or in 32s where they make one operand
    group (`operand_group`: the same children, no gates -- `gated` --, the chained capture taking the step's slots;
    the other captures' rows are in 64s). Node blocks of 32 each their own operand -- a gated layer's always are --
    were MEASURED slower deferred than through the per-step kernels (toast, one GPU no one else used, a tied HMM 32 x L,
    backward ms, deferred against per-step: BlockScale's, L 4096 at batch 16 10.36 / 6.08, at 64 13.25 / 9.60, L 8192
    30.03 / 27.56 and 43.42 / 39.45; a plain one forced to per-block operands, L 4096 at batch 16 5.68 / 5.17): a
    32-row node block is a thin slice of each flush tile."""
    blk, ne = ctx["block_size"], ctx["num_edges"]
    flows = ctx["nids"].numel() * blk * ne
    if not (mod is not None and n_node_gates == 1 and ne % 64 == 0
            and (flows >= MIN_FLOWS or flows * ctx["batch_size"] >= MIN_FLOW_SAMPLES)):
        return False
    return blk % 64 == 0 or (blk % 32 == 0 and operand_group(ctx, gated) > 1)


def try_defer(mod, layer, ctx, gate_table, ext, node_cbs, gate_cbs, ext_base, n_node_gates) -> bool:
    """Capture this step's parameter flows into its tied group's slots (the flush deferred to the end of the pass)
    and return True; or return False, the caller then launching the step's own kernel. `gate_table` / `ext`: a gated
    layer's gate table and log-gates (None for a regular layer: log phi 0)."""
    if not may_defer(mod, ctx, n_node_gates, gate_table is not None) or not getattr(layer, "_defer_param_flows", False):
        return False
    scope = current_scope()
    if scope is None:
        return False
    pf, params = ctx["param_flows"], ctx["params"]
    batch, blk, ne = ctx["batch_size"], ctx["block_size"], ctx["num_edges"]
    nb = ctx["nids"].numel()

    # This layer's group, resolved once per (partition, batch): None where it does not defer
    cache = layer.__dict__.setdefault("_bs_pd_member", {})
    mkey = (ctx["partition_id"], batch, id(ctx["pfids"]))
    member = cache.get(mkey)
    if member is None:
        if torch.cuda.is_current_stream_capturing():
            return False                                       # (the checks below sync)
        tied = any(ns.is_tied() for ns in layer.nodes)
        pfids, pids = _long(ctx["pfids"]).contiguous(), _long(ctx["pids"]).contiguous()
        G = operand_group(ctx, gate_table is not None)          # (node blocks an operand group of the slots holds)
        gkey = (int(pfids.min()), int(pfids.max()), nb, blk, ne, batch, params.dtype, G)
        groups = _groups_of(pf, create = True)
        group = groups.get(gkey)
        if group is None and tied:
            group = groups[gkey] = _Group(pfids, pids, nb, blk, ne, batch, pf, G)
        ok = (group is not None and group.S >= 2 * group.cps and torch.equal(group.pfids, pfids)
              and torch.equal(group.pids, pids))
        cids = _long(ctx["cids"]).contiguous().view(nb, -1)
        # (`shared`: every node block's edges the same children -- an HMM's blocks --, which the capture's warps use)
        member = cache[mkey] = ((gkey, _long(ctx["nids"]).contiguous(), cids,
                                 _long(gate_table).contiguous() if gate_table is not None else None,
                                 bool((cids == cids[:1]).all())) if ok else False)
    if member is False:
        return False
    gkey, nids, cids, gtab, shared = member
    group = _groups_of(pf, create = False).get(gkey)
    if group is None:
        return False

    if group.scope is not scope:                               # a new backward: whatever a past one left is dropped
        group.slots, group.scope = 0, scope
        group.mode = None
        scope.defer(lambda: group.flush(mod, pf, params))
        scope.at_close(("join_side", pf.device), functools.partial(join_side, pf.device))
    if group.slots + group.cps > group.S:                      # full: add what the slots hold, then go on
        group.flush(mod, pf, params)
    gated = gtab is not None                                   # (a regular layer's gate table and log-gates: unread)
    children = ctx.get("children")
    import pyjuice.layer.kernels.c as cuda_kernels
    cmod = cuda_kernels.get_par_deferred_capture_module()
    if group.steps_ok is None:
        group.steps_ok = cmod is not None and bool(cmod.fits(group.W, batch))
        assert group.steps_ok or (group.G == 1 and blk % 64 == 0), \
            "an operand group of node blocks, or node blocks of 32, take the chained capture"
    if group.steps_ok:
        # (the same operands in two chained launches, a step's column maxima its own: `par_deferred_fp16_capture.cu`)
        if group.slots == 0:
            group.Ms.zero_()
        fused = children is not None
        args = (group.A, group.Bop, group.fA, group.fB, group.Ms, ctx["node_flows"], ctx["node_mars"],
                ctx["node_mars"] if fused else ctx["element_mars"], children[0] if fused else group.Ms,
                int(children[1]) if fused else 0, ext if gated else ctx["node_mars"], nids, cids,
                gtab if gated else nids, batch, group.slots, node_cbs, gate_cbs, ext_base, gated, fused, shared)
        # (a side stream only for the children's values: `element_mars` rows are rewritten by the next layers'
        # products, and a gated capture reads values its backward shifts)
        if group.on_side(fused and not gated):
            side = _side_stream(pf.device)
            cmod.capture_steps(*args, side[0].cuda_stream)    # (forked there after the current stream's queue)
            side[1] = True
        else:
            cmod.capture_steps(*args, 0)
    elif children is not None:
        # (the product layer below fused: each edge's element value formed from its product's two children,
        # `par_deferred_fp16_fused.cu` -- the same operands, the product layer's forward not run for them)
        fmod = cuda_kernels.get_par_deferred_fused_module()
        if fmod is None:
            return False
        fmod.capture_fused(group.A, group.Bop, group.fA, group.fB, group.M, ctx["node_flows"], ctx["node_mars"],
                           children[0], int(children[1]), ext if gated else ctx["node_mars"], nids, cids,
                           gtab if gated else nids, batch, group.slots, node_cbs, gate_cbs, ext_base, gated)
    else:
        mod.capture(group.A, group.Bop, group.fA, group.fB, group.M, ctx["node_flows"], ctx["node_mars"],
                    ctx["element_mars"], ext if gated else ctx["node_mars"], nids, cids, gtab if gated else nids, batch,
                    group.slots, node_cbs, gate_cbs, ext_base, gated)
    group.slots += group.cps
    return True
