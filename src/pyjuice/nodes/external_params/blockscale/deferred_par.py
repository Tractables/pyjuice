"""
A tied group's gated parameter flows, deferred to one contraction (kernels/c/blockscale_par_deferred.cu).

Tied copies of a gated sum node -- a homogeneous HMM's steps -- write their parameter flows into ONE flow block
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
  * the step's kernel writes without either hazard (`_par_hook`'s guard, or `_par_write_flags` (0, 0)), one gate
    per node block, `blk % 64 == 0` and `num_edges % 64 == 0`, and the group is at least `MIN_FLOWS` flows (a
    smaller one's backward is host-bound, and there the flush after the last layer costs more than it saves);
  * the slots fit `PAR_DEFER_MB` with room for two steps (a full group flushes and goes on).
A group's slots belong to the scope that filled them: a new backward -- also after one that raised -- starts empty.
"""

from __future__ import annotations

import os
import weakref

import torch

from pyjuice.utils.deferred import current_scope

#: PYJUICE_BLOCKSCALE_PAR_DEFER=0 keeps every step's parameter flows in its own launch.
PAR_DEFER = os.environ.get("PYJUICE_BLOCKSCALE_PAR_DEFER", "1") != "0"

#: The operand slots a group may hold (MB).
PAR_DEFER_MB = float(os.environ.get("PYJUICE_BLOCKSCALE_PAR_DEFER_MB", "128"))

#: The smallest group (flows per step) that defers. The kernels win from ~1M flows (MEASURED, gate width 64, B 128,
#: their time against the per-step Triton kernels': 262K flows (HMM-512 in 2 blocks, 15 steps) 67 -> 81 us; 1M
#: (HMM-1024, 7 steps) 74 -> 56, 31 steps 345 -> 219; 16.8M (HMM-4096 in 4 blocks, 7 steps) 769 -> 364), but a
#: host-bound backward hides the per-step kernels in the GPU's idle time and exposes the flush after its last layer:
#: the whole tied backward, off -> on, HMM-4096 in 4 blocks 7 steps 1.87 -> 1.47 ms (B 512 5.01 -> 3.99, fp32
#: 2.11 -> 1.52) against HMM-2048 in 2 blocks (4M flows) 15 steps 2.26 -> 2.40 and HMM-1024 31 steps 4.64 -> 4.47.
MIN_FLOWS = 1 << 23

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

    def __init__(self, pfids, pids, nb, blk, ne, batch, device):
        self.Bp = (batch + 31) // 32 * 32
        per_slot = nb * (blk + ne) * self.Bp * 4
        self.S = int(min(64, (PAR_DEFER_MB * 2 ** 20) // per_slot))
        self.pfids, self.pids = pfids, pids
        self.slots, self.scope, self.cfg = 0, None, None
        if self.S >= 2:
            self.A = torch.zeros(nb, blk, self.S * self.Bp, device = device)
            self.Bop = torch.zeros(nb, ne, self.S * self.Bp, device = device)
            self.M = torch.empty(nb, self.Bp, device = device, dtype = torch.int32)

    def flush(self, mod, pf, params):
        if self.slots == 0:
            return
        if self.cfg is None:
            self.cfg = self._pick_cfg(mod, pf, params)
        mod.flush(pf, self.A, self.Bop, self.pfids, self.pids, params, self.slots, self.Bp, self.cfg)
        self.slots = 0

    def _pick_cfg(self, mod, pf, params):
        """The flush config, measured once into a scratch copy of the flows (the flush read-adds-writes them)."""
        import pyjuice.layer.kernels.c as ck

        blk, ne = self.A.size(1), self.Bop.size(1)
        cands = [i for i, (bm, bn, bk) in enumerate(mod.flush_configs())
                 if blk % bm == 0 and ne % bn == 0 and (self.slots * self.Bp) % bk == 0]
        if len(cands) == 1 or torch.cuda.is_current_stream_capturing():
            return cands[0]
        scr = pf.clone()
        trials = [(c, (lambda c = c: mod.flush(scr, self.A, self.Bop, self.pfids, self.pids, params, self.slots,
                                                self.Bp, c))) for c in cands]
        return ck.autotune(trials) or cands[0]


def _long(t):
    return t if t.dtype == torch.long else t.long()


def try_defer(mod, layer, ctx, gate_table, ext, node_cbs, gate_cbs, ext_base, n_node_gates) -> bool:
    """Capture this step's parameter flows into its tied group's slots (the flush deferred to the end of the pass)
    and return True; or return False, the caller then launching the step's own kernel."""
    if mod is None or not PAR_DEFER or not getattr(layer, "_bs_defer", False) or n_node_gates != 1:
        return False
    scope = current_scope()
    if scope is None:
        return False
    pf, params = ctx["param_flows"], ctx["params"]
    batch, blk, ne = ctx["batch_size"], ctx["block_size"], ctx["num_edges"]
    nb = ctx["nids"].numel()
    if blk % 64 != 0 or ne % 64 != 0 or nb * blk * ne < MIN_FLOWS:
        return False

    # This layer's group, resolved once per (partition, batch): None where it does not defer
    cache = layer.__dict__.setdefault("_bs_pd_member", {})
    mkey = (ctx["partition_id"], batch, id(ctx["pfids"]))
    member = cache.get(mkey)
    if member is None:
        if torch.cuda.is_current_stream_capturing():
            return False                                       # (the checks below sync)
        tied = any(info.ns.is_tied() for info in layer.external_node_infos)
        pfids, pids = _long(ctx["pfids"]).contiguous(), _long(ctx["pids"]).contiguous()
        gkey = (int(pfids.min()), int(pfids.max()), nb, blk, ne, batch, params.dtype)
        groups = _groups_of(pf, create = True)
        group = groups.get(gkey)
        if group is None and tied:
            group = groups[gkey] = _Group(pfids, pids, nb, blk, ne, batch, pf.device)
        ok = (group is not None and group.S >= 2 and torch.equal(group.pfids, pfids) and torch.equal(group.pids, pids))
        member = cache[mkey] = ((gkey, _long(ctx["nids"]).contiguous(), _long(ctx["cids"]).contiguous(),
                                 _long(gate_table).contiguous()) if ok else False)
    if member is False:
        return False
    gkey, nids, cids, gtab = member
    group = _groups_of(pf, create = False).get(gkey)
    if group is None:
        return False

    if group.scope is not scope:                               # a new backward: whatever a past one left is dropped
        group.slots, group.scope = 0, scope
        scope.defer(lambda: group.flush(mod, pf, params))
    if group.slots == group.S:                                 # full: add what the slots hold, then go on
        group.flush(mod, pf, params)
    mod.capture(group.A, group.Bop, group.M, ctx["node_flows"], ctx["node_mars"], ctx["element_mars"], ext, nids,
                cids, gtab, batch, group.slots, node_cbs, gate_cbs, ext_base)
    group.slots += 1
    return True
