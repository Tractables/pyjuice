"""
Evidence pruning. A column at boundary ``t`` is LIVE for a sample when the automaton can be there given the
sample's evidence: reachable from the initial state reading tokens of the observed classes on ``[0, t)`` (any class
where a token is missing), and still able to reach acceptance reading such tokens on ``[t, n)``. A slot ``(i, j)`` of
a block over ``[a, b)`` is live when ``i`` is live at ``a`` and ``j`` at ``b``. Every kernel computes the live slots
only and writes -inf to the others, which is exact: a live result reads a dead slot only as a term that is -inf
anyway --

* ``input @ block`` reads ``R[q, j]`` for a successor ``q`` of a live ``i`` through an allowed class: ``q`` is
  reachable, so if it is dead it cannot accept, and neither can any path ``q -> j -> acceptance``;
* ``block @ input`` reads ``L[i, q]`` for a predecessor ``q`` of a live ``j``: ``q`` can accept (through ``j``), so if
  it is dead it is unreachable, and so is any path ``initial -> i -> q``;
* ``block @ block`` reads both at a shared column, which, dead, is unreachable or cannot accept;
* a sum acts on each slot alone.

So an observed prefix pins every block before it to one column per sample, and a sample whose evidence breaks the
constraint keeps no live slot at all (its marginal is -inf).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import triton
import triton.language as tl

from .prod import _exit_width, _interval, _pair, grid_chunks


@triton.jit
def _liveness_kernel(live, back, obs_class, next_col, succ_ptr, succ, width, n, W, C, TILE: tl.constexpr):
    """One program per sample: ``live[b, t, q]`` (int8 [B, n + 1, W]; boundary ``n`` is the sequence end, column 0),
    forward reachability by a sweep over boundaries -- every live column marks its successors -- then ANDed with
    ``back`` (the same shape, scratch), whether acceptance can be reached, by a sweep the other way -- every column
    reads its successors. ``succ_ptr`` / ``succ``: every column's distinct successors (a CSR as
    :class:`~.prod.TransitionTables` grouped). A barrier makes each boundary visible to every lane before the next."""
    b = tl.program_id(0).to(tl.int64)
    base = b * (n + 1) * W
    lanes = tl.arange(0, TILE)
    ones = tl.full([TILE], 1, dtype = tl.int8)
    tl.store(live + base + lanes, (lanes == 0).to(tl.int8), mask = lanes < W)          # the initial column
    for t in range(0, n):
        nxt = _exit_width(width, t + 1, n)
        for j0 in range(0, nxt, TILE):
            tl.store(live + base + (t + 1) * W + j0 + lanes, tl.zeros([TILE], dtype = tl.int8), mask = j0 + lanes < nxt)
        tl.debug_barrier()
        oc = tl.load(obs_class + b * n + t)
        for q0 in range(0, tl.load(width + t), TILE):
            q = q0 + lanes
            f = (q < tl.load(width + t)) & (tl.load(live + base + t * W + q, mask = q < W, other = 0) != 0)
            if oc >= 0:                                                     # through the observed class only
                s = tl.load(next_col + (t * W + q) * C + oc, mask = f, other = -1)
                s = tl.where((t + 1 == n) & (s >= 0), 0, s)
                tl.store(live + base + (t + 1) * W + s, ones, mask = f & (s >= 0))
            else:                                                           # to every successor
                e0 = tl.load(succ_ptr + t * (W + 1) + q, mask = f, other = 0)
                e1 = tl.load(succ_ptr + t * (W + 1) + q + 1, mask = f, other = 0)
                for k in range(0, tl.max(e1 - e0, axis = 0)):
                    em = f & (e0 + k < e1)
                    tl.store(live + base + (t + 1) * W + tl.load(succ + e0 + k, mask = em, other = 0), ones, mask = em)
        tl.debug_barrier()
    tl.store(back + base + n * W, tl.full([], 1, dtype = tl.int8))
    for tt in range(0, n):
        t = n - 1 - tt
        tl.debug_barrier()
        oc = tl.load(obs_class + b * n + t)
        E = tl.load(width + t)
        for q0 in range(0, E, TILE):
            q = q0 + lanes
            qm = q < E
            g = tl.zeros([TILE], dtype = tl.int32)
            if oc >= 0:                                                     # the observed class's successor
                s = tl.load(next_col + (t * W + q) * C + oc, mask = qm, other = -1)
                s = tl.where((t + 1 == n) & (s >= 0), 0, s)
                g = (qm & (s >= 0) & (tl.load(back + base + (t + 1) * W + s, mask = qm & (s >= 0), other = 0) != 0)
                     ).to(tl.int32)
            else:                                                           # any successor
                e0 = tl.load(succ_ptr + t * (W + 1) + q, mask = qm, other = 0)
                e1 = tl.load(succ_ptr + t * (W + 1) + q + 1, mask = qm, other = 0)
                for k in range(0, tl.max(e1 - e0, axis = 0)):
                    em = qm & (e0 + k < e1)
                    s = tl.load(succ + e0 + k, mask = em, other = 0)
                    g = g | (em & (tl.load(back + base + (t + 1) * W + s, mask = em, other = 0) != 0)).to(tl.int32)
            tl.store(back + base + t * W + q, g.to(tl.int8), mask = qm)
            fwd = tl.load(live + base + t * W + q, mask = qm, other = 0)
            tl.store(live + base + t * W + q, ((fwd != 0) & (g != 0)).to(tl.int8), mask = qm)


@triton.jit
def _live_slot_kernel(flag, live, ivs, iv_of, iv_info, iv_table, width, B, n, W, pid0, TILE: tl.constexpr):
    """One program per (interval, tile of its ``B * slots`` columns): ``flag`` = 1 where the column's slot is live."""
    k = tl.program_id(0)
    tile = pid0 + tl.program_id(1)
    a = tl.load(ivs + k * 3)
    e = tl.load(ivs + k * 3 + 1)
    first = tl.load(ivs + k * 3 + 2).to(tl.int64)
    S, _, pairs = _interval(iv_of, iv_info, a, e, n)
    X = _exit_width(width, e, n)
    if tile * TILE < B * S:
        col = tile * TILE + tl.arange(0, TILE)
        m = col < B * S
        b = (col // S).to(tl.int64)
        v = _pair(iv_table, pairs, col % S, m, False)
        row = live + b * (n + 1) * W
        alive = (tl.load(row + a * W + v // X, mask = m, other = 0) != 0) & \
            (tl.load(row + e * W + v % X, mask = m, other = 0) != 0)
        tl.store(flag + first + col, alive.to(tl.int32), mask = m)


@triton.jit
def _live_order_kernel(flag, cum, perm, rank, ivs, iv_of, iv_info, B, n, pid0, TILE: tl.constexpr):
    """One program per (interval, tile of its columns): ``perm`` lists the interval's live columns in order, then its
    dead ones; ``rank`` holds every column's place among the live ones, -1 if dead (``cum``: inclusive prefix sums of
    ``flag`` over all intervals)."""
    k = tl.program_id(0)
    tile = pid0 + tl.program_id(1)
    a = tl.load(ivs + k * 3)
    e = tl.load(ivs + k * 3 + 1)
    first = tl.load(ivs + k * 3 + 2).to(tl.int64)
    S, _, _ = _interval(iv_of, iv_info, a, e, n)
    if tile * TILE < B * S:
        col = tile * TILE + tl.arange(0, TILE)
        m = col < B * S
        before = tl.load(cum + first) - tl.load(flag + first)              # live columns of earlier intervals
        num_live = tl.load(cum + first + B * S - 1) - before
        f = tl.load(flag + first + col, mask = m, other = 0)
        r = tl.load(cum + first + col, mask = m, other = 0) - f - before    # live columns before col
        tl.store(perm + first + tl.where(f != 0, r, num_live + col - r), col.to(tl.int32), mask = m)
        tl.store(rank + first + col, tl.where(f != 0, r, -1), mask = m)


@dataclass
class Pruning:
    """
    The live slots of one query (see the module docstring).

    :ivar live: int8 [B, n + 1, W], every column's liveness per sample and boundary
    :ivar perm: int32, per sum interval (``first[k]`` on), its ``B * slots`` columns: the live ones in order, then the
        dead ones
    :ivar rank: int32, the same columns' places among the live ones, -1 if dead
    :ivar first: per sum interval, where its columns start in ``perm`` and ``rank``
    :ivar counts: per sum interval, its live columns (a device tensor; :attr:`num_live` copies them to the host)
    :ivar reg_first, reg_live: per sum region, ``first`` and the live columns of its interval (device tensors)
    """

    live: torch.Tensor
    perm: torch.Tensor
    rank: torch.Tensor
    first: List[int]
    counts: torch.Tensor
    reg_first: torch.Tensor
    reg_live: torch.Tensor
    _num_live: Optional[List[int]] = None

    @property
    def num_live(self) -> List[int]:
        """Per sum interval, its live columns, on the host: copied on first use, the query's one synchronization
        (only a dense sum, whose product's size the host sets, needs them)."""
        if self._num_live is None:
            self._num_live = self.counts.tolist()
        return self._num_live

    def segment(self, k: int, ncols: int):
        """``(perm, rank, num_live)`` of sum interval ``k`` (``ncols``: its ``B * slots``)."""
        f = self.first[k]
        return self.perm[f:f + ncols], self.rank[f:f + ncols], self.num_live[k]


#: lanes of the list kernels
LIST_TILE = 512


def prune(prog, obs_class: torch.Tensor, B: int) -> Pruning:
    """The :class:`Pruning` of a query whose observed token classes are ``obs_class`` ([B, n] int32, -1 where
    missing). Nothing waits for the device."""
    n, W, C = prog.n, prog.next_col.size(1), prog.next_col.size(2)
    dev, ivs = prog.dev, prog.intervals
    live = torch.empty(B, n + 1, W, dtype = torch.int8, device = dev)
    back = torch.empty_like(live)
    TILE = min(1024, max(32, triton.next_power_of_2(W)))
    _liveness_kernel[(B,)](live, back, obs_class, prog.next_col, prog.succ.entry_ptr, prog.succ.entry_succ,
                           prog.width_t, n, W, C, TILE = TILE, num_warps = 4)

    if B not in prog.live_tables:                                           # where every sum interval's columns go
        sizes = [B * s for s in prog.sum_iv_slots]
        first = [0]
        for s in sizes:
            first.append(first[-1] + s)
        table = torch.tensor([(a, e, f) for (a, e), f in zip(prog.sum_ivs, first)], dtype = torch.int32).to(dev)
        starts = torch.tensor(first[:-1], dtype = torch.long, device = dev)
        prog.live_tables[B] = (sizes, first[:-1], first[-1], table, starts, starts + torch.tensor(sizes, device = dev))
    sizes, first, total, table, starts, ends = prog.live_tables[B]
    flag = torch.empty(total, dtype = torch.int32, device = dev)
    tiles = triton.cdiv(max(sizes), LIST_TILE)
    for f0, size in grid_chunks(tiles, 1):
        _live_slot_kernel[(len(sizes), size)](flag, live, table, ivs.of, ivs.info, ivs.table, prog.width_t, B, n, W,
                                              f0, TILE = LIST_TILE)
    cum = torch.cumsum(flag, 0, dtype = torch.int32)
    counts = cum[ends - 1] - cum[starts] + flag[starts]
    perm = torch.empty(total, dtype = torch.int32, device = dev)
    rank = torch.empty(total, dtype = torch.int32, device = dev)
    for f0, size in grid_chunks(tiles, 1):
        _live_order_kernel[(len(sizes), size)](flag, cum, perm, rank, table, ivs.of, ivs.info, B, n, f0,
                                               TILE = LIST_TILE)
    reg_first = starts[prog.reg_iv]
    return Pruning(live = live, perm = perm, rank = rank, first = first, counts = counts, reg_first = reg_first,
                   reg_live = counts[prog.reg_iv].to(torch.int32))
