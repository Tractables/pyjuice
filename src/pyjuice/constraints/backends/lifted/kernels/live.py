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

A block kept once for every sample (``prior_mars``, see :mod:`..query_plan`) is computed without pruning, so its dead
slots hold finite values. The arguments above still make every such slot a term of a dead product -- it is only ever
added to -inf -- so the values stay exact. They do enter the maxima ``block @ block`` shifts its exponentials by, which
can flush a live entry's terms: the tiles where that happens are computed again exactly (see
:func:`~.prod._block_block_tile`).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from .prod import _exit_width, _interval, _pair, grid_chunks
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2


@triton.jit
def _mark_successors(live, base, q, f, oc, next_col, succ_ptr, succ, t, n, W, C, TILE: tl.constexpr):
    """Forward sweep, boundary ``t`` -> ``t + 1``: the live columns ``q`` (``f``) mark their successors live at
    ``t + 1``, through the observed class ``oc`` only, or every successor when the token is missing (``oc`` -1)."""
    ones = tl.full([TILE], 1, dtype = tl.int8)
    if oc >= 0:                                                             # through the observed class only
        s = tl.load(next_col + (t * W + q) * C + oc, mask = f, other = -1)
        s = tl.where((t + 1 == n) & (s >= 0), 0, s)
        tl.store(live + base + (t + 1) * W + s, ones, mask = f & (s >= 0))
    else:                                                                   # to every successor
        e0 = tl.load(succ_ptr + t * (W + 1) + q, mask = f, other = 0)
        e1 = tl.load(succ_ptr + t * (W + 1) + q + 1, mask = f, other = 0)
        for k in range(0, tl.max(e1 - e0, axis = 0)):
            em = f & (e0 + k < e1)
            tl.store(live + base + (t + 1) * W + tl.load(succ + e0 + k, mask = em, other = 0), ones, mask = em)


@triton.jit
def _reach_back(live, back, base, q, qm, oc, next_col, succ_ptr, succ, t, n, W, C, TILE: tl.constexpr):
    """Backward sweep, boundary ``t + 1`` -> ``t``: the columns ``q`` (``qm``) can reach acceptance if a successor
    can (``back``, through the observed class ``oc`` or any class); stored in ``back`` and ANDed into ``live``."""
    g = tl.zeros([TILE], dtype = tl.int32)
    if oc >= 0:                                                             # the observed class's successor
        s = tl.load(next_col + (t * W + q) * C + oc, mask = qm, other = -1)
        s = tl.where((t + 1 == n) & (s >= 0), 0, s)
        g = (qm & (s >= 0) & (tl.load(back + base + (t + 1) * W + s, mask = qm & (s >= 0), other = 0) != 0)
             ).to(tl.int32)
    else:                                                                   # any successor
        e0 = tl.load(succ_ptr + t * (W + 1) + q, mask = qm, other = 0)
        e1 = tl.load(succ_ptr + t * (W + 1) + q + 1, mask = qm, other = 0)
        for k in range(0, tl.max(e1 - e0, axis = 0)):
            em = qm & (e0 + k < e1)
            s = tl.load(succ + e0 + k, mask = em, other = 0)
            g = g | (em & (tl.load(back + base + (t + 1) * W + s, mask = em, other = 0) != 0)).to(tl.int32)
    tl.store(back + base + t * W + q, g.to(tl.int8), mask = qm)
    fwd = tl.load(live + base + t * W + q, mask = qm, other = 0)
    tl.store(live + base + t * W + q, ((fwd != 0) & (g != 0)).to(tl.int8), mask = qm)


@triton_jit
def _liveness_kernel(live, back, obs_class, next_col, succ_ptr, succ, width, n, W, C, TILE: tl.constexpr):
    """One program per sample: ``live[b, t, q]`` (int8 [B, n + 1, W]; boundary ``n`` is the sequence end, column 0),
    forward reachability by a sweep over boundaries -- every live column marks its successors -- then ANDed with
    ``back`` (the same shape, scratch), whether acceptance can be reached, by a sweep the other way -- every column
    reads its successors. ``succ_ptr`` / ``succ``: every column's distinct successors (a CSR as
    :class:`~.prod.TransitionTables` grouped). A barrier makes each boundary visible to every lane before the next."""
    b = tl.program_id(0).to(tl.int64)
    base = b * (n + 1) * W
    lanes = tl.arange(0, TILE)
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
            _mark_successors(live, base, q, f, oc, next_col, succ_ptr, succ, t, n, W, C, TILE)
        tl.debug_barrier()
    tl.store(back + base + n * W, tl.full([], 1, dtype = tl.int8))
    for tt in range(0, n):
        t = n - 1 - tt
        tl.debug_barrier()
        oc = tl.load(obs_class + b * n + t)
        E = tl.load(width + t)
        for q0 in range(0, E, TILE):
            q = q0 + lanes
            _reach_back(live, back, base, q, q < E, oc, next_col, succ_ptr, succ, t, n, W, C, TILE)


@triton_jit
def _live_forward_kernel(live, obs_class, next_col, succ_ptr, succ, width, t, n, W, C, TILE: tl.constexpr):
    """Boundary ``t`` -> ``t + 1`` of :func:`_liveness_kernel`'s forward sweep as one launch, programs over (sample,
    tile of the boundary's columns): ``live`` holds the boundaries up to ``t``, and zeros after (concurrent marks of a
    column all write 1)."""
    b = tl.program_id(0).to(tl.int64)
    q = tl.program_id(1) * TILE + tl.arange(0, TILE)
    base = b * (n + 1) * W
    qm = q < tl.load(width + t)
    f = qm & (tl.load(live + base + t * W + q, mask = qm, other = 0) != 0)
    _mark_successors(live, base, q, f, tl.load(obs_class + b * n + t), next_col, succ_ptr, succ, t, n, W, C, TILE)


@triton_jit
def _live_backward_kernel(live, back, obs_class, next_col, succ_ptr, succ, width, t, n, W, C, TILE: tl.constexpr):
    """Boundary ``t + 1`` -> ``t`` of :func:`_liveness_kernel`'s backward sweep as one launch, programs over (sample,
    tile of the boundary's columns): ``back`` holds the boundaries after ``t``."""
    b = tl.program_id(0).to(tl.int64)
    q = tl.program_id(1) * TILE + tl.arange(0, TILE)
    base = b * (n + 1) * W
    _reach_back(live, back, base, q, q < tl.load(width + t), tl.load(obs_class + b * n + t), next_col, succ_ptr, succ,
                t, n, W, C, TILE)


@triton.jit
def _live_columns(live, ivs, iv_of, iv_info, iv_table, width, k, tile, B, n, W, TILE: tl.constexpr):
    """Tile ``tile`` of sum interval ``k``'s ``B * slots`` columns (sample-major): ``(columns, in range, live, first
    column in the lists, slots)`` -- a column is live where its slot's two ends are live for its sample."""
    a = tl.load(ivs + k * 4)
    e = tl.load(ivs + k * 4 + 1)
    first = tl.load(ivs + k * 4 + 2).to(tl.int64)
    S, _, pairs = _interval(iv_of, iv_info, a, e, n)
    X = _exit_width(width, e, n)
    col = tile * TILE + tl.arange(0, TILE)
    m = col < B * S
    b = (col // S).to(tl.int64)
    v = _pair(iv_table, pairs, col % S, m, False)
    row = live + b * (n + 1) * W
    alive = m & (tl.load(row + a * W + v // X, mask = m, other = 0) != 0) & \
        (tl.load(row + e * W + v % X, mask = m, other = 0) != 0)
    return col, m, alive, first, S


@triton_jit
def _live_count_kernel(tile_live, live, ivs, iv_of, iv_info, iv_table, width, B, n, W, pid0, TILE: tl.constexpr):
    """One program per (sum interval, tile of its columns): the tile's live columns, at ``tile_live[ivs[k, 3] +
    tile]``."""
    k = tl.program_id(0)
    tile = pid0 + tl.program_id(1)
    col, m, alive, first, S = _live_columns(live, ivs, iv_of, iv_info, iv_table, width, k, tile, B, n, W, TILE)
    if tile * TILE < B * S:
        tl.store(tile_live + tl.load(ivs + k * 4 + 3) + tile, tl.sum(alive.to(tl.int32), axis = 0))


@triton_jit
def _live_order_kernel(tile_live, counts, perm, rank, reg_live, iv_regs, iv_reg_ptr, live, ivs, iv_of, iv_info,
                       iv_table, width, B, n, W, pid0, TILE: tl.constexpr):
    """One program per (sum interval, tile of its columns): ``perm`` lists the interval's live columns in order, then
    its dead ones, and ``rank`` holds every column's place among the live ones (-1 if dead), from the earlier tiles'
    live counts (:func:`_live_count_kernel`) and the tile's own prefix sums; the first tile writes the interval's
    count to ``counts`` and to its regions' ``reg_live``."""
    k = tl.program_id(0)
    tile = pid0 + tl.program_id(1)
    col, m, alive, first, S = _live_columns(live, ivs, iv_of, iv_info, iv_table, width, k, tile, B, n, W, TILE)
    t0 = tl.load(ivs + k * 4 + 3)
    tiles = (B * S + TILE - 1) // TILE
    before = tl.zeros([], dtype = tl.int32)
    total = tl.zeros([], dtype = tl.int32)
    for c0 in range(0, tiles, TILE):                                       # the interval's tiles' live counts
        c = c0 + tl.arange(0, TILE)
        cnt = tl.load(tile_live + t0 + c, mask = c < tiles, other = 0)
        before += tl.sum(tl.where(c < tile, cnt, 0), axis = 0)
        total += tl.sum(cnt, axis = 0)
    if tile * TILE < B * S:
        f = alive.to(tl.int32)
        r = before + tl.cumsum(f, axis = 0) - f                            # live columns before each column
        tl.store(perm + first + tl.where(alive, r, total + col - r), col.to(tl.int32), mask = m)
        tl.store(rank + first + col, tl.where(alive, r, -1), mask = m)
    if tile == 0:                                                          # (an interval without slots: 0)
        tl.store(counts + k, total)
        for j in range(tl.load(iv_reg_ptr + k), tl.load(iv_reg_ptr + k + 1)):
            tl.store(reg_live + tl.load(iv_regs + j), total)


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

#: The liveness sweeps run as a launch per boundary (:func:`_live_forward_kernel`, :func:`_live_backward_kernel`, on
#: samples x tiles of LIVE_STEP_TILE columns) when the widest boundary has at least this many columns; below, one
#: program per sample sweeps them all (:func:`_liveness_kernel`). HMM4096, batch 1: the sweep of a 47k-column
#: automaton took 5-7 ms in one program per sample, 0.66 ms per boundary; below ~4k columns the 2n launches (~0.65 ms)
#: cost more than the one-program sweep (0.12-0.2 ms).
LIVE_STEPS_MIN_WIDTH = 4096
LIVE_STEP_TILE = 256


def prune(prog, obs_class: torch.Tensor, B: int) -> Pruning:
    """The :class:`Pruning` of a query whose observed token classes are ``obs_class`` ([B, n] int32, -1 where
    missing). Nothing waits for the device."""
    n, W, C = prog.n, prog.next_col.size(1), prog.next_col.size(2)
    dev, ivs = prog.dev, prog.intervals
    live = torch.empty(B, n + 1, W, dtype = torch.int8, device = dev)
    back = torch.empty_like(live)
    tables = (obs_class, prog.next_col, prog.succ.entry_ptr, prog.succ.entry_succ, prog.width_t)
    if W >= LIVE_STEPS_MIN_WIDTH:                    # a launch per boundary, programs over samples and columns
        live.zero_()
        live[:, 0, 0] = 1                                                   # the initial column
        back[:, n, 0] = 1                                                   # the sequence end
        for t in range(n):
            _live_forward_kernel[(B, host_cdiv(prog.width[t], LIVE_STEP_TILE))](
                live, *tables, t, n, W, C, TILE = LIVE_STEP_TILE, num_warps = 4)
        for t in range(n - 1, -1, -1):
            _live_backward_kernel[(B, host_cdiv(prog.width[t], LIVE_STEP_TILE))](
                live, back, *tables, t, n, W, C, TILE = LIVE_STEP_TILE, num_warps = 4)
    else:                                            # one program per sample, sweeping every boundary
        TILE = min(1024, max(32, host_next_power_of_2(W)))
        _liveness_kernel[(B,)](live, back, *tables, n, W, C, TILE = TILE, num_warps = 4)

    if B not in prog.live_tables:                    # where every sum interval's columns and their tiles' counts go
        sizes = [B * s for s in prog.sum_iv_slots]
        first, tiles = np.cumsum([0] + sizes), np.cumsum([0] + [host_cdiv(z, LIST_TILE) for z in sizes])
        table = torch.tensor([(a, e, f, t) for (a, e), f, t in zip(prog.sum_ivs, first[:-1].tolist(), tiles[:-1].tolist())],
                             dtype = torch.int32).to(dev)
        regs = np.argsort(np.asarray(prog.reg_iv_host), kind = "stable")  # every interval's regions
        reg_ptr = np.searchsorted(np.asarray(prog.reg_iv_host)[regs], np.arange(len(sizes) + 1))
        reg_first = torch.from_numpy(first[:-1][np.asarray(prog.reg_iv_host)]).to(dev)
        prog.live_tables[B] = (sizes, first[:-1].tolist(), int(first[-1]), int(tiles[-1]), table,
                               torch.from_numpy(regs.astype(np.int32)).to(dev),
                               torch.from_numpy(reg_ptr.astype(np.int32)).to(dev), reg_first)
    sizes, first, total, num_tiles, table, iv_regs, iv_reg_ptr, reg_first = prog.live_tables[B]
    K = len(sizes)
    tile_live = torch.empty(num_tiles, dtype = torch.int32, device = dev)
    counts = torch.empty(K, dtype = torch.int32, device = dev)
    reg_live = torch.empty(len(prog.reg_iv_host), dtype = torch.int32, device = dev)
    perm = torch.empty(total, dtype = torch.int32, device = dev)
    rank = torch.empty(total, dtype = torch.int32, device = dev)
    lists = (live, table, ivs.of, ivs.info, ivs.table, prog.width_t)
    for f0, size in grid_chunks(host_cdiv(max(sizes), LIST_TILE), 1):
        _live_count_kernel[(K, size)](tile_live, *lists, B, n, W, f0, TILE = LIST_TILE)
    for f0, size in grid_chunks(host_cdiv(max(sizes), LIST_TILE), 1):
        _live_order_kernel[(K, size)](tile_live, counts, perm, rank, reg_live, iv_regs, iv_reg_ptr, *lists, B, n, W,
                                      f0, TILE = LIST_TILE)
    return Pruning(live = live, perm = perm, rank = rank, first = first, counts = counts, reg_first = reg_first,
                   reg_live = reg_live)
