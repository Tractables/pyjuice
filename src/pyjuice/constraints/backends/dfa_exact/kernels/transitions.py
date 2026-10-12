"""
Transition masses: for a missing token at a position, a column and one of its successors, the log-mass of the token
classes that lead there, from the class masses (:func:`transition_masses`). The products' ``input @ block`` and
``block @ input`` steps read them, so a missing token costs a gather per successor, not per class (see
:mod:`.prod`). Their tables (:class:`TransitionTables`) are built once per backend.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ....distributions.categorical import _exact_dot
from .common import grid_chunks



@dataclass(frozen = True)
class TransitionTables:
    """
    Every boundary's transitions as entries (column, successor, mass), the successor of a sequence end being its one
    column. Ungrouped, one entry per (column, class) that leads somewhere, its mass the class's mass. Grouped, one
    per (column, successor) pair, its mass the pair's transition mass: the mass of every class that leads there
    (:func:`transition_masses`) -- a missing token then costs a gather per successor, not per class.

    :ivar grouped: which of the two
    :ivar entry_ptr: [n, W + 1] int32, the entries of column ``q`` at ``t``: ``entry_ptr[t, q] .. entry_ptr[t, q + 1]``
        (global numbers; ``entry_ptr[t, 0]``: the boundary's first)
    :ivar entry_succ, entry_x: [entries] int32, every entry's successor column and where its mass is: its class,
        or its pair's number within the boundary
    :ivar pred_ptr, pred_col, pred_x: the entries leading to column ``j`` at ``t + 1``, as a CSR (``pred_ptr``
        [n, W + 1]) into their columns and their mass indices
    :ivar pair_of: grouped only, [n, W, C] int32: the pair (within the boundary) every class of every column leads
        to, -1 where it leads nowhere
    :ivar num_pairs: grouped only, pairs per boundary (Python ints)
    :ivar max_pairs: grouped only, the most pairs of any column
    """

    grouped: bool
    entry_ptr: torch.Tensor
    entry_succ: torch.Tensor
    entry_x: torch.Tensor
    pred_ptr: torch.Tensor
    pred_col: torch.Tensor
    pred_x: torch.Tensor
    pair_of: Optional[torch.Tensor]
    num_pairs: list
    max_pairs: int


def max_successors(next_col: torch.Tensor, width, n: int) -> int:
    """The most distinct columns one column reaches in one token, at any boundary (a sequence end being one)."""
    W = next_col.size(1)
    nc, most = next_col.cpu().long(), 0
    for t in range(n - 1):                                                  # the last boundary: one column
        succ = nc[t, :int(width[t])]
        q, c = torch.nonzero(succ >= 0, as_tuple = True)
        if q.numel():
            pairs = torch.unique(q * (W + 1) + succ[q, c])
            most = max(most, int(torch.bincount(pairs // (W + 1)).max()))
    return max(1, most)


def transition_tables(next_col: torch.Tensor, width, n: int, grouped: bool) -> TransitionTables:
    """The :class:`TransitionTables` of a layout's ``next_col`` [n, W, C] and boundary widths."""
    W, C = next_col.size(1), next_col.size(2)
    nc = next_col.cpu().long()
    entry_ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    pred_ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    pair_of = torch.full((n, W, C), -1, dtype = torch.int64) if grouped else None
    succs, xs, cols, pxs, num_pairs = [], [], [], [], []
    base = max_pairs = 0
    for t in range(n):
        wt = int(width[t])
        succ = nc[t, :wt]                                                    # [wt, C]
        if t + 1 == n:
            succ = torch.where(succ >= 0, 0, -1)                             # a sequence end: one column
        q, c = torch.nonzero(succ >= 0, as_tuple = True)                     # by column, then class
        if grouped:
            key, inverse = torch.unique(q * (W + 1) + succ[q, c], return_inverse = True)    # by column, successor
            eq, es, ex = key // (W + 1), key % (W + 1), torch.arange(key.numel())
            pair_of[t, q, c] = inverse
            num_pairs.append(key.numel())
            max_pairs = max(max_pairs, int(torch.bincount(eq, minlength = 1).max()) if key.numel() else 0)
        else:
            eq, es, ex = q, succ[q, c], c
        entry_ptr[t] = base + torch.searchsorted(eq, torch.arange(W + 1))
        by_succ = torch.argsort(es * (W + 1) + eq, stable = True)
        pred_ptr[t] = base + torch.searchsorted(es[by_succ], torch.arange(W + 1))
        succs.append(es); xs.append(ex); cols.append(eq[by_succ]); pxs.append(ex[by_succ])
        base += eq.numel()
    dev = next_col.device
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    cat = lambda xs: i32(torch.cat(xs)) if base > 0 else torch.zeros(1, dtype = torch.int32, device = dev)
    return TransitionTables(grouped = grouped, entry_ptr = i32(entry_ptr), entry_succ = cat(succs), entry_x = cat(xs),
                            pred_ptr = i32(pred_ptr), pred_col = cat(cols), pred_x = cat(pxs),
                            pair_of = i32(pair_of) if grouped else None, num_pairs = num_pairs, max_pairs = max_pairs)


#: :func:`transition_masses`' tiles: the most rows (input steps at one position) and columns per program, classes per
#: product, warps. Rows and columns shrink until there are enough programs (they never change a result); the pairs
#: per product follow the widest column, up to 64.
TRANSITION_TILES = dict(TS = 64, QT = 4, TC = 16, warps = 4, programs = 1024)


def transition_job(rows: torch.Tensor, dev):
    """
    The steps whose transition masses one :func:`transition_masses` call builds: ``rows`` [N, 3] (input row,
    position, offset of its masses), as ``(rows, tiles, TS)`` on ``dev``, the rows ordered by position and ``tiles``
    [tiles, 3] (first row, number of rows, position) cutting them into tiles of at most ``TS`` rows at one position
    (fewer when there are few rows, so that there are enough programs).
    """
    TS = min(TRANSITION_TILES["TS"], max(16, host_next_power_of_2(rows.size(0)) // 32))
    rows = rows[torch.argsort(rows[:, 1], stable = True)]
    pos, counts = torch.unique_consecutive(rows[:, 1], return_counts = True)
    tiles, start = [], 0
    for t, k in zip(pos.tolist(), counts.tolist()):
        tiles += [(start + i, min(TS, k - i), t) for i in range(0, k, TS)]
        start += k
    i32 = lambda x: torch.as_tensor(x, dtype = torch.int32).reshape(-1, 3).to(dev).contiguous()
    return i32(rows), i32(tiles), TS


@triton_jit
def _transition_kernel(class_mars, mass_row, trans, rows, tiles, pair_of, entry_ptr, width, W, C, input_start,
                       num_qtiles, pid0, TS: tl.constexpr, QT: tl.constexpr, TP: tl.constexpr, TC: tl.constexpr):
    """
    One program per (tile of rows at one position, ``QT`` columns): per column, its pairs ``TP`` at a time, as the
    product of the rows' class masses -- linear, scaled by 2^64, in natural class order -- with the 0/1 table of
    which of the pairs every class leads to. A class mass is a sum of fp32 probabilities, so 0 or at least 2^-149:
    scaled, a normal float, which tensor cores do not flush. Its log is the transition mass.
    """
    pid = pid0 + tl.program_id(0)
    tile = pid // num_qtiles
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = tl.arange(0, TS)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    off = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)
    masses = class_mars + tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64) * C
    base = tl.load(entry_ptr + t * (W + 1))                                  # the boundary's first pair
    jj = tl.arange(0, TP)
    for qi in range(QT):
        q = (pid % num_qtiles) * QT + qi
        if q < tl.load(width + t):
            g_end = tl.load(entry_ptr + t * (W + 1) + q + 1) - base
            col = pair_of + (t * W + q).to(tl.int64) * C
            for g0 in range(tl.load(entry_ptr + t * (W + 1) + q) - base, g_end, TP):
                acc = tl.zeros([TS, TP], dtype = tl.float32)
                for c0 in range(0, C, TC):
                    c = c0 + tl.arange(0, TC)
                    cmask = c < C
                    x = tl.exp(tl.load(masses[:, None] + c[None, :], mask = rmask[:, None] & cmask[None, :],
                                       other = float("-inf")) + 44.3614195558365)                # x 2^64
                    g = tl.load(col + c, mask = cmask, other = -1) - g0                          # pair in the tile
                    acc += _exact_dot(x, tl.where(g[:, None] == jj[None, :], 1.0, 0.0))
                tl.store(trans + off[:, None] + (g0 + jj)[None, :], tl.log(acc) - 44.3614195558365,
                         mask = rmask[:, None] & (g0 + jj < g_end)[None, :])


def transition_masses(trans, job, class_mars, prog):
    """
    Write the transition masses of every step of ``job`` (from :func:`transition_job`): for each column ``q`` at the
    input's position ``t`` and each successor ``s``, ``log sum_{c: next_col[t, q, c] = s} exp(class_mars[r, c])``
    (``r``: input ``u``'s row of the class masses), at ``trans[offset + p]`` for the pair's number ``p`` within the
    boundary.
    """
    rows, tiles, TS = job
    tt, cfg = prog.trans, TRANSITION_TILES
    W, C = prog.next_col.size(1), prog.next_col.size(2)
    QT = cfg["QT"]
    while QT > 1 and tiles.size(0) * host_cdiv(max(prog.width), QT) < cfg["programs"]:
        QT //= 2
    num_qtiles = host_cdiv(max(prog.width), QT)
    TP = min(64, max(16, host_next_power_of_2(tt.max_pairs)))
    for first, size in grid_chunks(tiles.size(0) * num_qtiles, 0):
        _transition_kernel[(size,)](class_mars, prog.mass_row, trans, rows, tiles, tt.pair_of, tt.entry_ptr,
                                    prog.width_t, W, C,
                                    prog.input_start, num_qtiles, first, TS = TS, QT = QT, TP = TP, TC = cfg["TC"],
                                    num_warps = cfg["warps"])
