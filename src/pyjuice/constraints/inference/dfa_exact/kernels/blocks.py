"""
Moving the per-sample blocks of sum regions between batch layouts (:func:`move_blocks`). A decoder that keeps values in
the buffers between its queries (:mod:`..queries.decoder`) moves those a call reuses into that call's rows -- rows
reordered, duplicated or dropped, the batch size changed: every row's blocks are gathered from its parent's, from the
old storage into new storage laid out for the call.

Given the liveness of the call that left the values (:class:`~.live.Liveness`), only its live lanes are read, and
the others are written -inf: the next call's live lanes are among them (more evidence, fewer live lanes), and -inf is
never above a lane's true value (see :mod:`.live`). A block's live lanes are few, so this reads little, but every lane
is still written.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

from .common import _exit_width, _interval, _pair

#: rows of a region one program moves, and its columns
MOVE_ROWS, MOVE_TILE = 16, 1024


@triton.jit
def _move_rows(src, dst, so, sw, do, dw, nrows, cols, scol, m, alive):
    for r in range(nrows):
        tl.store(dst + do + r * dw + cols, tl.load(src + so + r * sw + scol, mask = alive, other = float("-inf")),
                 mask = m)


@triton_jit
def _move_blocks_kernel(src0, dst0, src1, dst1, table, parents, live, iv_of, iv_info, iv_table, width, B, n, W,
                        GATHER: tl.constexpr, LIVE: tl.constexpr, ROWS: tl.constexpr, TILE: tl.constexpr):
    """One program per (region, ``ROWS`` of its rows, ``TILE`` of its columns); ``table`` as :func:`move_blocks`'s.
    Column ``b * slots + s`` of a destination row is column ``parents[b] * slots + s`` of the source row (the same
    column without ``GATHER``) -- with ``LIVE``, where that lane is live for source sample ``parents[b]`` (``live``,
    the source call's flags: both ends of the slot's pair, over the region's interval ``[a, e)``), else -inf."""
    t = tl.program_id(0).to(tl.int64) * 9
    r0 = tl.program_id(1).to(tl.int64) * ROWS
    cols = tl.program_id(2) * TILE + tl.arange(0, TILE)          # 32-bit lane math, as the lane order's
    rows, slots = tl.load(table + t + 5), tl.load(table + t + 4).to(tl.int32)
    if r0 < rows:
        sw, dw = tl.load(table + t + 1), tl.load(table + t + 3)
        so, do = tl.load(table + t) + r0 * sw, tl.load(table + t + 2) + r0 * dw
        nrows = tl.minimum(ROWS, rows - r0)
        m = cols < B * slots
        scol = cols
        if GATHER or LIVE:
            b = cols // slots
            s = cols - b * slots
            p = b
            if GATHER:
                p = tl.load(parents + b, mask = m, other = 0).to(tl.int32)
                scol = p * slots + s
        if LIVE:
            a, e = tl.load(table + t + 7), tl.load(table + t + 8)
            _, _, pairs = _interval(iv_of, iv_info, a, e, n)
            X = _exit_width(width, e, n)
            v = _pair(iv_table, pairs, s, m, False)
            row = live + p.to(tl.int64) * (n + 1) * W
            alive = m & (tl.load(row + a * W + v // X, mask = m, other = 0) != 0) & \
                (tl.load(row + e * W + v % X, mask = m, other = 0) != 0)
        else:
            alive = m
        if tl.load(table + t + 6) == 0:
            _move_rows(src0, dst0, so, sw, do, dw, nrows, cols, scol, m, alive)
        else:
            _move_rows(src1, dst1, so, sw, do, dw, nrows, cols, scol, m, alive)


def move_table(src_offset: np.ndarray, src_width: np.ndarray, dst_offset: np.ndarray, dst_width: np.ndarray,
               slots: np.ndarray, rows: np.ndarray, buffer: np.ndarray,
               bounds: Optional[np.ndarray] = None) -> np.ndarray:
    """int64 [regions x 9]: :func:`move_blocks`'s table, every argument one entry per region (int64 [regions]): where
    its first row is in the source and in the destination, and their rows' widths there (floats), its slots per
    sample, its rows, which source and destination it moves between (0 or 1), and its interval's two ends (``bounds``
    [regions, 2]; None: no liveness)."""
    bounds = np.zeros((len(rows), 2), dtype = np.int64) if bounds is None else np.asarray(bounds).reshape(-1, 2)
    return np.stack([src_offset, src_width, dst_offset, dst_width, slots, rows, buffer, bounds[:, 0], bounds[:, 1]],
                    axis = 1).astype(np.int64)


def move_blocks(srcs: Sequence[torch.Tensor], dsts: Sequence[torch.Tensor], table: torch.Tensor, B: int,
                max_rows: int, max_slots: int, parents: Optional[torch.Tensor] = None,
                live: Optional[torch.Tensor] = None, prog = None):
    """
    Every region's per-sample blocks of the ``B`` destination samples, from ``srcs[k]`` into ``dsts[k]`` (``k`` the
    region's buffer): sample ``b``'s in a destination row are sample ``parents[b]``'s in the source row (``b``'s
    without ``parents``) -- given ``live``, at the lanes live there, and -inf at the others. A source and its
    destination must not overlap.

    :param srcs: one or two flat float tensors
    :param dsts: as many
    :param table: :func:`move_table`'s, on the device (int64 [regions x 9], flat or not)
    :param max_rows: the most rows of any region
    :param max_slots: the most slots of any region
    :param parents: None, or int64 [B] on the device
    :param live: None (every lane), or the source call's :class:`~.live.Liveness` ``live`` flags, with ``prog``, the
        lifted :class:`~..compiled.program.Program` (its interval tables)
    """
    regions = table.numel() // 9
    if regions == 0:
        return
    src0, src1 = srcs[0], srcs[-1]
    dst0, dst1 = dsts[0], dsts[-1]
    grid = (regions, host_cdiv(max_rows, MOVE_ROWS), host_cdiv(B * max_slots, MOVE_TILE))
    if live is not None:
        ivs = prog.intervals
        tables = (live, ivs.of, ivs.info, ivs.table, prog.width_t, prog.n, live.size(2))
    else:
        tables = (table,) * 5 + (0, 0)
    _move_blocks_kernel[grid](src0, dst0, src1, dst1, table, table if parents is None else parents, tables[0],
                              *tables[1:5], B, *tables[5:], GATHER = parents is not None, LIVE = live is not None,
                              ROWS = MOVE_ROWS, TILE = MOVE_TILE, num_warps = 4)
