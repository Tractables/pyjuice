"""
Moving the per-sample blocks of sum regions between batch layouts (:func:`move_blocks`). A decoder that keeps values in
the buffers between its queries (:mod:`..queries.decoder`) moves those a call reuses into that call's rows -- rows
reordered, duplicated or dropped, the batch size changed: every row's blocks are gathered from its parent's, from the
old storage into new storage laid out for the call.

Given the liveness of the call that left the values (:class:`~.live.Pruning`), only its live lanes are read, and the
others are written -inf: the next call's live lanes are among them (more evidence, fewer live lanes), and -inf is never
above a lane's true value (see :mod:`.live`). A block's live lanes are few, so this reads little, but every lane is
still written.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

#: rows of a region one program moves, and its columns
MOVE_ROWS, MOVE_TILE = 16, 1024


@triton.jit
def _move_rows(src, dst, so, sw, do, dw, nrows, cols, scol, m, alive):
    for r in range(nrows):
        tl.store(dst + do + r * dw + cols, tl.load(src + so + r * sw + scol, mask = alive, other = float("-inf")),
                 mask = m)


@triton_jit
def _move_blocks_kernel(src0, dst0, src1, dst1, table, parents, rank, B, GATHER: tl.constexpr, LIVE: tl.constexpr,
                        ROWS: tl.constexpr, TILE: tl.constexpr):
    """One program per (region, ``ROWS`` of its rows, ``TILE`` of its columns); ``table`` as :func:`move_blocks`'s.
    Column ``b * slots + s`` of a destination row is column ``parents[b] * slots + s`` of the source row (the same
    column without ``GATHER``) -- with ``LIVE``, where that column is live (``rank`` >= 0 at the region's start plus
    the column), else -inf."""
    t = tl.program_id(0).to(tl.int64) * 8
    r0 = tl.program_id(1).to(tl.int64) * ROWS
    cols = tl.program_id(2).to(tl.int64) * TILE + tl.arange(0, TILE)
    rows, slots = tl.load(table + t + 5), tl.load(table + t + 4)
    if r0 < rows:
        sw, dw = tl.load(table + t + 1), tl.load(table + t + 3)
        so, do = tl.load(table + t) + r0 * sw, tl.load(table + t + 2) + r0 * dw
        nrows = tl.minimum(ROWS, rows - r0)
        m = cols < B * slots
        if GATHER:
            b = cols // slots
            scol = tl.load(parents + b, mask = m, other = 0) * slots + (cols - b * slots)
        else:
            scol = cols
        if LIVE:
            alive = m & (tl.load(rank + tl.load(table + t + 7) + scol, mask = m, other = -1) >= 0)
        else:
            alive = m
        if tl.load(table + t + 6) == 0:
            _move_rows(src0, dst0, so, sw, do, dw, nrows, cols, scol, m, alive)
        else:
            _move_rows(src1, dst1, so, sw, do, dw, nrows, cols, scol, m, alive)


def move_table(src_offset: np.ndarray, src_width: np.ndarray, dst_offset: np.ndarray, dst_width: np.ndarray,
               slots: np.ndarray, rows: np.ndarray, buffer: np.ndarray,
               rank_first: Optional[np.ndarray] = None) -> np.ndarray:
    """int64 [regions x 8]: :func:`move_blocks`'s table, every argument one entry per region (int64 [regions]): where
    its first row is in the source and in the destination, and their rows' widths there (floats), its slots per
    sample, its rows, which source and destination it moves between (0 or 1), and where its source columns start in
    ``rank`` (None: no liveness)."""
    rank_first = np.zeros_like(rows) if rank_first is None else rank_first
    return np.stack([src_offset, src_width, dst_offset, dst_width, slots, rows, buffer, rank_first],
                    axis = 1).astype(np.int64)


def move_blocks(srcs: Sequence[torch.Tensor], dsts: Sequence[torch.Tensor], table: torch.Tensor, B: int,
                max_rows: int, max_slots: int, parents: Optional[torch.Tensor] = None,
                rank: Optional[torch.Tensor] = None):
    """
    Every region's per-sample blocks of the ``B`` destination samples, from ``srcs[k]`` into ``dsts[k]`` (``k`` the
    region's buffer): sample ``b``'s in a destination row are sample ``parents[b]``'s in the source row (``b``'s
    without ``parents``) -- given ``rank``, at the source columns live there, and -inf at the others. A source and its
    destination must not overlap.

    :param srcs: one or two flat float tensors
    :param dsts: as many
    :param table: :func:`move_table`'s, on the device (int64 [regions x 8], flat or not)
    :param max_rows: the most rows of any region
    :param max_slots: the most slots of any region
    :param parents: None, or int64 [B] on the device
    :param rank: None (every column), or a :class:`~.live.Pruning`'s ``rank`` of the source samples (int32, -1 at a
        dead column), each region's columns from the table's start on
    """
    regions = table.numel() // 8
    if regions == 0:
        return
    src0, src1 = srcs[0], srcs[-1]
    dst0, dst1 = dsts[0], dsts[-1]
    grid = (regions, host_cdiv(max_rows, MOVE_ROWS), host_cdiv(B * max_slots, MOVE_TILE))
    _move_blocks_kernel[grid](src0, dst0, src1, dst1, table, table if parents is None else parents,
                              table if rank is None else rank, B, GATHER = parents is not None,
                              LIVE = rank is not None, ROWS = MOVE_ROWS, TILE = MOVE_TILE, num_warps = 4)
