"""
Moving the per-sample blocks of sum regions between batch layouts (:func:`move_blocks`). A decoder that keeps values in
the buffers between its queries (:mod:`..queries.decoder`) moves those a call reuses into that call's rows -- rows
reordered, duplicated or dropped, the batch size changed: every row's blocks are gathered from its parent's, from the
old storage into new storage laid out for the call.
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
def _move_rows(src, dst, so, sw, do, dw, nrows, cols, scol, m):
    for r in range(nrows):
        tl.store(dst + do + r * dw + cols, tl.load(src + so + r * sw + scol, mask = m), mask = m)


@triton_jit
def _move_blocks_kernel(src0, dst0, src1, dst1, table, parents, B, GATHER: tl.constexpr, ROWS: tl.constexpr,
                        TILE: tl.constexpr):
    """One program per (region, ``ROWS`` of its rows, ``TILE`` of its columns); ``table`` as :func:`move_blocks`'s.
    Column ``b * slots + s`` of a destination row is column ``parents[b] * slots + s`` of the source row (the same
    column without ``GATHER``)."""
    t = tl.program_id(0).to(tl.int64) * 7
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
        if tl.load(table + t + 6) == 0:
            _move_rows(src0, dst0, so, sw, do, dw, nrows, cols, scol, m)
        else:
            _move_rows(src1, dst1, so, sw, do, dw, nrows, cols, scol, m)


def move_table(src_offset: np.ndarray, src_width: np.ndarray, dst_offset: np.ndarray, dst_width: np.ndarray,
               slots: np.ndarray, rows: np.ndarray, buffer: np.ndarray) -> np.ndarray:
    """int64 [regions x 7]: :func:`move_blocks`'s table, every argument one entry per region (int64 [regions]): where
    its first row is in the source and in the destination, and their rows' widths there (floats), its slots per
    sample, its rows, and which source and destination it moves between (0 or 1)."""
    return np.stack([src_offset, src_width, dst_offset, dst_width, slots, rows, buffer], axis = 1).astype(np.int64)


def move_blocks(srcs: Sequence[torch.Tensor], dsts: Sequence[torch.Tensor], table: torch.Tensor, B: int,
                max_rows: int, max_slots: int, parents: Optional[torch.Tensor] = None):
    """
    Every region's per-sample blocks of the ``B`` destination samples, from ``srcs[k]`` into ``dsts[k]`` (``k`` the
    region's buffer): sample ``b``'s in a destination row are sample ``parents[b]``'s in the source row (``b``'s
    without ``parents``). A source and its destination must not overlap.

    :param srcs: one or two flat float tensors
    :param dsts: as many
    :param table: :func:`move_table`'s, on the device (int64 [regions x 7], flat or not)
    :param max_rows: the most rows of any region
    :param max_slots: the most slots of any region
    :param parents: None, or int64 [B] on the device
    """
    regions = table.numel() // 7
    if regions == 0:
        return
    src0, src1 = srcs[0], srcs[-1]
    dst0, dst1 = dsts[0], dsts[-1]
    grid = (regions, host_cdiv(max_rows, MOVE_ROWS), host_cdiv(B * max_slots, MOVE_TILE))
    _move_blocks_kernel[grid](src0, dst0, src1, dst1, table, table if parents is None else parents, B,
                              GATHER = parents is not None, ROWS = MOVE_ROWS, TILE = MOVE_TILE, num_warps = 4)
