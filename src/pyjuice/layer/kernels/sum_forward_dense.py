"""
The two element-wise passes around the cuBLAS product of a dense sum layer (`SumLayer._forward_dense`): shift the
children by their column maximum and exponentiate, then take the log of the product and shift back. One launch
each, so that a small layer is not dominated by launches.

The rows may be several groups of `rows_per_group` consecutive rows, each with its own column maxima (`colmax` one
row per group): the groups of a batched product. `GROUPED` is a compile-time switch so that the one-group case
stays a single broadcast row. Their maxima come from `dense_colmax`: torch's `amax` over the middle dimension of a
[groups, rows, batch] view copies its input (128 MB for 32 HCLT-1024 groups at batch 512), the kernel nothing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fw_dense_colmax_kernel(element_mars, colmax, row_start, rows_per_group, batch_size,
                            TILE_R: tl.constexpr, TILE_B: tl.constexpr):
    g = tl.program_id(0)
    cols = tl.program_id(1) * TILE_B + tl.arange(0, TILE_B)
    cmask = cols < batch_size
    acc = tl.full([TILE_B], float("-inf"), dtype = tl.float32)
    for r0 in range(0, rows_per_group, TILE_R):
        rows = r0 + tl.arange(0, TILE_R)
        mask = (rows < rows_per_group)[:, None] & cmask[None, :]
        x = tl.load(element_mars + (row_start + g * rows_per_group + rows).to(tl.int64)[:, None] * batch_size + cols[None, :],
                    mask = mask, other = float("-inf"))
        acc = tl.maximum(acc, tl.max(x, axis = 0))
    tl.store(colmax + g.to(tl.int64) * batch_size + cols, acc, mask = cmask)


@triton.jit
def _fw_dense_exp_kernel(element_mars, colmax, ex, row_start, num_rows, batch_size, rows_per_group,
                         TILE_R: tl.constexpr, TILE_B: tl.constexpr, GROUPED: tl.constexpr):
    rows = tl.program_id(0) * TILE_R + tl.arange(0, TILE_R)
    cols = tl.program_id(1) * TILE_B + tl.arange(0, TILE_B)
    mask = (rows < num_rows)[:, None] & (cols < batch_size)[None, :]
    if GROUPED:
        m = tl.load(colmax + (rows // rows_per_group).to(tl.int64)[:, None] * batch_size + cols[None, :], mask = mask,
                    other = 0.0)
    else:
        m = tl.load(colmax + cols, mask = cols < batch_size, other = 0.0)[None, :]
    shift = tl.where(m == float("-inf"), 0.0, m)
    x = tl.load(element_mars + (row_start + rows).to(tl.int64)[:, None] * batch_size + cols[None, :], mask = mask,
                other = float("-inf"))
    tl.store(ex + rows.to(tl.int64)[:, None] * batch_size + cols[None, :], tl.exp(x - shift), mask = mask)


@triton.jit
def _fw_dense_log_kernel(node_mars, colmax, row_start, num_rows, batch_size, rows_per_group,
                         TILE_R: tl.constexpr, TILE_B: tl.constexpr, GROUPED: tl.constexpr):
    rows = tl.program_id(0) * TILE_R + tl.arange(0, TILE_R)
    cols = tl.program_id(1) * TILE_B + tl.arange(0, TILE_B)
    mask = (rows < num_rows)[:, None] & (cols < batch_size)[None, :]
    if GROUPED:
        m = tl.load(colmax + (rows // rows_per_group).to(tl.int64)[:, None] * batch_size + cols[None, :], mask = mask,
                    other = 0.0)
    else:
        m = tl.load(colmax + cols, mask = cols < batch_size, other = 0.0)[None, :]
    shift = tl.where(m == float("-inf"), 0.0, m)
    ptrs = node_mars + (row_start + rows).to(tl.int64)[:, None] * batch_size + cols[None, :]
    tl.store(ptrs, tl.log(tl.load(ptrs, mask = mask, other = 1.0)) + shift, mask = mask)


def dense_colmax(element_mars, row_start: int, num_groups: int, rows_per_group: int, batch_size: int):
    """The column max of each group of ``rows_per_group`` rows from ``row_start``: ``[num_groups, batch_size]``."""
    TILE_R, TILE_B = 32, 32
    colmax = torch.empty(num_groups, batch_size, device = element_mars.device)
    grid = (num_groups, triton.cdiv(batch_size, TILE_B))
    _fw_dense_colmax_kernel[grid](element_mars, colmax, row_start, rows_per_group, batch_size, TILE_R = TILE_R,
                                  TILE_B = TILE_B, num_warps = 4)
    return colmax


def dense_exp(element_mars, colmax, ex, row_start: int, num_rows: int, batch_size: int, rows_per_group: int = None):
    """
    ``ex = exp(element_mars[row_start:row_start + num_rows] - shift)``, ``shift`` the column max (0 where -inf):
    ``colmax`` is ``[batch_size]``, or with ``rows_per_group`` one row per group of that many consecutive rows.
    """
    TILE_R, TILE_B = 32, 128
    grouped = rows_per_group is not None and rows_per_group < num_rows
    grid = (triton.cdiv(num_rows, TILE_R), triton.cdiv(batch_size, TILE_B))
    _fw_dense_exp_kernel[grid](element_mars, colmax, ex, row_start, num_rows, batch_size,
                               rows_per_group if grouped else num_rows, TILE_R = TILE_R, TILE_B = TILE_B,
                               GROUPED = grouped, num_warps = 4)


def dense_log(node_mars, colmax, row_start: int, num_rows: int, batch_size: int, rows_per_group: int = None):
    """``node_mars[rows] = log(node_mars[rows]) + shift`` in place, ``shift`` and ``colmax`` as in :func:`dense_exp`."""
    TILE_R, TILE_B = 32, 128
    grouped = rows_per_group is not None and rows_per_group < num_rows
    grid = (triton.cdiv(num_rows, TILE_R), triton.cdiv(batch_size, TILE_B))
    _fw_dense_log_kernel[grid](node_mars, colmax, row_start, num_rows, batch_size,
                               rows_per_group if grouped else num_rows, TILE_R = TILE_R, TILE_B = TILE_B,
                               GROUPED = grouped, num_warps = 4)
