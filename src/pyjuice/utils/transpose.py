"""
A token-major copy of a parameter table's rows, ``out[v, k] = table[rows[k], v]``, by a tiled kernel: a [rows,
tokens] tile read along the tokens and written along the rows, at a plain copy's rate, where torch's transposed copy
(``copy_`` of a ``.t()`` view, or ``.t().contiguous()``) runs at about half of it. MEASURED (RTX PRO 6000, a [4096,
126464] fp32 table, the whole table): 2.85 ms to fp32 and 2.13 to bf16, against torch's 5.20 and 4.58 and a plain
copy of the table's bytes in 2.84. The same values as torch's: a copy, cast to ``out``'s type (round to nearest
even).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

#: rows and tokens a program copies, and its warps (MEASURED on the table above: 128 x 64 at 4 warps the fastest of
#: 32-128 x 32-128 at 4 and 8 warps, all within 5% of a plain copy)
TILES = dict(TR = 128, TV = 64, warps = 4)
#: the most rows one launch takes (a grid's second axis: at most 65535 programs)
GRID_ROWS = 65535


@triton_jit
def _transposed_rows_kernel(table, rows, out, r0, num_rows, V, ld, GATHER: tl.constexpr, TR: tl.constexpr,
                            TV: tl.constexpr):
    """``out[v, k] = table[rows[k], v]`` (``GATHER``; else row ``r0 + k``), cast to ``out``'s type, for ``k <
    num_rows``; ``out``'s rows ``ld`` apart."""
    k = tl.program_id(1).to(tl.int64) * TR + tl.arange(0, TR)
    v = tl.program_id(0).to(tl.int64) * TV + tl.arange(0, TV)
    km = k < num_rows
    if GATHER:
        r = tl.load(rows + k, mask = km, other = 0)
    else:
        r = r0 + k
    m = km[:, None] & (v < V)[None, :]
    x = tl.load(table + r[:, None] * V + v[None, :], mask = m)
    tl.store(out + v[None, :] * ld + k[:, None], x.to(out.dtype.element_ty), mask = m)


def transposed_rows(table: torch.Tensor, out: torch.Tensor, rows: Optional[torch.Tensor] = None,
                    first: int = 0) -> torch.Tensor:
    """
    ``out[v, k] = table[rows[k], v]`` -- ``rows`` int64 on ``table``'s device, or None for the consecutive rows
    ``first .. first + out.size(1)`` --, cast to ``out``'s type; returns ``out``.

    :param table: [num_rows, V], contiguous, any floating type
    :param out: [V, number of rows], its columns adjacent (``stride(1) == 1``, e.g. columns of a wider table); else,
        or on the CPU, torch's copy
    """
    K, V = out.size(1), table.size(1)
    assert out.size(0) == V and (rows is None or rows.numel() == K), (tuple(out.shape), tuple(table.shape))
    if K == 0 or V == 0:
        return out
    if not (table.is_cuda and table.is_contiguous() and out.stride(1) == 1):
        return out.copy_((table[first:first + K] if rows is None else table[rows]).t())
    cfg = TILES
    step = GRID_ROWS * cfg["TR"]
    for k0 in range(0, K, step):
        n = min(step, K - k0)
        part = out[:, k0:k0 + n]
        idx = table if rows is None else rows[k0:k0 + n]
        _transposed_rows_kernel[(host_cdiv(V, cfg["TV"]), host_cdiv(n, cfg["TR"]))](
            table, idx, part, first + k0, n, V, out.stride(0), GATHER = rows is not None, TR = cfg["TR"],
            TV = cfg["TV"], num_warps = cfg["warps"])
    return out
