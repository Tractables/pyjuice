"""
What the kernels share: the operand kinds of a step, CUDA's grid limits, the int32 index lists through which a launch
runs a subset of its steps (16-byte aligned, :func:`pack_aligned`), and the Triton device functions several kernels
use -- a block's address, an interval's tables, slots and column pairs, log-space accumulation.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import triton
import triton.language as tl


#: operand kinds: a sum node's block in ``node_mars``, an intermediate block in scratch, the identity
SUM, TEMP, IDENTITY = 0, 1, 2

#: CUDA's limit on the programs along a grid's first axis, and along each of the others
MAX_GRID = (2 ** 31 - 1, 65535)


def grid_chunks(count: int, axis: int):
    """``(first, size)`` of every launch that ``count`` programs along grid ``axis`` take: one launch, unless the
    count passes CUDA's limit there (each kernel adds ``first`` to its program id on that axis)."""
    limit = MAX_GRID[axis]
    return [(first, min(limit, count - first)) for first in range(0, count, limit)]


def pack_aligned(arrays: Sequence[np.ndarray], dtype) -> Tuple[np.ndarray, np.ndarray]:
    """``(buffer, starts)``: ``arrays`` concatenated as ``dtype``, each starting 16 bytes into the buffer: Triton
    specializes a pointer argument on that alignment, so a view at another offset would compile and run a slower
    variant of its kernel."""
    step = 16 // np.dtype(dtype).itemsize
    sizes = np.fromiter((len(a) for a in arrays), dtype = np.int64, count = len(arrays))
    starts = np.zeros(len(arrays) + 1, dtype = np.int64)
    np.cumsum(-(-sizes // step) * step, out = starts[1:])
    buf = np.zeros(int(starts[-1]), dtype = dtype)
    for a, s, k in zip(arrays, starts.tolist(), sizes.tolist()):
        buf[s:s + k] = a
    return buf, starts


class IndexList:
    """An int32 index list of a query plan: ``host`` (numpy) until :meth:`IndexLists.upload` copies it to the device
    (``tensor``) and drops it -- a cached plan holds the device copy alone (the host one may view a planner's whole
    output)."""

    __slots__ = ("host", "tensor")

    def __init__(self, host):
        self.host, self.tensor = host, None


def tensor(item: Optional[IndexList]):
    """An index list's device tensor, or None (every row)."""
    return None if item is None else item.tensor


@triton.jit
def _child(idx, c, INDEXED: tl.constexpr):
    """The row of a table that program ``c`` owns: the ``c``-th of a query plan's index list, or ``c``."""
    if INDEXED:
        return tl.load(idx + c)
    else:
        return c


@triton.jit
def _block_ptr(row, reg, node_mars, prior_mars, temp, reg_offset, reg_width, reg_first, prior_offset, prior_width,
               reg_prior, temp_width, KIND: tl.constexpr):
    """Start of a block operand, and whether every sample has a block of its own there (sample ``b``'s then ``b *
    slots`` further): a sum node's row -- in its region of ``node_mars``, or of ``prior_mars`` when the query plan
    keeps the region's values there (``reg_prior``, see :mod:`.plans`), its one block read by every sample -- or
    a scratch row."""
    if KIND == 0:
        k = row - tl.load(reg_first + reg)
        shared = tl.load(reg_prior + reg) != 0
        own = node_mars + tl.load(reg_offset + reg) + k * tl.load(reg_width + reg)
        prior = prior_mars + tl.load(prior_offset + reg) + k * tl.load(prior_width + reg)
        return tl.where(shared, prior, own), ~shared
    else:
        return temp + row.to(tl.int64) * temp_width, row >= 0


@triton.jit
def _exit_width(width, t2, n):
    """Columns at the exit boundary ``t2``: a sequence end is one column (summed over the accepting states)."""
    return tl.where(t2 == n, 1, tl.load(width + tl.minimum(t2, n)))


@triton.jit
def _interval(iv_of, iv_info, a, b, n):
    """``(slots, positions, pairs)`` of the interval ``[a, b)``: its blocks' slots per sample, and where its
    position and pair tables start (-1 when it keeps every pair; see :class:`~.layout.IntervalTables`)."""
    k = tl.load(iv_of + a * (n + 1) + b).to(tl.int64) * 3
    return tl.load(iv_info + k).to(tl.int32), tl.load(iv_info + k + 1), tl.load(iv_info + k + 2)  # 32-bit index math


@triton.jit
def _slot(iv_table, positions, r, c, ncols, mask, FULL: tl.constexpr):
    """The slot of pair ``(r, c)`` in a block with ``ncols`` exit columns: negative where the automaton does not join
    them, or ``r`` is negative. ``FULL``: the block's interval keeps every pair (known when the kernel is compiled);
    else the interval's tables say (``positions`` -1: it keeps every pair)."""
    if FULL:
        return r * ncols + c
    else:
        return tl.where(positions < 0, r * ncols + c,
                        tl.load(iv_table + positions + r * ncols + c, mask = mask & (r >= 0) & (positions >= 0),
                                other = -1))


@triton.jit
def _pair(iv_table, pairs, p, mask, FULL: tl.constexpr):
    """``i * X + j`` of the pair ``(i, j)`` in slot ``p`` (``FULL`` as for :func:`_slot`)."""
    if FULL:
        return p
    else:
        return tl.where(pairs < 0, p, tl.load(iv_table + pairs + p, mask = mask & (pairs >= 0), other = 0))


@triton.jit
def _lse_step(m, acc, v):
    """One step of an online log-sum-exp (``m``: running max, ``acc``: running sum scaled by ``exp(-m)``)."""
    m_new = tl.maximum(m, v)
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new)
    return m_new, acc * scale + tl.exp(v - shift)


# -------------------------------------------------------------------------------------------------
# The top-down pass: a sum's children from the sum
# -------------------------------------------------------------------------------------------------

@triton.jit
def _logaddexp_tile(a, b):
    hi = tl.maximum(a, b)
    s = tl.where(hi == float("-inf"), 0.0, hi)
    return tl.where(hi == float("-inf"), float("-inf"), tl.log(tl.exp(a - s) + tl.exp(b - s)) + s)
