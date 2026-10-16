"""
What the kernels share: the operand kinds of a step, CUDA's grid limits, the int32 index lists through which a launch
runs a subset of its steps (16-byte aligned, :func:`pack_aligned`), the node tiles of the walk kernels
(:func:`node_tiles`), and the Triton device functions several kernels use -- a block's address, an interval's tables,
slots and column pairs, log-space accumulation, a TF32 dot's operands.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import triton
import triton.language as tl
from numba import njit


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
    output). ``height``: for a list of node tiles (:func:`node_tiles`), their kernels' tile height, else None."""

    __slots__ = ("host", "tensor", "height")

    def __init__(self, host, height: Optional[int] = None):
        self.host, self.tensor, self.height = host, None, height


def tensor(item: Optional[IndexList]):
    """An index list's device tensor, or None (every row)."""
    return None if item is None else item.tensor


#: the rows of a node tile (:func:`node_tiles`): the most children (or destinations) of a walk kernel's table that one
#: program computes together, walking their shared indices once
NODE_TILE = 16
_NODE_TILE = tl.constexpr(NODE_TILE)


def node_groups(*keys: np.ndarray) -> np.ndarray:
    """Every row's group (int64 [rows]): the rows with equal ``keys`` (int arrays, a value per row each, or [rows, k])
    share one."""
    cols = np.concatenate([np.asarray(k, dtype = np.int64).reshape(len(keys[0]), -1) for k in keys], axis = 1)
    return np.unique(cols, axis = 0, return_inverse = True)[1].reshape(-1)


@njit(cache = True)
def _node_tiles(group, pos, tile):
    """:func:`node_tiles`' tiles (flat) and their most rows: a counting sort of ``pos`` by group."""
    groups = 0
    for p in pos:
        groups = max(groups, group[p] + 1)
    count = np.zeros(groups, dtype = np.int64)
    for p in pos:
        count[group[p]] += 1
    first = np.zeros(groups + 1, dtype = np.int64)                 # every group's first tile
    for g in range(groups):
        first[g + 1] = first[g] + (count[g] + tile - 1) // tile
    out = np.full(first[groups] * tile, -1, dtype = np.int32)
    fill = np.zeros(groups, dtype = np.int64)
    most = 1
    for p in pos:
        g = group[p]
        r = fill[g]
        out[(first[g] + r // tile) * tile + r % tile] = p
        fill[g] = r + 1
        most = max(most, min(r + 1, tile))
    return out, most


def node_tiles(group: np.ndarray, sel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """The rows ``sel`` (all if None) of a walk kernel's table in tiles of :data:`NODE_TILE`, each of one ``group``
    (:func:`node_groups`; groups in order, a group's rows in ``sel``'s): int32 [tiles * NODE_TILE], -1 past a tile's
    last row, and the kernels' tile height -- the most rows of a tile, 1, 4 or NODE_TILE (few variants to compile)."""
    pos = np.arange(group.size, dtype = np.int64) if sel is None else np.asarray(sel, dtype = np.int64)
    tiles, most = _node_tiles(np.asarray(group, dtype = np.int64), pos, NODE_TILE)
    return tiles, 1 if most == 1 else 4 if most <= 4 else NODE_TILE


@triton.jit
def _tile_rows(tiles, pid, TN: tl.constexpr, SUBS: tl.constexpr):
    """The rows of a walk kernel's table that program ``pid`` computes, from its node tile (:func:`node_tiles`; the
    tile's ``SUBS`` programs ``TN`` rows each): ``(rows, real)`` -- a padding row repeats the tile's first (every
    index read stays valid; its values are neither read nor written) -- and the tile's first row."""
    tile = pid // SUBS
    first = tl.load(tiles + tile * _NODE_TILE)
    rows = tl.load(tiles + tile * _NODE_TILE + (pid % SUBS) * TN + tl.arange(0, TN))
    real = rows >= 0
    return tl.where(real, rows, first), real, first


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


@triton.jit
def _tf32_operand(x, PRECISION: tl.constexpr):
    """A dot's operand ``x`` as a dot of ``PRECISION`` should read it: under "tf32" an fp32 ``x`` rounded to the
    nearest TF32 value (ties away from zero, as ``cvt.rna.tf32.f32``; finite ``x``). A Triton TF32 dot hands the
    tensor cores the fp32 bits, of which they drop the low 13 -- every operand truncated, every product biased low
    (by 6.5e-4 relative on uniform operands), a bias that compounds through a circuit's layers. Otherwise ``x``
    unchanged (other precisions; narrower types are exact in TF32)."""
    if PRECISION == "tf32":
        if x.dtype == tl.float32:
            x = ((x.to(tl.int32, bitcast = True) + 0x1000) & -8192).to(tl.float32, bitcast = True)
    return x


# -------------------------------------------------------------------------------------------------
# The top-down pass: a sum's children from the sum
# -------------------------------------------------------------------------------------------------

@triton.jit
def _logaddexp_tile(a, b):
    hi = tl.maximum(a, b)
    s = tl.where(hi == float("-inf"), 0.0, hi)
    return tl.where(hi == float("-inf"), float("-inf"), tl.log(tl.exp(a - s) + tl.exp(b - s)) + s)
