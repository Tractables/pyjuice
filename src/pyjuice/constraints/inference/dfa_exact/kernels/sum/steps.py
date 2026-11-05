"""
What the fused sum kernels of both directions are made of (:mod:`.fused_forward`, :mod:`.fused_top_down`): the steps of the online
log-sum-exp, a sample's columns, a launch's gates.
"""

from typing import Optional
import torch
import triton
import triton.language as tl

from pyjuice.utils.util import host_next_power_of_2
from ...external_params.block_scale import _row_gates, _segment_cell
from ..common import _tf32_operand
from ..compact import _lane_sample
from .tile_rules import _SPLIT_TILES


@triton.jit
def _lse_dot_step(m, acc, x, w, PRECISION: tl.constexpr):
    """One chunk of children of the online log-sum-exp: ``x`` [children, columns] in log space, ``w`` [nodes,
    children] the weights (the dot's operands rounded for ``PRECISION``, :func:`~.common._tf32_operand`)."""
    m_new = tl.maximum(m, tl.max(x, axis = 0))
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new)
    return m_new, acc * scale[None, :] + tl.dot(_tf32_operand(w, PRECISION),
                                                 _tf32_operand(tl.exp(x - shift[None, :]), PRECISION),
                                                 input_precision = PRECISION)


@triton.jit
def _lse_dot_step_biased(m, acc, x, w, bias, PRECISION: tl.constexpr):
    """:func:`_lse_dot_step` of ``x + bias[None, :]`` (``bias`` [columns], a chunk's gate per column, see
    :mod:`..external_params.block_scale`), the bias taken into the shift rather than added to
    every element."""
    m_new = tl.maximum(m, tl.max(x, axis = 0) + bias)
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new - bias)
    return m_new, acc * scale[None, :] + tl.dot(_tf32_operand(w, PRECISION),
                                                 _tf32_operand(tl.exp(x - shift[None, :]), PRECISION),
                                                 input_precision = PRECISION)


@triton.jit
def _lse_dot_step_rows(m, acc, x, w, gates, gidx, offs, emask, samp, gstride, goff, mmask, cmask,
                       PRECISION: tl.constexpr, TILE_E: tl.constexpr, SEG: tl.constexpr, SHARED: tl.constexpr):
    """One chunk of edges of the online log-sum-exp when the tile's rows read several node gates (node gates finer than
    the tile, see :mod:`..external_params.block_scale`; row ``r``'s cells are ``goff[r]`` from the block's first
    row's): the running maximum ``m`` per row and column, from -FLT_MAX (nothing: no rescale, no NaN). Every segment of
    ``SEG`` edges (one gate cell; the whole chunk when it reads one) is shifted by its own maximum per column -- an
    entry is lost only far below one of the same gate, as in :func:`_lse_dot_step` -- and is a product of its own,
    scaled per row and column by its gate: of the running sum's and the segment's factors one is 1, the other
    ``exp(-|t - m|)`` (one exp per row and column)."""
    NSEG: tl.constexpr = TILE_E // SEG
    TILE_B: tl.constexpr = x.shape[1]
    if NSEG == 1:
        xm = tl.max(x, axis = 0)                                                     # -inf: nothing
        e = tl.exp(x - tl.where(xm == float("-inf"), 0.0, xm)[None, :])
    else:
        xm3 = tl.max(tl.reshape(x, [NSEG, SEG, TILE_B]), axis = 1)                     # [NSEG, TILE_B]
        sh3 = tl.where(xm3 == float("-inf"), 0.0, xm3)
        e = tl.exp(x - tl.reshape(tl.broadcast_to(sh3[:, None, :], [NSEG, SEG, TILE_B]), [TILE_E, TILE_B]))
    e = _tf32_operand(e, PRECISION)                                                  # (once for every segment)
    w = _tf32_operand(w, PRECISION)
    for s in tl.static_range(NSEG):
        if NSEG == 1:
            seg = emask
            ws = w
        else:
            seg = emask & (tl.arange(0, TILE_E) // SEG == s)
            ws = tl.where(seg[None, :], w, 0.0)
            xm = tl.max(tl.where((tl.arange(0, NSEG) == s)[:, None], xm3, float("-inf")), axis = 0)  # segment s's
        g = _row_gates(gates, _segment_cell(gidx, offs, seg), goff, samp, gstride, mmask, cmask, SHARED)
        part = tl.dot(ws, e, input_precision = PRECISION)
        t = xm[None, :] + g                                                          # the segment's log-bound
        up = t > m
        r = tl.exp(tl.where(up, m - t, t - m))                                       # -inf - finite: 0
        acc = acc * tl.where(up, r, 1.0) + part * tl.where(up, 1.0, r)
        m = tl.maximum(m, t)
    return m, acc


@triton.jit
def _lse_dot_step_pre(m, acc, e, t, w, PRECISION: tl.constexpr):
    """:func:`_lse_dot_step` of a chunk whose shifted exponentials ``e`` [children, columns] and log-bound ``t``
    [columns] (the chunk's maximum per column, plus a gate) were computed beforehand, ``e`` rounded for ``PRECISION``
    (:func:`_forward_prepass_kernel`, :func:`_top_down_prepass_kernel`): the running maximum and the rescale on
    columns only, then one dot."""
    m_new = tl.maximum(m, t)
    live = m_new != float("-inf")
    scale = tl.where(live, tl.exp(m - m_new), 1.0)
    f = tl.where(live, tl.exp(t - m_new), 0.0)
    part = tl.dot(_tf32_operand(w, PRECISION), e, input_precision = PRECISION)
    return m_new, acc * scale[None, :] + part * f[None, :]


@triton.jit
def _pre_chunk(pre_buf, pre_cm, pre_ld, p, offs_i, c, im, k, cmask, BS: tl.constexpr, NCH: tl.constexpr):
    """``(e, t)``: parent entry ``p``'s chunk ``c`` (rows ``offs_i``) from the pre-pass, at column positions ``k``."""
    p64 = p.to(tl.int64)
    e = tl.load(pre_buf + (p64 * BS + offs_i)[:, None] * pre_ld + k[None, :], mask = im[:, None] & cmask[None, :],
                other = 0.0)
    t = tl.load(pre_buf + pre_cm + (p64 * NCH + c) * pre_ld + k, mask = cmask, other = float("-inf"))
    return e, t


@triton.jit
def _split_merge(out, buf, tile, s, TILE_R: tl.constexpr, TILE_B: tl.constexpr, SPLIT: tl.constexpr):
    """``(merged, last)``: a program's share ``s`` of a reduction split ``SPLIT`` ways (``out`` [rows, columns], its
    log-sum-exp) stored in the split buffer ``buf`` (:func:`split_buffer`) at output tile ``tile``, which the tile's
    counter counts; the last share in (``last``) reads every share back in order -- the merge does not depend on
    which is last -- and sets the counter back to 0 for the next launch."""
    size: tl.constexpr = TILE_R * TILE_B
    offs = tl.arange(0, TILE_R)[:, None] * TILE_B + tl.arange(0, TILE_B)[None, :]
    count = buf.to(tl.pointer_type(tl.int32)) + tile
    base = buf + _SPLIT_TILES + tile.to(tl.int64) * (SPLIT * size)
    tl.store(base + s * size + offs, out)
    tl.debug_barrier()                                    # every thread's share stored before the count releases it
    last = tl.atomic_add(count, 1, sem = "acq_rel") == SPLIT - 1
    m = tl.full([TILE_R, TILE_B], float("-inf"), dtype = tl.float32)
    for j in tl.static_range(SPLIT):
        m = tl.maximum(m, tl.load(base + j * size + offs, mask = last, other = float("-inf"), cache_modifier = ".cg"))
    sh = tl.where(m == float("-inf"), 0.0, m)
    acc = tl.zeros([TILE_R, TILE_B], dtype = tl.float32)
    for j in tl.static_range(SPLIT):
        acc += tl.exp(tl.load(base + j * size + offs, mask = last, other = float("-inf"), cache_modifier = ".cg") - sh)
    tl.store(count, 0, mask = last)
    return tl.log(acc) + sh, last


@triton.jit
def _column_sample(cols, reg, reg_slots, coff, reg_ivk, Bq, steps, LANES: tl.constexpr):
    """The sample of every column ``cols`` of a row of sum region ``reg`` (a gated layer's): its block of the region's
    slots, or ``LANES`` (compact rows, see :mod:`.compact`), the sample whose lanes hold it."""
    if LANES:
        return _lane_sample(coff, tl.load(reg_ivk + reg), cols, Bq, steps).to(tl.int32)
    else:
        return (cols // tl.load(reg_slots + reg)).to(tl.int32)


def _gate_tile(tile: int, uniform: dict) -> int:
    """A gated launch's chunk of edges (tile of children): ``tile``, or the widest narrower one, down to 16, whose
    every chunk reads one gate (``uniform``, see ``block_scale.uniform_chunks`` / ``uniform_tiles``), if there is."""
    for t in (tile, tile // 2, tile // 4, tile // 8):
        if t >= 16 and uniform.get(t, False):
            return t
    return tile


def _gate_segment(tile: int, uniform: dict) -> int:
    """The segments of a chunk of ``tile`` edges that read one gate cell each: the widest aligned run of edges, a power
    of two, whose every one does (``uniform``, see ``block_scale.uniform_chunks``)."""
    t = tile
    while t > 1 and not uniform.get(t, False):
        t //= 2
    return t


def _node_gates_per_tile(block_size: int, node_gate: int, tile: int) -> int:
    """The node gates a tile of ``tile`` rows of a node block spans at most (1: every tile lies in one)."""
    if node_gate >= block_size:
        return 1
    return max((min(block_size, t + tile) - 1) // node_gate - t // node_gate + 1 for t in range(0, block_size, tile))


def _row_tile(tile: int, block_size: int, node_gate: int) -> int:
    """A gated launch's tile of node rows when node gates are finer than the block: within one node gate where the
    products can be (16 rows or more) -- a tile of several reads a gate per row and keeps a maximum per row."""
    return min(tile, max(16, host_next_power_of_2(node_gate)))


def gate_launch(gated: Optional[dict], dummy: torch.Tensor, idx = None, a_idx = None, ck = None) -> tuple:
    """The sum kernels' gate arguments, ``(idx, a_idx, ck, gates, gstride, lz, lzstride, coff, reg_ivk, Bq, steps)``,
    from a part's gates (see :func:`~..queries.backend.forward.part_gates`; ``dummy`` where nothing is read)."""
    if gated is None:
        return (dummy,) * 3 + (dummy, 0, dummy, 0, dummy, dummy, 1, 1)
    off, reg_ivk, Bq, steps = gated["lanes"] if gated["lanes"] is not None else (dummy, dummy, 1, 1)
    return (idx if idx is not None else dummy, a_idx if a_idx is not None else dummy, ck if ck is not None else dummy,
            gated["values"], gated["gstride"], gated["lz"], gated["lzstride"], off, reg_ivk, Bq, steps)
