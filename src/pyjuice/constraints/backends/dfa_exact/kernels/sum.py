"""
The lifted sum layer. A sum node and its children (products over the same scope) keep blocks of the same
shape, so a sum acts on every column independently: it is pyjuice's sum over ``B * slots`` columns, where
``slots`` is the block size of the node group's scope. The edges are read from the sum layer's own partitions
(``partitioned_nids / cids / pids``): node ``i`` of the node block at ``nids[k]`` adds child element
``cids[k, e]`` with weight ``params[pids[k, e] + i]``.

Under evidence pruning (:mod:`.live`) only the live columns are computed, and the dead ones set to -inf. Two
paths:

* DENSE node blocks -- children a contiguous run of element rows, weights strided by the block size (an HMM's
  transitions) -- are a matrix product with weights read in place from the parameter buffer: a node block's
  weights are the ``[E, BS]`` matrix at ``params[pids[k, 0]]``. The children are shifted by their column
  maximum and exponentiated once (one pass), multiplied by cuBLAS, and the log taken and the maximum added back
  (one pass over the product, in place). In fp32 this is cuBLAS's exact fp32
  product (measured faster than pyjuice's own sum layer at every batch size on a 4096-state HMM, with ~100x
  smaller error). Node blocks that read the same children from consecutive rows can instead multiply their
  weights stacked into one ``[K * BS, E]`` matrix (:func:`stacked_weights`, a copy) in one product;
  :mod:`.forward` decides when the copy pays for itself;
* every other node block runs the fused Triton kernel: an online log-sum-exp over chunks of children, with
  the products on tensor cores. Padded edges point at pyjuice's dummy element rows, which lie below the
  product layer group's region, and are skipped. A gated sum layer's node blocks
  (:mod:`..external_params.block_scale`) run it with their gates (``GATED``, that module's kernel
  helpers), every one of them: the dense path has no per-sample weights.
"""

import contextlib
import functools
import hashlib
from typing import Optional

import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ..external_params.block_scale import (_add_gated_parent, _chunk_gates, _edge_gates, _node_log_z, _row_gates,
                                           _segment_cell, _tile_gates)
from .common import _child, _logaddexp_tile, _tf32_operand, grid_chunks
from .compact import _lane_sample


#: The fused kernel's tiles (TILE_M, TILE_E, TILE_B, num_warps, num_stages) per product precision, from a sweep on
#: a 4096-state HMM layer (4 x 1024 nodes, 4096 children) at 69 to 8832 columns.
_TILES = {"tf32": (128, 32, 128, 8, 3), "tf32x3": (64, 32, 64, 4, 3), "bf16x3": (64, 32, 64, 4, 3)}

#: Dense node blocks smaller than this run the fused kernel: one cuBLAS call per node block only pays for itself
#: on large blocks (a partition of many small blocks is one fused launch instead).
DENSE_MIN_BLOCK = 64


def _tile_warps(warps: int, rows: int) -> int:
    """A fused launch's warps for output tiles of ``rows`` rows: the table's (:data:`_TILES`, tuned on tiles of 128
    rows), at most 4 for tiles of 32 rows or fewer (node blocks of 32: at 8 warps a launch took up to 2.4x as long as
    at 4, with the same results; tiles of 128 rows and few columns lost 7% at 4)."""
    return warps if rows > 32 else min(warps, 4)


#: A fused launch of fewer programs than this many per SM splits every program's reduction (over its parents or
#: children) in shares, each a program of its own (:func:`_split_merge`), as long as a share keeps
#: ``SPLIT_MIN_SHARE`` chunks of it (dots); a power of two, at most ``SPLIT_MAX``. A launch of few, long programs (an
#: HMM's layer of 1024 nodes in blocks of 32 at small batch: one program per tile of 32 nodes, 32 chunks of children
#: each) leaves most of the GPU idle. (2 and 2 before: shares of 2 chunks merged at a cost above their gain -- an LL1024
#: top-down launch took 3.6x as long; 4 and 8 ran HMM, PD and Codpula-shaped queries as fast or faster.)
SPLIT_PROGRAMS_PER_SM = 4
SPLIT_MIN_SHARE = 8
SPLIT_MAX = 16


def split_count(programs: int, chunks: int, dev) -> int:
    """Into how many shares a fused launch of ``programs`` programs, each reducing over ``chunks`` chunks (the most
    any has), splits every program's reduction: 1 (none) when the launch fills the GPU."""
    want = host_cdiv(SPLIT_PROGRAMS_PER_SM * _num_sms(dev), max(1, programs))
    split = 1
    while split * 2 <= min(want, SPLIT_MAX, chunks // SPLIT_MIN_SHARE):
        split *= 2
    return split


#: The most output tiles a split launch has (a launch of more does not split): their counters lead the split buffer.
SPLIT_TILES = 4096
_SPLIT_TILES = tl.constexpr(SPLIT_TILES)

_SPLIT_BUFFERS = {}
_NO_SPLIT = {}

#: A fused top-down launch whose parents are read by at least this many tiles of children gets a pre-pass
#: (:func:`_top_down_prepass_kernel`): every parent chunk's values less their log Z, their maximum per column and
#: their shifted exponentials, once instead of once per tile of children.
PRE_MIN_TILES = 4

_PRE_BUFFERS = {}


def pre_buffer(dev, size: int) -> torch.Tensor:
    """The pre-pass's buffer on ``dev``, of ``size`` floats at least, kept per device and grown as needed."""
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _PRE_BUFFERS.get(key)
    if buf is None or buf.numel() < size:
        buf = torch.empty(max(size, 1 << 20), dtype = torch.float32, device = dev)
        _PRE_BUFFERS[key] = buf
    return buf


def split_buffer(dev, tiles: int, per_tile: int) -> torch.Tensor:
    """The split launches' buffer on ``dev`` (float32, zeros when made): ``SPLIT_TILES`` counters (int32 bits, 0
    between launches: the last share in resets its tile's), then ``per_tile`` floats for each of ``tiles`` output
    tiles; kept per device, grown as needed."""
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _SPLIT_BUFFERS.get(key)
    if buf is None or buf.numel() < SPLIT_TILES + tiles * per_tile:
        buf = torch.zeros(SPLIT_TILES + max(tiles * per_tile, 1 << 20), dtype = torch.float32, device = dev)
        _SPLIT_BUFFERS[key] = buf
    return buf


@functools.lru_cache(maxsize = None)
def _num_sms_of(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _num_sms(dev) -> int:
    return _num_sms_of(dev.index if dev.index is not None else torch.cuda.current_device())


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


@triton.jit
def _lifted_sum_body(node_mars, element_mars, weights, nids, cids, pids, nb_reg, sel,
                       reg_offset, reg_width, reg_first, reg_slots, a_rows, a_regs, a_pids, perm, reg_perm, reg_live,
                       gidx, a_gidx, gck, gates, gstride, lz, lzstride, coff, reg_ivk, Bq, steps,
                       split_buf, pre_buf, pre_cm, pre_ld, elem_first, elem_width, B, num_edges, num_alias,
                       num_mtiles, pid0,
                       BS: tl.constexpr, TILE_M: tl.constexpr, TILE_E: tl.constexpr, TILE_A: tl.constexpr,
                       TILE_B: tl.constexpr, PRECISION: tl.constexpr, ALIAS: tl.constexpr, PRUNE: tl.constexpr,
                       INDEXED: tl.constexpr, GATED: tl.constexpr, LANES: tl.constexpr, CHUNK_GATE: tl.constexpr,
                       A_CHUNK_GATE: tl.constexpr, SHARED_GATES: tl.constexpr, NG: tl.constexpr, NGT: tl.constexpr,
                       SEG: tl.constexpr, A_SEG: tl.constexpr, SPLIT: tl.constexpr, PRE: tl.constexpr):
    """One program per (node block -- of the plan's index list ``sel`` if ``INDEXED`` --, tile of its nodes, tile of
    columns). Pruned (``PRUNE``), the program's columns are positions in the region's interval's order of columns
    (live ones first, see :mod:`.live`): a tile past the live ones is all -inf, without reading anything. ``GATED``
    (see :mod:`..external_params.block_scale`): every edge's child values plus its gate of the
    column's sample (``gidx`` / ``a_gidx`` the edges' cells in ``gates``, ``gstride`` floats per sample, 0 for every
    sample's), and every node's ``log Z`` of the sample (``lz``, by node row, ``lzstride`` per sample) taken off;
    ``CHUNK_GATE`` / ``A_CHUNK_GATE``: every chunk of edges (aliased children) reads one cell; ``SHARED_GATES``: the
    gates are every sample's; ``NG``: the node gate size, below ``BS`` a node block's rows read several -- node gate
    ``m // NG`` of row ``m`` is ``gck`` (its node group's grid row length) cells per gate from the first row's: a tile
    of rows within one (``NGT`` 1) reads its cells, else its rows read their own, every chunk in segments of ``SEG``
    (``A_SEG``) edges of one cell each (:func:`_lse_dot_step_rows`). ``SPLIT``: every (node block, tile of its nodes,
    tile of columns) has as many programs, each over its share of the chunks of children, the aliased ones with the
    first (:func:`_split_merge`, ``split_buf``)."""
    pid_nm = tl.program_id(0)                                     # (node block, tile of its nodes, share)
    pid_c = pid0 + tl.program_id(1)                               # tile of columns
    s = pid_nm % SPLIT                                            # the program's share of the children
    pid_t = pid_nm // SPLIT
    pid_n = _child(sel, pid_t // num_mtiles, INDEXED)
    mt = pid_t % num_mtiles
    reg = tl.load(nb_reg + pid_n)
    ncols = B * tl.load(reg_slots + reg)
    col0 = pid_c * TILE_B
    if col0 < ncols:
        k = col0 + tl.arange(0, TILE_B)
        kmask = k < ncols
        n_edges, n_alias = num_edges, num_alias
        if PRUNE:
            num_live = tl.load(reg_live + reg)
            cols = tl.load(perm + tl.load(reg_perm + reg) + k, mask = kmask, other = 0)
            cmask = kmask & (k < num_live)
            n_edges = tl.where(col0 < num_live, num_edges, 0)                # a dead tile reads nothing
            n_alias = tl.where(col0 < num_live, num_alias, 0)
        else:
            cols = k
            cmask = kmask
        samp = cols
        if GATED and not SHARED_GATES:
            samp = _column_sample(cols, reg, reg_slots, coff, reg_ivk, Bq, steps, LANES)
        offs_m = mt * TILE_M + tl.arange(0, TILE_M)
        mmask = offs_m < BS
        m = tl.full([TILE_B], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE_M, TILE_B], dtype = tl.float32)
        if GATED and NG < BS:                                              # node gates finer than the block
            if NGT == 1:                                                   # the tile's one: its cells
                gates += (mt * TILE_M) // NG * tl.load(gck + pid_n)
            else:                                                          # several: per row
                m2 = tl.full([TILE_M, TILE_B], -3.4028234663852886e38, dtype = tl.float32)
                goff = (offs_m // NG) * tl.load(gck + pid_n)
        e_lo, e_hi = 0, n_edges
        if SPLIT > 1:                                     # whole chunks: a share, aliased children with the first
            share = ((num_edges + TILE_E - 1) // TILE_E + SPLIT - 1) // SPLIT * TILE_E
            e_lo = s * share
            e_hi = tl.minimum(n_edges, e_lo + share)
            n_alias = tl.where(s == 0, n_alias, 0)
        for e0 in range(e_lo, e_hi, TILE_E):
            offs_e = e0 + tl.arange(0, TILE_E)
            emask = offs_e < num_edges
            cid = tl.load(cids + pid_n * num_edges + offs_e, mask = emask, other = 0).to(tl.int64)
            pid = tl.load(pids + pid_n * num_edges + offs_e, mask = emask, other = 0).to(tl.int64)
            valid = emask & (cid >= elem_first)
            w = tl.load(weights + pid[None, :] + offs_m[:, None],
                        mask = mmask[:, None] & valid[None, :], other = 0.0)                    # [TILE_M, TILE_E]
            if PRE:                                            # the pre-pass's chunk (node blocks share their children)
                e = tl.load(pre_buf + offs_e.to(tl.int64)[:, None] * pre_ld + k[None, :],
                            mask = valid[:, None] & cmask[None, :], other = 0.0)
                t = tl.load(pre_buf + pre_cm + (e0 // TILE_E) * pre_ld + k, mask = cmask, other = float("-inf"))
                if GATED:
                    t += _chunk_gates(gates, gidx + pid_n * num_edges, offs_e, emask, samp, gstride, cmask,
                                      SHARED_GATES)
                m, acc = _lse_dot_step_pre(m, acc, e, t, w, PRECISION)
            else:
                x = tl.load(element_mars + (cid - elem_first)[:, None] * elem_width + cols[None, :],
                            mask = valid[:, None] & cmask[None, :], other = float("-inf"))       # [TILE_E, TILE_B]
                if GATED and not CHUNK_GATE and NGT == 1:
                    x += _edge_gates(gates, gidx + pid_n * num_edges, offs_e, emask, samp, gstride, cmask,
                                     SHARED_GATES)
                if GATED and NGT > 1:                                      # gates per row and column
                    m2, acc = _lse_dot_step_rows(m2, acc, x, w, gates, gidx + pid_n * num_edges, offs_e, emask, samp,
                                                 gstride, goff, mmask, cmask, PRECISION, TILE_E, SEG, SHARED_GATES)
                elif GATED and CHUNK_GATE:                                 # one gate per column
                    m, acc = _lse_dot_step_biased(m, acc, x, w, _chunk_gates(gates, gidx + pid_n * num_edges, offs_e,
                                                                             emask, samp, gstride, cmask,
                                                                             SHARED_GATES), PRECISION)
                else:
                    m, acc = _lse_dot_step(m, acc, x, w, PRECISION)
        if ALIAS:
            # children that copy a sum: read in the sum's own row (row a_rows within region a_regs)
            for a0 in range(0, n_alias, TILE_A):
                offs_a = a0 + tl.arange(0, TILE_A)
                amask = offs_a < num_alias
                row = tl.load(a_rows + pid_n * num_alias + offs_a, mask = amask, other = -1)
                areg = tl.load(a_regs + pid_n * num_alias + offs_a, mask = amask, other = 0)
                pid = tl.load(a_pids + pid_n * num_alias + offs_a, mask = amask, other = 0).to(tl.int64)
                valid = amask & (row >= 0)
                base = tl.load(reg_offset + areg, mask = valid, other = 0) + \
                    row.to(tl.int64) * tl.load(reg_width + areg, mask = valid, other = 0)
                base = tl.multiple_of(base, 16)                       # regions start and rows are padded to 16 floats
                x = tl.load(node_mars + base[:, None] + cols[None, :], mask = valid[:, None] & cmask[None, :],
                            other = float("-inf"))                                             # [TILE_A, TILE_B]
                if GATED and not A_CHUNK_GATE and NGT == 1:
                    x += _edge_gates(gates, a_gidx + pid_n * num_alias, offs_a, amask, samp, gstride, cmask,
                                     SHARED_GATES)
                w = tl.load(weights + pid[None, :] + offs_m[:, None],
                            mask = mmask[:, None] & valid[None, :], other = 0.0)                # [TILE_M, TILE_A]
                if GATED and NGT > 1:
                    m2, acc = _lse_dot_step_rows(m2, acc, x, w, gates, a_gidx + pid_n * num_alias, offs_a, amask, samp,
                                                 gstride, goff, mmask, cmask, PRECISION, TILE_A, A_SEG, SHARED_GATES)
                elif GATED and A_CHUNK_GATE:
                    m, acc = _lse_dot_step_biased(m, acc, x, w, _chunk_gates(gates, a_gidx + pid_n * num_alias, offs_a,
                                                                             amask, samp, gstride, cmask, SHARED_GATES),
                                                  PRECISION)
                else:
                    m, acc = _lse_dot_step(m, acc, x, w, PRECISION)
        if GATED and NGT > 1:
            out = tl.log(acc) + m2                                         # nothing: log 0 = -inf
        else:
            out = tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)[None, :]
        mask = mmask[:, None] & kmask[None, :]
        if SPLIT > 1:                                     # a share of the children: the last one in merges them all
            out, last = _split_merge(out, split_buf, pid_t + pid_c * (tl.num_programs(0) // SPLIT), s, TILE_M,
                                     TILE_B, SPLIT)
            mask = mask & last
        nid = tl.load(nids + pid_n).to(tl.int64)
        if GATED:
            out -= _node_log_z(lz, samp, lzstride, nid + offs_m, mmask, kmask, SHARED_GATES)
        width = tl.load(reg_width + reg)
        base = tl.load(reg_offset + reg) + (nid - tl.load(reg_first + reg)) * width
        tl.store(node_mars + base + offs_m.to(tl.int64)[:, None] * width + cols[None, :], out, mask = mask)


@triton_jit
def _lifted_sum_kernel(node_mars, element_mars, weights, nids, cids, pids, nb_reg, sel, reg_offset, reg_width,
                       reg_first, reg_slots, a_rows, a_regs, a_pids, perm, reg_perm, reg_live, gidx, a_gidx, gck, gates,
                       gstride, lz, lzstride, coff, reg_ivk, Bq, steps, elem_first, elem_width, B, num_edges, num_alias,
                       num_mtiles, pid0, BS: tl.constexpr, TILE_M: tl.constexpr, TILE_E: tl.constexpr,
                       TILE_A: tl.constexpr, TILE_B: tl.constexpr, PRECISION: tl.constexpr, ALIAS: tl.constexpr,
                       PRUNE: tl.constexpr, INDEXED: tl.constexpr, GATED: tl.constexpr, LANES: tl.constexpr,
                       CHUNK_GATE: tl.constexpr, A_CHUNK_GATE: tl.constexpr, SHARED_GATES: tl.constexpr,
                       NG: tl.constexpr, NGT: tl.constexpr, SEG: tl.constexpr, A_SEG: tl.constexpr):
    """:func:`_lifted_sum_body`, unsplit, without a pre-pass."""
    _lifted_sum_body(node_mars = node_mars, element_mars = element_mars, weights = weights, nids = nids, cids = cids,
                     pids = pids, nb_reg = nb_reg, sel = sel, reg_offset = reg_offset, reg_width = reg_width,
                     reg_first = reg_first, reg_slots = reg_slots, a_rows = a_rows, a_regs = a_regs, a_pids = a_pids,
                     perm = perm, reg_perm = reg_perm, reg_live = reg_live, gidx = gidx, a_gidx = a_gidx, gck = gck,
                     gates = gates, gstride = gstride, lz = lz, lzstride = lzstride, coff = coff, reg_ivk = reg_ivk,
                     Bq = Bq, steps = steps, split_buf = node_mars, pre_buf = node_mars, pre_cm = 0, pre_ld = 0,
                     elem_first = elem_first, elem_width = elem_width, B = B, num_edges = num_edges,
                     num_alias = num_alias, num_mtiles = num_mtiles, pid0 = pid0, BS = BS, TILE_M = TILE_M,
                     TILE_E = TILE_E, TILE_A = TILE_A, TILE_B = TILE_B, PRECISION = PRECISION, ALIAS = ALIAS,
                     PRUNE = PRUNE, INDEXED = INDEXED, GATED = GATED, LANES = LANES, CHUNK_GATE = CHUNK_GATE,
                     A_CHUNK_GATE = A_CHUNK_GATE, SHARED_GATES = SHARED_GATES, NG = NG, NGT = NGT, SEG = SEG,
                     A_SEG = A_SEG, SPLIT = 1, PRE = False)


@triton_jit
def _lifted_sum_aux_kernel(node_mars, element_mars, weights, nids, cids, pids, nb_reg, sel, reg_offset, reg_width,
                           reg_first, reg_slots, a_rows, a_regs, a_pids, perm, reg_perm, reg_live, gidx, a_gidx, gck,
                           gates, gstride, lz, lzstride, coff, reg_ivk, Bq, steps, split_buf, pre_buf, pre_cm, pre_ld,
                           elem_first, elem_width, B, num_edges, num_alias, num_mtiles, pid0, BS: tl.constexpr,
                           TILE_M: tl.constexpr, TILE_E: tl.constexpr, TILE_A: tl.constexpr, TILE_B: tl.constexpr,
                           PRECISION: tl.constexpr, ALIAS: tl.constexpr, PRUNE: tl.constexpr, INDEXED: tl.constexpr,
                           GATED: tl.constexpr, LANES: tl.constexpr, CHUNK_GATE: tl.constexpr,
                           A_CHUNK_GATE: tl.constexpr, SHARED_GATES: tl.constexpr, NG: tl.constexpr, NGT: tl.constexpr,
                           SEG: tl.constexpr, A_SEG: tl.constexpr, SPLIT: tl.constexpr, PRE: tl.constexpr):
    """:func:`_lifted_sum_body` split ``SPLIT`` ways (``split_buf``: :func:`split_buffer`) and / or with ``PRE``,
    the pre-pass's chunks (:func:`_forward_prepass_kernel`, ``pre_buf``)."""
    _lifted_sum_body(node_mars = node_mars, element_mars = element_mars, weights = weights, nids = nids, cids = cids,
                     pids = pids, nb_reg = nb_reg, sel = sel, reg_offset = reg_offset, reg_width = reg_width,
                     reg_first = reg_first, reg_slots = reg_slots, a_rows = a_rows, a_regs = a_regs, a_pids = a_pids,
                     perm = perm, reg_perm = reg_perm, reg_live = reg_live, gidx = gidx, a_gidx = a_gidx, gck = gck,
                     gates = gates, gstride = gstride, lz = lz, lzstride = lzstride, coff = coff, reg_ivk = reg_ivk,
                     Bq = Bq, steps = steps, split_buf = split_buf, pre_buf = pre_buf, pre_cm = pre_cm, pre_ld = pre_ld,
                     elem_first = elem_first, elem_width = elem_width, B = B, num_edges = num_edges,
                     num_alias = num_alias, num_mtiles = num_mtiles, pid0 = pid0, BS = BS, TILE_M = TILE_M,
                     TILE_E = TILE_E, TILE_A = TILE_A, TILE_B = TILE_B, PRECISION = PRECISION, ALIAS = ALIAS,
                     PRUNE = PRUNE, INDEXED = INDEXED, GATED = GATED, LANES = LANES, CHUNK_GATE = CHUNK_GATE,
                     A_CHUNK_GATE = A_CHUNK_GATE, SHARED_GATES = SHARED_GATES, NG = NG, NGT = NGT, SEG = SEG,
                     A_SEG = A_SEG, SPLIT = SPLIT, PRE = PRE)


@triton.jit
def _frag_offsets(i, k):
    """Where a chunk's e of reduction rows ``i`` (32) and column positions ``k`` lie from the chunk's first (``32
    pre_ld``, ``pre_ld`` a multiple of 8) for the CUDA kernel (``c/fused_sum.cu``): a group of 8 columns' 32 rows at a
    time, every tensor-core B fragment's two values side by side -- of lane ``(k % 8) 4 + i % 4``, its k8 step ``i //
    8`` of the group, rows ``i % 8 < 4`` and ``+ 4``."""
    return ((k // 8) * 256 + (k % 8) * 8)[None, :] + ((i // 8) * 64 + (i % 4) * 2 + (i % 8) // 4)[:, None]


@triton.jit
def _frag_offsets_bf16(i, k):
    """:func:`_frag_offsets` for bf16 e (the CUDA kernel's mma m16n8k16, in elements): a group of 8 columns' 32 rows at
    a time, every B fragment's four values side by side -- of lane ``(k % 8) 4 + (i % 8) // 2``, its k16 step ``i //
    16`` of the group, rows ``2 (lane % 4)``, ``+ 1``, ``+ 8``, ``+ 9``."""
    return ((k // 8) * 256 + (k % 8) * 16)[None, :] + \
        ((i // 16) * 128 + ((i % 8) // 2) * 4 + ((i % 16) // 8) * 2 + i % 2)[:, None]


@triton_jit
def _forward_prepass_kernel(element_mars, pre_buf, pre_cm, pre_ld, cids, nb_reg, reg_slots, perm, reg_perm, reg_live,
                            elem_first, elem_width, B, num_edges, pid0, TILE_E: tl.constexpr, TILE_B: tl.constexpr,
                            PRUNE: tl.constexpr, PRECISION: tl.constexpr, FRAG: tl.constexpr = 0):
    """One program per (chunk of ``TILE_E`` edges, tile of columns) of a fused forward launch whose node blocks share
    their children (``cids`` row 0, region ``nb_reg[0]``): the children's values, their maximum per column ``t`` and the
    shifted exponentials ``e = exp(x - t)``, stored at ``pre_buf[edge * pre_ld + k]`` and
    ``pre_buf[pre_cm + chunk * pre_ld + k]`` by column position ``k``, as the forward kernel reads them -- e, for the
    CUDA kernel (chunks of 32), in its order: ``FRAG`` 1 (:func:`_frag_offsets`), or 2 in bf16
    (:func:`_frag_offsets_bf16`, from the buffer's first on, in elements); e rounded for the products' ``PRECISION``
    (:func:`~.common._tf32_operand`), which read it as it is."""
    chunk = tl.program_id(0)
    reg = tl.load(nb_reg)
    ncols = B * tl.load(reg_slots + reg)
    col0 = (pid0 + tl.program_id(1)) * TILE_B
    if col0 < ncols:
        k = col0 + tl.arange(0, TILE_B)
        kmask = k < ncols
        if PRUNE:
            num_live = tl.load(reg_live + reg)
            cols = tl.load(perm + tl.load(reg_perm + reg) + k, mask = kmask, other = 0)
            cmask = kmask & (k < num_live)
        else:
            cols = k
            cmask = kmask
        offs_e = chunk * TILE_E + tl.arange(0, TILE_E)
        emask = offs_e < num_edges
        cid = tl.load(cids + offs_e, mask = emask, other = 0).to(tl.int64)
        valid = emask & (cid >= elem_first)
        x = tl.load(element_mars + (cid - elem_first)[:, None] * elem_width + cols[None, :],
                    mask = valid[:, None] & cmask[None, :], other = float("-inf"))
        t = tl.max(x, axis = 0)
        e = tl.exp(x - tl.where(t == float("-inf"), 0.0, t)[None, :])
        if FRAG == 2:
            at = chunk.to(tl.int64) * TILE_E * pre_ld + _frag_offsets_bf16(tl.arange(0, TILE_E), k)
            tl.store(pre_buf.to(tl.pointer_type(tl.bfloat16)) + at, e.to(tl.bfloat16),
                     mask = emask[:, None] & kmask[None, :])
        else:
            if FRAG == 1:
                at = chunk.to(tl.int64) * TILE_E * pre_ld + _frag_offsets(tl.arange(0, TILE_E), k)
            else:
                at = offs_e.to(tl.int64)[:, None] * pre_ld + k[None, :]
            tl.store(pre_buf + at, _tf32_operand(e, PRECISION), mask = emask[:, None] & kmask[None, :])
        tl.store(pre_buf + pre_cm + chunk * pre_ld + k, t, mask = kmask)


_SHARED_CHILDREN = {}


def shared_children(cids: torch.Tensor, nb_reg: torch.Tensor) -> bool:
    """Whether every node block of a fused part reads the same children in the same region (an HMM layer's): read once
    per part (its tables are the program's, kept alive here with the answer)."""
    hit = _SHARED_CHILDREN.get(id(cids))
    if hit is None or hit[0] is not cids or hit[1] is not nb_reg:
        same = bool((cids == cids[:1]).all() and (nb_reg == nb_reg[:1]).all()) if cids.size(0) > 1 else True
        hit = _SHARED_CHILDREN[id(cids)] = (cids, nb_reg, same)
    return hit[2]


#: A fused forward launch the pre-pass serves runs on the CUDA kernel (``c/fused_sum.cu``: node blocks of 32 over the
#: pre-pass's chunks of 32 children, every chunk's sums on tensor cores times its factors) when it has at least this
#: many tiles of 4 node blocks x its columns (:data:`_CUDA_TNS`).
CUDA_MIN_TILES = 0

#: The CUDA kernel's tiles of columns: a launch's narrowest that covers its columns in as few tiles as 128 would.
_CUDA_TNS = (64, 80, 96, 112, 128)

#: Bytes of the shares' sums the merge reads in a chunk's time (L2 at ~4.7 TB/s, a chunk's products ~1.6 us): the
#: merge's cost in :func:`_cuda_split`.
_CUDA_MERGE_BYTES = 7.5e6

#: A CUDA launch splits only while its shares' sums take at most this many bytes of the split buffer.
_CUDA_SPLIT_BYTES = 1 << 28

_CUDA_TABLES = {}
_INT32_DUMMIES = {}
_CTAS_PER_SM = {}


def _cuda_tables(pids: torch.Tensor, cells_of: Optional[torch.Tensor]):
    """``(wbase, cells, nch, affine)`` of a fused part for the CUDA kernel -- every (node block, chunk of 32 edges)'s
    first weight, the cell of its external parameters' log-scales (``cells_of``: every edge's, None without), over the
    ``nch`` chunks before the tables' padding (a zero suffix: weight 0 of the -inf dummy child, nothing to add), and
    whether every node block's weights are one run (chunk ``c`` at its first + 1024 ``c``) -- or None if the weights
    are not laid out as the kernel reads them (a chunk's 32 x 32 contiguous, child-major, 16 bytes aligned). Kept per
    part (its tables are the program's, held alive here with the answer)."""
    hit = _CUDA_TABLES.get(id(pids))
    if hit is None or hit[0] is not pids or hit[1] is not cells_of:
        nb = pids.size(0)
        real = int((pids != 0).any(dim = 0).sum())
        nch = real // 32
        p = pids[:, :nch * 32].reshape(nb, nch, 32).long()
        wbase = p[:, :, 0].contiguous()
        ok = real % 32 == 0 and nch > 0 and bool((wbase % 4 == 0).all()) and \
            bool(((p - wbase[:, :, None]) == 32 * torch.arange(32, device = p.device)).all())
        cells, affine = None, False
        if ok:
            affine = bool((wbase == wbase[:, :1] + 1024 * torch.arange(nch, device = p.device)).all())
            if cells_of is not None:
                cells = cells_of[:, :nch * 32].reshape(nb, nch, 32).max(dim = 2).values.to(torch.int32).contiguous()
        hit = _CUDA_TABLES[id(pids)] = (pids, cells_of, (wbase, cells, nch, affine) if ok else None)
    return hit[2]


def _int32_dummy(dev) -> torch.Tensor:
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    if key not in _INT32_DUMMIES:
        _INT32_DUMMIES[key] = torch.zeros(1, dtype = torch.int32, device = dev)
    return _INT32_DUMMIES[key]


def _no_fragments(params: torch.Tensor) -> torch.Tensor:
    """An empty tensor: no fragment-ordered weights (the CUDA kernel reads ``params`` as they are)."""
    return params[:0]


def _ctas_per_sm(mod, dev, td: bool, ext: bool, x3: bool, fa: bool, bf: bool, tn: int, affine: bool) -> int:
    """How many CTAs of the CUDA kernel (``td``: top-down; ``fa``: fragment-ordered TF32 weights; ``bf``: bf16
    operands) an SM of ``dev`` holds at once (at a share's most chunks)."""
    key = (dev.index if dev.index is not None else torch.cuda.current_device(), td, ext, x3, fa, bf, tn, affine)
    if key not in _CTAS_PER_SM:
        with torch.cuda.device(key[0]):
            _CTAS_PER_SM[key] = max(1, int(mod.ctas_per_sm(td, ext, x3, fa, bf, tn, 256 if ext else 128, affine)))
    return _CTAS_PER_SM[key]


def _cuda_split(tiles: int, nch: int, ctas: int, tn: int, cap: int) -> Optional[int]:
    """Into how many shares a CUDA launch of ``tiles`` tiles of ``nch`` chunks splits every tile's chunks (shares of at
    most ``cap``; None if none fits): the split whose waves of ``ctas`` CTAs take the fewest chunks of products, plus
    the merge's reads (:data:`_CUDA_MERGE_BYTES`); within 5% of the fewest, the fewest shares."""
    lo = host_cdiv(nch, cap)
    hi = max(lo, min(SPLIT_MAX, nch // SPLIT_MIN_SHARE))
    share_bytes = 4 * (4 * 32 * tn + 4 * tn)
    costs = []
    for split in range(lo, hi + 1):
        share = host_cdiv(nch, split)
        split = host_cdiv(nch, share)                       # (every share some chunks)
        if split > 1 and tiles * split * share_bytes > _CUDA_SPLIT_BYTES:
            break
        merge = tiles * split * share_bytes / _CUDA_MERGE_BYTES if split > 1 else 0.0
        costs.append((host_cdiv(tiles * split, ctas) * share + merge, split))
    if not costs or costs[0][1] > SPLIT_MAX:
        return None
    least = min(c for c, _ in costs)
    return min(sp for c, sp in costs if c <= 1.05 * least)


def _cuda_shape(num_blocks: int, max_cols: int) -> Optional[tuple]:
    """``(tn, tiles)`` of a CUDA launch of ``num_blocks`` blocks over ``max_cols`` columns (:data:`_CUDA_TNS`), or
    None if it is not the CUDA kernel's (:data:`CUDA_MIN_TILES`)."""
    col_tiles = host_cdiv(max_cols, 128)
    per = host_cdiv(max_cols, col_tiles)
    tn = next(t for t in _CUDA_TNS if t >= per)
    col_tiles = host_cdiv(max_cols, tn)
    tiles = host_cdiv(num_blocks, 4) * col_tiles
    return None if tiles < CUDA_MIN_TILES or col_tiles > 65535 else (tn, tiles)


def _cuda_shares(mod, dev, td: bool, ext: bool, precision: str, ops: Optional[str], tn: int, tiles: int, nch: int,
                 affine: bool) -> Optional[int]:
    """:func:`_cuda_split` of a CUDA launch (``td``: top-down; ``ops``: :func:`_cuda_operands`) of ``tiles`` tiles of
    ``tn`` columns over ``nch`` chunks."""
    cap = min(256 if ext else nch, nch if affine else 128)
    ctas = _ctas_per_sm(mod, dev, td, ext, precision == "tf32x3", ops == "tf32", ops == "bf16", tn, affine) * \
        _num_sms(dev)
    key = (tiles, nch, ctas, tn, cap, SPLIT_MAX, SPLIT_MIN_SHARE, _CUDA_SPLIT_BYTES, _CUDA_MERGE_BYTES)
    if key not in _CUDA_SPLITS:                         # (by the launch's shape: a few per program)
        if len(_CUDA_SPLITS) >= 4096:
            _CUDA_SPLITS.clear()
        _CUDA_SPLITS[key] = _cuda_split(tiles, nch, ctas, tn, cap)
    return _CUDA_SPLITS[key]


def _cuda_operands(precision: str, frags: Optional[dict], bf16: bool) -> Optional[str]:
    """The operands of a CUDA launch's products besides the pre-pass's e: "bf16" (bf16 weights and e, a launch of a
    ``bf16`` query: TF32 ``precision`` and its copies in ``frags``), "tf32" (the TF32 weights' fragment-ordered copy,
    in ``frags``), or None (the weights as they are)."""
    if frags is None or precision != "tf32":
        return None
    return "bf16" if bf16 else "tf32"


def _cuda_plan(params, pids, num_blocks, gated, g, num_edges, max_cols, block_size, TILE_E, precision, aliased,
               dev, frags: Optional[dict] = None, bf16: bool = False) -> Optional[tuple]:
    """``(module, tables, tn, split, wfrag)`` if a fused forward launch the pre-pass serves is the CUDA kernel's (see
    :data:`CUDA_MIN_TILES`), else None. ``g``: the launch's :func:`gate_launch` arguments; ``wfrag``: the weights in
    the kernel's fragment order (:func:`_cuda_operands`, from ``frags``: :func:`_cuda_fragments`; bf16: the products
    on bf16 operands), or None."""
    if block_size != 32 or TILE_E != 32 or num_edges % 32 or aliased is not None or \
            precision not in ("tf32", "tf32x3") or \
            (gated is not None and (gated["node_gate"] < block_size or not gated["uniform"].get(32, False))):
        return None                                      # (external parameters: a cell per node block and chunk)
    shape = _cuda_shape(num_blocks, max_cols)
    if shape is None:
        return None
    tn, tiles = shape
    from .c import get_module
    mod = get_module()
    tables = _cuda_tables(pids, g[0] if gated is not None else None) if mod is not None else None
    if tables is None:
        return None
    ops = _cuda_operands(precision, frags, bf16)
    split = _cuda_shares(mod, dev, False, gated is not None, precision, ops, tn, tiles, tables[2], tables[3])
    if split is None:
        return None
    return mod, tables, tn, split, None if ops is None else _cuda_fragments(mod, frags, params, tables[0], False,
                                                                           ops == "bf16")


_TILE_KEYS = {}


def _tiles_key(wbase: torch.Tensor) -> str:
    """A digest of the tiles a part's launches read (``wbase``, the program's: kept alive here with it), the same
    for parts that read the same tiles (an HMM's tied transitions). Read off the device once per part."""
    hit = _TILE_KEYS.get(id(wbase))
    if hit is None or hit[0] is not wbase:
        hit = _TILE_KEYS[id(wbase)] = (wbase, hashlib.sha1(wbase.cpu().numpy().tobytes()).hexdigest())
    return hit[1]


def _cuda_fragments(mod, frags: dict, params: torch.Tensor, wbase: torch.Tensor, td: bool,
                    bf16: bool = False) -> torch.Tensor:
    """The weights of a CUDA launch (its every 32 x 32 tile's first ``wbase``) in the kernel's A-fragment order
    (``c/fused_sum.cu``, ``fragment_weights``): rounded to TF32, or (``bf16``) a bf16 copy in mma m16n8k16's order --
    a copy of ``params`` per direction (``td``), every tile in its own place, kept in ``frags``: the stacked weights
    of a :func:`pyjuice.fast_inference` scope that allows parameter copies, or a bf16 query's own
    (:meth:`~..state.State.weight_stacks`). A part's tiles are copied by the first launch that reads them (parts
    reading the same tiles share the copy), the parameters not changing inside the scope or query. At TF32 the
    fragments are the products' own, so the sums do not change."""
    key = ("cuda_fragments", td, bf16)
    if key not in frags:
        frags[key] = (torch.empty(params.numel(), dtype = torch.bfloat16 if bf16 else params.dtype,
                                  device = params.device), set())
    frag, done = frags[key]
    tiles = _tiles_key(wbase)
    if tiles not in done:
        mod.fragment_weights(params, frag, wbase, td)
        done.add(tiles)
    return frag


def _cuda_split_buffer(dev, num_blocks: int, max_cols: int, tn: int, nch: int, split: int, fallback: torch.Tensor):
    """The split buffer of a CUDA launch (``fallback`` unsplit: never read)."""
    if split == 1:
        return fallback
    share = host_cdiv(nch, split)
    per_tile = host_cdiv(4 * tn + host_cdiv(nch, share) * (4 * 32 * tn + 4 * tn), 4) * 4
    return split_buffer(dev, host_cdiv(num_blocks, 4) * host_cdiv(max_cols, tn), per_tile)


def _cuda_forward(plan, node_mars, params, nids, nb_reg, sel, num_blocks, regions, lanes, gated, g, pre_args,
                  batch_size, max_cols, precision):
    """The fused forward launch ``plan`` (:func:`_cuda_plan`) on the CUDA kernel, from the pre-pass's chunks
    (``pre_args``: its buffer, ``pre_cm``, ``pre_ld``)."""
    mod, (wbase, cells, nch, affine), tn, split, wfrag = plan
    sbuf = _cuda_split_buffer(node_mars.device, num_blocks, max_cols, tn, nch, split, node_mars)
    pre, pre_cm, pre_ld = pre_args
    _, _, _, _, sstride, lz, lzstride, coff, reg_ivk, Bq, steps = g
    ext = gated is not None
    perm, reg_perm, reg_live = (lanes.perm, lanes.reg_first, lanes.reg_live) if lanes is not None else (nids,) * 3
    mod.fused_sum_cuda(False, node_mars, node_mars, params, _no_fragments(params) if wfrag is None else wfrag, wbase,
                       1024, pre, pre_ld, pre_cm, nids, nb_reg, nb_reg, nids if sel is None else sel, regions["offset"],
                       regions["width"], regions["first"], regions["slots"], perm, reg_perm, reg_live, 0, 0,
                       gated["minor"] if ext else node_mars,
                       cells if ext else _int32_dummy(node_mars.device), gated["mstride"] if ext else 0,
                       lz if ext else node_mars, lzstride if ext else 0, coff, reg_ivk, Bq, steps, batch_size,
                       num_blocks, nch, max_cols, ext, ext and sstride == 0, ext and gated["lanes"] is not None,
                       lanes is not None, sel is not None, precision == "tf32x3", affine, tn, sbuf, split)


_CUDA_TD_TABLES = {}
_CUDA_SPLITS = {}


def _cuda_td_tables(tables: dict, gt: Optional[dict], cbs: int):
    """The CUDA kernel's tables of a fused top-down part whose child blocks (of ``cbs`` children) it reads in slices
    of 32 children -- its row blocks: every slice's first row, region and region of columns, its first weight for
    every parent entry and (``gt``: the part's gate tables) the cell of its log-scales -- as ``(rows, rreg, colreg,
    wbase, cells, npar, affine, wstride, slices)``, or None unless every child block reads the same parents (node
    blocks and regions, in the same order: one region of columns) at 16-byte aligned weights and none is log-added to
    values written before (``cb_add``: with blocks of 32 a child's parents are one launch's).
    ``affine``: every slice's weights one run (entry ``j``'s at its first + ``wstride`` ``j``). Kept per part (its
    tables are the program's, held alive here with the answer)."""
    pid = tables["par_pid"]
    hit = _CUDA_TD_TABLES.get(id(pid))
    if hit is None or hit[0] is not pid or hit[1] is not gt:
        ptr = tables["par_ptr"].long()
        n_cb, sl = ptr.numel() - 1, cbs // 32
        counts = ptr[1:] - ptr[:-1]
        npar = int(counts[0]) if n_cb > 0 else 0
        out = None
        if npar > 0 and bool((counts == npar).all()) and not bool(tables["cb_add"].any()):
            dev = ptr.device
            at = ptr[:-1, None] + torch.arange(npar, device = dev)[None, :]                     # [child blocks, j]
            nid, reg, ivreg = tables["par_nid"].long()[at], tables["par_reg"].long()[at], tables["cb_ivreg"].long()
            ar = torch.arange(sl, device = dev)
            wbase = (pid.long()[at][:, None, :] + 1024 * ar[None, :, None]).reshape(n_cb * sl, npar).contiguous()
            if bool((nid == nid[:1]).all()) and bool((reg == reg[:1]).all()) and bool((ivreg == ivreg[0]).all()) and \
                    bool((wbase % 4 == 0).all()):
                d = wbase[:, 1:] - wbase[:, :-1]
                wstride = int(d[0, 0]) if npar > 1 else 0
                affine = npar == 1 or bool((d == wstride).all())
                cells = None
                if gt is not None:                       # (the slice's child gate)
                    lc = gt["lc"].long()[at][:, None, :] + 32 * ar[None, :, None]
                    cells = (gt["row"].long()[at][:, None, :] + lc // gt["cbs"].long()[at][:, None, :])
                    cells = cells.reshape(n_cb * sl, npar).to(torch.int32).contiguous()
                each = lambda t: t.long().repeat_interleave(sl).contiguous()
                rows = (tables["cb_row"].long()[:, None] + 32 * ar[None, :]).reshape(-1).contiguous()
                out = (rows, each(tables["cb_reg"]), each(tables["cb_ivreg"]), wbase, cells, npar, affine, wstride,
                       sl)
        hit = _CUDA_TD_TABLES[id(pid)] = (pid, gt, out)
    return hit[2]


def _cuda_td_plan(tables, gt, idx, block_size, cbs, max_cols, TILE_I, precision, dev, params,
                  frags: Optional[dict] = None, bf16: bool = False) -> Optional[tuple]:
    """``(module, tables, tn, split, wfrag)`` if a fused top-down launch is the CUDA kernel's: child blocks of 32
    children or a multiple (read in slices of 32, :func:`_cuda_td_tables`) under the same parent node blocks of 32,
    none log-added (with external parameters, every slice in one child gate and a parent block in one node gate), see
    :data:`CUDA_MIN_TILES` -- else None. ``wfrag``: the weights ``params`` in the kernel's fragment order
    (:func:`_cuda_operands`, from ``frags``: :func:`_cuda_fragments`), or None."""
    if block_size != 32 or cbs % 32 or TILE_I != 32 or precision not in ("tf32", "tf32x3") or \
            (gt is not None and (gt["node_gate"] < block_size or not gt["uniform"].get(32, False))):
        return None
    shape = _cuda_shape((tables["cb_row"].numel() if idx is None else idx.numel()) * (cbs // 32), max_cols)
    if shape is None:
        return None
    tn, tiles = shape
    from .c import get_module
    mod = get_module()
    td = _cuda_td_tables(tables, gt, cbs) if mod is not None else None
    if td is None:
        return None
    ops = _cuda_operands(precision, frags, bf16)
    split = _cuda_shares(mod, dev, True, gt is not None, precision, ops, tn, tiles, td[5], td[6])
    if split is None:
        return None
    return mod, td, tn, split, None if ops is None else _cuda_fragments(mod, frags, params, td[3], True,
                                                                       ops == "bf16")


def _cuda_top_down(plan, element_flows, node_flows, params, tables, regions, elem_first, elem_width, batch_size,
                   max_cols, precision, liveness, idx, gated, gl):
    """The fused top-down launch ``plan`` (:func:`_cuda_td_plan`) on the CUDA kernel: the parents' chunks once, by the
    pre-pass, then the products. ``gl``: the launch's :func:`gate_launch` arguments past the edges'."""
    mod, (rows, rreg, colreg, wbase, cells, npar, affine, wstride, sl), tn, split, wfrag = plan
    dev = element_flows.device
    sel = None
    if idx is not None:                                  # (the index list's child blocks' slices)
        sel = (idx.long()[:, None] * sl + torch.arange(sl, device = dev)[None, :]).reshape(-1)
    num_blocks = rows.numel() if sel is None else sel.numel()
    lanes = liveness.lanes if liveness is not None else None
    perm, reg_perm, reg_live = (lanes.perm, lanes.reg_first, lanes.reg_live) if lanes is not None else (rows,) * 3
    TILE_B = min(_TILES[precision][2], max(16, host_next_power_of_2(max_cols)))
    pre_ld = host_cdiv(max_cols, 8) * 8                                 # (groups of 8 columns, :func:`_frag_offsets`)
    pre = pre_buffer(dev, npar * (32 + 1) * pre_ld)
    for first, size in grid_chunks(host_cdiv(max_cols, TILE_B), 1):      # (one child block's parents: every one's)
        _top_down_prepass_kernel[(npar, size)](
            node_flows, pre, npar * 32 * pre_ld, pre_ld, tables["cb_ivreg"], tables["cb_row"] if idx is None else idx,
            tables["par_ptr"], tables["par_nid"], tables["par_reg"], regions["offset"], regions["width"],
            regions["first"], regions["slots"], perm, reg_perm, reg_live, gl[2], gl[3], gl[4], gl[5], gl[6], gl[7],
            batch_size, npar, first, BS = 32, NGC = 32, TILE_I = 32, TILE_B = TILE_B, PRUNE = liveness is not None,
            INDEXED = idx is not None, GATED = gated is not None,
            LANES = gated is not None and gated["lanes"] is not None,
            SHARED_GATES = gated is not None and gated["gstride"] == 0, REL = True, PRECISION = precision,
            FRAG = 2 if wfrag is not None and wfrag.dtype == torch.bfloat16 else 1, num_warps = 4)
    ext = gated is not None
    sbuf = _cuda_split_buffer(dev, num_blocks, max_cols, tn, npar, split, element_flows)
    mod.fused_sum_cuda(True, element_flows, node_flows, params, _no_fragments(params) if wfrag is None else wfrag,
                       wbase, wstride, pre, pre_ld, npar * 32 * pre_ld, rows, rreg, colreg,
                       rows if sel is None else sel, regions["offset"], regions["width"], regions["first"],
                       regions["slots"], perm, reg_perm, reg_live, elem_first, elem_width,
                       gated["minor"] if ext else element_flows, cells if ext else _int32_dummy(dev),
                       gated["mstride"] if ext else 0,
                       element_flows, 0, gl[4], gl[5], gl[6], gl[7], batch_size, num_blocks, npar, max_cols, ext,
                       ext and gl[1] == 0, ext and gated["lanes"] is not None, liveness is not None, sel is not None,
                       precision == "tf32x3", affine, tn, sbuf, split)


def fused_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, nids: torch.Tensor,
              cids: torch.Tensor, pids: torch.Tensor, nb_reg: torch.Tensor, regions: dict, elem_first: int,
              elem_width: int, batch_size: int, block_size: int, max_cols: int, precision: str, aliased = None,
              liveness = None, sel = None, count: Optional[int] = None, gated: Optional[dict] = None,
              frags: Optional[dict] = None, bf16: bool = False):
    """
    Node blocks of a sum layer partition with the fused Triton kernel, into ``node_mars`` -- or, a launch of node
    blocks of 32 the pre-pass serves, the CUDA kernel (:func:`_cuda_plan`).

    :param nids, cids, pids: the node blocks' tables (see the module docstring)
    :param nb_reg: [num node blocks] the sum region every node block's rows belong to
    :param regions: ``offset``, ``width``, ``first`` and ``slots`` of every sum region, as device tensors
    :param elem_first, elem_width: the first row and row width of the product layer group's region
    :param max_cols: the most columns any of the node blocks has (``batch_size`` x its slots)
    :param aliased: None, or the node blocks' children that copy a sum and are read in the sum's own row:
        ``(rows, regions, pids)``, each [num node blocks, num aliased] int32 (``rows`` within the region, ``-1`` for
        padding; ``pids`` as for ``cids``)
    :param liveness: None, or the query's :class:`~.live.Liveness` (its lane order covers the regions computed)
    :param sel, count: None for every node block, or a query plan's int32 index list of them and its length
    :param gated: None, or the gates of a gated layer's node blocks (:func:`~..queries.backend.forward.part_gates`):
        ``idx`` / ``a_idx`` (the edges' cells, as ``cids`` / the aliased children), the session's gates and normalizers
    :param frags: None, or where the CUDA kernel's fragment-ordered weights are kept (:func:`_cuda_fragments`)
    :param bf16: a CUDA launch's products on bf16 operands (a ``precision = "bf16"`` query's: ``precision`` "tf32",
        ``frags`` given)
    """
    num_blocks, num_edges = cids.shape
    if sel is not None:
        num_blocks = count
    tm, te, tb, warps, stages = _TILES[precision]
    TILE_M = min(tm, max(16, host_next_power_of_2(block_size)))
    TILE_E = min(te, max(16, host_next_power_of_2(num_edges)))
    TILE_B = min(tb, max(16, host_next_power_of_2(max_cols)))
    NG = gated["node_gate"] if gated is not None else block_size
    if NG < block_size:
        TILE_M = _row_tile(TILE_M, block_size, NG)
    NGT = _node_gates_per_tile(block_size, NG, TILE_M)
    warps = _tile_warps(warps, TILE_M)
    rows = NGT > 1                                       # a tile's rows read several node gates
    if gated is not None:                                # a chunk of edges within one gate, if one fits
        TILE_E = _gate_tile(TILE_E, gated["uniform"])
        if rows and not gated["uniform"].get(TILE_E, False):            # else a product per segment of one gate:
            TILE_E = 16                                                 # chunks of the fewest
    num_mtiles = host_cdiv(block_size, TILE_M)
    a_rows, a_regs, a_pids = aliased if aliased is not None else (cids, cids, cids)
    num_alias = a_rows.size(1) if aliased is not None else 0
    TILE_A = min(te, max(16, host_next_power_of_2(num_alias)))
    if rows and aliased is not None:
        TILE_A = _gate_tile(TILE_A, gated["a_uniform"])
        if not gated["a_uniform"].get(TILE_A, False):
            TILE_A = 16
    lanes = liveness.lanes if liveness is not None else None
    perm, reg_perm, reg_live = (lanes.perm, lanes.reg_first, lanes.reg_live) if lanes is not None else \
        (cids, cids, cids)
    g = gate_launch(gated, cids, gated.get("idx") if gated else None,
                    gated.get("a_idx") if gated and aliased is not None else None, gated.get("ck") if gated else None)
    num_col_tiles = host_cdiv(max_cols, TILE_B)
    tiles = num_blocks * num_mtiles * num_col_tiles
    split, buf, split_kw = 1, (), _NO_SPLIT                 # unsplit: the kernel without the split's arguments
    if num_edges >= 2 * SPLIT_MIN_SHARE * TILE_E and tiles <= SPLIT_TILES and \
            tiles < SPLIT_PROGRAMS_PER_SM * _num_sms(node_mars.device):                  # a launch that may split
        split = split_count(tiles, host_cdiv(num_edges, TILE_E), node_mars.device)
        if split > 1:
            buf, split_kw = (split_buffer(node_mars.device, tiles, split * TILE_M * TILE_B),), dict(SPLIT = split)
    pre_kw = _NO_SPLIT
    if not rows and (gated is None or gated["uniform"].get(TILE_E, False)) and \
            num_blocks * num_mtiles >= PRE_MIN_TILES and shared_children(cids, nb_reg):   # the children's chunks once
        nch = host_cdiv(num_edges, TILE_E)
        plan = _cuda_plan(params, pids, num_blocks, gated, g, num_edges, max_cols, block_size, TILE_E, precision,
                          aliased, node_mars.device, frags, bf16)
        al = 4 if plan is None else 8                   # (the CUDA kernel's groups of 8 columns, :func:`_frag_offsets`)
        pre_ld = host_cdiv(max_cols, al) * al
        pre = pre_buffer(node_mars.device, (nch * TILE_E + nch) * pre_ld)
        for first, size in grid_chunks(num_col_tiles, 1):
            _forward_prepass_kernel[(nch, size)](
                element_mars, pre, nch * TILE_E * pre_ld, pre_ld, cids, nb_reg, regions["slots"], perm, reg_perm,
                reg_live, elem_first, elem_width, batch_size, num_edges, first, TILE_E = TILE_E, TILE_B = TILE_B,
                PRUNE = liveness is not None, PRECISION = precision,
                FRAG = 0 if plan is None else 2 if plan[4] is not None and plan[4].dtype == torch.bfloat16 else 1,
                num_warps = 4)
        pre_kw = dict(pre_args = (pre, nch * TILE_E * pre_ld, pre_ld))
        if plan is not None:
            _cuda_forward(plan, node_mars, params, nids, nb_reg, sel, num_blocks, regions, lanes, gated, g,
                          pre_kw["pre_args"], batch_size, max_cols, precision)
            return
    if split > 1 or pre_kw:
        buf = (buf[0] if buf else node_mars,) + (pre_kw["pre_args"] if pre_kw else (node_mars, 0, 0))
        split_kw = dict(SPLIT = split, PRE = bool(pre_kw))
    for first, size in grid_chunks(num_col_tiles, 1):
        (_lifted_sum_aux_kernel if buf else _lifted_sum_kernel)[(num_blocks * num_mtiles * split, size)](
            node_mars, element_mars, params, nids, cids, pids, nb_reg, nids if sel is None else sel, regions["offset"],
            regions["width"],
            regions["first"], regions["slots"], a_rows, a_regs, a_pids, perm, reg_perm, reg_live, *g, *buf,
            elem_first, elem_width, batch_size, num_edges, num_alias, num_mtiles, first,
            BS = block_size, TILE_M = TILE_M,
            TILE_E = TILE_E, TILE_A = TILE_A, TILE_B = TILE_B, PRECISION = precision, ALIAS = aliased is not None,
            PRUNE = liveness is not None, INDEXED = sel is not None, GATED = gated is not None,
            LANES = gated is not None and gated["lanes"] is not None,
            CHUNK_GATE = gated is not None and gated["uniform"].get(TILE_E, False),
            A_CHUNK_GATE = gated is not None and aliased is not None and gated["a_uniform"].get(TILE_A, False),
            SHARED_GATES = gated is not None and gated["gstride"] == 0, NG = NG, NGT = NGT,
            SEG = _gate_segment(TILE_E, gated["uniform"]) if gated is not None else 1,
            A_SEG = _gate_segment(TILE_A, gated["a_uniform"]) if gated is not None and aliased is not None else 1,
            **split_kw, num_warps = warps, num_stages = stages)


# -------------------------------------------------------------------------------------------------
# Dense node blocks
# -------------------------------------------------------------------------------------------------

def dense_groups(nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor, block_size: int, elem_first: int):
    """
    Split a partition's node blocks into dense ones, grouped by their run of children, and the rest.

    :returns: ``(groups, rest)``: ``groups`` maps the first child row to ``[(node block index, nid, pid0)]`` for
        node blocks whose children are ``E`` contiguous element rows and whose weights are ``params[pid0 + e * BS
        + i]``; ``rest`` is a long tensor of the other node blocks' indices
    """
    E = cids.size(1)
    e = torch.arange(E, device = cids.device)
    dense = (cids == cids[:, :1] + e).all(dim = 1) & (pids == pids[:, :1] + e * block_size).all(dim = 1) & \
        (cids[:, 0] >= elem_first)
    if block_size < DENSE_MIN_BLOCK:
        dense[:] = False
    groups = {}
    for k, ok, nid, c0, p0 in zip(range(cids.size(0)), dense.tolist(), nids.tolist(), cids[:, 0].tolist(),
                                  pids[:, 0].tolist()):
        if ok:
            groups.setdefault(c0, []).append((k, nid, p0))
    rest = torch.nonzero(~dense).flatten()
    return groups, rest


@contextlib.contextmanager
def _matmul_precision(precision: str):
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


#: The dense path's elementwise passes: (rows, columns) per program.
_EW_TILE = (32, 128)


@triton_jit
def _shifted_exp_kernel(x, ex, m, rows, ncols, x_ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``ex = exp(x - m)`` (``ex`` contiguous [rows, ncols]), ``m`` the column maxima of ``x``; a column of -inf
    is shifted by 0 and stays all 0."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    shift = tl.where(shift == float("-inf"), 0.0, shift)
    r = r.to(tl.int64)
    v = tl.load(x + r[:, None] * x_ld + c[None, :], mask = mask, other = float("-inf"))
    tl.store(ex + r[:, None] * ncols + c[None, :], tl.exp(v - shift[None, :]), mask = mask)


@triton_jit
def _live_shifted_exp_kernel(x, perm, ex_t, m, rows, x_ld, TILE_R: tl.constexpr):
    """One program per live column ``k`` (``perm[k]`` of ``x``, [rows, *] with row stride ``x_ld``): its maximum
    ``m[k]`` and ``ex_t[k, :] = exp(x[:, perm[k]] - m[k])`` -- the live columns gathered, shifted and exponentiated in
    one pass, stored transposed (contiguous per column) for the product to read as ``ex_t.t()``."""
    k = tl.program_id(0)
    col = tl.load(perm + k).to(tl.int64)
    big = tl.full([TILE_R], float("-inf"), dtype = tl.float32)
    for r0 in range(0, rows, TILE_R):
        r = r0 + tl.arange(0, TILE_R)
        big = tl.maximum(big, tl.load(x + r.to(tl.int64) * x_ld + col, mask = r < rows, other = float("-inf")))
    top = tl.max(big, axis = 0)
    tl.store(m + k, top)
    shift = tl.where(top == float("-inf"), 0.0, top)
    for r0 in range(0, rows, TILE_R):
        r = r0 + tl.arange(0, TILE_R)
        v = tl.load(x + r.to(tl.int64) * x_ld + col, mask = r < rows, other = float("-inf"))
        tl.store(ex_t + k.to(tl.int64) * rows + r, tl.exp(v - shift), mask = r < rows)


def _live_shifted_exp(x: torch.Tensor, perm: torch.Tensor, num_live: int):
    """``(ex_t, m)`` of :func:`_live_shifted_exp_kernel`: [num_live, rows] and [num_live] for ``x`` [rows, *]."""
    rows = x.size(0)
    ex_t = torch.empty(num_live, rows, dtype = x.dtype, device = x.device)
    m = torch.empty(num_live, dtype = x.dtype, device = x.device)
    if num_live > 0:
        _live_shifted_exp_kernel[(num_live,)](x, perm, ex_t, m, rows, x.stride(0), TILE_R = 1024, num_warps = 4)
    return ex_t, m


@triton_jit
def _log_add_kernel(out, m, rows, ncols, ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``out = log(out) + m`` in place, ``m`` the column maxima :func:`_shifted_exp_kernel` shifted by (a column of
    -inf has a product of 0, and log 0 - inf = -inf: no guard needed)."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    ptr = out + r.to(tl.int64)[:, None] * ld + c[None, :]
    tl.store(ptr, tl.log(tl.load(ptr, mask = mask, other = 1.0)) + shift[None, :], mask = mask)


@triton_jit
def _log_scatter_kernel(out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R: tl.constexpr,
                        TILE_C: tl.constexpr):
    """``out[:, c] = log(prod[:, rank[c]]) + m[rank[c]]`` for every live column ``c`` of ``out``, -inf for the others
    (``prod`` [rows, num_live], the products of the live columns)."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    k = tl.load(rank + c, mask = cmask, other = -1)
    alive = k >= 0
    shift = tl.load(m + k, mask = cmask & alive, other = 0.0)
    r = r.to(tl.int64)
    v = tl.load(prod + r[:, None] * num_live + k[None, :], mask = mask & alive[None, :], other = 1.0)
    tl.store(out + r[:, None] * ld + c[None, :], tl.where(alive[None, :], tl.log(v) + shift[None, :], float("-inf")),
             mask = mask)


def stacked_weights(params: torch.Tensor, p0s, E: int, block_size: int) -> torch.Tensor:
    """``[K * block_size, E]``: the weights of K dense node blocks over the same ``E`` children (node block ``k``'s
    ``[E, block_size]`` matrix at ``params[p0s[k]]``), transposed and stacked in the order of ``p0s`` -- a copy, as
    the blocks' matrices are not one strided view of the parameter buffer."""
    return torch.cat([params[p0:p0 + E * block_size].view(E, block_size).t() for p0 in p0s], dim = 0)


def _products(blocks, params, E, BS, stacked):
    """``(weights [rows, E], first node row)`` of every matrix product of a run of node blocks: one with their
    ``stacked`` weights (the blocks then occupy consecutive rows), else one per block, read in place."""
    if stacked is not None:
        return [(stacked, blocks[0][1])]
    return [(params[p0:p0 + E * BS].view(E, BS).t(), nid) for _, nid, p0 in blocks]


def dense_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, groups: dict, E: int,
              block_size: int, region: tuple, elem_first: int, elem_width: int, ncols: int, precision: str,
              live = None, stacked: dict = None):
    """
    Dense node blocks of one sum region (see :func:`dense_groups`), into ``node_mars``: per run of children,
    ``log(W_k @ exp(X - m)) + m`` with ``m`` the column maximum of the children ``X``.

    :param region: ``(offset, width, first)`` of the node blocks' sum region (Python ints)
    :param live: None (every column), or the region's interval's ``(perm, rank, num_live)`` (see
        :meth:`~.live.LaneOrder.segment`): the live columns are gathered and multiplied, the others set to -inf
    :param stacked: first child row -> the :func:`stacked_weights` of that run's node blocks, which occupy
        consecutive rows in the order listed: one product for the run instead of one per block
    """
    offset, width, first = region
    BS = block_size
    stacked = stacked or {}
    elems = element_mars[:element_mars.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_mars[offset:offset + (node_mars.numel() - offset) // width * width].view(-1, width)
    TILE_R, TILE_C = _EW_TILE
    num_ctiles = host_cdiv(ncols, TILE_C)
    if live is not None:
        _dense_sum_live(nodes, elems, params, groups, E, BS, first, elem_first, ncols, precision, live, stacked)
        return
    with _matmul_precision(precision):
        for c0, blocks in groups.items():
            x = elems[c0 - elem_first:c0 - elem_first + E, :ncols]
            m = x.amax(dim = 0)
            ex = torch.empty(E, ncols, dtype = x.dtype, device = x.device)
            _shifted_exp_kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](x, ex, m, E, ncols, elem_width, num_ctiles,
                                                                     TILE_R = TILE_R, TILE_C = TILE_C)
            for w, nid in _products(blocks, params, E, BS, stacked.get(c0)):
                rows = w.size(0)
                out = nodes[nid - first:nid - first + rows, :ncols]
                torch.mm(w, ex, out = out)
                _log_add_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](out, m, rows, ncols, width, num_ctiles,
                                                                        TILE_R = TILE_R, TILE_C = TILE_C)


def _dense_sum_live(nodes, elems, params, groups, E, BS, first, elem_first, ncols, precision, live, stacked):
    perm, rank, num_live = live
    width = nodes.size(1)
    TILE_R, TILE_C = _EW_TILE
    with _matmul_precision(precision):
        for c0, blocks in groups.items():
            ex_t, m = _live_shifted_exp(elems[c0 - elem_first:c0 - elem_first + E], perm, num_live)    # [live, E]
            for w, nid in _products(blocks, params, E, BS, stacked.get(c0)):
                rows = w.size(0)
                prod = torch.mm(w, ex_t.t())                                       # [rows, live]
                num_ctiles = host_cdiv(ncols, TILE_C)
                _log_scatter_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](
                    nodes[nid - first:nid - first + rows], prod, m, rank, rows, ncols, width, num_live, num_ctiles,
                    TILE_R = TILE_R, TILE_C = TILE_C)


@triton.jit
def _lifted_sum_top_down_body(element_flows, node_flows, weights, cb_row, cb_reg, cb_add, cb_ivreg, cb_idx, par_ptr,
                               par_nid, par_reg, par_pid, reg_offset, reg_width, reg_first, reg_slots, perm, reg_perm,
                               reg_live, par_lc, par_grow, par_gcbs, par_gck, gates, gstride, lz, lzstride, coff,
                               reg_ivk, Bq, steps, split_buf, pre_buf, pre_cm, pre_ld, elem_first, elem_width, B,
                               pid0, BS: tl.constexpr, CBS: tl.constexpr, TILE_C: tl.constexpr, TILE_I: tl.constexpr,
                               TILE_B: tl.constexpr, PRECISION: tl.constexpr, PRUNE: tl.constexpr,
                               INDEXED: tl.constexpr, GATED: tl.constexpr, LANES: tl.constexpr,
                               GATE_TILE: tl.constexpr, SHARED_GATES: tl.constexpr, NG: tl.constexpr,
                               SPLIT: tl.constexpr, PRE: tl.constexpr):
    """One program per (child block, tile of its children, tile of columns): ``O_child[c, col] = log sum over the
    block's parent node blocks and their nodes i of w[c, i] exp(O_node[i, col])``, an online log-sum-exp over tiles
    of parent nodes with the products on tensor cores -- the forward's fused kernel with children and nodes
    exchanged (``w[c, i] = weights[pid + c * BS + i]``). A child block is element rows (``cb_reg`` -1), written to
    ``element_flows``, or a sum node block's rows (an aliased copy), written to ``node_flows``; written, or log-added
    to what is there (``cb_add``). Pruned, the columns are the interval's live ones first, as in the forward.
    ``INDEXED``: the child blocks are those of the plan's index list ``cb_idx``. ``GATED`` (see
    :mod:`..external_params.block_scale`):
    every parent node's value less its ``log Z`` of the column's sample, and a parent block's sum times the gate of
    every child and sample -- its grid row ``par_grow``, the child gate of the block's first child's place ``par_lc``
    over ``par_gcbs`` --, log-added per child and column (the gates vary with both); ``GATE_TILE``: every tile of
    children lies in one child gate, whose gate the parent's values take as they come (one per column);
    ``SHARED_GATES``: the gates are every sample's; ``NG``: the node gate size -- below ``BS`` a parent block's rows
    read several (node gate ``i // NG`` of row ``i``, ``par_gck`` cells per gate from the first row's): taken by the
    parent values per row with ``GATE_TILE``, else a sum of its own per node gate. ``SPLIT``: every (child block, tile
    of its children, tile of columns) has as many programs, each over its share of the parents (:func:`_split_merge`,
    ``split_buf``). ``PRE``: every parent chunk's shifted exponentials and maximum per column were computed once by
    :func:`_top_down_prepass_kernel` (``pre_buf``, ``pre_cm`` / ``pre_ld``): a chunk is one dot and a rescale of
    columns (:func:`_lse_dot_step_pre`)."""
    pid_cm = tl.program_id(0)
    pid_c = pid0 + tl.program_id(1)
    s = pid_cm % SPLIT                                                        # the program's share of the parents
    pid_t = pid_cm // SPLIT
    cb = pid_t // ((CBS + TILE_C - 1) // TILE_C)
    ct = pid_t % ((CBS + TILE_C - 1) // TILE_C)
    if GATED and not GATE_TILE:                                               # the pre-pass's chunks of a parent:
        NCPG: tl.constexpr = (NG + TILE_I - 1) // TILE_I                      # TILE_I rows within each node gate
        NCH: tl.constexpr = (BS // NG) * NCPG
    else:
        NCPG: tl.constexpr = (BS + TILE_I - 1) // TILE_I
        NCH: tl.constexpr = NCPG
    if INDEXED:
        cb = tl.load(cb_idx + cb)
    ivreg = tl.load(cb_ivreg + cb)
    ncols = B * tl.load(reg_slots + ivreg)
    col0 = pid_c * TILE_B
    if col0 < ncols:
        k = col0 + tl.arange(0, TILE_B)
        kmask = k < ncols
        if PRUNE:
            num_live = tl.load(reg_live + ivreg)
            cols = tl.load(perm + tl.load(reg_perm + ivreg) + k, mask = kmask, other = 0)
            cmask = kmask & (k < num_live)
            skip = col0 >= num_live                                           # a dead tile reads nothing
        else:
            cols = k
            cmask = kmask
            skip = False
        offs_c = ct * TILE_C + tl.arange(0, TILE_C)
        cm = offs_c < CBS
        m = tl.full([TILE_B], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE_C, TILE_B], dtype = tl.float32)
        samp = cols
        if GATED and not SHARED_GATES:
            samp = _column_sample(cols, ivreg, reg_slots, coff, reg_ivk, Bq, steps, LANES)
        if GATED:
            if not GATE_TILE:                                                 # per child and column
                m2 = tl.full([TILE_C, TILE_B], float("-inf"), dtype = tl.float32)
                acc2 = tl.zeros([TILE_C, TILE_B], dtype = tl.float32)
        p0 = tl.load(par_ptr + cb)
        p1 = tl.load(par_ptr + cb + 1)
        if SPLIT > 1:
            share = (p1 - p0 + SPLIT - 1) // SPLIT
            p0 = p0 + s * share
            p1 = tl.minimum(p1, p0 + share)
        if not skip:
            for p in range(p0, p1):
                nid = tl.load(par_nid + p).to(tl.int64)
                reg = tl.load(par_reg + p)
                pid = tl.load(par_pid + p).to(tl.int64)
                base = tl.load(reg_offset + reg) + (nid - tl.load(reg_first + reg)) * tl.load(reg_width + reg)
                width = tl.load(reg_width + reg)
                if GATED and not GATE_TILE:                                   # per node gate: its own sum, times
                    grow = tl.load(par_grow + p)                              # its gates, log-added
                    for g0 in range(0, BS, NG):
                        m = tl.full([TILE_B], float("-inf"), dtype = tl.float32)
                        acc = tl.zeros([TILE_C, TILE_B], dtype = tl.float32)
                        for i0 in range(g0, g0 + NG, TILE_I):
                            offs_i = i0 + tl.arange(0, TILE_I)
                            im = (offs_i < g0 + NG) & (offs_i < BS)
                            w = tl.load(weights + pid + offs_c[:, None] * BS + offs_i[None, :],
                                        mask = cm[:, None] & im[None, :], other = 0.0)        # [TILE_C, TILE_I]
                            if PRE:                                   # the pre-pass's chunk: log Z already off
                                e, t = _pre_chunk(pre_buf, pre_cm, pre_ld, p, offs_i, (g0 // NG) * NCPG +
                                                  (i0 - g0) // TILE_I, im, k, cmask, BS, NCH)
                                m, acc = _lse_dot_step_pre(m, acc, e, t, w, PRECISION)
                            else:
                                x = tl.load(node_flows + base + offs_i.to(tl.int64)[:, None] * width + cols[None, :],
                                            mask = im[:, None] & cmask[None, :], other = float("-inf"))
                                x -= _node_log_z(lz, samp, lzstride, nid + offs_i, im, cmask, SHARED_GATES)
                                m, acc = _lse_dot_step(m, acc, x, w, PRECISION)
                        m2, acc2 = _add_gated_parent(m2, acc2, m, acc, gates, samp, gstride,
                                                     grow + (g0 // NG) * tl.load(par_gck + p), tl.load(par_lc + p),
                                                     tl.load(par_gcbs + p), offs_c, cm, cmask, SHARED_GATES)
                else:
                    if GATED:                                                 # the tile's one child gate
                        gcell = tl.load(par_grow + p) + (tl.load(par_lc + p) + ct * TILE_C) // tl.load(par_gcbs + p)
                        if NG == BS:                                          # ... and node gate: per column
                            gb = _tile_gates(gates, samp, gstride, tl.load(par_grow + p),
                                             tl.load(par_lc + p) + ct * TILE_C, tl.load(par_gcbs + p), cmask,
                                             SHARED_GATES)
                        else:                                                 # ... per row and column
                            gck = tl.load(par_gck + p)
                    for i0 in range(0, BS, TILE_I):
                        offs_i = i0 + tl.arange(0, TILE_I)
                        im = offs_i < BS
                        w = tl.load(weights + pid + offs_c[:, None] * BS + offs_i[None, :],
                                    mask = cm[:, None] & im[None, :], other = 0.0)            # [TILE_C, TILE_I]
                        if PRE:                                       # the pre-pass's chunk (never row gates)
                            e, t = _pre_chunk(pre_buf, pre_cm, pre_ld, p, offs_i, i0 // TILE_I, im, k, cmask, BS,
                                              NCH)
                            if GATED:
                                t += gb
                            m, acc = _lse_dot_step_pre(m, acc, e, t, w, PRECISION)
                        else:
                            x = tl.load(node_flows + base + offs_i.to(tl.int64)[:, None] * width + cols[None, :],
                                        mask = im[:, None] & cmask[None, :], other = float("-inf"))   # [TILE_I, TILE_B]
                            if GATED:
                                x -= _node_log_z(lz, samp, lzstride, nid + offs_i, im, cmask, SHARED_GATES)
                                if NG < BS:
                                    x += _row_gates(gates, gcell, (offs_i // NG) * gck, samp, gstride, im, cmask,
                                                    SHARED_GATES)
                            if GATED and NG == BS:
                                m, acc = _lse_dot_step_biased(m, acc, x, w, gb, PRECISION)
                            else:
                                m, acc = _lse_dot_step(m, acc, x, w, PRECISION)
        if GATED and not GATE_TILE:
            out = tl.log(acc2) + tl.where(m2 == float("-inf"), 0.0, m2)
        else:
            out = tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)[None, :]
        mask = cm[:, None] & kmask[None, :]
        if SPLIT > 1:                                     # a share of the parents: the last one in merges them all
            out, last = _split_merge(out, split_buf, pid_t + pid_c * (tl.num_programs(0) // SPLIT), s, TILE_C,
                                     TILE_B, SPLIT)
            mask = mask & last
        row = tl.load(cb_row + cb).to(tl.int64)
        creg = tl.load(cb_reg + cb)
        add = tl.load(cb_add + cb)
        if creg < 0:                                                          # element rows
            ptr = element_flows + (row - elem_first + offs_c.to(tl.int64))[:, None] * elem_width + cols[None, :]
        else:                                                                 # an aliased sum's rows
            crow = row - tl.load(reg_first + creg) + offs_c.to(tl.int64)
            ptr = node_flows + tl.load(reg_offset + creg) + crow[:, None] * tl.load(reg_width + creg) + cols[None, :]
        if add:
            out = _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), out)
        tl.store(ptr, out, mask = mask)


@triton_jit
def _lifted_sum_top_down_kernel(element_flows, node_flows, weights, cb_row, cb_reg, cb_add, cb_ivreg, cb_idx, par_ptr,
                                par_nid, par_reg, par_pid, reg_offset, reg_width, reg_first, reg_slots, perm, reg_perm,
                                reg_live, par_lc, par_grow, par_gcbs, par_gck, gates, gstride, lz, lzstride, coff,
                                reg_ivk, Bq, steps, elem_first, elem_width, B, pid0, BS: tl.constexpr,
                                CBS: tl.constexpr, TILE_C: tl.constexpr, TILE_I: tl.constexpr, TILE_B: tl.constexpr,
                                PRECISION: tl.constexpr, PRUNE: tl.constexpr, INDEXED: tl.constexpr,
                                GATED: tl.constexpr, LANES: tl.constexpr, GATE_TILE: tl.constexpr,
                                SHARED_GATES: tl.constexpr, NG: tl.constexpr):
    """:func:`_lifted_sum_top_down_body`, unsplit, without a pre-pass."""
    _lifted_sum_top_down_body(element_flows = element_flows, node_flows = node_flows, weights = weights,
                              cb_row = cb_row, cb_reg = cb_reg, cb_add = cb_add, cb_ivreg = cb_ivreg, cb_idx = cb_idx,
                              par_ptr = par_ptr, par_nid = par_nid, par_reg = par_reg, par_pid = par_pid,
                              reg_offset = reg_offset, reg_width = reg_width, reg_first = reg_first,
                              reg_slots = reg_slots, perm = perm, reg_perm = reg_perm, reg_live = reg_live,
                              par_lc = par_lc, par_grow = par_grow, par_gcbs = par_gcbs, par_gck = par_gck,
                              gates = gates, gstride = gstride, lz = lz, lzstride = lzstride, coff = coff,
                              reg_ivk = reg_ivk, Bq = Bq, steps = steps, split_buf = element_flows,
                              pre_buf = element_flows, pre_cm = 0, pre_ld = 0, elem_first = elem_first,
                              elem_width = elem_width, B = B, pid0 = pid0, BS = BS, CBS = CBS, TILE_C = TILE_C,
                              TILE_I = TILE_I, TILE_B = TILE_B, PRECISION = PRECISION, PRUNE = PRUNE, INDEXED = INDEXED,
                              GATED = GATED, LANES = LANES, GATE_TILE = GATE_TILE, SHARED_GATES = SHARED_GATES, NG = NG,
                              SPLIT = 1, PRE = False)


@triton_jit
def _lifted_sum_top_down_aux_kernel(element_flows, node_flows, weights, cb_row, cb_reg, cb_add, cb_ivreg, cb_idx,
                                    par_ptr, par_nid, par_reg, par_pid, reg_offset, reg_width, reg_first, reg_slots,
                                    perm, reg_perm, reg_live, par_lc, par_grow, par_gcbs, par_gck, gates, gstride, lz,
                                    lzstride, coff, reg_ivk, Bq, steps, split_buf, pre_buf, pre_cm, pre_ld, elem_first,
                                    elem_width, B, pid0, BS: tl.constexpr, CBS: tl.constexpr, TILE_C: tl.constexpr,
                                    TILE_I: tl.constexpr, TILE_B: tl.constexpr, PRECISION: tl.constexpr,
                                    PRUNE: tl.constexpr, INDEXED: tl.constexpr, GATED: tl.constexpr,
                                    LANES: tl.constexpr, GATE_TILE: tl.constexpr, SHARED_GATES: tl.constexpr,
                                    NG: tl.constexpr, SPLIT: tl.constexpr, PRE: tl.constexpr):
    """:func:`_lifted_sum_top_down_body` split ``SPLIT`` ways (``split_buf``: :func:`split_buffer`) and / or with
    ``PRE``, the pre-pass's chunks (:func:`_top_down_prepass_kernel`, ``pre_buf``)."""
    _lifted_sum_top_down_body(element_flows = element_flows, node_flows = node_flows, weights = weights,
                              cb_row = cb_row, cb_reg = cb_reg, cb_add = cb_add, cb_ivreg = cb_ivreg, cb_idx = cb_idx,
                              par_ptr = par_ptr, par_nid = par_nid, par_reg = par_reg, par_pid = par_pid,
                              reg_offset = reg_offset, reg_width = reg_width, reg_first = reg_first,
                              reg_slots = reg_slots, perm = perm, reg_perm = reg_perm, reg_live = reg_live,
                              par_lc = par_lc, par_grow = par_grow, par_gcbs = par_gcbs, par_gck = par_gck,
                              gates = gates, gstride = gstride, lz = lz, lzstride = lzstride, coff = coff,
                              reg_ivk = reg_ivk, Bq = Bq, steps = steps, split_buf = split_buf, pre_buf = pre_buf,
                              pre_cm = pre_cm, pre_ld = pre_ld, elem_first = elem_first, elem_width = elem_width, B = B,
                              pid0 = pid0, BS = BS, CBS = CBS, TILE_C = TILE_C, TILE_I = TILE_I, TILE_B = TILE_B,
                              PRECISION = PRECISION, PRUNE = PRUNE, INDEXED = INDEXED, GATED = GATED, LANES = LANES,
                              GATE_TILE = GATE_TILE, SHARED_GATES = SHARED_GATES, NG = NG, SPLIT = SPLIT, PRE = PRE)


@triton_jit
def _top_down_prepass_kernel(node_flows, pre_buf, pre_cm, pre_ld, cb_ivreg, cb_idx, par_ptr, par_nid, par_reg,
                             reg_offset, reg_width, reg_first, reg_slots, perm, reg_perm, reg_live, lz, lzstride, coff,
                             reg_ivk, Bq, steps, B, max_par, pid0, BS: tl.constexpr, NGC: tl.constexpr,
                             TILE_I: tl.constexpr, TILE_B: tl.constexpr, PRUNE: tl.constexpr, INDEXED: tl.constexpr,
                             GATED: tl.constexpr, LANES: tl.constexpr, SHARED_GATES: tl.constexpr,
                             PRECISION: tl.constexpr, REL: tl.constexpr = False, FRAG: tl.constexpr = 0):
    """One program per (child block -- of the index list ``cb_idx`` if ``INDEXED`` --, its ``j``-th parent entry
    ``p``, chunk ``c`` of the parent's rows, tile of columns): the chunk's values -- less their log Z of the column's
    sample when ``GATED`` --, their maximum per column ``t`` and the shifted exponentials ``e = exp(x - t)``, stored at
    ``pre_buf[(p * BS + row) * pre_ld + k]`` and ``pre_buf[pre_cm + (p * NCH + c) * pre_ld + k]`` by column position
    ``k`` (pruned: the interval's live columns first, as the top-down kernel reads them). The chunks are the top-down
    kernel's: ``TILE_I`` rows within every group of ``NGC`` (its node gate, or the block). ``REL``: stored by the
    entry's place ``j`` among the child block's (every child block's parents the same: the CUDA kernel's, one child
    block's programs) -- e, for the CUDA kernel (chunks of 32 rows), in its order: ``FRAG`` 1 (:func:`_frag_offsets`),
    or 2 in bf16 (:func:`_frag_offsets_bf16`); e rounded for the products' ``PRECISION``
    (:func:`~.common._tf32_operand`), which read it as it is."""
    NCPG: tl.constexpr = (NGC + TILE_I - 1) // TILE_I
    NCH: tl.constexpr = (BS // NGC) * NCPG
    pid = tl.program_id(0)
    c = pid % NCH
    pj = pid // NCH
    j = pj % max_par
    cb = pj // max_par
    if INDEXED:
        cb = tl.load(cb_idx + cb)
    p0 = tl.load(par_ptr + cb)
    ivreg = tl.load(cb_ivreg + cb)
    ncols = B * tl.load(reg_slots + ivreg)
    col0 = (pid0 + tl.program_id(1)) * TILE_B
    if (j < tl.load(par_ptr + cb + 1) - p0) & (col0 < ncols):
        p = p0 + j
        k = col0 + tl.arange(0, TILE_B)
        kmask = k < ncols
        if PRUNE:
            num_live = tl.load(reg_live + ivreg)
            cols = tl.load(perm + tl.load(reg_perm + ivreg) + k, mask = kmask, other = 0)
            cmask = kmask & (k < num_live)
        else:
            cols = k
            cmask = kmask
        samp = cols
        if GATED and not SHARED_GATES:
            samp = _column_sample(cols, ivreg, reg_slots, coff, reg_ivk, Bq, steps, LANES)
        nid = tl.load(par_nid + p).to(tl.int64)
        reg = tl.load(par_reg + p)
        width = tl.load(reg_width + reg)
        base = tl.load(reg_offset + reg) + (nid - tl.load(reg_first + reg)) * width
        g0 = (c // NCPG) * NGC
        offs_i = g0 + (c % NCPG) * TILE_I + tl.arange(0, TILE_I)
        im = (offs_i < g0 + NGC) & (offs_i < BS)
        x = tl.load(node_flows + base + offs_i.to(tl.int64)[:, None] * width + cols[None, :],
                    mask = im[:, None] & cmask[None, :], other = float("-inf"))
        if GATED:
            x -= _node_log_z(lz, samp, lzstride, nid + offs_i, im, cmask, SHARED_GATES)
        t = tl.max(x, axis = 0)
        e = tl.exp(x - tl.where(t == float("-inf"), 0.0, t)[None, :])
        p64 = (j if REL else p).to(tl.int64)
        if FRAG == 2:
            tl.store(pre_buf.to(tl.pointer_type(tl.bfloat16)) + p64 * BS * pre_ld + _frag_offsets_bf16(offs_i, k),
                     e.to(tl.bfloat16), mask = im[:, None] & kmask[None, :])
        else:
            if FRAG == 1:
                at = p64 * BS * pre_ld + _frag_offsets(offs_i, k)
            else:
                at = (p64 * BS + offs_i)[:, None] * pre_ld + k[None, :]
            tl.store(pre_buf + at, _tf32_operand(e, PRECISION), mask = im[:, None] & kmask[None, :])
        tl.store(pre_buf + pre_cm + (p64 * NCH + c) * pre_ld + k, t, mask = kmask)


def fused_sum_top_down(element_flows: torch.Tensor, node_flows: torch.Tensor, params: torch.Tensor, tables: dict,
                      regions: dict, elem_first: int, elem_width: int, batch_size: int, block_size: int, cbs: int,
                      max_cols: int, precision: str, liveness = None, idx: Optional[torch.Tensor] = None,
                      gated: Optional[dict] = None, frags: Optional[dict] = None, bf16: bool = False):
    """
    The top-down values of the children of a sum layer's fused node blocks (:func:`_lifted_sum_top_down_kernel`) --
    or, child blocks of 32 children or a multiple under the same parent node blocks of 32, the CUDA kernel
    (:func:`_cuda_td_plan`).

    :param tables: child-block-major: ``cb_row``, ``cb_reg`` (-1 for element rows), ``cb_add``, ``cb_ivreg`` (a region
        of the children's interval, for pruning), ``par_ptr``, ``par_nid``, ``par_reg``, ``par_pid`` (int32 / int64
        device tensors, see :mod:`.queries.backend.top_down`), and a gated layer's ``gate``: its parents' ``lc``,
        ``row`` and ``cbs`` (see :mod:`..compiled.top_down_program`)
    :param cbs: the child block size of these children
    :param max_cols: the most columns any child block has (``batch_size`` x its slots)
    :param idx: None (every child block), or int32 on the device: the plan's index list of the child blocks to compute
    :param gated: None, or the session's gates (:func:`~..queries.backend.forward.part_gates`):
        the layer's gates applied
    :param frags: None, or where the CUDA kernel's fragment-ordered weights are kept (:func:`_cuda_fragments`)
    :param bf16: a CUDA launch's products on bf16 operands (see :func:`fused_sum`)
    """
    num_blocks = tables["cb_row"].numel() if idx is None else idx.numel()
    tm, te, tb, warps, stages = _TILES[precision]
    TILE_C = min(tm, max(16, host_next_power_of_2(cbs)))
    TILE_I = min(te, max(16, host_next_power_of_2(block_size)))
    TILE_B = min(tb, max(16, host_next_power_of_2(max_cols)))
    gt = tables["gate"] if gated is not None else None
    if gt is not None:                                   # a tile of children within one gate, if one fits
        TILE_C = _gate_tile(TILE_C, gt["uniform"])
        if gt["node_gate"] < block_size and not gt["uniform"].get(TILE_C, False):   # a sum per node gate: its rows
            TILE_I = min(TILE_I, max(16, gt["node_gate"]))                         # at once, when 16 or more
    num_ctiles = host_cdiv(cbs, TILE_C)
    num_col_tiles = host_cdiv(max_cols, TILE_B)
    plan = _cuda_td_plan(tables, gt, idx, block_size, cbs, max_cols, TILE_I, precision, element_flows.device, params,
                         frags, bf16)
    if plan is not None:
        _cuda_top_down(plan, element_flows, node_flows, params, tables, regions, elem_first, elem_width, batch_size,
                       max_cols, precision, liveness, idx, gated, gate_launch(gated, tables["cb_row"])[3:])
        return
    warps = _tile_warps(warps, TILE_C)
    tiles = num_blocks * num_ctiles * num_col_tiles
    split, buf, split_kw = 1, (), _NO_SPLIT                 # unsplit: the kernel without the split's arguments
    chunks = tables["max_parents"] * host_cdiv(block_size, TILE_I)
    if chunks >= 2 * SPLIT_MIN_SHARE and tiles <= SPLIT_TILES and \
            tiles < SPLIT_PROGRAMS_PER_SM * _num_sms(element_flows.device):              # a launch that may split
        split = split_count(tiles, chunks, element_flows.device)
        if split > 1:
            buf, split_kw = (split_buffer(element_flows.device, tiles, split * TILE_C * TILE_B),), dict(SPLIT = split)
    lanes = liveness.lanes if liveness is not None else None
    perm, reg_perm, reg_live = (lanes.perm, lanes.reg_first, lanes.reg_live) if lanes is not None else \
        (tables["cb_row"], tables["cb_row"], tables["cb_row"])
    g = (gt["lc"], gt["row"], gt["cbs"], gt["ck"]) if gt is not None else (tables["cb_row"],) * 4
    gate_tile = gt is not None and gt["uniform"].get(TILE_C, False)
    NG = gt["node_gate"] if gt is not None else block_size
    gl = gate_launch(gated, tables["cb_row"])[3:]
    aux, aux_kw = (), _NO_SPLIT
    if num_ctiles >= PRE_MIN_TILES and not (gate_tile and NG < block_size):   # the parents' chunks once: a pre-pass
        NGC = NG if gt is not None and not gate_tile else block_size            # (row gates are a tile's own)
        nch = (block_size // NGC) * host_cdiv(NGC, TILE_I)
        entries = tables["par_nid"].numel()
        pre_ld = host_cdiv(max_cols, 4) * 4
        pre = pre_buffer(element_flows.device, entries * (block_size + nch) * pre_ld)
        for first, size in grid_chunks(num_col_tiles, 1):
            _top_down_prepass_kernel[(num_blocks * tables["max_parents"] * nch, size)](
                node_flows, pre, entries * block_size * pre_ld, pre_ld, tables["cb_ivreg"],
                tables["cb_row"] if idx is None else idx, tables["par_ptr"], tables["par_nid"], tables["par_reg"],
                regions["offset"], regions["width"], regions["first"], regions["slots"], perm, reg_perm, reg_live,
                gl[2], gl[3], gl[4], gl[5], gl[6], gl[7], batch_size, tables["max_parents"], first, BS = block_size,
                NGC = NGC, TILE_I = TILE_I, TILE_B = TILE_B, PRUNE = liveness is not None, INDEXED = idx is not None,
                GATED = gated is not None, LANES = gated is not None and gated["lanes"] is not None,
                SHARED_GATES = gated is not None and gated["gstride"] == 0, PRECISION = precision, num_warps = 4)
        aux, aux_kw = (pre, entries * block_size * pre_ld, pre_ld), dict(PRE = True)
    if split > 1 or aux:
        aux = (buf[0] if buf else element_flows,) + (aux or (element_flows, 0, 0))
        aux_kw = dict(SPLIT = split, PRE = bool(aux_kw))
    for first, size in grid_chunks(num_col_tiles, 1):
        (_lifted_sum_top_down_aux_kernel if aux else _lifted_sum_top_down_kernel)[
                (num_blocks * num_ctiles * split, size)](
            element_flows, node_flows, params, tables["cb_row"], tables["cb_reg"], tables["cb_add"], tables["cb_ivreg"],
            tables["cb_row"] if idx is None else idx, tables["par_ptr"], tables["par_nid"], tables["par_reg"],
            tables["par_pid"], regions["offset"], regions["width"], regions["first"], regions["slots"], perm, reg_perm,
            reg_live, *g, *gl, *aux, elem_first, elem_width, batch_size, first, BS = block_size, CBS = cbs,
            TILE_C = TILE_C, TILE_I = TILE_I, TILE_B = TILE_B, PRECISION = precision, PRUNE = liveness is not None,
            INDEXED = idx is not None, GATED = gated is not None,
            LANES = gated is not None and gated["lanes"] is not None, GATE_TILE = gate_tile,
            SHARED_GATES = gated is not None and gated["gstride"] == 0, NG = NG, **aux_kw, num_warps = warps,
            num_stages = stages)


@triton_jit
def _log_accumulate_kernel(out, prod, m, rows, ncols, ld, num_ctiles, TILE_R: tl.constexpr, TILE_C: tl.constexpr):
    """``out = logaddexp(out, log(prod) + m)`` (``prod`` contiguous [rows, ncols])."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    mask = (r < rows)[:, None] & cmask[None, :]
    shift = tl.load(m + c, mask = cmask, other = 0.0)
    r = r.to(tl.int64)
    v = tl.log(tl.load(prod + r[:, None] * ncols + c[None, :], mask = mask, other = 0.0)) + shift[None, :]
    ptr = out + r[:, None] * ld + c[None, :]
    tl.store(ptr, _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), v), mask = mask)


@triton_jit
def _log_scatter_accumulate_kernel(out, prod, m, rank, rows, ncols, ld, num_live, num_ctiles, TILE_R: tl.constexpr,
                                   TILE_C: tl.constexpr):
    """``out[:, c] = logaddexp(out[:, c], log(prod[:, rank[c]]) + m[rank[c]])`` for every live column ``c``; dead
    columns untouched (they hold -inf from the first writer)."""
    r = (tl.program_id(0) // num_ctiles) * TILE_R + tl.arange(0, TILE_R)
    c = (tl.program_id(0) % num_ctiles) * TILE_C + tl.arange(0, TILE_C)
    cmask = c < ncols
    k = tl.load(rank + c, mask = cmask, other = -1)
    alive = k >= 0
    mask = (r < rows)[:, None] & (cmask & alive)[None, :]
    shift = tl.load(m + k, mask = cmask & alive, other = 0.0)
    r = r.to(tl.int64)
    v = tl.log(tl.load(prod + r[:, None] * num_live + k[None, :], mask = mask, other = 0.0)) + shift[None, :]
    ptr = out + r[:, None] * ld + c[None, :]
    tl.store(ptr, _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), v), mask = mask)


def dense_sum_top_down(element_flows: torch.Tensor, node_flows: torch.Tensor, params: torch.Tensor, blocks, E: int,
                      block_size: int, region: tuple, c0: int, elem_first: int, elem_width: int, ncols: int,
                      precision: str, add: bool, live = None, stacked = None, piece = None):
    """
    The top-down values of the ``E`` children ``c0 ..`` of a run of dense node blocks of one sum region:
    ``O_elem = log(W^T @ exp(O_node - m)) + m`` with ``m`` the column maximum of the node blocks' top-down values,
    written to ``element_flows``, or log-added to it (``add``); ``W^T`` is the parameter buffer's own ``[E, BS]``
    view per block (``stacked``: the blocks' :func:`stacked_weights`, transposed for free).

    :param blocks: ``[(k, nid, p0)]`` as in :func:`dense_sum`'s groups (consecutive rows when ``stacked``)
    :param region: ``(offset, width, first)`` of the node blocks' sum region
    :param live: None, or the interval's ``(perm, rank, num_live)`` (see :meth:`~.live.LaneOrder.segment`)
    :param piece: None, or ``(lo, hi)``: only the children ``c0 + lo .. c0 + hi``
    """
    offset, width, first = region
    BS = block_size
    lo, hi = piece if piece is not None else (0, E)
    elems = element_flows[:element_flows.numel() // elem_width * elem_width].view(-1, elem_width)
    nodes = node_flows[offset:offset + (node_flows.numel() - offset) // width * width].view(-1, width)
    out = elems[c0 + lo - elem_first:c0 + hi - elem_first]
    E = hi - lo
    TILE_R, TILE_C = _EW_TILE
    products = [(stacked.t()[lo:hi], blocks[0][1], stacked.size(0))] if stacked is not None else \
        [(params[p0 + lo * BS:p0 + hi * BS].view(E, BS), nid, BS) for _, nid, p0 in blocks]
    with _matmul_precision(precision):
        for wt, nid, rows in products:
            x = nodes[nid - first:nid - first + rows, :ncols]
            if live is None:
                m = x.amax(dim = 0)
                ex = torch.empty(rows, ncols, dtype = x.dtype, device = x.device)
                num_ctiles = host_cdiv(ncols, TILE_C)
                _shifted_exp_kernel[(host_cdiv(rows, TILE_R) * num_ctiles,)](x, ex, m, rows, ncols, width, num_ctiles,
                                                                            TILE_R = TILE_R, TILE_C = TILE_C)
                if add:
                    prod = torch.mm(wt, ex)
                    _log_accumulate_kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](
                        out, prod, m, E, ncols, elem_width, num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C)
                else:
                    torch.mm(wt, ex, out = out[:, :ncols])
                    _log_add_kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](out, m, E, ncols, elem_width, num_ctiles,
                                                                         TILE_R = TILE_R, TILE_C = TILE_C)
            else:
                perm, rank, num_live = live
                ex_t, m = _live_shifted_exp(x, perm, num_live)                                  # [live, rows]
                prod = torch.mm(wt, ex_t.t())                                                   # [E, live]
                num_ctiles = host_cdiv(ncols, TILE_C)
                kernel = _log_scatter_accumulate_kernel if add else _log_scatter_kernel
                kernel[(host_cdiv(E, TILE_R) * num_ctiles,)](out, prod, m, rank, E, ncols, elem_width, num_live,
                                                            num_ctiles, TILE_R = TILE_R, TILE_C = TILE_C)
            add = True                                                        # the run's later blocks add to the first


@triton_jit
def _fill_dead_rows_kernel(node_flows, rows, reg_offset, reg_width, reg_first, num_tiles, TILE: tl.constexpr):
    """Sum rows of ``node_flows`` set to -inf over their whole width: one program per (``(row, region)`` of ``rows``,
    tile of ``TILE`` columns)."""
    k = tl.program_id(0) // num_tiles
    row = tl.load(rows + k * 2)
    reg = tl.load(rows + k * 2 + 1)
    width = tl.load(reg_width + reg)
    base = node_flows + tl.load(reg_offset + reg) + (row - tl.load(reg_first + reg)).to(tl.int64) * width
    c = (tl.program_id(0) % num_tiles) * TILE + tl.arange(0, TILE)
    tl.store(base + c, tl.full([TILE], float("-inf"), dtype = tl.float32), mask = c < width)


def fill_dead_rows(node_flows: torch.Tensor, rows: torch.Tensor, regions: dict, max_width: int):
    """
    The top-down value -inf on the sum rows no chain of edges from the root reaches (no launch writes them).

    :param rows: int32 [D, 2], ``(row, region)`` per dead row
    :param regions: ``offset``, ``width`` and ``first`` of every sum region (device tensors)
    :param max_width: the widest of their regions' rows
    """
    TILE = 1024
    num_tiles = host_cdiv(max_width, TILE)
    _fill_dead_rows_kernel[(rows.size(0) * num_tiles,)](node_flows, rows, regions["offset"], regions["width"],
                                                        regions["first"], num_tiles, TILE = TILE)
