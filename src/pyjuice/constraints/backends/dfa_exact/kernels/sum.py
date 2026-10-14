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
from typing import Optional

import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ..external_params.block_scale import (_add_gated_parent, _chunk_gates, _edge_gates, _node_log_z, _row_gates,
                                           _segment_cell, _tile_gates)
from .common import _child, _logaddexp_tile, grid_chunks
from .compact import _lane_sample


#: The fused kernel's tiles (TILE_M, TILE_E, TILE_B, num_warps, num_stages) per product precision, from a sweep on
#: a 4096-state HMM layer (4 x 1024 nodes, 4096 children) at 69 to 8832 columns.
_TILES = {"tf32": (128, 32, 128, 8, 3), "tf32x3": (64, 32, 64, 4, 3), "bf16x3": (64, 32, 64, 4, 3)}

#: Dense node blocks smaller than this run the fused kernel: one cuBLAS call per node block only pays for itself
#: on large blocks (a partition of many small blocks is one fused launch instead).
DENSE_MIN_BLOCK = 64


@triton.jit
def _lse_dot_step(m, acc, x, w, PRECISION: tl.constexpr):
    """One chunk of children of the online log-sum-exp: ``x`` [children, columns] in log space, ``w`` [nodes,
    children] the weights."""
    m_new = tl.maximum(m, tl.max(x, axis = 0))
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new)
    return m_new, acc * scale[None, :] + tl.dot(w, tl.exp(x - shift[None, :]), input_precision = PRECISION)


@triton.jit
def _lse_dot_step_biased(m, acc, x, w, bias, PRECISION: tl.constexpr):
    """:func:`_lse_dot_step` of ``x + bias[None, :]`` (``bias`` [columns], a chunk's gate per column, see
    :mod:`..external_params.block_scale`), the bias taken into the shift rather than added to
    every element."""
    m_new = tl.maximum(m, tl.max(x, axis = 0) + bias)
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new - bias)
    return m_new, acc * scale[None, :] + tl.dot(w, tl.exp(x - shift[None, :]), input_precision = PRECISION)


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
    from a part's gates (see :func:`~..queries.forward.part_gates`; ``dummy`` where nothing is read)."""
    if gated is None:
        return (dummy,) * 3 + (dummy, 0, dummy, 0, dummy, dummy, 1, 1)
    off, reg_ivk, Bq, steps = gated["lanes"] if gated["lanes"] is not None else (dummy, dummy, 1, 1)
    return (idx if idx is not None else dummy, a_idx if a_idx is not None else dummy, ck if ck is not None else dummy,
            gated["values"], gated["gstride"], gated["lz"], gated["lzstride"], off, reg_ivk, Bq, steps)


@triton_jit
def _lifted_sum_kernel(node_mars, element_mars, weights, nids, cids, pids, nb_reg, sel,
                       reg_offset, reg_width, reg_first, reg_slots, a_rows, a_regs, a_pids, perm, reg_perm, reg_live,
                       gidx, a_gidx, gck, gates, gstride, lz, lzstride, coff, reg_ivk, Bq, steps,
                       elem_first, elem_width, B, num_edges, num_alias, num_mtiles, pid0,
                       BS: tl.constexpr, TILE_M: tl.constexpr, TILE_E: tl.constexpr, TILE_A: tl.constexpr,
                       TILE_B: tl.constexpr, PRECISION: tl.constexpr, ALIAS: tl.constexpr, PRUNE: tl.constexpr,
                       INDEXED: tl.constexpr, GATED: tl.constexpr, LANES: tl.constexpr, CHUNK_GATE: tl.constexpr,
                       A_CHUNK_GATE: tl.constexpr, SHARED_GATES: tl.constexpr, NG: tl.constexpr, NGT: tl.constexpr,
                       SEG: tl.constexpr, A_SEG: tl.constexpr):
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
    (``A_SEG``) edges of one cell each (:func:`_lse_dot_step_rows`)."""
    pid_nm = tl.program_id(0)                                     # (node block, tile of its nodes)
    pid_c = pid0 + tl.program_id(1)                               # tile of columns
    pid_n = _child(sel, pid_nm // num_mtiles, INDEXED)
    mt = pid_nm % num_mtiles
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
        for e0 in range(0, n_edges, TILE_E):
            offs_e = e0 + tl.arange(0, TILE_E)
            emask = offs_e < num_edges
            cid = tl.load(cids + pid_n * num_edges + offs_e, mask = emask, other = 0).to(tl.int64)
            pid = tl.load(pids + pid_n * num_edges + offs_e, mask = emask, other = 0).to(tl.int64)
            valid = emask & (cid >= elem_first)
            x = tl.load(element_mars + (cid - elem_first)[:, None] * elem_width + cols[None, :],
                        mask = valid[:, None] & cmask[None, :], other = float("-inf"))           # [TILE_E, TILE_B]
            if GATED and not CHUNK_GATE and NGT == 1:
                x += _edge_gates(gates, gidx + pid_n * num_edges, offs_e, emask, samp, gstride, cmask, SHARED_GATES)
            w = tl.load(weights + pid[None, :] + offs_m[:, None],
                        mask = mmask[:, None] & valid[None, :], other = 0.0)                    # [TILE_M, TILE_E]
            if GATED and NGT > 1:                                          # gates per row and column
                m2, acc = _lse_dot_step_rows(m2, acc, x, w, gates, gidx + pid_n * num_edges, offs_e, emask, samp,
                                             gstride, goff, mmask, cmask, PRECISION, TILE_E, SEG, SHARED_GATES)
            elif GATED and CHUNK_GATE:                                     # one gate per column
                m, acc = _lse_dot_step_biased(m, acc, x, w, _chunk_gates(gates, gidx + pid_n * num_edges, offs_e, emask,
                                                                         samp, gstride, cmask, SHARED_GATES), PRECISION)
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
        nid = tl.load(nids + pid_n).to(tl.int64)
        if GATED:
            out -= _node_log_z(lz, samp, lzstride, nid + offs_m, mmask, kmask, SHARED_GATES)
        width = tl.load(reg_width + reg)
        base = tl.load(reg_offset + reg) + (nid - tl.load(reg_first + reg)) * width
        tl.store(node_mars + base + offs_m.to(tl.int64)[:, None] * width + cols[None, :], out,
                 mask = mmask[:, None] & kmask[None, :])


def fused_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, nids: torch.Tensor,
              cids: torch.Tensor, pids: torch.Tensor, nb_reg: torch.Tensor, regions: dict, elem_first: int,
              elem_width: int, batch_size: int, block_size: int, max_cols: int, precision: str, aliased = None,
              liveness = None, sel = None, count: Optional[int] = None, gated: Optional[dict] = None):
    """
    Node blocks of a sum layer partition with the fused Triton kernel, into ``node_mars``.

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
    :param gated: None, or the gates of a gated layer's node blocks (:func:`~..queries.forward.part_gates`): ``idx``
        / ``a_idx`` (the edges' cells, as ``cids`` / the aliased children), the session's gates and normalizers
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
    for first, size in grid_chunks(host_cdiv(max_cols, TILE_B), 1):
        _lifted_sum_kernel[(num_blocks * num_mtiles, size)](
            node_mars, element_mars, params, nids, cids, pids, nb_reg, nids if sel is None else sel, regions["offset"],
            regions["width"],
            regions["first"], regions["slots"], a_rows, a_regs, a_pids, perm, reg_perm, reg_live, *g, elem_first,
            elem_width, batch_size, num_edges, num_alias, num_mtiles, first, BS = block_size, TILE_M = TILE_M,
            TILE_E = TILE_E, TILE_A = TILE_A, TILE_B = TILE_B, PRECISION = precision, ALIAS = aliased is not None,
            PRUNE = liveness is not None, INDEXED = sel is not None, GATED = gated is not None,
            LANES = gated is not None and gated["lanes"] is not None,
            CHUNK_GATE = gated is not None and gated["uniform"].get(TILE_E, False),
            A_CHUNK_GATE = gated is not None and aliased is not None and gated["a_uniform"].get(TILE_A, False),
            SHARED_GATES = gated is not None and gated["gstride"] == 0, NG = NG, NGT = NGT,
            SEG = _gate_segment(TILE_E, gated["uniform"]) if gated is not None else 1,
            A_SEG = _gate_segment(TILE_A, gated["a_uniform"]) if gated is not None and aliased is not None else 1,
            num_warps = warps, num_stages = stages)


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


@triton_jit
def _lifted_sum_top_down_kernel(element_flows, node_flows, weights, cb_row, cb_reg, cb_add, cb_ivreg, cb_idx, par_ptr,
                               par_nid, par_reg, par_pid, reg_offset, reg_width, reg_first, reg_slots, perm, reg_perm,
                               reg_live, par_lc, par_grow, par_gcbs, par_gck, gates, gstride, lz, lzstride, coff,
                               reg_ivk, Bq, steps, elem_first, elem_width, B, pid0, BS: tl.constexpr,
                               CBS: tl.constexpr, TILE_C: tl.constexpr, TILE_I: tl.constexpr, TILE_B: tl.constexpr,
                               PRECISION: tl.constexpr, PRUNE: tl.constexpr, INDEXED: tl.constexpr,
                               GATED: tl.constexpr, LANES: tl.constexpr, GATE_TILE: tl.constexpr,
                               SHARED_GATES: tl.constexpr, NG: tl.constexpr):
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
    parent values per row with ``GATE_TILE``, else a sum of its own per node gate."""
    pid_cm = tl.program_id(0)
    pid_c = pid0 + tl.program_id(1)
    cb = pid_cm // ((CBS + TILE_C - 1) // TILE_C)
    ct = pid_cm % ((CBS + TILE_C - 1) // TILE_C)
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
        row = tl.load(cb_row + cb).to(tl.int64)
        creg = tl.load(cb_reg + cb)
        add = tl.load(cb_add + cb)
        if creg < 0:                                                          # element rows
            ptr = element_flows + (row - elem_first + offs_c.to(tl.int64))[:, None] * elem_width + cols[None, :]
        else:                                                                 # an aliased sum's rows
            crow = row - tl.load(reg_first + creg) + offs_c.to(tl.int64)
            ptr = node_flows + tl.load(reg_offset + creg) + crow[:, None] * tl.load(reg_width + creg) + cols[None, :]
        mask = cm[:, None] & kmask[None, :]
        if add:
            out = _logaddexp_tile(tl.load(ptr, mask = mask, other = float("-inf")), out)
        tl.store(ptr, out, mask = mask)


def fused_sum_top_down(element_flows: torch.Tensor, node_flows: torch.Tensor, params: torch.Tensor, tables: dict,
                      regions: dict, elem_first: int, elem_width: int, batch_size: int, block_size: int, cbs: int,
                      max_cols: int, precision: str, liveness = None, idx: Optional[torch.Tensor] = None,
                      gated: Optional[dict] = None):
    """
    The top-down values of the children of a sum layer's fused node blocks (:func:`_lifted_sum_top_down_kernel`).

    :param tables: child-block-major: ``cb_row``, ``cb_reg`` (-1 for element rows), ``cb_add``, ``cb_ivreg`` (a region
        of the children's interval, for pruning), ``par_ptr``, ``par_nid``, ``par_reg``, ``par_pid`` (int32 / int64
        device tensors, see :mod:`.queries.top_down`), and a gated layer's ``gate``: its parents' ``lc``, ``row`` and
        ``cbs`` (see :mod:`..compiled.top_down_program`)
    :param cbs: the child block size of these children
    :param max_cols: the most columns any child block has (``batch_size`` x its slots)
    :param idx: None (every child block), or int32 on the device: the plan's index list of the child blocks to compute
    :param gated: None, or the session's gates (:func:`~..queries.forward.part_gates`): the layer's gates applied
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
    lanes = liveness.lanes if liveness is not None else None
    perm, reg_perm, reg_live = (lanes.perm, lanes.reg_first, lanes.reg_live) if lanes is not None else \
        (tables["cb_row"], tables["cb_row"], tables["cb_row"])
    g = (gt["lc"], gt["row"], gt["cbs"], gt["ck"]) if gt is not None else (tables["cb_row"],) * 4
    for first, size in grid_chunks(host_cdiv(max_cols, TILE_B), 1):
        _lifted_sum_top_down_kernel[(num_blocks * num_ctiles, size)](
            element_flows, node_flows, params, tables["cb_row"], tables["cb_reg"], tables["cb_add"], tables["cb_ivreg"],
            tables["cb_row"] if idx is None else idx, tables["par_ptr"], tables["par_nid"], tables["par_reg"],
            tables["par_pid"], regions["offset"], regions["width"], regions["first"], regions["slots"], perm, reg_perm,
            reg_live, *g, *gate_launch(gated, tables["cb_row"])[3:], elem_first, elem_width, batch_size,
            first, BS = block_size, CBS = cbs, TILE_C = TILE_C, TILE_I = TILE_I, TILE_B = TILE_B,
            PRECISION = precision, PRUNE = liveness is not None, INDEXED = idx is not None, GATED = gated is not None,
            LANES = gated is not None and gated["lanes"] is not None,
            GATE_TILE = gt is not None and gt["uniform"].get(TILE_C, False),
            SHARED_GATES = gated is not None and gated["gstride"] == 0,
            NG = gt["node_gate"] if gt is not None else block_size, num_warps = warps, num_stages = stages)


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
