"""
The fused forward sum: an online log-sum-exp over chunks of children, the products on tensor cores (see
:mod:`pyjuice.constraints.inference.dfa_exact.kernels.sum`).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ...external_params.block_scale import _chunk_gates, _edge_gates, _node_log_z
from ..common import _child, _tf32_operand, grid_chunks
from . import tile_rules
from .fused_cuda import _cuda_forward, _cuda_plan
from .steps import (_column_sample, _gate_segment, _gate_tile, _lse_dot_step, _lse_dot_step_biased, _lse_dot_step_pre,
                    _lse_dot_step_rows, _node_gates_per_tile, _row_tile, _split_merge, gate_launch)
from .tile_rules import SPLIT_TILES, _NO_SPLIT, _TILES, _num_sms, _tile_warps, pre_buffer, split_buffer, split_count


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
    if num_edges >= 2 * tile_rules.SPLIT_MIN_SHARE * TILE_E and tiles <= SPLIT_TILES and \
            tiles < tile_rules.SPLIT_PROGRAMS_PER_SM * _num_sms(node_mars.device):                  # a launch that may split
        split = split_count(tiles, host_cdiv(num_edges, TILE_E), node_mars.device)
        if split > 1:
            buf, split_kw = (split_buffer(node_mars.device, tiles, split * TILE_M * TILE_B),), dict(SPLIT = split)
    pre_kw = _NO_SPLIT
    if not rows and (gated is None or gated["uniform"].get(TILE_E, False)) and \
            num_blocks * num_mtiles >= tile_rules.PRE_MIN_TILES and shared_children(cids, nb_reg):   # the children's chunks once
        nch = host_cdiv(num_edges, TILE_E)
        plan = _cuda_plan(params, pids, num_blocks, gated, g, num_edges, max_cols, block_size, TILE_E, precision,
                          aliased, node_mars.device, frags, bf16)
        al = 4 if plan is None else 8                   # (the CUDA kernel's groups of 8 columns, :func:`_frag_offsets`)
        pre_ld = host_cdiv(max_cols, al) * al
        pre = pre_buffer(node_mars.device, (nch * TILE_E + nch) * pre_ld)
        if plan is not None:                            # the CUDA kernel's call runs its own pre-pass
            _cuda_forward(plan, node_mars, element_mars, params, nids, cids, nb_reg, sel, num_blocks, regions, lanes,
                          gated, g, (pre, nch * TILE_E * pre_ld, pre_ld), elem_first, elem_width, batch_size, max_cols,
                          precision)
            return
        for first, size in grid_chunks(num_col_tiles, 1):
            _forward_prepass_kernel[(nch, size)](
                element_mars, pre, nch * TILE_E * pre_ld, pre_ld, cids, nb_reg, regions["slots"], perm, reg_perm,
                reg_live, elem_first, elem_width, batch_size, num_edges, first, TILE_E = TILE_E, TILE_B = TILE_B,
                PRUNE = liveness is not None, PRECISION = precision, num_warps = 4)
        pre_kw = dict(pre_args = (pre, nch * TILE_E * pre_ld, pre_ld))
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
