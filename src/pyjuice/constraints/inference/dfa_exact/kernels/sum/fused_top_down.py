"""
The fused sum's top-down pass: a parent's value and its gates, an edge's weight and a child's inside value to a child's flow.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ...external_params.block_scale import _add_gated_parent, _node_log_z, _row_gates, _tile_gates
from ..common import _logaddexp_tile, _tf32_operand, grid_chunks
from . import tile_rules
from .fused_cuda import _cuda_td_plan, _cuda_top_down
from .fused_forward import _frag_offsets, _frag_offsets_bf16
from .steps import (_column_sample, _gate_tile, _lse_dot_step, _lse_dot_step_biased, _lse_dot_step_pre, _pre_chunk,
                    _split_merge, gate_launch)
from .tile_rules import SPLIT_TILES, _NO_SPLIT, _TILES, _num_sms, _tile_warps, pre_buffer, split_buffer, split_count


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
    if chunks >= 2 * tile_rules.SPLIT_MIN_SHARE and tiles <= SPLIT_TILES and \
            tiles < tile_rules.SPLIT_PROGRAMS_PER_SM * _num_sms(element_flows.device):              # a launch that may split
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
    if num_ctiles >= tile_rules.PRE_MIN_TILES and not (gate_tile and NG < block_size):   # the parents' chunks once: a pre-pass
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
