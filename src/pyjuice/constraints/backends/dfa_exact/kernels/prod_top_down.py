"""
The lifted product layer's top-down pass (see :mod:`.queries.top_down`): the top-down value of every child of every
product, from the products' own (``element_flows``, or ``temp_flows`` for a fold's scratch row) and the other children's
inside values. One kernel per contraction of :mod:`.prod`, each the transpose of its forward:

* ``copy``: the child's top-down value is the copy's;
* ``input @ block`` (``out[i, j] = sum_c mass(i, c) R[next(t, i, c), j]``): ``O_R[q, j] = sum_(i, c) -> q mass(i, c)
  O_out[i, j]``, a gather over the predecessors of ``q`` at boundary ``t`` (:func:`_input_block_top_down_kernel`);
* ``block @ input`` (``out[i, j] = sum_(q, c) -> j L[i, q] mass(q, c)``): ``O_L[i, q] = sum_c mass(q, c)
  O_out[i, next(t, q, c)]``, a gather over the successors of ``q`` (:func:`_block_input_top_down_kernel`);
* ``block @ block`` (``out = L @ R``): ``O_L = O_out @ R^T`` and ``O_R = L^T @ O_out``, log-space matrix products with
  one operand read transposed (:func:`_block_block_top_down_kernel`), skipping the (output tile, shared chunk) pairs
  the automaton cannot join and recomputing exactly the tiles whose entries the shifts flush, as the forward does.

A child's value is the log-sum over every parent product (several, in a PD) of its contribution: the tables are
CHILD-MAJOR (:class:`TopDownLaunch`), one program owning a child's tile and looping over its parents, so no
accumulation needs atomics or scratch. Across launches -- a sum node is a child of products in several layers --
the first launch to reach a child writes its value and the later ones log-add to it (``add`` per child), so the
buffers are never filled beforehand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from pyjuice.layer.kernels import autotune
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from .common import (SUM, _block_ptr, _child, _exit_width, _interval, _logaddexp_tile, _lse_step, _pair, _slot,
                     grid_chunks, tensor)
from .compact import _lane, _lane_ends
from .prod import (BLOCK_BLOCK_PROGRAMS_PER_SM, BLOCK_BLOCK_TILES, EXACT_BELOW, PRODUCT_PRECISION, SKIP_EMPTY_TILES,
                   _chunk_needed, _flat_tile, _live_lanes, _lse_step_one_exp, _lse_value, _tile, _tile_load)


#: fields of a child-major table's parent rows, per contraction: the forward step's fields (see :mod:`.prod`), the
#: child being the operand named by the side
PARENT_FIELDS = {"copy": 5, "input_block": 7, "block_input": 7, "block_block": 8}


@dataclass
class TopDownLaunch:
    """
    One launch of the top-down pass: the children of one side of one forward launch, each with its parents.

    :ivar form: the forward contraction
    :ivar side: ``"L"`` or ``"R"``, which operand the children are
    :ivar kinds: ``(child kind, parent kind)``: the child's rows are sum rows (``SUM``) or scratch rows
        (``TEMP``); the parents' top-down values are in ``element_flows`` (False) or ``temp_flows`` (True); and for
        ``input_block`` / ``block_input``, which of the child's and output's intervals keep every pair
    :ivar children: int32 [C, 4] on the device: row, region, ``add`` (1: log-add to the value already there), and the
        child's triple index for the skip masks (``block_block``) or 0
    :ivar ptr: int32 [C + 1], every child's parents in ``parents``
    :ivar parents: int32 [P, fields], the forward step rows, every child's contiguous; ``block_block`` rows carry the
        parent's triple index in a 9th field
    :ivar E_max, X_max, S_max: the most entry columns, exit columns and slots of the children's blocks
    :ivar host: for query plans, numpy copies: ``children`` [C, 3] (row, region, add), ``ptr``, ``parents``, and every
        child's (``child_iv``) and parent's (``parent_iv``) interval, by ``prog.intervals``' numbers
    """

    form: str
    side: str
    kinds: tuple
    children: torch.Tensor
    ptr: torch.Tensor
    parents: torch.Tensor
    E_max: int
    X_max: int
    S_max: int
    host: Any


@triton.jit
def _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, temp_width, KIND: tl.constexpr):
    """Start of a child's top-down block: a sum node's row in its region of ``node_flows`` (every sample's block
    ``slots`` further), or a scratch row of ``temp_flows``."""
    if KIND == 0:
        return node_flows + tl.load(reg_offset + reg) + (row - tl.load(reg_first + reg)) * tl.load(reg_width + reg)
    else:
        return temp_flows + row.to(tl.int64) * temp_width


@triton.jit
def _out_flows_ptr(row, element_flows, temp_flows, out_first, out_width, temp_width, OUT_TEMP: tl.constexpr):
    """Start of a parent product's top-down block (``element_flows``, or ``temp_flows`` for a fold's row)."""
    if OUT_TEMP:
        return temp_flows + row.to(tl.int64) * temp_width
    else:
        return element_flows + (row - out_first).to(tl.int64) * out_width


@triton.jit
def _once(once_iv, iv_of, a, b, n):
    """Whether this query keeps the top-down values of interval ``[a, b)`` once (a batch of one, sample stride 0)."""
    return tl.load(once_iv + tl.load(iv_of + a * (n + 1) + b)) != 0


@triton.jit
def _child_lanes(tile, B, S, reg, lanes, lane_first, lane_live, TILE: tl.constexpr, ORDERED: tl.constexpr):
    """A tile of a child's lanes, flattened (sample, slot): ``(lane, in range, live, all dead)``. ``ORDERED``: through
    the query's lane order of the child's interval (its live lanes first, :class:`~.live.LaneOrder`), the first
    ``lane_live`` live -- every tile past them dead; else in order, liveness left to the caller."""
    k = tile * TILE + tl.arange(0, TILE)
    m = k < B * S
    if ORDERED:
        nl = tl.load(lane_live + reg)
        return tl.load(lanes + tl.load(lane_first + reg) + k, mask = m, other = 0), m, m & (k < nl), tile * TILE >= nl
    else:
        return k, m, m, False


@triton.jit
def _store_flows(ptr, idx, val, m, add):
    """Write ``val`` at ``ptr + idx``, or log-add it to the value there (``add``)."""
    if add:
        val = _logaddexp_tile(tl.load(ptr + idx, mask = m, other = float("-inf")), val)
    tl.store(ptr + idx, val, mask = m)


@triton.jit
def _parent_base(b, ostride, COMPACT: tl.constexpr):
    """Where samples ``b``'s values start in a parent's row: their block (``ostride`` floats each, 0 for values kept
    once) when dense; the row's start when ``COMPACT`` (a lane says where, :func:`_parent_lane`)."""
    if COMPACT:
        return b * 0
    else:
        return b * ostride


@triton.jit
def _parent_lane(sl, b, once, coff, ccount, crank, kp, a, i, e, j, m, B, n, W, COMPACT: tl.constexpr):
    """Where a parent's value of the pair ``(i at a, j at e)`` of samples ``b`` is from :func:`_parent_base` (its slot
    ``sl`` in the sample's block; negative: the automaton does not join them): dense, slot ``sl``; ``COMPACT`` (see
    :mod:`.compact`), the pair's lane in the row of the parent's interval ``kp``, or slot ``sl`` of the one block kept
    once (``once``), -1 if nowhere."""
    if COMPACT:
        ok = (sl >= 0) & (i >= 0) & (j >= 0)
        lane = _lane(coff, ccount, crank, kp, b, a, i, e, j, m & ok, B, n, W)
        return tl.where(ok, tl.where(once, sl, lane), -1)
    else:
        return sl


@triton_jit
def _copy_top_down_kernel(element_flows, temp_flows, node_flows, iv_of, iv_info, once_iv, coff, children,
                         children_idx, ptr, parents, reg_offset, reg_width, reg_first, out_first, out_width, temp_width,
                         B, n, pid0, C_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, COMPACT: tl.constexpr,
                         INDEXED: tl.constexpr, TILE: tl.constexpr):
    """One program per child and tile of its lanes (``B * slots``, or ``COMPACT`` its interval's, see
    :mod:`.compact`): the log-sum over its parents' copies. A copy's child and parents share their interval, so they
    are kept once together: a per-sample child's parents are per sample, a compact one's lanes theirs."""
    c = _child(children_idx, tl.program_id(0), INDEXED)
    pid = pid0 + tl.program_id(1)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c * 4 + 2)
    p0 = tl.load(ptr + c)
    p1 = tl.load(ptr + c + 1)
    t0 = tl.load(parents + p0 * 5 + 3)
    t2 = tl.load(parents + p0 * 5 + 4)
    S, _, _ = _interval(iv_of, iv_info, t0, t2, n)
    if COMPACT:
        k = tl.load(iv_of + t0 * (n + 1) + t2)
        ncols = tl.load(coff + k.to(tl.int64) * (B + 1) + B).to(tl.int32)
    else:
        ncols = S * B
    if pid * TILE < ncols:
        cols = pid * TILE + tl.arange(0, TILE)
        m = cols < ncols
        if COMPACT:
            ocols = cols
        else:
            ocols = tl.where(_once(once_iv, iv_of, t0, t2, n), cols % S, cols)  # the parents' values kept once?
        mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for p in range(p0, p1):
            op = _out_flows_ptr(tl.load(parents + p * 5 + 0), element_flows, temp_flows, out_first, out_width,
                                temp_width, OUT_TEMP)
            mx, acc = _lse_step(mx, acc, tl.load(op + ocols, mask = m, other = float("-inf")))
        cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, temp_width, C_KIND)
        _store_flows(cp, cols, _lse_value(mx, acc), m, add)


@triton_jit
def _input_block_top_down_kernel(element_flows, temp_flows, node_flows, node_mars, obs_class, next_col, class_mars,
                                mass_row, trans, pred_ptr, pred_col, pred_x, width, iv_of, iv_info, iv_table, once_iv,
                                live, lanes, lane_first, lane_live, coff, ccount, cstart, ccolumns, crank, children,
                                children_idx, ptr, parents, reg_offset, reg_width, reg_first, out_first, out_width,
                                temp_width, B, n, W, C, input_start, csteps, pid0, C_KIND: tl.constexpr,
                                OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr, PRUNE: tl.constexpr,
                                ORDERED: tl.constexpr, COMPACT: tl.constexpr, C_FULL: tl.constexpr,
                                OUT_FULL: tl.constexpr, INDEXED: tl.constexpr, TILE: tl.constexpr):
    """The right operand ``R`` over ``[t + 1, t2)`` of ``input @ block`` steps: ``O_R[q, j] = log sum over the
    parents, and over the entries ``(i, class)`` leading to ``q`` at boundary ``t``, of mass + O_out[i, j]``. One
    program per child and tile of its lanes, flattened (sample, slot). ``ORDERED`` (a sum child under evidence pruning):
    the lanes in the query's lane order, live ones first -- a tile of dead lanes only stores -inf (a first writer)
    or nothing (a later one), since evidence usually leaves a few live (one per sample through an observed prefix).
    ``COMPACT``: the child's and the parents' values compact (:mod:`.compact`), a lane per pair of live columns, -inf
    at a pair the automaton does not join."""
    c = _child(children_idx, tl.program_id(0), INDEXED)
    tile = pid0 + tl.program_id(1)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c * 4 + 2)
    p0 = tl.load(ptr + c)
    p1 = tl.load(ptr + c + 1)
    t = tl.load(parents + p0 * 7 + 2)
    t2 = tl.load(parents + p0 * 7 + 5)
    X = _exit_width(width, t2, n)
    S, cpos, pairs = _interval(iv_of, iv_info, t + 1, t2, n)               # the child's slots
    OS, opos, _ = _interval(iv_of, iv_info, t, t2, n)                        # the parents' slots
    once = _once(once_iv, iv_of, t, t2, n)                                   # ... kept once?
    ostride = tl.where(once, 0, OS)
    kp = tl.load(iv_of + t * (n + 1) + t2)
    cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, temp_width, C_KIND)
    if COMPACT:
        kc = tl.load(iv_of + (t + 1) * (n + 1) + t2)
        ncols = tl.load(coff + kc.to(tl.int64) * (B + 1) + B).to(tl.int32)
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < ncols
        dead = False
    else:
        ncols = B * S
        idx, m, ml, dead = _child_lanes(tile, B, S, reg, lanes, lane_first, lane_live, TILE, ORDERED)
    if dead:                                                                 # ORDERED: no live lane in the tile
        if add == 0:
            tl.store(cp + idx, tl.full([TILE], float("-inf"), dtype = tl.float32), mask = m)
    elif tile * TILE < ncols:
        if COMPACT:
            b, q, j = _lane_ends(coff, ccount, cstart, ccolumns, kc, t + 1, t2, idx, m, B, n, csteps)
            b = b.to(tl.int64)
            ml = m & (_slot(iv_table, cpos, q, j, X, m, C_FULL) >= 0)        # a hole: -inf
        else:
            b = (idx // S).to(tl.int64)
            v = _pair(iv_table, pairs, idx % S, m, C_FULL)
            q = v // X
            j = v % X
            if not ORDERED:
                ml = _live_lanes(live, m, b, t + 1, q, t2, j, n, W, PRUNE)
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)
        e0 = tl.load(pred_ptr + t * (W + 1) + q, mask = ml, other = 0)
        e1 = tl.load(pred_ptr + t * (W + 1) + q + 1, mask = ml, other = 0)
        mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for p in range(p0, p1):
            u = tl.load(parents + p * 7 + 1).to(tl.int64)
            own = tl.load(node_mars + u * B + b, mask = m, other = float("-inf"))   # the observed token's log-prob
            op = _out_flows_ptr(tl.load(parents + p * 7 + 0), element_flows, temp_flows, out_first, out_width,
                                temp_width, OUT_TEMP) + _parent_base(b, ostride, COMPACT)
            if GROUPED:
                masses = trans + tl.load(parents + p * 7 + 6).to(tl.int64)
            else:
                masses = class_mars + tl.load(mass_row + (u - input_start)).to(tl.int64) * C
            for k in range(0, tl.max(e1 - e0, axis = 0)):                    # entry (i, x) -> q
                em = ml & (e0 + k < e1)
                i = tl.load(pred_col + e0 + k, mask = em, other = 0)
                x = tl.load(pred_x + e0 + k, mask = em, other = 0)
                if GROUPED:                                                  # the observed class leads to q?
                    holds = tl.load(next_col + (t * W + i) * C + tl.maximum(oc, 0), mask = em, other = -1) == q
                else:                                                        # is it this entry's class?
                    holds = x == oc
                mass = tl.where(oc >= 0, tl.where(holds, own, float("-inf")),
                                tl.load(masses + x, mask = em, other = float("-inf")))
                sl = _parent_lane(_slot(iv_table, opos, i, j, X, em, OUT_FULL), b, once, coff, ccount, crank, kp, t,
                                  i, t2, j, em, B, n, W, COMPACT)
                ov = tl.load(op + sl, mask = em & (sl >= 0), other = float("-inf"))
                mx, acc = _lse_step_one_exp(mx, acc, mass + ov)
        val = _lse_value(mx, acc)
        if COMPACT:
            val = tl.where(ml, val, float("-inf"))
        _store_flows(cp, idx, val, m, add)


@triton_jit
def _block_input_top_down_kernel(element_flows, temp_flows, node_flows, node_mars, obs_class, next_col, class_mars,
                                mass_row, trans, entry_ptr, entry_succ, entry_x, width, iv_of, iv_info, iv_table,
                                once_iv, live, lanes, lane_first, lane_live, coff, ccount, cstart, ccolumns, crank,
                                children, children_idx, ptr, parents, reg_offset, reg_width, reg_first, out_first,
                                out_width, temp_width, B, n, W, C, input_start, csteps, pid0, C_KIND: tl.constexpr,
                                OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr, PRUNE: tl.constexpr,
                                ORDERED: tl.constexpr, COMPACT: tl.constexpr, C_FULL: tl.constexpr,
                                OUT_FULL: tl.constexpr, INDEXED: tl.constexpr, TILE: tl.constexpr):
    """The left operand ``L`` over ``[t0, t)`` of ``block @ input`` steps (output over ``[t0, t + 1)``): ``O_L[i, q] =
    log sum over the parents and over the entries ``(successor, class)`` of ``q`` at boundary ``t`` of mass +
    O_out[i, successor]``. ``ORDERED``, ``COMPACT``: as :func:`_input_block_top_down_kernel`."""
    c = _child(children_idx, tl.program_id(0), INDEXED)
    tile = pid0 + tl.program_id(1)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c * 4 + 2)
    p0 = tl.load(ptr + c)
    p1 = tl.load(ptr + c + 1)
    t0 = tl.load(parents + p0 * 7 + 3)
    t = tl.load(parents + p0 * 7 + 5)
    M = tl.load(width + t)                                                   # the child's exit columns
    X = _exit_width(width, t + 1, n)                                         # the parents' exit columns
    S, cpos, pairs = _interval(iv_of, iv_info, t0, t, n)
    OS, opos, _ = _interval(iv_of, iv_info, t0, t + 1, n)
    once = _once(once_iv, iv_of, t0, t + 1, n)                               # the parents' values kept once?
    ostride = tl.where(once, 0, OS)
    kp = tl.load(iv_of + t0 * (n + 1) + t + 1)
    end = t + 1 == n
    cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, temp_width, C_KIND)
    if COMPACT:
        kc = tl.load(iv_of + t0 * (n + 1) + t)
        ncols = tl.load(coff + kc.to(tl.int64) * (B + 1) + B).to(tl.int32)
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < ncols
        dead = False
    else:
        ncols = B * S
        idx, m, ml, dead = _child_lanes(tile, B, S, reg, lanes, lane_first, lane_live, TILE, ORDERED)
    if dead:                                                                 # ORDERED: no live lane in the tile
        if add == 0:
            tl.store(cp + idx, tl.full([TILE], float("-inf"), dtype = tl.float32), mask = m)
    elif tile * TILE < ncols:
        if COMPACT:
            b, i, q = _lane_ends(coff, ccount, cstart, ccolumns, kc, t0, t, idx, m, B, n, csteps)
            b = b.to(tl.int64)
            ml = m & (_slot(iv_table, cpos, i, q, M, m, C_FULL) >= 0)        # a hole: -inf
        else:
            b = (idx // S).to(tl.int64)
            v = _pair(iv_table, pairs, idx % S, m, C_FULL)
            i = v // M
            q = v % M
            if not ORDERED:
                ml = _live_lanes(live, m, b, t0, i, t, q, n, W, PRUNE)
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)
        mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for p in range(p0, p1):
            u = tl.load(parents + p * 7 + 4).to(tl.int64)
            own = tl.load(node_mars + u * B + b, mask = m, other = float("-inf"))
            op = _out_flows_ptr(tl.load(parents + p * 7 + 0), element_flows, temp_flows, out_first, out_width,
                                temp_width, OUT_TEMP) + _parent_base(b, ostride, COMPACT)
            # an observed token: its own log-probability, then the output at the column its class leads to
            obs = ml & (oc >= 0)
            s = tl.load(next_col + (t * W + q) * C + oc, mask = obs, other = -1)
            s = tl.where(end & (s >= 0), 0, s)
            sl = _parent_lane(_slot(iv_table, opos, i, s, X, obs, OUT_FULL), b, once, coff, ccount, crank, kp, t0,
                              i, t + 1, s, obs, B, n, W, COMPACT)
            ov = tl.load(op + sl, mask = obs & (sl >= 0), other = float("-inf"))
            mx, acc = _lse_step_one_exp(mx, acc, tl.where(obs, own + ov, float("-inf")))
            # a missing token: every entry (successor, mass) of q
            miss = ml & (oc < 0)
            if tl.max(tl.where(miss, 1, 0), axis = 0) > 0:
                if GROUPED:
                    masses = trans + tl.load(parents + p * 7 + 6).to(tl.int64)
                else:
                    masses = class_mars + tl.load(mass_row + (u - input_start)).to(tl.int64) * C
                e0 = tl.load(entry_ptr + t * (W + 1) + q, mask = miss, other = 0)
                e1 = tl.load(entry_ptr + t * (W + 1) + q + 1, mask = miss, other = 0)
                for k in range(0, tl.max(e1 - e0, axis = 0)):
                    em = miss & (e0 + k < e1)
                    s = tl.load(entry_succ + e0 + k, mask = em, other = -1)
                    s = tl.where(end & (s >= 0), 0, s)
                    sl = _parent_lane(_slot(iv_table, opos, i, s, X, em, OUT_FULL), b, once, coff, ccount, crank,
                                      kp, t0, i, t + 1, s, em, B, n, W, COMPACT)
                    ov = tl.load(op + sl, mask = em & (sl >= 0), other = float("-inf"))
                    mass = tl.load(masses + tl.load(entry_x + e0 + k, mask = em, other = 0), mask = em,
                                   other = float("-inf"))
                    mx, acc = _lse_step_one_exp(mx, acc, mass + ov)
        val = _lse_value(mx, acc)
        if COMPACT:
            val = tl.where(ml, val, float("-inf"))
        _store_flows(cp, idx, val, m, add)


@triton.jit
def _tile_load_t(ptr, iv_table, positions, rows, cols, ncols, rm, cm):
    """The ``[len(rows), len(cols)]`` tile whose entry ``(r, c)`` is ``S[c, r]`` of a block ``S`` with ``ncols``
    columns: the transposed operand, read without a transposition in registers."""
    mask = rm[:, None] & cm[None, :]
    sl = _slot(iv_table, positions, cols[None, :], rows[:, None], ncols, mask, False)
    return tl.load(ptr + sl, mask = mask & (sl >= 0), other = float("-inf"))


@triton.jit
def _top_down_tile(ap, bp, iv_table, apos, bpos, mb, live_k, r, c, tr, tc, Rn, K, Cn, TILES_I, TILES_J, WORDS,
                  TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                  TRANS_A: tl.constexpr, TRANS_B: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr,
                  PRECISION: tl.constexpr):
    """``log(sum_k exp(A[r, k] + B[k, c]))`` for one output tile, as :func:`~.prod._block_block_tile`: ``A`` is
    ``[Rn, K]`` (stored ``[K, Rn]`` when ``TRANS_A``), ``B`` is ``[K, Cn]`` (stored ``[Cn, K]`` when ``TRANS_B``);
    ``live_k``: the shared boundary's liveness (pruned). Returns the tile and its shifted sums."""
    rm, cm = r < Rn, c < Cn
    mA = tl.full([TM], float("-inf"), dtype = tl.float32)
    mB = tl.full([TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for ch in range(0, tl.cdiv(K, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, tr, tc, ch, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            ks = ch * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_k + ks, mask = ks < K, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            k = ch * TK + tl.arange(0, TK)
            km = k < K
            if TRANS_A:
                at = _tile_load_t(ap, iv_table, apos, r, k, Rn, rm, km)
            else:
                at = _tile_load(ap, iv_table, apos, r, k, K, rm, km)
            if TRANS_B:
                bt = _tile_load_t(bp, iv_table, bpos, k, c, K, km, cm)
            else:
                bt = _tile_load(bp, iv_table, bpos, k, c, Cn, km, cm)
            nA = tl.maximum(mA, tl.max(at, axis = 1))
            nB = tl.maximum(mB, tl.max(bt, axis = 0))
            sA = tl.where(nA == float("-inf"), 0.0, nA)
            sB = tl.where(nB == float("-inf"), 0.0, nB)
            fA = tl.where(nA == float("-inf"), 1.0, tl.exp(mA - sA))
            fB = tl.where(nB == float("-inf"), 1.0, tl.exp(mB - sB))
            acc = acc * fA[:, None] * fB[None, :] + tl.dot(tl.exp(at - sA[:, None]), tl.exp(bt - sB[None, :]),
                                                            input_precision = PRECISION)
            mA, mB = nA, nB
    sA = tl.where(mA == float("-inf"), 0.0, mA)
    sB = tl.where(mB == float("-inf"), 0.0, mB)
    return tl.log(acc) + sA[:, None] + sB[None, :], acc


@triton.jit
def _top_down_tile_exact(ap, bp, iv_table, apos, bpos, mb, live_k, r, c, tr, tc, Rn, K, Cn, TILES_I, TILES_J, WORDS,
                        TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                        TRANS_A: tl.constexpr, TRANS_B: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr):
    """:func:`_top_down_tile` by an online log-sum-exp per entry over the shared columns one at a time (see
    :func:`~.prod._block_block_tile_exact`)."""
    rm, cm = r < Rn, c < Cn
    m = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for ch in range(0, tl.cdiv(K, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, tr, tc, ch, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            ks = ch * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_k + ks, mask = ks < K, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            for kk in range(0, TK):
                k = ch * TK + kk
                ok = k < K
                if PRUNE:
                    ok = ok & (tl.load(live_k + k, mask = k < K, other = 0) != 0)
                if ok:
                    if TRANS_A:
                        sa = _slot(iv_table, apos, k, r, Rn, rm, False)
                    else:
                        sa = _slot(iv_table, apos, r, k, K, rm, False)
                    av = tl.load(ap + sa, mask = rm & (sa >= 0), other = float("-inf"))
                    if TRANS_B:
                        sb = _slot(iv_table, bpos, c, k, K, cm, False)
                    else:
                        sb = _slot(iv_table, bpos, k, c, Cn, cm, False)
                    bv = tl.load(bp + sb, mask = cm & (sb >= 0), other = float("-inf"))
                    m, acc = _lse_step(m, acc, av[:, None] + bv[None, :])
    return _lse_value(m, acc)


@triton.jit
def _parent_tile(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width, iv_of, iv_info, iv_table,
                 once_iv, lrow, skip_bits, reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior,
                 out_first, out_width, temp_width, b, n, W, r, cc, tr, tc, Rn, Cn, valid, exact_below, TILES_I, TILES_J,
                 WORDS,
                 O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, SIDE_L: tl.constexpr, TM: tl.constexpr,
                 TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr, SKIP: tl.constexpr,
                 PRUNE: tl.constexpr, PRECISION: tl.constexpr, RECHECK: tl.constexpr):
    """Parent ``p``'s contribution to a child's output tile on the parent's own intervals -- the child's interval is
    every parent's, the other operand's and the product's are not (a sum shared by products reaching different
    boundaries) -- as ``(out, shifted sums, fast)``: ``RECHECK``, ``out`` is computed again exactly where the shifts
    flush a valid entry and ``fast`` is the shifted product's value; else both are the shifted product's."""
    t0 = tl.load(parents + p * 9 + 5)
    t1 = tl.load(parents + p * 9 + 6)
    t2 = tl.load(parents + p * 9 + 7)
    OS, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    op = _out_flows_ptr(tl.load(parents + p * 9 + 0), element_flows, temp_flows, out_first, out_width, temp_width,
                        OUT_TEMP) + tl.where(_once(once_iv, iv_of, t0, t2, n), 0, b * OS)   # kept once: stride 0
    mb = skip_bits + tl.load(parents + p * 9 + 8).to(tl.int64) * TILES_I * TILES_J * WORDS
    if SIDE_L:                                                               # O_out @ R^T, shared: the exit columns
        K = _exit_width(width, t2, n)
        XS, xpos, _ = _interval(iv_of, iv_info, t1, t2, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 3), tl.load(parents + p * 9 + 4), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        xp = xp + tl.where(x_own, b * XS, 0)
        out, acc = _top_down_tile(op, xp, iv_table, opos, xpos, mb, lrow + t2 * W, r, cc, tr, tc, Rn, K, Cn, TILES_I,
                                  TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, False, True, SKIP, PRUNE, PRECISION)
        fast = out
        if RECHECK:
            if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                out = _top_down_tile_exact(op, xp, iv_table, opos, xpos, mb, lrow + t2 * W, r, cc, tr, tc, Rn, K, Cn,
                                           TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, False, True, SKIP, PRUNE)
    else:                                                                    # L^T @ O_out, shared: the entry columns
        K = tl.load(width + t0)
        XS, xpos, _ = _interval(iv_of, iv_info, t0, t1, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 1), tl.load(parents + p * 9 + 2), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        xp = xp + tl.where(x_own, b * XS, 0)
        out, acc = _top_down_tile(xp, op, iv_table, xpos, opos, mb, lrow + t0 * W, r, cc, tr, tc, Rn, K, Cn, TILES_I,
                                  TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, True, False, SKIP, PRUNE, PRECISION)
        fast = out
        if RECHECK:
            if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                out = _top_down_tile_exact(xp, op, iv_table, xpos, opos, mb, lrow + t0 * W, r, cc, tr, tc, Rn, K, Cn,
                                           TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, True, False, SKIP, PRUNE)
    return out, acc, fast


@triton_jit
def _block_block_top_down_kernel(element_flows, temp_flows, node_flows, node_mars, prior_mars, temp, width, iv_of,
                                iv_info, iv_table, once_iv, live, children, children_idx, ptr, parents, skip_bits,
                                reg_offset, reg_width,
                                reg_first, prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, B, n,
                                W, TILES_I, TILES_J, WORDS, SPLIT, GROUP, exact_below, pid0, C_KIND: tl.constexpr,
                                O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, SIDE_L: tl.constexpr, TM: tl.constexpr,
                                TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                                SKIP: tl.constexpr, PRUNE: tl.constexpr, EXACT: tl.constexpr, PRECISION: tl.constexpr,
                                INDEXED: tl.constexpr):
    """One program per (child, sample, group of ``GROUP`` output tiles): the child's top-down tile is the log-sum over
    its parents of ``O_out @ R^T`` (``SIDE_L``: the child is the left operand ``L`` over ``[t0, t1)``, the other
    operand ``R`` over ``[t1, t2)``, the shared columns the exit ``j``) or ``L^T @ O_out`` (the child is ``R`` over
    ``[t1, t2)``, the shared columns the entry ``i``), each parent on its own triple (:func:`_parent_tile`). ``O_KIND``:
    the other operand's kind (a sum's block in ``node_mars`` or ``prior_mars``, or a fold's scratch row). ``EXACT``: as
    :func:`~.prod._block_block_kernel`, per (parent, tile)."""
    pid = pid0 + tl.program_id(0)
    g = pid % SPLIT
    c = _child(children_idx, pid // SPLIT // B, INDEXED)
    b = (pid // SPLIT % B).to(tl.int64)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c * 4 + 2)
    p0 = tl.load(ptr + c)
    p1 = tl.load(ptr + c + 1)
    if SIDE_L:                                                               # output (i, q): the child [t0, t1)
        ra = tl.load(parents + p0 * 9 + 5)
        rb = tl.load(parents + p0 * 9 + 6)
        Rn = tl.load(width + ra)
        Cn = tl.load(width + rb)
    else:                                                                    # output (q, j): the child [t1, t2)
        ra = tl.load(parents + p0 * 9 + 6)
        rb = tl.load(parents + p0 * 9 + 7)
        Rn = tl.load(width + ra)
        Cn = _exit_width(width, rb, n)
    CS, cpos, _ = _interval(iv_of, iv_info, ra, rb, n)
    lrow = live + b * (n + 1) * W
    cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, temp_width, C_KIND) + b * CS
    NJ = tl.cdiv(Cn, TN)
    if EXACT:
        bad = tl.zeros([TM, TN], dtype = tl.int32)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(Rn, TM) * NJ)):
        tr = t // NJ
        tc = t % NJ
        r = tr * TM + tl.arange(0, TM)
        cc = tc * TN + tl.arange(0, TN)
        om = (r < Rn)[:, None] & (cc < Cn)[None, :]
        sl = _slot(iv_table, cpos, r[:, None], cc[None, :], Cn, om, False)
        valid = om & (sl >= 0)
        if PRUNE:
            lr = tl.load(lrow + ra * W + r, mask = r < Rn, other = 0) != 0
            lc = tl.load(lrow + rb * W + cc, mask = cc < Cn, other = 0) != 0
            valid = valid & lr[:, None] & lc[None, :]
        tot = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
        for p in range(p0, p1):
            out, acc, fast = _parent_tile(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width,
                                          iv_of, iv_info, iv_table, once_iv, lrow, skip_bits, reg_offset, reg_width,
                                          reg_first, prior_offset, prior_width, reg_prior, out_first, out_width,
                                          temp_width, b, n, W, r, cc, tr, tc, Rn, Cn, valid, exact_below, TILES_I,
                                          TILES_J, WORDS, O_KIND, OUT_TEMP, SIDE_L, TM, TN, TK, SUB_I, SUB_J, SKIP,
                                          PRUNE, PRECISION, False)
            if EXACT:
                bad = bad | (valid & (acc < exact_below)).to(tl.int32)
            tot = _logaddexp_tile(tot, out)
        _store_flows(cp, sl, tl.where(valid, tot, float("-inf")), om & (sl >= 0), add)
    if EXACT:
        # an entry flushed (rare): the program's tiles again, exactly where so. A child written here is written again;
        # one an earlier launch wrote holds the earlier value log-added to the shifted products' sum, which the exact
        # sum replaces in it (the shifted one is computed again, bit for bit)
        if tl.max(bad) > 0:
            for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(Rn, TM) * NJ)):
                tr = t // NJ
                tc = t % NJ
                r = tr * TM + tl.arange(0, TM)
                cc = tc * TN + tl.arange(0, TN)
                om = (r < Rn)[:, None] & (cc < Cn)[None, :]
                sl = _slot(iv_table, cpos, r[:, None], cc[None, :], Cn, om, False)
                valid = om & (sl >= 0)
                if PRUNE:
                    lr = tl.load(lrow + ra * W + r, mask = r < Rn, other = 0) != 0
                    lc = tl.load(lrow + rb * W + cc, mask = cc < Cn, other = 0) != 0
                    valid = valid & lr[:, None] & lc[None, :]
                tot = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
                tot_fast = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
                for p in range(p0, p1):
                    out, acc, fast = _parent_tile(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp,
                                                  width, iv_of, iv_info, iv_table, once_iv, lrow, skip_bits,
                                                  reg_offset, reg_width, reg_first, prior_offset, prior_width,
                                                  reg_prior, out_first, out_width, temp_width, b, n, W, r, cc, tr, tc,
                                                  Rn, Cn, valid,
                                                  exact_below, TILES_I, TILES_J, WORDS, O_KIND, OUT_TEMP, SIDE_L, TM,
                                                  TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION, True)
                    tot = _logaddexp_tile(tot, out)
                    tot_fast = _logaddexp_tile(tot_fast, fast)
                tot = tl.where(valid, tot, float("-inf"))
                tot_fast = tl.where(valid, tot_fast, float("-inf"))
                m = om & (sl >= 0)
                if add:                       # stored = log(earlier + exp(tot_fast)): swap tot_fast for tot in it
                    old = tl.load(cp + sl, mask = m, other = float("-inf"))
                    s = tl.where(old == float("-inf"), 0.0, old)
                    v = tl.maximum(1.0 - tl.exp(tot_fast - s) + tl.exp(tot - s), tl.exp(tot - s))
                    tot = tl.where(old == float("-inf"), tot, tl.log(v) + s)
                tl.store(cp + sl, tot, mask = m)


def run_top_down(launches, bufs, prog, regions, element_flows, temp_flows, node_flows, node_mars, class_mars, obs_class,
                live, elem_first: int, elem_width: int, B: int, prior: dict, skip, exact: bool, once_iv: torch.Tensor,
                lanes = None, compact = None):
    """
    The top-down pass of one product layer's stage: every :class:`TopDownLaunch` in ``launches``, each with the plan's
    index list of the children it computes (int32 on the device) or None for every one.

    :param bufs: ``(temp, temp_width)``, the layer's scratch rows, holding the folds' inside values (recomputed)
    :param regions: ``offset``, ``width``, ``first`` and ``slots`` of every sum region (device tensors) of the buffer
        the children are written to (``node_flows``, or ``prior_flows`` for a batch of one)
    :param element_flows, temp_flows, node_flows: the top-down buffers (``temp_flows`` as wide as ``temp``)
    :param prior: where the sum operands the plan keeps once are (see :func:`~.common._block_ptr`)
    :param skip: the top-down pass's :class:`~.prod.SkipMasks` (its triples are the parents' ``(interval A, B)`` specs)
    :param exact: see :func:`~.prod.run_products`
    :param once_iv: int32 [intervals] (device, by ``prog.intervals``' numbers): 1 where the query keeps an interval's
        top-down values once, for a batch of one: a parent there is read with sample stride 0
    :param lanes: None, or under evidence pruning the query's :class:`~.live.LaneOrder` ``(perm, reg_first,
        reg_live)``: a sum child's lanes run in that order, its dead ones only set to -inf
    :param compact: None, or the query's :class:`~.compact.CompactLayout`: the per-sample values (children and
        parents) are compact -- ``regions``, ``elem_width`` and ``bufs``' width are their rows'; the children are
        written, and the parents read, a lane per pair of live columns
    """
    temp, temp_width = bufs
    common = dict(reg_offset = regions["offset"], reg_width = regions["width"], reg_first = regions["first"],
                  out_first = elem_first, out_width = elem_width, temp_width = temp_width)
    n, W, C = prog.n, prog.next_col.size(1), prog.next_col.size(2)
    ivs = prog.intervals
    prune, live = live is not None, live if live is not None else obs_class
    tt, grouped = prog.trans_top_down, prog.trans_top_down.grouped
    trans = prog.transition_buffer()
    if compact is not None:
        cl = compact
        lists = dict(coff = cl.off, ccount = cl.count, cstart = cl.start, ccolumns = cl.columns, csteps = cl.steps)
    else:
        lists = dict(coff = live, ccount = live, cstart = live, ccolumns = live, csteps = 0)
    for la, idx in launches:
        idx = tensor(idx)
        Cn = la.children.size(0) if idx is None else idx.numel()
        indexed = dict(children_idx = la.children if idx is None else idx, INDEXED = idx is not None)
        lanes_max = compact.lanes(la.host["child_iv"]) if compact is not None else B * la.S_max   # a child's most
        if la.form == "copy":
            c_kind, out_temp = la.kinds
            tile = _tile(max(lanes_max, 1))
            for first, size in grid_chunks(host_cdiv(lanes_max, tile), 1):
                _copy_top_down_kernel[(Cn, size)](
                    element_flows, temp_flows, node_flows, ivs.of, ivs.info, once_iv, lists["coff"],
                    children = la.children, ptr = la.ptr, parents = la.parents, B = B, n = n, pid0 = first,
                    C_KIND = c_kind, OUT_TEMP = out_temp, COMPACT = compact is not None, TILE = tile, num_warps = 4,
                    **indexed, **common)
        elif la.form in ("input_block", "block_input"):
            c_kind, out_temp, (c_full, out_full) = la.kinds
            ordered = prune and lanes is not None and c_kind == SUM and compact is None
            tile = _flat_tile(max(lanes_max, 1))
            kernel = _input_block_top_down_kernel if la.form == "input_block" else _block_input_top_down_kernel
            tables = (tt.pred_ptr, tt.pred_col, tt.pred_x) if la.form == "input_block" else \
                (tt.entry_ptr, tt.entry_succ, tt.entry_x)
            for first, size in grid_chunks(host_cdiv(lanes_max, tile), 1):
                kernel[(Cn, size)](
                    element_flows, temp_flows, node_flows, node_mars, obs_class, prog.next_col, class_mars,
                    prog.mass_row, trans, *tables, prog.width_t, ivs.of, ivs.info, ivs.table, once_iv, live,
                    *(lanes if ordered else (live, live, live)), crank = compact.rank if compact is not None else live,
                    children = la.children, ptr = la.ptr, parents = la.parents, B = B, n = n, W = W, C = C,
                    input_start = prog.input_start, pid0 = first, C_KIND = c_kind, OUT_TEMP = out_temp,
                    GROUPED = grouped, PRUNE = prune, ORDERED = ordered, COMPACT = compact is not None,
                    C_FULL = c_full, OUT_FULL = out_full, TILE = tile, num_warps = 4, **indexed, **common, **lists)
        elif compact is not None:
            raise NotImplementedError("Compact storage of the top-down values is not supported by block @ block "
                                      "products yet: use storage = \"dense\".")
        else:
            c_kind, o_kind, out_temp = la.kinds
            side_l = la.side == "L"

            def block_block(cfg, nf, tf, la = la, Cn = Cn, indexed = indexed, c_kind = c_kind, o_kind = o_kind,
                            out_temp = out_temp, side_l = side_l):
                TM, TN, TK, warps = cfg
                tiles = host_cdiv(la.E_max, TM) * host_cdiv(la.X_max, TN)
                split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, Cn * B))
                group = host_cdiv(tiles, max(1, split))
                split = host_cdiv(tiles, group)
                for first, size in grid_chunks(Cn * B * split, 0):
                    _block_block_top_down_kernel[(size,)](
                        element_flows, tf, nf, node_mars, prior["mars"], temp, prog.width_t, ivs.of, ivs.info,
                        ivs.table, once_iv, live, la.children, ptr = la.ptr, parents = la.parents,
                        skip_bits = skip.bits, prior_offset = prior["offset"], prior_width = prior["width"],
                        reg_prior = prior["reg_prior"], B = B, n = n, W = W, TILES_I = skip.tiles_i,
                        TILES_J = skip.tiles_j, WORDS = skip.words, SPLIT = split, GROUP = group,
                        exact_below = EXACT_BELOW, pid0 = first, C_KIND = c_kind, O_KIND = o_kind, OUT_TEMP = out_temp,
                        SIDE_L = side_l, TM = TM, TN = TN, TK = TK, SUB_I = TM // skip.tm, SUB_J = TN // skip.tn,
                        SKIP = SKIP_EMPTY_TILES, PRUNE = prune, EXACT = exact, PRECISION = PRODUCT_PRECISION,
                        num_warps = warps, num_stages = 1, **indexed, **common)

            # a child reached before log-adds to its value, so the launch is not idempotent: the candidates are timed
            # into throwaway copies of the outputs (and not at all when those cannot be allocated)
            key = ("lifted_block_block_top_down", la.kinds, la.side, prune, exact, SKIP_EMPTY_TILES, PRODUCT_PRECISION,
                   B, host_next_power_of_2(Cn), la.E_max, la.X_max)
            cfg = BLOCK_BLOCK_TILES[0]
            if autotune.should_tune(key, len(BLOCK_BLOCK_TILES)):
                nf_s, tf_s = autotune.scratch_like(node_flows), autotune.scratch_like(temp_flows)
                if nf_s is not None and tf_s is not None:
                    cfg = autotune.pick(key, BLOCK_BLOCK_TILES, lambda c: block_block(c, nf_s, tf_s))
                del nf_s, tf_s
            else:
                cfg = autotune.pick(key, BLOCK_BLOCK_TILES, None)
            block_block(cfg, node_flows, temp_flows)
