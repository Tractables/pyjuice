"""
The class weights' launches and their kernels: Triton's walk over a class's columns (:func:`_class_weight_kernel`) and the same in CUDA
(``c/class_weights.cu``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

from ..common import (pack_aligned, _block_ptr, _exit_width, _interval, _next_col, _logaddexp_tile, _slot, _tile_rows,
                      node_groups, node_tiles)
from ..prod.launches import _lse_step_one_exp
from ..prod_top_down.launches import _once, _out_flows_ptr, _parent_base, _parent_lane
from .tables import ScoreTables


@dataclass
class ClassWeightLaunch:
    """
    One launch of class weights: the input operands of one forward launch (``input_block`` or ``block_input``), by
    DESTINATION -- (position, column of the position's row group) -- each with its steps, so that one program owns a
    destination and loops over its steps (no atomics; a destination reached by several launches is log-added to).

    :ivar form: the forward contraction
    :ivar kinds: the forward launch's kinds (the other operand's kind first, then whether the output is a scratch row,
        then which of the output's and operand's intervals keep every pair)
    :ivar dests: int32 [D, 2] on the device: position, column
    :ivar ptr: int32 [D + 1], every destination's steps in ``parents``
    :ivar parents: int32 [P, 7], the forward step rows (see :mod:`.prod`), every destination's contiguous
    :ivar host: for query plans, numpy copies of ``dests``, ``ptr`` and ``parents``, and every destination's node tile
        ``group`` (:func:`~.common.node_groups`: its position, and its steps' other boundaries in order)
    :ivar tiles, tile_height: every destination in node tiles (int32 on the device) and their height
        (:func:`~.common.node_tiles`) -- a query plan's lists of destinations are tiles too
    :ivar number: its place among its circuit's: ``prog.cw_most[number]``, the most exit columns (``input_block``) or
        entry columns (``block_input``) of the outputs (what an automaton says of them)
    """

    form: str
    kinds: tuple
    dests: torch.Tensor
    ptr: torch.Tensor
    parents: torch.Tensor
    host: Any
    tiles: torch.Tensor
    tile_height: int
    number: int = 0


def class_weight_launch(launch, alive: np.ndarray, prog, score: ScoreTables, tables):
    """``(launch, boundaries)``, the :class:`ClassWeightLaunch` of a forward launch's input operands and its outputs'
    ``boundaries``, ``alive`` marking the steps whose output has a top-down value (the others add nothing); None when
    there is no such step or no input operand. Its tables are places of ``tables`` (a :class:`~.common.OneCopy`) until
    they are uploaded."""
    if launch.form not in ("input_block", "block_input") or not alive.any():
        return None
    rows = launch.rows.numpy()[alive]
    u, t = (rows[:, 1], rows[:, 2]) if launch.form == "input_block" else (rows[:, 4], rows[:, 5])
    k = score.col_of[u - prog.input_start]
    order = np.lexsort((k, t))
    rows, t, k = rows[order], t[order], k[order]
    first = np.ones(len(t), dtype = bool)
    first[1:] = (t[1:] != t[:-1]) | (k[1:] != k[:-1])
    starts = np.nonzero(first)[0]
    ptr = np.append(starts, len(t)).astype(np.int64)
    dests = np.stack([t[starts], k[starts]], axis = 1)
    boundaries = np.unique(rows[:, 5 if launch.form == "input_block" else 3]).tolist()   # exit columns at t2, entry at t0
    count = np.diff(ptr)                                    # destinations walking the same columns, together: at
    other = np.full((len(starts), int(count.max())), -1, dtype = np.int64)   # one position, steps to the same ends
    other[np.repeat(np.arange(len(starts)), count), np.arange(len(t)) - np.repeat(starts, count)] = \
        rows[:, 5 if launch.form == "input_block" else 3]
    group = node_groups(dests[:, 0], other)
    tiles, height = node_tiles(group)
    i32 = lambda x: tables.add(np.asarray(x, dtype = np.int32))
    return ClassWeightLaunch(launch.form, launch.kinds, i32(dests), i32(ptr), i32(rows),
                             dict(dests = dests, ptr = ptr, parents = rows, group = group, walk = {}), i32(tiles), height), boundaries


@triton.jit
def _walk_column(columns, cstart, ccount, cm, ii, PRUNE: tl.constexpr, UNIFORM: tl.constexpr):
    """Step ``ii`` of every lane's walk over its sample's columns at the input's boundary: ``(column, active)`` -- under
    evidence pruning the ``ii``-th of its sample's live columns (ascending, see :meth:`~.live.Liveness.lists`), else
    column ``ii``. ``UNIFORM`` (always without pruning, or one sample per program): every lane walks the same columns,
    and the column is a scalar."""
    if PRUNE:
        if UNIFORM:
            ok = cm
            i = tl.load(columns + cstart + ii)
        else:
            ok = cm & (ii < ccount)
            i = tl.load(columns + cstart + ii, mask = ok, other = 0)
    else:
        ok = cm
        i = ii
    return i, ok


@triton.jit
def _walk_slots(iv_table, positions, i, other, ncols, ok, om, I_FIRST: tl.constexpr, FULL: tl.constexpr,
                UNIFORM: tl.constexpr):
    """The slots of the pairs ``(i, other)`` (``I_FIRST``) or ``(other, i)``, ``i`` the walk's column
    (:func:`_walk_column`, ``ok`` its lanes), ``other`` a vector of columns (``om`` the real ones): a [lanes,
    len(other)] tile -- one row for every lane when the column is ``UNIFORM`` (one lookup per column, not per lane),
    else a row per lane."""
    if UNIFORM:
        r = i
        c = other
        m = om
    else:
        r = i[:, None]
        c = other[None, :]
        m = ok[:, None] & om[None, :]
    if I_FIRST:
        s = _slot(iv_table, positions, r, c, ncols, m, FULL)
    else:
        s = _slot(iv_table, positions, c, r, ncols, m, FULL)
    if UNIFORM:
        s = s[None, :]
    return s


@triton.jit
def _cw_block_input_term(mx, acc, i, ok, q, op, lp, iv_table, lpos, opos, coff, count, crank, kl, kp, b, vm, once,
                         l_own, M, E, X, t0, t, end, B, n, W, TJ: tl.constexpr, S_FULL: tl.constexpr,
                         OUT_FULL: tl.constexpr, COMPACT: tl.constexpr, MARS: tl.constexpr, UNIFORM: tl.constexpr):
    """One column ``i`` of :func:`_class_weight_kernel`'s walk, ``block @ input`` form: the terms ``L[i0, i] +
    O_out[i0, q]`` over the block's entry columns ``i0`` (``TJ`` at a time), ``q`` the lanes' successors (-1: none;
    a sequence end's folded to its one column), ``ok`` the lanes on; ``i`` a scalar (``UNIFORM``) or one per lane."""
    q = tl.where(end & (q >= 0), 0, q)
    for e0 in range(0, E, TJ):
        i0 = e0 + tl.arange(0, TJ)
        m1 = ok[:, None] & (i0 < E)[None, :]
        if UNIFORM:
            ir = i
        else:
            ir = i[:, None]
        sl = _parent_lane(_walk_slots(iv_table, lpos, i, i0, M, ok, i0 < E, False, S_FULL, UNIFORM), b[:, None],
                          (~l_own)[:, None, None], coff, count, crank, kl, t0, i0[None, :], t, ir, m1, B, n, W,
                          MARS)                                                 # (a block kept once: its slot)
        lv = tl.load(lp + sl, mask = vm & m1[None, :, :] & (sl >= 0), other = float("-inf"))       # L[i0, i]
        m2 = (ok & (q >= 0))[:, None] & (i0 < E)[None, :]
        so = _parent_lane(_slot(iv_table, opos, i0[None, :], q[:, None], X, m2, OUT_FULL), b[:, None], once, coff,
                          count, crank, kp, t0, i0[None, :], t + 1, q[:, None], m2, B, n, W, COMPACT)
        ov = tl.load(op + so[None, :, :], mask = vm & (m2 & (so >= 0))[None, :, :], other = float("-inf"))
        mx, acc = _lse_step_one_exp(mx, acc, lv + ov)                           # O_out[i0, next]
    return mx, acc


@triton.jit
def _cw_input_block_term(mx, acc, i, ok, q, op, rp, iv_table, opos, rpos, coff, count, crank, kp, kr, b, vm, once,
                         r_own, X, t, t2, end, B, n, W, TJ: tl.constexpr, S_KIND: tl.constexpr, S_FULL: tl.constexpr,
                         OUT_FULL: tl.constexpr, COMPACT: tl.constexpr, MARS: tl.constexpr, UNIFORM: tl.constexpr):
    """One column ``i`` of :func:`_class_weight_kernel`'s walk, ``input @ block`` form: the identity block
    (``S_KIND`` 2), the term ``O_out[i, q]``; else the terms ``O_out[i, j] + R[q, j]`` over the exit columns ``j``
    (``TJ`` at a time). Arguments as :func:`_cw_block_input_term`."""
    if S_KIND == 2:                                                             # the identity: the exit column is q
        jq = tl.where(end & (q >= 0), 0, q)
        qm = ok & (q >= 0)
        so = _parent_lane(_slot(iv_table, opos, i, jq, X, qm, OUT_FULL), b, once, coff, count, crank, kp, t, i, t2,
                          jq, qm, B, n, W, COMPACT)
        ov = tl.load(op + so[None, :, None], mask = vm & (qm & (so >= 0))[None, :, None], other = float("-inf"))
        mx, acc = _lse_step_one_exp(mx, acc, ov + tl.zeros_like(mx))
    else:
        for j0 in range(0, X, TJ):
            j = j0 + tl.arange(0, TJ)
            mo = ok[:, None] & (j < X)[None, :]
            if UNIFORM:
                ir = i
            else:
                ir = i[:, None]
            so = _parent_lane(_walk_slots(iv_table, opos, i, j, X, ok, j < X, True, OUT_FULL, UNIFORM), b[:, None],
                              once, coff, count, crank, kp, t, ir, t2, j[None, :], mo, B, n, W, COMPACT)
            ov = tl.load(op + so[None, :, :], mask = vm & (mo & (so >= 0))[None, :, :], other = float("-inf"))
            m2 = (ok & (q >= 0))[:, None] & (j < X)[None, :]                   # O_out[i, j]
            sr = _parent_lane(_slot(iv_table, rpos, q[:, None], j[None, :], X, m2, S_FULL), b[:, None],
                              (~r_own)[:, None, None], coff, count, crank, kr, t + 1, q[:, None], t2, j[None, :], m2,
                              B, n, W, MARS)                                    # (a block kept once: its slot)
            rv = tl.load(rp + sr, mask = vm & m2[None, :, :] & (sr >= 0), other = float("-inf"))   # R[next, j]
            mx, acc = _lse_step_one_exp(mx, acc, ov + rv)
    return mx, acc


@triton_jit
def _class_weight_kernel(dfa_class_flow, cw_col, cw_K, cw_count, cw_cls, cw_off, element_flows, temp_flows, node_mars,
                         prior_mars, temp, next_col, delta, state_id, col_of_state, cls_ptr, cls_col, cls_succ, live, width, iv_of,
                         iv_info, iv_table,
                         once_iv, count,
                         start, columns, coff, crank, dests, tiles, ptr, parents, reg_offset, reg_width, reg_first,
                         prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, tflow_width, B, n, W,
                         C, K, pid0, HOMOGENEOUS: tl.constexpr, BLOCK_INPUT: tl.constexpr, S_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                         PRUNE: tl.constexpr, OUT_FULL: tl.constexpr, S_FULL: tl.constexpr, UNIFORM: tl.constexpr,
                         COMPACT: tl.constexpr, MARS: tl.constexpr, STORE: tl.constexpr, WALK: tl.constexpr,
                         TN: tl.constexpr, SUBS: tl.constexpr, TB: tl.constexpr, TC: tl.constexpr, TJ: tl.constexpr,
                         PDL: tl.constexpr = False):
    """One program per (``TN`` destinations of a node tile, ``TB`` samples, ``TC`` classes), a lane per (sample,
    class): the log class weights of its steps' inputs (see the module docstring) -- a walk over the columns ``i`` of
    the input's boundary where the lane's class leads somewhere, then over the other boundary's columns ``TJ`` at a
    time -- log-added to the lane's row of ``dfa_class_flow`` -- its column's (the destination's position's first,
    ``cw_col``, plus its sample) live classes and their rows, ``cw_count`` / ``cw_cls`` / ``cw_off`` (see
    :class:`ClassWeightRows`; rows of ``cw_K`` floats per position) -- or stored there (``STORE``: the launch writes
    each of its destinations first in its query, see :mod:`..queries.backend.plans`). A node tile's destinations (see
    :func:`~.common.node_tiles`) are at one position, with steps to the same other boundaries: they walk the same
    columns, once for all of them; one exp a term (:func:`~.prod._lse_step_one_exp`).

    Two walks, the program choosing per its lanes (``WALK`` 0; 1 and 2 force one): over the CLASS's columns -- those
    where it leads somewhere, ``cls_ptr`` / ``cls_col`` / ``cls_succ`` (:class:`~.transitions.TransitionTables`),
    under evidence pruning those live for the lane's sample (``live`` [B, n + 1, W]) --, or over the SAMPLE's columns,
    every one of the boundary (``width``) or, under pruning, its live ones (``count`` / ``start`` / ``columns``, see
    :meth:`~.live.Liveness.lists`), the class's successor through the automaton's transitions (``delta``,
    ``state_id``, ``col_of_state``, see :func:`~.common._next_col`). The first costs the longest class list
    among the lanes, the second the columns walked: a sparse automaton with every column live (nothing observed) takes
    the first, a few live columns the second -- the sample walk's lanes sharing the column (``UNIFORM``: a scalar, see
    :func:`_walk_column`; ``TB`` 1 under pruning) look slots up once per column, so the class walk must be
    :data:`CLASS_WALK_UNIFORM_RATIO` times shorter to win there. A program's lanes take their column's live classes in
    the boundary's ``cls_order`` (most columns first), so that their lists are of a length; a dead class has no lane
    (nor row). Dead (column, class) pairs are -inf terms either way, so both walks give the same bits. ``COMPACT``: the
    outputs' top-down values compact (:mod:`.compact`; ``tflow_width`` the scratch rows' width there, ``temp_width``
    their inside values'); ``MARS``: the other operands' inside values too (by the same layout)."""
    if PDL:                                             # (the launch before it, done)
        gdc_wait()
    d, real, d0 = _tile_rows(tiles, tl.program_id(0), TN, SUBS)
    lane = tl.arange(0, TB * TC)
    b = ((pid0 + tl.program_id(1)) * TB + lane // TC).to(tl.int64)
    t = tl.load(dests + d0 * 2)
    rank = tl.program_id(2) * TC + lane % TC
    cm = (b < B) & (rank < C)
    nb = tl.load(cw_col + t) + b                                               # the lanes' columns: their live
    cm = cm & (rank < tl.load(cw_count + nb, mask = cm, other = 0))            # classes (:class:`ClassWeightRows`)
    cls = tl.load(cw_cls + nb * C + rank, mask = cm, other = 0)
    vm = real[:, None, None] & cm[None, :, None]                               # (destination, lane, column)
    k = tl.load(dests + d * 2 + 1)
    end = t + 1 == n
    M = tl.load(width + t).to(tl.int32)
    if PRUNE:                                                                  # the lanes' samples' live columns at t
        if UNIFORM:                                                            # (the program's one sample)
            row = (pid0 + tl.program_id(1)).to(tl.int64) * (n + 1) + t
            ccount = tl.load(count + row)
            cstart = tl.load(start + row)
            steps = ccount
        else:
            ccount = tl.load(count + b * (n + 1) + t, mask = cm, other = 0)
            cstart = tl.load(start + b * (n + 1) + t, mask = cm, other = 0)
            steps = tl.max(ccount, axis = 0)
    else:
        ccount = 0
        cstart = 0
        steps = M
    p0c = tl.load(cls_ptr + t * (C + 1) + cls, mask = cm, other = 0)           # the lanes' classes' columns
    p1c = tl.load(cls_ptr + t * (C + 1) + cls + 1, mask = cm, other = 0)
    L = tl.max(p1c - p0c, axis = 0)
    if WALK == 0:
        if UNIFORM:
            by_class = L * CLASS_WALK_UNIFORM_RATIO < steps
        else:
            by_class = L < steps
    elif WALK == 1:
        by_class = L >= 0
    else:
        by_class = L < 0
    mx = tl.full([TN, TB * TC, TJ], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TN, TB * TC, TJ], dtype = tl.float32)
    p0 = tl.load(ptr + d)
    f0 = tl.load(ptr + d0)
    for r in range(0, tl.load(ptr + d0 + 1) - f0):
        p = p0 + r
        f = f0 + r                                                             # (the tile's first destination's step)
        op = _out_flows_ptr(tl.load(parents + p * 7 + 0), element_flows, temp_flows, out_first, out_width,
                            tflow_width, OUT_TEMP)[:, None, None]
        if BLOCK_INPUT:                                                        # a block over [t0, t), then the input
            t0 = tl.load(parents + f * 7 + 3)
            E = tl.load(width + t0)
            X = _exit_width(width, t + 1, n)
            OS, opos, _ = _interval(iv_of, iv_info, t0, t + 1, n)
            LS, lpos, _ = _interval(iv_of, iv_info, t0, t, n)
            once = _once(once_iv, iv_of, t0, t + 1, n)
            ostride = tl.where(once, 0, OS)
            kp = tl.load(iv_of + t0 * (n + 1) + t + 1)
            op = op + _parent_base(b, ostride, COMPACT)[None, :, None]
            lp, l_own = _block_ptr(tl.load(parents + p * 7 + 1), tl.load(parents + p * 7 + 2), node_mars, prior_mars,
                                   temp, reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior,
                                   temp_width, S_KIND)
            lp = lp[:, None, None] + _parent_base(b[None, :], tl.where(l_own, LS, 0)[:, None], MARS)[:, :, None]
            kl = tl.load(iv_of + t0 * (n + 1) + t)
            if by_class:
                for ii in range(0, L):
                    ok = cm & (p0c + ii < p1c)
                    i = tl.load(cls_col + p0c + ii, mask = ok, other = 0)
                    q = tl.load(cls_succ + p0c + ii, mask = ok, other = -1)
                    if PRUNE:
                        ok = ok & (tl.load(live + (b * (n + 1) + t) * W + i, mask = ok, other = 0) != 0)
                    mx, acc = _cw_block_input_term(mx, acc, i, ok, q, op, lp, iv_table, lpos, opos, coff, count, crank,
                                                   kl, kp, b, vm, once, l_own, M, E, X, t0, t, end, B, n, W, TJ,
                                                   S_FULL, OUT_FULL, COMPACT, MARS, False)
            else:
                for ii in range(0, steps):
                    i, ok = _walk_column(columns, cstart, ccount, cm, ii, PRUNE, UNIFORM)
                    q = _next_col(next_col, delta, state_id, col_of_state, t, i, cls, ok, K, W, C, HOMOGENEOUS)
                    mx, acc = _cw_block_input_term(mx, acc, i, ok, q, op, lp, iv_table, lpos, opos, coff, count, crank,
                                                   kl, kp, b, vm, once, l_own, M, E, X, t0, t, end, B, n, W, TJ,
                                                   S_FULL, OUT_FULL, COMPACT, MARS, UNIFORM)
        else:                                                                  # the input, then a block over [t+1, t2)
            t2 = tl.load(parents + f * 7 + 5)
            X = _exit_width(width, t2, n)
            OS, opos, _ = _interval(iv_of, iv_info, t, t2, n)
            once = _once(once_iv, iv_of, t, t2, n)
            ostride = tl.where(once, 0, OS)
            kp = tl.load(iv_of + t * (n + 1) + t2)
            op = op + _parent_base(b, ostride, COMPACT)[None, :, None]
            if S_KIND != 2:
                RS, rpos, _ = _interval(iv_of, iv_info, t + 1, t2, n)
                rp, r_own = _block_ptr(tl.load(parents + p * 7 + 3), tl.load(parents + p * 7 + 4), node_mars,
                                       prior_mars, temp, reg_offset, reg_width, reg_first, prior_offset, prior_width,
                                       reg_prior, temp_width, S_KIND)
                rp = rp[:, None, None] + _parent_base(b[None, :], tl.where(r_own, RS, 0)[:, None], MARS)[:, :, None]
                kr = tl.load(iv_of + (t + 1) * (n + 1) + t2)
            else:                                                              # (unread)
                rpos = opos
                rp = op
                r_own = once
                kr = kp
            if by_class:
                for ii in range(0, L):
                    ok = cm & (p0c + ii < p1c)
                    i = tl.load(cls_col + p0c + ii, mask = ok, other = 0)
                    q = tl.load(cls_succ + p0c + ii, mask = ok, other = -1)
                    if PRUNE:
                        ok = ok & (tl.load(live + (b * (n + 1) + t) * W + i, mask = ok, other = 0) != 0)
                    mx, acc = _cw_input_block_term(mx, acc, i, ok, q, op, rp, iv_table, opos, rpos, coff, count, crank,
                                                   kp, kr, b, vm, once, r_own, X, t, t2, end, B, n, W, TJ, S_KIND,
                                                   S_FULL, OUT_FULL, COMPACT, MARS, False)
            else:
                for ii in range(0, steps):
                    i, ok = _walk_column(columns, cstart, ccount, cm, ii, PRUNE, UNIFORM)
                    q = _next_col(next_col, delta, state_id, col_of_state, t, i, cls, ok, K, W, C, HOMOGENEOUS)
                    mx, acc = _cw_input_block_term(mx, acc, i, ok, q, op, rp, iv_table, opos, rpos, coff, count, crank,
                                                   kp, kr, b, vm, once, r_own, X, t, t2, end, B, n, W, TJ, S_KIND,
                                                   S_FULL, OUT_FULL, COMPACT, MARS, UNIFORM)
    top = tl.max(mx, axis = 2)
    shift = tl.where(top == float("-inf"), 0.0, top)
    val = tl.log(tl.sum(acc * tl.exp(mx - shift[:, :, None]), axis = 2)) + shift
    dest = dfa_class_flow + (tl.load(cw_off + nb, mask = cm, other = 0) + rank * tl.load(cw_K + t))[None, :] + \
        k[:, None]
    dm = real[:, None] & cm[None, :]
    if STORE:
        tl.store(dest, val, mask = dm)
    else:
        tl.store(dest, _logaddexp_tile(tl.load(dest, mask = dm, other = float("-inf")), val), mask = dm)


#: Whether the class weights of ``input @ block`` steps with one exit column and one step a destination run the CUDA
#: kernels (kernels/c/class_weights.cu, :func:`_class_weight_cuda`: the destinations' rows staged in shared memory, the
#: class lists pre-filtered and interleaved by lane) where they apply -- compact storage under evidence pruning, a
#: non-identity block, at most ``CLASS_WEIGHT_CUDA_CLASSES`` classes, the launch's longest live walk (the most live
#: columns of a sample at its positions) at least ``CLASS_WEIGHT_CUDA_WALK`` --, the Triton kernel elsewhere. The two
#: give the same bits (the kernels' terms in the Triton kernel's order with its arithmetic). The CUDA launch has a floor
#: (~70 us: 16k CTAs staging their rows) where the Triton one costs ~16-26 us on a few live columns; past ~40 live
#: columns the Triton kernel is bound by the L2 traffic of its per-lane gathers (ncu: 77% of the L2's peak, 5 of every
#: 32 bytes used) and the CUDA kernels are 2-5x faster. MEASURED (HMM-4096 x hard_0 B8 diffusion, 32 calls, nsys,
#: RTX PRO 6000; launches paired between the two sessions): class weights 1230 ms on Triton, 783 on CUDA, 457 under the
#: rule (walk >= 41: the best threshold; the per-launch best of both 454).
CLASS_WEIGHT_CUDA = os.environ.get("PYJUICE_CW_CUDA", "1") == "1"
CLASS_WEIGHT_CUDA_CLASSES = 4096
CLASS_WEIGHT_CUDA_WALK = int(os.environ.get("PYJUICE_CW_WALK", "40"))
#: destinations a CTA (1, 2 or 4, the most whose rows fit in shared memory) and threads a CTA; the entry tables'
#: work space at most (bytes)
CLASS_WEIGHT_CUDA_DESTS = int(os.environ.get("PYJUICE_CW_DESTS", "2"))
CLASS_WEIGHT_CUDA_THREADS = int(os.environ.get("PYJUICE_CW_THREADS", "256"))
CLASS_WEIGHT_CUDA_TABLE_MAX = 1 << 28
#: the shared memory a CTA's staged rows may take (bytes; sm_80+ opt in to ~99 KB, less the kernel's own 128 B)
CLASS_WEIGHT_CUDA_SMEM = 96 * 1024


def _long_walk(la, rows) -> bool:
    """Whether the launch's longest live walk reaches :data:`CLASS_WEIGHT_CUDA_WALK` (host values, no wait)."""
    bounds = la.host["walk"].get("bounds")
    if bounds is None:
        bounds = la.host["walk"]["bounds"] = sorted(set(la.host["dests"][:, 0].tolist()))
    walk = rows.walk
    return max(walk[t] for t in bounds) >= CLASS_WEIGHT_CUDA_WALK


def _class_weight_boundaries(prog) -> None:
    """Into ``host["walk"]`` of every class weight launch of ``prog``'s :class:`~..compiled.circuit.Assembly`, what its
    CUDA tables say whatever the automaton (a launch's dict, the same for its program of any constraint): ``ts``, its
    destinations' boundaries, ``tlist`` / ``t2i`` the same on the device, in one copy (copies wait for the stream), and
    the most steps of a destination."""
    launches = [cw.host for kind, _, layers in prog.assembly.top_down.steps if kind == "prod" for _, stages, _, _ in layers
                for _, _, _, cws in stages for cw in cws]
    n, arrays, ts = prog.n, [], []
    for h in launches:
        x = np.unique(h["dests"][:, 0])
        t2i = np.full(n + 1, -1, dtype = np.int32)
        t2i[x] = np.arange(x.size, dtype = np.int32)
        ts.append(x.tolist())
        arrays += [x.astype(np.int32), t2i]
    buf, starts = pack_aligned(arrays, np.int32)
    tables, starts = torch.from_numpy(buf).to(prog.dev), starts.tolist()
    for k, (h, x) in enumerate(zip(launches, ts)):
        ptr = np.asarray(h["ptr"])
        h["walk"].update(ts = x, tlist = tables[starts[2 * k]:starts[2 * k] + len(x)],
                         t2i = tables[starts[2 * k + 1]:starts[2 * k + 1] + n + 1], T = len(x),
                         steps = int((ptr[1:] - ptr[:-1]).max(initial = 0)))


def _class_weight_cuda(mod, la, tiles, height, cw, rows, prog, bufs, element_flows, temp_flows, node_mars, liveness,
                       compact, prior, regions, elem_first, elem_width, temp_width, tflow_width, B, s_kind, out_temp,
                       out_full, s_full, inside, store) -> bool:
    """One launch on the CUDA kernels (see :data:`CLASS_WEIGHT_CUDA`), or False where they do not apply (several steps
    a destination, rows too wide for shared memory, entry tables past :data:`CLASS_WEIGHT_CUDA_TABLE_MAX`, a CUDA graph
    being captured before the launch's tables exist): the caller runs the Triton kernel then."""
    st = la.host["walk"]
    if "ts" not in st:
        if torch.cuda.is_current_stream_capturing():
            return False
        _class_weight_boundaries(prog)
    # (what the automaton says of its boundaries, of a program's launches: once)
    kept = prog.cw_walks.get(la.number)
    if kept is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        ts = st["ts"]
        maxl = getattr(prog, "_cw_max_list", None)
        if maxl is None:                                               # every boundary's longest class list (once)
            cp = prog.trans.cls_ptr.view(-1, prog.C + 1)
            maxl = prog._cw_max_list = (cp[:, 1:] - cp[:, :-1]).amax(1).cpu().numpy()
        kept = prog.cw_walks[la.number] = (max(1, int(max(maxl[t] for t in ts))), max(prog.width[t] for t in ts),
                                           max(prog.width[t + 1] for t in ts))
    if st["steps"] != 1:
        return False
    longest, wt, wt1 = kept
    lanes = host_cdiv(max(rows.most, 1), 32) * 32
    if st["T"] * B * lanes * longest * 4 > CLASS_WEIGHT_CUDA_TABLE_MAX:
        return False
    fits = [g for g in (4, 2, 1) if g <= CLASS_WEIGHT_CUDA_DESTS and height % g == 0 and
            mod.cw_stage_smem(g, wt, wt1) <= CLASS_WEIGHT_CUDA_SMEM]
    if not fits:
        return False
    TN = fits[0]
    temp, once_iv = bufs
    count, start, columns = liveness.lists()
    tab = torch.empty(st["T"] * B * lanes * longest, dtype = torch.int32, device = prog.dev)
    cnt = torch.empty(st["T"] * B * lanes, dtype = torch.int32, device = prog.dev)
    mod.cw_class_cuda(cw["buf"], cw["col"], cw["K"], rows.count, rows.cls, rows.off, element_flows, temp_flows,
                      node_mars, prior["mars"], temp, prog.width_t, prog.intervals.of, prog.intervals.info,
                      prog.intervals.table, once_iv, prog.trans.cls_ptr, prog.trans.cls_col, prog.trans.cls_succ,
                      liveness.live, count, start, columns, compact.rank, compact.off, la.dests, tiles, la.ptr,
                      la.parents, regions["offset"], regions["width"], regions["first"], prior["offset"],
                      prior["width"], prior["reg_prior"], st["tlist"], st["t2i"], tab, cnt, int(elem_first),
                      int(elem_width), int(temp_width), int(tflow_width), B, prog.n, prog.W, prog.C, lanes,
                      longest, wt, wt1, height // TN, TN, CLASS_WEIGHT_CUDA_THREADS, int(s_kind),
                      bool(out_temp), bool(out_full), bool(s_full), bool(inside), bool(store))
    return True


#: ... a column of the sample walk whose lanes share it costs this many times less than one of the class walk (ex108
#: l2r, every column live at the first call: the class walk over 90% of the 13348 columns took 240 ms against 52)
CLASS_WALK_UNIFORM_RATIO = tl.constexpr(4)
