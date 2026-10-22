"""
The conditionals' kernels (see :mod:`.queries.conditional`): the class weights of the target positions' input nodes,
read off the top-down pass, and the scoring of every token from them.

Token ``v`` of class ``c`` at position ``t`` moves column ``i`` to ``next_col[t, i, c]`` with probability ``p_u(v)`` at
input node ``u``, so ``p(x_t = v, e, C) = sum_u p_u(v) A_u[c]``, with ``A_u[c] = sum_i exp(O_u[i, next_col[t, i, c]])``
the CLASS WEIGHTS of ``u``'s top-down (log outside) block ``O_u``. An input node has no block of its own in the lifted
circuit: it is an operand of a product's step (:mod:`.prod`), and its top-down block is the transpose of that step,
from the step's output's top-down value and the other operand's inside value:

* ``input @ block`` (``out[i, j] = sum_q In[i, q] R[q, j]``): ``A[c] = sum_(i, j) exp(O_out[i, j] + R[next(t, i, c), j])``;
  the identity block (a product of one input, or an input next to another): ``A[c] = sum_i exp(O_out[i, next(t, i,
  c)])``;
* ``block @ input`` (``out[i0, q] = sum_i L[i0, i] In[i, q]``): ``A[c] = sum_(i0, i) exp(L[i0, i] + O_out[i0, next(t, i,
  c)])``,

a sequence end's successors being its one column. Tied input nodes share their parameters' row: the weights are summed
per (position, row of the class masses), :data:`dfa_class_flow` -- ``[target positions x samples, classes, rows of the
position's row group]`` per row group, log -- and scored per row group against the rows' token-major probabilities:
``log p(x_t = v | e, C) = log sum_k T[v, k] A[k, class(v)] - log p(e, C)``, with ``p(e, C) = sum_(k, c) A[k, c]
mass_k(c)``: the stored classes' terms from the class masses (no second pass over the tokens), an enumerated class's
(one token, no column in the masses) from its token's unnormalized score, the dot the scoring computes anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.constraints.distributions import find_token
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from .common import (IDENTITY, NODE_TILE, _block_ptr, _exit_width, _interval, _next_col, _logaddexp_tile, _lse_step, _slot,
                     _tf32_operand, _tile_rows, grid_chunks, node_groups, node_tiles, tensor)
from .prod import _lse_step_one_exp
from .prod_top_down import _once, _out_flows_ptr, _parent_base, _parent_lane
from .sum import _num_sms



@dataclass
class ScoreTables:
    """
    Where every position's conditionals are scored. The input nodes over a position read rows of the class masses
    (:meth:`~..state.State.class_mass_rows`; tied nodes one row), the rows of their
    parameter tables: positions whose input nodes read the same rows form a ROW GROUP, scored together against one
    token-major table of those rows (an HMM with tied emissions: one group).

    :ivar group_of: int64 [n] (host): every position's row group
    :ivar rows: per row group, the sorted rows its positions' input nodes read (host int64)
    :ivar rows_t: the same on the device
    :ivar col_of: int64 [input nodes] (host, by ``node_mars`` row from the inputs' first): every input node's column
        in its position's row group, -1 for an input node no product step reads
    :ivar nodes: per row group, int64 [n, rows of the group] (host): the input node (from the inputs' first) at every
        position of the group and column, -1 for none
    """

    group_of: np.ndarray
    rows: list
    rows_t: list
    col_of: np.ndarray
    nodes: list


def score_tables(prog) -> ScoreTables:
    """The :class:`ScoreTables` of a lifted :class:`~.program.Program` (built on first use, kept on it): the input
    operands of its product steps, by position."""
    if getattr(prog, "score", None) is not None:
        return prog.score
    mass_row = prog.mass_row.cpu().numpy().astype(np.int64)
    u, t = [], []
    for kind, _, layers in prog.steps:
        if kind != "prod":
            continue
        for stages, _, _ in layers:
            for launches in stages:
                for la in launches:
                    cols = {"input_block": (1, 2), "block_input": (4, 5)}.get(la.form)
                    if cols is not None:
                        u.append(la.rows[:, cols[0]].numpy())
                        t.append(la.rows[:, cols[1]].numpy())
    u = np.concatenate(u) - prog.input_start if u else np.zeros(0, dtype = np.int64)
    t = np.concatenate(t) if t else np.zeros(0, dtype = np.int64)
    group_of = np.full(prog.n, -1, dtype = np.int64)
    col_of = np.full(prog.input_end - prog.input_start, -1, dtype = np.int64)
    groups, rows, nodes = {}, [], []
    for pos in range(prog.n):
        at = u[t == pos]
        if at.size == 0:
            continue
        r = np.unique(mass_row[at])
        g = groups.setdefault(r.tobytes(), len(rows))
        if g == len(rows):
            rows.append(r)
            nodes.append(np.full((prog.n, len(r)), -1, dtype = np.int64))
        group_of[pos] = g
        col_of[at] = np.searchsorted(r, mass_row[at])
        nodes[g][pos, col_of[at]] = at
    prog.score = ScoreTables(group_of = group_of, rows = rows,
                             rows_t = [torch.from_numpy(r).to(prog.dev) for r in rows], col_of = col_of, nodes = nodes)
    return prog.score


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
    :ivar J_max: the most exit columns (``input_block``) or entry columns (``block_input``) of the outputs
    :ivar host: for query plans, numpy copies of ``dests``, ``ptr`` and ``parents``, and every destination's node tile
        ``group`` (:func:`~.common.node_groups`: its position, and its steps' other boundaries in order)
    :ivar tiles, tile_height: every destination in node tiles (int32 on the device) and their height
        (:func:`~.common.node_tiles`) -- a query plan's lists of destinations are tiles too
    """

    form: str
    kinds: tuple
    dests: torch.Tensor
    ptr: torch.Tensor
    parents: torch.Tensor
    J_max: int
    host: Any
    tiles: torch.Tensor
    tile_height: int


def class_weight_launch(launch, alive: np.ndarray, prog, score: ScoreTables) -> "ClassWeightLaunch | None":
    """The :class:`ClassWeightLaunch` of a forward launch's input operands, ``alive`` marking the steps whose output
    has a top-down value (the others add nothing); None when there is no such step or no input operand."""
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
    if launch.form == "input_block":                                           # exit columns at t2
        J = np.where(rows[:, 5] == prog.n, 1, np.asarray(prog.width)[rows[:, 5]])
    else:                                                                      # entry columns at t0
        J = np.asarray(prog.width)[rows[:, 3]]
    count = np.diff(ptr)                                    # destinations walking the same columns, together: at
    other = np.full((len(starts), int(count.max())), -1, dtype = np.int64)   # one position, steps to the same ends
    other[np.repeat(np.arange(len(starts)), count), np.arange(len(t)) - np.repeat(starts, count)] = \
        rows[:, 5 if launch.form == "input_block" else 3]
    group = node_groups(dests[:, 0], other)
    tiles, height = node_tiles(group)
    i32 = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(prog.dev, torch.int32)
    return ClassWeightLaunch(launch.form, launch.kinds, i32(dests), i32(ptr), i32(rows), int(J.max()),
                             dict(dests = dests, ptr = ptr, parents = rows, group = group), i32(tiles), height)


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
                         TN: tl.constexpr, SUBS: tl.constexpr, TB: tl.constexpr, TC: tl.constexpr, TJ: tl.constexpr):
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


#: :func:`_class_weight_kernel`'s walk: 0, every program chooses (its lanes' longest class list against the columns
#: it would walk otherwise); 1, the classes' columns; 2, the sample's columns (tests)
CLASS_WEIGHT_WALK = 0
#: ... a column of the sample walk whose lanes share it costs this many times less than one of the class walk (ex108
#: l2r, every column live at the first call: the class walk over 90% of the 13348 columns took 240 ms against 52)
CLASS_WALK_UNIFORM_RATIO = tl.constexpr(4)


#: destinations (of a node tile), and destinations x lanes (sample, class) x other-boundary columns, per
#: :func:`_class_weight_kernel` program, at most (Codpula diffusion, 4096 latents: 16 x 32 lanes 1.09 ms per call,
#: 8 x 32 1.16, 8 x 128 1.34, 16 x 128 1.65; 1 x 128, every destination its own walk, 2.29)
CLASS_WEIGHT_NODES = 16
CLASS_WEIGHT_LANES = 512

#: :func:`_live_classes_kernel`'s classes at a time (at most; fewer until its programs fill the device), and its lanes:
#: classes x their pairs (ex108 l2r, 15 classes over boundaries of up to 13824 columns: a lane a class, walking its list
#: alone, 1.2 ms a call; 1024 lanes over a column's classes, 16 programs, 0.19)
LIVE_CLASS_TILE = 64
LIVE_CLASS_LANES = 1024


@dataclass
class ClassWeightRows:
    """
    Where a conditional query keeps its class weights in ``dfa_class_flow`` (:func:`class_weight_rows`): the LIVE
    rows alone. A target column's (position ``t``, sample ``b``) class ``c`` is live when one of the (column,
    successor) pairs of ``c``'s list at ``t`` (``Program.trans``' by-class lists) is live at both ends for the sample;
    only then can its weights be finite -- and, under a session's per-sample inputs, when the sample allows ``c`` there
    (:attr:`~.live.Liveness.allowed`): a class they rule out has no token of mass, so its tokens score -inf whatever
    its weights. Every column's live classes, in the boundary's class order (``cls_order``), are consecutive rows of
    ``K`` floats (its row group's rows).

    :ivar count: int32 [N], every column's live classes
    :ivar cls: int32 [N, C], every column's live classes: ``cls[m, :count[m]]``
    :ivar off: int64 [N + 1], every column's first row's float in ``dfa_class_flow`` (the last: the size)
    :ivar row: int32 [N, C], every (column, class)'s row among its row group's live ones, -1 if dead
    :ivar starts: every row group's first float, then the size (host; read on first use, see :attr:`most`)
    :ivar most: the most live classes of any column (host): copied with :attr:`starts` as the rows are built, and
        waited for on first use -- a query asks after its forward pass's launches, so the wait is the rows' alone
    """

    count: torch.Tensor
    cls: torch.Tensor
    off: torch.Tensor
    row: torch.Tensor
    _host: torch.Tensor
    _done: Any
    _read: Optional[tuple] = None

    def _sizes(self):
        if self._read is None:
            self._done.synchronize()
            v = self._host.tolist()
            self._read = (v[:-1], int(v[-1]))
        return self._read

    @property
    def starts(self) -> list:
        return self._sizes()[0]

    @property
    def most(self) -> int:
        return self._sizes()[1]


@triton_jit
def _live_classes_kernel(live, cls_ptr, cls_col, cls_succ, col_pos, col_K, row, floats, n, W, C, B, allowed, a_stride,
                         TC: tl.constexpr, TP: tl.constexpr, PRUNE: tl.constexpr, ALLOWED: tl.constexpr):
    """One program per target column ``m`` (sample ``m % B`` of position ``col_pos[m // B]``) and tile of ``TC``
    classes: which of them are live -- a class whose list (``cls_ptr`` / ``cls_col`` / ``cls_succ``) has a pair live
    at both ends for the sample (``live`` int8 [B, n + 1, W]; without ``PRUNE``, every class with a pair), its pairs
    ``TP`` at a time, and -- ``ALLOWED`` -- that the sample allows there (``allowed`` int8 [Be, n, C], ``a_stride`` a
    sample's flags apart; a class it rules out is not walked) -- into ``row[m, c]`` (0 live, -1 dead), and their
    rows' floats (``col_K[m]`` a row) added to ``floats[m]``."""
    m = tl.program_id(0).to(tl.int64)
    t = tl.load(col_pos + m // B)
    b = m % B
    c = tl.program_id(1) * TC + tl.arange(0, TC)
    cm = c < C
    if ALLOWED:
        cm = cm & (tl.load(allowed + b * a_stride + t * C + c, mask = cm, other = 0) != 0)
    p0 = tl.load(cls_ptr + t * (C + 1) + c, mask = cm, other = 0)
    p1 = tl.load(cls_ptr + t * (C + 1) + c + 1, mask = cm, other = 0)
    if PRUNE:
        hit = tl.zeros([TC], dtype = tl.int32)
        for k0 in range(0, tl.max(p1 - p0, axis = 0), TP):
            k = p0[:, None] + k0 + tl.arange(0, TP)[None, :]                     # [TC, TP]: the classes' pairs
            pm = cm[:, None] & (k < p1[:, None])
            i = tl.load(cls_col + k, mask = pm, other = 0)
            q = tl.load(cls_succ + k, mask = pm, other = 0)
            a = tl.load(live + (b * (n + 1) + t) * W + i, mask = pm, other = 0)
            z = tl.load(live + (b * (n + 1) + t + 1) * W + q, mask = pm, other = 0)
            hit = hit | tl.max(((a != 0) & (z != 0)).to(tl.int32), axis = 1)
        on = cm & (hit != 0)
    else:
        on = cm & (p1 > p0)
    tl.store(row + m * C + c, tl.where(on, 0, -1), mask = c < C)
    tl.atomic_add(floats + m, tl.sum(on.to(tl.int64), axis = 0) * tl.load(col_K + m))


@triton_jit
def _class_rows_kernel(off, cls_order, col_pos, col_K, col_first, group_first, row, count, cls, sizes, N, G, C, B,
                       TC: tl.constexpr):
    """One program per target column ``m`` (from ``off``, every column's first float): its live classes (``row[m, c]
    >= 0``), in the boundary's class order, into ``cls[m, :count[m]]``, and their rows among their row group's
    (``col_first[m]`` its first column) into ``row[m]``; the most live classes of a column into ``sizes[G + 1]`` (0
    already) -- and program ``g <= G`` the ``g``-th row group's first float (``group_first``) into ``sizes[g]``."""
    m = tl.program_id(0).to(tl.int64)
    if m <= G:
        tl.store(sizes + m, tl.load(off + tl.load(group_first + m)))
    if m < N:
        t = tl.load(col_pos + m // B)
        K = tl.load(col_K + m)
        first = tl.load(off + m)
        local = (first - tl.load(off + tl.load(col_first + m))) // K
        tl.store(count + m, ((tl.load(off + m + 1) - first) // K).to(tl.int32))
        kept = tl.zeros([], dtype = tl.int32)
        for c0 in range(0, C, TC):
            rank = c0 + tl.arange(0, TC)
            cm = rank < C
            c = tl.load(cls_order + t * C + rank, mask = cm, other = 0)
            on = cm & (tl.load(row + m * C + c, mask = cm, other = -1) >= 0)
            v = on.to(tl.int32)
            r = kept + tl.cumsum(v, axis = 0) - v
            tl.store(row + m * C + c, (local + r).to(tl.int32), mask = on)
            tl.store(cls + m * C + r, c, mask = on)
            kept += tl.sum(v, axis = 0)
        tl.atomic_max(sizes + G + 1, kept.to(tl.int64))


def class_weight_rows(prog, liveness, col_pos: torch.Tensor, col_K: torch.Tensor, col_first: torch.Tensor,
                      group_first: torch.Tensor, B: int) -> ClassWeightRows:
    """
    The :class:`ClassWeightRows` of a query's target columns (the row groups' consecutive, a position's ``B``
    samples consecutive in a row group's), from its :class:`~.live.Liveness` (None: every column live). Nothing
    waits for the device: the row groups' sizes and the most live classes of a column are copied as the rows are
    built, and read on first use (:attr:`ClassWeightRows.most`).

    :param col_pos: int64 [N / B] on the device, every block of ``B`` columns' position
    :param col_K, col_first: int64 [N] on the device, every column's row group's rows and first column
    :param group_first: int64 [row groups + 1] on the device, every row group's first column, then ``N``
    """
    tt, n, W, C = prog.trans, prog.n, prog.W, prog.C
    N, G, dev = col_K.numel(), group_first.numel() - 1, prog.dev
    M, a, b = max(N, 1), N + 1 + (N + 1) % 2, G + 2 + G % 2                 # (pieces 16 B aligned)
    z = torch.zeros(a + b + M, dtype = torch.int64, device = dev)
    off, sizes, floats = z.as_strided((N + 1,), (1,)), z.as_strided((G + 2,), (1,), a), z.as_strided((M,), (1,), a + b)
    NC = M * C + (-M * C) % 4
    e = torch.empty(2 * NC + M, dtype = torch.int32, device = dev)
    cls, row, count = e.as_strided((M, C), (C, 1)), e.as_strided((M, C), (C, 1), NC), e.as_strided((M,), (1,), 2 * NC)
    if N > 0:
        prune = liveness is not None
        classes = prune and liveness.allowed is not None                    # (a session's per-sample classes)
        a_stride = n * C if classes and liveness.allowed.size(0) > 1 else 0  # (one set: every sample's)
        TC = min(LIVE_CLASS_TILE, host_next_power_of_2(C))
        while prune and TC > 1 and N * host_cdiv(C, TC) < 4 * _num_sms(dev):   # (programs filling the device)
            TC //= 2
        _live_classes_kernel[(N, host_cdiv(C, TC))](liveness.live if prune else row, tt.cls_ptr, tt.cls_col,
                                                     tt.cls_succ, col_pos, col_K, row, floats, n, W, C, B,
                                                     liveness.allowed if classes else row, a_stride, TC = TC,
                                                     TP = max(1, LIVE_CLASS_LANES // TC), PRUNE = prune,
                                                     ALLOWED = classes, num_warps = 4)
        torch.cumsum(floats, dim = 0, out = off[1:])
    TC = min(LIVE_CLASS_TILE, host_next_power_of_2(C))
    _class_rows_kernel[(max(N, G + 1),)](off, tt.cls_order, col_pos, col_K, col_first, group_first, row, count, cls,
                                         sizes, N, G, C, B, TC = TC)
    host = torch.empty(G + 2, dtype = torch.int64, pin_memory = True)
    host.copy_(sizes, non_blocking = True)
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(dev.index))
    return ClassWeightRows(count = count, cls = cls, off = off, row = row, _host = host, _done = done)


def run_class_weights(launches, cw: dict, prog, bufs, element_flows, temp_flows, node_mars, liveness, elem_first: int,
                      elem_width: int, B: int, prior: dict, once_iv: torch.Tensor, regions: dict, compact = None,
                      tflow_width: Optional[int] = None, inside: bool = False):
    """
    The class weights of one product layer stage's input operands at the query's target positions: every
    :class:`ClassWeightLaunch` in ``launches`` as ``(launch, index list of destinations (None: all), store)`` --
    ``store``: the launch writes each of its destinations first in the query, and stores the weights there; the others
    log-add them.

    :param cw: ``buf`` (``dfa_class_flow``: -inf where a launch log-adds and nothing was written yet), ``rows`` (its
        :class:`ClassWeightRows`: every target column's live classes, and where their rows are), and per position its
        first target column (``col``, int64 [n], -1 for a position no sample targets) and its row group's rows (``K``,
        int64 [n])
    :param bufs: ``(temp, temp_width)``, the layer's scratch rows (the folds' inside values)
    :param liveness: None, or the query's :class:`~.live.Liveness` (its live columns' lists are built on first use)
    :param prior: where the sum operands kept once are (see :func:`~.common._block_ptr`)
    :param once_iv: the plan's per-interval flag of top-down values kept once (read with sample stride 0)
    :param compact: None, or the query's :class:`~.compact.CompactLayout`: the outputs' top-down values are compact,
        ``elem_width`` and ``tflow_width`` the widths of their rows (``regions``: the sum operands' inside values')
    :param tflow_width: the scratch rows' width in ``temp_flows`` (None: ``bufs``')
    :param inside: whether the other operands' per-sample inside values are compact too (by ``compact``'s layout;
        ``regions`` and ``bufs`` are their rows')
    """
    temp, temp_width = bufs
    tflow_width = temp_width if tflow_width is None else tflow_width
    n, W, C = prog.n, prog.W, prog.C
    ivs, tt = prog.intervals, prog.trans
    prune = liveness is not None
    lists = liveness.lists() if prune else (cw["K"],) * 3
    ctables = (compact.off, compact.rank) if compact is not None else (cw["K"],) * 2
    rows = cw["rows"]
    TC = min(64, host_next_power_of_2(max(rows.most, 1)))                   # (a lane a live class of a column)
    for la, item, store in launches:
        idx = tensor(item)
        tiles, height = (la.tiles, la.tile_height) if idx is None else (idx, item.height)
        s_kind, out_temp, (out_full, *s_full) = la.kinds
        identity = la.form == "input_block" and s_kind == IDENTITY
        TJ = 1 if identity else min(64, host_next_power_of_2(la.J_max))
        TN = min(height, CLASS_WEIGHT_NODES, max(1, CLASS_WEIGHT_LANES // (TC * TJ)))   # (powers of 2)
        TB = max(1, min(host_next_power_of_2(B), 128 // TC, CLASS_WEIGHT_LANES // (TC * TJ * TN)))
        lanes = TN * TB * TC * TJ
        warps = 1 if lanes <= 128 else 2 if lanes <= 512 else 4
        for first, size in grid_chunks(host_cdiv(B, TB), 1):
            _class_weight_kernel[(tiles.numel() // NODE_TILE * (height // TN), size, host_cdiv(max(rows.most, 1), TC))](
                cw["buf"], cw["col"], cw["K"], rows.count, rows.cls, rows.off, element_flows, temp_flows, node_mars,
                prior["mars"], temp, prog.next_col_arg, prog.delta, prog.state_id, prog.col_of_state, tt.cls_ptr,
                tt.cls_col, tt.cls_succ,
                liveness.live if prune else cw["K"], prog.width_t,
                ivs.of, ivs.info, ivs.table, once_iv, *lists, *ctables,
                la.dests, tiles, la.ptr, la.parents, regions["offset"], regions["width"], regions["first"],
                prior["offset"], prior["width"], prior["reg_prior"], elem_first, elem_width, temp_width, tflow_width,
                B, n, W, C, prog.K, first, BLOCK_INPUT = la.form == "block_input", S_KIND = s_kind,
                OUT_TEMP = out_temp,
                PRUNE = prune, OUT_FULL = out_full, S_FULL = all(s_full), UNIFORM = not prune or TB == 1,
                COMPACT = compact is not None, MARS = inside, STORE = store, WALK = CLASS_WEIGHT_WALK, TN = TN,
                HOMOGENEOUS = prog.homogeneous,
                SUBS = height // TN, TB = TB, TC = TC, TJ = TJ, num_warps = warps)


# -------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------

@triton.jit
def _column_rows(nodes, mrow, mstride, b, ks, km, STRIDE: tl.constexpr):
    """The class mass rows of a column's input nodes ``ks`` (its position's ``nodes``, its sample ``b``)."""
    u = tl.load(nodes + ks, mask = km, other = 0)
    r = tl.load(mrow + u, mask = km, other = 0).to(tl.int64)
    if STRIDE:
        r += b * tl.load(mstride + u, mask = km, other = 0).to(tl.int64)
    return r


@triton_jit
def _score_weights_kernel(w, row, class_mars, rows, nodes, positions, mrow, mstride, lognorm, shift, zc, part,
                          class_of_stored, C, K, B, KS, Cs, TC: tl.constexpr, TK: tl.constexpr,
                          PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr, SHARES: tl.constexpr, FINISH: tl.constexpr):
    """One program per (column ``n``, tile of ``TC`` STORED classes, share ``s`` of the rows ``[s KS, (s + 1) KS)``) of a
    row group's log class weights ``w`` [live rows, K], (column, class)'s row ``row[n, c]`` (-1: dead, its weights
    -inf; see :class:`ClassWeightRows`) -- an enumerated class's weights are left as they are: its token's
    dot takes them in log (:func:`_log_dot`). For stored column ``x`` (class ``class_of_stored[x]``): ``shift[n, c]``,
    the maximum over the rows of the class's weights (``PER_COLUMN``, the class masses of each column's own, see
    :class:`~..state.ClassMasses`: less the row's ``lognorm`` -- a token's own weight ``q(v) = p(v) w(v) / Z`` is
    ``p(v)``'s times ``w(v) / Z``), the weights shifted by it and exponentiated in place, and ``zc[n, x] = log sum_k
    exp(w[n, c, k] + mass_k(c))``, the class's part of ``p(e, C)``, from the mass table. Column ``n`` is sample ``n %
    B`` of position ``positions[n // B]``, whose input nodes are that row of ``nodes`` ([positions of the PC, K]):
    node ``u`` reads row ``mrow[u] + b mstride[u]`` (``STRIDE``; else ``mrow[u]``) of the masses; without per-column
    masses every column reads ``rows``. One share (``SHARES`` 1): all of it. Several: the shares' maxima, running
    maxima and sums into ``part`` ([N, class tiles, SHARES, 3, TC]), then (``FINISH``, a second launch) every share
    combines them in order and writes its rows. A program none of whose classes is live for its column (most of them:
    a column has a few live stored classes, an observed one none) skips its walks over the rows -- every load of
    them would be masked off -- and writes what they would have left: -inf shifts and parts."""
    n = tl.program_id(0).to(tl.int64)
    ct = tl.program_id(1)
    sh_ = tl.program_id(2)
    xs = ct * TC + tl.arange(0, TC)
    cm = xs < Cs
    cs = tl.load(class_of_stored + xs, mask = cm, other = 0).to(tl.int64)
    rr = tl.load(row + n * C + cs, mask = cm, other = -1)                    # the classes' rows (-1: dead, -inf)
    lv = cm & (rr >= 0)
    any_live = tl.max(lv.to(tl.int32), axis = 0) > 0
    base = w + tl.maximum(rr, 0).to(tl.int64) * K
    t = tl.load(positions + n // B)
    b = n % B
    k_lo = sh_ * KS
    k_hi = tl.where(any_live, tl.minimum(K, k_lo + KS), k_lo)               # (no live class: no rows to walk)
    if not FINISH:
        big = tl.full([TC], float("-inf"), dtype = tl.float32)
        mz = tl.full([TC], float("-inf"), dtype = tl.float32)
        az = tl.zeros([TC], dtype = tl.float32)
        for k0 in range(k_lo, k_hi, TK):
            ks = k0 + tl.arange(0, TK)
            km = ks < k_hi
            if PER_COLUMN:
                r = _column_rows(nodes + t * K, mrow, mstride, b, ks, km, STRIDE)
                lz = tl.load(lognorm + r, mask = km, other = 0.0)
            else:
                r = tl.load(rows + ks, mask = km, other = 0).to(tl.int64)
                lz = tl.zeros([TK], dtype = tl.float32)
            mk = lv[:, None] & km[None, :]
            v = tl.load(base[:, None] + ks[None, :], mask = mk, other = float("-inf"))                  # [TC, TK]
            x = v + tl.load(class_mars + r[None, :] * Cs + xs[:, None], mask = mk, other = float("-inf"))
            mr = tl.max(x, axis = 1)                     # (selects, not maxima: Triton's thread-locality pass asserts
            m_new = tl.where(mr > mz, mr, mz)            # on a reduction accumulated by max whose result has several
            sh = tl.where(m_new == float("-inf"), 0.0, m_new)                             # uses after the loop)
            az = az * tl.exp(mz - sh) + tl.sum(tl.exp(x - sh[:, None]), axis = 1)
            mz = m_new
            br = tl.max(v - lz[None, :], axis = 1)
            big = tl.where(br > big, br, big)
        if SHARES > 1:
            out = part + (((n * tl.num_programs(1) + ct) * SHARES + sh_) * 3) * TC + tl.arange(0, TC)
            tl.store(out, big)
            tl.store(out + TC, mz)
            tl.store(out + 2 * TC, az)
            return
    else:
        big = tl.full([TC], float("-inf"), dtype = tl.float32)
        mz = tl.full([TC], float("-inf"), dtype = tl.float32)
        for j in range(SHARES):                                             # (in order: the same in every share)
            src = part + (((n * tl.num_programs(1) + ct) * SHARES + j) * 3) * TC + tl.arange(0, TC)
            big = tl.maximum(big, tl.load(src))
            mz = tl.maximum(mz, tl.load(src + TC))
        zs0 = tl.where(mz == float("-inf"), 0.0, mz)
        az = tl.zeros([TC], dtype = tl.float32)
        for j in range(SHARES):
            src = part + (((n * tl.num_programs(1) + ct) * SHARES + j) * 3) * TC + tl.arange(0, TC)
            az += tl.load(src + 2 * TC) * tl.exp(tl.load(src + TC) - zs0)
    if sh_ == 0:
        tl.store(shift + n * C + cs, big, mask = cm)
        tl.store(zc + n * Cs + xs, tl.log(az) + tl.where(mz == float("-inf"), 0.0, mz), mask = cm)
    s = tl.where(big == float("-inf"), 0.0, big)
    for k0 in range(k_lo, k_hi, TK):
        ks = k0 + tl.arange(0, TK)
        km = ks < k_hi
        if PER_COLUMN:
            r = _column_rows(nodes + t * K, mrow, mstride, b, ks, km, STRIDE)
            lz = tl.load(lognorm + r, mask = km, other = 0.0)
        else:
            lz = tl.zeros([TK], dtype = tl.float32)
        mk = lv[:, None] & km[None, :]
        ptr = base[:, None] + ks[None, :]
        v = tl.load(ptr, mask = mk, other = float("-inf")) - lz[None, :]
        tl.store(ptr, tl.exp(v - s[:, None]), mask = mk)


#: :func:`_score_weights_kernel`: the fewest rows of a share (the rows are split only when the columns x class tiles
#: are fewer than twice the SMs)
SCORE_SHARE_ROWS = 256


@triton.jit
def _plain_dot(trow, wrow, ok, K, BH: tl.constexpr, EVEN_K: tl.constexpr):
    """``sum_k T[v, k] a[k]`` of ``BV`` lanes, over the lanes' rows of the table (``trow`` [BV, 1] pointers) and
    their exponentiated weights (``wrow``), in fp32; ``ok`` the lanes on; ``EVEN_K``: ``BH`` divides ``K``."""
    acc = tl.zeros([trow.shape[0]], dtype = tl.float32)
    for k0 in range(0, K, BH):
        ks = k0 + tl.arange(0, BH)
        if EVEN_K:
            mk = ok[:, None]
        else:
            mk = ok[:, None] & (ks < K)[None, :]
        tv = tl.load(trow + ks[None, :], mask = mk, other = 0.0).to(tl.float32)                     # [BV, BH]
        a = tl.load(wrow + ks[None, :], mask = mk, other = 0.0)                                     # [BV, BH]
        acc += tl.sum(tv * a, axis = 1)
    return acc


@triton.jit
def _log_dot(trow, wrow, nodes_t, mrow, mstride, lognorm, b, ok, enum, K, BH: tl.constexpr, EVEN_K: tl.constexpr,
             ENUM: tl.constexpr, PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr):
    """``(dot, shift)`` of ``BV`` lanes: ``sum_k T[v, k] a[k]`` over the lanes' rows of the table (``trow`` [BV, 1]
    pointers) and weights (``wrow``) -- the weights as :func:`_score_weights_kernel` left them, exponentiated, for a
    stored class's lane (:func:`_plain_dot`); ``enum`` (``ENUM``: some class is enumerated), an enumerated class's:
    in log, so the lane first takes their maximum (less the rows' ``lognorm`` under per-column masses,
    ``PER_COLUMN``; the rows of the column's position's input nodes ``nodes_t``, see :func:`_column_rows`) as its
    shift, then exponentiates them under it as the dot goes -- a program without such a lane takes the plain dot.
    The dot in fp32; ``ok`` the lanes on; ``EVEN_K``: ``BH`` divides ``K``."""
    m = tl.full([trow.shape[0]], float("-inf"), dtype = tl.float32)
    if ENUM:
        if tl.max(tl.where(enum, 1, 0), axis = 0) > 0:
            for k0 in range(0, K, BH):
                ks = k0 + tl.arange(0, BH)
                km = ks < K
                a = tl.load(wrow + ks[None, :], mask = enum[:, None] & km[None, :], other = float("-inf"))
                if PER_COLUMN:
                    a = a - tl.load(lognorm + _column_rows(nodes_t, mrow, mstride, b, ks, km, STRIDE), mask = km,
                                    other = 0.0)[None, :]
                mr = tl.max(a, axis = 1)
                m = tl.where(mr > m, mr, m)                      # (a select: see _score_weights_kernel)
            s = tl.where(m == float("-inf"), 0.0, m)
            acc = tl.zeros([trow.shape[0]], dtype = tl.float32)
            for k0 in range(0, K, BH):
                ks = k0 + tl.arange(0, BH)
                if EVEN_K:
                    mk = ok[:, None]
                else:
                    mk = ok[:, None] & (ks < K)[None, :]
                tv = tl.load(trow + ks[None, :], mask = mk, other = 0.0).to(tl.float32)             # [BV, BH]
                a = tl.load(wrow + ks[None, :], mask = mk, other = 0.0)                             # [BV, BH]
                if PER_COLUMN:
                    lz = tl.load(lognorm + _column_rows(nodes_t, mrow, mstride, b, ks, ks < K, STRIDE),
                                 mask = ks < K, other = 0.0)
                    e = tl.exp(a - lz[None, :] - s[:, None])
                else:
                    e = tl.exp(a - s[:, None])
                acc += tl.sum(tv * tl.where(enum[:, None] & mk, e, a), axis = 1)   # (a row past K adds its 0:
                                                       # exp(0 - s) is inf for a lane's shift below -88, 0 x inf NaN)
        else:
            acc = _plain_dot(trow, wrow, ok, K, BH, EVEN_K)
    else:
        acc = _plain_dot(trow, wrow, ok, K, BH, EVEN_K)
    return acc, m


@triton_jit
def _score_tokens_kernel(table, w, row, enum_tokens, token_class, code, tok_ptr, enum_logw, positions, nodes, mrow,
                         mstride, lognorm, out_row, out, uz, N, K, V, C, B, E, logw_stride, BV: tl.constexpr,
                         BH: tl.constexpr, EVEN_K: tl.constexpr, LOGW: tl.constexpr, WRITE: tl.constexpr,
                         PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr):
    """One program per (column ``n``, tile of ``BV`` of the ``E`` enumerated classes' tokens): ``u_v = log sum_k T[v, k]
    exp(w[row[n, c(v)], k])`` (:func:`_log_dot`) -- ``LOGW``: plus the token's log-weight for the column's sample,
    ``enum_logw[b, tok_ptr[t] + e]``, ``e`` its number among the live enumerated tokens of the column's position ``t``
    (from its class's code); a token weighing nothing, or of a class dead at ``t`` (or for the column, its row -1),
    -inf, its dot skipped -- the
    token's UNNORMALIZED log score, into ``out[out_row[n], v]`` (``WRITE``; ``out`` [M, V]), and ``uz[n, tile] = log
    sum_v exp(u_v)`` over the tile: the tokens' part of ``p(e, C)`` (see :func:`_logz_kernel`). Columns of no row
    (``out_row[n]`` -1): -inf, nothing written."""
    n = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    es = tile * BV + tl.arange(0, BV)
    em = es < E
    r = tl.load(out_row + n)
    t = tl.load(positions + n // B)
    b = n % B
    tk = tl.load(enum_tokens + es, mask = em, other = 0).to(tl.int64)
    c = tl.load(token_class + tk, mask = em, other = 0).to(tl.int64)
    x = tl.load(code + t * C + c, mask = em, other = 0)
    t0 = tl.load(tok_ptr + t)
    e = tl.maximum(-1 - x, 0)
    live = em & (r >= 0) & (x < 0) & (e < tl.load(tok_ptr + t + 1) - t0)
    rw = tl.load(row + n * C + c, mask = live, other = -1)                    # its class's row (-1: dead, -inf)
    live = live & (rw >= 0)
    if LOGW:
        lw = tl.load(enum_logw + b * logw_stride + t0 + e, mask = live, other = float("-inf"))
    else:
        lw = tl.where(live, 0.0, float("-inf"))
    ok = lw > float("-inf")
    KD = tl.where(tl.max(ok.to(tl.int32), axis = 0) > 0, K, 0)              # (no lane on: no dot, every lane -inf)
    acc, m = _log_dot(table + tk[:, None] * K, w + tl.maximum(rw, 0).to(tl.int64)[:, None] * K, nodes + t * K, mrow,
                      mstride, lognorm, b, ok, ok, KD, BH, EVEN_K, True, PER_COLUMN, STRIDE)
    u = tl.where(ok, tl.log(acc) + m + lw, float("-inf"))
    top = tl.max(u, axis = 0)
    s = tl.where(top == float("-inf"), 0.0, top)
    tl.store(uz + n * tl.num_programs(1) + tile, tl.log(tl.sum(tl.exp(u - s), axis = 0)) + s)
    if WRITE:
        tl.store(out + tl.maximum(r, 0).to(tl.int64) * V + tk, u, mask = em & (r >= 0))


@triton_jit
def _logz_kernel(zc, uz, logz, Cs, UT, TC: tl.constexpr):
    """One program per column ``n``: ``logz[n] = log (sum_x exp(zc[n, x]) + sum_j exp(uz[n, j]))``, ``log p(e, C)`` --
    the ``Cs`` stored classes' parts and the enumerated tokens' (``UT`` tiles of them, see :func:`_score_tokens_kernel`)."""
    n = tl.program_id(0).to(tl.int64)
    m = tl.full([TC], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TC], dtype = tl.float32)
    for c0 in range(0, Cs, TC):
        cs = c0 + tl.arange(0, TC)
        m, acc = _lse_step(m, acc, tl.load(zc + n * Cs + cs, mask = cs < Cs, other = float("-inf")))
    for j0 in range(0, UT, TC):
        js = j0 + tl.arange(0, TC)
        m, acc = _lse_step(m, acc, tl.load(uz + n * UT + js, mask = js < UT, other = float("-inf")))
    top = tl.max(m, axis = 0)
    s = tl.where(top == float("-inf"), 0.0, top)
    tl.store(logz + n, tl.log(tl.sum(acc * tl.exp(m - s), axis = 0)) + s)


@triton_jit
def _normalize_kernel(out, out_row, logz, idx, off, L, V, KC, BV: tl.constexpr, MODE: tl.constexpr):
    """One program per (column ``n``, tile of ``BV`` of its row's ``L`` scored entries): less the column's ``logz``
    (-inf where ``p(e, C)`` is 0). The entries: ``MODE`` 0, ``out[r, j]`` (``out`` [M, KC], the row's candidates); 1,
    ``out[r, idx[off[n] + j]]`` (``out`` [M, V], the run of tokens its weights list, none for a column whose offset is
    -1); 2, ``out[r, idx[j]]`` (the enumerated classes' tokens)."""
    n = tl.program_id(0).to(tl.int64)
    js = tl.program_id(1) * BV + tl.arange(0, BV)
    r = tl.load(out_row + n)
    jm = (js < L) & (r >= 0)
    rr = tl.maximum(r, 0).to(tl.int64)
    if MODE == 0:
        ptr = out + rr * KC + js
    elif MODE == 1:
        wo = tl.load(off + n)
        jm = jm & (wo >= 0)
        ptr = out + rr * V + tl.load(idx + tl.maximum(wo, 0) + js, mask = jm, other = 0).to(tl.int64)
    else:
        ptr = out + rr * V + tl.load(idx + js, mask = jm, other = 0).to(tl.int64)
    lz = tl.load(logz + n)
    val = tl.load(ptr, mask = jm, other = float("-inf"))
    tl.store(ptr, tl.where(lz == float("-inf"), float("-inf"), val - lz), mask = jm)


@triton_jit
def _score_kernel(table, w, row, shift, logz, toks, tile_cls, out_row, out, logw, logw_off, N, K, V, C, KW,
                  BB: tl.constexpr, BV: tl.constexpr, BH: tl.constexpr, EVEN_K: tl.constexpr,
                  PRECISION: tl.constexpr, WEIGHTS: tl.constexpr):
    """One program per (``BB`` columns, tile of ``BV`` tokens of one stored class): ``log sum_k T[v, k] w[row[n, c],
    k] + shift[n, c] - logz[n]`` (a dead row, -1: -inf) from the token-major table ``T`` [V, K] (fp32 or bf16) and
    the shifted, exponentiated weights (:func:`_score_weights_kernel`), into row ``out_row[n]`` of ``out`` [M, V] (-1:
    not requested); -inf where ``p(e, C)`` is 0. ``EVEN_K``: ``BH`` divides ``K``. ``WEIGHTS``: plus the column's
    tokens' log-weights, ``logw[logw_off[n] + v]`` for ``v < KW`` (-inf beyond; none for a column whose offset is
    -1)."""
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    tk = tl.load(toks + pid_v * BV + tl.arange(0, BV))                        # -1: padding
    ok = tk >= 0
    c = tl.load(tile_cls + pid_v)
    ns = pid_n * BB + tl.arange(0, BB)
    nm = ns < N
    trow = table + tk.to(tl.int64)[None, :] * K
    rw = tl.load(row + ns.to(tl.int64) * C + c, mask = nm, other = -1)       # the columns' rows of the class
    nl = nm & (rw >= 0)                                                       # (-1: dead, weights -inf)
    wrow = w + tl.maximum(rw, 0).to(tl.int64)[:, None] * K
    acc = tl.zeros([BB, BV], dtype = tl.float32)
    KD = tl.where(tl.max(nl.to(tl.int32), axis = 0) > 0, K, 0)             # (the class dead for every column: no dot)
    for k0 in range(0, KD, BH):
        ks = k0 + tl.arange(0, BH)
        if EVEN_K:
            bt = tl.load(trow + ks[:, None], mask = ok[None, :], other = 0.0)                           # [BH, BV]
            a = tl.load(wrow + ks[None, :], mask = nl[:, None], other = 0.0)                           # [BB, BH]
        else:
            km = ks < K
            bt = tl.load(trow + ks[:, None], mask = km[:, None] & ok[None, :], other = 0.0)
            a = tl.load(wrow + ks[None, :], mask = nl[:, None] & km[None, :], other = 0.0)
        acc += tl.dot(_tf32_operand(a, PRECISION), _tf32_operand(bt, PRECISION).to(tl.float32),   # (bf16: exact)
                      input_precision = PRECISION)
    sc = tl.load(shift + ns.to(tl.int64) * C + c, mask = nm, other = float("-inf"))
    lz = tl.load(logz + ns, mask = nm, other = float("-inf"))
    val = tl.where((lz == float("-inf"))[:, None], float("-inf"), tl.log(acc) + (sc - lz)[:, None])
    if WEIGHTS:
        wo = tl.load(logw_off + ns, mask = nm, other = -1)
        lw = tl.load(logw + tl.maximum(wo, 0)[:, None] + tk[None, :],
                     mask = (wo >= 0)[:, None] & (ok & (tk < KW))[None, :], other = float("-inf"))
        val = tl.where((wo >= 0)[:, None], val + lw, val)
    r = tl.load(out_row + ns, mask = nm, other = -1)
    tl.store(out + r.to(tl.int64)[:, None] * V + tk[None, :], val, mask = (r >= 0)[:, None] & ok[None, :])


@triton_jit
def _score_candidates_kernel(table, w, row, shift, token_class, code, positions, nodes, mrow, mstride, lognorm, cand,
                             out_row, out, uz, logw, logw_off, ids, N, K, V, C, B, KC, KW, BV: tl.constexpr,
                             BH: tl.constexpr, EVEN_K: tl.constexpr, WEIGHTS: tl.constexpr, STEPS: tl.constexpr,
                             UZ: tl.constexpr, ENUM: tl.constexpr, PER_COLUMN: tl.constexpr, STRIDE: tl.constexpr):
    """One program per (column ``n``, tile of ``BV`` of its row's candidates): ``log sum_k T[v, k] exp(w[row[n,
    c(v)], k])`` (a dead row, -1: -inf), the UNNORMALIZED log score (less ``logz`` afterwards,
    :func:`_normalize_kernel`), for every candidate ``v = cand[r, j]`` of row ``r = out_row[n]`` (-1: not requested),
    into ``out[r, j]`` (``out`` [M, KC]); -inf where ``v`` is outside the vocabulary (padding). The weights as
    :func:`_score_weights_kernel` left them -- ``ENUM``: an enumerated class's (by its code at the column's position,
    ``code[t, c]`` < 0) in log, see :func:`_log_dot` --, the dot in fp32 over the candidates' rows of the table
    alone. ``EVEN_K``: ``BH`` divides ``K``. ``WEIGHTS``:
    plus the column's tokens' log-weights (none for a column whose offset ``logw_off[n]`` is -1) -- 1: as
    :func:`_score_kernel`; 2: of the tokens its ascending run of ``KW`` in ``ids`` lists, ``logw`` aligned with it
    (found in ``STEPS`` halvings; the others -inf); 3: the candidates are that run's tokens themselves, each scored
    into ``out[r, v]`` (``out`` [M, V]; the rest of the row is not written). ``UZ``: ``uz[n, tile] = log sum exp`` of
    the tile's enumerated candidates' scores: the tokens' part of ``p(e, C)`` (see :func:`_logz_kernel`) -- when the
    candidates are every token with a weight (3)."""
    n = tl.program_id(0).to(tl.int64)
    js = tl.program_id(1) * BV + tl.arange(0, BV)
    r = tl.load(out_row + n)
    rr = tl.where(r >= 0, r, 0).to(tl.int64)
    t = tl.load(positions + n // B)
    b = n % B
    if WEIGHTS != 0:
        wo = tl.load(logw_off + n)
        wb = tl.maximum(wo, 0)
    if WEIGHTS == 3:
        jm = (js < KW) & (r >= 0) & (wo >= 0)
        tk = tl.load(ids + wb + js, mask = jm, other = -1)
    else:
        jm = (js < KC) & (r >= 0)
        tk = tl.load(cand + rr * KC + js, mask = jm, other = -1)
    ok = jm & (tk >= 0) & (tk < V)
    tks = tl.where(ok, tk, 0).to(tl.int64)
    c = tl.load(token_class + tks, mask = ok, other = 0).to(tl.int64)
    rw = tl.load(row + n * C + c, mask = ok, other = -1)                      # its class's row (-1: dead, -inf)
    on = ok & (rw >= 0)
    if ENUM:
        enum = on & (tl.load(code + t * C + c, mask = on, other = 0) < 0)
    else:
        enum = on & (c < 0)
    KD = tl.where(tl.max(on.to(tl.int32), axis = 0) > 0, K, 0)              # (no lane on: no dot, every lane -inf)
    acc, m = _log_dot(table + tks[:, None] * K, w + tl.maximum(rw, 0).to(tl.int64)[:, None] * K, nodes + t * K, mrow,
                      mstride, lognorm, b, on, enum, KD, BH, EVEN_K, ENUM, PER_COLUMN, STRIDE)
    sc = tl.where(enum, m, tl.load(shift + n * C + c, mask = on & ~enum, other = float("-inf")))
    val = tl.where(on, tl.log(acc) + sc, float("-inf"))
    if WEIGHTS == 1:
        lw = tl.load(logw + wb + tks, mask = ok & (wo >= 0) & (tks < KW), other = float("-inf"))
        val = tl.where(wo >= 0, val + lw, val)
    elif WEIGHTS == 2:
        k = find_token(ids, wb + tl.zeros_like(tks), KW, tks, ok & (wo >= 0), STEPS)
        lw = tl.load(logw + wb + k, mask = k >= 0, other = float("-inf"))
        val = tl.where(wo >= 0, val + lw, val)
    elif WEIGHTS == 3:
        val = val + tl.load(logw + wb + js, mask = ok, other = float("-inf"))
    if UZ:
        u = tl.where(enum, val, float("-inf"))
        top = tl.max(u, axis = 0)
        s = tl.where(top == float("-inf"), 0.0, top)
        tl.store(uz + n * tl.num_programs(1) + tl.program_id(1), tl.log(tl.sum(tl.exp(u - s), axis = 0)) + s)
    if WEIGHTS == 3:
        tl.store(out + rr * V + tks, val, mask = ok)
    else:
        tl.store(out + rr * KC + js, val, mask = jm)


#: :func:`_score_candidates_kernel`'s and :func:`_score_tokens_kernel`'s ``(BV, BH, warps)`` -- tokens per program,
#: the dot's chunk of rows -- for at most :data:`SCORE_CANDIDATE_PAIRS` (column, token) pairs, and for more: from a
#: sweep of the cached decode step on Ctrl-G's HMM4096 (4096 rows) at 1/16/128 columns x 50/1024 candidates, the first
#: within 2% of the best everywhere but 128 x 1024, where the second takes the step from 3.32 to 2.87 ms (scoring every
#: token: 2.92)
SCORE_CANDIDATE_TILES = ((16, 256, 4), (64, 64, 2))
SCORE_CANDIDATE_PAIRS = 1 << 16


#: :func:`_score_kernel`'s tiles ``(BB, BV, BH, warps, stages)`` by table dtype and dot precision, for at most 16, 64 and
#: more columns: from a sweep of the kernel on Ctrl-G's HMM4096 (V = 50257, 4096 rows) and first example's 10 classes at
#: 1/16, 64 and 128 columns, ``BH`` (the dots' reduction chunk) fixed per dtype. (16, 64, 32, 4, 2) -- the fastest
#: before the epilogue read the shifts -- lost 16% with it, to code generation: no other tile changed.
SCORE_TILES = {("fp32", "tf32x3"): ((16, 128, 32, 4, 2), (32, 64, 32, 8, 3), (32, 128, 32, 4, 3)),
               ("fp32", "tf32"): ((16, 64, 32, 4, 2), (64, 64, 32, 4, 3), (64, 64, 32, 4, 3)),
               ("bf16", "tf32"): ((16, 64, 64, 4, 2), (32, 64, 64, 4, 2), (32, 128, 64, 8, 3))}


def score_group(out: torch.Tensor, table: torch.Tensor, w: torch.Tensor, row: torch.Tensor, rows: torch.Tensor,
                class_mars: torch.Tensor, out_row: torch.Tensor, classes, precision: str, positions: torch.Tensor,
                enum: dict, candidates: Optional[torch.Tensor] = None, sample: Optional[dict] = None) -> torch.Tensor:
    """
    Score one row group: ``out[out_row[n], v] = log p(x_t = v | e, C)`` for every column ``n`` (a target position's
    sample) that ``out_row`` asks for, from its log class weights -- ``w`` [live rows, K] (the stored classes'
    overwritten), column ``n``'s class ``c``'s at row ``row[n, c]`` (int32 [N, C]; -1: dead, its weights -inf, see
    :class:`ClassWeightRows`) --, the group's token-major table [V, K] and its rows of the class masses (``rows``,
    int64 [K] on the device).

    ``p(e, C) = sum_c sum_k A[k, c] mass_k(c)``: the stored classes' parts from the class masses
    (:func:`_score_weights_kernel`), an enumerated class's from its token's unnormalized score -- the dot the scoring
    computes anyway (:func:`_score_tokens_kernel`; under top-k weights, of the tokens with a weight alone, scored by
    :func:`_score_candidates_kernel`) -- so no second pass over the tokens, and no mass per (row, enumerated class).
    An enumerated class's weights are read once, by its token's dot, in log (:func:`_log_dot`): the shift-and-exp
    pass is over the stored classes alone.

    :param classes: the constraint's :class:`~pyjuice.constraints.distributions.TokenClasses`
    :param precision: the dot's: "tf32x3" (fp32-level) or "tf32"
    :param positions: int64 [N / columns of a position] on the device: the columns' positions (the samples of a
        position its consecutive columns)
    :param enum: how the class masses are read by class code (:func:`~.transitions.enumerated_args`)
    :param candidates: None (every token), or int32 [M, k] on the device: row ``r`` scores ``candidates[r]`` alone,
        into ``out[r]`` ([M, k]), by :func:`_score_candidates_kernel` (in fp32 whatever ``precision``)
    :param sample: None, or the columns' own class masses under per-sample inputs (see
        :class:`~..state.ClassMasses`): a dict with ``nodes`` (int64 [positions of the PC, K] on the device: every
        position's input nodes), ``row`` and ``stride`` (int32 per node; ``stride`` None
        for the same rows for every sample: node ``u`` of sample ``b`` reads row ``row[u] + b stride[u]`` of
        ``class_mars``), ``lognorm`` (every row's ``log Z``) and the tokens' weights, ``logw`` (flat), ``offset`` (int64
        [N], every column's, -1 for none), ``width`` (the tokens with a weight) and ``ids`` (None: the first ``width``
        tokens; else int32, flat as ``logw``, the tokens of its ascending runs -- every column then has an offset, and
        without ``candidates`` only those tokens are written: the rest of ``out``'s rows must be -inf already)
    :returns: ``log p(e, C)`` of every column, [N]
    """
    (N, C), K = row.shape, w.size(1)
    V = table.size(0)
    B = N // positions.numel()
    dev = w.device
    Cs, E = classes.num_stored, classes.num_enumerated                        # (one token each)
    shift = torch.empty(N, C, dtype = torch.float32, device = dev)
    zc = torch.empty(N, max(Cs, 1), dtype = torch.float32, device = dev)      # the stored classes' parts of p(e, C)
    logz = torch.empty(N, dtype = torch.float32, device = dev)
    per_column = sample is not None
    ids = sample["ids"] if per_column else None
    weights = dict(logw = sample["logw"] if per_column else w, logw_off = sample["offset"] if per_column else rows,
                   KW = sample["width"] if per_column else 0)
    stride = per_column and sample["stride"] is not None
    columns = dict(nodes = sample["nodes"] if per_column else rows, mrow = sample["row"] if per_column else rows,
                   mstride = sample["stride"] if stride else rows, lognorm = sample["lognorm"] if per_column else w)
    modes = dict(PER_COLUMN = per_column, STRIDE = stride)

    # the stored classes: the shifts, the weights exponentiated, their parts of p(e, C)
    if Cs > 0:
        TC, TK = min(64, max(16, host_next_power_of_2(Cs))), 64
        ctiles = host_cdiv(Cs, TC)
        shares = max(1, min(host_cdiv(2 * _num_sms(dev), N * ctiles), host_cdiv(K, SCORE_SHARE_ROWS)))
        KS = host_cdiv(host_cdiv(K, shares), TK) * TK
        shares = host_cdiv(K, KS)
        part = torch.empty(N * ctiles * shares * 3 * TC if shares > 1 else 1, dtype = torch.float32, device = dev)
        args = (w, row, class_mars, rows, columns["nodes"], positions, columns["mrow"], columns["mstride"],
                columns["lognorm"], shift, zc, part, classes.class_of_stored32, C, K, B, KS, Cs)
        consts = dict(TC = TC, TK = TK, SHARES = shares, num_warps = 4, **modes)
        _score_weights_kernel[(N, ctiles, shares)](*args, FINISH = False, **consts)
        if shares > 1:
            _score_weights_kernel[(N, ctiles, shares)](*args, FINISH = True, **consts)

    # the scores: the candidates', or the enumerated tokens' (unnormalized), with the enumerated tokens' parts of p(e, C)
    run = ids is not None and candidates is None                           # the weights' run of tokens, every one
    by_tokens = E > 0 and not run                                           # with a weight: their parts come with
    KC = (candidates.size(1) if candidates is not None else weights["KW"]) if candidates is not None or ids is not None \
        else 0
    tokens_tile = SCORE_CANDIDATE_TILES[N * E > SCORE_CANDIDATE_PAIRS]
    cand_tile = SCORE_CANDIDATE_TILES[N * KC > SCORE_CANDIDATE_PAIRS]
    UT = host_cdiv(E, tokens_tile[0]) if by_tokens else host_cdiv(KC, cand_tile[0]) if run and E > 0 else 0
    uz = torch.empty(N, max(UT, 1), dtype = torch.float32, device = dev)
    if candidates is not None or ids is not None:                           # top-k weights: their tokens alone
        mode = (2 if ids is not None else int(per_column)) if candidates is not None else 3
        BV, BH, warps = cand_tile
        _score_candidates_kernel[(N, host_cdiv(KC, BV))](
            table, w, row, shift, classes.token_class, enum["code"], positions, columns["nodes"], columns["mrow"],
            columns["mstride"], columns["lognorm"], candidates if candidates is not None else rows, out_row, out, uz,
            ids = ids if ids is not None else rows, N = N, K = K, V = V, C = C, B = B, KC = KC, BV = BV, BH = BH,
            EVEN_K = K % BH == 0, WEIGHTS = mode, STEPS = max(1, int(weights["KW"]).bit_length()),
            UZ = run and E > 0, ENUM = E > 0, num_warps = warps, **weights, **modes)
    if by_tokens:
        BV, BH, warps = tokens_tile
        _score_tokens_kernel[(N, UT)](table, w, row, classes.enumerated_tokens, classes.token_class, enum["code"],
                                      enum["tok_ptr"], enum["enum_logw"], positions, columns["nodes"], columns["mrow"],
                                      columns["mstride"], columns["lognorm"], out_row, out, uz, N = N, K = K, V = V,
                                      C = C, B = B, E = E, logw_stride = enum["logw_stride"], BV = BV, BH = BH,
                                      EVEN_K = K % BH == 0, LOGW = enum["LOGW"] and per_column,
                                      WRITE = candidates is None and ids is None, num_warps = warps, **modes)
    _logz_kernel[(N,)](zc, uz, logz, Cs, UT, TC = min(1024, max(16, host_next_power_of_2(max(Cs, UT, 1)))),
                       num_warps = 1)
    if candidates is not None or ids is not None:
        BV = cand_tile[0]
        _normalize_kernel[(N, host_cdiv(KC, BV))](out, out_row, logz, ids if run else rows, weights["logw_off"], KC,
                                                  V, KC, BV = BV, MODE = 1 if run else 0, num_warps = 1)
        return logz
    if E > 0:                                                               # (the enumerated tokens: written already)
        BV = tokens_tile[0]
        _normalize_kernel[(N, UT)](out, out_row, logz, classes.enumerated_tokens, rows, E, V, V, BV = BV, MODE = 2,
                                   num_warps = 1)
    if Cs > 0:                                                              # the stored classes' tokens, in tiles
        dtype = "bf16" if table.dtype == torch.bfloat16 else "fp32"
        prec = "tf32" if dtype == "bf16" else precision
        BB, BV, BH, warps, stages = SCORE_TILES[(dtype, prec)][0 if N <= 16 else 1 if N <= 64 else 2]
        BV = min(BV, max(16, host_next_power_of_2(-(-V // Cs))))            # many small classes: narrower tiles
        toks, tile_cls = classes.class_tiles(BV, stored = True)
        _score_kernel[(host_cdiv(N, BB), tile_cls.numel())](table, w, row, shift, logz, toks, tile_cls, out_row, out,
                                                            N = N, K = K, V = V, C = C, BB = BB, BV = BV, BH = BH,
                                                            EVEN_K = K % BH == 0, PRECISION = prec,
                                                            WEIGHTS = per_column, num_warps = warps,
                                                            num_stages = stages, **weights)
    return logz
