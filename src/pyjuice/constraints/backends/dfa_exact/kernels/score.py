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
mass_k(c)`` from the class masses (no second pass over the tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from .common import (IDENTITY, _block_ptr, _child, _exit_width, _interval, _logaddexp_tile, _lse_step, _slot,
                     grid_chunks, tensor)
from .prod_top_down import _once, _out_flows_ptr



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
    """

    group_of: np.ndarray
    rows: list
    rows_t: list
    col_of: np.ndarray


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
    groups, rows = {}, []
    for pos in range(prog.n):
        at = u[t == pos]
        if at.size == 0:
            continue
        r = np.unique(mass_row[at])
        g = groups.setdefault(r.tobytes(), len(rows))
        if g == len(rows):
            rows.append(r)
        group_of[pos] = g
        col_of[at] = np.searchsorted(r, mass_row[at])
    prog.score = ScoreTables(group_of = group_of, rows = rows,
                             rows_t = [torch.from_numpy(r).to(prog.dev) for r in rows], col_of = col_of)
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
    :ivar host: for query plans, numpy copies of ``dests``, ``ptr`` and ``parents``
    """

    form: str
    kinds: tuple
    dests: torch.Tensor
    ptr: torch.Tensor
    parents: torch.Tensor
    J_max: int
    host: Any


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
    i32 = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(prog.dev, torch.int32)
    return ClassWeightLaunch(launch.form, launch.kinds, i32(dests), i32(ptr), i32(rows), int(J.max()),
                             dict(dests = dests, ptr = ptr, parents = rows))


@triton_jit
def _class_weight_kernel(dfa_class_flow, cw_base, cw_K, element_flows, temp_flows, node_mars, prior_mars, temp,
                         next_col, width, iv_of, iv_info, iv_table, once_iv, live, dests, dests_idx, ptr, parents,
                         reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, out_first, out_width,
                         temp_width, B, n, W, C, pid0, BLOCK_INPUT: tl.constexpr, S_KIND: tl.constexpr,
                         OUT_TEMP: tl.constexpr, PRUNE: tl.constexpr, OUT_FULL: tl.constexpr, S_FULL: tl.constexpr,
                         INDEXED: tl.constexpr, TB: tl.constexpr, TC: tl.constexpr, TJ: tl.constexpr):
    """One program per (destination, ``TB`` samples, ``TC`` classes), a lane per (sample, class): the log class
    weights of its steps' inputs (see the module docstring) -- a loop over the columns ``i`` of the input's boundary,
    then over the other boundary's columns ``TJ`` at a time -- log-added to ``dfa_class_flow`` at the destination's
    position block (``cw_base``, ``cw_K`` per position). Under evidence pruning, a column ``i`` dead for every lane's
    sample is skipped (every term through it is -inf)."""
    d = _child(dests_idx, tl.program_id(0), INDEXED)
    lane = tl.arange(0, TB * TC)
    b = ((pid0 + tl.program_id(1)) * TB + lane // TC).to(tl.int64)
    cls = tl.program_id(2) * TC + lane % TC
    cm = (b < B) & (cls < C)
    t = tl.load(dests + d * 2)
    k = tl.load(dests + d * 2 + 1)
    end = t + 1 == n
    lrow = live + (b * (n + 1) + t) * W                                         # the lanes' liveness at t (pruned)
    mx = tl.full([TB * TC, TJ], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TB * TC, TJ], dtype = tl.float32)
    for p in range(tl.load(ptr + d), tl.load(ptr + d + 1)):
        out_row = tl.load(parents + p * 7 + 0)
        if BLOCK_INPUT:                                                        # a block over [t0, t), then the input
            t0 = tl.load(parents + p * 7 + 3)
            E = tl.load(width + t0)
            M = tl.load(width + t)
            X = _exit_width(width, t + 1, n)
            OS, opos, _ = _interval(iv_of, iv_info, t0, t + 1, n)
            LS, lpos, _ = _interval(iv_of, iv_info, t0, t, n)
            op = _out_flows_ptr(out_row, element_flows, temp_flows, out_first, out_width, temp_width, OUT_TEMP) + \
                b * tl.where(_once(once_iv, iv_of, t0, t + 1, n), 0, OS)
            lp, l_own = _block_ptr(tl.load(parents + p * 7 + 1), tl.load(parents + p * 7 + 2), node_mars, prior_mars,
                                   temp, reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior,
                                   temp_width, S_KIND)
            lp = lp + tl.where(l_own, b * LS, 0)
            for i in range(0, M):
                ok = cm
                if PRUNE:
                    ok = ok & (tl.load(lrow + i, mask = cm, other = 0) != 0)
                if tl.max(ok.to(tl.int32), axis = 0) > 0:
                    q = tl.load(next_col + (t * W + i) * C + cls, mask = ok, other = -1)
                    q = tl.where(end & (q >= 0), 0, q)                         # a sequence end: its one column
                    for e0 in range(0, E, TJ):
                        i0 = e0 + tl.arange(0, TJ)
                        im = i0 < E
                        sl = _slot(iv_table, lpos, i0, i, M, im, S_FULL)
                        lv = tl.load(lp[:, None] + sl[None, :], mask = ok[:, None] & (im & (sl >= 0))[None, :],
                                     other = float("-inf"))                    # L[i0, i]
                        m2 = (ok & (q >= 0))[:, None] & im[None, :]
                        so = _slot(iv_table, opos, i0[None, :], q[:, None], X, m2, OUT_FULL)
                        ov = tl.load(op[:, None] + so, mask = m2 & (so >= 0), other = float("-inf"))   # O_out[i0, next]
                        mx, acc = _lse_step(mx, acc, lv + ov)
        else:                                                                  # the input, then a block over [t+1, t2)
            t2 = tl.load(parents + p * 7 + 5)
            X = _exit_width(width, t2, n)
            OS, opos, _ = _interval(iv_of, iv_info, t, t2, n)
            op = _out_flows_ptr(out_row, element_flows, temp_flows, out_first, out_width, temp_width, OUT_TEMP) + \
                b * tl.where(_once(once_iv, iv_of, t, t2, n), 0, OS)
            if S_KIND != 2:
                RS, rpos, _ = _interval(iv_of, iv_info, t + 1, t2, n)
                rp, r_own = _block_ptr(tl.load(parents + p * 7 + 3), tl.load(parents + p * 7 + 4), node_mars,
                                       prior_mars, temp, reg_offset, reg_width, reg_first, prior_offset, prior_width,
                                       reg_prior, temp_width, S_KIND)
                rp = rp + tl.where(r_own, b * RS, 0)
            for i in range(0, tl.load(width + t)):
                ok = cm
                if PRUNE:
                    ok = ok & (tl.load(lrow + i, mask = cm, other = 0) != 0)
                if tl.max(ok.to(tl.int32), axis = 0) > 0:
                    q = tl.load(next_col + (t * W + i) * C + cls, mask = ok, other = -1)
                    if S_KIND == 2:                                            # the identity: the exit column is q
                        jq = tl.where(end & (q >= 0), 0, q)
                        qm = ok & (q >= 0)
                        so = _slot(iv_table, opos, i, jq, X, qm, OUT_FULL)
                        ov = tl.load(op + so, mask = qm & (so >= 0), other = float("-inf"))
                        mx, acc = _lse_step(mx, acc, ov[:, None] + tl.zeros([TB * TC, TJ], dtype = tl.float32))
                    else:
                        for j0 in range(0, X, TJ):
                            j = j0 + tl.arange(0, TJ)
                            jm = j < X
                            so = _slot(iv_table, opos, i, j, X, jm, OUT_FULL)
                            ov = tl.load(op[:, None] + so[None, :], mask = ok[:, None] & (jm & (so >= 0))[None, :],
                                         other = float("-inf"))               # O_out[i, j]
                            m2 = (ok & (q >= 0))[:, None] & jm[None, :]
                            sr = _slot(iv_table, rpos, q[:, None], j[None, :], X, m2, S_FULL)
                            rv = tl.load(rp[:, None] + sr, mask = m2 & (sr >= 0), other = float("-inf"))   # R[next, j]
                            mx, acc = _lse_step(mx, acc, ov + rv)
    top = tl.max(mx, axis = 1)
    shift = tl.where(top == float("-inf"), 0.0, top)
    val = tl.log(tl.sum(acc * tl.exp(mx - shift[:, None]), axis = 1)) + shift
    dest = dfa_class_flow + tl.load(cw_base + t) + (b * C + cls) * tl.load(cw_K + t) + k
    tl.store(dest, _logaddexp_tile(tl.load(dest, mask = cm, other = float("-inf")), val), mask = cm)


#: lanes (sample, class) x other-boundary columns per :func:`_class_weight_kernel` program, at most
CLASS_WEIGHT_LANES = 2048


def run_class_weights(launches, cw: dict, prog, bufs, element_flows, temp_flows, node_mars, live, elem_first: int,
                      elem_width: int, B: int, prior: dict, once_iv: torch.Tensor, regions: dict):
    """
    The class weights of one product layer stage's input operands at the query's target positions: every
    :class:`ClassWeightLaunch` in ``launches``, each with its plan's index list of destinations (None: all).

    :param cw: ``buf`` (``dfa_class_flow``, -inf where nothing was added yet), and per position the start of its block
        (``base``, int64 [n], -1 for a position no sample targets) and its row group's rows (``K``, int64 [n])
    :param bufs: ``(temp, temp_width)``, the layer's scratch rows (the folds' inside values)
    :param live: None, or the query's liveness (:class:`~.live.Pruning`'s ``live``)
    :param prior: where the sum operands kept once are (see :func:`~.common._block_ptr`)
    :param once_iv: the plan's per-interval flag of top-down values kept once (read with sample stride 0)
    """
    temp, temp_width = bufs
    n, W, C = prog.n, prog.next_col.size(1), prog.next_col.size(2)
    ivs = prog.intervals
    prune = live is not None
    TC = min(64, host_next_power_of_2(C))
    for la, idx in launches:
        idx = tensor(idx)
        D = la.dests.size(0) if idx is None else idx.numel()
        s_kind, out_temp, (out_full, *s_full) = la.kinds
        identity = la.form == "input_block" and s_kind == IDENTITY
        TJ = 1 if identity else min(64, host_next_power_of_2(la.J_max))
        TB = max(1, min(host_next_power_of_2(B), 128 // TC, CLASS_WEIGHT_LANES // (TC * TJ)))
        lanes = TB * TC * TJ
        warps = 1 if lanes <= 128 else 2 if lanes <= 512 else 4
        for first, size in grid_chunks(host_cdiv(B, TB), 1):
            _class_weight_kernel[(D, size, host_cdiv(C, TC))](
                cw["buf"], cw["base"], cw["K"], element_flows, temp_flows, node_mars, prior["mars"], temp,
                prog.next_col, prog.width_t, ivs.of, ivs.info, ivs.table, once_iv, live if prune else cw["K"],
                la.dests, la.dests if idx is None else idx, la.ptr, la.parents, regions["offset"], regions["width"],
                regions["first"], prior["offset"], prior["width"], prior["reg_prior"], elem_first, elem_width,
                temp_width, B, n, W, C, first, BLOCK_INPUT = la.form == "block_input", S_KIND = s_kind,
                OUT_TEMP = out_temp, PRUNE = prune, OUT_FULL = out_full, S_FULL = all(s_full),
                INDEXED = idx is not None, TB = TB, TC = TC, TJ = TJ, num_warps = warps)


# -------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------

@triton_jit
def _score_weights_kernel(w, class_mars, rows, shift, zc, C, K, TK: tl.constexpr):
    """One program per (column ``n``, class ``c``) of a row group's log class weights ``w`` [N, C, K]: their maximum
    over the rows (``shift[n, c]``), the weights shifted by it and exponentiated in place, and ``zc[n, c] = log sum_k
    exp(w[n, c, k] + class_mars[rows[k], c])``, the class's part of ``p(e, C)``."""
    nc = tl.program_id(0).to(tl.int64)
    c = nc % C
    base = w + nc * K
    big = tl.full([TK], float("-inf"), dtype = tl.float32)
    mz = tl.full([TK], float("-inf"), dtype = tl.float32)
    az = tl.zeros([TK], dtype = tl.float32)
    for k0 in range(0, K, TK):
        ks = k0 + tl.arange(0, TK)
        km = ks < K
        v = tl.load(base + ks, mask = km, other = float("-inf"))
        big = tl.maximum(big, v)
        mass = tl.load(class_mars + tl.load(rows + ks, mask = km, other = 0).to(tl.int64) * C + c, mask = km,
                       other = float("-inf"))
        mz, az = _lse_step(mz, az, v + mass)
    top = tl.max(big, axis = 0)
    tl.store(shift + nc, top)
    zt = tl.max(mz, axis = 0)
    zs = tl.where(zt == float("-inf"), 0.0, zt)
    tl.store(zc + nc, tl.log(tl.sum(az * tl.exp(mz - zs), axis = 0)) + zs)
    s = tl.where(top == float("-inf"), 0.0, top)
    for k0 in range(0, K, TK):
        ks = k0 + tl.arange(0, TK)
        km = ks < K
        tl.store(base + ks, tl.exp(tl.load(base + ks, mask = km, other = float("-inf")) - s), mask = km)


@triton_jit
def _logz_kernel(zc, logz, C, TC: tl.constexpr):
    """One program per column ``n``: ``logz[n] = log sum_c exp(zc[n, c])``, ``log p(e, C)``."""
    n = tl.program_id(0).to(tl.int64)
    m = tl.full([TC], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TC], dtype = tl.float32)
    for c0 in range(0, C, TC):
        cs = c0 + tl.arange(0, TC)
        m, acc = _lse_step(m, acc, tl.load(zc + n * C + cs, mask = cs < C, other = float("-inf")))
    top = tl.max(m, axis = 0)
    s = tl.where(top == float("-inf"), 0.0, top)
    tl.store(logz + n, tl.log(tl.sum(acc * tl.exp(m - s), axis = 0)) + s)


@triton_jit
def _score_kernel(table, w, shift, logz, toks, tile_cls, out_row, out, N, K, V, C, BB: tl.constexpr,
                  BV: tl.constexpr, BH: tl.constexpr, EVEN_K: tl.constexpr, PRECISION: tl.constexpr):
    """One program per (``BB`` columns, tile of ``BV`` tokens of one class): ``log sum_k T[v, k] w[n, c, k] +
    shift[n, c] - logz[n]`` from the token-major table ``T`` [V, K] (fp32 or bf16) and the shifted, exponentiated
    weights (:func:`_score_weights_kernel`), into row ``out_row[n]`` of ``out`` [M, V] (-1: not requested); -inf where
    ``p(e, C)`` is 0. ``EVEN_K``: ``BH`` divides ``K``."""
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    tk = tl.load(toks + pid_v * BV + tl.arange(0, BV))                        # -1: padding
    ok = tk >= 0
    c = tl.load(tile_cls + pid_v)
    ns = pid_n * BB + tl.arange(0, BB)
    nm = ns < N
    trow = table + tk.to(tl.int64)[None, :] * K
    wrow = w + (ns.to(tl.int64)[:, None] * C + c) * K
    acc = tl.zeros([BB, BV], dtype = tl.float32)
    for k0 in range(0, K, BH):
        ks = k0 + tl.arange(0, BH)
        if EVEN_K:
            bt = tl.load(trow + ks[:, None], mask = ok[None, :], other = 0.0).to(tl.float32)            # [BH, BV]
            a = tl.load(wrow + ks[None, :], mask = nm[:, None], other = 0.0)                           # [BB, BH]
        else:
            km = ks < K
            bt = tl.load(trow + ks[:, None], mask = km[:, None] & ok[None, :], other = 0.0).to(tl.float32)
            a = tl.load(wrow + ks[None, :], mask = nm[:, None] & km[None, :], other = 0.0)
        acc += tl.dot(a, bt, input_precision = PRECISION)
    sc = tl.load(shift + ns.to(tl.int64) * C + c, mask = nm, other = float("-inf"))
    lz = tl.load(logz + ns, mask = nm, other = float("-inf"))
    val = tl.where((lz == float("-inf"))[:, None], float("-inf"), tl.log(acc) + (sc - lz)[:, None])
    r = tl.load(out_row + ns, mask = nm, other = -1)
    tl.store(out + r.to(tl.int64)[:, None] * V + tk[None, :], val, mask = (r >= 0)[:, None] & ok[None, :])


@triton_jit
def _score_candidates_kernel(table, w, shift, logz, token_class, cand, out_row, out, N, K, V, C, KC,
                             BV: tl.constexpr, BH: tl.constexpr, EVEN_K: tl.constexpr):
    """One program per (column ``n``, tile of ``BV`` of its row's candidates): ``log sum_k T[v, k] w[n, c(v), k] +
    shift[n, c(v)] - logz[n]`` for every candidate ``v = cand[r, j]`` of row ``r = out_row[n]`` (-1: not requested),
    into ``out[r, j]`` (``out`` [M, KC]); -inf where ``v`` is outside the vocabulary (padding) or ``p(e, C)`` is 0.
    The weights as :func:`_score_kernel` reads them; the dot in fp32, over the candidates' rows of the table alone.
    ``EVEN_K``: ``BH`` divides ``K``."""
    n = tl.program_id(0).to(tl.int64)
    js = tl.program_id(1) * BV + tl.arange(0, BV)
    r = tl.load(out_row + n)
    jm = (js < KC) & (r >= 0)
    rr = tl.where(r >= 0, r, 0).to(tl.int64)
    tk = tl.load(cand + rr * KC + js, mask = jm, other = -1)
    ok = jm & (tk >= 0) & (tk < V)
    tks = tl.where(ok, tk, 0).to(tl.int64)
    c = tl.load(token_class + tks, mask = ok, other = 0).to(tl.int64)
    trow = table + tks[:, None] * K                                           # [BV, 1]: every candidate's row
    wrow = w + (n * C + c)[:, None] * K                                       # [BV, 1]: its class's weights
    acc = tl.zeros([BV], dtype = tl.float32)
    for k0 in range(0, K, BH):
        ks = k0 + tl.arange(0, BH)
        if EVEN_K:
            m = ok[:, None]
        else:
            m = ok[:, None] & (ks < K)[None, :]
        t = tl.load(trow + ks[None, :], mask = m, other = 0.0).to(tl.float32)                       # [BV, BH]
        a = tl.load(wrow + ks[None, :], mask = m, other = 0.0)                                      # [BV, BH]
        acc += tl.sum(t * a, axis = 1)
    sc = tl.load(shift + n * C + c, mask = ok, other = float("-inf"))
    lz = tl.load(logz + n)
    val = tl.where(ok & (lz != float("-inf")), tl.log(acc) + sc - lz, float("-inf"))
    tl.store(out + rr * KC + js, val, mask = jm)


#: :func:`_score_candidates_kernel`'s ``(BV, BH, warps)`` -- candidates per program, the dot's chunk of rows -- for at
#: most :data:`SCORE_CANDIDATE_PAIRS` (column, candidate) pairs, and for more: from a sweep of the cached decode step on
#: Ctrl-G's HMM4096 (4096 rows) at 1/16/128 columns x 50/1024 candidates, the first within 2% of the best everywhere
#: but 128 x 1024, where the second takes the step from 3.32 to 2.87 ms (scoring every token: 2.92)
SCORE_CANDIDATE_TILES = ((16, 256, 4), (64, 64, 2))
SCORE_CANDIDATE_PAIRS = 1 << 16


#: :func:`_score_kernel`'s tiles ``(BB, BV, BH, warps, stages)`` by table dtype and dot precision, for at most 16, 64 and
#: more columns: from a sweep of the kernel on Ctrl-G's HMM4096 (V = 50257, 4096 rows) and first example's 10 classes at
#: 1/16, 64 and 128 columns, ``BH`` (the dots' reduction chunk) fixed per dtype. (16, 64, 32, 4, 2) -- the fastest
#: before the epilogue read the shifts -- lost 16% with it, to code generation: no other tile changed.
SCORE_TILES = {("fp32", "tf32x3"): ((16, 128, 32, 4, 2), (32, 64, 32, 8, 3), (32, 128, 32, 4, 3)),
               ("fp32", "tf32"): ((16, 64, 32, 4, 2), (64, 64, 32, 4, 3), (64, 64, 32, 4, 3)),
               ("bf16", "tf32"): ((16, 64, 64, 4, 2), (32, 64, 64, 4, 2), (32, 128, 64, 8, 3))}


def score_group(out: torch.Tensor, table: torch.Tensor, w: torch.Tensor, rows: torch.Tensor, class_mars: torch.Tensor,
                out_row: torch.Tensor, classes, precision: str, candidates: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
    """
    Score one row group: ``out[out_row[n], v] = log p(x_t = v | e, C)`` for every column ``n`` (a target position's
    sample) that ``out_row`` asks for, from its log class weights ``w`` [N, C, K] (overwritten), the group's
    token-major table [V, K] and its rows of the class masses (``rows``, int64 [K] on the device).

    :param classes: the constraint's :class:`~pyjuice.constraints.distributions.TokenClasses`
    :param precision: the dot's: "tf32x3" (fp32-level) or "tf32"
    :param candidates: None (every token), or int32 [M, k] on the device: row ``r`` scores ``candidates[r]`` alone,
        into ``out[r]`` ([M, k]), by :func:`_score_candidates_kernel` (in fp32 whatever ``precision``)
    :returns: ``log p(e, C)`` of every column, [N]
    """
    N, C, K = w.shape
    V = table.size(0)
    shift = torch.empty(2, N, C, dtype = torch.float32, device = w.device)    # the shifts, then the classes' parts
    logz = torch.empty(N, dtype = torch.float32, device = w.device)
    _score_weights_kernel[(N * C,)](w, class_mars, rows, shift[0], shift[1], C, K,
                                    TK = min(1024, max(16, host_next_power_of_2(K))), num_warps = 4)
    _logz_kernel[(N,)](shift[1], logz, C, TC = min(1024, max(16, host_next_power_of_2(C))), num_warps = 1)
    if candidates is not None:
        KC = candidates.size(1)
        BV, BH, warps = SCORE_CANDIDATE_TILES[N * KC > SCORE_CANDIDATE_PAIRS]
        _score_candidates_kernel[(N, host_cdiv(KC, BV))](table, w, shift[0], logz, classes.token_class, candidates,
                                                         out_row, out, N, K, V, C, KC, BV = BV, BH = BH,
                                                         EVEN_K = K % BH == 0, num_warps = warps)
        return logz
    dtype = "bf16" if table.dtype == torch.bfloat16 else "fp32"
    prec = "tf32" if dtype == "bf16" else precision
    BB, BV, BH, warps, stages = SCORE_TILES[(dtype, prec)][0 if N <= 16 else 1 if N <= 64 else 2]
    BV = min(BV, max(16, host_next_power_of_2(-(-V // C))))                    # many small classes: narrower tiles
    toks, tile_cls = classes.class_tiles(BV)
    _score_kernel[(host_cdiv(N, BB), tile_cls.numel())](table, w, shift[0], logz, toks, tile_cls, out_row, out, N, K, V,
                                                        C, BB = BB, BV = BV, BH = BH, EVEN_K = K % BH == 0,
                                                        PRECISION = prec, num_warps = warps, num_stages = stages)
    return logz
