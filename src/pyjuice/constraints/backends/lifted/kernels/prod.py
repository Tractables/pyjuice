"""
The lifted product layer. A product chains its children's blocks in scope order: a log-space product over the
boundary columns that adjacent children share. There is one table-driven implementation for every circuit
shape. :class:`~pyjuice.constraints.backends.lifted.forward.Program` folds each product into binary steps,
right to left. Each step is one of three contractions, chosen by what its two operands are, never by the
circuit's shape:

* ``input @ block`` (:func:`_input_block_kernel`): an input at position ``t``, then a block from ``t + 1``.
  Class ``c`` of the token moves column ``i`` at ``t`` to ``next_col[t, i, c]``, so this is a gather over
  the successors of ``i``. The block may be the identity, which materializes the input's own block (a product
  with a single input child, or an input next to another input);
* ``block @ input`` (:func:`_block_input_kernel`): the mirror case, a gather over the predecessors of each
  column at ``t + 1``;
* ``block @ block`` (:func:`_block_block_kernel`): a per-sample log-space matrix product over the shared
  boundary, skipping the (output tile, shared chunk) pairs the automaton cannot join (:class:`SkipMasks`).

Blocks are either sum nodes' rows in ``node_mars`` or intermediate results in a scratch buffer. Outputs go
to the product layer's ``element_mars`` rows, or to scratch when another step reads them. A row holds one
block per sample, sample-major: sample ``b``'s block in columns ``[b * slots, (b + 1) * slots)``. A block keeps
only the (entry column ``i``, exit column ``j``) pairs the automaton joins, in row-major order: slot ``i * X +
j`` when it joins them all, else through its interval's tables (:class:`~..plan.IntervalTables`). An input child
is read only as the log-probability of its observed token (pyjuice's own input layers wrote it) or, when the
token is missing, through its transition masses (:func:`transition_masses`): for a column and one of its
successors, the log-mass of the token classes that lead there -- so a missing token costs a gather per
successor, not per class. Under evidence pruning (``live``, see :mod:`.live`), a dead slot is -inf and costs no
work.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ....distributions.categorical import _exact_dot
from ..plan import Reachability, _tile_any

#: operand kinds: a sum node's block in ``node_mars``, an intermediate block in scratch, the identity
SUM, TEMP, IDENTITY = 0, 1, 2

#: CUDA's limit on the programs along a grid's first axis, and along each of the others
MAX_GRID = (2 ** 31 - 1, 65535)


def grid_chunks(count: int, axis: int):
    """``(first, size)`` of every launch that ``count`` programs along grid ``axis`` take: one launch, unless the
    count passes CUDA's limit there (each kernel adds ``first`` to its program id on that axis)."""
    limit = MAX_GRID[axis]
    return [(first, min(limit, count - first)) for first in range(0, count, limit)]

#: fields of a step, per contraction (int32 rows of the step tables the Program builds)
INPUT_BLOCK_FIELDS = ("out", "u", "t", "r_row", "r_reg", "t2", "t_off")    # input u at t, then block over [t+1, t2)
BLOCK_INPUT_FIELDS = ("out", "l_row", "l_reg", "t0", "u", "t", "t_off")    # block over [t0, t), then input u at t
# (``t_off``: where the step's transition masses start in the transition buffer)
BLOCK_BLOCK_FIELDS = ("out", "l_row", "l_reg", "r_row", "r_reg", "t0", "t1", "t2")
COPY_FIELDS = ("out", "l_row", "l_reg", "t0", "t2")


@triton.jit
def _block_ptr(row, reg, node_mars, temp, reg_offset, reg_width, reg_first, temp_width, KIND: tl.constexpr):
    """Start of a block operand: a sum node's row in its region of ``node_mars``, or a scratch row."""
    if KIND == 0:
        base = tl.load(reg_offset + reg) + (row - tl.load(reg_first + reg)) * tl.load(reg_width + reg)
        return node_mars + base
    else:
        return temp + row.to(tl.int64) * temp_width


@triton.jit
def _out_ptr(row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP: tl.constexpr):
    if OUT_TEMP:
        return temp + row.to(tl.int64) * temp_width
    else:
        return element_mars + (row - out_first).to(tl.int64) * out_width


@triton.jit
def _exit_width(width, t2, n):
    """Columns at the exit boundary ``t2``: a sequence end is one column (summed over the accepting states)."""
    return tl.where(t2 == n, 1, tl.load(width + tl.minimum(t2, n)))


@triton.jit
def _interval(iv_of, iv_info, a, b, n):
    """``(slots, positions, pairs)`` of the interval ``[a, b)``: its blocks' slots per sample, and where its
    position and pair tables start (-1 when it keeps every pair; see :class:`~..plan.IntervalTables`)."""
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
def _live_lanes(live, m, b, a, i, e, j, n, W, PRUNE: tl.constexpr):
    """The lanes of ``m`` whose slot ``(i at boundary a, j at boundary e)`` of sample ``b`` is live (all of them
    unless ``PRUNE``)."""
    if PRUNE:
        row = live + b * (n + 1) * W
        return m & (tl.load(row + a * W + i, mask = m, other = 0) != 0) & \
            (tl.load(row + e * W + j, mask = m, other = 0) != 0)
    else:
        return m


@triton.jit
def _lse_step(m, acc, v):
    """One step of an online log-sum-exp (``m``: running max, ``acc``: running sum scaled by ``exp(-m)``)."""
    m_new = tl.maximum(m, v)
    scale = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))
    shift = tl.where(m_new == float("-inf"), 0.0, m_new)
    return m_new, acc * scale + tl.exp(v - shift)


@triton.jit
def _lse_value(m, acc):
    return tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)


@triton.jit
def _input_block_kernel(element_mars, node_mars, temp, obs_class, next_col, class_mars, mass_row, trans, entry_ptr,
                        entry_succ, entry_x, width, iv_of, iv_info, iv_table, live, steps, reg_offset, reg_width,
                        reg_first, out_first, out_width, temp_width, B, n, W, C, input_start, pid0,
                        R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr, PRUNE: tl.constexpr,
                        OUT_FULL: tl.constexpr, R_FULL: tl.constexpr, TILE: tl.constexpr):
    """One program per step and tile of its whole output, flattened (sample, slot): with the samples' blocks
    contiguous, a step's output is one run of columns, and the lanes stay full whatever the widths."""
    s = tl.program_id(0)
    tile = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 7 + 0)
    u = tl.load(steps + s * 7 + 1).to(tl.int64)
    t = tl.load(steps + s * 7 + 2)
    t2 = tl.load(steps + s * 7 + 5)
    X = _exit_width(width, t2, n)
    S, _, pairs = _interval(iv_of, iv_info, t, t2, n)
    if tile * TILE < B * S:
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < B * S
        b = (idx // S).to(tl.int64)
        v = _pair(iv_table, pairs, idx % S, m, OUT_FULL)
        i = v // X
        j = v % X
        ml = _live_lanes(live, m, b, t, i, t2, j, n, W, PRUNE)
        end = t + 1 == n
        if R_KIND != 2:
            RS, rpos, _ = _interval(iv_of, iv_info, t + 1, t2, n)            # the block's slots
            rp = _block_ptr(tl.load(steps + s * 7 + 3), tl.load(steps + s * 7 + 4), node_mars, temp, reg_offset,
                            reg_width, reg_first, temp_width, R_KIND) + b * RS
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)            # observed token's class, -1 if missing

        # an observed token: its own log-probability (the input region of node_mars), then the block from the
        # column its class leads to
        obs = ml & (oc >= 0)
        q = tl.load(next_col + (t * W + i) * C + oc, mask = obs, other = -1)
        if R_KIND == 2:
            rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
        else:
            sl = _slot(iv_table, rpos, q, j, X, obs, R_FULL)
            rv = tl.load(rp + sl, mask = obs & (sl >= 0), other = float("-inf"))
        val = tl.load(node_mars + u * B + b, mask = obs, other = float("-inf")) + rv

        # a missing token: log-sum over i's transitions of their mass + the block from their successor (skipped when
        # every lane is observed)
        miss = ml & (oc < 0)
        if tl.max(tl.where(miss, 1, 0), axis = 0) > 0:
            mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
            acc = tl.zeros([TILE], dtype = tl.float32)
            if GROUPED:                                                      # i's entries: successor, transition mass
                p0 = tl.load(entry_ptr + t * (W + 1) + i, mask = miss, other = 0)
                p1 = tl.load(entry_ptr + t * (W + 1) + i + 1, mask = miss, other = 0)
                masses = trans + tl.load(steps + s * 7 + 6).to(tl.int64)      # the step's transition masses
                for k in range(0, tl.max(p1 - p0, axis = 0)):
                    pm = miss & (p0 + k < p1)
                    q = tl.load(entry_succ + p0 + k, mask = pm, other = -1)
                    if R_KIND == 2:
                        rv = tl.where(pm & (q == j), 0.0, float("-inf"))
                    else:
                        sl = _slot(iv_table, rpos, q, j, X, pm, R_FULL)
                        rv = tl.load(rp + sl, mask = pm & (sl >= 0), other = float("-inf"))
                    mass = tl.load(masses + tl.load(entry_x + p0 + k, mask = pm, other = 0), mask = pm,
                                   other = float("-inf"))
                    mx, acc = _lse_step(mx, acc, mass + rv)
            else:                                                            # every class, in order: the same class
                masses = class_mars + tl.load(mass_row + (u - input_start)).to(tl.int64) * C    # (and mass) in
                for c in range(C):                                           # every lane
                    q = tl.load(next_col + (t * W + i) * C + c, mask = miss, other = -1)
                    if R_KIND == 2:
                        rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
                    else:
                        sl = _slot(iv_table, rpos, q, j, X, miss, R_FULL)
                        rv = tl.load(rp + sl, mask = miss & (sl >= 0), other = float("-inf"))
                    mx, acc = _lse_step(mx, acc, tl.load(masses + c) + rv)
            val = tl.where(oc >= 0, val, _lse_value(mx, acc))
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + idx, val, mask = m)


@triton.jit
def _block_input_kernel(element_mars, node_mars, temp, obs_class, next_col, class_mars, mass_row, trans, pred_ptr,
                        pred_col, pred_x, width, iv_of, iv_info, iv_table, live, steps, reg_offset, reg_width,
                        reg_first, out_first, out_width, temp_width, B, n, W, C, input_start, pid0,
                        L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr, PRUNE: tl.constexpr,
                        OUT_FULL: tl.constexpr, L_FULL: tl.constexpr, TILE: tl.constexpr):
    """One program per step and tile of its whole output, flattened (sample, slot) as in
    :func:`_input_block_kernel`: every lane takes a log-sum over its exit column's predecessor entries (the
    program loops to the longest list among its lanes), each through its mass for a missing token, or the
    observed token's own log-probability when the entry holds its class."""
    s = tl.program_id(0)
    tile = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 7 + 0)
    t0 = tl.load(steps + s * 7 + 3)
    u = tl.load(steps + s * 7 + 4).to(tl.int64)
    t = tl.load(steps + s * 7 + 5)
    M = tl.load(width + t)
    X = _exit_width(width, t + 1, n)
    S, _, pairs = _interval(iv_of, iv_info, t0, t + 1, n)
    LS, lpos, _ = _interval(iv_of, iv_info, t0, t, n)                       # the block's slots
    if tile * TILE < B * S:
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < B * S
        b = (idx // S).to(tl.int64)
        v = _pair(iv_table, pairs, idx % S, m, OUT_FULL)
        i = v // X
        j = v % X                                                            # exit column (at boundary t + 1)
        ml = _live_lanes(live, m, b, t0, i, t + 1, j, n, W, PRUNE)
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)
        own = tl.load(node_mars + u * B + b, mask = m, other = float("-inf"))  # the input region of node_mars
        lp = _block_ptr(tl.load(steps + s * 7 + 1), tl.load(steps + s * 7 + 2), node_mars, temp, reg_offset,
                        reg_width, reg_first, temp_width, L_KIND) + b * LS
        if GROUPED:
            masses = trans + tl.load(steps + s * 7 + 6).to(tl.int64)          # the step's transition masses
        else:                                                                # the input's class masses
            masses = class_mars + tl.load(mass_row + (u - input_start)).to(tl.int64) * C
        end = t + 1 == n
        e0 = tl.load(pred_ptr + t * (W + 1) + j, mask = ml, other = 0)
        e1 = tl.load(pred_ptr + t * (W + 1) + j + 1, mask = ml, other = 0)
        mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TILE], dtype = tl.float32)
        for k in range(0, tl.max(e1 - e0, axis = 0)):                        # column q at t -> j
            em = ml & (e0 + k < e1)
            q = tl.load(pred_col + e0 + k, mask = em, other = 0)
            x = tl.load(pred_x + e0 + k, mask = em, other = 0)
            if GROUPED:                                                      # does the observed class lead to j?
                nq = tl.load(next_col + (t * W + q) * C + tl.maximum(oc, 0), mask = em, other = -1)
                holds = (nq == j) | (end & (nq >= 0))
            else:                                                            # is it this entry's class?
                holds = x == oc
            mass = tl.where(oc >= 0, tl.where(holds, own, float("-inf")),
                            tl.load(masses + x, mask = em, other = float("-inf")))
            sl = _slot(iv_table, lpos, i, q, M, em, L_FULL)
            lv = tl.load(lp + sl, mask = em & (sl >= 0), other = float("-inf"))
            mx, acc = _lse_step(mx, acc, lv + mass)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + idx, _lse_value(mx, acc), mask = m)


@triton.jit
def _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I: tl.constexpr, SUB_J: tl.constexpr):
    """Whether shared chunk ``c`` can contribute to output tile ``(ti, tj)``: the OR of its ``SUB_I x SUB_J``
    mask tiles' bits (``mb``: the step's triple's masks)."""
    need = 0
    for si in tl.static_range(SUB_I):
        for sj in tl.static_range(SUB_J):
            bi = ti * SUB_I + si
            bj = tj * SUB_J + sj
            w = tl.load(mb + (bi * TILES_J + bj) * WORDS + c // 32, mask = (bi < TILES_I) & (bj < TILES_J), other = 0)
            need = need | ((w >> (c % 32)) & 1)
    return need != 0


@triton.jit
def _tile_load(ptr, iv_table, positions, r, c, ncols, rm, cm):
    """The tile of a block at rows ``r`` and columns ``c``, -inf where not joined."""
    mask = rm[:, None] & cm[None, :]
    sl = _slot(iv_table, positions, r[:, None], c[None, :], ncols, mask, False)
    return tl.load(ptr + sl, mask = mask & (sl >= 0), other = float("-inf"))


@triton.jit
def _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, live_q, i, j, ti, tj, E, M, X, TILES_I, TILES_J, WORDS,
                      TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr, SUB_J: tl.constexpr,
                      SKIP: tl.constexpr, PRUNE: tl.constexpr, PRECISION: tl.constexpr):
    """``log(sum_q exp(L[i, q] + R[q, j]))`` for one output tile, in one pass over the shared chunks it needs: the
    product of the operands' exponentials, shifted by every row's and column's running maximum, the sum rescaled
    whenever a maximum grows (``lpos`` and ``rpos``: the operands' position tables; pruned, ``live_q``: the shared
    boundary's liveness, a chunk without a live column skipped)."""
    im, jm = i < E, j < X
    mL = tl.full([TM], float("-inf"), dtype = tl.float32)
    mR = tl.full([TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for c in range(0, tl.cdiv(M, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            qs = c * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_q + qs, mask = qs < M, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            q = c * TK + tl.arange(0, TK)
            qm = q < M
            lt = _tile_load(lp, iv_table, lpos, i, q, M, im, qm)
            rt = _tile_load(rp, iv_table, rpos, q, j, X, qm, jm)
            nL = tl.maximum(mL, tl.max(lt, axis = 1))
            nR = tl.maximum(mR, tl.max(rt, axis = 0))
            sL = tl.where(nL == float("-inf"), 0.0, nL)
            sR = tl.where(nR == float("-inf"), 0.0, nR)
            fL = tl.where(nL == float("-inf"), 1.0, tl.exp(mL - sL))
            fR = tl.where(nR == float("-inf"), 1.0, tl.exp(mR - sR))
            acc = acc * fL[:, None] * fR[None, :] + tl.dot(tl.exp(lt - sL[:, None]), tl.exp(rt - sR[None, :]),
                                                            input_precision = PRECISION)
            mL, mR = nL, nR
    sL = tl.where(mL == float("-inf"), 0.0, mL)
    sR = tl.where(mR == float("-inf"), 0.0, mR)
    return tl.log(acc) + sL[:, None] + sR[None, :]


@triton.jit
def _block_block_kernel(element_mars, node_mars, temp, width, iv_of, iv_info, iv_table, live, steps, triples,
                        skip_bits, reg_offset, reg_width, reg_first, out_first, out_width, temp_width, B, n, W, TILES_I,
                        TILES_J, WORDS, SPLIT, GROUP,
                        pid0, L_KIND: tl.constexpr, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                        TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr,
                        SUB_J: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr, PRECISION: tl.constexpr):
    """One program per (step, sample, group of ``GROUP`` consecutive output tiles), its tiles in turn: the sample's
    two blocks are contiguous and stay in L1 across them. A (step, sample) has ``SPLIT`` groups: 1 unless the
    launch has too few (step, sample) pairs to fill the GPU."""
    pid = pid0 + tl.program_id(0)
    g = pid % SPLIT
    s = pid // SPLIT // B
    b = (pid // SPLIT % B).to(tl.int64)
    out_row = tl.load(steps + s * 8 + 0)
    t0 = tl.load(steps + s * 8 + 5)
    t1 = tl.load(steps + s * 8 + 6)
    t2 = tl.load(steps + s * 8 + 7)
    E = tl.load(width + t0)
    M = tl.load(width + t1)
    X = _exit_width(width, t2, n)
    LS, lpos, _ = _interval(iv_of, iv_info, t0, t1, n)
    RS, rpos, _ = _interval(iv_of, iv_info, t1, t2, n)
    OS, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    lp = _block_ptr(tl.load(steps + s * 8 + 1), tl.load(steps + s * 8 + 2), node_mars, temp, reg_offset,
                    reg_width, reg_first, temp_width, L_KIND) + b * LS
    rp = _block_ptr(tl.load(steps + s * 8 + 3), tl.load(steps + s * 8 + 4), node_mars, temp, reg_offset,
                    reg_width, reg_first, temp_width, R_KIND) + b * RS
    op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP) + b * OS
    mb = skip_bits + tl.load(triples + s).to(tl.int64) * TILES_I * TILES_J * WORDS
    lrow = live + b * (n + 1) * W                                            # the sample's liveness (if pruned)
    NJ = tl.cdiv(X, TN)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(E, TM) * NJ)):
        ti = t // NJ
        tj = t % NJ
        i = ti * TM + tl.arange(0, TM)
        j = tj * TN + tl.arange(0, TN)
        if PRUNE:                                                            # a tile without a live slot: -inf
            lr = tl.load(lrow + t0 * W + i, mask = i < E, other = 0) != 0
            lc = tl.load(lrow + t2 * W + j, mask = j < X, other = 0) != 0
            out = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
            if (tl.max(lr.to(tl.int32), axis = 0) > 0) & (tl.max(lc.to(tl.int32), axis = 0) > 0):
                out = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow + t1 * W, i, j, ti, tj, E, M, X,
                                        TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
            out = tl.where(lr[:, None] & lc[None, :], out, float("-inf"))
        else:
            out = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow, i, j, ti, tj, E, M, X, TILES_I, TILES_J,
                                    WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
        om = (i < E)[:, None] & (j < X)[None, :]
        sl = _slot(iv_table, opos, i[:, None], j[None, :], X, om, False)
        tl.store(op + sl, out, mask = om & (sl >= 0))


@triton.jit
def _copy_kernel(element_mars, node_mars, temp, iv_of, iv_info, steps, reg_offset, reg_width, reg_first, out_first,
                 out_width, temp_width, B, n, pid0, L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                 TILE: tl.constexpr):
    s = tl.program_id(0)
    pid = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 5 + 0)
    t0 = tl.load(steps + s * 5 + 3)
    t2 = tl.load(steps + s * 5 + 4)
    S, _, _ = _interval(iv_of, iv_info, t0, t2, n)
    ncols = S * B
    if pid * TILE < ncols:
        cols = pid * TILE + tl.arange(0, TILE)
        m = cols < ncols
        lp = _block_ptr(tl.load(steps + s * 5 + 1), tl.load(steps + s * 5 + 2), node_mars, temp, reg_offset,
                        reg_width, reg_first, temp_width, L_KIND)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + cols, tl.load(lp + cols, mask = m), mask = m)


@dataclass(frozen = True)
class TransitionTables:
    """
    Every boundary's transitions as entries (column, successor, mass), the successor of a sequence end being its one
    column. Ungrouped, one entry per (column, class) that leads somewhere, its mass the class's mass. Grouped, one
    per (column, successor) pair, its mass the pair's transition mass: the mass of every class that leads there
    (:func:`transition_masses`) -- a missing token then costs a gather per successor, not per class.

    :ivar grouped: which of the two
    :ivar entry_ptr: [n, W + 1] int32, the entries of column ``q`` at ``t``: ``entry_ptr[t, q] .. entry_ptr[t, q + 1]``
        (global numbers; ``entry_ptr[t, 0]``: the boundary's first)
    :ivar entry_succ, entry_x: [entries] int32, every entry's successor column and where its mass is: its class,
        or its pair's number within the boundary
    :ivar pred_ptr, pred_col, pred_x: the entries leading to column ``j`` at ``t + 1``, as a CSR (``pred_ptr``
        [n, W + 1]) into their columns and their mass indices
    :ivar pair_of: grouped only, [n, W, C] int32: the pair (within the boundary) every class of every column leads
        to, -1 where it leads nowhere
    :ivar num_pairs: grouped only, pairs per boundary (Python ints)
    :ivar max_pairs: grouped only, the most pairs of any column
    """

    grouped: bool
    entry_ptr: torch.Tensor
    entry_succ: torch.Tensor
    entry_x: torch.Tensor
    pred_ptr: torch.Tensor
    pred_col: torch.Tensor
    pred_x: torch.Tensor
    pair_of: Optional[torch.Tensor]
    num_pairs: list
    max_pairs: int


def max_successors(next_col: torch.Tensor, width, n: int) -> int:
    """The most distinct columns one column reaches in one token, at any boundary (a sequence end being one)."""
    W = next_col.size(1)
    nc, most = next_col.cpu().long(), 0
    for t in range(n - 1):                                                  # the last boundary: one column
        succ = nc[t, :int(width[t])]
        q, c = torch.nonzero(succ >= 0, as_tuple = True)
        if q.numel():
            pairs = torch.unique(q * (W + 1) + succ[q, c])
            most = max(most, int(torch.bincount(pairs // (W + 1)).max()))
    return max(1, most)


def transition_tables(next_col: torch.Tensor, width, n: int, grouped: bool) -> TransitionTables:
    """The :class:`TransitionTables` of a layout's ``next_col`` [n, W, C] and boundary widths."""
    W, C = next_col.size(1), next_col.size(2)
    nc = next_col.cpu().long()
    entry_ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    pred_ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    pair_of = torch.full((n, W, C), -1, dtype = torch.int64) if grouped else None
    succs, xs, cols, pxs, num_pairs = [], [], [], [], []
    base = max_pairs = 0
    for t in range(n):
        wt = int(width[t])
        succ = nc[t, :wt]                                                    # [wt, C]
        if t + 1 == n:
            succ = torch.where(succ >= 0, 0, -1)                             # a sequence end: one column
        q, c = torch.nonzero(succ >= 0, as_tuple = True)                     # by column, then class
        if grouped:
            key, inverse = torch.unique(q * (W + 1) + succ[q, c], return_inverse = True)    # by column, successor
            eq, es, ex = key // (W + 1), key % (W + 1), torch.arange(key.numel())
            pair_of[t, q, c] = inverse
            num_pairs.append(key.numel())
            max_pairs = max(max_pairs, int(torch.bincount(eq, minlength = 1).max()) if key.numel() else 0)
        else:
            eq, es, ex = q, succ[q, c], c
        entry_ptr[t] = base + torch.searchsorted(eq, torch.arange(W + 1))
        by_succ = torch.argsort(es * (W + 1) + eq, stable = True)
        pred_ptr[t] = base + torch.searchsorted(es[by_succ], torch.arange(W + 1))
        succs.append(es); xs.append(ex); cols.append(eq[by_succ]); pxs.append(ex[by_succ])
        base += eq.numel()
    dev = next_col.device
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    cat = lambda xs: i32(torch.cat(xs)) if base > 0 else torch.zeros(1, dtype = torch.int32, device = dev)
    return TransitionTables(grouped = grouped, entry_ptr = i32(entry_ptr), entry_succ = cat(succs), entry_x = cat(xs),
                            pred_ptr = i32(pred_ptr), pred_col = cat(cols), pred_x = cat(pxs),
                            pair_of = i32(pair_of) if grouped else None, num_pairs = num_pairs, max_pairs = max_pairs)


#: :func:`transition_masses`' tiles: the most rows (input steps at one position) and columns per program, classes per
#: product, warps. Rows and columns shrink until there are enough programs (they never change a result); the pairs
#: per product follow the widest column, up to 64.
TRANSITION_TILES = dict(TS = 64, QT = 4, TC = 16, warps = 4, programs = 1024)


def transition_job(rows: torch.Tensor, dev):
    """
    The steps whose transition masses one :func:`transition_masses` call builds: ``rows`` [N, 3] (input row,
    position, offset of its masses), as ``(rows, tiles, TS)`` on ``dev``, the rows ordered by position and ``tiles``
    [tiles, 3] (first row, number of rows, position) cutting them into tiles of at most ``TS`` rows at one position
    (fewer when there are few rows, so that there are enough programs).
    """
    TS = min(TRANSITION_TILES["TS"], max(16, triton.next_power_of_2(rows.size(0)) // 32))
    rows = rows[torch.argsort(rows[:, 1], stable = True)]
    pos, counts = torch.unique_consecutive(rows[:, 1], return_counts = True)
    tiles, start = [], 0
    for t, k in zip(pos.tolist(), counts.tolist()):
        tiles += [(start + i, min(TS, k - i), t) for i in range(0, k, TS)]
        start += k
    i32 = lambda x: torch.as_tensor(x, dtype = torch.int32).reshape(-1, 3).to(dev).contiguous()
    return i32(rows), i32(tiles), TS


@triton.jit
def _transition_kernel(class_mars, mass_row, trans, rows, tiles, pair_of, entry_ptr, width, W, C, input_start,
                       num_qtiles, pid0, TS: tl.constexpr, QT: tl.constexpr, TP: tl.constexpr, TC: tl.constexpr):
    """
    One program per (tile of rows at one position, ``QT`` columns): per column, its pairs ``TP`` at a time, as the
    product of the rows' class masses -- linear, scaled by 2^64, in natural class order -- with the 0/1 table of
    which of the pairs every class leads to. A class mass is a sum of fp32 probabilities, so 0 or at least 2^-149:
    scaled, a normal float, which tensor cores do not flush. Its log is the transition mass.
    """
    pid = pid0 + tl.program_id(0)
    tile = pid // num_qtiles
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = tl.arange(0, TS)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    off = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)
    masses = class_mars + tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64) * C
    base = tl.load(entry_ptr + t * (W + 1))                                  # the boundary's first pair
    jj = tl.arange(0, TP)
    for qi in range(QT):
        q = (pid % num_qtiles) * QT + qi
        if q < tl.load(width + t):
            g_end = tl.load(entry_ptr + t * (W + 1) + q + 1) - base
            col = pair_of + (t * W + q).to(tl.int64) * C
            for g0 in range(tl.load(entry_ptr + t * (W + 1) + q) - base, g_end, TP):
                acc = tl.zeros([TS, TP], dtype = tl.float32)
                for c0 in range(0, C, TC):
                    c = c0 + tl.arange(0, TC)
                    cmask = c < C
                    x = tl.exp(tl.load(masses[:, None] + c[None, :], mask = rmask[:, None] & cmask[None, :],
                                       other = float("-inf")) + 44.3614195558365)                # x 2^64
                    g = tl.load(col + c, mask = cmask, other = -1) - g0                          # pair in the tile
                    acc += _exact_dot(x, tl.where(g[:, None] == jj[None, :], 1.0, 0.0))
                tl.store(trans + off[:, None] + (g0 + jj)[None, :], tl.log(acc) - 44.3614195558365,
                         mask = rmask[:, None] & (g0 + jj < g_end)[None, :])


def transition_masses(trans, job, class_mars, prog):
    """
    Write the transition masses of every step of ``job`` (from :func:`transition_job`): for each column ``q`` at the
    input's position ``t`` and each successor ``s``, ``log sum_{c: next_col[t, q, c] = s} exp(class_mars[r, c])``
    (``r``: input ``u``'s row of the class masses), at ``trans[offset + p]`` for the pair's number ``p`` within the
    boundary.
    """
    rows, tiles, TS = job
    tt, cfg = prog.trans, TRANSITION_TILES
    W, C = prog.next_col.size(1), prog.next_col.size(2)
    QT = cfg["QT"]
    while QT > 1 and tiles.size(0) * triton.cdiv(max(prog.width), QT) < cfg["programs"]:
        QT //= 2
    num_qtiles = triton.cdiv(max(prog.width), QT)
    TP = min(64, max(16, triton.next_power_of_2(tt.max_pairs)))
    for first, size in grid_chunks(tiles.size(0) * num_qtiles, 0):
        _transition_kernel[(size,)](class_mars, prog.mass_row, trans, rows, tiles, tt.pair_of, tt.entry_ptr,
                                    prog.width_t, W, C,
                                    prog.input_start, num_qtiles, first, TS = TS, QT = QT, TP = TP, TC = cfg["TC"],
                                    num_warps = cfg["warps"])


@dataclass(frozen = True)
class SkipMasks:
    """
    Which chunks of the shared boundary every output tile of a ``block @ block`` step needs. For triple ``s``
    (boundaries ``(t0, t1, t2)``), word ``(s * tiles_i + ti) * tiles_j * words + tj * words + c // 32`` of
    ``bits`` has bit ``c % 32`` set when some column of entry tile ``ti`` (at ``t0``) reaches some column of chunk
    ``c`` (at ``t1``) that reaches some column of exit tile ``tj`` (at ``t2``). Every other (tile, chunk) pair is
    structurally ``-inf`` on one side and adds exactly zero. Tiles are ``tm x tn`` columns, chunks ``tk``.

    :ivar bits: int32 ``[triples * tiles_i * tiles_j * words]`` (one word if there are no triples)
    """

    bits: torch.Tensor
    tiles_i: int
    tiles_j: int
    words: int
    tm: int
    tn: int
    tk: int


def skip_masks(reach: Reachability, triples: Sequence[Tuple[int, int, int]], tm: int = 16, tn: int = 16,
               tk: int = 16, device = None) -> SkipMasks:
    """
    The :class:`SkipMasks` of ``triples`` (``(t0, t1, t2)`` boundaries, in the order their index refers to), from
    the occupancy ``reach`` keeps; ``tm``, ``tn`` and ``tk`` are multiples of its tile.
    """
    T = reach.tile
    if tm % T or tn % T or tk % T:
        raise ValueError(f"Tiles ({tm}, {tn}, {tk}) must be multiples of the occupancy tile {T}.")
    if len(triples) == 0:
        return SkipMasks(torch.zeros(1, dtype = torch.int32, device = device), 0, 0, 1, tm, tn, tk)
    lefts = sorted({(a, b) for a, b, _ in triples})
    rights = sorted({(b, c) for _, b, c in triples})
    L = [_tile_any(reach.occupancy(a, b), tm // T, tk // T) for a, b in lefts]     # [entry tiles, chunks]
    R = [_tile_any(reach.occupancy(b, c), tk // T, tn // T) for b, c in rights]    # [chunks, exit tiles]
    I = max(m.size(0) for m in L)
    J = max(m.size(1) for m in R)
    words = max(1, math.ceil(max(m.size(1) for m in L) / 32))
    NC = 32 * words

    def stack(ms, shape):
        out = torch.zeros(len(ms), *shape, dtype = torch.bool)
        for k, m in enumerate(ms):
            out[k, :m.size(0), :m.size(1)] = m
        return out

    Ls, Rs = stack(L, (I, NC)), stack(R, (NC, J))
    lpos = {iv: k for k, iv in enumerate(lefts)}
    rpos = {iv: k for k, iv in enumerate(rights)}
    li = torch.tensor([lpos[a, b] for a, b, _ in triples], dtype = torch.long)
    ri = torch.tensor([rpos[b, c] for _, b, c in triples], dtype = torch.long)
    shifts = torch.arange(32, dtype = torch.int64)
    chunk = max(1, (1 << 26) // (I * J * NC))
    parts = []
    for g0 in range(0, len(triples), chunk):
        need = Ls[li[g0:g0 + chunk], :, None, :] & Rs[ri[g0:g0 + chunk]].transpose(1, 2)[:, None, :, :]
        w = (need.view(need.size(0), I, J, words, 32).to(torch.int64) << shifts).sum(dim = -1)
        parts.append(torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32).flatten())
    return SkipMasks(torch.cat(parts).to(device), I, J, words, tm, tn, tk)


#: :func:`_block_block_kernel`'s output tile (entry x exit columns), shared chunk and warps; the tile is a multiple
#: of :class:`SkipMasks`'s, and the chunk equal to it
BLOCK_BLOCK_TILES = dict(TM = 16, TN = 32, TK = 16, warps = 4)
#: the products' dot precision: "ieee" (exact fp32, CUDA cores) or "tf32x3" (three TF32 tensor-core passes)
PRODUCT_PRECISION = "ieee"
#: skip the (output tile, shared chunk) pairs the automaton cannot join; False computes every pair
SKIP_EMPTY_TILES = True
#: a ``block @ block`` launch with fewer (step, sample) pairs than this many per SM spreads each pair's output tiles
#: over several programs (a tile's result does not depend on which program computes it). 16 from a sweep: a
#: 600-state automaton x30 at batch 1, x8 at 4; Ctrl-G's on a 16-latent PD unchanged (more loses there: cheap tiles
#: lose their operands' L1 reuse)
BLOCK_BLOCK_PROGRAMS_PER_SM = 16


def _tile(cols: int) -> int:
    return min(256, max(16, triton.next_power_of_2(cols)))


def _flat_tile(lanes: int) -> int:
    """Lanes per program of a step whose output is flattened (sample, entry column, exit column)."""
    return min(512, max(128, triton.next_power_of_2(lanes) // 8))


def run_products(stages, bufs, prog, regions, element_mars, node_mars, class_mars, obs_class, live,
                 elem_first: int, elem_width: int, B: int):
    """
    Every step of one product layer, stage by stage (a stage reads only earlier stages' scratch rows). Products
    are fp32-level whatever the query's precision, which applies to the sum layers: ``block @ block`` dots in
    :data:`PRODUCT_PRECISION` (exact fp32 by default), the other contractions exact fp32 log-sum-exps.

    :param stages: the layer's stages from :class:`~pyjuice.constraints.backends.lifted.forward.Program`: lists
        of ``(form, kinds, steps, E_max, X_max, S_max)`` launches (the most entry columns, exit columns and slots of
        their steps' outputs; ``block @ block`` steps: ``(steps, triple index)``,
        the latter indexing ``Program.skip``; ``input @ block`` and ``block @ input`` steps: ``(steps, chunks)``,
        ``(first, end, job)`` step ranges, ``job`` the :func:`transition_job` that builds their transition masses
        just before them, or None when the Program keeps them all)
    :param bufs: ``(temp, temp_width)``, the layer's scratch rows
    :param live: None, or the query's liveness (:class:`~.live.Pruning`'s ``live``)
    """
    temp, temp_width = bufs
    common = dict(reg_offset = regions["offset"], reg_width = regions["width"], reg_first = regions["first"],
                  out_first = elem_first, out_width = elem_width, temp_width = temp_width)
    n, W, C = prog.n, prog.next_col.size(1), prog.next_col.size(2)
    ivs = prog.intervals
    prune, live = live is not None, live if live is not None else obs_class
    for launches in stages:
        for form, kinds, steps, E_max, X_max, S_max in launches:
            S = steps[0].size(0) if isinstance(steps, tuple) else steps.size(0)
            if form in ("input_block", "block_input"):
                steps, chunks = steps
                trans, tt = prog.transition_buffer(), prog.trans
                for s0, s1, job in chunks:
                    part = steps[s0:s1]
                    if job is not None:                              # this chunk's transition masses, in scratch
                        transition_masses(trans, job, class_mars, prog)
                    if form == "input_block":
                        r_kind, out_temp, (out_full, *r_full) = kinds
                        tile = _flat_tile(B * S_max)
                        for first, size in grid_chunks(triton.cdiv(B * S_max, tile), 1):
                            _input_block_kernel[(s1 - s0, size)](
                                element_mars, node_mars, temp, obs_class, prog.next_col, class_mars, prog.mass_row,
                                trans, tt.entry_ptr, tt.entry_succ, tt.entry_x, prog.width_t, ivs.of, ivs.info,
                                ivs.table, live, part, B = B, n = n, W = W,
                                C = C, input_start = prog.input_start, pid0 = first, R_KIND = r_kind,
                                OUT_TEMP = out_temp, GROUPED = tt.grouped, PRUNE = prune, OUT_FULL = out_full,
                                R_FULL = all(r_full), TILE = tile, num_warps = 4, **common)
                    else:
                        l_kind, out_temp, (out_full, l_full) = kinds
                        tile = _flat_tile(B * S_max)
                        for first, size in grid_chunks(triton.cdiv(B * S_max, tile), 1):
                            _block_input_kernel[(s1 - s0, size)](
                                element_mars, node_mars, temp, obs_class, prog.next_col, class_mars, prog.mass_row,
                                trans, tt.pred_ptr, tt.pred_col, tt.pred_x, prog.width_t, ivs.of, ivs.info, ivs.table,
                                live, part, B = B, n = n, W = W,
                                C = C, input_start = prog.input_start, pid0 = first, L_KIND = l_kind,
                                OUT_TEMP = out_temp, GROUPED = tt.grouped, PRUNE = prune, OUT_FULL = out_full,
                                L_FULL = l_full, TILE = tile, num_warps = 4, **common)
            elif form == "block_block":
                l_kind, r_kind, out_temp = kinds
                steps, triples = steps
                cfg, skip = BLOCK_BLOCK_TILES, prog.skip
                assert cfg["TK"] == skip.tk and cfg["TM"] % skip.tm == 0 and cfg["TN"] % skip.tn == 0
                tiles = triton.cdiv(E_max, cfg["TM"]) * triton.cdiv(X_max, cfg["TN"])
                split = min(tiles, triton.cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, S * B))
                group = triton.cdiv(tiles, max(1, split))
                split = triton.cdiv(tiles, group)                         # no group past the widest step's tiles
                for first, size in grid_chunks(S * B * split, 0):
                    _block_block_kernel[(size,)](
                        element_mars, node_mars, temp, prog.width_t, ivs.of, ivs.info, ivs.table, live, steps,
                        triples, skip.bits, B = B, n = n, W = W,
                        TILES_I = skip.tiles_i, TILES_J = skip.tiles_j, WORDS = skip.words, SPLIT = split,
                        GROUP = group, pid0 = first, L_KIND = l_kind, R_KIND = r_kind, OUT_TEMP = out_temp,
                        TM = cfg["TM"], TN = cfg["TN"], TK = cfg["TK"], SUB_I = cfg["TM"] // skip.tm,
                        SUB_J = cfg["TN"] // skip.tn, SKIP = SKIP_EMPTY_TILES, PRUNE = prune,
                        PRECISION = PRODUCT_PRECISION,
                        num_warps = cfg["warps"], num_stages = 1, **common)
            else:                                                            # "copy"
                l_kind, out_temp = kinds
                tile = _tile(S_max * B)
                for first, size in grid_chunks(triton.cdiv(S_max * B, tile), 1):
                    _copy_kernel[(S, size)](
                        element_mars, node_mars, temp, ivs.of, ivs.info, steps, B = B, n = n, pid0 = first,
                        L_KIND = l_kind, OUT_TEMP = out_temp, TILE = tile, num_warps = 4, **common)
