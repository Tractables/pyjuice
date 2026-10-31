"""
The lifted product layer's top-down pass (see :mod:`.queries.backend.top_down`): the top-down value of every child of
every product, from the products' own (``element_flows``, or ``temp_flows`` for a fold's scratch row) and the other
children's inside values. One kernel per contraction of :mod:`.prod`, each the transpose of its forward:

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
from typing import Any, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.layer.kernels import autotune
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from .common import (NODE_TILE, SUM, _block_ptr, _child, _exit_width, _next_col, _interval, _logaddexp_tile, _lse_step, _pair,
                     _slot, _tf32_operand, _tile_rows, grid_chunks, tensor)
from .compact import _lane, _lane_ends
from .transitions import enumerated_args
from .prod import (BLOCK_BLOCK_PROGRAMS_PER_SM, BLOCK_BLOCK_TILES, EXACT_BELOW, PRODUCT_PRECISION, SKIP_EMPTY_TILES,
                   _chunk_needed, _flat_tile, _live_lanes, _lse_step_one_exp, _lse_value, _rect_product,
                   _rect_product_exact, _tile, _tile_load, compact_tiles)


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
        child's (``child_iv``) and parent's (``parent_iv``) interval, by ``prog.intervals``' numbers -- and those
        intervals each once (``child_ivs``, ``parent_ivs``; for compact storage, see :mod:`.compact`); for
        ``input_block`` / ``block_input``, every child's node tile ``group`` (:func:`~.common.node_groups`: its
        interval, number of parents and ``add``)
    :ivar tiles, tile_height: for ``input_block`` / ``block_input``, every child in node tiles (int32 on the device)
        and their height (:func:`~.common.node_tiles`) -- a query plan's lists of these launches are tiles too
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
    tiles: Optional[torch.Tensor] = None
    tile_height: int = 1


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


@triton.jit
def _entry_mass(masses, x, code_oc, eprobs, pr, lz, lw0, eidx0, num_enum, oc, own, vm, lane, ENUM: tl.constexpr,
                LOGW: tl.constexpr, OBSERVED: tl.constexpr):
    """The mass of an entry of class code ``x`` ([lanes]) at the parents' rows (``masses`` [TN, lanes] the rows'
    stored masses; ``ENUM``: ``eprobs`` + ``pr`` [TN, lanes] the rows' parameter rows of the enumerated tokens'
    probabilities, ``eidx0`` the columns there of the boundary's enumerated tokens, ``LOGW``: ``lw0`` their weights for
    the lane's sample and ``lz`` [TN, lanes] the rows' ``log Z``) -- ``OBSERVED``: or, where the lane observes a token
    (``oc`` >= 0), the token's own log-probability ``own`` if the entry's class is the observed one (``code_oc`` its
    code), else -inf. ``vm`` [TN, lanes] the parents and lanes on, ``lane`` [lanes] the lanes: the weights are read
    where a lane is on alone (a lane off the walk may have a sample past the last)."""
    if OBSERVED:
        miss, lmiss = vm & (oc < 0)[None, :], lane & (oc < 0)
    else:
        miss, lmiss = vm, lane
    if ENUM:
        enum = (x < 0)[None, :]
        stored = tl.load(masses + tl.maximum(x, 0)[None, :], mask = miss & ~enum, other = float("-inf"))
        e = tl.maximum(-1 - x, 0).to(tl.int64)             # the e-th enumerated token (a stored entry: the boundary's
        g = tl.load(eidx0 + e).to(tl.int64)                # first, or the padding past the last; unread)
        val = tl.log(tl.load(eprobs + pr * num_enum + g[None, :], mask = miss & enum, other = 0.0))
        if LOGW:
            val = val + tl.load(lw0 + e, mask = lmiss & (x < 0), other = 0.0)[None, :] - lz
        mass = tl.where(enum, val, stored)
    else:
        mass = tl.load(masses + x[None, :], mask = miss, other = float("-inf"))
    if OBSERVED:
        holds = x == code_oc
        return tl.where((oc >= 0)[None, :], tl.where(holds[None, :], own, float("-inf")), mass)
    return mass


@triton_jit
def _input_block_top_down_kernel(element_flows, temp_flows, node_flows, node_mars, obs_class, next_col, delta, state_id, col_of_state, class_mars,
                                mass_row, mass_stride, trans, mbase, mcount, mrank, bbase, mb_stride, mrank_stride,
                                pred_ptr, pred_col, pred_x, width, iv_of,
                                iv_info, iv_table, once_iv, live, lanes, lane_first, lane_live, coff, ccount, cstart,
                                ccolumns, crank, children, tiles, ptr, parents, reg_offset, reg_width, reg_first,
                                code, tok_ptr, eidx, eprobs, prow, lognorm, enum_logw, Cs, num_enum, logw_stride,
                                out_first, out_width, temp_width, B, n, W, C, K, HOMOGENEOUS: tl.constexpr, input_start, csteps, pid0,
                                C_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr,
                                PRUNE: tl.constexpr, ORDERED: tl.constexpr, COMPACT: tl.constexpr,
                                C_FULL: tl.constexpr, OUT_FULL: tl.constexpr, SAMPLE: tl.constexpr, ENUM: tl.constexpr,
                                LOGW: tl.constexpr, SAMPLE_MASSES: tl.constexpr, TN: tl.constexpr, SUBS: tl.constexpr,
                                TILE: tl.constexpr):
    """The right operand ``R`` over ``[t + 1, t2)`` of ``input @ block`` steps: ``O_R[q, j] = log sum over the
    parents, and over the entries ``(i, class)`` leading to ``q`` at boundary ``t``, of mass + O_out[i, j]``. One
    program per ``TN`` children of a node tile (:func:`~.common.node_tiles`: children of one interval with as many
    parents and one ``add``, whose lanes walk the same entries -- walked once for all of them) and tile of their
    lanes, flattened (sample, slot). ``ORDERED`` (a sum child under evidence pruning): the lanes in the query's lane
    order, live ones first -- a tile of dead lanes only stores -inf (a first writer) or nothing (a later one), since
    evidence usually leaves a few live (one per sample through an observed prefix). ``COMPACT``: the child's and the
    parents' values compact (:mod:`.compact`), a lane per pair of live columns, -inf at a pair the automaton does not
    join. ``SAMPLE``: class masses of each sample's own (see :class:`~..state.ClassMasses`; the transition masses as
    :meth:`~..compiled.program.Program.mass_args` lays them out, ``SAMPLE_MASSES`` each sample's own over its
    entries). Ungrouped, an entry's mass is read by its class's code (``ENUM``: an enumerated
    class's from its token, see :func:`_entry_mass`; ``Cs`` the stored classes, the table's width)."""
    c, real, c0 = _tile_rows(tiles, tl.program_id(0), TN, SUBS)
    tile = pid0 + tl.program_id(1)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c0 * 4 + 2)                                     # (the tile's: one for all)
    p0 = tl.load(ptr + c)
    f0 = tl.load(ptr + c0)                                                   # the tile's first child's parents:
    t = tl.load(parents + f0 * 7 + 2)                                        # every child's have its interval
    t2 = tl.load(parents + f0 * 7 + 5)
    X = _exit_width(width, t2, n)
    S, cpos, pairs = _interval(iv_of, iv_info, t + 1, t2, n)               # the children's slots
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
        idx, m, ml, dead = _child_lanes(tile, B, S, tl.load(children + c0 * 4 + 1), lanes, lane_first, lane_live,
                                        TILE, ORDERED)                     # (an interval's lane order)
    out = cp[:, None] + idx[None, :]
    om = real[:, None] & m[None, :]
    if dead:                                                                 # ORDERED: no live lane in the tile
        if add == 0:
            tl.store(out, tl.full([TN, TILE], float("-inf"), dtype = tl.float32), mask = om)
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
        mx = tl.full([TN, TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TN, TILE], dtype = tl.float32)
        for r in range(0, tl.load(ptr + c0 + 1) - f0):
            p = p0 + r
            u = tl.load(parents + p * 7 + 1).to(tl.int64)
            own = tl.load(node_mars + u[:, None] * B + b[None, :], mask = om & (oc >= 0)[None, :],
                          other = float("-inf"))                             # the observed token's log-prob
            op = _out_flows_ptr(tl.load(parents + p * 7 + 0), element_flows, temp_flows, out_first, out_width,
                                temp_width, OUT_TEMP)[:, None] + _parent_base(b, ostride, COMPACT)[None, :]
            if GROUPED:                                                      # the parents' transition masses: each
                mt = b.to(tl.int64) * mb_stride + t                          # step's rank's block of the boundary's
                masses = trans + tl.load(mbase + mt)[None, :] + \
                    tl.load(parents + p * 7 + 6).to(tl.int64)[:, None] * tl.load(mcount + mt).to(tl.int64)[None, :]
                bbt = tl.load(bbase + t)                                     # pairs, or of the lane's sample's entries
            else:
                mrow = tl.load(mass_row + (u - input_start)).to(tl.int64)[:, None]
                if SAMPLE:
                    mrow = mrow + b[None, :] * tl.load(mass_stride + (u - input_start)).to(tl.int64)[:, None]
                else:
                    mrow = mrow + (b * 0)[None, :]
                masses = class_mars + mrow * Cs
                if ENUM:                                                     # the rows' parameter rows, log Z
                    pr = tl.load(prow + mrow, mask = om, other = 0).to(tl.int64)
                    if LOGW:
                        lz = tl.load(lognorm + mrow, mask = om, other = 0.0)
                    else:
                        lz = pr
                    code_oc = tl.load(code + t * C + tl.maximum(oc, 0), mask = m, other = -1)
            for k in range(0, tl.max(e1 - e0, axis = 0)):                    # entry (i, x) -> q
                em = ml & (e0 + k < e1)
                i = tl.load(pred_col + e0 + k, mask = em, other = 0)
                x = tl.load(pred_x + e0 + k, mask = em, other = 0)
                sl = _parent_lane(_slot(iv_table, opos, i, j, X, em, OUT_FULL), b, once, coff, ccount, crank, kp, t,
                                  i, t2, j, em, B, n, W, COMPACT)
                ev = em & (sl >= 0)                                          # a dead parent lane: no term, nothing read
                vm = real[:, None] & ev[None, :]
                if GROUPED:                                                  # the observed class leads to q?
                    holds = _next_col(next_col, delta, state_id, col_of_state, t, i, tl.maximum(oc, 0), ev, K, W, C,
                                      HOMOGENEOUS) == q
                    if SAMPLE_MASSES:                                        # the sample's entry, or none: no mass
                        x = tl.load(mrank + b * mrank_stride + bbt + x, mask = ev, other = -1)
                    mass = tl.where((oc >= 0)[None, :], tl.where(holds[None, :], own, float("-inf")),
                                    tl.load(masses + x[None, :], mask = vm & ((oc < 0) & (x >= 0))[None, :],
                                            other = float("-inf")))
                elif ENUM:                                                   # is it this entry's class?
                    tk = tl.load(tok_ptr + t)
                    mass = _entry_mass(masses, x, code_oc, eprobs, pr, lz, enum_logw + b * logw_stride + tk, eidx + tk,
                                       num_enum, oc, own, vm, ev, ENUM, LOGW, True)
                else:
                    mass = _entry_mass(masses, x, oc, eprobs, masses, masses, enum_logw, eidx, num_enum, oc, own, vm,
                                       ev, ENUM, LOGW, True)
                ov = tl.load(op + sl[None, :], mask = vm, other = float("-inf"))
                mx, acc = _lse_step_one_exp(mx, acc, mass + ov)
        val = _lse_value(mx, acc)
        if COMPACT:
            val = tl.where(ml[None, :], val, float("-inf"))
        _store_flows(out, 0, val, om, add)


@triton_jit
def _block_input_top_down_kernel(element_flows, temp_flows, node_flows, node_mars, obs_class, next_col, delta, state_id, col_of_state, class_mars,
                                mass_row, mass_stride, trans, mbase, mcount, mrank, bbase, mb_stride, mrank_stride,
                                entry_ptr, entry_succ, entry_x, width,
                                iv_of, iv_info, iv_table, once_iv, live, lanes, lane_first, lane_live, coff, ccount,
                                cstart, ccolumns, crank, children, tiles, ptr, parents, reg_offset, reg_width,
                                reg_first, code, tok_ptr, eidx, eprobs, prow, lognorm, enum_logw, Cs, num_enum,
                                logw_stride, out_first, out_width, temp_width, B, n, W, C, K, HOMOGENEOUS: tl.constexpr, input_start, csteps, pid0,
                                C_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr,
                                PRUNE: tl.constexpr, ORDERED: tl.constexpr, COMPACT: tl.constexpr,
                                C_FULL: tl.constexpr, OUT_FULL: tl.constexpr, SAMPLE: tl.constexpr, ENUM: tl.constexpr,
                                LOGW: tl.constexpr, SAMPLE_MASSES: tl.constexpr, TN: tl.constexpr, SUBS: tl.constexpr,
                                TILE: tl.constexpr):
    """The left operand ``L`` over ``[t0, t)`` of ``block @ input`` steps (output over ``[t0, t + 1)``): ``O_L[i, q] =
    log sum over the parents and over the entries ``(successor, class)`` of ``q`` at boundary ``t`` of mass +
    O_out[i, successor]``. Node tiles, ``ORDERED``, ``COMPACT``, ``SAMPLE``, codes (``ENUM``): as
    :func:`_input_block_top_down_kernel`."""
    c, real, c0 = _tile_rows(tiles, tl.program_id(0), TN, SUBS)
    tile = pid0 + tl.program_id(1)
    row = tl.load(children + c * 4 + 0)
    reg = tl.load(children + c * 4 + 1)
    add = tl.load(children + c0 * 4 + 2)                                     # (the tile's: one for all)
    p0 = tl.load(ptr + c)
    f0 = tl.load(ptr + c0)
    t0 = tl.load(parents + f0 * 7 + 3)
    t = tl.load(parents + f0 * 7 + 5)
    M = tl.load(width + t)                                                   # the children's exit columns
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
        idx, m, ml, dead = _child_lanes(tile, B, S, tl.load(children + c0 * 4 + 1), lanes, lane_first, lane_live,
                                        TILE, ORDERED)
    out = cp[:, None] + idx[None, :]
    om = real[:, None] & m[None, :]
    if dead:                                                                 # ORDERED: no live lane in the tile
        if add == 0:
            tl.store(out, tl.full([TN, TILE], float("-inf"), dtype = tl.float32), mask = om)
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
        # an observed token: its own log-probability, then the output at the column its class leads to
        obs = ml & (oc >= 0)
        s = _next_col(next_col, delta, state_id, col_of_state, t, q, oc, obs, K, W, C, HOMOGENEOUS)
        s = tl.where(end & (s >= 0), 0, s)
        so = _parent_lane(_slot(iv_table, opos, i, s, X, obs, OUT_FULL), b, once, coff, ccount, crank, kp, t0, i, t + 1,
                          s, obs, B, n, W, COMPACT)
        miss = ml & (oc < 0)                                                 # a missing token: every entry (successor,
        any_miss = tl.max(tl.where(miss, 1, 0), axis = 0) > 0                # mass) of q
        e0 = tl.load(entry_ptr + t * (W + 1) + q, mask = miss, other = 0)
        e1 = tl.load(entry_ptr + t * (W + 1) + q + 1, mask = miss, other = 0)
        mx = tl.full([TN, TILE], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TN, TILE], dtype = tl.float32)
        for r in range(0, tl.load(ptr + c0 + 1) - f0):
            p = p0 + r
            u = tl.load(parents + p * 7 + 4).to(tl.int64)
            own = tl.load(node_mars + u[:, None] * B + b[None, :], mask = om & obs[None, :], other = float("-inf"))
            op = _out_flows_ptr(tl.load(parents + p * 7 + 0), element_flows, temp_flows, out_first, out_width,
                                temp_width, OUT_TEMP)[:, None] + _parent_base(b, ostride, COMPACT)[None, :]
            ov = tl.load(op + so[None, :], mask = real[:, None] & (obs & (so >= 0))[None, :], other = float("-inf"))
            mx, acc = _lse_step_one_exp(mx, acc, tl.where(obs[None, :], own + ov, float("-inf")))
            if any_miss:
                if GROUPED:                                                  # the parents' transition masses (as in
                    mt = b.to(tl.int64) * mb_stride + t                      # _input_block_top_down_kernel)
                    masses = trans + tl.load(mbase + mt)[None, :] + \
                        tl.load(parents + p * 7 + 6).to(tl.int64)[:, None] * tl.load(mcount + mt).to(tl.int64)[None, :]
                    bbt = tl.load(bbase + t)
                else:
                    mrow = tl.load(mass_row + (u - input_start)).to(tl.int64)[:, None]
                    if SAMPLE:
                        mrow = mrow + b[None, :] * tl.load(mass_stride + (u - input_start)).to(tl.int64)[:, None]
                    else:
                        mrow = mrow + (b * 0)[None, :]
                    masses = class_mars + mrow * Cs
                    if ENUM:
                        pr = tl.load(prow + mrow, mask = om, other = 0).to(tl.int64)
                        if LOGW:
                            lz = tl.load(lognorm + mrow, mask = om, other = 0.0)
                        else:
                            lz = pr
                for k in range(0, tl.max(e1 - e0, axis = 0)):
                    em = miss & (e0 + k < e1)
                    s = tl.load(entry_succ + e0 + k, mask = em, other = -1)
                    s = tl.where(end & (s >= 0), 0, s)
                    sl = _parent_lane(_slot(iv_table, opos, i, s, X, em, OUT_FULL), b, once, coff, ccount, crank,
                                      kp, t0, i, t + 1, s, em, B, n, W, COMPACT)
                    vm = real[:, None] & (em & (sl >= 0))[None, :]           # a live parent lane: else nothing read
                    ov = tl.load(op + sl[None, :], mask = vm, other = float("-inf"))
                    x = tl.load(entry_x + e0 + k, mask = em & (sl >= 0), other = 0)
                    if GROUPED:
                        if SAMPLE_MASSES:                                    # the sample's entry, or none: no mass
                            x = tl.load(mrank + b * mrank_stride + bbt + x, mask = em & (sl >= 0), other = -1)
                        mass = tl.load(masses + x[None, :], mask = vm & (x >= 0)[None, :], other = float("-inf"))
                    elif ENUM:                                               # (no lane here observes a token)
                        tk = tl.load(tok_ptr + t)                            # (`t0`: the block's start)
                        mass = _entry_mass(masses, x, x, eprobs, pr, lz, enum_logw + b * logw_stride + tk, eidx + tk,
                                           num_enum, x, ov, vm, em & (sl >= 0), ENUM, LOGW, False)
                    else:
                        mass = tl.load(masses + x[None, :], mask = vm, other = float("-inf"))
                    mx, acc = _lse_step_one_exp(mx, acc, mass + ov)
        val = _lse_value(mx, acc)
        if COMPACT:
            val = tl.where(ml[None, :], val, float("-inf"))
        _store_flows(out, 0, val, om, add)


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
            acc = acc * fA[:, None] * fB[None, :] + tl.dot(_tf32_operand(tl.exp(at - sA[:, None]), PRECISION),
                                                            _tf32_operand(tl.exp(bt - sB[None, :]), PRECISION),
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
                 out_first, out_width, temp_width, tflow_width, b, n, W, r, cc, tr, tc, Rn, Cn, valid, exact_below,
                 TILES_I, TILES_J, WORDS,
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
    op = _out_flows_ptr(tl.load(parents + p * 9 + 0), element_flows, temp_flows, out_first, out_width, tflow_width,
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
                                reg_first, prior_offset, prior_width, reg_prior, out_first, out_width, temp_width,
                                tflow_width, B, n, W, TILES_I, TILES_J, WORDS, SPLIT, GROUP, exact_below, pid0,
                                C_KIND: tl.constexpr,
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
    cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, tflow_width, C_KIND) + b * CS
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
                                          temp_width, tflow_width, b, n, W, r, cc, tr, tc, Rn, Cn, valid, exact_below,
                                          TILES_I, TILES_J, WORDS, O_KIND, OUT_TEMP, SIDE_L, TM, TN, TK, SUB_I, SUB_J,
                                          SKIP, PRUNE, PRECISION, False)
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
                                                  reg_prior, out_first, out_width, temp_width, tflow_width, b, n, W, r,
                                                  cc, tr, tc, Rn, Cn, valid,
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


@triton.jit
def _bb_compact_parent(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width, iv_of, iv_info,
                       iv_table, once_iv, coff, ccount, cstart, ccolumns, mars_offset, mars_width, reg_first,
                       prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, tflow_width, b, B, n,
                       rid, cid, pr, pc, rm, cm, valid, exact_below, O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                       SIDE_L: tl.constexpr, MARS: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr,
                       TK: tl.constexpr, PRECISION: tl.constexpr):
    """Parent ``p``'s contribution to a compact child's output tile (rows ``rid``, columns ``cid``; ``pr``, ``pc``:
    their places among the sample's live columns), a log-space matrix product over the sample's live columns at the
    shared boundary -- the exit ``t2`` (``SIDE_L``: ``O_out @ R^T``) or the entry ``t0`` (``L^T @ O_out``) -- of
    :func:`~.prod._rect_product`, computed again exactly (one shared column at a time) where its shifts flush a
    ``valid`` entry. The parent's values are compact, or kept once (by slot); the other operand's inside values
    compact too (``MARS``), or the sample's dense block, or kept once (by slot)."""
    t0 = tl.load(parents + p * 9 + 5)
    t1 = tl.load(parents + p * 9 + 6)
    t2 = tl.load(parents + p * 9 + 7)
    _, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    once = _once(once_iv, iv_of, t0, t2, n)
    op = _out_flows_ptr(tl.load(parents + p * 9 + 0), element_flows, temp_flows, out_first, out_width, tflow_width,
                        OUT_TEMP)
    o0 = tl.load(coff + tl.load(iv_of + t0 * (n + 1) + t2).to(tl.int64) * (B + 1) + b)
    ostride = tl.load(ccount + b * (n + 1) + t2)                             # the parent's exit columns live
    X_o = _exit_width(width, t2, n)
    if SIDE_L:                                                               # the other operand R over [t1, t2)
        ks = t2
        X_x = X_o
        XS, xpos, _ = _interval(iv_of, iv_info, t1, t2, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 3), tl.load(parents + p * 9 + 4), node_mars, prior_mars, temp,
                               mars_offset, mars_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        kx = tl.load(iv_of + t1 * (n + 1) + t2)
        xstride = ostride                                                    # its exit columns live
    else:                                                                    # ... or L over [t0, t1)
        ks = t0
        X_x = tl.load(width + t1)
        XS, xpos, _ = _interval(iv_of, iv_info, t0, t1, n)
        xp, x_own = _block_ptr(tl.load(parents + p * 9 + 1), tl.load(parents + p * 9 + 2), node_mars, prior_mars, temp,
                               mars_offset, mars_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               O_KIND)
        kx = tl.load(iv_of + t0 * (n + 1) + t1)
        xstride = tl.load(ccount + b * (n + 1) + t1)
    if MARS:
        x0 = tl.load(coff + kx.to(tl.int64) * (B + 1) + b)
        x_slot = ~x_own
    else:
        xp = xp + tl.where(x_own, b * XS, 0)
        x0 = 0
        x_slot = True
    krow = b * (n + 1) + ks
    ck = tl.load(ccount + krow)
    k0p = tl.load(cstart + krow)
    if SIDE_L:                                       # A[r, k] = O_out[r, k], B[k, c] = R[c, k]
        out, acc = _rect_product(op, o0, ostride, once, opos, X_o, xp, x0, xstride, x_slot, xpos, X_x, iv_table, rid,
                                 pr, rm, cid, pc, cm, ccolumns, k0p, ck, False, True, TM, TN, TK, PRECISION)
        if tl.min(tl.where(valid, acc, 1.0)) < exact_below:                  # flushed: one shared column at a time
            out = _rect_product_exact(op, o0, ostride, once, opos, X_o, xp, x0, xstride, x_slot, xpos, X_x, iv_table,
                                      rid, pr, rm, cid, pc, cm, ccolumns, k0p, ck, False, True, TM, TN)
    else:                                            # A[r, k] = L[k, r], B[k, c] = O_out[k, c]
        out, acc = _rect_product(xp, x0, xstride, x_slot, xpos, X_x, op, o0, ostride, once, opos, X_o, iv_table, rid,
                                 pr, rm, cid, pc, cm, ccolumns, k0p, ck, True, False, TM, TN, TK, PRECISION)
        if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
            out = _rect_product_exact(xp, x0, xstride, x_slot, xpos, X_x, op, o0, ostride, once, opos, X_o, iv_table,
                                      rid, pr, rm, cid, pc, cm, ccolumns, k0p, ck, True, False, TM, TN)
    return out


@triton_jit
def _block_block_top_down_compact_kernel(element_flows, temp_flows, node_flows, node_mars, prior_mars, temp, width,
                                        iv_of, iv_info, iv_table, once_iv, coff, ccount, cstart, ccolumns, children,
                                        children_idx, ptr, parents, reg_offset, reg_width, reg_first, mars_offset,
                                        mars_width, prior_offset, prior_width, reg_prior, out_first, out_width,
                                        temp_width, tflow_width, B, n, SPLIT, GROUP, exact_below, pid0,
                                        C_KIND: tl.constexpr, O_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                                        SIDE_L: tl.constexpr, MARS: tl.constexpr, TM: tl.constexpr,
                                        TN: tl.constexpr, TK: tl.constexpr, PRECISION: tl.constexpr,
                                        INDEXED: tl.constexpr):
    """:func:`_block_block_top_down_kernel` on compact values (:mod:`.compact`): one program per (child, sample, group
    of ``GROUP`` tiles of the sample's rectangle), each tile the log-sum over the child's parents of a log-space matrix
    product over the sample's live columns at the shared boundary (:func:`_bb_compact_parent`), -inf at the pairs the
    automaton does not join; the inside operands (``mars_offset``, ``mars_width``; ``temp_width``) compact too
    (``MARS``) or dense."""
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
    else:                                                                    # output (q, j): the child [t1, t2)
        ra = tl.load(parents + p0 * 9 + 6)
        rb = tl.load(parents + p0 * 9 + 7)
    Cn = _exit_width(width, rb, n)
    _, cpos, _ = _interval(iv_of, iv_info, ra, rb, n)
    kc = tl.load(iv_of + ra * (n + 1) + rb)
    cr = tl.load(ccount + b * (n + 1) + ra)                                  # the sample's rectangle
    cc_ = tl.load(ccount + b * (n + 1) + rb)
    r0 = tl.load(cstart + b * (n + 1) + ra)
    c0 = tl.load(cstart + b * (n + 1) + rb)
    cp = _flows_ptr(row, reg, node_flows, temp_flows, reg_offset, reg_width, reg_first, tflow_width, C_KIND) + \
        tl.load(coff + kc.to(tl.int64) * (B + 1) + b)
    NJ = tl.cdiv(cc_, TN)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(cr, TM) * NJ)):
        pr = (t // NJ) * TM + tl.arange(0, TM)
        pc = (t % NJ) * TN + tl.arange(0, TN)
        rm, cm = pr < cr, pc < cc_
        rid = tl.load(ccolumns + r0 + pr, mask = rm, other = 0)
        cid = tl.load(ccolumns + c0 + pc, mask = cm, other = 0)
        om = rm[:, None] & cm[None, :]
        valid = om & (_slot(iv_table, cpos, rid[:, None], cid[None, :], Cn, om, False) >= 0)
        tot = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
        for p in range(p0, p1):
            out = _bb_compact_parent(p, parents, element_flows, temp_flows, node_mars, prior_mars, temp, width, iv_of,
                                     iv_info, iv_table, once_iv, coff, ccount, cstart, ccolumns, mars_offset,
                                     mars_width, reg_first, prior_offset, prior_width, reg_prior, out_first, out_width,
                                     temp_width, tflow_width, b, B, n, rid, cid, pr, pc, rm, cm, valid, exact_below,
                                     O_KIND, OUT_TEMP, SIDE_L, MARS, TM, TN, TK, PRECISION)
            tot = _logaddexp_tile(tot, out)
        _store_flows(cp, pr[:, None].to(tl.int64) * cc_ + pc[None, :], tl.where(valid, tot, float("-inf")), om, add)


#: children (of a node tile) and (children x lanes) per program of the ``input @ block`` and ``block @ input``
#: top-down kernels, at most, and its warps: many small programs (Codpula diffusion, 4096 latents: 8 x 16 lanes 0.92
#: ms per call, 16 x 16 1.09, 16 x 64 1.89; 1 x 128, every child its own walk, 2.46)
TOP_DOWN_NODES = 8
TOP_DOWN_TILE_LANES = 128
TOP_DOWN_WARPS = 4

#: ``input @ block`` launches on compact values run the CUDA kernels (``c/walk.cu``) when those take them
#: (:func:`_input_block_cuda`); False: the Triton kernel always
INPUT_BLOCK_CUDA = True
#: the most children of a node tile a CTA of the CUDA kernel takes (a node tile's at most), and the most dynamic
#: shared memory (a sample's lanes of its children's parents, their class masses; 48 KB less the kernel's own): past
#: it, CTAs of fewer -- and past it for one child, CTAs of one child with as much as the device takes
INPUT_BLOCK_CUDA_NODES = NODE_TILE
INPUT_BLOCK_CUDA_SMEM = 45 * 1024
#: (lane, 4 children) items of a sample from which its CTA has 256 threads, not 128; up to which a CTA takes 8 samples
#: instead, a warp each, their columns not ordered by their terms (most of a CTA's threads would idle: left to right,
#: B 64, 61 -> 32 us a call)
INPUT_BLOCK_CUDA_WIDE = 512
INPUT_BLOCK_CUDA_WARP_ITEMS = 64
#: the most int32 of work space a launch takes (every sample's copy of its boundaries' predecessor lists): past it, the
#: Triton kernel
INPUT_BLOCK_CUDA_SPACE = 1 << 26
#: the codes of a unit's class masses (a sample's, at a CTA's children) the CUDA kernel's CTAs stage -- its terms' alone,
#: by rank --: a unit reading more (or a kept-once parent's every entry) is summed by the deferred kernel, every code
#: staged (`c/walk.cu`). Codpula-4096 x JSON hard_0 B64 diffusion (1224 codes, a unit's terms 43 at most): every code
#: staged took 39.5 KB a CTA, 2 CTAs an SM, the launches of a call 91-104 ms; 32 / 64 / 128 / 256 / 512 codes 26-29 /
#: 26-29 / 33-36 / 33-37 / 45-51 ms (8 samples a CTA, a warp each, up to 128)
INPUT_BLOCK_CUDA_CODES = 64
#: the deferred kernel's CTAs (0: as many as the device holds at once, at most one a deferred unit)
INPUT_BLOCK_CUDA_DEFER_CTAS = 0


def _input_block_boundaries(la: TopDownLaunch, prog) -> dict:
    """An ``input @ block`` launch's boundaries for the CUDA kernels, built on first use: ``ts`` (its parents' ``t``,
    ascending; ``tlist`` int32 on the device), ``t2i`` (int32 [n] on the device: every one's place among them),
    ``ecum`` (int64 [T + 1], host and ``ecum_t`` device: the predecessor entries at the boundaries before), and every
    child's interval as ``(t, t2)`` (``ta``, ``te``)."""
    st = la.host.get("cuda")
    if st is None:
        n, W = prog.n, prog.W
        counts = getattr(prog, "pred_counts", None)
        if counts is None:                                                   # every boundary's predecessor entries
            pp = prog.ungrouped_tables().pred_ptr.view(n, W + 1).cpu().numpy().astype(np.int64)   # (the CUDA walk
            counts = prog.pred_counts = pp[:, W] - pp[:, 0]                                        # runs ungrouped)
        ts = np.unique(la.host["parents"][:, 2]).astype(np.int64)
        ecum = np.zeros(ts.size + 1, dtype = np.int64)
        np.cumsum(counts[ts], out = ecum[1:])
        t2i = np.full(n, -1, dtype = np.int32)
        t2i[ts] = np.arange(ts.size, dtype = np.int32)
        ends = prog.iv_ends_np[np.asarray(la.host["child_ivs"], dtype = np.int64)]
        dev = prog.dev
        st = la.host["cuda"] = dict(ts = ts, tlist = torch.from_numpy(ts.astype(np.int32)).to(dev),
                                    t2i = torch.from_numpy(t2i).to(dev), ecum = ecum,
                                    ecum_t = torch.from_numpy(ecum).to(dev), ta = ends[:, 0] - 1, te = ends[:, 1])
    return st


def _work_space(prog, size: int, counters: int):
    """``(scratch, counters)``: int32 scratch of at least ``size`` on ``prog``'s device, and ``counters`` int32 zeros
    (each launch leaves them zero), both kept for the next launch (launches on one stream use them one after the
    other)."""
    ws, done = getattr(prog, "walk_space", None), getattr(prog, "walk_done", None)
    if ws is None or ws.numel() < size:
        ws = prog.walk_space = torch.empty(max(size, 1 << 16), dtype = torch.int32, device = prog.dev)
    if done is None or done.numel() < counters:
        done = prog.walk_done = torch.zeros(max(counters, 1 << 10), dtype = torch.int32, device = prog.dev)
    return ws, done


def _defer_space(prog, size: int) -> torch.Tensor:
    """The CUDA walk's list of deferred units: int32 of at least ``size`` on ``prog``'s device, zeros where it counts
    them (each launch leaves them zero), kept for the next launch."""
    d = getattr(prog, "walk_defer", None)
    if d is None or d.numel() < size:
        d = prog.walk_defer = torch.zeros(max(size, 1 << 12), dtype = torch.int32, device = prog.dev)
    return d


def _ibtd_smem_max(mod) -> int:
    """The most dynamic shared memory of a CTA of the CUDA kernel of one child on the current device (kept per
    device)."""
    dev = torch.cuda.current_device()
    kept = _IBTD_SMEM_MAX.get(dev)
    if kept is None:
        kept = _IBTD_SMEM_MAX[dev] = int(mod.ibtd_smem_max())
    return kept


_IBTD_SMEM_MAX = {}


def _input_block_cuda(la: TopDownLaunch, tiles: torch.Tensor, height: int, prog, compact, element_flows, temp_flows,
                      node_flows, node_mars, masses, obs_class, once_iv, common: dict, B: int,
                      allowed: Optional[torch.Tensor] = None) -> Optional[dict]:
    """
    Run an ``input @ block`` launch on compact values with the CUDA kernels (``c/walk.cu``: every sample's live terms
    per column, its columns by their terms, then a CTA per (node tile's children, sample) summing them from shared
    memory), if they take it -- ungrouped transitions, a term's place and class in 32 bits, its CTAs' shared memory
    within :data:`INPUT_BLOCK_CUDA_SMEM` (or a CTA of one child within the device's), its work space within
    :data:`INPUT_BLOCK_CUDA_SPACE` --, and return the CTAs' shape (``tn`` children, ``threads``, ``warps``: a warp per
    sample, ``smem``: bytes), else None. The values are the Triton kernel's, bit for bit. ``allowed``: None, or the
    classes a session's per-sample inputs leave each sample (:attr:`~.live.Liveness.allowed`): where a token is missing,
    the terms of a class ruled out (-inf, nothing to add) are not summed.
    """
    if not INPUT_BLOCK_CUDA or prog.trans_top_down.grouped or not obs_class.is_contiguous():
        return None
    from .c import get_module
    mod = get_module()
    n, W, C = prog.n, prog.W, prog.C
    ea = enumerated_args(prog, masses)
    ptr = prog.enum.ptr_host
    mw = int(ea["Cs"]) + (int((ptr[1:] - ptr[:-1]).max()) if prog.enum.num > 0 else 0)   # a child's staged masses:
    xb = max(1, (mw - 1).bit_length())                      # the stored columns, a boundary's most enumerated tokens
    if mod is None or xb >= 32 or W >= 1 << (32 - xb) or B > 65535:
        return None
    st = _input_block_boundaries(la, prog)
    mc = compact.max_count
    cq = int(mc[st["ts"] + 1].max())                                         # a sample's most live columns at t + 1
    lp = max(int((mc[st["ta"]] * mc[st["te"]]).max()), 1)                   # ... lanes of a parent's row
    team = lambda tn: (4 * tn * (lp + mw + 1) + 15) // 16 * 16             # a sample's shared memory (bytes:
    ms = min(mw, INPUT_BLOCK_CUDA_CODES)                                    # c/walk.cu's `ibtd_smem`), and its
    staged = lambda tn: (4 * tn * (lp + ms + 1) + 15) // 16 * 16           # CTAs' (`ibtd_smem_compact`)
    tn = min(height, INPUT_BLOCK_CUDA_NODES)
    while tn > 1 and team(tn) > INPUT_BLOCK_CUDA_SMEM:
        tn //= 2
    if team(tn) > INPUT_BLOCK_CUDA_SMEM and team(tn) > _ibtd_smem_max(mod):  # (one child: the device's most)
        return None
    items = cq * int(mc[st["te"]].max()) * max(tn // 4, 1)                  # a sample's most (lane, 4 children)
    warps = items <= INPUT_BLOCK_CUDA_WARP_ITEMS and 8 * staged(tn) <= INPUT_BLOCK_CUDA_SMEM
    threads = 256 if warps or items >= INPUT_BLOCK_CUDA_WIDE else 128
    T, nterms = st["ts"].size, B * int(st["ecum"][-1])
    space = nterms + 2 * T * B * W + T * B * (mw + 1)                       # (terms, cnt, perm, codes, ncodes)
    if space > INPUT_BLOCK_CUDA_SPACE or 4 * (2 * ((mw + 31) // 32) + 1) > 48 * 1024:
        return None
    ws, done = _work_space(prog, space, T * B)
    defer = _defer_space(prog, 2 + 2 * (tiles.numel() // NODE_TILE) * (height // tn) * B)
    tt, ivs = prog.trans_top_down, prog.intervals
    c_kind, out_temp, (c_full, out_full) = la.kinds
    sample = masses.batch > 1
    classes = allowed is not None and tt.pred_class is not None
    a_stride = n * C if classes and allowed.size(0) > 1 else 0               # (one set: every sample's)
    mod.ibtd_cuda(element_flows, temp_flows, node_flows, node_mars, obs_class, masses.table, masses.row,
                  masses.stride if sample else masses.row, tt.pred_ptr, tt.pred_col, tt.pred_x, prog.width_t, ivs.of,
                  ivs.info, ivs.table, once_iv, compact.off, compact.count, compact.start, compact.columns,
                  compact.rank, la.children, tiles, la.ptr, la.parents, common["reg_offset"], common["reg_width"],
                  common["reg_first"], st["tlist"], st["t2i"], st["ecum_t"], ws[:nterms],
                  ws[nterms:nterms + T * B * W], ws[nterms + T * B * W:nterms + 2 * T * B * W], done,
                  ws[nterms + 2 * T * B * W:space - T * B], ws[space - T * B:space],
                  common["out_first"], common["out_width"], common["temp_width"], B, n, W, C, prog.input_start,
                  height // tn, tn, threads, warps, not warps, lp, cq, sample, c_kind, out_temp, c_full, out_full,
                  ea["code"], ea["tok_ptr"], ea["eidx"], ea["eprobs"], ea["prow"], ea["lognorm"], ea["enum_logw"],
                  ea["Cs"], mw, ea["num_enum"], ea["logw_stride"], ea["LOGW"],
                  tt.pred_class if classes else tt.pred_x, allowed if classes else obs_class, a_stride, classes,
                  ms, defer, INPUT_BLOCK_CUDA_DEFER_CTAS)
    return dict(tn = tn, threads = threads, warps = warps, smem = staged(tn) * (8 if warps else 1), codes = ms)


def run_top_down(launches, bufs, prog, regions, element_flows, temp_flows, node_flows, node_mars, masses, obs_class,
                live, elem_first: int, elem_width: int, B: int, prior: dict, skip, exact: bool, once_iv: torch.Tensor,
                lanes = None, compact = None, tflow_width = None, inside_regions = None, inside: bool = False,
                allowed: Optional[torch.Tensor] = None):
    """
    The top-down pass of one product layer's stage: every :class:`TopDownLaunch` in ``launches``, each with the plan's
    index list of the children it computes (int32 on the device) or None for every one.

    :param bufs: ``(temp, temp_width)``, the layer's scratch rows, holding the folds' inside values (recomputed)
    :param regions: ``offset``, ``width``, ``first`` and ``slots`` of every sum region (device tensors) of the buffer
        the children are written to (``node_flows``, or ``prior_flows`` for a batch of one)
    :param element_flows, temp_flows, node_flows: the top-down buffers (``temp_flows`` as wide as ``temp``)
    :param masses: the query's :class:`~..state.ClassMasses` (per sample, the ``SAMPLE`` kernels)
    :param prior: where the sum operands the plan keeps once are (see :func:`~.common._block_ptr`)
    :param skip: the top-down pass's :class:`~.prod.SkipMasks` (its triples are the parents' ``(interval A, B)`` specs)
    :param exact: see :func:`~.prod.run_products`
    :param once_iv: int32 [intervals] (device, by ``prog.intervals``' numbers): 1 where the query keeps an interval's
        top-down values once, for a batch of one: a parent there is read with sample stride 0
    :param lanes: None, or under evidence pruning the query's :class:`~.live.LaneOrder` ``(perm, reg_first,
        reg_live)``: a sum child's lanes run in that order, its dead ones only set to -inf
    :param compact: None, or the query's :class:`~.compact.CompactLayout`: the per-sample values (children and
        parents) are compact -- ``regions``, ``elem_width`` and ``tflow_width`` are their rows'; the children are
        written, and the parents read, a lane per pair of live columns
    :param tflow_width: the scratch rows' width in ``temp_flows`` (None: ``bufs``')
    :param inside_regions: the sum regions' tables in ``node_mars`` (None: ``regions``), for the inside values a
        ``block @ block`` launch reads
    :param inside: whether those inside values are compact too (by ``compact``'s layout; ``inside_regions`` and
        ``bufs`` are their rows')
    :param allowed: None, or the classes a session's per-sample inputs leave each sample
        (:attr:`~.live.Liveness.allowed`), by which the CUDA ``input @ block`` walk drops terms
        (:func:`_input_block_cuda`)
    """
    temp, temp_width = bufs
    tflow_width = temp_width if tflow_width is None else tflow_width
    inside_regions = regions if inside_regions is None else inside_regions
    common = dict(reg_offset = regions["offset"], reg_width = regions["width"], reg_first = regions["first"],
                  out_first = elem_first, out_width = elem_width, temp_width = tflow_width)
    n, W, C = prog.n, prog.W, prog.C
    ivs = prog.intervals
    prune, live = live is not None, live if live is not None else obs_class
    tt, grouped = prog.trans_top_down, prog.trans_top_down.grouped
    margs = prog.mass_args()                                              # the query's transition masses
    mpos = tuple(margs[k] for k in ("trans", "mbase", "mcount", "mrank", "bbase", "mb_stride", "mrank_stride"))
    sample = masses.batch > 1
    if compact is not None:
        cl = compact
        lists = dict(coff = cl.off, ccount = cl.count, cstart = cl.start, ccolumns = cl.columns, csteps = cl.steps)
    else:
        lists = dict(coff = live, ccount = live, cstart = live, ccolumns = live, csteps = 0)
    for la, item in launches:
        idx = tensor(item)
        Cn = la.children.size(0) if idx is None else idx.numel()
        indexed = dict(children_idx = la.children if idx is None else idx, INDEXED = idx is not None)
        lanes_max = compact.lanes(la.host["child_ivs"]) if compact is not None else B * la.S_max   # a child's most
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
            tiles, height = (la.tiles, la.tile_height) if idx is None else (idx, item.height)
            if la.form == "input_block" and compact is not None and lanes_max > 0 and \
                    _input_block_cuda(la, tiles, height, prog, compact, element_flows, temp_flows, node_flows,
                                      node_mars, masses, obs_class, once_iv, common, B, allowed):
                continue
            tn = min(height, TOP_DOWN_NODES)
            tile = max(16, min(_flat_tile(max(lanes_max, 1)), TOP_DOWN_TILE_LANES // tn))
            kernel = _input_block_top_down_kernel if la.form == "input_block" else _block_input_top_down_kernel
            tables = (tt.pred_ptr, tt.pred_col, prog.mass_x("pred_x")) if la.form == "input_block" else \
                (tt.entry_ptr, tt.entry_succ, prog.mass_x("entry_x"))
            for first, size in grid_chunks(host_cdiv(lanes_max, tile), 1):
                kernel[(tiles.numel() // NODE_TILE * (height // tn), size)](
                    element_flows, temp_flows, node_flows, node_mars, obs_class, prog.next_col_arg, prog.delta,
                    prog.state_id, prog.col_of_state, masses.table,
                    masses.row, masses.stride if sample else masses.row, *mpos, *tables, prog.width_t,
                    ivs.of, ivs.info, ivs.table, once_iv, live,
                    *(lanes if ordered else (live, live, live)), crank = compact.rank if compact is not None else live,
                    children = la.children, tiles = tiles, ptr = la.ptr, parents = la.parents, B = B, n = n, W = W,
                    C = C, K = prog.K, input_start = prog.input_start, pid0 = first, C_KIND = c_kind,
                    OUT_TEMP = out_temp, HOMOGENEOUS = prog.homogeneous,
                    GROUPED = grouped, PRUNE = prune, ORDERED = ordered, COMPACT = compact is not None,
                    C_FULL = c_full, OUT_FULL = out_full, SAMPLE = sample, SAMPLE_MASSES = margs["SAMPLE_MASSES"],
                    TN = tn, SUBS = height // tn, TILE = tile,
                    num_warps = TOP_DOWN_WARPS, **common, **lists, **enumerated_args(prog, masses))
        elif compact is not None:                    # block @ block on compact values, a sample's rectangle at a time
            c_kind, o_kind, out_temp = la.kinds
            ends = prog.iv_ends_np[la.host["child_ivs"]]
            rmax, cmax = int(compact.max_count[ends[:, 0]].max()), int(compact.max_count[ends[:, 1]].max())
            TM, TN, TK, warps = compact_tiles(rmax, cmax)
            tiles = host_cdiv(rmax, TM) * host_cdiv(cmax, TN)
            if tiles == 0:
                continue
            split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, Cn * B))
            group = host_cdiv(tiles, max(1, split))
            split = host_cdiv(tiles, group)
            for first, size in grid_chunks(Cn * B * split, 0):
                _block_block_top_down_compact_kernel[(size,)](
                    element_flows, temp_flows, node_flows, node_mars, prior["mars"], temp, prog.width_t, ivs.of,
                    ivs.info, ivs.table, once_iv, compact.off, compact.count, compact.start, compact.columns,
                    la.children, ptr = la.ptr, parents = la.parents, mars_offset = inside_regions["offset"],
                    mars_width = inside_regions["width"], prior_offset = prior["offset"], prior_width = prior["width"],
                    reg_prior = prior["reg_prior"], tflow_width = tflow_width, B = B, n = n, SPLIT = split,
                    GROUP = group, exact_below = EXACT_BELOW, pid0 = first, C_KIND = c_kind, O_KIND = o_kind,
                    OUT_TEMP = out_temp, SIDE_L = la.side == "L", MARS = inside, TM = TM, TN = TN, TK = TK,
                    PRECISION = PRODUCT_PRECISION, num_warps = warps, num_stages = 1, **indexed,
                    **dict(common, temp_width = temp_width))
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
                        num_warps = warps, num_stages = 1, tflow_width = tflow_width, **indexed,
                        **dict(common, temp_width = temp_width))

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
