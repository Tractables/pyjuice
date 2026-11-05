"""
The top-down ``input @ block`` and ``block @ input`` kernels: a parent's flow to the entries it reaches, each entry's mass read per class.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from pyjuice.utils.kernel_launcher import triton_jit
from ..common import _exit_width, _next_col, _interval, _pair, _slot, _tile_rows
from ..compact import _lane_ends
from ..prod.launches import _live_lanes, _lse_step_one_exp, _lse_value
from .launches import _child_lanes, _flows_ptr, _once, _out_flows_ptr, _parent_base, _parent_lane, _store_flows


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
                                TILE: tl.constexpr, PDL: tl.constexpr = False):
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
    if PDL:                                             # (the launch before it, done)
        gdc_wait()
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
                # (masked: a lane past the tile's outputs has no sample, its `b` anywhere -- see _input_block_kernel)
                masses = trans + tl.load(mbase + mt, mask = m, other = 0)[None, :] + \
                    tl.load(parents + p * 7 + 6).to(tl.int64)[:, None] * \
                    tl.load(mcount + mt, mask = m, other = 0).to(tl.int64)[None, :]
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
                                TILE: tl.constexpr, PDL: tl.constexpr = False):
    """The left operand ``L`` over ``[t0, t)`` of ``block @ input`` steps (output over ``[t0, t + 1)``): ``O_L[i, q] =
    log sum over the parents and over the entries ``(successor, class)`` of ``q`` at boundary ``t`` of mass +
    O_out[i, successor]``. Node tiles, ``ORDERED``, ``COMPACT``, ``SAMPLE``, codes (``ENUM``): as
    :func:`_input_block_top_down_kernel`."""
    if PDL:                                             # (the launch before it, done)
        gdc_wait()
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
                    mt = b.to(tl.int64) * mb_stride + t                      # _input_block_top_down_kernel; masked)
                    masses = trans + tl.load(mbase + mt, mask = m, other = 0)[None, :] + \
                        tl.load(parents + p * 7 + 6).to(tl.int64)[:, None] * \
                        tl.load(mcount + mt, mask = m, other = 0).to(tl.int64)[None, :]
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
