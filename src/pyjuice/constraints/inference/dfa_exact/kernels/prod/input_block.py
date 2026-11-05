"""
The ``input @ block`` and ``block @ input`` kernels: an input's masses, read per class, against a block over the interval it joins.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from pyjuice.utils.kernel_launcher import triton_jit
from ..common import _block_ptr, _child, _exit_width, _interval, _lse_step, _next_col
from ..compact import _operand_slot
from .launches import _fresh_tile, _lane_rows, _lse_step_one_exp, _lse_value, _out_count, _out_lanes, _out_ptr


@triton.jit
def _list_range(succ_start, succ_count, b, t, i, miss, n, W, E):
    """``(l0, l1)``, int64: lanes' column ``i`` at boundary ``t`` (samples ``b``, ``miss`` the lanes on) -- its
    successor list's range in the :class:`~.live.SuccessorLists`' entries (those whose successor is live; ``start``
    counted from the sample's first)."""
    bt = b.to(tl.int64) * n + t
    l0 = b.to(tl.int64) * E + tl.load(succ_start + bt * W + i, mask = miss, other = 0)
    return l0, l0 + tl.load(succ_count + bt * W + i, mask = miss, other = 0)


@triton.jit
def _ib_term(mx, acc, ee, pm, entry_succ, entry_x, rp, iv_table, rpos, j, X, r_own, crank, b, t, rbase, ce, n, W,
             masses, mrank, mrank_stride, bbt, eprobs, tk, pr, lz, enum_logw, eidx, num_enum, logw_stride,
             R_KIND: tl.constexpr, R_FULL: tl.constexpr, COMPACT: tl.constexpr, GROUPED: tl.constexpr,
             SAMPLE_MASSES: tl.constexpr, ENUM: tl.constexpr, LOGW: tl.constexpr):
    """:func:`_input_block_kernel`'s term of entry ``ee`` (lanes ``pm``): its successor's block value plus its mass
    (a transition mass, ``GROUPED`` -- ``SAMPLE_MASSES``: the lane's sample's, at the entry's rank among the sample's,
    ``mrank`` [samples, entries] from ``bbt`` on for the boundary, see :meth:`~..compiled.program.Program.mass_args`;
    else the class's, :func:`_lane_mass`), log-summed into ``(mx, acc)``."""
    q = tl.load(entry_succ + ee, mask = pm, other = -1)
    if R_KIND == 2:
        rv = tl.where(pm & (q == j), 0.0, float("-inf"))
    else:
        rv = _block_value(rp, iv_table, rpos, q, j, X, pm, r_own, crank, b, t, rbase, ce, n, W, R_FULL, COMPACT)
    x = tl.load(entry_x + ee, mask = pm, other = 0)
    if GROUPED:
        if SAMPLE_MASSES:                                                    # the sample's entry, or none: no mass
            x = tl.load(mrank + b.to(tl.int64) * mrank_stride + bbt + x, mask = pm, other = -1)
            pm = pm & (x >= 0)
        mass = tl.load(masses + x, mask = pm, other = float("-inf"))
    else:
        mass = _lane_mass(masses, x, pm, eprobs, tk, pr, lz, enum_logw, eidx, b, num_enum, logw_stride, ENUM, LOGW)
    return _lse_step(mx, acc, mass + rv)


@triton.jit
def _ibg_term(mx, acc, ee, pm, lm, gm, entry_succ, entry_x, rp, iv_table, rpos, j, X, own, crank, b, t, rbase, ce, n, W,
              masses, eprobs, tk, pr, lz, enum_logw, eidx, num_enum, logw_stride, R_FULL: tl.constexpr,
              COMPACT: tl.constexpr, ENUM: tl.constexpr, LOGW: tl.constexpr, ONE_EXP: tl.constexpr):
    """:func:`_input_block_group_kernel`'s term of entry ``ee`` (lanes ``pm``) for the ``[G, TILE]`` lanes (``gm`` the
    group's steps, ``lm`` both): the successor's slot once, the steps' block values and masses (:func:`_group_mass`).
    The caller makes ``lm`` ahead of every load masked by ``pm``: Triton's layout pass may move such a load to the
    ``[G, TILE]`` layout, reading ``pm`` as ``lm`` has it, and some variants (``ENUM``) fail to compile if ``lm`` comes
    after the load ("operand #1 does not dominate this use")."""
    q = tl.load(entry_succ + ee, mask = pm, other = -1)
    x = tl.load(entry_x + ee, mask = pm, other = 0)
    sl = _operand_slot(iv_table, rpos, q, j, X, pm, own, crank, b, t + 1, q, rbase, ce, n, W, R_FULL, COMPACT)
    rv = tl.load(rp + sl[None, :], mask = gm[:, None] & (pm & (sl >= 0))[None, :], other = float("-inf"))
    mass = _group_mass(masses, x, lm, pm, tk, pr, lz, eidx, eprobs, enum_logw, b, logw_stride,
                       num_enum, ENUM, LOGW)
    if ONE_EXP:
        return _lse_step_one_exp(mx, acc, mass + rv)
    return _lse_step(mx, acc, mass + rv)


@triton.jit
def _lane_mass(masses, x, pm, eprobs, tk, pr, lz, enum_logw, eidx, b, num_enum, logw_stride, ENUM: tl.constexpr,
               LOGW: tl.constexpr):
    """The mass of class code ``x`` ([lanes]) at the lanes' rows of the class masses: ``masses`` the rows' pointers
    (``class_mars + mrow * Cs``), a stored class's column read there, an enumerated class's (``ENUM``, code ``-1 -
    e``: the e-th enumerated token of the boundary, from ``tk``) its token's probability ``eprobs[pr, eidx[tk + e]]``,
    ``LOGW``: times its weight for the lane's sample ``b`` over the row's ``Z`` (``lz`` its log) -- ``tk``, ``pr`` and
    ``lz`` from :func:`_lane_rows`; -inf where ``pm`` is off. The weights are read where a lane is on alone: a lane off
    the step's output has a sample past the last, a dead class the count past the boundary's tokens."""
    if ENUM:
        enum = x < 0
        stored = tl.load(masses + tl.maximum(x, 0), mask = pm & ~enum, other = float("-inf"))
        e = tl.maximum(-1 - x, 0).to(tl.int64)                                  # (a stored entry: token 0, unread)
        g = tl.load(eidx + tk + e).to(tl.int64)                              # (its column of the table)
        val = tl.log(tl.load(eprobs + pr * num_enum + g, mask = pm & enum, other = 0.0))
        if LOGW:
            val = val + tl.load(enum_logw + b * logw_stride + tk + e, mask = pm & enum, other = 0.0) - lz
        return tl.where(enum, val, stored)
    return tl.load(masses + x, mask = pm, other = float("-inf"))


@triton.jit
def _group_mass(masses, x, lm, pm, tk, pr, lz, eidx, eprobs, enum_logw, b, logw_stride, num_enum, ENUM: tl.constexpr,
                LOGW: tl.constexpr):
    """:func:`_lane_mass` for a group of steps: [G, TILE], the mass of class code ``x`` ([TILE]) at the lanes' rows
    (``masses`` [G, TILE] pointers), the lanes on ``lm`` [G, TILE] (the group's steps by ``pm`` [TILE]); ``ENUM``: ``tk``
    the boundary's first enumerated token, ``pr`` the rows' parameter rows, ``lz`` (``LOGW``) their log Z."""
    if ENUM:
        enum = (x < 0)[None, :]
        stored = tl.load(masses + tl.maximum(x, 0)[None, :], mask = lm & ~enum, other = float("-inf"))
        e = tl.maximum(-1 - x, 0).to(tl.int64)
        g = tl.load(eidx + tk + e).to(tl.int64)
        ev = tl.log(tl.load(eprobs + pr * num_enum + g[None, :], mask = lm & enum, other = 0.0))
        if LOGW:
            ev = ev + tl.load(enum_logw + b * logw_stride + tk + e, mask = pm & (x < 0), other = 0.0)[None, :] - lz
        return tl.where(enum, ev, stored)
    return tl.load(masses + x[None, :], mask = lm, other = float("-inf"))


@triton.jit
def _block_value(rp, iv_table, rpos, q, j, X, pm, r_own, crank, b, t, rbase, ce, n, W, R_FULL: tl.constexpr,
                 COMPACT: tl.constexpr):
    """[lanes]: the block's value (its rows at ``rp``) at successor column ``q`` of boundary ``t + 1`` and the lane's
    exit column ``j`` (see :func:`_operand_slot`), -inf where ``pm`` is off."""
    sl = _operand_slot(iv_table, rpos, q, j, X, pm, r_own, crank, b, t + 1, q, rbase, ce, n, W, R_FULL, COMPACT)
    return tl.load(rp + sl, mask = pm & (sl >= 0), other = float("-inf"))


@triton_jit
def _input_block_kernel(element_mars, node_mars, temp, obs_class, next_col, delta, state_id, col_of_state, class_mars, mass_row, mass_stride, trans,
                        mbase, mcount, mrank, bbase, mb_stride, mrank_stride, entry_ptr, entry_succ, entry_x, width,
                        iv_of, iv_info, iv_table, live, coff,
                        ccount, cstart, ccolumns, crank, steps, sel, reg_offset, reg_width, reg_first, prior_mars,
                        prior_offset, prior_width, reg_prior, code, tok_ptr, eidx, eprobs, prow, lognorm, enum_logw, Cs,
                        num_enum, logw_stride, out_first, out_width, temp_width, B, n, W, C, K, HOMOGENEOUS: tl.constexpr, input_start, csteps, pid0,
                        succ_start, succ_count, succ_use, succ_entries, E, R_KIND: tl.constexpr,
                        OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr, PRUNE: tl.constexpr, OUT_FULL: tl.constexpr,
                        R_FULL: tl.constexpr, COMPACT: tl.constexpr, INDEXED: tl.constexpr, SAMPLE: tl.constexpr,
                        ENUM: tl.constexpr, LOGW: tl.constexpr, ENTRIES: tl.constexpr, SUCC_LISTS: tl.constexpr,
                        SAMPLE_MASSES: tl.constexpr, TILE: tl.constexpr, hold_cum = None, HOLD: tl.constexpr = False,
                        PDL: tl.constexpr = False):
    """One program per step (of the plan's index list ``sel`` if ``INDEXED``) and tile of its whole output, flattened
    (sample, slot): with the samples' blocks contiguous, a step's output is one run of columns, and the lanes stay
    full whatever the widths. ``COMPACT``: the per-sample values compact (:mod:`.compact`), a lane per pair of the
    sample's live columns; the block's, kept once, by slot. ``SAMPLE``: class masses of each sample's own (see
    :class:`~..state.ClassMasses`; the transition masses as :meth:`~..compiled.program.Program.mass_args` lays them
    out, ``SAMPLE_MASSES`` each sample's own over its entries). A missing token's terms are the
    column's entries (:class:`~.transitions.TransitionTables`): grouped, a transition mass per successor; else a mass
    per live class, in class order, read by the class's code (:func:`_lane_mass`; ``ENUM``: enumerated classes from
    their tokens; ``Cs`` the stored classes, the table's width). ``SUCC_LISTS`` (2: every step, 1: a step whose
    boundary takes them, ``succ_use``): the lanes walk their column's entries whose successor is live alone
    (:class:`~.live.SuccessorLists`, ``E`` the tables' entries; a term through a dead successor is -inf, a no-op: the
    same bits)."""
    if PDL:                                             # (the launch before it, done)
        gdc_wait()
    s = _child(sel, tl.program_id(0), INDEXED)
    tile = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 7 + 0)
    u = tl.load(steps + s * 7 + 1).to(tl.int64)
    t = tl.load(steps + s * 7 + 2)
    t2 = tl.load(steps + s * 7 + 5)
    X = _exit_width(width, t2, n)
    S, opos, pairs = _interval(iv_of, iv_info, t, t2, n)
    ko, ncols = _out_count(iv_of, coff, t, t2, B, S, n, COMPACT)
    if (tile * TILE < ncols) & _fresh_tile(hold_cum, coff, ko, tile, TILE, ncols, B, csteps, HOLD):
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < ncols
        b, i, j, ml, pa, pe, ce = _out_lanes(idx, m, iv_table, opos, pairs, live, coff, ccount, cstart, ccolumns, ko,
                                            t, t2, S, X, B, n, W, csteps, PRUNE, OUT_FULL, COMPACT)
        end = t + 1 == n
        if R_KIND != 2:
            RS, rpos, _ = _interval(iv_of, iv_info, t + 1, t2, n)            # the block's slots
            rp, r_own = _block_ptr(tl.load(steps + s * 7 + 3), tl.load(steps + s * 7 + 4), node_mars, prior_mars, temp,
                                   reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                                   R_KIND)
            if COMPACT:                                                      # pair (q, j): the lane rbase + q's place
                kr = tl.load(iv_of + (t + 1) * (n + 1) + t2).to(tl.int64)    # x ce (the exit is the output's)
                rbase = tl.load(coff + kr * (B + 1) + b, mask = m, other = 0) + pe
            else:
                rp = rp + tl.where(r_own, b * RS, 0)
                rbase = 0
        else:                                                                # (the identity: no block, never read)
            rp, r_own, rpos, rbase = node_mars, m, opos, 0
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)            # observed token's class, -1 if missing

        # an observed token: its own log-probability (the input region of node_mars), then the block from the
        # column its class leads to
        obs = ml & (oc >= 0)
        q = _next_col(next_col, delta, state_id, col_of_state, t, i, oc, obs, K, W, C, HOMOGENEOUS)
        if R_KIND == 2:
            rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
        else:
            sl = _operand_slot(iv_table, rpos, q, j, X, obs, r_own, crank, b, t + 1, q, rbase, ce, n, W, R_FULL,
                               COMPACT)
            rv = tl.load(rp + sl, mask = obs & (sl >= 0), other = float("-inf"))
        val = tl.load(node_mars + u * B + b, mask = obs, other = float("-inf")) + rv

        # a missing token: log-sum over i's transitions of their mass + the block from their successor (skipped when
        # every lane is observed)
        miss = ml & (oc < 0)
        if tl.max(tl.where(miss, 1, 0), axis = 0) > 0:
            mx = tl.full([TILE], float("-inf"), dtype = tl.float32)
            acc = tl.zeros([TILE], dtype = tl.float32)
            if GROUPED:                                                      # the step's transition masses: its rank's
                mt = b.to(tl.int64) * mb_stride + t                          # block of the boundary's pairs, or of the
                # (masked: a lane past the tile's outputs has no sample, its `b` anywhere -- unmasked, these 8-byte
                # loads read past the tables and, on an unmapped page, faulted the context)
                masses = trans + tl.load(mbase + mt, mask = miss, other = 0) + \
                    tl.load(steps + s * 7 + 6).to(tl.int64) * tl.load(mcount + mt, mask = miss, other = 0).to(tl.int64)
                bbt = tl.load(bbase + t)                                     # entries there (see Program.mass_args)
                tk, pr, lz = 0, 0, 0.0                                       # (no class masses read)
            else:                                                            # the input's row of the class masses
                mrow = tl.load(mass_row + (u - input_start)).to(tl.int64) + b * 0     # (the lane's sample's)
                if SAMPLE:
                    mrow = mrow + b * tl.load(mass_stride + (u - input_start)).to(tl.int64)
                masses = class_mars + mrow * Cs
                bbt = 0
                tk, pr, lz = _lane_rows(tok_ptr, prow, lognorm, t, mrow, miss, ENUM, LOGW)   # (once, not per term)
            if GROUPED or ENTRIES:                                           # i's entries: a successor, and a
                if SUCC_LISTS == 2:                                          # (the boundary's successor lists)
                    listed = True
                elif SUCC_LISTS == 1:
                    listed = tl.load(succ_use + t) != 0
                else:
                    listed = False
                if listed:
                    l0, l1 = _list_range(succ_start, succ_count, b, t, i, miss, n, W, E)
                    for kk in range(0, tl.max(l1 - l0, axis = 0)):                  # (kk: not the int32 k below)
                        pm = miss & (l0 + kk < l1)
                        ee = tl.load(succ_entries + l0 + kk, mask = pm, other = 0)
                        mx, acc = _ib_term(mx, acc, ee, pm, entry_succ, entry_x, rp, iv_table, rpos, j, X, r_own,
                                           crank, b, t, rbase, ce, n, W, masses, mrank, mrank_stride, bbt, eprobs, tk,
                                           pr, lz, enum_logw, eidx, num_enum, logw_stride, R_KIND, R_FULL, COMPACT,
                                           GROUPED, SAMPLE_MASSES, ENUM, LOGW)
                else:                                                        # (or the entry tables)
                    p0 = tl.load(entry_ptr + t * (W + 1) + i, mask = miss, other = 0)   # transition mass or a class
                    p1 = tl.load(entry_ptr + t * (W + 1) + i + 1, mask = miss, other = 0)   # code
                    for k in range(0, tl.max(p1 - p0, axis = 0)):
                        pm = miss & (p0 + k < p1)
                        mx, acc = _ib_term(mx, acc, p0 + k, pm, entry_succ, entry_x, rp, iv_table, rpos, j, X, r_own,
                                           crank, b, t, rbase, ce, n, W, masses, mrank, mrank_stride, bbt, eprobs, tk,
                                           pr, lz, enum_logw, eidx, num_enum, logw_stride, R_KIND, R_FULL, COMPACT,
                                           GROUPED, SAMPLE_MASSES, ENUM, LOGW)
            else:                                                            # every class in class order, the dead
                for c in range(C):                                           # ones -inf (no entry tables read)
                    q = _next_col(next_col, delta, state_id, col_of_state, t, i, c, miss, K, W, C, HOMOGENEOUS)
                    q = tl.where(end & (q >= 0), 0, q)                       # (a sequence end: one column, as the
                    pm = miss & (q >= 0)                                     # entry tables have it)
                    if R_KIND == 2:
                        rv = tl.where(pm & (q == j), 0.0, float("-inf"))
                    else:
                        rv = _block_value(rp, iv_table, rpos, q, j, X, pm, r_own, crank, b, t, rbase, ce, n, W,
                                          R_FULL, COMPACT)
                    if ENUM:
                        x = tl.load(code + t * C + c + i * 0)                # the class's code at the boundary
                    else:
                        x = c + i * 0
                    mass = _lane_mass(masses, x, pm, eprobs, tk, pr, lz, enum_logw, eidx, b, num_enum, logw_stride,
                                      ENUM, LOGW)
                    mx, acc = _lse_step(mx, acc, mass + rv)
            val = tl.where(oc >= 0, val, _lse_value(mx, acc))
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + idx, val, mask = m)


@triton_jit
def _input_block_group_kernel(element_mars, node_mars, temp, obs_class, next_col, delta, state_id, col_of_state, class_mars, mass_row, mass_stride,
                              entry_ptr, entry_succ, entry_x, width, iv_of, iv_info, iv_table, live, coff, ccount,
                              cstart, ccolumns, crank, steps, groups, sel, reg_offset, reg_width, reg_first,
                              prior_mars, prior_offset, prior_width, reg_prior, code, tok_ptr, eidx, eprobs, prow,
                              lognorm, enum_logw, Cs, num_enum, logw_stride, out_first, out_width, temp_width, B, n, W,
                              C, K, HOMOGENEOUS: tl.constexpr, input_start, csteps, pid0, succ_start, succ_count, succ_use, succ_entries, E,
                              R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, PRUNE: tl.constexpr,
                              OUT_FULL: tl.constexpr, R_FULL: tl.constexpr, G: tl.constexpr, ONE_EXP: tl.constexpr,
                              COMPACT: tl.constexpr, INDEXED: tl.constexpr, SAMPLE: tl.constexpr, ENUM: tl.constexpr,
                              LOGW: tl.constexpr, ENTRIES: tl.constexpr, SUCC_LISTS: tl.constexpr,
                              TILE: tl.constexpr, hold_cum = None, HOLD: tl.constexpr = False,
                              PDL: tl.constexpr = False):
    """:func:`_input_block_kernel` for a group of steps per program, ungrouped transitions (a column's live entries,
    ``ENTRIES``, or every class, in class order), the block a sum's or a scratch row: group ``g`` is steps ``groups[g] .. groups[g +
    1]`` (at most ``G``, a power of 2), which share their position and block interval, so they share their lanes'
    columns, successors and slots. Those are read once for the ``[G, TILE]`` lanes; only the input's masses (by the
    class's code, ``ENUM``: an enumerated class's from its token) and the block's values are read per step.
    ``SUCC_LISTS`` (``ENTRIES``): as :func:`_input_block_kernel`'s.
    ``ONE_EXP``: the log-sum-exp by :func:`_lse_step_one_exp` (bit-identical). ``COMPACT``: as
    :func:`_input_block_kernel`, the group's blocks all kept once or none (their interval is shared). ``INDEXED``: the
    groups of the plan's index list ``sel``. ``SAMPLE``: as :func:`_input_block_kernel`."""
    if PDL:                                             # (the launch before it, done)
        gdc_wait()
    g = _child(sel, tl.program_id(0), INDEXED)
    first = tl.load(groups + g)
    gs = first + tl.arange(0, G)
    gm = tl.arange(0, G) < tl.load(groups + g + 1) - first                         # the group's steps
    tile = pid0 + tl.program_id(1)
    t = tl.load(steps + first * 7 + 2)
    t2 = tl.load(steps + first * 7 + 5)
    X = _exit_width(width, t2, n)
    S, opos, pairs = _interval(iv_of, iv_info, t, t2, n)
    ko, ncols = _out_count(iv_of, coff, t, t2, B, S, n, COMPACT)
    if (tile * TILE < ncols) & _fresh_tile(hold_cum, coff, ko, tile, TILE, ncols, B, csteps, HOLD):
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < ncols
        b, i, j, ml, pa, pe, ce = _out_lanes(idx, m, iv_table, opos, pairs, live, coff, ccount, cstart, ccolumns, ko,
                                            t, t2, S, X, B, n, W, csteps, PRUNE, OUT_FULL, COMPACT)
        RS, rpos, _ = _interval(iv_of, iv_info, t + 1, t2, n)                # the blocks' slots
        u = tl.load(steps + gs * 7 + 1, mask = gm, other = input_start).to(tl.int64)
        rp, r_own = _block_ptr(tl.load(steps + gs * 7 + 3, mask = gm, other = 0),
                               tl.load(steps + gs * 7 + 4, mask = gm, other = 0), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               R_KIND)
        if COMPACT:                                          # as in _input_block_kernel, the first step's for the group
            _rp, own = _block_ptr(tl.load(steps + first * 7 + 3), tl.load(steps + first * 7 + 4), node_mars, prior_mars,
                                temp, reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior,
                                temp_width, R_KIND)
            kr = tl.load(iv_of + (t + 1) * (n + 1) + t2).to(tl.int64)
            rbase = tl.load(coff + kr * (B + 1) + b, mask = m, other = 0) + pe
            rp = rp[:, None]
        else:
            own = True
            rbase = 0
            rp = rp[:, None] + tl.where(r_own[:, None], (b * RS)[None, :], 0)   # [G, TILE]
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)            # observed token's class, -1 if missing

        # an observed token, as in _input_block_kernel
        obs = ml & (oc >= 0)
        q = _next_col(next_col, delta, state_id, col_of_state, t, i, oc, obs, K, W, C, HOMOGENEOUS)
        sl = _operand_slot(iv_table, rpos, q, j, X, obs, own, crank, b, t + 1, q, rbase, ce, n, W, R_FULL, COMPACT)
        val = tl.load(node_mars + u[:, None] * B + b[None, :], mask = gm[:, None] & obs[None, :], other = float("-inf")) + \
            tl.load(rp + sl[None, :], mask = gm[:, None] & (obs & (sl >= 0))[None, :], other = float("-inf"))

        # a missing token, over the column's live entries or every class: each class's successor and slot once, its
        # mass and block value per step
        miss = ml & (oc < 0)
        if tl.max(tl.where(miss, 1, 0), axis = 0) > 0:
            mx = tl.full([G, TILE], float("-inf"), dtype = tl.float32)
            acc = tl.zeros([G, TILE], dtype = tl.float32)
            mrow = tl.load(mass_row + (u - input_start)).to(tl.int64)[:, None] + (b * 0)[None, :]   # [G, TILE]: the
            if SAMPLE:                                                                           # lanes' samples'
                mrow = mrow + b[None, :] * tl.load(mass_stride + (u - input_start)).to(tl.int64)[:, None]
            masses = class_mars + mrow * Cs
            gmiss = gm[:, None] & miss[None, :]
            if ENUM:                                                         # (see _group_mass)
                tk = tl.load(tok_ptr + t)
                pr = tl.load(prow + mrow, mask = gmiss, other = 0).to(tl.int64)
                if LOGW:
                    lz = tl.load(lognorm + mrow, mask = gmiss, other = 0.0)
                else:
                    lz = pr
            else:
                tk = 0
                pr = mrow
                lz = mrow
            if ENTRIES:                                                      # i's live entries, in class order
                if SUCC_LISTS == 2:                                          # (or those whose successor is live)
                    listed = True
                elif SUCC_LISTS == 1:
                    listed = tl.load(succ_use + t) != 0
                else:
                    listed = False
                if listed:
                    l0, l1 = _list_range(succ_start, succ_count, b, t, i, miss, n, W, E)
                    for kk in range(0, tl.max(l1 - l0, axis = 0)):                  # (kk: not the int32 k below)
                        pm = miss & (l0 + kk < l1)
                        lm = gm[:, None] & pm[None, :]
                        ee = tl.load(succ_entries + l0 + kk, mask = pm, other = 0)
                        mx, acc = _ibg_term(mx, acc, ee, pm, lm, gm, entry_succ, entry_x, rp, iv_table, rpos, j, X, own,
                                            crank, b, t, rbase, ce, n, W, masses, eprobs, tk, pr, lz, enum_logw, eidx,
                                            num_enum, logw_stride, R_FULL, COMPACT, ENUM, LOGW, ONE_EXP)
                else:
                    p0 = tl.load(entry_ptr + t * (W + 1) + i, mask = miss, other = 0)
                    p1 = tl.load(entry_ptr + t * (W + 1) + i + 1, mask = miss, other = 0)
                    for k in range(0, tl.max(p1 - p0, axis = 0)):
                        pm = miss & (p0 + k < p1)
                        lm = gm[:, None] & pm[None, :]
                        mx, acc = _ibg_term(mx, acc, p0 + k, pm, lm, gm, entry_succ, entry_x, rp, iv_table, rpos, j, X, own,
                                            crank, b, t, rbase, ce, n, W, masses, eprobs, tk, pr, lz, enum_logw, eidx,
                                            num_enum, logw_stride, R_FULL, COMPACT, ENUM, LOGW, ONE_EXP)
            else:                                                            # every class in class order, the dead
                for c in range(C):                                           # ones -inf (no entry tables read)
                    q = _next_col(next_col, delta, state_id, col_of_state, t, i, c, miss, K, W, C, HOMOGENEOUS)
                    q = tl.where((t + 1 == n) & (q >= 0), 0, q)              # (a sequence end: one column, as the
                    pm = miss & (q >= 0)                                     # entry tables have it)
                    if ENUM:
                        x = tl.load(code + t * C + c + i * 0)                # the class's code at the boundary
                    else:
                        x = c + i * 0
                    sl = _operand_slot(iv_table, rpos, q, j, X, pm, own, crank, b, t + 1, q, rbase, ce, n, W, R_FULL,
                                       COMPACT)
                    rv = tl.load(rp + sl[None, :], mask = gm[:, None] & (pm & (sl >= 0))[None, :],
                                 other = float("-inf"))
                    mass = _group_mass(masses, x, gm[:, None] & pm[None, :], pm, tk, pr, lz, eidx, eprobs, enum_logw, b,
                                       logw_stride, num_enum, ENUM, LOGW)
                    if ONE_EXP:
                        mx, acc = _lse_step_one_exp(mx, acc, mass + rv)
                    else:
                        mx, acc = _lse_step(mx, acc, mass + rv)
            val = tl.where((oc >= 0)[None, :], val, _lse_value(mx, acc))
        op = _out_ptr(tl.load(steps + gs * 7 + 0, mask = gm, other = 0), element_mars, temp, out_first, out_width,
                      temp_width, OUT_TEMP)
        tl.store(op[:, None] + idx[None, :], val, mask = gm[:, None] & m[None, :])


@triton_jit
def _block_input_kernel(element_mars, node_mars, temp, obs_class, next_col, delta, state_id, col_of_state, class_mars, mass_row, mass_stride, trans,
                        mbase, mcount, mrank, bbase, mb_stride, mrank_stride, pred_ptr, pred_col, pred_x, width, iv_of,
                        iv_info, iv_table, live, coff, ccount,
                        cstart, ccolumns, crank, steps, sel, reg_offset, reg_width, reg_first, prior_mars,
                        prior_offset, prior_width, reg_prior, code, tok_ptr, eidx, eprobs, prow, lognorm, enum_logw, Cs,
                        num_enum, logw_stride, out_first, out_width, temp_width, B, n, W, C, K, HOMOGENEOUS: tl.constexpr, input_start, csteps, pid0,
                        L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, GROUPED: tl.constexpr, PRUNE: tl.constexpr,
                        OUT_FULL: tl.constexpr, L_FULL: tl.constexpr, COMPACT: tl.constexpr, INDEXED: tl.constexpr,
                        SAMPLE: tl.constexpr, ENUM: tl.constexpr, LOGW: tl.constexpr, SAMPLE_MASSES: tl.constexpr,
                        TILE: tl.constexpr, hold_cum = None, HOLD: tl.constexpr = False, PDL: tl.constexpr = False):
    """One program per step and tile of its whole output, flattened (sample, slot) as in
    :func:`_input_block_kernel`: every lane takes a log-sum over its exit column's predecessor entries (the
    program loops to the longest list among its lanes), each through its mass for a missing token (ungrouped: by
    the class's code, :func:`_lane_mass`), or the observed token's own log-probability when the entry holds its
    class. ``COMPACT``, ``INDEXED``, ``SAMPLE``, ``ENUM``: as :func:`_input_block_kernel`."""
    if PDL:                                             # (the launch before it, done)
        gdc_wait()
    s = _child(sel, tl.program_id(0), INDEXED)
    tile = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 7 + 0)
    t0 = tl.load(steps + s * 7 + 3)
    u = tl.load(steps + s * 7 + 4).to(tl.int64)
    t = tl.load(steps + s * 7 + 5)
    M = tl.load(width + t)
    X = _exit_width(width, t + 1, n)
    S, opos, pairs = _interval(iv_of, iv_info, t0, t + 1, n)
    LS, lpos, _ = _interval(iv_of, iv_info, t0, t, n)                       # the block's slots
    ko, ncols = _out_count(iv_of, coff, t0, t + 1, B, S, n, COMPACT)
    if (tile * TILE < ncols) & _fresh_tile(hold_cum, coff, ko, tile, TILE, ncols, B, csteps, HOLD):
        idx = tile * TILE + tl.arange(0, TILE)
        m = idx < ncols
        b, i, j, ml, pa, pe, ce = _out_lanes(idx, m, iv_table, opos, pairs, live, coff, ccount, cstart, ccolumns, ko,
                                           t0, t + 1, S, X, B, n, W, csteps, PRUNE, OUT_FULL, COMPACT)   # j: at t + 1
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)
        own = tl.load(node_mars + u * B + b, mask = m, other = float("-inf"))  # the input region of node_mars
        lp, l_own = _block_ptr(tl.load(steps + s * 7 + 1), tl.load(steps + s * 7 + 2), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, L_KIND)
        if COMPACT:                                                          # pair (i, q): the lane lbase + q's place
            kl = tl.load(iv_of + t0 * (n + 1) + t).to(tl.int64)              # (the entry is the output's)
            lbase = tl.load(coff + kl * (B + 1) + b, mask = m, other = 0) + \
                pa * tl.load(ccount + b * (n + 1) + t, mask = m, other = 0)
        else:
            lp = lp + tl.where(l_own, b * LS, 0)
            lbase = 0
        if GROUPED:                                                          # the step's transition masses (as in
            mt = b.to(tl.int64) * mb_stride + t                              # _input_block_kernel; masked as there)
            masses = trans + tl.load(mbase + mt, mask = m, other = 0) + \
                tl.load(steps + s * 7 + 6).to(tl.int64) * tl.load(mcount + mt, mask = m, other = 0).to(tl.int64)
            bbt = tl.load(bbase + t)
        else:                                                                # the input's row of the class masses
            mrow = tl.load(mass_row + (u - input_start)).to(tl.int64) + b * 0
            if SAMPLE:
                mrow = mrow + b * tl.load(mass_stride + (u - input_start)).to(tl.int64)
            masses = class_mars + mrow * Cs
            bbt = 0
            tk, pr, lz = _lane_rows(tok_ptr, prow, lognorm, t, mrow, ml & (oc < 0), ENUM, LOGW)   # (once a lane)
            if ENUM:                                                         # the observed class's code
                code_oc = tl.load(code + t * C + tl.maximum(oc, 0), mask = m, other = -1)
            else:
                code_oc = oc
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
                nq = _next_col(next_col, delta, state_id, col_of_state, t, q, tl.maximum(oc, 0), em, K, W, C,
                               HOMOGENEOUS)
                holds = (nq == j) | (end & (nq >= 0))
                if SAMPLE_MASSES:                                            # the sample's entry, or none: no mass
                    x = tl.load(mrank + b.to(tl.int64) * mrank_stride + bbt + x, mask = em, other = -1)
                mass = tl.load(masses + x, mask = em & (oc < 0) & (x >= 0), other = float("-inf"))
            else:                                                            # is it this entry's class?
                holds = x == code_oc
                mass = _lane_mass(masses, x, em & (oc < 0), eprobs, tk, pr, lz, enum_logw, eidx, b, num_enum,
                                  logw_stride, ENUM, LOGW)
            mass = tl.where(oc >= 0, tl.where(holds, own, float("-inf")), mass)
            sl = _operand_slot(iv_table, lpos, i, q, M, em, l_own, crank, b, t, q, lbase, 1, n, W, L_FULL, COMPACT)
            lv = tl.load(lp + sl, mask = em & (sl >= 0), other = float("-inf"))
            mx, acc = _lse_step_one_exp(mx, acc, lv + mass)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + idx, _lse_value(mx, acc), mask = m)
