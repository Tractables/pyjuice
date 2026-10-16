"""
The lifted product layer. A product chains its children's blocks in scope order: a log-space product over the
boundary columns that adjacent children share. There is one table-driven implementation for every circuit
shape. :class:`~pyjuice.constraints.backends.dfa_exact.forward.Program` folds each product into binary steps,
right to left. Each step is one of three contractions, chosen by what its two operands are, never by the
circuit's shape:

* ``input @ block`` (:func:`_input_block_kernel`): an input at position ``t``, then a block from ``t + 1``.
  Class ``c`` of the token moves column ``i`` at ``t`` to ``next_col[t, i, c]``, so this is a gather over
  the successors of ``i``. The block may be the identity, which materializes the input's own block (a product
  with a single input child, or an input next to another input);
* ``block @ input`` (:func:`_block_input_kernel`): the mirror case, a gather over the predecessors of each
  column at ``t + 1``;
* ``block @ block`` (:func:`_block_block_kernel`): a per-sample log-space matrix product over the shared
  boundary, skipping the (output tile, shared chunk) pairs the automaton cannot join (:class:`SkipMasks`) -- or, the
  values compact (:mod:`.compact`), over the sample's live columns alone (:func:`_block_block_compact_kernel`).

Blocks are either sum nodes' rows in ``node_mars`` or intermediate results in a scratch buffer. Outputs go
to the product layer's ``element_mars`` rows, or to scratch when another step reads them. A row holds one
block per sample, sample-major: sample ``b``'s block in columns ``[b * slots, (b + 1) * slots)``. A sum node whose
scope no sample observes may keep one block for every sample instead, in ``prior_mars`` (the query plan says which,
see :mod:`.plans`). A block keeps
only the (entry column ``i``, exit column ``j``) pairs the automaton joins, in row-major order: slot ``i * X +
j`` when it joins them all, else through its interval's tables (:class:`~.layout.IntervalTables`). An input child
is read only as the log-probability of its observed token (pyjuice's own input layers wrote it) or, when the
token is missing, through its transition masses (:func:`transition_masses`): for a column and one of its
successors, the log-mass of the token classes that lead there -- so a missing token costs a gather per
successor, not per class. Under evidence pruning (``live``, see :mod:`.live`), a dead slot is -inf and costs no
work.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from pyjuice.layer.kernels import autotune
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ..compiled.layout import Reachability, _tile_any
from .common import _block_ptr, _child, _exit_width, _interval, _lse_step, _pair, _slot, _tf32_operand, grid_chunks
from .compact import _lane_pairs, _operand_slot
from .transitions import transition_masses


#: fields of a step, per contraction (int32 rows of the step tables the Program builds)
INPUT_BLOCK_FIELDS = ("out", "u", "t", "r_row", "r_reg", "t2", "t_off")    # input u at t, then block over [t+1, t2)
BLOCK_INPUT_FIELDS = ("out", "l_row", "l_reg", "t0", "u", "t", "t_off")    # block over [t0, t), then input u at t
# (``t_off``: where the step's transition masses start in the transition buffer)
BLOCK_BLOCK_FIELDS = ("out", "l_row", "l_reg", "r_row", "r_reg", "t0", "t1", "t2")
COPY_FIELDS = ("out", "l_row", "l_reg", "t0", "t2")


@dataclass
class Launch:
    """
    One launch of a product layer's stage, as :func:`run_products` runs it, and what a query plan needs to run a
    subset of its steps (:func:`~.plans.query_plan`).

    :ivar form: the contraction: ``input_block``, ``block_input``, ``block_block`` or ``copy``
    :ivar kinds: its operands' kinds (see :class:`~.program.Program`)
    :ivar steps: its device tables: ``input_block`` and ``block_input``, ``(steps, chunks, groups)``; ``block_block``,
        ``(steps, triple index)``; ``copy``, ``steps`` (int32 rows, see the ``*_FIELDS`` above)
    :ivar E_max, X_max, S_max: the most entry columns, exit columns and slots of its steps' outputs
    :ivar rows: [S, fields] int64, its steps on the host, in the device tables' order
    :ivar iv: [S] int64, the interval of every step's product, as its number in ``Program.sum_ivs`` (a scratch step
        follows the product it is a fold of)
    :ivar sizes: [S, 3] int64, every step's output's entry columns, exit columns and slots
    :ivar out_ivs: int64 (host): the intervals of its steps' outputs, each once, by ``prog.intervals``' numbers (for
        compact storage, see :mod:`.compact`)
    """

    form: str
    kinds: tuple
    steps: Any
    E_max: int
    X_max: int
    S_max: int
    rows: torch.Tensor
    iv: torch.Tensor
    sizes: torch.Tensor
    groups_host: Optional[dict] = None
    out_ivs: Any = None

    @property
    def out_temp(self) -> bool:
        """Whether its steps' outputs are scratch rows (a fold's), not the layer's ``element_mars`` rows."""
        return self.kinds[2 if self.form == "block_block" else 1]


@dataclass
class Cut:
    """
    The steps of a :class:`Launch` a query plan runs: per chunk of the launch (``chunks``: ``(first, end, job, index
    list or None for every step, count)``, see :func:`run_products`; the index list relative to the chunk's first); for
    ``input_block`` steps run in groups, per group size ``(index list or None, count)`` of its groups (a group's steps
    share their interval); the selected steps' most entry columns, exit columns and slots, and how many there are.
    """

    chunks: list
    groups: Optional[dict]
    E_max: int
    X_max: int
    S_max: int
    count: int


@triton.jit
def _out_ptr(row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP: tl.constexpr):
    if OUT_TEMP:
        return temp + row.to(tl.int64) * temp_width
    else:
        return element_mars + (row - out_first).to(tl.int64) * out_width


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
def _out_count(iv_of, coff, a, e, B, S, n, COMPACT: tl.constexpr):
    """``(k, lanes)`` of a step's output over ``[a, e)``: its interval's number and its row's lanes -- ``B * S``, or
    ``COMPACT`` the interval's (see :mod:`.compact`)."""
    k = tl.load(iv_of + a * (n + 1) + e)
    if COMPACT:
        return k, tl.load(coff + k.to(tl.int64) * (B + 1) + B).to(tl.int32)
    else:
        return k, B * S


@triton.jit
def _out_lanes(idx, m, iv_table, opos, pairs, live, coff, ccount, cstart, ccolumns, k, a, e, S, X, B, n, W, csteps,
               PRUNE: tl.constexpr, OUT_FULL: tl.constexpr, COMPACT: tl.constexpr):
    """``(b, i, j, live, pa, pe, ce)`` of the lanes ``idx`` (``m``: the real ones) of a step's output over ``[a, e)``:
    their sample and pair, which are live (the others -inf), and -- ``COMPACT`` -- the pair's places among the
    sample's live columns at the entry and the exit (``ce``: those at the exit). Dense, the samples' blocks of ``S``
    slots one after the other, live where both columns are (``PRUNE``); ``COMPACT``, the lanes of a row of the
    output's interval ``k`` (:mod:`.compact`), every one live but at a pair the automaton does not join."""
    if COMPACT:
        b, i, j, pa, pe, ce = _lane_pairs(coff, ccount, cstart, ccolumns, k, a, e, idx, m, B, n, csteps)
        return b.to(tl.int64), i, j, m & (_slot(iv_table, opos, i, j, X, m, OUT_FULL) >= 0), pa, pe, ce
    else:
        b = (idx // S).to(tl.int64)
        v = _pair(iv_table, pairs, idx % S, m, OUT_FULL)
        i = v // X
        j = v % X
        return b, i, j, _live_lanes(live, m, b, a, i, e, j, n, W, PRUNE), i, j, X


@triton.jit
def _lse_step_one_exp(m, acc, v):
    """:func:`_lse_step` with one exp, bit for bit: a new maximum rescales the sum by ``exp(m - v)`` and adds 1, any
    other term adds ``exp(v - m)`` -- both ``exp(-|v - m|)``, the float the two-exp form computes for the term it does
    not round to 1 (a ``-inf`` term adds nothing, and the first finite one makes the sum 1). Faster in
    :func:`_block_input_kernel` (x1.10) and in the class weights (:func:`~.score._class_weight_kernel`, 5-9% on every
    call of a diffusion decode); in the ``input @ block`` kernels it won with most lanes live and lost under evidence
    pruning, so they keep :func:`_lse_step`."""
    e = tl.exp(-tl.abs(v - m))
    acc = tl.where(v == float("-inf"), acc, tl.where(v > m, acc * e + 1.0, acc + e))
    return tl.maximum(m, v), acc


@triton.jit
def _lse_value(m, acc):
    return tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)


@triton_jit
def _input_block_kernel(element_mars, node_mars, temp, obs_class, next_col, class_mars, mass_row, mass_stride, trans,
                        trans_stride, entry_ptr, entry_succ, entry_x, width, iv_of, iv_info, iv_table, live, coff,
                        ccount, cstart, ccolumns, crank, steps, sel, reg_offset, reg_width, reg_first, prior_mars,
                        prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, B, n, W, C,
                        input_start, csteps, pid0, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                        GROUPED: tl.constexpr, PRUNE: tl.constexpr, OUT_FULL: tl.constexpr, R_FULL: tl.constexpr,
                        COMPACT: tl.constexpr, INDEXED: tl.constexpr, SAMPLE: tl.constexpr, TILE: tl.constexpr):
    """One program per step (of the plan's index list ``sel`` if ``INDEXED``) and tile of its whole output, flattened
    (sample, slot): with the samples' blocks contiguous, a step's output is one run of columns, and the lanes stay
    full whatever the widths. ``COMPACT``: the per-sample values compact (:mod:`.compact`), a lane per pair of the
    sample's live columns; the block's, kept once, by slot. ``SAMPLE``: class masses of each sample's own (see
    :class:`~..state.ClassMasses`; transition masses ``trans_stride`` floats apart)."""
    s = _child(sel, tl.program_id(0), INDEXED)
    tile = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 7 + 0)
    u = tl.load(steps + s * 7 + 1).to(tl.int64)
    t = tl.load(steps + s * 7 + 2)
    t2 = tl.load(steps + s * 7 + 5)
    X = _exit_width(width, t2, n)
    S, opos, pairs = _interval(iv_of, iv_info, t, t2, n)
    ko, ncols = _out_count(iv_of, coff, t, t2, B, S, n, COMPACT)
    if tile * TILE < ncols:
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
        oc = tl.load(obs_class + b * n + t, mask = m, other = -1)            # observed token's class, -1 if missing

        # an observed token: its own log-probability (the input region of node_mars), then the block from the
        # column its class leads to
        obs = ml & (oc >= 0)
        q = tl.load(next_col + (t * W + i) * C + oc, mask = obs, other = -1)
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
            if GROUPED:                                                      # i's entries: successor, transition mass
                p0 = tl.load(entry_ptr + t * (W + 1) + i, mask = miss, other = 0)
                p1 = tl.load(entry_ptr + t * (W + 1) + i + 1, mask = miss, other = 0)
                masses = trans + tl.load(steps + s * 7 + 6).to(tl.int64)      # the step's transition masses
                if SAMPLE:                                                   # (the lane's sample's)
                    masses = masses + tl.where(tl.load(mass_stride + (u - input_start)) > 0, b * trans_stride, 0)
                for k in range(0, tl.max(p1 - p0, axis = 0)):
                    pm = miss & (p0 + k < p1)
                    q = tl.load(entry_succ + p0 + k, mask = pm, other = -1)
                    if R_KIND == 2:
                        rv = tl.where(pm & (q == j), 0.0, float("-inf"))
                    else:
                        sl = _operand_slot(iv_table, rpos, q, j, X, pm, r_own, crank, b, t + 1, q, rbase, ce, n, W,
                                           R_FULL, COMPACT)
                        rv = tl.load(rp + sl, mask = pm & (sl >= 0), other = float("-inf"))
                    mass = tl.load(masses + tl.load(entry_x + p0 + k, mask = pm, other = 0), mask = pm,
                                   other = float("-inf"))
                    mx, acc = _lse_step(mx, acc, mass + rv)
            else:                                                            # every class, in order: the same class
                masses = class_mars + tl.load(mass_row + (u - input_start)).to(tl.int64) * C    # (and mass) in
                if SAMPLE:                                                   # every lane; the lane's sample's
                    masses = masses + b * tl.load(mass_stride + (u - input_start)).to(tl.int64) * C
                for c in range(C):
                    q = tl.load(next_col + (t * W + i) * C + c, mask = miss, other = -1)
                    if R_KIND == 2:
                        rv = tl.where((q == j) | (end & (q >= 0)), 0.0, float("-inf"))
                    else:
                        sl = _operand_slot(iv_table, rpos, q, j, X, miss, r_own, crank, b, t + 1, q, rbase, ce, n, W,
                                           R_FULL, COMPACT)
                        rv = tl.load(rp + sl, mask = miss & (sl >= 0), other = float("-inf"))
                    if SAMPLE:
                        mass = tl.load(masses + c, mask = miss, other = float("-inf"))
                    else:
                        mass = tl.load(masses + c)
                    mx, acc = _lse_step(mx, acc, mass + rv)
            val = tl.where(oc >= 0, val, _lse_value(mx, acc))
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + idx, val, mask = m)


@triton_jit
def _input_block_group_kernel(element_mars, node_mars, temp, obs_class, next_col, class_mars, mass_row, mass_stride,
                              width, iv_of, iv_info, iv_table, live, coff, ccount, cstart, ccolumns, crank, steps,
                              groups, sel, reg_offset, reg_width, reg_first, prior_mars, prior_offset, prior_width,
                              reg_prior, out_first, out_width, temp_width, B, n, W, C, input_start, csteps, pid0,
                              R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, PRUNE: tl.constexpr,
                              OUT_FULL: tl.constexpr, R_FULL: tl.constexpr, G: tl.constexpr, ONE_EXP: tl.constexpr,
                              COMPACT: tl.constexpr, INDEXED: tl.constexpr, SAMPLE: tl.constexpr,
                              TILE: tl.constexpr):
    """:func:`_input_block_kernel` for a group of steps per program, class by class, the block a sum's or a scratch
    row: group ``g`` is steps ``groups[g] .. groups[g + 1]`` (at most ``G``, a power of 2), which share their position
    and block interval, so they share their lanes' columns, successors and slots. Those are read once for the
    ``[G, TILE]`` lanes; only the input's masses and the block's values are read per step. ``ONE_EXP``: the
    log-sum-exp by :func:`_lse_step_one_exp` (bit-identical). ``COMPACT``: as :func:`_input_block_kernel`, the group's
    blocks all kept once or none (their interval is shared). ``INDEXED``: the groups of the plan's index list
    ``sel``. ``SAMPLE``: as :func:`_input_block_kernel`."""
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
    if tile * TILE < ncols:
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
        q = tl.load(next_col + (t * W + i) * C + oc, mask = obs, other = -1)
        sl = _operand_slot(iv_table, rpos, q, j, X, obs, own, crank, b, t + 1, q, rbase, ce, n, W, R_FULL, COMPACT)
        val = tl.load(node_mars + u[:, None] * B + b[None, :], mask = gm[:, None] & obs[None, :], other = float("-inf")) + \
            tl.load(rp + sl[None, :], mask = gm[:, None] & (obs & (sl >= 0))[None, :], other = float("-inf"))

        # a missing token, class by class: each class's successor and slot once, its mass and block value per step
        miss = ml & (oc < 0)
        if tl.max(tl.where(miss, 1, 0), axis = 0) > 0:
            mx = tl.full([G, TILE], float("-inf"), dtype = tl.float32)
            acc = tl.zeros([G, TILE], dtype = tl.float32)
            masses = class_mars + tl.load(mass_row + (u - input_start)).to(tl.int64) * C
            if SAMPLE:                                                   # [G, TILE]: the lane's sample's
                masses = masses[:, None] + \
                    (b[None, :] * tl.load(mass_stride + (u - input_start)).to(tl.int64)[:, None]) * C
            for c in range(C):
                q = tl.load(next_col + (t * W + i) * C + c, mask = miss, other = -1)
                sl = _operand_slot(iv_table, rpos, q, j, X, miss, own, crank, b, t + 1, q, rbase, ce, n, W, R_FULL,
                                   COMPACT)
                rv = tl.load(rp + sl[None, :], mask = gm[:, None] & (miss & (sl >= 0))[None, :], other = float("-inf"))
                if SAMPLE:
                    mass = tl.load(masses + c, mask = gm[:, None] & miss[None, :], other = float("-inf"))
                else:
                    mass = tl.load(masses + c)[:, None]
                if ONE_EXP:
                    mx, acc = _lse_step_one_exp(mx, acc, mass + rv)
                else:
                    mx, acc = _lse_step(mx, acc, mass + rv)
            val = tl.where((oc >= 0)[None, :], val, _lse_value(mx, acc))
        op = _out_ptr(tl.load(steps + gs * 7 + 0, mask = gm, other = 0), element_mars, temp, out_first, out_width,
                      temp_width, OUT_TEMP)
        tl.store(op[:, None] + idx[None, :], val, mask = gm[:, None] & m[None, :])


@triton_jit
def _block_input_kernel(element_mars, node_mars, temp, obs_class, next_col, class_mars, mass_row, mass_stride, trans,
                        trans_stride, pred_ptr, pred_col, pred_x, width, iv_of, iv_info, iv_table, live, coff, ccount,
                        cstart, ccolumns, crank, steps, sel, reg_offset, reg_width, reg_first, prior_mars,
                        prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, B, n, W, C,
                        input_start, csteps, pid0, L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                        GROUPED: tl.constexpr, PRUNE: tl.constexpr, OUT_FULL: tl.constexpr, L_FULL: tl.constexpr,
                        COMPACT: tl.constexpr, INDEXED: tl.constexpr, SAMPLE: tl.constexpr, TILE: tl.constexpr):
    """One program per step and tile of its whole output, flattened (sample, slot) as in
    :func:`_input_block_kernel`: every lane takes a log-sum over its exit column's predecessor entries (the
    program loops to the longest list among its lanes), each through its mass for a missing token, or the
    observed token's own log-probability when the entry holds its class. ``COMPACT``, ``INDEXED``, ``SAMPLE``: as
    :func:`_input_block_kernel`."""
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
    if tile * TILE < ncols:
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
        if GROUPED:
            masses = trans + tl.load(steps + s * 7 + 6).to(tl.int64)          # the step's transition masses
            if SAMPLE:                                                       # (the lane's sample's)
                masses = masses + tl.where(tl.load(mass_stride + (u - input_start)) > 0, b * trans_stride, 0)
        else:                                                                # the input's class masses
            masses = class_mars + tl.load(mass_row + (u - input_start)).to(tl.int64) * C
            if SAMPLE:
                masses = masses + b * tl.load(mass_stride + (u - input_start)).to(tl.int64) * C
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
            sl = _operand_slot(iv_table, lpos, i, q, M, em, l_own, crank, b, t, q, lbase, 1, n, W, L_FULL, COMPACT)
            lv = tl.load(lp + sl, mask = em & (sl >= 0), other = float("-inf"))
            mx, acc = _lse_step_one_exp(mx, acc, lv + mass)
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
    boundary's liveness, a chunk without a live column skipped). The shifts can lie farther above an entry's own terms
    than fp32's exponent reaches -- blocks of a wide dynamic range, or an operand kept for every sample (``prior_mars``),
    whose dead columns hold finite values the sample's evidence rules out -- and the entry's terms then flush: the
    returned sums tell (see :data:`EXACT_BELOW`). Below that, fp32 keeps its precision at any scale."""
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
            acc = acc * fL[:, None] * fR[None, :] + tl.dot(_tf32_operand(tl.exp(lt - sL[:, None]), PRECISION),
                                                            _tf32_operand(tl.exp(rt - sR[None, :]), PRECISION),
                                                            input_precision = PRECISION)
            mL, mR = nL, nR
    sL = tl.where(mL == float("-inf"), 0.0, mL)
    sR = tl.where(mR == float("-inf"), 0.0, mR)
    return tl.log(acc) + sL[:, None] + sR[None, :], acc


#: An output entry of :func:`_block_block_tile` whose sum of shifted exponentials is below this is computed again by
#: :func:`_block_block_tile_exact`. Every term the shifts flush out of fp32's range is below 2^-126 (so is a
#: subnormal term's error), at most 2^-112 together over a 16k-column boundary: above 2^-90 they are at most 2^-22
#: of the sum, fp32 rounding's order; below, they need not be.
EXACT_BELOW = 2.0 ** -90


@triton.jit
def _block_block_tile_exact(lp, rp, iv_table, lpos, rpos, mb, live_q, i, j, ti, tj, E, M, X, TILES_I, TILES_J, WORDS,
                            TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr,
                            SUB_J: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr):
    """:func:`_block_block_tile`'s output by an online log-sum-exp per entry over the shared columns, one at a time:
    exact whatever the blocks' dynamic range, at the cost of an exponential per (entry, column) instead of the
    tensor-core product. For the tiles whose entries the shifts flush (see :data:`EXACT_BELOW`): the entries of a
    block over a wide dynamic range that lie far below their row's and column's maxima -- those of a sample pinned by
    its evidence to columns of little mass, in a block kept for every sample."""
    im, jm = i < E, j < X
    m = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for c in range(0, tl.cdiv(M, TK)):
        need = True
        if SKIP:
            need = _chunk_needed(mb, ti, tj, c, TILES_I, TILES_J, WORDS, SUB_I, SUB_J)
        if PRUNE:
            qs = c * TK + tl.arange(0, TK)
            need = need & (tl.max(tl.load(live_q + qs, mask = qs < M, other = 0).to(tl.int32), axis = 0) > 0)
        if need:
            for k in range(0, TK):
                q = c * TK + k
                ok = q < M
                if PRUNE:
                    ok = ok & (tl.load(live_q + q, mask = q < M, other = 0) != 0)
                if ok:
                    sl = _slot(iv_table, lpos, i, q, M, im, False)
                    lv = tl.load(lp + sl, mask = im & (sl >= 0), other = float("-inf"))
                    sr = _slot(iv_table, rpos, q, j, X, jm, False)
                    rv = tl.load(rp + sr, mask = jm & (sr >= 0), other = float("-inf"))
                    m, acc = _lse_step(m, acc, lv[:, None] + rv[None, :])
    return _lse_value(m, acc)


@triton.jit
def _block_block_program(element_mars, node_mars, temp, width, iv_of, iv_info, live, steps, sel, triples, skip_bits,
                         reg_offset, reg_width, reg_first, prior_mars, prior_offset, prior_width, reg_prior, out_first,
                         out_width, temp_width, B, n, W, TILES_I, TILES_J, WORDS, SPLIT, pid,
                         L_KIND: tl.constexpr, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, INDEXED: tl.constexpr):
    """What program ``pid`` of a ``block @ block`` launch works on: its tile group ``g``, step and sample, the
    boundaries, widths and position tables of its three intervals, its operands' and output's blocks, its step's skip
    masks and its sample's liveness row."""
    g = pid % SPLIT
    s = _child(sel, pid // SPLIT // B, INDEXED)
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
    lp, l_own = _block_ptr(tl.load(steps + s * 8 + 1), tl.load(steps + s * 8 + 2), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, L_KIND)
    rp, r_own = _block_ptr(tl.load(steps + s * 8 + 3), tl.load(steps + s * 8 + 4), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, R_KIND)
    lp = lp + tl.where(l_own, b * LS, 0)
    rp = rp + tl.where(r_own, b * RS, 0)
    op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP) + b * OS
    mb = skip_bits + tl.load(triples + s).to(tl.int64) * TILES_I * TILES_J * WORDS
    lrow = live + b * (n + 1) * W                                            # the sample's liveness (if pruned)
    return g, t0, t1, t2, E, M, X, lpos, rpos, opos, lp, rp, op, mb, lrow


@triton_jit
def _block_block_kernel(element_mars, node_mars, temp, width, iv_of, iv_info, iv_table, live, steps, sel, triples,
                        skip_bits, reg_offset, reg_width, reg_first, prior_mars, prior_offset, prior_width, reg_prior,
                        out_first, out_width, temp_width, B, n, W, TILES_I, TILES_J, WORDS, SPLIT, GROUP, exact_below,
                        pid0, L_KIND: tl.constexpr, R_KIND: tl.constexpr, OUT_TEMP: tl.constexpr,
                        TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr, SUB_I: tl.constexpr,
                        SUB_J: tl.constexpr, SKIP: tl.constexpr, PRUNE: tl.constexpr, EXACT: tl.constexpr,
                        PRECISION: tl.constexpr, INDEXED: tl.constexpr):
    """One program per (step, sample, group of ``GROUP`` consecutive output tiles), its tiles in turn: the sample's
    two blocks are contiguous and stay in L1 across them. A (step, sample) has ``SPLIT`` groups: 1 unless the
    launch has too few (step, sample) pairs to fill the GPU. ``EXACT``: the program records, entry by entry, whether
    the shifts flushed any of its tiles' entries (see :data:`EXACT_BELOW`), and when they did -- rarely -- goes over
    its tiles again to find those (the shifted sums again, a reduction per tile) and compute them exactly
    (:func:`_block_block_tile_exact`). An elementwise record and one reduction per program in the common case: a
    pruned tile does little work, and a reduction per tile cost a fifth of it; a second launch for the flagged
    programs cost more still in launches."""
    g, t0, t1, t2, E, M, X, lpos, rpos, opos, lp, rp, op, mb, lrow = _block_block_program(
        element_mars, node_mars, temp, width, iv_of, iv_info, live, steps, sel, triples, skip_bits, reg_offset,
        reg_width, reg_first, prior_mars, prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, B, n,
        W, TILES_I, TILES_J, WORDS, SPLIT, pid0 + tl.program_id(0), L_KIND, R_KIND, OUT_TEMP, INDEXED)
    NJ = tl.cdiv(X, TN)
    if EXACT:
        bad = tl.zeros([TM, TN], dtype = tl.int32)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(E, TM) * NJ)):
        ti = t // NJ
        tj = t % NJ
        i = ti * TM + tl.arange(0, TM)
        j = tj * TN + tl.arange(0, TN)
        if PRUNE:                                                            # a tile without a live slot: -inf
            lr = tl.load(lrow + t0 * W + i, mask = i < E, other = 0) != 0
            lc = tl.load(lrow + t2 * W + j, mask = j < X, other = 0) != 0
            out = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
            if EXACT:
                acc = tl.full([TM, TN], 1.0, dtype = tl.float32)
            if (tl.max(lr.to(tl.int32), axis = 0) > 0) & (tl.max(lc.to(tl.int32), axis = 0) > 0):
                out, acc = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow + t1 * W, i, j, ti, tj, E, M, X,
                                             TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
            out = tl.where(lr[:, None] & lc[None, :], out, float("-inf"))
        else:
            out, acc = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow, i, j, ti, tj, E, M, X, TILES_I,
                                         TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
        om = (i < E)[:, None] & (j < X)[None, :]
        sl = _slot(iv_table, opos, i[:, None], j[None, :], X, om, False)
        if EXACT:
            valid = om & (sl >= 0)
            if PRUNE:
                valid = valid & lr[:, None] & lc[None, :]
            bad = bad | (valid & (acc < exact_below)).to(tl.int32)
        tl.store(op + sl, out, mask = om & (sl >= 0))
    if EXACT:
        if tl.max(bad) > 0:
            for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(E, TM) * NJ)):
                ti = t // NJ
                tj = t % NJ
                i = ti * TM + tl.arange(0, TM)
                j = tj * TN + tl.arange(0, TN)
                om = (i < E)[:, None] & (j < X)[None, :]
                sl = _slot(iv_table, opos, i[:, None], j[None, :], X, om, False)
                valid = om & (sl >= 0)
                if PRUNE:
                    lr = tl.load(lrow + t0 * W + i, mask = i < E, other = 0) != 0
                    lc = tl.load(lrow + t2 * W + j, mask = j < X, other = 0) != 0
                    valid = valid & lr[:, None] & lc[None, :]
                _, acc = _block_block_tile(lp, rp, iv_table, lpos, rpos, mb, lrow + t1 * W, i, j, ti, tj, E, M, X,
                                           TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE, PRECISION)
                if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                    out = _block_block_tile_exact(lp, rp, iv_table, lpos, rpos, mb, lrow + t1 * W, i, j, ti, tj, E, M,
                                                  X, TILES_I, TILES_J, WORDS, TM, TN, TK, SUB_I, SUB_J, SKIP, PRUNE)
                    tl.store(op + sl, tl.where(valid, out, float("-inf")), mask = om & (sl >= 0))


@triton.jit
def _rect_tile(p, base, stride, by_slot, positions, ncols, iv_table, x, y, xp, yp, xm, ym, T: tl.constexpr):
    """The ``[len(x), len(y)]`` tile of a block operand ``S``'s values ``S[x, y]`` (``T``: ``S[y, x]``, the operand
    read transposed), ``x`` and ``y`` columns of the automaton and ``xp`` and ``yp`` their places among the sample's
    live ones (``xm``, ``ym``: the real ones): compact (:mod:`.compact`), at ``p + base + (row's place) * stride +
    (column's place)`` -- ``base`` the sample's first lane in the row, ``stride`` its live columns at the block's
    exit --; ``by_slot``, at the pair's slot from ``p`` (a block kept once, or a sample's dense one), -inf where the
    automaton does not join them."""
    m = xm[:, None] & ym[None, :]
    if T:
        ri, ci, pr, pc = y[None, :], x[:, None], yp[None, :], xp[:, None]
    else:
        ri, ci, pr, pc = x[:, None], y[None, :], xp[:, None], yp[None, :]
    if by_slot:
        sl = _slot(iv_table, positions, ri, ci, ncols, m, False).to(tl.int64)
    else:
        sl = base + pr.to(tl.int64) * stride + pc
    return tl.load(p + sl, mask = m & (sl >= 0), other = float("-inf"))


@triton.jit
def _rect_product(ap, a0, a_stride, a_slot, a_pos, a_ncols, bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table, rid,
                  pr, rm, cid, pc, cm, columns, kstart, ck, A_T: tl.constexpr, B_T: tl.constexpr, TM: tl.constexpr,
                  TN: tl.constexpr, TK: tl.constexpr, PRECISION: tl.constexpr):
    """``log(sum_k exp(A[r, k] + B[k, c]))`` for one output tile -- rows ``rid`` (places ``pr``), columns ``cid``
    (places ``pc``) -- over the sample's ``ck`` live columns at the shared boundary (``columns[kstart]`` on), ``A``
    and ``B`` read by :func:`_rect_tile` (``A_T``, ``B_T``: stored transposed), shifted by every row's and column's
    running maximum as :func:`_block_block_tile` does. Returns the tile and its shifted sums (see :data:`EXACT_BELOW`)."""
    mA = tl.full([TM], float("-inf"), dtype = tl.float32)
    mB = tl.full([TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    for k0 in range(0, ck, TK):
        pk = k0 + tl.arange(0, TK)
        km = pk < ck
        kid = tl.load(columns + kstart + pk, mask = km, other = 0)
        at = _rect_tile(ap, a0, a_stride, a_slot, a_pos, a_ncols, iv_table, rid, kid, pr, pk, rm, km, A_T)
        bt = _rect_tile(bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table, kid, cid, pk, pc, km, cm, B_T)
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
def _rect_product_exact(ap, a0, a_stride, a_slot, a_pos, a_ncols, bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table,
                        rid, pr, rm, cid, pc, cm, columns, kstart, ck, A_T: tl.constexpr, B_T: tl.constexpr,
                        TM: tl.constexpr, TN: tl.constexpr):
    """:func:`_rect_product` by an online log-sum-exp per entry, one shared column at a time: exact whatever the
    operands' dynamic range."""
    m = tl.full([TM, TN], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TM, TN], dtype = tl.float32)
    one = tl.zeros([1], dtype = tl.int32)
    for k in range(0, ck):
        kid = one + tl.load(columns + kstart + k)
        at = _rect_tile(ap, a0, a_stride, a_slot, a_pos, a_ncols, iv_table, rid, kid, pr, one + k, rm, one == 0, A_T)
        bt = _rect_tile(bp, b0, b_stride, b_slot, b_pos, b_ncols, iv_table, kid, cid, one + k, pc, one == 0, cm, B_T)
        m, acc = _lse_step(m, acc, at + bt)                                  # [TM, 1] + [1, TN]
    return _lse_value(m, acc)


def compact_tiles(rows: int, cols: int):
    """``(TM, TN, TK, warps)`` of a compact ``block @ block`` launch whose samples have at most ``rows`` and ``cols``
    live columns at its outputs' ends (see :func:`_block_block_compact_kernel`)."""
    return (32, 32, 16, 4) if max(rows, cols) >= 32 else (16, 16, 16, 2)


@triton_jit
def _block_block_compact_kernel(element_mars, node_mars, temp, width, iv_of, iv_info, iv_table, coff, ccount, cstart,
                                ccolumns, steps, sel, reg_offset, reg_width, reg_first, prior_mars, prior_offset,
                                prior_width, reg_prior, out_first, out_width, temp_width, B, n, SPLIT, GROUP,
                                exact_below, pid0, L_KIND: tl.constexpr, R_KIND: tl.constexpr,
                                OUT_TEMP: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
                                PRECISION: tl.constexpr, INDEXED: tl.constexpr):
    """:func:`_block_block_kernel` on compact values (:mod:`.compact`): one program per (step, sample, group of
    ``GROUP`` tiles of the sample's rectangle of live columns at the output's ends), each tile a log-space matrix
    product over the sample's live columns at the shared boundary (:func:`_rect_product`), the operands' values by
    their places -- by slot, kept once --, -inf at the pairs the automaton does not join. The entries the shifts flush
    are recorded entry by entry and their tiles computed again exactly, as there."""
    pid = pid0 + tl.program_id(0)
    g = pid % SPLIT
    s = _child(sel, pid // SPLIT // B, INDEXED)
    b = (pid // SPLIT % B).to(tl.int64)
    t0 = tl.load(steps + s * 8 + 5)
    t1 = tl.load(steps + s * 8 + 6)
    t2 = tl.load(steps + s * 8 + 7)
    M = tl.load(width + t1)
    X = _exit_width(width, t2, n)
    _, lpos, _ = _interval(iv_of, iv_info, t0, t1, n)
    _, rpos, _ = _interval(iv_of, iv_info, t1, t2, n)
    _, opos, _ = _interval(iv_of, iv_info, t0, t2, n)
    lp, l_own = _block_ptr(tl.load(steps + s * 8 + 1), tl.load(steps + s * 8 + 2), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, L_KIND)
    rp, r_own = _block_ptr(tl.load(steps + s * 8 + 3), tl.load(steps + s * 8 + 4), node_mars, prior_mars, temp,
                           reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width, R_KIND)
    row = b * (n + 1)
    c0 = tl.load(ccount + row + t0)                                          # the sample's live columns at t0, t1, t2
    c1 = tl.load(ccount + row + t1)
    c2 = tl.load(ccount + row + t2)
    s0 = tl.load(cstart + row + t0)
    s1 = tl.load(cstart + row + t1)
    s2 = tl.load(cstart + row + t2)
    l0 = tl.load(coff + tl.load(iv_of + t0 * (n + 1) + t1).to(tl.int64) * (B + 1) + b)
    r0 = tl.load(coff + tl.load(iv_of + t1 * (n + 1) + t2).to(tl.int64) * (B + 1) + b)
    op = _out_ptr(tl.load(steps + s * 8 + 0), element_mars, temp, out_first, out_width, temp_width, OUT_TEMP) + \
        tl.load(coff + tl.load(iv_of + t0 * (n + 1) + t2).to(tl.int64) * (B + 1) + b)
    NJ = tl.cdiv(c2, TN)
    bad = tl.zeros([TM, TN], dtype = tl.int32)
    for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(c0, TM) * NJ)):
        pr = (t // NJ) * TM + tl.arange(0, TM)
        pc = (t % NJ) * TN + tl.arange(0, TN)
        rm, cm = pr < c0, pc < c2
        rid = tl.load(ccolumns + s0 + pr, mask = rm, other = 0)
        cid = tl.load(ccolumns + s2 + pc, mask = cm, other = 0)
        om = rm[:, None] & cm[None, :]
        valid = om & (_slot(iv_table, opos, rid[:, None], cid[None, :], X, om, False) >= 0)
        out, acc = _rect_product(lp, l0, c1, ~l_own, lpos, M, rp, r0, c2, ~r_own, rpos, X, iv_table, rid, pr, rm, cid,
                                 pc, cm, ccolumns, s1, c1, False, False, TM, TN, TK, PRECISION)
        bad = bad | (valid & (acc < exact_below)).to(tl.int32)
        tl.store(op + pr[:, None].to(tl.int64) * c2 + pc[None, :], tl.where(valid, out, float("-inf")), mask = om)
    if tl.max(bad) > 0:                                                      # an entry flushed (rare): its tiles
        for t in range(g * GROUP, tl.minimum(g * GROUP + GROUP, tl.cdiv(c0, TM) * NJ)):   # again, exactly where so
            pr = (t // NJ) * TM + tl.arange(0, TM)
            pc = (t % NJ) * TN + tl.arange(0, TN)
            rm, cm = pr < c0, pc < c2
            rid = tl.load(ccolumns + s0 + pr, mask = rm, other = 0)
            cid = tl.load(ccolumns + s2 + pc, mask = cm, other = 0)
            om = rm[:, None] & cm[None, :]
            valid = om & (_slot(iv_table, opos, rid[:, None], cid[None, :], X, om, False) >= 0)
            fast, acc = _rect_product(lp, l0, c1, ~l_own, lpos, M, rp, r0, c2, ~r_own, rpos, X, iv_table, rid, pr, rm,
                                      cid, pc, cm, ccolumns, s1, c1, False, False, TM, TN, TK, PRECISION)
            if tl.min(tl.where(valid, acc, 1.0)) < exact_below:
                out = _rect_product_exact(lp, l0, c1, ~l_own, lpos, M, rp, r0, c2, ~r_own, rpos, X, iv_table, rid, pr,
                                          rm, cid, pc, cm, ccolumns, s1, c1, False, False, TM, TN)
                tl.store(op + pr[:, None].to(tl.int64) * c2 + pc[None, :], tl.where(valid, out, float("-inf")),
                         mask = om)


@triton_jit
def _copy_kernel(element_mars, node_mars, temp, iv_of, iv_info, coff, steps, sel, reg_offset, reg_width, reg_first,
                 prior_mars, prior_offset, prior_width, reg_prior, out_first, out_width, temp_width, B, n, pid0,
                 L_KIND: tl.constexpr, OUT_TEMP: tl.constexpr, COMPACT: tl.constexpr, INDEXED: tl.constexpr,
                 TILE: tl.constexpr):
    """One program per step and tile of its output's lanes (``COMPACT``: its interval's, see :mod:`.compact`; the
    block's are the same)."""
    s = _child(sel, tl.program_id(0), INDEXED)
    pid = pid0 + tl.program_id(1)
    out_row = tl.load(steps + s * 5 + 0)
    t0 = tl.load(steps + s * 5 + 3)
    t2 = tl.load(steps + s * 5 + 4)
    S, _, _ = _interval(iv_of, iv_info, t0, t2, n)
    _, ncols = _out_count(iv_of, coff, t0, t2, B, S, n, COMPACT)
    if pid * TILE < ncols:
        cols = pid * TILE + tl.arange(0, TILE)
        m = cols < ncols
        # the block's scope is the output's: in prior_mars only when this is a batch of one (sample stride moot)
        lp, l_own = _block_ptr(tl.load(steps + s * 5 + 1), tl.load(steps + s * 5 + 2), node_mars, prior_mars, temp,
                               reg_offset, reg_width, reg_first, prior_offset, prior_width, reg_prior, temp_width,
                               L_KIND)
        op = _out_ptr(out_row, element_mars, temp, out_first, out_width, temp_width, OUT_TEMP)
        tl.store(op + cols, tl.load(lp + cols, mask = m), mask = m)


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
    return skip_masks_for(reach, [((a, b, False), (b, c, False)) for a, b, c in triples], tm, tn, tk, device)


def skip_masks_for(reach: Reachability, specs, tm: int = 16, tn: int = 16, tk: int = 16, device = None) -> SkipMasks:
    """
    The :class:`SkipMasks` of log-space products ``A @ B`` whose operands are blocks or transposed blocks: spec ``k``
    is ``((a, b, transposed), (c, d, transposed))``, ``A`` the block over ``[a, b)`` (rows: its entry columns, shared:
    its exit columns; transposed, the other way round) and ``B`` the block over ``[c, d)`` (shared: entry, output
    columns: exit; or the other way round). A forward step is ``((t0, t1, False), (t1, t2, False))``; the top-down
    pass reads one operand transposed.
    """
    T = reach.tile
    if tm % T or tn % T or tk % T:
        raise ValueError(f"Tiles ({tm}, {tn}, {tk}) must be multiples of the occupancy tile {T}.")
    if len(specs) == 0:
        return SkipMasks(torch.zeros(1, dtype = torch.int32, device = device), 0, 0, 1, tm, tn, tk)

    def occ(a, b, transposed):
        m = reach.occupancy(a, b)
        return m.t() if transposed else m

    lefts = sorted({a for a, _ in specs})
    rights = sorted({b for _, b in specs})
    L = [_tile_any(occ(*a), tm // T, tk // T) for a in lefts]                   # [output row tiles, chunks]
    R = [_tile_any(occ(*b), tk // T, tn // T) for b in rights]                  # [chunks, output column tiles]
    I = max(m.size(0) for m in L)
    J = max(m.size(1) for m in R)
    words = max(1, math.ceil(max(max(m.size(1) for m in L), max(m.size(0) for m in R)) / 32))
    NC = 32 * words

    def stack(ms, shape):
        out = torch.zeros(len(ms), *shape, dtype = torch.bool)
        for k, m in enumerate(ms):
            out[k, :m.size(0), :m.size(1)] = m
        return out

    Ls, Rs = stack(L, (I, NC)), stack(R, (NC, J))
    lpos = {a: k for k, a in enumerate(lefts)}
    rpos = {b: k for k, b in enumerate(rights)}
    li = torch.tensor([lpos[a] for a, _ in specs], dtype = torch.long)
    ri = torch.tensor([rpos[b] for _, b in specs], dtype = torch.long)
    shifts = torch.arange(32, dtype = torch.int64)
    chunk = max(1, (1 << 26) // (I * J * NC))
    parts = []
    for g0 in range(0, len(specs), chunk):
        need = Ls[li[g0:g0 + chunk], :, None, :] & Rs[ri[g0:g0 + chunk]].transpose(1, 2)[:, None, :, :]
        w = (need.view(need.size(0), I, J, words, 32).to(torch.int64) << shifts).sum(dim = -1)
        parts.append(torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32).flatten())
    return SkipMasks(torch.cat(parts).to(device), I, J, words, tm, tn, tk)


#: :func:`_block_block_kernel`'s ``(output tile entry x exit columns, shared chunk, warps)`` candidates, the first the
#: default, the others timed by the autotuner per launch shape (:mod:`pyjuice.layer.kernels.autotune`); every tile a
#: multiple of :class:`SkipMasks`'s, every chunk equal to it. The tile sets the dots' accumulation order, so
#: candidates agree to fp32 rounding (~1e-7 relative on a 16-latent PD), as the autotuner's contract allows. PD16,
#: tuned against the default: x1.03-1.22 (16 x 16 / 2 warps mostly)
BLOCK_BLOCK_TILES = [(16, 32, 16, 4), (16, 16, 16, 2), (32, 32, 16, 4), (16, 32, 16, 2)]
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
    return min(256, max(16, host_next_power_of_2(cols)))


def _flat_tile(lanes: int) -> int:
    """Lanes per program of a step whose output is flattened (sample, entry column, exit column)."""
    return min(512, max(128, host_next_power_of_2(lanes) // 8))


#: An ``input @ block`` launch runs its steps in groups (see :func:`_input_block_group_kernel`) only when a step's
#: output has more lanes than this (batch x slots): a step that fits one program shares little, and its launch would
#: have G times fewer programs (HMM4096 with 67- and 101-state automata at batch 1: 1-3% slower grouped).
GROUP_MIN_LANES = 128

#: The log-sum-exp steps the autotuner times for :func:`_input_block_group_kernel` (with every group size the
#: Program built): two exps (the default), and :func:`_lse_step_one_exp` -- bit-identical; x1.115 on the kernel with
#: most lanes live (one observed token), x0.91-0.97 under evidence pruning.
INPUT_BLOCK_ONE_EXP = (False, True)


def run_products(stages, bufs, prog, regions, element_mars, node_mars, masses, obs_class, live,
                 elem_first: int, elem_width: int, B: int, prior: dict, exact: bool, compact = None):
    """
    The steps a query plan runs of one product layer, stage by stage (a stage reads only earlier stages' scratch
    rows). Products are fp32-level whatever the query's precision, which applies to the sum layers: ``block @ block``
    dots in :data:`PRODUCT_PRECISION` (exact fp32 by default), the other contractions exact fp32 log-sum-exps.

    :param stages: the layer's stages: lists of ``(launch, cut)`` -- a :class:`Launch` and the :class:`Cut` of its
        steps the plan runs. ``block @ block`` steps: ``(steps, triple index)``, the latter indexing ``Program.skip``;
        ``input @ block`` and ``block @ input`` steps: ``(steps, chunks, groups)``, chunks ``(first, end, job)`` step
        ranges, ``job`` the :func:`transition_job` that builds their transition masses just before them, or None when
        the Program keeps them all; ``groups`` None (one step per program), or ``{G: firsts}`` for
        :func:`_input_block_group_kernel`, used above :data:`GROUP_MIN_LANES`
    :param bufs: ``(temp, temp_width)``, the layer's scratch rows
    :param masses: the query's :class:`~..state.ClassMasses` (per sample, the ``SAMPLE`` kernels)
    :param live: None, or the query's liveness (:class:`~.live.Liveness`'s ``live``)
    :param prior: where the sum operands the plan keeps once are (see :func:`_block_ptr`): ``mars``, the regions'
        ``offset`` and ``width`` there, and ``reg_prior`` (device tensors)
    :param exact: whether ``block @ block`` tiles whose entries its shifts flush are computed again exactly (see
        :data:`EXACT_BELOW`): for the blocks kept once, every entry of which some sample may need, and under evidence
        pruning, where a sample's few live columns may carry very different masses; a batch observing nothing reads
        its blocks' entries weighted by its own masses, which the shifts respect (and pays nothing)
    :param compact: None, or the query's :class:`~.compact.CompactLayout`, by which the per-sample inside values are
        compact (``regions``, ``elem_width`` and the scratch rows' width are their rows'): a lane per pair of live
        columns (``block @ block`` steps: :func:`_block_block_compact_kernel`, ``exact`` always)
    """
    temp, temp_width = bufs
    common = dict(reg_offset = regions["offset"], reg_width = regions["width"], reg_first = regions["first"],
                  prior_mars = prior["mars"], prior_offset = prior["offset"], prior_width = prior["width"],
                  reg_prior = prior["reg_prior"], out_first = elem_first, out_width = elem_width,
                  temp_width = temp_width)
    n, W, C = prog.n, prog.next_col.size(1), prog.next_col.size(2)
    ivs = prog.intervals
    prune, live = live is not None, live if live is not None else obs_class
    sample = masses.batch > 1
    inputs = dict(class_mars = masses.table, mass_row = masses.row,
                  mass_stride = masses.stride if sample else masses.row, SAMPLE = sample)
    if compact is not None:
        lists = dict(coff = compact.off, ccount = compact.count, cstart = compact.start, ccolumns = compact.columns,
                     crank = compact.rank, csteps = compact.steps)
    else:
        lists = dict(coff = live, ccount = live, cstart = live, ccolumns = live, crank = live, csteps = 0)
    for launches in stages:
        for launch, cut in launches:
            form, kinds = launch.form, launch.kinds
            E_max, X_max, S_max, S = cut.E_max, cut.X_max, cut.S_max, cut.count
            lanes = B * S_max if compact is None else compact.lanes(launch.out_ivs)   # an output's most
            if compact is not None and lanes == 0:                           # no sample has a live pair there
                continue
            if form in ("input_block", "block_input"):
                steps, _, groups = launch.steps
                trans, tt = prog.transition_buffer(masses.batch), prog.trans
                for s0, s1, job, sel, cnt in cut.chunks:
                    part = steps[s0:s1]
                    idx = dict(sel = part if sel is None else sel.tensor, INDEXED = sel is not None)
                    if job is not None:                              # this chunk's transition masses, in scratch
                        transition_masses(trans, job, masses, prog)
                    if form == "input_block" and groups is not None and lanes > GROUP_MIN_LANES:
                        r_kind, out_temp, (out_full, *r_full) = kinds
                        assert (s0, s1) == (0, launch.rows.size(0))   # groups index the launch's one chunk

                        def input_block_group(cfg):
                            G, one_exp = cfg
                            firsts = groups[G]
                            gsel, gcnt = cut.groups[G]
                            G = host_next_power_of_2(G)            # lanes per group (a group may be shorter)
                            tile = min(_flat_tile(lanes), 1024 // G)
                            for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                                _input_block_group_kernel[(gcnt, size)](
                                    element_mars, node_mars, temp, obs_class, prog.next_col,
                                    width = prog.width_t, iv_of = ivs.of, iv_info = ivs.info, iv_table = ivs.table,
                                    live = live, steps = part, groups = firsts,
                                    sel = firsts if gsel is None else gsel.tensor, B = B, n = n, W = W, C = C,
                                    input_start = prog.input_start, pid0 = first, R_KIND = r_kind,
                                    OUT_TEMP = out_temp, PRUNE = prune, OUT_FULL = out_full, R_FULL = all(r_full),
                                    G = G, ONE_EXP = one_exp, COMPACT = compact is not None,
                                    INDEXED = gsel is not None, TILE = tile, num_warps = 4, **inputs, **lists,
                                    **common)

                        # the group sizes the Program built (its default first), each step form: timed per shape (the
                        # steps rounded up to a power of 2, so that queries running subsets of a launch share choices)
                        cands = [(G, one_exp) for one_exp in INPUT_BLOCK_ONE_EXP for G in groups]
                        key = ("lifted_input_block_group", kinds, prune, compact is not None, sample, C,
                               host_next_power_of_2(S), host_next_power_of_2(lanes), tuple(groups))
                        input_block_group(autotune.pick(key, cands, input_block_group))   # outputs overwritten
                    elif form == "input_block":
                        r_kind, out_temp, (out_full, *r_full) = kinds
                        tile = _flat_tile(lanes)
                        for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                            _input_block_kernel[(cnt, size)](
                                element_mars, node_mars, temp, obs_class, prog.next_col, trans = trans,
                                trans_stride = prog.trans_size, entry_ptr = tt.entry_ptr, entry_succ = tt.entry_succ,
                                entry_x = tt.entry_x, width = prog.width_t, iv_of = ivs.of, iv_info = ivs.info,
                                iv_table = ivs.table, live = live, steps = part, B = B, n = n, W = W,
                                C = C, input_start = prog.input_start, pid0 = first, R_KIND = r_kind,
                                OUT_TEMP = out_temp, GROUPED = tt.grouped, PRUNE = prune, OUT_FULL = out_full,
                                R_FULL = all(r_full), COMPACT = compact is not None, TILE = tile, num_warps = 4,
                                **idx, **inputs, **lists, **common)
                    else:
                        l_kind, out_temp, (out_full, l_full) = kinds
                        tile = _flat_tile(lanes)
                        for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                            _block_input_kernel[(cnt, size)](
                                element_mars, node_mars, temp, obs_class, prog.next_col, trans = trans,
                                trans_stride = prog.trans_size, pred_ptr = tt.pred_ptr, pred_col = tt.pred_col,
                                pred_x = tt.pred_x, width = prog.width_t, iv_of = ivs.of, iv_info = ivs.info,
                                iv_table = ivs.table, live = live, steps = part, B = B, n = n, W = W,
                                C = C, input_start = prog.input_start, pid0 = first, L_KIND = l_kind,
                                OUT_TEMP = out_temp, GROUPED = tt.grouped, PRUNE = prune, OUT_FULL = out_full,
                                L_FULL = l_full, COMPACT = compact is not None, TILE = tile, num_warps = 4, **idx,
                                **inputs, **lists, **common)
            elif form == "block_block" and compact is not None:     # a sample's rectangle at a time
                l_kind, r_kind, out_temp = kinds
                steps, sel = launch.steps[0], cut.chunks[0][3]
                ends = prog.iv_ends_np[launch.out_ivs]
                rmax, cmax = int(compact.max_count[ends[:, 0]].max()), int(compact.max_count[ends[:, 1]].max())
                TM, TN, TK, warps = compact_tiles(rmax, cmax)
                tiles = host_cdiv(rmax, TM) * host_cdiv(cmax, TN)
                split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, S * B))
                group = host_cdiv(tiles, max(1, split))
                split = host_cdiv(tiles, group)
                for first, size in grid_chunks(S * B * split, 0):
                    _block_block_compact_kernel[(size,)](
                        element_mars, node_mars, temp, prog.width_t, ivs.of, ivs.info, ivs.table, compact.off,
                        compact.count, compact.start, compact.columns, steps, steps if sel is None else sel.tensor,
                        B = B, n = n, SPLIT = split, GROUP = group, exact_below = EXACT_BELOW, pid0 = first,
                        L_KIND = l_kind, R_KIND = r_kind, OUT_TEMP = out_temp, TM = TM, TN = TN, TK = TK,
                        PRECISION = PRODUCT_PRECISION, INDEXED = sel is not None, num_warps = warps, num_stages = 1,
                        **common)
            elif form == "block_block":
                l_kind, r_kind, out_temp = kinds
                steps, triples = launch.steps
                sel = cut.chunks[0][3]
                skip = prog.skip

                def block_block(cfg):
                    TM, TN, TK, warps = cfg
                    assert TK == skip.tk and TM % skip.tm == 0 and TN % skip.tn == 0
                    tiles = host_cdiv(E_max, TM) * host_cdiv(X_max, TN)
                    split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, S * B))
                    group = host_cdiv(tiles, max(1, split))
                    split = host_cdiv(tiles, group)                     # no group past the widest step's tiles
                    for first, size in grid_chunks(S * B * split, 0):
                        _block_block_kernel[(size,)](
                            element_mars, node_mars, temp, prog.width_t, ivs.of, ivs.info, ivs.table, live, steps,
                            steps if sel is None else sel.tensor, triples, skip.bits, B = B, n = n, W = W,
                            TILES_I = skip.tiles_i, TILES_J = skip.tiles_j, WORDS = skip.words, SPLIT = split,
                            GROUP = group, exact_below = EXACT_BELOW, pid0 = first, L_KIND = l_kind, R_KIND = r_kind,
                            OUT_TEMP = out_temp, TM = TM, TN = TN, TK = TK, SUB_I = TM // skip.tm, SUB_J = TN // skip.tn,
                            SKIP = SKIP_EMPTY_TILES, PRUNE = prune, EXACT = exact, PRECISION = PRODUCT_PRECISION,
                            INDEXED = sel is not None, num_warps = warps, num_stages = 1, **common)

                key = ("lifted_block_block", kinds, prune, exact, SKIP_EMPTY_TILES, PRODUCT_PRECISION, B,
                       host_next_power_of_2(S), E_max, X_max)
                block_block(autotune.pick(key, BLOCK_BLOCK_TILES, block_block))   # outputs overwritten: timed in place
            else:                                                            # "copy"
                l_kind, out_temp = kinds
                sel = cut.chunks[0][3]
                tile = _tile(lanes)
                for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                    _copy_kernel[(S, size)](
                        element_mars, node_mars, temp, ivs.of, ivs.info, lists["coff"], launch.steps,
                        launch.steps if sel is None else sel.tensor, B = B, n = n, pid0 = first, L_KIND = l_kind,
                        OUT_TEMP = out_temp, COMPACT = compact is not None, INDEXED = sel is not None, TILE = tile,
                        num_warps = 4, **common)
