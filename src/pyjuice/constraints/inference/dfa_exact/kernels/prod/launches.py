"""
What a product launch is: its steps' fields (:class:`Launch`) and how a query cuts it (:class:`Cut`), and the helpers every product kernel
reads its output with.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch
import triton
import triton.language as tl

from ..common import _pair, _slot
from ..compact import _lane_pairs, _lane_sample


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
    :ivar rows: [S, fields] int64, its steps on the host, in the device tables' order
    :ivar iv: [S] int64, the interval of every step's product, as its number in ``Program.sum_ivs`` (a scratch step
        follows the product it is a fold of)
    :ivar ends: the ends of its steps' outputs' intervals, each once (its ``parts``), and ``part_of`` [S], every step's
        among them (None: one part)
    :ivar out_ivs: int64 (host): the intervals of its steps' outputs, each once, by ``prog.intervals``' numbers (for
        compact storage, see :mod:`.compact`)
    :ivar number: its place among its :class:`~..compiled.circuit.Assembly`'s launches

    A launch is the circuit's, one for every automaton of a pattern: what an automaton says of it -- its parts' columns
    and slots -- is in the program's arrays, by ``number`` (:meth:`~..compiled.program.Program.most_of`).
    """

    form: str
    kinds: tuple
    steps: Any
    rows: torch.Tensor
    iv: torch.Tensor
    ends: list
    part_of: Optional[torch.Tensor] = None
    groups_host: Optional[dict] = None
    out_ivs: Any = None
    number: int = 0
    _input_boundaries: Any = None

    @property
    def out_temp(self) -> bool:
        """Whether its steps' outputs are scratch rows (a fold's), not the layer's ``element_mars`` rows."""
        return self.kinds[2 if self.form == "block_block" else 1]

    @property
    def input_boundaries(self) -> np.ndarray:
        """``input_block`` steps: the boundaries of their inputs, each once (int64, host; computed on first use)."""
        if self._input_boundaries is None:
            self._input_boundaries = np.unique(self.rows[:, 2].numpy())
        return self._input_boundaries


@dataclass
class Cut:
    """
    The steps of a :class:`Launch` a query plan runs: per chunk of the launch (``chunks``: ``(first, end, None, index
    list or None for every step, count)``, see :func:`run_products`; the index list relative to the chunk's first); for
    ``input_block`` steps run in groups, per group size ``(index list or None, count)`` of its groups (a group's steps
    share their interval); where the selected steps' most entry columns, exit columns and slots are in the plan's sizes
    (``at``: three numbers, what an automaton says of them), and how many steps there are.
    """

    chunks: list
    groups: Optional[dict]
    at: int
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
def _fresh_tile(hold_cum, coff, k, tile, TILE: tl.constexpr, ncols, B, steps, HOLD: tl.constexpr):
    """Whether a tile of a compact row's lanes (of interval ``k``, ``ncols`` of them) has one of a sample computing its
    values: ``HOLD``, by the samples of its first and last lanes (a sample's lanes are consecutive) and the interval's
    prefix counts of those computing theirs (``hold_cum`` [intervals, B + 1], :func:`~.compact.fresh_counts`); else
    every tile does."""
    if HOLD:
        ends = tile * TILE + tl.where(tl.arange(0, 2) == 0, 0, tl.minimum(TILE, ncols - tile * TILE) - 1)
        bs = _lane_sample(coff, k, ends, B, steps)
        base = hold_cum + k.to(tl.int64) * (B + 1)
        return tl.load(base + tl.max(bs, axis = 0) + 1) - tl.load(base + tl.min(bs, axis = 0)) > 0
    else:
        return tile >= 0


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


@triton.jit
def _lane_rows(tok_ptr, prow, lognorm, t, mrow, lm, ENUM: tl.constexpr, LOGW: tl.constexpr):
    """What :func:`_lane_mass` reads the same for every term of a lane, loaded once ahead of a loop over its terms:
    ``(tk, pr, lz)`` -- the first enumerated token of boundary ``t`` (``tok_ptr[t]``), and the lanes' parameter rows
    ``prow[mrow]`` and (``LOGW``) log Z ``lognorm[mrow]``, where ``lm`` (a mask every term's mask lies within);
    placeholders, nothing read, without ``ENUM`` (and ``lz`` without ``LOGW``)."""
    if ENUM:
        tk = tl.load(tok_ptr + t)
        pr = tl.load(prow + mrow, mask = lm, other = 0).to(tl.int64)
        if LOGW:
            lz = tl.load(lognorm + mrow, mask = lm, other = 0.0)
        else:
            lz = (mrow * 0).to(tl.float32)
    else:
        tk = t * 0
        pr = mrow
        lz = (mrow * 0).to(tl.float32)
    return tk, pr, lz
