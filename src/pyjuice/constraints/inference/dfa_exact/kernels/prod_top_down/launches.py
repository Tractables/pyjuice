"""
What a top-down product launch is (:class:`TopDownLaunch`), and how every top-down kernel reads a parent's values and stores a child's.
The copy kernel, the simplest of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from ..common import _child, _interval, _logaddexp_tile, _lse_step
from ..compact import _lane
from ..prod.launches import _lse_value


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
    :ivar host: for query plans, numpy copies: ``children`` [C, 3] (row, region, add), ``ptr``, ``parents``, and every
        child's (``child_iv``) and parent's (``parent_iv``) interval, by ``prog.intervals``' numbers -- and those
        intervals each once (``child_ivs``, ``parent_ivs``; for compact storage, see :mod:`.compact`); for
        ``input_block`` / ``block_input``, every child's node tile ``group`` (:func:`~.common.node_groups`: its
        interval, number of parents and ``add``)
    :ivar tiles, tile_height: for ``input_block`` / ``block_input``, every child in node tiles (int32 on the device)
        and their height (:func:`~.common.node_tiles`) -- a query plan's lists of these launches are tiles too
    :ivar number: its place among its circuit's: ``prog.td_most[3 * number:]``, the most entry columns, exit columns and
        slots of the children's blocks (what an automaton says of them)
    """

    form: str
    side: str
    kinds: tuple
    children: torch.Tensor
    ptr: torch.Tensor
    parents: torch.Tensor
    host: Any
    tiles: Optional[torch.Tensor] = None
    tile_height: int = 1
    number: int = 0


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
