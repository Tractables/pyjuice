"""
Compact storage of the per-sample values (a query's ``storage = "compact"``, see :mod:`..queries.conditional`).

Dense, a block over the interval ``[a, e)`` keeps ``slots`` lanes per sample -- every pair the automaton joins --, most
of them dead under evidence (:mod:`.live`). Compact, a row of an interval ``k`` keeps per sample ``b`` the RECTANGLE of
its live columns, ``count(b, a) x count(b, e)`` lanes, the samples one after the other: the pair ``(i, j)`` of sample
``b`` at ``off[k, b] + rank(b, a, i) * count(b, e) + rank(b, e, j)`` (``count``, ``rank``: the sample's live columns at
a boundary, and a column's place among them, see :meth:`~.live.Liveness.lists`). A pair the automaton does not join --
a HOLE, both ends live but no path between them inside the interval -- holds -inf. A row of interval ``k`` has
``T_k = off[k, B]`` lanes (padded), and a lane finds its sample by a binary search in ``off[k]``.

The per-sample top-down values are compact: ``node_flows``, the products' in ``element_flows`` and the folds' in the
scratch rows. So are the per-sample inside values (:attr:`CompactLayout.mars`): the sums' in ``node_mars`` (whose
input rows stay as they are, a value per sample), the products' in ``element_mars`` and the folds' in the scratch rows
(:func:`compact_buffers`). The values kept once stay dense (a batch of one, in ``prior_mars`` and
``prior_flows``); a row that may hold either (a product's, a fold's) is as wide as both need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_next_power_of_2

from ..compiled.tables import ALIGN
from .common import _slot, pack_aligned

#: the storages a query can keep its per-sample values in
STORAGES = ("dense", "compact")


@triton_jit
def _rectangles_kernel(off, count, ends, B, n, TILE: tl.constexpr):
    """One program per interval ``k`` (``ends``: int32 [intervals, 2]): ``off[k, b]``, the lanes of the samples before
    ``b`` in a row of ``k`` -- every sample's live columns at the entry times those at the exit --, and ``off[k, B]``
    the row's."""
    k = tl.program_id(0)
    a = tl.load(ends + k * 2)
    e = tl.load(ends + k * 2 + 1)
    base = off + k.to(tl.int64) * (B + 1)
    tl.store(base, tl.zeros([], dtype = tl.int64))
    total = tl.zeros([], dtype = tl.int64)
    for b0 in range(0, B, TILE):
        b = b0 + tl.arange(0, TILE)
        m = b < B
        rect = tl.load(count + b * (n + 1) + a, mask = m, other = 0).to(tl.int64) * \
            tl.load(count + b * (n + 1) + e, mask = m, other = 0).to(tl.int64)
        tl.store(base + 1 + b, total + tl.cumsum(rect, axis = 0), mask = m)
        total += tl.sum(rect, axis = 0)


@triton.jit
def _lane_sample(off, k, idx, B, steps):
    """The samples of the lanes ``idx`` of a row of interval ``k``: the last ``b`` with ``off[k, b] <= idx`` (a binary
    search of ``steps`` steps over the ``B + 1`` offsets; a sample without lanes is never the last)."""
    base = off + k.to(tl.int64) * (B + 1)
    lo = tl.zeros_like(idx)
    hi = lo + B
    for _ in range(steps):
        mid = (lo + hi + 1) // 2
        go = tl.load(base + mid) <= idx
        lo = tl.where(go, mid, lo)
        hi = tl.where(go, hi, mid - 1)
    return lo


@triton.jit
def _lane_pairs(off, count, start, columns, k, a, e, idx, m, B, n, steps):
    """``(b, i, j, pa, pe, ce)`` of the lanes ``idx`` (``m``: the real ones) of a row of interval ``k`` = ``[a, e)``:
    their sample, their pair's columns at the entry and the exit, and those columns' places among the sample's live
    ones there (``ce``: the sample's live columns at the exit)."""
    b = _lane_sample(off, k, idx, B, steps)
    local = (idx - tl.load(off + k.to(tl.int64) * (B + 1) + b, mask = m, other = 0)).to(tl.int32)
    ra = b.to(tl.int64) * (n + 1) + a
    re = b.to(tl.int64) * (n + 1) + e
    ce = tl.maximum(tl.load(count + re, mask = m, other = 1), 1)
    pa = local // ce
    pe = local - pa * ce
    i = tl.load(columns + tl.load(start + ra, mask = m, other = 0) + pa, mask = m, other = 0)
    j = tl.load(columns + tl.load(start + re, mask = m, other = 0) + pe, mask = m, other = 0)
    return b, i, j, pa, pe, ce


@triton.jit
def _lane_ends(off, count, start, columns, k, a, e, idx, m, B, n, steps):
    """``(b, i, j)`` of the lanes ``idx`` (``m``: the real ones) of a row of interval ``k`` = ``[a, e)``: their sample,
    and their pair's columns at the entry and the exit."""
    b, i, j, _, _, _ = _lane_pairs(off, count, start, columns, k, a, e, idx, m, B, n, steps)
    return b, i, j


@triton.jit
def _lane(off, count, rank, k, b, a, i, e, j, m, B, n, W):
    """The lane of the pair ``(i at a, j at e)`` of samples ``b`` in a row of interval ``k`` = ``[a, e)``, -1 where
    either column is dead for its sample (``m``: the real ones; shapes broadcast)."""
    row = rank + b.to(tl.int64) * (n + 1) * W
    ri = tl.load(row + a * W + i, mask = m, other = -1)
    rj = tl.load(row + e * W + j, mask = m, other = -1)
    ce = tl.load(count + b.to(tl.int64) * (n + 1) + e, mask = m, other = 0)
    lane = tl.load(off + k.to(tl.int64) * (B + 1) + b, mask = m, other = 0) + ri * ce + rj
    return tl.where((ri >= 0) & (rj >= 0), lane, -1)


@triton.jit
def _operand_slot(iv_table, positions, r, c, ncols, m, own, rank, b, t, q, base, mult, n, W, FULL: tl.constexpr,
                  COMPACT: tl.constexpr):
    """Where a product step's block operand holds its value of the pair ``(r, c)`` -- ``q`` its column at boundary
    ``t``, the other one the output's --, from where samples ``b``'s values start in its row: dense, the pair's slot
    in the sample's block (negative where the automaton does not join them, see :func:`~.common._slot`); ``COMPACT``,
    the pair's lane ``base + (q's place among the sample's live columns at t) x mult`` (-1 if ``q`` is dead) when the
    sample has values of its own there (``own``), else the pair's slot in the one block kept once."""
    if COMPACT:
        sl = _slot(iv_table, positions, r, c, ncols, m & ~own, FULL)
        rq = tl.load(rank + b * (n + 1) * W + t * W + q, mask = m & own & (q >= 0), other = -1)
        return tl.where(own, tl.where(rq >= 0, base + rq * mult, -1), sl)
    else:
        return _slot(iv_table, positions, r, c, ncols, m, FULL)


def _aligned(x):
    return (np.asarray(x, dtype = np.int64) + ALIGN - 1) // ALIGN * ALIGN


@dataclass
class CompactLayout:
    """
    Where a query keeps its per-sample top-down values compact (see the module docstring); :func:`compact_layout`.

    :ivar B: the batch size
    :ivar off: int64 [intervals, B + 1] on the device, by ``prog.intervals``' numbers: every sample's first lane in a
        row of the interval, then the row's lanes
    :ivar total: int64 [intervals] (host): every interval's lanes per row
    :ivar steps: the steps of a binary search over ``B + 1`` offsets
    :ivar count, start, columns, rank: the query's live lists and ranks (:meth:`~.live.Liveness.lists`)
    :ivar offset, width: per sum region, its rows' place and width in ``node_flows`` (host lists; width 0 for a region
        without per-sample values), and ``offset_t``, ``width_t``, ``lanes_t`` (int64 on the device; ``lanes_t``: its
        rows' lanes) and ``size``, the floats of ``node_flows``
    :ivar sum_lanes: the most lanes of a row of a region with per-sample values
    :ivar max_count: int64 [n + 1] (host): every boundary's most live columns of a sample
    :ivar elem_width: per product layer group, the width of its rows in ``element_flows`` (as wide as the compact values
        of its products under a sum with per-sample values -- a product's interval is its sum's --, or one sample's
        block kept once), and ``element_size``, the floats ``element_flows`` needs
    :ivar sample_ivs: bool [intervals] (host): those of the sum regions with per-sample values
    :ivar mars: None (the inside values dense), or where the query keeps its per-sample inside values compact, as
        :func:`~..compiled.tables.buffer_layout` says it for dense ones (:func:`compact_layout`), with ``sum_offsets_t``
        / ``sum_widths_t`` on the device, ``lanes`` (the most lanes of a row of a region with per-sample inside values),
        ``regions`` (bool [sum regions], host: those with rows) and ``ivs`` (bool [intervals], host: their intervals)
    """

    B: int
    off: torch.Tensor
    total: np.ndarray
    steps: int
    count: torch.Tensor
    start: torch.Tensor
    columns: torch.Tensor
    rank: torch.Tensor
    offset: list
    width: list
    offset_t: torch.Tensor
    width_t: torch.Tensor
    lanes_t: torch.Tensor
    size: int
    sum_lanes: int
    max_count: np.ndarray
    sample_ivs: np.ndarray
    elem_width: list
    element_size: int
    prog: Any
    mars: Optional[dict] = None

    def lanes(self, ivs) -> int:
        """The most lanes per row of the intervals ``ivs`` (``prog.intervals``' numbers; 0 for none)."""
        ivs = np.asarray(ivs, dtype = np.int64)
        return int(self.total[ivs].max()) if ivs.size else 0

    def row_width(self, ivs, slots: int) -> int:
        """The width of rows that hold the compact values of the intervals ``ivs``, or a batch of one's ``slots``."""
        return int(_aligned(max(self.lanes(ivs), slots)))

    def rows(self, kind: str):
        """``(offset, width)``: every sum region's rows' place and width (host lists), in ``node_flows`` (``kind``
        "flows") or ``node_mars`` ("mars")."""
        return (self.mars["sum_offsets"], self.mars["sum_widths"]) if kind == "mars" else (self.offset, self.width)

    def identity(self, size: int) -> torch.Tensor:
        """int32 ``0 .. size - 1`` on the device (the dense sum path's columns: a compact row's lanes in order)."""
        prog = self.prog
        ident = getattr(prog, "compact_identity", None)
        if ident is None or ident.numel() < size:
            ident = prog.compact_identity = torch.arange(max(size, 1024), dtype = torch.int32, device = prog.dev)
        return ident[:size]


def compact_layout(backend, liveness, regions: np.ndarray, inside: Optional[np.ndarray] = None) -> CompactLayout:
    """
    The :class:`CompactLayout` of a query: every interval's rectangles from its liveness (one launch, and the lists
    with their ranks), and the rows in ``node_flows`` of the sum ``regions`` (bool [sum regions], host) whose top-down
    values it keeps per sample; ``inside``: None, or the sum regions (bool) whose inside values it keeps per sample,
    compact (:attr:`CompactLayout.mars`). The one wait: the rows' lanes, for their widths, and every boundary's most
    live columns, for the grids.
    """
    prog = backend.program()
    B, n = liveness.B, prog.n
    count, start, columns = liveness.lists(rank = True)
    K = prog.iv_ends.size(0)
    off = torch.empty(K, B + 1, dtype = torch.int64, device = prog.dev)
    _rectangles_kernel[(K,)](off, count, prog.iv_ends, B, n, TILE = min(1024, max(32, host_next_power_of_2(B))),
                             num_warps = 4)
    host = torch.cat([off[:, B], count.view(B, n + 1).amax(dim = 0).long()]).cpu().numpy()
    total, max_count = host[:K], host[K:]
    lanes = total[prog.reg_iv_all]
    width = np.where(regions, _aligned(lanes), 0)
    offset = _row_offsets(prog.reg_rows, width)
    sample_ivs = _region_ivs(prog, regions, K)
    elem_width, element_size = _group_widths(prog, total, sample_ivs)
    arrays = [offset, width, lanes]
    mars = None
    if inside is not None:
        mars = _mars_layout(backend, prog, total, lanes, inside, B)
        arrays += [mars["sum_offsets"], mars["sum_widths"]]
    tables = _upload(prog, arrays)
    if mars is not None:
        mars.update(sum_offsets = mars["sum_offsets"].tolist(), sum_widths = mars["sum_widths"].tolist(),
                    sum_offsets_t = tables[3], sum_widths_t = tables[4])
    return CompactLayout(B = B, off = off, total = total, steps = max(1, int(np.ceil(np.log2(B + 1)))), count = count,
                         start = start, columns = columns, rank = liveness.rank, offset = offset.tolist(),
                         width = width.tolist(), offset_t = tables[0], width_t = tables[1], lanes_t = tables[2],
                         size = int((prog.reg_rows * width).sum()), sum_lanes = int(lanes[regions].max(initial = 0)),
                         max_count = max_count, sample_ivs = sample_ivs, elem_width = elem_width,
                         element_size = element_size, prog = prog, mars = mars)


def _row_offsets(rows: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Where every sum region's rows start (int64), the regions' ``rows`` of ``width`` floats one after the other."""
    out = np.zeros(len(width), dtype = np.int64)
    np.cumsum((rows * width)[:-1], out = out[1:])
    return out


def _region_ivs(prog, regions: np.ndarray, K: int) -> np.ndarray:
    """bool [intervals]: those of the sum ``regions`` (bool [sum regions])."""
    out = np.zeros(K, dtype = bool)
    out[prog.reg_iv_all[regions]] = True
    return out


def _group_widths(prog, total: np.ndarray, ivs: np.ndarray):
    """``(widths, size)``: per product layer group, the width of its rows (a list) -- as wide as the compact values of
    its products whose interval ``ivs`` marks (bool [intervals]; ``total``: every interval's lanes), or one sample's
    block kept once --, and the floats the widest group's rows take. The table's padding is one past the last
    interval: none marked, without lanes."""
    pad = prog.elem_iv_pad
    lanes = np.where(np.append(ivs, False)[pad], np.append(total, 0)[pad], 0).max(axis = 1)
    widths = _aligned(np.maximum(lanes, prog.elem_slots))
    return widths.tolist(), int((prog.elem_rows * widths).max(initial = 0))


def _upload(prog, arrays) -> list:
    """The int64 host ``arrays`` on the device, in one copy (each 16 bytes aligned, see
    :func:`~.common.pack_aligned`)."""
    buf, starts = pack_aligned(arrays, np.int64)
    dev = torch.from_numpy(buf).to(prog.dev)
    return [dev[s:s + len(a)] for a, s in zip(arrays, starts.tolist())]


def _mars_layout(backend, prog, total: np.ndarray, lanes: np.ndarray, inside: np.ndarray, B: int) -> dict:
    """:attr:`CompactLayout.mars`, but its rows' offsets and widths as arrays and not on the device: ``node_mars``'s
    input rows as dense ones (a value per sample, where pyjuice's input layers write them), then the compact rows of
    the sum regions ``inside`` (bool), none for the others; per product layer group, ``element_mars``'s rows as wide
    as the compact values of its products under those regions, or one sample's block kept once."""
    width = np.where(inside, _aligned(lanes), 0)
    base = int(_aligned(backend.input_range[1] * B))
    ivs = _region_ivs(prog, inside, len(total))
    elem, size = _group_widths(prog, total, ivs)
    return dict(input_offset = 0, sum_offsets = base + _row_offsets(prog.reg_rows, width), sum_widths = width,
                node_size = base + int((prog.reg_rows * width).sum()), element_widths = elem, element_size = size,
                lanes = int(lanes[inside].max(initial = 0)), ivs = ivs, regions = inside)


#: rows of a region one re-gather program copies, and its lanes; floats a segment copy's program copies
REGATHER_ROWS, REGATHER_TILE = 16, 512
COPY_TILE = 1024


@triton_jit
def _regather_kernel(src, dst, table, parents, old_off, old_count, old_rank, off, count, start, columns, B_old, B, n,
                     W, steps, ROWS: tl.constexpr, TILE: tl.constexpr):
    """One program per (``TILE`` of a region's lanes, region, ``ROWS`` of its rows; ``table`` as :func:`regather`'s --
    the lanes first, the grid's only axis long enough for every lane of a wide batch): every lane of the new layout
    -- sample ``b``, pair ``(i, j)`` -- gets the old one of sample ``parents[b]`` at the same pair (-inf where the old
    layout has none)."""
    t = tl.program_id(1).to(tl.int64) * 8
    r0 = tl.program_id(2).to(tl.int64) * ROWS
    rows, k = tl.load(table + t + 4), tl.load(table + t + 5)
    idx = tl.program_id(0) * TILE + tl.arange(0, TILE)
    lanes = tl.load(off + k * (B + 1) + B).to(tl.int32)
    if (r0 < rows) & (tl.program_id(0) * TILE < lanes):
        a, e = tl.load(table + t + 6), tl.load(table + t + 7)
        m = idx < lanes
        b, i, j = _lane_ends(off, count, start, columns, k, a, e, idx, m, B, n, steps)
        p = tl.load(parents + b, mask = m, other = 0)
        old = _lane(old_off, old_count, old_rank, k, p, a, i, e, j, m, B_old, n, W)
        sw, dw = tl.load(table + t + 1), tl.load(table + t + 3)
        so, do = tl.load(table + t) + r0 * sw, tl.load(table + t + 2) + r0 * dw
        for r in range(tl.minimum(ROWS, rows - r0)):
            tl.store(dst + do + r * dw + idx, tl.load(src + so + r * sw + old, mask = m & (old >= 0),
                                                      other = float("-inf")), mask = m)


def regather(prog, old: "CompactLayout", src: torch.Tensor, new: "CompactLayout", dst: torch.Tensor,
             regions: np.ndarray, parents: torch.Tensor, dst_offset: Optional[np.ndarray] = None, rows: str = "flows"):
    """
    The compact values of the sum ``regions`` (bool [sum regions]) from ``src``, laid out by ``old`` (a previous
    query's layout), into ``dst``, laid out by ``new`` (rows starting at ``dst_offset`` instead, int64 [sum regions], if
    given) -- top-down values, or inside values (``rows``, see :meth:`CompactLayout.rows`): every row of the new batch
    gets its ``parents`` row's values (int64 [B] on the device) at the same pairs of live columns -- one launch. A query
    of more evidence than its parent rows' has fewer live columns, so every new lane has an old one; a region's values
    must hold under the new evidence (none newly observed outside its scope -- top-down values -- or inside it, see
    :mod:`..queries.decoder`).
    """
    regs = np.nonzero(regions)[0]
    if regs.size == 0:
        return
    k = prog.reg_iv_all[regs]
    ends = prog.iv_ends_np[k]
    (old_offset, old_width), (new_offset, new_width) = old.rows(rows), new.rows(rows)
    dst_offset = np.asarray(new_offset if dst_offset is None else dst_offset)[regs]
    table = np.stack([np.asarray(old_offset)[regs], np.asarray(old_width)[regs], dst_offset,
                      np.asarray(new_width)[regs], prog.reg_rows[regs], k, ends[:, 0], ends[:, 1]], axis = 1)
    table = torch.from_numpy(np.ascontiguousarray(table, dtype = np.int64)).to(prog.dev)
    grid = (-(-max(new.lanes(k), 1) // REGATHER_TILE), regs.size, -(-int(prog.reg_rows[regs].max()) // REGATHER_ROWS))
    _regather_kernel[grid](src, dst, table, parents, old.off, old.count, old.rank, new.off, new.count, new.start,
                           new.columns, old.B, new.B, prog.n, prog.next_col.size(1), new.steps, ROWS = REGATHER_ROWS,
                           TILE = REGATHER_TILE, num_warps = 4)


@triton_jit
def _copy_segments_kernel(src, dst, table, TILE: tl.constexpr):
    """One program per (``TILE`` of a segment's floats, segment): ``table`` int64 [segments, 3], (source start,
    destination start, length)."""
    t = tl.program_id(1).to(tl.int64) * 3
    c = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    size = tl.load(table + t + 2)
    if tl.program_id(0).to(tl.int64) * TILE < size:
        m = c < size
        tl.store(dst + tl.load(table + t + 1) + c, tl.load(src + tl.load(table + t) + c, mask = m), mask = m)


def regather_into(state, prog, old: "CompactLayout", new: "CompactLayout", regions: np.ndarray,
                  parents: torch.Tensor, allocate, name: str = "node_flows", rows: str = "flows"):
    """
    :func:`regather` from the storage ``state`` holds as ``name`` (``node_flows`` or ``node_mars``, its ``rows`` laid
    out by ``old``) into the one ``allocate()`` returns (a dict holding it as ``name``, laid out by ``new``), never
    holding both: the ``regions``' values (some) are gathered into a scratch first, the old storage given up, and the
    scratch copied into the new one -- so a query whose rows hold little more than they did does not need their
    memory twice. Returns ``allocate()``'s result.
    """
    regs = np.nonzero(regions)[0]
    new_offset, new_width = new.rows(rows)
    sizes = prog.reg_rows[regs] * np.asarray(new_width, dtype = np.int64)[regs]
    starts = np.zeros(len(new_width), dtype = np.int64)
    starts[regs] = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    scratch = torch.empty(max(int(sizes.sum()), 1), device = prog.dev)
    src = state.replace_storage(name, None)
    regather(prog, old, src, new, scratch, regions, parents, dst_offset = starts, rows = rows)
    del src
    out = allocate()
    table = np.stack([starts[regs], np.asarray(new_offset, dtype = np.int64)[regs], sizes], axis = 1)
    table = torch.from_numpy(np.ascontiguousarray(table, dtype = np.int64)).to(prog.dev)
    _copy_segments_kernel[(-(-max(int(sizes.max()), 1) // COPY_TILE), regs.size)](scratch, out[name], table,
                                                                                TILE = COPY_TILE, num_warps = 4)
    return out


def compact_buffers(backend, liveness, plan, B: int, inputs_only: bool, regions: np.ndarray, held: Optional[np.ndarray],
                    carry: Optional[dict]):
    """
    ``(layout, bufs)`` of a compact query: its :class:`CompactLayout` -- its top-down values those of the sum
    ``regions`` (bool), its inside values those it computes per sample and those it holds (``held``: None or bool [sum
    regions]) -- and its buffers (:meth:`~..state.State.buffers`, ``inputs_only`` as there), the inside values it holds
    re-gathered into them (:func:`regather_into`) from the layout of the query that left them, ``carry`` (see
    :func:`~..queries.conditional.conditional`; required when the query holds some).
    """
    prog = backend.program()
    inside = plan.sample[prog.reg_iv_np]
    if held is not None:
        assert not held.any() or carry is not None, "compact values held without the layout they are in"
        inside = inside | held
    layout = compact_layout(backend, liveness, regions, inside = inside)
    allocate = lambda: backend.state.buffers(B, inputs_only, layout.mars)
    if carry is not None and carry["inside"].any():
        return layout, regather_into(backend.state, prog, carry["layout"], layout, carry["inside"], carry["parents"],
                                     allocate, "node_mars", "mars")
    return layout, allocate()
