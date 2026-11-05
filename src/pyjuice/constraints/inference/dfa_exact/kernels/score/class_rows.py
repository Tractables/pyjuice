"""
The class weights' rows: a batch's live classes, a row per class or per sample of it, those of the same terms shared (:class:`ClassWeightRows`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2
from ..common import _next_col
from ..sum.tile_rules import _num_sms


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
    its weights. A live class's weights are sums over those live pairs alone (a pair dead at either end adds -inf), in
    its list's order -- the same terms in the same order for every class with the same live pairs (its EFFECTIVE
    list): such classes share one row (``shared``, see :data:`CLASS_ROWS_SHARED`), that of the first of them in the
    boundary's class order (``cls_order``), their REPRESENTATIVE. Every column's rows, its representatives' in class
    order, are consecutive rows of ``K`` floats (its row group's rows). Few classes' lists differ under evidence
    (HMM4096 x JSON hard_0, B = 8: 912535 live classes of the 1024 target columns, 267218 lists, at a decode's call 1;
    296742 and 6824 at call 10).

    :ivar count: int32 [N], every column's rows (its live classes, or their distinct lists)
    :ivar cls: int32 [N, C], every column's rows' classes (the representatives): ``cls[m, :count[m]]``
    :ivar off: int64 [N + 1], every column's first row's float in ``dfa_class_flow`` (the last: the size)
    :ivar row: int32 [N, C], every (column, class)'s row among its row group's, -1 if dead
    :ivar rep: None, or (``shared``) int32 [N, C]: every live (column, class)'s representative (itself for one), -1 if
        dead -- a row's owner is its representative
    :ivar starts: every row group's first float, then the size (host; read on first use, see :attr:`most`)
    :ivar most: the most rows of any column (host): copied with :attr:`starts` as the rows are built, and waited for
        on first use -- a query asks after its forward pass's launches, so the wait is the rows' alone
    :ivar walk: per boundary, the most live columns of any sample there (host, every column under no pruning; set by
        :func:`class_weight_walk`): the class weights' longest walk at a position (:data:`CLASS_WEIGHT_CUDA_WALK`)
    """

    count: torch.Tensor
    cls: torch.Tensor
    off: torch.Tensor
    row: torch.Tensor
    _host: torch.Tensor
    _done: Any
    _groups: int
    rep: Optional[torch.Tensor] = None
    _read: Optional[tuple] = None
    _walk: Optional[list] = None
    _walk_copy: Optional[tuple] = None

    def _sizes(self):
        if self._read is None:
            self._done.synchronize()
            v = self._host.tolist()
            g = self._groups
            self._read = (v[:g + 1], int(v[g + 1]))
        return self._read

    @property
    def starts(self) -> list:
        return self._sizes()[0]

    @property
    def most(self) -> int:
        return self._sizes()[1]

    @property
    def walk(self) -> list:
        if self._walk is None:                          # (copied behind the launches, see `class_weight_walk`)
            host, done = self._walk_copy
            done.synchronize()
            self._walk = host.tolist()
        return self._walk


#: whether the live classes of a target column with the same live pairs share one row of class weights (see
#: :class:`ClassWeightRows`): their weights walked once, their tokens scored from it
CLASS_ROWS_SHARED = True
#: the bits of the classes' keys (:func:`_row_key`) the grouping compares: a key only proposes the classes that may
#: share a row, each is then checked exactly (:func:`_class_reps_kernel`); fewer bits (tests: 0, every class of a
#: column one candidate) cost merges, never values. At most 48 (:data:`_KEY_BITS`): a column's sort in registers packs
#: a class's rank below its key (:func:`_sorted_candidates_kernel`)
CLASS_ROW_KEY_MASK = (1 << 48) - 1
_KEY_BITS = tl.constexpr(48)
_KEY_NONE = tl.constexpr((1 << 63) - 1)                 # (a dead class's key: after every live one)


@triton.jit
def _mix64(x):
    """splitmix64's finalizer: a bijection of 64-bit integers whose outputs look independent."""
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    return x ^ (x >> 31)


@triton.jit
def _pair_hash(i, q, W):
    """The hash of the pair (column ``i``, successor ``q``) of a boundary of ``W`` columns: uint64."""
    return _mix64((i.to(tl.int64) * (W + 1) + q + 0x5BD1E995).to(tl.uint64, bitcast = True))


@triton.jit
def _row_key(h, cnt, stored, key_mask):
    """A live class's key: the sum of its live pairs' hashes ``h`` (uint64), their number ``cnt`` and whether the
    class is stored (``stored``: a stored class's row is exponentiated in place by the scoring, an enumerated one's
    read in log -- they never share), as a non-negative int64 masked by ``key_mask``."""
    salt = tl.where(stored, tl.full(cnt.shape, 0x9E3779B97F4A7C15, tl.uint64),
                    tl.full(cnt.shape, 0x2545F4914F6CDD1D, tl.uint64))
    k = h ^ _mix64(cnt.to(tl.uint64) + salt)
    return (k >> 2).to(tl.int64, bitcast = True) & key_mask


@triton_jit
def _live_classes_kernel(live, cls_ptr, cls_col, cls_succ, col_pos, col_K, row, floats, off, key, elen, stored, n, W,
                         C, B, N, allowed, a_stride, key_mask, TC: tl.constexpr, TP: tl.constexpr, PRUNE: tl.constexpr,
                         ALLOWED: tl.constexpr, KEYS: tl.constexpr):
    """One program per target column ``m`` (sample ``m % B`` of position ``col_pos[m // B]``) and tile of ``TC``
    classes: which of them are live -- a class whose list (``cls_ptr`` / ``cls_col`` / ``cls_succ``) has a pair live
    at both ends for the sample (``live`` int8 [B, n + 1, W]; without ``PRUNE``, every class with a pair), its pairs
    ``TP`` at a time, and -- ``ALLOWED`` -- that the sample allows there (``allowed`` int8 [Be, n, C], ``a_stride`` a
    sample's flags apart; a class it rules out is not walked) -- into ``row[m, c]`` (0 live, -1 dead). ``KEYS`` (the
    rows shared, see :class:`ClassWeightRows`): every live class's key (:func:`_row_key`; a dead one's
    :data:`_KEY_NONE`) into ``key[m, c]`` and its live pairs' number into ``elen[m, c]`` -- the rows are counted once
    the classes are grouped (:func:`_class_reps_kernel`); else the live classes' rows' floats (``col_K[m]`` a row)
    added to ``floats[m]``, and the ``N`` columns' first floats into ``off`` (:func:`_offsets_when_counted`)."""
    m = tl.program_id(0).to(tl.int64)
    t = tl.load(col_pos + m // B)
    b = m % B
    c = tl.program_id(1) * TC + tl.arange(0, TC)
    cm = c < C
    if ALLOWED:
        cm = cm & (tl.load(allowed + b * a_stride + t * C + c, mask = cm, other = 0) != 0)
    p0 = tl.load(cls_ptr + t * (C + 1) + c, mask = cm, other = 0)
    p1 = tl.load(cls_ptr + t * (C + 1) + c + 1, mask = cm, other = 0)
    h = tl.zeros([TC], dtype = tl.uint64)
    cnt = tl.zeros([TC], dtype = tl.int32)
    if PRUNE or KEYS:
        hit = tl.zeros([TC], dtype = tl.int32)
        for k0 in range(0, tl.max(p1 - p0, axis = 0), TP):
            k = p0[:, None] + k0 + tl.arange(0, TP)[None, :]                     # [TC, TP]: the classes' pairs
            pm = cm[:, None] & (k < p1[:, None])
            i = tl.load(cls_col + k, mask = pm, other = 0)
            q = tl.load(cls_succ + k, mask = pm, other = 0)
            if PRUNE:
                a = tl.load(live + (b * (n + 1) + t) * W + i, mask = pm, other = 0)
                z = tl.load(live + (b * (n + 1) + t + 1) * W + q, mask = pm, other = 0)
                lp = pm & (a != 0) & (z != 0)
            else:
                lp = pm
            hit = hit | tl.max(lp.to(tl.int32), axis = 1)
            if KEYS:
                h += tl.sum(tl.where(lp, _pair_hash(i, q, W), tl.zeros_like(h)[:, None]), axis = 1)
                cnt += tl.sum(lp.to(tl.int32), axis = 1)
        on = cm & (hit != 0)
    else:
        on = cm & (p1 > p0)
    tl.store(row + m * C + c, tl.where(on, 0, -1), mask = c < C)
    if KEYS:
        st = tl.load(stored + c, mask = c < C, other = -1) >= 0
        tl.store(key + m * C + c, tl.where(on, _row_key(h, cnt, st, key_mask), _KEY_NONE), mask = c < C)
        tl.store(elen + m * C + c, cnt, mask = c < C)
    else:
        tl.atomic_add(floats + m, tl.sum(on.to(tl.int64), axis = 0) * tl.load(col_K + m))
        _offsets_when_counted(floats, off, N)


_OFFSET_STEP = tl.constexpr(1024)


@triton.jit
def _offsets_when_counted(floats, off, N):
    """Once every program of the launch added its column's floats to ``floats[:N]`` -- counted at ``floats[N]`` (0
    before the launch) --, the last of them writes their running sums, every column's first float past the first,
    into ``off[1:]`` (``off[0]`` 0 already), 1024 columns a step, read from L2 (where the others' atomics are)."""
    tl.debug_barrier()                                                  # (this program's floats added)
    if tl.atomic_add(floats + N, 1) == tl.num_programs(0) * tl.num_programs(1) - 1:
        total = tl.zeros([], dtype = tl.int64)
        for s0 in range(0, N, _OFFSET_STEP):
            s = s0 + tl.arange(0, _OFFSET_STEP)
            v = tl.load(floats + s, mask = s < N, other = 0, cache_modifier = ".cg")
            tl.store(off + 1 + s, total + tl.cumsum(v, axis = 0), mask = s < N)
            total += tl.sum(v, axis = 0)


#: classes a step of :func:`_class_candidates_kernel` (one program per target column, every class of its boundary)
CLASS_CANDIDATE_TILE = 1024
#: the most classes (padded to a power of 2) whose candidates :func:`_sorted_candidates_kernel` finds, a column's keys
#: sorted in registers -- one launch for the three of :func:`_ordered_keys_kernel`, a sort and
#: :func:`_class_candidates_kernel` (~27 us of host time, 5 on its own) --, at any number of target columns up to
#: ``CLASS_CANDIDATE_ANY`` of them, past it at most ``CLASS_CANDIDATE_COLUMNS`` (None: the device's SMs, one wave).
#: Device time, the sort's vs the registers' (brezel): 100 classes x 8192 columns 99 vs 17 us, 500 x 8192 318 vs 119;
#: 617 x 1024 67 vs 60, x 8192 373 vs 406; 1224 x 1 42 vs 39, x 1024 120 vs 172; 3000 x 1 54 vs 88
CLASS_CANDIDATE_REGISTERS = 2048
CLASS_CANDIDATE_ANY = 512
CLASS_CANDIDATE_COLUMNS = None


@triton_jit
def _ordered_keys_kernel(key, cls_order, col_pos, out, C, B, TC: tl.constexpr):
    """One program per target column ``m`` and tile of ``TC`` ranks: ``out[m, r] = key[m, cls_order[t, r]]``, the
    column's keys in its boundary's class order (``t = col_pos[m // B]``)."""
    m = tl.program_id(0).to(tl.int64)
    t = tl.load(col_pos + m // B)
    r = tl.program_id(1) * TC + tl.arange(0, TC)
    rm = r < C
    c = tl.load(cls_order + t * C + r, mask = rm, other = 0)
    tl.store(out + m * C + r, tl.load(key + m * C + c, mask = rm), mask = rm)


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton_jit
def _class_candidates_kernel(keys, sidx, cls_order, col_pos, rep, C, B, TC: tl.constexpr):
    """One program per target column ``m``: from its keys in class order sorted, stably (``keys``, and ``sidx`` their
    ranks), every class's candidate -- the first class in the boundary's order (``cls_order[t]``) with its key --
    into ``rep[m, c]`` (int32), ``TC`` sorted keys a step: a key's first place carried over the steps."""
    m = tl.program_id(0).to(tl.int64)
    t = tl.load(col_pos + m // B)
    order, row = cls_order + t * C, m * C
    first = tl.zeros([], dtype = tl.int32)                         # the last key's first place so far
    for p0 in range(0, C, TC):
        p = p0 + tl.arange(0, TC)
        pm = p < C
        k = tl.load(keys + row + p, mask = pm, other = 0)
        prev = tl.load(keys + row + p - 1, mask = pm & (p > 0), other = 0)
        start = tl.where(k != prev, p, 0)                            # (a key's first place; 0's anyway)
        start = tl.maximum(tl.associative_scan(start, 0, _max_combine), first)
        first = tl.max(tl.where(pm, start, 0), axis = 0)
        r = tl.load(sidx + row + p, mask = pm, other = 0)
        cand = tl.load(sidx + row + start, mask = pm, other = 0)
        tl.store(rep + row + tl.load(order + r, mask = pm, other = 0),
                 tl.load(order + cand, mask = pm, other = 0), mask = pm)


@triton_jit
def _sorted_candidates_kernel(key, cls_order, col_pos, rep, C, B, CP: tl.constexpr, RB: tl.constexpr):
    """:func:`_class_candidates_kernel`'s candidates from the keys themselves, one program per target column ``m``: its
    ``C <= CP`` keys in class order with their ranks below them (``RB`` bits, ``2^RB = CP``; a live key of at most
    :data:`_KEY_BITS` bits, a dead one's :data:`_KEY_NONE` after every live one) sorted in registers -- the order of the
    stable sort --, every class's candidate the class at its key's first place."""
    m = tl.program_id(0).to(tl.int64)
    t = tl.load(col_pos + m // B)
    order, row = cls_order + t * C, m * C
    r = tl.arange(0, CP)
    rm = r < C
    k = tl.load(key + row + tl.load(order + r, mask = rm, other = 0), mask = rm, other = 0)
    k = tl.where(k == _KEY_NONE, 1 << _KEY_BITS, k)
    s = tl.sort(tl.where(rm, (k << RB) | r.to(tl.int64), _KEY_NONE))      # (past C: last)
    sk, sr = s >> RB, (s & (CP - 1)).to(tl.int32)
    start = tl.where((r == 0) | (sk != tl.gather(sk, tl.maximum(r - 1, 0), 0)), r, 0)      # (a key's first place)
    start = tl.associative_scan(start, 0, _max_combine)
    tl.store(rep + row + tl.load(order + sr, mask = rm, other = 0),
             tl.load(order + tl.gather(sr, start, 0), mask = rm, other = 0), mask = rm)


@triton_jit
def _class_reps_kernel(live, cls_ptr, cls_col, cls_succ, col_pos, col_K, row, rep, elen, stored, floats, off, next_col,
                       delta, state_id, col_of_state, n, W, C, B, K, N, TC: tl.constexpr, TP: tl.constexpr,
                       PRUNE: tl.constexpr, HOMOGENEOUS: tl.constexpr):
    """One program per target column ``m`` and tile of ``TC`` classes: every live class's representative -- the
    candidate ``rep[m, c]`` its key proposed (the first class of the same key in the boundary's class order), if their
    live pairs are the same, else itself -- into ``rep[m, c]`` (-1 for a dead class), and the column's
    representatives' rows' floats (``col_K[m]`` a row) added to ``floats[m]`` (the ``N`` columns' first floats then
    into ``off``, :func:`_offsets_when_counted`). The same live pairs: as many
    (``elen``), and each of the class's taken the same way by the candidate -- its column live at ``t``, its
    successor live at ``t + 1`` and the candidate's successor there (``next_col`` / ``delta``, see
    :func:`~.common._next_col`; the sequence end's one column) --, and the same kind (both stored, or both
    enumerated: ``stored``): so a key shared by different lists (a collision) merges nothing. The class's pairs ``TP``
    at a time, as :func:`_live_classes_kernel` walks them."""
    m = tl.program_id(0).to(tl.int64)
    t = tl.load(col_pos + m // B)
    b = m % B
    c = tl.program_id(1) * TC + tl.arange(0, TC)
    cm = c < C
    on = cm & (tl.load(row + m * C + c, mask = cm, other = -1) >= 0)
    r = tl.load(rep + m * C + c, mask = on, other = 0)
    dup = on & (r != c)
    bad = dup & (tl.load(elen + m * C + c, mask = dup, other = 0) != tl.load(elen + m * C + r, mask = dup, other = 0))
    bad = bad | (dup & ((tl.load(stored + c, mask = dup, other = -1) >= 0) != (tl.load(stored + r, mask = dup, other = -1)
                                                                              >= 0)))
    p0 = tl.load(cls_ptr + t * (C + 1) + c, mask = dup, other = 0)
    p1 = tl.load(cls_ptr + t * (C + 1) + c + 1, mask = dup, other = 0)
    end = t + 1 == n
    for k0 in range(0, tl.max(p1 - p0, axis = 0), TP):                       # (no candidate to check: no walk)
        k = p0[:, None] + k0 + tl.arange(0, TP)[None, :]
        pm = dup[:, None] & (k < p1[:, None])
        i = tl.load(cls_col + k, mask = pm, other = 0)
        q = tl.load(cls_succ + k, mask = pm, other = 0)
        if PRUNE:
            a = tl.load(live + (b * (n + 1) + t) * W + i, mask = pm, other = 0)
            z = tl.load(live + (b * (n + 1) + t + 1) * W + q, mask = pm, other = 0)
            lp = pm & (a != 0) & (z != 0)
        else:
            lp = pm
        qr = _next_col(next_col, delta, state_id, col_of_state, t, i, r[:, None], lp, K, W, C, HOMOGENEOUS)
        qr = tl.where(end & (qr >= 0), 0, qr)                                 # (the sequence end: its one column)
        bad = bad | (tl.max((lp & (qr != q)).to(tl.int32), axis = 1) != 0)
    fin = tl.where(on, tl.where(bad, c, r), -1)
    tl.store(rep + m * C + c, fin, mask = cm)
    tl.atomic_add(floats + m, tl.sum((on & (fin == c)).to(tl.int64), axis = 0) * tl.load(col_K + m))
    _offsets_when_counted(floats, off, N)


@triton_jit
def _class_rows_kernel(off, cls_order, col_pos, col_K, col_first, group_first, row, rep, count, cls, sizes, N, G, C, B,
                       TC: tl.constexpr, SHARED: tl.constexpr):
    """One program per target column ``m`` (from ``off``, every column's first float): its live classes (``row[m, c]
    >= 0``) -- ``SHARED``: the representatives among them (``rep[m, c] == c``) -- in the boundary's class order, into
    ``cls[m, :count[m]]``, and their rows among their row group's (``col_first[m]`` its first column) into ``row[m]``;
    ``SHARED``: every other live class its representative's row (a representative comes first in the class order: its
    row is written, in this tile or an earlier one, before it is read); the most rows of a column into ``sizes[G +
    1]`` (0 already) -- and program ``g <= G`` the ``g``-th row group's first float (``group_first``) into
    ``sizes[g]``."""
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
            if SHARED:
                rp = tl.load(rep + m * C + c, mask = on, other = -1)
                own = on & (rp == c)
            else:
                own = on
            v = own.to(tl.int32)
            r = kept + tl.cumsum(v, axis = 0) - v
            tl.store(row + m * C + c, (local + r).to(tl.int32), mask = own)
            tl.store(cls + m * C + r, c, mask = own)
            kept += tl.sum(v, axis = 0)
            if SHARED:
                tl.debug_barrier()                                            # (the representatives' rows written)
                dup = on & ~own
                tl.store(row + m * C + c, tl.load(row + m * C + rp, mask = dup, other = 0), mask = dup)
        tl.atomic_max(sizes + G + 1, kept.to(tl.int64))


def class_weight_rows(prog, liveness, col_pos: torch.Tensor, col_K: torch.Tensor, col_first: torch.Tensor,
                      group_first: torch.Tensor, B: int, stored: Optional[torch.Tensor] = None) -> ClassWeightRows:
    """
    The :class:`ClassWeightRows` of a query's target columns (the row groups' consecutive, a position's ``B``
    samples consecutive in a row group's), from its :class:`~.live.Liveness` (None: every column live). Nothing
    waits for the device: the row groups' sizes and the most rows of a column are copied as the rows are built, and
    read on first use (:attr:`ClassWeightRows.most`).

    With ``stored`` (int32 [C] on the device: every class's stored column, -1 for an enumerated class) and
    :data:`CLASS_ROWS_SHARED`, the live classes of a column with the same live pairs share a row: every class's key
    (:func:`_live_classes_kernel`), the classes of a key grouped per column in the boundary's class order (a sort),
    every class checked exactly against the first of its key (:func:`_class_reps_kernel`), then the rows -- the
    grouping in registers where it fits (:data:`CLASS_CANDIDATE_REGISTERS`).

    :param col_pos: int64 [N / B] on the device, every block of ``B`` columns' position
    :param col_K, col_first: int64 [N] on the device, every column's row group's rows and first column
    :param group_first: int64 [row groups + 1] on the device, every row group's first column, then ``N``
    """
    tt, n, W, C = prog.trans, prog.n, prog.W, prog.C
    N, G, dev = col_K.numel(), group_first.numel() - 1, prog.dev
    shared = CLASS_ROWS_SHARED and stored is not None and N > 0
    M, a, b = max(N, 1), N + 1 + (N + 1) % 2, G + 2 + G % 2                 # (pieces 16 B aligned)
    z = torch.zeros(a + b + M + 1, dtype = torch.int64, device = dev)      # (the floats' programs counted past them)
    off, sizes, floats = z.as_strided((N + 1,), (1,)), z.as_strided((G + 2,), (1,), a), z.as_strided((M,), (1,), a + b)
    NC, MP = M * C + (-M * C) % 4, M + (-M) % 4
    e = torch.empty((4 if shared else 2) * NC + MP, dtype = torch.int32, device = dev)
    cls, row, count = e.as_strided((M, C), (C, 1)), e.as_strided((M, C), (C, 1), NC), e.as_strided((M,), (1,), 2 * NC)
    if shared:                                      # every live class's representative, and its live pairs' number
        rep, elen = e.as_strided((M, C), (C, 1), 2 * NC + MP), e.as_strided((M, C), (C, 1), 3 * NC + MP)
        key = torch.empty(M, C, dtype = torch.int64, device = dev)
    else:
        rep, elen, key = row, row, floats
    if N > 0:                                       # (the last launch adding floats writes the columns' offsets)
        prune = liveness is not None
        classes = prune and liveness.allowed is not None                    # (a session's per-sample classes)
        a_stride = n * C if classes and liveness.allowed.size(0) > 1 else 0  # (one set: every sample's)
        TC = min(LIVE_CLASS_TILE, host_next_power_of_2(C))
        while (prune or shared) and TC > 1 and N * host_cdiv(C, TC) < 4 * _num_sms(dev):   # (programs filling the
            TC //= 2                                                                         # device)
        TP = max(1, LIVE_CLASS_LANES // TC)
        _live_classes_kernel[(N, host_cdiv(C, TC))](liveness.live if prune else row, tt.cls_ptr, tt.cls_col,
                                                     tt.cls_succ, col_pos, col_K, row, floats, off, key, elen,
                                                     stored if shared else row, n, W, C, B, N,
                                                     liveness.allowed if classes else row, a_stride, CLASS_ROW_KEY_MASK,
                                                     TC = TC, TP = TP, PRUNE = prune, ALLOWED = classes, KEYS = shared,
                                                     num_warps = 4)
        if shared:                                  # every class's candidate: the first of its key in class order
            CP = max(2, host_next_power_of_2(C))
            most = _num_sms(dev) if CLASS_CANDIDATE_COLUMNS is None else CLASS_CANDIDATE_COLUMNS
            if CP <= CLASS_CANDIDATE_REGISTERS and (CP <= CLASS_CANDIDATE_ANY or N <= most) and \
                    CLASS_ROW_KEY_MASK < (1 << _KEY_BITS.value):            # (8 warps the faster at 1024 only)
                _sorted_candidates_kernel[(N,)](key, tt.cls_order, col_pos, rep, C, B, CP = CP,
                                                RB = CP.bit_length() - 1, num_warps = 8 if CP == 1024 else 4)
            else:
                ordered = torch.empty(N, C, dtype = torch.int64, device = dev)
                TK = min(CLASS_CANDIDATE_TILE, host_next_power_of_2(C))
                _ordered_keys_kernel[(N, host_cdiv(C, TK))](key, tt.cls_order, col_pos, ordered, C, B, TC = TK)
                keys, sidx = torch.sort(ordered, dim = 1, stable = True)
                _class_candidates_kernel[(N,)](keys, sidx, tt.cls_order, col_pos, rep, C, B, TC = TK, num_warps = 4)
            _class_reps_kernel[(N, host_cdiv(C, TC))](liveness.live if prune else row, tt.cls_ptr, tt.cls_col,
                                                      tt.cls_succ, col_pos, col_K, row, rep, elen, stored, floats, off,
                                                      prog.next_col_arg, prog.delta, prog.state_id, prog.col_of_state,
                                                      n, W, C, B, prog.K, N, TC = TC, TP = TP, PRUNE = prune,
                                                      HOMOGENEOUS = prog.homogeneous, num_warps = 4)
    TC = min(LIVE_CLASS_TILE, host_next_power_of_2(C))
    _class_rows_kernel[(max(N, G + 1),)](off, tt.cls_order, col_pos, col_K, col_first, group_first, row, rep, count,
                                         cls, sizes, N, G, C, B, TC = TC, SHARED = shared)
    host = torch.empty(G + 2, dtype = torch.int64, pin_memory = True)
    host.copy_(sizes, non_blocking = True)
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(dev.index))
    return ClassWeightRows(count = count, cls = cls, off = off, row = row, _host = host, _done = done, _groups = G,
                           rep = rep if shared else None)
