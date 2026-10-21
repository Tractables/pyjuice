"""
Transition masses: for a missing token at a position, a column and one of its successors, the log-mass of the token
classes that lead there, from the class masses (:func:`transition_masses`). The products' ``input @ block`` and
``block @ input`` steps read them, so a missing token costs a gather per successor, not per class (see
:mod:`.prod`). Their tables (:class:`TransitionTables`) are built once per backend.

ENUMERATED CLASSES (see :class:`~pyjuice.constraints.distributions.TokenClasses`) have no class mass: where a kernel
read the mass of class ``c`` at row ``r`` it reads a CODE instead (:class:`EnumeratedTokens`): the class's stored
column, or ``-1 - e`` for the ``e``-th enumerated token live at the boundary, whose mass is its token's own
probability under the row's parameter row (``eprobs[prow[r], index]``, the enumerated tokens' probabilities, by
parameter row: a step's lanes read along one row, see :meth:`~..state.State.enumerated_probs`) times its per-sample
weight (``enum_logw``, see :class:`~..state.ClassMasses`) over the row's ``Z``.
"""

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ....distributions.categorical import _exact_dot
from .common import _lse_step, _next_col_of, grid_chunks



@dataclass(frozen = True)
class EnumeratedTokens:
    """
    The enumerated classes' tokens that are live at every boundary (lead somewhere from some active column), and
    every class's CODE at every boundary: its stored column, or ``-1 - e`` for the ``e``-th of the boundary's live
    enumerated tokens (a class dead at the boundary, enumerated: -1 - the count, matching no entry).

    :ivar ptr: [n + 1] int32, the boundary's tokens are ``tok[ptr[t] .. ptr[t + 1]]``; ``ptr_host`` the same on the host
    :ivar tok: [num + 1] int32, the tokens, then a 0 (kernels read a boundary's first token unmasked for a stored
        entry: past the last boundary's, this one)
    :ivar index: [num + 1] int32, every token's place among the automaton's enumerated tokens
        (``TokenClasses.enumerated_tokens``, in class order) -- its column of the enumerated tokens' probabilities
        (:meth:`~..state.State.enumerated_probs`) --, then a 0
    :ivar code: [n, C] int32
    :ivar num: the tokens over all boundaries (0: nothing enumerated; every code a stored column, the class itself)
    """

    ptr: torch.Tensor
    tok: torch.Tensor
    index: torch.Tensor
    code: torch.Tensor
    num: int
    ptr_host: np.ndarray


def enumerated_tokens(tr, classes) -> EnumeratedTokens:
    """The :class:`EnumeratedTokens` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`, on
    the device) under ``classes`` (a :class:`~pyjuice.constraints.distributions.TokenClasses`)."""
    n, C, dev = tr.n, tr.C, tr.device
    stored = classes.stored_of_class.to(dev).long()
    etok = classes.enumerated_token.to(dev).long()
    code = stored.expand(n, C).clone() if n > 0 else torch.zeros(0, C, dtype = torch.long, device = dev)
    ptr = torch.zeros(n + 1, dtype = torch.long)
    place = torch.cumsum(etok >= 0, dim = 0) - 1                             # an enumerated class's among them
    toks, places = [], []
    if classes.num_enumerated > 0:
        for t in range(n):
            live = (tr.at(t) >= 0).any(dim = 0) & (etok >= 0)                # enumerated classes live at t
            ids = torch.nonzero(live).view(-1)
            e = torch.full((C,), ids.numel(), dtype = torch.long, device = dev)   # (dead: the count, past the last)
            e[ids] = torch.arange(ids.numel(), device = dev)
            code[t] = torch.where(etok >= 0, -1 - e, stored)
            ptr[t + 1] = ptr[t] + ids.numel()
            toks.append(etok[ids])
            places.append(place[ids])
    zero = torch.zeros(1, dtype = torch.long, device = dev)
    tok = torch.cat(toks + [zero])
    index = torch.cat(places + [zero])
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    return EnumeratedTokens(ptr = i32(ptr), tok = i32(tok), index = i32(index), code = i32(code), num = int(ptr[-1]),
                            ptr_host = ptr.numpy())


@dataclass(frozen = True)
class TransitionTables:
    """
    Every boundary's transitions as entries (column, successor, mass), the successor of a sequence end being its one
    column. Ungrouped, one entry per (column, class) that leads somewhere, its mass the class's mass -- read by the
    class's code (:class:`EnumeratedTokens`). Grouped, one per (column, successor) pair, its mass the pair's
    transition mass: the mass of every class that leads there (:func:`transition_masses`) -- a missing token then
    costs a gather per successor, not per class.

    :ivar grouped: which of the two
    :ivar entry_ptr: [n, W + 1] int32, the entries of column ``q`` at ``t``: ``entry_ptr[t, q] .. entry_ptr[t, q + 1]``
        (global numbers; ``entry_ptr[t, 0]``: the boundary's first)
    :ivar entry_succ, entry_x: [entries] int32, every entry's successor column and where its mass is: its class's
        code, or its pair's number within the boundary
    :ivar pred_ptr, pred_col, pred_x: the entries leading to column ``j`` at ``t + 1``, as a CSR (``pred_ptr``
        [n, W + 1]) into their columns and their mass indices
    :ivar num_pairs: grouped only, pairs per boundary (Python ints)
    :ivar max_pairs: grouped only, the most pairs of any column
    :ivar enum_ptr, enum_e, enum_pair: grouped only, the enumerated classes' live entries of every column, as a CSR
        (``enum_ptr`` [n, W + 1] int32, global numbers): the token's number ``e`` at its boundary and the pair it leads
        to; ``max_enum`` the most of any column
    :ivar cls_ptr, cls_col, cls_succ: the same (column, class) entries BY CLASS: the columns where class ``c`` leads
        somewhere at ``t``, ascending, ``cls_col[cls_ptr[t, c] .. cls_ptr[t, c + 1]]`` (``cls_ptr`` [n, C + 1] int32,
        global numbers), and their successors (a sequence end's: its one column) -- the class weights' walk over a
        class's columns (:func:`~.score._class_weight_kernel`); ``max_cls`` the most of any (boundary, class);
        ``cls_order`` [n, C] int32, every boundary's classes by their columns, most first (the walk's lanes take
        classes in this order, so that a program's classes have lists of a length)
    :ivar entry_class, pred_class: ungrouped only, [entries] int32: every entry's class, in the entry and the
        predecessor order -- by which a session's per-sample inputs rule entries out (the ``input @ block`` entry loops'
        :class:`~.live.SuccessorLists`, the CUDA top-down walk's terms)
    :ivar boundary_base, num_entries: boundaries alike share their entries (see :func:`transition_tables`), so a
        structure with a value PER BOUNDARY AND ENTRY -- a sample's successor lists -- counts the boundaries' entries
        one after the other: ``boundary_base`` int64 [n + 1] on the device, boundary ``t``'s entries are
        ``boundary_base[t] .. boundary_base[t + 1]`` there (its entry ``e`` at ``boundary_base[t] + e - entry_ptr[t, 0]``),
        ``num_entries`` their total, ``boundary_base[n]``
    """

    grouped: bool
    entry_ptr: torch.Tensor
    entry_succ: torch.Tensor
    entry_x: torch.Tensor
    pred_ptr: torch.Tensor
    pred_col: torch.Tensor
    pred_x: torch.Tensor
    num_pairs: list
    max_pairs: int
    enum_ptr: Optional[torch.Tensor] = None
    enum_e: Optional[torch.Tensor] = None
    enum_pair: Optional[torch.Tensor] = None
    max_enum: int = 0
    cls_ptr: Optional[torch.Tensor] = None
    cls_col: Optional[torch.Tensor] = None
    cls_succ: Optional[torch.Tensor] = None
    max_cls: int = 0
    cls_order: Optional[torch.Tensor] = None
    entry_class: Optional[torch.Tensor] = None
    pred_class: Optional[torch.Tensor] = None
    boundary_base: Optional[torch.Tensor] = None
    num_entries: int = 0


def transition_stats(tr) -> Tuple[int, int, int]:
    """``(live, columns, most)`` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`): how many
    (column, class) pairs of the boundaries lead somewhere, how many columns the boundaries have, and the most distinct
    columns one column reaches in one token at any boundary but the last (a sequence end being one column; at least
    1) -- one pass over the boundaries' tables."""
    n, W = tr.n, tr.W
    live = columns = 0
    most = 1
    for t in range(n):
        nc = tr.at(t)
        alive = nc >= 0
        live += int(alive.sum())
        columns += nc.size(0)
        if t + 1 < n and nc.numel():
            q, c = torch.nonzero(alive, as_tuple = True)
            if q.numel():
                pairs = torch.unique(q * (W + 1) + nc[q, c])
                most = max(most, int(torch.bincount(pairs // (W + 1)).max()))
    return live, columns, most


def transition_tables(tr, grouped: bool, enum: Optional[EnumeratedTokens] = None) -> TransitionTables:
    """
    The :class:`TransitionTables` of a layout's transitions ``tr`` (a :class:`~..compiled.layout.Transitions`, on the
    device); ``enum`` the :class:`EnumeratedTokens` (None: nothing enumerated, every class's code the class itself).

    A boundary's entries depend on its possible states and the next boundary's alone (its columns, their successors'
    columns, the classes live there -- an enumerated token's code counts from the boundary's first), and on its being
    the sequence end. Boundaries alike SHARE their entries: one segment, built once, every one of them pointing into it
    (``entry_ptr[t]``, ``pred_ptr[t]``, ``cls_ptr[t]``, ``enum_ptr[t]``). Decoding at a length well past the shortest
    accepted string, most boundaries are alike: a JSON Schema automaton of 24k states has 81 distinct of 1024.
    """
    n, W, C, dev = tr.n, tr.W, tr.C, tr.device
    code = enum.code.long() if enum is not None and enum.num > 0 else None
    arange = lambda k: torch.arange(k, dtype = torch.int64, device = dev)
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    cat = lambda xs, total: i32(torch.cat(xs)) if total > 0 else torch.zeros(1, dtype = torch.int32, device = dev)

    sid = tr.state_id.cpu().numpy()                                          # the boundaries alike, by their states
    digests = [hashlib.blake2b(sid[t, :tr.width[t]].tobytes(), digest_size = 16).digest() for t in range(n + 1)]
    number, group, firsts = {}, [], []                                       # a key's number; every boundary's; its first
    for t in range(n):
        key = (digests[t], digests[t + 1], t + 1 == n)
        if key not in number:
            number[key] = len(firsts)
            firsts.append(t)
        group.append(number[key])
    G = len(firsts)
    at = torch.tensor(group, dtype = torch.long, device = dev)

    succs, xs, cols, pxs, pairs, ees, epairs, ccols, csuccs, clss, pclss = ([] for _ in range(11))
    entry_rows, pred_rows, enum_rows, cls_rows, orders = ([] for _ in range(5))
    sizes, csizes, esizes = [], [], []
    max_pairs = max_enum = max_cls = 0
    for t in firsts:                                                          # one segment per distinct boundary
        succ = tr.at(t)                                                      # [wt, C]
        if t + 1 == n:
            succ = torch.where(succ >= 0, 0, -1)                             # a sequence end: one column
        q, c = torch.nonzero(succ >= 0, as_tuple = True)                     # by column, then class
        by_cls = torch.argsort(c * (W + 1) + q, stable = True)               # by class, then column
        cls_rows.append(torch.searchsorted(c[by_cls], arange(C + 1)))
        ccols.append(q[by_cls]); csuccs.append(succ[q, c][by_cls])
        csizes.append(q.numel())
        lengths = torch.bincount(c, minlength = C)
        orders.append(torch.argsort(-lengths, stable = True))
        max_cls = max(max_cls, int(lengths.max()) if q.numel() else 0)
        if grouped:
            key, inverse = torch.unique(q * (W + 1) + succ[q, c], return_inverse = True)    # by column, successor
            eq, es, ex = key // (W + 1), key % (W + 1), arange(key.numel())
            pairs.append(key.numel())
            max_pairs = max(max_pairs, int(torch.bincount(eq, minlength = 1).max()) if key.numel() else 0)
            if code is not None:                                             # the enumerated classes' live entries
                en = code[t, c] < 0
                enum_rows.append(torch.searchsorted(q[en], arange(W + 1)))
                ees.append(-1 - code[t, c[en]]); epairs.append(inverse[en])
                esizes.append(int(en.sum()))
                max_enum = max(max_enum, int(torch.bincount(q[en], minlength = 1).max()) if en.any() else 0)
        else:
            eq, es, ex = q, succ[q, c], (c if code is None else code[t, c])
            clss.append(c)
        entry_rows.append(torch.searchsorted(eq, arange(W + 1)))
        by_succ = torch.argsort(es * (W + 1) + eq, stable = True)
        pred_rows.append(torch.searchsorted(es[by_succ], arange(W + 1)))
        succs.append(es); xs.append(ex); cols.append(eq[by_succ]); pxs.append(ex[by_succ])
        if not grouped:
            pclss.append(c[by_succ])
        sizes.append(eq.numel())

    def pointers(rows, counts):
        """[n, width] int32: every boundary's pointer row, its segment's relative row past the segments before."""
        if not rows:
            return torch.zeros(n, W + 1, dtype = torch.int32, device = dev)
        bases = torch.tensor([0] + list(np.cumsum(counts)[:-1]), dtype = torch.int64, device = dev)
        return i32(torch.stack(rows)[at] + bases[at][:, None])

    base, cbase, ebase = sum(sizes), sum(csizes), sum(esizes)
    enumerated = grouped and code is not None
    per_boundary = np.concatenate([[0], np.cumsum([sizes[g] for g in group])]).astype(np.int64)
    return TransitionTables(grouped = grouped, entry_ptr = pointers(entry_rows, sizes), entry_succ = cat(succs, base),
                            entry_x = cat(xs, base), pred_ptr = pointers(pred_rows, sizes), pred_col = cat(cols, base),
                            pred_x = cat(pxs, base), num_pairs = [pairs[g] for g in group] if grouped else [],
                            max_pairs = max_pairs,
                            enum_ptr = pointers(enum_rows, esizes) if enumerated else (pointers([], []) if grouped else None),
                            enum_e = cat(ees, ebase) if grouped else None,
                            enum_pair = cat(epairs, ebase) if grouped else None, max_enum = max_enum,
                            cls_ptr = pointers(cls_rows, csizes) if G else
                            torch.zeros(n, C + 1, dtype = torch.int32, device = dev),
                            cls_col = cat(ccols, cbase), cls_succ = cat(csuccs, cbase), max_cls = max_cls,
                            cls_order = i32(torch.stack(orders)[at]) if G else i32(arange(C).repeat(n, 1)),
                            entry_class = None if grouped else cat(clss, base),
                            pred_class = None if grouped else cat(pclss, base),
                            boundary_base = torch.from_numpy(per_boundary).to(dev), num_entries = int(per_boundary[-1]))


#: :func:`transition_masses`' tiles: the most rows (input steps at one position) and columns per program, classes per
#: product, warps. Rows and columns shrink until there are enough programs (they never change a result); the pairs
#: per product follow the widest column, up to 64.
TRANSITION_TILES = dict(TS = 64, QT = 4, TC = 16, warps = 4, programs = 1024)


def transition_job(rows: torch.Tensor, dev):
    """
    The steps whose transition masses one :func:`transition_masses` call builds: ``rows`` [N, 3] (input row,
    position, the step's rank among its position's -- which block of the boundary's pairs, or of a sample's entries
    there, its masses are), as ``(rows, tiles, TS)`` on ``dev``, the rows ordered by position and ``tiles``
    [tiles, 3] (first row, number of rows, position) cutting them into tiles of at most ``TS`` rows at one position
    (fewer when there are few rows, so that there are enough programs).
    """
    TS = min(TRANSITION_TILES["TS"], max(16, host_next_power_of_2(rows.size(0)) // 32))
    rows = rows[torch.argsort(rows[:, 1], stable = True)]
    pos, counts = torch.unique_consecutive(rows[:, 1], return_counts = True)
    tiles, start = [], 0
    for t, k in zip(pos.tolist(), counts.tolist()):
        tiles += [(start + i, min(TS, k - i), t) for i in range(0, k, TS)]
        start += k
    i32 = lambda x: torch.as_tensor(x, dtype = torch.int32).reshape(-1, 3).to(dev).contiguous()
    return i32(rows), i32(tiles), TS


@triton_jit
def _transition_kernel(class_mars, mass_row, mass_stride, trans, rows, tiles, delta, state_id, col_of_state,
                       entry_succ, entry_ptr, class_of_stored, enum_ptr, enum_e, enum_pair, tok_ptr, eidx, eprobs,
                       prow, lognorm, enum_logw, width, W, C, K, n, Cs, num_enum, logw_stride, input_start,
                       num_qtiles, pair_off, pairs, pid0, TS: tl.constexpr,
                       QT: tl.constexpr, TP: tl.constexpr, TC: tl.constexpr, TE: tl.constexpr, SAMPLE: tl.constexpr,
                       ENUM: tl.constexpr, LOGW: tl.constexpr):
    """
    One program per (tile of rows at one position, ``QT`` columns) -- ``SAMPLE``: and sample, for class masses of each
    sample's own (rows ``mass_row + b * mass_stride``; a row the same for every sample, stride 0, is built for sample
    0 alone; one set of masses is written either way): per column, its pairs ``TP`` at a time, as the product
    of the rows' class masses -- linear, scaled by 2^64, in column order over the ``Cs`` stored classes
    (``class_of_stored``: every column's class) -- with the 0/1 table of which of the pairs every class leads to: the
    class's successor column (the automaton's transitions ``delta`` through the boundaries' column maps ``state_id``
    and ``col_of_state``, a sequence end's being 0) against the pairs' (``entry_succ``, the boundary's entries being
    its pairs by column and successor). A class mass is a sum of fp32 probabilities, so 0 or at least 2^-149: scaled, a normal
    float, which tensor cores do not flush. ``ENUM``: then the column's enumerated entries (``enum_ptr`` / ``enum_e``
    / ``enum_pair``, see :class:`TransitionTables`), ``TE`` at a time, each the mass of its token: its probability
    under the row's parameter row (``eprobs[prow[row], eidx[e]]``, ``num_enum`` the table's row stride) times --
    ``LOGW`` -- its weight ``exp(enum_logw[b, e] - lognorm[row])`` (``logw_stride`` floats per sample). The log of the
    sum is the transition mass.
    """
    pid = pid0 + tl.program_id(0)
    tile = pid // num_qtiles
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = tl.arange(0, TS)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    rk = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)   # the step's rank at t: its
    off = tl.load(pair_off + t) + rk * tl.load(pairs + t)                                # block of the boundary's pairs
    row = tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    b = tl.program_id(1).to(tl.int64)
    if SAMPLE:
        st = tl.load(mass_stride + (u - input_start), mask = rmask, other = 0).to(tl.int64)
        rmask = rmask & ((b == 0) | (st > 0))
        row = row + b * st
    masses = class_mars + row * Cs
    if ENUM:
        pr = tl.load(prow + row, mask = rmask, other = 0).to(tl.int64)
        if LOGW:
            lz = tl.load(lognorm + row, mask = rmask, other = 0.0)
            lw0 = enum_logw + b * logw_stride + tl.load(tok_ptr + t)
        tk0 = eidx + tl.load(tok_ptr + t)
    base = tl.load(entry_ptr + t * (W + 1))                                  # the boundary's first pair
    jj = tl.arange(0, TP)
    ee = tl.arange(0, TE)
    for qi in range(QT):
        q = (pid % num_qtiles) * QT + qi
        if q < tl.load(width + t):
            g_end = tl.load(entry_ptr + t * (W + 1) + q + 1) - base
            sq = tl.load(state_id + t * W + q)                                                   # the column's state
            for g0 in range(tl.load(entry_ptr + t * (W + 1) + q) - base, g_end, TP):
                acc = tl.zeros([TS, TP], dtype = tl.float32)
                sg = tl.load(entry_succ + base + g0 + jj, mask = g0 + jj < g_end, other = -2)   # the pairs' successors
                for c0 in range(0, Cs, TC):
                    c = c0 + tl.arange(0, TC)
                    cmask = c < Cs
                    x = tl.exp(tl.load(masses[:, None] + c[None, :], mask = rmask[:, None] & cmask[None, :],
                                       other = float("-inf")) + 44.3614195558365)                # x 2^64
                    cls = tl.load(class_of_stored + c, mask = cmask, other = 0)
                    sc = _next_col_of(delta, col_of_state, sq, cls, cmask, t, K, C)             # the classes' successors
                    sc = tl.where((t + 1 == n) & (sc >= 0), 0, sc)
                    acc += _exact_dot(x, tl.where(sc[:, None] == sg[None, :], 1.0, 0.0))
                if ENUM:                                                     # the column's enumerated entries
                    e_end = tl.load(enum_ptr + t * (W + 1) + q + 1)
                    for e0 in range(tl.load(enum_ptr + t * (W + 1) + q), e_end, TE):
                        em = e0 + ee < e_end
                        e = tl.load(enum_e + e0 + ee, mask = em, other = 0).to(tl.int64)
                        g = tl.load(tk0 + e, mask = em, other = 0).to(tl.int64)
                        x = tl.load(eprobs + pr[:, None] * num_enum + g[None, :], mask = rmask[:, None] & em[None, :],
                                    other = 0.0) * 18446744073709551616.0                        # x 2^64
                        if LOGW:
                            x = x * tl.exp(tl.load(lw0 + e, mask = em, other = float("-inf"))[None, :] - lz[:, None])
                        g = tl.load(enum_pair + e0 + ee, mask = em, other = -1) - g0
                        acc += _exact_dot(x, tl.where(g[:, None] == jj[None, :], 1.0, 0.0))
                tl.store(trans + off[:, None] + (g0 + jj)[None, :], tl.log(acc) - 44.3614195558365,
                         mask = rmask[:, None] & (g0 + jj < g_end)[None, :])


def enumerated_args(prog, masses) -> dict:
    """The keyword arguments a kernel reading class masses by code takes (see :class:`EnumeratedTokens` and
    :class:`~..state.ClassMasses`): ``code, tok_ptr, eidx, eprobs, prow, lognorm, enum_logw, Cs, num_enum,
    logw_stride, ENUM, LOGW`` -- ``eidx`` the boundaries' tokens' columns of ``eprobs``, the enumerated tokens'
    probabilities by parameter row (``num_enum`` columns); unread placeholders where nothing is enumerated (``ENUM``
    False) or the masses are no sample's own (``LOGW`` False)."""
    en = prog.enum
    enum = en.num > 0
    logw = enum and masses.enum_logw is not None
    eprobs = masses.enum_probs if enum else masses.table
    return dict(code = en.code, tok_ptr = en.ptr, eidx = en.index, eprobs = eprobs, prow = masses.prow,
                lognorm = masses.lognorm if logw else masses.table,
                enum_logw = masses.enum_logw if logw else masses.table, Cs = masses.table.size(1),
                num_enum = eprobs.size(1), logw_stride = enum_logw_stride(masses), ENUM = enum, LOGW = logw)


def enum_logw_stride(masses) -> int:
    """How far apart the samples' rows of ``masses.enum_logw`` are: 0 when there is one row for every sample."""
    w = masses.enum_logw
    return 0 if w is None or w.size(0) == 1 else w.size(1)


def transition_masses(trans, job, masses, prog):
    """
    Write the transition masses of every step of ``job`` (from :func:`transition_job`): for each column ``q`` at the
    input's position ``t`` and each successor ``s``, ``log sum_{c: next_col[t, q, c] = s} exp(mass_r(c))`` (``r``:
    input ``u``'s row of the class masses ``masses``, a :class:`~..state.ClassMasses`, a stored class's mass from
    its table and an enumerated class's from its token; one set of masses for every sample -- a sample's own are
    :func:`sample_transition_masses`'), at ``trans[offset + p]`` for the pair's number ``p`` within the boundary,
    ``offset`` the step's block (``prog.pair_off[t] + rank * pairs[t]``).
    """
    rows, tiles, TS = job
    tt, cfg, en = prog.trans, TRANSITION_TILES, prog.enum
    W, C = prog.W, prog.C
    QT = cfg["QT"]
    while QT > 1 and tiles.size(0) * host_cdiv(max(prog.width), QT) < cfg["programs"]:
        QT //= 2
    num_qtiles = host_cdiv(max(prog.width), QT)
    TP = min(64, max(16, host_next_power_of_2(tt.max_pairs)))
    sample = masses.batch > 1
    stride = masses.stride if sample else masses.row
    enum = en.num > 0
    logw = enum and masses.enum_logw is not None
    eprobs = masses.enum_probs if enum else masses.table
    for first, size in grid_chunks(tiles.size(0) * num_qtiles, 0):
        _transition_kernel[(size, masses.batch)](
            masses.table, masses.row, stride, trans, rows, tiles, prog.delta, prog.state_id, prog.col_of_state,
            tt.entry_succ, tt.entry_ptr, prog.token_classes.class_of_stored32, tt.enum_ptr if enum else tt.entry_ptr,
            tt.enum_e if enum else tt.entry_ptr,
            tt.enum_pair if enum else tt.entry_ptr, en.ptr, en.index, eprobs, masses.prow,
            masses.lognorm if logw else masses.table, masses.enum_logw if logw else masses.table, prog.width_t, W, C,
            prog.K, prog.n, masses.table.size(1), eprobs.size(1), enum_logw_stride(masses), prog.input_start, num_qtiles,
            prog.pair_off, prog.pairs_t, first,
            TS = TS, QT = QT, TP = TP, TC = cfg["TC"], TE = min(64, max(16, host_next_power_of_2(max(tt.max_enum, 1)))),
            SAMPLE = sample, ENUM = enum, LOGW = logw, num_warps = cfg["warps"])


@dataclass
class SampleMasses:
    """
    Transition masses of each sample's own, over the sample's successor entries alone (:class:`~.live.SampleSuccessors`:
    the (column, successor) pairs some class the sample allows leads along) -- a session's soft evidence rules most
    classes out, so a sample's entries are few against the boundary's pairs. Sample ``b``'s masses of an input step at
    position ``t`` with rank ``r`` among the position's steps are the ``count[b, t]`` floats at ``base[b, t] + r *
    count[b, t]``, one per entry of its list at the boundary, in the list's order: the program entry ``g``'s at
    ``rank[b, g]`` (-1: none, see :attr:`~.live.SampleSuccessors.rank`).

    :ivar base: int64 [Be, n]; :ivar count: int32 [Be, n]; :ivar rank, ptr: the lists'
    :ivar cls_ptr, cls: every entry of the sample's lists (its number there, see ``ptr``) and the classes the sample
        allows that lead along it: ``cls[cls_ptr[b, e] .. cls_ptr[b, e + 1]]`` (``cls_ptr`` int32 [Be, E + 1] into the
        flat int32 ``cls``, one sample's after another's) -- what an entry's mass is the log-sum over (a session's
        inputs allow few of the hundreds of classes leading along a JSON automaton's entry)
    :ivar size: the floats of them all; :ivar Be: the samples with lists of their own (1: one set for every sample)
    :ivar version: the class masses' they were laid out for
    """

    base: torch.Tensor
    count: torch.Tensor
    rank: torch.Tensor
    ptr: torch.Tensor
    size: int
    Be: int
    version: int
    cls_ptr: Optional[torch.Tensor] = None                     # (built by sample_classes when the query takes the
    cls: Optional[torch.Tensor] = None                         # layout: a session over the budget never pays for them)


def sample_masses(masses, succ, prog) -> SampleMasses:
    """The :class:`SampleMasses` layout of ``masses`` (a :class:`~..state.ClassMasses`) over the successor lists
    ``succ`` (its :class:`~.live.SampleSuccessors`) for ``prog``'s input steps: one wait for the device (the size);
    the entries' classes come with :func:`sample_classes`."""
    Be, n, W = succ.ptr.size(0), prog.n, prog.W
    ptr = succ.ptr.view(Be, n, W + 1)
    width = prog.width_t[:n].long().view(1, n, 1).expand(Be, n, 1)
    count = (ptr.gather(2, width)[..., 0] - ptr[:, :, 0]).contiguous()               # the sample's entries at t
    sizes = count.long() * prog.pos_steps[None, :]                                    # floats per (sample, position)
    base = (torch.cumsum(sizes.reshape(-1), dim = 0) - sizes.reshape(-1)).view(Be, n)
    return SampleMasses(base = base, count = count, rank = succ.rank, ptr = succ.ptr, size = int(sizes.sum()),
                        Be = Be, version = masses.version)


def sample_classes(layout: SampleMasses, succ, prog) -> None:
    """Give ``layout`` every entry's allowed classes (``cls_ptr`` / ``cls``): the program's (entry, class) pairs
    (:func:`~.live.entry_classes`) whose class the sample allows (``succ.allowed``), by the entry's number in the
    sample's lists."""
    from .live import entry_classes
    Be, n, W, C, dev = layout.Be, prog.n, prog.W, prog.C, prog.dev
    ptr = succ.ptr.view(Be, n, W + 1)
    key, first = entry_classes(prog)                                                  # (entry, class) pairs by entry
    E = succ.rank.size(1)
    g = torch.repeat_interleave(torch.arange(E, device = dev), first[1:] - first[:-1])    # every pair's entry
    tb = torch.searchsorted(prog.succ.boundary_base[1:n + 1], g, right = True)        # ... its boundary
    cls_ptr = torch.empty(Be, E + 1, dtype = torch.int32, device = dev)
    classes, total = [], 0
    for b in range(Be):
        keep = succ.allowed.view(Be, n * C)[b, key] != 0                             # the sample allows the class
        e = ptr[b, :, 0].long()[tb[keep]] + succ.rank[b].long()[g[keep]]              # the entry's number in its lists
        cls_ptr[b, 0] = total
        cls_ptr[b, 1:] = total + torch.cumsum(torch.bincount(e, minlength = E), dim = 0)
        classes.append((key[keep] - tb[keep] * C).to(torch.int32))                   # (in entry order: by entry)
        total += int(classes[-1].numel())
    assert total < 1 << 31, total                                                     # (int32 pointers, as the tables')
    layout.cls_ptr = cls_ptr
    layout.cls = torch.cat(classes) if total else torch.zeros(1, dtype = torch.int32, device = dev)


#: :func:`sample_transition_masses`' tiles: entries of a sample per program (the rows are the job's tiles), warps
SAMPLE_TRANSITION_TILES = dict(TQ = 32, warps = 4)


@triton_jit
def _sample_transition_kernel(class_mars, mass_row, mass_stride, trans, rows, tiles, sptr, sbase, scount, cls_ptr, cls,
                              code, tok_ptr, eidx, eprobs, prow, lognorm, enum_logw, width, W, C, n, Cs, num_enum,
                              logw_stride, input_start, E, pid0, TS: tl.constexpr, TQ: tl.constexpr,
                              SAMPLE: tl.constexpr, ENUM: tl.constexpr, LOGW: tl.constexpr):
    """
    One program per (tile of ``TS`` rows at one position ``t``, sample ``b``): the sample's entries at boundary ``t``
    (``sptr`` [Be, n, W + 1], its lists' pointers), ``TQ`` at a time -- each the log-sum over the classes the sample
    allows that lead along it (``cls_ptr`` / ``cls``, see :class:`SampleMasses`) of the row's mass of the class: a
    stored class's from the row of the class masses (``SAMPLE``: the sample's row), an enumerated class's (``ENUM``,
    code < 0) its token's probability under the row's parameter row, times its weight (``LOGW``, as
    :func:`_transition_kernel`) --, stored at ``trans[sbase[b, t] + rank * scount[b, t] + e]`` for the entry's place
    ``e`` in the sample's list at ``t``.
    """
    tile = pid0 + tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = tl.arange(0, TS)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    rk = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)   # the step's rank at t
    row = tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    if SAMPLE:
        row = row + b * tl.load(mass_stride + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    masses = class_mars + row * Cs
    if ENUM:
        pr = tl.load(prow + row, mask = rmask, other = 0).to(tl.int64)
        tk = tl.load(tok_ptr + t).to(tl.int64)
        if LOGW:
            lz = tl.load(lognorm + row, mask = rmask, other = 0.0)
    bt = b * n + t
    e0 = tl.load(sptr + bt * (W + 1))                                        # the sample's entries at t
    e1 = tl.load(sptr + bt * (W + 1) + tl.load(width + t))
    out = trans + tl.load(sbase + bt) + rk[:, None] * tl.load(scount + bt).to(tl.int64)
    jj = tl.arange(0, TQ)
    for q0 in range(e0, e1, TQ):
        em = q0 + jj < e1
        f0 = tl.load(cls_ptr + b * (E + 1) + q0 + jj, mask = em, other = 0)   # the entries' allowed classes
        f1 = tl.load(cls_ptr + b * (E + 1) + q0 + jj + 1, mask = em, other = 0)
        mx = tl.full([TS, TQ], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TS, TQ], dtype = tl.float32)
        for k in range(0, tl.max(f1 - f0, axis = 0)):
            km = em & (f0 + k < f1)
            c = tl.load(cls + f0 + k, mask = km, other = 0)
            if ENUM:                                                         # (as :func:`~.prod_top_down._entry_mass`)
                x = tl.load(code + t * C + c, mask = km, other = -1)
                enum = (x < 0)[None, :]
                miss = rmask[:, None] & km[None, :]
                stored = tl.load(masses[:, None] + tl.maximum(x, 0)[None, :], mask = miss & ~enum, other = float("-inf"))
                e = tl.maximum(-1 - x, 0).to(tl.int64)                       # the e-th enumerated token at t (a stored
                gc = tl.load(eidx + tk + e).to(tl.int64)                     # class's: the boundary's first, unread)
                val = tl.log(tl.load(eprobs + pr[:, None] * num_enum + gc[None, :], mask = miss & enum, other = 0.0))
                if LOGW:
                    val = val + tl.load(enum_logw + b * logw_stride + tk + e, mask = km & (x < 0), other = 0.0)[None, :] - \
                        lz[:, None]
                mass = tl.where(enum, val, stored)
            else:
                mass = tl.load(masses[:, None] + c[None, :], mask = rmask[:, None] & km[None, :], other = float("-inf"))
            mx, acc = _lse_step(mx, acc, mass)
        val = tl.log(acc) + mx
        tl.store(out + (q0 + jj - e0)[None, :], val, mask = rmask[:, None] & em[None, :])


def sample_transition_masses(trans, job, masses, prog, layout: SampleMasses):
    """Write every sample's transition masses over its successor entries (``layout``, a :class:`SampleMasses`), for
    the steps of ``job`` (from :func:`transition_job`), from the class masses ``masses``: see
    :func:`_sample_transition_kernel`."""
    rows, tiles, TS = job
    cfg = SAMPLE_TRANSITION_TILES
    en = prog.enum
    enum = en.num > 0
    logw = enum and masses.enum_logw is not None
    eprobs = masses.enum_probs if enum else masses.table
    for first_tile, size in grid_chunks(tiles.size(0), 0):
        _sample_transition_kernel[(size, layout.Be)](
            masses.table, masses.row, masses.stride if masses.batch > 1 else masses.row, trans, rows, tiles,
            layout.ptr, layout.base, layout.count, layout.cls_ptr, layout.cls, en.code, en.ptr, en.index,
            eprobs, masses.prow, masses.lognorm if logw else masses.table,
            masses.enum_logw if logw else masses.table, prog.width_t, prog.W, prog.C, prog.n, masses.table.size(1),
            eprobs.size(1), enum_logw_stride(masses), prog.input_start, layout.rank.size(1), first_tile,
            TS = TS, TQ = cfg["TQ"], SAMPLE = masses.batch > 1, ENUM = enum, LOGW = logw, num_warps = cfg["warps"])
