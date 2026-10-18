"""
Transition masses: for a missing token at a position, a column and one of its successors, the log-mass of the token
classes that lead there, from the class masses (:func:`transition_masses`). The products' ``input @ block`` and
``block @ input`` steps read them, so a missing token costs a gather per successor, not per class (see
:mod:`.prod`). Their tables (:class:`TransitionTables`) are built once per backend.

ENUMERATED CLASSES (see :class:`~pyjuice.constraints.distributions.TokenClasses`) have no class mass: where a kernel
read the mass of class ``c`` at row ``r`` it reads a CODE instead (:class:`EnumeratedTokens`): the class's stored
column, or ``-1 - e`` for the ``e``-th enumerated token live at the boundary, whose mass is its token's own
probability (``tokmaj[token, prow[r]]``, the backend's token-major table) times its per-sample weight
(``enum_logw``, see :class:`~..state.ClassMasses`) over the row's ``Z``.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ....distributions.categorical import _exact_dot
from .common import grid_chunks



@dataclass(frozen = True)
class EnumeratedTokens:
    """
    The enumerated classes' tokens that are live at every boundary (lead somewhere from some active column), and
    every class's CODE at every boundary: its stored column, or ``-1 - e`` for the ``e``-th of the boundary's live
    enumerated tokens (a class dead at the boundary, enumerated: -1 - the count, matching no entry).

    :ivar ptr: [n + 1] int32, the boundary's tokens are ``tok[ptr[t] .. ptr[t + 1]]``; ``ptr_host`` the same on the host
    :ivar tok: [num + 1] int32, the tokens, then a 0 (kernels read a boundary's first token unmasked for a stored
        entry: past the last boundary's, this one)
    :ivar code: [n, C] int32
    :ivar num: the tokens over all boundaries (0: nothing enumerated; every code a stored column, the class itself)
    """

    ptr: torch.Tensor
    tok: torch.Tensor
    code: torch.Tensor
    num: int
    ptr_host: np.ndarray


def enumerated_tokens(next_col: torch.Tensor, width, n: int, classes) -> EnumeratedTokens:
    """The :class:`EnumeratedTokens` of a layout's ``next_col`` [n, W, C] under ``classes`` (a
    :class:`~pyjuice.constraints.distributions.TokenClasses`)."""
    C = next_col.size(2)
    dev = next_col.device
    nc = next_col.cpu()
    stored = classes.stored_of_class.cpu().long()
    etok = classes.enumerated_token.cpu().long()
    code = stored.expand(n, C).clone() if n > 0 else torch.zeros(0, C, dtype = torch.long)
    ptr = torch.zeros(n + 1, dtype = torch.long)
    toks = []
    if classes.num_enumerated > 0:
        for t in range(n):
            live = (nc[t, :int(width[t])] >= 0).any(dim = 0) & (etok >= 0)   # enumerated classes live at t
            ids = torch.nonzero(live).view(-1)
            e = torch.full((C,), ids.numel(), dtype = torch.long)              # (dead: the count, past the last)
            e[ids] = torch.arange(ids.numel())
            code[t] = torch.where(etok >= 0, -1 - e, stored)
            ptr[t + 1] = ptr[t] + ids.numel()
            toks.append(etok[ids])
    tok = torch.cat(toks + [torch.zeros(1, dtype = torch.long)])
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    return EnumeratedTokens(ptr = i32(ptr), tok = i32(tok), code = i32(code), num = int(ptr[-1]),
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
    :ivar pair_of: grouped only, [n, W, C] int32: the pair (within the boundary) every class of every column leads
        to, -1 where it leads nowhere
    :ivar num_pairs: grouped only, pairs per boundary (Python ints)
    :ivar max_pairs: grouped only, the most pairs of any column
    :ivar enum_ptr, enum_e, enum_pair: grouped only, the enumerated classes' live entries of every column, as a CSR
        (``enum_ptr`` [n, W + 1] int32, global numbers): the token's number ``e`` at its boundary and the pair it leads
        to; ``max_enum`` the most of any column
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
    enum_ptr: Optional[torch.Tensor] = None
    enum_e: Optional[torch.Tensor] = None
    enum_pair: Optional[torch.Tensor] = None
    max_enum: int = 0


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


def transition_tables(next_col: torch.Tensor, width, n: int, grouped: bool,
                      enum: Optional[EnumeratedTokens] = None) -> TransitionTables:
    """The :class:`TransitionTables` of a layout's ``next_col`` [n, W, C] and boundary widths; ``enum`` the
    :class:`EnumeratedTokens` (None: nothing enumerated, every class's code the class itself)."""
    W, C = next_col.size(1), next_col.size(2)
    nc = next_col.cpu().long()
    code = enum.code.cpu().long() if enum is not None and enum.num > 0 else None
    entry_ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    pred_ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    enum_ptr = torch.zeros(n, W + 1, dtype = torch.int64)
    pair_of = torch.full((n, W, C), -1, dtype = torch.int64) if grouped else None
    succs, xs, cols, pxs, num_pairs, ees, epairs = [], [], [], [], [], [], []
    base = max_pairs = ebase = max_enum = 0
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
            if code is not None:                                             # the enumerated classes' live entries
                en = code[t, c] < 0
                enum_ptr[t] = ebase + torch.searchsorted(q[en], torch.arange(W + 1))
                ees.append(-1 - code[t, c[en]]); epairs.append(inverse[en])
                ebase += int(en.sum())
                max_enum = max(max_enum, int(torch.bincount(q[en], minlength = 1).max()) if en.any() else 0)
        else:
            eq, es, ex = q, succ[q, c], (c if code is None else code[t, c])
        entry_ptr[t] = base + torch.searchsorted(eq, torch.arange(W + 1))
        by_succ = torch.argsort(es * (W + 1) + eq, stable = True)
        pred_ptr[t] = base + torch.searchsorted(es[by_succ], torch.arange(W + 1))
        succs.append(es); xs.append(ex); cols.append(eq[by_succ]); pxs.append(ex[by_succ])
        base += eq.numel()
    dev = next_col.device
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    cat = lambda xs, total: i32(torch.cat(xs)) if total > 0 else torch.zeros(1, dtype = torch.int32, device = dev)
    return TransitionTables(grouped = grouped, entry_ptr = i32(entry_ptr), entry_succ = cat(succs, base),
                            entry_x = cat(xs, base), pred_ptr = i32(pred_ptr), pred_col = cat(cols, base),
                            pred_x = cat(pxs, base), pair_of = i32(pair_of) if grouped else None,
                            num_pairs = num_pairs, max_pairs = max_pairs,
                            enum_ptr = i32(enum_ptr) if grouped else None,
                            enum_e = cat(ees, ebase) if grouped else None,
                            enum_pair = cat(epairs, ebase) if grouped else None, max_enum = max_enum)


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
def _transition_kernel(class_mars, mass_row, mass_stride, trans, rows, tiles, pair_of, entry_ptr, class_of_stored,
                       enum_ptr, enum_e, enum_pair, tok_ptr, tok, tokmaj, prow, lognorm, enum_logw, width, W, C, Cs,
                       num_rows, logw_stride, input_start, num_qtiles, trans_stride, pid0, TS: tl.constexpr,
                       QT: tl.constexpr, TP: tl.constexpr, TC: tl.constexpr, TE: tl.constexpr, SAMPLE: tl.constexpr,
                       ENUM: tl.constexpr, LOGW: tl.constexpr):
    """
    One program per (tile of rows at one position, ``QT`` columns) -- ``SAMPLE``: and sample, for class masses of each
    sample's own (rows ``mass_row + b * mass_stride``, transition masses ``trans_stride`` floats apart; a row the same
    for every sample, stride 0, is built for sample 0 alone): per column, its pairs ``TP`` at a time, as the product
    of the rows' class masses -- linear, scaled by 2^64, in column order over the ``Cs`` stored classes
    (``class_of_stored``: every column's class, where ``pair_of`` reads) -- with the 0/1 table of which of the pairs
    every class leads to. A class mass is a sum of fp32 probabilities, so 0 or at least 2^-149: scaled, a normal
    float, which tensor cores do not flush. ``ENUM``: then the column's enumerated entries (``enum_ptr`` / ``enum_e``
    / ``enum_pair``, see :class:`TransitionTables`), ``TE`` at a time, each the mass of its token: its probability
    under the row's parameter row (``tokmaj[token, prow[row]]``, ``num_rows`` the table's row stride) times --
    ``LOGW`` -- its weight ``exp(enum_logw[b, e] - lognorm[row])`` (``logw_stride`` floats per sample). The log of the
    sum is the transition mass.
    """
    pid = pid0 + tl.program_id(0)
    tile = pid // num_qtiles
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = tl.arange(0, TS)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    off = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)
    row = tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    b = tl.program_id(1).to(tl.int64)
    if SAMPLE:
        st = tl.load(mass_stride + (u - input_start), mask = rmask, other = 0).to(tl.int64)
        rmask = rmask & ((b == 0) | (st > 0))
        row = row + b * st
        off = off + tl.where(st > 0, b * trans_stride, 0)
    masses = class_mars + row * Cs
    if ENUM:
        pr = tl.load(prow + row, mask = rmask, other = 0).to(tl.int64)
        if LOGW:
            lz = tl.load(lognorm + row, mask = rmask, other = 0.0)
            lw0 = enum_logw + b * logw_stride + tl.load(tok_ptr + t)
        tk0 = tok + tl.load(tok_ptr + t)
    base = tl.load(entry_ptr + t * (W + 1))                                  # the boundary's first pair
    jj = tl.arange(0, TP)
    ee = tl.arange(0, TE)
    for qi in range(QT):
        q = (pid % num_qtiles) * QT + qi
        if q < tl.load(width + t):
            g_end = tl.load(entry_ptr + t * (W + 1) + q + 1) - base
            col = pair_of + (t * W + q).to(tl.int64) * C
            for g0 in range(tl.load(entry_ptr + t * (W + 1) + q) - base, g_end, TP):
                acc = tl.zeros([TS, TP], dtype = tl.float32)
                for c0 in range(0, Cs, TC):
                    c = c0 + tl.arange(0, TC)
                    cmask = c < Cs
                    x = tl.exp(tl.load(masses[:, None] + c[None, :], mask = rmask[:, None] & cmask[None, :],
                                       other = float("-inf")) + 44.3614195558365)                # x 2^64
                    cls = tl.load(class_of_stored + c, mask = cmask, other = 0)
                    g = tl.load(col + cls, mask = cmask, other = -1) - g0                        # pair in the tile
                    acc += _exact_dot(x, tl.where(g[:, None] == jj[None, :], 1.0, 0.0))
                if ENUM:                                                     # the column's enumerated entries
                    e_end = tl.load(enum_ptr + t * (W + 1) + q + 1)
                    for e0 in range(tl.load(enum_ptr + t * (W + 1) + q), e_end, TE):
                        em = e0 + ee < e_end
                        e = tl.load(enum_e + e0 + ee, mask = em, other = 0).to(tl.int64)
                        v = tl.load(tk0 + e, mask = em, other = 0).to(tl.int64)
                        x = tl.load(tokmaj + v[None, :] * num_rows + pr[:, None], mask = rmask[:, None] & em[None, :],
                                    other = 0.0) * 18446744073709551616.0                        # x 2^64
                        if LOGW:
                            x = x * tl.exp(tl.load(lw0 + e, mask = em, other = float("-inf"))[None, :] - lz[:, None])
                        g = tl.load(enum_pair + e0 + ee, mask = em, other = -1) - g0
                        acc += _exact_dot(x, tl.where(g[:, None] == jj[None, :], 1.0, 0.0))
                tl.store(trans + off[:, None] + (g0 + jj)[None, :], tl.log(acc) - 44.3614195558365,
                         mask = rmask[:, None] & (g0 + jj < g_end)[None, :])


def enumerated_args(prog, masses) -> dict:
    """The keyword arguments a kernel reading class masses by code takes (see :class:`EnumeratedTokens` and
    :class:`~..state.ClassMasses`): ``code, tok_ptr, tok, tokmaj, prow, lognorm, enum_logw, Cs, num_rows,
    logw_stride, ENUM, LOGW`` -- unread placeholders where nothing is enumerated (``ENUM`` False) or the masses are no
    sample's own (``LOGW`` False)."""
    en = prog.enum
    enum = en.num > 0
    logw = enum and masses.enum_logw is not None
    tokmaj = masses.tokmaj if enum else masses.table
    return dict(code = en.code, tok_ptr = en.ptr, tok = en.tok, tokmaj = tokmaj, prow = masses.prow,
                lognorm = masses.lognorm if logw else masses.table,
                enum_logw = masses.enum_logw if logw else masses.table, Cs = masses.table.size(1),
                num_rows = tokmaj.size(1), logw_stride = enum_logw_stride(masses), ENUM = enum, LOGW = logw)


def enum_logw_stride(masses) -> int:
    """How far apart the samples' rows of ``masses.enum_logw`` are: 0 when there is one row for every sample."""
    w = masses.enum_logw
    return 0 if w is None or w.size(0) == 1 else w.size(1)


def transition_masses(trans, job, masses, prog):
    """
    Write the transition masses of every step of ``job`` (from :func:`transition_job`): for each column ``q`` at the
    input's position ``t`` and each successor ``s``, ``log sum_{c: next_col[t, q, c] = s} exp(mass_r(c))`` (``r``:
    input ``u``'s row of the class masses ``masses``, a :class:`~..state.ClassMasses`, a stored class's mass from
    its table and an enumerated class's from its token; per sample when they are, ``prog.trans_size`` floats
    apart), at ``trans[offset + p]`` for the pair's number ``p`` within the boundary.
    """
    rows, tiles, TS = job
    tt, cfg, en = prog.trans, TRANSITION_TILES, prog.enum
    W, C = prog.next_col.size(1), prog.next_col.size(2)
    QT = cfg["QT"]
    while QT > 1 and tiles.size(0) * host_cdiv(max(prog.width), QT) < cfg["programs"]:
        QT //= 2
    num_qtiles = host_cdiv(max(prog.width), QT)
    TP = min(64, max(16, host_next_power_of_2(tt.max_pairs)))
    sample = masses.batch > 1
    stride = masses.stride if sample else masses.row
    enum = en.num > 0
    logw = enum and masses.enum_logw is not None
    tokmaj = masses.tokmaj if enum else masses.table
    for first, size in grid_chunks(tiles.size(0) * num_qtiles, 0):
        _transition_kernel[(size, masses.batch)](
            masses.table, masses.row, stride, trans, rows, tiles, tt.pair_of, tt.entry_ptr,
            prog.token_classes.class_of_stored32, tt.enum_ptr if enum else tt.entry_ptr,
            tt.enum_e if enum else tt.entry_ptr,
            tt.enum_pair if enum else tt.entry_ptr, en.ptr, en.tok, tokmaj, masses.prow,
            masses.lognorm if logw else masses.table, masses.enum_logw if logw else masses.table, prog.width_t, W, C,
            masses.table.size(1), tokmaj.size(1), enum_logw_stride(masses), prog.input_start, num_qtiles,
            prog.trans_size, first,
            TS = TS, QT = QT, TP = TP, TC = cfg["TC"], TE = min(64, max(16, host_next_power_of_2(max(tt.max_enum, 1)))),
            SAMPLE = sample, ENUM = enum, LOGW = logw, num_warps = cfg["warps"])
