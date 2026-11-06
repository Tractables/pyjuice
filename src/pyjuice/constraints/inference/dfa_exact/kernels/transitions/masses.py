"""
The masses of a query's transitions: a step's per (column, successor) pair, built from the class masses by :func:`transition_masses`, one set or
per sample.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from .....distributions.categorical import _exact_dot
from ..common import _next_col_of, grid_chunks


#: :func:`transition_masses`' tiles: the most rows (input steps at one position) and columns per program, classes per
#: product, warps. Rows and columns shrink until there are enough programs (they never change a result); the pairs
#: per product follow the widest column, up to 64.
#: :func:`_transition_kernel`'s tiles: TS rows a program (MEASURED on an HMM-4096 x hard_0 session at the pair tile 8:
#: 64 rows 320 ms a launch, 128 120, 256 285 -- a column's class successors and one-hot are per program, so more rows
#: amortize them until the registers run out; the rows' sums are the same whatever the tile), QT columns, TC classes
TRANSITION_TILES = dict(TS = 128, QT = 4, TC = 16, warps = 4, programs = 1024)
#: :func:`_transition_kernel`'s pair tile: None, the boundaries' mean pairs a column rounded up to a power of two
#: within [8, 64] -- a column's pairs are walked ``TP`` at a time, each pass over its masses and class successors
#: again, and the one-hot dot is ``TP`` wide, so a tile much wider than the columns' pairs is work on zeros (MEASURED
#: on an HMM-4096 x hard_0 session, 4.4 pairs a column: TP 64 321 ms a launch, 32 199, 16 178, 8 166, the same bits);
#: an int forces one (tests).
TRANSITION_PAIR_TILE = None


def transition_job(rows: torch.Tensor, dev):
    """
    The steps whose transition masses one :func:`transition_masses` call builds: ``rows`` [N, 3] (input row,
    position, the step's rank among its position's -- which block of the boundary's pairs, or of a sample's entries
    there, its masses are), as ``(rows, tiles, TS)`` on ``dev``, the rows ordered by position and ``tiles``
    [tiles, 3] (first row, number of rows, position) cutting them into tiles of at most ``TS`` rows at one position
    (fewer when there are few rows, so that there are enough programs).
    """
    TS = job_tile(rows.size(0))
    rows = rows.to(dev)
    rows = rows[torch.argsort(rows[:, 1], stable = True)]
    pos, counts = torch.unique_consecutive(rows[:, 1], return_counts = True)
    i32 = lambda x: x.to(dev, torch.int32).contiguous()
    return i32(rows), i32(job_tiles(pos, counts, TS)), TS


def job_tile(rows: int) -> int:
    """:func:`transition_job`'s most rows a tile, for so many rows."""
    return min(TRANSITION_TILES["TS"], max(16, host_next_power_of_2(rows) // 32))


def job_tiles(pos: torch.Tensor, counts: torch.Tensor, TS: int) -> torch.Tensor:
    """int64 [tiles, 3], ``(first row, number of rows, position)``: the rows of the positions ``pos`` (``counts`` of each,
    one position's after another's) cut into tiles of at most ``TS``."""
    per = (counts + TS - 1) // TS
    owner = torch.repeat_interleave(torch.arange(pos.numel(), device = pos.device), per)
    within = torch.arange(owner.numel(), device = pos.device) - (torch.cumsum(per, dim = 0) - per)[owner]
    first = (torch.cumsum(counts, dim = 0) - counts)[owner] + within * TS
    return torch.stack([first, (counts[owner] - within * TS).clamp(max = TS), pos[owner]], dim = 1)


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
    enum = en.on
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
    if TRANSITION_PAIR_TILE is None:
        mean_pairs = sum(tt.num_pairs) / max(1, sum(int(w) for w in prog.width[:prog.n]))
        TP = min(64, max(8, host_next_power_of_2(max(1, math.ceil(mean_pairs)))))
    else:
        TP = TRANSITION_PAIR_TILE
    sample = masses.batch > 1
    stride = masses.stride if sample else masses.row
    enum = en.on
    logw = enum and masses.enum_logw is not None
    eprobs = masses.enum_probs if enum else masses.table
    enum_ptr, enum_e, enum_pair = prog.enum_lists() if enum else (tt.entry_ptr, tt.entry_ptr, tt.entry_ptr)
    for first, size in grid_chunks(tiles.size(0) * num_qtiles, 0):
        _transition_kernel[(size, masses.batch)](
            masses.table, masses.row, stride, trans, rows, tiles, prog.delta, prog.state_id, prog.col_of_state,
            tt.entry_succ, tt.entry_ptr, prog.token_classes.class_of_stored32, enum_ptr, enum_e, enum_pair, en.ptr, en.index, eprobs, masses.prow,
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
    :ivar cls_ptr, xcode, wsel: every entry of the sample's lists (its number there, see ``ptr``) and the classes the
        sample allows that lead along it, ``cls_ptr[b, e] .. cls_ptr[b, e + 1]`` of the flat int32 ``xcode`` /
        ``wsel`` (``cls_ptr`` int32 [Be, E + 1], one sample's after another's) -- what an entry's mass is the log-sum
        over (a session's inputs allow few of the hundreds of classes leading along a JSON automaton's entry): every
        class's code at the entry's boundary, resolved -- the stored column (>= 0), or for an enumerated class ``-1 -
        j``, ``j`` its token's column of ``enum_probs`` -- and its token's place among the enumerated tokens live at
        any boundary (its ``enum_logw`` column; -1 for a stored class)
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
    xcode: Optional[torch.Tensor] = None                       # layout: a session over the budget never pays for them)
    wsel: Optional[torch.Tensor] = None


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


def _pair_tables(prog):
    """Of every (entry, class) pair of :func:`~.live.entry_classes`, built once per program: its class's code at its
    boundary resolved (``SampleMasses.xcode``) and its enumerated token's place (``SampleMasses.wsel``), int32 [P]."""
    kept = getattr(prog, "entry_class_tables", None)
    if kept is None:
        from ..live import entry_classes
        n, C, dev, en = prog.n, prog.C, prog.dev, prog.enum
        key, first = entry_classes(prog)
        E = first.numel() - 1
        g = torch.repeat_interleave(torch.arange(E, device = dev), first[1:] - first[:-1])    # every pair's entry
        tb = torch.searchsorted(prog.succ.boundary_base[1:n + 1], g, right = True)        # ... its boundary
        c = key - tb * C                                                                  # ... its class
        x = en.code.view(-1)[key].long() if en.on else c                                  # ... its code there
        e = (-1 - x).clamp(min = 0)
        at = en.ptr.long()[tb] + e if en.on else e
        enum = x < 0
        xcode = torch.where(enum, -1 - en.index.long()[at.clamp(max = en.index.numel() - 1)], x) if en.on else x
        wsel = torch.where(enum, at, -1)
        kept = prog.entry_class_tables = (xcode.to(torch.int32), wsel.to(torch.int32))
    return kept


def sample_classes(layout: SampleMasses, succ, prog) -> None:
    """Give ``layout`` every entry's allowed classes (``cls_ptr``, and their codes ``xcode`` / ``wsel``): the
    program's (entry, class) pairs (:func:`~.live.entry_classes`) whose class the sample allows
    (``succ.allowed``), by the entry's number in the sample's lists -- the samples :data:`~.live.SUPPORT_CHUNK` pairs
    at a time: the pairs by program entry, the sample's entries in program order
    (:attr:`~.live.SampleSuccessors.entry`), so an entry's classes end where the allowed pairs up to its program
    entry's last do. A wait for the device per chunk (its pairs' number)."""
    from ..live import SUPPORT_CHUNK, entry_classes
    Be, n, C, dev = layout.Be, prog.n, prog.C, prog.dev
    key, first = entry_classes(prog)                                                  # (entry, class) pairs by entry
    xcode, wsel = _pair_tables(prog)
    E, P = succ.rank.size(1), key.numel()
    allowed = succ.allowed.view(Be, n * C)
    cls_ptr = torch.empty(Be, E + 1, dtype = torch.int32, device = dev)
    j = torch.arange(E, device = dev)
    base = torch.zeros(1, dtype = torch.int64, device = dev)                          # the samples' before
    picked, total = [], 0
    step = max(1, SUPPORT_CHUNK // max(P, 1))
    for b0 in range(0, Be, step):
        bs = min(step, Be - b0)
        keep = allowed[b0:b0 + bs, key] != 0                                          # the sample allows the class
        hits = torch.zeros(bs * P + 1, dtype = torch.int64, device = dev)             # the chunk's allowed pairs before
        hits[1:] = torch.cumsum(keep.view(-1), dim = 0)                               # each (one scan over the chunk:
        hits = hits.as_strided((bs, P + 1), (P, 1))                                   # a scan along rows runs a row a
        num = (succ.rank[b0:b0 + bs] >= 0).sum(dim = 1)                               # block); the sample's entries
        ends = hits.gather(1, first[succ.entry[b0:b0 + bs, :E].long() + 1])           # ... each one's last pair's
        ends = torch.where(j[None, :] < num[:, None], ends, hits[:, P:])
        cls_ptr[b0:b0 + bs, 0] = base + hits[:, 0]                                    # (samples one after another)
        cls_ptr[b0:b0 + bs, 1:] = base + ends
        base = base + hits[-1, P]
        picked.append(torch.nonzero(keep.view(-1)).squeeze(1) % P)                   # (in entry order: by entry)
        total += picked[-1].numel()
    assert total < 1 << 31, total                                                     # (int32 pointers, as the tables')
    pair = torch.cat(picked) if total else torch.zeros(1, dtype = torch.int64, device = dev)
    layout.cls_ptr = cls_ptr
    layout.xcode, layout.wsel = xcode[pair], wsel[pair]
