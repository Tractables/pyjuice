"""
Transition masses by COLUMN SIGNATURE: a query's one set of transition masses (one mass a step and (column,
successor) pair, :func:`~.transitions.transition_masses`) computed once per distinct column instead of once per
boundary column.

A column's masses -- ``log sum_{c: next(q, c) = s} exp(mass_r(c))`` for its pairs ``s`` and a step's class-mass row
``r`` -- depend on the row, the column's automaton state, its pairs' successor states (the next boundary's live columns
among them) and its enumerated tokens alone: its SIGNATURE. A decode well past the shortest accepted string repeats
signatures across boundaries (HMM-4096 x JSON hard_0, n 128: 85,034 columns, 3,144 states, 6,185 signatures), and
under tied inputs every position's steps read the same class-mass rows (an HMM: 4096 rows for 128 x 4096 steps). The
masses are then laid out per (class of positions reading the same rows, rank, signature pair) -- a position's steps
in rank order, the class's rows -- and built once a signature and class, from a representative column: the same
arithmetic as :func:`~.kernels.transitions.masses._transition_kernel` over the same operands (the column's classes in the same
order, its pairs and enumerated tokens chunked alike), so every mass is the per-column kernel's own, bit for bit.

The readers (the ``input @ block`` / ``block @ input`` kernels and their top-down twins) address a step's masses as
before -- ``mbase[t] + rank * mcount[t]`` -- with the signature layout's per-position base and count, and an entry's
mass index from :attr:`SignatureMasses.entry_x` / :attr:`SignatureMasses.pred_x` (the pair's signature slot) in
place of the grouped tables' pair numbers (see :meth:`~..compiled.program.Program.mass_x`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from .....distributions.categorical import _exact_dot
from ..common import _next_col_of, grid_chunks, one_copy
from .host import host_tables
from .host_kernels import _column_signatures, _signature_representatives, _signature_slots
from .masses import TRANSITION_PAIR_TILE, TRANSITION_TILES, job_tile, job_tiles

#: Whether a query's one set of transition masses is laid out by column signature where that saves at least
#: ``SIGNATURE_MIN_SAVING`` times the floats (see the module docstring)
SIGNATURE_MASSES = True
SIGNATURE_MIN_SAVING = 1.5


@dataclass
class SignatureMasses:
    """
    A program's transition masses by column signature, for one class-mass row map (see the module docstring).

    :ivar pair_off, pairs: int64 [n] on the device: position ``t``'s steps' masses start at ``pair_off[t]``, a step's
        ``pairs[t]`` floats after the previous rank's (its class's first float; the signatures' slots)
    :ivar entry_x, pred_x: int32 on the device, the grouped tables' entries' and predecessors' mass indices: their
        pair's signature slot
    :ivar size: the floats of the masses
    :ivar job: the representative columns' steps, as :func:`~.transitions.transition_job` cuts them
    :ivar rep_ptr, rep_col, rep_shift: the representative columns at every position (``rep_ptr`` int32 [n + 1] into
        ``rep_col`` int32 / ``rep_shift`` int64: a column's first pair's slot less its first pair's number)
    :ivar max_reps: the most representative columns at a position
    :ivar saving: the per-column layout's floats over these
    """

    pair_off: torch.Tensor
    pairs: torch.Tensor
    entry_x: torch.Tensor
    pred_x: torch.Tensor
    size: int
    job: tuple
    rep_ptr: torch.Tensor
    rep_col: torch.Tensor
    rep_shift: torch.Tensor
    max_reps: int
    saving: float


def _signed(value: int) -> int:
    """A 64-bit constant as the int64 of its bits."""
    return value - (1 << 64) if value >= 1 << 63 else value


def _shift_right(z: torch.Tensor, bits: int) -> torch.Tensor:
    """The bits of ``z`` (int64) shifted right as unsigned."""
    return (z >> bits) & ((1 << (64 - bits)) - 1)


def _mix(v: torch.Tensor, salt: int) -> torch.Tensor:
    """A 64-bit mix of int64 values (splitmix64's finalizer, salted; wrapping)."""
    z = v + _signed(0x9E3779B97F4A7C15 ^ salt)
    z = (z ^ _shift_right(z, 30)) * _signed(0xBF58476D1CE4E5B9)
    z = (z ^ _shift_right(z, 27)) * _signed(0x94D049BB133111EB)
    return z ^ _shift_right(z, 31)


def _segment_hash(vals: torch.Tensor, seg: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor, longest: int,
                  salt: int) -> torch.Tensor:
    """Every segment's polynomial hash of its values (``vals`` int64, ``seg`` every value's segment, ``starts`` /
    ``lengths`` the segments' in ``vals``, ``longest`` the longest's length), wrapping: 0 for an empty segment."""
    out = torch.zeros(lengths.numel(), dtype = torch.int64, device = vals.device)
    if vals.numel() == 0:
        return out
    k = torch.arange(vals.numel(), device = vals.device) - starts[seg]
    base = 0x100000001B3 + 2 * salt
    powers, power = [], 1
    for _ in range(longest):
        powers.append(_signed(power))
        power = power * base % (1 << 64)
    pw = torch.tensor(powers, dtype = torch.int64, device = vals.device)
    return out.index_add_(0, seg, _mix(vals, salt) * pw[k])


def _spread(counts: torch.Tensor):
    """``(at, within)`` for runs of ``counts`` items: every item's run and its place in the run."""
    dev = counts.device
    at = torch.repeat_interleave(torch.arange(counts.numel(), device = dev), counts)
    return at, torch.arange(at.numel(), device = dev) - (torch.cumsum(counts, dim = 0) - counts)[at]


def _first_of(group: torch.Tensor, groups: int) -> torch.Tensor:
    """Every group's first index in ``group``."""
    first = torch.full((groups,), group.numel(), dtype = torch.int64, device = group.device)
    return first.scatter_reduce_(0, group, torch.arange(group.numel(), device = group.device), "amin")


def _unique_rows(columns: list) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(first, inverse)`` of the rows of ``columns`` (int64 [rows] each) as ``np.unique(axis = 0)`` gives them: the
    distinct rows in lexicographic order of signed values, every one's first row, and every row's number."""
    N = columns[0].numel()
    dev = columns[0].device
    order = torch.arange(N, device = dev)
    for col in reversed(columns):                                  # (stable sorts, the last column first)
        order = order[torch.argsort(col[order], stable = True)]
    new = torch.zeros(N, dtype = torch.bool, device = dev)
    if N:
        new[0] = True
    for col in columns:
        at = col[order]
        new[1:] |= at[1:] != at[:-1]
    number = torch.cumsum(new, dim = 0) - 1
    inverse = torch.empty_like(number)
    inverse[order] = number
    return order[new], inverse


class PositionClasses(NamedTuple):
    """The positions reading the same class-mass rows in rank order (:func:`position_classes`): ``of`` int64 [n], every
    position's class (-1: no step reads its masses), and ``steps`` int64 [classes], the steps of a position of each, for
    the row map ``row``."""

    row: torch.Tensor
    of: np.ndarray
    steps: np.ndarray


def position_classes(prog, masses) -> PositionClasses:
    """The :class:`PositionClasses` of ``prog``'s input steps under ``masses``' row map: every step's row, a look at each
    (HMM-4096: half a million) on the host, kept with the circuit's steps: they say nothing of the automaton."""
    kept = prog.assembly.position_classes
    if kept is not None and (kept.row is masses.row or torch.equal(kept.row, masses.row)):
        return kept
    rows = prog.trans_job[0].long()                                # (input row, position, rank), by position
    mass_row = masses.row.long()[rows[:, 0] - prog.input_start]
    rows_h, mass_h = rows.cpu().numpy(), mass_row.cpu().numpy()
    pos, counts = np.unique(rows_h[:, 1], return_counts = True)
    steps = np.zeros(prog.n, dtype = np.int64)
    steps[pos] = counts
    order = np.lexsort((rows_h[:, 2], rows_h[:, 1]))
    by_pos = np.split(mass_h[order], np.cumsum(counts)[:-1])
    of, classes = np.full(prog.n, -1, dtype = np.int64), {}
    for t, mr in zip(pos.tolist(), by_pos):
        of[t] = classes.setdefault(mr.tobytes(), len(classes))
    csteps = np.zeros(len(classes), dtype = np.int64)
    for t in pos.tolist():
        csteps[of[t]] = steps[t]
    kept = prog.assembly.position_classes = PositionClasses(masses.row, of, csteps)
    return kept


def signature_masses(prog, masses, trans_size: int) -> Optional[SignatureMasses]:
    """
    The :class:`SignatureMasses` of ``prog``'s grouped tables (``prog.succ``) for ``masses``' class-mass row map (one
    set: ``masses.batch`` 1, no per-sample token weights), or None where it saves less than
    :data:`SIGNATURE_MIN_SAVING` times the per-column layout's ``trans_size`` floats, nothing is to be built, or a check
    fails. On the device, once a program and row map.
    """
    tt, en, job = prog.succ, prog.enum, prog.trans_job
    if job is None or not tt.grouped or masses.batch != 1 or masses.enum_logw is not None:
        return None
    host = host_tables(prog._tr)
    if host is not None and host.pairs_host is not None:
        return _on_the_host(prog, masses, trans_size, host)
    n, W, dev = prog.n, prog.W, prog.dev
    long = lambda x: x.long()
    width = torch.tensor(prog.width, dtype = torch.int64, device = dev)
    sid = long(prog.state_id).view(n + 1, W)
    entry_ptr = long(tt.entry_ptr).view(n, W + 1)
    entry_succ, entry_x = long(tt.entry_succ), long(tt.entry_x)
    pred_ptr, pred_x = long(tt.pred_ptr).view(n, W + 1), long(tt.pred_x)
    enum = en.on

    # every column (t, q) and its pairs (entries: by column, then successor)
    T, Q = _spread(width[:n])
    es, ee = entry_ptr[T, Q], entry_ptr[T, Q + 1]
    npairs = ee - es
    cstart = torch.cumsum(npairs, dim = 0) - npairs
    ecol, ek = _spread(npairs)                                     # every (column, pair) visit
    eidx = es[ecol] + ek
    gfirst = es - entry_ptr[T, 0]                                  # a column's first pair's number (the kernels')
    if not torch.equal(entry_x[eidx], gfirst[ecol] + ek):          # (its pairs: consecutive numbers from it)
        return None
    end = T[ecol] + 1 == n
    next_t = (T[ecol] + 1).clamp(max = n)
    succ_state = torch.where(end, -7, sid[next_t, torch.where(end, torch.zeros_like(eidx), entry_succ[eidx])])
    # its enumerated tokens: (token's column of the probabilities, pair relative to the column's first)
    if enum:
        enum_ptr = long(tt.enum_ptr).view(n, W + 1)
        enum_e, enum_pair = long(tt.enum_e), long(tt.enum_pair)
        tok_ptr, tok_index = long(en.ptr), long(en.index)
        ns_, ne_ = enum_ptr[T, Q], enum_ptr[T, Q + 1]
        nenum = ne_ - ns_
        nstart = torch.cumsum(nenum, dim = 0) - nenum
        ncol, nk = _spread(nenum)
        nidx = ns_[ncol] + nk
        tok = tok_index[tok_ptr[T[ncol]] + enum_e[nidx]]
        rel = enum_pair[nidx] - gfirst[ncol]
        evals = tok * 65536 + rel
    else:
        nenum = torch.zeros(T.numel(), dtype = torch.int64, device = dev)
        nstart, ncol, evals = nenum, torch.zeros(0, dtype = torch.int64, device = dev), torch.zeros(0, dtype = torch.int64, device = dev)
    # a column's sizes, in that order, one key where they fit in it (the rows' order is theirs: one sort, not four)
    pair_bits, enum_bits = int(tt.max_pairs).bit_length(), int(tt.max_enum).bit_length()
    if int(prog.K).bit_length() + pair_bits + enum_bits < 62:
        sizes = [(((sid[T, Q] << pair_bits) | npairs) << enum_bits | nenum) << 1 | (T + 1 == n).long()]
    else:
        sizes = [sid[T, Q], npairs, nenum, (T + 1 == n).long()]
    # (the operands' hashes: no second one, exactly checked below)
    first, sig = _unique_rows(sizes + [_segment_hash(succ_state, ecol, cstart, npairs, tt.max_pairs, 1)] +
                              ([_segment_hash(evals, ncol, nstart, nenum, tt.max_enum, 3)] if enum else []))
    rep = first[sig]                                               # every column's signature's first column
    # exact: every column's operands are its first's (no hash collision)
    if not torch.equal(succ_state, succ_state[cstart[rep[ecol]] + ek]):
        return None
    if enum and not torch.equal(evals, evals[nstart[rep[ncol]] + nk]):
        return None
    S = first.numel()
    sig_pairs = npairs[first]
    sig_off = torch.cumsum(sig_pairs, dim = 0) - sig_pairs
    slots = int(sig_pairs.sum())

    # the positions' classes: the same class-mass rows in rank order
    classes = position_classes(prog, masses)
    cls_of, csteps = classes.of, classes.steps
    cbase = np.cumsum(csteps * slots) - csteps * slots
    size = int((csteps * slots).sum())
    if size * SIGNATURE_MIN_SAVING > trans_size:
        return None
    pair_off = np.where(cls_of >= 0, cbase[np.maximum(cls_of, 0)], 0)
    pairs = np.where(cls_of >= 0, slots, 0)

    # representatives: every (class, signature)'s first column at a position of the class's
    cls_t = torch.from_numpy(cls_of).to(dev)
    cand = torch.nonzero(cls_t[T] >= 0).flatten()
    _, group = torch.unique(cls_t[T[cand]] * S + sig[cand], return_inverse = True)
    reps = torch.sort(cand[_first_of(group, int(group.max()) + 1 if group.numel() else 0)]).values   # by position, then column
    rep_t, rep_q = T[reps], Q[reps]
    rep_shift = sig_off[sig[reps]] - gfirst[reps]
    rep_count = torch.bincount(rep_t, minlength = n)
    rep_ptr = torch.cat([torch.zeros(1, dtype = torch.int64, device = dev), torch.cumsum(rep_count, dim = 0)])

    # the entries' and predecessors' mass indices: their pair's slot (alike boundaries share entries: one value)
    exs = entry_x.clone()
    slot_e = sig_off[sig[ecol]] + ek
    exs[eidx] = slot_e
    if not torch.equal(exs[eidx], slot_e):
        return None
    pairbase = torch.cat([torch.zeros(1, dtype = torch.int64, device = dev),
                          torch.cumsum(torch.as_tensor(tt.num_pairs, dtype = torch.int64, device = dev), dim = 0)])
    pslot = torch.full((int(pairbase[-1]),), -1, dtype = torch.int64, device = dev)
    pslot[pairbase[T[ecol]] + entry_x[eidx]] = slot_e
    jw = torch.where(torch.arange(n, device = dev) + 1 == n, 1, width[1:n + 1])    # columns at t + 1 (a sequence end: one)
    boundaries = torch.arange(n, device = dev)
    ps, pe = pred_ptr[boundaries, 0], pred_ptr[boundaries, jw]
    pt, within = _spread(pe - ps)
    pidx = ps[pt] + within
    slot_p = pslot[pairbase[pt] + pred_x[pidx]]
    pxs = pred_x.clone()
    pxs[pidx] = slot_p
    if (slot_p < 0).any() or not torch.equal(pxs[pidx], slot_p):
        return None

    # the representatives' steps: every step of a position with one, as `transition_job` cuts them
    kept = torch.unique(rep_t)
    on = torch.zeros(n, dtype = torch.bool, device = dev)
    on[kept] = True
    rep_rows = job[0][on[job[0][:, 1].long()]].contiguous()
    TS = job_tile(rep_rows.size(0))
    i32 = lambda x: x.to(torch.int32).contiguous()
    return SignatureMasses(pair_off = torch.from_numpy(pair_off).to(dev), pairs = torch.from_numpy(pairs).to(dev),
                           entry_x = i32(exs), pred_x = i32(pxs), size = size,
                           job = (rep_rows, i32(job_tiles(kept, prog.pos_steps[kept], TS)), TS), rep_ptr = i32(rep_ptr),
                           rep_col = i32(rep_q), rep_shift = rep_shift.contiguous(),
                           max_reps = int(rep_count.max()) if rep_count.numel() else 0,
                           saving = trans_size / max(size, 1))


def _on_the_host(prog, masses, trans_size: int, host) -> Optional[SignatureMasses]:
    """:func:`signature_masses` of tables built on the host (:mod:`.host`): the same, by the groups of alike boundaries."""
    n, dev = prog.n, prog.dev
    pairs_host = host.pairs_host
    width, firsts, group = host.tr_width, host.firsts_host, host.group_host
    state_id = host.host[1]
    enumerated = pairs_host["enum"]
    none = np.zeros((host.G, prog.W + 1), dtype = np.int32)
    zero32, zero64 = np.zeros(1, dtype = np.int32), np.zeros(host.G + 1, dtype = np.int64)
    enum_base, enum_row, enum_e, enum_pair = (zero64, none, zero32, zero32) if enumerated is None else \
        (enumerated["base"], enumerated["row"], enumerated["e"], enumerated["pair"])
    tok_index, tok_ptr = (zero32, np.zeros(n + 1, dtype = np.int64)) if host.enum_lists is None else host.enum_lists
    pair_base, pair_row = pairs_host["base"], pairs_host["row"]
    most_pairs, most_enum = prog.succ.max_pairs, prog.succ.max_enum
    pair_bits, enum_bits = int(most_pairs).bit_length(), int(most_enum).bit_length()
    row_first, sig, first, ok = _column_signatures(firsts, width, state_id, n, pair_base, pair_row, pairs_host["s"], enum_base, enum_row, enum_e, enum_pair,
                                                   tok_index, tok_ptr, enumerated is not None, pair_bits, enum_bits, most_pairs, most_enum)
    if not ok:
        return None
    S = first.size
    sig_pairs = np.zeros(S, dtype = np.int64)
    if S:
        of_row = np.searchsorted(row_first, first, side = "right") - 1
        sig_pairs = (pair_row[of_row, first - row_first[of_row] + 1] - pair_row[of_row, first - row_first[of_row]]).astype(np.int64)
    sig_off = np.cumsum(sig_pairs) - sig_pairs
    slots = int(sig_pairs.sum())

    # the positions' classes: the same class-mass rows in rank order
    classes = position_classes(prog, masses)
    cls_of, csteps = classes.of, classes.steps
    cbase = np.cumsum(csteps * slots) - csteps * slots
    size = int((csteps * slots).sum())
    if size * SIGNATURE_MIN_SAVING > trans_size:
        return None
    pair_off = np.where(cls_of >= 0, cbase[np.maximum(cls_of, 0)], 0)
    pairs = np.where(cls_of >= 0, slots, 0)

    rep_t, rep_q, rep_shift = _signature_representatives(group, cls_of, width, row_first, sig, sig_off, pair_row, csteps.size, S, host.G)
    rep_count = np.bincount(rep_t, minlength = n)
    rep_ptr = np.concatenate([[0], np.cumsum(rep_count)]).astype(np.int32)
    exs, pxs = _signature_slots(pairs_host["x"], pairs_host["pred_x"], firsts, width, row_first, sig, sig_off, pair_base, pair_row)

    # the representatives' steps: every step of a position with one, as `transition_job` cuts them
    assembly = prog.assembly
    rows = assembly.trans_rows_host                                # (by position)
    kept = np.unique(rep_t)
    steps = assembly.steps_host.numpy()
    whole = kept.size == np.count_nonzero(steps)
    starts = np.cumsum(steps) - steps
    rep_rows = rows if whole else np.concatenate([rows[a:a + m] for a, m in zip(starts[kept].tolist(), steps[kept].tolist())]) if kept.size \
        else rows[:0]
    TS = job_tile(rep_rows.shape[0])
    counts = steps[kept]
    per = (counts + TS - 1) // TS
    owner = np.repeat(np.arange(kept.size), per)
    within = np.arange(owner.size) - (np.cumsum(per) - per)[owner]
    tiles = np.stack([(np.cumsum(counts) - counts)[owner] + within * TS, np.minimum(counts[owner] - within * TS, TS), kept[owner]], axis = 1).astype(np.int32)
    arrays = [exs if exs.size else zero32, pxs if pxs.size else zero32, tiles, rep_ptr, rep_q] + ([] if whole else [rep_rows.astype(np.int32)])
    moved = one_copy(arrays, dev)
    wide = one_copy([pair_off.astype(np.int64), pairs.astype(np.int64), rep_shift], dev, np.int64)
    return SignatureMasses(pair_off = wide[0], pairs = wide[1], entry_x = moved[0], pred_x = moved[1], size = size,
                           job = (prog.trans_job[0] if whole else moved[5], moved[2], TS), rep_ptr = moved[3], rep_col = moved[4],
                           rep_shift = wide[2], max_reps = int(rep_count.max()) if rep_count.size else 0, saving = trans_size / max(size, 1))


@triton_jit
def _signature_transition_kernel(class_mars, mass_row, trans, rows, tiles, delta, state_id, col_of_state, entry_succ,
                                 entry_ptr, class_of_stored, enum_ptr, enum_e, enum_pair, tok_ptr, eidx, eprobs, prow,
                                 rep_ptr, rep_col, rep_shift, width, W, C, K, n, Cs, num_enum, input_start, num_qtiles,
                                 pair_off, pairs, pid0, TS: tl.constexpr, QT: tl.constexpr, TP: tl.constexpr,
                                 TC: tl.constexpr, TE: tl.constexpr, ENUM: tl.constexpr):
    """:func:`~.kernels.transitions.masses._transition_kernel` (one set of masses, no per-sample weights) over every position's
    representative columns (``rep_ptr`` / ``rep_col``) alone, a column's masses stored at its signature's slots
    (``rep_shift``: the first's slot less the column's first pair's number) -- the same operations on the same
    operands, so the same values."""
    pid = pid0 + tl.program_id(0)
    tile = pid // num_qtiles
    start, count, t = tl.load(tiles + tile * 3), tl.load(tiles + tile * 3 + 1), tl.load(tiles + tile * 3 + 2)
    r = tl.arange(0, TS)
    rmask = r < count
    u = tl.load(rows + (start + r) * 3, mask = rmask, other = 0).to(tl.int64)
    rk = tl.load(rows + (start + r) * 3 + 2, mask = rmask, other = 0).to(tl.int64)   # the step's rank at t
    off = tl.load(pair_off + t) + rk * tl.load(pairs + t)
    row = tl.load(mass_row + (u - input_start), mask = rmask, other = 0).to(tl.int64)
    masses = class_mars + row * Cs
    if ENUM:
        pr = tl.load(prow + row, mask = rmask, other = 0).to(tl.int64)
        tk0 = eidx + tl.load(tok_ptr + t)
    base = tl.load(entry_ptr + t * (W + 1))                                  # the boundary's first pair
    jj = tl.arange(0, TP)
    ee = tl.arange(0, TE)
    r0 = tl.load(rep_ptr + t)
    nrep = tl.load(rep_ptr + t + 1) - r0
    for qi in range(QT):
        qq = (pid % num_qtiles) * QT + qi
        if qq < nrep:
            q = tl.load(rep_col + r0 + qq)
            shift = tl.load(rep_shift + r0 + qq)
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
                        g = tl.load(enum_pair + e0 + ee, mask = em, other = -1) - g0
                        acc += _exact_dot(x, tl.where(g[:, None] == jj[None, :], 1.0, 0.0))
                tl.store(trans + off[:, None] + shift + (g0 + jj)[None, :], tl.log(acc) - 44.3614195558365,
                         mask = rmask[:, None] & (g0 + jj < g_end)[None, :])


def signature_transition_masses(trans, sig: SignatureMasses, masses, prog):
    """Write the transition masses of ``sig``'s layout (see :func:`~.transitions.transition_masses`, whose tile
    choices it shares) from ``masses``' (a :class:`~..state.ClassMasses`, one set)."""
    rows, tiles, TS = sig.job
    tt, cfg, en = prog.succ, TRANSITION_TILES, prog.enum
    W, C = prog.W, prog.C
    QT = cfg["QT"]
    while QT > 1 and tiles.size(0) * host_cdiv(max(sig.max_reps, 1), QT) < cfg["programs"]:
        QT //= 2
    num_qtiles = host_cdiv(max(sig.max_reps, 1), QT)
    if TRANSITION_PAIR_TILE is None:                               # (the per-column kernel's pair tile: same chunks)
        mean_pairs = sum(tt.num_pairs) / max(1, sum(int(w) for w in prog.width[:prog.n]))
        TP = min(64, max(8, host_next_power_of_2(max(1, math.ceil(mean_pairs)))))
    else:
        TP = TRANSITION_PAIR_TILE
    enum = en.on
    eprobs = masses.enum_probs if enum else masses.table
    for first, size in grid_chunks(tiles.size(0) * num_qtiles, 0):
        _signature_transition_kernel[(size,)](
            masses.table, masses.row, trans, rows, tiles, prog.delta, prog.state_id, prog.col_of_state, tt.entry_succ,
            tt.entry_ptr, prog.token_classes.class_of_stored32, tt.enum_ptr if enum else tt.entry_ptr,
            tt.enum_e if enum else tt.entry_ptr, tt.enum_pair if enum else tt.entry_ptr, en.ptr, en.index, eprobs,
            masses.prow, sig.rep_ptr, sig.rep_col, sig.rep_shift, prog.width_t, W, C, prog.K, prog.n,
            masses.table.size(1), eprobs.size(1), prog.input_start, num_qtiles, sig.pair_off, sig.pairs, first,
            TS = TS, QT = QT, TP = TP, TC = cfg["TC"], TE = min(64, max(16, host_next_power_of_2(max(tt.max_enum, 1)))),
            ENUM = enum, num_warps = cfg["warps"])
