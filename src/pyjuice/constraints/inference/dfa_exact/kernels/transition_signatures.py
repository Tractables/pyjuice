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
arithmetic as :func:`~.transitions._transition_kernel` over the same operands (the column's classes in the same
order, its pairs and enumerated tokens chunked alike), so every mass is the per-column kernel's own, bit for bit.

The readers (the ``input @ block`` / ``block @ input`` kernels and their top-down twins) address a step's masses as
before -- ``mbase[t] + rank * mcount[t]`` -- with the signature layout's per-position base and count, and an entry's
mass index from :attr:`SignatureMasses.entry_x` / :attr:`SignatureMasses.pred_x` (the pair's signature slot) in
place of the grouped tables' pair numbers (see :meth:`~..compiled.program.Program.mass_x`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ....distributions.categorical import _exact_dot
from .common import _next_col_of, grid_chunks
from .transitions import TRANSITION_PAIR_TILE, TRANSITION_TILES, transition_job

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


def _mix(v: np.ndarray, salt: int) -> np.ndarray:
    """A 64-bit mix of int64 values (splitmix64's finalizer, salted)."""
    with np.errstate(over = "ignore"):
        z = v.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15 ^ salt)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _segment_hash(vals: np.ndarray, seg: np.ndarray, starts: np.ndarray, lengths: np.ndarray, salt: int) -> np.ndarray:
    """Every segment's polynomial hash of its values (``vals`` int64, ``seg`` every value's segment, ``starts`` /
    ``lengths`` the segments' in ``vals``): 0 for an empty segment."""
    out = np.zeros(lengths.size, dtype = np.uint64)
    if vals.size == 0:
        return out
    k = np.arange(vals.size, dtype = np.int64) - starts[seg]
    base = np.uint64(0x100000001B3 + 2 * salt)
    pw = np.ones(int(k.max()) + 1, dtype = np.uint64)
    with np.errstate(over = "ignore"):
        for i in range(1, pw.size):
            pw[i] = pw[i - 1] * base
        terms = _mix(vals, salt) * pw[k]
    np.add.at(out, seg, terms)                                     # (uint64: wraps)
    return out


def signature_masses(prog, masses, trans_size: int) -> Optional[SignatureMasses]:
    """
    The :class:`SignatureMasses` of ``prog``'s grouped tables (``prog.succ``) for ``masses``' class-mass row map (one
    set: ``masses.batch`` 1, no per-sample token weights), or None where it saves less than
    :data:`SIGNATURE_MIN_SAVING` times the per-column layout's ``trans_size`` floats, nothing is to be built, or a check
    fails. On the host, once a program and row map (two device copies: the tables and the rows).
    """
    tt, en, job = prog.succ, prog.enum, prog.trans_job
    if job is None or not tt.grouped or masses.batch != 1 or masses.enum_logw is not None:
        return None
    n, W, dev = prog.n, prog.W, prog.dev
    width = np.asarray(prog.width, dtype = np.int64)
    host = lambda x: x.cpu().numpy().astype(np.int64)
    sid = host(prog.state_id).reshape(n + 1, W)
    entry_ptr = host(tt.entry_ptr).reshape(n, W + 1)
    entry_succ, entry_x = host(tt.entry_succ), host(tt.entry_x)
    pred_ptr, pred_x = host(tt.pred_ptr).reshape(n, W + 1), host(tt.pred_x)
    enum = en.num > 0
    rows = host(job[0])                                            # (input row, position, rank), by position
    mass_row = host(masses.row)[rows[:, 0] - prog.input_start]

    # every column (t, q) and its pairs (entries: by column, then successor)
    T = np.repeat(np.arange(n, dtype = np.int64), width[:n])
    Q = np.arange(T.size, dtype = np.int64) - np.repeat(np.cumsum(width[:n]) - width[:n], width[:n])
    es, ee = entry_ptr[T, Q], entry_ptr[T, Q + 1]
    npairs = ee - es
    cstart = np.cumsum(npairs) - npairs
    ecol = np.repeat(np.arange(T.size), npairs)                    # every (column, pair) visit
    ek = np.arange(ecol.size, dtype = np.int64) - cstart[ecol]
    eidx = es[ecol] + ek
    gfirst = es - entry_ptr[T, 0]                                  # a column's first pair's number (the kernels')
    if not np.array_equal(entry_x[eidx], gfirst[ecol] + ek):      # (its pairs: consecutive numbers from it)
        return None
    end = T[ecol] + 1 == n
    succ_state = np.where(end, -7, sid[np.minimum(T[ecol] + 1, n), np.where(end, 0, entry_succ[eidx])])
    # its enumerated tokens: (token's column of the probabilities, pair relative to the column's first)
    if enum:
        enum_ptr = host(tt.enum_ptr).reshape(n, W + 1)
        enum_e, enum_pair = host(tt.enum_e), host(tt.enum_pair)
        tok_ptr, tok_index = host(en.ptr), host(en.index)
        ns_, ne_ = enum_ptr[T, Q], enum_ptr[T, Q + 1]
        nenum = ne_ - ns_
        nstart = np.cumsum(nenum) - nenum
        ncol = np.repeat(np.arange(T.size), nenum)
        nk = np.arange(ncol.size, dtype = np.int64) - nstart[ncol]
        nidx = ns_[ncol] + nk
        tok = tok_index[tok_ptr[T[ncol]] + enum_e[nidx]]
        rel = enum_pair[nidx] - gfirst[ncol]
        evals = tok * 65536 + rel
    else:
        nenum = np.zeros(T.size, dtype = np.int64)
        nstart, ncol, evals = nenum, np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64)
    keys = np.stack([sid[T, Q], npairs, nenum, (T + 1 == n).astype(np.int64)] +
                    [_segment_hash(succ_state, ecol, cstart, npairs, s).view(np.int64) for s in (1, 2)] +
                    [_segment_hash(evals, ncol, nstart, nenum, s).view(np.int64) for s in (3, 4)], axis = 1)
    _, first, sig = np.unique(keys, axis = 0, return_index = True, return_inverse = True)
    sig = sig.reshape(-1)
    rep = first[sig]                                               # every column's signature's first column
    # exact: every column's operands are its first's (no hash collision)
    if not np.array_equal(succ_state, succ_state[cstart[rep[ecol]] + ek]):
        return None
    if enum and not np.array_equal(evals, evals[nstart[rep[ncol]] + nk]):
        return None
    S = first.size
    sig_pairs = npairs[first]
    sig_off = np.cumsum(sig_pairs) - sig_pairs
    slots = int(sig_pairs.sum())

    # the positions' classes: the same class-mass rows in rank order
    pos, counts = np.unique(rows[:, 1], return_counts = True)
    steps = np.zeros(n, dtype = np.int64)
    steps[pos] = counts
    order = np.lexsort((rows[:, 2], rows[:, 1]))
    by_pos = np.split(mass_row[order], np.cumsum(counts)[:-1])
    cls_of, classes = np.full(n, -1, dtype = np.int64), {}
    for t, mr in zip(pos.tolist(), by_pos):
        cls_of[t] = classes.setdefault(mr.tobytes(), len(classes))
    csteps = np.zeros(len(classes), dtype = np.int64)
    for t in pos.tolist():
        csteps[cls_of[t]] = steps[t]
    cbase = np.cumsum(csteps * slots) - csteps * slots
    size = int((csteps * slots).sum())
    if size * SIGNATURE_MIN_SAVING > trans_size:
        return None
    pair_off = np.where(cls_of >= 0, cbase[np.maximum(cls_of, 0)], 0)
    pairs = np.where(cls_of >= 0, slots, 0)

    # representatives: every (class, signature)'s first column at a position of the class's
    live = cls_of[T] >= 0
    cand = np.nonzero(live)[0]
    _, rfirst = np.unique(np.stack([cls_of[T[cand]], sig[cand]], axis = 1), axis = 0, return_index = True)
    reps = np.sort(cand[rfirst])                                   # by position, then column
    rep_t, rep_q = T[reps], Q[reps]
    rep_shift = sig_off[sig[reps]] - gfirst[reps]
    rep_count = np.bincount(rep_t, minlength = n)
    rep_ptr = np.concatenate([[0], np.cumsum(rep_count)])

    # the entries' and predecessors' mass indices: their pair's slot (alike boundaries share entries: one value)
    exs = entry_x.copy()
    slot_e = sig_off[sig[ecol]] + ek
    exs[eidx] = slot_e
    if not np.array_equal(exs[eidx], slot_e):
        return None
    pairbase = np.concatenate([[0], np.cumsum(np.asarray(tt.num_pairs, dtype = np.int64))])
    pslot = np.full(int(pairbase[-1]), -1, dtype = np.int64)
    pslot[pairbase[T[ecol]] + entry_x[eidx]] = slot_e
    jw = np.where(np.arange(n) + 1 == n, 1, width[1:n + 1])       # columns at t + 1 (a sequence end: one)
    ps, pe = pred_ptr[np.arange(n), 0], pred_ptr[np.arange(n), jw]
    pt = np.repeat(np.arange(n, dtype = np.int64), pe - ps)
    pidx = ps[pt] + (np.arange(pt.size, dtype = np.int64) - np.repeat(np.cumsum(pe - ps) - (pe - ps), pe - ps))
    slot_p = pslot[pairbase[pt] + pred_x[pidx]]
    pxs = pred_x.copy()
    pxs[pidx] = slot_p
    if (slot_p < 0).any() or not np.array_equal(pxs[pidx], slot_p):
        return None

    keep = np.isin(rows[:, 1], np.unique(rep_t))
    i32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype = np.int32)).to(dev)
    i64 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype = np.int64)).to(dev)
    return SignatureMasses(pair_off = i64(pair_off), pairs = i64(pairs), entry_x = i32(exs), pred_x = i32(pxs),
                           size = size, job = transition_job(torch.from_numpy(rows[keep]), dev),
                           rep_ptr = i32(rep_ptr), rep_col = i32(rep_q), rep_shift = i64(rep_shift),
                           max_reps = int(rep_count.max(initial = 0)), saving = trans_size / max(size, 1))


@triton_jit
def _signature_transition_kernel(class_mars, mass_row, trans, rows, tiles, delta, state_id, col_of_state, entry_succ,
                                 entry_ptr, class_of_stored, enum_ptr, enum_e, enum_pair, tok_ptr, eidx, eprobs, prow,
                                 rep_ptr, rep_col, rep_shift, width, W, C, K, n, Cs, num_enum, input_start, num_qtiles,
                                 pair_off, pairs, pid0, TS: tl.constexpr, QT: tl.constexpr, TP: tl.constexpr,
                                 TC: tl.constexpr, TE: tl.constexpr, ENUM: tl.constexpr):
    """:func:`~.transitions._transition_kernel` (one set of masses, no per-sample weights) over every position's
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
    enum = en.num > 0
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
