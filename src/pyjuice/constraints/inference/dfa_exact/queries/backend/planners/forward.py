"""
The forward pass's plan: from a query's sum intervals to compute, per launch and mode (once, per sample), the steps to run and their sizes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit

from ....kernels.prod import Cut
from .cuts import IndexLists, _Flat, _Packed


#: the arrays of :func:`_forward_tables`, in the order they are packed: per product launch (in the Program's order) its
#: chunks (``(first, end)`` steps), each in runs of steps of one product interval, and its group sizes, each in runs of
#: groups of one interval; per sum layer part its dense runs' intervals and its fused node blocks in runs of one interval
_FW_FIELDS = ("launch_chunk_ptr", "chunk_lo", "chunk_hi", "chunk_run_ptr", "run_iv", "run_lo", "run_hi",
              "launch_gsize_ptr", "gsize_run_ptr", "grun_iv", "grun_lo", "grun_hi", "part_dense_ptr",
              "dense_iv", "part_brun_ptr", "brun_iv", "brun_lo", "brun_hi")


def _runs(keys: np.ndarray):
    """``(starts, ends)`` of the maximal runs of equal values in ``keys``."""
    if keys.size == 0:
        return np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return starts, np.r_[starts[1:], keys.size]


class _ForwardShape(NamedTuple):
    """What :func:`_forward_modes` and :func:`_forward_steps` read off the launches of a circuit's
    :class:`~...compiled.circuit.Assembly`, whatever the automaton (kept there).

    :ivar tables: :data:`_FW_FIELDS`, packed
    :ivar run_ptr, run_part: int64: the launches' parts (``Assembly.part_ptr``'s numbers) of every run of steps (a CSR)
    :ivar run_launch, brun_part: int64: the launch of every run of steps, the sum layer part of every run of node blocks
    :ivar full: per launch and mode, the :class:`~...kernels.prod.Cut` running all of its steps
    :ivar runs_of, ivs_of: per sum layer part, its dense runs, and the intervals of its fused node blocks' regions
    :ivar idle: every step with nothing to run (never mutated: a pass reads them)
    :ivar step_ranges: int64 [steps, 4]: every step's launches and sum layer parts ``(first, end, first, end)``
    """

    tables: "_Packed"
    run_ptr: np.ndarray
    run_part: np.ndarray
    run_launch: np.ndarray
    brun_part: np.ndarray
    full: list
    runs_of: list
    ivs_of: list
    idle: list
    step_ranges: np.ndarray


def _forward_shape(prog) -> _ForwardShape:
    """The :class:`_ForwardShape` of ``prog``'s :class:`~...compiled.circuit.Assembly`."""
    assembly = prog.assembly
    F = _Flat(*_FW_FIELDS)
    reg_iv = prog.reg_iv_np
    full, runs_of, ivs_of, ranges, idle = [], [], [], [], []
    run_ptr, run_part, run_launch, brun_part = [0], [], [], []
    part_first = assembly.part_ptr.tolist()
    nc = nr = ng = ngr = nd = nbr = 0
    L = P = 0
    for kind, prod_index, layers in prog.steps:
        L0, P0 = L, P
        if kind == "prod":
            idle.append((kind, prod_index, [([], [], 0, 0) for _ in layers]))
            for stages, _, _ in assembly.layers.get(prod_index, ()):
                for t in (t for st in stages for t in st):
                    S, iv = t.rows.size(0), t.iv.numpy()
                    walks = t.form in ("input_block", "block_input")
                    chunks = t.steps[1] if walks else [(0, S, None)]
                    part0 = part_first[L]
                    of_step = None if t.part_of is None else t.part_of.numpy()
                    F.add("launch_chunk_ptr", nc)
                    for s0, s1, _ in chunks:
                        F.add("chunk_lo", s0)
                        F.add("chunk_hi", s1)
                        F.add("chunk_run_ptr", nr)
                        a, b = _runs(iv[s0:s1])
                        F.add("run_iv", iv[s0:s1][a])
                        F.add("run_lo", a)
                        F.add("run_hi", b)
                        for x, y in zip(a.tolist(), b.tolist()):        # (a run's outputs: of the launch's parts)
                            run_part += [part0] if of_step is None else (part0 + np.unique(of_step[s0 + x:s0 + y])).tolist()
                            run_ptr.append(len(run_part))
                        run_launch += [L] * len(a)
                        nr += len(a)
                        nc += 1
                    F.add("launch_gsize_ptr", ng)
                    for firsts in (t.groups_host or {}).values():
                        giv = iv[firsts[:-1]]                         # a group's steps share their interval
                        a, b = _runs(giv)
                        F.add("gsize_run_ptr", ngr)
                        F.add("grun_iv", giv[a])
                        F.add("grun_lo", a)
                        F.add("grun_hi", b)
                        ngr += len(a)
                        ng += 1
                    groups = None if t.groups_host is None else {G: (None, firsts.size - 1) for G, firsts in t.groups_host.items()}
                    every = [(s0, s1, job, None, s1 - s0) for s0, s1, job in chunks]
                    full.append((Cut(every, groups, 6 * L, S), Cut(every, groups, 6 * L + 3, S)))
                    L += 1
        else:
            idle.append((kind, prod_index, [(block_size, [], []) for block_size, _ in layers]))
            for _, parts in layers:
                for dense, _, fused in parts:
                    runs = list(dense.items())
                    F.add("part_dense_ptr", nd)
                    F.add("dense_iv", [reg_iv[r] for (r, _), _ in runs])
                    nd += len(runs)
                    F.add("part_brun_ptr", nbr)
                    ivs_of.append(None if fused is None else np.unique(prog.reg_iv_all[fused[3].cpu().numpy()]))
                    if fused is not None:
                        biv = reg_iv[fused[3].cpu().numpy()]
                        a, b = _runs(biv)
                        F.add("brun_iv", biv[a])
                        F.add("brun_lo", a)
                        F.add("brun_hi", b)
                        brun_part += [P] * len(a)
                        nbr += len(a)
                    runs_of.append(runs)
                    P += 1
        ranges.append((L0, L, P0, P))
    for name, value in (("launch_chunk_ptr", nc), ("chunk_run_ptr", nr), ("launch_gsize_ptr", ng),
                        ("gsize_run_ptr", ngr), ("part_dense_ptr", nd), ("part_brun_ptr", nbr)):
        F.add(name, value)
    int64 = lambda x: np.asarray(x, dtype = np.int64)
    return _ForwardShape(_Packed(_FW_FIELDS, F.done()), int64(run_ptr), int64(run_part), int64(run_launch), int64(brun_part),
                         full, runs_of, ivs_of, idle, int64(ranges).reshape(-1, 4))


def _forward_tables(prog) -> _ForwardShape:
    """The :class:`_ForwardShape` of ``prog``'s :class:`~...compiled.circuit.Assembly`: built on first use, whatever the
    automaton."""
    shape = prog.assembly.forward_shape
    if shape is None:
        shape = prog.assembly.forward_shape = _forward_shape(prog)
    return shape


def forward_sizes(prog, todo: np.ndarray, sample: np.ndarray) -> tuple:
    """What ``prog``'s automaton says of a plan's forward (``todo`` / ``sample``: :func:`_forward_steps`'): ``(sizes,
    part slots)``, flat: per launch and mode (0: ``todo``) the most entry columns, exit columns and slots of its steps
    in that mode (:attr:`~...kernels.prod.Cut.at`), per sum layer part and mode the most slots of its fused node
    blocks there."""
    shape = _forward_tables(prog)
    t = shape.tables
    sizes = np.zeros((len(shape.full), 2, 3), dtype = np.int64)
    slots = np.zeros((len(shape.runs_of), 2), dtype = np.int64)
    counts = np.diff(shape.run_ptr)
    for m, keep in enumerate((todo, sample)):
        sel = keep[t.run_iv]
        if sel.any():
            owner = np.repeat(shape.run_launch[sel], counts[sel])
            first = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
            sizes[owner[first], m] = np.maximum.reduceat(prog.launch_parts[shape.run_part[np.repeat(sel, counts)]], first, axis = 0)
        sel = keep[t.brun_iv]
        if sel.any():
            owner = shape.brun_part[sel]
            first = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
            slots[owner[first], m] = np.maximum.reduceat(prog.iv_slots[t.brun_iv[sel]], first)
    return sizes.ravel().tolist(), slots.ravel().tolist()


@njit(cache = True)
def _forward_modes(todo, sample, data, offsets):
    """Per mode (0: ``todo``, the values kept once; 1: ``sample``, the per-sample ones; bool per sum interval) and per
    product launch of :func:`_forward_tables`: its steps and every chunk's and group size's in that mode -- count, and
    where their index lists (relative to the chunk, or the group number; written only when not every one is in) start
    in the returned buffer; per sum layer part, its dense runs in that mode, and its fused node blocks' count and index
    list. Every decision is per run of one interval: work proportional to the runs, and to the steps listed."""
    launch_chunk_ptr = data[offsets[0]:offsets[1]]
    chunk_lo = data[offsets[1]:offsets[2]]
    chunk_hi = data[offsets[2]:offsets[3]]
    chunk_run_ptr = data[offsets[3]:offsets[4]]
    run_iv = data[offsets[4]:offsets[5]]
    run_lo = data[offsets[5]:offsets[6]]
    run_hi = data[offsets[6]:offsets[7]]
    launch_gsize_ptr = data[offsets[7]:offsets[8]]
    gsize_run_ptr = data[offsets[8]:offsets[9]]
    grun_iv = data[offsets[9]:offsets[10]]
    grun_lo = data[offsets[10]:offsets[11]]
    grun_hi = data[offsets[11]:offsets[12]]
    part_dense_ptr = data[offsets[12]:offsets[13]]
    dense_iv = data[offsets[13]:offsets[14]]
    part_brun_ptr = data[offsets[14]:offsets[15]]
    brun_iv = data[offsets[15]:offsets[16]]
    brun_lo = data[offsets[16]:offsets[17]]
    brun_hi = data[offsets[17]:offsets[18]]
    NL, NC, NG, NP = launch_chunk_ptr.size - 1, chunk_lo.size, gsize_run_ptr.size - 1, part_brun_ptr.size - 1
    launch_count = np.zeros((NL, 2), dtype = np.int64)
    chunk_count = np.zeros((NC, 2), dtype = np.int64)
    chunk_start = np.zeros((NC, 2), dtype = np.int64)
    gsize_count = np.zeros((NG, 2), dtype = np.int64)
    gsize_start = np.zeros((NG, 2), dtype = np.int64)
    dense_sel = np.zeros((dense_iv.size, 2), dtype = np.bool_)
    part_count = np.zeros((NP, 2), dtype = np.int64)
    part_start = np.zeros((NP, 2), dtype = np.int64)
    total = 0
    for c in range(NC):
        total += chunk_hi[c] - chunk_lo[c]
    for j in range(grun_iv.size):
        total += grun_hi[j] - grun_lo[j]
    for j in range(brun_iv.size):
        total += brun_hi[j] - brun_lo[j]
    buf = np.empty(2 * total, dtype = np.int64)
    k = 0
    for m in range(2):
        keep = todo if m == 0 else sample
        for L in range(NL):
            for c in range(launch_chunk_ptr[L], launch_chunk_ptr[L + 1]):
                n = 0
                for j in range(chunk_run_ptr[c], chunk_run_ptr[c + 1]):
                    if keep[run_iv[j]]:
                        n += run_hi[j] - run_lo[j]
                chunk_count[c, m] = n
                launch_count[L, m] += n
                chunk_start[c, m] = k
                if 0 < n < chunk_hi[c] - chunk_lo[c]:                  # partly in: list its steps
                    for j in range(chunk_run_ptr[c], chunk_run_ptr[c + 1]):
                        if keep[run_iv[j]]:
                            for s in range(run_lo[j], run_hi[j]):
                                buf[k] = s
                                k += 1
            for G in range(launch_gsize_ptr[L], launch_gsize_ptr[L + 1]):
                n, groups = 0, 0
                for j in range(gsize_run_ptr[G], gsize_run_ptr[G + 1]):
                    groups += grun_hi[j] - grun_lo[j]
                    if keep[grun_iv[j]]:
                        n += grun_hi[j] - grun_lo[j]
                gsize_count[G, m] = n
                gsize_start[G, m] = k
                if 0 < n < groups:
                    for j in range(gsize_run_ptr[G], gsize_run_ptr[G + 1]):
                        if keep[grun_iv[j]]:
                            for g in range(grun_lo[j], grun_hi[j]):
                                buf[k] = g
                                k += 1
        for P in range(NP):
            for r in range(part_dense_ptr[P], part_dense_ptr[P + 1]):
                dense_sel[r, m] = keep[dense_iv[r]]
            n, blocks = 0, 0
            for j in range(part_brun_ptr[P], part_brun_ptr[P + 1]):
                blocks += brun_hi[j] - brun_lo[j]
                if keep[brun_iv[j]]:
                    n += brun_hi[j] - brun_lo[j]
            part_count[P, m] = n
            part_start[P, m] = k
            if 0 < n < blocks:
                for j in range(part_brun_ptr[P], part_brun_ptr[P + 1]):
                    if keep[brun_iv[j]]:
                        for b in range(brun_lo[j], brun_hi[j]):
                            buf[k] = b
                            k += 1
    return (launch_count, chunk_count, chunk_start, gsize_count, gsize_start, dense_sel, part_count, part_start, buf[:k])


def _forward_steps(prog, todo: np.ndarray, sample: np.ndarray, lists: IndexLists) -> tuple:
    """The steps of a :class:`QueryPlan` (``todo`` / ``sample``: the intervals whose values it computes once / per
    sample), from :func:`_forward_modes`, and the indices of those with anything to run, ascending; their index lists
    go into ``lists``. They say nothing of the automaton: what it says is by places in the plan's sizes
    (:func:`forward_sizes`)."""
    shape = _forward_tables(prog)
    t, launches, full, runs_of, ivs_of = shape.tables, prog.assembly.launches, shape.full, shape.runs_of, shape.ivs_of
    lc, cc, cs, gc, gs, run_sel, pc, ps, buf = _forward_modes(todo, sample, t.data, t.offsets)
    counts, part_counts, run_sel, run_ptr = lc.tolist(), pc.tolist(), run_sel.tolist(), t.part_dense_ptr.tolist()
    active = _forward_active(prog, lc, pc, run_sel, t.part_dense_ptr)       # (the steps with anything to run)

    def cut(L, m):
        la, count = launches[L], counts[L][m]
        if count == la.rows.size(0):
            return full[L][m]
        chunks = []
        for c, (s0, s1, job, _, _) in zip(range(t.launch_chunk_ptr[L], t.launch_chunk_ptr[L + 1]),
                                          full[L][0].chunks):
            n = cc[c, m]
            if n:
                chunks.append((s0, s1, job, None if n == s1 - s0 else lists.add(buf[cs[c, m]:cs[c, m] + n]), n))
        groups = None
        if la.groups_host is not None:
            groups = {}
            for G, (size, firsts) in zip(range(t.launch_gsize_ptr[L], t.launch_gsize_ptr[L + 1]),
                                         la.groups_host.items()):
                n = gc[G, m]
                groups[size] = (None if n == len(firsts) - 1 else lists.add(buf[gs[G, m]:gs[G, m] + n]), n)
        return Cut(chunks, groups, 3 * (2 * L + m), int(count))

    steps, busy = list(shape.idle), np.flatnonzero(active).tolist()
    for si in busy:
        kind, prod_index, layers = prog.steps[si]
        L, P = int(shape.step_ranges[si, 0]), int(shape.step_ranges[si, 2])
        layers_out = []
        if kind == "prod":
            for stages, num_temps, temp_no in layers:
                out = ([], [])
                for st in stages:
                    st_out = ([], [])
                    for la in st:
                        for m in (0, 1):
                            if counts[L][m]:
                                st_out[m].append((la, cut(L, m)))
                        L += 1
                    for m in (0, 1):
                        if st_out[m]:
                            out[m].append(st_out[m])
                layers_out.append((out[0] or (), out[1] or (), num_temps, temp_no))
        else:
            for block_size, parts in layers:
                out = ([], [])
                for _, num_edges, fused in parts:
                    r0 = run_ptr[P]
                    for m in (0, 1):
                        dense = [(r, c0, blocks, key) for k, ((r, c0), (blocks, key)) in enumerate(runs_of[P])
                                 if run_sel[r0 + k][m]]
                        n, f = part_counts[P][m], None
                        if n:
                            every = n == len(fused[3])
                            f = (fused, None if every else lists.add(buf[ps[P, m]:ps[P, m] + n]), int(n), 2 * P + m,
                                 ivs_of[P])
                        if dense or f is not None:
                            out[m].append((dense or (), num_edges, f))
                    P += 1
                layers_out.append((block_size, out[0] or (), out[1] or ()))
        steps[si] = (kind, prod_index, layers_out)
    return steps, busy


def _forward_active(prog, lc, pc, run_sel, part_dense_ptr) -> np.ndarray:
    """bool [steps]: the Program's steps with a launch computing something in a mode (``lc`` [launches, 2] counts),
    or a sum part with fused blocks (``pc`` [parts, 2]) or dense runs (``run_sel`` [runs, 2]) in one -- by prefix
    sums over the steps' ranges of launches and parts (:attr:`_ForwardShape.step_ranges`), so that a query whose steps
    mostly do nothing (decoding left to right: all but the target's) plans in time proportional to the rest."""
    ranges = prog.assembly.forward_shape.step_ranges
    lcum = np.concatenate([[0], np.cumsum(lc.any(axis = 1))])
    sel = np.asarray(run_sel).reshape(-1, 2).any(axis = 1) if len(run_sel) else np.zeros(0, dtype = bool)
    rcum = np.concatenate([[0], np.cumsum(sel)])
    dense = rcum[part_dense_ptr[1:]] - rcum[part_dense_ptr[:-1]] > 0 if pc.shape[0] else np.zeros(0, dtype = bool)
    pcum = np.concatenate([[0], np.cumsum(pc.any(axis = 1) | dense)])
    return (lcum[ranges[:, 1]] > lcum[ranges[:, 0]]) | (pcum[ranges[:, 3]] > pcum[ranges[:, 2]])
