"""
How a query plan (:mod:`.plans`) is built fast. The Program's launch tables are flattened once into packed int64
arrays (:func:`_forward_tables`, :func:`_top_down_tables`), and numba planners decide, in one pass over runs of steps
of one interval or source, which values every launch, dense run and node block computes and in which mode
(:func:`_forward_modes`, :func:`_top_down_modes`). Their output becomes the plan's steps (:func:`_forward_steps`,
:func:`_top_down_steps`): index lists into the Program's fixed launch tables (:class:`IndexLists`,
:func:`cut_launch`), copied to the device together.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from numba import njit

from ...kernels.common import IndexList, SUM, TEMP, node_tiles, pack_aligned
from ...kernels.prod import Cut, Launch



class IndexLists:
    """The index lists of one query plan, copied to the device together (a copy from the host waits for the stream:
    one per plan)."""

    def __init__(self):
        self.items = []

    def add(self, idx, height: Optional[int] = None) -> IndexList:
        item = IndexList(idx, height)
        self.items.append(item)
        return item

    def upload(self, dev):
        if self.items:
            buf, starts = pack_aligned([i.host for i in self.items], np.int32)
            buf = torch.from_numpy(buf).to(dev)
            for i, s in zip(self.items, starts.tolist()):
                i.tensor, i.host = buf[s:s + len(i.host)], None


def cut_launch(launch: Launch, sel: np.ndarray, lists: IndexLists) -> Optional[Cut]:
    """The :class:`Cut` of ``launch`` running the steps ``sel`` marks (bool [S], host), None if none; its index lists
    go into ``lists``."""
    count = int(sel.sum())
    if count == 0:
        return None
    every = count == sel.size
    if every:
        E_max, X_max, S_max = launch.E_max, launch.X_max, launch.S_max
    else:
        sizes = launch.sizes.numpy()[sel]
        E_max, X_max, S_max = (int(sizes[:, k].max()) for k in range(3))
    chunks, groups = [(0, sel.size, None)], None
    if launch.form in ("input_block", "block_input"):
        _, chunks, groups = launch.steps
    cuts = []
    for s0, s1, job in chunks:
        part = sel[s0:s1]
        c = int(part.sum())
        if c:
            cuts.append((s0, s1, job, None if c == s1 - s0 else lists.add(np.nonzero(part)[0]), c))
    gcuts = None
    if groups is not None:
        gcuts = {}
        for G, firsts in launch.groups_host.items():
            gsel = sel[firsts[:-1]]
            c = int(gsel.sum())
            gcuts[G] = (None if c == gsel.size else lists.add(np.nonzero(gsel)[0]), c)
    return Cut(cuts, gcuts, E_max, X_max, S_max, count)


#: the arrays of :func:`_forward_tables`, in the order they are packed: per product launch (in the Program's order) its
#: chunks (``(first, end)`` steps), each in runs of steps of one product interval with their most entry columns, exit
#: columns and slots, and its group sizes, each in runs of groups of one interval; per sum layer part its dense runs'
#: intervals and its fused node blocks in runs of one interval with their most slots
_FW_FIELDS = ("launch_chunk_ptr", "chunk_lo", "chunk_hi", "chunk_run_ptr", "run_iv", "run_lo", "run_hi", "run_E",
              "run_X", "run_S", "launch_gsize_ptr", "gsize_run_ptr", "grun_iv", "grun_lo", "grun_hi", "part_dense_ptr",
              "dense_iv", "part_brun_ptr", "brun_iv", "brun_lo", "brun_hi", "brun_slots")


class _Packed:
    """Int64 arrays packed into one (``data``, ``offsets``: a numba call pays per argument), readable by name."""

    def __init__(self, names, arrays: dict):
        parts = [np.asarray(arrays[k], dtype = np.int64).reshape(-1) for k in names]
        self.offsets = np.cumsum([0] + [p.size for p in parts]).astype(np.int64)
        self.data = np.concatenate(parts)
        for k, a, b in zip(names, self.offsets[:-1], self.offsets[1:]):
            setattr(self, k, self.data[a:b])


def _runs(keys: np.ndarray):
    """``(starts, ends)`` of the maximal runs of equal values in ``keys``."""
    if keys.size == 0:
        return np.zeros(0, dtype = np.int64), np.zeros(0, dtype = np.int64)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return starts, np.r_[starts[1:], keys.size]


def _forward_tables(prog):
    """``(tables, launches, full cuts, parts, part intervals)``: what :func:`_forward_modes` reads
    (:data:`_FW_FIELDS`), every product launch in the Program's order with the :class:`~.prod.Cut` running all of its
    steps, every sum layer part's dense runs, and the intervals of its fused node blocks' regions (``prog.intervals``'
    numbers, as a :class:`~...kernels.compact.CompactLayout` counts their lanes; None for none) -- built on first use,
    kept on the Program."""
    if getattr(prog, "fw_flat", None) is not None:
        return prog.fw_flat
    F = _Flat(*_FW_FIELDS)
    reg_iv, slots = np.asarray(prog.reg_iv_host), np.asarray(prog.slots)
    launches, full, runs_of, ivs_of = [], [], [], []
    nc = nr = ng = ngr = nd = nbr = 0
    for kind, _, layers in prog.steps:
        if kind == "prod":
            for stages, _, _ in layers:
                for launch in (la for st in stages for la in st):
                    S, iv, sizes = launch.rows.size(0), launch.iv.numpy(), launch.sizes.numpy()
                    chunks = launch.steps[1] if launch.form in ("input_block", "block_input") else [(0, S, None)]
                    F.add("launch_chunk_ptr", nc)
                    for s0, s1, _ in chunks:
                        F.add("chunk_lo", s0)
                        F.add("chunk_hi", s1)
                        F.add("chunk_run_ptr", nr)
                        a, b = _runs(iv[s0:s1])
                        F.add("run_iv", iv[s0:s1][a])
                        F.add("run_lo", a)
                        F.add("run_hi", b)
                        for k, name in enumerate(("run_E", "run_X", "run_S")):
                            F.add(name, [sizes[s0 + x:s0 + y, k].max() for x, y in zip(a, b)])
                        nr += len(a)
                        nc += 1
                    F.add("launch_gsize_ptr", ng)
                    for firsts in (launch.groups_host or {}).values():
                        giv = iv[firsts[:-1]]                         # a group's steps share their interval
                        a, b = _runs(giv)
                        F.add("gsize_run_ptr", ngr)
                        F.add("grun_iv", giv[a])
                        F.add("grun_lo", a)
                        F.add("grun_hi", b)
                        ngr += len(a)
                        ng += 1
                    launches.append(launch)
                    full.append(cut_launch(launch, np.ones(S, dtype = bool), IndexLists()))
        else:
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
                        bsl = slots[fused[3].cpu().numpy()]
                        a, b = _runs(biv)
                        F.add("brun_iv", biv[a])
                        F.add("brun_lo", a)
                        F.add("brun_hi", b)
                        F.add("brun_slots", [bsl[x:y].max() for x, y in zip(a, b)])
                        nbr += len(a)
                    runs_of.append(runs)
    for name, value in (("launch_chunk_ptr", nc), ("chunk_run_ptr", nr), ("launch_gsize_ptr", ng),
                        ("gsize_run_ptr", ngr), ("part_dense_ptr", nd), ("part_brun_ptr", nbr)):
        F.add(name, value)
    prog.fw_flat = (_Packed(_FW_FIELDS, F.done()), launches, full, runs_of, ivs_of)
    return prog.fw_flat


@njit(cache = True)
def _forward_modes(todo, sample, data, offsets):
    """Per mode (0: ``todo``, the values kept once; 1: ``sample``, the per-sample ones; bool per sum interval) and per
    product launch of :func:`_forward_tables`: its steps and every chunk's and group size's in that mode -- count, and
    where their index lists (relative to the chunk, or the group number; written only when not every one is in) start
    in the returned buffer -- and the most entry columns, exit columns and slots of its steps; per sum layer part, its
    dense runs in that mode, and its fused node blocks' count, most slots and index list. Every decision is per run of
    one interval: work proportional to the runs, and to the steps listed."""
    launch_chunk_ptr = data[offsets[0]:offsets[1]]
    chunk_lo = data[offsets[1]:offsets[2]]
    chunk_hi = data[offsets[2]:offsets[3]]
    chunk_run_ptr = data[offsets[3]:offsets[4]]
    run_iv = data[offsets[4]:offsets[5]]
    run_lo = data[offsets[5]:offsets[6]]
    run_hi = data[offsets[6]:offsets[7]]
    run_E = data[offsets[7]:offsets[8]]
    run_X = data[offsets[8]:offsets[9]]
    run_S = data[offsets[9]:offsets[10]]
    launch_gsize_ptr = data[offsets[10]:offsets[11]]
    gsize_run_ptr = data[offsets[11]:offsets[12]]
    grun_iv = data[offsets[12]:offsets[13]]
    grun_lo = data[offsets[13]:offsets[14]]
    grun_hi = data[offsets[14]:offsets[15]]
    part_dense_ptr = data[offsets[15]:offsets[16]]
    dense_iv = data[offsets[16]:offsets[17]]
    part_brun_ptr = data[offsets[17]:offsets[18]]
    brun_iv = data[offsets[18]:offsets[19]]
    brun_lo = data[offsets[19]:offsets[20]]
    brun_hi = data[offsets[20]:offsets[21]]
    brun_slots = data[offsets[21]:offsets[22]]
    NL, NC, NG, NP = launch_chunk_ptr.size - 1, chunk_lo.size, gsize_run_ptr.size - 1, part_brun_ptr.size - 1
    launch_count = np.zeros((NL, 2), dtype = np.int64)
    launch_max = np.zeros((NL, 2, 3), dtype = np.int64)
    chunk_count = np.zeros((NC, 2), dtype = np.int64)
    chunk_start = np.zeros((NC, 2), dtype = np.int64)
    gsize_count = np.zeros((NG, 2), dtype = np.int64)
    gsize_start = np.zeros((NG, 2), dtype = np.int64)
    dense_sel = np.zeros((dense_iv.size, 2), dtype = np.bool_)
    part_count = np.zeros((NP, 2), dtype = np.int64)
    part_slots = np.zeros((NP, 2), dtype = np.int64)
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
                        launch_max[L, m, 0] = max(launch_max[L, m, 0], run_E[j])
                        launch_max[L, m, 1] = max(launch_max[L, m, 1], run_X[j])
                        launch_max[L, m, 2] = max(launch_max[L, m, 2], run_S[j])
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
                    part_slots[P, m] = max(part_slots[P, m], brun_slots[j])
            part_count[P, m] = n
            part_start[P, m] = k
            if 0 < n < blocks:
                for j in range(part_brun_ptr[P], part_brun_ptr[P + 1]):
                    if keep[brun_iv[j]]:
                        for b in range(brun_lo[j], brun_hi[j]):
                            buf[k] = b
                            k += 1
    return (launch_count, launch_max, chunk_count, chunk_start, gsize_count, gsize_start, dense_sel, part_count,
            part_slots, part_start, buf[:k])


def _forward_steps(prog, todo: np.ndarray, sample: np.ndarray, lists: IndexLists) -> list:
    """The steps of a :class:`QueryPlan` (``todo`` / ``sample``: the intervals whose values it computes once / per
    sample), from :func:`_forward_modes`; their index lists go into ``lists``."""
    t, launches, full, runs_of, ivs_of = _forward_tables(prog)
    lc, lmax, cc, cs, gc, gs, run_sel, pc, pslots, ps, buf = _forward_modes(todo, sample, t.data, t.offsets)
    counts, part_counts, run_sel, run_ptr = lc.tolist(), pc.tolist(), run_sel.tolist(), t.part_dense_ptr.tolist()
    active = _forward_active(prog, lc, pc, run_sel, t.part_dense_ptr)       # (the steps with anything to run)

    def cut(L, m):
        la, count = launches[L], counts[L][m]
        if count == la.rows.size(0):
            return full[L]
        chunks = []
        for c, (s0, s1, job, _, _) in zip(range(t.launch_chunk_ptr[L], t.launch_chunk_ptr[L + 1]),
                                          full[L].chunks):
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
        return Cut(chunks, groups, int(lmax[L, m, 0]), int(lmax[L, m, 1]), int(lmax[L, m, 2]), int(count))

    idle = getattr(prog, "fw_idle_steps", None)             # every step with nothing to run, built once (never
    if idle is None:                                          # mutated: a pass reads them)
        idle = prog.fw_idle_steps = [(kind, prod_index, [([], [], 0, 0) for _ in layers] if kind == "prod" else
                                      [(block_size, [], []) for block_size, _ in layers])
                                     for kind, prod_index, layers in prog.steps]
    steps = list(idle)
    for si in np.flatnonzero(active).tolist():
        kind, prod_index, layers = prog.steps[si]
        L, P = int(prog.fw_step_ranges[si, 0]), int(prog.fw_step_ranges[si, 2])
        layers_out = []
        if kind == "prod":
            for stages, num_temps, temp_slots in layers:
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
                layers_out.append((out[0], out[1], num_temps, temp_slots))
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
                            f = (fused, None if every else lists.add(buf[ps[P, m]:ps[P, m] + n]), int(n),
                                 int(pslots[P, m]), ivs_of[P])
                        if dense or f is not None:
                            out[m].append((dense, num_edges, f))
                    P += 1
                layers_out.append((block_size, out[0], out[1]))
        steps[si] = (kind, prod_index, layers_out)
    return steps


def _forward_active(prog, lc, pc, run_sel, part_dense_ptr) -> np.ndarray:
    """bool [steps]: the Program's steps with a launch computing something in a mode (``lc`` [launches, 2] counts),
    or a sum part with fused blocks (``pc`` [parts, 2]) or dense runs (``run_sel`` [runs, 2]) in one -- by prefix
    sums over the steps' ranges of launches and parts (``prog.fw_step_ranges``, built once), so that a query whose
    steps mostly do nothing (decoding left to right: all but the target's) plans in time proportional to the rest."""
    ranges = getattr(prog, "fw_step_ranges", None)
    if ranges is None:
        ranges, L, P = [], 0, 0
        for kind, _, layers in prog.steps:
            L0, P0 = L, P
            if kind == "prod":
                L += sum(len(st) for stages, _, _ in layers for st in stages)
            else:
                P += sum(len(parts) for _, parts in layers)
            ranges.append((L0, L, P0, P))
        prog.fw_step_ranges = ranges = np.asarray(ranges, dtype = np.int64).reshape(-1, 4)
    lcum = np.concatenate([[0], np.cumsum(lc.any(axis = 1))])
    sel = np.asarray(run_sel).reshape(-1, 2).any(axis = 1) if len(run_sel) else np.zeros(0, dtype = bool)
    rcum = np.concatenate([[0], np.cumsum(sel)])
    dense = rcum[part_dense_ptr[1:]] - rcum[part_dense_ptr[:-1]] > 0 if pc.shape[0] else np.zeros(0, dtype = bool)
    pcum = np.concatenate([[0], np.cumsum(pc.any(axis = 1) | dense)])
    return (lcum[ranges[:, 1]] > lcum[ranges[:, 0]]) | (pcum[ranges[:, 3]] > pcum[ranges[:, 2]])


class _Flat:
    """Arrays built by appending, then concatenated (see :func:`_top_down_tables`)."""

    def __init__(self, *names):
        self._parts = {name: [] for name in names}

    def add(self, name, values):
        self._parts[name].append(np.asarray(values, dtype = np.int64).reshape(-1))

    def count(self, name) -> int:
        return sum(a.size for a in self._parts[name])

    def done(self) -> dict:
        return {k: np.concatenate(v) if v else np.zeros(0, dtype = np.int64) for k, v in self._parts.items()}


#: the arrays of :func:`_top_down_tables` (the :class:`~.top_down_program.TopDownProgram` flattened, in the order
#: :func:`_top_down_modes` walks it: its product layers from the bottom up, each a range of stages in the forward's
#: order, each a range of launches -- their children in runs of one source, a sum region or a scratch row, every run's
#: parents' output rows as ranges and its siblings' distinct operands --, of input steps, of class weight launches --
#: their destinations, each a range of edges -- and of fold rows -- each a range of sum regions; every layer's fold
#: launches' rows; then the sum layers' dense runs and fused child blocks, in groups), in the order they are packed;
#: last, a stage's input steps and a class weight launch's destinations BY POSITION (:func:`_position_groups`): a
#: query visits its target positions' alone, not every position's
_TD_FIELDS = ("layer_pi", "layer_temps", "layer_stage_ptr", "layer_foldrow_ptr", "stage_unit_ptr", "stage_input_ptr", "stage_cw_ptr", "stage_fold_ptr", "unit_child_ptr", "unit_run_ptr", "run_child_ptr", "run_src", "run_iv", "run_par_ptr", "par_lo", "par_hi", "run_sib_ptr", "sib_id", "unit_child_temp", "unit_out_temp", "unit_bb", "unit_sib_temp", "input_out_temp", "input_row", "input_pos", "cw_dest_ptr", "cw_temp", "dest_pos", "dest_edge_ptr", "dest_edge_temp", "fold_out", "fold_temp", "fold_reg_ptr", "fold_reg", "foldrow_out", "elem_base", "sumlayer_pi", "sumlayer_run_ptr", "sumlayer_group_ptr", "group_block_ptr", "run_reg", "run_lo", "run_hi", "block_elem", "block_lo", "block_hi", "block_reg", "block_ivreg", "reg_iv", "ipos_ptr", "ipos_pos", "ipos_first", "ipos_end", "ipos_idx", "dpos_ptr", "dpos_pos", "dpos_first", "dpos_end", "dpos_idx")


def _position_groups(ptr: np.ndarray, pos: np.ndarray, n: int):
    """The items of every range of the CSR ``ptr`` (int64 [m + 1]) grouped by their position ``pos`` (int64 [items]):
    ``(gptr, gpos, gfirst, gend, idx)`` -- range ``r``'s groups are ``gptr[r] .. gptr[r + 1]``, group ``g`` the items
    ``idx[gfirst[g] .. gend[g]]`` at position ``gpos[g]``."""
    m = ptr.size - 1
    owner = np.repeat(np.arange(m, dtype = np.int64), np.diff(ptr))
    idx = np.lexsort((pos, owner))                                            # by range, then position
    keys = owner[idx] * (n + 1) + pos[idx]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if idx.size else np.zeros(0, dtype = np.int64)
    gptr = np.searchsorted(owner[idx][starts], np.arange(m + 1, dtype = np.int64)) if idx.size else \
        np.zeros(m + 1, dtype = np.int64)
    return gptr, pos[idx][starts], starts, np.r_[starts[1:], idx.size] if idx.size else starts, idx


def _top_down_tables(prog, td):
    """``(tables, index)``: the :func:`_top_down_tables` of ``td`` and, per step, where its launches are in them --
    per product layer ``([(first launch, first class weight launch)] per stage, [[(first row, end row)] per fold
    launch] per fold stage)``, per sum layer ``(first run, [(group, first block, end block)])`` (built on first use,
    kept on ``td``)."""
    if getattr(td, "flat", None) is not None:
        return td.flat
    F = _Flat("layer_pi", "layer_temps", "layer_stage_ptr", "layer_foldrow_ptr", "stage_unit_ptr", "stage_input_ptr",
              "stage_cw_ptr", "stage_fold_ptr", "unit_child_ptr", "unit_run_ptr", "run_child_ptr", "run_src",
              "run_iv", "run_par_ptr", "par_lo", "par_hi", "run_sib_ptr", "sib_id", "unit_child_temp", "unit_out_temp",
              "unit_bb",
              "unit_sib_temp", "input_out_temp",
              "input_row", "input_pos", "cw_dest_ptr", "cw_temp", "dest_pos", "dest_edge_ptr", "dest_edge_temp",
              "fold_out", "fold_temp", "fold_reg_ptr", "fold_reg", "foldrow_out", "sumlayer_pi", "sumlayer_run_ptr",
              "sumlayer_group_ptr", "group_block_ptr", "run_reg", "run_lo", "run_hi", "block_elem", "block_lo",
              "block_hi", "block_reg", "block_ivreg")
    index = [None] * len(td.steps)
    elem_base = np.cumsum([0] + [end - first for first, end in td.elem_regions])
    n_units = n_children = n_runs_c = n_cws = n_dests = n_dedges = n_folds = n_fregs = n_frows = 0
    n_pars = n_sibs = 0
    n_stages = 0
    for si in range(len(td.steps) - 1, -1, -1):           # the product layers, from the bottom up
        kind, pi, layers = td.steps[si]
        if kind != "prod":
            continue
        first = td.elem_regions[pi][0]
        recs = []
        for fold_stages, stages, num_temps, _ in layers:
            F.add("layer_pi", pi)
            F.add("layer_temps", max(num_temps, 1))
            F.add("layer_stage_ptr", n_stages)
            F.add("layer_foldrow_ptr", n_frows)
            stage_ids = []
            for launches, inputs, folds, cws in stages:
                F.add("stage_unit_ptr", n_units)
                F.add("stage_input_ptr", F.count("input_row"))
                F.add("stage_cw_ptr", n_cws)
                F.add("stage_fold_ptr", n_folds)
                stage_ids.append((n_units, n_cws))
                n_stages += 1
                for la in launches:
                    h = la.host
                    child_temp = la.kinds[0] != SUM
                    out_temp = _out_temp(la)
                    bb = 0 if la.form != "block_block" else (1 if la.side == "L" else 2)
                    sib_temp = bb == 1 and la.kinds[1] == TEMP
                    par = h["parents"]
                    src = h["children"][:, 0 if child_temp else 1]
                    starts = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])    # runs of one source
                    F.add("unit_child_ptr", n_children)
                    F.add("unit_run_ptr", n_runs_c)
                    F.add("run_child_ptr", n_children + starts)
                    F.add("run_src", src[starts])
                    F.add("run_iv", h["child_iv"][starts])
                    n_runs_c += len(starts)
                    out = par[:, 0] if out_temp else par[:, 0] - first
                    sib = par[:, 2] if bb == 2 else par[:, 3] if sib_temp else par[:, 4] if bb == 1 else None
                    ends = np.r_[starts[1:], len(src)]
                    for c0, c1 in zip(starts, ends):                    # a run's parents (ranges), its siblings
                        e0, e1 = h["ptr"][c0], h["ptr"][c1]
                        rows = np.unique(out[e0:e1])
                        cut = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1] + 1])
                        F.add("run_par_ptr", n_pars)
                        F.add("par_lo", rows[cut])
                        F.add("par_hi", np.r_[rows[cut[1:] - 1], rows[-1]] + 1 if rows.size else rows)
                        n_pars += len(cut)
                        sibs = np.unique(sib[e0:e1]) if sib is not None else np.zeros(0, dtype = np.int64)
                        F.add("run_sib_ptr", n_sibs)
                        F.add("sib_id", sibs)
                        n_sibs += len(sibs)
                    F.add("unit_child_temp", int(child_temp))
                    F.add("unit_out_temp", int(out_temp))
                    F.add("unit_bb", bb)
                    F.add("unit_sib_temp", int(sib_temp))
                    n_units += 1
                    n_children += len(h["children"])
                for out_temp, rows, pos in inputs:
                    F.add("input_out_temp", np.full(len(rows), int(out_temp)))
                    F.add("input_row", rows if out_temp else rows - first)
                    F.add("input_pos", pos)
                for cw in cws:
                    h = cw.host
                    temp = cw.form == "input_block" and cw.kinds[0] == TEMP
                    F.add("cw_dest_ptr", n_dests)
                    F.add("cw_temp", int(temp))
                    F.add("dest_pos", h["dests"][:, 0])
                    F.add("dest_edge_ptr", n_dedges + h["ptr"][:-1])
                    F.add("dest_edge_temp", h["parents"][:, 3] if temp else np.zeros(len(h["parents"]), dtype = np.int64))
                    n_cws += 1
                    n_dests += len(h["dests"])
                    n_dedges += len(h["parents"])
                for out_rows, temp, sums in folds:
                    F.add("fold_out", out_rows)
                    F.add("fold_temp", temp if temp is not None else np.full(len(out_rows), -1))
                    regs = np.stack(sums, axis = 1) if sums else np.zeros((len(out_rows), 0), dtype = np.int64)
                    F.add("fold_reg_ptr", n_fregs + np.arange(len(out_rows)) * regs.shape[1])
                    F.add("fold_reg", regs)
                    n_folds += len(out_rows)
                    n_fregs += regs.size
            fold_ids = []
            for launches in fold_stages:
                ids = []
                for la in launches:
                    rows = la.rows[:, 0].numpy()
                    F.add("foldrow_out", rows)
                    ids.append((n_frows, n_frows + len(rows)))
                    n_frows += len(rows)
                fold_ids.append(ids)
            recs.append((stage_ids, fold_ids))
        index[si] = recs
    F.add("layer_stage_ptr", n_stages)
    F.add("layer_foldrow_ptr", n_frows)
    F.add("stage_unit_ptr", n_units)
    F.add("stage_input_ptr", F.count("input_row"))
    F.add("stage_cw_ptr", n_cws)
    F.add("stage_fold_ptr", n_folds)
    F.add("unit_child_ptr", n_children)
    F.add("unit_run_ptr", n_runs_c)
    F.add("run_child_ptr", n_children)
    F.add("run_par_ptr", n_pars)
    F.add("run_sib_ptr", n_sibs)
    F.add("cw_dest_ptr", n_dests)
    F.add("dest_edge_ptr", n_dedges)
    F.add("fold_reg_ptr", n_fregs)
    n_runs = n_groups = n_blocks = 0
    for si, (kind, pi, layers) in enumerate(td.steps):  # the sum layers
        if kind != "sum":
            continue
        first, end = td.elem_regions[pi]
        recs = []
        for _, dense_runs, fused_groups in layers:
            F.add("sumlayer_pi", pi)
            F.add("sumlayer_run_ptr", n_runs)
            F.add("sumlayer_group_ptr", n_groups)
            for r, c0, _, (lo, hi), _, _, _ in dense_runs:
                F.add("run_reg", r)
                F.add("run_lo", c0 + lo - first)
                F.add("run_hi", c0 + hi - first)
            groups = []
            for cbs, tables in fused_groups.items():
                h = tables["host"]
                lo = np.clip(h["cb_row"] - first, 0, end - first)
                F.add("group_block_ptr", n_blocks)
                F.add("block_elem", h["cb_reg"] < 0)
                F.add("block_lo", lo)
                F.add("block_hi", np.minimum(lo + cbs, end - first))
                F.add("block_reg", np.maximum(h["cb_reg"], 0))
                F.add("block_ivreg", h["cb_ivreg"])
                groups.append((n_groups, n_blocks, n_blocks + len(lo)))
                n_groups += 1
                n_blocks += len(lo)
            recs.append((n_runs, groups))
            n_runs += len(dense_runs)
        index[si] = recs
    F.add("sumlayer_run_ptr", n_runs)
    F.add("sumlayer_group_ptr", n_groups)
    F.add("group_block_ptr", n_blocks)
    arrays = F.done()
    arrays.update(elem_base = elem_base, reg_iv = td.reg_iv)
    for name, ptr, pos in (("ipos", arrays["stage_input_ptr"], arrays["input_pos"]),
                           ("dpos", arrays["cw_dest_ptr"], arrays["dest_pos"])):
        groups = _position_groups(ptr, pos, prog.n)
        arrays.update({f"{name}_{k}": v for k, v in zip(("ptr", "pos", "first", "end", "idx"), groups)})
    td.flat = (_Packed(_TD_FIELDS, arrays), index)
    return td.flat


def _top_down_active(td, index, unit_count, cw_count, foldrow_mode, run_mode, group_count) -> np.ndarray:
    """bool [steps]: the top-down program's steps with anything computed -- a launch's child in a mode (``unit_count``
    [units, 2]), a class weight destination (``cw_count`` [launches]; None: no scoring), a fold row
    (``foldrow_mode`` >= 0), a dense run (``run_mode`` >= 0) or a fused block (``group_count`` [groups, 2]) -- by
    prefix sums over the steps' ranges in the flat tables (``td.step_ranges``, built once from ``index``)."""
    ranges = getattr(td, "step_ranges", None)
    if ranges is None:
        ranges = np.zeros((len(td.steps), 10), dtype = np.int64)            # u0 u1 w0 w1 f0 f1 r0 r1 g0 g1
        for si, (kind, pi, layers) in enumerate(td.steps):
            r = ranges[si]
            if kind == "prod":
                units = cws = 0
                u0 = w0 = f0 = f1 = -1
                for (fold_stages, stages, _, _), (stage_ids, fold_ids) in zip(layers, index[si]):
                    for (launches, _, _, cw_launches), (u, w) in zip(stages, stage_ids):
                        if u0 < 0:
                            u0, w0 = u, w
                        units += len(launches)
                        cws += len(cw_launches)
                    for ids in fold_ids:
                        for a, b in ids:
                            f0, f1 = (a if f0 < 0 else min(f0, a)), max(f1, b)
                r[:] = (max(u0, 0), max(u0, 0) + units, max(w0, 0), max(w0, 0) + cws, max(f0, 0), max(f1, 0),
                        0, 0, 0, 0)
            else:
                runs, groups = 0, []
                r0 = -1
                for (_, dense_runs, _), (first, grp) in zip(layers, index[si]):
                    if r0 < 0:
                        r0 = first
                    runs += len(dense_runs)
                    groups += [g for g, _, _ in grp]
                g0 = min(groups) if groups else 0
                r[:] = (0, 0, 0, 0, 0, 0, max(r0, 0), max(r0, 0) + runs, g0, g0 + len(groups))
        td.step_ranges = ranges
    cum = lambda flags: np.concatenate([[0], np.cumsum(flags)])
    on = lambda c, lo, hi: c[ranges[:, hi]] > c[ranges[:, lo]]
    active = on(cum(unit_count.any(axis = 1)), 0, 1) | on(cum(foldrow_mode >= 0), 4, 5) | \
        on(cum(run_mode >= 0), 6, 7) | on(cum(group_count.any(axis = 1)), 8, 9)
    if cw_count is not None:
        active |= on(cum(cw_count > 0), 2, 3)
    return active


def _mode_workspace(t, td):
    """The work arrays of :func:`_top_down_modes` sized by the program, kept on ``td`` across queries: the ones
    indexed by every launch child, class weight destination or product row -- a query writes a few of their entries
    and the next resets those (:func:`_top_down_modes` keeps the runs and launches it touched), so that a query pays
    for what it computes, not for the program's size."""
    work = getattr(td, "mode_work", None)
    if work is None:
        temps = int(t.layer_temps.max()) if t.layer_temps.size else 1
        work = td.mode_work = (np.full(int(t.unit_child_ptr[-1]), -1, dtype = np.int8),            # child_mode
                               np.zeros(t.run_src.size, dtype = np.bool_),                        # run_on
                               np.zeros(t.dest_pos.size, dtype = np.bool_),                        # dest_sel
                               np.zeros(t.cw_temp.size, dtype = np.int64),                        # cw_count
                               np.zeros(int(t.elem_base[-1]), dtype = np.bool_),                   # ereq
                               np.zeros(int(t.elem_base[-1]) + t.elem_base.size - 1, dtype = np.int64),   # E
                               np.zeros(t.elem_base.size - 1, dtype = np.bool_),                   # touched
                               np.zeros(temps, dtype = np.bool_), np.full(temps, -1, dtype = np.int8))   # treq, tin
    return work


@njit(cache = True)
def _top_down_modes(req, positions, once, scoring, data, offsets, child_mode, run_on, dest_sel, cw_count, ereq, E,
                    touched, treq_all, tin_all):
    """The plan of :func:`top_down_plan` over :func:`_top_down_tables`, in one pass from the bottom up: per product
    layer, the stages in the forward's order (a child computed -- a sum region ``req`` marks, or a fold a later
    stage's computed child reads -- marks its parents' outputs; an input at a target position its product's), then in
    the order they run (every computed child's mode, 1 kept once if its interval is, else 0 per sample; the siblings'
    and folds' inside values read in that mode); then the sum layers (a run or child block computed when one of its
    children's values is). Returns the modes (-1 not computed) of every launch child, fold row, dense run and child
    block, per launch and group the count in each mode, the class weight destinations at target positions with their
    count per launch, and the sum regions whose inside values a launch reads for a batch of one."""
    layer_pi = data[offsets[0]:offsets[1]]
    layer_temps = data[offsets[1]:offsets[2]]
    layer_stage_ptr = data[offsets[2]:offsets[3]]
    layer_foldrow_ptr = data[offsets[3]:offsets[4]]
    stage_unit_ptr = data[offsets[4]:offsets[5]]
    stage_input_ptr = data[offsets[5]:offsets[6]]
    stage_cw_ptr = data[offsets[6]:offsets[7]]
    stage_fold_ptr = data[offsets[7]:offsets[8]]
    unit_child_ptr = data[offsets[8]:offsets[9]]
    unit_run_ptr = data[offsets[9]:offsets[10]]
    run_child_ptr = data[offsets[10]:offsets[11]]
    run_src = data[offsets[11]:offsets[12]]
    run_iv = data[offsets[12]:offsets[13]]
    run_par_ptr = data[offsets[13]:offsets[14]]
    par_lo = data[offsets[14]:offsets[15]]
    par_hi = data[offsets[15]:offsets[16]]
    run_sib_ptr = data[offsets[16]:offsets[17]]
    sib_id = data[offsets[17]:offsets[18]]
    unit_child_temp = data[offsets[18]:offsets[19]]
    unit_out_temp = data[offsets[19]:offsets[20]]
    unit_bb = data[offsets[20]:offsets[21]]
    unit_sib_temp = data[offsets[21]:offsets[22]]
    input_out_temp = data[offsets[22]:offsets[23]]
    input_row = data[offsets[23]:offsets[24]]
    input_pos = data[offsets[24]:offsets[25]]
    cw_dest_ptr = data[offsets[25]:offsets[26]]
    cw_temp = data[offsets[26]:offsets[27]]
    dest_pos = data[offsets[27]:offsets[28]]
    dest_edge_ptr = data[offsets[28]:offsets[29]]
    dest_edge_temp = data[offsets[29]:offsets[30]]
    fold_out = data[offsets[30]:offsets[31]]
    fold_temp = data[offsets[31]:offsets[32]]
    fold_reg_ptr = data[offsets[32]:offsets[33]]
    fold_reg = data[offsets[33]:offsets[34]]
    foldrow_out = data[offsets[34]:offsets[35]]
    elem_base = data[offsets[35]:offsets[36]]
    sumlayer_pi = data[offsets[36]:offsets[37]]
    sumlayer_run_ptr = data[offsets[37]:offsets[38]]
    sumlayer_group_ptr = data[offsets[38]:offsets[39]]
    group_block_ptr = data[offsets[39]:offsets[40]]
    run_reg = data[offsets[40]:offsets[41]]
    run_lo = data[offsets[41]:offsets[42]]
    run_hi = data[offsets[42]:offsets[43]]
    block_elem = data[offsets[43]:offsets[44]]
    block_lo = data[offsets[44]:offsets[45]]
    block_hi = data[offsets[45]:offsets[46]]
    block_reg = data[offsets[46]:offsets[47]]
    block_ivreg = data[offsets[47]:offsets[48]]
    reg_iv = data[offsets[48]:offsets[49]]
    ipos_ptr = data[offsets[49]:offsets[50]]                                 # (input steps by position, per stage)
    ipos_pos = data[offsets[50]:offsets[51]]
    ipos_first = data[offsets[51]:offsets[52]]
    ipos_end = data[offsets[52]:offsets[53]]
    ipos_idx = data[offsets[53]:offsets[54]]
    dpos_ptr = data[offsets[54]:offsets[55]]                                 # (destinations by position, per launch)
    dpos_pos = data[offsets[55]:offsets[56]]
    dpos_first = data[offsets[56]:offsets[57]]
    dpos_end = data[offsets[57]:offsets[58]]
    dpos_idx = data[offsets[58]:offsets[59]]
    # the work arrays (:func:`_mode_workspace`): what the query before wrote, reset
    for j in range(run_on.size):
        if run_on[j]:
            child_mode[run_child_ptr[j]:run_child_ptr[j + 1]] = -1
            run_on[j] = False
    for w in range(cw_count.size):
        if cw_count[w] > 0:
            dest_sel[cw_dest_ptr[w]:cw_dest_ptr[w + 1]] = False
            cw_count[w] = 0
    for pi in range(touched.size):
        if touched[pi]:
            ereq[elem_base[pi]:elem_base[pi + 1]] = False
            touched[pi] = False
    unit_count = np.zeros((unit_child_temp.size, 2), dtype = np.int64)
    foldrow_mode = np.full(foldrow_out.size, -1, dtype = np.int8)
    prior = np.zeros(reg_iv.size, dtype = np.bool_)
    for L in range(layer_pi.size):
        base = elem_base[layer_pi[L]]
        treq = treq_all[:layer_temps[L]]
        tin = tin_all[:layer_temps[L]]
        treq[:] = False
        tin[:] = -1
        for s in range(layer_stage_ptr[L], layer_stage_ptr[L + 1]):   # the forward's order: what is computed
            for u in range(stage_unit_ptr[s], stage_unit_ptr[s + 1]):
                for j in range(unit_run_ptr[u], unit_run_ptr[u + 1]):         # children of one source
                    if treq[run_src[j]] if unit_child_temp[u] else req[run_src[j]]:
                        run_on[j] = True                                    # computed (its mode set below)
                        for k in range(run_par_ptr[j], run_par_ptr[j + 1]):
                            if unit_out_temp[u]:
                                treq[par_lo[k]:par_hi[k]] = True
                            else:
                                ereq[base + par_lo[k]:base + par_hi[k]] = True
                                touched[layer_pi[L]] = True
            for g in range(ipos_ptr[s], ipos_ptr[s + 1]):                 # the stage's inputs at target positions
                if positions[ipos_pos[g]]:
                    for k in range(ipos_first[g], ipos_end[g]):
                        i = ipos_idx[k]
                        if input_out_temp[i]:
                            treq[input_row[i]] = True
                        else:
                            ereq[base + input_row[i]] = True
                            touched[layer_pi[L]] = True
        for s in range(layer_stage_ptr[L + 1] - 1, layer_stage_ptr[L] - 1, -1):    # the order they run in
            if scoring:
                for w in range(stage_cw_ptr[s], stage_cw_ptr[s + 1]):
                    for g in range(dpos_ptr[w], dpos_ptr[w + 1]):          # its destinations at target positions
                        if not positions[dpos_pos[g]]:
                            continue
                        for k in range(dpos_first[g], dpos_end[g]):
                            d = dpos_idx[k]
                            dest_sel[d] = True
                            cw_count[w] += 1
                            if cw_temp[w]:                                  # a fold operand: per sample
                                for e in range(dest_edge_ptr[d], dest_edge_ptr[d + 1]):
                                    if tin[dest_edge_temp[e]] == 1:
                                        raise ValueError("a fold's inside value read once and per sample")
                                    tin[dest_edge_temp[e]] = 0
            for u in range(stage_unit_ptr[s], stage_unit_ptr[s + 1]):
                for j in range(unit_run_ptr[u], unit_run_ptr[u + 1]):
                    if not run_on[j]:
                        continue
                    mode = 1 if once[run_iv[j]] else 0
                    c0, c1 = run_child_ptr[j], run_child_ptr[j + 1]
                    child_mode[c0:c1] = mode
                    unit_count[u, mode] += c1 - c0
                    for k in range(run_sib_ptr[j], run_sib_ptr[j + 1]):   # the siblings' inside values, in that mode
                        if unit_sib_temp[u]:
                            tin[sib_id[k]] = mode
                        elif mode == 1:
                            prior[sib_id[k]] = True
            for f in range(stage_fold_ptr[s], stage_fold_ptr[s + 1]):    # a fold computed again: its operands too
                mode = tin[fold_out[f]]
                if mode >= 0:
                    if fold_temp[f] >= 0:
                        tin[fold_temp[f]] = mode
                    if mode == 1:
                        for k in range(fold_reg_ptr[f], fold_reg_ptr[f + 1]):
                            prior[fold_reg[k]] = True
        for r in range(layer_foldrow_ptr[L], layer_foldrow_ptr[L + 1]):
            foldrow_mode[r] = tin[foldrow_out[r]]
    run_mode = np.full(run_reg.size, -1, dtype = np.int8)
    block_mode = np.full(block_reg.size, -1, dtype = np.int8)
    group_count = np.zeros((group_block_ptr.size - 1, 2), dtype = np.int64)
    for pi in range(touched.size):                       # per touched group, from 0 at base + pi (never written):
        if touched[pi]:                                   # prefix sums of the products computed
            for k in range(elem_base[pi], elem_base[pi + 1]):
                E[k + pi + 1] = E[k + pi] + ereq[k]
    for S in range(sumlayer_pi.size):
        pi = sumlayer_pi[S]
        base = elem_base[pi] + pi
        for r in range(sumlayer_run_ptr[S], sumlayer_run_ptr[S + 1]):
            if touched[pi] and E[base + run_hi[r]] > E[base + run_lo[r]]:
                run_mode[r] = 1 if once[reg_iv[run_reg[r]]] else 0
        for g in range(sumlayer_group_ptr[S], sumlayer_group_ptr[S + 1]):
            for k in range(group_block_ptr[g], group_block_ptr[g + 1]):
                if block_elem[k]:
                    rq = touched[pi] and E[base + block_hi[k]] > E[base + block_lo[k]]
                else:
                    rq = req[block_reg[k]]
                if rq:
                    mode = 1 if once[reg_iv[block_ivreg[k]]] else 0
                    block_mode[k] = mode
                    group_count[g, mode] += 1
    return child_mode, unit_count, dest_sel, cw_count, foldrow_mode, run_mode, block_mode, group_count, prior


def _top_down_steps(td, t, index, m, scoring: bool, lists: IndexLists) -> list:
    """The steps of a top-down plan (:func:`~.plans.top_down_plan`), from :func:`_top_down_modes`'s
    output ``m`` over :func:`_top_down_tables`' ``t`` and ``index``: per product layer, its fold launches and, per stage
    in the order they run, its launches and class weight launches, each with the children it computes in a mode (kept
    once, per sample) as an index list added to ``lists`` (None for all of them); per sum layer, its dense runs and
    fused node blocks in each mode."""
    child_mode, unit_count, dest_sel, cw_count, foldrow_mode, run_mode, block_mode, group_count, _ = m
    active = _top_down_active(td, index, unit_count, cw_count if scoring else None, foldrow_mode, run_mode,
                              group_count)
    unit_count, cw_count, run_mode, group_count = (a.tolist() for a in (unit_count, cw_count, run_mode, group_count))
    child_ptr, dest_ptr = t.unit_child_ptr.tolist(), t.cw_dest_ptr.tolist()

    def selection(modes, mode, count):
        return None if count == modes.size else lists.add(np.nonzero(modes == mode)[0])

    def tiled(la, modes, mode, count):                 # (a walk kernel's: in node tiles)
        if la.form not in ("input_block", "block_input"):
            return selection(modes, mode, count)
        return None if count == modes.size else lists.add(*node_tiles(la.host["group"], np.nonzero(modes == mode)[0]))

    idle = getattr(td, "idle_steps", None)                  # every step with nothing computed, built once (never
    if idle is None:                                          # mutated: a pass reads them)
        idle = td.idle_steps = [(kind, pi, [(([], []), [], 0, 0) for _ in layers] if kind == "prod" else
                                 [(block_size, ([], []), ([], [])) for block_size, _, _ in layers])
                                for kind, pi, layers in td.steps]
    out = list(idle)
    for si in np.flatnonzero(active).tolist():
        kind, pi, layers = td.steps[si]
        layers_out = []
        if kind == "prod":
            for (fold_stages, stages, num_temps, temp_slots), (stage_ids, fold_ids) in zip(layers, index[si]):
                stages_out = []
                for (launches, _, _, cws), (u0, w0) in zip(stages[::-1], stage_ids[::-1]):   # the order they run in
                    once_l, sample_l, cw_l = [], [], []
                    for k, la in enumerate(launches):
                        u = u0 + k
                        for mode, lst in ((1, once_l), (0, sample_l)):
                            if unit_count[u][mode]:
                                modes = child_mode[child_ptr[u]:child_ptr[u + 1]]
                                lst.append((la, tiled(la, modes, mode, unit_count[u][mode])))
                    for k, cw in enumerate(cws if scoring else ()):
                        w = w0 + k
                        if cw_count[w]:
                            sel = dest_sel[dest_ptr[w]:dest_ptr[w + 1]]
                            cw_l.append((cw, tiled(cw, sel, 1, cw_count[w])))
                    stages_out.append((once_l, sample_l, cw_l))
                folds_out = ([], [])
                for launches, ids in zip(fold_stages, fold_ids):
                    cut = ([], [])
                    for la, (r0, r1) in zip(launches, ids):
                        for mode, lst in ((1, cut[0]), (0, cut[1])):
                            c = cut_launch(la, foldrow_mode[r0:r1] == mode, lists)
                            if c is not None:
                                lst.append((la, c))
                    for mode in (0, 1):
                        if cut[mode]:
                            folds_out[mode].append(cut[mode])
                layers_out.append((folds_out, stages_out, num_temps, temp_slots))
        else:
            for (block_size, dense_runs, fused_groups), (r0, groups) in zip(layers, index[si]):
                dense = ([], [])
                for k, run in enumerate(dense_runs):
                    if run_mode[r0 + k] >= 0:
                        dense[1 - run_mode[r0 + k]].append(run)
                fused = ([], [])
                for (cbs, tables), (g, b0, b1) in zip(fused_groups.items(), groups):
                    for mode, lst in ((1, fused[0]), (0, fused[1])):
                        if group_count[g][mode]:
                            lst.append((cbs, tables, selection(block_mode[b0:b1], mode, group_count[g][mode])))
                layers_out.append((block_size, dense, fused))
        out[si] = (kind, pi, layers_out)
    return out


def class_weight_writes(prog, steps: list, positions: np.ndarray) -> bool:
    """
    The class weight launches of a scoring plan's top-down ``steps`` (:func:`_top_down_steps`' output, its index lists
    not uploaded yet) as ``(launch, index list, store)``, in place: ``store`` where the launch is the first in the
    order they run to write every destination it writes -- it stores the weights, the later ones log-add to them.
    Returns whether ``dfa_class_flow`` must be filled with -inf first: some launch log-adds to a destination nothing
    wrote before, or some (target position, row) of the buffer (``positions``: bool [n]) is never written.
    """
    from ...kernels.score import score_tables
    score = score_tables(prog)
    K = np.array([len(score.rows[g]) if g >= 0 else 0 for g in score.group_of], dtype = np.int64)
    seen = np.zeros((prog.n, max(int(K.max(initial = 0)), 1)), dtype = bool)
    fill = False
    for step in steps:
        if step is None or step[0] != "prod":
            continue
        for _, stages, _, _ in step[2]:
            for _, _, cw_l in stages:
                for i, (la, item) in enumerate(cw_l):
                    d = la.host["dests"]
                    if item is not None:
                        rows = np.asarray(item.host).ravel()
                        d = d[rows[rows >= 0]]
                    first = ~seen[d[:, 0], d[:, 1]]
                    store = bool(first.all())
                    fill |= not store and bool(first.any())
                    seen[d[:, 0], d[:, 1]] = True
                    cw_l[i] = (la, item, store)
    covered = (np.arange(seen.shape[1])[None, :] < K[:, None]) <= seen
    return fill or not covered[np.asarray(positions, dtype = bool)].all()


def _out_temp(la) -> bool:
    """Whether a :class:`~.prod_top_down.TopDownLaunch`'s parents are scratch rows."""
    return bool(la.kinds[2] if la.form == "block_block" else la.kinds[1])
