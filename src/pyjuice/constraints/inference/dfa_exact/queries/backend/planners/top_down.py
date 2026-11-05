"""
The top-down pass's plan: from a query's values to keep, per launch and mode, the steps to run and their sizes; the class weights' writes.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ....kernels.common import SUM, TEMP, node_tiles
from .cuts import IndexLists, _Flat, _Packed, cut_launch


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
    kept with the circuit's tables, ``td.shared``: they say nothing of the automaton)."""
    if "flat" in td.shared:
        return td.shared["flat"]
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
    td.shared["flat"] = (_Packed(_TD_FIELDS, arrays), index)
    return td.shared["flat"]


def _top_down_active(td, index, unit_count, cw_count, foldrow_mode, run_mode, group_count) -> np.ndarray:
    """bool [steps]: the top-down program's steps with anything computed -- a launch's child in a mode (``unit_count``
    [units, 2]), a class weight destination (``cw_count`` [launches]; None: no scoring), a fold row
    (``foldrow_mode`` >= 0), a dense run (``run_mode`` >= 0) or a fused block (``group_count`` [groups, 2]) -- by
    prefix sums over the steps' ranges in the flat tables (``td.shared``' ``step_ranges``, built once from ``index``)."""
    ranges = td.shared.get("step_ranges")
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
        td.shared["step_ranges"] = ranges
    cum = lambda flags: np.concatenate([[0], np.cumsum(flags)])
    on = lambda c, lo, hi: c[ranges[:, hi]] > c[ranges[:, lo]]
    active = on(cum(unit_count.any(axis = 1)), 0, 1) | on(cum(foldrow_mode >= 0), 4, 5) | \
        on(cum(run_mode >= 0), 6, 7) | on(cum(group_count.any(axis = 1)), 8, 9)
    if cw_count is not None:
        active |= on(cum(cw_count > 0), 2, 3)
    return active


def _mode_workspace(t, td):
    """The work arrays of :func:`_top_down_modes` sized by the program, the ones indexed by every launch child, class
    weight destination or product row -- a query writes a few of their entries and the next resets those
    (:func:`_top_down_modes` keeps the runs and launches it touched), so that a query pays for what it computes, not for
    the program's size. Kept with the circuit's tables, taken from there while a plan reads them (the caller gives them
    back in ``td.shared``: a plan of another thread takes its own)."""
    work = td.shared.pop("mode_work", None)
    if work is None:
        temps = int(t.layer_temps.max()) if t.layer_temps.size else 1
        work = (np.full(int(t.unit_child_ptr[-1]), -1, dtype = np.int8),            # child_mode
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


def _top_down_steps(td, t, index, m, scoring: bool, lists: IndexLists) -> tuple:
    """The steps of a top-down plan (:func:`~.plans.top_down_plan`), from :func:`_top_down_modes`'s
    output ``m`` over :func:`_top_down_tables`' ``t`` and ``index``: per product layer, its fold launches and, per stage
    in the order they run, its launches and class weight launches, each with the children it computes in a mode (kept
    once, per sample) as an index list added to ``lists`` (None for all of them); per sum layer, its dense runs and
    fused node blocks in each mode -- the indices of the steps with anything to run, ascending, and what the fold
    launches' sizes are the most of (:func:`cut_parts`)."""
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

    idle = td.shared.get("idle_steps")                      # every step with nothing computed, built once (never
    if idle is None:                                          # mutated: a pass reads them)
        idle = td.shared["idle_steps"] = [(kind, pi, [(([], []), [], 0, 0) for _ in layers] if kind == "prod" else
                                           [(block_size, ([], []), ([], [])) for block_size, _, _ in layers])
                                          for kind, pi, layers in td.steps]
    out, busy = list(idle), np.flatnonzero(active).tolist()
    parts = []
    for si in busy:
        kind, pi, layers = td.steps[si]
        layers_out = []
        if kind == "prod":
            for (fold_stages, stages, num_temps, temp_no), (stage_ids, fold_ids) in zip(layers, index[si]):
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
                    stages_out.append((once_l or (), sample_l or (), cw_l or ()))
                folds_out = ([], [])
                for launches, ids in zip(fold_stages, fold_ids):
                    cut = ([], [])
                    for la, (r0, r1) in zip(launches, ids):
                        for mode, lst in ((1, cut[0]), (0, cut[1])):
                            c = cut_launch(la, foldrow_mode[r0:r1] == mode, lists, parts)
                            if c is not None:
                                lst.append((la, c))
                    for mode in (0, 1):
                        if cut[mode]:
                            folds_out[mode].append(cut[mode])
                layers_out.append(((folds_out[0] or (), folds_out[1] or ()), stages_out, num_temps, temp_no))
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
                layers_out.append((block_size, (dense[0] or (), dense[1] or ()), (fused[0] or (), fused[1] or ())))
        out[si] = (kind, pi, layers_out)
    return out, busy, parts


def class_weight_writes(prog, steps: list, positions: np.ndarray) -> bool:
    """
    The class weight launches of a scoring plan's top-down ``steps`` (:func:`_top_down_steps`' output, or those of its
    steps with anything to run, in order; the index lists not uploaded yet) as ``(launch, index list, store)``, in place: ``store`` where the launch is the first in the
    order they run to write every destination it writes -- it stores the weights, the later ones log-add to them.
    Returns whether ``dfa_class_flow`` must be filled with -inf first: some launch log-adds to a destination nothing
    wrote before, or some (target position, row) of the buffer (``positions``: bool [n]) is never written.
    """
    from ....kernels.score import score_tables
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
