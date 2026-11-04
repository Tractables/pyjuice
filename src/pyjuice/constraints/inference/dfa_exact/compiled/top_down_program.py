"""
What the top-down pass (:mod:`.queries.backend.top_down`) reads besides the buffers and the parameters, built once per
backend from the forward's :class:`~.program.Program` (:class:`TopDownProgram`).

Every table is CHILD-MAJOR: a program owns a child and sums over its parents, so nothing is accumulated with atomics
or scratch; a child reached by several launches (a sum shared by products in several layers) is written by the
first and log-added to by the rest (``add`` flags fixed when the tables are built, in launch order), so no buffer is
filled first. A node no chain of edges from the root reaches (pyjuice's PD has such nodes) has top-down value -inf:
nothing reads it, and its sum rows are filled with -inf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..external_params import block_scale
from ..kernels.common import IDENTITY, SUM, TEMP, first_occurrence_groups, node_groups, node_tiles, packed
from ..kernels.prod import skip_masks_for
from ..kernels.prod_top_down import TopDownLaunch
from ..kernels.score import class_weight_launch, score_tables



@dataclass
class TopDownProgram:
    """
    Everything the top-down pass reads besides buffers and parameters, built from the forward's
    :class:`~.program.Program`.

    :ivar steps: the layers from the top down: ``("sum", prod_index, [(block_size, dense_runs, fused_groups)])`` --
        ``dense_runs`` a list of ``(region, c0, E, (lo, hi), blocks, key, add)`` (children ``c0 + lo .. c0 + hi`` of a
        dense run over ``E``), ``fused_groups`` ``{child block size: tables}``
        for :func:`~.sum.fused_sum_top_down` (with host copies under ``"host"``) -- and ``("prod", prod_index,
        [(fold_stages, stages, num_temps, temp_slots)])``: ``fold_stages`` the forward launches that write scratch rows,
        ``stages`` per forward stage, in the forward's order, ``(launches, inputs, folds, class_weights)`` -- the
        stage's :class:`~.prod_top_down.TopDownLaunch` es, its products' input operands ``(out_temp, out rows,
        positions)``, per fold launch, ``(out rows, scratch operand rows or None, [sum operands' regions])``, and the
        :class:`~.score.ClassWeightLaunch` es of its input operands (a conditional's scoring reads them)
    :ivar skip: the skip masks of the ``block @ block`` top-down steps (their specs' index is the parents' 9th field)
    :ivar iv_bounds: int64 [intervals, 2] (host): ``prog.intervals``' intervals, by number
    :ivar reg_iv: int64 [sum regions] (host): every sum region's interval, by that number
    :ivar elem_regions: per product layer group, ``(first, end)`` element rows
    :ivar dead: int64 [D, 2] (host): the ``(row, region)`` of every sum node no chain of edges from the root reaches
        (pyjuice's PD has such nodes); no launch writes them: :func:`run_backward` fills those of the regions it
        computes with -inf (``prior_flows`` is -inf where never written)
    :ivar dead_region: bool [sum regions] (host): those with a dead row
    """

    steps: list
    skip: Any
    dead: np.ndarray
    dead_region: np.ndarray
    iv_bounds: np.ndarray
    reg_iv: np.ndarray
    elem_regions: list


def top_down_program(backend) -> TopDownProgram:
    prog = backend.program()
    dev, n = prog.dev, prog.n
    reach = prog.reach
    firsts = prog.firsts
    score = score_tables(prog)
    of = prog.intervals.of.cpu().numpy()
    iv_index = lambda a, b: of[np.asarray(a) * (n + 1) + np.asarray(b)].astype(np.int64)
    width = np.asarray(prog.width, dtype = np.int64)
    exits = width.copy()
    exits[n] = 1                                                      # a sequence end: one column
    # what already has its top-down value: sum rows, every product layer group's element rows and every product layer's
    # scratch rows; what is never written has none (a node no chain of edges reaches) and is not read either
    sums = np.zeros(max(end for _, end, _ in backend.sum_regions), dtype = bool)
    sums[slice(*prog.root_rows)] = True
    elem_firsts = [first for first, _, _ in backend.element_regions]
    elements = [np.zeros(end - first, dtype = bool) for first, end, _ in backend.element_regions]
    temps = {}
    specs, spec_of = [], {}

    def spec_index(spec):
        if spec not in spec_of:
            spec_of[spec] = len(specs)
            specs.append(spec)
        return spec_of[spec]

    def out_temp_of(launch):
        return launch.kinds[2] if launch.form == "block_block" else launch.kinds[1]

    def outputs_written(launch, rows, prod_index, layer):
        """bool [steps]: whether every step's output (``rows`` the launch's, numpy) has a top-down value."""
        if out_temp_of(launch):
            return temps[layer][rows[:, 0]]
        return elements[prod_index][rows[:, 0] - elem_firsts[prod_index]]

    def child_launch(launch, side, prod_index, layer):
        """The :class:`TopDownLaunch` of one side of a forward launch, or None when that operand is an input or no
        product of the launch has a top-down value."""
        form, kinds = launch.form, launch.kinds
        rows = launch.rows.numpy()
        live = outputs_written(launch, rows, prod_index, layer)
        if not live.any():
            return None
        rows = rows[live]
        if form == "copy":
            c_kind, out_temp = kinds
            crow, creg = rows[:, 1], rows[:, 2]
            a, b = rows[:, 3], rows[:, 4]
            spec = None
        elif form == "input_block":
            c_kind, out_temp = kinds[0], kinds[1]
            if c_kind == IDENTITY:
                return None
            crow, creg = rows[:, 3], rows[:, 4]
            a, b = rows[:, 2] + 1, rows[:, 5]
            spec = None
        elif form == "block_input":
            c_kind, out_temp = kinds[0], kinds[1]
            crow, creg = rows[:, 1], rows[:, 2]
            a, b = rows[:, 3], rows[:, 5]
            spec = None
        else:
            l_kind, r_kind, out_temp = kinds
            t0, t1, t2 = rows[:, 5], rows[:, 6], rows[:, 7]
            if side == "L":
                c_kind, crow, creg, a, b = l_kind, rows[:, 1], rows[:, 2], t0, t1
                spec = lambda r: ((int(r[5]), int(r[7]), False), (int(r[6]), int(r[7]), True))
            else:
                c_kind, crow, creg, a, b = r_kind, rows[:, 3], rows[:, 4], t1, t2
                spec = lambda r: ((int(r[5]), int(r[6]), True), (int(r[5]), int(r[7]), False))
        key_reg = creg if c_kind == SUM else np.zeros_like(creg)
        order = np.lexsort((key_reg, crow))                           # by child, the parents of one in launch order
        crow, key_reg = crow[order], key_reg[order]
        first = np.ones(crow.size, dtype = bool)
        first[1:] = (crow[1:] != crow[:-1]) | (key_reg[1:] != key_reg[:-1])
        starts = np.nonzero(first)[0]
        child_row, child_reg = crow[starts], key_reg[starts]
        ptr_np = np.append(starts, crow.size).astype(np.int64)
        kept = sums if c_kind == SUM else temps[layer]
        add = kept[child_row].astype(np.int64)                        # written by an earlier launch: log-added to
        kept[child_row] = True
        par = rows[order]
        if spec is not None:
            _, at, inverse = np.unique(packed(par[:, 5:8], n + 2), return_index = True, return_inverse = True)
            index = np.empty(at.size, dtype = np.int64)
            for u in np.argsort(at).tolist():                          # (numbered as they first occur)
                index[u] = spec_index(spec(par[at[u]]))
            par = np.concatenate([par, index[inverse][:, None]], axis = 1)
        a, b = a[order], b[order]
        out_a, out_b = {"copy": (3, 4), "input_block": (2, 5), "block_input": (3, None), "block_block": (5, 7)}[form]
        pa = par[:, out_a]
        pb = par[:, 5] + 1 if form == "block_input" else par[:, out_b]    # the product's interval
        ivs = [(int(v) // (n + 2), int(v) % (n + 2)) for v in np.unique(a * (n + 2) + b).tolist()]
        S_max = max(reach.slots(x, y) for x, y in ivs)
        if form == "copy":
            okinds = (c_kind, out_temp)
        elif form in ("input_block", "block_input"):
            c_full = all(reach.positions(x, y) is None for x, y in ivs)
            out_full = kinds[2][0]
            okinds = (c_kind, out_temp, (c_full, out_full))
        else:
            okinds = (c_kind, r_kind if side == "L" else l_kind, out_temp)
        i32 = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev, torch.int32)
        ch = i32(np.stack([child_row, child_reg, add, np.zeros_like(add)], axis = 1))
        child_iv, parent_iv = iv_index(a[ptr_np[:-1]], b[ptr_np[:-1]]), iv_index(pa, pb)
        host = dict(children = np.stack([child_row, child_reg, add], axis = 1).astype(np.int64),
                    ptr = ptr_np, parents = par, parent_iv = parent_iv, child_iv = child_iv,
                    parent_ivs = np.unique(parent_iv), child_ivs = np.unique(child_iv))
        tiles, height = None, 1
        if form in ("input_block", "block_input"):                # children walking the same entries, together
            host["group"] = node_groups(a[ptr_np[:-1]], b[ptr_np[:-1]], np.diff(ptr_np), add)
            tiles, height = node_tiles(host["group"])
            tiles = torch.from_numpy(tiles).to(dev)
        return TopDownLaunch(form, side, okinds, ch, i32(ptr_np), i32(par), int(width[a].max()), int(exits[b].max()),
                             S_max, host, tiles, height)

    steps = []
    for kind, prod_index, layers in reversed(prog.steps):
        if kind == "sum":
            layers_out = []
            elem_first = backend.element_regions[prod_index][0]
            for block_size, parts in layers:                # in the order run_backward launches: dense, then fused
                dense_runs = []
                for dense, E, _ in parts:
                    for (r, c0), (blocks, key) in dense.items():
                        nid = np.asarray([block[1] for block in blocks], dtype = np.int64)
                        if not sums[nid[:, None] + np.arange(block_size)].any():
                            continue                             # no node of the run has a top-down value
                        children = elements[prod_index][c0 - elem_first:c0 - elem_first + E]
                        hit = children.copy()
                        children[:] = True
                        for lo, hi, add in _segments(hit):       # written by an earlier launch: log-added to
                            dense_runs.append((r, c0, E, (lo, hi), blocks, key, add))
                fused_groups = _fused_tables(prog, [fused for _, _, fused in parts if fused is not None], block_size,
                                             elem_first, firsts, sums, elements[prod_index], dev)
                layers_out.append((block_size, dense_runs, fused_groups))
            steps.append(("sum", prod_index, layers_out))
        else:
            layers_out = []
            for li, (stages, num_temps, temp_slots) in enumerate(layers):
                temps[(prod_index, li)] = np.zeros(num_temps, dtype = bool)
                fold_stages = [[l for l in launches if out_temp_of(l)] for launches in stages]
                fold_stages = [s for s in fold_stages if s]
                info = []
                for launches in reversed(stages):            # in the order run_backward launches them
                    out, cws = [], []
                    for launch in launches:
                        for side in (("L", "R") if launch.form == "block_block" else ("L",)):
                            la = child_launch(launch, side, prod_index, (prod_index, li))
                            if la is not None:
                                out.append(la)
                        alive = outputs_written(launch, launch.rows.numpy(), prod_index, (prod_index, li))
                        cw = class_weight_launch(launch, alive, prog, score)
                        if cw is not None:
                            cws.append(cw)
                    info.append((out, _inputs(launches), _folds([l for l in launches if out_temp_of(l)]), cws))
                layers_out.append((fold_stages, info[::-1], num_temps, temp_slots))
            steps.append(("prod", prod_index, layers_out))
    region_first = np.asarray([first for first, _, _ in backend.sum_regions], dtype = np.int64)
    region_end = np.asarray([end for _, end, _ in backend.sum_regions], dtype = np.int64)
    rows = np.nonzero(~sums)[0]
    regs = np.searchsorted(region_first, rows, side = "right") - 1
    inside = regs >= 0
    inside[inside] = rows[inside] < region_end[regs[inside]]
    dead = np.stack([rows[inside], regs[inside]], axis = 1).astype(np.int64)
    keys = np.nonzero(of >= 0)[0]
    iv_bounds = np.zeros((int(of.max()) + 1, 2), dtype = np.int64)
    iv_bounds[of[keys]] = np.stack([keys // (n + 1), keys % (n + 1)], axis = 1)
    dead_region = np.zeros(len(backend.sum_regions), dtype = bool)
    dead_region[dead[:, 1]] = True
    region_iv = prog.iv_bounds[prog.reg_iv_np]
    return TopDownProgram(steps = steps, skip = skip_masks_for(reach, specs, device = dev),
                          dead = dead, dead_region = dead_region, iv_bounds = iv_bounds,
                          reg_iv = iv_index(region_iv[:, 0], region_iv[:, 1]),
                          elem_regions = [(first, end) for first, end, _ in backend.element_regions])


def _inputs(launches) -> list:
    """The input operands of a stage's forward launches: ``(out_temp, out rows, positions)`` (host)."""
    out = []
    for la in launches:
        if la.form == "input_block":                     # input u at t, then a block (or nothing)
            out.append((bool(la.kinds[1]), la.rows[:, 0].numpy(), la.rows[:, 2].numpy()))
        elif la.form == "block_input":                   # a block, then input u at t
            out.append((bool(la.kinds[1]), la.rows[:, 0].numpy(), la.rows[:, 5].numpy()))
    return out


def _folds(launches) -> list:
    """Per forward launch writing scratch rows (a fold): ``(out rows, scratch operand rows or None, [sum operands'
    regions])`` (host) -- what computing its rows again reads."""
    out = []
    for la in launches:
        r, sums, temp = la.rows, [], None
        if la.form == "block_block":
            sums.append(r[:, 2].numpy())
            if la.kinds[1] == SUM:
                sums.append(r[:, 4].numpy())
            else:
                temp = r[:, 3].numpy()
        elif la.form == "input_block":
            if la.kinds[0] == SUM:
                sums.append(r[:, 4].numpy())
            elif la.kinds[0] == TEMP:
                temp = r[:, 3].numpy()
        elif la.form == "block_input":
            sums.append(r[:, 2].numpy())
        out.append((r[:, 0].numpy(), temp, sums))
    return out


def _fused_runs(cols: np.ndarray, pids: np.ndarray, valid: np.ndarray, BS: int, group = None, lc = None):
    """Per row of ``cols`` [K, E], its maximal runs of valid consecutive children with weights ``BS`` apart (and, given
    ``group``, in one group: the sum region of aliased children; given ``lc``, consecutive places), row by row: ``(row,
    first child, length, pid base, group, first place)``, arrays of a value per run."""
    cont = valid[:, 1:] & valid[:, :-1] & (cols[:, 1:] == cols[:, :-1] + 1) & (pids[:, 1:] == pids[:, :-1] + BS)
    if group is not None:
        cont &= group[:, 1:] == group[:, :-1]
    if lc is not None:
        cont &= lc[:, 1:] == lc[:, :-1] + 1
    none = np.zeros((valid.shape[0], 1), dtype = bool)
    k, a = np.nonzero(valid & ~np.concatenate([none, cont], axis = 1))
    _, b = np.nonzero(valid & ~np.concatenate([cont, none], axis = 1))
    return (k, cols[k, a], b - a + 1, pids[k, a], np.full(k.size, -1, dtype = np.int64) if group is None else group[k, a],
            np.zeros(k.size, dtype = np.int64) if lc is None else lc[k, a])


def _fused_tables(prog, parts, block_size, elem_first, firsts, sums, elements, dev) -> dict:
    """Child-block-major tables of a sum layer's fused parts (see :func:`~.sum.fused_sum_top_down`), by child
    block size. A node block reads runs of consecutive children whose weights are ``BS`` apart; the runs of every node
    block of the layer are cut where any of them starts or ends, so two child blocks are equal or disjoint (one
    launch's programs never write the same row), and where an earlier launch's children end, so a child block is
    written or log-added to as a whole. Node blocks with no top-down value (no row written, ``sums``) add nothing: left
    out. ``elements``: which of the product layer group's rows (from ``elem_first``) have a top-down value.
    A gated layer's runs are also cut where their children's places among their node group's children stop being
    consecutive: every parent of a child block has its gate row, child gate size, grid row length and its first child's
    place (``gate``, see ``block_scale.edge_tables`` in :mod:`..external_params`), and the layer its node gate size.
    A layer under external parameters that is one dense run (``block_scale.dense_run``) whose child blocks tile its
    children, none of them written by an earlier launch, has in its one group's ``dense`` the run and its forward
    part -- what its products by sample need (see :func:`~..kernels.sum.per_sample_dense_sum_top_down`); else None."""
    BS = block_size
    found = {"E": [], "S": []}           # kind -> per part, a value per run: first row, length, pid base, region, nid,
    firsts_np = np.asarray(firsts)       # reg, place, gate row, gate size, grid row length
    gated = False
    for nids, cids, pids, nb_reg, _, aliased, gate in parts:
        gated = gate is not None
        nids_h = nids.cpu().long().numpy()
        at = np.nonzero(sums[nids_h[:, None] + np.arange(BS)].any(axis = 1))[0]
        if at.size == 0:
            continue
        nids_h, regs_h = nids_h[at], nb_reg.cpu().long().numpy()[at]
        cids_h, pids_h = cids.cpu().long().numpy()[at], pids.cpu().long().numpy()[at]
        if gated:
            g_row, g_cbs, g_ck = gate["row"][at], gate["cbs"][at], gate["ck_host"][at]
        else:
            g_row, g_cbs, g_ck = (np.full(at.size, v, dtype = np.int64) for v in (0, 1, 0))

        def collect(kind, run):
            k, c0, length, p0, creg, lc0 = run
            found[kind].append((c0, length, p0, creg, nids_h[k], regs_h[k], lc0, g_row[k], g_cbs[k], g_ck[k]))

        collect("E", _fused_runs(cids_h, pids_h, cids_h >= elem_first, BS,        # padding: below the region
                                 lc = gate["lc"][at] if gated else None))
        if aliased is not None:
            a_rows, a_regs, a_pids = (t.cpu().long().numpy()[at] for t in aliased)
            valid = a_rows >= 0
            collect("S", _fused_runs(np.where(valid, a_rows + firsts_np[a_regs], -1), a_pids, valid, BS, a_regs,
                                     lc = gate["a_lc"][at] if gated else None))
    # the child blocks: per kind, as they first occur; a block's parents, in their runs' order
    tables = []
    for kind, runs in found.items():
        if not runs:
            continue
        c0, length, p0, creg, nid, reg, lc0, g_row, g_cbs, g_ck = (np.concatenate(col) for col in zip(*runs))
        if c0.size == 0:
            continue
        cuts = np.unique(np.concatenate([c0, c0 + length]))
        lo, hi = np.searchsorted(cuts, c0), np.searchsorted(cuts, c0 + length)
        count = hi - lo
        run = np.repeat(np.arange(c0.size), count)
        cut = lo[run] + np.arange(run.size) - np.repeat(np.cumsum(count) - count, count)
        a, size = cuts[cut], cuts[cut + 1] - cuts[cut]
        block, first = first_occurrence_groups(a, creg[run], size)
        order = np.argsort(block, kind = "stable")
        p_count = np.bincount(block, minlength = first.size)
        p_first = np.cumsum(p_count) - p_count
        r = run[order]
        shift = a[order] - c0[r]
        tables.append(dict(kind = kind, row = a[first], reg = creg[run[first]], size = size[first], p_first = p_first,
                           p_count = p_count, nid = nid[r], par_reg = reg[r], pid = p0[r] + shift * BS, lc = lc0[r] + shift,
                           grow = g_row[r], gcbs = g_cbs[r], gck = g_ck[r]))
    # ... cut where an earlier launch's children end; by their size, as the sizes first occur
    segments = []
    for t in tables:
        size = t["size"]
        block = np.repeat(np.arange(size.size), size)
        c = np.arange(block.size) - np.repeat(np.cumsum(size) - size, size)
        rows = t["row"][block] + c
        kept, rows = (elements, rows - elem_first) if t["kind"] == "E" else (sums, rows)
        hit = kept[rows].copy()
        kept[rows] = True
        new = np.ones(block.size, dtype = bool)
        new[1:] = (hit[1:] != hit[:-1]) | (block[1:] != block[:-1])
        starts = np.nonzero(new)[0]
        ends = np.append(starts[1:], block.size)
        segments.append((t, block[starts], c[starts], ends - starts, hit[starts].astype(np.int64)))
    groups = {}
    if not segments:
        return groups
    sizes = np.concatenate([seg[3] for seg in segments])
    distinct = pd_unique(sizes)
    for length in distinct.tolist():
        cb_row, cb_reg, cb_add, cb_ivreg, counts = [], [], [], [], []
        p_nid, p_reg, p_pid, p_lc, p_grow, p_gcbs, p_gck = [], [], [], [], [], [], []
        for t, block, lo, seg_len, add in segments:
            at = np.nonzero(seg_len == length)[0]
            if at.size == 0:
                continue
            b, l = block[at], lo[at]
            cb_row.append(t["row"][b] + l)
            cb_reg.append(t["reg"][b])
            cb_add.append(add[at])
            cb_ivreg.append(t["par_reg"][t["p_first"][b]])
            count = t["p_count"][b]
            counts.append(count)
            idx = np.repeat(t["p_first"][b], count) + np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
            shift = np.repeat(l, count)
            p_nid.append(t["nid"][idx])
            p_reg.append(t["par_reg"][idx])
            p_pid.append(t["pid"][idx] + shift * BS)
            p_lc.append(t["lc"][idx] + shift)
            p_grow.append(t["grow"][idx])
            p_gcbs.append(t["gcbs"][idx])
            p_gck.append(t["gck"][idx])
        cb_row, cb_reg, cb_add, cb_ivreg, counts = (np.concatenate(x) for x in (cb_row, cb_reg, cb_add, cb_ivreg, counts))
        p_nid, p_reg, p_pid, p_lc, p_grow, p_gcbs, p_gck = (np.concatenate(x) for x in (p_nid, p_reg, p_pid, p_lc, p_grow,
                                                                                         p_gcbs, p_gck))
        ptr = np.concatenate([[0], np.cumsum(counts)])
        t32 = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev, torch.int32)
        dense = None              # one dense run under external parameters: its children these child blocks, unwritten
        run = parts[0][6].get("dense_run") if gated and len(parts) == 1 and distinct.size == 1 else None
        if run is not None and run["E"] % length == 0 and not cb_add.any() and (cb_reg < 0).all() and \
                np.array_equal(np.sort(cb_row), run["c0"] + length * np.arange(run["E"] // length)):
            dense = dict(run = run, part = parts[0])
        groups[length] = dict(cb_row = t32(cb_row), cb_reg = t32(cb_reg), cb_add = t32(cb_add), cb_ivreg = t32(cb_ivreg),
                              par_ptr = t32(ptr), par_nid = t32(p_nid), par_reg = t32(p_reg),
                              par_pid = torch.from_numpy(p_pid).to(dev, torch.int64),
                              max_slots = max(prog.slots[r] for r in np.unique(cb_ivreg).tolist()),
                              max_parents = int(np.diff(ptr).max()),
                              dense = dense,
                              gate = None if not gated else dict(
                                  layer = parts[0][6]["layer"], lc = t32(p_lc), row = t32(p_grow), cbs = t32(p_gcbs),
                                  ck = t32(p_gck), node_gate = parts[0][6]["node_gate"],
                                  uniform = block_scale.uniform_tiles(p_lc, p_gcbs, length)),
                              host = dict(cb_row = cb_row, cb_reg = cb_reg, cb_ivreg = cb_ivreg,
                                          ivs = np.unique(prog.reg_iv_all[cb_ivreg])))
    return groups


def pd_unique(values: np.ndarray) -> np.ndarray:
    """The distinct ``values``, as they first occur."""
    u, at = np.unique(values, return_index = True)
    return u[np.argsort(at)]


def _segments(flags: np.ndarray) -> list:
    """The maximal runs of equal values in ``flags``: ``[(start, end, value)]``."""
    flags = np.asarray(flags)
    cuts = (np.nonzero(flags[1:] != flags[:-1])[0] + 1).tolist()
    return [(s, e, int(flags[s])) for s, e in zip([0] + cuts, cuts + [flags.size])]
