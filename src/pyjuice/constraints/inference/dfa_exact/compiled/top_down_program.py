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
from typing import Any, NamedTuple, Optional

import numpy as np
import torch

from ..external_params import block_scale
from ..kernels.common import IDENTITY, SUM, TEMP, OneCopy, first_occurrence_groups, most_in_runs, node_groups, node_tiles, packed, \
    settled
from ..kernels.prod import skip_masks_for
from ..kernels.prod_top_down import TopDownLaunch
from ..kernels.score import ClassWeightLaunch, class_weight_launch, score_tables



@dataclass
class TopDownProgram:
    """
    Everything the top-down pass reads besides buffers and parameters, built from the forward's
    :class:`~.program.Program`.

    :ivar steps: the layers from the top down: ``("sum", prod_index, [(block_size, dense_runs, fused_groups)])`` --
        ``dense_runs`` a list of ``(region, c0, E, (lo, hi), blocks, key, add)`` (children ``c0 + lo .. c0 + hi`` of a
        dense run over ``E``), ``fused_groups`` ``{child block size: tables}``
        for :func:`~.sum.fused_sum_top_down` (with host copies under ``"host"``) -- and ``("prod", prod_index,
        [(fold_stages, stages, num_temps, layer's number)])``: ``fold_stages`` the forward launches that write scratch rows,
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
    :ivar shared: what query plans keep of these tables whatever the automaton (:mod:`..queries.backend.planners`), the
        circuit's: one dict for the programs of one :class:`~.circuit.Assembly`

    Its ``steps`` and ``shared`` are the :class:`~.circuit.Assembly`'s, one for every automaton of its pattern: what an
    automaton says of them -- the launches' most columns and slots -- is in ``prog.td_most``, ``prog.cw_most`` and
    ``prog.td_slots``, by their ``number``.
    """

    steps: list
    skip: Any
    dead: np.ndarray
    dead_region: np.ndarray
    iv_bounds: np.ndarray
    reg_iv: np.ndarray
    elem_regions: list
    shared: dict


def top_down_program(backend) -> TopDownProgram:
    """The :class:`TopDownProgram` of a backend: its circuit's child-major tables, shared with its other backends
    (:func:`top_down_template`), and what its layout says of them."""
    prog = backend.program()
    t = prog.assembly.top_down
    if t is None:
        t = prog.assembly.top_down = top_down_template(backend, prog)
    # what the automaton says of its launches, at once: their most columns and slots, a class weight launch's outputs'
    # columns, a fused group's regions' slots
    ptr = t.child_ptr
    prog.td_most = np.stack([most_in_runs(prog.entries[t.child_a], ptr), most_in_runs(prog.exits[t.child_b], ptr),
                             most_in_runs(prog.all_iv_slots[t.child_iv], ptr)], axis = 1).ravel().tolist()
    prog.cw_most = most_in_runs(prog.exits[t.cw_bound], t.cw_ptr).tolist()
    prog.td_slots = most_in_runs(np.asarray(prog.slots, dtype = np.int64)[t.group_reg], t.group_ptr).tolist()
    return TopDownProgram(steps = t.steps, skip = skip_masks_for(prog.reach, t.spec_table, device = prog.dev), dead = t.dead,
                          dead_region = t.dead_region, iv_bounds = t.iv_bounds, reg_iv = t.reg_iv,
                          elem_regions = t.elem_regions, shared = t.shared)


class _Child(NamedTuple):
    """A :class:`TopDownLaunch` but for what its automaton says: its columns and slots (of its children's distinct
    intervals: ``number``, its among the template's child launches, see :class:`_Template`), and, where ``c_full_at`` is
    not None, whether every pair of those intervals is kept (``kinds``: ``(child kind, parent kind, whether every pair of
    the output's is kept)`` then)."""

    form: str
    side: str
    kinds: tuple
    c_full_at: Optional[int]
    children: torch.Tensor
    ptr: torch.Tensor
    parents: torch.Tensor
    number: int
    host: dict
    tiles: Optional[torch.Tensor]
    height: int


class _Template(NamedTuple):
    """:func:`top_down_template`'s tables: ``steps``, :attr:`TopDownProgram.steps`, the circuit's alone; ``child_a``,
    ``child_b``, ``child_iv`` int64: the ends and numbers of every child launch's children's distinct intervals, the
    launch ``number``'s from ``child_ptr[number]``; ``cw_bound``, ``cw_ptr``: every class weight launch's outputs'
    boundaries (:class:`~..kernels.score.ClassWeightLaunch`, by its ``number``); ``group_reg``, ``group_ptr``: every fused
    group's sum regions (its ``number``)."""

    steps: list
    spec_table: np.ndarray
    dead: np.ndarray
    dead_region: np.ndarray
    iv_bounds: np.ndarray
    reg_iv: np.ndarray
    elem_regions: list
    shared: dict
    child_a: np.ndarray
    child_b: np.ndarray
    child_iv: np.ndarray
    child_ptr: np.ndarray
    cw_bound: np.ndarray
    cw_ptr: np.ndarray
    group_reg: np.ndarray
    group_ptr: np.ndarray


def _out_temp(launch) -> bool:
    return launch.kinds[2] if launch.form == "block_block" else launch.kinds[1]


def top_down_template(backend, prog) -> _Template:
    """The child-major tables of the launches of ``prog``'s :class:`~.circuit.Assembly`: the circuit's alone."""
    circuit, assembly = prog.assembly.circuit, prog.assembly
    dev, n = prog.dev, prog.n
    firsts = prog.firsts
    score = score_tables(prog)
    of = circuit.of
    iv_index = lambda a, b: of[np.asarray(a) * (n + 1) + np.asarray(b)].astype(np.int64)
    # what already has its top-down value: sum rows, every product layer group's element rows and every product layer's
    # scratch rows; what is never written has none (a node no chain of edges reaches) and is not read either
    sums = np.zeros(max(end for _, end, _ in backend.sum_regions), dtype = bool)
    sums[slice(*prog.root_rows)] = True
    elem_firsts = [first for first, _, _ in backend.element_regions]
    elements = [np.zeros(end - first, dtype = bool) for first, end, _ in backend.element_regions]
    temps = {}
    specs, spec_of = [], {}
    child_ends, group_regs, cw_bounds = [], [], []
    tables = OneCopy()                                    # (the launches' tables on the device: in one copy)

    def spec_index(spec):
        if spec not in spec_of:
            spec_of[spec] = len(specs)
            specs.append(spec)
        return spec_of[spec]

    def outputs_written(launch, rows, prod_index, layer):
        """bool [steps]: whether every step's output (``rows`` the launch's, numpy) has a top-down value."""
        if _out_temp(launch):
            return temps[layer][rows[:, 0]]
        return elements[prod_index][rows[:, 0] - elem_firsts[prod_index]]

    def child_launch(launch, side, prod_index, layer):
        """The :class:`_Child` of one side of a forward launch, or None when that operand is an input or no product of
        the launch has a top-down value."""
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
        ends = np.unique(a * (n + 2) + b)                             # (the children's distinct intervals)
        iv_a, iv_b = ends // (n + 2), ends % (n + 2)
        c_full_at = None
        if form == "copy":
            okinds = (c_kind, out_temp)
        elif form in ("input_block", "block_input"):
            c_full_at = 0
            okinds = (c_kind, out_temp, kinds[2][0])
        else:
            okinds = (c_kind, r_kind if side == "L" else l_kind, out_temp)
        i32 = lambda x: tables.add(np.asarray(x, dtype = np.int32))
        ch = i32(np.stack([child_row, child_reg, add, np.zeros_like(add)], axis = 1))
        child_iv, parent_iv = iv_index(a[ptr_np[:-1]], b[ptr_np[:-1]]), iv_index(pa, pb)
        host = dict(children = np.stack([child_row, child_reg, add], axis = 1).astype(np.int64),
                    ptr = ptr_np, parents = par, parent_iv = parent_iv, child_iv = child_iv,
                    parent_ivs = np.unique(parent_iv), child_ivs = np.unique(child_iv))
        tiles, height = None, 1
        if form in ("input_block", "block_input"):                # children walking the same entries, together
            host["walk"] = {}
            host["group"] = node_groups(a[ptr_np[:-1]], b[ptr_np[:-1]], np.diff(ptr_np), add)
            tiles, height = node_tiles(host["group"])
            tiles = tables.add(tiles)
        child_ends.append((iv_a, iv_b, iv_index(iv_a, iv_b)))
        return _Child(form, side, okinds, c_full_at, ch, i32(ptr_np), i32(par), len(child_ends) - 1, host, tiles, height)

    steps = []
    for kind, prod_index, layers in reversed(circuit.steps):
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
                for group in fused_groups.values():
                    group["number"] = len(group_regs)
                    group_regs.append(group["regions"])
                layers_out.append((block_size, dense_runs, fused_groups))
            steps.append(("sum", prod_index, layers_out))
        else:
            layers_out = []
            for li, (stages, num_temps, _) in enumerate(assembly.layers.get(prod_index, ())):
                temps[(prod_index, li)] = np.zeros(num_temps, dtype = bool)
                info = []
                for launches in reversed(stages):            # in the order run_backward launches them
                    out, cws = [], []
                    for launch in launches:
                        for side in (("L", "R") if launch.form == "block_block" else ("L",)):
                            la = child_launch(launch, side, prod_index, (prod_index, li))
                            if la is not None:
                                out.append(la)
                        alive = outputs_written(launch, launch.rows.numpy(), prod_index, (prod_index, li))
                        cw = class_weight_launch(launch, alive, prog, score, tables)
                        if cw is not None:
                            cw, boundaries = cw
                            cw.number = len(cw_bounds)
                            cw_bounds.append(boundaries)
                            cws.append(cw)
                    info.append((out, _inputs(launches), _folds([l for l in launches if _out_temp(l)]), cws))
                layers_out.append(info[::-1])
            steps.append(("prod", prod_index, layers_out))
    tables.upload(dev)
    joined = lambda k: np.concatenate([ends[k] for ends in child_ends]) if child_ends else np.zeros(0, dtype = np.int64)
    child_iv, child_ptr = joined(2), np.cumsum([0] + [ends[0].size for ends in child_ends], dtype = np.int64)
    all_full = np.logical_and.reduceat(prog.all_iv_full[child_iv], child_ptr[:-1]).tolist() if child_ends else []

    def child(c: _Child) -> TopDownLaunch:                     # (a pattern says which intervals keep every pair)
        kinds = c.kinds if c.c_full_at is None else (c.kinds[0], c.kinds[1], (all_full[c.number], c.kinds[2]))
        return TopDownLaunch(c.form, c.side, kinds, c.children.tensor, c.ptr.tensor, c.parents.tensor, c.host,
                             settled(c.tiles), c.height, c.number)

    for k, (kind, prod_index, layers_out), (_, _, layers) in zip(range(len(steps)), list(steps), reversed(assembly.steps)):
        if kind != "prod":
            continue
        out = []
        for info, (stages, num_temps, temp_no) in zip(layers_out, layers):
            fold_stages = [[l for l in launches if _out_temp(l)] for launches in stages]
            for _, _, _, cws in info:
                for cw in cws:
                    cw.dests, cw.ptr, cw.parents, cw.tiles = cw.dests.tensor, cw.ptr.tensor, cw.parents.tensor, cw.tiles.tensor
            out.append(([s for s in fold_stages if s], [([child(c) for c in children], inputs, folds, cws)
                                                        for children, inputs, folds, cws in info], num_temps, temp_no))
        steps[k] = ("prod", prod_index, out)
    int64 = lambda parts: np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype = np.int64)
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
    return _Template(steps, np.asarray(specs, dtype = np.int64).reshape(len(specs), 6), dead, dead_region, iv_bounds, iv_index(region_iv[:, 0], region_iv[:, 1]),
                     [(first, end) for first, end, _ in backend.element_regions], {}, joined(0), joined(1), child_iv, child_ptr,
                     int64([np.asarray(b, dtype = np.int64) for b in cw_bounds]),
                     np.cumsum([0] + [len(b) for b in cw_bounds], dtype = np.int64), int64(group_regs),
                     np.cumsum([0] + [r.size for r in group_regs], dtype = np.int64))


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


def _fused_runs(cols: torch.Tensor, pids: torch.Tensor, valid: torch.Tensor, BS: int, group = None, lc = None):
    """Per row of ``cols`` [K, E], its maximal runs of valid consecutive children with weights ``BS`` apart (and, given
    ``group``, in one group: the sum region of aliased children; given ``lc``, consecutive places), row by row: ``(row,
    first child, length, pid base, group, first place)``, int64 arrays of a value per run, on the host (the tables on
    the device, every edge of a layer: a few runs per row, usually)."""
    cont = valid[:, 1:] & valid[:, :-1] & (cols[:, 1:] == cols[:, :-1] + 1) & (pids[:, 1:] == pids[:, :-1] + BS)
    if group is not None:
        cont &= group[:, 1:] == group[:, :-1]
    if lc is not None:
        cont &= lc[:, 1:] == lc[:, :-1] + 1
    none = torch.zeros(valid.size(0), 1, dtype = torch.bool, device = valid.device)
    k, a = torch.nonzero(valid & ~torch.cat([none, cont], dim = 1), as_tuple = True)
    _, b = torch.nonzero(valid & ~torch.cat([cont, none], dim = 1), as_tuple = True)
    host = lambda x: x.long().cpu().numpy()
    return (host(k), host(cols[k, a]), host(b - a + 1), host(pids[k, a]),
            np.full(k.numel(), -1, dtype = np.int64) if group is None else host(group[k, a]),
            np.zeros(k.numel(), dtype = np.int64) if lc is None else host(lc[k, a]))


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
    firsts_t = torch.as_tensor(firsts, dtype = torch.long, device = dev)   # reg, place, gate row, gate size, grid row length
    gated = False
    for nids, cids, pids, nb_reg, _, aliased, gate in parts:
        gated = gate is not None
        nids_h = nids.cpu().long().numpy()
        at = np.nonzero(sums[nids_h[:, None] + np.arange(BS)].any(axis = 1))[0]
        if at.size == 0:
            continue
        kept = None if at.size == nids_h.size else torch.from_numpy(at).to(dev)
        rows_of = (lambda t: t) if kept is None else (lambda t: t[kept])   # (of the node blocks with a top-down value)
        nids_h, regs_h = nids_h[at], nb_reg.cpu().long().numpy()[at]
        if gated:
            g_row, g_cbs, g_ck = gate["row"][at], gate["cbs"][at], gate["ck_host"][at]
        else:
            g_row, g_cbs, g_ck = (np.full(at.size, v, dtype = np.int64) for v in (0, 1, 0))

        def collect(kind, run):
            k, c0, length, p0, creg, lc0 = run
            found[kind].append((c0, length, p0, creg, nids_h[k], regs_h[k], lc0, g_row[k], g_cbs[k], g_ck[k]))

        cols = rows_of(cids)
        collect("E", _fused_runs(cols, rows_of(pids), cols >= elem_first, BS,     # padding: below the region
                                 lc = rows_of(block_scale.child_places(gate, cids)) if gated else None))
        if aliased is not None:
            a_rows, a_regs, a_pids = (rows_of(t) for t in aliased)
            valid = a_rows >= 0
            collect("S", _fused_runs(torch.where(valid, a_rows + firsts_t[a_regs], -1), a_pids, valid, BS, a_regs,
                                     lc = rows_of(gate["a_lc"]) if gated else None))
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
                              regions = np.unique(cb_ivreg),               # (their slots: ``prog.td_slots``, by ``number``)
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
