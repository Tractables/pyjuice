"""
The lifted top-down pass: every sum node's TOP-DOWN value in ``node_flows`` (laid out as ``node_mars``), from the root
down -- the log of what ``p(C, e)`` multiplies the node's inside value by, over the positions outside its scope: its
log outside value, or pyjuice's top-down probability (:func:`~pyjuice.model.backend.eval_top_down_probs`) with the
evidence and the automaton's columns. The lifted circuit is a smooth, decomposable circuit (the PC times the
automaton's), so this is its backward pass, the pyjuice flow of a node being ``exp(node_mars + node_flows - log p(C,
e))``; the top-down value is kept rather than the flow because it depends only on the evidence outside the node's
scope, which a cache can key on.

Layer by layer from the top, mirroring :mod:`.forward`:

* a sum layer gives its children (the products below) their top-down values: dense node blocks by the transposed
  matrix product on the parameter buffer's own weight views, the others by a child-major kernel over pyjuice's edge
  tables (:func:`~.kernels.sum.dense_sum_top_down`, :func:`~.kernels.sum.fused_sum_top_down`); the values go to
  ``element_flows`` -- ``element_mars``'s storage, whose product insides the pass never reads -- or, for a product
  that was a copy of a sum (aliased, never materialized), straight to that sum's rows;
* a product layer gives its children their top-down values from its own: the transposed contractions of
  :mod:`.kernels.prod_top_down`, reading the siblings' inside values (sum rows, or a fold's scratch row, which the
  layer's fold steps compute again first).

Every table is CHILD-MAJOR: a program owns a child and sums over its parents, so nothing is accumulated with atomics
or scratch; a child reached by several launches (a sum shared by products in several layers) is written by the
first and log-added to by the rest (``add`` flags fixed when the tables are built, in launch order), so no buffer is
filled first. A node no chain of edges from the root reaches (pyjuice's PD has such nodes) has top-down value -inf:
nothing reads it, and its sum rows are filled with -inf.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from .kernels.prod import IDENTITY, SUM, TEMP, skip_masks_for, transition_tables
from .kernels.prod import run_products as products
from .kernels.prod_top_down import TopDownLaunch, run_top_down
from .kernels.sum import dense_sum_top_down, fill_dead_rows, fused_sum_top_down, stacked_weights
from .plan import _align
from . import forward as F


@dataclass
class TopDownProgram:
    """
    Everything the top-down pass reads besides buffers and parameters, built from the forward's
    :class:`~.forward.Program`.

    :ivar steps: the layers from the top down: ``("sum", prod_index, [(block_size, dense_runs, fused_groups)])`` --
        ``dense_runs`` a list of ``(region, c0, E, (lo, hi), blocks, key, add)`` (children ``c0 + lo .. c0 + hi`` of a
        dense run over ``E``), ``fused_groups`` ``{child block size: tables}``
        for :func:`~.kernels.sum.fused_sum_top_down` (with host copies under ``"host"``) -- and ``("prod", prod_index,
        [(fold_stages, stages, num_temps, temp_slots)])``: ``fold_stages`` the forward launches that write scratch rows,
        ``stages`` per forward stage, in the forward's order, ``(launches, inputs, folds)`` -- the stage's
        :class:`~.kernels.prod_top_down.TopDownLaunch` es, its products' input operands ``(out_temp, out rows,
        positions)`` and, per fold launch, ``(out rows, scratch operand rows or None, [sum operands' regions])``
    :ivar skip: the skip masks of the ``block @ block`` top-down steps (their specs' index is the parents' 9th field)
    :ivar iv_bounds: int64 [intervals, 2] (host): ``prog.intervals``' intervals, by number
    :ivar reg_iv: int64 [sum regions] (host): every sum region's interval, by that number
    :ivar elem_regions: per product layer group, ``(first, end)`` element rows
    :ivar dead: int64 [D, 2] (host): the ``(row, region)`` of every sum node no chain of edges from the root reaches
        (pyjuice's PD has such nodes); no launch writes them: :func:`run_backward` fills those of the regions it
        computes with -inf (``prior_flows`` is -inf where never written)
    """

    steps: list
    skip: Any
    dead: np.ndarray
    iv_bounds: np.ndarray
    reg_iv: np.ndarray
    elem_regions: list


def top_down_program(cc) -> TopDownProgram:
    prog = cc._lifted_program()
    dev, n = prog.dev, prog.n
    if not hasattr(prog, "trans_top_down"):                   # the transitions as the top-down kernels read them
        prog.trans_top_down = prog.trans if (not prog.trans.grouped or prog.trans_persistent) else \
            transition_tables(prog.next_col, prog.width, n, grouped = False)
    reach = prog.reach
    firsts = prog.firsts
    of = prog.intervals.of.cpu().numpy()
    iv_index = lambda a, b: of[np.asarray(a) * (n + 1) + np.asarray(b)].astype(np.int64)
    # what already has its top-down value: ("S", row, region) sum rows, ("E", group, row) element rows and ("T", layer,
    # row) scratch rows; what is never written has none (a node no chain of edges reaches) and is not read either
    written = {("S", row, prog.root_region) for row in range(*prog.root_rows)}
    specs, spec_of = [], {}

    def spec_index(spec):
        if spec not in spec_of:
            spec_of[spec] = len(specs)
            specs.append(spec)
        return spec_of[spec]

    def child_launch(launch, side, prod_index, layer):
        """The :class:`TopDownLaunch` of one side of a forward launch, or None when that operand is an input or no
        product of the launch has a top-down value."""
        form, kinds, rows = launch.form, launch.kinds, launch.rows
        out_temp = out_temp_of(launch)
        live = torch.tensor([(("T", layer, r) if out_temp else ("E", prod_index, r)) in written
                             for r in rows[:, 0].tolist()], dtype = torch.bool)
        if not live.any():
            return None
        rows, iv = rows[live], launch.iv[live]
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
        key_reg = creg if c_kind == SUM else torch.zeros_like(creg)
        keys = list(zip(crow.tolist(), key_reg.tolist()))
        order = sorted(range(len(keys)), key = lambda k: keys[k])
        children, ptr, add = [], [0], []
        for k in order:
            if not children or children[-1] != keys[k]:
                children.append(keys[k])
                ptr.append(ptr[-1])
                wk = ("S", keys[k][0], keys[k][1]) if c_kind == SUM else ("T", layer, keys[k][0])
                add.append(int(wk in written))
                written.add(wk)
            ptr[-1] += 1
        par = rows[order]
        if spec is not None:
            tri = torch.tensor([spec_index(spec(r)) for r in par.tolist()], dtype = torch.long)
            par = torch.cat([par, tri[:, None]], dim = 1)
        a, b = a[order], b[order]
        out_a, out_b = {"copy": (3, 4), "input_block": (2, 5), "block_input": (3, None), "block_block": (5, 7)}[form]
        pa = par[:, out_a].numpy()
        pb = (par[:, 5] + 1).numpy() if form == "block_input" else par[:, out_b].numpy()   # the product's interval
        ptr_np = np.asarray(ptr, dtype = np.int64)
        entry = torch.tensor([prog._entry(int(x)) for x in a])
        exit_ = torch.tensor([prog._exit(int(x)) for x in b])
        slots = torch.tensor([reach.slots(int(x), int(y)) for x, y in zip(a, b)])
        if form == "copy":
            okinds = (c_kind, out_temp)
        elif form in ("input_block", "block_input"):
            c_full = all(reach.positions(int(x), int(y)) is None for x, y in zip(a, b))
            out_full = kinds[2][0]
            okinds = (c_kind, out_temp, (c_full, out_full))
        else:
            okinds = (c_kind, r_kind if side == "L" else l_kind, out_temp)
        ch = torch.tensor([[r, g, ad, 0] for (r, g), ad in zip(children, add)], dtype = torch.int32, device = dev)
        host = dict(children = np.asarray([[r, g, ad] for (r, g), ad in zip(children, add)], dtype = np.int64),
                    ptr = ptr_np, parents = par.numpy(), parent_iv = iv_index(pa, pb),
                    child_iv = iv_index(a.numpy()[ptr_np[:-1]], b.numpy()[ptr_np[:-1]]))
        return TopDownLaunch(form, side, okinds, ch, torch.tensor(ptr, dtype = torch.int32, device = dev),
                             par.to(dev, torch.int32).contiguous(), int(entry.max()), int(exit_.max()),
                             int(slots.max()), host)

    def out_temp_of(launch):
        return launch.kinds[2] if launch.form == "block_block" else launch.kinds[1]

    steps = []
    for kind, prod_index, layers in reversed(prog.steps):
        if kind == "sum":
            layers_out = []
            for block_size, parts in layers:                # in the order run_backward launches: dense, then fused
                dense_runs = []
                for dense, E, _ in parts:
                    for (r, c0), (blocks, key) in dense.items():
                        if not any(("S", nid + i, r) in written for _, nid, _ in blocks for i in range(block_size)):
                            continue                             # no node of the run has a top-down value
                        wk = [("E", prod_index, c0 + e) for e in range(E)]
                        hit = [k in written for k in wk]
                        written.update(wk)
                        for lo, hi, add in _segments(hit):       # written by an earlier launch: log-added to
                            dense_runs.append((r, c0, E, (lo, hi), blocks, key, add))
                fused_groups = _fused_tables(prog, [fused for _, _, fused in parts if fused is not None], block_size,
                                             prod_index, cc.element_regions[prod_index][0], firsts, written, dev)
                layers_out.append((block_size, dense_runs, fused_groups))
            steps.append(("sum", prod_index, layers_out))
        else:
            layers_out = []
            for li, (stages, num_temps, temp_slots) in enumerate(layers):
                fold_stages = [[l for l in launches if out_temp_of(l)] for launches in stages]
                fold_stages = [s for s in fold_stages if s]
                info = []
                for launches in reversed(stages):            # in the order run_backward launches them
                    out = []
                    for launch in launches:
                        for side in (("L", "R") if launch.form == "block_block" else ("L",)):
                            la = child_launch(launch, side, prod_index, (prod_index, li))
                            if la is not None:
                                out.append(la)
                    info.append((out, _inputs(launches), _folds([l for l in launches if out_temp_of(l)])))
                layers_out.append((fold_stages, info[::-1], num_temps, temp_slots))
            steps.append(("prod", prod_index, layers_out))
    dead = [(row, r) for r, (first, end, _) in enumerate(cc.sum_regions) for row in range(first, end)
            if ("S", row, r) not in written]
    iv_bounds = np.zeros((int(of.max()) + 1, 2), dtype = np.int64)
    for k in np.nonzero(of >= 0)[0]:
        iv_bounds[of[k]] = (k // (n + 1), k % (n + 1))
    return TopDownProgram(steps = steps, skip = skip_masks_for(reach, specs, device = dev),
                          dead = np.asarray(dead, dtype = np.int64).reshape(-1, 2), iv_bounds = iv_bounds,
                          reg_iv = iv_index(*np.asarray([prog.sum_ivs[k] for k in prog.reg_iv_host]).T),
                          elem_regions = [(first, end) for first, end, _ in cc.element_regions])


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


def _fused_tables(prog, parts, block_size, prod_index, elem_first, firsts, written, dev) -> dict:
    """Child-block-major tables of a sum layer's fused parts (see :func:`~.kernels.sum.fused_sum_top_down`), by child
    block size. A node block reads runs of consecutive children whose weights are ``BS`` apart; the runs of every node
    block of the layer are cut where any of them starts or ends, so two child blocks are equal or disjoint (one
    launch's programs never write the same row), and where an earlier launch's children end, so a child block is
    written or log-added to as a whole. Node blocks with no top-down value (no row written) add nothing: left out."""
    BS = block_size

    def runs(cols, pid_row, valid, group = None):
        """Maximal runs of valid consecutive children with weights ``BS`` apart (and, given ``group``, in one group:
        the sum region of aliased children): [(first, length, pid base, group)]."""
        cont = valid[1:] & valid[:-1] & (cols[1:] == cols[:-1] + 1) & (pid_row[1:] == pid_row[:-1] + BS)
        if group is not None:
            cont &= group[1:] == group[:-1]
        starts = np.nonzero(valid & ~np.concatenate([[False], cont]))[0]
        ends = np.nonzero(valid & ~np.concatenate([cont, [False]]))[0]
        return [(int(cols[a]), int(b - a + 1), int(pid_row[a]), -1 if group is None else int(group[a]))
                for a, b in zip(starts, ends)]

    found = {"E": [], "S": []}                           # kind -> [(first row, length, pid base, region, nid, reg)]
    firsts_np = np.asarray(firsts)
    for nids, cids, pids, nb_reg, _, aliased in parts:
        nids_h, regs_h = nids.cpu().long().tolist(), nb_reg.cpu().long().tolist()
        cids_h, pids_h = cids.cpu().long().numpy(), pids.cpu().long().numpy()
        if aliased is not None:
            a_rows, a_regs, a_pids = (t.cpu().long().numpy() for t in aliased)
        for k, (nid, reg) in enumerate(zip(nids_h, regs_h)):
            if not any(("S", nid + i, reg) in written for i in range(BS)):
                continue
            for c0, length, p0, _ in runs(cids_h[k], pids_h[k], cids_h[k] >= elem_first):   # padding: below the region
                found["E"].append((c0, length, p0, -1, nid, reg))
            if aliased is not None:
                valid = a_rows[k] >= 0
                rows = np.where(valid, a_rows[k] + firsts_np[a_regs[k]], -1)
                for r0, length, p0, creg in runs(rows, a_pids[k], valid, a_regs[k]):
                    found["S"].append((r0, length, p0, creg, nid, reg))
    blocks = defaultdict(list)                           # (kind, first row, region, size) -> [(nid, reg, pid base)]
    for kind, rs in found.items():
        cuts = sorted({x for c0, length, *_ in rs for x in (c0, c0 + length)})
        for c0, length, p0, creg, nid, reg in rs:
            lo, hi = bisect_left(cuts, c0), bisect_left(cuts, c0 + length)
            for a, b in zip(cuts[lo:hi], cuts[lo + 1:hi + 1]):
                blocks[(kind, a, creg, b - a)].append((nid, reg, p0 + (a - c0) * BS))
    by_size = defaultdict(list)                          # child block size -> [(kind, first row, region, add, parents)]
    for (kind, row, creg, length), parents in blocks.items():
        keys = [(("E", prod_index, row + c) if kind == "E" else ("S", row + c, creg)) for c in range(length)]
        hit = [k in written for k in keys]
        written.update(keys)
        for lo, hi, add in _segments(hit):               # written by an earlier launch: log-added to
            by_size[hi - lo].append((kind, row + lo, creg, add, [(nid, reg, p0 + lo * BS) for nid, reg, p0 in parents]))
    groups = {}
    for length, items in by_size.items():
        cb_row, cb_reg, cb_add, cb_ivreg, ptr, p_nid, p_reg, p_pid = [], [], [], [], [0], [], [], []
        for kind, row, creg, add, parents in items:
            cb_row.append(row); cb_reg.append(creg); cb_add.append(add); cb_ivreg.append(parents[0][1])
            for nid, reg, p0 in parents:
                p_nid.append(nid); p_reg.append(reg); p_pid.append(p0)
            ptr.append(len(p_nid))
        t = lambda xs, dt = torch.int32: torch.tensor(xs, dtype = dt, device = dev)
        groups[length] = dict(cb_row = t(cb_row), cb_reg = t(cb_reg), cb_add = t(cb_add), cb_ivreg = t(cb_ivreg),
                              par_ptr = t(ptr), par_nid = t(p_nid), par_reg = t(p_reg), par_pid = t(p_pid, torch.int64),
                              max_slots = max(prog.slots[r] for r in cb_ivreg),
                              host = dict(cb_row = np.asarray(cb_row), cb_reg = np.asarray(cb_reg),
                                          cb_ivreg = np.asarray(cb_ivreg)))
    return groups


def _segments(flags) -> list:
    """The maximal runs of equal values in ``flags``: ``[(start, end, value)]``."""
    out, k = [], 0
    for j in range(1, len(flags) + 1):
        if j == len(flags) or flags[j] != flags[k]:
            out.append((k, j, int(flags[k])))
            k = j
    return out


def run_backward(cc, bufs: dict, flows: dict, plan, x: torch.Tensor, missing: torch.Tensor, obs_class: torch.Tensor,
                 pruning, precision: str, visit: Optional[Callable[[int], None]] = None):
    """
    The top-down values a query plan computes (``plan.top_down``, see :mod:`.query_plan`), after
    :func:`~.forward.run_forward` filled the inside values of ``bufs`` under ``plan``: the per-sample ones into
    ``flows["node_flows"]``, those kept once into ``prior_flows`` (:meth:`~pyjuice.constraints.ConstrainedCircuit.
    _prior_flows`), for a batch of one. Products' values pass through ``flows["element_flows"]`` and the folds' through
    the layer's scratch rows (per sample, or a batch of one in the same rows).

    :param visit: None, or called with a product layer group's index once its products' top-down values are in
        ``element_flows`` (per sample, or for a batch of one where the plan keeps them once), before the next group's
        overwrite them: where the values of the products of the targets' inputs are read
    """
    pc, prog, td = cc.pc, cc._lifted_program(), plan.top_down
    out = cc._top_down_program()
    B = x.size(0)
    lay = bufs["layout"]
    node_mars, element_mars = bufs["node_mars"], bufs["element_mars"]
    node_flows, element_flows = flows["node_flows"], flows["element_flows"]
    regions = dict(offset = lay["sum_offsets_t"], width = lay["sum_widths_t"], first = prog.reg_first,
                   slots = prog.reg_slots)
    prior = cc._prior_buffer() if plan.prior else None
    srcs = dict(mars = node_mars, offset = regions["offset"], width = regions["width"], reg_prior = plan.reg_prior)
    if prior is not None:
        srcs.update(mars = prior["mars"], offset = prior["offset_t"], width = prior["width_t"])
    pf = cc._prior_flows(td.once) if td.once.any() else None
    if pf is not None:                                   # where the values kept once go
        once = dict(offset = pf["offset_t"], width = pf["width_t"], first = prog.reg_first, slots = prog.reg_slots)
    dot = F.PRECISIONS[precision]
    live = pruning.live if pruning is not None else None
    lanes = (pruning.perm, pruning.reg_first, pruning.reg_live) if pruning is not None else None
    exact = pruning is not None
    class_mars = cc._class_masses()
    prog.transitions(class_mars, cc._class_mars_version)
    stacks, kept = cc._weight_stacks()

    r = prog.root_region                                 # the root: top-down value 0 (log 1) on its slot
    computed, root_once = td.root
    if computed:
        dest, offset, width, cols = (pf["flows"], pf["offset"][r], pf["width"][r], prog.slots[r]) if root_once else \
            (node_flows, lay["sum_offsets"][r], lay["sum_widths"][r], B * prog.slots[r])
        base = offset + (prog.root_rows[0] - prog.firsts[r]) * width
        for row in range(prog.root_rows[1] - prog.root_rows[0]):
            dest[base + row * width:base + row * width + cols] = 0.0
    if td.dead is not None:                              # rows no launch writes, of the regions computed per sample
        fill_dead_rows(node_flows, td.dead, regions, max(lay["sum_widths"][r] for r in td.dead_regions))

    for kind, prod_index, layers in td.steps:
        elem_first = cc.element_regions[prod_index][0]
        elem_width = lay["element_widths"][prod_index]
        if kind == "sum":
            for block_size, dense_runs, fused_groups in layers:
                for part in (0, 1):                      # once, then per sample
                    if part == 0 and not (dense_runs[0] or fused_groups[0]):
                        continue
                    nf, regs, nb = (pf["flows"], once, 1) if part == 0 else (node_flows, regions, B)
                    offsets, widths = (pf["offset"], pf["width"]) if part == 0 else \
                        (lay["sum_offsets"], lay["sum_widths"])
                    for r, c0, E, piece, blocks, key, add in dense_runs[part]:
                        ncols = nb * prog.slots[r]
                        cols = None
                        if part == 1 and pruning is not None and \
                                F.prune_dense(ncols, pruning.num_live[prog.reg_iv_host[r]]):
                            cols = pruning.segment(prog.reg_iv_host[r], ncols)
                        stacked = None
                        width_ = cols[2] if cols is not None else ncols
                        if key is not None and width_ <= F.STACK_MAX_COLUMNS[precision] and \
                                (kept or prog.stack_uses[key] > 1):
                            if key not in stacks:
                                stacks[key] = stacked_weights(pc.params, key[2], key[0], block_size)
                            stacked = stacks[key]
                        dense_sum_top_down(element_flows, nf, pc.params, blocks, E, block_size,
                                          (offsets[r], widths[r], prog.firsts[r]), c0, elem_first, elem_width, ncols,
                                          dot, add, live = cols, stacked = stacked, piece = piece)
                    for cbs, tables, idx in fused_groups[part]:
                        fused_sum_top_down(element_flows, nf, pc.params, tables, regs, elem_first, elem_width, nb,
                                          block_size, cbs, max_cols = nb * tables["max_slots"], precision = dot,
                                          pruning = pruning if part == 1 else None, idx = idx)
            if visit is not None:
                visit(prod_index)
        else:
            for folds, stages, num_temps, temp_slots in layers:
                temp, temp_flows, temp_width = node_mars, node_mars, 0
                if num_temps > 0:
                    temp_width = _align(B * temp_slots)
                    temp = cc._storage_view("product_scratch", num_temps * temp_width)
                    temp_flows = cc._storage_view("product_scratch_flows", num_temps * temp_width)
                if folds[0]:                             # the folds' inside values, for a batch of one ...
                    products(folds[0], (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             prog.no_evidence, None, elem_first, elem_width, 1, srcs, exact = True)
                if folds[1]:                             # ... and per sample
                    products(folds[1], (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             obs_class, live, elem_first, elem_width, B, srcs, exact = exact)
                for once_launches, sample_launches in stages:
                    if once_launches:
                        run_top_down(once_launches, (temp, temp_width), prog, once, element_flows, temp_flows,
                                     pf["flows"], node_mars, class_mars, prog.no_evidence, None, elem_first,
                                     elem_width, 1, srcs, out.skip, True, td.once_iv)
                    if sample_launches:
                        run_top_down(sample_launches, (temp, temp_width), prog, regions, element_flows, temp_flows,
                                     node_flows, node_mars, class_mars, obs_class, live, elem_first, elem_width, B,
                                     srcs, out.skip, exact, td.once_iv, lanes = lanes)
    if pf is not None:
        pf["valid"] |= td.todo
