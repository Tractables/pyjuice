"""
What a query computes, and where: the lifted backend's partial evaluation.

Every sum region holds, per sample, the INSIDE values of its node group's scope: the log-probability of the evidence
on the scope, for every pair of automaton columns at its ends. Two facts decide which of them a query computes:

* a region's inside values are NEEDED unless its scope covers every target of the query: a conditional of the
  targets reads the inside values of the nodes whose scope misses one of them (and of their descendants, whose
  scopes miss it too), a marginal reads every one;
* they are INPUT-INDEPENDENT when no sample of the batch observes a position of the scope: they are then those of
  the scope with nothing observed, the same for every sample and every query. Such a region may be kept ONCE, in
  ``prior_mars`` (a batch of one, laid out as ``node_mars``), which every sample reads (see
  :func:`~.kernels.prod._block_ptr`), and computed only when ``prior_mars`` does not hold it yet -- inside
  :func:`pyjuice.fast_inference` it keeps them across queries, as the class masses, else every query starts afresh
  (see :meth:`~pyjuice.constraints.ConstrainedCircuit._prior_validity`).

Keeping a region once computes it without evidence pruning -- every column, where the samples' evidence may leave
very few live: :func:`prior_regions` keeps a region once when that is kept across queries, or costs no more than the
samples' live columns would; the other input-independent regions are computed per sample, pruned, like the rest
(a region kept once needs its descendants kept once too: its launch reads them for a batch of one).

A query computes its needed values that are input-dependent per sample, into ``node_mars``, those kept once that
``prior_mars`` does not hold yet, once, and the input-independent ones not kept once per sample as well. A product
follows its sum regions (they share their scope), and a product's intermediate folds follow the product. The
:class:`QueryPlan` is the subsets of the Program's launches, dense runs and fused parts that do so, cached on the
Program by what they depend on.

A query with a top-down pass (:mod:`.backward`) plans it the same way, mirrored: a node's top-down value is NEEDED when
its scope meets a target (the scoring reads the top-down values of the target positions' products, each from its
parents'), and INPUT-INDEPENDENT when no sample observes a position OUTSIDE its scope -- kept once then, in
``prior_flows``, when that is kept across queries or costs less than the samples' live columns, and with it every
interval holding it (:func:`top_down_prior`). The intervals are those of every value the pass writes, the folds' of a
product of several children included. A value kept once reads its siblings' inside values for a batch of one, so
those are kept once too. :class:`TopDownPlan` holds the pass's launches as index lists into the Program's tables.

Where a query reads values another query left, it says so: ``held`` -- the per-sample values the buffers hold for it,
by sum node (region) -- is computed by none of its launches (a caching engine's to decide; nothing by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .kernels.prod import SUM, TEMP

#: query plans a Program keeps (the most recently used)
PLAN_CACHE_SIZE = 64


@dataclass
class QueryPlan:
    """
    :ivar steps: the Program's ``steps``, every part split in two, ``(once, sample)``: its values kept once, computed
        for a batch of one into ``prior_mars``, and its per-sample ones -- a product layer ``(once stages, sample
        stages, num_temps, max_temp_slots)``, a sum layer ``(block_size, once parts, sample parts)``, each part
        ``(dense runs, num_edges, fused part or None)``
    :ivar todo: bool [sum intervals] (host tensor): the values kept once the query computes
    :ivar prior: whether the query reads or writes ``prior_mars`` at all
    :ivar root_prior: whether the root's value is kept once
    :ivar reg_prior: int32 [sum regions] (device): 1 where a region's values are in ``prior_mars`` for this query
    :ivar sample: bool [sum intervals] (host): the inside values the query computes per sample
    :ivar top_down: None, or the query's :class:`TopDownPlan`
    """

    steps: list
    todo: torch.Tensor
    prior: bool
    root_prior: bool
    reg_prior: torch.Tensor
    sample: np.ndarray
    top_down: Optional["TopDownPlan"] = None


@dataclass
class TopDownPlan:
    """
    The top-down pass of a query (:func:`top_down_plan`), from the :class:`~.backward.TopDownProgram`'s tables: every
    launch the query runs, as ``(launch or tables, index list)`` -- the int32 device list of the children (child
    blocks) it computes, or None for all -- split in two, ``(once, sample)``: the values kept once, for a batch of one
    in ``prior_flows``, and the per-sample ones.

    :ivar steps: from the top down, ``("sum", prod_index, [(block_size, (once, sample) dense runs, (once, sample)
        fused [(child block size, tables, index list)])])`` and ``("prod", prod_index, [((once, sample) fold stages,
        stages [(once launches, sample launches)] in launch order, num_temps, temp_slots)])``
    :ivar root: ``(computed, once)`` of the root's value
    :ivar once: bool [sum regions] (host): those whose values the query keeps once (reads or writes in ``prior_flows``)
    :ivar todo: bool [sum regions] (host): those it writes in ``prior_flows``
    :ivar sample: bool [sum regions] (host): those it writes in ``node_flows``
    :ivar once_iv: int32 [intervals] (device, by ``prog.intervals``' numbers): 1 where the values are kept once
    :ivar prior: bool [sum intervals] (host): the inside values its launches for a batch of one read
    :ivar dead: None, or int32 [D, 2] (device): the rows of the regions it computes per sample that no launch writes
        (see :class:`~.backward.TopDownProgram`), and ``dead_regions``, their regions
    """

    steps: list
    root: tuple
    once: np.ndarray
    todo: np.ndarray
    sample: np.ndarray
    once_iv: torch.Tensor
    prior: np.ndarray
    dead: Optional[torch.Tensor]
    dead_regions: list


def _prefix(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.size + 1, dtype = np.int64)
    np.cumsum(mask, out = out[1:])
    return out


def free_regions(prog, observed: np.ndarray) -> np.ndarray:
    """bool [sum intervals]: the input-independent ones -- no sample observes a position of the interval."""
    P = _prefix(observed)
    return P[prog.iv_bounds[:, 1]] == P[prog.iv_bounds[:, 0]]


def prior_regions(prog, observed: np.ndarray, B: int, kept: bool, pruning) -> np.ndarray:
    """
    bool [sum intervals]: the intervals a query of ``B`` samples keeps once (see the module docstring): every
    input-independent one (:func:`free_regions`) when ``kept`` (``prior_mars`` outlives the query: inside
    :func:`pyjuice.fast_inference`); else those whose block computed once is cheaper than the samples' live slots
    (``pruning``, the query's :class:`~.kernels.live.Pruning`, whose counts this reads -- the query's one wait for the
    device; None: every slot live), and every interval inside one of those. One sample's live slots never exceed a
    block's, so a query of one sample keeps nothing once (and reads no counts).

    :param observed: bool [n] (host): the positions some sample observes
    """
    K = prog.iv_bounds.shape[0]
    if not kept and B == 1:
        return np.zeros(K, dtype = bool)
    free = free_regions(prog, observed)
    if kept:
        return free
    slots = prog.iv_slots
    live = np.asarray(pruning.num_live, dtype = np.int64) if pruning is not None else B * slots
    worth = free & (live > slots)
    if not worth.any():
        return worth
    return free & prog.iv_inside[worth].any(axis = 0)                    # ... and every interval inside one of them


def query_plan(prog, observed: np.ndarray, targets: Optional[np.ndarray], valid: np.ndarray, prior: np.ndarray,
               held: Optional[np.ndarray] = None, top_down: Optional[dict] = None) -> QueryPlan:
    """
    The plan of a query (see the module docstring), from the Program's cache when it holds it.

    :param observed: bool [n] (host): the positions some sample observes
    :param targets: None for a marginal (every inside value is needed), or bool [n] (host): the positions some
        sample targets
    :param valid: bool [sum intervals] (host): the values ``prior_mars`` holds
    :param prior: bool [sum intervals] (host): the intervals kept once (:func:`prior_regions`)
    :param held: None, or bool [sum regions] (host): the per-sample inside values ``node_mars`` holds for this query
    :param top_down: None (no top-down pass), or the arguments of :func:`top_down_plan` but ``prog`` and ``targets``,
        as a dict: ``program``, ``once``, ``valid`` and ``held``
    """
    a, b = prog.iv_bounds[:, 0], prog.iv_bounds[:, 1]
    if targets is None:
        needed = np.ones_like(prior)
    else:
        Q = _prefix(targets)
        needed = (Q[a] > 0) | (Q[-1] > Q[b])                         # a target outside the scope
    td, td_key = None, None
    if top_down is not None:
        h = top_down["held"]
        td_key = ("top_down", None if targets is None else targets.tobytes(), top_down["once"].tobytes(),
                  top_down["valid"].tobytes(), None if h is None else h.tobytes())
        td = prog.plans.get(td_key)
        if td is None:
            td = top_down_plan(prog, top_down["program"], targets, top_down["once"], top_down["valid"], h)
            _keep(prog, td_key, td)
        else:
            prog.plans.move_to_end(td_key)
        if td.prior.any():                       # read for a batch of one: kept once, and every interval inside
            free = free_regions(prog, observed)
            assert not (td.prior & ~free).any(), "a value kept once reads an input-dependent inside value"
            prior = prior | (free & prog.iv_inside[td.prior].any(axis = 0))
    todo = needed & prior & ~valid
    sample = needed & ~prior
    if held is not None:
        sample &= ~held_intervals(prog, held)
    key = (observed.tobytes(), None if targets is None else targets.tobytes(), todo.tobytes(), sample.tobytes(),
           prior.tobytes(), td_key)
    plan = prog.plans.get(key)
    if plan is not None:
        prog.plans.move_to_end(key)
        return plan

    todo_t, sample_t = torch.from_numpy(todo), torch.from_numpy(sample)
    steps = []
    for kind, prod_index, layers in prog.steps:
        if kind == "prod":
            steps.append((kind, prod_index, [(_cut_stages(prog, stages, todo_t), _cut_stages(prog, stages, sample_t),
                                              num_temps, temp_slots) for stages, num_temps, temp_slots in layers]))
        else:
            steps.append((kind, prod_index, [(block_size, _cut_parts(prog, parts, todo_t),
                                              _cut_parts(prog, parts, sample_t)) for block_size, parts in layers]))
    plan = QueryPlan(steps = steps, todo = todo_t, prior = bool((needed & prior).any()),
                     root_prior = bool(prior[prog.reg_iv_host[prog.root_region]]),
                     reg_prior = torch.from_numpy(prior[prog.reg_iv_host].astype(np.int32)).to(prog.dev),
                     sample = sample, top_down = td)
    _keep(prog, key, plan)
    return plan


def _keep(prog, key, plan):
    prog.plans[key] = plan
    if len(prog.plans) > PLAN_CACHE_SIZE:
        prog.plans.popitem(last = False)


def held_intervals(prog, held: np.ndarray) -> np.ndarray:
    """bool [sum intervals]: those whose every sum region ``held`` (bool [sum regions]) marks."""
    out = np.ones(prog.iv_bounds.shape[0], dtype = bool)
    np.logical_and.at(out, np.asarray(prog.reg_iv_host), held)
    return out


def _cut_stages(prog, stages, keep):
    """The launches of a product layer's ``stages`` cut to the steps whose product's interval ``keep`` holds (a
    launch kept whole is reused); stages and launches left empty dropped."""
    out = []
    for launches in stages:
        cut = []
        for launch in launches:
            mask = keep[launch.iv]
            if bool(mask.all()):
                cut.append(launch)
            elif bool(mask.any()):
                cut.append(prog._launch(launch.form, launch.kinds, launch.rows[mask], launch.iv[mask],
                                        launch.sizes[mask]))
        if cut:
            out.append(cut)
    return out


def _cut_parts(prog, parts, keep):
    """A sum layer's partitions cut to the node blocks whose region's interval ``keep`` holds."""
    out = []
    iv = lambda r: prog.reg_iv_host[r]
    for dense, num_edges, fused in parts:
        dense = {rc: v for rc, v in dense.items() if keep[iv(rc[0])]}
        if fused is not None:
            nids, cids, pids, nb_reg, max_slots, aliased = fused
            regs = nb_reg.tolist()
            mask = [bool(keep[iv(r)]) for r in regs]
            if not any(mask):
                fused = None
            elif not all(mask):
                idx = torch.tensor([k for k, m in enumerate(mask) if m], dtype = torch.long, device = nids.device)
                if aliased is not None:
                    aliased = tuple(t[idx].contiguous() for t in aliased)
                fused = (nids[idx].contiguous(), cids[idx].contiguous(), pids[idx].contiguous(), nb_reg[idx].contiguous(),
                         max(prog.slots[r] for r, m in zip(regs, mask) if m), aliased)
        if dense or fused is not None:
            out.append((dense, num_edges, fused))
    return out


def top_down_free(td, observed: np.ndarray) -> np.ndarray:
    """bool [intervals] (by ``prog.intervals``' numbers): the input-independent top-down values -- no sample observes
    a position outside the interval."""
    P = _prefix(observed)
    return (P[td.iv_bounds[:, 0]] == 0) & (P[-1] == P[td.iv_bounds[:, 1]])


def top_down_prior(prog, td, observed: np.ndarray, B: int, kept: bool, pruning) -> np.ndarray:
    """
    bool [intervals] (by ``prog.intervals``' numbers): the top-down values a query of ``B`` samples keeps once, as
    :func:`prior_regions` decides for inside values -- every input-independent one (:func:`top_down_free`) when
    ``kept``; else those of the sum intervals whose block computed once is cheaper than the samples' live slots, and
    every interval holding one of them (a value kept once reads its parents' for a batch of one); none for a query of
    one sample.

    :param td: the :class:`~.backward.TopDownProgram`
    """
    I = td.iv_bounds.shape[0]
    if not kept and B == 1:
        return np.zeros(I, dtype = bool)
    free = top_down_free(td, observed)
    if kept:
        return free
    slots = prog.iv_slots
    live = np.asarray(pruning.num_live, dtype = np.int64) if pruning is not None else B * slots
    sums = td.reg_iv[np.unique(np.asarray(prog.reg_iv_host), return_index = True)[1]]    # sum interval -> number
    worth = sums[free[sums] & (live > slots)]
    if worth.size == 0:
        return np.zeros(I, dtype = bool)
    wa, wb = td.iv_bounds[worth, 0], td.iv_bounds[worth, 1]
    order = np.argsort(wa, kind = "stable")
    wa, wb = wa[order], wb[order]
    low = np.minimum.accumulate(wb[::-1])[::-1]               # the least end among those starting at wa[k] or later
    k = np.searchsorted(wa, td.iv_bounds[:, 0], side = "left")
    holds = np.zeros(I, dtype = bool)
    inside = k < wa.size
    holds[inside] = low[k[inside]] <= td.iv_bounds[inside, 1]
    return free & holds


def top_down_plan(prog, td, targets: Optional[np.ndarray], once: np.ndarray, valid: np.ndarray,
                  held: Optional[np.ndarray]) -> TopDownPlan:
    """
    The top-down pass of a query (see the module docstring and :class:`TopDownPlan`). A sum node's value is computed
    when it is needed and the buffer of its kind (``prior_flows`` once, ``node_flows`` per sample) does not hold it; a
    product's (a fold's) when a child computed reads it, or a target position is one of its inputs. From the bottom
    up, which values are computed; then the launches computing them, split by where they go.

    :param td: the :class:`~.backward.TopDownProgram`
    :param targets: None (every value), or bool [n] (host): the positions some sample targets
    :param once: bool [intervals]: the values kept once (:func:`top_down_prior`)
    :param valid: bool [sum regions]: those ``prior_flows`` holds
    :param held: None, or bool [sum regions]: those ``node_flows`` holds for this query
    """
    n, dev = prog.n, prog.dev
    a, b = td.iv_bounds[:, 0], td.iv_bounds[:, 1]
    if targets is None:
        needed, positions = np.ones(a.size, dtype = bool), np.ones(n, dtype = bool)
    else:
        Q = _prefix(targets)
        needed, positions = Q[b] > Q[a], targets
    r_once = once[td.reg_iv]
    req = needed[td.reg_iv] & ~np.where(r_once, valid, False if held is None else held)   # sum regions computed
    prior = np.zeros(len(td.reg_iv), dtype = bool)        # sum regions read for a batch of one

    def idx(sel):
        return None if sel.all() else torch.from_numpy(np.nonzero(sel)[0].astype(np.int32)).to(dev)

    def mark(rows, out_temp, first, treq, ereq):
        if out_temp:
            treq[rows] = True
        else:
            ereq[rows - first] = True

    elem_req, out = {}, [None] * len(td.steps)
    for si in range(len(td.steps) - 1, -1, -1):          # from the bottom up: what each computed value reads
        kind, pi, layers = td.steps[si]
        first, end = td.elem_regions[pi]
        if kind == "prod":
            ereq = np.zeros(end - first, dtype = bool)
            layers_out = []
            for fold_stages, stages, num_temps, temp_slots in layers:
                treq = np.zeros(max(num_temps, 1), dtype = bool)
                tin = np.full(max(num_temps, 1), -1)      # the folds' inside values: needed per sample (0) or once (1)
                masks = []
                for launches, inputs, _ in stages:        # the forward's order: a fold before the steps reading it
                    ms = []
                    for la in launches:
                        h = la.host
                        rq = req[h["children"][:, 1]] if la.kinds[0] == SUM else treq[h["children"][:, 0]]
                        if rq.any():
                            mark(h["parents"][np.repeat(rq, np.diff(h["ptr"])), 0], _out_temp(la), first, treq,
                                 ereq)
                        ms.append(rq)
                    for out_temp, rows, pos in inputs:
                        mark(rows[positions[pos]], out_temp, first, treq, ereq)
                    masks.append(ms)
                stages_out = []
                for (launches, _, folds), ms in zip(stages[::-1], masks[::-1]):    # the order they run in
                    once_l, sample_l = [], []
                    for la, rq in zip(launches, ms):
                        if not rq.any():
                            continue
                        h = la.host
                        co = once[h["child_iv"]]
                        for mode, sel, lst in ((1, rq & co, once_l), (0, rq & ~co, sample_l)):
                            if not sel.any():
                                continue
                            lst.append((la, idx(sel)))
                            if la.form == "block_block":  # the sibling's inside value, in the child's mode
                                par = h["parents"][np.repeat(sel, np.diff(h["ptr"]))]
                                if la.side == "R":
                                    prior[par[:, 2]] |= mode == 1
                                elif la.kinds[1] == TEMP:
                                    tin[par[:, 3]] = mode
                                else:
                                    prior[par[:, 4]] |= mode == 1
                    for out_rows, temp, sums in folds:    # a fold computed again reads its operands in its mode
                        mode = tin[out_rows]
                        if temp is not None:
                            tin[temp[mode >= 0]] = mode[mode >= 0]
                        for regs in sums:
                            prior[regs[mode == 1]] = True
                    stages_out.append((once_l, sample_l))
                folds_out = ([], [])
                for launches in fold_stages:
                    cut = ([], [])
                    for la in launches:
                        mode = tin[la.rows[:, 0].numpy()]
                        for m, lst in ((1, cut[0]), (0, cut[1])):
                            sel = mode == m
                            if sel.all():
                                lst.append(la)
                            elif sel.any():
                                t = torch.from_numpy(sel)
                                lst.append(prog._launch(la.form, la.kinds, la.rows[t], la.iv[t], la.sizes[t]))
                    for m in (0, 1):
                        if cut[m]:
                            folds_out[m].append(cut[m])
                layers_out.append((folds_out, stages_out, num_temps, temp_slots))
            elem_req[pi] = ereq
            out[si] = ("prod", pi, layers_out)
        else:
            ereq = elem_req.get(pi, np.zeros(end - first, dtype = bool))
            E = _prefix(ereq)
            layers_out = []
            for block_size, dense_runs, fused_groups in layers:
                dense = ([], [])
                for run in dense_runs:
                    r, c0, _, (lo, hi), _, _, _ = run
                    if E[c0 + hi - first] > E[c0 + lo - first]:
                        dense[0 if once[td.reg_iv[r]] else 1].append(run)
                fused = ([], [])
                for cbs, tables in fused_groups.items():
                    h = tables["host"]
                    elem = h["cb_reg"] < 0
                    lo = np.clip(h["cb_row"] - first, 0, ereq.size)
                    rq = np.where(elem, E[np.minimum(lo + cbs, ereq.size)] > E[lo], req[np.maximum(h["cb_reg"], 0)])
                    co = once[td.reg_iv[h["cb_ivreg"]]]
                    for sel, lst in ((rq & co, fused[0]), (rq & ~co, fused[1])):
                        if sel.any():
                            lst.append((cbs, tables, idx(sel)))
                layers_out.append((block_size, dense, fused))
            out[si] = ("sum", pi, layers_out)
    root = prog.root_region
    sample = req & ~r_once
    dead = td.dead[sample[td.dead[:, 1]]]
    reads = np.zeros(prog.iv_bounds.shape[0], dtype = bool)                    # by sum interval
    reads[np.asarray(prog.reg_iv_host)[prior]] = True
    return TopDownPlan(steps = out, root = (bool(req[root]), bool(r_once[root])),
                       once = r_once & needed[td.reg_iv], todo = req & r_once, sample = sample,
                       once_iv = torch.from_numpy(once.astype(np.int32)).to(dev), prior = reads,
                       dead = torch.from_numpy(dead.astype(np.int32)).to(dev) if dead.size else None,
                       dead_regions = sorted(set(dead[:, 1].tolist())))


def _out_temp(la) -> bool:
    """Whether a :class:`~.kernels.prod_top_down.TopDownLaunch`'s parents are scratch rows."""
    return bool(la.kinds[2] if la.form == "block_block" else la.kinds[1])
