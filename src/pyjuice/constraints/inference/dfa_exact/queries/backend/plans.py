"""
What a query computes, and where: the dfa_exact backend's partial evaluation.

Every sum region holds, per sample, the INSIDE values of its node group's scope: the log-probability of the evidence
on the scope, for every pair of automaton columns at its ends. Two facts decide which of them a query computes:

* a region's inside values are NEEDED unless its scope covers every target of the query: a conditional of the
  targets reads the inside values of the nodes whose scope misses one of them (and of their descendants, whose
  scopes miss it too), a marginal reads every one;
* they are INPUT-INDEPENDENT when no sample of the batch observes a position of the scope: they are then those of
  the scope with nothing observed, the same for every sample and every query. Such a region may be kept ONCE, in
  ``prior_mars`` (a batch of one, laid out as ``node_mars``), which every sample reads (see
  :func:`~.common._block_ptr`), and computed only when ``prior_mars`` does not hold it yet -- inside
  :func:`pyjuice.fast_inference` it keeps them across queries, as the class masses, else every query starts afresh
  (see :meth:`~.state.State.prior_validity`).

Keeping a region once computes it without evidence pruning -- every column, where the samples' evidence may leave
very few live: :func:`prior_regions` keeps a region once when that is kept across queries, or costs no more than the
samples' live columns would; the other input-independent regions are computed per sample, pruned, like the rest
(a region kept once needs its descendants kept once too: its launch reads them for a batch of one).

A query computes its needed values that are input-dependent per sample, into ``node_mars``, and the
input-independent ones not kept once per sample as well. Of the values kept once it computes those its launches READ
-- the children of the products it computes per sample, of the top-down values it computes, the root's for a
marginal -- that ``prior_mars`` does not hold, and, so that each can be computed, their children kept once that it
does not hold either, down to the ones it holds or the inputs (:attr:`QueryPlan.prior_read`: the values read, held
or computed, which the storage keeps while the query runs, see :class:`~...state.PriorPool`; when they do not all fit
it, the lower ones are computed first, by plans of their own, :func:`kept_plan`). A product follows its
sum regions (they share their scope), and a product's intermediate folds follow the product. The :class:`QueryPlan`
is the subsets of the Program's launches, dense runs and fused parts that do so, cached on the Program by what they
depend on.

A query with a top-down pass (:mod:`.top_down`) plans it the same way, mirrored: a node's top-down value is
NEEDED when its scope meets a target (the scoring reads the top-down values of the target positions' products, each from
its parents'), and INPUT-INDEPENDENT when no sample observes a position OUTSIDE its scope -- kept once then, in
``prior_flows``, when that is kept across queries or costs less than the samples' live columns, and with it every
interval holding it (:func:`top_down_prior`). The intervals are those of every value the pass writes, the folds' of a
product of several children included. A value kept once reads its siblings' inside values for a batch of one, so those
are kept once too. :class:`TopDownPlan` holds the pass's launches as index lists into the Program's tables.

Where a query reads values another query left, it says so: ``held`` -- the per-sample values the buffers hold for it,
by sum node (region) -- is computed by none of its launches (a caching engine's to decide; nothing by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .planners import IndexLists, _forward_steps, _mode_workspace, _top_down_modes, _top_down_steps, _top_down_tables, \
    class_weight_writes


#: query plans a Program keeps (the most recently used): at least this many, and at least ``PLANS_PER_POSITION`` per
#: position -- a decoding step is a plan of its own (a forward one and a top-down one), so decoding every position
#: keeps all of its plans
PLAN_CACHE_SIZE = 64
PLANS_PER_POSITION = 4


@dataclass
class QueryPlan:
    """
    :ivar steps: the Program's ``steps``, every part split in two, ``(once, sample)``: its values kept once, computed
        for a batch of one into ``prior_mars``, and its per-sample ones -- a product layer ``(once stages, sample
        stages, num_temps, max_temp_slots)``, its stages lists of ``(launch, cut)`` (:class:`~.prod.Cut`), a
        sum layer ``(block_size, once parts, sample parts)``, each part ``(dense runs [(region, c0, blocks, key)],
        num_edges, None or (fused part, index list of its node blocks or None for all, count, most slots, its
        regions' intervals))``
    :ivar todo: bool [sum intervals] (host tensor): the values kept once the query computes
    :ivar prior_read: bool [sum intervals] (host): the values kept once the query reads (``todo`` among them)
    :ivar prior: whether the query reads or writes ``prior_mars`` at all
    :ivar root_prior: whether the root's value is kept once
    :ivar reg_prior: int32 [sum regions] (device): 1 where a region's values are in ``prior_mars`` for this query
    :ivar sample: bool [sum intervals] (host): the inside values the query computes per sample
    :ivar top_down: None, or the query's :class:`TopDownPlan`
    """

    steps: list
    todo: torch.Tensor
    prior_read: np.ndarray
    prior: bool
    root_prior: bool
    reg_prior: torch.Tensor
    sample: np.ndarray
    top_down: Optional["TopDownPlan"] = None


@dataclass
class TopDownPlan:
    """
    The top-down pass of a query (:func:`top_down_plan`), from the :class:`~.top_down_program.TopDownProgram`'s tables:
    every launch the query runs, as ``(launch or tables, index list)`` -- the int32 device list of the children (child
    blocks) it computes, or None for all; for the walk kernels' launches (``input_block`` / ``block_input`` and class
    weights) the children (destinations) in node tiles (:func:`~.common.node_tiles`) -- split in two, ``(once,
    sample)``: the values kept once, for a batch of one in ``prior_flows``, and the per-sample ones.

    :ivar steps: from the top down, ``("sum", prod_index, [(block_size, (once, sample) dense runs, (once, sample)
        fused [(child block size, tables, index list)])])`` and ``("prod", prod_index, [((once, sample) fold stages,
        stages [(once launches, sample launches, class weight launches)] in launch order, num_temps, temp_slots)])``
        (class weight launches: a scoring plan's, ``(launch, index list, store)``, see ``cw_fill``; else none)
    :ivar root: ``(computed, once)`` of the root's value
    :ivar once: bool [sum regions] (host): those whose values the query keeps once (reads or writes in ``prior_flows``)
    :ivar todo: bool [sum regions] (host): those it writes in ``prior_flows``
    :ivar sample: bool [sum regions] (host): those it writes in ``node_flows``
    :ivar per_sample: bool [sum regions] (host): those whose values it needs per sample, written or held in
        ``node_flows`` -- the regions whose lanes its sum launches walk
    :ivar once_iv: int32 [intervals] (device, by ``prog.intervals``' numbers): 1 where the values are kept once
    :ivar prior: bool [sum intervals] (host): the inside values its launches for a batch of one read
    :ivar dead: None, or int32 [D, 2] (device): the rows of the regions it computes per sample that no launch writes
        (see :class:`~.top_down_program.TopDownProgram`), and ``dead_regions``, their regions
    :ivar cw_fill: a scoring plan's: whether ``dfa_class_flow`` must hold -inf before its class weight launches run
        (:func:`~.planners.class_weight_writes`; their entries ``(launch, index list, store)``)
    """

    steps: list
    root: tuple
    once: np.ndarray
    todo: np.ndarray
    sample: np.ndarray
    per_sample: np.ndarray
    once_iv: torch.Tensor
    prior: np.ndarray
    dead: Optional[torch.Tensor]
    dead_regions: list
    cw_fill: bool = True


def _prefix(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.size + 1, dtype = np.int64)
    np.cumsum(mask, out = out[1:])
    return out


def free_regions(prog, observed: np.ndarray) -> np.ndarray:
    """bool [sum intervals]: the input-independent ones -- no sample observes a position of the interval."""
    P = _prefix(observed)
    return P[prog.iv_bounds[:, 1]] == P[prog.iv_bounds[:, 0]]


def prior_regions(prog, observed: np.ndarray, B: int, kept: bool, liveness) -> np.ndarray:
    """
    bool [sum intervals]: the intervals a query of ``B`` samples keeps once (see the module docstring): every
    input-independent one (:func:`free_regions`) when ``kept`` (``prior_mars`` outlives the query: inside
    :func:`pyjuice.fast_inference`); else those whose block computed once is cheaper than the samples' live slots
    (``liveness``, the query's :class:`~...kernels.live.Liveness`, whose counts this reads -- a wait for the device;
    None: every slot live), and every interval inside one of those. One sample's live slots never exceed a
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
    live = np.asarray(liveness.num_live, dtype = np.int64) if liveness is not None else B * slots
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
        as a dict: ``program``, ``once``, ``valid``, ``held`` and optionally ``scoring``
    """
    td, td_key = None, None
    tbytes = None if targets is None else targets.tobytes()
    if top_down is not None:
        h, scoring = top_down["held"], top_down.get("scoring", False)
        td_key = ("top_down", tbytes, top_down["once"].tobytes(), top_down["valid"].tobytes(),
                  None if h is None else h.tobytes(), scoring)
        td = prog.plans.get(td_key)
        if td is None:
            td = top_down_plan(prog, top_down["program"], targets, top_down["once"], top_down["valid"], h, scoring)
            _keep(prog, td_key, td)
        else:
            prog.plans.move_to_end(td_key)
    key = (observed.tobytes(), tbytes, valid.tobytes(), prior.tobytes(), None if held is None else held.tobytes(),
           td_key)                                       # what the plan is a function of
    plan = prog.plans.get(key)
    if plan is not None:
        prog.plans.move_to_end(key)
        return plan

    a, b = prog.iv_bounds[:, 0], prog.iv_bounds[:, 1]
    if targets is None:
        needed = np.ones_like(prior)
    else:
        Q = _prefix(targets)
        needed = (Q[a] > 0) | (Q[-1] > Q[b])                         # a target outside the scope
    if td is not None and td.prior.any():        # read for a batch of one: kept once, and every interval inside
        free = free_regions(prog, observed)
        assert not (td.prior & ~free).any(), "a value kept once reads an input-dependent inside value"
        prior = prior | (free & prog.iv_inside[td.prior].any(axis = 0))
    sample = needed & ~prior
    if held is not None:
        sample &= ~held_intervals(prog, held)
    # the values kept once the launches read: the children of what is computed per sample or top down, the root's
    # for a marginal -- and, to compute the ones not held, their children not held, down to held ones
    read_ivs = sample.copy()
    if td is not None:
        read_ivs[prog.reg_iv_np[td.per_sample | td.once]] = True
    wanted = np.zeros_like(prior)
    wanted[prog.reg_iv_np[prog.operands[read_ivs].any(axis = 0)]] = True
    if td is not None:
        wanted |= td.prior
    if targets is None:
        wanted[prog.reg_iv_host[prog.root_region]] = True
    wanted &= prior
    todo, frontier, read = np.zeros_like(prior), wanted & ~valid, wanted.copy()
    while frontier.any():
        todo |= frontier
        below = np.zeros_like(prior)
        for k in np.nonzero(frontier)[0]:
            below[prog.iv_children[k]] = True
        read |= below & prior                            # (their children kept once: held, or computed too)
        frontier = below & prior & ~valid & ~todo
    prior_read = read | todo

    lists = IndexLists()                                 # every index list, copied to the device together
    steps = _forward_steps(prog, todo, sample, lists)
    reg_prior = lists.add(prior[prog.reg_iv_np].astype(np.int32))
    lists.upload(prog.dev)
    plan = QueryPlan(steps = steps, todo = torch.from_numpy(todo), prior_read = prior_read,
                     prior = bool(prior_read.any()), root_prior = bool(prior[prog.reg_iv_host[prog.root_region]]),
                     reg_prior = reg_prior.tensor, sample = sample, top_down = td)
    _keep(prog, key, plan)
    return plan


def kept_plan(prog, todo: np.ndarray, prior: np.ndarray, keep: np.ndarray) -> QueryPlan:
    """
    A plan computing the values kept once of the intervals ``todo`` (bool [sum intervals], host) and nothing per
    sample: a piece of a query's values kept once, computed before the query when they do not all fit their pool
    (:func:`~.forward.kept_values`). It reads them and their children kept once (``prior``, bool [sum intervals]: the
    intervals kept once), which must be held or among ``todo``; its ``prior_read`` has ``keep`` too (bool [sum
    intervals]: the held values the query reads, kept through its pieces).
    """
    lists = IndexLists()
    sample = np.zeros_like(todo)
    steps = _forward_steps(prog, todo, sample, lists)
    reg_prior = lists.add(prior[prog.reg_iv_np].astype(np.int32))
    lists.upload(prog.dev)
    read = todo.copy()
    for k in np.nonzero(todo)[0]:
        read[prog.iv_children[k]] = True
    return QueryPlan(steps = steps, todo = torch.from_numpy(todo), prior_read = (read & prior) | keep, prior = True,
                     root_prior = False, reg_prior = reg_prior.tensor, sample = sample)


def read_regions(prog, plan: QueryPlan, root: bool = False) -> np.ndarray:
    """bool [sum regions]: those whose per-sample inside values the query's launches may read -- the children of the
    products of the intervals it computes per sample (:attr:`QueryPlan.sample`), and of those whose top-down values it
    needs, per sample or kept once (:attr:`TopDownPlan.per_sample`, ``once``: a product's top-down step reads its other
    children's inside values, and so do the class weights of its input at a target, per sample even under a value
    kept once) -- and the ``root``'s if asked. Held values of other regions may stay where they are (see
    :mod:`...kernels.compact`)."""
    ivs = plan.sample.copy()
    if plan.top_down is not None:
        ivs[prog.reg_iv_np[plan.top_down.per_sample | plan.top_down.once]] = True
    out = prog.operands[ivs].any(axis = 0)
    if root:
        out[prog.root_region] = True
    return out


def per_sample_intervals(prog, plan: QueryPlan, top_down: bool = True) -> np.ndarray:
    """bool [sum intervals]: those whose columns the query's kernels walk in lane order
    (:meth:`~...kernels.live.Liveness.order`): the inside values ``plan`` computes per sample
    (:attr:`QueryPlan.sample`), and the top-down values it needs per sample (:attr:`TopDownPlan.per_sample` -- held ones
    included: a sum launch gives its products their values from them), unless ``top_down`` is False (they are compact,
    see :mod:`...kernels.compact`, and walked in their own order)."""
    out = plan.sample.copy()
    if plan.top_down is not None and top_down:
        out[prog.reg_iv_np[plan.top_down.per_sample]] = True
    return out


def _keep(prog, key, plan):
    prog.plans[key] = plan
    if len(prog.plans) > max(PLAN_CACHE_SIZE, PLANS_PER_POSITION * prog.n):
        prog.plans.popitem(last = False)


def held_intervals(prog, held: np.ndarray) -> np.ndarray:
    """bool [sum intervals]: those whose every sum region ``held`` (bool [sum regions]) marks."""
    out = np.ones(prog.iv_bounds.shape[0], dtype = bool)
    np.logical_and.at(out, prog.reg_iv_np, held)
    return out


def top_down_free(td, observed: np.ndarray) -> np.ndarray:
    """bool [intervals] (by ``prog.intervals``' numbers): the input-independent top-down values -- no sample observes
    a position outside the interval."""
    P = _prefix(observed)
    return (P[td.iv_bounds[:, 0]] == 0) & (P[-1] == P[td.iv_bounds[:, 1]])


def scope_intervals(td, scopes) -> np.ndarray:
    """bool [intervals] (by ``prog.intervals``' numbers, ``td`` the :class:`~.top_down_program.TopDownProgram`): those
    equal to one of ``scopes`` (``(a, b)`` pairs) -- under per-sample inputs of a sum node group (gates, see
    :mod:`...external_params`), its scope: the top-down values there (its children's) read them, so
    they are input-dependent whatever is observed."""
    if not scopes:
        return np.zeros(td.iv_bounds.shape[0], dtype = bool)
    se = np.asarray(scopes, dtype = np.int64).reshape(-1, 2)
    big = int(max(td.iv_bounds.max(), se.max())) + 1
    return np.isin(td.iv_bounds[:, 0] * big + td.iv_bounds[:, 1], se[:, 0] * big + se[:, 1])


def top_down_prior(prog, td, observed: np.ndarray, B: int, kept: bool, liveness) -> np.ndarray:
    """
    bool [intervals] (by ``prog.intervals``' numbers): the top-down values a query of ``B`` samples keeps once, as
    :func:`prior_regions` decides for inside values -- every input-independent one (:func:`top_down_free`) when
    ``kept``; else those of the sum intervals whose block computed once is cheaper than the samples' live slots, and
    every interval holding one of them (a value kept once reads its parents' for a batch of one); none for a query of
    one sample.

    :param td: the :class:`~.top_down_program.TopDownProgram`
    """
    I = td.iv_bounds.shape[0]
    if not kept and B == 1:
        return np.zeros(I, dtype = bool)
    free = top_down_free(td, observed)
    if kept:
        return free
    slots = prog.iv_slots
    live = np.asarray(liveness.num_live, dtype = np.int64) if liveness is not None else B * slots
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
                  held: Optional[np.ndarray], scoring: bool = False) -> TopDownPlan:
    """
    The top-down pass of a query (see the module docstring and :class:`TopDownPlan`). A sum node's value is computed
    when it is needed and the buffer of its kind (``prior_flows`` once, ``node_flows`` per sample) does not hold it; a
    product's (a fold's) when a child computed reads it, or a target position is one of its inputs. From the bottom
    up, which values are computed (:func:`_top_down_modes`, over the program's flat tables); then the launches
    computing them, split by where they go. A ``scoring`` plan also runs the class weights of the target positions'
    input nodes (:mod:`~.score`), every sample's: their products' values are computed (see above), and a fold
    they read as an operand is computed again per sample.

    :param td: the :class:`~.top_down_program.TopDownProgram`
    :param targets: None (every value), or bool [n] (host): the positions some sample targets
    :param once: bool [intervals]: the values kept once (:func:`top_down_prior`)
    :param valid: bool [sum regions]: those ``prior_flows`` holds
    :param held: None, or bool [sum regions]: those ``node_flows`` holds for this query
    :param scoring: whether the plan runs the class weights (``targets`` given)
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
    t, index = _top_down_tables(prog, td)
    m = _top_down_modes(req, positions, once, scoring, t.data, t.offsets, *_mode_workspace(t, td))
    lists = IndexLists()                                 # every index list, copied to the device together
    out = _top_down_steps(td, t, index, m, scoring, lists)
    cw_fill = class_weight_writes(prog, out, positions) if scoring else True
    prior = m[-1]
    root = prog.root_region
    sample = req & ~r_once
    dead = td.dead[sample[td.dead[:, 1]]] if td.dead.size else td.dead
    reads = np.zeros(prog.iv_bounds.shape[0], dtype = bool)                    # by sum interval
    reads[prog.reg_iv_np[prior]] = True
    once_iv, dead_rows = lists.add(once.astype(np.int32)), lists.add(dead.ravel()) if dead.size else None
    lists.upload(dev)
    return TopDownPlan(steps = out, root = (bool(req[root]), bool(r_once[root])),
                       once = r_once & needed[td.reg_iv], todo = req & r_once, sample = sample,
                       per_sample = needed[td.reg_iv] & ~r_once,
                       once_iv = once_iv.tensor, prior = reads,
                       dead = None if dead_rows is None else dead_rows.tensor.view(-1, 2),
                       dead_regions = np.nonzero(sample & td.dead_region)[0].tolist(), cw_fill = cw_fill)
