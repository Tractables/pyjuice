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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

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
    """

    steps: list
    todo: torch.Tensor
    prior: bool
    root_prior: bool
    reg_prior: torch.Tensor


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


def query_plan(prog, observed: np.ndarray, targets: Optional[np.ndarray], valid: np.ndarray,
               prior: np.ndarray) -> QueryPlan:
    """
    The plan of a query (see the module docstring), from the Program's cache when it holds it.

    :param observed: bool [n] (host): the positions some sample observes
    :param targets: None for a marginal (every inside value is needed), or bool [n] (host): the positions some
        sample targets
    :param valid: bool [sum intervals] (host): the values ``prior_mars`` holds
    :param prior: bool [sum intervals] (host): the intervals kept once (:func:`prior_regions`)
    """
    a, b = prog.iv_bounds[:, 0], prog.iv_bounds[:, 1]
    if targets is None:
        needed = np.ones_like(prior)
    else:
        Q = _prefix(targets)
        needed = (Q[a] > 0) | (Q[-1] > Q[b])                         # a target outside the scope
    todo = needed & prior & ~valid
    sample = needed & ~prior
    key = (observed.tobytes(), None if targets is None else targets.tobytes(), todo.tobytes(), sample.tobytes(),
           prior.tobytes())
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
                     reg_prior = torch.from_numpy(prior[prog.reg_iv_host].astype(np.int32)).to(prog.dev))
    prog.plans[key] = plan
    if len(prog.plans) > PLAN_CACHE_SIZE:
        prog.plans.popitem(last = False)
    return plan


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
