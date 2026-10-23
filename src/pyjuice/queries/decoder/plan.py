"""
The decoder's planner: from which variables a call changed (in any sample: one change set for the whole batch) and
what it asks, which node groups (:class:`~pyjuice.nodes.CircuitNodes`) the call recomputes, which cached values it
reads, and where a conditional's top-down pass starts.

Validity, per group, for the whole batch (smooth and decomposable circuits):

* ``in_ok``: its node values (insides) are those of the current evidence on its scope. Tracked for sum and input
  groups; a product's values are recomputed from its children whenever they are needed.
* ``out_ok``: its stored log-outsides are those of the current evidence outside its scope. Sum and input groups; the
  root's outside is 0, always.

A call that changes the variables ``S`` keeps ``in_ok`` only where the scope misses ``S``, and ``out_ok`` only where
the scope contains ``S``.

Plans:

* :meth:`Planner.marginal` -- the cheapest of: recomputing every stale group up to the root, whose value is the
  marginal; or a cut -- groups every induced tree passes through exactly one node of, all with valid outsides, ``log p
  = logsumexp over the cut of in + out``, recomputing only the stale groups below it. The cuts tried: a variable's
  input groups (every tree has exactly one leaf of it), for every variable whose input groups' outsides are valid; and
  the deepest groups whose scope covers the variables changed since the root was last current, lifted to their parent
  sums when every child of those is among them.
* :meth:`Planner.conditional` -- the targets' input groups need outsides. Walk up through every parent until groups
  whose outsides are valid (the seeds; the root's always is). Every group the walk passes through (visited) gets its
  outside computed; the pass reads the insides of the visited groups, of every child of a visited product, and of the
  targets' input groups, so the stale ones among those are recomputed first. A seed's inside is not read: the top-down
  pass starts it from its stored outside.

The structure is flattened once into int64 arrays (:class:`Structure`); the passes are numba functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from pyjuice.nodes import CircuitNodes


INPUT, PROD, SUM = 0, 1, 2


class Structure:
    """
    A circuit's node groups, children before parents (``groups``; the root last), flattened:

    * ``kind`` [G]: :data:`INPUT`, :data:`PROD` or :data:`SUM`;
    * ``scope`` [G, words] uint64: every group's variables as a bitset;
    * ``ch_ptr`` / ``ch_idx``, ``par_ptr`` / ``par_idx``: children and parents (CSR);
    * ``var_ptr`` / ``var_idx``: the groups whose scope contains each variable; ``inp_ptr`` / ``inp_idx``: its input
      groups;
    * ``cost`` [G] float64: a group's work -- its edges for a sum or product, its nodes for an input group;
    * ``num_nodes`` [G].
    """

    def __init__(self, root_ns: CircuitNodes):
        groups = _post_order(root_ns)
        index = {id(ns): g for g, ns in enumerate(groups)}
        G = len(groups)

        self.groups = groups
        self.root = G - 1
        self.kind = np.array([INPUT if ns.is_input() else (PROD if ns.is_prod() else SUM) for ns in groups],
                             dtype = np.int64)
        self.num_nodes = np.array([ns.num_nodes for ns in groups], dtype = np.int64)
        self.cost = np.array([float(ns.num_nodes if ns.is_input() else ns.num_edges) for ns in groups],
                             dtype = np.float64)

        scopes = [ns.scope.to_list() for ns in groups]
        self.num_vars = max(max(s) for s in scopes if len(s) > 0) + 1
        words = (self.num_vars + 63) // 64
        self.scope = np.zeros([G, words], dtype = np.uint64)
        for g, s in enumerate(scopes):
            for v in s:
                self.scope[g, v >> 6] |= np.uint64(1) << np.uint64(v & 63)

        children = [[index[id(cs)] for cs in ns.chs] if not ns.is_input() else [] for ns in groups]
        parents = [[] for _ in range(G)]
        for g, chs in enumerate(children):
            for c in chs:
                parents[c].append(g)
        self.ch_ptr, self.ch_idx = _csr(children)
        self.par_ptr, self.par_idx = _csr(parents)

        var_groups = [[] for _ in range(self.num_vars)]
        var_inputs = [[] for _ in range(self.num_vars)]
        for g, s in enumerate(scopes):
            for v in s:
                var_groups[v].append(g)
                if groups[g].is_input():
                    var_inputs[v].append(g)
        self.var_ptr, self.var_idx = _csr(var_groups)
        self.inp_ptr, self.inp_idx = _csr(var_inputs)

    def input_groups(self, variables) -> np.ndarray:
        """The input groups of ``variables``, ascending."""
        variables = np.asarray(variables, dtype = np.int64)
        out = [self.inp_idx[self.inp_ptr[v]:self.inp_ptr[v + 1]] for v in variables]
        return np.unique(np.concatenate(out)) if len(out) > 0 else np.zeros([0], dtype = np.int64)


def _post_order(root_ns: CircuitNodes) -> list:
    """The groups under ``root_ns``, children before parents (the root last) -- iteratively: a deep chain's recursion
    would exceed Python's limit."""
    order, seen, stack = [], set(), [(root_ns, False)]
    while stack:
        ns, expanded = stack.pop()
        if expanded:
            order.append(ns)
            continue
        if id(ns) in seen:
            continue
        seen.add(id(ns))
        stack.append((ns, True))
        if not ns.is_input():
            for cs in reversed(ns.chs):
                if id(cs) not in seen:
                    stack.append((cs, False))
    return order


def _csr(lists):
    ptr = np.zeros([len(lists) + 1], dtype = np.int64)
    ptr[1:] = np.cumsum([len(l) for l in lists])
    idx = np.array([x for l in lists for x in l], dtype = np.int64)
    return ptr, idx


@dataclass
class MarginalPlan:
    """``recompute``: sum and input groups whose insides the call recomputes, children first. ``cut``: empty when the
    marginal is the root's value; else the groups ``log p = logsumexp over their nodes of in + out`` runs over."""
    recompute: np.ndarray
    cut: np.ndarray
    cost: float


@dataclass
class ConditionalPlan:
    """``recompute``: as for a marginal. ``seeds``: groups whose stored outsides start the top-down pass (the root's
    is 0). ``visited``: the groups whose outsides the pass computes, parents first (products among them).
    ``targets``: the targets' input groups."""
    recompute: np.ndarray
    seeds: np.ndarray
    visited: np.ndarray
    targets: np.ndarray


class Planner:
    """
    Validity of a circuit's cached values across a decoder session's calls, and the plan of every call. Per call:
    :meth:`observe` the variables whose evidence changed in any sample, plan with :meth:`marginal` or
    :meth:`conditional`, run the plan, then :meth:`commit` it. Nothing is valid before the first call.
    """

    def __init__(self, root_ns: CircuitNodes):
        self.structure = s = Structure(root_ns)
        G = s.kind.size
        self.in_ok = np.zeros([G], dtype = np.uint8)
        self.out_ok = np.zeros([G], dtype = np.uint8)
        self.out_ok[s.root] = 1
        self.pending = np.zeros([s.num_vars], dtype = np.uint8)     # changed since the root's inside was last valid

    def reset(self) -> None:
        """Nothing valid, as before the first call."""
        self.in_ok[:] = 0
        self.out_ok[:] = 0
        self.out_ok[self.structure.root] = 1
        self.pending[:] = 0

    def observe(self, changed) -> None:
        """The variables whose evidence (value, or missing or not) changed in any sample since the last call."""
        s = self.structure
        changed = np.asarray(changed, dtype = np.int64)
        _invalidate(changed, s.var_ptr, s.var_idx, s.scope, self.in_ok, self.out_ok, s.root)
        self.pending[changed] = 1

    def marginal(self, cuts: bool = True) -> MarginalPlan:
        """The plan of a marginal; ``cuts`` False: always up to the root."""
        s = self.structure
        recompute, cut, cost = _marginal(np.flatnonzero(self.pending).astype(np.int64), s.kind, s.scope, s.ch_ptr,
                                         s.ch_idx, s.par_ptr, s.par_idx, s.inp_ptr, s.inp_idx, self.in_ok,
                                         self.out_ok, s.cost, s.num_nodes.astype(np.float64), s.root, cuts)
        return MarginalPlan(recompute, cut, cost)

    def conditional(self, target_vars, reuse: bool = True) -> ConditionalPlan:
        """The plan of a conditional of ``target_vars``; ``reuse`` False: no stored outside is read -- the pass
        starts at the root, from its inside (recomputed if stale), as a full backward does."""
        s = self.structure
        targets = s.input_groups(target_vars)
        out_ok = self.out_ok if reuse else np.zeros_like(self.out_ok)
        recompute, seeds, visited = _conditional(targets, s.kind, s.ch_ptr, s.ch_idx, s.par_ptr, s.par_idx,
                                                 self.in_ok, out_ok, s.root, not reuse)
        return ConditionalPlan(recompute, seeds, visited, targets)

    def commit(self, plan, outsides: bool = True) -> None:
        """``plan`` ran: its recomputed insides, and a conditional's visited outsides (unless ``outsides`` is
        False: they were not stored), are valid."""
        self.in_ok[plan.recompute] = 1
        if isinstance(plan, ConditionalPlan) and outsides:
            visited = plan.visited[self.structure.kind[plan.visited] != PROD]
            self.out_ok[visited] = 1
        if self.in_ok[self.structure.root]:
            self.pending[:] = 0


@njit(cache = True)
def _invalidate(changed, var_ptr, var_idx, scope, in_ok, out_ok, root):
    G, W = scope.shape
    words = np.zeros(W, dtype = np.uint64)
    for v in changed:
        words[v >> 6] |= np.uint64(1) << np.uint64(v & 63)
        for j in range(var_ptr[v], var_ptr[v + 1]):
            in_ok[var_idx[j]] = 0
    for g in range(G):
        if g != root and out_ok[g]:
            for w in range(W):
                if words[w] & ~scope[g, w]:
                    out_ok[g] = 0
                    break


@njit(cache = True)
def _closure(required, kind, ch_ptr, ch_idx, in_ok):
    """The sum and input groups to recompute so that the values of the ``required`` groups can be read: the stale ones
    among them and, recursively, among what a recomputed sum or a product reads."""
    G = kind.size
    need = required.copy()
    recompute = np.zeros(G, dtype = np.uint8)
    for g in range(G - 1, -1, -1):                       # parents before children
        if not need[g]:
            continue
        if kind[g] == PROD or not in_ok[g]:
            if kind[g] != PROD:
                recompute[g] = 1
            for j in range(ch_ptr[g], ch_ptr[g + 1]):
                need[ch_idx[j]] = 1
    return recompute


@njit(cache = True)
def _marginal(pending, kind, scope, ch_ptr, ch_idx, par_ptr, par_idx, inp_ptr, inp_idx, in_ok, out_ok, cost, num_nodes,
              root, cuts):
    G, W = scope.shape

    # To the root
    required = np.zeros(G, dtype = np.uint8)
    required[root] = 1
    best = _closure(required, kind, ch_ptr, ch_idx, in_ok)
    best_cost = 0.0
    for g in range(G):
        if best[g]:
            best_cost += cost[g]
    best_cut = np.zeros(G, dtype = np.uint8)
    if not cuts:
        return np.nonzero(best)[0].astype(np.int64), np.nonzero(best_cut)[0].astype(np.int64), best_cost

    # A variable's input groups, where their outsides are valid: recompute the stale ones among them (no children)
    for v in range(inp_ptr.size - 1):
        if inp_ptr[v] == inp_ptr[v + 1]:
            continue
        ok, c = 1, 0.0
        for j in range(inp_ptr[v], inp_ptr[v + 1]):
            g = inp_idx[j]
            if not out_ok[g]:
                ok = 0
                break
            c += num_nodes[g]
            if not in_ok[g]:
                c += cost[g]
        if ok and c < best_cost:
            best_cost = c
            best = np.zeros(G, dtype = np.uint8)
            best_cut = np.zeros(G, dtype = np.uint8)
            for j in range(inp_ptr[v], inp_ptr[v + 1]):
                g = inp_idx[j]
                best_cut[g] = 1
                if not in_ok[g]:
                    best[g] = 1

    # The deepest groups covering the pending variables, lifted (`_pending_cut`)
    if pending.size > 0:
        cut, valid = _pending_cut(pending, kind, scope, ch_ptr, ch_idx, par_ptr, par_idx, out_ok)
        if valid:
            recompute = _closure(cut, kind, ch_ptr, ch_idx, in_ok)
            c = 0.0
            for g in range(G):
                if recompute[g]:
                    c += cost[g]
                if cut[g]:
                    c += num_nodes[g]
            if c < best_cost:
                best_cost, best, best_cut = c, recompute, cut

    return np.nonzero(best)[0].astype(np.int64), np.nonzero(best_cut)[0].astype(np.int64), best_cost


@njit(cache = True)
def _pending_cut(pending, kind, scope, ch_ptr, ch_idx, par_ptr, par_idx, out_ok):
    """The cut of the variables ``pending``: the input and product groups whose scope covers them and of which no child
    does -- every induced tree passes through exactly one of their nodes -- if they are input groups; if products, their
    parent sums, when every child of those is among them (else the lift is no cut). And whether it is usable: a cut, its
    outsides valid."""
    G, W = scope.shape
    words = np.zeros(W, dtype = np.uint64)
    for v in pending:
        words[v >> 6] |= np.uint64(1) << np.uint64(v & 63)
    covers = np.zeros(G, dtype = np.uint8)
    for g in range(G):
        ok = 1
        for w in range(W):
            if words[w] & ~scope[g, w]:
                ok = 0
                break
        covers[g] = ok
    low = np.zeros(G, dtype = np.uint8)
    n_low, n_inputs = 0, 0
    for g in range(G):
        if covers[g] and kind[g] != SUM:
            deepest = 1
            for j in range(ch_ptr[g], ch_ptr[g + 1]):
                if covers[ch_idx[j]]:
                    deepest = 0
                    break
            if deepest:
                low[g] = 1
                n_low += 1
                if kind[g] == INPUT:
                    n_inputs += 1

    cut = np.zeros(G, dtype = np.uint8)
    valid = n_low > 0
    if n_inputs == n_low:
        cut[:] = low                                     # input groups: their own outsides
    elif n_inputs == 0:
        for g in range(G):                               # products: lifted to their parent sums ...
            if low[g]:
                for j in range(par_ptr[g], par_ptr[g + 1]):
                    p = par_idx[j]
                    if kind[p] != SUM:
                        valid = False
                    cut[p] = 1
        for g in range(G):                               # ... every child of which is among them
            if cut[g]:
                for j in range(ch_ptr[g], ch_ptr[g + 1]):
                    if not low[ch_idx[j]]:
                        valid = False
    else:
        valid = False
    for g in range(G):
        if cut[g] and not out_ok[g]:
            valid = False
    return cut, valid


@njit(cache = True)
def _conditional(targets, kind, ch_ptr, ch_idx, par_ptr, par_idx, in_ok, out_ok, root, seed_insides):
    G = kind.size
    need = np.zeros(G, dtype = np.uint8)                 # needs an outside
    for g in targets:
        need[g] = 1
    seed = np.zeros(G, dtype = np.uint8)
    visited = np.zeros(G, dtype = np.uint8)
    for g in range(G):                                   # children before parents: a group's need is final here
        if not need[g]:
            continue
        if kind[g] != PROD and (out_ok[g] or g == root):
            seed[g] = 1
            continue
        visited[g] = 1
        for j in range(par_ptr[g], par_ptr[g + 1]):
            need[par_idx[j]] = 1

    # The insides the pass reads: the visited groups', every child of a visited product's, the targets'. (The visited
    # groups' follow from the other two wherever a sum's children are inputs or products, as pyjuice requires: a
    # visited sum's parents are products, a visited input group is a target. Kept so the plan does not rest on it.)
    required = np.zeros(G, dtype = np.uint8)
    for g in targets:
        required[g] = 1
    if seed_insides:                                     # the pass reads the seeds' insides too
        for g in range(G):
            if seed[g]:
                required[g] = 1
    for g in range(G):
        if visited[g]:
            if kind[g] == PROD:
                for j in range(ch_ptr[g], ch_ptr[g + 1]):      # a seed's too: it is read above its own layer
                    required[ch_idx[j]] = 1
            else:
                required[g] = 1
    recompute = _closure(required, kind, ch_ptr, ch_idx, in_ok)

    order = np.nonzero(visited)[0][::-1].astype(np.int64)  # parents first
    return np.nonzero(recompute)[0].astype(np.int64), np.nonzero(seed)[0].astype(np.int64), order
