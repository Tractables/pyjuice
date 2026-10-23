"""
The decoder's planner (:mod:`pyjuice.queries.decoder.plan`) against full evaluation. A float64 evaluator over the node
groups runs every plan literally: it recomputes only the groups the plan lists and reads every other value from its
cache -- and every value it reads must have been computed under the current evidence (on the group's scope for an
inside, outside it for an outside), checked against the evidence each value was computed under, not against the
planner's own bookkeeping. The answers, marginals and the conditionals of target variables, must equal a full
evaluation's. Sequences: left to right, diffusion-like (several positions per step, different per sample), and random
(values changed, observed, marginalized again), marginals and conditionals mixed; on an HMM, an HCLT, a PD, and a
hand-built circuit with a sum over inputs, a three-child product, a group shared by two products, and two
decompositions of the root's scope. The cuts of the variables changed since the root was last valid against the
circuit's induced trees, enumerated: each must contain exactly one of a cut's groups.
"""

import itertools

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.queries.decoder.plan import INPUT, PROD, SUM, Planner, _pending_cut


NUM_CATS = 3


class Reference:
    """Float64 log-values of a circuit's node groups (Categorical inputs) under evidence ``x`` [B, V], ``miss`` [B, V]."""

    def __init__(self, structure):
        self.s = s = structure
        self.info = []
        for g, ns in enumerate(s.groups):
            ch = s.ch_idx[s.ch_ptr[g]:s.ch_ptr[g + 1]]
            if s.kind[g] == INPUT:
                P = ns.get_source_ns().get_params().view(ns.num_nodes, NUM_CATS).double().numpy()
                self.info.append((ns.scope.to_list()[0], np.log(P)))
            elif s.kind[g] == SUM:
                W = ns.get_source_ns().get_params(as_matrix = True).double().numpy()
                offsets = np.cumsum([0] + [s.num_nodes[c] for c in ch])
                self.info.append((np.log(W), ch, offsets))
            else:
                E, bs = ns.edge_ids.numpy(), ns.block_size
                maps = [(E[:, j][:, None] * bs + np.arange(bs)[None, :]).reshape(-1) if E.shape[0] == ns.num_node_blocks
                        else E[:, j] for j in range(len(ch))]
                self.info.append((maps, ch))
        self.scope_vars = [np.array(ns.scope.to_list()) for ns in s.groups]

    def inside(self, g, get, x, miss):
        """Group ``g``'s values from its children's (``get``)."""
        kind = self.s.kind[g]
        if kind == INPUT:
            v, logP = self.info[g]
            val = logP[:, x[:, v]]
            val[:, miss[:, v]] = 0.0
            return val
        if kind == SUM:
            logW, ch, _ = self.info[g]
            C = np.concatenate([get(c) for c in ch], axis = 0)
            return np.logaddexp.reduce(logW[:, :, None] + C[None, :, :], axis = 1)
        maps, ch = self.info[g]
        return sum(get(c)[m, :] for c, m in zip(ch, maps))

    def outside_from(self, g, p, out_p, get):
        """The contribution of parent ``p`` (its outsides ``out_p``) to group ``g``'s outsides."""
        if self.s.kind[p] == SUM:
            logW, ch, offsets = self.info[p]
            j = list(ch).index(g)
            cols = slice(offsets[j], offsets[j + 1])
            return np.logaddexp.reduce(out_p[:, None, :] + logW[:, cols, None], axis = 0)
        maps, ch = self.info[p]
        j = list(ch).index(g)
        contrib = out_p + sum(get(c)[m, :] for k, (c, m) in enumerate(zip(ch, maps)) if k != j)
        acc = np.full([self.s.num_nodes[g], out_p.shape[1]], -np.inf)
        np.logaddexp.at(acc, maps[j], contrib)
        return acc

    def full(self, x, miss):
        s = self.s
        IN = {}
        for g in range(len(s.groups)):
            IN[g] = self.inside(g, lambda c: IN[c], x, miss)
        OUT = {s.root: np.zeros_like(IN[s.root])}
        for g in range(len(s.groups) - 2, -1, -1):
            acc = np.full_like(IN[g], -np.inf)
            for p in s.par_idx[s.par_ptr[g]:s.par_ptr[g + 1]]:
                acc = np.logaddexp(acc, self.outside_from(g, p, OUT[p], lambda c: IN[c]))
            OUT[g] = acc
        return IN, OUT

    def conditional(self, target_vars, out):
        """[B, targets, NUM_CATS]: P(x_t = v | e) from the outsides ``out`` of the targets' input groups."""
        rows = []
        for t in target_vars:
            logits = None
            for g in self.s.input_groups([t]):
                _, logP = self.info[g]
                lg = np.logaddexp.reduce(out[g][:, :, None] + logP[:, None, :], axis = 0)
                logits = lg if logits is None else np.logaddexp(logits, lg)
            rows.append(np.exp(logits - np.logaddexp.reduce(logits, axis = 1, keepdims = True)))
        return np.stack(rows, axis = 1)


class Session:
    """Runs plans on a cache, reading a cached value only where it was computed under the current evidence."""

    def __init__(self, planner, ref):
        self.planner, self.ref, self.s = planner, ref, planner.structure
        self.IN, self.OUT = {}, {}          # group -> (values, x, miss they were computed under)

    def _same(self, x0, m0, x, miss, vars):
        return np.array_equal(m0[:, vars], miss[:, vars]) and \
            np.array_equal(np.where(m0, -1, x0)[:, vars], np.where(miss, -1, x)[:, vars])

    def get_in(self, g, x, miss):
        if self.s.kind[g] == PROD:
            return self.ref.inside(g, lambda c: self.get_in(c, x, miss), x, miss)
        assert g in self.IN, f"read the inside of group {g}, never computed"
        val, x0, m0 = self.IN[g]
        assert self._same(x0, m0, x, miss, self.ref.scope_vars[g]), f"read a stale inside of group {g}"
        return val

    def get_out(self, g, x, miss):
        if g == self.s.root:
            return np.zeros([1, x.shape[0]])
        assert g in self.OUT, f"read the outside of group {g}, never computed"
        val, x0, m0 = self.OUT[g]
        outside = np.setdiff1d(np.arange(self.s.num_vars), self.ref.scope_vars[g])
        assert self._same(x0, m0, x, miss, outside), f"read a stale outside of group {g}"
        return val

    def recompute(self, plan, x, miss):
        assert np.all(np.diff(plan.recompute) > 0)
        for g in plan.recompute:
            assert self.s.kind[g] != PROD
            self.IN[g] = (self.ref.inside(g, lambda c: self.get_in(c, x, miss), x, miss), x.copy(), miss.copy())

    def marginal(self, plan, x, miss):
        self.recompute(plan, x, miss)
        if plan.cut.size == 0:
            return self.get_in(self.s.root, x, miss)[0]
        acc = np.full([x.shape[0]], -np.inf)
        for k in plan.cut:
            acc = np.logaddexp(acc, np.logaddexp.reduce(self.get_in(k, x, miss) + self.get_out(k, x, miss), axis = 0))
        return acc

    def conditional(self, plan, target_vars, x, miss):
        self.recompute(plan, x, miss)
        # What the flow executor reads besides: it divides by every visited group's inside, and a visited product's
        # value is the sum of all its children's
        for g in plan.visited:
            if self.s.kind[g] == PROD:
                for c in self.s.ch_idx[self.s.ch_ptr[g]:self.s.ch_ptr[g + 1]]:
                    self.get_in(c, x, miss)
            else:
                self.get_in(g, x, miss)
        out = {g: self.get_out(g, x, miss) for g in plan.seeds}
        assert np.all(np.diff(plan.visited) < 0)                   # parents first
        for g in plan.visited:
            acc = np.full([self.s.num_nodes[g], x.shape[0]], -np.inf)
            for p in self.s.par_idx[self.s.par_ptr[g]:self.s.par_ptr[g + 1]]:
                assert p in out, f"group {g}'s parent {p} has no outside in this pass"
                acc = np.logaddexp(acc, self.ref.outside_from(g, p, out[p], lambda c: self.get_in(c, x, miss)))
            out[g] = acc
            if self.s.kind[g] != PROD:
                self.OUT[g] = (acc, x.copy(), miss.copy())
        for g in plan.targets:
            assert g in out, f"target group {g} has no outside"
        return self.ref.conditional(target_vars, out)


def _changed(prev, x, miss):
    if prev is None:
        return np.arange(x.shape[1])
    x0, m0 = prev
    differ = (m0 != miss) | (~miss & (x0 != x))
    return np.flatnonzero(differ.any(axis = 0))


def _run(root, steps):
    """Run ``steps`` -- (x, miss, query, targets) -- through a planner and a session, each answer against full
    evaluation. Returns how many marginals took a cut."""
    planner = Planner(root)
    ref = Reference(planner.structure)
    sess = Session(planner, ref)
    prev, cuts = None, 0
    for x, miss, query, targets in steps:
        planner.observe(_changed(prev, x, miss))
        prev = (x.copy(), miss.copy())
        IN, OUT = ref.full(x, miss)
        if query == "marginal":
            plan = planner.marginal()
            got = sess.marginal(plan, x, miss)
            np.testing.assert_allclose(got, IN[planner.structure.root][0], rtol = 1e-9, atol = 1e-9)
            cuts += plan.cut.size > 0
        else:
            plan = planner.conditional(targets)
            got = sess.conditional(plan, targets, x, miss)
            np.testing.assert_allclose(got, ref.conditional(targets, OUT), rtol = 1e-9, atol = 1e-12)
        planner.commit(plan)
    return cuts


def _circuit(kind):
    torch.manual_seed(0)
    if kind == "hmm":
        root = juice.structures.HMM(seq_length = 6, num_latents = 4, num_emits = NUM_CATS)
    elif kind == "hclt":
        x = torch.randint(0, NUM_CATS, (256, 7))
        root = juice.structures.HCLT(x.float(), num_latents = 4, input_dist = dists.Categorical(num_cats = NUM_CATS))
    elif kind == "pd":
        root = juice.structures.PD(data_shape = (4, 4), num_latents = 2, split_intervals = 2,
                                   input_dist = dists.Categorical(num_cats = NUM_CATS))
    else:
        i = [inputs(v, num_node_blocks = 1, block_size = 2, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(4)]
        s0 = summate(i[0], num_node_blocks = 1, block_size = 2)                        # a sum over inputs
        s12 = summate(multiply(i[1], i[2]), num_node_blocks = 1, block_size = 2)
        s23 = summate(multiply(i[2], i[3]), num_node_blocks = 1, block_size = 2)
        root = summate(multiply(s0, s12, i[3]), multiply(s0, i[1], s23),           # s0 shared; two decompositions
                       num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root


def _num_vars(root):
    return len(root.scope.to_list())


def _left_to_right(V, rng, B = 3):
    x = rng.integers(0, NUM_CATS, (B, V))
    miss = np.ones((B, V), dtype = bool)
    steps = []
    for t in range(V):
        steps.append((x.copy(), miss.copy(), "conditional", [t]))
        steps.append((x.copy(), miss.copy(), "marginal", None))
        miss[:, t] = False
    steps.append((x.copy(), miss.copy(), "marginal", None))
    return steps


def _diffusion(V, rng, B = 3, k = 2):
    x = rng.integers(0, NUM_CATS, (B, V))
    miss = np.ones((B, V), dtype = bool)
    steps = []
    while miss.any():
        targets = np.flatnonzero(miss.all(axis = 0)).tolist()
        if targets:
            steps.append((x.copy(), miss.copy(), "conditional", targets))
        steps.append((x.copy(), miss.copy(), "marginal", None))
        for b in range(B):
            masked = np.flatnonzero(miss[b])
            miss[b, rng.choice(masked, size = min(k, masked.size), replace = False)] = False
    steps.append((x.copy(), miss.copy(), "marginal", None))
    return steps


def _random(V, rng, B = 3, n = 40):
    x = rng.integers(0, NUM_CATS, (B, V))
    miss = rng.random((B, V)) < 0.5
    steps = []
    for _ in range(n):
        for _ in range(rng.integers(0, 3)):
            v = rng.integers(0, V)
            op = rng.integers(0, 3)
            if op == 0:
                x[:, v] = rng.integers(0, NUM_CATS, B)
            elif op == 1:
                miss[rng.integers(0, B), v] = True
            else:
                miss[:, v] = False
        if rng.random() < 0.5:
            steps.append((x.copy(), miss.copy(), "marginal", None))
        else:
            targets = sorted(rng.choice(V, size = rng.integers(1, 3), replace = False).tolist())
            m = miss.copy()
            m[:, targets] = True
            x_t = x.copy()
            steps.append((x_t, m, "conditional", targets))
    return steps


@pytest.mark.parametrize("kind", ["hmm", "hclt", "pd", "hand"])
@pytest.mark.parametrize("sequence", ["left_to_right", "diffusion", "random"])
def test_plans_against_full_evaluation(kind, sequence):
    root = _circuit(kind)
    V = _num_vars(root)
    rng = np.random.default_rng(0)
    steps = dict(left_to_right = _left_to_right, diffusion = _diffusion, random = _random)[sequence](V, rng)
    cuts = _run(root, steps)
    if sequence == "left_to_right" and kind == "hmm":
        assert cuts > 0, "no marginal took a cut"


def test_left_to_right_on_an_hmm_reuses():
    """Left to right on an HMM: after the first calls, a conditional recomputes only the newly observed variable's
    input group and visits a constant number of groups; a marginal right after it takes the cut at that input group
    (once recomputing up to the root costs more)."""
    root = juice.structures.HMM(seq_length = 12, num_latents = 4, num_emits = NUM_CATS)
    root.init_parameters(perturbation = 2.0)
    planner = Planner(root)
    s = planner.structure
    V = s.num_vars
    rng = np.random.default_rng(1)
    x = rng.integers(0, NUM_CATS, (2, V))
    miss = np.ones((2, V), dtype = bool)
    prev = None
    visited = []
    for t in range(V):
        planner.observe(_changed(prev, x, miss))
        prev = (x.copy(), miss.copy())
        plan = planner.conditional([t])
        if t >= 2:
            assert set(plan.recompute.tolist()) <= set(s.input_groups([t - 1]).tolist())
        visited.append(plan.visited.size)
        planner.commit(plan)
        miss[:, t] = False
        planner.observe(_changed(prev, x, miss))
        prev = (x.copy(), miss.copy())
        plan = planner.marginal()
        if t >= 2:                          # nearer the root, recomputing up to it costs no more than the cut
            assert plan.cut.size > 0 and set(plan.cut.tolist()) == set(s.input_groups([t]).tolist())
        planner.commit(plan)
    assert max(visited[2:]) <= 4, visited


def _induced_trees(structure, limit = 200000):
    """Every induced tree of the circuit at the level of groups (a sum picks one child group, a product all of them),
    as sets of groups -- a tree never holds two nodes of one group -- or None past ``limit``."""
    s = structure
    trees = []
    for g in range(len(s.groups)):
        ch = s.ch_idx[s.ch_ptr[g]:s.ch_ptr[g + 1]]
        if s.kind[g] == INPUT:
            t = [frozenset([g])]
        elif s.kind[g] == SUM:
            t = [tr | {g} for c in ch for tr in trees[c]]
        else:
            t = [frozenset().union(*combo) | {g} for combo in itertools.product(*[trees[c] for c in ch])]
        if len(t) > limit:
            return None
        trees.append(t)
    return trees[s.root]


@pytest.mark.parametrize("kind", ["hmm", "hclt", "pd", "hand"])
def test_pending_cuts_are_cuts(kind):
    """For every set of up to three variables (or 300 sets drawn), the lifted cut `_pending_cut` calls usable (all
    outsides valid) holds exactly one group of every induced tree."""
    root = _circuit(kind)
    s = Planner(root).structure
    trees = _induced_trees(s)
    assert trees is not None
    V = s.num_vars
    sets = [c for k in (1, 2, 3) for c in itertools.combinations(range(V), k)]
    if len(sets) > 300:
        rng = np.random.default_rng(2)
        sets = [sets[i] for i in rng.choice(len(sets), size = 300, replace = False)]
    out_ok = np.ones(s.kind.size, dtype = np.uint8)
    usable = 0
    for S in sets:
        cut, valid = _pending_cut(np.array(S, dtype = np.int64), s.kind, s.scope, s.ch_ptr, s.ch_idx, s.par_ptr,
                                  s.par_idx, out_ok)
        if valid:
            usable += 1
            members = set(np.flatnonzero(cut).tolist())
            for tr in trees:
                assert len(tr & members) == 1, (S, sorted(members), sorted(tr))
    assert usable > 0


def test_a_lift_that_is_no_cut_is_refused():
    """The hand-built circuit's variables 1 and 2: covered deepest by the product under s12 and by the root's second
    decomposition -- whose parent, the root, has the first decomposition (not among them) for a child too. The lift
    {s12, root} would count the first decomposition's trees twice."""
    root = _circuit("hand")
    s = Planner(root).structure
    cut, valid = _pending_cut(np.array([1, 2], dtype = np.int64), s.kind, s.scope, s.ch_ptr, s.ch_idx, s.par_ptr,
                              s.par_idx, np.ones(s.kind.size, dtype = np.uint8))
    assert not valid
