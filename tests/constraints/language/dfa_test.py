import itertools
import random

import numpy as np
import pytest
import torch

from pyjuice.constraints import DFA


def strings(V, max_len):
    for L in range(max_len + 1):
        yield from itertools.product(range(V), repeat = L)


def contains_ref(patterns, t):
    return any(tuple(t[i:i + len(p)]) == tuple(p) for p in patterns for i in range(len(t) - len(p) + 1))


def random_dfa(K, V, seed):
    rng = random.Random(seed)
    dense = [[rng.randrange(K) for _ in range(V)] for _ in range(K)]
    accept = rng.sample(range(K), rng.randint(1, K))
    return DFA.from_dense(V, dense, 0, accept), dense, set(accept)


def run_dense(dense, accept, t):
    q = 0
    for x in t:
        q = dense[q][x]
    return q in accept


# -------------------------------------------------------------------------------------------------
# Construction and membership
# -------------------------------------------------------------------------------------------------

def test_basic_languages():
    V = 3
    for t in strings(V, 4):
        assert DFA.anything(V).accepts(t)
        assert not DFA.nothing(V).accepts(t)
    for seed in range(5):
        dfa, dense, accept = random_dfa(4, V, seed)
        for t in strings(V, 5):
            assert dfa.accepts(t) == run_dense(dense, accept, t)


# The last three are self-overlapping: after a partial match fails, the right state is a proper suffix
# of the partial match, so they only pass with correct Aho-Corasick failure links.
@pytest.mark.parametrize("patterns", [[[1, 2]], [[0, 1], [2]], [[1, 1, 0], [1, 0]], [[2, 2, 2]],
                                      [[1, 1, 0]], [[0, 1, 0, 1, 1]], [[1, 2, 1, 2, 0], [2, 0, 0]]])
def test_contains_matches_substring_semantics(patterns):
    V = 3
    dfa = DFA.contains(patterns, V)
    for t in strings(V, 7):
        assert dfa.accepts(t) == contains_ref(patterns, t), t


def test_token_classes_are_compressed():
    V = 50
    dfa = DFA.contains([[3, 7], [7, 9]], V)
    assert dfa.num_classes == 4                          # 3, 7, 9 and "everything else"
    sizes = dfa.class_sizes()
    assert sizes.sum() == V and sorted(sizes.tolist()) == [1, 1, 1, 47]
    # tokens in one class behave identically from every state
    dense = dfa.to_dense()
    for c in range(dfa.num_classes):
        toks = (dfa.token_class == c).nonzero().flatten()
        assert (dense[:, toks] == dense[:, toks[:1]]).all()


def test_from_transitions_partial_map_with_arbitrary_ids():
    # "a token 1 somewhere, then a token 2 later", over V = 4, with outlines-style sparse ids
    V = 4
    trans = {512: {0: 512, 2: 512, 3: 512, 1: 640}, 640: {0: 640, 1: 640, 3: 640, 2: 768},
             768: {t: 768 for t in range(V)}}
    dfa = DFA.from_transitions(V, trans, initial = 512, accept = [768])
    ref = lambda t: any(t[i] == 1 and 2 in t[i + 1:] for i in range(len(t)))
    for t in strings(V, 5):
        assert dfa.accepts(t) == ref(t), t
    # a missing (state, token) pair goes to a dead state
    sparse = DFA.from_transitions(V, {0: {1: 1}, 1: {}}, initial = 0, accept = [1])
    assert sparse.accepts([1]) and not sparse.accepts([1, 0]) and not sparse.accepts([0, 1])


def test_from_ctrlg_graph():
    V = 4
    rng = np.random.default_rng(0)
    # Ctrl-G style: edges (u, v, bitset) with tuple state names
    bits = lambda toks: np.isin(np.arange(V), toks)
    graph = {"edges": [(("a",), ("a",), bits([0, 2, 3])), (("a",), ("b",), bits([1])),
                       (("b",), ("b",), bits([0, 1, 2, 3]))],
             "initial_state": ("a",), "accept_states": {("b",)}}
    dfa = DFA.from_ctrlg(graph, V)
    for t in strings(V, 4):
        assert dfa.accepts(t) == (1 in t)
    with pytest.raises(ValueError, match = "not complete"):
        DFA.from_ctrlg({"edges": [(0, 1, bits([1]))], "initial_state": 0, "accept_states": {1}}, V)


def test_invalid_tables_raise():
    with pytest.raises(ValueError, match = "complete"):
        DFA(2, [0, 0], [[1]], 0, [0])
    with pytest.raises(ValueError, match = "token_class"):
        DFA(2, [0, 1], [[0]], 0, [0])
    with pytest.raises(ValueError, match = "initial"):
        DFA(2, [0, 0], [[0]], 1, [0])


# -------------------------------------------------------------------------------------------------
# Minimisation
# -------------------------------------------------------------------------------------------------

def test_minimize_preserves_language_and_is_canonical():
    V = 3
    for seed in range(20):
        dfa, dense, accept = random_dfa(6, V, seed)
        m = dfa.minimize()
        assert m.num_states <= dfa.num_states
        for t in strings(V, 5):
            assert m.accepts(t) == dfa.accepts(t)
        assert m.minimize() == m                              # idempotent
        # a relabelled copy (states permuted) minimises to the identical table
        perm = list(range(6)); random.Random(seed).shuffle(perm)
        inv = {p: i for i, p in enumerate(perm)}
        dense2 = [[perm[dense[inv[i]][x]] for x in range(V)] for i in range(6)]
        dfa2 = DFA.from_dense(V, dense2, perm[0], [perm[a] for a in accept])
        assert dfa2.minimize() == m and dfa2.equivalent(dfa)


def test_minimize_agrees_with_automata_lib_state_count():
    from automata.fa.dfa import DFA as RefDFA
    V = 3
    for seed in range(20):
        dfa, dense, accept = random_dfa(7, V, seed)
        ref = RefDFA(states = set(range(7)), input_symbols = set(range(V)),
                     transitions = {q: {x: dense[q][x] for x in range(V)} for q in range(7)},
                     initial_state = 0, final_states = set(accept)).minify()
        assert dfa.minimize().num_states == len(ref.states), seed


def test_inequivalent_dfas_are_told_apart():
    V = 3
    assert not DFA.contains([[1, 2]], V).equivalent(DFA.contains([[2, 1]], V))
    assert DFA.contains([[1], [1, 2]], V).equivalent(DFA.contains([[1]], V))


# -------------------------------------------------------------------------------------------------
# Matcher
# -------------------------------------------------------------------------------------------------

def brute_allowed(dfa, prefix, V, remaining, max_extra):
    """Token v is allowed iff some continuation (of exactly `remaining` tokens incl. v, or of any
    length up to max_extra if remaining is None) of prefix + v is accepted."""
    out = torch.zeros(V, dtype = torch.bool)
    for v in range(V):
        lengths = [remaining - 1] if remaining is not None else range(max_extra + 1)
        for L in lengths:
            if any(dfa.accepts(list(prefix) + [v] + list(w)) for w in itertools.product(range(V), repeat = L)):
                out[v] = True
                break
    return out


def test_matcher_masks_are_exact():
    V = 3
    for seed, dfa in [(0, DFA.contains([[1, 2]], V)), (1, random_dfa(5, V, 1)[0]),
                      (2, DFA.contains([[0, 0], [2, 1]], V)), (3, random_dfa(4, V, 7)[0])]:
        for prefix in strings(V, 3):
            m = dfa.matcher()
            ok = all(m.advance(t) for t in prefix)
            if not ok:
                continue
            # dead-state-free automaton of <= 5 states: any accepting continuation has length < 5
            assert torch.equal(m.allowed_next(), brute_allowed(dfa, prefix, V, None, 5)), (seed, prefix)
            for r in (1, 2, 3):
                assert torch.equal(m.allowed_next(remaining = r), brute_allowed(dfa, prefix, V, r, 0)), (seed, prefix, r)
            assert m.is_accepting() == dfa.accepts(prefix)


def test_matcher_advance_rollback_clone():
    V = 3
    dfa = DFA.from_transitions(V, {0: {1: 1}, 1: {2: 2}, 2: {}}, initial = 0, accept = [2])   # exactly [1, 2]
    m = dfa.matcher()
    assert not m.advance(0) and m.num_consumed == 0         # rejected tokens leave the state alone
    assert m.advance(1) and m.advance(2) and m.is_accepting()
    assert not m.allowed_next().any()                        # nothing may follow [1, 2]
    c = m.clone()
    m.rollback(2)
    assert m.num_consumed == 0 and not m.is_accepting() and c.is_accepting() and c.num_consumed == 2
    with pytest.raises(ValueError):
        m.rollback(1)


# -------------------------------------------------------------------------------------------------
# Factorised-weight primitives
# -------------------------------------------------------------------------------------------------

def brute_wmc(dfa, lw):
    B, n, V = lw.shape
    out = torch.full((B,), float("-inf"), dtype = lw.dtype)
    for x in itertools.product(range(V), repeat = n):
        if dfa.accepts(x):
            out = torch.logaddexp(out, lw[:, torch.arange(n), list(x)].sum(dim = 1))
    return out


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_wmc_matches_brute_force(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")
    V, n = 3, 5
    torch.manual_seed(0)
    for dfa in [DFA.contains([[1, 2]], V), DFA.contains([[0, 0], [2, 1, 2]], V), random_dfa(5, V, 3)[0],
                DFA.nothing(V)]:
        lw = torch.randn(4, n, V, dtype = torch.float64)
        lw[0, 2, 1] = float("-inf")                          # a forbidden token
        got = dfa.wmc(lw.to(device)).cpu()
        want = brute_wmc(dfa, lw)
        assert torch.equal(torch.isfinite(got), torch.isfinite(want))
        fin = torch.isfinite(want)
        assert torch.allclose(got[fin], want[fin], atol = 1e-10)


def test_sample_matches_exact_distribution():
    V, n, S = 3, 4, 20000
    dfa = DFA.contains([[1, 2]], V)
    torch.manual_seed(0)
    lw = torch.randn(2, n, V, dtype = torch.float64)
    gen = torch.Generator().manual_seed(1)
    xs = dfa.sample(lw, S, generator = gen)
    assert xs.shape == (2, S, n)
    for b in range(2):
        target = {}
        for x in itertools.product(range(V), repeat = n):
            if dfa.accepts(x):
                target[x] = float(lw[b, torch.arange(n), list(x)].sum().exp())
        Z = sum(target.values())
        counts = {}
        for x in map(tuple, xs[b].tolist()):
            assert x in target, "sampled a rejected sequence"
            counts[x] = counts.get(x, 0) + 1
        tv = 0.5 * sum(abs(counts.get(x, 0) / S - p / Z) for x, p in target.items())
        assert tv < 3 * (len(target) / (2 * np.pi * S)) ** 0.5, tv


def test_sample_raises_when_nothing_is_accepted():
    with pytest.raises(ValueError, match = "No sequence"):
        DFA.nothing(3).sample(torch.zeros(1, 2, 3), 1)


# -------------------------------------------------------------------------------------------------
# Composition (automata built by And / Or / Not / Concat nodes)
# -------------------------------------------------------------------------------------------------

def test_composite_automata_match_definitions_on_all_short_strings():
    V = 3
    a = random_dfa(3, V, 11)[0]
    b = DFA.contains([[1, 2]], V)
    c = DFA.contains([[0, 0]], V)
    d = random_dfa(4, V, 12)[0]
    cases = {
        "and": (a & b, lambda t: a.accepts(t) and b.accepts(t)),
        "or3": (a | b | d, lambda t: a.accepts(t) or b.accepts(t) or d.accepts(t)),
        "not": (~b, lambda t: not b.accepts(t)),
        "concat": (b.concat(c), lambda t: any(b.accepts(t[:k]) and c.accepts(t[k:]) for k in range(len(t) + 1))),
        "concat_random": (a.concat(d), lambda t: any(a.accepts(t[:k]) and d.accepts(t[k:]) for k in range(len(t) + 1))),
        "concat3": (a.concat(b).concat(d),
                    lambda t: any(a.accepts(t[:i]) and b.accepts(t[i:j]) and d.accepts(t[j:])
                                  for i in range(len(t) + 1) for j in range(i, len(t) + 1))),
        "nested": ((~(a & b)).concat(c) | d,
                   lambda t: any(not (a.accepts(t[:k]) and b.accepts(t[:k])) and c.accepts(t[k:])
                                 for k in range(len(t) + 1)) or d.accepts(t)),
    }
    for name, (node, ref) in cases.items():
        dfa = node.automaton()
        assert dfa.minimize() == dfa, name                     # built automata are canonical
        for t in strings(V, 7):
            want = ref(t)
            assert node.accepts(t) == want and dfa.accepts(t) == want, (name, t)


def test_equivalent_compositions_give_identical_automata():
    V = 3
    a, b = DFA.contains([[1, 2]], V), DFA.contains([[0, 0]], V)
    assert (a & b).automaton() == (b & a).automaton()
    assert (~(a | b)).automaton() == ((~a) & (~b)).automaton()          # De Morgan
    assert (a & DFA.anything(V)).automaton() == a.minimize()
    assert (a | DFA.nothing(V)).automaton() == a.minimize()


def test_capabilities():
    assert DFA.anything(3).capabilities() == {"accepts", "matcher", "wmc", "sample", "automaton", "relax"}
    d = DFA.contains([[1]], 3)
    assert d.automaton() is d and d.relax() is d
