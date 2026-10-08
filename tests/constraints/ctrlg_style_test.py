"""
Compiling Ctrl-G-style constraints against an HMM, checked part by part against brute force.

The cases are those of Ctrl-G's own correctness tests (constraints-in-pyjuice,
baselines/ctrlg/test_ctrlg_reference.py): six DFAs over a 4-token vocabulary, each placed in a
"prefix . x . suffix" window. Ctrl-G's DFA reads only ``x``; here the constraint reads all ``n`` PC
variables, so a setting ``(m, L, s)`` becomes ``n = m + L + s`` and the constraint
"any m tokens . M . any s tokens". (Ctrl-G's ``lo < hi`` sums over lengths, which a fixed window does
not express, so ``|x| = L`` exactly.)

Every string of length ``n`` is enumerated (at most 4^7), and admissibility comes from the definitions
(substring tests, splits for concatenation, the random DFAs' own tables), never from our automata. Each
part of compilation is then checked for the job it does:

* the constraint: its automaton accepts exactly the admissible strings;
* the PC structure: an HMM is right-linear;
* the layout: the columns at each boundary are exactly the automaton states admissible strings pass
  through there, and following ``next_col`` keeps a string alive exactly while its prefix can still be
  completed to an admissible string, landing on the automaton's own state at every step.
"""
import functools
import itertools
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.backends.lifted.plan import build_layout

V = 4

# (prefix length m, |x| = L, suffix length s), from Ctrl-G's settings with lo = hi; the last one is too
# short for some cases (e.g. "2 then 1 1" needs 3 tokens), so they are unsatisfiable
SETTINGS = [(0, 4, 0), (1, 3, 1), (2, 3, 2), (0, 3, 2), (1, 2, 0)]


# -------------------------------------------------------------------------------------------------
# Definitions (independent of the automata)
# -------------------------------------------------------------------------------------------------

def contains_any(X, patterns):
    """[N] whether each row of X [N, L] contains one of the patterns as a contiguous substring."""
    out = torch.zeros(X.size(0), dtype = torch.bool)
    for p in patterns:
        if len(p) <= X.size(1):
            out |= (X.unfold(1, len(p), 1) == torch.tensor(p)).all(dim = 2).any(dim = 1)
    return out


def then(first, second):
    """Concatenation by definition: some split X = u . v with u in `first` and v in `second`."""
    return lambda X: functools.reduce(torch.logical_or,
                                      [first(X[:, :j]) & second(X[:, j:]) for j in range(X.size(1) + 1)])


def run_table(table, accept, X):
    q = torch.zeros(X.size(0), dtype = torch.long)
    for t in range(X.size(1)):
        q = table[q, X[:, t]]
    return accept[q]


def random_table(K, rng):
    """A random complete DFA over the tokens (as in Ctrl-G's tests): table [K, V], accepting mask [K]."""
    table = torch.tensor([[rng.randrange(K) for _ in range(V)] for _ in range(K)])
    accept = torch.zeros(K, dtype = torch.bool)
    accept[rng.sample(range(K), rng.randint(1, K))] = True
    return table, accept


def make_cases():
    rng = random.Random(5)                      # a seed whose random DFAs reject some strings in every setting
    t3, a3 = random_table(3, rng)
    t5, a5 = random_table(5, rng)
    tb, ab = random_table(3, rng)
    rand = lambda t, a: jc.DFA.from_dense(V, t, initial = 0, accept = a)
    keyword = lambda *ps: (lambda X: contains_any(X, ps))
    return {
        # name: (constraint M on x, its definition on x)
        "random3": (rand(t3, a3), lambda X: run_table(t3, a3, X)),
        "random5": (rand(t5, a5), lambda X: run_table(t5, a5, X)),
        "keyword": (jc.DFA.contains([[1, 2]], V), keyword([1, 2])),
        "two_keywords_or": (jc.DFA.contains([[0, 1], [3]], V), keyword([0, 1], [3])),
        "ordered_keywords": (jc.DFA.contains([[2]], V).concat(jc.DFA.contains([[1, 1]], V)),
                             then(keyword([2]), keyword([1, 1]))),
        "and_with_random": (jc.DFA.contains([[2, 0]], V) & rand(tb, ab),
                            lambda X: contains_any(X, [[2, 0]]) & run_table(tb, ab, X)),
    }


CASES = make_cases()


def exactly(m):
    """Any m tokens."""
    dense = torch.tensor([[min(q + 1, m + 1)] * V for q in range(m + 2)])      # state m + 1 is dead
    return jc.DFA.from_dense(V, dense, initial = 0, accept = [m])


def windowed(M, m, s):
    parts = ([exactly(m)] if m else []) + [M] + ([exactly(s)] if s else [])
    return functools.reduce(lambda a, b: a.concat(b), parts)


def all_strings(n):
    return torch.tensor(list(itertools.product(range(V), repeat = n)), dtype = torch.long)


def automaton_states(dfa, X):
    """[N, n+1] the automaton's state at every boundary of every string."""
    q = torch.full((X.size(0),), dfa.initial, dtype = torch.long)
    states = [q]
    for t in range(X.size(1)):
        q = dfa.delta[q, dfa.token_class[X[:, t]]]
        states.append(q)
    return torch.stack(states, dim = 1)


def hmm(n, seed = 0):
    torch.manual_seed(seed)
    return juice.compile(juice.structures.HMM(seq_length = n, num_latents = 4, num_emits = V), verbose = False)


# -------------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("setting", SETTINGS, ids = lambda st: "m{}-L{}-s{}".format(*st))
@pytest.mark.parametrize("case", list(CASES))
def test_compile_a_ctrlg_case(case, setting):
    m, L, s = setting
    n = m + L + s
    M, definition = CASES[case]
    c = windowed(M, m, s)
    X = all_strings(n)
    want = definition(X[:, m:m + L])                                       # [N] admissible

    # the constraint: its automaton accepts exactly the admissible strings (and so does `accepts`)
    dfa = c.automaton()
    states = automaton_states(dfa, X)
    assert torch.equal(dfa.accept[states[:, n]], want)
    for i in random.Random(1).sample(range(X.size(0)), min(100, X.size(0))):
        assert c.accepts(X[i]) == bool(want[i])

    # compile binds the PC's structure and the automaton; an HMM is right-linear
    cc = jc.compile(c, hmm(n))
    assert cc.n == n and cc.automaton is dfa and cc.satisfiable == bool(want.any())
    assert cc.structure.right_linear and cc.structure.unsupported == ()
    assert {k for k, v in cc.shape_counts.items() if v > 0} == {"suffix", "whole"}

    # the layout: active columns are exactly the states admissible strings pass through
    lay = cc.layout
    for t in range(n + 1):
        w = int(lay.width[t])
        active = lay.state_id[t, :w]
        assert active.tolist() == sorted(set(states[want, t].tolist())), t      # increasing state id
        assert (lay.state_id[t, w:] == -1).all()

    # walking next_col: alive exactly while the prefix can still be completed, on the automaton's state
    prefix_code = lambda Y, t: (Y[:, :t] * V ** torch.arange(t)).sum(dim = 1)   # injective on length-t prefixes
    col = torch.full((X.size(0),), 0 if cc.satisfiable else -1, dtype = torch.long)
    for t in range(n + 1):
        alive = col >= 0
        completable = torch.isin(prefix_code(X, t), prefix_code(X[want], t))
        assert torch.equal(alive, completable), t
        assert torch.equal(lay.state_id[t, col[alive]], states[alive, t]), t
        if t < n:
            nxt = torch.full_like(col, -1)
            nxt[alive] = lay.next_col[t, col[alive], dfa.token_class[X[alive, t]]]
            col = nxt
    assert torch.equal(col >= 0, want)


def test_the_cases_cover_both_outcomes_and_pruning():
    """Guards the test matrix itself: no case accepts every string, some pairs are unsatisfiable, and the
    layout prunes states that are reachable but cannot finish in time."""
    satisfiable, pruned = set(), False
    for (M, definition), (m, L, s) in itertools.product(CASES.values(), SETTINGS):
        assert not definition(all_strings(L)).all()
        dfa = windowed(M, m, s).automaton()
        n = m + L + s
        layout = build_layout(dfa, n)
        satisfiable.add(layout.satisfiable)
        reachable = automaton_states(dfa, all_strings(n))
        pruned |= any(int(layout.width[t]) < len(set(reachable[:, t].tolist())) for t in range(n + 1))
    assert satisfiable == {True, False} and pruned


def test_a_ctrlg_case_rebinds_to_an_hmm_with_other_parameters():
    """A drafter/verifier-style pair (same structure, different parameters) shares one plan."""
    m, L, s = 1, 3, 1
    c = windowed(CASES["ordered_keywords"][0], m, s)
    pa, pb = hmm(m + L + s, seed = 0), hmm(m + L + s, seed = 1)
    assert not torch.equal(pa.params, pb.params)
    ca = jc.compile(c, pa)
    cb = ca.with_pc(pb)
    assert cb.pc is pb and cb.layout is ca.layout and cb.automaton is ca.automaton
    assert jc.compile(c, pb).layout is ca.layout                  # recompiling hits the layout cache
