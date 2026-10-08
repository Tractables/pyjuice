import itertools
import random

import torch

import pyjuice.constraints as jc
from pyjuice.constraints.backends.lifted.plan import build_layout


def random_dfa(K, V, seed):
    rng = random.Random(seed)
    dense = [[rng.randrange(K) for _ in range(V)] for _ in range(K)]
    return jc.DFA.from_dense(V, dense, 0, rng.sample(range(K), rng.randint(1, K)))


def accepts_from(dfa, q, suffix):
    return bool(dfa.accept[dfa.run(suffix, state = q)])


def brute_active(dfa, n, V):
    """States reachable in exactly t tokens AND accepting some continuation of exactly n - t tokens."""
    active = []
    for t in range(n + 1):
        reach = {dfa.run(p) for p in itertools.product(range(V), repeat = t)}
        active.append({q for q in reach
                       if any(accepts_from(dfa, q, s) for s in itertools.product(range(V), repeat = n - t))})
    return active


def test_running_example():
    # n = 4, "contains 1 2"; states 0 = nothing yet, 1 = just saw 1, 2 = found (absorbing, accepting)
    dfa = jc.DFA.contains([[1, 2]], 6)
    L = build_layout(dfa, 4)
    assert L.width.tolist() == [1, 2, 3, 2, 1] and L.satisfiable
    state = lambda t, k: int(L.state_id[t, k])
    assert [state(0, 0)] == [dfa.initial]
    # boundary 2 holds states {0, 1, 2} in order; boundary 3 holds {1, 2}
    assert L.state_id[2].tolist() == [0, 1, 2] and L.state_id[3, :2].tolist() == [1, 2]
    c0, c1, c2 = (int(dfa.token_class[v]) for v in (0, 1, 2))       # other, token 1, token 2
    nxt = lambda k, c: int(L.next_col[2, k, c])
    col3 = lambda s: int(L.col_of_state[3, s])
    # state 0 with two tokens left must read "1" now; state 1 continues with 1 or finishes with 2
    assert (nxt(0, c0), nxt(0, c1), nxt(0, c2)) == (-1, col3(1), -1)
    assert (nxt(1, c0), nxt(1, c1), nxt(1, c2)) == (-1, col3(1), col3(2))
    assert {nxt(2, c0), nxt(2, c1), nxt(2, c2)} == {col3(2)}


def test_active_states_match_brute_force():
    V = 3
    for seed in range(15):
        dfa = random_dfa(5, V, seed)
        for n in (0, 1, 3, 5):
            L = build_layout(dfa, n)
            want = brute_active(dfa, n, V)
            for t in range(n + 1):
                got = {int(s) for s in L.state_id[t] if s >= 0}
                assert got == want[t], (seed, n, t)
                assert int(L.width[t]) == len(want[t])
                for s in got:                                       # inverse map is consistent
                    assert int(L.state_id[t, L.col_of_state[t, s]]) == s
            assert L.satisfiable == (len(want[0]) > 0)


def test_successors_are_consistent():
    V = 3
    for seed in range(15):
        dfa = random_dfa(5, V, seed)
        n = 5
        L = build_layout(dfa, n)
        if not L.satisfiable:
            continue
        delta = dfa.delta
        for t in range(n):
            for k in range(int(L.width[t])):
                s = int(L.state_id[t, k])
                cols = L.next_col[t, k]
                assert (cols >= 0).any(), "an active state must have an active successor"
                for c in range(dfa.num_classes):
                    succ = int(delta[s, c])
                    expect = int(L.col_of_state[t + 1, succ])
                    assert int(cols[c]) == expect                    # -1 exactly when inactive at t + 1
            assert (L.next_col[t, int(L.width[t]):] == -1).all()     # padding columns lead nowhere


def test_unsatisfiable_constraint_has_empty_boundaries():
    L = build_layout(jc.DFA.contains([[1, 2, 1]], 4), 2)
    assert not L.satisfiable and (L.width == 0).all() and (L.next_col == -1).all()


def test_layouts_are_cached_by_fingerprint():
    a = build_layout(jc.DFA.contains([[1, 2]], 6), 4)
    b = build_layout(jc.DFA.contains([[1, 2]], 6), 4)               # separately built, equal automaton
    assert a is b
    assert build_layout(jc.DFA.contains([[1, 2]], 6), 5) is not a


def test_pruning_on_a_word_count_constraint():
    """Report how much pruning narrows the columns for a Ctrl-G-style constraint (keyword AND exactly 8
    words) over a small vocabulary; the active width must stay below the automaton's state count."""
    V = 6
    kinds = torch.tensor([0, 3, 3, 3, 1, 2])          # 0 special, 1-3 words, 4 glued, 5 separator
    c = (jc.DFA.contains([[1, 2]], V) & jc.DFA.word_count(8, 8, kinds)).automaton()
    L = build_layout(c, 12)
    print(f"{c.num_states} states; active width per boundary {L.width.tolist()}")
    assert L.satisfiable and int(L.width.max()) < c.num_states
