import itertools
import random

import pytest
import torch

import pyjuice as juice
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


# -------------------------------------------------------------------------------------------------
# The PC side: build_pc_tables, read through the constrained circuit's attributes
# -------------------------------------------------------------------------------------------------

V = 3
PC_KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "pd": 8, "pd_prod_dominated": 8,
            "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4}      # see conftest.py


def rows_by_element(cc):
    """(product layer index, element row) -> (pattern, child rows, boundaries), padding removed. Element
    rows are only unique within a layer: every product layer reuses element_mars from its start."""
    out = {}
    for li, layer in enumerate(cc.product_rows):
        for pattern, (rows, child, bounds) in layer.items():
            for r, ch, bd in zip(rows.tolist(), child.tolist(), bounds.tolist()):
                assert (li, r) not in out                                  # every product node once
                k = sum(c >= 0 for c in ch)
                assert ch[k:] == [-1] * (len(ch) - k) and bd[k + 1:] == [-1] * (len(bd) - k - 1)
                out[li, r] = (pattern, ch[:k], bd[:k + 1])
    return out


@pytest.mark.parametrize("kind", list(PC_KINDS))
def test_product_rows_follow_the_edges_in_scope_order(kind, build_pc, reference):
    n = PC_KINDS[kind]
    pc = build_pc(kind, n, V)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    st = cc.structure
    product_layers = [layer for lg in pc.inner_layer_groups if lg.is_prod() for layer in lg.layers]
    layer_of = {ns: li for li, layer in enumerate(product_layers) for ns in layer.nodes}
    got = rows_by_element(cc)
    num_rows = 0
    for ns in pc.root_ns:
        if not ns.is_prod():
            continue
        (a, b), = st.node(ns).scope_runs
        order = sorted(range(len(ns.chs)), key = lambda k: st.node(ns.chs[k]).scope_runs[0][0])
        chs = [ns.chs[k] for k in order]
        child_rows = [ns.chs[k]._output_ind_range[0] + reference.product_child_index(ns, k) for k in order]
        bounds = [st.node(cs).scope_runs[0][0] for cs in chs] + [b + 1]
        pattern = ("input_suffix" if len(chs) == 2 and chs[0].is_input() and not chs[1].is_input() and b == n - 1
                   else "chain")
        for p in range(ns.num_nodes):
            pat, ch, bd = got[layer_of[ns], ns._output_ind_range[0] + p]
            assert pat == pattern and bd == bounds
            assert ch == [int(c[p]) for c in child_rows]
            num_rows += 1
    assert num_rows == len(got)


def test_hmm_products_are_input_suffix_except_the_last_position(build_pc):
    n = 6
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    got = rows_by_element(cc)
    lo, hi = cc.input_range
    for pattern, ch, bd in got.values():
        if bd == [n - 1, n]:                       # pyjuice's one-child product over the last input node
            assert pattern == "chain" and len(ch) == 1 and lo <= ch[0] < hi
        else:
            a = bd[0]
            assert pattern == "input_suffix" and bd == [a, a + 1, n]
            assert lo <= ch[0] < hi and not lo <= ch[1] < hi            # the input node first, then the suffix


def test_columns_per_sample(build_pc):
    # an HMM keeps one vector over the entry columns per node: S = the widest boundary after the first
    n = 6
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    assert cc.columns_per_sample == max(cc.width_per_boundary[1:n].tolist())
    # the hand-built circuit under "contains 1 1": widths 1, 2, 3, 3, 2, 1, and its interval nodes over
    # [1, 2] (2 x 3 columns) and the one-child product over x3 (3 x 2) are the largest blocks
    cc = jc.compile(jc.DFA.contains([[1, 1]], V), build_pc("hand", 5, V))
    assert cc.width_per_boundary.tolist() == [1, 2, 3, 3, 2, 1] and cc.columns_per_sample == 6
    # a one-state constraint keeps one column per sample everywhere
    for kind, n in PC_KINDS.items():
        assert jc.compile(jc.DFA.anything(V), build_pc(kind, n, V)).columns_per_sample == 1


def test_tables_live_on_the_pcs_device_and_with_pc_reads_the_new_pc(build_pc):
    c = jc.DFA.contains([[1, 2]], V)
    # the same structure (PD draws its edges at random, so the same seed), another device and other
    # compile options
    pa = build_pc("pd", 8, V, seed = 0)
    pb = build_pc("pd", 8, V, seed = 0, device = torch.device("cpu"), layer_sparsity_tol = 0.05)
    ca = jc.compile(c, pa)
    cb = ca.with_pc(pb)
    assert cb.layout is ca.layout and cb.automaton is ca.automaton
    assert cb.columns_per_sample == ca.columns_per_sample
    assert [sorted(layer) for layer in cb.product_rows] == [sorted(layer) for layer in ca.product_rows]
    for la, lb in zip(ca.product_rows, cb.product_rows):
        for pattern in la:
            for ta, tb in zip(la[pattern], lb[pattern]):
                assert ta.dtype == tb.dtype == torch.int32
                assert ta.device == pa.device and tb.device.type == "cpu"     # each PC's own tables
                assert torch.equal(ta.cpu(), tb)


def test_a_graph_compiled_again_is_refused():
    torch.manual_seed(0)
    ns = juice.structures.PD(data_shape = (8,), num_latents = 4, split_intervals = 1,
                             input_node_params = {"num_cats": V})
    first = juice.compile(ns, verbose = False)
    juice.compile(ns, verbose = False)            # the same graph again: its node groups now describe this one
    with pytest.raises(jc.ConstraintCompileError, match = "compiled again"):
        jc.compile(jc.DFA.contains([[1, 2]], V), first)
    # an HMM graph compiled again keeps its rows, so nothing is refused
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 6, num_latents = 4, num_emits = V)
    first = juice.compile(ns, verbose = False)
    juice.compile(ns, verbose = False)
    jc.compile(jc.DFA.contains([[1, 2]], V), first)
