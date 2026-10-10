import itertools
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
from pyjuice.constraints.backends.lifted.kernels.prod import skip_masks
from pyjuice.constraints.backends.lifted.plan import Reachability, _tile_any, build_layout


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


def to_accept(dfa, n):
    """The fewest tokens from every state to an accepting state, n + 1 if more than n."""
    acc = torch.stack([dfa.accept_within(r) for r in range(n + 1)])
    return torch.where(acc.any(dim = 0), acc.to(torch.uint8).argmax(dim = 0), n + 1)


def test_columns_are_ordered_by_distance_to_acceptance():
    for seed in range(15):
        dfa = random_dfa(6, 3, seed)
        for n in (1, 3, 6):
            L = build_layout(dfa, n)
            d = to_accept(dfa, n)
            for t in range(n + 1):
                keys = [(-int(d[s]), s) for s in L.state_id[t, :int(L.width[t])].tolist()]
                assert keys == sorted(keys), (seed, n, t)                  # farthest first, then by state id


def brute_pairs(dfa, L, a, b, V):
    """[width at a, exit width at b]: some string of b - a tokens leads column i to column j (b == n: accepts)."""
    n = L.n
    wb = 1 if b == n else int(L.width[b])
    out = torch.zeros(int(L.width[a]), wb, dtype = torch.bool)
    for i in range(int(L.width[a])):
        for tokens in itertools.product(range(V), repeat = b - a):
            e = dfa.run(list(tokens), state = int(L.state_id[a, i]))
            if b == n:
                out[i, 0] |= bool(dfa.accept[e])
            elif L.col_of_state[b, e] >= 0:
                out[i, L.col_of_state[b, e]] = True
    return out


def test_reachability_matches_brute_force():
    V, n = 3, 5
    for seed in range(10):
        dfa = random_dfa(5, V, seed)
        L = build_layout(dfa, n)
        if not L.satisfiable:
            continue
        reach = Reachability(L, tile = 2)
        for a in range(n + 1):
            for b in range(a, n + 1):
                want = brute_pairs(dfa, L, a, b, V)
                assert torch.equal(reach.pairs(a, b), want), (seed, a, b)
                assert torch.equal(reach.occupancy(a, b), _tile_any(want, 2, 2)), (seed, a, b)


def definition_masks(reach, triples, tm, tn, tk):
    """The SkipMasks bits, straight from the pairs: [triples, tiles_i, tiles_j, chunks] bool."""
    out = []
    for a, b, c in triples:
        lp, rp = reach.pairs(a, b), reach.pairs(b, c)
        I, J, C = -(-lp.size(0) // tm), -(-rp.size(1) // tn), -(-lp.size(1) // tk)
        out.append(torch.tensor([[[bool(lp[ti * tm:(ti + 1) * tm, ch * tk:(ch + 1) * tk].any()) and
                                   bool(rp[ch * tk:(ch + 1) * tk, tj * tn:(tj + 1) * tn].any())
                                   for ch in range(C)] for tj in range(J)] for ti in range(I)]).view(I, J, C))
    return out


def mask_bits(m, s, ti, tj, c):
    w = int(m.bits[((s * m.tiles_i + ti) * m.tiles_j + tj) * m.words + c // 32])
    return bool((w >> (c % 32)) & 1)


@pytest.mark.parametrize("tiles", [(1, 1, 1), (2, 2, 2), (2, 1, 3), (4, 2, 2)])
def test_skip_masks_are_exact(tiles):
    V, n = 3, 6
    for seed in range(6):
        dfa = random_dfa(6, V, seed)
        L = build_layout(dfa, n)
        if not L.satisfiable:
            continue
        reach = Reachability(L, tile = 1)
        triples = [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n + 1)]
        m = skip_masks(reach, triples, *tiles)
        for s, want in enumerate(definition_masks(reach, triples, *tiles)):
            I, J, C = want.shape
            for ti in range(m.tiles_i):
                for tj in range(m.tiles_j):
                    for c in range(32 * m.words):
                        expect = ti < I and tj < J and c < C and bool(want[ti, tj, c])
                        assert mask_bits(m, s, ti, tj, c) == expect, (seed, triples[s], ti, tj, c)


def test_larger_tiles_or_their_sub_tiles():
    """A kernel with 2x2 base tiles per output tile may OR the base masks: the result is the larger tile's mask."""
    V, n = 3, 6
    for seed in range(6):
        dfa = random_dfa(8, V, seed)
        L = build_layout(dfa, n)
        if not L.satisfiable:
            continue
        reach = Reachability(L, tile = 1)
        triples = [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n + 1)]
        base, big = skip_masks(reach, triples, 1, 1, 1), skip_masks(reach, triples, 2, 2, 1)
        for s in range(len(triples)):
            for ti in range(big.tiles_i):
                for tj in range(big.tiles_j):
                    for c in range(32 * big.words):
                        ored = any(mask_bits(base, s, 2 * ti + di, 2 * tj + dj, c)
                                   for di in range(2) for dj in range(2)
                                   if 2 * ti + di < base.tiles_i and 2 * tj + dj < base.tiles_j)
                        assert mask_bits(big, s, ti, tj, c) == ored, (seed, s, ti, tj, c)


def test_skip_masks_span_several_words():
    """A boundary with more than 32 chunks (here 1-column chunks of a wide automaton) needs several words."""
    V, n = 4, 5
    dfa = random_dfa(120, V, 3)
    L = build_layout(dfa, n)
    assert int(L.width.max()) > 64
    reach = Reachability(L, tile = 1)
    triples = [(1, 3, 5), (0, 2, 4), (2, 3, 4)]
    m = skip_masks(reach, triples, 8, 8, 1)
    assert m.words >= 2
    for s, want in enumerate(definition_masks(reach, triples, 8, 8, 1)):
        I, J, C = want.shape
        for ti in range(I):
            for tj in range(J):
                assert [mask_bits(m, s, ti, tj, c) for c in range(C)] == want[ti, tj].tolist()


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
            "pd_blockified": 8, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5}      # see conftest.py


def rows_by_element(cc):
    """(product layer index, element row) -> (child rows, boundaries), padding removed. Element rows are only
    unique within a layer: every product layer reuses element_mars from its start."""
    out = {}
    for li, (rows, child, bounds) in enumerate(cc.product_rows):
        for r, ch, bd in zip(rows.tolist(), child.tolist(), bounds.tolist()):
            assert (li, r) not in out                                      # every product node once
            k = sum(c >= 0 for c in ch)
            assert ch[k:] == [-1] * (len(ch) - k) and bd[k + 1:] == [-1] * (len(bd) - k - 1)
            out[li, r] = (ch[:k], bd[:k + 1])
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
        for p in range(ns.num_nodes):
            ch, bd = got[layer_of[ns], ns._output_ind_range[0] + p]
            assert bd == bounds
            assert ch == [int(c[p]) for c in child_rows]
            num_rows += 1
    assert num_rows == len(got)


def test_hmm_products_are_an_input_then_a_suffix(build_pc):
    n = 6
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", n, V))
    got = rows_by_element(cc)
    lo, hi = cc.input_range
    for ch, bd in got.values():
        if bd == [n - 1, n]:                       # pyjuice's one-child product over the last input node
            assert len(ch) == 1 and lo <= ch[0] < hi
        else:
            a = bd[0]
            assert bd == [a, a + 1, n]
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
    assert len(cb.product_rows) == len(ca.product_rows)
    for la, lb in zip(ca.product_rows, cb.product_rows):
        for ta, tb in zip(la, lb):
            assert ta.dtype == tb.dtype == torch.int32
            assert ta.device == pa.device and tb.device.type == "cpu"         # each PC's own tables
            assert torch.equal(ta.cpu(), tb)


@pytest.mark.parametrize("kind", ["pd", "hand", "hand_left", "hmm"])
def test_every_block_block_step_indexes_its_triple(kind, build_pc):
    n = PC_KINDS[kind]
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V))
    prog = cc._lifted_program()
    seen = 0
    for k, _, layers in prog.steps:
        if k != "prod":
            continue
        for stages, _, _ in layers:
            for launches in stages:
                for form, _, steps, _, _ in launches:
                    if form == "block_block":
                        tri = [prog.skip_triples[i] for i in steps[3].tolist()]
                        assert tri == [tuple(r) for r in steps[0][:, 5:8].tolist()]
                        seen += 1
    m = prog.skip
    assert m.bits.numel() == max(1, len(prog.skip_triples) * m.tiles_i * m.tiles_j * m.words)
    assert (seen > 0) == (len(prog.skip_triples) > 0)


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
