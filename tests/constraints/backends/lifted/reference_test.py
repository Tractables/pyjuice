"""
Validates the test-side reference for constrained marginals (the `reference` fixture in conftest.py),
which the library's GPU implementation is tested against. Like the compiler, the reference handles
Categorical input nodes only, so every PC here has Categorical input nodes. Checks:

* its sum weights are normalized for every node, and equal the node groups' own after
  `pc.update_parameters()` (where their edge blocks are distinct);
* with the one-state constraint (anything goes) it reproduces `juice.queries.marginal` on every kind of
  PC in conftest.py's `PC_KINDS`: HMMs (tied, untied, with block-sparse transitions), 1-D PDs (plain,
  sum-sharing with block-sparse sum edges, blockified), and hand-built circuits with three- and four-child
  products, a sum over an input node, an input node next to an interval node, node-level and permuted
  block-level product edges, explicit block-sparse sum edges and block size 1;
* under real constraints it equals brute force -- the PC's own probabilities of every string, summed
  over the accepted ones consistent with the evidence -- including after an in-place parameter update;
* its log-space block operations stay exact far outside exp's range.
"""
import itertools
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc

DEV = torch.device("cuda:0")


# kind (see `PC_KINDS` in conftest.py) -> n, for the identity tests and for brute force
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "pd": 8, "pd_prod_dominated": 8, "pd_blockified": 8,
         "hand": 5, "hand_permuted": 5, "hand_unit": 4}
TINY = {"hmm": 5, "hmm_untied": 5, "hmm_block_sparse": 5, "pd": 4, "pd_prod_dominated": 5, "pd_blockified": 5,
        "hand": 5, "hand_permuted": 5, "hand_unit": 4}


def evidence(n, V, B = 6, seed = 0):
    """Sample 0 has everything missing (p(C) alone), sample 1 everything observed, the rest random."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.5
    missing[0], missing[1] = True, False
    return data, missing


@pytest.mark.parametrize("kind", list(KINDS))
def test_parameters_are_read_from_the_compiled_pc(kind, reference, build_pc):
    pc = build_pc(kind, KINDS[kind], V = 5)
    pc.update_parameters()                                              # writes the node groups' own copies
    for ns in pc.root_ns:
        src = ns.get_source_ns()
        if ns.is_sum():
            weights = reference.sum_weights(pc, ns).cpu()
            assert torch.allclose(weights.sum(dim = 1), torch.ones(ns.num_nodes))     # every node normalized
            # `get_params(as_matrix = True)` keeps one copy of a repeated edge block, so it is the expected
            # value only where the edge blocks are distinct
            if torch.unique(ns.edge_ids, dim = 1).size(1) == ns.edge_ids.size(1):
                assert torch.equal(weights, src.get_params(as_matrix = True))
        elif ns.is_input():
            assert torch.equal(reference.categorical_probs(pc, ns).cpu(), src._params.reshape(ns.num_nodes, -1))


@pytest.mark.parametrize("kind", list(KINDS))
def test_one_state_constraint_reproduces_the_marginal(kind, reference, build_pc):
    n, V = KINDS[kind], 5
    pc = build_pc(kind, n, V)
    cc = jc.compile(jc.DFA.anything(V), pc)
    data, missing = evidence(n, V)
    for mask in (None, missing, missing[2]):                            # no mask, [B, n], [n]
        want = juice.queries.marginal(pc, data = data.to(DEV), missing_mask = None if mask is None else mask.to(DEV))
        got = reference.marginal(cc, data, mask)
        assert got.shape == want.shape and got.dtype == torch.float32
        assert (got - want).abs().max() < 1e-5, (kind, mask)


def test_log_space_block_operations_are_stable(reference):
    """Far outside exp's range (and with -inf entries) the float32 block operations equal a direct
    float64 logsumexp, up to float32 rounding of values in the thousands."""
    g = torch.Generator().manual_seed(0)
    A = torch.rand(3, 2, 4, 5, generator = g) * 10 - 3000
    B = torch.rand(3, 2, 5, 6, generator = g) * 10 + 2000
    A[0, 0, 1] = -float("inf"); B[1, 1, :, 2] = -float("inf"); A[2, 0, 0, :2] = -float("inf")
    want = torch.logsumexp(A.double()[..., :, :, None] + B.double()[..., None, :, :], dim = -2)
    got = reference.log_matmul(A, B)
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)) and not torch.isnan(got).any()
    fin = torch.isfinite(want)
    assert torch.allclose(got[fin].double(), want[fin], rtol = 1e-6, atol = 0)

    Wt = torch.rand(7, 3, generator = g)
    Wt[0, 1] = 0.0
    X = torch.rand(3, 2, 4, 5, generator = g) * 10 - 3000
    X[:, 1, 2] = -float("inf")                                                         # every child -inf
    want = torch.logsumexp(torch.log(Wt.double())[:, :, None, None, None] + X.double()[None], dim = 1)
    got = reference.log_weighted_sum(Wt, X)
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)) and not torch.isnan(got).any()
    fin = torch.isfinite(want)
    assert torch.allclose(got[fin].double(), want[fin], rtol = 1e-6, atol = 0)


def test_long_sequences_do_not_underflow(reference):
    """log p(x) of a 300-token sequence is far below exp's range (about -745)."""
    n, V = 300, 50
    torch.manual_seed(0)
    pc = juice.compile(juice.structures.HMM(seq_length = n, num_latents = 8, num_emits = V), verbose = False).to(DEV)
    cc = jc.compile(jc.DFA.anything(V), pc)
    data, missing = evidence(n, V, B = 3)
    want = juice.queries.marginal(pc, data = data.to(DEV), missing_mask = missing.to(DEV))
    got = reference.marginal(cc, data, missing)
    assert want[1, 0] < -745                                                            # fully observed sample
    assert ((got - want).abs() <= 1e-5 * want.abs().clamp(min = 1)).all()               # float32, 300 steps


def random_dfa(V, K, seed):
    rng = random.Random(seed)
    dense = [[rng.randrange(K) for _ in range(V)] for _ in range(K)]
    return jc.DFA.from_dense(V, dense, 0, rng.sample(range(K), rng.randint(1, K - 1)))


def constraints(V, n):
    return {
        "keyword": jc.DFA.contains([[1, 2]], V),
        "ordered_keywords": jc.DFA.contains([[2]], V).concat(jc.DFA.contains([[1, 1]], V)),
        "random": random_dfa(V, 4, seed = 3),
        "negation": ~jc.DFA.contains([[0, 0]], V),
        "and_with_random": jc.DFA.contains([[2, 0]], V) & random_dfa(V, 3, seed = 4),
        "unsatisfiable": jc.DFA.contains([[1] * (n + 1)], V),
    }


def brute_force(pc, cc, data, missing, V):
    """log sum over every accepted string consistent with the evidence of the PC's own p(x)."""
    n = cc.n
    X = torch.tensor(list(itertools.product(range(V), repeat = n)), device = DEV)
    lls = pc(X)[:, 0].double()
    dfa = cc.automaton
    q = torch.full((X.size(0),), dfa.initial, device = DEV)
    delta, cls = dfa.delta.to(DEV), dfa.token_class.to(DEV)
    for t in range(n):
        q = delta[q, cls[X[:, t]]]
    accepted = dfa.accept.to(DEV)[q]
    out = []
    for b in range(data.size(0)):
        ok = accepted & ((X == data[b].to(DEV)) | missing[b].to(DEV)).all(dim = 1)
        out.append(torch.logsumexp(lls[ok], dim = 0) if ok.any() else torch.tensor(-float("inf"), device = DEV,
                                                                                    dtype = torch.float64))
    return torch.stack(out)


@pytest.mark.parametrize("name", list(constraints(3, 5)))
@pytest.mark.parametrize("kind", list(TINY))
def test_reference_equals_brute_force(kind, name, reference, build_pc):
    n, V = TINY[kind], 3
    pc = build_pc(kind, n, V)
    c = constraints(V, n)[name]
    cc = jc.compile(c, pc)
    data, missing = evidence(n, V)
    got = reference.marginal(cc, data, missing)[:, 0]
    want = brute_force(pc, cc, data, missing, V)
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)), (got, want)
    fin = torch.isfinite(want)
    assert (got[fin] - want[fin]).abs().max() < 1e-5 if fin.any() else name == "unsatisfiable"


def test_reference_follows_in_place_parameter_updates(reference, build_pc):
    """An in-place update, as training does between steps without recompiling: copy in the (normalized)
    parameters of the same structure built with another seed."""
    n, V = 5, 3
    pc, other = build_pc("hand", n, V, seed = 0), build_pc("hand", n, V, seed = 1)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    data, missing = evidence(n, V)
    before = reference.marginal(cc, data, missing)[:, 0]
    with torch.no_grad():
        pc.params.copy_(other.params)
        for layer, src in zip(pc.input_layer_group, other.input_layer_group):
            layer.params.copy_(src.params)
    after = reference.marginal(cc, data, missing)[:, 0]
    want = brute_force(pc, cc, data, missing, V)
    fin = torch.isfinite(want)                                          # some samples contradict the constraint
    assert torch.equal(torch.isneginf(after), ~fin) and torch.equal(torch.isneginf(before), ~fin)
    assert (after[fin] - before[fin]).abs().max() > 1e-2                # the update matters ...
    assert (after[fin] - want[fin]).abs().max() < 1e-5                  # ... and the reference follows it
