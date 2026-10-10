"""
The lifted forward pass (`cc.marginal`) against the reference in conftest.py, on every kind of PC there:
HMMs (tied, untied, block-sparse transitions), their left-linear mirror, 1-D PDs (plain, sum-sharing with block-sparse sum edges,
blockified) and hand-built circuits (three- and four-child products, a sum over an input node, permuted and
node-level product edges, an edge block listed twice, block size 1). Also: the one-state constraint gives
pyjuice's own marginal, an unsatisfiable constraint or evidence that breaks the constraint gives -inf, and
TF32 products stay close to the fp32 default.
"""
import math
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 8, "pd_prod_dominated": 8, "pd_blockified": 8,
         "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5}                     # see conftest.py

CONSTRAINTS = {
    "contains": lambda: jc.DFA.contains([[1, 2]], V),
    "contains_either_and_not": lambda: jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)),
}


def evidence(n, B, seed = 0):
    """Row 0 all missing (p(C) alone), row 1 all observed, the rest random."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.5
    missing[0] = True
    if B > 1:
        missing[1] = False
    return data, missing


def assert_close(got, want, n):
    assert got.shape == want.shape
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert torch.isfinite(got[fin]).all()
    tol = 1e-5 * want[fin].abs() + n * 1e-6
    assert ((got[fin] - want[fin]).abs() <= tol).all(), (got[fin] - want[fin]).abs().max()


@pytest.fixture(params = ["fused", "dense"])
def sum_path(request, monkeypatch):
    """Run the sum layers' node blocks through the fused kernel only, or through the dense (cuBLAS) path wherever
    they qualify -- the test circuits' blocks are far below the size the dense path is used from by default."""
    from pyjuice.constraints.backends.lifted.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 if request.param == "dense" else 1 << 30)
    return request.param


@pytest.fixture(params = ["classes", "grouped"])
def transitions(request, monkeypatch):
    """A missing token's transitions one class at a time, or grouped by successor through transition masses (the
    test automata have far fewer classes than grouping is used from by default)."""
    import pyjuice.constraints.backends.lifted.forward as F
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 1 << 30 if request.param == "classes" else 0)
    return request.param


@pytest.mark.parametrize("constraint", list(CONSTRAINTS))
@pytest.mark.parametrize("B", [1, 3, 16])
@pytest.mark.parametrize("kind", list(KINDS))
def test_marginal_matches_the_reference(kind, B, constraint, sum_path, transitions, build_pc, reference):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[constraint](), build_pc(kind, n, V))
    data, missing = evidence(n, B)
    assert_close(cc.marginal(data, missing), reference.marginal(cc, data, missing), n)


@pytest.mark.parametrize("kind", ["pd", "pd_prod_dominated", "pd_blockified", "hand"])
def test_a_wide_automaton_matches_the_reference(kind, sum_path, build_pc, reference):
    """A 48-state random automaton: interval nodes keep only the pairs of columns it joins -- a fraction of them on
    some intervals -- and the marginal matches the reference, which keeps every pair."""
    rng = random.Random(0)
    K, n = 48, KINDS[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    data, missing = evidence(n, 4)
    got = cc.marginal(data, missing)
    assert (cc._lifted_program().intervals.info[:, 1] >= 0).any()           # some interval keeps a fraction
    assert_close(got, reference.marginal(cc, data, missing), n)


@pytest.mark.parametrize("kind", list(KINDS))
def test_the_one_state_constraint_gives_pyjuices_marginal(kind, build_pc):
    n = KINDS[kind]
    pc = build_pc(kind, n, V)
    data, missing = evidence(n, 8, seed = 1)
    want = juice.queries.marginal(pc, data = data.cuda(), missing_mask = missing.cuda())
    cc = jc.compile(jc.DFA.anything(V), pc)
    assert_close(cc.marginal(data, missing), want.float(), n)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand"])
def test_missing_mask_forms(kind, build_pc):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, _ = evidence(n, 4)
    shared = torch.rand(n) < 0.5
    assert torch.equal(cc.marginal(data, shared), cc.marginal(data, shared[None].expand(4, n)))
    assert torch.equal(cc.marginal(data), cc.marginal(data, torch.zeros(4, n, dtype = torch.bool)))
    with pytest.raises(ValueError, match = "missing_mask"):
        cc.marginal(data, torch.zeros(4, n + 1, dtype = torch.bool))


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_unit"])
def test_impossible_is_minus_inf(kind, build_pc):
    n = KINDS[kind]
    pc = build_pc(kind, n, V)
    data, missing = evidence(n, 4)
    assert torch.isneginf(jc.compile(jc.DFA.nothing(V), pc).marginal(data, missing)).all()
    too_long = jc.DFA.contains([[1] * (n + 1)], V)                        # cannot fit in n tokens
    assert torch.isneginf(jc.compile(too_long, pc).marginal(data, missing)).all()
    cc = jc.compile(jc.Not(jc.DFA.contains([[1]], V)), pc)                 # no token 1 anywhere
    data[:, 0], missing[:, 0] = 1, False                                   # ... but every row observed one
    assert torch.isneginf(cc.marginal(data, missing)).all()


@pytest.mark.parametrize("kind", ["hmm_block_sparse", "pd_prod_dominated", "hand_permuted"])
def test_tf32_products_stay_close(kind, sum_path, build_pc):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence(n, 16)
    fp32, tf32 = cc.marginal(data, missing), cc.marginal(data, missing, precision = "tf32")
    fin = torch.isfinite(fp32)
    assert torch.equal(fin, torch.isfinite(tf32))
    assert ((fp32 - tf32)[fin].abs() <= 1e-2).all()
    with pytest.raises(ValueError, match = "precision"):
        cc.marginal(data, missing, precision = "bf16")


def sum_blocks(cc, B, data, missing):
    """Run the marginal; every sum region's rows, B * slots columns (cloned: the buffers are reused)."""
    cc.marginal(data, missing)
    bufs = cc._buffers(B)
    lay, nm = bufs["layout"], bufs["node_mars"]
    out = []
    for r, (first, end, slots) in enumerate(cc.sum_regions):
        off, wid = lay["sum_offsets"][r], lay["sum_widths"][r]
        out.append(nm[off:off + (end - first) * wid].view(end - first, wid)[:, :B * slots].clone())
    return out


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand", "hand_left"])
def test_every_sample_is_a_contiguous_block(kind, build_pc):
    """Sample b's block of every sum node is columns [b * slots, (b + 1) * slots) of its row: what a batch of one
    puts in columns [0, slots)."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence(n, 3)
    batched = sum_blocks(cc, 3, data, missing)
    for b in range(3):
        single = sum_blocks(cc, 1, data[b:b + 1], missing[b:b + 1])
        for (first, end, slots), rb, rs in zip(cc.sum_regions, batched, single):
            got = rb[:, b * slots:(b + 1) * slots]
            assert torch.equal(torch.isneginf(got), torch.isneginf(rs))
            fin = torch.isfinite(rs)
            assert torch.allclose(got[fin], rs[fin], rtol = 1e-6, atol = 1e-6), (kind, b, first)


def test_skipping_empty_tiles_changes_nothing(build_pc, monkeypatch):
    """A wide automaton, so that blocks span several tiles and some (output tile, shared chunk) pairs cannot be
    joined: skipping them gives what computing them gives."""
    from pyjuice.constraints.backends.lifted.kernels import prod
    from pyjuice.constraints.backends.lifted.plan import Reachability
    rng = random.Random(0)
    K, n = 48, 8
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc("pd", n, V))
    prog = cc._lifted_program()
    assert prog.skip.tiles_i > 1                                         # blocks wider than one tile
    reach = Reachability(cc.layout)
    kept = total = 0
    for a, b, c in prog.skip_triples:
        lt, rt = reach.occupancy(a, b).float(), reach.occupancy(b, c).float()
        kept += float((lt @ rt).sum())
        total += lt.size(0) * lt.size(1) * rt.size(1)
    assert kept < total                                                  # something is skipped
    data, missing = evidence(n, 4)
    with_skip = cc.marginal(data, missing)
    monkeypatch.setattr(prod, "SKIP_EMPTY_TILES", False)
    assert_close(with_skip, cc.marginal(data, missing), n)


@pytest.mark.parametrize("kind", ["hmm", "left_linear", "hand_left"])
def test_every_token_its_own_class(kind, transitions, build_pc, reference):
    """An automaton that sends every token of a 600-token vocabulary its own way: 600 token classes, so the
    class masses take the class-order pass and every column has far fewer successors than classes (an HMM reads
    its inputs before a block, hand_left after one), one class at a time or grouped. The marginal still matches
    the reference."""
    from pyjuice.constraints.distributions import categorical
    rng = random.Random(0)
    V2, K, n = 600, 6, KINDS[kind]
    dfa = jc.DFA.from_dense(V2, [[rng.randrange(K) for _ in range(V2)] for _ in range(K)], 0, rng.sample(range(K), 3))
    cc = jc.compile(dfa, build_pc(kind, n, V2))
    assert cc.num_classes > categorical.NATURAL_ORDER_MAX_CLASSES             # the class-order pass
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V2, (3, n), generator = g)
    missing = torch.rand(3, n, generator = g) < 0.5
    missing[0] = True
    assert_close(cc.marginal(data, missing), reference.marginal(cc, data, missing), n)


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_fast_inference_keeps_the_class_masses(kind, build_pc, reference, monkeypatch):
    """Inside pyjuice.fast_inference (parameters do not change) the class masses are computed by the first query
    and kept, through a nested scope, until the outer scope exits. Outside one every query computes them, so an
    in-place parameter update -- here after the scope -- is read."""
    from pyjuice.constraints.distributions import categorical
    n = KINDS[kind]
    pc, other = build_pc(kind, n, V, seed = 0), build_pc(kind, n, V, seed = 1)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    data, missing = evidence(n, 4)
    import pyjuice.constraints.backends.lifted.forward as F
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 0)                       # transitions grouped
    calls, computed = [], categorical.class_masses
    monkeypatch.setattr(categorical, "class_masses", lambda *a, **k: calls.append(1) or computed(*a, **k))
    builds, built = [], F.transition_masses                              # the transition masses follow them
    monkeypatch.setattr(F, "transition_masses", lambda *a, **k: builds.append(1) or built(*a, **k))
    per_query = len(pc.input_layer_group)                                # one call per input layer

    want = reference.marginal(cc, data, missing)
    for _ in range(2):
        assert_close(cc.marginal(data, missing), want, n)
    assert len(calls) == 2 * per_query and len(builds) == 2              # outside a scope: every query
    with juice.fast_inference():
        for _ in range(2):
            assert_close(cc.marginal(data, missing), want, n)
        with juice.fast_inference():
            assert_close(cc.marginal(data, missing), want, n)
        assert_close(cc.marginal(data, missing), want, n)
    assert len(calls) == 3 * per_query and len(builds) == 3              # inside: the first query only

    with torch.no_grad():                                                # an in-place update after the scope
        pc.params.copy_(other.params)
        for layer, src in zip(pc.input_layer_group, other.input_layer_group):
            layer.params.copy_(src.params)
    updated = reference.marginal(cc, data, missing)
    assert (updated - want)[torch.isfinite(want)].abs().max() > 1e-2
    assert_close(cc.marginal(data, missing), updated, n)
    assert len(calls) == 4 * per_query and len(builds) == 4


def blocks_hmm(n, tied, seed):
    """An HMM whose 128 latents are two node blocks of 64 per position (dense at the library's own threshold): every
    transition is a run of two node blocks over the same children."""
    torch.manual_seed(seed)
    ns = juice.structures.HMM(seq_length = n, num_latents = 128, num_emits = V, homogeneous = tied, block_size = 64)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to("cuda:0")


def stack_keys(cc):
    """The stack key of every dense run the forward multiplies (None where its node blocks cannot be stacked)."""
    return [key for kind, _, layers in cc._lifted_program().steps if kind == "sum" for _, parts in layers
            for dense, _, _ in parts for _, key in dense.values()]


def spy_dense_sums(monkeypatch):
    """Record, per dense_sum call of the forward, whether it got stacked weights and live columns; and count the
    stacked weights built."""
    import pyjuice.constraints.backends.lifted.forward as F
    calls, builds = [], []
    dense_sum, stacked_weights = F.dense_sum, F.stacked_weights
    monkeypatch.setattr(F, "dense_sum", lambda *a, **k: calls.append((k["stacked"] is not None, k["live"] is not None))
                        or dense_sum(*a, **k))
    monkeypatch.setattr(F, "stacked_weights", lambda *a: builds.append(1) or stacked_weights(*a))
    return calls, builds


@pytest.mark.parametrize("dense_pruning", [False, True])
@pytest.mark.parametrize("tied", [True, False])
def test_stacked_dense_weights(tied, dense_pruning, reference, monkeypatch):
    """Dense node blocks over the same children multiply their stacked weights in one product. Outside
    pyjuice.fast_inference a query stacks only the weights it reads more than once (an HMM's tied transitions),
    copying each once; inside a scope every run is stacked, by the first query, and kept until the scope exits --
    an in-place parameter update after it is read. The marginal matches the reference throughout, on all columns
    and on the live ones only (evidence pruning)."""
    import pyjuice.constraints.backends.lifted.forward as F
    if dense_pruning:
        monkeypatch.setattr(F, "DENSE_PRUNE_MIN_COLUMNS", 0)              # the live-column path at these sizes
    n = 6
    pc, other = blocks_hmm(n, tied, seed = 0), blocks_hmm(n, tied, seed = 1)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    data, missing = evidence(n, 4)
    keys = [key for key in stack_keys(cc) if key is not None]
    assert len(keys) == n - 1 and len(set(keys)) == (1 if tied else n - 1)       # one run per transition
    calls, builds = spy_dense_sums(monkeypatch)
    stacked = lambda: sum(s for s, _ in calls)

    want = reference.marginal(cc, data, missing)
    for _ in range(2):
        assert_close(cc.marginal(data, missing), want, n)
    assert any(p for _, p in calls) == dense_pruning
    assert stacked() == (2 * len(keys) if tied else 0) and len(builds) == (2 if tied else 0)
    calls.clear(), builds.clear()
    with juice.fast_inference():
        for _ in range(2):
            assert_close(cc.marginal(data, missing), want, n)
    assert stacked() == 2 * len(keys) and len(builds) == len(set(keys))      # built by the first query only
    assert cc._kept_stacks is None                                            # and dropped when the scope exits

    calls.clear(), builds.clear()
    with torch.no_grad():                                                     # an in-place update after the scope
        pc.params.copy_(other.params)
        for layer, src in zip(pc.input_layer_group, other.input_layer_group):
            layer.params.copy_(src.params)
    updated = reference.marginal(cc, data, missing)
    assert (updated - want)[torch.isfinite(want)].abs().max() > 1e-2
    assert_close(cc.marginal(data, missing), updated, n)
    assert len(builds) == (1 if tied else 0)


@pytest.mark.parametrize("dense_pruning", [False, True])
def test_wide_products_are_not_stacked(dense_pruning, reference, monkeypatch):
    """A product wider than STACK_MAX_COLUMNS (here every one) runs one product per block, inside a
    pyjuice.fast_inference scope too, and copies no weights."""
    import pyjuice.constraints.backends.lifted.forward as F
    if dense_pruning:
        monkeypatch.setattr(F, "DENSE_PRUNE_MIN_COLUMNS", 0)
    monkeypatch.setattr(F, "STACK_MAX_COLUMNS", {"fp32": -1, "tf32": -1})
    n = 6
    cc = jc.compile(CONSTRAINTS["contains"](), blocks_hmm(n, tied = True, seed = 0))
    data, missing = evidence(n, 4)
    calls, builds = spy_dense_sums(monkeypatch)
    want = reference.marginal(cc, data, missing)
    assert_close(cc.marginal(data, missing), want, n)
    with juice.fast_inference():
        assert_close(cc.marginal(data, missing), want, n)
    assert calls and not any(s for s, _ in calls) and not builds


@pytest.mark.parametrize("dense_pruning", [False, True])
def test_the_stacked_weights_are_what_is_multiplied(dense_pruning, monkeypatch):
    """Doubling every stacked copy (and nothing else) doubles the sums of every stacked run: every path of the HMM
    crosses each transition once, so the marginal moves by log 2 per run -- the product reads the copy."""
    import pyjuice.constraints.backends.lifted.forward as F
    if dense_pruning:
        monkeypatch.setattr(F, "DENSE_PRUNE_MIN_COLUMNS", 0)
    n = 6
    cc = jc.compile(CONSTRAINTS["contains"](), blocks_hmm(n, tied = True, seed = 0))
    data, missing = evidence(n, 4)
    want = cc.marginal(data, missing)
    stacked_weights = F.stacked_weights
    monkeypatch.setattr(F, "stacked_weights", lambda *a: 2 * stacked_weights(*a))
    runs = sum(key is not None for key in stack_keys(cc))
    assert_close(cc.marginal(data, missing), want + runs * math.log(2), n)


def test_transition_masses_match_brute_force(build_pc, monkeypatch):
    """Every transition mass -- log sum of the class masses that lead from a column to a successor -- against
    float64, on an automaton whose columns have more successors than one pair tile takes and more classes than one
    chunk, with token probabilities down to e^-100 and one emission row below e^-90 altogether (pairs whose class
    masses are all below the fp32 normal range)."""
    import pyjuice.constraints.backends.lifted.forward as F
    from pyjuice.constraints.backends.lifted.kernels import prod
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 0)
    rng = random.Random(0)
    V2, K, n = 600, 48, 6
    dfa = jc.DFA.from_dense(V2, [[rng.randrange(K) for _ in range(V2)] for _ in range(K)], 0, rng.sample(range(K), 12))
    pc = build_pc("hmm", n, V2)
    g = torch.Generator(device = pc.params.device).manual_seed(0)
    with torch.no_grad():
        for layer in pc.input_layer_group:
            layer.params.copy_(torch.exp(-torch.rand(layer.params.shape, device = pc.params.device, generator = g) * 100))
            row = layer.s_pids[0]
            layer.params[row:row + V2] = torch.exp(-90 - torch.rand(V2, device = pc.params.device, generator = g))
    cc = jc.compile(dfa, pc)
    cc.marginal(torch.zeros(1, n, dtype = torch.long), torch.ones(1, n, dtype = torch.bool))
    prog = cc._lifted_program()
    tt, width, nc = prog.trans, prog.width, cc.layout.next_col
    ptr, succ = tt.entry_ptr.long().cpu(), tt.entry_succ.long().cpu()
    assert tt.grouped and prog.trans_persistent and tt.max_pairs > 16    # several pair tiles at the narrowest
    assert nc.size(2) > prod.TRANSITION_TILES["TC"]
    T, cm = prog.transition_buffer().double().cpu(), cc._class_mars.double().cpu()
    mass_row = prog.mass_row.long().cpu()
    assert (cm[torch.isfinite(cm)] < -87.4).any()                       # class masses below the normal range
    assert (cm.max(dim = 1).values < -87.4).any()                       # ... every class of a row
    for u, t, off in prog.trans_job[0].long().cpu().tolist():
        for q in range(width[t]):
            to = nc[t, q].long()
            for g in range(int(ptr[t, q]), int(ptr[t, q + 1])):
                classes = (to >= 0) if t + 1 == n else (to == succ[g])
                want = torch.logsumexp(cm[mass_row[u - prog.input_start], classes], dim = 0)
                got = T[off + g - int(ptr[t, 0])]
                assert abs(float(got - want)) < 1e-5, (u, t, q, g, float(got), float(want))


@pytest.mark.parametrize("kind", ["hmm", "left_linear", "pd", "hand_left"])
def test_chunked_transition_masses_change_nothing(kind, build_pc, monkeypatch):
    """Transitions grouped, with a budget of one float: every input step's transition masses are built on their own,
    just before the step, into one shared scratch -- the marginal of keeping them all, bit for bit (a wide
    automaton; the buffers filled with NaN before each call)."""
    import pyjuice.constraints.backends.lifted.forward as F
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 0)
    rng = random.Random(0)
    K, n = 48, KINDS[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    data, missing = evidence(n, 4)

    def marginal(budget):
        monkeypatch.setattr(F, "TRANSITION_BUDGET", budget)
        cc._program = None
        bufs = cc._buffers(data.size(0))
        bufs["node_mars"].fill_(float("nan"))
        bufs["element_mars"].fill_(float("nan"))
        out = cc.marginal(data, missing)
        assert cc._lifted_program().trans_persistent == (budget > 1)
        return out

    assert torch.equal(marginal(1), marginal(1 << 26))


def test_splitting_block_block_tiles_changes_nothing(build_pc, monkeypatch):
    """Spreading every (step, sample)'s output tiles over as many programs as it has tiles gives what one program
    per (step, sample) gives, bit for bit (a wide automaton: blocks of several tiles). The buffers are filled with
    NaN before each call, so a tile no program writes shows."""
    from pyjuice.constraints.backends.lifted.kernels import prod
    rng = random.Random(0)
    K, n = 48, 8
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc("pd", n, V))
    assert cc._lifted_program().skip.tiles_i > 1                          # blocks wider than one tile
    data, missing = evidence(n, 4)

    def marginal(per_sm):
        monkeypatch.setattr(prod, "BLOCK_BLOCK_PROGRAMS_PER_SM", per_sm)
        bufs = cc._buffers(data.size(0))
        bufs["node_mars"].fill_(float("nan"))
        bufs["element_mars"].fill_(float("nan"))
        return cc.marginal(data, missing)

    assert torch.equal(marginal(1 << 20), marginal(0))


def input_block_launches(cc):
    """Every ``input @ block`` launch: ``(kinds, steps, groups)``."""
    return [(kinds, steps, groups) for kind, _, layers in cc._lifted_program().steps if kind == "prod"
            for stages, _, _ in layers for launches in stages
            for _, kinds, (steps, _, groups), *_ in (l for l in launches if l[0] == "input_block")]


def input_block_groups(cc):
    """``(steps [S, 7], firsts)`` of every ``input @ block`` launch whose steps run in groups."""
    return [(steps.cpu(), groups[1].cpu()) for _, steps, groups in input_block_launches(cc) if groups is not None]


class CountingKernel:
    """A Triton kernel that counts its launches."""

    def __init__(self, kernel):
        self.kernel, self.launches = kernel, 0

    def __getitem__(self, grid):
        self.launches += 1
        return self.kernel[grid]


@pytest.mark.parametrize("G", [3, 8])
@pytest.mark.parametrize("B", [1, 3, 16])
@pytest.mark.parametrize("kind", list(KINDS))
def test_grouped_input_steps_change_nothing(kind, B, G, build_pc, monkeypatch):
    """Running ``input @ block`` steps G to a program (the class-by-class path) gives what one step per program gives,
    bit for bit, with evidence (pruned) and without. Every launch whose block is not the identity runs in groups
    (the PDs and the left-linear chain have none), here at any width. The buffers are filled with NaN before each
    call, so an output no group writes shows."""
    import pyjuice.constraints.backends.lifted.forward as F
    from pyjuice.constraints.backends.lifted.kernels import prod
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 1 << 30)                       # transitions class by class
    monkeypatch.setattr(prod, "GROUP_MIN_LANES", 0)                          # groups however narrow the launch
    kernel = CountingKernel(prod._input_block_group_kernel)
    monkeypatch.setattr(prod, "_input_block_group_kernel", kernel)
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence(n, B)

    def marginal(group, missing):
        monkeypatch.setattr(F, "INPUT_BLOCK_GROUP", group)
        cc._program = None                                                    # built again with this grouping
        bufs = cc._buffers(B)
        bufs["node_mars"].fill_(float("nan"))
        bufs["element_mars"].fill_(float("nan"))
        kernel.launches = 0
        out = cc.marginal(data, missing)
        blocks = sum(kinds[0] != prod.IDENTITY for kinds, _, _ in input_block_launches(cc))
        assert len(input_block_groups(cc)) == kernel.launches == (blocks if group > 1 else 0)
        return out

    for miss in (missing, torch.ones_like(missing)):
        assert torch.equal(marginal(G, miss), marginal(1, miss))


def test_narrow_launches_run_one_step_per_program(build_pc, reference, monkeypatch):
    """A launch whose steps' outputs have at most GROUP_MIN_LANES lanes runs one step per program even where the
    Program grouped its steps; one just above it runs the groups."""
    import pyjuice.constraints.backends.lifted.forward as F
    from pyjuice.constraints.backends.lifted.kernels import prod
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 1 << 30)
    kernel = CountingKernel(prod._input_block_group_kernel)
    monkeypatch.setattr(prod, "_input_block_group_kernel", kernel)
    n = KINDS["hmm"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("hmm", n, V))
    data, missing = evidence(n, 4)
    assert input_block_groups(cc)
    lanes = max(4 * l[5] for kind, _, layers in cc._lifted_program().steps if kind == "prod" for stages, _, _ in layers
                for launches in stages for l in launches if l[0] == "input_block" and l[2][2] is not None)
    want = reference.marginal(cc, data, missing)
    for limit, launched in ((lanes, False), (lanes - 1, True)):
        monkeypatch.setattr(prod, "GROUP_MIN_LANES", limit)
        kernel.launches = 0
        assert_close(cc.marginal(data, missing), want, n)
        assert (kernel.launches > 0) == launched


@pytest.mark.parametrize("kind", ["hmm", "hand", "hand_permuted"])
def test_input_steps_group_by_position(kind, build_pc, monkeypatch):
    """Groups of ``input @ block`` steps never mix positions or block ends, hold at most INPUT_BLOCK_GROUP steps,
    and cover every step once -- also where one launch holds steps at several positions (one of "hand"'s)."""
    import pyjuice.constraints.backends.lifted.forward as F
    monkeypatch.setattr(F, "GROUP_MIN_RATIO", 1 << 30)
    monkeypatch.setattr(F, "INPUT_BLOCK_GROUP", 3)
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    launches = input_block_groups(cc)
    assert launches
    mixed = False
    for steps, firsts in launches:
        assert firsts[0] == 0 and firsts[-1] == steps.size(0) and (firsts[1:] > firsts[:-1]).all()
        assert (firsts[1:] - firsts[:-1] <= 3).all()
        for a, b in zip(firsts[:-1].tolist(), firsts[1:].tolist()):
            assert len({(t, t2) for t, t2 in steps[a:b][:, [2, 5]].tolist()}) == 1
        mixed |= len({(t, t2) for t, t2 in steps[:, [2, 5]].tolist()}) > 1
    assert mixed == (kind == "hand")


@pytest.mark.parametrize("kind", ["hmm", "left_linear", "pd", "hand_left"])
def test_kernels_launch_past_the_grid_limits(kind, sum_path, build_pc, monkeypatch):
    """With CUDA's grid limits lowered to 3 programs on the first axis and 2 on the others, every kernel whose
    program count grows with the batch or the automaton is launched in several parts: the same marginal, bit for
    bit. A wide automaton gives each of them far more work than that (an HMM: input @ block; a PD: block
    @ block; left_linear and hand_left: block @ input; the dense path keeps copies). The buffers are filled with NaN before each
    call, so a slot left unwritten shows instead of keeping the previous call's value."""
    from pyjuice.constraints.backends.lifted.kernels import prod
    rng = random.Random(0)
    K, n = 48, KINDS[kind]
    dfa = jc.DFA.from_dense(V, [[rng.randrange(K) for _ in range(V)] for _ in range(K)], 0, rng.sample(range(K), 12))
    cc = jc.compile(dfa, build_pc(kind, n, V))
    data, missing = evidence(n, 4)

    def marginal():
        bufs = cc._buffers(data.size(0))
        bufs["node_mars"].fill_(float("nan"))
        bufs["element_mars"].fill_(float("nan"))
        return cc.marginal(data, missing)

    want = marginal()
    monkeypatch.setattr(prod, "MAX_GRID", (3, 2))
    assert torch.equal(marginal(), want)


def marginal_with_aliasing(cc, on, data, missing, monkeypatch):
    """The marginal with copies of sums aliased or materialized (the Program is rebuilt either way)."""
    import pyjuice.constraints.backends.lifted.forward as F
    monkeypatch.setattr(F, "ALIAS_COPIES", on)
    cc._program = None
    return cc.marginal(data, missing), cc._lifted_program()


def copy_launches(prog):
    return [steps for k, _, layers in prog.steps if k == "prod" for stages, _, _ in layers
            for launches in stages for form, _, steps, *_ in launches if form == "copy"]


@pytest.mark.parametrize("kind", list(KINDS))
def test_aliased_copies_give_the_materialized_marginal(kind, sum_path, build_pc, monkeypatch):
    """A sum layer reading a copied sum's row in place of the copy reads the same numbers (its aliased children
    are mixed after the others, so up to the accumulation order)."""
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc(kind, n, V))
    data, missing = evidence(n, 3)
    copied, _ = marginal_with_aliasing(cc, False, data, missing, monkeypatch)
    aliased, prog = marginal_with_aliasing(cc, True, data, missing, monkeypatch)
    assert_close(aliased, copied, n)
    # a dense node block reads its children as one run of rows: none of them may be aliased
    for k, (kind_, prod_index, layers) in enumerate(prog.steps[:-1]):
        if kind_ == "prod" and prod_index in prog.alias:
            alias_row = prog.alias[prod_index][0].cpu()
            first = cc.element_regions[prod_index][0]
            for _, parts in prog.steps[k + 1][2]:
                for dense, E, _ in parts:
                    for (_, c0) in dense:
                        assert (alias_row[c0 - first:c0 - first + E] < 0).all()


def test_copies_of_sums_are_not_materialized(build_pc, monkeypatch):
    """On a PD (small node blocks: the fused sum path) every copy of a sum is aliased and no copy is launched."""
    n = KINDS["pd"]
    cc = jc.compile(CONSTRAINTS["contains"](), build_pc("pd", n, V))
    data, missing = evidence(n, 2)
    _, prog = marginal_with_aliasing(cc, False, data, missing, monkeypatch)
    copies = sum(st.size(0) for st in copy_launches(prog))
    assert copies > 0 and not prog.alias
    _, prog = marginal_with_aliasing(cc, True, data, missing, monkeypatch)
    assert copy_launches(prog) == []
    assert sum(int((a >= 0).sum()) for a, _ in prog.alias.values()) == copies
