"""
The lifted forward pass (`cc.marginal`) against the reference in conftest.py, on every kind of PC there:
HMMs (tied, untied, block-sparse transitions), 1-D PDs (plain, sum-sharing with block-sparse sum edges,
blockified) and hand-built circuits (three- and four-child products, a sum over an input node, permuted and
node-level product edges, an edge block listed twice, block size 1). Also: the one-state constraint gives
pyjuice's own marginal, an unsatisfiable constraint or evidence that breaks the constraint gives -inf, and
TF32 products stay close to the fp32 default.
"""
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "pd": 8, "pd_prod_dominated": 8, "pd_blockified": 8,
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


@pytest.mark.parametrize("constraint", list(CONSTRAINTS))
@pytest.mark.parametrize("B", [1, 3, 16])
@pytest.mark.parametrize("kind", list(KINDS))
def test_marginal_matches_the_reference(kind, B, constraint, sum_path, build_pc, reference):
    n = KINDS[kind]
    cc = jc.compile(CONSTRAINTS[constraint](), build_pc(kind, n, V))
    data, missing = evidence(n, B)
    assert_close(cc.marginal(data, missing), reference.marginal(cc, data, missing), n)


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


def test_every_token_its_own_class(build_pc, reference):
    """An automaton that sends every token of a 600-token vocabulary its own way: 600 token classes, so the
    class masses take the class-order pass. The marginal still matches the reference."""
    from pyjuice.constraints.distributions import categorical
    rng = random.Random(0)
    V2, K, n = 600, 6, 6
    dfa = jc.DFA.from_dense(V2, [[rng.randrange(K) for _ in range(V2)] for _ in range(K)], 0, rng.sample(range(K), 3))
    cc = jc.compile(dfa, build_pc("hmm", n, V2))
    assert cc.num_classes > categorical.NATURAL_ORDER_MAX_CLASSES             # the class-order pass
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V2, (3, n), generator = g)
    missing = torch.rand(3, n, generator = g) < 0.5
    missing[0] = True
    assert_close(cc.marginal(data, missing), reference.marginal(cc, data, missing), n)


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


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_kernels_launch_past_the_grid_limits(kind, sum_path, build_pc, monkeypatch):
    """With CUDA's grid limits lowered to 3 programs on the first axis and 2 on the others, every kernel whose
    program count grows with the batch or the automaton is launched in several parts: the same marginal, bit for
    bit. A wide automaton gives each of them far more work than that (an HMM: input @ block; a PD: block
    @ block; hand_left: block @ input; the dense path keeps copies). The buffers are filled with NaN before each
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
            for launches in stages for form, _, steps, _, _ in launches if form == "copy"]


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
