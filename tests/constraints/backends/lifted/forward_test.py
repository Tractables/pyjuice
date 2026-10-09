"""
The lifted forward pass (`cc.marginal`) against the reference in conftest.py, on every kind of PC there:
HMMs (tied, untied, block-sparse transitions), 1-D PDs (plain, sum-sharing with block-sparse sum edges,
blockified) and hand-built circuits (three- and four-child products, a sum over an input node, permuted and
node-level product edges, an edge block listed twice, block size 1). Also: the one-state constraint gives
pyjuice's own marginal, an unsatisfiable constraint or evidence that breaks the constraint gives -inf, and
TF32 products stay close to the fp32 default.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "pd": 8, "pd_prod_dominated": 8, "pd_blockified": 8,
         "hand": 5, "hand_permuted": 5, "hand_unit": 4}                     # see conftest.py

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
