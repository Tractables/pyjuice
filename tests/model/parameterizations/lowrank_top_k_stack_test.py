"""
`LowRank(top_k)` sums' STACKED forward (`lowrank/top_k_stack.py`): a pass's operands that no other layer's values
enter, made once for every sum reading a leaf layer, and each sum's forward two launches from the stack's plan --
against the unstacked fused forward (`top_k_fused.sum_forward`, the reference of its operands): the values bit for bit
(the same device code), whatever the masks, the map of rows and the parameters' sources; the stash a backward takes
from it (the conditional's answers, the gradients: up to the backward kernels' order of adds); the settled forward
(the layer's next calls straight to the stack) taking every call of a pass it serves, and none it does not.
"""

import contextlib
import importlib.util
import os

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.parameterizations import LowRank
from pyjuice.queries import conditional

# (the top_k tests' circuit and batch, loaded by path: the suite's importlib mode puts no test directory on `sys.path`)
_spec = importlib.util.spec_from_file_location(
    "_lowrank_top_k_reference_for_stack", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowrank_top_k_test.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEV = H.DEV
T, N, C, K, R, RE, V2 = 4, 128, 70, 64, 20, 8, 80            # (the kernels' shapes: 128 nodes, top_k 64)


def _modules():
    import pyjuice.parameterizations.lowrank.top_k_fused as tkf
    import pyjuice.parameterizations.lowrank.top_k_leaf_fused as tlf
    import pyjuice.parameterizations.lowrank.top_k_stack as tks
    if tkf.top_k_module() is None or tlf.module() is None:
        pytest.skip("the CUDA modules did not build")
    return tkf, tlf, tks


def _untied_pc(tab, seed):
    """`H._hmm_pc` with every transition its own parameters (the stack's sources one a position)"""
    g = torch.Generator().manual_seed(seed)
    with juice.set_block_size(N):
        leaf = inputs(T - 1, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V2),
                      parameterization = LowRank(rank = RE, top_k = K))
        leaves, trans, zs = {T - 1: leaf}, {}, leaf
        for var in range(T - 2, -1, -1):
            xs = leaf.duplicate(var, tie_params = True)
            leaves[var] = xs
            ns = summate(zs, num_node_blocks = 1, parameterization = LowRank(rank = R, top_k = K))
            ns.set_params(torch.softmax(torch.randn(N, N, generator = g, dtype = torch.float64), -1).float())
            trans[var] = ns
            zs = multiply(xs, ns)
        root = summate(zs, num_node_blocks = 1, block_size = 1)
    root.set_params(tab["lpi"].exp().float().cpu()[None, :])
    leaf.set_params(tab["lE"].exp().float().cpu().reshape(-1))
    pc = juice.compile(root)
    pc.to(DEV)
    pc.register_external_params_group("tr", [trans[t] for t in range(T - 1)])
    pc.register_external_params_group("em", [leaves[t] for t in range(T)])
    return pc


def _case(b, masked, seed = 7, tied = True, rows = None):
    """a circuit at the kernels' shapes and a call's arguments: `rows` the map of rows (`external_params_rows`), the
    evidence of the rows' batch then"""
    tab = H._tables(N, seed, V2)
    pc = H._hmm_pc(T, N, K, R, RE, tab, V2)[0] if tied else _untied_pc(tab, seed)
    ext, ids, lp, x, miss = H._batch(b, T, N, C, K, R, RE, seed + 1, masked = masked, V = V2)
    kw = dict(external_params = H._pc_ext(ext, T, N), categorical_evidence_logp = lp.float(), soft_evidence_cat_ids = ids)
    if rows is not None:
        rows = torch.tensor(rows, device = DEV)
        x, miss = x[rows], miss[rows]
        kw["external_params_rows"] = rows
    return pc, kw, x, miss


@contextlib.contextmanager
def _spy_stacks(tks):
    """the node vectors the stacks' plans took, a list of `id(ns)` in call order"""
    taken, real = [], tks.SumStack.forward

    def forward(self, ns):
        got = real(self, ns)
        if got:
            taken.append(id(ns))
        return got

    tks.SumStack.forward = forward
    try:
        yield taken
    finally:
        tks.SumStack.forward = real


@pytest.mark.parametrize("masked", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rows", [None, [0, 1, 1, 3, 0, 2]])
@pytest.mark.parametrize("tied", [True, False])
def test_the_stacked_forward_is_the_unstacked_one(masked, rows, tied):
    """The values bit for bit -- every token observed, half of them masked, all of them; a sample its own tensors, or
    the samples of a row of tensors (and of evidence) sharing it; the transitions' parameters one source, or one a
    position -- and every transition taken by its pass's stack, the second pass's by the settled forwards."""
    tkf, tlf, tks = _modules()
    pc, kw, x, miss = _case(4 if rows else 5, masked, tied = tied, rows = rows)
    with torch.no_grad():
        tks.STACKED = False
        try:
            ref = pc(x, missing_mask = miss, **kw).clone()
        finally:
            tks.STACKED = True
        settled, real = [], tks.settled_forward
        tks.settled_forward = lambda *a, **k: (lambda got: (settled.append(got), got)[1])(real(*a, **k))
        try:
            with _spy_stacks(tks) as taken:
                got = [pc(x, missing_mask = miss, **kw).clone() for _ in range(2)]
        finally:
            tks.settled_forward = real
    for g in got:
        assert torch.equal(g, ref), float((g - ref).abs().max())
    assert len(taken) == 2 * (T - 1), taken                         # (every transition, both passes)
    # (the second pass: each layer's call its settled forward's -- the first pass's calls, settling, the full path's)
    assert settled.count(True) >= T - 1 and False not in settled, settled


@pytest.mark.parametrize("masked", [0.5, 1.0])
def test_the_stash_a_backward_takes_from_the_stack(masked):
    """The conditional (its forward keeps the stack's stash for its backward) and the gradients of the external tensors
    and the evidence through the circuit's backward: the unstacked path's, up to the backward kernels' order of adds."""
    tkf, tlf, tks = _modules()
    pc, kw, x, miss = _case(4, masked, seed = 11)

    def run():
        probs = conditional(pc, x, missing_mask = miss, **kw)
        ts = {g: tuple(t.clone().requires_grad_(True) for t in v) for g, v in kw["external_params"].items()}
        lpg = kw["categorical_evidence_logp"].clone().requires_grad_(True)
        ll = pc(x, missing_mask = miss, external_params = ts, categorical_evidence_logp = lpg,
                soft_evidence_cat_ids = kw["soft_evidence_cat_ids"], differentiable = True).view(-1)
        ll.sum().backward()
        return [probs, ll.detach()] + [t.grad for v in ts.values() for t in v] + [lpg.grad]

    tks.STACKED = False
    try:
        ref = run()
    finally:
        tks.STACKED = True
    settled, real = [], tks.settled_backward
    tks.settled_backward = lambda *a, **k: (lambda got: (settled.append(got), got)[1])(real(*a, **k))
    try:
        with _spy_stacks(tks) as taken:
            got = run()
    finally:
        tks.settled_backward = real
    assert len(taken) == 2 * (T - 1), taken                         # (the conditional's forward, the differentiable one)
    # (the conditional's backward by the settled backwards -- settled by its forward --, the gradients' by the full path)
    assert settled.count(True) == T - 1 and settled.count(False) == T - 1, settled
    assert torch.equal(got[1], ref[1])                              # (the values: the forward's, bit for bit)
    assert H._rel(got[0], ref[0]) < 1e-5, H._rel(got[0], ref[0])    # (the conditional's answers)
    for k, (u, v) in enumerate(zip(got[2:], ref[2:])):
        if masked < 1.0:
            assert H._rel(u, v) < 1e-5, (k, H._rel(u, v))
        else:                                                       # (every token summed out: each likelihood 1
            assert float((u - v).abs().max()) < 1e-7, k             #  whatever the tensors, its gradients 0 up to
                                                                    #  the roundoff, ~1e-10: compared absolutely)


def test_the_settled_forward_declines_what_its_stack_does_not_serve():
    """A settled layer's call the pass's stack does not take -- the stacked passes off (the unstacked kernels then), the
    fused kernels off (the torch passes) -- goes the full path: the values its own."""
    tkf, tlf, tks = _modules()
    pc, kw, x, miss = _case(3, 0.5, seed = 5)
    with torch.no_grad():
        first = pc(x, missing_mask = miss, **kw).clone()           # (settles the transitions' layers)
        lays = [l for grp in pc.inner_layer_groups for l in grp if l.__dict__.get("_settled_forward") is not None]
        assert len(lays) == T - 1, len(lays)
        with _spy_stacks(tks) as taken:
            tks.STACKED = False
            try:
                off = pc(x, missing_mask = miss, **kw).clone()
            finally:
                tks.STACKED = True
            tkf.TOP_K_FUSED = False
            try:
                torch_ = pc(x, missing_mask = miss, **kw).clone()
            finally:
                tkf.TOP_K_FUSED = True
            again = pc(x, missing_mask = miss, **kw).clone()
    assert torch.equal(off, first) and torch.equal(again, first)
    assert H._rel(torch_, first) < 2e-3, H._rel(torch_, first)      # (the kernels' fp16 logits against fp32)
    assert len(taken) == T - 1, taken                               # (the last call's alone)
