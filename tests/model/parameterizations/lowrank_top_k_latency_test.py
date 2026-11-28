"""
`LowRank(top_k)` sums' LATENCY mode (`lowrank/top_k_stack.py`, `kernels/c/lowrank_top_k_latency.cuh`): where no
backward follows and the batch is small, a stack's forward is one launch a position on the chain, each node's slot sums
made before the chain -- the throughput mode's operands and roundings, its sums reassociated (the children a chunk at a
time, each chunk shifted by its own largest). Against the throughput mode (the unstacked path's launches): the values up
to that reassociation, whatever the masks, the map of rows and the parameters' sources, and the same -inf entries;
every token masked, each likelihood 1 (the slots' sums and the chain's made of the same fp16 operands); the same bits
call to call, the pass whose product layer decisions ran sums before it included; an impossible sample's -inf; the
mode taken where it serves and nowhere else.
"""

import contextlib
import importlib.util
import os

import pytest
import torch

from pyjuice.queries import conditional

# (the stack tests' circuits and batches, loaded by path: the suite's importlib mode puts no test directory on
# `sys.path`)
_spec = importlib.util.spec_from_file_location(
    "_lowrank_top_k_stack_for_latency", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "lowrank_top_k_stack_test.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

T = S.T


@contextlib.contextmanager
def _modes(tks, latency = True, max_b = 8):
    """`top_k_stack.LATENCY` set to `latency` (and `LATENCY_MAX_B` to `max_b`, None: as it is); yields each stacked
    forward's mode in call order, True the latency mode's"""
    got, real, was = [], tks.SumStack.forward, (tks.LATENCY, tks.LATENCY_MAX_B)

    def forward(self, ns, pairs = None):
        lat = self.plan is not None and self.plan.latency()
        taken = real(self, ns, pairs)
        if taken:
            got.append(lat)
        return taken

    tks.SumStack.forward, tks.LATENCY = forward, latency
    if max_b is not None:
        tks.LATENCY_MAX_B = max_b
    try:
        yield got
    finally:
        tks.SumStack.forward, (tks.LATENCY, tks.LATENCY_MAX_B) = real, was


def _run(pc, x, miss, kw, calls = 2):
    """the log-likelihoods and the node values of the last of `calls` no-grad calls (from the second, the settled
    forwards')"""
    with torch.no_grad():
        for _ in range(calls):
            ll = pc(x, missing_mask = miss, **kw).view(-1).clone()
    return ll, pc.node_mars.clone()


def _sum_rows(pc):
    """the node rows of the circuit's `LowRank(top_k)` sums"""
    return [l._layer_nid_range for grp in pc.inner_layer_groups for l in grp
            if getattr(getattr(l, "parameterization", None), "top_k", None) is not None]


# MEASURED (the stack tests' circuit, b 1 .. 8, the maps and sources below): the largest |latency - throughput| of the
# log-likelihoods and of the sums' values, 1.9e-6 and 2.9e-6 every token observed, 4.4e-5 and 5.8e-4 half of them
# masked, 1.6e-6 and 2.6e-4 all of them -- each mode 6.8e-4 and 2.9e-3 from the torch passes at most, alike
_BOUNDS = {0.0: (1e-5, 2e-5), 0.5: (2e-4, 3e-3), 1.0: (1e-5, 1e-3)}


@pytest.mark.parametrize("masked", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("rows", [None, [0, 2, 2, 1]])
@pytest.mark.parametrize("tied", [True, False])
def test_the_latency_forward_is_the_throughput_one(masked, rows, tied):
    """The values against the throughput mode's -- every token observed, half of them masked, all of them; a sample its
    own tensors, or the samples of a row of tensors sharing it; the transitions' parameters one source, or one a
    position --, the same entries -inf; every token masked, each likelihood 1; every transition by the latency mode,
    both passes (the second by the settled forwards)."""
    tkf, tlf, tks = S._modules()
    pc, kw, x, miss = S._case(3 if rows else 4, masked, tied = tied, rows = rows)
    with _modes(tks, latency = False) as modes:
        thr = _run(pc, x, miss, kw)
    assert modes == [False] * (2 * (T - 1)), modes
    with _modes(tks) as modes:
        lat = _run(pc, x, miss, kw)
    assert modes == [True] * (2 * (T - 1)), modes
    bl, bv = _BOUNDS[masked]
    assert torch.equal(torch.isfinite(lat[0]), torch.isfinite(thr[0])) and bool(torch.isfinite(thr[0]).all())
    assert float((lat[0] - thr[0]).abs().max()) < bl, float((lat[0] - thr[0]).abs().max())
    for s, e in _sum_rows(pc):
        a, b = lat[1][s:e], thr[1][s:e]
        fin = torch.isfinite(b)
        assert torch.equal(torch.isfinite(a), fin)
        assert float((a - b)[fin].abs().max()) < bv, (s, float((a - b)[fin].abs().max()))
    if masked == 1.0:                                               # (every token summed out: Z and U of each chunk
        assert float(lat[0].abs().max()) < 1e-5, lat[0]             #  the same fp16 W, at the reference the slot
                                                                    #  sums kept for it -- MEASURED 2.4e-6; Z's W at
                                                                    #  another reference than U's, 5e-5)


def test_the_latency_forward_is_the_same_call_to_call(monkeypatch):
    """Three calls, the same bits (the chunks' shares added in their order: no atomics on values) -- the second the pass
    whose product layer decisions' trials ran its sums before it (`TensorCircuit._stream_fusion_groups`: the stack's
    buffers given back after them, the stack made again), the third the fused passes'."""
    import pyjuice.layer.kernels.autotune as autotune
    real = autotune.pick

    def pick(key, cands, bench, *a, **k):                           # (every candidate's trial, as a tuner with none
        if key[0] != "prod_stream_fusion":                          #  stored runs them -- a stored decision runs
            return real(key, cands, bench, *a, **k)                 #  none -- then the fused passes)
        for c in cands:
            bench(c)
        return "fused"
    monkeypatch.setattr(autotune, "pick", pick)
    tkf, tlf, tks = S._modules()
    pc, kw, x, miss = S._case(4, 0.5, fuses = True)
    with _modes(tks) as modes:
        got = [_run(pc, x, miss, kw, calls = 1) for _ in range(3)]
    assert modes and all(modes), modes
    rows = _sum_rows(pc)
    for ll, nm in got[1:]:
        assert torch.equal(ll, got[0][0]), float((ll - got[0][0]).abs().max())
        for s, e in rows:
            assert torch.equal(nm[s:e], got[0][1][s:e]), s


def test_an_impossible_sample_is_minus_infinity():
    """A sample whose soft evidence is impossible (every candidate's -inf) at a masked position: its log-likelihood
    -inf, not NaN (its slots' weights 0, its children's values -inf through the chunks' shifts), every other sample's
    as without it, bit for bit."""
    tkf, tlf, tks = S._modules()
    pc, kw, x, miss = S._case(4, 0.5, seed = 13)
    miss = miss.clone()
    miss[0, 1] = True
    bad = dict(kw, categorical_evidence_logp = kw["categorical_evidence_logp"].clone())
    bad["categorical_evidence_logp"][0] = -float("inf")
    with _modes(tks) as modes:
        good = _run(pc, x, miss, kw)[0]
        got = _run(pc, x, miss, bad)[0]
    assert modes and all(modes), modes
    assert float(got[0]) == -float("inf") and bool(torch.isfinite(good).all()), (got, good)
    assert torch.equal(got[1:], good[1:])


def test_the_latency_mode_where_no_backward_follows():
    """Taken by a no-grad forward of at most `LATENCY_MAX_B` samples; not by a larger batch's, nor where a backward may
    follow (the conditional's forward, which keeps the stash; a differentiable forward), nor with it off."""
    tkf, tlf, tks = S._modules()
    nb = tks.LATENCY_MAX_B
    pc, kw, x, miss = S._case(nb, 0.5)
    with _modes(tks, max_b = None) as modes:
        _run(pc, x, miss, kw, calls = 1)
    assert modes == [True] * (T - 1), modes
    with _modes(tks, latency = False) as modes:
        _run(pc, x, miss, kw, calls = 1)
    assert modes == [False] * (T - 1), modes
    with _modes(tks, max_b = None) as modes:
        conditional(pc, x, missing_mask = miss, **kw)
    assert modes == [False] * (T - 1), modes
    ts = {g: tuple(t.clone().requires_grad_(True) for t in v) for g, v in kw["external_params"].items()}
    with _modes(tks, max_b = None) as modes:
        ll = pc(x, missing_mask = miss, external_params = ts, soft_evidence_cat_ids = kw["soft_evidence_cat_ids"],
                categorical_evidence_logp = kw["categorical_evidence_logp"], differentiable = True)
    ll.sum().backward()
    assert modes == [False] * (T - 1), modes
    pc2, kw2, x2, miss2 = S._case(nb + 1, 0.5)
    with _modes(tks, max_b = None) as modes:
        _run(pc2, x2, miss2, kw2, calls = 1)
    assert modes == [False] * (T - 1), modes
