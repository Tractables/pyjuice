"""
The decoder session's fused glue (`pyjuice.queries.decoder.kernels`) against the torch forms it replaces:

* `normalize_target_flows` -- the segmented log-sum-exp over the targets' rows and the rows shifted by it -- against
  the torch form of `Decoder._normalize_target_flows`, for one target (a slice of rows, an index) and several of as
  many rows each, on flows with -inf entries, columns all -inf, batch sizes off the tiles;
* `categorical_conditional` against `queries.distributions.categorical.backward` (its matmul form) on the same flows,
  and declining where that form does not apply as it is (fewer categories, several targets);
* a decoder session left to right on circuits that take the fused path (128 and 256 categories) against the queries.
"""

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.queries.decoder import kernels
from pyjuice.queries.distributions import categorical

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _torch_normalize(nf, rows, num_targets):
    """`Decoder._normalize_target_flows`' torch form (rows: a slice or an index, every target as many rows, in order)."""
    vals = nf[rows]
    B = vals.size(1)
    if num_targets == 1:
        m = vals.amax(dim = 0, keepdim = True)
        m = torch.where(torch.isfinite(m), m, 0.0)
        tot = torch.exp(vals - m).sum(dim = 0, keepdim = True)
        log_p = torch.where(tot > 0, m + torch.log(tot), 0.0)
        nf[rows] = vals - log_p
        return log_p[0]
    seg = torch.arange(num_targets, device = nf.device).repeat_interleave(vals.size(0) // num_targets)
    m = vals.view(num_targets, -1, B).amax(dim = 1)
    m = torch.where(torch.isfinite(m), m, 0.0)
    tot = torch.zeros([num_targets, B], dtype = vals.dtype, device = vals.device).index_add_(0, seg, torch.exp(vals - m[seg]))
    log_p = torch.where(tot > 0, m + torch.log(tot), 0.0)
    nf[rows] = vals - log_p[seg]
    return log_p[0]


@cuda_only
@pytest.mark.parametrize("B", [1, 5, 16, 70])
@pytest.mark.parametrize("layout", ["slice", "index_one", "index_several"])
@pytest.mark.parametrize("per", [1, 3, 1024, 5000])
def test_normalize_target_flows_is_the_torch_form(B, layout, per):
    g = torch.Generator(device = DEV).manual_seed(B * 31 + per)
    N = 3 * per + 40
    nf = torch.randn([N, B], device = DEV, generator = g) * 4.0 - 10.0
    nf[torch.rand([N, B], device = DEV, generator = g) < 0.3] = -float("inf")
    if B > 1:
        nf[:, 0] = -float("inf")                                       # a sample nothing weighs
    if layout == "slice":
        rows, T = slice(17, 17 + per), 1
    elif layout == "index_one":
        rows, T = torch.randperm(N, device = DEV, generator = g)[:per], 1
    else:
        rows, T = torch.randperm(N, device = DEV, generator = g)[:3 * per], 3
    # Against float64: the kernel's error no larger than the torch form's own (each sums thousands of terms in fp32, in
    # its own order) -- and the same infinities
    ref, ref64 = nf.clone(), nf.double()
    want = _torch_normalize(ref, rows, T)
    want64 = _torch_normalize(ref64, rows, T)
    got = kernels.normalize_target_flows(nf, rows, T)
    assert torch.equal(torch.isinf(got), torch.isinf(want)) and torch.equal(torch.isinf(nf), torch.isinf(ref))
    fin = torch.isfinite(ref)
    for mine, theirs, exact in ((got, want, want64), (nf[fin], ref[fin], ref64[fin])):
        err, err_torch = float((mine.double() - exact).abs().max()), float((theirs.double() - exact).abs().max())
        assert err <= max(2.0 * err_torch, 2e-6), (err, err_torch)


def _hmm(num_cats, latents = 64, T = 12, seed = 0):
    torch.manual_seed(seed)
    ns = juice.structures.HMM(seq_length = T, num_latents = latents, num_emits = num_cats)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), T


def _hclt(num_cats, V = 16, seed = 0):
    torch.manual_seed(seed)
    x = torch.randint(0, num_cats, (512, V))
    ns = juice.structures.HCLT(x.float(), num_latents = 32, input_dist = dists.Categorical(num_cats = num_cats))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV), V


@cuda_only
@pytest.mark.parametrize("model", ["hmm", "hclt"])
@pytest.mark.parametrize("num_cats", [128, 256])
@pytest.mark.parametrize("B", [1, 16, 33])
def test_categorical_conditional_is_the_matmul_form(model, num_cats, B):
    """On the flows of a conditional's backward, the fused form against `categorical.backward` for every variable: the
    weights to the bit (libdevice's exp), the matmul the same, the normalization's sum in another order."""
    pc, V = _hmm(num_cats) if model == "hmm" else _hclt(num_cats)
    x = torch.randint(0, num_cats, (B, V), device = DEV)
    miss = torch.rand(B, V, device = DEV) < 0.5
    juice.queries.conditional(pc, data = x.clone(), missing_mask = miss, target_vars = [0])     # (flows in place)
    layer = pc.input_layer_group[0]
    targets = kernels.CategoricalTargets(layer)
    for v in range(V):
        got = kernels.categorical_conditional(targets, pc.node_flows, [v])
        assert got is not None, v                                                # (the fused form applies)
        want = categorical.backward(layer, x, pc.node_flows, pc.node_mars, layer.params, target_vars = [v],
                                    logspace_flows = True)
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol = 1e-7, rtol = 1e-6), (v, float((got - want).abs().max()))


@cuda_only
def test_categorical_conditional_declines_what_it_does_not_compute():
    pc, V = _hmm(64)                                                             # below the matmul form's categories
    t = kernels.CategoricalTargets(pc.input_layer_group[0])
    assert kernels.categorical_conditional(t, pc.node_flows, [0]) is None
    pc, V = _hmm(256)
    t = kernels.CategoricalTargets(pc.input_layer_group[0])
    assert kernels.categorical_conditional(t, pc.node_flows, [0, 1]) is None     # several targets
    assert kernels.categorical_conditional(t, pc.node_flows, None) is None       # every variable
    assert kernels.categorical_conditional(t, pc.node_flows, [V + 3]) is None    # not a variable of the layer


@cuda_only
@pytest.mark.parametrize("model", ["hmm", "hclt"])
@pytest.mark.parametrize("early_stop", [False, True])
def test_a_session_on_the_fused_path_answers_as_the_queries(model, early_stop, monkeypatch):
    """Left to right (one target a call) and a call of several targets, on circuits of 256 categories: the session's
    conditionals against `queries.conditional`; the fused input-layer form taken on every one-target call."""
    pc, V = _hmm(256) if model == "hmm" else _hclt(256)
    B = 8
    x = torch.randint(0, 256, (B, V), device = DEV, generator = torch.Generator(device = DEV).manual_seed(1))
    taken = []
    real = kernels.categorical_conditional
    monkeypatch.setattr(kernels, "categorical_conditional", lambda *a, **k: taken.append(real(*a, **k) is not None) or real(*a, **k))
    with juice.decoder(pc, early_stop = early_stop) as dec:
        for t in list(range(V)) + [None]:
            m = torch.ones(B, V, dtype = torch.bool, device = DEV)
            m[:, :t if t is not None else V // 2] = False
            targets = [t] if t is not None else list(range(V // 2, V))
            got = dec.conditional(x, missing_mask = m, target_vars = targets)
            want = juice.queries.conditional(pc, data = x.clone(), missing_mask = m, target_vars = targets)
            assert torch.allclose(got, want, atol = 1e-5 if early_stop else 1e-6, rtol = 0.0), \
                (t, float((got - want).abs().max()))
    assert taken == [True] * V + [False], taken


@cuda_only
def test_a_target_listed_twice_takes_the_torch_normalization(monkeypatch):
    """Rows in two targets' sets (a variable listed twice) would be read and written by two programs of the fused
    normalization at once: the session keeps the torch form there -- and answers as the query does."""
    pc, V = _hmm(256)
    B = 4
    x = torch.randint(0, 256, (B, V), device = DEV)
    m = torch.ones(B, V, dtype = torch.bool, device = DEV); m[:, :3] = False
    calls = []
    real = kernels.normalize_target_flows
    monkeypatch.setattr(kernels, "normalize_target_flows", lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    with juice.decoder(pc, early_stop = True) as dec:
        dec.conditional(x, missing_mask = m, target_vars = [3])
        got = dec.conditional(x, missing_mask = m, target_vars = [4, 5, 4])
    want = juice.queries.conditional(pc, data = x.clone(), missing_mask = m, target_vars = [4, 5, 4])
    assert torch.allclose(got, want, atol = 1e-5, rtol = 0.0), float((got - want).abs().max())
    assert calls == [1], calls                                         # the one-target call only
