"""
The fused dense live-column product (kernels/c/dense_live_mma.cu: a bf16 query's ``log(W @ exp(x[:, perm] - m)) + m``
over the live lanes on tensor cores, in two forms -- the gather fused into every row tile, or the exponentials computed
once --, the log and scatter fused) against float64 at every arm the dispatch offers (form x config x split), both weight
layouts (K-major, MN-major), the three modes
(overwrite with -inf at the dead columns, keep them, log-add), rows and K off the tiles, a live column of -inf, a row
of zero weights, a single live lane -- held to the chain's own bf16 class (its operands are bf16 too). Then the
dispatch: a bf16 HMM decode session with the fused kernel forced against the float64 reference (strong_check's bars)
and against the chain forced; the same bits twice; what it declines (tf32 queries, 64 lanes and more, misaligned
weights). Skipped where the extension cannot build.
"""
import importlib.util
import os

import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
from pyjuice.constraints.inference.dfa_exact.kernels import c as dfa_c
from pyjuice.layer.kernels import autotune

# strong_check_test's session harness and float64 reference (the tests import under importlib: by path)
_spec = importlib.util.spec_from_file_location("strong_check_test", os.path.join(os.path.dirname(__file__), "strong_check_test.py"))
strong = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strong)
Float64Reference, check_against_reference, run_session, schedule, small_case = (
    strong.Float64Reference, strong.check_against_reference, strong.run_session, strong.schedule, strong.small_case)

DEV = torch.device("cuda:0")
NEG = -float("inf")

needs_kernel = pytest.mark.skipif(not torch.cuda.is_available() or dfa_c.get_dense_live_module() is None,
                                  reason = "needs the fused dense live product (nvcc + CUTLASS + sm_90+)")


# --------------------------------------------------------------------------------- the kernel

def _reference(A, x, perm, rank, L, ncols, mode, out0):
    """``out`` after the step in float64: ``log(A @ exp(x[:, perm] - m)) + m`` scattered by ``rank`` -- the dead columns
    -inf (mode 0), kept (1), or every live column log-added (2)."""
    xs = x.double()[:, perm[:L].long()]
    m = xs.max(0).values
    shift = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    v = torch.log(A.double() @ torch.exp(xs - shift)) + shift                                  # [M, L]
    out = out0.double().clone()
    rk = rank[:ncols].long()
    live = rk >= 0
    vals = v[:, rk.clamp(min = 0)]
    if mode == 0:
        out[:, :ncols] = torch.where(live[None], vals, torch.full_like(vals, NEG))
    elif mode == 1:
        out[:, :ncols] = torch.where(live[None], vals, out[:, :ncols])
    else:
        out[:, :ncols] = torch.where(live[None], torch.logaddexp(out[:, :ncols], vals), out[:, :ncols])
    return out


def _case(M, K, L, ncols, k_major, scattered, seed = 0):
    """bf16 weights [M, K] (a row of zeros) in the layout, values x [K, ncols + 7] with -inf in places and a live column
    of -inf, the live lanes (scattered over the columns, or one run) as int32 perm / rank."""
    g = torch.Generator(device = DEV).manual_seed(seed)
    x = torch.randn(K, ncols + 7, device = DEV, generator = g) * 3
    x[torch.rand(K, ncols + 7, device = DEV, generator = g) < 0.2] = NEG
    if scattered:
        perm = torch.randperm(ncols, device = DEV, generator = g)[:L].sort().values
    else:
        perm = torch.arange(L, device = DEV) + (ncols - L) // 2
    x[:, perm[0]] = NEG
    rank = torch.full((ncols,), -1, dtype = torch.int32, device = DEV)
    rank[perm] = torch.arange(L, dtype = torch.int32, device = DEV)
    w = torch.rand(M, K, device = DEV, generator = g) / K
    w[0] = 0
    A = w.to(torch.bfloat16) if k_major else w.t().contiguous().to(torch.bfloat16).t()
    return A, x, perm.to(torch.int32), rank


SHAPES = [(4096, 4096, 8, 16, False), (64, 4096, 3, 1584, True), (104, 1000, 17, 50, True), (200, 72, 63, 63, False),
          (8, 8, 1, 4, False), (304, 520, 40, 97, True), (128, 136, 16, 16, False)]


@needs_kernel
@pytest.mark.parametrize("M,K,L,ncols,scattered", SHAPES, ids = [f"M{s[0]}K{s[1]}L{s[2]}" for s in SHAPES])
@pytest.mark.parametrize("k_major", [True, False], ids = ["K-major", "MN-major"])
def test_every_arm_matches_float64(M, K, L, ncols, scattered, k_major):
    mod = dfa_c.get_dense_live_module()
    A, x, perm, rank = _case(M, K, L, ncols, k_major, scattered)
    op = lifted_sum._dense_live_operand(A)
    assert op is not None and op[0] == k_major
    arms = dfa_c.dense_live_arms(M, K, L)
    assert arms, "no arm offered"
    knt = -(-K // 64)
    codes = sorted({c | (sp << 8) for c in range(len(mod.dense_live_mma_configs())) if mod.dense_live_mma_fits(c)
                    for sp in (1, 2, min(16, knt)) if sp <= knt} | set(arms))
    for mode in (0, 1, 2):
        out0 = torch.randn(M, ncols + 3, device = DEV)
        out0[:, ::5] = NEG
        want = _reference(A, x, perm, rank, L, ncols, mode, out0)
        fin = torch.isfinite(want)
        assert fin.any()
        chain = out0.clone()
        lifted_sum._bf16_live_chain(x, perm, rank, L, ncols, [(A, chain, chain.stride(0), mode)])
        d_chain = float((chain.double() - want)[fin].abs().max())
        for code in codes:
            out = out0.clone()
            mod.dense_live_mma(out, out.stride(0), A, op[1], k_major, x, x.stride(0), perm, rank, M, K, L, ncols, mode, code)
            torch.cuda.synchronize()
            what = f"mode {mode}, config {code & 255} {tuple(int(v) for v in mod.dense_live_mma_configs()[code & 255][:3])} split {code >> 8}"
            assert not torch.isnan(out).any(), f"{what}: NaN"
            assert torch.equal(torch.isneginf(out), torch.isneginf(want)), f"{what}: the -inf pattern differs"
            d = float((out.double() - want)[fin].abs().max())
            # bf16 operands: the chain's class (its worst error here, plus the summation orders' difference)
            assert d < 4e-3 and d <= 1.5 * d_chain + 2e-4, f"{what}: off by {d:.2e} (the chain {d_chain:.2e})"


@needs_kernel
def test_same_bits_twice():
    mod = dfa_c.get_dense_live_module()
    A, x, perm, rank = _case(512, 1000, 20, 40, True, True)
    outs = []
    for _ in range(2):
        out = torch.randn(512, 40, device = DEV, generator = torch.Generator(device = DEV).manual_seed(3))
        for code in dfa_c.dense_live_arms(512, 1000, 20):
            mod.dense_live_mma(out, 40, A, A.stride(0), True, x, x.stride(0), perm, rank, 512, 1000, 20, 40, 2, code)
        outs.append(out.clone())
    assert torch.equal(outs[0], outs[1])


# --------------------------------------------------------------------------------- the dispatch

def _force(monkeypatch, which):
    """`autotune.choose` picking ``which`` at the dense live gate without a trial -- "chain", or "form0" / "form1": the
    first arm of that form (the gather fused, or the exponentials once) --, the real one elsewhere; returns the keys
    it saw."""
    real, seen = autotune.choose, []
    forms = [int(v[4]) for v in dfa_c.get_dense_live_module().dense_live_mma_configs()]

    def choose(key, candidates):
        if isinstance(key, tuple) and key and key[0] == "dfa_dense_live":
            seen.append(key)
            keys = [k for k, _ in candidates]
            if which == "chain":
                return "chain"
            return next(k for k in keys if k != "chain" and forms[k & 255] == int(which[-1]))
        return real(key, candidates)

    monkeypatch.setattr(autotune, "choose", choose)
    autotune.set_autotune(clear_cache = True)
    return seen


def _calls(monkeypatch):
    """Every call of the extension's product, recorded."""
    mod = dfa_c.get_dense_live_module()
    real, calls = mod.dense_live_mma, []

    def record(*a):
        calls.append(a[-6:])
        return real(*a)

    monkeypatch.setattr(mod, "dense_live_mma", record)
    return calls


@needs_kernel
@pytest.mark.parametrize("form", ["form0", "form1"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("mode", ["diffusion", "l2r"])
def test_a_bf16_session_against_float64_and_the_chain(mode, storage, form, monkeypatch):
    """A bf16 HMM session with the fused kernel forced: within strong_check's bf16 bars of the float64 reference, -inf
    where it has it; and within bf16's rounding of the same session with the chain forced. On compact rows every dense
    step is a live step (the row's lanes); in dense storage only the steps liveness prunes are, which this small case
    has none of -- the fused kernel then serves nothing, and the session is the chain's."""
    pc, dfa, data, inputs = small_case("hmm")
    cc = jc.compile(dfa, pc)
    masks = schedule(mode, data.size(0), data.size(1))
    calls = _calls(monkeypatch)
    _force(monkeypatch, form)
    check_against_reference(cc, Float64Reference(pc, dfa, data.size(1)), data, masks, inputs, storage, "bf16")
    assert bool(calls) == (storage == "compact"), "the fused kernel serves every compact step, no step of this dense case"
    fused = run_session(cc, data, masks, inputs, storage, "bf16")
    _force(monkeypatch, "chain")
    n = len(calls)
    chain = run_session(cc, data, masks, inputs, storage, "bf16")
    assert len(calls) == n, "the chain forced, the fused kernel still ran"
    assert fused.keys() == chain.keys()
    for i in fused:
        for u, v in zip(fused[i], chain[i]):
            assert torch.equal(torch.isneginf(u), torch.isneginf(v)), f"call {i}: the -inf pattern differs"
            fin = torch.isfinite(u)
            assert float((u - v)[fin].abs().max()) < 1.5e-2, f"call {i}"


@needs_kernel
def test_it_declines_what_it_does_not_serve(monkeypatch):
    """A tf32 session never reaches it; 64 lanes and more take no arm; misaligned weights take the chain."""
    pc, dfa, data, inputs = small_case("hmm")
    cc = jc.compile(dfa, pc)
    calls = _calls(monkeypatch)
    _force(monkeypatch, "form0")
    run_session(cc, data, inputs = inputs, masks = schedule("l2r", data.size(0), data.size(1)), storage = "compact",
                precision = "tf32")
    assert not calls
    assert dfa_c.dense_live_arms(256, 256, 64) == [] and dfa_c.dense_live_arms(256, 256, 1000) == []
    assert dfa_c.dense_live_arms(256, 256, 63) and dfa_c.dense_live_arms(1, 1, 1)
    A, x, perm, rank = _case(64, 72, 5, 8, True, False)
    assert lifted_sum._dense_live_operand(A[:, 1:]) is None                              # a base off 16 bytes
    off = torch.zeros(72, 70, dtype = torch.bfloat16, device = DEV)[:, :64].t()          # MN-major, columns of 70
    assert lifted_sum._dense_live_operand(off) is None
    out = torch.zeros(64, 8, device = DEV)
    lifted_sum._bf16_live_step(x[1:], perm, rank, 5, 8, [(A[:, 1:], out, 8, 0)])        # K = 71, off: the chain
    assert not calls and torch.isfinite(out[1:, perm[1:].long()]).all()          # (row 0: zero weights, -inf)
