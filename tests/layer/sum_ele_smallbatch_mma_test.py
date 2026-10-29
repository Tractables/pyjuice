"""
The tensor-core small-batch CUDA element-flow backward (c/smallbatch_ele_backward_mma.cu: a child tile x a batch tile x
a range of 64-parent tiles per block, the ranges merged by a dependent launch) against a float64 reference, at every
(config, split) code the arms offer, on layouts of several element blocks with scattered parent tiles and parameter
blocks of 64 to 1024, at batches 1..63, with -inf log ratios (and a column of impossible parents), fp32 parameters (both
operands hi + lo bf16 halves: held to 1e-4, which one bf16 each misses) and bf16 ones (one-bf16 exponentials); and the layer's dispatch, told the gate chose a code, reaches it on whole launches (batch 1 to
40, where the element tiles are 64 and where the tile family picks another width) and agrees with Triton; under
`precision = "fp32"` it is not offered. Skipped where the CUDA kernels cannot build (CUTLASS, sm_90+).
"""
import itertools
import pytest
import torch

import pyjuice as juice
from pyjuice.layer import sum_layer as SL
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import c as ck

DEV = torch.device("cuda:0")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _available():
    return torch.cuda.is_available() and ck.smallbatch_ele_mma_is_available()


def _layout(neb, cs, kt, bsk, batch, dt, neg_inf, group = 1):
    """Element blocks of `cs` children over `kt` tiles of 64 parents (scattered over the node rows; `group` > 1: aligned
    runs of that many blocks share them, an HCLT's), the parameters of a tile in one block of `bsk` >= 64 or in two of
    32 (`bsk` 32: two bases a tile)."""
    pbt = 2 if bsk == 32 else 1
    n_rows = (neb * kt + 5) * 64
    nflows = torch.randn(n_rows, batch, device = DEV) * 2
    nmars = torch.randn(n_rows, batch, device = DEV) * 2
    if neg_inf:
        m = torch.rand(n_rows, batch, device = DEV) < 0.2
        nmars[m] = -float("inf"); nflows[m] = -float("inf")
        if batch > 1:
            nflows[:, -1] = -float("inf"); nmars[:, -1] = -float("inf")     # last column: every parent impossible
    perm = torch.randperm(neb * kt + 5, device = DEV)[:neb * kt]
    ebase = (perm * 64).view(neb, kt)
    ebase = ebase[(torch.arange(neb, device = DEV) // group) * group].contiguous()   # a group's blocks share parents
    pbase = (torch.arange(neb * kt * pbt, device = DEV) * cs * max(bsk, 64)).view(neb, kt * pbt).contiguous()
    params = (torch.rand(neb * kt * pbt * cs * max(bsk, 64), device = DEV) + 1e-3).to(dt)
    emars = torch.randn(neb * cs + 9, batch, device = DEV)
    chids = torch.arange(neb, device = DEV) * cs + 9
    eflows = torch.full((neb * cs + 9, batch), float("nan"), device = DEV)
    return eflows, emars, nflows, nmars, params, chids, ebase, pbase, pbt


def _reference(emars, nflows, nmars, params, chids, ebase, pbase, cs, kt, bsk, pbt):
    out, p = [], params.double()
    w = 64 // pbt                                                            # parents of a tile in one parameter block
    for eb in range(chids.numel()):
        rows = (ebase[eb][:, None] + torch.arange(64, device = DEV)[None, :]).reshape(-1)
        nm, nf = nmars[rows], nflows[rows]
        lf = torch.where(torch.isneginf(nm), torch.full_like(nm, -float("inf")), nf - nm).double()
        W = torch.stack([torch.cat([p[pbase[eb, t * pbt + h]:].as_strided((cs, w), (bsk, 1)) for h in range(pbt)], 1)
                         for t in range(kt)], 1).reshape(cs, -1)
        mx = lf.max(dim = 0, keepdim = True).values
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        out.append(torch.log(W @ torch.exp(lf - mx)) + mx + emars[chids[eb]:chids[eb] + cs].double())
    return torch.cat(out)


LAYOUTS = {"3x128x4x128": (3, 128, 4, 128), "2x64x3x64": (2, 64, 3, 64), "1x1024x8x1024": (1, 1024, 8, 1024),
           "2x256x1x256": (2, 256, 1, 256), "8x32x4x32-g4": (8, 32, 4, 32, 4), "4x32x2x32-g2": (4, 32, 2, 32, 2),
           "4x64x3x32": (4, 64, 3, 32), "8x32x2x64-g4": (8, 32, 2, 64, 4)}


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch element backward cannot build here")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("neg_inf", [False, True])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_every_arm_matches_float64(layout, neg_inf, dtype):
    torch.manual_seed(0)
    neb, cs, kt, bsk, *grp = LAYOUTS[layout]
    group = grp[0] if grp else 1
    for batch in (1, 3, 8, 15, 16, 17, 33, 63):
        eflows, emars, nflows, nmars, params, chids, ebase, pbase, pbt = _layout(neb, cs, kt, bsk, batch, DTYPES[dtype],
                                                                                neg_inf, group)
        ref = _reference(emars, nflows, nmars, params, chids, ebase, pbase, cs, kt, max(bsk, 64) if pbt == 1 else bsk, pbt)
        codes = ck.smallbatch_ele_mma_arms(cs, bsk, batch, kt * 64, neb, group)
        assert codes
        if group > 1:
            assert any(ck.smallbatch_ele_mma_rows(c, cs) > 1 for c in codes)
        for code in codes:
            ef = eflows.clone()
            ck.smallbatch_ele_backward_sum_mma(ef, emars, nflows, nmars, params, chids, ebase, pbase, batch,
                                               max(bsk, 64) if pbt == 1 else bsk, cs, kt * 64, code, None, pbt)
            got = ef[9:].double()
            assert not torch.isnan(got).any(), (batch, code)
            assert torch.equal(torch.isneginf(got), torch.isneginf(ref)), (batch, code)
            fin = torch.isfinite(ref)
            err = float(((got - ref).abs() / ref.abs().clamp_min(1.0))[fin].max())
            assert err < (3e-3 if dtype == "bf16" else 1e-4), (batch, code, err)
            assert torch.isnan(ef[:9]).all()                                    # rows before the layer untouched


class _ForceEle:
    """`autotune.best_of` with the small-batch element pick forced to a choice (recording what was offered)."""

    def __init__(self, real, choice):
        self.real, self.choice, self.offered = real, choice, []

    def __getattr__(self, name):
        return getattr(self.real, name)

    def best_of(self, candidates, *args, **kwargs):
        keys = [k for k, _ in candidates]
        if keys and keys[0] == ("triton", -1) and any(k[0] in ("cuda", "cuda_mma") for k in keys[1:]):
            self.offered.append(keys)
            if self.choice == "first_mma":
                return next(k for k in keys if k[0] == "cuda_mma")
            return self.choice if self.choice in keys else keys[0]
        return self.real.best_of(candidates, *args, **kwargs)


def _flows(pc, x, monkeypatch, choice):
    force = _ForceEle(autotune, choice)
    monkeypatch.setattr(SL, "autotune", force)
    from pyjuice.utils.launch_plans import LaunchPlans
    for lg in pc.inner_layer_groups:
        for layer in lg:
            if hasattr(layer, "_cached_bk_ele_choice"):
                layer._cached_bk_ele_choice.clear()
            LaunchPlans.of(layer).clear()                                       # (its plans replay the choice)
    pc(x)
    pc.backward(x, compute_param_flows = False)
    return pc.element_flows.detach().clone(), force


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch element backward cannot build here")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("batch", [1, 5, 16, 40])
def test_dispatch_reaches_it_and_agrees_with_triton(batch, dtype, monkeypatch):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 5, num_latents = 512, num_emits = 16, homogeneous = True, block_size = 128)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = DTYPES[dtype]).to(DEV)
    x = torch.randint(0, 16, (batch, 5), device = DEV)
    ref, _ = _flows(pc, x, monkeypatch, ("triton", -1))
    calls = []
    real = ck.smallbatch_ele_backward_sum_mma

    def spy(*args, **kwargs):
        calls.append(args[12]); return real(*args, **kwargs)
    monkeypatch.setattr(SL.cuda_kernels, "smallbatch_ele_backward_sum_mma", spy)
    got, force = _flows(pc, x, monkeypatch, "first_mma")
    assert force.offered and calls
    fin = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(got), fin)
    assert (got[fin] - ref[fin]).abs().max() < 2e-2, float((got[fin] - ref[fin]).abs().max())


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch element backward cannot build here")
def test_not_offered_at_fp32_precision(monkeypatch):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 4, num_latents = 256, num_emits = 16, homogeneous = True, block_size = 128)
    ns.init_parameters(perturbation = 2.0)
    for precision, expect in (("auto", True), ("tf32", True), ("fp32", False)):
        pc = juice.compile(ns, verbose = False, precision = precision).to(DEV)
        x = torch.randint(0, 16, (8, 4), device = DEV)
        _, force = _flows(pc, x, monkeypatch, ("triton", -1))
        offered = any(k[0] == "cuda_mma" for keys in force.offered for k in keys)
        assert offered == expect, (precision, offered)
