"""
The tensor-core small-batch CUDA sum forward (c/smallbatch_forward_sum_mma.cu: a node tile x a batch tile x an edge range
per block, bf16 operands, the ranges merged by a dependent launch) against a float64 reference, at every (config, split)
code the arms offer, on block-sparse layouts of 64-, 128- and 1024-node blocks at batches 1..63 (tile tails included),
with -inf children (a leading run, a whole column, a whole node block), fp32 and bf16 parameters; its agreement with the
FMA small-batch kernel it competes with; and the layer's dispatch, told the gate chose a code, reaches it on a whole
launch and agrees with the Triton forward. Skipped where the CUDA kernels cannot build (CUTLASS, sm_90+).
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
    return torch.cuda.is_available() and ck.smallbatch_fw_is_available() and ck.smallbatch_fw_mma_is_available()


def _layout(ng, block_size, num_edges, batch, dt, neg_inf):
    node_mars = torch.full((ng * block_size + 7, batch), float("nan"), device = DEV)
    element_mars = torch.randn(ng * num_edges + 3, batch, device = DEV) * 3
    if neg_inf:
        element_mars[torch.rand(element_mars.shape, device = DEV) < 0.3] = -float("inf")
        element_mars[3:3 + num_edges // 2, 0] = -float("inf")                 # block 0, column 0: a leading -inf run
        element_mars[3 + num_edges // 2:3 + num_edges, 0] = -1.0
        if ng > 1:
            element_mars[3 + num_edges:3 + 2 * num_edges] = -float("inf")    # block 1: every child -inf
        if batch > 1:
            element_mars[:, -1] = -float("inf")                              # last column: every child -inf
    params = (torch.rand(ng * num_edges * block_size, device = DEV) + 1e-3).to(dt)
    nids = torch.arange(ng, device = DEV) * block_size + 7
    ebase = torch.arange(ng, device = DEV) * num_edges + 3
    pbase = torch.arange(ng, device = DEV) * (num_edges * block_size)
    return node_mars, element_mars, params, nids, ebase, pbase


def _reference(element_mars, params, nids, ebase, pbase, block_size, num_edges):
    em, p = element_mars.double(), params.double()
    out = []
    for nb in range(nids.numel()):
        e = em[ebase[nb]:ebase[nb] + num_edges]
        w = p[pbase[nb]:pbase[nb] + num_edges * block_size].view(num_edges, block_size)
        mx = e.max(dim = 0, keepdim = True).values
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        out.append(torch.log(w.t() @ torch.exp(e - mx)) + mx)
    return torch.cat(out)


LAYOUTS = {"3x128x256": (3, 128, 256), "2x64x192": (2, 64, 192), "1x1024x512": (1, 1024, 512), "2x256x64": (2, 256, 64)}
BATCHES = [1, 3, 5, 8, 15, 16, 17, 33, 48, 63]


def _codes(block_size, batch, num_edges, ng):
    codes = ck.smallbatch_fw_mma_arms(block_size, batch, num_edges, ng)
    assert codes, (block_size, batch, num_edges)
    return codes


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch forward cannot build here")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("neg_inf", [False, True])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_every_arm_matches_float64(layout, neg_inf, dtype):
    torch.manual_seed(0)
    ng, bs, ne = LAYOUTS[layout]
    dt = DTYPES[dtype]
    for batch in BATCHES:
        node_mars, element_mars, params, nids, ebase, pbase = _layout(ng, bs, ne, batch, dt, neg_inf)
        ref = _reference(element_mars, params, nids, ebase, pbase, bs, ne)
        for code in _codes(bs, batch, ne, ng):
            nm = node_mars.clone()
            ck.smallbatch_forward_sum_mma(nm, element_mars, params, nids, ebase, pbase, batch, bs, ne, code)
            got = nm[7:].double()
            assert not torch.isnan(got).any(), (batch, code)
            assert torch.equal(torch.isneginf(got), torch.isneginf(ref)), (batch, code)
            fin = torch.isfinite(ref)
            # bf16 operands: ~2e-4 relative on the node values (the FMA kernels 1e-7)
            assert ((got - ref).abs() / ref.abs().clamp_min(1.0))[fin].max() < 2e-3, (batch, code)
            assert torch.isnan(nm[:7]).all()                                  # rows before the layer untouched


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch forward cannot build here")
def test_agrees_with_the_fma_kernel():
    torch.manual_seed(1)
    for (ng, bs, ne), batch, dt in itertools.product(LAYOUTS.values(), [2, 7, 16, 40, 63], DTYPES.values()):
        node_mars, element_mars, params, nids, ebase, pbase = _layout(ng, bs, ne, batch, dt, True)
        old = node_mars.clone()
        ck.smallbatch_forward_sum(old, element_mars, params, nids, ebase, pbase, batch, bs, ne, 0)
        for code in _codes(bs, batch, ne, ng):
            new = node_mars.clone()
            ck.smallbatch_forward_sum_mma(new, element_mars, params, nids, ebase, pbase, batch, bs, ne, code)
            a, b = old[7:], new[7:]
            assert torch.equal(torch.isneginf(a), torch.isneginf(b))
            fin = torch.isfinite(a)
            assert ((a - b).abs() / a.abs().clamp_min(1.0))[fin].max() < 2e-3, (ng, bs, ne, batch, dt, code)


def test_arms_follow_the_shape():
    if not _available():
        pytest.skip("the tensor-core small-batch forward cannot build here")
    assert ck.smallbatch_fw_mma_arms(1024, 16, 4096, 4)                      # the HMM-4096 layer
    assert not ck.smallbatch_fw_mma_arms(32, 16, 4096, 4)                    # 32-node blocks: no tile divides them
    assert not ck.smallbatch_fw_mma_arms(1024, 16, 4000, 4)                  # not whole edge tiles
    assert not ck.smallbatch_fw_mma_arms(1024, 64, 4096, 4)                  # the large-batch kernels' regime
    for code in ck.smallbatch_fw_mma_arms(1024, 16, 4096, 4):
        assert (code >> 8) in (0, 1, 2, 4, 8)


class _Forced:
    """`autotune` with the small-batch gate's pick forced to a tensor-core code."""

    def __init__(self, real, code):
        self.real, self.code = real, code

    def __getattr__(self, name):
        return getattr(self.real, name)

    def _gate(self, key):
        return isinstance(key, tuple) and key and key[0] == "sum_fw_smallbatch_cuda"

    def cached(self, key):
        return ("cuda_mma", self.code) if self._gate(key) else self.real.cached(key)

    def pick(self, key, candidates, bench, *args, **kwargs):
        if not self._gate(key):
            return self.real.pick(key, candidates, bench, *args, **kwargs)
        assert ("cuda_mma", self.code) in candidates, (self.code, [c for c in candidates if c[0] == "cuda_mma"])
        return ("cuda_mma", self.code)


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch forward cannot build here")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("batch", [1, 5, 16, 40])
def test_dispatch_reaches_it_and_agrees_with_triton(batch, dtype, monkeypatch):
    dt = DTYPES[dtype]
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 6, num_latents = 256, num_emits = 16, homogeneous = True, block_size = 128)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = dt).to(DEV)
    x = torch.randint(0, 16, (batch, 6), device = DEV)
    codes = ck.smallbatch_fw_mma_arms(128, batch, 256, 2)
    saved = SL.FORWARD_SUM_CUDA
    try:
        monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
        ref = pc(x).clone()
        monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", True)
        calls = []
        real = ck.smallbatch_forward_sum_mma

        def spy(*args, **kwargs):
            calls.append(kwargs["code"] if "code" in kwargs else args[9]); return real(*args, **kwargs)
        monkeypatch.setattr(SL.cuda_kernels, "smallbatch_forward_sum_mma", spy)
        for code in codes:
            calls.clear()
            monkeypatch.setattr(SL, "autotune", _Forced(autotune, code))
            got = pc(x)
            assert calls and all(c == code for c in calls), (code, calls)
            assert ((got - ref).abs() / ref.abs().clamp_min(1.0)).max() < 2e-3, (batch, dtype, code)
    finally:
        SL.FORWARD_SUM_CUDA = saved
