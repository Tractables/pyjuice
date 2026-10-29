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


class _Recorder:
    """`autotune` recording the small-batch gate's keys and candidate lists (the real pick runs)."""

    def __init__(self, real):
        self.real, self.seen = real, []

    def __getattr__(self, name):
        return getattr(self.real, name)

    def cached(self, key):
        if isinstance(key, tuple) and key and key[0] == "sum_fw_smallbatch_cuda":
            return None                                                  # always reach the pick
        return self.real.cached(key)

    def pick(self, key, candidates, bench, *args, **kwargs):
        if isinstance(key, tuple) and key and key[0] == "sum_fw_smallbatch_cuda":
            self.seen.append((key, list(candidates)))
        return self.real.pick(key, candidates, bench, *args, **kwargs)


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch forward cannot build here")
@pytest.mark.parametrize("precision", ["auto", "tf32", "fp32"])
def test_offered_only_at_the_default_precision(precision, monkeypatch):
    """The tensor-core arms round the operands to bf16: like the other tensor-core forwards they are candidates under
    `precision = "auto"` only, and the choice is keyed on it ("tf32" and "auto" share the rest of the key here)."""
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = 4, num_latents = 256, num_emits = 16, homogeneous = True, block_size = 128)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, precision = precision).to(DEV)
    rec = _Recorder(autotune)
    monkeypatch.setattr(SL, "autotune", rec)
    pc(torch.randint(0, 16, (8, 4), device = DEV))
    assert rec.seen, "the small-batch gate was not reached"
    offered = any(c[0] == "cuda_mma" for _, cands in rec.seen for c in cands)
    assert offered == (precision == "auto"), (precision, offered)
    assert all(key[2] == (precision == "auto") for key, _ in rec.seen)


@pytest.mark.skipif(not _available(), reason = "the tensor-core small-batch forward cannot build here")
def test_not_offered_on_misaligned_bf16_parameters(monkeypatch):
    """Its 16-byte parameter loads take 8 bf16: a layer whose parameter range starts at 4 mod 8 (compilation packs the
    ranges without padding) must not take it -- and the forward agrees with the same values stored in fp32."""
    from pyjuice.nodes import inputs, multiply, summate
    import pyjuice.nodes.distributions as dists
    torch.manual_seed(0)
    C = dists.Categorical(num_cats = 16)
    s0 = summate(inputs(0, num_node_blocks = 3, block_size = 1, dist = C), num_node_blocks = 16, block_size = 4)
    s1 = summate(inputs(1, num_node_blocks = 3, block_size = 1, dist = C), num_node_blocks = 16, block_size = 4)
    t2 = summate(inputs(2, num_node_blocks = 3, block_size = 1, dist = C), num_node_blocks = 1, block_size = 4)
    u3 = summate(inputs(3, num_node_blocks = 2, block_size = 2, dist = C), num_node_blocks = 1, block_size = 4)
    a = summate(multiply(s0, s1), num_node_blocks = 16, block_size = 4)
    b = summate(multiply(t2, u3), num_node_blocks = 16, block_size = 4)
    top = summate(multiply(a, b), num_node_blocks = 4, block_size = 64)
    root = summate(multiply(top), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc16 = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(DEV)
    pc32 = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(DEV).set_param_dtype(torch.float32)
    assert top._param_range[0] % 8 == 4                                  # (the construction holds)
    rec = _Recorder(autotune)
    monkeypatch.setattr(SL, "autotune", rec)
    x = torch.randint(0, 16, (8, 4), device = DEV)
    ll16 = pc16(x)
    keys64 = [(key, cands) for key, cands in rec.seen if key[4] == 64]   # (name, num_edges, mma_ok, kernel, block_size, ...)
    assert keys64, "the 64-node layer did not reach the small-batch gate"
    assert all(not key[2] and not any(c[0] == "cuda_mma" for c in cands) for key, cands in keys64)
    ll32 = pc32(x)
    assert torch.allclose(ll16, ll32, atol = 1e-3, rtol = 1e-4), float((ll16 - ll32).abs().max())
