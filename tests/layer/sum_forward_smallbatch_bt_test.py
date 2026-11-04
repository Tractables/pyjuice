"""
The batch-tiled small-batch CUDA sum forward (c/smallbatch_forward_sum_bt.cu: the parameters loaded once per BT batch
columns) against a float64 reference and the per-sample small-batch kernel, on block-sparse layouts of 32-, 64-, 128-
and 1024-node blocks at batches 1..63 (tile tails included), with -inf children (a leading run, a whole column), at fp32
and bf16 parameters; and the layer's dispatch, told the gate chose a batch-tiled config, reaches it on a whole launch
and on runs and agrees with the Triton forward. Skipped where the CUDA kernels cannot build.
"""
import itertools
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import sum_layer as SL
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import c as ck

DEV = torch.device("cuda:0")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _available():
    return torch.cuda.is_available() and ck.smallbatch_fw_is_available() and ck.smallbatch_fw_bt_is_available()


def _layout(ng, block_size, num_edges, batch, dt, neg_inf):
    """ng node blocks (nodes nids[nb] + m), each with num_edges contiguous children (ebase[nb] + e) and parameters
    pbase[nb] + e * block_size + m; a few unrelated rows before the layer's."""
    node_mars = torch.full((ng * block_size + 7, batch), float("nan"), device = DEV)
    element_mars = torch.randn(ng * num_edges + 3, batch, device = DEV) * 3
    if neg_inf:
        element_mars[torch.rand(element_mars.shape, device = DEV) < 0.3] = -float("inf")
        element_mars[3:3 + num_edges // 2, 0] = -float("inf")                 # block 0, column 0: a leading -inf run
        element_mars[3 + num_edges // 2:3 + num_edges, 0] = -1.0
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
        e = em[ebase[nb]:ebase[nb] + num_edges]                                             # [E, B]
        w = p[pbase[nb]:pbase[nb] + num_edges * block_size].view(num_edges, block_size)      # [E, M]
        mx = e.max(dim = 0, keepdim = True).values
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        out.append(torch.log(w.t() @ torch.exp(e - mx)) + mx)                               # [M, B]
    return torch.cat(out)


LAYOUTS = {"3x32x64": (3, 32, 64), "2x128x256": (2, 128, 256), "1x1024x512": (1, 1024, 512), "4x64x48": (4, 64, 48)}
BATCHES = [1, 2, 3, 5, 8, 15, 16, 17, 33, 48, 63]


@pytest.mark.skipif(not _available(), reason = "the small-batch CUDA forward kernels cannot build here")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("neg_inf", [False, True])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_every_config_matches_float64(layout, neg_inf, dtype):
    torch.manual_seed(0)
    ng, bs, ne = LAYOUTS[layout]
    dt = DTYPES[dtype]
    for batch in BATCHES:
        node_mars, element_mars, params, nids, ebase, pbase = _layout(ng, bs, ne, batch, dt, neg_inf)
        ref = _reference(element_mars, params, nids, ebase, pbase, bs, ne)
        for cfg in range(len(ck.smallbatch_fw_bt_configs())):
            nm = node_mars.clone()
            ck.smallbatch_forward_sum_bt(nm, element_mars, params, nids, ebase, pbase, batch, bs, ne, cfg)
            got = nm[7:].double()
            assert not torch.isnan(got).any(), (batch, cfg)
            assert torch.equal(torch.isneginf(got), torch.isneginf(ref)), (batch, cfg)
            fin = torch.isfinite(ref)
            assert ((got - ref).abs() / ref.abs().clamp_min(1.0))[fin].max() < 2e-5, (batch, cfg)
            assert torch.isnan(nm[:7]).all()                                   # rows before the layer untouched


@pytest.mark.skipif(not _available(), reason = "the small-batch CUDA forward kernels cannot build here")
def test_agrees_with_the_per_sample_kernel():
    """The two kernels are the autotuner's arms for one launch: they must agree to the ~1e-6 the arms already differ."""
    torch.manual_seed(1)
    for (ng, bs, ne), batch, dt in itertools.product(LAYOUTS.values(), [1, 4, 7, 16, 40], DTYPES.values()):
        node_mars, element_mars, params, nids, ebase, pbase = _layout(ng, bs, ne, batch, dt, True)
        old = node_mars.clone()
        ck.smallbatch_forward_sum(old, element_mars, params, nids, ebase, pbase, batch, bs, ne, 0)
        for cfg in range(len(ck.smallbatch_fw_bt_configs())):
            new = node_mars.clone()
            ck.smallbatch_forward_sum_bt(new, element_mars, params, nids, ebase, pbase, batch, bs, ne, cfg)
            a, b = old[7:], new[7:]
            assert torch.equal(torch.isneginf(a), torch.isneginf(b))
            fin = torch.isfinite(a)
            assert (a[fin] - b[fin]).abs().max() < 1e-4, (ng, bs, ne, batch, dt, cfg)


class _Forced:
    """`autotune` with the small-batch gate's pick forced to a batch-tiled config."""

    def __init__(self, real, cfg):
        self.real, self.cfg, self.picked = real, cfg, []

    def __getattr__(self, name):
        return getattr(self.real, name)

    def _gate(self, key):
        return isinstance(key, tuple) and key and key[0] == "sum_fw_smallbatch_cuda"

    def cached(self, key):
        return ("cuda_bt", self.cfg) if self._gate(key) else self.real.cached(key)

    def pick(self, key, candidates, bench, *args, **kwargs):
        if not self._gate(key):
            return self.real.pick(key, candidates, bench, *args, **kwargs)
        assert ("cuda_bt", self.cfg) in candidates
        self.picked.append(key)
        return ("cuda_bt", self.cfg)


def _hmm(latents, dt, T = 6):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = latents, num_emits = 16, homogeneous = True,
                              block_size = latents // 2)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = dt).to(DEV), T


@pytest.mark.skipif(not _available(), reason = "the small-batch CUDA forward kernels cannot build here")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("batch", [1, 5, 16, 40])
def test_dispatch_reaches_it_and_agrees_with_triton(batch, dtype, monkeypatch):
    dt = DTYPES[dtype]
    pc, T = _hmm(256, dt)
    x = torch.randint(0, 16, (batch, T), device = DEV)
    saved = SL.FORWARD_SUM_CUDA
    try:
        monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", False)
        ref = pc(x).clone()
        monkeypatch.setattr(SL, "FORWARD_SUM_CUDA", True)
        calls = []
        real = ck.smallbatch_forward_sum_bt

        def spy(*args, **kwargs):
            calls.append(args[-1]); return real(*args, **kwargs)
        monkeypatch.setattr(SL.cuda_kernels, "smallbatch_forward_sum_bt", spy)
        for cfg in range(len(ck.smallbatch_fw_bt_configs())):
            calls.clear()
            monkeypatch.setattr(SL, "autotune", _Forced(autotune, cfg))
            got = pc(x)
            assert calls and all(c == cfg for c in calls), (cfg, calls)
            assert (got - ref).abs().max() < 1e-3, (batch, dtype, cfg)
    finally:
        SL.FORWARD_SUM_CUDA = saved
