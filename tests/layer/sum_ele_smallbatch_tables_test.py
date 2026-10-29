"""
The small-batch CUDA element backward (c/smallbatch_ele_backward.cu) on layers of SEVERAL parent blocks, which it
addresses through the layer's per-tile parent / parameter tables: the dispatch offers it there (it used to require the
tiles to chain across the whole row, which no such layer satisfies), and forced to each config it gives the Triton
path's element flows, at batches 1 to 15, fp32 and bf16 parameters, on HMMs of four parent blocks (block 256 of 1024
latents; block 64 of 256) and of one. Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

import pyjuice as juice
from pyjuice.layer import sum_layer as SL
from pyjuice.layer.kernels import autotune
from pyjuice.layer.kernels import c as ck

DEV = torch.device("cuda:0")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _available():
    return torch.cuda.is_available() and ck.smallbatch_ele_is_available()


def _hmm(latents, block_size, dt, T = 5):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = latents, num_emits = 16, homogeneous = True,
                              block_size = block_size)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = dt).to(DEV), T


class _ForceEle:
    """`autotune.best_of` with the small-batch element-backward pick forced to a config (or to Triton)."""

    def __init__(self, real, choice):
        self.real, self.choice, self.seen = real, choice, []

    def __getattr__(self, name):
        return getattr(self.real, name)

    def best_of(self, candidates, *args, **kwargs):
        keys = [k for k, _ in candidates]
        if keys and keys[0] == ("triton", -1) and any(k[0] == "cuda" for k in keys[1:]):
            self.seen.append(keys)
            assert self.choice in keys, (self.choice, keys)
            return self.choice
        return self.real.best_of(candidates, *args, **kwargs)


def _flows(pc, x, monkeypatch, choice):
    monkeypatch.setattr(SL, "autotune", _ForceEle(autotune, choice))
    for lg in pc.inner_layer_groups:
        for layer in lg:
            if hasattr(layer, "_cached_bk_ele_choice"):
                layer._cached_bk_ele_choice.clear()                   # the forced pick is re-made
    pc(x)
    pc.backward(x, compute_param_flows = False)
    return pc.element_flows.detach().clone(), SL.autotune


@pytest.mark.skipif(not _available(), reason = "the small-batch CUDA element backward cannot build here")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("shape", ["L1024-b256", "L256-b64", "L256-b256"])
def test_offered_on_several_parent_blocks_and_matches_triton(shape, dtype, monkeypatch):
    latents, block = {"L1024-b256": (1024, 256), "L256-b64": (256, 64), "L256-b256": (256, 256)}[shape]
    pc, T = _hmm(latents, block, DTYPES[dtype])
    n_cfg = len(ck.smallbatch_ele_configs())
    for B in (1, 3, 8, 15):
        x = torch.randint(0, 16, (B, T), device = DEV)
        ref, forced = _flows(pc, x, monkeypatch, ("triton", -1))
        assert forced.seen, "the small-batch element backward was not offered"
        for cfg in range(n_cfg):
            got, _ = _flows(pc, x, monkeypatch, ("cuda", cfg))
            fin = torch.isfinite(ref)
            assert torch.equal(torch.isfinite(got), fin), (B, cfg)
            assert (got[fin] - ref[fin]).abs().max() < 1e-3, (B, cfg)
    # the tables it took: one per tile, the layer's own
    layers = [l for g in pc.inner_layer_groups if g.is_sum() for l in g]
    sb = [v for l in layers for v in getattr(l, "_cached_bk_ele_sb", {}).values()]
    assert sb and all(ok and e.dim() == 2 for e, _, ok in sb)
