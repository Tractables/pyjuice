"""Tests for the block-sparse parameter-flow backward optimizations:
  - the collision-free gate that decides atomic vs non-atomic accumulation,
  - the non-atomic read-add-store (RMW) variant matching the atomic kernel,
  - the automatic fallback when the tuned launch exceeds GPU shared memory.
"""
import warnings
import torch
import pytest

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.layer.kernels.sum_backward_param_block_sparse as parmod
from pyjuice.layer.sum_layer import SumLayer
from pyjuice.layer.kernels import autotune


def _build_small_hclt(num_latents=64, num_cats=8, num_vars=16, device="cuda:0"):
    torch.manual_seed(42)
    device = torch.device(device)
    x = torch.randint(0, num_cats, (2000, num_vars), device=device)
    ns = juice.structures.HCLT(x.float(), num_latents=num_latents, num_bins=16, sigma=0.1,
                               chunk_size=16, input_dist=dists.Categorical(num_cats=num_cats))
    pc = juice.compile(ns)
    pc.to(device)
    return pc, device


def _backward(pc, x):
    pc.init_param_flows(flows_memory=0.0)
    pc(x, propagation_alg="LL")
    pc.backward(x, flows_memory=1.0, allow_modify_flows=False,
                propagation_alg="LL", logspace_flows=True)
    return pc.param_flows.clone()


def test_par_flow_collision_free_gate():
    # The gate must accept untied param-flow layouts (distinct, block_size-spaced) and
    # reject colliding ones (duplicate or sub-block-size spacing).
    pc, device = _build_small_hclt()
    layer = [l for g in pc.inner_layer_groups for l in g if l.is_sum() and l.block_size > 1][0]
    bs = layer.block_size

    untied = torch.arange(0, 8 * bs, bs, device=device)
    repeated = torch.tensor([0, 0, bs, 2 * bs], device=device)
    close = torch.arange(0, 8 * (bs // 2), bs // 2, device=device)

    assert layer._par_flow_collision_free(untied)
    assert not layer._par_flow_collision_free(repeated)
    assert not layer._par_flow_collision_free(close)


def test_par_flow_rmw_matches_atomic():
    # For an untied PC the non-atomic RMW kernel (default) must match the atomic kernel.
    pc, device = _build_small_hclt()
    x = torch.randint(0, 8, (64, pc.num_vars), device=device)

    pf_rmw = _backward(pc, x)  # untied -> RMW variant

    orig = SumLayer._par_flow_collision_free
    try:
        SumLayer._par_flow_collision_free = lambda self, pfids: False  # force atomic kernel
        pf_atomic = _backward(pc, x)
    finally:
        SumLayer._par_flow_collision_free = orig

    assert torch.all((pf_rmw - pf_atomic).abs() < 1e-3)


@pytest.mark.parametrize("tuning", [True, False])
def test_par_flow_oom_fallback(tuning):
    # If the tuned launch raises OutOfResources (simulating a smaller GPU), the backward must
    # transparently avoid that configuration and stay correct. Two mechanisms cover this, and both
    # are exercised here: with the autotuner ON, the offending config simply loses the benchmark
    # (`OutOfResources` is raised at compile time, so the candidate is skipped); with it OFF, the
    # launch itself raises and `_par_tuning_oom` triggers a retry on the untuned config.
    from triton.runtime.errors import OutOfResources

    pc, device = _build_small_hclt()
    x = torch.randint(0, 8, (64, pc.num_vars), device=device)
    ref = _backward(pc, x)

    real_rmw = parmod._bk_triton_block_sparse_par_kernel_rmw

    class OOMProxy:
        def __getitem__(self, grid):
            realk = real_rmw[grid]
            def run(*a, **k):
                if k.get("num_warps", None) == 8:           # the tuned launch
                    raise OutOfResources(200000, 101376, "shared memory")
                return realk(*a, **k)
            return run

    for g in pc.inner_layer_groups:
        for l in g:
            if l.is_sum() and hasattr(l, "_par_tuning_oom"):
                del l._par_tuning_oom

    was_enabled, cache = autotune.ENABLED, dict(autotune._CACHE)
    try:
        # A fresh cache, so the proxy is actually benchmarked rather than served a prior choice.
        autotune.ENABLED, autotune._CACHE = tuning, dict()
        parmod._bk_triton_block_sparse_par_kernel_rmw = OOMProxy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = _backward(pc, x)    # must avoid the failing config, not crash
            got2 = _backward(pc, x)   # cached choice / cached fallback path
    finally:
        parmod._bk_triton_block_sparse_par_kernel_rmw = real_rmw
        autotune.ENABLED, autotune._CACHE = was_enabled, cache

    assert torch.all((got - ref).abs() < 1e-3)
    assert torch.all((got2 - ref).abs() < 1e-3)
    if not tuning:
        n_oom = sum(1 for g in pc.inner_layer_groups for l in g
                    if l.is_sum() and getattr(l, "_par_tuning_oom", False))
        assert n_oom >= 1


if __name__ == "__main__":
    test_par_flow_collision_free_gate()
    test_par_flow_rmw_matches_atomic()
    test_par_flow_oom_fallback(True)
    test_par_flow_oom_fallback(False)
