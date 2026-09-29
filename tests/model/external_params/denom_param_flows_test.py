"""
Regression tests for the `denom_param_flows` infrastructure -- the optional second parameter-flow
buffer `F-` that a layer requests when its M-step is a conditional dual-flow update
`theta <- normalize(theta * F+ / F-)` (see `ExternalSumParams.requests_denom_param_flows`).

These pin the generic PLUMBING, independent of the F- math (which a requesting layer's own backward
writes, and which is validated where that layer is tested):

  * the request signal defaults False on every descriptor and every layer, so an ordinary PC is
    untouched and pays nothing;
  * `pc.denom_param_flows` is allocated only when some layer requests it, mirrors `param_flows`
    exactly (shape / dtype / device), and rides the same zero / `flows_memory`-scale cadence;
  * a backward threads `denom_param_flows` from the PC through `layer.backward` to the descriptor's
    `post_backward_layer`, while the plain and ungated paths keep threading `None`.

`BlockScaleSumParams(apply_z_correction = True)` is the one shipped requester. Its correction math is
not implemented yet (the descriptor's backward raises), so the single test that needs a live non-None
buffer to reach `post_backward_layer` isolates the PASS-THROUGH by stubbing the descriptor's two
backward hooks; the F- math itself is validated separately once it lands.
"""

import math
import os

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
from pyjuice.nodes.external_params.external_params import ExternalSumParams
from pyjuice.nodes.external_params.lowrank import LowRankSumParams
from pyjuice.layer.layer import Layer
from pyjuice.layer.sum_layer import SumLayer
from pyjuice.layer.prod_layer import ProdLayer
from pyjuice.layer.input_layer import InputLayer
from pyjuice.layer.external_sum_layer import ExternalParamsSumLayer


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_CATS = 5


def _build(gated = None, block_size = 4, n_blocks = 2, seed = 0):
    """Small PC. `gated=None` -> plain sum layer; else a BlockScale gate with `apply_z_correction=gated`."""
    torch.manual_seed(seed)
    with juice.set_block_size(block_size):
        i0 = inputs(0, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
        i1 = inputs(1, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
        m = multiply(i0, i1)
        if gated is None:
            ns = summate(m, num_node_blocks = n_blocks)
        else:
            ns = summate(m, num_node_blocks = n_blocks,
                         external_params = BlockScaleSumParams(ch_block_size = 2, apply_z_correction = gated))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, ns


def _gate(ns, batch, device):
    return torch.zeros(ns.external_params.tensor_shapes(ns, batch)[0], device = device)


def _ext_layer(pc):
    layers = [l for g in pc.inner_layer_groups for l in g.layers
              if isinstance(l, ExternalParamsSumLayer)]
    assert len(layers) == 1
    return layers[0]


# ---------------------------------------------------------------- step 1: the request signal

def test_request_signal_defaults_false_on_descriptors():
    assert ExternalSumParams.requests_denom_param_flows is False
    assert LowRankSumParams(rank = 4).requests_denom_param_flows is False
    assert BlockScaleSumParams().requests_denom_param_flows is False          # correction off by default


def test_blockscale_requests_denom_iff_correction():
    assert BlockScaleSumParams(apply_z_correction = False).requests_denom_param_flows is False
    assert BlockScaleSumParams(apply_z_correction = True).requests_denom_param_flows is True


def test_every_layer_class_defaults_false():
    for L in (Layer, SumLayer, ProdLayer, InputLayer):
        assert L.requests_denom_param_flows is False


@pytest.mark.parametrize("corr", [False, True])
def test_external_layer_delegates_request_to_descriptor(corr):
    # The layer must READ the descriptor, not fall back to the Layer class default.
    root, _ = _build(gated = corr)
    pc = juice.compile(root, verbose = False)
    assert _ext_layer(pc).requests_denom_param_flows is corr


# ---------------------------------------------------------------- step 2: allocation + lifecycle

def test_plain_pc_allocates_no_denom_buffer():
    root, _ = _build(gated = None)
    pc = juice.compile(root, verbose = False)
    assert pc._requests_denom_param_flows is False
    pc.init_param_flows(flows_memory = 0.0)
    assert pc.param_flows is not None                                          # unchanged
    assert pc.denom_param_flows is None                                        # pays nothing


def test_gated_without_correction_allocates_no_denom_buffer():
    root, _ = _build(gated = False)
    pc = juice.compile(root, verbose = False)
    assert pc._requests_denom_param_flows is False
    pc.init_param_flows(flows_memory = 0.0)
    assert pc.denom_param_flows is None


def test_correction_pc_allocates_denom_matching_param_flows():
    root, _ = _build(gated = True)
    pc = juice.compile(root, verbose = False)
    assert pc._requests_denom_param_flows is True
    pc.init_param_flows(flows_memory = 0.0)
    d, p = pc.denom_param_flows, pc.param_flows
    assert d is not None
    assert d is not p and d.data_ptr() != p.data_ptr()          # a SEPARATE buffer, not an alias of F+
    assert d.shape == p.shape and d.dtype == p.dtype and d.device == p.device
    assert torch.count_nonzero(d) == 0
    # independent storage: writing one must not disturb the other (an alias would fail this)
    d[:] = 5.0
    p[:] = 9.0
    assert bool((pc.denom_param_flows == 5.0).all()) and bool((pc.param_flows == 9.0).all())


def test_denom_rides_zero_and_scale_cadence():
    root, _ = _build(gated = True)
    pc = juice.compile(root, verbose = False)
    pc.init_param_flows(flows_memory = 0.0)

    # zero_param_flows() must zero denom too (it routes through init_param_flows(0.0))
    pc.denom_param_flows[:] = 7.0
    pc.param_flows[:] = 3.0
    pc.zero_param_flows()
    assert torch.count_nonzero(pc.denom_param_flows) == 0
    assert torch.count_nonzero(pc.param_flows) == 0

    # flows_memory scales denom exactly as it scales param_flows
    pc.denom_param_flows[:] = 4.0
    pc.param_flows[:] = 4.0
    pc.init_param_flows(flows_memory = 0.5)
    assert torch.allclose(pc.denom_param_flows, torch.full_like(pc.denom_param_flows, 2.0))
    assert torch.allclose(pc.param_flows, torch.full_like(pc.param_flows, 2.0))


# ---------------------------------------------------------------- step 4: pass-through to the descriptor

@cuda_only
def test_plain_backward_unaffected_by_new_signature():
    dev = torch.device("cuda:0")
    root, _ = _build(gated = None)
    pc = juice.compile(root, verbose = False).to(dev)
    x = torch.randint(0, NUM_CATS, [16, 2], device = dev)
    pc(x)
    pc.backward(x, flows_memory = 1.0)                                          # threads None, no error
    assert pc.denom_param_flows is None
    p0 = pc.params.detach().clone()
    pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    assert torch.isfinite(pc.params).all() and not torch.allclose(p0, pc.params)


@cuda_only
def test_gated_off_threads_none_to_post_backward_layer():
    dev = torch.device("cuda:0")
    root, ns = _build(gated = False)
    pc = juice.compile(root, verbose = False).to(dev)
    x = torch.randint(0, NUM_CATS, [16, 2], device = dev)
    phi = _gate(ns, 16, dev)

    seen = {}
    orig = BlockScaleSumParams.post_backward_layer
    def cap(self, *a, **k):
        seen["denom"] = k.get("denom_param_flows", "MISSING")
        return orig(self, *a, **k)
    BlockScaleSumParams.post_backward_layer = cap
    try:
        pc(x, sum_external_params = {ns: phi})
        pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True, flows_memory = 1.0)
    finally:
        BlockScaleSumParams.post_backward_layer = orig

    assert seen["denom"] is None                       # kwarg threaded; None because it was not requested
    assert pc.denom_param_flows is None


@cuda_only
def test_correction_on_threads_the_denom_buffer_to_post_backward_layer():
    """Step-4 pass-through in ISOLATION: stub the descriptor's two backward hooks so the check depends
    only on the plumbing (not on the F- math, which its own tests below cover) -- the allocated
    `denom_param_flows` buffer, the exact PC object, must reach `post_backward_layer`."""
    dev = torch.device("cuda:0")
    root, ns = _build(gated = True)
    pc = juice.compile(root, verbose = False).to(dev)
    x = torch.randint(0, NUM_CATS, [16, 2], device = dev)
    phi = _gate(ns, 16, dev)

    seen = {}
    orig_pre, orig_post = BlockScaleSumParams.pre_backward_layer, BlockScaleSumParams.post_backward_layer
    BlockScaleSumParams.pre_backward_layer = lambda self, *a, **k: None         # skip the WIP raise + setup
    def cap(self, *a, **k):
        seen["denom"] = k.get("denom_param_flows", "MISSING")
        return None
    BlockScaleSumParams.post_backward_layer = cap
    try:
        assert pc.denom_param_flows is None                                    # lazy: allocated in backward
        pc(x, sum_external_params = {ns: phi})
        pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True, flows_memory = 1.0)
    finally:
        BlockScaleSumParams.pre_backward_layer = orig_pre
        BlockScaleSumParams.post_backward_layer = orig_post

    d = seen.get("denom")
    assert torch.is_tensor(d)
    assert d is pc.denom_param_flows                                           # the exact allocated buffer
    assert d.shape == pc.param_flows.shape


def _build_mixed(block_size = 4, n_blocks = 2, seed = 0):
    """Two sum layers at the SAME depth and block size -- one plain, one gated (correction off) -- so
    they land in ONE `LayerGroup`, and the group's backward hands the `denom_param_flows` kwarg to the
    plain `SumLayer` too."""
    torch.manual_seed(seed)
    with juice.set_block_size(block_size):
        i0 = inputs(0, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
        i1 = inputs(1, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
        m = multiply(i0, i1)
        ns_plain = summate(m, num_node_blocks = n_blocks)
        ns_gated = summate(m, num_node_blocks = n_blocks,
                           external_params = BlockScaleSumParams(ch_block_size = 2))
        root = summate(multiply(ns_plain), multiply(ns_gated), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, ns_gated


@cuda_only
def test_mixed_gated_and_plain_sum_layers_in_one_group():
    dev = torch.device("cuda:0")
    root, ns_gated = _build_mixed()
    pc = juice.compile(root, verbose = False).to(dev)

    # exactly one sum group holds BOTH a plain SumLayer and an ExternalParamsSumLayer
    sum_groups = [g for g in pc.inner_layer_groups if g.is_sum()]
    mixed = [g for g in sum_groups
             if any(isinstance(l, ExternalParamsSumLayer) for l in g.layers)
             and any(not isinstance(l, ExternalParamsSumLayer) for l in g.layers)]
    assert len(mixed) == 1, [[type(l).__name__ for l in g.layers] for g in sum_groups]

    x = torch.randint(0, NUM_CATS, [16, 2], device = dev)
    phi = _gate(ns_gated, 16, dev)
    pc(x, sum_external_params = {ns_gated: phi})
    # the plain SumLayer in the group also receives `denom_param_flows` (via **kwargs) -- must not choke
    pc.backward(x, sum_external_params = {ns_gated: phi}, logspace_flows = True, flows_memory = 1.0)
    assert pc.denom_param_flows is None                # correction off -> not requested


# ------------------------------------------------------ step 5: the F- accumulation (apply_z_correction)

def _phi(ns, batch, dev, scale = 1.5):
    return torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], device = dev) * scale


def _corr_flows(pc, ns, x, phi, ref = False):
    """One gated fwd+bwd with `apply_z_correction`; returns (F+, F-) fresh for this batch.
    `ref=True` forces the torch reference `_accumulate_denom_torch` via the env switch."""
    os.environ["PYJUICE_BLOCKSCALE_DENOM_REF"] = "1" if ref else "0"
    try:
        pc(x, sum_external_params = {ns: phi})
        pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True, flows_memory = 0.0)
    finally:
        os.environ.pop("PYJUICE_BLOCKSCALE_DENOM_REF", None)
    return pc.param_flows.clone(), pc.denom_param_flows.clone()


@cuda_only
def test_denom_accumulation_matches_finite_differences():
    """The decisive check: `F+ - F-` equals the exact gradient d(sum_b log P(x_b)) / d log theta[n,c],
    validated numerically. This is what makes `F-` the right denominator for the conditional M-step
    `theta <- normalize(theta * F+ / F-)` -- and it shares no indexing with the kernel."""
    dev = torch.device("cuda:0")
    root, ns = _build(gated = True)                                   # apply_z_correction = True
    pc = juice.compile(root, verbose = False).to(dev)
    lay = pc.external_params_nodes[ns]
    B = 32
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)
    Fp, Fm = _corr_flows(pc, ns, x, phi)

    pids, pfids, cids = lay.partitioned_pids[0], lay.partitioned_pfids[0], lay.partitioned_cids[0]
    rows, E = pids.shape

    def ll():
        with torch.no_grad():                                         # perturb theta, no autograd graph
            return pc(x, sum_external_params = {ns: phi}).double().sum().item()

    eps, worst, n = 1e-2, 0.0, 0
    for r in range(min(rows, 2)):
        for e in range(min(E, 4)):
            if int(cids[r, e]) == 0:
                continue
            for m in (0, 2):
                pid, pf = int(pids[r, e]) + m, int(pfids[r, e]) + m
                g_an = float(Fp[pf] - Fm[pf])
                with torch.no_grad():
                    o = float(pc.params[pid]); pc.params[pid] = o * math.exp(eps); lp = ll()
                    pc.params[pid] = o * math.exp(-eps); lm = ll(); pc.params[pid] = o
                g_fd = (lp - lm) / (2 * eps)
                worst = max(worst, abs(g_fd - g_an) / max(abs(g_fd), abs(g_an), 1e-6))
                n += 1
    assert n > 0
    assert worst < 5e-2, worst


@cuda_only
def test_denom_kernel_matches_torch_reference():
    """The shipped Triton kernel must agree with the finite-difference-validated torch reference."""
    dev = torch.device("cuda:0")
    root, ns = _build(gated = True)
    pc = juice.compile(root, verbose = False).to(dev)
    B = 32
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)

    Fp_k, Fm_k = _corr_flows(pc, ns, x, phi, ref = False)             # Triton kernel
    Fp_r, Fm_r = _corr_flows(pc, ns, x, phi, ref = True)             # torch reference
    assert torch.allclose(Fp_k, Fp_r)                                # numerator unaffected by denom path
    assert torch.count_nonzero(Fm_k) > 0
    rel = ((Fm_k - Fm_r).abs() / (Fm_r.abs() + 1e-6)).max().item()
    assert rel < 1e-4, rel


@cuda_only
def test_denom_conservation_per_node():
    """`sum_c F+ == sum_c F-` for every node (both == sum_b f_b): the correction only REALLOCATES a
    node's mass across its gates, so a one-gate node is an exact no-op."""
    dev = torch.device("cuda:0")
    root, ns = _build(gated = True)
    pc = juice.compile(root, verbose = False).to(dev)
    lay = pc.external_params_nodes[ns]
    B = 32
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)
    Fp, Fm = _corr_flows(pc, ns, x, phi)
    pids, pfids, cids = lay.partitioned_pids[0], lay.partitioned_pfids[0], lay.partitioned_cids[0]
    rows, E = pids.shape
    worst = 0.0
    for r in range(rows):
        for m in range(4):
            sp = sum(float(Fp[int(pfids[r, e]) + m]) for e in range(E) if int(cids[r, e]) != 0)
            sm = sum(float(Fm[int(pfids[r, e]) + m]) for e in range(E) if int(cids[r, e]) != 0)
            worst = max(worst, abs(sp - sm))
    assert worst < 1e-3, worst


@cuda_only
def test_apply_z_correction_node_axis_gate_raises():
    """`F-` reweights gate MASSES per node block, so a gate finer than the block along the NODE axis is
    refused (as `d LL / d log phi` is) -- and BEFORE `node_mars` is perturbed."""
    dev = torch.device("cuda:0")
    with juice.set_block_size(8):
        i0 = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        i1 = inputs(1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        ns = summate(multiply(i0, i1), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(block_size = 4, apply_z_correction = True))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)
    B = 16
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = torch.zeros(ns.external_params.tensor_shapes(ns, B)[0], device = dev)
    pc(x, sum_external_params = {ns: phi})
    with pytest.raises(NotImplementedError, match = "NODE axis"):
        pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True, flows_memory = 1.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
