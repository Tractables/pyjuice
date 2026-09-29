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


# ------------------------------------------------------ step 6: the conditional dual-flow M-step

def _build_one_gate(corr, seed = 0):
    """One node block, one child block -> ONE edge block -> one gate (an exact no-op gate)."""
    torch.manual_seed(seed)
    with juice.set_block_size(4):
        i0 = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        i1 = inputs(1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        ns = summate(multiply(i0, i1), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(apply_z_correction = corr))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, ns


@cuda_only
def test_one_gate_correction_matches_standard_em():
    """With one gate the correction is an exact no-op, so the dual M-step must equal pyjuice's standard
    M-step (correction off) -- validated against the trusted standard update, not a re-derivation."""
    import warnings
    dev = torch.device("cuda:0")
    B = 64
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                              # the "single gate is a no-op" warning
        r_off, ns_off = _build_one_gate(corr = False); pc_off = juice.compile(r_off, verbose = False).to(dev)
        r_on, ns_on = _build_one_gate(corr = True); pc_on = juice.compile(r_on, verbose = False).to(dev)
        phi = torch.randn(B, ns_off.num_nodes // 4, ns_off.num_ch_nodes // ns_off.ch_block_size,
                          device = dev) * 1.5
        for pc, ns in [(pc_off, ns_off), (pc_on, ns_on)]:
            pc(x, sum_external_params = {ns: phi})
            pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True, flows_memory = 0.0)
            pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    assert torch.allclose(pc_on.params, pc_off.params, atol = 1e-4, rtol = 1e-4), \
        (pc_on.params - pc_off.params).abs().max().item()


@cuda_only
def test_anemone_step_size_rescaling_with_correction_raises():
    """`step_size_rescaling` (Anemone) with `apply_z_correction` must refuse: its top-down pass does not
    yet adjust `denom_param_flows`, so the correction would divide by an inconsistent denominator."""
    dev = torch.device("cuda:0")
    root, ns = _build(gated = True); pc = juice.compile(root, verbose = False).to(dev)
    B = 32
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)
    pc(x, sum_external_params = {ns: phi})
    pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True, flows_memory = 1.0)
    with pytest.raises(NotImplementedError, match = "step_size_rescaling"):
        pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1, step_size_rescaling = True)


@cuda_only
def test_corrected_em_is_monotone_under_a_live_gate():
    """Exact EM (pseudocount 0) with the correction on a live multi-gate must not decrease the train LL."""
    dev = torch.device("cuda:0")
    root, ns = _build(gated = True, seed = 1); pc = juice.compile(root, verbose = False).to(dev)
    B = 64
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)
    lls = []
    for _ in range(6):
        lls.append(pc(x, sum_external_params = {ns: phi}).mean().item())
        pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True, flows_memory = 0.0)
        pc.mini_batch_em(step_size = 1.0, pseudocount = 0.0)
    assert min(lls[i + 1] - lls[i] for i in range(len(lls) - 1)) > -1e-3, lls
    assert torch.isfinite(pc.params).all()


# ------------------------------------------------- F- across the shape space
#
# The tests above all run ONE shape (block_size 4, 2 node blocks, ch_block_size 2, batch 32), which
# leaves most of `_bs_triton_denom_kernel` unexercised: batch 32 is a single batch tile, so the
# kernel's online-max rescaling never runs; 2 node blocks is a power of two, so the gate table is
# never narrower than the (power-of-two padded) edge count; and `log phi ~ N(0, 1.5)` never
# approaches the range where `exp` would overflow. Each of those is a place a kernel goes wrong
# silently, so they get their own coverage here.

def _build_shape(block_size, n_blocks, ch_block_size, seed = 0):
    """A gated PC with `apply_z_correction`, with every shape axis of the kernel exposed."""
    torch.manual_seed(seed)
    with juice.set_block_size(block_size):
        i0 = inputs(0, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
        i1 = inputs(1, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
        ns = summate(multiply(i0, i1), num_node_blocks = n_blocks,
                     external_params = BlockScaleSumParams(ch_block_size = ch_block_size,
                                                           apply_z_correction = True))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, ns


def _run(pc, ns, x, phi, ref = False):
    """One gated fwd+bwd; returns `(F+, F-, sum_b f_b per node)`.

    `allow_modify_flows = False` is REQUIRED for the third return value: otherwise the backward
    overwrites `node_flows` in place with the `log f - log m` form and the "flow" read back is a
    different quantity entirely.
    """
    os.environ["PYJUICE_BLOCKSCALE_DENOM_REF"] = "1" if ref else "0"
    try:
        pc(x, sum_external_params = {ns: phi})
        pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True,
                    flows_memory = 0.0, allow_modify_flows = False)
    finally:
        os.environ.pop("PYJUICE_BLOCKSCALE_DENOM_REF", None)

    lay = pc.external_params_nodes[ns]
    bs = ns.block_size
    ar = torch.arange(bs, device = pc.params.device)
    gid = torch.cat([(lay.partitioned_nids[p].long()[:, None] + ar[None, :]).reshape(-1)
                     for p in range(len(lay.partitioned_nids))])
    flow_sum = pc.node_flows[gid].double().exp().sum(-1)
    return pc.param_flows.clone(), pc.denom_param_flows.clone(), flow_sum


def _sum_over_children(pc, ns, F):
    """`sum_c F[n,c]` per node, accumulated over every partition (a node's children can split)."""
    lay = pc.external_params_nodes[ns]
    bs = ns.block_size
    ar = torch.arange(bs, device = F.device)
    parts = range(len(lay.partitioned_pfids))
    gmin = min(int(lay.partitioned_nids[p].min()) for p in parts)
    out = torch.zeros(ns.num_node_blocks * bs, device = F.device, dtype = torch.float64)
    for p in parts:
        pfids, cids = lay.partitioned_pfids[p], lay.partitioned_cids[p]
        nids = lay.partitioned_nids[p].long()
        rows, E = pfids.shape
        idx = (pfids[:, None, :] + ar[None, :, None]).long()
        real = (cids != 0)[:, None, :].expand(rows, bs, E)
        loc = (nids[:, None] + ar[None, :] - gmin).reshape(-1)
        out.index_add_(0, loc, (F[idx].double() * real).sum(-1).reshape(-1))
    return out


# block_size, n_blocks, ch_block_size, batch
_SHAPES = [
    (2,  2, 1,  32),     # narrowest block
    (4,  2, 2,  96),     # > 1 batch tile
    (4,  2, 2,  65),     # partial batch tile with a single live lane
    (4,  3, 2,  96),     # RAGGED: 3 node blocks -> the gate table is narrower than padded num_edges
    (4,  5, 2,  64),     # ragged again, wider
    (8,  2, 2, 128),
    (8,  4, 4, 257),     # partial tile, several node blocks, coarse gate
    (16, 2, 2,  64),
    (16, 8, 2,  96),
    (32, 2, 4,  33),     # partial tile far smaller than the tile size
]


@cuda_only
@pytest.mark.parametrize("block_size,n_blocks,ch_block_size,batch", _SHAPES)
def test_denom_kernel_matches_reference_across_shapes(block_size, n_blocks, ch_block_size, batch):
    """The Triton `F-` and the torch reference must agree at every shape, not just the one the
    original tests used. The ragged rows (`n_blocks` 3 and 5) are the regression for the reference's
    unbounded gate-column gather: the compiled `num_edges` is padded to a power of two while the
    gate table is only as wide as the widest row's edge-block count, so `e // node_cbs` ran PAST the
    table and raised a device-side assert (the kernel already clamped, only the reference did not).
    """
    dev = torch.device("cuda:0")
    root, ns = _build_shape(block_size, n_blocks, ch_block_size)
    pc = juice.compile(root, verbose = False).to(dev)
    torch.manual_seed(3)
    x = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    phi = _phi(ns, batch, dev)

    _, Fm_k, _ = _run(pc, ns, x, phi, ref = False)
    _, Fm_r, _ = _run(pc, ns, x, phi, ref = True)
    assert torch.isfinite(Fm_k).all()
    assert torch.count_nonzero(Fm_k) > 0
    rel = ((Fm_k - Fm_r).abs() / (Fm_r.abs() + 1e-6)).max().item()
    assert rel < 1e-4, rel


@cuda_only
@pytest.mark.parametrize("batch", [32, 64, 65, 96, 128, 257])
def test_denom_conserves_node_flow_across_batch_tiling(batch):
    """`sum_c F-[n,c] == sum_b f_b[n]`, because `sum_c theta_b[n,c] == 1` for every sample.

    Reference-free, so it cannot be satisfied by the kernel and the reference being wrong together
    -- and it is exactly what a dropped or double-counted batch tile breaks. `F-`'s launcher uses
    `cdiv` and the kernel re-masks the batch each iteration, which is what the non-multiples of the
    64-wide batch tile (65, 96, 257) pin here.
    """
    dev = torch.device("cuda:0")
    root, ns = _build_shape(4, 2, 2)
    pc = juice.compile(root, verbose = False).to(dev)
    torch.manual_seed(3)
    x = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    phi = _phi(ns, batch, dev)

    _, Fm, flow_sum = _run(pc, ns, x, phi)
    got = _sum_over_children(pc, ns, Fm)
    keep = flow_sum.abs() > 1e-8
    rel = ((got - flow_sum).abs()[keep] / flow_sum.abs()[keep]).max().item()
    assert rel < 1e-4, f"F- does not conserve node flow at batch={batch} (relmax={rel})"


@cuda_only
@pytest.mark.parametrize("scale", [0.0, 20.0, 90.0, 300.0])
def test_denom_survives_extreme_gate_logits(scale):
    """`log phi` is a router logit and therefore UNBOUNDED, which is the trap that already bit the
    log-Z half of `d LL / d log phi` (there `exp(nf - log Z)` underflowed to 0 past ~88 and the term
    silently vanished with no inf/NaN to show for it).

    `F-` is built to be safe by construction -- `log phi` is kept INSIDE the exponent, where it
    cancels against `log Z`, so the summand never exceeds `f_b[n]` however large the logit. This
    pins that: the result stays finite AND still conserves, at logits far past the overflow point.
    `scale = 0` is the neutral gate at the other end.
    """
    dev = torch.device("cuda:0")
    root, ns = _build_shape(4, 2, 2)
    pc = juice.compile(root, verbose = False).to(dev)
    torch.manual_seed(3)
    x = torch.randint(0, NUM_CATS, [96, 2], device = dev)
    phi = _phi(ns, 96, dev, scale = scale)

    _, Fm, flow_sum = _run(pc, ns, x, phi)
    assert torch.isfinite(Fm).all(), f"non-finite F- at gate scale {scale}"
    got = _sum_over_children(pc, ns, Fm)
    keep = flow_sum.abs() > 1e-8
    rel = ((got - flow_sum).abs()[keep] / flow_sum.abs()[keep]).max().item()
    assert rel < 1e-3, f"F- lost mass at gate scale {scale} (relmax={rel})"


@cuda_only
def test_denom_matches_finite_differences_at_a_larger_shape():
    """`F+ - F- == d(sum_b log P(x_b)) / d log theta` at a shape with several batch tiles and a
    coarser gate -- the original FD test ran only the single-tile shape."""
    dev = torch.device("cuda:0")
    root, ns = _build_shape(8, 2, 4)
    pc = juice.compile(root, verbose = False).to(dev)
    lay = pc.external_params_nodes[ns]
    B = 96
    torch.manual_seed(3)
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)
    Fp, Fm, _ = _run(pc, ns, x, phi)

    pids, pfids, cids = lay.partitioned_pids[0], lay.partitioned_pfids[0], lay.partitioned_cids[0]

    def ll():
        with torch.no_grad():
            return pc(x, sum_external_params = {ns: phi}).double().sum().item()

    eps, n, dmax, gmax = 1e-2, 0, 0.0, 0.0
    for e in range(0, pids.size(1), max(1, pids.size(1) // 5)):
        if int(cids[0, e]) == 0:
            continue
        for m in (0, ns.block_size // 2):
            pid, pf = int(pids[0, e]) + m, int(pfids[0, e]) + m
            g_an = float(Fp[pf] - Fm[pf])
            with torch.no_grad():
                o = float(pc.params[pid])
                pc.params[pid] = o * math.exp(eps); lp = ll()
                pc.params[pid] = o * math.exp(-eps); lm = ll()
                pc.params[pid] = o
            g_fd = (lp - lm) / (2 * eps)
            dmax = max(dmax, abs(g_fd - g_an))
            gmax = max(gmax, abs(g_an), abs(g_fd))
            n += 1
    assert n >= 4
    # Judged against the LARGEST gradient probed, not per-edge. A per-edge relative error is
    # meaningless where the gradient is near zero -- the central difference resolves about 1e-4 here
    # (the LL is accumulated in fp32), so an edge with |g| ~ 1e-4 reads as 100% error no matter how
    # correct the kernel is. MEASURED: shapes whose worst per-edge ratio was 0.46 and 1.00 had a
    # `dmax` of 1e-4, i.e. they were exact; the genuinely broken shape had `dmax` 3.06 against a
    # `gmax` of 3.2. Scaling by `gmax` separates those cleanly.
    assert dmax / gmax < 5e-2, f"dmax={dmax:.3e} gmax={gmax:.3e} ratio={dmax / gmax:.3e}"


# ------------------------------------------------- the dual M-step, at its corners

@cuda_only
@pytest.mark.parametrize("step_size,pseudocount", [(1.0, 0.0), (0.5, 0.0), (0.5, 0.1), (1.0, 2.0)])
def test_em_correction_keeps_parameters_normalized(step_size, pseudocount):
    """The corrected M-step must leave each node's parameters summing to 1 over its children, for
    every `(step_size, pseudocount)` -- the dual update renormalizes through `cum`, and an error
    there shows up as drift rather than as anything obviously wrong."""
    dev = torch.device("cuda:0")
    root, ns = _build_shape(4, 2, 2)
    pc = juice.compile(root, verbose = False).to(dev)
    B = 96
    torch.manual_seed(3)
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)

    for _ in range(3):
        pc(x, sum_external_params = {ns: phi})
        pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True,
                    flows_memory = 0.0, allow_modify_flows = False)
        pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)
        assert torch.isfinite(pc.params).all(), "non-finite parameters after the corrected M-step"

    ps, pe = ns._param_range
    E, cbs, bs = ns.edge_ids.size(1), ns.ch_block_size, ns.block_size
    theta = pc.params[ps:pe].reshape(E, cbs, bs)
    nb = ns.edge_ids[0].to(device = theta.device, dtype = torch.long)
    tot = torch.zeros(ns.num_node_blocks, bs, device = theta.device, dtype = theta.dtype)
    tot.index_add_(0, nb, theta.sum(dim = 1))
    assert (tot - 1.0).abs().max().item() < 1e-4, (tot - 1.0).abs().max().item()


@cuda_only
def test_em_correction_finite_when_a_parameter_is_zero():
    """A structurally zero `theta` makes the dual ratio `(F+ + pc/K) / (F- + pc*theta)` a finite
    number over a clamped zero, and `theta * ratio` must stay 0 rather than becoming `0 * inf`.
    With `keep_zero_params` the zeros must also survive the update."""
    dev = torch.device("cuda:0")
    root, ns = _build_shape(4, 2, 2)
    pc = juice.compile(root, verbose = False).to(dev)
    B = 64
    torch.manual_seed(3)
    x = torch.randint(0, NUM_CATS, [B, 2], device = dev)
    phi = _phi(ns, B, dev)

    ps, pe = ns._param_range
    with torch.no_grad():                          # zero out a slice of this ns's parameters
        pc.params[ps:ps + (pe - ps) // 4] = 0.0
    zeroed = (pc.params[ps:pe] == 0.0).clone()
    assert bool(zeroed.any())

    pc(x, sum_external_params = {ns: phi})
    pc.backward(x, sum_external_params = {ns: phi}, logspace_flows = True,
                flows_memory = 0.0, allow_modify_flows = False)
    pc.mini_batch_em(step_size = 1.0, pseudocount = 1.0, keep_zero_params = True)

    assert torch.isfinite(pc.params).all(), "the corrected M-step produced inf/NaN on a zero parameter"
    assert bool((pc.params[ps:pe][zeroed] == 0.0).all()), "`keep_zero_params` did not hold the zeros"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
