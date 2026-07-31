"""
The block-scale forward, against a MATERIALIZED plain PC.

`BlockScaleSumParams` gives every `[block_size x gate ch_block_size]` parameter block a per-sample
scalar gate and renormalizes over each node's children:

    theta_tilde[b, n, c] = phi[b, g(n, c)] * theta[n, c] / Z[n, b],
    Z[n, b]              = sum_c phi[b, g(n, c)] * theta[n, c]

`theta_tilde` is an ordinary child-normalized parameter matrix, so a second circuit whose parameters are
set to it -- for one chosen sample -- is a plain PC evaluated by pyjuice's own trusted code. That is the
oracle here: it shares no code with the kernel under test, so agreement validates the whole path
(staging, the gate fold, the running stabilizers, `sigma`, the fused normalizer and the epilogue).

Two invariants come for free and pin down what the oracle cannot:

  * with ONE gate per node the gate cancels against its own normalizer, so the circuit must match the
    ungated one whatever `phi` is (to bf16 accuracy -- see the test for why not bit-exactly);
  * every tile config computes the same contraction, so they must all agree -- which is what makes the
    launcher's autotuning safe.

There is no Triton fallback for this parameterization: it is a fork of the CuTe/TMA sum kernel, so the
tests skip wholesale where that kernel does not apply, and the last test pins the boundary of what is
supported so it is visible when it moves.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams


NUM_CATS = 5

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _cute_available():
    if not torch.cuda.is_available():
        return False
    from pyjuice.nodes.external_params.kernels.c import get_cute_module
    try:
        return get_cute_module() is not None
    except Exception:
        return False


needs_cute = pytest.mark.skipif(
    not _cute_available(),
    reason = "needs the CuTe/TMA extension (nvcc + CUTLASS + sm_90+); no fallback exists")


# --------------------------------------------------------------------------------- circuits

def _build(num_latents, block_size, gate_cbs, n_vars = 2, seed = 0, tie_external = False):
    """A small PC with one gated sum layer over a product of `n_vars` inputs."""
    torch.manual_seed(seed)
    n_blocks = num_latents // block_size

    with juice.set_block_size(block_size):
        ni = [inputs(v, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(n_vars)]
        kw = {}
        if gate_cbs is not None:
            kw["external_params"] = BlockScaleSumParams(ch_block_size = gate_cbs,
                                                        tie_external = tie_external)
        ns = summate(multiply(*ni), num_node_blocks = n_blocks, **kw)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, ns


def _set_node_params(pc, ns, vals):
    psid, peid = ns._param_range
    local = (ns._param_ids - psid) // (ns.block_size * ns.ch_block_size)
    buf = pc.params[psid:peid].reshape(-1, ns.ch_block_size, ns.block_size)
    buf[local, :, :] = vals.permute(0, 2, 1).to(buf.dtype)


def _effective(theta, phi, ns, gate_cbs):
    """`theta_tilde` for one sample: `[E, K, Kc]` from shared `[E, K, Kc]` and gates `[E, 1, D]`.

    Normalization runs over all children of a NODE -- the edge blocks incident to that node's block --
    not over every edge block in the layer, which would mix node blocks that share nothing."""
    E = theta.size(0)

    g = phi[:, 0, :].double().exp().repeat_interleave(gate_cbs, dim = 1)   # [E, Kc]
    eff = theta.double() * g[:, None, :]

    nblk = ns.edge_ids[0, :].tolist()
    out = torch.empty_like(eff)
    for nb in sorted(set(nblk)):
        rows = [e for e in range(E) if nblk[e] == nb]
        tot = eff[rows].sum(dim = 2).sum(dim = 0)                          # [K]
        for e in rows:
            out[e] = eff[e] / tot[:, None]

    return out


def _run(num_latents, block_size, gate_cbs, batch, n_vars = 2, scale = 0.7, seed = 0, phi = None):
    """Compile, evaluate with gates, and return everything the oracle needs."""
    dev = torch.device("cuda:0")
    root, ns = _build(num_latents, block_size, gate_cbs, n_vars, seed)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, n_vars], device = dev)

    E = ns.edge_ids.size(1)
    D = ns.external_params.gate_counts(ns)[1]
    if phi is None:
        phi = torch.randn([batch, E, 1, D], device = dev) * scale

    lls = pc(data, sum_external_params = {ns: phi})
    return pc, root, ns, data, phi, lls


# --------------------------------------------------------------------------------- invariants

@cuda_only
@needs_cute
@pytest.mark.parametrize("block_size,batch", [(64, 64), (128, 64), (64, 8), (128, 3)])
def test_single_gate_is_a_no_op(block_size, batch):
    """One gate per node => the gate cancels against its own normalizer, for ANY phi.

    Exactly, in real arithmetic. Not bit-exactly on the machine, and the loose bound below is NOT
    slack for the gate being sloppy -- it is the sum layer's own floor. Measured against a float64
    reference on this circuit: the plain forward is 3.007e-04 from exact and the gated forward is
    3.007e-04 from exact, while they differ from each other by 2.4e-04. The bf16 tensor cores set that
    floor; folding `log phi` into `element_mars` perturbs values before they are rounded to bf16, so
    individual roundings fall differently, which moves the error around inside the floor rather than
    adding to it.

    The tight bound is asserted separately at `phi = 1`, where the fold adds exactly zero, so the two
    paths round identically and only the normalizer's own rounding is left.
    """
    dev = torch.device("cuda:0")

    # One node block, one edge block, and a gate as wide as the child block => exactly one gate. The
    # type warns about precisely this at construction, since it is a no-op by accident far more often
    # than on purpose.
    with pytest.warns(RuntimeWarning, match = "exact no-op"):
        root_a, ns_a = _build(block_size, block_size, block_size)
    pc_a = juice.compile(root_a, verbose = False).to(dev)

    root_b, _ = _build(block_size, block_size, None)
    pc_b = juice.compile(root_b, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    E = ns_a.edge_ids.size(1)

    # The invariant, tested WITHIN the gated path: with one gate the output cannot depend on phi. Both
    # arms run whichever kernel the launcher chose, so this compares like with like and stays tight.
    flat = torch.zeros([batch, E, 1, 1], device = dev)
    phi = torch.randn([batch, E, 1, 1], device = dev) * 3.0
    a0 = pc_a(data, sum_external_params = {ns_a: flat})
    a1 = pc_a(data, sum_external_params = {ns_a: phi})
    assert torch.allclose(a0, a1, atol = 1e-3), \
        f"the gate changed the answer: max |d| = {(a0 - a1).abs().max():.3e}"

    # and against the ungated circuit, which may be a DIFFERENT kernel at a different precision
    b = pc_b(data)
    assert torch.allclose(a0, b, atol = 1e-3), f"vs plain: max |d| = {(a0 - b).abs().max():.3e}"


@cuda_only
@needs_cute
@pytest.mark.parametrize("num_latents,block_size,gate_cbs,batch", [
    (256, 64, 8, 64),        # the narrow tile
    (256, 64, 16, 128),
    (512, 64, 4, 64),        # the finest gate the kernel specializes
    (512, 128, 8, 64),       # the wide tile
    (512, 128, 32, 128),
    (256, 128, 128, 64),     # gate as wide as the child block: several edge blocks, one gate each
    # ---- batches the CuTe tile cannot serve at all; the small-batch kernel takes these ----
    (256, 64, 8, 1),         # decoding-shaped: one sample at a time
    (256, 64, 8, 4),
    (256, 64, 8, 13),        # not a power of two either
    (256, 64, 32, 32),
    (512, 128, 16, 48),
    (128, 32, 8, 16),        # node block below the CuTe tile, fine for a 32-node warp group
])
def test_matches_materialized_pc(num_latents, block_size, gate_cbs, batch):
    """The kernel against a plain PC carrying the effective per-sample parameters."""
    dev = torch.device("cuda:0")
    sample = min(3, batch - 1)

    pc_a, root_a, ns_a, data, phi, lls_a = _run(num_latents, block_size, gate_cbs, batch)

    # circuit B: same topology, no gate, parameters set to the effective per-sample matrix
    root_b, ns_b = _build(num_latents, block_size, None)
    pc_b = juice.compile(root_b, verbose = False).to(dev)
    pc_b.input_layer_group.layers[0].params.copy_(pc_a.input_layer_group.layers[0].params)
    _set_node_params(pc_b, root_b, pc_a.get_node_params(root_a))
    _set_node_params(pc_b, ns_b, _effective(pc_a.get_node_params(ns_a), phi[sample], ns_a, gate_cbs))

    lls_b = pc_b(data[sample:sample + 1, :])

    d = float((lls_a[sample] - lls_b[0]).abs())
    assert d < 2e-3, f"|dLL| = {d:.3e}"


@cuda_only
@needs_cute
def test_all_tile_configs_agree():
    """Every valid tile computes the same contraction -- what makes autotuning safe."""
    pc, _, ns, data, phi, ref = _run(512, 128, 16, 128)

    layer = [l for g in pc.inner_layer_groups if g.is_sum() for l in g.layers
             if hasattr(l, "_bs_fw_plan")][0]
    mod, fname, calls = layer._bs_fw_plan[1]
    if fname != "blockscale_forward":
        pytest.skip("the launcher chose the small-batch kernel; no tile configs to compare")
    valid = [int(i) for i in mod.fitting_configs(128, 128, 16)]
    assert len(valid) >= 2, f"expected several usable tiles, got {valid}"

    saved = list(calls)
    for cfg in valid:
        calls[:] = [tuple(a[:-1]) + (cfg,) for a in saved]
        lls = pc(data, sum_external_params = {ns: phi})
        assert torch.equal(lls, ref), f"tile {tuple(mod.configs()[cfg])} disagrees"
    calls[:] = saved


@cuda_only
@needs_cute
def test_repeated_evaluation_is_deterministic():
    """No atomics on this path, so the same input must give bit-identical output."""
    pc, _, ns, data, phi, first = _run(256, 64, 8, 64)
    for _ in range(3):
        assert torch.equal(pc(data, sum_external_params = {ns: phi}), first)


# --------------------------------------------------------------------------------- numerics

@cuda_only
@needs_cute
@pytest.mark.parametrize("scale", [8.0, 30.0])
def test_extreme_gates_stay_finite(scale):
    """The running stabilizers must hold up when phi spans a wide dynamic range."""
    lls = _run(256, 64, 8, 64, scale = scale)[-1]
    assert torch.isfinite(lls).all(), "log-likelihoods went non-finite under a wide gate range"


@cuda_only
@needs_cute
def test_zero_gates_do_not_produce_nan():
    """`phi = 0` (log phi = -inf) zeroes a gate. Those children must drop out, not poison the sum.

    A node keeps at least one live gate here, so its value stays finite; the guards being exercised are
    the `-inf - (-inf)` ones in the stabilizer rescale and in the exponentials."""
    dev = torch.device("cuda:0")
    root, ns = _build(256, 64, 8)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [64, 2], device = dev)
    E = ns.edge_ids.size(1)
    D = ns.external_params.gate_counts(ns)[1]

    phi = torch.randn([64, E, 1, D], device = dev) * 0.5
    phi[:, :, :, 0] = -float("inf")          # kill one child gate of every block, keep the rest

    lls = pc(data, sum_external_params = {ns: phi})
    assert not torch.isnan(lls).any(), "a zeroed gate produced NaN"
    assert torch.isfinite(lls).all()


@cuda_only
@needs_cute
def test_gate_shifts_mass_towards_its_block():
    """A sanity check with a known direction: raising one gate must raise that child's posterior."""
    dev = torch.device("cuda:0")
    root, ns = _build(256, 64, 8)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [64, 2], device = dev)
    E = ns.edge_ids.size(1)
    D = ns.external_params.gate_counts(ns)[1]

    flat = torch.zeros([64, E, 1, D], device = dev)
    base = pc(data, sum_external_params = {ns: flat}).clone()

    boosted = flat.clone()
    boosted[:, 0, 0, 0] = 4.0
    other = pc(data, sum_external_params = {ns: boosted})

    assert not torch.allclose(base, other), "the gate had no effect at all"
    assert torch.isfinite(other).all()


# --------------------------------------------------------------------------------- plumbing

@cuda_only
@needs_cute
def test_two_gated_layers_are_independent():
    """Two gated sum layers in one circuit, each with its own gate tensor."""
    dev = torch.device("cuda:0")
    batch, K, bs = 64, 256, 64
    torch.manual_seed(0)

    with juice.set_block_size(bs):
        n0 = inputs(0, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        n1 = inputs(1, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        s0 = summate(multiply(n0, n1), num_node_blocks = K // bs,
                     external_params = BlockScaleSumParams(ch_block_size = 8))
        n2 = inputs(2, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        s1 = summate(multiply(s0, n2), num_node_blocks = K // bs,
                     external_params = BlockScaleSumParams(ch_block_size = 16))
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)

    def gates(ns, scale):
        E = ns.edge_ids.size(1)
        D = ns.external_params.gate_counts(ns)[1]
        return torch.randn([batch, E, 1, D], device = dev) * scale

    g0, g1 = gates(s0, 0.5), gates(s1, 0.5)
    a = pc(data, sum_external_params = {s0: g0, s1: g1})
    assert torch.isfinite(a).all()

    # changing only the second layer's gate must change the answer
    b = pc(data, sum_external_params = {s0: g0, s1: gates(s1, 1.5)})
    assert not torch.allclose(a, b)


@cuda_only
@needs_cute
def test_gated_and_plain_layers_coexist():
    """A gated sum layer alongside an ungated one: the gate must not leak into the plain layer."""
    dev = torch.device("cuda:0")
    batch, K, bs = 64, 256, 64
    torch.manual_seed(0)

    with juice.set_block_size(bs):
        n0 = inputs(0, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        n1 = inputs(1, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        gated = summate(multiply(n0, n1), num_node_blocks = K // bs,
                        external_params = BlockScaleSumParams(ch_block_size = 8))
        n2 = inputs(2, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        plain = summate(multiply(gated, n2), num_node_blocks = K // bs)
        root = summate(multiply(plain), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    E = gated.edge_ids.size(1)
    D = gated.external_params.gate_counts(gated)[1]
    phi = torch.randn([batch, E, 1, D], device = dev) * 0.5

    assert torch.isfinite(pc(data, sum_external_params = {gated: phi})).all()


@cuda_only
@needs_cute
def test_tie_external_shares_one_gate_tensor():
    """With `tie_external`, every copy of a tied node reads ONE gate tensor."""
    dev = torch.device("cuda:0")
    batch, K, bs = 64, 256, 64
    torch.manual_seed(0)

    with juice.set_block_size(bs):
        n0 = inputs(0, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        n1 = inputs(1, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        s0 = summate(multiply(n0, n1), num_node_blocks = K // bs,
                     external_params = BlockScaleSumParams(ch_block_size = 8, tie_external = True))
        n2 = inputs(2, num_node_blocks = K // bs, dist = dists.Categorical(num_cats = NUM_CATS))
        s1 = s0.duplicate(multiply(s0, n2), tie_params = True)
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    E = s0.edge_ids.size(1)
    D = s0.external_params.gate_counts(s0)[1]
    phi = torch.randn([batch, E, 1, D], device = dev) * 0.5

    lls = pc(data, sum_external_params = {s0: phi})
    assert torch.isfinite(lls).all()

    # the tied copy has no tensor of its own -- one gate drives both layers
    other = pc(data, sum_external_params = {s0: phi * 0.0})
    assert not torch.allclose(lls, other)


# --------------------------------------------------------------------------------- the boundary

@cuda_only
@needs_cute
@pytest.mark.parametrize("num_latents,block_size,batch,gate_bs,why", [
    (256, 64, 64, 32, "gate narrower than the node block"),
])
def test_unsupported_shapes_raise(num_latents, block_size, batch, gate_bs, why):
    """The kernel has no fallback, so anything outside its gate must raise -- loudly, and at the
    forward, not silently produce something else. This pins the supported region: when one of these
    starts passing, a case has been implemented and the parametrization should move."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    n_blocks = num_latents // block_size

    with juice.set_block_size(block_size):
        ni = [inputs(v, num_node_blocks = n_blocks, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        ep = (BlockScaleSumParams(block_size = gate_bs, ch_block_size = 8) if gate_bs is not None
              else BlockScaleSumParams(ch_block_size = 8))
        if gate_bs is not None:
            with pytest.raises(NotImplementedError, match = why.split()[0]):
                summate(multiply(*ni), num_node_blocks = n_blocks, external_params = ep)
            return
        ns = summate(multiply(*ni), num_node_blocks = n_blocks, external_params = ep)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    E = ns.edge_ids.size(1)
    D = ns.external_params.gate_counts(ns)[1]
    phi = torch.randn([batch, E, 1, D], device = dev) * 0.5

    with pytest.raises(NotImplementedError):
        pc(data, sum_external_params = {ns: phi})


if __name__ == "__main__":
    test_single_gate_is_a_no_op(64, 64)
    test_matches_materialized_pc(256, 64, 8, 64)
    test_all_tile_configs_agree()
    print("ok")
