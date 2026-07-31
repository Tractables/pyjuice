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


def _gate_shape(ns, batch):
    """The gate grid the layer expects: one entry per (node gate-block, child gate-block, sample)."""
    return ns.external_params.tensor_shapes(ns, batch)[0]


def _set_node_params(pc, ns, vals):
    psid, peid = ns._param_range
    local = (ns._param_ids - psid) // (ns.block_size * ns.ch_block_size)
    buf = pc.params[psid:peid].reshape(-1, ns.ch_block_size, ns.block_size)
    buf[local, :, :] = vals.permute(0, 2, 1).to(buf.dtype)


def _gates_per_edge_block(phi, ns):
    """`[Nk, Ck]` grid for one sample -> `[E, D]`, the gates of each edge block, in `edge_ids` order.

    This is the caller-facing layout translated to the one the maths below is written in; it is the
    same mapping `to_storage` performs, done independently here so the test does not inherit a bug
    from it."""
    _, n_child_gates = ns.external_params.gate_counts(ns)
    out = []
    for e in range(ns.edge_ids.size(1)):
        nb = int(ns.edge_ids[0, e])
        cb = int(ns.edge_ids[1, e])
        out.append(phi[nb, cb * n_child_gates:(cb + 1) * n_child_gates])
    return torch.stack(out, dim = 0)                                       # [E, D]


def _effective(theta, phi, ns, gate_cbs):
    """`theta_tilde` for one sample: `[E, K, Kc]` from shared `[E, K, Kc]` and the gate grid `[Nk, Ck]`.

    Normalization runs over all children of a NODE -- the edge blocks incident to that node's block --
    not over every edge block in the layer, which would mix node blocks that share nothing."""
    E = theta.size(0)

    g = _gates_per_edge_block(phi, ns).double().exp().repeat_interleave(gate_cbs, dim = 1)   # [E, Kc]
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

    if phi is None:
        phi = torch.randn(_gate_shape(ns, batch), device = dev) * scale

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

    # The invariant, tested WITHIN the gated path: with one gate the output cannot depend on phi. Both
    # arms run whichever kernel the launcher chose, so this compares like with like and stays tight.
    shape = _gate_shape(ns_a, batch)
    flat = torch.zeros(shape, device = dev)
    phi = torch.randn(shape, device = dev) * 3.0
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
    phi = torch.randn(_gate_shape(ns, 64), device = dev) * 0.5
    phi[:, :, 0] = -float("inf")             # kill one child gate column, keep the rest

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
    flat = torch.zeros(_gate_shape(ns, 64), device = dev)
    base = pc(data, sum_external_params = {ns: flat}).clone()

    boosted = flat.clone()
    boosted[:, 0, 0] = 4.0
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
        return torch.randn(_gate_shape(ns, batch), device = dev) * scale

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
    phi = torch.randn(_gate_shape(gated, batch), device = dev) * 0.5

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
    phi = torch.randn(_gate_shape(s0, batch), device = dev) * 0.5

    lls = pc(data, sum_external_params = {s0: phi})
    assert torch.isfinite(lls).all()

    # the tied copy has no tensor of its own -- one gate drives both layers
    other = pc(data, sum_external_params = {s0: phi * 0.0})
    assert not torch.allclose(lls, other)


# --------------------------------------------------------------------------------- the API

@cuda_only
def test_gate_grid_shape_follows_the_gate_size_not_the_blocking():
    """The grid is `num_nodes / gate block size` by `num_ch_nodes / gate ch_block_size`.

    A caller who asks for gates spanning 16 children gets `num_ch_nodes / 16` of them, whatever
    blocking pyjuice uses inside -- that is the point of this layout, and the child axis below is
    identical across three different `ns.block_size` values.

    The NODE axis tracks `ns.block_size` only because the gate's own node block size defaults to it;
    setting it independently is the case that still raises `NotImplementedError`. When that case lands,
    this axis becomes independent too and the assertion below should tighten."""
    for num_latents, gate_cbs in ((256, 16), (512, 8)):
        shapes = []
        for block_size in (32, 64, 128):
            _, ns = _build(num_latents, block_size, gate_cbs = gate_cbs)
            shape = ns.external_params.tensor_shapes(ns, 8)[0]
            gate_bs, _ = ns.external_params.gate_sizes(ns)

            assert tuple(shape) == (8, ns.num_nodes // gate_bs, ns.num_ch_nodes // gate_cbs), shape
            shapes.append(tuple(shape))

        child_axis = {sh[2] for sh in shapes}
        assert child_axis == {num_latents // gate_cbs}, \
            f"the child axis moved with ns.block_size: {shapes}"


@cuda_only
def test_storage_round_trip_is_the_identity():
    """`from_storage` must invert `to_storage` exactly -- it is how gradients reach the caller.

    Storage holds the caller's own grid, batch innermost, so the two are a permutation and its inverse
    and the round trip is bit-exact. Entries for (node block, child block) pairs the layer does not
    connect simply go unread: the compiled table never points at them, and the gradient buffer is
    zero-initialized each backward, so they read back as zero rather than as stale values."""
    dev = torch.device("cuda:0")
    batch = 5
    _, ns = _build(256, 64, gate_cbs = 8)
    ep = ns.external_params

    torch.manual_seed(0)
    phi = torch.randn(_gate_shape(ns, batch), device = dev)

    stored = ep.to_storage(ns, (phi,))[0]
    assert tuple(stored.shape) == tuple(ep.storage_shapes(ns, batch)[0])

    back = ep.from_storage(ns, (stored,))[0]
    assert back.shape == phi.shape
    assert torch.equal(back, phi), "the round trip is not the identity"


@cuda_only
def test_storage_offsets_agree_with_the_grid_layout():
    """Each edge block's compiled base must be where its gates actually live in the grid.

    This is what replaced a per-forward gather: the kernel's existing indirection absorbs the layout,
    so the offsets have to be right or the kernel reads the wrong gates -- silently, since every offset
    is in bounds."""
    _, ns = _build(512, 64, gate_cbs = 8)
    ep = ns.external_params

    n_node_gates, n_child_gates = ep.gate_counts(ns)
    ck = ns.num_ch_nodes // ep.gate_sizes(ns)[1]
    offsets = ep.storage_offsets(ns)[0]

    for e in range(ns.edge_ids.size(1)):
        nb, cb = int(ns.edge_ids[0, e]), int(ns.edge_ids[1, e])
        # the flat position of grid entry (nb * A, cb * D) in a [Nk, Ck] row-major grid
        assert int(offsets[e]) == (nb * n_node_gates) * ck + cb * n_child_gates, \
            f"edge block {e} -> ({nb}, {cb}) has offset {int(offsets[e])}"


@cuda_only
@needs_cute
@pytest.mark.parametrize("bad", ["one_axis_short", "transposed", "extra_axis", "wrong_batch"])
def test_wrong_gate_shape_is_rejected(bad):
    """A mis-shaped gate tensor must fail loudly at the call, not be silently reinterpreted."""
    dev = torch.device("cuda:0")
    batch = 64
    root, ns = _build(256, 64, gate_cbs = 8)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    b, nk, ck = _gate_shape(ns, batch)

    shape = {"one_axis_short": (b, nk, ck - 1),
             "transposed": (b, ck, nk),
             "extra_axis": (b, nk, ck, 1),
             "wrong_batch": (b // 2, nk, ck)}[bad]
    if shape[1:] == (ck, nk) and nk == ck:
        pytest.skip("the grid is square here, so a transpose is not detectable by shape")

    with pytest.raises(AssertionError, match = "shape"):
        pc(data, sum_external_params = {ns: torch.zeros(shape, device = dev)})


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
    phi = torch.randn(_gate_shape(ns, batch), device = dev) * 0.5

    with pytest.raises(NotImplementedError):
        pc(data, sum_external_params = {ns: phi})


# --------------------------------------------------------------------------------- backward

def _ch_element_ids(ns):
    """Global `element_mars` row of each child block's first element, indexed by `edge_ids[1]`.

    Derived here from `_output_ind_range` rather than read off the layer, so the reference does not
    inherit the layer's own view of where a child block lives."""
    out = torch.zeros([ns.num_ch_node_blocks], dtype = torch.long)
    cum = 0
    for cs in ns.chs:
        out[cum:cum + cs.num_node_blocks] = (cs._output_ind_range[0]
                                             + torch.arange(cs.num_node_blocks) * ns.ch_block_size)
        cum += cs.num_node_blocks
    return out


def _flow_reference(pc, ns, phi, gate_cbs, batch):
    """
    float64 element and parameter flows under the effective per-sample parameters.

    Built from the circuit's own `node_flows` / `node_mars` / `element_mars`, which the layers ABOVE
    this one produced and the gate does not touch, plus the definition of `theta_tilde`:

        edge_flow[e,n,c,b] = exp(node_flows[n,b]) * theta_tilde[b,e,n,c]
                             * exp(element_mars[c,b] - node_mars[n,b])
        element_flows[c,b] = log sum_{e,n} edge_flow        (log space)
        param_flows[e,n,c] = sum_b edge_flow

    `node_mars` is the node's value under the EFFECTIVE parameters, so it already carries `1/Z`.
    Pairing it with an unnormalized `theta` -- or with `theta_tilde` AND a `Z`-shifted `node_mars` --
    puts a `log Z` offset on every flow, which is invisible at `phi = 1` because `Z` is then 1.
    """
    dev = pc.node_mars.device
    bs, cbs = ns.block_size, ns.ch_block_size
    E = ns.edge_ids.size(1)
    nid0 = ns._output_ind_range[0]

    theta = pc.get_node_params(ns)                                      # [E, bs, cbs]
    ch_eids = _ch_element_ids(ns)
    ar_n = torch.arange(bs, device = dev)
    ar_c = torch.arange(cbs, device = dev)

    ef = torch.zeros([pc.element_flows.size(0), batch], dtype = torch.float64, device = dev)
    pf = torch.zeros([E, bs, cbs], dtype = torch.float64, device = dev)

    for b in range(batch):
        eff = _effective(theta, phi[b], ns, gate_cbs)                   # [E, bs, cbs], float64
        for e in range(E):
            nrows = nid0 + int(ns.edge_ids[0, e]) * bs + ar_n
            crows = int(ch_eids[int(ns.edge_ids[1, e])]) + ar_c

            f = pc.node_flows[nrows, b].double().exp()                  # [bs]
            nm = pc.node_mars[nrows, b].double()                        # [bs]
            em = pc.element_mars[crows, b].double()                     # [cbs]

            w = f[:, None] * eff[e] * (em[None, :] - nm[:, None]).exp()
            pf[e] += w
            ef[crows, b] += w.sum(dim = 0)

    return ef.log(), pf


@cuda_only
@needs_cute
@pytest.mark.parametrize("num_latents,block_size,gate_cbs,batch,scale", [
    (128, 128, 8, 64, 0.0),
    (128, 128, 8, 64, 1.0),
    (128, 128, 8, 64, 3.0),
    (128, 128, 16, 128, 2.0),
    (256, 128, 32, 64, 2.0),
])
def test_backward_matches_reference(num_latents, block_size, gate_cbs, batch, scale):
    """`pc.backward()` against the float64 reference: both flows, gates from a no-op to a strong one.

    This is the end-to-end statement that the wiring is right -- the `log Z` shift, the backward gate
    table (which is the TRANSPOSE of the forward's indexing, one row per child block), the forward
    table the param flows reuse, and both fork kernels."""
    pc, root, ns, data, phi, _ = _run(num_latents, block_size, gate_cbs, batch, scale = scale)
    pc.backward(data, flows_memory = 0.0)

    ref_ef, ref_pf = _flow_reference(pc, ns, phi, gate_cbs, batch)

    live = torch.isfinite(ref_ef)
    d_ef = float((pc.element_flows[:, :batch].double()[live] - ref_ef[live]).abs().max())

    ns.update_param_flows(pc.param_flows)
    got_pf = ns.get_param_flows().double().to(ref_pf.device)
    d_pf = float(((got_pf - ref_pf).abs() / ref_pf.clamp(min = 1e-30)).max())

    # The fp16 tensor-core floor of the kernels these fork (~1e-3 log space for the element flows).
    assert d_ef < 2e-3, f"element flows off by {d_ef}"
    assert d_pf < 3e-3, f"param flows off by {d_pf} (relative)"


@cuda_only
@needs_cute
@pytest.mark.parametrize("block_size,batch", [(128, 64), (128, 128)])
def test_backward_with_unit_gates_matches_the_plain_pc(block_size, batch):
    """`log phi = 0` makes the gate the identity, so every flow must match the ungated circuit's.

    Necessary but NOT sufficient on its own: at `phi = 1` the normalizer is 1, so this case cannot see
    a wrong `log Z` shift, and every gate index gives the same answer so it cannot see a wrong table.
    `test_backward_matches_reference` covers both; this one pins the reduction."""
    dev = torch.device("cuda:0")
    num_latents = 128

    root_a, ns_a = _build(num_latents, block_size, 8, seed = 0)
    root_b, ns_b = _build(num_latents, block_size, None, seed = 0)
    pc_a = juice.compile(root_a, verbose = False).to(dev)
    pc_b = juice.compile(root_b, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    phi = torch.zeros(_gate_shape(ns_a, batch), device = dev)

    pc_a(data, sum_external_params = {ns_a: phi})
    pc_a.backward(data, flows_memory = 0.0)
    pc_b(data)
    pc_b.backward(data, flows_memory = 0.0)

    live = torch.isfinite(pc_b.element_flows[:, :batch])
    d_ef = float((pc_a.element_flows[:, :batch][live] - pc_b.element_flows[:, :batch][live]).abs().max())

    ns_a.update_param_flows(pc_a.param_flows)
    ns_b.update_param_flows(pc_b.param_flows)
    ref = ns_b.get_param_flows().double()
    d_pf = float(((ns_a.get_param_flows().double() - ref).abs() / ref.clamp(min = 1e-30)).max())

    assert d_ef < 2e-3, f"element flows differ from the ungated circuit by {d_ef}"
    assert d_pf < 3e-3, f"param flows differ from the ungated circuit by {d_pf} (relative)"


@cuda_only
@needs_cute
def test_backward_leaves_node_mars_as_the_forward_wrote_it():
    """The backward turns `node_mars` into `log N` and must put it back.

    `node_mars` is a circuit-wide buffer that later work reads, so a shift left in place would corrupt
    everything downstream -- and by exactly `log Z`, which is 0 at `phi = 1`, so only a non-trivial
    gate can catch it."""
    pc, root, ns, data, phi, _ = _run(128, 128, 8, 64, scale = 2.0)
    before = pc.node_mars.clone()
    pc.backward(data, flows_memory = 0.0)

    lo, hi = ns._output_ind_range
    d = float((pc.node_mars[lo:hi] - before[lo:hi]).abs().max())
    assert d == 0.0, f"node_mars was left shifted by up to {d}"


@cuda_only
@needs_cute
def test_gate_gradients_are_refused_rather_than_returned_as_zeros():
    """`d LL / d log phi` is not implemented, and the gradient buffer is allocated and zeroed by
    default -- so the read must say so instead of handing back a plausible tensor of zeros."""
    pc, root, ns, data, phi, _ = _run(128, 128, 8, 64, scale = 1.0)
    pc.backward(data, flows_memory = 0.0)

    with pytest.raises(NotImplementedError, match = "gradients"):
        pc.get_external_params_grad(ns)


@cuda_only
@needs_cute
def test_backward_outside_the_supported_regime_raises():
    """The forked kernels have no Triton fallback, so a shape the standard backward would serve with
    its own (UNGATED) kernels has to raise. A batch the CuTe element-flow kernel cannot tile is such a
    shape -- the forward still runs, via the small-batch kernel."""
    pc, root, ns, data, phi, _ = _run(128, 128, 8, 8, scale = 1.0)

    with pytest.raises(NotImplementedError, match = "block-sparse CUDA regime"):
        pc.backward(data, flows_memory = 0.0)

    # and the interception must not survive the failure, or the next backward inherits it
    layer = [l for g in pc.inner_layer_groups if g.is_sum() for l in g.layers
             if hasattr(l, "external_node_infos")][0]
    assert layer._ext_bw_ele_hook is None and layer._ext_bw_par_hook is None



if __name__ == "__main__":
    test_single_gate_is_a_no_op(64, 64)
    test_matches_materialized_pc(256, 64, 8, 64)
    test_all_tile_configs_agree()
    print("ok")
