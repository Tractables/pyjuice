"""
`d LL / d log phi` -- the gradient with respect to the per-sample gate.

    d LL / d log phi[b,g] = sum_n f_b[n] * [ sum_{c in g} P(c|n,b)  -  phi_b[g]*sigma[g,n]/Z_b[n] ]
                                             \\___ Ntilde term ___/     \\___ log Z term ___/

Checked three ways, because each catches something the others do not:

  * against a float64 REFERENCE, on shapes where it is short enough to be obviously right;
  * against the ZERO-SUM INVARIANT -- the bracket sums to 1 - 1 = 0 over a node's gates, because
    scaling all of them by a constant cancels in `theta_b`. This needs no reference, holds on every
    shape, and is what caught the two terms being added instead of subtracted;
  * across FORKS: Triton, CuTe and the small-batch kernels must agree, since the launcher picks
    between them by measurement and any of them can run.

Kept fast: tiny circuits, and the reference is vectorized over the batch.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _cute_available():
    if not torch.cuda.is_available():
        return False
    from pyjuice.nodes.external_params.kernels.c import get_cute_module
    try:
        return get_cute_module() is not None
    except Exception:
        return False


needs_cute = pytest.mark.skipif(not _cute_available(), reason = "needs the CuTe/TMA extension")


def _build(K, gate_cbs, seed = 0):
    """One gated layer, one node block, one child block -- so a gate is (0, ck) and the reference short."""
    torch.manual_seed(seed)
    with juice.set_block_size(K):
        ni = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        prod = multiply(*ni)
        ns = summate(prod, num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gate_cbs))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, ns, prod


def _reference(pc, ns, prod, phi, gate_cbs, batch):
    """Both terms in float64, from the circuit's own buffers after a forward+backward."""
    lo, hi = ns._output_ind_range
    elo = prod._output_ind_range[0]
    K = hi - lo

    nm = pc.node_mars[lo:hi, :batch].double()                 # [n, B] = log N
    em = pc.element_mars[elo:elo + K, :batch].double()        # [c, B]
    f = pc.node_flows[lo:hi, :batch].double().exp()           # [n, B]

    theta = pc.get_node_params(ns).double()[0]                # [n, c]
    ephi = phi.double()[:, 0, :].exp()                        # [B, g]
    gphi = ephi.repeat_interleave(gate_cbs, dim = 1)          # [B, c]

    tt = gphi[:, None, :] * theta[None, :, :]                 # [B, n, c]  (unnormalized)
    Z = tt.sum(dim = 2)                                       # [B, n]
    tb = tt / Z[:, :, None]                                   # theta_b

    P = tb * (em.T[:, None, :] - nm.T[:, :, None]).exp()      # theta_b * e^{em - log N}
    t1 = torch.einsum("nb,bnc->bc", f, P).view(batch, -1, gate_cbs).sum(dim = 2)

    sigma = theta.view(K, -1, gate_cbs).sum(dim = 2)          # [n, g]
    t2 = torch.einsum("nb,bng->bg", f, ephi[:, None, :] * sigma[None, :, :] / Z[:, :, None])

    return t1 - t2


def _run(K, gate_cbs, batch, scale, seed = 0):
    dev = torch.device("cuda:0")
    root, ns, prod = _build(K, gate_cbs, seed)
    pc = juice.compile(root, verbose = False).to(dev)
    torch.manual_seed(seed + 7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    torch.manual_seed(seed + 11)
    phi = torch.randn([batch, 1, K // gate_cbs], device = dev) * scale

    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)
    return pc, ns, prod, data, phi


@cuda_only
@needs_cute
@pytest.mark.parametrize("K,gate_cbs,batch,scale", [
    (64, 8, 8, 1.0), (64, 8, 8, 3.0), (64, 4, 4, 2.0), (64, 16, 15, 1.5),
    (128, 8, 64, 1.0), (128, 8, 64, 3.0), (128, 16, 128, 2.0), (128, 32, 64, 1.0),
    (128, 8, 1, 2.0), (256, 32, 64, 1.5), (256, 8, 8, 1.0),
])
def test_gradient_matches_the_reference(K, gate_cbs, batch, scale):
    """Across gate widths, batch sizes and both fp32 and tensor-core regimes."""
    pc, ns, prod, data, phi = _run(K, gate_cbs, batch, scale)
    ref = _reference(pc, ns, prod, phi, gate_cbs, batch)
    got = pc.get_external_params_grad(ns)[0].double()[:, 0, :]

    # The bar is looser than the element backward's own ~1e-3, and deliberately so: the gradient is a
    # DIFFERENCE of two terms of nearly equal magnitude (|t1| ~ |t2|), so cancellation amplifies each
    # term's relative error in the result. At small batch the whole path is fp32, there is no
    # tensor-core error to amplify, and it lands near 1e-6 -- hence the two very different bars.
    tol = 2e-2 if batch >= 16 else 1e-4
    d = float((got - ref).abs().max() / ref.abs().max())
    assert d < tol, f"gradient off by {d} (relative)"


@cuda_only
@needs_cute
@pytest.mark.parametrize("K,block_size,gate_cbs,batch", [
    (128, 128, 8, 64), (256, 128, 16, 64), (256, 256, 8, 4), (512, 128, 32, 128), (128, 64, 8, 64),
])
def test_gradient_sums_to_zero_over_a_nodes_gates(K, block_size, gate_cbs, batch):
    """Reference-free, so it holds on shapes a short reference does not cover -- several node blocks,
    several edge blocks. Scaling all of a node's gates by a constant is a no-op, so the gradient along
    the child-gate axis must cancel."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    nb = K // block_size
    with juice.set_block_size(block_size):
        ni = [inputs(v, num_node_blocks = nb, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        ns = summate(multiply(*ni), num_node_blocks = nb,
                     external_params = BlockScaleSumParams(ch_block_size = gate_cbs))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    phi = torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], device = dev) * 1.5

    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)
    g = pc.get_external_params_grad(ns)[0]

    scale = float(g.abs().max())
    assert scale > 0.0, "the gradient buffer was never written"
    # Relative to the gradient's own scale, and loose for the same cancellation reason as above: the
    # residual being checked IS the leftover of a near-exact cancellation between two large terms.
    assert float(g.sum(dim = 2).abs().max()) < 5e-2 * scale, \
        "the gradient does not cancel along the child-gate axis"


@cuda_only
@needs_cute
def test_every_fork_computes_the_same_gradient():
    """The launcher picks between forks by measurement, so any of them can run on a given shape."""
    pc, ns, prod, data, phi = _run(128, 8, 64, 1.5)
    ref = _reference(pc, ns, prod, phi, 8, 64)

    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]

    for forced in ("triton", "cute"):
        for k, v in layer._bs_bw_gate_cache.items():
            if isinstance(k, tuple) and k and k[0] == "eleplan":
                v["kind"] = forced
        pc(data, sum_external_params = {ns: phi})
        pc.backward(data, flows_memory = 0.0)
        g = pc.get_external_params_grad(ns)[0].double()[:, 0, :]
        d = float((g - ref).abs().max() / ref.abs().max())
        assert d < 2e-2, f"the '{forced}' fork's gradient is off by {d} (relative)"


@cuda_only
@needs_cute
def test_tied_gates_accumulate_across_every_layer_that_shares_them():
    """`tie_external` points every copy of a tied node at the SOURCE's gate tensor, so one gradient row
    is written by one layer per copy. The buffer is zeroed once per backward precisely so those add up,
    and the tied gradient must therefore equal the SUM of what the copies contribute separately.

    This is the invariant behind the emission being an atomic rather than a store, and it is checked
    against an UNTIED build of the same circuit fed the same gates -- an oracle that needs no reference
    implementation and no assumption about what the right answer is."""
    dev = torch.device("cuda:0")
    K, gate_cbs, batch, steps = 128, 8, 64, 4

    def build(tie):
        torch.manual_seed(0)
        with juice.set_block_size(K):
            ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
            src, copies = None, []
            for t in range(1, steps):
                emit = inputs(t, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
                prod = multiply(ns, emit)
                if src is None:
                    ns = summate(prod, num_node_blocks = 1,
                                 external_params = BlockScaleSumParams(ch_block_size = gate_cbs,
                                                                       tie_external = tie))
                    src = ns
                else:
                    ns = src.duplicate(prod, tie_params = True)
                copies.append(ns)
            root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
        torch.manual_seed(0)
        root.init_parameters(perturbation = 2.0)
        return juice.compile(root, verbose = False).to(dev), copies

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, steps], device = dev)
    torch.manual_seed(11)
    phi = torch.randn([batch, 1, K // gate_cbs], device = dev) * 0.5

    def run(tie):
        pc, copies = build(tie)
        pc(data, sum_external_params = {n: phi for n in copies})
        pc.backward(data, flows_memory = 0.0)
        return [pc.get_external_params_grad(n)[0].clone() for n in copies]

    tied, untied = run(True), run(False)

    # Every copy reads back the SAME storage when tied -- if they did not, there would be nothing to
    # accumulate and the rest of the check would pass vacuously.
    assert all(torch.equal(tied[0], g) for g in tied), "tied copies do not share one gradient tensor"
    assert not torch.equal(untied[0], untied[1]), "untied copies unexpectedly share a gradient"

    total = torch.stack(untied).sum(dim = 0)
    d = float((tied[0] - total).abs().max() / total.abs().max())
    assert d < 1e-4, f"the tied gradient is not the sum over its copies (off by {d} relative)"


@cuda_only
@needs_cute
def test_a_block_size_no_log_z_tile_fits_widely_still_runs():
    """REGRESSION. The log-Z tile is autotuned, and the candidates are ranked by a traffic model before
    the finalists are timed. That model favours WIDE tiles, so at a large block size every shortlisted
    candidate can be one that does not fit in shared memory (136-264 KB against 101 KB here), leaving
    nothing measurable -- and the fallback then handed back one of the candidates that had just been
    refused, so the real launch raised `OutOfResources`. The narrowest pair now always stays on the
    shortlist."""
    dev = torch.device("cuda:0")
    K, gate_cbs, batch = 1024, 4, 64
    torch.manual_seed(0)
    with juice.set_block_size(K):
        ni = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        ns = summate(multiply(*ni), num_node_blocks = 2,
                     external_params = BlockScaleSumParams(ch_block_size = gate_cbs))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    phi = torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], device = dev) * 0.5

    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)               # used to raise OutOfResources
    g = pc.get_external_params_grad(ns)[0]

    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0.0
    assert float(g.sum(dim = 2).abs().max()) < 5e-2 * float(g.abs().max())


@cuda_only
@needs_cute
def test_partial_supply_of_a_shared_layer_is_refused():
    """REGRESSION. `ext_base` is measured from the first SUPPLIED node, but the gate tables it is added
    to already carry the cursor over ALL of the layer's gated nodes -- so supplying gates for only a
    later node shifted every address by the earlier nodes' slabs. That gave a finite, plausible, WRONG
    log-likelihood (the forward read past the end of the buffer) and then an illegal access in the
    gradient write. Refused up front now."""
    dev = torch.device("cuda:0")
    batch, K, gcbs = 32, 64, 8
    torch.manual_seed(0)
    with juice.set_block_size(K):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(2)]
        # Same variable scope and depth, so the two land on ONE layer
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gcbs))
        s1 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gcbs))
        root = summate(multiply(s0), multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    if len([i for i in layer.external_node_infos]) < 2:
        pytest.skip("this build did not place both gated nodes on one layer")

    data = torch.randint(0, NUM_CATS, [batch, 2], device = dev)
    g1 = torch.randn(s1.external_params.tensor_shapes(s1, batch)[0], device = dev) * 0.5

    with pytest.raises(NotImplementedError, match = "Partial supply"):
        pc(data, sum_external_params = {s1: g1})


@cuda_only
@needs_cute
@pytest.mark.parametrize("scale", [40.0, 120.0])
def test_gradient_survives_gates_far_outside_the_exp_range(scale):
    """REGRESSION. `log phi` is a raw router output; nothing bounds it to the ~88 where `exp`
    overflows in fp32, and at scale 120 these gates reach 438.

    The log-Z term is a matmul whose [M, B] operand has no gate axis, so `phi` cannot ride inside the
    exponent the way a per-element form lets it. Unshifted, `exp(node_flows - log Z)` underflows to
    exactly 0 for every such gate -- `log Z >= log phi` -- and the entire log-Z half silently
    vanishes, leaving the Ntilde half alone. Note what that looks like: no inf, no NaN, just a
    plausible finite number, with the zero-sum residual at 1.0 instead of 0."""
    K, gate_cbs, batch = 128, 8, 64
    pc, ns, prod, data, phi = _run(K, gate_cbs, batch, scale)
    ref = _reference(pc, ns, prod, phi, gate_cbs, batch)
    got = pc.get_external_params_grad(ns)[0]

    assert torch.isfinite(got).all(), "the gradient is not finite"
    d = float((got.double()[:, 0, :] - ref).abs().max() / ref.abs().max())
    assert d < 2e-2, f"gradient off by {d} (relative)"
    # The invariant that catches a vanished term, which the relative check above can be too loose for
    assert float(got.sum(dim = 2).abs().max()) < 5e-2 * float(got.abs().max())


@cuda_only
@needs_cute
@pytest.mark.parametrize("K,gate_cbs,batch", [(64, 16, 8), (128, 32, 8), (64, 16, 15)])
def test_the_triton_fork_accumulates_when_a_gate_spans_several_element_tiles(K, gate_cbs, batch):
    """REGRESSION. The element backward emits the Ntilde half of the gradient once per parent block,
    into row `(tile_id * TILE_SIZE_M) // GATE_CBS`. When the element tile is NARROWER than a gate --
    which the small-batch heuristic forces, pinning `TILE_SIZE_M` to 8 -- that maps several distinct
    `pid_m` programs onto ONE row, each holding a partial sum over its own slice of the gate's
    children. Emitting with a plain store let the last writer win and silently dropped the rest.

    Forcing the fork is what gives this teeth: at these shapes the small-batch CUDA fork, which
    accumulates, normally wins the autotune, so the default path hid the bug. `(64, 16, 15)` is
    already in the reference sweep above and passed throughout."""
    pc, ns, prod, data, phi = _run(K, gate_cbs, batch, 1.5)
    ref = _reference(pc, ns, prod, phi, gate_cbs, batch)

    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]
    for k, v in layer._bs_bw_gate_cache.items():
        if isinstance(k, tuple) and k and k[0] == "eleplan":
            v["kind"] = "triton"

    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 0.0)
    got = pc.get_external_params_grad(ns)[0].double()[:, 0, :]

    d = float((got - ref).abs().max() / ref.abs().max())
    assert d < 1e-4, f"the Triton fork's gradient is off by {d} (relative)"


@cuda_only
@needs_cute
def test_autotuning_does_not_accumulate_into_the_gradient():
    """REGRESSION. The gradient writes are atomic ACCUMULATIONS into the real buffer, so measuring
    several candidates over warmup and reps once added it tens of times over -- ~128x too large, with
    the zero-sum invariant blown. The flow output already went to a scratch buffer for this reason;
    the gradient did not.

    A freshly compiled PC autotunes on its first backward; a second backward reuses the cached choice
    and does not. The two must agree."""
    pc, ns, prod, data, phi = _run(128, 8, 64, 1.5)      # first backward: autotunes
    first = pc.get_external_params_grad(ns)[0].clone()

    pc(data, sum_external_params = {ns: phi})           # second: choice cached, no trials
    pc.backward(data, flows_memory = 0.0)
    second = pc.get_external_params_grad(ns)[0]

    d = float((first - second).abs().max() / float(second.abs().max()))
    assert d < 1e-5, f"the autotuned backward's gradient differs from the cached one by {d}"


@cuda_only
@needs_cute
def test_gradient_is_replaced_not_accumulated_across_backwards():
    """Unlike the parameter flows, the gradient buffer is zeroed by each backward -- so two identical
    backwards give the same gradient, not twice it."""
    pc, ns, prod, data, phi = _run(128, 8, 64, 1.5)
    once = pc.get_external_params_grad(ns)[0].clone()

    pc(data, sum_external_params = {ns: phi})
    pc.backward(data, flows_memory = 1.0)               # param flows accumulate; the gradient must not
    twice = pc.get_external_params_grad(ns)[0]

    d = float((once - twice).abs().max() / float(once.abs().max()))
    assert d < 1e-5, f"the gradient accumulated across backwards (off by {d})"


@cuda_only
@needs_cute
def test_group_gradient_is_the_concatenation_of_its_members():
    dev, batch, K, gcbs = torch.device("cuda:0"), 64, 128, 8
    torch.manual_seed(0)
    with juice.set_block_size(K):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(3)]
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gcbs))
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gcbs))
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0 = torch.randn(s0.external_params.tensor_shapes(s0, batch)[0], device = dev) * 1.5
    g1 = torch.randn(s1.external_params.tensor_shapes(s1, batch)[0], device = dev) * 1.5

    pc(data, sum_external_params = {s0: g0, s1: g1})
    pc.backward(data, flows_memory = 0.0)
    a, b = pc.get_external_params_grad(s0)[0].clone(), pc.get_external_params_grad(s1)[0].clone()

    pc.register_external_params_group("both", [s0, s1])
    pc(data, sum_external_params = {"both": torch.cat([g0, g1], dim = 1)})
    pc.backward(data, flows_memory = 0.0)
    cat = pc.get_external_params_grad("both")[0]

    assert torch.equal(cat, torch.cat([a, b], dim = 1))
