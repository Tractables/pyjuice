"""The per-block element fork of `BlockScale` (`kernels/blockscale_backward_blocks.py`, the element plan's kind
("blocks", splits)): where a gated layer's parent blocks are narrower than its k-tile (`ptr_inc_step > 1`), the parents a
block at a time, split into ranges over the grid and merged. Against the Triton fork it was cut from, forced in turn
through the plan's cached choice (as `external_blockscale_grad_test.py` forces the forks): the same element flows (seen
through the node flows below them), gate gradient and parameter flows."""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.parameterizations import BlockScale
from pyjuice.nodes.parameterized_sum_nodes import ParameterizedSumNodes
from pyjuice.utils.launch_plans import LaunchPlans

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _build(L, T, seed = 0):
    torch.manual_seed(seed)
    ns = juice.structures.GeneralizedHMM(seq_length = T, num_latents = L, homogeneous = True, block_size = 32,
                                         input_dist = dists.Categorical(num_cats = 24))
    for s in ns:
        if s.is_sum() and s.num_nodes == L:
            s.__class__ = ParameterizedSumNodes
            s.parameterization = BlockScale(ch_block_size = 32)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(torch.device("cuda:0"))


def _plans(pc):
    out = []
    for lg in pc.inner_layer_groups:
        for l in lg.layers:
            for k, v in (getattr(l, "_bs_bw_gate_cache", None) or {}).items():
                if isinstance(k, tuple) and k and k[0] == "eleplan":
                    out.append((l, v))
    return out


def _step(pc, x, gates, with_grad, kind):
    """A forward and backward with every gated layer's element fork forced to `kind` (None: as chosen)."""
    if kind is not None:
        for layer, plan in _plans(pc):
            plan["kind"] = kind
            LaunchPlans.of(layer).clear()
        LaunchPlans.of(pc).clear()
    grads = {s: torch.zeros_like(p) for s, p in gates.items()} if with_grad else None
    pc.init_param_flows(flows_memory = 0.0)
    pc(x, sum_external_params = gates)
    if with_grad:
        pc.backward(x, sum_external_params = gates, sum_external_params_grad = grads)
    else:
        pc.backward(x, sum_external_params = gates)
    g = torch.cat([v.flatten() for v in grads.values()]) if with_grad else None
    return pc.node_flows.clone(), g, pc.param_flows.clone()


@cuda_only
@pytest.mark.parametrize("L,B", [(256, 64), (256, 16), (512, 24), (128, 13), (384, 32)])
@pytest.mark.parametrize("with_grad", [True, False])
def test_the_per_block_fork_is_the_triton_forks(L, B, with_grad):
    pc = _build(L, T = 6)
    x = torch.randint(0, 24, (B, 6), device = torch.device("cuda:0"))
    g = torch.Generator(device = "cuda").manual_seed(3)
    gates = {s: 2.0 * torch.randn(s.parameterization.tensor_shapes(s, B)[0], device = "cuda", generator = g)
             for s in pc.external_params_nodes}
    _step(pc, x, gates, with_grad, None)                  # builds the plans
    plans = _plans(pc)
    assert plans and all("blocks" in p and p["blocks"] is not None for _, p in plans), \
        "the per-block fork was not offered: no layer with parent blocks narrower than its k-tile"
    steps = plans[0][1]["blocks"][0].size(1)
    assert steps > 1

    nf_t, g_t, pf_t = _step(pc, x, gates, with_grad, "triton")
    fin = torch.isfinite(nf_t)
    # (12 parent blocks at 384 latents: ranges of 2 by 8 splits, the last two past the end -- the range guard)
    for splits in sorted({min(sp, steps) for sp in (1, 2, 4, 8)} | {steps}):
        nf, gg, pf = _step(pc, x, gates, with_grad, ("blocks", splits))
        assert torch.equal(torch.isfinite(nf), fin), f"splits {splits}: the node flows' -inf pattern differs"
        assert (nf[fin] - nf_t[fin]).abs().max() < 1e-4 * max(1.0, float(nf_t[fin].abs().max())), \
            f"splits {splits}: node flows off by {(nf[fin] - nf_t[fin]).abs().max():.3e}"
        # From batch 16 both forks' dots are TF32 (rounded operands), each step's operands normalized at its own maximum
        # -- a k-tile's there, a parent block's here --, so the parameter flows below differ at TF32's rounding; below
        # 16 both are fp32, and agree to fp32's
        rtol = 3e-3 if B >= 16 else 1e-4
        assert torch.allclose(pf, pf_t, rtol = rtol, atol = 1e-7), \
            f"splits {splits}: parameter flows off by {((pf - pf_t).abs() / pf_t.abs().clamp(min = 1e-7)).max():.3e} (relative)"
        if with_grad:
            # both forks TF32-round the same products; the gradient is a difference of near-equal terms
            assert (gg - g_t).abs().max() < 2e-3 * float(g_t.abs().max()), \
                f"splits {splits}: gate gradient off by {(gg - g_t).abs().max() / g_t.abs().max():.3e} (relative)"


@cuda_only
@pytest.mark.parametrize("nb,B", [(4, 64), (4, 16), (8, 24), (2, 13)])
def test_the_per_block_fork_against_float64(nb, B):
    """One gated layer of `nb` blocks of 32 over a product of two inputs (its element flows are the inputs' node flows),
    against float64 from the circuit's own buffers: no less accurate than the Triton fork."""
    from pyjuice.nodes import inputs, multiply, summate
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ni = [inputs(v, num_node_blocks = nb, dist = dists.Categorical(num_cats = 24)) for v in range(2)]
        prod = multiply(*ni)
        ns = summate(prod, num_node_blocks = nb, parameterization = BlockScale(ch_block_size = 32))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)
    x = torch.randint(0, 24, (B, 2), device = dev)
    g = torch.Generator(device = "cuda").manual_seed(5)
    phi = 2.0 * torch.randn(ns.parameterization.tensor_shapes(ns, B)[0], device = "cuda", generator = g)
    gates = {ns: phi}
    _step(pc, x, gates, True, None)
    plans = _plans(pc)
    assert plans and plans[0][1].get("blocks") is not None, "the per-block fork was not offered"

    K = nb * 32
    lo, hi = ns._output_ind_range
    elo = prod._output_ind_range[0]
    ilo = ni[0]._output_ind_range[0]
    errs = {}
    for kind in ["triton"] + [("blocks", s) for s in (1, 2)]:
        _step(pc, x, gates, True, kind)
        nm = pc.node_mars[lo:hi, :B].double()                     # log N - log Z, as the forward left it
        em = pc.element_mars[elo:elo + K, :B].double()
        f = pc.node_flows[lo:hi, :B].double()
        theta = torch.zeros(K, K, dtype = torch.float64, device = dev)             # [n, c] from the edge blocks
        for e, (i, j) in enumerate(ns.edge_ids.t().tolist()):
            theta[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = pc.get_node_params(ns).double()[e]
        lphi = phi.double().view(B, nb, K // 32)                   # [B, node block, child gate]
        tt = lphi.repeat_interleave(32, dim = 1).repeat_interleave(32, dim = 2).exp() * theta[None]   # [B, n, c]
        tb = tt / tt.sum(dim = 2, keepdim = True)
        ref = torch.logsumexp(f.T[:, :, None] + tb.log() + em.T[:, None, :] - nm.T[:, :, None], dim = 1)   # [B, c]
        got = pc.node_flows[ilo:ilo + K, :B].double().T
        errs[kind] = float((got - ref).abs().max())
    for kind, e in errs.items():
        assert e < max(2.0 * errs["triton"], 2e-5), f"{kind}: {e:.3e} against float64 (Triton fork {errs['triton']:.3e})"
