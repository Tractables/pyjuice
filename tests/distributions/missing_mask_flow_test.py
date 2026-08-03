import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists


# `bk_flow_mask_fn` spreads a marginalized leaf's flow over its whole support. For a `Categorical` that
# means `param_flows[n, c] += flow[n, b] * theta[n, c]` for every category, which used to run once per
# (node, batch, category). It is linear in `flow`, so `_backward_missing` sums the batch axis away first
# and walks (node, category) once. These tests pin the result against the closed form.


def _build_pc(num_vars, num_nodes, num_cats, seed = 0):
    torch.manual_seed(seed)
    nis = [juice.inputs(v, num_nodes = num_nodes, dist = dists.Categorical(num_cats = num_cats))
           for v in range(num_vars)]
    ms = juice.multiply(*nis)
    ns = juice.summate(ms, num_node_blocks = 1, block_size = 1)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(torch.device("cuda:0"))
    return pc


def _run(pc, data, mask, logspace_flows = False):
    pc.init_param_flows(flows_memory = 0.0)
    kw = {} if mask is None else dict(missing_mask = mask)
    pc(data, **kw)
    pc.backward(data, allow_modify_flows = False, logspace_flows = logspace_flows, **kw)
    node_flows = pc.node_flows
    if logspace_flows:
        node_flows = node_flows.exp()
    return pc.input_layer_group[0].param_flows.clone(), node_flows.clone()


def _reference(layer, node_flows, data, node_miss, num_cats):
    """`param_flows` computed straight from the definition, one node at a time."""
    sid = layer._output_ind_range[0]
    vids = layer.vids.view(-1)
    ref = torch.zeros_like(layer.param_flows)
    for n in range(vids.numel()):
        v = int(vids[n])
        pf0, p0 = int(layer.s_pfids[n]), int(layer.s_pids[n])
        flow = node_flows[sid + n]                       # [batch_size]
        theta = layer.params[p0 : p0 + num_cats]
        miss = node_miss[n]                              # [batch_size] bool

        # marginalized: the flow is spread over the support in proportion to theta
        ref[pf0 : pf0 + num_cats] += flow[miss].sum() * theta
        # observed: the flow lands on the observed category
        obs = (~miss).nonzero().flatten()
        for b in obs.tolist():
            ref[pf0 + int(data[b, v])] += flow[b]
    return ref


@pytest.mark.parametrize("logspace_flows", [False, True])
@pytest.mark.parametrize("mask_layout", ["per_var", "per_batch_var", "none_masked", "all_masked"])
def test_categorical_missing_mask_matches_the_closed_form(mask_layout, logspace_flows):
    device = torch.device("cuda:0")
    V, N, C, B = 4, 3, 6, 5

    pc = _build_pc(V, N, C)
    layer = pc.input_layer_group[0]
    data = torch.randint(0, C, (B, V), device = device)

    if mask_layout == "per_var":
        mask = torch.tensor([True, False, True, False], device = device)
        node_miss_full = mask.view(1, V).expand(B, V)
    elif mask_layout == "per_batch_var":
        torch.manual_seed(7)
        mask = torch.randint(0, 2, (B, V), device = device).bool()
        node_miss_full = mask
    elif mask_layout == "none_masked":
        mask = torch.zeros(B, V, dtype = torch.bool, device = device)
        node_miss_full = mask
    else:
        mask = torch.ones(B, V, dtype = torch.bool, device = device)
        node_miss_full = mask

    pf, node_flows = _run(pc, data, mask, logspace_flows)

    vids = layer.vids.view(-1)
    node_miss = node_miss_full[:, vids].t().contiguous()          # [num_nodes, batch_size]
    ref = _reference(layer, node_flows, data, node_miss, C)

    assert torch.isfinite(pf).all()
    assert torch.allclose(pf, ref, rtol = 1e-4, atol = 1e-5), \
        f"max abs diff {float((pf - ref).abs().max()):.3e}"


def test_empty_mask_matches_no_mask():
    device = torch.device("cuda:0")
    V, N, C, B = 4, 3, 6, 5

    pc = _build_pc(V, N, C)
    data = torch.randint(0, C, (B, V), device = device)

    pf_none, _ = _run(pc, data, None)
    pf_false, _ = _run(pc, data, torch.zeros(B, V, dtype = torch.bool, device = device))

    assert torch.allclose(pf_none, pf_false, rtol = 1e-5, atol = 1e-6)


def test_gaussian_missing_mask_only_credits_masked_positions():
    """`Gaussian.bk_flow_mask_fn` does not consult `missing_mask` itself -- it just adds `mu * flow`,
    `(sigma^2 + mu^2) * flow`, `flow`. The old per-(node, batch) launch left `mask` un-narrowed for the
    mask pass, so it fired at OBSERVED positions too, on top of the ordinary flow kernel. Summing the
    masked flow before the call makes the mask structural: an all-False mask must be inert."""
    device = torch.device("cuda:0")
    V, N, B = 3, 2, 4

    torch.manual_seed(0)
    nis = [juice.inputs(v, num_nodes = N, dist = dists.Gaussian(mu = 0.0, sigma = 1.0)) for v in range(V)]
    ns = juice.summate(juice.multiply(*nis), num_node_blocks = 1, block_size = 1)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randn(B, V, device = device)

    pf_none, _ = _run(pc, data, None)
    pf_false, _ = _run(pc, data, torch.zeros(B, V, dtype = torch.bool, device = device))

    assert torch.allclose(pf_none, pf_false, rtol = 1e-5, atol = 1e-6), \
        f"an all-False mask changed the Gaussian statistics (max abs diff {float((pf_none - pf_false).abs().max()):.3e})"


if __name__ == "__main__":
    for layout in ("per_var", "per_batch_var", "none_masked", "all_masked"):
        for ls in (False, True):
            test_categorical_missing_mask_matches_the_closed_form(layout, ls)
    test_empty_mask_matches_no_mask()
    test_gaussian_missing_mask_only_credits_masked_positions()
