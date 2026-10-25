"""
The input layers' partition flows of `eval_partition_grad` (what SGD / Adam subtract from the data flows) against torch:
a node's category c gets exp(node flow) * p_c / sum_c' p_c' * B (with `input_layer_norm_params`, else p_c unnormalized),
negated with `negate_pflows` -- at category counts that are and are not powers of two (the kernel's tile is the next one:
the categories past the count must stay out of the normalizer, the last node's past the parameters' end), with the
caching allocator's free blocks filled with NaN first (what a read past the parameters would see in a long-lived
process).
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.model.backend.partition_grad import eval_partition_grad

DEV = torch.device("cuda:0")


def _poison_free_blocks():
    blocks = [torch.empty((1 << k) // 4, device = DEV).fill_(float("nan")) for k in range(9, 25) for _ in range(2)]
    torch.cuda.synchronize()
    del blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("num_cats", [3, 5, 8, 200])
def test_input_partition_flows_are_the_normalized_parameters(num_cats, negate, norm):
    torch.manual_seed(0)
    x = torch.randint(0, num_cats, (256, 8))
    ns = juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.Categorical(num_cats = num_cats))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to(DEV)
    pc(x[:16].to(DEV))
    pc.init_param_flows(flows_memory = 0.0)
    _poison_free_blocks()
    eval_partition_grad(pc, negate_pflows = negate, input_layer_norm_params = norm)
    B = pc.node_mars.size(1)
    checked = 0
    for layer in pc.input_layer_group:
        nsid, neid = layer._output_ind_range
        cats = torch.arange(num_cats, device = DEV)[None, :]
        p = layer.params[layer.s_pids[:, None] + cats].float()
        want = torch.exp(pc.node_flows[nsid:neid, 0])[:, None] * (p / p.sum(dim = 1, keepdim = True) if norm else p) * B
        if negate:
            want = -want
        got = layer.param_flows[layer.s_pfids[:, None] + cats]
        assert torch.isfinite(got).all(), (num_cats, negate, norm)
        assert torch.allclose(got, want, rtol = 1e-5, atol = 1e-7), \
            (num_cats, negate, norm, float(((got - want).abs() / want.abs().clamp_min(1e-30)).max()))
        checked += 1
    assert checked > 0
