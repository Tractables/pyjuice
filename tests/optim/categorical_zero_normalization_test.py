"""Categorical EM must normalize each node over its own retained categories."""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer import InputLayer


@pytest.mark.parametrize("distribution", ["categorical", "external", "soft_evidence"])
@pytest.mark.parametrize("num_cats", [256, 257, 512, 50262])
@pytest.mark.parametrize("step_size,pseudocount", [(1.0, 0.0), (1.0, 0.3), (0.25, 0.3)])
@pytest.mark.parametrize("keep_zero_params", [False, True])
def test_categorical_em_normalizes_retained_support(
    distribution, num_cats, step_size, pseudocount, keep_zero_params
):
    # Eleven nodes exercise both a full eight-node tile and a partial final tile.
    # 257/50262 categories also exercise masked lanes in the last category tile.
    num_nodes = 11
    if distribution == "categorical":
        dist = dists.Categorical(num_cats = num_cats)
    elif distribution == "external":
        dist = dists.ExternProductCategorical(num_cats = num_cats)
    else:
        # The dual-flow path has a different EM rule and already reduces on axis 1.
        dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False)

    nodes = juice.inputs(0, num_nodes = num_nodes, block_size = 1, dist = dist)
    layer = InputLayer([nodes], pc_num_vars = 1)
    layer._init_parameters(perturbation = 0.0)
    layer.to(torch.device("cuda:0"))
    layer.init_param_flows(flows_memory = 0.0)

    rng = torch.Generator().manual_seed(7)
    row_ids = torch.arange(num_nodes)[:, None]
    zero_mask = (row_ids + torch.arange(num_cats)[None, :]) % 3 == 0
    before = torch.rand(num_nodes, num_cats, generator = rng) + 0.1
    before[zero_mask] = 0.0
    before /= before.sum(dim = 1, keepdim = True)
    # Unequal row totals catch a shared denominator even when it is rescaled.
    flows = (torch.rand(num_nodes, num_cats, generator = rng) + 0.1) * (row_ids + 1)
    layer.params.data.copy_(before.flatten().to(layer.device))
    layer.param_flows.copy_(flows.flatten().to(layer.device))

    weights = flows.double() + pseudocount / num_cats
    if keep_zero_params:
        weights[zero_mask] = 0.0
    target = weights / weights.sum(dim = 1, keepdim = True)
    expected = before.double()

    # Repeated full/partial updates must preserve both normalization and support.
    for _ in range(2):
        expected = (1.0 - step_size) * expected + step_size * target
        layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount,
                            keep_zero_params = keep_zero_params)
        actual = layer.params.detach().cpu().reshape(num_nodes, num_cats)
        torch.testing.assert_close(actual.sum(dim = 1), torch.ones(num_nodes),
                                   atol = 2e-6, rtol = 0)
        torch.testing.assert_close(actual.double(), expected, atol = 2e-7, rtol = 2e-5)
        if keep_zero_params:
            assert torch.equal(actual[zero_mask], torch.zeros_like(actual[zero_mask]))
        else:
            assert bool((actual[zero_mask] > 0).all())
