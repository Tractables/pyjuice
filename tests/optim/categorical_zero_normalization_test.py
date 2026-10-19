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
    # Tiny positive entries are live parameters, not an implicit pruning mask.
    tiny_mask = (~zero_mask) & (torch.arange(num_cats)[None, :] < 5)
    before[tiny_mask] = 1e-16
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
        assert bool((actual[tiny_mask] > 0).all())
        if keep_zero_params:
            assert torch.equal(actual[zero_mask], torch.zeros_like(actual[zero_mask]))
        else:
            assert bool((actual[zero_mask] > 0).all())


@pytest.mark.parametrize("distribution", ["categorical", "external", "soft_evidence"])
@pytest.mark.parametrize("num_cats", [256, 50262])
@pytest.mark.parametrize("retained", [1, 7])
def test_categorical_zero_flow_normalizer(distribution, num_cats, retained):
    """A pseudocount-only update must normalize even very sparse rows."""
    if distribution == "categorical":
        dist = dists.Categorical(num_cats = num_cats)
    elif distribution == "external":
        dist = dists.ExternProductCategorical(num_cats = num_cats)
    else:
        dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False)
    nodes = juice.inputs(0, num_nodes = 11, block_size = 1, dist = dist)
    layer = InputLayer([nodes], pc_num_vars = 1)
    layer._init_parameters(perturbation = 0.0)
    layer.to(torch.device("cuda:0"))
    layer.init_param_flows(flows_memory = 0.0)
    before = torch.zeros(11, num_cats, device = layer.device)
    before[:, :retained] = torch.arange(1, retained + 1, device = layer.device)
    before /= before.sum(1, keepdim = True)
    layer.params.data.copy_(before.flatten())
    layer.mini_batch_em(step_size = 1.0, pseudocount = 1e-6, keep_zero_params = True)
    actual = layer.params.detach().reshape_as(before)
    expected = (before > 0).float() / retained
    torch.testing.assert_close(actual, expected, atol = 2e-7, rtol = 2e-6)
    torch.testing.assert_close(actual.sum(1), torch.ones(11, device = layer.device), atol = 2e-6, rtol = 0)


@pytest.mark.parametrize("num_cats", [256, 257])
@pytest.mark.parametrize("step_size", [1.0, 0.25])
@pytest.mark.parametrize("keep_zero_params", [False, True])
def test_dual_flow_categorical_preserves_tiny_positive_parameters(num_cats, step_size, keep_zero_params):
    nodes = juice.inputs(0, num_nodes = 11, block_size = 1,
                         dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True))
    layer = InputLayer([nodes], pc_num_vars = 1)
    layer._init_parameters(perturbation = 0.0)
    layer.to(torch.device("cuda:0"))
    layer.init_param_flows(flows_memory = 0.0)
    before = torch.ones(11, num_cats)
    before[:, 0] = 0.0
    before[:, 1:3] = torch.tensor([1e-16, 5e-13])
    before /= before.sum(1, keepdim = True)
    layer.params.data.copy_(before.flatten().to(layer.device))
    fp = torch.arange(1, num_cats + 1).float()[None, :].expand_as(before) / num_cats
    fm = torch.ones_like(before)
    layer.param_flows.copy_(torch.stack([fp, fm], dim = 1).flatten().to(layer.device))
    expected = before.double()
    for _ in range(2):
        ratio = (fp.double() + 0.3 / num_cats) / (fm.double() + 0.3 * expected)
        weights = expected * ((1.0 - step_size) + step_size * ratio)
        zero_mask = expected == 0
        expected = (weights / weights.sum(1, keepdim = True)).clamp_min(1e-30)
        if keep_zero_params:
            expected[zero_mask] = 0.0
        layer.mini_batch_em(step_size = step_size, pseudocount = 0.3, keep_zero_params = keep_zero_params)
        actual = layer.params.detach().cpu().reshape_as(before)
        torch.testing.assert_close(actual.double(), expected, atol = 2e-7, rtol = 2e-5)
        # Absolute tolerances alone would incorrectly accept deleting these entries.
        torch.testing.assert_close(actual[:, 1:3].double(), expected[:, 1:3], atol = 0, rtol = 2e-5)
        if keep_zero_params:
            assert bool((actual[:, 0] == 0).all())
        else:
            assert bool((actual[:, 0] > 0).all())
