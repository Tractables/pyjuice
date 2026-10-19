"""The zero-preserving sum update must not apply a magnitude pruning threshold."""
import pytest
import torch
import pyjuice as juice
from pyjuice.model.backend.par_update import em_par_update, sgd_par_update


def _circuit():
    inputs = juice.inputs(0, num_nodes = 8, block_size = 1,
                          dist = juice.distributions.Categorical(num_cats = 2))
    nodes = juice.summate(inputs, num_nodes = 11, block_size = 1)
    root = juice.summate(juice.multiply(nodes), num_nodes = 1, block_size = 1)
    root.init_parameters(perturbation = 0.0)
    before = torch.ones(11, 8)
    before[:, :3] = torch.tensor([0.0, 1e-16, 5e-13])
    before /= before.sum(1, keepdim = True)
    nodes.set_params(before)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    pc.init_param_flows(flows_memory = 0.0)
    return pc, nodes, before


@pytest.mark.parametrize("keep_zero_params", [False, True])
@pytest.mark.parametrize("step_size,pseudocount", [(1.0, 0.0), (1.0, 0.3), (0.25, 0.3)])
def test_sum_em_preserves_exact_zeros_without_pruning_tiny_weights(keep_zero_params, step_size, pseudocount):
    pc, nodes, before = _circuit()
    pc.param_flows.fill_(1.0)
    # Keep one positive entry below the former cutoff across successive updates.
    flows = torch.ones_like(before)
    flows[:, 1] = 1e-15
    pfs, pfe = nodes._param_flow_range
    pc.param_flows[pfs:pfe] = flows.flatten().cuda()
    expected = before.double()
    weights = flows.double() + pseudocount / before.shape[1]
    if keep_zero_params:
        weights[before == 0] = 0
    target = weights / weights.sum(1, keepdim = True)
    for _ in range(2):
        expected = (1 - step_size) * expected + step_size * target
        em_par_update(pc.params, pc.param_flows, pc.par_update_kwargs,
                      step_size = step_size, pseudocount = pseudocount, keep_zero_params = keep_zero_params)
        pc.update_parameters()
        actual = nodes.get_params(as_matrix = True).cpu()
        torch.testing.assert_close(actual.double(), expected, atol = 2e-7, rtol = 2e-5)
        torch.testing.assert_close(actual[:, 1:3].double(), expected[:, 1:3], atol = 0, rtol = 2e-5)
        if keep_zero_params:
            assert bool((actual[:, 0] == 0).all())
        else:
            assert bool((actual[:, 0] > 0).all())


@pytest.mark.parametrize("keep_zero_params", [False, True])
def test_sum_sgd_does_not_prune_tiny_positive_weights(keep_zero_params):
    pc, nodes, before = _circuit()
    gradients = torch.ones_like(pc.param_flows)
    sgd_par_update(pc.params, gradients, pc.par_update_kwargs, lr = 0.1, keep_zero_params = keep_zero_params)
    pc.update_parameters()
    actual = nodes.get_params(as_matrix = True).cpu()
    expected = before.double() * torch.exp(torch.tensor(0.1, dtype = torch.float64))
    torch.testing.assert_close(actual.double(), expected, atol = 0, rtol = 5e-6)
