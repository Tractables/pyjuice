import pyjuice as juice
import torch
import numpy as np
from copy import deepcopy

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit
from pyjuice.nodes.methods import get_subsumed_scopes

import pytest


def partial_eval_forward_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni2 = inputs(2, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni3 = inputs(3, num_nodes = 2, dist = dists.Categorical(num_cats = 2))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    lls = pc(data)

    pseudo_data = data.clone()
    pseudo_data[:,1] = 0
    _, cache = pc(pseudo_data, return_cache = True)

    scopes = get_subsumed_scopes(n, [1])
    pc.enable_partial_evaluation(scopes = scopes, forward = True)

    lls2 = pc(data, cache = cache)

    assert torch.all(torch.abs(lls - lls2) < 1e-4)

    assert (pc.input_layers[0].fw_local_ids.cpu() == torch.tensor([2, 3])).all()
    assert (pc.inner_layers[0].fw_group_local_ids[0].cpu() == torch.tensor([0, 1, 2, 3])).all()
    assert (pc.inner_layers[1].fw_group_local_ids[1].cpu() == torch.tensor([0, 1])).all()

    for var in range(4):
        pseudo_data = data.clone()
        pseudo_data[:,var] = 0
        _, cache = pc(pseudo_data, return_cache = True)

        scopes = get_subsumed_scopes(n, [var])
        pc.enable_partial_evaluation(scopes = scopes, forward = True)

        lls2 = pc(data, cache = cache)

        assert torch.all(torch.abs(lls - lls2) < 1e-4)


def partial_eval_backward_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni2 = inputs(2, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni3 = inputs(3, num_nodes = 2, dist = dists.Categorical(num_cats = 2))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    lls, cache = pc(data, return_cache = True)
    cache = pc.backward(data, cache = cache, return_cache = True)

    scopes = get_subsumed_scopes(n, [1])
    pc.enable_partial_evaluation(scopes = scopes, forward = False, backward = True)

    cache2 = pc.backward(data, cache = deepcopy(cache), return_cache = True)

    assert torch.all(cache["node_flows"] == cache2["node_flows"])

    assert (pc.input_layers[0].bk_local_ids.cpu() == torch.tensor([2, 3])).all()
    assert (pc.inner_layers[0].bk_group_local_ids[1].cpu() == torch.tensor([2, 3])).all()
    assert (pc.inner_layers[1].bk_group_local_ids[0].cpu() == torch.tensor([0, 1, 2, 3])).all()
    assert (pc.inner_layers[2].bk_group_local_ids[0].cpu() == torch.tensor([0, 1])).all()


if __name__ == "__main__":
    partial_eval_forward_test()
    partial_eval_backward_test()
