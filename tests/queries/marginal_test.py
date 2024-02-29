import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def cat_hard_marginal_test():

    device = torch.device("cuda:0")

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 4))

    m = multiply(ni0, ni1)
    n = summate(m, num_nodes = 1)

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [3, 2]).to(device)
    missing_mask = torch.tensor([[False, True], [False, True], [False, True]]).to(device) # True for variables to be conditioned on

    lls = juice.queries.marginal(
        pc, data = data, missing_mask = missing_mask
    )

    p0 = pc.input_layer_group[0].params[data[:,0]]
    p1 = pc.input_layer_group[0].params[2+data[:,0]]

    assert torch.all((torch.log(p0 * pc.params[1] + p1 * pc.params[2]) - lls[:,0]).abs() < 1e-4)


def cat_soft_marginal_test():

    device = torch.device("cuda:0")

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 4))

    m = multiply(ni0, ni1)
    n = summate(m, num_nodes = 1)

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.rand([3, 2, 4]).to(device) # data shape: (B, num_vars, num_cats)
    data[:,0,2:] = 0.0
    missing_mask = torch.tensor([[False, True], [False, True], [False, True]]).to(device) # True for variables to be conditioned on

    lls = juice.queries.marginal(
        pc, data = data, missing_mask = missing_mask
    )

    p0 = data[:,0,0] * pc.input_layer_group[0].params[0] + data[:,0,1] * pc.input_layer_group[0].params[1]
    p1 = data[:,0,0] * pc.input_layer_group[0].params[2] + data[:,0,1] * pc.input_layer_group[0].params[3]

    assert torch.all((torch.log(p0 * pc.params[1] + p1 * pc.params[2]) - lls[:,0]).abs() < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    cat_hard_marginal_test()
    cat_soft_marginal_test()