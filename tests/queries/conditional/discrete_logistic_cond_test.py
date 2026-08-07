import pyjuice as juice
import torch
import math

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_discrete_logistic_soft_cond():

    device = torch.device("cuda:0")

    ni0 = inputs(0, num_nodes = 2, dist = dists.DiscreteLogistic(num_cats = 2, val_range = (0.0, 1.0)))
    ni1 = inputs(1, num_nodes = 2, dist = dists.DiscreteLogistic(num_cats = 4, val_range = (0.0, 1.0)))

    m = multiply(ni0, ni1)
    n = summate(m, num_nodes = 1)

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.rand([3, 2, 4]).to(device) # data shape: (B, num_vars, num_cats)
    data[:,0,2:] = 0.0
    missing_mask = torch.tensor([[False, True], [False, True], [False, True]]).to(device) # True for variables to be conditioned on

    outputs = juice.queries.conditional(
        pc, data = data, missing_mask = missing_mask
    ) # output shape: (B, num_vars, num_cats)

    mu0 = pc.input_layer_group[0].params[0]
    s0 = pc.input_layer_group[0].params[1]

    p00 = 1.0 / (1.0 + math.exp((mu0 - 0.5) / s0))
    p01 = 1.0 - p00

    mu1 = pc.input_layer_group[0].params[2]
    s1 = pc.input_layer_group[0].params[3]

    p10 = 1.0 / (1.0 + math.exp((mu1 - 0.5) / s1))
    p11 = 1.0 - p10

    p0 = data[0,0,0] * p00 + data[0,0,1] * p01
    p1 = data[0,0,0] * p10 + data[0,0,1] * p11

    assert (pc.node_mars[1,0].exp() - p0).abs() < 1e-4
    assert (pc.node_mars[2,0].exp() - p1).abs() < 1e-4

    f1 = pc.node_flows[1,0]
    f2 = pc.node_flows[2,0]
    unnorm_o = f1 * torch.tensor([p00, p01], device = device) + \
               f2 * torch.tensor([p10, p11], device = device)
    target_o = unnorm_o / unnorm_o.sum()

    assert torch.all((target_o - outputs[0,0,:2]) < 1e-4)

    mu0 = pc.input_layer_group[0].params[4]
    s0 = pc.input_layer_group[0].params[5]

    p00 = 1.0 / (1.0 + math.exp((mu0 - 0.25) / s0))
    p01 = 1.0 / (1.0 + math.exp((mu0 - 0.5) / s0)) - p00
    p02 = 1.0 / (1.0 + math.exp((mu0 - 0.75) / s0)) - p01 - p00
    p03 = 1.0 - p02 - p01 - p00

    mu1 = pc.input_layer_group[0].params[6]
    s1 = pc.input_layer_group[0].params[7]

    p10 = 1.0 / (1.0 + math.exp((mu1 - 0.25) / s1))
    p11 = 1.0 / (1.0 + math.exp((mu1 - 0.5) / s1)) - p10
    p12 = 1.0 / (1.0 + math.exp((mu1 - 0.75) / s1)) - p11 - p10
    p13 = 1.0 - p12 - p11 - p10

    f1 = pc.node_flows[3,0]
    f2 = pc.node_flows[4,0]
    unnorm_o = f1 * torch.tensor([p00, p01, p02, p03], device = device) + \
               f2 * torch.tensor([p10, p11, p12, p13], device = device)
    target_o = unnorm_o / unnorm_o.sum()

    assert torch.all((target_o - outputs[0,1,:]) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    test_discrete_logistic_soft_cond()
