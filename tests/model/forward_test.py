import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def forward_test():

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

    ## Unit tests for forward pass ##

    assert torch.abs(pc.node_mars[1,0] - torch.log(pc.input_layers[0].params[data[0,0]])) < 1e-4
    assert torch.abs(pc.node_mars[2,0] - torch.log(pc.input_layers[0].params[2+data[0,0]])) < 1e-4
    assert torch.abs(pc.node_mars[3,0] - torch.log(pc.input_layers[0].params[4+data[0,1]])) < 1e-4
    assert torch.abs(pc.node_mars[4,0] - torch.log(pc.input_layers[0].params[6+data[0,1]])) < 1e-4
    assert torch.abs(pc.node_mars[5,0] - torch.log(pc.input_layers[0].params[8+data[0,2]])) < 1e-4
    assert torch.abs(pc.node_mars[6,0] - torch.log(pc.input_layers[0].params[10+data[0,2]])) < 1e-4
    assert torch.abs(pc.node_mars[7,0] - torch.log(pc.input_layers[0].params[12+data[0,3]])) < 1e-4
    assert torch.abs(pc.node_mars[8,0] - torch.log(pc.input_layers[0].params[14+data[0,3]])) < 1e-4

    p1 = torch.exp(pc.node_mars[1,0] + pc.node_mars[3,0])
    p2 = torch.exp(pc.node_mars[1,0] + pc.node_mars[4,0])
    p3 = torch.exp(pc.node_mars[2,0] + pc.node_mars[3,0])
    p4 = torch.exp(pc.node_mars[2,0] + pc.node_mars[4,0])
    s11 = p1 * pc.params[1] + p2 * pc.params[2] + p3 * pc.params[3] + p4 * pc.params[4]
    s12 = p1 * pc.params[5] + p2 * pc.params[6] + p3 * pc.params[7] + p4 * pc.params[8]

    assert torch.abs(pc.node_mars[9,0] - torch.log(s11)) < 1e-4
    assert torch.abs(pc.node_mars[10,0] - torch.log(s12)) < 1e-4

    p5 = torch.exp(pc.node_mars[5,0] + pc.node_mars[7,0])
    p6 = torch.exp(pc.node_mars[6,0] + pc.node_mars[8,0])
    s13 = p5 * pc.params[9] + p6 * pc.params[10]
    s14 = p5 * pc.params[11] + p6 * pc.params[12]

    assert torch.abs(pc.node_mars[11,0] - torch.log(s13)) < 1e-4
    assert torch.abs(pc.node_mars[12,0] - torch.log(s14)) < 1e-4

    p7 = torch.exp(pc.node_mars[9,0] + pc.node_mars[11,0])
    p8 = torch.exp(pc.node_mars[10,0] + pc.node_mars[12,0])
    s = p7 * pc.params[13] + p8 * pc.params[14]

    assert torch.abs(pc.node_mars[13,0] - torch.log(s)) < 1e-4


if __name__ == "__main__":
    forward_test()