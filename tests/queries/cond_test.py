import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def cond_test():

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

    outputs = juice.queries.conditional(
        pc, missing_mask, soft_evidence = data
    ) # output shape: (B, num_vars, num_cats)

    p0 = data[0,0,0] * pc.input_layers[0].params[0] + data[0,0,1] * pc.input_layers[0].params[1]
    p1 = data[0,0,0] * pc.input_layers[0].params[2] + data[0,0,1] * pc.input_layers[0].params[3]

    assert (pc.node_mars[1,0].exp() - p0).abs() < 1e-4
    assert (pc.node_mars[2,0].exp() - p1).abs() < 1e-4

    import pdb; pdb.set_trace()

    f1 = pc.node_flows[3,0]
    f2 = pc.node_flows[4,0]
    unnorm_o = f1 * pc.input_layers[0].params[4:8] + f2 * pc.input_layers[0].params[8:12]
    target_o = unnorm_o / unnorm_o.sum()

    assert torch.all((target_o - outputs[0,1,:]) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    cond_test()