import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def non_sd_test():
    ni1 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni2 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni3 = inputs(2, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni4 = inputs(3, num_nodes = 2, dist = dists.Categorical(num_cats = 2))

    np12 = multiply(ni1, ni2)
    np23 = multiply(ni2, ni3)
    np34 = multiply(ni3, ni4)

    ns12 = summate(np12, num_nodes = 2)
    ns23 = summate(np23, num_nodes = 2)
    ns34 = summate(np34, num_nodes = 2)

    np1 = multiply(ns12, ns34)
    np2 = multiply(ni1, ns23, ni4)
    np3 = multiply(ni1, ni2, ns34)

    ns = summate(np1, np2, np3, num_nodes = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    pc.update_parameters()

    ## Unit tests for forward pass ##

    fw12 = ni1._params.reshape(2, 2)[:,data[0,0]]
    fw34 = ni2._params.reshape(2, 2)[:,data[0,1]]
    fw56 = ni3._params.reshape(2, 2)[:,data[0,2]]
    fw78 = ni4._params.reshape(2, 2)[:,data[0,3]]

    assert torch.all(torch.abs(fw12 - pc.node_mars[1:3,0].exp().cpu()) < 1e-4)
    assert torch.all(torch.abs(fw34 - pc.node_mars[3:5,0].exp().cpu()) < 1e-4)
    assert torch.all(torch.abs(fw56 - pc.node_mars[5:7,0].exp().cpu()) < 1e-4)
    assert torch.all(torch.abs(fw78 - pc.node_mars[7:9,0].exp().cpu()) < 1e-4)

    fw910 = torch.matmul(ns12._params.reshape(2, 2), (fw12 * fw34).reshape(2, 1)).reshape(-1)
    fw1112 = torch.matmul(ns34._params.reshape(2, 2), (fw56 * fw78).reshape(2, 1)).reshape(-1)
    fw1314 = torch.matmul(ns23._params.reshape(2, 2), (fw34 * fw56).reshape(2, 1)).reshape(-1)

    assert torch.all(torch.abs(fw910.log() - pc.node_mars[9:11,0].cpu()) < 1e-4)
    assert torch.all(torch.abs(fw1112.log() - pc.node_mars[11:13,0].cpu()) < 1e-4)
    assert torch.all(torch.abs(fw1314.log() - pc.node_mars[13:15,0].cpu()) < 1e-4)

    fwnp1 = fw910 * fw1112
    fwnp2 = fw12 * fw1314 * fw78
    fwnp3 = fw12 * fw34 * fw1112
    fw15 = (ns._params * torch.cat((fwnp1, fwnp2, fwnp3), dim = 0)).sum()

    assert torch.all(torch.abs(fw15.log() - pc.node_mars[15,0].cpu()) < 1e-4)

    ## Unit tests for backward pass ##

    assert torch.abs(pc.node_flows[15,0] - 1.0) < 1e-4

    bknp1 = ns._params[0:2] * fwnp1 / fw15
    bknp2 = ns._params[2:4] * fwnp2 / fw15
    bknp3 = ns._params[4:6] * fwnp3 / fw15

    bk910 = bknp1
    bk1112 = bknp1 + bknp3
    bk1314 = bknp2

    assert torch.all(torch.abs(bk910 - pc.node_flows[9:11,0].cpu()) < 1e-4)
    assert torch.all(torch.abs(bk910 - pc.node_flows[9:11,0].cpu()) < 1e-4)
    assert torch.all(torch.abs(bk910 - pc.node_flows[9:11,0].cpu()) < 1e-4)

    bknp12 = bk910[0] * ns12._params[0:2] * fw12 * fw34 / fw910[0] + bk910[1] * ns12._params[2:4] * fw12 * fw34 / fw910[1]
    bknp23 = bk1314[0] * ns23._params[0:2] * fw34 * fw56 / fw1314[0] + bk1314[1] * ns23._params[2:4] * fw34 * fw56 / fw1314[1]
    bknp34 = bk1112[0] * ns34._params[0:2] * fw56 * fw78 / fw1112[0] + bk1112[1] * ns34._params[2:4] * fw56 * fw78 / fw1112[1]

    bk12 = bknp12 + bknp2 + bknp3
    bk34 = bknp12 + bknp23 + bknp3
    bk56 = bknp34 + bknp23
    bk78 = bknp34 + bknp2

    assert torch.all(torch.abs(bk12 - pc.node_flows[1:3,0].cpu()) < 1e-4)
    assert torch.all(torch.abs(bk34 - pc.node_flows[3:5,0].cpu()) < 1e-4)
    assert torch.all(torch.abs(bk56 - pc.node_flows[5:7,0].cpu()) < 1e-4)
    assert torch.all(torch.abs(bk78 - pc.node_flows[7:9,0].cpu()) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(129)
    non_sd_test()