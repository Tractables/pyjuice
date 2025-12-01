import pyjuice as juice
import torch
import numpy as np
import math

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_soft_evi_fw():
    
    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.External())

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 2]).to(device)

    lls = pc(data)

    ## Input node forward tests ##

    for i in range(16):
        assert torch.abs(pc.node_mars[1,i] - torch.log(pc.input_layer_group[0].params[data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[2,i] - torch.log(pc.input_layer_group[0].params[2+data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[3,i]) < 1e-4
        assert torch.abs(pc.node_mars[4,i]) < 1e-4

    external_soft_evi = torch.rand([16, 1, 2], dtype = torch.float32).to(device)

    lls = pc(data, external_soft_evi = external_soft_evi)

    for i in range(16):
        assert torch.abs(pc.node_mars[1,i] - torch.log(pc.input_layer_group[0].params[data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[2,i] - torch.log(pc.input_layer_group[0].params[2+data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[3,i] - external_soft_evi[i,0,0]) < 1e-4
        assert torch.abs(pc.node_mars[4,i] - external_soft_evi[i,0,1]) < 1e-4


def test_soft_evi_bp():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.External())

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 2]).to(device)

    lls = pc(data)

    pc.backward(data)

    ## Input node backward tests ##

    gt_param_flows = torch.zeros([4], device = pc.node_flows.device)

    for i in range(16):
        gt_param_flows[data[i,0]] += pc.node_flows[1,i]
        gt_param_flows[2+data[i,0]] += pc.node_flows[2,i]

    assert torch.all(torch.abs(gt_param_flows - pc.input_layer_group[0].param_flows) < 1e-4)

    external_soft_evi = torch.rand([16, 1, 2], dtype = torch.float32).to(device)
    external_soft_evi_grad = torch.zeros([16, 1, 2], dtype = torch.float32).to(device)

    lls = pc(data, external_soft_evi = external_soft_evi)

    pc.backward(data, external_soft_evi_grad = external_soft_evi_grad)

    for i in range(16):
        assert torch.abs(pc.node_flows[3,i] - external_soft_evi_grad[i,0,0]) < 1e-4
        assert torch.abs(pc.node_flows[4,i] - external_soft_evi_grad[i,0,1]) < 1e-4


if __name__ == "__main__":
    test_soft_evi_fw()
    test_soft_evi_bp()
