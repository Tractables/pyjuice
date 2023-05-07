import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def backward_test():

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

    pc.backward(data)

    ## Unit tests for backward pass ##

    assert torch.abs(pc.node_flows[13,0] - 1.0) < 1e-4
    p11 = torch.exp(pc.node_mars[9,0] + pc.node_mars[11,0])
    p12 = torch.exp(pc.node_mars[10,0] + pc.node_mars[12,0])
    f11 = pc.node_flows[13,0] * p11 * pc.params[13] / torch.exp(pc.node_mars[13,0])
    f12 = pc.node_flows[13,0] * p12 * pc.params[14] / torch.exp(pc.node_mars[13,0])
    f21, f22, f23, f24 = f11, f12, f11, f12
    assert torch.abs(pc.node_flows[9,0] - f21) < 1e-4
    assert torch.abs(pc.node_flows[10,0] - f22) < 1e-4
    assert torch.abs(pc.node_flows[11,0] - f23) < 1e-4
    assert torch.abs(pc.node_flows[12,0] - f24) < 1e-4
    f31 = pc.node_flows[9,0] * torch.exp(pc.node_mars[1,0] + pc.node_mars[3,0]) * pc.params[1] / torch.exp(pc.node_mars[9,0]) + \
          pc.node_flows[10,0] * torch.exp(pc.node_mars[1,0] + pc.node_mars[3,0]) * pc.params[5] / torch.exp(pc.node_mars[10,0])
    f32 = pc.node_flows[9,0] * torch.exp(pc.node_mars[1,0] + pc.node_mars[4,0]) * pc.params[2] / torch.exp(pc.node_mars[9,0]) + \
          pc.node_flows[10,0] * torch.exp(pc.node_mars[1,0] + pc.node_mars[4,0]) * pc.params[6] / torch.exp(pc.node_mars[10,0])
    f33 = pc.node_flows[9,0] * torch.exp(pc.node_mars[2,0] + pc.node_mars[3,0]) * pc.params[3] / torch.exp(pc.node_mars[9,0]) + \
          pc.node_flows[10,0] * torch.exp(pc.node_mars[2,0] + pc.node_mars[3,0]) * pc.params[7] / torch.exp(pc.node_mars[10,0])
    f34 = pc.node_flows[9,0] * torch.exp(pc.node_mars[2,0] + pc.node_mars[4,0]) * pc.params[4] / torch.exp(pc.node_mars[9,0]) + \
          pc.node_flows[10,0] * torch.exp(pc.node_mars[2,0] + pc.node_mars[4,0]) * pc.params[8] / torch.exp(pc.node_mars[10,0])
    assert torch.abs(pc.node_flows[1,0] - (f31 + f32)) < 1e-4
    assert torch.abs(pc.node_flows[2,0] - (f33 + f34)) < 1e-4
    assert torch.abs(pc.node_flows[3,0] - (f31 + f33)) < 1e-4
    assert torch.abs(pc.node_flows[4,0] - (f32 + f34)) < 1e-4
    f35 = pc.node_flows[11,0] * torch.exp(pc.node_mars[5,0] + pc.node_mars[7,0]) * pc.params[9] / torch.exp(pc.node_mars[11,0]) + \
          pc.node_flows[12,0] * torch.exp(pc.node_mars[5,0] + pc.node_mars[7,0]) * pc.params[11] / torch.exp(pc.node_mars[12,0])
    f36 = pc.node_flows[11,0] * torch.exp(pc.node_mars[6,0] + pc.node_mars[8,0]) * pc.params[10] / torch.exp(pc.node_mars[11,0]) + \
          pc.node_flows[12,0] * torch.exp(pc.node_mars[6,0] + pc.node_mars[8,0]) * pc.params[12] / torch.exp(pc.node_mars[12,0])
    assert torch.abs(pc.node_flows[5,0] - f35) < 1e-4
    assert torch.abs(pc.node_flows[6,0] - f36) < 1e-4
    assert torch.abs(pc.node_flows[7,0] - f35) < 1e-4
    assert torch.abs(pc.node_flows[8,0] - f36) < 1e-4

    pf11 = (pc.node_flows[13,:] * torch.exp(pc.node_mars[9,:] + pc.node_mars[11,:]) * pc.params[13].unsqueeze(0) / torch.exp(pc.node_mars[13,:])).sum(dim = 0)
    pf12 = (pc.node_flows[13,:] * torch.exp(pc.node_mars[10,:] + pc.node_mars[12,:]) * pc.params[14].unsqueeze(0) / torch.exp(pc.node_mars[13,:])).sum(dim = 0)
    assert torch.abs(pc.param_flows[13] - pf11) < 1e-4
    assert torch.abs(pc.param_flows[14] - pf12) < 1e-4
    pf21 = pc.node_flows[9,:] * torch.exp(pc.node_mars[1,:] + pc.node_mars[3,:]) * pc.params[1].unsqueeze(0) / torch.exp(pc.node_mars[9,:])
    pf22 = pc.node_flows[9,:] * torch.exp(pc.node_mars[1,:] + pc.node_mars[4,:]) * pc.params[2].unsqueeze(0) / torch.exp(pc.node_mars[9,:])
    pf23 = pc.node_flows[9,:] * torch.exp(pc.node_mars[2,:] + pc.node_mars[3,:]) * pc.params[3].unsqueeze(0) / torch.exp(pc.node_mars[9,:])
    pf24 = pc.node_flows[9,:] * torch.exp(pc.node_mars[2,:] + pc.node_mars[4,:]) * pc.params[4].unsqueeze(0) / torch.exp(pc.node_mars[9,:])
    pf25 = pc.node_flows[10,:] * torch.exp(pc.node_mars[1,:] + pc.node_mars[3,:]) * pc.params[5].unsqueeze(0) / torch.exp(pc.node_mars[10,:])
    pf26 = pc.node_flows[10,:] * torch.exp(pc.node_mars[1,:] + pc.node_mars[4,:]) * pc.params[6].unsqueeze(0) / torch.exp(pc.node_mars[10,:])
    pf27 = pc.node_flows[10,:] * torch.exp(pc.node_mars[2,:] + pc.node_mars[3,:]) * pc.params[7].unsqueeze(0) / torch.exp(pc.node_mars[10,:])
    pf28 = pc.node_flows[10,:] * torch.exp(pc.node_mars[2,:] + pc.node_mars[4,:]) * pc.params[8].unsqueeze(0) / torch.exp(pc.node_mars[10,:])
    assert torch.abs(pc.param_flows[1] - pf21.sum(dim = 0)) < 1e-4
    assert torch.abs(pc.param_flows[2] - pf22.sum(dim = 0)) < 1e-4
    assert torch.abs(pc.param_flows[3] - pf23.sum(dim = 0)) < 1e-4
    assert torch.abs(pc.param_flows[4] - pf24.sum(dim = 0)) < 1e-4
    assert torch.abs(pc.param_flows[5] - pf25.sum(dim = 0)) < 1e-4
    assert torch.abs(pc.param_flows[6] - pf26.sum(dim = 0)) < 1e-4
    assert torch.abs(pc.param_flows[7] - pf27.sum(dim = 0)) < 1e-4
    assert torch.abs(pc.param_flows[8] - pf28.sum(dim = 0)) < 1e-4

    assert torch.abs((pc.node_flows[1,:] * (data[:,0] == 0)).sum() - pc.input_layers[0].param_flows[0]) < 1e-4
    assert torch.abs((pc.node_flows[1,:] * (data[:,0] == 1)).sum() - pc.input_layers[0].param_flows[1]) < 1e-4
    assert torch.abs((pc.node_flows[2,:] * (data[:,0] == 0)).sum() - pc.input_layers[0].param_flows[2]) < 1e-4
    assert torch.abs((pc.node_flows[2,:] * (data[:,0] == 1)).sum() - pc.input_layers[0].param_flows[3]) < 1e-4
    assert torch.abs((pc.node_flows[3,:] * (data[:,1] == 0)).sum() - pc.input_layers[0].param_flows[4]) < 1e-4
    assert torch.abs((pc.node_flows[3,:] * (data[:,1] == 1)).sum() - pc.input_layers[0].param_flows[5]) < 1e-4
    assert torch.abs((pc.node_flows[4,:] * (data[:,1] == 0)).sum() - pc.input_layers[0].param_flows[6]) < 1e-4
    assert torch.abs((pc.node_flows[4,:] * (data[:,1] == 1)).sum() - pc.input_layers[0].param_flows[7]) < 1e-4
    assert torch.abs((pc.node_flows[5,:] * (data[:,2] == 0)).sum() - pc.input_layers[0].param_flows[8]) < 1e-4
    assert torch.abs((pc.node_flows[5,:] * (data[:,2] == 1)).sum() - pc.input_layers[0].param_flows[9]) < 1e-4
    assert torch.abs((pc.node_flows[6,:] * (data[:,2] == 0)).sum() - pc.input_layers[0].param_flows[10]) < 1e-4
    assert torch.abs((pc.node_flows[6,:] * (data[:,2] == 1)).sum() - pc.input_layers[0].param_flows[11]) < 1e-4
    assert torch.abs((pc.node_flows[7,:] * (data[:,3] == 0)).sum() - pc.input_layers[0].param_flows[12]) < 1e-4
    assert torch.abs((pc.node_flows[7,:] * (data[:,3] == 1)).sum() - pc.input_layers[0].param_flows[13]) < 1e-4
    assert torch.abs((pc.node_flows[8,:] * (data[:,3] == 0)).sum() - pc.input_layers[0].param_flows[14]) < 1e-4
    assert torch.abs((pc.node_flows[8,:] * (data[:,3] == 1)).sum() - pc.input_layers[0].param_flows[15]) < 1e-4

    ## Unit tests for params ##

    inner_param_flows = pc.param_flows.clone()
    pc._normalize_parameters(inner_param_flows)
    assert torch.abs(pf21.sum(dim = 0) / pf22.sum(dim = 0) - inner_param_flows[1] / inner_param_flows[2]) < 1e-4
    assert torch.abs(pf11 / pf12 - inner_param_flows[13] / inner_param_flows[14]) < 1e-4
    assert torch.abs(inner_param_flows[13] + inner_param_flows[14] - 1.0) < 1e-4


if __name__ == "__main__":
    backward_test()