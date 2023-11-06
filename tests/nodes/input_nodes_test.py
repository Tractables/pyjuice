import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def categorical_nodes_test():

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

    ## Input node forward tests ##

    for i in range(16):
        assert torch.abs(pc.node_mars[1,i] - torch.log(pc.input_layers[0].params[data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[2,i] - torch.log(pc.input_layers[0].params[2+data[i,0]])) < 1e-4
        assert torch.abs(pc.node_mars[3,i] - torch.log(pc.input_layers[0].params[4+data[i,1]])) < 1e-4
        assert torch.abs(pc.node_mars[4,i] - torch.log(pc.input_layers[0].params[6+data[i,1]])) < 1e-4
        assert torch.abs(pc.node_mars[5,i] - torch.log(pc.input_layers[0].params[8+data[i,2]])) < 1e-4
        assert torch.abs(pc.node_mars[6,i] - torch.log(pc.input_layers[0].params[10+data[i,2]])) < 1e-4
        assert torch.abs(pc.node_mars[7,i] - torch.log(pc.input_layers[0].params[12+data[i,3]])) < 1e-4
        assert torch.abs(pc.node_mars[8,i] - torch.log(pc.input_layers[0].params[14+data[i,3]])) < 1e-4

    ## Input node backward tests ##

    gt_param_flows = torch.zeros([16], device = pc.node_flows.device)

    for i in range(16):
        gt_param_flows[data[i,0]] += pc.node_flows[1,i]
        gt_param_flows[2+data[i,0]] += pc.node_flows[2,i]
        gt_param_flows[4+data[i,1]] += pc.node_flows[3,i]
        gt_param_flows[6+data[i,1]] += pc.node_flows[4,i]
        gt_param_flows[8+data[i,2]] += pc.node_flows[5,i]
        gt_param_flows[10+data[i,2]] += pc.node_flows[6,i]
        gt_param_flows[12+data[i,3]] += pc.node_flows[7,i]
        gt_param_flows[14+data[i,3]] += pc.node_flows[8,i]

    assert torch.all(torch.abs(gt_param_flows - pc.input_layers[0].param_flows) < 1e-4)

    ## EM tests ##

    original_params = pc.input_layers[0].params.clone()

    step_size = 0.3
    pseudocount = 0.1

    par_flows = pc.input_layers[0].param_flows.clone().reshape(8, 2)
    new_params = (1.0 - step_size) * original_params + step_size * ((par_flows + pseudocount / 2) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount)).reshape(-1)

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(new_params - pc.input_layers[0].params) < 1e-4)


def bernoulli_nodes_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Bernoulli())
    ni1 = inputs(1, num_nodes = 2, dist = dists.Bernoulli())
    ni2 = inputs(2, num_nodes = 2, dist = dists.Bernoulli())
    ni3 = inputs(3, num_nodes = 2, dist = dists.Bernoulli())

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

    ## Input node forward tests ##

    for i in range(16):
        assert torch.abs(pc.node_mars[1,i].exp() - (pc.input_layers[0].params[0] if data[i,0] == 1 else (1.0 - pc.input_layers[0].params[0]))) < 1e-4
        assert torch.abs(pc.node_mars[2,i].exp() - (pc.input_layers[0].params[1] if data[i,0] == 1 else (1.0 - pc.input_layers[0].params[1]))) < 1e-4
        assert torch.abs(pc.node_mars[3,i].exp() - (pc.input_layers[0].params[2] if data[i,1] == 1 else (1.0 - pc.input_layers[0].params[2]))) < 1e-4
        assert torch.abs(pc.node_mars[4,i].exp() - (pc.input_layers[0].params[3] if data[i,1] == 1 else (1.0 - pc.input_layers[0].params[3]))) < 1e-4
        assert torch.abs(pc.node_mars[5,i].exp() - (pc.input_layers[0].params[4] if data[i,2] == 1 else (1.0 - pc.input_layers[0].params[4]))) < 1e-4
        assert torch.abs(pc.node_mars[6,i].exp() - (pc.input_layers[0].params[5] if data[i,2] == 1 else (1.0 - pc.input_layers[0].params[5]))) < 1e-4
        assert torch.abs(pc.node_mars[7,i].exp() - (pc.input_layers[0].params[6] if data[i,3] == 1 else (1.0 - pc.input_layers[0].params[6]))) < 1e-4
        assert torch.abs(pc.node_mars[8,i].exp() - (pc.input_layers[0].params[7] if data[i,3] == 1 else (1.0 - pc.input_layers[0].params[7]))) < 1e-4

    ## Input node backward tests ##

    gt_param_flows = torch.zeros([16], device = pc.node_flows.device)

    offsets = 1 - data
    for i in range(16):
        gt_param_flows[offsets[i,0]] += pc.node_flows[1,i]
        gt_param_flows[2+offsets[i,0]] += pc.node_flows[2,i]
        gt_param_flows[4+offsets[i,1]] += pc.node_flows[3,i]
        gt_param_flows[6+offsets[i,1]] += pc.node_flows[4,i]
        gt_param_flows[8+offsets[i,2]] += pc.node_flows[5,i]
        gt_param_flows[10+offsets[i,2]] += pc.node_flows[6,i]
        gt_param_flows[12+offsets[i,3]] += pc.node_flows[7,i]
        gt_param_flows[14+offsets[i,3]] += pc.node_flows[8,i]

    assert torch.all(torch.abs(gt_param_flows - pc.input_layers[0].param_flows) < 1e-4)

    ## EM tests ##

    original_params = pc.input_layers[0].params.clone()

    step_size = 0.3
    pseudocount = 0.1

    par_flows = pc.input_layers[0].param_flows.clone().reshape(8, 2)
    new_params = (1.0 - step_size) * original_params + step_size * ((par_flows + pseudocount / 2) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount))[:,0]

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(new_params - pc.input_layers[0].params) < 1e-4)


if __name__ == "__main__":
    # categorical_nodes_test()
    bernoulli_nodes_test()