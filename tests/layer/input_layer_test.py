import pyjuice as juice
import torch
import numpy as np
import time

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

from pyjuice.layer import InputLayer

import pytest


def input_layer_test():

    device = torch.device("cuda:0")

    group_size = 4
    batch_size = 16
    
    with juice.set_group_size(group_size):

        ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))

    layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = 1)

    layer._init_parameters(perturbation = 2.0)

    assert torch.all(layer.vids == torch.tensor([0,1,2,3]).unsqueeze(1).repeat(1, 8).reshape(-1, 1))
    assert torch.all(layer.s_pids == torch.arange(0, 32 * 2, 2))
    assert torch.all(layer.s_pfids == torch.arange(0, 32 * 2, 2))
    assert torch.all(layer.metadata == torch.ones([4]) * 2.0)
    assert torch.all(layer.s_mids == torch.tensor([0,1,2,3]).unsqueeze(1).repeat(1, 8).reshape(-1))
    assert torch.all(layer.source_nids == torch.arange(0, 32))

    layer.to(device)

    data = torch.randint(0, 2, (4, batch_size)).to(device)
    node_mars = torch.zeros([33, batch_size]).to(device)

    ## Forward tests ##

    layer(data, node_mars)

    for i in range(16):
        for j in range(4 * 2 * group_size):
            assert torch.abs(node_mars[j+1,i].exp() - layer.params[j*2+data[j//(2*group_size),i]]) < 1e-4

    ## Forward with mask tests ##

    missing_mask = torch.tensor([0,1,0,1]).bool().to(device)

    layer(data, node_mars, missing_mask = missing_mask)

    for i in range(16):
        for j in range(4 * 2 * group_size):
            v = j//(2*group_size)
            if v == 0 or v == 2:
                assert torch.abs(node_mars[j+1,i].exp() - layer.params[j*2+data[v,i]]) < 1e-4
            else:
                assert torch.abs(node_mars[j+1,i].exp() - 1.0) < 1e-4

    missing_mask = torch.randint(0, 2, (4, batch_size)).bool().to(device)

    layer(data, node_mars, missing_mask = missing_mask)

    for i in range(16):
        for j in range(4 * 2 * group_size):
            v = j // (2*group_size)
            if not missing_mask[v,i]:
                assert torch.abs(node_mars[j+1,i].exp() - layer.params[j*2+data[v,i]]) < 1e-4
            else:
                assert torch.abs(node_mars[j+1,i].exp() - 1.0) < 1e-4

    ## Backward tests ##

    node_flows = torch.rand([33, batch_size]).to(device)

    layer.init_param_flows(flows_memory = 0.0)

    layer(data, node_mars)
    layer.backward(data, node_flows, node_mars)

    param_flows = torch.zeros([group_size * 2 * 4 * 2]).to(device)

    for i in range(16):
        for j in range(4 * 2 * group_size):
            v = j // (2*group_size)

            param_flows[j*2+data[j//(2*group_size),i]] += node_flows[j+1,i]

    assert torch.all(torch.abs(layer.param_flows - param_flows) < 1e-4)

    ## EM tests ##

    original_params = layer.params.clone()

    step_size = 0.3
    pseudocount = 0.1

    par_flows = layer.param_flows.clone().reshape(32, 2)
    new_params = (1.0 - step_size) * original_params + step_size * ((par_flows + pseudocount / 2) / (par_flows.sum(dim = 1, keepdim = True) + pseudocount)).reshape(-1)

    layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(new_params - layer.params) < 1e-4)


def tied_bp_test():

    device = torch.device("cuda:0")

    group_size = 4
    batch_size = 16
    
    with juice.set_group_size(group_size):

        ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = ni1.duplicate(3, tie_params = True)

    layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = 1, max_tied_ns_per_parflow_group = 1.0)

    layer._init_parameters(perturbation = 2.0)

    assert torch.all(layer.vids == torch.tensor([0,1,2,3]).unsqueeze(1).repeat(1, 8).reshape(-1, 1))
    s_pids = torch.arange(0, 32 * 2, 2)
    s_pids[24:32] = s_pids[8:16]
    assert torch.all(layer.s_pids == s_pids)
    assert torch.all(layer.s_pfids == torch.arange(0, 32 * 2, 2))
    assert torch.all(layer.metadata == torch.ones([4]) * 2.0)
    assert torch.all(layer.s_mids == torch.tensor([0,1,2,3]).unsqueeze(1).repeat(1, 8).reshape(-1))
    assert torch.all(layer.source_nids == torch.arange(0, 24))

    assert layer.tied2source_nids[0][0] == 16
    assert layer.tied2source_nids[0][1] == 16
    assert torch.all(layer.tied2source_nids[0][2] == torch.tensor([16, 48]))

    layer.to(device)

    data = torch.randint(0, 2, (4, batch_size)).to(device)
    node_mars = torch.zeros([33, batch_size]).to(device)
    node_flows = torch.rand([33, batch_size]).to(device)

    step_size = 0.3
    pseudocount = 0.1

    ## EM tests ##

    layer.init_param_flows(flows_memory = 0.0)

    layer(data, node_mars)
    layer.backward(data, node_flows, node_mars)

    param_flows = layer.param_flows.detach().clone()
    param_flows[16:32] += param_flows[48:64]

    layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    assert torch.all(torch.abs(param_flows - layer.param_flows) < 1e-4)


def speed_test():

    device = torch.device("cuda:0")

    group_size = 4
    num_vars = 28*28
    num_node_groups = 256 // group_size

    batch_size = 512
    
    with juice.set_group_size(group_size):

        nis = []
        for v in range(num_vars):
            nis.append(inputs(v, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 64)))

    layer = InputLayer(nis, cum_nodes = 1)

    layer._init_parameters(perturbation = 2.0)

    layer.to(device)

    data = torch.randint(0, 64, (num_vars, batch_size)).to(device)
    node_mars = torch.zeros([1 + group_size * num_node_groups * num_vars, batch_size]).to(device)

    ## Forward tests ##

    layer(data, node_mars)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(data, node_mars)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Forward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 0.533ms.")
    print("--------------------------------------------------------------")

    ## Forward with mask tests ##

    missing_mask = torch.randint(0, 2, (num_vars,)).bool().to(device)

    layer(data, node_mars, missing_mask = missing_mask)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(data, node_mars, missing_mask = missing_mask)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Forward pass (w/ sample independent mask) on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 1.434ms.")
    print("--------------------------------------------------------------")

    missing_mask = torch.randint(0, 2, (num_vars, batch_size)).bool().to(device)

    layer(data, node_mars, missing_mask = missing_mask)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(data, node_mars, missing_mask = missing_mask)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Forward pass (w/ sample dependent mask) on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 1.431ms.")
    print("--------------------------------------------------------------")

    ## Backward tests ##

    node_flows = torch.rand([1 + group_size * num_node_groups * num_vars, batch_size]).to(device)

    layer.init_param_flows(flows_memory = 0.0)

    layer(data, node_mars)
    layer.backward(data, node_flows, node_mars)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer.backward(data, node_flows, node_mars)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Backward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 0.825ms.")
    print("--------------------------------------------------------------")

    ## EM tests ##

    step_size = 0.01
    pseudocount = 0.1

    layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer.mini_batch_em(step_size = step_size, pseudocount = pseudocount)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"EM on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 0.784ms.")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    input_layer_test()
    tied_bp_test()
    speed_test()