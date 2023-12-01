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

    layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = 1, maximize_group_size = False)

    layer._init_parameters(perturbation = 2.0)

    assert torch.all(layer.vids == torch.tensor([0,0,1,1,2,2,3,3]).reshape(-1, 1))
    npars_per_group = group_size * ni0.dist.num_parameters()
    assert torch.all(layer.s_pids == torch.arange(0, npars_per_group * 8, npars_per_group))
    assert torch.all(layer.inc_pids == ni0.dist.num_parameters())
    npflows_per_group = group_size * ni0.dist.num_param_flows()
    assert torch.all(layer.s_pfids == torch.arange(0, npflows_per_group * 8, npflows_per_group))
    assert torch.all(layer.inc_pfids == ni0.dist.num_param_flows())
    assert torch.all(layer.metadata == torch.ones([4]) * 2.0)
    assert torch.all(layer.s_mids == torch.tensor([0,0,1,1,2,2,3,3]))
    assert torch.all(layer.source_ngids == torch.arange(0, 8))

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

    import pdb; pdb.set_trace()


def speed_test():

    device = torch.device("cuda:0")

    group_size = 4
    num_vars = 28*28
    num_node_groups = 256 // group_size

    batch_size = 512
    
    with juice.set_group_size(group_size):

        nis = []
        for v in range(num_vars):
            nis.append(inputs(0, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 64)))

    layer = InputLayer(nis, cum_nodes = 1, maximize_group_size = False)

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
    print("Reference computation time on RTX 4090: 0.048ms.")
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
    print("Reference computation time on RTX 4090: 0.062ms.")
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
    print("Reference computation time on RTX 4090: 0.086ms.")
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
    print("Reference computation time on RTX 4090: 0.086ms.")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    # input_layer_test()
    speed_test()