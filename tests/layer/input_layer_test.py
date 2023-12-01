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
    
    with juice.set_group_size(group_size):

        ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))

    layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = 1)

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

    data = torch.randint(0, 2, (4, 16)).to(device)
    node_mars = torch.zeros([33, 16]).to(device)

    ## Forward tests ##

    layer(data, node_mars)

    for i in range(16):
        for j in range(4 * 2 * group_size):
            assert torch.abs(node_mars[j+1,i].exp() - layer.params[j*2+data[j//(2*group_size),i]]) < 1e-4

    ## Forward with mask tests ##



    import pdb; pdb.set_trace()


def speed_test():

    device = torch.device("cuda:0")

    group_size = 128
    num_vars = 16*16*3
    num_node_groups = 256 // group_size

    batch_size = 512
    
    with juice.set_group_size(group_size):

        nis = []
        for v in range(num_vars):
            nis.append(inputs(0, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 64)))

    layer = InputLayer(nis, cum_nodes = 1)

    layer._init_parameters(perturbation = 2.0)

    layer.to(device)

    data = torch.randint(0, 64, (num_vars, batch_size)).to(device)
    node_mars = torch.zeros([1 + group_size * num_node_groups * num_vars, 16]).to(device)

    ## Forward tests ##

    layer(data, node_mars)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(data, node_mars)
    torch.cuda.synchronize()
    t1 = time.time()
    print((t1 - t0) / 100 * 1000)


if __name__ == "__main__":
    # input_layer_test()
    speed_test()