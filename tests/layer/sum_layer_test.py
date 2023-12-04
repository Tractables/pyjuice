import pyjuice as juice
import torch
import numpy as np
import time
import random

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

from pyjuice.layer import InputLayer, ProdLayer, SumLayer

import pytest


def sum_layer_test():

    device = torch.device("cuda:0")

    group_size = 16
    batch_size = 16
    
    with juice.set_group_size(group_size):

        ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)
        np2 = multiply(ni1, ni2)

        ns0 = summate(np0, num_node_groups = 2)
        ns1 = summate(np1, num_node_groups = 2)
        ns2 = summate(np2, num_node_groups = 2)

    input_layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = group_size)

    prod_layer = ProdLayer([np0, np1, np2])

    layer = SumLayer([ns0, ns1, ns2], global_nid_start = group_size,
                     param_ends = [1], tied_param_ids = [],
                     tied_param_group_ids = [], tied_param_ends = [],
                     ch_prod_layer_size = prod_layer.num_nodes + group_size)

    assert torch.all(layer.partitioned_nids[0] == torch.arange(group_size, 7 * group_size, group_size))
    assert torch.all(layer.partitioned_cids[0][0:2,0] == group_size)
    assert torch.all(layer.partitioned_cids[0][2:4,0] == 3 * group_size)
    assert torch.all(layer.partitioned_cids[0][4:6,0] == 5 * group_size)
    assert torch.all(layer.partitioned_cids[0][0:2,1] == group_size + 1)
    assert torch.all(layer.partitioned_cids[0][2:4,1] == 3 * group_size + 1)
    assert torch.all(layer.partitioned_cids[0][4:6,1] == 5 * group_size + 1)
    assert torch.all(layer.partitioned_pids[0][:,0] == torch.arange(group_size, (group_size * 2 * 6 + 1) * group_size, 2 * group_size * group_size) - group_size + 1)
    assert torch.all(layer.partitioned_pids[0][:,1] == torch.arange(group_size, (group_size * 2 * 6 + 1) * group_size, 2 * group_size * group_size) + 1)

    layer.to(device)

    ## Forward tests ##

    element_mars = torch.rand([group_size + 3 * 2 * 2 * group_size, batch_size]).log().to(device)
    element_mars[:group_size,:] = -float("inf")
    node_mars = torch.zeros([group_size + group_size * 2 * 3, batch_size]).to(device)

    params = torch.rand([1 + 3 * 4 * group_size * group_size]).to(device)

    layer(node_mars, element_mars, params)

    for i in range(group_size):
        for j in range(6):
            cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
            epars = params[layer.partitioned_pids[0][j,:]+i]
            assert torch.all(torch.abs(node_mars[(j+1)*group_size+i,:] - (epars[:,None] * cmars).sum(dim = 0).log()) < 1e-3)


def speed_test():

    device = torch.device("cuda:0")

    group_size = 32
    num_vars = 28*28
    num_node_groups = 256 // group_size
    num_prod_nodes = 200

    batch_size = 512

    with juice.set_group_size(group_size):

        nis = []
        for v in range(num_vars):
            nis.append(inputs(v, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 64)))

        nps = []
        for i in range(num_prod_nodes):
            v1 = random.randint(0, num_vars - 1)
            v2 = random.randint(0, num_vars - 1)
            if v1 == v2:
                if v1 == num_vars - 1:
                    v1 -= 2
                v2 = v1 + 1

            nps.append(multiply(nis[v1], nis[v2]))

        nodes = [summate(np, num_node_groups = num_node_groups) for np in nps]

    input_layer = InputLayer(nis, cum_nodes = group_size)

    prod_layer = ProdLayer(nps, layer_sparsity_tol = 0.1)

    layer = SumLayer(nodes, global_nid_start = group_size,
                         param_ends = [1], tied_param_ids = [],
                         tied_param_group_ids = [], tied_param_ends = [],
                         ch_prod_layer_size = prod_layer.num_nodes + group_size)

    # import pdb; pdb.set_trace()

    layer.to(device)

    node_mars = torch.zeros([group_size + group_size * num_node_groups * num_prod_nodes, batch_size]).to(device)
    element_mars = torch.rand([group_size + num_prod_nodes * group_size * num_node_groups, batch_size]).log().to(device)
    params = torch.rand([layer.partitioned_pids[0].max() + group_size]).to(device)

    ## Forward tests ##

    layer(node_mars, element_mars, params)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(node_mars, element_mars, params)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Forward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 11.255ms.")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    # sum_layer_test()
    speed_test()