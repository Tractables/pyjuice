import pyjuice as juice
import torch
import numpy as np
import time
import random

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

from pyjuice.layer import InputLayer, ProdLayer

import pytest


def prod_layer_test():

    device = torch.device("cuda:0")

    for (group_size, batch_size) in [(1, 16), (8, 512)]:
    
        with juice.set_group_size(group_size):

            ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
            ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
            ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
            ni3 = inputs(3, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))

            np0 = multiply(ni0, ni1)
            np1 = multiply(ni2, ni3)
            np2 = multiply(ni1, ni2)

        input_layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = group_size)

        layer = ProdLayer([np0, np1, np2])

        assert torch.all(layer.partitioned_nids[0] == torch.arange(group_size, 7*group_size, group_size))
        assert layer.partitioned_cids[0][0,0] == group_size
        assert layer.partitioned_cids[0][0,1] == 3 * group_size
        assert layer.partitioned_cids[0][1,0] == 2 * group_size
        assert layer.partitioned_cids[0][1,1] == 4 * group_size
        assert layer.partitioned_cids[0][2,0] == 5 * group_size
        assert layer.partitioned_cids[0][2,1] == 7 * group_size
        assert layer.partitioned_cids[0][3,0] == 6 * group_size
        assert layer.partitioned_cids[0][3,1] == 8 * group_size
        assert layer.partitioned_cids[0][4,0] == 3 * group_size
        assert layer.partitioned_cids[0][4,1] == 5 * group_size
        assert layer.partitioned_cids[0][5,0] == 4 * group_size
        assert layer.partitioned_cids[0][5,1] == 6 * group_size

        layer.to(device)

        node_mars = torch.rand([group_size + group_size * 2 * 4, batch_size]).log().to(device)
        element_mars = torch.zeros([group_size + 3 * 2 * 2 * group_size, batch_size]).to(device)

        ## Forward tests ##

        layer(node_mars, element_mars)

        for i in range(group_size):
            assert torch.all(torch.abs(element_mars[group_size+i,:] - (node_mars[group_size+i,:] + node_mars[3*group_size+i,:])) < 1e-4)
            assert torch.all(torch.abs(element_mars[2*group_size+i,:] - (node_mars[2*group_size+i,:] + node_mars[4*group_size+i,:])) < 1e-4)

            assert torch.all(torch.abs(element_mars[3*group_size+i,:] - (node_mars[5*group_size+i,:] + node_mars[7*group_size+i,:])) < 1e-4)
            assert torch.all(torch.abs(element_mars[4*group_size+i,:] - (node_mars[6*group_size+i,:] + node_mars[8*group_size+i,:])) < 1e-4)

            assert torch.all(torch.abs(element_mars[5*group_size+i,:] - (node_mars[3*group_size+i,:] + node_mars[5*group_size+i,:])) < 1e-4)
            assert torch.all(torch.abs(element_mars[6*group_size+i,:] - (node_mars[4*group_size+i,:] + node_mars[6*group_size+i,:])) < 1e-4)

        ## Backward tests ##

        element_flows = torch.rand([group_size + 3 * 2 * 2 * group_size, batch_size]).to(device)
        element_flows[:group_size,:] = 0.0
        node_flows = torch.zeros([group_size + group_size * 2 * 4, batch_size]).to(device)

        layer(node_mars, element_mars)
        layer.backward(node_flows, element_flows)

        for i in range(group_size):
            assert torch.all(torch.abs(node_flows[group_size+i,:] - element_flows[group_size+i,:]) < 1e-4)
            assert torch.all(torch.abs(node_flows[2*group_size+i,:] - element_flows[2*group_size+i,:]) < 1e-4)

            assert torch.all(torch.abs(node_flows[3*group_size+i,:] - (element_flows[group_size+i,:] + element_flows[5*group_size+i,:])) < 1e-4)
            assert torch.all(torch.abs(node_flows[4*group_size+i,:] - (element_flows[2*group_size+i,:] + element_flows[6*group_size+i,:])) < 1e-4)

            assert torch.all(torch.abs(node_flows[5*group_size+i,:] - (element_flows[3*group_size+i,:] + element_flows[5*group_size+i,:])) < 1e-4)
            assert torch.all(torch.abs(node_flows[6*group_size+i,:] - (element_flows[4*group_size+i,:] + element_flows[6*group_size+i,:])) < 1e-4)

            assert torch.all(torch.abs(node_flows[7*group_size+i,:] - element_flows[3*group_size+i,:]) < 1e-4)
            assert torch.all(torch.abs(node_flows[8*group_size+i,:] - element_flows[4*group_size+i,:]) < 1e-4)


def speed_test():

    device = torch.device("cuda:0")

    group_size = 16
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

    input_layer = InputLayer(nis, cum_nodes = group_size)

    layer = ProdLayer(nps, layer_sparsity_tol = 0.1)

    layer.to(device)

    node_mars = torch.rand([group_size + group_size * num_node_groups * num_vars, batch_size]).log().to(device)
    element_mars = torch.zeros([group_size + num_prod_nodes * group_size * num_node_groups, batch_size]).to(device)

    ## Forward tests ##

    layer(node_mars, element_mars)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(node_mars, element_mars)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Forward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 0.330ms.")
    print("--------------------------------------------------------------")

    element_flows = torch.rand([group_size + num_prod_nodes * num_node_groups * group_size, batch_size]).to(device)
    element_flows[:group_size,:] = 0.0
    node_flows = torch.zeros([group_size + group_size * num_node_groups * num_vars, batch_size]).to(device)

    layer(node_mars, element_mars)
    layer.backward(node_flows, element_flows)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer.backward(node_flows, element_flows)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Backward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 0.533ms.")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    torch.manual_seed(2390)
    prod_layer_test()
    speed_test()
