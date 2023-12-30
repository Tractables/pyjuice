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


def sparse_prod_layer_test():

    device = torch.device("cuda:0")

    group_size = 16
    batch_size = 16

    with juice.set_group_size(group_size):

        ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_groups = 2, dist = dists.Categorical(num_cats = 2))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3, edge_ids = torch.arange(0, group_size * 2)[:,None].repeat(1, 2), sparse_edges = True)
        np2 = multiply(ni1, ni2)

    input_layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = group_size)

    layer = ProdLayer([np0, np1, np2])

    assert torch.all(layer.partitioned_nids[0] == torch.arange(1, 1+6*group_size))
    assert torch.all(layer.partitioned_cids[0][0,:] == torch.tensor([1 * group_size, 3 * group_size]))
    assert torch.all(layer.partitioned_cids[0][1:16,:] - layer.partitioned_cids[0][0:15,:] == 1)
    assert torch.all(layer.partitioned_cids[0][16,:] == torch.tensor([2 * group_size, 4 * group_size]))
    assert torch.all(layer.partitioned_cids[0][17:32,:] - layer.partitioned_cids[0][16:31,:] == 1)
    assert torch.all(layer.partitioned_cids[0][32,:] == torch.tensor([5 * group_size, 7 * group_size]))
    assert torch.all(layer.partitioned_cids[0][33:48,:] - layer.partitioned_cids[0][32:47,:] == 1)
    assert torch.all(layer.partitioned_cids[0][48,:] == torch.tensor([6 * group_size, 8 * group_size]))
    assert torch.all(layer.partitioned_cids[0][49:64,:] - layer.partitioned_cids[0][48:63,:] == 1)
    assert torch.all(layer.partitioned_cids[0][64,:] == torch.tensor([3 * group_size, 5 * group_size]))
    assert torch.all(layer.partitioned_cids[0][65:80,:] - layer.partitioned_cids[0][64:79,:] == 1)
    assert torch.all(layer.partitioned_cids[0][80,:] == torch.tensor([4 * group_size, 6 * group_size]))
    assert torch.all(layer.partitioned_cids[0][81:96,:] - layer.partitioned_cids[0][80:95,:] == 1)

    assert torch.all(layer.partitioned_u_cids[0] == torch.arange(16, 144))
    assert torch.all(layer.partitioned_parids[0][0:32,0] == torch.arange(1, 32+1))
    assert torch.all(layer.partitioned_parids[0][0:32,1] == 0)
    assert torch.all(layer.partitioned_parids[0][32:64,0] == torch.arange(1, 32+1))
    assert torch.all(layer.partitioned_parids[0][32:64,1] == torch.arange(64+1, 96+1))
    assert torch.all(layer.partitioned_parids[0][64:96,0] == torch.arange(32+1, 64+1))
    assert torch.all(layer.partitioned_parids[0][64:96,1] == torch.arange(64+1, 96+1))
    assert torch.all(layer.partitioned_parids[0][96:128,0] == torch.arange(32+1, 64+1))
    assert torch.all(layer.partitioned_parids[0][96:128,1] == 0)

    layer.to(device)

    node_mars = torch.rand([group_size + group_size * 2 * 4, batch_size]).log().to(device)
    element_mars = torch.zeros([1 + 3 * 2 * 2 * group_size, batch_size]).to(device)

    ## Forward tests ##

    layer(node_mars, element_mars)

    for i in range(group_size):
        assert torch.all(torch.abs(element_mars[1+i,:] - (node_mars[group_size+i,:] + node_mars[3*group_size+i,:])) < 1e-4)
        assert torch.all(torch.abs(element_mars[1+1*group_size+i,:] - (node_mars[2*group_size+i,:] + node_mars[4*group_size+i,:])) < 1e-4)

        assert torch.all(torch.abs(element_mars[1+2*group_size+i,:] - (node_mars[5*group_size+i,:] + node_mars[7*group_size+i,:])) < 1e-4)
        assert torch.all(torch.abs(element_mars[1+3*group_size+i,:] - (node_mars[6*group_size+i,:] + node_mars[8*group_size+i,:])) < 1e-4)

        assert torch.all(torch.abs(element_mars[1+4*group_size+i,:] - (node_mars[3*group_size+i,:] + node_mars[5*group_size+i,:])) < 1e-4)
        assert torch.all(torch.abs(element_mars[1+5*group_size+i,:] - (node_mars[4*group_size+i,:] + node_mars[6*group_size+i,:])) < 1e-4)

    ## Backward tests ##

    element_flows = torch.rand([1 + 3 * 2 * 2 * group_size, batch_size]).to(device)
    element_flows[0,:] = 0.0
    node_flows = torch.zeros([group_size + group_size * 2 * 4, batch_size]).to(device)

    layer(node_mars, element_mars)
    layer.backward(node_flows, element_flows)

    for i in range(group_size):
        assert torch.all(torch.abs(node_flows[group_size+i,:] - element_flows[1+i,:]) < 1e-4)
        assert torch.all(torch.abs(node_flows[2*group_size+i,:] - element_flows[1+1*group_size+i,:]) < 1e-4)

        assert torch.all(torch.abs(node_flows[3*group_size+i,:] - (element_flows[1+i,:] + element_flows[1+4*group_size+i,:])) < 1e-4)
        assert torch.all(torch.abs(node_flows[4*group_size+i,:] - (element_flows[1+1*group_size+i,:] + element_flows[1+5*group_size+i,:])) < 1e-4)

        assert torch.all(torch.abs(node_flows[5*group_size+i,:] - (element_flows[1+2*group_size+i,:] + element_flows[1+4*group_size+i,:])) < 1e-4)
        assert torch.all(torch.abs(node_flows[6*group_size+i,:] - (element_flows[1+3*group_size+i,:] + element_flows[1+5*group_size+i,:])) < 1e-4)

        assert torch.all(torch.abs(node_flows[7*group_size+i,:] - element_flows[1+2*group_size+i,:]) < 1e-4)
        assert torch.all(torch.abs(node_flows[8*group_size+i,:] - element_flows[1+3*group_size+i,:]) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(2390)
    sparse_prod_layer_test()
