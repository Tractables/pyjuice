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


def general_ll_prop_test():
    
    device = torch.device("cuda:0")

    batch_size = 16

    for block_size in [1, 4, 8, 16]:
    
        with juice.set_block_size(block_size):

            ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
            ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
            ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
            ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

            np0 = multiply(ni0, ni1)
            np1 = multiply(ni2, ni3)
            np2 = multiply(ni1, ni2)

            ns0 = summate(np0, num_node_blocks = 2)
            ns1 = summate(np1, num_node_blocks = 2)
            ns2 = summate(np2, num_node_blocks = 2)

        input_layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = block_size)

        prod_layer = ProdLayer([np0, np1, np2])

        layer = SumLayer([ns0, ns1, ns2], global_nid_start = block_size,
                        global_pid_start = block_size ** 2, 
                        global_pfid_start = 0, node2tiednodes = dict())

        layer.to(device)

        ## Forward pass ##

        element_mars = torch.rand([block_size + 3 * 2 * 2 * block_size, batch_size]).log().to(device)
        element_mars[:block_size,:] = -float("inf")
        node_mars = torch.zeros([block_size + block_size * 2 * 3, batch_size]).to(device)

        params = torch.rand([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

        alphas = [1.2, 2.0, 3.0]

        for alpha in alphas:
            layer(node_mars, element_mars, params, propagation_alg = "GeneralLL", alpha = alpha)

            for i in range(block_size):
                for j in range(6):
                    cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                    epars = params[layer.partitioned_pids[0][j,:]+i]
                    scaled_lls = (epars[:,None]**alpha * cmars**alpha).sum(dim = 0).log() * (1.0 / alpha)
                    assert torch.all(torch.abs(node_mars[(j+1)*block_size+i,:] - scaled_lls) < 1e-3)


def mpe_prop_test():

    device = torch.device("cuda:0")

    batch_size = 16

    for block_size in [1, 4, 8, 16]:
    
        with juice.set_block_size(block_size):

            ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
            ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
            ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
            ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

            np0 = multiply(ni0, ni1)
            np1 = multiply(ni2, ni3)
            np2 = multiply(ni1, ni2)

            ns0 = summate(np0, num_node_blocks = 2)
            ns1 = summate(np1, num_node_blocks = 2)
            ns2 = summate(np2, num_node_blocks = 2)

        input_layer = InputLayer([ni0, ni1, ni2, ni3], cum_nodes = block_size)

        prod_layer = ProdLayer([np0, np1, np2])

        layer = SumLayer([ns0, ns1, ns2], global_nid_start = block_size,
                        global_pid_start = block_size ** 2, 
                        global_pfid_start = 0, node2tiednodes = dict())

        layer.to(device)

        ## Forward pass ##

        element_mars = torch.rand([block_size + 3 * 2 * 2 * block_size, batch_size]).log().to(device)
        element_mars[:block_size,:] = -float("inf")
        node_mars = torch.zeros([block_size + block_size * 2 * 3, batch_size]).to(device)

        params = torch.rand([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

        layer(node_mars, element_mars, params, propagation_alg = "MPE")

        for i in range(block_size):
            for j in range(6):
                cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                epars = params[layer.partitioned_pids[0][j,:]+i]
                scaled_lls = (epars[:,None] * cmars).max(dim = 0).values.log()
                assert torch.all(torch.abs(node_mars[(j+1)*block_size+i,:] - scaled_lls) < 1e-3)


if __name__ == "__main__":
    general_ll_prop_test()
    mpe_prop_test()
