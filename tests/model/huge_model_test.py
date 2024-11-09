import pyjuice as juice
import torch
import numpy as np
import time
import random

import pyjuice.nodes.distributions as dists

import pytest


def test_huge_model():

    device = torch.device("cuda:0")
    
    for n_blocks in [100000, 200000, 500000]:
        for block_size in [4, 8, 16]:
    
            ns = juice.inputs(var = 0, num_node_blocks = n_blocks, block_size = 8, dist = dists.Categorical(num_cats = 2))
            ms = juice.multiply(ns)
            ns = juice.summate(ms, num_node_blocks = n_blocks, block_size = block_size, 
                            edge_ids = torch.arange(0, n_blocks)[None,:].repeat(2, 1))

            pc = juice.compile(ns)
            pc.to(device)

            x = torch.zeros((16, 1), dtype = torch.long).to(device)
            
            lls = pc(x, propagation_alg = "LL")
            pc.backward(x, flows_memory = 1.0, allow_modify_flows = False,
                        propagation_alg = "LL", logspace_flows = True)

            assert (lls < 0.0).all()


if __name__ == "__main__":
    test_huge_model()
