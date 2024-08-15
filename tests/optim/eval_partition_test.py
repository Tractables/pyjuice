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
from pyjuice.model.backend import eval_partition_fn

import pytest


def test_simple_model_partition():

    device = torch.device("cuda:0")

    block_size = 16
    
    with juice.set_block_size(block_size):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 4))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 4))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 6))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 6))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)
        np2 = multiply(ni1, ni2)
        np3 = multiply(ni0, ni1)

        ns0 = summate(np0, np3, num_node_blocks = 2)
        ns1 = summate(np1, num_node_blocks = 2)
        ns2 = summate(np2, num_node_blocks = 2)

        np4 = multiply(ns0, ni2, ni3)
        np5 = multiply(ns1, ni0, ni1)
        np6 = multiply(ns2, ni0, ni3)

        ns = summate(np4, np5, np6, num_node_blocks = 1, block_size = 1)

    ns.init_parameters()

    pc = TensorCircuit(ns, layer_sparsity_tol = 0.1)
    pc.to(device)

    pc.init_param_flows()
    pc._init_buffer(name = "node_mars", shape = (pc.num_nodes, 1), set_value = 0.0)
    pc._init_buffer(name = "element_mars", shape = (pc.num_elements, 1), set_value = 0.0)

    partition_fn = eval_partition_fn(pc)

    assert torch.all(torch.abs(partition_fn) < 1e-5)


if __name__ == "__main__":
    test_simple_model_partition()
