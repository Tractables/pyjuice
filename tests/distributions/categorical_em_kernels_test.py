import pyjuice as juice
import torch
import numpy as np
import math
import time

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_categorical_em():

    num_cats = 50257

    ni0 = inputs(0, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))
    ni2 = inputs(2, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))
    ni3 = inputs(3, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))

    m1 = multiply(ni0, ni1)
    n1 = summate(m1, num_node_blocks = 1, block_size = 1024)

    m2 = multiply(ni2, ni3)
    n2 = summate(m2, num_node_blocks = 1, block_size = 1024)

    m = multiply(n1, n2)
    n = summate(m, num_node_blocks = 1, block_size = 1)

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    ## EM tests ##

    original_params = pc.input_layer_group[0].params.clone().reshape(4, 1024, num_cats)

    step_size = 0.3
    pseudocount = 0.1

    par_flows = pc.input_layer_group[0].param_flows.clone().reshape(4, 1024, num_cats)
    new_params = (1.0 - step_size) * original_params + step_size * (par_flows + pseudocount / num_cats) / (par_flows.sum(dim = 2, keepdim = True) + pseudocount)

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    updated_params = pc.input_layer_group[0].params.reshape(4, 1024, num_cats)

    assert torch.all(torch.abs(new_params - updated_params) < 1e-4)


def test_categorical_em_speed():

    num_cats = 50257

    ni0 = inputs(0, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))
    ni2 = inputs(2, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))
    ni3 = inputs(3, num_node_blocks = 1, block_size = 1024, dist = dists.Categorical(num_cats = num_cats))

    m1 = multiply(ni0, ni1)
    n1 = summate(m1, num_node_blocks = 1, block_size = 1024)

    m2 = multiply(ni2, ni3)
    n2 = summate(m2, num_node_blocks = 1, block_size = 1024)

    m = multiply(n1, n2)
    n = summate(m, num_node_blocks = 1, block_size = 1)

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 4]).to(device)

    lls = pc(data)

    pc.backward(data)

    ## EM tests ##

    original_params = pc.input_layer_group[0].params.clone().reshape(4, 1024, num_cats)

    step_size = 0.3
    pseudocount = 0.1

    for _ in range(3):
        pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    print("==============================================================")
    print(f"- num_nodes=4x1024, num_cats={num_cats}")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)
    torch.cuda.synchronize()
    t1 = time.time()
    aveg_fw_ms = (t1 - t0) * 1000 / 100

    print(f"EM on average takes {aveg_fw_ms:.3f}ms.")
    print(f"Reference computation time on RTX 4090: 4.937ms.")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    test_categorical_em()
    test_categorical_em_speed()
