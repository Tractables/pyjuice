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


@pytest.mark.parametrize("keep_zero_params", [False, True])
@pytest.mark.parametrize("num_cats", [200, 300])       # the kernel for at most 256 categories, and the one past it
def test_categorical_em_update(num_cats, keep_zero_params):
    """One mini-batch EM step of Categorical input nodes, some of whose parameters are exactly zero, against its
    definition: ``(1 - s) p + s (f + c / V) / Z`` for flows ``f``, step size ``s`` and pseudocount ``c``, with ``Z``
    the node's flows plus ``c``. Under ``keep_zero_params`` a zero parameter stays zero and ``Z`` sums the flows and
    pseudocounts of the others alone; every node's parameters sum to 1 either way."""

    torch.manual_seed(0)
    nodes = 16

    ns = [inputs(v, num_node_blocks = 1, block_size = nodes, dist = dists.Categorical(num_cats = num_cats))
          for v in range(4)]
    n1 = summate(multiply(ns[0], ns[1]), num_node_blocks = 1, block_size = nodes)
    n2 = summate(multiply(ns[2], ns[3]), num_node_blocks = 1, block_size = nodes)
    n = summate(multiply(n1, n2), num_node_blocks = 1, block_size = 1)

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    # Every node zeroes a few categories of its own (no category is zero under every node of a variable, so every
    # sample has a finite likelihood), renormalized
    layer = pc.input_layer_group[0]
    params = layer.params.view(-1, num_cats)
    zero = torch.zeros_like(params, dtype = torch.bool)
    for i in range(params.size(0)):
        zero[i, (torch.arange(3, device = device) * 37 + i) % num_cats] = True
    with torch.no_grad():
        params[zero] = 0.0
        params /= params.sum(dim = 1, keepdim = True)

    data = torch.randint(0, num_cats, [64, 4]).to(device)

    lls = pc(data)
    assert torch.isfinite(lls).all()

    pc.backward(data)

    ## EM tests ##

    step_size = 0.5
    pseudocount = 0.1

    p = params.clone().double()
    f = layer.param_flows.view(-1, num_cats).clone().double()
    if keep_zero_params:
        g = torch.where(zero, 0.0, f + pseudocount / num_cats)
        new_params = torch.where(zero, 0.0, (1.0 - step_size) * p + step_size * g / g.sum(dim = 1, keepdim = True))
    else:
        new_params = (1.0 - step_size) * p + step_size * (f + pseudocount / num_cats) / \
            (f.sum(dim = 1, keepdim = True) + pseudocount)

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount, keep_zero_params = keep_zero_params)

    updated_params = layer.params.view(-1, num_cats).double()

    assert torch.allclose(updated_params.sum(dim = 1), torch.ones_like(updated_params[:, 0]), atol = 1e-5)
    if keep_zero_params:
        assert torch.all(updated_params[zero] == 0.0)
    assert torch.allclose(updated_params, new_params, rtol = 1e-4, atol = 1e-6)


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
