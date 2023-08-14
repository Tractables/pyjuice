import pyjuice as juice
import torch
import numpy as np
import torchvision
import time

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit
from pyjuice.nodes.methods import get_subsumed_scopes

import pytest


def cond_test():

    device = torch.device("cuda:0")

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 4))

    m = multiply(ni0, ni1)
    n = summate(m, num_nodes = 1)

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.rand([3, 2, 4]).to(device) # data shape: (B, num_vars, num_cats)
    data[:,0,2:] = 0.0
    missing_mask = torch.tensor([[False, True], [False, True], [False, True]]).to(device) # True for variables to be conditioned on

    outputs = juice.queries.conditional(
        pc, missing_mask = missing_mask, soft_evidence = data
    ) # output shape: (B, num_vars, num_cats)

    p0 = data[0,0,0] * pc.input_layers[0].params[0] + data[0,0,1] * pc.input_layers[0].params[1]
    p1 = data[0,0,0] * pc.input_layers[0].params[2] + data[0,0,1] * pc.input_layers[0].params[3]

    assert (pc.node_mars[1,0].exp() - p0).abs() < 1e-4
    assert (pc.node_mars[2,0].exp() - p1).abs() < 1e-4

    f1 = pc.node_flows[3,0]
    f2 = pc.node_flows[4,0]
    unnorm_o = f1 * pc.input_layers[0].params[4:8] + f2 * pc.input_layers[0].params[8:12]
    target_o = unnorm_o / unnorm_o.sum()

    assert torch.all((target_o - outputs[0,1,:]) < 1e-4)


def partial_eval_cond_test():

    ni0 = inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni2 = inputs(2, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
    ni3 = inputs(3, num_nodes = 2, dist = dists.Categorical(num_cats = 2))

    m1 = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    n1 = summate(m1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    m2 = multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n2 = summate(m2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    m = multiply(n1, n2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    n = summate(m, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long))

    pc = TensorCircuit(n)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, 2, [16, 4]).to(device)
    missing_mask = torch.zeros([16, 4], dtype = torch.bool).to(device)
    missing_mask[:,1] = True

    cond_ps = juice.queries.conditional(pc, target_vars = [1], tokens = data, missing_mask = missing_mask)

    cond_ps2 = juice.queries.conditional(pc, tokens = data, missing_mask = missing_mask)

    assert torch.all(torch.abs(cond_ps[:,0,:] - cond_ps2[:,1,:]) < 1e-6)

    cloned_data = data.clone()
    cloned_data[:,2] = 1

    cond_ps3, cache = juice.queries.conditional(pc, target_vars = [1], tokens = cloned_data, missing_mask = missing_mask, cache = dict())

    cond_ps4, cache = juice.queries.conditional(pc, target_vars = [1], tokens = data, missing_mask = missing_mask, cache = cache, fw_delta_vars = [2])

    assert torch.all(torch.abs(cond_ps[:,0,:] - cond_ps4[:,0,:]) < 1e-6)

    # Soft evidence

    soft_data = torch.rand([16, 4, 2]).to(device)
    soft_data /= soft_data.sum(dim = 2, keepdim = True)

    cache["node_mars"][:,:] = 0.0

    cond_ps5, cache = juice.queries.conditional(pc, target_vars = [1], soft_evidence = soft_data, missing_mask = missing_mask, cache = cache, fw_delta_vars = [2])

    assert torch.all(cache["node_mars"][:5,:] < 1e-4)
    assert torch.all(cache["node_mars"][7:11,:] < 1e-4)


def cond_prob_speed_test():

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    train_data = train_dataset.data.reshape(60000, 28*28)

    ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = 64, 
        chunk_size = 32
    )
    pc = juice.TensorCircuit(ns)

    pc.to(device)

    data = train_data[:1024,:].long().to(device)
    missing_mask = torch.zeros([1024, 28*28], dtype = torch.bool, device = device)

    probs = juice.queries.conditional(
        pc, tokens = data, missing_mask = missing_mask
    )

    t0 = time.time()

    probs = juice.queries.conditional(
        pc, tokens = data, missing_mask = missing_mask
    )

    torch.cuda.synchronize()
    t1 = time.time()

    probs, cache = juice.queries.conditional(
        pc, target_vars = [10], tokens = data, missing_mask = missing_mask, cache = dict()
    )

    torch.cuda.synchronize()
    t2 = time.time()

    probs, cache = juice.queries.conditional(
        pc, target_vars = [10], tokens = data, missing_mask = missing_mask, cache = dict()
    )

    torch.cuda.synchronize()
    t3 = time.time()

    probs, cache = juice.queries.conditional(
        pc, target_vars = [10], tokens = data, missing_mask = missing_mask, 
        fw_delta_vars = [5], cache = cache
    )

    fw_scopes = get_subsumed_scopes(pc, [5], type = "any")
    bk_scopes = get_subsumed_scopes(pc, [10], type = "any")

    torch.cuda.synchronize()
    t4 = time.time()

    probs, cache = juice.queries.conditional(
        pc, target_vars = [10], tokens = data, missing_mask = missing_mask, 
        fw_delta_vars = [5], cache = cache, fw_scopes = fw_scopes, bk_scopes = bk_scopes
    )

    torch.cuda.synchronize()
    t5 = time.time()

    probs, cache = juice.queries.conditional(
        pc, target_vars = [10], tokens = data, missing_mask = missing_mask, 
        fw_delta_vars = [5], cache = cache, fw_scopes = fw_scopes, bk_scopes = bk_scopes,
        overwrite_partial_eval = False
    )

    torch.cuda.synchronize()
    t6 = time.time()

    print(t1 - t0, t3 - t2, t5 - t4, t6 - t5)

    assert (t6 - t5) * 5 < (t1 - t0)


if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    cond_test()
    partial_eval_cond_test()
    cond_prob_speed_test()