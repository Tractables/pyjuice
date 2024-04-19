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


def test_sum_layer():

    device = torch.device("cuda:0")

    block_size = 16
    batch_size = 16
    
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

    assert torch.all(layer.partitioned_nids[0] == torch.arange(block_size, 7 * block_size, block_size))
    assert torch.all(layer.partitioned_cids[0][0:2,0] == block_size)
    assert torch.all(layer.partitioned_cids[0][2:4,0] == 3 * block_size)
    assert torch.all(layer.partitioned_cids[0][4:6,0] == 5 * block_size)
    assert torch.all(layer.partitioned_cids[0][0:2,1] == block_size + 1)
    assert torch.all(layer.partitioned_cids[0][2:4,1] == 3 * block_size + 1)
    assert torch.all(layer.partitioned_cids[0][4:6,1] == 5 * block_size + 1)
    assert torch.all(layer.partitioned_pids[0][:,0] == torch.arange(block_size, (block_size * 2 * 6 + 1) * block_size, 2 * block_size * block_size) - block_size + block_size ** 2)
    assert torch.all(layer.partitioned_pids[0][:,1] == torch.arange(block_size, (block_size * 2 * 6 + 1) * block_size, 2 * block_size * block_size) + block_size ** 2)
    assert torch.all(layer.partitioned_pfids[0][:,0] == torch.arange(block_size, (block_size * 2 * 6 + 1) * block_size, 2 * block_size * block_size) - block_size)
    assert torch.all(layer.partitioned_pfids[0][:,1] == torch.arange(block_size, (block_size * 2 * 6 + 1) * block_size, 2 * block_size * block_size))

    assert torch.all(layer.partitioned_chids[0] == torch.arange(block_size, 7 * block_size, block_size))
    assert torch.all(layer.partitioned_parids[0][0:2,0] == block_size)
    assert torch.all(layer.partitioned_parids[0][0:2,1] == 2 * block_size)
    assert torch.all(layer.partitioned_parids[0][2:4,0] == 3 * block_size)
    assert torch.all(layer.partitioned_parids[0][2:4,1] == 4 * block_size)
    assert torch.all(layer.partitioned_parids[0][4:6,0] == 5 * block_size)
    assert torch.all(layer.partitioned_parids[0][4:6,1] == 6 * block_size)
    assert torch.all(layer.partitioned_parpids[0][0,0] == block_size**2)
    assert torch.all(layer.partitioned_parpids[0][1,0] == 2 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][0,1] == 3 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][1,1] == 4 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][2,0] == 5 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][3,0] == 6 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][2,1] == 7 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][3,1] == 8 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][4,0] == 9 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][5,0] == 10 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][4,1] == 11 * block_size**2)
    assert torch.all(layer.partitioned_parpids[0][5,1] == 12 * block_size**2)

    layer.to(device)

    ## Forward tests ##

    element_mars = torch.rand([block_size + 3 * 2 * 2 * block_size, batch_size]).log().to(device)
    element_mars[:block_size,:] = -float("inf")
    node_mars = torch.zeros([block_size + block_size * 2 * 3, batch_size]).to(device)

    params = torch.rand([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

    layer(node_mars, element_mars, params)

    for i in range(block_size):
        for j in range(6):
            cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
            epars = params[layer.partitioned_pids[0][j,:]+i]
            assert torch.all(torch.abs(node_mars[(j+1)*block_size+i,:] - (epars[:,None] * cmars).sum(dim = 0).log()) < 1e-3)

    ## Backward tests ##

    node_flows = torch.rand([block_size + block_size * 2 * 3, batch_size]).to(device)
    element_flows = torch.zeros([block_size + 3 * 2 * 2 * block_size, batch_size]).to(device)

    param_flows = torch.zeros([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

    layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows)

    chids = layer.partitioned_chids[0]
    parids = layer.partitioned_parids[0]
    parpids = layer.partitioned_parpids[0]

    num_nblocks = chids.size(0)
    num_eblocks = parids.size(1)
    parids = (parids[:,:,None].repeat(1, 1, block_size) + torch.arange(0, block_size, device = parids.device)).reshape(num_nblocks, num_eblocks * block_size)
    parpids_start = (parpids[:,:,None] + torch.arange(0, block_size, device = parids.device)).reshape(
        num_nblocks, num_eblocks * block_size)

    for j in range(6):
        parpids = parpids_start.clone()
        for i in range(block_size):
            nmars = node_mars[parids[j,:]].exp()
            nflows = node_flows[parids[j,:]]
            emars = element_mars[(j+1)*block_size+i,:].exp()
            epars = params[parpids[j,:]]
            eflows = (nflows * epars[:,None] * emars[None,:] / nmars).sum(dim = 0)

            assert torch.all(torch.abs(eflows - element_flows[(j+1)*block_size+i,:]) < 1e-2)

            parpids += block_size

    my_pflows = torch.zeros_like(param_flows)

    for i in range(block_size):
        for j in range(6):
            emars = element_mars[layer.partitioned_cids[0][j,:]].exp()
            epars = params[layer.partitioned_pids[0][j,:]+i]
            nmars = node_mars[(j+1)*block_size+i,:].exp()
            nflows = node_flows[(j+1)*block_size+i,:]
            pflows = epars * (nflows[None,:] * emars / nmars[None,:]).sum(dim = 1)

            my_pflows[layer.partitioned_pfids[0][j,:]+i] = pflows

    assert torch.all(torch.abs(my_pflows - param_flows) < 2e-3)


def test_corner_case():

    device = torch.device("cuda:0")

    block_sizes = [2, 4, 4,  4, 8,  8, 16, 32, 32, 32]
    batch_sizes = [4, 4, 8, 16, 8, 16,  8,  8, 16, 32]
    
    for block_size, batch_size in zip(block_sizes, batch_sizes):
        for force_use_fp16, force_use_fp32 in ((False, False), (True, False), (False, True)):
    
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

            ## Compilation tests ##

            assert torch.all(layer.partitioned_nids[0] == torch.arange(block_size, block_size * 7, block_size))

            assert torch.all(layer.partitioned_cids[0][0,:] == torch.arange(block_size, block_size * 3))
            assert torch.all(layer.partitioned_cids[0][1,:] == torch.arange(block_size, block_size * 3))
            assert torch.all(layer.partitioned_cids[0][2,:] == torch.arange(block_size * 3, block_size * 5))
            assert torch.all(layer.partitioned_cids[0][3,:] == torch.arange(block_size * 3, block_size * 5))
            assert torch.all(layer.partitioned_cids[0][4,:] == torch.arange(block_size * 5, block_size * 7))
            assert torch.all(layer.partitioned_cids[0][5,:] == torch.arange(block_size * 5, block_size * 7))

            assert torch.all(layer.partitioned_pids[0][0,:] == torch.arange(block_size**2, block_size**2 * 3, block_size))
            assert torch.all(layer.partitioned_pids[0][1,:] == torch.arange(block_size**2 * 3, block_size**2 * 5, block_size))
            assert torch.all(layer.partitioned_pids[0][2,:] == torch.arange(block_size**2 * 5, block_size**2 * 7, block_size))
            assert torch.all(layer.partitioned_pids[0][3,:] == torch.arange(block_size**2 * 7, block_size**2 * 9, block_size))
            assert torch.all(layer.partitioned_pids[0][4,:] == torch.arange(block_size**2 * 9, block_size**2 * 11, block_size))
            assert torch.all(layer.partitioned_pids[0][5,:] == torch.arange(block_size**2 * 11, block_size**2 * 13, block_size))

            assert torch.all(layer.partitioned_pfids[0][0,:] == torch.arange(0, block_size**2 * 2, block_size))
            assert torch.all(layer.partitioned_pfids[0][1,:] == torch.arange(block_size**2 * 2, block_size**2 * 4, block_size))
            assert torch.all(layer.partitioned_pfids[0][2,:] == torch.arange(block_size**2 * 4, block_size**2 * 6, block_size))
            assert torch.all(layer.partitioned_pfids[0][3,:] == torch.arange(block_size**2 * 6, block_size**2 * 8, block_size))
            assert torch.all(layer.partitioned_pfids[0][4,:] == torch.arange(block_size**2 * 8, block_size**2 * 10, block_size))
            assert torch.all(layer.partitioned_pfids[0][5,:] == torch.arange(block_size**2 * 10, block_size**2 * 12, block_size))

            assert torch.all(layer.partitioned_chids[0] == torch.arange(block_size, block_size * 7, block_size))

            assert torch.all(layer.partitioned_parids[0][0,:] == torch.tensor([block_size, block_size * 2]))
            assert torch.all(layer.partitioned_parids[0][1,:] == torch.tensor([block_size, block_size * 2]))
            assert torch.all(layer.partitioned_parids[0][2,:] == torch.tensor([block_size * 3, block_size * 4]))
            assert torch.all(layer.partitioned_parids[0][3,:] == torch.tensor([block_size * 3, block_size * 4]))
            assert torch.all(layer.partitioned_parids[0][4,:] == torch.tensor([block_size * 5, block_size * 6]))
            assert torch.all(layer.partitioned_parids[0][5,:] == torch.tensor([block_size * 5, block_size * 6]))

            assert torch.all(layer.partitioned_parpids[0][0,:] == torch.tensor([block_size**2 * 1, block_size**2 * 3]))
            assert torch.all(layer.partitioned_parpids[0][1,:] == torch.tensor([block_size**2 * 2, block_size**2 * 4]))
            assert torch.all(layer.partitioned_parpids[0][2,:] == torch.tensor([block_size**2 * 5, block_size**2 * 7]))
            assert torch.all(layer.partitioned_parpids[0][3,:] == torch.tensor([block_size**2 * 6, block_size**2 * 8]))
            assert torch.all(layer.partitioned_parpids[0][4,:] == torch.tensor([block_size**2 * 9, block_size**2 * 11]))
            assert torch.all(layer.partitioned_parpids[0][5,:] == torch.tensor([block_size**2 * 10, block_size**2 * 12]))

            layer.to(device)

            ## Forward tests ##

            element_mars = torch.rand([block_size + 3 * 2 * 2 * block_size, batch_size]).log().to(device)
            element_mars[:block_size,:] = -float("inf")
            node_mars = torch.zeros([block_size + block_size * 2 * 3, batch_size]).to(device)

            params = torch.rand([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

            layer(node_mars, element_mars, params, force_use_fp16 = force_use_fp16, force_use_fp32 = force_use_fp32)

            for i in range(block_size):
                for j in range(6):
                    cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                    epars = params[layer.partitioned_pids[0][j,:]+i]
                    assert torch.all(torch.abs(node_mars[(j+1)*block_size+i,:] - (epars[:,None] * cmars).sum(dim = 0).log()) < 2e-3)

            ## Backward tests ##

            node_flows = torch.rand([block_size + block_size * 2 * 3, batch_size]).to(device)
            element_flows = torch.zeros([block_size + 3 * 2 * 2 * block_size, batch_size]).to(device)

            param_flows = torch.zeros([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

            layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows)

            chids = layer.partitioned_chids[0]
            parids = layer.partitioned_parids[0]
            parpids = layer.partitioned_parpids[0]

            num_nblocks = chids.size(0)
            num_eblocks = parids.size(1)
            parids = (parids[:,:,None].repeat(1, 1, block_size) + torch.arange(0, block_size, device = parids.device)).reshape(num_nblocks, num_eblocks * block_size)
            parpids_start = (parpids[:,:,None] + torch.arange(0, block_size, device = parids.device)).reshape(
                num_nblocks, num_eblocks * block_size)

            for j in range(6):
                parpids = parpids_start.clone()
                for i in range(block_size):
                    nmars = node_mars[parids[j,:]].exp()
                    nflows = node_flows[parids[j,:]]
                    emars = element_mars[(j+1)*block_size+i,:].exp()
                    epars = params[parpids[j,:]]
                    eflows = (nflows * epars[:,None] * emars[None,:] / nmars).sum(dim = 0)

                    assert torch.all(torch.abs(eflows - element_flows[(j+1)*block_size+i,:]) < 1e-2)

                    parpids += block_size

            my_pflows = torch.zeros_like(param_flows)

            for i in range(block_size):
                for j in range(6):
                    emars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                    epars = params[layer.partitioned_pids[0][j,:]+i]
                    nmars = node_mars[(j+1)*block_size+i,:].exp()
                    nflows = node_flows[(j+1)*block_size+i,:]
                    pflows = epars * (nflows[None,:] * emars / nmars[None,:]).sum(dim = 1)

                    my_pflows[layer.partitioned_pfids[0][j,:]+i] = pflows

            assert torch.all(torch.abs(my_pflows - param_flows) < 2e-3)


@pytest.mark.slow
def test_speed():

    device = torch.device("cuda:0")

    block_size = 32
    num_vars = 28*28
    num_node_blocks = 256 // block_size
    num_prod_nodes = 200

    batch_size = 512

    with juice.set_block_size(block_size):

        nis = []
        for v in range(num_vars):
            nis.append(inputs(v, num_node_blocks = num_node_blocks, dist = dists.Categorical(num_cats = 64)))

        nps = []
        for i in range(num_prod_nodes):
            v1 = random.randint(0, num_vars - 1)
            v2 = random.randint(0, num_vars - 1)
            if v1 == v2:
                if v1 == num_vars - 1:
                    v1 -= 2
                v2 = v1 + 1

            nps.append(multiply(nis[v1], nis[v2]))

        nodes = [summate(np, num_node_blocks = num_node_blocks) for np in nps]

    input_layer = InputLayer(nis, cum_nodes = block_size)

    prod_layer = ProdLayer(nps, layer_sparsity_tol = 0.1)

    layer = SumLayer(nodes, global_nid_start = block_size,
                     global_pid_start = block_size ** 2, global_pfid_start = 0, node2tiednodes = dict(), )

    layer.to(device)

    node_mars = torch.zeros([block_size + block_size * num_node_blocks * num_prod_nodes, batch_size]).to(device)
    element_mars = torch.rand([block_size + num_prod_nodes * block_size * num_node_blocks, batch_size]).log().to(device)
    params = torch.rand([layer.partitioned_pids[0].max() + block_size ** 2]).to(device)

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
    print("Reference computation time on RTX 4090: 0.635ms.")
    print("--------------------------------------------------------------")

    node_flows = torch.rand([block_size + block_size * num_node_blocks * num_prod_nodes, batch_size]).to(device)
    element_flows = torch.zeros([block_size + num_prod_nodes * block_size * num_node_blocks, batch_size]).log().to(device)
    param_flows = torch.zeros([block_size ** 2 + layer.partitioned_pids[0].max() + block_size]).to(device)

    layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows,
                   allow_modify_flows = True)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows,
                       allow_modify_flows = True)
    torch.cuda.synchronize()
    t1 = time.time()
    backward_ms = (t1 - t0) / 100 * 1000

    print(f"Backward pass on average takes {backward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 1.544ms.")
    print("--------------------------------------------------------------")


@pytest.mark.slow
def test_block_sparse_speed():

    device = torch.device("cuda:0")

    block_size = 32
    num_vars = 28*28
    num_node_blocks = 1024 // block_size
    num_prod_nodes = 200

    batch_size = 512

    with juice.set_block_size(block_size):

        nis = []
        for v in range(num_vars):
            nis.append(inputs(v, num_node_blocks = num_node_blocks, dist = dists.Categorical(num_cats = 64)))

        nps = []
        for i in range(num_prod_nodes):
            v1 = random.randint(0, num_vars - 1)
            v2 = random.randint(0, num_vars - 1)
            if v1 == v2:
                if v1 == num_vars - 1:
                    v1 -= 2
                v2 = v1 + 1

            nps.append(multiply(nis[v1], nis[v2]))

        nodes = []
        for np in nps:
            edge_ids = torch.rand([num_node_blocks, num_node_blocks]) < 0.2
            edge_ids[:,0] = True
            edge_ids = torch.nonzero(edge_ids, as_tuple = False).permute(1, 0)
            nodes.append(summate(np, num_node_blocks = num_node_blocks, edge_ids = edge_ids))

    input_layer = InputLayer(nis, cum_nodes = block_size)

    prod_layer = ProdLayer(nps, layer_sparsity_tol = 0.1)

    layer = SumLayer(nodes, global_nid_start = block_size,
                     global_pid_start = block_size ** 2, global_pfid_start = 0, node2tiednodes = dict(), 
                     layer_sparsity_tol = 0.1)

    layer.to(device)

    node_mars = torch.zeros([block_size + block_size * num_node_blocks * num_prod_nodes, batch_size]).to(device)
    element_mars = torch.rand([block_size + num_prod_nodes * block_size * num_node_blocks, batch_size]).log().to(device)
    params = torch.rand([layer.partitioned_pids[0].max() + block_size ** 2]).to(device)

    ## Forward tests ##

    layer(node_mars, element_mars, params)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer(node_mars, element_mars, params)
    torch.cuda.synchronize()
    t1 = time.time()
    forward_ms = (t1 - t0) / 100 * 1000

    print(f"Sparse forward pass on average takes {forward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 2.594ms.")
    print("--------------------------------------------------------------")

    node_flows = torch.rand([block_size + block_size * num_node_blocks * num_prod_nodes, batch_size]).to(device)
    element_flows = torch.zeros([block_size + num_prod_nodes * block_size * num_node_blocks, batch_size]).log().to(device)
    param_flows = torch.zeros([block_size ** 2 + layer.partitioned_pids[0].max() + block_size]).to(device)

    layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows,
                   allow_modify_flows = True)

    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(100):
        layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows,
                       allow_modify_flows = True)
    torch.cuda.synchronize()
    t1 = time.time()
    backward_ms = (t1 - t0) / 100 * 1000

    print(f"Sparse backward pass on average takes {backward_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 8.528ms.")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    torch.manual_seed(3890)
    test_sum_layer()
    test_corner_case()
    test_speed()
    test_block_sparse_speed()