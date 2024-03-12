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


def test_ll_prop():
    
    device = torch.device("cuda:0")

    batch_size = 16

    for block_size in [1, 4, 8, 16]:

        for allow_modify_flows in [True, False]:
    
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

            layer(node_mars, element_mars, params, propagation_alg = "LL")

            for i in range(block_size):
                for j in range(6):
                    cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                    epars = params[layer.partitioned_pids[0][j,:]+i]
                    scaled_lls = (epars[:,None] * cmars).sum(dim = 0).log()
                    
                    assert torch.all(torch.abs(node_mars[(j+1)*block_size+i,:] - scaled_lls) < 2e-3)

            ## Backward pass ##

            node_flows = torch.rand([block_size + block_size * 2 * 3, batch_size]).to(device)
            element_flows = torch.zeros([block_size + 3 * 2 * 2 * block_size, batch_size]).to(device)

            param_flows = torch.zeros([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

            origin_node_flows = node_flows.clone()

            layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows, 
                           allow_modify_flows = allow_modify_flows, propagation_alg = "LL")

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
                    nflows = origin_node_flows[parids[j,:]]
                    emars = element_mars[(j+1)*block_size+i,:].exp()
                    epars = params[parpids[j,:]]
                    eflows = (nflows * (epars[:,None] * emars[None,:]) / nmars).sum(dim = 0)

                    if allow_modify_flows:
                        uflows1 = node_flows[parids[j,:]]
                        uflows2 = origin_node_flows[parids[j,:]].log() - nmars.log()

                        assert torch.all(torch.abs(uflows1 - uflows2) < 1e-3)

                    assert torch.all(torch.abs(eflows - element_flows[(j+1)*block_size+i,:]) < 1e-2)

                    parpids += block_size

            my_pflows = torch.zeros_like(param_flows)

            for i in range(block_size):
                for j in range(6):
                    emars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                    epars = params[layer.partitioned_pids[0][j,:]+i]
                    nmars = node_mars[(j+1)*block_size+i,:].exp()
                    nflows = origin_node_flows[(j+1)*block_size+i,:]
                    pflows = epars * (nflows[None,:] * emars / nmars[None,:]).sum(dim = 1)

                    my_pflows[layer.partitioned_pfids[0][j,:]+i] = pflows

            assert torch.all(torch.abs(my_pflows - param_flows) < 2e-3)


def test_general_ll_prop():
    
    device = torch.device("cuda:0")

    batch_size = 16

    for block_size in [1, 4, 8, 16]:

        for allow_modify_flows in [True, False]:
    
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

            alphas = [1.2, 2.0, 3.0]

            for alpha in alphas:

                ## Forward pass ##

                element_mars = torch.rand([block_size + 3 * 2 * 2 * block_size, batch_size]).log().to(device)
                element_mars[:block_size,:] = -float("inf")
                node_mars = torch.zeros([block_size + block_size * 2 * 3, batch_size]).to(device)

                params = torch.rand([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)
            
                layer(node_mars, element_mars, params, propagation_alg = "GeneralLL", alpha = alpha)

                for i in range(block_size):
                    for j in range(6):
                        cmars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                        epars = params[layer.partitioned_pids[0][j,:]+i]
                        scaled_lls = (epars[:,None]**alpha * cmars**alpha).sum(dim = 0).log() * (1.0 / alpha)

                        assert torch.all(torch.abs(node_mars[(j+1)*block_size+i,:] - scaled_lls) < 2e-3)

                ## Backward pass ##

                node_flows = torch.rand([block_size + block_size * 2 * 3, batch_size]).to(device)
                element_flows = torch.zeros([block_size + 3 * 2 * 2 * block_size, batch_size]).to(device)

                param_flows = torch.zeros([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

                origin_node_flows = node_flows.clone()

                layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows, 
                               allow_modify_flows = allow_modify_flows, propagation_alg = "GeneralLL", alpha = alpha)

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
                        nflows = origin_node_flows[parids[j,:]]
                        emars = element_mars[(j+1)*block_size+i,:].exp()
                        epars = params[parpids[j,:]]
                        eflows = (nflows * (epars[:,None] * emars[None,:]) ** alpha / nmars ** alpha).sum(dim = 0)

                        if allow_modify_flows:
                            uflows1 = node_flows[parids[j,:]]
                            uflows2 = origin_node_flows[parids[j,:]].log() - nmars.log() * alpha
                            assert torch.all(torch.abs(uflows1 - uflows2) < 1e-3)

                        assert torch.all(torch.abs(eflows - element_flows[(j+1)*block_size+i,:]) < 1e-2)

                        parpids += block_size

                my_pflows = torch.zeros_like(param_flows)

                for i in range(block_size):
                    for j in range(6):
                        emars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                        epars = params[layer.partitioned_pids[0][j,:]+i]
                        nmars = node_mars[(j+1)*block_size+i,:].exp()
                        nflows = origin_node_flows[(j+1)*block_size+i,:]
                        pflows = epars * (nflows[None,:] * emars / nmars[None,:]).sum(dim = 1)

                        my_pflows[layer.partitioned_pfids[0][j,:]+i] = pflows

                assert torch.all(torch.abs(my_pflows - param_flows) < 4e-3)


def test_mpe_prop():

    device = torch.device("cuda:0")

    batch_size = 16

    for block_size in [1, 4, 8, 16]:

        for allow_modify_flows in [True, False]:
    
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

            ## Backward pass ##

            node_flows = torch.rand([block_size + block_size * 2 * 3, batch_size]).to(device)
            element_flows = torch.zeros([block_size + 3 * 2 * 2 * block_size, batch_size]).to(device)

            param_flows = torch.zeros([block_size ** 2 + 3 * 4 * block_size * block_size]).to(device)

            origin_node_flows = node_flows.clone()

            layer.backward(node_flows, element_flows, node_mars, element_mars, params, param_flows, 
                           allow_modify_flows = allow_modify_flows, propagation_alg = "MPE")

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
                    nflows = origin_node_flows[parids[j,:]]
                    emars = element_mars[(j+1)*block_size+i,:].exp()
                    epars = params[parpids[j,:]]
                    eflows = (nflows * (((epars[:,None] * emars[None,:]) - nmars).abs() < 1e-6).float()).sum(dim = 0)

                    assert torch.all(torch.abs(eflows - element_flows[(j+1)*block_size+i,:]) < 1e-2)

                    parpids += block_size

            my_pflows = torch.zeros_like(param_flows)

            for i in range(block_size):
                for j in range(6):
                    emars = element_mars[layer.partitioned_cids[0][j,:]].exp()
                    epars = params[layer.partitioned_pids[0][j,:]+i]
                    nmars = node_mars[(j+1)*block_size+i,:].exp()
                    nflows = origin_node_flows[(j+1)*block_size+i,:]
                    pflows = (nflows[None,:] * ((epars[:,None] * emars - nmars[None,:]).abs() < 1e-6).float()).sum(dim = 1)

                    my_pflows[layer.partitioned_pfids[0][j,:]+i] = pflows

            assert torch.all(torch.abs(my_pflows - param_flows) < 2e-3)


if __name__ == "__main__":
    torch.manual_seed(280)
    test_ll_prop()
    test_general_ll_prop()
    test_mpe_prop()
