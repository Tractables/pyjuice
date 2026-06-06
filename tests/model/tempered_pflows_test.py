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


def test_tempered_pflows_forward():

    torch.manual_seed(63892)

    device = torch.device("cuda:0")

    block_size = 16
    batch_size = 32
    temperature = 0.5
    
    with juice.set_block_size(block_size):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)

        ns0 = summate(np0, num_node_blocks = 2)
        ns1 = summate(np1, num_node_blocks = 2)

        np2 = multiply(ns0, ns1)
        ns = summate(np2, num_node_blocks = 1, block_size = 1)

    ns.init_parameters(perturbation = 4.0)
    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, 2, [batch_size, 4]).to(device)

    lls = pc(data, pflow_temperature = temperature)

    ni0_mars = pc.node_mars[ni0._output_ind_range[0]:ni0._output_ind_range[1],:]
    ni1_mars = pc.node_mars[ni1._output_ind_range[0]:ni1._output_ind_range[1],:]
    np0_mars = ni0_mars + ni1_mars

    ns0_params = ns0.get_params(as_matrix = True).to(device)

    ns0_mars = pc.node_mars[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]
    ns0_mars_target = torch.matmul(ns0_params, np0_mars.exp()).log()
    assert torch.all(torch.abs(ns0_mars - ns0_mars_target) < 1e-2)

    ns0_mars_tempered = pc.node_mars_tempered[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]
    ns0_mars_tempered_target = torch.matmul(ns0_params.pow(1.0 / temperature), (np0_mars / temperature).exp()).log()
    assert torch.all(torch.abs(ns0_mars_tempered - ns0_mars_tempered_target) < 1e-2)

    ns1_mars = pc.node_mars[ns1._output_ind_range[0]:ns1._output_ind_range[1],:]
    np2_mars = ns0_mars + ns1_mars

    ns_params = ns.get_params(as_matrix = True).to(device)

    ns_mars = pc.node_mars[ns._output_ind_range[0]:ns._output_ind_range[1],:]
    ns_mars_target = torch.matmul(ns_params, np2_mars.exp()).log()
    assert torch.all(torch.abs(ns_mars - ns_mars_target) < 1e-2)

    ns_mars_tempered = pc.node_mars_tempered[ns._output_ind_range[0]:ns._output_ind_range[1],:]
    ns_mars_tempered_target = torch.matmul(ns_params.pow(1.0 / temperature), (np2_mars / temperature).exp()).log()
    assert torch.all(torch.abs(ns_mars_tempered - ns_mars_tempered_target) < 1e-2)


def test_tempered_pflows_backward():

    torch.manual_seed(63892)

    device = torch.device("cuda:0")

    block_size = 16
    batch_size = 32
    temperature = 0.5
    
    with juice.set_block_size(block_size):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)

        ns0 = summate(np0, num_node_blocks = 2)
        ns1 = summate(np1, num_node_blocks = 2)

        np2 = multiply(ns0, ns1)
        ns = summate(np2, num_node_blocks = 1, block_size = 1)

    ns.init_parameters(perturbation = 4.0)
    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, 2, [batch_size, 4]).to(device)

    pc.init_param_flows()

    lls = pc(data, pflow_temperature = temperature)
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, pflow_temperature = temperature)

    pc.update_param_flows()

    ns_param_flows = ns.get_param_flows(as_matrix = True).to(device)[0,:]
    ns_params = ns.get_params(as_matrix = True).to(device)[0,:]

    ns0_mars = pc.node_mars[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]
    ns1_mars = pc.node_mars[ns1._output_ind_range[0]:ns1._output_ind_range[1],:]
    np2_mars = ns0_mars + ns1_mars

    unnorm_flows = (ns_params[:,None] * np2_mars.exp()).pow(1.0 / temperature)
    ns_parflows_target = (unnorm_flows / unnorm_flows.sum(dim = 0, keepdim = True)).sum(dim = 1)
    assert torch.all(torch.abs(ns_parflows_target - ns_param_flows) < 1e-2)

    ns0_param_flows = ns0.get_param_flows(as_matrix = True).to(device)
    ns0_params = ns0.get_params(as_matrix = True).to(device)

    ni0_mars = pc.node_mars[ni0._output_ind_range[0]:ni0._output_ind_range[1],:]
    ni1_mars = pc.node_mars[ni1._output_ind_range[0]:ni1._output_ind_range[1],:]
    np0_mars = ni0_mars + ni1_mars

    ns0_flows = pc.node_flows[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]
    ns0_mars_tempered = pc.node_mars_tempered[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]

    ns0_parflows_target = (ns0_flows[:,None,:] + ns0_params[:,:,None].pow(1.0 / temperature).log() + \
        (np0_mars[None,:,:] / temperature) - ns0_mars_tempered[:,None,:]).exp().sum(dim = 2)
    assert torch.all(torch.abs(ns0_parflows_target - ns0_param_flows) < 1e-3)


def test_tempered_eflows_backward():

    torch.manual_seed(63892)

    device = torch.device("cuda:0")

    block_size = 16
    batch_size = 32
    temperature = 0.5
    
    with juice.set_block_size(block_size):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)

        ns0 = summate(np0, num_node_blocks = 2)
        ns1 = summate(np1, num_node_blocks = 2)

        np2 = multiply(ns0, ns1)
        ns = summate(np2, num_node_blocks = 1, block_size = 1)

    ns.init_parameters(perturbation = 4.0)
    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, 2, [batch_size, 4]).to(device)

    pc.init_param_flows()

    lls = pc(data, pflow_temperature = temperature)
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                pflow_temperature = temperature, temper_eflow = True)

    ns_params = ns.get_params(as_matrix = True).to(device)[0,:]

    ns0_mars = pc.node_mars[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]
    ns1_mars = pc.node_mars[ns1._output_ind_range[0]:ns1._output_ind_range[1],:]
    np2_mars = ns0_mars + ns1_mars

    unnorm_flows = (ns_params[:,None] * np2_mars.exp()).pow(1.0 / temperature)
    np2_eflows_target = (unnorm_flows / unnorm_flows.sum(dim = 0, keepdim = True)).log()

    ns0_mars_tempered = pc.node_mars_tempered[ns._output_ind_range[0]:ns._output_ind_range[1],:]
    assert torch.all(torch.abs(unnorm_flows.sum(dim = 0, keepdim = True).log() - ns0_mars_tempered) < 1e-3)

    ns0_flows = pc.node_flows[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]
    ns1_flows = pc.node_flows[ns1._output_ind_range[0]:ns1._output_ind_range[1],:]
    assert torch.all(torch.abs(np2_eflows_target - ns0_flows) < 1e-2)

    ns0_params = ns0.get_params(as_matrix = True).to(device)

    ni0_mars = pc.node_mars[ni0._output_ind_range[0]:ni0._output_ind_range[1],:]
    ni1_mars = pc.node_mars[ni1._output_ind_range[0]:ni1._output_ind_range[1],:]
    np0_mars = ni0_mars + ni1_mars

    ns0_flows = pc.node_flows[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]
    ns0_mars_tempered = pc.node_mars_tempered[ns0._output_ind_range[0]:ns0._output_ind_range[1],:]

    np0_eflows_target = (ns0_flows[:,None,:] + ns0_params[:,:,None].pow(1.0 / temperature).log() + \
        (np0_mars[None,:,:] / temperature) - ns0_mars_tempered[:,None,:]).exp().sum(dim = 0).log()
    ni0_eflows = pc.node_flows[ni0._output_ind_range[0]:ni0._output_ind_range[1],:]
    assert torch.all(torch.abs(np0_eflows_target - ni0_eflows) < 1e-2)

    ns1_params = ns1.get_params(as_matrix = True).to(device)

    ni2_mars = pc.node_mars[ni2._output_ind_range[0]:ni2._output_ind_range[1],:]
    ni3_mars = pc.node_mars[ni3._output_ind_range[0]:ni3._output_ind_range[1],:]
    np1_mars = ni2_mars + ni3_mars

    ns1_flows = pc.node_flows[ns1._output_ind_range[0]:ns1._output_ind_range[1],:]
    ns1_mars_tempered = pc.node_mars_tempered[ns1._output_ind_range[0]:ns1._output_ind_range[1],:]

    np1_eflows_target = (ns1_flows[:,None,:] + ns1_params[:,:,None].pow(1.0 / temperature).log() + \
        (np1_mars[None,:,:] / temperature) - ns1_mars_tempered[:,None,:]).exp().sum(dim = 0).log()
    ni2_eflows = pc.node_flows[ni2._output_ind_range[0]:ni2._output_ind_range[1],:]
    assert torch.all(torch.abs(np1_eflows_target - ni2_eflows) < 1e-2)


if __name__ == "__main__":
    test_tempered_pflows_forward()
    test_tempered_pflows_backward()
    test_tempered_eflows_backward()
