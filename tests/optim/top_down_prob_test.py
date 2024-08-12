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
from pyjuice.model.backend import eval_top_down_probs

import pytest


def test_simple_model_tdp():

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
    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, 1), set_value = 0.0)
    pc._init_buffer(name = "element_flows", shape = (pc.num_elements, 1), set_value = 0.0)

    eval_top_down_probs(pc, update_pflow = True, scale = 1.0)

    ns_params = ns._params.reshape(-1)

    np4_tdp = ns_params[:32]
    ns0_tdp = np4_tdp
    sid, eid = ns0._output_ind_range
    assert torch.all(torch.abs(ns0_tdp - pc.node_flows[sid:eid,0].cpu()) < 1e-5)

    np5_tdp = ns_params[32:64]
    ns1_tdp = np5_tdp
    sid, eid = ns1._output_ind_range
    assert torch.all(torch.abs(ns1_tdp - pc.node_flows[sid:eid,0].cpu()) < 1e-5)

    np6_tdp = ns_params[64:96]
    ns2_tdp = np6_tdp
    sid, eid = ns2._output_ind_range
    assert torch.all(torch.abs(ns2_tdp - pc.node_flows[sid:eid,0].cpu()) < 1e-5)

    sid, eid = ns._param_flow_range
    assert torch.all(torch.abs(ns_params - pc.param_flows[sid:eid].cpu()) < 1e-5)

    ns0_params = ns0._params.reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)
    ns0_epars_tdp = ns0_tdp[:,None] * ns0_params
    np03_tdp = (ns0_tdp[None,:] @ ns0_params).squeeze(0)
    np0_tdp = np03_tdp[:32]
    np3_tdp = np03_tdp[32:]

    sid, eid = np0._output_ind_range
    assert torch.all(torch.abs(np0_tdp - pc.element_flows[sid:eid,0].cpu()) < 1e-5)

    sid, eid = np3._output_ind_range
    assert torch.all(torch.abs(np3_tdp - pc.element_flows[sid:eid,0].cpu()) < 1e-5)

    sid, eid = ns0._param_flow_range
    epars_flow = pc.param_flows[sid:eid].reshape(2, 4, 16, 16).permute(0, 3, 1, 2).reshape(32, 64)
    assert torch.all(torch.abs(ns0_epars_tdp - epars_flow.cpu()) < 1e-5)

    ns1_params = ns1._params.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    ns1_epars_tdp = ns1_tdp[:,None] * ns1_params
    np1_tdp = (ns1_tdp[None,:] @ ns1_params).squeeze(0)
    
    sid, eid = np1._output_ind_range
    assert torch.all(torch.abs(np1_tdp - pc.element_flows[sid:eid,0].cpu()) < 1e-5)

    sid, eid = ns1._param_flow_range
    epars_flow = pc.param_flows[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32)
    assert torch.all(torch.abs(ns1_epars_tdp - epars_flow.cpu()) < 1e-5)

    ns2_params = ns2._params.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    ns2_epars_tdp = ns2_tdp[:,None] * ns2_params
    np2_tdp = (ns2_tdp[None,:] @ ns2_params).squeeze(0)

    sid, eid = ns2._param_flow_range
    epars_flow = pc.param_flows[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32)
    assert torch.all(torch.abs(ns2_epars_tdp - epars_flow.cpu()) < 1e-5)

    sid, eid = np2._output_ind_range
    assert torch.all(torch.abs(np2_tdp - pc.element_flows[sid:eid,0].cpu()) < 1e-5)

    ni0_tdp = np0_tdp + np3_tdp + np5_tdp + np6_tdp
    ni1_tdp = np0_tdp + np2_tdp + np3_tdp + np5_tdp
    ni2_tdp = np1_tdp + np2_tdp + np4_tdp
    ni3_tdp = np1_tdp + np4_tdp + np6_tdp

    sid, eid = ni0._output_ind_range
    assert torch.all(torch.abs(ni0_tdp - pc.node_flows[sid:eid,0].cpu()) < 1e-5)

    sid, eid = ni1._output_ind_range
    assert torch.all(torch.abs(ni1_tdp - pc.node_flows[sid:eid,0].cpu()) < 1e-5)

    sid, eid = ni2._output_ind_range
    assert torch.all(torch.abs(ni2_tdp - pc.node_flows[sid:eid,0].cpu()) < 1e-5)

    sid, eid = ni3._output_ind_range
    assert torch.all(torch.abs(ni3_tdp - pc.node_flows[sid:eid,0].cpu()) < 1e-5)

    input_layer = pc.input_layer_group[0]

    ni0_epars_tdp = ni0_tdp[:,None] * input_layer.params[:32*4].reshape(32, 4).cpu()
    assert torch.all(torch.abs(ni0_epars_tdp - input_layer.param_flows[:32*4].reshape(32, 4).cpu()) < 1e-5)

    ni1_epars_tdp = ni1_tdp[:,None] * input_layer.params[32*4:32*4*2].reshape(32, 4).cpu()
    assert torch.all(torch.abs(ni1_epars_tdp - input_layer.param_flows[32*4:32*4*2].reshape(32, 4).cpu()) < 1e-5)

    ni2_epars_tdp = ni2_tdp[:,None] * input_layer.params[32*4*2:32*4*2+32*6].reshape(32, 6).cpu()
    assert torch.all(torch.abs(ni2_epars_tdp - input_layer.param_flows[32*4*2:32*4*2+32*6].reshape(32, 6).cpu()) < 1e-5)

    ni3_epars_tdp = ni3_tdp[:,None] * input_layer.params[32*4*2+32*6:32*4*2+32*6*2].reshape(32, 6).cpu()
    assert torch.all(torch.abs(ni3_epars_tdp - input_layer.param_flows[32*4*2+32*6:32*4*2+32*6*2].reshape(32, 6).cpu()) < 1e-5)


def test_scaled_mini_batch_em():

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
    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, 1), set_value = 0.0)
    pc._init_buffer(name = "element_flows", shape = (pc.num_elements, 1), set_value = 0.0)

    eval_top_down_probs(pc, update_pflow = True, scale = 1.0)

    sid, eid = ns._param_flow_range
    ns_epars_tdp = pc.param_flows[sid:eid].clone()

    sid, eid = ns0._param_flow_range
    ns0_epars_tdp = pc.param_flows[sid:eid].reshape(2, 4, 16, 16).permute(0, 3, 1, 2).reshape(32, 64).clone()

    sid, eid = ns1._param_flow_range
    ns1_epars_tdp = pc.param_flows[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32).clone()

    sid, eid = ns2._param_flow_range
    ns2_epars_tdp = pc.param_flows[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32).clone()

    input_layer = pc.input_layer_group[0]

    ni0_epars_tdp = input_layer.param_flows[:32*4].reshape(32, 4).clone()
    ni1_epars_tdp = input_layer.param_flows[32*4:32*4*2].reshape(32, 4).clone()
    ni2_epars_tdp = input_layer.param_flows[32*4*2:32*4*2+32*6].reshape(32, 6).clone()
    ni3_epars_tdp = input_layer.param_flows[32*4*2+32*6:32*4*2+32*6*2].reshape(32, 6).clone()

    batch_size = 16
    data = torch.randint(0, 4, [batch_size, 4], device = device)

    pc.init_param_flows()

    lls = pc(data)
    pc.backward(data, flows_memory = 0.0)

    sid, eid = ns._param_flow_range
    ns_epars_flow = pc.param_flows[sid:eid].clone()

    sid, eid = ns0._param_flow_range
    ns0_epars_flow = pc.param_flows[sid:eid].reshape(2, 4, 16, 16).permute(0, 3, 1, 2).reshape(32, 64).clone()

    sid, eid = ns1._param_flow_range
    ns1_epars_flow = pc.param_flows[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32).clone()

    sid, eid = ns2._param_flow_range
    ns2_epars_flow = pc.param_flows[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32).clone()

    input_layer = pc.input_layer_group[0]

    ni0_epars_flow = input_layer.param_flows[:32*4].reshape(32, 4).clone()
    ni1_epars_flow = input_layer.param_flows[32*4:32*4*2].reshape(32, 4).clone()
    ni2_epars_flow = input_layer.param_flows[32*4*2:32*4*2+32*6].reshape(32, 6).clone()
    ni3_epars_flow = input_layer.param_flows[32*4*2+32*6:32*4*2+32*6*2].reshape(32, 6).clone()

    alpha = 0.1

    ns_params = ns_epars_flow * alpha / batch_size + ns_epars_tdp * (1.0 - alpha)
    ns_params /= ns_params.sum()

    ns0_params = ns0_epars_flow * alpha / batch_size + ns0_epars_tdp * (1.0 - alpha)
    ns0_params /= ns0_params.sum(dim = 1, keepdim = True)

    ns1_params = ns1_epars_flow * alpha / batch_size + ns1_epars_tdp * (1.0 - alpha)
    ns1_params /= ns1_params.sum(dim = 1, keepdim = True)

    ns2_params = ns2_epars_flow * alpha / batch_size + ns2_epars_tdp * (1.0 - alpha)
    ns2_params /= ns2_params.sum(dim = 1, keepdim = True)

    ni0_params = ni0_epars_flow * alpha / batch_size + ni0_epars_tdp * (1.0 - alpha)
    ni0_params /= ni0_params.sum(dim = 1, keepdim = True)

    ni1_params = ni1_epars_flow * alpha / batch_size + ni1_epars_tdp * (1.0 - alpha)
    ni1_params /= ni1_params.sum(dim = 1, keepdim = True)

    ni2_params = ni2_epars_flow * alpha / batch_size + ni2_epars_tdp * (1.0 - alpha)
    ni2_params /= ni2_params.sum(dim = 1, keepdim = True)

    ni3_params = ni3_epars_flow * alpha / batch_size + ni3_epars_tdp * (1.0 - alpha)
    ni3_params /= ni3_params.sum(dim = 1, keepdim = True)

    pc.mini_batch_em(step_size = alpha, step_size_rescaling = True)

    sid, eid = ns._param_range
    epars = pc.params[sid:eid]
    assert torch.all(torch.abs(ns_params - epars) < 1e-5)

    sid, eid = ns0._param_range
    epars = pc.params[sid:eid].reshape(2, 4, 16, 16).permute(0, 3, 1, 2).reshape(32, 64).clone()
    assert torch.all(torch.abs(ns0_params - epars) < 1e-5)

    sid, eid = ns1._param_range
    epars = pc.params[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32).clone()
    assert torch.all(torch.abs(ns1_params - epars) < 1e-5)

    sid, eid = ns2._param_range
    epars = pc.params[sid:eid].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(32, 32).clone()
    assert torch.all(torch.abs(ns2_params - epars) < 1e-5)

    epars = input_layer.params[:32*4].reshape(32, 4).clone()
    assert torch.all(torch.abs(ni0_params - epars) < 1e-5)

    epars = input_layer.params[32*4:32*4*2].reshape(32, 4).clone()
    assert torch.all(torch.abs(ni1_params - epars) < 1e-5)

    epars = input_layer.params[32*4*2:32*4*2+32*6].reshape(32, 6).clone()
    assert torch.all(torch.abs(ni2_params - epars) < 1e-5)

    epars = input_layer.params[32*4*2+32*6:32*4*2+32*6*2].reshape(32, 6).clone()
    assert torch.all(torch.abs(ni3_params - epars) < 1e-5)


@pytest.mark.slow
def test_tdp_speed():

    device = torch.device("cuda:0")
    
    root_ns = juice.structures.HMM(seq_length = 32, num_latents = 4096, num_emits = 50000, homogeneous = True)

    pc = TensorCircuit(root_ns, layer_sparsity_tol = 0.1)
    pc.to(device)

    pc.init_param_flows()
    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, 1), set_value = 0.0)
    pc._init_buffer(name = "element_flows", shape = (pc.num_elements, 1), set_value = 0.0)

    eval_top_down_probs(pc, update_pflow = True, scale = 1.0)

    t0 = time.time()
    for _ in range(100):
        eval_top_down_probs(pc, update_pflow = True, scale = 1.0)
    t1 = time.time()

    tdp_ms = (t1 - t0) / 100 * 1000

    print(f"Computing TDP on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 31.423ms.")
    print("--------------------------------------------------------------")



if __name__ == "__main__":
    torch.set_num_threads(8)
    test_simple_model_tdp()
    test_scaled_mini_batch_em()
    test_tdp_speed()
