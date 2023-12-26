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


def simple_model_test():
    
    device = torch.device("cuda:0")

    group_size = 16
    
    with juice.set_group_size(group_size):

        ni0 = inputs(0, num_node_groups = 2, dist = dists.Categorical(num_cats = 4))
        ni1 = inputs(1, num_node_groups = 2, dist = dists.Categorical(num_cats = 4))
        ni2 = inputs(2, num_node_groups = 2, dist = dists.Categorical(num_cats = 6))
        ni3 = inputs(3, num_node_groups = 2, dist = dists.Categorical(num_cats = 6))

        np0 = multiply(ni0, ni1)
        np1 = multiply(ni2, ni3)
        np2 = multiply(ni1, ni2)
        np3 = multiply(ni0, ni1)

        ns0 = summate(np0, np3, num_node_groups = 2)
        ns1 = summate(np1, num_node_groups = 2)
        ns2 = summate(np2, num_node_groups = 2)

        np4 = multiply(ns0, ni2, ni3)
        np5 = multiply(ns1, ni0, ni1)
        np6 = multiply(ns2, ni0, ni3)

        ns = summate(np4, np5, np6, num_node_groups = 1, group_size = 1)

    ns.init_parameters()

    pc = TensorCircuit(ns, layer_sparsity_tol = 0.1)

    ## Test all compilation-related stuff ##

    input_layer = pc.input_layer_group[0]

    assert torch.all(input_layer.vids[0:32,0] == 0)
    assert torch.all(input_layer.vids[32:64,0] == 1)
    assert torch.all(input_layer.vids[64:96,0] == 2)
    assert torch.all(input_layer.vids[96:128,0] == 3)

    assert torch.all(input_layer.s_pids[:64] == torch.arange(0, 64*4, 4))
    assert torch.all(input_layer.s_pids[64:] == torch.arange(64*4, 64*(4+6), 6))

    assert torch.all(input_layer.s_pfids[:64] == torch.arange(0, 64*4, 4))
    assert torch.all(input_layer.s_pfids[64:] == torch.arange(64*4, 64*(4+6), 6))

    assert torch.all(input_layer.s_mids[0:32] == 0)
    assert torch.all(input_layer.s_mids[32:64] == 1)
    assert torch.all(input_layer.s_mids[64:96] == 2)
    assert torch.all(input_layer.s_mids[96:128] == 3)

    assert torch.all(input_layer.source_nids == torch.arange(0, 128))

    assert input_layer.num_parameters == 64 * (4 + 6)

    assert torch.all(torch.abs(input_layer.params[:64*4].reshape(64, 4).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(input_layer.params[64*4:].reshape(64, 6).sum(dim = 1) - 1.0) < 1e-4)

    prod_layer0 = pc.inner_layer_groups[0][0]

    assert prod_layer0.num_nodes == 4 * 16 * 2
    assert prod_layer0.num_edges == 8 * 16 * 2

    assert torch.all(prod_layer0.partitioned_nids[0] == torch.arange(16, 144, 16))

    assert torch.all(prod_layer0.partitioned_cids[0][0:2,:] == torch.tensor([[16, 48], [32, 64]]))
    assert torch.all(prod_layer0.partitioned_cids[0][2:4,:] == torch.tensor([[16, 48], [32, 64]]))
    assert torch.all(prod_layer0.partitioned_cids[0][4:6,:] == torch.tensor([[80, 112], [96, 128]]))
    assert torch.all(prod_layer0.partitioned_cids[0][6:8,:] == torch.tensor([[48, 80], [64, 96]]))

    assert torch.all(prod_layer0.partitioned_u_cids[0] == torch.tensor([16, 32, 80, 96, 112, 128]))
    assert torch.all(prod_layer0.partitioned_u_cids[1] == torch.tensor([48, 64]))

    assert torch.all(prod_layer0.partitioned_parids[0] == torch.tensor([[16, 48], [32, 64], [80, 112], [96, 128], [80, 0], [96, 0]]))
    assert torch.all(prod_layer0.partitioned_parids[1] == torch.tensor([[16, 48, 112, 0], [32, 64, 128, 0]]))

    sum_layer0 = pc.inner_layer_groups[1][0]

    assert torch.all(sum_layer0.partitioned_nids[0] == torch.tensor([176, 192, 208, 224]))
    assert torch.all(sum_layer0.partitioned_nids[1] == torch.tensor([144, 160]))

    assert torch.all(sum_layer0.partitioned_cids[0][0,:] == torch.arange(80, 112))
    assert torch.all(sum_layer0.partitioned_cids[0][1,:] == torch.arange(80, 112))
    assert torch.all(sum_layer0.partitioned_cids[0][2,:] == torch.arange(112, 144))
    assert torch.all(sum_layer0.partitioned_cids[0][3,:] == torch.arange(112, 144))
    assert torch.all(sum_layer0.partitioned_cids[1][0,:] == torch.arange(16, 80))
    assert torch.all(sum_layer0.partitioned_cids[1][1,:] == torch.arange(16, 80))

    assert torch.all(sum_layer0.partitioned_pids[0][0,:] == torch.arange(2304, 2816, 16))
    assert torch.all(sum_layer0.partitioned_pids[0][1,:] == torch.arange(2816, 3328, 16))
    assert torch.all(sum_layer0.partitioned_pids[0][2,:] == torch.arange(3328, 3840, 16))
    assert torch.all(sum_layer0.partitioned_pids[0][3,:] == torch.arange(3840, 4352, 16))
    assert torch.all(sum_layer0.partitioned_pids[1][0,:] == torch.arange(256, 1280, 16))
    assert torch.all(sum_layer0.partitioned_pids[1][1,:] == torch.arange(1280, 2304, 16))

    assert torch.all(sum_layer0.partitioned_pfids[0][0,:] == torch.arange(2048, 2560, 16))
    assert torch.all(sum_layer0.partitioned_pfids[0][1,:] == torch.arange(2560, 3072, 16))
    assert torch.all(sum_layer0.partitioned_pfids[0][2,:] == torch.arange(3072, 3584, 16))
    assert torch.all(sum_layer0.partitioned_pfids[0][3,:] == torch.arange(3584, 4096, 16))
    assert torch.all(sum_layer0.partitioned_pfids[1][0,:] == torch.arange(0, 1024, 16))
    assert torch.all(sum_layer0.partitioned_pfids[1][1,:] == torch.arange(1024, 2048, 16))

    assert torch.all(sum_layer0.partitioned_chids[0] == torch.arange(16, 144, 16))

    assert torch.all(sum_layer0.partitioned_parids[0][:4] == torch.tensor([[144, 160]]))
    assert torch.all(sum_layer0.partitioned_parids[0][4:6] == torch.tensor([[176, 192]]))
    assert torch.all(sum_layer0.partitioned_parids[0][6:8] == torch.tensor([[208, 224]]))

    assert torch.all(sum_layer0.partitioned_parpids[0][0,:] == torch.tensor([256, 1280]))
    assert torch.all(sum_layer0.partitioned_parpids[0][1,:] == torch.tensor([512, 1536]))
    assert torch.all(sum_layer0.partitioned_parpids[0][2,:] == torch.tensor([768, 1792]))
    assert torch.all(sum_layer0.partitioned_parpids[0][3,:] == torch.tensor([1024, 2048]))
    assert torch.all(sum_layer0.partitioned_parpids[0][4,:] == torch.tensor([2304, 2816]))
    assert torch.all(sum_layer0.partitioned_parpids[0][5,:] == torch.tensor([2560, 3072]))
    assert torch.all(sum_layer0.partitioned_parpids[0][6,:] == torch.tensor([3328, 3840]))
    assert torch.all(sum_layer0.partitioned_parpids[0][7,:] == torch.tensor([3584, 4096]))

    assert torch.all(torch.abs(ns0._params.reshape(2, 4, 16, 16).sum(dim = 3).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(ns1._params.reshape(2, 2, 16, 16).sum(dim = 3).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(ns2._params.reshape(2, 2, 16, 16).sum(dim = 3).sum(dim = 1) - 1.0) < 1e-4)

    assert torch.all(torch.abs(pc.params[:256] - 0.0) < 1e-4)
    assert torch.all(torch.abs(pc.params[256:1280].reshape(1, 4, 16, 16).sum(dim = 2).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(pc.params[1280:2304].reshape(1, 4, 16, 16).sum(dim = 2).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(pc.params[2304:2816].reshape(1, 2, 16, 16).sum(dim = 2).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(pc.params[2816:3328].reshape(1, 2, 16, 16).sum(dim = 2).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(pc.params[3328:3840].reshape(1, 2, 16, 16).sum(dim = 2).sum(dim = 1) - 1.0) < 1e-4)
    assert torch.all(torch.abs(pc.params[3840:4352].reshape(1, 2, 16, 16).sum(dim = 2).sum(dim = 1) - 1.0) < 1e-4)

    prod_layer1 = pc.inner_layer_groups[2][0]

    assert torch.all(prod_layer1.partitioned_nids[0] == torch.arange(16, 112, 16))

    assert torch.all(prod_layer1.partitioned_cids[0][0,:] == torch.tensor([144, 80, 112, 0]))
    assert torch.all(prod_layer1.partitioned_cids[0][1,:] == torch.tensor([160, 96, 128, 0]))
    assert torch.all(prod_layer1.partitioned_cids[0][2,:] == torch.tensor([176, 16, 48, 0]))
    assert torch.all(prod_layer1.partitioned_cids[0][3,:] == torch.tensor([192, 32, 64, 0]))
    assert torch.all(prod_layer1.partitioned_cids[0][4,:] == torch.tensor([208, 16, 112, 0]))
    assert torch.all(prod_layer1.partitioned_cids[0][5,:] == torch.tensor([224, 32, 128, 0]))

    assert torch.all(prod_layer1.partitioned_u_cids[0] == torch.tensor([48, 64, 80, 96, 144, 160, 176, 192, 208, 224]))
    assert torch.all(prod_layer1.partitioned_u_cids[1] == torch.tensor([16, 32, 112, 128]))

    assert torch.all(prod_layer1.partitioned_parids[0][0:2,:] == torch.tensor([[48], [64]]))
    assert torch.all(prod_layer1.partitioned_parids[0][2:4,:] == torch.tensor([[16], [32]]))
    assert torch.all(prod_layer1.partitioned_parids[0][4:6,:] == torch.tensor([[16], [32]]))
    assert torch.all(prod_layer1.partitioned_parids[0][6:8,:] == torch.tensor([[48], [64]]))
    assert torch.all(prod_layer1.partitioned_parids[0][8:10,:] == torch.tensor([[80], [96]]))
    assert torch.all(prod_layer1.partitioned_parids[1][0:2,:] == torch.tensor([[48, 80], [64, 96]]))
    assert torch.all(prod_layer1.partitioned_parids[1][2:4,:] == torch.tensor([[16, 80], [32, 96]]))

    sum_layer1 = pc.inner_layer_groups[3][0]

    assert sum_layer1.group_size == 1

    assert torch.all(sum_layer1.partitioned_nids[0] == torch.tensor([240]))

    assert torch.all(sum_layer1.partitioned_cids[0][0,:96] == torch.arange(16, 112))
    assert torch.all(sum_layer1.partitioned_cids[0][0,96:] == 0)

    assert torch.all(sum_layer1.partitioned_pids[0][0,:96] == torch.arange(4352, 4448))
    assert torch.all(sum_layer1.partitioned_pids[0][0,96:] == 0)

    assert torch.all(sum_layer1.partitioned_chids[0] == torch.arange(16, 112, 16))

    assert torch.all(sum_layer1.partitioned_parids[0] == 240)

    assert torch.all(sum_layer1.partitioned_parpids[0] == torch.arange(4352, 4448, 16)[:,None])

    assert torch.abs(pc.params[4352:4448].sum() - 1.0) < 1e-4

    ## Forward pass ##

    pc.to(device)

    data = torch.randint(0, 4, [512, 4], device = device)

    lls = pc(data)

    node_mars = pc.node_mars.cpu()
    data_cpu = data.cpu()

    sid, eid = ni0._output_ind_range
    ni0_lls = node_mars[sid:eid,:]
    assert torch.all(torch.abs(ni0_lls - ni0._params.reshape(-1, 4)[:,data_cpu[:,0]].log()) < 1e-4)

    sid, eid = ni1._output_ind_range
    ni1_lls = node_mars[sid:eid,:]
    assert torch.all(torch.abs(ni1_lls - ni1._params.reshape(-1, 4)[:,data_cpu[:,1]].log()) < 1e-4)

    sid, eid = ni2._output_ind_range
    ni2_lls = node_mars[sid:eid,:]
    assert torch.all(torch.abs(ni2_lls - ni2._params.reshape(-1, 6)[:,data_cpu[:,2]].log()) < 1e-4)

    sid, eid = ni3._output_ind_range
    ni3_lls = node_mars[sid:eid,:]
    assert torch.all(torch.abs(ni3_lls - ni3._params.reshape(-1, 6)[:,data_cpu[:,3]].log()) < 1e-4)

    np0_lls = ni0_lls + ni1_lls
    np1_lls = ni2_lls + ni3_lls
    np2_lls = ni1_lls + ni2_lls
    np3_lls = ni0_lls + ni1_lls

    pc.inner_layer_groups[0][0].forward(pc.node_mars, pc.element_mars)
    element_mars = pc.element_mars.cpu()

    sid, eid = np0._output_ind_range
    assert torch.all(torch.abs(np0_lls - element_mars[sid:eid,:]) < 1e-4)

    sid, eid = np1._output_ind_range
    assert torch.all(torch.abs(np1_lls - element_mars[sid:eid,:]) < 1e-4)

    sid, eid = np2._output_ind_range
    assert torch.all(torch.abs(np2_lls - element_mars[sid:eid,:]) < 1e-4)

    sid, eid = np3._output_ind_range
    assert torch.all(torch.abs(np3_lls - element_mars[sid:eid,:]) < 1e-4)

    ch_lls = torch.cat((np0_lls, np3_lls), dim = 0)
    epars = ns0._params.reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)
    ns0_lls = torch.matmul(epars, ch_lls.exp()).log()
    sid, eid = ns0._output_ind_range
    assert torch.all(torch.abs(ns0_lls - node_mars[sid:eid,:]) < 1e-3)
    
    epars = ns1._params.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    ns1_lls = torch.matmul(epars, np1_lls.exp()).log()
    sid, eid = ns1._output_ind_range
    assert torch.all(torch.abs(ns1_lls - node_mars[sid:eid,:]) < 1e-3)

    epars = ns2._params.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    ns2_lls = torch.matmul(epars, np2_lls.exp()).log()
    sid, eid = ns2._output_ind_range
    assert torch.all(torch.abs(ns2_lls - node_mars[sid:eid,:]) < 1e-3)

    np4_lls = ns0_lls + ni2_lls + ni3_lls
    np5_lls = ns1_lls + ni0_lls + ni1_lls
    np6_lls = ns2_lls + ni0_lls + ni3_lls

    pc.inner_layer_groups[2][0].forward(pc.node_mars, pc.element_mars)
    element_mars = pc.element_mars.cpu()

    sid, eid = np4._output_ind_range
    assert torch.all(torch.abs(np4_lls - element_mars[sid:eid,:]) < 1e-3)

    sid, eid = np5._output_ind_range
    assert torch.all(torch.abs(np5_lls - element_mars[sid:eid,:]) < 1e-3)

    sid, eid = np6._output_ind_range
    assert torch.all(torch.abs(np6_lls - element_mars[sid:eid,:]) < 1e-3)

    ch_lls = torch.cat((np4_lls, np5_lls, np6_lls), dim = 0)
    epars = ns._params.reshape(1, 6, 1, 16).permute(0, 2, 1, 3).reshape(1, 96)
    ns_lls = torch.matmul(epars, ch_lls.exp()).log()
    sid, eid = ns._output_ind_range
    assert torch.all(torch.abs(ns_lls - node_mars[sid:eid,:]) < 1e-3)

    ## Backward pass ##

    pc.backward(data.permute(1, 0), allow_modify_flows = False)

    node_flows = pc.node_flows.cpu()
    param_flows = pc.param_flows.cpu()

    sid, eid = ns._output_ind_range
    assert torch.all(torch.abs(node_flows[sid:eid,:] - 1.0) < 1e-4)

    pc.inner_layer_groups[2][0].forward(pc.node_mars, pc.element_mars)
    pc.inner_layer_groups[3][0].backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params, pc.param_flows)
    element_flows = pc.element_flows.cpu()

    ch_lls = torch.cat((np4_lls, np5_lls, np6_lls), dim = 0)
    epars = ns._params.reshape(1, 6, 1, 16).permute(0, 2, 1, 3).reshape(96, 1)
    eflows = epars * (ch_lls - ns_lls).exp()

    sid, eid = np4._output_ind_range
    np4_flows = eflows[0:32,:]
    assert torch.all(torch.abs(np4_flows - element_flows[sid:eid,:]) < 1e-4)

    sid, eid = np5._output_ind_range
    np5_flows = eflows[32:64,:]
    assert torch.all(torch.abs(np5_flows - element_flows[sid:eid,:]) < 1e-4)

    sid, eid = np6._output_ind_range
    np6_flows = eflows[64:96,:]
    assert torch.all(torch.abs(np6_flows - element_flows[sid:eid,:]) < 1e-4)

    ns_parflows = eflows.sum(dim = 1)
    ref_parflows = param_flows[4096:4192]
    assert torch.all(torch.abs(ns_parflows - ref_parflows) < 1e-3)

    sid, eid = ns0._output_ind_range
    ns0_flows = np4_flows
    assert torch.all(torch.abs(ns0_flows - node_flows[sid:eid,:]) < 1e-4)

    sid, eid = ns1._output_ind_range
    ns1_flows = np5_flows
    assert torch.all(torch.abs(ns1_flows - node_flows[sid:eid,:]) < 1e-4)

    sid, eid = ns2._output_ind_range
    ns2_flows = np6_flows
    assert torch.all(torch.abs(ns2_flows - node_flows[sid:eid,:]) < 1e-4)

    pc.inner_layer_groups[0][0].forward(pc.node_mars, pc.element_mars)
    pc.inner_layer_groups[1][0].backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params, pc.param_flows)
    element_flows = pc.element_flows.cpu()

    ch_lls = torch.cat((np0_lls, np3_lls), dim = 0)
    epars = ns0._params.reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)
    log_n_fdm = ns0_flows.log() - ns0_lls
    log_n_fdm_max = log_n_fdm.max(dim = 0).values
    n_fdm_sub = (log_n_fdm - log_n_fdm_max[None,:]).exp()
    emars = (ch_lls + log_n_fdm_max[None,:]).exp()
    eflows = emars * torch.matmul(epars.permute(1, 0), n_fdm_sub)

    sid, eid = np0._output_ind_range
    np0_flows = eflows[0:32,:]
    assert torch.all(torch.abs(np0_flows - element_flows[sid:eid,:]) < 1e-4)

    sid, eid = np3._output_ind_range
    np3_flows = eflows[32:64,:]
    assert torch.all(torch.abs(np3_flows - element_flows[sid:eid,:]) < 1e-4)

    ns0_parflows = epars * torch.matmul(n_fdm_sub, emars.permute(1, 0))
    ref_parflows = param_flows[0:2048].reshape(2, 64, 16).permute(0, 2, 1).reshape(32, 64)
    assert torch.all(torch.abs(ns0_parflows - ref_parflows) < 1e-3)

    ch_lls = np1_lls
    epars = ns1._params.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    log_n_fdm = ns1_flows.log() - ns1_lls
    log_n_fdm_max = log_n_fdm.max(dim = 0).values
    n_fdm_sub = (log_n_fdm - log_n_fdm_max[None,:]).exp()
    emars = (ch_lls + log_n_fdm_max[None,:]).exp()
    eflows = emars * torch.matmul(epars.permute(1, 0), n_fdm_sub)

    sid, eid = np1._output_ind_range
    np1_flows = eflows
    assert torch.all(torch.abs(np1_flows - element_flows[sid:eid,:]) < 1e-4)

    ns1_parflows = epars * torch.matmul(n_fdm_sub, emars.permute(1, 0))
    ref_parflows = param_flows[2048:3072].reshape(2, 32, 16).permute(0, 2, 1).reshape(32, 32)
    assert torch.all(torch.abs(ns1_parflows - ref_parflows) < 1e-3)

    ch_lls = np2_lls
    epars = ns2._params.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    log_n_fdm = ns2_flows.log() - ns2_lls
    log_n_fdm_max = log_n_fdm.max(dim = 0).values
    n_fdm_sub = (log_n_fdm - log_n_fdm_max[None,:]).exp()
    emars = (ch_lls + log_n_fdm_max[None,:]).exp()
    eflows = emars * torch.matmul(epars.permute(1, 0), n_fdm_sub)

    sid, eid = np2._output_ind_range
    np2_flows = eflows
    assert torch.all(torch.abs(np2_flows - element_flows[sid:eid,:]) < 1e-4)

    ns2_parflows = epars * torch.matmul(n_fdm_sub, emars.permute(1, 0))
    ref_parflows = param_flows[3072:4096].reshape(2, 32, 16).permute(0, 2, 1).reshape(32, 32)
    assert torch.all(torch.abs(ns2_parflows - ref_parflows) < 1e-3)

    sid, eid = ni0._output_ind_range
    ni0_flows = np0_flows + np3_flows + np5_flows + np6_flows
    assert torch.all(torch.abs(ni0_flows - node_flows[sid:eid,:]) < 2e-4)

    sid, eid = ni1._output_ind_range
    ni1_flows = np0_flows + np2_flows + np3_flows + np5_flows
    assert torch.all(torch.abs(ni1_flows - node_flows[sid:eid,:]) < 2e-4)

    sid, eid = ni2._output_ind_range
    ni2_flows = np1_flows + np2_flows + np4_flows
    assert torch.all(torch.abs(ni2_flows - node_flows[sid:eid,:]) < 2e-4)

    sid, eid = ni3._output_ind_range
    ni3_flows = np1_flows + np4_flows + np6_flows
    assert torch.all(torch.abs(ni3_flows - node_flows[sid:eid,:]) < 2e-4)

    input_layer = pc.input_layer_group[0]
    input_pflows = input_layer.param_flows.cpu()
    data_cpu = data.cpu()

    ni0_pflows = input_pflows[0:128].reshape(32, 4)
    ref_pflows = torch.zeros_like(ni0_pflows)
    for b in range(512):
        ref_pflows[:,data_cpu[b,0]] += ni0_flows[:,b]
    assert torch.all(torch.abs(ni0_pflows - ref_pflows) < 6e-3)

    ni1_pflows = input_pflows[128:256].reshape(32, 4)
    ref_pflows = torch.zeros_like(ni1_pflows)
    for b in range(512):
        ref_pflows[:,data_cpu[b,1]] += ni1_flows[:,b]
    assert torch.all(torch.abs(ni1_pflows - ref_pflows) < 6e-3)

    ni2_pflows = input_pflows[256:448].reshape(32, 6)
    ref_pflows = torch.zeros_like(ni2_pflows)
    for b in range(512):
        ref_pflows[:,data_cpu[b,2]] += ni2_flows[:,b]
    assert torch.all(torch.abs(ni2_pflows - ref_pflows) < 6e-3)

    ni3_pflows = input_pflows[448:640].reshape(32, 6)
    ref_pflows = torch.zeros_like(ni3_pflows)
    for b in range(512):
        ref_pflows[:,data_cpu[b,3]] += ni3_flows[:,b]
    assert torch.all(torch.abs(ni3_pflows - ref_pflows) < 6e-3)

    ## EM Optimization tests ##

    pc.backward(data.permute(1, 0), flows_memory = 0.0)

    ns0_old_params = ns0._params.clone().reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)
    ns1_old_params = ns1._params.clone().reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    ns2_old_params = ns2._params.clone().reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)

    ns_old_params = ns._params.clone().reshape(96)

    pc.update_param_flows()

    ref_parflows = ns0._param_flows.reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)
    assert torch.all(torch.abs(ns0_parflows - ref_parflows) < 1e-3)

    ref_parflows = ns1._param_flows.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    assert torch.all(torch.abs(ns1_parflows - ref_parflows) < 1e-3)

    ref_parflows = ns2._param_flows.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    assert torch.all(torch.abs(ns2_parflows - ref_parflows) < 1e-3)

    par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata = pc.par_update_kwargs

    if metadata["BLOCK_SIZE"] == 32:
        par_start_ids = par_start_ids.cpu()
        assert torch.all(par_start_ids[0:16] == torch.arange(256, 272))
        assert torch.all(par_start_ids[16:32] == torch.arange(768, 784))
        assert torch.all(par_start_ids[32:48] == torch.arange(1280, 1296))
        assert torch.all(par_start_ids[48:64] == torch.arange(1792, 1808))
        assert torch.all(par_start_ids[64:80] == torch.arange(2304, 2320))
        assert torch.all(par_start_ids[80:96] == torch.arange(2816, 2832))
        assert torch.all(par_start_ids[96:112] == torch.arange(3328, 3344))
        assert torch.all(par_start_ids[112:128] == torch.arange(3840, 3856))
        assert torch.all(par_start_ids[128:131] == torch.tensor([4352, 4384, 4416]))

        pflow_start_ids = pflow_start_ids.cpu()
        assert torch.all(par_start_ids - pflow_start_ids == 256)

        blk_sizes = blk_sizes.cpu()
        assert torch.all(blk_sizes[0:128] == 32)
        assert torch.all(blk_sizes[128:131] == 32)

        blk_intervals = blk_intervals.cpu()
        assert torch.all(blk_intervals[0:128] == 16)
        assert torch.all(blk_intervals[128:131] == 1)

        global_nids = global_nids.cpu()
        assert torch.all(global_nids[0:16] == torch.arange(0, 16))
        assert torch.all(global_nids[16:32] == torch.arange(0, 16))
        assert torch.all(global_nids[32:48] == torch.arange(16, 32))
        assert torch.all(global_nids[48:64] == torch.arange(16, 32))
        assert torch.all(global_nids[64:80] == torch.arange(32, 48))
        assert torch.all(global_nids[80:96] == torch.arange(48, 64))
        assert torch.all(global_nids[96:112] == torch.arange(64, 80))
        assert torch.all(global_nids[112:128] == torch.arange(80, 96))
        assert torch.all(global_nids[128:131] == 96)

        nchs = nchs.cpu()
        assert torch.all(nchs[0:32] == 64)
        assert torch.all(nchs[32:64] == 64)
        assert torch.all(nchs[64:128] == 32)
        assert torch.all(nchs[128:131] == 96)

        assert cum_pflows.size(0) == 97

    step_size = 0.25
    pseudocount = 0.01

    pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    cum_pflows = pc.par_update_kwargs[6].cpu()

    ns0_parflows = ns0._param_flows.reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)
    ns1_parflows = ns1._param_flows.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    ns2_parflows = ns2._param_flows.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)

    assert torch.all(torch.abs(ns0_parflows.sum(dim = 1) - cum_pflows[0:32]) < 1e-3)
    assert torch.all(torch.abs(ns1_parflows.sum(dim = 1) - cum_pflows[32:64]) < 1e-3)
    assert torch.all(torch.abs(ns2_parflows.sum(dim = 1) - cum_pflows[64:96]) < 1e-3)
    assert torch.abs(ns_parflows.sum() - cum_pflows[96]) < 1e-3

    ns0_new_params = (ns0_parflows + pseudocount / 64) / (ns0_parflows.sum(dim = 1, keepdim = True) + pseudocount)
    ns1_new_params = (ns1_parflows + pseudocount / 32) / (ns1_parflows.sum(dim = 1, keepdim = True) + pseudocount)
    ns2_new_params = (ns2_parflows + pseudocount / 32) / (ns2_parflows.sum(dim = 1, keepdim = True) + pseudocount)
    ns_new_params = (ns_parflows + pseudocount / 96) / (ns_parflows.sum() + pseudocount)

    ns0_updated_params = (1.0 - step_size) * ns0_old_params + step_size * ns0_new_params
    ns1_updated_params = (1.0 - step_size) * ns1_old_params + step_size * ns1_new_params
    ns2_updated_params = (1.0 - step_size) * ns2_old_params + step_size * ns2_new_params
    ns_updated_params = (1.0 - step_size) * ns_old_params + step_size * ns_new_params

    pc.update_parameters()

    ref_params = ns0._params.clone().reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)
    assert torch.all(torch.abs(ns0_updated_params - ref_params) < 1e-4)

    ref_params = ns1._params.clone().reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    assert torch.all(torch.abs(ns1_updated_params - ref_params) < 1e-4)

    ref_params = ns2._params.clone().reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    assert torch.all(torch.abs(ns2_updated_params - ref_params) < 1e-4)

    ref_params = ns._params.clone().reshape(96)
    assert torch.all(torch.abs(ns_updated_params - ref_params) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(23892)
    simple_model_test()
