import pyjuice as juice
import torch
import math
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_structured_blk_sparse_pc():

    torch.manual_seed(2893)

    device = torch.device("cuda:0")

    block_size = 16
    num_node_blocks = 8
    num_nodes = block_size * num_node_blocks

    mask0 = torch.zeros([num_node_blocks, num_node_blocks], dtype = torch.long)
    mask1 = torch.zeros([num_node_blocks, num_node_blocks], dtype = torch.long)

    mask0[0, [0, 1, 2, 3]] = 1
    mask1[0, [0, 1, 2, 3]] = 1

    mask0[1, [2, 3, 4, 5]] = 1
    mask1[1, [2, 3, 4, 5]] = 1

    mask0[2, [1, 2, 5, 6]] = 1
    mask1[2, [1, 2, 5, 6]] = 1

    mask0[3, [0, 1, 6, 7]] = 1
    mask1[3, [0, 1, 6, 7]] = 1

    mask0[4, [0, 2, 5, 7]] = 1
    mask1[4, [0, 2, 5, 7]] = 1

    mask0[5, [4, 5, 6, 7]] = 1
    mask1[5, [4, 5, 6, 7]] = 1

    mask0[6, [0, 1, 6, 7]] = 1
    mask1[6, [3, 4]] = 1

    mask0[7, [3, 4]] = 1
    mask1[7, [0, 1, 6, 7]] = 1

    input_mask0 = mask0[:,None,:,None].repeat(1, block_size, 1, 256 // num_node_blocks).reshape(num_nodes, 256)
    input_mask1 = mask1[:,None,:,None].repeat(1, block_size, 1, 256 // num_node_blocks).reshape(num_nodes, 256)

    sum_edges0 = torch.nonzero(mask0, as_tuple = False).permute(1, 0)
    sum_edges1 = torch.nonzero(mask1, as_tuple = False).permute(1, 0)
    
    with juice.set_block_size(block_size = block_size):

        ni0 = inputs(0, num_node_blocks = num_node_blocks, dist = dists.MaskedCategorical(num_cats = 256, mask_mode = "full_mask"), mask = input_mask0)
        ni1 = inputs(1, num_node_blocks = num_node_blocks, dist = dists.MaskedCategorical(num_cats = 256, mask_mode = "full_mask"), mask = input_mask1)

        rearranged_nids = torch.arange(0, num_nodes).reshape(num_node_blocks, block_size).permute(1, 0).reshape(-1)
        edge_ids = rearranged_nids[:,None].repeat(1, 2)
        np01 = multiply(ni0, ni1, edge_ids = edge_ids, sparse_edges = True)

        ns01 = summate(np01, edge_ids = sum_edges0)

        ni2 = inputs(2, num_node_blocks = num_node_blocks, dist = dists.MaskedCategorical(num_cats = 256, mask_mode = "full_mask"), mask = input_mask1)

        rearranged_nids = torch.arange(0, num_nodes).reshape(num_node_blocks, block_size).permute(1, 0).reshape(-1)
        edge_ids = rearranged_nids[:,None].repeat(1, 2)
        np012 = multiply(ns01, ni2, edge_ids = edge_ids, sparse_edges = True)

        ns012 = summate(np012, num_node_blocks = 1, block_size = 1)

    pc = TensorCircuit(ns012)

    ## Compilation tests ##

    ni0_params = ni0._params.reshape(num_nodes, 2 * 256 + 1)
    assert torch.all(ni0_params[:,256:512][::block_size,::(256 // num_node_blocks)].long() == mask0)
    assert torch.all(pc.input_layer_group[0].vids.reshape(3, num_nodes)[0,:] == 0)
    assert torch.all(pc.input_layer_group[0].vids.reshape(3, num_nodes)[1,:] == 1)
    assert torch.all(pc.input_layer_group[0].vids.reshape(3, num_nodes)[2,:] == 2)
    assert torch.all(pc.input_layer_group[0].s_pids == torch.arange(0, 65664 * 3, 513))
    assert torch.all(pc.input_layer_group[0].s_pfids == torch.arange(0, 32768 * 3, 256))
    assert torch.all(pc.input_layer_group[0].s_mids.reshape(3, num_nodes)[0,:] == 0)
    assert torch.all(pc.input_layer_group[0].s_mids.reshape(3, num_nodes)[1,:] == 1)
    assert torch.all(pc.input_layer_group[0].s_mids.reshape(3, num_nodes)[2,:] == 2)

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_nids[0] == torch.arange(16, 144))
    assert ni0._output_ind_range[0] == 16
    assert ni0._output_ind_range[1] == 144
    assert ni1._output_ind_range[0] == 144
    assert ni1._output_ind_range[1] == 272

    cids = torch.tensor([[16, 144], [32, 160], [48, 176], [64, 192], [80, 208], [96, 224], [112, 240], [128, 256]])
    for i in range(16):
        assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][i*8:(i+1)*8,:] == cids + i)

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_u_cids[0] == torch.arange(16, 272))

    parids = torch.arange(16, 144, 8).unsqueeze(1)
    for i in range(8):
        assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][i*16:(i+1)*16,:] == parids + i)

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_nids[0] == torch.arange(400, 528, 16))
    assert torch.all((pc.inner_layer_groups[1][0].partitioned_cids[0][:,::16] // 16).reshape(-1)[:-2] == sum_edges0[1,:] + 1)
    assert torch.all((pc.inner_layer_groups[1][0].partitioned_cids[0][:,::16] // 16).reshape(-1)[-2:] == 0)
    cids = pc.inner_layer_groups[1][0].partitioned_cids[0].reshape(8, 4, 16)
    assert torch.all((cids[:,:,1:] - cids[:,:,:-1])[:-1,:,:] == 1)
    assert torch.all((cids[:,:,1:] - cids[:,:,:-1])[-1,:2,:] == 1)
    assert torch.all((cids[:,:,1:] - cids[:,:,:-1])[-1,2:,:] == 0)

    ## Forward tests ##

    pc.to(device)

    data = torch.randint(0, 256, (512, 3)).to(device)
    data_cpu = data.cpu()

    lls = pc(data, force_use_fp32 = True)

    node_mars = pc.node_mars.clone().cpu()

    ni0_params = ni0._params.reshape(num_nodes, 2 * 256 + 1)[:,:256]
    ni0_lls = ni0_params[:,data_cpu[:,0]].log()
    ni0_lls = torch.where(ni0_lls == -float("inf"), math.log(1e-10), ni0_lls)
    assert torch.all(torch.abs(ni0_lls.exp() - node_mars[16:144,:].exp()) < 1e-4)

    ni1_params = ni1._params.reshape(num_nodes, 2 * 256 + 1)[:,:256]
    ni1_lls = ni1_params[:,data_cpu[:,1]].log()
    ni1_lls = torch.where(ni1_lls == -float("inf"), math.log(1e-10), ni1_lls)
    assert torch.all(torch.abs(ni1_lls.exp() - node_mars[144:272,:].exp()) < 1e-4)

    ni2_params = ni2._params.reshape(num_nodes, 2 * 256 + 1)[:,:256]
    ni2_lls = ni2_params[:,data_cpu[:,2]].log()
    ni2_lls = torch.where(ni2_lls == -float("inf"), math.log(1e-10), ni2_lls)
    assert torch.all(torch.abs(ni2_lls.exp() - node_mars[272:400,:].exp()) < 1e-4)

    pc.inner_layer_groups[0][0].forward(pc.node_mars, pc.element_mars)
    element_mars = pc.element_mars.clone().cpu()

    np01_lls = (ni0_lls + ni1_lls).reshape(8, 16, 512).permute(1, 0, 2).reshape(128, 512)
    assert torch.all(torch.abs(np01_lls - element_mars[16:144,:]) < 1e-4)
    np01_vals = np01_lls.reshape(8, 16, 512).exp()

    ns01_vals = torch.zeros([8, 16, 512])
    for i in range(ns01.edge_ids.size(1)):
        ngid = ns01.edge_ids[0,i]
        cgid = ns01.edge_ids[1,i]
        
        ns01_vals[ngid,:,:] += torch.matmul(ns01._params[i,:,:], np01_vals[cgid,:,:])

    ns01_lls = ns01_vals.reshape(128, 512).log()

    assert torch.all(torch.abs(ns01_lls - node_mars[400:528,:]) < 2e-3)

    pc.inner_layer_groups[2][0].forward(pc.node_mars, pc.element_mars)
    element_mars = pc.element_mars.clone().cpu()

    np012_lls = (ns01_lls + ni2_lls).reshape(8, 16, 512).permute(1, 0, 2).reshape(128, 512)
    assert torch.all(torch.abs(np012_lls - element_mars[16:144,:]) < 2e-3)

    ns012_lls = torch.logsumexp(ns012._params.reshape(-1, 1).log() + np012_lls, dim = 0)
    assert torch.all(torch.abs(ns012_lls - node_mars[528,:]) < 2e-3)

    ## Backward tests ##

    pc.backward(data, allow_modify_flows = False)

    node_flows = pc.node_flows.clone().cpu()
    param_flows = pc.param_flows.clone().cpu()

    assert torch.all(torch.abs(node_flows[528,:] - 1.0) < 1e-4)

    pc.inner_layer_groups[2].forward(pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[3].backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params, 
                                      param_flows = pc.param_flows, allow_modify_flows = False)
    element_flows = pc.element_flows.clone().cpu()

    np012_flows = (ns012._params.reshape(-1, 1).log() + np012_lls - ns012_lls).exp()
    assert torch.all(torch.abs(np012_flows - element_flows[16:144,:]) < 1e-4)

    ns01_flows = np012_flows.reshape(16, 8, 512).permute(1, 0, 2).reshape(128, 512)
    ni2_flows = np012_flows.reshape(16, 8, 512).permute(1, 0, 2).reshape(128, 512)

    assert torch.all(torch.abs(ns01_flows - node_flows[400:528,:]) < 1e-4)
    assert torch.all(torch.abs(ni2_flows - node_flows[272:400,:]) < 1e-4)

    pc.inner_layer_groups[0].forward(pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[1].backward(pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, pc.params, 
                                      param_flows = pc.param_flows, allow_modify_flows = False)
    element_flows = pc.element_flows.clone().cpu()

    ns01_flows = ns01_flows.reshape(8, 16, 512)
    ref_ns01_pflows = param_flows[0:7680].reshape(ns01.edge_ids.size(1), 16, 16).permute(0, 2, 1)

    np01_flows = torch.zeros([8, 16, 512])
    for i in range(ns01.edge_ids.size(1)):
        ngid = ns01.edge_ids[0,i]
        cgid = ns01.edge_ids[1,i]
        
        np01_flows[cgid,:,:] += torch.matmul(ns01._params[i,:,:].permute(1, 0), ns01_flows[ngid,:,:] / ns01_vals[ngid,:,:]) * np01_vals[cgid,:,:]

        ns01_pflows = torch.matmul(ns01_flows[ngid,:,:] / ns01_vals[ngid,:,:], np01_vals[cgid,:,:].permute(1, 0)) * ns01._params[i,:,:]
        assert torch.all(torch.abs(ns01_pflows - ref_ns01_pflows[i,:,:]) < 1e-2)

    np01_flows = np01_flows.reshape(128, 512)

    assert torch.all(torch.abs(np01_flows - element_flows[16:144,:]) < 1e-3)

    ni0_flows = np01_flows.reshape(16, 8, 512).permute(1, 0, 2).reshape(128, 512)
    ni1_flows = np01_flows.reshape(16, 8, 512).permute(1, 0, 2).reshape(128, 512)

    assert torch.all(torch.abs(ni0_flows - node_flows[16:144,:]) < 1e-3)
    assert torch.all(torch.abs(ni1_flows - node_flows[144:272,:]) < 1e-3)


if __name__ == "__main__":
    torch.manual_seed(89172)
    test_structured_blk_sparse_pc()
