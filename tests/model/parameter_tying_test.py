import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs, set_block_size
from pyjuice.model import TensorCircuit
from pyjuice.model.backend import compute_cum_par_flows, em_par_update

import pytest


def test_simple_structure_block1():

    block_size = 1
    
    with set_block_size(block_size = block_size):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))

        np01 = multiply(ni0, ni1)
        np12 = multiply(ni1, ni2)
        np23 = multiply(ni2, ni3)

        ns01 = summate(np01, num_node_blocks = 2)
        ns12 = ns01.duplicate(np12, tie_params = True)
        ns23 = ns01.duplicate(np23, tie_params = True)

        np012_0 = multiply(ns01, ni2)
        np012_1 = multiply(ns12, ni0)
        ns012 = summate(np012_0, np012_1, num_node_blocks = 2)

        np123_0 = multiply(ns12, ni3)
        np123_1 = multiply(ns23, ni1)
        ns123 = ns012.duplicate(np123_0, np123_1, tie_params = True)

        np0123_0 = multiply(ns012, ni3)
        np0123_1 = multiply(ns123, ni0)
        ns0123 = ns123.duplicate(np0123_0, np0123_1, tie_params = True)

    pc = TensorCircuit(ns0123, max_tied_ns_per_parflow_block = 2)

    device = torch.device("cuda:0")

    ## Compilation tests ##

    assert torch.all(pc.input_layer_group[0].vids == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).reshape(8, 1))
    assert torch.all(pc.input_layer_group[0].s_pids == torch.tensor([0, 5, 10, 15, 20, 25, 30, 35]))
    assert torch.all(pc.input_layer_group[0].s_pfids == torch.tensor([0, 5, 10, 15, 20, 25, 30, 35]))
    assert torch.all(pc.input_layer_group[0].metadata == torch.tensor([5.0, 5.0, 5.0, 5.0]))
    assert torch.all(pc.input_layer_group[0].s_mids == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    assert torch.all(pc.input_layer_group[0].source_nids == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
    assert pc.input_layer_group[0]._output_ind_range[0] == 1
    assert pc.input_layer_group[0]._output_ind_range[1] == 9

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_nids[0] == torch.tensor([1, 2, 3, 4, 5, 6]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][0,:] == torch.tensor([1, 3]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][1,:] == torch.tensor([2, 4]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][2,:] == torch.tensor([3, 5]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][3,:] == torch.tensor([4, 6]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][4,:] == torch.tensor([5, 7]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][5,:] == torch.tensor([6, 8]))

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_u_cids[0] == torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][0,:] == torch.tensor([1, 0]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][1,:] == torch.tensor([2, 0]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][2,:] == torch.tensor([1, 3]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][3,:] == torch.tensor([2, 4]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][4,:] == torch.tensor([3, 5]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][5,:] == torch.tensor([4, 6]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][6,:] == torch.tensor([5, 0]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][7,:] == torch.tensor([6, 0]))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_nids[0] == torch.arange(9, 15))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_cids[0][0:2,:] == torch.tensor([1, 2]).reshape(1, 2))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_cids[0][2:4,:] == torch.tensor([3, 4]).reshape(1, 2))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_cids[0][4:6,:] == torch.tensor([5, 6]).reshape(1, 2))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][0:2,:] == torch.tensor([[1, 2], [3, 4]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][2:4,:] == torch.tensor([[1, 2], [3, 4]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][4:6,:] == torch.tensor([[1, 2], [3, 4]]))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][0:2,:] == torch.tensor([[0, 1], [2, 3]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][2:4,:] == torch.tensor([[0, 1], [2, 3]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][4:6,:] == torch.tensor([[4, 5], [6, 7]]))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_chids[0] == torch.arange(1, 7))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parids[0][0:2,:] == torch.tensor([9, 10]).reshape(1, 2))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parids[0][2:4,:] == torch.tensor([11, 12]).reshape(1, 2))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parids[0][4:6,:] == torch.tensor([13, 14]).reshape(1, 2))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parpids[0][0:2,:] == torch.tensor([[1, 3], [2, 4]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parpids[0][2:4,:] == torch.tensor([[1, 3], [2, 4]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parpids[0][4:6,:] == torch.tensor([[1, 3], [2, 4]]))

    assert torch.all(pc.inner_layer_groups[2][0].partitioned_nids[0] == torch.arange(1, 9))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][0:2,:] == torch.tensor([[9, 5], [10, 6]]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][2:4,:] == torch.tensor([[11, 1], [12, 2]]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][4:6,:] == torch.tensor([[11, 7], [12, 8]]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][6:8,:] == torch.tensor([[13, 3], [14, 4]]))

    assert torch.all(pc.inner_layer_groups[2][0].partitioned_u_cids[0] == torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_u_cids[1] == torch.tensor([11, 12]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_parids[0] == torch.tensor([3, 4, 7, 8, 1, 2, 5, 6, 1, 2, 7, 8]).reshape(12, 1))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_parids[1] == torch.tensor([[3, 5], [4, 6]]))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_nids[0] == torch.arange(15, 19))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_cids[0][0:2,:] == torch.tensor([1, 2, 3, 4]).reshape(1, 4))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_cids[0][2:4,:] == torch.tensor([5, 6, 7, 8]).reshape(1, 4))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pids[0][0:2,:] == torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pids[0][2:4,:] == torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]]))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pfids[0][0:2,:] == torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pfids[0][2:4,:] == torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]]))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_chids[0] == torch.arange(1, 9))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parids[0][0:4,:] == torch.tensor([15, 16]).reshape(1, 2))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parids[0][4:8,:] == torch.tensor([17, 18]).reshape(1, 2))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parpids[0][0:4,:] == torch.tensor([[5, 9], [6, 10], [7, 11], [8, 12]]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parpids[0][4:8,:] == torch.tensor([[5, 9], [6, 10], [7, 11], [8, 12]]))

    assert torch.all(pc.inner_layer_groups[4][0].partitioned_nids[0] == torch.tensor([1, 2, 3, 4]))
    assert torch.all(pc.inner_layer_groups[4][0].partitioned_cids[0] == torch.tensor([[15, 7], [16, 8], [17, 1], [18, 2]]))

    assert torch.all(pc.inner_layer_groups[4][0].partitioned_u_cids[0] == torch.tensor([1, 2, 7, 8, 15, 16, 17, 18]))
    assert torch.all(pc.inner_layer_groups[4][0].partitioned_parids[0] == torch.tensor([3, 4, 1, 2, 1, 2, 3, 4]).reshape(8, 1))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_nids[0] == torch.arange(19, 21))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_cids[0] == torch.tensor([1, 2, 3, 4]).reshape(1, 4))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pids[0] == torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]]))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pfids[0] == torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]]))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_chids[0] == torch.arange(1, 5))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_parids[0] == torch.tensor([19, 20]).reshape(1, 2))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_parpids[0] == torch.tensor([[5, 9], [6, 10], [7, 11], [8, 12]]))

    pc.to(device)

    ## Forward tests ##

    data = torch.randint(0, 5, [16, 4]).to(device)

    lls = pc(data)

    node_mars = pc.node_mars.detach().cpu()
    params = pc.params.detach().cpu()

    params0 = params[1:5].reshape(2, 2)

    np01_lls = node_mars[1:3,:] + node_mars[3:5,:]
    ns01_lls = torch.matmul(params0, np01_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[9:11,:] - ns01_lls) < 1e-4)

    np12_lls = node_mars[3:5,:] + node_mars[5:7,:]
    ns12_lls = torch.matmul(params0, np12_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[11:13,:] - ns12_lls) < 1e-4)

    np23_lls = node_mars[5:7,:] + node_mars[7:9,:]
    ns23_lls = torch.matmul(params0, np23_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[13:15,:] - ns23_lls) < 1e-4)

    params1 = params[5:13].reshape(2, 4)

    np012_0_lls = ns01_lls + node_mars[5:7,:]
    np012_1_lls = ns12_lls + node_mars[1:3,:]
    np012_lls = torch.cat((np012_0_lls, np012_1_lls), dim = 0)
    ns012_lls = torch.matmul(params1, np012_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[15:17,:] - ns012_lls) < 1e-4)

    np123_0_lls = ns12_lls + node_mars[7:9,:]
    np123_1_lls = ns23_lls + node_mars[3:5,:]
    np123_lls = torch.cat((np123_0_lls, np123_1_lls), dim = 0)
    ns123_lls = torch.matmul(params1, np123_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[17:19,:] - ns123_lls) < 1e-4)

    np0123_0_lls = ns012_lls + node_mars[7:9,:]
    np0123_1_lls = ns123_lls + node_mars[1:3,:]
    np0123_lls = torch.cat((np0123_0_lls, np0123_1_lls), dim = 0)
    ns0123_lls = torch.matmul(params1, np0123_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[19:21,:] - ns0123_lls) < 1e-4)

    ## Backward tests ##

    pc.backward(data.permute(1, 0), allow_modify_flows = False)

    node_flows = pc.node_flows.detach().cpu().clone()
    param_flows = pc.param_flows.detach().cpu().clone()

    assert torch.all(torch.abs(node_flows[19:21,:] - 1.0) < 1e-4)

    pc.inner_layer_groups[4][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[5][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np0123_flows = torch.matmul(params1.permute(1, 0), 1.0 / ns0123_lls.exp()) * np0123_lls.exp()
    assert torch.all(torch.abs(element_flows[1:5,:] - np0123_flows) < 1e-4)

    param_flows1 = torch.matmul(1.0 / ns0123_lls.exp(), np0123_lls.exp().permute(1, 0)) * params1

    ns012_flows = element_flows[1:3,:]
    assert torch.all(torch.abs(node_flows[15:17,:] - ns012_flows) < 1e-4)

    ns123_flows = element_flows[3:5,:]
    assert torch.all(torch.abs(node_flows[17:19,:] - ns123_flows) < 1e-4)

    ni0_flows = element_flows[3:5,:].clone()
    ni3_flows = element_flows[1:3,:].clone()

    pc.inner_layer_groups[2][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[3][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np012_flows = torch.matmul(params1.permute(1, 0), ns012_flows / ns012_lls.exp()) * np012_lls.exp()
    assert torch.all(torch.abs(element_flows[1:5,:] - np012_flows) < 1e-4)

    param_flows1 += torch.matmul(ns012_flows / ns012_lls.exp(), np012_lls.exp().permute(1, 0)) * params1

    np123_flows = torch.matmul(params1.permute(1, 0), ns123_flows / ns123_lls.exp()) * np123_lls.exp()
    assert torch.all(torch.abs(element_flows[5:9,:] - np123_flows) < 1e-4)

    param_flows1 += torch.matmul(ns123_flows / ns123_lls.exp(), np123_lls.exp().permute(1, 0)) * params1

    ns01_flows = np012_flows[0:2,:]
    assert torch.all(torch.abs(node_flows[9:11,:] - ns01_flows) < 1e-4)

    ns12_flows = np012_flows[2:4,:] + np123_flows[0:2,:]
    assert torch.all(torch.abs(node_flows[11:13,:] - ns12_flows) < 1e-4)

    ns23_flows = np123_flows[2:4,:]
    assert torch.all(torch.abs(node_flows[13:15,:] - ns23_flows) < 1e-4)

    ni2_flows = np012_flows[0:2,:].clone()
    ni0_flows += np012_flows[2:4,:].clone()
    ni3_flows += np123_flows[0:2,:].clone()
    ni1_flows = np123_flows[2:4,:].clone()

    pc.inner_layer_groups[0][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[1][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np01_flows = torch.matmul(params0.permute(1, 0), ns01_flows / ns01_lls.exp()) * np01_lls.exp()
    assert torch.all(torch.abs(element_flows[1:3,:] - np01_flows) < 1e-4)

    param_flows0 = torch.matmul(ns01_flows / ns01_lls.exp(), np01_lls.exp().permute(1, 0)) * params0

    np12_flows = torch.matmul(params0.permute(1, 0), ns12_flows / ns12_lls.exp()) * np12_lls.exp()
    assert torch.all(torch.abs(element_flows[3:5,:] - np12_flows) < 1e-4)

    param_flows0 += torch.matmul(ns12_flows / ns12_lls.exp(), np12_lls.exp().permute(1, 0)) * params0

    np23_flows = torch.matmul(params0.permute(1, 0), ns23_flows / ns23_lls.exp()) * np23_lls.exp()
    assert torch.all(torch.abs(element_flows[5:7,:] - np23_flows) < 1e-4)

    param_flows0 += torch.matmul(ns23_flows / ns23_lls.exp(), np23_lls.exp().permute(1, 0)) * params0

    ni0_flows += np01_flows.clone()
    ni1_flows += np01_flows.clone() + np12_flows.clone()
    ni2_flows += np12_flows.clone() + np23_flows.clone()
    ni3_flows += np23_flows.clone()

    assert torch.all(torch.abs(node_flows[1:3,:] - ni0_flows) < 1e-4)
    assert torch.all(torch.abs(node_flows[3:5,:] - ni1_flows) < 1e-4)
    assert torch.all(torch.abs(node_flows[5:7,:] - ni2_flows) < 1e-4)
    assert torch.all(torch.abs(node_flows[7:9,:] - ni3_flows) < 1e-4)

    assert torch.all(torch.abs(param_flows0.reshape(-1) - (param_flows[0:4] + param_flows[4:8])) < 1e-4)
    assert torch.all(torch.abs(param_flows1.reshape(-1) - param_flows[8:16]) < 1e-4)

    ## Parameter learning & flow aggregation tests ##

    temp_param_flows = param_flows.clone().to(device)

    compute_cum_par_flows(temp_param_flows, pc.parflow_fusing_kwargs)

    assert torch.all(torch.abs(param_flows0.reshape(-1) - temp_param_flows[0:4].cpu()) < 1e-4)
    assert torch.all(torch.abs(param_flows1.reshape(-1) - temp_param_flows[8:16].cpu()) < 1e-4)

    em_par_update(pc.params, temp_param_flows, pc.par_update_kwargs, step_size = 1.0, pseudocount = 0.0)

    param_flows0 /= param_flows0.sum(dim = 1, keepdim = True)
    assert torch.all(torch.abs(param_flows0.reshape(-1) - pc.params[1:5].cpu()) < 1e-4)

    param_flows1 /= param_flows1.sum(dim = 1, keepdim = True)
    assert torch.all(torch.abs(param_flows1.reshape(-1) - pc.params[5:13].cpu()) < 1e-4)


def test_simple_structure_block16():

    block_size = 16
    
    with set_block_size(block_size = block_size):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))
        ni1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))
        ni2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))
        ni3 = inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))

        np01 = multiply(ni0, ni1)
        np12 = multiply(ni1, ni2)
        np23 = multiply(ni2, ni3)

        ns01 = summate(np01, num_node_blocks = 2)
        ns12 = ns01.duplicate(np12, tie_params = True)
        ns23 = ns01.duplicate(np23, tie_params = True)

        np012_0 = multiply(ns01, ni2)
        np012_1 = multiply(ns12, ni0)
        ns012 = summate(np012_0, np012_1, num_node_blocks = 2)

        np123_0 = multiply(ns12, ni3)
        np123_1 = multiply(ns23, ni1)
        ns123 = ns012.duplicate(np123_0, np123_1, tie_params = True)

        np0123_0 = multiply(ns012, ni3)
        np0123_1 = multiply(ns123, ni0)
        ns0123 = ns123.duplicate(np0123_0, np0123_1, tie_params = True)

    ns0123.init_parameters()
    pc = TensorCircuit(ns0123, max_tied_ns_per_parflow_block = 2)

    device = torch.device("cuda:0")

    ## Compilation tests ##

    assert pc.input_layer_group[0]._output_ind_range[0] == 16
    assert pc.input_layer_group[0]._output_ind_range[1] == 144

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_nids[0] == torch.tensor([16, 32, 48, 64, 80, 96]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][0,:] == torch.tensor([16, 48]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][1,:] == torch.tensor([32, 64]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][2,:] == torch.tensor([48, 80]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][3,:] == torch.tensor([64, 96]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][4,:] == torch.tensor([80, 112]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0][5,:] == torch.tensor([96, 128]))

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_u_cids[0] == torch.tensor([16, 32, 48, 64, 80, 96, 112, 128]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][0,:] == torch.tensor([16, 0]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][1,:] == torch.tensor([32, 0]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][2,:] == torch.tensor([16, 48]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][3,:] == torch.tensor([32, 64]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][4,:] == torch.tensor([48, 80]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][5,:] == torch.tensor([64, 96]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][6,:] == torch.tensor([80, 0]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0][7,:] == torch.tensor([96, 0]))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_nids[0] == torch.arange(144, 240, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_cids[0][0:2,:] == torch.arange(16, 48).reshape(1, 32))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_cids[0][2:4,:] == torch.arange(48, 80).reshape(1, 32))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_cids[0][4:6,:] == torch.arange(80, 112).reshape(1, 32))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][0,:] == torch.arange(256, 768, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][1,:] == torch.arange(768, 1280, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][2,:] == torch.arange(256, 768, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][3,:] == torch.arange(768, 1280, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][4,:] == torch.arange(256, 768, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0][5,:] == torch.arange(768, 1280, 16))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][0,:] == torch.arange(0, 512, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][1,:] == torch.arange(512, 1024, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][2,:] == torch.arange(0, 512, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][3,:] == torch.arange(512, 1024, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][4,:] == torch.arange(1024, 1536, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0][5,:] == torch.arange(1536, 2048, 16))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_chids[0] == torch.arange(16, 112, 16))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parids[0][0:2,:] == torch.tensor([144, 160]).reshape(1, 2))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parids[0][2:4,:] == torch.tensor([176, 192]).reshape(1, 2))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parids[0][4:6,:] == torch.tensor([208, 224]).reshape(1, 2))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parpids[0][0:2,:] == torch.tensor([[256, 768], [512, 1024]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parpids[0][2:4,:] == torch.tensor([[256, 768], [512, 1024]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parpids[0][4:6,:] == torch.tensor([[256, 768], [512, 1024]]))

    assert torch.all(pc.inner_layer_groups[2][0].partitioned_nids[0] == torch.arange(16, 144, 16))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][0:2,:] == torch.tensor([[144, 80], [160, 96]]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][2:4,:] == torch.tensor([[176, 16], [192, 32]]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][4:6,:] == torch.tensor([[176, 112], [192, 128]]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0][6:8,:] == torch.tensor([[208, 48], [224, 64]]))

    assert torch.all(pc.inner_layer_groups[2][0].partitioned_u_cids[0] == torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]) * 16)
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_u_cids[1] == torch.tensor([11, 12]) * 16)
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_parids[0] == torch.tensor([3, 4, 7, 8, 1, 2, 5, 6, 1, 2, 7, 8]).reshape(12, 1) * 16)
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_parids[1] == torch.tensor([[3, 5], [4, 6]]) * 16)

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_nids[0] == torch.arange(15, 19) * 16)
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_cids[0][0:2,:] == torch.arange(16, 80).reshape(1, 64))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_cids[0][2:4,:] == torch.arange(80, 144).reshape(1, 64))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pids[0][0,:] == torch.arange(1280, 2304, 16))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pids[0][1,:] == torch.arange(2304, 3328, 16))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pids[0][2,:] == torch.arange(1280, 2304, 16))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pids[0][3,:] == torch.arange(2304, 3328, 16))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pfids[0][0,:] == torch.arange(2048, 3072, 16))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pfids[0][1,:] == torch.arange(3072, 4096, 16))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pfids[0][2,:] == torch.arange(2048, 3072, 16))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pfids[0][3,:] == torch.arange(3072, 4096, 16))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_chids[0] == torch.arange(1, 9) * 16)
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parids[0][0:4,:] == torch.tensor([15, 16]).reshape(1, 2) * 16)
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parids[0][4:8,:] == torch.tensor([17, 18]).reshape(1, 2) * 16)

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parpids[0][0:4,:] == torch.tensor([[5, 9], [6, 10], [7, 11], [8, 12]]) * 256)
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parpids[0][4:8,:] == torch.tensor([[5, 9], [6, 10], [7, 11], [8, 12]]) * 256)

    assert torch.all(pc.inner_layer_groups[4][0].partitioned_nids[0] == torch.tensor([1, 2, 3, 4]) * 16)
    assert torch.all(pc.inner_layer_groups[4][0].partitioned_cids[0] == torch.tensor([[15, 7], [16, 8], [17, 1], [18, 2]]) * 16)

    assert torch.all(pc.inner_layer_groups[4][0].partitioned_u_cids[0] == torch.tensor([1, 2, 7, 8, 15, 16, 17, 18]) * 16)
    assert torch.all(pc.inner_layer_groups[4][0].partitioned_parids[0] == torch.tensor([3, 4, 1, 2, 1, 2, 3, 4]).reshape(8, 1) * 16)

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_nids[0] == torch.arange(19, 21) * 16)
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_cids[0] == torch.arange(16, 80).reshape(1, 64))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pids[0][0,:] == torch.arange(1280, 2304, 16))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pids[0][1,:] == torch.arange(2304, 3328, 16))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pfids[0][0,:] == torch.arange(2048, 3072, 16))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pfids[0][1,:] == torch.arange(3072, 4096, 16))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_chids[0] == torch.arange(1, 5) * 16)
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_parids[0] == torch.tensor([19, 20]).reshape(1, 2) * 16)

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_parpids[0] == torch.tensor([[5, 9], [6, 10], [7, 11], [8, 12]]) * 256)

    pc.to(device)

    ## Forward tests ##

    data = torch.randint(0, 5, [16, 4]).to(device)

    lls = pc(data)

    node_mars = pc.node_mars.detach().cpu()
    params = pc.params.detach().cpu()

    params0 = ns01.get_source_ns()._params.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)

    np01_lls = node_mars[16:48,:] + node_mars[48:80,:]
    ns01_lls = torch.matmul(params0, np01_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[144:176,:] - ns01_lls) < 1e-3)

    np12_lls = node_mars[48:80,:] + node_mars[80:112,:]
    ns12_lls = torch.matmul(params0, np12_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[176:208,:] - ns12_lls) < 1e-3)

    np23_lls = node_mars[80:112,:] + node_mars[112:144,:]
    ns23_lls = torch.matmul(params0, np23_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[208:240,:] - ns23_lls) < 1e-3)

    params1 = ns0123.get_source_ns()._params.reshape(2, 4, 16, 16).permute(0, 2, 1, 3).reshape(32, 64)

    np012_0_lls = ns01_lls + node_mars[80:112,:]
    np012_1_lls = ns12_lls + node_mars[16:48,:]
    np012_lls = torch.cat((np012_0_lls, np012_1_lls), dim = 0)
    ns012_lls = torch.matmul(params1, np012_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[240:272,:] - ns012_lls) < 1e-3)

    np123_0_lls = ns12_lls + node_mars[112:144,:]
    np123_1_lls = ns23_lls + node_mars[48:80,:]
    np123_lls = torch.cat((np123_0_lls, np123_1_lls), dim = 0)
    ns123_lls = torch.matmul(params1, np123_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[272:304,:] - ns123_lls) < 1e-3)

    np0123_0_lls = ns012_lls + node_mars[112:144,:]
    np0123_1_lls = ns123_lls + node_mars[16:48,:]
    np0123_lls = torch.cat((np0123_0_lls, np0123_1_lls), dim = 0)
    ns0123_lls = torch.matmul(params1, np0123_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[304:336,:] - ns0123_lls) < 1e-3)

    ## Backward tests ##

    pc.backward(data.permute(1, 0), allow_modify_flows = False)

    node_flows = pc.node_flows.detach().cpu().clone()
    param_flows = pc.param_flows.detach().cpu().clone()

    assert torch.all(torch.abs(node_flows[304:336,:] - 1.0) < 1e-4)

    pc.inner_layer_groups[4][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[5][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np0123_flows = torch.matmul(params1.permute(1, 0), 1.0 / ns0123_lls.exp()) * np0123_lls.exp()
    assert torch.all(torch.abs(element_flows[16:80,:] - np0123_flows) < 4e-3)

    param_flows1 = torch.matmul(1.0 / ns0123_lls.exp(), np0123_lls.exp().permute(1, 0)) * params1

    ns012_flows = element_flows[16:48,:]
    assert torch.all(torch.abs(node_flows[240:272,:] - ns012_flows) < 4e-3)

    ns123_flows = element_flows[48:80,:]
    assert torch.all(torch.abs(node_flows[272:304,:] - ns123_flows) < 4e-3)

    ni0_flows = element_flows[48:80,:].clone()
    ni3_flows = element_flows[16:48,:].clone()

    pc.inner_layer_groups[2][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[3][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np012_flows = torch.matmul(params1.permute(1, 0), ns012_flows / ns012_lls.exp()) * np012_lls.exp()
    assert torch.all(torch.abs(element_flows[16:80,:] - np012_flows) < 4e-3)

    param_flows1 += torch.matmul(ns012_flows / ns012_lls.exp(), np012_lls.exp().permute(1, 0)) * params1

    np123_flows = torch.matmul(params1.permute(1, 0), ns123_flows / ns123_lls.exp()) * np123_lls.exp()
    assert torch.all(torch.abs(element_flows[80:144,:] - np123_flows) < 4e-3)

    param_flows1 += torch.matmul(ns123_flows / ns123_lls.exp(), np123_lls.exp().permute(1, 0)) * params1

    ns01_flows = np012_flows[0:32,:]
    assert torch.all(torch.abs(node_flows[144:176,:] - ns01_flows) < 4e-3)

    ns12_flows = np012_flows[32:64,:] + np123_flows[0:32,:]
    assert torch.all(torch.abs(node_flows[176:208,:] - ns12_flows) < 4e-3)

    ns23_flows = np123_flows[32:64,:]
    assert torch.all(torch.abs(node_flows[208:240,:] - ns23_flows) < 1e-3)

    ni2_flows = np012_flows[0:32,:].clone()
    ni0_flows += np012_flows[32:64,:].clone()
    ni3_flows += np123_flows[0:32,:].clone()
    ni1_flows = np123_flows[32:64,:].clone()

    pc.inner_layer_groups[0][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[1][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np01_flows = torch.matmul(params0.permute(1, 0), ns01_flows / ns01_lls.exp()) * np01_lls.exp()
    assert torch.all(torch.abs(element_flows[16:48,:] - np01_flows) < 1e-2)

    param_flows0 = torch.matmul(ns01_flows / ns01_lls.exp(), np01_lls.exp().permute(1, 0)) * params0

    np12_flows = torch.matmul(params0.permute(1, 0), ns12_flows / ns12_lls.exp()) * np12_lls.exp()
    assert torch.all(torch.abs(element_flows[48:80,:] - np12_flows) < 1e-2)

    param_flows0 += torch.matmul(ns12_flows / ns12_lls.exp(), np12_lls.exp().permute(1, 0)) * params0

    np23_flows = torch.matmul(params0.permute(1, 0), ns23_flows / ns23_lls.exp()) * np23_lls.exp()
    assert torch.all(torch.abs(element_flows[80:112,:] - np23_flows) < 1e-2)

    param_flows0 += torch.matmul(ns23_flows / ns23_lls.exp(), np23_lls.exp().permute(1, 0)) * params0

    ni0_flows += np01_flows.clone()
    ni1_flows += np01_flows.clone() + np12_flows.clone()
    ni2_flows += np12_flows.clone() + np23_flows.clone()
    ni3_flows += np23_flows.clone()

    assert torch.all(torch.abs(node_flows[16:48,:] - ni0_flows) < 1e-2)
    assert torch.all(torch.abs(node_flows[48:80,:] - ni1_flows) < 1e-2)
    assert torch.all(torch.abs(node_flows[80:112,:] - ni2_flows) < 1e-2)
    assert torch.all(torch.abs(node_flows[112:144,:] - ni3_flows) < 1e-2)

    ref_param_flows0 = (param_flows[0:1024] + param_flows[1024:2048]).reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(-1)
    assert torch.all(torch.abs(param_flows0.reshape(-1) - ref_param_flows0) < 1e-2)

    ref_param_flows1 = param_flows[2048:4096].reshape(2, 4, 16, 16).permute(0, 3, 1, 2).reshape(-1)
    assert torch.all(torch.abs(param_flows1.reshape(-1) - ref_param_flows1) < 1e-2)

    ## Parameter learning & flow aggregation tests ##

    temp_param_flows = param_flows.clone().to(device)

    compute_cum_par_flows(temp_param_flows, pc.parflow_fusing_kwargs)

    ref_param_flows0 = temp_param_flows[0:1024].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(-1)
    assert torch.all(torch.abs(param_flows0.reshape(-1) - ref_param_flows0.cpu()) < 1e-2)

    ref_param_flows1 = temp_param_flows[2048:4096].reshape(2, 4, 16, 16).permute(0, 3, 1, 2).reshape(-1)
    assert torch.all(torch.abs(param_flows1.reshape(-1) - ref_param_flows1.cpu()) < 1e-2)

    em_par_update(pc.params, temp_param_flows, pc.par_update_kwargs, step_size = 1.0, pseudocount = 0.0)

    param_flows0 /= param_flows0.sum(dim = 1, keepdim = True)
    ref_params0 = pc.params[256:1280].reshape(2, 2, 16, 16).permute(0, 3, 1, 2).reshape(-1)
    assert torch.all(torch.abs(param_flows0.reshape(-1) - ref_params0.cpu()) < 1e-4)

    param_flows1 /= param_flows1.sum(dim = 1, keepdim = True)
    ref_params1 = pc.params[1280:3328].reshape(2, 4, 16, 16).permute(0, 3, 1, 2).reshape(-1)
    assert torch.all(torch.abs(param_flows1.reshape(-1) - ref_params1.cpu()) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(2390)
    test_simple_structure_block1()
    test_simple_structure_block16()
