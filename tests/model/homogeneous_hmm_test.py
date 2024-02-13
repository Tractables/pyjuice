import pyjuice as juice
import torch
import numpy as np

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs, set_block_size
from pyjuice.model import TensorCircuit
from pyjuice.model.backend import compute_cum_par_flows, em_par_update

import pytest


def homogeneous_hmm_test():

    block_size = 1

    with set_block_size(block_size = block_size):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))
        ni1 = ni0.duplicate(1, tie_params = True)
        ni2 = ni0.duplicate(2, tie_params = True)
        ni3 = ni0.duplicate(3, tie_params = True)

        np01 = multiply(ni0, ni1)
        ns01 = summate(np01, num_node_blocks = 2)

        np012 = multiply(ns01, ni2)
        ns012 = ns01.duplicate(np012, tie_params = True)

        np0123 = multiply(ns012, ni3)
        ns0123 = ns012.duplicate(np0123, tie_params = True)

    ns0123.init_parameters()

    pc = TensorCircuit(ns0123, max_tied_ns_per_parflow_block = 2)

    device = torch.device("cuda:0")

    ## Compilation tests ##

    assert torch.all(pc.input_layer_group[0].vids == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).reshape(8, 1))
    assert torch.all(pc.input_layer_group[0].s_pids == torch.tensor([0, 5, 0, 5, 0, 5, 0, 5]))
    assert torch.all(pc.input_layer_group[0].s_pfids == torch.tensor([0, 5, 0, 5, 10, 15, 10, 15]))
    assert torch.all(pc.input_layer_group[0].metadata == torch.tensor([5.0, 5.0, 5.0, 5.0]))
    assert torch.all(pc.input_layer_group[0].s_mids == torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    assert torch.all(pc.input_layer_group[0].source_nids == torch.tensor([0, 1]))
    assert pc.input_layer_group[0]._output_ind_range[0] == 1
    assert pc.input_layer_group[0]._output_ind_range[1] == 9

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_nids[0] == torch.tensor([1, 2]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_cids[0] == torch.tensor([[1, 3], [2, 4]]))

    assert torch.all(pc.inner_layer_groups[0][0].partitioned_u_cids[0] == torch.tensor([1, 2, 3, 4]))
    assert torch.all(pc.inner_layer_groups[0][0].partitioned_parids[0] == torch.tensor([[1], [2], [1], [2]]))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_nids[0] == torch.tensor([9, 10]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_cids[0] == torch.tensor([[1, 2], [1, 2]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pids[0] == torch.tensor([[1, 2], [3, 4]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_pfids[0] == torch.tensor([[0, 1], [2, 3]]))

    assert torch.all(pc.inner_layer_groups[1][0].partitioned_chids[0] == torch.tensor([1, 2]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parids[0] == torch.tensor([[9, 10], [9, 10]]))
    assert torch.all(pc.inner_layer_groups[1][0].partitioned_parpids[0] == torch.tensor([[1, 3], [2, 4]]))

    assert torch.all(pc.inner_layer_groups[2][0].partitioned_nids[0] == torch.tensor([1, 2]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_cids[0] == torch.tensor([[9, 5], [10, 6]]))

    assert torch.all(pc.inner_layer_groups[2][0].partitioned_u_cids[0] == torch.tensor([5, 6, 9, 10]))
    assert torch.all(pc.inner_layer_groups[2][0].partitioned_parids[0] == torch.tensor([[1], [2], [1], [2]]))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_nids[0] == torch.tensor([11, 12]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_cids[0] == torch.tensor([[1, 2], [1, 2]]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pids[0] == torch.tensor([[1, 2], [3, 4]]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_pfids[0] == torch.tensor([[0, 1], [2, 3]]))

    assert torch.all(pc.inner_layer_groups[3][0].partitioned_chids[0] == torch.tensor([1, 2]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parids[0] == torch.tensor([[11, 12], [11, 12]]))
    assert torch.all(pc.inner_layer_groups[3][0].partitioned_parpids[0] == torch.tensor([[1, 3], [2, 4]]))

    assert torch.all(pc.inner_layer_groups[4][0].partitioned_nids[0] == torch.tensor([1, 2]))
    assert torch.all(pc.inner_layer_groups[4][0].partitioned_cids[0] == torch.tensor([[11, 7], [12, 8]]))

    assert torch.all(pc.inner_layer_groups[4][0].partitioned_u_cids[0] == torch.tensor([7, 8, 11, 12]))
    assert torch.all(pc.inner_layer_groups[4][0].partitioned_parids[0] == torch.tensor([[1], [2], [1], [2]]))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_nids[0] == torch.tensor([13, 14]))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_cids[0] == torch.tensor([[1, 2], [1, 2]]))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pids[0] == torch.tensor([[1, 2], [3, 4]]))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_pfids[0] == torch.tensor([[0, 1], [2, 3]]))

    assert torch.all(pc.inner_layer_groups[5][0].partitioned_chids[0] == torch.tensor([1, 2]))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_parids[0] == torch.tensor([[13, 14], [13, 14]]))
    assert torch.all(pc.inner_layer_groups[5][0].partitioned_parpids[0] == torch.tensor([[1, 3], [2, 4]]))

    pc.to(device)

    ## Forward tests ##

    data = torch.randint(0, 5, [16, 4]).to(device)
    data_cpu = data.cpu()

    lls = pc(data)

    node_mars = pc.node_mars.detach().cpu()
    params = pc.params.detach().cpu()

    params0 = ni0._params.reshape(2, 5)

    assert torch.all(torch.abs(node_mars[1,:].exp() - params0[0, data_cpu[:,0]]) < 1e-4)
    assert torch.all(torch.abs(node_mars[2,:].exp() - params0[1, data_cpu[:,0]]) < 1e-4)

    assert torch.all(torch.abs(node_mars[3,:].exp() - params0[0, data_cpu[:,1]]) < 1e-4)
    assert torch.all(torch.abs(node_mars[4,:].exp() - params0[1, data_cpu[:,1]]) < 1e-4)

    assert torch.all(torch.abs(node_mars[5,:].exp() - params0[0, data_cpu[:,2]]) < 1e-4)
    assert torch.all(torch.abs(node_mars[6,:].exp() - params0[1, data_cpu[:,2]]) < 1e-4)

    assert torch.all(torch.abs(node_mars[7,:].exp() - params0[0, data_cpu[:,3]]) < 1e-4)
    assert torch.all(torch.abs(node_mars[8,:].exp() - params0[1, data_cpu[:,3]]) < 1e-4)

    params1 = ns01.get_source_ns()._params.reshape(2, 2)

    np01_lls = node_mars[1:3,:] + node_mars[3:5,:]
    ns01_lls = torch.matmul(params1, np01_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[9:11,:] - ns01_lls) < 1e-4)

    np012_lls = node_mars[5:7,:] + node_mars[9:11,:]
    ns012_lls = torch.matmul(params1, np012_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[11:13,:] - ns012_lls) < 1e-4)

    np0123_lls = node_mars[7:9,:] + node_mars[11:13,:]
    ns0123_lls = torch.matmul(params1, np0123_lls.exp()).log()
    assert torch.all(torch.abs(node_mars[13:15,:] - ns0123_lls) < 1e-4)

    ## Backward tests ##

    pc.backward(data.permute(1, 0), allow_modify_flows = False)

    node_flows = pc.node_flows.detach().cpu().clone()
    param_flows = pc.param_flows.detach().cpu().clone()

    assert torch.all(torch.abs(node_flows[13:15,:] - 1.0) < 1e-4)

    pc.inner_layer_groups[4][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[5][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np0123_flows = torch.matmul(params1.permute(1, 0), 1.0 / ns0123_lls.exp()) * np0123_lls.exp()
    assert torch.all(torch.abs(element_flows[1:3,:] - np0123_flows) < 1e-4)

    param_flows1 = torch.matmul(1.0 / ns0123_lls.exp(), np0123_lls.exp().permute(1, 0)) * params1

    ni3_flows = element_flows[1:3,:]
    ns012_flows = element_flows[1:3,:]
    assert torch.all(torch.abs(node_flows[11:13,:] - ns012_flows) < 1e-4)

    pc.inner_layer_groups[2][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[3][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np012_flows = torch.matmul(params1.permute(1, 0), ns012_flows / ns012_lls.exp()) * np012_lls.exp()
    assert torch.all(torch.abs(element_flows[1:3,:] - np012_flows) < 1e-4)

    param_flows1 += torch.matmul(ns012_flows / ns012_lls.exp(), np012_lls.exp().permute(1, 0)) * params1

    ni2_flows = element_flows[1:3,:]
    ns01_flows = element_flows[1:3,:]
    assert torch.all(torch.abs(node_flows[9:11,:] - ns01_flows) < 1e-4)

    pc.inner_layer_groups[0][0](pc.node_mars, pc.element_mars, _for_backward = True)
    pc.inner_layer_groups[1][0].backward(
        pc.node_flows, pc.element_flows, pc.node_mars, pc.element_mars, 
        pc.params, pc.param_flows
    )
    element_flows = pc.element_flows.detach().cpu()

    np01_flows = torch.matmul(params1.permute(1, 0), ns01_flows / ns01_lls.exp()) * np01_lls.exp()
    assert torch.all(torch.abs(element_flows[1:3,:] - np01_flows) < 1e-4)

    param_flows1 += torch.matmul(ns01_flows / ns01_lls.exp(), np01_lls.exp().permute(1, 0)) * params1

    ni0_flows = element_flows[1:3,:]
    ni1_flows = element_flows[1:3,:]

    assert torch.all(torch.abs(param_flows1.reshape(-1) - param_flows[0:4]) < 1e-4)

    ## Parameter learning & flow aggregation tests ##

    temp_param_flows = param_flows.clone().to(device)

    compute_cum_par_flows(temp_param_flows, pc.parflow_fusing_kwargs)

    assert torch.all(torch.abs(param_flows1.reshape(-1) - temp_param_flows[0:4].cpu()) < 1e-4)

    gt_param_flows0 = pc.input_layer_group[0].param_flows[0:10] + pc.input_layer_group[0].param_flows[10:20]
    param_flows0 = torch.zeros([2, 5])

    for i in range(16):
        param_flows0[0, data_cpu[i,0]] += ni0_flows[0,i]
        param_flows0[1, data_cpu[i,0]] += ni0_flows[1,i]
        param_flows0[0, data_cpu[i,1]] += ni1_flows[0,i]
        param_flows0[1, data_cpu[i,1]] += ni1_flows[1,i]
        param_flows0[0, data_cpu[i,2]] += ni2_flows[0,i]
        param_flows0[1, data_cpu[i,2]] += ni2_flows[1,i]
        param_flows0[0, data_cpu[i,3]] += ni3_flows[0,i]
        param_flows0[1, data_cpu[i,3]] += ni3_flows[1,i]

    assert torch.all(torch.abs(param_flows0.reshape(-1) - gt_param_flows0.cpu()) < 1e-4)


if __name__ == "__main__":
    torch.manual_seed(2390)
    homogeneous_hmm_test()
