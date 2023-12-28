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


def block_sparse_pc_test():
    
    device = torch.device("cuda:0")

    num_node_groups = 4
    batch_size = 512

    for group_size in [16, 8, 1]:
    
        with juice.set_group_size(group_size):

            ni00 = inputs(0, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 4))
            ni10 = inputs(1, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 4))
            np0 = multiply(ni00, ni10)

            ni01 = inputs(0, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 4))
            ni11 = inputs(1, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 4))
            np1 = multiply(ni01, ni11)

            ni02 = inputs(0, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 4))
            ni12 = inputs(1, num_node_groups = num_node_groups, dist = dists.Categorical(num_cats = 4))
            np2 = multiply(ni02, ni12)

            edge_indicators = torch.rand([num_node_groups, 3 * num_node_groups]) < 0.3
            edge_indicators[:,0] = True
            edge_ids = torch.nonzero(edge_indicators, as_tuple = False).permute(1, 0)

            ns = summate(np0, np1, np2, edge_ids = edge_ids)

        ns.init_parameters()

        pc = TensorCircuit(ns, layer_sparsity_tol = 1.0)

        pc.to(device)

        data = torch.randint(0, 4, [batch_size, 2], device = device)

        ## Forward tests ##

        lls = pc(data, force_use_fp32 = True)

        node_mars = pc.node_mars.cpu()
        element_mars = pc.element_mars.cpu()

        np0_vals = element_mars[group_size:group_size*(num_node_groups+1),:].exp().reshape(num_node_groups, group_size, batch_size)
        np1_vals = element_mars[group_size*(num_node_groups+1):group_size*(num_node_groups*2+1),:].exp().reshape(num_node_groups, group_size, batch_size)
        np2_vals = element_mars[group_size*(num_node_groups*2+1):group_size*(num_node_groups*3+1),:].exp().reshape(num_node_groups, group_size, batch_size)

        params = ns._params

        ns_vals = torch.zeros([num_node_groups, group_size, batch_size])

        for i in range(edge_ids.size(1)):
            ni, ci = edge_ids[0,i], edge_ids[1,i]
            if ci < num_node_groups:
                ns_vals[ni,:,:] += torch.matmul(params[i], np0_vals[ci])
            elif ci < num_node_groups * 2:
                ns_vals[ni,:,:] += torch.matmul(params[i], np1_vals[ci-num_node_groups])
            else:
                ns_vals[ni,:,:] += torch.matmul(params[i], np2_vals[ci-num_node_groups*2])

        sid, eid = (num_node_groups * 6 + 1) * group_size, (num_node_groups * 7 + 1) * group_size
        ref_ns_vals = node_mars[sid:eid,:].exp().reshape(num_node_groups, group_size, batch_size)

        assert torch.all(torch.abs(ns_vals - ref_ns_vals) < 1e-4)

        ## Backward tests ##

        pc.backward(data.permute(1, 0), allow_modify_flows = False)

        node_flows = pc.node_flows.cpu()
        element_flows = pc.element_flows.cpu()
        param_flows = pc.param_flows.cpu()

        np0_flows = torch.zeros([num_node_groups, group_size, batch_size])
        np1_flows = torch.zeros([num_node_groups, group_size, batch_size])
        np2_flows = torch.zeros([num_node_groups, group_size, batch_size])

        for i in range(edge_ids.size(1)):
            ni, ci = edge_ids[0,i], edge_ids[1,i]
            if ci < num_node_groups:
                np0_flows[ci] += torch.matmul(params[i].permute(1, 0), 1.0 / ns_vals[ni]) * np0_vals[ci]
            elif ci < num_node_groups * 2:
                np1_flows[ci-num_node_groups] += torch.matmul(params[i].permute(1, 0), 1.0 / ns_vals[ni]) * np1_vals[ci-num_node_groups]
            else:
                np2_flows[ci-num_node_groups*2] += torch.matmul(params[i].permute(1, 0), 1.0 / ns_vals[ni]) * np2_vals[ci-num_node_groups*2]

        ref_np0_flows = element_flows[group_size:group_size*(num_node_groups+1),:].reshape(num_node_groups, group_size, batch_size)
        ref_np1_flows = element_flows[group_size*(num_node_groups+1):group_size*(num_node_groups*2+1),:].reshape(num_node_groups, group_size, batch_size)
        ref_np2_flows = element_flows[group_size*(num_node_groups*2+1):group_size*(num_node_groups*3+1),:].reshape(num_node_groups, group_size, batch_size)

        assert torch.all(torch.abs(np0_flows - ref_np0_flows) < 1e-3)
        assert torch.all(torch.abs(np1_flows - ref_np1_flows) < 1e-3)
        assert torch.all(torch.abs(np2_flows - ref_np2_flows) < 1e-3)

        param_flows = param_flows.reshape(edge_ids.size(1), group_size, group_size).permute(0, 2, 1)

        for i in range(edge_ids.size(1)):
            ni, ci = edge_ids[0,i], edge_ids[1,i]
            if ci < num_node_groups:
                curr_par_flows = torch.matmul(1.0 / ns_vals[ni], np0_vals[ci].permute(1, 0)) * params[i]
            elif ci < num_node_groups * 2:
                curr_par_flows = torch.matmul(1.0 / ns_vals[ni], np1_vals[ci-num_node_groups].permute(1, 0)) * params[i]
            else:
                curr_par_flows = torch.matmul(1.0 / ns_vals[ni], np2_vals[ci-num_node_groups*2].permute(1, 0)) * params[i]

            assert torch.all(torch.abs(param_flows[i] - curr_par_flows) < 1e-2)


if __name__ == "__main__":
    torch.manual_seed(3890)
    block_sparse_pc_test()
