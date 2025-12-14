import pyjuice as juice
import torch
import numpy as np
import math

import pyjuice.nodes.distributions as dists
from pyjuice.utils import BitSet
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.model import TensorCircuit

import pytest


def test_external_categorical_dist_fw():

    num_cats = 3298
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 2]).to(device)

    external_categorical_logps = torch.rand([16, 2, num_cats], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll")

    unnorm_node_mars = pc.node_mars.detach().clone()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalized_ll")

    norm_node_mars = pc.node_mars.detach().clone()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalizing_constant")

    logz_node_mars = pc.node_mars.detach().clone()

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        internal_params = layer.params[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]
        external_params = external_categorical_logps[:,v,:].exp()

        params = internal_params[None,:,:] * external_params[:,None,:]

        normalized_params = params.log() - params.log().logsumexp(dim = 2, keepdim = True)

        logz = params.sum(dim = 2).log()

        normalized_logp = normalized_params.gather(2, data[:,v][:,None,None].expand(16, num_latents, 1)).squeeze(-1)
        unnormalized_logp = params.gather(2, data[:,v][:,None,None].expand(16, num_latents, 1)).squeeze(-1).log()

        pc_mars = unnorm_node_mars[sid:eid,:].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - unnormalized_logp) < 1e-4)

        pc_mars = norm_node_mars[sid:eid,:].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - normalized_logp) < 1e-4)

        pc_mars = logz_node_mars[sid:eid,:].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - logz) < 1e-4)


def test_external_categorical_dist_fw_dim2():

    num_cats = 3298
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 2]).to(device)

    external_categorical_logps = torch.rand([16, 2], device = device).log()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll")

    unnorm_node_mars = pc.node_mars.detach().clone()

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        internal_params = layer.params[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]
        external_params = external_categorical_logps[:,v].exp()

        params = internal_params[None,:,:] * external_params[:,None,None]

        unnormalized_logp = params.gather(2, data[:,v][:,None,None].expand(16, num_latents, 1)).squeeze(-1).log()

        pc_mars = unnorm_node_mars[sid:eid,:].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - unnormalized_logp) < 1e-4)


def test_external_categorical_dist_fw_w_mask():

    num_cats = 3298
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 2]).to(device)

    external_categorical_logps = torch.rand([16, 2, num_cats], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    external_categorical_value_mask = torch.zeros([16, 2], dtype = torch.bool, device = device)
    external_categorical_value_mask[8:,:] = True

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll",
             external_categorical_value_mask = external_categorical_value_mask)

    unnorm_node_mars = pc.node_mars.detach().clone()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalized_ll",
             external_categorical_value_mask = external_categorical_value_mask)

    norm_node_mars = pc.node_mars.detach().clone()

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        internal_params = layer.params[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]
        external_params = external_categorical_logps[:,v,:].exp()

        params = internal_params[None,:,:] * external_params[:,None,:]

        normalized_params = params.log() - params.log().logsumexp(dim = 2, keepdim = True)

        logz = params.sum(dim = 2).log()

        normalized_logp = normalized_params.gather(2, data[:,v][:,None,None].expand(16, num_latents, 1)).squeeze(-1)
        unnormalized_logp = params.gather(2, data[:,v][:,None,None].expand(16, num_latents, 1)).squeeze(-1).log()

        pc_mars = unnorm_node_mars[sid:eid,:8].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - logz[:8,:]) < 1e-4)

        pc_mars = unnorm_node_mars[sid:eid,8:].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - unnormalized_logp[8:,:]) < 1e-4)

        pc_mars = norm_node_mars[sid:eid,:8].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - logz[:8,:]) < 1e-4)

        pc_mars = norm_node_mars[sid:eid,8:].permute(1, 0)
        assert torch.all(torch.abs(pc_mars - normalized_logp[8:,:]) < 1e-4)


def test_external_categorical_dist_bk_param_only():

    num_cats = 3298
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 2]).to(device)

    external_categorical_logps = torch.rand([16, 2, num_cats], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    pc.zero_param_flows()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll")
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll")

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        param_flows = layer.param_flows[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]

        my_param_flows = torch.zeros_like(param_flows)

        for b in range(16):
            flows = pc.node_flows[sid:eid,b]
            my_param_flows[:,data[b,v]] += flows.exp()

        assert torch.all(torch.abs(param_flows - my_param_flows) < 1e-5)

    pc.zero_param_flows()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalized_ll")
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalized_ll")

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        param_flows = layer.param_flows[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]

        my_param_flows = torch.zeros_like(param_flows)

        for b in range(16):
            flows = pc.node_flows[sid:eid,b]
            my_param_flows[:,data[b,v]] += flows.exp()

        assert torch.all(torch.abs(param_flows - my_param_flows) < 1e-5)

    pc.zero_param_flows()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalizing_constant")
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalizing_constant")

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        internal_params = layer.params[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]
        external_params = external_categorical_logps[:,v,:].exp()

        params = internal_params[None,:,:] * external_params[:,None,:]
        normalized_params = params.log() - params.log().logsumexp(dim = 2, keepdim = True)

        param_flows = layer.param_flows[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]

        my_param_flows = torch.zeros_like(param_flows)

        for b in range(16):
            flows = pc.node_flows[sid:eid,b].exp()

            my_param_flows[:,:] += normalized_params[b,:,:].exp() * flows[:,None]

        assert torch.all(torch.abs(param_flows - my_param_flows) < 1e-5)


def test_external_categorical_dist_bk_param_only_w_mask():

    num_cats = 3298
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 2]).to(device)

    external_categorical_logps = torch.rand([16, 2, num_cats], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    external_categorical_value_mask = torch.zeros([16, 2], dtype = torch.bool, device = device)
    external_categorical_value_mask[8:,:] = True

    pc.zero_param_flows()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll",
             external_categorical_value_mask = external_categorical_value_mask)
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll",
                external_categorical_value_mask = external_categorical_value_mask)

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        internal_params = layer.params[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]
        external_params = external_categorical_logps[:,v,:].exp()

        params = internal_params[None,:,:] * external_params[:,None,:]
        normalized_params = params.log() - params.log().logsumexp(dim = 2, keepdim = True)

        param_flows = layer.param_flows[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]

        my_param_flows = torch.zeros_like(param_flows)

        for b in range(8, 16):
            flows = pc.node_flows[sid:eid,b]
            my_param_flows[:,data[b,v]] += flows.exp()

        for b in range(0, 8):
            flows = pc.node_flows[sid:eid,b].exp()
            my_param_flows[:,:] += normalized_params[b,:,:].exp() * flows[:,None]

        assert torch.all(torch.abs(param_flows - my_param_flows) < 1e-5)


def test_external_categorical_dist_bk_ext_grad():

    num_cats = 3298
    
    ni0 = inputs(0, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))
    ni1 = inputs(1, num_node_blocks = 2, block_size = 32, dist = dists.ExternProductCategorical(num_cats = num_cats))

    ms = multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns = summate(ms, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long), block_size = 1)

    pc = TensorCircuit(ns)

    device = torch.device("cuda:0")
    pc.to(device)

    data = torch.randint(0, num_cats, [16, 2]).to(device)

    external_categorical_logps = torch.rand([16, 2, num_cats], device = device)
    external_categorical_logps /= external_categorical_logps.sum(dim = 2, keepdim = True)
    external_categorical_logps = external_categorical_logps.log()

    external_categorical_logps_grad = torch.zeros_like(external_categorical_logps)

    pc.zero_param_flows()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll")
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "unnormalized_ll",
                external_categorical_logps_grad = external_categorical_logps_grad)

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        my_external_categorical_logps_grad = torch.zeros_like(external_categorical_logps_grad[:,v,:])

        for b in range(16):
            flows = pc.node_flows[sid:eid,b]
            my_external_categorical_logps_grad[b,data[b,v]] += flows.exp().sum()

        assert torch.all(torch.abs(external_categorical_logps_grad[:,v,:] - my_external_categorical_logps_grad) < 1e-5)

    pc.zero_param_flows()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalized_ll")
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalized_ll",
                external_categorical_logps_grad = external_categorical_logps_grad)

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        internal_params = layer.params[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]
        external_params = external_categorical_logps[:,v,:].exp()

        params = internal_params[None,:,:] * external_params[:,None,:]
        normalized_params = params.log() - params.log().logsumexp(dim = 2, keepdim = True) # [B, N, C]

        my_external_categorical_logps_grad = torch.zeros_like(external_categorical_logps_grad[:,v,:])

        for b in range(16):
            flows = pc.node_flows[sid:eid,b]
            my_external_categorical_logps_grad[b,data[b,v]] += flows.exp().sum()

            my_external_categorical_logps_grad[b,:] -= (normalized_params[b,:,:] + flows[:,None]).exp().sum(dim = 0)

        assert torch.all(torch.abs(external_categorical_logps_grad[:,v,:] - my_external_categorical_logps_grad) < 1e-5)

    pc.zero_param_flows()

    lls = pc(data, external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalizing_constant")
    pc.backward(data, logspace_flows = True, allow_modify_flows = False, 
                external_categorical_logps = external_categorical_logps, extern_product_categorical_mode = "normalizing_constant",
                external_categorical_logps_grad = external_categorical_logps_grad)

    num_latents = 64

    layer = pc.input_layer_group[0]
    vids = layer.vids[::num_latents,0]
    for i in range(pc.num_vars):
        v = vids[i]
        sid = layer._output_ind_range[0] + i * num_latents
        eid = layer._output_ind_range[0] + (i + 1) * num_latents

        internal_params = layer.params[layer.s_pids[i*num_latents:(i+1)*num_latents][:,None] + torch.arange(0, num_cats, device = device)[None,:]]
        external_params = external_categorical_logps[:,v,:].exp()

        params = internal_params[None,:,:] * external_params[:,None,:]
        normalized_params = params.log() - params.log().logsumexp(dim = 2, keepdim = True) # [B, N, C]

        my_external_categorical_logps_grad = torch.zeros_like(external_categorical_logps_grad[:,v,:])

        for b in range(16):
            flows = pc.node_flows[sid:eid,b]

            my_external_categorical_logps_grad[b,:] += (normalized_params[b,:,:] + flows[:,None]).exp().sum(dim = 0)

        assert torch.all(torch.abs(external_categorical_logps_grad[:,v,:] - my_external_categorical_logps_grad) < 1e-5)


if __name__ == "__main__":
    # test_external_categorical_dist_fw()
    # test_external_categorical_dist_fw_dim2()
    # test_external_categorical_dist_fw_w_mask()
    # test_external_categorical_dist_bk_param_only()
    test_external_categorical_dist_bk_param_only_w_mask()
    test_external_categorical_dist_bk_ext_grad()
