import torch
import pyjuice as juice
import time
import pyjuice.nodes.distributions as dists

import pytest


def test_codd_hmm_no_filter():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1278
    num_latents = 1024

    root_ns = juice.structures.GeneralizedHMM(
        seq_length = num_vars,
        num_latents = num_latents,
        homogeneous = True,
        input_dist = dists.SoftEvidenceCategorical(num_cats = num_cats)
    )
    root_ns.init_parameters(perturbation = 32.0)

    pc = juice.compile(root_ns)
    pc.to(device)

    # Extract parameters
    emit_params = root_ns.chs[0].chs[0].get_source_ns()._params.reshape(num_latents, num_cats).to(device)

    var_order = pc.input_layer_group[0].vids[::num_latents]

    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)

    soft_evidence_logp = torch.log_softmax(
        torch.rand([batch_size, num_vars, num_cats], device = device), 
        dim = 2
    )

    ###########################
    ## Forward pass runtests ##
    ###########################

    lls = pc(
        data,
        soft_evidence_logp = soft_evidence_logp
    )

    input_layer = pc.input_layer_group[0]
    sid, eid = input_layer._output_ind_range

    mars = pc.node_mars[sid:eid,:].reshape(num_vars, num_latents, batch_size).permute(2, 0, 1)

    for i in range(num_vars):
        v = var_order[i,0]
        
        log_params = soft_evidence_logp[:,v,:][:,None,:] + emit_params[None,:,:].log()
        unnorm_mars = log_params.gather(2, data[:,v][:,None,None].repeat(1, num_latents, 1)).squeeze()

        logZ = torch.logsumexp(log_params, dim = 2)
        target_mars = unnorm_mars - logZ
        
        v_mars = mars[:,i,:]

        assert torch.all(torch.abs(target_mars - v_mars) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    soft_evidence_logp_grad = torch.zeros_like(soft_evidence_logp)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        soft_evidence_logp = soft_evidence_logp,
        soft_evidence_logp_grad = soft_evidence_logp_grad
    )

    flows = pc.node_flows[sid:eid,:].reshape(num_vars, num_latents, batch_size).permute(2, 0, 1) # [B, V, N]

    pflows = input_layer.param_flows.reshape(-1, num_latents, num_cats).sum(dim = 0)

    my_pflows = torch.zeros_like(pflows)
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_pflows[:,data[b,v]] += flows[b,i,:].exp()

    assert torch.all(torch.abs(pflows - my_pflows) < 1e-5)

    my_grads = torch.zeros_like(soft_evidence_logp_grad) # [B, V, K]
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_grads[b,v,data[b,v]] += flows[b,i,:].exp().sum()

            log_params = soft_evidence_logp[b,v,:][None,:] + emit_params.log() # [N, C]
            logZ = torch.logsumexp(log_params, dim = 1) # [N]

            my_grads[b,v,:] -= (flows[b,i,:][:,None] + log_params - logZ[:,None]).exp().sum(dim = 0)

    assert torch.all(torch.abs(soft_evidence_logp_grad - my_grads) < 1e-4)


def test_codd_hmm_with_filter():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1278
    num_latents = 1024
    num_selected_cats = 128

    root_ns = juice.structures.GeneralizedHMM(
        seq_length = num_vars,
        num_latents = num_latents,
        homogeneous = True,
        input_dist = dists.SoftEvidenceCategorical(num_cats = num_cats)
    )
    root_ns.init_parameters(perturbation = 32.0)

    pc = juice.compile(root_ns)
    pc.to(device)

    # Extract parameters
    emit_params = root_ns.chs[0].chs[0].get_source_ns()._params.reshape(num_latents, num_cats).to(device)

    var_order = pc.input_layer_group[0].vids[::num_latents]

    soft_evidence_logp = torch.log_softmax(
        torch.rand([batch_size, num_vars, num_cats], device = device), 
        dim = 2
    )

    soft_evidence_cat_ids = torch.multinomial(
        torch.ones(batch_size * num_vars, num_cats, dtype = torch.float32, device = device),
        num_samples = num_selected_cats,
        replacement = False
    ).reshape(batch_size, num_vars, num_selected_cats).contiguous()
    
    soft_evidence_logp = soft_evidence_logp.gather(2, soft_evidence_cat_ids).contiguous()

    data = soft_evidence_cat_ids[:,:,0].contiguous()

    ###########################
    ## Forward pass runtests ##
    ###########################

    unnorm_lls = pc(
        data,
        soft_evidence_logp = soft_evidence_logp,
        soft_evidence_cat_ids = soft_evidence_cat_ids
    )

    input_layer = pc.input_layer_group[0]
    sid, eid = input_layer._output_ind_range

    mars = pc.node_mars[sid:eid,:].reshape(num_vars, num_latents, batch_size).permute(2, 0, 1)

    for i in range(num_vars):
        v = var_order[i,0]
        
        log_params = soft_evidence_logp[:,v,:][:,None,:] + emit_params[None,:,:].repeat(batch_size, 1, 1).gather(2, soft_evidence_cat_ids[:,v,:][:,None,:].repeat(1, num_latents, 1)).log()
        unnorm_mars = log_params[:,:,0]

        logZ = torch.logsumexp(log_params, dim = 2)
        target_mars = unnorm_mars - logZ
        
        v_mars = mars[:,i,:]

        assert torch.all(torch.abs(target_mars - v_mars) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    soft_evidence_logp_grad = torch.zeros_like(soft_evidence_logp)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        soft_evidence_logp = soft_evidence_logp,
        soft_evidence_logp_grad = soft_evidence_logp_grad,
        soft_evidence_cat_ids = soft_evidence_cat_ids
    )

    flows = pc.node_flows[sid:eid,:].reshape(num_vars, num_latents, batch_size).permute(2, 0, 1) # [B, V, N]

    pflows = input_layer.param_flows.reshape(-1, num_latents, num_cats).sum(dim = 0)

    my_pflows = torch.zeros_like(pflows)
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_pflows[:,data[b,v]] += flows[b,i,:].exp()

    assert torch.all(torch.abs(pflows - my_pflows) < 1e-5)

    my_grads = torch.zeros_like(soft_evidence_logp_grad) # [B, V, K]
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_grads[b,v,0] += flows[b,i,:].exp().sum()

            log_params = soft_evidence_logp[b,v,:][None,:] + emit_params[:,soft_evidence_cat_ids[b,v,:]].log() # [N, C]
            logZ = torch.logsumexp(log_params, dim = 1) # [N]

            my_grads[b,v,:] -= (flows[b,i,:][:,None] + log_params - logZ[:,None]).exp().sum(dim = 0)

    assert torch.all(torch.abs(soft_evidence_logp_grad - my_grads) < 1e-4)


if __name__ == "__main__":
    test_codd_hmm_no_filter()
    test_codd_hmm_with_filter()
