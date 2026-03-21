import torch
import pyjuice as juice
import time
import pyjuice.nodes.distributions as dists

import pytest


def test_codd_hmm():

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

    unnorm_lls = pc(
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


if __name__ == "__main__":
    test_codd_hmm()
