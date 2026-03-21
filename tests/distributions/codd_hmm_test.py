import torch
import pyjuice as juice
import time
import pyjuice.nodes.distributions as dists

import pytest


def test_codd_hmm():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1437

    root_ns = juice.structures.GeneralizedHMM(
        seq_length = num_vars,
        num_latents = 1024,
        homogeneous = True,
        input_dist = dists.ExternProductCategorical(num_cats = num_cats)
    )
    root_ns.init_parameters(perturbation = 4.0)

    pc = juice.compile(root_ns)
    pc.to(device)

    # Extract parameters
    emit_params = root_ns.chs[0].chs[0].get_source_ns()._params.reshape(1024, num_cats).to(device)

    var_order = pc.input_layer_group[0].vids[::1024]

    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
    ext_logps = torch.log_softmax(
        torch.rand([batch_size, num_vars, num_cats], device = device), 
        dim = 2
    )


    ###########################
    ## Forward pass runtests ##
    ###########################

    unnorm_lls = pc(
        data,
        external_categorical_logps = ext_logps,
        extern_product_categorical_mode = "unnormalized_ll"
    )

    input_layer = pc.input_layer_group[0]
    sid, eid = input_layer._output_ind_range

    mars = pc.node_mars[sid:eid,:].reshape(num_vars, 1024, batch_size).permute(2, 0, 1)

    for i in range(num_vars):
        v = var_order[i,0]
        
        log_params = ext_logps[:,v,:][:,None,:] + emit_params[None,:,:].log()
        target_mars = log_params.gather(2, data[:,v][:,None,None].repeat(1, 1024, 1)).squeeze()
        
        v_mars = mars[:,i,:]

        assert torch.all(torch.abs(target_mars - v_mars) < 1e-4)


if __name__ == "__main__":
    test_codd_hmm()
