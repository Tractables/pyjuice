import torch
import pyjuice as juice
import time
import pyjuice.nodes.distributions as dists
import math

import pytest


def test_soft_evidence_categorical_dist():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1278

    # Construct PC
    nis = [
        juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats)) for v in range(num_vars)
    ]

    np = juice.multiply(*nis)
    ns = juice.summate(np, num_nodes = 1)

    ns.init_parameters(perturbation = 0.0)

    pc = juice.compile(ns)
    pc.to(device)

    # Inputs
    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
    logits = torch.rand([batch_size, num_vars, num_cats], device = device) * 2 - 1

    # Compute groundtruth
    logits.requires_grad = True
    logits.retain_grad()
    logps = torch.log_softmax(logits, dim = 2)
    target_lls = logps.gather(2, data.unsqueeze(2)).squeeze(2).sum(dim = 1)

    target_lls.sum().backward()
    target_logits_grad = logits.grad

    ###########################
    ## Forward pass runtests ##
    ###########################

    lls = pc(
        data,
        soft_evidence_logp = logits
    )

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    logits_grad = torch.zeros_like(logits)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        soft_evidence_logp = logits,
        soft_evidence_logp_grad = logits_grad
    )
    
    assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)


def test_soft_evidence_categorical_dist_varied():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1000

    # Construct PC
    nis = [
        juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats)) for v in range(num_vars)
    ]

    np = juice.multiply(*nis)
    ns = juice.summate(np, num_nodes = 1)

    ns.init_parameters(perturbation = 0.0)
    for ni in nis:
        ni._params[:500] *= 3
        ni._params /= 4

    pc = juice.compile(ns)
    pc.to(device)

    # Inputs
    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
    logits = torch.rand([batch_size, num_vars, num_cats], device = device) * 2 - 1
    add_logits = torch.zeros_like(logits)
    add_logits[:,:,:500] = math.log(3)

    # Compute groundtruth
    logits.requires_grad = True
    logits.retain_grad()
    logps = torch.log_softmax(logits + add_logits, dim = 2)
    target_lls = logps.gather(2, data.unsqueeze(2)).squeeze(2).sum(dim = 1)

    target_lls.sum().backward()
    target_logits_grad = logits.grad

    ###########################
    ## Forward pass runtests ##
    ###########################

    lls = pc(
        data,
        soft_evidence_logp = logits
    )

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    logits_grad = torch.zeros_like(logits)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        soft_evidence_logp = logits,
        soft_evidence_logp_grad = logits_grad
    )
    
    assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)


def test_soft_evidence_categorical_dist_multi_nodes():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1278

    for num_nodes in [2, 4, 8, 16, 32]:

        # Construct PC
        nis = [
            juice.inputs(v, num_nodes = num_nodes, dist = dists.SoftEvidenceCategorical(num_cats = num_cats)) for v in range(num_vars)
        ]

        np = juice.multiply(*nis)
        ns = juice.summate(np, num_nodes = 1)

        ns.init_parameters(perturbation = 0.0)

        pc = juice.compile(ns)
        pc.to(device)

        # Inputs
        data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
        logits = torch.rand([batch_size, num_vars, num_cats], device = device) * 2 - 1

        # Compute groundtruth
        logits.requires_grad = True
        logits.retain_grad()
        logps = torch.log_softmax(logits, dim = 2)
        target_lls = logps.gather(2, data.unsqueeze(2)).squeeze(2).sum(dim = 1)

        target_lls.sum().backward()
        target_logits_grad = logits.grad

        ###########################
        ## Forward pass runtests ##
        ###########################

        lls = pc(
            data,
            soft_evidence_logp = logits
        )

        assert torch.all(torch.abs(lls.view(-1) - target_lls) / num_vars < 1e-3)

        ############################
        ## Backward pass runtests ##
        ############################

        logits_grad = torch.zeros_like(logits)

        pc.backward(
            data, allow_modify_flows = False, logspace_flows = True,
            soft_evidence_logp = logits,
            soft_evidence_logp_grad = logits_grad
        )
        
        assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)


if __name__ == "__main__":
    torch.manual_seed(4343442)
    torch.cuda.manual_seed(5434)
    test_soft_evidence_categorical_dist()
    test_soft_evidence_categorical_dist_varied()
    test_soft_evidence_categorical_dist_multi_nodes()
