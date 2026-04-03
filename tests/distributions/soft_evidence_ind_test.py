import torch
import pyjuice as juice
import time
import pyjuice.nodes.distributions as dists
import math

import pytest


def test_soft_evidence_indicator_dist():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 7
    num_nodes = 16

    # Construct PC
    nis = [
        juice.inputs(v, num_nodes = num_nodes, dist = dists.SoftEvidenceIndicator(num_states = num_nodes)) for v in range(num_vars)
    ]

    np = juice.multiply(*nis)
    ns = juice.summate(np, num_nodes = 1)

    ns.init_parameters(perturbation = 4.0)

    pc = juice.compile(ns)
    pc.to(device)

    # Inputs
    data = torch.zeros([batch_size, num_vars], device = device)
    logits = torch.rand([batch_size, num_vars, num_nodes], device = device).log()

    input_layer = pc.input_layer_group[0]

    ###########################
    ## Forward pass runtests ##
    ###########################

    lls = pc(
        data,
        indicator_evidence_logp = logits
    )

    target_lls = (logits.sum(dim = 1) + ns._params.view(-1).log().to(device)[None,:]).logsumexp(dim = 1)

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    logits_grad = torch.zeros_like(logits)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        indicator_evidence_logp = logits,
        indicator_evidence_logp_grad = logits_grad
    )

    vids = input_layer.vids[::num_nodes].view(-1).sort().indices
    target_logits_grad = pc.node_flows[1:1+num_vars*num_nodes,:].reshape(num_vars, num_nodes, batch_size).permute(2, 0, 1)[:,vids,:]
    
    assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)


if __name__ == "__main__":
    test_soft_evidence_indicator_dist()
