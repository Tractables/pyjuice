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
        juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False)) for v in range(num_vars)
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
        categorical_evidence_logp = logits
    )

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    logits_grad = torch.zeros_like(logits)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        categorical_evidence_logp = logits,
        categorical_evidence_logp_grad = logits_grad
    )
    
    assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)


def test_soft_evidence_categorical_dist_varied():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1000

    # Construct PC
    nis = [
        juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False)) for v in range(num_vars)
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
        categorical_evidence_logp = logits
    )

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    logits_grad = torch.zeros_like(logits)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        categorical_evidence_logp = logits,
        categorical_evidence_logp_grad = logits_grad
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
            juice.inputs(v, num_nodes = num_nodes, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False)) for v in range(num_vars)
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
            categorical_evidence_logp = logits
        )

        assert torch.all(torch.abs(lls.view(-1) - target_lls) / num_vars < 1e-3)

        ############################
        ## Backward pass runtests ##
        ############################

        logits_grad = torch.zeros_like(logits)

        pc.backward(
            data, allow_modify_flows = False, logspace_flows = True,
            categorical_evidence_logp = logits,
            categorical_evidence_logp_grad = logits_grad
        )
        
        assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)


def test_soft_evidence_categorical_dist_sample():

    device = torch.device("cuda:0")

    batch_size = 64
    num_cats = 43

    # Construct PC
    ni0 = juice.inputs(0, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False))
    ni1 = juice.inputs(1, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False))

    np = juice.multiply(ni0, ni1)
    ns = juice.summate(np, num_nodes = 1)

    ns.init_parameters(perturbation = 0.0)

    pc = juice.compile(ns)
    pc.to(device)

    probs = torch.rand([batch_size, 2, num_cats], device = device)
    probs /= probs.sum(dim = 2, keepdim = True)
    probs *= 0.1
    probs[:,:,10] += 0.9
    logits = probs.log()

    samples = juice.queries.sample(
        pc, 
        num_samples = batch_size,
        categorical_evidence_logp = logits
    )

    assert (samples == 10).float().mean() > 0.8


def test_soft_evidence_categorical_dist_dual_flow():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1278

    # Construct PC
    nis = [
        juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)) for v in range(num_vars)
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
        categorical_evidence_logp = logits
    )

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    logits_grad = torch.zeros_like(logits)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        categorical_evidence_logp = logits,
        categorical_evidence_logp_grad = logits_grad
    )
    
    assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)

    ## Runtest for pflow
    input_layer = pc.input_layer_group[0]
    sid, eid = input_layer._output_ind_range
    flows = pc.node_flows[sid:eid,:].reshape(num_vars, batch_size).permute(1, 0) # [B, V]

    params = input_layer.params.reshape(num_vars, num_cats)

    pflows = input_layer.param_flows.reshape(num_vars, num_cats * 2)
    pflows_num = pflows[:,:num_cats]
    pflows_denom = pflows[:,num_cats:]

    # Numerator
    var_order = pc.input_layer_group[0].vids
    my_pflows_num = torch.zeros_like(pflows_num)
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_pflows_num[i,data[b,v]] += flows[b,i].exp()

    assert torch.all(torch.abs(pflows_num - my_pflows_num) < 1e-5)

    # Denominator
    logps = logits + params[None,:,:].log()
    logps -= logps.logsumexp(dim = 2, keepdim = True)

    my_pflows_denom = torch.zeros_like(pflows_denom)
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_pflows_denom[i,:] += (flows[b,i] + logps[b,i,:]).exp()

    assert torch.all(torch.abs(pflows_denom - my_pflows_denom) < 1e-5)


def test_soft_evidence_categorical_dist_dual_flow_filtered():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32
    num_cats = 1278

    # Construct PC
    nis = [
        juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)) for v in range(num_vars)
    ]

    np = juice.multiply(*nis)
    ns = juice.summate(np, num_nodes = 1)

    ns.init_parameters(perturbation = 0.0)

    pc = juice.compile(ns)
    pc.to(device)

    # Inputs
    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
    logits = torch.rand([batch_size, num_vars, num_cats], device = device) * 2 - 1
    ids = torch.arange(0, num_cats, device = device)[None,None,:].repeat(batch_size, num_vars, 1).contiguous()

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
        categorical_evidence_logp = logits,
        soft_evidence_cat_ids = ids
    )

    # import pdb; pdb.set_trace()

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    logits_grad = torch.zeros_like(logits)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        categorical_evidence_logp = logits,
        categorical_evidence_logp_grad = logits_grad,
        soft_evidence_cat_ids = ids
    )
    
    assert torch.all(torch.abs(logits_grad - target_logits_grad) < 1e-5)

    ## Runtest for pflow
    input_layer = pc.input_layer_group[0]
    sid, eid = input_layer._output_ind_range
    flows = pc.node_flows[sid:eid,:].reshape(num_vars, batch_size).permute(1, 0) # [B, V]

    params = input_layer.params.reshape(num_vars, num_cats)

    pflows = input_layer.param_flows.reshape(num_vars, num_cats * 2)
    pflows_num = pflows[:,:num_cats]
    pflows_denom = pflows[:,num_cats:]

    # Numerator
    var_order = pc.input_layer_group[0].vids
    my_pflows_num = torch.zeros_like(pflows_num)
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_pflows_num[i,data[b,v]] += flows[b,i].exp()

    assert torch.all(torch.abs(pflows_num - my_pflows_num) < 1e-5)

    # Denominator
    logps = logits + params[None,:,:].log()
    logps -= logps.logsumexp(dim = 2, keepdim = True)

    my_pflows_denom = torch.zeros_like(pflows_denom)
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_pflows_denom[i,:] += (flows[b,i] + logps[b,i,:]).exp()

    assert torch.all(torch.abs(pflows_denom - my_pflows_denom) < 1e-5)


def test_soft_evidence_categorical_dist_dual_flow_em():

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 32

    for num_cats in [256, 2381]:

        # Construct PC
        nis = [
            juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)) for v in range(num_vars)
        ]

        np = juice.multiply(*nis)
        ns = juice.summate(np, num_nodes = 1)

        ns.init_parameters(perturbation = 2.0)

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

        lls = pc(
            data,
            categorical_evidence_logp = logits
        )

        logits_grad = torch.zeros_like(logits)

        pc.backward(
            data, allow_modify_flows = False, logspace_flows = True,
            categorical_evidence_logp = logits,
            categorical_evidence_logp_grad = logits_grad
        )

        pc.mini_batch_em(step_size = 0.1, pseudocount = 1e-6)

        input_layer = pc.input_layer_group[0]
        params = input_layer.params.reshape(num_vars, num_cats)

        # assert torch.all(torch.abs(params.sum(dim = 1) - 1.0) < 1e-6)

        import pdb; pdb.set_trace()
        a = 3


if __name__ == "__main__":
    torch.manual_seed(4343442)
    torch.cuda.manual_seed(5434)
    # test_soft_evidence_categorical_dist()
    # test_soft_evidence_categorical_dist_varied()
    # test_soft_evidence_categorical_dist_multi_nodes()
    # test_soft_evidence_categorical_dist_sample()
    # test_soft_evidence_categorical_dist_dual_flow()
    # test_soft_evidence_categorical_dist_dual_flow_filtered()
    test_soft_evidence_categorical_dist_dual_flow_em()
