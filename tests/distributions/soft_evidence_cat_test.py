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


def test_soft_evidence_categorical_dist_dual_flow_topk():
    # Guards the F- (denominator) buffer offset in `bk_softevi_kernel` for the *true* top-k path,
    # i.e. when the number of candidate ids `k` is strictly smaller than `num_cats`. The dual-flow
    # param-flow buffer is laid out per node as [F+ over num_cats | F- over num_cats], so F- starts
    # at offset `num_cats`. A previous bug used the top-k tile width `k` as that offset, scattering
    # the denominator flows into the wrong slots (the existing `_filtered` test uses ids == arange,
    # i.e. k == num_cats, so it could not catch this).

    torch.manual_seed(230)

    device = torch.device("cuda:0")

    batch_size = 16
    num_vars = 8
    num_cats = 300   # > 256 to also exercise the large-tile backward path
    k = 32           # k < num_cats: the actual top-k regime

    # Construct PC
    nis = [
        juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)) for v in range(num_vars)
    ]
    np = juice.multiply(*nis)
    ns = juice.summate(np, num_nodes = 1)

    ns.init_parameters(perturbation = 2.0)

    pc = juice.compile(ns)
    pc.to(device)

    # Inputs: candidate id sets that contain the ground-truth token and are distinct per (b, v)
    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
    logp = torch.rand([batch_size, num_vars, k], device = device) * 2 - 1

    rnd = torch.rand([batch_size, num_vars, num_cats], device = device)
    rnd.scatter_(2, data.unsqueeze(2), 2.0)  # force the ground-truth token into the top-k
    ids = rnd.topk(k, dim = 2).indices.contiguous()

    ###########################
    ## Forward pass runtests ##
    ###########################

    lls = pc(
        data,
        categorical_evidence_logp = logp,
        soft_evidence_cat_ids = ids
    )

    ############################
    ## Backward pass runtests ##
    ############################

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        categorical_evidence_logp = logp,
        soft_evidence_cat_ids = ids
    )

    input_layer = pc.input_layer_group[0]
    var_order = input_layer.vids
    params = input_layer.params.reshape(num_vars, num_cats)

    sid, eid = input_layer._output_ind_range
    flows = pc.node_flows[sid:eid,:].reshape(num_vars, batch_size).permute(1, 0)  # [B, num_vars(layer order)]

    pflows = input_layer.param_flows.reshape(num_vars, num_cats * 2)
    pflows_num = pflows[:,:num_cats]
    pflows_denom = pflows[:,num_cats:]

    # Reference forward LL + numerator/denominator flows, restricting the local partition Z to the
    # candidate set (the kernel only normalizes over the provided soft-evidence ids).
    p = logp.exp()
    ref_lls = torch.zeros(batch_size, device = device)
    ref_num = torch.zeros_like(pflows_num)
    ref_denom = torch.zeros_like(pflows_denom)
    for i in range(num_vars):
        v = int(var_order[i, 0])
        beta = params[i]
        for b in range(batch_size):
            cands = ids[b, v]                # [k]
            pw = p[b, v]                     # [k]
            beta_c = beta[cands]             # [k]
            Z = (beta_c * pw).sum()

            d = int(data[b, v])
            slot = int((cands == d).nonzero()[0, 0])
            ref_lls[b] += torch.log(beta[d] * pw[slot] / Z)

            f = flows[b, i].exp()
            ref_num[i, d] += f
            ref_denom[i, cands] += f * beta_c * pw / Z

    assert torch.all(torch.abs(lls.view(-1) - ref_lls) < 1e-3)
    assert torch.all(torch.abs(pflows_num - ref_num) < 1e-4)
    assert torch.all(torch.abs(pflows_denom - ref_denom) < 1e-4)


def test_soft_evidence_categorical_dist_dual_flow_em_uniform():
    # Guards the principled MAP denominator (F- + pseudocount * beta). With uniform soft evidence
    # (constant logits => p_theta uniform => Z_n = 1, F- = beta * Gamma), the dual-flow EM update
    # must reduce EXACTLY to the standard categorical MAP update (F+ + pc/K) / (sum F+ + pc).

    torch.manual_seed(99)

    device = torch.device("cuda:0")

    batch_size = 32
    num_vars = 8

    for num_cats in [200, 400]:   # small (<=256) and large (>256) EM kernel paths

        nis = [
            juice.inputs(v, num_nodes = 1, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)) for v in range(num_vars)
        ]
        np = juice.multiply(*nis)
        ns = juice.summate(np, num_nodes = 1)

        ns.init_parameters(perturbation = 2.0)

        pc = juice.compile(ns)
        pc.to(device)

        data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
        logits = torch.zeros([batch_size, num_vars, num_cats], device = device)  # uniform evidence

        pc(data, categorical_evidence_logp = logits)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, categorical_evidence_logp = logits)

        step_size = 1.0
        pseudocount = 0.1

        input_layer = pc.input_layer_group[0]
        pflows = input_layer.param_flows.reshape(num_vars, num_cats * 2)
        F_pos = pflows[:,:num_cats]
        ref_params = (F_pos + pseudocount / num_cats) / (F_pos.sum(dim = 1, keepdim = True) + pseudocount)

        pc.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

        updated_params = input_layer.params.reshape(num_vars, num_cats)

        assert torch.all(torch.abs(updated_params - ref_params) < 1e-4)


def test_soft_evidence_categorical_dist_dual_flow_em_monotonic():
    # Guards the dual-flow EM end to end (offset + MAP denominator + padding-lane handling). By the
    # EM theorem a correct full-batch M-step is non-decreasing in the LL of a fixed dataset; we check
    # monotonic ascent across small/large num_cats and full-vocab/top-k evidence (peaked, like a dLLM).

    torch.manual_seed(1234)

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 8
    num_latents = 4

    for num_cats, k in [(200, None), (200, 64), (400, None), (400, 64)]:

        nis = [
            juice.inputs(v, num_nodes = num_latents, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)) for v in range(num_vars)
        ]
        np = juice.multiply(*nis)
        ns = juice.summate(np, num_nodes = 1)

        ns.init_parameters(perturbation = 2.0)

        pc = juice.compile(ns)
        pc.to(device)

        data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
        evi_full = torch.log_softmax(8.0 * torch.randn([batch_size, num_vars, num_cats], device = device), dim = 2)

        if k is None:
            kwargs = dict(categorical_evidence_logp = evi_full)
        else:
            vals, ids = evi_full.topk(k, dim = 2)
            xid = data.unsqueeze(2)
            miss = ~(ids == xid).any(dim = 2, keepdim = True)
            ids[..., -1:] = torch.where(miss, xid, ids[..., -1:])
            vals[..., -1:] = torch.where(miss, evi_full.gather(2, xid), vals[..., -1:])
            kwargs = dict(categorical_evidence_logp = vals.contiguous(), soft_evidence_cat_ids = ids.contiguous())

        pc.zero_param_flows()
        prev_ll = None
        for step in range(8):
            ll = pc(data, **kwargs).mean().item()

            assert not math.isnan(ll)
            if prev_ll is not None:
                assert ll >= prev_ll - 1e-3, f"EM not monotonic (num_cats={num_cats}, k={k}): {prev_ll:.4f} -> {ll:.4f}"
            prev_ll = ll

            pc.backward(data, allow_modify_flows = False, logspace_flows = True, **kwargs)
            pc.mini_batch_em(step_size = 1.0, pseudocount = 1e-4)
            pc.zero_param_flows()


def test_soft_evidence_categorical_dist_dual_flow_em_momentum_no_nan():
    # Guards the numerical floor in the dual EM. The MAP denominator self-floors only when F- is
    # computed fresh from the current beta; flow momentum smears a stale (larger) F- over a fast-
    # collapsing beta, which underflows float32 -> NaN without the floor. This mirrors the training
    # loop's bias-corrected flow momentum on the (packed) flow buffers, with peaked top-k evidence.

    torch.manual_seed(77)

    device = torch.device("cuda:0")

    batch_size = 64
    num_vars = 8
    num_latents = 4
    num_cats = 400   # large EM path
    k = 64
    momentum = 0.9

    nis = [
        juice.inputs(v, num_nodes = num_latents, dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = True)) for v in range(num_vars)
    ]
    np = juice.multiply(*nis)
    ns = juice.summate(np, num_nodes = 1)

    ns.init_parameters(perturbation = 2.0)

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)
    evi_full = torch.log_softmax(10.0 * torch.randn([batch_size, num_vars, num_cats], device = device), dim = 2)  # very peaked
    vals, ids = evi_full.topk(k, dim = 2)
    xid = data.unsqueeze(2)
    miss = ~(ids == xid).any(dim = 2, keepdim = True)
    ids[..., -1:] = torch.where(miss, xid, ids[..., -1:])
    vals[..., -1:] = torch.where(miss, evi_full.gather(2, xid), vals[..., -1:])
    kwargs = dict(categorical_evidence_logp = vals.contiguous(), soft_evidence_cat_ids = ids.contiguous())

    pc.zero_param_flows()
    mflows = {"sum": torch.zeros(pc.param_flows.size(), dtype = torch.float32, device = device)}
    for i, layer in enumerate(pc.input_layer_group):
        mflows[i] = torch.zeros(layer.param_flows.size(), dtype = torch.float32, device = device)

    for step in range(20):
        pc(data, **kwargs)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, **kwargs)
        pc._cum_flow = batch_size

        with torch.no_grad():
            pc.param_flows.mul_(1.0 - momentum)
            mflows["sum"].mul_(momentum); mflows["sum"].add_(pc.param_flows)
            pc.param_flows[:] = mflows["sum"]
            pc.param_flows.div_(1.0 - math.pow(momentum, step + 1))
            for i, layer in enumerate(pc.input_layer_group):
                layer.param_flows.mul_(1.0 - momentum)
                mflows[i].mul_(momentum); mflows[i].add_(layer.param_flows)
                layer.param_flows[:] = mflows[i]
                layer.param_flows.div_(1.0 - math.pow(momentum, step + 1))

        pc.mini_batch_em(step_size = 0.8, pseudocount = 1e-6, step_size_rescaling = True)
        pc.zero_param_flows()

        for layer in pc.input_layer_group:
            assert not torch.isnan(layer.params).any()
            assert torch.all(layer.params >= 0.0)

    assert not math.isnan(pc(data, **kwargs).mean().item())


if __name__ == "__main__":
    torch.manual_seed(4343442)
    torch.cuda.manual_seed(5434)
    test_soft_evidence_categorical_dist()
    test_soft_evidence_categorical_dist_varied()
    test_soft_evidence_categorical_dist_multi_nodes()
    test_soft_evidence_categorical_dist_sample()
    test_soft_evidence_categorical_dist_dual_flow()
    test_soft_evidence_categorical_dist_dual_flow_filtered()
    test_soft_evidence_categorical_dist_dual_flow_topk()
    test_soft_evidence_categorical_dist_dual_flow_em_uniform()
    test_soft_evidence_categorical_dist_dual_flow_em_monotonic()
    test_soft_evidence_categorical_dist_dual_flow_em_momentum_no_nan()
