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
        input_dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False)
    )
    root_ns.init_parameters(perturbation = 32.0)

    pc = juice.compile(root_ns)
    pc.to(device)

    # Extract parameters
    emit_params = root_ns.chs[0].chs[0].get_source_ns()._params.reshape(num_latents, num_cats).to(device)

    var_order = pc.input_layer_group[0].vids[::num_latents]

    data = torch.randint(0, num_cats, [batch_size, num_vars], device = device)

    categorical_evidence_logp = torch.log_softmax(
        torch.rand([batch_size, num_vars, num_cats], device = device), 
        dim = 2
    )

    ###########################
    ## Forward pass runtests ##
    ###########################

    lls = pc(
        data,
        categorical_evidence_logp = categorical_evidence_logp
    )

    input_layer = pc.input_layer_group[0]
    sid, eid = input_layer._output_ind_range

    mars = pc.node_mars[sid:eid,:].reshape(num_vars, num_latents, batch_size).permute(2, 0, 1)

    for i in range(num_vars):
        v = var_order[i,0]
        
        log_params = categorical_evidence_logp[:,v,:][:,None,:] + emit_params[None,:,:].log()
        unnorm_mars = log_params.gather(2, data[:,v][:,None,None].repeat(1, num_latents, 1)).squeeze()

        logZ = torch.logsumexp(log_params, dim = 2)
        target_mars = unnorm_mars - logZ
        
        v_mars = mars[:,i,:]

        assert torch.all(torch.abs(target_mars - v_mars) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    categorical_evidence_logp_grad = torch.zeros_like(categorical_evidence_logp)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        categorical_evidence_logp = categorical_evidence_logp,
        categorical_evidence_logp_grad = categorical_evidence_logp_grad
    )

    flows = pc.node_flows[sid:eid,:].reshape(num_vars, num_latents, batch_size).permute(2, 0, 1) # [B, V, N]

    pflows = input_layer.param_flows.reshape(-1, num_latents, num_cats).sum(dim = 0)

    my_pflows = torch.zeros_like(pflows)
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_pflows[:,data[b,v]] += flows[b,i,:].exp()

    assert torch.all(torch.abs(pflows - my_pflows) < 1e-5)

    my_grads = torch.zeros_like(categorical_evidence_logp_grad) # [B, V, K]
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_grads[b,v,data[b,v]] += flows[b,i,:].exp().sum()

            log_params = categorical_evidence_logp[b,v,:][None,:] + emit_params.log() # [N, C]
            logZ = torch.logsumexp(log_params, dim = 1) # [N]

            my_grads[b,v,:] -= (flows[b,i,:][:,None] + log_params - logZ[:,None]).exp().sum(dim = 0)

    assert torch.all(torch.abs(categorical_evidence_logp_grad - my_grads) < 1e-4)


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
        input_dist = dists.SoftEvidenceCategorical(num_cats = num_cats, _dual_flow_backward = False)
    )
    root_ns.init_parameters(perturbation = 32.0)

    pc = juice.compile(root_ns)
    pc.to(device)

    # Extract parameters
    emit_params = root_ns.chs[0].chs[0].get_source_ns()._params.reshape(num_latents, num_cats).to(device)

    var_order = pc.input_layer_group[0].vids[::num_latents]

    categorical_evidence_logp = torch.log_softmax(
        torch.rand([batch_size, num_vars, num_cats], device = device), 
        dim = 2
    )

    soft_evidence_cat_ids = torch.multinomial(
        torch.ones(batch_size * num_vars, num_cats, dtype = torch.float32, device = device),
        num_samples = num_selected_cats,
        replacement = False
    ).reshape(batch_size, num_vars, num_selected_cats).contiguous()
    
    categorical_evidence_logp = categorical_evidence_logp.gather(2, soft_evidence_cat_ids).contiguous()

    data = soft_evidence_cat_ids[:,:,0].contiguous()

    ###########################
    ## Forward pass runtests ##
    ###########################

    unnorm_lls = pc(
        data,
        categorical_evidence_logp = categorical_evidence_logp,
        soft_evidence_cat_ids = soft_evidence_cat_ids
    )

    input_layer = pc.input_layer_group[0]
    sid, eid = input_layer._output_ind_range

    mars = pc.node_mars[sid:eid,:].reshape(num_vars, num_latents, batch_size).permute(2, 0, 1)

    for i in range(num_vars):
        v = var_order[i,0]
        
        log_params = categorical_evidence_logp[:,v,:][:,None,:] + emit_params[None,:,:].repeat(batch_size, 1, 1).gather(2, soft_evidence_cat_ids[:,v,:][:,None,:].repeat(1, num_latents, 1)).log()
        unnorm_mars = log_params[:,:,0]

        logZ = torch.logsumexp(log_params, dim = 2)
        target_mars = unnorm_mars - logZ
        
        v_mars = mars[:,i,:]

        assert torch.all(torch.abs(target_mars - v_mars) < 1e-3)

    ############################
    ## Backward pass runtests ##
    ############################

    categorical_evidence_logp_grad = torch.zeros_like(categorical_evidence_logp)

    pc.backward(
        data, allow_modify_flows = False, logspace_flows = True,
        categorical_evidence_logp = categorical_evidence_logp,
        categorical_evidence_logp_grad = categorical_evidence_logp_grad,
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

    my_grads = torch.zeros_like(categorical_evidence_logp_grad) # [B, V, K]
    for i in range(num_vars):
        v = var_order[i,0]

        for b in range(batch_size):
            my_grads[b,v,0] += flows[b,i,:].exp().sum()

            log_params = categorical_evidence_logp[b,v,:][None,:] + emit_params[:,soft_evidence_cat_ids[b,v,:]].log() # [N, C]
            logZ = torch.logsumexp(log_params, dim = 1) # [N]

            my_grads[b,v,:] -= (flows[b,i,:][:,None] + log_params - logZ[:,None]).exp().sum(dim = 0)

    assert torch.all(torch.abs(categorical_evidence_logp_grad - my_grads) < 1e-4)


@pytest.mark.parametrize("dual_flow,use_topk", [(True, True), (True, False), (False, True), (False, False)])
def test_codd_hmm_evidence_grad_finite_diff(dual_flow, use_topk):
    # INDEPENDENT (finite-difference) verification of the dLLM gradient used by the joint-training
    # workflow: d log q(x) / d categorical_evidence_logp, produced by bk_softevi's `update_extflows`.
    #
    # Why this is needed beyond test_codd_hmm_{no,with}_filter: those check the gradient against the
    # SAME analytic formula the kernel implements (self-consistent), and only with _dual_flow_backward
    # = False. The joint-training workflow runs with _dual_flow_backward = True (so `update_extflows`
    # executes ALONGSIDE the F+/F- accumulation in the same kernel) and on the top-k layout where the
    # ground-truth token is forced into the candidate set (as chop_into_blocks does). Here we treat the
    # forward as a black box and check a random-projection directional derivative by central differences,
    # which is an independent ground truth (and has strong float32 signal-to-noise).

    torch.manual_seed(123)
    device = torch.device("cuda:0")

    S, L, V, B, k = 4, 8, 64, 8, 16

    root_ns = juice.structures.GeneralizedHMM(
        seq_length = S, num_latents = L, homogeneous = True,
        input_dist = dists.SoftEvidenceCategorical(num_cats = V, _dual_flow_backward = dual_flow)
    )
    root_ns.init_parameters(perturbation = 2.0)  # non-uniform emissions
    pc = juice.compile(root_ns)
    pc.to(device)

    data = torch.randint(0, V, (B, S), device = device)
    full_logp = torch.log_softmax(torch.randn(B, S, V, device = device), dim = 2)

    if use_topk:
        # top-k candidates with the ground-truth token forced into the last slot (chop_into_blocks)
        vals, ids = full_logp.topk(k, dim = 2)
        xid = data.unsqueeze(2)
        miss = ~(ids == xid).any(dim = 2, keepdim = True)
        ids = torch.cat([ids[..., :-1], torch.where(miss, xid, ids[..., -1:])], dim = 2).long().contiguous()
        vals = torch.cat([vals[..., :-1], torch.where(miss, full_logp.gather(2, xid), vals[..., -1:])], dim = 2)
        evi0 = vals.contiguous()
        kw = dict(soft_evidence_cat_ids = ids)
    else:
        evi0 = full_logp.contiguous()
        kw = dict()

    def logq_sum(evi):
        with torch.cuda.device(device):
            return float(pc(data, categorical_evidence_logp = evi.contiguous(), **kw).sum())

    # analytic gradient from the kernel (the value the workflow backprops to the dLLM)
    grad = torch.zeros_like(evi0)
    with torch.cuda.device(device):
        pc(data, categorical_evidence_logp = evi0, **kw)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True,
                    categorical_evidence_logp = evi0, categorical_evidence_logp_grad = grad, **kw)

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    # central-difference directional derivative along a random direction
    torch.manual_seed(7)
    direction = torch.randn_like(evi0)
    eps = 1e-3
    fd = (logq_sum(evi0 + eps * direction) - logq_sum(evi0 - eps * direction)) / (2.0 * eps)
    analytic = float((grad * direction).sum())

    rel_err = abs(fd - analytic) / (abs(analytic) + 1e-6)
    assert rel_err < 1e-2, f"dual_flow={dual_flow} topk={use_topk}: FD {fd:.5f} vs analytic {analytic:.5f} (rel err {rel_err:.2e})"


def test_codd_hmm_evidence_grad_dualflow_matches_singleflow():
    # The evidence gradient must NOT depend on whether F-/F+ are also being accumulated, i.e. it must be
    # identical for _dual_flow_backward True vs False (update_extflows is independent of update_pflows).
    # This directly validates the joint-training config (True) against the well-tested config (False).

    torch.manual_seed(321)
    device = torch.device("cuda:0")

    S, L, V, B, k = 4, 8, 80, 8, 24

    data = torch.randint(0, V, (B, S), device = device)
    full_logp = torch.log_softmax(torch.randn(B, S, V, device = device), dim = 2)
    vals, ids = full_logp.topk(k, dim = 2)
    xid = data.unsqueeze(2)
    miss = ~(ids == xid).any(dim = 2, keepdim = True)
    ids = torch.cat([ids[..., :-1], torch.where(miss, xid, ids[..., -1:])], dim = 2).long().contiguous()
    vals = torch.cat([vals[..., :-1], torch.where(miss, full_logp.gather(2, xid), vals[..., -1:])], dim = 2).contiguous()

    def grad_for(dual_flow, params_ref = None):
        root_ns = juice.structures.GeneralizedHMM(
            seq_length = S, num_latents = L, homogeneous = True,
            input_dist = dists.SoftEvidenceCategorical(num_cats = V, _dual_flow_backward = dual_flow)
        )
        root_ns.init_parameters(perturbation = 2.0)
        pc = juice.compile(root_ns)
        pc.to(device)
        # force identical parameters across the two PCs
        if params_ref is None:
            params_ref = (pc.params.clone(), pc.input_layer_group[0].params.clone())
        else:
            with torch.no_grad():
                pc.params[:] = params_ref[0]
                pc.input_layer_group[0].params[:] = params_ref[1]
        g = torch.zeros_like(vals)
        with torch.cuda.device(device):
            pc(data, categorical_evidence_logp = vals, soft_evidence_cat_ids = ids)
            pc.backward(data, allow_modify_flows = False, logspace_flows = True,
                        categorical_evidence_logp = vals, soft_evidence_cat_ids = ids,
                        categorical_evidence_logp_grad = g)
        return g, params_ref

    g_false, ref = grad_for(False)
    g_true, _ = grad_for(True, params_ref = ref)

    assert torch.all(torch.abs(g_false - g_true) < 1e-5)


if __name__ == "__main__":
    test_codd_hmm_no_filter()
    test_codd_hmm_with_filter()
    for df in (True, False):
        for tk in (True, False):
            test_codd_hmm_evidence_grad_finite_diff(df, tk)
    test_codd_hmm_evidence_grad_dualflow_matches_singleflow()
