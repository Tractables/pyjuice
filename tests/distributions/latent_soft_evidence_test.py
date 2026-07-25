import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists

import pytest


def _build_pc(num_latents, num_cats, emission_dist, seed = 42):
    """
    A single-block-position PC:

        summate( multiply( emission(var 0), LatentSoftEvidence(var 1) ) )

    The `multiply` is index-wise by default, so product node `i` pairs emission node `i` with latent
    node `i` -- the state-aligned wiring the latent-evidence channel assumes.
    """
    torch.manual_seed(seed)

    ni_emit = juice.inputs(0, num_nodes = num_latents, dist = emission_dist)
    ni_lat = juice.inputs(1, num_nodes = num_latents, dist = dists.LatentSoftEvidence())

    np_ = juice.multiply(ni_emit, ni_lat)
    ns = juice.summate(np_, num_nodes = 1)

    ns.init_parameters(perturbation = 2.0)

    return ns, ni_emit, ni_lat


def _log_w(ns, device):
    return ns.get_params(as_matrix = True).view(-1).log().to(device)


def _log_beta(ni_emit, num_latents, num_cats, device):
    return ni_emit._params.view(num_latents, num_cats).log().to(device)


########################################
## HMM with external latent evidence  ##
########################################

# The structural use the latent-evidence channel is built for (spec Option B): at every block position
# `v` of an HMM, a `LatentSoftEvidence` node is multiplied into the *latent* branch, state-aligned with
# the transition sum node, exactly the way `GeneralizedHMM` builds its chain. Emission variables are
# `0 .. T-1`; the latent-evidence variable of position `v` is `T + v`, so the whole chain shares one
# latent-evidence input layer with `ext_num_vars == T`.
HMM_SEQ_LENGTH = 5
HMM_NUM_LATENTS = 16
HMM_BLOCK_SIZE = 4
HMM_NUM_CATS = 6


def _build_hmm(emission_dist_fn, with_latent_evidence = True, seed = 1234,
               seq_length = HMM_SEQ_LENGTH, num_latents = HMM_NUM_LATENTS,
               block_size = HMM_BLOCK_SIZE, num_cats = HMM_NUM_CATS):
    """
    A homogeneous HMM mirroring `juice.structures.GeneralizedHMM`, optionally with a
    `LatentSoftEvidence` node multiplied into the latent branch at every position.

    Returns `(root, alpha, beta, gamma)` with the transition/emission/init parameters as plain torch
    tensors, so the reference below never has to reverse-engineer the compiled parameter layout.
    """
    torch.manual_seed(seed)

    num_node_blocks = num_latents // block_size

    with juice.set_block_size(block_size = block_size):
        ns_input = juice.inputs(seq_length - 1, num_node_blocks = num_node_blocks,
                                dist = emission_dist_fn())

        def _latent_ns(v):
            return juice.inputs(seq_length + v, num_node_blocks = num_node_blocks,
                                dist = dists.LatentSoftEvidence())

        if with_latent_evidence:
            curr_zs = juice.multiply(ns_input, _latent_ns(seq_length - 1))
        else:
            curr_zs = ns_input

        ns_sum = None
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = True)

            if ns_sum is None:
                ns_sum = juice.summate(curr_zs, num_node_blocks = num_node_blocks)
                ns = ns_sum
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params = True)

            if with_latent_evidence:
                curr_zs = juice.multiply(curr_xs, ns, _latent_ns(var))
            else:
                curr_zs = juice.multiply(curr_xs, ns)

        root = juice.summate(curr_zs, num_node_blocks = 1, block_size = 1)

    # Initialize everything first -- `SumNodes.init_parameters` overwrites `_params` unconditionally,
    # so the explicit parameters have to be set afterwards.
    root.init_parameters(perturbation = 2.0)

    alpha = torch.rand([num_latents, num_latents])
    alpha /= alpha.sum(dim = 1, keepdim = True)

    beta = torch.rand([num_latents, num_cats])
    beta /= beta.sum(dim = 1, keepdim = True)

    gamma = torch.rand([num_latents])
    gamma /= gamma.sum()

    ns_input.set_params(beta)
    ns_sum.set_params(alpha)
    root.set_params(gamma.unsqueeze(0))

    return root, alpha, beta, gamma


def _hmm_reference_lls(log_emit, alpha, gamma, E, seq_length = HMM_SEQ_LENGTH):
    """
    Forward recursion of the HMM the builder above defines:

        V_{T-1}[b,j] = log_emit[b, T-1, j] + E[b, T-1, j]
        V_v[b,j]     = log_emit[b, v, j] + logsumexp_i( log alpha[j,i] + V_{v+1}[b,i] ) + E[b, v, j]
        ll[b]        = logsumexp_j( log gamma[j] + V_0[b,j] )

    `log_emit[b, v, j]` is the value of emission node `j` at position `v` (whatever the emission
    distribution computes), and `E` the additive latent log-potential -- pass zeros to switch it off.
    """
    log_alpha = alpha.log()
    log_gamma = gamma.log()

    V = log_emit[:,seq_length - 1,:] + E[:,seq_length - 1,:]
    for v in range(seq_length - 2, -1, -1):
        trans = torch.logsumexp(log_alpha[None,:,:] + V[:,None,:], dim = 2)
        V = log_emit[:,v,:] + trans + E[:,v,:]

    return torch.logsumexp(log_gamma[None,:] + V, dim = 1)


def _categorical_log_emit(beta, data, seq_length = HMM_SEQ_LENGTH):
    """`log beta_j(x_v)` for a plain `Categorical` emission -- [B, T, num_latents]."""
    return beta.log()[None,None,:,:].expand(data.size(0), seq_length, -1, -1).gather(
        3, data[:,:seq_length,None,None].expand(-1, -1, beta.size(0), -1)).squeeze(3)


def _latent_input_layer(pc):
    for layer in pc.input_layer_group:
        if isinstance(layer.nodes[0].dist, dists.LatentSoftEvidence):
            return layer

    raise AssertionError("No `LatentSoftEvidence` input layer found.")


def test_hmm_latent_soft_evidence_forward():
    """HMM forward with per-position latent evidence matches a torch forward recursion."""

    device = torch.device("cuda:0")

    batch_size = 16
    T, L, C = HMM_SEQ_LENGTH, HMM_NUM_LATENTS, HMM_NUM_CATS

    root, alpha, beta, gamma = _build_hmm(lambda: dists.Categorical(num_cats = C))

    pc = juice.compile(root)
    pc.to(device)

    alpha, beta, gamma = alpha.to(device), beta.to(device), gamma.to(device)

    data = torch.randint(0, C, [batch_size, 2 * T], device = device)
    latent_evidence_logp = torch.randn([batch_size, T, L], device = device).log_softmax(dim = 2).contiguous()

    lls = pc(data, latent_evidence_logp = latent_evidence_logp)

    target_lls = _hmm_reference_lls(_categorical_log_emit(beta, data), alpha, gamma, latent_evidence_logp)

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ## The latent layer's `node_mars` are the supplied potentials, position- and state-aligned ##

    layer = _latent_input_layer(pc)
    sid, eid = layer._output_ind_range

    var_order = layer.vids[::L].view(-1) - T # layer slot -> position
    mars = pc.node_mars[sid:eid,:].reshape(T, L, batch_size).permute(2, 0, 1) # [B, slot, state]

    for i in range(T):
        assert torch.all(torch.abs(mars[:,i,:] - latent_evidence_logp[:,var_order[i],:]) < 1e-5)


@pytest.mark.parametrize("logspace_flows", [True, False])
def test_hmm_latent_soft_evidence_grad(logspace_flows):
    """The HMM's latent-evidence gradient matches autograd through the reference recursion."""

    device = torch.device("cuda:0")

    batch_size = 16
    T, L, C = HMM_SEQ_LENGTH, HMM_NUM_LATENTS, HMM_NUM_CATS

    root, alpha, beta, gamma = _build_hmm(lambda: dists.Categorical(num_cats = C))

    pc = juice.compile(root)
    pc.to(device)

    alpha, beta, gamma = alpha.to(device), beta.to(device), gamma.to(device)

    data = torch.randint(0, C, [batch_size, 2 * T], device = device)
    latent_evidence_logp = torch.randn([batch_size, T, L], device = device).log_softmax(dim = 2).contiguous()

    pc(data, latent_evidence_logp = latent_evidence_logp)

    latent_evidence_logp_grad = torch.zeros_like(latent_evidence_logp)
    pc.backward(
        data, logspace_flows = logspace_flows,
        latent_evidence_logp = latent_evidence_logp,
        latent_evidence_logp_grad = latent_evidence_logp_grad
    )

    E = latent_evidence_logp.clone().requires_grad_(True)
    target_lls = _hmm_reference_lls(_categorical_log_emit(beta, data), alpha, gamma, E)
    target_grad, = torch.autograd.grad(target_lls.sum(), E)

    assert torch.all(torch.abs(latent_evidence_logp_grad - target_grad) < 1e-4)

    # The flow of a position is a proper posterior over latent states, so it sums to 1 per (b, v)
    assert torch.all(torch.abs(latent_evidence_logp_grad.sum(dim = 2) - 1.0) < 1e-4)


def test_hmm_latent_soft_evidence_coexists_with_soft_evidence_categorical():
    """
    The target configuration: a `SoftEvidenceCategorical` emission carrying `p_theta(x)` and a
    `LatentSoftEvidence` node carrying `p_theta(z)` at every position of one HMM, one forward.
    """

    device = torch.device("cuda:0")

    batch_size = 16
    T, L, C = HMM_SEQ_LENGTH, HMM_NUM_LATENTS, HMM_NUM_CATS

    root, alpha, beta, gamma = _build_hmm(lambda: dists.SoftEvidenceCategorical(num_cats = C))

    pc = juice.compile(root)
    pc.to(device)

    alpha, beta, gamma = alpha.to(device), beta.to(device), gamma.to(device)

    data = torch.randint(0, C, [batch_size, 2 * T], device = device)
    categorical_evidence_logp = torch.randn([batch_size, T, C], device = device).log_softmax(dim = 2).contiguous()
    latent_evidence_logp = torch.randn([batch_size, T, L], device = device).log_softmax(dim = 2).contiguous()

    lls = pc(
        data,
        categorical_evidence_logp = categorical_evidence_logp,
        latent_evidence_logp = latent_evidence_logp
    )

    # `SoftEvidenceCategorical` node `j` at position `v`: log beta_j(d) + log p_theta(d) - logZ_{v,j}
    log_beta_d = _categorical_log_emit(beta, data) # [B, T, L]
    log_ptheta_d = torch.gather(categorical_evidence_logp, 2, data[:,:T,None]) # [B, T, 1]
    logZ = torch.logsumexp(beta.log()[None,None,:,:] + categorical_evidence_logp[:,:,None,:], dim = 3) # [B, T, L]

    log_emit = log_beta_d + log_ptheta_d - logZ

    target_lls = _hmm_reference_lls(log_emit, alpha, gamma, latent_evidence_logp)

    # `SoftEvidenceCategorical`'s leaf-local logZ carries a ~7e-4 fp32/tensor-core bias per node (the
    # same magnitude `codd_hmm_test` allows for on per-node mars), which accumulates additively down
    # the chain -- hence a per-position budget. A state or position misalignment is O(1) either way.
    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1.5e-3 * T)

    # The latent-evidence leaf itself contributes no error of its own
    layer = _latent_input_layer(pc)
    sid, eid = layer._output_ind_range

    var_order = layer.vids[::L].view(-1) - T
    mars = pc.node_mars[sid:eid,:].reshape(T, L, batch_size).permute(2, 0, 1)

    for i in range(T):
        assert torch.equal(mars[:,i,:], latent_evidence_logp[:,var_order[i],:])


def test_hmm_latent_soft_evidence_off_matches_plain_hmm():
    """
    An HMM carrying latent-evidence nodes but run without `latent_evidence_logp` is the plain HMM: the
    latent channel can be ablated without rebuilding the circuit.
    """

    device = torch.device("cuda:0")

    batch_size = 16
    T, L, C = HMM_SEQ_LENGTH, HMM_NUM_LATENTS, HMM_NUM_CATS

    root_lat, alpha, beta, gamma = _build_hmm(lambda: dists.Categorical(num_cats = C), with_latent_evidence = True)
    root_plain, alpha_p, beta_p, gamma_p = _build_hmm(lambda: dists.Categorical(num_cats = C), with_latent_evidence = False)

    # Same seed in the builder => same parameters; assert it rather than trust it
    assert torch.equal(alpha, alpha_p) and torch.equal(beta, beta_p) and torch.equal(gamma, gamma_p)

    pc_lat = juice.compile(root_lat)
    pc_lat.to(device)

    pc_plain = juice.compile(root_plain)
    pc_plain.to(device)

    data = torch.randint(0, C, [batch_size, 2 * T], device = device)

    lls_lat = pc_lat(data)
    lls_plain = pc_plain(data[:,:T])

    assert torch.all(torch.abs(lls_lat.view(-1) - lls_plain.view(-1)) < 1e-4)

    # ... and supplying zero evidence is the same thing
    lls_zero = pc_lat(data, latent_evidence_logp = torch.zeros([batch_size, T, L], device = device))

    assert torch.all(torch.abs(lls_lat.view(-1) - lls_zero.view(-1)) < 1e-5)


def test_hmm_latent_soft_evidence_does_not_perturb_em():
    """
    On the real (tied-parameter) HMM: requesting the latent-evidence gradient leaves every parameter
    flow bit-identical, and a full EM step lands on the same parameters.
    """

    device = torch.device("cuda:0")

    batch_size = 16
    T, L, C = HMM_SEQ_LENGTH, HMM_NUM_LATENTS, HMM_NUM_CATS

    root, _, _, _ = _build_hmm(lambda: dists.Categorical(num_cats = C))

    pc = juice.compile(root)
    pc.to(device)

    data = torch.randint(0, C, [batch_size, 2 * T], device = device)
    latent_evidence_logp = torch.randn([batch_size, T, L], device = device).log_softmax(dim = 2).contiguous()

    # EM updates the input-layer parameters too, so both have to be restored for the two runs to start
    # from an identical state
    params_before = pc.params.clone()
    input_params_before = [layer.params.clone() for layer in pc.input_layer_group]

    def _run(with_grad):
        pc.params[:] = params_before
        for layer, saved in zip(pc.input_layer_group, input_params_before):
            layer.params[:] = saved

        pc.init_param_flows(flows_memory = 0.0)

        pc(data, latent_evidence_logp = latent_evidence_logp)

        kwargs = dict()
        if with_grad:
            kwargs["latent_evidence_logp_grad"] = torch.zeros_like(latent_evidence_logp)

        pc.backward(data, logspace_flows = True, latent_evidence_logp = latent_evidence_logp, **kwargs)

        flows = [pc.param_flows.clone()]
        for layer in pc.input_layer_group:
            if layer.param_flows is not None and layer.param_flows.numel() > 0:
                flows.append(layer.param_flows.clone())

        pc.mini_batch_em(step_size = 0.5, pseudocount = 0.01)

        params = [pc.params.clone()] + [layer.params.clone() for layer in pc.input_layer_group]

        return flows, params

    flows_without, params_without = _run(with_grad = False)
    flows_with, params_with = _run(with_grad = True)

    assert len(flows_without) == len(flows_with) and len(flows_without) > 1
    for a, b in zip(flows_without, flows_with):
        assert torch.equal(a, b)

    for a, b in zip(params_without, params_with):
        assert torch.equal(a, b)


def test_latent_soft_evidence_forward():
    """Forward matches `logsumexp_i(log w_i + log beta_i(x) + E[b,i])`, incl. `nids` alignment."""

    device = torch.device("cuda:0")

    batch_size = 16
    num_latents = 8
    num_cats = 5

    ns, ni_emit, _ = _build_pc(num_latents, num_cats, dists.Categorical(num_cats = num_cats))

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, 2], device = device)

    # Deliberately distinct per-state values: a `nids` mis-alignment would permute these and break the test
    latent_evidence_logp = torch.randn([batch_size, 1, num_latents], device = device).log_softmax(dim = 2).contiguous()

    lls = pc(data, latent_evidence_logp = latent_evidence_logp)

    log_w = _log_w(ns, device)
    log_beta = _log_beta(ni_emit, num_latents, num_cats, device)

    target_lls = (log_w[None,:] + log_beta[:,data[:,0]].permute(1, 0) + latent_evidence_logp[:,0,:]).logsumexp(dim = 1)

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)


@pytest.mark.parametrize("logspace_flows", [True, False])
def test_latent_soft_evidence_grad(logspace_flows):
    """`latent_evidence_logp_grad` holds the LINEAR-space flow under both flow storage modes."""

    device = torch.device("cuda:0")

    batch_size = 16
    num_latents = 8
    num_cats = 5

    ns, ni_emit, _ = _build_pc(num_latents, num_cats, dists.Categorical(num_cats = num_cats))

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, 2], device = device)
    latent_evidence_logp = torch.randn([batch_size, 1, num_latents], device = device).log_softmax(dim = 2).contiguous()

    pc(data, latent_evidence_logp = latent_evidence_logp)

    latent_evidence_logp_grad = torch.zeros_like(latent_evidence_logp)
    pc.backward(
        data, logspace_flows = logspace_flows,
        latent_evidence_logp = latent_evidence_logp,
        latent_evidence_logp_grad = latent_evidence_logp_grad
    )

    ## Closed form: the flow is the posterior responsibility of state `i` ##

    log_w = _log_w(ns, device)
    log_beta = _log_beta(ni_emit, num_latents, num_cats, device)

    logits = log_w[None,:] + log_beta[:,data[:,0]].permute(1, 0) + latent_evidence_logp[:,0,:]
    target_grad = logits.softmax(dim = 1)

    assert torch.all(torch.abs(latent_evidence_logp_grad[:,0,:] - target_grad) < 1e-4)

    ## Central-difference directional derivative of `sum_b log f` ##

    torch.manual_seed(1234)
    direction = torch.randn_like(latent_evidence_logp)
    eps = 1e-2

    ll_plus = pc(data, latent_evidence_logp = (latent_evidence_logp + eps * direction).contiguous()).sum()
    ll_minus = pc(data, latent_evidence_logp = (latent_evidence_logp - eps * direction).contiguous()).sum()

    fd = ((ll_plus - ll_minus) / (2 * eps)).item()
    analytic = (latent_evidence_logp_grad * direction).sum().item()

    assert abs(fd - analytic) / max(abs(fd), 1e-6) < 1e-2


def test_latent_soft_evidence_coexists_with_soft_evidence_categorical():
    """A `SoftEvidenceCategorical` emission and `LatentSoftEvidence` in one PC, one forward."""

    device = torch.device("cuda:0")

    batch_size = 16
    num_latents = 8
    num_cats = 5

    ns, ni_emit, _ = _build_pc(num_latents, num_cats, dists.SoftEvidenceCategorical(num_cats = num_cats))

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, 2], device = device)

    categorical_evidence_logp = torch.randn([batch_size, 1, num_cats], device = device).log_softmax(dim = 2).contiguous()
    latent_evidence_logp = torch.randn([batch_size, 1, num_latents], device = device).log_softmax(dim = 2).contiguous()

    lls = pc(
        data,
        categorical_evidence_logp = categorical_evidence_logp,
        latent_evidence_logp = latent_evidence_logp
    )

    log_w = _log_w(ns, device)
    log_beta = _log_beta(ni_emit, num_latents, num_cats, device)

    # `SoftEvidenceCategorical` emission node `i`: log beta_i(d) + log p_theta(d) - logZ_i
    logZ = (log_beta[None,:,:] + categorical_evidence_logp[:,0,None,:]).logsumexp(dim = 2) # [B, num_latents]
    log_emit = log_beta[:,data[:,0]].permute(1, 0) + \
        torch.gather(categorical_evidence_logp[:,0,:], 1, data[:,0:1]) - logZ

    target_lls = (log_w[None,:] + log_emit + latent_evidence_logp[:,0,:]).logsumexp(dim = 1)

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)


def test_latent_soft_evidence_survives_missing_mask():
    """Marginalizing the emission variable keeps the latent evidence in the marginal."""

    device = torch.device("cuda:0")

    batch_size = 16
    num_latents = 8
    num_cats = 5

    ns, _, _ = _build_pc(num_latents, num_cats, dists.Categorical(num_cats = num_cats))

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, 2], device = device)
    latent_evidence_logp = torch.randn([batch_size, 1, num_latents], device = device).log_softmax(dim = 2).contiguous()

    # Marginalize the emission variable (var 0); the latent-evidence variable (var 1) is never masked
    missing_mask = torch.tensor([True, False], device = device)

    lls_marg = pc(data, missing_mask = missing_mask, latent_evidence_logp = latent_evidence_logp)

    log_w = _log_w(ns, device)
    target_lls = (log_w[None,:] + latent_evidence_logp[:,0,:]).logsumexp(dim = 1)

    assert torch.all(torch.abs(lls_marg.view(-1) - target_lls) < 1e-3)

    # The latent evidence still makes the marginal batch-dependent
    assert lls_marg.view(-1).std().item() > 1e-3

    # Masking a latent-evidence variable is not intended usage, but must drop its evidence rather than
    # crash: the node falls back to 0, leaving `logsumexp_j(log w_j) == 0`
    lls_all_marg = pc(
        data,
        missing_mask = torch.tensor([True, True], device = device),
        latent_evidence_logp = latent_evidence_logp
    )

    assert torch.all(torch.abs(lls_all_marg.view(-1)) < 1e-3)


def test_latent_soft_evidence_does_not_perturb_param_flows():
    """Requesting the latent-evidence gradient leaves the PC's parameter flows bit-identical."""

    device = torch.device("cuda:0")

    batch_size = 16
    num_latents = 8
    num_cats = 5

    ns, _, _ = _build_pc(num_latents, num_cats, dists.Categorical(num_cats = num_cats))

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, 2], device = device)
    latent_evidence_logp = torch.randn([batch_size, 1, num_latents], device = device).log_softmax(dim = 2).contiguous()

    def _run(with_grad):
        pc.init_param_flows(flows_memory = 0.0)
        pc(data, latent_evidence_logp = latent_evidence_logp)

        kwargs = dict()
        if with_grad:
            kwargs["latent_evidence_logp_grad"] = torch.zeros_like(latent_evidence_logp)

        pc.backward(data, logspace_flows = True, latent_evidence_logp = latent_evidence_logp, **kwargs)

        flows = [pc.param_flows.clone()]
        for layer in pc.input_layer_group:
            if layer.param_flows is not None and layer.param_flows.numel() > 0:
                flows.append(layer.param_flows.clone())

        return flows

    flows_without = _run(with_grad = False)
    flows_with = _run(with_grad = True)

    assert len(flows_without) == len(flows_with)
    for a, b in zip(flows_without, flows_with):
        assert torch.equal(a, b)


def test_latent_soft_evidence_off_without_evidence():
    """Without `latent_evidence_logp` the latent nodes contribute 0, i.e. the channel is off."""

    device = torch.device("cuda:0")

    batch_size = 16
    num_latents = 8
    num_cats = 5

    ns, ni_emit, _ = _build_pc(num_latents, num_cats, dists.Categorical(num_cats = num_cats))

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, 2], device = device)

    lls = pc(data)

    log_w = _log_w(ns, device)
    log_beta = _log_beta(ni_emit, num_latents, num_cats, device)

    target_lls = (log_w[None,:] + log_beta[:,data[:,0]].permute(1, 0)).logsumexp(dim = 1)

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    # Explicitly-zero evidence is the same thing
    zero_evidence = torch.zeros([batch_size, 1, num_latents], device = device)
    lls_zero = pc(data, latent_evidence_logp = zero_evidence)

    assert torch.all(torch.abs(lls.view(-1) - lls_zero.view(-1)) < 1e-5)


@pytest.mark.parametrize("logspace_flows", [True, False])
def test_latent_soft_evidence_multi_position(logspace_flows):
    """
    Several block positions sharing one latent-evidence input layer, so the kernels have to resolve
    `ext_num_vars > 1` through `var_idmapping`. Forward and gradient are both checked against a torch
    reference (the gradient via autograd through that reference).
    """

    device = torch.device("cuda:0")

    batch_size = 16
    num_positions = 3
    num_latents = 32
    num_cats = 4

    torch.manual_seed(7)

    # Position `v` uses variable `2v` for the emission and `2v + 1` for the latent evidence
    ni_emits = [juice.inputs(2 * v, num_nodes = num_latents, dist = dists.Categorical(num_cats = num_cats))
                for v in range(num_positions)]
    ni_lats = [juice.inputs(2 * v + 1, num_nodes = num_latents, dist = dists.LatentSoftEvidence())
               for v in range(num_positions)]

    nss = [juice.summate(juice.multiply(ni_emits[v], ni_lats[v]), num_nodes = num_latents)
           for v in range(num_positions)]

    ns = juice.summate(juice.multiply(*nss), num_nodes = 1)

    ns.init_parameters(perturbation = 2.0)

    pc = juice.compile(ns)
    pc.to(device)

    data = torch.randint(0, num_cats, [batch_size, 2 * num_positions], device = device)
    latent_evidence_logp = torch.randn(
        [batch_size, num_positions, num_latents], device = device).log_softmax(dim = 2).contiguous()

    lls = pc(data, latent_evidence_logp = latent_evidence_logp)

    ## Torch reference ##

    E = latent_evidence_logp.clone().requires_grad_(True)

    log_w = _log_w(ns, device) # [num_latents]

    target_lls = torch.zeros([batch_size, num_latents], device = device)
    for v in range(num_positions):
        log_A = nss[v].get_params(as_matrix = True).log().to(device) # [num_latents, num_latents]
        log_beta = _log_beta(ni_emits[v], num_latents, num_cats, device) # [num_latents, num_cats]

        # log m_v[b, j] = logsumexp_i( log A[j,i] + log beta_i(x_{b,v}) + E[b,v,i] )
        logits = log_A[None,:,:] + log_beta[:,data[:,2 * v]].permute(1, 0)[:,None,:] + E[:,v,None,:]
        target_lls = target_lls + logits.logsumexp(dim = 2)

    target_lls = (log_w[None,:] + target_lls).logsumexp(dim = 1)

    assert torch.all(torch.abs(lls.view(-1) - target_lls) < 1e-3)

    ## Gradient ##

    latent_evidence_logp_grad = torch.zeros_like(latent_evidence_logp)
    pc.backward(
        data, logspace_flows = logspace_flows,
        latent_evidence_logp = latent_evidence_logp,
        latent_evidence_logp_grad = latent_evidence_logp_grad
    )

    target_grad, = torch.autograd.grad(target_lls.sum(), E)

    assert torch.all(torch.abs(latent_evidence_logp_grad - target_grad) < 1e-4)


if __name__ == "__main__":
    test_latent_soft_evidence_forward()
    test_latent_soft_evidence_grad(True)
    test_latent_soft_evidence_grad(False)
    test_latent_soft_evidence_coexists_with_soft_evidence_categorical()
    test_latent_soft_evidence_survives_missing_mask()
    test_latent_soft_evidence_does_not_perturb_param_flows()
    test_latent_soft_evidence_off_without_evidence()
    test_latent_soft_evidence_multi_position(True)
    test_latent_soft_evidence_multi_position(False)
    test_hmm_latent_soft_evidence_forward()
    test_hmm_latent_soft_evidence_grad(True)
    test_hmm_latent_soft_evidence_grad(False)
    test_hmm_latent_soft_evidence_coexists_with_soft_evidence_categorical()
    test_hmm_latent_soft_evidence_off_matches_plain_hmm()
    test_hmm_latent_soft_evidence_does_not_perturb_em()
