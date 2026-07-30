import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, LowRankSumParams
from pyjuice.layer import ExternalParamsSumLayer

import pytest


SEQ_LENGTH = 6
NUM_LATENTS = 32
NUM_EMITS = 8
RANK = 4
BATCH_SIZE = 4


def _build_hmm(external: bool, seed: int = 0):
    """
    A homogeneous HMM, built exactly as `pyjuice.structures.HMM` does, except that the transition may
    carry an external low-rank parameterization.

    The transition is a dense `num_latents x num_latents` sum node at `block_size = num_latents`, so it
    has a single edge block -- and `LowRankSumParams`' per-edge-block layout collapses to the familiar
    `[B, S, r]` (with a singleton edge-block axis). Every timestep after the first is a TIED duplicate,
    so they all share one parameter block but compile into their own layer.
    """

    torch.manual_seed(seed)

    ext = dict(external_params = LowRankSumParams(rank = RANK)) if external else dict()

    with juice.set_block_size(NUM_LATENTS):

        ns_input = inputs(SEQ_LENGTH - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))

        ns_sum, curr_zs = None, ns_input
        for var in range(SEQ_LENGTH - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = True)

            if ns_sum is None:
                ns_sum = summate(curr_zs, num_node_blocks = 1, **ext)
                ns = ns_sum
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params = True)

            curr_zs = multiply(curr_xs, ns)

        root_ns = summate(curr_zs, num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root_ns.init_parameters(perturbation = 2.0)

    return root_ns, ns_sum


def _compiled(external: bool, seed: int = 0):
    root_ns, trans_ns = _build_hmm(external, seed = seed)
    pc = juice.compile(root_ns)
    pc.to(torch.device("cuda:0"))
    return pc, root_ns, trans_ns


def _router_factors(batch_size, num_copies, fill = None, seed = 0, device = "cuda:0"):
    """
    Factors shaped the way `LowRankSumParams` declares them for this HMM: `[B, 1, S, r]` per copy,
    carved out of one `[B, T, 2, 1, S, r]` tensor the way a router head would emit them.
    """

    torch.manual_seed(seed)

    shape = [batch_size, num_copies, 2, 1, NUM_LATENTS, RANK]
    big = torch.full(shape, fill, device = device) if fill is not None else torch.randn(shape, device = device)

    return big, [(big[:,t,0], big[:,t,1]) for t in range(num_copies)]


def _reference_theta(theta_shared, U_b, V_b):
    """
    The per-example transition the kernels must reproduce, for one batch element:

        theta_b = normalize_over_children( theta_shared + exp(U_b) @ exp(V_b).T )

    `theta_shared` is `[child, parent]` and normalized over children, so `U_b` is indexed by the CHILD
    state and `V_b` by the node's own state -- exactly the orientation `LowRankSumParams` declares
    (`U` sized by `ch_block_size`, `V` by `block_size`).

    :returns: `(theta_b, correction)`, both `[child, parent]`
    """
    correction = torch.exp(U_b) @ torch.exp(V_b).t()

    theta = theta_shared + correction

    return theta / theta.sum(dim = 0, keepdim = True), correction


def _reference_lls(pc, trans_ns, U, V, data):
    """
    The semantics the kernels have to reproduce, computed the slow way: materialize the per-example
    transition and run the ordinary PC on one example at a time.

        theta_b = normalize_over_children( theta_shared + exp(U_b) @ exp(V_b).T )

    :param U: `[B, S, r]` child-side factors (the singleton edge-block axis squeezed out)
    :param V: `[B, S, r]` node-side factors
    """

    param_sid, param_eid = trans_ns._param_range

    theta_shared = pc.params[param_sid:param_eid].view(NUM_LATENTS, NUM_LATENTS).clone()

    lls = []
    with torch.no_grad():
        for b in range(data.size(0)):
            theta, _ = _reference_theta(theta_shared, U[b], V[b])

            pc.params[param_sid:param_eid] = theta.reshape(-1)

            lls.append(pc(data[b:b+1,:]))

        pc.params[param_sid:param_eid] = theta_shared.reshape(-1)

    return torch.cat(lls, dim = 0)


def test_lowrank_hmm_compilation():
    """
    A homogeneous HMM whose transition takes external low-rank parameters: one external layer per
    timestep, all tied to one shared parameter block.
    """

    pc, root_ns, trans_ns = _compiled(external = True)

    ext_layers = [layer for layer_group in pc.inner_layer_groups if layer_group.is_sum()
                  for layer in layer_group.layers if isinstance(layer, ExternalParamsSumLayer)]

    assert len(ext_layers) == SEQ_LENGTH - 1
    assert len(pc.external_params_nodes) == SEQ_LENGTH - 1

    # All the per-timestep copies are tied to the one source, so there is a single transition matrix
    copies = list(pc.external_params_nodes)

    assert trans_ns in copies
    assert all([ns.get_source_ns() is trans_ns for ns in copies])
    assert sum([ns.is_tied() for ns in copies]) == SEQ_LENGTH - 2

    param_sid, param_eid = trans_ns._param_range
    assert param_eid - param_sid == NUM_LATENTS * NUM_LATENTS

    # A dense transition at block_size == num_latents is a single edge block, so the declared layout
    # is the familiar [B, S, r] with a singleton edge-block axis
    for ns in copies:
        shape_U, shape_V = ns.external_params.tensor_shapes(ns, BATCH_SIZE)

        assert ns.edge_ids.size(1) == 1
        assert shape_U == (BATCH_SIZE, 1, NUM_LATENTS, RANK)
        assert shape_V == (BATCH_SIZE, 1, NUM_LATENTS, RANK)

    # The staging buffer is sized accordingly: two factors per copy
    total_numel, _ = pc._external_params_layout(BATCH_SIZE)
    assert total_numel == 2 * (SEQ_LENGTH - 1) * BATCH_SIZE * NUM_LATENTS * RANK


def test_lowrank_hmm_reference_semantics():
    """
    Pin down the semantics the kernels must reproduce, using the plain PC and a materialized
    per-example transition. This is the oracle the forward will be validated against.
    """

    device = torch.device("cuda:0")

    pc, root_ns, trans_ns = _compiled(external = False)

    param_sid, param_eid = trans_ns._param_range
    theta_shared = pc.params[param_sid:param_eid].view(NUM_LATENTS, NUM_LATENTS).clone()

    # `pc.params[trans].view(S, S)` is [child, parent]: normalized over children, i.e. down dim 0
    assert torch.all(torch.abs(theta_shared.sum(dim = 0) - 1.0) < 1e-4)

    torch.manual_seed(1)
    data = torch.randint(0, NUM_EMITS, [BATCH_SIZE, SEQ_LENGTH]).to(device)

    baseline_lls = pc(data)

    ## A vanishing correction recovers the shared model ##

    neg_inf = torch.full([BATCH_SIZE, NUM_LATENTS, RANK], -float("inf"), device = device)
    noop_lls = _reference_lls(pc, trans_ns, neg_inf, neg_inf, data)

    assert torch.all(torch.abs(noop_lls - baseline_lls) < 1e-4)

    ## A finite correction is nonnegative, keeps the transition normalized, and moves the answer ##

    torch.manual_seed(2)
    U = torch.randn([BATCH_SIZE, NUM_LATENTS, RANK], device = device) - 2.0
    V = torch.randn([BATCH_SIZE, NUM_LATENTS, RANK], device = device) - 2.0

    theta, correction = _reference_theta(theta_shared, U[0], V[0])

    assert torch.all(correction > 0.0)                                  # nonnegative => no clamping needed
    assert torch.all(theta > 0.0)
    assert torch.all(torch.abs(theta.sum(dim = 0) - 1.0) < 1e-4)

    ## Orientation: `U` indexes the CHILD state, `V` the node's own state ##

    # Give exactly one child state mass. The correction must then be confined to that child's ROW,
    # which is what distinguishes the intended orientation from its transpose.
    child_id = 3
    child_mask = torch.arange(NUM_LATENTS, device = device) != child_id

    U_one = torch.full([BATCH_SIZE, NUM_LATENTS, RANK], -float("inf"), device = device)
    U_one[:,child_id,:] = 0.0
    V_one = torch.full([BATCH_SIZE, NUM_LATENTS, RANK], -1.0, device = device)

    one_theta, one_correction = _reference_theta(theta_shared, U_one[0], V_one[0])

    assert torch.all(one_correction[child_id,:] > 0.0)
    assert torch.all(one_correction[child_mask,:] == 0.0)

    # Only the child that received mass gains probability, and it does so for every parent
    assert torch.all(one_theta[child_id,:] > theta_shared[child_id,:])
    assert torch.all(one_theta[child_mask,:] < theta_shared[child_mask,:])

    corrected_lls = _reference_lls(pc, trans_ns, U, V, data)

    assert torch.all(torch.isfinite(corrected_lls))
    assert not torch.allclose(corrected_lls, baseline_lls)

    # The correction is per-example: two different factors give two different answers
    other_lls = _reference_lls(pc, trans_ns, U - 1.0, V, data)
    assert not torch.allclose(corrected_lls, other_lls)

    # ... and the shared parameters are left exactly as they were
    assert torch.equal(pc.params[param_sid:param_eid].view(NUM_LATENTS, NUM_LATENTS), theta_shared)


def test_lowrank_hmm_staging():
    """
    Router-shaped factors reach the layers intact at HMM scale, for both a correction shared across
    timesteps and one that varies per timestep.
    """

    device = torch.device("cuda:0")

    pc, root_ns, trans_ns = _compiled(external = True)

    copies = list(pc.external_params_nodes)

    torch.manual_seed(3)
    data = torch.randint(0, NUM_EMITS, [BATCH_SIZE, SEQ_LENGTH]).to(device)

    ## A time-varying correction: one slice of the router's output per timestep ##

    _, per_step = _router_factors(BATCH_SIZE, len(copies), seed = 4)
    mapping = {ns: per_step[t] for t, ns in enumerate(copies)}

    assert not any([t.is_contiguous() for t in mapping[copies[0]]])      # batch-major slices are strided

    pc(data, sum_external_params = mapping)

    staged = pc._staged_external_params

    assert set(staged.keys()) == set(copies)

    # Staged in the kernels' storage order (batch-innermost), so compare against the permuted source
    perm = trans_ns.external_params.storage_perm()
    for ns, tensors in mapping.items():
        for staged_tensor, tensor in zip(staged[ns], tensors):
            assert torch.equal(staged_tensor, tensor.permute(perm))
            assert staged_tensor.is_contiguous()

    # Timesteps do not share storage, so a per-timestep correction really is per-timestep
    assert len(set([staged[ns][0].data_ptr() for ns in copies])) == len(copies)

    ## One correction shared across every timestep -- the same tensors passed for each copy ##

    big, _ = _router_factors(BATCH_SIZE, 1, seed = 5)
    U, V = big[:,0,0].contiguous(), big[:,0,1].contiguous()

    pc(data, sum_external_params = {ns: (U, V) for ns in copies})

    perm = trans_ns.external_params.storage_perm()
    for ns in copies:
        assert torch.equal(pc._staged_external_params[ns][0], U.permute(perm))
        assert torch.equal(pc._staged_external_params[ns][1], V.permute(perm))


def test_lowrank_hmm_shared_params_unaffected():
    """
    Staging external parameters must not disturb the shared model: an HMM with an external transition
    and no factors supplied is the plain HMM through forward, backward and EM, to float32 rounding.
    """

    device = torch.device("cuda:0")

    pc_plain, _, trans_plain = _compiled(external = False)
    pc_ext, _, trans_ext = _compiled(external = True)

    assert trans_plain._param_range == trans_ext._param_range
    assert torch.equal(pc_plain.params, pc_ext.params)

    torch.manual_seed(6)
    data = torch.randint(0, NUM_EMITS, [BATCH_SIZE, SEQ_LENGTH]).to(device)

    # NOT bit equality. The external signature is part of the layer-grouping key, so the two circuits
    # group differently and pyjuice may pick a different kernel implementation for the transition -- its
    # CUDA fast path versus the Triton one, chosen by a runtime autotune. MEASURED difference: 9.5e-07 at
    # a log-likelihood of ~12.5, i.e. exactly one float32 ULP.
    assert torch.allclose(pc_plain(data), pc_ext(data), atol = 1e-5, rtol = 1e-5)

    pc_plain.backward(data)
    pc_ext.backward(data)

    # Same reason as the forward, plus a flow accumulation order that differs between the two kernel
    # implementations. MEASURED at 3.7e-08 absolute on a 2.8e-01 scale, i.e. float32 rounding.
    assert torch.allclose(pc_plain.param_flows, pc_ext.param_flows, atol = 1e-6, rtol = 1e-4)

    pc_plain.mini_batch_em(step_size = 0.5, pseudocount = 0.01)
    pc_ext.mini_batch_em(step_size = 0.5, pseudocount = 0.01)

    assert torch.allclose(pc_plain.params, pc_ext.params, atol = 1e-6, rtol = 1e-4)


def test_lowrank_hmm_forward_matches_reference():
    """
    The two acceptance criteria for the forward, against the materialized oracle on the plain PC:
    a vanishing correction reproduces the baseline, and a finite one reproduces the per-example
    transition it stands for.
    """

    device = torch.device("cuda:0")

    pc_plain, _, trans_plain = _compiled(external = False)
    pc_ext, _, trans_ext = _compiled(external = True)

    copies = list(pc_ext.external_params_nodes)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_EMITS, [BATCH_SIZE, SEQ_LENGTH]).to(device)

    ## A vanishing correction leaves the shared model untouched, exactly ##

    neg_inf = torch.full([BATCH_SIZE, 1, NUM_LATENTS, RANK], -float("inf"), device = device)

    baseline_lls = pc_ext(data)
    noop_lls = pc_ext(data, sum_external_params = {ns: (neg_inf, neg_inf) for ns in copies})

    assert torch.equal(noop_lls, baseline_lls)

    ## A finite correction reproduces the materialized per-example transition ##

    torch.manual_seed(8)
    U = torch.randn([BATCH_SIZE, NUM_LATENTS, RANK], device = device) - 2.0
    V = torch.randn([BATCH_SIZE, NUM_LATENTS, RANK], device = device) - 2.0

    reference_lls = _reference_lls(pc_plain, trans_plain, U, V, data)
    lls = pc_ext(data, sum_external_params = {ns: (U[:,None].contiguous(), V[:,None].contiguous())
                                              for ns in copies})

    assert torch.all(torch.abs(reference_lls - lls) < 1e-4)
    assert not torch.allclose(lls, baseline_lls)


def test_lowrank_hmm_distinct_per_timestep():
    """
    A correction that DIFFERS per timestep.

    This is the case that catches a wrong staging offset: passing the same factors to every copy makes
    every slot hold identical data, so reading the wrong one still gives the right answer. Only
    distinct factors per copy expose it.
    """

    device = torch.device("cuda:0")

    batch_size = 64

    pc_plain, _, trans_plain = _compiled(external = False)
    pc_ext, _, trans_ext = _compiled(external = True)

    copies = list(pc_ext.external_params_nodes)

    torch.manual_seed(21)
    data = torch.randint(0, NUM_EMITS, [batch_size, SEQ_LENGTH]).to(device)

    # One distinct factor pair per timestep
    _, per_step = _router_factors(batch_size, len(copies), seed = 22)
    mapping = {ns: (tensors[0].contiguous(), tensors[1].contiguous())
               for ns, tensors in zip(copies, per_step)}

    kernel_lls = pc_ext(data, sum_external_params = mapping)

    applicable = LowRankSumParams._kernel_applicable
    try:
        LowRankSumParams._kernel_applicable = lambda *args, **kwargs: False
        torch_lls = pc_ext(data, sum_external_params = mapping)
    finally:
        LowRankSumParams._kernel_applicable = applicable

    assert torch.all(torch.abs(kernel_lls - torch_lls) < 1e-4)

    # Every timestep's factors must actually matter: perturbing just one changes the answer
    perturbed = dict(mapping)
    perturbed[copies[-1]] = (mapping[copies[-1]][0] + 1.0, mapping[copies[-1]][1])

    assert not torch.allclose(pc_ext(data, sum_external_params = perturbed), kernel_lls)


def test_lowrank_hmm_kernel_matches_torch_reference():
    """
    At a batch size the Triton kernel covers, it must agree with the torch reference it replaces.

    The comparison is against the torch path rather than the per-example oracle because the SHARED
    forward itself changes numerics with batch size (a bf16 tensor-core dot above the small-batch
    threshold); that difference is present with no external parameters at all, so comparing to a
    batch-1 oracle here would measure the shared kernel, not this correction.
    """

    device = torch.device("cuda:0")

    batch_size = 64

    pc, root_ns, trans_ns = _compiled(external = True)

    copies = list(pc.external_params_nodes)

    torch.manual_seed(11)
    data = torch.randint(0, NUM_EMITS, [batch_size, SEQ_LENGTH]).to(device)

    U = torch.randn([batch_size, 1, NUM_LATENTS, RANK], device = device) - 2.0
    V = torch.randn([batch_size, 1, NUM_LATENTS, RANK], device = device) - 2.0

    mapping = {ns: (U, V) for ns in copies}

    ext_layer = [layer for layer_group in pc.inner_layer_groups if layer_group.is_sum()
                 for layer in layer_group.layers if getattr(layer, "ext_xu", None) is not None][0]

    assert trans_ns.external_params._kernel_applicable(
        ext_layer, [(None, None)] * len(ext_layer.external_node_infos),
        torch.zeros([1, batch_size], device = device)
    )

    kernel_lls = pc(data, sum_external_params = mapping)

    # Same computation through the reference path
    applicable = LowRankSumParams._kernel_applicable
    try:
        LowRankSumParams._kernel_applicable = lambda *args, **kwargs: False
        torch_lls = pc(data, sum_external_params = mapping)
    finally:
        LowRankSumParams._kernel_applicable = applicable

    assert torch.all(torch.abs(kernel_lls - torch_lls) < 1e-4)


if __name__ == "__main__":
    test_lowrank_hmm_compilation()
    test_lowrank_hmm_reference_semantics()
    test_lowrank_hmm_staging()
    test_lowrank_hmm_shared_params_unaffected()
    test_lowrank_hmm_reaches_kernels()
