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

    with pytest.raises(NotImplementedError):
        pc(data, sum_external_params = mapping)

    staged = pc._staged_external_params

    assert set(staged.keys()) == set(copies)
    for ns, tensors in mapping.items():
        for staged_tensor, tensor in zip(staged[ns], tensors):
            assert torch.equal(staged_tensor, tensor)
            assert staged_tensor.is_contiguous()

    # Timesteps do not share storage, so a per-timestep correction really is per-timestep
    assert len(set([staged[ns][0].data_ptr() for ns in copies])) == len(copies)

    ## One correction shared across every timestep -- the same tensors passed for each copy ##

    big, _ = _router_factors(BATCH_SIZE, 1, seed = 5)
    U, V = big[:,0,0].contiguous(), big[:,0,1].contiguous()

    with pytest.raises(NotImplementedError):
        pc(data, sum_external_params = {ns: (U, V) for ns in copies})

    for ns in copies:
        assert torch.equal(pc._staged_external_params[ns][0], U)
        assert torch.equal(pc._staged_external_params[ns][1], V)


def test_lowrank_hmm_shared_params_unaffected():
    """
    Staging external parameters must not disturb the shared model: an HMM with an external transition
    and no factors supplied is the plain HMM, bit for bit, through forward, backward and EM.
    """

    device = torch.device("cuda:0")

    pc_plain, _, trans_plain = _compiled(external = False)
    pc_ext, _, trans_ext = _compiled(external = True)

    assert trans_plain._param_range == trans_ext._param_range
    assert torch.equal(pc_plain.params, pc_ext.params)

    torch.manual_seed(6)
    data = torch.randint(0, NUM_EMITS, [BATCH_SIZE, SEQ_LENGTH]).to(device)

    assert torch.equal(pc_plain(data), pc_ext(data))

    pc_plain.backward(data)
    pc_ext.backward(data)

    assert torch.equal(pc_plain.param_flows, pc_ext.param_flows)

    pc_plain.mini_batch_em(step_size = 0.5, pseudocount = 0.01)
    pc_ext.mini_batch_em(step_size = 0.5, pseudocount = 0.01)

    assert torch.equal(pc_plain.params, pc_ext.params)


def test_lowrank_hmm_reaches_kernels():
    """
    The forward reaches the low-rank kernels with everything staged.

    Once they are implemented, this is where the equivalence assertion goes:
    `pc(data, sum_external_params = ...)` must match `_reference_lls(...)` on the plain PC, and
    `-inf` factors must reproduce the baseline exactly.
    """

    device = torch.device("cuda:0")

    pc, root_ns, trans_ns = _compiled(external = True)

    copies = list(pc.external_params_nodes)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_EMITS, [BATCH_SIZE, SEQ_LENGTH]).to(device)

    big, _ = _router_factors(BATCH_SIZE, 1, seed = 8)
    U, V = big[:,0,0].contiguous(), big[:,0,1].contiguous()

    mapping = {ns: (U, V) for ns in copies}

    with pytest.raises(NotImplementedError):
        pc(data, sum_external_params = mapping)

    # Everything up to the kernel is in place
    assert pc.external_params is not None
    assert pc.external_params.numel() == pc._external_params_layout(BATCH_SIZE)[0]
    assert all([torch.equal(pc._staged_external_params[ns][0], U) for ns in copies])


if __name__ == "__main__":
    test_lowrank_hmm_compilation()
    test_lowrank_hmm_reference_semantics()
    test_lowrank_hmm_staging()
    test_lowrank_hmm_shared_params_unaffected()
    test_lowrank_hmm_reaches_kernels()
