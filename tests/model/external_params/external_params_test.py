import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, LowRankSumParams

import pytest


def _build_hmm(external, seed = 0, seq_length = 4, num_latents = 8, num_emits = 5, rank = 3):
    """
    A homogeneous-HMM-shaped PC whose transition either does or does not take external parameters.
    Built at one seed so the two variants are directly comparable.
    """

    torch.manual_seed(seed)

    ext = dict(external_params = LowRankSumParams(rank = rank)) if external else dict()

    with juice.set_block_size(num_latents):

        ns_input = inputs(seq_length - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = num_emits))

        ns_sum, curr_zs = None, ns_input
        for var in range(seq_length - 2, -1, -1):
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


def _colsums(pc, ns):
    # `get_node_params` -> [edge_block, parent, child]; sum nodes normalize over children AND the
    # edge blocks that connect them
    return pc.get_node_params(ns).sum(dim = 2).sum(dim = 0)


def test_external_params_pc_matches_plain():
    """
    An external parameterization must not change the `params` / `param_flows` setup at all: the layer
    inherits `SumLayer`'s compilation, so with no external tensors supplied the PC is the plain one,
    bit for bit.
    """

    device = torch.device("cuda:0")

    root_plain, trans_plain = _build_hmm(external = False)
    root_ext, trans_ext = _build_hmm(external = True)

    pc_plain = juice.compile(root_plain).to(device)
    pc_ext = juice.compile(root_ext).to(device)

    ## Identical parameter layout ##

    assert pc_plain.num_sum_params == pc_ext.num_sum_params
    assert pc_plain.num_param_flows == pc_ext.num_param_flows
    assert trans_plain._param_range == trans_ext._param_range
    assert trans_plain._param_flow_range == trans_ext._param_flow_range
    assert torch.equal(pc_plain.params, pc_ext.params)

    # The tied per-timestep copies share the source's parameter block, as they would for a plain node
    tied_nss = [ns for ns in root_ext if ns.is_sum() and ns.is_tied()]
    assert len(tied_nss) == 2
    assert all([ns.get_source_ns() is trans_ext for ns in tied_nss])

    ## Identical forward / backward ##

    data = torch.randint(0, 5, [32, len(root_ext.scope)]).to(device)

    assert torch.equal(pc_plain(data), pc_ext(data))

    pc_plain.backward(data)
    pc_ext.backward(data)

    assert torch.equal(pc_plain.param_flows, pc_ext.param_flows)
    assert pc_plain._cum_flow == pc_ext._cum_flow
    for layer_plain, layer_ext in zip(pc_plain.input_layer_group, pc_ext.input_layer_group):
        assert torch.equal(layer_plain.param_flows, layer_ext.param_flows)

    ## Identical EM update ##

    pc_plain.mini_batch_em(step_size = 0.5, pseudocount = 0.01)
    pc_ext.mini_batch_em(step_size = 0.5, pseudocount = 0.01)

    assert torch.equal(pc_plain.params, pc_ext.params)
    assert torch.all(torch.abs(_colsums(pc_ext, trans_ext) - 1.0) < 1e-5)


@pytest.mark.parametrize("optimizer", ["FullBatchEM", "MiniBatchEM", "Anemone"])
def test_external_params_em_optimizers(optimizer):
    """
    The EM-family optimizers consume `param_flows` and normalize `params`; neither depends on how the
    effective parameters are formed, so an external node trains exactly like a plain one.

    This matters for Anemone in particular: its `step_size_rescaling` evaluates top-down probabilities
    assuming the PC is normalized, which holds because the shared parameters stay child-normalized.
    """

    device = torch.device("cuda:0")

    def make_optimizer(pc):
        if optimizer == "FullBatchEM":
            return juice.optim.FullBatchEM(pc)
        elif optimizer == "MiniBatchEM":
            return juice.optim.MiniBatchEM(pc, step_size = 0.3, pseudocount = 0.01)
        else:
            return juice.optim.Anemone(pc, step_size = 0.4, momentum = 0.9,
                                       niters_per_update = 2, pseudocount = 1e-6)

    root_plain, _ = _build_hmm(external = False)
    root_ext, trans_ext = _build_hmm(external = True)

    pc_plain = juice.compile(root_plain).to(device)
    pc_ext = juice.compile(root_ext).to(device)

    opt_plain, opt_ext = make_optimizer(pc_plain), make_optimizer(pc_ext)

    torch.manual_seed(1)
    data = torch.randint(0, 5, [32, len(root_ext.scope)]).to(device)

    lls_plain, lls_ext = [], []
    for it in range(6):
        batch = data[(it % 2) * 16:(it % 2) * 16 + 16,:]

        for pc, opt, lls_list in ((pc_plain, opt_plain, lls_plain), (pc_ext, opt_ext, lls_ext)):
            lls = pc(batch)
            pc.backward(batch, flows_memory = 1.0)
            opt.step()
            lls_list.append(lls.mean().item())

    assert lls_plain == lls_ext
    assert torch.equal(pc_plain.params, pc_ext.params)

    # The shared parameters stay child-normalized throughout, which is what makes the correction's
    # normalizer independent of them -- and hence the EM update exact
    assert torch.all(torch.abs(_colsums(pc_ext, trans_ext) - 1.0) < 1e-4)


def test_external_params_node_accessors():
    """
    The per-node parameter views work for an external node exactly as for a plain one.
    """

    device = torch.device("cuda:0")

    root_ns, trans_ns = _build_hmm(external = True)

    pc = juice.compile(root_ns).to(device)

    data = torch.randint(0, 5, [32, len(root_ns.scope)]).to(device)

    pc(data)
    pc.backward(data)

    num_latents = trans_ns.num_nodes

    ns_params = pc.get_node_params(trans_ns)
    ns_param_flows = pc.get_node_param_flows(trans_ns)

    assert ns_params is not None and ns_param_flows is not None
    assert ns_params.size(1) == num_latents and ns_params.size(2) == num_latents
    assert ns_param_flows.size() == ns_params.size()
    assert torch.all(ns_param_flows >= 0.0)

    # Tied nodes hold no parameters of their own, here as anywhere else
    for ns in root_ns:
        if ns.is_sum() and ns.is_tied():
            assert pc.get_node_params(ns) is None

    pc.update_parameters()
    pc.update_param_flows()

    assert trans_ns.get_params() is not None
    assert trans_ns.get_param_flows() is not None
    assert trans_ns.get_params(as_matrix = True).size() == (num_latents, num_latents)

    # Writing the node's parameters back into the PC reproduces what is already there
    params_before = pc.params.clone()
    trans_ns.gather_parameters(pc.params)

    assert torch.all(torch.abs(params_before - pc.params) < 1e-6)


if __name__ == "__main__":
    test_external_params_pc_matches_plain()
    for optimizer in ["FullBatchEM", "MiniBatchEM", "Anemone"]:
        test_external_params_em_optimizers(optimizer)
    test_external_params_node_accessors()
