import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, set_block_size
from pyjuice.parameterizations import LowRank

import pytest


# A product node vector that is a child of sum nodes landing in DIFFERENT sum layers of one depth (another block size,
# or a parameterization: `TensorCircuit._sum_layer_key`). Every sum layer's backward stores the element flows of its
# children, so before compilation gave each further layer its own copy of such a product, the product kept only the
# last layer's flow and everything below it received too little -- while the forward pass was right.
#
# The invariant tested: in a smooth, decomposable, normalized circuit every variable's input nodes receive a total
# flow of exactly 1 a sample.


def _shared_product_pc(kind):
    with set_block_size(4):
        i0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
        i1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
        i2a = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
        p = multiply(i0, i1)                                    # the shared product
        ext = None
        if kind == "one_layer":                                 # (the control: both parents in one sum layer)
            sa = summate(p, num_node_blocks = 2)
            sb = summate(p, num_node_blocks = 2)
            i2b = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
            root = summate(multiply(sa, i2a), multiply(sb, i2b), num_node_blocks = 1, block_size = 1)
        elif kind == "lowrank_and_plain":
            sa = ext = summate(p, num_node_blocks = 2, parameterization = LowRank(rank = 1, form = "exp_sum"))
            sb = summate(p, num_node_blocks = 2)
            i2b = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
            root = summate(multiply(sa, i2a), multiply(sb, i2b), num_node_blocks = 1, block_size = 1)
        else:
            assert kind == "two_block_sizes"
            sa = summate(p, num_node_blocks = 2)
            sb = summate(p, num_node_blocks = 4, block_size = 2)
            i2b = inputs(2, num_node_blocks = 4, block_size = 2, dist = dists.Categorical(num_cats = 3))
            ra = summate(multiply(sa, i2a), num_node_blocks = 1, block_size = 1)
            rb = summate(multiply(sb, i2b), num_node_blocks = 1, block_size = 1)
            i3a = inputs(3, num_node_blocks = 1, block_size = 1, dist = dists.Categorical(num_cats = 3))
            i3b = inputs(3, num_node_blocks = 1, block_size = 1, dist = dists.Categorical(num_cats = 3))
            root = summate(multiply(ra, i3a), multiply(rb, i3b), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, ext, (i0, i1)


@pytest.mark.parametrize("kind", ["one_layer", "lowrank_and_plain", "two_block_sizes"])
@pytest.mark.parametrize("logspace_flows", [True, False])
def test_product_shared_by_sum_layers_passes_all_flow(kind, logspace_flows):
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    if kind == "lowrank_and_plain" and not logspace_flows:
        pytest.skip("`LowRank` requires `logspace_flows = True`")

    root, ext, shared_inputs = _shared_product_pc(kind)
    num_vars = len(root.scope)
    pc = juice.compile(root)
    pc.to(device)

    B = 32
    x = torch.randint(0, 3, (B, num_vars)).to(device)

    fw_kwargs, bk_kwargs = dict(), dict()
    if ext is not None:
        A = 0.3 * torch.randn(B, ext.num_nodes, 1, device = device)
        Bf = 0.3 * torch.randn(B, 1, ext.num_ch_nodes, device = device)
        fw_kwargs = dict(sum_external_params = {ext: (A, Bf)})
        bk_kwargs = dict(sum_external_params_grad = {ext: (torch.zeros_like(A), torch.zeros_like(Bf))})

    lls = pc(x, **fw_kwargs)
    assert torch.isfinite(lls).all()

    pc.backward(x, allow_modify_flows = False, logspace_flows = logspace_flows, **bk_kwargs)

    for ns in shared_inputs:
        s, e = ns._output_ind_range
        flows = pc.node_flows[s:e].double()
        per_sample = (flows.exp() if logspace_flows else flows).sum(dim = 0)
        assert torch.allclose(per_sample, torch.ones_like(per_sample), atol = 1e-4), \
            f"{kind}: the inputs below the shared product got {per_sample.min().item():.4f} .. " \
            f"{per_sample.max().item():.4f} of a sample's flow instead of 1"


@pytest.mark.parametrize("kind", ["one_layer", "lowrank_and_plain", "two_block_sizes"])
def test_top_down_probs_with_several_sum_layers_a_depth(kind):
    # The top-down pass (`eval_top_down_probs`, the `(1 - step_size) * P_td * theta` term of a mini-batch EM step)
    # zeroed the element flows ahead of EVERY sum layer of a depth, so only the last layer's children kept a top-down
    # probability. In a normalized circuit every variable's input nodes have a total top-down probability of 1.
    from pyjuice.model.backend.top_down_prob import eval_top_down_probs

    torch.manual_seed(0)
    device = torch.device("cuda:0")

    root, ext, shared_inputs = _shared_product_pc(kind)
    num_vars = len(root.scope)
    pc = juice.compile(root)
    pc.to(device)

    B = 8
    x = torch.randint(0, 3, (B, num_vars)).to(device)
    fw_kwargs, bk_kwargs = dict(), dict()
    if ext is not None:
        A = torch.zeros(B, ext.num_nodes, 1, device = device)
        Bf = torch.zeros(B, 1, ext.num_ch_nodes, device = device)
        fw_kwargs = dict(sum_external_params = {ext: (A, Bf)})
        bk_kwargs = dict(sum_external_params_grad = {ext: (torch.zeros_like(A), torch.zeros_like(Bf))})
    pc(x, **fw_kwargs)
    pc.backward(x, allow_modify_flows = False, logspace_flows = True, **bk_kwargs)      # (allocates the flow buffers)

    eval_top_down_probs(pc, update_pflow = False, scale = 1.0, use_cudagraph = False)

    for ns in shared_inputs:
        s, e = ns._output_ind_range
        total = pc._single_node_flows[s:e].double().sum().item()
        assert abs(total - 1.0) < 1e-4, \
            f"{kind}: the inputs of variable {ns.scope.to_list()[0]} have a total top-down probability of {total:.4f}"


def test_small_step_em_with_a_dead_branch_matches_the_plain_circuit():
    # End to end: one mini-batch EM step with step size < 1 on (a) a circuit with one sum layer a depth, and (b) the
    # same circuit with an extra branch of weight exactly 0 that puts a second sum layer (another block size) at that
    # depth -- the shared parameters must end up the same. (Whether the live layer lost its `(1 - step_size)` term
    # before the top-down fix depended on the layers' order, and this circuit's spared it: the test that fails without
    # the fix is `test_top_down_probs_with_several_sum_layers_a_depth`.)
    device = torch.device("cuda:0")

    def build(dead_branch):
        with set_block_size(4):
            i0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
            i1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
            s1 = summate(multiply(i0, i1), num_node_blocks = 2)
            i2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 3))
            top = summate(multiply(s1, i2), num_node_blocks = 1, block_size = 1)
            if not dead_branch:
                root = top
            else:
                # a second sum over (X0, X1), of block size 2 -> its own sum layer at s1's depth -- reached only
                # through a root edge whose parameter is exactly 0
                j0 = inputs(0, num_node_blocks = 4, block_size = 2, dist = dists.Categorical(num_cats = 3))
                j1 = inputs(1, num_node_blocks = 4, block_size = 2, dist = dists.Categorical(num_cats = 3))
                dead = summate(multiply(j0, j1), num_node_blocks = 4, block_size = 2)
                j2 = inputs(2, num_node_blocks = 4, block_size = 2, dist = dists.Categorical(num_cats = 3))
                rb = summate(multiply(dead, j2), num_node_blocks = 1, block_size = 1)
                k3a = inputs(3, num_node_blocks = 1, block_size = 1, dist = dists.Categorical(num_cats = 2))
                k3b = inputs(3, num_node_blocks = 1, block_size = 1, dist = dists.Categorical(num_cats = 2))
                root = summate(multiply(top, k3a), multiply(rb, k3b), num_node_blocks = 1, block_size = 1)
        return root, dict(i0 = i0, i1 = i1, i2 = i2, s1 = s1, top = top)

    torch.manual_seed(1)
    root_a, shared_a = build(False)
    root_a.init_parameters(perturbation = 2.0)
    root_b, shared_b = build(True)
    root_b.init_parameters(perturbation = 2.0)
    for k in shared_a:                                                                   # the same shared parameters
        shared_b[k].set_params(shared_a[k].get_params().clone(), normalize = False)
    root_b.set_params(torch.tensor([[1.0, 0.0]]), normalize = False)                     # the dead branch: weight 0
    before = shared_a["s1"].get_params().double().clone()

    x3 = torch.randint(0, 3, (64, 3))
    outs = []
    for root, shared, x in ((root_a, shared_a, x3),
                            (root_b, shared_b, torch.cat([x3, torch.zeros(64, 1, dtype = torch.long)], 1))):
        pc = juice.compile(root)
        pc.to(device)
        pc(x.to(device))
        pc.backward(x.to(device), allow_modify_flows = False, logspace_flows = True)
        pc.mini_batch_em(step_size = 0.1, pseudocount = 0.0, keep_zero_params = True, step_size_rescaling = True)
        pc.update_parameters()
        outs.append(shared["s1"].get_params().double().cpu())

    moved = (outs[0] - before).abs().max().item()
    assert moved > 1e-4                                                                  # (the step did something)
    assert torch.allclose(outs[0], outs[1], atol = 0.02 * moved), \
        f"the shared sum's parameters differ by {(outs[0] - outs[1]).abs().max().item():.3e} after one small EM " \
        f"step that moved them by {moved:.3e}"


def test_shared_product_gets_one_copy_a_further_layer():
    # (the structure the fix makes: the first layer keeps the product, each further layer one copy of it)
    root, ext, _ = _shared_product_pc("lowrank_and_plain")
    pc = juice.compile(root)

    prods = [ns for ns in pc.root_ns if ns.is_prod() and len(ns.scope) == 2]
    assert len(prods) == 2
    assert prods[0].chs[0] is prods[1].chs[0] and prods[0].chs[1] is prods[1].chs[1]

    root, ext, _ = _shared_product_pc("one_layer")
    pc = juice.compile(root)

    prods = [ns for ns in pc.root_ns if ns.is_prod() and len(ns.scope) == 2]
    assert len(prods) == 1


if __name__ == "__main__":
    for kind in ["one_layer", "lowrank_and_plain", "two_block_sizes"]:
        for logspace_flows in [True, False]:
            test_product_shared_by_sum_layers_passes_all_flow(kind, logspace_flows)
        test_top_down_probs_with_several_sum_layers_a_depth(kind)
    test_small_step_em_with_a_dead_branch_matches_the_plain_circuit()
    test_shared_product_gets_one_copy_a_further_layer()
