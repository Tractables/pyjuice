"""
Addressing several nodes' external parameters with ONE concatenated tensor, under a name.

Nodes sharing a parameterization usually share a shape -- an HMM's per-timestep transitions being the
case this exists for -- and a head producing them emits one tensor already. A group lets that tensor be
handed over as-is instead of being sliced apart before every call.

The claim under test is equivalence: a group must be exactly the per-node API with the slicing moved
inside. So the checks compare the two forms THROUGH ONE PC and demand bit-identity. Two PCs would not
do: each autotunes its backward independently and may pick a different (equally valid) kernel, which
shows up as ~1e-3 of rounding and would mask a real difference.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import BlockScaleSumParams, LowRankSumParams


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _two_gated(K = 128, gate_cbs = 8, seed = 0, n_vars = 3):
    """Two gated sum layers of identical gate shape, stacked."""
    torch.manual_seed(seed)
    with juice.set_block_size(K):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(n_vars)]
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gate_cbs))
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gate_cbs))
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, s0, s1


def _gates(ns, batch, dev, seed):
    torch.manual_seed(seed)
    return torch.randn(ns.external_params.tensor_shapes(ns, batch)[0], device = dev) * 1.5


@cuda_only
def test_group_is_exactly_the_per_node_api():
    """Forward and backward, both forms, one PC: bit-identical or the group is doing something else."""
    dev, batch = torch.device("cuda:0"), 64
    root, s0, s1 = _two_gated()
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0, g1 = _gates(s0, batch, dev, 11), _gates(s1, batch, dev, 12)

    pc.register_external_params_group("router", [s0, s1])
    ll_g = pc(data, sum_external_params = {"router": torch.cat([g0, g1], dim = 1)}).clone()
    pc.backward(data, flows_memory = 0.0)
    ef_g, pf_g = pc.element_flows.clone(), pc.param_flows.clone()

    ll_p = pc(data, sum_external_params = {s0: g0, s1: g1}).clone()
    pc.backward(data, flows_memory = 0.0)

    assert torch.equal(ll_g, ll_p), "the grouped forward differs from the per-node one"
    assert torch.equal(ef_g, pc.element_flows), "the grouped element flows differ"
    assert torch.equal(pf_g, pc.param_flows), "the grouped param flows differ"


@cuda_only
def test_a_group_and_a_bare_node_mix_in_one_dict():
    """Keys are nodes or names, told apart by type, so both forms coexist."""
    dev, batch = torch.device("cuda:0"), 64
    root, s0, s1 = _two_gated()
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0, g1 = _gates(s0, batch, dev, 11), _gates(s1, batch, dev, 12)

    pc.register_external_params_group("first", [s0])
    ll_mixed = pc(data, sum_external_params = {"first": g0, s1: g1}).clone()
    ll_plain = pc(data, sum_external_params = {s0: g0, s1: g1}).clone()

    assert torch.equal(ll_mixed, ll_plain)


@cuda_only
def test_the_dim_lever():
    """`dim` defaults to 1, the first non-batch axis, and any other parameter axis works too."""
    dev, batch = torch.device("cuda:0"), 64
    root, s0, s1 = _two_gated()
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0, g1 = _gates(s0, batch, dev, 11), _gates(s1, batch, dev, 12)

    ref = pc(data, sum_external_params = {s0: g0, s1: g1}).clone()

    pc.register_external_params_group("axis2", [s0, s1], dim = 2)
    got = pc(data, sum_external_params = {"axis2": torch.cat([g0, g1], dim = 2)}).clone()

    # concatenating along the CHILD axis puts each member's gates where its own tensor had them only
    # if the split is by that member's own extent -- which is what makes unequal extents legal below
    assert torch.equal(got, ref)


@cuda_only
def test_members_may_differ_in_extent_along_the_group_axis():
    """Members are split by their OWN extents, so only the other axes have to agree."""
    dev, batch = torch.device("cuda:0"), 64
    torch.manual_seed(0)
    with juice.set_block_size(128):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(3)]
        # different gate widths -> different Ck, the axis being concatenated
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = 8))
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = 8))
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0, g1 = _gates(s0, batch, dev, 11), _gates(s1, batch, dev, 12)

    pc.register_external_params_group("mixed", [s0, s1], dim = 2)
    got = pc(data, sum_external_params = {"mixed": torch.cat([g0, g1], dim = 2)}).clone()
    ref = pc(data, sum_external_params = {s0: g0, s1: g1}).clone()
    assert torch.equal(got, ref)


@cuda_only
def test_group_of_multi_tensor_nodes_and_its_gradients():
    """A parameterization with several tensors per node takes one concatenated tensor PER SLOT, and
    the gradient comes back in the same layout."""
    dev, batch, K, rank = torch.device("cuda:0"), 32, 32, 4
    torch.manual_seed(0)
    with juice.set_block_size(K):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(3)]
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = LowRankSumParams(rank = rank))
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1,
                     external_params = LowRankSumParams(rank = rank))
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    torch.manual_seed(11)
    f0 = tuple(torch.rand(sh, device = dev) * 0.1
               for sh in s0.external_params.tensor_shapes(s0, batch))
    f1 = tuple(torch.rand(sh, device = dev) * 0.1
               for sh in s1.external_params.tensor_shapes(s1, batch))

    pc.register_external_params_group("factors", [s0, s1])
    cat = tuple(torch.cat([a, b], dim = 1) for a, b in zip(f0, f1))

    ll_g = pc(data, sum_external_params = {"factors": cat}).clone()
    pc.backward(data, flows_memory = 0.0)
    grad_g = tuple(g.clone() for g in pc.get_external_params_grad("factors"))

    ll_p = pc(data, sum_external_params = {s0: f0, s1: f1}).clone()
    pc.backward(data, flows_memory = 0.0)
    grad_0 = tuple(g.clone() for g in pc.get_external_params_grad(s0))
    grad_1 = tuple(g.clone() for g in pc.get_external_params_grad(s1))

    assert torch.equal(ll_g, ll_p)
    assert len(grad_g) == len(cat), "one gradient tensor per slot"
    for slot, (a, b) in enumerate(zip(grad_0, grad_1)):
        assert grad_g[slot].shape == cat[slot].shape, f"slot {slot} gradient is not the group's shape"
        assert torch.equal(grad_g[slot], torch.cat([a, b], dim = 1)), f"slot {slot} gradient differs"


@cuda_only
def test_a_grouped_hmm_feeds_every_timestep_from_one_tensor():
    """The case this exists for: one tensor of per-timestep gates instead of T separate ones."""
    dev, batch, T, K = torch.device("cuda:0"), 64, 5, 128
    torch.manual_seed(0)
    with juice.set_block_size(K):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        trans = []
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
            ns = summate(multiply(ns, emit), num_node_blocks = 1,
                         external_params = BlockScaleSumParams(ch_block_size = 8))
            trans.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, T], device = dev)
    gates = [_gates(t, batch, dev, 20 + i) for i, t in enumerate(trans)]

    pc.register_external_params_group("transitions", trans)
    ll_g = pc(data, sum_external_params = {"transitions": torch.cat(gates, dim = 1)}).clone()
    ll_p = pc(data, sum_external_params = dict(zip(trans, gates))).clone()

    assert torch.equal(ll_g, ll_p)
    assert torch.isfinite(ll_g).all()


# --------------------------------------------------------------------------------- rejections

@cuda_only
def test_group_registration_rejects_bad_definitions():
    dev = torch.device("cuda:0")
    root, s0, s1 = _two_gated()
    pc = juice.compile(root, verbose = False).to(dev)

    with pytest.raises(AssertionError, match = "batch axis"):
        pc.register_external_params_group("d0", [s0, s1], dim = 0)

    with pytest.raises(AssertionError, match = "empty"):
        pc.register_external_params_group("none", [])

    with pytest.raises(AssertionError, match = "same node twice"):
        pc.register_external_params_group("dup", [s0, s0])

    pc.register_external_params_group("ok", [s0, s1])
    with pytest.raises(AssertionError, match = "already registered"):
        pc.register_external_params_group("ok", [s0])

    pc.unregister_external_params_group("ok")
    pc.register_external_params_group("ok", [s0])          # the name is free again


@cuda_only
def test_group_use_rejects_the_wrong_tensor():
    dev, batch = torch.device("cuda:0"), 64
    root, s0, s1 = _two_gated()
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0, g1 = _gates(s0, batch, dev, 11), _gates(s1, batch, dev, 12)

    with pytest.raises(AssertionError, match = "unregistered external-parameter group"):
        pc(data, sum_external_params = {"nope": g0})

    pc.register_external_params_group("router", [s0, s1])

    # the un-concatenated tensor: right rank, wrong extent along the group axis
    with pytest.raises(AssertionError, match = "concatenation"):
        pc(data, sum_external_params = {"router": g0})

    with pytest.raises(AssertionError, match = "both directly and through group"):
        pc(data, sum_external_params = {"router": torch.cat([g0, g1], dim = 1), s0: g0})


@cuda_only
def test_a_group_cannot_split_tensors_that_are_tied_together():
    """`tie_external` makes a node and its copy read ONE tensor, supplied through the storage owner.
    Handing them two slices of a concatenation contradicts that, so the group is refused."""
    dev, K = torch.device("cuda:0"), 128
    torch.manual_seed(0)
    with juice.set_block_size(K):
        n0 = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        n1 = inputs(1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        s0 = summate(multiply(n0, n1), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = 8, tie_external = True))
        n2 = inputs(2, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        s1 = s0.duplicate(multiply(s0, n2), tie_params = True)
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    with pytest.raises(AssertionError, match = "tie_external"):
        pc.register_external_params_group("tied", [s0, s1])


@cuda_only
def test_a_contiguous_group_stages_in_one_copy():
    """A group whose members land on adjacent slots must take the whole-group transpose.

    White-box on purpose. Every other test here compares results, and results are identical whichever
    path runs -- so a silent fall back to the per-member path (which is ~1.75x SLOWER at T=32 than the
    per-node API the group is meant to improve on) would go unnoticed."""
    dev, batch, T, K = torch.device("cuda:0"), 64, 6, 128
    torch.manual_seed(0)
    with juice.set_block_size(K):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        trans = []
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
            ns = summate(multiply(ns, emit), num_node_blocks = 1,
                         external_params = BlockScaleSumParams(ch_block_size = 8))
            trans.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    gates = [_gates(t, batch, dev, 30 + i) for i, t in enumerate(trans)]
    cat = torch.cat(gates, dim = 1)
    pc.register_external_params_group("tr", trans)

    views = pc._external_params_views(name = "external_params", batch_size = batch)
    whole = pc._group_fast_stage("tr", cat, views, batch)
    assert whole is not None, "the whole-group staging path was refused for a contiguous group"
    dst, src = whole
    assert dst.numel() == cat.numel(), "the group's destination does not cover the whole tensor"
    assert src is cat, "the fast path must copy the caller's tensor directly, without a repack"

    # and reversing the order breaks adjacency, which must be DETECTED, not mis-staged
    pc.register_external_params_group("rev", list(reversed(trans)))
    assert pc._group_fast_stage("rev", cat, views, batch) is None, \
        "a group whose order does not match the buffer must fall back, not stage wholesale"

    # either way the answer is the same
    ll_g = pc(data_ := torch.randint(0, NUM_CATS, [batch, T], device = dev),
              sum_external_params = {"tr": cat}).clone()
    ll_r = pc(data_, sum_external_params = {"rev": torch.cat(list(reversed(gates)), dim = 1)}).clone()
    ll_p = pc(data_, sum_external_params = dict(zip(trans, gates))).clone()
    assert torch.equal(ll_g, ll_p) and torch.equal(ll_r, ll_p)
