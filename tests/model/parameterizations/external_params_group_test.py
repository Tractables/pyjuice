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
from pyjuice.parameterizations import BlockScale, LowRank, SumParameterization


NUM_CATS = 4

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _two_gated(K = 128, gate_cbs = 8, seed = 0, n_vars = 3):
    """Two gated sum layers of identical gate shape, stacked."""
    torch.manual_seed(seed)
    with juice.set_block_size(K):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(n_vars)]
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     parameterization = BlockScale(ch_block_size = gate_cbs))
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1,
                     parameterization = BlockScale(ch_block_size = gate_cbs))
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, s0, s1


def _gates(ns, batch, dev, seed):
    torch.manual_seed(seed)
    return torch.randn(ns.parameterization.tensor_shapes(ns, batch)[0], device = dev) * 1.5


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
                     parameterization = BlockScale(ch_block_size = 8))
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1,
                     parameterization = BlockScale(ch_block_size = 8))
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
                     parameterization = LowRank(rank = rank))
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1,
                     parameterization = LowRank(rank = rank))
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    torch.manual_seed(11)
    f0 = tuple(torch.rand(sh, device = dev) * 0.1
               for sh in s0.parameterization.tensor_shapes(s0, batch))
    f1 = tuple(torch.rand(sh, device = dev) * 0.1
               for sh in s1.parameterization.tensor_shapes(s1, batch))

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
                         parameterization = BlockScale(ch_block_size = 8))
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
                     parameterization = BlockScale(ch_block_size = 8, tie_external = True))
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
                         parameterization = BlockScale(ch_block_size = 8))
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


def _counting_staging(monkeypatch):
    """The launches of the batched transpose (the pairs of each) and the generic copies from here on."""
    from pyjuice.parameterizations.kernels.c import get_module
    mod = get_module()
    if mod is None or not hasattr(mod, "staging_transpose_batched"):
        pytest.skip("the CUDA extension did not build")
    calls = dict(batched = [], copies = 0)
    real_batched, real_copy = mod.staging_transpose_batched, torch._foreach_copy_

    def batched(rows, max_b, max_n):
        calls["batched"].append(len(rows) // 4)
        return real_batched(rows, max_b, max_n)

    def copy(dsts, srcs):
        calls["copies"] += 1
        return real_copy(dsts, srcs)

    monkeypatch.setattr(mod, "staging_transpose_batched", batched)
    monkeypatch.setattr(torch, "_foreach_copy_", copy)
    return calls


def _lowrank_chain(T = 4, K = 8, rank = 3, form = "sum_exp"):
    """A chain of `T` sums under LowRank (two tensors a node, whose slots alternate in the buffer)."""
    torch.manual_seed(0)
    with juice.set_block_size(K):
        ns = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
        members = []
        for t in range(1, T + 1):
            emit = inputs(t, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
            ns = summate(multiply(ns, emit), num_node_blocks = 2,
                         parameterization = LowRank(rank = rank, form = form))
            members.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    return root, members


def _factors(members, batch, dev, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    return [tuple((0.3 * torch.randn(sh, generator = g)).to(dev) for sh in m.parameterization.tensor_shapes(m, batch))
            for m in members]


def _cats(factors):
    return tuple(torch.cat(slot, dim = 1) for slot in zip(*factors))


@cuda_only
@pytest.mark.parametrize("form", ["sum_exp", "exp_sum"])
def test_a_group_of_several_tensors_goes_in_one_launch_each_way(monkeypatch, form):
    """A member's slot is one run in every sample of the concatenated tensor: every slot of every member goes to the
    buffer in one launch -- of the batched transpose where the storage puts the batch innermost ("sum_exp"), one
    strided copy a slot where it is the caller's own layout ("exp_sum": `_group_strided`) --, no view of a member made,
    and back to the caller's gradient tensors the same way -- bit for bit what the per-node form stages and computes,
    through one PC. A backward with no destinations after one with them writes nothing into them."""
    dev, batch, T = torch.device("cuda:0"), 40, 4                  # (a batch not a multiple of 32)
    root, members = _lowrank_chain(T = T, form = form)
    pc = juice.compile(root, verbose = False).to(dev)
    pc.register_external_params_group("tr", members)
    data = torch.randint(0, NUM_CATS, [batch, T + 1], device = dev)
    factors = _factors(members, batch, dev, 5)

    ll_p = pc(data, sum_external_params = dict(zip(members, factors))).clone()
    staged_p = pc.external_params.clone()
    pc.backward(data, flows_memory = 0.0)
    grads_p = [tuple(g.clone() for g in pc.get_external_params_grad(m)) for m in members]

    calls = _counting_staging(monkeypatch)
    cats = _cats(factors)
    one_launch = dict(batched = [2 * T] if form == "sum_exp" else [], copies = 0)    # (exp_sum: no per-node copies)
    if form == "exp_sum":
        assert pc._group_strided("tr", cats, batch, pc.external_params) is not None
    ll_g = pc(data, sum_external_params = {"tr": cats}).clone()
    assert calls == one_launch, calls
    assert torch.equal(pc.external_params, staged_p) and torch.equal(ll_g, ll_p)

    calls.update(batched = [], copies = 0)
    grads = tuple(torch.full_like(c, float("nan")) for c in cats)
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"tr": grads})
    torch.cuda.synchronize()
    assert calls == one_launch, calls
    for slot, got in enumerate(grads):
        assert torch.equal(got, torch.cat([g[slot] for g in grads_p], dim = 1)), f"slot {slot}"
    assert float(grads[0].abs().max()) > 0.0
    assert pc._external_grad_groups is None, "the caller's tensors are let go"

    kept = tuple(g.clone() for g in grads)
    pc(data, sum_external_params = {"tr": tuple(0.5 * c for c in cats)})
    pc.backward(data, flows_memory = 0.0)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(grads, kept)), "a backward with no destinations wrote into the last's"


@cuda_only
def test_a_group_in_one_launch_at_other_batches_and_where_the_buffer_moved(monkeypatch):
    """What is kept of a group is per batch size and holds no address: other batches -- one large enough to move the
    staging buffer -- and fresh tensors each call stage what they should, in one launch each, gradients coming back too."""
    dev, T = torch.device("cuda:0"), 3
    root, members = _lowrank_chain(T = T)
    pc = juice.compile(root, verbose = False).to(dev)
    pc.register_external_params_group("tr", members)
    calls = _counting_staging(monkeypatch)
    for seed, batch in enumerate((40, 8, 1024, 8, 40)):
        data = torch.randint(0, NUM_CATS, [batch, T + 1], device = dev)
        factors = _factors(members, batch, dev, 100 + seed)
        calls.update(batched = [], copies = 0)
        pc(data, sum_external_params = {"tr": _cats(factors)})
        assert calls == dict(batched = [2 * T], copies = 0), f"batch {batch}: {calls}"
        for m, tensors in zip(members, factors):
            for staged, t in zip(pc._staged_external_params[m], tensors):
                assert torch.equal(staged, t.permute(1, 2, 0)), f"batch {batch}"
        grads = tuple(torch.empty_like(c) for c in _cats(factors))
        pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"tr": grads})
        torch.cuda.synchronize()
        for slot, got in enumerate(grads):
            want = torch.cat([pc.get_external_params_grad(m)[slot] for m in members], dim = 1)
            assert torch.equal(got, want), f"batch {batch}, slot {slot}"


@cuda_only
def test_a_group_whose_members_differ_along_the_group_axis_in_one_launch(monkeypatch):
    """Members of different node counts: the runs have their own lengths, the grid spans the longest, the other
    pair's extra blocks return (the batched transpose: a storage with the batch innermost, LowRank's "sum_exp")."""
    dev, batch, K, rank = torch.device("cuda:0"), 40, 8, 3
    torch.manual_seed(0)
    with juice.set_block_size(K):
        n = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(4)]
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1, parameterization = LowRank(rank = rank, form = "sum_exp"))
        s1 = summate(multiply(n[2], n[3]), num_node_blocks = 3, parameterization = LowRank(rank = rank, form = "sum_exp"))
        top = multiply(s0, s1, edge_ids = torch.tensor([[0, 0], [0, 1], [0, 2]]))
        root = summate(top, num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)
    pc.register_external_params_group("two", [s0, s1])
    data = torch.randint(0, NUM_CATS, [batch, 4], device = dev)
    factors = _factors([s0, s1], batch, dev, 7)
    assert factors[0][0].size(1) != factors[1][0].size(1)

    ll_p = pc(data, sum_external_params = {s0: factors[0], s1: factors[1]}).clone()
    staged_p = pc.external_params.clone()
    pc.backward(data, flows_memory = 0.0)
    want = tuple(torch.cat([pc.get_external_params_grad(m)[slot] for m in (s0, s1)], dim = 1) for slot in range(2))

    calls = _counting_staging(monkeypatch)
    cats = _cats(factors)
    assert torch.equal(pc(data, sum_external_params = {"two": cats}), ll_p)
    assert torch.equal(pc.external_params, staged_p) and calls == dict(batched = [4], copies = 0), calls
    grads = tuple(torch.empty_like(c) for c in cats)
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"two": grads})
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(grads, want))


@cuda_only
def test_a_group_of_one_tensor_a_node_out_of_the_buffers_order_in_one_launch(monkeypatch):
    """Members that are not adjacent in the group's order are not one transpose, but they are one launch."""
    dev, batch = torch.device("cuda:0"), 40
    root, s0, s1 = _two_gated()
    pc = juice.compile(root, verbose = False).to(dev)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0, g1 = _gates(s0, batch, dev, 11), _gates(s1, batch, dev, 12)
    ll_p = pc(data, sum_external_params = {s0: g0, s1: g1}).clone()
    staged_p = pc.external_params.clone()
    pc.backward(data, flows_memory = 0.0)
    want = torch.cat([pc.get_external_params_grad(s1)[0], pc.get_external_params_grad(s0)[0]], dim = 1)

    pc.register_external_params_group("rev", [s1, s0])
    calls = _counting_staging(monkeypatch)
    cat = torch.cat([g1, g0], dim = 1)
    assert torch.equal(pc(data, sum_external_params = {"rev": cat}), ll_p)
    assert torch.equal(pc.external_params, staged_p) and calls == dict(batched = [2], copies = 0), calls
    got = torch.empty_like(cat)
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"rev": got})
    torch.cuda.synchronize()
    assert torch.equal(got, want)


class _CallersOrder(SumParameterization):
    """Two tensors a node, stored in the caller's order (`storage_perm` None): not a transpose, whatever it holds."""

    def get_signature(self):
        return "CallersOrder"

    def tensor_shapes(self, ns, batch_size):
        return ((batch_size, ns.num_nodes, 2), (batch_size, 3, ns.num_ch_nodes))

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs):
        pass

    def pre_backward(self, layer, ns_info, tensors, node_flows, element_flows, node_mars, element_mars, params, **kwargs):
        pass

    def post_backward(self, layer, ns_info, tensors, grad_tensors, node_flows, element_flows, node_mars, element_mars,
                      params, param_flows = None, **kwargs):
        pass


@cuda_only
@pytest.mark.parametrize("order", ["buffer", "reversed"])
def test_a_group_stored_in_the_callers_order_goes_as_one_strided_copy_a_slot(monkeypatch, order):
    """Storage that is the caller's own layout: a group of members of one shape, their slots evenly spaced in the
    buffer in the group's order, is one strided copy a slot (`_group_strided`: no batched transpose, no per-node
    copies) -- the members named in the other order, the per-node way (one batched copy of every member's view)."""
    dev, batch = torch.device("cuda:0"), 40
    torch.manual_seed(0)
    with juice.set_block_size(8):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(3)]
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1, parameterization = _CallersOrder())
        s1 = summate(multiply(s0, n[2]), num_node_blocks = 1, parameterization = _CallersOrder())
        root = summate(multiply(s1), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)
    members = [s0, s1]
    _, ns2slots = pc._external_params_layout(batch)
    if (ns2slots[s0][0][0] < ns2slots[s1][0][0]) != (order == "buffer"):
        members = members[::-1]
    pc.register_external_params_group("two", members)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    factors = _factors(members, batch, dev, 3)
    calls = _counting_staging(monkeypatch)
    pc(data, sum_external_params = {"two": _cats(factors)})
    assert calls["batched"] == [] and calls["copies"] == (0 if order == "buffer" else 1), calls
    for m, tensors in zip(members, factors):
        for staged, t in zip(pc._staged_external_params[m], tensors):
            assert torch.equal(staged, t)


@cuda_only
def test_what_a_group_in_one_launch_does_not_take_goes_the_per_node_way(monkeypatch):
    """Tensors that are not contiguous fp32 of the concatenation's exact shape, on the buffer's device, are the per-node
    path's: which stages what it can, and says what is wrong with the rest. So does a node named twice, among
    the gradients' destinations as among the tensors, a group member that got no external parameters, and a group mixed
    with the nodes it leaves out."""
    dev, batch, T = torch.device("cuda:0"), 40, 4
    root, members = _lowrank_chain(T = T)
    pc = juice.compile(root, verbose = False).to(dev)
    pc.register_external_params_group("tr", members)
    pc.register_external_params_group("head", members[:2])
    data = torch.randint(0, NUM_CATS, [batch, T + 1], device = dev)
    factors = _factors(members, batch, dev, 9)
    cats = _cats(factors)
    pc(data, sum_external_params = dict(zip(members, factors)))
    staged_p = pc.external_params.clone()
    calls = _counting_staging(monkeypatch)

    # not contiguous: every other row of a taller tensor
    def spaced(c):
        big = torch.empty(2 * c.size(0), *c.shape[1:], device = dev)
        big[::2] = c
        return big[::2]
    pc(data, sum_external_params = {"tr": tuple(spaced(c) for c in cats)})
    assert calls["batched"] == [] and calls["copies"] == 1 and torch.equal(pc.external_params, staged_p), calls

    # a group with the nodes it leaves out, in one dict
    calls.update(batched = [], copies = 0)
    pc(data, sum_external_params = {"head": _cats(factors[:2]), **dict(zip(members[2:], factors[2:]))})
    assert calls["batched"] == [4, 4] and torch.equal(pc.external_params, staged_p), calls

    # another dtype, another device: refused, never read as float32 bytes or through a host address
    with pytest.raises(AssertionError, match = "float32"):
        pc(data, sum_external_params = {"tr": tuple(c.double() for c in cats)})
    with pytest.raises(AssertionError, match = "is on"):
        pc(data, sum_external_params = {"tr": tuple(c.cpu() for c in cats)})

    # the slots the wrong way round: right rank, wrong extents
    with pytest.raises(AssertionError, match = "concatenation"):
        pc(data, sum_external_params = {"tr": cats[::-1]})
    with pytest.raises(AssertionError, match = "expects 2 concatenated"):
        pc(data, sum_external_params = {"tr": cats[0]})
    with pytest.raises(AssertionError, match = "both directly and through group"):
        pc(data, sum_external_params = {"tr": cats, members[0]: factors[0]})

    # the gradients' destinations
    pc(data, sum_external_params = {"tr": cats})
    grads = tuple(torch.empty_like(c) for c in cats)
    with pytest.raises(AssertionError, match = "both directly and through group"):
        pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"tr": grads, members[0]: factors[0]})
    pc(data, sum_external_params = dict(zip(members[:3], factors[:3])))
    with pytest.raises(AssertionError, match = "not given external"):
        pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"tr": grads})
    pc(data, sum_external_params = {"tr": cats})
    with pytest.raises(AssertionError, match = "shape"):
        pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"tr": grads[::-1]})


@cuda_only
def test_backward_can_fill_caller_supplied_gradient_buffers():
    """`sum_external_params_grad` mirrors the forward: a dict keyed by `ns` OR by group name, whose
    tensors the backward fills.

    The kernels always write the PC's internal buffer -- they address it by compiled offset -- so this
    is a copy-out afterwards, and `get_external_params_grad` keeps working alongside it. Asking for
    destinations implies `compute_external_grads`, so there is no second flag to remember."""
    dev, batch = torch.device("cuda:0"), 64
    root, s0, s1 = _two_gated()
    pc = juice.compile(root, verbose = False).to(dev)

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [batch, 3], device = dev)
    g0, g1 = _gates(s0, batch, dev, 11), _gates(s1, batch, dev, 12)

    pc(data, sum_external_params = {s0: g0, s1: g1})
    pc.backward(data, flows_memory = 0.0)
    ref0 = pc.get_external_params_grad(s0)[0].clone()
    ref1 = pc.get_external_params_grad(s1)[0].clone()
    assert float(ref0.abs().max()) > 0.0, "the gradient is identically zero; nothing was computed"

    mine0, mine1 = torch.zeros_like(ref0), torch.zeros_like(ref1)
    pc(data, sum_external_params = {s0: g0, s1: g1})
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {s0: mine0, s1: mine1})
    assert torch.equal(mine0, ref0) and torch.equal(mine1, ref1)
    # the internal read still works after a copy-out
    assert torch.equal(pc.get_external_params_grad(s0)[0], ref0)

    # and by group name, in the group's own layout
    pc.register_external_params_group("both", [s0, s1])
    cat = torch.zeros_like(torch.cat([ref0, ref1], dim = 1))
    pc(data, sum_external_params = {"both": torch.cat([g0, g1], dim = 1)})
    pc.backward(data, flows_memory = 0.0, sum_external_params_grad = {"both": cat})
    assert torch.equal(cat, torch.cat([ref0, ref1], dim = 1))

    with pytest.raises(AssertionError, match = "shape"):
        pc(data, sum_external_params = {s0: g0, s1: g1})
        pc.backward(data, flows_memory = 0.0,
                    sum_external_params_grad = {s0: torch.zeros([batch, 1, 1], device = dev)})
