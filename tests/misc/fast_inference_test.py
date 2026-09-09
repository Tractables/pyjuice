"""
`pyjuice.fast_inference()`: the scope in which layers may hold derived parameter copies.

Nothing READS the copy yet -- the transposed soft-evidence forward is a separate change -- so these
tests are about its lifetime, which is the part that can go wrong silently. The copy is only sound
because it cannot outlive a parameter change, and there is no way to detect one after the fact:
`mini_batch_em` writes `params` through a Triton kernel and `tensor._version` never moves (measured;
so does `params.data[...] = x`). Every guarantee therefore comes from the scope and from the hooks
on the in-repo write paths, and that is what is pinned here.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.utils.fast_inference import is_active, param_copies_allowed


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_CATS = 32


def _softevi_pc(num_vars = 4, states = 8):
    with juice.set_block_size(states):
        ns = inputs(0, num_node_blocks = 1,
                    dist = dists.SoftEvidenceCategorical(num_cats = NUM_CATS))
        for v in range(1, num_vars):
            ns = summate(multiply(ns, inputs(v, num_node_blocks = 1,
                                             dist = dists.SoftEvidenceCategorical(num_cats = NUM_CATS))),
                         num_node_blocks = 1)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _copies(pc):
    return [l._fast_inference_params for l in pc.input_layer_group]


def _data(pc, batch = 8):
    return torch.randint(0, NUM_CATS, [batch, pc.num_vars], device = torch.device("cuda:0"))


def test_the_scope_is_a_mode_and_nests():
    assert not is_active() and not param_copies_allowed()
    with juice.fast_inference():
        assert is_active() and param_copies_allowed()
        with juice.fast_inference(allow_param_copy = False):
            # an inner scope may NARROW what the outer one permitted ...
            assert is_active() and not param_copies_allowed()
        # ... and the outer scope's permission comes back when it exits
        assert param_copies_allowed()
    assert not is_active() and not param_copies_allowed()


def test_an_inner_scope_cannot_widen_an_outer_one():
    """A caller who bounded memory at the outer scope should not be overruled from inside."""
    with juice.fast_inference(allow_param_copy = False):
        with juice.fast_inference(allow_param_copy = True):
            assert not param_copies_allowed()


def test_the_scope_unwinds_on_an_exception():
    with pytest.raises(RuntimeError):
        with juice.fast_inference():
            assert is_active()
            raise RuntimeError("boom")
    assert not is_active()


@cuda_only
def test_a_copy_is_built_lazily_and_freed_on_exit():
    pc = _softevi_pc()
    data = _data(pc)

    assert all(c is None for c in _copies(pc))
    with juice.fast_inference():
        # LAZY: entering the scope alone builds nothing ...
        assert all(c is None for c in _copies(pc))
        pc(data)
        # ... the first forward inside it does
        built = _copies(pc)
        assert all(c is not None for c in built)
        layer = pc.input_layer_group[0]
        assert built[0].shape == (NUM_CATS, layer.params.numel() // NUM_CATS)
    assert all(c is None for c in _copies(pc))


@cuda_only
def test_no_copy_when_the_scope_forbids_it():
    pc = _softevi_pc()
    with juice.fast_inference(allow_param_copy = False):
        pc(_data(pc))
        assert all(c is None for c in _copies(pc))


@cuda_only
def test_no_copy_outside_a_scope():
    pc = _softevi_pc()
    pc(_data(pc))
    assert all(c is None for c in _copies(pc))


@cuda_only
def test_the_copy_is_the_transposed_emission_table():
    pc = _softevi_pc()
    layer = pc.input_layer_group[0]
    with juice.fast_inference():
        pc(_data(pc))
        rows = layer.params.numel() // NUM_CATS
        assert torch.equal(layer._fast_inference_params,
                           layer.params.detach().view(rows, NUM_CATS).t().contiguous())


@cuda_only
@pytest.mark.parametrize("write", ["mini_batch_em", "_init_parameters", "to"])
def test_every_in_repo_parameter_write_drops_the_copy(write):
    """
    The hooks are the defence against a copy outliving a parameter change WITHIN a scope.

    They are not the reason the design is safe -- the scope's lifetime is -- but a stale copy here
    would be silent, so each path is pinned separately rather than trusted.
    """
    pc = _softevi_pc()
    data = _data(pc)
    with juice.fast_inference():
        pc(data)
        assert pc.input_layer_group[0]._fast_inference_params is not None

        if write == "mini_batch_em":
            pc.backward(data, flows_memory = 0.0)
            pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
        elif write == "_init_parameters":
            for layer in pc.input_layer_group:
                layer._init_parameters(perturbation = 1.0)
        else:
            for layer in pc.input_layer_group:
                layer.to(torch.device("cuda:0"))

        assert all(c is None for c in _copies(pc)), f"{write} left a stale copy behind"


@cuda_only
def test_the_copy_does_not_change_the_answer():
    """Nothing reads it yet, so the forward must be unaffected either way."""
    pc = _softevi_pc()
    data = _data(pc)
    outside = pc(data).clone()
    with juice.fast_inference():
        inside = pc(data).clone()
    after = pc(data).clone()
    assert torch.equal(outside, inside) and torch.equal(outside, after)


@cuda_only
def test_the_memory_is_actually_returned():
    """A freed copy has to give the allocator its block back, or the scope leaks per entry."""
    pc = _softevi_pc(num_vars = 6, states = 32)
    data = _data(pc)
    pc(data)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    with juice.fast_inference():
        pc(data)
        torch.cuda.synchronize()
        inside = torch.cuda.memory_allocated()
        expected = sum(l.params.numel() * l.params.element_size() for l in pc.input_layer_group)
        assert inside - base >= expected * 0.9, "the copy does not look allocated"

    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= base + expected * 0.1, "the copy was not released"


@cuda_only
def test_repeated_entries_do_not_accumulate():
    pc = _softevi_pc()
    data = _data(pc)
    pc(data); torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for _ in range(5):
        with juice.fast_inference():
            pc(data)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= base + 1024, "entries leak"


@cuda_only
def test_a_distribution_with_no_use_for_a_copy_builds_nothing():
    with juice.set_block_size(8):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        for v in range(1, 4):
            ns = summate(multiply(ns, inputs(v, num_node_blocks = 1,
                                             dist = dists.Categorical(num_cats = NUM_CATS))),
                         num_node_blocks = 1)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    with juice.fast_inference():
        pc(_data(pc))
        assert all(c is None for c in _copies(pc))
