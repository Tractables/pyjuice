"""
`is_structured_decomposable`, and the `TensorCircuit` flag derived from it.

A PC is structured decomposable when every product node decomposes its scope the same way -- for any
two products over the same scope, the partition into their children's scopes is identical -- which is
exactly the condition under which a single vtree exists.

The flag is not decorative: the top-down sampler can precompute its entire index plan when it holds,
because the frontier's shape is then a function of the scopes alone and never of which node a draw
selected. A flag that is wrongly `True` would enable that cache on a circuit whose plan actually
varies, so these tests pin both directions -- and in particular the two cases that made an earlier
version of the check wrong in each direction.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes.methods import is_structured_decomposable


def _leaves(num_vars = 4, num_node_blocks = 2):
    return [inputs(v, num_node_blocks = num_node_blocks, dist = dists.Categorical(num_cats = 5))
            for v in range(num_vars)]


def test_one_split_per_scope_is_structured_decomposable():
    with juice.set_block_size(4):
        x = _leaves()
        a0 = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        a1 = summate(multiply(x[0], x[1]), num_node_blocks = 2)      # same split, twice
        b = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        root = summate(multiply(a0, b), multiply(a1, b), num_node_blocks = 1, block_size = 1)

    assert is_structured_decomposable(root)


def test_two_splits_of_one_scope_are_not():
    """`{0,1}{2,3}` and `{0,2}{1,3}` over the same scope: no vtree admits both."""
    with juice.set_block_size(4):
        x = _leaves()
        a = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        b = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        c = summate(multiply(x[0], x[2]), num_node_blocks = 2)
        d = summate(multiply(x[1], x[3]), num_node_blocks = 2)
        root = summate(multiply(a, b), multiply(c, d), num_node_blocks = 1, block_size = 1)

    assert not is_structured_decomposable(root)


def test_a_one_child_product_is_not_a_split():
    """
    REGRESSION, false negative. pyjuice caps a PC with `summate(multiply(ns), ...)`, so every circuit
    has a UNARY product over the full scope. Counting it as a decomposition made it conflict with the
    real split of the same scope, and even an HMM -- which follows an obvious linear vtree -- came out
    non-decomposable.
    """
    with juice.set_block_size(4):
        x = _leaves()
        a = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        b = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        full = summate(multiply(a, b), num_node_blocks = 2)           # scope {0,1,2,3}
        root = summate(multiply(full), num_node_blocks = 1, block_size = 1)

    assert is_structured_decomposable(root)


def test_a_sum_node_mixing_a_flat_component_with_a_split_is_not():
    """
    REGRESSION, false positive. Ignoring unary products (the fix above) opens exactly one hole: a sum
    node offering BOTH a flat component and a decomposed one over the same scope. Whichever the
    sampler draws pushes a different number of nodes onto the frontier, so the plan is not invariant
    and the flag must say so.
    """
    with juice.set_block_size(4):
        x = _leaves()
        a = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        b = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        flat = summate(multiply(a, b), num_node_blocks = 2)
        root = summate(multiply(flat), multiply(a, b), num_node_blocks = 1, block_size = 1)

    assert not is_structured_decomposable(root)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
@pytest.mark.parametrize("name,expected", [("hmm", True), ("hclt", True), ("rat_spn", False)])
def test_library_structures(name, expected):
    """
    `HMM` and `HCLT` follow one vtree. `RAT_SPN` does not: `num_repetitions` builds several
    independent random splits of the same scope, which is the point of the structure. (`PD` is the
    same story via `split_intervals`, and is left out here only because it is slow to build.)
    """
    torch.manual_seed(0)
    if name == "hmm":
        ns = juice.structures.HMM(seq_length = 6, num_latents = 32, num_emits = 8, homogeneous = True)
    elif name == "hclt":
        data = torch.randint(0, 16, [512, 16]).float().to(torch.device("cuda:0"))
        ns = juice.structures.HCLT(data, num_bins = 8, sigma = 0.5 / 8, num_latents = 8,
                                   chunk_size = 8)
    else:
        ns = juice.structures.RAT_SPN(num_vars = 16, num_latents = 8, depth = 2,
                                      num_repetitions = 2, num_pieces = 2,
                                      input_dist = dists.Categorical(num_cats = 8))

    assert is_structured_decomposable(ns) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_the_flag_is_set_on_a_compiled_circuit():
    torch.manual_seed(0)
    with juice.set_block_size(4):
        x = _leaves()
        a = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        b = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        root = summate(multiply(a, b), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)

    pc = juice.compile(root, verbose = False)
    assert pc.is_structured_decomposable is True
    assert is_structured_decomposable(pc) is True        # also accepts a compiled circuit
