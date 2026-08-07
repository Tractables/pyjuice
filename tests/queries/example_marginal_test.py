"""
A worked example: marginal probabilities.

`juice.queries.marginal(pc, data, missing_mask)` returns `log P(e)` -- the probability of the
variables you left OBSERVED, with the ones you marked missing summed out. Marginalising is exact and
costs one forward pass, whatever the subset, which is the property probabilistic circuits are for.

The convention that matters: **`missing_mask = True` means "sum this variable out"**. Observed
variables are the ones marked `False`, and their values come from `data`.

Every claim below is checked against a brute-force sum over the marginalised variable, which is only
practical because the circuit here is deliberately tiny. The thorough tests live in
`queries/marginal/`.
"""

import itertools

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_VARS = 3
NUM_CATS = 3


def _build():
    """A small mixture over three variables -- little enough to enumerate by hand."""
    with juice.set_block_size(2):
        i = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(NUM_VARS)]
        left = summate(multiply(i[0], i[1]), num_node_blocks = 2)
        root = summate(multiply(left, i[2]), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 4.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _all_observed(pc):
    return torch.zeros([NUM_VARS], dtype = torch.bool, device = pc.device)


@cuda_only
def test_a_full_assignment_gives_the_joint_probability():
    """With nothing marked missing, `marginal` is just the joint `log P(x)`."""
    torch.manual_seed(0)
    pc = _build()

    data = torch.randint(0, NUM_CATS, [8, NUM_VARS], device = pc.device)
    log_p = juice.queries.marginal(pc, data = data, missing_mask = _all_observed(pc)).flatten()

    assert log_p.shape == (8,)
    assert bool((log_p <= 1e-5).all()), "a probability cannot exceed 1"


@cuda_only
def test_the_whole_distribution_sums_to_one():
    """Enumerate every assignment of all three variables; the joint must sum to exactly 1."""
    torch.manual_seed(0)
    pc = _build()

    every = torch.tensor(list(itertools.product(range(NUM_CATS), repeat = NUM_VARS)),
                         device = pc.device)
    total = juice.queries.marginal(pc, data = every,
                               missing_mask = _all_observed(pc)).detach().exp().sum()

    assert abs(float(total) - 1.0) < 1e-4, f"the joint sums to {float(total):.6f}"


@cuda_only
def test_marginalising_a_variable_equals_summing_it_out():
    """
    The defining property, checked directly.

    `P(x_0, x_2)` with variable 1 marked missing must equal the sum of `P(x_0, x_1, x_2)` over every
    value of variable 1 -- which is what "exact marginal inference" means.
    """
    torch.manual_seed(0)
    pc = _build()

    observed = torch.randint(0, NUM_CATS, [16, NUM_VARS], device = pc.device)

    missing = _all_observed(pc)
    missing[1] = True                                   # sum variable 1 out
    marginalised = juice.queries.marginal(pc, data = observed, missing_mask = missing).exp().flatten()

    by_hand = torch.zeros_like(marginalised)
    for value in range(NUM_CATS):
        filled = observed.clone()
        filled[:, 1] = value                            # ... by enumerating it instead
        by_hand += juice.queries.marginal(pc, data = filled,
                                          missing_mask = _all_observed(pc)).exp().flatten()

    assert float((marginalised - by_hand).abs().max()) < 1e-5


@cuda_only
def test_marginalising_everything_gives_one():
    """Nothing observed, so there is nothing left to be uncertain about: `P() = 1`."""
    torch.manual_seed(0)
    pc = _build()

    data = torch.randint(0, NUM_CATS, [4, NUM_VARS], device = pc.device)
    missing = torch.ones([NUM_VARS], dtype = torch.bool, device = pc.device)

    log_p = juice.queries.marginal(pc, data = data, missing_mask = missing).flatten()

    assert float(log_p.abs().max()) < 1e-5, "summing every variable out should give log 1 = 0"
