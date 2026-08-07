"""
A worked example: conditional distributions.

`juice.queries.conditional(pc, data, missing_mask)` returns `P(x_v = k | e)` for every variable and
every category at once, shaped `[batch, num_vars, num_cats]`. One forward and one backward pass gives
the whole table -- you do not ask per variable.

Same convention as `marginal`: **`missing_mask = True` marks the variables to sum out**, so those are
the ones you get a distribution over, and the `False` ones are the evidence.

The interesting check is the last one, which recovers the same numbers from `marginal` alone via

    P(x_v = k | e)  =  P(e, x_v = k) / sum_j P(e, x_v = j)

so the example doubles as a statement of how the two queries relate. Thorough tests live in
`queries/conditional/`.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_VARS = 3
NUM_CATS = 3
BATCH = 8


def _build():
    with juice.set_block_size(2):
        i = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(NUM_VARS)]
        left = summate(multiply(i[0], i[1]), num_node_blocks = 2)
        root = summate(multiply(left, i[2]), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 4.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _evidence(pc):
    """Observe variables 0 and 2; ask about variable 1."""
    torch.manual_seed(1)
    data = torch.randint(0, NUM_CATS, [BATCH, NUM_VARS], device = pc.device)
    missing = torch.zeros([NUM_VARS], dtype = torch.bool, device = pc.device)
    missing[1] = True
    return data, missing


@cuda_only
def test_the_shape_and_the_convention():
    """One row per (sample, variable), holding a distribution over the categories."""
    torch.manual_seed(0)
    pc = _build()
    data, missing = _evidence(pc)

    probs = juice.queries.conditional(pc, data = data, missing_mask = missing)

    assert probs.shape == (BATCH, NUM_VARS, NUM_CATS)
    assert bool((probs >= -1e-6).all()), "probabilities cannot be negative"


@cuda_only
def test_the_queried_variable_gets_a_proper_distribution():
    """Variable 1 was marked missing, so its row is `P(x_1 = k | x_0, x_2)` and must sum to 1."""
    torch.manual_seed(0)
    pc = _build()
    data, missing = _evidence(pc)

    probs = juice.queries.conditional(pc, data = data, missing_mask = missing)
    total = probs[:, 1, :].sum(dim = 1)

    assert float((total - 1.0).abs().max()) < 1e-4, \
        f"the queried variable's distribution sums to {float(total.min()):.6f}..{float(total.max()):.6f}"


@cuda_only
def test_it_agrees_with_the_marginal_query():
    """
    The relation that defines a conditional, checked end to end against a DIFFERENT query.

    `P(x_1 = k | e)` is `P(e, x_1 = k)` renormalised over `k`, and every term on the right is
    something `marginal` can answer -- so this pins `conditional` against machinery it does not
    share, rather than against itself.
    """
    torch.manual_seed(0)
    pc = _build()
    data, missing = _evidence(pc)

    probs = juice.queries.conditional(pc, data = data, missing_mask = missing)[:, 1, :]

    all_observed = torch.zeros([NUM_VARS], dtype = torch.bool, device = pc.device)
    joint = []
    for value in range(NUM_CATS):
        filled = data.clone()
        filled[:, 1] = value
        joint.append(juice.queries.marginal(pc, data = filled,
                                            missing_mask = all_observed).exp().flatten())

    joint = torch.stack(joint, dim = 1)                 # [batch, num_cats]
    expected = joint / joint.sum(dim = 1, keepdim = True)

    assert float((probs - expected).abs().max()) < 1e-5


@cuda_only
def test_evidence_actually_changes_the_answer():
    """
    A guard on the example itself: if the conditional came back the same whatever was observed, every
    check above would still pass while the query did nothing.
    """
    torch.manual_seed(0)
    pc = _build()
    _, missing = _evidence(pc)

    first = torch.zeros([BATCH, NUM_VARS], dtype = torch.long, device = pc.device)
    second = torch.full([BATCH, NUM_VARS], NUM_CATS - 1, dtype = torch.long, device = pc.device)

    a = juice.queries.conditional(pc, data = first, missing_mask = missing)[:, 1, :]
    b = juice.queries.conditional(pc, data = second, missing_mask = missing)[:, 1, :]

    assert float((a - b).abs().max()) > 1e-3, "the evidence made no difference to the answer"
