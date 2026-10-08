"""
`conditional(pc, ..., target_vars = T)` must equal `conditional(pc, ...)[:, T, :]`.

Issue #21 reported that it did not. For Categorical leaves the original report was fixed by PR #40 (input
layers used block ids as node offsets); the DiscreteLogistic path had the same class of bug -- its
kernel read each target node's parameters and metadata through the position in the target list instead
of the node id, so every node was combined with another node's mean, scale and value range.
"""
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def build(kind, dist):
    torch.manual_seed(0); random.seed(0)              # RAT-SPN shuffles scopes with `random`
    if kind == "hmm":
        ns = juice.structures.GeneralizedHMM(seq_length = 12, num_latents = 32, homogeneous = False, input_dist = dist)
        V = 12
    elif kind == "hclt":
        ns = juice.structures.HCLT(torch.randint(0, 8, (300, 12)), num_latents = 32, input_dist = dist)
        V = 12
    elif kind == "pd":
        ns = juice.structures.PD(data_shape = (4, 4), num_latents = 16, split_intervals = (2, 2), input_dist = dist)
        V = 16
    else:
        ns = juice.structures.RAT_SPN(num_vars = 8, num_latents = 16, depth = 2, num_repetitions = 2, input_dist = dist)
        V = 8
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(torch.device("cuda:0")), V


DISTS = {
    "categorical": lambda: dists.Categorical(num_cats = 8),
    "discrete_logistic": lambda: dists.DiscreteLogistic(val_range = (-1.0, 1.0), num_cats = 8),
}


@pytest.mark.parametrize("dist", list(DISTS))
@pytest.mark.parametrize("kind", ["hmm", "hclt", "pd", "rat_spn"])
def test_target_vars_match_the_full_conditional(kind, dist):
    pc, V = build(kind, DISTS[dist]())
    dev = torch.device("cuda:0")
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 8, (5, V), generator = g).to(dev)
    missing = torch.zeros(5, V, dtype = torch.bool)
    missing[:, ::3] = True
    missing = missing.to(dev)

    targets = sorted(random.Random(2).sample(range(V), max(2, V // 4)))
    for T in (targets, targets[::-1], [targets[0]]):              # sorted, unsorted, single
        sub = juice.queries.conditional(pc, data = x, missing_mask = missing, target_vars = T)   # subset first
        full = juice.queries.conditional(pc, data = x, missing_mask = missing)
        assert sub.shape == (5, len(T), 8)
        assert torch.allclose(sub, full[:, T, :], atol = 1e-5), (kind, dist, T, float((sub - full[:, T, :]).abs().max()))


def test_a_target_vars_query_leaves_later_passes_unchanged():
    pc, V = build("hmm", DISTS["categorical"]())
    x = torch.randint(0, 8, (5, V), generator = torch.Generator().manual_seed(3)).to(torch.device("cuda:0"))
    lls = pc(x).clone()
    full = juice.queries.conditional(pc, data = x)
    juice.queries.conditional(pc, data = x, target_vars = [1, 4])
    assert torch.equal(pc(x), lls)
    assert torch.allclose(juice.queries.conditional(pc, data = x), full, atol = 1e-6)
