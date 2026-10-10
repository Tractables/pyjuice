"""
Categorical class masses (:mod:`pyjuice.constraints.distributions.categorical`) against a float64 brute force: per
node, the log of the summed probabilities of each class's tokens. Both passes -- the natural token order (up to
NATURAL_ORDER_MAX_CLASSES classes) and the class order (above) -- on tied (homogeneous HMM) and untied input layers
(and one whose tied nodes come out of their rows' order), for one class, two, a skewed split, a class with no
token, both sides of the boundary between the passes, and every token its own class; and on probabilities small
enough to be subnormal, which tensor cores would flush to zero.
"""
import math

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.constraints import distributions
from pyjuice.constraints.distributions import TokenClasses, categorical

V = 600
DEV = torch.device("cuda:0")
LABELS = ["one", "two", "skewed", "empty_class", "boundary", "past_boundary", "many", "each_token"]


def build(kind):
    torch.manual_seed(0)
    if kind == "tied":
        ns = juice.structures.HMM(seq_length = 4, num_latents = 8, num_emits = V)
    elif kind == "untied":
        ns = juice.structures.GeneralizedHMM(seq_length = 4, num_latents = 8, homogeneous = False,
                                             input_dist = dists.Categorical(num_cats = V))
    else:
        # nodes out of their rows' order: variable 2's nodes are tied to variable 1's, which follow variable 0's
        cat = lambda: dists.Categorical(num_cats = V)
        x0 = inputs(0, num_node_blocks = 1, block_size = 2, dist = cat())
        x1 = inputs(1, num_node_blocks = 2, block_size = 2, dist = cat())
        x2 = x1.duplicate(2, tie_params = True)
        m = [multiply(summate(x0, num_node_blocks = 2, block_size = 2), x1, x2)]
        ns = summate(*m, num_node_blocks = 1, block_size = 1)
    ns.init_parameters(perturbation = 4.0)
    return juice.compile(ns, verbose = False).to(DEV)


def labels(name):
    """(token_class [V], number of classes)."""
    g = torch.Generator().manual_seed(1)
    M = categorical.NATURAL_ORDER_MAX_CLASSES
    skew = lambda C: torch.cat([torch.arange(C), torch.where(torch.rand(V - C, generator = g) < 0.7, 0,
                                                             torch.randint(1, C, (V - C,), generator = g))])
    if name == "one":
        return torch.zeros(V, dtype = torch.long), 1
    if name == "two":
        return torch.randint(0, 2, (V,), generator = g), 2
    if name == "skewed":                       # like Ctrl-G's: most tokens in one class, a few in many small ones
        return skew(10)[torch.randperm(V, generator = g)], 10
    if name == "empty_class":
        return torch.randint(0, 3, (V,), generator = g), 4
    if name == "boundary":
        return skew(M)[torch.randperm(V, generator = g)], M
    if name == "past_boundary":
        return skew(M + 1)[torch.randperm(V, generator = g)], M + 1
    if name == "many":                         # the class order, with classes spanning many chunks and empty ones
        lab = skew(100)[torch.randperm(V, generator = g)]
        return lab, 103
    assert name == "each_token"
    return torch.randperm(V, generator = g), V


def brute_force(layer, token_class, num_classes):
    probs = layer.params.double()[layer.s_pids[:, None] + torch.arange(V, device = DEV)[None, :]]    # [nodes, V]
    masses = torch.zeros(probs.size(0), num_classes, dtype = torch.float64, device = DEV)
    return masses.index_add_(1, token_class.long(), probs).log()


def check(got, want):
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert (got[fin].double() - want[fin]).abs().max() < 1e-5


@pytest.mark.parametrize("name", LABELS)
@pytest.mark.parametrize("kind", ["tied", "untied", "tied_out_of_order"])
def test_class_masses_match_brute_force(kind, name):
    pc = build(kind)
    token_class, C = labels(name)
    classes = TokenClasses(token_class.to(DEV), C)
    for layer in pc.input_layer_group:
        if kind == "tied_out_of_order":
            rows = layer.s_pids // V
            assert not torch.equal(rows, torch.arange(rows.numel(), device = DEV) % (rows.max() + 1))
        want = brute_force(layer, classes.token_class, C)
        got = categorical.class_masses(layer, classes)
        assert got.shape == (layer.num_nodes, C) and got.dtype == torch.float32
        check(got, want)
        assert torch.isneginf(want).any() == (name in ("empty_class", "many"))
        out = torch.full((layer.num_nodes, C), float("nan"), device = DEV)
        assert categorical.class_masses(layer, classes, out = out) is out
        check(out, want)
    # the class order is built only past the boundary, once
    assert bool(classes._by_class) == (C > categorical.NATURAL_ORDER_MAX_CLASSES)


@pytest.mark.parametrize("name", ["skewed", "many"])
def test_subnormal_probabilities_are_kept(name):
    """Probabilities down to 1e-44 (subnormal in fp32): every class keeps its mass, even one whose tokens are all
    subnormal."""
    pc = build("untied")
    layer = pc.input_layer_group[0]
    token_class, C = labels(name)
    g = torch.Generator(device = DEV).manual_seed(0)
    with torch.no_grad():
        layer.params.copy_(torch.exp(-torch.rand(layer.params.shape, device = DEV, generator = g) * 100.0))
        layer.params[layer.s_pids[0] + torch.nonzero(token_class == 1).flatten().to(DEV)] = 1e-44
    assert (layer.params < 1.17e-38).any()
    classes = TokenClasses(token_class.to(DEV), C)
    want = brute_force(layer, classes.token_class, C)
    assert want[0, 1] < math.log(1.17e-38)                             # class 1 of node 0: a subnormal mass
    check(categorical.class_masses(layer, classes), want)


def test_the_class_order_is_built_once():
    classes = TokenClasses(torch.randperm(V, device = DEV), V)
    assert classes.by_class(categorical.CLASS_CHUNK) is classes.by_class(categorical.CLASS_CHUNK)
    moved = classes.to(torch.device("cpu"))
    assert moved.token_class.device.type == "cpu" and not moved._by_class


def test_supported_distributions_are_matched_by_exact_type():
    class MyCategorical(dists.Categorical):
        pass

    assert distributions.lookup(dists.Categorical(num_cats = 5)) is categorical
    assert distributions.lookup(MyCategorical(num_cats = 5)) is None
    assert distributions.lookup(dists.Gaussian(mu = 0.0, sigma = 1.0)) is None
    assert categorical.num_values(dists.Categorical(num_cats = 7)) == 7
