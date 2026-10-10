"""
Categorical class masses (:mod:`pyjuice.constraints.distributions.categorical`) against a float64 brute force: per
node, the log of the summed probabilities of each class's tokens. On tied (homogeneous HMM) and untied input
layers (and one whose tied nodes come out of their rows' order), for one class, two, a skewed split, a class
with no token, and every token its own class (more classes than one matrix product takes).
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.constraints import distributions
from pyjuice.constraints.distributions import categorical

V = 600
DEV = torch.device("cuda:0")


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
    if name == "one":
        return torch.zeros(V, dtype = torch.long), 1
    if name == "two":
        return torch.randint(0, 2, (V,), generator = g), 2
    if name == "skewed":                       # like Ctrl-G's: most tokens in one class, a few in many small ones
        u = torch.rand(V, generator = g)
        return torch.where(u < 0.7, 0, torch.where(u < 0.9, 1, 2 + (u * 1e4).long() % 8)), 10
    if name == "empty_class":
        return torch.randint(0, 3, (V,), generator = g), 4
    assert name == "each_token"
    return torch.randperm(V, generator = g), V


def brute_force(layer, token_class, num_classes):
    probs = layer.params.double()[layer.s_pids[:, None] + torch.arange(V, device = DEV)[None, :]]    # [nodes, V]
    masses = torch.zeros(probs.size(0), num_classes, dtype = torch.float64, device = DEV)
    return masses.index_add_(1, token_class, probs).log()


@pytest.mark.parametrize("name", ["one", "two", "skewed", "empty_class", "each_token"])
@pytest.mark.parametrize("kind", ["tied", "untied", "tied_out_of_order"])
def test_class_masses_match_brute_force(kind, name):
    pc = build(kind)
    token_class, C = labels(name)
    token_class = token_class.to(DEV)
    if name == "each_token":
        assert C > categorical.CLASS_CHUNK                                # several class chunks
    for layer in pc.input_layer_group:
        if kind == "tied_out_of_order":
            rows = layer.s_pids // V
            assert not torch.equal(rows, torch.arange(rows.numel(), device = DEV) % (rows.max() + 1))
        got = categorical.class_masses(layer, token_class, C)
        want = brute_force(layer, token_class, C)
        assert got.shape == (layer.num_nodes, C) and got.dtype == torch.float32
        assert torch.equal(torch.isneginf(got), torch.isneginf(want))
        assert torch.isneginf(want).any() == (name == "empty_class")
        fin = torch.isfinite(want)
        assert (got[fin].double() - want[fin]).abs().max() < 1e-5


def test_a_device_move_recomputes_the_rows():
    """The node rows kept on the layer follow the circuit when it moves (here to the CPU)."""
    pc = build("tied")
    layer = pc.input_layer_group[0]
    token_class, C = labels("skewed")
    want = categorical.class_masses(layer, token_class.to(DEV), C).cpu()
    pc.to(torch.device("cpu"))
    got = categorical.class_masses(layer, token_class, C)
    assert got.device.type == "cpu" and (got - want).abs().max() < 1e-5


def test_supported_distributions_are_matched_by_exact_type():
    class MyCategorical(dists.Categorical):
        pass

    assert distributions.lookup(dists.Categorical(num_cats = 5)) is categorical
    assert distributions.lookup(MyCategorical(num_cats = 5)) is None
    assert distributions.lookup(dists.Gaussian(mu = 0.0, sigma = 1.0)) is None
    assert categorical.num_values(dists.Categorical(num_cats = 7)) == 7


def test_a_layer_of_several_widths_is_refused():
    """Categorical nodes with different num_cats share an input layer (the signature ignores num_cats), whose
    parameters are then not one [rows, num_cats] table."""
    x = [inputs(v, num_node_blocks = 1, block_size = 2, dist = dists.Categorical(num_cats = 3 + v)) for v in range(2)]
    ns = summate(multiply(*x), num_node_blocks = 1, block_size = 1)
    pc = juice.compile(ns, verbose = False).to(DEV)
    layer, = pc.input_layer_group
    with pytest.raises(NotImplementedError, match = r"num_cats \[3, 4\]"):
        categorical.class_masses(layer, torch.zeros(4, dtype = torch.long, device = DEV), 1)
