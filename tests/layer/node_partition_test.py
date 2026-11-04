"""
`partition_nodes_by_n_edges`: the block sizes a layer's nodes are padded to. The dynamic program finds the sizes with
the least padding, and they are read back through its table: its column must step down one a partition. Reading one
column throughout gave other sizes from 3 partitions on: MEASURED on random degree lists, 1615 of 2228 past the
requested tolerance, mean padding 18.3% against 11.6%. Padding changes speed and memory, never a value: the circuit test
checks it.
"""

import itertools

import numpy as np
import pytest
import torch

import pyjuice as juice
from pyjuice.layer.backend.node_partition import partition_nodes_by_n_edges
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes.distributions import Categorical


def _cost(degrees, sizes):
    """Every node padded to the smallest size that holds it."""
    sizes = np.sort(np.asarray(sizes))
    return int(sizes[np.searchsorted(sizes, degrees)].sum())


def _degrees(rng, unique = 10, high = 64):
    values = rng.choice(np.arange(1, high + 1), size = unique, replace = False)
    return rng.choice(values, size = int(rng.integers(unique, 4 * unique)), replace = True)


@pytest.mark.parametrize("algorithm", ["dp_with_coalesce", "dp_simple"])
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_the_sizes_have_the_least_padding_of_any_k_sizes(algorithm, k):
    rng = np.random.default_rng(k)
    for _ in range(60):
        degrees = _degrees(rng)
        sizes = partition_nodes_by_n_edges(torch.from_numpy(degrees), max_num_partitions = k, algorithm = algorithm)
        unique = np.unique(degrees)
        best = min(_cost(degrees, list(rest) + [unique[-1]]) for n in range(k)
                   for rest in itertools.combinations(unique[:-1], n))
        assert _cost(degrees, sizes.numpy()) == best, (degrees.tolist(), sizes.tolist())


@pytest.mark.parametrize("tolerance", [0.01, 0.05, 0.25, 0.5])
def test_the_padding_stays_within_the_tolerance(tolerance):
    """At most 16 distinct degrees: 16 sizes pad nothing, so the tolerance can always be met."""
    rng = np.random.default_rng(int(tolerance * 100))
    for _ in range(100):
        degrees = _degrees(rng, unique = int(rng.integers(3, 17)), high = 512)
        sizes = partition_nodes_by_n_edges(torch.from_numpy(degrees), sparsity_tolerance = tolerance)
        assert _cost(degrees, sizes.numpy()) <= degrees.sum() * (1 + tolerance) + 1e-9, degrees.tolist()
        assert int(sizes.max()) == int(degrees.max()), "a size must hold the widest node"
        assert torch.equal(sizes, torch.sort(sizes).values)


def _irregular(tolerance):
    """A sum layer whose nodes take 1 to 64 children: several block sizes, at a tight tolerance."""
    torch.manual_seed(0)
    degrees = [1, 2, 3, 5, 8, 13, 21, 34, 55, 64]
    with juice.set_block_size(1):
        a = inputs(0, num_node_blocks = 64, dist = Categorical(num_cats = 8))
        b = inputs(1, num_node_blocks = 64, dist = Categorical(num_cats = 8))
        products = multiply(a, b, edge_ids = torch.arange(64)[:, None].repeat(1, 2))
        pars = torch.cat([torch.full((d,), i) for i, d in enumerate(degrees)])
        chs = torch.cat([torch.arange(d) for d in degrees])
        ns = summate(products, num_node_blocks = len(degrees), edge_ids = torch.stack((pars, chs)))
        root = summate(multiply(ns), num_node_blocks = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False, layer_sparsity_tol = tolerance).to(torch.device("cuda:0"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_a_circuit_computes_the_same_with_any_block_sizes():
    tight, one = _irregular(0.01), _irregular(None)
    sum_layers = [l for g in tight.inner_layer_groups for l in g.layers if isinstance(l, juice.SumLayer)]
    assert max(l.num_fw_partitions for l in sum_layers) >= 3, "the case the read-back gets wrong"
    x = torch.randint(0, 8, (32, 2), device = torch.device("cuda:0"))
    lls = []
    for pc in (tight, one):
        lls.append(pc(x).detach().clone())
        pc.backward(x, flows_memory = 0.0)
    assert torch.allclose(lls[0], lls[1], atol = 1e-5), float((lls[0] - lls[1]).abs().max())
    assert torch.allclose(tight.param_flows.sum(), one.param_flows.sum(), rtol = 1e-5)
