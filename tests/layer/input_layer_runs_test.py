"""
An input layer's forward over some runs of its nodes only (`InputLayer.forward(..., node_runs = ...)`): every node a run
covers -- rounded up to a power of two of nodes within the layer -- gets the bits the whole layer's launch gives it,
and every other row of `node_mars` is left as it was. Also the queries' soft-data forwards, which take the same runs,
and the refusal of a layer whose post-processing kernels cover all of its nodes.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.queries.distributions as qd
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.queries.distributions.common import runs as rounded_runs


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEV = torch.device("cuda:0")
V, B, C = 12, 8, 5

DISTS = {
    "categorical": (lambda: dists.Categorical(num_cats = C), lambda g: torch.randint(0, C, (B, V), generator = g)),
    "discrete_logistic": (lambda: dists.DiscreteLogistic(val_range = (0.0, 1.0), num_cats = C),
                          lambda g: torch.randint(0, C, (B, V), generator = g)),
    "gaussian": (lambda: dists.Gaussian(mu = 0.0, sigma = 1.0), lambda g: torch.randn(B, V, generator = g)),
    "bernoulli": (lambda: dists.Bernoulli(), lambda g: torch.randint(0, 2, (B, V), generator = g)),
}

# Layer-local node ranges of a layer of 12 groups x 12 nodes: a group, two groups apart, the layer's last nodes (rounded
# up past its end, so moved left), a few in the middle, the last three, one node
RUNS = [[(0, 12)], [(12, 24), (60, 72)], [(130, 144)], [(100, 105)], [(141, 144)], [(5, 6)]]


def _circuit(dist_fn):
    torch.manual_seed(0)
    with juice.set_block_size(4):
        root = summate(multiply(*[inputs(v, num_node_blocks = 3, dist = dist_fn()) for v in range(V)]),
                       num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(DEV)


def _missing(kind, g):
    if kind is None:
        return None
    return (torch.rand(V, generator = g) < 0.3 if kind == "1d" else torch.rand(B, V, generator = g) < 0.3).to(DEV)


def _covered(layer, runs, num_rows):
    """Every row of `node_mars` the rounded `runs` write."""
    sid, eid = layer._output_ind_range
    rows = torch.zeros([num_rows], dtype = torch.bool, device = DEV)
    for start, n in rounded_runs(runs, eid - sid):
        rows[sid + start:sid + start + n] = True
    return rows


@cuda_only
@pytest.mark.parametrize("runs", RUNS)
@pytest.mark.parametrize("missing", [None, "1d", "2d"])
@pytest.mark.parametrize("kind", DISTS)
def test_runs_write_the_whole_layers_bits(kind, missing, runs):
    dist_fn, data_fn = DISTS[kind]
    pc = _circuit(dist_fn)
    layer = pc.input_layer_group[0]
    assert layer._output_ind_range[1] - layer._output_ind_range[0] == V * 12
    g = torch.Generator().manual_seed(1)
    data = data_fn(g).to(DEV).permute(1, 0).contiguous()
    miss = _missing(missing, g)

    want = torch.full([pc.num_nodes, B], float("nan"), device = DEV)
    layer.forward(data, want, missing_mask = miss)
    got = torch.full_like(want, float("nan"))
    layer.forward(data, got, missing_mask = miss, node_runs = runs)

    rows = _covered(layer, runs, pc.num_nodes)
    assert torch.equal(got[rows], want[rows])
    assert torch.isnan(got[~rows]).all()                             # nothing else written


@cuda_only
@pytest.mark.parametrize("runs", RUNS)
@pytest.mark.parametrize("kind", ["categorical", "discrete_logistic"])
def test_soft_data_runs_write_the_whole_layers_bits(kind, runs):
    """The queries' soft-evidence forwards ([B, V, num_cats] data), whose kernels are their own."""
    pc = _circuit(DISTS[kind][0])
    layer = pc.input_layer_group[0]
    g = torch.Generator().manual_seed(2)
    soft = torch.rand(B, V, C, generator = g).to(DEV)
    soft /= soft.sum(dim = 2, keepdim = True)
    miss = _missing("2d", g)

    want = torch.full([pc.num_nodes, B], float("nan"), device = DEV)
    qd.forward(layer, soft.clone(), want, missing_mask = miss)
    got = torch.full_like(want, float("nan"))
    qd.forward(layer, soft.clone(), got, missing_mask = miss, node_runs = runs)

    rows = _covered(layer, runs, pc.num_nodes)
    assert torch.equal(got[rows], want[rows])
    assert torch.isnan(got[~rows]).all()


@cuda_only
def test_a_layer_with_post_processing_kernels_runs_whole():
    pc = _circuit(lambda: dists.SoftEvidenceCategorical(num_cats = C))
    layer = pc.input_layer_group[0]
    assert not qd.takes_node_runs(layer)
    data = torch.zeros(V, B, dtype = torch.long, device = DEV)
    with pytest.raises(ValueError, match = "whole layer"):
        layer.forward(data, torch.zeros(pc.num_nodes, B, device = DEV), node_runs = [(0, 4)],
                      categorical_evidence_logp = torch.zeros(B, V, C, device = DEV))
