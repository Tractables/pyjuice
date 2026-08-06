"""
`sample()` across PC structures and batch sizes.

The top-down sampler picks its Triton tile sizes from the shape of each layer and from how many nodes
the frontier holds, so its launch configuration varies with BOTH the circuit and `num_samples`. Some
of those configurations do not compile at all, and the failure is a hard `PassManager::run failed`
rather than a wrong answer -- which means a structure either works or crashes outright, and nothing
in between warns you. Two were live:

  * a tile whose node-block axis is 1 (a single node block per layer) with `BLOCK_S == 32`
    (`num_samples >= 4096`) -- every HMM-shaped chain, at large batch;
  * a tile whose sample axis is 1 (`num_samples < 256`) -- `PD`-structured circuits at small batch,
    including every conditional draw on them, since those run at the evidence's batch size.

Both are gone (all tile dimensions are floored at 2), and this file is what keeps them gone. It is
deliberately a COVERAGE test rather than a correctness one: the distributions are checked in
`sample_test.py` and `blockscale_sample_test.py`, while what is easy to break here is a launch
configuration nobody happened to instantiate.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

#: 1 and 16 sit below the `num_samples // 128` threshold where the sample tile collapses to one lane;
#: 4096 is where it reaches 32. Both ends have broken before.
BATCHES = [1, 16, 512, 4096]


def _hmm(num_states = 64, seq_len = 6):
    with juice.set_block_size(num_states):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = 8))
        for t in range(1, seq_len):
            emit = inputs(t, num_node_blocks = 1, dist = dists.Categorical(num_cats = 8))
            ns = summate(multiply(ns, emit), num_node_blocks = 1)
        return summate(multiply(ns), num_node_blocks = 1, block_size = 1)


def _hclt(num_vars = 16, num_latents = 8):
    data = torch.randint(0, 16, [512, num_vars]).float().to(torch.device("cuda:0"))
    return juice.structures.HCLT(data, num_bins = 8, sigma = 0.5 / 8,
                                 num_latents = num_latents, chunk_size = 8)


def _pd(num_vars = 64):
    """
    Multi-PARTITION layers, and the structure the small-batch crash was found on.

    Sized to REPRODUCE it: at 16 variables / 8 latents the layer shapes never instantiate the tile
    that failed, and this file passed with the floors removed. Verified by removing them again.
    """
    return juice.structures.PD(data_shape = (num_vars,), num_latents = 32, split_intervals = (8,))


def _rat(num_vars = 16):
    return juice.structures.RAT_SPN(num_vars = num_vars, num_latents = 8, depth = 2,
                                    num_repetitions = 2, num_pieces = 2,
                                    input_dist = dists.Categorical(num_cats = 8))


def _ragged():
    """Hand-built and deliberately irregular: unequal children per node block, ragged edge sets."""
    with juice.set_block_size(4):
        i = [inputs(v, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        m0 = multiply(i[0], i[1], edge_ids = torch.tensor([[0, 0], [1, 2], [2, 1]], dtype = torch.long))
        m1 = multiply(i[2], i[3])
        s0 = summate(m0, edge_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]], dtype = torch.long))
        s1 = summate(m1, num_node_blocks = 2)
        m2 = multiply(s0, s1, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
        return summate(m2, num_node_blocks = 1, block_size = 1)


STRUCTURES = {"hmm": _hmm, "hclt": _hclt, "pd": _pd, "rat_spn": _rat, "ragged": _ragged}


@pytest.fixture(scope = "module")
def circuits():
    """Compiled once per structure -- building an HCLT / PD / RAT is the slow part of this file."""
    out = {}
    for name, build in STRUCTURES.items():
        torch.manual_seed(0)
        ns = build()
        ns.init_parameters(perturbation = 2.0)
        out[name] = juice.compile(ns, verbose = False).to(torch.device("cuda:0"))
    return out


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
@pytest.mark.parametrize("num_samples", BATCHES)
def test_unconditional_sampling_runs(circuits, name, num_samples):
    pc = circuits[name]
    samples = juice.queries.sample(pc, num_samples = num_samples)

    assert samples.shape == (num_samples, pc.num_vars)
    assert bool((samples >= 0).all()), "a draw fell outside the input distributions' support"


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
@pytest.mark.parametrize("batch_size", [1, 16, 512])
def test_conditional_sampling_runs(circuits, name, batch_size):
    """Conditional draws inherit the FORWARD's batch, so a small evidence batch forces the
    one-lane tile that used to fail to compile."""
    pc = circuits[name]
    torch.manual_seed(1)

    x = torch.randint(0, 5, [batch_size, pc.num_vars], device = pc.device)
    missing = torch.zeros([pc.num_vars], dtype = torch.bool, device = pc.device)
    missing[::2] = True

    pc(x, missing_mask = missing)
    samples = juice.queries.sample(pc, conditional = True)

    assert samples.shape == (batch_size, pc.num_vars)
    assert bool((samples >= 0).all())


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_the_frontier_ends_with_one_input_node_per_variable(circuits, name):
    """A structural check the shape assertions cannot make: every variable must be reached exactly
    once. A mis-sized tile that silently dropped a lane would leave a variable unassigned."""
    pc = circuits[name]
    frontier = juice.queries.sample(pc, num_samples = 64, _sample_input_ns = False)

    # `_sample_input_ns = False` stops before the input nodes emit and returns the frontier of
    # selected INPUT node ids, compacted to the front of each column. A complete pass leaves exactly
    # one per variable; a dropped lane leaves fewer and a double-expanded product node leaves more.
    selected = (frontier != -1).sum(dim = 0)

    assert bool((selected == pc.num_vars).all()), \
        f"selected {selected.min().item()}..{selected.max().item()} input nodes per sample, " \
        f"expected exactly {pc.num_vars} (one per variable)"
