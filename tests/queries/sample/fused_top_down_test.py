"""
`sample(fuse_top_down = True)`: the whole conditional top-down as one kernel.

The fused pass replicates the unfused RNG scheme exactly -- the same per-(row, sample) offset -- so
under one seed the two draw the SAME frontier, and these tests can assert equality rather than
settle for a distributional argument. They differ only where the inverse-CDF boundary falls within
float error of the drawn uniform (the unfused walk accumulates the CDF in tiles, the fused one in a
single cumsum), which is rare enough to state as a rate: MEASURED at 0.017% of entries, on 2 of 200
seeds, and both passes are individually deterministic.

The batch > 1 tests are the load-bearing ones. The level-to-level handover goes through global
memory, and the threads that write a level's child rows are not the threads that read them next --
without an explicit barrier the pass is silently wrong in a way that HIDES AT BATCH 1: measured, two
of four columns came back entirely `-1` while column 2 was still perfectly correct.
"""

import random

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.queries.sample import _scope_plan, _fused_plan
from pyjuice.queries.sampling.fused import fusion_applicability


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_CATS = 5


def _hmm(num_vars = 6, states = 32):
    """A deep narrow chain -- the shape the fused pass exists for."""
    with juice.set_block_size(states):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        for v in range(1, num_vars):
            ns = summate(multiply(ns, inputs(v, num_node_blocks = 1,
                                             dist = dists.Categorical(num_cats = NUM_CATS))),
                         num_node_blocks = 1)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _draw(pc, data, fuse, seed, graph = False, **kw):
    random.seed(seed)
    torch.manual_seed(seed)
    pc(data, **kw)
    return juice.queries.sample(pc, conditional = True, _sample_input_ns = False,
                                use_cudagraph = graph, fuse_top_down = fuse, **kw).clone()


def _ready(batch, num_vars = 6, graph = False):
    """
    A circuit, its data, and BOTH passes already warmed.

    The first sampler call on a circuit allocates its per-shape state, and does not reproduce against
    later calls even under a fixed seed -- measured here as `first != second == third`. Comparing a
    cold draw against a warm one is a property of that warm-up, not of either pass, and it will fail
    every one of these tests for the wrong reason.
    """
    pc = _hmm(num_vars)
    data = torch.randint(0, NUM_CATS, [batch, pc.num_vars], device = torch.device("cuda:0"))
    for fuse in (False, True):
        _draw(pc, data, fuse, 0, graph = graph)
    return pc, data


@cuda_only
def test_the_gate_accepts_a_deep_chain_and_says_why_when_it_declines():
    pc = _hmm()
    ok, why = fusion_applicability(pc, _scope_plan(pc))
    assert ok and why is None
    assert _fused_plan(pc, _scope_plan(pc)) is not None

    # A circuit whose layer groups do not alternate product/sum is outside what the fused walk
    # assumes, and it has to decline with a reason rather than produce a wrong draw.
    with juice.set_block_size(4):
        i = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
             for v in range(4)]
        left = summate(multiply(i[0], i[1]), num_node_blocks = 2)
        right = summate(multiply(i[2], i[3]), num_node_blocks = 2)
        root = summate(multiply(left, right), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    other = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    ok2, why2 = fusion_applicability(other, _scope_plan(other))
    if not ok2:
        assert isinstance(why2, str) and why2


@cuda_only
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 8, 16])
def test_fused_matches_the_unfused_draw(batch):
    """The whole point: same seed, same frontier -- at every batch size, not just batch 1."""
    pc, data = _ready(batch)
    for seed in range(6):
        a = _draw(pc, data, False, seed)
        b = _draw(pc, data, True, seed)
        assert torch.equal(a, b), \
            f"batch {batch}, seed {seed}: {int((a != b).sum())} of {a.numel()} entries differ"


@cuda_only
@pytest.mark.parametrize("batch", [1, 4])
def test_every_column_is_a_complete_path(batch):
    """
    The barrier regression, stated as an invariant rather than a comparison.

    A dropped handover shows up as a column with no live entries at all, which is exactly what the
    unsynchronised version produced for 2 of 4 columns while the others looked perfect.
    """
    pc, data = _ready(batch)
    frontier = _draw(pc, data, True, 0)
    live = (frontier >= 0).sum(dim = 0)
    assert int(live.min()) > 0, f"a column came back empty: {live.tolist()}"
    assert len(set(live.tolist())) == 1, \
        f"columns disagree on how many nodes are live: {live.tolist()}"
    assert torch.equal(live, (_draw(pc, data, False, 0) >= 0).sum(dim = 0))


@cuda_only
def test_the_fused_pass_is_deterministic():
    """A residual race would show as two draws at one seed disagreeing."""
    pc, data = _ready(8)
    first = _draw(pc, data, True, 3)
    for _ in range(5):
        assert torch.equal(first, _draw(pc, data, True, 3))


@cuda_only
def test_fused_and_unfused_agree_under_a_cuda_graph():
    """The two passes are different kernel sequences captured into the same state slot, so the
    capture signature has to tell them apart -- otherwise flipping the flag replays the other one."""
    pc, data = _ready(4, graph = True)
    for seed in range(4):
        assert torch.equal(_draw(pc, data, True, seed, graph = True),
                           _draw(pc, data, False, seed, graph = True))


@cuda_only
def test_a_declined_circuit_still_draws():
    """When the gate says no, the flag is a no-op and the ordinary pass runs."""
    pc, data = _ready(4)
    pc.__dict__["_sample_fused_plan"] = False          # force the decline
    assert torch.equal(_draw(pc, data, True, 1), _draw(pc, data, False, 1))


@cuda_only
@pytest.mark.parametrize("num_node_blocks", [3, 5, 6])
def test_a_non_power_of_two_node_block_count_still_works(num_node_blocks):
    """
    Every extent the fused kernel turns into a `tl.arange` must be a power of two -- Triton requires
    it. The plan's extents come from the circuit's own table widths, which are under no such
    obligation: a circuit with 3 node blocks is entirely ordinary, passed the gate, and then failed
    to COMPILE. `scoped.py` rounds each of its own extents up for exactly this reason.
    """
    with juice.set_block_size(4):
        ni = [inputs(v, num_node_blocks = num_node_blocks,
                     dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        ns = summate(multiply(*ni), num_node_blocks = num_node_blocks)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))

    data = torch.randint(0, NUM_CATS, [4, pc.num_vars], device = torch.device("cuda:0"))
    for fuse in (False, True):
        _draw(pc, data, fuse, 0)                      # warm both passes
    for seed in range(4):
        a = _draw(pc, data, False, seed)
        b = _draw(pc, data, True, seed)
        assert torch.equal(a, b), \
            f"{num_node_blocks} node blocks: {int((a != b).sum())} of {a.numel()} entries differ"
