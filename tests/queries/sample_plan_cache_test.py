"""
The sampler's index-plan cache.

Most of a top-down pass is spent deriving WHERE things go rather than drawing them: per layer a
`torch.where`, a device-to-host copy, a serial slot allocation and a copy back. On a structured
decomposable circuit those indices are the same on every call -- one vtree means the frontier's shape
after each layer is a function of the scopes alone, never of which node a draw selected -- so they are
recorded once and replayed.

The correctness risk is entirely in the GATE. Replaying a plan on a circuit whose plan actually
varies would not be slightly stale, it would be wrong: the sampler would write children into slots
belonging to other nodes. So these tests pin (a) that the cache engages only when
`pc.is_structured_decomposable`, (b) that cached draws follow the same distribution as uncached ones,
and (c) that plans of different shapes never serve each other.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _sd_circuit():
    """One vtree: `{0,1}` and `{2,3}`, always split the same way -> structured decomposable."""
    with juice.set_block_size(4):
        x = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        a = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        b = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        root = summate(multiply(a, b), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _non_sd_circuit():
    """A sum node choosing between products of DIFFERENT ARITY over one scope: the number of nodes
    pushed onto the frontier depends on the draw, so the plan genuinely varies."""
    with juice.set_block_size(4):
        x = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        s01 = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        s23 = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        s2 = summate(multiply(x[2]), num_node_blocks = 2)
        s3 = summate(multiply(x[3]), num_node_blocks = 2)
        root = summate(multiply(s01, s23), multiply(s01, s2, s3),
                       num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _plans(pc):
    return pc.__dict__.get("_sample_plans", {})


@cuda_only
def test_the_cache_engages_on_a_structured_decomposable_circuit():
    torch.manual_seed(0)
    pc = _sd_circuit()
    assert pc.is_structured_decomposable

    juice.queries.sample(pc, num_samples = 64)
    assert (64, False) in _plans(pc), "the first pass should have recorded a plan"
    assert not _plans(pc)[(64, False)].replaying

    juice.queries.sample(pc, num_samples = 64)
    assert _plans(pc)[(64, False)].replaying, "the second pass should have replayed it"


@cuda_only
def test_the_cache_refuses_a_non_structured_decomposable_circuit():
    """
    The gate is the whole safety argument: replaying a varying plan writes children into slots that
    belong to other nodes.

    MEASURED rather than assumed -- forcing the cache on with `pc.is_structured_decomposable = True`
    shifts this circuit's per-variable marginals by 48 sigma. Note that this test catches a removed
    gate by observing that a plan was recorded at all, not by observing a wrong answer: the wrong
    answer is a distribution shift, which a shape assertion would sail past.
    """
    torch.manual_seed(0)
    pc = _non_sd_circuit()
    assert not pc.is_structured_decomposable

    for _ in range(3):
        juice.queries.sample(pc, num_samples = 64)

    assert not _plans(pc), "a circuit whose plan varies must not get a cached one"


@cuda_only
@pytest.mark.parametrize("conditional", [False, True])
def test_cached_draws_match_uncached_ones(conditional):
    """Same distribution, per variable, in units of the standard error."""
    torch.manual_seed(0)
    pc = _sd_circuit()
    N = 40_000

    def draw():
        if conditional:
            x = torch.randint(0, 5, [N, pc.num_vars], device = pc.device)
            missing = torch.zeros([pc.num_vars], dtype = torch.bool, device = pc.device)
            missing[::2] = True
            pc(x, missing_mask = missing)
            return juice.queries.sample(pc, conditional = True).float()
        return juice.queries.sample(pc, num_samples = N).float()

    torch.manual_seed(1)
    draw()                                                  # records the plan
    cached = draw()                                         # replays it

    pc.__dict__.pop("_sample_plans", None)
    pc.is_structured_decomposable = False                   # force the uncached path
    torch.manual_seed(1)
    fresh = draw()

    se = ((cached.var(dim = 0) + fresh.var(dim = 0)) / N).sqrt().clamp(min = 1e-9)
    z = float(((cached.mean(dim = 0) - fresh.mean(dim = 0)) / se).abs().max())
    assert z < 5.0, f"cached and uncached draws differ: max |z| = {z:.2f}"


@cuda_only
def test_plans_do_not_leak_across_shapes():
    """A plan is keyed by `(num_samples, conditional)`. Serving one shape's indices to another would
    index past the end of the frontier."""
    torch.manual_seed(0)
    pc = _sd_circuit()

    for n in (16, 64, 256):
        juice.queries.sample(pc, num_samples = n)
        juice.queries.sample(pc, num_samples = n)

    x = torch.randint(0, 5, [64, pc.num_vars], device = pc.device)
    missing = torch.ones([pc.num_vars], dtype = torch.bool, device = pc.device)
    pc(x, missing_mask = missing)
    juice.queries.sample(pc, conditional = True)

    keys = set(_plans(pc))
    assert {(16, False), (64, False), (256, False), (64, True)} <= keys
    # the conditional plan must be its own entry, not the unconditional one at the same batch
    assert _plans(pc)[(64, True)] is not _plans(pc)[(64, False)]


@cuda_only
def test_the_cache_is_bounded():
    """One plan per shape, held for the circuit's lifetime, would grow without limit for a caller
    sweeping batch sizes."""
    from pyjuice.queries.sample import _PLAN_CACHE_SIZE

    torch.manual_seed(0)
    pc = _sd_circuit()

    for n in range(1, _PLAN_CACHE_SIZE + 5):
        juice.queries.sample(pc, num_samples = n)

    assert len(_plans(pc)) <= _PLAN_CACHE_SIZE


@cuda_only
def test_a_replayed_pass_still_reaches_every_variable():
    """Structural check: a stale plan would leave variables unassigned or doubly expanded."""
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 32, _sample_input_ns = False)
    frontier = juice.queries.sample(pc, num_samples = 32, _sample_input_ns = False)

    selected = (frontier != -1).sum(dim = 0)
    assert bool((selected == pc.num_vars).all())
