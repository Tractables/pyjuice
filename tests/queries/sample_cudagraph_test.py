"""
`sample(use_cudagraph = True)`.

Replaying the top-down pass from a CUDA graph removes the per-launch host cost, which is what the
pass is bound by. It is OPT-IN because a graph owns a private memory pool and pins the frontier
buffers it was captured with for the circuit's lifetime -- worth it inside a sampling loop, not for a
one-off draw.

Two things make it delicate, and both are pinned here:

  * **a graph freezes scalar kernel arguments.** The sum kernels used to take a seed and call
    `tl.rand(seed, ...)`, so a captured pass would have redrawn the identical sample on every replay.
    Uniforms now come from a buffer refilled outside the graph;
  * **it is only correct where the index plan repeats**, i.e. on a structured-decomposable circuit,
    exactly as for the plan cache -- a captured pass replays one specific plan.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _sd_circuit():
    with juice.set_block_size(4):
        x = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5)) for v in range(4)]
        a = summate(multiply(x[0], x[1]), num_node_blocks = 2)
        b = summate(multiply(x[2], x[3]), num_node_blocks = 2)
        root = summate(multiply(a, b), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _non_sd_circuit():
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


@cuda_only
def test_no_graph_is_captured_by_default():
    """Opt-in: an ordinary draw must not leave a graph or a pinned frontier behind."""
    torch.manual_seed(0)
    pc = _sd_circuit()

    for _ in range(3):
        juice.queries.sample(pc, num_samples = 64)

    assert not pc.__dict__.get("_sample_plans", {}), "an ordinary draw recorded a pair-list plan"
    assert not pc.__dict__.get("_sample_scoped_states", {}), \
        "an ordinary draw pinned buffers or captured a graph"


@cuda_only
def test_replays_draw_fresh_samples():
    """
    REGRESSION for the trap this feature is built around: a CUDA graph bakes scalar kernel arguments
    in, so seeding the RNG inside the kernels would make every replay return the identical sample.
    """
    torch.manual_seed(0)
    pc = _sd_circuit()

    draws = [juice.queries.sample(pc, num_samples = 256, use_cudagraph = True).clone()
             for _ in range(4)]

    for i in range(1, len(draws)):
        assert not torch.equal(draws[0], draws[i]), \
            "consecutive graph replays returned identical samples -- the RNG is frozen in the graph"


@cuda_only
@pytest.mark.parametrize("conditional", [False, True])
def test_graphed_draws_match_ungraphed_ones(conditional):
    torch.manual_seed(0)
    pc = _sd_circuit()
    N = 40_000

    def draw(use_graph):
        if conditional:
            x = torch.randint(0, 5, [N, pc.num_vars], device = pc.device)
            missing = torch.zeros([pc.num_vars], dtype = torch.bool, device = pc.device)
            missing[::2] = True
            pc(x, missing_mask = missing)
            return juice.queries.sample(pc, conditional = True, use_cudagraph = use_graph).float()
        return juice.queries.sample(pc, num_samples = N, use_cudagraph = use_graph).float()

    torch.manual_seed(1)
    draw(True)                                          # capture
    graphed = draw(True)                                # replay
    torch.manual_seed(1)
    plain = draw(False)

    se = ((graphed.var(dim = 0) + plain.var(dim = 0)) / N).sqrt().clamp(min = 1e-9)
    z = float(((graphed.mean(dim = 0) - plain.mean(dim = 0)) / se).abs().max())
    assert z < 5.0, f"graphed and ungraphed draws differ: max |z| = {z:.2f}"


@cuda_only
def test_a_non_structured_decomposable_circuit_can_be_captured_too():
    """
    The default pass derives its frontier layout from the circuit, so its shapes are static for ANY
    circuit and capture no longer needs structured decomposability.

    That requirement belonged to the PAIR-LIST pass, which replays a recorded index plan and is
    therefore only correct where the plan repeats -- it is still enforced for that path. Getting this
    backwards is a wrong answer rather than a crash, so both directions are pinned: here, and in
    `test_the_pair_list_pass_still_requires_it` below.
    """
    torch.manual_seed(0)
    pc = _non_sd_circuit()
    assert not pc.is_structured_decomposable

    N = 40_000
    juice.queries.sample(pc, num_samples = N, use_cudagraph = True)         # capture
    graphed = juice.queries.sample(pc, num_samples = N, use_cudagraph = True).float()
    plain = juice.queries.sample(pc, num_samples = N).float()

    se = ((graphed.var(dim = 0) + plain.var(dim = 0)) / N).sqrt().clamp(min = 1e-9)
    z = float(((graphed.mean(dim = 0) - plain.mean(dim = 0)) / se).abs().max())
    assert z < 5.0, f"captured draws on a non-SD circuit differ: max |z| = {z:.2f}"


@cuda_only
def test_the_pair_list_pass_still_requires_it():
    """The legacy pass replays one specific plan, so capturing it where the plan varies is wrong."""
    torch.manual_seed(0)
    pc = _non_sd_circuit()

    with pytest.raises(AssertionError, match = "structured-decomposable"):
        juice.queries.sample(pc, num_samples = 64, use_cudagraph = True, _use_scope_plan = False)


@cuda_only
def test_a_graph_is_reused_rather_than_recaptured():
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 64, use_cudagraph = True)
    graph = pc.__dict__["_sample_scoped_states"][(64, False)]["graph"]
    assert graph is not None

    juice.queries.sample(pc, num_samples = 64, use_cudagraph = True)
    assert pc.__dict__["_sample_scoped_states"][(64, False)]["graph"] is graph


@cuda_only
def test_a_replayed_pass_still_reaches_every_variable():
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 32, use_cudagraph = True, _sample_input_ns = False)
    frontier = juice.queries.sample(pc, num_samples = 32, use_cudagraph = True,
                                    _sample_input_ns = False)

    selected = (frontier != -1).sum(dim = 0)
    assert bool((selected == pc.num_vars).all())
