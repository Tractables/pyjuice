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
    graph = pc.__dict__["_sample_scoped_states"][(None, False)]["graph"]
    assert graph is not None

    juice.queries.sample(pc, num_samples = 64, use_cudagraph = True)
    assert pc.__dict__["_sample_scoped_states"][(None, False)]["graph"] is graph


@cuda_only
def test_a_replayed_pass_still_reaches_every_variable():
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 32, use_cudagraph = True, _sample_input_ns = False)
    frontier = juice.queries.sample(pc, num_samples = 32, use_cudagraph = True,
                                    _sample_input_ns = False)

    selected = (frontier != -1).sum(dim = 0)
    assert bool((selected == pc.num_vars).all())


# ------------------------------------------------------------------ what a capture bakes in

def _gated_circuit():
    from pyjuice.nodes import BlockScaleSumParams

    with juice.set_block_size(2):
        i = [inputs(v, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5)) for v in range(2)]
        gated = summate(multiply(i[0], i[1]), num_node_blocks = 2,
                        external_params = BlockScaleSumParams(ch_block_size = 1))
        root = summate(multiply(gated), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)

    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    num_node_gates = gated.num_nodes // gated.external_params.gate_sizes(gated)[0]
    num_ch_gates = gated.num_ch_nodes // gated.external_params.gate_sizes(gated)[1]
    return pc, gated, (num_node_gates, num_ch_gates)


def _max_z(a, b, num_samples):
    se = ((a.float().var(dim = 0) + b.float().var(dim = 0)) / num_samples).sqrt().clamp(min = 1e-9)
    return float(((a.float().mean(dim = 0) - b.float().mean(dim = 0)) / se).abs().max())


@cuda_only
def test_a_graph_is_recaptured_when_the_circuits_buffers_move():
    """
    REGRESSION. A captured pass holds the addresses of `node_mars` / `element_mars`, and those are
    `_init_buffer`-backed: a forward at another batch size REALLOCATES them, so the replay would go
    on reading memory the allocator has since handed to somebody else.

    Interleaving two batch sizes on one circuit is ordinary in a decode loop, which is how this is
    reached. It fails SILENTLY -- how wrong the draw comes out depends on what landed in the recycled
    pages, so it can pass on one run and not the next; measured at 10.5 sigma before the fix.
    """
    torch.manual_seed(0)
    pc = _sd_circuit()
    N = 4096

    def condition(evidence, batch = N):
        x = torch.zeros([batch, pc.num_vars], dtype = torch.long, device = pc.device)
        x[:, 0] = evidence
        missing = torch.ones([pc.num_vars], dtype = torch.bool, device = pc.device)
        missing[0] = False
        pc(x, missing_mask = missing)

    condition(0)
    torch.manual_seed(1)
    juice.queries.sample(pc, conditional = True, use_cudagraph = True)          # capture

    # HOLD the captured buffer alive. Without this the caching allocator is free to hand the same
    # block back when `node_mars` is reallocated below, and then the stale pointer still happens to
    # be valid and the test silently proves nothing -- which is exactly how it failed under `-n 8`,
    # where memory pressure changed which block came back.
    keepalive = pc.node_mars
    before = keepalive.data_ptr()

    # a forward at another batch size, then back -- what a decode loop does between draws
    pc(torch.zeros([777, pc.num_vars], dtype = torch.long, device = pc.device),
       missing_mask = torch.ones([pc.num_vars], dtype = torch.bool, device = pc.device))
    condition(3)
    assert pc.node_mars.data_ptr() != before, \
        "the allocator reused the same address despite the keepalive -- this test cannot see the " \
        "defect it exists for"

    torch.manual_seed(2)
    graphed = juice.queries.sample(pc, conditional = True, use_cudagraph = True)
    condition(3)
    torch.manual_seed(2)
    live = juice.queries.sample(pc, conditional = True)

    z = _max_z(graphed, live, N)
    assert z < 5.0, f"the replay read stale buffers: max |z| = {z:.2f}"
    del keepalive


@cuda_only
def test_a_graph_captured_without_gates_is_not_replayed_for_a_gated_draw():
    """
    A pass captured with no external parameters has no gated kernels in it at all, so replaying it
    for a gated draw drops the gate ENTIRELY rather than reading it from the wrong place -- the draw
    comes back looking like a perfectly ordinary ungated one.
    """
    torch.manual_seed(0)
    pc, gated_ns, (num_node_gates, num_ch_gates) = _gated_circuit()
    N = 4096

    torch.manual_seed(3)
    gate = (torch.randn([1, num_node_gates, num_ch_gates], device = pc.device) * 3.0)
    gate = gate.expand(N, -1, -1).contiguous()

    torch.manual_seed(1)
    juice.queries.sample(pc, num_samples = N, use_cudagraph = True)             # capture UNGATED

    torch.manual_seed(2)
    replayed = juice.queries.sample(pc, num_samples = N, use_cudagraph = True,
                                    sum_external_params = {gated_ns: gate})
    torch.manual_seed(2)
    live = juice.queries.sample(pc, num_samples = N, sum_external_params = {gated_ns: gate})
    torch.manual_seed(2)
    ungated = juice.queries.sample(pc, num_samples = N)

    assert _max_z(replayed, ungated, N) > 5.0, \
        "the gated replay is indistinguishable from an ungated draw -- the gate was dropped"
    z = _max_z(replayed, live, N)
    assert z < 5.0, f"the gated replay differs from a live gated draw: max |z| = {z:.2f}"


@cuda_only
def test_a_graph_survives_repeated_draws_at_one_batch_size():
    """The other side of the recapture check: nothing moved, so the graph must be REUSED. Recapturing
    on every call would be correct and would silently cost the whole point of the feature."""
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 64, use_cudagraph = True)
    graph = pc.__dict__["_sample_scoped_states"][(None, False)]["graph"]

    for _ in range(4):
        juice.queries.sample(pc, num_samples = 64, use_cudagraph = True)
        assert pc.__dict__["_sample_scoped_states"][(None, False)]["graph"] is graph


# ------------------------------------------------------ one graph across a shrinking batch size

def _unconditional_state(pc):
    return pc.__dict__.get("_sample_scoped_states", {}).get((None, False))


@cuda_only
def test_a_shrinking_batch_reuses_one_graph():
    """
    A decode loop's batch falls as sequences finish. Capturing a graph per size costs three live
    passes each and then thrashes an 8-entry cache, which is slower than not using a graph at all --
    so a smaller request replays the wider graph and reads the first `num_samples` columns.
    """
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 4096, use_cudagraph = True)
    graph = _unconditional_state(pc)["graph"]
    assert graph is not None

    for n in (2048, 1024, 512, 64, 3):
        samples = juice.queries.sample(pc, num_samples = n, use_cudagraph = True)
        assert samples.size(0) == n, f"asked for {n} samples, got {samples.size(0)}"
        assert _unconditional_state(pc)["graph"] is graph, f"recaptured for batch {n}"

    assert len(pc.__dict__["_sample_scoped_states"]) == 1, \
        "a shrinking batch should not accumulate one state per size"


@cuda_only
def test_a_growing_batch_recaptures():
    """The other direction has to reallocate: the buffers a wider draw needs do not exist yet, and
    the old graph holds pointers into the ones being replaced."""
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 256, use_cudagraph = True)
    graph = _unconditional_state(pc)["graph"]

    samples = juice.queries.sample(pc, num_samples = 1024, use_cudagraph = True)
    assert samples.size(0) == 1024
    assert _unconditional_state(pc)["graph"] is not graph, "a wider draw must not replay the old graph"


@cuda_only
def test_a_reused_graph_still_draws_the_right_distribution():
    """
    The correctness half. Replaying a wider graph computes lanes this draw does not want; if the
    slicing were off by so much as a column the answer would be another sample's.
    """
    torch.manual_seed(0)
    pc = _sd_circuit()
    N = 40_000

    juice.queries.sample(pc, num_samples = 200_000, use_cudagraph = True)    # capture wide

    torch.manual_seed(1)
    narrow = juice.queries.sample(pc, num_samples = N, use_cudagraph = True).float()
    torch.manual_seed(1)
    plain = juice.queries.sample(pc, num_samples = N).float()

    se = ((narrow.var(dim = 0) + plain.var(dim = 0)) / N).sqrt().clamp(min = 1e-9)
    z = float(((narrow.mean(dim = 0) - plain.mean(dim = 0)) / se).abs().max())
    assert z < 5.0, f"a draw off the shared buffer differs from a plain one: max |z| = {z:.2f}"


@cuda_only
def test_a_reused_graph_still_reaches_every_variable():
    """`_sample_input_ns = False` returns the frontier itself, which is where an off-by-one column
    would show as a missing variable rather than as a shifted distribution."""
    torch.manual_seed(0)
    pc = _sd_circuit()

    juice.queries.sample(pc, num_samples = 2048, use_cudagraph = True, _sample_input_ns = False)
    frontier = juice.queries.sample(pc, num_samples = 512, use_cudagraph = True,
                                    _sample_input_ns = False)

    assert frontier.size(1) == 512
    assert bool(((frontier != -1).sum(dim = 0) == pc.num_vars).all())


@cuda_only
def test_conditional_graphs_stay_keyed_by_batch_size():
    """
    Conditional draws must NOT share a buffer: their kernels read `pc.node_mars` with the batch size
    frozen at capture, so a graph replayed against a forward pass run at another batch would read it
    with the wrong stride. Pinned as a wrong ANSWER, not just as a cache-key check.
    """
    torch.manual_seed(0)
    pc = _sd_circuit()

    def condition(batch):
        x = torch.randint(0, 5, [batch, pc.num_vars], device = pc.device)
        missing = torch.zeros([pc.num_vars], dtype = torch.bool, device = pc.device)
        missing[::2] = True
        pc(x, missing_mask = missing)

    condition(4096)
    juice.queries.sample(pc, conditional = True, use_cudagraph = True)
    condition(1024)
    graphed = juice.queries.sample(pc, conditional = True, use_cudagraph = True).float()
    condition(1024)
    plain = juice.queries.sample(pc, conditional = True).float()

    assert graphed.size(0) == 1024
    se = ((graphed.var(dim = 0) + plain.var(dim = 0)) / 1024).sqrt().clamp(min = 1e-9)
    z = float(((graphed.mean(dim = 0) - plain.mean(dim = 0)) / se).abs().max())
    assert z < 5.0, f"a conditional replay at another batch size is wrong: max |z| = {z:.2f}"

    keys = set(pc.__dict__["_sample_scoped_states"])
    assert (4096, True) in keys and (1024, True) in keys


# --------------------------------------------------- gates, graphs and a changing batch size

@cuda_only
def test_a_gated_draw_survives_a_shrinking_batch_under_a_graph():
    """
    REGRESSION. The shared workspace above is keyed `(None, False)` and stays at the widest batch
    seen, but the GATED kernels take the frontier's width as their sample count -- using it both to
    validate the staged tensors and to stride the gate table. A width-4096 replay of an 8-sample
    gated draw was therefore rejected outright ("staged ... for a batch of 8 ... needed 4096"), and
    had it not been, it would have read past the end of the gate buffer.

    Gated draws consequently keep one workspace per batch size; only ungated ones share. Both
    directions are exercised, because growing reallocates and shrinking is what a decode loop does.
    """
    pc, gated_ns, (num_node_gates, num_ch_gates) = _gated_circuit()

    def gate(n):
        torch.manual_seed(9)
        return (torch.randn([1, num_node_gates, num_ch_gates], device = pc.device) * 2.0
                ).expand(n, -1, -1).contiguous()

    for n in (4096, 512, 8, 4096, 16384):
        samples = juice.queries.sample(pc, num_samples = n, use_cudagraph = True,
                                       sum_external_params = {gated_ns: gate(n)})
        assert samples.size(0) == n, f"asked for {n} samples, got {samples.size(0)}"


@cuda_only
def test_a_returned_frontier_is_not_the_live_workspace():
    """
    REGRESSION. `_sample_input_ns = False` returns the frontier itself, and under `use_cudagraph`
    that buffer is pinned for the circuit's lifetime -- so the caller's result was being overwritten
    by their next draw. Two frontiers held at once must not share storage.

    :note: the first attempt at this fix used `.contiguous()`, which on an already-contiguous slice
           returns the SAME object and so changed nothing. It needs an explicit copy.
    """
    torch.manual_seed(0)
    pc = _sd_circuit()

    first = juice.queries.sample(pc, num_samples = 256, use_cudagraph = True, _sample_input_ns = False)
    snapshot = first.clone()
    second = juice.queries.sample(pc, num_samples = 256, use_cudagraph = True, _sample_input_ns = False)

    assert first.data_ptr() != second.data_ptr(), "two frontiers share one buffer"
    assert torch.equal(first, snapshot), "the first frontier was overwritten by the second draw"
