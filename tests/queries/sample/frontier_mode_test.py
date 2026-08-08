"""
`sample(..., _sample_input_ns = False)` -- the FRONTIER mode.

An ordinary draw finishes by emitting a value from each selected input node and returns
`[num_samples, num_vars]` tokens. In frontier mode it stops one step earlier and hands back the
selected NODE IDS as `[rows, num_samples]`, with `-1` for "nothing here". A decoder that wants the
latent, not the token, uses this exclusively -- it reads the ids, maps them to variables through
`layer.vids`, and gathers emission parameters through `layer.s_pids`.

That makes four unwritten assumptions load-bearing, and this file writes them down:

  * the compaction leaves each column as a dense prefix of live ids followed by nothing but `-1`,
    so a `>= 0` filter is exactly "the live entries";
  * each sample resolves each variable EXACTLY once -- never twice, never zero times;
  * the live count is the same in every column of a draw, so a caller may size buffers by it;
  * every live id belongs to SOME input layer, and which one has to be resolved before subtracting
    a start index. Ids are GLOBAL. See `test_the_frontier_spans_every_input_layer`, which exists
    because the natural shortcut -- subtract `input_layer_group[0]`'s start and index its `vids` --
    reads out of bounds the moment a circuit has more than one input layer.

:note: the emission step is what is skipped, not the bookkeeping: `sample()` still fills
       `pc.node_flows` before the branch, so a frontier draw overwrites it. A caller holding flows
       from a backward gets them replaced. Pinned in `test_a_frontier_draw_does_not_disturb_a_backward`
       -- what matters is that a later `backward()` recomputes rather than trusting them.
"""

import random

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.queries.sampling.frontier import push_non_neg_ones_to_front


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_CATS = 5


def _hmm(num_vars = 6, states = 32):
    with juice.set_block_size(states):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        for v in range(1, num_vars):
            ns = summate(multiply(ns, inputs(v, num_node_blocks = 1,
                                             dist = dists.Categorical(num_cats = NUM_CATS))),
                         num_node_blocks = 1)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _mixed_distributions():
    """Categorical on some variables, Bernoulli on others -- which compiles to TWO input layers."""
    with juice.set_block_size(4):
        i = [inputs(v, num_node_blocks = 2,
                    dist = dists.Categorical(num_cats = NUM_CATS) if v % 2 == 0 else dists.Bernoulli())
             for v in range(4)]
        left = summate(multiply(i[0], i[1]), num_node_blocks = 2)
        right = summate(multiply(i[2], i[3]), num_node_blocks = 2)
        root = summate(multiply(left, right), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0"))


def _owning_layer(pc, node_id):
    for layer in pc.input_layer_group:
        start, end = layer._output_ind_range
        if start <= node_id < end:
            return layer, start
    return None, None


# ----------------------------------------------------------------- the compaction, on its own

@cuda_only
@pytest.mark.parametrize("case", ["mixed", "all_dead", "all_live", "one_at_top", "one_at_bottom",
                                  "only_trailing", "single_row", "wide"])
def test_the_compaction_matches_a_per_column_reference(case):
    """
    `push_non_neg_ones_to_front` against an independent per-column implementation.

    It is a scatter to `cumsum - 1` through a scratch row, which is efficient and not obviously
    correct; the degenerate columns are where a hand-written version of this usually goes wrong.
    """
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    matrices = {
        "mixed":        torch.tensor([[5, -1, 9], [-1, -1, 8], [7, 3, -1], [-1, 4, 6]]),
        "all_dead":     torch.full([4, 3], -1),
        "all_live":     torch.arange(12).reshape(4, 3),
        "one_at_top":   torch.tensor([[5, 6, 7], [-1, -1, -1], [-1, -1, -1]]),
        "one_at_bottom": torch.tensor([[-1, -1, -1], [-1, -1, -1], [5, 6, 7]]),
        "only_trailing": torch.tensor([[-1, -1], [-1, 9], [8, 7]]),
        "single_row":   torch.tensor([[3, -1, 5]]),
        "wide":         torch.where(torch.rand(7, 4095) < 0.5, torch.randint(0, 99, [7, 4095]),
                                    torch.full([7, 4095], -1)),
    }
    matrix = matrices[case].to(torch.long).to(device)
    reference = matrix.clone()

    counts, _ = push_non_neg_ones_to_front(matrix)

    for j in range(reference.size(1)):
        live = [int(v) for v in reference[:, j] if int(v) != -1]
        got = matrix[:, j]

        assert int(counts[j]) == len(live)
        assert [int(v) for v in got[:len(live)]] == live, \
            f"column {j}: order not preserved, {got[:len(live)].tolist()} vs {live}"
        assert bool((got[len(live):] == -1).all()), \
            f"column {j}: rows below the prefix are not -1 -- a `>= 0` filter would pick up stale ids"


# ------------------------------------------------------------------ the frontier's shape invariants

@cuda_only
@pytest.mark.parametrize("num_samples", [1, 2, 3, 4095, 4096])
def test_the_frontier_is_a_dense_prefix_of_exactly_one_node_per_variable(num_samples):
    """
    The three shape facts a caller relies on, at sizes that straddle the kernels' tiling.

    A partial compaction, an off-by-one in the slice, or a variable resolved twice would each leave
    a frontier that still looks plausible; each is caught here.
    """
    torch.manual_seed(0)
    pc = _hmm()

    frontier = juice.queries.sample(pc, num_samples = num_samples, _sample_input_ns = False)
    assert frontier.size(1) == num_samples

    live = (frontier != -1).sum(dim = 0)
    assert bool((live == pc.num_vars).all()), \
        f"live counts {sorted(set(live.tolist()))}, expected exactly {pc.num_vars} everywhere"

    rows = torch.arange(frontier.size(0), device = frontier.device)[:, None]
    assert bool((frontier[rows < live[None, :]] != -1).all()), "a -1 inside the live prefix"
    assert bool((frontier[rows >= live[None, :]] == -1).all()), "a non -1 below the live prefix"


@cuda_only
def test_every_variable_is_resolved_exactly_once():
    """
    Never twice. A scope reachable by more than one route -- a mixture over the same variables, or a
    circuit that is not structured decomposable -- is where a double-resolve would come from, and it
    would silently give a decoder two emissions for one position.
    """
    torch.manual_seed(0)
    pc = _hmm()

    frontier = juice.queries.sample(pc, num_samples = 2048, _sample_input_ns = False)
    layer = pc.input_layer_group[0]
    start, _ = layer._output_ind_range

    for j in range(0, 2048, 97):                         # a stride, so this stays quick
        column = frontier[:, j]
        vids = layer.vids[column[column >= 0] - start, 0]
        assert sorted(vids.tolist()) == list(range(pc.num_vars)), \
            f"sample {j} resolved {sorted(vids.tolist())}"


# ------------------------------------------------------------ ids are GLOBAL, across input layers

@cuda_only
def test_the_frontier_spans_every_input_layer():
    """
    REGRESSION for a decode that looks right and is not.

    The returned ids are GLOBAL node ids. With a single input layer -- the common case, and every
    all-`Categorical` circuit -- `id - input_layer_group[0]._output_ind_range[0]` happens to be a
    valid index into that layer's `vids`, so the shortcut works and keeps working. Give a circuit
    two input layers (different distribution TYPES) and half the ids belong to the second: the
    subtraction then indexes past the end of the first layer's `vids`, which is a device-side assert
    that poisons the CUDA context rather than a clean error.

    This test pins the CORRECT decode -- attribute each id to its owning layer first -- and that the
    frontier really does span both, so the hazard cannot quietly disappear from the fixture.
    """
    torch.manual_seed(0)
    pc = _mixed_distributions()

    layers = [l for l in pc.input_layer_group]
    assert len(layers) > 1, "this circuit no longer has two input layers -- the test is vacuous"

    frontier = juice.queries.sample(pc, num_samples = 256, _sample_input_ns = False)
    live = frontier[frontier >= 0]

    first_start, first_end = layers[0]._output_ind_range
    assert int((live >= first_end).sum()) > 0, \
        "no id came from the second layer -- the fixture no longer exercises the hazard"

    # the correct decode: resolve the owning layer, THEN subtract its start
    for j in range(0, 256, 31):
        column = frontier[:, j]
        seen = []
        for node_id in column[column >= 0].tolist():
            layer, start = _owning_layer(pc, node_id)
            assert layer is not None, f"id {node_id} belongs to no input layer"
            local = node_id - start
            assert local < layer.vids.size(0), \
                f"id {node_id} is out of range for its own layer"
            seen.append(int(layer.vids[local, 0]))

        assert sorted(seen) == list(range(pc.num_vars)), f"sample {j} resolved {sorted(seen)}"


# ------------------------------------------------------------------------ ownership and sequencing

@cuda_only
@pytest.mark.parametrize("use_cudagraph", [False, True])
def test_two_frontiers_held_at_once_do_not_share_storage(use_cudagraph):
    """A caller that keeps one draw while making another must not see the first change. Under
    `use_cudagraph` the workspace is persistent, so only an explicit copy prevents this."""
    torch.manual_seed(0)
    pc = _hmm()

    first = juice.queries.sample(pc, num_samples = 256, use_cudagraph = use_cudagraph,
                                 _sample_input_ns = False)
    snapshot = first.clone()
    second = juice.queries.sample(pc, num_samples = 256, use_cudagraph = use_cudagraph,
                                  _sample_input_ns = False)

    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, snapshot), "the first frontier was overwritten by the second draw"


@cuda_only
def test_a_frontier_draw_does_not_disturb_a_backward():
    """
    Frontier mode still fills `pc.node_flows` before returning -- the emission step is what it skips,
    not the bookkeeping -- so a draw between a forward and its backward overwrites the flows a
    caller may be holding. What must NOT happen is the backward itself coming out different, which
    it does not, because it recomputes.
    """
    torch.manual_seed(0)
    data = torch.randint(0, NUM_CATS, [512, _hmm().num_vars], device = torch.device("cuda:0"))

    def flows(interpose):
        # a FRESH circuit each time: `pc.param_flows` accumulates across backward calls (measured,
        # exactly 2.0x after two), so reusing one circuit would compare one backward against two
        torch.manual_seed(0)
        pc = _hmm()
        pc(data)
        if interpose:
            juice.queries.sample(pc, num_samples = 333, _sample_input_ns = False)
        pc.backward(data)
        return pc.param_flows.clone()

    assert torch.allclose(flows(False), flows(True), atol = 1e-5), \
        "a frontier draw between a forward and its backward changed the flows"


@cuda_only
def test_repeated_frontier_draws_do_not_accumulate_state():
    """The mode returns early; anything it skipped resetting would show up as drift."""
    torch.manual_seed(0)
    pc = _hmm()

    def seeded_draw():
        # BOTH generators: `sample()` takes its kernel seed from the `random` MODULE, which
        # `torch.manual_seed` does not touch, so seeding torch alone leaves the draw depending on how
        # many draws happened before it -- which is precisely what this test varies.
        torch.manual_seed(5)
        random.seed(5)
        return juice.queries.sample(pc, num_samples = 128, _sample_input_ns = False).clone()

    first = seeded_draw()
    for _ in range(50):
        juice.queries.sample(pc, num_samples = 128, _sample_input_ns = False)
    last = seeded_draw()

    assert torch.equal(first, last), "the 1st and 51st draw at one seed differ -- state accumulated"
