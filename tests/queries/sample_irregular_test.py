"""
Sampling from IRREGULAR and BLOCK-SPARSE circuits, against an oracle that is not the other sampler.

The layout the default pass samples through is derived from the circuit (`scope_plan.py`), so its
correctness is a claim about structure: which scopes a layer owns, which rows they get, where a drawn
child lands. Structures that are all alike therefore prove very little about it. That is not
hypothetical -- the layout shipped giving each `(layer, scope)` its own element row, which is wrong
whenever one product GROUP owns a scope in several layers, and the whole suite passed anyway because
no structure it exercised had such a group.

So the structures here are chosen to be awkward on one axis each:

  * `ragged`          -- block-sparse `edge_ids` in the product AND the sum layers, uneven block counts
  * `mixed_arity`     -- one scope split two ways, so the decomposition depends on the draw (not
                         structured decomposable, and the frontier's shape genuinely varies)
  * `unbalanced`      -- the same variable reached at different DEPTHS down the two branches
  * `unblocked`       -- `block_size = 1` throughout with sparse edges, so no blocking hides an
                         addressing error
  * `hetero`          -- three branches over one scope at block sizes 4 / 2 / 1, which puts one scope
                         in three layers of one product group
  * `pd` / `rat_spn`  -- the library's own non-structured-decomposable structures

**The oracle is the forward pass, not the other sampler.** `P(x_v = k)` is read off `pc()` with every
other variable marginalized out, so a fault would have to corrupt sampling and marginalization
identically and in the same direction to survive. Comparing against the pair-list sampler instead
would only pin that the two agree, and they share the compiled tables.

:note: the conditional check covers the MISSING variables only. `sample(conditional = True)` does not
       hand back the evidence for an observed variable, it redraws it from the input node the
       posterior selected -- both sampling paths do this identically, so it is the vanilla semantics
       rather than anything about the layout, and asserting the evidence comes back verbatim would
       pin a behaviour this suite does not own.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams, SumNodes


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_CATS = 4
N_DRAWS = 40_000
#: Judged in standard errors. Over the few dozen (variable, category) cells per structure a 5 sigma
#: bar is ~1 false failure in 1e5 runs, while the layout faults this file exists for miss by 70-220.
Z_BAR = 5.0


# --------------------------------------------------------------------------------- structures

def _I(v, num_node_blocks, block_size):
    return inputs(v, num_node_blocks = num_node_blocks, block_size = block_size,
                  dist = dists.Categorical(num_cats = NUM_CATS))


def _sparse_edges(num_par_blocks, num_ch_blocks, num_edges):
    """A deterministic block-sparse edge list that leaves every parent block with at least one edge."""
    return torch.tensor([[i % num_par_blocks for i in range(num_edges)],
                         [(i * 3) % num_ch_blocks for i in range(num_edges)]], dtype = torch.long)


def _ragged():
    i = [_I(v, 3, 2) for v in range(4)]
    s0 = summate(multiply(i[0], i[1], edge_ids = torch.tensor([[0, 0], [1, 2], [2, 1]])),
                 edge_ids = _sparse_edges(2, 3, 5), num_node_blocks = 2, block_size = 2)
    s1 = summate(multiply(i[2], i[3], edge_ids = torch.tensor([[0, 1], [1, 0], [2, 2]])),
                 edge_ids = _sparse_edges(3, 3, 7), num_node_blocks = 3, block_size = 2)
    return summate(multiply(s0, s1, edge_ids = torch.tensor([[0, 0], [1, 1], [1, 2]])),
                   num_node_blocks = 1, block_size = 1)


def _mixed_arity():
    i = [_I(v, 2, 4) for v in range(4)]
    a = summate(multiply(i[0], i[1]), num_node_blocks = 2, block_size = 4)
    b = summate(multiply(i[2], i[3]), num_node_blocks = 2, block_size = 4)
    c = summate(multiply(i[2]), num_node_blocks = 2, block_size = 4)
    d = summate(multiply(i[3]), num_node_blocks = 2, block_size = 4)
    return summate(multiply(a, b), multiply(a, c, d), num_node_blocks = 1, block_size = 1)


def _unbalanced():
    i = [_I(v, 2, 2) for v in range(4)]
    inner = summate(multiply(i[0], i[1]), num_node_blocks = 2, block_size = 2)
    inner = summate(multiply(inner, i[2]), num_node_blocks = 2, block_size = 2)
    deep = summate(multiply(inner, i[3]), num_node_blocks = 1, block_size = 1)
    shallow = summate(multiply(i[0], i[1], i[2], i[3]), num_node_blocks = 1, block_size = 1)
    return summate(multiply(deep), multiply(shallow), num_node_blocks = 1, block_size = 1)


def _unblocked():
    i = [_I(v, 4, 1) for v in range(4)]
    a = summate(multiply(i[0], i[1]), edge_ids = _sparse_edges(3, 4, 9), num_node_blocks = 3, block_size = 1)
    b = summate(multiply(i[2], i[3]), edge_ids = _sparse_edges(3, 4, 8), num_node_blocks = 3, block_size = 1)
    return summate(multiply(a, b), edge_ids = _sparse_edges(1, 3, 3), num_node_blocks = 1, block_size = 1)


def _hetero():
    def branch(block_size, num_node_blocks):
        i = [_I(v, num_node_blocks, block_size) for v in range(4)]
        a = summate(multiply(i[0], i[1]), num_node_blocks = num_node_blocks, block_size = block_size)
        b = summate(multiply(i[2], i[3]), num_node_blocks = num_node_blocks, block_size = block_size)
        return summate(multiply(a, b), num_node_blocks = 1, block_size = 1)

    return summate(multiply(branch(4, 2)), multiply(branch(2, 4)), multiply(branch(1, 8)),
                   num_node_blocks = 1, block_size = 1)


def _pd():
    return juice.structures.PD(data_shape = (4,), num_latents = 6, split_intervals = (1,),
                               input_dist = dists.Categorical(num_cats = NUM_CATS))


def _rat():
    return juice.structures.RAT_SPN(num_vars = 4, num_latents = 4, depth = 1, num_repetitions = 3,
                                    num_pieces = 2, input_dist = dists.Categorical(num_cats = NUM_CATS))


STRUCTURES = {"ragged": _ragged, "mixed_arity": _mixed_arity, "unbalanced": _unbalanced,
              "unblocked": _unblocked, "hetero": _hetero, "pd": _pd, "rat_spn": _rat}


@pytest.fixture(scope = "module")
def circuits():
    out = {}
    for name, build in STRUCTURES.items():
        torch.manual_seed(0)
        ns = build()
        ns.init_parameters(perturbation = 2.0)
        out[name] = (ns, juice.compile(ns, verbose = False).to(torch.device("cuda:0")))
    return out


# --------------------------------------------------------------------------------- the oracle

def _exact_marginals(pc, observed = None, external = None):
    """
    `P(x_v = k | evidence)` straight from the forward pass, with every other variable marginalized.

    Independent of both samplers: it shares the compiled parameters but nothing about how a draw is
    routed through the frontier.
    """
    observed = observed or {}
    kwargs = {"sum_external_params": external} if external is not None else {}
    out = torch.zeros([pc.num_vars, NUM_CATS], device = pc.device)

    for v in range(pc.num_vars):
        if v in observed:
            continue
        x = torch.zeros([NUM_CATS, pc.num_vars], dtype = torch.long, device = pc.device)
        x[:, v] = torch.arange(NUM_CATS, device = pc.device)
        for var, val in observed.items():
            x[:, var] = val

        missing = torch.ones([pc.num_vars], dtype = torch.bool, device = pc.device)
        missing[v] = False
        for var in observed:
            missing[var] = False

        out[v] = pc(x, missing_mask = missing, **kwargs).exp().flatten().detach()

    keep = [v for v in range(pc.num_vars) if v not in observed]
    out[keep] /= out[keep].sum(dim = 1, keepdim = True)
    return out, keep


def _frequencies(samples):
    return torch.stack([(samples == k).float().mean(dim = 0) for k in range(NUM_CATS)], dim = 1)


def _max_z(empirical, exact, rows, num_samples):
    se = (exact[rows] * (1.0 - exact[rows]) / num_samples).sqrt().clamp(min = 1e-9)
    return float(((empirical[rows] - exact[rows]) / se).abs().max())


# --------------------------------------------------------------------------------- the structures bite

@cuda_only
def test_the_structures_are_actually_irregular(circuits):
    """
    These tests are only worth their runtime while the structures stay awkward. Each property is
    READ BACK off the compiled circuit rather than trusted from the builder, because a change to
    compilation could quietly regularize one -- which is exactly how the suite came to have five
    structures that all looked different and all exercised the same layout.
    """
    def block_sparse(ns):
        for node in ns:
            if isinstance(node, SumNodes) and node.edge_ids is not None:
                if node.edge_ids.size(1) < node.num_node_blocks * node.num_ch_node_blocks:
                    return True
        return False

    def overlapping_product_layers(pc):
        worst = 0
        for group in pc.inner_layer_groups:
            if not group.is_prod():
                continue
            counts = {}
            for layer in group:
                for key in {tuple(sorted(scope)) for scope in layer.scopes}:
                    counts[key] = counts.get(key, 0) + 1
            worst = max(worst, max(counts.values(), default = 0))
        return worst

    ns, pc = circuits["ragged"]
    assert block_sparse(ns), "'ragged' no longer has block-sparse edges"
    ns, pc = circuits["unblocked"]
    assert block_sparse(ns) and any(node.block_size == 1 for node in ns if isinstance(node, SumNodes))

    assert overlapping_product_layers(circuits["hetero"][1]) >= 3, \
        "'hetero' no longer puts one scope in three product layers of a group"

    non_sd = [name for name, (_, pc) in circuits.items() if not pc.is_structured_decomposable]
    assert len(non_sd) >= 3, f"only {non_sd} are non-structured-decomposable"


# --------------------------------------------------------------------------------- draws

@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_a_draw_reaches_every_variable(circuits, name):
    """
    A branch the layout fails to route leaves its rows untouched, and an untouched row comes back as
    the buffer's `0` -- a perfectly plausible category. Coverage is checked on the FRONTIER, where an
    unset entry is `-1` and cannot be mistaken for a draw.
    """
    _, pc = circuits[name]

    frontier = juice.queries.sample(pc, num_samples = 2048, _sample_input_ns = False)
    covered = (frontier != -1).sum(dim = 0)

    assert bool((covered == pc.num_vars).all()), \
        f"{int((covered < pc.num_vars).sum())} of 2048 draws are missing a variable"


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_unconditional_draws_match_the_exact_marginals(circuits, name):
    _, pc = circuits[name]

    exact, rows = _exact_marginals(pc)
    torch.manual_seed(1)
    samples = juice.queries.sample(pc, num_samples = N_DRAWS)

    z = _max_z(_frequencies(samples), exact, rows, N_DRAWS)
    assert z < Z_BAR, f"'{name}' draws differ from the exact marginals: max |z| = {z:.2f}"


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_conditional_draws_match_the_exact_posterior(circuits, name):
    """Evidence on variable 0; the variables that were actually missing are compared against the
    exact posterior. Conditioning changes which nodes are live, so the frontier is a different shape
    from the unconditional one and is worth pinning separately."""
    _, pc = circuits[name]

    # The oracle FIRST: it runs its own forwards, and a conditional draw reads whatever the last one
    # left behind -- building it after the conditioning pass silently samples against that instead,
    # at the oracle's batch size rather than this one's.
    exact, rows = _exact_marginals(pc, observed = {0: 1})

    x = torch.zeros([N_DRAWS, pc.num_vars], dtype = torch.long, device = pc.device)
    x[:, 0] = 1
    missing = torch.ones([pc.num_vars], dtype = torch.bool, device = pc.device)
    missing[0] = False
    pc(x, missing_mask = missing)

    torch.manual_seed(2)
    samples = juice.queries.sample(pc, conditional = True)
    assert samples.size(0) == N_DRAWS, "the conditional draw did not follow the conditioning pass"

    z = _max_z(_frequencies(samples), exact, rows, N_DRAWS)
    assert z < Z_BAR, f"'{name}' conditional draws differ from the exact posterior: max |z| = {z:.2f}"


# --------------------------------------------------------------------------------- gated + irregular

def _gated_irregular():
    """A `BlockScaleSumParams` node with block-sparse edges, under a root whose branches overlap --
    the intersection of this branch's feature with the structures above."""
    with juice.set_block_size(2):
        i = [_I(v, 3, 2) for v in range(4)]
        gated = summate(multiply(i[0], i[1], edge_ids = torch.tensor([[0, 0], [1, 2], [2, 1]])),
                        edge_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 2, 0, 1, 2]]),
                        num_node_blocks = 2,
                        external_params = BlockScaleSumParams(ch_block_size = 1))
        other = summate(multiply(i[2], i[3]), num_node_blocks = 2, block_size = 2)
        branch_a = summate(multiply(gated, other), num_node_blocks = 1, block_size = 1)

    with juice.set_block_size(4):
        j = [_I(v, 2, 4) for v in range(4)]
        a = summate(multiply(j[0], j[1]), num_node_blocks = 2, block_size = 4)
        b = summate(multiply(j[2], j[3]), num_node_blocks = 2, block_size = 4)
        branch_b = summate(multiply(a, b), num_node_blocks = 1, block_size = 1)

    root = summate(multiply(branch_a), multiply(branch_b), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(torch.device("cuda:0")), gated


@pytest.fixture(scope = "module")
def gated():
    torch.manual_seed(0)
    pc, ns = _gated_irregular()
    num_node_gates = ns.num_nodes // ns.external_params.gate_sizes(ns)[0]
    num_ch_gates = ns.num_ch_nodes // ns.external_params.gate_sizes(ns)[1]

    torch.manual_seed(3)
    # ONE gate held across the batch, so the gated circuit is a fixed distribution the forward pass
    # can be asked for exactly. Large enough to move the answer well clear of the ungated one.
    gate = torch.randn([1, num_node_gates, num_ch_gates], device = pc.device) * 3.0
    return pc, ns, gate


@cuda_only
def test_gated_draws_on_an_irregular_circuit_match_the_exact_gated_marginals(gated):
    pc, ns, gate = gated

    exact, rows = _exact_marginals(pc, external = {ns: gate.expand(NUM_CATS, -1, -1).contiguous()})
    torch.manual_seed(1)
    samples = juice.queries.sample(pc, num_samples = N_DRAWS,
                                   sum_external_params = {ns: gate.expand(N_DRAWS, -1, -1).contiguous()})

    z = _max_z(_frequencies(samples), exact, rows, N_DRAWS)
    assert z < Z_BAR, f"gated draws differ from the exact gated marginals: max |z| = {z:.2f}"


@cuda_only
def test_the_gate_is_load_bearing(gated):
    """The check above passes trivially if the gate is ignored AND the oracle ignores it too -- both
    read it through the same staging. So the gated draw is also required to differ from the ungated
    one, which fixes the direction."""
    pc, ns, gate = gated

    torch.manual_seed(1)
    with_gate = juice.queries.sample(pc, num_samples = N_DRAWS,
                                     sum_external_params = {ns: gate.expand(N_DRAWS, -1, -1).contiguous()})
    torch.manual_seed(1)
    without = juice.queries.sample(pc, num_samples = N_DRAWS)

    exact_gated, rows = _exact_marginals(pc, external = {ns: gate.expand(NUM_CATS, -1, -1).contiguous()})
    exact_plain, _ = _exact_marginals(pc)

    assert float((exact_gated[rows] - exact_plain[rows]).abs().max()) > 0.02, \
        "this gate barely moves the distribution -- the test cannot tell the paths apart"

    se = ((with_gate.float().var(dim = 0) + without.float().var(dim = 0)) / N_DRAWS).sqrt().clamp(min = 1e-9)
    z = float(((with_gate.float().mean(dim = 0) - without.float().mean(dim = 0)) / se).abs().max())
    assert z > Z_BAR, f"gated and ungated draws are indistinguishable: max |z| = {z:.2f}"


@cuda_only
def test_a_gated_draw_reaches_every_variable(gated):
    pc, ns, gate = gated

    frontier = juice.queries.sample(pc, num_samples = 2048, _sample_input_ns = False,
                                    sum_external_params = {ns: gate.expand(2048, -1, -1).contiguous()})
    covered = (frontier != -1).sum(dim = 0)

    assert bool((covered == pc.num_vars).all())
