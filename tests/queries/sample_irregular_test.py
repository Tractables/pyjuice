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

and then four ORDINARY DENSE ones at a realistic size -- a 12-variable 32-state chain, a 16-variable
HCLT, and PD / RAT-SPN at 16 variables -- because everything above is four variables wide, which says
nothing about a circuit anyone would actually build. They run against the same oracle.

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

import math

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams, SumNodes
from pyjuice.nodes.input_nodes import InputNodes


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_CATS = 4
N_DRAWS = 40_000

#: How peaked the randomly-drawn parameters are: `_params = exp(rand * -perturbation)`, so 2.0 spans
#: at most 7.4x between the edges of one parent and 6.0 spans ~400x.
#:
#: TUNED, not inherited. Near-uniform parameters hide a wrong choice -- in the limit of a uniform
#: circuit a completely broken sum sampler is indistinguishable from a correct one. MEASURED against a
#: seeded wrong-child fault: 22 sigma at perturbation 2.0 on `ragged`, 137 at 8.0. It is not pushed
#: higher because peaked parameters also thin the rare categories, and the statistic below needs an
#: expected count of at least 5 per cell -- at 8.0 the 256-category HCLT falls to 3.3.
PERTURBATION = 6.0

#: Weight ratios for the MANUAL parameterization; see `_set_parameters`.
DECAY = 0.35        # between successive edges of one parent
SPREAD = 0.55       # across positions inside a parameter block
TAU = 0.7           # emission peak width, in categories

#: Mass held OUT of the peak and spread evenly over the categories. A total, not a per-category
#: constant: a constant floor divided by 256 categories is negligible, while a constant floor ADDED
#: to each of 4 categories swamps the peak -- the first version of this set 0.30 per category and
#: made the manual parameters LESS discriminating than random ones (leaf separation 0.21 against
#: 0.47-0.94). Holding the total fixed keeps the peak sharp at every category count while still
#: guaranteeing no cell is too rare for the statistic.
FLOOR_MASS = 0.30


def _set_parameters(root):
    """
    A deterministic, strongly non-uniform parameter structure -- no RNG anywhere.

    Two properties, and the second is the one that is easy to get wrong.

    SEPARATION -- each input node peaks on its own category, so an emitted value all but names the
    node the path went through, and each parent's edges are strongly unequal so which edge was taken
    shows too.

    IRREGULARITY -- the profile differs from parent to parent and from group to group. A structure
    that is merely non-uniform but SYMMETRIC is a trap: give every parent the same decreasing edge
    profile and cycle the emission centres in step, and a bug that permutes children maps the circuit
    onto a near-copy of itself and cancels in the marginals. MEASURED on a seeded wrong-child fault,
    the regular version of this scored 2.7 sigma on `hmm_dense` and 2.9 on `pd_dense` -- it would have
    let the fault through -- where random parameters scored 21 and 63.
    """
    for group_idx, ns in enumerate(root):
        if getattr(ns, "_source_node", None) is not None:
            continue                                        # tied: follows its source

        if isinstance(ns, SumNodes):
            num_edges, block_size, ch_block_size = ns._params.shape
            parents = ns.edge_ids[0].tolist()

            rank, seen = torch.zeros(num_edges, dtype = torch.long), {}
            for k, parent in enumerate(parents):            # this edge's position among its parent's
                rank[k] = seen.get(parent, 0)
                seen[parent] = int(rank[k]) + 1

            grid = (torch.arange(block_size)[:, None] * ch_block_size
                    + torch.arange(ch_block_size)[None, :])
            # the exponent mixes the edge's rank, WHICH parent it belongs to, its position in the
            # block and which group this is, so no two parents share a profile
            exponent = (rank[:, None, None] * 1
                        + torch.tensor(parents)[:, None, None] * 2
                        + grid[None, :, :]
                        + group_idx * 3) % 5
            w = DECAY ** exponent.double()

            for parent in set(parents):                     # normalize each PARENT NODE's edges
                sel = [k for k, p in enumerate(parents) if p == parent]
                w[sel] = w[sel] / w[sel].sum(dim = (0, 2))[None, :, None]
            ns._params = w.float()

        elif isinstance(ns, InputNodes) and isinstance(ns.dist, dists.Categorical):
            num_nodes, num_cats = ns.num_nodes, ns.dist.num_cats
            step = max(1, num_cats // max(num_nodes, 1))
            # a stride coprime-ish with the category count, offset per group: centres do not line up
            # between one variable and the next
            centre = ((torch.arange(num_nodes) * step + torch.arange(num_nodes) * 5
                       + group_idx * 3) % num_cats)[:, None]
            d = (torch.arange(num_cats)[None, :] - centre).abs()
            d = torch.minimum(d, num_cats - d)              # circular, so no node sits at an edge

            # widths vary too, so two nodes differ in shape and not only in position
            tau = TAU * (1.0 + 0.4 * ((torch.arange(num_nodes) + group_idx) % 3).double())[:, None]
            peak = torch.exp(-d.double() / tau)
            peak = peak / peak.sum(dim = 1, keepdim = True)
            p = FLOOR_MASS / num_cats + (1.0 - FLOOR_MASS) * peak
            ns._params = (p / p.sum(dim = 1, keepdim = True)).float().flatten()

#: Family-wise false-failure rate the bar is calibrated to. A draw is compared cell by cell over
#: (variable, category), so the bar has to depend on HOW MANY cells -- a constant does not survive the
#: range here. MEASURED on the 4096-cell HCLT: the null maximum has median 3.95 and the expected
#: maximum of `n` standard normals is sqrt(2 ln n) = 4.08, so a flat 5 sigma would false-fail once in
#: 426 runs. Bonferroni against this alpha gives ~4.98 at 16 cells and ~5.97 at 4096.
ALPHA = 1e-5


def _z_bar(num_cells):
    return float(math.sqrt(2) * torch.erfinv(torch.tensor(1.0 - ALPHA / num_cells,
                                                          dtype = torch.float64)))


def _num_cats(pc):
    """
    Asked of the compiled circuit, never assumed.

    Both `PD` and `HCLT` ignore a `num_bins` argument for their input distribution and default to
    `Categorical(num_cats = 256)`. Comparing a draw from those against a 4-category reference covers
    1.4% of the mass and reports a confident 145-175 sigma "failure" that is entirely the oracle's.
    """
    return pc.input_layer_group[0].nodes[0].dist.num_cats


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


def _hmm_dense():
    """A DENSE structure at realistic size: 12 variables over a 32-state chain."""
    with juice.set_block_size(32):
        ns = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        for v in range(1, 12):
            ns = summate(multiply(ns, inputs(v, num_node_blocks = 1,
                                             dist = dists.Categorical(num_cats = NUM_CATS))),
                         num_node_blocks = 1)
        return summate(multiply(ns), num_node_blocks = 1, block_size = 1)


def _hclt_dense():
    """16 variables, 16 latents, and 256 categories -- 4096 cells, which is what forced the bar to be
    calibrated rather than constant."""
    data = torch.randint(0, 8, [2048, 16]).float().to(torch.device("cuda:0"))
    return juice.structures.HCLT(data, num_bins = 8, sigma = 0.5 / 8, num_latents = 16,
                                 chunk_size = 16)


def _pd_dense():
    return juice.structures.PD(data_shape = (16,), num_latents = 16, split_intervals = (4,),
                               input_dist = dists.Categorical(num_cats = NUM_CATS))


def _rat_dense():
    return juice.structures.RAT_SPN(num_vars = 16, num_latents = 8, depth = 2, num_repetitions = 3,
                                    num_pieces = 2, input_dist = dists.Categorical(num_cats = NUM_CATS))


STRUCTURES = {"ragged": _ragged, "mixed_arity": _mixed_arity, "unbalanced": _unbalanced,
              "unblocked": _unblocked, "hetero": _hetero, "pd": _pd, "rat_spn": _rat,
              # the awkward structures above are all 4 variables, which says nothing about a circuit
              # of a realistic size. These are ordinary DENSE ones, checked against the same oracle.
              "hmm_dense": _hmm_dense, "hclt_dense": _hclt_dense,
              "pd_dense": _pd_dense, "rat_dense": _rat_dense}


#: Every structural test below runs under BOTH parameterizations, because NEITHER DOMINATES --
#: measured, against a seeded wrong-child fault, in sigma:
#:
#:                     random p=6.0   manual
#:     unblocked             1262       236
#:     hetero                 362       119
#:     rat_dense               45        14
#:     hmm_dense               19.5       5.6
#:     hclt_dense              39       358
#:
#: Random parameters win nearly everywhere; the deliberate ones win by an order of magnitude on the
#: 256-category HCLT, where random initialization leaves the rare categories both flat and thinly
#: sampled (11.8 expected counts against 46.8). Running both makes the suite's sensitivity the better
#: of the two, and costs one extra compile per structure.
PARAMETERIZATIONS = ("random", "manual")


@pytest.fixture(scope = "module")
def _compiled():
    """Every (parameterization, structure), compiled once for the module."""
    out = {}
    for mode in PARAMETERIZATIONS:
        for name, build in STRUCTURES.items():
            torch.manual_seed(0)
            ns = build()
            ns.init_parameters(perturbation = PERTURBATION)     # allocates the tensors
            if mode == "manual":
                _set_parameters(ns)                             # then overwrites them
            out[(mode, name)] = (ns, juice.compile(ns, verbose = False).to(torch.device("cuda:0")))
    return out


@pytest.fixture(params = PARAMETERIZATIONS)
def circuits(request, _compiled):
    return {name: _compiled[(request.param, name)] for name in STRUCTURES}


# --------------------------------------------------------------------------------- the oracle

def _exact_marginals(pc, observed = None, external = None):
    """
    `P(x_v = k | evidence)` straight from the forward pass, with every other variable marginalized.

    Independent of both samplers: it shares the compiled parameters but nothing about how a draw is
    routed through the frontier.
    """
    observed = observed or {}
    kwargs = {"sum_external_params": external} if external is not None else {}
    num_cats = _num_cats(pc)
    out = torch.zeros([pc.num_vars, num_cats], device = pc.device)

    for v in range(pc.num_vars):
        if v in observed:
            continue
        x = torch.zeros([num_cats, pc.num_vars], dtype = torch.long, device = pc.device)
        x[:, v] = torch.arange(num_cats, device = pc.device)
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


def _frequencies(samples, num_cats):
    return torch.stack([(samples == k).float().mean(dim = 0) for k in range(num_cats)], dim = 1)


def _assert_matches(pc, samples, exact, rows, label):
    """Compare a draw against the exact reference, at a bar calibrated to the number of cells."""
    empirical = _frequencies(samples, _num_cats(pc))
    se = (exact[rows] * (1.0 - exact[rows]) / N_DRAWS).sqrt().clamp(min = 1e-9)
    z = float(((empirical[rows] - exact[rows]) / se).abs().max())

    cells = len(rows) * _num_cats(pc)
    bar = _z_bar(cells)
    assert z < bar, f"{label}: max |z| = {z:.2f} over {cells} cells (bar {bar:.2f})"


def _leaf_separation(root):
    """
    How distinguishable two nodes of one variable are, minimised over the variables.

    This is the MECHANISM that makes a wrong choice visible: a draw reveals which node the path went
    through only to the extent that nodes emit differently. Judged here rather than on how far the
    aggregate marginal sits from uniform, which is a bad proxy -- a mixture over many peaked nodes
    averages back to nearly uniform, so HCLT's 256-category marginal spans 0.013 while being a
    perfectly discriminating circuit.
    """
    worst = 1.0
    for ns in root:
        if isinstance(ns, InputNodes) and isinstance(ns.dist, dists.Categorical) and ns.num_nodes > 1:
            p = ns._params.reshape(ns.num_nodes, ns.dist.num_cats).double()
            tv = 0.5 * (p[:, None, :] - p[None, :, :]).abs().sum(dim = -1)
            worst = min(worst, float(tv.max()))
    return worst


@cuda_only
@pytest.mark.parametrize("name", list(STRUCTURES))
def test_the_parameters_can_show_a_wrong_choice(circuits, name):
    """
    A guard on the INSTRUMENT rather than on the sampler.

    Every check in this file compares a draw against a reference, and a circuit close to uniform has
    almost the same distribution whichever child a sum node picks -- so a thoroughly broken sampler
    would pass. Two things are required of the parameters: that nodes of one variable emit
    differently enough for the choice between them to show, and that no cell is so rare the normal
    approximation behind the z-statistic stops holding.

    MEASURED against a seeded wrong-child fault, to show the first of those is not merely plausible:
    22 sigma at perturbation 2.0 on `ragged`, 137 at 8.0.
    """
    root, pc = circuits[name]

    separation = _leaf_separation(root)
    assert separation > 0.25, \
        f"'{name}' has two nodes of one variable only {separation:.3f} apart -- a wrong choice between " \
        f"them would barely move the answer, so this circuit cannot police the sampler"

    exact, rows = _exact_marginals(pc)
    smallest = float(exact[rows].min()) * N_DRAWS
    assert smallest >= 5.0, \
        f"'{name}' has a cell with {smallest:.1f} expected counts -- the z-statistic is not valid"


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

    _assert_matches(pc, samples, exact, rows, f"'{name}' draws differ from the exact marginals")


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

    _assert_matches(pc, samples, exact, rows, f"'{name}' conditional draws differ from the posterior")


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

    _assert_matches(pc, samples, exact, rows, "gated draws differ from the exact gated marginals")


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
    assert z > _z_bar(pc.num_vars), f"gated and ungated draws are indistinguishable: max |z| = {z:.2f}"


@cuda_only
def test_a_gated_draw_reaches_every_variable(gated):
    pc, ns, gate = gated

    frontier = juice.queries.sample(pc, num_samples = 2048, _sample_input_ns = False,
                                    sum_external_params = {ns: gate.expand(2048, -1, -1).contiguous()})
    covered = (frontier != -1).sum(dim = 0)

    assert bool((covered == pc.num_vars).all())
