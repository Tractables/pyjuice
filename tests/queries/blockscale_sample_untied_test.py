"""
Gated sampling with UNTIED per-node external parameters, one axis of the gate grid at a time.

The two existing gated files each leave a hole here. `blockscale_sample_hmm_test.py` feeds a whole
chain through ONE registered group tensor, and its block size equals the state count, so its gate
grid is `[batch, 1, n_ch_gates]` -- the node axis is degenerate and the per-`ns` supply path is never
taken. `blockscale_sample_test.py` does reach the node axis, but on a single gated node, so it cannot
confuse one `ns`'s parameters with another's.

Here every transition of a chain carries its OWN `BlockScaleSumParams` and is supplied as its own
entry in `sum_external_params`, and the grid is `[batch, 3, 3]` -- so `batch`, `ns`, and the two
within-`ns` axes are all non-degenerate and all distinguishable from each other.

**Everything is checked deterministically.** A one-hot gate at +-`GATE_SCALE` makes the child block a
FUNCTION of the gate, so each test asserts the drawn block per sample with no statistics at all --
which is both sharper than a frequency check and immune to the flakiness a near-deterministic
emission would bring. The block, not the child: a gate column spans `GATE_CBS` children that share
it, and which of those is drawn is still up to `theta`.

**How a draw is read.** `_sample_input_ns = False` returns the selected input node ids. The product
under transition `t` pairs node `j` of the previous chain with node `j` of emission `t`, so the
emission node id at variable `t` IS the child index that transition drew -- and the child index drawn
at `t + 1` is the index of the PARENT node that transition `t` was drawn from, which is what makes
the node axis observable.

Every test that asserts a gate is obeyed is paired with a check that it could have come out
otherwise: an axis test that permutes the targets and requires the draw to follow, so a kernel
ignoring that axis fails rather than passing on a coincidence.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

BLOCK_SIZE = 4          # states per block
NUM_BLOCKS = 3          # -> 12 states, and a 3-wide gate grid on BOTH axes
NUM_STATES = BLOCK_SIZE * NUM_BLOCKS
NUM_EMITS = 8
SEQ_LEN = 5
GATE_SCALE = 40.0       # exp(+-40) -- a one-hot column wins by ~1e34, so the block is deterministic
BATCH = 512

#: Two gate GRANULARITIES, because they take different addressing paths in the kernel.
#:
#: `block_aligned` puts exactly one gate row per node block and one column per child block, which is
#: the everyday case -- and the one whose sub-block refinements (`node_gate_off`, and the column
#: WITHIN an edge block) are identically zero. Mutating either to zero leaves that configuration
#: bit-identical, so on its own this file would claim node/child coverage it did not have.
#:
#: `sub_block` makes both gate sizes half the node/child block, so one node block spans two gate rows
#: and one edge block spans two columns, and those refinements carry the answer.
CONFIGS = {"block_aligned": {"gate_bs": None, "gate_cbs": BLOCK_SIZE},
           "sub_block":     {"gate_bs": BLOCK_SIZE // 2, "gate_cbs": BLOCK_SIZE // 2}}


class Chain():
    """
    The circuit plus its gate granularity, so a test can turn a drawn node id into the gate row or
    column that decided it.

    The granularity is ASKED OF the parameterization rather than recomputed from what was passed in:
    `block_size = None` means "the node's own", a default that belongs to `BlockScaleSumParams`, and
    a test that restates it would keep agreeing with itself after the real one changed.
    """

    def __init__(self, pc, emits, transitions):
        self.pc, self.emits, self.transitions = pc, emits, transitions

        ns = transitions[0]
        self.gate_bs, self.gate_cbs = ns.external_params.gate_sizes(ns)
        self.num_node_gates = ns.num_nodes // self.gate_bs
        self.num_ch_gates = ns.num_ch_nodes // self.gate_cbs


def _build(gate_bs, gate_cbs, seed = 0):
    """A chain whose transitions are UNTIED: each carries its own external parameters."""
    torch.manual_seed(seed)
    with juice.set_block_size(BLOCK_SIZE):
        base = inputs(0, num_node_blocks = NUM_BLOCKS, dist = dists.Categorical(num_cats = NUM_EMITS))
        emits, transitions, ns = [base], [], base
        for t in range(1, SEQ_LEN):
            emit = inputs(t, num_node_blocks = NUM_BLOCKS, dist = dists.Categorical(num_cats = NUM_EMITS))
            emits.append(emit)
            ns = summate(multiply(ns, emit), num_node_blocks = NUM_BLOCKS,
                         external_params = BlockScaleSumParams(block_size = gate_bs,
                                                               ch_block_size = gate_cbs))
            transitions.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    return Chain(pc, emits, transitions)


@pytest.fixture(scope = "module")
def built():
    """Both configurations, compiled ONCE for the module.

    Parametrizing a module-scoped fixture directly looks equivalent and is not: this suite's conftest
    reorders tests by cached duration, which interleaves the two parameters, and pytest keeps only
    the most recent one -- MEASURED at 16 rebuilds of a 2-configuration file. Caching the pair and
    parametrizing a cheap lookup makes the count 2 whatever order the tests run in.
    """
    return {name: _build(**config) for name, config in CONFIGS.items()}


@pytest.fixture(params = list(CONFIGS))
def chain(request, built):
    return built[request.param]


def _one_hot(c, targets):
    """
    Gates that pin the drawn child gate-column.

    `targets`: `[batch, num_node_gates]` long, the hot column for each (sample, node gate row).
    Taking a full per-(sample, row) target -- rather than one scalar broadcast -- is what lets a
    single helper drive the batch axis, the node axis, and both at once.
    """
    batch, num_node_gates = targets.shape
    gates = torch.full([batch, num_node_gates, c.num_ch_gates], -GATE_SCALE, device = c.pc.device)
    gates.scatter_(2, targets.unsqueeze(-1).to(c.pc.device), GATE_SCALE)
    return gates


def _uniform_targets(c, value, batch = BATCH):
    return torch.full([batch, c.num_node_gates], int(value), dtype = torch.long, device = c.pc.device)


def _same_for_every_ns(c, gates):
    return {ns: gates for ns in c.transitions}


def _frontier(c, **kwargs):
    return juice.queries.sample(c.pc, _sample_input_ns = False, **kwargs)


def _child_at(c, frontier, t):
    """The child index transition `t` drew, per sample, read exactly off the frontier."""
    lo, hi = c.emits[t]._output_ind_range

    hit = (frontier >= lo) & (frontier < hi)
    assert bool(hit.sum(dim = 0).eq(1).all()), "each sample should select exactly one node of this variable"

    rows, cols = hit.nonzero(as_tuple = True)
    out = torch.zeros([frontier.size(1)], dtype = torch.long, device = frontier.device)
    out[cols] = frontier[rows, cols] - lo
    return out


def _column_at(c, frontier, t):
    """Which gate COLUMN the drawn child fell in."""
    return _child_at(c, frontier, t) // c.gate_cbs


def _row_at(c, frontier, t):
    """Which gate ROW transition `t` was drawn from -- i.e. the gate row of its parent node, which is
    the child index drawn one level up."""
    return _child_at(c, frontier, t + 1) // c.gate_bs


# --------------------------------------------------------------------------------- the batch axis

@cuda_only
def test_the_grid_is_non_degenerate_on_both_within_ns_axes(chain):
    """Guard on the instrument. Every axis test below is vacuous on a grid that is 1 wide, and the
    node axis of the existing chain test is exactly that."""
    c = chain
    assert c.num_node_gates > 1 and c.num_ch_gates > 1
    for ns in c.transitions:
        assert ns.num_nodes // ns.external_params.gate_sizes(ns)[0] == c.num_node_gates
        assert ns.num_ch_nodes // ns.external_params.gate_sizes(ns)[1] == c.num_ch_gates


@cuda_only
def test_the_batch_axis_gives_every_sample_its_own_gate(chain):
    """Each sample gets a different one-hot column, so a kernel that broadcast row 0 across the batch
    -- or transposed the batch and node axes -- gets the wrong column for all but a few samples."""
    c = chain
    torch.manual_seed(0)
    per_sample = torch.randint(0, c.num_ch_gates, [BATCH], device = c.pc.device)

    gates = _same_for_every_ns(c, _one_hot(c, per_sample[:, None].expand(-1, c.num_node_gates)))
    frontier = _frontier(c, num_samples = BATCH, sum_external_params = gates)

    for t in range(1, SEQ_LEN):
        assert torch.equal(_column_at(c, frontier, t), per_sample), \
            f"transition {t} did not follow the per-sample gate"


@cuda_only
def test_the_batch_axis_is_not_ignored(chain):
    """The paired discrimination check: permuting which sample gets which gate must permute the draw
    the same way. Without it, a kernel reading one shared row could pass the test above whenever the
    targets happened to agree."""
    c = chain
    torch.manual_seed(1)
    per_sample = torch.randint(0, c.num_ch_gates, [BATCH], device = c.pc.device)
    rolled = torch.roll(per_sample, shifts = 1)

    def draw(targets):
        gates = _same_for_every_ns(c, _one_hot(c, targets[:, None].expand(-1, c.num_node_gates)))
        return _column_at(c, _frontier(c, num_samples = BATCH, sum_external_params = gates), 1)

    assert torch.equal(draw(per_sample), per_sample)
    assert torch.equal(draw(rolled), rolled)
    assert not torch.equal(per_sample, rolled), "the roll was a no-op -- this check proves nothing"


# ------------------------------------------------------------------------------------ the ns axis

@cuda_only
def test_each_ns_reads_its_own_tensor(chain):
    """
    UNTIED supply: every transition gets its own entry in `sum_external_params`, each with a
    different target. A kernel that took the first tensor for every layer, or resolved a layer's slab
    of the staging buffer by position rather than by node, sends every transition to one column.
    """
    c = chain
    per_ns = [t % c.num_ch_gates for t in range(len(c.transitions))]
    gates = {ns: _one_hot(c, _uniform_targets(c, target))
             for ns, target in zip(c.transitions, per_ns)}

    frontier = _frontier(c, num_samples = BATCH, sum_external_params = gates)
    for t in range(1, SEQ_LEN):
        columns = _column_at(c, frontier, t)
        assert bool((columns == per_ns[t - 1]).all()), \
            f"transition {t} drew from column {int(columns[columns != per_ns[t - 1]][0])}, wanted {per_ns[t - 1]}"


@cuda_only
def test_rotating_the_targets_across_ns_rotates_the_draw(chain):
    """The `ns` axis is load-bearing: hand each transition the target meant for the next one and every
    level must move with it. Two transitions sharing a tensor would not budge."""
    c = chain

    def draw_all(targets_per_ns):
        gates = {ns: _one_hot(c, _uniform_targets(c, target))
                 for ns, target in zip(c.transitions, targets_per_ns)}
        frontier = _frontier(c, num_samples = BATCH, sum_external_params = gates)
        return [int(_column_at(c, frontier, t)[0]) for t in range(1, SEQ_LEN)]

    base = [t % c.num_ch_gates for t in range(len(c.transitions))]
    rotated = base[1:] + base[:1]

    assert draw_all(base) == base
    assert draw_all(rotated) == rotated
    assert base != rotated, "the rotation was a no-op -- this check proves nothing"


# ------------------------------------------------------------------------- the within-ns node axis

@cuda_only
def test_the_node_axis_selects_per_parent_gate_row(chain):
    """
    The axis the existing chain test cannot reach, since its grid is one row wide.

    Giving each gate row a different target makes the drawn column a function of which row the PARENT
    fell in. A kernel that indexed the gate table by child alone -- or swapped the two within-`ns`
    axes -- picks the wrong row for most parents.
    """
    c = chain
    per_row = torch.tensor([(k + 1) % c.num_ch_gates for k in range(c.num_node_gates)],
                           device = c.pc.device)
    gates = _same_for_every_ns(c, _one_hot(c, per_row[None, :].expand(BATCH, -1)))

    # ONE draw, or the parent and the child come from different samples
    frontier = _frontier(c, num_samples = BATCH, sum_external_params = gates)

    for t in range(1, SEQ_LEN - 1):
        assert torch.equal(_column_at(c, frontier, t), per_row[_row_at(c, frontier, t)]), \
            f"transition {t} ignored which gate row its parent was in"


# ------------------------------------------------------------------------ the within-ns child axis

@cuda_only
def test_the_child_axis_reaches_every_column(chain):
    """
    Sweeps the hot column across the WHOLE gate table -- a stride or off-by-one on the child axis
    survives any single fixed column and fails somewhere in a sweep.

    Swept inside the test rather than parametrized over a fixed range, because the two configurations
    have different widths and a shared range would report the wider one's extra columns as skips on
    the narrower one, which reads as missing coverage rather than as a table that is simply smaller.
    """
    c = chain
    assert c.num_ch_gates > 1, "a one-column table cannot show a stride error"

    for hot in range(c.num_ch_gates):
        gates = _same_for_every_ns(c, _one_hot(c, _uniform_targets(c, hot)))
        frontier = _frontier(c, num_samples = BATCH, sum_external_params = gates)

        for t in range(1, SEQ_LEN):
            assert bool((_column_at(c, frontier, t) == hot).all()), \
                f"transition {t} missed gate column {hot} of {c.num_ch_gates}"


# ------------------------------------------------------------------------------- all axes together

def _independent_targets(c, seed):
    """A target per (sample, ns, gate row) -- no two axes share a pattern, so any confusion between
    them changes at least one entry."""
    generator = torch.Generator(device = "cpu").manual_seed(seed)
    return {ns: torch.randint(0, c.num_ch_gates, [BATCH, c.num_node_gates],
                              generator = generator).to(c.pc.device)
            for ns in c.transitions}


def _assert_every_axis_honoured(c, targets, frontier):
    samples = torch.arange(frontier.size(1), device = frontier.device)
    for t in range(1, SEQ_LEN - 1):
        expected = targets[c.transitions[t - 1]][samples, _row_at(c, frontier, t)]
        drawn = _column_at(c, frontier, t)
        assert torch.equal(drawn, expected), \
            f"transition {t} disagreed on {int((drawn != expected).sum())} of {frontier.size(1)} samples"


@cuda_only
def test_every_axis_at_once(chain):
    """
    Batch, `ns` and node axes varying INDEPENDENTLY and simultaneously.

    Each test above holds two axes constant, which a kernel could survive by collapsing the axis
    under test onto one that happens to agree. Here every entry of every grid is drawn on its own.
    """
    c = chain
    targets = _independent_targets(c, seed = 7)
    gates = {ns: _one_hot(c, target) for ns, target in targets.items()}

    _assert_every_axis_honoured(c, targets,
                                _frontier(c, num_samples = BATCH, sum_external_params = gates))


@cuda_only
def test_every_axis_at_once_conditionally(chain):
    """The same on a conditional draw, where the gate has to reach both the forward pass that built
    `element_mars` and the top-down pass that reads it."""
    c = chain
    targets = _independent_targets(c, seed = 11)
    gates = {ns: _one_hot(c, target) for ns, target in targets.items()}

    torch.manual_seed(3)
    x = torch.randint(0, NUM_EMITS, [BATCH, SEQ_LEN], device = c.pc.device)
    missing = torch.zeros([SEQ_LEN], dtype = torch.bool, device = c.pc.device)
    missing[1::2] = True
    c.pc(x, missing_mask = missing, sum_external_params = gates)

    _assert_every_axis_honoured(c, targets, _frontier(c, conditional = True))


# ------------------------------------------------------------------------------------- invariants

@cuda_only
def test_untied_unit_gates_reproduce_the_plain_sampler(chain):
    """A zero log-gate is the identity, so an untied supply of them must give the plain draw. This is
    what says the untied path adds nothing of its own when the gates say nothing."""
    c = chain
    gates = {ns: torch.zeros([BATCH, c.num_node_gates, c.num_ch_gates], device = c.pc.device)
             for ns in c.transitions}

    torch.manual_seed(5)
    gated = juice.queries.sample(c.pc, num_samples = BATCH, sum_external_params = gates).float()
    torch.manual_seed(5)
    plain = juice.queries.sample(c.pc, num_samples = BATCH).float()

    se = ((gated.var(dim = 0) + plain.var(dim = 0)) / BATCH).sqrt().clamp(min = 1e-9)
    z = float(((gated.mean(dim = 0) - plain.mean(dim = 0)) / se).abs().max())
    assert z < 5.0, f"unit gates changed the draw: max |z| = {z:.2f}"
