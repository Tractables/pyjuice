"""
Conditional sampling on a circuit whose input layers carry SOFT EVIDENCE.

This is the shape a diffusion-LM decoder uses -- an external model produces per-position
log-probabilities, the PC is run forward on them, and a conditional draw then reports which latent
each position resolved to. Nothing in the suite covered it: the external-categorical sampling tests
are all UNCONDITIONAL (evidence handed to `sample()` and `num_samples` given), while the conditional
tests all use plain `Categorical` with a `missing_mask`. The combination -- evidence to the FORWARD,
then `sample(conditional = True, _sample_input_ns = False)` -- had no test at all.

Four things the caller depends on, and none of them were pinned:

  * the draw follows the LATENT POSTERIOR the evidence induces, not the prior;
  * the returned frontier decodes through `id - layer._output_ind_range[0]` into `layer.vids` and
    `layer.s_pids`, which is how the caller recovers a position and its emission parameters;
  * a decode loop that interleaves batch sizes -- one batch for the block, another for a
    dependency probe -- does not let one call's state leak into the next;
  * gates supplied per step through a registered GROUP steer the draw, and gates handed to the
    conditional draw itself are ignored (callers pass them, so the no-op has to stay a no-op).

**The oracle.** For the circuit below the exact answer is available in closed form. The root is a
single sum node whose children are products pairing node `j` of variable 0 with node `j` of variable
1, so `P(child j | evidence)` is `theta_j * exp(element_mars_j)` renormalised -- and because the
product pairs like with like, the child index IS the input node index the frontier reports. The draw
can therefore be checked against the posterior the forward pass itself computed, rather than against
another sampler.
"""

import math

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_CATS = 6
NUM_LATENTS = 8         # nodes per variable, and children of the root
NUM_VARS = 2
ALPHA = 1e-5            # family-wise false-failure rate the z-bars are calibrated to


def _z_bar(num_cells):
    return float(math.sqrt(2) * torch.erfinv(torch.tensor(1.0 - ALPHA / num_cells,
                                                          dtype = torch.float64)))


def _build(gated = False, seed = 0):
    """
    Root -> product(i0, i1) with `NUM_LATENTS` children, one per latent.

    Deliberately shaped so the root's chosen child names the input node of every variable: the
    product pairs node `j` of `i0` with node `j` of `i1`, so one draw resolves both positions and
    the frontier is directly comparable to a closed-form posterior over `j`.
    """
    torch.manual_seed(seed)
    i0 = inputs(0, num_node_blocks = NUM_LATENTS, block_size = 1,
                dist = dists.SoftEvidenceCategorical(num_cats = NUM_CATS))
    i1 = inputs(1, num_node_blocks = NUM_LATENTS, block_size = 1,
                dist = dists.SoftEvidenceCategorical(num_cats = NUM_CATS))

    # the implicit pairing: block j of i0 with block j of i1, which is what makes the root's chosen
    # child index equal to the input node index the frontier reports
    pairs = multiply(i0, i1)
    kwargs = {"external_params": BlockScaleSumParams(ch_block_size = 1)} if gated else {}
    root = summate(pairs, edge_ids = torch.stack([torch.zeros(NUM_LATENTS, dtype = torch.long),
                                                  torch.arange(NUM_LATENTS)]),
                   num_node_blocks = 1, block_size = 1, **kwargs)

    root.init_parameters(perturbation = 4.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    return pc, root, (i0, i1)


#: The batch carries only this many DISTINCT evidence patterns, assigned by row index. Every check
#: below is then per pattern rather than over the batch as a whole -- which matters: a batch of
#: independent random evidence averages to nearly the same posterior whatever it is, so a
#: batch-averaged comparison would pass a sampler that shuffled evidence between rows.
NUM_PATTERNS = 2


def _evidence(pc, batch, seed = 1, sharpness = 3.0):
    """Per-position log-probabilities from an external model, plus the observed/masked split.

    Rows cycle through `NUM_PATTERNS` distinct patterns, so each row's posterior is known and rows
    sharing a pattern can be pooled into one clean frequency estimate.
    """
    torch.manual_seed(seed)
    distinct = torch.randn([NUM_PATTERNS, NUM_VARS, NUM_CATS], device = pc.device) * sharpness
    which = torch.arange(batch, device = pc.device) % NUM_PATTERNS
    evidence = torch.log_softmax(distinct[which], dim = -1)

    # the OBSERVED TOKEN is part of the pattern too. Drawing it per row would make every row's
    # posterior different even where the evidence matched, and pooling rows would then compare a
    # draw against one arbitrary row's answer -- which is what made the first version of this file
    # report a 243-sigma "failure" against a sampler that was in fact correct.
    distinct_tokens = torch.randint(0, NUM_CATS, [NUM_PATTERNS, NUM_VARS], device = pc.device)
    tokens = distinct_tokens[which]

    observed = torch.zeros([batch, NUM_VARS], dtype = torch.bool, device = pc.device)
    observed[:, 0] = True                                   # position 0 decoded, position 1 still masked
    return evidence, tokens, observed


def _pattern_of(batch, device):
    return torch.arange(batch, device = device) % NUM_PATTERNS


def _forward(pc, tokens, evidence, observed, gates = None):
    extra = {"sum_external_params": gates} if gates is not None else {}
    return pc(tokens, categorical_evidence_logp = evidence,
              soft_evidence_value_mask = observed, **extra)


def _exact_child_posterior(pc, log_gate = None):
    """
    `P(root child j | evidence)` from the state the forward pass just left behind.

    Ungated this is `theta_j * exp(element_mars_j)` renormalised. GATED it is
    `phi_j * theta_j * exp(element_mars_j)` -- the external parameter multiplies the edge, so
    omitting it would compare a gated draw against the ungated posterior.

    :param log_gate: `[batch, num_ch_gates]` LOG gate for the root's edges, or None when ungated.
    """
    layer = pc.inner_layer_groups[-1][0]
    pids = layer.partitioned_pids[0][0]
    cids = layer.partitioned_cids[0][0]

    theta = pc.params[pids]                                             # [NUM_LATENTS]
    elem = pc.element_mars[cids, :]                                     # [NUM_LATENTS, batch]
    w = theta[:, None].log() + elem
    if log_gate is not None:
        w = w + log_gate.permute(1, 0)
    return torch.softmax(w, dim = 0).permute(1, 0)                      # [batch, NUM_LATENTS]


def _drawn_child(pc, inputs_ns, **kwargs):
    """Which latent each sample resolved to, decoded from the frontier the way a caller does."""
    frontier = juice.queries.sample(pc, conditional = True, _sample_input_ns = False, **kwargs)

    layer = pc.input_layer_group[0]
    layer_sid, _ = layer._output_ind_range
    lo, hi = inputs_ns[0]._output_ind_range                             # variable 0's nodes

    hit = (frontier >= lo) & (frontier < hi)
    assert bool(hit.sum(dim = 0).eq(1).all()), "each sample must select exactly one node of variable 0"

    rows, cols = hit.nonzero(as_tuple = True)
    out = torch.zeros([frontier.size(1)], dtype = torch.long, device = frontier.device)
    out[cols] = frontier[rows, cols] - lo
    return out, frontier


# ------------------------------------------------------------------ the draw follows the evidence

@cuda_only
def test_a_conditional_draw_follows_the_posterior_the_evidence_induces():
    """
    The core claim, against a closed form rather than another sampler.

    A sampler that ignored the evidence entirely -- drawing from the prior `theta` -- would still
    produce a valid-looking frontier covering every variable, so shape checks cannot see this.
    """
    pc, _, ins = _build()
    batch = 30_000

    evidence, tokens, observed = _evidence(pc, batch)
    _forward(pc, tokens, evidence, observed)
    exact = _exact_child_posterior(pc)                                  # [batch, NUM_LATENTS]

    torch.manual_seed(3)
    drawn, _ = _drawn_child(pc, ins)

    # PER PATTERN, not over the batch: rows sharing a pattern share a posterior, and a sampler that
    # applied the right evidence to the wrong rows would still get the batch average right.
    pattern = _pattern_of(batch, pc.device)
    worst, bar = 0.0, _z_bar(NUM_PATTERNS * NUM_LATENTS)
    for p in range(NUM_PATTERNS):
        rows = pattern == p
        n = int(rows.sum())
        reference = exact[rows][0]                                      # identical across these rows
        assert float((exact[rows] - reference).abs().max()) < 1e-5, "rows of one pattern disagree"

        frequency = torch.bincount(drawn[rows], minlength = NUM_LATENTS).float() / n
        se = (reference * (1.0 - reference) / n).sqrt().clamp(min = 1e-9)
        worst = max(worst, float(((frequency - reference) / se).abs().max()))

    assert worst < bar, f"drawn latents differ from the exact posterior: max |z| = {worst:.2f} (bar {bar:.2f})"


@cuda_only
def test_changing_only_the_evidence_changes_the_draw():
    """
    A guard on the test above: it is only meaningful if the posterior actually moves with the
    evidence. The two PATTERNS in one batch must draw differently -- which also proves the check is
    sensitive to which row got which evidence, the thing a batch average cannot see.
    """
    pc, _, ins = _build()
    batch = 30_000

    evidence, tokens, observed = _evidence(pc, batch)
    _forward(pc, tokens, evidence, observed)
    torch.manual_seed(3)
    drawn, _ = _drawn_child(pc, ins)

    pattern = _pattern_of(batch, pc.device)
    freqs = []
    for p in range(NUM_PATTERNS):
        rows = pattern == p
        freqs.append(torch.bincount(drawn[rows], minlength = NUM_LATENTS).float() / int(rows.sum()))

    a, b = freqs[0], freqs[1]
    n = batch // NUM_PATTERNS
    se = ((a * (1 - a) + b * (1 - b)) / n).sqrt().clamp(min = 1e-9)
    z = float(((a - b) / se).abs().max())
    assert z > _z_bar(NUM_LATENTS), \
        f"the two evidence patterns drew the same latents: max |z| = {z:.2f} -- this circuit cannot " \
        f"police the sampler"


# --------------------------------------------------------------- the frontier decodes as documented

@cuda_only
def test_the_frontier_decodes_into_variables_and_emission_parameters():
    """
    Exactly the decode a caller performs: subtract the input layer's start index, then index
    `layer.vids` and `layer.s_pids`. If the frontier ever carried an id from outside that layer, or
    a stale sum-node id, this is where it would go wrong -- silently, since the arithmetic still
    produces an in-range-looking index.
    """
    pc, _, _ = _build()
    batch = 512

    evidence, tokens, observed = _evidence(pc, batch)
    _forward(pc, tokens, evidence, observed)
    frontier = juice.queries.sample(pc, conditional = True, _sample_input_ns = False)

    layer = pc.input_layer_group[0]
    layer_sid, layer_eid = layer._output_ind_range

    live = frontier[frontier >= 0]
    assert live.numel() > 0
    assert bool(((live >= layer_sid) & (live < layer_eid)).all()), \
        "the frontier carries ids outside the input layer -- `id - layer_sid` would mis-index"

    node_ids = live - layer_sid
    assert int(node_ids.max()) < layer.vids.size(0)

    # every sample resolves every variable exactly once
    for j in range(batch):
        column = frontier[:, j]
        ids = column[column >= 0] - layer_sid
        got = layer.vids[ids, 0]
        assert sorted(got.tolist()) == list(range(NUM_VARS)), \
            f"sample {j} resolved variables {sorted(got.tolist())}"

    # the emission parameters a caller gathers must lie inside the parameter tensor
    psids = layer.s_pids[node_ids]
    assert int((psids + NUM_CATS).max()) <= layer.params.numel()


# ------------------------------------------------------------------ a decode loop's state hygiene

@cuda_only
def test_a_dependency_probe_between_steps_does_not_disturb_the_next_draw():
    """
    REGRESSION for the decode loop's real shape. A step runs a forward at the block's batch, draws,
    and then runs a SECOND forward at a different batch to score candidate dependencies. That second
    forward reallocates the circuit's buffers and restages, and it is the pattern that previously
    produced a shape error from deep inside the sampler.

    Each draw must equal what it would have been with no probe in between.
    """
    pc, _, ins = _build()
    batch, probe_batch = 8192, 37

    evidence, tokens, observed = _evidence(pc, batch)

    def step(with_probe):
        if with_probe:                                       # the dependency probe, at another batch
            p_evi, p_tok, p_obs = _evidence(pc, probe_batch, seed = 5)
            _forward(pc, p_tok, p_evi, p_obs)
        _forward(pc, tokens, evidence, observed)
        torch.manual_seed(3)
        drawn, _ = _drawn_child(pc, ins)
        return torch.bincount(drawn, minlength = NUM_LATENTS).float() / batch

    clean = step(with_probe = False)
    probed = step(with_probe = True)

    se = ((clean * (1 - clean) + probed * (1 - probed)) / batch).sqrt().clamp(min = 1e-9)
    z = float(((clean - probed) / se).abs().max())
    assert z < _z_bar(NUM_LATENTS), \
        f"an interleaved probe at another batch changed the draw: max |z| = {z:.2f}"


@cuda_only
def test_many_steps_at_alternating_batch_sizes():
    """The loop run for real: alternating batches, a draw each time. Nothing may accumulate."""
    pc, _, ins = _build()

    for step in range(6):
        batch = (512, 37, 2048, 91, 512, 8)[step]
        evidence, tokens, observed = _evidence(pc, batch, seed = step)
        _forward(pc, tokens, evidence, observed)

        drawn, frontier = _drawn_child(pc, ins)
        assert frontier.size(1) == batch, f"step {step}: drew {frontier.size(1)} of {batch}"
        assert int(drawn.max()) < NUM_LATENTS and int(drawn.min()) >= 0


# --------------------------------------------------------------------------- gates, per step

@cuda_only
def test_a_group_gate_supplied_per_step_steers_each_draw():
    """
    Gates arrive through a registered GROUP and change every step, which is how a router feeds them.
    A one-hot gate makes the answer a function of the gate, so this needs no statistics.
    """
    pc, root, ins = _build(gated = True)
    pc.register_external_params_group("tr", [root])
    batch = 2048

    num_node_gates = root.num_nodes // root.external_params.gate_sizes(root)[0]
    num_ch_gates = root.num_ch_nodes // root.external_params.gate_sizes(root)[1]
    assert num_ch_gates == NUM_LATENTS, "one gate per child, so a one-hot gate pins the draw"

    evidence, tokens, observed = _evidence(pc, batch)

    for target in (0, 3, NUM_LATENTS - 1):
        gate = torch.full([batch, num_node_gates, num_ch_gates], -30.0, device = pc.device)
        gate[:, :, target] = 30.0

        _forward(pc, tokens, evidence, observed, gates = {"tr": gate})
        drawn, _ = _drawn_child(pc, ins)

        assert bool((drawn == target).all()), \
            f"gate pinned child {target}, but {int((drawn != target).sum())} of {batch} went elsewhere"


@cuda_only
def test_gates_handed_to_the_conditional_draw_are_ignored():
    """
    Callers pass their gate dict to BOTH the forward and the draw. The draw's copy is never read --
    it takes the values the forward staged, because those are what `node_mars` was built with.

    Pinned because it is relied upon: a change that started honouring the draw's copy would silently
    give a caller two different gates on the two halves of one step.
    """
    pc, root, ins = _build(gated = True)
    pc.register_external_params_group("tr", [root])
    batch = 4096

    num_node_gates = root.num_nodes // root.external_params.gate_sizes(root)[0]
    num_ch_gates = root.num_ch_nodes // root.external_params.gate_sizes(root)[1]

    def one_hot(target):
        g = torch.full([batch, num_node_gates, num_ch_gates], -30.0, device = pc.device)
        g[:, :, target] = 30.0
        return {"tr": g}

    evidence, tokens, observed = _evidence(pc, batch)
    _forward(pc, tokens, evidence, observed, gates = one_hot(2))

    plain, _ = _drawn_child(pc, ins)
    contradicted, _ = _drawn_child(pc, ins, sum_external_params = one_hot(5))

    assert bool((plain == 2).all())
    assert bool((contradicted == 2).all()), \
        "the draw honoured the gate it was handed instead of the one the forward staged"


@cuda_only
def test_a_gate_naming_the_wrong_nodes_is_refused_not_ignored():
    """The one thing that IS checked about a conditional draw's gate is its key set. A caller who
    passes the wrong group should hear about it rather than get a silently ungated draw."""
    pc, root, ins = _build(gated = True)
    pc.register_external_params_group("tr", [root])
    batch = 256

    evidence, tokens, observed = _evidence(pc, batch)
    _forward(pc, tokens, evidence, observed,
             gates = {"tr": torch.zeros([batch, 1, NUM_LATENTS], device = pc.device)})

    with pytest.raises(AssertionError):
        juice.queries.sample(pc, conditional = True, _sample_input_ns = False,
                             sum_external_params = {"nonexistent": torch.zeros([batch, 1, NUM_LATENTS],
                                                                               device = pc.device)})


@cuda_only
def test_a_gated_conditional_draw_matches_the_GATED_posterior():
    """
    The gated posterior's MAGNITUDES, not just which child wins.

    The other gated tests use one-hot gates at +-30, which reduce the draw to an argmax: they would
    pass a kernel that combined the gate with the parameters wrongly -- `phi^2 * theta`, or `phi`
    with `theta` dropped -- so long as the winner were unchanged. Here the gate is SOFT, so the
    answer depends on the exact product

        P(child j | evidence)  proportional to  phi_j * theta_j * exp(element_mars_j)

    and any mis-weighting shows up as a distribution shift rather than not at all.
    """
    pc, root, ins = _build(gated = True)
    pc.register_external_params_group("tr", [root])
    batch = 30_000

    num_node_gates = root.num_nodes // root.external_params.gate_sizes(root)[0]
    num_ch_gates = root.num_ch_nodes // root.external_params.gate_sizes(root)[1]
    assert num_ch_gates == NUM_LATENTS and num_node_gates == 1

    # one soft gate per pattern, so rows sharing a pattern share a posterior
    torch.manual_seed(11)
    per_pattern = torch.randn([NUM_PATTERNS, num_ch_gates], device = pc.device) * 1.2
    pattern = _pattern_of(batch, pc.device)
    gate = per_pattern[pattern][:, None, :].contiguous()                # [batch, 1, NUM_LATENTS]

    evidence, tokens, observed = _evidence(pc, batch)
    _forward(pc, tokens, evidence, observed, gates = {"tr": gate})

    exact = _exact_child_posterior(pc, log_gate = gate[:, 0, :])
    torch.manual_seed(3)
    drawn, _ = _drawn_child(pc, ins)

    worst, bar = 0.0, _z_bar(NUM_PATTERNS * NUM_LATENTS)
    for p in range(NUM_PATTERNS):
        rows = pattern == p
        n = int(rows.sum())
        reference = exact[rows][0]
        assert float((exact[rows] - reference).abs().max()) < 1e-5, "rows of one pattern disagree"

        frequency = torch.bincount(drawn[rows], minlength = NUM_LATENTS).float() / n
        se = (reference * (1.0 - reference) / n).sqrt().clamp(min = 1e-9)
        worst = max(worst, float(((frequency - reference) / se).abs().max()))

    assert worst < bar, f"gated draw differs from phi*theta*exp(mars): max |z| = {worst:.2f} (bar {bar:.2f})"


@cuda_only
def test_the_soft_gate_is_not_effectively_one_hot():
    """A guard on the test above: a gate sharp enough to pin one child would collapse it back into
    the argmax check it exists to replace."""
    pc, root, ins = _build(gated = True)
    pc.register_external_params_group("tr", [root])
    batch = 4096

    num_ch_gates = root.num_ch_nodes // root.external_params.gate_sizes(root)[1]
    torch.manual_seed(11)
    per_pattern = torch.randn([NUM_PATTERNS, num_ch_gates], device = pc.device) * 1.2
    pattern = _pattern_of(batch, pc.device)
    gate = per_pattern[pattern][:, None, :].contiguous()

    evidence, tokens, observed = _evidence(pc, batch)
    _forward(pc, tokens, evidence, observed, gates = {"tr": gate})
    exact = _exact_child_posterior(pc, log_gate = gate[:, 0, :])

    biggest = float(exact.max(dim = 1).values.max())
    assert biggest < 0.9, \
        f"the gate pins a child with probability {biggest:.3f} -- too sharp to test proportions"
