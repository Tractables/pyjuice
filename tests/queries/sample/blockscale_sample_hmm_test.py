"""
End-to-end conditional sampling from a gated HMM -- the model `BlockScaleSumParams` exists for.

Every transition of a `SEQ_LEN`-step chain carries a per-sample gate, and all of them are supplied as
ONE tensor through a registered group, which is how a router head would actually feed them. That is
the part a two-variable circuit cannot exercise: the gates have to be split across `SEQ_LEN - 1`
DIFFERENT layers at different depths, each layer has to pick up its own slab of the staging buffer,
and the top-down pass has to compose the whole chain.

The reference is an exact dense recursion over the same circuit, written from the NODE-level
parameters (`ns._params`, `ns.edge_ids`) rather than the compiled `nids`/`cids`/`pids`, so it shares
no indexing with the kernel:

    A_t[j,i]  =  phi_t[g(i)] theta_t[j,i] / sum_i'                 effective transition
    V_0       =  E_0,        V_t[j] = sum_i A_t[j,i] V_{t-1}[i] E_t[i]        bottom-up
    Q_{T-1}   =  r V_{T-1} / R,   D_t[i] = sum_j Q_t[j] A_t[j,i] V_{t-1}[i] E_t[i] / V_t[j]

`D_t` is the posterior over the child index chosen at level `t`, which the sampler draws from, and
`D_t @ P(x_t | .)` is then the exact posterior predictive of variable `t`. Note the sampler re-emits
OBSERVED variables from the selected node too, so the same predictive covers every position rather
than just the missing ones.

Emissions are deliberately SOFT here (8 categories over 64 states): sharp emissions would let the
evidence pin the latent state and the gate would stop mattering, which is the opposite of what this
file is for.

**Scope, measured rather than assumed.** Seeding faults into the kernel, this file catches the ones
about composition -- the gate arithmetic, each transition reading its own slice of the group tensor,
and `element_mars` entering the conditional draw -- and is blind to the ones about SHAPE and
NUMERICS, because the chain is a single dense 64x64 block with power-of-two counts and moderate
gates. Node-axis gates, padded edge columns, ragged rows, the two-pass path and the overflow
stabilizer are covered by `blockscale_sample_test.py` instead, whose circuits are built to reach
them. Neither file subsumes the other; run both.

The oracle here is itself checked for discrimination: scoring a batch half against the OTHER half's
gate, against a unit gate, or against a gate whose group axis is rolled by one timestep disagrees by
17-22 sigma, so agreement in the tests below is evidence rather than insensitivity.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

NUM_STATES = 64        # hidden states; also the block size, so the chain is one node block wide
NUM_EMITS = 8          # observation alphabet -- small, so the evidence tilts without pinning
SEQ_LEN = 6            # timesteps
GATE_CBS = 8           # a gate spans 8 children -> 8 gates per transition
Z_BAR = 5.0


def _build(seed = 0):
    torch.manual_seed(seed)
    with juice.set_block_size(NUM_STATES):
        base = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))
        emits, transitions, ns = [base], [], base
        for t in range(1, SEQ_LEN):
            emit = inputs(t, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))
            emits.append(emit)
            ns = summate(multiply(ns, emit), num_node_blocks = 1,
                         external_params = BlockScaleSumParams(ch_block_size = GATE_CBS))
            transitions.append(ns)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    pc.register_external_params_group("transitions", transitions)

    return pc, root, emits, transitions


def _exact_predictive(root, emits, transitions, log_phi, obs, missing):
    """
    Exact posterior predictive of every variable, for ONE sample's gate and evidence.

    `log_phi`: [SEQ_LEN - 1, n_gates], `obs`: [SEQ_LEN], `missing`: [SEQ_LEN] bool.
    Built from node-level parameters only, in float64.
    """
    K, T = NUM_STATES, SEQ_LEN
    emit_p = [e._params.reshape(K, NUM_EMITS).double() for e in emits]

    ev = [torch.ones([K], dtype = torch.float64) if missing[t] else emit_p[t][:, obs[t]]
          for t in range(T)]

    gate_of = torch.arange(K) // GATE_CBS               # one child block, so the gate is i // GATE_CBS
    A = []
    for t in range(1, T):
        theta = transitions[t - 1]._params.reshape(K, K).double()          # [j, i]
        w = theta * torch.exp(log_phi[t - 1].double())[gate_of][None, :]
        A.append(w / w.sum(dim = 1, keepdim = True))

    V, W = [ev[0]], [None]
    for t in range(1, T):
        W.append(V[t - 1] * ev[t])
        V.append(A[t - 1] @ W[t])

    r = root._params.reshape(K).double()
    r = r / r.sum()
    Q = r * V[T - 1] / (r * V[T - 1]).sum()

    D = [None] * T
    for t in range(T - 1, 0, -1):
        D[t] = (Q[:, None] * A[t - 1] * W[t][None, :] / V[t][:, None]).sum(dim = 0)
        Q = D[t]

    # Variable 0 is emitted by the base input node, whose index is the child chosen at level 1
    return [(D[1] if t == 0 else D[t]) @ emit_p[t] for t in range(T)]


def _observe(pc, obs, missing, log_phi):
    x = obs[None, :].expand(log_phi.size(0), -1).contiguous().to(pc.device)
    pc(x, missing_mask = missing.to(pc.device), sum_external_params = {"transitions": log_phi})


def _max_z(drawn, expect, N):
    """Worst deviation of the drawn category frequencies from `expect`, in standard errors."""
    freq = torch.bincount(drawn, minlength = NUM_EMITS).double() / N
    e = expect.to(freq.device)
    se = (e * (1 - e) / N).sqrt().clamp(min = 1e-12)
    return float(((freq - e) / se).abs().max())


def _states_at(pc, emits, t, **kwargs):
    """The hidden index chosen at level `t`, read exactly off the frontier."""
    frontier = juice.queries.sample(pc, _sample_input_ns = False, **kwargs)
    lo, hi = emits[t]._output_ind_range

    hit = (frontier >= lo) & (frontier < hi)
    assert bool(hit.sum(dim = 0).eq(1).all())

    rows, cols = hit.nonzero(as_tuple = True)
    out = torch.zeros([frontier.size(1)], dtype = torch.long, device = frontier.device)
    out[cols] = frontier[rows, cols] - lo
    return out


@cuda_only
def test_conditional_samples_match_the_exact_posterior_predictive():
    """
    The whole feature end to end: a gated forward on partial evidence, then a conditional draw, per
    variable, against the exact predictive.

    The batch carries TWO DIFFERENT gates, one per half, and each half is checked against its own
    reference. A gate that leaked across samples -- or a layer reading the wrong slab of the staging
    buffer -- would agree with neither.
    """
    pc, root, emits, transitions = _build(seed = 0)
    n_gates = NUM_STATES // GATE_CBS
    half = 50_000

    torch.manual_seed(100)
    obs = torch.randint(0, NUM_EMITS, [SEQ_LEN])
    missing = torch.tensor([False, True, False, True, True, False])       # observe t = 0, 2, 5

    phis = [torch.randn([SEQ_LEN - 1, n_gates]) * 1.5,
            torch.randn([SEQ_LEN - 1, n_gates]) * 1.5]
    log_phi = torch.cat([p[None].expand(half, -1, -1) for p in phis], dim = 0).contiguous().to(pc.device)

    _observe(pc, obs, missing, log_phi)
    samples = juice.queries.sample(pc, conditional = True)

    for h, phi in enumerate(phis):
        expect = _exact_predictive(root, emits, transitions, phi, obs, missing)
        drawn = samples[h * half:(h + 1) * half]
        for t in range(SEQ_LEN):
            z = _max_z(drawn[:, t].long(), expect[t], half)
            assert z < Z_BAR, f"gate {h}, variable {t}: max |z| = {z:.2f}"


@cuda_only
def test_a_one_hot_gate_routes_the_trajectory_at_its_own_timestep():
    """
    Each transition must read ITS OWN slice of the group tensor. A one-hot gate at one timestep
    confines the state chosen there and leaves the others free, so getting the slice wrong -- an
    off-by-one along the group axis, say -- moves the constraint to the neighbouring timestep.
    """
    pc, root, emits, transitions = _build(seed = 1)
    n_gates = NUM_STATES // GATE_CBS
    B = 512

    torch.manual_seed(101)
    obs = torch.randint(0, NUM_EMITS, [SEQ_LEN])
    missing = torch.zeros([SEQ_LEN], dtype = torch.bool)
    missing[1::2] = True

    for level in range(1, SEQ_LEN):
        hot = torch.arange(B, device = pc.device) % n_gates          # a different gate per SAMPLE
        log_phi = torch.zeros([B, SEQ_LEN - 1, n_gates], device = pc.device)
        log_phi[:, level - 1, :] = -40.0
        log_phi[torch.arange(B, device = pc.device), level - 1, hot] = 40.0

        _observe(pc, obs, missing, log_phi)
        states = _states_at(pc, emits, level, conditional = True)

        lo, hi = hot * GATE_CBS, (hot + 1) * GATE_CBS
        assert bool(((states >= lo) & (states < hi)).all()), \
            f"level {level}: hot={hot[:6].tolist()} states={states[:6].tolist()}"


@cuda_only
def test_unit_gates_reproduce_the_plain_sampler():
    """A chain of unit gates leaves every transition's effective parameters unchanged."""
    pc, root, emits, transitions = _build(seed = 2)
    n_gates = NUM_STATES // GATE_CBS
    N = 60_000

    torch.manual_seed(102)
    obs = torch.randint(0, NUM_EMITS, [SEQ_LEN])
    missing = torch.tensor([True, False, True, False, True, False])

    _observe(pc, obs, missing, torch.zeros([N, SEQ_LEN - 1, n_gates], device = pc.device))
    gated = juice.queries.sample(pc, conditional = True)

    x = obs[None, :].expand(N, -1).contiguous().to(pc.device)
    pc(x, missing_mask = missing.to(pc.device))
    plain = juice.queries.sample(pc, conditional = True)

    for t in range(SEQ_LEN):
        a = torch.bincount(gated[:, t].long(), minlength = NUM_EMITS).double() / N
        b = torch.bincount(plain[:, t].long(), minlength = NUM_EMITS).double() / N
        se = ((a * (1 - a) + b * (1 - b)) / N).sqrt().clamp(min = 1e-12)
        z = float(((a - b) / se).abs().max())
        assert z < Z_BAR, f"variable {t}: max |z| = {z:.2f}"


@cuda_only
def test_unconditional_gated_hmm_sampling():
    """The same chain with no evidence at all: gates still apply, and every draw is a valid symbol."""
    pc, root, emits, transitions = _build(seed = 3)
    n_gates = NUM_STATES // GATE_CBS
    N = 20_000

    torch.manual_seed(103)
    phi = torch.randn([SEQ_LEN - 1, n_gates]) * 1.5
    log_phi = phi[None].expand(N, -1, -1).contiguous().to(pc.device)

    samples = juice.queries.sample(pc, num_samples = N,
                                   sum_external_params = {"transitions": log_phi})
    assert samples.shape == (N, SEQ_LEN)
    assert bool(((samples >= 0) & (samples < NUM_EMITS)).all())

    missing = torch.ones([SEQ_LEN], dtype = torch.bool)
    expect = _exact_predictive(root, emits, transitions, phi,
                               torch.zeros([SEQ_LEN], dtype = torch.long), missing)
    for t in range(SEQ_LEN):
        z = _max_z(samples[:, t].long(), expect[t], N)
        assert z < Z_BAR, f"variable {t}: max |z| = {z:.2f}"
