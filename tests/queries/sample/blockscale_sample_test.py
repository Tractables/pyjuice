"""
Ancestral sampling from a `BlockScaleSumParams` sum layer, conditional and unconditional.

**How a draw is made observable.** Every test builds the same shape: two variables, a gated sum node
over `multiply(i0, i1)`, and a root. The drawn child is read off the sampler's FRONTIER --
`_sample_input_ns = False` returns the selected input node ids rather than emissions -- and the
product layer pairs node `j` of one variable with node `j` of the other, so the id names the child
element the gated node chose. The root is usually pinned to one element, which fixes WHICH node of
the gated block is under test and removes the need to marginalize over that choice.

Reading the frontier rather than a near-deterministic emission is not fussiness: an input node with
`p = 1 - 1e-6` on its own index reports the wrong child about once in a million draws, which over
100k samples is a ~10%-per-run flaky failure. The frontier is exact.

**Three tiers of check**, deliberately failing in different ways:

1. *deterministic* -- one-hot gates make the draw a function of the gate, so gate indexing is checked
   per sample with no statistics at all;
2. *distributional* -- drawn frequencies against `P(c | n, b)` computed in torch from the compiled
   tables and the caller's gate grid, judged in units of the standard error rather than against an
   eyeballed tolerance;
3. *invariants* -- a unit gate must reproduce the plain sampler, and adding a constant to every log
   gate must be an exact no-op, since it cancels in the normalizer.

Tier 3 is not decoration. The kernel accumulates its normalizer against a running max, and the gates
these tests would otherwise use (`+-40`) do NOT overflow `exp` in fp32 -- removing the stabilization
entirely passed every other check here. `test_one_hot_gates_survive_exp_overflow` and
`test_a_constant_gate_shift_is_a_no_op` exist because mutation testing showed the rest were blind
to it. `test_a_padded_edge_axis_does_not_read_past_the_gate_table` has the same origin: every other
shape here has a power-of-two child-block count, where the gate column can never run off the table.

**Known gaps**, deliberate rather than overlooked. Mutating the kernel catches 6 of 9 seeded faults.
The three it does not are all defensive guards whose removal provably cannot change an answer, so no
output-based test can detect them:

  * the `offs_gcol < gate_stride` bound, which stops a padded edge column reading past the gate
    table. The lane it protects has the dummy parameter, and `params[0:block_size]` is exactly zero,
    so its weight is zero whatever gate it reads -- the bound buys MEMORY SAFETY, not correctness;
  * the `mask_nids` term in the `nids` scan, which needs a sum layer whose node ids are below
    `block_size` -- impossible while the dummy and input nodes occupy the start of `node_mars`. The
    same guard in the shared-parameter PRODUCT kernel is very much load-bearing, since element ids
    do start low; it is there because omitting it corrupted the frontier outright;
  * the `chids` fallback, which needs `r == sum` in fp32 after the two passes accumulate in
    different orders.

Say so rather than implying coverage. An earlier version of this file appeared to catch the first of
them, but that turned out to depend on what the allocator had left past the table -- it passed or
failed with the test ORDERING, not with the kernel.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams, LowRankSumParams
from pyjuice.layer.external_sum_layer import ExternalParamsSumLayer


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

#: Frequency checks are judged in standard errors. 5 sigma over a few hundred children is ~1 false
#: failure in 1e5 runs, and the mutants these tests exist to catch miss by 20-100 sigma.
Z_BAR = 5.0


# --------------------------------------------------------------------------------- construction

def _build(block_size = 4, gate_bs = None, gate_cbs = 2, n_ch_blocks = 4, n_node_blocks = 1,
           edge_ids = None, seed = 0):
    """A gated sum node whose drawn child is readable off the sampler's frontier."""
    torch.manual_seed(seed)
    K = n_ch_blocks * block_size
    with juice.set_block_size(block_size):
        i0 = inputs(0, num_node_blocks = n_ch_blocks, dist = dists.Categorical(num_cats = K))
        i1 = inputs(1, num_node_blocks = n_ch_blocks, dist = dists.Categorical(num_cats = K))
        ns = summate(multiply(i0, i1), num_node_blocks = n_node_blocks, edge_ids = edge_ids,
                     external_params = BlockScaleSumParams(block_size = gate_bs,
                                                           ch_block_size = gate_cbs))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 2.0)

    # Evidence that TILTS a conditional draw without determining it, so the gate still matters.
    # Variable 1 is the one that is always missing, so its parameters never enter anything.
    p = torch.rand([K, K]) + 0.05
    i0._params = (p / p.sum(1, keepdim = True)).flatten().clone()

    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    return pc, ns, i0, K


def _drawn(pc, ns, **kwargs):
    """
    The child element each sample's gated node chose, read EXACTLY off the frontier.

    `_sample_input_ns = False` stops the pass before the input nodes emit and hands back the selected
    input node ids instead, so the answer does not depend on any input distribution being sharp.
    """
    frontier = juice.queries.sample(pc, _sample_input_ns = False, **kwargs)
    lo, hi = ns.chs[0].chs[1]._output_ind_range                # input nodes of variable 1

    hit = (frontier >= lo) & (frontier < hi)
    assert bool(hit.sum(dim = 0).eq(1).all()), "each sample should select exactly one node of var 1"

    rows, cols = hit.nonzero(as_tuple = True)
    out = torch.zeros([frontier.size(1)], dtype = torch.long, device = frontier.device)
    out[cols] = frontier[rows, cols] - lo
    return out


def _pin_root(pc, ele_idx):
    """Make the root take element `ele_idx` with probability 1, fixing the gated node under test."""
    pids = pc.inner_layer_groups[-1][0].partitioned_pids[0][0]
    pc.params[pids] = 1e-30
    pc.params[pids[ele_idx]] = 1.0


def _gated_layer(pc):
    return [l for g in pc.inner_layer_groups for l in g if isinstance(l, ExternalParamsSumLayer)][0]


def _gate_shape(ns):
    return ns.num_nodes // ns.external_params.gate_sizes(ns)[0], \
           ns.num_ch_nodes // ns.external_params.gate_sizes(ns)[1]


# --------------------------------------------------------------------------------- the oracle

def _exact_for_node(pc, ns, node_id, log_phi, log_ev = None):
    """
    `P(edge slot | node, sample)` for one node, and the input-node index each slot reveals.

    Built from the compiled tables (`pids` for `theta`, `par_ptr` / `eblk_ids` for the topology) and
    the CALLER's gate grid -- not from the kernel's addressing -- so it is an independent statement
    of what the draw should be.
    """
    layer = _gated_layer(pc)
    info = layer.external_node_infos[0]
    bs, cbs = ns.block_size, ns.ch_block_size
    gate_bs, gate_cbs = ns.external_params.gate_sizes(ns)
    n_child_gates = cbs // gate_cbs

    row = None
    for pid in range(layer.num_fw_partitions):
        nids = layer.partitioned_nids[pid]
        hit = ((nids <= node_id) & (node_id < nids + bs)).nonzero().flatten()
        if hit.numel() > 0:
            row, m, part = int(hit[0]), int(node_id - nids[hit[0]]), pid
            break
    assert row is not None, f"node {node_id} is not in this layer"

    nb = (int(layer.partitioned_nids[part][row]) - info.nid_start) // bs
    n_eblks = int(info.par_ptr[nb + 1] - info.par_ptr[nb])
    K = n_eblks * cbs

    theta = pc.params[layer.partitioned_pids[part][row][:K] + m]

    gate_of = torch.empty([K], dtype = torch.long)
    ele_of = torch.empty([K], dtype = torch.long)
    for k in range(K):
        eblk = int(info.eblk_ids[int(info.par_ptr[nb]) + k // cbs])
        cb = int(ns.edge_ids[1, eblk])
        gate_of[k] = cb * n_child_gates + (k % cbs) // gate_cbs
        ele_of[k] = cb * cbs + (k % cbs)

    node_gate = nb * (bs // gate_bs) + m // gate_bs
    a = log_phi[:, node_gate, :][:, gate_of.to(log_phi.device)]
    if log_ev is not None:
        a = a + log_ev.to(log_phi.device)[ele_of.to(log_phi.device)][None, :]

    w = theta[None] * torch.exp(a - a.amax(dim = 1, keepdim = True))
    return w / w.sum(dim = 1, keepdim = True), ele_of.to(theta.device)


def _max_z(drawn, ele_of, expect, N):
    """Worst per-child deviation of the drawn frequencies from `expect`, in standard errors."""
    K = expect.numel()
    dev = drawn.device
    slot_of = torch.full([int(ele_of.max()) + 1], -1, dtype = torch.long, device = dev)
    slot_of[ele_of] = torch.arange(K, device = dev)

    assert int(drawn.max()) < slot_of.numel() and bool((slot_of[drawn] >= 0).all()), \
        "a draw landed outside the children of the node under test"

    freq = torch.bincount(slot_of[drawn], minlength = K).float() / N
    se = (expect * (1 - expect) / N).sqrt().clamp(min = 1e-12)
    return float(((freq - expect) / se).abs().max())


def _two_sample_z(a, b, K, N):
    """Worst deviation between two sets of draws that should follow the SAME distribution."""
    ca = torch.bincount(a, minlength = K).float() / N
    cb = torch.bincount(b, minlength = K).float() / N
    se = ((ca * (1 - ca) + cb * (1 - cb)) / N).sqrt().clamp(min = 1e-12)
    return float(((ca - cb) / se).abs().max())


def _one_hot_gates(B, Nk, Ck, hot, node_gate = None, scale = 40.0, device = "cuda:0"):
    """`log phi` that is `+scale` on gate `hot` and `-scale` elsewhere, per sample."""
    g = torch.full([B, Nk, Ck], -scale, device = device)
    if node_gate is None:
        g[torch.arange(B, device = device), :, hot] = scale
    else:
        g[:, node_gate, hot] = scale
    return g


def _observe(pc, ns, log_phi, K, v, N):
    """One gated forward with variable 0 observed at `v` and variable 1 missing."""
    dev = torch.device("cuda:0")
    x = torch.full([N, 2], v, dtype = torch.long, device = dev)
    missing = torch.tensor([False, True], device = dev)
    pc(x, missing_mask = missing, sum_external_params = {ns: log_phi})


# --------------------------------------------------------------------------------- unconditional

@cuda_only
def test_one_hot_gates_pin_the_drawn_child():
    """A gate that is on for exactly one child gate makes the draw deterministic. No statistics."""
    pc, ns, _, K = _build()
    Nk, Ck = _gate_shape(ns)
    B = 8 * Ck
    hot = torch.arange(B, device = pc.device) % Ck                # a DIFFERENT gate per sample

    drawn = _drawn(pc, ns, num_samples = B,
                   sum_external_params = {ns: _one_hot_gates(B, Nk, Ck, hot)})
    assert bool(((drawn >= hot * 2) & (drawn < hot * 2 + 2)).all()), \
        f"hot={hot[:8].tolist()} drawn={drawn[:8].tolist()}"


@cuda_only
def test_one_hot_gates_survive_exp_overflow():
    """`exp(+-120)` overflows fp32; only the running-max shift keeps the draw well defined.

    REGRESSION for a real blind spot: with gates at `+-40` -- which do not overflow -- removing the
    stabilization from the kernel passed every other test in this file."""
    pc, ns, _, K = _build()
    Nk, Ck = _gate_shape(ns)
    B = 8 * Ck
    hot = torch.arange(B, device = pc.device) % Ck

    drawn = _drawn(pc, ns, num_samples = B,
                   sum_external_params = {ns: _one_hot_gates(B, Nk, Ck, hot, scale = 120.0)})
    assert bool(((drawn >= hot * 2) & (drawn < hot * 2 + 2)).all()), \
        f"hot={hot[:8].tolist()} drawn={drawn[:8].tolist()}"


@cuda_only
def test_a_constant_gate_shift_is_a_no_op():
    """`phi -> phi + c` cancels in the normalizer, so the distribution must not move at all."""
    pc, ns, _, K = _build()
    Nk, Ck = _gate_shape(ns)
    N = 100_000

    base = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
    a = _drawn(pc, ns, num_samples = N, sum_external_params = {ns: base.expand(N, -1, -1).contiguous()})
    b = _drawn(pc, ns, num_samples = N,
               sum_external_params = {ns: (base + 300.0).expand(N, -1, -1).contiguous()})

    assert _two_sample_z(a, b, K, N) < Z_BAR


@cuda_only
def test_unit_gates_reproduce_the_plain_sampler():
    """`log phi = 0` leaves the effective parameters unchanged, gate or no gate."""
    pc, ns, _, K = _build()
    Nk, Ck = _gate_shape(ns)
    N = 100_000

    gated = _drawn(pc, ns, num_samples = N,
                   sum_external_params = {ns: torch.zeros([N, Nk, Ck], device = pc.device)})
    plain = _drawn(pc, ns, num_samples = N)

    assert _two_sample_z(gated, plain, K, N) < Z_BAR


@cuda_only
def test_frequencies_match_the_exact_gated_distribution():
    pc, ns, _, K = _build(seed = 1)
    _pin_root(pc, 2)
    Nk, Ck = _gate_shape(ns)
    N = 100_000

    phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
    nid = _gated_layer(pc).external_node_infos[0].nid_start + 2
    expect, ele_of = _exact_for_node(pc, ns, nid, phi)

    drawn = _drawn(pc, ns, num_samples = N,
                   sum_external_params = {ns: phi.expand(N, -1, -1).contiguous()})
    assert _max_z(drawn, ele_of, expect.squeeze(0), N) < Z_BAR


@cuda_only
def test_node_axis_gates_index_the_right_row():
    """A gate FINER than the node block: each node gate must read its own row of the grid."""
    pc, ns, _, K = _build(block_size = 8, gate_bs = 4, gate_cbs = 2, n_ch_blocks = 2)
    Nk, Ck = _gate_shape(ns)
    assert Nk == 2
    B = 256

    for pinned in range(ns.block_size):
        _pin_root(pc, pinned)
        ng = pinned // 4
        hot = (3 * ng + 1) % Ck                          # a different hot gate per node gate
        drawn = _drawn(pc, ns, num_samples = B,
                       sum_external_params = {ns: _one_hot_gates(B, Nk, Ck, hot, node_gate = ng)})
        assert bool(((drawn >= hot * 2) & (drawn < hot * 2 + 2)).all()), \
            f"node {pinned} (node gate {ng}, hot {hot}): drawn {drawn[:8].tolist()}"


@cuda_only
def test_a_multi_tile_edge_axis_matches_the_exact_distribution():
    """`K = 1024 > BLOCK_K` takes the TWO-PASS path, where the walk runs against a sum the first
    pass computed -- the one place the two could disagree."""
    pc, ns, _, K = _build(block_size = 4, gate_cbs = 2, n_ch_blocks = 256, seed = 4)
    _pin_root(pc, 1)
    Nk, Ck = _gate_shape(ns)
    N = 200_000

    phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
    nid = int(_gated_layer(pc).partitioned_nids[0][0]) + 1
    expect, ele_of = _exact_for_node(pc, ns, nid, phi)

    drawn = _drawn(pc, ns, num_samples = N,
                   sum_external_params = {ns: phi.expand(N, -1, -1).contiguous()})
    assert _max_z(drawn, ele_of, expect.squeeze(0), N) < Z_BAR

    # The two passes must also agree on the SHIFT, which a frequency check alone is weak against
    M = 50_000
    a = _drawn(pc, ns, num_samples = M, sum_external_params = {ns: phi.expand(M, -1, -1).contiguous()})
    b = _drawn(pc, ns, num_samples = M,
               sum_external_params = {ns: (phi + 300.0).expand(M, -1, -1).contiguous()})
    assert _two_sample_z(a, b, K, M) < Z_BAR


@cuda_only
def test_a_ragged_topology_matches_the_exact_distribution():
    """Node blocks with 4 / 2 / 3 edge blocks: the short rows carry `-1` gate slots, whose children
    must take exactly zero probability."""
    edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2],
                             [0, 1, 2, 3, 1, 3, 0, 2, 3]], dtype = torch.long)
    pc, ns, _, K = _build(block_size = 4, gate_cbs = 2, n_ch_blocks = 4, n_node_blocks = 3,
                          edge_ids = edge_ids, seed = 5)
    Nk, Ck = _gate_shape(ns)
    N = 100_000

    for pinned in (1, 5, 9):                              # one node from each node block
        _pin_root(pc, pinned)
        phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
        nid = _gated_layer(pc).external_node_infos[0].nid_start + pinned
        expect, ele_of = _exact_for_node(pc, ns, nid, phi)

        drawn = _drawn(pc, ns, num_samples = N,
                       sum_external_params = {ns: phi.expand(N, -1, -1).contiguous()})
        assert _max_z(drawn, ele_of, expect.squeeze(0), N) < Z_BAR


@cuda_only
@pytest.mark.parametrize("n_ch_blocks", [3, 5])
def test_a_padded_edge_axis_does_not_read_past_the_gate_table(n_ch_blocks):
    """
    A child-block count that is not a power of two pads `num_edges` (3 blocks -> 12 edges -> 16), so
    the last edge tile derives gate COLUMNS beyond the table's width.

    This is the only shape in this file that exercises those columns at all -- every other one has a
    power-of-two child count, where `num_edges` is exactly `n_eblks * NODE_CBS` and the column can
    never run past the table.

    :note: it does NOT prove the bound in `_gate_weights` is load-bearing, and measurement says it is
           not, for the OUTPUT: a padded slot's `pids` is the dummy parameter and `params[0:block_size]`
           is exactly zero, so such a lane has weight zero and is unselectable no matter which gate it
           reads. What the bound buys is memory safety -- on the last row an unbounded column reads
           past the end of the `gate` tensor. Removing it is invisible to every assertion here, and
           two node blocks rather than one does not change that; it was tried. Recorded so nobody
           re-derives it.
    """
    pc, ns, _, K = _build(block_size = 4, gate_cbs = 2, n_ch_blocks = n_ch_blocks,
                          n_node_blocks = 2, seed = 15)
    _pin_root(pc, 1)
    Nk, Ck = _gate_shape(ns)
    N = 100_000

    phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
    nid = int(_gated_layer(pc).partitioned_nids[0][0]) + 1
    expect, ele_of = _exact_for_node(pc, ns, nid, phi)

    drawn = _drawn(pc, ns, num_samples = N,
                   sum_external_params = {ns: phi.expand(N, -1, -1).contiguous()})
    assert _max_z(drawn, ele_of, expect.squeeze(0), N) < Z_BAR


# --------------------------------------------------------------------------------- conditional

@cuda_only
def test_conditional_frequencies_match_the_exact_posterior():
    """`theta * phi * p(x0 | child)`, with variable 1 marginalized. Soft input parameters on
    variable 0 so the evidence tilts the posterior without swamping the gate."""
    pc, ns, i0, K = _build(gate_cbs = 2, seed = 3)
    _pin_root(pc, 2)
    Nk, Ck = _gate_shape(ns)
    N, v = 100_000, 5

    phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
    log_ev = torch.log(i0._params.to(pc.device).view(K, K)[:, v])
    nid = _gated_layer(pc).external_node_infos[0].nid_start + 2
    expect, ele_of = _exact_for_node(pc, ns, nid, phi, log_ev = log_ev)

    _observe(pc, ns, phi.expand(N, -1, -1).contiguous(), K, v, N)
    assert _max_z(_drawn(pc, ns, conditional = True), ele_of, expect.squeeze(0), N) < Z_BAR


@cuda_only
def test_conditional_one_hot_gates_pin_the_drawn_child():
    """The gate is per SAMPLE in a conditional draw too: with `+-40` against evidence worth ~17 nats
    the gate decides, so each sample's child must sit in that sample's own hot gate."""
    pc, ns, _, K = _build(seed = 6)
    Nk, Ck = _gate_shape(ns)
    B, v = 8 * Ck, 3
    hot = torch.arange(B, device = pc.device) % Ck

    log_phi = _one_hot_gates(B, Nk, Ck, hot)
    _observe(pc, ns, log_phi, K, v, B)
    drawn = _drawn(pc, ns, conditional = True)
    assert bool(((drawn >= hot * 2) & (drawn < hot * 2 + 2)).all()), \
        f"hot={hot[:8].tolist()} drawn={drawn[:8].tolist()}"


@cuda_only
def test_conditional_one_hot_gates_survive_exp_overflow():
    pc, ns, _, K = _build(seed = 7)
    Nk, Ck = _gate_shape(ns)
    B, v = 8 * Ck, 3
    hot = torch.arange(B, device = pc.device) % Ck

    log_phi = _one_hot_gates(B, Nk, Ck, hot, scale = 120.0)
    _observe(pc, ns, log_phi, K, v, B)
    drawn = _drawn(pc, ns, conditional = True)
    assert bool(((drawn >= hot * 2) & (drawn < hot * 2 + 2)).all()), \
        f"hot={hot[:8].tolist()} drawn={drawn[:8].tolist()}"


@cuda_only
def test_conditional_unit_gates_reproduce_the_plain_sampler():
    """With `log phi = 0` the gated conditional draw must match an ungated forward's."""
    pc, ns, _, K = _build(seed = 8)
    Nk, Ck = _gate_shape(ns)
    N, v = 100_000, 4

    _observe(pc, ns, torch.zeros([N, Nk, Ck], device = pc.device), K, v, N)
    gated = _drawn(pc, ns, conditional = True)

    x = torch.full([N, 2], v, dtype = torch.long, device = pc.device)
    pc(x, missing_mask = torch.tensor([False, True], device = pc.device))
    plain = _drawn(pc, ns, conditional = True)

    assert _two_sample_z(gated, plain, K, N) < Z_BAR


@cuda_only
def test_conditional_node_axis_gates_index_the_right_row():
    pc, ns, _, K = _build(block_size = 8, gate_bs = 4, gate_cbs = 2, n_ch_blocks = 2, seed = 9)
    Nk, Ck = _gate_shape(ns)
    B, v = 256, 2

    for pinned in range(ns.block_size):
        _pin_root(pc, pinned)
        ng = pinned // 4
        hot = (3 * ng + 1) % Ck
        log_phi = _one_hot_gates(B, Nk, Ck, hot, node_gate = ng)
        _observe(pc, ns, log_phi, K, v, B)
        drawn = _drawn(pc, ns, conditional = True)
        assert bool(((drawn >= hot * 2) & (drawn < hot * 2 + 2)).all()), \
            f"node {pinned} (node gate {ng}, hot {hot}): drawn {drawn[:8].tolist()}"


@cuda_only
def test_conditional_multi_tile_edge_axis():
    pc, ns, i0, K = _build(block_size = 4, gate_cbs = 2, n_ch_blocks = 256, seed = 10)
    _pin_root(pc, 1)
    Nk, Ck = _gate_shape(ns)
    N, v = 200_000, 17

    phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
    log_ev = torch.log(i0._params.to(pc.device).view(K, K)[:, v])
    nid = int(_gated_layer(pc).partitioned_nids[0][0]) + 1
    expect, ele_of = _exact_for_node(pc, ns, nid, phi, log_ev = log_ev)

    _observe(pc, ns, phi.expand(N, -1, -1).contiguous(), K, v, N)
    assert _max_z(_drawn(pc, ns, conditional = True), ele_of, expect.squeeze(0), N) < Z_BAR


@cuda_only
def test_conditional_ragged_topology():
    edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2],
                             [0, 1, 2, 3, 1, 3, 0, 2, 3]], dtype = torch.long)
    pc, ns, i0, K = _build(block_size = 4, gate_cbs = 2, n_ch_blocks = 4, n_node_blocks = 3,
                           edge_ids = edge_ids, seed = 11)
    Nk, Ck = _gate_shape(ns)
    N, v = 100_000, 6

    for pinned in (1, 5, 9):
        _pin_root(pc, pinned)
        phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
        log_ev = torch.log(i0._params.to(pc.device).view(K, K)[:, v])
        nid = _gated_layer(pc).external_node_infos[0].nid_start + pinned
        expect, ele_of = _exact_for_node(pc, ns, nid, phi, log_ev = log_ev)

        _observe(pc, ns, phi.expand(N, -1, -1).contiguous(), K, v, N)
        assert _max_z(_drawn(pc, ns, conditional = True), ele_of, expect.squeeze(0), N) < Z_BAR


@cuda_only
def test_conditional_padded_edge_axis():
    """The padded-column case again, conditionally: `element_mars` is read on the same lanes."""
    pc, ns, i0, K = _build(block_size = 4, gate_cbs = 2, n_ch_blocks = 3, seed = 16)
    _pin_root(pc, 1)
    Nk, Ck = _gate_shape(ns)
    N, v = 100_000, 4

    phi = torch.randn([1, Nk, Ck], device = pc.device) * 1.5
    log_ev = torch.log(i0._params.to(pc.device).view(K, K)[:, v])
    nid = int(_gated_layer(pc).partitioned_nids[0][0]) + 1
    expect, ele_of = _exact_for_node(pc, ns, nid, phi, log_ev = log_ev)

    _observe(pc, ns, phi.expand(N, -1, -1).contiguous(), K, v, N)
    assert _max_z(_drawn(pc, ns, conditional = True), ele_of, expect.squeeze(0), N) < Z_BAR


# --------------------------------------------------------------------------------- the contract

@cuda_only
def test_a_gated_pc_sampled_without_gates_uses_the_shared_parameters():
    """Supplying nothing makes a gated layer a plain sum layer, which is what an ungated FORWARD
    computes for it too. Consistency between the two is the point."""
    pc, ns, _, K = _build(seed = 12)
    s = juice.queries.sample(pc, num_samples = 64)
    assert ((s >= 0) & (s < K)).all()


@cuda_only
def test_an_ungated_forward_makes_the_following_draw_ungated():
    """
    A conditional draw uses whatever the forward staged, so an ungated forward -- which runs the
    gated layers as plain sum layers -- must leave nothing staged, and the draw that follows is from
    the shared parameters. Consistent with the `node_mars` it conditions on, which is the point.

    This used to raise instead: the staged tensors were never dropped, so the draw found an earlier
    forward's gate at an unrelated batch size. Read at a matching batch they would have been in
    bounds and plausible, which is why the mismatch is now made unreachable rather than merely
    detected. See `tests/model/external_params/external_params_staging_test.py`.
    """
    pc, ns, _, K = _build(seed = 13)
    Nk, Ck = _gate_shape(ns)
    dev = pc.device

    pc(torch.randint(0, K, [8, 2], device = dev),
       sum_external_params = {ns: torch.randn([8, Nk, Ck], device = dev)})
    assert pc._staged_external_params is not None

    # A LATER, ungated forward at a different batch size
    pc(torch.randint(0, K, [32, 2], device = dev))
    assert pc._staged_external_params is None

    samples = juice.queries.sample(pc, conditional = True)
    assert samples.shape == (32, pc.num_vars)


@cuda_only
def test_conditional_sampling_rejects_a_different_set_of_nodes():
    pc, ns, _, K = _build(seed = 14)
    Nk, Ck = _gate_shape(ns)
    dev = pc.device

    x = torch.randint(0, K, [16, 2], device = dev)
    pc(x, sum_external_params = {ns: torch.randn([16, Nk, Ck], device = dev)})

    with pytest.raises(AssertionError, match = "different set of nodes"):
        juice.queries.sample(pc, conditional = True, sum_external_params = {})


@cuda_only
def test_partial_supply_is_refused():
    """Two gated nodes on one layer, one of them supplied: the kernels run over every row of the
    layer, so the unsupplied node has no gate to read."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    K, gcbs = 64, 8
    with juice.set_block_size(K):
        n = [inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = 4)) for v in range(2)]
        s0 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gcbs))
        s1 = summate(multiply(n[0], n[1]), num_node_blocks = 1,
                     external_params = BlockScaleSumParams(ch_block_size = gcbs))
        root = summate(multiply(s0), multiply(s1), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    layer = _gated_layer(pc)
    if len(layer.external_node_infos) < 2:
        pytest.skip("this build did not place both gated nodes on one layer")

    g1 = torch.randn(s1.external_params.tensor_shapes(s1, 32)[0], device = dev) * 0.5
    with pytest.raises(NotImplementedError, match = "Partial supply"):
        juice.queries.sample(pc, num_samples = 32, sum_external_params = {s1: g1})


@cuda_only
def test_a_parameterization_without_a_sampler_refuses_rather_than_ignoring_its_parameters():
    """`LowRankSumParams` has no `sample_layer`. Drawing with the shared-parameter kernel would
    ignore the per-sample parameters and quietly return samples from a different distribution, so
    the base class raises instead."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    K = 16
    with juice.set_block_size(K):
        i0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 4))
        i1 = inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 4))
        ns = summate(multiply(i0, i1), num_node_blocks = 1,
                     external_params = LowRankSumParams(rank = 2))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)

    tensors = tuple(torch.randn(shape, device = dev) * 0.1
                    for shape in ns.external_params.tensor_shapes(ns, 32))
    with pytest.raises(NotImplementedError, match = "does not implement ancestral sampling"):
        juice.queries.sample(pc, num_samples = 32, sum_external_params = {ns: tensors})
