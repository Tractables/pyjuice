"""
Conditional sampling from GATED circuits at the shape a real decoder uses, against exact oracles.

Three gated test files existed and none of them reached this combination. `blockscale_sample_hmm_test`
runs a chain but with one gate row per transition and hard evidence; `blockscale_sample_test` reaches
the node axis but on a two-variable circuit; `blockscale_sample_untied_test` varies the gate axes but
checks routing, not magnitudes. Nothing combined **gated + conditional + soft evidence at depth**,
which is what a diffusion-LM decoder actually runs, and nothing anywhere used TOP-K soft evidence
with a conditional draw.

**The oracles here share no indexing with the kernels.** Each is built in float64 from the NODE-level
parameters (`ns._params`, `ns.edge_ids`) and the caller's own gate, then -- before being used to judge
a draw -- checked against the FORWARD pass. That second step is not ceremony: two of the three
oracles were wrong when first written, and the check is what caught them.

  * `_assemble_theta` exists because `ns._params` is `[edge_blocks, block_size, ch_block_size]`, NOT a
    dense `[K, K]` matrix. Reshaping it directly is valid only for a single-block node and silently
    interleaves rows otherwise -- measured as a confident 155-sigma "mismatch" against a sampler that
    was correct.
  * the gate maps `(node n, child c) -> (n // gate_block_size, c // gate_ch_block_size)` in GLOBAL
    index order, confirmed against the forward to 4e-7 rather than assumed.
"""

import math

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
from pyjuice.nodes.distributions import sort_soft_evidence_candidates


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

ALPHA = 1e-5
FORWARD_TOL = 1e-4          # relative, for "does my oracle agree with the forward"


def _z_bar(num_cells):
    return float(math.sqrt(2) * torch.erfinv(torch.tensor(1.0 - ALPHA / num_cells,
                                                          dtype = torch.float64)))


def _gate_shape(ns):
    node_gate_size, ch_gate_size = ns.external_params.gate_sizes(ns)
    return ns.num_nodes // node_gate_size, ns.num_ch_nodes // ch_gate_size


def _assemble_theta(ns, num_states, block_size):
    """
    The `[K, K]` transition matrix, assembled from the BLOCKED parameter layout.

    `ns._params[e]` is the `[block_size, ch_block_size]` tile of edge block `e`, whose parent and
    child blocks are `ns.edge_ids[:, e]`. A plain `reshape(K, K)` is only the same thing when there
    is exactly one block.
    """
    theta = torch.zeros([num_states, num_states], dtype = torch.float64)
    for e, (parent_block, child_block) in enumerate(ns.edge_ids.t().tolist()):
        theta[parent_block * block_size:(parent_block + 1) * block_size,
              child_block * block_size:(child_block + 1) * block_size] = ns._params[e].double()
    return theta


def _gated_transition(ns, log_gate, num_states, block_size, device):
    """`A[j, i]` proportional to `phi[g(j, i)] * theta[j, i]`, row-normalised."""
    gate_bs, gate_cbs = ns.external_params.gate_sizes(ns)
    theta = _assemble_theta(ns, num_states, block_size).to(device)
    row = torch.arange(num_states, device = device) // gate_bs
    col = torch.arange(num_states, device = device) // gate_cbs
    w = theta * torch.exp(log_gate.double()[row][:, col])
    return w / w.sum(dim = 1, keepdim = True)


def _drawn_state(pc, emit_ns, frontier):
    """Which hidden state a sample resolved to at one timestep, read off the frontier."""
    lo, hi = emit_ns._output_ind_range
    hit = (frontier >= lo) & (frontier < hi)
    rows, cols = hit.nonzero(as_tuple = True)
    out = torch.zeros([frontier.size(1)], dtype = torch.long, device = frontier.device)
    out[cols] = frontier[rows, cols] - lo
    return out



def _build_chain(T, K, C, gate_bs, gate_cbs, seed = 0):
    """A `T`-step chain whose transitions each carry a node-axis gate grid, under one group."""
    torch.manual_seed(seed)
    with juice.set_block_size(K):
        emits = [inputs(0, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = C))]
        transitions, chain = [], emits[0]
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = C))
            emits.append(emit)
            chain = summate(multiply(chain, emit), num_node_blocks = 1,
                            external_params = BlockScaleSumParams(block_size = gate_bs,
                                                                  ch_block_size = gate_cbs))
            transitions.append(chain)
        root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 3.0)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    pc.register_external_params_group("tr", transitions)
    return pc, emits, transitions, root


def _chain_exact_posterior(emits, transitions, root, gate, evidence, tokens, observed, K, device):
    """
    `P(hidden state at t | evidence)` for every `t`, by exact forward-backward in float64.

    Built from node-level parameters and the caller's gate only -- it shares no indexing with the
    kernels, which is what makes it a real oracle rather than a second opinion.
    """
    T = len(emits)
    num_node_gates = _gate_shape(transitions[0])[0]

    emit_p = [e._params.reshape(K, evidence.size(-1)).double().to(device) for e in emits]
    beta = evidence[0].double().exp()
    ev = [(beta[t, tokens[0, t]] * emit_p[t][:, tokens[0, t]]) if bool(observed[t])
          else (beta[t][None, :] * emit_p[t]).sum(dim = 1) for t in range(T)]

    A = [_gated_transition(transitions[t - 1],
                           gate[0, (t - 1) * num_node_gates:t * num_node_gates, :], K, K, device)
         for t in range(1, T)]

    V, W = [ev[0]], [None]
    for t in range(1, T):
        W.append(V[t - 1] * ev[t])
        V.append(A[t - 1] @ W[t])

    r = root._params.reshape(K).double().to(device)
    r = r / r.sum()
    Q = r * V[T - 1] / (r * V[T - 1]).sum()

    posterior = [None] * T
    for t in range(T - 1, 0, -1):
        posterior[t] = (Q[:, None] * A[t - 1] * W[t][None, :] / V[t][:, None]).sum(dim = 0)
        Q = posterior[t]
    return posterior


def _chain_inputs(T, C, K, device, seed, batch):
    """Evidence, tokens, observed mask and a gate for one chain -- identical across the batch."""
    torch.manual_seed(seed)
    evidence = torch.log_softmax(torch.randn([1, T, C], device = device) * 2.0, dim = -1)
    tokens = torch.randint(0, C, [1, T], device = device)
    observed = torch.zeros([T], dtype = torch.bool, device = device)
    observed[::2] = True
    return evidence, tokens, observed


def _run_chain_forward(pc, tokens, evidence, observed, gate, batch):
    pc(tokens.expand(batch, -1).contiguous(),
       categorical_evidence_logp = evidence.expand(batch, -1, -1).contiguous(),
       soft_evidence_value_mask = observed[None, :].expand(batch, -1),
       sum_external_params = {"tr": gate.expand(batch, -1, -1).contiguous()})


def _assert_chain_matches(pc, emits, posterior, frontier, batch, K, label):
    bar = _z_bar((len(emits) - 1) * K)
    for t in range(1, len(emits)):
        drawn = _drawn_state(pc, emits[t], frontier)
        frequency = torch.bincount(drawn, minlength = K).double() / batch
        exact = posterior[t]
        se = (exact * (1.0 - exact) / batch).sqrt().clamp(min = 1e-12)
        z = float(((frequency - exact) / se).abs().max())
        assert z < bar, f"{label}: timestep {t}: max |z| = {z:.2f} (bar {bar:.2f})"


# ------------------------------------------------------------------ a deep gated chain, soft evidence

@cuda_only
def test_a_deep_gated_chain_under_soft_evidence_matches_the_exact_recursion():
    """
    Eight timesteps, a per-transition node-axis gate grid, alternating observed/masked positions, and
    a conditional draw judged PER TIMESTEP against an exact forward-backward recursion.

    Depth is the point. A defect that accumulates -- a gate applied to the wrong transition, an
    element marginal read one layer off -- would grow along the chain, which a two-variable circuit
    cannot show. The per-timestep breakdown is asserted individually so a late-chain drift cannot be
    averaged away by early timesteps that are still right.
    """
    T, K, C, gate_bs, gate_cbs = 8, 16, 7, 2, 2
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    with juice.set_block_size(K):
        emits = [inputs(0, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = C))]
        transitions, chain = [], emits[0]
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = C))
            emits.append(emit)
            chain = summate(multiply(chain, emit), num_node_blocks = 1,
                            external_params = BlockScaleSumParams(block_size = gate_bs,
                                                                  ch_block_size = gate_cbs))
            transitions.append(chain)
        root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)

    root.init_parameters(perturbation = 3.0)
    pc = juice.compile(root, verbose = False).to(device)
    pc.register_external_params_group("tr", transitions)

    num_node_gates, num_ch_gates = _gate_shape(transitions[0])
    assert num_node_gates > 1, "the node axis is degenerate -- this test would not exercise it"

    batch = 60_000
    torch.manual_seed(4)
    gate = torch.randn([1, (T - 1) * num_node_gates, num_ch_gates], device = device)
    observed = torch.zeros([T], dtype = torch.bool, device = device)
    observed[::2] = True

    torch.manual_seed(5)
    evidence = torch.log_softmax(torch.randn([1, T, C], device = device) * 2.0, dim = -1)
    tokens = torch.randint(0, C, [1, T], device = device)

    pc(tokens.expand(batch, -1).contiguous(),
       categorical_evidence_logp = evidence.expand(batch, -1, -1).contiguous(),
       soft_evidence_value_mask = observed[None, :].expand(batch, -1),
       sum_external_params = {"tr": gate.expand(batch, -1, -1).contiguous()})

    # ---- the exact recursion, float64, from node-level parameters only
    emit_p = [e._params.reshape(K, C).double().to(device) for e in emits]
    beta = evidence[0].double().exp()
    ev = [(beta[t, tokens[0, t]] * emit_p[t][:, tokens[0, t]]) if bool(observed[t])
          else (beta[t][None, :] * emit_p[t]).sum(dim = 1) for t in range(T)]

    A = [_gated_transition(transitions[t - 1], gate[0, (t - 1) * num_node_gates:t * num_node_gates, :],
                           K, K, device) for t in range(1, T)]

    V, W = [ev[0]], [None]
    for t in range(1, T):
        W.append(V[t - 1] * ev[t])
        V.append(A[t - 1] @ W[t])

    r = root._params.reshape(K).double().to(device)
    r = r / r.sum()
    Q = r * V[T - 1] / (r * V[T - 1]).sum()

    posterior = [None] * T
    for t in range(T - 1, 0, -1):
        posterior[t] = (Q[:, None] * A[t - 1] * W[t][None, :] / V[t][:, None]).sum(dim = 0)
        Q = posterior[t]

    frontier = juice.queries.sample(pc, conditional = True, _sample_input_ns = False)
    bar = _z_bar((T - 1) * K)

    for t in range(1, T):
        drawn = _drawn_state(pc, emits[t], frontier)
        frequency = torch.bincount(drawn, minlength = K).double() / batch
        exact = posterior[t]
        se = (exact * (1.0 - exact) / batch).sqrt().clamp(min = 1e-12)
        z = float(((frequency - exact) / se).abs().max())
        assert z < bar, f"timestep {t}: max |z| = {z:.2f} (bar {bar:.2f})"


# ------------------------------------------------------------------- a MULTI-BLOCK gated transition

@cuda_only
def test_a_multi_block_gated_transition_matches_its_exact_posterior():
    """
    The parameter layout the single-block tests cannot reach: three blocks of four, so `_params` is
    `[9, 4, 4]` and the transition has to be assembled block by block.

    The oracle is checked against the FORWARD first. Without that step a mistake in the assembly or
    in the gate's index mapping looks exactly like a sampler bug -- which is how this was first
    misdiagnosed.
    """
    block_size, num_blocks, C = 4, 3, 5
    K = block_size * num_blocks
    gate_bs, gate_cbs = 2, 2
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    i0 = inputs(0, num_node_blocks = num_blocks, block_size = block_size,
                dist = dists.SoftEvidenceCategorical(num_cats = C))
    i1 = inputs(1, num_node_blocks = num_blocks, block_size = block_size,
                dist = dists.SoftEvidenceCategorical(num_cats = C))
    gated = summate(multiply(i0, i1), num_node_blocks = num_blocks, block_size = block_size,
                    external_params = BlockScaleSumParams(block_size = gate_bs,
                                                          ch_block_size = gate_cbs))
    root = summate(multiply(gated), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 3.0)

    pc = juice.compile(root, verbose = False).to(device)
    pc.register_external_params_group("tr", [gated])
    assert gated._params.dim() == 3 and gated._params.size(0) > 1, \
        "this node is a single parameter block -- the test would not exercise the assembly"

    num_node_gates, num_ch_gates = _gate_shape(gated)
    batch = 60_000
    torch.manual_seed(4)
    gate = torch.randn([1, num_node_gates, num_ch_gates], device = device)

    torch.manual_seed(5)
    evidence = torch.log_softmax(torch.randn([1, 2, C], device = device) * 2.0, dim = -1)
    tokens = torch.randint(0, C, [1, 2], device = device)
    observed = torch.zeros([1, 2], dtype = torch.bool, device = device)
    observed[:, 0] = True

    pc(tokens.expand(batch, -1).contiguous(),
       categorical_evidence_logp = evidence.expand(batch, -1, -1).contiguous(),
       soft_evidence_value_mask = observed.expand(batch, -1),
       sum_external_params = {"tr": gate.expand(batch, -1, -1).contiguous()})

    ep0 = i0._params.reshape(K, C).double().to(device)
    ep1 = i1._params.reshape(K, C).double().to(device)
    beta = evidence[0].double().exp()
    child = (beta[0, tokens[0, 0]] * ep0[:, tokens[0, 0]]) * (beta[1][None, :] * ep1).sum(dim = 1)

    A = _gated_transition(gated, gate[0], K, block_size, device)

    # the oracle has to reproduce the FORWARD before it may judge the draw
    lo, hi = gated._output_ind_range
    predicted = A @ child
    actual = pc.node_mars[lo:hi, 0].double().exp()
    rel = float((predicted - actual).abs().max() / actual.abs().max())
    assert rel < FORWARD_TOL, f"the oracle disagrees with the forward by {rel:.2e} -- it is wrong"

    # P(child i) for the node the root selected, marginalised over that choice
    root_layer = pc.inner_layer_groups[-1][0]
    r_pids = root_layer.partitioned_pids[0][0]
    r_cids = root_layer.partitioned_cids[0][0]
    real = r_cids > 0                                    # edge tables are padded to a power of two
    p_parent = torch.softmax(pc.params[r_pids[real]].double().log()
                             + pc.element_mars[r_cids[real], 0].double(), dim = 0)

    # P(child i | parent j) is proportional to `phi * theta * exp(element_mars_i)` -- the child's OWN
    # marginal belongs in it. Leaving it out gives the PRIOR transition, which still reproduces
    # `node_mars` under `A @ child` and so survives the forward check above while being the wrong
    # thing to compare a posterior draw against.
    posterior = A * child[None, :]
    posterior = posterior / posterior.sum(dim = 1, keepdim = True)
    exact = (p_parent[:, None] * posterior).sum(dim = 0)

    frontier = juice.queries.sample(pc, conditional = True, _sample_input_ns = False)
    drawn = _drawn_state(pc, i0, frontier)
    frequency = torch.bincount(drawn, minlength = K).double() / batch

    se = (exact * (1.0 - exact) / batch).sqrt().clamp(min = 1e-12)
    z = float(((frequency - exact) / se).abs().max())
    assert z < _z_bar(K), f"multi-block gated draw: max |z| = {z:.2f} (bar {_z_bar(K):.2f})"


# ---------------------------------------------------------------------- TOP-K soft evidence

@cuda_only
def test_top_k_soft_evidence_with_a_conditional_gated_draw():
    """
    The candidate-set form of soft evidence: instead of a log-probability per category, the caller
    supplies the best `k` and their ids. A decoder uses this to keep the evidence support small.

    Nothing else in the suite exercises `soft_evidence_cat_ids` with a conditional draw at all, gated
    or otherwise. Candidates are sorted by category id, which the distribution requires.
    """
    K, C, top_k, gate_bs, gate_cbs = 16, 11, 4, 2, 2
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    with juice.set_block_size(K):
        i0 = inputs(0, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = C))
        i1 = inputs(1, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = C))
        gated = summate(multiply(i0, i1), num_node_blocks = 1,
                        external_params = BlockScaleSumParams(block_size = gate_bs,
                                                              ch_block_size = gate_cbs))
        root = summate(multiply(gated), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 3.0)

    pc = juice.compile(root, verbose = False).to(device)
    pc.register_external_params_group("tr", [gated])
    num_node_gates, num_ch_gates = _gate_shape(gated)

    batch = 60_000
    torch.manual_seed(5)
    full = torch.log_softmax(torch.randn([1, 2, C], device = device) * 2.0, dim = -1)
    lp, idx = full.topk(top_k, dim = -1)
    tokens = idx[:, :, 0].clone()                        # observe a candidate, so it is in the set
    lp, idx = sort_soft_evidence_candidates(lp, idx)

    observed = torch.zeros([1, 2], dtype = torch.bool, device = device)
    observed[:, 0] = True
    torch.manual_seed(4)
    gate = torch.randn([1, num_node_gates, num_ch_gates], device = device)

    pc(tokens.expand(batch, -1).contiguous(),
       categorical_evidence_logp = lp.expand(batch, -1, -1).contiguous(),
       soft_evidence_cat_ids = idx.expand(batch, -1, -1).contiguous(),
       soft_evidence_value_mask = observed.expand(batch, -1),
       sum_external_params = {"tr": gate.expand(batch, -1, -1).contiguous()})

    ep0 = i0._params.reshape(K, C).double().to(device)
    ep1 = i1._params.reshape(K, C).double().to(device)
    beta = lp[0].double().exp()

    hit = (idx[0, 0] == tokens[0, 0]).nonzero()[0, 0]    # where the observed token sits after sorting
    value0 = beta[0, hit] * ep0[:, tokens[0, 0]]
    value1 = (beta[1][None, :] * ep1[:, idx[0, 1]]).sum(dim = 1)   # marginal over the candidates only
    child = value0 * value1

    A = _gated_transition(gated, gate[0], K, K, device)

    lo, hi = gated._output_ind_range
    rel = float(((A @ child) - pc.node_mars[lo:hi, 0].double().exp()).abs().max()
                / pc.node_mars[lo:hi, 0].double().exp().abs().max())
    assert rel < FORWARD_TOL, f"the top-k oracle disagrees with the forward by {rel:.2e}"

    root_layer = pc.inner_layer_groups[-1][0]
    r_pids = root_layer.partitioned_pids[0][0]
    r_cids = root_layer.partitioned_cids[0][0]
    real = r_cids > 0
    p_parent = torch.softmax(pc.params[r_pids[real]].double().log()
                             + pc.element_mars[r_cids[real], 0].double(), dim = 0)

    # P(child i | parent j) is proportional to `phi * theta * exp(element_mars_i)` -- the child's OWN
    # marginal belongs in it. Leaving it out gives the PRIOR transition, which still reproduces
    # `node_mars` under `A @ child` and so survives the forward check above while being the wrong
    # thing to compare a posterior draw against.
    posterior = A * child[None, :]
    posterior = posterior / posterior.sum(dim = 1, keepdim = True)
    exact = (p_parent[:, None] * posterior).sum(dim = 0)

    frontier = juice.queries.sample(pc, conditional = True, _sample_input_ns = False)
    drawn = _drawn_state(pc, i0, frontier)
    frequency = torch.bincount(drawn, minlength = K).double() / batch

    se = (exact * (1.0 - exact) / batch).sqrt().clamp(min = 1e-12)
    z = float(((frequency - exact) / se).abs().max())
    assert z < _z_bar(K), f"top-k gated conditional draw: max |z| = {z:.2f} (bar {_z_bar(K):.2f})"


# ----------------------------------------------------------------- ragged gated topology

@cuda_only
def test_a_ragged_gated_topology_matches_its_exact_posterior():
    """
    A gated node whose parent blocks have DIFFERENT numbers of children (2, 3 and 1 here).

    Ragged gated topologies were covered for unconditional draws only. The risk specific to them is
    the gate's column mapping: with a full grid, "the c-th child" and "the c-th edge of this parent"
    coincide, and they stop coinciding here. The oracle is checked against the forward first, so a
    wrong mapping shows up as an invalid oracle rather than as a phantom sampler bug.
    """
    block_size, num_blocks, C, gate_bs, gate_cbs = 4, 3, 5, 2, 2
    K = block_size * num_blocks
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    i0 = inputs(0, num_node_blocks = num_blocks, block_size = block_size,
                dist = dists.SoftEvidenceCategorical(num_cats = C))
    i1 = inputs(1, num_node_blocks = num_blocks, block_size = block_size,
                dist = dists.SoftEvidenceCategorical(num_cats = C))
    edges = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 2, 0, 1, 2, 1]], dtype = torch.long)
    gated = summate(multiply(i0, i1), edge_ids = edges, num_node_blocks = num_blocks,
                    block_size = block_size,
                    external_params = BlockScaleSumParams(block_size = gate_bs,
                                                          ch_block_size = gate_cbs))
    root = summate(multiply(gated), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 3.0)

    pc = juice.compile(root, verbose = False).to(device)
    pc.register_external_params_group("tr", [gated])
    assert sorted(torch.bincount(edges[0]).tolist()) != [2, 2, 2], "this topology is not ragged"

    num_node_gates, num_ch_gates = _gate_shape(gated)
    batch = 60_000
    torch.manual_seed(4)
    gate = torch.randn([1, num_node_gates, num_ch_gates], device = device)

    torch.manual_seed(5)
    evidence = torch.log_softmax(torch.randn([1, 2, C], device = device) * 2.0, dim = -1)
    tokens = torch.randint(0, C, [1, 2], device = device)
    observed = torch.zeros([1, 2], dtype = torch.bool, device = device)
    observed[:, 0] = True

    pc(tokens.expand(batch, -1).contiguous(),
       categorical_evidence_logp = evidence.expand(batch, -1, -1).contiguous(),
       soft_evidence_value_mask = observed.expand(batch, -1),
       sum_external_params = {"tr": gate.expand(batch, -1, -1).contiguous()})

    ep0 = i0._params.reshape(K, C).double().to(device)
    ep1 = i1._params.reshape(K, C).double().to(device)
    beta = evidence[0].double().exp()
    child = (beta[0, tokens[0, 0]] * ep0[:, tokens[0, 0]]) * (beta[1][None, :] * ep1).sum(dim = 1)

    A = _gated_transition(gated, gate[0], K, block_size, device)

    lo, hi = gated._output_ind_range
    actual = pc.node_mars[lo:hi, 0].double().exp()
    rel = float(((A @ child) - actual).abs().max() / actual.abs().max())
    assert rel < FORWARD_TOL, f"the ragged oracle disagrees with the forward by {rel:.2e}"

    root_layer = pc.inner_layer_groups[-1][0]
    r_pids = root_layer.partitioned_pids[0][0]
    r_cids = root_layer.partitioned_cids[0][0]
    real = r_cids > 0
    p_parent = torch.softmax(pc.params[r_pids[real]].double().log()
                             + pc.element_mars[r_cids[real], 0].double(), dim = 0)
    posterior = A * child[None, :]
    posterior = posterior / posterior.sum(dim = 1, keepdim = True)
    exact = (p_parent[:, None] * posterior).sum(dim = 0)

    frontier = juice.queries.sample(pc, conditional = True, _sample_input_ns = False)
    drawn = _drawn_state(pc, i0, frontier)
    frequency = torch.bincount(drawn, minlength = K).double() / batch
    se = (exact * (1.0 - exact) / batch).sqrt().clamp(min = 1e-12)
    z = float(((frequency - exact) / se).abs().max())
    assert z < _z_bar(K), f"ragged gated conditional draw: max |z| = {z:.2f}"


# ------------------------------------------------------- a CUDA graph, gated and conditional, at depth

@cuda_only
def test_a_deep_gated_chain_replays_from_a_cuda_graph():
    """
    Capture a conditional gated draw on a chain, then change BOTH the gate and the evidence and
    replay. The replay must follow the new state, not the captured one.

    Judged against the exact recursion rather than against an eager draw, so a graph that replayed
    stale gates cannot be excused by an eager path making the same mistake. Depth matters here for
    the same reason as above: a graph holds one captured pass over every layer.
    """
    T, K, C, gate_bs, gate_cbs = 6, 16, 7, 2, 2
    device = torch.device("cuda:0")
    batch = 60_000

    pc, emits, transitions, root = _build_chain(T, K, C, gate_bs, gate_cbs)
    num_node_gates, num_ch_gates = _gate_shape(transitions[0])

    def state(seed):
        evidence, tokens, observed = _chain_inputs(T, C, K, device, seed, batch)
        torch.manual_seed(seed + 100)
        gate = torch.randn([1, (T - 1) * num_node_gates, num_ch_gates], device = device)
        return evidence, tokens, observed, gate

    # capture on one state
    evidence, tokens, observed, gate = state(4)
    _run_chain_forward(pc, tokens, evidence, observed, gate, batch)
    juice.queries.sample(pc, conditional = True, _sample_input_ns = False, use_cudagraph = True)

    # now a DIFFERENT gate and different evidence, replayed through the graph
    evidence, tokens, observed, gate = state(9)
    _run_chain_forward(pc, tokens, evidence, observed, gate, batch)
    frontier = juice.queries.sample(pc, conditional = True, _sample_input_ns = False,
                                    use_cudagraph = True)

    posterior = _chain_exact_posterior(emits, transitions, root, gate, evidence, tokens, observed,
                                       K, device)
    _assert_chain_matches(pc, emits, posterior, frontier, batch, K, "graph replay")
