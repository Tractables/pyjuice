"""
Gated sum nodes (:class:`~pyjuice.nodes.external_params.BlockScaleSumParams`) under a constraint, their gates
(``sum_external_params``) bound to a decoder session
(:mod:`pyjuice.constraints.inference.dfa_exact.external_params`). Sample ``b``'s sum node ``n`` weighs child ``c`` by
``theta[n, c] phi_b[g(n, c)] / Z_b[n]``, so the truth is pyjuice's own gated forward: under the one-state constraint
``pc(x, missing_mask, sum_external_params)`` itself, under any other the log-sum of it over the completions the
constraint accepts, and for conditionals the float64 reference with every sample's weights. Every kind of PC of
conftest.py with its sums gated -- node gates of a node block or half one, child gates of a child block or finer --,
gates per row or one row for all, dense and compact storage, together with soft evidence, sessions, some layers gated
or given gates only, registered groups and tied gates; and what a session refuses.
"""
import itertools

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import BlockScaleSumParams, LowRankSumParams, inputs, multiply, summate

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 6, "pd_prod_dominated": 6,
         "pd_blockified": 6, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5, "hand_shared": 4,
         "hand_fold": 4}
CONSTRAINTS = {
    "anything": lambda: jc.DFA.anything(V),
    "contains": lambda: jc.DFA.contains([[1, 2]], V),
    "contains_either_and_not": lambda: jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)),
}
DEV = torch.device("cuda:0")
#: the gates' child block sizes: the child blocks', or the finest that divides them (one child per gate at most two)
CHILD_GATES = {"block": lambda ns: ns.ch_block_size,
               "fine": lambda ns: next(d for d in (2, 1) if ns.ch_block_size % d == 0)}
#: the gates' node block sizes: the node blocks', or half of them (a node block's rows read two node gates)
NODE_GATES = {"block": lambda ns: ns.block_size, "half": lambda ns: max(1, ns.block_size // 2)}


def gated_pc(build_pc, kind, child_gate = "fine", n = None, node_gate = "block", **gates):
    return build_pc(kind, KINDS[kind] if n is None else n, V, perturbation = 4.0,
                    gates = dict(child_gate = CHILD_GATES[child_gate], node_gate = NODE_GATES[node_gate], **gates))


def make_gates(pc, B, seed = 0, scale = 2.0, nodes = None):
    """Log gates [B, node gates, child gates] for every gated node group of ``pc`` (or ``nodes``; under tied gates
    their storage owners), from N(0, scale^2)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for ns in (pc.external_params_nodes if nodes is None else nodes):
        owner = ns.external_params.storage_owner(ns)
        if owner in out:
            continue
        shape = ns.external_params.tensor_shapes(owner, B)[0]
        out[owner] = (torch.randn(shape, generator = g) * scale).to(DEV)
    return out


def expand(gates, B):
    return {ns: t.expand(B, -1, -1).contiguous() for ns, t in gates.items()}


def evidence(n, B, seed = 0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.5
    missing[0] = True
    return data.to(DEV), missing.to(DEV)


def assert_close(got, want, n):
    got, want = got.view(-1), want.view(-1)
    assert torch.equal(torch.isneginf(got), torch.isneginf(want)), (got, want)
    fin = torch.isfinite(want)
    assert torch.isfinite(got[fin]).all()
    tol = 1e-5 * want[fin].abs() + n * 4e-6
    assert ((got[fin] - want[fin]).abs() <= tol).all(), (got[fin] - want[fin]).abs().max()


def assert_rows_close(got, want, n):
    assert got.shape == want.shape
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert ((got[fin] - want[fin]).abs() <= 1e-4 + n * 4e-6 + 1e-5 * want[fin].abs()).all(), \
        (got[fin] - want[fin]).abs().max()


@pytest.fixture(params = ["fused", "dense"])
def sum_path(request, monkeypatch):
    """The sum layers through the fused kernel only, or the dense (cuBLAS) path wherever they qualify -- gated layers
    never do: their weights differ per sample."""
    from pyjuice.constraints.inference.dfa_exact.kernels import sum as lifted_sum
    monkeypatch.setattr(lifted_sum, "DENSE_MIN_BLOCK", 1 if request.param == "dense" else 1 << 30)
    return request.param


def brute_force(pc, constraint, data, missing, gates, other = None):
    """log sum over the completions the constraint accepts of pyjuice's gated forward, each sample under its row of
    the gates (and of ``other`` keywords of ``pc(x, ...)``)."""
    out = []
    for b in range(data.size(0)):
        miss = missing[b].nonzero().view(-1).tolist()
        fills = list(itertools.product(range(V), repeat = len(miss)))
        xs = data[b].repeat(len(fills), 1)
        if miss:
            xs[:, miss] = torch.tensor(fills, device = DEV)
        xs = xs[torch.tensor([constraint.accepts(row) for row in xs.tolist()], device = DEV)]
        if xs.size(0) == 0:
            out.append(torch.tensor(-float("inf"), device = DEV))
            continue
        row = lambda t: t[b if t.size(0) > 1 else 0][None].expand(xs.size(0), *t.shape[1:]).contiguous()
        kw = {key: row(t) for key, t in (other or {}).items()}
        lls = pc(xs, sum_external_params = {ns: row(t) for ns, t in gates.items()}, **kw)
        out.append(torch.logsumexp(lls.view(-1).double(), 0).float())
    return torch.stack(out)


# -------------------------------------------------------------------------------------------------
# Marginals
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("node_gate", list(NODE_GATES))
@pytest.mark.parametrize("child_gate", list(CHILD_GATES))
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_the_one_state_constraint_is_the_pcs_own_marginal(kind, shared, child_gate, node_gate, build_pc):
    """Under ``DFA.anything`` the session's marginal is ``pc(x, missing_mask, sum_external_params)``: gates per row,
    or one row for all."""
    pc = gated_pc(build_pc, kind, child_gate, node_gate = node_gate)
    n = KINDS[kind]
    data, missing = evidence(n, 4)
    gates = make_gates(pc, 1 if shared else 4)
    cc = jc.compile(jc.DFA.anything(V), pc)
    with cc.decoder(sum_external_params = gates) as decoder:
        got = decoder.marginal(data, missing)
    want = pc(data, missing_mask = missing, sum_external_params = expand(gates, 4))
    if any(gates[ns].size(2) > 1 for ns in gates):                           # a node of several gates: they matter
        assert (want - pc(data, missing_mask = missing)).abs().max() > 1e-2
    assert_close(got, want, n)


@pytest.mark.parametrize("constraint", ["contains", "contains_either_and_not"])
@pytest.mark.parametrize("kind", list(KINDS))
def test_marginal_matches_a_brute_force(kind, constraint, sum_path, build_pc):
    """``log p(C, x_observed)`` under gates per row: the log-sum of pyjuice's gated forward over the completions the
    constraint accepts, dense and compact storage, every path of the sums."""
    pc = gated_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing = evidence(n, 3, seed = 1)
    gates = make_gates(pc, 3, seed = 1)
    c = CONSTRAINTS[constraint]()
    cc = jc.compile(c, pc)
    want = brute_force(pc, c, data, missing, gates)
    for storage in ("dense", "compact"):
        with cc.decoder(sum_external_params = gates, storage = storage) as decoder:
            assert_close(decoder.marginal(data, missing), want, n)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand", "hand_shared"])
def test_storage_and_shared_gates(kind, build_pc):
    """Compact storage gives the dense one's marginal; gates of one row read by every row give what the same row
    repeated per row gives (kept once, or per sample); inside fast_inference too."""
    pc = gated_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing = evidence(n, 5, seed = 2)
    gates = make_gates(pc, 1, seed = 2)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    got = {}
    for storage, rows, fi in itertools.product(("dense", "compact"), ("shared", "per_row"), (False, True)):
        with juice.fast_inference() if fi else torch.no_grad():
            with cc.decoder(sum_external_params = gates if rows == "shared" else expand(gates, 5),
                            storage = storage) as decoder:
                got[storage, rows, fi] = decoder.marginal(data, missing)
    want = brute_force(pc, CONSTRAINTS["contains"](), data, missing, gates)
    for key, value in got.items():
        assert_close(value, want, n)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_left"])
def test_shared_gates_are_kept_once(kind, build_pc):
    """Gates of one row for all: the values over positions no row observes are every sample's, kept once for the
    session (the brute force's value); gates per row: none kept once over them."""
    from pyjuice.constraints.inference.dfa_exact.queries.marginal import marginal
    pc = gated_pc(build_pc, kind)
    n = KINDS[kind]
    data, missing = evidence(n, 3)
    free = torch.arange(n, device = DEV) >= n // 2
    missing[:] = ~free if kind == "hand_left" else free
    c = CONSTRAINTS["contains"]()
    cc = jc.compile(c, pc)
    for Bg in (1, 3):
        gates = make_gates(pc, Bg)
        external = cc.backend.external_inputs(dict(sum_external_params = gates))
        with juice.fast_inference():
            report = {}
            got = marginal(cc.backend, data, missing, report = report, external = external)
        assert bool(report["plan"].prior) == (Bg == 1)
        assert_close(got, brute_force(pc, c, data, missing, gates), n)


def test_gates_and_soft_evidence(build_pc, reference):
    """Gated sums over SoftEvidenceCategorical inputs, the evidence dense or top-k, per row or shared, the gates per row
    or shared: the reference's marginals and conditionals."""
    n = KINDS["hmm"]
    Vm = 12
    pc = build_pc("hmm", n, Vm, perturbation = 4.0, input_dist = dists.SoftEvidenceCategorical,
                  gates = dict(child_gate = CHILD_GATES["fine"]))
    c = jc.DFA.contains([[3, 7]], Vm)
    cc = jc.compile(c, pc)
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, Vm, (3, n), generator = g).to(DEV)
    m = (torch.rand(3, n, generator = g) < 0.5).to(DEV)
    m[0] = True
    for Be, Bg, topk in itertools.product((1, 3), (1, 3), (False, True)):
        E = (torch.randn(Be, n, Vm, generator = g) * 2).to(DEV)
        ev = dict(categorical_evidence_logp = E)
        if topk:
            ev = dict(zip(("categorical_evidence_logp", "soft_evidence_cat_ids"), torch.topk(E, Vm - 2, dim = 2)))
        gates = make_gates(pc, Bg, seed = Be + Bg)
        inputs = dict(ev, sum_external_params = gates)
        with cc.decoder(precision = "fp32", **inputs) as decoder:
            marg = decoder.marginal(data, m)
            cond = decoder.conditional(data, m)
        assert_close(marg, reference.marginal(cc, data, m, inputs), n)
        assert_rows_close(cond, reference.conditional(cc, data, m, inputs = inputs), n)


# -------------------------------------------------------------------------------------------------
# Conditionals
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("node_gate", list(NODE_GATES))
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_conditional_matches_the_reference(kind, shared, node_gate, build_pc, reference):
    """``log p(x_t = v | e, C)`` under gates, every missing position: the reference's (every sample's weights), dense
    and compact storage; the log marginal it returns is the marginal's; candidates alone score as the full rows."""
    pc = gated_pc(build_pc, kind, node_gate = node_gate)
    n = KINDS[kind]
    data, m = evidence(n, 3, seed = 3)
    gates = make_gates(pc, 1 if shared else 3, seed = 3)
    c = CONSTRAINTS["contains_either_and_not"]()
    cc = jc.compile(c, pc)
    inputs = dict(sum_external_params = gates)
    want = reference.conditional(cc, data, m, inputs = inputs)
    for storage in ("dense", "compact"):
        with cc.decoder(storage = storage, precision = "fp32", **inputs) as decoder:
            got, lm = decoder.conditional(data, m, return_log_marginal = True)
            marg = decoder.marginal(data, m)
            cand = torch.randint(0, V + 2, (3, n, 4), device = DEV)
            got_c = decoder.conditional(data, m, candidates = cand)
        assert_rows_close(got, want, n)
        rows = torch.nonzero(m)
        assert_close(lm, marg.view(-1)[rows[:, 0]], n)
        cr = cand[rows[:, 0], rows[:, 1]]
        assert_rows_close(got_c, torch.where(cr < V, got.gather(1, cr.clamp(max = V - 1)), -float("inf")), n)


@pytest.mark.parametrize("latents, block, gate, node_gate", [(64, 32, 8, 32), (64, 32, 32, 32), (128, 64, 64, 64),
                                                             (64, 32, 32, 16), (64, 32, 8, 8), (128, 64, 64, 16),
                                                             (128, 64, 16, 8)])
def test_wide_gates(latents, block, gate, node_gate, gate_sums, reference):
    """HMMs of wider blocks, gates of 8 children (an edge chunk and a tile of children read several), 32 (a chunk
    reads one) and 64 (a tile too), of a node block or of 16 or 8 of its nodes (its rows read several): the
    reference's marginals and conditionals, per-row and shared gates."""
    torch.manual_seed(0)
    n = 5
    ns = juice.structures.HMM(seq_length = n, num_latents = latents, num_emits = V, block_size = block)
    gate_sums(ns, child_gate = lambda s: min(gate, s.ch_block_size), node_gate = lambda s: min(node_gate, s.block_size),
              which = lambda s: s.num_nodes == latents)
    ns.init_parameters(perturbation = 4.0)
    pc = juice.compile(ns, verbose = False).to(DEV)
    cc = jc.compile(CONSTRAINTS["contains_either_and_not"](), pc)
    data, m = evidence(n, 3, seed = 6)
    for Bg in (3, 1):
        inputs_ = dict(sum_external_params = make_gates(pc, Bg, seed = Bg))
        for storage in ("dense", "compact"):
            with cc.decoder(storage = storage, precision = "fp32", **inputs_) as decoder:
                assert_close(decoder.marginal(data, m), reference.marginal(cc, data, m, inputs_), n)
                assert_rows_close(decoder.conditional(data, m), reference.conditional(cc, data, m, inputs = inputs_), n)


@pytest.mark.parametrize("child_gate", [1, 4])
@pytest.mark.parametrize("node_gate", [32, 16, 8])
def test_gates_far_from_the_values(node_gate, child_gate):
    """One gated block of 32 nodes over 32 children ``x0[c] x1[c]``, every other child gate's children ~85 nats less
    likely per input and gated 200 nats above the others: those far children are the nodes' values, so a sum that
    shifts its children before their gates scale them loses them. pyjuice's gated forward is the truth: the marginals,
    and the conditionals as its log-softmax over the completions."""
    x = [inputs(v, num_node_blocks = 1, block_size = 32, dist = dists.Categorical(num_cats = V)) for v in range(2)]
    gate = BlockScaleSumParams(block_size = None if node_gate == 32 else node_gate, ch_block_size = child_gate)
    s = summate(multiply(x[0], x[1]), num_node_blocks = 1, block_size = 32, external_params = gate)
    root = summate(multiply(s), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    far = (torch.arange(32) // child_gate) % 2 == 1
    for xi in x:
        p = torch.full((32, V), 0.5)
        p[far, 0], p[~far, 0], p[~far, 1:] = 1e-37, 1.0 - 2e-7, 1e-7
        xi._params = (p / p.sum(1, keepdim = True)).reshape(-1)
    pc = juice.compile(root, verbose = False).to(DEV)
    (ns,) = pc.external_params_nodes
    lphi = torch.zeros(ns.external_params.tensor_shapes(ns, 2)[0])
    lphi[..., 1::2] = 200.0
    lphi[1] += torch.randn(lphi.shape[1:], generator = torch.Generator().manual_seed(0)) * 5
    gates = {ns: lphi.to(DEV)}
    data = torch.zeros(2, 2, dtype = torch.long, device = DEV)
    m = torch.tensor([[False, False], [False, True]], device = DEV)
    cc = jc.compile(jc.DFA.anything(V), pc)
    with cc.decoder(sum_external_params = gates) as decoder:
        marg = decoder.marginal(data, m)
        cond = decoder.conditional(data, m)
    want = pc(data, missing_mask = m, sum_external_params = gates)
    assert want.max() < -80                                                  # the far children's values
    assert_close(marg, want, 2)
    xs = torch.tensor([[0, v] for v in range(V)], device = DEV)
    lls = pc(xs, sum_external_params = {ns: lphi[1:].expand(V, -1, -1).to(DEV)}).view(1, V)
    assert_rows_close(cond, torch.log_softmax(lls, dim = 1), 2)


@pytest.mark.parametrize("child_gate, node_gate", [("fine", "block"), ("block", "half")])
@pytest.mark.parametrize("kind", ["hmm", "hmm_block_sparse", "pd"])
def test_log_normalizers_against_float64(kind, child_gate, node_gate, build_pc, reference):
    """Every gated node's log Z of every sample (``block_scale.log_normalizers``) against ``log sum_c w[n, c]
    phi_b[gate(n, c)]`` in float64, straight from the weights, within the normalizers' error bound
    (``block_scale.logz_error_bound``): gates N(0, 2^2), every other child gate 200 nats up, and -- where a gate row
    has a gate its nodes have no edge in (block-sparse sums) -- that gate 1000 nats above the row's others, so the
    nodes' Z is 1000 nats below the row's largest gate: a sum that underflows even in float64, computed exactly."""
    from pyjuice.constraints.inference.dfa_exact.external_params import block_scale
    pc = gated_pc(build_pc, kind, child_gate = child_gate, node_gate = node_gate)
    layout = jc.compile(jc.DFA.anything(V), pc).backend.program().gate_cells
    B, deep = 3, 0
    for case in ("normal", "far", "deep"):
        gates = make_gates(pc, B, seed = 1)
        for ns in pc.external_params_nodes:
            owner = ns.external_params.storage_owner(ns)
            if case == "far":
                gates[owner][:, :, 1::2] += 200.0
            elif case == "deep":                    # a gate its row's nodes have no edge in: 1000 nats up
                ng, cg = ns.external_params.gate_sizes(ns)
                W = reference.sum_weights(pc, ns)
                empty = (W.view(W.size(0), -1, cg) == 0).all(dim = 2)                   # [nodes, child gates]
                for r in range(gates[owner].size(1)):
                    rows = empty[r * ng:(r + 1) * ng]
                    hit = rows.all(dim = 0).nonzero().view(-1) if (~rows).any() else rows.new_zeros(0)
                    if hit.numel():
                        gates[owner][:, r, hit[0]] += 1000.0
                        deep += 1
        check_normalizers(pc, layout, gates, reference, (kind, case))
    assert kind != "hmm_block_sparse" or deep > 0


def check_normalizers(pc, layout, gates, reference, label):
    """``block_scale.log_normalizers`` of ``gates`` against float64 straight from the weights, within its bound."""
    from pyjuice.constraints.inference.dfa_exact.external_params import block_scale
    session = block_scale.session_inputs(pc, layout, gates)
    got = block_scale.log_normalizers(pc, layout, session)
    worst = 0.0
    for ns in session.nodes:
        ng, cg = ns.external_params.gate_sizes(ns)
        W = reference.sum_weights(pc, ns).double()
        lphi = gates[ns.external_params.storage_owner(ns)].double()
        g = lphi[:, torch.arange(W.size(0), device = DEV) // ng][:, :, torch.arange(W.size(1), device = DEV) // cg]
        want = torch.logsumexp(torch.log(W)[None] + g, dim = 2)                            # [B, nodes]
        first = ns._output_ind_range[0]
        have = got[:, first:first + ns.num_nodes].double()
        assert torch.equal(torch.isneginf(have), torch.isneginf(want)), label
        fin = torch.isfinite(want)
        err = (have[fin] - want[fin]).abs()
        bound = block_scale.logz_error_bound(W.size(1) // cg, want[fin])
        assert (err <= bound).all(), (label, float((err / bound).max()))
        worst = max(worst, float((err / bound).max()) if err.numel() else 0.0)
    return worst


@pytest.mark.parametrize("nats", [40, 60, 70, 80, 90, 95, 100, 110])
def test_log_normalizers_across_the_exact_threshold(nats, build_pc, reference):
    """A gate each row's nodes have no edge in (block-sparse sums) ``nats`` above the row's others: the nodes' fp32
    sums, their terms shifted by the row's largest gate, ``nats`` below 1 -- in the normal range, subnormal, and zero
    -- within the bound either way: computed in fp32 above the normalizers' threshold, exactly below it (a subnormal
    sum keeps a few bits: 95-100 nats would be off by 1e-4 to 1e-2)."""
    pc = gated_pc(build_pc, "hmm_block_sparse", child_gate = "fine", node_gate = "block")
    layout = jc.compile(jc.DFA.anything(V), pc).backend.program().gate_cells
    gates, raised = make_gates(pc, 3, seed = 2), 0
    for ns in pc.external_params_nodes:
        owner = ns.external_params.storage_owner(ns)
        ng, cg = ns.external_params.gate_sizes(ns)
        W = reference.sum_weights(pc, ns)
        empty = (W.view(W.size(0), -1, cg) == 0).all(dim = 2)                               # [nodes, child gates]
        for r in range(gates[owner].size(1)):
            rows = empty[r * ng:(r + 1) * ng]
            hit = rows.all(dim = 0).nonzero().view(-1) if (~rows).any() else rows.new_zeros(0)
            if hit.numel():
                gates[owner][:, r, hit[0]] = gates[owner][:, r].max(dim = 1).values + float(nats)
                raised += 1
    assert raised > 0
    check_normalizers(pc, layout, gates, reference, nats)


def test_children_in_another_order(reference):
    """Two gated sums over the same two products in opposite orders, a gate per child: the second's children are
    rows in the first's order, so its edges' places among its own children are not their rows' order -- the reference's
    marginals and conditionals."""
    x = [inputs(v, num_node_blocks = 1, block_size = 2, dist = dists.Categorical(num_cats = V)) for v in range(3)]
    pA, pB = multiply(x[0], x[1]), multiply(x[0], x[1])
    gate = lambda: BlockScaleSumParams(ch_block_size = 1)
    s1 = summate(pA, pB, num_node_blocks = 1, block_size = 2, external_params = gate())
    s2 = summate(pB, pA, num_node_blocks = 1, block_size = 2, edge_ids = torch.tensor([[0, 0], [1, 0]]),
                 external_params = gate())
    root = summate(multiply(s1, x[2]), multiply(s2, x[2]), num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 4.0)
    pc = juice.compile(root, verbose = False).to(DEV)
    assert pA._output_ind_range[1] == pB._output_ind_range[0]               # pA's rows, then pB's
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    data, m = evidence(3, 3)
    gates = make_gates(pc, 3, scale = 3.0)
    inputs_ = dict(sum_external_params = gates)
    with cc.decoder(precision = "fp32", **inputs_) as decoder:
        assert_close(decoder.marginal(data, m), reference.marginal(cc, data, m, inputs_), 3)
        assert_rows_close(decoder.conditional(data, m), reference.conditional(cc, data, m, inputs = inputs_), 3)


def test_sessions_in_one_scope(build_pc):
    """Inside one fast_inference scope, which keeps the values no sample's evidence reaches: a plain query, then
    sessions of gates of one row for all, each another -- every one reads only the values kept under its own."""
    pc = gated_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, missing = evidence(n, 3)
    missing[:] = torch.arange(n, device = DEV) >= n // 2                       # a suffix no row observes
    c = CONSTRAINTS["contains"]()
    cc = jc.compile(c, pc)
    with juice.fast_inference():
        assert_close(cc.marginal(data, missing), brute_force(pc, c, data, missing, {}), n)
        for seed in (1, 2):
            gates = make_gates(pc, 1, seed = seed)
            with cc.decoder(sum_external_params = gates) as decoder:
                assert_close(decoder.marginal(data, missing), brute_force(pc, c, data, missing, gates), n)


def test_a_session_reuses_only_what_its_rows_gates_allow(build_pc):
    """Prefix decoding with marginals and conditionals in one session, then the rows reordered as beams are: every
    call is what a fresh session gives -- with gates per row too, where a row taking another row's values would
    read that row's gates."""
    pc = gated_pc(build_pc, "hmm")
    n = KINDS["hmm"]
    data, _ = evidence(n, 4)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    prefix = lambda t: (torch.arange(n, device = DEV)[None, :] >= t).expand(4, n)
    order = torch.tensor([2, 0, 1, 3], device = DEV)
    calls = [(data, prefix(t)) for t in range(n)] + [(data[order], prefix(n - 1))]
    query = lambda d, x, m: (d.marginal(x, m), d.conditional(x, m, targets = [min(n - 1, int(m[0].int().argmax()))]))
    for Bg in (4, 1):
        gates = make_gates(pc, Bg)
        with cc.decoder(sum_external_params = gates) as decoder:
            session = [query(decoder, x, m) for x, m in calls]
        for (x, m), (marg, cond) in zip(calls, session):
            with cc.decoder(sum_external_params = gates) as decoder:
                fresh = query(decoder, x, m)
            assert_close(marg, fresh[0], n)
            assert_rows_close(cond, fresh[1], n)


# -------------------------------------------------------------------------------------------------
# Some layers gated, groups, tied gates
# -------------------------------------------------------------------------------------------------

def test_some_layers_gated(build_pc, reference):
    """A PD with only its sums over intervals of two or more positions gated, and gates given to only some of the
    gated layers: the others are plain sum layers -- the library's forward, the reference, plain queries alike."""
    kind = "pd"
    n = KINDS[kind]
    pc = build_pc(kind, n, V, perturbation = 4.0,
                  gates = dict(child_gate = CHILD_GATES["fine"], which = lambda ns: len(ns.scope) >= 2))
    gated_layers = [layer for lg in pc.inner_layer_groups if lg.is_sum() for layer in lg.layers
                    if getattr(layer, "external_params", None) is not None]
    assert len(gated_layers) > 1
    some = make_gates(pc, 3, nodes = [ns for layer in gated_layers[::2] for ns in layer.nodes])
    data, m = evidence(n, 3, seed = 4)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    for gates in (some, None):
        kw = {} if gates is None else dict(sum_external_params = gates)
        with cc.decoder(precision = "fp32", **kw) as decoder:
            marg = decoder.marginal(data, m)
            cond = decoder.conditional(data, m)
        assert_close(marg, brute_force(pc, CONSTRAINTS["contains"](), data, m, gates or {}), n)
        assert_rows_close(cond, reference.conditional(cc, data, m, inputs = kw), n)
    assert_close(cc.marginal(data, m), marg, n)                              # no gates: plain sums


def test_a_registered_group_and_tied_gates(build_pc):
    """A registered group of gated nodes gives what their own entries give; under tied gates (``tie_external``) the
    copies of a tied node read their source's: one tensor, as the library takes it."""
    n = KINDS["hmm"]
    pc = gated_pc(build_pc, "hmm")
    data, m = evidence(n, 3)
    gates = make_gates(pc, 3)
    nodes = list(gates)
    pc.register_external_params_group("chain", nodes)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    with cc.decoder(sum_external_params = gates) as decoder:
        per_node = decoder.marginal(data, m)
    with cc.decoder(sum_external_params = {"chain": torch.cat([gates[ns] for ns in nodes], dim = 1)}) as decoder:
        assert torch.equal(decoder.marginal(data, m), per_node)

    tied = gated_pc(build_pc, "hmm", tie_external = True)
    gates = make_gates(tied, 3, seed = 5)
    assert len(gates) < len(tied.external_params_nodes)                       # the copies read their source's
    cc = jc.compile(CONSTRAINTS["contains"](), tied)
    with cc.decoder(sum_external_params = gates) as decoder:
        got = decoder.marginal(data, m)
    assert_close(got, brute_force(tied, CONSTRAINTS["contains"](), data, m, gates), n)


# -------------------------------------------------------------------------------------------------
# Codpula's shape
# -------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_codpula_shape_gated(reference, gate_sums):
    """Codpula's gated PC: an HMM of 1024 latents over 32 positions, gates of 32 x 32, under language-model soft
    evidence over 126464 tokens; prefix decoding in a session against the reference."""
    torch.manual_seed(0)
    n, Vc = 32, 126464
    ns = juice.structures.GeneralizedHMM(seq_length = n, num_latents = 1024, block_size = 32,
                                         input_dist = dists.SoftEvidenceCategorical(num_cats = Vc))
    gate_sums(ns, child_gate = lambda s: 32, which = lambda s: s.num_nodes == 1024)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False).to(DEV)
    c = jc.DFA.contains([[50000, 100000]], Vc) & jc.Not(jc.DFA.contains([[126000]], Vc))
    cc = jc.compile(c, pc)
    E = torch.log_softmax(torch.randn(1, n, Vc, device = DEV) * 3, dim = -1)
    E[:, 5, 50000] += 20.0
    E[:, 6, 100000] += 20.0
    data = torch.multinomial(E[0].exp(), 1).view(1, n)
    gates = make_gates(pc, 1)
    inputs = dict(categorical_evidence_logp = E, sum_external_params = gates)
    with cc.decoder(**inputs) as decoder:
        for t in range(n):
            m = (torch.arange(n, device = DEV) >= t)[None, :]
            got = decoder.marginal(data, m)
            if t in (0, 7, 31):
                assert_close(got, reference.marginal(cc, data, m, inputs), n)


# -------------------------------------------------------------------------------------------------
# What a session refuses
# -------------------------------------------------------------------------------------------------

def test_gates_checked_and_refused(build_pc):
    pc = gated_pc(build_pc, "pd")
    n = KINDS["pd"]
    data, missing = evidence(n, 2)
    cc = jc.compile(CONSTRAINTS["contains"](), pc)
    gates = make_gates(pc, 2)
    ns = next(iter(gates))
    with pytest.raises(ValueError, match = "shape"):
        cc.decoder(sum_external_params = {**gates, ns: gates[ns][..., None]})
    with pytest.raises(ValueError, match = "dict"):
        cc.decoder(sum_external_params = [gates[ns]])
    plain = build_pc("pd", n, V)
    with pytest.raises(ValueError, match = "no external parameters"):
        jc.compile(CONSTRAINTS["contains"](), plain).decoder(sum_external_params = {next(iter(plain.root_ns.chs)): gates[ns]})
    layer = next(layer for lg in pc.inner_layer_groups if lg.is_sum() for layer in lg.layers
                 if getattr(layer, "external_params", None) is not None and len(layer.nodes) > 1)
    with pytest.raises(NotImplementedError, match = "all or none"):
        cc.decoder(sum_external_params = {k: v for k, v in gates.items() if k is not layer.nodes[0]})
    with cc.decoder(sum_external_params = gates) as decoder:
        with pytest.raises(ValueError, match = "batch of 2"):
            decoder.marginal(data[:1], missing[:1])
    soft = build_pc("pd", n, V, gates = dict(), input_dist = dists.SoftEvidenceCategorical)
    with pytest.raises(ValueError, match = "batch"):                             # evidence of another batch
        jc.compile(CONSTRAINTS["contains"](), soft).decoder(sum_external_params = make_gates(soft, 2),
                                                           categorical_evidence_logp = torch.zeros(3, n, V, device = DEV))


def test_unsupported_gates_refused_at_compile():
    """LowRank sums cannot be compiled under a constraint."""
    x = [inputs(v, num_node_blocks = 2, block_size = 4, dist = dists.Categorical(num_cats = V)) for v in range(2)]
    root = summate(multiply(x[0], x[1]), num_node_blocks = 2, block_size = 4,
                   external_params = LowRankSumParams(rank = 2))
    root = summate(multiply(root), num_node_blocks = 1, block_size = 1)
    root.init_parameters()
    pc = juice.compile(root, verbose = False).to(DEV)
    with pytest.raises(Exception, match = "LowRank"):
        jc.compile(jc.DFA.anything(V), pc)
