"""
`LowRankSumParams`: the multiplicative low-rank scale, both forms, against a MATERIALIZED plain circuit.

Two circuits of one structure:

  A -- its sums carry `LowRankSumParams`, and are given factors `(A_b, B_b)` per sample;
  B -- plain and untied, its parameters set, for ONE sample at a time, to that sample's effective ones

        theta_b[i, j] = theta[i, j] * g_b[i, j] / sum_j' theta[i, j'] * g_b[i, j']

      computed here in float64 from the definition of the form.

B is an ordinary circuit, so its values and flows are pyjuice's own shared-parameter code: A's node values, node
flows, `F+`, `F-` and the factors' gradients on that sample must be B's (the last two by the definitions, in float64,
from B's flows). Nothing of the descriptor's passes is shared with the reference.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import LowRankSumParams


NUM_CATS = 6
FORMS = ("exp_sum", "sum_exp")


# ---------------------------------------------------------------------------------------------- structures

def _ext(external, rank, form, **kw):
    return dict(external_params = LowRankSumParams(rank = rank, form = form, **kw)) if external else dict()


def _hmm(external, rank = 3, form = "exp_sum", seq_length = 4, num_latents = 8, tied = True, tie_plain = False, **kw):
    """An HMM; its transitions carry the factors. `tied`: one transition matrix (the plain twin is untied whatever
    A is, to take every position's effective parameters, unless `tie_plain`)."""
    with juice.set_block_size(num_latents):
        ni = inputs(seq_length - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        sums, cur = [], ni
        for var in range(seq_length - 2, -1, -1):
            cx = ni.duplicate(var, tie_params = True)
            if sums and tied and (external or tie_plain):
                ns = sums[0].duplicate(cur, tie_params = True)
            else:
                ns = summate(cur, num_node_blocks = 1, **_ext(external, rank, form, **kw))
            sums.append(ns)
            cur = multiply(cx, ns)
        root = summate(cur, num_node_blocks = 1, block_size = 1)
    return root, sums


def _sparse(external, rank = 2, form = "exp_sum", **kw):
    """One sum of 3 node blocks of 4 over 4 child blocks of 2, block-sparse and ragged (1, 3 and 2 edge blocks)."""
    with juice.set_block_size(2):
        ins = [inputs(v, num_node_blocks = 4, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(3)]
        prod = multiply(*ins)
    edge_ids = torch.tensor([[0, 1, 1, 1, 2, 2], [2, 0, 1, 3, 0, 3]])
    ns = summate(prod, num_node_blocks = 3, block_size = 4, edge_ids = edge_ids, **_ext(external, rank, form, **kw))
    with juice.set_block_size(4):
        top = multiply(ns, inputs(3, num_node_blocks = 3, dist = dists.Categorical(num_cats = NUM_CATS)))
    root = summate(top, num_node_blocks = 1, block_size = 1)
    return root, [ns]


def _shared_children(external, rank = 2, form = "exp_sum", **kw):
    """Two sums of one layer over the same products: their element flows add."""
    with juice.set_block_size(4):
        ins = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        prod = multiply(*ins)
        n1 = summate(prod, num_node_blocks = 2, **_ext(external, rank, form, **kw))
        n2 = summate(prod, num_node_blocks = 1, **_ext(external, rank, form, **kw))
        i2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
        i3 = inputs(2, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        top = summate(multiply(n1, i2), multiply(n2, i3, edge_ids = torch.zeros(1, 2, dtype = torch.long)),
                      num_node_blocks = 1, block_size = 1)
    return top, [n1, n2]


def _unreached(external, rank = 2, form = "exp_sum", **kw):
    """Two sums of one layer over three child blocks: the first reaches blocks 0 and 1 only, the second 1 and 2."""
    with juice.set_block_size(4):
        ins = [inputs(v, num_node_blocks = 3, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        prod = multiply(*ins)
        n1 = summate(prod, num_node_blocks = 2, edge_ids = torch.tensor([[0, 1], [0, 1]]),
                     **_ext(external, rank, form, **kw))
        n2 = summate(prod, num_node_blocks = 1, edge_ids = torch.tensor([[0, 0], [1, 2]]),
                     **_ext(external, rank, form, **kw))
        i2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
        i3 = inputs(2, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        top = summate(multiply(n1, i2), multiply(n2, i3, edge_ids = torch.zeros(1, 2, dtype = torch.long)),
                      num_node_blocks = 1, block_size = 1)
    return top, [n1, n2]


STRUCTURES = dict(hmm_tied = lambda **kw: _hmm(tied = True, **kw),
                  hmm_untied = lambda **kw: _hmm(tied = False, **kw),
                  sparse = _sparse, shared_children = _shared_children, unreached = _unreached)


def _pair(name, form, seed = 0, **kw):
    """`(pc_a, sums_a, pc_b, sums_b)`: the circuit with factors and its plain twin, the same shared parameters."""
    out = []
    for external in (True, False):
        torch.manual_seed(seed)
        root, sums = STRUCTURES[name](external = external, form = form, **kw)
        torch.manual_seed(seed)
        root.init_parameters(perturbation = 2.0)
        out += [root, sums]
    root_a, sums_a, root_b, sums_b = out
    dev = torch.device("cuda:0")
    pc_a = juice.compile(root_a, verbose = False).to(dev)
    pc_b = juice.compile(root_b, verbose = False).to(dev)
    # The twin's other parameters are A's (tying changes what the same seed draws)
    for la, lb in zip(pc_a.input_layer_group.layers, pc_b.input_layer_group.layers):
        lb.params.copy_(la.params)
    carried = set(map(id, sums_b))
    for na, nb in zip(list(root_a), list(root_b)):
        assert type(na) is type(nb) or (na.is_sum() and nb.is_sum())
        if nb.is_sum() and not nb.is_tied() and id(nb) not in carried:
            _set_params(pc_b, nb, pc_a.get_node_params(_source(na)))
    return pc_a, sums_a, pc_b, sums_b, root_a, root_b


# ---------------------------------------------------------------------------------------------- the reference

def _dense(ns, per_edge):
    """`[E, block_size, ch_block_size]` (as `get_node_params`) on the node vector's `[M, N]` grid, 0 elsewhere."""
    bs, cbs = ns.block_size, ns.ch_block_size
    out = torch.zeros(ns.num_node_blocks, ns.num_ch_node_blocks, bs, cbs, dtype = torch.float64, device = per_edge.device)
    out[ns.edge_ids[0], ns.edge_ids[1]] = per_edge.double()
    return out.permute(0, 2, 1, 3).reshape(ns.num_nodes, ns.num_ch_nodes)


def _per_edge(ns, dense):
    """The inverse of :func:`_dense`: `[E, block_size, ch_block_size]`."""
    bs, cbs = ns.block_size, ns.ch_block_size
    grid = dense.reshape(ns.num_node_blocks, bs, ns.num_ch_node_blocks, cbs).permute(0, 2, 1, 3)
    return grid[ns.edge_ids[0], ns.edge_ids[1]]


def _set_params(pc, ns, per_edge):
    """Write `[E, block_size, ch_block_size]` into the circuit: an edge block holds `[ch_block_size, block_size]`."""
    ps, _ = ns._param_range
    idx = ((ns._param_ids - ps) // (ns.block_size * ns.ch_block_size)).to(per_edge.device)
    pc.params[slice(*ns._param_range)].reshape(-1, ns.ch_block_size, ns.block_size)[idx] = \
        per_edge.permute(0, 2, 1).to(pc.params.dtype)


def _source(ns):
    return ns.get_source_ns() if ns.is_tied() else ns


def _flows_at(pc, ns, flow_range):
    """`[E, block_size, ch_block_size]` of one range of the circuit's parameter flows: it lies as the node vector's
    (source's) parameters do in theirs."""
    bs, cbs = ns.block_size, ns.ch_block_size
    src = _source(ns)
    c = torch.arange(cbs, device = pc.param_flows.device)
    m = torch.arange(bs, device = pc.param_flows.device)
    pf = (src._param_ids - src._param_range[0] + flow_range[0]).to(pc.param_flows.device)[:, None, None] \
        + c[None, :, None] * bs + m[None, None, :]
    return pc.param_flows[pf].permute(0, 2, 1).double()


def _flows(pc, ns):
    """`F+` of the node vector as `[E, block_size, ch_block_size]`, from the circuit's parameter flows."""
    return _flows_at(pc, ns, ns._param_flow_range)


def _scale(form, A, B):
    """`g[i, j]` of one sample in float64, from the definition."""
    A, B = A.double(), B.double()
    if form == "exp_sum":
        return (A @ B).exp()
    return A.exp() @ B.exp()


def _factors(ns, rank, batch, dev, spread = 1.0, seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    A = (torch.randn(batch, ns.num_nodes, rank, generator = g) * spread).to(dev)
    B = (torch.randn(batch, rank, ns.num_ch_nodes, generator = g) * spread).to(dev)
    return A, B


def _reference(pc_a, sums_a, pc_b, sums_b, data, ext, form):
    """
    Sample by sample through the plain circuit at that sample's effective parameters. Returns, per node vector of A,
    `F+` and `F-` on its dense grid summed over the samples (and over tied copies where A ties them), and per node
    vector the gradients of its factors; and the log-likelihoods, node values and node flows of every sample.
    """
    batch = data.size(0)
    theta = [_dense(a, pc_a.get_node_params(_source(a))) for a in sums_a]
    Fp = [torch.zeros_like(t) for t in theta]
    Fm = [torch.zeros_like(t) for t in theta]
    dA = [torch.zeros_like(ext[a][0], dtype = torch.float64) for a in sums_a]
    dB = [torch.zeros_like(ext[a][1], dtype = torch.float64) for a in sums_a]
    lls, mars, flows, every = [], [], [], []
    for b in range(batch):
        eff = []
        for a, bn, t in zip(sums_a, sums_b, theta):
            w = t * _scale(form, ext[a][0][b], ext[a][1][b])
            eff.append(w / w.sum(dim = 1, keepdim = True))
            _set_params(pc_b, bn, _per_edge(bn, eff[-1]))
        lls.append(pc_b(data[b:b + 1]).reshape(-1)[0].double())
        pc_b.init_param_flows(flows_memory = 0.0)
        pc_b.backward(data[b:b + 1], allow_modify_flows = False, logspace_flows = True)
        mars.append([pc_b.node_mars[slice(*bn._output_ind_range), 0].double() for bn in sums_b])
        flows.append([pc_b.node_flows[slice(*bn._output_ind_range), 0].double() for bn in sums_b])
        every.append(pc_b.node_flows[:, 0].double().clone())
        for k, (a, bn, e) in enumerate(zip(sums_a, sums_b, eff)):
            q = _dense(bn, _flows(pc_b, bn))                            # f theta_b c / n of this sample
            t = flows[-1][k].exp()[:, None] * e                         # f theta_b
            Fp[k] += q
            Fm[k] += t
            G = q - t                                                   # dLL / d log g
            Ab, Bb = ext[a][0][b].double(), ext[a][1][b].double()
            if form == "exp_sum":
                dA[k][b] = G @ Bb.t()
                dB[k][b] = Ab.t() @ G
            else:
                R = G / _scale(form, Ab, Bb)                            # dLL / dg
                dA[k][b] = Ab.exp() * (R @ Bb.exp().t())
                dB[k][b] = Bb.exp() * (Ab.exp().t() @ R)
    return dict(Fp = Fp, Fm = Fm, dA = dA, dB = dB, lls = torch.stack(lls), mars = mars, flows = flows, every = every)


def _tie_sum(sums_a, per_ns):
    """What A accumulates per SOURCE node vector: the sum over the copies tied to it."""
    out = dict()
    for a, v in zip(sums_a, per_ns):
        src = _source(a)
        out[src] = out[src] + v if src in out else v.clone()
    return out


def _close(a, b, rtol = 2e-3, atol = 2e-4):
    a, b = a.double(), b.double()
    return bool(((a - b).abs() <= atol + rtol * b.abs()).all())


def _worst(a, b):
    return float((a.double() - b.double()).abs().max())


# ---------------------------------------------------------------------------------------------- tests

@pytest.mark.parametrize("name", sorted(STRUCTURES))
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("batch", [1, 5])
def test_values_flows_and_gradients_are_the_materialized_circuits(name, form, batch):
    rank = 3
    pc_a, sums_a, pc_b, sums_b, root_a, root_b = _pair(name, form, rank = rank)
    dev = pc_a.params.device
    torch.manual_seed(1)
    data = torch.randint(0, NUM_CATS, (batch, pc_a.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 10 + i) for i, a in enumerate(sums_a)}
    grads = {a: (torch.zeros_like(ext[a][0]), torch.zeros_like(ext[a][1])) for a in sums_a}

    lls = pc_a(data, sum_external_params = ext).reshape(-1)
    pc_a.init_param_flows(flows_memory = 0.0)
    pc_a.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)

    ref = _reference(pc_a, sums_a, pc_b, sums_b, data, ext, form)
    assert _close(lls, ref["lls"], rtol = 0, atol = 2e-3), _worst(lls, ref["lls"])

    for k, a in enumerate(sums_a):
        rows = slice(*a._output_ind_range)
        for b in range(batch):
            assert _close(pc_a.node_mars[rows, b], ref["mars"][b][k], rtol = 0, atol = 2e-3), "node values"
            assert _close(pc_a.node_flows[rows, b].exp(), ref["flows"][b][k].exp()), "node flows"
        assert _close(grads[a][0], ref["dA"][k]), f"dA: {_worst(grads[a][0], ref['dA'][k])}"
        assert _close(grads[a][1], ref["dB"][k]), f"dB: {_worst(grads[a][1], ref['dB'][k])}"

    # Every other node's flow too: the inputs' below are the element flows the layer wrote, summed where two of
    # its node vectors share a child
    compared = 0
    for na, nb in zip(list(root_a), list(root_b)):
        if na.is_prod():
            continue
        ra, rb = slice(*na._output_ind_range), slice(*nb._output_ind_range)
        for b in range(batch):
            assert _close(pc_a.node_flows[ra, b].exp(), ref["every"][b][rb].exp()), f"flows of {type(na).__name__}"
        compared += 1
    assert compared > len(sums_a) + 1

    # A child no edge reaches takes no gradient
    for a in sums_a:
        layer = pc_a.external_params_nodes[a]
        geo = a.external_params._geometry(layer, a.external_params._info(layer, a), dev)
        if name == "unreached":
            assert not geo["full"] and int(geo["has"].sum()) == 8
        assert bool((grads[a][1][:, :, ~geo["has"]] == 0).all())

    Fp, Fm = _tie_sum(sums_a, ref["Fp"]), _tie_sum(sums_a, ref["Fm"])
    desc = sums_a[0].external_params
    for src in Fp:
        got = sum(_dense(src, _flows_at(pc_a, src, r)) for r in {a._param_flow_range for a in sums_a if _source(a) is src})
        assert _close(got, Fp[src]), f"F+: {_worst(got, Fp[src])}"
        sources = [(pc_a.external_params_nodes[a], a) for a in sums_a if _source(a) is src]
        fm = desc.materialize_denom_flows(src, pc_a.params, pc_a.denom_param_flows, sources)
        E = src.edge_ids.size(1)
        got = _dense(src, fm.reshape(E, src.ch_block_size, src.block_size).permute(0, 2, 1))
        assert _close(got, Fm[src]), f"F-: {_worst(got, Fm[src])}"


@pytest.mark.parametrize("form", FORMS)
def test_tie_external_shares_one_pair_of_factors(form):
    """With `tie_external` every copy reads the source's tensors, and their gradients add up there."""
    rank, batch = 2, 3
    pc_t, sums_t, pc_b, sums_b, _, _ = _pair("hmm_tied", form, rank = rank, tie_external = True)
    pc_a, sums_a, _, _, _, _ = _pair("hmm_tied", form, rank = rank)
    dev = pc_t.params.device
    torch.manual_seed(2)
    data = torch.randint(0, NUM_CATS, (batch, pc_t.num_vars), device = dev)
    A, B = _factors(sums_t[0], rank, batch, dev)

    gt = (torch.zeros_like(A), torch.zeros_like(B))
    lls_t = pc_t(data, sum_external_params = {sums_t[0]: (A, B)})
    pc_t.init_param_flows(flows_memory = 0.0)
    pc_t.backward(data, allow_modify_flows = False, logspace_flows = True,
                  sum_external_params_grad = {sums_t[0]: gt})

    ext = {a: (A, B) for a in sums_a}
    ga = {a: (torch.zeros_like(A), torch.zeros_like(B)) for a in sums_a}
    lls_a = pc_a(data, sum_external_params = ext)
    pc_a.init_param_flows(flows_memory = 0.0)
    pc_a.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = ga)

    assert _close(lls_t, lls_a, rtol = 0, atol = 1e-4)
    assert _close(gt[0], sum(ga[a][0] for a in sums_a), rtol = 1e-4, atol = 1e-5)
    assert _close(gt[1], sum(ga[a][1] for a in sums_a), rtol = 1e-4, atol = 1e-5)
    assert _close(pc_t.param_flows, pc_a.param_flows, rtol = 1e-4, atol = 1e-5)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("rescale", [False, True], ids = ["mini_batch", "anemone"])
def test_constant_factors_leave_the_standard_em_step(form, rescale):
    """With a scale that is one constant for every pair, `theta_b = theta`, `F- = theta * sum_j F+`, and the
    conditional M-step is the standard one -- the top-down term of the rescaled step included."""
    rank, batch = 2, 16
    pc_a, sums_a, pc_b, sums_b, _, _ = _pair("hmm_tied", form, rank = rank, tie_plain = True)
    dev = pc_a.params.device
    torch.manual_seed(3)
    data = torch.randint(0, NUM_CATS, (batch, pc_a.num_vars), device = dev)
    ext = {a: (torch.zeros(batch, a.num_nodes, rank, device = dev), torch.zeros(batch, rank, a.num_ch_nodes, device = dev))
           for a in sums_a}

    for pc, kw in ((pc_a, dict(sum_external_params = ext)), (pc_b, dict())):
        pc(data, **kw)
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True)
        pc.mini_batch_em(step_size = 0.3, pseudocount = 0.01, step_size_rescaling = rescale)

    assert _close(pc_a.get_node_params(sums_a[0]), pc_b.get_node_params(sums_b[0]), rtol = 1e-3, atol = 1e-6)
    assert _close(pc_a.params, pc_b.params, rtol = 1e-3, atol = 1e-6)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("step_size,pseudocount", [(1.0, 0.0), (0.4, 0.05)])
def test_the_conditional_m_step(form, step_size, pseudocount):
    """`theta <- normalize(theta * F+ / F-)` in the damped MAP form, from the reference's `F+` and `F-`."""
    rank, batch = 3, 6
    pc_a, sums_a, pc_b, sums_b, _, _ = _pair("sparse", form, rank = rank)
    dev = pc_a.params.device
    torch.manual_seed(4)
    data = torch.randint(0, NUM_CATS, (batch, pc_a.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 20 + i) for i, a in enumerate(sums_a)}

    pc_a(data, sum_external_params = ext)
    pc_a.init_param_flows(flows_memory = 0.0)
    pc_a.backward(data, allow_modify_flows = False, logspace_flows = True)
    ref = _reference(pc_a, sums_a, pc_b, sums_b, data, ext, form)
    ns = sums_a[0]
    theta = _dense(ns, pc_a.get_node_params(ns))
    pc_a.mini_batch_em(step_size = step_size, pseudocount = pseudocount)

    has = _dense(ns, torch.ones(ns.edge_ids.size(1), ns.block_size, ns.ch_block_size, device = dev)) > 0
    K = has.sum(dim = 1, keepdim = True).double()
    ratio = torch.where(has, (ref["Fp"][0] + pseudocount / K) / (ref["Fm"][0] + pseudocount * theta).clamp_min(1e-38),
                        torch.zeros_like(theta))
    cum = (1.0 - step_size) + step_size * (theta * ratio).sum(dim = 1, keepdim = True)
    want = theta * ((1.0 - step_size) + step_size * ratio) / cum
    got = _dense(ns, pc_a.get_node_params(ns))
    assert _close(got, want, rtol = 3e-3, atol = 1e-5), _worst(got, want)
    assert _close(got.sum(dim = 1), torch.ones_like(got.sum(dim = 1)), rtol = 0, atol = 1e-5)


@pytest.mark.parametrize("form", FORMS)
def test_children_without_an_edge_read_as_absent(form):
    """A child block none of a node vector's edges reach has no element of its own in the gathers: whatever stands
    there is masked to -inf, so it enters no maximum, no shift and no sum -- here an enormous value."""
    pc_a, sums_a, _, _, _, _ = _pair("unreached", form, rank = 2)
    dev = pc_a.params.device
    desc, a = sums_a[0].external_params, sums_a[0]
    layer = pc_a.external_params_nodes[a]
    geo = desc._geometry(layer, desc._info(layer, a), dev)
    torch.manual_seed(5)
    fake = torch.randn(pc_a.num_elements, 3, device = dev)
    lc = desc._child_values(geo, fake)
    assert lc.shape == (3, a.num_ch_nodes)
    assert bool(torch.isneginf(lc[:, ~geo["has"]]).all())
    assert torch.equal(lc[:, geo["has"]], fake[geo["kids"]].t())

    # ... and through a pass: the same node values whatever the stand-in element holds
    A, B = _factors(a, 2, 3, dev)
    theta = desc._dense_params(geo, pc_a.params)
    values = []
    for stand_in in (0.0, 5000.0):
        fake[int(geo["eid"][~geo["has"]][0])] = stand_in
        lc = desc._child_values(geo, fake)
        ln, lz = desc._forward_exp_sum(A, B, theta.log(), lc) if form == "exp_sum" else \
            desc._forward_sum_exp(A, B, theta.double(), lc)
        values.append(ln)
    assert bool(torch.isfinite(values[0]).all()) and torch.equal(values[0], values[1])


@pytest.mark.parametrize("form", FORMS)
def test_top_down_term_reaches_both_flows(form):
    """The rescaled step's top-down pass adds `scale * P_td[n] * theta[n, c]` to `F+`; the same has to reach `F-`,
    kept here per node and multiplied by the parameters when `F-` is read. Factors that vary: with constant ones
    any denominator of the form `theta` times a factor per node gives the same update."""
    from pyjuice.model.backend.top_down_prob import eval_top_down_probs

    rank, batch = 2, 8
    pc_a, sums_a, _, _, _, _ = _pair("hmm_tied", form, rank = rank)
    dev = pc_a.params.device
    torch.manual_seed(6)
    data = torch.randint(0, NUM_CATS, (batch, pc_a.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 30 + i) for i, a in enumerate(sums_a)}
    pc_a(data, sum_external_params = ext)
    pc_a.backward(data, allow_modify_flows = False, logspace_flows = True, flows_memory = 0.0)

    pc_a.param_flows[:] = 0.0
    pc_a.denom_param_flows[:] = 0.0
    eval_top_down_probs(pc_a, update_pflow = True, scale = 0.25, use_cudagraph = False)

    src = _source(sums_a[0])
    desc = src.external_params
    fp = sum(_dense(src, _flows_at(pc_a, src, r)) for r in {a._param_flow_range for a in sums_a})
    sources = [(pc_a.external_params_nodes[a], a) for a in sums_a]
    fm = desc.materialize_denom_flows(src, pc_a.params, pc_a.denom_param_flows, sources)
    fm = _dense(src, fm.reshape(src.edge_ids.size(1), src.ch_block_size, src.block_size).permute(0, 2, 1))
    assert float(fp.sum()) > 0.1, "the top-down pass wrote nothing"
    assert _close(fm, fp, rtol = 1e-5, atol = 1e-9), _worst(fm, fp)


def test_a_batch_in_chunks_is_the_batch(monkeypatch):
    """`exp_sum` holds `[samples, M, N]` for a bounded number of cells and takes a larger batch in chunks of
    samples: the same values, flows and gradients, here two samples at a time with a last chunk of one."""
    import pyjuice.nodes.external_params.lowrank as lowrank

    rank, batch = 3, 7
    pc_a, sums_a, _, _, _, _ = _pair("sparse", "exp_sum", rank = rank)
    dev = pc_a.params.device
    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, (batch, pc_a.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 40 + i) for i, a in enumerate(sums_a)}

    def run():
        grads = {a: (torch.zeros_like(ext[a][0]), torch.zeros_like(ext[a][1])) for a in sums_a}
        lls = pc_a(data, sum_external_params = ext).clone()
        pc_a.backward(data, allow_modify_flows = False, logspace_flows = True, flows_memory = 0.0,
                      sum_external_params_grad = grads)
        return lls, pc_a.param_flows.clone(), pc_a.denom_param_flows.clone(), pc_a.node_flows.clone(), grads

    whole = run()
    a = sums_a[0]
    monkeypatch.setattr(lowrank, "LOWRANK_CHUNK_CELLS", 2 * a.num_nodes * a.num_ch_nodes)
    assert a.external_params._chunks(batch, a.num_nodes * a.num_ch_nodes) == [(0, 2), (2, 4), (4, 6), (6, 7)]
    chunked = run()
    for x, y in zip(whole[:4], chunked[:4]):
        assert torch.allclose(x, y, rtol = 1e-5, atol = 1e-6, equal_nan = True)
    for k in sums_a:
        assert torch.allclose(whole[4][k][0], chunked[4][k][0], rtol = 1e-5, atol = 1e-6)
        assert torch.allclose(whole[4][k][1], chunked[4][k][1], rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("form", FORMS)
def test_without_the_denominator_flow(form):
    """`apply_z_correction = False`: no buffer, nothing accumulated, the plain `normalize(F+)` step -- and the
    same values, flows and gradients as with it."""
    rank, batch = 2, 6
    pc_on, sums_on, _, _, _, _ = _pair("sparse", form, rank = rank)
    pc_off, sums_off, _, _, _, _ = _pair("sparse", form, rank = rank, apply_z_correction = False)
    dev = pc_on.params.device
    assert pc_off.denom_param_flows is None and pc_on.denom_param_flows is None      # (allocated with the flows)
    torch.manual_seed(8)
    data = torch.randint(0, NUM_CATS, (batch, pc_on.num_vars), device = dev)
    A, B = _factors(sums_on[0], rank, batch, dev)
    out = []
    for pc, ns in ((pc_on, sums_on[0]), (pc_off, sums_off[0])):
        g = (torch.zeros_like(A), torch.zeros_like(B))
        lls = pc(data, sum_external_params = {ns: (A, B)}).clone()
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, flows_memory = 0.0,
                    sum_external_params_grad = {ns: g})
        out.append((lls, pc.param_flows.clone(), g))
    assert pc_on.denom_param_flows is not None and float(pc_on.denom_param_flows.abs().sum()) > 0
    assert getattr(pc_off, "denom_param_flows", None) is None
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])
    assert torch.equal(out[0][2][0], out[1][2][0]) and torch.equal(out[0][2][1], out[1][2][1])

    before = _dense(sums_off[0], pc_off.get_node_params(sums_off[0]))
    flows = _dense(sums_off[0], _flows(pc_off, sums_off[0]))
    pc_off.mini_batch_em(step_size = 1.0, pseudocount = 0.0)
    after = _dense(sums_off[0], pc_off.get_node_params(sums_off[0]))
    assert _close(after, flows / flows.sum(dim = 1, keepdim = True), rtol = 1e-4, atol = 1e-6)
    assert not _close(after, before)


def test_refusals():
    with pytest.raises(AssertionError):
        LowRankSumParams(rank = 0)
    with pytest.raises(AssertionError):
        LowRankSumParams(rank = 2, form = "additive")
    assert LowRankSumParams(rank = 2).form == "exp_sum"
    assert LowRankSumParams(rank = 2, form = "sum_exp") != LowRankSumParams(rank = 2)
    assert LowRankSumParams(rank = 2).requests_denom_param_flows
    assert not LowRankSumParams(rank = 2, apply_z_correction = False).requests_denom_param_flows

    pc_a, sums_a, _, _, _, _ = _pair("shared_children", "exp_sum", rank = 2)
    dev = pc_a.params.device
    data = torch.randint(0, NUM_CATS, (2, pc_a.num_vars), device = dev)
    with pytest.raises(NotImplementedError, match = "Supply all"):
        pc_a(data, sum_external_params = {sums_a[0]: _factors(sums_a[0], 2, 2, dev)})
