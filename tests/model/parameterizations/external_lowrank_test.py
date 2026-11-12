"""
`LowRank`: the multiplicative low-rank scale, both forms, against a MATERIALIZED plain circuit.

Two circuits of one structure:

  A -- its sums carry `LowRank`, and are given factors `(A_b, B_b)` per sample;
  B -- plain and untied, its parameters set, for ONE sample at a time, to that sample's effective ones

        theta_b[i, j] = theta[i, j] * g_b[i, j] / sum_j' theta[i, j'] * g_b[i, j']

      computed here in float64 from the definition of the form.

B is an ordinary circuit, so its values and flows are pyjuice's own shared-parameter code: A's node values, node
flows, `F+`, `F-` and the factors' gradients on that sample must be B's (the last two by the definitions, in float64,
from B's flows). Nothing of the descriptor's passes is shared with the reference.
"""

import pytest
import math
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.parameterizations import LowRank, BlockScale


NUM_CATS = 6
FORMS = ("exp_sum", "sum_exp")
OTHER_KIND = dict(apply_z_correction = False)       # (another parameterization: a layer of its own)


# ---------------------------------------------------------------------------------------------- structures

def _ext(external, rank, form, **kw):
    return dict(parameterization = LowRank(rank = rank, form = form, **kw)) if external else dict()


def _hmm(external, rank = 3, form = "exp_sum", seq_length = 4, num_latents = 8, tied = True, tie_plain = False,
         ranks = None, **kw):
    """An HMM; its transitions carry the factors. `tied`: one transition matrix (the plain twin is untied whatever
    A is, to take every position's effective parameters, unless `tie_plain`). `ranks`: untied, the transitions'
    ranks from the deepest up."""
    with juice.set_block_size(num_latents):
        ni = inputs(seq_length - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        sums, cur = [], ni
        for var in range(seq_length - 2, -1, -1):
            cx = ni.duplicate(var, tie_params = True)
            if sums and tied and (external or tie_plain):
                ns = sums[0].duplicate(cur, tie_params = True)
            else:
                ns = summate(cur, num_node_blocks = 1,
                             **_ext(external, rank if ranks is None else ranks[len(sums)], form, **kw))
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


def _two_disjoint(external, rank = 2, form = "exp_sum", **kw):
    """Two sums of one layer, each over products of its own: of 2 and 1 node blocks, the second ragged."""
    with juice.set_block_size(4):
        ins = [inputs(v, num_node_blocks = 3, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(2)]
        p1, p2 = multiply(*ins), multiply(ins[0], ins[1].duplicate(1, tie_params = False))
        n1 = summate(p1, num_node_blocks = 2, **_ext(external, rank, form, **kw))
        n2 = summate(p2, num_node_blocks = 2, edge_ids = torch.tensor([[0, 0, 1], [0, 2, 1]]),
                     **_ext(external, rank, form, **kw))
        i2 = inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))
        top = summate(multiply(n1, i2), multiply(n2, i2.duplicate(2, tie_params = False)),
                      num_node_blocks = 1, block_size = 1)
    return top, [n1, n2]


def _two_chains(external, rank = 2, form = "exp_sum", seq_length = 4, num_latents = 4, second = None, **kw):
    """Two tied HMMs side by side, over variables of their own: every layer holds a transition of each, every
    transition tied to its chain's others. `second`: what the second chain's factors are built with besides -- another
    parameterization then, so every layer group holds a layer of each chain."""
    tops, sums = [], []
    with juice.set_block_size(num_latents):
        for chain in range(2):
            first = chain * seq_length
            ni = inputs(first + seq_length - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
            mine, cur = [], ni
            for var in range(seq_length - 2, -1, -1):
                cx = ni.duplicate(first + var, tie_params = True)
                ns = mine[0].duplicate(cur, tie_params = True) if mine and external else \
                    summate(cur, num_node_blocks = 1, **_ext(external, rank, form, **dict(kw, **(second or {}) if chain else {})))
                mine.append(ns)
                cur = multiply(cx, ns)
            tops.append(summate(cur, num_node_blocks = 1))
            sums += mine
        root = summate(multiply(*tops), num_node_blocks = 1, block_size = 1)
    return root, sums


def _hmm_side(external, rank = 2, form = "exp_sum", seq_length = 4, num_latents = 8, side = None, side_first = True,
              **kw):
    """A tied HMM with a plain sum over one more variable beside it, joined at the top: that sum's product stands in
    the lowest product layer, before the chain's, so the deepest transition's children are other element rows than
    the other transitions'. `side`: that sum carries factors too, built with these besides (another
    parameterization: a layer of its own in the deepest transition's layer group), and is the last of the sums
    returned; `side_first`: built before the chain or after it."""
    def beside():
        return summate(multiply(inputs(seq_length, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))),
                       num_node_blocks = 1, **(_ext(external, rank, form, **dict(kw, **side)) if side is not None else {}))

    with juice.set_block_size(num_latents):
        other = beside() if side_first else None
        ni = inputs(seq_length - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        sums, cur = [], ni
        for var in range(seq_length - 2, -1, -1):
            cx = ni.duplicate(var, tie_params = True)
            ns = sums[0].duplicate(cur, tie_params = True) if sums and external else \
                summate(cur, num_node_blocks = 1, **_ext(external, rank, form, **kw))
            sums.append(ns)
            if var == 0:
                other = other if side_first else beside()
                cur = multiply(other, cx, ns)
            else:
                cur = multiply(cx, ns)
        root = summate(cur, num_node_blocks = 1, block_size = 1)
    return root, sums + ([other] if side is not None else [])


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


def _fan_in(external, rank = 2, form = "exp_sum", **kw):
    """Two sums of one layer over products of their own, and above them ONE sum over products of each: its children's
    children lie in both."""
    with juice.set_block_size(4):
        a, b = ([inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(3)]
                for _ in range(2))
        n1 = summate(multiply(a[0], a[1]), num_node_blocks = 2, **_ext(external, rank, form, **kw))
        n2 = summate(multiply(b[0], b[1]), num_node_blocks = 2, **_ext(external, rank, form, **kw))
        top = summate(multiply(n1, a[2]), multiply(n2, b[2]), num_node_blocks = 2,
                      **_ext(external, rank, form, **kw))
        root = summate(multiply(top, inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS))),
                       num_node_blocks = 1, block_size = 1)
    return root, [n1, n2, top]


def _fan_out(external, rank = 2, form = "exp_sum", **kw):
    """One sum, and above it TWO sums of one layer, each over products of it of its own."""
    with juice.set_block_size(4):
        ins = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS)) for v in range(4)]
        low = summate(multiply(ins[0], ins[1]), num_node_blocks = 2, **_ext(external, rank, form, **kw))
        n1 = summate(multiply(low, ins[2]), num_node_blocks = 2, **_ext(external, rank, form, **kw))
        n2 = summate(multiply(low, ins[3]), num_node_blocks = 2, **_ext(external, rank, form, **kw))
        root = summate(multiply(n1, ins[3].duplicate(3, tie_params = False)),
                       multiply(n2, ins[2].duplicate(2, tie_params = False)), num_node_blocks = 1, block_size = 1)
    return root, [low, n1, n2]


def _beside(external, rank = 2, form = "exp_sum", seq_length = 4, num_latents = 8, flip = False, **kw):
    """A tied HMM and, at the depth of its middle transition, a sum with factors of ANOTHER parameterization over a
    plain sum of two more variables: a layer of its own in that transition's layer group, over children no layer with
    factors computes. `flip`: the two joined at the top in the other order."""
    with juice.set_block_size(num_latents):
        ni = inputs(seq_length - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
        sums, cur = [], ni
        for var in range(seq_length - 2, -1, -1):
            cx = ni.duplicate(var, tie_params = True)
            ns = sums[0].duplicate(cur, tie_params = True) if sums and external else \
                summate(cur, num_node_blocks = 1, **_ext(external, rank, form, **kw))
            sums.append(ns)
            cur = multiply(cx, ns)
        chain = summate(cur, num_node_blocks = 1)
        extra = [inputs(seq_length + v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
                 for v in range(3)]
        low = summate(multiply(extra[0], extra[1]), num_node_blocks = 1)
        side = summate(multiply(low, extra[2]), num_node_blocks = 1, **_ext(external, rank, form, **dict(kw, **OTHER_KIND)))
        root = summate(multiply(side, chain) if flip else multiply(chain, side), num_node_blocks = 1, block_size = 1)
    return root, sums + [side]


STRUCTURES = dict(hmm_tied = lambda **kw: _hmm(tied = True, **kw),
                  hmm_untied = lambda **kw: _hmm(tied = False, **kw),
                  sparse = _sparse, shared_children = _shared_children, unreached = _unreached,
                  two_disjoint = _two_disjoint, two_chains = _two_chains, hmm_side = _hmm_side, fan_in = _fan_in,
                  fan_out = _fan_out)


def _pair(name, form, seed = 0, prepare = None, precision = "auto", **kw):
    """`(pc_a, sums_a, pc_b, sums_b)`: the circuit with factors and its plain twin, the same shared parameters.
    `name`: a structure's, or its builder. `prepare(root)`: applied to both before they are compiled. `precision`:
    both circuits'."""
    out = []
    for external in (True, False):
        torch.manual_seed(seed)
        root, sums = (STRUCTURES[name] if isinstance(name, str) else name)(external = external, form = form, **kw)
        torch.manual_seed(seed)
        root.init_parameters(perturbation = 2.0)
        if prepare is not None:
            prepare(root)
        out += [root, sums]
    root_a, sums_a, root_b, sums_b = out
    dev = torch.device("cuda:0")
    pc_a = juice.compile(root_a, verbose = False, precision = precision).to(dev)
    pc_b = juice.compile(root_b, verbose = False, precision = precision).to(dev)
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
    # (exp_sum at `precision = "fp32"`: the semantics; its default precision's bf16 products against the torch passes
    # in the fused tests below)
    rank = 3
    pc_a, sums_a, pc_b, sums_b, root_a, root_b = _pair(name, form, rank = rank,
                                                       precision = "fp32" if form == "exp_sum" else "auto")
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
        geo = a.parameterization._geometry(layer, a.parameterization._info(layer, a), dev)
        if name == "unreached":
            assert not geo["full"] and int(geo["has"].sum()) == 8
        assert bool((grads[a][1][:, :, ~geo["has"]] == 0).all())

    Fp, Fm = _tie_sum(sums_a, ref["Fp"]), _tie_sum(sums_a, ref["Fm"])
    desc = sums_a[0].parameterization
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
    desc, a = sums_a[0].parameterization, sums_a[0]
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
    desc = src.parameterization
    fp = sum(_dense(src, _flows_at(pc_a, src, r)) for r in {a._param_flow_range for a in sums_a})
    sources = [(pc_a.external_params_nodes[a], a) for a in sums_a]
    fm = desc.materialize_denom_flows(src, pc_a.params, pc_a.denom_param_flows, sources)
    fm = _dense(src, fm.reshape(src.edge_ids.size(1), src.ch_block_size, src.block_size).permute(0, 2, 1))
    assert float(fp.sum()) > 0.1, "the top-down pass wrote nothing"
    assert _close(fm, fp, rtol = 1e-5, atol = 1e-9), _worst(fm, fp)


def test_a_batch_in_chunks_is_the_batch(monkeypatch):
    """`exp_sum` holds `[samples, M, N]` for a bounded number of cells and takes a larger batch in chunks of
    samples: the same values, flows and gradients, here two samples at a time with a last chunk of one."""
    import pyjuice.parameterizations.lowrank.forward as lowrank_forward

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
    monkeypatch.setattr(lowrank_forward, "LOWRANK_CHUNK_CELLS", 2 * a.num_nodes * a.num_ch_nodes)
    assert a.parameterization._chunks(batch, a.num_nodes * a.num_ch_nodes) == [(0, 2), (2, 4), (4, 6), (6, 7)]
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


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("supplied", ["none", "the_corrected", "the_other"])
def test_a_layer_without_the_denominator_flow_beside_one_with_it(form, supplied):
    """Two chains, the first's factors with the denominator flow and the second's without: the circuit has the buffer,
    the second chain's layers no part of it. A backward that supplies nothing for a layer accumulates that layer's
    denominator only where its parameterization keeps one (it raised for the other before: the buffer was taken for
    every layer's); the first chain's `F-` then has its `F+`'s sum per node -- the node's flow -- supplied or not,
    and the conditional step leaves normalized, finite parameters."""
    rank, batch = 2, 6
    pc, sums, _, _, _, _ = _pair("two_chains", form, rank = rank, second = OTHER_KIND)
    dev = pc.params.device
    first = [a for a in sums if a.parameterization.apply_z_correction]
    second = [a for a in sums if not a.parameterization.apply_z_correction]
    assert len(first) == len(second) == 3
    torch.manual_seed(13)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    given = dict(none = [], the_corrected = first, the_other = second)[supplied]
    ext = _own_factors(given, batch, dev, 300)
    lls = pc(data, **(dict(sum_external_params = ext) if ext else {}))
    pc.init_param_flows(flows_memory = 0.0)
    pc.backward(data, allow_modify_flows = False, logspace_flows = True)
    assert torch.isfinite(lls).all()

    src = _source(first[0])
    sources = [(pc.external_params_nodes[a], a) for a in first]
    fm = src.parameterization.materialize_denom_flows(src, pc.params, pc.denom_param_flows, sources)
    E = src.edge_ids.size(1)
    fm = _dense(src, fm.reshape(E, src.ch_block_size, src.block_size).permute(0, 2, 1))
    ranges = {tuple(a._param_flow_range): a for a in first}          # (tied copies may share a range of the flows)
    fp = _dense(src, sum(_flows(pc, a) for a in ranges.values()))
    assert float(fp.sum()) > 0 and _close(fm.sum(dim = 1), fp.sum(dim = 1), rtol = 1e-4, atol = 1e-5)
    assert all(not hasattr(pc.external_params_nodes[a], "denom_flow_slices") for a in second)

    pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    for a in (first[0], second[0]):
        theta = _dense(_source(a), pc.get_node_params(_source(a)))
        assert torch.isfinite(theta).all() and _close(theta.sum(dim = 1), torch.ones_like(theta.sum(dim = 1)), rtol = 1e-5)


# ---------------------------------------------------------------- the low-rank scale that is a BlockScale gate

GATE_T, GATE_L, GATE_BS = 5, 16, 4
GATE_NK = GATE_L // GATE_BS


def _gated_chain(desc, precision = "auto"):
    """A tied HMM of 16 latents in blocks of 4, its transitions carrying `desc`, registered as the group "tr"."""
    torch.manual_seed(0)
    trans = []
    with juice.set_block_size(GATE_BS):
        ni = inputs(GATE_T - 1, num_node_blocks = GATE_NK, dist = dists.Categorical(num_cats = NUM_CATS))
        src, cur = None, ni
        for var in range(GATE_T - 2, -1, -1):
            cx = ni.duplicate(var, tie_params = True)
            ns = summate(cur, num_node_blocks = GATE_NK, parameterization = desc) if src is None \
                else src.duplicate(cur, tie_params = True)
            src = src or ns
            trans.append(ns)
            cur = multiply(cx, ns)
        root = summate(cur, num_node_blocks = 1, block_size = 1)
    torch.manual_seed(0)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, precision = precision).to(torch.device("cuda:0"))
    pc.register_external_params_group("tr", trans[::-1])
    return pc, src


def _gate_as_factors(gate, form):
    """A BlockScale log gate `[B, n * NK, CK]` as low-rank factors of rank NK with the same scale: `A` picks a
    node's block (its one-hot; in the mixture form its log), `B` holds the block's gate at every child."""
    batch, n = gate.size(0), gate.size(1) // GATE_NK
    onehot = torch.eye(GATE_NK, device = gate.device).repeat_interleave(GATE_BS, dim = 0)        # [L, NK]
    A = onehot if form == "exp_sum" else torch.where(onehot > 0, 0.0, -60.0)
    A = A[None, None].expand(batch, n, GATE_L, GATE_NK).reshape(batch, n * GATE_L, GATE_NK)
    B = gate.view(batch, n, GATE_NK, GATE_NK).repeat_interleave(GATE_BS, dim = 3)                # a child's block
    return A.contiguous(), B.reshape(batch, n * GATE_NK, GATE_L).contiguous()


@pytest.mark.parametrize("form,virtual", [("exp_sum", False), ("sum_exp", False), ("sum_exp", True)])
@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_a_blockscale_gate_as_factors_trains_as_blockscale(form, virtual, momentum, monkeypatch):
    """A low-rank scale of rank = the node blocks can BE a BlockScale gate: the same effective parameters per
    sample, so the same training -- the shared parameters after every Anemone step, with momentum too. The momentum
    is what tells how the denominator flow is kept: averaged over steps as `F-`, it carries the parameters of past
    steps (3e-2 from BlockScale's after 16 steps, measured); kept free of them, as BlockScale's is, it is the same
    update. The gate's gradient is the factors', summed over a gate's children. (exp_sum at `precision = "fp32"`: its
    default precision's bf16 products are within their tolerance of the torch passes, not of BlockScale's gradient's.)"""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    if virtual and not _virtual_available():
        pytest.skip("the low-rank mixture's CUDA extension is not built")
    monkeypatch.setattr(lowrank_virtual, "ENABLED", virtual)
    batch = 32
    pc_g, src_g = _gated_chain(BlockScale(block_size = GATE_BS, ch_block_size = GATE_BS))
    pc_f, src_f = _gated_chain(LowRank(rank = GATE_NK, form = form), precision = "fp32" if form == "exp_sum" else "auto")
    assert torch.equal(pc_g.params, pc_f.params)
    dev = pc_g.params.device
    opts = [juice.optim.Anemone(pc, step_size = 0.5, momentum = momentum, niters_per_update = 1, pseudocount = 1e-6)
            for pc in (pc_g, pc_f)]
    g = torch.Generator(device = "cpu").manual_seed(1)
    for step in range(8):
        x = torch.randint(0, NUM_CATS, (batch, GATE_T), generator = g).to(dev)
        gate = torch.log_softmax(1.5 * torch.randn(batch, (GATE_T - 1) * GATE_NK, GATE_NK, generator = g), -1).to(dev)
        gate = gate.contiguous()
        A, B = _gate_as_factors(gate, form)
        grad_g, grad_f = torch.zeros_like(gate), (torch.zeros_like(A), torch.zeros_like(B))
        ll_g = pc_g(x, sum_external_params = {"tr": gate})
        ll_f = pc_f(x, sum_external_params = {"tr": (A, B)})
        assert torch.allclose(ll_g, ll_f, rtol = 0, atol = 5e-3), f"step {step}: {float((ll_g - ll_f).abs().max())}"
        pc_g.backward(x, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = {"tr": grad_g})
        pc_f.backward(x, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = {"tr": grad_f})
        assert type(pc_f.external_params_nodes[src_f].__dict__["_lr_state"]) is (tuple if virtual else dict)
        # d LL / d log gate[block, child block] = the sum of d LL / d B[block, child] over the block's children
        summed = grad_f[1].view(batch, GATE_T - 1, GATE_NK, GATE_NK, GATE_BS).sum(-1).reshape(gate.shape)
        assert float((summed - grad_g).abs().max()) <= 3e-3 * float(grad_g.abs().max()), f"step {step}: gradient"
        for opt in opts:
            opt.step()
        ps, pe = src_g._param_range
        worst = float((pc_g.params[ps:pe] - pc_f.params[ps:pe]).abs().max())
        assert worst < 2e-5, f"step {step + 1}: the shared parameters {worst} apart"


@pytest.mark.parametrize("form", FORMS)
def test_the_denominator_statistic_is_free_of_the_parameters(form):
    """What the backward accumulates is `W = F- / theta`: read with other parameters, `F-` is those times the same
    `W` -- the parameters of the pass that made it are not in it."""
    rank, batch = 3, 6
    pc, sums, _, _, _, _ = _pair("sparse", form, rank = rank)
    dev = pc.params.device
    torch.manual_seed(21)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 150 + i) for i, a in enumerate(sums)}
    _passes(pc, sums, data, ext)
    a = sums[0]
    desc, sources = a.parameterization, [(pc.external_params_nodes[a], a)]
    ps, pe = a._param_range
    before = desc.materialize_denom_flows(a, pc.params, pc.denom_param_flows, sources).clone()
    other = pc.params.clone()
    other[ps:pe] *= torch.rand(pe - ps, device = dev) + 0.5
    after = desc.materialize_denom_flows(a, other, pc.denom_param_flows, sources)
    assert float(before.sum()) > 1.0
    assert torch.allclose(after / other[ps:pe], before / pc.params[ps:pe], rtol = 1e-5, atol = 1e-7)
    assert not torch.allclose(after, before, rtol = 1e-2)


# ---------------------------------------------------------------- "sum_exp" on the virtual batch (`lowrank.virtual`)

def _virtual_available():
    from pyjuice.parameterizations.lowrank.kernels.c import get_lowrank_module
    return torch.cuda.is_available() and get_lowrank_module() is not None


needs_virtual = pytest.mark.skipif(not _virtual_available(), reason = "the low-rank mixture's CUDA extension is not built")


def _passes(pc, sums, data, ext, param_flows = True):
    """One forward and backward with gradient buffers: everything they leave. The buffers start as NaN: whatever the
    backward does not write -- a destination it was handed unzeroed (`StagedExternalParams.fresh`) -- shows."""
    grads = {a: (torch.full_like(ext[a][0], float("nan")), torch.full_like(ext[a][1], float("nan"))) for a in sums}
    lls = pc(data, sum_external_params = ext).clone()
    pc.init_param_flows(flows_memory = 0.0)
    pc.backward(data, allow_modify_flows = False, logspace_flows = True, compute_param_flows = param_flows,
                sum_external_params_grad = grads)
    path = {type(pc.external_params_nodes[a].__dict__["_lr_state"]) for a in sums}
    return dict(lls = lls, mars = pc.node_mars.clone(), flows = pc.node_flows.clone(), fp = pc.param_flows.clone(),
                fm = _denominators(pc, sums), dA = [grads[a][0] for a in sums], dB = [grads[a][1] for a in sums],
                path = path)


def _denominators(pc, sums):
    """`F-` of every source node vector as the M-step reads it: summed over its copies, whichever of their slices
    of the buffer holds what."""
    out = []
    for src in dict.fromkeys(_source(a) for a in sums):
        desc = src.parameterization
        if desc.apply_z_correction:
            sources = [(pc.external_params_nodes[a], a) for a in sums if _source(a) is src]
            out.append(desc.materialize_denom_flows(src, pc.params, pc.denom_param_flows, sources).clone())
    return torch.cat(out)


def _same(x, y, rtol, atol, grad_scale = None):
    """Everything two runs left alike, to `(rtol, atol)`; the gradients, given `grad_scale`, to that share of their
    largest entry instead (bf16 products: each term ~2^-8 off, the sums over children cancelling)."""
    for k in ("lls", "mars", "fp", "fm"):
        assert torch.allclose(x[k], y[k], rtol = rtol, atol = atol, equal_nan = True), \
            f"{k}: {float((x[k] - y[k])[torch.isfinite(x[k]) & torch.isfinite(y[k])].abs().max())}"
    assert torch.allclose(x["flows"].exp(), y["flows"].exp(), rtol = rtol, atol = atol), "node flows"
    for k in ("dA", "dB"):
        for a, b in zip(x[k], y[k]):
            if grad_scale is None:
                assert torch.allclose(a, b, rtol = rtol, atol = max(atol, 1e-6)), f"{k}: {float((a - b).abs().max())}"
            else:
                worst, top = float((a - b).abs().max()), float(b.abs().max())
                assert worst <= grad_scale * top + 1e-6, f"{k}: {worst} against {top}"


#: The fused exp_sum kernels' tolerances against the torch passes: `precision = "fp32"` (three-term TF32) as fp32;
#: the default (bf16 products) the same but for the gradients, a share of their largest entry
FUSED_TOL = {"fp32": dict(rtol = 2e-4, atol = 2e-5), "auto": dict(rtol = 2e-4, atol = 2e-5, grad_scale = 2e-2)}


@needs_virtual
@pytest.mark.parametrize("name,served", [("hmm_tied", True), ("hmm_untied", True), ("sparse", True),
                                         ("two_disjoint", True), ("two_chains", True), ("hmm_side", True),
                                         ("fan_in", True), ("fan_out", True), ("shared_children", False),
                                         ("unreached", False)])
@pytest.mark.parametrize("rank,batch", [(1, 3), (3, 5), (4, 1)])
def test_the_virtual_batch_is_the_torch_pass(name, served, rank, batch, monkeypatch):
    """`sum_exp` through the plain kernels on (sample, rank) columns against its torch passes: the same values, flows,
    denominators and gradients -- taken where no child of a layer is shared by two of its node vectors, and the torch
    passes kept where one is."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    pc, sums, _, _, _, _ = _pair(name, "sum_exp", rank = rank)
    dev = pc.params.device
    torch.manual_seed(9)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 50 + i) for i, a in enumerate(sums)}

    got = _passes(pc, sums, data, ext)
    assert got["path"] == ({tuple} if served else {dict})
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    assert want["path"] == {dict}
    _same(got, want, rtol = 2e-4, atol = 2e-5)

    # Without parameter flows: the element flows and the gradients alone, nothing accumulated
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    again = _passes(pc, sums, data, ext, param_flows = False)
    assert float(again["fp"].abs().sum()) == 0.0 and float(pc.denom_param_flows.abs().sum()) == 0.0
    assert torch.allclose(again["flows"].exp(), want["flows"].exp(), rtol = 2e-4, atol = 2e-5)
    for k in ("dA", "dB"):
        for a, b in zip(again[k], want[k]):
            assert torch.allclose(a, b, rtol = 2e-4, atol = 2e-5)


@needs_virtual
@pytest.mark.parametrize("precision,rtol", [("auto", 5e-3), ("fp32", 2e-4)])
def test_the_virtual_batch_on_tensor_core_shapes(precision, rtol, monkeypatch):
    """A tied HMM of 128 latents in blocks of 32 at batch 24: the shapes the plain layer's tensor-core kernels take,
    whose products are bf16 at the default precision -- within their tolerance of the torch passes there, and close at
    `precision = "fp32"`."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 4, 24
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ni = inputs(4, num_node_blocks = 4, dist = dists.Categorical(num_cats = NUM_CATS))
        sums, cur = [], ni
        for var in range(3, -1, -1):
            cx = ni.duplicate(var, tie_params = True)
            ns = sums[0].duplicate(cur, tie_params = True) if sums else \
                summate(cur, num_node_blocks = 4, parameterization = LowRank(rank = rank, form = "sum_exp"))
            sums.append(ns)
            cur = multiply(cx, ns)
        root = summate(cur, num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, precision = precision).to(torch.device("cuda:0"))
    dev = pc.params.device
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, spread = 0.5, seed = 60 + i) for i, a in enumerate(sums)}

    got = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple}
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    assert torch.allclose(got["lls"], want["lls"], rtol = 0, atol = 20 * rtol)
    assert torch.allclose(got["fp"], want["fp"], rtol = 10 * rtol, atol = 10 * rtol * float(want["fp"].max()))
    assert torch.allclose(got["fm"], want["fm"], rtol = 10 * rtol, atol = 10 * rtol * float(want["fm"].max()))
    for k in ("dA", "dB"):
        for a, b in zip(got[k], want[k]):
            assert float((a - b).abs().max()) <= 10 * rtol * float(b.abs().max())


@needs_virtual
def test_the_virtual_buffers_stay_what_the_plain_kernels_expect(monkeypatch):
    """The plain kernels pad a ragged row's edges with the circuit's dummy rows: the virtual buffers hold -inf there
    (zero value, zero flow) from their allocation on, whatever memory they were given; the stand-in node values of
    the backward stay 0. A pass over them after the torch passes dirtied the allocator is the torch passes' result.
    The forward's node values (each rank's log S) lie in a node buffer of the twin's rows, the dummy ones -inf; what
    the backward takes of them, the share differences, is each layer's own (`_kept`): the twin's rows, the dummy ones
    0."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 3, 6
    pc, sums, _, _, _, _ = _pair("sparse", "sum_exp", rank = rank)
    dev = pc.params.device
    torch.manual_seed(4)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 20 + i) for i, a in enumerate(sums)}

    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    junk = [torch.full((1 << 18,), float("nan"), device = dev) for _ in range(8)]      # what freed blocks may hold
    del junk
    pc.node_mars.__dict__.pop("_lowrank_virtual", None)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    got = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple}
    _same(got, want, rtol = 2e-4, atol = 2e-5)

    families = pc.node_mars._lowrank_virtual                         # (kept on the circuit's own buffer)
    layers = list({id(l): l for l in (pc.external_params_nodes[a] for a in sums)}.values())
    ranges = {lowrank_virtual._twin(l)[0]._layer_nid_range for l in layers}
    cols = batch * rank
    assert set(families) == {(f, cols, r) for f in ("fw", "bw") for r in [None, *ranges]}
    bufs = {name: buf for (_, _, r), family in families.items() if r is None for name, buf in family.items()}
    layer = pc.external_params_nodes[sums[0]]
    kids = layer.parameterization._geometry(layer, layer.parameterization._info(layer, sums[0]), dev)["kids"]
    other = torch.ones(pc.element_mars.size(0), dtype = torch.bool, device = dev)
    other[kids] = False
    for name in ("vem", "vem_d", "vef_d"):
        assert bool(torch.isneginf(bufs[name][other]).all()), name
    for (f, _, r), family in families.items():                       # the node ones: one layer's rows, dummies first
        if r is None:
            continue
        nodes = ("vnm",) if f == "fw" else ("vnf_d", "zero")
        assert all(family[name].size(0) == r[1] < pc.node_mars.size(0) for name in nodes)
        assert all(family[name] is buf for name, buf in families[(f, cols, None)].items())   # (the element ones: one set)
        assert bool(torch.isneginf(family[nodes[0]][:r[0]]).all())
        assert f == "fw" or float(family["zero"].abs().sum()) == 0.0
    for layer in layers:                                             # the kept share differences: the layer's own
        kept, nd = layer.__dict__["_lrv_kd"], lowrank_virtual._twin(layer)[0]._layer_nid_range
        assert kept.shape == (nd[1], cols) and float(kept[:nd[0]].abs().sum()) == 0.0


@needs_virtual
@pytest.mark.parametrize("name", ["sparse", "hmm_tied"])
def test_the_plain_kernels_run_on_a_twin_of_the_layer_one_layer_tall(name, monkeypatch):
    """The virtual batch runs the plain kernels on the layer's twin (`_twin`): a plain `SumLayer` whose node ids are the
    layer's `shift` rows lower -- its own nodes from row `nd` on, a dummy id as it is, inside the dummy rows --, every
    other table the layer's own tensor, its caches its own. The torch passes' result; after the layer's device caches
    are dropped a new twin, the same result."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.sum_layer import SumLayer

    rank, batch = 2, 5
    pc, sums, _, _, _, _ = _pair(name, "sum_exp", rank = rank)
    dev = pc.params.device
    torch.manual_seed(51)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 280 + i) for i, a in enumerate(sums)}
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    got = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple}
    _same(got, want, rtol = 2e-4, atol = 2e-5)

    dummies = 0
    for layer in {id(l): l for l in (pc.external_params_nodes[a] for a in sums)}.values():
        twin, shift = lowrank_virtual._twin(layer)
        lo, hi = layer._layer_nid_range
        nd = twin._layer_nid_range[0]
        assert type(twin) is SumLayer and twin._layer_nid_range == (lo - shift, hi - shift)
        assert nd % 32 == 0 and nd >= max(32, layer.block_size)
        for t, u in zip(layer.partitioned_nids, twin.partitioned_nids):
            assert torch.equal(u, t - shift)
        for t, u in zip(layer.partitioned_parids, twin.partitioned_parids):
            assert torch.equal(u, torch.where(t >= lo, t - shift, t))
            assert bool((u[t < lo] + layer.block_size <= nd).all())        # (a dummy block inside the dummy rows)
            dummies += int((t < lo).sum())
        for table in ("partitioned_cids", "partitioned_pids", "partitioned_pfids", "partitioned_chids",
                      "partitioned_parpids"):
            assert getattr(twin, table) is getattr(layer, table)
        caches = [k for k, v in twin.__dict__.items() if k.startswith("_cached_") and isinstance(v, dict)]
        assert caches and all(twin.__dict__[k] is not layer.__dict__[k] for k in caches)
        assert any(twin.__dict__[k] for k in caches)                       # (the twin's passes filled its own)
    if name == "sparse":
        assert dummies > 0                                                 # (what this structure is here for)

    layer = pc.external_params_nodes[sums[0]]
    old = lowrank_virtual._twin(layer)[0]
    layer._drop_device_caches()
    assert "_lrv_twin" not in layer.__dict__
    again = _passes(pc, sums, data, ext)
    assert lowrank_virtual._twin(layer)[0] is not old
    _same(again, got, rtol = 1e-6, atol = 1e-7)


@needs_virtual
def test_the_node_buffers_of_two_layers_are_shared_only_where_their_rows_lie_alike():
    """`_buffers` keys a layer's node buffers by its twin's rows, the dummy rows included: two layers of one height
    whose dummy rows differ never share one (one would write the other's dummy rows); the element ones, one per family
    and width, are the same for both -- the layer below writes the next layer's."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    dev = torch.device("cuda:0")
    node_mars, element_mars = torch.zeros(1000, 3, device = dev), torch.zeros(200, 3, device = dev)
    a = lowrank_virtual._buffers(node_mars, element_mars, 6, "bw", (32, 128))
    b = lowrank_virtual._buffers(node_mars, element_mars, 6, "bw", (64, 128))
    c = lowrank_virtual._buffers(node_mars, element_mars, 6, "bw", (32, 128))
    assert a["vnf_d"] is not b["vnf_d"] and a["zero"] is not b["zero"] and a["vnf_d"] is c["vnf_d"]
    assert a["vef_d"] is b["vef_d"] and a["vem_d"] is b["vem_d"]
    assert a["vnf_d"].shape == (128, 6) and a["vef_d"].shape == (200, 6)
    assert bool(torch.isneginf(a["vnf_d"]).all()) and float(a["zero"].abs().sum()) == 0.0
    f = lowrank_virtual._buffers(node_mars, element_mars, 6, "fw")
    assert set(f) == {"vem"}
    assert f["vem"] is lowrank_virtual._buffers(node_mars, element_mars, 6, "fw", (32, 128))["vem"]


@needs_virtual
def test_a_twin_refuses_tables_past_the_layers_rows():
    """The twin shifts the layer's own node ids and keeps the dummies; a table naming a node past the layer's rows --
    neither -- is refused, not shifted into another row."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = 2)
    layer = pc.external_params_nodes[sums[0]]
    layer.__dict__.pop("_lrv_twin", None)
    real = layer._modules["partitioned_parids"]
    bad = real[0].clone()
    bad.view(-1)[0] = layer._layer_nid_range[1]                              # (the first node after the layer's)
    layer._modules["partitioned_parids"] = type(real)([torch.nn.Parameter(bad, requires_grad = False), *list(real)[1:]])
    try:
        with pytest.raises(RuntimeError, match = "past its own rows"):
            lowrank_virtual._twin(layer)
    finally:
        layer._modules["partitioned_parids"] = real
    assert "_lrv_twin" not in layer.__dict__ and lowrank_virtual._twin(layer)[0] is not None


def _grid_reference(ns, M, N, device):
    """The node vector's `(cell i * N + j, its parameter's local index)` per stored parameter, from its edge blocks and
    its source's parameter ids -- what the gather and the scatter use, built here on its own."""
    src = ns.get_source_ns() if ns.is_tied() else ns
    bs, cbs = ns.block_size, ns.ch_block_size
    nb, cb = ns.edge_ids[0].to(device).long(), ns.edge_ids[1].to(device).long()
    m, c = torch.arange(bs, device = device), torch.arange(cbs, device = device)
    rows = (nb[:, None, None] * bs + m[None, None, :]).expand(-1, cbs, -1)
    cols = (cb[:, None, None] * cbs + c[None, :, None]).expand(-1, -1, bs)
    pid = src._param_ids.to(device).long()[:, None, None] + (c[:, None] * bs + m[None, :])[None]
    return (rows * N + cols).reshape(-1), (pid - src._param_range[0]).reshape(-1), src._param_range[0]


@pytest.mark.parametrize("name", ["hmm_tied", "hmm_untied", "sparse"])
def test_a_block_major_node_vector_reads_and_writes_its_parameters_as_a_strided_grid(name):
    """Where a node vector's parameters lie block-major in one run -- every cell's edge block in its place, node block
    by node block, each in child order -- its dense `theta` is a strided copy of the run, in fp32 or bf16, and a grid's
    values -- times its parameters, read in place, or not -- go onto its parameters' slots as one strided add, no
    `[M N]` index table made; a node vector that is not (ragged) keeps the gather and the scatter. Both against the
    cells and slots built here from the edge blocks and the parameter ids."""
    pc, sums, _, _, _, _ = _pair(name, "sum_exp", rank = 2)
    dev = pc.params.device
    for k, a in enumerate(sums):
        layer = pc.external_params_nodes[a]
        desc = a.parameterization
        geo = desc._geometry(layer, desc._info(layer, a), dev)
        M, N = geo["M"], geo["N"]
        flat, local, p0 = _grid_reference(a, M, N, dev)
        if name == "sparse":
            assert geo["grid"] is None and torch.equal(geo["flat"], flat) and torch.equal(geo["local"], local)
        else:
            assert geo["grid"] is not None and "flat" not in geo and "pid" not in geo, k

        want = torch.zeros(M * N, device = dev)
        want[flat] = pc.params[p0 + local].float()
        assert torch.equal(desc._dense_params(geo, pc.params), want.view(M, N))
        assert torch.equal(desc._dense_params(geo, pc.params, torch.bfloat16), want.view(M, N).to(torch.bfloat16))

        g = torch.Generator(device = "cpu").manual_seed(k)
        values = torch.randn(M, N, generator = g).to(dev)
        size = a._param_flow_range[1] - a._param_flow_range[0]
        before = torch.randn(size + 7, generator = g).to(dev)
        for base, params in ((0, None), (7, pc.params)):
            got, ref = before.clone(), before.clone()
            desc._scatter_add(geo, got, base, values, params)
            v = values.reshape(-1)[flat] * (1.0 if params is None else pc.params[p0 + local])
            ref.index_add_(0, base + local, v)
            assert torch.allclose(got, ref, rtol = 1e-6, atol = 1e-6) and \
                (params is not None or torch.equal(got, ref)), (k, base)


@pytest.mark.parametrize("listing,grid", [(((1, 0, 1, 0), (0, 0, 1, 1)), True), (((0, 0, 1, 1), (1, 0, 1, 0)), False),
                                          (((0, 0, 1), (0, 1, 0)), False)])
def test_the_grid_is_read_where_the_parameters_lie_block_major_however_the_edges_are_listed(listing, grid):
    """Edge blocks out of block-major order. Compilation lays a node block's parameters out in the order its edge
    blocks are listed, node block by node block: every node block with every child block, listed node block 1 first,
    child-major, they still lie block-major -- the grid, its run from the least id, not the first edge block's --;
    with each node block's children reversed they do not -- read as one run, the grid would take other cells --, and
    the gather and the scatter stay; nor where a cell has no edge block, the others a block-major prefix (the run
    read as the grid would reach past the node vector's parameters). Each against the cells and slots built from the
    edge blocks and the parameter ids."""
    dev = torch.device("cuda:0")
    with juice.set_block_size(4):
        prod = multiply(inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS)),
                        inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = NUM_CATS)))
        ns = summate(prod, num_node_blocks = 2, edge_ids = torch.tensor(listing),
                     parameterization = LowRank(rank = 2, form = "sum_exp"))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(dev)
    layer = pc.external_params_nodes[ns]
    desc = ns.parameterization
    geo = desc._geometry(layer, desc._info(layer, ns), dev)
    M, N = geo["M"], geo["N"]
    flat, local, p0 = _grid_reference(ns, M, N, dev)
    if grid:
        assert geo["grid"] is not None and "flat" not in geo and int(ns._param_ids[0]) != p0 + geo["grid"][0]
    else:
        assert geo["grid"] is None and torch.equal(geo["flat"], flat) and torch.equal(geo["local"], local)
    want = torch.zeros(M * N, device = dev)
    want[flat] = pc.params[p0 + local].float()
    assert torch.equal(desc._dense_params(geo, pc.params), want.view(M, N))
    values = torch.randn(M, N, device = dev)
    got, ref = torch.zeros(local.numel() + 3, device = dev), torch.zeros(local.numel() + 3, device = dev)
    desc._scatter_add(geo, got, 3, values)
    ref.index_add_(0, 3 + local, values.reshape(-1)[flat])
    assert torch.equal(got, ref)


@pytest.mark.parametrize("form", ["exp_sum", "sum_exp"])
def test_a_node_whose_parameters_are_all_zero_has_no_mass(form, monkeypatch):
    """A node every shared parameter of which is 0 has no mass whatever its factors: the value -inf (as a plain sum
    node's), not `log S - log Z = -inf - -inf`; the circuit's values, flows, denominators and gradients finite
    elsewhere and, for the mixture form, the virtual batch's the torch passes' -- its collapse, where the plain
    kernels give the node's virtual values -FLT_MAX, by itself and with the layer above's `expand` taken along
    (`collapse_expand`: the product layers fused in, the layers linked by the passes before)."""
    import pyjuice.layer.kernels.autotune as autotune
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused" if key[0] == "prod_stream_fusion"
                        else "unfused" if key[0] == "prod_stream_fusion_bk" else pick(key, cands, bench, **k))
    real, merged = lowrank_virtual._module, []

    class Counted:
        def __getattr__(self, attr):
            fn = getattr(real(), attr)
            if attr != "lowrank_collapse_expand":
                return fn

            def counted(*args):
                merged.append(attr)
                return fn(*args)
            return counted

    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)

    rank, batch, dead = 2, 6, 1                                             # (`dead`: a node of the tied transition)
    out = dict()
    for arm in ("torch", "virtual") if form == "sum_exp" else ("torch",):
        # (a circuit each: the first passes decide the product layers' fusion, which the virtual batch's link needs)
        monkeypatch.setattr(lowrank_virtual, "ENABLED", arm == "virtual")
        pc, sums, _, _, _, _ = _pair("hmm_tied", form, rank = rank)
        dev = pc.params.device
        a = sums[0]
        layer = pc.external_params_nodes[a]
        desc = a.parameterization
        geo = desc._geometry(layer, desc._info(layer, a), dev)
        flat, local, p0 = _grid_reference(a, geo["M"], geo["N"], dev)
        with torch.no_grad():
            pc.params[p0 + local[flat // geo["N"] == dead]] = 0.0
        torch.manual_seed(57)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {s: _factors(s, rank, batch, dev, seed = 300 + i) for i, s in enumerate(sums)}
        for k in range(3):                                                  # (linked from the third on)
            merged.clear()
            got = out[arm, k] = _passes(pc, sums, data, ext)
            for s in sums:
                info = desc._info(pc.external_params_nodes[s], s)
                assert bool(torch.isneginf(got["mars"][info.nid_start + dead]).all()), (arm, k, "the dead node's value")
            for key in ("lls", "fp", "fm"):
                assert bool(torch.isfinite(got[key]).all()), (arm, k, key)
            assert not bool(torch.isnan(got["mars"]).any()) and not bool(torch.isnan(got["flows"]).any()), (arm, k)
            assert all(bool(torch.isfinite(g).all()) for g in got["dA"] + got["dB"]), (arm, k)
        if arm == "virtual":
            assert merged, "the linked collapse not taken"
    if form == "sum_exp":
        for k in range(3):
            _same(out["virtual", k], out["torch", k], rtol = 2e-4, atol = 2e-5)


@needs_virtual
def test_the_virtual_batch_hands_the_circuits_bf16_copy_to_the_plain_kernels(monkeypatch):
    """Where the circuit makes its bf16 copy of tied fp32 parameters (`TensorCircuit._param_operands`, forced here),
    the virtual batch's plain kernels are handed it through the layer's twin, as a plain layer's are -- the forward's
    `op_params`, the backward's per-call copy --, the twin marked for it at every call (made here before the circuit
    marked the layers, so its mark is the one each call carries over); the values, flows and gradients within bf16's
    rounding of the torch passes', which read the fp32 parameters."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.sum_layer import SumLayer

    rank, batch = 2, 16                                          # (forced: made from 16 virtual columns on)
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank)
    dev = pc.params.device
    layers = list({id(l): l for l in (pc.external_params_nodes[a] for a in sums)}.values())
    twins = {id(lowrank_virtual._twin(l)[0]) for l in layers}             # (before the circuit's first pass)
    torch.manual_seed(53)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 290 + i) for i, a in enumerate(sums)}
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)

    calls, real_fw, real_bw = [], SumLayer.forward, SumLayer.backward

    def fw(self, *args, op_params = None, **kwargs):
        calls.append(("fw", self, op_params, self.__dict__.get("_reads_param_operands", False)))
        return real_fw(self, *args, op_params = op_params, **kwargs)

    def bw(self, *args, op_params = None, **kwargs):
        calls.append(("bw", self, op_params, self.__dict__.get("_reads_param_operands", False)))
        return real_bw(self, *args, op_params = op_params, **kwargs)

    monkeypatch.setattr(SumLayer, "forward", fw)
    monkeypatch.setattr(SumLayer, "backward", bw)
    monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)
    got = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple}
    assert all(l.__dict__.get("_reads_param_operands") for l in layers)
    assert {id(lowrank_virtual._twin(l)[0]) for l in layers} == twins
    seen = [(kind, op, marked) for kind, layer, op, marked in calls if id(layer) in twins]
    assert {kind for kind, _, _ in seen} == {"fw", "bw"}
    assert all(op is not None and op.dtype == torch.bfloat16 and marked for _, op, marked in seen), \
        [(k, op is not None, marked) for k, op, marked in seen]
    _same(got, want, rtol = 2e-2, atol = 1e-3)


@needs_virtual
def test_the_rule_on_the_copy_reckons_the_virtual_columns(monkeypatch):
    """The circuit's rule on its bf16 copy (`TensorCircuit._param_operand_groups`, not forced) reckons a mixture-form
    layer's work, and the 16 columns it takes the copy from, on the columns its plain kernels run at -- `rank` a
    sample (`LowRank.param_operand_columns`) --, not the batch: rank 4 at batch 4 is 16 columns, given the copy where
    the layer's parameters times 16 reach `_COPY_MIN_WORK` (set here), the batch alone below it; batch 3 (12 columns)
    never; the work's bound kept (16 columns short of a threshold of 32, batch 8 reaching it). The twins are handed
    the copy, the values, flows and gradients within bf16's rounding of the torch passes'. The `exp_sum` form's
    layers, whose torch passes read `params`, are not marked. A group of a plain layer and a mixture-form one takes
    the least of their columns, the plain layer's batch: none of its layers handed the copy at batch 4, both at 16.
    Forced (`_param_operands_force`), the copy goes to no group below 16 columns either: the columns' least kept."""
    import pyjuice.model.tensorcircuit as tcm
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.sum_layer import SumLayer

    monkeypatch.setattr(tcm.TensorCircuit, "_param_operands_force", None)
    rank = 4
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank)
    dev = pc.params.device
    groups, _ = pc._param_operand_plan()
    sizes = pc.__dict__["_param_operand_sizes_"]
    assert groups and set(sizes) == set(groups)
    edges = max(e for e, _, _ in sizes.values())
    layers = list({id(l): l for l in (pc.external_params_nodes[a] for a in sums)}.values())
    assert all(l.__dict__.get("_reads_param_operands") for l in layers)

    seen, real_fw = [], SumLayer.forward

    def fw(self, *args, op_params = None, **kwargs):
        seen.append((self, op_params))
        return real_fw(self, *args, op_params = op_params, **kwargs)

    monkeypatch.setattr(SumLayer, "forward", fw)

    def run(batch):
        torch.manual_seed(60 + batch)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 310 + i) for i, a in enumerate(sums)}
        monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
        want = _passes(pc, sums, data, ext)
        monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
        seen.clear()
        got = _passes(pc, sums, data, ext)
        assert got["path"] == {tuple}
        twins = {id(lowrank_virtual._twin(l)[0]) for l in layers}
        ops = [op for layer, op in seen if id(layer) in twins]
        assert ops, batch
        made = pc.__dict__["_param_operands_made"]
        assert all((op is not None) is made for op in ops), batch
        if made:
            assert all(op.dtype == torch.bfloat16 for op in ops)
            _same(got, want, rtol = 2e-2, atol = 1e-3)
        else:
            _same(got, want, rtol = 2e-4, atol = 2e-5)
        return made

    monkeypatch.setattr(tcm, "_COPY_MIN_WORK", edges * 16)
    assert edges * 4 < tcm._COPY_MIN_WORK                                # (the batch alone: below it)
    assert run(4) and pc._param_operand_groups(4) == groups
    assert not run(3) and pc._param_operands(3) is None and not pc._param_operand_groups(3)
    pc.__dict__.pop("_param_operand_rule_")                               # (the rule again, at another threshold)
    monkeypatch.setattr(tcm, "_COPY_MIN_WORK", edges * 32)
    assert not run(4) and pc._param_operands(4) is not None              # (16 columns: it applies, not given)
    assert run(8)

    pc_e, sums_e, _, _, _, _ = _pair("hmm_tied", "exp_sum", rank = rank)
    assert not pc_e._param_operand_plan()[0] and pc_e._param_operands(64) is None
    assert not any(pc_e.external_params_nodes[a].__dict__.get("_reads_param_operands") for a in sums_e)

    # (two tied chains side by side, the first's transitions mixture-form, the second's plain: a layer of each a group)
    from pyjuice.layer.parameterized_sum_layer import ParameterizedSumLayer
    torch.manual_seed(0)
    tops, sums_m = [], []
    with juice.set_block_size(8):
        for chain in range(2):
            ni = inputs(chain * 4 + 3, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
            mine, cur = [], ni
            for var in range(2, -1, -1):
                cx = ni.duplicate(chain * 4 + var, tie_params = True)
                ns = mine[0].duplicate(cur, tie_params = True) if mine else \
                    summate(cur, num_node_blocks = 1, **(_ext(True, rank, "sum_exp") if chain == 0 else {}))
                mine.append(ns)
                cur = multiply(cx, ns)
            tops.append(summate(cur, num_node_blocks = 1))
            sums_m += mine if chain == 0 else []
        root = summate(multiply(*tops), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc_m = juice.compile(root, verbose = False).to(dev)
    monkeypatch.setattr(tcm, "_COPY_MIN_WORK", 1)
    groups_m, _ = pc_m._param_operand_plan()
    mixed = [g for g in pc_m.inner_layer_groups if id(g) in groups_m and any(type(l) is SumLayer for l in g)
             and any(isinstance(l, ParameterizedSumLayer) for l in g)]
    assert mixed
    for batch, given in ((4, False), (16, True)):
        assert all((id(g) in pc_m._param_operand_groups(batch)) is given for g in mixed), batch
        data = torch.randint(0, NUM_CATS, (batch, pc_m.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 330 + i) for i, a in enumerate(sums_m)}
        seen.clear()
        with torch.no_grad():
            pc_m(data, sum_external_params = ext)
        twins = {id(lowrank_virtual._twin(l)[0]) for g in mixed for l in g if isinstance(l, ParameterizedSumLayer)}
        plain = {id(l) for g in mixed for l in g if type(l) is SumLayer}
        ops = [op for layer, op in seen if id(layer) in twins | plain]
        assert {id(layer) for layer, _ in seen} >= twins | plain and all((op is not None) is given for op in ops), batch
    monkeypatch.setattr(pc_m, "_param_operands_force", True, raising = False)
    for batch, given in ((4, False), (16, True)):
        assert all((id(g) in pc_m._param_operand_groups(batch)) is given for g in mixed), ("forced", batch)


@needs_virtual
def test_the_virtual_batch_across_batch_sizes_and_its_bound(monkeypatch):
    """Forwards at two batch sizes in turn, each followed by its backward: the buffers follow the batch. Past
    `VIRTUAL_BYTES_MOST` the torch passes take the layer, with the same result."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank = 2
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank)
    dev = pc.params.device
    out = dict()
    for batch in (7, 3, 7):
        torch.manual_seed(batch)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 70 + i) for i, a in enumerate(sums)}
        got = _passes(pc, sums, data, ext)
        assert got["path"] == {tuple} and {c for _, c, _ in pc.node_mars._lowrank_virtual} == {batch * rank}
        if batch in out:
            _same(got, out[batch], rtol = 1e-6, atol = 1e-7)
        out[batch] = got

    rows = lowrank_virtual._twin(pc.external_params_nodes[sums[0]])[0]._layer_nid_range[1]     # (a layer's: all alike)
    need = lowrank_virtual._bytes(rows, pc.element_mars, 7 * rank)
    monkeypatch.setattr(lowrank_virtual, "VIRTUAL_BYTES_MOST", need - 1)
    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, (7, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, 7, dev, seed = 70 + i) for i, a in enumerate(sums)}
    over = _passes(pc, sums, data, ext)
    assert over["path"] == {dict}
    _same(over, out[7], rtol = 2e-4, atol = 2e-5)
    monkeypatch.setattr(lowrank_virtual, "VIRTUAL_BYTES_MOST", need)
    assert _passes(pc, sums, data, ext)["path"] == {tuple}


def _plain_backwards(monkeypatch):
    """Counts the plain layer's backward calls and their batch (the virtual batch's columns)."""
    from pyjuice.layer.sum_layer import SumLayer

    calls, plain = [], SumLayer.backward

    def counted(self, node_flows, *args, **kwargs):
        calls.append(node_flows.size(1))
        return plain(self, node_flows, *args, **kwargs)

    monkeypatch.setattr(SumLayer, "backward", counted)
    return calls


def _normalizer_calls(monkeypatch):
    """Counts the calls of the normalizer half's operand kernel -- from the factors (`lowrank_u_operand`) or from the
    exponentials the pass keeps (`lowrank_kept_operands`, its U given) --, with the rows of each (copies x batch x
    rank)."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    real, calls = lowrank_virtual._module, []

    class Counted:
        def __getattr__(self, attr):
            fn = getattr(real(), attr)
            if attr == "lowrank_kept_operands":
                def kept(Ud, Vd, Un, *args):
                    if Un is not None:
                        calls.append((Un.size(1), Un.dtype))
                    return fn(Ud, Vd, Un, *args)
                return kept
            if attr != "lowrank_u_operand":
                return fn

            def counted(Ut, *args):                                  # (`Ut` [M, rows]: the rows are the copies')
                calls.append((Ut.size(1), Ut.dtype))
                return fn(Ut, *args)
            return counted

    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)
    return calls


@needs_virtual
@pytest.mark.parametrize("operands", ["fp32", "bf16"])
def test_the_normalizer_half_is_two_products_over_all_the_copies(operands, monkeypatch):
    """Five tied copies, one a layer: each layer's backward runs the plain backward once, on its data half, and at the
    pass's end ONE pair of matrix products over the five copies' rows gives the denominator statistic (written into
    the slice of the first copy the backward met, the topmost; the others' hold nothing) and the gradients'
    normalizer part. The torch passes' result,
    with fp32 operands and with bf16 ones (`GEMM_BF16_WORK` forced to 0); a second backward of the same forward the
    same again, nothing left over from the first."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 2, 4
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank, seq_length = 6)
    dev = pc.params.device
    assert len(sums) == 5
    torch.manual_seed(13)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 80 + i) for i, a in enumerate(sums)}
    if operands == "bf16":
        monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)
    tol = dict(rtol = 2e-4, atol = 2e-5) if operands == "fp32" else dict(rtol = 1e-2, atol = 1e-3)

    cols = batch * rank
    plain, norm = _plain_backwards(monkeypatch), _normalizer_calls(monkeypatch)
    got = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple}
    assert [c for c in plain if c != batch] == [cols] * 5             # (the root's layer runs at the batch itself)
    assert norm == [(5 * cols, torch.float32 if operands == "fp32" else torch.bfloat16)]
    desc = sums[0].parameterization
    held = [float(pc.denom_param_flows[slice(o, o + sums[0].num_edges)].abs().sum()) > 0
            for o in (desc._denom_offsets(pc.external_params_nodes[a])[id(a)][0] for a in sums)]
    assert held == [False, False, False, False, True]                # (the top transition's: the backward's first)

    plain.clear()
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    assert not [c for c in plain if c != batch]
    _same(got, want, **tol)

    # A second backward of the same forward: the same again
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    grads = {a: (torch.zeros_like(ext[a][0]), torch.zeros_like(ext[a][1])) for a in sums}
    pc(data, sum_external_params = ext)
    for _ in range(2):
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)
        assert torch.allclose(pc.param_flows, want["fp"], **tol)
        assert torch.allclose(_denominators(pc, sums), want["fm"], **tol)
        for k, a in enumerate(sums):
            assert torch.allclose(grads[a][0], want["dA"][k], **tol)
            assert torch.allclose(grads[a][1], want["dB"][k], **tol)


@needs_virtual
def test_the_normalizer_products_with_fewer_copies_and_without_some_gradients(monkeypatch):
    """A pass that gives factors to some copies only, and one that asks for the gradients of some: the products run
    over the copies in the pass, the gradients reach the buffers given; the torch passes' result."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 2, 4
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank, seq_length = 6)
    dev = pc.params.device
    torch.manual_seed(14)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 90 + i) for i, a in enumerate(sums)}
    _passes(pc, sums, data, ext)
    norm = _normalizer_calls(monkeypatch)
    for kept in ([0, 2, 4], [1], [3, 4]):
        few = [sums[i] for i in kept]
        sub = {a: ext[a] for a in few}
        monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
        norm.clear()
        got = _passes(pc, few, data, sub)
        assert got["path"] == {tuple} and norm == [(len(kept) * batch * rank, torch.float32)]
        monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
        want = _passes(pc, few, data, sub)
        _same(got, want, rtol = 2e-4, atol = 2e-5)

    # gradients for the second and the fourth copy only
    out = dict()
    for enabled in (True, False):
        monkeypatch.setattr(lowrank_virtual, "ENABLED", enabled)
        grads = {a: (torch.zeros_like(ext[a][0]), torch.zeros_like(ext[a][1])) for a in (sums[1], sums[3])}
        pc(data, sum_external_params = ext)
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)
        out[enabled] = (grads, _denominators(pc, sums))
    for a in (sums[1], sums[3]):
        for k in (0, 1):
            assert torch.allclose(out[True][0][a][k], out[False][0][a][k], rtol = 2e-4, atol = 2e-5)
    assert torch.allclose(out[True][1], out[False][1], rtol = 2e-4, atol = 2e-5)


@needs_virtual
def test_the_normalizer_products_shift_large_factors(monkeypatch):
    """Factors whose entries pass 89: `exp(B)` alone overflows fp32 (past 88.7); shifted by each row's largest entry,
    the products are the torch passes' (which work in float64)."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 2, 4
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank)
    dev = pc.params.device
    torch.manual_seed(21)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, spread = 40.0, seed = 300 + i) for i, a in enumerate(sums)}
    assert max(float(B.max()) for _, B in ext.values()) > 89
    got = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple} and all(torch.isfinite(t).all() for t in (got["fp"], got["fm"], *got["dB"]))
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    _same(got, _passes(pc, sums, data, ext), rtol = 1e-3, atol = 1e-4)


@needs_virtual
@pytest.mark.parametrize("M, N, K, B, copies", [(3, 300, 2, 37, 2),     # children past one warp's share, samples past one block's
                                                (5, 3, 3, 4, 70)])      # few children, copies past one launch's
def test_the_normalizer_operands_and_gradients_follow_their_formulas(M, N, K, B, copies):
    """The kernels' contract, checked directly. `Vt[j, (c, b, r)] = exp(B_c[r, j, b] - shift)` with the shift the row's
    largest `B_c[r, :, b]`, and `Ut[i, (c, b, r)] = exp(lf_c[i, b] + A_c[i, r, b] - log Z_c[i, b] + shift)` -- a copy's
    rows `c B K + b K + r`, a sample's ranks adjacent --; a node
    without flow (`lf = -inf`) gives 0 -- also where it has no mass at all (factors of -inf, `log Z = -inf`, so its
    value and its flow are -inf: `-inf - -inf` would be NaN). The gradients: `dB_c[r, j, b] -= Vt * Gt` on the copy's
    rows, a copy without a buffer skipped. float32 and bfloat16 operands."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    mod, dev = lowrank_virtual._module(), torch.device("cuda")
    g = torch.Generator(device = "cpu").manual_seed(7)
    fa = [(torch.randn(M, K, B, generator = g) * 3).to(dev) for _ in range(copies)]
    fb = [(torch.randn(K, N, B, generator = g) * 3).to(dev) for _ in range(copies)]
    lz = [(torch.randn(M, B, generator = g) * 3).to(dev) for _ in range(copies)]
    node_flows = (torch.randn(2 + copies * M, B, generator = g) - 1.0).to(dev)        # log flows, the copies' rows below 2
    nid0 = [2 + c * M for c in range(copies)]
    node_flows[nid0[0] + 1, 2] = -float("inf")                                        # copy 0, node 1, sample 2: no flow
    lz[1][0, 3] = -float("inf")                                                       # copy 1, node 0, sample 3: no mass
    fa[1][0, :, 3] = -float("inf")
    node_flows[nid0[1] + 0, 3] = -float("inf")                                        # (so no flow either)
    fb[0][1, :, 1] = -float("inf")                                                    # copy 0, rank 1, sample 1: no mass

    rows = copies * B * K
    for dtype in (torch.float32, torch.bfloat16):
        tol = dict(rtol = 1e-5, atol = 1e-6) if dtype is torch.float32 else dict(rtol = 1e-2, atol = 1e-3)
        Ut = torch.full((M, rows), float("nan"), dtype = dtype, device = dev)
        Vt = torch.full((N, rows), float("nan"), dtype = dtype, device = dev)
        shift = torch.full((rows,), float("nan"), device = dev)
        row0 = [c * B * K for c in range(copies)]
        mod.lowrank_z_operand(Vt, shift, fb, B, K)
        mod.lowrank_u_operand(Ut, shift, node_flows, fa, lz, nid0, row0, M, K)
        torch.cuda.synchronize()
        for c in range(copies):
            lf = node_flows[nid0[c]:nid0[c] + M]                                      # [M, B]
            for r in range(K):
                row = c * B * K + torch.arange(B, device = dev) * K + r
                sh = fb[c][r].max(dim = 0).values                                     # [B]
                assert torch.equal(shift[row], sh), f"shift: copy {c}, rank {r}"
                want_v = torch.where(torch.isneginf(fb[c][r]), torch.zeros_like(fb[c][r]), torch.exp(fb[c][r] - sh))
                assert torch.allclose(Vt[:, row].float(), want_v, **tol), f"V: copy {c}, rank {r}"
                assert torch.equal(Vt[:, row].float().max(dim = 0).values, torch.isfinite(sh).float()), "not shifted"
                want_u = torch.exp(lf + fa[c][:, r] - lz[c] + sh)
                want_u = torch.where(torch.isneginf(lf) | torch.isneginf(lz[c]), torch.zeros_like(want_u), want_u)
                assert torch.allclose(Ut[:, row].float(), want_u, **tol), f"U: copy {c}, rank {r}"
        assert torch.isfinite(Ut.float()).all() and torch.isfinite(Vt.float()).all()
        assert all(float(Ut[1, 2 * K + r]) == 0.0 for r in range(K))
        assert all(float(Ut[0, 1 * B * K + 3 * K + r]) == 0.0 for r in range(K))

        # Each copy its own buffer, one with none...
        Gt = torch.randn(N, rows, generator = g).to(dev)
        before = [(torch.randn(K, N, B, generator = g)).to(dev) for _ in range(copies)]
        grads = [t.clone() for t in before]
        passed = [None if c == 1 else grads[c] for c in range(copies)]
        mod.lowrank_normalizer_grads(passed, row0, Vt, Gt, B, K)
        torch.cuda.synchronize()
        assert torch.equal(grads[1], before[1]), "a copy without a buffer was written"
        for c in range(copies):
            if c == 1:
                continue
            for r in range(K):
                row = c * B * K + torch.arange(B, device = dev) * K + r
                assert torch.allclose(grads[c][r], before[c][r] - Vt[:, row].float() * Gt[:, row], rtol = 1e-5, atol = 1e-5)
        # ... and every copy one buffer (`tie_external`): all the subtractions land, none lost to another
        one = before[0].clone()
        mod.lowrank_normalizer_grads([one] * copies, row0, Vt, Gt, B, K)
        torch.cuda.synchronize()
        want = before[0].double()
        for c in range(copies):
            for r in range(K):
                row = c * B * K + torch.arange(B, device = dev) * K + r
                want[r] -= (Vt[:, row].float() * Gt[:, row]).double()
        assert torch.allclose(one.double(), want, rtol = 1e-3, atol = 1e-3)


def _stream_form(monkeypatch, form):
    """Forces the virtual batch's fused stream forward to one form, its tuner's choice (`virtual._stream_run`): the
    pre-pass chain's first arm (`form` "chain") or the one-launch form ("om"). Returns the list each choice goes to:
    `(code, whether the layer took the product layer below fused in)`."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.kernels import autotune

    taken, choose = [], autotune.choose

    def forced(key, cands, *args, **kwargs):
        if key[0] != "lowrank_stream_fw6":
            return choose(key, cands, *args, **kwargs)
        codes = [c for c, _ in cands if (c == lowrank_virtual._OM_ARM) == (form == "om")]
        assert codes, f"no {form} arm among {[c for c, _ in cands]}"
        taken.append((codes[0], key[9]))                               # (the key's `fused`: `virtual._stream_run`)
        return codes[0]

    monkeypatch.setattr(autotune, "choose", forced)
    return taken


@needs_virtual
@pytest.mark.parametrize("fusion", ["fused", "unfused"])
@pytest.mark.parametrize("form,bs,batch,rank", [("om", 32, 16, 16), ("om", 64, 16, 16), ("om", 128, 4, 16),
                                                ("om", 32, 12, 16), ("chain", 32, 16, 16), ("chain", 64, 32, 8),
                                                ("chain", 128, 16, 4)])
def test_the_fused_stream_forward_is_the_plain_virtual_path(form, bs, batch, rank, fusion, monkeypatch):
    """The virtual batch's fused stream forward (`virtual._stream_run`, `kernels/c/lowrank_stream.cu`: the virtual child
    values never stored), each form forced -- the pre-pass chain; the one-launch form, A by TMA in blocks of 32 and by
    cp.async in blocks of 64 (two a tile) and 128, at batches it alone serves (4, 12) --, the product layers handed to
    the sum layers or not: against the plain virtual path on the same circuit (bf16 products both, under other
    shifts) every value, flow, denominator and gradient, the same bits from run to run, and the torch passes (fp32
    parameters) within the bf16 copy's rounding."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.kernels import autotune

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: fusion if key[0] == "prod_stream_fusion"
                        else "unfused" if key[0] == "prod_stream_fusion_bk" else pick(key, cands, bench, **k))
    pc, sums, _, _, _, _ = _pair(_hmm_blocks(256, bs), "sum_exp", rank = rank)
    monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)   # (the bf16 copy: the stream path's)
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)                # (bf16 normalizers: the one-launch form's V)
    dev = pc.params.device
    torch.manual_seed(31)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 410 + i) for i, a in enumerate(sums)}
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    torch_pass = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM", False)
    plain = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM", True)
    taken = _stream_form(monkeypatch, form)
    got = _passes(pc, sums, data, ext)
    again = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple}
    # every layer took the form, the first (its children the inputs) without products, the others as the circuit chose
    # (a choice per layer the first pass; the next keeps it on the plan, asking again only where the tuner's cache
    # holds another arm for the shape -- what another test's tuner chose)
    assert len(taken) >= len(sums) and all((c == lowrank_virtual._OM_ARM) == (form == "om") for c, _ in taken)
    assert {f for _, f in taken} == ({True, False} if fusion == "fused" else {False})
    _same(got, plain, rtol = 2e-3, atol = 2e-3, grad_scale = 2e-2)
    for k in ("lls", "mars", "flows", "fp", "fm"):
        assert torch.equal(got[k], again[k]), k
    for k in ("dA", "dB"):
        assert all(torch.equal(a, b) for a, b in zip(got[k], again[k])), k
    _same(got, torch_pass, rtol = 2e-2, atol = 1e-3, grad_scale = 2e-2)


@needs_virtual
@pytest.mark.parametrize("bs,batch,param_flows,plain_middle,da_ctas", [(32, 16, True, False, 0), (64, 16, True, False, 0),
                                                                        (128, 4, True, False, 0), (32, 12, True, False, 0),
                                                                        (32, 16, False, False, 0), (32, 16, True, True, 0),
                                                                        (32, 16, False, True, 0), (64, 16, True, False, 1)])
def test_the_one_launch_backward_is_the_plain_data_half(bs, batch, param_flows, plain_middle, da_ctas, monkeypatch):
    """The virtual batch's one-launch backward (`virtual.FUSED_STREAM_BK`, `kernels/c/lowrank_stream.cu`'s
    `lr_om_bw_kernel`: the data half's element flows -- the product layer below handed to the backward, so its children's
    node flows --, dB's data half and its normalizer half, and, by the grid's first CTAs, dA), where it serves -- blocks of
    32, 64 and 128 (a child tile of 4, 2 and 1 child blocks, a parent block of 1, 2 and 4 k-steps), batches of 16, 12
    and 4 -- against the plain data half on the same circuit: every value, flow, denominator and gradient; the same bits
    from run to run; and the torch passes (fp32 parameters) within the bf16 copy's rounding. The first layer (its
    children the inputs: no product layer below) takes it too, writing the element flows, its children's values kept
    for the pass's end's parameter flows. `plain_middle`: a middle layer's backward on the plain path (its plan
    withheld): its copy alone owes the normalizer half, past the normalizers' first rows -- with the parameter flows
    (U's rows every copy's: the denominators read them) and without (the owing copy's rows alone). `da_ctas`: the dA
    CTAs capped (one), so it strides over the tiles of the (node, sample) pairs (at these sizes a CTA has one tile
    otherwise)."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.kernels import autotune

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused"
                        if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk") else pick(key, cands, bench, **k))
    rank = 16
    pc, sums, _, _, _, _ = _pair(_hmm_blocks(256, bs), "sum_exp", rank = rank)
    monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)   # (the bf16 copy: the stream path's)
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)                # (bf16 normalizers, as at real sizes)
    dev = pc.params.device
    torch.manual_seed(37)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 470 + i) for i, a in enumerate(sums)}
    # (a child of the first and of the last node vector cut off at sample 1 -- its B -inf at every rank: the node below
    # gets no flow there, its dA pairs take nothing, what the one-launch backward stores as 0 into the NaN-filled buffers)
    for a in (sums[0], sums[-1]):
        ext[a][1][1, :, 0] = -float("inf")
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    torch_pass = _passes(pc, sums, data, ext, param_flows = param_flows)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM_BK", False)
    plain = _passes(pc, sums, data, ext, param_flows = param_flows)
    smod = lowrank_virtual._stream_module()
    calls, real = [], smod.lowrank_backward_om

    def counted(*args):
        calls.append(args[22] is not None)                              # (whether it took dA: `grad_a`)
        return real(*args)

    monkeypatch.setattr(smod, "lowrank_backward_om", counted)
    smod.set_da_ctas_most(da_ctas)
    if plain_middle:
        middle, plan_bk = pc.external_params_nodes[sums[len(sums) // 2]], lowrank_virtual._stream_plan_bk
        monkeypatch.setattr(lowrank_virtual, "_stream_plan_bk",
                            lambda desc, layer, *args: None if layer is middle else plan_bk(desc, layer, *args))
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM_BK", True)
    try:
        got = _passes(pc, sums, data, ext, param_flows = param_flows)
        again = _passes(pc, sums, data, ext, param_flows = param_flows)
    finally:
        smod.set_da_ctas_most(0)
    assert got["path"] == {tuple}
    # every layer took it, each with dA -- but the middle one, `plain_middle`
    assert len(calls) == 2 * (len(sums) - plain_middle) and all(calls), calls
    _same(got, plain, rtol = 2e-3, atol = 2e-3, grad_scale = 2e-2)
    for k in ("lls", "mars", "flows", "fp", "fm"):
        assert torch.equal(got[k], again[k]), k
    for k in ("dA", "dB"):
        assert all(torch.equal(a, b) for a, b in zip(got[k], again[k])), k
    _same(got, torch_pass, rtol = 2e-2, atol = 1e-3, grad_scale = 2e-2)


@needs_virtual
def test_tied_copies_through_the_one_launch_backward_into_one_destination(monkeypatch):
    """`tie_external` through the one-launch backward (rank 16, blocks of 32): every copy reads the source's factors
    and adds its gradients into the source's one destination, handed over unzeroed (`StagedExternalParams.fresh`;
    NaN-filled here) -- the first copy's launch stores into it (it claims it), every other one adds --: the sums of the
    untied circuit's gradients over its copies, to fp32's order of the additions; every layer took the form."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.kernels import autotune

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused"
                        if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk") else pick(key, cands, bench, **k))
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)
    rank, batch = 16, 8
    pc_t, sums_t, _, _, _, _ = _pair(_hmm_blocks(256, 32), "sum_exp", rank = rank, tie_external = True)
    pc_a, sums_a, _, _, _, _ = _pair(_hmm_blocks(256, 32), "sum_exp", rank = rank)
    for pc in (pc_t, pc_a):
        monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)
    dev = pc_t.params.device
    torch.manual_seed(5)
    data = torch.randint(0, NUM_CATS, (batch, pc_t.num_vars), device = dev)
    A, B = _factors(sums_t[0], rank, batch, dev)
    smod = lowrank_virtual._stream_module()
    stores, real = [], smod.lowrank_backward_om

    def counted(*args):
        stores.append(args[-1])                                       # (whether the launch stored)
        return real(*args)

    monkeypatch.setattr(smod, "lowrank_backward_om", counted)
    nan = float("nan")
    for _ in range(2):                                                # (the circuit settles its choices at the first)
        gt = (torch.full_like(A, nan), torch.full_like(B, nan))
        pc_t(data, sum_external_params = {sums_t[0]: (A, B)})
        pc_t.init_param_flows(flows_memory = 0.0)
        stores.clear()
        pc_t.backward(data, allow_modify_flows = False, logspace_flows = True,
                      sum_external_params_grad = {sums_t[0]: gt})
    assert len(stores) == len(sums_t) and stores.count(True) == 1 and stores[0], stores
    ga = {a: (torch.full_like(A, nan), torch.full_like(B, nan)) for a in sums_a}
    for _ in range(2):
        ga = {a: (torch.full_like(A, nan), torch.full_like(B, nan)) for a in sums_a}
        pc_a(data, sum_external_params = {a: (A, B) for a in sums_a})
        pc_a.init_param_flows(flows_memory = 0.0)
        pc_a.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = ga)
    torch.cuda.synchronize()
    for k in range(2):
        want = sum(ga[a][k].double() for a in sums_a)
        assert torch.isfinite(gt[k]).all(), k
        err = float((gt[k].double() - want).abs().max() / want.abs().max())
        assert err < 1e-5, f"slot {k}: {err}"


@needs_virtual
def test_copies_in_layers_of_several_node_vectors(monkeypatch):
    """Two tied chains side by side: every layer holds two node vectors, each with copies in the pass. One plain
    backward a layer (the data half of both), the normalizer products once a source: two."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 2, 4
    pc, sums, _, _, _, _ = _pair("two_chains", "sum_exp", rank = rank)
    dev = pc.params.device
    assert len(sums) == 6 and all(len(pc.external_params_nodes[a].external_node_infos) == 2 for a in sums)
    torch.manual_seed(15)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 100 + i) for i, a in enumerate(sums)}
    plain, norm = _plain_backwards(monkeypatch), _normalizer_calls(monkeypatch)
    got = _passes(pc, sums, data, ext)
    assert got["path"] == {tuple} and [c for c in plain if c != batch] == [batch * rank] * 3
    assert sorted(norm) == [(3 * batch * rank, torch.float32)] * 2
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    _same(got, _passes(pc, sums, data, ext), rtol = 2e-4, atol = 2e-5)


@needs_virtual
def test_a_backward_after_one_that_stopped_midway(monkeypatch):
    """A backward that raised after some copies were noted for the normalizer products leaves them noted. The next
    backward of the same forward starts over -- the first node vector seen again resets what the pass keeps -- and is
    a whole one: the torch passes' flows and gradients, one plain backward a layer, one pair of products."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.sum_layer import SumLayer

    # (no trials of the circuit's tuner: one that raised would be a candidate dropped, not the backward stopping)
    import pyjuice.layer.kernels.autotune as autotune
    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **kw:
                        "fused" if key[0] == "prod_stream_fusion_bk" else pick(key, cands, bench, **kw))

    rank, batch = 2, 4
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank, seq_length = 6)
    dev = pc.params.device
    torch.manual_seed(17)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 120 + i) for i, a in enumerate(sums)}
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)

    grads = {a: (torch.zeros_like(ext[a][0]), torch.zeros_like(ext[a][1])) for a in sums}
    pc(data, sum_external_params = ext)
    plain, seen = SumLayer.backward, []

    def stops(self, node_flows, *args, **kwargs):
        seen.append(node_flows.size(1))
        if len([c for c in seen if c == batch * rank]) == 3:        # the third transition's data half
            raise RuntimeError("stopped")
        return plain(self, node_flows, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(SumLayer, "backward", stops)
        pc.init_param_flows(flows_memory = 0.0)
        with pytest.raises(RuntimeError, match = "stopped"):
            pc.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)

    calls, norm = _plain_backwards(monkeypatch), _normalizer_calls(monkeypatch)
    for _ in range(2):
        calls.clear(); norm.clear()
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)
        assert len([c for c in calls if c != batch]) == 5 and len(norm) == 1      # five data halves, one pair
        assert torch.allclose(pc.param_flows, want["fp"], rtol = 2e-4, atol = 2e-5)
        assert torch.allclose(_denominators(pc, sums), want["fm"], rtol = 2e-4, atol = 2e-5)
        for k, a in enumerate(sums):
            assert torch.allclose(grads[a][0], want["dA"][k], rtol = 2e-4, atol = 2e-5)
            assert torch.allclose(grads[a][1], want["dB"][k], rtol = 2e-4, atol = 2e-5)


@pytest.mark.parametrize("form", FORMS)
def test_nodes_no_child_of_which_is_possible(form, monkeypatch):
    """A node block that reaches one child block, impossible for some samples (an input of probability 0): its
    nodes' values are -inf there and they carry no flow -- finite flows and gradients everywhere, the materialized
    circuit's, through the virtual batch and the torch passes alike."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    def impossible(root):
        for ns in root:
            if ns.is_input() and not ns.is_tied() and ns.scope.to_list() == [0]:
                p = ns.get_params().reshape(ns.num_nodes, NUM_CATS).clone()
                p[4:6, 0] = 0.0                                     # (child block 2's nodes of variable 0)
                ns.set_params(p)

    rank, batch = 3, 6
    for enabled in (True, False):
        monkeypatch.setattr(lowrank_virtual, "ENABLED", enabled)
        pc_a, sums_a, pc_b, sums_b, _, _ = _pair("sparse", form, rank = rank, prepare = impossible)
        dev = pc_a.params.device
        torch.manual_seed(18)
        data = torch.randint(0, NUM_CATS, (batch, pc_a.num_vars), device = dev)
        data[:3, 0] = 0
        ext = {a: _factors(a, rank, batch, dev, seed = 130 + i) for i, a in enumerate(sums_a)}
        grads = {a: (torch.zeros_like(ext[a][0]), torch.zeros_like(ext[a][1])) for a in sums_a}
        lls = pc_a(data, sum_external_params = ext).reshape(-1)
        pc_a.init_param_flows(flows_memory = 0.0)
        pc_a.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)
        a = sums_a[0]
        rows = slice(*a._output_ind_range)
        assert bool(torch.isneginf(pc_a.node_mars[rows][:4, :3]).all()) and bool(torch.isfinite(lls).all())
        assert bool(torch.isneginf(pc_a.node_flows[rows][:4, :3]).all())
        for x in (pc_a.param_flows, pc_a.denom_param_flows, grads[a][0], grads[a][1]):
            assert bool(torch.isfinite(x).all())
        ref = _reference(pc_a, sums_a, pc_b, sums_b, data, ext, form)
        assert _close(lls, ref["lls"], rtol = 0, atol = 2e-3)
        assert _close(_dense(a, _flows(pc_a, a)), ref["Fp"][0]) and _close(grads[a][0], ref["dA"][0])
        assert _close(grads[a][1], ref["dB"][0])
        fm = a.parameterization.materialize_denom_flows(a, pc_a.params, pc_a.denom_param_flows,
                                                        [(pc_a.external_params_nodes[a], a)])
        E = a.edge_ids.size(1)
        assert _close(_dense(a, fm.reshape(E, a.ch_block_size, a.block_size).permute(0, 2, 1)), ref["Fm"][0])


@needs_virtual
@pytest.mark.parametrize("name,offered", [("hmm_tied", True), ("two_disjoint", True), ("two_chains", True),
                                          ("sparse", False)])
def test_the_forward_with_the_product_layer_fused_in(name, offered, monkeypatch):
    """Where the circuit leaves the product layer below to this layer's forward (its tuner's choice, forced here),
    the virtual batch reads every product's value from its two children: the bits of the forward that was handed the
    products' values, forward after forward; and a layer that does not go through the virtual batch after all forms
    the products' values itself. Products of two children only (`sparse`: three, never handed over)."""
    import pyjuice.layer.kernels.autotune as autotune
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    pick = autotune.pick
    choice = ["fused"]
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **kw:
                        choice[0] if key[0] == "prod_stream_fusion" else pick(key, cands, bench, **kw))
    fused = []
    expand = lowrank_virtual._module().lowrank_expand_fused

    class Counted:
        def __getattr__(self, attr):
            return getattr(lowrank_virtual._module.__wrapped__(), attr)

        def lowrank_expand_fused(self, *args):
            fused.append(1)
            return expand(*args)

    real = lowrank_virtual._module
    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)
    lowrank_virtual._module.__wrapped__ = real

    rank, batch = 2, 5
    out = dict()
    for which in ("fused", "unfused"):
        choice[0] = which
        pc, sums, _, _, _, _ = _pair(name, "sum_exp", rank = rank)
        dev = pc.params.device
        torch.manual_seed(19)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 140 + i) for i, a in enumerate(sums)}
        fused.clear()
        lls = [pc(data, sum_external_params = ext).clone() for _ in range(4)]
        assert all(torch.equal(l, lls[0]) for l in lls)
        assert (len(fused) > 0) == (which == "fused" and offered)
        out[which] = (lls[0], _passes(pc, sums, data, ext))
        # Other data next: the products' values the last pass left in the circuit's buffer are not these
        torch.manual_seed(20)
        other = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        out[which] += (pc(other, sum_external_params = ext).clone(),)
        if which == "fused":
            # ... and the torch passes, handed the product layer the same way, form its values first
            monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
            torch_pass = _passes(pc, sums, data, ext)
            monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
            assert torch_pass["path"] == {dict}
            _same(out[which][1], torch_pass, rtol = 2e-4, atol = 2e-5)
    assert torch.equal(out["fused"][0], out["unfused"][0]) and torch.equal(out["fused"][2], out["unfused"][2])
    assert not torch.equal(out["fused"][0], out["fused"][2])
    _same(out["fused"][1], out["unfused"][1], rtol = 1e-6, atol = 1e-7)


def _calls(own = 0, own_fused = 0, merged = 0, collapse = 0):
    out = dict(lowrank_expand = own, lowrank_expand_fused = own_fused, lowrank_collapse_expand = merged,
               lowrank_collapse = collapse)
    return {k: v for k, v in out.items() if v}


AHEAD = dict(
    # case -> (the structure, how it is built, a forward's calls once the layers know each other): the deepest
    # layer's own `expand` (products of one child: not handed over), the top layer's collapse alone
    hmm_tied = ("hmm_tied", dict(), _calls(own = 1, merged = 2, collapse = 1)),
    hmm_untied = ("hmm_untied", dict(), _calls(own = 1, merged = 2, collapse = 1)),
    hmm_side = ("hmm_side", dict(), _calls(own = 1, merged = 2, collapse = 1)),
    two_chains = ("two_chains", dict(), _calls(own = 2, merged = 4, collapse = 2)),
    # one node vector above two of a layer: its `expand` after both collapses, from the layer below's call
    fan_in = ("fan_in", dict(), _calls(own_fused = 3, collapse = 3)),
    # two node vectors above one: the first in its launch, the second's `expand` after it
    fan_out = ("fan_out", dict(), _calls(own_fused = 2, merged = 1, collapse = 2)),
    # ranks of their own (2 under 3 under 2): the launch writes the layer above's columns, not its own
    two_ranks = ("hmm_untied", dict(ranks = (2, 3, 2)), _calls(own = 1, merged = 2, collapse = 1)),
    # two layers with factors in every layer group, one buffer: each overwrites what the other was left -- given up
    two_kinds = ("two_chains", dict(second = OTHER_KIND), _calls(own = 2, own_fused = 4, collapse = 6)),
    # a layer with factors beside the deepest transition's, run before it: nothing between the transitions
    side = ("hmm_side", dict(side = OTHER_KIND), _calls(own = 2, merged = 2, collapse = 2)),
    # one beside the middle transition, over children of no layer with factors: run before or after it, it comes
    # between a transition and the one above -- that pair gives up, the other pair keeps its launch
    beside = (_beside, dict(), _calls(own = 1, own_fused = 2, merged = 1, collapse = 3)),
    beside_flip = (_beside, dict(flip = True), _calls(own = 1, own_fused = 2, merged = 1, collapse = 3)),
)


def _own_factors(sums, batch, dev, seed):
    return {a: _factors(a, a.parameterization.rank, batch, dev, seed = seed + i) for i, a in enumerate(sums)}


@needs_virtual
@pytest.mark.parametrize("case", sorted(AHEAD))
def test_a_collapse_takes_the_next_layers_expand_along(case, monkeypatch):
    """A layer whose nodes are the children of the next layer's products writes that layer's virtual child values in
    the launch of its own collapse, once that layer was handed the product group between (the circuit's fused
    forward, its choice forced here after its trials): the bits of the forward in which every layer expands its own
    -- values, and through the backward the flows, both parameter flows and the gradients (the collapse's kept
shares) --, on other data, at another batch size (the trials there run every layer several times: the launches
    stay), and where the last node vector's layer is given no factors (nothing is written for it). A node vector of the next
    layer whose children lie in two of this layer's is expanded after both collapses (`fan_in`), as is the second of
    two over one (`fan_out`); where another layer with factors writes the buffer in between, the values left there
    are not read, and are no longer written after the second time (`two_kinds`, `beside` -- a layer under the one
    below, run again by the circuit's trials, is not "in between")."""
    import pyjuice.layer.kernels.autotune as autotune
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    name, kw, want = AHEAD[case]
    pick = autotune.pick

    def forced(key, cands, bench, **k):
        if key[0] == "prod_stream_fusion":
            for trial in ("fused", "unfused", "fused", "fused"):
                bench(trial)
            return "fused"
        return "unfused" if key[0] == "prod_stream_fusion_bk" else pick(key, cands, bench, **k)

    monkeypatch.setattr(autotune, "pick", forced)
    real = lowrank_virtual._module
    counts = dict()

    class Counted:
        def __getattr__(self, attr):
            fn = getattr(real(), attr)
            if attr not in ("lowrank_collapse_expand", "lowrank_expand_fused", "lowrank_expand", "lowrank_collapse"):
                return fn

            def counted(*args):
                counts[attr] = counts.get(attr, 0) + 1
                return fn(*args)
            return counted

    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)

    def forward_calls(pc, x, ext):
        pc(x, sum_external_params = ext)                         # (the circuit's trials, if any, in this one)
        counts.clear()
        pc(x, sum_external_params = ext)
        return dict(counts)

    batch = 5
    out = dict()
    for linked in (True, False):
        monkeypatch.setattr(lowrank_virtual, "LINK_TRIES", 64 if linked else 0)
        pc, sums, _, _, _, _ = _pair(name, "sum_exp", rank = 2, **kw)
        dev = pc.params.device
        torch.manual_seed(27)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        other = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = _own_factors(sums, batch, dev, 190)
        passes, seen = [], []
        for x in (data, data, other, data, data, data):               # (other data right after the layers linked)
            passes.append(_passes(pc, sums, x, ext))
        seen.append(forward_calls(pc, data, ext))
        # Another batch size: the circuit decides again, running every layer several times
        small = torch.randint(0, NUM_CATS, (3, pc.num_vars), device = dev)
        ext3 = _own_factors(sums, 3, dev, 230)
        for _ in range(3):
            passes.append(_passes(pc, sums, small, ext3))
        seen.append(forward_calls(pc, small, ext3))
        # The last node vector's layer without factors: a plain one in this pass
        given = [a for a in sums if pc.external_params_nodes[a] is not pc.external_params_nodes[sums[-1]]]
        part = {a: ext[a] for a in given}
        passes.append(_passes(pc, given, data, part))
        seen.append(forward_calls(pc, data, part))
        passes.append(_passes(pc, sums, data, ext))
        out[linked] = (passes, seen)
        if case == "fan_in" and linked:
            link = pc.external_params_nodes[sums[0]].__dict__["_lrv_link"]
            assert link["pairs"] == dict() and link["rest"] == [0]
    assert all("lowrank_collapse_expand" not in c for c in out[False][1])
    assert out[True][1][0] == want and out[True][1][1] == want, out[True][1]
    for a, b in zip(out[True][0], out[False][0]):
        for k in ("lls", "mars", "flows", "fp", "fm"):
            assert torch.equal(a[k], b[k]), k
        for k in ("dA", "dB"):
            assert all(torch.equal(x, y) for x, y in zip(a[k], b[k])), k
    assert not torch.equal(out[True][0][1]["lls"], out[True][0][2]["lls"])


@needs_virtual
def test_nothing_is_written_for_a_layer_above_that_keeps_its_torch_passes(monkeypatch):
    """A layer whose virtual buffers would pass the bound keeps its torch passes; the layer below it then writes
    nothing for it. Ranks 2 under 3 under 2, the bound between the two ranks' needs, set after the layers know
    each other."""
    import pyjuice.layer.kernels.autotune as autotune
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused" if key[0] == "prod_stream_fusion"
                        else "unfused" if key[0] == "prod_stream_fusion_bk" else pick(key, cands, bench, **k))
    real = lowrank_virtual._module
    merged = []

    class Counted:
        def __getattr__(self, attr):
            fn = getattr(real(), attr)
            if attr != "lowrank_collapse_expand":
                return fn

            def counted(*args):
                merged.append(args[-1])                              # (the rank written for)
                return fn(*args)
            return counted

    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)

    batch = 5
    pc, sums, _, _, _, _ = _pair("hmm_untied", "sum_exp", rank = 2, ranks = (2, 3, 2))
    dev = pc.params.device
    torch.manual_seed(33)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = _own_factors(sums, batch, dev, 250)
    for _ in range(4):
        free = _passes(pc, sums, data, ext)
    merged.clear()
    pc(data, sum_external_params = ext)
    assert sorted(merged) == [2, 3]

    rows = lowrank_virtual._twin(pc.external_params_nodes[sums[0]])[0]._layer_nid_range[1]     # (a layer's: all alike)
    need = lambda rank: lowrank_virtual._bytes(rows, pc.element_mars, batch * rank)
    monkeypatch.setattr(lowrank_virtual, "VIRTUAL_BYTES_MOST", (need(2) + need(3)) // 2)
    merged.clear()
    bound = _passes(pc, sums, data, ext)
    paths = [type(pc.external_params_nodes[a].__dict__["_lr_state"]) for a in sums]
    assert paths == [tuple, dict, tuple] and merged == []
    _same(bound, free, rtol = 2e-4, atol = 2e-5)


@needs_virtual
@pytest.mark.parametrize("between", [False, True])
def test_child_values_left_by_the_layer_below_are_read_only_if_nothing_wrote_since(between, monkeypatch):
    """The layers run by hand, on the pass's staged tensors: a transition, then the one above it, which finds its
    virtual child values written and launches no `expand`; with another layer with factors run in between -- its
    `expand` writes the same rows of the buffer -- the one above expands its own again. The circuit's node values
    either way."""
    import pyjuice.layer.kernels.autotune as autotune
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused" if key[0] == "prod_stream_fusion"
                        else "unfused" if key[0] == "prod_stream_fusion_bk" else pick(key, cands, bench, **k))
    real = lowrank_virtual._module
    expands = []

    class Counted:
        def __getattr__(self, attr):
            fn = getattr(real(), attr)
            if attr not in ("lowrank_expand", "lowrank_expand_fused"):
                return fn

            def counted(*args):
                expands.append(attr)
                return fn(*args)
            return counted

    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)

    rank, batch = 2, 5
    pc, sums, _, _, _, _ = _pair("hmm_side", "sum_exp", rank = rank, side = OTHER_KIND)
    dev = pc.params.device
    torch.manual_seed(31)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 210 + i) for i, a in enumerate(sums)}
    for _ in range(4):
        lls = pc(data, sum_external_params = ext)
    want = pc.node_mars.clone()

    layer = {a: pc.external_params_nodes[a] for a in sums}
    low, up, side = layer[sums[0]], layer[sums[1]], layer[sums[-1]]
    assert low.__dict__["_lrv_link"]["above"]() is up and side is not low
    group = lambda lay: next(i for i, g in enumerate(pc.inner_layer_groups) if any(l is lay for l in g.layers))
    assert group(side) == group(low) and group(up) == group(low) + 2
    kwargs = dict(sum_external_params = pc._staged_external_params)
    products = lambda lay: pc.inner_layer_groups[group(lay) - 1].forward(pc.node_mars, pc.element_mars)

    products(low)
    low.forward(pc.node_mars, pc.element_mars, pc.params, **kwargs)
    if between:
        side.forward(pc.node_mars, pc.element_mars, pc.params, **kwargs)
    products(up)
    expands.clear()
    up.forward(pc.node_mars, pc.element_mars, pc.params, **kwargs)
    assert expands == (["lowrank_expand"] if between else [])
    rows = slice(*up._layer_nid_range)
    assert torch.equal(pc.node_mars[rows], want[rows])
    # (the rows that layer writes are the ones the upper transition's children stand in)
    kids = lambda lay: lay.parameterization._geometry(lay, lay.external_node_infos[0], dev)["rows"]
    assert set(kids(side).tolist()) == set(kids(up).tolist())


@needs_virtual
@pytest.mark.parametrize("name,offered", [("hmm_tied", True), ("two_chains", True), ("hmm_side", True),
                                          ("two_disjoint", False), ("sparse", False)])
@pytest.mark.parametrize("forward", ["unfused", "fused"])
def test_the_backward_with_the_product_layer_fused_in(name, offered, forward, monkeypatch):
    """Where the circuit leaves the product layer below to this layer's backward (its tuner's choice, forced here, after
    its trials on scratch flows), the virtual batch reads every product's value from its two children and writes the
    product's flow as theirs: the bits of the backward between a product layer's recompute and its own backward --
    node flows, both parameter flows, the gradients --, pass after pass, on other data too, whichever way the forward
    went; and a layer that does not go through the virtual batch after all is given the products' values and its
    element flows are written as the children's. Products of two children, each the child of that one product
    (`sparse`: three children; `two_disjoint`: an input the child of two products -- never handed over)."""
    import pyjuice.layer.kernels.autotune as autotune
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    pick = autotune.pick
    choice = ["fused"]

    def forced(key, cands, bench, **kw):
        if key[0] == "prod_stream_fusion":
            return forward
        if key[0] == "prod_stream_fusion_bk":
            count = len(fused)
            for trial in ("fused", "unfused", "fused"):       # (the circuit's trials: this pass's layers, other flows)
                bench(trial)
            del fused[count:]
            return choice[0]
        return pick(key, cands, bench, **kw)

    monkeypatch.setattr(autotune, "pick", forced)
    fused = []
    real = lowrank_virtual._module
    collapse = real().lowrank_collapse_bw_fused

    class Counted:
        def __getattr__(self, attr):
            return getattr(real(), attr)

        def lowrank_collapse_bw_fused(self, *args):
            fused.append(1)
            return collapse(*args)

    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)

    rank, batch = 2, 5
    out = dict()
    for which in ("fused", "unfused"):
        choice[0] = which
        pc, sums, _, _, _, _ = _pair(name, "sum_exp", rank = rank)
        dev = pc.params.device
        torch.manual_seed(23)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 170 + i) for i, a in enumerate(sums)}
        fused.clear()
        first = _passes(pc, sums, data, ext)                     # (the circuit decides at its second backward)
        assert not fused
        passes = [_passes(pc, sums, data, ext) for _ in range(3)]
        assert (len(fused) > 0) == (which == "fused" and offered)
        for got in passes:
            for k in ("flows", "fm"):
                assert torch.equal(got[k], first[k]), k
            for k in ("dA", "dB"):
                assert all(torch.equal(a, b) for a, b in zip(got[k], first[k])), k
            # (the parameter flows: one matrix product where the layer is fused, the plain kernels' before it decided)
            assert torch.equal(got["fp"], passes[0]["fp"])
            assert torch.allclose(got["fp"], first["fp"], rtol = 1e-4, atol = 1e-6)
        # Other data next: the products' values and flows the last pass left in the circuit's buffers are not these
        torch.manual_seed(24)
        other = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        out[which] = _passes(pc, sums, other, ext)
        assert not torch.equal(out[which]["flows"], first["flows"])
        if which == "fused":
            # ... and the torch passes, handed the product layer the same way
            monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
            torch_pass = _passes(pc, sums, other, ext)
            monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
            assert torch_pass["path"] == {dict}
            _same(out[which], torch_pass, rtol = 2e-4, atol = 2e-5)
    for k in ("flows", "fm"):
        assert torch.equal(out["fused"][k], out["unfused"][k]), k
    assert torch.allclose(out["fused"]["fp"], out["unfused"]["fp"], rtol = 1e-4, atol = 1e-6)
    for k in ("dA", "dB"):
        assert all(torch.equal(a, b) for a, b in zip(out["fused"][k], out["unfused"][k])), k


def _force_fused_backward(monkeypatch):
    """The circuit hands every product layer it can to the sum layer above in the backward, whatever its tuner would
    choose (it decides at its second backward)."""
    import pyjuice.layer.kernels.autotune as autotune

    pick = autotune.pick

    def forced(key, cands, bench, **kw):
        return "fused" if key[0] == "prod_stream_fusion_bk" else pick(key, cands, bench, **kw)

    monkeypatch.setattr(autotune, "pick", forced)


@needs_virtual
@pytest.mark.parametrize("name,sources,fused_copies", [("hmm_tied", 1, 2), ("two_chains", 2, 4), ("hmm_side", 1, 2)])
@pytest.mark.parametrize("operands", ["fp32", "bf16"])
def test_the_parameter_flows_of_fused_layers_are_one_matrix_product(name, sources, fused_copies, operands, monkeypatch):
    """Where the product layer below is fused in, a layer's plain backward adds no parameter flows: all the fused copies
    of a source are one matrix product at the pass's end, `theta * (U^T V)` over their rows -- `F+` within rounding of
    the plain kernels' on the same forward and of the torch passes', with `F-`, the gradients and the node flows
    untouched; a layer not fused keeps its plain kernel."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.sum_layer import SumLayer

    _force_fused_backward(monkeypatch)
    if operands == "bf16":
        monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)
    rtol = 2e-4 if operands == "fp32" else 6e-3

    real, calls = lowrank_virtual._module, []

    class Counted:
        def __getattr__(self, attr):
            fn = getattr(real(), attr)
            if attr not in ("lowrank_data_operands", "lowrank_kept_operands"):
                return fn

            def counted(Ut, Vt, *args):                          # (the data half's U: from the factors, or kept)
                if Ut is not None:
                    calls.append((Ut.size(1), Ut.dtype))
                return fn(Ut, Vt, *args)
            return counted

    counted = Counted()
    monkeypatch.setattr(lowrank_virtual, "_module", lambda: counted)
    plain, with_flows = SumLayer.backward, []

    def plain_counted(self, node_flows, *args, **kwargs):
        if node_flows.size(1) == batch * rank:                   # (a virtual batch's: the root's and a plain sum's keep theirs)
            with_flows.append(kwargs.get("param_flows") is not None)
        return plain(self, node_flows, *args, **kwargs)

    monkeypatch.setattr(SumLayer, "backward", plain_counted)

    rank, batch = 2, 5
    pc, sums, _, _, _, _ = _pair(name, "sum_exp", rank = rank)
    dev = pc.params.device
    torch.manual_seed(31)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 210 + i) for i, a in enumerate(sums)}
    unfused = _passes(pc, sums, data, ext)                       # (the circuit decides at its second backward)
    assert calls == []
    with_flows.clear()
    got = _passes(pc, sums, data, ext)
    dtype = torch.float32 if operands == "fp32" else torch.bfloat16
    assert len(calls) == sources and sum(r for r, _ in calls) == fused_copies * batch * rank, calls
    assert all(t == dtype for _, t in calls), calls
    assert with_flows == [False, False, True], with_flows         # (a layer's plain backward, the deepest's not fused)

    assert torch.allclose(got["fp"], unfused["fp"], rtol = rtol, atol = 1e-5), \
        f"{float(((got['fp'] - unfused['fp']).abs() / unfused['fp'].abs().clamp_min(1e-5)).max())}"
    assert torch.equal(got["flows"], unfused["flows"]) and torch.equal(got["mars"], unfused["mars"])
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    ref = _passes(pc, sums, data, ext)
    assert ref["path"] == {dict}
    assert torch.allclose(got["fp"], ref["fp"], rtol = rtol, atol = 1e-5)


@needs_virtual
def test_the_parameter_flows_of_copies_in_several_flow_blocks(monkeypatch):
    """Tied copies in one layer past `max_tied_ns_per_parflow_block` have flow blocks of their own, summed before the
    parameters read them. The product lands in the first's: the EM step takes the parameters where the torch passes do."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    _force_fused_backward(monkeypatch)
    rank, batch, dev, K = 2, 5, torch.device("cuda:0"), 8
    out = dict()
    for arm in ("virtual", "torch"):
        monkeypatch.setattr(lowrank_virtual, "ENABLED", arm == "virtual")
        torch.manual_seed(0)
        with juice.set_block_size(K):
            sums, parts = [], []
            for t in range(4):                                    # one layer of tied sums over two inputs each
                m = multiply(*[inputs(2 * t + v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
                               for v in range(2)])
                ns = summate(m, num_node_blocks = 1, parameterization = LowRank(rank = rank, form = "sum_exp")) \
                    if not sums else sums[0].duplicate(m, tie_params = True)
                sums.append(ns)
            root = summate(multiply(*sums), num_node_blocks = 1, block_size = 1)
        torch.manual_seed(0)
        root.init_parameters(perturbation = 2.0)
        pc = juice.compile(root, verbose = False, max_tied_ns_per_parflow_block = 2).to(dev)
        assert len({a._param_flow_range for a in sums}) > 1, "one flow block: nothing is tested"
        torch.manual_seed(3)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 230 + i) for i, a in enumerate(sums)}
        for _ in range(3):                                        # (the circuit decides at its second backward)
            _passes(pc, sums, data, ext)
            pc.mini_batch_em(step_size = 0.5, pseudocount = 1e-6)
        out[arm] = pc.params.clone()
    assert torch.allclose(out["virtual"], out["torch"], rtol = 1e-3, atol = 1e-5)


def _kept_children(node_mars, table, first, eid):
    """What a copy without its product layer fused in keeps of its children, `[N, B]`: the products' values (here the
    sum of their two children's node values, the kernels' own fp32 add), -inf at a child no edge reaches."""
    at = (eid.clamp_min(first) - first).long()
    cv = node_mars[table[at, 0].long()] + node_mars[table[at, 1].long()]
    return torch.where((eid >= 0)[:, None], cv, torch.full_like(cv, -float("inf"))).contiguous()


@needs_virtual
@pytest.mark.parametrize("M, N, K, B, copies", [(3, 300, 2, 37, 2),     # children past one warp's share, samples past a block's
                                                (5, 3, 3, 4, 70)])      # few children, copies past one launch's
def test_the_data_operands_follow_their_formulas(M, N, K, B, copies):
    """The kernel's contract, checked directly. With `v[r, j, b] = c_b[j] + B[r, j, b]`, `c_b[j]` the sum of the node values
    of the product's two children and the shift the row's largest `v`: `Vt[j, (c, b, r)] = exp(v[r, j, b] - shift)`, and
    `Ut[i, (c, b, r)] = exp(lf[i, b] + A[i, r, b] - log S[i, b] + shift)`, `log S = node_mars + log Z`. A child no edge
    reaches, a child without mass: 0. A node without flow or mass: 0. A rank with no mass at a sample: zeros, no NaN. float32
    and bfloat16 operands."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    mod, dev = lowrank_virtual._module(), torch.device("cuda")
    g = torch.Generator(device = "cpu").manual_seed(11)
    children = 13                                                      # the rows children stand in
    fa = [(torch.randn(M, K, B, generator = g) * 2).to(dev) for _ in range(copies)]
    fb = [(torch.randn(K, N, B, generator = g) * 2).to(dev) for _ in range(copies)]
    lz = [(torch.randn(M, B, generator = g) * 2).to(dev) for _ in range(copies)]
    rows = children + copies * M
    node_flows = (torch.randn(rows, B, generator = g) - 1.0).to(dev)
    node_mars = (torch.randn(rows, B, generator = g) - 3.0).to(dev)
    nid0 = [children + c * M for c in range(copies)]
    first, elements = 5, N + 7
    table = torch.randint(0, children, (elements, 2), generator = g, dtype = torch.int32).to(dev)
    eids = [(first + torch.randperm(elements, generator = g)[:N]).to(dev) for _ in range(copies)]
    eids[0][1] = -1                                                    # copy 0, child 1: no edge reaches it
    node_mars[int(table[int(eids[1][0]) - first, 0]), 2] = -float("inf")   # a child without mass (and whatever shares its row)
    node_flows[nid0[0] + 1, 2] = -float("inf")                         # copy 0, node 1, sample 2: no flow
    node_mars[nid0[1] + 0, 3] = -float("inf")                          # copy 1, node 0, sample 3: no mass
    fb[0][1, :, 1] = -float("inf")                                     # copy 0, rank 1, sample 1: the rank has no mass

    total = copies * B * K
    for dtype in (torch.float32, torch.bfloat16):
        tol = dict(rtol = 1e-5, atol = 1e-6) if dtype is torch.float32 else dict(rtol = 1e-2, atol = 1e-3)
        Ut = torch.full((M, total), float("nan"), dtype = dtype, device = dev)
        Vt = torch.full((N, total), float("nan"), dtype = dtype, device = dev)
        mod.lowrank_data_operands(Ut, Vt, node_flows, node_mars, fa, lz, fb, nid0, eids, [table] * copies,
                                  [first] * copies, M, K, [None] * copies)
        torch.cuda.synchronize()
        assert torch.isfinite(Ut.float()).all() and torch.isfinite(Vt.float()).all()
        # (the children's values kept instead of the table -- a copy whose product layer was not fused in: the same bits)
        Uk, Vk = torch.full_like(Ut, float("nan")), torch.full_like(Vt, float("nan"))
        mod.lowrank_data_operands(Uk, Vk, node_flows, node_mars, fa, lz, fb, nid0, eids, [None] * copies,
                                  [0] * copies, M, K, [_kept_children(node_mars, table, first, e) for e in eids])
        assert torch.equal(Uk, Ut) and torch.equal(Vk, Vt), "the kept children's values"
        for c in range(copies):
            has = eids[c] >= 0
            at = (eids[c].clamp_min(first) - first).long()
            cv = node_mars[table[at, 0].long()] + node_mars[table[at, 1].long()]    # [N, B]
            cv = torch.where(has[:, None], cv, torch.full_like(cv, -float("inf")))
            lf = node_flows[nid0[c]:nid0[c] + M]
            ls = node_mars[nid0[c]:nid0[c] + M] + lz[c]
            for r in range(K):
                row = c * B * K + torch.arange(B, device = dev) * K + r
                v = cv + fb[c][r]                                              # [N, B]
                sh = v.max(dim = 0).values                                     # [B]
                want_v = torch.where(torch.isneginf(v), torch.zeros_like(v), torch.exp(v - sh))
                assert torch.allclose(Vt[:, row].float(), want_v, **tol), f"V: copy {c}, rank {r}"
                want_u = torch.exp(lf + fa[c][:, r] - ls + sh)
                dead = torch.isneginf(lf) | torch.isneginf(ls)
                want_u = torch.where(dead, torch.zeros_like(want_u), want_u)
                assert torch.allclose(Ut[:, row].float(), want_u, **tol), f"U: copy {c}, rank {r}"
        assert all(float(Ut[1, 2 * K + r]) == 0.0 for r in range(K))
        assert all(float(Ut[0, B * K + 3 * K + r]) == 0.0 for r in range(K))
        assert float(Vt[1, :B * K].abs().max()) == 0.0
        assert float(Vt[:, K + 1].abs().max()) == 0.0 and float(Ut[:, K + 1].abs().max()) == 0.0


@needs_virtual
@pytest.mark.parametrize("in_place", [True, False], ids = ["in_place", "batch_innermost"])
@pytest.mark.parametrize("M, K, B, copies", [(5, 16, 7, 3), (40, 8, 33, 2), (3, 32, 4, 70), (6, 4, 9, 2), (7, 3, 5, 2)])
def test_the_a_operand_is_exp_a_under_each_node_and_samples_largest(M, K, B, copies, in_place):
    """`lowrank_a_operand` (A batch innermost, [M, K, B]) and `lowrank_a_operand_in_place` (A as the caller gives it,
    [B, M, K], a wider batch stride), bf16 and fp32: `am[c, i, b]` each (node, sample)'s largest A over its K ranks,
    `Ua[i, (c, b, r)] = exp(A[c][b, i, r] - am[c, i, b])` -- ranks 8, 16, 32 held in registers, 3 and 4 not; copies past
    one launch's --; a (node, sample) of no rank with mass (every A -inf): am -inf, its row of Ua 0, no NaN."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    mod, dev = lowrank_virtual._module(), torch.device("cuda")
    if in_place and not mod.lowrank_in_place_rank(K):
        pytest.skip("the operands in place serve ranks 4, 8, 16, 32")
    g = torch.Generator(device = "cpu").manual_seed(29)
    fa = [(torch.randn(B, M, K, generator = g) * 3).to(dev) for _ in range(copies)]   # A[b, i, r]
    fa[0][1, M - 1, :] = -float("inf")                                  # (copy 0, its last node, sample 1: no mass)
    fa[-1][0, 0, 0] = -float("inf")                                     # (one rank of no mass)
    if in_place:   # (a wider batch stride: slices of a bigger tensor)
        given = [torch.cat([a, torch.zeros_like(a)], dim = 1)[:, :M] for a in fa]
    else:
        given = [a.permute(1, 2, 0).contiguous() for a in fa]          # [M, K, B]
    for dtype in (torch.bfloat16, torch.float32):
        Ua = torch.full((M, copies * B * K), float("nan"), dtype = dtype, device = dev)
        am = torch.full((copies, M, B), float("nan"), device = dev)
        (mod.lowrank_a_operand_in_place if in_place else mod.lowrank_a_operand)(Ua, am, given, B, K)
        torch.cuda.synchronize()
        for c, a in enumerate(fa):
            want_am = a.max(dim = 2).values.t()                         # [M, B]
            assert torch.equal(am[c], want_am), c
            want = torch.where(want_am.t()[:, :, None] > -float("inf"), torch.exp(a - want_am.t()[:, :, None]),
                               torch.zeros_like(a)).nan_to_num(0.0)     # [B, M, K]
            got = Ua[:, c * B * K:(c + 1) * B * K].float().view(M, B, K).permute(1, 0, 2)
            assert not torch.isnan(got).any(), c
            tol = 4e-3 if dtype is torch.bfloat16 else 1e-6
            assert torch.allclose(got, want, rtol = tol, atol = 1e-30), (c, float((got - want).abs().max()))
        assert bool((Ua[:, 0 * B * K + 1 * K:0 * B * K + 2 * K][M - 1] == 0).all())   # (copy 0's (last node, sample 1))
        assert am[0, M - 1, 1] == -float("inf")


@needs_virtual
@pytest.mark.parametrize("M, N, K, B, copies", [(3, 300, 2, 37, 2),     # children past one warp's share, ragged runs
                                                (6, 40, 16, 4, 3),      # whole 16-byte runs
                                                (5, 3, 3, 4, 70)])      # few children, copies past one launch's
def test_the_kept_operands_follow_their_formulas(M, N, K, B, copies):
    """`lowrank_kept_operands`, the pass's end from the exponentials the pass keeps -- `Ua = exp(A - am)`
    (`lowrank_a_operand`, am each (node, sample)'s largest A) and `Vt = exp(B - shB)` (`lowrank_z_operand`), bf16 --,
    checked directly: on every row (copy,
    b, r), `Ud[i] Vd[j] = f_b[i] exp(A[i, r, b] + B[r, j, b]) c_b[j] / S_b[i]` for the copies with a data half (their rows
    of Ud / Vd in their own order), `Un[i] Vt[j] = f_b[i] exp(A[i, r, b] + B[r, j, b]) / Z_b[i]` for those whose U is
    asked for, `c_b[j]` the product's two children's node values, `log S = node_mars + log Z`; Vd <= 1. A child no edge
    reaches, a child without mass, a node without flow, mass or normalizer, a rank without mass: 0, never NaN; the rows
    no copy writes untouched."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    mod, dev = lowrank_virtual._module(), torch.device("cuda")
    g = torch.Generator(device = "cpu").manual_seed(13)
    children = 13
    fa = [(torch.randn(M, K, B, generator = g) * 2).to(dev) for _ in range(copies)]
    fb = [(torch.randn(K, N, B, generator = g) * 2).to(dev) for _ in range(copies)]
    lz = [(torch.randn(M, B, generator = g) * 2).to(dev) for _ in range(copies)]
    rows = children + copies * M
    node_flows = (torch.randn(rows, B, generator = g) - 1.0).to(dev)
    node_mars = (torch.randn(rows, B, generator = g) - 3.0).to(dev)
    nid0 = [children + c * M for c in range(copies)]
    first, elements = 5, N + 7
    table = torch.randint(0, children, (elements, 2), generator = g, dtype = torch.int32).to(dev)
    eids = [(first + torch.randperm(elements, generator = g)[:N]).to(dev) for _ in range(copies)]
    eids[0][1] = -1                                                    # copy 0, child 1: no edge reaches it
    node_mars[int(table[int(eids[0][0]) - first, 0]), 2] = -float("inf")   # a child without mass (and whatever shares its row)
    node_flows[nid0[0] + 1, 2] = -float("inf")                         # copy 0, node 1, sample 2: no flow
    node_mars[nid0[1] + 0, 3 % B] = -float("inf")                      # copy 1, node 0: no mass
    node_mars[nid0[0] + M - 1, 3 % B] = -float("inf")                  # copy 0 (a data half), its last node: no mass
    lz[1][M - 1, 1] = -float("inf")                                    # copy 1, its last node, sample 1: no normalizer
    fb[0][1 % K, :, 1] = -float("inf")                                 # copy 0, a rank without mass at sample 1

    total = copies * B * K
    Ua = torch.empty(M, total, dtype = torch.bfloat16, device = dev)
    am = torch.empty(copies, M, B, device = dev)
    Vt = torch.empty(N, total, dtype = torch.bfloat16, device = dev)
    shb = torch.empty(total, device = dev)
    mod.lowrank_a_operand(Ua, am, fa, B, K)
    mod.lowrank_z_operand(Vt, shb, fb, B, K)
    data = [c for c in range(copies) if c != 1]                        # (copy 1: no data half)
    want_un = [c for c in range(copies) if c != 0]                     # (copy 0: no U asked for)
    drow0 = [data.index(c) * B * K if c in data else -1 for c in range(copies)]
    Ud = torch.full((M, len(data) * B * K), float("nan"), dtype = torch.bfloat16, device = dev)
    Vd = torch.full((N, len(data) * B * K), float("nan"), dtype = torch.bfloat16, device = dev)
    Un = torch.full((M, total), float("nan"), dtype = torch.bfloat16, device = dev)
    mod.lowrank_kept_operands(Ud, Vd, Un, Ua, am, Vt, shb, node_flows, node_mars, lz, nid0,
                              [eids[c] if c in data else None for c in range(copies)],
                              [table if c in data else None for c in range(copies)], [first] * copies,
                              [c * B * K for c in range(copies)], drow0, [int(c in want_un) for c in range(copies)], M, K,
                              [None] * copies)
    torch.cuda.synchronize()
    assert torch.isfinite(Ud.float()).all() and torch.isfinite(Vd.float()).all()
    # (the children's values kept instead of the table -- a copy whose product layer was not fused in: the same bits)
    Udk, Vdk, Unk = torch.full_like(Ud, float("nan")), torch.full_like(Vd, float("nan")), torch.full_like(Un, float("nan"))
    mod.lowrank_kept_operands(Udk, Vdk, Unk, Ua, am, Vt, shb, node_flows, node_mars, lz, nid0,
                              [eids[c] if c in data else None for c in range(copies)], [None] * copies, [0] * copies,
                              [c * B * K for c in range(copies)], drow0, [int(c in want_un) for c in range(copies)], M, K,
                              [_kept_children(node_mars, table, first, eids[c]) if c in data else None
                               for c in range(copies)])
    assert torch.equal(Udk, Ud) and torch.equal(Vdk, Vd) and torch.equal(Unk.isnan(), Un.isnan()) and \
        torch.equal(Unk.nan_to_num(), Un.nan_to_num()), "the kept children's values"
    assert float(Vd.float().max()) <= 1.0
    for c in range(copies):
        has = eids[c] >= 0
        at = (eids[c].clamp_min(first) - first).long()
        cv = node_mars[table[at, 0].long()] + node_mars[table[at, 1].long()]    # [N, B]
        cv = torch.where(has[:, None], cv, torch.full_like(cv, -float("inf"))).double()
        lf = node_flows[nid0[c]:nid0[c] + M].double()
        ls = node_mars[nid0[c]:nid0[c] + M].double() + lz[c].double()
        for r in range(K):
            row = c * B * K + torch.arange(B, device = dev) * K + r
            a, bb = fa[c][:, r].double(), fb[c][r].double()            # [M, B], [N, B]
            if c in want_un:
                want = torch.exp((lf + a - lz[c].double())[:, None] + bb[None])   # [M, N, B]
                dead = (torch.isneginf(lf) | torch.isinf(lz[c].double()))[:, None] | torch.isneginf(bb)[None]
                want = torch.where(dead, torch.zeros_like(want), want)
                got = Un[:, row].double()[:, None] * Vt[:, row].double()[None]
                assert torch.allclose(got, want, rtol = 3e-2, atol = 1e-30), f"Un: copy {c}, rank {r}"
            else:
                assert torch.isnan(Un[:, row].float()).all(), f"Un: copy {c}'s rows written"
            if c in data:
                d = data.index(c) * B * K + torch.arange(B, device = dev) * K + r
                want = torch.exp((lf + a - ls)[:, None] + (cv + bb)[None])
                dead = (torch.isneginf(lf) | torch.isneginf(ls))[:, None] | torch.isneginf(cv + bb)[None]
                want = torch.where(dead, torch.zeros_like(want), want)
                got = Ud[:, d].double()[:, None] * Vd[:, d].double()[None]
                assert torch.allclose(got, want, rtol = 3e-2, atol = 1e-30), f"Ud Vd: copy {c}, rank {r}"
    assert torch.isfinite(Un[:, B * K:].float()).all()


def _fp8_rows(q8, scales):
    """e4m3 operand rows back in float64: each row's codes times its scale."""
    return q8.float().double() * scales.double()[:, None]


@needs_virtual
@pytest.mark.parametrize("M, N, K, B, copies", [(3, 300, 2, 37, 2),     # children past one warp's share, ragged runs
                                                (6, 40, 16, 4, 3),      # whole runs (8-byte stores)
                                                (5, 3, 3, 4, 70)])      # few children, copies past one launch's
@pytest.mark.parametrize("from_g", [False, True])
def test_the_fp8_operands_are_the_kept_ones_in_e4m3(M, N, K, B, copies, from_g):
    """`lowrank_kept_operands_fp8` (`LowRank(flow_precision = "fp8")`) against `lowrank_kept_operands` on the same
    exponentials: each operand row -- Ud's, Un's (a node), Vd's, Vt's (a child) -- under a power-of-2 scale whose row's
    largest code lies in [224, 448] (1 for a row of zeros); every entry the bf16 one within e4m3's rounding (2^-4 of it,
    2^-10 of the scale below e4m3's normal range) and bf16's, 0 where it is 0; Vt's copy written for the copies whose U
    is asked for, the other columns untouched; copies past one launch (the rows' maxima over every launch); the
    children's values kept instead of the table the same bits. The edge cases of
    `test_the_kept_operands_follow_their_formulas`: finite, 0 where a factor of the term is. `from_g`: the rows' maxima
    from each (child, copy, sample)'s largest V over its ranks (what the in-place V operand's twin gives: here made from
    V) and the nodes' runs of exp(A - shift), not from a pass over the operands."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    mod, mod8, dev = lowrank_virtual._module(), lowrank_virtual._fp8_module(), torch.device("cuda")
    if mod8 is None:
        pytest.skip("the e4m3 pass's end did not build")
    f8 = torch.float8_e4m3fn
    g = torch.Generator(device = "cpu").manual_seed(13)
    children = 13
    fa = [(torch.randn(M, K, B, generator = g) * 2).to(dev) for _ in range(copies)]
    fb = [(torch.randn(K, N, B, generator = g) * 2).to(dev) for _ in range(copies)]
    lz = [(torch.randn(M, B, generator = g) * 2).to(dev) for _ in range(copies)]
    rows = children + copies * M
    node_flows = (torch.randn(rows, B, generator = g) - 1.0).to(dev)
    node_mars = (torch.randn(rows, B, generator = g) - 3.0).to(dev)
    nid0 = [children + c * M for c in range(copies)]
    first, elements = 5, N + 7
    table = torch.randint(0, children, (elements, 2), generator = g, dtype = torch.int32).to(dev)
    eids = [(first + torch.randperm(elements, generator = g)[:N]).to(dev) for _ in range(copies)]
    eids[0][1] = -1                                                    # copy 0, child 1: no edge reaches it
    node_mars[int(table[int(eids[0][0]) - first, 0]), 2] = -float("inf")   # a child without mass
    node_flows[nid0[0] + 1, 2] = -float("inf")                         # copy 0, node 1, sample 2: no flow
    node_mars[nid0[1] + 0, 3 % B] = -float("inf")                      # copy 1, node 0: no mass
    node_mars[nid0[0] + M - 1, 3 % B] = -float("inf")                  # copy 0 (a data half), its last node: no mass
    lz[1][M - 1, 1] = -float("inf")                                    # copy 1, its last node, sample 1: no normalizer
    fb[0][1 % K, :, 1] = -float("inf")                                 # copy 0, a rank without mass at sample 1
    for c in range(copies):                                            # node 2: no flow anywhere -- rows of zeros
        node_flows[nid0[c] + 2 % M] = -float("inf")

    total = copies * B * K
    Ua = torch.empty(M, total, dtype = torch.bfloat16, device = dev)
    am = torch.empty(copies, M, B, device = dev)
    Vt = torch.empty(N, total, dtype = torch.bfloat16, device = dev)
    shb = torch.empty(total, device = dev)
    mod.lowrank_a_operand(Ua, am, fa, B, K)
    mod.lowrank_z_operand(Vt, shb, fb, B, K)
    data = [c for c in range(copies) if c != 1]                        # (copy 1: no data half)
    want_un = [c for c in range(copies) if c != 0]                     # (copy 0: no U asked for)
    drow0 = [data.index(c) * B * K if c in data else -1 for c in range(copies)]

    def args(cv):
        return (Ua, am, Vt, shb, node_flows, node_mars, lz, nid0,
                [eids[c] if c in data else None for c in range(copies)],
                [table if c in data and cv is None else None for c in range(copies)], [first] * copies,
                [c * B * K for c in range(copies)], drow0, [int(c in want_un) for c in range(copies)], M, K,
                [None] * copies if cv is None else [cv[c] if c in data else None for c in range(copies)])

    Ud = torch.zeros(M, len(data) * B * K, dtype = torch.bfloat16, device = dev)
    Vd = torch.zeros(N, len(data) * B * K, dtype = torch.bfloat16, device = dev)
    Un = torch.zeros(M, total, dtype = torch.bfloat16, device = dev)
    mod.lowrank_kept_operands(Ud, Vd, Un, *args(None))
    # (each (child, copy, sample)'s largest V over its ranks)
    g = (Vt.float().view(N, total // K, K).amax(2).contiguous() if from_g else None,)

    def run(cv):
        q = dict(Ud = torch.full(Ud.shape, float("nan"), dtype = f8, device = dev),
                 Vd = torch.full(Vd.shape, float("nan"), dtype = f8, device = dev),
                 Un = torch.full(Un.shape, float("nan"), dtype = f8, device = dev),
                 Vt = torch.full(Vt.shape, float("nan"), dtype = f8, device = dev))
        sc = torch.full((2 * M + 2 * N,), float("nan"), device = dev)
        mod8.lowrank_kept_operands_fp8(q["Ud"], q["Vd"], q["Un"], q["Vt"], sc, *args(cv), *g)
        torch.cuda.synchronize()
        return q, sc

    q, sc = run(None)
    scales = dict(Ud = sc[:M], Un = sc[M:2 * M], Vd = sc[2 * M:2 * M + N], Vt = sc[2 * M + N:])
    assert torch.isfinite(sc).all() and torch.equal(torch.exp2(torch.log2(sc).round()), sc), "powers of 2"
    un_cols = torch.cat([torch.arange(c * B * K, (c + 1) * B * K, device = dev) for c in want_un])
    off_cols = torch.arange(0, B * K, device = dev)                    # (copy 0's: no U asked for)
    refs = dict(Ud = (Ud, slice(None)), Vd = (Vd, slice(None)), Un = (Un, un_cols), Vt = (Vt, un_cols))
    for name, (ref, cols) in refs.items():
        got, s, want = q[name][:, cols], scales[name], ref[:, cols].double()
        assert torch.isfinite(got.float()).all(), f"{name}: a written entry not finite"
        codes = got.float().abs().amax(1)
        live = want.abs().amax(1) > 0
        assert ((codes[live] >= 224) & (codes[live] <= 448)).all(), f"{name}: a row's largest code {codes[live]}"
        assert (s[~live] == 1).all() and (codes[~live] == 0).all(), f"{name}: a row of zeros"
        deq = _fp8_rows(got, s)
        assert bool((deq[want == 0] == 0).all()), f"{name}: an entry 0 in bf16 not 0"
        tol = want.abs() * (2 ** -4 + 2 ** -8) + s.double()[:, None] * 2 ** -10
        assert bool(((deq - want).abs() <= tol).all()), \
            f"{name}: off by {float(((deq - want).abs() / tol).max())} tolerances"
    assert not bool((scales["Ud"] == 1).all()) and not bool((scales["Vd"] == 1).all())
    assert torch.isnan(q["Un"][:, off_cols].float()).all() and torch.isnan(q["Vt"][:, off_cols].float()).all(), \
        "the columns of a copy whose U is not asked for written"
    # (the children's values kept instead of the table -- a copy whose product layer was not fused in: the same bits)
    qk, sck = run([_kept_children(node_mars, table, first, e) for e in eids])
    assert torch.equal(sck, sc), "the kept children's scales"
    for name in q:
        assert torch.equal(qk[name].view(torch.uint8), q[name].view(torch.uint8)), f"the kept children's {name}"


@needs_virtual
@pytest.mark.parametrize("L, bs, rank, batch, fp8", [(256, 32, 16, 8, True),     # the one-launch passes, e4m3 products
                                                     (160, 32, 2, 7, False)])    # rows not a multiple of 16: bf16
def test_the_fp8_flow_products_change_the_statistics_alone(monkeypatch, L, bs, rank, batch, fp8):
    """`LowRank(flow_precision = "fp8")` against the default on the same circuit and inputs: the log-likelihoods,
    node values, flows and gradients bit for bit, the parameter flows and the denominators within e4m3's rounding of
    the pass's end's products (an entry two roundings of each of its terms at most, 1e-5 of the largest below e4m3's
    range; their rel-l2 under 1e-2, not 0); the e4m3 products taken -- two a pass, F+ and W -- where they serve, the
    bf16 ones (the same values) where they do not."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.kernels import autotune

    if lowrank_virtual._fp8_module() is None:
        pytest.skip("the e4m3 pass's end did not build")
    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused"
                        if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk") else pick(key, cands, bench, **k))
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)
    calls, real = [], lowrank_virtual._product_fp8

    def counted(*a):
        calls.append(tuple(a[0].shape))
        return real(*a)

    monkeypatch.setattr(lowrank_virtual, "_product_fp8", counted)
    twins, real_twin = [], lowrank_virtual._fp8_module().lowrank_z_operand_in_place_g

    def counted_twin(*a):
        twins.append(a[0].shape)
        return real_twin(*a)

    monkeypatch.setattr(lowrank_virtual._fp8_module(), "lowrank_z_operand_in_place_g", counted_twin)
    out = {}
    for prec in ("bf16", "fp8"):
        twins.clear()
        pc, sums, _, _, _, _ = _pair(_hmm_blocks(L, bs), "sum_exp", rank = rank, flow_precision = prec)
        monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)
        dev = pc.params.device
        torch.manual_seed(37)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 470 + i) for i, a in enumerate(sums)}
        _passes(pc, sums, data, ext)
        calls.clear()
        out[prec] = _passes(pc, sums, data, ext)
        out[prec]["calls"], out[prec]["twins"] = list(calls), list(twins)
    got, want = out["fp8"], out["bf16"]
    assert want["calls"] == [] and want["twins"] == [], "the default took the e4m3 products or V's twin"
    assert len(got["calls"]) == (2 if fp8 else 0), got["calls"]
    # (rank 16, the factors read in place: V's twin makes the children's maxima -- no pass over the operands)
    assert bool(got["twins"]) == (rank in (4, 8, 16, 32)), got["twins"]
    for k in ("lls", "mars", "flows"):
        assert torch.equal(got[k], want[k]), k
    for k in ("dA", "dB"):
        assert all(torch.equal(a, b) for a, b in zip(got[k], want[k])), k
    for k in ("fp", "fm"):
        if not fp8:
            assert torch.allclose(got[k], want[k], rtol = 1e-6, atol = 0), k       # (atomic adds' order alone)
            continue
        # (a term's two e4m3 roundings at most 2^-4 each, bf16's 2^-8 under them; the sums -- 384 terms here --
        # average them out: their l2 a few 1e-3)
        top = float(want[k].abs().max())
        assert torch.allclose(got[k], want[k], rtol = 2 * 2 ** -4 + 2 ** -6, atol = 1e-5 * top), \
            f"{k}: {float(((got[k] - want[k]).abs() / (want[k].abs() + 1e-5 * top)).max())}"
        l2 = float((got[k] - want[k]).norm() / want[k].norm())
        assert 0 < l2 < 1e-2, f"{k}: rel-l2 {l2}"


def test_flow_precision_is_part_of_the_parameterization():
    """`flow_precision` is the "sum_exp" form's, in the parameterization's identity (signature, constructor, pickling);
    anything but "bf16" / "fp8" refused."""
    import pickle

    a, b = LowRank(rank = 4, form = "sum_exp"), LowRank(rank = 4, form = "sum_exp", flow_precision = "fp8")
    assert a.flow_precision == "bf16" and b.flow_precision == "fp8"
    assert a.get_signature() != b.get_signature() and b.get_signature().endswith("_f8")
    cls, kw = b._get_constructor()
    assert cls(**kw).get_signature() == b.get_signature()
    assert pickle.loads(pickle.dumps(b)).flow_precision == "fp8"
    with pytest.raises(AssertionError):
        LowRank(rank = 4, form = "exp_sum", flow_precision = "fp8")
    with pytest.raises(AssertionError):
        LowRank(rank = 4, form = "sum_exp", flow_precision = "fp16")


@needs_virtual
def test_fp8_flow_products_with_tied_copies_that_do_not_go_through_the_virtual_batch(monkeypatch):
    """`test_a_tied_copy_that_does_not_go_through_the_virtual_batch` at `flow_precision = "fp8"` (blocks of 16, 8
    samples at rank 2: every dimension of the e4m3 products a multiple of 16): the forward's normalizers hold rows for
    the three copies, one of them comes through the virtual batch's backward -- the e4m3 operands' columns of the other
    two 0, so W holds the one's terms alone --, on the plain path (its dB's normalizer half owed to the pass's end:
    the bf16 theta^T U beside the e4m3 W): against "bf16" on the same circuit, everything bit for bit but the
    parameter flows and the denominators, those within e4m3's rounding; W the e4m3 product's."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    if lowrank_virtual._fp8_module() is None:
        pytest.skip("the e4m3 pass's end did not build")
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)
    rank, batch, dev, K = 2, 8, torch.device("cuda:0"), 16
    calls, real = [], lowrank_virtual._product_fp8

    def counted(*a):
        calls.append(tuple(a[0].shape))
        return real(*a)

    monkeypatch.setattr(lowrank_virtual, "_product_fp8", counted)

    def build(prec):
        torch.manual_seed(0)
        with juice.set_block_size(K):
            leaf = lambda v: inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
            below = summate(multiply(leaf(0), leaf(1)), num_node_blocks = 1)
            s0 = summate(multiply(below, leaf(2)), num_node_blocks = 1,
                         parameterization = LowRank(rank = rank, form = "sum_exp", flow_precision = prec))
            shared = multiply(leaf(3), leaf(4))
            s1, s2 = s0.duplicate(shared, tie_params = True), s0.duplicate(shared, tie_params = True)
            mix = summate(multiply(s1), multiply(s2), num_node_blocks = 1)        # (the two over one scope: mixed)
            root = summate(multiply(s0, mix), num_node_blocks = 1, block_size = 1)
        torch.manual_seed(0)
        root.init_parameters(perturbation = 2.0)
        return juice.compile(root, verbose = False).to(dev), [s0, s1, s2]

    out = dict()
    for prec in ("bf16", "fp8"):
        pc, sums = build(prec)
        torch.manual_seed(41)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 260 + i) for i, a in enumerate(sums)}
        calls.clear()
        out[prec] = _passes(pc, sums, data, ext)
        out[prec]["calls"] = list(calls)
        kinds = [type(pc.external_params_nodes[a].__dict__["_lr_state"]) for a in sums]
        assert kinds == [tuple, dict, dict], kinds
    got, want = out["fp8"], out["bf16"]
    # (W's alone: the copy's parameter flows are its plain backward's, and its dB's normalizer half -- owed -- the
    # bf16 product's, as the default's)
    assert want["calls"] == [] and got["calls"] == [(16, 3 * batch * rank)], (want["calls"], got["calls"])
    for k in ("lls", "mars", "flows"):
        assert torch.equal(got[k], want[k]), k
    for k in ("dA", "dB"):
        assert all(torch.equal(a, b) for a, b in zip(got[k], want[k])), k
    for k in ("fp", "fm"):
        top = float(want[k].abs().max())
        assert torch.isfinite(got[k]).all(), k
        assert torch.allclose(got[k], want[k], rtol = 2 * 2 ** -4 + 2 ** -6, atol = 1e-5 * top), \
            f"{k}: {float(((got[k] - want[k]).abs() / (want[k].abs() + 1e-5 * top)).max())}"


@needs_virtual
@pytest.mark.parametrize("M, N, K, B, copies", [(40, 300, 16, 5, 3),      # a node's ranks over lanes, children ragged
                                                (1100, 1100, 16, 2, 2),   # past the kept registers' share (2nd loops)
                                                (8, 8, 4, 2, 70)])        # copies past one launch's
def test_the_fp8_in_place_operands_give_their_ranks_maxima(M, N, K, B, copies):
    """`lowrank_z_operand_in_place_g` (`LowRank(flow_precision = "fp8")`'s V): V and its shifts the bits of
    `lowrank_z_operand_in_place`; besides, each (child, copy, sample)'s largest V over its ranks, as stored. A rank
    without mass (its V 0), a sample without any (its maxima 0)."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    mod, mod8, dev = lowrank_virtual._module(), lowrank_virtual._fp8_module(), torch.device("cuda")
    if mod8 is None:
        pytest.skip("the e4m3 pass's end did not build")
    g = torch.Generator(device = "cpu").manual_seed(29)
    fa = [(torch.randn(B, M, K, generator = g) * 2).to(dev) for _ in range(copies)]
    fb = [(torch.randn(B, K, N, generator = g) * 2).to(dev) for _ in range(copies)]
    fb[0][B - 1, K - 1, :] = -float("inf")                              # a rank without mass
    fb[copies - 1][0] = -float("inf")                                   # a sample without any: its maxima 0
    total = copies * B * K
    out = {}
    for twin in (False, True):
        Vt = torch.full((N, total), float("nan"), dtype = torch.bfloat16, device = dev)
        shb = torch.full((total,), float("nan"), device = dev)
        if twin:
            gv = torch.full((N, total // K), float("nan"), device = dev)
            mod8.lowrank_z_operand_in_place_g(Vt, shb, gv, fb, B, K)
            out[twin] = (Vt, shb, gv)
        else:
            mod.lowrank_z_operand_in_place(Vt, shb, fb, B, K)
            out[twin] = (Vt, shb)
    torch.cuda.synchronize()
    assert torch.equal(out[False][0].view(torch.int16), out[True][0].view(torch.int16)), "V"
    assert torch.equal(out[False][1].view(torch.int32), out[True][1].view(torch.int32)), "V's shifts"
    Vt, shb, gv = out[True]
    assert torch.equal(gv, Vt.float().view(N, total // K, K).amax(2)), "V's maxima"
    assert bool((gv[:, (copies - 1) * B] == 0).all()), "a sample without mass: its maxima 0"


@needs_virtual
def test_a_backward_at_another_precision_than_its_forward(monkeypatch):
    """The forward's operands are the backward's where their dtype is what the backward's precision asks for. A backward at
    fp32 after a forward whose products took bf16 operands makes its own `V` and `theta` (from the parameters, not the
    forward's rounded one), and gives the torch passes' to fp32's rounding of what the forward left."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 2, 5
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank)
    dev = pc.params.device
    torch.manual_seed(37)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 250 + i) for i, a in enumerate(sums)}
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)

    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)
    made, real = [], lowrank_virtual._v_rows
    thetas, real_theta = [], LowRank._dense_params

    def counted(mod, fbs, N, K, batch, dtype, dev, *rest):
        made.append(dtype)
        return real(mod, fbs, N, K, batch, dtype, dev, *rest)

    def theta(geo, params, dtype = torch.float32):
        thetas.append((params.dtype, dtype, params is pc.params))
        return real_theta(geo, params, dtype)

    monkeypatch.setattr(lowrank_virtual, "_v_rows", counted)
    monkeypatch.setattr(LowRank, "_dense_params", staticmethod(theta))
    grads = {a: (torch.zeros_like(ext[a][0]), torch.zeros_like(ext[a][1])) for a in sums}
    pc(data, sum_external_params = ext)
    assert made == [torch.bfloat16] and thetas == [(torch.float32, torch.bfloat16, True)], thetas
    pc.precision = "fp32"                                       # (the circuit's, read by each pass as it runs)
    pc.init_param_flows(flows_memory = 0.0)
    pc.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)
    pc.precision = "auto"
    assert made == [torch.bfloat16, torch.float32], made
    assert thetas == [(torch.float32, torch.bfloat16, True), (torch.float32, torch.float32, True)], thetas
    # (the node values are the bf16 product's: the flows read them)
    assert torch.allclose(_denominators(pc, sums), want["fm"], rtol = 2e-2, atol = 1e-3)
    assert torch.allclose(pc.param_flows, want["fp"], rtol = 2e-2, atol = 1e-3)
    for k, a in enumerate(sums):
        assert torch.allclose(grads[a][1], want["dB"][k], rtol = 2e-2, atol = 1e-3)
    # ... and one of the default precision keeps the forward's
    pc(data, sum_external_params = ext)
    made.clear()
    thetas.clear()
    pc.init_param_flows(flows_memory = 0.0)
    pc.backward(data, allow_modify_flows = False, logspace_flows = True, sum_external_params_grad = grads)
    assert made == [] and thetas == [], (made, thetas)


@needs_virtual
def test_the_normalizers_theta_is_made_in_its_products_dtype(monkeypatch):
    """The forward's normalizer product and the backward's gradient read `theta` dense in their operands' dtype, made
    once a pass: at bf16 operands (`GEMM_BF16_WORK` set to 0) from the circuit's bf16 copy where the forward was
    handed it (forced here: its values as they are), by a copy that rounds the fp32 parameters else; at fp32 from the
    parameters, handed the copy or not. `F+` takes its factor from the parameters. Each pass's values, flows and
    gradients within its operands' rounding of the torch passes'. (A backward at another precision than its forward
    makes its own: `test_a_backward_at_another_precision_than_its_forward`.)"""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 2, 16
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank)
    dev = pc.params.device
    torch.manual_seed(39)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 350 + i) for i, a in enumerate(sums)}
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)

    made, real = [], LowRank._dense_params

    def dense(geo, params, dtype = torch.float32):
        made.append((params.dtype, dtype, params is pc.params))
        return real(geo, params, dtype)

    monkeypatch.setattr(LowRank, "_dense_params", staticmethod(dense))
    for force, source in ((True, torch.bfloat16), (False, torch.float32)):
        monkeypatch.setattr(pc, "_param_operands_force", force, raising = False)
        made.clear()
        got = _passes(pc, sums, data, ext)
        assert got["path"] == {tuple} and pc.__dict__["_param_operands_made"] is force
        assert made == [(source, torch.bfloat16, not force)], (force, made)     # (one source: the tied transition)
        _same(got, want, rtol = 2e-2, atol = 1e-3)
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 1 << 62)            # (fp32 operands)
    for force in (True, False):
        monkeypatch.setattr(pc, "_param_operands_force", force, raising = False)
        made.clear()
        got = _passes(pc, sums, data, ext)
        assert pc.__dict__["_param_operands_made"] is force
        assert made == [(torch.float32, torch.float32, True)], (force, made)
        # (handed the copy, the plain kernels may read it)
        _same(got, want, **(dict(rtol = 2e-2, atol = 1e-3) if force else dict(rtol = 2e-4, atol = 2e-5)))


@needs_virtual
def test_a_tied_copy_that_does_not_go_through_the_virtual_batch(monkeypatch):
    """Tied copies in a layer the virtual batch does not serve (two node vectors sharing a child) keep the torch passes, a
    copy of theirs in another layer goes through it: the forward's products hold rows for all of them, the rows of those
    that did not come through are no part of the backward's -- the torch passes' flows and gradients, all told."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch, dev, K = 2, 5, torch.device("cuda:0"), 8
    def build():
        torch.manual_seed(0)
        with juice.set_block_size(K):
            leaf = lambda v: inputs(v, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_CATS))
            below = summate(multiply(leaf(0), leaf(1)), num_node_blocks = 1)
            s0 = summate(multiply(below, leaf(2)), num_node_blocks = 1,
                         parameterization = LowRank(rank = rank, form = "sum_exp"))
            shared = multiply(leaf(3), leaf(4))
            s1, s2 = s0.duplicate(shared, tie_params = True), s0.duplicate(shared, tie_params = True)
            mix = summate(multiply(s1), multiply(s2), num_node_blocks = 1)        # (the two over one scope: mixed)
            root = summate(multiply(s0, mix), num_node_blocks = 1, block_size = 1)
        torch.manual_seed(0)
        root.init_parameters(perturbation = 2.0)
        return juice.compile(root, verbose = False).to(dev), [s0, s1, s2]

    out = dict()
    for arm in ("virtual", "torch"):
        monkeypatch.setattr(lowrank_virtual, "ENABLED", arm == "virtual")
        pc, sums = build()
        torch.manual_seed(41)
        data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
        ext = {a: _factors(a, rank, batch, dev, seed = 260 + i) for i, a in enumerate(sums)}
        out[arm] = got = _passes(pc, sums, data, ext)
        kinds = [type(pc.external_params_nodes[a].__dict__["_lr_state"]) for a in sums]
        assert kinds == ([tuple, dict, dict] if arm == "virtual" else [dict, dict, dict]), kinds
    _same(out["virtual"], out["torch"], rtol = 2e-4, atol = 2e-5)


@needs_virtual
def test_a_backward_that_wants_neither_the_denominator_nor_gradients_makes_no_normalizer(monkeypatch):
    """Without the denominator statistic (`apply_z_correction = False`) and without the factors' gradients
    (`compute_external_grads = False`) the normalizer half has nothing to write: the pass's end makes none of its
    operands, and the flows are the torch passes'. With the gradients (the default) it does, once a source."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 2, 5
    pc, sums, _, _, _, _ = _pair("hmm_tied", "sum_exp", rank = rank, apply_z_correction = False)
    dev = pc.params.device
    torch.manual_seed(43)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 270 + i) for i, a in enumerate(sums)}

    def flows(**kw):
        pc(data, sum_external_params = ext)
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(data, allow_modify_flows = False, logspace_flows = True, **kw)
        return pc.param_flows.clone(), pc.node_flows.clone()

    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    want = flows(compute_external_grads = False)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    norm = _normalizer_calls(monkeypatch)
    got = flows(compute_external_grads = False)
    assert {type(pc.external_params_nodes[a].__dict__["_lr_state"]) for a in sums} == {tuple}
    assert norm == [], norm
    for g, w in zip(got, want):
        assert torch.allclose(g, w, rtol = 2e-4, atol = 2e-5)

    flows()
    assert len(norm) == 1, norm


def test_refusals():
    with pytest.raises(AssertionError):
        LowRank(rank = 0)
    with pytest.raises(AssertionError):
        LowRank(rank = 2, form = "additive")
    assert LowRank(rank = 2).form == "exp_sum"
    assert LowRank(rank = 2, form = "sum_exp") != LowRank(rank = 2)
    assert LowRank(rank = 2).requests_denom_param_flows
    assert not LowRank(rank = 2, apply_z_correction = False).requests_denom_param_flows

    pc_a, sums_a, _, _, _, _ = _pair("shared_children", "exp_sum", rank = 2)
    dev = pc_a.params.device
    data = torch.randint(0, NUM_CATS, (2, pc_a.num_vars), device = dev)
    with pytest.raises(NotImplementedError, match = "Supply all"):
        pc_a(data, sum_external_params = {sums_a[0]: _factors(sums_a[0], 2, 2, dev)})


# ---------------------------------------------------------------- "exp_sum" through its fused kernels

def _fused_available():
    from pyjuice.parameterizations.lowrank.kernels.c import get_exp_sum_module
    return torch.cuda.is_available() and get_exp_sum_module() is not None


needs_fused = pytest.mark.skipif(not _fused_available(), reason = "the exp_sum form's CUDA extension is not built")


def _fused_calls(monkeypatch):
    """Every call of the fused kernels from now on, by name (the module wrapped where both passes look it up)."""
    import pyjuice.parameterizations.lowrank.forward as lr_forward
    import pyjuice.parameterizations.lowrank.backward as lr_backward
    real, calls = lr_forward.exp_sum_module(), []

    class Spy:
        def __getattr__(self, name):
            fn = getattr(real, name)
            return lambda *a, **k: calls.append(name) or fn(*a, **k)

    spy = Spy()
    monkeypatch.setattr(lr_forward, "exp_sum_module", lambda: spy)
    monkeypatch.setattr(lr_backward, "exp_sum_module", lambda: spy)
    return calls


@needs_fused
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("name", sorted(STRUCTURES))
@pytest.mark.parametrize("rank,batch", [(1, 3), (3, 5), (4, 1), (8, 33), (16, 2), (24, 3), (64, 2)])
def test_the_fused_exp_sum_is_the_torch_pass(name, rank, batch, precision, monkeypatch):
    """`exp_sum` through its fused kernels against its torch passes: the same values, flows, denominators and
    gradients (`FUSED_TOL`: bf16 products at the default precision, three-term TF32 at `"fp32"`), on every structure
    -- the kernels taken for the node vectors whose parameters lie block-major in one run (`LowRank._geometry`'s grid),
    the torch passes kept for the others."""
    import pyjuice.parameterizations.lowrank.forward as lr_forward

    pc, sums, _, _, _, _ = _pair(name, "exp_sum", rank = rank, precision = precision)
    dev = pc.params.device
    torch.manual_seed(9)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 50 + i) for i, a in enumerate(sums)}

    calls = _fused_calls(monkeypatch)
    got = _passes(pc, sums, data, ext)
    gridded = 0                                            # (the node vectors whose parameters lie block-major)
    for a in sums:
        layer = pc.external_params_nodes[a]
        desc = _source(a).parameterization
        gridded += desc._geometry(layer, desc._info(layer, a), dev)["grid"] is not None
    assert calls.count("exp_sum_forward") == gridded and calls.count("exp_sum_backward") == gridded, (calls, gridded)
    monkeypatch.setattr(lr_forward, "EXP_SUM_FUSED", False)
    del calls[:]
    want = _passes(pc, sums, data, ext)
    assert calls == []
    _same(got, want, **FUSED_TOL[precision])

    # Without parameter flows: the element flows and the gradients alone, nothing accumulated -- the gradients those
    # of the run with them, to the bit (the same kernels' products and sums)
    monkeypatch.setattr(lr_forward, "EXP_SUM_FUSED", True)
    again = _passes(pc, sums, data, ext, param_flows = False)
    assert float(again["fp"].abs().sum()) == 0.0 and float(pc.denom_param_flows.abs().sum()) == 0.0
    assert torch.allclose(again["flows"].exp(), want["flows"].exp(), rtol = 2e-4, atol = 2e-5)
    for k in ("dA", "dB"):
        for a, b in zip(again[k], got[k]):
            assert torch.allclose(a, b, rtol = 1e-5, atol = 1e-7)


def _hmm_blocks(L, bs, T = 4):
    """A tied HMM of `L` latents in blocks of `bs`, its transitions LowRank exp_sum's: the fused kernels' tiles (64
    nodes by 32 children; the bf16 backward's from rank 9 on 128 by 64) many, ragged where `L` is not a multiple of
    them."""
    def build(external, rank = 2, form = "exp_sum", **kw):
        with juice.set_block_size(bs):
            ni = inputs(T - 1, num_node_blocks = L // bs, dist = dists.Categorical(num_cats = NUM_CATS))
            sums, cur = [], ni
            for var in range(T - 2, -1, -1):
                cx = ni.duplicate(var, tie_params = True)
                ns = sums[0].duplicate(cur, tie_params = True) if sums and external else \
                    summate(cur, num_node_blocks = L // bs, **_ext(external, rank, form, **kw))
                sums.append(ns)
                cur = multiply(cx, ns)
            root = summate(cur, num_node_blocks = 1, block_size = 1)
        return root, sums
    return build


@needs_fused
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("L,bs,rank,batch,zsplit", [(160, 32, 3, 7, 0), (160, 32, 3, 7, 1), (300, 4, 16, 5, 0),
                                                    (256, 64, 8, 33, 2), (72, 8, 4, 2, 0), (160, 32, 40, 3, 0),
                                                    (256, 64, 33, 4, 1), (300, 4, 17, 2, 0), (300, 4, 16, 5, 3),
                                                    (160, 32, 24, 7, 2), (160, 32, 17, 1, 2)])
def test_the_fused_exp_sum_on_many_tiles(L, bs, rank, batch, zsplit, precision, monkeypatch):
    """The fused kernels where a node vector spans many of their tiles -- nodes past 64 and children past 32, ragged
    where `L` is not a multiple of either, the forward's column tiles combined, the backward's partial sums over tiles
    added, its samples split over CTAs (`EXP_SUM_ZSPLIT`: 0 the default, 1 none -- F+ and W added by the one CTA
    holding them --, 2 two), ranks past 16 (40, 33, 17: padded to 32 and 64) -- against the torch passes, at both
    precisions. The bf16 backward's big tiles (ranks past 8) take `EXP_SUM_ZSPLIT` as their persistent CTAs' count at
    most: 1 to 3 put all the units on as many runs, each crossing tiles (a tile's first stage and A fragments fetched in
    the last sample of the tile before, F+ and W added where a run leaves a tile, a tile's samples split between runs;
    at batch 1 every unit a tile of its own)."""
    import pyjuice.parameterizations.lowrank.forward as lr_forward

    monkeypatch.setattr(lr_forward, "EXP_SUM_ZSPLIT", zsplit)
    pc, sums, _, _, _, _ = _pair(_hmm_blocks(L, bs), "exp_sum", rank = rank, precision = precision)
    dev = pc.params.device
    torch.manual_seed(11)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, spread = 0.6, seed = 80 + i) for i, a in enumerate(sums)}
    calls = _fused_calls(monkeypatch)
    got = _passes(pc, sums, data, ext)
    assert calls.count("exp_sum_forward") == len(sums) and calls.count("exp_sum_backward") == len(sums), calls
    monkeypatch.setattr(lr_forward, "EXP_SUM_FUSED", False)
    want = _passes(pc, sums, data, ext)
    _same(got, want, **FUSED_TOL[precision])


@needs_fused
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("rank", [4, 24])
def test_the_fused_exp_sum_reads_bf16_parameters(rank, precision, monkeypatch):
    """A circuit whose parameters are bf16 (`param_dtype`): the fused kernels read them as the torch passes do (the
    same values, widened) -- values, flows, denominators and gradients those of the torch passes."""
    import pyjuice.parameterizations.lowrank.forward as lr_forward

    torch.manual_seed(0)
    root, sums = _hmm_blocks(160, 32)(external = True, rank = rank)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = torch.bfloat16, precision = precision).to(torch.device("cuda:0"))
    assert pc.params.dtype == torch.bfloat16
    dev = pc.params.device
    data = torch.randint(0, NUM_CATS, (6, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, 6, dev, spread = 0.5, seed = 90 + i) for i, a in enumerate(sums)}
    calls = _fused_calls(monkeypatch)
    got = _passes(pc, sums, data, ext)
    assert calls.count("exp_sum_forward") == len(sums) and calls.count("exp_sum_backward") == len(sums), calls
    monkeypatch.setattr(lr_forward, "EXP_SUM_FUSED", False)
    want = _passes(pc, sums, data, ext)
    _same(got, want, **FUSED_TOL[precision])


@needs_fused
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("rank,batch", [(4, 3), (16, 4), (64, 2), (16, 100), (64, 99)])
def test_the_fused_exp_sum_forward_far_below_its_shift(rank, batch, precision):
    """The forward's S terms are Z's times the children's factors 2^(lc - mc), mc a tile's largest child value: where a
    tile's large weights and large child values sit on different children -- rank 0 puts every other child's scale at
    e^-95 (2^-137 of the node's largest term: flushed), every other child's value is 100 nats down (its factor
    2^-144: flushed), so the factorized sum is 0 though S is not -- the tile is made again exactly under its own max (a
    warp's vote). The node values and log Z against float64; the parameters all fp32 normal (the kernels flush
    subnormals)."""
    from pyjuice.parameterizations.lowrank.kernels.c import get_exp_sum_module
    mod, dev = get_exp_sum_module(), torch.device("cuda:0")
    M = N = 128
    bs = cbs = 32
    g = torch.Generator(device = "cpu").manual_seed(3)
    A = 0.3 * torch.randn(batch, M, rank, generator = g)
    Bm = 0.3 * torch.randn(batch, rank, N, generator = g)
    A[:, :, 0] = 1.0                                       # (phi[i, j] = B[0, j] + the other ranks' small terms)
    Bm[:, 0, 0::2] = -95.0
    Bm[:, 0, 1::2] = 0.0
    theta = torch.rand(M, N, generator = g).add_(0.05)
    theta = theta / theta.sum(1, keepdim = True)
    emars = -5.0 + torch.randn(N, batch, generator = g)
    emars[1::2] -= 100.0
    params = theta.view(M // bs, bs, N // cbs, cbs).permute(0, 2, 3, 1).contiguous().view(-1).to(dev)
    nmars = torch.empty(M, batch, device = dev)
    lzs = mod.exp_sum_forward(A.to(dev).contiguous(), Bm.to(dev).contiguous(), params, 0, N // cbs, cbs, bs,
                              emars.to(dev).contiguous(), torch.arange(N, device = dev), nmars, 0, M, N, rank,
                              precision != "fp32")
    lw = torch.bmm(A.double(), Bm.double()) + theta.double().log()[None]
    lz = torch.logsumexp(lw, 2)
    ln = torch.logsumexp(lw + emars.t().double()[:, None, :], 2) - lz
    worst = float((nmars.t().double().cpu() - ln).abs().max())
    assert worst < 1e-3, worst
    assert float((lzs.t().double().cpu() - lz).abs().max()) < 1e-3


@needs_fused
@pytest.mark.parametrize("rank,batch,nslice", [(64, 160, 4), (16, 300, 3)])
def test_a_batch_past_the_l2_runs_the_bf16_passes_in_chunks(rank, batch, nslice):
    """A batch whose working set exceeds the L2 runs the bf16 passes a chunk of samples at a time (`sample_chunk`: at
    1024 x 1024 nodes, rank 64 is ~2 MB a sample forward and 1.5 MB backward -- chunks of 54 and 80 at a 128 MB L2 --,
    rank 16 forward ~0.75 MB -- chunks of 100): every output against the same passes over the batch's slices, a call a
    slice -- the forward's node values and log Z to the bit (a sample's work is the same in any chunk), the backward's
    element flows, F+, W and gradients to fp32 (their sums over tiles and samples are atomic adds, in any order)."""
    from pyjuice.parameterizations.lowrank.kernels.c import get_exp_sum_module
    mod, dev = get_exp_sum_module(), torch.device("cuda:0")
    M = N = 1024
    bs = cbs = 32
    g = torch.Generator(device = "cpu").manual_seed(7)
    A = (0.3 * torch.randn(batch, M, rank, generator = g)).to(dev)
    Bm = (0.3 * torch.randn(batch, rank, N, generator = g)).to(dev)
    theta = torch.rand(M, N, generator = g).add_(0.05)
    theta = theta / theta.sum(1, keepdim = True)
    params = theta.view(M // bs, bs, N // cbs, cbs).permute(0, 2, 3, 1).contiguous().view(-1).to(dev)
    emars = (-5.0 + torch.randn(N, batch, generator = g)).to(dev)
    nflows = (torch.randn(M, batch, generator = g) - 7.0).to(dev)
    rows = torch.arange(N, device = dev)

    def passes(lo, hi):
        b = hi - lo
        nmars = torch.empty(M, b, device = dev)
        lzs = mod.exp_sum_forward(A[lo:hi].contiguous(), Bm[lo:hi].contiguous(), params, 0, N // cbs, cbs, bs,
                                  emars[:, lo:hi].contiguous(), rows, nmars, 0, M, N, rank, True)
        eflows = torch.full((N, b), -float("inf"), device = dev)
        pf, dn = torch.zeros(M * N, device = dev), torch.zeros(M * N, device = dev)
        gA, gB = torch.zeros(b, M, rank, device = dev), torch.zeros(b, rank, N, device = dev)
        mod.exp_sum_backward(A[lo:hi].contiguous(), Bm[lo:hi].contiguous(), params, 0, N // cbs, cbs, bs,
                             emars[:, lo:hi].contiguous(), rows, nflows[:, lo:hi].contiguous(), nmars, 0, lzs, eflows,
                             pf, 0, dn, 0, gA, gB, M, N, rank, 0, True)
        return nmars, lzs, eflows, pf, dn, gA, gB

    whole = passes(0, batch)
    cuts = [batch * i // nslice for i in range(nslice + 1)]
    parts = [passes(lo, hi) for lo, hi in zip(cuts[:-1], cuts[1:])]
    for k in (0, 1):                                       # node values, log Z: bit for bit
        assert torch.equal(whole[k], torch.cat([p[k] for p in parts], 1)), k
    ef = torch.cat([p[2] for p in parts], 1)
    assert torch.isfinite(whole[2]).all()
    assert float((whole[2] - ef).abs().max()) < 1e-5     # (log element flows)
    for k in (3, 4):                                       # F+, W: summed over the slices
        want = sum(p[k] for p in parts)
        assert float((whole[k] - want).abs().max()) <= 1e-5 * float(want.abs().max()), k
    for k in (5, 6):                                       # dA, dB
        want = torch.cat([p[k] for p in parts], 0)
        assert float((whole[k] - want).abs().max()) <= 1e-5 * float(want.abs().max()), k


@needs_fused
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("side", ["low", "high"])
@pytest.mark.parametrize("rank,batch", [(16, 4), (64, 100)])
def test_the_fused_exp_sum_forward_out_of_its_frame(rank, batch, side, precision):
    """The bf16 forward sums a row's terms 2^(lw' - cm) unshifted, cm the row's largest log2 theta in the tile (once a
    tile), where its sums z and s lie in [2^-92, 2^100]: rank 0 puts every child's scale of half the nodes at e^-95
    (their terms flushed: z = 0, "low") or at e^80 (z past 2^100, "high"), so those rows take the shifted path, under
    the row max of the sample. Batch 100 gives a CTA samples in pairs (its second sample's registers, the
    pair's fallbacks). The node values and log Z against float64."""
    from pyjuice.parameterizations.lowrank.kernels.c import get_exp_sum_module
    mod, dev = get_exp_sum_module(), torch.device("cuda:0")
    M = N = 128
    bs = cbs = 32
    g = torch.Generator(device = "cpu").manual_seed(5)
    A = 0.3 * torch.randn(batch, M, rank, generator = g)
    Bm = 0.3 * torch.randn(batch, rank, N, generator = g)
    A[:, :, 0] = 0.0
    A[:, 0::2, 0] = 1.0                                    # (phi[i, j] = B[0, j] + small for the even nodes)
    Bm[:, 0, :] = -95.0 if side == "low" else 80.0
    theta = torch.rand(M, N, generator = g).add_(0.05)
    theta = theta / theta.sum(1, keepdim = True)
    emars = -5.0 + torch.randn(N, batch, generator = g)
    params = theta.view(M // bs, bs, N // cbs, cbs).permute(0, 2, 3, 1).contiguous().view(-1).to(dev)
    nmars = torch.empty(M, batch, device = dev)
    lzs = mod.exp_sum_forward(A.to(dev).contiguous(), Bm.to(dev).contiguous(), params, 0, N // cbs, cbs, bs,
                              emars.to(dev).contiguous(), torch.arange(N, device = dev), nmars, 0, M, N, rank,
                              precision != "fp32")
    lw = torch.bmm(A.double(), Bm.double()) + theta.double().log()[None]
    lz = torch.logsumexp(lw, 2)
    ln = torch.logsumexp(lw + emars.t().double()[:, None, :], 2) - lz
    assert torch.isfinite(nmars).all() and torch.isfinite(lzs).all()
    assert float((nmars.t().double().cpu() - ln).abs().max()) < 1e-3
    assert float(((lzs.t().double().cpu() - lz) / lz.abs().clamp_min(1.0)).abs().max()) < 1e-5


@needs_fused
@pytest.mark.parametrize("precision", ["auto", "fp32"])
@pytest.mark.parametrize("rank,batch", [(16, 4), (64, 100)])
def test_the_fused_exp_sum_forward_with_mostly_zero_parameters(rank, batch, precision):
    """97% of the parameters 0 (their terms -inf), every node keeping its first child's: a row's four lanes in a tile
    see few or none of its live terms -- the row's frame (its largest log2 theta in the tile) and liveness are taken
    over all four, so a lane with no live term still stores its row's sums in the frame they were made in. The node
    values and log Z against float64."""
    from pyjuice.parameterizations.lowrank.kernels.c import get_exp_sum_module
    mod, dev = get_exp_sum_module(), torch.device("cuda:0")
    M = N = 128
    bs = cbs = 32
    g = torch.Generator(device = "cpu").manual_seed(9)
    A = 0.3 * torch.randn(batch, M, rank, generator = g)
    Bm = 0.3 * torch.randn(batch, rank, N, generator = g)
    theta = torch.rand(M, N, generator = g).add_(0.05)
    theta[torch.rand(M, N, generator = g) < 0.97] = 0.0
    theta[:, 0] += 0.01
    emars = -5.0 + torch.randn(N, batch, generator = g)
    params = theta.view(M // bs, bs, N // cbs, cbs).permute(0, 2, 3, 1).contiguous().view(-1).to(dev)
    nmars = torch.empty(M, batch, device = dev)
    lzs = mod.exp_sum_forward(A.to(dev).contiguous(), Bm.to(dev).contiguous(), params, 0, N // cbs, cbs, bs,
                              emars.to(dev).contiguous(), torch.arange(N, device = dev), nmars, 0, M, N, rank,
                              precision != "fp32")
    lw = torch.bmm(A.double(), Bm.double()) + theta.double().log()[None]
    lz = torch.logsumexp(lw, 2)
    ln = torch.logsumexp(lw + emars.t().double()[:, None, :], 2) - lz
    assert torch.isfinite(nmars).all() and torch.isfinite(lzs).all()
    assert float((nmars.t().double().cpu() - ln).abs().max()) < 1e-3
    assert float((lzs.t().double().cpu() - lz).abs().max()) < 1e-3


@needs_virtual
@pytest.mark.parametrize("batch", [16, 12])
def test_the_one_launch_passes_where_a_stages_values_lie_deep_and_apart(batch, monkeypatch):
    """The one-launch forward and backward (`lr_om_kernel`, `lr_om_bw_kernel`) where a stage's values lie deep below 0
    and far apart: a chain of 64 steps (the values ~100 nats down), in every node block of 32 half the states emitting
    half the categories with probability 0 and the other half the others (an HMM's sparse emissions: half a k-step's
    children impossible whatever the token), and the factors matching each half of a block's nodes to the same half
    of the child blocks' (a node's other children ~60 nats down: half the parents' values ~60 below the others'). A
    stage's maxima, as the converters take them (an integer redux of the floats' order-preserving keys), must be the
    largest (a shift of 0 loses every term); the normalizer half's running max rises where the data half's does not
    (a stage's flags take either). Against the plain path and the torch passes on the same circuit; the A operand the
    forward's collapse's (no launch of its own)."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.kernels import autotune

    def sparse_emissions(root):
        for ns in root:
            if ns.is_input() and not ns.is_tied():
                p = ns.get_params().reshape(ns.num_nodes, NUM_CATS).clone()
                low = (torch.arange(ns.num_nodes, device = p.device) % 32) < 16
                p[low, NUM_CATS // 2:] = 0.0
                p[~low, :NUM_CATS // 2] = 0.0
                ns.set_params(p / p.sum(dim = 1, keepdim = True))

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused"
                        if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk") else pick(key, cands, bench, **k))
    rank = 16
    pc, sums, _, _, _, _ = _pair(_hmm_blocks(256, 32, T = 64), "sum_exp", rank = rank, prepare = sparse_emissions)
    monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)   # (the bf16 copy: the stream path's)
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)                # (bf16 normalizers, as at real sizes)
    dev = pc.params.device
    torch.manual_seed(43)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 490 + i) for i, a in enumerate(sums)}
    for a in sums:                                      # (ranks 0..7 a block's first half's, 8..15 its second's)
        lo_i = (torch.arange(a.num_nodes, device = dev) % 32) < 16
        lo_j = (torch.arange(a.num_ch_nodes, device = dev) % 32) < 16
        lo_r = torch.arange(rank, device = dev) < rank // 2
        ext[a][0][:] += torch.where(lo_i[:, None] == lo_r[None, :], 30.0, -30.0)
        ext[a][1][:] += torch.where(lo_r[:, None] == lo_j[None, :], 30.0, -30.0)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", False)
    torch_pass = _passes(pc, sums, data, ext)
    monkeypatch.setattr(lowrank_virtual, "ENABLED", True)
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM", False)
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM_BK", False)
    plain = _passes(pc, sums, data, ext)
    smod = lowrank_virtual._stream_module()
    calls, real = [], smod.lowrank_backward_om

    def counted(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(smod, "lowrank_backward_om", counted)
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM", True)
    monkeypatch.setattr(lowrank_virtual, "FUSED_STREAM_BK", True)
    taken = _stream_form(monkeypatch, "om")
    # (every copy's forward the one-launch form, the factors read in place: the A operand its collapse's -- the
    # backward and the pass's end launch none of their own)
    mod, ops = lowrank_virtual._module(), []
    real_a = mod.lowrank_a_operand_in_place

    def counted_a(*args):
        ops.append(1)
        return real_a(*args)

    monkeypatch.setattr(mod, "lowrank_a_operand_in_place", counted_a)
    got = _passes(pc, sums, data, ext)
    assert len(taken) >= len(sums) and all(c == lowrank_virtual._OM_ARM for c, _ in taken)
    assert len(calls) == len(sums), calls
    assert not ops, ops
    assert torch.isneginf(plain["mars"]).any() and torch.isfinite(plain["lls"]).all()   # (impossible nodes, possible data)
    assert float(plain["lls"].max()) < -88.0, plain["lls"]                    # (every value past a shift of 0's range)
    for k in ("lls", "mars", "flows", "fp", "fm", "dA", "dB"):
        for t in (got[k] if isinstance(got[k], list) else [got[k]]):
            assert not torch.isnan(t).any(), k
    _same(got, plain, rtol = 2e-3, atol = 2e-3, grad_scale = 2e-2)
    _same(got, torch_pass, rtol = 2e-2, atol = 1e-3, grad_scale = 2e-2)


@needs_virtual
@pytest.mark.parametrize("bs,batch,group,da_ctas,fw_ctas", [(32, 16, 1, 0, 0), (32, 16, 3, 0, 1), (64, 12, 2, 1, 3),
                                                             (128, 16, 3, 3, 5)])
def test_the_one_launch_grids_order_changes_no_bit(bs, batch, group, da_ctas, fw_ctas, monkeypatch):
    """The one-launch kernels' grids (`kernels/c/lowrank_stream.cu`): the tiles in groups of sample tiles (`tile_of`: a
    quarter of the SMs a group -- at these sizes one group, the grid's plain order), the backward's dA CTAs first in
    the grid, the launcher's share of the SMs, and the forward PERSISTENT, a CTA an SM taking every grid-th tile, its
    ring running on across them. A tile's sums and a pair's dA are its own whatever block of the grid takes them: the
    passes give the same BITS in groups of 1, 2 or 3 sample tiles of the 4 or 3 (a last group of fewer), with the dA
    CTAs capped (1: one CTA strides over every tile of the pairs; 3: unevenly) and the forward's CTAs capped (1: one
    CTA takes a layer's 8 or 6 tiles, its stages' phases flipping across them; 3, 5: unevenly -- at these sizes a CTA
    has one tile otherwise)."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual
    from pyjuice.layer.kernels import autotune

    pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, **k: "fused"
                        if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk") else pick(key, cands, bench, **k))
    rank = 16
    pc, sums, _, _, _, _ = _pair(_hmm_blocks(256, bs), "sum_exp", rank = rank)
    monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)   # (the bf16 copy: the stream path's)
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)                # (bf16 normalizers, as at real sizes)
    dev = pc.params.device
    torch.manual_seed(47)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 530 + i) for i, a in enumerate(sums)}
    smod = lowrank_virtual._stream_module()
    calls, real = [], smod.lowrank_backward_om

    def counted(*args):
        calls.append(args[22] is not None)                              # (whether it took dA: `grad_a`)
        return real(*args)

    monkeypatch.setattr(smod, "lowrank_backward_om", counted)
    taken = _stream_form(monkeypatch, "om")
    want = _passes(pc, sums, data, ext)
    assert len(taken) >= len(sums) and all(c == lowrank_virtual._OM_ARM for c, _ in taken)
    assert len(calls) == len(sums) and all(calls), calls
    smod.set_tile_group(group)
    smod.set_da_ctas_most(da_ctas)
    smod.set_om_ctas_most(fw_ctas)
    try:
        got = _passes(pc, sums, data, ext)
    finally:
        smod.set_tile_group(0)
        smod.set_da_ctas_most(0)
        smod.set_om_ctas_most(0)
    # (the hooks took: the last launches' grids -- every layer's the same shape -- the forward's CTAs, the dA CTAs and
    # both kernels' groups as set)
    fw_n, fw_group, da_n, bw_group = smod.om_last_grids()
    assert fw_group == group and bw_group == group and (fw_n == fw_ctas if fw_ctas else fw_n >= 6), (fw_n, fw_group)
    assert da_n == da_ctas if da_ctas else da_n >= 1, da_n
    assert len(calls) == 2 * len(sums) and all(calls), calls
    for k in ("lls", "mars", "flows", "fp", "fm"):
        assert not torch.isnan(got[k]).any() and torch.equal(got[k], want[k]), k
    for k in ("dA", "dB"):
        assert all(not torch.isnan(a).any() and torch.equal(a, b) for a, b in zip(got[k], want[k])), k


@needs_virtual
@pytest.mark.parametrize("rounds", [2, 4])
def test_the_pass_ends_products_in_rounds_are_torch_mms(rounds, monkeypatch):
    """The pass's end's two products (`virtual._product_nt`: F+ = Ud Vd^T, W = Un Vt^T, bf16) in rounds of a batch-8
    torch.bmm over 8 R chunks of K, each round's 8 summed in fp32 (`PRODUCT_ROUNDS` forced: the tuner takes them where
    they are faster, at large batches): the parameter flows and their denominators those of torch.mm within fp32's
    rounding of the sums in another order."""
    import pyjuice.parameterizations.lowrank.virtual as lowrank_virtual

    rank, batch = 16, 16
    pc, sums, _, _, _, _ = _pair(_hmm_blocks(256, 32), "sum_exp", rank = rank)
    monkeypatch.setattr(pc, "_param_operands_force", True, raising = False)   # (the bf16 copy: the kept operands)
    monkeypatch.setattr(lowrank_virtual, "GEMM_BF16_WORK", 0)                # (bf16 products)
    dev = pc.params.device
    torch.manual_seed(53)
    data = torch.randint(0, NUM_CATS, (batch, pc.num_vars), device = dev)
    ext = {a: _factors(a, rank, batch, dev, seed = 560 + i) for i, a in enumerate(sums)}
    calls, real = [], lowrank_virtual._product_nt

    def counted(a, b):
        calls.append(a.size(1))
        return real(a, b)

    monkeypatch.setattr(lowrank_virtual, "_product_nt", counted)
    monkeypatch.setattr(lowrank_virtual, "PRODUCT_ROUNDS", 0)
    want = _passes(pc, sums, data, ext)
    took = len(calls)
    monkeypatch.setattr(lowrank_virtual, "PRODUCT_ROUNDS", rounds)
    got = _passes(pc, sums, data, ext)
    # (F+ and W, the rounds' chunks whole: K a multiple of 8 R)
    assert took >= 1 and len(calls) == 2 * took and all(k % (8 * rounds) == 0 for k in calls), calls
    for k in ("fp", "fm"):
        top = float(want[k].abs().max())
        assert top > 0 and torch.allclose(got[k], want[k], rtol = 1e-5, atol = 1e-6 * top), \
            f"{k}: {float((got[k] - want[k]).abs().max())} against {top}"
    for k in ("lls", "mars", "flows"):
        assert torch.equal(got[k], want[k]), k
