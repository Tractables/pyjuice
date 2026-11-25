"""
`LowRank(top_k)`: the per-candidate forms -- the tilted soft-evidence leaf and the sum node whose parameters read the
token of the leaf beside it -- against their DEFINITION, a dynamic program over every candidate in float64 (here, by
autograd for its gradients): nothing of the descriptor's passes is shared with the reference. The leaf at position t,
state z, candidate c (slot s(c) = min(place, K)) and the transition out of it given the candidate it emits:

    log P(c | z)       = log_softmax_c( log E[z, ids_c] + lp_c + U[z] . D[:, s(c)] )
    log P(z' | z, c)   = log_softmax_z'( log theta[z, z'] + A[z] . B[:, z'] + G[s(c), z'] )

a missing token summed over all its candidates. Checked: the values, the gradients in every external tensor and in
the evidence, the EM statistics (F+ - F- = d / d log-parameters, the tables' rows normalized), the differentiable
conditional, a tree whose product holds a reading sum beside a plain one, and the structures the compiler refuses.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.parameterizations import LowRank


DEV = torch.device("cuda")
V = 12                                                   # (the vocabulary; every position has C candidates of it)


# ---------------------------------------------------------------------------------------------- the reference

def _leaf_logp(lE, lp, ids, U, D, K):
    """[b, N, C]: log P(c | z) of one position (lE [N, V] log E; lp, ids [b, C]; U [b, N, re], D [b, re, K + 1])."""
    C = ids.size(1)
    slot = torch.arange(C, device = ids.device).clamp(max = K)
    logits = lE[:, ids].permute(1, 0, 2) + lp[:, None, :] + torch.bmm(U, D)[:, :, slot]
    return torch.log_softmax(logits, dim = 2)


def _trans_logp(lT, A, B, G):
    """[b, N, S, N]: log P(z' | z, slot s) of one transition (lT [N, N]; A [b, N, r], B [b, r, N], G [b, S, N])."""
    lt = lT[None, :, None, :] + torch.bmm(A, B)[:, :, None, :] + G[:, None, :, :]
    return torch.log_softmax(lt, dim = 3)


def _emit_terms(logP, ids, x, miss):
    """[b, N, C]: log P(c | z) times the evidence's indicator (every candidate where the token is missing)."""
    hit = (ids == x[:, None]) | miss[:, None]
    return logP + torch.where(hit, 0.0, -float("inf")).to(logP.dtype)[:, None, :]


def _ref_hmm(tab, ext, ids, lp, x, miss, K):
    """[b] log p(the observed tokens) of the HMM, float64: the leaf's candidates one by one at every position."""
    lT, lpi, lE = tab["lT"], tab["lpi"], tab["lE"]
    A, B, G, U, D = ext
    T = ids.size(1)
    slot = torch.arange(ids.size(2), device = ids.device).clamp(max = K)
    lv = None
    for t in range(T - 1, -1, -1):
        e = _emit_terms(_leaf_logp(lE, lp[:, t], ids[:, t], U[:, t], D[:, t], K), ids[:, t], x[:, t], miss[:, t])
        if lv is None:
            lv = torch.logsumexp(e, dim = 2)                                            # [b, N]
        else:
            inner = torch.logsumexp(_trans_logp(lT, A[:, t], B[:, t], G[:, t]) + lv[:, None, None, :], dim = 3)
            lv = torch.logsumexp(e + inner[:, :, slot], dim = 2)
    return torch.logsumexp(lpi[None, :] + lv, dim = 1)


def _definition_stats(tab, ext, ids, lp, x, miss, K):
    """The EM statistics by the definition, each on its own (float64 autograd): the tables enter the logits as a
    numerator copy and a normalizer copy, d / d numerator = F+, -d / d normalizer = F-; the start's F+ = d / d its
    log-probabilities. `Tp` [N, N], `Tm` [T - 1, N, N] (a position each), `Ep`, `Em` [N, V], `pi` [N], summed over the rows."""
    A, B, G, U, D = ext
    T, C = ids.size(1), ids.size(2)
    lTn = tab["lT"].clone().requires_grad_(True)
    lTd = tab["lT"][None].repeat(T - 1, 1, 1).requires_grad_(True)          # (a normalizer copy a position)
    lEn, lEd = tab["lE"].clone().requires_grad_(True), tab["lE"].clone().requires_grad_(True)
    lpi = tab["lpi"].clone().requires_grad_(True)
    slot = torch.arange(C, device = ids.device).clamp(max = K)
    lv = None
    for t in range(T - 1, -1, -1):
        tau = torch.bmm(U[:, t], D[:, t])[:, :, slot]
        num = lEn[:, ids[:, t]].permute(1, 0, 2) + lp[:, t, None, :] + tau
        den = lEd[:, ids[:, t]].permute(1, 0, 2) + lp[:, t, None, :] + tau
        e = _emit_terms(num - torch.logsumexp(den, dim = 2, keepdim = True), ids[:, t], x[:, t], miss[:, t])
        if lv is None:
            lv = torch.logsumexp(e, dim = 2)
        else:
            AB = torch.bmm(A[:, t], B[:, t])[:, :, None, :] + G[:, t][:, None, :, :]
            lt = lTn[None, :, None, :] + AB - torch.logsumexp(lTd[t][None, :, None, :] + AB, dim = 3, keepdim = True)
            inner = torch.logsumexp(lt + lv[:, None, None, :], dim = 3)
            lv = torch.logsumexp(e + inner[:, :, slot], dim = 2)
    torch.logsumexp(lpi[None, :] + lv, dim = 1).sum().backward()
    return dict(Tp = lTn.grad, Tm = -lTd.grad, Ep = lEn.grad, Em = -lEd.grad, pi = lpi.grad)


# ---------------------------------------------------------------------------------------------- the circuits

def _hmm_pc(T, N, K, r, re, tab):
    with juice.set_block_size(N):
        leaf = inputs(T - 1, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V),
                      parameterization = LowRank(rank = re, top_k = K))
        leaves, trans = {T - 1: leaf}, {}
        zs, first = leaf, None
        for var in range(T - 2, -1, -1):
            xs = leaf.duplicate(var, tie_params = True)
            leaves[var] = xs
            ns = summate(zs, num_node_blocks = 1, parameterization = LowRank(rank = r, top_k = K)) if first is None \
                else first.duplicate(zs, tie_params = True)
            first = first or ns
            trans[var] = ns
            zs = multiply(xs, ns)
        root = summate(zs, num_node_blocks = 1, block_size = 1)
    first.set_params(tab["lT"].exp().float().cpu())
    root.set_params(tab["lpi"].exp().float().cpu()[None, :])
    leaf.set_params(tab["lE"].exp().float().cpu().reshape(-1))
    pc = juice.compile(root)
    pc.to(DEV)
    pc.register_external_params_group("tr", [trans[t] for t in range(T - 1)])
    pc.register_external_params_group("em", [leaves[t] for t in range(T)])
    return pc, trans, leaves


def _tables(N, seed):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.log_softmax(torch.randn(*s, generator = g, dtype = torch.float64), dim = -1).to(DEV)
    return dict(lT = mk(N, N), lpi = mk(N), lE = mk(N, V))


def _batch(b, T, N, C, K, r, re, seed, masked = 0.5, scale = 0.7):
    """the external tensors (A, B, G, U, D) in the reference's layout, the candidates and their evidence, the tokens
    (each among its position's candidates) and a mask"""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: (scale * torch.randn(*s, generator = g, dtype = torch.float64)).to(DEV)
    ext = (rn(b, T - 1, N, r), rn(b, T - 1, r, N), rn(b, T - 1, K + 1, N), rn(b, T, N, re), rn(b, T, re, K + 1))
    ids = torch.stack([torch.stack([torch.randperm(V, generator = g)[:C] for _ in range(T)]) for _ in range(b)]).to(DEV)
    lp = torch.log_softmax(torch.randn(b, T, C, generator = g, dtype = torch.float64), dim = -1).to(DEV)
    pick = torch.randint(0, C, (b, T), generator = g).to(DEV)
    x = ids.gather(2, pick[..., None])[..., 0]
    miss = (torch.rand(b, T, generator = g) < masked).to(DEV)
    return ext, ids, lp, x, miss


def _pc_ext(ext, T, N):
    A, B, G, U, D = ext
    b = A.size(0)
    return {"tr": (A.reshape(b, (T - 1) * N, -1).float(), B.reshape(b, -1, N).float(), G.reshape(b, -1, N).float()),
            "em": (U.reshape(b, T * N, -1).float(), D.reshape(b, -1, D.size(-1)).float())}


def _rel(a, b):
    return float((a.double() - b.double()).abs().max()) / max(float(b.abs().max()), 1e-12)


# ---------------------------------------------------------------------------------------------- the HMM

@pytest.mark.parametrize("C, K", [(5, 2), (4, 4), (3, 6)])        # (a shared slot; none; more slots than candidates)
@pytest.mark.parametrize("masked", [0.0, 0.5, 1.0])
def test_hmm_values_are_the_definition(C, K, masked):
    T, N, r, re, b = 4, 4, 3, 2, 6
    tab = _tables(N, 0)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 1, masked = masked)
    with torch.no_grad():
        ll = pc(x, missing_mask = miss, external_params = _pc_ext(ext, T, N), categorical_evidence_logp = lp.float(),
                soft_evidence_cat_ids = ids).view(-1)
    ref = _ref_hmm(tab, ext, ids, lp, x, miss, K)
    assert torch.allclose(ll.double(), ref, atol = 1e-4, rtol = 1e-5), (ll, ref)


@pytest.mark.parametrize("masked", [0.0, 0.5])
def test_hmm_gradients_and_em_statistics_are_the_definitions(masked):
    T, N, C, K, r, re, b = 4, 4, 5, 2, 3, 2, 6
    tab = _tables(N, 2)
    pc, trans, leaves = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 3, masked = masked)

    # the reference: autograd of the definition, the tables as free logits (their rows' normalization inside)
    psiT, psiE = tab["lT"].clone().requires_grad_(True), tab["lE"].clone().requires_grad_(True)
    leaves_ref = [t.clone().requires_grad_(True) for t in ext] + [lp.clone().requires_grad_(True)]
    ref_tab = dict(lT = torch.log_softmax(psiT, -1), lpi = tab["lpi"], lE = torch.log_softmax(psiE, -1))
    _ref_hmm(ref_tab, leaves_ref[:5], ids, leaves_ref[5], x, miss, K).sum().backward()

    pext = _pc_ext(ext, T, N)
    g = {k_: tuple(torch.zeros_like(t) for t in v) for k_, v in pext.items()}
    glp = torch.zeros_like(lp, dtype = torch.float32)
    evi = dict(categorical_evidence_logp = lp.float(), soft_evidence_cat_ids = ids)
    pc.init_param_flows(flows_memory = 0.0)
    with torch.no_grad():
        pc(x, missing_mask = miss, external_params = pext, **evi)
    pc.backward(x, missing_mask = miss, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                allow_modify_flows = False, external_params_grad = g, categorical_evidence_logp_grad = glp, **evi)

    mine = [g["tr"][0].view(b, T - 1, N, r), g["tr"][1].view(b, T - 1, r, N), g["tr"][2].view(b, T - 1, K + 1, N),
            g["em"][0].view(b, T, N, re), g["em"][1].view(b, T, re, K + 1), glp]
    for name, a, ref in zip(("A", "B", "G", "U", "D", "lp"), mine, leaves_ref):
        assert _rel(a, ref.grad) < 2e-4, (name, _rel(a, ref.grad))

    # the tables: F+ - F- (the transition's flows and denominators; the leaf's two phases) summed over the copies
    layer_of = {ns: l for grp in pc.inner_layer_groups if grp.is_sum() for l in grp.layers for ns in l.nodes}
    gT, seen = torch.zeros(N, N, dtype = torch.float64, device = DEV), set()
    theta = tab["lT"].exp()
    for t in range(T - 1):
        ns, layer = trans[t], layer_of[trans[t]]
        desc = layer.parameterization
        info = [i for i in layer.external_node_infos if i.ns is ns][0]
        geo = desc._geometry(layer, info, DEV)
        off, Mb, Nb, cbs, bs = geo["grid"]
        dense = lambda buf, base: buf[base + off:base + off + N * N].view(Mb, Nb, cbs, bs).permute(0, 3, 1, 2).reshape(N, N).double()
        if ns._param_flow_range not in seen:
            seen.add(ns._param_flow_range)
            gT += dense(pc.param_flows, ns._param_flow_range[0])
        gT -= theta * dense(pc.denom_param_flows, desc._denom_offsets(layer)[id(ns)][0])
    il = pc.input_layer_group[0]
    gE, seen = torch.zeros(N, V, dtype = torch.float64, device = DEV), set()
    for t in range(T):
        rg = leaves[t]._param_flow_range
        if rg not in seen:
            seen.add(rg)
            rows = il.param_flows[rg[0]:rg[0] + N * 2 * V].view(N, 2 * V).double()
            gE += rows[:, :V] - rows[:, V:]
    assert _rel(gT, psiT.grad) < 2e-4, _rel(gT, psiT.grad)
    assert _rel(gE, psiE.grad) < 2e-4, _rel(gE, psiE.grad)


def test_hmm_differentiable_conditional():
    T, N, C, K, r, re, b = 4, 4, 5, 2, 3, 2, 6
    tab = _tables(N, 4)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 5)
    w = torch.linspace(-1.0, 2.0, b, device = DEV, dtype = torch.float64)       # (both signs)

    leaves_ref = [t.clone().requires_grad_(True) for t in ext] + [lp.clone().requires_grad_(True)]
    none = torch.zeros_like(miss)
    ref = _ref_hmm(tab, leaves_ref[:5], ids, leaves_ref[5], x, none, K) - _ref_hmm(tab, leaves_ref[:5], ids, leaves_ref[5], x, miss, K)
    (w * ref).sum().backward()

    leaves_pc = [t.float().clone().requires_grad_(True) for t in ext] + [lp.float().clone().requires_grad_(True)]
    kw = dict(external_params = _pc_ext(leaves_pc[:5], T, N), categorical_evidence_logp = leaves_pc[5],
              soft_evidence_cat_ids = ids, differentiable = True)
    cond = pc(x, **kw).view(-1) - pc(x, missing_mask = miss, **kw).view(-1)
    (w.float() * cond).sum().backward()
    assert torch.allclose(cond.double(), ref.detach(), atol = 1e-4)
    for name, a, b_ in zip(("A", "B", "G", "U", "D", "lp"), leaves_pc, leaves_ref):
        assert _rel(a.grad, b_.grad) < 2e-4, (name, _rel(a.grad, b_.grad))


def test_zero_tensors_are_the_plain_circuit():
    """A, B, G, U, D all zero: the circuit of plain soft-evidence leaves and plain sums, as pyjuice evaluates it."""
    T, N, C, K, r, re, b = 4, 4, 5, 2, 3, 2, 6
    tab = _tables(N, 6)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 7, scale = 0.0)
    with juice.set_block_size(N):
        leaf = inputs(T - 1, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V))
        zs, first = leaf, None
        for var in range(T - 2, -1, -1):
            xs = leaf.duplicate(var, tie_params = True)
            ns = summate(zs, num_node_blocks = 1) if first is None else first.duplicate(zs, tie_params = True)
            first = first or ns
            zs = multiply(xs, ns)
        root = summate(zs, num_node_blocks = 1, block_size = 1)
    first.set_params(tab["lT"].exp().float().cpu())
    root.set_params(tab["lpi"].exp().float().cpu()[None, :])
    leaf.set_params(tab["lE"].exp().float().cpu().reshape(-1))
    plain = juice.compile(root)
    plain.to(DEV)
    evi = dict(categorical_evidence_logp = lp.float(), soft_evidence_cat_ids = ids)
    with torch.no_grad():
        a = pc(x, missing_mask = miss, external_params = _pc_ext(ext, T, N), **evi).view(-1)
        b_ = plain(x, missing_mask = miss, **evi).view(-1)
    assert torch.allclose(a, b_, atol = 1e-5), (a, b_)


# ---------------------------------------------------------------------------------------------- a tree

def test_a_tree_with_a_reading_sum_beside_a_plain_one():
    """root -> z0; z0's product holds the leaf of x0, a sum READING x0's token over the subtree of x1, and a plain sum
    over the subtree of x2 (not a chain: the reading sum shares its product with another sum)."""
    N, C, K, r, re, b = 4, 4, 2, 2, 2, 5
    g = torch.Generator().manual_seed(8)
    mk = lambda *s: torch.log_softmax(torch.randn(*s, generator = g, dtype = torch.float64), dim = -1).to(DEV)
    lE, lTa, lTb, lpi = mk(N, V), mk(N, N), mk(N, N), mk(N)
    with juice.set_block_size(N):
        leaf0 = inputs(0, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V),
                       parameterization = LowRank(rank = re, top_k = K))
        leaf1, leaf2 = leaf0.duplicate(1, tie_params = True), leaf0.duplicate(2, tie_params = True)
        sa = summate(leaf1, num_node_blocks = 1, parameterization = LowRank(rank = r, top_k = K))
        sb = summate(leaf2, num_node_blocks = 1)
        root = summate(multiply(leaf0, sa, sb), num_node_blocks = 1, block_size = 1)
    sa.set_params(lTa.exp().float().cpu()); sb.set_params(lTb.exp().float().cpu())
    root.set_params(lpi.exp().float().cpu()[None, :]); leaf0.set_params(lE.exp().float().cpu().reshape(-1))
    pc = juice.compile(root)
    pc.to(DEV)

    rn = lambda *s: (0.7 * torch.randn(*s, generator = g, dtype = torch.float64)).to(DEV)
    A, B, G = rn(b, N, r), rn(b, r, N), rn(b, K + 1, N)
    U, D = rn(b, 3, N, re), rn(b, 3, re, K + 1)
    ids = torch.stack([torch.stack([torch.randperm(V, generator = g)[:C] for _ in range(3)]) for _ in range(b)]).to(DEV)
    lp = torch.log_softmax(torch.randn(b, 3, C, generator = g, dtype = torch.float64), dim = -1).to(DEV)
    x = ids.gather(2, torch.randint(0, C, (b, 3, 1), generator = g).to(DEV))[..., 0]
    miss = (torch.rand(b, 3, generator = g) < 0.5).to(DEV)

    def ref(A, B, G, U, D, lp):
        slot = torch.arange(C, device = DEV).clamp(max = K)
        e = [_emit_terms(_leaf_logp(lE, lp[:, v], ids[:, v], U[:, v], D[:, v], K), ids[:, v], x[:, v], miss[:, v]) for v in range(3)]
        l1, l2 = torch.logsumexp(e[1], dim = 2), torch.logsumexp(e[2], dim = 2)                   # [b, N]
        ina = torch.logsumexp(_trans_logp(lTa, A, B, G) + l1[:, None, None, :], dim = 3)          # [b, N, S]
        inb = torch.logsumexp(lTb[None] + l2[:, None, :], dim = 2)                                # [b, N]
        lv = torch.logsumexp(e[0] + ina[:, :, slot], dim = 2) + inb
        return torch.logsumexp(lpi[None] + lv, dim = 1)

    leaves_ref = [t.clone().requires_grad_(True) for t in (A, B, G, U, D, lp)]
    r_ll = ref(*leaves_ref)
    r_ll.sum().backward()
    gA, gB, gG = torch.zeros(b, N, r, device = DEV), torch.zeros(b, r, N, device = DEV), torch.zeros(b, K + 1, N, device = DEV)
    gU, gD = [torch.zeros(b, N, re, device = DEV) for _ in range(3)], [torch.zeros(b, re, K + 1, device = DEV) for _ in range(3)]
    glp = torch.zeros(b, 3, C, device = DEV)
    leafs = (leaf0, leaf1, leaf2)
    ext = {sa: (A.float(), B.float(), G.float()), **{leafs[v]: (U[:, v].float(), D[:, v].float()) for v in range(3)}}
    grads = {sa: (gA, gB, gG), **{leafs[v]: (gU[v], gD[v]) for v in range(3)}}
    evi = dict(categorical_evidence_logp = lp.float(), soft_evidence_cat_ids = ids)
    with torch.no_grad():
        ll = pc(x, missing_mask = miss, external_params = ext, **evi).view(-1)
    pc.backward(x, missing_mask = miss, compute_param_flows = False, logspace_flows = True, allow_modify_flows = False,
                external_params_grad = grads, categorical_evidence_logp_grad = glp, **evi)
    assert torch.allclose(ll.double(), r_ll.detach(), atol = 1e-4), (ll, r_ll)
    mine = (gA, gB, gG, torch.stack(gU, 1), torch.stack(gD, 1), glp)
    for name, a, b_ in zip(("A", "B", "G", "U", "D", "lp"), mine, leaves_ref):
        assert _rel(a, b_.grad) < 2e-4, (name, _rel(a, b_.grad))


# ---------------------------------------------------------------------------------------------- what is refused

def test_compile_refuses_a_reading_sum_without_one_leaf_beside_it():
    N = 4
    with juice.set_block_size(N):
        l0 = inputs(0, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V))
        l1 = inputs(1, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V))
        l2 = inputs(2, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V))
        s = summate(l2, num_node_blocks = 1, parameterization = LowRank(rank = 2, top_k = 2))
        root = summate(multiply(l0, l1, s), num_node_blocks = 1, block_size = 1)    # (two leaves: which token?)
    with pytest.raises(AssertionError, match = "exactly one input node vector"):
        juice.compile(root)


def test_a_leaf_parameterization_needs_soft_evidence_and_top_k():
    with pytest.raises(AssertionError, match = "SoftEvidenceCategorical"):
        inputs(0, num_nodes = 4, dist = dists.Categorical(num_cats = V), parameterization = LowRank(rank = 2, top_k = 2))
    with pytest.raises(AssertionError, match = "top_k"):
        inputs(0, num_nodes = 4, dist = dists.SoftEvidenceCategorical(num_cats = V), parameterization = LowRank(rank = 2))


@pytest.mark.parametrize("masked", [0.0, 0.5, 1.0])
def test_hmm_em_numerator_and_denominator_each_are_the_definitions(masked):
    """F+ and F- SEPARATELY (the M-step divides them, so their difference -- the gradient -- is not enough): the tables
    enter the definition's logits as a numerator copy and a normalizer copy, d / d numerator = F+, -d / d normalizer =
    F-. Every token observed: the pass EM trains on. Every token missing: the counts expected under each sample's own
    model, which an Anemone step is anchored to. Mixed: a missing token adds its posterior counts to both phases --
    the leaf's per-sample probabilities where no sum reads it (here the last position), not its shared parameters."""
    T, N, C, K, r, re, b = 4, 4, 5, 2, 3, 2, 6
    tab = _tables(N, 10)
    pc, trans, leaves = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 11, masked = masked)
    assert masked not in (0.0, 1.0) or bool((miss == (masked == 1.0)).all())
    assert masked != 0.5 or (bool(miss[:, T - 1].any()) and bool(miss[:, :T - 1].any()))   # (both kinds of leaf)
    ref = _definition_stats(tab, ext, ids, lp, x, miss, K)

    evi = dict(categorical_evidence_logp = lp.float(), soft_evidence_cat_ids = ids)
    pc.init_param_flows(flows_memory = 0.0)
    with torch.no_grad():
        lls = pc(x, missing_mask = miss, external_params = _pc_ext(ext, T, N), **evi)
    pc.backward(x, missing_mask = miss, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                allow_modify_flows = False, **evi)
    if masked == 1.0:                                                       # (every sample's model is normalized)
        assert float(lls.abs().max()) < 1e-5, float(lls.abs().max())

    layer_of = {ns: l for grp in pc.inner_layer_groups if grp.is_sum() for l in grp.layers for ns in l.nodes}
    Fp, Fm, seen = torch.zeros(N, N, dtype = torch.float64, device = DEV), torch.zeros(N, N, dtype = torch.float64, device = DEV), set()
    for t in range(T - 1):
        ns, layer = trans[t], layer_of[trans[t]]
        desc = layer.parameterization
        info = [i for i in layer.external_node_infos if i.ns is ns][0]
        off, Mb, Nb, cbs, bs = desc._geometry(layer, info, DEV)["grid"]
        dense = lambda buf, base: buf[base + off:base + off + N * N].view(Mb, Nb, cbs, bs).permute(0, 3, 1, 2).reshape(N, N).double()
        if ns._param_flow_range not in seen:
            seen.add(ns._param_flow_range)
            Fp += dense(pc.param_flows, ns._param_flow_range[0])
        Fm += tab["lT"].exp() * dense(pc.denom_param_flows, desc._denom_offsets(layer)[id(ns)][0])
    assert _rel(Fp, ref["Tp"]) < 2e-4, _rel(Fp, ref["Tp"])
    assert _rel(Fm, ref["Tm"].sum(0)) < 2e-4, _rel(Fm, ref["Tm"].sum(0))

    il = pc.input_layer_group[0]
    Ep, Em, seen = torch.zeros(N, V, dtype = torch.float64, device = DEV), torch.zeros(N, V, dtype = torch.float64, device = DEV), set()
    for t in range(T):
        rg = leaves[t]._param_flow_range
        if rg not in seen:
            seen.add(rg)
            rows = il.param_flows[rg[0]:rg[0] + N * 2 * V].view(N, 2 * V).double()
            Ep += rows[:, :V]; Em += rows[:, V:]
    assert _rel(Ep, ref["Ep"]) < 2e-4, _rel(Ep, ref["Ep"])
    assert _rel(Em, ref["Em"]) < 2e-4, _rel(Em, ref["Em"])
