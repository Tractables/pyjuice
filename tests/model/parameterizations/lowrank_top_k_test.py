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

import contextlib

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

def _hmm_pc(T, N, K, r, re, tab, V = V):
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


def _tables(N, seed, V = V):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.log_softmax(torch.randn(*s, generator = g, dtype = torch.float64), dim = -1).to(DEV)
    return dict(lT = mk(N, N), lpi = mk(N), lE = mk(N, V))


def _batch(b, T, N, C, K, r, re, seed, masked = 0.5, scale = 0.7, V = V):
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


def test_the_autograd_backward_leaves_no_gradients_on_the_circuit():
    """Once the autograd backward of a differentiable call returns, the gradients are autograd's alone: the circuit
    keeps none of the destinations its backward wrote (they would stay allocated until its next backward staged
    others) -- they are freed with their last reader."""
    import gc, weakref
    from pyjuice.layer.parameterized_sum_layer import EXTERNAL_PARAMS_GRAD_KWARG
    T, N, C, K, r, re, b = 4, 4, 5, 2, 3, 2, 6
    pc, _, _ = _hmm_pc(T, N, K, r, re, _tables(N, 4))
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 5)
    leaves = [t.float().clone().requires_grad_(True) for t in ext] + [lp.float().clone().requires_grad_(True)]
    seen, real = [], pc.backward

    def spy(*args, **kwargs):
        for ts in kwargs[EXTERNAL_PARAMS_GRAD_KWARG].values():
            seen.extend(weakref.ref(t) for t in ((ts,) if torch.is_tensor(ts) else ts))
        return real(*args, **kwargs)

    pc.backward = spy
    try:
        ll = pc(x, missing_mask = miss, external_params = _pc_ext(leaves[:5], T, N), categorical_evidence_logp = leaves[5],
                soft_evidence_cat_ids = ids, differentiable = True)
        ll.sum().backward()
    finally:
        del pc.backward
    assert len(seen) > 0 and all(float(t.grad.abs().sum()) > 0 for t in leaves[:5])
    del ll
    for t in leaves:
        t.grad = None
    gc.collect()
    assert all(w() is None for w in seen), sum(w() is not None for w in seen)


# ---------------------------------------------------------------------------------------------- a map of rows

def test_hmm_conditional_in_one_batch():
    """Both passes of the conditional in ONE call -- the batch twice, each half reading the same rows of the tensors
    (`external_params_rows`) -- are the two calls: the values and every gradient, from one forward and one backward."""
    T, N, C, K, r, re, b = 4, 4, 5, 2, 3, 2, 6
    tab = _tables(N, 4)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 5)
    w = torch.linspace(-1.0, 2.0, b, device = DEV)                               # (both signs)

    def run(one):
        leaves = [t.float().clone().requires_grad_(True) for t in ext] + [lp.float().clone().requires_grad_(True)]
        pext = _pc_ext(leaves[:5], T, N)
        if one:
            ll = pc(x.repeat(2, 1), missing_mask = torch.cat([torch.zeros_like(miss), miss]), external_params = pext,
                    external_params_rows = torch.arange(b, device = DEV).repeat(2), soft_evidence_cat_ids = ids.repeat(2, 1, 1),
                    categorical_evidence_logp = leaves[5].repeat(2, 1, 1), differentiable = True).view(-1)
            cond = ll[:b] - ll[b:]
        else:
            kw = dict(external_params = pext, categorical_evidence_logp = leaves[5], soft_evidence_cat_ids = ids,
                      differentiable = True)
            cond = pc(x, **kw).view(-1) - pc(x, missing_mask = miss, **kw).view(-1)
        runs = pc._num_runs
        (w * cond).sum().backward()
        return cond.detach(), [t.grad for t in leaves], pc._num_runs - runs

    c1, g1, passes = run(True)
    c2, g2, _ = run(False)
    assert passes == 1, passes                                                    # (the backward alone: no forward again)
    assert torch.allclose(c1, c2, atol = 1e-5), (c1, c2)
    for name, a, b_ in zip(("A", "B", "G", "U", "D", "lp"), g1, g2):
        assert _rel(a, b_) < 1e-5, (name, _rel(a, b_))


def test_rows_map_reads_the_rows_it_names():
    """A map with repeats, in any order: the values, the per-sample gradients and the EM statistics of the gathered
    tensors."""
    T, N, C, K, r, re, b = 4, 4, 5, 2, 3, 2, 5
    tab = _tables(N, 7)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 8)
    rows = torch.tensor([3, 0, 0, 4, 2, 3, 1], device = DEV)
    x2, ids2, miss2 = x[rows], ids[rows], miss[rows]
    lp2 = lp[rows].float().contiguous()
    evi = dict(categorical_evidence_logp = lp2, soft_evidence_cat_ids = ids2)

    def run(pext, **kw):
        g = {k_: tuple(torch.zeros(rows.numel(), *t.shape[1:], device = DEV) for t in v) for k_, v in pext.items()}
        pc.init_param_flows(flows_memory = 0.0)
        with torch.no_grad():
            ll = pc(x2, missing_mask = miss2, external_params = pext, **kw, **evi).view(-1)
        pc.backward(x2, missing_mask = miss2, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                    allow_modify_flows = False, external_params_grad = g, **evi)
        il = pc.input_layer_group[0]
        return ll, g, pc.param_flows.clone(), pc.denom_param_flows.clone(), il.param_flows.clone()

    got = run(_pc_ext(ext, T, N), external_params_rows = rows)
    want = run(_pc_ext(tuple(t[rows] for t in ext), T, N))
    assert torch.allclose(got[0], want[0], atol = 1e-6), (got[0], want[0])
    for k_ in got[1]:
        for a, b_ in zip(got[1][k_], want[1][k_]):
            assert _rel(a, b_) < 1e-6, (k_, _rel(a, b_))
    for a, b_ in zip(got[2:], want[2:]):
        assert _rel(a, b_) < 1e-6, _rel(a, b_)


def test_rows_map_needs_a_parameterization_that_reads_through_it():
    with juice.set_block_size(4):
        leaf = inputs(0, num_node_blocks = 1, dist = dists.Categorical(num_cats = V))
        ns = summate(leaf, num_node_blocks = 1, parameterization = LowRank(rank = 2))
        root = summate(multiply(ns, inputs(1, num_node_blocks = 1, dist = dists.Categorical(num_cats = V))),
                       num_node_blocks = 1, block_size = 1)
    pc = juice.compile(root)
    pc.to(DEV)
    x = torch.randint(0, V, (3, 2), device = DEV)
    ext = {ns: (torch.randn(2, 4, 2, device = DEV), torch.randn(2, 2, 4, device = DEV))}
    with pytest.raises(NotImplementedError, match = "external_params_rows"):
        pc(x, external_params = ext, external_params_rows = torch.tensor([0, 1, 1], device = DEV))
    with pytest.raises(AssertionError, match = "external_params_rows"):
        pc(x, external_params_rows = torch.tensor([0, 1, 1], device = DEV))


def test_the_fused_forward_is_the_reference():
    """The fused forward (`lowrank/top_k_fused.py`: the kernels' shapes, 128 nodes, top_k 64) against the float64
    reference -- the circuit's values (fp16 logits, TF32 slots: ~1e-2 nats) -- and, both passes of the conditional one
    batch through the map, the two calls' values (up to the kernels' rounding)."""
    import pyjuice.parameterizations.lowrank.top_k_fused as tkf
    if tkf.top_k_module() is None:
        pytest.skip("the CUDA module did not build")
    T, N, C, K, r, re, b, V2 = 3, 128, 70, 64, 20, 8, 4, 80
    tab = _tables(N, 11, V2)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab, V2)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 12, V = V2)
    pext, lp = _pc_ext(ext, T, N), lp.float()
    calls = []
    real = tkf.forward
    tkf.forward = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        vals = {}
        for fused in (False, True):
            tkf.TOP_K_FUSED = fused
            with torch.no_grad():
                full = pc(x, external_params = pext, categorical_evidence_logp = lp, soft_evidence_cat_ids = ids).view(-1)
                vis = pc(x, missing_mask = miss, external_params = pext, categorical_evidence_logp = lp,
                         soft_evidence_cat_ids = ids).view(-1)
                one = pc(x.repeat(2, 1), missing_mask = torch.cat([torch.zeros_like(miss), miss]), external_params = pext,
                         external_params_rows = torch.arange(b, device = DEV).repeat(2),
                         categorical_evidence_logp = lp.repeat(2, 1, 1), soft_evidence_cat_ids = ids.repeat(2, 1, 1)).view(-1)
            vals[fused] = (full, vis, one)
    finally:
        tkf.TOP_K_FUSED = True
        tkf.forward = real
    assert len(calls) == 3 * (T - 1), len(calls)                                  # (every transition, every call, fused)
    for a, ref in zip(vals[True][:2], vals[False][:2]):
        assert float((a - ref).abs().max()) < 2e-2, float((a - ref).abs().max())
    full, vis, one = vals[True]
    # (the same values up to the kernels' rounding: a batch's split into the CTAs' shares sets where each part's
    # running maximum -- and so W's TF32 rounding -- falls)
    assert torch.allclose(one, torch.cat([full, vis]), atol = 2e-3), (one, torch.cat([full, vis]))


@pytest.mark.parametrize("C", [40, 200])                         # (fewer candidates than slots; a shared slot)
def test_the_fused_leaf_is_the_torch_passes(C):
    """The leaves by the kernels (`lowrank/top_k_leaf_fused.py`: 128 nodes) against their torch passes -- the sums'
    passes the torch reference in both, to see the leaves alone --: the differentiable conditional (one call, the map)
    and its gradients, and an EM backward's statistics on the all-observed pass and on the visible-only one (the
    missing tokens' posterior counts)."""
    import pyjuice.parameterizations.lowrank.top_k_fused as tkf
    import pyjuice.parameterizations.lowrank.top_k_leaf_fused as tlf
    import pyjuice.parameterizations.lowrank.top_k_backward as tkb
    if tlf.module() is None:
        pytest.skip("the CUDA module did not build")
    T, N, K, r, re, b, V2 = 3, 128, 64, 20, 8, 4, 300
    tab = _tables(N, 21, V2)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab, V2)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 22, V = V2)
    pext, lp = _pc_ext(ext, T, N), lp.float()
    rows = torch.arange(b, device = DEV).repeat(2)
    ran = []
    real_sum, real_leaf, real_bw = tkf.sum_forward, tlf.leaf_forward, tkb.serves
    tkf.sum_forward, tkb.serves = (lambda *a, **k: False), (lambda *a, **k: False)
    tlf.leaf_forward = lambda *a, **k: (lambda got: (ran.append(got), got)[1])(real_leaf(*a, **k))

    def passes():
        ts = {g_: tuple(t.clone().requires_grad_(True) for t in v) for g_, v in pext.items()}
        lpg = lp.clone().requires_grad_(True)
        ll = pc(x.repeat(2, 1), missing_mask = torch.cat([torch.zeros_like(miss), miss]), external_params = ts,
                external_params_rows = rows, categorical_evidence_logp = lpg.repeat(2, 1, 1),
                soft_evidence_cat_ids = ids.repeat(2, 1, 1), differentiable = True).view(-1)
        (-(ll[:b] - ll[b:]).sum()).backward()
        out = [ll.detach()] + [t.grad for v in ts.values() for t in v] + [lpg.grad]
        for m_ in (None, miss):
            pc.init_param_flows(flows_memory = 0.0)
            with torch.no_grad():
                pc(x, missing_mask = m_, external_params = pext, categorical_evidence_logp = lp, soft_evidence_cat_ids = ids)
            pc.backward(x, missing_mask = m_, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                        allow_modify_flows = False, external_params = pext, categorical_evidence_logp = lp,
                        soft_evidence_cat_ids = ids)
            out += [pc.input_layer_group[0].param_flows.clone(), pc.param_flows.clone()]
        return out

    try:
        tkf.TOP_K_FUSED = False
        ref = passes()
        tkf.TOP_K_FUSED = True
        got = passes()
    finally:
        tkf.TOP_K_FUSED = True
        tkf.sum_forward, tlf.leaf_forward, tkb.serves = real_sum, real_leaf, real_bw
    assert ran == [False] * 3 + [True] * 3, ran                                     # (the torch passes, then the kernels)
    assert torch.allclose(got[0], ref[0], atol = 1e-4), float((got[0] - ref[0]).abs().max())
    names = ("A", "B", "G", "U", "D", "lp", "leaf F (all observed)", "sums' F (all observed)", "leaf F (visible)",
             "sums' F (visible)")
    for name, a, r_ in zip(names, got[1:], ref[1:]):
        assert _rel(a, r_) < 1e-4, (name, _rel(a, r_))


@pytest.mark.parametrize("re, S, layout, T", [(64, 65, "groups", 4), (20, 9, "groups", 4), (6, 65, "ranks strided", 4),
                                              (8, 33, "slots strided", 70)])
def test_the_tilt_kernels_are_the_products(re, S, layout, T):
    """The leaves' tilt products by their kernels (`kernels/c/lowrank_top_k_tilt.cu`) against float64: the forward's
    log2 e (A B)^T, and the gradients of destination rows reading the tensors' rows through a map -- into destinations
    of NaN handed over unzeroed (each one's first writer stores it whole), and added into others; two positions sharing
    one destination (the second adds) -- at the groups' layout (A [n, T M, re], B [n, T re, S] split along their axis)
    and with the ranks or the slots strided (the scalar paths); rank counts and slots below the kernels' 64 and 65, and
    more positions than a launch takes."""
    import math
    import pyjuice.parameterizations.lowrank.top_k_leaf_fused as tlf
    from pyjuice.layer.parameterized_sum_layer import StagedExternalParams
    if tlf.tilt_module() is None:
        pytest.skip("the CUDA module did not build")
    Bx, M = 3, 256
    g = torch.Generator(device = DEV).manual_seed(100 * re + S)
    rn = lambda *s: torch.randn(*s, device = DEV, generator = g)
    U = rn(Bx, re, T * M).transpose(1, 2) if layout == "ranks strided" else rn(Bx, T * M, re)
    Dd = rn(Bx, S, T * re).transpose(1, 2) if layout == "slots strided" else rn(Bx, T * re, S)
    tilts = [(U[:, t * M:(t + 1) * M], Dd[:, t * re:(t + 1) * re]) for t in range(T)]
    tau = torch.empty(T, Bx, S, M, device = DEV)
    tlf._tilts(tilts, tau)
    ref = torch.stack([torch.bmm(Bt.double().transpose(1, 2), A.double().transpose(1, 2)) for A, Bt in tilts])
    assert _rel(tau, ref / math.log(2.0)) < 1e-6, _rel(tau, ref / math.log(2.0))

    emap = torch.tensor([2, 0, 1, 1, 0, 2], device = DEV)
    D = rn(T, emap.numel(), S, M)
    want = [[torch.bmm(D[t].double().transpose(1, 2), Bt.double()[emap].transpose(1, 2)),
             torch.bmm(A.double()[emap].transpose(1, 2), D[t].double().transpose(1, 2))]
            for t, (A, Bt) in enumerate(tilts)]
    want[T - 2] = [a + b for a, b in zip(want[T - 2], want[T - 1])]       # (the last position writes the one before's)
    nodes = [dict(ns = t) for t in range(T)]
    for fresh, fill in ((True, float("nan")), (False, 1.0)):
        gU = torch.full((emap.numel(), T * M, re), fill, device = DEV)
        gD = torch.full((emap.numel(), T * re, S), fill, device = DEV)
        dst = [(gU[:, t * M:(t + 1) * M], gD[:, t * re:(t + 1) * re]) for t in range(T - 1)]
        gv = StagedExternalParams({t: dst[min(t, T - 2)] for t in range(T)})
        gv.fresh = fresh
        tlf._tilt_grads(nodes, nodes, D, dict(enumerate(tilts)), gv, fresh, emap)
        base = 0.0 if fresh else 1.0
        for t in range(T - 1):
            for name, got, w in zip(("A", "B"), dst[t], want[t]):
                assert _rel(got - base, w) < 2e-6, (fresh, t, name, _rel(got - base, w))


def test_the_kernels_initialize_the_gradient_destinations():
    """Destinations handed over unzeroed (`StagedExternalParams.fresh`: every node's backward initializes what it
    writes) keep whatever they held -- NaN here -- until written: the kernels' backward (the leaves' tilt products
    storing theirs whole, the sums' zeroing theirs before they add) gives the gradients of zeroed destinations and of
    the circuit's own buffer (zeroed, added into)."""
    import pyjuice.parameterizations.lowrank.top_k_leaf_fused as tlf
    pc, (ext, ids, lp, x, miss), (T, N, b) = _mask_case(True)
    if tlf.tilt_module() is None:
        pytest.skip("the CUDA module did not build")
    pext, lp = _pc_ext(ext, T, N), lp.float()
    evi = dict(categorical_evidence_logp = lp, soft_evidence_cat_ids = ids)
    tm, stores = tlf.tilt_module(), []
    real = tm.top_k_leaf_tilt_grads
    tm.top_k_leaf_tilt_grads = lambda *a: (stores.append(a[5]), real(*a))[1]

    def grads(fill):
        with torch.no_grad():
            pc(x, missing_mask = miss, external_params = pext, **evi)
        kw = dict(missing_mask = miss, compute_param_flows = False, logspace_flows = True, allow_modify_flows = False,
                  **evi)
        if fill is None:
            pc.backward(x, **kw)
            return [t.clone() for grp in ("tr", "em") for t in pc.get_external_params_grad(grp)]
        g = {k_: tuple(torch.full_like(t, fill) for t in v) for k_, v in pext.items()}
        pc.backward(x, external_params_grad = g, **kw)
        return [t for v in g.values() for t in v]

    try:
        with _paths(True):
            got = [grads(float("nan")), grads(0.0), grads(None)]
    finally:
        tm.top_k_leaf_tilt_grads = real
    assert stores == [True, True, False], stores                    # (stored over the unzeroed; added into the buffer)
    for name, a, z, o in zip(("A", "B", "G", "U", "D"), *got):
        assert bool(torch.isfinite(a).all()), name
        assert _rel(a, z) < 1e-5 and _rel(o, z) < 1e-5, (name, _rel(a, z), _rel(o, z))


def test_the_backward_kernels_are_the_reference():
    """The sums' backward by the kernels (`lowrank/top_k_backward.py`, `kernels/c/lowrank_top_k.cu`: 128 nodes, 128
    children) against the reference
    (`_sum_backward_terms`) -- the forward passes the torch references in both, the leaves too --: the differentiable
    conditional's gradients (one call, the map; the kernels' W from the forward's fp16 logits: ~1e-3), a backward's
    node flows without gradients or EM statistics (the element flows alone), and the EM statistics of a backward on
    the all-observed pass and on the visible-only one (the sums' F+, F- / theta and the leaves')."""
    import pyjuice.parameterizations.lowrank.top_k_fused as tkf
    import pyjuice.parameterizations.lowrank.top_k_leaf_fused as tlf
    import pyjuice.parameterizations.lowrank.top_k_backward as tkb
    if tkf.top_k_module() is None or tkb.triton is None:
        pytest.skip("the CUDA module did not build, or no Triton")
    T, N, C, K, r, re, b, V2 = 3, 128, 70, 64, 20, 8, 4, 80
    tab = _tables(N, 31, V2)
    pc, _, _ = _hmm_pc(T, N, K, r, re, tab, V2)
    ext, ids, lp, x, miss = _batch(b, T, N, C, K, r, re, 32, V = V2)
    pext, lp = _pc_ext(ext, T, N), lp.float()
    rows = torch.arange(b, device = DEV).repeat(2)
    ran = []
    real_sum, real_leaf, real_terms = tkf.sum_forward, tlf.leaf_forward, tkb.terms
    tkf.sum_forward, tlf.leaf_forward = (lambda *a, **k: False), (lambda *a, **k: False)
    tkb.terms = lambda *a, **k: (ran.append(1), real_terms(*a, **k))[1]

    def passes():
        ts = {g_: tuple(t.clone().requires_grad_(True) for t in v) for g_, v in pext.items()}
        lpg = lp.clone().requires_grad_(True)
        ll = pc(x.repeat(2, 1), missing_mask = torch.cat([torch.zeros_like(miss), miss]), external_params = ts,
                external_params_rows = rows, categorical_evidence_logp = lpg.repeat(2, 1, 1),
                soft_evidence_cat_ids = ids.repeat(2, 1, 1), differentiable = True).view(-1)
        (-(ll[:b] - ll[b:]).sum()).backward()
        out = [t.grad for v in ts.values() for t in v] + [lpg.grad]
        with torch.no_grad():
            pc(x, missing_mask = miss, external_params = pext, categorical_evidence_logp = lp, soft_evidence_cat_ids = ids)
        pc.backward(x, missing_mask = miss, compute_param_flows = False, logspace_flows = True, allow_modify_flows = False,
                    external_params = pext, categorical_evidence_logp = lp, soft_evidence_cat_ids = ids)
        out.append(pc.node_flows.clone())
        for m_ in (None, miss):
            pc.init_param_flows(flows_memory = 0.0)
            with torch.no_grad():
                pc(x, missing_mask = m_, external_params = pext, categorical_evidence_logp = lp, soft_evidence_cat_ids = ids)
            pc.backward(x, missing_mask = m_, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                        allow_modify_flows = False, external_params = pext, categorical_evidence_logp = lp,
                        soft_evidence_cat_ids = ids)
            out += [pc.param_flows.clone(), pc.input_layer_group[0].param_flows.clone()]
            if pc.denom_param_flows is not None:
                out.append(pc.denom_param_flows.clone())
        return out

    try:
        tkf.TOP_K_FUSED = False
        ref = passes()
        tkf.TOP_K_FUSED = True
        got = passes()
    finally:
        tkf.TOP_K_FUSED = True
        tkf.sum_forward, tlf.leaf_forward, tkb.terms = real_sum, real_leaf, real_terms
    assert len(ran) == 4 * (T - 1), len(ran)                              # (the kernels: every transition, every backward)
    for name, a, r_ in zip(("A", "B", "G", "U", "D", "lp"), got[:6], ref[:6]):
        assert _rel(a, r_) < 5e-3, (name, _rel(a, r_))
    fin = torch.isfinite(ref[6])
    assert bool((torch.isfinite(got[6]) == fin).all())
    assert float((got[6][fin] - ref[6][fin]).abs().max()) < 5e-3, float((got[6][fin] - ref[6][fin]).abs().max())
    assert len(got) == len(ref) and len(got) > 7
    for k, (a, r_) in enumerate(zip(got[7:], ref[7:])):                    # (the EM statistics: the sums', the leaves')
        assert _rel(a, r_) < 5e-3, (k, _rel(a, r_))


@pytest.mark.parametrize("keep", [False, True])
def test_the_backward_kernels_give_a_sample_its_own_gradients_through_the_map(keep):
    """The kernels' backward with a map of rows and a gradient a SAMPLE (no fold: a sample a work item, whatever the
    items of the forward whose sums' stash it takes -- `keep`: the forward ran with grad on and kept it, else the
    backward makes its own) against the torch passes: every gradient and the EM statistics."""
    import pyjuice.parameterizations.lowrank.top_k_fused as tkf
    import pyjuice.parameterizations.lowrank.top_k_backward as tkb
    if tkf.top_k_module() is None or tkb.triton is None:
        pytest.skip("the CUDA module did not build, or no Triton")
    T, N, C, K, r, re, V2 = 3, 128, 70, 64, 20, 8, 80
    pc, _, _ = _hmm_pc(T, N, K, r, re, _tables(N, 33, V2), V2)
    ext = _batch(4, T, N, C, K, r, re, 34, V = V2)[0]
    rows = torch.tensor([3, 0, 0, 2, 1, 3, 0], device = DEV)                  # (rows read by one, two and three samples)
    _, ids, lp, x, miss = _batch(rows.numel(), T, N, C, K, r, re, 35, V = V2)
    pext, evi = _pc_ext(ext, T, N), dict(categorical_evidence_logp = lp.float(), soft_evidence_cat_ids = ids)
    ran, real_terms = [], tkb.terms
    tkb.terms = lambda *a, **k: (ran.append(k.get("kept", a[-1]) is not None), real_terms(*a, **k))[1]

    def run():
        g = {k_: tuple(torch.zeros(rows.numel(), *t.shape[1:], device = DEV) for t in v) for k_, v in pext.items()}
        pc.init_param_flows(flows_memory = 0.0)
        with torch.set_grad_enabled(keep):
            pc(x, missing_mask = miss, external_params = pext, external_params_rows = rows, **evi)
        pc.backward(x, missing_mask = miss, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                    allow_modify_flows = False, external_params_grad = g, **evi)
        return [t for v in g.values() for t in v] + [pc.param_flows.clone(), pc.denom_param_flows.clone(),
                                                     pc.input_layer_group[0].param_flows.clone()]

    try:
        tkf.TOP_K_FUSED = False
        ref = run()
        tkf.TOP_K_FUSED = True
        got = run()
    finally:
        tkf.TOP_K_FUSED = True
        tkb.terms = real_terms
    assert ran == [keep] * (T - 1), ran                                      # (the kernels; the stash kept or made)
    assert len(got) == len(ref)
    for k, (a, r_) in enumerate(zip(got, ref)):
        assert _rel(a, r_) < 5e-3, (k, _rel(a, r_))


# ---------------------------------------------------------------------------------------------- param_flows_mask

def _mask_case(kernels):
    """A circuit and a batch for the masked backward: the kernels' shapes (128 nodes, top_k 64), else small ones."""
    if kernels:
        T, N, C, K, r, re, b, V2 = 3, 128, 70, 64, 20, 8, 4, 80
    else:
        T, N, C, K, r, re, b, V2 = 4, 4, 5, 2, 3, 2, 6, V
    pc, _, _ = _hmm_pc(T, N, K, r, re, _tables(N, 41, V2), V2)
    return pc, _batch(b, T, N, C, K, r, re, 42, V = V2), (T, N, b)


@contextlib.contextmanager
def _paths(kernels):
    """The sums and the leaves by the kernels (each asserted to have run), or by the torch passes."""
    import pyjuice.parameterizations.lowrank.top_k_fused as tkf
    import pyjuice.parameterizations.lowrank.top_k_leaf_fused as tlf
    import pyjuice.parameterizations.lowrank.top_k_backward as tkb
    if kernels and (tkf.top_k_module() is None or tlf.module() is None or tkb.triton is None):
        pytest.skip("the CUDA modules did not build, or no Triton")
    ran = set()
    real_terms, real_leaf = tkb.terms, tlf.leaf_backward
    tkb.terms = lambda *a, **k: (ran.add("sums"), real_terms(*a, **k))[1]
    tlf.leaf_backward = lambda *a, **k: (lambda got: (ran.add("leaves") if got else None, got)[1])(real_leaf(*a, **k))
    tkf.TOP_K_FUSED = kernels
    try:
        yield
    finally:
        tkf.TOP_K_FUSED = True
        tkb.terms, tlf.leaf_backward = real_terms, real_leaf
    assert ran == ({"sums", "leaves"} if kernels else set()), ran


def _stats(pc):
    return [pc.param_flows.clone(), pc.denom_param_flows.clone(), pc.input_layer_group[0].param_flows.clone()]


@pytest.mark.parametrize("kernels", [False, True])
def test_masked_em_statistics_are_the_sub_batch(kernels):
    """`param_flows_mask` on a batch of both kinds of sample (every token observed, some missing): the EM statistics --
    the sums' F+ and F- / theta, the leaves' at observed and at missing tokens -- and `_cum_flow` are those of a backward
    of the samples it holds; the node flows and every gradient stay the whole batch's."""
    pc, (ext, ids, lp, x, miss), (T, N, b) = _mask_case(kernels)
    X, MISS = x.repeat(2, 1), torch.cat([torch.zeros_like(miss), miss])
    E = tuple(t.repeat(2, *([1] * (t.dim() - 1))) for t in ext)
    IDS, LP = ids.repeat(2, 1, 1), lp.float().repeat(2, 1, 1)
    m = torch.arange(2 * b, device = DEV) < b
    m[1], m[b], m[b + 2] = False, True, True                                    # (an observed sample out, two masked in)
    w = torch.linspace(0.5, 2.0, 2 * b, device = DEV)

    def run(sel, pmask = None):
        rows = torch.arange(2 * b, device = DEV) if sel is None else sel.nonzero().reshape(-1)
        pext = _pc_ext(tuple(t[rows] for t in E), T, N)
        evi = dict(categorical_evidence_logp = LP[rows].contiguous(), soft_evidence_cat_ids = IDS[rows].contiguous())
        g = {k_: tuple(torch.zeros_like(t) for t in v) for k_, v in pext.items()}
        glp = torch.zeros_like(evi["categorical_evidence_logp"])
        pc.init_param_flows(flows_memory = 0.0)
        pc._cum_flow = 0.0
        with torch.no_grad():
            pc(X[rows], missing_mask = MISS[rows], external_params = pext, **evi)
        pc.backward(X[rows], ll_weights = w[rows], missing_mask = MISS[rows], compute_param_flows = True,
                    logspace_flows = True, allow_modify_flows = False, external_params_grad = g,
                    categorical_evidence_logp_grad = glp, param_flows_mask = pmask, **evi)
        return _stats(pc), pc._cum_flow, pc.node_flows.clone(), [t for v in g.values() for t in v] + [glp]

    with _paths(kernels):
        got, cum, nf, grads = run(None, m)
        _, _, nf_all, grads_all = run(None)
        want, cum_want, _, _ = run(m)
    assert torch.allclose(nf, nf_all, rtol = 1e-6, atol = 1e-6)
    for a, b_ in zip(grads, grads_all):                                     # (the kernels' adds in any order)
        assert _rel(a, b_) < 1e-5, _rel(a, b_)
    for k, (a, w_) in enumerate(zip(got, want)):
        assert _rel(a, w_) < 1e-5, (k, _rel(a, w_))
    assert abs(cum - cum_want) < 1e-5 * cum_want, (cum, cum_want)


@pytest.mark.parametrize("kernels", [False, True])
def test_one_backward_for_the_gradients_and_the_em_statistics(kernels):
    """The differentiable conditional in one call (the map) with `param_flows_mask` on its all-observed pass: the
    gradients are those without the mask, and the EM statistics (the sums', their denominators, the leaves') and
    `_cum_flow` those of the EM pass on the all-observed samples, weighted by |the upstream gradient|."""
    pc, (ext, ids, lp, x, miss), (T, N, b) = _mask_case(kernels)
    pext, lp = _pc_ext(ext, T, N), lp.float()
    rows = torch.arange(b, device = DEV).repeat(2)
    wt = torch.linspace(0.5, 2.0, b, device = DEV)
    evi = dict(categorical_evidence_logp = lp, soft_evidence_cat_ids = ids)

    def conditional(pmask):
        ts = {g_: tuple(t.clone().requires_grad_(True) for t in v) for g_, v in pext.items()}
        lpg = lp.clone().requires_grad_(True)
        pc.init_param_flows(flows_memory = 0.0)
        pc._cum_flow = 0.0
        ll = pc(x.repeat(2, 1), missing_mask = torch.cat([torch.zeros_like(miss), miss]), external_params = ts,
                external_params_rows = rows, categorical_evidence_logp = lpg.repeat(2, 1, 1),
                soft_evidence_cat_ids = ids.repeat(2, 1, 1), differentiable = True,
                **({} if pmask is None else {"param_flows_mask": pmask})).view(-1)
        (-(wt * (ll[:b] - ll[b:])).sum()).backward()
        return [t.grad for v in ts.values() for t in v] + [lpg.grad], _stats(pc), pc._cum_flow

    with _paths(kernels):
        grads, got, cum = conditional(torch.arange(2 * b, device = DEV) < b)
        grads0, _, _ = conditional(None)
        pc.init_param_flows(flows_memory = 0.0)
        pc._cum_flow = 0.0
        with torch.no_grad():
            pc(x, external_params = pext, **evi)
        pc.backward(x, ll_weights = wt, compute_param_flows = True, logspace_flows = True, allow_modify_flows = False,
                    **evi)
        want, cum_want = _stats(pc), pc._cum_flow
    for name, a, b_ in zip(("A", "B", "G", "U", "D", "lp"), grads, grads0):   # (the kernels' adds in any order)
        assert _rel(a, b_) < 1e-5, (name, _rel(a, b_))
    for k, (a, w_) in enumerate(zip(got, want)):
        assert _rel(a, w_) < 1e-5, (k, _rel(a, w_))
    assert abs(cum - cum_want) < 1e-5 * cum_want, (cum, cum_want)


def test_the_kernels_together_are_the_torch_passes():
    """Every pass by the kernels -- the sums' (their counts written straight into the leaves' buffer) and the leaves' --
    against every pass by torch: the differentiable conditional in one call (the map) with the EM statistics of its
    all-observed pass (`param_flows_mask`), its gradients and statistics."""
    import pyjuice.parameterizations.lowrank.top_k_fused as tkf
    import pyjuice.parameterizations.lowrank.top_k_leaf_fused as tlf
    import pyjuice.parameterizations.lowrank.top_k_backward as tkb
    if tkf.top_k_module() is None or tlf.module() is None or tkb.triton is None:
        pytest.skip("the CUDA modules did not build, or no Triton")
    pc, (ext, ids, lp, x, miss), (T, N, b) = _mask_case(True)
    pext, lp = _pc_ext(ext, T, N), lp.float()
    rows = torch.arange(b, device = DEV).repeat(2)
    wt = torch.linspace(0.5, 2.0, b, device = DEV)

    def conditional():
        ts = {g_: tuple(t.clone().requires_grad_(True) for t in v) for g_, v in pext.items()}
        lpg = lp.clone().requires_grad_(True)
        pc.init_param_flows(flows_memory = 0.0)
        ll = pc(x.repeat(2, 1), missing_mask = torch.cat([torch.zeros_like(miss), miss]), external_params = ts,
                external_params_rows = rows, categorical_evidence_logp = lpg.repeat(2, 1, 1),
                soft_evidence_cat_ids = ids.repeat(2, 1, 1), differentiable = True,
                param_flows_mask = torch.arange(2 * b, device = DEV) < b).view(-1)
        (-(wt * (ll[:b] - ll[b:])).sum()).backward()
        return [ll.detach()] + [t.grad for v in ts.values() for t in v] + [lpg.grad] + _stats(pc)

    with _paths(False):
        ref = conditional()
    with _paths(True):
        got = conditional()
    assert torch.allclose(got[0], ref[0], atol = 1e-3), float((got[0] - ref[0]).abs().max())
    names = ("A", "B", "G", "U", "D", "lp", "sums' F+", "sums' F- / theta", "leaves' F")
    for name, a, r_ in zip(names, got[1:], ref[1:]):
        assert _rel(a, r_) < 5e-3, (name, _rel(a, r_))


@pytest.mark.parametrize("kernels", [False, True])
def test_evidence_of_the_rows_batch_is_read_through_the_map(kernels):
    """The conditional in one call with its evidence given a row each -- the rows' batch, read through
    `external_params_rows` -- is the call with it repeated a sample each: the values, every gradient (the evidence's
    folded into its rows, each sample's times its sign) and the EM statistics of the all-observed pass."""
    pc, (ext, ids, lp, x, miss), (T, N, b) = _mask_case(kernels)
    pext, lp = _pc_ext(ext, T, N), lp.float()
    rows = torch.arange(b, device = DEV).repeat(2)
    wt = torch.linspace(0.5, 2.0, b, device = DEV)

    def conditional(by_row):
        ts = {g_: tuple(t.clone().requires_grad_(True) for t in v) for g_, v in pext.items()}
        lpg = lp.clone().requires_grad_(True)
        evi = dict(categorical_evidence_logp = lpg, soft_evidence_cat_ids = ids) if by_row else \
            dict(categorical_evidence_logp = lpg.repeat(2, 1, 1), soft_evidence_cat_ids = ids.repeat(2, 1, 1))
        pc.init_param_flows(flows_memory = 0.0)
        pc._cum_flow = 0.0
        ll = pc(x.repeat(2, 1), missing_mask = torch.cat([torch.zeros_like(miss), miss]), external_params = ts,
                external_params_rows = rows, differentiable = True,
                param_flows_mask = torch.arange(2 * b, device = DEV) < b, **evi).view(-1)
        (-(wt * (ll[:b] - ll[b:])).sum()).backward()
        return [ll.detach()] + [t.grad for v in ts.values() for t in v] + [lpg.grad] + _stats(pc)

    with _paths(kernels):
        got = conditional(True)
        want = conditional(False)
    names = ("ll", "A", "B", "G", "U", "D", "lp", "sums' F+", "sums' F- / theta", "leaves' F")
    for name, a, w_ in zip(names, got, want):
        assert _rel(a, w_) < 1e-5, (name, _rel(a, w_))


def test_a_plain_soft_evidence_leaf_refuses_evidence_of_another_batch():
    """Evidence of the rows' batch is read through the map by a `LowRank(top_k)` leaf only: a plain
    `SoftEvidenceCategorical` reads it a sample each, and refuses another batch rather than read past it."""
    with juice.set_block_size(4):
        xs = [inputs(v, num_node_blocks = 1, dist = dists.SoftEvidenceCategorical(num_cats = V)) for v in range(2)]
        root = summate(multiply(*xs), num_node_blocks = 1, block_size = 1)
    pc = juice.compile(root).to(DEV)
    x = torch.randint(0, 4, (6, 2), device = DEV)
    ids = torch.arange(4, device = DEV).repeat(3, 2, 1)                          # [3, 2, 4]: the batch of 3 rows
    lp = torch.log_softmax(torch.randn(3, 2, 4, device = DEV), dim = -1)
    with pytest.raises(ValueError, match = "rows' batch"):
        pc(x, categorical_evidence_logp = lp, soft_evidence_cat_ids = ids)


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


# ---------------------------------------------------------------------------------------------- Anemone's top-down term

def _tables_of(pc, trans):
    """The circuit's transitions [N, N], emissions [N, V] and start [N], float64."""
    layer = [l for g in pc.inner_layer_groups if g.is_sum() for l in g.layers if trans[0] in l.nodes][0]
    desc = layer.parameterization
    info = [i for i in layer.external_node_infos if i.ns is trans[0]][0]
    lt = desc._dense_params(desc._geometry(layer, info, DEV), pc.params, dtype = torch.float64)
    N = lt.size(0)
    ps, pe = pc.root_ns._param_range
    return lt, pc.input_layer_group[0].params[:N * V].view(N, V).double(), pc.params[ps:pe].double()


def _em_pass(pc, chunk, T, N):
    """The EM pass of one chunk (its own missing mask); returns the per-sample inputs it ran with."""
    ext, ids, lp, x, miss = chunk
    inputs = dict(external_params = _pc_ext(ext, T, N), categorical_evidence_logp = lp.float(),
                  soft_evidence_cat_ids = ids)
    evi = {k: v for k, v in inputs.items() if k != "external_params"}
    with torch.no_grad():
        pc(x, missing_mask = miss, **inputs)
    pc.backward(x, missing_mask = miss, compute_param_flows = True, flows_memory = 1.0, logspace_flows = True,
                allow_modify_flows = False, **evi)
    return inputs


def _conditional(p, num, den):
    """The conditional M-step: theta F+ / F-, renormalized; a ratio of 1 where both are 0."""
    q = p * torch.where(den > 0, num / den.clamp_min(1e-300), torch.ones_like(den))
    return q / q.sum(-1, keepdim = True)


def _anemone_update(cur, st, s, rows):
    """One Anemone update by the definition, from the statistics `st` of `rows` rows (`_definition_stats`' layout) at
    the tables `cur`: EM on s x the rows plus 1 - s x the model they collapse to -- at each state the posterior-weighted
    mean of the rows' own transition rows (its F- row at that position, normalized; the shared row where no flow came),
    the top-down probabilities one pass through those from the start; a leaf's the mean of the rows' own emission
    probabilities (its F- row over all positions, normalized). The term's row: half that, half the shared one."""
    theta, em, pi = cur["lT"].exp(), cur["lE"].exp(), cur["lpi"].exp()
    hat = s / rows
    P, Ptot = pi.clone(), torch.zeros_like(pi)
    Tp, Tm = hat * st["Tp"], torch.zeros_like(theta)
    for t in range(st["Tm"].size(0)):
        Fm = hat * st["Tm"][t]
        f = Fm.sum(1)
        live = (f > 0)[:, None]
        safe = f.clamp_min(1e-300)[:, None]
        term = (1 - s) * P[:, None] * torch.where(live, 0.5 * (Fm / safe + theta), theta)
        Tp, Tm = Tp + term, Tm + Fm + term
        Ptot = Ptot + P
        P = P @ torch.where(live, Fm / safe, theta)
    Ptot = Ptot + P
    Fe = hat * st["Em"]
    f = Fe.sum(1)
    term = (1 - s) * Ptot[:, None] * torch.where((f > 0)[:, None], 0.5 * (Fe / f.clamp_min(1e-300)[:, None] + em), em)
    start = hat * st["pi"] + (1 - s) * pi                 # (the start: a plain sum)
    return dict(lT = _conditional(theta, Tp, Tm).log(), lE = _conditional(em, hat * st["Ep"] + term, Fe + term).log(),
                lpi = (start / start.sum()).log())


def _two_chunks(T, N, C, K, r, re):
    return [_batch(3, T, N, C, K, r, re, 21, masked = 0.0), _batch(3, T, N, C, K, r, re, 22, masked = 0.5)]


def _same(pc, trans, cur, tol = 1e-4):
    for a, b in zip(_tables_of(pc, trans), (cur["lT"].exp(), cur["lE"].exp(), cur["lpi"].exp())):
        assert _rel(a, b) < tol, _rel(a, b)


def test_anemone_top_down_term_is_the_collapsed_models():
    """One Anemone update over two chunks (every token observed; half masked) of rows that ran under external
    parameters equals the definition's: the top-down term that of the model the rows collapse to."""
    T, N, C, K, r, re, s = 4, 4, 5, 2, 3, 2, 0.3
    tab = _tables(N, 10)
    pc, trans, leaves = _hmm_pc(T, N, K, r, re, tab)
    chunks = _two_chunks(T, N, C, K, r, re)
    opt = juice.optim.Anemone(pc, step_size = s, momentum = 0.0, niters_per_update = 2, pseudocount = 0.0)
    for chunk in chunks:
        _em_pass(pc, chunk, T, N)
        opt.step()
    st = [_definition_stats(tab, *c[:4], c[4], K) for c in chunks]
    _same(pc, trans, _anemone_update(tab, {k: st[0][k] + st[1][k] for k in st[0]}, s, 6))


def test_anemone_top_down_term_under_momentum():
    """Under momentum the flows are averaged over updates and the top-down term is made from the averaged flows, at the
    update: two updates (momentum 0.5) equal the definition's -- the transitions' F- averaged as the circuit keeps it,
    W = F- / theta, and multiplied by the tables of the update."""
    T, N, C, K, r, re, s, m = 4, 4, 5, 2, 3, 2, 0.3, 0.5
    tab = _tables(N, 10)
    pc, trans, leaves = _hmm_pc(T, N, K, r, re, tab)
    opt = juice.optim.Anemone(pc, step_size = s, momentum = m, niters_per_update = 1, pseudocount = 0.0)
    cur, buf = dict(tab), None
    for k, chunk in enumerate(_two_chunks(T, N, C, K, r, re)):
        _em_pass(pc, chunk, T, N)
        opt.step()
        S = dict(_definition_stats(cur, *chunk[:4], chunk[4], K))
        S["Tm"] = S["Tm"] / cur["lT"].exp()
        buf = {key: (1 - m) * S[key] + (0.0 if buf is None else m * buf[key]) for key in S}
        avg = {key: v / (1 - m ** (k + 1)) for key, v in buf.items()}
        avg["Tm"] = avg["Tm"] * cur["lT"].exp()
        cur = _anemone_update(cur, avg, s, 3)
    _same(pc, trans, cur)


def test_anemone_without_external_params_is_the_shared_top_down_term():
    """Rows that ran without external parameters collapse to the circuit's own parameters at the sums: the term there
    is the shared `P_td theta` (the update equals the definition's with zero external tensors)."""
    T, N, C, K, r, re, s = 4, 4, 5, 2, 3, 2, 0.3
    tab = _tables(N, 10)
    pc, trans, leaves = _hmm_pc(T, N, K, r, re, tab)
    ext, ids, lp, x, miss = _batch(3, T, N, C, K, r, re, 21, masked = 0.0)
    zero = tuple(torch.zeros_like(t) for t in ext)
    opt = juice.optim.Anemone(pc, step_size = s, momentum = 0.0, pseudocount = 0.0)
    _em_pass(pc, (zero, ids, lp, x, miss), T, N)
    opt.step()
    _same(pc, trans, _anemone_update(tab, _definition_stats(tab, zero, ids, lp, x, miss, K), s, 3))


def test_anemone_keeps_the_categories_no_row_can_emit():
    """The categories that are no row's candidate -- no data in either phase -- keep their parameters (up to the row's
    normalizer) under Anemone's default pseudocount: the top-down term's shared half covers them, and the prior enters
    with the step's weight (alone, at full strength, it took every one of them to its 1 / V on each update)."""
    T, N, C, K, r, re = 4, 4, 5, 2, 3, 2
    tab = _tables(N, 10)
    pc, trans, leaves = _hmm_pc(T, N, K, r, re, tab)
    ext, _, lp, _, miss = _batch(3, T, N, C, K, r, re, 21, masked = 0.0)
    g = torch.Generator().manual_seed(31)
    ids = torch.stack([torch.stack([torch.randperm(V - 2, generator = g)[:C] for _ in range(T)]) for _ in range(3)]).to(DEV)
    x = ids.gather(2, torch.randint(0, C, (3, T), generator = g).to(DEV)[..., None])[..., 0]
    opt = juice.optim.Anemone(pc, step_size = 0.3, momentum = 0.0, pseudocount = 1e-6)
    _em_pass(pc, (ext, ids, lp, x, miss), T, N)
    opt.step()
    _, le, _ = _tables_of(pc, trans)
    e0 = tab["lE"].exp()
    kept = le[:, V - 2:] / e0[:, V - 2:]                  # (categories V - 2, V - 1: no row's candidate)
    assert _rel(kept[:, 0], kept[:, 1]) < 1e-4, (kept, "the two unseen categories were moved apart")


def test_anemone_tiny_step_moves_nothing():
    """The prior enters an Anemone update with the step's weight: at a tiny step every parameter stays put, the
    categories with little statistics included (a full-strength pseudocount moved those to the prior's value)."""
    T, N, C, K, r, re = 4, 4, 5, 2, 3, 2
    tab = _tables(N, 10)
    pc, trans, leaves = _hmm_pc(T, N, K, r, re, tab)
    chunk = _two_chunks(T, N, C, K, r, re)[0]
    before = _tables_of(pc, trans)
    opt = juice.optim.Anemone(pc, step_size = 1e-6, momentum = 0.0, pseudocount = 1e-2)
    _em_pass(pc, chunk, T, N)
    opt.step()
    for a, b in zip(_tables_of(pc, trans), before):
        assert float(((a - b).abs() / b).max()) < 1e-4, float(((a - b).abs() / b).max())
