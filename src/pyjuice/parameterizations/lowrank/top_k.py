"""
`LowRank(top_k = K)`: the per-candidate forms of the scale -- their passes in plain torch, in `PASSES_DTYPE`: float32
where the kernels do not serve yet (the default), float64 the REFERENCE the kernels are checked against.

CANDIDATES are those of a `SoftEvidenceCategorical` variable, `soft_evidence_cat_ids[b, v]` in the caller's order, with
their evidence `lp_b(c)` (`categorical_evidence_logp`) beside them; the k-th has the SLOT min(k, K): one of its own for
each of the first K, one shared by all the others.

On a LEAF of variable v (the emission adjustment), node i, the tensors `A [B, M, r]`, `B [B, r, K + 1]`:

    log P_b(c | i) = log theta[i, c] + lp_b(c) + A_b[i] . B_b[:, slot(c)] - log sum_c' (the same)      over the candidates

and the node's value is log P_b(x | i) for the observed token x (-inf if x is not a candidate), 0 where x is missing.

On a SUM node multiplied with node i of such a leaf (the token term), child j, the tensors `A [B, M, r]`, `B [B, r, N]`,
`G [B, K + 1, N]`, the token x the leaf emits:

    log theta_b[i, j | s] = log theta[i, j] + A_b[i] . B_b[:, j] + G_b[s, j] - log Z_b[i, s]          over the children

and the node's value is log sum_j theta_b[i, j | slot(x)] c_b[j] for an observed x; for a missing one the token is
summed out under the leaf's own probabilities of the slots, log sum_s P_b(s | i) sum_j theta_b[i, j | s] c_b[j], P_b(s | i)
those of the leaf's first K candidates one by one and the rest summed. The leaf's value is 1 there, so the product of
the two is exactly the circuit with a state for every (node, slot).

The sum's values and its backward are made from the shifted exponentials' products (`_products`), as the kernels will
make them: the per-(node, slot) normalizers `Z = W E^T` and the evidence's `U = W (E * c)^T`, `W = exp(log theta + A B -
max)`, `E = exp(G - max)`; the backward in closed form (`_sum_backward_terms`).
"""

from __future__ import annotations

import torch
from typing import Optional

#: the torch passes' dtype: float32 (fast: the default where no kernel serves) or float64 (the reference: tests, checks)
PASSES_DTYPE = torch.float32

#: A leaf layer's record of the torch forward's tokens: `dict(B, place = {id(node vector): [B] long})`, each sample's
#: token's place among its candidates, -1 none of them, -2 missing
TOKENS = "_lr_top_k_tokens"


def _d(t: torch.Tensor) -> torch.Tensor:
    return t.to(PASSES_DTYPE)


_KW = {}


def _kw(name: str) -> str:
    """The circuit's option `parameterized_sum_layer.<name>` (imported when first asked for: an import a call costs ~1 us
    of a position's host time)"""
    got = _KW.get(name)
    if got is None:
        import pyjuice.layer.parameterized_sum_layer as psl
        got = _KW[name] = getattr(psl, name)
    return got


def _staged(kwargs):
    return kwargs.get(_kw("EXTERNAL_PARAMS_KWARG"))


def external_rows(kwargs):
    """`[B]` long: the row of the external tensors each sample reads (`external_params_rows`), or None (its own)."""
    return kwargs.get(_kw("EXTERNAL_PARAMS_ROWS_KWARG"))


def per_sample(tensors, rows):
    """The external tensors' rows the samples read: themselves, or gathered through the map `rows`."""
    return tensors if rows is None else tuple(t.index_select(0, rows) for t in tensors)


def evidence_rows(kwargs):
    """`[B]` long: the row of the leaves' evidence (`categorical_evidence_logp`, `soft_evidence_cat_ids`) each sample
    reads where the evidence has the rows' batch -- that of the external tensors, through `external_params_rows`:
    samples sharing a row share its evidence --, or None (the samples' batch: a sample's own)."""
    lp, rows = kwargs.get("categorical_evidence_logp"), external_rows(kwargs)
    if lp is None or rows is None or lp.size(0) == rows.numel():
        return None
    return rows


def per_sample_evidence(kwargs):
    """`(lp, ids)` a sample each: the evidence as given, or its rows gathered (`evidence_rows`)."""
    lp, ids, er = kwargs.get("categorical_evidence_logp"), kwargs.get("soft_evidence_cat_ids"), evidence_rows(kwargs)
    if er is None or lp is None or ids is None:
        return lp, ids
    return lp.index_select(0, er), ids.index_select(0, er)


def add_evidence_grad(glp, lvid: int, values, kwargs, samples = None) -> None:
    """`glp[:, lvid, :]` += the samples' `values [b, C]` (the samples `samples`, or every one): each into its own
    row, or (evidence of the rows' batch, `evidence_rows`) into the row it read, times its sign where the backward folds
    the gradients (`EXTERNAL_PARAMS_GRAD_SIGNS_KWARG`)."""
    from pyjuice.layer.parameterized_sum_layer import EXTERNAL_PARAMS_GRAD_SIGNS_KWARG
    er = evidence_rows(kwargs)
    if er is None:
        if samples is None:
            glp[:, lvid, :] += values.to(glp.dtype)
        else:
            glp[samples, lvid, :] += values.to(glp.dtype)
        return
    sg = kwargs.get(EXTERNAL_PARAMS_GRAD_SIGNS_KWARG)
    if samples is not None:
        er, sg = er[samples], (None if sg is None else sg[samples])
    v = values if sg is None else values * sg.to(values.dtype)[:, None]
    glp[:, lvid, :].index_add_(0, er, v.to(glp.dtype))


def _leaf_rows(layer, ns):
    """`(first node row, end, variable, its layer-local id, the nodes' first parameter ids [M])` of a soft-evidence
    leaf node vector of input layer `layer`."""
    s, e = ns._output_ind_range
    l0 = layer._output_ind_range[0]
    v = ns.scope.to_list()[0]
    return s, e, v, int(layer.var_idmapping[v]), layer.s_pids[s - l0:e - l0]


def _missing(missing_mask, v: int, B: int, device, batch_first: bool = True) -> torch.Tensor:
    """`[B]` bool: variable `v` missing in each sample (`missing_mask` as the circuit takes it: `[num_vars]`, `[B,
    num_vars]`, or `[num_vars, B]` where not batch first)."""
    if missing_mask is None:
        return torch.zeros(B, dtype = torch.bool, device = device)
    if missing_mask.dim() == 1:
        return missing_mask[v].to(device = device, dtype = torch.bool).expand(B)
    return (missing_mask[:, v] if batch_first else missing_mask[v, :]).to(device = device, dtype = torch.bool)


def _candidate_place(ids_v: torch.Tensor, x: torch.Tensor):
    """`(place [B], found [B])`: where the token `x` first is among the candidates `ids_v [B, C]`."""
    hit = ids_v.long() == x.long()[:, None]
    return hit.int().argmax(dim = 1), hit.any(dim = 1)


def leaf_log_probs(layer, ns, lp_v: torch.Tensor, ids_v: torch.Tensor, tilt = None, K: Optional[int] = None) -> torch.Tensor:
    """`[B, M, C]` (`PASSES_DTYPE`): log P_b(c | i) of the leaf node vector `ns` over its candidates `ids_v [B, C]`, the evidence
    `lp_v [B, C]`; `tilt` its tensors `(A [B, M, r], B [B, r, K + 1])` (slots capped at `K`), or None (the plain leaf)."""
    _, _, _, _, spids = _leaf_rows(layer, ns)
    C = ids_v.size(1)
    lt = _d(layer.params[spids[None, :, None] + ids_v.long()[:, None, :]]).log()             # [B, M, C]
    lt = lt + _d(lp_v)[:, None, :]
    if tilt is not None:
        tau = torch.bmm(_d(tilt[0]), _d(tilt[1]))                                              # [B, M, K + 1]
        slot = torch.arange(C, device = lt.device).clamp(max = K)
        lt = lt + tau[:, :, slot]
    z = torch.logsumexp(lt, dim = 2, keepdim = True)
    return lt - z.masked_fill(z == -float("inf"), 0.0)           # (no candidate with mass: every log-probability -inf)


def leaf_forward(desc, layer, data: torch.Tensor, node_mars: torch.Tensor, kwargs: dict) -> None:
    """The values of the input layer's node vectors that were given tensors, written over the shared forward's (the
    layer's missing mask applies after). `data` `[num_vars, B]`, `kwargs` the evidence as the caller gave it."""
    from . import top_k_leaf_fused
    layer.__dict__.pop(TOKENS, None)
    if top_k_leaf_fused.leaf_forward(desc, layer, data, node_mars, kwargs):
        return None
    staged = _staged(kwargs)
    if not staged:
        return None
    lp, ids = per_sample_evidence(kwargs)
    places = {}
    for ns in layer.nodes:
        tilt = staged.get(ns)
        if tilt is None:
            continue
        tilt = per_sample(tilt, external_rows(kwargs))
        assert ids is not None and lp is not None, \
            "`LowRank(top_k)` on a soft-evidence leaf needs `soft_evidence_cat_ids` and `categorical_evidence_logp`."
        s, e, v, lvid, _ = _leaf_rows(layer, ns)
        logp = leaf_log_probs(layer, ns, lp[:, lvid, :], ids[:, lvid, :], tilt, desc.top_k)    # [B, M, C]
        place, found = _candidate_place(ids[:, lvid, :], data[v, :])
        val = logp.gather(2, place[:, None, None].expand(-1, logp.size(1), 1))[:, :, 0]         # [B, M]
        val = torch.where(found[:, None], val, torch.full_like(val, -float("inf")))
        node_mars[s:e, :] = val.t().to(node_mars.dtype)
        miss = _missing(kwargs.get("missing_mask"), v, val.size(0), val.device, kwargs.get("_batch_first", True))
        places[id(ns)] = torch.where(miss, -2, torch.where(found, place, -1))
    # (the tokens' places, for a conditional draw on this forward: `top_k_queries.forward_places`; the kernels' forward
    # keeps them in its state)
    layer.__dict__[TOKENS] = dict(B = node_mars.size(1), place = places)
    return None


def slot_parts(desc, sib: dict, batch_size: int, kwargs: dict, device):
    """`(slot [B], rows, logw_rows, logp, tilt)`: what the token's slots weigh for each node of a `LowRank(top_k)` sum
    node vector, `sib` its leaf (`_sibling_leaves`) -- the observed token's slot in each sample (its first place among
    the candidates, capped at K; 0 where it is none of them: the leaf's value is -inf there); the samples `rows` whose
    token is missing, and there the slots' log-weights `[b, M, K + 1]` (the leaf's own log P_b(s | i), its first K
    candidates one by one and the rest summed), the leaf's log P_b(c | i) `[b, M, C]` and its tilt's rows (None
    without one); `PASSES_DTYPE` (None where no token is missing). Where the leaf's layer ran by the kernels, the slots
    and the weights its forward made (`top_k_leaf_fused.slot_rows`), and no log P_b(c | i) or tilt."""
    from . import top_k_leaf_fused
    made = top_k_leaf_fused.slot_rows(sib, desc.top_k, batch_size)
    if made is not None:
        return (*made, None, None)
    leaf_layer, xs, v = sib["layer"], sib["ns"], sib["var"]
    s, e, _, lvid, _ = _leaf_rows(leaf_layer, xs)
    lp, ids = per_sample_evidence(kwargs)
    assert ids is not None and lp is not None, \
        "A `LowRank(top_k)` sum reads its leaf's candidates: `soft_evidence_cat_ids` and `categorical_evidence_logp`."
    K, M = desc.top_k, e - s
    lp_v, ids_v = lp[:, lvid, :], ids[:, lvid, :]
    C = ids_v.size(1)

    place, _ = _candidate_place(ids_v, kwargs["_inputs"][:, v])
    miss = _missing(kwargs.get("missing_mask"), v, batch_size, device, kwargs.get("_batch_first", True))
    rows, logw_rows, logp, tilt = torch.nonzero(miss).reshape(-1), None, None, None
    if rows.numel() > 0:
        staged = _staged(kwargs)
        tilt = None if staged is None else staged.get(xs)
        if tilt is not None:
            ext = external_rows(kwargs)
            er = rows if ext is None else ext[rows]                            # (the rows of the tensors they read)
            tilt = (_d(tilt[0][er]), _d(tilt[1][er]))
        logp = leaf_log_probs(leaf_layer, xs, lp_v[rows], ids_v[rows], tilt,
                              None if tilt is None else xs.parameterization.top_k)                # [b, M, C]
        own = logp[:, :, :min(K, C)]
        if C < K:
            own = torch.cat([own, own.new_full((own.size(0), M, K - C), -float("inf"))], dim = 2)
        rest = torch.logsumexp(logp[:, :, K:], dim = 2, keepdim = True) if C > K else \
            logp.new_full((logp.size(0), M, 1), -float("inf"))
        logw_rows = torch.cat([own, rest], dim = 2)
    return place.clamp(max = K), rows, logw_rows, logp, tilt


def slot_weights(slot, rows, logw_rows, batch_size: int, M: int, K: int) -> torch.Tensor:
    """`logw [B, M, K + 1]` (`PASSES_DTYPE`) from `slot_parts`' pieces: 0 at the observed token's slot and -inf
    elsewhere; at the samples `rows` whose token is missing, their slots' log-weights `logw_rows`."""
    logw = torch.full((batch_size, M, K + 1), -float("inf"), dtype = PASSES_DTYPE, device = slot.device)
    logw.scatter_(2, slot[:, None, None].expand(-1, M, 1), 0.0)
    if rows.numel() > 0:
        logw[rows] = logw_rows.to(PASSES_DTYPE)
    return logw


def slot_log_weights(desc, sib: dict, batch_size: int, kwargs: dict, device):
    """`(logw [B, M, K + 1], rows, logp, tilt)` (`PASSES_DTYPE`): the log-weights of the token's slots for each node of a
    `LowRank(top_k)` sum node vector (`slot_parts`, `slot_weights`) -- 0 at the observed token's slot and -inf
    elsewhere; where the token is missing the leaf's own log P_b(s | i) --; the samples `rows` whose token is missing,
    and there the leaf's log P_b(c | i) `[b, M, C]` and its tilt's rows (None without one)."""
    slot, rows, logw_rows, logp, tilt = slot_parts(desc, sib, batch_size, kwargs, device)
    s, e, _, _, _ = _leaf_rows(sib["layer"], sib["ns"])
    return slot_weights(slot, rows, logw_rows, batch_size, e - s, desc.top_k), rows, logp, tilt


def _finite_or_zero(t: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t)).detach()


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a / b` where `b > 0`, 0 elsewhere."""
    pos = b > 0
    return torch.where(pos, a / torch.where(pos, b, torch.ones_like(b)), torch.zeros_like(a))


def _products(A, Bm, G, ltheta, lc):
    """The shifted exponentials and their products (`PASSES_DTYPE`): `W [B, M, N]` = exp(log theta + A B) (a row's largest 1),
    `E [B, S, N]` = exp(G) (a slot's largest 1), `ev [B, N]` = the children's values over their largest, `c [B, 1]` that
    largest's log; `Z [B, M, S]` = W E^T (the per-(node, slot) normalizers) and `U` = W (E ev)^T, each up to its row's
    and slot's shifts -- which every ratio below cancels."""
    lw = torch.bmm(A, Bm) + ltheta[None]
    W = (lw - _finite_or_zero(lw.amax(dim = 2, keepdim = True))).exp()
    E = (G - _finite_or_zero(G.amax(dim = 2, keepdim = True))).exp()
    c = _finite_or_zero(lc.amax(dim = 1, keepdim = True))
    ev = (lc - c).exp()
    Z = torch.bmm(W, E.transpose(1, 2))
    U = torch.bmm(W, (E * ev[:, None, :]).transpose(1, 2))
    return W, E, ev, c, Z, U


def _sum_values(A, Bm, G, ltheta, lc, logw) -> torch.Tensor:
    """`[B, M]` (`PASSES_DTYPE`): log sum_s w_b[i, s] sum_j theta_b[i, j | s] c_b[j], from `A [B, M, r]`, `B [B, r, N]`, `G [B,
    S, N]`, `log theta [M, N]`, the children's log-values `lc [B, N]` and the slots' log-weights `logw [B, M, S]`: -inf
    where no term is positive."""
    W, E, ev, c, Z, U = _products(A, Bm, G, ltheta, lc)
    R = (logw.exp() * _safe_div(U, Z)).sum(dim = 2)                                 # [B, M]
    val = torch.where(R > 0, R, torch.ones_like(R)).log() + c
    none = (R <= 0) | ~torch.isfinite(lc.amax(dim = 1, keepdim = True))
    return torch.where(none, torch.full_like(val, -float("inf")), val)


def _sum_backward_terms(A, Bm, G, ltheta, lc, logw, f, pm = None):
    """The backward of one `LowRank(top_k)` node vector (`PASSES_DTYPE`). With `pi_b[i, s]` the posterior of slot s given node
    i (`w r / sum_s w r`, r = U / Z) and `a = f pi` the flow into each (node, slot), the counts along the edges and
    their normalizers' -- q[i, s, j] = a W E c / U, t[i, s, j] = a W E / Z -- make everything (F+, F- over the samples
    `pm` [B] holds, if given):

        ef [B, N]     = sum_i sum_s q                 the element flows (d / d log c)
        Fp, Fm [M, N] = sum_b sum_s q,  sum_b sum_s t   F+ and F- (d / d log theta in the values' / normalizers' terms)
        Gl [B, M, N]  = sum_s (q - t)                 d / d (A B): dA = Gl B^T, dB = A^T Gl
        dG [B, S, N]  = sum_i (q - t)
        a  [B, M, S]                                  d / d logw: where the token is missing, the leaf's posterior
                                                      counts of its slots"""
    W, E, ev, c, Z, U = _products(A, Bm, G, ltheta, lc)
    wr = logw.exp() * _safe_div(U, Z)                                               # [B, M, S]
    a = f[:, :, None] * _safe_div(wr, wr.sum(dim = 2, keepdim = True))
    alpha, beta = _safe_div(a, U), _safe_div(a, Z)
    Q = W * torch.bmm(alpha, E) * ev[:, None, :]                                    # sum_s q
    Tm = W * torch.bmm(beta, E)                                                     # sum_s t
    dG = E * (torch.bmm(alpha.transpose(1, 2), W) * ev[:, None, :] - torch.bmm(beta.transpose(1, 2), W))
    keep = 1.0 if pm is None else pm[:, None, None].to(Q.dtype)
    return Q.sum(dim = 1), (Q * keep).sum(dim = 0), (Tm * keep).sum(dim = 0), Q - Tm, dG, a


def sum_forward(desc, layer, ns_tensors, node_mars, element_mars, params, kwargs) -> None:
    """The node values of a `LowRank(top_k)` sum layer, written over its rows of `node_mars`: by the pass's stack of the
    sums reading the leaf beside it where it takes the node vector (`top_k_stack.py`), else by the fused kernels where
    they serve (`top_k_fused.py`), else here. A layer whose every node vector a stack took is settled
    (`top_k_stack.settle`: its next calls straight to the stack)."""
    top_k_fused, top_k_stack = _kernel_modules()[2], _stack_module()
    sibs = layer.__dict__.get("_sibling_leaves")
    assert sibs is not None, "A `LowRank(top_k)` sum layer compiled without its leaves (`_resolve_top_k_siblings`)."
    assert "_inputs" in kwargs, "A `LowRank(top_k)` sum layer reads the tokens; the circuit's forward passes them."
    B = node_mars.size(1)
    layer.__dict__["_lr_state"] = dict()
    layer.__dict__["_lr_topk_batch"] = B
    stacked = []
    for ns_info, tensors in ns_tensors:
        sib = sibs[id(ns_info.ns)]
        stack = top_k_stack.stack_of(sib, B, kwargs, node_mars, element_mars, params)
        if stack is not None and stack.forward(ns_info.ns):
            stacked.append((ns_info.ns, sib))
            continue
        geo = desc._geometry(layer, ns_info, node_mars.device)
        if top_k_fused.sum_forward(desc, layer, ns_info, tensors, geo, params, element_mars, node_mars, sib, kwargs):
            continue
        A, Bm, G = (_d(t) for t in per_sample(tensors, external_rows(kwargs)))
        ltheta = desc._dense_params(geo, params, dtype = PASSES_DTYPE).log()        # [M, N]
        lc = _d(desc._child_values(geo, element_mars))                              # [B, N]
        logw = slot_log_weights(desc, sibs[id(ns_info.ns)], A.size(0), kwargs, node_mars.device)[0]
        ln = _sum_values(A, Bm, G, ltheta, lc, logw)
        node_mars[ns_info.nid_start:ns_info.nid_end, :] = ln.t().to(node_mars.dtype)
    if stacked and len(stacked) == len(ns_tensors):
        top_k_stack.settle(layer, stacked)
    return None


def _grads_kw(kwargs):
    gviews = kwargs.get(_kw("EXTERNAL_PARAMS_GRAD_KWARG"))
    return gviews, getattr(gviews, "fresh", False)


def _store(gviews, fresh, grads, values, rows = None) -> None:
    """Add `values` -- a sample each: every sample, or the samples `rows` -- into the gradient destinations `grads`;
    the destinations' first writer of the pass initializes them where they were handed over unzeroed (`fresh`). Where
    the destinations fold (`StagedExternalParams.fold`: the rows of the map, the samples' signs) each sample's value
    goes into the row it read, times its sign."""
    if fresh and gviews.claim(grads):
        for g in grads:
            g.zero_()
    fold = getattr(gviews, "fold", None)
    if fold is not None:
        into, sign = fold if rows is None else (fold[0][rows], fold[1][rows])
    for g, v in zip(grads, values):
        if fold is not None:
            g.index_add_(0, into, (v * sign.view(-1, *([1] * (v.dim() - 1))).to(v.dtype)).to(g.dtype))
        elif rows is None:
            g.add_(v.to(g.dtype))
        else:
            g.index_add_(0, rows, v.to(g.dtype))


_MODS = {}


def _kernel_modules():
    """`(top_k_leaf_fused, top_k_backward, top_k_fused)` (imported when first asked for)"""
    got = _MODS.get("kernels")
    if got is None:
        from . import top_k_leaf_fused, top_k_backward, top_k_fused
        got = _MODS["kernels"] = (top_k_leaf_fused, top_k_backward, top_k_fused)
    return got


def _stack_module():
    """`top_k_stack` (imported when first asked for)"""
    got = _MODS.get("stack")
    if got is None:
        from . import top_k_stack as got
        _MODS["stack"] = got
    return got


def sum_backward(desc, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows, node_mars, element_mars, params,
                 param_flows, denom_param_flows, kwargs) -> None:
    """
    The backward of a `LowRank(top_k)` sum layer (`_sum_backward_terms`), from the node flows `f_b[i]`: the element
    flows; F+ into the parameter flows and W = F- / theta into the denominator flows (with `apply_z_correction`); the
    gradients of `A`, `B`, `G`; and where the leaf's token is missing, what reaches the leaf through its probabilities of
    the slots -- with rho[i, c] = d / d log P_b(c | i) (the posterior counts: a slot's own candidate gets the slot's,
    the shared slot's spread over its candidates as P_b(c | i)), d / d logit(c) = rho - f P_b(c | i): the evidence's
    gradient (summed over i), the tilt's, and the leaf's EM statistics there, F+ += rho, F- += f P_b(c | i).
    """
    top_k_leaf_fused, top_k_backward, top_k_fused = _kernel_modules()
    dev = node_flows.device
    sibs = layer.__dict__["_sibling_leaves"]
    gviews, fresh = _grads_kw(kwargs)
    want_flows = param_flows is not None
    want_denom = want_flows and denom_param_flows is not None and desc.apply_z_correction
    pm = kwargs.get("param_flows_mask") if want_flows else None          # (the samples the EM statistics take)
    glp = kwargs.get("categorical_evidence_logp_grad")

    geos = [desc._geometry(layer, ns_info, dev) for ns_info, _ in ns_tensors]
    # the children's element flows: where no child is reached by two of the layer's node vectors (`_kids_own`), each
    # one's rows written by its node vector alone (the kernels' scatter stores them: no -inf fill first, a launch a
    # position); else filled here and each node vector's added in
    alone = desc._kids_own(layer, dev)
    if not alone:
        for geo in geos:
            element_flows.index_fill_(0, geo["kids"], -float("inf"))
    ef_kernel = element_flows.dtype == torch.float32 and element_flows.stride(1) == 1
    # (a backward asking for no gradients nor EM statistics -- a query's -- of a node vector whose forward a stack took:
    # by the stack, one call; `top_k_stack`)
    plain = not want_flows and ef_kernel and node_flows.dtype == torch.float32 and node_flows.stride(1) == 1
    B = node_flows.size(1)

    for (ns_info, tensors), grads, geo in zip(ns_tensors, ns_grad_tensors, geos):
        sib = sibs[id(ns_info.ns)]
        if plain and grads is None:
            stack = _stack_module().stack_made(sib, B)
            if stack is not None and stack.backward(ns_info.ns, node_flows, element_flows, alone):
                continue
        nf = node_flows[ns_info.nid_start:ns_info.nid_end, :]                        # [M, B] (log-space flows)
        f = None                                                                      # (f [B, M]: where torch needs it)
        kernels = top_k_backward.serves(desc, geo["M"], geo["N"], tensors[0].size(2), B, dev, kwargs, want_flows) \
            and element_mars.stride(1) == 1 and nf.dtype == torch.float32 and nf.stride(1) == 1
        # (the leaf's parts as its forward state holds them, made once a pass -- the forward's --: one look-up)
        cache = top_k_leaf_fused.pass_cache(sib, B) if kernels else None
        parts = None if cache is None else \
            top_k_leaf_fused.sum_parts(sib, desc.top_k, B, external_rows(kwargs), cache)
        if parts is not None:
            slot, rows, logw_rows, _, Rm, t = parts
            logp = tilt = None
        else:
            slot, rows, logw_rows, logp, tilt = slot_parts(desc, sib, B, kwargs, dev)
            Rm, t = rows.numel(), None
        # (the parameters: the kernels' fp32 grid and its tables, once a pass -- the forward's --; torch's in PASSES_DTYPE)
        tab = top_k_fused.theta_tables(desc, sib, ns_info.ns, geo, params, B, cache) if kernels else None
        theta = tab["theta"] if kernels else desc._dense_params(geo, params, dtype = PASSES_DTYPE)
        pf0 = ns_info.ns._param_flow_range[0] if want_flows else None
        d0 = desc._denom_offsets(layer)[id(ns_info.ns)][0] if want_denom else None
        em_out = None
        if kernels:
            # (the gradients added into their destinations by the kernels; the counts into the leaf's buffer; the EM
            # statistics into a pass's accumulator of the flow block, where there is one: scattered at the pass's end;
            # the element flows by the same call where they can)
            counts = None if Rm == 0 else (top_k_leaf_fused.counts_buffer(sib, desc.top_k, B) if t is None else
                                           top_k_leaf_fused.counts_rows(sib, desc.top_k, B, t))
            if want_flows:
                em_out = top_k_backward.em_accumulator(geo, params, param_flows, pf0,
                                                       denom_param_flows if want_denom else None, d0, dev)
            ef, a_rows, flows = top_k_backward.terms(*tensors, tab, element_mars, geo, slot, rows, logw_rows, nf,
                                                     external_rows(kwargs), grads, gviews, fresh, want_flows, pm, counts,
                                                     top_k_fused.kept_of(layer, ns_info.ns, B), em_out = em_out,
                                                     element_flows = element_flows if ef_kernel else None,
                                                     store = alone)
            Fp, Fm = flows if want_flows else (None, None)
        else:
            f = _d(nf.t()).exp()
            A, Bm, G = (_d(t) for t in per_sample(tensors, external_rows(kwargs)))
            lc = _d(desc._child_values(geo, element_mars))
            logw = slot_weights(slot, rows, logw_rows, B, geo["M"], desc.top_k)
            ef, Fp, Fm, Gl, dG, a = _sum_backward_terms(A, Bm, G, theta.log(), lc, logw, f, pm)
            if grads is not None:
                _store(gviews, fresh, grads, [torch.bmm(Gl, Bm.transpose(1, 2)), torch.bmm(A.transpose(1, 2), Gl), dG])
            a_rows = a[rows]

        kids = geo["kids"]
        if not (kernels and ef_kernel):                                # (else written by `terms`' call)
            if alone:
                element_flows.index_fill_(0, kids, -float("inf"))
            element_flows[kids, :] = torch.logaddexp(element_flows[kids, :],
                                                     ef.log().t()[geo["cols"], :].to(element_flows.dtype))
        if want_flows and em_out is None:
            if not (kernels and top_k_backward.scatter_statistics(geo, Fp, Fm, params, param_flows, pf0,
                                                                    denom_param_flows if want_denom else None, d0)):
                desc._scatter_add(geo, param_flows, pf0, Fp.float())
                if want_denom:
                    desc._scatter_add(geo, denom_param_flows, d0, _safe_div(Fm, theta).float())
        if Rm > 0 and a_rows is not None and not top_k_leaf_fused.add_counts(sib, desc.top_k, B, a_rows):
            records = kwargs.get(_kw("QUERY_RECORDS_KWARG"))           # (a conditional query's: the leaf's read-out)
            if records is not None:
                records[("top_k_slot_counts", id(sib["ns"]))] = (rows, a_rows)
            f = _d(nf.t()).exp() if f is None else f
            _leaf_terms(desc, sib, rows, logp, tilt, logw_rows, a_rows, f[rows], kwargs, glp, gviews, fresh, want_flows,
                        pm)
    return None


def _leaf_terms(desc, sib, rows, logp, tilt, logw, a, f, kwargs, glp, gviews, fresh, want_flows, pm = None) -> None:
    """What a `LowRank(top_k)` sum hands its leaf at the samples `rows` whose token is missing (`sum_backward`): `logp`
    the leaf's log P_b(c | i) there, `logw` its slots', `a` the posterior counts of the slots, `f` the node flows; the
    EM statistics of the samples `pm` [B] holds, if given."""
    leaf_layer, xs = sib["layer"], sib["ns"]
    s, e, _, lvid, _ = _leaf_rows(leaf_layer, xs)
    K = desc.top_k
    b, M, C = logp.shape
    k = torch.arange(C, device = logp.device)
    # rho: a slot's own candidate takes the slot's count; the shared slot's is spread as P_b(c | i) / P_b(rest | i)
    rho = torch.where((k < K)[None, None, :], a[:, :, k.clamp(max = K)],
                      a[:, :, K:K + 1] * (logp - logw[:, :, K:K + 1]).exp().nan_to_num(0.0))
    P = logp.exp()
    dlogit = rho - f[:, :, None] * P                                                  # [b, M, C]
    if glp is not None:
        add_evidence_grad(glp, lvid, dlogit.sum(dim = 1), kwargs, samples = rows)
    leaf_grads = gviews.get(xs) if gviews is not None else None
    if tilt is not None and leaf_grads is not None:
        Kl = xs.parameterization.top_k
        dtau = torch.zeros(b, M, Kl + 1, dtype = PASSES_DTYPE, device = logp.device).index_add_(2, k.clamp(max = Kl), dlogit)
        _store(gviews, fresh, leaf_grads, [torch.bmm(dtau, tilt[1].transpose(1, 2)), torch.bmm(tilt[0].transpose(1, 2), dtau)],
               rows = rows)
    if want_flows and leaf_layer.param_flows is not None:
        pf, V = leaf_layer.param_flows, xs.dist.num_cats
        l0 = leaf_layer._output_ind_range[0]
        er = evidence_rows(kwargs)
        ids = kwargs["soft_evidence_cat_ids"][rows if er is None else er[rows], lvid, :]
        idx = leaf_layer.s_pfids[s - l0:e - l0][None, :, None] + ids.long()[:, None, :]  # [b, M, C]
        keep = 1.0 if pm is None else pm[rows][:, None, None].to(rho.dtype)
        pf.index_add_(0, idx.reshape(-1), (rho * keep).reshape(-1).to(pf.dtype))        # F+: the posterior counts
        pf.index_add_(0, (idx + V).reshape(-1), (f[:, :, None] * P * keep).reshape(-1).to(pf.dtype))   # F-


def leaf_backward(desc, layer, data, node_flows, kwargs, logspace_flows: bool, missing_mask, batch_first: bool,
                  compute_param_flows: bool) -> bool:
    """
    The backward of a soft-evidence input layer whose node vectors were all given tensors; False where none were (the
    plain leaf's backward then). At an observed token x, with the node flow `f_b[i]` and P = P_b(. | i):

        d / d logit_b(i, c) = f_b[i] (1[c = x] - P(c))     -> the evidence's gradient (summed over i) and the tilt's
        F+[i, x] += f_b[i]          F-[i, c] += f_b[i] P(c)          (the dual-flow statistics: Anemone's M-step)

    A missing token's leaf is 1 whatever its parameters: no gradient here. Its EM statistics are its posterior counts
    in both phases: where a `LowRank(top_k)` sum beside it reads its token, the sum's (`_leaf_terms`); else the token's
    posterior is the leaf's own P, so f P (the counts expected under the sample's model, not under the shared
    parameters: with every token missing, the pass gives exactly what Anemone's step is anchored to).
    """
    staged = _staged(kwargs)
    supplied = [ns for ns in layer.nodes if staged is not None and ns in staged]
    if not supplied:
        return False
    assert len(supplied) == len(layer.nodes), \
        "`LowRank` on input nodes: supply every node vector of the layer (a registered group), or none."
    assert layer.dist._dual_flow_backward, "`LowRank` on a soft-evidence leaf: its dual-flow statistics (the default)."
    from . import top_k_leaf_fused
    if top_k_leaf_fused.leaf_backward(desc, layer, data, node_flows, kwargs, logspace_flows, missing_mask, batch_first,
                                      compute_param_flows):
        return True
    gviews, fresh = _grads_kw(kwargs)
    lp, ids = per_sample_evidence(kwargs)
    glp = kwargs.get("categorical_evidence_logp_grad")
    pf = layer.param_flows if compute_param_flows else None
    pm = kwargs.get("param_flows_mask") if pf is not None else None     # (the samples the EM statistics take)
    dev = node_flows.device
    l0 = layer._output_ind_range[0]
    K = desc.top_k
    for ns in layer.nodes:
        tilt = per_sample(staged[ns], external_rows(kwargs))
        s, e, v, lvid, _ = _leaf_rows(layer, ns)
        f = _d(node_flows[s:e, :].t())
        f = f.exp() if logspace_flows else f
        B, M = f.shape
        miss = _missing(missing_mask, v, B, dev, batch_first)
        lp_v, ids_v = lp[:, lvid, :], ids[:, lvid, :]
        C = ids_v.size(1)
        P = leaf_log_probs(layer, ns, lp_v, ids_v, tilt, K).exp()                       # [B, M, C]
        fe = f if pm is None else f * _d(pm[:, None])                                 # [B, M]: the EM statistics'
        if pf is not None and bool(miss.any()) and id(ns) not in layer.__dict__.get("_read_by_sums", ()):
            V = ns.dist.num_cats
            idx = layer.s_pfids[s - l0:e - l0][None, :, None] + ids_v.long()[:, None, :]  # [B, M, C]
            fP = ((fe * _d(miss[:, None]))[:, :, None] * P).reshape(-1).to(pf.dtype)      # the posterior counts f P
            pf.index_add_(0, idx.reshape(-1), fP)                                       # F+
            pf.index_add_(0, (idx + V).reshape(-1), fP)                                 # F-
        place, found = _candidate_place(ids_v, data[v, :])
        fo = f * _d(((~miss) & found)[:, None])                                       # [B, M]
        onehot = torch.zeros(B, C, dtype = PASSES_DTYPE, device = dev).scatter_(1, place[:, None], 1.0)
        dlogit = fo[:, :, None] * (onehot[:, None, :] - P)                             # [B, M, C]
        if glp is not None:
            add_evidence_grad(glp, lvid, dlogit.sum(dim = 1), kwargs)
        grads = gviews.get(ns) if gviews is not None else None
        if grads is not None:
            slot = torch.arange(C, device = dev).clamp(max = K)
            dtau = torch.zeros(B, M, K + 1, dtype = PASSES_DTYPE, device = dev).index_add_(2, slot, dlogit)
            _store(gviews, fresh, grads, [torch.bmm(dtau, _d(tilt[1]).transpose(1, 2)),
                                          torch.bmm(_d(tilt[0]).transpose(1, 2), dtau)])
        if pf is not None:
            V = ns.dist.num_cats
            spf = layer.s_pfids[s - l0:e - l0]                                          # [M]
            foe = fo if pm is None else fo * _d(pm[:, None])
            keep = foe > 0
            idx = (spf[None, :] + data[v, :].long()[:, None])                           # [B, M]: F+ at the token
            pf.index_add_(0, idx[keep], foe[keep].to(pf.dtype))
            idx2 = spf[None, :, None] + V + ids_v.long()[:, None, :]                     # [B, M, C]: F- over the candidates
            pf.index_add_(0, idx2.reshape(-1), (foe[:, :, None] * P).reshape(-1).to(pf.dtype))
    return True
