"""
The queries of `LowRank(top_k)` circuits (`top_k.py`'s forms): `juice.queries.conditional`'s read-out at the leaves and
`juice.queries.sample`'s draws at the sums and the leaves.

The conditional of a leaf's variable over its candidates, from the backward's flows F. Where the token is missing and a
`LowRank(top_k)` sum beside the leaf reads it, the sum's parameters depend on the slot the token takes, so the token's
posterior is that of the slots -- the sum's posterior counts a[i, s] = F(i) P(s | i, e), which its backward makes --
spread over each slot's candidates as the leaf's own probabilities:

    P(x = c | e)  =  sum_i a[i, s(c)] P(c | i) / P(s(c) | i)            (the ratio 1 for a slot's own candidate)

s(c) the SUM's slots (its top_k). Elsewhere -- an observed token, or a missing one no sum reads -- the leaf's emission
under each node's flow, sum_i F(i) P(c | i), as `SoftEvidenceCategorical`'s conditional gives it.

A draw: the token beside a reading sum's node i decides the slot its child is drawn under, so the sum draws the two
together, (s, j) in proportion to P(s | i) theta(j | i, s) (times exp(c[j]), the child's value, when conditioning; an
observed token's slot then fixed), and the leaf its candidate within that slot -- the slot's own, or one of the rest in
proportion to P(c | i). A leaf no sum reads draws from P(c | i) whole; an observed token, conditioning, is itself.
"""

from __future__ import annotations

import numpy as np
import torch

from .top_k import PASSES_DTYPE, TOKENS, _d, _finite_or_zero, _leaf_rows, _staged, evidence_rows, external_rows, \
    leaf_log_probs, per_sample, per_sample_evidence


def slot_counts(layer, ns, batch_size: int, kwargs: dict):
    """`(rows [b], a [b, M, K + 1])`: the posterior counts of the slots of leaf node vector `ns`'s token, at the samples
    `rows` whose token is missing, as the sum reading it made them in the backward -- the kernels into the leaf's
    forward state (`top_k_leaf_fused.counts_buffer`: each node's posterior of the slots, then its flow), torch into the
    query's records (`QUERY_RECORDS_KWARG`) --; None where no sum read it (or no token of it is missing)."""
    from . import top_k_leaf_fused
    from pyjuice.layer.parameterized_sum_layer import QUERY_RECORDS_KWARG
    st = top_k_leaf_fused._state(layer, batch_size)
    if st is not None and st["counts"] is not None:
        t = top_k_leaf_fused._plan(layer)["index"][id(ns)]
        if t in st["written"]:
            o0, o1, K = st["offs"][t], st["offs"][t + 1], st["K"]
            c = st["counts"][o0:o1]
            return st["rows"][o0:o1], c[:, :, :K + 1] * c[:, :, K + 1:K + 2]
    records = kwargs.get(QUERY_RECORDS_KWARG)
    return None if records is None else records.get(("top_k_slot_counts", id(ns)))


def conditional(desc, layer, node_flows: torch.Tensor, kwargs: dict, rows: dict, num_rows: int,
                over_candidates: bool = False) -> torch.Tensor:
    """`[B, num_rows, V]` float32: each target variable's posterior over its candidates in its row `rows[v]`, at their
    tokens -- with `over_candidates`, `[B, num_rows, C]` at the candidates' places -- (`SumParameterization.
    input_conditional`), from the log-space `node_flows`: by the read-out's kernel from the kernels' forward state
    where there is one (`top_k_query_kernels`), else `conditional_reference`."""
    from . import top_k_leaf_fused, top_k_query_kernels
    B = node_flows.size(1)
    st = top_k_leaf_fused._state(layer, B)
    if st is not None:
        plan = top_k_leaf_fused._plan(layer)
        targets = [(t, rows[n["v"]]) for t, n in enumerate(plan["nodes"]) if n["v"] in rows]
        out = top_k_query_kernels.conditional(st, plan, node_flows, _staged(kwargs), targets, num_rows,
                                              layer.dist.num_cats, over_candidates)
        if out is not None:
            return out
    probs, ids = conditional_reference(desc, layer, node_flows, kwargs, rows, num_rows)
    if over_candidates:
        return probs
    out = torch.zeros(B, num_rows, layer.dist.num_cats, dtype = torch.float32, device = node_flows.device)
    return out.scatter_add_(2, ids, probs)


def conditional_reference(desc, layer, node_flows: torch.Tensor, kwargs: dict, rows: dict, num_rows: int):
    """`(probs [B, num_rows, C], ids [B, num_rows, C])` float32 / long: `conditional`'s posteriors over the candidates
    and their tokens, by torch from the leaf's parameters (the definition the read-out's kernel is checked against)."""
    staged = _staged(kwargs)
    lp, ids = per_sample_evidence(kwargs)
    B, dev = node_flows.size(1), node_flows.device
    C = ids.size(2)
    probs = torch.zeros(B, num_rows, C, dtype = torch.float32, device = dev)
    out_ids = torch.zeros(B, num_rows, C, dtype = torch.long, device = dev)
    k = torch.arange(C, device = dev)
    for ns in layer.nodes:
        s, e, v, lvid, _ = _leaf_rows(layer, ns)
        r = rows.get(v)
        if r is None:
            continue
        logp = leaf_log_probs(layer, ns, lp[:, lvid, :], ids[:, lvid, :], per_sample(staged[ns], external_rows(kwargs)),
                              desc.top_k)                                                  # [B, M, C]
        f = _d(node_flows[s:e, :].t()).exp()                                               # [B, M]
        P = torch.bmm(f[:, None, :], logp.exp())[:, 0, :]                                  # [B, C]
        sc = slot_counts(layer, ns, B, kwargs)
        if sc is not None and sc[0].numel() > 0:
            srows, a = sc
            K = a.size(2) - 1
            lq = logp[srows]                                                               # [b, M, C]
            share = torch.ones_like(lq)
            if C > K:                                                                      # (the rest slot's candidates)
                rest = torch.logsumexp(lq[:, :, K:], dim = 2, keepdim = True)
                share[:, :, K:] = torch.where(torch.isfinite(rest), (lq[:, :, K:] - rest).exp(), 0.0)
            P[srows] = (_d(a)[:, :, k.clamp(max = K)] * share).sum(dim = 1)
        P = P / P.sum(dim = 1, keepdim = True).clamp_min(1e-30)
        probs[:, r, :] = P.float()
        out_ids[:, r, :] = ids[:, lvid, :].long()
    return probs, out_ids


# ---------------------------------------------------------------------------------------------- sampling

def forward_places(layer, ns, batch_size: int):
    """`[B]` long: where each sample's token was among its candidates in the layer's last forward -- -1 none of them,
    -2 missing -- (the kernels' forward state, or the torch passes' record, `top_k.TOKENS`); None where no forward of
    this batch left either."""
    from . import top_k_leaf_fused
    st = top_k_leaf_fused._state(layer, batch_size)
    if st is not None:
        return st["place"][top_k_leaf_fused._plan(layer)["index"][id(ns)]].long()
    rec = layer.__dict__.get(TOKENS)
    if rec is not None and rec["B"] == batch_size and id(ns) in rec["place"]:
        return rec["place"][id(ns)]
    return None


def _draw(logits: torch.Tensor) -> torch.Tensor:
    """`[n]` long: an index of each row of `logits [n, k]` drawn in proportion to exp(logits) (the Gumbel-max draw)."""
    u = torch.rand(logits.shape, dtype = logits.dtype, device = logits.device).clamp_(min = 1e-30)
    return (logits - (-u.log()).log()).argmax(dim = 1)


def _leaf_logq(layer, ns, bs: torch.Tensor, i: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """`[n, C]` (`PASSES_DTYPE`): log P_b(c | i) of leaf node vector `ns` at the samples `bs`, node `i` of each."""
    _, _, _, lvid, _ = _leaf_rows(layer, ns)
    lp, ids = per_sample_evidence(kwargs)
    tilt = per_sample(_staged(kwargs)[ns], external_rows(kwargs))
    logp = leaf_log_probs(layer, ns, lp[bs, lvid, :], ids[bs, lvid, :], (tilt[0][bs], tilt[1][bs]),
                          ns.parameterization.top_k)                                       # [n, M, C]
    return logp[torch.arange(bs.numel(), device = bs.device), i]


def _slot_log_weights(lq: torch.Tensor, K: int) -> torch.Tensor:
    """`[n, K + 1]`: log P(s | i) of the slots (the first K candidates one by one, the rest summed) from `lq [n, C]`."""
    n, C = lq.shape
    own = lq[:, :min(K, C)]
    if C < K:
        own = torch.cat([own, own.new_full((n, K - C), -float("inf"))], dim = 1)
    rest = torch.logsumexp(lq[:, K:], dim = 1, keepdim = True) if C > K else lq.new_full((n, 1), -float("inf"))
    return torch.cat([own, rest], dim = 1)


def _cand_record(records: dict, leaf_layer, batch_size: int, dev) -> torch.Tensor:
    """`[T, B]` int32 (the leaf layer's node vectors in its plan's order): the candidate place a sum's draw took for the
    leaf beside it, -1 none -- made by the first sum that draws for the layer (`QUERY_RECORDS_KWARG`)."""
    from . import top_k_leaf_fused
    key = ("top_k_cand", id(leaf_layer))
    rec = records.get(key)
    if rec is None or rec.size(1) != batch_size:
        T = len(top_k_leaf_fused._plan(leaf_layer)["nodes"])
        rec = records[key] = torch.full((T, batch_size), -1, dtype = torch.int32, device = dev)
    return rec


def _kernels_serve(*ts) -> bool:
    return all(t.is_cuda for t in ts)


def sample_sums(desc, layer, ns_tensors, element_mars, params, node_samples, element_samples, rows, erows, seed_ptr,
                conditional: bool, kwargs: dict) -> None:
    """A `LowRank(top_k)` sum layer's draw against the structural frontier layout (`SumParameterization.sample_layer`):
    for every live sample of the rows `rows`, its node's slot and child together (the child at `erows`) and the leaf's
    candidate within that slot, recorded for the leaf's draw (`_cand_record`) -- by `_tk_sample_sum_kernel`, else
    `sample_sums_reference`."""
    from pyjuice.layer.parameterized_sum_layer import QUERY_RECORDS_KWARG
    from . import top_k_leaf_fused, top_k_query_kernels as qk
    import triton
    records = kwargs[QUERY_RECORDS_KWARG]
    sibs = layer.__dict__["_sibling_leaves"]
    B, dev = node_samples.size(1), node_samples.device
    K = desc.top_k
    lp, ids = kwargs.get("categorical_evidence_logp"), kwargs.get("soft_evidence_cat_ids")
    ev_mapped = evidence_rows(kwargs) is not None
    emap = external_rows(kwargs)
    staged = _staged(kwargs)
    for ns_info, tensors in ns_tensors:
        geo = desc._geometry(layer, ns_info, dev)
        sib = sibs[id(ns_info.ns)]
        leaf_layer, leaf_ns = sib["layer"], sib["ns"]
        tilt = staged[leaf_ns]
        if not _kernels_serve(node_samples, *tensors, *tilt, leaf_layer.params) or geo["eid"].numel() > 8192 or \
                max(ids.size(2), K + 1) > 1024:
            sample_sums_reference(desc, layer, [(ns_info, tensors)], element_mars, params, node_samples, element_samples,
                                  rows, erows, conditional, kwargs)
            continue
        src = ns_info.ns.get_source_ns() if ns_info.ns.is_tied() else ns_info.ns
        lth = records.get(("top_k_ltheta", id(src)))
        if lth is None:
            lth = records[("top_k_ltheta", id(src))] = \
                desc._dense_params(geo, params, dtype = torch.float32).log().contiguous()
        A, Bm, G = (t if t.dtype == torch.float32 else t.float() for t in tensors)
        U, D = (t if t.dtype == torch.float32 else t.float() for t in tilt)
        lplan = top_k_leaf_fused._plan(leaf_layer)                     # (the leaf's rows kept: no device read a call)
        t_leaf = lplan["index"][id(leaf_ns)]
        lvid, spids = lplan["nodes"][t_leaf]["lvid"], lplan["tables"][lplan["nodes"][t_leaf]["tab"]]
        place = forward_places(leaf_layer, leaf_ns, B) if conditional else None
        if conditional and place is None:
            raise RuntimeError("A conditional draw of a `LowRank(top_k)` sum reads the tokens of the forward pass it "
                               "conditions on: run that forward (at this batch) first.")
        cand = _cand_record(records, leaf_layer, B, dev)[t_leaf]
        em = element_mars if conditional else torch.empty(0, dtype = torch.float32, device = dev)
        assert not conditional or (em.dtype == torch.float32 and em.stride(1) == 1 and em.stride(0) == B), \
            "the element values [elements, B] float32, batch innermost"
        none_l = torch.empty(0, dtype = torch.long, device = dev)
        N = geo["eid"].numel()
        BLOCK_N = triton.next_power_of_2(N)
        qk._tk_sample_sum_kernel[(rows.numel(), B)](
            node_samples, element_samples, rows, erows, B, ns_info.nid_start, ns_info.nid_end - ns_info.nid_start,
            lth, N, A, *A.stride(), Bm, *Bm.stride(), G, *G.stride(), A.size(2),
            none_l if emap is None else emap, int(emap is not None), geo["eid"], em, leaf_layer.params, spids,
            ids, lp.float() if lp.dtype != torch.float32 else lp, int(ev_mapped), ids.size(1), lvid, ids.size(2),
            U, *U.stride(), D, *D.stride(), U.size(2), leaf_ns.parameterization.top_k,
            none_l if place is None else place, cand, seed_ptr, K,
            COND = bool(conditional), HAS_PLACE = place is not None, BLOCK_N = BLOCK_N,
            BLOCK_C = triton.next_power_of_2(max(ids.size(2), K + 1)), RB = 16, SB = 16,
            BLOCK_RE = triton.next_power_of_2(U.size(2)), num_warps = 8)


def sample_sums_reference(desc, layer, ns_tensors, element_mars, params, node_samples, element_samples, rows, erows,
                          conditional: bool, kwargs: dict) -> None:
    """`sample_sums` by torch (the definition its kernel is checked against): row by row, its live samples' joint draws
    of (slot, child) and then of the leaf's candidate within the slot."""
    from pyjuice.layer.parameterized_sum_layer import QUERY_RECORDS_KWARG
    from . import top_k_leaf_fused
    records = kwargs[QUERY_RECORDS_KWARG]
    sibs = layer.__dict__["_sibling_leaves"]
    B, dev = node_samples.size(1), node_samples.device
    K = desc.top_k
    for ns_info, tensors in ns_tensors:
        geo = desc._geometry(layer, ns_info, dev)
        sib = sibs[id(ns_info.ns)]
        A, Bm, G = (_d(t) for t in per_sample(tensors, external_rows(kwargs)))
        ltheta = desc._dense_params(geo, params, dtype = PASSES_DTYPE).log()                # [M, N]
        lc = _d(desc._child_values(geo, element_mars)) if conditional else None              # [B, N]
        place = forward_places(sib["layer"], sib["ns"], B) if conditional else None
        if conditional and place is None:
            raise RuntimeError("A conditional draw of a `LowRank(top_k)` sum reads the tokens of the forward pass it "
                               "conditions on: run that forward (at this batch) first.")
        cand = _cand_record(records, sib["layer"], B, dev)[top_k_leaf_fused._plan(sib["layer"])["index"][id(sib["ns"])]]
        N = ltheta.size(1)
        for row, erow in zip(rows.tolist(), erows.tolist()):
            nid = node_samples[row]
            bs = torch.nonzero((nid >= ns_info.nid_start) & (nid < ns_info.nid_end))[:, 0]
            if bs.numel() == 0:
                continue
            i = nid[bs] - ns_info.nid_start
            lt = torch.bmm(A[bs, i][:, None, :], Bm[bs])[:, 0] + ltheta[i]                   # [n, N]
            lt = lt[:, None, :] + G[bs]                                                    # [n, S, N]
            lt = lt - _finite_or_zero(torch.logsumexp(lt, dim = 2, keepdim = True))        # log theta_b(j | i, s)
            lq = _leaf_logq(sib["layer"], sib["ns"], bs, i, kwargs)                        # [n, C]
            lw = _slot_log_weights(lq, K)                                                  # [n, S]: log P(s | i)
            obs = None
            if conditional:
                pl = place[bs]
                obs = pl != -2
                fixed = torch.full_like(lw, -float("inf")).scatter_(1, pl.clamp(0, K)[:, None], 0.0)
                lw = torch.where(obs[:, None], fixed, lw)                                  # (an observed token's slot)
                lt = lt + lc[bs][:, None, :]
            idx = _draw((lw[:, :, None] + lt).reshape(bs.numel(), -1))
            element_samples[erow, bs] = geo["eid"][idx % N]
            s_ = idx // N
            k = torch.arange(lq.size(1), device = dev)
            c = torch.where(s_ < K, s_, _draw(lq.masked_fill(k[None, :] < K, -float("inf"))))   # (within the slot)
            if obs is not None:
                c = torch.where(obs, pl.clamp(min = 0), c)                                 # (an observed token)
            cand[bs] = c.to(cand.dtype)


def sample_leaves(desc, layer, samples, node_flows, kwargs: dict) -> None:
    """A `LowRank(top_k)` leaf layer's draw (`SumParameterization.input_sample`), its tokens into `samples [num_vars,
    B]`: a node vector a sum reads, the candidates the sum's draw took (`_cand_record`); another, each sample's
    selected node's candidate drawn as its P(c | i) (`_tk_sample_leaf_kernel`, else torch) -- conditioning, an
    observed token itself."""
    import random
    import triton
    from pyjuice.layer.parameterized_sum_layer import QUERY_RECORDS_KWARG
    from . import top_k_leaf_fused, top_k_query_kernels as qk
    records = kwargs.get(QUERY_RECORDS_KWARG) or {}
    conditional = records.get("conditional", False)
    plan = top_k_leaf_fused._plan(layer)
    B, dev = node_flows.size(1), node_flows.device
    lp, ids = kwargs.get("categorical_evidence_logp"), kwargs.get("soft_evidence_cat_ids")
    er = evidence_rows(kwargs)
    cand = records.get(("top_k_cand", id(layer)))
    read = [t for t, n in enumerate(plan["nodes"]) if n["K"]] if cand is not None else []
    if read:
        tr = plan.get("_read_index")
        if tr is None or tr.device != dev:
            tr = plan["_read_index"] = torch.tensor(read, dtype = torch.long, device = dev)
        c = cand[tr].long()                                                                # [Tr, B]
        rows_b = torch.arange(B, device = dev) if er is None else er
        tok = ids[rows_b[None, :], plan["nv_lvid"][tr].long()[:, None], c.clamp(min = 0)]
        v = plan["vars"][tr]
        samples[v] = torch.where(c >= 0, tok.to(samples.dtype), samples[v])
    emap = external_rows(kwargs)
    staged = _staged(kwargs)
    for t, n in enumerate(plan["nodes"]):
        if t in read:
            continue
        ns = n["ns"]
        U, D = staged[ns]
        place = forward_places(layer, ns, B) if conditional else None
        if _kernels_serve(node_flows, U, D, layer.params) and ids.size(2) <= 1024 and n["e"] - n["s"] <= 8192:
            U, D = (x if x.dtype == torch.float32 else x.float() for x in (U, D))
            none_l = torch.empty(0, dtype = torch.long, device = dev)
            nf = node_flows if node_flows.dtype == torch.float32 else node_flows.float()
            qk._tk_sample_leaf_kernel[(B,)](
                samples, n["v"], nf, n["s"], n["e"] - n["s"], B, layer.params, plan["tables"][n["tab"]],
                ids, lp.float() if lp.dtype != torch.float32 else lp, int(er is not None), ids.size(1), n["lvid"],
                ids.size(2), U, *U.stride(), D, *D.stride(), U.size(2), ns.parameterization.top_k,
                none_l if emap is None else emap, int(emap is not None), none_l if place is None else place,
                random.randint(0, 2 ** 31 - 1), HAS_PLACE = place is not None,
                BLOCK_M = triton.next_power_of_2(n["e"] - n["s"]), BLOCK_C = triton.next_power_of_2(ids.size(2)),
                BLOCK_RE = triton.next_power_of_2(U.size(2)), num_warps = 8)
            continue
        _, idsg = per_sample_evidence(kwargs)
        fl = node_flows[n["s"]:n["e"], :]                                                  # [M, B]: 1 where selected
        bs = torch.nonzero(fl.amax(dim = 0) > 0.5)[:, 0]
        if bs.numel() == 0:
            continue
        i = fl[:, bs].argmax(dim = 0)
        c = _draw(_leaf_logq(layer, ns, bs, i, kwargs))
        if place is not None:
            pl = place[bs]
            c = torch.where(pl >= 0, pl, c)                                                # (an observed token)
        samples[n["v"], bs] = idsg[bs, n["lvid"], :].gather(1, c[:, None])[:, 0].to(samples.dtype)


# ---------------------------------------------------------------------------------------------- the walk

def forward_places_all(layer, batch_size: int):
    """`[T, B]` long: `forward_places` of every node vector of the leaf layer (its plan's order), or None."""
    from . import top_k_leaf_fused
    st = top_k_leaf_fused._state(layer, batch_size)
    if st is not None:
        return st["place"].long()
    rec = layer.__dict__.get(TOKENS)
    if rec is None or rec["B"] != batch_size:
        return None
    nodes = top_k_leaf_fused._plan(layer)["nodes"]
    if not all(id(n["ns"]) in rec["place"] for n in nodes):
        return None
    return torch.stack([rec["place"][id(n["ns"])] for n in nodes])


def walk_levels(pc, plan, layer_id: int) -> list:
    """The run of levels the walk can take from inner layer group `layer_id` (a sum group) down -- each a sum group of
    one `LowRank(top_k)` layer of one node vector over one frontier row, then a product group of one layer of one
    partition, every child of the sum an element of it --, its static tables (the children's element ids, their own
    children's node ids and frontier rows); [] where none starts there. Kept on the circuit, a device."""
    from pyjuice.layer.parameterized_sum_layer import ParameterizedSumLayer
    import triton
    # (keyed by the parameters' device: `pc.device` may be index-less, `cuda`, and never equal a tensor's `cuda:0`)
    dev = pc.params.device
    cache = pc.__dict__.setdefault("_tk_walk_levels", {})
    hit = cache.get((layer_id, dev))
    if hit is not None:
        return hit
    groups = pc.inner_layer_groups
    levels, idx = [], layer_id
    while idx >= 1:
        sg, pg = groups[idx], groups[idx - 1]
        if not (sg.is_sum() and pg.is_prod()):
            break
        sls, pls = list(sg), list(pg)
        if len(sls) != 1 or len(pls) != 1:
            break
        sl, pl = sls[0], pls[0]
        par = getattr(sl, "parameterization", None)
        if not isinstance(sl, ParameterizedSumLayer) or getattr(par, "top_k", None) is None or \
                len(sl.external_node_infos) != 1 or pl.num_fw_partitions != 1 or \
                plan.sum_rows[id(sl)].numel() != 1:
            break
        ns_info = sl.external_node_infos[0]
        geo = par._geometry(sl, ns_info, dev)
        eid = geo["eid"]
        p_nids = pl.partitioned_nids[0].to(dev).long()
        p_cids = pl.partitioned_cids[0].to(dev).long()
        p_crows = plan.prod_crows[id(pl)][0].to(dev).long()
        nb = torch.searchsorted(p_nids, eid.clamp(min = 0), right = True) - 1
        nbc = nb.clamp(min = 0)
        off = eid - p_nids[nbc]
        inside = (eid >= 0) & (nb >= 0) & (off >= 0) & (off < pl.block_size)
        if not bool((inside == (eid >= 0)).all()):
            break                                                  # (a child no element of the product layer below)
        cn = torch.where(inside[:, None] & (p_cids[nbc] > 0), p_cids[nbc] + off[:, None], torch.zeros_like(p_cids[nbc]))
        cr = torch.where(inside[:, None], p_crows[nbc], torch.full_like(p_crows[nbc], -1))
        levels.append(dict(sl = sl, ns_info = ns_info, geo = geo, row = int(plan.sum_rows[id(sl)][0]),
                           eid = eid.contiguous(), cn = cn, cr = cr, sib = sl.__dict__["_sibling_leaves"][id(ns_info.ns)]))
        idx -= 2
    if levels:
        E_P = triton.next_power_of_2(max(lv["cn"].size(1) for lv in levels))
        for lv in levels:
            e = lv["cn"].size(1)
            lv["cn"] = torch.cat([lv["cn"], lv["cn"].new_zeros(lv["cn"].size(0), E_P - e)], 1).contiguous()
            lv["cr"] = torch.cat([lv["cr"], lv["cr"].new_full((lv["cr"].size(0), E_P - e), -1)], 1).contiguous()
    cache[(layer_id, dev)] = levels
    return levels


#: the walk kernel's ranks a tile, slots a tile and warps (the Triton walk's)
WALK_TILES = (16, 16, 8)

#: whether a run of levels is walked by one kernel (`walk`; else the scoped pass's per-layer draws, `sample_sums`), and
#: by the CUDA kernel where it serves (else the Triton one)
WALK, WALK_CUDA = True, True


def _walk_module():
    if not WALK_CUDA:
        return None
    from .kernels.c import get_top_k_walk_module
    return get_top_k_walk_module()


def walk(pc, plan, layer_id: int, node_samples, seed_ptr, conditional: bool, kwargs: dict) -> int:
    """`SumParameterization.sample_levels` of `LowRank(top_k)`: the run of levels from group `layer_id` down
    (`walk_levels`) drawn by `_tk_walk_kernel` in one launch -- the groups it took (0: none, the scoped pass's own then)."""
    import triton
    from pyjuice.layer.parameterized_sum_layer import QUERY_RECORDS_KWARG
    from . import top_k_leaf_fused, top_k_query_kernels as qk
    if not WALK or not _kernels_serve(node_samples):
        return 0
    levels = walk_levels(pc, plan, layer_id)
    staged = _staged(kwargs)
    lp, ids = kwargs.get("categorical_evidence_logp"), kwargs.get("soft_evidence_cat_ids")
    if not levels or staged is None or ids is None or lp is None or not node_samples.is_cuda or \
            node_samples.stride(1) != 1 or ids.size(2) > 1024 or any(lv["eid"].numel() > 4096 for lv in levels):
        return 0
    B, dev = node_samples.size(1), node_samples.device
    records = kwargs[QUERY_RECORDS_KWARG]
    emap = external_rows(kwargs)
    nm = pc.node_mars
    if conditional and not (nm.dtype == torch.float32 and nm.is_contiguous() and nm.size(1) == B):
        return 0
    f32 = lambda t: t if t.dtype == torch.float32 else t.float()
    lvi, lvp, keep, places = [], [], [], {}
    for lv in levels:
        sl, ns_info, sib = lv["sl"], lv["ns_info"], lv["sib"]
        tensors, tilt = staged.get(ns_info.ns), staged.get(sib["ns"])
        if tensors is None or tilt is None:
            return 0
        A, Bm, G = (f32(t) for t in tensors)
        U, D = (f32(t) for t in tilt)
        leaf_layer, leaf_ns = sib["layer"], sib["ns"]
        lplan = top_k_leaf_fused._plan(leaf_layer)
        t_leaf = lplan["index"][id(leaf_ns)]
        n = lplan["nodes"][t_leaf]
        src = ns_info.ns.get_source_ns() if ns_info.ns.is_tied() else ns_info.ns
        lth = records.get(("top_k_ltheta", id(src)))
        if lth is None:
            lth = records[("top_k_ltheta", id(src))] = \
                sl.parameterization._dense_params(lv["geo"], pc.params, dtype = torch.float32).log().contiguous()
        cand = _cand_record(records, leaf_layer, B, dev)
        place = 0
        if conditional:
            if id(leaf_layer) not in places:
                places[id(leaf_layer)] = forward_places_all(leaf_layer, B)
            pa = places[id(leaf_layer)]
            if pa is None:
                raise RuntimeError("A conditional draw of a `LowRank(top_k)` sum reads the tokens of the forward pass "
                                   "it conditions on: run that forward (at this batch) first.")
            place = pa[t_leaf].data_ptr()
        keep += [A, Bm, G, U, D, lth, cand]
        lvi.append([lv["row"], ns_info.nid_start, ns_info.nid_end - ns_info.nid_start, lv["eid"].numel(), A.size(2),
                    U.size(2), leaf_ns.parameterization.top_k, n["lvid"], sl.parameterization.top_k,
                    *A.stride(), *Bm.stride(), *G.stride(), *U.stride(), *D.stride()])
        lvp.append([A.data_ptr(), Bm.data_ptr(), G.data_ptr(), U.data_ptr(), D.data_ptr(), lth.data_ptr(),
                    leaf_layer.params.data_ptr(), lplan["tables"][n["tab"]].data_ptr(), lv["eid"].data_ptr(),
                    lv["cn"].data_ptr(), lv["cr"].data_ptr(), place, cand[t_leaf].data_ptr()])
    # (both tables one tensor, one copy: the pointers' rows padded to the integers' width, read at that stride; numpy
    # makes it -- `torch.tensor` of nested lists took 0.17 ms)
    tab = np.zeros((2 * len(lvi), qk.WALK_NI), dtype = np.int64)
    tab[:len(lvi)] = lvi
    tab[len(lvi):, :qk.WALK_NP] = lvp
    tabs = top_k_leaf_fused._to_device(torch.from_numpy(tab), dev)
    L = len(levels)
    lvi_t, lvp_t = tabs[:L], tabs[L:]
    K = max(lv["sl"].parameterization.top_k for lv in levels)
    maxN = max(lv["eid"].numel() for lv in levels)
    none_l = torch.empty(0, dtype = torch.long, device = dev)
    cm = _walk_module()
    if cm is not None and ids.size(2) <= 512 and maxN <= 2048 and all(row[4] <= 128 and row[5] <= 128 for row in lvi):
        # (the CUDA walk: a CTA a sample, a warp a slot -- `lowrank_top_k_walk.cu`)
        cm.top_k_walk(lvi_t, lvp_t, L, node_samples, nm if conditional else torch.empty(0, device = dev), ids.long(),
                      f32(lp).contiguous(), none_l if emap is None else emap.long(), int(evidence_rows(kwargs) is not None),
                      seed_ptr, bool(conditional), bool(conditional), levels[0]["cn"].size(1), maxN,
                      max(ids.size(2), K + 1))
        return 2 * L
    BLOCK_N = triton.next_power_of_2(maxN)
    qk._tk_walk_kernel[(B,)](
        lvi_t, lvp_t, L, node_samples, nm if conditional else torch.empty(0, dtype = torch.float32, device = dev), B,
        ids, f32(lp), int(evidence_rows(kwargs) is not None), ids.size(1), ids.size(2),
        none_l if emap is None else emap, int(emap is not None), seed_ptr,
        NI = qk.WALK_NI, NP = qk.WALK_NI, COND = bool(conditional), HAS_PLACE = bool(conditional), BLOCK_N = BLOCK_N,
        BLOCK_C = triton.next_power_of_2(max(ids.size(2), K + 1)), RB = WALK_TILES[0], SB = WALK_TILES[1],
        BLOCK_RE = triton.next_power_of_2(max(lv["sib"]["ns"].parameterization.rank for lv in levels)),
        E_P = levels[0]["cn"].size(1), num_warps = WALK_TILES[2])
    return 2 * L
