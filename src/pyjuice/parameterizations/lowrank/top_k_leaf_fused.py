"""
`LowRank(top_k)` soft-evidence leaves' passes by the CUDA kernels (`kernels/c/lowrank_top_k_leaf.cu`) where they serve
(`top_k_fused.TOP_K_FUSED`, the module built, the shapes they hold, the default precision); else the torch passes
(`top_k.py`), the reference. An input layer's node vectors -- every one given tensors -- take one launch a pass:

* the forward (`leaf_forward`): the transposed log2 tables of the parameters they read (one a block of parameters, made
  by every forward: nothing tells a changed parameter), each node's normalizer over its candidates, the observed
  tokens' values, and where the token is missing and a sum reads it, the slots' log-weights that sum takes in its
  forward and its backward (`slot_rows`) -- the forward's state, kept on the layer for the passes that follow;
* a reading sum's backward hands the leaf its posterior counts of those slots (`add_counts`);
* the backward (`leaf_backward`): the evidence's gradient (summed over the nodes) and the tilt's, every sample; the
  parameters' EM statistics (`compute_param_flows`) by torch passes, from the same counts.

Measured precision (fp32 sums in log2 units, `ex2.approx`) against float64: ~1e-5 nats on the values and the slot
weights, ~1e-6 relative on the gradients.
"""

from __future__ import annotations

import math
import torch

from . import top_k_fused

_L2E = 1.0 / math.log(2.0)
_STATE = "_lr_top_k_leaf"           # the forward's state on the input layer
_ONCE = {}


def module():
    """The leaves' kernels' module, or None (no CUDA toolchain, a failed build)."""
    if "leaf" not in _ONCE:
        from .kernels.c import get_top_k_leaf_module
        _ONCE["leaf"] = get_top_k_leaf_module()
    return _ONCE["leaf"]


def _plan(layer) -> dict:
    """The layer's node vectors as the kernels take them (made once a device): each one's rows, variable, its block of
    parameters (tied node vectors read one: one table) and its readers' slots (`K`: the reading sums' top_k);
    the kernels' per-node-vector arrays."""
    dev = layer.params.device
    plan = layer.__dict__.get("_lr_top_k_leaf_plan")
    if plan is not None and plan["device"] == dev:
        return plan
    from .top_k import _leaf_rows
    readers = layer.__dict__.get("_read_by_sums", {})
    nodes, tables, keys = [], [], {}
    for ns in layer.nodes:
        s, e, v, lvid, spids = _leaf_rows(layer, ns)
        key = (int(spids[0]), e - s)
        if key not in keys:
            keys[key] = len(tables)
            tables.append(spids.contiguous())
        nodes.append(dict(ns = ns, s = s, e = e, v = v, lvid = lvid, tab = keys[key],
                          K = sorted({pz.top_k for pz in readers.get(id(ns), ())})))
    i32 = lambda xs: torch.tensor(xs, dtype = torch.int32, device = dev)
    plan = layer.__dict__["_lr_top_k_leaf_plan"] = dict(
        device = dev, nodes = nodes, tables = tables, index = {id(n["ns"]): t for t, n in enumerate(nodes)},
        nv_tab = i32([n["tab"] for n in nodes]), nv_lvid = i32([n["lvid"] for n in nodes]),
        nv_row0 = i32([n["s"] for n in nodes]), vars = torch.tensor([n["v"] for n in nodes], dtype = torch.long, device = dev),
        read = torch.tensor([len(n["K"]) > 0 for n in nodes], dtype = torch.bool, device = dev),
        Ks = sorted({k for n in nodes for k in n["K"]}), M = sorted({n["e"] - n["s"] for n in nodes}))
    return plan


def serves(desc, layer, node_mars: torch.Tensor, kwargs: dict, staged) -> bool:
    """Whether the kernels take the layer's forward (and so its other passes): every node vector given a tilt, the
    same number of nodes (a multiple of 128) and one top_k for the sums that read them."""
    if not (top_k_fused.TOP_K_FUSED and node_mars.is_cuda and node_mars.dtype == torch.float32 and
            node_mars.is_contiguous() and kwargs.get("precision", "auto") != "fp32" and
            layer.params.dtype == torch.float32 and module() is not None):
        return False
    ids, lp = kwargs.get("soft_evidence_cat_ids"), kwargs.get("categorical_evidence_logp")
    if ids is None or lp is None or ids.dim() != 3 or lp.shape != ids.shape or not staged:
        return False
    plan = _plan(layer)
    if len(plan["M"]) != 1 or plan["M"][0] % 128 != 0 or len(plan["Ks"]) > 1:
        return False
    for n in plan["nodes"]:
        t = staged.get(n["ns"])
        if t is None or len(t) != 2 or t[1].size(2) != desc.top_k + 1:
            return False
    return True


def _missing_all(missing_mask, vars_: torch.Tensor, B: int, dev, batch_first: bool) -> torch.Tensor:
    """`[T, B]` bool: each node vector's variable missing in each sample."""
    if missing_mask is None:
        return torch.zeros(vars_.numel(), B, dtype = torch.bool, device = dev)
    m = missing_mask.to(device = dev, dtype = torch.bool)
    if m.dim() == 1:
        return m[vars_][:, None].expand(-1, B)
    return (m[:, vars_].t() if batch_first else m[vars_, :])


def leaf_forward(desc, layer, data: torch.Tensor, node_mars: torch.Tensor, kwargs: dict) -> bool:
    """The layer's node values by the kernels (`top_k.leaf_forward`'s), its state kept for the passes that follow;
    False where they do not serve (the torch passes then, and no state)."""
    from .top_k import _staged, external_rows
    layer.__dict__.pop(_STATE, None)
    staged = _staged(kwargs)
    if not serves(desc, layer, node_mars, kwargs, staged):
        return False
    mod, plan = module(), _plan(layer)
    nodes = plan["nodes"]
    T, B, dev = len(nodes), node_mars.size(1), node_mars.device
    M, K = plan["M"][0], (plan["Ks"][0] if plan["Ks"] else desc.top_k)

    # the missing tokens a sum reads: their rows of the slot weights (the pass's one synchronization)
    miss = _missing_all(kwargs.get("missing_mask"), plan["vars"], B, dev, kwargs.get("_batch_first", True))
    rmask = (miss & plan["read"][:, None]).cpu()
    counts = rmask.sum(dim = 1).tolist()
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    R = offs[-1]
    nz = rmask.nonzero()
    wrow = torch.full((T, B), -1, dtype = torch.int32)
    wrow[nz[:, 0], nz[:, 1]] = torch.arange(R, dtype = torch.int32)
    wrow, rows = wrow.to(dev), nz[:, 1].to(dev)

    V = nodes[0]["ns"].dist.num_cats
    tables = [mod.top_k_leaf_table(layer.params, spids, V) for spids in plan["tables"]]
    tab = tables[0][None] if len(tables) == 1 else torch.stack(tables)
    emap = external_rows(kwargs)
    tilts = [staged[n["ns"]] for n in nodes]
    Bx, S = tilts[0][0].size(0), tilts[0][1].size(2)
    tau = torch.empty(T, Bx, S, M, dtype = torch.float32, device = dev)
    for t, (A, Bt) in enumerate(tilts):
        torch.bmm(Bt.float().transpose(1, 2), A.float().transpose(1, 2), out = tau[t])   # (A B)^T [Bx, S, M]
    tau.mul_(_L2E)
    ids = kwargs["soft_evidence_cat_ids"].to(torch.int32).contiguous()
    lp = kwargs["categorical_evidence_logp"].float().contiguous()
    emap32 = torch.empty(0, dtype = torch.int32, device = dev) if emap is None else emap.to(torch.int32).contiguous()
    tok = data.index_select(0, plan["vars"]).to(torch.int32).contiguous()
    lz, lb, place, logw = mod.top_k_leaf_forward(tab, plan["nv_tab"], tau, emap32, ids, lp, plan["nv_lvid"],
                                                 plan["nv_row0"], tok, miss.to(torch.uint8).contiguous(), wrow, R, S, K,
                                                 node_mars)
    layer.__dict__[_STATE] = dict(B = B, K = K, S = S, tab = tab, tau = tau, emap = emap32, ids = ids, lp = lp, lz = lz,
                                  lb = lb, place = place, miss = miss, wrow = wrow, logw = logw, offs = offs, rows = rows,
                                  counts = None)
    return True


def _state(layer, B: int):
    st = layer.__dict__.get(_STATE)
    return st if st is not None and st["B"] == B else None


def slot_rows(sib: dict, K: int, batch_size: int):
    """`(slot [B], rows, logw_rows)` of a sum's leaf `sib` (`_sibling_leaves`) from its layer's forward state (as
    `top_k.slot_parts` makes them), or None (no state: the torch passes)."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K:
        return None
    t = _plan(sib["layer"])["index"][id(sib["ns"])]
    o0, o1 = st["offs"][t], st["offs"][t + 1]
    return st["place"][t].long().clamp(0, K), st["rows"][o0:o1], st["logw"][o0:o1]


def add_counts(sib: dict, K: int, batch_size: int, a_rows: torch.Tensor) -> bool:
    """A reading sum's posterior counts of its slots `a_rows [b, M, K + 1]` at the leaf's missing tokens (the rows
    `slot_rows` named), for the leaf's backward; False where its layer has no forward state (`top_k._leaf_terms`
    then)."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K:
        return False
    if st["counts"] is None:
        st["counts"] = torch.zeros_like(st["logw"])
    t = _plan(sib["layer"])["index"][id(sib["ns"])]
    st["counts"][st["offs"][t]:st["offs"][t + 1]] = a_rows
    return True


def leaf_backward(desc, layer, data, node_flows: torch.Tensor, kwargs: dict, logspace_flows: bool, missing_mask,
                  batch_first: bool, compute_param_flows: bool) -> bool:
    """`top_k.leaf_backward` by the kernels, from the layer's forward state and the reading sums' counts; False where
    the forward left none (the torch passes then)."""
    from .top_k import _staged, _grads_kw, _store, per_sample, external_rows
    st = _state(layer, node_flows.size(1))
    if st is None or node_flows.dtype != torch.float32 or not node_flows.is_contiguous():
        return False
    mod, plan = module(), _plan(layer)
    nodes = plan["nodes"]
    gviews, fresh = _grads_kw(kwargs)
    staged = _staged(kwargs)
    glp_out = kwargs.get("categorical_evidence_logp_grad")
    glp = None
    if glp_out is not None:
        glp = glp_out if glp_out.dtype == torch.float32 and glp_out.is_contiguous() else torch.zeros_like(st["lp"])
    counts = st["counts"] if st["counts"] is not None else torch.empty(0, device = node_flows.device)
    dtau = mod.top_k_leaf_backward(st["tab"], plan["nv_tab"], st["tau"], st["emap"], st["ids"], st["lp"], plan["nv_lvid"],
                                   plan["nv_row0"], st["lz"], st["lb"], st["place"], node_flows, logspace_flows,
                                   st["wrow"], st["logw"], counts, st["S"], st["K"],
                                   glp if glp is not None else torch.empty(0, device = node_flows.device))
    if glp is not None and glp is not glp_out:
        glp_out += glp.to(glp_out.dtype)
    if gviews is not None:
        emap = external_rows(kwargs)
        for t, n in enumerate(nodes):
            grads = gviews.get(n["ns"])
            if grads is None:
                continue
            A, Bt = (x.float() for x in per_sample(staged[n["ns"]], emap))
            D = dtau[t].transpose(1, 2)                                                     # [B, M, S]
            _store(gviews, fresh, grads, [torch.bmm(D, Bt.transpose(1, 2)), torch.bmm(A.transpose(1, 2), D)])
    if compute_param_flows and layer.param_flows is not None:
        _param_flows(desc, layer, st, node_flows, kwargs, logspace_flows)
    st["counts"] = None
    return True


def _param_flows(desc, layer, st, node_flows, kwargs, logspace_flows) -> None:
    """The parameters' EM statistics by torch passes (`top_k.leaf_backward`'s and `_leaf_terms`'): F+ and F- of the
    observed tokens (f at the token, f P), of the missing ones a sum reads (its posterior counts rho, f P) and of the
    others (f P both)."""
    from .top_k import leaf_log_probs, per_sample, external_rows, _staged, _d
    plan = _plan(layer)
    pf, l0 = layer.param_flows, layer._output_ind_range[0]
    staged, emap = _staged(kwargs), external_rows(kwargs)
    lp, ids = kwargs["categorical_evidence_logp"], kwargs["soft_evidence_cat_ids"]
    K = st["K"]
    for t, n in enumerate(plan["nodes"]):
        ns, s, e, lvid = n["ns"], n["s"], n["e"], n["lvid"]
        V = ns.dist.num_cats
        f = _d(node_flows[s:e, :].t())
        f = f.exp() if logspace_flows else f
        logp = leaf_log_probs(layer, ns, lp[:, lvid, :], ids[:, lvid, :], per_sample(staged[ns], emap), desc.top_k)
        P = logp.exp()                                                                      # [B, M, C]
        place = st["place"][t].long()
        fo = f * (place >= 0)[:, None]
        C = P.size(2)
        Fp = torch.zeros_like(P).scatter_(2, place.clamp(min = 0)[:, None, None].expand(-1, P.size(1), 1), fo[:, :, None])
        Fm = fo[:, :, None] * P
        miss = st["miss"][t]
        o0, o1 = st["offs"][t], st["offs"][t + 1]
        if o1 > o0:                                                                         # missing, a sum reads it
            rows, a = st["rows"][o0:o1], _d(st["counts"][o0:o1]) if st["counts"] is not None else None
            if a is not None:
                k = torch.arange(C, device = P.device)
                lw = _d(st["logw"][o0:o1])
                rho = torch.where((k < K)[None, None, :], a[:, :, k.clamp(max = K)],
                                  a[:, :, K:K + 1] * (logp[rows] - lw[:, :, K:K + 1]).exp().nan_to_num(0.0))
                Fp[rows] = rho
                Fm[rows] = a.sum(dim = 2, keepdim = True) * P[rows]
        other = miss & ~plan["read"][t]                                                     # missing, no one reads it
        if bool(other.any()):
            fP = (f * other[:, None])[:, :, None] * P
            Fp, Fm = Fp + fP, Fm + fP
        idx = layer.s_pfids[s - l0:e - l0][None, :, None] + ids[:, lvid, :].long()[:, None, :]   # [B, M, C]
        pf.index_add_(0, idx.reshape(-1), Fp.reshape(-1).to(pf.dtype))
        pf.index_add_(0, (idx + V).reshape(-1), Fm.reshape(-1).to(pf.dtype))
