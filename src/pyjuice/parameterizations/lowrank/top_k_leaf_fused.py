"""
`LowRank(top_k)` soft-evidence leaves' passes by the CUDA kernels (`kernels/c/lowrank_top_k_leaf.cu`) where they serve
(`top_k_fused.TOP_K_FUSED`, the module built, the shapes they hold, the default precision); else the torch passes
(`top_k.py`), the reference. An input layer's node vectors -- every one given tensors -- take one launch a pass:

* the forward (`leaf_forward`): the transposed log2 tables of the parameters they read (one a block of parameters, made
  by every forward: nothing tells a changed parameter), each node's normalizer over its candidates, the observed
  tokens' values, and where the token is missing and a sum reads it, the slots' log-weights that sum takes in its
  forward and its backward (`slot_rows`) -- the forward's state, kept on the layer for the passes that follow;
* a reading sum's backward hands the leaf its posterior counts of those slots (`add_counts`);
* the backward (`leaf_backward`): the evidence's gradient (summed over the nodes) and the tilt's, every sample, and the
  parameters' EM statistics (`compute_param_flows`): F+ and F- into transposed buffers a block of parameters, then
  added into the layer's flows.

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
    nodes, tables, flows, keys = [], [], [], {}
    l0 = layer._output_ind_range[0]
    for ns in layer.nodes:
        s, e, v, lvid, spids = _leaf_rows(layer, ns)
        key = (int(spids[0]), e - s)
        if key not in keys:
            keys[key] = len(tables)
            tables.append(spids.contiguous())
            flows.append(layer.s_pfids[s - l0:e - l0].contiguous())
        nodes.append(dict(ns = ns, s = s, e = e, v = v, lvid = lvid, tab = keys[key],
                          K = sorted({pz.top_k for pz in readers.get(id(ns), ())})))
    i32 = lambda xs: torch.tensor(xs, dtype = torch.int32, device = dev)
    plan = layer.__dict__["_lr_top_k_leaf_plan"] = dict(
        device = dev, nodes = nodes, tables = tables, flows = flows, index = {id(n["ns"]): t for t, n in enumerate(nodes)},
        nv_tab = i32([n["tab"] for n in nodes]), nv_lvid = i32([n["lvid"] for n in nodes]),
        nv_row0 = i32([n["s"] for n in nodes]), vars = torch.tensor([n["v"] for n in nodes], dtype = torch.long, device = dev),
        read = torch.tensor([len(n["K"]) > 0 for n in nodes], dtype = torch.bool, device = dev),
        Ks = sorted({k for n in nodes for k in n["K"]}), M = sorted({n["e"] - n["s"] for n in nodes}))
    return plan


def serves(desc, layer, node_mars: torch.Tensor, kwargs: dict, staged) -> bool:
    """Whether the kernels take the layer's forward (and so its other passes): every node vector given a tilt, the
    same number of nodes (a multiple of 128) and one top_k for the sums that read them."""
    pf = layer.param_flows
    if not (top_k_fused.TOP_K_FUSED and node_mars.is_cuda and node_mars.dtype == torch.float32 and
            node_mars.is_contiguous() and kwargs.get("precision", "auto") != "fp32" and
            layer.params.dtype == torch.float32 and (pf is None or (pf.dtype == torch.float32 and pf.is_contiguous()))
            and module() is not None):
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
    for t, (A, Bt) in enumerate(tilts):                                                  # log2 e (A B)^T [Bx, S, M]
        torch.bmm((Bt.float() * _L2E).transpose(1, 2), A.float().transpose(1, 2), out = tau[t])   # (the scale on Bt)
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
    if st.get("slot") is None:                                     # (every position's, once a forward)
        st["slot"] = st["place"].long().clamp_(0, K)
    return st["slot"][t], st["rows"][o0:o1], st["logw"][o0:o1]


#: The counts' rows as the sums' backward kernels write them (`SA` of `kernels/c/lowrank_top_k.cu`: K + 1 padded)
COUNTS_STRIDE = 72


def counts_buffer(sib: dict, K: int, batch_size: int):
    """The leaf's buffer `[b, M, COUNTS_STRIDE]` of a reading sum's posterior counts at its missing tokens (the rows
    `slot_rows` named) -- the sum's backward kernels write them there --, or None where its layer has no forward state
    or K is past the stride."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K or K + 1 > COUNTS_STRIDE:
        return None
    t = _plan(sib["layer"])["index"][id(sib["ns"])]
    return _counts(st)[st["offs"][t]:st["offs"][t + 1]]


def _counts(st: dict) -> torch.Tensor:
    if st["counts"] is None:                                       # (zeros: a row no sum writes reads none)
        R, M, S1 = st["logw"].shape
        st["counts"] = torch.zeros(R, M, max(S1, COUNTS_STRIDE), dtype = st["logw"].dtype, device = st["logw"].device)
    return st["counts"]


def add_counts(sib: dict, K: int, batch_size: int, a_rows: torch.Tensor) -> bool:
    """A reading sum's posterior counts of its slots `a_rows [b, M, K + 1]` at the leaf's missing tokens (the rows
    `slot_rows` named), for the leaf's backward -- nothing to do where they are already its buffer's
    (`counts_buffer`); False where its layer has no forward state (`top_k._leaf_terms` then)."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K:
        return False
    t = _plan(sib["layer"])["index"][id(sib["ns"])]
    dst = _counts(st)[st["offs"][t]:st["offs"][t + 1]]
    if dst.numel() > 0 and a_rows.data_ptr() != dst.data_ptr():
        dst[:, :, :a_rows.size(2)] = a_rows
    return True



def _tilt_grads(mod, nodes, graded, dtau, staged, gviews, fresh, emap) -> None:
    """The tilt's gradients from dtau [T, B or Bx (folded), S, M] (a row of destinations each, reading the row of
    tensors `emap` names -- None: its own), added into the destinations: one pass over dtau and the factors a run of
    positions whose operands share their strides (`top_k_leaf_tilt_grads`), else cuBLAS's products into them."""
    dests = [gviews.get(n["ns"]) for n in graded]
    if fresh:
        for g in dests:
            gviews.init_grads(g)
    idx = [nodes.index(n) for n in graded]
    ops = [tuple(x.float() for x in staged[n["ns"]]) + tuple(g) for n, g in zip(graded, dests)]
    A0, B0 = ops[0][0], ops[0][1]
    fits = (A0.size(2) <= 64 and B0.size(2) <= 65 and A0.size(1) % 64 == 0 and dtau.dtype == torch.float32 and
            all(x.dtype == torch.float32 for o in ops for x in o))
    em = torch.empty(0, dtype = torch.int32, device = dtau.device) if emap is None else emap.to(torch.int32).contiguous()
    k = 0
    while k < len(ops):
        if not fits:
            t = idx[k]
            A, Bt, gA, gB = ops[k]
            if emap is not None:
                A, Bt = A.index_select(0, emap), Bt.index_select(0, emap)
            D = dtau[t].transpose(1, 2)                                                     # [B or Bx, M, S]
            gA.baddbmm_(D, Bt.transpose(1, 2))
            gB.baddbmm_(A.transpose(1, 2), D)
            k += 1
            continue
        # a run of consecutive positions with the same strides (one launch)
        j = k + 1
        while j < len(ops) and idx[j] == idx[j - 1] + 1 and all(x.stride() == y.stride() for x, y in zip(ops[j], ops[k])):
            j += 1
        run = ops[k:j]
        ptrs = torch.tensor([[o[q].data_ptr() for o in run] for q in range(4)], dtype = torch.int64, device = dtau.device)
        strides = torch.tensor([s for x in run[0] for s in x.stride()], dtype = torch.int64)
        mod.top_k_leaf_tilt_grads(dtau[idx[k]:idx[j - 1] + 1], ptrs, strides, em, A0.size(2))
        k = j

def leaf_backward(desc, layer, data, node_flows: torch.Tensor, kwargs: dict, logspace_flows: bool, missing_mask,
                  batch_first: bool, compute_param_flows: bool) -> bool:
    """`top_k.leaf_backward` by the kernels, from the layer's forward state and the reading sums' counts; False where
    the forward left none (the torch passes then)."""
    from .top_k import _staged, _grads_kw, external_rows
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
    dev = node_flows.device
    none = torch.empty(0, device = dev)
    counts = st["counts"] if st["counts"] is not None else none
    em = compute_param_flows and layer.param_flows is not None
    fpT, fmT = (torch.zeros_like(st["tab"]), torch.zeros_like(st["tab"])) if em else (none, none)
    pm = kwargs.get("param_flows_mask") if em else None                 # (the samples the EM statistics take)
    # the tilt's gradients' dtau: where any is wanted; where the destinations fold, each sample's times its sign summed
    # into the row it read by the kernel -- a row's gradients then one product
    graded = [n for n in nodes if gviews is not None and gviews.get(n["ns"]) is not None]
    fold = getattr(gviews, "fold", None) if graded else None
    drow, dsgn, Bd = none.int(), none, 0
    if fold is not None:
        drow, dsgn, Bd = fold[0].to(torch.int32).contiguous(), fold[1].float().contiguous(), staged[graded[0]["ns"]][0].size(0)
    dtau = mod.top_k_leaf_backward(st["tab"], plan["nv_tab"], st["tau"], st["emap"], st["ids"], st["lp"], plan["nv_lvid"],
                                   plan["nv_row0"], st["lz"], st["lb"], st["place"], node_flows, logspace_flows,
                                   st["wrow"], st["logw"], counts, st["S"], st["K"], glp if glp is not None else none,
                                   fpT, fmT, none.bool() if pm is None else pm.contiguous(), bool(graded), drow, dsgn, Bd)
    if em:                                                                              # (F+ then F- of each block's rows)
        V = nodes[0]["ns"].dist.num_cats
        for k, spf in enumerate(plan["flows"]):
            mod.top_k_leaf_fold(fpT[k], spf, layer.param_flows, 0)
            mod.top_k_leaf_fold(fmT[k], spf, layer.param_flows, V)
    if glp is not None and glp is not glp_out:
        glp_out += glp.to(glp_out.dtype)
    if graded:
        _tilt_grads(mod, nodes, graded, dtau, staged, gviews, fresh, None if fold is not None else external_rows(kwargs))
    st["counts"] = None
    return True
