"""
`LowRank(top_k)` soft-evidence leaves' passes by the CUDA kernels (`kernels/c/lowrank_top_k_leaf.cu`) where they serve
(`top_k_fused.TOP_K_FUSED`, the module built, the shapes they hold, the default precision); else the torch passes
(`top_k.py`), the reference. An input layer's node vectors -- every one given tensors -- take one launch a pass:

* the forward (`leaf_forward`): the transposed log2 tables of the parameters they read (one a block of parameters, made
  by every forward: nothing tells a changed parameter), each node's normalizer over its candidates, the observed
  tokens' values, and where the token is missing and a sum reads it, the slots' log-weights that sum takes in its
  forward and its backward (`slot_rows`) -- the forward's state, kept on the layer for the passes that follow (until
  the leaf's backward has used it: then released, and a second backward of the same forward takes the torch passes);
* a reading sum hands the leaf its posterior counts of those slots (`counts_buffer`: the slots' posterior a node and
  the node's flow, which its kernels write; `add_counts`);
* the backward (`leaf_backward`): the evidence's gradient (summed over the nodes) and the tilt's, every sample, and the
  parameters' EM statistics (`compute_param_flows`): the candidates of the samples they take sorted by category, then
  a category's sums over them added into the layer's flows -- no buffer of every category.

The tilt's products -- the forward's log2 e (A B)^T, the backward's gradients of A and B from d / d tau -- by their own
kernels (`kernels/c/lowrank_top_k_tilt.cu`: exact fp32, a launch a run of positions), else cuBLAS's.

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


def tilt_module():
    """The tilt products' module (`kernels/c/lowrank_top_k_tilt.cu`), or None (cuBLAS's products then)."""
    if "tilt" not in _ONCE:
        from .kernels.c import get_top_k_tilt_module
        _ONCE["tilt"] = get_top_k_tilt_module()
    return _ONCE["tilt"]


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
    spf = torch.stack(flows) if len({f.numel() for f in flows}) == 1 else None       # [ntab, M] (the kernels' M one)
    plan = layer.__dict__["_lr_top_k_leaf_plan"] = dict(
        device = dev, nodes = nodes, tables = tables, index = {id(n["ns"]): t for t, n in enumerate(nodes)},
        spf = spf, spf_aligned = spf is not None and bool((spf % 4 == 0).all()),
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

    # one call into the module (`top_k_leaf_pass`): the categories some tile takes among its candidates and their rows
    # of the tables (the others none -- never read: the tables hold the pass's categories alone, MEASURED 48 blocks:
    # 494 -> 193 MiB on the dump's rows, -> 383 with DISTINCT=1 rows); the missing tokens a sum reads, their rows of the
    # slot weights -- the pass's one synchronization, waited on after the work that needs none of it --; the tables;
    # the kernel. Its ~40 operations one by one from Python were ~450 us of host time a pass at 2 samples (the pass's
    # start host-bound); the tilts made in the forward's kernel where it can (no tau [T, Bx, S, M] made, read, nor kept
    # for the backward: its rest slot alone, tauK), else by their own kernel first
    V = nodes[0]["ns"].dist.num_cats
    miss = _missing_all(kwargs.get("missing_mask"), plan["vars"], B, dev, kwargs.get("_batch_first", True))
    tilts = [staged[n["ns"]] for n in nodes]
    tA, tB = [t[0] for t in tilts], [t[1] for t in tilts]
    args = (layer.params, plan["tables"], plan["nv_tab"], plan["nv_lvid"], plan["nv_row0"], plan["vars"], plan["read"],
            miss, kwargs["soft_evidence_cat_ids"], kwargs["categorical_evidence_logp"], data, external_rows(kwargs))
    tau = None
    got = mod.top_k_leaf_pass(*args, None, tA, tB, node_mars, V, K, tB[0].size(2), M)
    if not got:
        tau = torch.empty(T, tA[0].size(0), tB[0].size(2), M, dtype = torch.float32, device = dev)
        _tilts(tilts, tau)
        got = mod.top_k_leaf_pass(*args, tau, [], [], node_mars, V, K, tau.size(2), M)
    tab, tauK, emap32, ids, lp, lz, lb, place, wrow, logw, rows, ord_, rank, rowstart, crow, offs = got
    layer.__dict__[_STATE] = dict(B = B, K = K, S = tB[0].size(2), tab = tab, tau = tau, tauK = None if tau is not None
                                  else tauK, emap = emap32, ids = ids, lp = lp, lz = lz, lb = lb, place = place,
                                  miss = miss, wrow = wrow, logw = logw, offs = offs.tolist(), rows = rows,
                                  counts = None, ord = ord_, rank = rank, rowstart = rowstart, crow = crow)
    return True


def _to_device(t: torch.Tensor, dev) -> torch.Tensor:
    """A host tensor `t` on `dev` without a synchronization: staged in pinned memory and copied asynchronously (the
    caching host allocator keeps the pinned block until the copy is done)."""
    if dev.type != "cuda":
        return t.to(dev)
    h = torch.empty(t.shape, dtype = t.dtype, pin_memory = True)
    h.copy_(t)
    return h.to(dev, non_blocking = True)


def _state(layer, B: int):
    st = layer.__dict__.get(_STATE)
    return st if st is not None and st["B"] == B else None


def pass_cache(sib: dict, batch_size: int):
    """A dict for what the sums reading the leaf share in a pass (their parameters' tables: `top_k_fused.theta_tables`),
    kept in the leaf's forward state -- made again by every forward, so nothing outlives the parameters it was made from
    --, or None where the leaf's layer has no state for this batch."""
    st = _state(sib["layer"], batch_size)
    return None if st is None else st.setdefault("shared", {})


def slot_rows(sib: dict, K: int, batch_size: int):
    """`(slot [B], rows, logw_rows)` of a sum's leaf `sib` (`_sibling_leaves`) from its layer's forward state (as
    `top_k.slot_parts` makes them), or None (no state: the torch passes)."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K:
        return None
    return _slot_views(st, _plan(sib["layer"])["index"][id(sib["ns"])], K)


def _slot_views(st: dict, t: int, K: int):
    # (every position's, made once a forward -- its sums' forward and backward both read them --, a call a kind of view:
    # ~2 us a view made one by one in Python)
    views = st.get("slot_views")
    if views is None:
        if st.get("slot") is None:
            st["slot"] = st["place"].long().clamp_(0, K)
        offs = st["offs"]
        sizes = [offs[u + 1] - offs[u] for u in range(len(offs) - 1)]
        views = st["slot_views"] = list(zip(st["slot"].unbind(0), st["rows"].split(sizes),
                                            st["logw"][:, :, :K + 1].split(sizes)))       # (rows 72 wide: 0 .. K)
    return views[t]


def sum_parts(sib: dict, K: int, batch_size: int, emap, cache: dict):
    """`(slot [B], rows, logw_rows, items, Rm, t)`: what a reading sum's kernels take of its leaf `sib` from its layer's
    forward state, in one look-up -- `slot_rows`' three, its paired work items (`top_k_items`' for every sum reading the
    layer: one launch a pass, kept in `cache`, the pass's shared dict, `pass_cache`), its masked samples' count, its
    place among the layer's leaves (`counts_rows`) --, made once a pass for its forward and its backward: a position's
    host time is what the small batches' passes take (MEASURED, the Codpula head's circuit at 2 samples: these
    look-ups one by one ~14 us a position, ~0.5 ms a forward). None where the layer has no forward state for this
    batch, or for another K. `emap` [B] long the row each sample reads (None: its own)."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K:
        return None
    parts = st.setdefault("parts", {})
    got = parts.get(id(sib["ns"]))
    if got is None:
        t = _plan(sib["layer"])["index"][id(sib["ns"])]
        items = work_items(st, emap, cache)[1]
        got = parts[id(sib["ns"])] = (*_slot_views(st, t, K), items[t], st["offs"][t + 1] - st["offs"][t], t)
    return got


def work_items(st: dict, emap, cache: dict):
    """`(flat, per_leaf)`: the paired work items of every sum reading the leaf layer, from its forward state (its slots,
    its masked rows) -- `top_k_items`' four buffers `[mitems, oitems, noitems, slot32]`, a leaf's after the earlier
    leaves', and each leaf's four slices of them --, one launch a pass, kept in the pass's `cache` (`pass_cache`).
    `emap` [B] long the row each sample reads (None: its own)."""
    got = cache.get("items")
    if got is None:
        offs, dev = st["offs"], st["rows"].device
        if st.get("slot") is None:
            st["slot"] = st["place"].long().clamp_(0, st["K"])
        roff = _to_device(torch.tensor(offs, dtype = torch.int32), dev)
        made = top_k_fused.top_k_module().top_k_items(
            torch.empty(0, dtype = torch.long, device = dev) if emap is None else emap.long().contiguous(),
            st["rows"].long().contiguous(), roff, offs, st["slot"].contiguous())
        cache["items_flat"] = made[:4]
        got = cache["items"] = [made[4 + 4 * t:8 + 4 * t] for t in range(len(offs) - 1)]
    return cache["items_flat"], got


#: The counts' rows as the sums' kernels write them (`SA` of `kernels/c/lowrank_top_k.cu`: K + 2 padded)
COUNTS_STRIDE = 72


def counts_buffer(sib: dict, K: int, batch_size: int):
    """The leaf's buffer `[b, M, COUNTS_STRIDE]` of a reading sum's posterior counts at its missing tokens (the rows
    `slot_rows` named), which the sum's kernels write: a node's row the posterior of its K + 1 slots, then its flow
    (their product the counts) -- or None where its layer has no forward state or K is past the stride."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K or K + 2 > COUNTS_STRIDE:
        return None
    return _counts_rows(st, _plan(sib["layer"])["index"][id(sib["ns"])])


def counts_rows(sib: dict, K: int, batch_size: int, t: int):
    """`counts_buffer`'s, of the leaf `t` of its layer (`sum_parts`' place: no look-up)."""
    st = _state(sib["layer"], batch_size)
    return None if st is None or st["K"] != K or K + 2 > COUNTS_STRIDE else _counts_rows(st, t)


def _counts_rows(st: dict, t: int) -> torch.Tensor:
    buf = _counts(st)
    st["written"].add(t)                                           # (its rows the sum's kernels' to write)
    return buf[st["offs"][t]:st["offs"][t + 1]]


def _counts(st: dict) -> torch.Tensor:
    # (not zeroed: a leaf's rows are written whole -- the posterior and the flow, entries 0 .. K + 1 of a node, the
    # rest never read -- by the sum that reads it, before the leaf's backward; `written` the leaves whose rows a sum
    # took, the others' zeroed there)
    if st["counts"] is None:
        R, M, _ = st["logw"].shape
        st["counts"] = torch.empty(R, M, max(st["K"] + 2, COUNTS_STRIDE), dtype = st["logw"].dtype,
                                   device = st["logw"].device)
        st["written"] = set()
    return st["counts"]


def add_counts(sib: dict, K: int, batch_size: int, a_rows: torch.Tensor) -> bool:
    """A reading sum's posterior counts of its slots `a_rows [b, M, K + 1]` at the leaf's missing tokens (the rows
    `slot_rows` named), for the leaf's backward, into its buffer as that holds them (`counts_buffer`); False where
    its layer has no forward state (`top_k._leaf_terms` then)."""
    st = _state(sib["layer"], batch_size)
    if st is None or st["K"] != K:
        return False
    t = _plan(sib["layer"])["index"][id(sib["ns"])]
    dst = _counts(st)[st["offs"][t]:st["offs"][t + 1]]
    st["written"].add(t)
    if dst.numel() > 0:
        f = a_rows.sum(dim = 2, keepdim = True)                    # (a node's flow; its slots' posterior)
        dst[:, :, :K + 1] = torch.where(f > 0, a_rows / f, torch.zeros_like(a_rows))
        dst[:, :, K + 1] = f[:, :, 0]
    return True



def _vec(*ts) -> bool:
    """Whether the kernels take these `[b, M, re]` tensors' rows 16 B at a time: their ranks contiguous and a multiple
    of 4, every row 16-B aligned."""
    return all(t.stride(2) == 1 and t.size(2) % 4 == 0 and t.stride(0) % 4 == 0 and t.stride(1) % 4 == 0 and
               t.data_ptr() % 16 == 0 for t in ts)


def _tilts(tilts, tau: torch.Tensor) -> None:
    """tau [T, Bx, S, M] = log2 e (A B)^T of each position's tensors `(A [Bx, M, re], B [Bx, re, S])`: one launch a run
    of positions whose tensors share their strides (`top_k_leaf_tau`), else cuBLAS's product (the scale on B)."""
    tm = tilt_module()
    ops = [tuple(x.float() for x in tl) for tl in tilts]
    fits = tm is not None and ops[0][0].size(2) <= 64 and tau.size(2) <= 65 and tau.size(3) % 128 == 0
    k = 0
    while k < len(ops):
        if not fits:
            A, Bt = ops[k]
            torch.bmm((Bt * _L2E).transpose(1, 2), A.transpose(1, 2), out = tau[k])
            k += 1
            continue
        j = k + 1
        while j < len(ops) and all(x.stride() == y.stride() for x, y in zip(ops[j], ops[k])):
            j += 1
        run = ops[k:j]
        tm.top_k_leaf_tau(tau[k:j], [o[q].data_ptr() for q in range(2) for o in run],
                          [s for x in run[0] for s in x.stride()], run[0][0].size(2), all(_vec(o[0]) for o in run))
        k = j


def _tilt_grads(nodes, graded, dtau, staged, gviews, fresh, emap) -> None:
    """The tilt's gradients from dtau [T, B or Bx (folded), S, M] (a row of destinations each, reading the row of
    tensors `emap` names -- None: its own): one pass over dtau and the factors a run of positions whose operands share
    their strides (`top_k_leaf_tilt_grads`), a destination STORED whole by its first writer of the pass (`claim`: not
    zeroed first) and added into by the others; else cuBLAS's products into them."""
    tm = tilt_module()
    dests = [gviews.get(n["ns"]) for n in graded]
    store = [bool(fresh) and gviews.claim(g) for g in dests]
    idx = [nodes.index(n) for n in graded]
    ops = [tuple(x.float() for x in staged[n["ns"]]) + tuple(g) for n, g in zip(graded, dests)]
    A0, B0 = ops[0][0], ops[0][1]
    fits = (tm is not None and A0.size(2) <= 64 and B0.size(2) <= 65 and A0.size(1) % 64 == 0 and
            dtau.dtype == torch.float32 and all(x.dtype == torch.float32 for o in ops for x in o))
    em = torch.empty(0, dtype = torch.int32, device = dtau.device) if emap is None else emap.to(torch.int32).contiguous()
    k = 0
    while k < len(ops):
        if not fits:
            t = idx[k]
            A, Bt, gA, gB = ops[k]
            if store[k]:
                torch._foreach_zero_([gA, gB])
            if emap is not None:
                A, Bt = A.index_select(0, emap), Bt.index_select(0, emap)
            D = dtau[t].transpose(1, 2)                                                     # [B or Bx, M, S]
            gA.baddbmm_(D, Bt.transpose(1, 2))
            gB.baddbmm_(A.transpose(1, 2), D)
            k += 1
            continue
        # a run of consecutive positions with the same strides and the same first writing, no destination twice (one
        # launch: its CTAs write their rows unsynchronized)
        j = k + 1
        while (j < len(ops) and idx[j] == idx[j - 1] + 1 and store[j] == store[k] and
               all(x.stride() == y.stride() for x, y in zip(ops[j], ops[k])) and
               all(ops[j][2].data_ptr() != o[2].data_ptr() for o in ops[k:j])):
            j += 1
        run = ops[k:j]
        tm.top_k_leaf_tilt_grads(dtau[idx[k]:idx[j - 1] + 1], [o[q].data_ptr() for q in range(4) for o in run],
                                 [s for x in run[0] for s in x.stride()], em, A0.size(2), store[k],
                                 all(_vec(o[0], o[2]) for o in run))
        k = j


def _head_grad_args(nodes, graded, staged, gviews, fresh, st, Bd: int):
    """The tilt's gradients with the head's walk in them (`tk_leaf_head_grad_kernel`: no dtau made, written and read
    back) -- where every position is graded, the gradients fold into the rows of tensors the evidence was given at (a
    row's samples then one tile: the fold's rows are the forward's map) and the operands fit the kernel --: `(ptrs,
    strides, ranks, stores)`, the destinations claimed; else None (dtau then, and `_tilt_grads`)."""
    # (the sums' top_k the leaves' own -- K != Kl's slots, past the tests' reach, keep dtau --; the kernel's head: 80
    # candidates)
    if not graded or len(graded) != len(nodes) or len(nodes) > 64 or st["rowstart"].numel() != Bd + 1 or \
            st["K"] != st["S"] - 1 or min(st["ids"].size(2), st["K"]) > 80:
        return None
    dests = [gviews.get(n["ns"]) for n in graded]
    ops = [tuple(staged[n["ns"]]) + tuple(g) for n, g in zip(graded, dests)]
    A0, B0 = ops[0][0], ops[0][1]
    if not (A0.size(2) <= 64 and A0.size(2) % 4 == 0 and B0.size(2) <= 65 and A0.size(1) % 64 == 0 and
            all(x.dtype == torch.float32 for o in ops for x in o) and
            all(x.stride() == y.stride() for o in ops for x, y in zip(o, ops[0])) and all(_vec(o[0], o[2]) for o in ops)
            and len({o[2].data_ptr() for o in ops}) == len(ops) and len({o[3].data_ptr() for o in ops}) == len(ops)):
        return None
    stores = [bool(fresh) and gviews.claim(g) for g in dests]
    return [o[q].data_ptr() for q in range(4) for o in ops], [s for x in ops[0] for s in x.stride()], A0.size(2), stores


def leaf_backward(desc, layer, data, node_flows: torch.Tensor, kwargs: dict, logspace_flows: bool, missing_mask,
                  batch_first: bool, compute_param_flows: bool) -> bool:
    """`top_k.leaf_backward` by the kernels, from the layer's forward state and the reading sums' counts; False where
    the forward left none (the torch passes then)."""
    from .top_k import _staged, _grads_kw, external_rows, evidence_rows
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
    if counts.numel() > 0:                                             # (the rows of a leaf no sum wrote: none read)
        for t in range(len(st["offs"]) - 1):
            if st["offs"][t + 1] > st["offs"][t] and t not in st["written"]:
                counts[st["offs"][t]:st["offs"][t + 1]].zero_()
    em = compute_param_flows and layer.param_flows is not None
    pm = kwargs.get("param_flows_mask") if em else None                 # (the samples the EM statistics take)
    # the tilt's gradients' dtau: where any is wanted; where the destinations fold, each sample's times its sign summed
    # into the row it read by the kernel -- a row's gradients then one product
    graded = [n for n in nodes if gviews is not None and gviews.get(n["ns"]) is not None]
    fold = getattr(gviews, "fold", None) if graded else None
    drow, dsgn, Bd = none.int(), none, 0
    if fold is not None:
        drow, dsgn, Bd = fold[0].to(torch.int32).contiguous(), fold[1].float().contiguous(), staged[graded[0]["ns"]][0].size(0)
    # (evidence of the rows' batch: each sample's gradient into the row it read, times its sign where the backward folds)
    gsgn = none
    if glp is not None and evidence_rows(kwargs) is not None:
        from pyjuice.layer.parameterized_sum_layer import EXTERNAL_PARAMS_GRAD_SIGNS_KWARG
        sg = kwargs.get(EXTERNAL_PARAMS_GRAD_SIGNS_KWARG)
        gsgn = torch.ones(node_flows.size(1), device = dev) if sg is None else sg.float().contiguous()
    # (the gradients folded, the evidence a row each: the head's walk in the tilt's gradients, no dtau)
    hg = _head_grad_args(nodes, graded, staged, gviews, fresh, st, Bd) \
        if fold is not None and st["ids"].size(0) != node_flows.size(1) else None
    tptrs, tstrides, re, tstores = hg if hg is not None else ([], [], 0, [])
    tau, tauK = st["tau"], st["tauK"]
    if tau is None:                                                    # (made in the forward's kernel: made again)
        tau = torch.empty(len(nodes), tauK.size(1), st["S"], tauK.size(2), dtype = torch.float32, device = dev)
        _tilts([staged[n["ns"]] for n in nodes], tau)
    dtau = mod.top_k_leaf_backward(st["tab"], plan["nv_tab"], tau, st["emap"], st["ids"], st["lp"], plan["nv_lvid"],
                                   plan["nv_row0"], st["lz"], st["lb"], st["place"], node_flows, logspace_flows,
                                   st["wrow"], st["logw"], counts, st["S"], st["K"], glp if glp is not None else none,
                                   layer.param_flows if em else none, plan["spf"] if em else none.long(),
                                   plan["spf_aligned"], none.bool() if pm is None else pm.contiguous(), gsgn,
                                   bool(graded), drow, dsgn, Bd, st["ord"], st["rank"], tptrs, tstrides,
                                   st["rowstart"] if hg is not None else none.int(), re, tstores, st["crow"],
                                   none if tauK is None else tauK)
    if glp is not None and glp is not glp_out:
        glp_out += glp.to(glp_out.dtype)
    if graded and hg is None:
        _tilt_grads(nodes, graded, dtau, staged, gviews, fresh, None if fold is not None else external_rows(kwargs))
    # the forward's state used: released, nothing of the pass held after it (MEASURED, 48 blocks: 1.3 GB -- the tables,
    # the tilts, the slots' log-weights). A second backward of the same forward takes the torch passes (no state).
    layer.__dict__.pop(_STATE, None)
    return True
