"""
`LowRank`' "sum_exp" form on a VIRTUAL BATCH: the plain sum-layer kernels do the work.

The mixture scale `g_b[i, j] = sum_r exp(A_b[i, r]) exp(B_b[r, j])` is, per rank, a rescaling of the children by
`exp(B_b[r, :])` followed by a plain sum node, so each (sample, rank) pair becomes a column of a virtual batch for the
node's unnormalized value, and the layer's own forward and backward run on it as they stand: the shared parameters are
read once for all the columns, every kernel, tuning and launch plan of the plain layer applies, and the pass stays in
the regime where its cost barely depends on the batch (MEASURED, a tied HMM of 1024 latents and 31 layers, forward /
backward: batch 16 0.64 / 1.20 ms, 128 0.53 / 1.23, 256 0.60 / 1.63). What is specific to the parameterization is
elementwise and lives in `kernels/c/lowrank_mixture.cu`, whose header has the formulas.

The node's normalizer reads no child values and tied copies share the parameters: it is not run through the plain kernels
at all. For a node vector and every tied copy of it the pass was given, before the first of them runs, each rank's
normalizer is ONE matrix product over the rows `(copy, rank, sample)` (`_normalizers`).

The virtual buffers are `K` times as wide as the circuit's and scratch, rewritten by each layer's pass. The element ones
are as tall as the circuit's (whose element rows every product layer reuses: one layer's worth); the NODE ones are one
layer's -- a few dummy rows, then the layer's nodes --, the plain kernels run on the layer's TWIN (`_twin`): a plain
`SumLayer` of its compiled tables, the node rows shifted to match, its caches its own. What a layer's backward needs of
its forward -- `log Z` and each rank's share of the two sums -- is kept per node vector, on the layer.

The backward's normalizer half reads no child values and tied copies share the parameters: it is not run through
the plain kernels at all. For a node vector and every tied copy of it in the pass, at the pass's end, it is TWO MATRIX
PRODUCTS over the rows `(copy, sample, rank)` (`kernels/c/lowrank_mixture.cu`'s header has them): `W = U^T V`, the
denominator statistic of all the copies at once, and `U theta`, whose product with `V` is the normalizer half's part
of the factors' gradients -- cuBLAS, bf16 operands with fp32 accumulation under the default precision where the
product is large, fp32 else or asked for. The data half's parameter flows are a third product where the layer's
product layer is fused in. (What a pass is: the staged tensors of one forward, on which the copies that went through
here are noted.)

Served: CUDA, the extension built, no child of the layer shared by two of its node vectors (each would want its own
rescaling of it), the virtual buffers within `VIRTUAL_BYTES_MOST`. Every other case keeps the torch passes of
`lowrank.py`, which are also what these are tested against.
"""

from __future__ import annotations

import weakref
from typing import Optional

import torch

#: `False`: the torch passes always (the reference the virtual batch is tested against)
ENABLED = True
#: The most bytes the virtual buffers may take; a pass that would need more keeps the torch passes
VIRTUAL_BYTES_MOST = 4 << 30
#: The normalizer half's matrix products take bf16 operands (fp32 accumulation) from this many multiply-adds on, under
#: the default precision (the plain kernels' own rule for their operands: `TensorCircuit._COPY_MIN_WORK`)
GEMM_BF16_WORK = 1 << 28

_EXTERNAL_PARAMS_KWARG = "sum_external_params"


_MODULE = None


def _module():
    """The extension, or None. (Kept: a `from ... import` is 1.7 us, and a pass asks about 60 times.)"""
    global _MODULE
    if _MODULE is None:
        from .kernels.c import get_lowrank_module
        _MODULE = get_lowrank_module()
    return _MODULE


def _columns(layer, node_mars: torch.Tensor) -> int:
    return node_mars.size(1) * layer.parameterization.rank


def _bytes(node_rows: int, element_mars: torch.Tensor, cols: int) -> int:
    """Every family of `_buffers` a layer's pass takes at once: the forward's values; the backward's (flows, values,
    zeros) -- the node rows the layer's (`_twin`), the element rows the circuit's."""
    return 4 * cols * (3 * node_rows + 3 * element_mars.size(0))


def serves(desc, layer, node_mars: torch.Tensor, element_mars: torch.Tensor) -> bool:
    """Whether this layer's pass at this batch goes through the virtual batch."""
    if not ENABLED or desc.form != "sum_exp" or not node_mars.is_cuda or node_mars.dtype != torch.float32:
        return False
    if _module() is None or not _disjoint(desc, layer, node_mars.device):
        return False
    rows = _twin(layer)[0]._layer_nid_range[1]
    return _bytes(rows, element_mars, _columns(layer, node_mars)) <= VIRTUAL_BYTES_MOST


def note_compiled(layer) -> None:
    """`LowRank.compile`'s: which attributes of the layer, as compilation leaves it, are caches -- an empty container or
    None: the twin gets a fresh one -- and which are its compiled tables and settings, which the twin shares."""
    caches, compiled = dict(), []
    for k, v in layer.__dict__.items():
        if v is None or (isinstance(v, (dict, list, set)) and len(v) == 0):
            caches[k] = type(v)
        else:
            compiled.append(k)
    layer.__dict__["_lr_compiled"] = (tuple(compiled), caches)


def _twin(layer):
    """`(twin, shift)`: the layer's TWIN, on which the virtual batch runs the plain kernels -- a plain `SumLayer` with
    the layer's compiled tables and settings (shared), caches of its own (fresh: what the layer's would hold is derived
    from other node rows), and node ids `shift` rows lower: each id of the layer's own nodes, `nids` and `parids`
    alike; a dummy id, below the layer's rows, unchanged -- so the virtual node buffers hold the dummy rows `[0, nd)`,
    then the layer's nodes, a node's row there its row in the circuit's less `shift`. `nd` covers every dummy block the
    tables name, one block at least, a multiple of 32. Made when first asked for, on the tables' device; dropped with
    the layer's device caches (`LowRank.drop_device_caches`)."""
    hit = layer.__dict__.get("_lrv_twin")
    if hit is not None:
        return hit
    from pyjuice.layer.sum_layer import SumLayer
    import torch.nn as nn

    compiled, caches = layer.__dict__["_lr_compiled"]
    twin = SumLayer.__new__(SumLayer)
    own = twin.__dict__
    for k in compiled:
        if k in layer.__dict__:
            own[k] = layer.__dict__[k]
    for k, kind in caches.items():
        own[k] = kind()

    lo, hi = layer._layer_nid_range
    nids, parids = list(layer.partitioned_nids), list(layer.partitioned_parids)
    if not all(bool(((t >= lo) & (t < hi)).all()) for t in nids) or not all(bool((t < hi).all()) for t in parids):
        raise RuntimeError("LowRank: a sum layer's tables name nodes past its own rows.")
    extent = max((int(t[t < lo].max()) + layer.block_size for t in parids if bool((t < lo).any())), default = 0)
    nd = -(-max(32, layer.block_size, extent) // 32) * 32
    shift = lo - nd
    lists = type(layer.partitioned_nids)
    shifted = lambda ts, dummies: lists([nn.Parameter((torch.where(t >= lo, t - shift, t) if dummies else t - shift)
                                                      .contiguous(), requires_grad = False) for t in ts])
    modules = type(layer._modules)(layer._modules)
    modules["partitioned_nids"], modules["partitioned_parids"] = shifted(nids, False), shifted(parids, True)
    own["_modules"] = modules
    own["_layer_nid_range"] = (lo - shift, hi - shift)
    hit = layer.__dict__["_lrv_twin"] = (twin, shift)
    return hit


def _disjoint(desc, layer, device) -> bool:
    """Whether no child of the layer is shared by two of its node vectors: one would need two rescalings in one row
    of the virtual buffer."""
    ok = layer.__dict__.get("_lrv_disjoint")
    if ok is None:
        rows = torch.cat([desc._geometry(layer, info, device)["kids"] for info in layer.external_node_infos])
        ok = layer.__dict__["_lrv_disjoint"] = bool(torch.unique(rows).numel() == rows.numel())
    return ok


# The families of virtual buffers: name -> (rows of nodes (True) or elements, columns in units of a half)
_FAMILIES = {
    "fw": dict(vnm = (True, 1), vem = (False, 1)),
    "bw": dict(vnf_d = (True, 1), vem_d = (False, 1), vef_d = (False, 1), zero = (True, 1)),
}


def _buffers(node_mars: torch.Tensor, element_mars: torch.Tensor, cols: int, family: str,
             node_rows: Optional[tuple] = None) -> dict:
    """The virtual buffers beside `node_mars` for `cols` columns a half, of one family (allocated when first asked
    for): the element ones as tall as `element_mars`, one set per (family, columns) -- the layer below writes the next
    layer's (`_link`) --, the node ones for a layer's twin's rows (`_twin`: `node_rows` its `_layer_nid_range`, the
    dummy rows before it; None: the element ones only), one set per (family, columns, rows): the layers whose dummy rows
    and nodes lie alike take turns, none writing another's dummy rows. One dict a key, the element ones in it the
    same tensors whatever the rows (one lookup a call: the pass is host-bound at small shapes). Kept ON that tensor:
    a circuit's rows are its own -- which are dummies, which a layer's --, and it makes another buffer when its batch
    changes, these gone with the old one."""
    store = node_mars.__dict__.get("_lowrank_virtual")
    if store is None:
        store = node_mars.__dict__["_lowrank_virtual"] = dict()
    key = (family, cols, node_rows)
    made = store.get(key)
    if made is None:
        # Filled once, as the circuit fills its own: the plain kernels pad a ragged row's edges with DUMMY rows -- the
        # first elements, of value 0 (-inf), the first nodes, of flow 0 -- which nothing here ever writes. (Node
        # values: 0, which the backward's stand-ins are throughout.)
        made, dev = dict(), node_mars.device
        if node_rows is not None:
            made.update(_buffers(node_mars, element_mars, cols, family))
        for name, (of_nodes, width) in _FAMILIES[family].items():
            if of_nodes != (node_rows is not None):
                continue
            shape = (node_rows[1] if of_nodes else element_mars.size(0), width * cols)
            zero = name in ("vnm", "zero")
            made[name] = torch.zeros(shape, device = dev) if zero else torch.full(shape, -float("inf"), device = dev)
        store[key] = made
    return made


def _state(layer, info, batch: int, device) -> tuple:
    """`(log Z [M, B], each rank's log share of S [M, K, B], of Z [M, K, B])` of one node vector -- batch innermost,
    as the factors are stored: written by its forward, read by its backward."""
    states = layer.__dict__.setdefault("_lrv_state", {})
    key = id(info.ns)
    hit = states.get(key)
    M, K = info.ns.num_nodes, layer.parameterization.rank
    if hit is None or hit[0].size(1) != batch or hit[0].device != device:
        hit = states[key] = (torch.empty(M, batch, device = device), torch.empty(M, K, batch, device = device),
                             torch.empty(M, K, batch, device = device))
    return hit


def _pass(kwargs) -> dict:
    """What is kept of a PASS: `ran`, the node vectors whose forward went through here; of its backward `done`, the
    ones whose backward has, and `norm`, per source of tied copies, what the normalizer half needs of each copy that
    ran (its layer, node vector, factors, kept `log Z`, gradient buffers). Kept on the staged tensors of the forward,
    which every layer of the pass and of its backward is handed. A layer's forward may run more than once on them --
    the circuit times its fused forward on the live pass (`TensorCircuit._stream_fusion_groups`) -- so `ran` is a set
    that only grows."""
    staged = kwargs[_EXTERNAL_PARAMS_KWARG]
    st = staged.__dict__.get("_lowrank_pass")
    if st is None:
        st = staged.__dict__["_lowrank_pass"] = dict(ran = dict(), done = set(), norm = dict())
    return st


def _source(ns):
    return ns.get_source_ns() if ns.is_tied() else ns


def _dtype(precision, work: int):
    """The operands of the pass's matrix products: bf16 (fp32 accumulation) under the default precision where the work
    is `GEMM_BF16_WORK` or more, fp32 else."""
    return torch.bfloat16 if precision != "fp32" and work >= GEMM_BF16_WORK else torch.float32


def _product(a, b, dtype):
    return torch.mm(a, b, out_dtype = torch.float32) if dtype is torch.bfloat16 else torch.mm(a, b)


def fuses(desc, layer, children) -> bool:
    """Whether a forward of this layer through the virtual batch takes the product group below fused into its
    `expand` (`children`: the group's `(table, first_row)`): products of two children, as the fused kernels of the
    plain layer take them. (Asked by the circuit's decision, without the pass's buffers: a pass that turns out too
    large for the virtual buffers forms the products' values itself, `LowRank.forward_layer`.)"""
    table = children[0]
    if not ENABLED or desc.form != "sum_exp" or not table.is_cuda or table.size(1) != 2 or _module() is None:
        return False
    return _disjoint(desc, layer, table.device)


def fuses_bk() -> bool:
    """Whether a backward through the virtual batch takes the product group below fused in: its `backward_pre` reads
    each product's value from its two children, its `collapse_bw` writes the product's flow as theirs. (For a layer
    whose forward went through here: the extension is built, no child is shared.)"""
    return ENABLED and _module() is not None


#: How many forwards a layer handed the product group below looks for the layer under it (`_link`) before it stops
LINK_TRIES = 64
#: How many times a layer's virtual child values may be written by the layer below for nothing -- another layer
#: wrote the buffer in between -- before that stops
WASTED_MOST = 2


def _link(desc, layer, ran, stream_children, device) -> None:
    """A layer handed the product group below (`stream_children`) looks for the layer whose nodes are its products'
    children: that layer's `collapse` can then write this layer's virtual child values in the same launch
    (`lowrank_collapse_expand`), every pass after. The layer is the one holding the highest child row, among those
    that went through the virtual batch on this circuit's buffers (`ran`); every other child then lies below that
    layer's rows -- computed before it runs, rows being handed out in the order the layers run in. A node vector of
    this layer is PAIRED with the one of that layer holding all of its children that layer has (its launch reads
    those from its own collapse, nothing another launch of that layer writes); one without such a pair gets its
    `expand` after that layer's collapses. Kept on the layer below (`_lrv_link`); looked for until found, `LINK_TRIES`
    forwards at most (the buffers may be new ones, on which that layer has not run yet)."""
    table, first = stream_children
    want = layer.__dict__.get("_lrv_want")
    if want is None or want["table"] is not table:
        kids = []
        for info in layer.external_node_infos:
            rows = desc._geometry(layer, info, device)["rows"]
            kids.append(torch.unique(table[rows[rows >= 0] - first].long()))
        want = layer.__dict__["_lrv_want"] = dict(
            table = table, kids = kids, tries = LINK_TRIES,
            top = max((int(k.max()) for k in kids if k.numel() > 0), default = -1))
    if want["tries"] <= 0:
        return
    top = want["top"]
    below = next((other for other in ran.values()
                  if other is not layer and other._layer_nid_range[0] <= top < other._layer_nid_range[1]), None)
    if below is None:
        want["tries"] -= 1
        return
    want["tries"], kids, want["kids"] = 0, want["kids"], None
    if below.__dict__.get("_lrv_link") is not None:                    # (taken: one layer above a layer)
        return
    lo = below._layer_nid_range[0]
    pairs, rest = dict(), []
    for at, k in enumerate(kids):
        inside = k[k >= lo]
        mine = None if inside.numel() == 0 else next(
            (info for info in below.external_node_infos
             if info.nid_start <= int(inside.min()) and int(inside.max()) < info.nid_end), None)
        if mine is None or mine.nid_start in pairs:
            rest.append(at)
        else:
            pairs[mine.nid_start] = at
    below.__dict__["_lrv_link"] = dict(above = weakref.ref(layer), table = table, first = first, pairs = pairs,
                                       rest = rest)


def _v_rows(mod, fbs, N, K, batch, dtype, dev) -> tuple:
    """`(V^T [N, rows], each row's shift)`: `exp(B - shift)`, the rows `(copy, rank, sample)`, the shift the row's largest
    `B`."""
    rows = len(fbs) * batch * K
    Vt = torch.empty(N, rows, dtype = dtype, device = dev)
    shift = torch.empty(rows, device = dev)
    mod.lowrank_z_operand(Vt, shift, fbs, batch, K)
    return Vt, shift


def _normalizers(desc, layer, info, st, kwargs, params, batch, dev) -> dict:
    """What a pass keeps of a node vector's source, made when the first of its copies runs. For every tied copy the pass
    was given, each rank's normalizer is ONE matrix product `theta @ exp(B - shift)` over the rows `(copy, rank, sample)`,
    `log_z`, the plain kernels then never seeing the normalizer half; `row`: each copy's first row. The product's operands
    are the backward's too: `V`, `shift` and `theta`, and `fb`, the factors. `theta` dense in the product's dtype: at
    bf16, from the circuit's bf16 copy where the forward was handed it (`op_params`, the layer marked for it: its
    source's range copied in this forward); else from the parameters, by one strided copy that converts."""
    source = _source(info.ns)
    zs = st.setdefault("z", dict())
    hit = zs.get(source)
    if hit is None or info.ns not in hit["row"]:
        copies = [(c, t[1]) for c, t in kwargs[_EXTERNAL_PARAMS_KWARG].items() if _source(c) is source]
        K, mod, geo = desc.rank, _module(), desc._geometry(layer, info, dev)
        fbs = [fb for _, fb in copies]
        dtype = _dtype(kwargs.get("precision", "auto"), len(copies) * batch * K * geo["M"] * geo["N"])
        Vt, shift = _v_rows(mod, fbs, geo["N"], K, batch, dtype, dev)
        op = kwargs.get("op_params")
        theta = desc._dense_params(geo, op if op is not None and dtype is torch.bfloat16 else params, dtype)
        Zt = _product(theta, Vt, dtype)
        mod.lowrank_z_log(Zt, shift)
        hit = zs[source] = dict(log_z = Zt, row = {c: at * batch * K for at, (c, _) in enumerate(copies)}, V = Vt,
                                shift = shift, dtype = dtype, fb = fbs, theta = theta)
    return hit


def _above(layer, kwargs, node_mars, element_mars):
    """`(the layer above, its node vectors' tensors, its forward buffers)` where this layer's collapse takes that
    layer's `expand` along in this pass (`_link`): that layer goes through the virtual batch at this batch too, on
    tensors the circuit staged for all of its node vectors. Else None."""
    link = layer.__dict__.get("_lrv_link")
    if link is None:
        return None
    from pyjuice.layer.parameterized_sum_layer import StagedExternalParams
    above, staged = link["above"](), kwargs[_EXTERNAL_PARAMS_KWARG]
    if above is None or not isinstance(staged, StagedExternalParams) or link["table"].device != node_mars.device \
            or above.__dict__.get("_lrv_wasted", 0) >= WASTED_MOST \
            or not serves(above.parameterization, above, node_mars, element_mars):
        return None
    tensors = [staged.get(info.ns) for info in above.external_node_infos]
    if any(t is None for t in tensors):
        return None
    return above, tensors, _buffers(node_mars, element_mars, _columns(above, node_mars), "fw")


def forward(desc, layer, ns_tensors, node_mars, element_mars, params, kwargs, stream_children = None) -> None:
    from pyjuice.layer.sum_layer import SumLayer

    mod, K, batch, dev = _module(), desc.rank, node_mars.size(1), node_mars.device
    cols = batch * K
    twin, shift = _twin(layer)
    bufs = _buffers(node_mars, element_mars, cols, "fw", twin._layer_nid_range)
    st = _pass(kwargs)

    # The virtual child values: written already where the layer below took them along with its collapse. The mark it
    # left names this layer; whichever layer comes through here next takes the mark away -- another's `expand` writes
    # the same buffer, the values written for nothing then. Counted against the pair where that other layer comes
    # AFTER the one below (its rows are handed out later): one before it is the circuit's trials running a layer
    # group again, not a layer between the two.
    mark = st.pop("ahead", None)
    if mark is not None and mark[0] is not layer and mark[1] is not layer \
            and layer._layer_nid_range[0] > mark[1]._layer_nid_range[0]:
        mark[0].__dict__["_lrv_wasted"] = mark[0].__dict__.get("_lrv_wasted", 0) + 1
    if mark is None or mark[0] is not layer:
        for info, (_, fb) in ns_tensors:
            rows = desc._geometry(layer, info, dev)["rows"]
            if stream_children is None:
                mod.lowrank_expand(bufs["vem"], element_mars, fb, rows, K)
            else:                                                       # (the product layer's forward was not run)
                mod.lowrank_expand_fused(bufs["vem"], node_mars, stream_children[0], stream_children[1], fb, rows, K)
    ran = node_mars.__dict__.setdefault("_lowrank_layers", dict())      # (the layers through here on these buffers)
    if stream_children is not None:
        _link(desc, layer, ran, stream_children, dev)

    # (the circuit's bf16 copy of the tied parameters, where the circuit marked the layer for it: the plain kernels'
    # operands, as for a plain layer)
    twin.__dict__["_reads_param_operands"] = layer.__dict__.get("_reads_param_operands", False)
    SumLayer.forward(twin, bufs["vnm"], bufs["vem"], params, precision = kwargs.get("precision", "auto"),
                     op_params = kwargs.get("op_params"))

    # The node values -- and, in the same launch, the virtual child values of the layer above (`_link`)
    nxt = _above(layer, kwargs, node_mars, element_mars)
    if nxt is None:
        for info, (fa, _) in ns_tensors:
            z = _normalizers(desc, layer, info, st, kwargs, params, batch, dev)
            mod.lowrank_collapse(node_mars, bufs["vnm"], fa, *_state(layer, info, batch, dev), z["log_z"], z["row"][info.ns],
                                 info.nid_start, info.nid_start - shift, info.ns.num_nodes, K)
    else:
        above, tensors, bufa = nxt
        link, adesc = layer.__dict__["_lrv_link"], above.parameterization
        infos = above.external_node_infos
        geo = lambda at: adesc._geometry(above, infos[at], dev)["rows"]
        for info, (fa, _) in ns_tensors:
            at = link["pairs"].get(info.nid_start)
            z = _normalizers(desc, layer, info, st, kwargs, params, batch, dev)
            lzr, row0 = z["log_z"], z["row"][info.ns]
            if at is None:
                mod.lowrank_collapse(node_mars, bufs["vnm"], fa, *_state(layer, info, batch, dev), lzr, row0,
                                     info.nid_start, info.nid_start - shift, info.ns.num_nodes, K)
            else:
                mod.lowrank_collapse_expand(node_mars, bufs["vnm"], fa, *_state(layer, info, batch, dev), lzr, row0,
                                            info.nid_start, info.nid_start - shift, info.ns.num_nodes, K, bufa["vem"],
                                            link["table"],
                                            link["first"], tensors[at][1], geo(at), adesc.rank)
        for at in link["rest"]:
            mod.lowrank_expand_fused(bufa["vem"], node_mars, link["table"], link["first"], tensors[at][1], geo(at),
                                     adesc.rank)
        st["ahead"] = (above, layer)

    for info, _ in ns_tensors:
        st["ran"][info.ns] = True
    ran[id(layer)] = layer
    layer.__dict__["_lr_state"] = ("virtual", batch)
    return None


def _end_of_pass(mod, z, copies, node_flows, node_mars, params, param_flows, denom_param_flows, want_denom, want_grads,
                 precision, batch) -> None:
    """What a node vector and its copies in the pass owe at the pass's end, as matrix products over the rows `(copy,
    rank, sample)` (`kernels/c/lowrank_mixture.cu`'s header has them), of cuBLAS: bf16 operands (fp32 accumulation) under
    the default precision where the work is `GEMM_BF16_WORK` or more, fp32 else.

    The normalizer half: `W = U^T V` into the denominator statistic (the first copy's slice: `materialize_denom_flows`
    sums the copies'), `V * (U theta)` off each copy's `dB`, `V` and the rows the forward's (`z`, `_normalizers`). The
    data half: `F+ = theta * (U^T V)`, `theta` read from `params` where they lie, for the copies whose product layer was
    fused in (`copies`' last entry; another's plain backward has added its flows), into the first of them's flow block:
    the blocks of a source's copies are summed before the parameters read them, as those of the plain kernels.
    (`copies`: one source's, of one parameterization: its rank, its denominator or none.)"""
    layer0, info0, _, _, _, _ = copies[0]
    desc, K = layer0.parameterization, layer0.parameterization.rank
    dev = node_flows.device
    geo = desc._geometry(layer0, info0, dev)
    M, N = geo["M"], geo["N"]
    fp = [] if param_flows is None else [c for c in copies if c[5] is not None]
    dtype = _dtype(precision, len(z["fb"]) * batch * K * M * N)

    if fp:
        rows = len(fp) * batch * K
        Ud, Vd = torch.empty(M, rows, dtype = dtype, device = dev), torch.empty(N, rows, dtype = dtype, device = dev)
        mod.lowrank_data_operands(Ud, Vd, node_flows, node_mars, [fa for _, _, (fa, _), _, _, _ in fp],
                                  [lz for _, _, _, lz, _, _ in fp], [fb for _, _, (_, fb), _, _, _ in fp],
                                  [info.nid_start for _, info, _, _, _, _ in fp],
                                  [desc._geometry(layer, info, dev)["rows"] for layer, info, _, _, _, _ in fp],
                                  [sc[0] for _, _, _, _, _, sc in fp], [sc[1] for _, _, _, _, _, sc in fp], M, K)
        desc._scatter_add(geo, param_flows, fp[0][1].ns._param_flow_range[0], _product(Ud, Vd.t(), dtype), params)

    if not (want_denom or want_grads):
        return
    Vt, shift = (z["V"], z["shift"]) if dtype is z["dtype"] else _v_rows(mod, z["fb"], N, K, batch, dtype, dev)
    # (a copy the forward was given that did not come through here: no rows of U, so nothing of it in the products)
    Ut = (torch.empty if len(copies) == len(z["row"]) else torch.zeros)(M, Vt.size(1), dtype = dtype, device = dev)
    row0 = [z["row"][info.ns] for _, info, _, _, _, _ in copies]
    mod.lowrank_u_operand(Ut, shift, node_flows, [fa for _, _, (fa, _), _, _, _ in copies],
                          [lz for _, _, _, lz, _, _ in copies], [info.nid_start for _, info, _, _, _, _ in copies], row0, M, K)
    if want_denom:
        desc._scatter_add(geo, denom_param_flows, desc._denom_offsets(layer0)[id(info0.ns)][0],
                          _product(Ut, Vt.t(), dtype))
    if want_grads:
        theta = z["theta"] if z["theta"].dtype == dtype else desc._dense_params(geo, params, dtype)
        Gt = _product(theta.t(), Ut, dtype)                                  # [N, rows]: (U theta)^T
        mod.lowrank_normalizer_grads([None if grads is None else grads[1] for _, _, _, _, grads, _ in copies], row0, Vt, Gt,
                                     batch, K)


def backward(desc, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows, node_mars, element_mars, params,
             param_flows, denom_param_flows, kwargs, stream_children = None) -> None:
    """`stream_children`: None, or the product group below's `(table, first_row)` -- its forward (the recompute of
    `element_mars`) and its backward not run: the virtual child values are formed from the products' children's node
    values, and each element's flow is written as its two children's node flows, `element_flows` not touched."""
    from pyjuice.layer.sum_layer import SumLayer

    mod, K, batch, dev = _module(), desc.rank, node_mars.size(1), node_mars.device
    cols = batch * K
    twin, shift = _twin(layer)
    bufs = _buffers(node_mars, element_mars, cols, "bw", twin._layer_nid_range)
    precision = kwargs.get("precision", "auto")
    want_denom = param_flows is not None and denom_param_flows is not None and desc.apply_z_correction

    # Another backward of the same pass: a node vector's again, or one into other flows -- the circuit times its fused
    # backward on scratch flows before the pass's own (`TensorCircuit._stream_fusion_bk_groups`)
    st = _pass(kwargs)
    flows = st.get("flows")
    if flows is None or flows() is not node_flows or any(info.ns in st["done"] for info, _ in ns_tensors):
        st.update(done = set(), norm = dict(), flows = weakref.ref(node_flows))

    # The data half: the virtual child values and node flows (and dA), the plain backward, the element flows (and dB's
    # data part)
    vectors = []
    for (info, (fa, fb)), grads in zip(ns_tensors, ns_grad_tensors):
        rows = desc._geometry(layer, info, dev)["rows"]
        state = _state(layer, info, batch, dev)
        vectors.append((info, fa, fb, grads, rows, state))
        grad_a = None if grads is None else grads[0]
        if stream_children is None:
            mod.lowrank_backward_pre(bufs["vem_d"], element_mars, fb, rows, bufs["vnf_d"], node_flows, node_mars, fa,
                                     *state, grad_a, info.nid_start, info.nid_start - shift, info.ns.num_nodes, K)
        else:
            mod.lowrank_backward_pre_fused(bufs["vem_d"], stream_children[0], stream_children[1], fb, rows,
                                           bufs["vnf_d"], node_flows, node_mars, fa, *state,
                                           grad_a, info.nid_start, info.nid_start - shift, info.ns.num_nodes, K)

    # (the parameter flows: one matrix product at the pass's end where the product layer is fused in: its children's
    # values are still at hand then)
    twin.__dict__["_reads_param_operands"] = layer.__dict__.get("_reads_param_operands", False)
    SumLayer.backward(twin, bufs["vnf_d"], bufs["vef_d"], bufs["zero"], bufs["vem_d"], params,
                      param_flows = param_flows if stream_children is None else None, logspace_flows = True,
                      precision = precision, op_params = layer.__dict__.get("_bk_op_params"))

    for info, fa, fb, grads, rows, state in vectors:
        grad_b = None if grads is None else grads[1]
        if stream_children is None:
            mod.lowrank_collapse_bw(element_flows, bufs["vef_d"], rows, grad_b, K)
        else:
            mod.lowrank_collapse_bw_fused(node_flows, bufs["vef_d"], rows, stream_children[0], stream_children[1],
                                          grad_b, K)
        st["norm"].setdefault(_source(info.ns), []).append((layer, info, (fa, fb), state[0], grads, stream_children))
        st["done"].add(info.ns)

    # The pass's last node vector: every source's products, all of its copies at once (each source's own
    # parameterization: its rank, whether it keeps a denominator)
    if len(st["done"]) == len(st["ran"]):
        for src, copies in st.pop("norm").items():
            own = copies[0][0].parameterization
            denom = param_flows is not None and denom_param_flows is not None and own.apply_z_correction
            grads = any(c[4] is not None for c in copies)
            _end_of_pass(mod, st["z"][src], copies, node_flows, node_mars, params, param_flows, denom_param_flows,
                         denom, grads, precision, batch)
        st["norm"] = dict()

    return None
