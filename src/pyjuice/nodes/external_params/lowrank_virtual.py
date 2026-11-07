"""
`LowRankSumParams`' "sum_exp" form on a VIRTUAL BATCH: the plain sum-layer kernels do the work.

The mixture scale `g_b[i, j] = sum_r exp(A_b[i, r]) exp(B_b[r, j])` is, per rank, a rescaling of the children by
`exp(B_b[r, :])` followed by a plain sum node, so each (sample, rank) pair becomes two columns of a virtual batch -- one
for the node's unnormalized value, one for its normalizer -- and the layer's own forward and backward run on it as
they stand: the shared parameters are read once for all the columns, every kernel, tuning and launch plan of the plain
layer applies, and the pass stays in the regime where its cost barely depends on the batch (MEASURED, a tied HMM of
1024 latents and 31 layers, forward / backward: batch 16 0.64 / 1.20 ms, 128 0.53 / 1.23, 256 0.60 / 1.63). What is
specific to the parameterization is elementwise and lives in `kernels/c/lowrank_mixture.cu`, whose header has the
formulas.

The virtual buffers are as tall as the circuit's own (the plain kernels address rows by their ids), `K` or `2 K`
times as wide, and shared by every layer: scratch, rewritten by each layer's pass. What a layer's backward needs of
its forward -- `log Z` and each rank's share of the two sums -- is kept per node vector, on the layer.

The backward's normalizer half reads no child values, and tied copies share the parameters: where a layer holds one
node vector and other copies of it are in the pass, their normalizer columns are written into the rows of one of them
(the HOST, the first of every `COLUMNS_MOST` columns' worth) at their own column offsets, and one plain backward of the
host's layer gives the denominator flow and the gradients' normalizer part of all of them -- flushed when the columns
are full and when the pass's last copy has run. (What a pass is: the staged tensors of one forward, on which the
copies that went through here are noted.) MEASURED, the same HMM, batch 16, rank 4: 31 such calls 1.3 ms, 4 calls of
8 copies each 0.3 ms.

Served: CUDA, the extension built, no child of the layer shared by two of its node vectors (each would want its own
rescaling of it), the virtual buffers within `VIRTUAL_BYTES_MOST`. Every other case keeps the torch passes of
`lowrank.py`, which are also what these are tested against.
"""

from __future__ import annotations

import weakref

import torch

#: `False`: the torch passes always (the reference the virtual batch is tested against)
ENABLED = True
#: The most bytes the virtual buffers may take; a pass that would need more keeps the torch passes
VIRTUAL_BYTES_MOST = 4 << 30
#: The most columns of a host's normalizer half: as many tied copies' as fit share one plain backward. The plain
#: layer's cost is nearly flat in the batch up to here.
COLUMNS_MOST = 512

_EXTERNAL_PARAMS_KWARG = "sum_external_params"

# Per device: a zeroed stand-in for the parameter flows. (The virtual buffers are kept on the circuit's buffer of node
# values itself, `_buffers`.)
_FLOW_SCRATCH = dict()


def _module():
    from .kernels.c import get_lowrank_module
    return get_lowrank_module()


def _columns(layer, node_mars: torch.Tensor) -> int:
    return node_mars.size(1) * layer.external_params.rank


def _host_columns(cols: int) -> int:
    """The width of a host's normalizer buffers for `cols` columns a copy: whole copies, at least one."""
    return max(1, COLUMNS_MOST // cols) * cols


def _bytes(node_mars: torch.Tensor, element_mars: torch.Tensor, cols: int) -> int:
    """Every family of `_buffers` at once: the forward's two halves side by side; the backward's data half (flows,
    values, zeros); the normalizer half a layer of its own, and a host's."""
    hn, he, wide = node_mars.size(0), element_mars.size(0), _host_columns(cols)
    return 4 * (2 * cols * (hn + he) + cols * (2 * hn + 2 * he) + cols * (hn + 2 * he) + wide * (2 * hn + 2 * he))


def serves(desc, layer, node_mars: torch.Tensor, element_mars: torch.Tensor) -> bool:
    """Whether this layer's pass at this batch goes through the virtual batch."""
    if not ENABLED or desc.form != "sum_exp" or not node_mars.is_cuda or node_mars.dtype != torch.float32:
        return False
    if not _disjoint(desc, layer, node_mars.device) \
            or _bytes(node_mars, element_mars, _columns(layer, node_mars)) > VIRTUAL_BYTES_MOST:
        return False
    return _module() is not None


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
    "fw": dict(vnm = (True, 2), vem = (False, 2)),
    "bw": dict(vnf_d = (True, 1), vem_d = (False, 1), vef_d = (False, 1), zero = (True, 1)),
    "bw_z": dict(vnf_z = (True, 1), vem_z = (False, 1), vef_z = (False, 1)),
    "bw_host": dict(vnf_h = (True, 0), vem_h = (False, 0), vef_h = (False, 0), zero_h = (True, 0)),
}


def _buffers(node_mars: torch.Tensor, element_mars: torch.Tensor, cols: int, family: str) -> dict:
    """The virtual buffers beside `node_mars` for `cols` columns a half, of one family (allocated when first asked
    for). Kept ON that tensor: a circuit's rows are its own -- which are dummies, which a layer's --, and it makes
    another buffer when its batch changes, these gone with the old one."""
    store = node_mars.__dict__.get("_lowrank_virtual")
    if store is None:
        store = node_mars.__dict__["_lowrank_virtual"] = dict()
    made = store.get((family, cols))
    if made is None:
        # Filled once, as the circuit fills its own: the plain kernels pad a ragged row's edges with the circuit's
        # DUMMY rows -- the first elements, of value 0 (-inf), the first nodes, of flow 0 -- which nothing here ever
        # writes. (Node values: 0, which the backward's stand-ins are throughout.)
        made, dev = dict(), node_mars.device
        for name, (nodes, width) in _FAMILIES[family].items():
            shape = (node_mars.size(0) if nodes else element_mars.size(0), width * cols if width else _host_columns(cols))
            zero = name in ("vnm", "zero", "zero_h")
            made[name] = torch.zeros(shape, device = dev) if zero else torch.full(shape, -float("inf"), device = dev)
        made["used"] = dict()                                           # (a host's rows: the columns written last)
        store[(family, cols)] = made
    return made


def _flow_scratch(param_flows: torch.Tensor) -> torch.Tensor:
    """A zeroed tensor shaped like the parameter flows, for the normalizer half's: zero again when handed back."""
    hit = _FLOW_SCRATCH.get(param_flows.device)
    if hit is None or hit.shape != param_flows.shape or hit.dtype != param_flows.dtype:
        hit = _FLOW_SCRATCH[param_flows.device] = torch.zeros_like(param_flows)
    return hit


def drop_device_caches() -> None:
    _FLOW_SCRATCH.clear()


def _state(layer, info, batch: int, device) -> tuple:
    """`(log Z [B, M], each rank's log share of S [B, M, K], of Z [B, M, K])` of one node vector: written by its
    forward, read by its backward."""
    states = layer.__dict__.setdefault("_lrv_state", {})
    key = id(info.ns)
    hit = states.get(key)
    M, K = info.ns.num_nodes, layer.external_params.rank
    if hit is None or hit[0].size(0) != batch or hit[0].device != device:
        hit = states[key] = (torch.empty(batch, M, device = device), torch.empty(batch, M, K, device = device),
                             torch.empty(batch, M, K, device = device))
    return hit


def _pass(kwargs) -> dict:
    """What is kept of a PASS: `ran`, the node vectors whose forward went through here; of its backward `done`, the
    ones whose backward has, `copies`, per source of tied copies, how many of them ran, and `pending`, per source, a
    host and the copies written into it so far. Kept on the staged tensors of the forward, which every layer of the
    pass and of its backward is handed. A layer's forward may run more than once on them -- the circuit times its
    fused forward on the live pass (`TensorCircuit._stream_fusion_groups`) -- so `ran` is a set that only grows."""
    staged = kwargs[_EXTERNAL_PARAMS_KWARG]
    st = staged.__dict__.get("_lowrank_pass")
    if st is None:
        st = staged.__dict__["_lowrank_pass"] = dict(ran = dict(), done = set(), pending = dict(), copies = dict())
    return st


def _source(ns):
    return ns.get_source_ns() if ns.is_tied() else ns


def fuses(desc, layer, children) -> bool:
    """Whether a forward of this layer through the virtual batch takes the product group below fused into its
    `expand` (`children`: the group's `(table, first_row)`): products of two children, as the fused kernels of the
    plain layer take them. (Asked by the circuit's decision, without the pass's buffers: a pass that turns out too
    large for the virtual buffers forms the products' values itself, `LowRankSumParams.forward_layer`.)"""
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


def _above(layer, kwargs, node_mars, element_mars):
    """`(the layer above, its node vectors' tensors, its forward buffers)` where this layer's collapse takes that
    layer's `expand` along in this pass (`_link`): that layer goes through the virtual batch at this batch too, on
    tensors the circuit staged for all of its node vectors. Else None."""
    link = layer.__dict__.get("_lrv_link")
    if link is None:
        return None
    from pyjuice.layer.external_sum_layer import StagedExternalParams
    above, staged = link["above"](), kwargs[_EXTERNAL_PARAMS_KWARG]
    if above is None or not isinstance(staged, StagedExternalParams) or link["table"].device != node_mars.device \
            or above.__dict__.get("_lrv_wasted", 0) >= WASTED_MOST \
            or not serves(above.external_params, above, node_mars, element_mars):
        return None
    tensors = [staged.get(info.ns) for info in above.external_node_infos]
    if any(t is None for t in tensors):
        return None
    return above, tensors, _buffers(node_mars, element_mars, _columns(above, node_mars), "fw")


def forward(desc, layer, ns_tensors, node_mars, element_mars, params, kwargs, stream_children = None) -> None:
    from pyjuice.layer.sum_layer import SumLayer

    mod, K, batch, dev = _module(), desc.rank, node_mars.size(1), node_mars.device
    cols = batch * K
    bufs = _buffers(node_mars, element_mars, cols, "fw")
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
                mod.lowrank_expand(bufs["vem"], 0, bufs["vem"], cols, element_mars, fb, rows, rows, K)
            else:                                                       # (the product layer's forward was not run)
                mod.lowrank_expand_fused(bufs["vem"], 0, bufs["vem"], cols, node_mars, stream_children[0],
                                         stream_children[1], fb, rows, rows, K)
    ran = node_mars.__dict__.setdefault("_lowrank_layers", dict())      # (the layers through here on these buffers)
    if stream_children is not None:
        _link(desc, layer, ran, stream_children, dev)

    SumLayer.forward(layer, bufs["vnm"], bufs["vem"], params, precision = kwargs.get("precision", "auto"))

    # The node values -- and, in the same launch, the virtual child values of the layer above (`_link`)
    nxt = _above(layer, kwargs, node_mars, element_mars)
    if nxt is None:
        for info, (fa, _) in ns_tensors:
            mod.lowrank_collapse(node_mars, bufs["vnm"], fa, *_state(layer, info, batch, dev), info.nid_start,
                                 info.ns.num_nodes, K)
    else:
        above, tensors, bufa = nxt
        link, adesc = layer.__dict__["_lrv_link"], above.external_params
        infos = above.external_node_infos
        geo = lambda at: adesc._geometry(above, infos[at], dev)["rows"]
        for info, (fa, _) in ns_tensors:
            at = link["pairs"].get(info.nid_start)
            if at is None:
                mod.lowrank_collapse(node_mars, bufs["vnm"], fa, *_state(layer, info, batch, dev), info.nid_start,
                                     info.ns.num_nodes, K)
            else:
                mod.lowrank_collapse_expand(node_mars, bufs["vnm"], fa, *_state(layer, info, batch, dev),
                                            info.nid_start, info.ns.num_nodes, K, bufa["vem"], link["table"],
                                            link["first"], tensors[at][1], geo(at), adesc.rank)
        for at in link["rest"]:
            rows = geo(at)
            mod.lowrank_expand_fused(bufa["vem"], 0, bufa["vem"], batch * adesc.rank, node_mars, link["table"],
                                     link["first"], tensors[at][1], rows, rows, adesc.rank)
        st["ahead"] = (above, layer)

    for info, _ in ns_tensors:
        st["ran"][info.ns] = True
    ran[id(layer)] = layer
    layer.__dict__["_lr_state"] = ("virtual", batch)
    return None


def _take_denominators(mod, desc, layer, infos, scratch, denom_param_flows, params) -> None:
    """The normalizer half's parameter flows, accumulated in `scratch` by a plain backward of `layer` -- `F-`, with
    the parameters of this pass in it --, into the layer's denominator statistic `W = F- / theta` (its node vectors'
    own slices, which lie as their parameter flows do, as their parameters do); `scratch` zero again."""
    offsets, taken = desc._denom_offsets(layer), set()
    for info in infos:
        pfs, pfe = info.ns._param_flow_range
        if (pfs, pfe) in taken:                                         # (tied copies of one layer writing one range)
            continue
        taken.add((pfs, pfe))
        off = offsets[id(info.ns)][0]
        ps, pe = _source(info.ns)._param_range
        mod.lowrank_take(denom_param_flows[off:off + pfe - pfs], scratch[pfs:pfe], params[ps:pe])


def _flush(mod, desc, group, bufs, cols, K, batch, params, param_flows, denom_param_flows, want_denom, precision) -> None:
    """One plain backward of the host's layer over the normalizer columns of the copies written into its rows: its
    parameter flows are the copies' `F-` summed (`W`, once divided by the parameters), its element flows their
    gradients' normalizer part."""
    from pyjuice.layer.sum_layer import SumLayer

    layer, info, rows = group["host"]
    members = group["members"]
    used = len(members) * cols
    # Columns past the copies written this time may hold an earlier pass's: no flow there
    dirty = bufs["used"].get(info.nid_start, 0)
    if dirty > used:
        bufs["vnf_h"][info.nid_start:info.nid_end, used:dirty] = -float("inf")
    bufs["used"][info.nid_start] = used

    want_grads = any(grads is not None for grads in members)
    if want_denom or want_grads:
        scratch = _flow_scratch(param_flows) if want_denom else None
        SumLayer.backward(layer, bufs["vnf_h"], bufs["vef_h"], bufs["zero_h"], bufs["vem_h"], params,
                          param_flows = scratch, logspace_flows = True, precision = precision)
        if want_denom:
            _take_denominators(mod, desc, layer, [info], scratch, denom_param_flows, params)
        if want_grads:
            mod.lowrank_sub_b_hosted([None if grads is None else grads[1] for grads in members], bufs["vef_h"], rows,
                                     batch, K)
    return None


def backward(desc, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows, node_mars, element_mars, params,
             param_flows, denom_param_flows, kwargs, stream_children = None) -> None:
    """`stream_children`: None, or the product group below's `(table, first_row)` -- its forward (the recompute of
    `element_mars`) and its backward not run: the virtual child values are formed from the products' children's node
    values, and each element's flow is written as its two children's node flows, `element_flows` not touched."""
    from pyjuice.layer.sum_layer import SumLayer

    mod, K, batch, dev = _module(), desc.rank, node_mars.size(1), node_mars.device
    cols = batch * K
    bufs = _buffers(node_mars, element_mars, cols, "bw")
    precision = kwargs.get("precision", "auto")
    want_denom = param_flows is not None and denom_param_flows is not None and desc.apply_z_correction
    flush = lambda group, hosted: _flush(mod, desc, group, hosted, cols, K, batch, params, param_flows,
                                         denom_param_flows, want_denom, precision)

    # Another backward of the same pass: a node vector's again, or one into other flows -- the circuit times its fused
    # backward on scratch flows before the pass's own (`TensorCircuit._stream_fusion_bk_groups`)
    st = _pass(kwargs)
    flows = st.get("flows")
    if flows is None or flows() is not node_flows or any(info.ns in st["done"] for info, _ in ns_tensors):
        st.update(done = set(), pending = dict(), flows = weakref.ref(node_flows))

    def pre(vemz, oz, fb, rows, rows_z, vnfz, nid0z, fa, state, grad_a, info):
        if stream_children is None:
            mod.lowrank_backward_pre(bufs["vem_d"], vemz, oz, element_mars, fb, rows, rows_z, bufs["vnf_d"], vnfz,
                                     nid0z, node_flows, node_mars, fa, *state, grad_a, info.nid_start,
                                     info.ns.num_nodes, K)
        else:
            mod.lowrank_backward_pre_fused(bufs["vem_d"], vemz, oz, stream_children[0], stream_children[1], fb, rows,
                                           rows_z, bufs["vnf_d"], vnfz, nid0z, node_flows, node_mars, fa, *state,
                                           grad_a, info.nid_start, info.ns.num_nodes, K)

    def collapse(rows, vefz, grad_b):
        if stream_children is None:
            mod.lowrank_collapse_bw(element_flows, bufs["vef_d"], rows, vefz, grad_b, K)
        else:
            mod.lowrank_collapse_bw_fused(node_flows, bufs["vef_d"], rows, stream_children[0], stream_children[1],
                                          vefz, grad_b, K)
    if not st["done"]:                                                  # a backward's first: its copies per source
        copies = st["copies"] = dict()
        for ns in st["ran"]:
            copies[_source(ns)] = copies.get(_source(ns), 0) + 1

    # A layer of one node vector with other copies in the pass: its normalizer half goes to a host's rows
    capacity = _host_columns(cols) // cols
    hosted = None
    if len(ns_tensors) == 1 and len(layer.external_node_infos) == 1 and capacity > 1 \
            and st["copies"].get(_source(ns_tensors[0][0].ns), 0) > 1:
        info, (fa, fb) = ns_tensors[0]
        grads = ns_grad_tensors[0]
        geo = desc._geometry(layer, info, dev)
        hosted = _buffers(node_mars, element_mars, cols, "bw_host")
        src = _source(info.ns)
        group = st["pending"].get(src)
        if group is None:
            group = st["pending"][src] = dict(host = (layer, info, geo["rows"]), members = [])
        _, host, host_rows = group["host"]
        at = len(group["members"]) * cols
        pre(hosted["vem_h"], at, fb, geo["rows"], host_rows, hosted["vnf_h"], host.nid_start, fa,
            _state(layer, info, batch, dev), None if grads is None else grads[0], info)
        group["members"].append(grads)

        # The data half: the layer's element flows, a column per rank, and `F+`
        SumLayer.backward(layer, bufs["vnf_d"], bufs["vef_d"], bufs["zero"], bufs["vem_d"], params,
                          param_flows = param_flows, logspace_flows = True, precision = precision)
        collapse(geo["rows"], None, None if grads is None else grads[1])

        st["done"].add(info.ns)
        if len(group["members"]) == capacity:
            flush(st["pending"].pop(src), hosted)
    else:
        bufz = _buffers(node_mars, element_mars, cols, "bw_z")
        want_grads = any(grads is not None for grads in ns_grad_tensors)
        for (info, (fa, fb)), grads in zip(ns_tensors, ns_grad_tensors):
            rows = desc._geometry(layer, info, dev)["rows"]
            pre(bufz["vem_z"], 0, fb, rows, rows, bufz["vnf_z"], info.nid_start, fa, _state(layer, info, batch, dev),
                None if grads is None else grads[0], info)

        SumLayer.backward(layer, bufs["vnf_d"], bufs["vef_d"], bufs["zero"], bufs["vem_d"], params,
                          param_flows = param_flows, logspace_flows = True, precision = precision)

        # The normalizer half: its parameter flows are `F-`, its element flows the gradient of `B`'s other term
        if want_denom or want_grads:
            scratch = _flow_scratch(param_flows) if want_denom else None
            SumLayer.backward(layer, bufz["vnf_z"], bufz["vef_z"], bufs["zero"], bufz["vem_z"], params,
                              param_flows = scratch, logspace_flows = True, precision = precision)
            if want_denom:
                _take_denominators(mod, desc, layer, [info for info, _ in ns_tensors], scratch, denom_param_flows,
                                   params)

        for (info, _), grads in zip(ns_tensors, ns_grad_tensors):
            rows = desc._geometry(layer, info, dev)["rows"]
            if grads is None:
                collapse(rows, None, None)
            else:
                collapse(rows, bufz["vef_z"], grads[1])
            st["done"].add(info.ns)

    # The pass's last node vector: every host still holding copies
    if len(st["done"]) == len(st["ran"]) and st["pending"]:
        hosted = hosted if hosted is not None else _buffers(node_mars, element_mars, cols, "bw_host")
        for src in list(st["pending"]):
            flush(st["pending"].pop(src), hosted)

    return None
