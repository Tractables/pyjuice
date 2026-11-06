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


def forward(desc, layer, ns_tensors, node_mars, element_mars, params, kwargs, stream_children = None) -> None:
    from pyjuice.layer.sum_layer import SumLayer

    mod, K, batch = _module(), desc.rank, node_mars.size(1)
    cols = batch * K
    bufs = _buffers(node_mars, element_mars, cols, "fw")
    st = _pass(kwargs)

    for info, (_, fb) in ns_tensors:
        rows = desc._geometry(layer, info, node_mars.device)["rows"]
        if stream_children is None:
            mod.lowrank_expand(bufs["vem"], 0, bufs["vem"], cols, element_mars, fb, rows, rows, K)
        else:                                                           # (the product layer's forward was not run)
            mod.lowrank_expand_fused(bufs["vem"], 0, bufs["vem"], cols, node_mars, stream_children[0],
                                     stream_children[1], fb, rows, rows, K)

    SumLayer.forward(layer, bufs["vnm"], bufs["vem"], params, precision = kwargs.get("precision", "auto"))

    for info, (fa, _) in ns_tensors:
        lz, lrn, lrz = _state(layer, info, batch, node_mars.device)
        mod.lowrank_collapse(node_mars, bufs["vnm"], fa, lz, lrn, lrz, info.nid_start, info.ns.num_nodes, K)
        st["ran"][info.ns] = True

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
             param_flows, denom_param_flows, kwargs) -> None:
    from pyjuice.layer.sum_layer import SumLayer

    mod, K, batch, dev = _module(), desc.rank, node_mars.size(1), node_mars.device
    cols = batch * K
    bufs = _buffers(node_mars, element_mars, cols, "bw")
    precision = kwargs.get("precision", "auto")
    want_denom = param_flows is not None and denom_param_flows is not None and desc.apply_z_correction
    flush = lambda group, hosted: _flush(mod, desc, group, hosted, cols, K, batch, params, param_flows,
                                         denom_param_flows, want_denom, precision)

    st = _pass(kwargs)
    if any(info.ns in st["done"] for info, _ in ns_tensors):           # (another backward of the same pass)
        st.update(done = set(), pending = dict())
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
        lz, lrn, lrz = _state(layer, info, batch, dev)
        mod.lowrank_backward_pre(bufs["vem_d"], hosted["vem_h"], at, element_mars, fb, geo["rows"], host_rows,
                                 bufs["vnf_d"], hosted["vnf_h"], host.nid_start,
                                 node_flows, node_mars, fa, lz, lrn, lrz, None if grads is None else grads[0],
                                 info.nid_start, info.ns.num_nodes, K)
        group["members"].append(grads)

        # The data half: the layer's element flows, a column per rank, and `F+`
        SumLayer.backward(layer, bufs["vnf_d"], bufs["vef_d"], bufs["zero"], bufs["vem_d"], params,
                          param_flows = param_flows, logspace_flows = True, precision = precision)
        mod.lowrank_collapse_bw(element_flows, bufs["vef_d"], geo["rows"], None, None if grads is None else grads[1], K)

        st["done"].add(info.ns)
        if len(group["members"]) == capacity:
            flush(st["pending"].pop(src), hosted)
    else:
        bufz = _buffers(node_mars, element_mars, cols, "bw_z")
        want_grads = any(grads is not None for grads in ns_grad_tensors)
        for (info, (fa, fb)), grads in zip(ns_tensors, ns_grad_tensors):
            rows = desc._geometry(layer, info, dev)["rows"]
            lz, lrn, lrz = _state(layer, info, batch, dev)
            mod.lowrank_backward_pre(bufs["vem_d"], bufz["vem_z"], 0, element_mars, fb, rows, rows,
                                     bufs["vnf_d"], bufz["vnf_z"], info.nid_start, node_flows,
                                     node_mars, fa, lz, lrn, lrz, None if grads is None else grads[0],
                                     info.nid_start, info.ns.num_nodes, K)

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
                mod.lowrank_collapse_bw(element_flows, bufs["vef_d"], rows, None, None, K)
            else:
                mod.lowrank_collapse_bw(element_flows, bufs["vef_d"], rows, bufz["vef_z"], grads[1], K)
            st["done"].add(info.ns)

    # The pass's last node vector: every host still holding copies
    if len(st["done"]) == len(st["ran"]) and st["pending"]:
        hosted = hosted if hosted is not None else _buffers(node_mars, element_mars, cols, "bw_host")
        for src in list(st["pending"]):
            flush(st["pending"].pop(src), hosted)

    return None
