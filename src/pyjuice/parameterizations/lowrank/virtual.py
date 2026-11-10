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
normalizer is ONE matrix product over the rows `(copy, sample, rank)` (`_normalizers`).

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
#: `False`: a forward through the virtual batch keeps the plain path (`expand`, then the twin's forward) where the
#: fused stream forward would serve (`kernels/c/lowrank_stream.cu`, the virtual child values never stored: the `expand`
#: in the stream kernel's pre-pass -- rank <= 16, B K % 64 == 0, B % 16 == 0 -- or the layer one launch, the operand
#: formed in the GEMM -- rank 16, B % 4 == 0, the normalizers in bf16; a layer of one node vector, bf16 parameters)
FUSED_STREAM = True
#: `False`: a backward through the virtual batch keeps the plain path where the one-launch backward would serve
#: (`kernels/c/lowrank_stream.cu`'s `lr_om_bw_kernel`: the data half's element flows, dB's data half and, in the grid's
#: free SMs, dA, one launch a layer; a layer of one node vector with the product layer below handed to the backward,
#: rank 16, B % 4 == 0, bf16 parameters). MEASURED (HMM 1024 x 31, rank 16, backward wall, plain -> one launch): B16
#: 1.83 -> 1.55 ms, B64 4.13 -> 3.57, B256 15.09 -> 13.73
FUSED_STREAM_BK = True


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
    """Every family of `_buffers` a layer's pass takes at once -- the forward's values (children's, nodes'), the
    backward's (flows, values, zeros) -- and the layer's own kept share differences (`_kept`): the node rows the
    layer's (`_twin`), the element rows the circuit's."""
    return 4 * cols * (4 * node_rows + 3 * element_mars.size(0))


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
    # (the forward's node values, each rank's log S: read by the collapse alone, in the launch after; what the
    # backward takes of them is the layer's own, `_kept`)
    "fw": dict(vem = (False, 1), vnm = (True, 1)),
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
            zero = name == "zero"
            made[name] = torch.zeros(shape, device = dev) if zero else torch.full(shape, -float("inf"), device = dev)
        store[key] = made
    return made


def _lz(layer, info, batch: int, device) -> torch.Tensor:
    """`log Z [M, B]` of one node vector -- batch innermost, as the factors are stored: written by its forward, read
    by its backward."""
    states = layer.__dict__.setdefault("_lrv_state", {})
    key = id(info.ns)
    hit = states.get(key)
    if hit is None or hit.size(1) != batch or hit.device != device:
        hit = states[key] = torch.empty(info.ns.num_nodes, batch, device = device)
    return hit


def _kept(layer, twin, cols: int, device) -> torch.Tensor:
    """What the layer's backward takes of its forward, `kd [twin rows, cols]`: each rank's share of S less its share
    of Z, `D_r = exp(A_r) S_r / S - exp(A_r) Z_r / Z` (the twin's row of a node, column `b K + r`; 0 where the node has
    no mass), dLL / dA_r = f D_r -- written by the forward's collapse (the one-launch form's epilogue), read by dA
    alone: neither the forward's log S_r, A nor the normalizers' product again. Its dummy rows hold 0 and are never
    written. Made when first asked for, per (columns, device); dropped with the layer's device caches."""
    hit = layer.__dict__.get("_lrv_kd")
    rows = twin._layer_nid_range[1]
    if hit is None or hit.shape != (rows, cols) or hit.device != device:
        hit = layer.__dict__["_lrv_kd"] = torch.zeros(rows, cols, device = device)
    return hit


def _pass(kwargs) -> dict:
    """What is kept of a PASS: `ran`, the node vectors whose forward went through here; of its backward `done`, the
    ones whose backward has, and `norm`, per source of tied copies, what the normalizer half needs of each copy that
    ran (its layer, node vector, factors, kept `log Z`, gradient buffers, the product group fused in, whether its
    backward took dB's normalizer half off already). Kept on the staged tensors of the forward,
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
    """`(V^T [N, rows], each row's shift)`: `exp(B - shift)`, the rows `(copy, sample, rank)` -- a sample's ranks adjacent,
    as in the virtual batch's columns --, the shift the row's largest `B`."""
    rows = len(fbs) * batch * K
    Vt = torch.empty(N, rows, dtype = dtype, device = dev)
    shift = torch.empty(rows, device = dev)
    mod.lowrank_z_operand(Vt, shift, fbs, batch, K)
    return Vt, shift


def _normalizers(desc, layer, info, st, kwargs, params, batch, dev) -> dict:
    """What a pass keeps of a node vector's source, made when the first of its copies runs. For every tied copy the pass
    was given, each rank's normalizer is a matrix product `theta @ exp(B - shift)` over the rows `(copy, sample, rank)`,
    `zt` -- its readers take the log and the row's shift back (`log Z_r = log zt + shift`: no pass of its own) --, the
    plain kernels then never seeing the normalizer half; `row`: each copy's first row. A copy's rows of `zt` are made
    as its layer's collapse needs them (`zt_done`: the copies whose rows are made, or not needed -- the one-launch
    forward forms its normalizers beside its own product, `lowrank_forward_om`, and keeps none), by `_zt_rows` -- ONE
    product for every copy where none has its rows yet, else the copy's own; `zt` itself made then (a pass whose layers
    all take the one-launch form makes none). The product's operands are the backward's too: `V`, `shift` and `theta`,
    and `fb`, the factors. `theta` dense in the product's dtype: at
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
        hit = zs[source] = dict(zt = None, zt_done = set(), row = {c: at * batch * K for at, (c, _) in enumerate(copies)},
                                V = Vt, shift = shift, dtype = dtype, fb = fbs, theta = theta, cols = batch * K,
                                fa = [t[0] for c, t in kwargs[_EXTERNAL_PARAMS_KWARG].items() if _source(c) is source])
    return hit


def _zt_rows(z, ns, every: bool = True) -> None:
    """The copy `ns`'s rows of the normalizers' product `zt` (`_normalizers`), made where they are not yet: the product of
    every copy at once while no copy has its rows (`every`: allowed), else the copy's own."""
    if ns in z["zt_done"]:
        return
    if z["zt"] is None:
        z["zt"] = torch.empty(z["theta"].size(0), z["V"].size(1), device = z["V"].device)
    if every and not z["zt_done"]:
        z["zt"].copy_(_product(z["theta"], z["V"], z["dtype"]))
        z["zt_done"].update(z["row"])
        return
    a, b = z["row"][ns], z["row"][ns] + z["cols"]
    z["zt"][:, a:b] = _product(z["theta"], z["V"][:, a:b], z["dtype"])
    z["zt_done"].add(ns)


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


_STREAM = "unset"
#: The most split ranges a row of the fused stream forward's GEMM takes (`lowrank_stream.cu`'s MAX_SPLIT)
_STREAM_MAX_SPLIT = 12
#: The pass's end forms its products' bf16 operands from the exponentials the pass keeps (`lowrank_kept_operands`:
#: the backward's `exp(A - shift)`, the forward's V), not from the factors (`lowrank_data_operands`'s shifts and
#: operands, `lowrank_u_operand`)
KEPT_OPERANDS = True

#: The tuner's code of the one-launch form (`lowrank_forward_om`); the pre-pass chain's arms are config id * 16 + split
_OM_ARM = 4096


def _stream_module():
    """The fused stream forward's extension, or None (kept, as `_module` is)."""
    global _STREAM
    if _STREAM == "unset":
        from .kernels.c import get_lowrank_stream_module
        _STREAM = get_lowrank_stream_module()
    return _STREAM


def _om_slabs(pbase: torch.Tensor, knt: int, block_size: int) -> tuple:
    """The one-launch form's parameter tiles: `(p0, bstr)` where every slab's base is `p0 + block bstr + edge tile 64
    block_size` (blocks of 32, the bases 8-aligned: one TMA box a k-step), else `(-1, 0)` (cp.async). A device sync."""
    rows = pbase.size(0)
    if block_size != 32 or rows == 0:
        return (-1, 0)
    p0 = int(pbase[0, 0])
    bstr = int(pbase[1, 0]) - p0 if rows > 1 else knt * 64 * block_size
    want = p0 + torch.arange(rows, device = pbase.device)[:, None] * bstr + \
        torch.arange(knt, device = pbase.device)[None, :] * (64 * block_size)
    if p0 % 8 != 0 or bstr % 8 != 0 or bstr <= 0 or not bool(torch.equal(pbase, want)):
        return (-1, 0)
    return (p0, bstr)


def _stream_plan(desc, layer, twin, info, node_mars, kwargs, stream_children, dev):
    """What the fused stream forward of this layer (one node vector `info`) launches, or None where it does not serve:
    per partition of the twin, the stream kernel's tables -- `nids` (the twin's rows), `ebase` / `pbase` [rows, knt] of
    64-child edge tiles (contiguous children, parameters strided by the block size, 8-aligned: the plain dispatch's
    conditions), the row groups, the pre-pass chain's arms (`arms`) and the one-launch form's parameter tiles (`om`,
    None where it does not serve), `jpos` (the child at each edge position of a group's rows, -1: none) --, each
    element row's child `jrow` from `e_lo` on, and, handed the product group below, `cbase` (None where a tile's rows
    are not two runs of children: the products' values are then formed first). Built once per (layer, device, product
    table) -- a device sync --, kept on the layer, dropped with its device caches."""
    mod = _stream_module()
    op = kwargs.get("op_params")
    K, B = desc.rank, node_mars.size(1)
    if mod is None or op is None or op.dtype != torch.bfloat16 or twin.block_size % 32 != 0:
        return None
    chain = K <= 16 and (B * K) % 64 == 0 and B % 16 == 0
    if not chain and not mod.om_serves(K, B, twin.block_size, 128):    # (128 blocks: whole tiles; each partition's below)
        return None
    cache = layer.__dict__.setdefault("_lrv_stream", {})
    key = (dev, id(stream_children[0]) if stream_children is not None else None, B)   # (the arms fit the batch's columns)
    plan = cache.get(key)
    if plan is None:
        from pyjuice.layer.kernels import c as cuda_kernels
        from pyjuice.layer.sum_layer import SumLayer
        plan = False
        rows = desc._geometry(layer, info, dev)["rows"]
        has = rows >= 0
        if bool(has.any()) and cuda_kernels.stream_is_available():
            e_lo, e_hi = int(rows[has].min()), int(rows[has].max()) + 1
            jrow = torch.full((e_hi - e_lo,), -1, dtype = torch.int32, device = dev)
            jrow[rows[has] - e_lo] = torch.arange(rows.numel(), device = dev, dtype = torch.int32)[has]
            parts, ok = [], twin.block_size % 8 == 0
            ar = torch.arange(64, device = dev, dtype = torch.int64)
            for p in range(twin.num_fw_partitions):
                if not ok:
                    break
                nids, cids, pids = (t[p].data for t in (twin.partitioned_nids, twin.partitioned_cids, twin.partitioned_pids))
                if cids.dim() != 2 or cids.size(1) % 64 != 0 or cids.size(1) == 0:   # ([blocks, edges]: 64-child tiles)
                    ok = False
                    break
                knt = cids.size(1) // 64
                cids, pids = cids.view(cids.size(0), knt, 64), pids.view(pids.size(0), knt, 64)
                ebase = cids[:, :, 0].contiguous().to(torch.int64)
                pbase = pids[:, :, 0].contiguous().to(torch.int64)
                ok = bool(torch.equal(cids.to(torch.int64), ebase[:, :, None] + ar)) and \
                    bool(torch.equal(pids.to(torch.int64), pbase[:, :, None] + ar * twin.block_size)) and \
                    bool((pbase % 8 == 0).all())
                if not ok:
                    break
                groups = cuda_kernels.row_groups(ebase)
                arms = cuda_kernels.stream_arms(twin.block_size, B * K, knt, nids.numel(), groups[2]) if chain else []
                # (the one-launch form: a tile of 128 nodes -- of whole blocks of one group, or inside one block)
                om = _om_slabs(pbase, knt, twin.block_size) \
                    if mod.om_serves(K, B, twin.block_size, nids.numel()) and \
                    (twin.block_size >= 128 or 128 // twin.block_size <= groups[2]) else None
                if not arms and om is None:
                    ok = False
                    break
                cbase = None if stream_children is None else \
                    SumLayer._stream_fusion_cbase(ebase, stream_children[0], stream_children[1])
                # the child at each edge position of a group's rows (-1: a row of no child): [groups, edges]
                epos = ebase[groups[1].long()][:, :, None] + ar
                inside = (epos >= e_lo) & (epos < e_hi)
                jpos = torch.where(inside, jrow[(epos - e_lo).clamp(0, jrow.numel() - 1)], jrow.new_full((), -1))
                # the column maxima of the pre-pass, at every split an arm may take: -inf between calls (the GEMM sets
                # them back)
                cmax = torch.full((groups[1].numel() * _STREAM_MAX_SPLIT * B,), -float("inf"), device = dev)
                parts.append(dict(nids = nids.contiguous().to(torch.int64), ebase = ebase, pbase = pbase, knt = knt,
                                  groups = groups, arms = arms, om = om, cbase = cbase, tuned = dict(),
                                  jpos = jpos.reshape(groups[1].numel(), -1).contiguous(), cmax = cmax))
            if ok:
                plan = dict(parts = parts, jrow = jrow, e_lo = e_lo)
        cache[key] = plan
    return plan or None


def _stream_run(plan, info, fb, fa, z, row0, vnm, kd, node_mars, element_mars, op, stream_children, twin, K, lz,
                nshift) -> bool:
    """The fused stream forward of one node vector, the whole layer: per partition the pre-pass chain or the one-launch
    form (it takes the normalizers' V: bf16 ones), and whether it ran -- False, nothing launched: no form serves this
    call (the plain path then). The one-launch form also makes the copy's normalizers and the collapse
    (`lowrank_forward_om`: node values, log Z, the kept share differences `kd`); the chain writes each rank's log S
    into `vnm`, then the copy's rows of the product (`_zt_rows`, its own: what the tuner times) and the collapse. The
    arm is tuned once per shape
    (`autotune.choose`; the pre-pass chain charged its two extra launches) and kept on the plan -- a call then costs one
    lookup of what the tuner holds (a cleared or retuned cache tunes again), not a list of candidates. `z`: the pass's
    normalizers of this node vector's source -- their row shifts the column bound, their V the one-launch form's
    operand; `row0` this copy's first row; `nshift` the circuit's node rows past the twin's. Handed the product group
    below, the kernels form the products' values from their children where every tile allows (`cbase`); else they are
    formed into `element_mars` first (`form_elements`), as the plain path does."""
    from pyjuice.layer.kernels import autotune
    mod = _stream_module()
    vt = z["V"] if z["dtype"] is torch.bfloat16 else None
    # (the one-launch form a layer of one partition: it collapses the node vector; the chain on several, then the
    # copy's rows of the product and the collapse once every partition's virtual node values are in)
    one = len(plan["parts"]) == 1
    om_ok = one and vt is not None and plan["parts"][0]["om"] is not None
    if not om_ok and not all(p["arms"] for p in plan["parts"]):
        return False
    fused = stream_children is not None and all(p["cbase"] is not None for p in plan["parts"])
    if stream_children is not None and not fused:
        from pyjuice.layer.fused_children import form_elements
        form_elements(element_mars, node_mars, stream_children)
    vals = node_mars if fused else element_mars
    shift = z["shift"]
    vn0, M = info.nid_start - nshift, info.ns.num_nodes

    def collapse():
        z["zt_done"].discard(info.ns)                   # (the copy's own rows: what the tuner times)
        _zt_rows(z, info.ns, every = False)
        _module().lowrank_collapse(node_mars, vnm, kd, fa, lz, z["zt"], shift, row0, info.nid_start, vn0, M, K)

    for p in plan["parts"]:
        def launch(code, p = p):
            cb = p["cbase"] if fused else None
            if code == _OM_ARM:
                mod.lowrank_forward_om(kd, vals, cb, p["ebase"], p["jpos"], vt, row0, shift, op, p["nids"],
                                       p["pbase"], p["groups"][0], twin.block_size, p["knt"], p["om"][0], p["om"][1],
                                       node_mars, lz, fa, nshift, vn0)
            else:
                mod.lowrank_forward_stream(vnm, vals, cb, p["ebase"], p["jpos"], fb, shift, row0, op, p["nids"],
                                           p["pbase"], p["groups"][0], p["groups"][1], p["cmax"], twin.block_size,
                                           p["knt"], code)
                if one:                                 # (as the arm costs: the tuner times it so)
                    collapse()
        hit = p["tuned"].get((fused, om_ok))
        if hit is None or autotune.cached(hit[0]) != hit[1]:
            key = ("lowrank_stream_fw6", info.ns.num_nodes, info.ns.num_ch_nodes, K, node_mars.size(1),
                   twin.block_size, p["knt"], p["nids"].numel(), p["groups"][2], fused, om_ok, len(plan["parts"]))
            cands = [(c, autotune.charged(lambda c = c: launch(c), 2 * autotune.LAUNCH_S)) for c in p["arms"]]
            if om_ok:
                cands.append((_OM_ARM, lambda: launch(_OM_ARM)))
            code = autotune.choose(key, cands)
            hit = p["tuned"][(fused, om_ok)] = (key, code)
        launch(hit[1])
        if hit[1] == _OM_ARM:
            z["zt_done"].add(info.ns)                   # (its rows not needed: nothing reads them)
    if not one:
        collapse()
    return True


def _stream_plan_bk(desc, layer, twin, info, node_mars, stream_children, dev):
    """What the one-launch backward of this layer (one node vector `info`, the product layer below fused in) launches, or
    None where it does not serve: per backward partition of the twin, its child blocks' first element rows (`chids`),
    each one's parent blocks' first twin rows and parameter slabs (`ppar`, `ppid` [blocks, parent blocks]) and the
    block sizes; each element row's child `jrow` from `e_lo` on. Served where every child block's rows are the node
    vector's children, every parent a node of it (no dummy), the child blocks a tile spans of equal parents, the slabs
    8-aligned (`lowrank_stream.cu`'s `om_bw_serves` for the rest). Built once per (layer, device, product table, batch) --
    a device sync --, kept on the layer, dropped with its device caches."""
    mod = _stream_module()
    K, B = desc.rank, node_mars.size(1)
    if mod is None or stream_children is None:
        return None
    cache = layer.__dict__.setdefault("_lrv_stream_bk", {})
    key = (dev, id(stream_children[0]), B)
    plan = cache.get(key)
    if plan is None:
        plan = False
        rows = desc._geometry(layer, info, dev)["rows"]
        has = rows >= 0
        vn0 = info.nid_start - _twin(layer)[1]
        M = info.ns.num_nodes
        if bool(has.any()):
            e_lo, e_hi = int(rows[has].min()), int(rows[has].max()) + 1
            jrow = torch.full((e_hi - e_lo,), -1, dtype = torch.int32, device = dev)
            jrow[rows[has] - e_lo] = torch.arange(rows.numel(), device = dev, dtype = torch.int32)[has]
            parts, ok = [], True
            for p in range(twin.num_bk_partitions):
                chids, ppar, ppid = (t[p].data.to(torch.int64).contiguous()
                                     for t in (twin.partitioned_chids, twin.partitioned_parids, twin.partitioned_parpids))
                cbs, pbs = int(twin.cs_block_sizes[p]), twin.block_size
                if ppar.dim() != 2 or not mod.om_bw_serves(K, B, cbs, pbs, chids.numel()):
                    ok = False
                    break
                bpt = 128 // cbs if cbs < 128 else 1
                ok = bool((chids >= e_lo).all()) and bool((chids + cbs <= e_hi).all()) and \
                    bool((ppar >= vn0).all()) and bool((ppar + pbs <= vn0 + M).all()) and \
                    bool((ppid % 8 == 0).all()) and \
                    (bpt == 1 or bool((ppar.view(-1, bpt, ppar.size(1)) == ppar.view(-1, bpt, ppar.size(1))[:, :1]).all()))
                if not ok:
                    break
                parts.append(dict(chids = chids, ppar = ppar, ppid = ppid, cbs = cbs, pbs = pbs))
            if ok and parts:
                plan = dict(parts = parts, jrow = jrow, e_lo = e_lo, vn0 = vn0)
        cache[key] = plan
    return plan or None


def _a_rows(mod, z, batch, K, dev):
    """The pass's `exp(A - shift)` of a source's copies (bf16 [M, rows], the rows the normalizers'), and each row's shift:
    the one-launch backward's operand, made when its first layer runs."""
    hit = z.get("ua")
    if hit is None:
        fas = z["fa"]
        Ut = torch.empty(fas[0].size(0), len(fas) * batch * K, dtype = torch.bfloat16, device = dev)
        shift = torch.empty(Ut.size(1), device = dev)
        mod.lowrank_a_operand(Ut, shift, fas, batch, K)
        hit = z["ua"] = (Ut, shift)
    return hit


def forward(desc, layer, ns_tensors, node_mars, element_mars, params, kwargs, stream_children = None) -> None:
    from pyjuice.layer.sum_layer import SumLayer

    mod, K, batch, dev = _module(), desc.rank, node_mars.size(1), node_mars.device
    cols = batch * K
    twin, shift = _twin(layer)
    bufs = _buffers(node_mars, element_mars, cols, "fw", twin._layer_nid_range)
    vnm, kd = bufs["vnm"], _kept(layer, twin, cols, dev)
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
    # The fused stream forward where it serves: the virtual child values formed in its pre-pass, never stored (and the
    # layer below's link, which would write them for nothing, dropped)
    plan = _stream_plan(desc, layer, twin, ns_tensors[0][0], node_mars, kwargs, stream_children, dev) \
        if FUSED_STREAM and len(ns_tensors) == 1 else None
    if plan is not None:
        if mark is not None and mark[0] is layer:
            mark[1].__dict__.pop("_lrv_link", None)
        info0, (fa0, fb0) = ns_tensors[0]
        z0 = _normalizers(desc, layer, info0, st, kwargs, params, batch, dev)    # (its shifts, its V: the operands')
        # (the whole layer, its collapse too)
        if not _stream_run(plan, info0, fb0, fa0, z0, z0["row"][info0.ns], vnm, kd, node_mars, element_mars,
                           kwargs["op_params"], stream_children, twin, K, _lz(layer, info0, batch, dev), shift):
            plan = None                                                 # (no form serves this call: the plain path)
    if plan is None:
        if mark is None or mark[0] is not layer:
            for info, (_, fb) in ns_tensors:
                rows = desc._geometry(layer, info, dev)["rows"]
                if stream_children is None:
                    mod.lowrank_expand(bufs["vem"], element_mars, fb, rows, K)
                else:                                                   # (the product layer's forward was not run)
                    mod.lowrank_expand_fused(bufs["vem"], node_mars, stream_children[0], stream_children[1], fb, rows, K)
    ran = node_mars.__dict__.setdefault("_lowrank_layers", dict())      # (the layers through here on these buffers)
    if stream_children is not None and plan is None:
        _link(desc, layer, ran, stream_children, dev)

    if plan is None:
        # (the circuit's bf16 copy of the tied parameters, where the circuit marked the layer for it: the plain kernels'
        # operands, as for a plain layer)
        twin.__dict__["_reads_param_operands"] = layer.__dict__.get("_reads_param_operands", False)
        SumLayer.forward(twin, vnm, bufs["vem"], params, precision = kwargs.get("precision", "auto"),
                         op_params = kwargs.get("op_params"))

    # The node values -- and, in the same launch, the virtual child values of the layer above (`_link`; the stream
    # forward collapsed already, and leaves the layer above's to its own pre-pass)
    nxt = None if plan is not None else _above(layer, kwargs, node_mars, element_mars)
    if plan is not None:
        pass
    elif nxt is None:
        for info, (fa, _) in ns_tensors:
            z = _normalizers(desc, layer, info, st, kwargs, params, batch, dev)
            _zt_rows(z, info.ns)
            mod.lowrank_collapse(node_mars, vnm, kd, fa, _lz(layer, info, batch, dev), z["zt"], z["shift"],
                                 z["row"][info.ns], info.nid_start, info.nid_start - shift, info.ns.num_nodes, K)
    else:
        above, tensors, bufa = nxt
        link, adesc = layer.__dict__["_lrv_link"], above.parameterization
        infos = above.external_node_infos
        geo = lambda at: adesc._geometry(above, infos[at], dev)["rows"]
        for info, (fa, _) in ns_tensors:
            at = link["pairs"].get(info.nid_start)
            z = _normalizers(desc, layer, info, st, kwargs, params, batch, dev)
            _zt_rows(z, info.ns)
            zt, zsh, row0 = z["zt"], z["shift"], z["row"][info.ns]
            if at is None:
                mod.lowrank_collapse(node_mars, vnm, kd, fa, _lz(layer, info, batch, dev), zt, zsh, row0,
                                     info.nid_start, info.nid_start - shift, info.ns.num_nodes, K)
            else:
                mod.lowrank_collapse_expand(node_mars, vnm, kd, fa, _lz(layer, info, batch, dev), zt, zsh, row0,
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
    sample, rank)` (`kernels/c/lowrank_mixture.cu`'s header has them), of cuBLAS: bf16 operands (fp32 accumulation) under
    the default precision where the work is `GEMM_BF16_WORK` or more, fp32 else.

    The normalizer half: `W = U^T V` into the denominator statistic (the first copy's slice: `materialize_denom_flows`
    sums the copies'), `V * (U theta)` off the `dB` of each copy whose backward did not take it off already (`copies`'
    last entry: the one-launch form, `lowrank_backward_om`, does), `V` and the rows the forward's (`z`,
    `_normalizers`). The data half: `F+ = theta * (U^T V)`, `theta` read from `params` where they lie, for the copies
    whose product layer was fused in (`copies`' fifth entry; another's plain backward has added its flows), into the
    first of them's flow block: the blocks of a source's copies are summed before the parameters read them, as those of
    the plain kernels. The operands: at bf16, from the exponentials the pass keeps -- the backward's `exp(A - shift)`
    (`_a_rows`) and the forward's V -- in one launch (`lowrank_kept_operands`); else from the factors
    (`lowrank_data_operands`, `lowrank_u_operand`). (`copies`: one source's, of one parameterization: its rank, its
    denominator or none.)"""
    layer0, info0 = copies[0][0], copies[0][1]
    desc, K = layer0.parameterization, layer0.parameterization.rank
    dev = node_flows.device
    geo = desc._geometry(layer0, info0, dev)
    M, N = geo["M"], geo["N"]
    fp = [] if param_flows is None else [c for c in copies if c[5] is not None]
    dtype = _dtype(precision, len(z["fb"]) * batch * K * M * N)
    # The copies whose rows of U are read: every one for W; else those whose dB's normalizer half is still owed (their
    # backward did not take it off), U's other rows not made. (A copy the forward was given that did not come through
    # here: no rows of U, so nothing of it in W.)
    owed = [c for c in copies if c[4] is not None and not c[6]] if want_grads else []
    rows_of = copies if want_denom else owed
    if not (fp or rows_of):
        return
    cols = batch * K
    kept = KEPT_OPERANDS and dtype is torch.bfloat16 and z["dtype"] is torch.bfloat16
    Vt, shift = (z["V"], z["shift"]) if dtype is z["dtype"] else _v_rows(mod, z["fb"], N, K, batch, dtype, dev)
    zeros = want_denom and len(copies) < len(z["row"])
    Ut = (torch.zeros if zeros else torch.empty)(M, Vt.size(1), dtype = dtype, device = dev) if rows_of else None
    Ud = Vd = None
    if fp:
        Ud = torch.empty(M, len(fp) * cols, dtype = dtype, device = dev)
        Vd = torch.empty(N, len(fp) * cols, dtype = dtype, device = dev)
    if kept:
        ua, sha = _a_rows(mod, z, batch, K, dev)
        at = {id(c): k * cols for k, c in enumerate(fp)}
        un = {id(c) for c in rows_of}
        sel = [c for c in copies if id(c) in at or id(c) in un]
        geos = [desc._geometry(c[0], c[1], dev)["rows"] if id(c) in at else None for c in sel]
        mod.lowrank_kept_operands(Ud, Vd, Ut, ua, sha, Vt, shift, node_flows, node_mars, [c[3] for c in sel],
                                  [c[1].nid_start for c in sel], geos, [c[5][0] if id(c) in at else None for c in sel],
                                  [c[5][1] if id(c) in at else 0 for c in sel], [z["row"][c[1].ns] for c in sel],
                                  [at.get(id(c), -1) for c in sel], [int(id(c) in un) for c in sel], M, K)
    else:
        if fp:
            mod.lowrank_data_operands(Ud, Vd, node_flows, node_mars, [c[2][0] for c in fp], [c[3] for c in fp],
                                      [c[2][1] for c in fp], [c[1].nid_start for c in fp],
                                      [desc._geometry(c[0], c[1], dev)["rows"] for c in fp], [c[5][0] for c in fp],
                                      [c[5][1] for c in fp], M, K)
        if rows_of:
            mod.lowrank_u_operand(Ut, shift, node_flows, [c[2][0] for c in rows_of], [c[3] for c in rows_of],
                                  [c[1].nid_start for c in rows_of], [z["row"][c[1].ns] for c in rows_of], M, K)
    if fp:
        desc._scatter_add(geo, param_flows, fp[0][1].ns._param_flow_range[0], _product(Ud, Vd.t(), dtype), params)
    if want_denom:
        desc._scatter_add(geo, denom_param_flows, desc._denom_offsets(layer0)[id(info0.ns)][0],
                          _product(Ut, Vt.t(), dtype))
    if owed:
        theta = z["theta"] if z["theta"].dtype == dtype else desc._dense_params(geo, params, dtype)
        if len(owed) == len(z["row"]):
            Gt = _product(theta.t(), Ut, dtype)                              # [N, rows]: (U theta)^T
        else:                                                                # (the owed copies' rows alone)
            Gt = torch.empty(N, Vt.size(1), device = dev)
            for c in owed:
                a = z["row"][c[1].ns]
                Gt[:, a:a + cols] = _product(theta.t(), Ut[:, a:a + cols], dtype)
        mod.lowrank_normalizer_grads([c[4][1] for c in owed], [z["row"][c[1].ns] for c in owed], Vt, Gt, batch, K)


def backward(desc, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows, node_mars, element_mars, params,
             param_flows, denom_param_flows, kwargs, stream_children = None) -> None:
    """`stream_children`: None, or the product group below's `(table, first_row)` -- its forward (the recompute of
    `element_mars`) and its backward not run: the virtual child values are formed from the products' children's node
    values, and each element's flow is written as its two children's node flows, `element_flows` not touched."""
    mod, K, batch, dev = _module(), desc.rank, node_mars.size(1), node_mars.device
    cols = batch * K
    twin, shift = _twin(layer)
    precision = kwargs.get("precision", "auto")
    want_denom = param_flows is not None and denom_param_flows is not None and desc.apply_z_correction

    # Another backward of the same pass: a node vector's again, or one into other flows -- the circuit times its fused
    # backward on scratch flows before the pass's own (`TensorCircuit._stream_fusion_bk_groups`)
    st = _pass(kwargs)
    flows = st.get("flows")
    if flows is None or flows() is not node_flows or any(info.ns in st["done"] for info, _ in ns_tensors):
        st.update(done = set(), norm = dict(), flows = weakref.ref(node_flows))

    kd = _kept(layer, twin, cols, dev)                                  # (the forward's: dLL / dA_r = f D_r)
    # The one-launch backward where it serves: dA from the kept responsibilities, then the data half's element flows (as
    # the products' children's node flows) and dB's data part in one launch -- no virtual buffers
    op = layer.__dict__.get("_bk_op_params")
    op = op if op is not None else (params if params.dtype == torch.bfloat16 else None)
    plan = _stream_plan_bk(desc, layer, twin, ns_tensors[0][0], node_mars, stream_children, dev) \
        if FUSED_STREAM_BK and len(ns_tensors) == 1 and op is not None else None
    if plan is not None:
        smod = _stream_module()
        (info, (fa, fb)), grads = ns_tensors[0], ns_grad_tensors[0]
        lz = _lz(layer, info, batch, dev)
        z = st["z"][_source(info.ns)]
        row0 = z["row"][info.ns]
        ua, sha = _a_rows(mod, z, batch, K, dev)
        for at, part in enumerate(plan["parts"]):
            # (dA by the first partition's launch, in its grid's free SMs)
            da = grads is not None and at == 0
            smod.lowrank_backward_om(node_flows, node_mars, stream_children[0], stream_children[1], fb, plan["jrow"],
                                     plan["e_lo"], None if grads is None else grads[1], ua, row0, sha, node_flows,
                                     node_mars, lz, shift, plan["vn0"], op, part["chids"], part["ppar"],
                                     part["ppid"], part["cbs"], part["pbs"], grads[0] if da else None,
                                     kd if da else None)
        # (dB's normalizer half taken off by the launches: `grads` given)
        st["norm"].setdefault(_source(info.ns), []).append((layer, info, (fa, fb), lz, grads, stream_children,
                                                            grads is not None))
        st["done"].add(info.ns)
    else:
        bufs = _buffers(node_mars, element_mars, cols, "bw", twin._layer_nid_range)
        _backward_data(desc, layer, twin, shift, ns_tensors, ns_grad_tensors, node_flows, element_flows, node_mars,
                       element_mars, params, param_flows, kwargs, stream_children, bufs, kd, st, mod, K, batch, dev,
                       precision)

    # The pass's last node vector: every source's products, all of its copies at once (each source's own
    # parameterization: its rank, whether it keeps a denominator)
    if len(st["done"]) == len(st["ran"]):
        for src, copies in st.pop("norm").items():
            own = copies[0][0].parameterization
            denom = param_flows is not None and denom_param_flows is not None and own.apply_z_correction
            grads = any(c[4] is not None and not c[6] for c in copies)          # (dB's normalizer half still owed)
            _end_of_pass(mod, st["z"][src], copies, node_flows, node_mars, params, param_flows, denom_param_flows,
                         denom, grads, precision, batch)
        st["norm"] = dict()

    return None


def _backward_data(desc, layer, twin, shift, ns_tensors, ns_grad_tensors, node_flows, element_flows, node_mars,
                   element_mars, params, param_flows, kwargs, stream_children, bufs, kd, st, mod, K, batch, dev,
                   precision) -> None:
    """The plain path of a layer's backward's data half: the virtual child values and node flows (and dA), the plain
    backward on the twin, the element flows (and dB's data part)."""
    from pyjuice.layer.sum_layer import SumLayer

    vectors = []
    for (info, (fa, fb)), grads in zip(ns_tensors, ns_grad_tensors):
        rows = desc._geometry(layer, info, dev)["rows"]
        lz = _lz(layer, info, batch, dev)
        vectors.append((info, fa, fb, grads, rows, lz))
        grad_a = None if grads is None else grads[0]
        if stream_children is None:
            mod.lowrank_backward_pre(bufs["vem_d"], element_mars, fb, rows, bufs["vnf_d"], node_flows, node_mars, fa,
                                     lz, kd, grad_a, info.nid_start, info.nid_start - shift, info.ns.num_nodes, K)
        else:
            mod.lowrank_backward_pre_fused(bufs["vem_d"], stream_children[0], stream_children[1], fb, rows,
                                           bufs["vnf_d"], node_flows, node_mars, fa, lz, kd, grad_a, info.nid_start,
                                           info.nid_start - shift, info.ns.num_nodes, K)

    # (the parameter flows: one matrix product at the pass's end where the product layer is fused in: its children's
    # values are still at hand then)
    twin.__dict__["_reads_param_operands"] = layer.__dict__.get("_reads_param_operands", False)
    SumLayer.backward(twin, bufs["vnf_d"], bufs["vef_d"], bufs["zero"], bufs["vem_d"], params,
                      param_flows = param_flows if stream_children is None else None, logspace_flows = True,
                      precision = precision, op_params = layer.__dict__.get("_bk_op_params"))

    for info, fa, fb, grads, rows, lz in vectors:
        grad_b = None if grads is None else grads[1]
        if stream_children is None:
            mod.lowrank_collapse_bw(element_flows, bufs["vef_d"], rows, grad_b, K)
        else:
            mod.lowrank_collapse_bw_fused(node_flows, bufs["vef_d"], rows, stream_children[0], stream_children[1],
                                          grad_b, K)
        st["norm"].setdefault(_source(info.ns), []).append((layer, info, (fa, fb), lz, grads, stream_children, False))
        st["done"].add(info.ns)
