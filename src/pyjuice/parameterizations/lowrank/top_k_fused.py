"""
`LowRank(top_k)` sum layers' forward by the fused CUDA kernels (`kernels/c/lowrank_top_k.cu`) where they serve
(`TOP_K_FUSED`, the module built, the shapes they hold, the default precision); else the float64 reference
(`top_k.py`), which stays what both are checked against. A node vector takes two launches: the prep (`top_k_prep`:
every operand from the caller's tensors, the circuit's element values and the shared parameters -- fp16 factors,
TF32 token rows (their slots also in fp16, scaled: the masked rows' MMAs), the children's values over their largest,
log2 theta in the MMAs' fragment order, the masked rows' slot weights) and the values (`top_k_forward`: the masked
rows' kernel, the observed rows' beside it on a side stream). Measured precision (the best Codpula head vs float64):
~3e-5 nats a masked token's read-out.
"""

from __future__ import annotations

import torch

#: `LowRank(top_k)` sum layers' forward by the fused kernels (`kernels/c/lowrank_top_k.cu`) where they serve; off, the
#: float64 reference (`top_k.py`)
TOP_K_FUSED = True

_ONCE = {}

#: the most samples a call the kernels take (their work items' scans)
MAX_SAMPLES = 4096


def top_k_module():
    """The fused forward's module, or None (no CUDA toolchain, a failed build, compute capability below 8.0)."""
    if "tk" not in _ONCE:
        from .kernels.c import get_top_k_module
        _ONCE["tk"] = get_top_k_module()
    return _ONCE["tk"]


def serves(desc, M: int, N: int, rank: int, node_mars: torch.Tensor, kwargs: dict) -> bool:
    """Whether the fused forward takes a node vector of `M` nodes over `N` children, its factors of rank `rank`."""
    return (TOP_K_FUSED and node_mars.is_cuda and node_mars.dtype == torch.float32 and node_mars.is_contiguous()
            and desc.top_k == 64 and M % 128 == 0 and N % 64 == 0 and rank <= 128 and node_mars.size(1) <= MAX_SAMPLES
            and kwargs.get("precision", "auto") != "fp32" and top_k_module() is not None)


def tf32_round(x: torch.Tensor) -> torch.Tensor:
    """`x` (float32, finite) rounded to TF32 as `cvt.rna` rounds (to nearest, ties away from zero), as float32."""
    i = x.contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)


def _empty(device, dtype):
    key = ("empty", device, dtype)
    t = _ONCE.get(key)
    if t is None:
        t = _ONCE[key] = torch.empty(0, dtype = dtype, device = device)
    return t


def _rows(t: torch.Tensor) -> torch.Tensor:
    """`t` [batch, rows, n] float32 as the kernels read it: its rows contiguous and adjacent, 16-byte aligned with a
    batch stride of a multiple of 4 floats -- a view of a group's tensor as it lies, else a contiguous copy."""
    t = t.float()
    ok = t.stride(2) == 1 and t.stride(1) == t.size(2) and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0
    return t if ok else t.contiguous()


def theta_tables(desc, sib, ns, geo, params, batch_size: int) -> dict:
    """The node vector's shared parameters as the kernels read them -- `{"theta": [M, N] fp32}`, and each table of them
    once a kernel made it ("LTh": the forward's l2t; "L2T", "LTc": the backward's) -- kept for the pass in its leaf's
    state (`top_k_leaf_fused.pass_cache`: made again every forward) and shared by the node vectors of one source of
    parameters (tied copies); a dict of this call alone where the leaf has no state."""
    from . import top_k_leaf_fused
    cache = top_k_leaf_fused.pass_cache(sib, batch_size)
    src = ns.get_source_ns() if ns.is_tied() else ns
    key = ("theta", id(src), params.data_ptr(), params.dtype, geo["M"], geo["N"])
    tab = None if cache is None else cache.get(key)
    if tab is None:
        tab = dict(theta = desc._dense_params(geo, params, dtype = torch.float32))
        if cache is not None:
            cache[key] = tab
    return tab


class Operands(list):
    """`prep`'s operands: the list the forward and the backward both read -- (BP, E, ev, cmax, LTh, mitems, oitems,
    noitems, slot32, logw, flags) -- and the masked forward's own beside it, `e16` (E's slots 0 .. 63 in fp16, times
    2^15: its MMAs' operand)."""
    e16 = None


def prep(Bm, G, element_mars, eid, has, theta, slot, rows_m, logw_m, emap = None, pair: bool = True, lth = None,
         items = None):
    """`top_k_prep`'s operands: `B [Bx, r, N]`, `G [Bx, K + 1, N]` a row of tensors each; the children's values
    `element_mars [eid, :]` (`has` [N] bool where an edge reaches them, None where every child is reached); `theta [M,
    N]` the shared parameters (0 where none); a sample each: `slot [B]` the observed token's slot, `emap [B]` the row of
    tensors it reads (None: its own, Bx = B); `rows_m` the samples whose token is missing and their slots' log-weights
    `logw_m [b, M, K + 1]` (None where none); `pair`: the samples reading a row share its work items (else a sample an
    item); `lth` theta's l2t as an earlier call made it (`theta_tables`' "LTh": not made again), or None; `items` its
    work items as `work_items` made them, or None (made by the prep)."""
    dev = Bm.device
    f = lambda t: t.float().contiguous()
    got = top_k_module().top_k_prep(
        _rows(Bm), _rows(G), element_mars, eid.long().contiguous(), _empty(dev, torch.bool) if has is None else has, f(theta),
        slot.long().contiguous(), _empty(dev, torch.long) if emap is None else emap.long().contiguous(),
        rows_m.long().contiguous(), _empty(dev, torch.float32) if logw_m is None else _weights_operand(logw_m), pair,
        _empty(dev, torch.int32) if lth is None else lth, [] if items is None else items)
    ops = Operands(got[:11])
    ops.e16 = got[11]
    return ops


def work_items(sib: dict, batch_size: int, emap):
    """The paired work items of a sum reading the leaf `sib` -- `top_k_items`' for every node vector reading the leaf's
    layer, one launch a pass (kept in the leaf's forward state, whose slots and masked rows they are made from) --, or
    None where the leaf's layer has no forward state (the prep makes its own)."""
    from . import top_k_leaf_fused
    cache = top_k_leaf_fused.pass_cache(sib, batch_size)
    st = top_k_leaf_fused._state(sib["layer"], batch_size)
    if cache is None or st is None or st.get("slot") is None:
        return None
    got = cache.get("items")
    if got is None:
        offs, B, dev = st["offs"], batch_size, st["slot"].device
        roff = top_k_leaf_fused._to_device(torch.tensor(offs, dtype = torch.int32), dev)
        mi, oi, ni, s32 = top_k_module().top_k_items(_empty(dev, torch.long) if emap is None else emap.long().contiguous(),
                                                     st["rows"].long().contiguous(), roff, offs, st["slot"].contiguous())
        got, mo, oo = [], 0, 0
        for t in range(len(offs) - 1):                                     # (a leaf's after the earlier leaves')
            rm = offs[t + 1] - offs[t]
            m1, o1 = mo + max(rm, 1), oo + 2 * max((B - rm + 1) // 2, 1) + 1
            got.append([mi[mo:m1], oi[oo:o1], ni[t:t + 1], s32[t]])
            mo, oo = m1, o1
        cache["items"] = got
    return got[top_k_leaf_fused._plan(sib["layer"])["index"][id(sib["ns"])]]


#: The masked samples' weights' row stride in the kernels' operand (`SP` of `kernels/c/lowrank_top_k.cu`)
WEIGHTS_STRIDE = 72


def _weights_operand(logw: torch.Tensor) -> torch.Tensor:
    """`logw [b, M, K + 1]` as the kernels read it: where it is a view of rows `WEIGHTS_STRIDE` wide (the leaf's,
    `top_k_leaf_fused.slot_rows`), those rows -- no copy --, else contiguous (the kernels pad it)."""
    b, M, S = logw.shape
    if logw.dtype == torch.float32 and S <= WEIGHTS_STRIDE and logw.stride() == (M * WEIGHTS_STRIDE, WEIGHTS_STRIDE, 1):
        return logw.as_strided((b, M, WEIGHTS_STRIDE), (M * WEIGHTS_STRIDE, WEIGHTS_STRIDE, 1))
    return logw.float().contiguous()


def forward(A, ops, node_mars, nid0: int, stash: bool = False, counts = None):
    """The values from the caller's `A [Bx, M, r]` and `prep`'s operands into `node_mars` from row `nid0` on (None: no
    values). With `stash`, what a backward takes from the sums, returned (`top_k_forward`'s notes; the node flows are
    not in it: `top_k_backward.terms` puts them in) -- the masked samples' posterior of the slots into `counts` where
    given (their leaf's buffer: `top_k_leaf_fused.counts_buffer`)."""
    none = _empty(A.device, torch.float32)
    got = top_k_module().top_k_forward(_rows(A), *ops, ops.e16, none if node_mars is None else node_mars, nid0, stash,
                                       none if counts is None else counts)
    return got if stash else None


def sum_forward(desc, layer, ns_info, tensors, geo, params, element_mars, node_mars, sib, kwargs) -> bool:
    """A `LowRank(top_k)` sum node vector's values by the fused kernels, written over its rows of `node_mars`; False
    where they do not serve (the reference then). Samples reading the same row of tensors (`external_params_rows`)
    share its tile."""
    from .top_k import slot_parts, external_rows
    from . import top_k_backward, top_k_leaf_fused
    B = node_mars.size(1)
    emap = external_rows(kwargs)
    A, Bm, G = tensors
    if not serves(desc, geo["M"], geo["N"], A.size(2), node_mars, kwargs) or element_mars.stride(1) != 1 or \
            (emap is None and A.size(0) != B):
        return False
    slot, rows_m, logw_m, _, _ = slot_parts(desc, sib, B, kwargs, node_mars.device)
    tab = theta_tables(desc, sib, ns_info.ns, geo, params, B)              # (the parameters' tables: once a pass)
    ops = prep(Bm, G, element_mars, geo["eid"], None if geo["full"] else geo["has"], tab["theta"], slot, rows_m, logw_m,
               emap, lth = tab.get("LTh"), items = work_items(sib, B, emap))
    tab.setdefault("LTh", ops[4])
    # (a backward may follow -- the call had grad on -- and its kernels serve: the operands and the sums' stash kept
    # for it, which `top_k_backward.terms` takes: `kept_of`)
    keep = kwargs.get("_grad_enabled", False) and \
        top_k_backward.serves(desc, geo["M"], geo["N"], A.size(2), B, node_mars.device, kwargs, False)
    counts = top_k_leaf_fused.counts_buffer(sib, desc.top_k, B) if keep and rows_m.numel() > 0 else None
    stash = forward(A, ops, node_mars, ns_info.nid_start, keep, counts)
    if keep:
        layer.__dict__["_lr_state"][("kept", id(ns_info.ns))] = (B, ops, stash)
    return True


def kept_of(layer, ns, B: int):
    """`(ops, stash)`: the forward's operands of the node vector (`prep`'s) and its sums' stash (`forward`'s), kept
    where a backward may follow (`sum_forward`), taken -- a backward uses them once (the flows go into the stash), a
    second one makes its own --; None where the forward kept none, or for another batch."""
    st = layer.__dict__.get("_lr_state")
    got = st.pop(("kept", id(ns)), None) if isinstance(st, dict) else None
    return got[1:] if got is not None and got[0] == B else None
