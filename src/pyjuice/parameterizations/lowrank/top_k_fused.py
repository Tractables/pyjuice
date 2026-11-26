"""
`LowRank(top_k)` sum layers' forward by the fused CUDA kernels (`kernels/c/lowrank_top_k.cu`) where they serve
(`TOP_K_FUSED`, the module built, the shapes they hold, the default precision); else the float64 reference
(`top_k.py`), which stays what both are checked against. A node vector takes two launches: the prep (`top_k_prep`:
every operand from the caller's tensors, the circuit's element values and the shared parameters -- fp16 factors,
TF32 token rows, the children's values over their largest, log2 theta in the MMAs' fragment order, the masked rows'
slot weights) and the values (`top_k_forward`: the masked rows' kernel, the observed rows' beside it on a side
stream). Measured precision (the best Codpula head vs float64): ~3e-5 nats a masked token's read-out.
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


def prep(Bm, G, element_mars, eid, has, theta, slot, rows_m, logw_m, emap = None, pair: bool = True):
    """`top_k_prep`'s operands: `B [Bx, r, N]`, `G [Bx, K + 1, N]` a row of tensors each; the children's values
    `element_mars [eid, :]` (`has` [N] bool where an edge reaches them, None where every child is reached); `theta [M,
    N]` the shared parameters (0 where none); a sample each: `slot [B]` the observed token's slot, `emap [B]` the row of
    tensors it reads (None: its own, Bx = B); `rows_m` the samples whose token is missing and their slots' log-weights
    `logw_m [b, M, K + 1]` (None where none); `pair`: the samples reading a row share its work items (else a sample an
    item)."""
    dev = Bm.device
    f = lambda t: t.float().contiguous()
    return top_k_module().top_k_prep(
        _rows(Bm), _rows(G), element_mars, eid.long().contiguous(), _empty(dev, torch.bool) if has is None else has, f(theta),
        slot.long().contiguous(), _empty(dev, torch.long) if emap is None else emap.long().contiguous(),
        rows_m.long().contiguous(), _empty(dev, torch.float32) if logw_m is None else logw_m.float().contiguous(), pair)


def forward(A, ops, node_mars: torch.Tensor, nid0: int) -> None:
    """The values from the caller's `A [Bx, M, r]` and `prep`'s operands into `node_mars` from row `nid0` on."""
    top_k_module().top_k_forward(_rows(A), *ops, node_mars, nid0)


def sum_forward(desc, layer, ns_info, tensors, geo, params, element_mars, node_mars, sib, kwargs) -> bool:
    """A `LowRank(top_k)` sum node vector's values by the fused kernels, written over its rows of `node_mars`; False
    where they do not serve (the reference then). Samples reading the same row of tensors (`external_params_rows`)
    share its tile."""
    from .top_k import slot_parts, external_rows
    B = node_mars.size(1)
    emap = external_rows(kwargs)
    A, Bm, G = tensors
    if not serves(desc, geo["M"], geo["N"], A.size(2), node_mars, kwargs) or element_mars.stride(1) != 1 or \
            (emap is None and A.size(0) != B):
        return False
    slot, rows_m, logw_m, _, _ = slot_parts(desc, sib, B, kwargs, node_mars.device)
    theta = desc._dense_params(geo, params, dtype = torch.float32)
    ops = prep(Bm, G, element_mars, geo["eid"], None if geo["full"] else geo["has"], theta, slot, rows_m, logw_m, emap)
    forward(A, ops, node_mars, ns_info.nid_start)
    return True
