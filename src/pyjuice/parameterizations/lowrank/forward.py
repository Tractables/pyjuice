"""
The forward of :class:`~pyjuice.parameterizations.low_rank.LowRank` (:class:`_ForwardMixin`, one of its bases): the
forward: the circuit's fusion of the product layer below, the layer's pass -- the virtual batch (`.virtual`) where it
serves, the "exp_sum" form's fused kernels where they serve (`exp_sum_fused`), the torch passes of both forms else
(`exp_sum` in chunks of samples).
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple

from . import virtual as lowrank_virtual

#: The most (sample, node, child) triples of one node vector the `exp_sum` passes hold at a time: a larger batch is
#: taken in chunks of samples
LOWRANK_CHUNK_CELLS = 1 << 25

#: The "exp_sum" form's fused kernels (`kernels/c/lowrank_exp_sum.cu`: the scale made tile by tile, never stored) where
#: they serve (`exp_sum_fused`); off, the torch passes
EXP_SUM_FUSED = True
#: ... up to this rank (the kernels' MMAs: ranks padded to 8, 16, 32 or 64)
EXP_SUM_FUSED_RANK_MOST = 64
#: The backward's samples split over this many CTAs a tile (0: as many as fill the GPU twice; F+ and W are then added
#: atomically, a CTA holding them for its samples only) -- ranks to 8 and the TF32 path; the bf16 big tiles (ranks 9 ..
#: 64) take it as their persistent CTAs' count at most (0: the GPU's)
EXP_SUM_ZSPLIT = 0
#: At the default precision, the kernels of the exp_sum of the fp16-ROUNDED factors (`kernels/c/lowrank_exp_sum_h.cu`:
#: A' = log2 e A and B rounded to fp16 once, the ranks padded with zeros to 16, 32 or 64 -- one MMA a 16 x 8 tile a 16
#: of them --, fp32 sums) where the fused kernels serve (`exp_sum_h`); off, the bf16 three-term kernels of the
#: unrounded factors. They read fp16 factors of a padded rank where they lie (`LowRank.external_dtypes`), others (fp32,
#: bf16, fp16 of other ranks) through an fp16 copy
EXP_SUM_H = True
#: ... up to this rank
EXP_SUM_H_RANK_MOST = 64
#: ... their forward's tiles' children
EXP_SUM_H_TN = 64
#: ... the forward's fp16 copy of the factors (made where they are fp32, or fp16 the kernels cannot read in place) kept
#: for the backward, which reads it instead of converting the factors again (its prologue -61% at B1024; the copy half
#: the fp32 factors' bytes, held from the forward to the node vector's backward); off, each pass converts its own
EXP_SUM_H_KEEP = True
#: ... the NEXT node vector's fp32 factors converted into its fp16 copy under this one's main kernel (one more warp a
#: CTA, `lowrank_exp_sum_h.cu`'s `hf_ahead`) where the pass's order is known (the node vector that came after this one
#: in the last forward pass, `_ahead_next`): that one's prologue then skips its DRAM-bound conversion pass; off, each
#: converts its own
EXP_SUM_H_AHEAD = True
#: ... a pass's launches of them chained by programmatic dependent launch (each kernel's launch and prologue under the
#: tail of the one before; `lowrank_exp_sum_h.cu`'s `launch_h`); off, plain launches
EXP_SUM_H_PDL = True
#: ... the forward's children's factors (WF) made by its main kernel's producer warp (`hf_fw_kernel`'s WP): its
#: prologue then launched only to convert the factors; off, the prologue makes them
EXP_SUM_H_WP = True
#: ... the product layer group below taken over by the backward (`stream_fusion_bk`: the circuit's choice, timed): its
#: children's values made from their products' two children in its prologue (coalesced: a warp a child's 32 samples),
#: its element flows written as those children's node flows -- the product layer's recompute and backward (two
#: launches a layer) not run; off, they run. (Not the forward's: its producer would gather one sample's 64 children
#: from 128 rows of the node values a warp -- MEASURED L 256, B 1024: the forward 1.82 -> 2.19 ms, the sectors 8x the
#: bytes, from DRAM: the input layer's rows are cold)
EXP_SUM_H_STREAM = True
#: ... the backward of a node vector one row tile tall (128 nodes at most) with its dB stored, not added (each sample's
#: column tile one CTA's: no zeros first), and its element-flow sums so; off, the general kernels
EXP_SUM_H_ONE = True


def exp_sum_bf16(kwargs) -> bool:
    """The fused kernels' products in bf16 (the default precision) or three-term TF32 (`precision = "fp32"`)."""
    return kwargs.get("precision", "auto") != "fp32"


# (what the per-call checks look up once: the modules, a device's compute capability, the staging types -- an import
#  statement or a capability query a call each cost the host ~1 us apiece, at every layer of every pass)
_ONCE = {}


def exp_sum_module():
    """The fused kernels' module, or None (no CUDA toolchain, a failed build: the torch passes)."""
    if "es" not in _ONCE:
        from .kernels.c import get_exp_sum_module
        _ONCE["es"] = get_exp_sum_module()
    return _ONCE["es"]


def exp_sum_h_module():
    """The fp16-rounded factors' kernels' module, or None (the bf16 three-term kernels then)."""
    if "esh" not in _ONCE:
        from .kernels.c import get_exp_sum_h_module
        _ONCE["esh"] = get_exp_sum_h_module()
    return _ONCE["esh"]


def _capability(device) -> tuple:
    cap = _ONCE.get(("cap", device.index))
    if cap is None:
        cap = _ONCE[("cap", device.index)] = torch.cuda.get_device_capability(device)
    return cap


def _staging_types() -> tuple:
    """`(EXTERNAL_PARAMS_KWARG, StagedExternalParams)`, imported when first asked (the layer module imports this
    one)."""
    st = _ONCE.get("st")
    if st is None:
        from pyjuice.layer.parameterized_sum_layer import EXTERNAL_PARAMS_KWARG, StagedExternalParams
        st = _ONCE["st"] = (EXTERNAL_PARAMS_KWARG, StagedExternalParams)
    return st


def exp_sum_h(desc, kwargs) -> bool:
    """Whether a node vector the fused kernels take (`exp_sum_fused`) goes through the fp16-rounded factors' kernels:
    a rank they hold (`EXP_SUM_H_RANK_MOST`), the default precision, their module built (`EXP_SUM_H`)."""
    return EXP_SUM_H and desc.rank <= EXP_SUM_H_RANK_MOST and exp_sum_bf16(kwargs) and exp_sum_h_module() is not None


def exp_sum_h_theta(geo, params, kwargs) -> list:
    """The fp16 forward's tiles' theta fragments of a node vector's parameters (`exp_sum_h_theta`), made once a forward
    pass for every node vector reading the same run -- tied copies -- and kept on the pass's staged-tensors mapping
    (one a call: gone with it); keyed by the run, the geometry and the parameters' version. Not staged: made for the
    node vector alone."""
    EXTERNAL_PARAMS_KWARG, StagedExternalParams = _staging_types()
    _, _, Nb, cbs, bs = geo["grid"]
    make = lambda: exp_sum_h_module().exp_sum_h_theta(params, geo["p0"], Nb, cbs, bs, geo["M"], geo["N"], EXP_SUM_H_TN)
    holder = kwargs.get(EXTERNAL_PARAMS_KWARG)
    if not isinstance(holder, StagedExternalParams):
        return make()
    cache = holder.__dict__.setdefault("_lr_h_theta", {})
    key = (params.data_ptr(), params._version, params.dtype, geo["p0"], geo["M"], geo["N"], Nb, cbs, bs, EXP_SUM_H_TN)
    frags = cache.get(key)
    if frags is None:
        frags = cache[key] = make()
    return frags


def _staged(kwargs):
    """The pass's staged external tensors (`StagedExternalParams`: every node's, the pass's own object), or None."""
    EXTERNAL_PARAMS_KWARG, StagedExternalParams = _staging_types()
    holder = kwargs.get(EXTERNAL_PARAMS_KWARG)
    return holder if isinstance(holder, StagedExternalParams) else None


def _ahead_key(tensors) -> tuple:
    """What a copy made ahead was made from: the factors' storage and versions (a mapping of staged tensors handed to a
    later pass too: its copies taken only of factors unchanged since)."""
    return tuple((t.data_ptr(), t._version) for t in tensors[:2])


def _ahead_take(holder, ns, tensors) -> list:
    """The fp16 copy of a node vector's factors the forward before made (`_ahead_next`), where made from these factors
    as they are: `[AP, BP]`, else `[]`."""
    made = None if holder is None else holder.__dict__.get("_lr_ahead", {}).pop(id(ns), None)
    return made[0] if made is not None and made[1] == _ahead_key(tensors) else []


def _ahead_next(layer, ns, holder):
    """The node vector to convert ahead under `ns`'s forward -- the one that came after it in the last forward pass,
    staged in this one, its descriptor's form and rank the fp16 kernels' (the kernels decline the rest: other dtypes,
    layouts, sizes) --: `(ns2, tensors2)`, else None. And notes `ns` as the successor of the node vector before it in
    this pass (each pass teaches the next one its order)."""
    if holder is None:
        return None
    prev = holder.__dict__.get("_lr_prev")
    if prev is not None:
        prev[0].__dict__.setdefault("_lr_succ", {})[id(prev[1])] = ns
    holder.__dict__["_lr_prev"] = (layer, ns)
    if not EXP_SUM_H_AHEAD:
        return None
    ns2 = layer.__dict__.get("_lr_succ", {}).get(id(ns))
    t2 = None if ns2 is None else holder.get(ns2)
    if t2 is None or len(t2) < 2:
        return None
    desc = getattr(ns2, "parameterization", None)
    if getattr(desc, "form", None) != "exp_sum" or not 1 <= getattr(desc, "rank", 0) <= EXP_SUM_H_RANK_MOST:
        return None
    if t2[0].is_inference() or t2[1].is_inference():           # (no version counter to tell a copy stale by)
        return None
    return ns2, t2


def samples_contiguous(t: torch.Tensor) -> bool:
    """`[B, X, Y]` whose samples each lie contiguous, at any batch stride: the caller's tensors as read in place."""
    sz, st = t.shape, t.stride()                               # (two queries, not six: every layer's every pass asks)
    return len(sz) == 3 and (sz[2] <= 1 or st[2] == 1) and (sz[1] <= 1 or st[1] == sz[2])


def exp_sum_fused(desc, geo, tensors, params, node_mars, element_mars) -> bool:
    """Whether a node vector's pass goes through the "exp_sum" form's fused kernels: the form, a rank they hold, the
    shared parameters block-major in one run (`LowRank._geometry`'s grid), the circuit's buffers fp32 on a GPU of
    compute capability 8.0 on (the kernels' MMAs), the parameters fp32 or bf16, the factors fp32, fp16 or bf16 with
    each sample's block contiguous (the caller's, read in place, any batch stride), and the module built."""
    if not (EXP_SUM_FUSED and desc.form == "exp_sum" and desc.rank <= EXP_SUM_FUSED_RANK_MOST
            and geo["grid"] is not None and node_mars.is_cuda):
        return False
    if _capability(node_mars.device) < (8, 0):
        return False
    if node_mars.dtype != torch.float32 or element_mars.dtype != torch.float32 \
            or params.dtype not in (torch.float32, torch.bfloat16):
        return False
    if not all(t.dtype in (torch.float32, torch.float16, torch.bfloat16) and samples_contiguous(t) for t in tensors):
        return False
    return exp_sum_module() is not None


class _ForwardMixin:
    """The forward of :class:`~pyjuice.parameterizations.low_rank.LowRank`, a base of it."""

    # ------------------------------------------------------------------ forward

    def stream_fusion_plan(self, layer, fast_key: tuple, children) -> Optional[tuple]:
        """What `ParameterizedSumLayer._stream_fusion` asks for the circuit's decision to leave the product group
        below (``children``) to this layer's forward: taken where the forward goes through the mixture's virtual batch,
        whose `expand` then reads each product's value from its two children."""
        return (children,) if lowrank_virtual.fuses(self, layer, children) else None

    def stream_fusion_bk(self, layer, batch_size: int, dtype, op: bool, param_flows: bool) -> bool:
        """What `ParameterizedSumLayer._stream_fusion_bk` asks, for the circuit's decision and at every backward
        handed the product group below (`_bk_stream_children`: its forward and backward not run): taken where this
        layer's last forward went through the virtual batch (a backward at another batch size than that forward's is
        refused whatever this says), whose backward then reads each product's value from its two children and writes
        its flow as theirs; or through the fp16 exp_sum kernels (`_h_stream_bk`), whose backward does."""
        if isinstance(layer.__dict__.get("_lr_state"), tuple):
            return lowrank_virtual.fuses_bk()
        return self._h_stream_bk(layer, batch_size)

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs) -> None:
        raise NotImplementedError("`LowRank` is served by `forward_layer`.")

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        The node values `log sum_j theta_b[i, j] c_b[j]`, written over the layer's rows of `node_mars`; every node
        vector's `log Z_b` (and `sum_exp`'s shifts) kept on the layer for the backward.
        """
        self._require_full_supply(layer, ns_tensors)

        children = kwargs.pop("stream_children", None)
        if lowrank_virtual.serves(self, layer, node_mars, element_mars):
            return lowrank_virtual.forward(self, layer, ns_tensors, node_mars, element_mars, params, kwargs,
                                           stream_children = children)
        if children is not None:                                # (handed the product layer: its values, as it forms them)
            from pyjuice.layer.fused_children import form_elements
            form_elements(element_mars, node_mars, children)

        state = layer.__dict__["_lr_state"] = dict()
        rounded = layer.__dict__["_lr_h"] = dict()             # (the node vectors the fp16-rounded kernels took: their
                                                                #  factors' fp16 copy, or empty -- read in place)
        holder = _staged(kwargs)
        for ns_info, tensors in ns_tensors:
            geo = self._geometry(layer, ns_info, node_mars.device)
            if exp_sum_fused(self, geo, tensors, params, node_mars, element_mars):
                _, _, Nb, cbs, bs = geo["grid"]
                if exp_sum_h(self, kwargs):
                    given = _ahead_take(holder, ns_info.ns, tensors)       # (its copy, made by the forward before)
                    nxt = _ahead_next(layer, ns_info.ns, holder)          # (the next one's factors, converted here)
                    lzs, *copy = exp_sum_h_module().exp_sum_h_forward(
                        tensors[0], tensors[1], params, geo["p0"], Nb, cbs, bs, element_mars, geo["rows"], node_mars,
                        ns_info.nid_start, geo["M"], geo["N"], self.rank, EXP_SUM_H_TN,
                        exp_sum_h_theta(geo, params, kwargs), EXP_SUM_H_KEEP, given, [] if nxt is None else nxt[1][:2],
                        EXP_SUM_H_PDL, EXP_SUM_H_WP)
                    if nxt is not None and copy[2].numel() > 0:  # (made where the kernels could: `hf_ahead`)
                        holder.__dict__.setdefault("_lr_ahead", {})[id(nxt[0])] = (copy[2:4], _ahead_key(nxt[1]))
                    copy = copy[:2]
                    rounded[id(ns_info.ns)] = copy if copy[0].numel() > 0 else []
                else:
                    lzs = exp_sum_module().exp_sum_forward(tensors[0], tensors[1], params, geo["p0"], Nb, cbs, bs,
                                                           element_mars, geo["rows"], node_mars, ns_info.nid_start,
                                                           geo["M"], geo["N"], self.rank, exp_sum_bf16(kwargs))
                state[id(ns_info.ns)] = lzs.t()                 # (`[B, M]` as the torch passes keep it: a view)
                continue
            A, Bm = (t.float() for t in self._as_given(tensors))  # (fp16 factors as float32; views as they lie)
            theta = self._dense_params(geo, params)
            lc = self._child_values(geo, element_mars)

            if self.form == "exp_sum":
                ln, lz = self._forward_exp_sum(A, Bm, theta.log(), lc)
            else:
                ln, lz = self._forward_sum_exp(A, Bm, theta.double(), lc)

            node_mars[ns_info.nid_start:ns_info.nid_end, :] = ln.t()
            state[id(ns_info.ns)] = lz

        return None

    @staticmethod
    def _chunks(batch: int, cells: int):
        step = max(1, LOWRANK_CHUNK_CELLS // max(cells, 1))
        return [(s, min(s + step, batch)) for s in range(0, batch, step)]

    def _forward_exp_sum(self, A, Bm, ltheta, lc) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(log n, log Z)`, both `[B, M]`: `log w = log theta + A @ B` per sample, `Z = sum_j w`, `n = sum_j w c / Z`."""
        ln, lz = [], []
        for s, e in self._chunks(A.size(0), ltheta.numel()):
            lw = torch.bmm(A[s:e], Bm[s:e])                                  # [b, M, N]
            lw += ltheta[None, :, :]
            lz.append(torch.logsumexp(lw, dim = 2))
            lw += lc[s:e, None, :]
            ls = torch.logsumexp(lw, dim = 2)
            ln.append(torch.where(ls > -float("inf"), ls - lz[-1], ls))     # (no mass: 0, not 0 / 0)
        return torch.cat(ln, dim = 0), torch.cat(lz, dim = 0)

    @staticmethod
    def _mixture_terms(Bm, add: Optional[torch.Tensor], theta64):
        """
        For `sum_exp`: `x[b, r, j] = exp(B[b, r, j] (+ add[b, j]) - shift[b, r])`, float64, the shift each (sample,
        rank)'s largest exponent; and `y[b, r, i] = sum_j theta[i, j] x[b, r, j]`, `[B, k, M]`.
        """
        x = Bm if add is None else Bm + add[:, None, :]
        shift = x.amax(dim = 2, keepdim = True)
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
        x = (x - shift).double().exp()
        return x, shift, torch.matmul(x, theta64.t())

    def _forward_sum_exp(self, A, Bm, theta64, lc) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(log n, log Z)`: `S[b, i] = sum_r e^A[b, i, r] * sum_j theta[i, j] e^B[b, r, j] c[b, j]`, `Z` the same
        without `c`, `n = S / Z`."""
        At = A.transpose(1, 2)                                               # [B, k, M]
        _, sc, yc = self._mixture_terms(Bm, lc, theta64)
        _, sb, yb = self._mixture_terms(Bm, None, theta64)
        ls = torch.logsumexp(At.double() + yc.log() + sc, dim = 1)           # [B, M]
        lz = torch.logsumexp(At.double() + yb.log() + sb, dim = 1)
        return torch.where(ls > -float("inf"), ls - lz, ls).float(), lz.float()    # (no mass: 0, not 0 / 0)
