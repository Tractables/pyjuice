"""
`LowRank(top_k)` sum layers' BACKWARD by the CUDA kernels of `kernels/c/lowrank_top_k.cu` where they serve
(`top_k_fused.TOP_K_FUSED`, the module built, the fused forward's shapes, at most 4096 samples): three passes over
the forward's work items -- ROWS (the forward's sums again -> a = f pi a (node, slot), alpha = a / U, beta = a / Z,
W's shift), dA (a CTA 128 nodes), COLS (a CTA 128 children: the element flows, dB, dG) -- their notes at the top of
the .cu file. The gradients go straight into their destinations (each sample's into the row of tensors it read, times
its sign, where `external_params_rows` folds them; else into its own); the EM statistics (F+, F- over the samples, a
tile of the parameters a CTA) by a CUDA kernel for the observed samples, the Triton one below for the masked.
Else `top_k._sum_backward_terms`, the reference.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                              # pragma: no cover
    triton = None

BM, BN = 64, 64                                                  # the EM kernel's tile: nodes, children
SA = 72                                                          # alpha's, beta's, the counts' row stride (65 slots)
MAX_SAMPLES = 4096                                               # (the work items' scans)


if triton is not None:

    @triton.jit
    def _tf32(x):
        """fp32 `x` (finite) rounded to the nearest TF32 value, as `cvt.rna` rounds: a TF32 dot then drops nothing"""
        return ((x.to(tl.int32, bitcast = True) + 0x1000) & -8192).to(tl.float32, bitcast = True)

    @triton.jit
    def _bw_em_kernel(AH, BH, L2T, E, EV, ALPHA, BETA, HROW, SLOT, LWIDX, EMAP, FP, FM, M, N, B,
                      KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      SA: tl.constexpr):
        """a CTA a 64 x 64 tile of the parameters: over the masked samples, F+ = sum_b Q, F- = sum_b T ([M, N]: q, t
        summed over the slots), stored -- the observed samples' then added by `top_k_bw_em_observed` (CUDA). W from the
        CUDA passes' operands -- AH = fp16(log2 e A) [Bx, M, KP], BH [Bx, KP, N] fp16, l2t [M, N] fp16 -- and the rows
        pass's shift HROW [B, M]; a masked sample's alpha, beta [Rm, M, SA] at its place LWIDX."""
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        s = tl.arange(0, 64)
        k = tl.arange(0, K0)
        l2t = tl.load(L2T + rows[:, None] * N + cols[None, :]).to(tl.float32)
        Fp = tl.zeros([BM, BN], tl.float32)
        Fm = tl.zeros([BM, BN], tl.float32)
        for b in range(0, B):
            b64 = b.to(tl.int64)
            if tl.load(SLOT + b64) < 0:                                         # (a masked sample)
                eb = tl.load(EMAP + b64).to(tl.int64)
                lg = tl.dot(tl.load(AH + (eb * M + rows[:, None]) * KP + k[None, :]),
                            tl.load(BH + (eb * KP + k[:, None]) * N + cols[None, :]))
                if K1 > 0:
                    k1 = K0 + tl.arange(0, K1)
                    lg += tl.dot(tl.load(AH + (eb * M + rows[:, None]) * KP + k1[None, :]),
                                 tl.load(BH + (eb * KP + k1[:, None]) * N + cols[None, :]))
                W = tl.exp2(lg + l2t - tl.load(HROW + b64 * M + rows)[:, None])
                ev = tl.load(EV + b64 * N + cols)
                base = (tl.load(LWIDX + b64).to(tl.int64) * M + rows) * SA
                Es = tl.load(E + (eb * 65 + s[:, None]) * N + cols[None, :])
                er = tl.load(E + (eb * 65 + 64) * N + cols)
                X = tl.dot(_tf32(tl.load(ALPHA + base[:, None] + s[None, :])), Es, input_precision = "tf32") + \
                    tl.load(ALPHA + base + 64)[:, None] * er[None, :]
                Y = tl.dot(_tf32(tl.load(BETA + base[:, None] + s[None, :])), Es, input_precision = "tf32") + \
                    tl.load(BETA + base + 64)[:, None] * er[None, :]
                Fp += W * X * ev[None, :]
                Fm += W * Y
        out = rows[:, None] * N + cols[None, :]
        tl.store(FP + out, Fp)
        tl.store(FM + out, Fm)


def serves(desc, M: int, N: int, r: int, B: int, device, kwargs: dict, want_flows: bool) -> bool:
    """Whether the kernels take a node vector's backward (`B` samples): its element flows, gradients and (`want_flows`:
    by Triton) EM statistics."""
    from . import top_k_fused
    return (top_k_fused.TOP_K_FUSED and device.type == "cuda" and desc.top_k == 64 and M % 128 == 0 and N % 128 == 0
            and r <= 128 and B <= MAX_SAMPLES and kwargs.get("precision", "auto") != "fp32"
            and (triton is not None or not want_flows) and top_k_fused.top_k_module() is not None)


def _dest_ok(g: torch.Tensor) -> bool:
    """`g` [batch, rows, n] as the kernels add into it: float32, its rows contiguous and adjacent, 16-byte aligned with a
    batch stride of a multiple of 4 floats"""
    return (g.dtype == torch.float32 and g.is_cuda and g.dim() == 3 and g.stride(2) == 1 and g.stride(1) == g.size(2)
            and g.stride(0) % 4 == 0 and g.data_ptr() % 16 == 0)


def terms(A, Bm, G, theta, element_mars, geo, slot, rows, logw_rows, f, emap, grads, gviews, fresh: bool,
          want_flows: bool = False):
    """`top_k._sum_backward_terms`' element flows `[B, N]`, the posterior counts `a` of the masked samples `rows`
    `[b, M, 65]` and the EM statistics `(F+, F-)` `[M, N]` (None unless `want_flows`) by the kernels; the gradients of
    `A [Bx, M, r]`, `Bm [Bx, r, N]`, `G [Bx, 65, N]` (a row of tensors each) ADDED into the destinations `grads` (None:
    none) -- where they fold (`StagedExternalParams.fold`) each sample's times its sign into the row it read, else
    into its own. `emap [B]` the row a sample reads (None: its own); the parameters `theta [M, N]` (0 where none); the
    children's values `element_mars [geo["eid"], :]`; `f [B, M]` the node flows; the observed samples' slots `slot
    [B]`, the masked ones' weights `logw_rows`."""
    from . import top_k_fused as tkf
    mod = tkf.top_k_module()
    Bx, M, r = A.shape
    N = Bm.size(2)
    B = f.size(0)
    dev = A.device
    fold = getattr(gviews, "fold", None) if grads is not None else None
    # (the samples reading a row share one item -- its tile -- where their gradients merge: a fold, or none asked)
    pair = emap is None or fold is not None or grads is None
    A_ = tkf._rows(A)
    ops = tkf.prep(Bm, G, element_mars, geo["eid"], None if geo["full"] else geo["has"], theta, slot, rows, logw_rows,
                   emap, pair = pair)
    BP, E, ev, cmax, LTh, mitems, oitems, noitems, slot32, logw, flags = ops
    AH, L2T, LTc = mod.top_k_bw_prep(A_, theta.float().contiguous(), BP.size(1))
    alpha, beta, acnt, ao, bo, hrow = mod.top_k_bw_rows(A_, *ops, f.float().contiguous())
    EF = torch.zeros(B, N, dtype = torch.float32, device = dev)
    if grads is not None:
        gviews.init_grads(grads) if fresh else None
        drow, sgn = fold if fold is not None else (torch.arange(B, device = dev), None)
        sgn = torch.ones(B, device = dev) if sgn is None else sgn
        dests = [g if _dest_ok(g) else torch.zeros(g.shape, dtype = torch.float32, device = dev) for g in grads]
        mod.top_k_bw_grads(A_, AH, BP, E, ev, LTh, LTc, mitems, oitems, noitems, slot32, alpha, beta, ao, bo, hrow,
                           drow.to(torch.int32).contiguous(), sgn.float().contiguous(), EF, *dests)
        for g, d in zip(grads, dests):
            if d is not g:
                g.add_(d.to(g.dtype))
    else:
        none = torch.empty(0, dtype = torch.float32, device = dev)
        mod.top_k_bw_grads(A_, AH, BP, E, ev, LTh, LTc, mitems, oitems, noitems, slot32, alpha, beta, ao, bo, hrow,
                           none, none, EF, none, none, none)
    Rm = rows.numel()
    flows = None
    if want_flows:
        KP = BP.size(1)
        K0 = min(KP, 64)
        EMAP = (torch.arange(B, device = dev) if emap is None else emap).to(torch.int32).contiguous()
        LWIDX = torch.full((B,), -1, dtype = torch.int32, device = dev)
        LWIDX[rows] = torch.arange(Rm, dtype = torch.int32, device = dev)
        SLOT = torch.where(LWIDX >= 0, -1, slot.to(torch.int32)).to(torch.int32).contiguous()
        # the masked samples' (Triton: stored, or zeros where none), then the observed ones' added (CUDA: a tile of
        # the parameters a CTA walking the samples)
        if Rm > 0:
            FP = torch.empty(M, N, dtype = torch.float32, device = dev)
            FM = torch.empty_like(FP)
            _bw_em_kernel[(M // BM, N // BN)](AH, BP, L2T, E, ev, alpha, beta, hrow, SLOT, LWIDX, EMAP, FP, FM, M, N,
                                              B, KP = KP, K0 = K0, K1 = KP - K0, BM = BM, BN = BN, SA = SA, num_warps = 4,
                                              num_stages = 1)
        else:
            FP = torch.zeros(M, N, dtype = torch.float32, device = dev)
            FM = torch.zeros_like(FP)
        if B > Rm:
            mod.top_k_bw_em_observed(AH, BP, E, ev, LTh, ao, bo, hrow, SLOT,
                                     torch.empty(0, dtype = torch.int32, device = dev) if emap is None else EMAP, FP, FM)
        flows = (FP, FM)
    return EF, acnt[:Rm, :, :65], flows
