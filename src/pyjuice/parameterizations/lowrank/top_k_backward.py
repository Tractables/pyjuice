"""
`LowRank(top_k)` sum layers' BACKWARD by Triton kernels where they serve (`top_k_fused.TOP_K_FUSED`, CUDA, the fused
forward's shapes, no EM statistics asked); else `top_k._sum_backward_terms`, the reference. With the tile W of the
forward (`top_k_fused.py`: fp16 logits A' B + l2t, 2^(. - m) a row) and a = f pi the flow into each (node, slot),
alpha = a / U, beta = a / Z (`top_k._sum_backward_terms`'s terms):

    Gl [i, j] = W [i, j] (ev[j] sum_s alpha[i, s] E[s, j] - sum_s beta[i, s] E[s, j])      d / d (A B)
    ef [j]    = sum_i W [i, j] ev[j] sum_s alpha[i, s] E[s, j]                          the element flows
    dG [s, j] = E[s, j] sum_i (alpha[i, s] ev[j] - beta[i, s]) W[i, j]
    dA = Gl B^T,  dB = A^T Gl

the tile never stored, each sample its own, three launches a node vector (two without gradients):

* ROWS (a CTA 64 nodes of a sample, the children in chunks): the forward's Z and U again (TF32 dots, the rest slot by
  FP32 sums; the running maximum m a row) -> a, alpha, beta, m;
* COLUMNS (a CTA 64 children of a sample, the nodes in chunks): W again, Gl, the element flows, dB, dG -- the sums over
  the nodes in the CTA, no atomics;
* dA (a CTA 64 nodes): W again, Gl, dA.

Operands as the forward rounds them -- the factors of the logits fp16, l2t fp16, E TF32 --; the dots' other operands
(W, alpha, beta, Gl, A and B for dA, dB) rounded to the nearest TF32 (a Triton TF32 dot truncates), every sum fp32.
MEASURED on the best Codpula head's tensors against float64 (relative to the largest): the element flows 3-6e-4, dA,
dB, dG 3e-4 .. 2e-3 (the forward's fp16 logits: the same as its float64 emulation), the counts a 7e-5; a transition at
96 samples 3.6 ms (the float32 torch reference 27.6).
"""

from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                              # pragma: no cover
    triton = None

_L2E = 1.0 / math.log(2.0)
BM, BN = 64, 64                                                  # nodes, children a tile


def _parts(r: int):
    """The rank's split into the dots' K blocks (powers of two): (K0, K1), K0 + K1 >= r."""
    for k0, k1 in ((16, 0), (32, 0), (64, 0), (64, 32), (64, 64)):
        if r <= k0 + k1:
            return k0, k1
    return None


if triton is not None:

    @triton.jit
    def _tf32(x):
        """fp32 `x` (finite) rounded to the nearest TF32 value, as `cvt.rna` rounds: a TF32 dot then drops nothing"""
        return ((x.to(tl.int32, bitcast = True) + 0x1000) & -8192).to(tl.float32, bitcast = True)

    @triton.jit
    def _safe_div(a, b):
        return tl.where(b > 0, a / tl.where(b > 0, b, 1.0), 0.0)

    @triton.jit
    def _logits(AH, BH, L2T, eb, rows, cols, M, N, KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr):
        """[rows, cols] of A' B + l2t (log2 units): fp16 factors AH [Bx, M, KP], BH [Bx, KP, N], l2t [M, N] fp16"""
        k = tl.arange(0, K0)
        a = tl.load(AH + (eb * M + rows[:, None]) * KP + k[None, :])
        b = tl.load(BH + (eb * KP + k[:, None]) * N + cols[None, :])
        acc = tl.dot(a, b)
        if K1 > 0:
            k1 = K0 + tl.arange(0, K1)
            a1 = tl.load(AH + (eb * M + rows[:, None]) * KP + k1[None, :])
            b1 = tl.load(BH + (eb * KP + k1[:, None]) * N + cols[None, :])
            acc += tl.dot(a1, b1)
        return acc + tl.load(L2T + rows[:, None] * N + cols[None, :]).to(tl.float32)

    @triton.jit
    def _bw_rows_kernel(AH, BH, L2T, E, EV, LOGW, F, EMAP, ALPHA, BETA, ACNT, MROW, M, N,
                        KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        """a CTA 64 nodes of sample b: Z, U over the children -> a [B, M, 65], alpha, beta [B, M, 65] (the shift m a
        row's: W's), m [B, M]"""
        i0 = tl.program_id(0) * BM
        b = tl.program_id(1).to(tl.int64)
        eb = tl.load(EMAP + b).to(tl.int64)
        rows = i0 + tl.arange(0, BM)
        s = tl.arange(0, 64)
        m = tl.full([BM], -1e30, tl.float32)
        Z = tl.zeros([BM, 64], tl.float32)
        U = tl.zeros([BM, 64], tl.float32)
        Zr = tl.zeros([BM], tl.float32)
        Ur = tl.zeros([BM], tl.float32)
        for j0 in range(0, N, BN):
            cols = j0 + tl.arange(0, BN)
            lg = _logits(AH, BH, L2T, eb, rows, cols, M, N, KP, K0, K1)
            mx = tl.max(lg, axis = 1)
            m_new = tl.where(mx > m + 8.0, mx, m)
            sc = tl.exp2(m - m_new)
            Z, U, Zr, Ur = Z * sc[:, None], U * sc[:, None], Zr * sc, Ur * sc
            m = m_new
            W = tl.exp2(lg - m[:, None])
            Et = tl.load(E + (eb * 65 + s[None, :]) * N + cols[:, None])               # [BN, 64]
            er = tl.load(E + (eb * 65 + 64) * N + cols)
            ev = tl.load(EV + b * N + cols)
            Wev = W * ev[None, :]
            Z += tl.dot(_tf32(W), Et, input_precision = "tf32")
            U += tl.dot(_tf32(Wev), Et, input_precision = "tf32")
            Zr += tl.sum(W * er[None, :], axis = 1)
            Ur += tl.sum(Wev * er[None, :], axis = 1)
        base = (b * M + rows) * 65
        lw = tl.load(LOGW + base[:, None] + s[None, :])
        lwr = tl.load(LOGW + base + 64)
        wr = tl.exp(lw) * _safe_div(U, Z)
        wrr = tl.exp(lwr) * _safe_div(Ur, Zr)
        coef = _safe_div(tl.load(F + b * M + rows), tl.sum(wr, axis = 1) + wrr)
        a = wr * coef[:, None]
        ar = wrr * coef
        tl.store(ACNT + base[:, None] + s[None, :], a)
        tl.store(ACNT + base + 64, ar)
        tl.store(ALPHA + base[:, None] + s[None, :], _safe_div(a, U))
        tl.store(ALPHA + base + 64, _safe_div(ar, Ur))
        tl.store(BETA + base[:, None] + s[None, :], _safe_div(a, Z))
        tl.store(BETA + base + 64, _safe_div(ar, Zr))
        tl.store(MROW + b * M + rows, m)

    @triton.jit
    def _gl(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, b, eb, rows, cols, M, N,
            KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr):
        """the tile's W, alpha's and beta's contractions with E (X, Y: [rows, cols]) and ev"""
        s = tl.arange(0, 64)
        W = tl.exp2(_logits(AH, BH, L2T, eb, rows, cols, M, N, KP, K0, K1) - tl.load(MROW + b * M + rows)[:, None])
        base = (b * M + rows) * 65
        al = tl.load(ALPHA + base[:, None] + s[None, :])
        be = tl.load(BETA + base[:, None] + s[None, :])
        alr = tl.load(ALPHA + base + 64)
        ber = tl.load(BETA + base + 64)
        Es = tl.load(E + (eb * 65 + s[:, None]) * N + cols[None, :])                   # [64, BN]
        er = tl.load(E + (eb * 65 + 64) * N + cols)
        ev = tl.load(EV + b * N + cols)
        X = tl.dot(_tf32(al), Es, input_precision = "tf32") + alr[:, None] * er[None, :]
        Y = tl.dot(_tf32(be), Es, input_precision = "tf32") + ber[:, None] * er[None, :]
        return W, X, Y, ev, al, be, alr, ber, Es, er

    @triton.jit
    def _bw_cols_kernel(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, EMAP, ATF, EF, DB, DG, M, N, R,
                        KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        GRADS: tl.constexpr):
        """a CTA 64 children of sample b: over the nodes, the element flows EF [B, N] and (GRADS) dB [B, R, N], dG [B,
        65, N]"""
        j0 = tl.program_id(0) * BN
        b = tl.program_id(1).to(tl.int64)
        eb = tl.load(EMAP + b).to(tl.int64)
        cols = j0 + tl.arange(0, BN)
        k = tl.arange(0, K0)
        ef = tl.zeros([BN], tl.float32)
        dB0 = tl.zeros([K0, BN], tl.float32)
        dB1 = tl.zeros([K1 if K1 > 0 else 16, BN], tl.float32)
        P1 = tl.zeros([64, BN], tl.float32)
        P2 = tl.zeros([64, BN], tl.float32)
        P1r = tl.zeros([BN], tl.float32)
        P2r = tl.zeros([BN], tl.float32)
        for i0 in range(0, M, BM):
            rows = i0 + tl.arange(0, BM)
            W, X, Y, ev, al, be, alr, ber, Es, er = _gl(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, b, eb, rows, cols, M, N,
                                                         KP, K0, K1)
            Wev = W * ev[None, :]
            Q = Wev * X
            Gl = _tf32(Q - W * Y)
            ef += tl.sum(Q, axis = 0)
            if GRADS:
                a0 = tl.load(ATF + (eb * M + rows[:, None]) * KP + k[None, :])
                dB0 += tl.dot(tl.trans(a0), Gl, input_precision = "tf32")
                if K1 > 0:
                    k1 = K0 + tl.arange(0, K1)
                    a1 = tl.load(ATF + (eb * M + rows[:, None]) * KP + k1[None, :])
                    dB1 += tl.dot(tl.trans(a1), Gl, input_precision = "tf32")
                P1 += tl.dot(tl.trans(_tf32(al)), _tf32(Wev), input_precision = "tf32")
                P2 += tl.dot(tl.trans(_tf32(be)), _tf32(W), input_precision = "tf32")
                P1r += tl.sum(Wev * alr[:, None], axis = 0)
                P2r += tl.sum(W * ber[:, None], axis = 0)
        tl.store(EF + b * N + cols, ef)
        if GRADS:
            s = tl.arange(0, 64)
            Es = tl.load(E + (eb * 65 + s[:, None]) * N + cols[None, :])
            er = tl.load(E + (eb * 65 + 64) * N + cols)
            tl.store(DG + (b * 65 + s[:, None]) * N + cols[None, :], Es * (P1 - P2))
            tl.store(DG + (b * 65 + 64) * N + cols, er * (P1r - P2r))
            tl.store(DB + (b * R + k[:, None]) * N + cols[None, :], dB0, mask = (k < R)[:, None])
            if K1 > 0:
                k1 = K0 + tl.arange(0, K1)
                tl.store(DB + (b * R + k1[:, None]) * N + cols[None, :], dB1, mask = (k1 < R)[:, None])

    @triton.jit
    def _bw_da_kernel(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, EMAP, BTF, DA, M, N, R,
                      KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        """a CTA 64 nodes of sample b: over the children, dA [B, M, R]"""
        i0 = tl.program_id(0) * BM
        b = tl.program_id(1).to(tl.int64)
        eb = tl.load(EMAP + b).to(tl.int64)
        rows = i0 + tl.arange(0, BM)
        k = tl.arange(0, K0)
        dA0 = tl.zeros([BM, K0], tl.float32)
        dA1 = tl.zeros([BM, K1 if K1 > 0 else 16], tl.float32)
        for j0 in range(0, N, BN):
            cols = j0 + tl.arange(0, BN)
            W, X, Y, ev, al, be, alr, ber, Es, er = _gl(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, b, eb, rows, cols, M, N,
                                                         KP, K0, K1)
            Gl = _tf32(W * (X * ev[None, :] - Y))
            b0 = tl.load(BTF + (eb * KP + k[None, :]) * N + cols[:, None])            # [BN, K0]
            dA0 += tl.dot(Gl, b0, input_precision = "tf32")
            if K1 > 0:
                k1 = K0 + tl.arange(0, K1)
                b1 = tl.load(BTF + (eb * KP + k1[None, :]) * N + cols[:, None])
                dA1 += tl.dot(Gl, b1, input_precision = "tf32")
        tl.store(DA + (b * M + rows[:, None]) * R + k[None, :], dA0, mask = (k < R)[None, :])
        if K1 > 0:
            k1 = K0 + tl.arange(0, K1)
            tl.store(DA + (b * M + rows[:, None]) * R + k1[None, :], dA1, mask = (k1 < R)[None, :])


def serves(desc, M: int, N: int, r: int, logw: torch.Tensor, want_flows: bool, kwargs: dict) -> bool:
    """Whether the kernels take a node vector's backward (its gradients and element flows; not the EM statistics)."""
    from . import top_k_fused
    return (triton is not None and top_k_fused.TOP_K_FUSED and logw.is_cuda and not want_flows and desc.top_k == 64
            and M % BM == 0 and N % BN == 0 and _parts(r) is not None and kwargs.get("precision", "auto") != "fp32")


def _tf32_round(x: torch.Tensor) -> torch.Tensor:
    i = x.float().contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)


def terms(A, Bm, G, ltheta, lc, logw, f, emap, want_grads: bool):
    """`top_k._sum_backward_terms`' element flows, the gradients' values `[dA, dB, dG]` (None unless `want_grads`) and
    the posterior counts `a`, by the kernels: `A [Bx, M, r]`, `Bm [Bx, r, N]`, `G [Bx, 65, N]` a row of tensors each,
    `emap [B]` the row a sample reads (None: its own); `ltheta [M, N]`, `lc [B, N]`, `logw [B, M, 65]`, `f [B, M]`."""
    Bx, M, r = A.shape
    N = Bm.size(2)
    B = lc.size(0)
    dev = A.device
    K0, K1 = _parts(r)
    KP = K0 + K1
    pad = lambda t, dim: torch.nn.functional.pad(t, (0, KP - r) if dim == 2 else (0, 0, 0, KP - r))
    AH = pad((A.float() * _L2E).half(), 2).contiguous()                                # [Bx, M, KP] (the forward's A')
    BH = pad(Bm.half(), 1).contiguous()                                               # [Bx, KP, N]
    l2t = ltheta.float() * _L2E
    l2t = l2t - torch.where(torch.isfinite(l2t), l2t, torch.full_like(l2t, -1e30)).amax(dim = 1, keepdim = True).clamp_min(-1e30)
    L2T = l2t.half().contiguous()
    Gf = G.float()
    E = _tf32_round((Gf - torch.where(torch.isfinite(Gf), Gf, torch.full_like(Gf, -1e30)).amax(dim = 2, keepdim = True)
                     .clamp_min(-1e30)).exp()).contiguous()                           # [Bx, 65, N]
    lcf = lc.float()
    c = torch.where(torch.isfinite(lcf), lcf, torch.full_like(lcf, -1e30)).amax(dim = 1, keepdim = True).clamp_min(-1e30)
    EV = (lcf - c).exp().contiguous()                                                 # [B, N]
    EMAP = (torch.arange(B, device = dev) if emap is None else emap).to(torch.int32).contiguous()
    LOGW, F = logw.float().contiguous(), f.float().contiguous()
    ALPHA = torch.empty(B, M, 65, dtype = torch.float32, device = dev)
    BETA, ACNT = torch.empty_like(ALPHA), torch.empty_like(ALPHA)
    MROW = torch.empty(B, M, dtype = torch.float32, device = dev)
    _bw_rows_kernel[(M // BM, B)](AH, BH, L2T, E, EV, LOGW, F, EMAP, ALPHA, BETA, ACNT, MROW, M, N,
                                  KP = KP, K0 = K0, K1 = K1, BM = BM, BN = BN, num_warps = 4, num_stages = 1)
    EF = torch.empty(B, N, dtype = torch.float32, device = dev)
    vals = None
    if want_grads:
        ATF = pad(_tf32_round(A), 2).contiguous()
        BTF = pad(_tf32_round(Bm), 1).contiguous()
        DA = torch.empty(B, M, r, dtype = torch.float32, device = dev)
        DB = torch.empty(B, r, N, dtype = torch.float32, device = dev)
        DG = torch.empty(B, 65, N, dtype = torch.float32, device = dev)
        _bw_da_kernel[(M // BM, B)](AH, BH, L2T, E, EV, ALPHA, BETA, MROW, EMAP, BTF, DA, M, N, r,
                                    KP = KP, K0 = K0, K1 = K1, BM = BM, BN = BN, num_warps = 4, num_stages = 1)
        vals = [DA, DB, DG]
    else:
        ATF, DB, DG = AH, EF, EF                                                      # (not read)
    _bw_cols_kernel[(N // BN, B)](AH, BH, L2T, E, EV, ALPHA, BETA, MROW, EMAP, ATF, EF, DB, DG, M, N, r,
                                  KP = KP, K0 = K0, K1 = K1, BM = BM, BN = BN, GRADS = want_grads, num_warps = 4, num_stages = 1)
    return EF, vals, ACNT
