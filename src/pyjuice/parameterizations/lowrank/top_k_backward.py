"""
`LowRank(top_k)` sum layers' BACKWARD by Triton kernels where they serve (`top_k_fused.TOP_K_FUSED`, CUDA, the fused
forward's shapes, no EM statistics asked); else `top_k._sum_backward_terms`, the reference. With the tile W of the
forward (`top_k_fused.py`: fp16 logits A' B + l2t, 2^(. - m) a row) and a = f pi the flow into each (node, slot),
alpha = a / U, beta = a / Z (`top_k._sum_backward_terms`'s terms):

    Gl [i, j] = W [i, j] (ev[j] sum_s alpha[i, s] E[s, j] - sum_s beta[i, s] E[s, j])      d / d (A B)
    ef [j]    = sum_i W [i, j] ev[j] sum_s alpha[i, s] E[s, j]                          the element flows
    dG [s, j] = E[s, j] sum_i (alpha[i, s] ev[j] - beta[i, s]) W[i, j]
    dA = Gl B^T,  dB = A^T Gl

the tile never stored, each sample its own; each launch twice, once a kind of sample (a masked token's: the slots'
weights the leaf's; an OBSERVED token's: one slot, alpha and beta one number a node, no slot dots), the other kind's
CTAs leaving at once:

* ROWS (a CTA 64 nodes of a sample, the children in chunks): the forward's Z and U again (TF32 dots, the rest slot by
  FP32 sums; the running maximum m a row) -> a, alpha, beta, m;
* FLOWS (a CTA 64 children of a sample, the nodes in chunks): W again, Gl, the element flows and dB -- the sums over the
  nodes in the CTA, no atomics;
* dG (the same CTAs): W again, alpha^T (W ev) - beta^T W into one accumulator;
* dA (a CTA 64 nodes): W again, Gl, dA.

(dG apart from the flows and dB: one CTA keeping both accumulators ran at a third of the speed.)

Operands as the forward rounds them -- the factors of the logits fp16, l2t fp16, E TF32 --; the dots' other operands
(W, alpha, beta, Gl, A and B for dA, dB) rounded to the nearest TF32 (a Triton TF32 dot truncates), every sum fp32.
MEASURED on the best Codpula head's tensors against float64 (relative to the largest): the element flows 3-6e-4, dA,
dB, dG 3e-4 .. 2e-3 (the forward's fp16 logits: the same as its float64 emulation), the counts a 7e-5; a transition at
96 samples (48 rows, both passes) 1.96 ms (the float32 torch reference 28.6).
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
    def _bw_rows_kernel(AH, BH, L2T, E, EV, LOGW, LWIDX, SLOT, F, EMAP, ALPHA, BETA, ACNT, MROW, M, N,
                        KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        OBS: tl.constexpr):
        """a CTA 64 nodes of sample b (of its kind: a masked token's, OBS an observed one's): Z, U over the children ->
        alpha, beta [B, M, 65] (OBS: the observed slot's at column 0), the masked row's counts ACNT [Rm, M, 65], the
        shift m [B, M] (W's)"""
        i0 = tl.program_id(0) * BM
        b = tl.program_id(1).to(tl.int64)
        q = tl.load(LWIDX + b).to(tl.int64)
        so = tl.load(SLOT + b).to(tl.int64)
        if OBS:
            if so < 0:
                return
        else:
            if q < 0:
                return
        eb = tl.load(EMAP + b).to(tl.int64)
        rows = i0 + tl.arange(0, BM)
        s = tl.arange(0, 64)
        m = tl.full([BM], -1e30, tl.float32)
        if OBS:
            Z = tl.zeros([BM], tl.float32)
            U = tl.zeros([BM], tl.float32)
        else:
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
            m = m_new
            W = tl.exp2(lg - m[:, None])
            ev = tl.load(EV + b * N + cols)
            Wev = W * ev[None, :]
            if OBS:
                es = tl.load(E + (eb * 65 + so) * N + cols)
                Z = Z * sc + tl.sum(W * es[None, :], axis = 1)
                U = U * sc + tl.sum(Wev * es[None, :], axis = 1)
            else:
                Z, U, Zr, Ur = Z * sc[:, None], U * sc[:, None], Zr * sc, Ur * sc
                Et = tl.load(E + (eb * 65 + s[None, :]) * N + cols[:, None])           # [BN, 64]
                er = tl.load(E + (eb * 65 + 64) * N + cols)
                Z = tl.dot(_tf32(W), Et, Z, input_precision = "tf32")
                U = tl.dot(_tf32(Wev), Et, U, input_precision = "tf32")
                Zr += tl.sum(W * er[None, :], axis = 1)
                Ur += tl.sum(Wev * er[None, :], axis = 1)
        f = tl.load(F + b * M + rows)
        base = (b * M + rows) * 65
        if OBS:                                                     # one slot: a = f where the slot's ratio is positive
            a = tl.where(_safe_div(U, Z) > 0, f, 0.0)
            tl.store(ALPHA + base, _safe_div(a, U))
            tl.store(BETA + base, _safe_div(a, Z))
        else:
            qb = (q * M + rows) * 65
            wr = tl.exp(tl.load(LOGW + qb[:, None] + s[None, :])) * _safe_div(U, Z)
            wrr = tl.exp(tl.load(LOGW + qb + 64)) * _safe_div(Ur, Zr)
            coef = _safe_div(f, tl.sum(wr, axis = 1) + wrr)
            a = wr * coef[:, None]
            ar = wrr * coef
            tl.store(ACNT + qb[:, None] + s[None, :], a)
            tl.store(ACNT + qb + 64, ar)
            tl.store(ALPHA + base[:, None] + s[None, :], _safe_div(a, U))
            tl.store(ALPHA + base + 64, _safe_div(ar, Ur))
            tl.store(BETA + base[:, None] + s[None, :], _safe_div(a, Z))
            tl.store(BETA + base + 64, _safe_div(ar, Zr))
        tl.store(MROW + b * M + rows, m)

    @triton.jit
    def _gl(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, b, eb, so, rows, cols, M, N,
            KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, OBS: tl.constexpr):
        """the tile's W, ev, and alpha's and beta's contractions with E (X, Y [rows, cols]): Gl = W (X ev - Y)"""
        W = tl.exp2(_logits(AH, BH, L2T, eb, rows, cols, M, N, KP, K0, K1) - tl.load(MROW + b * M + rows)[:, None])
        ev = tl.load(EV + b * N + cols)
        base = (b * M + rows) * 65
        if OBS:
            es = tl.load(E + (eb * 65 + so) * N + cols)
            X = tl.load(ALPHA + base)[:, None] * es[None, :]
            Y = tl.load(BETA + base)[:, None] * es[None, :]
        else:
            s = tl.arange(0, 64)
            Es = tl.load(E + (eb * 65 + s[:, None]) * N + cols[None, :])               # [64, BN]
            er = tl.load(E + (eb * 65 + 64) * N + cols)
            X = tl.dot(_tf32(tl.load(ALPHA + base[:, None] + s[None, :])), Es, input_precision = "tf32") + \
                tl.load(ALPHA + base + 64)[:, None] * er[None, :]
            Y = tl.dot(_tf32(tl.load(BETA + base[:, None] + s[None, :])), Es, input_precision = "tf32") + \
                tl.load(BETA + base + 64)[:, None] * er[None, :]
        return W, ev, X, Y

    @triton.jit
    def _bw_db_kernel(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, LWIDX, SLOT, EMAP, ATF, EF, DB, M, N, R,
                      KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      OBS: tl.constexpr, GRADS: tl.constexpr):
        """a CTA 64 children of sample b (of its kind): over the nodes, the element flows EF [B, N] and (GRADS) dB [B, R,
        N] = A^T Gl"""
        j0 = tl.program_id(0) * BN
        b = tl.program_id(1).to(tl.int64)
        so = tl.load(SLOT + b).to(tl.int64)
        if OBS:
            if so < 0:
                return
        else:
            if tl.load(LWIDX + b) < 0:
                return
        eb = tl.load(EMAP + b).to(tl.int64)
        cols = j0 + tl.arange(0, BN)
        k = tl.arange(0, K0)
        ef = tl.zeros([BN], tl.float32)
        dB0 = tl.zeros([K0, BN], tl.float32)
        dB1 = tl.zeros([K1 if K1 > 0 else 16, BN], tl.float32)
        for i0 in range(0, M, BM):
            rows = i0 + tl.arange(0, BM)
            W, ev, X, Y = _gl(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, b, eb, so, rows, cols, M, N, KP, K0, K1, OBS)
            Q = W * X * ev[None, :]
            ef += tl.sum(Q, axis = 0)
            if GRADS:
                Gl = _tf32(Q - W * Y)
                dB0 = tl.dot(tl.load(ATF + (eb * M + rows[None, :]) * KP + k[:, None]), Gl, dB0, input_precision = "tf32")
                if K1 > 0:
                    k1 = K0 + tl.arange(0, K1)
                    dB1 = tl.dot(tl.load(ATF + (eb * M + rows[None, :]) * KP + k1[:, None]), Gl, dB1,
                                 input_precision = "tf32")
        tl.store(EF + b * N + cols, ef)
        if GRADS:
            tl.store(DB + (b * R + k[:, None]) * N + cols[None, :], dB0, mask = (k < R)[:, None])
            if K1 > 0:
                k1 = K0 + tl.arange(0, K1)
                tl.store(DB + (b * R + k1[:, None]) * N + cols[None, :], dB1, mask = (k1 < R)[:, None])

    @triton.jit
    def _bw_dg_kernel(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, LWIDX, SLOT, EMAP, DG, M, N,
                      KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      OBS: tl.constexpr):
        """a CTA 64 children of sample b (of its kind): over the nodes, dG [B, 65, N] = E (alpha^T (W ev) - beta^T W)
        (OBS: the observed slot's row, the others 0)"""
        j0 = tl.program_id(0) * BN
        b = tl.program_id(1).to(tl.int64)
        so = tl.load(SLOT + b).to(tl.int64)
        if OBS:
            if so < 0:
                return
        else:
            if tl.load(LWIDX + b) < 0:
                return
        eb = tl.load(EMAP + b).to(tl.int64)
        cols = j0 + tl.arange(0, BN)
        s = tl.arange(0, 64)
        ev = tl.load(EV + b * N + cols)
        if OBS:
            PD = tl.zeros([BN], tl.float32)
        else:
            PD = tl.zeros([64, BN], tl.float32)
            PDr = tl.zeros([BN], tl.float32)
        for i0 in range(0, M, BM):
            rows = i0 + tl.arange(0, BM)
            W = tl.exp2(_logits(AH, BH, L2T, eb, rows, cols, M, N, KP, K0, K1) - tl.load(MROW + b * M + rows)[:, None])
            base = (b * M + rows) * 65
            if OBS:
                PD += tl.sum(W * (tl.load(ALPHA + base)[:, None] * ev[None, :] - tl.load(BETA + base)[:, None]), axis = 0)
            else:
                Wev = W * ev[None, :]
                PD = tl.dot(_tf32(tl.load(ALPHA + base[None, :] + s[:, None])), _tf32(Wev), PD, input_precision = "tf32")
                PD = tl.dot(_tf32(-tl.load(BETA + base[None, :] + s[:, None])), _tf32(W), PD, input_precision = "tf32")
                PDr += tl.sum(Wev * tl.load(ALPHA + base + 64)[:, None] - W * tl.load(BETA + base + 64)[:, None], axis = 0)
        if OBS:
            s65 = tl.arange(0, 128)
            es = tl.load(E + (eb * 65 + so) * N + cols)
            tl.store(DG + (b * 65 + s65[:, None]) * N + cols[None, :],
                     tl.where((s65 == so)[:, None], (es * PD)[None, :], 0.0), mask = (s65 < 65)[:, None])
        else:
            Es = tl.load(E + (eb * 65 + s[:, None]) * N + cols[None, :])
            tl.store(DG + (b * 65 + s[:, None]) * N + cols[None, :], Es * PD)
            tl.store(DG + (b * 65 + 64) * N + cols, tl.load(E + (eb * 65 + 64) * N + cols) * PDr)

    @triton.jit
    def _bw_da_kernel(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, LWIDX, SLOT, EMAP, BTF, DA, M, N, R,
                      KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      OBS: tl.constexpr):
        """a CTA 64 nodes of sample b (of its kind): over the children, dA [B, M, R] = Gl B^T"""
        i0 = tl.program_id(0) * BM
        b = tl.program_id(1).to(tl.int64)
        so = tl.load(SLOT + b).to(tl.int64)
        if OBS:
            if so < 0:
                return
        else:
            if tl.load(LWIDX + b) < 0:
                return
        eb = tl.load(EMAP + b).to(tl.int64)
        rows = i0 + tl.arange(0, BM)
        k = tl.arange(0, K0)
        dA0 = tl.zeros([BM, K0], tl.float32)
        dA1 = tl.zeros([BM, K1 if K1 > 0 else 16], tl.float32)
        for j0 in range(0, N, BN):
            cols = j0 + tl.arange(0, BN)
            W, ev, X, Y = _gl(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, b, eb, so, rows, cols, M, N, KP, K0, K1, OBS)
            Gl = _tf32(W * (X * ev[None, :] - Y))
            dA0 = tl.dot(Gl, tl.load(BTF + (eb * KP + k[None, :]) * N + cols[:, None]), dA0, input_precision = "tf32")
            if K1 > 0:
                k1 = K0 + tl.arange(0, K1)
                dA1 = tl.dot(Gl, tl.load(BTF + (eb * KP + k1[None, :]) * N + cols[:, None]), dA1, input_precision = "tf32")
        tl.store(DA + (b * M + rows[:, None]) * R + k[None, :], dA0, mask = (k < R)[None, :])
        if K1 > 0:
            k1 = K0 + tl.arange(0, K1)
            tl.store(DA + (b * M + rows[:, None]) * R + k1[None, :], dA1, mask = (k1 < R)[None, :])


    @triton.jit
    def _bw_em_kernel(AH, BH, L2T, E, EV, ALPHA, BETA, MROW, SLOT, EMAP, FP, FM, M, N, B,
                      KP: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        """a CTA a 64 x 64 tile of the parameters: over the samples, F+ = sum_b Q, F- = sum_b T ([M, N]: q, t summed
        over the slots)"""
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        s = tl.arange(0, 64)
        Fp = tl.zeros([BM, BN], tl.float32)
        Fm = tl.zeros([BM, BN], tl.float32)
        for b in range(0, B):
            b64 = b.to(tl.int64)
            eb = tl.load(EMAP + b64).to(tl.int64)
            so = tl.load(SLOT + b64).to(tl.int64)
            W = tl.exp2(_logits(AH, BH, L2T, eb, rows, cols, M, N, KP, K0, K1) - tl.load(MROW + b64 * M + rows)[:, None])
            ev = tl.load(EV + b64 * N + cols)
            base = (b64 * M + rows) * 65
            if so >= 0:
                es = tl.load(E + (eb * 65 + so) * N + cols)
                X = tl.load(ALPHA + base)[:, None] * es[None, :]
                Y = tl.load(BETA + base)[:, None] * es[None, :]
            else:
                Es = tl.load(E + (eb * 65 + s[:, None]) * N + cols[None, :])
                er = tl.load(E + (eb * 65 + 64) * N + cols)
                X = tl.dot(_tf32(tl.load(ALPHA + base[:, None] + s[None, :])), Es, input_precision = "tf32") + \
                    tl.load(ALPHA + base + 64)[:, None] * er[None, :]
                Y = tl.dot(_tf32(tl.load(BETA + base[:, None] + s[None, :])), Es, input_precision = "tf32") + \
                    tl.load(BETA + base + 64)[:, None] * er[None, :]
            Fp += W * X * ev[None, :]
            Fm += W * Y
        tl.store(FP + rows[:, None] * N + cols[None, :], Fp)
        tl.store(FM + rows[:, None] * N + cols[None, :], Fm)


def serves(desc, M: int, N: int, r: int, device, kwargs: dict) -> bool:
    """Whether the kernels take a node vector's backward: its element flows, gradients and EM statistics."""
    from . import top_k_fused
    return (triton is not None and top_k_fused.TOP_K_FUSED and device.type == "cuda" and desc.top_k == 64 and
            M % BM == 0 and N % BN == 0 and _parts(r) is not None and kwargs.get("precision", "auto") != "fp32")


def _tf32_round(x: torch.Tensor) -> torch.Tensor:
    i = x.float().contiguous().view(torch.int32)
    return ((i + 0x1000) & ~0x1FFF).view(torch.float32)


def _shifted_exp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """exp(x - its largest along `dim`), the largest over the finite entries (0 where none is)"""
    big = torch.where(torch.isfinite(x), x, torch.full_like(x, -1e30)).amax(dim = dim, keepdim = True).clamp_min(-1e30)
    return (x - big).exp()


def terms(A, Bm, G, ltheta, lc, slot, rows, logw_rows, f, emap, want_grads: bool, want_flows: bool = False):
    """`top_k._sum_backward_terms`' element flows `[B, N]`, the gradients' values `[dA, dB, dG]` (None unless
    `want_grads`), the posterior counts `a` of the masked samples `rows` `[b, M, 65]` and the EM statistics `(F+, F-)`
    `[M, N]` (None unless `want_flows`), by the kernels: `A [Bx, M, r]`, `Bm [Bx, r, N]`, `G [Bx, 65, N]` a row of
    tensors each, `emap [B]` the row a sample reads (None: its own); `ltheta [M, N]`, `lc [B, N]`, `f [B, M]`; the
    observed samples' slots `slot [B]`, the masked ones' weights `logw_rows`."""
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
    L2T = (l2t - torch.where(torch.isfinite(l2t), l2t, torch.full_like(l2t, -1e30)).amax(dim = 1, keepdim = True)
           .clamp_min(-1e30)).half().contiguous()                                     # [M, N]
    E = _tf32_round(_shifted_exp(G.float(), 2)).contiguous()                          # [Bx, 65, N]
    EV = _shifted_exp(lc.float(), 1).contiguous()                                     # [B, N]
    EMAP = (torch.arange(B, device = dev) if emap is None else emap).to(torch.int32).contiguous()
    Rm = rows.numel()
    LWIDX = torch.full((B,), -1, dtype = torch.int32, device = dev)
    LWIDX[rows] = torch.arange(Rm, dtype = torch.int32, device = dev)
    SLOT = torch.where(LWIDX >= 0, -1, slot.to(torch.int32)).to(torch.int32).contiguous()
    LOGW = logw_rows.float().contiguous() if Rm > 0 else torch.zeros(1, M, 65, device = dev)
    F = f.float().contiguous()
    ALPHA = torch.empty(B, M, 65, dtype = torch.float32, device = dev)
    BETA = torch.empty_like(ALPHA)
    ACNT = torch.empty(max(Rm, 1), M, 65, dtype = torch.float32, device = dev)
    MROW = torch.empty(B, M, dtype = torch.float32, device = dev)
    EF = torch.empty(B, N, dtype = torch.float32, device = dev)
    opts = dict(KP = KP, K0 = K0, K1 = K1, BM = BM, BN = BN, num_warps = 4, num_stages = 1)
    for obs in (False, True):
        _bw_rows_kernel[(M // BM, B)](AH, BH, L2T, E, EV, LOGW, LWIDX, SLOT, F, EMAP, ALPHA, BETA, ACNT, MROW, M, N,
                                      OBS = obs, **opts)
    vals = None
    ATF, DB = AH, EF                                                                   # (not read without gradients)
    if want_grads:
        ATF = pad(_tf32_round(A), 2).contiguous()
        BTF = pad(_tf32_round(Bm), 1).contiguous()
        DA = torch.empty(B, M, r, dtype = torch.float32, device = dev)
        DB = torch.empty(B, r, N, dtype = torch.float32, device = dev)
        DG = torch.empty(B, 65, N, dtype = torch.float32, device = dev)
        for obs in (False, True):
            _bw_dg_kernel[(N // BN, B)](AH, BH, L2T, E, EV, ALPHA, BETA, MROW, LWIDX, SLOT, EMAP, DG, M, N, OBS = obs, **opts)
            _bw_da_kernel[(M // BM, B)](AH, BH, L2T, E, EV, ALPHA, BETA, MROW, LWIDX, SLOT, EMAP, BTF, DA, M, N, r,
                                        OBS = obs, **opts)
        vals = [DA, DB, DG]
    for obs in (False, True):
        _bw_db_kernel[(N // BN, B)](AH, BH, L2T, E, EV, ALPHA, BETA, MROW, LWIDX, SLOT, EMAP, ATF, EF, DB, M, N, r,
                                    OBS = obs, GRADS = want_grads, **opts)
    flows = None
    if want_flows:
        FP = torch.empty(M, N, dtype = torch.float32, device = dev)
        FM = torch.empty_like(FP)
        _bw_em_kernel[(M // BM, N // BN)](AH, BH, L2T, E, EV, ALPHA, BETA, MROW, SLOT, EMAP, FP, FM, M, N, B, **opts)
        flows = (FP, FM)
    return EF, vals, ACNT[:Rm], flows
