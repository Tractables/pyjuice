// `LowRank(top_k)` sum layers' FORWARD (`top_k.py`'s `_sum_values`): the node values of a node vector whose scale reads
// the token of the leaf beside it, made tile by tile, the tile never stored:
//
//     W_b[i, j] = 2^(phi'_b[i, j] + l2t[i, j] - m_b[i])        phi' = A' B, A' = log2 e A (fp16-rounded factors),
//                                                                l2t = log2 theta less its row's largest (fp16)
//     Z_b[i, s] = sum_j W_b[i, j] E_b[s, j]          U_b[i, s] = sum_j W_b[i, j] ev_b[j] E_b[s, j]
//     value_b[i] = log sum_s w_b[i, s] U_b[i, s] / Z_b[i, s] + c_b
//
// E = exp(G - its largest over the children) (a slot's largest 1), ev = exp(lc - c) (the children's values over their
// largest, c), w the slots' weights: where the leaf's token is MISSING its probabilities of the slots (`logw`), where it
// is OBSERVED one slot. The row's shifts -- log2 theta's largest, the running maximum m_b[i] kept over the children's
// chunks -- cancel in every ratio; they only keep W in range.
//
// PRECISION, measured on the best Codpula head against float64 (`check_tk_kernel_e2e.py`): phi' from fp16-rounded
// factors (one m16n8k16 MMA a 16 ranks), l2t in fp16 (its row's largest 0: the dominant entries near 0, where fp16 is
// finest), the contraction W [ev E, E] in TF32 (m16n8k8, round to nearest): ~3e-5 nats a masked token's read-out. A
// bf16 contraction moved it by 1.4e-4 (over the 1e-4 gate), an fp16 one underflows (a slot's weights E_b[s, j] W_b[i, j]
// peak where the row's and the slot's maxima do not meet).
//
// SAMPLES SHARING A ROW OF TENSORS (`external_params_rows`: both passes of a conditional one batch) share the tile W,
// which their tensors alone determine: the WORK ITEMS (`tk_items_kernel`) gather them -- a MASKED item a sample whose
// token is missing with up to two observed samples reading the same row (its partners), an OBSERVED item two slots of a
// row and up to two of its observed samples. Without the map every sample reads its own row: the same items.
//
// TWO PERSISTENT LAUNCHES a node vector (`tk_sk_kernel`, stream-K: its notes below), one a kind of item:
//
//  * MASKED items (~250 registers, one CTA an SM): 128 nodes (eight warps of 16) x one item, its children in chunks of
//    64 -- double-buffered cp.async stages of the row's B chunk (rank-major: `ldmatrix.trans` makes the MMA's
//    fragments), E's chunk and the samples' ev chunks. A warp keeps its A' fragments (made once from the caller's fp32
//    A), its accumulators, its rows' running maxima and l2t's fragments of the next chunk (read from global in FRAGMENT
//    ORDER, `LTh`: one 256-byte run a warp's load; the C operand the logits' MMAs start from). The masked sample's slots
//    0 .. 63 by TF32 MMAs, the shared "rest" slot -- the last of E's rows -- and the partners' sums (their slot's E row,
//    their ev) by FP32; ev on the A side (W ev beside W), so both contractions read the same B operand E; the logits'
//    accumulator fragment IS the TF32 MMA's A fragment with the k index permuted (column t <-> child 2t, t + 4 <-> child
//    2t + 1): E's B fragment is then two adjacent children of one slot, one 64-bit shared load, no shuffles.
//  * OBSERVED items: the same tile a slot (a row of tensors, its l2t's reads shared by the two slots), FP32 sums only
//    (z once where a slot's two samples take the same slot of E).
//
// MEASURED (RTX PRO 6000, one CTA an SM alone): the per-SM ingest (~62 GB/s from the L2) does not overlap the compute
// well -- the kernels' time ~ compute + their bytes' time -- so the bytes a CTA reads are cut where they can be: l2t
// fp16 and shared by two slots, A' made in the kernel, one barrier a chunk, the next stage's loads issued right after
// it. The conditional of the best Codpula head (48 blocks, 31 transitions): two passes 271 us a transition, one batch
// through the map 159 us.
//
// The PREP kernel (`tk_prep_kernel`, one launch a node vector) makes the other operands: a row of tensors each, B [Bx,
// KP, N] fp16 (ranks zero-padded to KP = 32, 64, 96 or 128) and E [Bx, S, N] fp32 (TF32-rounded; S = 65); a sample
// each, the children's values from the circuit's (ev [B, N], their largest c [B], -inf where every child's is) and the
// observed slots; l2t in fragment order; the masked samples' weights [Rm, M, SP] fp32 (SP = 72: aligned pairs).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <type_traits>

namespace {

constexpr int TM = 128;                // nodes a CTA
constexpr int NW = 8;                  // warps a CTA (16 nodes each)
constexpr int TN = 64;                 // children a chunk
constexpr int BROW = TN + 8;           // halves a staged B row (ldmatrix rows conflict-free)
constexpr int FS = TN + 8;             // a staged fp32 row's stride (64-bit fragment loads conflict-free)
constexpr int SP = 72;                 // the masked rows' weights' row stride (aligned float2 pairs)
constexpr int NS8 = 8;                 // slot tiles (top_k = 64; the 65th slot by FP32 sums)
constexpr int RPC = 2;                 // observed rows a CTA
constexpr float LOG2E = 1.4426950408889634f;
#define NEG (-CUDART_INF_F)

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// the first k-step: the accumulator starts from l2t's fragment (fp16 pairs: row g's, row g + 8's)
__device__ __forceinline__ void mma_f16_c(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1, uint2 c) {
    const float2 c0 = __half22float2(*reinterpret_cast<const __half2*>(&c.x));
    const float2 c1 = __half22float2(*reinterpret_cast<const __half2*>(&c.y));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// x rounded to TF32 as `cvt.rna` rounds a finite non-negative x (to nearest, ties away): half a TF32 ulp added to its
// bits, the MMA ignoring the low 13 (`cvt.rna.tf32.f32` itself compiles to three instructions, the inf / NaN guard)
__device__ __forceinline__ unsigned to_tf32(float x) { return __float_as_uint(x) + 0x1000u; }

__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// A' fragments of one row from the caller's fp32 A [B, M, r] (`p0` row g's first rank): rows g, g + 8 of the warp's
// 16, ranks (2t, 2t + 1) and (2t + 8, 2t + 9) of each 16 -- log2 e A rounded to fp16, zero past r
template <int NK>
__device__ __forceinline__ void a_frags(unsigned (&af)[NK][4], const float* __restrict__ p0, int r, int t) {
    const float* p1 = p0 + 8L * r;
    #pragma unroll
    for (int ks = 0; ks < NK; ++ks) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = ks * 16 + h * 8 + 2 * t;
            const float x0 = k < r ? __ldg(p0 + k) : 0.f, x1 = k + 1 < r ? __ldg(p0 + k + 1) : 0.f;
            const float y0 = k < r ? __ldg(p1 + k) : 0.f, y1 = k + 1 < r ? __ldg(p1 + k + 1) : 0.f;
            af[ks][2 * h] = h2u(__floats2half2_rn(x0 * LOG2E, x1 * LOG2E));
            af[ks][2 * h + 1] = h2u(__floats2half2_rn(y0 * LOG2E, y1 * LOG2E));
        }
    }
}

// The logits of one chunk into Sx (l2t's fragments the C operand of the first k-step), from the staged B chunk Bs
template <int NK>
__device__ __forceinline__ void logits(float (&Sx)[8][4], const unsigned (&af)[NK][4], const __half* Bs, const uint2 (&lt)[8],
                                       int lane) {
    #pragma unroll
    for (int np = 0; np < 4; ++np) {
        unsigned bf[4];
        ldsm4t(bf, smem_u32(Bs + (lane & 15) * BROW + np * 16 + (lane >> 4) * 8));
        mma_f16_c(Sx[2 * np], af[0], bf[0], bf[1], lt[2 * np]);
        mma_f16_c(Sx[2 * np + 1], af[0], bf[2], bf[3], lt[2 * np + 1]);
    }
    #pragma unroll
    for (int ks = 1; ks < NK; ++ks) {
        #pragma unroll
        for (int np = 0; np < 4; ++np) {
            unsigned bf[4];
            ldsm4t(bf, smem_u32(Bs + (ks * 16 + (lane & 15)) * BROW + np * 16 + (lane >> 4) * 8));
            mma_f16(Sx[2 * np], af[ks], bf[0], bf[1]);
            mma_f16(Sx[2 * np + 1], af[ks], bf[2], bf[3]);
        }
    }
}

// The rows' reference maxima (log2 units) moved -- `rescale(a0, a1)` applied to the accumulators -- only where the
// chunk's exceeds them by more than 2^8 (LAZY: W = 2^(logits - reference) stays below 256, the sums far from fp32's
// range); then W = 2^(logits - reference) in place.
template <typename F>
__device__ __forceinline__ void exp_tile(float (&Sx)[8][4], float& m0, float& m1, F&& rescale) {
    float x0 = NEG, x1 = NEG;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        x0 = fmaxf(x0, fmaxf(Sx[nt][0], Sx[nt][1]));
        x1 = fmaxf(x1, fmaxf(Sx[nt][2], Sx[nt][3]));
    }
    x0 = quad_max(x0); x1 = quad_max(x1);
    const bool up0 = x0 > m0 + 8.f, up1 = x1 > m1 + 8.f;
    if (__any_sync(0xffffffffu, up0 || up1)) {
        rescale(up0 ? exp2f(m0 - x0) : 1.f, up1 ? exp2f(m1 - x1) : 1.f);
        m0 = up0 ? x0 : m0; m1 = up1 ? x1 : m1;
    }
    const float h0 = m0 == NEG ? 0.f : m0, h1 = m1 == NEG ? 0.f : m1;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        Sx[nt][0] = exp2f(Sx[nt][0] - h0); Sx[nt][1] = exp2f(Sx[nt][1] - h0);
        Sx[nt][2] = exp2f(Sx[nt][2] - h1); Sx[nt][3] = exp2f(Sx[nt][3] - h1);
    }
}

template <int NK>
struct MaskedGeo {                     // a stage: { B chunk [KP][BROW] fp16, E chunk [65][FS] fp32, ev chunks [3][TN] fp32 }
    static constexpr int KP = 16 * NK;
    static constexpr int B_BYTES = KP * BROW * 2;
    static constexpr int E_BYTES = (8 * NS8 + 1) * FS * 4;
    static constexpr int STAGE = B_BYTES + E_BYTES + 3 * TN * 4;
    static constexpr int BYTES = 2 * STAGE;
};

template <int NK>
struct ObservedGeo {                   // a stage: 2 slots x { B chunk [KP][BROW] fp16, E rows [2][TN] fp32, ev [2][TN] fp32 }
    static constexpr int KP = 16 * NK;
    static constexpr int B_BYTES = KP * BROW * 2;
    static constexpr int ROW = B_BYTES + 4 * TN * 4;
    static constexpr int STAGE = 2 * ROW;
    static constexpr int BYTES = 2 * STAGE;
};

// A pass of a node vector is two persistent launches (`tk_sk_kernel`, STREAM-K) side by side on two streams, the masked
// items' tiles (128 nodes x an item) and the observed items', 16 chunks of 64 children each: the SMs split between the
// two kinds by their work (WM, WO: a chunk's cost each, measured in place) -- each launch a CTA an SM, the CTAs past
// its kind's share leaving at once (the counts are the device's) -- and each kind's tiles cut into one equal share a
// CTA: no wave's tail. (One launch of both kinds, weighted by their costs, was SLOWER: its code at the 255-register cap spilled
// in the observed loop, 13.7K instructions; the partners' sums stretched the masked tiles.) A tile cut between CTAs: its earlier parts'
// sums are PARTIALS (each CTA's at most one, in the scratch `ws`, a flag set), the CTA with its last chunk adds them in
// (each part's running maxima its own: rescaled to the larger) and makes the values. A CTA runs its partial first and
// waits on lower CTAs only (dispatched first): no deadlock, short waits.

constexpr int WM = 7, WO = 5;          // a masked chunk's cost and an observed one's (67 : 48 us a tile in place)

struct MState {                         // a thread's sums of a masked tile: its rows g, g + 8
    float m0, m1, zr0, zr1, ur0, ur1;
    float Zc[NS8][4], Uc[NS8][4];
    float pz[2][2], pu[2][2];
};
struct OState {                         // ... of an observed tile: two slots x two samples x its rows
    float m[2][2], zs[2][2][2], us[2][2][2];
};
constexpr int WS_M = 6 + 8 * NS8 * 2 + 8, WS_O = 4 + 16;   // a partial's floats a thread: masked, observed

struct TkArgs {
    const float* A; const __half* BP; const uint2* LTh; const float* E; const float* ev; const float* cmax;
    const int4* mitems; const int4* oitems; const int* noitems; const int* slot32; const float* logw;
    float* out; float* ws; int* flags;
    long nid0, ostride, sA; int r, M, N, S, Rm;                    // sA: A's batch stride (its rows contiguous)
};

// the masked item y's sums over its tile x's chunks c0 .. c1 - 1
template <int NK>
__device__ __forceinline__ void masked_run(MState& St, const TkArgs& a, int y, int x, int c0, int c1,
                                           unsigned char* smem) {
    using G = MaskedGeo<NK>;
    constexpr int KP = G::KP;
    const int4 item = a.mitems[y];
    const int b = item.x, e = item.y, o1 = item.z, o2 = item.w;
    const int s1 = o1 >= 0 ? a.slot32[o1] : 0, s2 = o2 >= 0 ? a.slot32[o2] : 0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int N = a.N, S = a.S, nchunks = N / TN;
    const int node0 = x * TM + warp * 16 + g;

    constexpr int NB = KP * (TN / 8), NE = (8 * NS8 + 1) * (TN / 4), NSEG = NB + NE + 3 * (TN / 4);
    auto load_seg = [&](int q, int st, int j0) {
        const unsigned bs = smem_u32(smem + st * G::STAGE);
        if (q < NB) {
            const int k = q / (TN / 8), s = q % (TN / 8);
            cp16(bs + (k * BROW + s * 8) * 2, a.BP + ((long)e * KP + k) * N + j0 + s * 8);
        } else if (q < NB + NE) {
            const int rr = (q - NB) / (TN / 4), s = (q - NB) % (TN / 4);
            const int srow = rr < 8 * NS8 ? rr : S - 1;
            cp16(bs + G::B_BYTES + (rr * FS + s * 4) * 4, a.E + ((long)e * S + srow) * N + j0 + s * 4);
        } else if (q < NSEG) {
            const int v = (q - NB - NE) / (TN / 4), s = (q - NB - NE) % (TN / 4);
            const int row = v == 0 ? b : (v == 1 ? o1 : o2);
            if (row >= 0) cp16(bs + G::B_BYTES + G::E_BYTES + (v * TN + s * 4) * 4, a.ev + (long)row * N + j0 + s * 4);
        }
    };
    __syncthreads();                                                         // (the last tile's stages read by all)
    for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, 0, c0 * TN);
    cp_commit();

    unsigned af[NK][4];
    a_frags<NK>(af, a.A + (long)e * a.sA + (long)node0 * a.r, a.r, t);
    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8) * 32 + lane;
    uint2 lt[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + (c0 * 8 + nt) * 32);

    St.m0 = St.m1 = NEG;
    St.zr0 = St.zr1 = St.ur0 = St.ur1 = 0.f;
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn)
        #pragma unroll
        for (int k = 0; k < 4; ++k) St.Zc[sn][k] = St.Uc[sn][k] = 0.f;
    #pragma unroll
    for (int k = 0; k < 2; ++k) St.pz[k][0] = St.pz[k][1] = St.pu[k][0] = St.pu[k][1] = 0.f;

    for (int jc = c0; jc < c1; ++jc) {
        const int st = (jc - c0) & 1;
        const bool more = jc + 1 < c1;
        cp_wait<0>();
        __syncthreads();                     // (this chunk's stage complete; the other free: every warp past the last chunk)
        if (more) {
            for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, st ^ 1, (jc + 1) * TN);
            cp_commit();
        }
        const unsigned char* p = smem + st * G::STAGE;
        const float* Es = reinterpret_cast<const float*>(p + G::B_BYTES);
        const float* Vs = reinterpret_cast<const float*>(p + G::B_BYTES + G::E_BYTES);

        float Sx[8][4];
        logits<NK>(Sx, af, reinterpret_cast<const __half*>(p), lt, lane);
        exp_tile(Sx, St.m0, St.m1, [&](float a0, float a1) {
            #pragma unroll
            for (int sn = 0; sn < NS8; ++sn) {
                St.Zc[sn][0] *= a0; St.Zc[sn][1] *= a0; St.Zc[sn][2] *= a1; St.Zc[sn][3] *= a1;
                St.Uc[sn][0] *= a0; St.Uc[sn][1] *= a0; St.Uc[sn][2] *= a1; St.Uc[sn][3] *= a1;
            }
            St.zr0 *= a0; St.ur0 *= a0; St.zr1 *= a1; St.ur1 *= a1;
            #pragma unroll
            for (int k = 0; k < 2; ++k) { St.pz[k][0] *= a0; St.pu[k][0] *= a0; St.pz[k][1] *= a1; St.pu[k][1] *= a1; }
        });

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            if (more) lt[nt] = __ldg(ltp + ((jc + 1) * 8 + nt) * 32);
            const float2 ec = *reinterpret_cast<const float2*>(Vs + nt * 8 + 2 * t);
            const float w00 = Sx[nt][0], w01 = Sx[nt][1], w10 = Sx[nt][2], w11 = Sx[nt][3];
            const float u00 = w00 * ec.x, u01 = w01 * ec.y, u10 = w10 * ec.x, u11 = w11 * ec.y;
            const unsigned az0 = to_tf32(w00), az1 = to_tf32(w10), az2 = to_tf32(w01), az3 = to_tf32(w11);
            const unsigned au0 = to_tf32(u00), au1 = to_tf32(u10), au2 = to_tf32(u01), au3 = to_tf32(u11);
            #pragma unroll
            for (int sn = 0; sn < NS8; ++sn) {
                const float2 bb = *reinterpret_cast<const float2*>(Es + (sn * 8 + g) * FS + nt * 8 + 2 * t);
                const unsigned b0 = __float_as_uint(bb.x), b1 = __float_as_uint(bb.y);
                mma_tf32(St.Zc[sn], az0, az1, az2, az3, b0, b1);
                mma_tf32(St.Uc[sn], au0, au1, au2, au3, b0, b1);
            }
            const float2 rr = *reinterpret_cast<const float2*>(Es + (8 * NS8) * FS + nt * 8 + 2 * t);
            St.zr0 += w00 * rr.x + w01 * rr.y; St.zr1 += w10 * rr.x + w11 * rr.y;
            St.ur0 += u00 * rr.x + u01 * rr.y; St.ur1 += u10 * rr.x + u11 * rr.y;
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                if ((k == 0 ? o1 : o2) < 0) continue;
                const float2 pr = *reinterpret_cast<const float2*>(Es + (k == 0 ? s1 : s2) * FS + nt * 8 + 2 * t);
                const float2 pv = *reinterpret_cast<const float2*>(Vs + (1 + k) * TN + nt * 8 + 2 * t);
                const float er0 = pv.x * pr.x, er1 = pv.y * pr.y;
                St.pz[k][0] += w00 * pr.x + w01 * pr.y; St.pz[k][1] += w10 * pr.x + w11 * pr.y;
                St.pu[k][0] += w00 * er0 + w01 * er1; St.pu[k][1] += w10 * er0 + w11 * er1;
            }
        }
    }
}

// the masked item y's values over its tile x from its sums: log sum_s w U / Z + c; the partners' log u / z + c
__device__ __forceinline__ void masked_values(const MState& St, const TkArgs& a, int y, int x) {
    const int4 item = a.mitems[y];
    const int b = item.x, o1 = item.z, o2 = item.w;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int M = a.M, node0 = x * TM + warp * 16 + g, node1 = node0 + 8;
    const float* lw0 = a.logw + ((long)y * M + node0) * SP + 2 * t;
    const float* lw1 = lw0 + 8L * SP;
    float2 w0[NS8], w1[NS8];
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        w0[sn] = __ldg(reinterpret_cast<const float2*>(lw0 + sn * 8));
        w1[sn] = __ldg(reinterpret_cast<const float2*>(lw1 + sn * 8));
    }
    const float wr0 = __ldg(lw0 - 2 * t + 8 * NS8), wr1 = __ldg(lw1 - 2 * t + 8 * NS8);
    const float zr0 = quad_sum(St.zr0), zr1 = quad_sum(St.zr1), ur0 = quad_sum(St.ur0), ur1 = quad_sum(St.ur1);
    float R0 = 0.f, R1 = 0.f;
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        if (St.Zc[sn][0] > 0.f) R0 += __expf(w0[sn].x) * (St.Uc[sn][0] / St.Zc[sn][0]);
        if (St.Zc[sn][1] > 0.f) R0 += __expf(w0[sn].y) * (St.Uc[sn][1] / St.Zc[sn][1]);
        if (St.Zc[sn][2] > 0.f) R1 += __expf(w1[sn].x) * (St.Uc[sn][2] / St.Zc[sn][2]);
        if (St.Zc[sn][3] > 0.f) R1 += __expf(w1[sn].y) * (St.Uc[sn][3] / St.Zc[sn][3]);
    }
    R0 = quad_sum(R0); R1 = quad_sum(R1);
    if (zr0 > 0.f) R0 += __expf(wr0) * (ur0 / zr0);
    if (zr1 > 0.f) R1 += __expf(wr1) * (ur1 / zr1);
    float q0[2], q1[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const float z0 = quad_sum(St.pz[k][0]), z1 = quad_sum(St.pz[k][1]);
        const float u0 = quad_sum(St.pu[k][0]), u1 = quad_sum(St.pu[k][1]);
        q0[k] = z0 > 0.f ? u0 / z0 : 0.f;
        q1[k] = z1 > 0.f ? u1 / z1 : 0.f;
    }
    if (t == 0) {
        float* out = a.out + a.nid0 * a.ostride;
        const float c = a.cmax[b];
        out[node0 * a.ostride + b] = (R0 > 0.f && c > NEG) ? __logf(R0) + c : NEG;
        out[node1 * a.ostride + b] = (R1 > 0.f && c > NEG) ? __logf(R1) + c : NEG;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int o = k == 0 ? o1 : o2;
            if (o < 0) continue;
            const float co = a.cmax[o];
            out[node0 * a.ostride + o] = (q0[k] > 0.f && co > NEG) ? __logf(q0[k]) + co : NEG;
            out[node1 * a.ostride + o] = (q1[k] > 0.f && co > NEG) ? __logf(q1[k]) + co : NEG;
        }
    }
}

// the observed item y's sums over its tile x's chunks c0 .. c1 - 1
template <int NK>
__device__ __forceinline__ void observed_run(OState& St, const TkArgs& a, int y, int x, int c0, int c1,
                                             unsigned char* smem) {
    using G = ObservedGeo<NK>;
    constexpr int KP = G::KP;
    const int4 sa = a.oitems[2 * y], sb = a.oitems[2 * y + 1];
    const int ek[2] = {sa.x, sb.x};
    const int rk[2][2] = {{sa.y, sa.z}, {sb.y, sb.z}};
    const bool two = sb.x >= 0;
    int sk[2][2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int j = 0; j < 2; ++j) sk[k][j] = rk[k][j] >= 0 ? a.slot32[rk[k][j]] : 0;
    const bool same[2] = {rk[0][1] >= 0 && sk[0][0] == sk[0][1], rk[1][1] >= 0 && sk[1][0] == sk[1][1]};
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, t = lane & 3, g = lane >> 2;
    const int N = a.N, S = a.S, nchunks = N / TN;
    const int node0 = x * TM + warp * 16 + g;

    constexpr int NB = KP * (TN / 8), NSR = NB + 4 * (TN / 4), NSEG = 2 * NSR;
    auto load_seg = [&](int q, int st, int j0) {
        const int k = q / NSR, qq = q % NSR;
        const int ee = k == 0 ? ek[0] : ek[1];
        if (ee < 0) return;
        const unsigned bs = smem_u32(smem + st * G::STAGE + k * G::ROW);
        if (qq < NB) {
            const int kk = qq / (TN / 8), s = qq % (TN / 8);
            cp16(bs + (kk * BROW + s * 8) * 2, a.BP + ((long)ee * KP + kk) * N + j0 + s * 8);
        } else {
            const int v = (qq - NB) / (TN / 4), s = (qq - NB) % (TN / 4);      // v: E row 0, 1, ev 0, 1
            const int j = v & 1;
            const int row = k == 0 ? (j == 0 ? rk[0][0] : rk[0][1]) : (j == 0 ? rk[1][0] : rk[1][1]);
            if (row < 0) return;
            const int sl = k == 0 ? (j == 0 ? sk[0][0] : sk[0][1]) : (j == 0 ? sk[1][0] : sk[1][1]);
            const float* src = v < 2 ? a.E + ((long)ee * S + sl) * N : a.ev + (long)row * N;
            cp16(bs + G::B_BYTES + (v * TN + s * 4) * 4, src + j0 + s * 4);
        }
    };
    __syncthreads();
    for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, 0, c0 * TN);
    cp_commit();

    unsigned af[2][NK][4];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        if (k == 0 || two) a_frags<NK>(af[k], a.A + (long)ek[k] * a.sA + (long)node0 * a.r, a.r, t);
    const uint2* ltp = a.LTh + ((long)(x * NW + warp) * nchunks * 8) * 32 + lane;
    uint2 lt[8];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + (c0 * 8 + nt) * 32);

    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        St.m[k][0] = St.m[k][1] = NEG;
        #pragma unroll
        for (int j = 0; j < 2; ++j) St.zs[k][j][0] = St.zs[k][j][1] = St.us[k][j][0] = St.us[k][j][1] = 0.f;
    }

    for (int jc = c0; jc < c1; ++jc) {
        const int st = (jc - c0) & 1;
        const bool more = jc + 1 < c1;
        cp_wait<0>();
        __syncthreads();
        if (more) {
            for (int q = tid; q < NSEG; q += NW * 32) load_seg(q, st ^ 1, (jc + 1) * TN);
            cp_commit();
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            if (k == 1 && !two) continue;                                    // (a constant trip count: the state in registers)
            const unsigned char* p = smem + st * G::STAGE + k * G::ROW;
            const float* Es = reinterpret_cast<const float*>(p + G::B_BYTES);
            float Sx[8][4];
            logits<NK>(Sx, af[k], reinterpret_cast<const __half*>(p), lt, lane);
            if (more && (k == 1 || !two)) {                                  // (l2t read by the last slot: the next chunk's)
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) lt[nt] = __ldg(ltp + ((jc + 1) * 8 + nt) * 32);
            }
            exp_tile(Sx, St.m[k][0], St.m[k][1], [&](float a0, float a1) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    St.zs[k][j][0] *= a0; St.us[k][j][0] *= a0; St.zs[k][j][1] *= a1; St.us[k][j][1] *= a1;
                }
            });
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (rk[k][j] < 0) continue;
                const bool zj = !(j == 1 && same[k]);
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const float2 rr = *reinterpret_cast<const float2*>(Es + j * TN + nt * 8 + 2 * t);
                    const float2 e2 = *reinterpret_cast<const float2*>(Es + (2 + j) * TN + nt * 8 + 2 * t);
                    const float er0 = e2.x * rr.x, er1 = e2.y * rr.y;
                    if (zj) {
                        St.zs[k][j][0] += Sx[nt][0] * rr.x + Sx[nt][1] * rr.y;
                        St.zs[k][j][1] += Sx[nt][2] * rr.x + Sx[nt][3] * rr.y;
                    }
                    St.us[k][j][0] += Sx[nt][0] * er0 + Sx[nt][1] * er1;
                    St.us[k][j][1] += Sx[nt][2] * er0 + Sx[nt][3] * er1;
                }
            }
        }
    }
}

__device__ __forceinline__ void observed_values(const OState& St, const TkArgs& a, int y, int x) {
    const int4 sa = a.oitems[2 * y], sb = a.oitems[2 * y + 1];
    const int rk[2][2] = {{sa.y, sa.z}, {sb.y, sb.z}};
    const bool two = sb.x >= 0;
    int sk[2][2];
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int j = 0; j < 2; ++j) sk[k][j] = rk[k][j] >= 0 ? a.slot32[rk[k][j]] : 0;
    const bool same[2] = {rk[0][1] >= 0 && sk[0][0] == sk[0][1], rk[1][1] >= 0 && sk[1][0] == sk[1][1]};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int node0 = x * TM + warp * 16 + g, node1 = node0 + 8;
    float* out = a.out + a.nid0 * a.ostride;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = rk[k][j];
            const bool z_of_0 = j == 1 && same[k];                           // (a select, not an index: in registers)
            const float z0 = quad_sum(z_of_0 ? St.zs[k][0][0] : St.zs[k][j][0]);
            const float z1 = quad_sum(z_of_0 ? St.zs[k][0][1] : St.zs[k][j][1]);
            const float u0 = quad_sum(St.us[k][j][0]), u1 = quad_sum(St.us[k][j][1]);
            if (t == 0 && row >= 0 && (k == 0 || two)) {
                const float R0 = z0 > 0.f ? u0 / z0 : 0.f, R1 = z1 > 0.f ? u1 / z1 : 0.f;
                const float c = a.cmax[row];
                out[node0 * a.ostride + row] = (R0 > 0.f && c > NEG) ? __logf(R0) + c : NEG;
                out[node1 * a.ostride + row] = (R1 > 0.f && c > NEG) ? __logf(R1) + c : NEG;
            }
        }
    }
}

// a partial's floats as one list, by value (`f(q, v, row)` returns the new value -- no member's address taken: a
// reference into the state kept it in local memory): the masked state's (rows -1: maxima; 0, 1: row g's, row g + 8's
// sums), the observed one's (maxima -1 .. -4; sums by their slot and row 0 .. 3)
template <typename F>
__device__ __forceinline__ void each_float(MState& S, F&& f) {
    S.m0 = f(0, S.m0, -1); S.m1 = f(1, S.m1, -1);
    S.zr0 = f(2, S.zr0, 0); S.zr1 = f(3, S.zr1, 1); S.ur0 = f(4, S.ur0, 0); S.ur1 = f(5, S.ur1, 1);
    #pragma unroll
    for (int sn = 0; sn < NS8; ++sn) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            S.Zc[sn][k] = f(6 + (sn * 4 + k) * 2, S.Zc[sn][k], k >> 1);
            S.Uc[sn][k] = f(7 + (sn * 4 + k) * 2, S.Uc[sn][k], k >> 1);
        }
    }
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            S.pz[k][h] = f(6 + 8 * NS8 + (k * 2 + h) * 2, S.pz[k][h], h);
            S.pu[k][h] = f(7 + 8 * NS8 + (k * 2 + h) * 2, S.pu[k][h], h);
        }
    }
}
template <typename F>
__device__ __forceinline__ void each_float(OState& S, F&& f) {
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        S.m[k][0] = f(2 * k, S.m[k][0], -1);
        S.m[k][1] = f(2 * k + 1, S.m[k][1], -1);
    }
    #pragma unroll
    for (int k = 0; k < 2; ++k)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int q = 4 + ((k * 2 + j) * 2 + h) * 2;
                S.zs[k][j][h] = f(q, S.zs[k][j][h], 2 * k + h);
                S.us[k][j][h] = f(q + 1, S.us[k][j][h], 2 * k + h);
            }
}

// the partial into this CTA's slot of the scratch, then its kind's flag
// a CTA's partial slot: the masked kind's from the scratch's start, the observed kind's after gridDim.x masked slots
__device__ __forceinline__ float* ws_slot(const TkArgs& a, int kp, bool masked) {
    return a.ws + (masked ? (long)kp * WS_M : (long)gridDim.x * WS_M + (long)kp * WS_O) * (NW * 32) + threadIdx.x;
}

template <typename St>
__device__ __forceinline__ void publish(St& S, const TkArgs& a, int* flags) {
    float* w = ws_slot(a, blockIdx.x, std::is_same<St, MState>::value);
    each_float(S, [&](int q, float v, int) { __stcg(w + q * (NW * 32), v); return v; });
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(flags + blockIdx.x, 1);
}

__device__ __forceinline__ void wait_flag(const int* f) {
    if (threadIdx.x == 0) {
        int v;
        do {
            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
            if (v == 0) __nanosleep(64);
        } while (v == 0);
    }
    __syncthreads();
}

// CTA kp's partial added into S: each row's sums rescaled to the larger of the two maxima
__device__ __forceinline__ void fold_in(MState& S, const TkArgs& a, int kp) {
    const float* w = ws_slot(a, kp, true);
    const float pm0 = __ldcg(w), pm1 = __ldcg(w + NW * 32);
    const float n0 = fmaxf(S.m0, pm0), n1 = fmaxf(S.m1, pm1);
    const float fs0 = n0 == NEG ? 1.f : exp2f(S.m0 - n0), fs1 = n1 == NEG ? 1.f : exp2f(S.m1 - n1);
    const float fp0 = n0 == NEG ? 1.f : exp2f(pm0 - n0), fp1 = n1 == NEG ? 1.f : exp2f(pm1 - n1);
    each_float(S, [&](int q, float v, int row) {
        if (row < 0) return v;
        return v * (row == 0 ? fs0 : fs1) + __ldcg(w + q * (NW * 32)) * (row == 0 ? fp0 : fp1);
    });
    S.m0 = n0; S.m1 = n1;
}
__device__ __forceinline__ void fold_in(OState& S, const TkArgs& a, int kp) {
    const float* w = ws_slot(a, kp, false);
    float fs0, fs1, fs2, fs3, fp0, fp1, fp2, fp3, nm0, nm1, nm2, nm3;
    auto pair = [&](float mine, float theirs, float& nm, float& fs, float& fp) {
        nm = fmaxf(mine, theirs);
        fs = nm == NEG ? 1.f : exp2f(mine - nm);
        fp = nm == NEG ? 1.f : exp2f(theirs - nm);
    };
    pair(S.m[0][0], __ldcg(w), nm0, fs0, fp0);
    pair(S.m[0][1], __ldcg(w + NW * 32), nm1, fs1, fp1);
    pair(S.m[1][0], __ldcg(w + 2 * NW * 32), nm2, fs2, fp2);
    pair(S.m[1][1], __ldcg(w + 3 * NW * 32), nm3, fs3, fp3);
    each_float(S, [&](int q, float v, int row) {
        if (row < 0) return v;
        const float a_ = row == 0 ? fs0 : (row == 1 ? fs1 : (row == 2 ? fs2 : fs3));
        const float b_ = row == 0 ? fp0 : (row == 1 ? fp1 : (row == 2 ? fp2 : fp3));
        return v * a_ + __ldcg(w + q * (NW * 32)) * b_;
    });
    S.m[0][0] = nm0; S.m[0][1] = nm1; S.m[1][0] = nm2; S.m[1][1] = nm3;
}

template <int NK, bool MASKED>
__global__ void __launch_bounds__(NW * 32, 1) tk_sk_kernel(const TkArgs a) {
    extern __shared__ __align__(128) unsigned char smem[];
    using St = typename std::conditional<MASKED, MState, OState>::type;
    const int nx = a.M / TM, nch = a.N / TN;
    const long Tm = (long)nx * a.Rm, To = (long)nx * (*a.noitems);
    // the SMs each kind takes: its share of both kinds' cost (all where the other has none), at most a CTA a tile -- a
    // kind's tiles split only where its share is short of them (a split's prologue and partial cost ~4 us a part)
    const long cm = Tm * WM, co = To * WO, G = gridDim.x;
    long Pm = cm == 0 ? 0 : (co == 0 ? G : max(1L, min(G - 1, (G * cm + (cm + co) / 2) / (cm + co))));
    long Po = G - Pm;
    Pm = min(Pm, Tm);
    Po = min(Po, To);
    const int P = (int)(MASKED ? Pm : Po), k = blockIdx.x;
    if (k >= P) return;
    const long T = MASKED ? Tm : To;
    const long W = T * nch;
    auto gstart = [&](long kk) -> long { return W * kk / P; };
    const long g0 = gstart(k), g1 = gstart(k + 1);
    if (g0 >= g1) return;
    const long ta = g0 / nch, tb = (g1 - 1) / nch;
    const int ca0 = (int)(g0 % nch), cb1 = (int)((g1 - 1) % nch) + 1;
    const bool part = cb1 < nch;                                             // (the share ends inside a tile)
    const long last = part ? tb - 1 : tb;                                    // (its last whole tile after the first)
    const int nseg = (part && tb == ta) ? 1 : (part ? 1 : 0) + 1 + (int)(last > ta ? last - ta : 0);
    int* flags = a.flags + (MASKED ? 0 : gridDim.x);                         // (each kind its own: by the grid, not the share)
    for (int si = 0; si < nseg; ++si) {
        long tt; int c0, c1, mode;                                           // mode 0 whole, 1 partial, 2 the tile's last part
        if (part && si == 0) {
            tt = tb; c0 = tb == ta ? ca0 : 0; c1 = cb1; mode = 1;
        } else {
            const int j = si - (part ? 1 : 0);
            if (j == 0) { tt = ta; c0 = ca0; c1 = ta == tb ? cb1 : nch; mode = ca0 > 0 ? 2 : 0; }
            else { tt = ta + j; c0 = 0; c1 = nch; mode = 0; }
        }
        const int item = (int)(tt / nx), x = (int)(tt % nx);
        St S;
        if constexpr (MASKED) masked_run<NK>(S, a, item, x, c0, c1, smem);
        else observed_run<NK>(S, a, item, x, c0, c1, smem);
        if (mode == 1) {
            publish(S, a, flags);
            continue;
        }
        if (mode == 2)
            for (int kp = k - 1; kp >= 0; --kp) {
                const long s0 = gstart(kp);
                if (s0 >= gstart(kp + 1)) continue;                          // (an empty share)
                wait_flag(flags + kp);
                fold_in(S, a, kp);
                if (s0 <= tt * nch) break;                                   // (the tile's first part)
            }
        if constexpr (MASKED) masked_values(S, a, item, x);
        else observed_values(S, a, item, x);
    }
}

// The operands, one launch: blockIdx.y the task, blockIdx.x grid-stride over its items. B and G a row of tensors each
// (Bx of them), the children's values a sample each (B).
__global__ void tk_prep_kernel(
        const float* __restrict__ Bm, long sB, const float* __restrict__ Gt, long sG, const float* __restrict__ em,
        long em_stride,
        const long* __restrict__ eid, const bool* __restrict__ has, const float* __restrict__ theta,
        const float* __restrict__ logwin, __half* __restrict__ BP, float* __restrict__ E, float* __restrict__ ev,
        float* __restrict__ cmax, uint2* __restrict__ LTh, float* __restrict__ logw,
        int Bx, int B, int M, int N, int r, int KP, int S, int Rm) {
    const int task = blockIdx.y;
    const long tid0 = (long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long)gridDim.x * blockDim.x;
    const long wid = tid0 >> 5, nwarp = nth >> 5;
    const int lane = threadIdx.x & 31;
    if (task == 0) {                                                         // B, fp16, its ranks padded
        const long total = (long)Bx * KP * (N / 2);
        for (long q = tid0; q < total; q += nth) {
            const int j2 = (int)(q % (N / 2));
            const long bk = q / (N / 2);
            const int k = (int)(bk % KP);
            const long bb = bk / KP;
            float2 v = make_float2(0.f, 0.f);
            if (k < r) v = reinterpret_cast<const float2*>(Bm + bb * sB + (long)k * N)[j2];
            reinterpret_cast<__half2*>(BP)[q] = __floats2half2_rn(v.x, v.y);
        }
    } else if (task == 1) {                                                  // E = exp(G - max) in TF32, a warp a row
        for (long row = wid; row < (long)Bx * S; row += nwarp) {
            const float* gr = Gt + (row / S) * sG + (row % S) * N;
            float mx = NEG;
            for (int j = lane; j < N; j += 32) mx = fmaxf(mx, gr[j]);
            mx = warp_max(mx);
            const float sh = isfinite(mx) ? mx : 0.f;
            for (int j = lane; j < N; j += 32) {
                const unsigned u = __float_as_uint(__expf(gr[j] - sh));
                E[row * N + j] = __uint_as_float((u + 0x1000u) & ~0x1FFFu);
            }
        }
    } else if (task == 2) {                                                  // the children's values, a warp a sample
        for (long bb = wid; bb < B; bb += nwarp) {
            float mx = NEG;
            for (int j = lane; j < N; j += 32)
                mx = fmaxf(mx, (has == nullptr || has[j]) ? em[eid[j] * em_stride + bb] : NEG);
            mx = warp_max(mx);
            const float sh = isfinite(mx) ? mx : 0.f;
            for (int j = lane; j < N; j += 32) {
                const float v = (has == nullptr || has[j]) ? em[eid[j] * em_stride + bb] : NEG;
                ev[bb * N + j] = __expf(v - sh);
            }
            if (lane == 0) cmax[bb] = mx;
        }
    } else if (task == 3) {                                                  // l2t in fragment order, a CTA a 16-row tile
        __shared__ float rmax[16];
        const int nch = N / TN;
        const int warp = threadIdx.x >> 5;
        for (int tile = blockIdx.x; tile < M / 16; tile += gridDim.x) {
            for (int rw = warp; rw < 16; rw += blockDim.x >> 5) {
                const float* tr = theta + ((long)tile * 16 + rw) * N;
                float mx = 0.f;
                for (int j = lane; j < N; j += 32) mx = fmaxf(mx, tr[j]);
                mx = warp_max(mx);
                if (lane == 0) rmax[rw] = mx > 0.f ? log2f(mx) : 0.f;
            }
            __syncthreads();
            for (int q = threadIdx.x; q < nch * 8 * 32; q += blockDim.x) {
                const int ln = q & 31, nt = (q >> 5) & 7, jc = q >> 8;
                const int g = ln >> 2, t = ln & 3;
                const long i = (long)tile * 16 + g;
                const int j = jc * TN + nt * 8 + 2 * t;
                const float2 u = *reinterpret_cast<const float2*>(theta + i * N + j);
                const float2 w = *reinterpret_cast<const float2*>(theta + (i + 8) * N + j);
                const float s0 = rmax[g], s1 = rmax[g + 8];
                uint2 o;
                o.x = h2u(__floats2half2_rn(log2f(u.x) - s0, log2f(u.y) - s0));
                o.y = h2u(__floats2half2_rn(log2f(w.x) - s1, log2f(w.y) - s1));
                LTh[((long)tile * nch * 8) * 32 + q] = o;
            }
            __syncthreads();
        }
    } else {                                                                 // the masked samples' weights, fp32, padded
        const long total = (long)Rm * M * SP;
        for (long q = tid0; q < total; q += nth) {
            const int s = (int)(q % SP);
            logw[q] = s < S ? logwin[(q / SP) * S + s] : NEG;
        }
    }
}

// The work items (one CTA): a masked item a masked sample, its row of tensors and up to two observed samples reading
// that row (its partners: the row's observed samples 2i, 2i + 1 for its i-th masked one); the row's other observed
// samples in slots of two, an observed item two slots. Every sample's place among its row's samples of its kind by
// scans in shared memory (deterministic: in the samples' order). `emap` the row each sample reads (null: its own).
__global__ void __launch_bounds__(1024) tk_items_kernel(
        const long* __restrict__ emap, const long* __restrict__ rows, const long* __restrict__ slot64,
        int4* __restrict__ mitems, int4* __restrict__ oitems, int* __restrict__ noitems, int* __restrict__ slot32,
        int* __restrict__ flags, int B, int Rm, int P) {
    extern __shared__ int sm[];
    int* se = sm;                                                            // [B] the row each sample reads
    int* sk = se + B;                                                        // [B] its kind (1: masked)
    int* sp = sk + B;                                                        // [B] masked: its item
    int* sx = sp + B;                                                        // [B] its place among its row's of its kind
    int* sl = sx + B;                                                        // [B] observed: 1 a slot's first sample
    __shared__ int nslots;
    const int tid = threadIdx.x, nth = blockDim.x;
    for (int b = tid; b < B; b += nth) {
        se[b] = emap != nullptr ? (int)emap[b] : b;
        sk[b] = 0;
        sl[b] = 0;
        slot32[b] = (int)slot64[b];
    }
    if (tid == 0) nslots = 0;
    for (int i = tid; i < P; i += nth) flags[i] = 0;                         // (the forward's stream-K flags)
    __syncthreads();
    for (int q = tid; q < Rm; q += nth) {
        sk[rows[q]] = 1;
        sp[rows[q]] = q;
    }
    __syncthreads();
    for (int b = tid; b < B; b += nth) {
        const int e = se[b], k = sk[b];
        int ix = 0;
        for (int c = 0; c < b; ++c) ix += (se[c] == e && sk[c] == k);
        sx[b] = ix;
        if (k) mitems[sp[b]] = make_int4(b, e, -1, -1);
    }
    __syncthreads();
    for (int b = tid; b < B; b += nth) {
        if (sk[b]) continue;
        const int e = se[b], io = sx[b];
        int km = 0;
        for (int c = 0; c < B; ++c) km += (se[c] == e && sk[c]);
        if (io < 2 * km) {                                                   // a partner of the row's masked sample io / 2
            for (int c = 0; c < B; ++c)
                if (se[c] == e && sk[c] && sx[c] == io / 2) {
                    reinterpret_cast<int*>(mitems + sp[c])[2 + (io & 1)] = b;
                    break;
                }
        } else if (((io - 2 * km) & 1) == 0) {
            sl[b] = 1;
        }
    }
    __syncthreads();
    for (int b = tid; b < B; b += nth) {
        if (!sl[b]) continue;
        int s = 0;
        for (int c = 0; c < b; ++c) s += sl[c];
        int b2 = -1;
        for (int c = b + 1; c < B; ++c)
            if (se[c] == se[b] && !sk[c] && sx[c] == sx[b] + 1) {
                b2 = c;
                break;
            }
        oitems[s] = make_int4(se[b], b, b2, 0);
        atomicAdd(&nslots, 1);
    }
    __syncthreads();
    if (tid == 0) {
        const int ns = nslots;
        if (ns & 1) oitems[ns] = make_int4(-1, -1, -1, 0);
        *noitems = (ns + 1) / 2;
    }
}

void check(const torch::Tensor& x, at::ScalarType dt, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == dt, "top_k: ", name,
                " must be a contiguous CUDA tensor of ", dt);
}

// a [batch, rows, n] tensor whose rows are contiguous and adjacent (any batch stride: a view of a group's tensor)
void check_rows(const torch::Tensor& x, at::ScalarType dt, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.scalar_type() == dt && x.stride(2) == 1 && x.stride(1) == x.size(2) &&
                x.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "top_k: ", name,
                " must be a CUDA tensor of ", dt, " with contiguous rows, 16-byte aligned (batch stride a multiple of 4)");
}

int padded_rank(int r) { return r <= 32 ? 32 : (r <= 64 ? 64 : (r <= 96 ? 96 : 128)); }

template <typename K>
void smem_attr(K kern, int bytes, int dev, bool* done) {
    if (!done[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        done[dev] = true;
    }
}

}  // namespace

// The operands of one node vector: B [Bx, r, N] and G [Bx, S, N] (fp32) a row of tensors each; the circuit's element
// values `em` [elements, B'] with every child's element id `eid` [N] (and `has` [N], or an empty tensor where every
// child is reached); theta [M, N] (fp32); a sample each (B of them): the observed slots `slot` [B], the row each reads
// `emap` [B] (an empty tensor: its own); the masked samples `rows` [Rm] and their weights `logw` [Rm, M, S] (float32;
// empty where none): (BP, E, ev, cmax, LTh, mitems, oitems, noitems, slot32, logw, flags).
std::vector<torch::Tensor> top_k_prep(torch::Tensor Bm, torch::Tensor G, torch::Tensor em, torch::Tensor eid,
                                      torch::Tensor has, torch::Tensor theta, torch::Tensor slot64, torch::Tensor emap,
                                      torch::Tensor rows, torch::Tensor logwin) {
    const c10::cuda::CUDAGuard guard(Bm.device());
    check_rows(Bm, at::kFloat, "B"); check_rows(G, at::kFloat, "G"); check(theta, at::kFloat, "theta");
    check(eid, at::kLong, "eid"); check(slot64, at::kLong, "slot"); check(rows, at::kLong, "rows");
    TORCH_CHECK(em.is_cuda() && em.scalar_type() == at::kFloat && em.dim() == 2 && em.stride(1) == 1,
                "top_k: the element values, a CUDA float32 [elements, batch] tensor, batch innermost");
    const int Bx = Bm.size(0), r = Bm.size(1), N = Bm.size(2), M = theta.size(0), S = G.size(1), Rm = rows.size(0);
    const int B = slot64.size(0);
    const bool mapped = emap.numel() > 0;
    TORCH_CHECK(G.size(0) == Bx && G.size(2) == N && theta.size(1) == N && eid.size(0) == N && (mapped || Bx == B),
                "top_k: shapes");
    if (mapped) { check(emap, at::kLong, "emap"); TORCH_CHECK(emap.size(0) == B, "top_k: emap [B]"); }
    TORCH_CHECK(M % TM == 0 && N % TN == 0 && r <= 128 && S == 8 * NS8 + 1,
                "top_k: nodes a multiple of 128, children of 64, rank <= 128, top_k = 64");
    TORCH_CHECK(B <= 4096, "top_k: at most 4096 samples (the work items' scans)");
    const bool full = has.numel() == 0;
    if (!full) TORCH_CHECK(has.is_cuda() && has.scalar_type() == at::kBool && has.numel() == N, "top_k: has [N] bool");
    if (Rm > 0) {
        check(logwin, at::kFloat, "logw");
        TORCH_CHECK(logwin.size(0) == Rm && logwin.size(1) == M && logwin.size(2) == S, "top_k: logw [Rm, M, S]");
    }
    const int KP = padded_rank(r);
    auto f32 = Bm.options(), f16 = Bm.options().dtype(at::kHalf), i32 = Bm.options().dtype(at::kInt);
    auto BP = torch::empty({Bx, KP, N}, f16), E = torch::empty({Bx, S, N}, f32);
    auto ev = torch::empty({B, N}, f32), cmax = torch::empty({B}, f32);
    auto LTh = torch::empty({(long)M * N / 2}, i32);
    const long ocap = std::max((B - Rm + 1) / 2, 1);
    auto mitems = torch::empty({std::max(Rm, 1), 4}, i32), oitems = torch::empty({2 * ocap + 1, 4}, i32);
    auto noitems = torch::empty({1}, i32), slot32 = torch::empty({B}, i32);
    auto logw = torch::empty({std::max(Rm, 1), M, SP}, f32);
    // the forward's grid (each kind's launch): a CTA an SM, at least ~4 chunks a CTA (the observed items' count an
    // upper bound here); its flags, the masked launch's and the observed one's
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long chunks = (long)(M / TM) * (N / TN) * std::max((long)Rm, ocap);
    const int P = (int)std::max(2L, std::min((long)sms, chunks / 4));       // (two at least: a CTA for each kind)
    auto flags = torch::empty({2 * P}, i32);
    auto stream = at::cuda::getCurrentCUDAStream();
    tk_prep_kernel<<<dim3(296, 5), 256, 0, stream>>>(
        Bm.data_ptr<float>(), Bm.stride(0), G.data_ptr<float>(), G.stride(0), em.data_ptr<float>(), em.stride(0),
        eid.data_ptr<long>(),
        full ? nullptr : has.data_ptr<bool>(), theta.data_ptr<float>(), Rm > 0 ? logwin.data_ptr<float>() : nullptr,
        reinterpret_cast<__half*>(BP.data_ptr<at::Half>()), E.data_ptr<float>(), ev.data_ptr<float>(),
        cmax.data_ptr<float>(), reinterpret_cast<uint2*>(LTh.data_ptr<int>()), logw.data_ptr<float>(),
        Bx, B, M, N, r, KP, S, Rm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    static bool done[64] = {};
    const int ibytes = 5 * B * 4;
    if (ibytes > 48 * 1024) smem_attr(tk_items_kernel, ibytes, Bm.get_device(), done);
    tk_items_kernel<<<1, 1024, ibytes, stream>>>(
        mapped ? emap.data_ptr<long>() : nullptr, rows.data_ptr<long>(), slot64.data_ptr<long>(),
        reinterpret_cast<int4*>(mitems.data_ptr<int>()), reinterpret_cast<int4*>(oitems.data_ptr<int>()),
        noitems.data_ptr<int>(), slot32.data_ptr<int>(), flags.data_ptr<int>(), B, Rm, 2 * P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (Rm == 0) mitems = mitems.narrow(0, 0, 0);
    return {BP, E, ev, cmax, LTh, mitems, oitems, noitems, slot32, logw, flags};
}

// One node vector's values from the caller's A [Bx, M, r] (fp32) and `top_k_prep`'s operands, into `out` [nodes,
// batch] from node `nid0` on: the masked items' persistent launch and the observed items' beside it on a side stream
// (joined), each taking its kind's share of the SMs (`tk_sk_kernel`).
void top_k_forward(torch::Tensor A, torch::Tensor BP, torch::Tensor E, torch::Tensor ev, torch::Tensor cmax,
                   torch::Tensor LTh, torch::Tensor mitems, torch::Tensor oitems, torch::Tensor noitems,
                   torch::Tensor slot32, torch::Tensor logw, torch::Tensor flags, torch::Tensor out, int64_t nid0) {
    const c10::cuda::CUDAGuard guard(A.device());
    check_rows(A, at::kFloat, "A");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.size(1) == ev.size(0),
                "top_k: the node values, a contiguous CUDA float32 [nodes, batch] tensor");
    const int M = A.size(1), r = A.size(2), KP = BP.size(1), N = BP.size(2), S = E.size(1);
    TORCH_CHECK(KP == padded_rank(r) && BP.size(0) == A.size(0), "top_k: A and the prep's operands disagree");
    TORCH_CHECK(nid0 >= 0 && nid0 + M <= out.size(0), "top_k: the node vector's rows of the node values");
    const int P = flags.size(0) / 2, dev = A.get_device();
    const int Rm = (int)mitems.size(0), B = (int)ev.size(0);
    auto ws = torch::empty({(long)P * (WS_M + WS_O) * NW * 32}, A.options());
    TkArgs args{A.data_ptr<float>(), reinterpret_cast<const __half*>(BP.data_ptr<at::Half>()),
                reinterpret_cast<const uint2*>(LTh.data_ptr<int>()), E.data_ptr<float>(), ev.data_ptr<float>(),
                cmax.data_ptr<float>(), reinterpret_cast<const int4*>(mitems.data_ptr<int>()),
                reinterpret_cast<const int4*>(oitems.data_ptr<int>()), noitems.data_ptr<int>(), slot32.data_ptr<int>(),
                logw.data_ptr<float>(), out.data_ptr<float>(), ws.data_ptr<float>(), flags.data_ptr<int>(),
                (long)nid0, (long)out.size(1), (long)A.stride(0), r, M, N, S, Rm};
    cudaStream_t main = at::cuda::getCurrentCUDAStream(), side = main;
    static cudaEvent_t fork[64] = {}, join[64] = {};
    const bool two = Rm > 0 && B > Rm;
    if (two) {
        if (fork[dev] == nullptr) {
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&fork[dev], cudaEventDisableTiming));
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&join[dev], cudaEventDisableTiming));
        }
        side = at::cuda::getStreamFromPool(false, dev).stream();
        C10_CUDA_CHECK(cudaEventRecord(fork[dev], main));
        C10_CUDA_CHECK(cudaStreamWaitEvent(side, fork[dev], 0));
    }
#define TK_CASE(NKV)                                                                                               \
    case NKV: {                                                                                                    \
        static bool dm[64] = {}, dob[64] = {};                                                                     \
        if (Rm > 0) {                                                                                              \
            smem_attr(tk_sk_kernel<NKV, true>, MaskedGeo<NKV>::BYTES, dev, dm);                                    \
            tk_sk_kernel<NKV, true><<<P, NW * 32, MaskedGeo<NKV>::BYTES, main>>>(args);                            \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                        \
        }                                                                                                          \
        if (B > Rm) {                                                                                              \
            smem_attr(tk_sk_kernel<NKV, false>, ObservedGeo<NKV>::BYTES, dev, dob);                                \
            tk_sk_kernel<NKV, false><<<P, NW * 32, ObservedGeo<NKV>::BYTES, side>>>(args);                         \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                        \
        }                                                                                                          \
        break;                                                                                                     \
    }
    switch (KP / 16) {
        TK_CASE(2)
        TK_CASE(4)
        TK_CASE(6)
        TK_CASE(8)
        default: TORCH_CHECK(false, "top_k: ranks padded to 32, 64, 96 or 128");
    }
#undef TK_CASE
    if (two) {
        C10_CUDA_CHECK(cudaEventRecord(join[dev], side));
        C10_CUDA_CHECK(cudaStreamWaitEvent(main, join[dev], 0));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_prep", &top_k_prep, "a LowRank(top_k) sum node vector's operands and work items for `top_k_forward`");
    m.def("top_k_forward", &top_k_forward,
          "a LowRank(top_k) sum node vector's values: a persistent stream-K launch for each kind of work item");
}
