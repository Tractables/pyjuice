// `LowRank`'s "exp_sum" form at the default precision, its scale from fp16-ROUNDED factors: phi_b = A'_b @ B_b with
// A' = log2 e A and B each rounded to fp16 once, the product exact in fp32 accumulation (one m16n8k16 MMA a 16 x 8 tile)
// -- the exp_sum of the rounded factors, the forward's and the backward's alike -- where `lowrank_exp_sum.cu`'s bf16 path
// splits each factor into hi + lo (three MMAs) to make the unrounded one. The rest is that file's formulation:
//
//     lw_b[i, j] = phi_b[i, j] + log theta[i, j]                    (-inf where theta is 0)
//     Z_b[i] = sum_j exp(lw_b[i, j])        S_b[i] = sum_j exp(lw_b[i, j] + c_b[j])        node value S / Z
//
// everything in base 2 (A' = log2 e A, lc' = log2 e c: exp2f); a row's terms 2^(lw' - cm), cm its largest log2 theta in
// the tile, S's terms times w_j = 2^(lc'_j - mc) (mc the tile's largest lc'), the unshifted sums exact to fp32 where they
// lie in [2^-92, 2^100] and the shifted / exact paths where not (`bf_fw2_kernel`'s note); each tile's (shift, sum)
// pairs, folded over the column tiles by the combine.
//
// FORWARD (`hf_fw_kernel`): a CTA holds a tile of 128 nodes x TN children (TN = 8 NT8) -- eight consumer warps of 16
// nodes, each warp all TN children (NT8 n-tiles of 8), its pairs' log2 theta (less the row's frame) in registers, the
// first MMA's C operand -- and loops over its samples; a producer warp streams each sample's operands into an NST-stage
// ring of shared memory by bulk copies (full / empty mbarriers): the CTA's rows' A' fragments, its children's B
// fragments, their factors w and the tile's mc, laid out by the prologue (`hf_fw_prep_kernel`) in fragment order so a
// lane reads its registers in 16-byte (8-byte) words.
//
// Layouts: the circuit's buffers [rows, batch], batch innermost; the factors as the caller gives them, A [B, M, K],
// B [B, K, N]; the shared parameters block-major (`LowRank._geometry`'s grid), edge block (i / bs, j / cbs) from
// `p0 + ((i / bs) Nb + j / cbs) cbs bs` on, child then node within it. Rank 16; node and child counts padded to whole
// tiles (zeros, -inf past the real ones).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

constexpr int HK = 16;                         // the rank these kernels take
constexpr int HTM = 128;                       // a forward tile's nodes (eight warps of 16)
constexpr int HNW = 8;                         // the forward's consumer warps
#define NEG (-CUDART_INF_F)                    // (device code only)
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

template <typename TP> __device__ __forceinline__ float to_f(TP v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

// The offset of theta[i, j] (and of its flow, its denominator) in the node vector's block-major run
__device__ __forceinline__ long grid_at(int i, int j, int Nb, int cbs, int bs) {
    const int lbs = __ffs(bs) - 1, lcbs = __ffs(cbs) - 1;                   // (block sizes are powers of two)
    return (((long)(i >> lbs) * Nb + (j >> lcbs)) << (lcbs + lbs)) + ((long)(j & (cbs - 1)) << lbs) + (i & (bs - 1));
}

// theta[i, j], 0 outside the node vector: the index clamped and the value selected (no branch around the load)
template <typename TP>
__device__ __forceinline__ float theta_at(const TP* __restrict__ params, long p0, int i, int j, int M, int N, int Nb,
                                          int cbs, int bs) {
    const float v = to_f(params[p0 + grid_at(min(i, M - 1), min(j, N - 1), Nb, cbs, bs)]);
    return (i < M && j < N) ? v : 0.f;
}

// a running (max, sum) of base-2 exponentials folded with another
__device__ __forceinline__ void fold_ms(float& m, float& s, float m2, float s2) {
    const float n = fmaxf(m, m2);
    if (n == NEG) return;
    s = s * exp2f(m - n) + s2 * exp2f(m2 - n);
    m = n;
}

__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

// fp16 m16n8k16 (fp32 accumulation), C given apart from D (the tile's log2 theta) or accumulated into D
__device__ __forceinline__ void mma_h_c(float (&d)[4], const uint4& a, unsigned b0, unsigned b1, const float (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void mbar_init(uint32_t a, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t a, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
    asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
                 ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// node_mars[nid0 + i, b] = (log2 S - log2 Z) ln 2 (-inf where S is 0), lzs[i, b] = log2 Z ln 2, from the column
// tiles' base-2 (max, sum) pairs: four lanes an output, each up to 8 tiles (its loads issued together), folded by
// two shuffles (`lowrank_exp_sum.cu`'s `fw_comb_kernel`)
__global__ void __launch_bounds__(128) hf_comb_kernel(const float4* __restrict__ part, int njt, float* __restrict__ nmars,
                                                      long nid0, float* __restrict__ lzs, int M, int Mp, int B, int Bs,
                                                      int b0) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x, e = t >> 2;
    const int q = (int)(t & 3);
    const bool live = e < (long)B * M;
    const int b = live ? (int)(e / M) : 0, i = live ? (int)(e % M) : 0;
    const float4* p = part + (long)b * njt * Mp + i;
    float mz = NEG, z = 0.f, ms = NEG, s = 0.f;
    for (int t0 = q * 8; t0 < njt; t0 += 32) {
        float4 v[8];
#pragma unroll
        for (int u = 0; u < 8; u++)
            v[u] = live && t0 + u < njt ? p[(long)(t0 + u) * Mp] : make_float4(NEG, 0.f, NEG, 0.f);
#pragma unroll
        for (int u = 0; u < 8; u++) {
            fold_ms(mz, z, v[u].x, v[u].y);
            fold_ms(ms, s, v[u].z, v[u].w);
        }
    }
#pragma unroll
    for (int d = 1; d <= 2; d <<= 1) {
        fold_ms(mz, z, __shfl_xor_sync(0xffffffffu, mz, d), __shfl_xor_sync(0xffffffffu, z, d));
        fold_ms(ms, s, __shfl_xor_sync(0xffffffffu, ms, d), __shfl_xor_sync(0xffffffffu, s, d));
    }
    if (!live || q != 0) return;
    const float lz = z > 0.f ? (mz + __log2f(z)) * LN2 : NEG;
    const float ls = s > 0.f ? (ms + __log2f(s)) * LN2 : NEG;
    nmars[(nid0 + i) * Bs + b0 + b] = ls > NEG ? ls - lz : ls;                   // (no mass: -inf, not 0 / 0)
    lzs[(long)i * Bs + b0 + b] = lz;
}

// ============================================================================================== forward
// Fragment order (a 32-bit register two fp16, the lower index low), per sample:
//   AF [rt][w][lane] uint4: register r -> node 128 rt + 16 w + g + 8 (r & 1), ranks 2 tq + 8 (r >> 1) + {0, 1}
//   BF [jt][t][lane] uint2: register r -> child TN jt + 8 t + g, ranks 2 tq + 8 r + {0, 1}
//   WF [jt][72] float: [tq][2 NT8] the factors w_j = 2^(lc'_j - mc) of a lane's children (TN jt + 8 t + 2 tq + p at
//                       2 t + p), then mc (the tile's largest lc'; -inf: no child of the tile has a value) at 64
//                       (NT8 8; at 4 NT8 2 + 0 for other NT8), padded to 16 bytes
template <int NT8>
struct HfGeo {
    static constexpr int TN = 8 * NT8;
    static constexpr int WFW = 4 * 2 * NT8 + 4;                              // WF floats a tile (mc at 8 NT8)
    static constexpr int A_BYTES = HNW * 32 * 16, B_BYTES = NT8 * 32 * 8, W_BYTES = WFW * 4;
    static constexpr int ST = A_BYTES + B_BYTES + W_BYTES;                   // a stage (each piece 16-byte aligned)
};

// A tile's B fragments by n-tile pairs: n-tile t's uint2 of a lane at index ((t / 2) 32 + lane) 2 + t % 2 of the
// tile's run (a lane's pair one 16-byte word)
__host__ __device__ __forceinline__ int hf_bf_at(int t, int lane) { return ((t >> 1) * 32 + lane) * 2 + (t & 1); }

// The tiles' theta in the forward consumers' order, a block a tile (a thread a consumer lane): LTF [tile][warp][t][lane]
// float4 = the lane's log2 theta less its row's frame (the largest log2 theta of the row in the tile; 0 for a row
// without one), the MMA's C operand, and CMF [tile][warp][lane] = (cm of rows g, g + 8, live flags) -- the main
// kernel's tile change two runs of 16-byte loads, no index math, no logs. Made once for every sample (and every node
// vector of a forward pass reading the same run: `exp_sum_h_theta`)
template <typename TP, int NT8>
__global__ void __launch_bounds__(256) hf_theta_kernel(const TP* __restrict__ params, long p0, int Nb, int cbs, int bs,
                                                       float4* __restrict__ LTF, float4* __restrict__ CMF, int M, int N,
                                                       int Np) {
    using Gm = HfGeo<NT8>;
    const int njt = Np / Gm::TN, tile = blockIdx.x;
    {
        const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
        const int jt = tile % njt, j0 = jt * Gm::TN, rA = (tile / njt) * HTM + 16 * w + g, rB = rA + 8;
        float lt[NT8][4], cm[2] = {NEG, NEG};
        bool live[2] = {false, false};
#pragma unroll
        for (int t = 0; t < NT8; t++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float th = theta_at(params, p0, e & 2 ? rB : rA, j0 + 8 * t + 2 * tq + (e & 1), M, N, Nb, cbs, bs);
                live[e >> 1] |= th > 0.f;
                lt[t][e] = th > 0.f ? __log2f(th) : NEG;
                cm[e >> 1] = fmaxf(cm[e >> 1], lt[t][e]);
            }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int lv = live[r] ? 1 : 0;
            lv |= __shfl_xor_sync(0xffffffffu, lv, 1);
            lv |= __shfl_xor_sync(0xffffffffu, lv, 2);
            live[r] = lv != 0;
            cm[r] = fmaxf(cm[r], __shfl_xor_sync(0xffffffffu, cm[r], 1));
            cm[r] = fmaxf(cm[r], __shfl_xor_sync(0xffffffffu, cm[r], 2));
            if (!live[r]) cm[r] = 0.f;
        }
        float4* lp = LTF + ((long)tile * HNW + w) * NT8 * 32 + lane;
#pragma unroll
        for (int t = 0; t < NT8; t++)
            lp[t * 32] = make_float4(lt[t][0] - cm[0], lt[t][1] - cm[0], lt[t][2] - cm[1], lt[t][3] - cm[1]);
        CMF[((long)tile * HNW + w) * 32 + lane] = make_float4(cm[0], cm[1], live[0] ? 1.f : 0.f, live[1] ? 1.f : 0.f);
    }
}

// The prologue: AF, BF, WF and lcs [B, Np] = lc' (the exact fallback's) for samples b0 .. b0 + B of the chunk, a grid
// row a sample: blocks of 256 threads over AF's registers, then BF's, then WF's tiles (a warp a tile)
template <int NT8>
__global__ void __launch_bounds__(256) hf_fw_prep_kernel(
        const float* __restrict__ A, const float* __restrict__ Bm, uint4* __restrict__ AF, uint2* __restrict__ BF,
        float* __restrict__ WF, float* __restrict__ lcs, const float* __restrict__ emars, const long* __restrict__ rows,
        int M, int N, int Mp, int Np, int nbA, int nbB, int Bs, int b0, long sA, long sB) {
    using Gm = HfGeo<NT8>;
    const int b = blockIdx.y, njt = Np / Gm::TN;
    int blk = blockIdx.x;
    if (blk < nbA) {                                          // AF: a thread a register quadruple (a lane's uint4)
        const int e = blk * 256 + threadIdx.x;               // e = (rt * 8 + w) * 32 + lane
        if (e >= Mp / 16 * 32) return;
        const int lane = e & 31, rg = e >> 5, g = lane >> 2, tq = lane & 3;
        unsigned r4[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const int i = 16 * rg + g + 8 * (r & 1), k = 2 * tq + 8 * (r >> 1);
            const float* src = A + (long)b * sA + (long)i * HK + k;
            const float v0 = i < M ? __ldg(src) * LOG2E : 0.f, v1 = i < M ? __ldg(src + 1) * LOG2E : 0.f;
            r4[r] = pack_h2(v0, v1);
        }
        AF[(long)b * (Mp / 16 * 32) + e] = make_uint4(r4[0], r4[1], r4[2], r4[3]);
        return;
    }
    blk -= nbA;
    if (blk < nbB) {                                          // BF: a thread a lane's uint2
        const int e = blk * 256 + threadIdx.x;               // e = (jt * NT8 + t) * 32 + lane
        if (e >= Np / 8 * 32) return;
        const int lane = e & 31, j = 8 * (e >> 5) + (lane >> 2), tq = lane & 3;
        const int tt = e >> 5, t = tt % NT8;                 // (stored at hf_bf_at's place: n-tiles in pairs)
        const long dst = (long)(tt - t) * 32 + hf_bf_at(t, lane);
        unsigned r2[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int k = 2 * tq + 8 * r;
            const float* src = Bm + (long)b * sB + (long)k * N + j;
            const float v0 = j < N ? __ldg(src) : 0.f, v1 = j < N ? __ldg(src + N) : 0.f;
            r2[r] = pack_h2(v0, v1);
        }
        BF[(long)b * (Np / 8 * 32) + dst] = make_uint2(r2[0], r2[1]);
        return;
    }
    blk -= nbB;                                               // WF: a warp a tile, a lane NT8 / 4 children
    const int jt = blk * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (jt >= njt) return;
    constexpr int PER = Gm::TN / 32;
    float lc[PER], mc = NEG;
#pragma unroll
    for (int u = 0; u < PER; u++) {
        const int j = jt * Gm::TN + u * 32 + lane;
        const long rr = j < N ? rows[j] : -1;
        lc[u] = rr >= 0 ? emars[rr * Bs + b0 + b] * LOG2E : NEG;
        lcs[(long)b * Np + j] = lc[u];
        mc = fmaxf(mc, lc[u]);
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, d));
    float* wp = WF + ((long)b * njt + jt) * Gm::WFW;
#pragma unroll
    for (int u = 0; u < PER; u++) {
        const int jl = u * 32 + lane, t = jl >> 3, tq = (jl & 7) >> 1, p = jl & 1;   // (child jl = 8 t + 2 tq + p)
        wp[tq * 2 * NT8 + 2 * t + p] = mc == NEG ? 0.f : exp2f(lc[u] - mc);
    }
    if (lane < 4) wp[8 * NT8 + lane] = lane == 0 ? mc : 0.f;
}

// A warp's (shift, sum) pairs of its two rows (`hf_fw_kernel`'s registers a[r], z[r], ms[r], s[r])
struct HfRows { float a0, z0, ms0, s0, a1, z1, ms1, s1; };

// The forward's rare paths for one sample of a warp's rows: z out of range (the shifted path: a row max of lw') and S
// below its range (exactly: x' = lw' + lc' under its own max) -- phi made again a tile at a time from the stage's
// fragments, log2 theta read again from the parameters. Out of line (`__noinline__`): the hot loop holds nothing for it
// (in line, its live arrays spilled the loop). Every lane of the warp calls it.
template <typename TP, int NT8>
__device__ __noinline__ HfRows hf_fw_rare(const unsigned char* sp, const TP* __restrict__ params, long p0, int Nb,
                                          int cbs, int bs, const float* lcb, int M, int N, int rA, int rB, int j0,
                                          bool needz, bool needs, HfRows in) {
    using Gm = HfGeo<NT8>;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, tq = lane & 3;
    const uint4 af = *reinterpret_cast<const uint4*>(sp + (w * 32 + lane) * 16);
    const float* wv = reinterpret_cast<const float*>(sp + Gm::A_BYTES + Gm::B_BYTES);
    const float mc = wv[8 * NT8];
    auto lw = [&](int t, float (&x)[4]) {                  // lw' (unframed) of the lane's four pairs of n-tile t
        float c[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float th = theta_at(params, p0, e & 2 ? rB : rA, j0 + 8 * t + 2 * tq + (e & 1), M, N, Nb, cbs, bs);
            c[e] = th > 0.f ? __log2f(th) : NEG;
        }
        const uint2 bf = reinterpret_cast<const uint2*>(sp + Gm::A_BYTES)[hf_bf_at(t, lane)];
        mma_h_c(x, af, bf.x, bf.y, c);
    };
    float a[2] = {in.a0, in.a1}, z[2] = {in.z0, in.z1}, ms[2] = {in.ms0, in.ms1}, s[2] = {in.s0, in.s1};
    if (__any_sync(0xffffffffu, needz)) {                 // (z out of range: the shifted path, a row max)
        float am[2] = {NEG, NEG};
        for (int t = 0; t < NT8; t++) {
            float x[4];
            lw(t, x);
#pragma unroll
            for (int e = 0; e < 4; e++) am[e >> 1] = fmaxf(am[e >> 1], x[e]);
        }
#pragma unroll
        for (int dd = 1; dd <= 2; dd <<= 1) {
            am[0] = fmaxf(am[0], __shfl_xor_sync(0xffffffffu, am[0], dd));
            am[1] = fmaxf(am[1], __shfl_xor_sync(0xffffffffu, am[1], dd));
        }
        const float sa0 = am[0] == NEG ? 0.f : am[0], sa1 = am[1] == NEG ? 0.f : am[1];
        z[0] = z[1] = s[0] = s[1] = 0.f;
        for (int t = 0; t < NT8; t++) {
            const float2 ww = *reinterpret_cast<const float2*>(wv + tq * 2 * NT8 + 2 * t);
            float x[4];
            lw(t, x);
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float v = exp2f(x[e] - (e & 2 ? sa1 : sa0));
                z[e >> 1] += v;
                s[e >> 1] = fmaf(v, e & 1 ? ww.y : ww.x, s[e >> 1]);
            }
        }
#pragma unroll
        for (int dd = 1; dd <= 2; dd <<= 1) {
            z[0] += __shfl_xor_sync(0xffffffffu, z[0], dd); z[1] += __shfl_xor_sync(0xffffffffu, z[1], dd);
            s[0] += __shfl_xor_sync(0xffffffffu, s[0], dd); s[1] += __shfl_xor_sync(0xffffffffu, s[1], dd);
        }
        a[0] = am[0]; a[1] = am[1];
        ms[0] = s[0] > 0.f ? am[0] + mc : NEG;
        ms[1] = s[1] > 0.f ? am[1] + mc : NEG;
        needs = mc > NEG && ((s[0] < 0x1p-97f && am[0] > NEG) || (s[1] < 0x1p-97f && am[1] > NEG));
    }
    if (__any_sync(0xffffffffu, needs)) {                 // (S exactly: x' = lw' + lc' under its own max)
        float m2[2] = {NEG, NEG};
        for (int t = 0; t < NT8; t++) {
            const float2 lc = *reinterpret_cast<const float2*>(lcb + 8 * t);
            float x[4];
            lw(t, x);
#pragma unroll
            for (int e = 0; e < 4; e++) m2[e >> 1] = fmaxf(m2[e >> 1], x[e] + (e & 1 ? lc.y : lc.x));
        }
#pragma unroll
        for (int dd = 1; dd <= 2; dd <<= 1) {
            m2[0] = fmaxf(m2[0], __shfl_xor_sync(0xffffffffu, m2[0], dd));
            m2[1] = fmaxf(m2[1], __shfl_xor_sync(0xffffffffu, m2[1], dd));
        }
        const float sc0 = m2[0] == NEG ? 0.f : m2[0], sc1 = m2[1] == NEG ? 0.f : m2[1];
        float s2[2] = {0.f, 0.f};
        for (int t = 0; t < NT8; t++) {
            const float2 lc = *reinterpret_cast<const float2*>(lcb + 8 * t);
            float x[4];
            lw(t, x);
#pragma unroll
            for (int e = 0; e < 4; e++) s2[e >> 1] += exp2f(x[e] + (e & 1 ? lc.y : lc.x) - (e & 2 ? sc1 : sc0));
        }
#pragma unroll
        for (int dd = 1; dd <= 2; dd <<= 1) {
            s2[0] += __shfl_xor_sync(0xffffffffu, s2[0], dd);
            s2[1] += __shfl_xor_sync(0xffffffffu, s2[1], dd);
        }
        ms[0] = s2[0] > 0.f ? m2[0] : NEG;
        ms[1] = s2[1] > 0.f ? m2[1] : NEG;
        s[0] = s2[0];
        s[1] = s2[1];
    }
    return HfRows{a[0], z[0], ms[0], s[0], a[1], z[1], ms[1], s[1]};
}

// The forward: a persistent grid (the GPU's SMs times the CTAs an SM holds, at most one CTA an item), NW + 1 warps (the
// last the producer). The work is the (tile, sample) items in tile-major order, each CTA an even contiguous run of them
// (its tile's theta set up again where its run enters the next tile). Partials part[b][jt][row] = (cm, z, ms, s) -- Z's
// and S's base-2 (shift, sum) over the tile's children.
template <typename TP, int NT8, int NST>
__global__ void __launch_bounds__((HNW + 1) * 32, NT8 <= 8 ? 2 : 1) hf_fw_kernel(
        const uint4* __restrict__ AF, const uint2* __restrict__ BF, const float* __restrict__ WF,
        const float* __restrict__ lcs, const TP* __restrict__ params, long p0, int Nb, int cbs, int bs,
        const float4* __restrict__ LTF, const float4* __restrict__ CMF, float4* __restrict__ part, int M, int N,
        int Mp, int Np, int B, int njt, long total) {
    using Gm = HfGeo<NT8>;
    constexpr int TN = Gm::TN;
    extern __shared__ __align__(128) unsigned char sm[];
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
    const uint32_t bfull = sbase + NST * Gm::ST, bempty = bfull + NST * 8;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const long it0 = (long)blockIdx.x * total / gridDim.x, it1 = (long)(blockIdx.x + 1) * total / gridDim.x;
    if (it0 >= it1) return;
    if (tid == 0) {
        for (int s = 0; s < NST; s++) { mbar_init(bfull + 8 * s, 1); mbar_init(bempty + 8 * s, HNW); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (w == HNW) {                                           // the producer: an item's operands a stage
        if (lane == 0) {                                      // (the item's tile and sample by counters: no division)
            const long astr = (long)Mp / 16 * 32, bstr = (long)Np / 8 * 32, wstr = (long)njt * Gm::WFW;
            const int tile0 = (int)(it0 / B);
            int b = (int)(it0 - (long)tile0 * B), jt = tile0 % njt, rt = tile0 / njt, st = 0, ph = 0;
            for (long it = it0; it < it1; it++) {
                mbar_wait(bempty + 8 * st, ph ^ 1);
                const uint32_t dst = sbase + st * Gm::ST, bar = bfull + 8 * st;
                mbar_expect(bar, Gm::ST);
                bulk_g2s(dst, AF + b * astr + (long)rt * HNW * 32, Gm::A_BYTES, bar);
                bulk_g2s(dst + Gm::A_BYTES, BF + b * bstr + (long)jt * NT8 * 32, Gm::B_BYTES, bar);
                bulk_g2s(dst + Gm::A_BYTES + Gm::B_BYTES, WF + b * wstr + (long)jt * Gm::WFW, Gm::W_BYTES, bar);
                if (++b == B) { b = 0; if (++jt == njt) { jt = 0; rt++; } }
                if (++st == NST) { st = 0; ph ^= 1; }
            }
        }
        return;
    }

    float lt[NT8][4], cm[2] = {0.f, 0.f};
    bool live[2] = {false, false};
    int cur = -1, jt = 0, rA = 0, rB = 0, j0 = 0;
    int tile = (int)(it0 / B), b = (int)(it0 - (long)tile * B), st = 0, ph = 0;   // (counters: no division an item)
    for (long it = it0; it < it1; it++) {
        if (tile != cur) {                                    // the tile's log2 theta less each row's frame cm (its
            cur = tile;                                       //  largest in the tile), from the prologue's fragments
            jt = tile % njt;
            j0 = jt * TN;
            rA = (tile / njt) * HTM + 16 * w + g;
            rB = rA + 8;
            const float4* lp = LTF + ((long)tile * HNW + w) * NT8 * 32 + lane;
#pragma unroll
            for (int t = 0; t < NT8; t++) {
                const float4 v = __ldg(lp + t * 32);
                lt[t][0] = v.x; lt[t][1] = v.y; lt[t][2] = v.z; lt[t][3] = v.w;
            }
            const float4 cv = __ldg(CMF + ((long)tile * HNW + w) * 32 + lane);
            cm[0] = cv.x; cm[1] = cv.y;
            live[0] = cv.z != 0.f; live[1] = cv.w != 0.f;
        }
        mbar_wait(bfull + 8 * st, ph);
        const unsigned char* sp = sm + st * Gm::ST;
        const uint4 af = *reinterpret_cast<const uint4*>(sp + (w * 32 + lane) * 16);
        const float* wv = reinterpret_cast<const float*>(sp + Gm::A_BYTES + Gm::B_BYTES);
        const float mc = wv[8 * NT8];
        float z[2] = {0.f, 0.f}, s[2] = {0.f, 0.f};
#pragma unroll
        for (int t2 = 0; t2 < NT8; t2 += 2) {                // (n-tiles in pairs: a 16-byte load each of B, w)
            const uint4 bf2 = *reinterpret_cast<const uint4*>(sp + Gm::A_BYTES + ((t2 >> 1) * 32 + lane) * 16);
            const float4 ww2 = *reinterpret_cast<const float4*>(wv + tq * 2 * NT8 + 2 * t2);
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int t = t2 + u;
                const unsigned b0 = u ? bf2.z : bf2.x, b1 = u ? bf2.w : bf2.y;
                const float wx = u ? ww2.z : ww2.x, wy = u ? ww2.w : ww2.y;
                float d[4];
                mma_h_c(d, af, b0, b1, lt[t]);
                const float e0 = exp2f(d[0]), e1 = exp2f(d[1]), e2 = exp2f(d[2]), e3 = exp2f(d[3]);
                z[0] += e0; z[0] += e1; z[1] += e2; z[1] += e3;
                s[0] = fmaf(e0, wx, s[0]); s[0] = fmaf(e1, wy, s[0]);
                s[1] = fmaf(e2, wx, s[1]); s[1] = fmaf(e3, wy, s[1]);
            }
        }
#pragma unroll
        for (int dd = 1; dd <= 2; dd <<= 1) {
            z[0] += __shfl_xor_sync(0xffffffffu, z[0], dd); z[1] += __shfl_xor_sync(0xffffffffu, z[1], dd);
            s[0] += __shfl_xor_sync(0xffffffffu, s[0], dd); s[1] += __shfl_xor_sync(0xffffffffu, s[1], dd);
        }
        float a[2] = {z[0] > 0.f ? cm[0] : NEG, z[1] > 0.f ? cm[1] : NEG};
        float ms[2] = {s[0] > 0.f ? cm[0] + mc : NEG, s[1] > 0.f ? cm[1] + mc : NEG};
        bool needz = false, needs = false;                    // (2^-92: a row's largest term over 2^-97, so any term
#pragma unroll                                                //  flushed under 2^-126 is under 2^-29 of it)
        for (int r = 0; r < 2; r++) {
            needz |= live[r] && !(z[r] >= 0x1p-92f && z[r] <= 0x1p100f);
            needs |= live[r] && mc > NEG && !(s[r] >= 0x1p-92f);
        }
        if (__any_sync(0xffffffffu, needz || needs)) {       // (rare: out of line)
            const HfRows r = hf_fw_rare<TP, NT8>(sp, params, p0, Nb, cbs, bs, lcs + (long)b * Np + j0 + 2 * tq, M, N, rA,
                                                 rB, j0, needz, needs,
                                                 HfRows{a[0], z[0], ms[0], s[0], a[1], z[1], ms[1], s[1]});
            a[0] = r.a0; z[0] = r.z0; ms[0] = r.ms0; s[0] = r.s0;
            a[1] = r.a1; z[1] = r.z1; ms[1] = r.ms1; s[1] = r.s1;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(bempty + 8 * st);
        if (tq == 0) {
            if (rA < M) part[((long)b * njt + jt) * Mp + rA] = make_float4(a[0], z[0], ms[0], s[0]);
            if (rB < M) part[((long)b * njt + jt) * Mp + rB] = make_float4(a[1], z[1], ms[1], s[1]);
        }
        if (++b == B) { b = 0; tile++; }
        if (++st == NST) { st = 0; ph ^= 1; }
    }
}


// ================================================================================================= backward
// `lowrank_exp_sum.cu`'s big-tile backward (`bf_bw2_kernel`, its head has the design) at rank 16 with phi from ONE fp16
// MMA (the forward's rounded factors) and G, B, A' fp16 in the dA / dB products: a CTA of 8 warps a tile of 128 nodes x
// 64 children over a run of (tile, sample) units, F+ and W in registers over a tile's samples. PIPELINED, one barrier
// a sample: iteration b (parity par) makes phi_b, b's element terms -- each n-tile with one k-step of dB of sample
// b - 1 (from the other parity's G^T / A', every warp's written a barrier back), each second n-tile with b's dA
// k-step and G^T m-tile --, and writes b's G^T / A' / element-flow partials into parity par; the next [A] publishes
// them. A phase saturating one unit (shared memory, the tensor pipe, MUFU) with the others idle, one after the other,
// is what bound the two-barrier form: interleaved, the units work at once (-16% B64, -21% B256). The gradients' sums
// leave as fp32 bulk adds of a CTA's partials after the next [A]: dA's rows (contiguous) one plain bulk op, dB's 16
// rank rows of 64 children one TENSOR reduce through a 3-D map of the caller's [B, 16, N] (16 bulk ops of 256 bytes
// cost 12 us at B64, 48 at B256: the L2's small-op rate).
// Fragment order of the operands (a 32-bit register two fp16, the lower index low), per sample:
//   AF [rg][lane][4]: register r -> node 16 rg + g + 8 (r & 1), ranks 2 tq + 8 (r >> 1) + {0, 1} (the forward's AF)
//   BF [jt][lane][8] (32-child tiles): register u -> child 32 jt + 8 (u >> 1) + g, ranks 2 tq + 8 (u & 1) + {0, 1}
constexpr int TM2 = 128, TN2 = 64, NW2 = 8, BN = 32;
struct Hb2Geo {
    static constexpr int NT = 2;
    static constexpr int SL = 2 * TM2 + 2 * 36 + 4;            // a stage's terms: LU (128 float2), LC (2 x 36), SC
    static constexpr int SB = 2 * 256 + SL;                    // words: a sample's stage -- B (2 jt) + its terms
    static constexpr int SA = NW2 * NT * 64;                   // A' fragments transposed: [warp][nt][2][32]
    static constexpr int ST = NW2 * 16 * 32;                   // G^T blocks: [warp][2 nf + h][32]
    static constexpr int SE = NW2 * TN2;                       // element flows: [warp][64]
    static constexpr int SDA = TM2 * HK;                       // dA's partials (fp32): [node 128][rank 16]
    static constexpr int SDB = HK * TN2;                       // dB's: [rank 16][child 64]
    static constexpr int MAIN = 2 * SDA + 2 * SDB + 2 * SB + 2 * SA + 2 * ST + 2 * SE;   // (each by parity; the
                                                               //  partials first: the TMA source 128-byte aligned)
    static constexpr int FLUSH = NW2 * 32 * 20;                // F+ / W: [warp][child 32][node 16 + 4], a half at a time
    static constexpr int WORDS = MAIN;
    static_assert(FLUSH <= 2 * SA + 2 * ST, "the F+ / W flush within the A' / G^T buffers");
    static_assert(SB % 4 == 0 && SDA % 4 == 0, "16-byte aligned pieces");
};

// Atomic adds (vector forms: sm_90 on)
__device__ __forceinline__ void red1(float* p, float a) {
    asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(a) : "memory");
}
__device__ __forceinline__ void red4(float* p, float4 v) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}
// fp32 bulk adds from shared memory into global (the L2's adds), a bulk group each; the issuing thread waits for its
// groups' reads before their buffer is written again
__device__ __forceinline__ void bulk_add_f(void* gdst, const void* ssrc, int bytes) {
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                 ::"l"(gdst), "r"((unsigned)__cvta_generic_to_shared(ssrc)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
// a box of a 3-D tensor (its map a kernel parameter) added from shared memory (128-byte aligned, the box's layout)
__device__ __forceinline__ void tma_add_3d(const CUtensorMap* m, const void* ssrc, int c0, int c1, int c2) {
    asm volatile("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
                 ::"l"(m), "r"((unsigned)__cvta_generic_to_shared(ssrc)), "r"(c0), "r"(c1), "r"(c2) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
// (a thread's shared-memory writes made visible to the async proxy: before the barrier ahead of a bulk add)
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ unsigned movt(unsigned x) {
    unsigned y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
// fp16 m16n8k16 accumulating into D (the four A registers given apart)
__device__ __forceinline__ void mma_h16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The backward's prologue, one launch, a grid row a sample: AF, BF (fp16, the orders above), the nodes' terms LU
// [B, Mp, 2] = (lu', lv') -- lu' = log2 (f / (n Z)), lv' = log2 (f / Z), -inf for a node without flow --, the children's
// LC [B, njt, 36] (per tq the lane's 8 children's lc'), the element-flow scratch ef [B, Np] zeroed and the sample's
// scale SC [B] (its last block): E = ceil(log2 max_i f_i) -> (E, 2^E, ln 2 2^E, 0). The main kernel takes q and t (so
// G, the fp16 operand of the dA / dB products) in units of 2^E -- |G| <= 1: fp16's range whatever the flows' scale,
// a sample's common scale carried apart in fp32 -- and its sums back by 2^E
__global__ void __launch_bounds__(256) hb_bw_prep_kernel(
        const float* __restrict__ A, const float* __restrict__ Bm, uint4* __restrict__ AF, unsigned* __restrict__ BF,
        float* __restrict__ LU, float* __restrict__ LC, float* __restrict__ ef, float4* __restrict__ SC,
        const float* __restrict__ nflows, const float* __restrict__ nmars, long nid0,
        const float* __restrict__ lzs, const float* __restrict__ emars, const long* __restrict__ rows, int M, int N,
        int Mp, int Np, int nbA, int nbB, int nT, int Bs, int b0, long sA, long sB) {
    const int b = blockIdx.y, njt = Np / BN;
    int blk = blockIdx.x;
    if (blk < nbA) {                                          // AF: a thread a lane's uint4
        const int e = blk * 256 + threadIdx.x;
        if (e >= Mp / 16 * 32) return;
        const int lane = e & 31, rg = e >> 5, g = lane >> 2, tq = lane & 3;
        unsigned r4[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const int i = 16 * rg + g + 8 * (r & 1), k = 2 * tq + 8 * (r >> 1);
            const float* src = A + (long)b * sA + (long)i * HK + k;
            const float v0 = i < M ? __ldg(src) * LOG2E : 0.f, v1 = i < M ? __ldg(src + 1) * LOG2E : 0.f;
            r4[r] = pack_h2(v0, v1);
        }
        AF[(long)b * (Mp / 16 * 32) + e] = make_uint4(r4[0], r4[1], r4[2], r4[3]);
        return;
    }
    blk -= nbA;
    if (blk < nbB) {                                          // BF: a thread a register
        const int e = blk * 256 + threadIdx.x;               // e = (jt * 32 + lane) * 8 + u
        if (e >= Np / BN * 256) return;
        const int u = e & 7, lane = (e >> 3) & 31, jt = e >> 8, g = lane >> 2, tq = lane & 3;
        const int j = BN * jt + 8 * (u >> 1) + g, k = 2 * tq + 8 * (u & 1);
        const float* src = Bm + (long)b * sB + (long)k * N + j;
        const float v0 = j < N ? __ldg(src) : 0.f, v1 = j < N ? __ldg(src + N) : 0.f;
        BF[(long)b * (Np / BN * 256) + e] = pack_h2(v0, v1);
        return;
    }
    blk -= nbB;
    if (blk >= nT) {                                          // the scale: the largest log flow
        __shared__ float red[8];
        const int lane = threadIdx.x & 31, wq = threadIdx.x >> 5;
        float m = NEG;
        for (int o = threadIdx.x; o < M; o += 256) {
            const float lf = nflows[(nid0 + o) * Bs + b0 + b];
            if (isfinite(lf)) m = fmaxf(m, lf);
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
        if (lane == 0) red[wq] = m;
        __syncthreads();
        if (threadIdx.x == 0) {
#pragma unroll
            for (int q = 1; q < 8; q++) m = fmaxf(m, red[q]);
            m = fmaxf(m, red[0]);
            const float E = m > NEG ? fminf(fmaxf(ceilf(m * LOG2E), -120.f), 120.f) : 0.f;   // (no flow: any)
            SC[b] = make_float4(E, exp2f(E), LN2 * exp2f(E), 0.f);
        }
        return;
    }
    const int o = blk * 256 + threadIdx.x;
    if (o < Mp) {
        float lu = NEG, lv = NEG;
        if (o < M) {
            const float lf = nflows[(nid0 + o) * Bs + b0 + b];
            if (isfinite(lf)) {
                const float lz = lzs[(long)o * Bs + b0 + b];
                lu = (lf - (nmars[(nid0 + o) * Bs + b0 + b] + lz)) * LOG2E;
                lv = (lf - lz) * LOG2E;
            }
        }
        reinterpret_cast<float2*>(LU)[(long)b * Mp + o] = make_float2(lu, lv);
    } else if (o < Mp + njt * 36) {
        const int q = o - Mp, jt = q / 36, w = q % 36;
        float v = 0.f;
        if (w < 32) {                                          // (slot 8 tq + 2 nf + p: child 8 nf + 2 tq + p)
            const int tq = w >> 3, nf = (w & 7) >> 1, p = w & 1, j = BN * jt + 8 * nf + 2 * tq + p;
            const long r = j < N ? rows[j] : -1;
            v = r >= 0 ? emars[r * Bs + b0 + b] * LOG2E : NEG;
        }
        LC[((long)b * njt + jt) * 36 + w] = v;
    } else if (o < Mp + njt * 36 + Np) {
        ef[(long)b * Np + (o - Mp - njt * 36)] = 0.f;
    }
}

// The backward's epilogue: the element flows to the log, log-added into the circuit's (a thread a (child, sample))
__global__ void __launch_bounds__(256) hb_ef_epi_kernel(const float* __restrict__ ef, const long* __restrict__ rows,
                                                        float* __restrict__ eflows, int N, int Np, int B, int lb, int Bs,
                                                        int b0) {
    const long t = (long)blockIdx.x * 256 + threadIdx.x;
    const int b = (int)(t & ((1 << lb) - 1));
    const long j = t >> lb;
    if (b >= B || j >= N) return;
    const long er = rows[j];
    const float v = ef[(long)b * Np + j];
    if (er < 0 || !(v > 0.f)) return;
    const float lv = logf(v);
    float* p = eflows + er * Bs + b0 + b;
    const float o = *p;
    *p = o == NEG ? lv : fmaxf(o, lv) + log1pf(__expf(-fabsf(o - lv)));
}

template <typename TP, bool FLOWS, bool DENOM, bool GRADS>
__global__ void __launch_bounds__(256, 1) hb_bw_kernel(
        const unsigned* __restrict__ AFh, const unsigned* __restrict__ BFh, const TP* __restrict__ params, long p0,
        int Nb, int cbs, int bs, const float* __restrict__ LU, const float* __restrict__ LC, float* __restrict__ ef,
        float* __restrict__ pflows, long pf0, float* __restrict__ denom, long dn0, float* __restrict__ gA, long sgA,
        int ma, const __grid_constant__ CUtensorMap tmB, const float4* __restrict__ SC, int M, int N, int Mp, int Np,
        int B, int nct, long units) {
    constexpr int KP = HK;
    using Gm = Hb2Geo;
    constexpr int NT = Gm::NT;
    extern __shared__ __align__(128) unsigned smw[];
    float* const sDA = reinterpret_cast<float*>(smw);          // [par][node 128][rank 16]
    float* const sDB = sDA + 2 * Gm::SDA;                      // [par][rank 16][child 64]
    unsigned* const sB = reinterpret_cast<unsigned*>(sDB + 2 * Gm::SDB);   // [par][SB]: B fragments, LU, LC, SC
    unsigned* const sA2 = sB + 2 * Gm::SB;                     // [par][SA]
    unsigned* const sT2 = sA2 + 2 * Gm::SA;                    // [par][ST]
    float* const sE2 = reinterpret_cast<float*>(sT2 + 2 * Gm::ST);   // [par][SE]
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const bool issuer = GRADS && tid == 0;                     // (the bulk adds' thread)
    const long astr = (long)Mp * KP / 2, bstr = (long)Np * KP / 2;
    const int njt = Np / BN;
    const long u0 = (long)blockIdx.x * units / gridDim.x, u1 = (long)(blockIdx.x + 1) * units / gridDim.x;
    // sample b's B fragments (child tile jcs's two 32-child blocks, the lanes' two 16-byte halves apart: a warp's
    // reads of one half conflict-free) and its terms (row tile its's LU rows, the 2 jt's LC, SC) into stage par
    auto stage = [&](int b, int par, int jcs, int its) {
        if (tid < 128) {
            const int r = tid, blk = r >> 6, ln = (r >> 1) & 31, half = r & 1;
            const unsigned* src = BFh + (long)b * bstr + ((long)2 * jcs + blk) * 256 + ln * 8 + half * 4;
            cp_async16(sB + par * Gm::SB + blk * 256 + half * 128 + ln * 4, src);
        } else if (tid < 128 + 83) {
            const int r = tid - 128;
            const float* src = r < 64 ? LU + ((long)b * Mp + (long)its * TM2) * 2 + r * 4
                             : r < 82 ? LC + ((long)b * njt + 2 * jcs) * 36 + (r - 64) * 4
                                      : reinterpret_cast<const float*>(SC + b);
            cp_async16(sB + par * Gm::SB + 512 + r * 4, src);
        }
        cp_async_commit();
    };
    uint4 nah;                                                 // (the warp's A fragments, a sample ahead)
    auto fetchA = [&](int b, int rgs) {
        nah = __ldg(reinterpret_cast<const uint4*>(AFh + (long)b * astr + ((long)rgs * 32 + lane) * 4));
    };
    // dB of a sample from parity pp's G^T / A' (all 8 warps' blocks written, a barrier back): unit w = (child m-tile
    // w & 3, rank tile w >> 2), two split accumulators
    auto dB_mma = [&](int pp, float (&dq)[2][4]) {
        const unsigned* const sA = sA2 + pp * Gm::SA;
        const unsigned* const sT = sT2 + pp * Gm::ST;
        const int mg = w & 3, nt = w >> 2;
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
            for (int c = 0; c < 4; c++) dq[h][c] = 0.f;
#pragma unroll
        for (int kw = 0; kw < NW2; kw++) {
            const uint2 bb = reinterpret_cast<const uint2*>(sA)[(kw * NT + nt) * 32 + lane];
            const uint4 aa = reinterpret_cast<const uint4*>(sT)[(kw * 4 + mg) * 32 + lane];
            mma_h16(dq[kw & 1], aa.x, aa.y, aa.z, aa.w, bb.x, bb.y);
        }
    };
    // its unit's sums (A' carried log2 e, G units of 2^E: times scz = ln 2 2^E) into parity pp's [rank][child] rows:
    // (children 16 mg + g / + 8, ranks 8 nt + 2 tq / + 1)
    auto dB_out = [&](int pp, const float (&dq)[2][4], float scz) {
        const int mg = w & 3, nt = w >> 2, ch = 16 * mg + g, rk = 8 * nt + 2 * tq;
        float* const db = sDB + pp * Gm::SDB;
        db[rk * TN2 + ch] = (dq[0][0] + dq[1][0]) * scz;
        db[(rk + 1) * TN2 + ch] = (dq[0][1] + dq[1][1]) * scz;
        db[rk * TN2 + ch + 8] = (dq[0][2] + dq[1][2]) * scz;
        db[(rk + 1) * TN2 + ch + 8] = (dq[0][3] + dq[1][3]) * scz;
        fence_async_smem();
    };
    // the bulk adds of a sample's dA rows (the tile's 128, clipped at ma) and dB's box (16 ranks x 64 children at
    // child j0, sample b; children past the map's dropped), from parity pp (written and fenced a barrier back)
    auto out_dA = [&](int pp, int b, int i0) {
        const int rows = min(TM2, ma - i0);
        if (issuer && rows > 0) bulk_add_f(gA + (long)b * sgA + (long)i0 * HK, sDA + pp * Gm::SDA, rows * HK * 4);
    };
    auto out_dB = [&](int pp, int b, int j0) {
        if (issuer) tma_add_3d(&tmB, sDB + pp * Gm::SDB, j0, 0, b);
    };
    // the element flows of a sample from parity pp's partials (its 8 warps' sums), one add
    auto flows_out = [&](int pp, int b, int j0, float scy) {
        if (tid < TN2) {
            const float* const sE = sE2 + pp * Gm::SE;
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < NW2; ww++) s += sE[ww * TN2 + tid];
            if (s != 0.f && j0 + tid < N) red1(ef + (long)b * Np + j0 + tid, s * scy);
        }
    };
    int pc = 0;                                                // the stage parity of the sample at hand
    if (u0 < u1) {                                             // the run's first sample: its stage, its A fragments
        const int t = (int)(u0 / B), b = (int)(u0 - (long)t * B), it = t / nct;
        stage(b, 0, t - it * nct, it);
        fetchA(b, it * NW2 + w);
    }
#pragma unroll 1
    for (long u = u0; u < u1;) {
        const int t = (int)(u / B), b0 = (int)(u - (long)t * B);
        const int b1 = (int)min((long)B, (long)b0 + (u1 - u));
        u += b1 - b0;
        const bool more = u < u1;                              // (the run's next tile: t + 1, from its sample 0)
        const int it = t / nct, jc = t - it * nct;
        const int i0 = it * TM2, j0 = jc * TN2;
        const int rg = it * NW2 + w, rA = i0 + 16 * w + g, rB = rA + 8;

        // the tile's theta (its log2 taken under the first sample's phi), F+, W: element e = 4 nf + c -> row
        // c & 2 ? rB : rA, child j0 + 8 nf + 2 tq + (c & 1)
        float lt[32], fp[32], tw[32];
#pragma unroll
        for (int e = 0; e < 32; e++) {
            const int nf = e >> 2, c = e & 3, i = c & 2 ? rB : rA, j = j0 + 8 * nf + 2 * tq + (c & 1);
            lt[e] = theta_at(params, p0, i, j, M, N, Nb, cbs, bs);
            fp[e] = 0.f;
            tw[e] = 0.f;
        }
        float scy_prev = 0.f, scz_prev = 0.f;                  // (the previous sample's scales)

#pragma unroll 1
        for (int b = b0; b < b1; b++, pc ^= 1) {
            const int par = pc;
            cp_async_wait<0>();
            if (issuer) bulk_wait_read0();                     // (the adds issued at the last [A] have read their rows)
            __syncthreads();                                   // [A] b's stage; b - 1's G^T / A' / partials written
            if (GRADS) {
                if (b > b0) out_dA(par ^ 1, b - 1, i0);
                if (b > b0 + 1) out_dB(par, b - 2, j0);
            }
            if (b > b0) flows_out(par ^ 1, b - 1, j0, scy_prev);
            if (b + 1 < b1) stage(b + 1, par ^ 1, jc, it);   // (its last readers: b - 1's, before [A])
            else if (more) {                                   // (the next tile's first sample)
                const int itn = (t + 1) / nct;
                stage(0, par ^ 1, t + 1 - itn * nct, itn);
            }
            const unsigned* const sbh = sB + par * Gm::SB;
            const float* const sl = reinterpret_cast<const float*>(sbh + 512);
            // the sample's scale 2^E: q, t, G in its units (lu', lv' less E), F+ / W / the sums times 2^E
            const float4 scl = *reinterpret_cast<const float4*>(sl + 2 * TM2 + 72);
            float2 tA = reinterpret_cast<const float2*>(sl)[16 * w + g], tB = reinterpret_cast<const float2*>(sl)[16 * w + g + 8];
            tA.x -= scl.x; tA.y -= scl.x; tB.x -= scl.x; tB.y -= scl.x;
            float lcv[16];                                     // lc' of children j0 + 8 nf + 2 tq + p at 2 nf + p
#pragma unroll
            for (int jl = 0; jl < 2; jl++) {
                const float* lc0 = sl + 2 * TM2 + jl * 36 + tq * 8;
                const float4 c0 = *reinterpret_cast<const float4*>(lc0);
                const float4 c1 = *reinterpret_cast<const float4*>(lc0 + 4);
                lcv[8 * jl + 0] = c0.x; lcv[8 * jl + 1] = c0.y; lcv[8 * jl + 2] = c0.z; lcv[8 * jl + 3] = c0.w;
                lcv[8 * jl + 4] = c1.x; lcv[8 * jl + 5] = c1.y; lcv[8 * jl + 6] = c1.z; lcv[8 * jl + 7] = c1.w;
            }
            // ---- phi_b (one fp16 term): the warp's A fragments, the staged B fragments
            float S[8][4];
#pragma unroll
            for (int nf = 0; nf < 8; nf++)
#pragma unroll
                for (int c = 0; c < 4; c++) S[nf][c] = 0.f;
            {
                const uint4 xh = nah;
                if (b + 1 < b1) fetchA(b + 1, rg);
                else if (more) fetchA(0, (t + 1) / nct * NW2 + w);
                const unsigned ah[4] = {xh.x, xh.y, xh.z, xh.w};
                if (GRADS) {                                   // dB's A' operand of sample b, parity par
#pragma unroll
                    for (int r = 0; r < 2; r++)
                        reinterpret_cast<uint2*>(sA2 + par * Gm::SA)[(w * NT + r) * 32 + lane] =
                            make_uint2(movt(ah[2 * r]), movt(ah[2 * r + 1]));
                }
#pragma unroll
                for (int jl = 0; jl < 2; jl++) {
                    const uint4* bp = reinterpret_cast<const uint4*>(sbh + jl * 256) + lane;
                    const uint4 h0 = bp[0], h1 = bp[32];
                    const unsigned bh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
                    for (int q = 0; q < 4; q++)
                        mma_h16(S[4 * jl + q], ah[0], ah[1], ah[2], ah[3], bh[2 * q], bh[2 * q + 1]);
                }
            }
            float dq[2][4];                                    // (dB of sample b - 1, its k-steps in the loop below)
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
                for (int c = 0; c < 4; c++) dq[h][c] = 0.f;
            if (b == b0) {                                     // (theta's loads under the first phi)
#pragma unroll
                for (int e = 0; e < 32; e++) lt[e] = lt[e] > 0.f ? __log2f(lt[e]) : NEG;
            }
            // ---- the element terms of b: q = 2^(lw' + lu' + lc'), t = 2^(lw' + lv'); each n-tile with dB_{b-1}'s
            // k-step nf, each pair of n-tiles followed by b's dA k-step and G^T m-tile
            unsigned R[8][2];                                  // (q - t) as fp16 pairs: (n-tile, node half)
            float v[16];                                       // q over the lane's two rows, its children 2 nf + p
            float dd[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) dd[nt][c] = 0.f;
#pragma unroll
            for (int nf = 0; nf < 8; nf++) {
                if (GRADS && b > b0) {
                    const unsigned* const sAp = sA2 + (par ^ 1) * Gm::SA;
                    const unsigned* const sTp = sT2 + (par ^ 1) * Gm::ST;
                    const int mg = w & 3, nt = w >> 2;
                    const uint2 bb = reinterpret_cast<const uint2*>(sAp)[(nf * NT + nt) * 32 + lane];
                    const uint4 aa = reinterpret_cast<const uint4*>(sTp)[(nf * 4 + mg) * 32 + lane];
                    mma_h16(dq[nf & 1], aa.x, aa.y, aa.z, aa.w, bb.x, bb.y);
                }
                float gq[4], qq[4];
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const float lw = S[nf][c] + lt[4 * nf + c];
                    const float q = exp2f(lw + (c & 2 ? tB.x : tA.x) + lcv[2 * nf + (c & 1)]);
                    const float tt = exp2f(lw + (c & 2 ? tB.y : tA.y));
                    if (FLOWS) fp[4 * nf + c] = fmaf(q, scl.y, fp[4 * nf + c]);
                    if (DENOM) tw[4 * nf + c] = fmaf(tt, scl.y, tw[4 * nf + c]);
                    qq[c] = q;
                    gq[c] = q - tt;
                }
                v[2 * nf] = qq[0] + qq[2];
                v[2 * nf + 1] = qq[1] + qq[3];
                R[nf][0] = pack_h2(gq[0], gq[1]);
                R[nf][1] = pack_h2(gq[2], gq[3]);
                if (GRADS && (nf & 1)) {
                    // dA += G B^T over children 16 kk ..: the A operand R[2 kk][0], R[2 kk][1], R[2 kk + 1][0],
                    // R[2 kk + 1][1]; B^T the staged fragments of block kk / 2, half kk & 1, transposed
                    const int kk = nf >> 1;
                    const uint4 hw = reinterpret_cast<const uint4*>(sbh + (kk >> 1) * 256)[(kk & 1) * 32 + lane];
                    const unsigned m0 = movt(hw.x), m1 = movt(hw.y), m2 = movt(hw.z), m3 = movt(hw.w);
                    mma_h16(dd[0], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], m0, m2);
                    mma_h16(dd[1], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], m1, m3);
                    // dB's G^T block (child g, nodes 2 tq ..) of m-tile kk, in the A operand's order
                    reinterpret_cast<uint4*>(sT2 + par * Gm::ST)[(w * 4 + kk) * 32 + lane] =
                        make_uint4(movt(R[2 * kk][0]), movt(R[2 * kk + 1][0]), movt(R[2 * kk][1]), movt(R[2 * kk + 1][1]));
                }
            }
            // ---- the element flows: each child's sum over the warp's 16 nodes, a reduce-scatter over the 8 lanes of a tq
            {
                float* const sE = sE2 + par * Gm::SE;
                const bool h2 = g & 4, h1 = g & 2, h0 = g & 1;
                float u8[8], u4[4], u2[2];
#pragma unroll
                for (int m = 0; m < 8; m++) {
                    const float send = h2 ? v[m] : v[m + 8], keep = h2 ? v[m + 8] : v[m];
                    u8[m] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
                }
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    const float send = h1 ? u8[m] : u8[m + 4], keep = h1 ? u8[m + 4] : u8[m];
                    u4[m] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
                }
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    const float send = h0 ? u4[m] : u4[m + 2], keep = h0 ? u4[m + 2] : u4[m];
                    u2[m] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
                }
                const int base = (h2 ? 8 : 0) + (h1 ? 4 : 0) + (h0 ? 2 : 0);   // v's indices this lane holds: base, + 1
#pragma unroll
                for (int m = 0; m < 2; m++) {
                    const int idx = base + m;
                    sE[w * TN2 + 8 * (idx >> 1) + 2 * tq + (idx & 1)] = u2[m];
                }
            }
            if (GRADS) {
                // b's dA rows (units of 2^E: times scl.y) into parity par: (nodes g / g + 8, ranks 8 nt + 2 tq, + 1)
                float* const da = sDA + par * Gm::SDA;
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    *reinterpret_cast<float2*>(da + (16 * w + g) * HK + 8 * nt + 2 * tq) =
                        make_float2(dd[nt][0] * scl.y, dd[nt][1] * scl.y);
                    *reinterpret_cast<float2*>(da + (16 * w + g + 8) * HK + 8 * nt + 2 * tq) =
                        make_float2(dd[nt][2] * scl.y, dd[nt][3] * scl.y);
                }
                // ---- dB_{b-1}'s epilogue (its MMAs long done)
                if (b > b0) dB_out(par ^ 1, dq, scz_prev);
                else fence_async_smem();
            }
            scy_prev = scl.y; scz_prev = scl.z;
        }
        // the tile's drain: the last sample's G^T / A' / partials published, its dB and element flows, then F+ / W
        const int pl = pc ^ 1;                                 // (the last sample's parity)
        if (issuer) bulk_wait_read0();
        __syncthreads();
        if (GRADS) {
            out_dA(pl, b1 - 1, i0);
            if (b1 - 1 > b0) out_dB(pl ^ 1, b1 - 2, j0);
        }
        flows_out(pl, b1 - 1, j0, scy_prev);
        if (GRADS) {
            float dq[2][4];
            dB_mma(pl, dq);
            dB_out(pl, dq, scz_prev);                          // (pl's dB rows: their last add waited before the drain)
        }
        __syncthreads();                                       // (the drain's reads of sA2 / sT2 and dB's writes done)
        if (GRADS) out_dB(pl, b1 - 1, j0);
        // F+ and W of the warp's 16 x 64 cells, half the children at a time through its own shared rows (the A' /
        // G^T buffers: the next tile's first stage lands meanwhile): a lane four consecutive nodes of a child, one
        // 16-byte add (scalar adds where the node blocks are under 4 or the runs unaligned)
        if (FLOWS || DENOM) {
            float* const st = reinterpret_cast<float*>(sA2) + w * (32 * 20);   // [child 32][node 16 + 4]
            const bool vec = (bs & 3) == 0 && ((pf0 | dn0) & 3) == 0;
#pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                if ((pass == 0 && !FLOWS) || (pass == 1 && !DENOM)) continue;
                float* const dst = pass == 0 ? pflows + pf0 : denom + dn0;
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
#pragma unroll
                    for (int e4 = 0; e4 < 16; e4++) {
                        const int e = 16 * hh + e4, nf = e >> 2, c = e & 3;
                        float val = pass == 0 ? fp[e] : tw[e];
                        if (pass == 1) {
                            const float th = exp2f(lt[e]);
                            val = th > 0.f ? val / th : 0.f;
                        }
                        st[(8 * (nf - 4 * hh) + 2 * tq + (c & 1)) * 20 + g + 8 * (c >> 1)] = val;
                    }
                    __syncwarp();
#pragma unroll
                    for (int uu = 0; uu < 4; uu++) {           // (8 children an instruction, 4 lanes a child)
                        const int col = 8 * uu + (lane >> 2), r4 = 4 * (lane & 3), i = i0 + 16 * w + r4;
                        const int j = j0 + 32 * hh + col;
                        const float4 vv = *reinterpret_cast<const float4*>(st + col * 20 + r4);
                        if (j >= N || i >= M) continue;
                        const long o = grid_at(i, j, Nb, cbs, bs);
                        if (vec && i + 3 < M) {
                            if (vv.x != 0.f || vv.y != 0.f || vv.z != 0.f || vv.w != 0.f) red4(dst + o, vv);
                        } else {
                            const float v4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
                            for (int q = 0; q < 4; q++)
                                if (i + q < M && v4[q] != 0.f) red1(dst + grid_at(i + q, j, Nb, cbs, bs), v4[q]);
                        }
                    }
                    __syncwarp();
                }
            }
        }
        __syncthreads();                                       // (the flush's region, the stages: the next tile's)
    }
    if (issuer) bulk_wait_all();                               // (the rows' last reads before the CTA ends)
}


// Samples a pass takes at once: its per-sample working set `per` (bytes) kept within the L2, the batch split evenly
static int64_t sample_chunk(int64_t B, double per) {
    constexpr int NDEV = 64;
    static int l2s[NDEV] = {0};
    const int dev = at::cuda::current_device();
    int l2 = dev < NDEV ? l2s[dev] : 0;
    if (l2 == 0) {
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
        if (dev < NDEV) l2s[dev] = l2;
    }
    const int64_t cmax = std::max<int64_t>(16, (int64_t)(l2 / per));
    const int64_t nch = (B + cmax - 1) / cmax;
    return (B + nch - 1) / nch;
}

// A [B, X, Y] whose samples each lie contiguous (any batch stride: the caller's tensors read in place)
static bool inner_contiguous(const torch::Tensor& t) {
    return t.dim() == 3 && (t.size(2) <= 1 || t.stride(2) == 1) && (t.size(1) <= 1 || t.stride(1) == t.size(2));
}

void check_common(const torch::Tensor& A, const torch::Tensor& Bm, const torch::Tensor& params,
                  const torch::Tensor& emars, const torch::Tensor& rows, int64_t M, int64_t N, int64_t K, int64_t B) {
    TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kFloat32 && inner_contiguous(A), "A: fp32 [B, M, K], a sample's contiguous");
    TORCH_CHECK(Bm.is_cuda() && Bm.scalar_type() == torch::kFloat32 && inner_contiguous(Bm), "B: fp32 [B, K, N], a sample's contiguous");
    TORCH_CHECK(A.dim() == 3 && A.size(0) == B && A.size(1) == M && A.size(2) == K, "A's shape");
    TORCH_CHECK(Bm.dim() == 3 && Bm.size(0) == B && Bm.size(1) == K && Bm.size(2) == N, "B's shape");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32 || params.scalar_type() == torch::kBFloat16, "params: fp32 or bf16");
    TORCH_CHECK(emars.scalar_type() == torch::kFloat32 && emars.is_contiguous() && emars.size(1) == B, "element_mars");
    TORCH_CHECK(rows.scalar_type() == torch::kInt64 && rows.numel() == N, "rows: int64 [N]");
    TORCH_CHECK(K == HK, "the fp16 exp_sum kernels take rank 16");
}

#define HF_TP_SWITCH(PARAMS, ...)                                                                                  \
    if (PARAMS.scalar_type() == torch::kFloat32) {                                                                 \
        using TP = float;                                                                                          \
        const TP* pp = PARAMS.template data_ptr<float>();                                                                 \
        __VA_ARGS__;                                                                                               \
    } else {                                                                                                       \
        using TP = __nv_bfloat16;                                                                                  \
        const TP* pp = reinterpret_cast<const TP*>(PARAMS.template data_ptr<at::BFloat16>());                            \
        __VA_ARGS__;                                                                                               \
    }

template <int NT8, int NST, typename TP>
void launch_hf_fw(cudaStream_t stream, const uint4* af, const uint2* bf, const float* wf, const float* lc, const TP* pp,
                  long p0, int Nb, int cbs, int bs, const float4* ltf, const float4* cmf, float4* part, int M, int N,
                  int Mp, int Np, int Bc, int njt, long total) {
    using Gm = HfGeo<NT8>;
    constexpr int bytes = NST * Gm::ST + 2 * NST * 8;
    auto kern = hf_fw_kernel<TP, NT8, NST>;
    static int per_sm[64] = {0};                              // (a device's CTAs an SM; 0: not asked yet)
    const int dev = at::cuda::current_device();
    int occ = dev < 64 ? per_sm[dev] : 0;
    if (occ == 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, (HNW + 1) * 32, bytes));
        occ = std::max(occ, 1);
        if (dev < 64) per_sm[dev] = occ;
    }
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long grid = std::max<long>(1, std::min<long>(total, (long)sms * occ));
    kern<<<(unsigned)grid, (HNW + 1) * 32, bytes, stream>>>(af, bf, wf, lc, pp, p0, Nb, cbs, bs, ltf, cmf, part, M, N,
                                                           Mp, Np, Bc, njt, total);
}

// The forward of one node vector: its node values into `node_mars` rows `nid0 .. nid0 + M`; returns log Z [M, B].
// `tn`: the tiles' children, 64 or 128.
// The tiles' theta fragments of a node vector (`hf_theta_kernel`): {LTF, CMF}
template <int NT8>
std::vector<torch::Tensor> theta_impl(torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs, int64_t bs, int64_t M,
                                      int64_t N) {
    using Gm = HfGeo<NT8>;
    const at::cuda::CUDAGuard guard(params.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int Mp = (int)((M + HTM - 1) / HTM * HTM), Np = (int)((N + Gm::TN - 1) / Gm::TN * Gm::TN);
    const int nbT = (Np / Gm::TN) * (Mp / HTM);
    auto LTF = torch::empty({(long)nbT * HNW * NT8 * 32 * 4}, params.options().dtype(torch::kFloat32));
    auto CMF = torch::empty({(long)nbT * HNW * 32 * 4}, params.options().dtype(torch::kFloat32));
    HF_TP_SWITCH(params, hf_theta_kernel<TP, NT8><<<(unsigned)nbT, 256, 0, stream>>>(
        pp, p0, (int)Nb, (int)cbs, (int)bs, reinterpret_cast<float4*>(LTF.template data_ptr<float>()),
        reinterpret_cast<float4*>(CMF.template data_ptr<float>()), (int)M, (int)N, Np));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {LTF, CMF};
}

template <int NT8>
torch::Tensor forward_impl(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb,
                           int64_t cbs, int64_t bs, torch::Tensor emars, torch::Tensor rows, torch::Tensor node_mars,
                           int64_t nid0, int64_t M, int64_t N, torch::Tensor LTF, torch::Tensor CMF) {
    using Gm = HfGeo<NT8>;
    constexpr int NST = 3, TN = Gm::TN;
    const int64_t B = node_mars.size(1);
    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int Mp = (int)((M + HTM - 1) / HTM * HTM), Np = (int)((N + TN - 1) / TN * TN), njt = Np / TN, nit = Mp / HTM;
    auto lzs = torch::empty({M, B}, A.options());
    const long* rw = reinterpret_cast<const long*>(rows.template data_ptr<int64_t>());
    // a sample's working set: AF, BF (fp16), WF, lcs, the partials
    const int64_t BC = sample_chunk(B, 2.0 * HK * (Mp + Np) + 4.0 * njt * Gm::WFW + 4.0 * Np + 16.0 * njt * Mp);
    auto AF = torch::empty({BC, (long)Mp / 16 * 32 * 4}, A.options().dtype(torch::kInt32));
    auto BF = torch::empty({BC, (long)Np / 8 * 32 * 2}, A.options().dtype(torch::kInt32));
    auto WF = torch::empty({BC, (long)njt * Gm::WFW}, A.options());
    auto lcs = torch::empty({BC, (long)Np}, A.options());
    auto part = torch::empty({BC, (long)njt * Mp * 4}, A.options());
    const int nbT = njt * nit;                                 // (the tiles' theta fragments: given, or made here)
    if (LTF.numel() == 0) {
        auto tc = theta_impl<NT8>(params, p0, Nb, cbs, bs, M, N);
        LTF = tc[0];
        CMF = tc[1];
    }
    TORCH_CHECK(LTF.numel() == (long)nbT * HNW * NT8 * 32 * 4 && CMF.numel() == (long)nbT * HNW * 32 * 4,
                "theta fragments: exp_sum_h_theta's, for this node vector and tile width");
    const float4* ltf = reinterpret_cast<const float4*>(LTF.template data_ptr<float>());
    const float4* cmf = reinterpret_cast<const float4*>(CMF.template data_ptr<float>());
    uint4* af = reinterpret_cast<uint4*>(AF.template data_ptr<int>());
    uint2* bf = reinterpret_cast<uint2*>(BF.template data_ptr<int>());
    float4* pp4 = reinterpret_cast<float4*>(part.template data_ptr<float>());
    const int nbA = (Mp / 16 * 32 + 255) / 256, nbB = (Np / 8 * 32 + 255) / 256, nbW = (njt + 7) / 8;
    for (int64_t b0 = 0; b0 < B; b0 += BC) {
        const int Bc = (int)std::min<int64_t>(BC, B - b0);
        hf_fw_prep_kernel<NT8><<<dim3(nbA + nbB + nbW, (unsigned)Bc), 256, 0, stream>>>(
            A.template data_ptr<float>() + b0 * A.stride(0), Bm.template data_ptr<float>() + b0 * Bm.stride(0), af, bf,
            WF.template data_ptr<float>(), lcs.template data_ptr<float>(), emars.template data_ptr<float>(), rw,
            (int)M, (int)N, Mp, Np, nbA, nbB, (int)B, (int)b0, (long)A.stride(0), (long)Bm.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const long total = (long)njt * nit * Bc;
        HF_TP_SWITCH(params, launch_hf_fw<NT8, NST, TP>(stream, af, bf, WF.template data_ptr<float>(),
                                                        lcs.template data_ptr<float>(), pp, p0, (int)Nb, (int)cbs,
                                                        (int)bs, ltf, cmf, pp4, (int)M, (int)N, Mp, Np, Bc, njt, total));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hf_comb_kernel<<<(unsigned)((4L * Bc * M + 127) / 128), 128, 0, stream>>>(pp4, njt, node_mars.template data_ptr<float>(),
                                                                                nid0, lzs.template data_ptr<float>(), (int)M, Mp,
                                                                                Bc, (int)B, (int)b0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return lzs;
}

// The tiles' theta fragments of a node vector's parameters for the forward at tile width `tn` (made once a forward
// pass for every node vector reading the same run: tied copies): {LTF, CMF}
std::vector<torch::Tensor> exp_sum_h_theta(torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs, int64_t bs,
                                           int64_t M, int64_t N, int64_t tn) {
    TORCH_CHECK(params.scalar_type() == torch::kFloat32 || params.scalar_type() == torch::kBFloat16, "params: fp32 or bf16");
    TORCH_CHECK(tn == 64 || tn == 128, "tn: 64 or 128");
    return tn == 128 ? theta_impl<16>(params, p0, Nb, cbs, bs, M, N) : theta_impl<8>(params, p0, Nb, cbs, bs, M, N);
}

// The forward of one node vector (`theta`: exp_sum_h_theta's fragments of its parameters, or empty: made here)
torch::Tensor exp_sum_h_forward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb,
                                int64_t cbs, int64_t bs, torch::Tensor emars, torch::Tensor rows, torch::Tensor node_mars,
                                int64_t nid0, int64_t M, int64_t N, int64_t K, int64_t tn,
                                std::vector<torch::Tensor> theta) {
    const int64_t B = node_mars.size(1);
    check_common(A, Bm, params, emars, rows, M, N, K, B);
    TORCH_CHECK(node_mars.scalar_type() == torch::kFloat32 && node_mars.is_contiguous(), "node_mars");
    TORCH_CHECK(tn == 64 || tn == 128, "tn: 64 or 128");
    torch::Tensor ltf = theta.size() == 2 ? theta[0] : torch::empty({0}, A.options());
    torch::Tensor cmf = theta.size() == 2 ? theta[1] : torch::empty({0}, A.options());
    if (tn == 128) return forward_impl<16>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_mars, nid0, M, N, ltf, cmf);
    return forward_impl<8>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_mars, nid0, M, N, ltf, cmf);
}


#define HB_FLAGS_SWITCH(FLOWS, DN, GRADS, ...)                                                                     \
    if (FLOWS && DN && GRADS) { constexpr bool FL = true, DE = true, GR = true; __VA_ARGS__; }                     \
    else if (FLOWS && DN) { constexpr bool FL = true, DE = true, GR = false; __VA_ARGS__; }                        \
    else if (FLOWS && GRADS) { constexpr bool FL = true, DE = false, GR = true; __VA_ARGS__; }                     \
    else if (FLOWS) { constexpr bool FL = true, DE = false, GR = false; __VA_ARGS__; }                             \
    else if (DN && GRADS) { constexpr bool FL = false, DE = true, GR = true; __VA_ARGS__; }                        \
    else if (DN) { constexpr bool FL = false, DE = true, GR = false; __VA_ARGS__; }                                \
    else if (GRADS) { constexpr bool FL = false, DE = false, GR = true; __VA_ARGS__; }                             \
    else { constexpr bool FL = false, DE = false, GR = false; __VA_ARGS__; }

// cuTensorMapEncodeTiled (the driver's, found through the runtime once: no link to libcuda)
static PFN_cuTensorMapEncodeTiled_v12000 tmap_encode() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
    if (fn == nullptr) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
        TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr, "cuTensorMapEncodeTiled: not found");
        fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
    }
    return fn;
}

// dB's destination as a 3-D tensor of fp32 -- n children (a rank's row, contiguous), 16 ranks `ldb` apart, bc samples
// `sgb` apart (in floats; 16-byte multiples) -- in boxes of the backward's 64 children x 16 ranks x 1 sample
static CUtensorMap db_map(float* base, int64_t n, int64_t ldb, int64_t sgb, int64_t bc) {
    CUtensorMap m;
    cuuint64_t dim[3] = {(cuuint64_t)n, (cuuint64_t)HK, (cuuint64_t)bc};
    cuuint64_t str[2] = {(cuuint64_t)ldb * 4, (cuuint64_t)sgb * 4};
    cuuint32_t box[3] = {(cuuint32_t)TN2, (cuuint32_t)HK, 1};
    cuuint32_t es[3] = {1, 1, 1};
    const CUresult r = tmap_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, base, dim, str, box, es,
                                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                     CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "dB's tensor map: cuTensorMapEncodeTiled failed (", (int)r, ")");
    return m;
}

// The backward on its persistent grid: the GPU's SMs times the CTAs an SM holds (asked once a device), at most a CTA a
// unit
template <typename TP, bool FL, bool DE, bool GR>
void launch_hb_bw(cudaStream_t stream, const unsigned* af, const unsigned* bf, const TP* pp, long p0, int Nb, int cbs,
                  int bs, const float* LU, const float* LC, float* ef, float* pf, long pf0, float* dnp, long dn0,
                  float* ga, long sga, int ma, const CUtensorMap& tmb, const float4* sc, int M, int N, int Mp, int Np,
                  int B, int nct, long units) {
    constexpr int bytes = Hb2Geo::WORDS * 4;
    auto kern = hb_bw_kernel<TP, FL, DE, GR>;
    static int per_sm[64] = {0};
    const int dev = at::cuda::current_device();
    int occ = dev < 64 ? per_sm[dev] : 0;
    if (occ == 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, 256, bytes));
        occ = std::max(occ, 1);
        if (dev < 64) per_sm[dev] = occ;
    }
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long grid = std::max<long>(1, std::min<long>(units, (long)sms * occ));
    kern<<<(unsigned)grid, 256, bytes, stream>>>(af, bf, pp, p0, Nb, cbs, bs, LU, LC, ef, pf, pf0, dnp, dn0, ga, sga, ma,
                                                tmb, sc, M, N, Mp, Np, B, nct, units);
}

// The backward of one node vector: the element flows of its children log-added into `element_flows` (the caller sets
// them to -inf first, once a layer), and -- each where its tensor is given (not empty) -- F+ added into `param_flows`
// from `pf0`, W into `denom` from `dn0`, dLL/dA and dLL/dB into the gradient buffers (the caller's layout, [B, M, K],
// [B, K, N]): `lowrank_exp_sum.cu`'s `exp_sum_backward` at rank 16 with phi of the fp16-rounded factors
void exp_sum_h_backward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs,
                        int64_t bs, torch::Tensor emars, torch::Tensor rows, torch::Tensor node_flows,
                        torch::Tensor node_mars, int64_t nid0, torch::Tensor lzs, torch::Tensor element_flows,
                        torch::Tensor pflows, int64_t pf0, torch::Tensor denom, int64_t dn0, torch::Tensor gA,
                        torch::Tensor gB, int64_t M, int64_t N, int64_t K) {
    const int64_t B = node_mars.size(1);
    check_common(A, Bm, params, emars, rows, M, N, K, B);
    TORCH_CHECK(node_mars.scalar_type() == torch::kFloat32 && node_mars.is_contiguous(), "node_mars");
    TORCH_CHECK(node_flows.scalar_type() == torch::kFloat32 && node_flows.is_contiguous() && node_flows.size(1) == B,
                "node_flows");
    TORCH_CHECK(element_flows.scalar_type() == torch::kFloat32 && element_flows.is_contiguous() &&
                element_flows.size(1) == B, "element_flows");
    TORCH_CHECK(lzs.scalar_type() == torch::kFloat32 && lzs.is_contiguous() && lzs.size(0) == M && lzs.size(1) == B,
                "log Z: [M, B]");
    const bool flows = pflows.numel() > 0, dn = denom.numel() > 0, grads = gA.numel() > 0;
    if (flows) TORCH_CHECK(pflows.scalar_type() == torch::kFloat32 && pflows.is_contiguous(), "param_flows");
    if (dn) TORCH_CHECK(denom.scalar_type() == torch::kFloat32 && denom.is_contiguous(), "denominators");
    if (grads) {
        TORCH_CHECK(gB.numel() > 0, "both gradients or neither");
        TORCH_CHECK(gA.scalar_type() == torch::kFloat32 && inner_contiguous(gA) && gA.sizes() == A.sizes(),
                    "dA: fp32 [B, M, K], a sample's contiguous");
        TORCH_CHECK(gB.scalar_type() == torch::kFloat32 && inner_contiguous(gB) && gB.sizes() == Bm.sizes(),
                    "dB: fp32 [B, K, N], a sample's contiguous");
    }
    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const long* rw = reinterpret_cast<const long*>(rows.data_ptr<int64_t>());
    float* pf = flows ? pflows.data_ptr<float>() : nullptr;
    float* dnp = dn ? denom.data_ptr<float>() : nullptr;
    const int Mp2 = (int)((M + TM2 - 1) / TM2 * TM2), Np2 = (int)((N + TN2 - 1) / TN2 * TN2), njt2 = Np2 / BN;
    const int nct = Np2 / TN2, nit2 = Mp2 / TM2;
    const int64_t BC = sample_chunk(B, 2.0 * HK * (Mp2 + Np2) + 8.0 * K * (M + N));   // (a chunk of samples at a time)
    const long na2 = (long)Mp2 / 16 * 32 * 4, nb2 = (long)Np2 / BN * 256;       // (words a sample)
    auto AF2 = torch::empty({BC, na2}, A.options().dtype(torch::kInt32));
    auto BF2 = torch::empty({BC, nb2}, A.options().dtype(torch::kInt32));
    auto LU2 = torch::empty({BC, Mp2, 2}, A.options());
    auto LC2 = torch::empty({BC, njt2, 36}, A.options());
    auto efs2 = torch::empty({BC, Np2}, A.options());
    auto SC2 = torch::empty({BC, 4}, A.options());             // (the samples' scales)
    unsigned* afh2 = reinterpret_cast<unsigned*>(AF2.data_ptr<int>());
    unsigned* bfh2 = reinterpret_cast<unsigned*>(BF2.data_ptr<int>());
    float4* sc2 = reinterpret_cast<float4*>(SC2.data_ptr<float>());
    const int nbA2 = (int)((Mp2 / 16 * 32 + 255) / 256), nbB2 = (int)((nb2 + 255) / 256);
    const int nT2 = (Mp2 + njt2 * 36 + Np2 + 255) / 256;
    // the gradients' bulk adds take the caller's buffers where each sample's rows lie 16-byte aligned (dA's [M, 16]
    // rows, dB's 16 rank rows of N); else padded temporaries [B, Mp2, 16], [B, 16, Np2], added in afterwards
    const auto al16 = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
    const bool direct = !grads || (al16(gA) && gA.stride(0) % 4 == 0 && al16(gB) && gB.stride(0) % 4 == 0 && N % 4 == 0);
    torch::Tensor tA2, tB2;
    if (grads && !direct) {
        tA2 = torch::zeros({BC, Mp2, HK}, A.options());
        tB2 = torch::zeros({BC, HK, Np2}, A.options());
    }
    const long sga = !grads ? 0L : direct ? (long)gA.stride(0) : (long)Mp2 * HK;
    const long sgb = !grads ? 0L : direct ? (long)gB.stride(0) : (long)HK * Np2;
    const int ma = direct ? (int)M : Mp2, ldb = direct ? (int)N : Np2, nbw = direct ? (int)N : Np2;
    for (int64_t b0 = 0; b0 < B; b0 += BC) {
        const int Bc = (int)std::min<int64_t>(BC, B - b0);
        int lbc = 0;
        while ((1 << lbc) < Bc) lbc++;
        hb_bw_prep_kernel<<<dim3(nbA2 + nbB2 + nT2 + 1, (unsigned)Bc), 256, 0, stream>>>(
            A.data_ptr<float>() + b0 * A.stride(0), Bm.data_ptr<float>() + b0 * Bm.stride(0), reinterpret_cast<uint4*>(afh2),
            bfh2, LU2.data_ptr<float>(), LC2.data_ptr<float>(), efs2.data_ptr<float>(), sc2, node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), nid0, lzs.data_ptr<float>(), emars.data_ptr<float>(), rw, (int)M, (int)N, Mp2,
            Np2, nbA2, nbB2, nT2, (int)B, (int)b0, (long)A.stride(0), (long)Bm.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (grads && !direct) {
            tA2.zero_();
            tB2.zero_();
        }
        float* ga2 = !grads ? nullptr : direct ? gA.data_ptr<float>() + b0 * gA.stride(0) : tA2.data_ptr<float>();
        float* gb2 = !grads ? nullptr : direct ? gB.data_ptr<float>() + b0 * gB.stride(0) : tB2.data_ptr<float>();
        CUtensorMap tmb;
        if (grads) tmb = db_map(gb2, nbw, ldb, sgb, Bc);
        else std::memset(&tmb, 0, sizeof(tmb));
        const long units = (long)nit2 * nct * Bc;
        if (params.scalar_type() == torch::kFloat32) {
            using TP = float;
            const TP* pp = params.data_ptr<float>();
            HB_FLAGS_SWITCH(flows, dn, grads, launch_hb_bw<TP, FL, DE, GR>(stream, afh2, bfh2, pp, p0, (int)Nb, (int)cbs,
                (int)bs, LU2.data_ptr<float>(), LC2.data_ptr<float>(), efs2.data_ptr<float>(), pf, pf0, dnp, dn0, ga2,
                sga, ma, tmb, sc2, (int)M, (int)N, Mp2, Np2, Bc, nct, units))
        } else {
            using TP = __nv_bfloat16;
            const TP* pp = reinterpret_cast<const TP*>(params.data_ptr<at::BFloat16>());
            HB_FLAGS_SWITCH(flows, dn, grads, launch_hb_bw<TP, FL, DE, GR>(stream, afh2, bfh2, pp, p0, (int)Nb, (int)cbs,
                (int)bs, LU2.data_ptr<float>(), LC2.data_ptr<float>(), efs2.data_ptr<float>(), pf, pf0, dnp, dn0, ga2,
                sga, ma, tmb, sc2, (int)M, (int)N, Mp2, Np2, Bc, nct, units))
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hb_ef_epi_kernel<<<(unsigned)((((long)N << lbc) + 255) / 256), 256, 0, stream>>>(
            efs2.data_ptr<float>(), rw, element_flows.data_ptr<float>(), (int)N, Np2, Bc, lbc, (int)B, (int)b0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (grads && !direct) {                                // (the temporaries' sums into the caller's buffers)
            gA.narrow(0, b0, Bc).add_(tA2.narrow(0, 0, Bc).narrow(1, 0, M));
            gB.narrow(0, b0, Bc).add_(tB2.narrow(0, 0, Bc).narrow(2, 0, N));
        }
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_sum_h_theta", &exp_sum_h_theta,
          "the forward's tiles' theta fragments of a node vector's parameters (make once, pass to exp_sum_h_forward)");
    m.def("exp_sum_h_forward", &exp_sum_h_forward,
          "a node vector's values (and log Z) under the exp_sum scale of fp16-rounded factors, the scale made tile by tile");
    m.def("exp_sum_h_backward", &exp_sum_h_backward,
          "a node vector's element flows, F+, W and dLL/dA, dLL/dB under the exp_sum scale of fp16-rounded factors");
}
