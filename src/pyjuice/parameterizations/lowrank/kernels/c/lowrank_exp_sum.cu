// `LowRank`'s "exp_sum" form, fused: the per-sample scale `g_b[i, j] = exp(phi_b[i, j])`, `phi_b = A_b @ B_b`, made
// tile by tile and used where it is made, never stored.
//
//     lw_b[i, j] = phi_b[i, j] + log theta[i, j]                    (-inf where theta is 0)
//     Z_b[i] = sum_j exp(lw_b[i, j])        S_b[i] = sum_j exp(lw_b[i, j] + c_b[j])        node value S / Z
//
//     q_b[i, j] = exp(lw + log f - log n - log Z + c_b[j])           the flow along the edge (f the node flow)
//     t_b[i, j] = exp(lw + log f - log Z)                            f theta_b
//     element flow[j, b] = sum_i q        F+[i, j] += sum_b q        W[i, j] += sum_b t / theta
//     dLL/dA_b = G B_b^T        dLL/dB_b = A_b^T G        G = q - t
//
// Everything in base 2 (A' = log2 e A, lc' = log2 e c: exp2f). A TILE is 64 nodes by 32 children (a CTA, four warps
// of 16 nodes, each warp the tile's 32 children); a CTA loops over its samples, its tile's log2 theta in registers.
//
// THE PRODUCTS, two precisions (the circuit's `precision`):
//   bf16 (the default, "auto"): phi from bf16 hi + lo (A' = Ah + Al, B = Bh + Bl; Ah Bh + Ah Bl + Al Bh, fp32
//     accumulation: ~2^-16 of each product), m16n8k16 MMAs (m16n8k8 at ranks <= 8); dA, dB one bf16 term each.
//   tf32 ("fp32"): three-term TF32 (hi + lo, ~fp32) everywhere, m16n8k8 MMAs (m16n8k4 at ranks <= 4 in the forward).
//
// FORWARD (`bf_fw_kernel` / `tf_fw_kernel`, then `fw_comb_kernel`): a warp's MMA fragments read straight from global
// memory one sample ahead, in FRAGMENT ORDER (the prologue lays each sample's A' and B out lane by lane: a 16-byte
// load a lane a fragment), no shared memory, no barriers. Z's terms 2^(lw' - a), a the node's max over the tile; S's
// the same terms times the children's factors: e w_j = 2^(lw' + lc' - a - mc), mc the tile's max lc' -- one
// exponential an element. A term below 2^-126 of the reference 2^(a + mc) flushes; at most BN of them, so a sum of at
// least 2^-97 is exact to fp32 (the flushed mass under 2^-24 of it), and a tile whose sum falls below that (its large
// terms' lw' and lc' apart by > 97 bits) is made again exactly, x' = lw' + lc' under its own max (a warp's vote). Each
// warp writes its tiles' (max, sum) pairs; the combine folds the column tiles' into the node values and log Z.
//
// BACKWARD, bf16, ranks 9 .. 64 (`bf_bw2_kernel`, below: tiles of 128 nodes x 64 children, persistent CTAs of 8 warps);
// ranks to 8 (`bf_bw_kernel`): phi again from its own prologue's fragments (the forward keeps none), then in
// registers: dA = G B^T -- G's accumulators ARE the m16n8k16 A operand (bf16 pairs), B^T phi's hi B fragments
// transposed (`movmatrix`); dB^T = G^T A' over the CTA's 64 nodes -- each warp stages its G^T blocks (`movmatrix`)
// and its A' fragments (phi's hi A fragments transposed) a sample parity, a barrier, then units (child m-tile, rank
// tile) over the four warps' nodes as k-steps; the element flows a reduce-scatter over the warp's nodes, summed over
// the four warps through the same stage. F+ and W in registers over the samples, added once at the end through each
// warp's own shared rows (a lane 4 consecutive nodes of a child: 16-byte adds).
// BACKWARD, tf32 (`tf_bw_kernel`): a sample's A', B, node and child terms staged in shared memory (cp.async, double
// buffered), G^T staged for dB over the CTA's nodes, dA in registers; gradients into per-sample scratch, then the
// epilogue adds them into the caller's layout.
//
// Layouts: the circuit's buffers [rows, batch], batch innermost; the factors and their gradients AS THE CALLER GIVES
// THEM, A [B, M, K], B [B, K, N] (`LowRank.storage_shapes`: a sample's block contiguous, so the fragment order is a
// permutation within 16-node (32-child) blocks). The shared parameters where they lie, block-major
// (`LowRank._geometry`'s grid): edge block (i / bs, j / cbs) from `p0 + ((i / bs) Nb + j / cbs) cbs bs` on, child
// then node within it -- the flows' and the denominators' ranges alike. Ranks up to 64; node and child counts padded
// to whole tiles (zeros, -inf past the real ones).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <math_constants.h>

namespace {

constexpr int BN = 32;                         // a tile's children (a warp's: four n-tiles of 8)
constexpr int TBM = 64;                        // a tile's nodes (four warps of 16)
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

// Atomic adds; the vector forms sm_90 on (elsewhere one a float)
__device__ __forceinline__ void red1(float* p, float a) {
    asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(a) : "memory");
}
__device__ __forceinline__ void red2(float* p, float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(p), "f"(a), "f"(b) : "memory");
#else
    red1(p, a);
    red1(p + 1, b);
#endif
}
__device__ __forceinline__ void red4(float* p, float4 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
#else
    red1(p, v.x);
    red1(p + 1, v.y);
    red1(p + 2, v.z);
    red1(p + 3, v.w);
#endif
}
// a pair of adjacent ranks (the second only where it is a real one): one 8-byte add where aligned
__device__ __forceinline__ void add_pair(float* p, float a, float b, bool both) {
    if (both && (reinterpret_cast<unsigned long>(p) & 7) == 0) red2(p, a, b);
    else {
        red1(p, a);
        if (both) red1(p + 1, b);
    }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// a running (max, sum) of base-2 exponentials folded with another
__device__ __forceinline__ void fold_ms(float& m, float& s, float m2, float s2) {
    const float n = fmaxf(m, m2);
    if (n == NEG) return;
    s = s * exp2f(m - n) + s2 * exp2f(m2 - n);
    m = n;
}

// TF32 MMAs (fp32 accumulation), m16n8k8 and m16n8k4; the C fragment (g = lane / 4, tq = lane % 4): c0 (g, 2 tq),
// c1 (g, 2 tq + 1), c2 (g + 8, 2 tq), c3 (g + 8, 2 tq + 1)
__device__ __forceinline__ void mma_tf32(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_tf32_k4(float (&d)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
// three-term TF32 (x = hi + lo; hi hi + hi lo + lo hi, ~fp32): hi rounded to nearest (bits + 2^12, masked: finite
// values), lo = x - hi passed whole (the tensor core reads its TF32 bits)
__device__ __forceinline__ void tf32_split_fast(float x, unsigned& hi, unsigned& lo) {
    hi = (__float_as_uint(x) + 0x1000u) & 0xffffe000u;
    lo = __float_as_uint(x - __uint_as_float(hi));
}
__device__ __forceinline__ void mma3(float (&d)[4], const unsigned (&ah)[4], const unsigned (&al)[4], unsigned bh0,
                                     unsigned bh1, unsigned bl0, unsigned bl1) {
    mma_tf32(d, al[0], al[1], al[2], al[3], bh0, bh1);
    mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
    mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}

// bf16 MMAs (fp32 accumulation), m16n8k16 and m16n8k8, and the packing of bf16 pairs
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&v);
}
__device__ __forceinline__ void split_bf16(float v0, float v1, unsigned& h, unsigned& l) {
    const __nv_bfloat16 h0 = __float2bfloat16_rn(v0), h1 = __float2bfloat16_rn(v1);
    h = pack_bf16(__bfloat162float(h0), __bfloat162float(h1));
    l = pack_bf16(v0 - __bfloat162float(h0), v1 - __bfloat162float(h1));
}
__device__ __forceinline__ void mma_bf16_k16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16_k8(float (&d)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ unsigned movt(unsigned x) {
    unsigned y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}

// node_mars[nid0 + i, b] = (log2 S - log2 Z) ln 2 (-inf where S is 0), lzs[i, b] = log2 Z ln 2, from the column
// tiles' base-2 (max, sum) pairs: four lanes an output, each up to 8 tiles (its loads issued together), folded by
// two shuffles
__global__ void __launch_bounds__(128) fw_comb_kernel(const float4* __restrict__ part, int njt, float* __restrict__ nmars,
                                                        long nid0, float* __restrict__ lzs, int M, int Mp, int B) {
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
    nmars[(nid0 + i) * B + b] = ls > NEG ? ls - lz : ls;                   // (no mass: -inf, not 0 / 0)
    lzs[(long)i * B + b] = lz;
}

// The backward's epilogue: the element flows to the log, log-added into the circuit's (a thread a (child, sample))
__global__ void __launch_bounds__(256) ef_epi_kernel(const float* __restrict__ ef, const long* __restrict__ rows,
                                                     float* __restrict__ eflows, int N, int Np, int B, int lb) {
    const long t = (long)blockIdx.x * 256 + threadIdx.x;
    const int b = (int)(t & ((1 << lb) - 1));
    const long j = t >> lb;
    if (b >= B || j >= N) return;
    const long er = rows[j];
    const float v = ef[(long)b * Np + j];
    if (er < 0 || !(v > 0.f)) return;
    const float lv = logf(v);
    float* p = eflows + er * B + b;
    const float o = *p;
    *p = o == NEG ? lv : fmaxf(o, lv) + log1pf(__expf(-fabsf(o - lv)));
}

// =================================================================== bf16 products (the default precision, "auto")
// Fragment order, a 32-bit register two bf16 (the lower index low):
//   AF [b][rg][ks][lane][NRA]: register r -> node 16 rg + g + 8 (r & 1), ranks 16 ks + 2 tq + 8 (r >> 1) + {0, 1}
//                                (k8: NRA 2, node 16 rg + g + 8 r, ranks 2 tq + {0, 1})
//   BF [b][jt][ks][lane][NRB]: register 2 nf + r -> child BN jt + 8 nf + g, ranks 16 ks + 2 tq + 8 r + {0, 1}
//                                (k8: NRB 4, register nf, ranks 2 tq + {0, 1})
template <int KP> struct BfGeo {
    static constexpr bool K8 = KP <= 8;
    static constexpr int KS = K8 ? 1 : KP / 16;     // k-steps
    static constexpr int NRA = K8 ? 2 : 4;          // A registers a lane a k-step
    static constexpr int NRB = K8 ? 4 : 8;          // B registers a lane a k-step (4 n-tiles)
};

// The factors as the caller gives them, A [B, M, K], B [B, K, N]: a sample's block contiguous, so the fragment order is
// a permutation within 16-node (32-child) blocks -- reads and writes both along the sample's own run.
//   AF, BF (hi, lo) in the fragment order above; WF [B, njt, 36]: per tq the 8 factors w_j = 2^(lc'_j - mc) of the
//   lane's children (8 nf + 2 tq + p), then mc, the tile's max lc' (-inf: no edge reaches the tile's children);
//   lcs [B, Np] = lc' (the exact fallback's). A grid row a sample.
__global__ void __launch_bounds__(256) bf_fw_prep_kernel(
        const float* __restrict__ A, const float* __restrict__ Bm, unsigned* __restrict__ AFh, unsigned* __restrict__ AFl,
        unsigned* __restrict__ BFh, unsigned* __restrict__ BFl, float* __restrict__ WF, float* __restrict__ lcs,
        const float* __restrict__ emars, const long* __restrict__ rows, int M, int N, int Mp, int Np, int K, int KP,
        int B, int nbA, int nbB) {
    const int b = blockIdx.y;
    const bool k8 = KP <= 8;
    const int KS = k8 ? 1 : KP / 16, NRA = k8 ? 2 : 4, NRB = k8 ? 4 : 8;
    const long na = (long)Mp * KP / 2, nb = (long)Np * KP / 2;
    int blk = blockIdx.x;
    if (blk < nbA) {
        const int e = blk * 256 + threadIdx.x;
        if (e >= na) return;
        const int r = e % NRA, lane = (e / NRA) & 31, ks = (e / (NRA * 32)) % KS, rg = e / (NRA * 32 * KS);
        const int g = lane >> 2, tq = lane & 3;
        const int i = k8 ? 16 * rg + g + 8 * r : 16 * rg + g + 8 * (r & 1);
        const int k = k8 ? 2 * tq : 16 * ks + 2 * tq + 8 * (r >> 1);
        const float* src = A + ((long)b * M + i) * K + k;
        const float v0 = (i < M && k < K) ? __ldg(src) * LOG2E : 0.f;
        const float v1 = (i < M && k + 1 < K) ? __ldg(src + 1) * LOG2E : 0.f;
        unsigned h, l;
        split_bf16(v0, v1, h, l);
        AFh[(long)b * na + e] = h;
        AFl[(long)b * na + e] = l;
        return;
    }
    blk -= nbA;
    if (blk < nbB) {
        const int e = blk * 256 + threadIdx.x;
        if (e >= nb) return;
        const int u = e % NRB, lane = (e / NRB) & 31, ks = (e / (NRB * 32)) % KS, jt = e / (NRB * 32 * KS);
        const int g = lane >> 2, tq = lane & 3;
        const int nf = k8 ? u : u >> 1, j = BN * jt + 8 * nf + g;
        const int k = k8 ? 2 * tq : 16 * ks + 2 * tq + 8 * (u & 1);
        const float* src = Bm + ((long)b * K + k) * N + j;
        const float v0 = (j < N && k < K) ? __ldg(src) : 0.f;
        const float v1 = (j < N && k + 1 < K) ? __ldg(src + N) : 0.f;
        unsigned h, l;
        split_bf16(v0, v1, h, l);
        BFh[(long)b * nb + e] = h;
        BFl[(long)b * nb + e] = l;
        return;
    }
    blk -= nbB;                                                // (the column tiles' factors: a warp each)
    const int njt = Np / BN, jt = blk * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (jt >= njt) return;
    const int j = jt * BN + lane;
    const long rr = j < N ? rows[j] : -1;
    const float lc = rr >= 0 ? emars[rr * B + b] * LOG2E : NEG;
    float mc = lc;
#pragma unroll
    for (int d = 16; d; d >>= 1) mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, d));
    lcs[(long)b * Np + j] = lc;
    float* wp = WF + ((long)b * njt + jt) * 36;
    const int nf = lane >> 3, tq = (lane & 7) >> 1, p = lane & 1;
    wp[tq * 8 + 2 * nf + p] = mc == NEG ? 0.f : exp2f(lc - mc);
    if (lane < 4) wp[32 + lane] = lane == 0 ? mc : 0.f;
}

// The bf16 forward: `fwtc3_kernel` with phi from bf16 hi + lo MMAs, its fragments fetched one k-step ahead over the
// (sample, k-step) sequence
template <int KP, typename TP>
__global__ void __launch_bounds__(128) bf_fw_kernel(
        const unsigned* __restrict__ AFh, const unsigned* __restrict__ AFl, const unsigned* __restrict__ BFh,
        const unsigned* __restrict__ BFl, const TP* __restrict__ params, long p0, int Nb, int cbs, int bs,
        const float* __restrict__ WF, const float* __restrict__ lcs, float4* __restrict__ part, int M, int N, int Mp,
        int Np, int B, int bper) {
    using Gm = BfGeo<KP>;
    constexpr int KS = Gm::KS, NRA = Gm::NRA, NRB = Gm::NRB;
    constexpr bool K8 = Gm::K8;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int i0 = blockIdx.y * 64, j0 = blockIdx.x * BN, jt = blockIdx.x, njt = gridDim.x;
    const int rg = blockIdx.y * 4 + w, rA = i0 + 16 * w + g, rB = rA + 8;
    const int b0 = blockIdx.z * bper, b1 = min(B, b0 + bper);
    if (b0 >= b1) return;

    float lt[16];
#pragma unroll
    for (int e = 0; e < 16; e++) {
        const int nf = e >> 2, c = e & 3, i = c & 2 ? rB : rA, j = j0 + 8 * nf + 2 * tq + (c & 1);
        lt[e] = (i < M && j < N) ? to_f(params[p0 + grid_at(i, j, Nb, cbs, bs)]) : 0.f;
    }
#pragma unroll
    for (int e = 0; e < 16; e++) lt[e] = lt[e] > 0.f ? __log2f(lt[e]) : NEG;

    const long astr = (long)Mp * KP / 2, bstr = (long)Np * KP / 2, wstr = (long)njt * 36;
    const unsigned* ah0 = AFh + ((long)rg * KS * 32 + lane) * NRA;
    const unsigned* al0 = AFl + ((long)rg * KS * 32 + lane) * NRA;
    const unsigned* bh0 = BFh + ((long)jt * KS * 32 + lane) * NRB;
    const unsigned* bl0 = BFl + ((long)jt * KS * 32 + lane) * NRB;
    const float* wf = WF + (long)jt * 36 + tq * 8;
    unsigned nah[NRA], nal[NRA], nbh[NRB], nbl[NRB];
    auto fetch = [&](int b, int ks) {
        const long ao = b * astr + ks * 32 * NRA, bo = b * bstr + ks * 32 * NRB;
        if constexpr (NRA == 4) {
            const uint4 x = __ldg(reinterpret_cast<const uint4*>(ah0 + ao)), y = __ldg(reinterpret_cast<const uint4*>(al0 + ao));
            nah[0] = x.x; nah[1] = x.y; nah[2] = x.z; nah[3] = x.w; nal[0] = y.x; nal[1] = y.y; nal[2] = y.z; nal[3] = y.w;
        } else {
            const uint2 x = __ldg(reinterpret_cast<const uint2*>(ah0 + ao)), y = __ldg(reinterpret_cast<const uint2*>(al0 + ao));
            nah[0] = x.x; nah[1] = x.y; nal[0] = y.x; nal[1] = y.y;
        }
#pragma unroll
        for (int u = 0; u < NRB; u += 4) {
            const uint4 x = __ldg(reinterpret_cast<const uint4*>(bh0 + bo + u));
            const uint4 y = __ldg(reinterpret_cast<const uint4*>(bl0 + bo + u));
            nbh[u] = x.x; nbh[u + 1] = x.y; nbh[u + 2] = x.z; nbh[u + 3] = x.w;
            nbl[u] = y.x; nbl[u + 1] = y.y; nbl[u + 2] = y.z; nbl[u + 3] = y.w;
        }
    };
    fetch(b0, 0);
#pragma unroll (KP >= 16 ? 2 : 1)                                // (two samples interleaved past rank 8)
    for (int b = b0; b < b1; b++) {
        float wv[8];
        {
            const float4 w0 = __ldg(reinterpret_cast<const float4*>(wf + b * wstr));
            const float4 w1 = __ldg(reinterpret_cast<const float4*>(wf + b * wstr + 4));
            wv[0] = w0.x; wv[1] = w0.y; wv[2] = w0.z; wv[3] = w0.w; wv[4] = w1.x; wv[5] = w1.y; wv[6] = w1.z; wv[7] = w1.w;
        }
        const float mc = __ldg(WF + b * wstr + (long)jt * 36 + 32);
        float S[4][4];
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) S[nf][c] = 0.f;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned ah[NRA], al[NRA], bh[NRB], bl[NRB];
#pragma unroll
            for (int u = 0; u < NRA; u++) { ah[u] = nah[u]; al[u] = nal[u]; }
#pragma unroll
            for (int u = 0; u < NRB; u++) { bh[u] = nbh[u]; bl[u] = nbl[u]; }
            if (ks + 1 < KS) fetch(b, ks + 1);                 // (the next k-step's, or the next sample's first)
            else if (b + 1 < b1) fetch(b + 1, 0);
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                if constexpr (K8) {
                    mma_bf16_k8(S[nf], al[0], al[1], bh[nf]);
                    mma_bf16_k8(S[nf], ah[0], ah[1], bl[nf]);
                    mma_bf16_k8(S[nf], ah[0], ah[1], bh[nf]);
                } else {
                    mma_bf16_k16(S[nf], al[0], al[1], al[2], al[3], bh[2 * nf], bh[2 * nf + 1]);
                    mma_bf16_k16(S[nf], ah[0], ah[1], ah[2], ah[3], bl[2 * nf], bl[2 * nf + 1]);
                    mma_bf16_k16(S[nf], ah[0], ah[1], ah[2], ah[3], bh[2 * nf], bh[2 * nf + 1]);
                }
            }
        }
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) S[nf][c] += lt[4 * nf + c];

        float a[2] = {NEG, NEG};
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) a[c >> 1] = fmaxf(a[c >> 1], S[nf][c]);
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            a[0] = fmaxf(a[0], __shfl_xor_sync(0xffffffffu, a[0], d));
            a[1] = fmaxf(a[1], __shfl_xor_sync(0xffffffffu, a[1], d));
        }
        const float sa0 = a[0] == NEG ? 0.f : a[0], sa1 = a[1] == NEG ? 0.f : a[1];
        float z[2] = {0.f, 0.f}, s[2] = {0.f, 0.f};
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int h = c >> 1;
                const float e = exp2f(S[nf][c] - (h ? sa1 : sa0));
                z[h] += e;
                s[h] = fmaf(e, wv[2 * nf + (c & 1)], s[h]);
            }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            z[0] += __shfl_xor_sync(0xffffffffu, z[0], d);
            z[1] += __shfl_xor_sync(0xffffffffu, z[1], d);
            s[0] += __shfl_xor_sync(0xffffffffu, s[0], d);
            s[1] += __shfl_xor_sync(0xffffffffu, s[1], d);
        }
        float ms[2] = {s[0] > 0.f ? a[0] + mc : NEG, s[1] > 0.f ? a[1] + mc : NEG};
        const bool need = mc > NEG && ((s[0] < 0x1p-97f && a[0] > NEG) || (s[1] < 0x1p-97f && a[1] > NEG));
        if (__any_sync(0xffffffffu, need)) {                   // (exactly: x' = lw' + lc' under its own max)
            float lcv[4][2];
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                const float2 v = *reinterpret_cast<const float2*>(lcs + (long)b * Np + j0 + 8 * nf + 2 * tq);
                lcv[nf][0] = v.x;
                lcv[nf][1] = v.y;
            }
            float m2[2] = {NEG, NEG};
#pragma unroll
            for (int nf = 0; nf < 4; nf++)
#pragma unroll
                for (int c = 0; c < 4; c++) m2[c >> 1] = fmaxf(m2[c >> 1], S[nf][c] + lcv[nf][c & 1]);
#pragma unroll
            for (int d = 1; d <= 2; d <<= 1) {
                m2[0] = fmaxf(m2[0], __shfl_xor_sync(0xffffffffu, m2[0], d));
                m2[1] = fmaxf(m2[1], __shfl_xor_sync(0xffffffffu, m2[1], d));
            }
            const float sc0 = m2[0] == NEG ? 0.f : m2[0], sc1 = m2[1] == NEG ? 0.f : m2[1];
            float s2[2] = {0.f, 0.f};
#pragma unroll
            for (int nf = 0; nf < 4; nf++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int h = c >> 1;
                    s2[h] += exp2f(S[nf][c] + lcv[nf][c & 1] - (h ? sc1 : sc0));
                }
#pragma unroll
            for (int d = 1; d <= 2; d <<= 1) {
                s2[0] += __shfl_xor_sync(0xffffffffu, s2[0], d);
                s2[1] += __shfl_xor_sync(0xffffffffu, s2[1], d);
            }
            ms[0] = s2[0] > 0.f ? m2[0] : NEG;
            ms[1] = s2[1] > 0.f ? m2[1] : NEG;
            s[0] = s2[0];
            s[1] = s2[1];
        }
        if (tq == 0) {
            if (rA < M) part[((long)b * njt + jt) * Mp + rA] = make_float4(a[0], z[0], ms[0], s[0]);
            if (rB < M) part[((long)b * njt + jt) * Mp + rB] = make_float4(a[1], z[1], ms[1], s[1]);
        }
    }
}


// The bf16 backward's prologue, one launch, a grid row a sample: AF, BF (hi, lo) as the forward makes them (its
// fragments again: nothing kept between the passes), the nodes' terms LU [B, Mp, 2] = (lu', lv') -- lu' = log2 (f /
// (n Z)), lv' = log2 (f / Z), -inf for a node without flow --, the children's LC [B, njt, 36] (per tq the lane's 8
// children's lc', as WF's factors) and the element-flow scratch ef [B, Np] zeroed
__global__ void __launch_bounds__(256) bf_bw_prep_kernel(
        const float* __restrict__ A, const float* __restrict__ Bm, unsigned* __restrict__ AFh, unsigned* __restrict__ AFl,
        unsigned* __restrict__ BFh, unsigned* __restrict__ BFl, float* __restrict__ LU, float* __restrict__ LC,
        float* __restrict__ ef, const float* __restrict__ nflows, const float* __restrict__ nmars, long nid0,
        const float* __restrict__ lzs, const float* __restrict__ emars, const long* __restrict__ rows, int M, int N,
        int Mp, int Np, int K, int KP, int B, int nbA, int nbB) {
    const int b = blockIdx.y, njt = Np / BN;
    const bool k8 = KP <= 8;
    const int KS = k8 ? 1 : KP / 16, NRA = k8 ? 2 : 4, NRB = k8 ? 4 : 8;
    const long na = (long)Mp * KP / 2, nb = (long)Np * KP / 2;
    int blk = blockIdx.x;
    if (blk < nbA) {
        const int e = blk * 256 + threadIdx.x;
        if (e >= na) return;
        const int r = e % NRA, lane = (e / NRA) & 31, ks = (e / (NRA * 32)) % KS, rg = e / (NRA * 32 * KS);
        const int g = lane >> 2, tq = lane & 3;
        const int i = k8 ? 16 * rg + g + 8 * r : 16 * rg + g + 8 * (r & 1);
        const int k = k8 ? 2 * tq : 16 * ks + 2 * tq + 8 * (r >> 1);
        const float* src = A + ((long)b * M + i) * K + k;
        const float v0 = (i < M && k < K) ? __ldg(src) * LOG2E : 0.f;
        const float v1 = (i < M && k + 1 < K) ? __ldg(src + 1) * LOG2E : 0.f;
        unsigned h, l;
        split_bf16(v0, v1, h, l);
        AFh[(long)b * na + e] = h;
        AFl[(long)b * na + e] = l;
        return;
    }
    blk -= nbA;
    if (blk < nbB) {
        const int e = blk * 256 + threadIdx.x;
        if (e >= nb) return;
        const int u = e % NRB, lane = (e / NRB) & 31, ks = (e / (NRB * 32)) % KS, jt = e / (NRB * 32 * KS);
        const int g = lane >> 2, tq = lane & 3;
        const int nf = k8 ? u : u >> 1, j = BN * jt + 8 * nf + g;
        const int k = k8 ? 2 * tq : 16 * ks + 2 * tq + 8 * (u & 1);
        const float* src = Bm + ((long)b * K + k) * N + j;
        const float v0 = (j < N && k < K) ? __ldg(src) : 0.f;
        const float v1 = (j < N && k + 1 < K) ? __ldg(src + N) : 0.f;
        unsigned h, l;
        split_bf16(v0, v1, h, l);
        BFh[(long)b * nb + e] = h;
        BFl[(long)b * nb + e] = l;
        return;
    }
    blk -= nbB;
    const int o = blk * 256 + threadIdx.x;
    if (o < Mp) {
        float lu = NEG, lv = NEG;
        if (o < M) {
            const float lf = nflows[(nid0 + o) * B + b];
            if (isfinite(lf)) {
                const float lz = lzs[(long)o * B + b];
                lu = (lf - (nmars[(nid0 + o) * B + b] + lz)) * LOG2E;
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
            v = r >= 0 ? emars[r * B + b] * LOG2E : NEG;
        }
        LC[((long)b * njt + jt) * 36 + w] = v;
    } else if (o < Mp + njt * 36 + Np) {
        ef[(long)b * Np + (o - Mp - njt * 36)] = 0.f;
    }
}

// The bf16 backward (the file's head has the design): a warp a 16-node x 32-child tile over the CTA's samples.
template <int KP, typename TP, bool FLOWS, bool DENOM, bool GRADS>
__global__ void __launch_bounds__(128) bf_bw_kernel(
        const unsigned* __restrict__ AFh, const unsigned* __restrict__ AFl, const unsigned* __restrict__ BFh,
        const unsigned* __restrict__ BFl, const TP* __restrict__ params, long p0, int Nb, int cbs, int bs, const float* __restrict__ LU,
        const float* __restrict__ LC, float* __restrict__ ef, float* __restrict__ pflows, long pf0,
        float* __restrict__ denom, long dn0, float* __restrict__ gA, float* __restrict__ gB, int M, int N, int Mp,
        int Np, int K, int B, int bper) {
    using Gm = BfGeo<KP>;
    constexpr int KS = Gm::KS, NRA = Gm::NRA, NRB = Gm::NRB, NT = KP / 8;
    constexpr bool K8 = Gm::K8;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int i0 = blockIdx.y * 64, j0 = blockIdx.x * BN, jt = blockIdx.x, njt = gridDim.x;
    const int rg = blockIdx.y * 4 + w, rA = i0 + 16 * w + g, rB = rA + 8;
    const int b0 = blockIdx.z * bper, b1 = min(B, b0 + bper);
    if (b0 >= b1) return;
    // stages a sample parity: the warps' G^T blocks [2][4][8][32], A' fragments [2][4][NT][2][32], element flows [2][4][BN]
    __shared__ __align__(16) unsigned sT[(GRADS ? 2 * 4 * 8 * 32 : 0) > 4 * 32 * 20 ? 2 * 4 * 8 * 32 : 4 * 32 * 20];
    __shared__ unsigned sA[GRADS ? 2 * 4 * NT * 64 : 1];
    __shared__ __align__(16) float sQ[GRADS ? 4 * 8 * 20 : 4];     // (each warp's dB unit, re-laid for gB's rows)
    __shared__ float sE[2 * 4 * BN];

    float lt[16], fp[16], tw[16];
#pragma unroll
    for (int e = 0; e < 16; e++) {
        const int nf = e >> 2, c = e & 3, i = c & 2 ? rB : rA, j = j0 + 8 * nf + 2 * tq + (c & 1);
        lt[e] = (i < M && j < N) ? to_f(params[p0 + grid_at(i, j, Nb, cbs, bs)]) : 0.f;
        fp[e] = 0.f;
        tw[e] = 0.f;
    }
#pragma unroll
    for (int e = 0; e < 16; e++) lt[e] = lt[e] > 0.f ? __log2f(lt[e]) : NEG;

    const long astr = (long)Mp * KP / 2, bstr = (long)Np * KP / 2;
    const unsigned* ah0 = AFh + ((long)rg * KS * 32 + lane) * NRA;
    const unsigned* al0 = AFl + ((long)rg * KS * 32 + lane) * NRA;
    const unsigned* bh0 = BFh + ((long)jt * KS * 32 + lane) * NRB;
    const unsigned* bl0 = BFl + ((long)jt * KS * 32 + lane) * NRB;
    const float2* lu0 = reinterpret_cast<const float2*>(LU);
    const float* lc0 = LC + (long)jt * 36 + tq * 8;
    unsigned nah[NRA], nal[NRA], nbh[NRB], nbl[NRB];
    auto fetch = [&](int b, int ks) {
        const long ao = b * astr + ks * 32 * NRA, bo = b * bstr + ks * 32 * NRB;
        if constexpr (NRA == 4) {
            const uint4 x = __ldg(reinterpret_cast<const uint4*>(ah0 + ao)), y = __ldg(reinterpret_cast<const uint4*>(al0 + ao));
            nah[0] = x.x; nah[1] = x.y; nah[2] = x.z; nah[3] = x.w; nal[0] = y.x; nal[1] = y.y; nal[2] = y.z; nal[3] = y.w;
        } else {
            const uint2 x = __ldg(reinterpret_cast<const uint2*>(ah0 + ao)), y = __ldg(reinterpret_cast<const uint2*>(al0 + ao));
            nah[0] = x.x; nah[1] = x.y; nal[0] = y.x; nal[1] = y.y;
        }
#pragma unroll
        for (int u = 0; u < NRB; u += 4) {
            const uint4 x = __ldg(reinterpret_cast<const uint4*>(bh0 + bo + u));
            const uint4 y = __ldg(reinterpret_cast<const uint4*>(bl0 + bo + u));
            nbh[u] = x.x; nbh[u + 1] = x.y; nbh[u + 2] = x.z; nbh[u + 3] = x.w;
            nbl[u] = y.x; nbl[u + 1] = y.y; nbl[u + 2] = y.z; nbl[u + 3] = y.w;
        }
    };
    fetch(b0, 0);
#pragma unroll 1
    for (int b = b0; b < b1; b++) {
        const int par = b & 1;
        // this sample's terms and gradient operands (used after phi: their latency under its MMAs)
        const float2 tA = __ldg(lu0 + (long)b * Mp + rA), tB = __ldg(lu0 + (long)b * Mp + rB);
        float lcv[8];
        {
            const float4 c0 = __ldg(reinterpret_cast<const float4*>(lc0 + (long)b * njt * 36));
            const float4 c1 = __ldg(reinterpret_cast<const float4*>(lc0 + (long)b * njt * 36 + 4));
            lcv[0] = c0.x; lcv[1] = c0.y; lcv[2] = c0.z; lcv[3] = c0.w; lcv[4] = c1.x; lcv[5] = c1.y; lcv[6] = c1.z; lcv[7] = c1.w;
        }
        // dA's B operand (B^T: children 2 tq .., rank g) and dB's (A': nodes 2 tq .., rank g): phi's hi fragments, each
        // 8 x 8 block transposed (`movmatrix`), one bf16 term
        unsigned bgv[GRADS ? NT : 1][4], agv[GRADS ? NT : 1][2];
        // ---- phi' (bf16 hi + lo, as the forward)
        float S[4][4];
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) S[nf][c] = 0.f;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            unsigned ah[NRA], al[NRA], bh[NRB], bl[NRB];
#pragma unroll
            for (int u = 0; u < NRA; u++) { ah[u] = nah[u]; al[u] = nal[u]; }
#pragma unroll
            for (int u = 0; u < NRB; u++) { bh[u] = nbh[u]; bl[u] = nbl[u]; }
            if (ks + 1 < KS) fetch(b, ks + 1);
            else if (b + 1 < b1) fetch(b + 1, 0);
            if (GRADS) {
                if constexpr (K8) {                            // (ranks 0..7: one rank tile)
                    agv[0][0] = movt(ah[0]);
                    agv[0][1] = movt(ah[1]);
#pragma unroll
                    for (int nf = 0; nf < 4; nf++) bgv[0][nf] = movt(bh[nf]);
                } else {                                       // (ranks 16 ks + 8 r ..: rank tile 2 ks + r)
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        agv[2 * ks + r][0] = movt(ah[2 * r]);
                        agv[2 * ks + r][1] = movt(ah[2 * r + 1]);
#pragma unroll
                        for (int nf = 0; nf < 4; nf++) bgv[2 * ks + r][nf] = movt(bh[2 * nf + r]);
                    }
                }
            }
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                if constexpr (K8) {
                    mma_bf16_k8(S[nf], al[0], al[1], bh[nf]);
                    mma_bf16_k8(S[nf], ah[0], ah[1], bl[nf]);
                    mma_bf16_k8(S[nf], ah[0], ah[1], bh[nf]);
                } else {
                    mma_bf16_k16(S[nf], al[0], al[1], al[2], al[3], bh[2 * nf], bh[2 * nf + 1]);
                    mma_bf16_k16(S[nf], ah[0], ah[1], ah[2], ah[3], bl[2 * nf], bl[2 * nf + 1]);
                    mma_bf16_k16(S[nf], ah[0], ah[1], ah[2], ah[3], bh[2 * nf], bh[2 * nf + 1]);
                }
            }
        }
        // ---- the element terms: q = 2^(lw' + lu' + lc'), t = 2^(lw' + lv')
        float Q[4][4];
        unsigned R[4][2];                                      // (q - t) as bf16 pairs: (n-tile, node half)
#pragma unroll
        for (int nf = 0; nf < 4; nf++) {
            float gq[4];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const float lw = S[nf][c] + lt[4 * nf + c];
                const float q = exp2f(lw + (c & 2 ? tB.x : tA.x) + lcv[2 * nf + (c & 1)]);
                const float t = exp2f(lw + (c & 2 ? tB.y : tA.y));
                if (FLOWS) fp[4 * nf + c] += q;
                if (DENOM) tw[4 * nf + c] += t;
                Q[nf][c] = q;
                gq[c] = q - t;
            }
            R[nf][0] = pack_bf16(gq[0], gq[1]);
            R[nf][1] = pack_bf16(gq[2], gq[3]);
        }
        // ---- the element flows: each child's sum over the warp's 16 nodes, a reduce-scatter over the 8 lanes of a tq
        {
            float v[8];
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                v[2 * nf] = Q[nf][0] + Q[nf][2];
                v[2 * nf + 1] = Q[nf][1] + Q[nf][3];
            }
            const bool h2 = g & 4, h1 = g & 2, h0 = g & 1;
            float u4[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const float send = h2 ? v[m] : v[m + 4], keep = h2 ? v[m + 4] : v[m];
                u4[m] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
            }
            float u2[2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                const float send = h1 ? u4[m] : u4[m + 2], keep = h1 ? u4[m + 2] : u4[m];
                u2[m] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
            }
            const float send = h0 ? u2[0] : u2[1], keep = h0 ? u2[1] : u2[0];
            const float tot = keep + __shfl_xor_sync(0xffffffffu, send, 4);
            const int m = (h2 ? 4 : 0) + (h1 ? 2 : 0) + (h0 ? 1 : 0);     // v's index this lane holds: m = 2 nf + p
            const int col = 8 * (m >> 1) + 2 * tq + (m & 1);
            sE[(par * 4 + w) * BN + col] = tot;
        }
        if (GRADS) {
            // ---- dA += G B^T: the A operand (k16 over children 16 ks..): R[2 ks][0], R[2 ks][1], R[2 ks + 1][0], R[2 ks + 1][1]
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                mma_bf16_k16(d, R[0][0], R[0][1], R[1][0], R[1][1], bgv[nt][0], bgv[nt][1]);
                mma_bf16_k16(d, R[2][0], R[2][1], R[3][0], R[3][1], bgv[nt][2], bgv[nt][3]);
                // (node g / g + 8, ranks 8 nt + 2 tq, + 1)
                const int k = 8 * nt + 2 * tq;                 // (into gA [B, M, K]: ranks past K padding)
                if (k < K) {
                    if (rA < M) add_pair(gA + ((long)b * M + rA) * K + k, d[0], d[1], k + 1 < K);
                    if (rB < M) add_pair(gA + ((long)b * M + rB) * K + k, d[2], d[3], k + 1 < K);
                }
            }
            // ---- dB^T += G^T A' over the CTA's 64 nodes: each warp stages its G^T blocks (child g, nodes 2 tq ..)
            // and its A' fragments, then units (child m-tile, rank tile) over the four warps' nodes as k-steps
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                sT[((par * 4 + w) * 8 + 2 * nf) * 32 + lane] = movt(R[nf][0]);
                sT[((par * 4 + w) * 8 + 2 * nf + 1) * 32 + lane] = movt(R[nf][1]);
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                sA[((par * 4 + w) * NT + nt) * 64 + lane] = agv[nt][0];
                sA[((par * 4 + w) * NT + nt) * 64 + 32 + lane] = agv[nt][1];
            }
        }
        __syncthreads();                                       // (the four warps' stages of this sample written)
        if (tid < BN) {                                        // the element flows: the four warps' sums, one add
            const float v = (sE[(par * 4) * BN + tid] + sE[(par * 4 + 1) * BN + tid]) +
                            (sE[(par * 4 + 2) * BN + tid] + sE[(par * 4 + 3) * BN + tid]);
            if (v != 0.f && j0 + tid < N) red1(ef + (long)b * Np + j0 + tid, v);
        }
        if (GRADS) {
            for (int u = w; u < 2 * NT; u += 4) {
                const int mt = u & 1, nt = u >> 1;
                float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kw = 0; kw < 4; kw++) {
                    const unsigned* tp = sT + (par * 4 + kw) * 8 * 32 + lane;
                    const unsigned* ap = sA + ((par * 4 + kw) * NT + nt) * 64 + lane;
                    mma_bf16_k16(d, tp[(4 * mt) * 32], tp[(4 * mt + 2) * 32], tp[(4 * mt + 1) * 32], tp[(4 * mt + 3) * 32],
                                 ap[0], ap[32]);
                }
                // (child 16 mt + g / + 8, ranks 8 nt + 2 tq, + 1)
                const int ja = j0 + 16 * mt + g, jb = ja + 8;
                // into gB [B, K, N] (A' carried log2 e): the unit's 8 ranks x 16 children through the warp's own
                // shared rows, then a lane 4 consecutive children of one rank -- 64-byte runs, one 16-byte add a lane
                {
                    constexpr float IL2 = 0.6931471805599453f;
                    float* wq = sQ + w * (8 * 20);                 // [rank 8][child 16 + 4]
                    wq[(2 * tq) * 20 + g] = d[0] * IL2;
                    wq[(2 * tq + 1) * 20 + g] = d[1] * IL2;
                    wq[(2 * tq) * 20 + g + 8] = d[2] * IL2;
                    wq[(2 * tq + 1) * 20 + g + 8] = d[3] * IL2;
                    __syncwarp();
                    const int rk = lane >> 2, c4 = 4 * (lane & 3), k = 8 * nt + rk, jc = j0 + 16 * mt + c4;
                    const float4 v = *reinterpret_cast<const float4*>(wq + rk * 20 + c4);
                    __syncwarp();
                    if (k < K) {
                        float* p = gB + ((long)b * K + k) * N + jc;
                        if (jc + 3 < N && (reinterpret_cast<unsigned long>(p) & 15) == 0) red4(p, v);
                        else {
                            if (jc < N) red1(p, v.x);
                            if (jc + 1 < N) red1(p + 1, v.y);
                            if (jc + 2 < N) red1(p + 2, v.z);
                            if (jc + 3 < N) red1(p + 3, v.w);
                        }
                    }
                }
            }
        }
    }
    // F+ and W of the warp's 16 x 32 cells: through its own shared rows (the stages free now), a child's 16 nodes
    // contiguous in the parameters' block-major runs -- a lane four of them, one 16-byte add (scalar adds where the
    // node blocks are under 4 or the runs unaligned)
    if (FLOWS || DENOM) {
        __syncthreads();                                       // (every warp's last reads of the stages done)
        float* st = reinterpret_cast<float*>(sT) + w * (32 * 20);     // [child 32][node 16 + 4]
        const bool vec = (bs & 3) == 0 && ((pf0 | dn0) & 3) == 0;
#pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            if ((pass == 0 && !FLOWS) || (pass == 1 && !DENOM)) continue;
#pragma unroll
            for (int e = 0; e < 16; e++) {
                const int nf = e >> 2, c = e & 3;
                float v = pass == 0 ? fp[e] : tw[e];
                if (pass == 1) {
                    const float th = exp2f(lt[e]);
                    v = th > 0.f ? v / th : 0.f;
                }
                st[(8 * nf + 2 * tq + (c & 1)) * 20 + g + 8 * (c >> 1)] = v;
            }
            __syncwarp();
            float* dst = pass == 0 ? pflows + pf0 : denom + dn0;
#pragma unroll
            for (int u = 0; u < 4; u++) {                      // (8 children an instruction, 4 lanes a child)
                const int col = 8 * u + (lane >> 2), r4 = 4 * (lane & 3), i = i0 + 16 * w + r4, j = j0 + col;
                const float4 v = *reinterpret_cast<const float4*>(st + col * 20 + r4);
                if (j >= N || i >= M) continue;
                const long o = grid_at(i, j, Nb, cbs, bs);
                if (vec && i + 3 < M) {
                    if (v.x != 0.f || v.y != 0.f || v.z != 0.f || v.w != 0.f) red4(dst + o, v);
                } else {
                    const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int q = 0; q < 4; q++)
                        if (i + q < M && vv[q] != 0.f) red1(dst + grid_at(i + q, j, Nb, cbs, bs), vv[q]);
                }
            }
            __syncwarp();
        }
    }
}


// ===================================== the bf16 backward on big tiles (`bf_bw2_kernel`, ranks 9 .. 64: padded 16, 32, 64)
// A CTA of 8 warps works a tile of 128 nodes x 64 children -- a warp 16 nodes x the tile's 64 children (eight n-tiles)
// -- over a run of (tile, sample) units: persistent, the units in tile-major order split evenly over the CTAs, so a
// CTA's run spans a tile or two (theta, F+ and W in registers over a tile's samples, F+ and W added where the run leaves
// the tile; a tile's theta loaded at its start, its log2 taken under the first sample's phi). A sample: B's fragments
// (hi, lo: the tile's 64 children) staged in shared memory a sample ahead (cp.async), read by the 8 warps; A's from
// global memory into each warp's registers (its own 16 nodes, one k-step ahead) -- the run's next tile's first stage and
// A fragments fetched in the current tile's last sample. dA = G B^T over the tile's 64 children, all its rank tiles at
// once (B^T: the staged hi fragments transposed), added through the warp's own shared rows in whole runs of a row's
// ranks; dB^T = G^T A' over the tile's 128 nodes -- the 8 warps' G^T blocks and A' fragments (transposed) staged in the
// MMA's operand order (one 16- / 8-byte load an operand), units (child m-tiles, rank tile) over the warps, a warp's
// units interleaved and a short one's sum over the 8 node blocks split over partial accumulators (4 MMA chains a warp).
// Two barriers a sample. The dA and dB partial sums (4 K (128 + 64) bytes a tile and sample) are its L2 traffic's
// largest part: L2 takes adds at ~3 TB/s (loads ~6.3).
constexpr int TM2 = 128, TN2 = 64, NW2 = 8;
template <int KP> struct Bw2Geo {
    static_assert(KP >= 16 && KP % 16 == 0, "the big tiles: padded ranks 16, 32, 64");
    static constexpr int KS = KP / 16, NT = KP / 8;
    static constexpr int SB = 2 * KS * 256;                    // words: a (sample, hi | lo) stage -- 2 jt x KS blocks
    static constexpr int SA = NW2 * NT * 64;                   // A' hi fragments transposed: [warp][nt][2][32]
    static constexpr int ST = NW2 * 16 * 32;                   // G^T blocks: [warp][2 nf + h][32]
    static constexpr int SE = NW2 * TN2;                       // element flows: [warp][64]
    static constexpr int PW = KP < 32 ? KP : 32;               // dA: a row's ranks a flush
    static constexpr int SP = NW2 * 16 * (PW + 4);             // dA rows: [warp][node 16][PW + 4]
    static constexpr int UM = 2 * NT >= NW2 ? 2 : 1;           // dB: m-tiles a unit (128-byte runs at 2)
    static constexpr int UW = 16 * UM + 4;
    static constexpr int SQ = NW2 * 8 * UW;                    // dB rows: [warp][rank 8][16 UM + 4]
    static constexpr int MAIN = 4 * SB + SA + ST + SE + SP + SQ;
    static constexpr int FLUSH = NW2 * 32 * 20;                // F+ / W: [warp][child 32][node 16 + 4], a half at a time
    static constexpr int WORDS = MAIN > 4 * SB + FLUSH ? MAIN : 4 * SB + FLUSH;
};

template <int KP, typename TP, bool FLOWS, bool DENOM, bool GRADS>
__global__ void __launch_bounds__(256, 1) bf_bw2_kernel(
        const unsigned* __restrict__ AFh, const unsigned* __restrict__ AFl, const unsigned* __restrict__ BFh,
        const unsigned* __restrict__ BFl, const TP* __restrict__ params, long p0, int Nb, int cbs, int bs,
        const float* __restrict__ LU, const float* __restrict__ LC, float* __restrict__ ef, float* __restrict__ pflows,
        long pf0, float* __restrict__ denom, long dn0, float* __restrict__ gA, float* __restrict__ gB, int M, int N,
        int Mp, int Np, int K, int B, int nct, long units) {
    using Gm = Bw2Geo<KP>;
    constexpr int KS = Gm::KS, NT = Gm::NT, PW = Gm::PW, UM = Gm::UM, UW = Gm::UW;
    extern __shared__ __align__(16) unsigned smw[];
    unsigned* const sB = smw;                                  // [par][hi | lo][blk = jl KS + ks][half][lane][4]
    unsigned* const sA = sB + 4 * Gm::SB;
    unsigned* const sT = sA + Gm::SA;
    float* const sE = reinterpret_cast<float*>(sT + Gm::ST);
    float* const sP = sE + Gm::SE;
    float* const sQ = sP + Gm::SP;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const long astr = (long)Mp * KP / 2, bstr = (long)Np * KP / 2;
    const int njt = Np / BN;
    const float2* lu0 = reinterpret_cast<const float2*>(LU);
    const long u0 = (long)blockIdx.x * units / gridDim.x, u1 = (long)(blockIdx.x + 1) * units / gridDim.x;
    // sample b's B fragments (hi and lo, child tile jcs's two 32-child blocks) into stage `par`, the lanes' two 16-byte
    // halves apart (a warp's reads of one half conflict-free)
    auto stage = [&](int b, int par, int jcs) {
        constexpr int CH = 2 * KS * 64;                        // 16-byte chunks a (hi | lo) stage
#pragma unroll
        for (int q0 = 0; q0 < 2 * CH; q0 += 256) {
            const int q = q0 + tid, hl = q / CH, r = q - hl * CH, blk = r >> 6, ln = (r >> 1) & 31, half = r & 1;
            const unsigned* src = (hl ? BFl : BFh) + (long)b * bstr + ((long)2 * jcs * KS + blk) * 256 + ln * 8 + half * 4;
            cp_async16(sB + (2 * par + hl) * Gm::SB + blk * 256 + half * 128 + ln * 4, src);
        }
        cp_async_commit();
    };
    // a warp's A fragments: k-step ks of sample b, row group rgs
    uint4 nah, nal;
    auto fetchA = [&](int b, int ks, int rgs) {
        const long o = (long)b * astr + ((long)rgs * KS * 32 + lane) * 4 + ks * 128;
        nah = __ldg(reinterpret_cast<const uint4*>(AFh + o));
        nal = __ldg(reinterpret_cast<const uint4*>(AFl + o));
    };
    int pc = 0;                                                // the stage parity of the sample at hand
    if (u0 < u1) {                                             // the run's first sample: its stage, its A fragments
        const int t = (int)(u0 / B), b = (int)(u0 - (long)t * B), it = t / nct;
        stage(b, 0, t - it * nct);
        fetchA(b, 0, it * NW2 + w);
    }
#pragma unroll 1
    for (long u = u0; u < u1;) {
        const int t = (int)(u / B), b0 = (int)(u - (long)t * B);
        const int b1 = (int)min((long)B, (long)b0 + (u1 - u));
        u += b1 - b0;
        const bool more = u < u1;                              // (the run's next tile: t + 1, from its sample 0)
        const int it = t / nct, jc = t - it * nct;
        const int i0 = it * TM2, j0 = jc * TN2, jt0 = 2 * jc;
        const int rg = it * NW2 + w, rA = i0 + 16 * w + g, rB = rA + 8;
        const int itn = (t + 1) / nct, jcn = (t + 1) - itn * nct, rgn = itn * NW2 + w;

        // the tile's theta (its log2 taken under the first sample's phi), F+, W: element e = 4 nf + c -> row
        // c & 2 ? rB : rA, child j0 + 8 nf + 2 tq + (c & 1)
        float lt[32], fp[32], tw[32];
#pragma unroll
        for (int e = 0; e < 32; e++) {
            const int nf = e >> 2, c = e & 3, i = c & 2 ? rB : rA, j = j0 + 8 * nf + 2 * tq + (c & 1);
            lt[e] = (i < M && j < N) ? to_f(params[p0 + grid_at(i, j, Nb, cbs, bs)]) : 0.f;
            fp[e] = 0.f;
            tw[e] = 0.f;
        }

#pragma unroll 1
        for (int b = b0; b < b1; b++, pc ^= 1) {
            const int par = pc;
            cp_async_wait<0>();
            __syncthreads();                                   // [A] b's stage landed; b - 1's dB done (sA, sT, sE free)
            if (b + 1 < b1) stage(b + 1, par ^ 1, jc);         // (its last readers: b - 1's phi and dA, before [A])
            else if (more) stage(0, par ^ 1, jcn);             // (the next tile's first sample)
            const unsigned* const sbh = sB + 2 * par * Gm::SB;
            const unsigned* const sbl = sbh + Gm::SB;
            const float2 tA = __ldg(lu0 + (long)b * Mp + rA), tB = __ldg(lu0 + (long)b * Mp + rB);
            float lcv[16];                                     // lc' of children j0 + 8 nf + 2 tq + p at 2 nf + p
#pragma unroll
            for (int jl = 0; jl < 2; jl++) {
                const float* lc0 = LC + ((long)b * njt + jt0 + jl) * 36 + tq * 8;
                const float4 c0 = __ldg(reinterpret_cast<const float4*>(lc0));
                const float4 c1 = __ldg(reinterpret_cast<const float4*>(lc0 + 4));
                lcv[8 * jl + 0] = c0.x; lcv[8 * jl + 1] = c0.y; lcv[8 * jl + 2] = c0.z; lcv[8 * jl + 3] = c0.w;
                lcv[8 * jl + 4] = c1.x; lcv[8 * jl + 5] = c1.y; lcv[8 * jl + 6] = c1.z; lcv[8 * jl + 7] = c1.w;
            }
            // ---- phi' (bf16 hi + lo): the warp's A fragments, the staged B fragments
            float S[8][4];
#pragma unroll
            for (int nf = 0; nf < 8; nf++)
#pragma unroll
                for (int c = 0; c < 4; c++) S[nf][c] = 0.f;
#pragma unroll
            for (int ks = 0; ks < KS; ks++) {
                const uint4 xh = nah, xl = nal;
                if (ks + 1 < KS) fetchA(b, ks + 1, rg);
                else if (b + 1 < b1) fetchA(b + 1, 0, rg);
                else if (more) fetchA(0, 0, rgn);
                const unsigned ah[4] = {xh.x, xh.y, xh.z, xh.w}, al[4] = {xl.x, xl.y, xl.z, xl.w};
                if (GRADS) {                                   // dB's A' operand: the warp's hi fragments transposed
#pragma unroll
                    for (int r = 0; r < 2; r++)
                        reinterpret_cast<uint2*>(sA)[(w * NT + 2 * ks + r) * 32 + lane] =
                            make_uint2(movt(ah[2 * r]), movt(ah[2 * r + 1]));
                }
#pragma unroll
                for (int jl = 0; jl < 2; jl++) {
                    const uint4* bp = reinterpret_cast<const uint4*>(sbh + (jl * KS + ks) * 256) + lane;
                    const uint4* lp = reinterpret_cast<const uint4*>(sbl + (jl * KS + ks) * 256) + lane;
                    const uint4 h0 = bp[0], h1 = bp[32], l0 = lp[0], l1 = lp[32];
                    const unsigned bh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                    const unsigned bl[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        mma_bf16_k16(S[4 * jl + q], al[0], al[1], al[2], al[3], bh[2 * q], bh[2 * q + 1]);
                        mma_bf16_k16(S[4 * jl + q], ah[0], ah[1], ah[2], ah[3], bl[2 * q], bl[2 * q + 1]);
                        mma_bf16_k16(S[4 * jl + q], ah[0], ah[1], ah[2], ah[3], bh[2 * q], bh[2 * q + 1]);
                    }
                }
            }
            if (b == b0) {                                     // (theta's loads under the first phi)
#pragma unroll
                for (int e = 0; e < 32; e++) lt[e] = lt[e] > 0.f ? __log2f(lt[e]) : NEG;
            }
            // ---- the element terms: q = 2^(lw' + lu' + lc'), t = 2^(lw' + lv')
            unsigned R[8][2];                                  // (q - t) as bf16 pairs: (n-tile, node half)
            float v[16];                                       // q over the lane's two rows, its children 2 nf + p
#pragma unroll
            for (int nf = 0; nf < 8; nf++) {
                float gq[4], qq[4];
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const float lw = S[nf][c] + lt[4 * nf + c];
                    const float q = exp2f(lw + (c & 2 ? tB.x : tA.x) + lcv[2 * nf + (c & 1)]);
                    const float tt = exp2f(lw + (c & 2 ? tB.y : tA.y));
                    if (FLOWS) fp[4 * nf + c] += q;
                    if (DENOM) tw[4 * nf + c] += tt;
                    qq[c] = q;
                    gq[c] = q - tt;
                }
                v[2 * nf] = qq[0] + qq[2];
                v[2 * nf + 1] = qq[1] + qq[3];
                R[nf][0] = pack_bf16(gq[0], gq[1]);
                R[nf][1] = pack_bf16(gq[2], gq[3]);
            }
            // ---- the element flows: each child's sum over the warp's 16 nodes, a reduce-scatter over the 8 lanes of a tq
            {
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
                // ---- dA += G B^T over the tile's 64 children: the A operand (k16 over children 16 kk ..) R[2 kk][0],
                // R[2 kk][1], R[2 kk + 1][0], R[2 kk + 1][1]; B^T the staged hi fragments transposed
                float* const pw = sP + w * 16 * (PW + 4);
                // all NT rank tiles at once (independent accumulators); k16 over children 16 kk .. -- B^T from the
                // staged hi fragments of block (kk / 2, ks), half kk & 1: words r (rank tile 2 ks + r: b0) and 2 + r (b1)
                float dd[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                    for (int c = 0; c < 4; c++) dd[nt][c] = 0.f;
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        const uint4 hw = reinterpret_cast<const uint4*>(sbh + ((kk >> 1) * KS + ks) * 256)[(kk & 1) * 32 + lane];
                        const unsigned m0 = movt(hw.x), m1 = movt(hw.y), m2 = movt(hw.z), m3 = movt(hw.w);
                        mma_bf16_k16(dd[2 * ks], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], m0, m2);
                        mma_bf16_k16(dd[2 * ks + 1], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], m1, m3);
                    }
                }
#pragma unroll
                for (int f0 = 0; f0 < NT; f0 += PW / 8) {       // the rows' ranks 8 f0 .. 8 f0 + PW: whole runs a row
#pragma unroll
                    for (int nt = f0; nt < f0 + PW / 8; nt++) {   // (nodes g / g + 8, ranks 8 nt + 2 tq, + 1)
                        const int col = (8 * nt) % PW + 2 * tq;
                        *reinterpret_cast<float2*>(pw + g * (PW + 4) + col) = make_float2(dd[nt][0], dd[nt][1]);
                        *reinterpret_cast<float2*>(pw + (g + 8) * (PW + 4) + col) = make_float2(dd[nt][2], dd[nt][3]);
                    }
                    {
                        __syncwarp();
                        constexpr int LPR = PW / 4, RPI = 32 / LPR;     // lanes a row, rows an instruction
                        const int k0 = 8 * f0, rr = lane / LPR, c4 = 4 * (lane % LPR), k = k0 + c4;
#pragma unroll
                        for (int r0 = 0; r0 < 16; r0 += RPI) {
                            const int row = r0 + rr, i = i0 + 16 * w + row;
                            const float4 val = *reinterpret_cast<const float4*>(pw + row * (PW + 4) + c4);
                            if (i < M && k < K) {
                                float* p = gA + ((long)b * M + i) * K + k;
                                if (k + 3 < K && (reinterpret_cast<unsigned long>(p) & 15) == 0) red4(p, val);
                                else {
                                    red1(p, val.x);
                                    if (k + 1 < K) red1(p + 1, val.y);
                                    if (k + 2 < K) red1(p + 2, val.z);
                                    if (k + 3 < K) red1(p + 3, val.w);
                                }
                            }
                        }
                        __syncwarp();
                    }
                }
                // ---- dB's G^T blocks (child g, nodes 2 tq ..), a child m-tile's four in the A operand's order
#pragma unroll
                for (int mt = 0; mt < 4; mt++)
                    reinterpret_cast<uint4*>(sT)[(w * 4 + mt) * 32 + lane] =
                        make_uint4(movt(R[2 * mt][0]), movt(R[2 * mt + 1][0]), movt(R[2 * mt][1]), movt(R[2 * mt + 1][1]));
            }
            __syncthreads();                                   // [B] the 8 warps' sA, sT, sE of sample b written
            if (tid < TN2) {                                   // the element flows: the 8 warps' sums, one add
                float s = 0.f;
#pragma unroll
                for (int ww = 0; ww < NW2; ww++) s += sE[ww * TN2 + tid];
                if (s != 0.f && j0 + tid < N) red1(ef + (long)b * Np + j0 + tid, s);
            }
            if (GRADS) {
                // ---- dB^T += G^T A' over the tile's 128 nodes (the 8 warps' blocks as k-steps): units (UM m-tiles,
                // rank tile nt) over the warps
                constexpr int MG = 4 / UM, NU = MG * NT;
                constexpr int UPW = NU > 0 ? (NU + NW2 - 1) / NW2 : 1, CHN = UPW * UM, KSPL = CHN >= 4 ? 1 : 4 / CHN;
                static_assert(NU % NW2 == 0 || NU == 0, "dB's units: a whole number a warp");
                float* const qw = sQ + w * 8 * UW;
                float dq[UPW][UM][KSPL][4];
#pragma unroll
                for (int uu = 0; uu < UPW; uu++)
#pragma unroll
                    for (int m = 0; m < UM; m++)
#pragma unroll
                        for (int h = 0; h < KSPL; h++)
#pragma unroll
                            for (int c = 0; c < 4; c++) dq[uu][m][h][c] = 0.f;
                if (NU > 0) {
#pragma unroll
                    for (int kw = 0; kw < NW2; kw++) {
#pragma unroll
                        for (int uu = 0; uu < UPW; uu++) {
                            const int un = w + NW2 * uu, mg = un % MG, nt = un / MG;
                            const uint2 bb = reinterpret_cast<const uint2*>(sA)[(kw * NT + nt) * 32 + lane];
#pragma unroll
                            for (int m = 0; m < UM; m++) {
                                const uint4 aa = reinterpret_cast<const uint4*>(sT)[(kw * 4 + UM * mg + m) * 32 + lane];
                                mma_bf16_k16(dq[uu][m][kw % KSPL], aa.x, aa.y, aa.z, aa.w, bb.x, bb.y);
                            }
                        }
                    }
                }
#pragma unroll
                for (int uu = 0; uu < (NU > 0 ? UPW : 0); uu++) {
                    const int un = w + NW2 * uu, mg = un % MG, nt = un / MG;
                    float d[UM][4];
#pragma unroll
                    for (int m = 0; m < UM; m++)
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            float a = dq[uu][m][0][c];
#pragma unroll
                            for (int h = 1; h < KSPL; h++) a += dq[uu][m][h][c];
                            d[m][c] = a;
                        }
                    // (children 16 mt + g / + 8, ranks 8 nt + 2 tq, + 1; A' carried log2 e) through the warp's rows
#pragma unroll
                    for (int m = 0; m < UM; m++) {
                        qw[(2 * tq) * UW + 16 * m + g] = d[m][0] * LN2;
                        qw[(2 * tq + 1) * UW + 16 * m + g] = d[m][1] * LN2;
                        qw[(2 * tq) * UW + 16 * m + g + 8] = d[m][2] * LN2;
                        qw[(2 * tq + 1) * UW + 16 * m + g + 8] = d[m][3] * LN2;
                    }
                    __syncwarp();
                    constexpr int LPR = 4 * UM, RPI = 32 / LPR;      // lanes a rank's row, rows an instruction
                    const int rr = lane / LPR, c4 = 4 * (lane % LPR), jc4 = j0 + 16 * UM * mg + c4;
#pragma unroll
                    for (int r0 = 0; r0 < 8; r0 += RPI) {
                        const int rk = r0 + rr, k = 8 * nt + rk;
                        const float4 val = *reinterpret_cast<const float4*>(qw + rk * UW + c4);
                        if (k < K) {
                            float* p = gB + ((long)b * K + k) * N + jc4;
                            if (jc4 + 3 < N && (reinterpret_cast<unsigned long>(p) & 15) == 0) red4(p, val);
                            else {
                                if (jc4 < N) red1(p, val.x);
                                if (jc4 + 1 < N) red1(p + 1, val.y);
                                if (jc4 + 2 < N) red1(p + 2, val.z);
                                if (jc4 + 3 < N) red1(p + 3, val.w);
                            }
                        }
                    }
                    __syncwarp();
                }
            }
        }
        // F+ and W of the warp's 16 x 64 cells, half the children at a time through its own shared rows (past the B
        // stages: the next tile's first lands meanwhile): a lane four consecutive nodes of a child, one 16-byte add
        // (scalar adds where the node blocks are under 4 or the runs unaligned)
        if (FLOWS || DENOM) {
            __syncthreads();                                   // (every warp's last reads of the stages done)
            float* const st = reinterpret_cast<float*>(smw + 4 * Gm::SB) + w * (32 * 20);   // [child 32][node 16 + 4]
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
        __syncthreads();                                       // (the stages reused by the run's next tile)
    }
}

// ======================================================================= TF32 products, three terms ("fp32")
template <int KP> struct Fw3Geo {
    static constexpr bool K4 = KP == 4;
    static constexpr int NKS = K4 ? 1 : KP / 8;      // k-steps
    static constexpr int NA = K4 ? 2 : 4;            // A values a lane a k-step
    static constexpr int NB = K4 ? 4 : 8;            // B values a lane a k-step (4 n-tiles)
};

// The TF32 forward's prologue (the caller's factors, A [B, M, K], B [B, K, N]): AF [B, Mp / 16, NKS, 32, NA], BF [B,
// Np / BN, NKS, 32, NB] fp32 in fragment order; WF, lcs as the bf16 path's. A grid row a sample
__global__ void __launch_bounds__(256) tf_fw_prep_kernel(
        const float* __restrict__ A, const float* __restrict__ Bm, float* __restrict__ AF, float* __restrict__ BF, float* __restrict__ WF, float* __restrict__ lcs,
        const float* __restrict__ emars, const long* __restrict__ rows, int M, int N, int Mp, int Np, int K, int lkp,
        int B, int nA, int nB) {
    const int b = blockIdx.y, KP = 1 << lkp;
    const bool k4 = KP == 4;
    const int NA = k4 ? 2 : 4, NB = k4 ? 4 : 8, NKS = k4 ? 1 : KP / 8;
    int blk = blockIdx.x;
    if (blk < nA) {                                            // AF[b][rg][ks][lane][u]
        const int e = blk * 256 + threadIdx.x;
        if (e >= Mp * KP) return;
        const int u = e % NA, lane = (e / NA) & 31, ks = (e / (NA * 32)) % NKS, rg = e / (NA * 32 * NKS);
        const int g = lane >> 2, tq = lane & 3;
        const int i = 16 * rg + g + (u & 1) * 8, k = k4 ? tq : 8 * ks + tq + (u >> 1) * 4;
        AF[(long)b * Mp * KP + e] = (k < K && i < M) ? A[((long)b * M + i) * K + k] * LOG2E : 0.f;
        return;
    }
    blk -= nA;
    if (blk < nB) {                                            // BF[b][jt][ks][lane][u]
        const int e = blk * 256 + threadIdx.x;
        if (e >= Np * KP) return;
        const int u = e % NB, lane = (e / NB) & 31, ks = (e / (NB * 32)) % NKS, jt = e / (NB * 32 * NKS);
        const int g = lane >> 2, tq = lane & 3;
        const int nf = k4 ? u : u >> 1, k = k4 ? tq : 8 * ks + tq + (u & 1) * 4, j = BN * jt + 8 * nf + g;
        BF[(long)b * Np * KP + e] = (k < K && j < N) ? Bm[((long)b * K + k) * N + j] : 0.f;
        return;
    }
    blk -= nB;                                                 // (the column tiles: a warp each, a child a lane)
    const int njt = Np / BN, jt = blk * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (jt >= njt) return;
    const int j = jt * BN + lane;
    const long r = j < N ? rows[j] : -1;
    const float lc = r >= 0 ? emars[r * B + b] * LOG2E : NEG;
    float mc = lc;
#pragma unroll
    for (int d = 16; d; d >>= 1) mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, d));
    lcs[(long)b * Np + j] = lc;
    float* wp = WF + ((long)b * njt + jt) * 36;
    const int nf = lane >> 3, tq = (lane & 7) >> 1, p = lane & 1;     // child lane = 8 nf + 2 tq + p
    wp[tq * 8 + 2 * nf + p] = mc == NEG ? 0.f : exp2f(lc - mc);
    if (lane < 4) wp[32 + lane] = lane == 0 ? mc : 0.f;
}

template <int KP, typename TP>
__global__ void __launch_bounds__(128) tf_fw_kernel(
        const float* __restrict__ AF, const float* __restrict__ BF, const TP* __restrict__ params, long p0, int Nb,
        int cbs, int bs, const float* __restrict__ WF, const float* __restrict__ lcs, float4* __restrict__ part, int M,
        int N, int Mp, int Np, int B, int bper) {
    using Gm = Fw3Geo<KP>;
    constexpr int NKS = Gm::NKS, NA = Gm::NA, NB = Gm::NB;
    constexpr bool K4 = Gm::K4;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int i0 = blockIdx.y * 64, j0 = blockIdx.x * BN, jt = blockIdx.x, njt = gridDim.x;
    const int rg = blockIdx.y * 4 + w, rA = i0 + 16 * w + g, rB = rA + 8;
    const int b0 = blockIdx.z * bper, b1 = min(B, b0 + bper);
    if (b0 >= b1) return;

    float lt[16];
#pragma unroll
    for (int e = 0; e < 16; e++) {
        const int nf = e >> 2, c = e & 3, i = c & 2 ? rB : rA, j = j0 + 8 * nf + 2 * tq + (c & 1);
        lt[e] = (i < M && j < N) ? to_f(params[p0 + grid_at(i, j, Nb, cbs, bs)]) : 0.f;
    }
#pragma unroll
    for (int e = 0; e < 16; e++) lt[e] = lt[e] > 0.f ? __log2f(lt[e]) : NEG;

    // a sample's fragments: A (NKS x NA), B (NKS x NB), the factors (8) and mc
    const float* af = AF + ((long)rg * NKS * 32 + lane) * NA;
    const float* bfp = BF + ((long)jt * NKS * 32 + lane) * NB;
    const float* wf = WF + (long)jt * 36 + tq * 8;
    const long astr = (long)Mp * KP, bstr = (long)Np * KP, wstr = (long)njt * 36;
    float an[NKS][NA], bn[NKS][NB], wn[8], mcn;
    auto fetch = [&](int b) {
#pragma unroll
        for (int ks = 0; ks < NKS; ks++) {
            const float* p = af + b * astr + ks * 32 * NA;
            if constexpr (NA == 4) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(p));
                an[ks][0] = v.x; an[ks][1] = v.y; an[ks][2] = v.z; an[ks][3] = v.w;
            } else {
                const float2 v = __ldg(reinterpret_cast<const float2*>(p));
                an[ks][0] = v.x; an[ks][1] = v.y;
            }
            const float* q = bfp + b * bstr + ks * 32 * NB;
#pragma unroll
            for (int u = 0; u < NB; u += 4) {
                const float4 v = __ldg(reinterpret_cast<const float4*>(q + u));
                bn[ks][u] = v.x; bn[ks][u + 1] = v.y; bn[ks][u + 2] = v.z; bn[ks][u + 3] = v.w;
            }
        }
        const float4 w0 = __ldg(reinterpret_cast<const float4*>(wf + b * wstr));
        const float4 w1 = __ldg(reinterpret_cast<const float4*>(wf + b * wstr + 4));
        wn[0] = w0.x; wn[1] = w0.y; wn[2] = w0.z; wn[3] = w0.w; wn[4] = w1.x; wn[5] = w1.y; wn[6] = w1.z; wn[7] = w1.w;
        mcn = __ldg(WF + b * wstr + (long)jt * 36 + 32);
    };
    fetch(b0);
#pragma unroll 1
    for (int b = b0; b < b1; b++) {
        float ac[NKS][NA], bc[NKS][NB], wv[8];
#pragma unroll
        for (int ks = 0; ks < NKS; ks++) {
#pragma unroll
            for (int u = 0; u < NA; u++) ac[ks][u] = an[ks][u];
#pragma unroll
            for (int u = 0; u < NB; u++) bc[ks][u] = bn[ks][u];
        }
#pragma unroll
        for (int u = 0; u < 8; u++) wv[u] = wn[u];
        const float mc = mcn;
        if (b + 1 < b1) fetch(b + 1);                          // (in flight while this sample computes)

        float S[4][4];
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) S[nf][c] = 0.f;
#pragma unroll
        for (int ks = 0; ks < NKS; ks++) {
            unsigned ah[NA], al[NA];
#pragma unroll
            for (int u = 0; u < NA; u++) tf32_split_fast(ac[ks][u], ah[u], al[u]);
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                if constexpr (K4) {
                    unsigned bh, bl;
                    tf32_split_fast(bc[ks][nf], bh, bl);
                    mma_tf32_k4(S[nf], al[0], al[1], bh);
                    mma_tf32_k4(S[nf], ah[0], ah[1], bl);
                    mma_tf32_k4(S[nf], ah[0], ah[1], bh);
                } else {
                    unsigned bh0, bl0, bh1, bl1;
                    tf32_split_fast(bc[ks][2 * nf], bh0, bl0);
                    tf32_split_fast(bc[ks][2 * nf + 1], bh1, bl1);
                    const unsigned ahh[4] = {ah[0], ah[1], ah[2], ah[3]}, all[4] = {al[0], al[1], al[2], al[3]};
                    mma3(S[nf], ahh, all, bh0, bh1, bl0, bl1);
                }
            }
        }
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) S[nf][c] += lt[4 * nf + c];

        float a[2] = {NEG, NEG};
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) a[c >> 1] = fmaxf(a[c >> 1], S[nf][c]);
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            a[0] = fmaxf(a[0], __shfl_xor_sync(0xffffffffu, a[0], d));
            a[1] = fmaxf(a[1], __shfl_xor_sync(0xffffffffu, a[1], d));
        }
        const float sa0 = a[0] == NEG ? 0.f : a[0], sa1 = a[1] == NEG ? 0.f : a[1];
        float z[2] = {0.f, 0.f}, s[2] = {0.f, 0.f};
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int h = c >> 1;
                const float e = exp2f(S[nf][c] - (h ? sa1 : sa0));
                z[h] += e;
                s[h] = fmaf(e, wv[2 * nf + (c & 1)], s[h]);
            }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            z[0] += __shfl_xor_sync(0xffffffffu, z[0], d);
            z[1] += __shfl_xor_sync(0xffffffffu, z[1], d);
            s[0] += __shfl_xor_sync(0xffffffffu, s[0], d);
            s[1] += __shfl_xor_sync(0xffffffffu, s[1], d);
        }
        float ms[2] = {s[0] > 0.f ? a[0] + mc : NEG, s[1] > 0.f ? a[1] + mc : NEG};
        const bool need = mc > NEG && ((s[0] < 0x1p-97f && a[0] > NEG) || (s[1] < 0x1p-97f && a[1] > NEG));
        if (__any_sync(0xffffffffu, need)) {                   // (exactly: x' = lw' + lc' under its own max)
            float lcv[4][2];
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                const float2 v = *reinterpret_cast<const float2*>(lcs + (long)b * Np + j0 + 8 * nf + 2 * tq);
                lcv[nf][0] = v.x;
                lcv[nf][1] = v.y;
            }
            float m2[2] = {NEG, NEG};
#pragma unroll
            for (int nf = 0; nf < 4; nf++)
#pragma unroll
                for (int c = 0; c < 4; c++) m2[c >> 1] = fmaxf(m2[c >> 1], S[nf][c] + lcv[nf][c & 1]);
#pragma unroll
            for (int d = 1; d <= 2; d <<= 1) {
                m2[0] = fmaxf(m2[0], __shfl_xor_sync(0xffffffffu, m2[0], d));
                m2[1] = fmaxf(m2[1], __shfl_xor_sync(0xffffffffu, m2[1], d));
            }
            const float sc0 = m2[0] == NEG ? 0.f : m2[0], sc1 = m2[1] == NEG ? 0.f : m2[1];
            float s2[2] = {0.f, 0.f};
#pragma unroll
            for (int nf = 0; nf < 4; nf++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int h = c >> 1;
                    s2[h] += exp2f(S[nf][c] + lcv[nf][c & 1] - (h ? sc1 : sc0));
                }
#pragma unroll
            for (int d = 1; d <= 2; d <<= 1) {
                s2[0] += __shfl_xor_sync(0xffffffffu, s2[0], d);
                s2[1] += __shfl_xor_sync(0xffffffffu, s2[1], d);
            }
            ms[0] = s2[0] > 0.f ? m2[0] : NEG;
            ms[1] = s2[1] > 0.f ? m2[1] : NEG;
            s[0] = s2[0];
            s[1] = s2[1];
        }
        if (tq == 0) {
            if (rA < M) part[((long)b * njt + jt) * Mp + rA] = make_float4(a[0], z[0], ms[0], s[0]);
            if (rB < M) part[((long)b * njt + jt) * Mp + rB] = make_float4(a[1], z[1], ms[1], s[1]);
        }
    }
}


constexpr int TNT = 2 * TBM;                     // ... and its threads
template <int KP> struct TcGeo {
    static constexpr int KPS = KP + 8;          // A' rows: the dB B-operand reads conflict-free (t * KPS = 8 t mod 32)
    static constexpr int BNS = BN + 8;          // B rows: the phi B-operand reads conflict-free
    static constexpr int GTS = TBM + 4;         // (q - t)^T rows: the dB A-operand reads conflict-free (4 g + t)
    static constexpr int NT = KP / 8;           // rank n-tiles
    static constexpr int NBUF = KP <= 16 ? 2 : 1;                       // the staged operands' buffers
    static constexpr int STAGE = TBM * KPS + KP * BNS + 2 * TBM + BN;   // a buffer: A', B, lu' lv' lc'
    static constexpr int STAGES = NBUF * STAGE > BN * GTS ? NBUF * STAGE : BN * GTS;   // (+ Gt: the two F+, W tiles)
    static constexpr size_t SMEM = (size_t)(STAGES + BN * GTS + (TBM / 16) * BN) * 4;
};

template <int KP, typename TP, bool FLOWS, bool DENOM, bool GRADS>
__global__ void __launch_bounds__(TNT) tf_bw_kernel(
        const float* __restrict__ At, const float* __restrict__ Bt, const TP* __restrict__ params, long p0, int Nb,
        int cbs, int bs, const float* __restrict__ terms, float* __restrict__ ef, float* __restrict__ pflows, long pf0,
        float* __restrict__ denom, long dn0, float* __restrict__ dAt, float* __restrict__ dBt, int M, int N, int Mp,
        int Np, int K, int B, int bper, int split) {
    using Gm = TcGeo<KP>;
    constexpr int KPS = Gm::KPS, BNS = Gm::BNS, GTS = Gm::GTS, NTR = Gm::NT, NBUF = Gm::NBUF, STAGE = Gm::STAGE;
    extern __shared__ __align__(16) float smem[];
    float* Gt = smem + Gm::STAGES;              // [BN][GTS]   (q - t)^T
    float* ef_s = Gt + BN * GTS;                // [TBM / 16][BN]  the warps' element-flow sums
    // a buffer: As [BM][KPS] (A' of the sample's nodes), Bs [KP][BNS] (B of the tile's children), lu' lv' [BM], lc' [BN]
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int i0 = blockIdx.y * TBM, j0 = blockIdx.x * BN, rw0 = 16 * w;
    const int b0 = blockIdx.z * bper, b1 = min(B, b0 + bper);
    const int rA = i0 + rw0 + g, rB = rA + 8;   // this thread's two nodes
    const long tper = 2L * Mp + Np;
    // a sample's operands into buffer `buf`: 16-byte asynchronous copies of contiguous runs (everything padded)
    auto issue = [&](int b, int buf) {
        float* st = smem + buf * STAGE;
        for (int e = tid; e < TBM * KP / 4; e += TNT) {
            const int rr = e / (KP / 4), c4 = (e % (KP / 4)) * 4;
            cp_async16(st + rr * KPS + c4, At + ((long)b * Mp + i0 + rr) * KP + c4);
        }
        for (int e = tid; e < KP * BN / 4; e += TNT) {
            const int kk = e / (BN / 4), c4 = (e % (BN / 4)) * 4;
            cp_async16(st + TBM * KPS + kk * BNS + c4, Bt + ((long)b * KP + kk) * Np + j0 + c4);
        }
        const float* tb = terms + b * tper;
        float* ts = st + TBM * KPS + KP * BNS;
        for (int e = tid; e < (2 * TBM + BN) / 4; e += TNT) {
            const int o = e * 4;
            const float* src = o < TBM ? tb + i0 + o
                                       : (o < 2 * TBM ? tb + Mp + i0 + (o - TBM) : tb + 2 * Mp + j0 + (o - 2 * TBM));
            cp_async16(ts + o, src);
        }
        cp_async_commit();
    };

    // log2 theta of the thread's 16 accumulators: (nf, c) at node rA + 8 (c >> 1), child j0 + 8 nf + 2 tq + (c & 1)
    float lt[16], fp[16], tw[16];
#pragma unroll
    for (int e = 0; e < 16; e++) {
        const int nf = e >> 2, c = e & 3, i = c & 2 ? rB : rA, j = j0 + 8 * nf + 2 * tq + (c & 1);
        lt[e] = (i < M && j < N) ? to_f(params[p0 + grid_at(i, j, Nb, cbs, bs)]) : 0.f;
        fp[e] = 0.f;
        tw[e] = 0.f;
    }
#pragma unroll
    for (int e = 0; e < 16; e++) lt[e] = lt[e] > 0.f ? __log2f(lt[e]) : NEG;

    if (NBUF == 2 && b0 < b1) issue(b0, 0);
    for (int b = b0; b < b1; b++) {
        const int buf = NBUF == 2 ? (b - b0) & 1 : 0;
        if (NBUF == 1) issue(b, 0);
        cp_async_wait<0>();
        __syncthreads();                                       // (the sample landed; every warp done with the last)
        if (NBUF == 2 && b + 1 < b1) issue(b + 1, buf ^ 1);    // (in flight while this sample computes)
        const float* As = smem + buf * STAGE;
        const float* Bs = As + TBM * KPS;
        const float* lu_s = Bs + KP * BNS;
        const float* lv_s = lu_s + TBM;
        const float* lc_s = lv_s + TBM;

        // ---- phi' = A' B: the warp's 16 nodes by the 32 children, K in steps of 8
        float S[4][4];
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++) S[nf][c] = 0.f;
#pragma unroll
        for (int ks = 0; ks < KP / 8; ks++) {
            unsigned ah[4], al[4];
            const float* a0 = As + (rw0 + g) * KPS + 8 * ks + tq;
            tf32_split_fast(a0[0], ah[0], al[0]);
            tf32_split_fast(a0[8 * KPS], ah[1], al[1]);
            tf32_split_fast(a0[4], ah[2], al[2]);
            tf32_split_fast(a0[8 * KPS + 4], ah[3], al[3]);
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                unsigned bh0, bl0, bh1, bl1;
                const float* bp = Bs + (8 * ks + tq) * BNS + 8 * nf + g;
                tf32_split_fast(bp[0], bh0, bl0);
                tf32_split_fast(bp[4 * BNS], bh1, bl1);
                mma3(S[nf], ah, al, bh0, bh1, bl0, bl1);
            }
        }

        // ---- the element terms
        const float luA = lu_s[rw0 + g], luB = lu_s[rw0 + g + 8], lvA = lv_s[rw0 + g], lvB = lv_s[rw0 + g + 8];
        float G[4][4], Q[4][4];
#pragma unroll
        for (int nf = 0; nf < 4; nf++) {
            const float lc0 = lc_s[8 * nf + 2 * tq], lc1 = lc_s[8 * nf + 2 * tq + 1];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const float lw = S[nf][c] + lt[4 * nf + c];
                const float q = exp2f(lw + (c & 2 ? luB : luA) + (c & 1 ? lc1 : lc0));
                const float t = exp2f(lw + (c & 2 ? lvB : lvA));
                if (FLOWS) fp[4 * nf + c] += q;
                if (DENOM) tw[4 * nf + c] += t;
                Q[nf][c] = q;
                G[nf][c] = q - t;
            }
        }

        // ---- dA += G B^T: G as the A operand (k slot tq <-> child 2 tq, slot tq + 4 <-> child 2 tq + 1)
        if (GRADS) {
            constexpr int TA = NTR >= 4 ? 1 : 3;              // (accumulators a rank tile: a term each, or one)
            float dA3[NTR][TA][4];
#pragma unroll
            for (int nt = 0; nt < NTR; nt++)
#pragma unroll
                for (int u = 0; u < TA; u++)
#pragma unroll
                    for (int c = 0; c < 4; c++) dA3[nt][u][c] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                unsigned ah[4], al[4];
                tf32_split_fast(G[kk][0], ah[0], al[0]);            // (node g, child 2 tq)
                tf32_split_fast(G[kk][2], ah[1], al[1]);            // (node g + 8, child 2 tq)
                tf32_split_fast(G[kk][1], ah[2], al[2]);            // (node g, child 2 tq + 1)
                tf32_split_fast(G[kk][3], ah[3], al[3]);            // (node g + 8, child 2 tq + 1)
#pragma unroll
                for (int nt = 0; nt < NTR; nt++) {
                    const float2 bv = *reinterpret_cast<const float2*>(Bs + (8 * nt + g) * BNS + 8 * kk + 2 * tq);
                    unsigned bh0, bl0, bh1, bl1;
                    tf32_split_fast(bv.x, bh0, bl0);
                    tf32_split_fast(bv.y, bh1, bl1);
                    if constexpr (TA == 3) {
                        mma_tf32(dA3[nt][0], al[0], al[1], al[2], al[3], bh0, bh1);
                        mma_tf32(dA3[nt][1], ah[0], ah[1], ah[2], ah[3], bl0, bl1);
                        mma_tf32(dA3[nt][2], ah[0], ah[1], ah[2], ah[3], bh0, bh1);
                    } else
                        mma3(dA3[nt][0], ah, al, bh0, bh1, bl0, bl1);
                }
            }
            float dA[NTR][4];
#pragma unroll
            for (int nt = 0; nt < NTR; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++)
                    dA[nt][c] = TA == 3 ? (dA3[nt][0][c] + dA3[nt][1][c]) + dA3[nt][TA - 1][c] : dA3[nt][0][c];
            // dA[nt]: (node g / g + 8, rank 8 nt + 2 tq / + 1)
#pragma unroll
            for (int nt = 0; nt < NTR; nt++) {
                const int k = 8 * nt + 2 * tq;
                if (k >= K) continue;
                if (rA < M) {
                    float* p = dAt + ((long)b * Mp + rA) * KP + k;
                    red2(p, dA[nt][0], dA[nt][1]);
                }
                if (rB < M) {
                    float* p = dAt + ((long)b * Mp + rB) * KP + k;
                    red2(p, dA[nt][2], dA[nt][3]);
                }
            }
        }

        // ---- the element flows: each column's sum over the warp's 16 nodes, a reduce-scatter over the 8 lanes of
        // a tq (lane bits 2..4): v[m], m = 2 nf + parity, ends as one column a lane
        {
            float v[8];
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                v[2 * nf] = Q[nf][0] + Q[nf][2];
                v[2 * nf + 1] = Q[nf][1] + Q[nf][3];
            }
            // round 1 (xor 16: g bit 2): keep m in [0, 4) if bit clear, [4, 8) if set
            const bool h2 = g & 4, h1 = g & 2, h0 = g & 1;
            float u4[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const float send = h2 ? v[m] : v[m + 4], keep = h2 ? v[m + 4] : v[m];
                u4[m] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
            }
            float u2[2];
#pragma unroll
            for (int m = 0; m < 2; m++) {
                const float send = h1 ? u4[m] : u4[m + 2], keep = h1 ? u4[m + 2] : u4[m];
                u2[m] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
            }
            const float send = h0 ? u2[0] : u2[1], keep = h0 ? u2[1] : u2[0];
            const float tot = keep + __shfl_xor_sync(0xffffffffu, send, 4);
            const int m = (h2 ? 4 : 0) + (h1 ? 2 : 0) + (h0 ? 1 : 0);   // the m this lane holds
            ef_s[w * BN + 8 * (m >> 1) + 2 * tq + (m & 1)] = tot;
        }

        // ---- (q - t)^T staged for dB
#pragma unroll
        for (int nf = 0; nf < 4; nf++)
#pragma unroll
            for (int c = 0; c < 4; c++)
                Gt[(8 * nf + 2 * tq + (c & 1)) * GTS + rw0 + g + 8 * (c >> 1)] = G[nf][c];
        __syncthreads();

        // the element flows: the 8 warps' sums, one atomic a child
        if (tid < BN) {
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < TBM / 16; ww++) s += ef_s[ww * BN + tid];
            if (j0 + tid < N && s != 0.f) atomicAdd(ef + (long)b * Np + j0 + tid, s);
        }
        // ---- dB'^T[child, rank] = (q - t)^T A': units (m-tile, rank n-tile, node quarter) over the warps
        if (GRADS) {
            static_assert(TBM == 64, "dB: four warps, two child m-tiles by two node halves");
            constexpr int TB = NTR >= 4 ? 1 : 3;
            const int mt = w & 1, kq = w >> 1;
            float d3[NTR][TB][4];
#pragma unroll
            for (int nt = 0; nt < NTR; nt++)
#pragma unroll
                for (int u = 0; u < TB; u++)
#pragma unroll
                    for (int c = 0; c < 4; c++) d3[nt][u][c] = 0.f;
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                const int r0 = 32 * kq + 8 * ks + tq;
                unsigned ah[4], al[4];
                const float* gp = Gt + (16 * mt + g) * GTS + r0;
                tf32_split_fast(gp[0], ah[0], al[0]);
                tf32_split_fast(gp[8 * GTS], ah[1], al[1]);
                tf32_split_fast(gp[4], ah[2], al[2]);
                tf32_split_fast(gp[8 * GTS + 4], ah[3], al[3]);
#pragma unroll
                for (int nt = 0; nt < NTR; nt++) {
                    unsigned bh0, bl0, bh1, bl1;
                    tf32_split_fast(As[r0 * KPS + 8 * nt + g], bh0, bl0);
                    tf32_split_fast(As[(r0 + 4) * KPS + 8 * nt + g], bh1, bl1);
                    if constexpr (TB == 3) {
                        mma_tf32(d3[nt][0], al[0], al[1], al[2], al[3], bh0, bh1);
                        mma_tf32(d3[nt][1], ah[0], ah[1], ah[2], ah[3], bl0, bl1);
                        mma_tf32(d3[nt][2], ah[0], ah[1], ah[2], ah[3], bh0, bh1);
                    } else
                        mma3(d3[nt][0], ah, al, bh0, bh1, bl0, bl1);
                }
            }
            // d: (child 16 mt + g / + 8, rank 8 nt + 2 tq / + 1)
            const int ja = j0 + 16 * mt + g, jb = ja + 8;
#pragma unroll
            for (int nt = 0; nt < NTR; nt++) {
                float d[4];
#pragma unroll
                for (int c = 0; c < 4; c++) d[c] = TB == 3 ? (d3[nt][0][c] + d3[nt][1][c]) + d3[nt][TB - 1][c] : d3[nt][0][c];
                const int k = 8 * nt + 2 * tq;
#pragma unroll
                for (int v = 0; v < 2; v++) {
                    if (k + v >= K) continue;
                    float* row = dBt + ((long)b * KP + k + v) * Np;
                    if (ja < N) atomicAdd(row + ja, d[v]);
                    if (jb < N) atomicAdd(row + jb, d[2 + v]);
                }
            }
        }
        if (NBUF == 1) __syncthreads();                        // (single buffer: read before the next sample's copies)
    }

    if (FLOWS || DENOM) {
        // F+ and W of the tile through shared memory (the staging buffers free now), then read-added-written along
        // the nodes -- a column's nodes contiguous in the parameters' block-major layout: 16-byte accesses, full
        // sectors (the accumulators' own layout puts 4 bytes a lane in four 32-byte runs)
        __syncthreads();                                       // (the loop's last reads of the shared buffers done)
        float* Tp = smem;                                      // [BN][GTS]: F+ of the tile, column by column
        float* Tw = smem + BN * GTS;                           // [BN][GTS]: W
#pragma unroll
        for (int e = 0; e < 16; e++) {
            const int nf = e >> 2, c = e & 3, col = 8 * nf + 2 * tq + (c & 1), row = rw0 + g + 8 * (c >> 1);
            if (FLOWS) Tp[col * GTS + row] = fp[e];
            if (DENOM) {
                const float th = exp2f(lt[e]);                 // (theta, as the kernel read it)
                Tw[col * GTS + row] = th > 0.f ? tw[e] / th : 0.f;
            }
        }
        __syncthreads();
        const bool vec = (bs & 3) == 0 && ((pf0 | dn0) & 3) == 0 && !split;
        for (int e = tid; e < BN * TBM / 4; e += TNT) {
            const int col = e / (TBM / 4), r4 = (e % (TBM / 4)) * 4, i = i0 + r4, j = j0 + col;
            if (j >= N || i >= M) continue;
            const float4 vp = *reinterpret_cast<const float4*>(Tp + col * GTS + r4);
            const float4 vw = *reinterpret_cast<const float4*>(Tw + col * GTS + r4);
            if (vec && i + 3 < M) {                            // (four nodes of one node block: bs a multiple of 4)
                const long o = grid_at(i, j, Nb, cbs, bs);
                if (FLOWS) {
                    float4* p = reinterpret_cast<float4*>(pflows + pf0 + o);
                    float4 v = *p;
                    v.x += vp.x; v.y += vp.y; v.z += vp.z; v.w += vp.w;
                    *p = v;
                }
                if (DENOM) {
                    float4* p = reinterpret_cast<float4*>(denom + dn0 + o);
                    float4 v = *p;
                    v.x += vw.x; v.y += vw.y; v.z += vw.z; v.w += vw.w;
                    *p = v;
                }
            } else {
                const float fpv[4] = {vp.x, vp.y, vp.z, vp.w}, wv[4] = {vw.x, vw.y, vw.z, vw.w};
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    if (i + u >= M) continue;
                    const long o = grid_at(i + u, j, Nb, cbs, bs);
                    if (FLOWS) {
                        if (split) atomicAdd(pflows + pf0 + o, fpv[u]);
                        else pflows[pf0 + o] += fpv[u];
                    }
                    if (DENOM) {
                        if (split) atomicAdd(denom + dn0 + o, wv[u]);
                        else denom[dn0 + o] += wv[u];
                    }
                }
            }
        }
    }
}

// One backward launch: the tensor-core kernel (its shared memory set past the static 48 KB where it needs more)
template <int KP, typename TP, bool FL, bool DN, bool GR>
void launch_tf_bw(dim3 grid, cudaStream_t st, const float* At, const float* Bt, const TP* params, long p0, int Nb, int cbs,
                int bs, const float* terms, float* ef, float* pflows, long pf0, float* denom, long dn0, float* dAt,
                float* dBt, int M, int N, int Mp, int Np, int K, int B, int bper, int split) {
    if constexpr (KP % 8 == 0) {
        constexpr size_t SM = TcGeo<KP>::SMEM;
        static bool set = false;
        if (!set && SM > 48 * 1024) {
            cudaFuncSetAttribute(tf_bw_kernel<KP, TP, FL, DN, GR>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SM);
            set = true;
        }
        tf_bw_kernel<KP, TP, FL, DN, GR><<<grid, TNT, SM, st>>>(At, Bt, params, p0, Nb, cbs, bs, terms, ef, pflows, pf0,
                                                                denom, dn0, dAt, dBt, M, N, Mp, Np, K, B, bper, split);
    }
}

// The backward's prologue, one launch, a sample a grid row (blockIdx.y): padded to whole tiles (Mp nodes, Np children,
// KP ranks; zeros and -inf past the real ones) -- A' = log2 e A [B, Mp, KP], B [B, KP, Np], the nodes' terms
// lu' = log2 (f / (n Z)) and lv' = log2 (f / Z) [B, Mp] (-inf for a node without flow), the children's lc' = log2 c
// [B, Np] -- and (row 0) the scratch the main kernel adds into zeroed. nA, nB, nT: the blocks a sample's A', B, terms take
__global__ void __launch_bounds__(256) tf_bw_prep_kernel(
        const float* __restrict__ A, const float* __restrict__ Bm, float* __restrict__ At, float* __restrict__ Bt,
        float* __restrict__ terms, float* __restrict__ zero, int grads, const float* __restrict__ nflows,
        const float* __restrict__ nmars, long nid0, const float* __restrict__ lzs, const float* __restrict__ emars,
        const long* __restrict__ rows, int M, int N, int Mp, int Np, int K, int lkp, int B, int nA, int nB, int nT) {
    const int b = blockIdx.y, KP = 1 << lkp;
    int blk = blockIdx.x;
    if (blk < nA) {                                            // A'[b, i, k] = log2 e A[b, i, k]
        const int e = blk * 256 + threadIdx.x;
        if (e >= Mp * KP) return;
        const int i = e >> lkp, k = e & (KP - 1);
        At[(long)b * Mp * KP + e] = (k < K && i < M) ? A[((long)b * M + i) * K + k] * LOG2E : 0.f;
        return;
    }
    blk -= nA;
    if (blk < nB) {                                            // B[b, k, j], padded: a block within one k
        const int per = (Np + 255) / 256, k = blk / per, j = (blk - k * per) * 256 + threadIdx.x;
        if (j >= Np) return;
        Bt[((long)b * KP + k) * Np + j] = (k < K && j < N) ? Bm[((long)b * K + k) * N + j] : 0.f;
        return;
    }
    blk -= nB;
    if (blk < nT) {                                            // terms[b] = [lu' (Mp) | lv' (Mp) | lc' (Np)]
        const int o = blk * 256 + threadIdx.x;
        if (o >= 2 * Mp + Np) return;
        float v = NEG;
        if (o < 2 * Mp) {
            const int i = o < Mp ? o : o - Mp;
            if (i < M) {
                const float lf = nflows[(nid0 + i) * B + b];
                if (isfinite(lf)) {
                    const float lz = lzs[(long)i * B + b];
                    v = (o < Mp ? lf - (nmars[(nid0 + i) * B + b] + lz) : lf - lz) * LOG2E;
                }
            }
        } else {
            const int j = o - 2 * Mp;
            const long r = j < N ? rows[j] : -1;
            if (r >= 0) v = emars[r * B + b] * LOG2E;
        }
        terms[(long)b * (2 * Mp + Np) + o] = v;
        return;
    }
    blk -= nT;                                                 // (the rest: this sample's slices of the scratch --
    const long per = (long)Np + (grads ? (long)Mp * KP + (long)KP * Np : 0);    // ef[b], dAt[b], dBt[b] -- zeroed)
    for (long e = (long)blk * 256 + threadIdx.x; e < per; e += (long)(gridDim.x - nA - nB - nT) * 256) {
        long at;
        if (e < Np) at = (long)b * Np + e;
        else if (e < Np + (long)Mp * KP) at = (long)B * Np + (long)b * Mp * KP + (e - Np);
        else at = (long)B * Np + (long)B * Mp * KP + (long)b * KP * Np + (e - Np - (long)Mp * KP);
        zero[at] = 0.f;
    }
}


// The TF32 backward's epilogue: the element flows to the log, log-added into the circuit's (a thread a (child,
// sample), the sample fastest: lb = log2 of B rounded up); the scratch gradients added into the caller-layout ones --
// gA [B, M, K] from dAt [B, Mp, KP], gB [B, K, N] from dBt [B, KP, Np] (A' carried log2 e) -- a thread an element
__global__ void __launch_bounds__(256) tf_bw_epi_kernel(
        const float* __restrict__ ef, const long* __restrict__ rows, float* __restrict__ eflows,
        const float* __restrict__ dAt, const float* __restrict__ dBt, float* __restrict__ gA, float* __restrict__ gB,
        int M, int N, int Mp, int Np, int K, int KP, int B, int lb, int grads) {
    long t = (long)blockIdx.x * 256 + threadIdx.x;
    const long nE = (long)N << lb;
    if (t < nE) {
        const int b = (int)(t & ((1 << lb) - 1));
        const long j = t >> lb;
        if (b >= B) return;
        const long er = rows[j];
        const float v = ef[(long)b * Np + j];
        if (er < 0 || !(v > 0.f)) return;
        const float lv = logf(v);
        float* p = eflows + er * B + b;
        const float o = *p;
        *p = o == NEG ? lv : fmaxf(o, lv) + log1pf(__expf(-fabsf(o - lv)));
        return;
    }
    if (!grads) return;
    t -= nE;
    const long nAo = (long)B * M * K;
    if (t < nAo) {                                             // gA[b, i, k]
        const int k = (int)(t % K), i = (int)((t / K) % M), b = (int)(t / ((long)M * K));
        gA[t] += dAt[((long)b * Mp + i) * KP + k];
        return;
    }
    t -= nAo;
    if (t < (long)B * K * N) {                                 // gB[b, k, j]
        const int j = (int)(t % N), k = (int)((t / N) % K), b = (int)(t / ((long)K * N));
        gB[t] += dBt[((long)b * KP + k) * Np + j] * LN2;
    }
}

// ===================================================================================================== the host

int grid_z(int ctas, int B) {
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int z = (2 * sms + ctas - 1) / ctas;
    return std::max(1, std::min(z, B));
}

void check_common(const torch::Tensor& A, const torch::Tensor& Bm, const torch::Tensor& params,
                  const torch::Tensor& emars, const torch::Tensor& rows, int64_t M, int64_t N, int64_t K, int64_t B) {
    TORCH_CHECK(A.is_cuda() && A.scalar_type() == torch::kFloat32 && A.is_contiguous(), "A: contiguous fp32 [B, M, K]");
    TORCH_CHECK(Bm.is_cuda() && Bm.scalar_type() == torch::kFloat32 && Bm.is_contiguous(), "B: contiguous fp32 [B, K, N]");
    TORCH_CHECK(A.dim() == 3 && A.size(0) == B && A.size(1) == M && A.size(2) == K, "A's shape");
    TORCH_CHECK(Bm.dim() == 3 && Bm.size(0) == B && Bm.size(1) == K && Bm.size(2) == N, "B's shape");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32 || params.scalar_type() == torch::kBFloat16, "params: fp32 or bf16");
    TORCH_CHECK(emars.scalar_type() == torch::kFloat32 && emars.is_contiguous() && emars.size(1) == B, "element_mars");
    TORCH_CHECK(rows.scalar_type() == torch::kInt64 && rows.numel() == N, "rows: int64 [N]");
    TORCH_CHECK(K >= 1 && K <= 64, "rank 1..64");
}

int rank_pad(int64_t K, int least) {                          // the MMAs' ranks: 4 (or 8), 8, 16, 32, 64
    const int r = K <= 4 ? 4 : (K <= 8 ? 8 : (K <= 16 ? 16 : (K <= 32 ? 32 : 64)));
    return std::max(r, least);
}

#define EXP_SUM_KP_SWITCH(KPV, ...)                                    \
    switch (KPV) {                                                     \
        case 4: { constexpr int KP = 4; __VA_ARGS__; break; }          \
        case 8: { constexpr int KP = 8; __VA_ARGS__; break; }          \
        case 16: { constexpr int KP = 16; __VA_ARGS__; break; }        \
        case 32: { constexpr int KP = 32; __VA_ARGS__; break; }        \
        default: { constexpr int KP = 64; __VA_ARGS__; break; }        \
    }
#define EXP_SUM_TP_SWITCH(PARAMS, ...)                                                                             \
    if (PARAMS.scalar_type() == torch::kFloat32) {                                                                 \
        using TP = float;                                                                                          \
        const TP* pp = PARAMS.data_ptr<float>();                                                                   \
        __VA_ARGS__;                                                                                               \
    } else {                                                                                                       \
        using TP = __nv_bfloat16;                                                                                  \
        const TP* pp = reinterpret_cast<const TP*>(PARAMS.data_ptr<at::BFloat16>());                              \
        __VA_ARGS__;                                                                                               \
    }
#define EXP_SUM_FLAGS_SWITCH(FLOWS, DN, GRADS, ...)                                                                \
    if (FLOWS && DN && GRADS) { constexpr bool FL = true, DE = true, GR = true; __VA_ARGS__; }                     \
    else if (FLOWS && DN) { constexpr bool FL = true, DE = true, GR = false; __VA_ARGS__; }                        \
    else if (FLOWS && GRADS) { constexpr bool FL = true, DE = false, GR = true; __VA_ARGS__; }                     \
    else if (FLOWS) { constexpr bool FL = true, DE = false, GR = false; __VA_ARGS__; }                             \
    else if (DN && GRADS) { constexpr bool FL = false, DE = true, GR = true; __VA_ARGS__; }                        \
    else if (DN) { constexpr bool FL = false, DE = true, GR = false; __VA_ARGS__; }                                \
    else if (GRADS) { constexpr bool FL = false, DE = false, GR = true; __VA_ARGS__; }                             \
    else { constexpr bool FL = false, DE = false, GR = false; __VA_ARGS__; }

// The forward of one node vector: its node values into `node_mars` rows `nid0..nid0 + M`; returns log Z [M, B]
torch::Tensor exp_sum_forward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb,
                              int64_t cbs, int64_t bs, torch::Tensor emars, torch::Tensor rows, torch::Tensor node_mars,
                              int64_t nid0, int64_t M, int64_t N, int64_t K, bool bf16) {
    const int64_t B = node_mars.size(1);
    check_common(A, Bm, params, emars, rows, M, N, K, B);
    TORCH_CHECK(node_mars.scalar_type() == torch::kFloat32 && node_mars.is_contiguous(), "node_mars");
    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int Mp = (int)((M + TBM - 1) / TBM * TBM), Np = (int)((N + BN - 1) / BN * BN), njt = Np / BN, nit = Mp / TBM;
    auto WF = torch::empty({B, njt, 36}, A.options());
    auto lcs = torch::empty({B, Np}, A.options());
    auto part = torch::empty({B, njt, Mp, 4}, A.options());
    auto lzs = torch::empty({M, B}, A.options());
    const int z = grid_z(njt * nit, (int)B), bper = (int)((B + z - 1) / z);
    const dim3 grid(njt, nit, (unsigned)((B + bper - 1) / bper));
    float4* pp4 = reinterpret_cast<float4*>(part.data_ptr<float>());
    const long* rw = reinterpret_cast<const long*>(rows.data_ptr<int64_t>());
    const int nbW = (njt + 7) / 8;
    if (bf16) {
        const int KPb = rank_pad(K, 8);
        const long na = (long)Mp * KPb / 2, nb = (long)Np * KPb / 2;
        auto AF = torch::empty({2, B, na}, A.options().dtype(torch::kInt32));
        auto BF = torch::empty({2, B, nb}, A.options().dtype(torch::kInt32));
        unsigned* afh = reinterpret_cast<unsigned*>(AF.data_ptr<int>());
        unsigned* bfh = reinterpret_cast<unsigned*>(BF.data_ptr<int>());
        const int nbA = (int)((na + 255) / 256), nbB = (int)((nb + 255) / 256);
        bf_fw_prep_kernel<<<dim3(nbA + nbB + nbW, (unsigned)B), 256, 0, stream>>>(
            A.data_ptr<float>(), Bm.data_ptr<float>(), afh, afh + B * na, bfh, bfh + B * nb, WF.data_ptr<float>(),
            lcs.data_ptr<float>(), emars.data_ptr<float>(), rw, (int)M, (int)N, Mp, Np, (int)K, KPb, (int)B, nbA, nbB);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        EXP_SUM_KP_SWITCH(KPb, if constexpr (KP >= 8) { EXP_SUM_TP_SWITCH(params,
            bf_fw_kernel<KP, TP><<<grid, 128, 0, stream>>>(afh, afh + B * na, bfh, bfh + B * nb, pp, p0, (int)Nb,
                                                            (int)cbs, (int)bs, WF.data_ptr<float>(), lcs.data_ptr<float>(),
                                                            pp4, (int)M, (int)N, Mp, Np, (int)B, bper)) })
    } else {
        const int KPf = rank_pad(K, 4), lkp = __builtin_ctz((unsigned)KPf);
        auto AF = torch::empty({B, (long)Mp * KPf}, A.options());
        auto BF = torch::empty({B, (long)Np * KPf}, A.options());
        const int nA = (int)(((long)Mp * KPf + 255) / 256), nB = (int)(((long)Np * KPf + 255) / 256);
        tf_fw_prep_kernel<<<dim3(nA + nB + nbW, (unsigned)B), 256, 0, stream>>>(
            A.data_ptr<float>(), Bm.data_ptr<float>(), AF.data_ptr<float>(), BF.data_ptr<float>(), WF.data_ptr<float>(),
            lcs.data_ptr<float>(), emars.data_ptr<float>(), rw, (int)M, (int)N, Mp, Np, (int)K, lkp, (int)B, nA, nB);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        EXP_SUM_KP_SWITCH(KPf, EXP_SUM_TP_SWITCH(params,
            tf_fw_kernel<KP, TP><<<grid, 128, 0, stream>>>(AF.data_ptr<float>(), BF.data_ptr<float>(), pp, p0, (int)Nb,
                                                            (int)cbs, (int)bs, WF.data_ptr<float>(), lcs.data_ptr<float>(),
                                                            pp4, (int)M, (int)N, Mp, Np, (int)B, bper)))
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    fw_comb_kernel<<<(unsigned)((4 * B * M + 127) / 128), 128, 0, stream>>>(pp4, njt, node_mars.data_ptr<float>(), nid0,
                                                                         lzs.data_ptr<float>(), (int)M, Mp, (int)B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return lzs;
}

// The bf16 backward on big tiles: its shared memory (set once a device), its persistent grid -- the GPU's SMs times the
// CTAs an SM holds (asked once a device), at most a CTA a unit, at most `cap` CTAs where `cap` > 0
template <int KP, typename TP, bool FL, bool DE, bool GR>
void launch_bf_bw2(cudaStream_t stream, const unsigned* afh, const unsigned* afl, const unsigned* bfh, const unsigned* bfl,
                   const TP* pp, long p0, int Nb, int cbs, int bs, const float* LU, const float* LC, float* ef, float* pf,
                   long pf0, float* dnp, long dn0, float* ga, float* gb, int M, int N, int Mp, int Np, int K, int B,
                   int nct, long units, int cap) {
    constexpr int bytes = Bw2Geo<KP>::WORDS * 4;
    auto kern = bf_bw2_kernel<KP, TP, FL, DE, GR>;
    constexpr int NDEV = 64;
    static int per_sm[NDEV] = {0};                             // (a device's CTAs an SM; 0: not asked yet)
    const int dev = at::cuda::current_device();
    int occ = dev < NDEV ? per_sm[dev] : 0;
    if (occ == 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, 256, bytes));
        occ = std::max(occ, 1);
        if (dev < NDEV) per_sm[dev] = occ;
    }
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    long grid = std::min<long>(units, (long)sms * occ);
    if (cap > 0) grid = std::min<long>(grid, cap);
    kern<<<(unsigned)std::max<long>(grid, 1), 256, bytes, stream>>>(afh, afl, bfh, bfl, pp, p0, Nb, cbs, bs, LU, LC, ef,
                                                                   pf, pf0, dnp, dn0, ga, gb, M, N, Mp, Np, K, B, nct,
                                                                   units);
}

// The backward of one node vector: the element flows of its children log-added into `element_flows` (the caller sets
// them to -inf first, once a layer), and -- each where its tensor is given (not empty) -- F+ added into `param_flows`
// from `pf0`, W into `denom` from `dn0`, dLL/dA and dLL/dB into the gradient buffers (the caller's layout, `[B, M, K]`,
// `[B, K, N]`). `zsplit`: ranks to 8 and the TF32 path, the samples split over this many CTAs a tile (0: as many as fill
// the GPU twice); the big tiles (bf16, ranks 9 .. 64), their persistent CTAs at most this many (0: the GPU's)
void exp_sum_backward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs,
                      int64_t bs, torch::Tensor emars, torch::Tensor rows, torch::Tensor node_flows,
                      torch::Tensor node_mars, int64_t nid0, torch::Tensor lzs, torch::Tensor element_flows,
                      torch::Tensor pflows, int64_t pf0, torch::Tensor denom, int64_t dn0, torch::Tensor gA,
                      torch::Tensor gB, int64_t M, int64_t N, int64_t K, int64_t zsplit, bool bf16) {
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
        TORCH_CHECK(gA.scalar_type() == torch::kFloat32 && gA.is_contiguous() && gA.sizes() == A.sizes(), "dA: [B, M, K]");
        TORCH_CHECK(gB.scalar_type() == torch::kFloat32 && gB.is_contiguous() && gB.sizes() == Bm.sizes(), "dB: [B, K, N]");
    }
    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int Mp = (int)((M + TBM - 1) / TBM * TBM), Np = (int)((N + BN - 1) / BN * BN), njt = Np / BN, nit = Mp / TBM;
    const int z = zsplit > 0 ? (int)std::min<int64_t>(zsplit, B) : grid_z(njt * nit, (int)B);
    const int bper = (int)((B + z - 1) / z), nz = (int)((B + bper - 1) / bper);
    const dim3 grid(njt, nit, (unsigned)nz);
    const long* rw = reinterpret_cast<const long*>(rows.data_ptr<int64_t>());
    float* pf = flows ? pflows.data_ptr<float>() : nullptr;
    float* dnp = dn ? denom.data_ptr<float>() : nullptr;
    int lb = 0;
    while ((1 << lb) < B) lb++;
    if (bf16) {
        const int KPb = rank_pad(K, 8);
        if (KPb >= 16) {                                       // ranks 9 .. 64: the big tiles (`bf_bw2_kernel`)
            const int Mp2 = (int)((M + TM2 - 1) / TM2 * TM2), Np2 = (int)((N + TN2 - 1) / TN2 * TN2), njt2 = Np2 / BN;
            const int nct = Np2 / TN2, nit2 = Mp2 / TM2;
            const long na2 = (long)Mp2 * KPb / 2, nb2 = (long)Np2 * KPb / 2;
            auto AF2 = torch::empty({2, B, na2}, A.options().dtype(torch::kInt32));
            auto BF2 = torch::empty({2, B, nb2}, A.options().dtype(torch::kInt32));
            auto LU2 = torch::empty({B, Mp2, 2}, A.options());
            auto LC2 = torch::empty({B, njt2, 36}, A.options());
            auto efs2 = torch::empty({B, Np2}, A.options());
            unsigned* afh2 = reinterpret_cast<unsigned*>(AF2.data_ptr<int>());
            unsigned* bfh2 = reinterpret_cast<unsigned*>(BF2.data_ptr<int>());
            const int nbA2 = (int)((na2 + 255) / 256), nbB2 = (int)((nb2 + 255) / 256);
            const int nT2 = (Mp2 + njt2 * 36 + Np2 + 255) / 256;
            bf_bw_prep_kernel<<<dim3(nbA2 + nbB2 + nT2, (unsigned)B), 256, 0, stream>>>(
                A.data_ptr<float>(), Bm.data_ptr<float>(), afh2, afh2 + B * na2, bfh2, bfh2 + B * nb2, LU2.data_ptr<float>(),
                LC2.data_ptr<float>(), efs2.data_ptr<float>(), node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
                nid0, lzs.data_ptr<float>(), emars.data_ptr<float>(), rw, (int)M, (int)N, Mp2, Np2, (int)K, KPb, (int)B,
                nbA2, nbB2);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            float* ga2 = grads ? gA.data_ptr<float>() : nullptr;
            float* gb2 = grads ? gB.data_ptr<float>() : nullptr;
            const long units = (long)nit2 * nct * B;
            EXP_SUM_KP_SWITCH(KPb, if constexpr (KP >= 16) { EXP_SUM_TP_SWITCH(params, EXP_SUM_FLAGS_SWITCH(flows, dn, grads,
                launch_bf_bw2<KP, TP, FL, DE, GR>(stream, afh2, afh2 + B * na2, bfh2, bfh2 + B * nb2, pp, p0, (int)Nb,
                    (int)cbs, (int)bs, LU2.data_ptr<float>(), LC2.data_ptr<float>(), efs2.data_ptr<float>(), pf, pf0, dnp,
                    dn0, ga2, gb2, (int)M, (int)N, Mp2, Np2, (int)K, (int)B, nct, units, (int)zsplit))) })
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            ef_epi_kernel<<<(unsigned)((((long)N << lb) + 255) / 256), 256, 0, stream>>>(
                efs2.data_ptr<float>(), rw, element_flows.data_ptr<float>(), (int)N, Np2, (int)B, lb);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        // ranks to 8 (`bf_bw_kernel`)
        const long na = (long)Mp * KPb / 2, nb = (long)Np * KPb / 2;
        auto AF = torch::empty({2, B, na}, A.options().dtype(torch::kInt32));
        auto BF = torch::empty({2, B, nb}, A.options().dtype(torch::kInt32));
        auto LU = torch::empty({B, Mp, 2}, A.options());
        auto LC = torch::empty({B, njt, 36}, A.options());
        auto efs = torch::empty({B, Np}, A.options());
        unsigned* afh = reinterpret_cast<unsigned*>(AF.data_ptr<int>());
        unsigned* bfh = reinterpret_cast<unsigned*>(BF.data_ptr<int>());
        const int nbA = (int)((na + 255) / 256), nbB = (int)((nb + 255) / 256), nT = (Mp + njt * 36 + Np + 255) / 256;
        bf_bw_prep_kernel<<<dim3(nbA + nbB + nT, (unsigned)B), 256, 0, stream>>>(
            A.data_ptr<float>(), Bm.data_ptr<float>(), afh, afh + B * na, bfh, bfh + B * nb, LU.data_ptr<float>(),
            LC.data_ptr<float>(), efs.data_ptr<float>(), node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), nid0,
            lzs.data_ptr<float>(), emars.data_ptr<float>(), rw, (int)M, (int)N, Mp, Np, (int)K, KPb, (int)B, nbA, nbB);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        float* ga = grads ? gA.data_ptr<float>() : nullptr;
        float* gb = grads ? gB.data_ptr<float>() : nullptr;
        EXP_SUM_KP_SWITCH(KPb, if constexpr (KP == 8) { EXP_SUM_TP_SWITCH(params, EXP_SUM_FLAGS_SWITCH(flows, dn, grads,
            bf_bw_kernel<KP, TP, FL, DE, GR><<<grid, 128, 0, stream>>>(afh, afh + B * na, bfh, bfh + B * nb, pp, p0,
                (int)Nb, (int)cbs, (int)bs, LU.data_ptr<float>(), LC.data_ptr<float>(), efs.data_ptr<float>(), pf, pf0,
                dnp, dn0, ga, gb, (int)M, (int)N, Mp, Np, (int)K, (int)B, bper))) })
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        ef_epi_kernel<<<(unsigned)((((long)N << lb) + 255) / 256), 256, 0, stream>>>(
            efs.data_ptr<float>(), rw, element_flows.data_ptr<float>(), (int)N, Np, (int)B, lb);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    const int KPv = rank_pad(K, 8), lkp = __builtin_ctz((unsigned)KPv);
    auto At = torch::empty({B, Mp, KPv}, A.options());
    auto Bt = torch::empty({B, KPv, Np}, A.options());
    auto terms = torch::empty({B, 2L * Mp + Np}, A.options());
    const long nscr = (long)B * Np + (grads ? (long)B * Mp * KPv + (long)B * KPv * Np : 0);
    auto scr = torch::empty({nscr}, A.options());              // ef [B, Np], then dAt [B, Mp, KP], dBt [B, KP, Np]
    float* efp = scr.data_ptr<float>();
    float* dat = grads ? efp + (long)B * Np : nullptr;
    float* dbt = grads ? dat + (long)B * Mp * KPv : nullptr;
    {
        const int nA = (Mp * KPv + 255) / 256, nB = KPv * ((Np + 255) / 256), nT = (2 * Mp + Np + 255) / 256;
        const long per = (long)Np + (grads ? (long)Mp * KPv + (long)KPv * Np : 0);
        const int nZ = (int)std::min<long>((per + 255) / 256, 64);
        tf_bw_prep_kernel<<<dim3(nA + nB + nT + nZ, (unsigned)B), 256, 0, stream>>>(
            A.data_ptr<float>(), Bm.data_ptr<float>(), At.data_ptr<float>(), Bt.data_ptr<float>(),
            terms.data_ptr<float>(), efp, grads ? 1 : 0, node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), nid0,
            lzs.data_ptr<float>(), emars.data_ptr<float>(), rw, (int)M, (int)N, Mp, Np, (int)K, lkp, (int)B, nA, nB, nT);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int sp = nz > 1 ? 1 : 0;
    EXP_SUM_KP_SWITCH(KPv, if constexpr (KP >= 8) { EXP_SUM_TP_SWITCH(params, EXP_SUM_FLAGS_SWITCH(flows, dn, grads,
        launch_tf_bw<KP, TP, FL, DE, GR>(grid, stream, At.data_ptr<float>(), Bt.data_ptr<float>(), pp, p0, (int)Nb,
                                         (int)cbs, (int)bs, terms.data_ptr<float>(), efp, pf, pf0, dnp, dn0, dat, dbt,
                                         (int)M, (int)N, Mp, Np, (int)K, (int)B, bper, sp))) })
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const long tot = ((long)N << lb) + (grads ? (long)B * M * K + (long)B * K * N : 0);
    tf_bw_epi_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, stream>>>(
        efp, rw, element_flows.data_ptr<float>(), dat, dbt, grads ? gA.data_ptr<float>() : nullptr,
        grads ? gB.data_ptr<float>() : nullptr, (int)M, (int)N, Mp, Np, (int)K, KPv, (int)B, lb, grads ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_sum_forward", &exp_sum_forward,
          "a node vector's values (and log Z) under the exp_sum scale, the scale made tile by tile and never stored");
    m.def("exp_sum_backward", &exp_sum_backward,
          "a node vector's element flows, F+, W and dLL/dA, dLL/dB under the exp_sum scale, one fused pass");
}
