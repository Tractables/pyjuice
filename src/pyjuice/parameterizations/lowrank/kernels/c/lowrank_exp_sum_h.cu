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
// THE FACTORS, fp32, fp16 or bf16 as the caller gives them, A [B, M, K], B [B, K, N] (a sample's block contiguous, any
// batch stride). fp16 ones are read IN PLACE where each sample's block is 16-byte aligned, M a multiple of 4 and N of 8
// (`h_in_place`), A made A' in the kernels' registers (`scale_h2`); others -- fp32, bf16, or fp16 not so aligned --
// through a COPY in fp16 the prologues make (AP [B, Mp, 16] = A', BP [B, 16, Np] = B, zero-padded: `conv_copy`), the
// forward's kept for the backward where asked (`keep`). Both kernels read either as they lie: plain row-major fp16, no
// fragment order -- ldmatrix of the staged rows.
//
// FORWARD (`hf_fw_kernel`): a CTA holds a tile of 128 nodes x TN children (TN = 8 NT8 = 64) -- eight consumer warps of
// 16 nodes, each warp all TN children (NT8 n-tiles of 8), its pairs' log2 theta (less the row's frame) in registers, the
// first MMA's C operand -- and loops over its samples; a producer warp streams each sample's operands into an
// NST-stage ring of shared memory (full / empty mbarriers): A's 128 rows and B's 64 children as boxes of tensor maps
// (TMA, 128-byte swizzled: ldmatrix rows conflict-free), their factors w and the tile's mc (`hf_fw_prep_kernel`'s WF)
// by a bulk copy.
//
// Layouts: the circuit's buffers [rows, batch], batch innermost; the shared parameters block-major
// (`LowRank._geometry`'s grid), edge block (i / bs, j / cbs) from `p0 + ((i / bs) Nb + j / cbs) cbs bs` on, child then
// node within it. Rank 16; node and child counts padded to whole tiles (zeros, -inf past the real ones).

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
__device__ __forceinline__ unsigned pack_hh(__half lo, __half hi) {
    const __half2 v = __halves2half2(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

// A' = log2 e a of a pair of fp16 a (the caller's fp16 A, read in place), in two fp16 instructions: fma(a, c1, a c2)
// with c1 = fl16(log2 e) and c2 = fl16(log2 e - c1) -- the rounded product fl16(log2 e a) for 98.4% of the normal fp16
// a, one ulp off for the rest (a c2's rounding; where it is subnormal, a under 0.2, most): the kernels of a pass all
// make A' so (and the copy of an fp16 A that cannot be read in place), so forward and backward agree
__device__ __forceinline__ unsigned scale_h2(unsigned a) {
    const __half2 x = *reinterpret_cast<const __half2*>(&a);
    const __half2 c1 = __float2half2_rn(0x1.714p+0f), c2 = __float2half2_rn(0x1.478p-12f);
    const __half2 r = __hfma2(x, c1, __hmul2(x, c2));
    return *reinterpret_cast<const unsigned*>(&r);
}

// The converted copy's words (`AP`: A' = log2 e A, a pair of ranks of a node; `BP`: B, a pair of children of a rank)
// from the caller's fp32 or bf16 (A' rounded once from fp32) or fp16 factors (scale_h2); zero where not live (scalar
// loads: a view may be 4-byte aligned)
__device__ __forceinline__ unsigned conv_a(const float* src, bool live) {
    return live ? pack_h2(__ldg(src) * LOG2E, __ldg(src + 1) * LOG2E) : 0u;
}
__device__ __forceinline__ unsigned conv_a(const __half* src, bool live) {
    return live ? scale_h2(pack_hh(src[0], src[1])) : 0u;
}
__device__ __forceinline__ unsigned conv_a(const __nv_bfloat16* src, bool live) {
    return live ? pack_h2(__bfloat162float(src[0]) * LOG2E, __bfloat162float(src[1]) * LOG2E) : 0u;
}
__device__ __forceinline__ unsigned conv_b(const float* src, bool l0, bool l1) {
    return pack_h2(l0 ? __ldg(src) : 0.f, l1 ? __ldg(src + 1) : 0.f);
}
__device__ __forceinline__ unsigned conv_b(const __nv_bfloat16* src, bool l0, bool l1) {
    return pack_h2(l0 ? __bfloat162float(src[0]) : 0.f, l1 ? __bfloat162float(src[1]) : 0.f);
}
__device__ __forceinline__ unsigned conv_b(const __half* src, bool l0, bool l1) {
    const __half z = __float2half_rn(0.f);
    return pack_hh(l0 ? src[0] : z, l1 ? src[1] : z);
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
__device__ __forceinline__ void st_shared(uint32_t a, unsigned v) {
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int parity) {
    asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
                 ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, int bytes, uint32_t bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// a box of a 3-D tensor (its map a kernel parameter) into shared memory, in the map's swizzle (zeros past its extents)
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* m, int c0, int c1, int c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, "
                 "%4}], [%5];" ::"r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}

// ldmatrix / stmatrix (8 x 8 b16 matrices; lane i addresses row i % 8 of matrix i / 8)
// (at shared addresses: one base and lane offsets, the steps' offsets immediates)
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm2t(unsigned (&r)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a) : "memory");
}
__device__ __forceinline__ void stsm4t(uint32_t p, unsigned a, unsigned b, unsigned c, unsigned d) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
                 ::"r"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// node_mars[nid0 + i, b] = (log2 S - log2 Z) ln 2 (-inf where S is 0), lzs[i, b] = log2 Z ln 2, from the column
// tiles' base-2 (shift, sum) pairs: part2 [b][jt][row] = (z, s) under the shifts (cm, cm + mc) -- cm the row's frame in
// the tile (CMR [tile][row]), mc the sample's largest lc' of the tile (WF's float `wmc` of its tile's `wfw`) -- or,
// z's sign bit set, a rare item's (|z|, s) under its own shifts (a, ms) = rsh [b][jt][row]. A block of 8 warps 32 rows x 8 / LPO samples
// of the chunk, LPO lanes an output (lane LPO row + h: tiles h, h + LPO, .. -- a load a warp's LPO 256 / LPO-byte runs
// --, 8 loads in flight, folded in order, the lanes' folds by shuffles); the outputs [node][batch] stored through
// shared memory, a row's samples contiguous. (A thread an output: 16 dependent folds a chain, 1.4x the old kernel at
// B64; 32 samples a block: too few blocks there, 2.5x.) LPO 4 where 2 would leave the SMs short of blocks
template <int LPO>
__global__ void __launch_bounds__(256) hf_comb_kernel(const float2* __restrict__ part2, const float2* __restrict__ rsh,
                                                      const float* __restrict__ CMR, const float* __restrict__ WF, int wfw,
                                                      int wmc, int njt, float* __restrict__ nmars, long nid0,
                                                      float* __restrict__ lzs, int M, int Mp, int Bc, int Bs, int b0) {
    constexpr int RW = 32 / LPO, SPB = 8 / LPO;                // (a warp's rows; a block's samples)
    __shared__ float2 out[32][SPB + 1];                        // [row][sample]: (node value, log Z)
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, h = lane % LPO;
    const int rl = RW * (w % LPO) + lane / LPO, k = w / LPO;   // (the thread's row in the block, its sample)
    const int i0 = blockIdx.x * 32, s0 = blockIdx.y * SPB;
    const int ic = min(i0 + rl, M - 1), bc = min(s0 + k, Bc - 1);    // (clamped: loads in range)
    const float* const cr = CMR + (long)(ic / HTM) * njt * HTM + ic % HTM;   // (row ic's frames: + HTM a tile)
    const float* const wm = WF + (long)bc * njt * wfw + wmc;                 // (the sample's mc: + wfw a tile)
    const long pb = (long)bc * njt * Mp + ic;
    float mz = NEG, z = 0.f, ms = NEG, s = 0.f;
    for (int t0 = h; t0 < njt; t0 += 8 * LPO) {
        float2 v[8];
        float c[8], mc[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {                          // (the 8 tiles' loads issued together)
            const int jt = t0 + LPO * u;
            const bool in = jt < njt;
            v[u] = in ? part2[pb + (long)jt * Mp] : make_float2(0.f, 0.f);
            c[u] = in ? cr[jt * HTM] : 0.f;
            mc[u] = in ? wm[jt * wfw] : NEG;
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            float4 r;
            if (signbit(v[u].x)) {                             // (a rare item: its own shifts)
                const float2 a = rsh[pb + (long)(t0 + LPO * u) * Mp];
                r = make_float4(a.x, fabsf(v[u].x), a.y, v[u].y);
            } else {
                r = make_float4(v[u].x > 0.f ? c[u] : NEG, v[u].x, v[u].y > 0.f ? c[u] + mc[u] : NEG, v[u].y);
            }
            fold_ms(mz, z, r.x, r.y);
            fold_ms(ms, s, r.z, r.w);
        }
    }
#pragma unroll
    for (int d = 1; d < LPO; d <<= 1) {
        fold_ms(mz, z, __shfl_xor_sync(0xffffffffu, mz, d), __shfl_xor_sync(0xffffffffu, z, d));
        fold_ms(ms, s, __shfl_xor_sync(0xffffffffu, ms, d), __shfl_xor_sync(0xffffffffu, s, d));
    }
    if (h == 0) {
        const float lz = z > 0.f ? (mz + __log2f(z)) * LN2 : NEG;
        const float ls = s > 0.f ? (ms + __log2f(s)) * LN2 : NEG;
        out[rl][k] = make_float2(ls > NEG ? ls - lz : ls, lz);   // (no mass: -inf, not 0 / 0)
    }
    __syncthreads();
    if (threadIdx.x < 32 * SPB) {                              // (a thread an output: row e / SPB, sample e % SPB)
        const int e = threadIdx.x, r = e / SPB, i = i0 + r, b = s0 + e % SPB;
        if (i < M && b < Bc) {
            const float2 o = out[r][e % SPB];
            nmars[(nid0 + i) * Bs + b0 + b] = o.x;
            lzs[(long)i * Bs + b0 + b] = o.y;
        }
    }
}

// ============================================================================================== forward
// A stage, per (tile, sample) item: the tile's 128 rows of A' -- or of the caller's fp16 A read in place, made A' in
// the consumers' registers (`scale_h2`) -- as [node][rank 16] fp16, a box of a tensor map of 128-byte lines (4 nodes
// a line: a box of 32-byte rows costs the copy engine a request a row, 4x the lines'); its TN children of B as [rank
// 16][child TN]; both 128-byte swizzled (each lane's ldmatrix row in a bank group of its own); and WF
// [jt][72] float: [tq][2 NT8] the factors w_j = 2^(lc'_j - mc) of a lane's children (TN jt + 8 t + 2 tq + p at 2 t + p),
// then mc (the tile's largest lc'; -inf: no child of the tile has a value) at 8 NT8, padded to 16 bytes
template <int NT8>
struct HfGeo {
    static_assert(NT8 == 8, "a tile's children: one 128-byte swizzled box");
    static constexpr int TN = 8 * NT8;
    static constexpr int WFW = 4 * 2 * NT8 + 4;                              // WF floats a tile (mc at 8 NT8)
    static constexpr int A_BYTES = HTM * HK * 2, B_BYTES = HK * TN * 2, W_BYTES = WFW * 4;
    static constexpr int ST = (A_BYTES + B_BYTES + W_BYTES + 1023) / 1024 * 1024;   // a stage (the boxes 1024-aligned)
};

// Where the 16 bytes an ldmatrix row reads lie in a stage's 128-byte swizzled boxes (a line's 16-byte piece c at c ^
// line % 8): A's half row h of node `row` (line row / 4, piece 2 (row % 4) + h) and B's children 8 c .. 8 c + 7 of
// rank row k (line k, piece c)
__device__ __forceinline__ int hf_a_at(int row, int h) { return (row >> 2) * 128 + (((2 * (row & 3) + h) ^ (row >> 2)) & 7) * 16; }
__device__ __forceinline__ int hf_b_at(int k, int c) { return k * 128 + ((c ^ k) & 7) * 16; }
// ... B's pieces of the n-tile pairs t2 = 0, 2, 4, 6 a lane reads (ldmatrix.trans: rank row k = 8 (lq & 1) + lr, piece
// t2 + (lq >> 1)): this base XOR 16 t2 -- (t2 + h) ^ k = (t2 ^ (k & 6)) | ((h ^ k) & 1), the base's bits 5, 6 those of
// 16 (k & 6) and the stage 1024-byte aligned: one XOR an access, no lane math in the loop
__device__ __forceinline__ int hf_b_base(int k, int h) { return k * 128 + 16 * ((h ^ k) & 1) + 16 * (k & 6); }

// The tiles' theta in the forward consumers' order, a block a tile (a thread a consumer lane): LTF [tile][warp][t][lane]
// float4 = the lane's log2 theta less its row's frame (the largest log2 theta of the row in the tile; 0 for a row
// without one), the MMA's C operand, and CMF [tile][warp][lane] = (cm of rows g, g + 8, live flags) -- the main
// kernel's tile change two runs of 16-byte loads, no index math, no logs. Made once for every sample (and every node
// vector of a forward pass reading the same run: `exp_sum_h_theta`)
template <typename TP, int NT8>
__global__ void __launch_bounds__(256) hf_theta_kernel(const TP* __restrict__ params, long p0, int Nb, int cbs, int bs,
                                                       float4* __restrict__ LTF, float4* __restrict__ CMF,
                                                       float* __restrict__ CMR, int M, int N, int Np) {
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
        if (tq == 0) {                                     // (the rows' frames for the comb kernel: CMR [tile][row])
            CMR[(long)tile * HTM + 16 * w + g] = cm[0];
            CMR[(long)tile * HTM + 16 * w + g + 8] = cm[1];
        }
    }
}

// A block of sample b's converted copy (the factors' fp16 operands where they are not read in place, the forward's and
// the backward's alike): blocks 0 .. nbA over AP [Mp, 16] = A' (a thread a pair of ranks of a node), then nbB over BP
// [16, Np] = B (a thread a pair of children of a rank); zero past the node vector's nodes / children. False: blk is
// past them
template <typename TI>
__device__ __forceinline__ bool conv_copy(int blk, int nbA, int nbB, const TI* __restrict__ A, const TI* __restrict__ Bm,
                                          unsigned* __restrict__ AP, unsigned* __restrict__ BP, int b, int M, int N,
                                          int Mp, int Np, long sA, long sB) {
    if (blk < nbA) {
        const int e = blk * 256 + threadIdx.x, i = e >> 3;   // (e = 8 i + kp)
        if (e < Mp * 8)
            AP[(long)b * Mp * 8 + e] = conv_a(A + (long)b * sA + (long)min(i, M - 1) * HK + 2 * (e & 7), i < M);
        return true;
    }
    blk -= nbA;
    if (blk < nbB) {
        const int e = blk * 256 + threadIdx.x;               // (e = k Np / 2 + jp)
        if (e < HK * (Np / 2)) {
            const int k = e / (Np / 2), j = 2 * (e - k * (Np / 2));
            BP[(long)b * HK * (Np / 2) + e] = conv_b(Bm + (long)b * sB + (long)k * N + min(j, N - 1), j < N, j + 1 < N);
        }
        return true;
    }
    return false;
}

// The prologue of a chunk, a grid row a sample: the converted copy's blocks (none where the factors are read in
// place), then WF's tiles (a warp a tile) with lcs [B, Np] = lc' (the exact fallback's)
template <typename TI, int NT8>
__global__ void __launch_bounds__(256) hf_fw_prep_kernel(
        const TI* __restrict__ A, const TI* __restrict__ Bm, unsigned* __restrict__ AP, unsigned* __restrict__ BP,
        float* __restrict__ WF, float* __restrict__ lcs, const float* __restrict__ emars, const long* __restrict__ rows,
        int M, int N, int Mp, int Np, int nbA, int nbB, int Bs, int b0, long sA, long sB) {
    using Gm = HfGeo<NT8>;
    const int b = blockIdx.y, njt = Np / Gm::TN;
    const int blk = blockIdx.x;
    if (conv_copy(blk, nbA, nbB, A, Bm, AP, BP, b, M, N, Mp, Np, sA, sB)) return;
    const int jt = (blk - nbA - nbB) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;   // WF: a lane NT8 / 4 children
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

// z (a sum of 2^x, >= 0) out of [2^-92, 2^100] -- or inf, NaN: its bits less 2^-92's beyond (2^100 - 2^-92)'s
__device__ __forceinline__ bool hf_out(float z) { return (unsigned)(__float_as_int(z) - 0x11800000) > 0x60000000u; }

// A warp's (shift, sum) pairs of its two rows (`hf_fw_kernel`'s registers a[r], z[r], ms[r], s[r])
struct HfRows { float a0, z0, ms0, s0, a1, z1, ms1, s1; };

// The forward's rare paths for one sample of a warp's rows: z out of range (the shifted path: a row max of lw') and S
// below its range (exactly: x' = lw' + lc' under its own max) -- phi made again a tile at a time from the warp's A'
// fragment `af` and B's rank rows of the sample at `bg` (fp16, rows bk apart, zeros from child bn), log2 theta read
// again from the parameters, the tile's w at `wv` (global). Every lane of the warp takes it. Not in the main loop: its
// items are deferred to a pass of their own (`hf_fw_kernel`) -- a call there, even one never taken, kept the loop's
// values in a stack frame (local loads every item: -14% with it gone)
template <typename TP, int NT8>
__device__ __forceinline__ HfRows hf_fw_rare(const __half* __restrict__ bg, int bk, int bn, const float* wv, uint4 af,
                                             const TP* __restrict__ params, long p0, int Nb, int cbs, int bs,
                                             const float* lcb, int M, int N, int rA, int rB, int j0, bool needz,
                                             bool needs, HfRows in) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const float mc = wv[8 * NT8];
    auto lw = [&](int t, float (&x)[4]) {                  // lw' (unframed) of the lane's four pairs of n-tile t
        float c[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float th = theta_at(params, p0, e & 2 ? rB : rA, j0 + 8 * t + 2 * tq + (e & 1), M, N, Nb, cbs, bs);
            c[e] = th > 0.f ? __log2f(th) : NEG;
        }
        const int j = j0 + 8 * t + g;                      // (n-tile t's B fragment: the lane's child, its rank pairs)
        const __half zh = __float2half_rn(0.f);
        __half v[4];
#pragma unroll
        for (int q = 0; q < 4; q++) v[q] = j < bn ? bg[(long)(2 * tq + (q & 1) + 8 * (q >> 1)) * bk + j] : zh;
        mma_h_c(x, af, pack_hh(v[0], v[1]), pack_hh(v[2], v[3]), c);
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
// last the producer: an item's boxes of A and B by its maps `tmA`, `tmB`, its WF by a bulk copy; its loads from a
// consumer warp's lane instead -- 8 warps, 128 registers a thread -- fell behind the items: +15%). The work is the
// (tile, sample) items in tile-major order, each CTA an even contiguous run of them (its tile's theta set up again where
// its run enters the next tile). `scaleA`: the A boxes are the caller's A (made A' by scale_h2), else the copy's A'.
// Partials part2[b][jt][row] = (z, s) -- Z's and S's base-2 sums over the tile's children under the shifts (cm, cm + mc)
// the comb kernel knows (CMR, WF): a quad's lanes tq 0 / 1 a row each, one 8-byte store (16-byte (shift, sum) pairs by
// the tq 0 lanes behind a branch cost 13% of the loop); a rare item's own shifts (a, ms) into rsh, z's sign set. The rare
// paths' items (`hf_fw_rare`) a warp notes in a bit mask over a stretch of 32 items and takes after it (a vote and a
// branch an item cost 14% in the items' serial tails), their operands from `Ag` / `Bg` (the boxes' tensors: A rows
// `sAg` a sample, zeros from row am; B rank rows bk apart, `sBg` a sample, zeros from child bn)
template <typename TP, int NT8, int NST>
__global__ void __launch_bounds__((HNW + 1) * 32, 2) hf_fw_kernel(
        const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, int scaleA,
        const __half* __restrict__ Ag, long sAg, int am, const __half* __restrict__ Bg, long sBg, int bk, int bn,
        const float* __restrict__ WF, const float* __restrict__ lcs, const TP* __restrict__ params, long p0, int Nb,
        int cbs, int bs, const float4* __restrict__ LTF, const float4* __restrict__ CMF, float2* __restrict__ part2,
        float2* __restrict__ rsh, int M, int N, int Mp, int Np, int B, int njt, long total) {
    using Gm = HfGeo<NT8>;
    constexpr int TN = Gm::TN;
    extern __shared__ __align__(128) unsigned char smraw[];
    const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smraw), sbase = (sraw + 1023u) & ~1023u;
    unsigned char* const sm = smraw + (sbase - sraw);         // (the stages 1024-byte aligned: the swizzles' frame)
    const uint32_t bfull = sbase + NST * Gm::ST, bempty = bfull + NST * 8, sguard = bempty + NST * 8;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3, lr = lane & 7, lq = lane >> 3;
    const long it0 = (long)blockIdx.x * total / gridDim.x, it1 = (long)(blockIdx.x + 1) * total / gridDim.x;
    if (it0 >= it1) return;
    if (tid == 0) {
        for (int s = 0; s < NST; s++) { mbar_init(bfull + 8 * s, 1); mbar_init(bempty + 8 * s, HNW); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (w == HNW) {                                           // the producer: an item's operands a stage
        if (lane == 0) {                                      // (the item's tile and sample by counters: no division)
            const long wstr = (long)njt * Gm::WFW;
            const int tile0 = (int)(it0 / B);
            int b = (int)(it0 - (long)tile0 * B), jt = tile0 % njt, rt = tile0 / njt, st = 0, ph = 0;
            for (long it = it0; it < it1; it++) {
                mbar_wait(bempty + 8 * st, ph ^ 1);
                const uint32_t dst = sbase + st * Gm::ST, bar = bfull + 8 * st;
                mbar_expect(bar, Gm::A_BYTES + Gm::B_BYTES + Gm::W_BYTES);
                tma_load_3d(dst, &tmA, 0, rt * (HTM / 4), b, bar);
                tma_load_3d(dst + Gm::A_BYTES, &tmB, jt * TN, 0, b, bar);
                bulk_g2s(dst + Gm::A_BYTES + Gm::B_BYTES, WF + b * wstr + (long)jt * Gm::WFW, Gm::W_BYTES, bar);
                if (++b == B) { b = 0; if (++jt == njt) { jt = 0; rt++; } }
                if (++st == NST) { st = 0; ph ^= 1; }
            }
        }
        return;
    }

    // the deferred items of a stretch of the warp's run (bit k of `bits`: item s0 + k): each one's sums as the main loop
    // left them in `part` (its tq 0 lanes' writes, a __syncwarp back), its operands from Ag / Bg, the rare paths
    auto rare_pass = [&](long s0, unsigned bits) {
        __syncwarp();
        while (bits) {
            const long itq = s0 + __ffs(bits) - 1;
            bits &= bits - 1;
            const int tl = (int)(itq / B), bq = (int)(itq - (long)tl * B), jq = tl % njt, j0q = jq * TN;
            const int rAq = (tl / njt) * HTM + 16 * w + g, rBq = rAq + 8;
            const float4 cv = __ldg(CMF + ((long)tl * HNW + w) * 32 + lane);
            const float* wq = WF + ((long)bq * njt + jq) * Gm::WFW;
            const float mcq = wq[8 * NT8];
            float2* const pq2 = part2 + ((long)bq * njt + jq) * Mp;
            float2* const pqs = rsh + ((long)bq * njt + jq) * Mp;
            const float2 u0 = pq2[rAq], u1 = pq2[rBq];       // (the main loop's sums: shifts cm, cm + mc)
            const float4 v0 = make_float4(u0.x > 0.f ? cv.x : NEG, u0.x, u0.y > 0.f ? cv.x + mcq : NEG, u0.y);
            const float4 v1 = make_float4(u1.x > 0.f ? cv.y : NEG, u1.x, u1.y > 0.f ? cv.y + mcq : NEG, u1.y);
            const bool l0 = cv.z != 0.f, l1 = cv.w != 0.f;
            const bool nz = (l0 && !(v0.y >= 0x1p-92f && v0.y <= 0x1p100f)) || (l1 && !(v1.y >= 0x1p-92f && v1.y <= 0x1p100f));
            const bool ns = mcq > NEG && ((l0 && !(v0.w >= 0x1p-92f)) || (l1 && !(v1.w >= 0x1p-92f)));
            const __half* const aq = Ag + (long)bq * sAg;     // (the warp's A' fragment: rows rAq / rBq, rank pairs)
            unsigned ah[4];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int row = r & 1 ? rBq : rAq, k = 2 * tq + 8 * (r >> 1);
                ah[r] = row < am ? *reinterpret_cast<const unsigned*>(aq + (long)row * HK + k) : 0u;
                if (scaleA) ah[r] = scale_h2(ah[r]);
            }
            const HfRows r = hf_fw_rare<TP, NT8>(Bg + (long)bq * sBg, bk, bn, wq, make_uint4(ah[0], ah[1], ah[2], ah[3]),
                                                 params, p0, Nb, cbs, bs, lcs + (long)bq * Np + j0q + 2 * tq, M, N, rAq,
                                                 rBq, j0q, nz, ns, HfRows{v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w});
            const int rq = tq ? rBq : rAq;                     // (a row by the lane that stored its sums: its sums,
            if (tq < 2 && rq < M) {                            //  z's sign bit set for the comb kernel, its shifts)
                pq2[rq] = tq ? make_float2(-fabsf(r.z1), r.s1) : make_float2(-fabsf(r.z0), r.s0);
                pqs[rq] = tq ? make_float2(r.a1, r.ms1) : make_float2(r.a0, r.ms0);
            }
        }
        __syncwarp();
    };

    float lt[NT8][4];
    bool liver;                                               // (the lane's row -- tq odd: rB, else rA -- has a theta)
    int cur, jt, rA, rB, j0;
    int pw;                                                   // (row rA's partial of the item: part2[pw] = [b][jt][rA])
    const int pstep = njt * Mp;
    int tile = (int)(it0 / B), b = (int)(it0 - (long)tile * B), st = 0, ph = 0;   // (counters: no division an item)
    const uint32_t a_ln = hf_a_at(16 * w + 8 * (lq & 1) + lr, lq >> 1);             // (the lane's ldmatrix rows)
    const uint32_t b_ln = Gm::A_BYTES + hf_b_base(8 * (lq & 1) + lr, lq >> 1);
    // the tile's log2 theta less each row's frame cm (its largest in the tile), from the prologue's fragments
    auto set_tile = [&]() {
        cur = tile;
        jt = tile % njt;
        j0 = jt * TN;
        rA = (tile / njt) * HTM + 16 * w + g;
        rB = rA + 8;
        pw = (b * njt + jt) * Mp + rA;
        const float4* lp = LTF + ((long)tile * HNW + w) * NT8 * 32 + lane;
#pragma unroll
        for (int t = 0; t < NT8; t++) {
            const float4 v = __ldg(lp + t * 32);
            lt[t][0] = v.x; lt[t][1] = v.y; lt[t][2] = v.z; lt[t][3] = v.w;
        }
        const float4 cv = __ldg(CMF + ((long)tile * HNW + w) * 32 + lane);
        liver = (tq & 1 ? cv.w : cv.z) != 0.f;
    };
    long it = it0;
    while (it < it1) {                                        // stretches of up to 32 items, each one's deferred items
        set_tile();                                           //  after it (each stretch from its own load: the tile's
        const long s0 = it, s1 = min(it1, it + 32);           //  registers free in the deferred pass)
        unsigned need = 0;                                    // (bit it - s0: the lane's rows of item it take the rare
        for (; it < s1; it++) {                               //  paths)
            if (tile != cur) set_tile();
            mbar_wait(bfull + 8 * st, ph);
            const unsigned char* sp = sm + st * Gm::ST;
            const uint32_t sb = sbase + st * Gm::ST;
            uint4 af;                                         // the warp's 16 rows of A' (ldmatrix of the half rows)
            {
                unsigned ah[4];
                ldsm4(ah, sb + a_ln);
                if (scaleA) {
#pragma unroll
                    for (int r = 0; r < 4; r++) ah[r] = scale_h2(ah[r]);
                }
                af = make_uint4(ah[0], ah[1], ah[2], ah[3]);
            }
            const float* wv = reinterpret_cast<const float*>(sp + Gm::A_BYTES + Gm::B_BYTES);
            const float mc = wv[8 * NT8];
            float z[2] = {0.f, 0.f}, s[2] = {0.f, 0.f};
#pragma unroll
            for (int t2 = 0; t2 < NT8; t2 += 2) {            // (n-tiles in pairs: an ldmatrix.trans of B, a 16-byte w)
                unsigned bh[4];
                ldsm4t(bh, (sb + b_ln) ^ (16u * t2));
                const float4 ww2 = *reinterpret_cast<const float4*>(wv + tq * 2 * NT8 + 2 * t2);
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int t = t2 + u;
                    const float wx = u ? ww2.z : ww2.x, wy = u ? ww2.w : ww2.y;
                    float d[4];
                    mma_h_c(d, af, bh[2 * u], bh[2 * u + 1], lt[t]);
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
            // the lane's row (tq odd: rB, else rA -- a quad's lanes hold both rows' sums): its rare-path test (2^-92: a
            // row's largest term over 2^-97, so any term flushed under 2^-126 is under 2^-29 of it; z, s >= 0: their
            // bits compared as integers, NaN / inf out -- bitwise, no branch) and its sums (z, s), stored by lanes tq
            // 0 and 1 (the shifts cm, cm + mc the comb kernel's: CMR, WF)
            const bool ro = tq & 1;
            const float zr = ro ? z[1] : z[0], sr = ro ? s[1] : s[0];
            const unsigned nd = (unsigned)liver & ((unsigned)hf_out(zr) |
                                                   ((unsigned)(mc > NEG) & (unsigned)(__float_as_int(sr) < 0x11800000)));
            need |= nd << (int)(it - s0);
            __syncwarp();
            if (lane == 0) {                                  // (the stage released after its reads return: ptxas
                st_shared(sguard + 4 * w, __float_as_uint(s[0]) ^ __float_as_uint(mc));   //  issues an arrive with
                mbar_arrive(bempty + 8 * st);                 //  loads in flight -- a store of their values first)
            }
            if (tq < 2 && (ro ? rB : rA) < M) part2[pw + 8 * ro] = make_float2(zr, sr);
            pw += pstep;                                      // (the next sample's; a tile change sets it anew)
            if (++b == B) { b = 0; tile++; }
            if (++st == NST) { st = 0; ph ^= 1; }
        }
        need = __reduce_or_sync(0xffffffffu, need);           // (the warp's: every lane takes each item)
        if (need) rare_pass(s0, need);
    }
}


// ================================================================================================= backward
// `lowrank_exp_sum.cu`'s big-tile backward (`bf_bw2_kernel`, its head has the design) at rank 16 with phi from ONE fp16
// MMA (the forward's rounded factors; log2 theta its C operand: lw' straight from the tensor pipe, one n-tile pair
// ahead of the element terms) and G, B, A' fp16 in the dA / dB products: a CTA of 8 warps a tile of 128 nodes x
// 64 children over a run of (tile, sample) units, F+ and W in registers over a tile's samples. PIPELINED, one barrier
// a sample: iteration b (parity par) makes phi_b, b's element terms -- each n-tile with one k-step of dB of sample
// b - 1 (from the other parity's G^T / A', every warp's written a barrier back), each second n-tile with b's dA
// k-step and G^T m-tile --, and writes b's G^T / A' / element-flow partials into parity par; the next [A] publishes
// them. A phase saturating one unit (shared memory, the tensor pipe, MUFU) with the others idle, one after the other,
// is what bound the two-barrier form: interleaved, the units work at once (-16% B64, -21% B256). The gradients' sums
// leave as fp32 bulk adds of a CTA's partials after the next [A]: dA's rows (contiguous) one plain bulk op, dB's 16
// rank rows of 64 children one TENSOR reduce through a 3-D map of the caller's [B, 16, N] (16 bulk ops of 256 bytes
// cost 12 us at B64, 48 at B256: the L2's small-op rate).
// The operands, staged by cp.async into padded rows, as they lie: the forward's converted copy (A' [Mp, 16], B [16, Np]
// a sample, fp16) or the caller's fp16 A [M, 16], B [16, N] read in place (pieces past M / N zero-filled; A made A'
// by scale_h2 in phi's registers, dB's product of A itself, so its sums without the ln 2)
constexpr int TM2 = 128, TN2 = 64, NW2 = 8, BN = 32;
static_assert(TM2 == HTM && TN2 == 64, "the forward's tiles: one copy of the factors serves both");
struct Hb2Geo {
    static constexpr int NT = 2;
    // (words; rows padded so each 8 x 8 ldmatrix / stmatrix access meets 8 distinct 16-byte bank groups)
    static constexpr int AROW = 12, BROW = 36, TROW = 68;    // a row: A' 16 halves (+ 8), B 64 (+ 8), G^T 128 (+ 8)
    static constexpr int SAH = TM2 * AROW;                     // A' rows of the tile: [node 128][rank 16 + pad]
    static constexpr int SBH = HK * BROW;                      // B of the tile: [rank 16][child 64 + pad]
    static constexpr int SL = 2 * TM2 + 2 * 36 + 4;            // a stage's terms: LU (128 float2), LC (2 x 36), SC
    static constexpr int ST = TN2 * TROW;                      // G^T of the tile: [child 64][node 128 + pad]
    static constexpr int SE = NW2 * TN2;                       // element flows: [warp][64]
    static constexpr int SDA = TM2 * HK;                       // dA's partials (fp32): [node 128][rank 16]
    static constexpr int SDB = HK * TN2;                       // dB's: [rank 16][child 64]
    // the partials first (the TMA source 128-byte aligned); A' in three buffers (sample b - 1's for dB, b's, b + 1's
    // landing), B, the terms, G^T and the element flows by parity
    static constexpr int MAIN = 2 * SDA + 2 * SDB + 3 * SAH + 2 * SBH + 2 * SL + 2 * ST + 2 * SE;
    static constexpr int FLUSH = NW2 * 32 * 20;                // F+ / W: [warp][child 32][node 16 + 4], a half at a time
    static constexpr int WORDS = MAIN;
    static_assert(FLUSH <= 2 * ST, "the F+ / W flush within the G^T buffers");
    static_assert(SDA % 32 == 0 && SDB % 32 == 0 && SAH % 4 == 0 && SBH % 4 == 0 && SL % 4 == 0 && ST % 4 == 0,
                  "aligned pieces");
};

// Atomic adds (vector forms: sm_90 on)
__device__ __forceinline__ void red1(float* p, float a) {
    asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(a) : "memory");
}
__device__ __forceinline__ void red4(float* p, float4 v) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
// (a thread's shared-memory writes made visible to the async proxy: before the barrier ahead of a bulk add)
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// (zeros where not live: nothing read)
__device__ __forceinline__ void cp_async16z(uint32_t sa, const void* gmem, bool live) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(gmem), "r"(live ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t sa, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(gmem) : "memory");
}
// fp32 bulk adds from shared memory into global (the L2's adds), a bulk group each -- a run of a buffer, or a box of a
// 3-D tensor (its map a kernel parameter; 128-byte aligned, the box's layout) --, the issuing thread waiting for its
// groups' reads before their buffer is written again. Predicated (`on`: one thread's): no branch around them
__device__ __forceinline__ void bulk_wait_read0_if(bool on) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %0, 0;\n @p cp.async.bulk.wait_group.read 0;\n}\n" ::"r"((int)on)
                 : "memory");
}
__device__ __forceinline__ void bulk_add_f_if(bool on, void* gdst, uint32_t ssrc, int bytes) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %0, 0;\n"
                 " @p cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%1], [%2], %3;\n"
                 " @p cp.async.bulk.commit_group;\n}\n" ::"r"((int)on), "l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_add_3d_if(bool on, const CUtensorMap* m, uint32_t ssrc, int c0, int c1, int c2) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %0, 0;\n"
                 " @p cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%1, {%3, %4, %5}], [%2];\n"
                 " @p cp.async.bulk.commit_group;\n}\n"
                 ::"r"((int)on), "l"(m), "r"(ssrc), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
// fp16 m16n8k16 accumulating into D (the four A registers given apart)
__device__ __forceinline__ void mma_h16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The backward's prologue, one launch: the converted copy's blocks where the forward's is not at hand and the factors
// are not read in place (`conv_copy`, (nbA + nbB) a sample: AP [B, Mp, 16] = A', BP [B, 16, Np] = B in fp16, zero
// past the node vector's nodes / children); then, in tiles of 32 rows x 32 samples (the circuit's [rows, batch] read a
// row's 32 samples a warp's 128 bytes, the tile turned through shared memory, written a sample's row at a time -- a
// thread a node read the rows 4 bytes a 32-byte sector, 4x the bytes: 65 us at B1024), the nodes' terms LU [B, Mp, 2]
// = (lu', lv') -- lu' = log2 (f / (n Z)), lv' = log2 (f / Z), -inf for a node without flow --, the children's LC [B,
// njt, 36] (per tq the lane's 8 children's lc') with the element-flow scratch ef [B, Np] zeroed; and a block of 8
// samples' scales SC [B]: E = ceil(log2 max_i f_i) -> (E, 2^E, ln 2 2^E, 0). The main kernel takes q and t (so G, the
// fp16 operand of the dA / dB products) in units of 2^E -- |G| <= 1: fp16's range whatever the flows' scale, a
// sample's common scale carried apart in fp32 -- and its sums back by 2^E
template <typename TI>
__global__ void __launch_bounds__(256) hb_bw_prep_kernel(
        const TI* __restrict__ A, const TI* __restrict__ Bm, unsigned* __restrict__ AP, unsigned* __restrict__ BP,
        float* __restrict__ LU, float* __restrict__ LC, float* __restrict__ ef, float4* __restrict__ SC,
        const float* __restrict__ nflows, const float* __restrict__ nmars, long nid0,
        const float* __restrict__ lzs, const float* __restrict__ emars, const long* __restrict__ rows, int M, int N,
        int Mp, int Np, int nbA, int nbB, int Bc, int Bs, int b0, long sA, long sB, float4* __restrict__ zA, long szA,
        float4* __restrict__ zB, long szB, int nbZ) {
    __shared__ float2 tile[32][33];                            // [sample][row] (padded)
    const int njt = Np / BN, nbg = (Bc + 31) / 32, lane = threadIdx.x & 31, wq = threadIdx.x >> 5;
    int blk = blockIdx.x;
    if (blk < nbZ * Bc) {                                      // the gradients' zeros (`init`: their first writer): a
        const int nA4 = M * (HK / 4), n4 = nA4 + N * (HK / 4), s = blk / nbZ;   // sample's dA then dB rows, 4096
        const int e1 = min(n4, (blk - s * nbZ + 1) * 4096);    // float4s a block -- in the L2 when the main kernel's
        const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);     // adds come (their DRAM round trip of zeros saved)
        for (int e = (blk - s * nbZ) * 4096 + threadIdx.x; e < e1; e += 256) {
            if (e < nA4) zA[s * szA + e] = z4;
            else zB[s * szB + (e - nA4)] = z4;
        }
        return;
    }
    blk -= nbZ * Bc;
    if (blk < (nbA + nbB) * Bc) {                              // the copy: a sample's blocks
        conv_copy(blk % (nbA + nbB), nbA, nbB, A, Bm, AP, BP, blk / (nbA + nbB), M, N, Mp, Np, sA, sB);
        return;
    }
    blk -= (nbA + nbB) * Bc;
    const int nLU = (Mp / 32) * nbg, nLC = njt * nbg;
    if (blk < nLU) {                                           // LU: nodes o0 .., samples s0 ..
        const int o0 = (blk / nbg) * 32, s0 = (blk % nbg) * 32, sb = s0 + lane;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int o = o0 + wq + 8 * k;
            float lu = NEG, lv = NEG;
            if (o < M && sb < Bc) {
                const long at = (nid0 + o) * Bs + b0 + sb;
                const float lf = nflows[at];
                if (isfinite(lf)) {
                    const float lz = lzs[(long)o * Bs + b0 + sb];
                    lu = (lf - (nmars[at] + lz)) * LOG2E;
                    lv = (lf - lz) * LOG2E;
                }
            }
            tile[lane][wq + 8 * k] = make_float2(lu, lv);
        }
        __syncthreads();
        for (int e = threadIdx.x; e < 32 * 32; e += 256) {
            const int i = e >> 5, r = e & 31;
            if (s0 + i < Bc) reinterpret_cast<float2*>(LU)[(long)(s0 + i) * Mp + o0 + r] = tile[i][r];
        }
        return;
    }
    blk -= nLU;
    if (blk < nLC) {                                           // LC and ef: child tile jt, samples s0 ..
        const int jt = blk / nbg, s0 = (blk % nbg) * 32, sb = s0 + lane;
        float* const tl = reinterpret_cast<float*>(&tile[0][0]);   // ([sample][slot 32 + 1])
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int jl = wq + 8 * k, j = BN * jt + jl;      // (child jl = 8 nf + 2 tq + p at slot 8 tq + 2 nf + p)
            const long r = j < N ? rows[j] : -1;
            const float v = r >= 0 && sb < Bc ? emars[r * Bs + b0 + sb] * LOG2E : NEG;
            tl[lane * 33 + 8 * ((jl >> 1) & 3) + 2 * (jl >> 3) + (jl & 1)] = v;
        }
        __syncthreads();
        for (int e = threadIdx.x; e < 32 * 36; e += 256) {
            const int i = e / 36, w = e - i * 36;
            if (s0 + i < Bc) LC[((long)(s0 + i) * njt + jt) * 36 + w] = w < 32 ? tl[i * 33 + w] : 0.f;
        }
        for (int e = threadIdx.x; e < 32 * 32; e += 256) {
            const int i = e >> 5;
            if (s0 + i < Bc) ef[(long)(s0 + i) * Np + BN * jt + (e & 31)] = 0.f;
        }
        return;
    }
    blk -= nLC;                                                // the scales: 8 samples a block (a lane's sample
    const int s0 = blk * 8, sl = lane & 7, ol = lane >> 3, sb = s0 + sl;   //  lane % 8, its rows lane / 8 + 4 (warp
    float m = NEG;                                                          //  + 8 k): 32-byte runs, 8 in flight)
    if (sb < Bc)
        for (int o0 = 4 * wq + ol; o0 < M; o0 += 256) {
            float v[8];
#pragma unroll
            for (int u = 0; u < 8; u++) v[u] = o0 + 32 * u < M ? nflows[(nid0 + o0 + 32 * u) * Bs + b0 + sb] : NEG;
#pragma unroll
            for (int u = 0; u < 8; u++)
                if (isfinite(v[u])) m = fmaxf(m, v[u]);
        }
    float* const red = reinterpret_cast<float*>(&tile[0][0]);   // ([warp][lane])
    red[wq * 32 + lane] = m;
    __syncthreads();
    if (threadIdx.x < 8 && s0 + threadIdx.x < Bc) {
        m = NEG;
        for (int q = 0; q < 32; q++) m = fmaxf(m, red[(q >> 2) * 32 + 8 * (q & 3) + threadIdx.x]);
        const float E = m > NEG ? fminf(fmaxf(ceilf(m * LOG2E), -120.f), 120.f) : 0.f;   // (no flow: any)
        SC[s0 + threadIdx.x] = make_float4(E, exp2f(E), LN2 * exp2f(E), 0.f);
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
        const __half* __restrict__ AP, long sAb, int am, const __half* __restrict__ BP, long sBb, int bk, int bn,
        int scaleA, const TP* __restrict__ params, long p0, int Nb, int cbs, int bs, const float* __restrict__ LU, const float* __restrict__ LC, float* __restrict__ ef,
        float* __restrict__ pflows, long pf0, float* __restrict__ denom, long dn0, float* __restrict__ gA, long sgA,
        int ma, const __grid_constant__ CUtensorMap tmB, const float4* __restrict__ SC, int M, int N, int Mp, int Np,
        int B, int nct, long units) {
    using Gm = Hb2Geo;
    constexpr int NT = Gm::NT;
    extern __shared__ __align__(128) unsigned smw[];
    float* const sDA = reinterpret_cast<float*>(smw);          // [par][node 128][rank 16]
    float* const sDB = sDA + 2 * Gm::SDA;                      // [par][rank 16][child 64]
    unsigned* const sAh = reinterpret_cast<unsigned*>(sDB + 2 * Gm::SDB);   // [3][node][AROW]: A' rows (fp16 pairs)
    unsigned* const sBh = sAh + 3 * Gm::SAH;                   // [par][rank][BROW]: B (fp16 pairs of children)
    float* const sL = reinterpret_cast<float*>(sBh + 2 * Gm::SBH);         // [par][SL]: LU, LC, SC
    unsigned* const sT2 = reinterpret_cast<unsigned*>(sL + 2 * Gm::SL);    // [par][child][TROW]: G^T (fp16 pairs)
    float* const sE2 = reinterpret_cast<float*>(sT2 + 2 * Gm::ST);         // [par][SE]
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int wu = __shfl_sync(0xffffffffu, w, 0);             // (the warp's index, provably uniform: uniform branches)
    const int lr = lane & 7, lq = lane >> 3;                   // (ldmatrix / stmatrix: the row a lane addresses, its matrix)
    const bool issuer = GRADS && tid == 0;                     // (the bulk adds' thread)
    const int njt = Np / BN;
    const long u0 = (long)blockIdx.x * units / gridDim.x, u1 = (long)(blockIdx.x + 1) * units / gridDim.x;
    // Shared addresses: one base and the lanes' byte offsets into each buffer (no address conversion an access, the
    // k-steps' and n-tiles' offsets immediates)
    const uint32_t uDA = smem_u32(smw), uDB = uDA + 2 * Gm::SDA * 4, uAh = uDB + 2 * Gm::SDB * 4;
    const uint32_t uBh = uAh + 3 * Gm::SAH * 4, uL = uBh + 2 * Gm::SBH * 4, uT2 = uL + 2 * Gm::SL * 4;
    const uint32_t oA = ((16 * w + 8 * (lq & 1) + lr) * Gm::AROW + 4 * (lq >> 1)) * 4;     // phi's A' rows
    const uint32_t oB = ((8 * (lq & 1) + lr) * Gm::BROW + 4 * (lq >> 1)) * 4;             // phi's B, + 16 t2
    const uint32_t oBt = ((8 * (lq >> 1) + lr) * Gm::BROW + 4 * (lq & 1)) * 4;            // dA's B^T, + 32 kk
    const uint32_t oT = ((8 * (lq >> 1) + lr) * Gm::TROW + 8 * w + 4 * (lq & 1)) * 4;    // G^T's rows, + 64 TROW kk
    const uint32_t oTa = ((16 * (w & 3) + 8 * (lq & 1) + lr) * Gm::TROW + 4 * (lq >> 1)) * 4;   // dB's G^T, + 32 kw
    const uint32_t oAb = ((8 * (lq & 1) + lr) * Gm::AROW + 4 * (w >> 2)) * 4;             // dB's A', + 64 AROW kw
    // sample b's operands into stage (ab: A' buffer, par: B and terms; row tile its, child tile jcs): the tile's 128
    // A rows (a thread a 16-byte half row; zeros from row am), B's 16 rank rows of its 64 children (warps 0 .. 3: 16
    // bytes a thread, rows bk halves apart; zeros from child bn), the terms (LU rows, the 2 jt's LC, SC: warps 4 .. 6).
    // A thread's sources a tile's base and its sample's stride: `src` of the tile, then base + b stride a sample
    struct Src { const char* a; const char* r; long ra; bool al, rl; };
    auto srcs = [&](int jcs, int its) {
        Src q;
        const int row = tid >> 1, half = tid & 1, i = its * TM2 + row;
        q.a = reinterpret_cast<const char*>(AP + (long)min(i, am - 1) * HK + 8 * half);
        q.al = i < am;
        q.r = nullptr; q.ra = 0; q.rl = false;
        if (wu < 4) {
            const int rk = tid >> 3, ch = tid & 7, j = jcs * TN2 + 8 * ch;
            q.r = reinterpret_cast<const char*>(BP + (long)rk * bk + min(j, bn - 8));
            q.ra = sBb * 2;
            q.rl = j < bn;
        } else if (wu < 7) {
            const int r = tid - 128;
            q.rl = r < 83;
            q.r = reinterpret_cast<const char*>(r < 64 ? LU + (long)its * TM2 * 2 + r * 4
                                                : r < 82 ? LC + (long)2 * jcs * 36 + (r - 64) * 4
                                                         : reinterpret_cast<const float*>(SC));
            q.ra = r < 64 ? (long)Mp * 2 * 4 : r < 82 ? (long)njt * 36 * 4 : 16;
        }
        return q;
    };
    const uint32_t dA0 = uAh + ((tid >> 1) * Gm::AROW + 4 * (tid & 1)) * 4;
    const uint32_t dR0 = wu < 4 ? uBh + ((tid >> 3) * Gm::BROW + 4 * (tid & 7)) * 4 : uL + (tid - 128) * 16;
    const uint32_t dRs = wu < 4 ? Gm::SBH * 4 : Gm::SL * 4;
    auto stage = [&](const Src& q, int b, int ab, int par) {
        cp_async16z(dA0 + ab * (Gm::SAH * 4), q.a + (long)b * sAb * 2, q.al);
        if (wu < 4) cp_async16z(dR0 + par * dRs, q.r + b * q.ra, q.rl);
        else if (wu < 7 && q.rl) cp_async16(dR0 + par * dRs, q.r + b * q.ra);
        cp_async_commit();
    };
    // dB of a sample from parity pp's G^T and A' buffer ab (all 8 warps' blocks written, a barrier back): unit w =
    // (child m-tile w & 3, rank tile w >> 2), k-step kw the 8 warps' 16-node blocks; A operand G^T [child][node] by
    // ldmatrix, B operand A' [node][rank] by ldmatrix.trans
    auto dB_step = [&](int pp, int ab, int kw, float (&acc)[4]) {
        unsigned aa[4], bb[2];
        ldsm4(aa, uT2 + pp * (Gm::ST * 4) + oTa + 32 * kw);
        ldsm2t(bb, uAh + ab * (Gm::SAH * 4) + oAb + 64 * Gm::AROW * kw);
        mma_h16(acc, aa[0], aa[1], aa[2], aa[3], bb[0], bb[1]);
    };
    auto dB_mma = [&](int pp, int ab, float (&dq)[4]) {
#pragma unroll
        for (int c = 0; c < 4; c++) dq[c] = 0.f;
#pragma unroll
        for (int kw = 0; kw < NW2; kw++) dB_step(pp, ab, kw, dq);
    };
    // its unit's sums (G in units of 2^E, A' carrying log2 e: times scz = ln 2 2^E; 2^E of the caller's A) into
    // parity pp's [rank][child] rows (children 16 mg + g / + 8, ranks 8 nt + 2 tq / + 1)
    auto dB_out = [&](int pp, const float (&dq)[4], float scz) {
        const int mg = w & 3, nt = w >> 2, ch = 16 * mg + g, rk = 8 * nt + 2 * tq;
        float* const db = sDB + pp * Gm::SDB;
        db[rk * TN2 + ch] = dq[0] * scz;
        db[(rk + 1) * TN2 + ch] = dq[1] * scz;
        db[rk * TN2 + ch + 8] = dq[2] * scz;
        db[(rk + 1) * TN2 + ch + 8] = dq[3] * scz;
        fence_async_smem();
    };
    // the bulk adds of a sample's dA rows (the tile's 128, clipped at ma) and dB's box (16 ranks x 64 children at
    // child j0, sample b; children past the map's dropped), from parity pp (written and fenced a barrier back)
    auto out_dA = [&](int pp, int b, int i0) {             // (warp 0's: its lane 0 by predicate)
        if (wu == 0) {
            const int rows = min(TM2, ma - i0);
            bulk_add_f_if(issuer && rows > 0, gA + (long)b * sgA + (long)i0 * HK, uDA + pp * (Gm::SDA * 4), rows * HK * 4);
        }
    };
    auto out_dB = [&](int pp, int b, int j0) {
        if (wu == 0) tma_add_3d_if(issuer, &tmB, uDB + pp * (Gm::SDB * 4), j0, 0, b);
    };
    // the element flows of a sample from parity pp's partials (its 8 warps' sums), one add
    auto flows_out = [&](int pp, int b, int j0, float scy) {
        if (wu < TN2 / 32) {
            const float* const sE = sE2 + pp * Gm::SE;
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < NW2; ww++) s += sE[ww * TN2 + tid];
            if (s != 0.f && j0 + tid < N) red1(ef + (long)b * Np + j0 + tid, s * scy);
        }
    };
    int pc = 0, pa = 0;                                        // the sample's B / terms parity, its A' buffer (mod 3)
    if (u0 < u1) {                                             // the run's first sample: its stage
        const int t = (int)(u0 / B), b = (int)(u0 - (long)t * B), it = t / nct;
        stage(srcs(t - it * nct, it), b, 0, 0);
    }
#pragma unroll 1
    for (long u = u0; u < u1;) {
        const int t = (int)(u / B), b0 = (int)(u - (long)t * B);
        const int b1 = (int)min((long)B, (long)b0 + (u1 - u));
        u += b1 - b0;
        const bool more = u < u1;                              // (the run's next tile: t + 1, from its sample 0)
        const int it = t / nct, jc = t - it * nct;
        const int i0 = it * TM2, j0 = jc * TN2;
        const int rA = i0 + 16 * w + g, rB = rA + 8;
        const Src sq = srcs(jc, it);                           // (this tile's sources)

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
        for (int b = b0; b < b1; b++, pc ^= 1, pa = pa == 2 ? 0 : pa + 1) {
            const int par = pc, pap = pa == 0 ? 2 : pa - 1, pan = pa == 2 ? 0 : pa + 1;
            cp_async_wait<0>();
            if (wu == 0) bulk_wait_read0_if(issuer);           // (the adds issued at the last [A] have read their rows)
            __syncthreads();                                   // [A] b's stage; b - 1's G^T / partials written
            if (GRADS) {
                if (b > b0) out_dA(par ^ 1, b - 1, i0);
                if (b > b0 + 1) out_dB(par, b - 2, j0);
            }
            if (b > b0) flows_out(par ^ 1, b - 1, j0, scy_prev);
            if (b + 1 < b1) stage(sq, b + 1, pan, par ^ 1);    // (its buffers' last readers: before [A])
            else if (more) {                                   // (the next tile's first sample)
                const int itn = (t + 1) / nct;
                stage(srcs(t + 1 - itn * nct, itn), 0, pan, par ^ 1);
            }
            const uint32_t uah = uAh + pa * (Gm::SAH * 4), ubh = uBh + par * (Gm::SBH * 4);
            const float* const sl = sL + par * Gm::SL;
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
            if (b == b0) {                                     // (the tile's log2 theta: the phi MMAs' C operand)
#pragma unroll
                for (int e = 0; e < 32; e++) lt[e] = lt[e] > 0.f ? __log2f(lt[e]) : NEG;
            }
            // ---- lw_b = phi_b + log2 theta (one fp16 term, theta the C operand): the warp's A' fragment (ldmatrix of
            // its 16 rows), B's (ldmatrix.trans of the rank rows: k pairs along the ranks), a pair of n-tiles at a
            // time, one pair ahead of the element terms (its registers live a pair at a time, not all eight)
            unsigned ah[4];
            ldsm4(ah, uah + oA);
            if (scaleA) {                                      // (the caller's A: A' as the forward made it)
#pragma unroll
                for (int r = 0; r < 4; r++) ah[r] = scale_h2(ah[r]);
            }
            float S[8][4];
            auto phi_pair = [&](int t2) {                      // (n-tiles t2, t2 + 1: their k halves)
                unsigned bh[4];
                ldsm4t(bh, ubh + oB + 16 * t2);
                const uint4 a4 = make_uint4(ah[0], ah[1], ah[2], ah[3]);
                mma_h_c(S[t2], a4, bh[0], bh[1], *reinterpret_cast<const float(*)[4]>(lt + 4 * t2));
                mma_h_c(S[t2 + 1], a4, bh[2], bh[3], *reinterpret_cast<const float(*)[4]>(lt + 4 * t2 + 4));
            };
            phi_pair(0);
            float dq[4] = {0.f, 0.f, 0.f, 0.f};                // (dB of sample b - 1, its k-steps in the loop below:
                                                               //  one chain, its MMAs an n-tile's work apart)
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
                if ((nf & 1) == 0 && nf + 2 < 8) phi_pair(nf + 2);
                if (GRADS && b > b0) dB_step(par ^ 1, pap, nf, dq);
                float gq[4], qq[4];
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const float lw = S[nf][c];
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
                    // R[2 kk + 1][1]; B^T by ldmatrix of B's rank rows (k pairs along the children), both rank tiles
                    const int kk = nf >> 1;
                    unsigned bt[4];
                    ldsm4(bt, ubh + oBt + 32 * kk);
                    mma_h16(dd[0], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], bt[0], bt[1]);
                    mma_h16(dd[1], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], bt[2], bt[3]);
                    // dB's G^T (children 16 kk .. 16 kk + 15, the warp's 16 nodes) by stmatrix.trans into parity par
                    stsm4t(uT2 + par * (Gm::ST * 4) + oT + 64 * Gm::TROW * kk,
                           R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1]);
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
            scy_prev = scl.y; scz_prev = scaleA ? scl.y : scl.z;   // (dB of A itself: no ln 2)
        }
        // the tile's drain: the last sample's G^T / partials published, its dB and element flows, then F+ / W
        const int pl = pc ^ 1, pal = pa == 0 ? 2 : pa - 1;     // (the last sample's parity, its A' buffer)
        if (wu == 0) bulk_wait_read0_if(issuer);
        __syncthreads();
        if (GRADS) {
            out_dA(pl, b1 - 1, i0);
            if (b1 - 1 > b0) out_dB(pl ^ 1, b1 - 2, j0);
        }
        flows_out(pl, b1 - 1, j0, scy_prev);
        if (GRADS) {
            float dq[4];
            dB_mma(pl, pal, dq);
            dB_out(pl, dq, scz_prev);                          // (pl's dB rows: their last add waited before the drain)
        }
        __syncthreads();                                       // (the drain's reads of G^T / A' and dB's writes done)
        if (GRADS) out_dB(pl, b1 - 1, j0);
        // F+ and W of the warp's 16 x 64 cells, half the children at a time through its own shared rows (the G^T
        // buffers: the next tile's first stage lands meanwhile in A' / B / terms): a lane four consecutive nodes of
        // a child, one 16-byte add (scalar adds where the node blocks are under 4 or the runs unaligned)
        if (FLOWS || DENOM) {
            float* const st = reinterpret_cast<float*>(sT2) + w * (32 * 20);   // [child 32][node 16 + 4]
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

// A sample's stride (halves or floats): a single sample's whatever its own, the dense one
static int64_t sstride(const torch::Tensor& t) { return t.size(0) <= 1 ? t.size(1) * t.size(2) : t.stride(0); }

// Whether the factors are read in place: fp16, each sample's block 16-byte aligned (the base, the batch stride a
// multiple of 8 halves), A's rows whole 128-byte lines (M a multiple of 4) and B's rank rows whole 8-child pieces (N a
// multiple of 8) -- the TMA boxes', the cp.async pieces' alignment. Else the kernels read a converted copy (`conv_copy`)
static bool h_in_place(const torch::Tensor& A, const torch::Tensor& Bm, int64_t M, int64_t N) {
    const auto ok = [](const torch::Tensor& t) {
        return t.scalar_type() == torch::kHalf && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 &&
               sstride(t) % 8 == 0;
    };
    return ok(A) && ok(Bm) && M % 4 == 0 && N % 8 == 0;
}

void check_common(const torch::Tensor& A, const torch::Tensor& Bm, const torch::Tensor& params,
                  const torch::Tensor& emars, const torch::Tensor& rows, int64_t M, int64_t N, int64_t K, int64_t B) {
    TORCH_CHECK(A.is_cuda() && (A.scalar_type() == torch::kFloat32 || A.scalar_type() == torch::kHalf ||
                                A.scalar_type() == torch::kBFloat16) && inner_contiguous(A),
                "A: fp32, fp16 or bf16 [B, M, K], a sample's contiguous");
    TORCH_CHECK(Bm.is_cuda() && Bm.scalar_type() == A.scalar_type() && inner_contiguous(Bm),
                "B: [B, K, N] of A's dtype, a sample's contiguous");
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

// the factors' element type (TI) and their data at sample b0
#define HF_TI_SWITCH(A, Bm, b0, ...)                                                                               \
    if (A.scalar_type() == torch::kHalf) {                                                                         \
        using TI = __half;                                                                                         \
        const TI* ai = reinterpret_cast<const TI*>(A.template data_ptr<at::Half>()) + (b0) * A.stride(0);          \
        const TI* bi = reinterpret_cast<const TI*>(Bm.template data_ptr<at::Half>()) + (b0) * Bm.stride(0);        \
        __VA_ARGS__;                                                                                               \
    } else if (A.scalar_type() == torch::kBFloat16) {                                                              \
        using TI = __nv_bfloat16;                                                                                  \
        const TI* ai = reinterpret_cast<const TI*>(A.template data_ptr<at::BFloat16>()) + (b0) * A.stride(0);      \
        const TI* bi = reinterpret_cast<const TI*>(Bm.template data_ptr<at::BFloat16>()) + (b0) * Bm.stride(0);    \
        __VA_ARGS__;                                                                                               \
    } else {                                                                                                       \
        using TI = float;                                                                                          \
        const TI* ai = A.template data_ptr<float>() + (b0) * A.stride(0);                                          \
        const TI* bi = Bm.template data_ptr<float>() + (b0) * Bm.stride(0);                                        \
        __VA_ARGS__;                                                                                               \
    }

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

// The forward's operands as tensor maps of fp16 in 128-byte swizzled boxes, zeros past the extents: A [bc][rows][16]
// (rows contiguous -- `rows` a multiple of 4 --, samples `ss` halves apart) as lines of 4 rows, boxes of 32 lines (128
// rows); B [bc][16][n] (rank rows of n children contiguous) in boxes of 64 children
static CUtensorMap h_map(const void* base, bool isA, int64_t ext, int64_t ss, int64_t bc) {
    CUtensorMap m;
    cuuint64_t dimA[3] = {64, (cuuint64_t)ext / 4, (cuuint64_t)bc};
    cuuint64_t dimB[3] = {(cuuint64_t)ext, (cuuint64_t)HK, (cuuint64_t)bc};
    cuuint64_t strA[2] = {128, (cuuint64_t)ss * 2}, strB[2] = {(cuuint64_t)ext * 2, (cuuint64_t)ss * 2};
    cuuint32_t boxA[3] = {64, (cuuint32_t)HTM / 4, 1}, boxB[3] = {64, (cuuint32_t)HK, 1};
    cuuint32_t es[3] = {1, 1, 1};
    const CUresult r = tmap_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<void*>(base), isA ? dimA : dimB,
                                     isA ? strA : strB, isA ? boxA : boxB, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "the forward's tensor map: cuTensorMapEncodeTiled failed (", (int)r, ")");
    return m;
}

// The factors' fp16 operands as the kernels read them (the forward's deferred items, the backward's stages): A rows (16
// halves) `sAb` apart a sample, zeros from row am; B rank rows `bk` apart, `sBb` a sample, zeros from child bn; scaleA:
// A is the caller's (made A' in the kernels), else the copy's A'
struct HbOps { const __half* a; long sAb; int am; const __half* b; long sBb; int bk, bn, scaleA; };

template <int NT8, int NST, typename TP>
void launch_hf_fw(cudaStream_t stream, const CUtensorMap& tma, const CUtensorMap& tmb, const HbOps& op, const float* wf,
                  const float* lc, const TP* pp, long p0, int Nb, int cbs, int bs, const float4* ltf, const float4* cmf,
                  float2* part2, float2* rsh, int M, int N, int Mp, int Np, int Bc, int njt, long total) {
    using Gm = HfGeo<NT8>;
    constexpr int bytes = NST * Gm::ST + 2 * NST * 8 + 4 * HNW + 1024;   // (+ the stages' alignment)
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
    kern<<<(unsigned)grid, (HNW + 1) * 32, bytes, stream>>>(tma, tmb, op.scaleA, op.a, op.sAb, op.am, op.b, op.sBb, op.bk,
                                                           op.bn, wf, lc, pp, p0, Nb, cbs, bs, ltf, cmf, part2, rsh, M,
                                                           N, Mp, Np, Bc, njt, total);
}

// The tiles' theta fragments of a node vector (`hf_theta_kernel`): {LTF, CMF, CMR}
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
    auto CMR = torch::empty({(long)nbT * HTM}, params.options().dtype(torch::kFloat32));
    HF_TP_SWITCH(params, hf_theta_kernel<TP, NT8><<<(unsigned)nbT, 256, 0, stream>>>(
        pp, p0, (int)Nb, (int)cbs, (int)bs, reinterpret_cast<float4*>(LTF.template data_ptr<float>()),
        reinterpret_cast<float4*>(CMF.template data_ptr<float>()), CMR.template data_ptr<float>(), (int)M, (int)N, Np));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {LTF, CMF, CMR};
}

// The forward of one node vector: its node values into `node_mars` rows `nid0 .. nid0 + M`; returns {log Z [M, B], AP,
// BP}: the converted copy of the factors where they are not read in place and `keep` (the backward's operands then:
// AP [B, Mp, 16] = A', BP [B, 16, Np] = B in fp16), else empty
template <int NT8>
std::vector<torch::Tensor> forward_impl(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb,
                                        int64_t cbs, int64_t bs, torch::Tensor emars, torch::Tensor rows,
                                        torch::Tensor node_mars, int64_t nid0, int64_t M, int64_t N, torch::Tensor LTF,
                                        torch::Tensor CMF, torch::Tensor CMR, bool keep) {
    using Gm = HfGeo<NT8>;
    constexpr int NST = 3, TN = Gm::TN;
    const int64_t B = node_mars.size(1);
    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int Mp = (int)((M + HTM - 1) / HTM * HTM), Np = (int)((N + TN - 1) / TN * TN), njt = Np / TN, nit = Mp / HTM;
    auto lzs = torch::empty({M, B}, emars.options());
    const long* rw = reinterpret_cast<const long*>(rows.template data_ptr<int64_t>());
    const bool direct = h_in_place(A, Bm, M, N);
    // a sample's working set: its fp16 operands, WF, lcs, the partials (z, s; a rare item's own pairs apart)
    const int64_t BC = sample_chunk(B, 2.0 * HK * (Mp + Np) + 4.0 * njt * Gm::WFW + 4.0 * Np + 8.0 * njt * Mp);
    torch::Tensor AP, BP;                                      // (the converted copy: the batch's where kept)
    if (!direct) {
        const int64_t cb = keep ? B : BC;
        AP = torch::empty({cb, (long)Mp, HK}, emars.options().dtype(torch::kHalf));
        BP = torch::empty({cb, HK, (long)Np}, emars.options().dtype(torch::kHalf));
    }
    auto WF = torch::empty({BC, (long)njt * Gm::WFW}, emars.options());
    auto lcs = torch::empty({BC, (long)Np}, emars.options());
    auto part2 = torch::empty({BC, (long)njt * Mp * 2}, emars.options());      // (every item's (z, s))
    auto rshs = torch::empty({BC, (long)njt * Mp * 2}, emars.options());       // (rare items' own shifts only)
    const int nbT = njt * nit;                                 // (the tiles' theta fragments: given, or made here)
    if (LTF.numel() == 0) {
        auto tc = theta_impl<NT8>(params, p0, Nb, cbs, bs, M, N);
        LTF = tc[0];
        CMF = tc[1];
        CMR = tc[2];
    }
    TORCH_CHECK(LTF.numel() == (long)nbT * HNW * NT8 * 32 * 4 && CMF.numel() == (long)nbT * HNW * 32 * 4 &&
                CMR.numel() == (long)nbT * HTM, "theta fragments: exp_sum_h_theta's, for this node vector and tile width");
    TORCH_CHECK((int64_t)BC * njt * Mp < (int64_t)INT_MAX, "the partials' index: 32 bits");
    const float4* ltf = reinterpret_cast<const float4*>(LTF.template data_ptr<float>());
    const float4* cmf = reinterpret_cast<const float4*>(CMF.template data_ptr<float>());
    float2* pp2 = reinterpret_cast<float2*>(part2.template data_ptr<float>());
    float2* prs = reinterpret_cast<float2*>(rshs.template data_ptr<float>());
    const int nbA = direct ? 0 : (Mp * 8 + 255) / 256, nbB = direct ? 0 : (HK * (Np / 2) + 255) / 256;
    const int nbW = (njt + 7) / 8;
    const int64_t sA = A.stride(0), sB = Bm.stride(0);
    for (int64_t b0 = 0; b0 < B; b0 += BC) {
        const int Bc = (int)std::min<int64_t>(BC, B - b0);
        const int64_t c0 = keep ? b0 : 0;                      // (the chunk's first sample in the copy)
        unsigned* ap = direct ? nullptr : reinterpret_cast<unsigned*>(AP.data_ptr<at::Half>()) + c0 * Mp * 8;
        unsigned* bp = direct ? nullptr : reinterpret_cast<unsigned*>(BP.data_ptr<at::Half>()) + c0 * HK * (Np / 2);
        HF_TI_SWITCH(A, Bm, b0, hf_fw_prep_kernel<TI, NT8><<<dim3(nbA + nbB + nbW, (unsigned)Bc), 256, 0, stream>>>(
            ai, bi, ap, bp, WF.template data_ptr<float>(), lcs.template data_ptr<float>(), emars.template data_ptr<float>(),
            rw, (int)M, (int)N, Mp, Np, nbA, nbB, (int)B, (int)b0, (long)sA, (long)sB));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const CUtensorMap tma = direct ? h_map(reinterpret_cast<const __half*>(A.data_ptr<at::Half>()) + b0 * sA, true, M,
                                               sstride(A), Bc)
                                       : h_map(ap, true, Mp, (int64_t)Mp * HK, Bc);
        const CUtensorMap tmb = direct ? h_map(reinterpret_cast<const __half*>(Bm.data_ptr<at::Half>()) + b0 * sB, false,
                                               N, sstride(Bm), Bc)
                                       : h_map(bp, false, Np, (int64_t)HK * Np, Bc);
        const HbOps op = direct ? HbOps{reinterpret_cast<const __half*>(A.data_ptr<at::Half>()) + b0 * sA, (long)sstride(A),
                                        (int)M, reinterpret_cast<const __half*>(Bm.data_ptr<at::Half>()) + b0 * sB,
                                        (long)sstride(Bm), (int)N, (int)N, 1}
                                : HbOps{reinterpret_cast<const __half*>(ap), (long)Mp * HK, Mp,
                                        reinterpret_cast<const __half*>(bp), (long)HK * Np, Np, Np, 0};
        const long total = (long)njt * nit * Bc;
        HF_TP_SWITCH(params, launch_hf_fw<NT8, NST, TP>(stream, tma, tmb, op, WF.template data_ptr<float>(),
                                                        lcs.template data_ptr<float>(), pp, p0, (int)Nb, (int)cbs,
                                                        (int)bs, ltf, cmf, pp2, prs, (int)M, (int)N, Mp, Np, Bc, njt,
                                                        total));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // (the combine: 2 lanes an output, 4 where that grid would leave an SM fewer than 2 blocks)
        const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        const bool wide = (long)((M + 31) / 32) * ((Bc + 3) / 4) < 2L * sms;
        auto comb = wide ? hf_comb_kernel<4> : hf_comb_kernel<2>;
        comb<<<dim3((unsigned)((M + 31) / 32), (unsigned)((Bc + (wide ? 1 : 3)) / (wide ? 2 : 4))), 256, 0, stream>>>(
            pp2, prs, CMR.template data_ptr<float>(), WF.template data_ptr<float>(), Gm::WFW, 8 * NT8, njt,
            node_mars.template data_ptr<float>(), nid0, lzs.template data_ptr<float>(), (int)M, Mp, Bc, (int)B, (int)b0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const auto none = torch::empty({0}, emars.options().dtype(torch::kHalf));
    return {lzs, keep && !direct ? AP : none, keep && !direct ? BP : none};
}

// The tiles' theta fragments of a node vector's parameters for the forward at tile width `tn` (made once a forward
// pass for every node vector reading the same run: tied copies): {LTF, CMF}
std::vector<torch::Tensor> exp_sum_h_theta(torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs, int64_t bs,
                                           int64_t M, int64_t N, int64_t tn) {
    TORCH_CHECK(params.scalar_type() == torch::kFloat32 || params.scalar_type() == torch::kBFloat16, "params: fp32 or bf16");
    TORCH_CHECK(tn == 64, "tn: 64");
    return theta_impl<8>(params, p0, Nb, cbs, bs, M, N);
}

// The forward of one node vector (`theta`: exp_sum_h_theta's fragments of its parameters, or empty: made here; `keep`:
// return the converted copy of factors not read in place, for the backward): {log Z [M, B], AP, BP}
std::vector<torch::Tensor> exp_sum_h_forward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0,
                                             int64_t Nb, int64_t cbs, int64_t bs, torch::Tensor emars,
                                             torch::Tensor rows, torch::Tensor node_mars, int64_t nid0, int64_t M,
                                             int64_t N, int64_t K, int64_t tn, std::vector<torch::Tensor> theta,
                                             bool keep) {
    const int64_t B = node_mars.size(1);
    check_common(A, Bm, params, emars, rows, M, N, K, B);
    TORCH_CHECK(node_mars.scalar_type() == torch::kFloat32 && node_mars.is_contiguous(), "node_mars");
    TORCH_CHECK(tn == 64, "tn: 64");
    const bool given = theta.size() == 3;
    torch::Tensor ltf = given ? theta[0] : torch::empty({0}, emars.options());
    torch::Tensor cmf = given ? theta[1] : torch::empty({0}, emars.options());
    torch::Tensor cmr = given ? theta[2] : torch::empty({0}, emars.options());
    return forward_impl<8>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_mars, nid0, M, N, ltf, cmf, cmr, keep);
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
void launch_hb_bw(cudaStream_t stream, const HbOps& op, const TP* pp, long p0, int Nb, int cbs, int bs, const float* LU,
                  const float* LC, float* ef, float* pf, long pf0, float* dnp, long dn0, float* ga, long sga, int ma,
                  const CUtensorMap& tmb, const float4* sc, int M, int N, int Mp, int Np, int B, int nct, long units) {
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
    kern<<<(unsigned)grid, 256, bytes, stream>>>(op.a, op.sAb, op.am, op.b, op.sBb, op.bk, op.bn, op.scaleA, pp, p0, Nb,
                                                cbs, bs, LU, LC, ef, pf, pf0, dnp, dn0, ga, sga, ma, tmb, sc, M, N, Mp, Np,
                                                B, nct, units);
}

// The backward of one node vector: the element flows of its children log-added into `element_flows` (the caller sets
// them to -inf first, once a layer), and -- each where its tensor is given (not empty) -- F+ added into `param_flows`
// from `pf0`, W into `denom` from `dn0`, dLL/dA and dLL/dB into the gradient buffers (fp32, the caller's layout, [B,
// M, K], [B, K, N]): `lowrank_exp_sum.cu`'s `exp_sum_backward` at rank 16 with phi of the fp16-rounded factors. Its
// operands: `copy` = the forward's {AP, BP} where given, else the factors in place (fp16, aligned: `h_in_place`), else
// converted here a chunk at a time. `init`: the gradient buffers hold anything (their first writer: every element
// written, not added into)
void exp_sum_h_backward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs,
                        int64_t bs, torch::Tensor emars, torch::Tensor rows, torch::Tensor node_flows,
                        torch::Tensor node_mars, int64_t nid0, torch::Tensor lzs, torch::Tensor element_flows,
                        torch::Tensor pflows, int64_t pf0, torch::Tensor denom, int64_t dn0, torch::Tensor gA,
                        torch::Tensor gB, int64_t M, int64_t N, int64_t K, std::vector<torch::Tensor> copy, bool init) {
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
    const bool given = copy.size() == 2 && copy[0].numel() > 0;
    if (given)
        TORCH_CHECK(copy[0].scalar_type() == torch::kHalf && copy[0].is_contiguous() && copy[0].numel() == B * Mp2 * HK &&
                    copy[1].scalar_type() == torch::kHalf && copy[1].is_contiguous() && copy[1].numel() == B * HK * Np2,
                    "copy: the forward's {AP [B, Mp, 16], BP [B, 16, Np]} of these factors");
    const bool direct = !given && h_in_place(A, Bm, M, N), convert = !given && !direct;
    // (a chunk's operands within the L2 beside its fp32 traffic -- the gradients' adds, a conversion's reads -- however
    // they come: read from the copy or in place, the operands of a chunk the whole batch's would be evicted between
    // their 16 / 8 readers, +28% at B1024)
    const int64_t BC = sample_chunk(B, 2.0 * HK * (Mp2 + Np2) + 8.0 * K * (M + N));
    torch::Tensor AP2, BP2;                                    // (converted here: a chunk's copy)
    if (convert) {
        AP2 = torch::empty({BC, (long)Mp2 * HK}, emars.options().dtype(torch::kHalf));
        BP2 = torch::empty({BC, (long)HK * Np2}, emars.options().dtype(torch::kHalf));
    }
    auto LU2 = torch::empty({BC, Mp2, 2}, emars.options());
    auto LC2 = torch::empty({BC, njt2, 36}, emars.options());
    auto efs2 = torch::empty({BC, Np2}, emars.options());
    auto SC2 = torch::empty({BC, 4}, emars.options());          // (the samples' scales)
    float4* sc2 = reinterpret_cast<float4*>(SC2.data_ptr<float>());
    const int nbA2 = convert ? (int)(((long)Mp2 * 8 + 255) / 256) : 0;
    const int nbB2 = convert ? (int)(((long)HK * (Np2 / 2) + 255) / 256) : 0;
    // the gradients' bulk adds take the caller's buffers where each sample's rows lie 16-byte aligned (dA's [M, 16]
    // rows, dB's 16 rank rows of N); else padded temporaries [B, Mp2, 16], [B, 16, Np2], added in afterwards
    const auto al16 = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
    const bool dgrad = !grads || (al16(gA) && gA.stride(0) % 4 == 0 && al16(gB) && gB.stride(0) % 4 == 0 && N % 4 == 0);
    torch::Tensor tA2, tB2;
    if (grads && !dgrad) {
        tA2 = torch::zeros({BC, Mp2, HK}, emars.options());
        tB2 = torch::zeros({BC, HK, Np2}, emars.options());
    }
    const long sga = !grads ? 0L : dgrad ? (long)gA.stride(0) : (long)Mp2 * HK;
    const long sgb = !grads ? 0L : dgrad ? (long)gB.stride(0) : (long)HK * Np2;
    const int ma = dgrad ? (int)M : Mp2, ldb = dgrad ? (int)N : Np2, nbw = dgrad ? (int)N : Np2;
    for (int64_t b0 = 0; b0 < B; b0 += BC) {
        const int Bc = (int)std::min<int64_t>(BC, B - b0);
        int lbc = 0;
        while ((1 << lbc) < Bc) lbc++;
        unsigned* ap = convert ? reinterpret_cast<unsigned*>(AP2.data_ptr<at::Half>()) : nullptr;
        unsigned* bp = convert ? reinterpret_cast<unsigned*>(BP2.data_ptr<at::Half>()) : nullptr;
        // (`init`: the chunk's slices of the caller's gradient buffers zeroed by the prep kernel, just ahead of the
        // adds -- their lines in the L2 when the adds come, no DRAM round trip of zeros: the previous chunk's final
        // lines leave as they come in (in the main kernel's background instead: as slow, its L2 shared); through
        // temporaries, those copied in rather than added)
        const int nbZ = init && grads && dgrad ? (int)((K / 4 * (M + N) + 4095) / 4096) : 0;
        float4* zA = nbZ ? reinterpret_cast<float4*>(gA.data_ptr<float>() + b0 * gA.stride(0)) : nullptr;
        float4* zB = nbZ ? reinterpret_cast<float4*>(gB.data_ptr<float>() + b0 * gB.stride(0)) : nullptr;
        const int nbg = (Bc + 31) / 32;
        const long nprep = (long)(nbZ + nbA2 + nbB2) * Bc + (Mp2 / 32 + njt2) * nbg + (Bc + 7) / 8;
        HF_TI_SWITCH(A, Bm, b0, hb_bw_prep_kernel<TI><<<(unsigned)nprep, 256, 0, stream>>>(
            ai, bi, ap, bp, LU2.data_ptr<float>(), LC2.data_ptr<float>(), efs2.data_ptr<float>(), sc2,
            node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), nid0, lzs.data_ptr<float>(),
            emars.data_ptr<float>(), rw, (int)M, (int)N, Mp2, Np2, nbA2, nbB2, Bc, (int)B, (int)b0,
            (long)A.stride(0), (long)Bm.stride(0), zA, nbZ ? gA.stride(0) / 4 : 0L, zB, nbZ ? gB.stride(0) / 4 : 0L,
            nbZ));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (grads && !dgrad) {
            tA2.zero_();
            tB2.zero_();
        }
        HbOps op;
        if (given)
            op = HbOps{reinterpret_cast<const __half*>(copy[0].data_ptr<at::Half>()) + b0 * Mp2 * HK, (long)Mp2 * HK, Mp2,
                       reinterpret_cast<const __half*>(copy[1].data_ptr<at::Half>()) + b0 * HK * Np2, (long)HK * Np2,
                       Np2, Np2, 0};
        else if (direct)
            op = HbOps{reinterpret_cast<const __half*>(A.data_ptr<at::Half>()) + b0 * A.stride(0), (long)sstride(A),
                       (int)M, reinterpret_cast<const __half*>(Bm.data_ptr<at::Half>()) + b0 * Bm.stride(0),
                       (long)sstride(Bm), (int)N, (int)N, 1};
        else
            op = HbOps{reinterpret_cast<const __half*>(ap), (long)Mp2 * HK, Mp2, reinterpret_cast<const __half*>(bp),
                       (long)HK * Np2, Np2, Np2, 0};
        float* ga2 = !grads ? nullptr : dgrad ? gA.data_ptr<float>() + b0 * gA.stride(0) : tA2.data_ptr<float>();
        float* gb2 = !grads ? nullptr : dgrad ? gB.data_ptr<float>() + b0 * gB.stride(0) : tB2.data_ptr<float>();
        CUtensorMap tmb;
        if (grads) tmb = db_map(gb2, nbw, ldb, sgb, Bc);
        else std::memset(&tmb, 0, sizeof(tmb));
        const long units = (long)nit2 * nct * Bc;
        HF_TP_SWITCH(params, HB_FLAGS_SWITCH(flows, dn, grads, launch_hb_bw<TP, FL, DE, GR>(stream, op, pp, p0, (int)Nb,
            (int)cbs, (int)bs, LU2.data_ptr<float>(), LC2.data_ptr<float>(), efs2.data_ptr<float>(), pf, pf0, dnp, dn0,
            ga2, sga, ma, tmb, sc2, (int)M, (int)N, Mp2, Np2, Bc, nct, units)))
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hb_ef_epi_kernel<<<(unsigned)((((long)N << lbc) + 255) / 256), 256, 0, stream>>>(
            efs2.data_ptr<float>(), rw, element_flows.data_ptr<float>(), (int)N, Np2, Bc, lbc, (int)B, (int)b0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (grads && !dgrad) {                                 // (the temporaries' sums into the caller's buffers)
            if (init) {
                gA.narrow(0, b0, Bc).copy_(tA2.narrow(0, 0, Bc).narrow(1, 0, M));
                gB.narrow(0, b0, Bc).copy_(tB2.narrow(0, 0, Bc).narrow(2, 0, N));
            } else {
                gA.narrow(0, b0, Bc).add_(tA2.narrow(0, 0, Bc).narrow(1, 0, M));
                gB.narrow(0, b0, Bc).add_(tB2.narrow(0, 0, Bc).narrow(2, 0, N));
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_sum_h_theta", &exp_sum_h_theta,
          "the forward's tiles' theta fragments of a node vector's parameters (make once, pass to exp_sum_h_forward)");
    m.def("exp_sum_h_forward", &exp_sum_h_forward,
          "a node vector's values (and log Z, the factors' fp16 copy where kept) under the exp_sum scale of fp16-rounded "
          "factors, the scale made tile by tile");
    m.def("exp_sum_h_backward", &exp_sum_h_backward,
          "a node vector's element flows, F+, W and dLL/dA, dLL/dB under the exp_sum scale of fp16-rounded factors");
}
