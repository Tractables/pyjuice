// `LowRank`'s "exp_sum" form at the default precision, its scale from fp16-ROUNDED factors: phi_b = A'_b @ B_b with
// A' = log2 e A and B each rounded to fp16 once, the product exact in fp32 accumulation (one m16n8k16 MMA a 16 x 8 tile
// a 16 ranks: the factors' K padded with zeros to KP = 16, 32 or 64) -- the exp_sum of the rounded factors, the
// forward's and the backward's alike -- where `lowrank_exp_sum.cu`'s bf16 path splits each factor into hi + lo (three
// MMAs) to make the unrounded one. The rest is that file's formulation:
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
// batch stride; K 1 .. 64). fp16 ones of a padded rank (K = KP) are read IN PLACE where each sample's block is 16-byte
// aligned, M a multiple of 64 / KP and N of 8 (`h_in_place`), A made A' in the kernels' registers (`scale_h2`); others
// -- fp32, bf16, fp16 of another rank or not so aligned -- through a COPY in fp16 the prologues make (AP [B, Mp, KP] =
// A', BP [B, KP, Np] = B, zero-padded: `conv_copy`), the forward's kept for the backward where asked (`keep`). The
// kernels read either as they lie: plain row-major fp16, no fragment order -- ldmatrix of the staged rows.
//
// FORWARD (`hf_fw_kernel`): a CTA holds a tile of 128 nodes x TN children (TN = 8 NT8 = 64) -- eight consumer warps of
// 16 nodes, each warp all TN children (NT8 n-tiles of 8, KP / 16 k-steps each), its pairs' log2 theta (less the row's
// frame) in registers, the first MMA's C operand -- and loops over its samples; a producer warp streams each sample's
// operands into an NST-stage ring of shared memory (full / empty mbarriers): A's 128 rows and B's 64 children as boxes
// of tensor maps (TMA, 128-byte swizzled: ldmatrix rows conflict-free), their factors w and the tile's mc
// (`hf_fw_prep_kernel`'s WF) by a bulk copy. Two CTAs an SM at KP 16 and 32, one at 64 (its 25 KB stages). Bound by its
// compute, not its loads: the A / B boxes skipped, KP 64 -15%, 32 -10%, 16 -8% (B1024; KP 64 ~1.45x its tensor floor).
//
// BACKWARD: KP 16 `hb_bw_kernel` (pipelined: dB of a sample inside the next one's element loop), KP 32 and 64
// `hbk_bw_kernel` (two barriers a sample: the 16-rank pipeline's buffers do not fit beside the wider operands).
//
// Layouts: the circuit's buffers [rows, batch], batch innermost; the shared parameters block-major
// (`LowRank._geometry`'s grid), edge block (i / bs, j / cbs) from `p0 + ((i / bs) Nb + j / cbs) cbs bs` on, child then
// node within it. Ranks padded to KP; node and child counts padded to whole tiles (zeros, -inf past the real ones).

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

constexpr int HK = 16;                         // the rank `hb_bw_kernel` takes (the others: KP template arguments)
constexpr int HTM = 128;                       // a forward tile's nodes (eight warps of 16)
constexpr int HNW = 8;                         // the forward's consumer warps
#define NEG (-CUDART_INF_F)                    // (device code only)
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

// Programmatic dependent launch: a pass's launches chained (`launch_h`'s `pdl`), each kernel waiting before its first
// read of what the launch before wrote -- pointers to such data not `const __restrict__` (an ld.global.nc may be issued
// above the wait) -- and letting the next launch at its start; a no-op where launched without the attribute
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" :::); }

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

// A (shift, sum) pair made NORMAL: its sum's binary exponent moved into the shift (s = f 2^e, f in [1, 2): (m + e,
// f); 0, subnormal, inf, NaN as they are) -- a fold of pairs so made loses no term over 2^-126 of the larger. The
// combine's pairs need it: a main-loop tile's sums lie in [2^-92, 2^100] under its frames, a shifted tile's S down to
// 2^-97 under Z's max; unmade, a pair 2^-132 under the other's shift flushed with its 2^(m2 - n) though its term was
// 2^-33 of the sum, the other's sum 2^-21 (S 2.8e-4 short). (e by a float subtraction: 2^23 + ex less 2^23 + 127)
__device__ __forceinline__ void norm_ms(float& m, float& s) {
    const unsigned u = __float_as_uint(s), ex = u >> 23;           // (s >= 0: its biased exponent)
    const bool nrm = ex - 1u < 254u;
    const float e = __uint_as_float(0x4b000000u | ex) - 8388735.0f;
    m = nrm ? m + e : m;
    s = nrm ? __uint_as_float((u & 0x007fffffu) | 0x3f800000u) : s;
}
// a running (max, sum) folded with a pair made normal first
__device__ __forceinline__ void fold_n(float& m, float& s, float m2, float s2) {
    norm_ms(m2, s2);
    fold_ms(m, s, m2, s2);
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
// from the caller's fp32 or bf16 (A' rounded once from fp32) or fp16 factors (scale_h2); zero where not live -- each
// of the pair's two ranks apart (a rank past the factors' K: their padding) -- (scalar loads: a view may be 4-byte
// aligned)
// (the second rank's address src + 1 only where it is live: a rank row's last element may end the tensor)
__device__ __forceinline__ unsigned conv_a(const float* src, bool l0, bool l1) {
    return pack_h2(l0 ? __ldg(src) * LOG2E : 0.f, l1 ? __ldg(src + (l1 ? 1 : 0)) * LOG2E : 0.f);
}
__device__ __forceinline__ unsigned conv_a(const __half* src, bool l0, bool l1) {
    const __half z = __float2half_rn(0.f);
    return scale_h2(pack_hh(l0 ? src[0] : z, l1 ? src[l1 ? 1 : 0] : z));
}
__device__ __forceinline__ unsigned conv_a(const __nv_bfloat16* src, bool l0, bool l1) {
    return pack_h2(l0 ? __bfloat162float(src[0]) * LOG2E : 0.f,
                   l1 ? __bfloat162float(src[l1 ? 1 : 0]) * LOG2E : 0.f);
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

// fp16 m16n8k16 accumulating into D (the four A registers given apart)
__device__ __forceinline__ void mma_h16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
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

// ============================================================================================== forward
// A stage, per (tile, sample) item: the tile's 128 rows of A' -- or of the caller's fp16 A read in place, made A' in
// the consumers' registers (`scale_h2`) -- as [node][rank KP] fp16, a box of a tensor map of 128-byte lines (64 / KP
// nodes a line: a box of 32-byte rows costs the copy engine a request a row, 4x the lines'); its TN children of B as
// [rank KP][child TN]; both 128-byte swizzled (each lane's ldmatrix row in a bank group of its own); and WF
// [jt][72] float: [tq][2 NT8] the factors w_j = 2^(lc'_j - mc) of a lane's children (TN jt + 8 t + 2 tq + p at 2 t + p),
// then mc (the tile's largest lc'; -inf: no child of the tile has a value) at 8 NT8, padded to 16 bytes. KP: the
// factors' rank padded (16, 32, 64: KP / 16 k-steps of the MMAs; zero ranks past the factors' K)
template <int KP, int NT8>
struct HfGeo {
    static_assert(NT8 == 8, "a tile's children: one 128-byte swizzled box");
    static_assert(KP == 16 || KP == 32 || KP == 64, "ranks padded to 16, 32 or 64");
    static constexpr int TN = 8 * NT8;
    static constexpr int KS = KP / 16;                                       // the MMAs' k-steps
    static constexpr int WFW = 4 * 2 * NT8 + 4;                              // WF floats a tile (mc at 8 NT8)
    static constexpr int A_BYTES = HTM * KP * 2, B_BYTES = KP * TN * 2, W_BYTES = WFW * 4;
    static constexpr int ST = (A_BYTES + B_BYTES + W_BYTES + 1023) / 1024 * 1024;   // a stage (the boxes 1024-aligned)
    static constexpr int NST = 3;                                            // the ring's stages
    static constexpr int CTAS = KP == 64 ? 1 : 2;                           // an SM's (the stages' shared memory)
};

// Where the 16 bytes an ldmatrix row reads lie in a stage's 128-byte swizzled boxes (a line's 16-byte piece c at c ^
// line % 8): A's 16-byte piece h of node `row` (line row / RPL, piece (row % RPL) KP / 8 + h; k-step s's pieces 2 s,
// 2 s + 1: the address of s = 0's XOR 32 s) and B's children 8 c .. 8 c + 7 of rank row k (line k, piece c; k-step s
// rows 16 s on: + 2048 s)
template <int KP>
__device__ __forceinline__ int hf_a_at(int row, int h) {
    if constexpr (KP == 16) {
        return (row >> 2) * 128 + (((2 * (row & 3) + h) ^ (row >> 2)) & 7) * 16;
    } else {
        constexpr int RPL = 64 / KP, LR = RPL == 2 ? 1 : 0;
        const int line = row >> LR;
        return line * 128 + ((((row & (RPL - 1)) * (KP / 8) + h) ^ line) & 7) * 16;
    }
}
__device__ __forceinline__ int hf_b_at(int k, int c) { return k * 128 + ((c ^ k) & 7) * 16; }
// ... B's pieces of the n-tile pairs t2 = 0, 2, 4, 6 a lane reads (ldmatrix.trans: rank row k = 8 (lq & 1) + lr, piece
// t2 + (lq >> 1)): this base XOR 16 t2 -- (t2 + h) ^ k = (t2 ^ (k & 6)) | ((h ^ k) & 1), the base's bits 5, 6 those of
// 16 (k & 6) and the stage 1024-byte aligned: one XOR an access, no lane math in the loop
__device__ __forceinline__ int hf_b_base(int k, int h) { return k * 128 + 16 * ((h ^ k) & 1) + 16 * (k & 6); }

// The tiles' theta in the forward consumers' order, a block a tile (a thread a consumer lane): LTF [tile][warp][t][lane]
// float4 = the lane's log2 theta less its row's frame (the largest log2 theta of the row in the tile; 0 for a row
// without one), the MMA's C operand -- the main kernel's tile change a run of 16-byte loads, no index math, no logs --,
// and the frames for the comb kernel, CMR [tile][row] (-inf: the row has no theta in the tile). Made once for every
// sample (and every node vector of a forward pass reading the same run: `exp_sum_h_theta`)
template <typename TP, int NT8>
__global__ void __launch_bounds__(256) hf_theta_kernel(const TP* __restrict__ params, long p0, int Nb, int cbs, int bs,
                                                       float4* __restrict__ LTF, float* __restrict__ CMR, int M, int N,
                                                       int Np) {
    using Gm = HfGeo<16, NT8>;                             // (the tile's children: any rank's)
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
        if (tq == 0) {                                     // (the rows' frames for the comb kernel: CMR [tile][row])
            CMR[(long)tile * HTM + 16 * w + g] = live[0] ? cm[0] : NEG;
            CMR[(long)tile * HTM + 16 * w + g + 8] = live[1] ? cm[1] : NEG;
        }
    }
}

// A block of sample b's converted copy (the factors' fp16 operands where they are not read in place, the forward's and
// the backward's alike) at KP ranks (the factors' K padded: 16, 32, 64): blocks 0 .. nbA over AP [Mp, KP] = A' (a
// thread a pair of ranks of a node), then nbB over BP [KP, Np] = B (a thread a pair of children of a rank); zero past
// the node vector's nodes / children and past the factors' K ranks. False: blk is past them
template <int KP, typename TI>
__device__ __forceinline__ bool conv_copy(int blk, int nbA, int nbB, const TI* __restrict__ A, const TI* __restrict__ Bm,
                                          unsigned* __restrict__ AP, unsigned* __restrict__ BP, int b, int M, int N, int K,
                                          int Mp, int Np, long sA, long sB) {
    constexpr int KW = KP / 2;                               // (a row's words)
    if (blk < nbA) {
        const int e = blk * 256 + threadIdx.x, i = e / KW, k = 2 * (e % KW);   // (e = KW i + k / 2)
        if (e < Mp * KW) {
            const bool l0 = i < M && k < K, l1 = i < M && k + 1 < K;
            AP[(long)b * Mp * KW + e] = conv_a(A + (long)b * sA + (long)min(i, M - 1) * K + (l0 ? k : 0), l0, l1);
        }
        return true;
    }
    blk -= nbA;
    if (blk < nbB) {
        const int e = blk * 256 + threadIdx.x;               // (e = k Np / 2 + jp)
        if (e < KP * (Np / 2)) {
            const int k = e / (Np / 2), j = 2 * (e - k * (Np / 2));
            BP[(long)b * KP * (Np / 2) + e] = conv_b(Bm + (long)b * sB + (long)min(k, K - 1) * N + min(j, N - 1),
                                                     j < N && k < K, j + 1 < N && k < K);
        }
        return true;
    }
    return false;
}

// The prologue of a chunk, a grid row a sample: the converted copy's blocks (none where the factors are read in
// place), then WF's tiles (a warp a tile) with lcs [B, Np] = lc' (the exact fallback's)
template <int KP, typename TI, int NT8>
__global__ void __launch_bounds__(256) hf_fw_prep_kernel(
        const TI* __restrict__ A, const TI* __restrict__ Bm, unsigned* __restrict__ AP, unsigned* __restrict__ BP,
        float* __restrict__ WF, float* __restrict__ lcs, const float* emars, const long* __restrict__ rows,
        int M, int N, int K, int Mp, int Np, int nbA, int nbB, int Bs, int b0, long sA, long sB) {
    pdl_wait();                                                // (the child values: the launch before's)
    pdl_trigger();
    using Gm = HfGeo<KP, NT8>;
    const int b = blockIdx.y, njt = Np / Gm::TN;
    const int blk = blockIdx.x;
    if (conv_copy<KP>(blk, nbA, nbB, A, Bm, AP, BP, b, M, N, K, Mp, Np, sA, sB)) return;
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

// A main-loop item of a live row (its frame cm > -inf) is RARE where its sums are not exact to fp32: z out of
// [2^-92, 2^100] -- or inf, NaN -- (`hf_out`), or s under 2^-92 where a child of the tile has a value (mc > -inf; 2^-92:
// a row's largest term over 2^-97, so any term flushed under 2^-126 is under 2^-29 of it). z, s >= 0: their bits
// compared as integers, NaN / inf out -- bitwise, no branch
__device__ __forceinline__ bool hf_rare_z(float z, float cm) { return (cm > NEG) & hf_out(z); }
__device__ __forceinline__ bool hf_rare_s(float s, float cm, float mc) {
    return (cm > NEG) & (mc > NEG) & (__float_as_int(s) < 0x11800000);
}

// The forward's rare paths for one item -- row i of chunk sample b, the children of tile jt -- from the main loop's
// sums (z, s) under the shifts (cm, cm + mc): `rz` the shifted path (Z's and S's sums under the row's largest lw'),
// `rs` -- or S under 2^-97 after the shifted path -- S exactly (x' = lw' + lc' under its own max). The deferred pass's
// arithmetic the main kernel had for its rows: lw' made again a tile at a time by the main loop's MMA, row i's A' in
// every row of its A fragment, its log2 theta the C operand (each quad the row: a lane children 8 t + 2 tq, + 1 of
// n-tile t), from the operands as the main kernel's stages hold them (A rows `sAg` a sample, zeros from row am, made
// A' by scale_h2 where `scaleA`; B rank rows bk apart, `sBg` a sample, zeros from child bn), theta from the
// parameters, w and lc' from the prologue's WF and lcs. Returns the item's (shift, sum) pairs, Z's and S's, in every
// lane (rz, rs warp-uniform)
template <typename TP>
struct HfRare {                                           // (`hf_rare_row`'s operands: a comb kernel parameter)
    const __half* Ag; long sAg; int am;
    const __half* Bg; long sBg; int bk, bn, scaleA;
    const float* WF; int wfw;
    const float* lcs;
    const TP* params; long p0; int Nb, cbs, bs, M, N, Np, njt;
};
template <typename TP, int KP, int NT8>
__device__ __forceinline__ float4 hf_rare_row(const HfRare<TP>& ra, int b, int jt, int i, float cm, float mc, float z,
                                              float s, bool rz, bool rs) {
    constexpr int KS = KP / 16;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, j0 = jt * 8 * NT8;
    uint4 af[KS];                                         // (row i's A' of k-step s: ranks 16 s + 2 tq, + 1, + 8, + 9)
#pragma unroll
    for (int ks = 0; ks < KS; ks++) {
        const __half* ar = ra.Ag + (long)b * ra.sAg + (long)i * KP + 16 * ks + 2 * tq;
        unsigned lo = i < ra.am ? *reinterpret_cast<const unsigned*>(ar) : 0u;
        unsigned hi = i < ra.am ? *reinterpret_cast<const unsigned*>(ar + 8) : 0u;
        if (ra.scaleA) {
            lo = scale_h2(lo);
            hi = scale_h2(hi);
        }
        af[ks] = make_uint4(lo, lo, hi, hi);
    }
    const __half* const bg = ra.Bg + (long)b * ra.sBg;
    auto lw = [&](int t, float (&x)[4]) {                  // lw' (unframed) of the lane's children of n-tile t: x[0, 1]
        float c[4];
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const float th = theta_at(ra.params, ra.p0, i, j0 + 8 * t + 2 * tq + e, ra.M, ra.N, ra.Nb, ra.cbs, ra.bs);
            c[e] = c[e + 2] = th > 0.f ? __log2f(th) : NEG;
        }
        const int j = j0 + 8 * t + g;                      // (n-tile t's B fragment: the lane's child, its rank pairs)
        const __half zh = __float2half_rn(0.f);
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            __half v[4];
#pragma unroll
            for (int q = 0; q < 4; q++)
                v[q] = j < ra.bn ? bg[(long)(16 * ks + 2 * tq + (q & 1) + 8 * (q >> 1)) * ra.bk + j] : zh;
            if (ks == 0) mma_h_c(x, af[0], pack_hh(v[0], v[1]), pack_hh(v[2], v[3]), c);
            else mma_h16(x, af[ks].x, af[ks].y, af[ks].z, af[ks].w, pack_hh(v[0], v[1]), pack_hh(v[2], v[3]));
        }
    };
    float za = z > 0.f ? cm : NEG, zs = z, sa = s > 0.f ? cm + mc : NEG, ss = s;   // (the main loop's pairs)
    if (rz) {                                             // (the shifted path: a row max)
        float m = NEG;
        for (int t = 0; t < NT8; t++) {
            float x[4];
            lw(t, x);
            m = fmaxf(fmaxf(m, x[0]), x[1]);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
        const float sh = m == NEG ? 0.f : m;
        const float* const wv = ra.WF + ((long)b * ra.njt + jt) * ra.wfw + tq * 2 * NT8;
        float z2 = 0.f, s2 = 0.f;
        for (int t = 0; t < NT8; t++) {
            const float2 ww = *reinterpret_cast<const float2*>(wv + 2 * t);
            float x[4];
            lw(t, x);
            const float e0 = exp2f(x[0] - sh), e1 = exp2f(x[1] - sh);
            z2 += e0;
            s2 = fmaf(e0, ww.x, s2);
            z2 += e1;
            s2 = fmaf(e1, ww.y, s2);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            z2 += __shfl_xor_sync(0xffffffffu, z2, d);
            s2 += __shfl_xor_sync(0xffffffffu, s2, d);
        }
        za = m;
        zs = z2;
        sa = s2 > 0.f ? m + mc : NEG;
        ss = s2;
        rs = mc > NEG && s2 < 0x1p-97f && m > NEG;
    }
    if (rs) {                                             // (S exactly: x' = lw' + lc' under its own max)
        const float* const lcb = ra.lcs + (long)b * ra.Np + j0 + 2 * tq;
        float m = NEG;
        for (int t = 0; t < NT8; t++) {
            const float2 lc = *reinterpret_cast<const float2*>(lcb + 8 * t);
            float x[4];
            lw(t, x);
            m = fmaxf(fmaxf(m, x[0] + lc.x), x[1] + lc.y);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
        const float sh = m == NEG ? 0.f : m;
        float s3 = 0.f;
        for (int t = 0; t < NT8; t++) {
            const float2 lc = *reinterpret_cast<const float2*>(lcb + 8 * t);
            float x[4];
            lw(t, x);
            s3 += exp2f(x[0] + lc.x - sh);
            s3 += exp2f(x[1] + lc.y - sh);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) s3 += __shfl_xor_sync(0xffffffffu, s3, d);
        sa = s3 > 0.f ? m : NEG;
        ss = s3;
    }
    return make_float4(za, zs, sa, ss);
}

// The NEXT node vector's conversion, made in this one's forward (`hf_fw_kernel`'s AH: one more warp a CTA, which
// converts while the consumers compute): its fp32 factors A2 [nb][m][k], B2 [nb][k][n] (samples `sa` / `sb` floats
// apart; rank k = its padded kp, m a multiple of 128, n of 64: a sample's block of either is its copy's, element for
// element) into its copy AP [nb][m][kp], BP [nb][kp][n] in fp16 -- `conv_copy`'s words -- in 4 KB pieces, the grid's
// warps a run each: each piece by a bulk copy into one of HAS slots of shared memory (HAS pieces in flight a warp, no
// registers held for them: a lane's own loads, 8 pieces in flight, made 1 TB/s -- the main kernel 2.6x slower), then
// converted (a lane 32 floats: 16-byte reads, 8-byte stores) and stored. The conversion pass that node vector's
// prologue would make before its main kernel (DRAM-bound) so made under this one's (compute-bound)
struct HfAhead {
    const float* a; const float* b;                           // (from the chunk's first sample)
    long sa, sb;
    __half* ap; __half* bp;                                   // (the chunk's samples of the copy)
    int pa, pb, nb;                                           // (a sample's pieces of A, of B; the samples)
};
constexpr int HAS = 4, HAP = 1024;                            // (the ring's slots; a piece's floats)
// (a bulk copy whose lines the L2 evicts first: the factors are read once -- left normal, their stream evicted the
// main kernel's operands at L 512 .. 1024, the kernel its parts' sum)
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, int bytes, uint32_t bar, uint64_t pol) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
                 "%4;" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void hf_ahead(const HfAhead& h, uint32_t ring, uint32_t bars, int lane) {
    const long per = h.pa + h.pb, P = (long)h.nb * per;
    const long q0 = (long)blockIdx.x * P / gridDim.x, q1 = (long)(blockIdx.x + 1) * P / gridDim.x;
    if (q0 >= q1) return;
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    // piece q: sample q / per, A's piece (q % per) where under pa, else B's
    auto src = [&](long q, bool& isA, long& o) -> const float* {
        const long sm = q / per, r = q - sm * per;
        isA = r < h.pa;
        o = (long)sm * (isA ? h.pa : h.pb) * HAP + (isA ? r : r - h.pa) * HAP;   // (the piece's first element)
        return isA ? h.a + (long)sm * h.sa + r * HAP : h.b + (long)sm * h.sb + (r - h.pa) * HAP;
    };
    if (lane == 0) {
        for (int k = 0; k < HAS && q0 + k < q1; k++) {
            bool isA; long o;
            const float* g = src(q0 + k, isA, o);
            mbar_expect(bars + 8 * k, HAP * 4);
            bulk_g2s_ef(ring + k * HAP * 4, g, HAP * 4, bars + 8 * k, pol);
        }
    }
    for (long q = q0, i = 0; q < q1; q++, i++) {
        const int sl = (int)(i % HAS);
        bool isA; long o;
        src(q, isA, o);
        mbar_wait(bars + 8 * sl, (int)((i / HAS) & 1));
        __half* dst = (isA ? h.ap : h.bp) + o;
        const float c = isA ? LOG2E : 1.f;
        float4 v[HAP / 128];
#pragma unroll
        for (int t = 0; t < HAP / 128; t++) {                 // (float4 lane + 32 t: conflict-free)
            const uint32_t a = ring + sl * HAP * 4 + (lane + 32 * t) * 16;
            asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(v[t].x), "=f"(v[t].y), "=f"(v[t].z), "=f"(v[t].w) : "r"(a) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // (the slot's reads before its next bulk copy)
        __syncwarp();
        if (lane == 0 && q + HAS < q1) {
            bool isA2; long o2;
            const float* g = src(q + HAS, isA2, o2);
            mbar_expect(bars + 8 * sl, HAP * 4);
            bulk_g2s_ef(ring + sl * HAP * 4, g, HAP * 4, bars + 8 * sl, pol);
        }
#pragma unroll
        for (int t = 0; t < HAP / 128; t++)
            *reinterpret_cast<uint2*>(dst + 4 * (lane + 32 * t)) = make_uint2(pack_h2(v[t].x * c, v[t].y * c),
                                                                              pack_h2(v[t].z * c, v[t].w * c));
    }
}

// The children's terms of the forward's items made by its producer warp (`hf_fw_kernel`'s WP: the prologue's WF part,
// its launch and its bulk copies gone): the tile's children's rows of the circuit's element values `emars` (rows [N],
// -1 where no edge reaches; batch `bs` columns, the chunk's samples from `b0`), and where the items of row tile 0 put
// the WF / lcs words the comb reads (`hf_fw_prep_kernel`'s)
struct HfWP {
    const float* emars; const long* rows; float* wf; float* lcs;
    int n, np, bs, b0;
};

// The forward: a persistent grid (the GPU's SMs times the CTAs an SM holds, at most one CTA an item), NW + 1 warps (the
// last the producer: an item's boxes of A and B by its maps `tmA`, `tmB`, its WF by a bulk copy; its loads from a
// consumer warp's lane instead -- 8 warps, 128 registers a thread -- fell behind the items: +15%). The work is the
// (tile, sample) items in tile-major order, each CTA an even contiguous run of them (its tile's theta set up again where
// its run enters the next tile). `scaleA`: the A boxes are the caller's A (made A' by scale_h2), else the copy's A'.
// Partials part2[b][jt][row] = (z, s) -- Z's and S's base-2 sums over the tile's children under the shifts (cm, cm + mc)
// the comb kernel knows (CMR, WF): a quad's lanes tq 0 / 1 a row each, one 8-byte store (16-byte (shift, sum) pairs by
// the tq 0 lanes behind a branch cost 13% of the loop). No test of its own: the comb kernel finds the RARE items
// (`hf_rare_z`, `hf_rare_s`) and makes them again -- here, a bit mask over stretches of 32 items and a pass of the
// warp's deferred items after each cost this kernel 12-14% (the test, the pass's registers, a stack frame).
// AH: one more warp, the next node vector's conversion (`hf_ahead` of `ah`). WP: the items' WF made by the producer's
// lanes from the children's values (`wp`: the prologue then launched only to convert), else bulk-copied from WF
template <int KP, int NT8, int NST, bool AH = false, bool WP = false>
__global__ void __launch_bounds__((HNW + 1 + AH) * 32, HfGeo<KP, NT8>::CTAS) hf_fw_kernel(
        const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, int scaleA,
        const float* __restrict__ WF, const float4* __restrict__ LTF, float2* __restrict__ part2, int M, int Mp, int B,
        int njt, long total, const __grid_constant__ HfAhead ah, const __grid_constant__ HfWP wp) {
    using Gm = HfGeo<KP, NT8>;
    constexpr int TN = Gm::TN;
    extern __shared__ __align__(128) unsigned char smraw[];
    const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smraw), sbase = (sraw + 1023u) & ~1023u;
    unsigned char* const sm = smraw + (sbase - sraw);         // (the stages 1024-byte aligned: the swizzles' frame)
    const uint32_t bfull = sbase + NST * Gm::ST, bempty = bfull + NST * 8, sguard = bempty + NST * 8;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3, lr = lane & 7, lq = lane >> 3;
    const long it0 = (long)blockIdx.x * total / gridDim.x, it1 = (long)(blockIdx.x + 1) * total / gridDim.x;
    pdl_trigger();
    if (it0 >= it1) return;
    if (tid == 0) {
        for (int s = 0; s < NST; s++) { mbar_init(bfull + 8 * s, WP ? 32 : 1); mbar_init(bempty + 8 * s, HNW); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if constexpr (AH) {
        if (w == HNW + 1) {                                   // (its ring after the guards, 128-byte aligned)
            const uint32_t ring = (sguard + 4 * HNW + 127u) & ~127u, bars = ring + HAS * HAP * 4;
            if (lane == 0) {
                for (int k = 0; k < HAS; k++) mbar_init(bars + 8 * k, 1);
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            }
            __syncwarp();
            hf_ahead(ah, ring, bars, lane);
            return;
        }
    }
    if constexpr (WP) {
        if (w == HNW) {                                       // the producer, every lane: an item's WF a stage (child
            pdl_wait();                                       //  jl = lane, lane + 32 of the tile), A / B by lane 0
            const int tile0 = (int)(it0 / B);
            int b = (int)(it0 - (long)tile0 * B), jt = tile0 % njt, rt = tile0 / njt, st = 0, ph = 0;
            long o[2];                                        // (the lane's children's emars rows, -1: none)
            auto rows_of = [&](int jtile) {
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const int j = jtile * Gm::TN + lane + 32 * u;
                    const long r = j < wp.n ? wp.rows[j] : -1;
                    o[u] = r >= 0 ? r * wp.bs + wp.b0 : -1;
                }
            };
            rows_of(jt);
            float nv[2];                                      // (the next item's values: its loads a stage ahead)
#pragma unroll
            for (int u = 0; u < 2; u++) nv[u] = o[u] >= 0 ? wp.emars[o[u] + b] : NEG;
            const int t0 = lane >> 3, q0 = (lane & 7) >> 1, p0 = lane & 1;   // (child jl = 8 t + 2 tq + p; u 1, t + 4)
            const int wo0 = q0 * 2 * NT8 + 2 * t0 + p0, wo1 = wo0 + 8;
            for (long it = it0; it < it1; it++) {
                const float v0 = nv[0], v1 = nv[1];
                const int bc = b, jc = jt, rc = rt;
                if (++b == B) {
                    b = 0;
                    if (++jt == njt) { jt = 0; rt++; }
                    if (it + 1 < it1) rows_of(jt);
                }
                if (it + 1 < it1) {
#pragma unroll
                    for (int u = 0; u < 2; u++) nv[u] = o[u] >= 0 ? wp.emars[o[u] + b] : NEG;
                }
                const float l0 = v0 * LOG2E, l1 = v1 * LOG2E;
                float mc = fmaxf(l0, l1);
#pragma unroll
                for (int d = 16; d; d >>= 1) mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, d));
                const float w0 = mc == NEG ? 0.f : exp2f(l0 - mc), w1 = mc == NEG ? 0.f : exp2f(l1 - mc);
                mbar_wait(bempty + 8 * st, ph ^ 1);
                const uint32_t dst = sbase + st * Gm::ST, bar = bfull + 8 * st;
                float* const ws = reinterpret_cast<float*>(sm + st * Gm::ST + Gm::A_BYTES + Gm::B_BYTES);
                ws[wo0] = w0;
                ws[wo1] = w1;
                if (lane < 4) ws[8 * NT8 + lane] = lane == 0 ? mc : 0.f;
                if (rc == 0) {                                // (the comb's: WF and lcs of (sample, column tile))
                    float* const wg = wp.wf + ((long)bc * njt + jc) * Gm::WFW;
                    wg[wo0] = w0;
                    wg[wo1] = w1;
                    if (lane < 4) wg[8 * NT8 + lane] = lane == 0 ? mc : 0.f;
                    float* const lg = wp.lcs + (long)bc * wp.np + jc * Gm::TN + lane;
                    lg[0] = l0;
                    lg[32] = l1;
                }
                __syncwarp();
                if (lane == 0) {                              // (its arrival the stage's 32nd: the lanes' WF before)
                    mbar_expect(bar, Gm::A_BYTES + Gm::B_BYTES);
                    tma_load_3d(dst, &tmA, 0, rc * (HTM * KP / 64), bc, bar);
                    tma_load_3d(dst + Gm::A_BYTES, &tmB, jc * TN, 0, bc, bar);
                } else {
                    mbar_arrive(bar);
                }
                if (++st == NST) { st = 0; ph ^= 1; }
            }
            return;
        }
    }
    if (w == HNW) {                                           // the producer: an item's operands a stage
        pdl_wait();                                           // (the prologue's WF, its copy; the consumers read only
        if (lane == 0) {                                      //  the stages and the theta fragments, older)
            const long wstr = (long)njt * Gm::WFW;
            const int tile0 = (int)(it0 / B);
            int b = (int)(it0 - (long)tile0 * B), jt = tile0 % njt, rt = tile0 / njt, st = 0, ph = 0;
            for (long it = it0; it < it1; it++) {
                mbar_wait(bempty + 8 * st, ph ^ 1);
                const uint32_t dst = sbase + st * Gm::ST, bar = bfull + 8 * st;
                mbar_expect(bar, Gm::A_BYTES + Gm::B_BYTES + Gm::W_BYTES);
                tma_load_3d(dst, &tmA, 0, rt * (HTM * KP / 64), b, bar);   // (a row tile's 128-byte lines)
                tma_load_3d(dst + Gm::A_BYTES, &tmB, jt * TN, 0, b, bar);
                bulk_g2s(dst + Gm::A_BYTES + Gm::B_BYTES, WF + b * wstr + (long)jt * Gm::WFW, Gm::W_BYTES, bar);
                if (++b == B) { b = 0; if (++jt == njt) { jt = 0; rt++; } }
                if (++st == NST) { st = 0; ph ^= 1; }
            }
        }
        return;
    }

    float lt[NT8][4];
    int cur, rA, rB;
    int pw;                                                   // (row rA's partial of the item: part2[pw] = [b][jt][rA])
    const int pstep = njt * Mp;
    int tile = (int)(it0 / B), b = (int)(it0 - (long)tile * B), st = 0, ph = 0;   // (counters: no division an item)
    const uint32_t a_ln = hf_a_at<KP>(16 * w + 8 * (lq & 1) + lr, lq >> 1);         // (the lane's ldmatrix rows)
    const uint32_t b_ln = Gm::A_BYTES + hf_b_base(8 * (lq & 1) + lr, lq >> 1);
    // the tile's log2 theta less each row's frame cm (its largest in the tile), from the prologue's fragments
    auto set_tile = [&]() {
        cur = tile;
        rA = (tile / njt) * HTM + 16 * w + g;
        rB = rA + 8;
        pw = (b * njt + tile % njt) * Mp + rA;
        const float4* lp = LTF + ((long)tile * HNW + w) * NT8 * 32 + lane;
#pragma unroll
        for (int t = 0; t < NT8; t++) {
            const float4 v = __ldg(lp + t * 32);
            lt[t][0] = v.x; lt[t][1] = v.y; lt[t][2] = v.z; lt[t][3] = v.w;
        }
    };
    set_tile();
    for (long it = it0; it < it1; it++) {
        if (tile != cur) set_tile();
        mbar_wait(bfull + 8 * st, ph);
        const unsigned char* sp = sm + st * Gm::ST;
        const uint32_t sb = sbase + st * Gm::ST;
        uint4 af[Gm::KS];                                     // the warp's 16 rows of A' (ldmatrix of the half rows),
#pragma unroll                                                //  a k-step's
        for (int ks = 0; ks < Gm::KS; ks++) {
            unsigned ah[4];
            ldsm4(ah, (sb + a_ln) ^ (32u * ks));
            if (scaleA) {
#pragma unroll
                for (int r = 0; r < 4; r++) ah[r] = scale_h2(ah[r]);
            }
            af[ks] = make_uint4(ah[0], ah[1], ah[2], ah[3]);
        }
        const float* wv = reinterpret_cast<const float*>(sp + Gm::A_BYTES + Gm::B_BYTES);
        const float mc = wv[8 * NT8];
        float z[2] = {0.f, 0.f}, s[2] = {0.f, 0.f};
#pragma unroll
        for (int t2 = 0; t2 < NT8; t2 += 2) {                // (n-tiles in pairs: an ldmatrix.trans of B a k-step, a
            unsigned bh[Gm::KS][4];                           //  16-byte w)
#pragma unroll
            for (int ks = 0; ks < Gm::KS; ks++) ldsm4t(bh[ks], (sb + b_ln + 2048u * ks) ^ (16u * t2));
            const float4 ww2 = *reinterpret_cast<const float4*>(wv + tq * 2 * NT8 + 2 * t2);
#pragma unroll
            for (int u = 0; u < 2; u++) {
                const int t = t2 + u;
                const float wx = u ? ww2.z : ww2.x, wy = u ? ww2.w : ww2.y;
                float d[4];
                mma_h_c(d, af[0], bh[0][2 * u], bh[0][2 * u + 1], lt[t]);
#pragma unroll
                for (int ks = 1; ks < Gm::KS; ks++)
                    mma_h16(d, af[ks].x, af[ks].y, af[ks].z, af[ks].w, bh[ks][2 * u], bh[ks][2 * u + 1]);
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
        // the lane's row (tq odd: rB, else rA -- a quad's lanes hold both rows' sums): its sums (z, s), stored by lanes
        // tq 0 and 1 (the shifts cm, cm + mc the comb kernel's: CMR, WF)
        const bool ro = tq & 1;
        __syncwarp();
        if (lane == 0) {                                      // (the stage released after its reads return: ptxas
            st_shared(sguard + 4 * w, __float_as_uint(s[0]) ^ __float_as_uint(mc));   //  issues an arrive with loads
            mbar_arrive(bempty + 8 * st);                     //  in flight -- a store of their values first)
        }
        if (tq < 2 && (ro ? rB : rA) < M) part2[pw + 8 * ro] = make_float2(ro ? z[1] : z[0], ro ? s[1] : s[0]);
        pw += pstep;                                          // (the next sample's; a tile change sets it anew)
        if (++b == B) { b = 0; tile++; }
        if (++st == NST) { st = 0; ph ^= 1; }
    }
}

// node_mars[nid0 + i, b] = (log2 S - log2 Z) ln 2 (-inf where S is 0), lzs[i, b] = log2 Z ln 2, from the column
// tiles' base-2 sums part2 [b][jt][row] = (z, s) under the shifts (cm, cm + mc) -- cm the row's frame in the tile (CMR
// [tile][row]; -inf: the row has no theta there), mc the sample's largest lc' of the tile (WF's float `wmc` of its
// tile's `wfw`). A block of 8 warps 32 rows x 8 / LPO samples of the chunk, LPO lanes an output (lane LPO row + h:
// tiles h, h + LPO, .. -- a load a warp's LPO 256 / LPO-byte runs --, 8 loads in flight, each pair made normal and
// folded in order (`fold_n`), the lanes' folds by shuffles); the outputs [node][batch] stored through shared memory, a
// row's samples contiguous. (A thread an output: 16 dependent folds a chain, 1.4x the old kernel at B64; 32 samples a
// block: too few blocks there, 2.5x.) LPO 4 where 2 would leave the SMs short of blocks. The RARE items (`hf_rare_z`,
// `hf_rare_s`) found here: an output with one made again after the block's folds by a warp, its tiles a lane each,
// each rare one by the warp (`hf_rare_row`, its operands `ra`). (The paths' code takes this kernel from 40 registers to
// 80, +6% at B256 .. 1024, the normal folds as much again -- the main kernel 12-13% faster without them; capped at 40
// or 48 the folds' loop spills, +24% .. +130%; the paths a call apart: 128 registers; one exponential a fold, the
// pairs folded as a tree, normal only where a sum lies past 2^+-30: no faster)
template <typename TP, int KP, int NT8, int LPO>
__global__ void __launch_bounds__(256) hf_comb_kernel(
        const float2* part2, const float* __restrict__ CMR, const float* WF, int wfw, int wmc,
        int njt, float* __restrict__ nmars, long nid0, float* __restrict__ lzs, int M, int Mp, int Bc, int Bs, int b0,
        const __grid_constant__ HfRare<TP> ra) {
    pdl_wait();                                                // (the partials, WF: the launches before's)
    constexpr int RW = 32 / LPO, SPB = 8 / LPO;                // (a warp's rows; a block's samples)
    __shared__ float2 out[32][SPB + 1];                        // [row][sample]: (node value, log Z)
    __shared__ unsigned char rf[32][SPB];                      // (the output has a rare item)
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, h = lane % LPO;
    const int rl = RW * (w % LPO) + lane / LPO, k = w / LPO;   // (the thread's row in the block, its sample)
    const int i0 = blockIdx.x * 32, s0 = blockIdx.y * SPB;
    const int ic = min(i0 + rl, M - 1), bc = min(s0 + k, Bc - 1);    // (clamped: loads in range)
    const float* const cr = CMR + (long)(ic / HTM) * njt * HTM + ic % HTM;   // (row ic's frames: + HTM a tile)
    const float* const wm = WF + (long)bc * njt * wfw + wmc;                 // (the sample's mc: + wfw a tile)
    const long pb = (long)bc * njt * Mp + ic;
    float mz = NEG, z = 0.f, ms = NEG, s = 0.f;
    bool rare = false;
    for (int t0 = h; t0 < njt; t0 += 8 * LPO) {
        float2 v[8];
        float c[8], mc[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {                          // (the 8 tiles' loads issued together)
            const int jt = t0 + LPO * u;
            const bool in = jt < njt;
            v[u] = in ? part2[pb + (long)jt * Mp] : make_float2(0.f, 0.f);
            c[u] = in ? cr[jt * HTM] : NEG;
            mc[u] = in ? wm[jt * wfw] : NEG;
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            rare |= hf_rare_z(v[u].x, c[u]) | hf_rare_s(v[u].y, c[u], mc[u]);
            fold_n(mz, z, v[u].x > 0.f ? c[u] : NEG, v[u].x);
            fold_n(ms, s, v[u].y > 0.f ? c[u] + mc[u] : NEG, v[u].y);
        }
    }
#pragma unroll
    for (int d = 1; d < LPO; d <<= 1) {
        fold_ms(mz, z, __shfl_xor_sync(0xffffffffu, mz, d), __shfl_xor_sync(0xffffffffu, z, d));
        fold_ms(ms, s, __shfl_xor_sync(0xffffffffu, ms, d), __shfl_xor_sync(0xffffffffu, s, d));
        rare |= __shfl_xor_sync(0xffffffffu, (int)rare, d) != 0;
    }
    rare &= i0 + rl < M && s0 + k < Bc;                        // (not the clamped loads' rows / samples)
    if (h == 0) {
        const float lz = z > 0.f ? (mz + __log2f(z)) * LN2 : NEG;
        const float ls = s > 0.f ? (ms + __log2f(s)) * LN2 : NEG;
        out[rl][k] = make_float2(ls > NEG ? ls - lz : ls, lz);   // (no mass: -inf, not 0 / 0)
        rf[rl][k] = rare;
    }
    if (__syncthreads_or(rare)) {                              // (the block's outputs with rare items: a warp each)
        for (int e = w; e < 32 * SPB; e += 8) {
            const int r = e / SPB, kk = e % SPB;
            if (!rf[r][kk]) continue;
            const int i = i0 + r, b = s0 + kk;
            const float* const cq = CMR + (long)(i / HTM) * njt * HTM + i % HTM;
            const float* const wq = WF + (long)b * njt * wfw + wmc;
            const long pq = (long)b * njt * Mp + i;
            float mz2 = NEG, z2 = 0.f, ms2 = NEG, s2 = 0.f;
            for (int u0 = 0; u0 < njt; u0 += 32) {
                const int jt = u0 + lane;
                const bool in = jt < njt;
                const float2 v = in ? part2[pq + (long)jt * Mp] : make_float2(0.f, 0.f);
                const float c = in ? cq[jt * HTM] : NEG, mc = in ? wq[jt * wfw] : NEG;
                const bool rz = hf_rare_z(v.x, c), rs = hf_rare_s(v.y, c, mc);
                float4 f = make_float4(v.x > 0.f ? c : NEG, v.x, v.y > 0.f ? c + mc : NEG, v.y);
                for (unsigned bal = __ballot_sync(0xffffffffu, rz | rs); bal; bal &= bal - 1) {
                    const int src = __ffs(bal) - 1;           // (each rare tile by the warp, its lane's pairs replaced)
                    const float4 q = hf_rare_row<TP, KP, NT8>(
                            ra, b, u0 + src, i, __shfl_sync(0xffffffffu, c, src), __shfl_sync(0xffffffffu, mc, src),
                            __shfl_sync(0xffffffffu, v.x, src), __shfl_sync(0xffffffffu, v.y, src),
                            __shfl_sync(0xffffffffu, (int)rz, src) != 0, __shfl_sync(0xffffffffu, (int)rs, src) != 0);
                    if (lane == src) f = q;
                }
                fold_n(mz2, z2, f.x, f.y);
                fold_n(ms2, s2, f.z, f.w);
            }
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                fold_ms(mz2, z2, __shfl_xor_sync(0xffffffffu, mz2, d), __shfl_xor_sync(0xffffffffu, z2, d));
                fold_ms(ms2, s2, __shfl_xor_sync(0xffffffffu, ms2, d), __shfl_xor_sync(0xffffffffu, s2, d));
            }
            if (lane == 0) {
                const float lz = z2 > 0.f ? (mz2 + __log2f(z2)) * LN2 : NEG;
                const float ls = s2 > 0.f ? (ms2 + __log2f(s2)) * LN2 : NEG;
                out[r][kk] = make_float2(ls > NEG ? ls - lz : ls, lz);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < 32 * SPB) {                              // (a thread an output: row e / SPB, sample e % SPB)
        const int e = threadIdx.x, r = e / SPB, i = i0 + r, b = s0 + e % SPB;
        if (i < M && b < Bc) {
            const float2 o = out[r][e % SPB];
            nmars[(nid0 + i) * Bs + b0 + b] = o.x;
            lzs[(long)i * Bs + b0 + b] = o.y;
        }
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
// ... a box stored instead (the box's destination this one's alone: no zeros, no reads of them)
__device__ __forceinline__ void tma_store_3d_if(bool on, const CUtensorMap* m, uint32_t ssrc, int c0, int c1, int c2) {
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %0, 0;\n"
                 " @p cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%1, {%3, %4, %5}], [%2];\n"
                 " @p cp.async.bulk.commit_group;\n}\n"
                 ::"r"((int)on), "l"(m), "r"(ssrc), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// The backward's prologue, one launch: the gradients' zeros where asked (nA4 + nB4 float4s a sample: its dA then dB
// rows); the converted copy's blocks where the forward's is not at hand and the factors are not read in place
// (`conv_copy`, (nbA + nbB) a sample: AP [B, Mp, KP] = A', BP [B, KP, Np] = B in fp16, zero past the node vector's
// nodes / children and the factors' K ranks); then, in tiles of 32 rows x 32 samples (the circuit's [rows, batch] read a
// row's 32 samples a warp's 128 bytes, the tile turned through shared memory, written a sample's row at a time -- a
// thread a node read the rows 4 bytes a 32-byte sector, 4x the bytes: 65 us at B1024), the nodes' terms LU [B, Mp, 2]
// = (lu', lv') -- lu' = log2 (f / (n Z)), lv' = log2 (f / Z), -inf for a node without flow --, the children's LC [B,
// njt, 36] (per tq the lane's 8 children's lc') with the element-flow scratch ef [B, Np] zeroed; and a block of 8
// samples' scales SC [B]: E = ceil(log2 max_i f_i) -> (E, 2^E, ln 2 2^E, 0). The main kernel takes q and t (so G, the
// fp16 operand of the dA / dB products) in units of 2^E -- |G| <= 1: fp16's range whatever the flows' scale, a
// sample's common scale carried apart in fp32 -- and its sums back by 2^E
template <int KP, typename TI>
__global__ void __launch_bounds__(256) hb_bw_prep_kernel(
        const TI* __restrict__ A, const TI* __restrict__ Bm, unsigned* __restrict__ AP, unsigned* __restrict__ BP,
        float* __restrict__ LU, float* __restrict__ LC, float* __restrict__ ef, float4* __restrict__ SC,
        const float* nflows, const float* __restrict__ nmars, long nid0,
        const float* __restrict__ lzs, const float* emars, const long* __restrict__ rows, int M, int N,
        int K, int Mp, int Np, int nbA, int nbB, int Bc, int Bs, int b0, long sA, long sB, float4* __restrict__ zA,
        long szA, int nA4, float4* __restrict__ zB, long szB, int nB4, int nbZ, const int* __restrict__ tab,
        long first) {
    pdl_wait();                                                // (the node flows, the children's values: the launch
    pdl_trigger();                                             //  before's)
    __shared__ float2 tile[32][33];                            // [sample][row] (padded)
    const int njt = Np / BN, nbg = (Bc + 31) / 32, lane = threadIdx.x & 31, wq = threadIdx.x >> 5;
    int blk = blockIdx.x;
    if (blk < nbZ * Bc) {                                      // the gradients' zeros (`init`: their first writer): a
        const int n4 = nA4 + nB4, s = blk / nbZ;               // sample's dA then dB rows, 4096
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
        conv_copy<KP>(blk % (nbA + nbB), nbA, nbB, A, Bm, AP, BP, blk / (nbA + nbB), M, N, K, Mp, Np, sA, sB);
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
            float v = NEG;                                     // (`tab`: the product's two children's node values)
            if (r >= 0 && sb < Bc)
                v = (tab == nullptr ? emars[r * Bs + b0 + sb]
                                    : nmars[(long)tab[2 * (r - first)] * Bs + b0 + sb] +
                                      nmars[(long)tab[2 * (r - first) + 1] * Bs + b0 + sb]) * LOG2E;
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

// The backward's epilogue: the element flows to the log, log-added into the circuit's (a thread a (child, sample));
// `store`: written instead (-inf for none) -- the node vector the only one of its layer to reach its children, whose
// rows then need no -inf fill first
__global__ void __launch_bounds__(256) hb_ef_epi_kernel(const float* ef, const long* __restrict__ rows,
                                                        float* __restrict__ eflows, int N, int Np, int B, int lb, int Bs,
                                                        int b0, int store, const int* __restrict__ tab, long first,
                                                        float* __restrict__ nflows) {
    pdl_wait();                                                // (the element flows' sums: the launch before's)
    const long t = (long)blockIdx.x * 256 + threadIdx.x;
    const int b = (int)(t & ((1 << lb) - 1));
    const long j = t >> lb;
    if (b >= B || j >= N) return;
    const long er = rows[j];
    if (er < 0) return;
    const float v = ef[(long)b * Np + j];
    if (tab != nullptr) {                                      // (the product group below handed over: its products'
        const float lv = v > 0.f ? logf(v) : NEG;              //  children's node flows, each child's its one parent
        nflows[(long)tab[2 * (er - first)] * Bs + b0 + b] = lv;     //  element's -- the product backward's writes)
        nflows[(long)tab[2 * (er - first) + 1] * Bs + b0 + b] = lv;
        return;
    }
    float* p = eflows + er * Bs + b0 + b;
    if (store) {
        *p = v > 0.f ? logf(v) : NEG;
        return;
    }
    if (!(v > 0.f)) return;
    const float lv = logf(v);
    const float o = *p;
    *p = o == NEG ? lv : fmaxf(o, lv) + log1pf(__expf(-fabsf(o - lv)));
}

// (DMAP: dA's rows through a box of tmA -- the caller's rows of K < 16 ranks --, else one run of the tile's rows. ONE:
// the node vector one row tile tall, each sample's dB box and element-flow sums of a column tile this kernel's alone:
// dB stored, not added -- the prologue zeroes none of it --, the sums stored into `ef`, not added)
template <typename TP, bool FLOWS, bool DENOM, bool GRADS, bool DMAP, bool ONE = false>
__global__ void __launch_bounds__(256, 1) hb_bw_kernel(
        const __half* __restrict__ AP, long sAb, int am, const __half* __restrict__ BP, long sBb, int bk, int bn,
        int scaleA, const TP* __restrict__ params, long p0, int Nb, int cbs, int bs, const float* __restrict__ LU, const float* __restrict__ LC, float* __restrict__ ef,
        float* __restrict__ pflows, long pf0, float* __restrict__ denom, long dn0, float* __restrict__ gA, long sgA,
        int ma, const __grid_constant__ CUtensorMap tmB, const __grid_constant__ CUtensorMap tmA,
        const float4* __restrict__ SC, int M, int N, int Mp, int Np, int B, int nct, long units) {
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
    // the bulk adds of a sample's dA rows (the tile's 128, clipped at ma; DMAP: a box of tmA, the caller's rows of
    // K < 16 ranks -- ranks and rows past its extents dropped) and dB's box (K ranks x 64 children at child j0, sample
    // b; children past the map's dropped), from parity pp (written and fenced a barrier back)
    auto out_dA = [&](int pp, int b, int i0) {             // (warp 0's: its lane 0 by predicate)
        if (wu == 0) {
            if constexpr (DMAP) {
                tma_add_3d_if(issuer, &tmA, uDA + pp * (Gm::SDA * 4), 0, i0, b);
            } else {
                const int rows = min(TM2, ma - i0);
                bulk_add_f_if(issuer && rows > 0, gA + (long)b * sgA + (long)i0 * HK, uDA + pp * (Gm::SDA * 4),
                              rows * HK * 4);
            }
        }
    };
    auto out_dB = [&](int pp, int b, int j0) {
        if constexpr (ONE) {
            if (wu == 0) tma_store_3d_if(issuer, &tmB, uDB + pp * (Gm::SDB * 4), j0, 0, b);
        } else {
            if (wu == 0) tma_add_3d_if(issuer, &tmB, uDB + pp * (Gm::SDB * 4), j0, 0, b);
        }
    };
    // the element flows of a sample from parity pp's partials (its 8 warps' sums), one add (ONE: one store)
    auto flows_out = [&](int pp, int b, int j0, float scy) {
        if (wu < TN2 / 32) {
            const float* const sE = sE2 + pp * Gm::SE;
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < NW2; ww++) s += sE[ww * TN2 + tid];
            if constexpr (ONE) {
                if (j0 + tid < N) ef[(long)b * Np + j0 + tid] = s * scy;
            } else {
                if (s != 0.f && j0 + tid < N) red1(ef + (long)b * Np + j0 + tid, s * scy);
            }
        }
    };
    int pc = 0, pa = 0;                                        // the sample's B / terms parity, its A' buffer (mod 3)
    pdl_trigger();
    pdl_wait();                                                // (the prologue's terms, scales, zeros, its copy)
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


// ======================================================================= backward at ranks padded to 32 and 64
// The rank-16 backward's tile (128 nodes x 64 children, 8 warps of 16 nodes, F+ / W / theta of the tile in registers
// over its samples, the operands staged by cp.async, G and the flows scaled by the sample's 2^E) at KP = 32 or 64
// ranks: phi KP / 16 MMAs a 16 x 8 tile (log2 theta the first's C operand), b's element terms as at rank 16 (G^T
// stmatrix'd, the element-flow partials), then -- after a second barrier [B] -- b's dA (the warp's own G rows by B^T)
// and dB (every warp's G^T by A', KP / 2 units of 16 children x 8 ranks a CTA) out of registers and shared memory, their
// partials added at the next sample's [A]. The rank-16 kernel's pipeline (dB of b - 1 inside b's element loop: G^T, A'
// a sample deeper, the partials by parity) would take 131 KB (KP 32) / 214 KB (64) of shared memory; this form ~75 /
// ~92 KB: one buffer of G^T, of the element-flow and fp32 partials (the issuer's bulk reads waited before [B]), two of
// A' / B / terms. dA's partials: KP 32 through shared memory, one bulk add of the tile's rows (a box of `tmA` where the
// caller's rows are K < 32 wide); KP 64 by 16-byte reductions from registers -- its 32 KB of shared memory do not fit
// --, a lane 4 ranks of a row (a quad's value pairs transposed: a warp's reduction 8 rows x 64 bytes, 2.6 TB/s of L2
// adds where 16 rows x 32 bytes made 1.9: -15% at B256). BOUND BY THE PARTIALS' ADDS (fp32: dA's 16 an element over the
// column tiles, dB's 8 over the row tiles -- 48 / 24 KB a CTA-sample at KP 64 / 32): a node vector at B256, adds alone
// (no MMAs, no exponentials) 690 of 765 us at KP 64, 367 of 391 at 32, compute alone 531 / 311 -- ~2.3 TB/s of L2 adds
// in situ. Tried at KP 64: dA by a warp's 2 KB bulk slots (rows swizzled instead of padded, no sE): +2%; dB before dA
// after [B]: 0.
template <int KP>
struct HbkGeo {
    static_assert(KP == 32 || KP == 64, "ranks padded to 32 or 64");
    static constexpr int KS = KP / 16, NT = KP / 8, NPW = NT / 2;   // k-steps; rank n-tiles; dB's n-tiles a warp
    static constexpr bool DA_SMEM = KP == 32;                  // (dA's partials through shared memory, one bulk add)
    static constexpr int AROW = KP / 2 + 4, BROW = 36, TROW = 68;   // (words; rows padded: conflict-free ldmatrix)
    static constexpr int SAH = TM2 * AROW, SBH = KP * BROW, SL = 2 * TM2 + 2 * 36 + 4, ST = TN2 * TROW, SE = NW2 * TN2;
    static constexpr int SDA = DA_SMEM ? TM2 * KP : 0, SDB = KP * TN2;
    static constexpr int FLUSH = NW2 * 16 * 20;                // F+ / W: [warp][child 16][node 16 + 4], a quarter at a time
    static constexpr int WORDS = SDA + SDB + 2 * SAH + 2 * SBH + 2 * SL + ST + SE;
    static_assert(FLUSH <= ST, "the F+ / W flush within the G^T buffer");
    static_assert(WORDS * 4 <= 101376, "a CTA's shared memory");
    static_assert(SDA % 32 == 0 && SDB % 32 == 0 && SAH % 4 == 0 && SBH % 4 == 0 && SL % 4 == 0 && ST % 4 == 0,
                  "aligned pieces");
};

// gA, gB: the gradients' destinations (dA rows `ka` floats, `kv` of them live, `ma` rows, samples `sgA` apart; dB by
// the tensor map tmB); `dmap` (KP 32): dA's rows through a box of tmA
template <int KP, typename TP, bool FLOWS, bool DENOM, bool GRADS, bool ONE = false>
__global__ void __launch_bounds__(256, 1) hbk_bw_kernel(
        const __half* __restrict__ AP, long sAb, int am, const __half* __restrict__ BP, long sBb, int bk, int bn,
        int scaleA, const TP* __restrict__ params, long p0, int Nb, int cbs, int bs, const float* __restrict__ LU,
        const float* __restrict__ LC, float* __restrict__ ef, float* __restrict__ pflows, long pf0,
        float* __restrict__ denom, long dn0, float* __restrict__ gA, long sgA, int ma, int ka, int kv,
        const __grid_constant__ CUtensorMap tmB, const __grid_constant__ CUtensorMap tmA, int dmap,
        const float4* __restrict__ SC, int M, int N, int Mp, int Np, int B, int nct, long units) {
    using Gm = HbkGeo<KP>;
    constexpr int KS = Gm::KS, NPW = Gm::NPW;
    extern __shared__ __align__(128) unsigned smw[];
    float* const sDA = reinterpret_cast<float*>(smw);          // [node 128][rank KP] (KP 32)
    float* const sDB = sDA + Gm::SDA;                          // [rank KP][child 64]
    unsigned* const sAh = reinterpret_cast<unsigned*>(sDB + Gm::SDB);   // [par][node][AROW]: A' rows (fp16 pairs)
    unsigned* const sBh = sAh + 2 * Gm::SAH;                   // [par][rank][BROW]: B (fp16 pairs of children)
    float* const sL = reinterpret_cast<float*>(sBh + 2 * Gm::SBH);         // [par][SL]: LU, LC, SC
    unsigned* const sT = reinterpret_cast<unsigned*>(sL + 2 * Gm::SL);     // [child][TROW]: G^T (fp16 pairs)
    float* const sE = reinterpret_cast<float*>(sT + Gm::ST);   // [warp][64]: element-flow partials
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, tq = lane & 3;
    const int wu = __shfl_sync(0xffffffffu, w, 0);             // (the warp's index, provably uniform: uniform branches)
    const int lr = lane & 7, lq = lane >> 3;                   // (ldmatrix / stmatrix: the row a lane addresses, its matrix)
    const bool issuer = GRADS && tid == 0;                     // (the bulk adds' thread)
    const int njt = Np / BN;
    const long u0 = (long)blockIdx.x * units / gridDim.x, u1 = (long)(blockIdx.x + 1) * units / gridDim.x;
    const uint32_t uDA = smem_u32(smw), uDB = uDA + Gm::SDA * 4, uAh = uDB + Gm::SDB * 4;
    const uint32_t uBh = uAh + 2 * Gm::SAH * 4, uL = uBh + 2 * Gm::SBH * 4, uT = uL + 2 * Gm::SL * 4;
    const uint32_t oA = ((16 * w + 8 * (lq & 1) + lr) * Gm::AROW + 4 * (lq >> 1)) * 4;    // phi's A' rows, + 32 ks
    const uint32_t oB = ((8 * (lq & 1) + lr) * Gm::BROW + 4 * (lq >> 1)) * 4;  // phi's B, + 16 t2, + 64 BROW ks
    const uint32_t oBt = ((8 * (lq >> 1) + lr) * Gm::BROW + 4 * (lq & 1)) * 4; // dA's B^T, + 32 kk, + 64 BROW p
    const uint32_t oT = ((8 * (lq >> 1) + lr) * Gm::TROW + 8 * w + 4 * (lq & 1)) * 4;    // G^T's rows, + 64 TROW kk
    const uint32_t oTa = ((16 * (w & 3) + 8 * (lq & 1) + lr) * Gm::TROW + 4 * (lq >> 1)) * 4;   // dB's G^T, + 32 kw
    const int nb = NPW * (w >> 2);                             // (dB: the warp's first rank n-tile, its child m-tile w & 3)
    const uint32_t oAb = ((8 * (lq & 1) + lr) * Gm::AROW + 4 * (nb + (lq >> 1))) * 4;    // dB's A', + 64 AROW kw, + 32 p
    // sample b's operands into stage par (row tile its, child tile jcs): the tile's 128 A' rows (a thread KP / 16
    // 16-byte pieces: rows tid / (KP / 8) + RP s, zeros from row am), B's KP rank rows of its 64 children (KP / 32
    // pieces: rank rows tid / 8 + 32 s, bk halves apart; zeros from child bn), the terms (LU rows, the 2 jt's LC, SC:
    // threads 0 .. 82). A thread's sources a tile's base and its sample's stride
    constexpr int CPR = KP / 8, RP = 256 / CPR;                // (A': a row's pieces, the rows a pass)
    struct Src { const char* a; const char* r; const char* t; long ta; bool rl, tl; };
    auto srcs = [&](int jcs, int its) {
        Src q;
        q.a = reinterpret_cast<const char*>(AP + 8 * (tid % CPR));   // (row 0's piece: a row's offset a pass)
        const int j = jcs * TN2 + 8 * (tid & 7);
        q.r = reinterpret_cast<const char*>(BP + (long)(tid >> 3) * bk + min(j, bn - 8));
        q.rl = j < bn;
        const int r = tid;
        q.tl = r < 83;
        q.t = reinterpret_cast<const char*>(r < 64 ? LU + (long)its * TM2 * 2 + r * 4
                                            : r < 82 ? LC + (long)2 * jcs * 36 + (r - 64) * 4
                                                     : reinterpret_cast<const float*>(SC));
        q.ta = r < 64 ? (long)Mp * 2 * 4 : r < 82 ? (long)njt * 36 * 4 : 16;
        return q;
    };
    const uint32_t dA0 = uAh + ((tid / CPR) * Gm::AROW + 4 * (tid % CPR)) * 4;
    const uint32_t dB0 = uBh + ((tid >> 3) * Gm::BROW + 4 * (tid & 7)) * 4;
    const uint32_t dT0 = uL + tid * 16;
    auto stage = [&](const Src& q, int its, int b, int par) {
        const int i = its * TM2 + tid / CPR;
#pragma unroll
        for (int s = 0; s < KS; s++) {                         // (rows past am: zeros, row am - 1's address)
            const int row = i + RP * s;
            cp_async16z(dA0 + par * (Gm::SAH * 4) + s * (RP * Gm::AROW * 4),
                        q.a + ((long)b * sAb + (long)min(row, am - 1) * KP) * 2, row < am);
        }
#pragma unroll
        for (int s = 0; s < KP / 32; s++)
            cp_async16z(dB0 + par * (Gm::SBH * 4) + s * (32 * Gm::BROW * 4), q.r + (long)b * sBb * 2 + (long)s * 32 * bk * 2,
                        q.rl);
        if (q.tl) cp_async16(dT0 + par * (Gm::SL * 4), q.t + b * q.ta);
        cp_async_commit();
    };
    // the bulk adds of a sample's dA rows (KP 32: the tile's 128, clipped at ma; `dmap`: a box of tmA) and dB's box (the
    // destination's ranks x 64 children at child j0, sample b; children past the map's dropped), from their buffers
    // (written and fenced a barrier back)
    auto out_dA = [&](int b, int i0) {
        if constexpr (Gm::DA_SMEM) {
            if (wu == 0) {
                if (dmap) tma_add_3d_if(issuer, &tmA, uDA, 0, i0, b);
                else {
                    const int rows = min(TM2, ma - i0);
                    bulk_add_f_if(issuer && rows > 0, gA + (long)b * sgA + (long)i0 * KP, uDA, rows * KP * 4);
                }
            }
        }
    };
    auto out_dB = [&](int b, int j0) {                     // (ONE: `hb_bw_kernel`'s -- dB stored)
        if constexpr (ONE) {
            if (wu == 0) tma_store_3d_if(issuer, &tmB, uDB, j0, 0, b);
        } else {
            if (wu == 0) tma_add_3d_if(issuer, &tmB, uDB, j0, 0, b);
        }
    };
    int pc = 0;                                                // the sample's operands' parity
    pdl_trigger();
    pdl_wait();                                                // (the prologue's terms, scales, zeros, its copy)
    if (u0 < u1) {                                             // the run's first sample: its stage
        const int t = (int)(u0 / B), b = (int)(u0 - (long)t * B), it = t / nct;
        stage(srcs(t - it * nct, it), it, b, 0);
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

#pragma unroll 1
        for (int b = b0; b < b1; b++, pc ^= 1) {
            const int par = pc;
            cp_async_wait<0>();
            __syncthreads();                                   // [A] b's stage; b - 1's partials written
            if (GRADS && b > b0) {
                out_dA(b - 1, i0);
                out_dB(b - 1, j0);
            }
            if (b + 1 < b1) stage(sq, it, b + 1, par ^ 1);    // (its buffers' last readers: b - 1's, before [A])
            else if (more) {                                   // (the next tile's first sample)
                const int itn = (t + 1) / nct;
                stage(srcs(t + 1 - itn * nct, itn), itn, 0, par ^ 1);
            }
            const uint32_t uah = uAh + par * (Gm::SAH * 4), ubh = uBh + par * (Gm::SBH * 4);
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
            // ---- lw_b = phi_b + log2 theta (KS fp16 k-steps, theta the first's C operand): the warp's A' fragments
            // (ldmatrix of its 16 rows, a k-step's), B's (ldmatrix.trans of the rank rows), a pair of n-tiles at a time,
            // one pair ahead of the element terms
            unsigned ah[KS][4];
#pragma unroll
            for (int ks = 0; ks < KS; ks++) {
                ldsm4(ah[ks], uah + oA + 32 * ks);
                if (scaleA) {                                  // (the caller's A: A' as the forward made it)
#pragma unroll
                    for (int r = 0; r < 4; r++) ah[ks][r] = scale_h2(ah[ks][r]);
                }
            }
            float S[8][4];
            auto phi_pair = [&](int t2) {                      // (n-tiles t2, t2 + 1: their k halves, a k-step's)
#pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    unsigned bh[4];
                    ldsm4t(bh, ubh + oB + 16 * t2 + 64 * Gm::BROW * ks);
                    const uint4 a4 = make_uint4(ah[ks][0], ah[ks][1], ah[ks][2], ah[ks][3]);
                    if (ks == 0) {
                        mma_h_c(S[t2], a4, bh[0], bh[1], *reinterpret_cast<const float(*)[4]>(lt + 4 * t2));
                        mma_h_c(S[t2 + 1], a4, bh[2], bh[3], *reinterpret_cast<const float(*)[4]>(lt + 4 * t2 + 4));
                    } else {
                        mma_h16(S[t2], a4.x, a4.y, a4.z, a4.w, bh[0], bh[1]);
                        mma_h16(S[t2 + 1], a4.x, a4.y, a4.z, a4.w, bh[2], bh[3]);
                    }
                }
            };
            phi_pair(0);
            // ---- the element terms of b: q = 2^(lw' + lu' + lc'), t = 2^(lw' + lv'); G^T a pair of n-tiles at a time
            unsigned R[8][2];                                  // (q - t) as fp16 pairs: (n-tile, node half)
            float v[16];                                       // q over the lane's two rows, its children 2 nf + p
#pragma unroll
            for (int nf = 0; nf < 8; nf++) {
                if ((nf & 1) == 0 && nf + 2 < 8) phi_pair(nf + 2);
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
                if (GRADS && (nf & 1)) {                       // (dB's G^T: children 16 kk .., the warp's 16 nodes)
                    const int kk = nf >> 1;
                    stsm4t(uT + oT + 64 * Gm::TROW * kk, R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1]);
                }
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
            if (wu == 0) bulk_wait_read0_if(issuer);           // (the adds issued at [A] have read their partials)
            __syncthreads();                                   // [B] b's G^T, element-flow partials written
            if (wu < TN2 / 32) {                               // b's element flows: the 8 warps' partials, one add
                float s = 0.f;
#pragma unroll
                for (int ww = 0; ww < NW2; ww++) s += sE[ww * TN2 + tid];
                if constexpr (ONE) {
                    if (j0 + tid < N) ef[(long)b * Np + j0 + tid] = s * scl.y;
                } else {
                    if (s != 0.f && j0 + tid < N) red1(ef + (long)b * Np + j0 + tid, s * scl.y);
                }
            }
            if (GRADS) {
                // ---- dA_b = G B^T over the tile's children (the warp's rows; units of 2^E: times 2^E), a pair of rank
                // n-tiles at a time: the A operand R[2 kk][0], R[2 kk][1], R[2 kk + 1][0], R[2 kk + 1][1]; B^T by
                // ldmatrix of B's rank rows (k pairs along the children)
#pragma unroll
                for (int p = 0; p < KP / 16; p++) {
                    float dd[2][4];
#pragma unroll
                    for (int nt = 0; nt < 2; nt++)
#pragma unroll
                        for (int c = 0; c < 4; c++) dd[nt][c] = 0.f;
#pragma unroll
                    for (int kk = 0; kk < 4; kk++) {
                        unsigned bt[4];
                        ldsm4(bt, ubh + oBt + 32 * kk + 64 * Gm::BROW * p);
                        mma_h16(dd[0], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], bt[0], bt[1]);
                        mma_h16(dd[1], R[2 * kk][0], R[2 * kk][1], R[2 * kk + 1][0], R[2 * kk + 1][1], bt[2], bt[3]);
                    }
                    if constexpr (Gm::DA_SMEM) {               // (nodes g / g + 8, ranks 16 p + 8 nt + 2 tq, + 1)
#pragma unroll
                        for (int nt = 0; nt < 2; nt++) {
                            const int rk = 16 * p + 8 * nt + 2 * tq;
                            *reinterpret_cast<float2*>(sDA + (16 * w + g) * KP + rk) =
                                make_float2(dd[nt][0] * scl.y, dd[nt][1] * scl.y);
                            *reinterpret_cast<float2*>(sDA + (16 * w + g + 8) * KP + rk) =
                                make_float2(dd[nt][2] * scl.y, dd[nt][3] * scl.y);
                        }
                    } else {
                        // (a quad's pairs transposed: lane tq holds pair tq of n-tile 2p (ranks 16 p + 2 tq, + 1) and
                        // pair 4 + tq of n-tile 2p + 1 and takes pairs 2 tq, 2 tq + 1 -- 4 ranks of a row, a 16-byte
                        // reduction; a warp instruction 8 rows x 64 bytes: 2.6 TB/s of L2 adds where 16 rows x 32
                        // bytes made 1.9). Round A: lanes 0, 2 offer their n-tile 2p + 1 pair, 1, 3 their n-tile 2p
                        // pair, lane tq reads lane {1, 3, 0, 2}[tq]; round B: lane 1 offers its n-tile 2p + 1 pair,
                        // lane 2 its n-tile 2p pair, read by each other
                        const int qb = lane & ~3;
                        const int srcA = qb + ((0x2031 >> (4 * tq)) & 0xf), srcB = qb + ((0x3120 >> (4 * tq)) & 0xf);
#pragma unroll
                        for (int h = 0; h < 2; h++) {          // (row g, then g + 8)
                            const float a0 = dd[0][2 * h], a1 = dd[0][2 * h + 1], b0v = dd[1][2 * h], b1v = dd[1][2 * h + 1];
                            const float oa0 = (tq & 1) ? a0 : b0v, oa1 = (tq & 1) ? a1 : b1v;
                            const float ob0 = tq == 1 ? b0v : a0, ob1 = tq == 1 ? b1v : a1;
                            const float ra0 = __shfl_sync(0xffffffffu, oa0, srcA), ra1 = __shfl_sync(0xffffffffu, oa1, srcA);
                            const float rb0 = __shfl_sync(0xffffffffu, ob0, srcB), rb1 = __shfl_sync(0xffffffffu, ob1, srcB);
                            const float l0 = tq == 0 ? a0 : tq == 1 ? rb0 : ra0, l1 = tq == 0 ? a1 : tq == 1 ? rb1 : ra1;
                            const float h0 = tq == 2 ? rb0 : tq == 3 ? b0v : ra0, h1 = tq == 2 ? rb1 : tq == 3 ? b1v : ra1;
                            const float4 v4 = make_float4(l0 * scl.y, l1 * scl.y, h0 * scl.y, h1 * scl.y);
                            const int row = rA + 8 * h, rk = 16 * p + 4 * tq;
                            if (row < ma && rk < kv) red4(gA + (long)b * sgA + (long)row * ka + rk, v4);
                        }
                    }
                }
                // ---- dB_b = A'^T G over the tile's nodes (k-steps kw: the 8 warps' 16-node blocks of G^T): the warp's
                // units (child m-tile w & 3, rank n-tiles nb ..), the G^T fragment shared, A' by ldmatrix.trans of the
                // node rows (two n-tiles a load); into [rank][child] rows (units of 2^E, A' carrying log2 e: times
                // ln 2 2^E; 2^E for the caller's A itself)
                float dq[NPW][4];
#pragma unroll
                for (int x = 0; x < NPW; x++)
#pragma unroll
                    for (int c = 0; c < 4; c++) dq[x][c] = 0.f;
#pragma unroll
                for (int kw = 0; kw < NW2; kw++) {
                    unsigned aa[4];
                    ldsm4(aa, uT + oTa + 32 * kw);
#pragma unroll
                    for (int pp = 0; pp < NPW / 2; pp++) {
                        unsigned bb[4];
                        ldsm4t(bb, uah + oAb + 64 * Gm::AROW * kw + 32 * pp);
                        mma_h16(dq[2 * pp], aa[0], aa[1], aa[2], aa[3], bb[0], bb[1]);
                        mma_h16(dq[2 * pp + 1], aa[0], aa[1], aa[2], aa[3], bb[2], bb[3]);
                    }
                }
                const float scz = scaleA ? scl.y : scl.z;
                const int ch = 16 * (w & 3) + g;
#pragma unroll
                for (int x = 0; x < NPW; x++) {
                    const int rk = 8 * (nb + x) + 2 * tq;
                    sDB[rk * TN2 + ch] = dq[x][0] * scz;
                    sDB[(rk + 1) * TN2 + ch] = dq[x][1] * scz;
                    sDB[rk * TN2 + ch + 8] = dq[x][2] * scz;
                    sDB[(rk + 1) * TN2 + ch + 8] = dq[x][3] * scz;
                }
                fence_async_smem();                            // (the partials to the bulk adds after the next barrier)
            }
        }
        // the tile's drain: the last sample's partials added; then F+ / W
        __syncthreads();
        if (GRADS) {
            out_dA(b1 - 1, i0);
            out_dB(b1 - 1, j0);
        }
        // F+ and W of the warp's 16 x 64 cells, a quarter of the children at a time through its own rows of the G^T
        // buffer (the next tile's first stage lands meanwhile in A' / B / terms): a lane four consecutive nodes of a
        // child, one 16-byte add (scalar adds where the node blocks are under 4 or the runs unaligned)
        if (FLOWS || DENOM) {
            float* const st = reinterpret_cast<float*>(sT) + w * (16 * 20);   // [child 16][node 16 + 4]
            const bool vec = (bs & 3) == 0 && ((pf0 | dn0) & 3) == 0;
#pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                if ((pass == 0 && !FLOWS) || (pass == 1 && !DENOM)) continue;
                float* const dst = pass == 0 ? pflows + pf0 : denom + dn0;
#pragma unroll
                for (int qq = 0; qq < 4; qq++) {
#pragma unroll
                    for (int e8 = 0; e8 < 8; e8++) {
                        const int e = 8 * qq + e8, nf = e >> 2, c = e & 3;
                        float val = pass == 0 ? fp[e] : tw[e];
                        if (pass == 1) {
                            const float th = exp2f(lt[e]);
                            val = th > 0.f ? val / th : 0.f;
                        }
                        st[(8 * (nf - 2 * qq) + 2 * tq + (c & 1)) * 20 + g + 8 * (c >> 1)] = val;
                    }
                    __syncwarp();
#pragma unroll
                    for (int uu = 0; uu < 2; uu++) {           // (8 children an instruction, 4 lanes a child)
                        const int col = 8 * uu + (lane >> 2), r4 = 4 * (lane & 3), i = i0 + 16 * w + r4;
                        const int j = j0 + 16 * qq + col;
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
        __syncthreads();                                       // (the flush's rows: the next tile's G^T)
    }
    if (issuer) bulk_wait_all();                               // (the partials' last reads before the CTA ends)
}


// The current device's L2 bytes (asked once a device)
static int l2_bytes() {
    constexpr int NDEV = 64;
    static int l2s[NDEV] = {0};
    const int dev = at::cuda::current_device();
    int l2 = dev < NDEV ? l2s[dev] : 0;
    if (l2 == 0) {
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
        if (dev < NDEV) l2s[dev] = l2;
    }
    return l2;
}

// Samples a pass takes at once: its per-sample working set `per` (bytes) kept within the L2, the batch split evenly
static int64_t sample_chunk(int64_t B, double per) {
    const int l2 = l2_bytes();
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

// The factors' rank K padded to the kernels' (16, 32 or 64: zero ranks past K)
static int rank_pad_h(int64_t K) { return K <= 16 ? 16 : K <= 32 ? 32 : 64; }

// Whether the factors are read in place: fp16, their rank the kernels' (K = KP: no padding), each sample's block
// 16-byte aligned (the base, the batch stride a multiple of 8 halves), A's rows whole 128-byte lines (M a multiple of
// 64 / KP) and B's rank rows whole 8-child pieces (N a multiple of 8) -- the TMA boxes', the cp.async pieces'
// alignment. Else the kernels read a converted copy (`conv_copy`)
static bool h_in_place(const torch::Tensor& A, const torch::Tensor& Bm, int64_t M, int64_t N, int64_t K, int KP) {
    const auto ok = [](const torch::Tensor& t) {
        return t.scalar_type() == torch::kHalf && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 &&
               sstride(t) % 8 == 0;
    };
    return K == KP && ok(A) && ok(Bm) && M % (64 / KP) == 0 && N % 8 == 0;
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
    TORCH_CHECK(K >= 1 && K <= 64, "the fp16 exp_sum kernels take ranks 1 .. 64");
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

// `kernel<<<grid, block, smem, st>>>(args...)`, as a programmatic dependent of the launch before it where `pdl` (the
// pass's launches chained: each one's launch and prologue under the tail of the one before -- MEASURED a forward at L
// 128 .. 512, B 1024 -0.14 .. -0.29 ms)
template <typename... P, typename... A>
static void launch_h(bool pdl, void (*kernel)(P...), dim3 grid, dim3 block, int smem, cudaStream_t st, A... args) {
    if (!pdl) {
        kernel<<<grid, block, smem, st>>>(static_cast<P>(args)...);
        return;
    }
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid; lc.blockDim = block; lc.dynamicSmemBytes = smem; lc.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at; lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kernel, static_cast<P>(args)...));
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

// The forward's operands as tensor maps of fp16 in 128-byte swizzled boxes, zeros past the extents: A [bc][rows][KP]
// (rows contiguous -- `rows` a multiple of 64 / KP --, samples `ss` halves apart) as lines of 64 / KP rows, boxes of
// 2 KP lines (128 rows); B [bc][KP][n] (rank rows of n children contiguous) in boxes of 64 children
template <int KP>
static CUtensorMap h_map(const void* base, bool isA, int64_t ext, int64_t ss, int64_t bc) {
    constexpr int RPL = 64 / KP;
    CUtensorMap m;
    cuuint64_t dimA[3] = {64, (cuuint64_t)ext / RPL, (cuuint64_t)bc};
    cuuint64_t dimB[3] = {(cuuint64_t)ext, (cuuint64_t)KP, (cuuint64_t)bc};
    cuuint64_t strA[2] = {128, (cuuint64_t)ss * 2}, strB[2] = {(cuuint64_t)ext * 2, (cuuint64_t)ss * 2};
    cuuint32_t boxA[3] = {64, (cuuint32_t)HTM / RPL, 1}, boxB[3] = {64, (cuuint32_t)KP, 1};
    cuuint32_t es[3] = {1, 1, 1};
    const CUresult r = tmap_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, const_cast<void*>(base), isA ? dimA : dimB,
                                     isA ? strA : strB, isA ? boxA : boxB, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "the forward's tensor map: cuTensorMapEncodeTiled failed (", (int)r, ")");
    return m;
}

// The factors' fp16 operands as the kernels read them (the comb's rare items, the backward's stages): A rows (KP
// halves) `sAb` apart a sample, zeros from row am; B rank rows `bk` apart, `sBb` a sample, zeros from child bn; scaleA:
// A is the caller's (made A' in the kernels), else the copy's A'
struct HbOps { const __half* a; long sAb; int am; const __half* b; long sBb; int bk, bn, scaleA; };

template <int KP, int NT8, int NST, bool AH = false, bool WP = false>
void launch_hf_fw(bool pdl, cudaStream_t stream, const CUtensorMap& tma, const CUtensorMap& tmb, int scaleA,
                  const float* wf, const float4* ltf, float2* part2, int M, int Mp, int Bc, int njt, long total,
                  const HfAhead& ah, const HfWP& wpp) {
    using Gm = HfGeo<KP, NT8>;
    constexpr int bytes = NST * Gm::ST + 2 * NST * 8 + 4 * HNW + 1024 +   // (+ the stages' alignment; AH: its ring)
                          (AH ? 128 + HAS * (HAP * 4 + 8) : 0);
    constexpr int threads = (HNW + 1 + AH) * 32;
    auto kern = hf_fw_kernel<KP, NT8, NST, AH, WP>;
    static int per_sm[64] = {0};                              // (a device's CTAs an SM; 0: not asked yet)
    const int dev = at::cuda::current_device();
    int occ = dev < 64 ? per_sm[dev] : 0;
    if (occ == 0) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, bytes));
        occ = std::max(occ, 1);
        if (dev < 64) per_sm[dev] = occ;
    }
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const long grid = std::max<long>(1, std::min<long>(total, (long)sms * occ));
    launch_h(pdl, kern, dim3((unsigned)grid), dim3(threads), bytes, stream, tma, tmb, scaleA, wf, ltf, part2, M, Mp, Bc,
             njt, total, ah, wpp);
}

// The tiles' theta fragments of a node vector (`hf_theta_kernel`): {LTF, CMR}
template <int NT8>
std::vector<torch::Tensor> theta_impl(torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs, int64_t bs, int64_t M,
                                      int64_t N) {
    using Gm = HfGeo<16, NT8>;                                 // (the tile's children: any rank's)
    const at::cuda::CUDAGuard guard(params.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int Mp = (int)((M + HTM - 1) / HTM * HTM), Np = (int)((N + Gm::TN - 1) / Gm::TN * Gm::TN);
    const int nbT = (Np / Gm::TN) * (Mp / HTM);
    auto LTF = torch::empty({(long)nbT * HNW * NT8 * 32 * 4}, params.options().dtype(torch::kFloat32));
    auto CMR = torch::empty({(long)nbT * HTM}, params.options().dtype(torch::kFloat32));
    HF_TP_SWITCH(params, hf_theta_kernel<TP, NT8><<<(unsigned)nbT, 256, 0, stream>>>(
        pp, p0, (int)Nb, (int)cbs, (int)bs, reinterpret_cast<float4*>(LTF.template data_ptr<float>()),
        CMR.template data_ptr<float>(), (int)M, (int)N, Np));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {LTF, CMR};
}

// Whether the forward converts the next node vector's factors `ahead` {A2 [B, M2, K2], B2 [B, K2, N2]} (`hf_ahead`):
// fp32, each sample's block contiguous, 16-byte aligned (base and sample stride), K2 a padded rank (16, 32, 64), M2 a
// multiple of 128 and N2 of 64 (its blocks its copy's element for element), its fp16 copy at most a quarter of the L2
static bool ahead_fits(const std::vector<torch::Tensor>& ahead, int64_t B, const c10::Device& dev) {
    if (ahead.size() != 2) return false;
    const torch::Tensor &a = ahead[0], &b = ahead[1];
    if (a.dim() != 3 || b.dim() != 3 || !a.is_cuda() || a.device() != dev || b.device() != dev) return false;
    if (a.scalar_type() != torch::kFloat32 || b.scalar_type() != torch::kFloat32) return false;
    const int64_t M2 = a.size(1), K2 = a.size(2), N2 = b.size(2);
    if (a.size(0) != B || b.size(0) != B || b.size(1) != K2) return false;
    if (!inner_contiguous(a) || !inner_contiguous(b)) return false;
    if (K2 != rank_pad_h(K2) || M2 % HTM != 0 || N2 % 64 != 0 || M2 == 0 || N2 == 0) return false;
    const auto al = [](const torch::Tensor& t) {
        return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 && sstride(t) % 4 == 0;
    };
    if (!al(a) || !al(b)) return false;
    return 2.0 * B * K2 * (M2 + N2) <= 0.25 * l2_bytes();
}

// The forward of one node vector at KP ranks (its K padded): its node values into `node_mars` rows `nid0 .. nid0 + M`;
// returns {log Z [M, B], AP, BP, AP2, BP2}: the converted copy of the factors where they are not read in place and
// `keep` (the backward's operands then: AP [B, Mp, KP] = A', BP [B, KP, Np] = B in fp16), else empty; `given`: that
// copy made already (`ahead` of the node vector before: {AP, BP}), else empty. `ahead`: the next node vector's fp32
// factors {A2 [B, M2, K2], B2 [B, K2, N2]}, converted into its copy {AP2, BP2} under this forward's main kernel (`AH`)
// where `ahead_fits`, else nothing made (empty)
template <int KP, int NT8>
std::vector<torch::Tensor> forward_impl(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb,
                                        int64_t cbs, int64_t bs, torch::Tensor emars, torch::Tensor rows,
                                        torch::Tensor node_mars, int64_t nid0, int64_t M, int64_t N, int64_t K,
                                        torch::Tensor LTF, torch::Tensor CMR, bool keep,
                                        const std::vector<torch::Tensor>& given,
                                        const std::vector<torch::Tensor>& ahead, bool pdl, bool wp) {
    using Gm = HfGeo<KP, NT8>;
    constexpr int NST = Gm::NST, TN = Gm::TN;
    const int64_t B = node_mars.size(1);
    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int Mp = (int)((M + HTM - 1) / HTM * HTM), Np = (int)((N + TN - 1) / TN * TN), njt = Np / TN, nit = Mp / HTM;
    auto lzs = torch::empty({M, B}, emars.options());
    const long* rw = reinterpret_cast<const long*>(rows.template data_ptr<int64_t>());
    const bool made = given.size() == 2 && given[0].numel() > 0;   // (the copy at hand: the batch's)
    if (made)
        TORCH_CHECK(given[0].scalar_type() == torch::kHalf && given[0].is_contiguous() &&
                    given[0].numel() == B * Mp * KP && given[1].scalar_type() == torch::kHalf &&
                    given[1].is_contiguous() && given[1].numel() == B * KP * Np,
                    "given: the copy {AP [B, Mp, KP], BP [B, KP, Np]} of these factors");
    const bool direct = !made && h_in_place(A, Bm, M, N, K, KP);
    // a sample's working set: its fp16 operands, WF, lcs, the partials (z, s)
    const int64_t BC = sample_chunk(B, 2.0 * KP * (Mp + Np) + 4.0 * njt * Gm::WFW + 4.0 * Np + 8.0 * njt * Mp);
    torch::Tensor AP, BP;                                      // (the converted copy: the batch's where kept or given)
    if (made) {
        AP = given[0];
        BP = given[1];
    } else if (!direct) {
        const int64_t cb = keep ? B : BC;
        AP = torch::empty({cb, (long)Mp, KP}, emars.options().dtype(torch::kHalf));
        BP = torch::empty({cb, KP, (long)Np}, emars.options().dtype(torch::kHalf));
    }
    // the next node vector's conversion under this one's main kernel (`hf_ahead`), where its blocks are its copy's
    // element for element -- fp32, its rank padded already, its nodes and children whole tiles, 16-byte aligned
    // samples -- and its copy fits a quarter of the L2 (MEASURED: a larger one, written under this kernel and read
    // under the next, is evicted between -- the kernel its parts' sum, +0.3 .. 2.3 ms a forward at a copy of 64 MB on);
    // else nothing made, that one converts its own. Its copy, the batch's
    const bool ah = ahead_fits(ahead, B, A.device());
    const int64_t M2 = ah ? ahead[0].size(1) : 0, N2 = ah ? ahead[1].size(2) : 0, K2 = ah ? ahead[0].size(2) : 0;
    const int KP2 = ah ? (int)K2 : 16, Mp2 = (int)M2, Np2 = (int)N2;
    torch::Tensor AP2, BP2;
    if (ah) {
        AP2 = torch::empty({B, (long)Mp2, KP2}, emars.options().dtype(torch::kHalf));
        BP2 = torch::empty({B, KP2, (long)Np2}, emars.options().dtype(torch::kHalf));
    }
    auto WF = torch::empty({BC, (long)njt * Gm::WFW}, emars.options());
    auto lcs = torch::empty({BC, (long)Np}, emars.options());
    auto part2 = torch::empty({BC, (long)njt * Mp * 2}, emars.options());      // (every item's (z, s))
    float* const wfp = WF.template data_ptr<float>(), * const lcp = lcs.template data_ptr<float>();
    float* const p2p = part2.template data_ptr<float>();
    const int nbT = njt * nit;                                 // (the tiles' theta fragments: given, or made here)
    if (LTF.numel() == 0) {
        auto tc = theta_impl<NT8>(params, p0, Nb, cbs, bs, M, N);
        LTF = tc[0];
        CMR = tc[1];
    }
    TORCH_CHECK(LTF.numel() == (long)nbT * HNW * NT8 * 32 * 4 && CMR.numel() == (long)nbT * HTM,
                "theta fragments: exp_sum_h_theta's, for this node vector and tile width");
    TORCH_CHECK((int64_t)BC * njt * Mp < (int64_t)INT_MAX, "the partials' index: 32 bits");
    const float4* ltf = reinterpret_cast<const float4*>(LTF.template data_ptr<float>());
    float2* pp2 = reinterpret_cast<float2*>(p2p);
    const int nbA = direct || made ? 0 : (Mp * (KP / 2) + 255) / 256;
    const int nbB = direct || made ? 0 : (KP * (Np / 2) + 255) / 256;
    // WF made by the main kernel's producer where the prologue would run for it alone (the copy at hand, or the factors
    // read in place: its launch saved); else by the prologue converting anyway (MEASURED L 1024, B 1024: the producer's
    // gathers made the main kernel 13% slower, the prologue's WF all but free beside its conversion)
    const bool wpr = wp && (direct || made);
    const int nbW = wpr ? 0 : (njt + 7) / 8;
    const int64_t sA = A.stride(0), sB = Bm.stride(0);
    for (int64_t b0 = 0; b0 < B; b0 += BC) {
        const int Bc = (int)std::min<int64_t>(BC, B - b0);
        const int64_t c0 = keep || made ? b0 : 0;              // (the chunk's first sample in the copy)
        unsigned* ap = direct ? nullptr : reinterpret_cast<unsigned*>(AP.data_ptr<at::Half>()) + c0 * Mp * (KP / 2);
        unsigned* bp = direct ? nullptr : reinterpret_cast<unsigned*>(BP.data_ptr<at::Half>()) + c0 * KP * (Np / 2);
        if (nbA + nbB + nbW > 0) {                             // (the prologue: none where nothing is to convert, the
            HF_TI_SWITCH(A, Bm, b0, launch_h(pdl, hf_fw_prep_kernel<KP, TI, NT8>, dim3(nbA + nbB + nbW, (unsigned)Bc),
                dim3(256), 0, stream, ai, bi, ap, bp, wfp, lcp,
                emars.template data_ptr<float>(), rw, (int)M, (int)N, (int)K, Mp, Np, nbA, nbB, (int)B, (int)b0,
                (long)sA, (long)sB));                          //  producer making WF)
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const HfWP wq{emars.template data_ptr<float>(), rw, wfp, lcp,
                      (int)N, Np, (int)B, (int)b0};
        const CUtensorMap tma = direct ? h_map<KP>(reinterpret_cast<const __half*>(A.data_ptr<at::Half>()) + b0 * sA, true,
                                                   M, sstride(A), Bc)
                                       : h_map<KP>(ap, true, Mp, (int64_t)Mp * KP, Bc);
        const CUtensorMap tmb = direct ? h_map<KP>(reinterpret_cast<const __half*>(Bm.data_ptr<at::Half>()) + b0 * sB,
                                                   false, N, sstride(Bm), Bc)
                                       : h_map<KP>(bp, false, Np, (int64_t)KP * Np, Bc);
        const HbOps op = direct ? HbOps{reinterpret_cast<const __half*>(A.data_ptr<at::Half>()) + b0 * sA, (long)sstride(A),
                                        (int)M, reinterpret_cast<const __half*>(Bm.data_ptr<at::Half>()) + b0 * sB,
                                        (long)sstride(Bm), (int)N, (int)N, 1}
                                : HbOps{reinterpret_cast<const __half*>(ap), (long)Mp * KP, Mp,
                                        reinterpret_cast<const __half*>(bp), (long)KP * Np, Np, Np, 0};
        const long total = (long)njt * nit * Bc;
        if (ah && wpr) {
            const HfAhead ha{ahead[0].data_ptr<float>() + b0 * sstride(ahead[0]),
                             ahead[1].data_ptr<float>() + b0 * sstride(ahead[1]), (long)sstride(ahead[0]),
                             (long)sstride(ahead[1]),
                             reinterpret_cast<__half*>(AP2.data_ptr<at::Half>()) + b0 * Mp2 * KP2,
                             reinterpret_cast<__half*>(BP2.data_ptr<at::Half>()) + b0 * KP2 * Np2,
                             Mp2 * KP2 / HAP, KP2 * Np2 / HAP, Bc};
            launch_hf_fw<KP, NT8, NST, true, true>(pdl, stream, tma, tmb, op.scaleA, wfp, ltf,
                                                   pp2, (int)M, Mp, Bc, njt, total, ha, wq);
        } else if (wpr) {
            launch_hf_fw<KP, NT8, NST, false, true>(pdl, stream, tma, tmb, op.scaleA, wfp, ltf,
                                                    pp2, (int)M, Mp, Bc, njt, total, HfAhead{}, wq);
        } else if (ah) {
            const HfAhead ha{ahead[0].data_ptr<float>() + b0 * sstride(ahead[0]),
                             ahead[1].data_ptr<float>() + b0 * sstride(ahead[1]), (long)sstride(ahead[0]),
                             (long)sstride(ahead[1]),
                             reinterpret_cast<__half*>(AP2.data_ptr<at::Half>()) + b0 * Mp2 * KP2,
                             reinterpret_cast<__half*>(BP2.data_ptr<at::Half>()) + b0 * KP2 * Np2,
                             Mp2 * KP2 / HAP, KP2 * Np2 / HAP, Bc};
            launch_hf_fw<KP, NT8, NST, true>(pdl, stream, tma, tmb, op.scaleA, wfp, ltf, pp2,
                                             (int)M, Mp, Bc, njt, total, ha, wq);
        } else {
            launch_hf_fw<KP, NT8, NST>(pdl, stream, tma, tmb, op.scaleA, wfp, ltf, pp2, (int)M,
                                       Mp, Bc, njt, total, HfAhead{}, wq);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // (the combine, the rare items' paths: 2 lanes an output, 4 where that grid would leave an SM fewer than 2
        // blocks)
        const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        const bool wide = (long)((M + 31) / 32) * ((Bc + 3) / 4) < 2L * sms;
        const dim3 cg((unsigned)((M + 31) / 32), (unsigned)((Bc + (wide ? 1 : 3)) / (wide ? 2 : 4)));
        HF_TP_SWITCH(params, {
            auto comb = wide ? hf_comb_kernel<TP, KP, NT8, 4> : hf_comb_kernel<TP, KP, NT8, 2>;
            const HfRare<TP> ra{op.a, op.sAb, op.am, op.b, op.sBb, op.bk, op.bn, op.scaleA, wfp,
                                Gm::WFW, lcp, pp, p0, (int)Nb, (int)cbs, (int)bs, (int)M,
                                (int)N, Np, njt};
            launch_h(pdl, comb, cg, dim3(256), 0, stream, pp2, CMR.template data_ptr<float>(), wfp,
                     Gm::WFW, 8 * NT8, njt, node_mars.template data_ptr<float>(), nid0, lzs.template data_ptr<float>(),
                     (int)M, Mp, Bc, (int)B, (int)b0, ra);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const auto none = torch::empty({0}, emars.options().dtype(torch::kHalf));
    return {lzs, keep && !direct ? AP : none, keep && !direct ? BP : none, ah ? AP2 : none, ah ? BP2 : none};
}

// The tiles' theta fragments of a node vector's parameters for the forward at tile width `tn` (made once a forward
// pass for every node vector reading the same run: tied copies): {LTF, CMR}
std::vector<torch::Tensor> exp_sum_h_theta(torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs, int64_t bs,
                                           int64_t M, int64_t N, int64_t tn) {
    TORCH_CHECK(params.scalar_type() == torch::kFloat32 || params.scalar_type() == torch::kBFloat16, "params: fp32 or bf16");
    TORCH_CHECK(tn == 64, "tn: 64");
    return theta_impl<8>(params, p0, Nb, cbs, bs, M, N);
}

// The forward of one node vector (`theta`: exp_sum_h_theta's fragments of its parameters, or empty: made here; `keep`:
// return the converted copy of factors not read in place, for the backward; `given`: that copy made already, by the
// forward before -- its `ahead`; `ahead`: the next node vector's fp32 factors, converted into its copy under this
// forward's main kernel where they can be): {log Z [M, B], AP, BP, AP2, BP2} (`forward_impl`)
std::vector<torch::Tensor> exp_sum_h_forward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0,
                                             int64_t Nb, int64_t cbs, int64_t bs, torch::Tensor emars,
                                             torch::Tensor rows, torch::Tensor node_mars, int64_t nid0, int64_t M,
                                             int64_t N, int64_t K, int64_t tn, std::vector<torch::Tensor> theta,
                                             bool keep, std::vector<torch::Tensor> given,
                                             std::vector<torch::Tensor> ahead, bool pdl, bool wp) {
    const int64_t B = node_mars.size(1);
    check_common(A, Bm, params, emars, rows, M, N, K, B);
    TORCH_CHECK(node_mars.scalar_type() == torch::kFloat32 && node_mars.is_contiguous(), "node_mars");
    TORCH_CHECK(tn == 64, "tn: 64");
    const bool frags = theta.size() == 2;
    torch::Tensor ltf = frags ? theta[0] : torch::empty({0}, emars.options());
    torch::Tensor cmr = frags ? theta[1] : torch::empty({0}, emars.options());
    switch (rank_pad_h(K)) {                                   // (the factors' rank padded: 16, 32, 64)
        case 16: return forward_impl<16, 8>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_mars, nid0, M, N, K, ltf, cmr,
                                            keep, given, ahead, pdl, wp);
        case 32: return forward_impl<32, 8>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_mars, nid0, M, N, K, ltf, cmr,
                                            keep, given, ahead, pdl, wp);
        default: return forward_impl<64, 8>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_mars, nid0, M, N, K, ltf,
                                            cmr, keep, given, ahead, pdl, wp);
    }
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

// dB's destination as a 3-D tensor of fp32 -- n children (a rank's row, contiguous), k ranks `ldb` apart, bc samples
// `sgb` apart (in floats; 16-byte multiples) -- in boxes of the backward's 64 children x k ranks x 1 sample (the first
// k rows of a CTA's partial [KP][64]: the caller's ranks, or all KP of a temporary)
static CUtensorMap db_map(float* base, int64_t n, int64_t k, int64_t ldb, int64_t sgb, int64_t bc) {
    CUtensorMap m;
    cuuint64_t dim[3] = {(cuuint64_t)n, (cuuint64_t)k, (cuuint64_t)bc};
    cuuint64_t str[2] = {(cuuint64_t)ldb * 4, (cuuint64_t)sgb * 4};
    cuuint32_t box[3] = {(cuuint32_t)TN2, (cuuint32_t)k, 1};
    cuuint32_t es[3] = {1, 1, 1};
    const CUresult r = tmap_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, base, dim, str, box, es,
                                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                     CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "dB's tensor map: cuTensorMapEncodeTiled failed (", (int)r, ")");
    return m;
}

// dA's destination where its rows are narrower than a CTA's partial rows (the caller's K < KP ranks, K a multiple of
// 4): fp32 [bc][m][K] (rows K floats, samples `sga` floats apart), in boxes of KP ranks x 128 rows x 1 sample -- the
// partial [128][KP] as it lies, its ranks past K and rows past m dropped
static CUtensorMap da_map(float* base, int64_t m, int64_t K, int64_t sga, int64_t bc, int KP) {
    CUtensorMap mp;
    cuuint64_t dim[3] = {(cuuint64_t)K, (cuuint64_t)m, (cuuint64_t)bc};
    cuuint64_t str[2] = {(cuuint64_t)K * 4, (cuuint64_t)sga * 4};
    cuuint32_t box[3] = {(cuuint32_t)KP, (cuuint32_t)TM2, 1};
    cuuint32_t es[3] = {1, 1, 1};
    const CUresult r = tmap_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, base, dim, str, box, es,
                                     CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                     CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "dA's tensor map: cuTensorMapEncodeTiled failed (", (int)r, ")");
    return mp;
}


// The backward on its persistent grid: the GPU's SMs times the CTAs an SM holds (asked once a device), at most a CTA a
// unit
template <typename TP, bool FL, bool DE, bool GR, bool DM, bool ONE = false>
void launch_hb_bw(bool pdl, cudaStream_t stream, const HbOps& op, const TP* pp, long p0, int Nb, int cbs, int bs,
                  const float* LU,
                  const float* LC, float* ef, float* pf, long pf0, float* dnp, long dn0, float* ga, long sga, int ma,
                  const CUtensorMap& tmb, const CUtensorMap& tma, const float4* sc, int M, int N, int Mp, int Np, int B,
                  int nct, long units) {
    constexpr int bytes = Hb2Geo::WORDS * 4;
    auto kern = hb_bw_kernel<TP, FL, DE, GR, DM, ONE>;
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
    launch_h(pdl, kern, dim3((unsigned)grid), dim3(256), bytes, stream, op.a, op.sAb, op.am, op.b, op.sBb, op.bk, op.bn,
             op.scaleA, pp, p0, Nb, cbs, bs, LU, LC, ef, pf, pf0, dnp, dn0, ga, sga, ma, tmb, tma, sc, M, N, Mp, Np, B,
             nct, units);
}

// ... at ranks padded to 32 or 64 (`hbk_bw_kernel`)
template <int KP, typename TP, bool FL, bool DE, bool GR, bool ONE = false>
void launch_hbk_bw(bool pdl, cudaStream_t stream, const HbOps& op, const TP* pp, long p0, int Nb, int cbs, int bs,
                   const float* LU,
                   const float* LC, float* ef, float* pf, long pf0, float* dnp, long dn0, float* ga, long sga, int ma,
                   int ka, int kv, const CUtensorMap& tmb, const CUtensorMap& tma, int dmap, const float4* sc, int M,
                   int N, int Mp, int Np, int B, int nct, long units) {
    constexpr int bytes = HbkGeo<KP>::WORDS * 4;
    auto kern = hbk_bw_kernel<KP, TP, FL, DE, GR, ONE>;
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
    launch_h(pdl, kern, dim3((unsigned)grid), dim3(256), bytes, stream, op.a, op.sAb, op.am, op.b, op.sBb, op.bk, op.bn,
             op.scaleA, pp, p0, Nb, cbs, bs, LU, LC, ef, pf, pf0, dnp, dn0, ga, sga, ma, ka, kv, tmb, tma, dmap, sc, M,
             N, Mp, Np, B, nct, units);
}

// The backward of one node vector at KP ranks (its K padded; see exp_sum_h_backward)
template <int KP>
void backward_impl(const torch::Tensor& A, const torch::Tensor& Bm, const torch::Tensor& params, int64_t p0, int64_t Nb,
                   int64_t cbs, int64_t bs, const torch::Tensor& emars, const torch::Tensor& rows,
                   const torch::Tensor& node_flows, const torch::Tensor& node_mars, int64_t nid0, const torch::Tensor& lzs,
                   const torch::Tensor& element_flows, const torch::Tensor& pflows, int64_t pf0, const torch::Tensor& denom,
                   int64_t dn0, const torch::Tensor& gA, const torch::Tensor& gB, int64_t M, int64_t N, int64_t K,
                   const std::vector<torch::Tensor>& copy, bool init, bool store, bool pdl, const torch::Tensor& tab,
                   int64_t first, bool one_tile) {
    const int64_t B = node_mars.size(1);
    const bool flows = pflows.numel() > 0, dn = denom.numel() > 0, grads = gA.numel() > 0;
    const at::cuda::CUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const long* rw = reinterpret_cast<const long*>(rows.data_ptr<int64_t>());
    float* pf = flows ? pflows.data_ptr<float>() : nullptr;
    float* dnp = dn ? denom.data_ptr<float>() : nullptr;
    const int Mp2 = (int)((M + TM2 - 1) / TM2 * TM2), Np2 = (int)((N + TN2 - 1) / TN2 * TN2), njt2 = Np2 / BN;
    const int nct = Np2 / TN2, nit2 = Mp2 / TM2;
    const bool given = copy.size() == 2 && copy[0].numel() > 0;
    if (given)
        TORCH_CHECK(copy[0].scalar_type() == torch::kHalf && copy[0].is_contiguous() && copy[0].numel() == B * Mp2 * KP &&
                    copy[1].scalar_type() == torch::kHalf && copy[1].is_contiguous() && copy[1].numel() == B * KP * Np2,
                    "copy: the forward's {AP [B, Mp, KP], BP [B, KP, Np]} of these factors");
    const bool direct = !given && h_in_place(A, Bm, M, N, K, KP), convert = !given && !direct;
    // (a chunk's operands within the L2 beside its fp32 traffic -- the gradients' adds, a conversion's reads -- however
    // they come: read from the copy or in place, the operands of a chunk the whole batch's would be evicted between
    // their 16 / 8 readers, +28% at B1024)
    const int64_t BC = sample_chunk(B, 2.0 * KP * (Mp2 + Np2) + 8.0 * K * (M + N));
    torch::Tensor AP2, BP2;                                    // (converted here: a chunk's copy)
    if (convert) {
        AP2 = torch::empty({BC, (long)Mp2 * KP}, emars.options().dtype(torch::kHalf));
        BP2 = torch::empty({BC, (long)KP * Np2}, emars.options().dtype(torch::kHalf));
    }
    // the scratch, one allocation: LU [B, Mp, 2], LC [B, njt, 36], the element flows' sums [B, Np], the samples' scales
    // [B, 4] -- each piece 16-byte aligned
    const long nLU = BC * Mp2 * 2, nLC2 = BC * njt2 * 36, nEF = BC * Np2;
    auto scratch = torch::empty({nLU + nLC2 + nEF + BC * 4}, emars.options());
    float* const lu2 = scratch.data_ptr<float>(), * const lc2 = lu2 + nLU, * const ef2 = lc2 + nLC2;
    float4* sc2 = reinterpret_cast<float4*>(ef2 + nEF);
    const int nbA2 = convert ? (int)(((long)Mp2 * (KP / 2) + 255) / 256) : 0;
    const int nbB2 = convert ? (int)(((long)KP * (Np2 / 2) + 255) / 256) : 0;
    // the gradients' bulk adds take the caller's buffers where each sample's rows lie 16-byte aligned -- dA's [M, K]
    // rows (K a multiple of 4: KP wide, one run of the tile's rows; narrower, a box of a tensor map, `dmap`), dB's K rank
    // rows of N --; else padded temporaries [B, Mp2, KP], [B, KP, Np2], added in afterwards
    const auto al16 = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
    const bool dga = grads && al16(gA) && gA.stride(0) % 4 == 0 && K % 4 == 0;
    const bool dgb = grads && al16(gB) && gB.stride(0) % 4 == 0 && N % 4 == 0;
    const int dmap = dga && K < KP && KP <= 32;                // (KP 64: dA by reductions from registers, any K)
    torch::Tensor tA2, tB2;
    if (grads && !dga) tA2 = torch::zeros({BC, Mp2, KP}, emars.options());
    if (grads && !dgb) tB2 = torch::zeros({BC, KP, Np2}, emars.options());
    const long sga = !grads ? 0L : dga ? (long)gA.stride(0) : (long)Mp2 * KP;
    const long sgb = !grads ? 0L : dgb ? (long)gB.stride(0) : (long)KP * Np2;
    const int ma = dga ? (int)M : Mp2, ldb = dgb ? (int)N : Np2, nbw = dgb ? (int)N : Np2, kb = dgb ? (int)K : KP;
    const int ka = dga ? (int)K : KP;                          // (dA's destination rows, their live ranks)
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
        // (one row tile, training -- gradients and parameter flows, this backward the destinations' first writer --:
        // each sample's dB and element-flow sums of a column tile one CTA's, stored -- dB not zeroed first. MEASURED
        // L 128: the backward -0.11 .. -0.18 ms, ranks 16 and 32, B 1024 and 2048)
        const bool one = one_tile && nit2 == 1 && grads && flows && init && dgb;
        const int nA4 = init && dga ? (int)(M * K / 4) : 0, nB4 = init && dgb && !one ? (int)(K * N / 4) : 0;
        const int nbZ = (nA4 + nB4 + 4095) / 4096;
        float4* zA = nA4 ? reinterpret_cast<float4*>(gA.data_ptr<float>() + b0 * gA.stride(0)) : nullptr;
        float4* zB = nB4 ? reinterpret_cast<float4*>(gB.data_ptr<float>() + b0 * gB.stride(0)) : nullptr;
        const int nbg = (Bc + 31) / 32;
        const long nprep = (long)(nbZ + nbA2 + nbB2) * Bc + (Mp2 / 32 + njt2) * nbg + (Bc + 7) / 8;
        HF_TI_SWITCH(A, Bm, b0, launch_h(pdl, hb_bw_prep_kernel<KP, TI>, dim3((unsigned)nprep), dim3(256), 0, stream,
            ai, bi, ap, bp, lu2, lc2, ef2, sc2,
            node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), nid0, lzs.data_ptr<float>(),
            emars.data_ptr<float>(), rw, (int)M, (int)N, (int)K, Mp2, Np2, nbA2, nbB2, Bc, (int)B, (int)b0,
            (long)A.stride(0), (long)Bm.stride(0), zA, nA4 ? gA.stride(0) / 4 : 0L, nA4, zB, nB4 ? gB.stride(0) / 4 : 0L,
            nB4, nbZ, tab.numel() > 0 ? tab.data_ptr<int>() : nullptr, (long)first));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (tA2.defined()) tA2.zero_();
        if (tB2.defined()) tB2.zero_();
        HbOps op;
        if (given)
            op = HbOps{reinterpret_cast<const __half*>(copy[0].data_ptr<at::Half>()) + b0 * Mp2 * KP, (long)Mp2 * KP, Mp2,
                       reinterpret_cast<const __half*>(copy[1].data_ptr<at::Half>()) + b0 * KP * Np2, (long)KP * Np2,
                       Np2, Np2, 0};
        else if (direct)
            op = HbOps{reinterpret_cast<const __half*>(A.data_ptr<at::Half>()) + b0 * A.stride(0), (long)sstride(A),
                       (int)M, reinterpret_cast<const __half*>(Bm.data_ptr<at::Half>()) + b0 * Bm.stride(0),
                       (long)sstride(Bm), (int)N, (int)N, 1};
        else
            op = HbOps{reinterpret_cast<const __half*>(ap), (long)Mp2 * KP, Mp2, reinterpret_cast<const __half*>(bp),
                       (long)KP * Np2, Np2, Np2, 0};
        float* ga2 = !grads ? nullptr : dga ? gA.data_ptr<float>() + b0 * gA.stride(0) : tA2.data_ptr<float>();
        float* gb2 = !grads ? nullptr : dgb ? gB.data_ptr<float>() + b0 * gB.stride(0) : tB2.data_ptr<float>();
        CUtensorMap tmb, tma;
        if (grads) tmb = db_map(gb2, nbw, kb, ldb, sgb, Bc);
        else std::memset(&tmb, 0, sizeof(tmb));
        if (dmap) tma = da_map(ga2, M, K, sga, Bc, KP);
        else std::memset(&tma, 0, sizeof(tma));
        const long units = (long)nit2 * nct * Bc;
        if (one) {                                             // (FL, GR: `one`'s; DE either)
#define HB_ONE(DE, DM)                                                                                             \
            launch_hb_bw<TP, true, DE, true, DM, true>(pdl, stream, op, pp, p0, (int)Nb, (int)cbs, (int)bs, lu2, lc2, \
                ef2, pf, pf0, dnp, dn0, ga2, sga, ma, tmb, tma, sc2, (int)M, (int)N, Mp2, Np2, Bc, nct, units)
#define HBK_ONE(DE)                                                                                                \
            launch_hbk_bw<KP, TP, true, DE, true, true>(pdl, stream, op, pp, p0, (int)Nb, (int)cbs, (int)bs, lu2,      \
                lc2, ef2, pf, pf0, dnp, dn0, ga2, sga, ma, ka, ka, tmb, tma, dmap, sc2, (int)M, (int)N, Mp2, Np2, Bc,  \
                nct, units)
            HF_TP_SWITCH(params, {
                if constexpr (KP == 16) {
                    if (dmap) {
                        if (dn) HB_ONE(true, true);
                        else HB_ONE(false, true);
                    } else {
                        if (dn) HB_ONE(true, false);
                        else HB_ONE(false, false);
                    }
                } else {
                    if (dn) HBK_ONE(true);
                    else HBK_ONE(false);
                }
            })
#undef HB_ONE
#undef HBK_ONE
        } else if constexpr (KP == 16) {
            if (dmap) {                                        // (the gradients wanted: dmap needs them)
                HF_TP_SWITCH(params, HB_FLAGS_SWITCH(flows, dn, true, launch_hb_bw<TP, FL, DE, true, true>(pdl,
                    stream, op, pp, p0, (int)Nb, (int)cbs, (int)bs, lu2, lc2, ef2, pf, pf0, dnp, dn0, ga2, sga, ma, tmb,
                    tma, sc2, (int)M, (int)N, Mp2, Np2, Bc, nct, units)))
            } else {
                HF_TP_SWITCH(params, HB_FLAGS_SWITCH(flows, dn, grads, launch_hb_bw<TP, FL, DE, GR, false>(pdl,
                    stream, op, pp, p0, (int)Nb, (int)cbs, (int)bs, lu2, lc2, ef2, pf, pf0, dnp, dn0, ga2, sga, ma, tmb,
                    tma, sc2, (int)M, (int)N, Mp2, Np2, Bc, nct, units)))
            }
        } else {
            HF_TP_SWITCH(params, HB_FLAGS_SWITCH(flows, dn, grads, launch_hbk_bw<KP, TP, FL, DE, GR>(pdl, stream, op,
                pp, p0, (int)Nb, (int)cbs, (int)bs, lu2, lc2, ef2, pf, pf0, dnp, dn0, ga2, sga, ma, ka, ka, tmb, tma,
                dmap, sc2, (int)M, (int)N, Mp2, Np2, Bc, nct, units)))
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        launch_h(pdl, hb_ef_epi_kernel, dim3((unsigned)((((long)N << lbc) + 255) / 256)), dim3(256), 0, stream,
                 (const float*)ef2, rw, element_flows.data_ptr<float>(), (int)N, Np2, Bc, lbc, (int)B,
                 (int)b0, store ? 1 : 0, tab.numel() > 0 ? tab.data_ptr<int>() : nullptr, (long)first,
                 node_flows.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (tA2.defined()) {                                   // (the temporaries' sums into the caller's buffers)
            auto src = tA2.narrow(0, 0, Bc).narrow(1, 0, M).narrow(2, 0, K);
            if (init) gA.narrow(0, b0, Bc).copy_(src);
            else gA.narrow(0, b0, Bc).add_(src);
        }
        if (tB2.defined()) {
            auto src = tB2.narrow(0, 0, Bc).narrow(1, 0, K).narrow(2, 0, N);
            if (init) gB.narrow(0, b0, Bc).copy_(src);
            else gB.narrow(0, b0, Bc).add_(src);
        }
    }
}

// The backward of one node vector: the element flows of its children log-added into `element_flows` (the caller sets
// them to -inf first, once a layer), and -- each where its tensor is given (not empty) -- F+ added into `param_flows`
// from `pf0`, W into `denom` from `dn0`, dLL/dA and dLL/dB into the gradient buffers (fp32, the caller's layout, [B,
// M, K], [B, K, N]): `lowrank_exp_sum.cu`'s `exp_sum_backward` with phi of the fp16-rounded factors (their K ranks
// padded to 16, 32 or 64). Its operands: `copy` = the forward's {AP, BP} where given, else the factors in place (fp16,
// aligned: `h_in_place`), else converted here a chunk at a time. `init`: the gradient buffers hold anything (their
// first writer: every element written, not added into). `store`: the element flows of the node vector's children
// written, not log-added (no other node vector of the layer reaches them; their rows need no -inf fill first)
void exp_sum_h_backward(torch::Tensor A, torch::Tensor Bm, torch::Tensor params, int64_t p0, int64_t Nb, int64_t cbs,
                        int64_t bs, torch::Tensor emars, torch::Tensor rows, torch::Tensor node_flows,
                        torch::Tensor node_mars, int64_t nid0, torch::Tensor lzs, torch::Tensor element_flows,
                        torch::Tensor pflows, int64_t pf0, torch::Tensor denom, int64_t dn0, torch::Tensor gA,
                        torch::Tensor gB, int64_t M, int64_t N, int64_t K, std::vector<torch::Tensor> copy, bool init,
                        bool store, bool pdl, std::optional<torch::Tensor> ochildren, int64_t first, bool one_tile) {
    torch::Tensor children = ochildren.has_value() ? *ochildren : torch::empty({0}, A.options().dtype(torch::kInt32));
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
    TORCH_CHECK(children.numel() == 0 ||
                (children.scalar_type() == torch::kInt32 && children.is_contiguous() && children.dim() == 2 &&
                 children.size(1) == 2 && children.device() == A.device()),
                "children: the product group's table, int32 [n, 2]");
    if (flows) TORCH_CHECK(pflows.scalar_type() == torch::kFloat32 && pflows.is_contiguous(), "param_flows");
    if (dn) TORCH_CHECK(denom.scalar_type() == torch::kFloat32 && denom.is_contiguous(), "denominators");
    if (grads) {
        TORCH_CHECK(gB.numel() > 0, "both gradients or neither");
        TORCH_CHECK(gA.scalar_type() == torch::kFloat32 && inner_contiguous(gA) && gA.sizes() == A.sizes(),
                    "dA: fp32 [B, M, K], a sample's contiguous");
        TORCH_CHECK(gB.scalar_type() == torch::kFloat32 && inner_contiguous(gB) && gB.sizes() == Bm.sizes(),
                    "dB: fp32 [B, K, N], a sample's contiguous");
    }
    switch (rank_pad_h(K)) {
        case 16: backward_impl<16>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_flows, node_mars, nid0, lzs,
                                   element_flows, pflows, pf0, denom, dn0, gA, gB, M, N, K, copy, init, store, pdl,
                                   children, first, one_tile); break;
        case 32: backward_impl<32>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_flows, node_mars, nid0, lzs,
                                   element_flows, pflows, pf0, denom, dn0, gA, gB, M, N, K, copy, init, store, pdl,
                                   children, first, one_tile); break;
        default: backward_impl<64>(A, Bm, params, p0, Nb, cbs, bs, emars, rows, node_flows, node_mars, nid0, lzs,
                                   element_flows, pflows, pf0, denom, dn0, gA, gB, M, N, K, copy, init, store, pdl,
                                   children, first, one_tile); break;
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_sum_h_theta", &exp_sum_h_theta,
          "the forward's tiles' theta fragments of a node vector's parameters (make once, pass to exp_sum_h_forward)");
    m.def("exp_sum_h_forward", &exp_sum_h_forward,
          "a node vector's values (and log Z, the factors' fp16 copy where kept) under the exp_sum scale of fp16-rounded "
          "factors, the scale made tile by tile",
          py::arg("A"), py::arg("Bm"), py::arg("params"), py::arg("p0"), py::arg("Nb"), py::arg("cbs"), py::arg("bs"),
          py::arg("emars"), py::arg("rows"), py::arg("node_mars"), py::arg("nid0"), py::arg("M"), py::arg("N"),
          py::arg("K"), py::arg("tn"), py::arg("theta"), py::arg("keep"),
          py::arg("given") = std::vector<torch::Tensor>(), py::arg("ahead") = std::vector<torch::Tensor>(),
          py::arg("pdl") = true, py::arg("wp") = true);
    m.def("exp_sum_h_backward", &exp_sum_h_backward,
          "a node vector's element flows, F+, W and dLL/dA, dLL/dB under the exp_sum scale of fp16-rounded factors",
          py::arg("A"), py::arg("Bm"), py::arg("params"), py::arg("p0"), py::arg("Nb"), py::arg("cbs"), py::arg("bs"),
          py::arg("emars"), py::arg("rows"), py::arg("node_flows"), py::arg("node_mars"), py::arg("nid0"),
          py::arg("lzs"), py::arg("element_flows"), py::arg("pflows"), py::arg("pf0"), py::arg("denom"),
          py::arg("dn0"), py::arg("gA"), py::arg("gB"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("copy"),
          py::arg("init"), py::arg("store") = false, py::arg("pdl") = true, py::arg("children") = py::none(),
          py::arg("first") = 0, py::arg("one_tile") = true);
}
