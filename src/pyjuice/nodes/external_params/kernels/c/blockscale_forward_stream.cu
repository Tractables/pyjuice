// The `BlockScaleSumParams` forward (bf16 parameters) as a pre-pass, one GEMM launch and a merge -- what the pre-pass
// copy (blockscale_forward_prepass.cu) computes, not to its bits:
//
//     node_mars[n, b] = log sum_c theta[n, c] phi_b[g(c)] exp(em[c, b])  -  log Z[n, b]
//     Z[n, b]         = sum_c theta[n, c] phi_b[g(c)]  =  sum_g sigma[n, g] phi_b[g]
//
// with sigma the per-gate parameter sums (`_DenominatorsMixin._sigma`). The fork and its pre-pass copy walk a row's
// edge tiles serially, each tile under its own stabilizer, merging the running sums tile by tile, and they carry the
// normalizer as a second MMA over every child. Here:
//   pre-pass  per (row, split, PB columns): the row's children are cut into `split` contiguous ranges of k-steps; over
//             a range, s = em + log phi (the gate folded in as the fork folds it), its max Mp, and u = bf16 exp(s - Mp)
//             (0 where Mp is -inf), laid out [row, child, b]. One stabilizer per range makes the GEMM a plain one;
//   GEMM      per (batch tile, node tile, split): L_p = theta[tile, range] . u, bf16 MMA in fp32 over the whole range
//             (producers cp.async the parameters and u into mbarrier'd stages, the consumers ldmatrix and MMA -- no
//             per-tile merges), stored as the range's partial. Blocks past the GEMM's (the SMs it leaves idle) compute
//             log Z from sigma: chunks of GC gates, the gates' log phi under their running max (the accumulator
//             rescaled where it grows), Z += sigma . exp(log phi - mz) as three bf16 MMAs on hi + lo halves (lo.hi +
//             hi.lo + hi.hi: ~2^-17 relative, against the bf16 normalizer operand of the fork);
//   merge     per 4 columns of a node: L = sum_p L_p exp(Mp - M), M = max_p Mp; the fork's epilogue on log L + M and
//             log Z (-inf where Z is 0 or N is -inf).
// Every sum is in a fixed order: the same bits from run to run.
//
// MEASURED (RTX PRO 6000, a gated HMM layer, bf16, the plan's choices against the pre-pass copy's best config, over
// L 1024 / 2048 / 4096 x 1 or 4 node blocks x B 64 .. 512 x gate width 16 / 64): L 4096 1.18 - 1.95x (gate width 64
// 1.48 - 1.95x, e.g. 4 blocks of 1024 B 128 46.8 -> 27.1 us), L 2048 0.96 - 1.72x, L 1024 0.80 - 1.11x (three short
// launches; the plan keeps the faster). Against float64 the node values' max error 2 - 5e-4 (the pre-pass copy's
// 5 - 15e-4), log Z's 3e-6 (8e-4).
//
// Scope (`fitting_configs`, and the Python plan for the rest): bf16 parameters with 8-aligned runs, one gate per node
// block, node_cbs >= 64 (an edge tile is one run of children), a power-of-two gate width >= 4 dividing node_cbs,
// batch % 64 == 0, block_size % 64 == 0 and a multiple of the tile's rows, split 1 .. 8. Scratch: one allocation per
// call (inside a CUDA graph capture, from its pool).

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"   // `param_ptr<TP>`: the stored dtype
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cmath>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64           // edge tile: one run of children (cids / pids one entry each)
#define CH 32           // the pre-pass's chunk: half an edge tile
#define GC 64           // log Z's chunk of gates
#define MAX_SPLIT 8

__device__ __forceinline__ void cpa16(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
// arrives on `bar` once every cp.async this thread issued before has landed (the barrier's count includes it)
__device__ __forceinline__ void cpa_mbar_arrive(uint64_t* bar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{\n .reg .pred p;\n W_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra W_%=;\n}\n" ::"r"(a), "r"(phase) : "memory");
}

// ---------------- pre-pass: a thread block per (row, split, PB columns) ----------------
// Thread (part, b) takes the split's 32-child chunks part, part + PARTS, ...: a chunk is one run of `element_mars` rows
// (half an edge tile) under CH / gc gates (gc = min(gate width, CH)), its s = em + log phi in registers; the split's
// max Mp over every chunk (shared memory); u = bf16 exp(s - Mp). A split past PARTS chunks loads its later chunks
// twice (the max, then the exponentials).
template <int NT, int PB>
__global__ void __launch_bounds__(NT) bs_stream_prepass_kernel(
        const float* __restrict__ em, const long* __restrict__ cids, const float* __restrict__ ext,
        const long* __restrict__ gate, __nv_bfloat16* __restrict__ u, float* __restrict__ mp, int batch, int num_edges,
        int knt, int ksteps, int kt_sh, int split, int gate_stride, int node_sh, int gate_sh, int node_cbs,
        long ext_base, int gc) {
    constexpr int PARTS = NT / PB, NG = CH / 4;          // a gate spans >= 4 children
    __shared__ float smx[PARTS][PB];
    const int tid = threadIdx.x, nbc = batch / PB;
    const int r = blockIdx.x / (split * nbc), sp = (blockIdx.x / nbc) % split, b0 = (blockIdx.x % nbc) * PB;
    const int b = tid % PB, part = tid / PB;
    const int q0 = (int)((((long)sp * ksteps / split) << kt_sh) / CH);
    const int q1 = (int)((((long)(sp + 1) * ksteps / split) << kt_sh) / CH);
    const long* ct = cids + (long)r * knt;
    const long* gt = gate + (long)r * gate_stride;
    const int gpc = CH / gc;                             // gates a chunk
    auto load_chunk = [&](int q, float* v) {
        const int pos0 = q * CH;
        float lq[NG];
        CUTE_UNROLL
        for (int k = 0; k < NG; k++) {
            if (k < gpc) {
                const int pos = pos0 + k * gc, j = pos >> node_sh, d = (pos & (node_cbs - 1)) >> gate_sh;
                const long gb = (j < gate_stride) ? gt[j] : -1;
                lq[k] = (gb >= 0) ? ext[(gb + ext_base + d) * (long)batch + b0 + b] : -INFINITY;
            }
        }
        const float* er = em + (ct[pos0 / BK] + pos0 % BK) * (long)batch + b0 + b;
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) v[i] = er[(long)i * batch];
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) {
            float l = lq[0];
            CUTE_UNROLL
            for (int k = 1; k < NG; k++)
                if (k < gpc && i >= k * gc) l = lq[k];
            v[i] += l;
        }
    };
    float v[CH];
    float m = -INFINITY;
    const int q = q0 + part;
    if (q < q1) {
        load_chunk(q, v);
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) m = fmaxf(m, v[i]);
    }
    for (int qq = q + PARTS; qq < q1; qq += PARTS) {
        float w[CH];
        load_chunk(qq, w);
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) m = fmaxf(m, w[i]);
    }
    smx[part][b] = m;
    __syncthreads();
    m = smx[0][b];
    CUTE_UNROLL
    for (int k = 1; k < PARTS; k++) m = fmaxf(m, smx[k][b]);
    __nv_bfloat16* ur = u + ((long)r * num_edges) * batch + b0 + b;
    if (q < q1) {
        CUTE_UNROLL
        for (int i = 0; i < CH; i++)
            ur[(long)(q * CH + i) * batch] = __float2bfloat16_rn(m == -INFINITY ? 0.f : __expf(v[i] - m));
    }
    for (int qq = q + PARTS; qq < q1; qq += PARTS) {
        float w[CH];
        load_chunk(qq, w);
        CUTE_UNROLL
        for (int i = 0; i < CH; i++)
            ur[(long)(qq * CH + i) * batch] = __float2bfloat16_rn(m == -INFINITY ? 0.f : __expf(w[i] - m));
    }
    if (part == 0) mp[((long)r * split + sp) * batch + b0 + b] = m;
}

extern __shared__ __align__(1024) char smem_raw[];

// The MN-major smem atom of an operand with 64 rows (m or b) a k: 128-byte lines, swizzled over the 16-byte chunks so
// the transposed ldmatrix's 8 k-rows of one chunk fall in distinct banks.
using Atom64 = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));

// ---------------- log Z: a ZMT x ZBT tile on a block past the GEMM's ----------------
__device__ __forceinline__ void split_bf16(float x, __nv_bfloat16& h, __nv_bfloat16& l) {
    h = __float2bfloat16_rn(x);
    l = __float2bfloat16_rn(x - __bfloat162float(h));
}
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int ZMT, int ZBT, int NTH>
constexpr int z_smem_bytes() {
    return 2 * ZMT * (GC + 8) * 2 + 2 * ZBT * (GC + 8) * 2 + (2 * ZBT + (NTH / 32) * ZBT) * 4 + GC * 8;
}
// Chunks of GC gates: their log phi row offsets (shared), sigma's columns (scalar loads, all in flight) split into bf16
// hi + lo, the gates' log phi (thread t < 256: column t % ZBT, KG gates, in registers); the running max mz per column,
// the accumulator rescaled where it grew; exp(log phi - mz) split into hi + lo (16-byte stores, k contiguous); the
// three MMAs -- 8 warps as 4 (ZMT / 4 nodes) x 2 (ZBT / 2 columns). Then log Z into log_z_out.
template <int NTH, int ZMT, int ZBT>
__device__ __forceinline__ void z_tile(char* smem, int zb, const float* __restrict__ sigma,
                                       const float* __restrict__ ext, const long* __restrict__ gate,
                                       float* __restrict__ log_z_out, int batch, int block_size, int n_gates,
                                       int n_child_gates, int gate_stride, long ext_base) {
    constexpr int LD = GC + 8, NW = NTH / 32, WTN = ZBT / 2, MT = ZMT / 64, NTN = WTN / 8;
    static_assert(NTH >= 256 && NTH % 32 == 0 && ZBT % 32 == 0 && (ZMT == 64 || ZMT == 128),
                  "8 MMA warps: 4 x ZMT / 4 nodes, 2 x ZBT / 2 columns");
    __nv_bfloat16* sAh = (__nv_bfloat16*)smem;               // [ZMT][LD]: sigma hi, k (gate) contiguous
    __nv_bfloat16* sAl = sAh + ZMT * LD;
    __nv_bfloat16* sBh = sAl + ZMT * LD;                     // [ZBT][LD]: phi' hi, k contiguous
    __nv_bfloat16* sBl = sBh + ZBT * LD;
    float* sMz = (float*)(sBl + ZBT * LD);                   // [ZBT] running max
    float* sScl = sMz + ZBT;                                 // [ZBT] its rescale this chunk
    float* sRed = sScl + ZBT;                                // [NW][ZBT]
    long* sRow = (long*)(sRed + NW * ZBT);                   // [GC]
    const int tid = threadIdx.x, nzb = batch / ZBT, nzm = block_size / ZMT;
    const int r = zb / (nzm * nzb), m0 = ((zb / nzb) % nzm) * ZMT, b0 = (zb % nzb) * ZBT;
    const long* gt = gate + (long)r * gate_stride;
    const int warp = tid / 32, lane = tid % 32, gq = lane >> 2, tq = lane & 3;
    const int wm = warp / 2, wn = warp % 2;                  // (warps 0 .. 7 multiply)
    float acc[MT][NTN][4];
    CUTE_UNROLL
    for (int i = 0; i < MT; i++) {
        CUTE_UNROLL
        for (int j = 0; j < NTN; j++) { acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f; }
    }
    if (tid < ZBT) sMz[tid] = -INFINITY;
    constexpr int NSI = (ZMT * GC + NTH - 1) / NTH;
    constexpr int NKG = 256 / ZBT, KG = GC / NKG;
    static_assert(KG % 8 == 0, "16-byte stores of 8 bf16 gates");
    const int pc = tid % ZBT, pk = (tid / ZBT) * KG;
    for (int g0 = 0; g0 < n_gates; g0 += GC) {
        const int gn = min(GC, n_gates - g0);
        __syncthreads();                                     // (the previous chunk read)
        for (int g = tid; g < GC; g += NTH) {               // each gate's log phi row offset (-1: no gate)
            const int j = (g0 + g) / n_child_gates;
            const long gb = (g < gn && j < gate_stride) ? gt[j] : -1;
            sRow[g] = gb >= 0 ? (gb + ext_base + (g0 + g) % n_child_gates) * (long)batch + b0 : -1;
        }
        float sv[NSI];
        CUTE_UNROLL
        for (int k = 0; k < NSI; k++) {
            const int i = tid + k * NTH, mm = i / GC, g = i % GC;
            sv[k] = (i < ZMT * GC && g < gn) ? sigma[((long)r * block_size + m0 + mm) * n_gates + g0 + g] : 0.f;
        }
        CUTE_UNROLL
        for (int k = 0; k < NSI; k++) {
            const int i = tid + k * NTH, mm = i / GC, g = i % GC;
            if (i < ZMT * GC) split_bf16(sv[k], sAh[mm * LD + g], sAl[mm * LD + g]);
        }
        __syncthreads();
        float pv[KG];
        if (tid < 256) {
            float pm = -INFINITY;
            CUTE_UNROLL
            for (int k = 0; k < KG; k++) {
                const long ro = sRow[pk + k];
                pv[k] = (ro >= 0) ? ext[ro + pc] : -INFINITY;
            }
            CUTE_UNROLL
            for (int k = 0; k < KG; k++) pm = fmaxf(pm, pv[k]);
            sRed[(tid / ZBT) * ZBT + pc] = pm;
        }
        __syncthreads();
        if (tid < ZBT) {
            float x = sRed[tid];
            for (int k = 1; k < NKG; k++) x = fmaxf(x, sRed[k * ZBT + tid]);
            const float mo = sMz[tid], mn = fmaxf(mo, x);
            sScl[tid] = (mo == -INFINITY) ? 1.f : __expf(mo - mn);
            sMz[tid] = mn;
        }
        __syncthreads();
        if (tid < 256) {
            const float m2 = sMz[pc];
            CUTE_UNROLL
            for (int k0 = 0; k0 < KG; k0 += 8) {
                __nv_bfloat16 h[8], l[8];
                CUTE_UNROLL
                for (int e = 0; e < 8; e++) {
                    const float x = pv[k0 + e];
                    split_bf16((x == -INFINITY || m2 == -INFINITY) ? 0.f : __expf(x - m2), h[e], l[e]);
                }
                *(uint4*)&sBh[pc * LD + pk + k0] = *(const uint4*)h;
                *(uint4*)&sBl[pc * LD + pk + k0] = *(const uint4*)l;
            }
        }
        __syncthreads();
        if (warp < 8) {
            if (g0 > 0) {                                    // the running max grew: rescale what was summed
                CUTE_UNROLL
                for (int j = 0; j < NTN; j++) {
                    const int n = wn * WTN + j * 8 + 2 * tq;
                    const float f0 = sScl[n], f1 = sScl[n + 1];
                    CUTE_UNROLL
                    for (int i = 0; i < MT; i++) {
                        acc[i][j][0] *= f0; acc[i][j][1] *= f1; acc[i][j][2] *= f0; acc[i][j][3] *= f1;
                    }
                }
            }
            CUTE_UNROLL
            for (int k0 = 0; k0 < GC; k0 += 16) {
                uint32_t ah[MT][4], al[MT][4];
                CUTE_UNROLL
                for (int i = 0; i < MT; i++) {
                    const int row = wm * (ZMT / 4) + i * 16 + gq;
                    const __nv_bfloat16* ph = sAh + row * LD + k0 + 2 * tq;
                    const __nv_bfloat16* pl = sAl + row * LD + k0 + 2 * tq;
                    ah[i][0] = *(const uint32_t*)ph; ah[i][1] = *(const uint32_t*)(ph + 8 * LD);
                    ah[i][2] = *(const uint32_t*)(ph + 8); ah[i][3] = *(const uint32_t*)(ph + 8 * LD + 8);
                    al[i][0] = *(const uint32_t*)pl; al[i][1] = *(const uint32_t*)(pl + 8 * LD);
                    al[i][2] = *(const uint32_t*)(pl + 8); al[i][3] = *(const uint32_t*)(pl + 8 * LD + 8);
                }
                CUTE_UNROLL
                for (int j = 0; j < NTN; j++) {
                    const int n = wn * WTN + j * 8 + gq;
                    const __nv_bfloat16* qh = sBh + n * LD + k0 + 2 * tq;
                    const __nv_bfloat16* ql = sBl + n * LD + k0 + 2 * tq;
                    const uint32_t bh[2] = {*(const uint32_t*)qh, *(const uint32_t*)(qh + 8)};
                    const uint32_t bl[2] = {*(const uint32_t*)ql, *(const uint32_t*)(ql + 8)};
                    CUTE_UNROLL
                    for (int i = 0; i < MT; i++) {
                        mma_bf16(acc[i][j], al[i], bh);
                        mma_bf16(acc[i][j], ah[i], bl);
                        mma_bf16(acc[i][j], ah[i], bh);
                    }
                }
            }
        }
    }
    if (warp < 8) {
        CUTE_UNROLL
        for (int j = 0; j < NTN; j++) {
            const int n = wn * WTN + j * 8 + 2 * tq;
            const float mz0 = sMz[n], mz1 = sMz[n + 1];
            CUTE_UNROLL
            for (int i = 0; i < MT; i++) {
                CUTE_UNROLL
                for (int h = 0; h < 2; h++) {
                    const int m = m0 + wm * (ZMT / 4) + i * 16 + gq + 8 * h;
                    const float z0 = acc[i][j][2 * h], z1 = acc[i][j][2 * h + 1];
                    const float l0 = (z0 <= 0.0f || mz0 == -INFINITY) ? -INFINITY : mz0 + __logf(z0);
                    const float l1 = (z1 <= 0.0f || mz1 == -INFINITY) ? -INFINITY : mz1 + __logf(z1);
                    *(float2*)&log_z_out[((long)r * block_size + m) * batch + b0 + n] = make_float2(l0, l1);
                }
            }
        }
    }
}

// ---------------- the GEMM launch: (batch tile, node tile, split) blocks, then the log Z tiles ----------------
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT, int ZMT, int ZBT>
__global__ void __launch_bounds__((WM * WN + PW) * 32) bs_stream_gemm_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ pids, const bfloat16_t* __restrict__ u,
        float* __restrict__ part, int knt, int mtiles, int block_size, int batch, int num_edges, int split,
        long rows_total, int n_gemm, const float* __restrict__ sigma, const float* __restrict__ ext,
        const long* __restrict__ gate, float* __restrict__ log_z_out, int n_gates, int n_child_gates, int gate_stride,
        long ext_base) {
    constexpr int NC = WM * WN * 32, NP = PW * 32;
    constexpr int ABYTES = KT * BM * 2, BBYTES = KT * BN * 2;
    constexpr int STAGE = (ABYTES + BBYTES + 1023) / 1024 * 1024;
    constexpr int SUB = BK / KT;                         // k-steps an edge tile is
    if ((int)blockIdx.x >= n_gemm) {
        z_tile<NC + NP, ZMT, ZBT>(smem_raw, blockIdx.x - n_gemm, sigma, ext, gate, log_z_out, batch, block_size,
                                  n_gates, n_child_gates, gate_stride, ext_base);
        return;
    }
    const int nbt = batch / BN, nmt = (int)(rows_total / BM);
    const int pid_b = blockIdx.x % nbt, pid_m = (blockIdx.x / nbt) % nmt, sp = blockIdx.x / (nbt * nmt);
    const int r = pid_m / mtiles, tile_id = pid_m % mtiles;
    const int ksteps = knt * SUB;
    const int k0 = (int)((long)sp * ksteps / split), k1 = (int)((long)(sp + 1) * ksteps / split);
    uint64_t* full = (uint64_t*)(smem_raw + ST * STAGE);
    uint64_t* empty = full + ST;
    const int tid = threadIdx.x;
    auto sAl = tile_to_shape(Atom64{}, make_shape(Int<BM>{}, Int<KT>{}));
    auto sBl = tile_to_shape(Atom64{}, make_shape(Int<BN>{}, Int<KT>{}));
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES); };
    if (tid == 0)
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], NP); mbar_init(&empty[s], NC); }
    __syncthreads();
    if (tid >= NC) {
        // ---------------- producers: k-step t into stage (t - k0) % ST once the consumers released it ----------------
        const int ptid = tid - NC;
        // a thread's 16-byte chunks are the same every k-step: their shared offsets (bytes, from a stage) and global
        // offsets (elements, from the k-step's bases), computed once; ~0u marks a chunk past the end
        constexpr int TA = ABYTES / 16, TB = BBYTES / 16, CA = (TA + NP - 1) / NP, CB = (TB + NP - 1) / NP;
        uint32_t oA[CA], oB[CB];
        int gA[CA], gB[CB];
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                oA[q] = i < TA ? (uint32_t)((char*)&sAt(mm, e) - smem_raw) : ~0u;
                gA[q] = e * block_size + mm;
            }
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) {
                int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - smem_raw) : ~0u;
                gB[q] = e * batch + bb;
            }
        }
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        const long* pr = pids + (long)r * knt;
        for (int t = k0; t < k1; t++) {
            const int j = t - k0, s = j % ST;
            if (j >= ST) mbar_wait(&empty[s], ((j / ST) - 1) & 1);
            const uint32_t ss = sbase + s * STAGE;
            const int t64 = t / SUB, sub = t % SUB;
            const bfloat16_t* pa = params + pr[t64] + (long)sub * KT * block_size + (long)tile_id * BM;
            const bfloat16_t* pb = u + ((long)r * num_edges + (long)t * KT) * batch + (long)pid_b * BN;
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) if (oA[q] != ~0u) cpa16(ss + oA[q], pa + gA[q]);
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) if (oB[q] != ~0u) cpa16(ss + oB[q], pb + gB[q]);
            cpa_mbar_arrive(&full[s]);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }
    // ---------------- consumers: the range's k-steps in order into one fp32 accumulator ----------------
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    auto tA = s2rA.get_thread_slice(tid);
    auto tB = s2rB.get_thread_slice(tid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);
    constexpr int KB = size<2>(tCrA);
    for (int t = k0; t < k1; t++) {
        const int j = t - k0, s = j % ST;
        mbar_wait(&full[s], (j / ST) & 1);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        auto tXsA = tA.partition_S(sAt);
        auto tXsB = tB.partition_S(sBt);
        copy(s2rA, tXsA(_, _, 0), tXrA(_, _, 0));
        copy(s2rB, tXsB(_, _, 0), tXrB(_, _, 0));
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {                // block kb + 1's fragments load while block kb multiplies
            if (kb + 1 < KB) {
                copy(s2rA, tXsA(_, _, kb + 1), tXrA(_, _, kb + 1));
                copy(s2rB, tXsB(_, _, kb + 1), tXrB(_, _, kb + 1));
            } else {
                mbar_arrive(&empty[s]);                  // the stage is read
            }
            cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
        }
    }
    float* o = part + ((long)sp * rows_total + (long)r * block_size + (long)tile_id * BM) * batch + pid_b * BN;
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i += 2) {            // (i, i + 1): columns b, b + 1 of one row
        *(float2*)&o[(long)get<0>(tCcC(i)) * batch + get<1>(tCcC(i))] = make_float2(tCrL(i), tCrL(i + 1));
    }
}

// ---------------- merge: a thread per 4 columns of a node ----------------
template <int S>
__global__ void __launch_bounds__(256) bs_stream_merge_kernel(
        float* __restrict__ node_mars, const float* __restrict__ log_z, const float* __restrict__ part,
        const float* __restrict__ mp, const long* __restrict__ nids, int batch, int block_size, long rows_total) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    const int bq = batch / 4;
    if (idx >= rows_total * bq) return;
    const long n = idx / bq;
    const int b = (int)(idx % bq) * 4;
    const int r = (int)(n / block_size);
    float4 lp[S], xm[S];
    CUTE_UNROLL
    for (int p = 0; p < S; p++) {                        // (every load issued before the arithmetic)
        lp[p] = *(const float4*)&part[((long)p * rows_total + n) * batch + b];
        xm[p] = *(const float4*)&mp[((long)r * S + p) * batch + b];
    }
    const float4 lz = *(const float4*)&log_z[n * batch + b];
    const long nid = nids[r] + n % block_size;
    float out[4];
    CUTE_UNROLL
    for (int c = 0; c < 4; c++) {
        float M = -INFINITY;
        CUTE_UNROLL
        for (int p = 0; p < S; p++) M = fmaxf(M, (&xm[p].x)[c]);
        float L = 0.f;
        CUTE_UNROLL
        for (int p = 0; p < S; p++) {
            const float x = (&xm[p].x)[c];
            L = fmaf((&lp[p].x)[c], x == -INFINITY ? 0.f : __expf(x - M), L);
        }
        const float ln = (M == -INFINITY) ? -INFINITY : (logf(L) + M), z = (&lz.x)[c];
        out[c] = (ln == -INFINITY || z == -INFINITY) ? -INFINITY : (ln - z);
    }
    *(float4*)&node_mars[nid * batch + b] = make_float4(out[0], out[1], out[2], out[3]);
}

// ---------------- host ----------------
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT, int ZMT, int ZBT>
static void launch_gemm(const bfloat16_t* params, const long* pids, const bfloat16_t* u, float* part, int knt, int rows,
                        int block_size, int batch, int num_edges, int split, const float* sigma, const float* ext,
                        const long* gate, float* log_z, int n_gates, int ncg, int gate_stride, long ext_base,
                        cudaStream_t st) {
    constexpr int STAGE = (KT * BM * 2 + KT * BN * 2 + 1023) / 1024 * 1024;
    constexpr int SMEM_G = ST * STAGE + 2 * ST * 8;
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr int SMEM_Z = z_smem_bytes<ZMT, ZBT, NTH>();
    constexpr int SMEM = SMEM_G > SMEM_Z ? SMEM_G : SMEM_Z;
    static_assert(SMEM <= 101376, "a config past sm_120's 99 KB of shared memory a block");
    auto kernel = bs_stream_gemm_kernel<BM, BN, WM, WN, PW, ST, KT, ZMT, ZBT>;
    // the shared-memory attribute once per device and instantiation (the call costs host time on every launch)
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if (dev < 64) attr_set[dev] = true;
    }
    const int mtiles = block_size / BM;
    const long n_gemm = (long)(batch / BN) * rows * mtiles * split;
    const long n_z = (long)rows * (block_size / ZMT) * (batch / ZBT);
    TORCH_CHECK(n_gemm + n_z < (1L << 31), "blockscale forward stream: a grid past 2^31 blocks");
    kernel<<<(unsigned)(n_gemm + n_z), NTH, SMEM, st>>>(params, pids, u, part, knt, mtiles, block_size, batch,
                                                        num_edges, split, (long)rows * block_size, (int)n_gemm, sigma,
                                                        ext, gate, log_z, n_gates, ncg, gate_stride, ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
static void launch_gemm_z(const bfloat16_t* params, const long* pids, const bfloat16_t* u, float* part, int knt,
                          int rows, int block_size, int batch, int num_edges, int split, const float* sigma,
                          const float* ext, const long* gate, float* log_z, int n_gates, int ncg, int gate_stride,
                          long ext_base, cudaStream_t st) {
    // log Z's tile: 128 nodes where the node block divides, else 64; 128 columns where the batch divides, else 64
#define L_(ZMT, ZBT) launch_gemm<BM, BN, WM, WN, PW, ST, KT, ZMT, ZBT>(params, pids, u, part, knt, rows, block_size, \
                         batch, num_edges, split, sigma, ext, gate, log_z, n_gates, ncg, gate_stride, ext_base, st)
    if (block_size % 128 == 0) {
        if (batch % 128 == 0) L_(128, 128); else L_(128, 64);
    } else {
        if (batch % 128 == 0) L_(64, 128); else L_(64, 64);
    }
#undef L_
}

// id -> {BM, BN, WM, WN, PW, ST, KT}. MEASURED (the sweep above): the 128 x 128 tiles take B >= 128 at most shapes
// (64-row k-steps in 3 stages, or 32-row ones in 6), 128 x 64 B 64 and the 512-node blocks, 64 x 128 the 256-node
// blocks at B 512. 64 x 64 serves 64-node blocks at a batch that is no multiple of 128.
#define CFG_TABLE(X)                     \
    X(0, 128, 128, 4, 2, 4, 3, 64)       \
    X(1, 128, 128, 2, 4, 4, 6, 32)       \
    X(2, 128, 64, 4, 2, 4, 4, 64)        \
    X(3, 64, 128, 2, 4, 4, 4, 64)        \
    X(4, 128, 64, 4, 2, 4, 8, 32)        \
    X(5, 64, 64, 2, 2, 4, 6, 64)

std::vector<std::vector<int64_t>> configs() {
    std::vector<std::vector<int64_t>> out;
#define X(id, bm, bn, wm, wn, pw, st, kt) out.push_back({bm, bn, kt});
    CFG_TABLE(X)
#undef X
    return out;
}

// The config ids that serve this layer: its tile divides the node block and the batch, and the layer is in scope.
std::vector<int64_t> fitting_configs(int64_t block_size, int64_t batch, int64_t node_cbs, int64_t gate_cbs) {
    std::vector<int64_t> out;
    if (batch % 64 != 0 || block_size % 64 != 0 || node_cbs < BK || gate_cbs < 4 ||
        (node_cbs & (node_cbs - 1)) != 0 || (gate_cbs & (gate_cbs - 1)) != 0 || node_cbs % gate_cbs != 0)
        return out;
    const auto cfgs = configs();
    for (int i = 0; i < (int)cfgs.size(); i++)
        if (block_size % cfgs[i][0] == 0 && batch % cfgs[i][1] == 0) out.push_back(i);
    return out;
}

// The pre-pass copy's arguments (`blockscale_forward_prepass`), `cfg` = config id * 16 + split, and sigma
// [rows * block_size, n_gates] (fp32, `_sigma`).
void blockscale_forward_stream(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                               torch::Tensor ext, torch::Tensor nids, torch::Tensor cids, torch::Tensor pids,
                               torch::Tensor gate, torch::Tensor log_z, int64_t block_size, int64_t num_edges,
                               int64_t node_cbs, int64_t gate_cbs, int64_t n_node_gates, int64_t ext_base, int64_t cfg,
                               torch::Tensor sigma) {
    const int batch = node_mars.size(1);
    const int knt = (int)(num_edges / BK), rows = (int)nids.size(0);
    const int cfg_id = (int)(cfg / 16), split = (int)(cfg % 16);
    const auto cfgs = configs();
    TORCH_CHECK(cfg_id >= 0 && cfg_id < (int)cfgs.size(), "blockscale forward stream: no config ", cfg_id);
    const int BMc = (int)cfgs[cfg_id][0], BNc = (int)cfgs[cfg_id][1], kt = (int)cfgs[cfg_id][2];
    const int ksteps = (int)(num_edges / kt), kt_sh = (int)std::log2((double)kt);
    TORCH_CHECK(n_node_gates == 1, "blockscale forward stream: one gate per node block; got ", n_node_gates);
    TORCH_CHECK(batch % 64 == 0 && num_edges % BK == 0 && block_size % 64 == 0 && block_size % BMc == 0 &&
                batch % BNc == 0, "blockscale forward stream: batch % 64, num_edges % 64, block_size % 64 and the "
                "config's tile dividing the node block and the batch");
    TORCH_CHECK(node_cbs >= BK && gate_cbs >= 4 && (node_cbs & (node_cbs - 1)) == 0 &&
                (gate_cbs & (gate_cbs - 1)) == 0 && node_cbs % gate_cbs == 0,
                "blockscale forward stream: node_cbs >= 64, a gate width >= 4, powers of two, the gate's dividing");
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= ksteps, "blockscale forward stream: split 1 .. ",
                MAX_SPLIT, " and at most the k-steps (", ksteps, "); got ", split);
    TORCH_CHECK(element_mars.is_contiguous() && node_mars.is_contiguous() && ext.is_contiguous() &&
                gate.is_contiguous() && nids.is_contiguous() && pids.is_contiguous() && cids.is_contiguous() &&
                sigma.is_contiguous() && log_z.is_contiguous(), "blockscale forward stream: contiguous tensors");
    TORCH_CHECK(cids.numel() == nids.numel() * knt && pids.numel() == cids.numel(),
                "blockscale forward stream: cids / pids one entry per (node block, edge tile)");
    const int n_gates = (int)(num_edges / gate_cbs), ncg = (int)(node_cbs / gate_cbs);
    TORCH_CHECK(sigma.scalar_type() == torch::kFloat && sigma.numel() == (long)rows * block_size * n_gates,
                "blockscale forward stream: sigma fp32 [rows * block_size, n_gates]");
    TORCH_CHECK(log_z.numel() == (long)rows * block_size * batch,
                "blockscale forward stream: log_z [rows * block_size, B]");
    const at::cuda::OptionalCUDAGuard guard(node_mars.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    const int node_sh = (int)std::log2((double)node_cbs), gate_sh = (int)std::log2((double)gate_cbs);
    // scratch, ONE allocation: u bf16 [rows, num_edges, B], Mp [rows, split, B], the partials [split, rows * bs, B]
    // (every region 16-byte aligned: B % 64 == 0)
    const long o_mp = (long)rows * num_edges * batch * 2;
    const long o_part = o_mp + (long)rows * split * batch * 4;
    const long total = o_part + (long)split * rows * block_size * batch * 4;
    torch::Tensor scratch = torch::empty({total}, element_mars.options().dtype(torch::kUInt8));
    char* base = reinterpret_cast<char*>(scratch.data_ptr());
    auto* u = reinterpret_cast<__nv_bfloat16*>(base);
    auto* mp = reinterpret_cast<float*>(base + o_mp);
    auto* part = reinterpret_cast<float*>(base + o_part);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    {
        constexpr int NT = 512, PB = 16;
        const int gc = (int)(gate_cbs < CH ? gate_cbs : CH);
        bs_stream_prepass_kernel<NT, PB><<<rows * split * (batch / PB), NT, 0, st>>>(
            element_mars.data_ptr<float>(), cids.data_ptr<long>(), ext.data_ptr<float>(), gate.data_ptr<long>(), u, mp,
            batch, (int)num_edges, knt, ksteps, kt_sh, split, (int)gate.size(1), node_sh, gate_sh, (int)node_cbs,
            (long)ext_base, gc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    switch (cfg_id) {
#define X(id, bm, bn, wm, wn, pw, stg, ktt) \
        case id: launch_gemm_z<bm, bn, wm, wn, pw, stg, ktt>(pp, pids.data_ptr<long>(), (const bfloat16_t*)u, part, \
                     knt, rows, (int)block_size, batch, (int)num_edges, split, sigma.data_ptr<float>(), \
                     ext.data_ptr<float>(), gate.data_ptr<long>(), log_z.data_ptr<float>(), n_gates, ncg, \
                     (int)gate.size(1), (long)ext_base, st); break;
        CFG_TABLE(X)
#undef X
    }
    {
        const long rt = (long)rows * block_size, n = rt * (batch / 4);
        const unsigned grid = (unsigned)((n + 255) / 256);
        float* nm = node_mars.data_ptr<float>();
        const float* lz = log_z.data_ptr<float>();
        const long* np = nids.data_ptr<long>();
        switch (split) {
#define M_(S) case S: bs_stream_merge_kernel<S><<<grid, 256, 0, st>>>(nm, lz, part, mp, np, batch, (int)block_size, \
                                                                      rt); break;
            M_(1) M_(2) M_(3) M_(4) M_(5) M_(6) M_(7) M_(8)
#undef M_
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_forward_stream", &blockscale_forward_stream,
          "the block-scale forward as a pre-pass, a split GEMM (log Z beside it) and a merge (log N - log Z)");
    m.def("configs", &configs, "(BM, BN, KT) per config id");
    m.def("fitting_configs", &fitting_configs,
          "the config ids that serve a layer (block_size, batch, node_cbs, gate_cbs)");
}
