// The "sum_exp" form's forward over the VIRTUAL BATCH, the virtual child values never stored.
//
// `lowrank/virtual.py` runs the plain sum-layer kernels on a virtual batch whose columns are the (sample, rank) pairs,
// `v = b K + r`, of virtual child values `vem[child, v] = element_mars[child, b] + B_b[r, child]` (the mixture scale's
// `exp(B)`, applied to the children), the node values `vnm[node, v] = log sum_j theta[node, j] exp(vem[j, v])` then
// collapsed over the ranks with `A`. The plain path materializes `vem` (K times the circuit's element buffer: written
// by `expand`, read by the stream kernel's pre-pass). Here, two forms take the virtual batch directly:
//
// THE PRE-PASS CHAIN (`lowrank_forward_stream`; the stream kernel of layer/kernels/c/tlmm_forward_sum_stream.cu: a
// pre-pass per split range, one split GEMM, a merge):
//   pre-pass  two launches. (1) The range's largest element value a sample (a float atomic max): with
//             the rank's largest B (the normalizers' `shift`) it bounds a column's largest virtual value
//             `c + B[r, j, b]` -- the column's shift, at most B's spread over the children above the true maximum: the
//             terms lie within bf16's range below 1 unless that spread passes ~80 nats. (2) A thread per (child row,
//             sample) forms its row's K virtual values -- `c` the element value, or the product's two children's node
//             values summed (FUSED, as the plain kernels' fused pre-pass forms them: `cbase`) -- and stores their
//             exponentials `u[group, child, b K .. b K + K)` in one run; a warp is PB consecutive samples of a row:
//             every load one line of B, every store a contiguous run. `jpos[group, edge]` names each row's child
//             `j` of the node vector (-1: a dummy row, 0).
//   GEMM      the stream kernel's, as it stands (`batch` = B K columns): `vnm[node, v]` (split 1) or the partials;
//   merge     the stream kernel's.
// The values are the plain path's class (the same fp32 sums and GEMM; the bf16 exponentials under another shift), in a
// fixed order: the same bits from run to run. Scope (`fitting_configs`, and `virtual.py`'s conditions): bf16
// parameters with 8-aligned runs, K <= 16, B K % 64 == 0, B % 16 == 0, block_size % 32 == 0, one node vector per launch
// (its `B` factors and `jpos`), split 1 .. 12.
//
// THE ONLINE-MAX FORM (`lowrank_forward_om`, `lr_om_kernel` below): one launch, the operand formed in the GEMM's own
// stages from the normalizers' V and the children's values, each sample's shift its running max (its header). Scope
// (`om_serves`): rank 16, B % 4 == 0, block_size % 32 == 0; the normalizers' V bf16.
//
// The collapse (`lowrank_mixture.cu`) follows either. MEASURED on the way (HMM 1024 x 1024, rank 16, B 64, us a
// layer): the exponentials with an exact column max -- a block per (group, split, 8 samples) walking the range 78;
// blocks of 64 rows x 8 samples 16.5 (two reads of B) --; the operand formed by the GEMM's producer warps in place of a
// pre-pass 88 with synchronous loads (they stall on the loads' latency) and 67 with the raw inputs cp.async'd two
// k-steps ahead into a ring of raw slots and formed in the stage (one launch a layer; B 16: 27, B 256: 191 -- against
// this chain's 18 / 27 / 69): the round trip of a stage's ~1100 16-byte requests is ~4 us under load, and 99 KB of
// shared memory hold two k-steps of raw inputs beside the stages, where the pre-pass's u is read seven stages ahead.
// (The online-max form forms the operand from V instead -- one 16-byte run of a precomputed V row a chunk, scaled in
// place by converter warps -- and keeps the producers' loops free of shared-memory loads.)

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"
#include <cuda_bf16.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64           // edge tile: one run of children (ebase / pbase one entry each)
#define CH 32           // the fused pre-pass's chunk: half an edge tile (cbase's runs)
#define MAX_SPLIT 12
#define KMAX 16         // the largest rank served

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
__device__ __forceinline__ void cpa_mbar_arrive(uint64_t* bar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ int ldv_s32(const int* p) {
    int v;
    asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ long ldv_s64(const long* p) {
    long v;
    asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{\n .reg .pred p;\n W_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra W_%=;\n}\n" ::"r"(a), "r"(phase) : "memory");
}
// ---------------- pre-pass: two launches, a thread per (row, sample), all K ranks of its row ----------------------
// A unit is ROWS consecutive rows of a range and PB consecutive samples, a thread each: a warp is 32 / PB rows x PB
// samples (PB 32: one row's 32 samples -- a load of B or of the element values one 128-byte line, the row's stores
// 1 KB in a run). Small blocks (4 rows x 32 samples): every SM takes the cold factors' loads (MEASURED, HMM 1024 x 1024,
// rank 16, B 64: blocks of 16 rows left 60 of 188 SMs idle). (1) `lr_cmax_kernel`: the range's largest element value a
// sample, `cmax[g, sp, b]` -- a float atomic max over the units into -inf, which the GEMM sets back once it has
// passed its wait for this chain (no memset of its own on the stream: an operation of its own costs a few us whatever
// it does). (2) `lr_exp_kernel`: u = bf16 exp(c + B[r, j, b] - M), M the column's shift (cmax + the rank's largest B),
// 0 where the row has no child, its value is -inf, or M is; the range's first row's threads write the shifts to `mp`
// for the GEMM's epilogue / the merge. `jpos` [groups, edges] names each row's child (-1: none). The values after each
// kernel's wait for the launch before, through pointers that are not __restrict__ (tlmm_forward_sum_stream.cu's note
// on dependent launches). (A cooperative launch doing both, a grid-wide barrier between, MEASURED: ~9 us of its own.)
__device__ __forceinline__ void atomic_max_float(float* addr, float v) {
    // the usual trick: non-negative floats order as ints, negative ones in reverse as unsigned
    if (v >= 0.f) atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
    else atomicMin(reinterpret_cast<unsigned*>(addr), __float_as_uint(v));
}

struct UnitGeo {
    int g, sp, b, pos, pos0, pos1;                           // a unit's group, range, this thread's sample and row
};

template <int ROWS, int PB>
__device__ __forceinline__ UnitGeo unit_geo(int unit, int B, int ksteps, int kt_sh, int split, int nchunk) {
    UnitGeo o;
    const int tid = threadIdx.x, nbt = B / PB;
    const int chunk = unit % nchunk, bt = (unit / nchunk) % nbt;
    o.sp = (unit / (nchunk * nbt)) % split;
    o.g = unit / (nchunk * nbt * split);
    o.b = bt * PB + tid % PB;
    o.pos0 = (int)(((long)o.sp * ksteps / split) << kt_sh);
    o.pos1 = (int)(((long)(o.sp + 1) * ksteps / split) << kt_sh);
    o.pos = o.pos0 + chunk * ROWS + tid / PB;
    return o;
}

// the element value of a row of a child (FUSED: the product's two children's node values summed; -inf for a dummy)
template <bool FUSED>
__device__ __forceinline__ float elem_value(const float* vals, const long* ebase, const int* cbase, const int* grep,
                                            int knt, int B, int g, int pos, int b) {
    if (FUSED) {
        const int* cq = cbase + (long)grep[g] * knt * 4 + (pos / BK) * 4 + ((pos % BK) / CH) * 2;
        const int c0 = cq[0];
        if (c0 < 0) return -INFINITY;
        return vals[(long)(c0 + pos % CH) * B + b] + vals[(long)(cq[1] + pos % CH) * B + b];
    }
    return vals[(ebase[(long)grep[g] * knt + pos / BK] + pos % BK) * B + b];
}

template <int ROWS, int PB, bool FUSED>
__global__ void __launch_bounds__(ROWS * PB) lr_cmax_kernel(
        const float* vals, const long* __restrict__ ebase, const int* __restrict__ cbase, const int* __restrict__ grep,
        const int* __restrict__ jpos, int B, float* cmax, int num_edges, int knt, int ksteps, int kt_sh, int split,
        int nchunk) {
    constexpr int NT = ROWS * PB, NW = NT / 32;
    __shared__ float smx[NW][PB];
    const UnitGeo o = unit_geo<ROWS, PB>(blockIdx.x, B, ksteps, kt_sh, split, nchunk);
    const int j = o.pos < o.pos1 ? jpos[(long)o.g * num_edges + o.pos] : -1;   // (a table: before the wait)
    cudaGridDependencySynchronize();                         // (`vals` and `cmax`' reset are the launches before's)
    cudaTriggerProgrammaticLaunchCompletion();               // the exponentials may launch
    float m = j >= 0 ? elem_value<FUSED>(vals, ebase, cbase, grep, knt, B, o.g, o.pos, o.b) : -INFINITY;
    CUTE_UNROLL
    for (int off = 16; off >= PB; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));   // (a warp's rows)
    const int tid = threadIdx.x;
    if (NW == 1) {
        if (tid < PB && m > -INFINITY) atomic_max_float(cmax + ((long)o.g * split + o.sp) * B + o.b, m);
        return;
    }
    if (tid % 32 < PB) smx[tid / 32][tid % PB] = m;
    __syncthreads();
    if (tid < PB) {
        float x = smx[0][tid];
        CUTE_UNROLL
        for (int w = 1; w < NW; w++) x = fmaxf(x, smx[w][tid]);
        if (x > -INFINITY) atomic_max_float(cmax + ((long)o.g * split + o.sp) * B + o.b, x);
    }
}

template <int ROWS, int PB, bool FUSED>
__global__ void __launch_bounds__(ROWS * PB) lr_exp_kernel(
        const float* vals, const long* __restrict__ ebase, const int* __restrict__ cbase, const int* __restrict__ grep,
        const int* __restrict__ jpos, const float* __restrict__ fb, int N, int B, int K,
        __nv_bfloat16* __restrict__ u, const float* cmax, const float* __restrict__ shift, long row0, float* mp, int cols,
        int num_edges, int knt, int ksteps, int kt_sh, int split, int nchunk) {
    const UnitGeo o = unit_geo<ROWS, PB>(blockIdx.x, B, ksteps, kt_sh, split, nchunk);
    const long sB = (long)N * B;
    // what no launch of the chain writes, before the wait: the row's child, the rank's largest B (the factors are the
    // pass's, staged before it)
    const int j = o.pos < o.pos1 ? jpos[(long)o.g * num_edges + o.pos] : -1;
    float sh[KMAX];
    {
        const float* shp = shift + row0 + (long)o.b * K;     // (copy, sample, rank) rows: the sample's ranks in a run
        CUTE_UNROLL
        for (int r = 0; r < KMAX; r++) sh[r] = r < K ? shp[r] : 0.f;
    }
    float w[KMAX];
    {
        const float* src = fb + (long)(j >= 0 ? j : 0) * B + o.b;
        CUTE_UNROLL
        for (int r = 0; r < KMAX; r++) w[r] = r < K && j >= 0 ? src[r * sB] : 0.f;
    }
    cudaGridDependencySynchronize();                         // (the maxima and `vals` are the launches before's)
    const float cm = cmax[((long)o.g * split + o.sp) * B + o.b];
    float m[KMAX];
    CUTE_UNROLL
    for (int r = 0; r < KMAX; r++) m[r] = r < K && cm > -INFINITY ? cm + sh[r] : -INFINITY;
    if (o.pos == o.pos0) {
        float* out = mp + ((long)o.g * split + o.sp) * cols + (long)o.b * K;
        for (int r = 0; r < K; r++) out[r] = m[r];
    }
    if (o.pos < o.pos1) {
        const float c = j >= 0 ? elem_value<FUSED>(vals, ebase, cbase, grep, knt, B, o.g, o.pos, o.b) : -INFINITY;
        const bool live = j >= 0 && c > -INFINITY;
        unsigned h[KMAX / 2];
        CUTE_UNROLL
        for (int r = 0; r < KMAX; r += 2) {
            const __nv_bfloat16 lo = __float2bfloat16_rn(live && m[r] > -INFINITY ? __expf(c + w[r] - m[r]) : 0.f);
            const __nv_bfloat16 hi = __float2bfloat16_rn(live && m[r + 1] > -INFINITY ? __expf(c + w[r + 1] - m[r + 1]) : 0.f);
            h[r / 2] = (unsigned)__bfloat16_as_ushort(lo) | ((unsigned)__bfloat16_as_ushort(hi) << 16);
        }
        __nv_bfloat16* out = u + ((long)o.g * num_edges + o.pos) * cols + (long)o.b * K;
        if (K % 8 == 0) {
            CUTE_UNROLL
            for (int q = 0; q < KMAX / 8; q++)
                if (q * 8 < K) *reinterpret_cast<uint4*>(out + q * 8) = make_uint4(h[4 * q], h[4 * q + 1], h[4 * q + 2], h[4 * q + 3]);
        } else if (K == 4) {
            *reinterpret_cast<uint2*>(out) = make_uint2(h[0], h[1]);
        } else if (K == 2) {
            *reinterpret_cast<unsigned*>(out) = h[0];
        } else {                                             // (K 1, and any other: element stores)
            for (int r = 0; r < K; r++)
                out[r] = __ushort_as_bfloat16((unsigned short)(r & 1 ? h[r / 2] >> 16 : h[r / 2] & 0xffffu));
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();               // the GEMM may launch (its consumers spin from then on)
}

extern __shared__ __align__(1024) char smem_raw[];

// The MN-major smem atom of an operand with 64 rows (m or b) a k: 128-byte lines, swizzled over the 16-byte chunks so
// the transposed ldmatrix's 8 k-rows of one chunk fall in distinct banks.
using Atom64 = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
using AtomA32 = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));   // (TMA's A)
// K-major A of 32-wide k-steps (the backward's theta^T: a child's 32 parents one swizzled 64-byte line)
using AtomAK32 = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
__device__ __forceinline__ void tma_load_3d(uint32_t smem, const CUtensorMap* desc, int c0, int c1, int c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 ::"r"(smem), "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}

// ---------------- the GEMM: (batch tile, node tile, split) blocks ----------------
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
__global__ void __launch_bounds__((WM * WN + PW) * 32) lr_gemm_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ pbase, const bfloat16_t* __restrict__ u,
        const int* __restrict__ ugrp, float* __restrict__ part, int knt, int mtiles, int block_size, int batch,
        int num_edges, int split, long rows_total, float* __restrict__ node_mars, const float* mp,
        const long* __restrict__ nids, float* cmax, int ncm) {
    constexpr int NC = WM * WN * 32, NP = PW * 32;
    constexpr int ABYTES = KT * BM * 2, BBYTES = KT * BN * 2;
    constexpr int STAGE = (ABYTES + BBYTES + 1023) / 1024 * 1024;
    constexpr int SUB = BK / KT;                         // k-steps an edge tile is
    // 16 consumer warps (the 256-row tile, 640 threads) run under a 96-register cap: the tile's columns in NH halves of
    // BNH, each half's fragments loaded as they are multiplied; the 8-warp tiles (NH 1) keep their fragments a k-block
    // ahead, their code as it was
    constexpr int NH = (WM * WN >= 16) ? 2 : 1, BNH = BN / NH, HBYTES = BBYTES / NH;
    const int nbt = batch / BN, nmt = (int)(rows_total / BM);
    const int pid_b = blockIdx.x % nbt, pid_m = (blockIdx.x / nbt) % nmt, sp = blockIdx.x / (nbt * nmt);
    // the node tile: one of `mtiles` of a node block, or -- a block smaller than the tile (an HCLT's 32-node blocks)
    // -- `bpt` consecutive blocks of one group, their parameters a slab each
    const int bpt = BM > block_size ? BM / block_size : 1;
    const int r = bpt > 1 ? pid_m * bpt : pid_m / mtiles, tile_id = bpt > 1 ? 0 : pid_m % mtiles;
    const int ksteps = knt * SUB;
    const int k0 = (int)((long)sp * ksteps / split), k1 = (int)((long)(sp + 1) * ksteps / split);
    uint64_t* full = (uint64_t*)(smem_raw + ST * STAGE);
    uint64_t* empty = full + ST;
    const int tid = threadIdx.x;
    auto sAl = tile_to_shape(Atom64{}, make_shape(Int<BM>{}, Int<KT>{}));
    auto sBl = tile_to_shape(Atom64{}, make_shape(Int<BNH>{}, Int<KT>{}));      // (a half's u tile)
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s, int h) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES + h * HBYTES); };
    if (tid == 0)
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], NP); mbar_init(&empty[s], NC); }
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();           // the merge may launch
    if (tid >= NC) {
        // ---------------- producers: k-step t into stage (t - k0) % ST once the consumers released it ----------------
        const int ptid = tid - NC;
        // a thread's 16-byte chunks are the same every k-step: their shared offsets (bytes, from a stage) and global
        // offsets (elements, from the k-step's bases), computed once; ~0u marks a chunk past the end
        constexpr int TA = ABYTES / 16, TB = BBYTES / 16, CA = (TA + NP - 1) / NP, CB = (TB + NP - 1) / NP;
        uint32_t oA[CA], oB[CB];
        int gA[CA], gB[CB];
        const long* prA[CA];                             // a chunk's node block's parameter table (its slab of the tile)
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(0, 0)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                oA[q] = i < TA ? (uint32_t)((char*)&sAt(mm, e) - smem_raw) : ~0u;
                const int bq = bpt > 1 ? mm / block_size : 0;
                gA[q] = e * block_size + (bpt > 1 ? mm - bq * block_size : mm) + tile_id * BM;
                prA[q] = pbase + (long)(r + bq) * knt;
            }
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) {
                int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8, h = bb / BNH;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb % BNH, e) - smem_raw) + h * HBYTES : ~0u;
                gB[q] = e * batch + bb;
            }
        }
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        const bfloat16_t* ug = u + ((long)ugrp[r] * num_edges) * batch + (long)pid_b * BN;
        // the first stages' parameter tiles while the pre-pass may still run (no launch of the chain writes them)
        const int t_pre = min(k1, k0 + ST);
        for (int t = k0; t < t_pre; t++) {
            const long psub = (long)(t % SUB) * KT * block_size;
            CUTE_UNROLL
            for (int q = 0; q < CA; q++)
                if (oA[q] != ~0u) cpa16(sbase + (t - k0) * STAGE + oA[q], params + prA[q][t / SUB] + psub + gA[q]);
        }
        cudaGridDependencySynchronize();                 // u (the pre-pass's) from here on
        if (blockIdx.x == 0)                             // the maxima back to -inf: every reader of them is done
            for (int i = ptid; i < ncm; i += NP) cmax[i] = -INFINITY;
        for (int t = k0; t < k1; t++) {
            const int j = t - k0, s = j % ST;
            if (j >= ST) mbar_wait(&empty[s], ((j / ST) - 1) & 1);
            const uint32_t ss = sbase + s * STAGE;
            const int t64 = t / SUB, sub = t % SUB;
            const long psub = (long)sub * KT * block_size;
            const bfloat16_t* pb = ug + (long)t * KT * batch;
            if (t >= t_pre) {
                CUTE_UNROLL
                for (int q = 0; q < CA; q++) if (oA[q] != ~0u) cpa16(ss + oA[q], params + prA[q][t64] + psub + gA[q]);
            }
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) if (oB[q] != ~0u) cpa16(ss + oB[q], pb + gB[q]);
            cpa_mbar_arrive(&full[s]);                   // (tracks every copy this thread issued: the stage's A too)
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }
    // ---------------- consumers: the range's k-steps in order into one fp32 accumulator ----------------
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0, 0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    auto tA = s2rA.get_thread_slice(tid);
    auto tB = s2rB.get_thread_slice(tid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BNH>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    Tensor tCrL1 = thr.partition_fragment_C(cC);         // (the second half's accumulator; unused where NH is 1)
    clear(tCrL);
    clear(tCrL1);
    constexpr int KB = size<2>(tCrA);
    for (int t = k0; t < k1; t++) {
        const int j = t - k0, s = j % ST;
        mbar_wait(&full[s], (j / ST) & 1);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s, 0)), sBl);
        auto tXsA = tA.partition_S(sAt);
        auto tXsB = tB.partition_S(sBt);
        if constexpr (NH == 1) {
            copy(s2rA, tXsA(_, _, 0), tXrA(_, _, 0));
            copy(s2rB, tXsB(_, _, 0), tXrB(_, _, 0));
            CUTE_UNROLL
            for (int kb = 0; kb < KB; kb++) {            // block kb + 1's fragments load while block kb multiplies
                if (kb + 1 < KB) {
                    copy(s2rA, tXsA(_, _, kb + 1), tXrA(_, _, kb + 1));
                    copy(s2rB, tXsB(_, _, kb + 1), tXrB(_, _, kb + 1));
                } else {
                    mbar_arrive(&empty[s]);              // the stage is read
                }
                cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
            }
        } else {
            auto tXsB1 = tB.partition_S(make_tensor(make_smem_ptr(stB(s, 1)), sBl));
            CUTE_UNROLL
            for (int kb = 0; kb < KB; kb++) {            // a k-block: A once, each half's u as it is multiplied
                copy(s2rA, tXsA(_, _, kb), tXrA(_, _, kb));
                copy(s2rB, tXsB(_, _, kb), tXrB(_, _, kb));
                cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
                copy(s2rB, tXsB1(_, _, kb), tXrB(_, _, kb));
                if (kb + 1 == KB) mbar_arrive(&empty[s]);   // the stage is read
                cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL1);
            }
        }
    }
    cudaGridDependencySynchronize();                     // (the producers waited; the epilogue's own guarantee)
    if (split > 1) {                                     // the range's partial, merged by the merge kernel
        CUTE_UNROLL
        for (int h = 0; h < NH; h++) {
            float* o = part + ((long)sp * rows_total + (long)r * block_size + (long)tile_id * BM) * batch + pid_b * BN
                       + h * BNH;
            auto& L = h == 0 ? tCrL : tCrL1;
            CUTE_UNROLL
            for (int i = 0; i < size(L); i += 2) {       // (i, i + 1): columns b, b + 1 of one row
                *(float2*)&o[(long)get<0>(tCcC(i)) * batch + get<1>(tCcC(i))] = make_float2(L(i), L(i + 1));
            }
        }
    } else {                                             // one range: the merge in the epilogue (its S = 1 arithmetic)
        const float* mpg = mp + (long)ugrp[r] * batch + (long)pid_b * BN;
        CUTE_UNROLL
        for (int h = 0; h < NH; h++) {
            auto& L = h == 0 ? tCrL : tCrL1;
            CUTE_UNROLL
            for (int i = 0; i < size(L); i += 2) {
                const int m = get<0>(tCcC(i)), c = get<1>(tCcC(i)) + h * BNH;
                const int bq = bpt > 1 ? m / block_size : 0;
                const long nid = nids[r + bq] + (bpt > 1 ? m - bq * block_size : tile_id * BM + m);
                const float2 M = *(const float2*)&mpg[c];
                const float o0 = M.x == -INFINITY ? -INFINITY : (logf(L(i)) + M.x);
                const float o1 = M.y == -INFINITY ? -INFINITY : (logf(L(i + 1)) + M.y);
                *(float2*)&node_mars[nid * batch + pid_b * BN + c] = make_float2(o0, o1);
            }
        }
    }
}

// ======================= the online-max form: ONE launch, the operand formed in the GEMM =======================
// `lr_om_kernel` takes the virtual batch without the pre-pass chain: the rows' exponentials of the normalizers (`V`,
// bf16 [N, rows]: `exp(B - shift)`, the rows (copy, sample, rank), `virtual._normalizers`) times the children's values
// ARE the operand, `u[j, (b, r)] = V[j, (b, r)] exp(c_b[j] - m_b)`, so a CTA forms it in its own stages: producers
// copy a k-step's parameter tile (cp.async, or one TMA box), V's rows of its children (one 16-byte run a row a chunk:
// a tile's columns are S samples x all K ranks, adjacent in V) and the children's values (16 bytes a row: the tile's
// S samples); converters scale V by the values in place; the consumers' MMAs as the stream kernel's. The column
// shift m_b is each sample's running max over the k-steps so far -- ONLINE: a converter warp takes its sample's max
// over a stage's rows and, where it rose, the factor exp(m_old - m_new) the consumers rescale their sums by before
// the stage (exactly 1 where it did not: the sums untouched) -- so no pass over the values comes first, and the
// epilogue writes `vnm = log(sum) + m_b + shift` (the sample's final max; shift V's row's). A row of no child (`jpos`
// -1) copies -inf for its value (its V row: zeros).
//
// MEASURED (croissant, one GPU no one else used, HMM 1024 x 1024 in blocks of 32, rank 16, batch 64, the layer's
// launches replayed in a CUDA graph, us a call; `knt` edge tiles of the layer's 16): the pre-pass chain 9.2 at one edge
// tile, 21.2 at 16 (cmax + exp + GEMM); this 4.2 / 16.6-16.9. Its main loop is the L2's: the A stream alone (TMA)
// reaches ~5.1 TB/s of the ~6.25 a 128-SM grid can draw. Not taken (each measured): the samples' maxima first, by a
// launch of their own (8.8 / 21.4) or by the GEMM's converters walking the range (7.1 / 22.0); 6 stages (no gain);
// V by TMA (16.7 -> 20.7: a 64-column box a k-step half); two-CTA clusters of 128 x 128 tiles summing two k-ranges
// (18.8-19.6: sm_120's distributed shared memory moves 32 KB a CTA in 4-14 us, so the halves went through L2;
// `setmaxnreg` is ignored in cluster kernels); TMA multicast (ptxas: "substantially reduced performance" on sm_120a).
__device__ __align__(16) float g_ninf[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};   // (a row of no child)

// The tile: OM_BM nodes x OM_BN columns (OM_BN / OM_K samples, every rank), KT = 32 children a k-step in ST stages;
// 8 consumer warps (4 x 2), 4 producer warps, 4 converter warps. TMAA: A by TMA (`tmapA`: 32 nodes x 32 k x the tile's
// 4 node blocks of 32, the slabs affine in (block, edge tile)), else cp.async.
constexpr int OM_BM = 128, OM_BN = 64, OM_K = 16, OM_ST = 5, OM_THREADS = 512;

template <int ST, bool FUSED, bool TMAA>
__global__ void __launch_bounds__(OM_THREADS, 1) lr_om_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ pbase, const bfloat16_t* __restrict__ vt,
        long vrows, long row0, const float* shift, const float* vals, const long* __restrict__ ebase,
        const int* __restrict__ cbase, const int* __restrict__ jpos, const int* __restrict__ ugrp, int knt, int mtiles,
        int block_size, int B, float* __restrict__ vnm, const long* __restrict__ nids, int num_edges,
        const __grid_constant__ CUtensorMap tmapA, const float* __restrict__ fa, float* node_mars,
        float* __restrict__ lz, long nshift, long vn0) {
    constexpr int BM = OM_BM, BN = OM_BN, K = OM_K, WM = 4, WN = 2;
    constexpr int KT = 32, NC = WM * WN * 32, NP = 128, NV = 128;
    static_assert(NC + NP + NV == OM_THREADS, "the warp roles");
    constexpr int S = BN / K;                            // the tile's samples
    static_assert(S % 4 == 0, "a row's values of the tile's samples in 16-byte runs");
    // a stage: A, V as copied (the normalizer's operand), u = V exp(c - m) (the converters'), the children's values
    constexpr int ABYTES = KT * BM * 2, UBYTES = KT * BN * 2, CBYTES = 2 * KT * S * 4;
    constexpr int STAGE = (ABYTES + 2 * UBYTES + CBYTES + 1023) / 1024 * 1024;
    constexpr int GS = BN + 4;                           // the epilogue's tiles of log S_r and Z_r, a row's stride
    static_assert(2 * BM * GS * 4 <= ST * STAGE, "the epilogue's tiles in the stages");
    constexpr int SUB = BK / KT;                         // k-steps an edge tile is
    const int cols = K * B;
    const int nbt = cols / BN;
    const int pid_b = blockIdx.x % nbt, pid_m = blockIdx.x / nbt;
    // the node tile: one of `mtiles` of a node block, or `bpt` consecutive blocks of one group (blocks of 32 or 64)
    const int bpt = BM > block_size ? BM / block_size : 1;
    const int r = bpt > 1 ? pid_m * bpt : pid_m / mtiles, tile_id = bpt > 1 ? 0 : pid_m % mtiles;
    const int nk = knt * SUB;
    const int g = ugrp[r];
    float* smsh = (float*)(smem_raw + ST * STAGE);       // [BN] the columns' final shifts (the converters')
    float* sfac = smsh + BN;                             // [ST][S] a stage's rescale factors (the converters')
    uint64_t* fullr = (uint64_t*)(sfac + ST * S);        // a stage's copies landed (the producers')
    uint64_t* full = fullr + ST;                         // ... and converted (the converters')
    uint64_t* empty = full + ST;                         // ... and read (the consumers')
    const int tid = threadIdx.x;
    auto sAl = [&]() {                                   // (TMAA: a node block's [KT k][32 nodes] a contiguous 2 KB)
        if constexpr (TMAA) return tile_to_shape(AtomA32{}, make_shape(Int<BM>{}, Int<KT>{}), Step<_2, _1>{});
        else return tile_to_shape(Atom64{}, make_shape(Int<BM>{}, Int<KT>{}));
    }();
    auto sBl = tile_to_shape(Atom64{}, make_shape(Int<BN>{}, Int<KT>{}));
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES); };             // V
    auto stU = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES + UBYTES); };    // u
    auto stC = [&](int s) { return (const float*)(smem_raw + s * STAGE + ABYTES + 2 * UBYTES); };   // [2][KT][S]
    if (tid == 0) {
        for (int s = 0; s < ST; s++) { mbar_init(&fullr[s], NP); mbar_init(&full[s], NV); mbar_init(&empty[s], NC); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");   // (visible to the TMA unit)
    }
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();           // the launch after may launch (it waits for this one)
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const long col0 = (long)pid_b * BN;
    const int b0 = pid_b * S;                            // the tile's first sample
    // The epilogue, every role's (a (node, sample) a thread, `BM S` of them), from the consumers' tiles of log S_r
    // (`gS`) and Z_r (`gZ`): the COLLAPSE -- the node's value lse_r(A + log S_r) - lse_r(A + log Z_r) and log Z, as
    // `lowrank_mixture.cu`'s `collapse_sums<16>` forms them (the same operations in the same order) -- and the share
    // differences the backward's dA takes (`vnm`: the layer's kept D, 64-byte runs). A's pieces were prefetched into L2
    // by the producers during the main loop.
    static_assert(BM * S == OM_THREADS && K == 16, "a thread an epilogue pair");
    float* gS = (float*)smem_raw;
    float* gZ = gS + BM * GS;
    auto node_of = [&](int m) {                          // a tile row's twin row
        const int bq = bpt > 1 ? m / block_size : 0;
        return nids[r + bq] + (bpt > 1 ? m - bq * block_size : tile_id * BM + m);
    };
    auto epilogue = [&]() {
        const int m = tid / S, sl = tid % S, b = b0 + sl;
        const long nid = node_of(m), i = nid - vn0;     // (the twin's row, the node vector's node)
        const long c0 = col0 + sl * K;                  // the sample's first column
        float av[K], sv[K], zv[K], hv[K];
        CUTE_UNROLL
        for (int q = 0; q < K; q++) { sv[q] = gS[m * GS + sl * K + q]; zv[q] = gZ[m * GS + sl * K + q]; }
        CUTE_UNROLL
        for (int q = 0; q < K / 4; q++) {
            const float4 x = *(const float4*)&shift[row0 + c0 + 4 * q];
            hv[4 * q] = x.x; hv[4 * q + 1] = x.y; hv[4 * q + 2] = x.z; hv[4 * q + 3] = x.w;
        }
        const float* ap = fa + (i * K) * B + b;          // A[i, r, b]: r at a stride of B
        CUTE_UNROLL
        for (int q = 0; q < K; q++) av[q] = ap[(long)q * B];
        float ms = -INFINITY, mz = -INFINITY;
        CUTE_UNROLL
        for (int q = 0; q < K; q++) {
            sv[q] = av[q] + sv[q];
            zv[q] = av[q] + (logf(zv[q]) + hv[q]);
            ms = fmaxf(ms, sv[q]);
            mz = fmaxf(mz, zv[q]);
        }
        float ss = 0.0f, sz = 0.0f;
        CUTE_UNROLL
        for (int q = 0; q < K; q++) {
            ss += expf(sv[q] - ms);
            sz += expf(zv[q] - mz);
        }
        const float ls = ms > -INFINITY ? ms + logf(ss) : -INFINITY;
        const float z = mz > -INFINITY ? mz + logf(sz) : -INFINITY;
        const bool live = ls > -INFINITY && z > -INFINITY;
        // (Z = 0, a node whose parameters are all 0: no mass, -inf -- as the collapse kernel)
        node_mars[(nid + nshift) * B + b] = live ? ls - z : -INFINITY;
        lz[i * B + b] = z;
        CUTE_UNROLL
        for (int q = 0; q < K; q++) sv[q] = live ? expf(sv[q] - ls) - expf(zv[q] - z) : 0.0f;
        CUTE_UNROLL
        for (int q = 0; q < K / 4; q++)
            *(float4*)&vnm[nid * cols + c0 + 4 * q] = make_float4(sv[4 * q], sv[4 * q + 1], sv[4 * q + 2], sv[4 * q + 3]);
    };
    auto epilogue_wait = [&]() { asm volatile("bar.sync 5, %0;" ::"n"(OM_THREADS) : "memory"); };   // (the tiles written)
    if (tid >= NC + NV) {
        // ---------------- producers: k-step t into stage t % ST once the consumers released it ----------------
        // Every per-chunk address made before the loop (pinned in registers: else the compiler remakes the swizzled
        // offsets from the thread id every stage), the slab bases and the rows' children loaded ahead, and no
        // shared-memory load in the loop (MEASURED: one costs a cp.async loop 34% to 2.1x, even predicated off).
        const int ptid = tid - NC - NV;
        constexpr int TA = ABYTES / 16, CA = (TA + NP - 1) / NP;
        constexpr int TB = UBYTES / 16, CB = (TB + NP - 1) / NP;
        constexpr int NC4 = (FUSED ? 2 : 1) * KT * (S / 4), CC = (NC4 + NP - 1) / NP;
        uint32_t oA[CA];
        int gA[CA], bA[CA];
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                const int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                oA[q] = i < TA ? (uint32_t)((char*)&sAt(mm, e) - smem_raw) : ~0u;
                const int bq = bpt > 1 ? mm / block_size : 0;
                gA[q] = e * block_size + (bpt > 1 ? mm - bq * block_size : mm) + tile_id * BM;
                bA[q] = r + bq;
            }
        }
        uint32_t oB[CB];
        int eB[CB];
        const bfloat16_t* vptr[CB];                      // V's row 0 at the chunk's columns (+ the child's row)
        {
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) {
                const int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - smem_raw) : ~0u;
                eB[q] = e;
                vptr[q] = vt + row0 + col0 + bb;
            }
        }
        uint32_t oC[CC];
        int cx[CC], ce[CC];
        const float* cptr[CC];                           // the values at the chunk's samples (+ the row's offset)
        CUTE_UNROLL
        for (int k = 0; k < CC; k++) {
            constexpr int s4 = S / 4;
            const int ci = ptid + k * NP;
            const bool on = ci < NC4;
            const int x = on ? ci / (KT * s4) : 0, rem = ci - x * (KT * s4);
            const int e = on ? rem / s4 : 0, q4 = on ? rem % s4 : 0;
            oC[k] = on ? (uint32_t)(ABYTES + 2 * UBYTES + ((x * KT + e) * S + 4 * q4) * 4) : ~0u;   // (after V, u)
            cx[k] = x; ce[k] = e;
            cptr[k] = vals + b0 + 4 * q4;
        }
        long vrows_l = vrows;                            // (strides in registers: else reloaded from the constant bank)
        int B_i = B;
        asm volatile("" : "+l"(vrows_l), "+r"(B_i));
        // the epilogue's pieces of A of tile row `ptid` (the tile's S samples of a rank one 16-byte piece), one
        // prefetched into L2 a k-step
        static_assert(NP == BM, "a producer a tile row's prefetches");
        const float* pfa = fa + (node_of(ptid) - vn0) * K * B + b0;
        auto prefetch = [&](int q) { asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(pfa + (long)q * B)); };
        CUTE_UNROLL
        for (int q = 0; q < CA; q++) asm volatile("" : "+r"(oA[q]), "+r"(gA[q]), "+r"(bA[q]));
        CUTE_UNROLL
        for (int q = 0; q < CB; q++) asm volatile("" : "+r"(oB[q]), "+r"(eB[q]), "+l"(vptr[q]));
        CUTE_UNROLL
        for (int k = 0; k < CC; k++) asm volatile("" : "+r"(oC[k]), "+r"(cx[k]), "+r"(ce[k]), "+l"(cptr[k]));
        // the parameter tiles (cp.async): each chunk's pointer at the current edge tile's slab, the next one's base
        // loaded an edge tile ahead
        int ecur = 0;
        const bfloat16_t* aptr[CA];
        long pnx[CA];
        if constexpr (!TMAA) {
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                aptr[q] = params + gA[q] + pbase[(long)bA[q] * knt];
                pnx[q] = knt > 1 ? pbase[(long)bA[q] * knt + 1] : 0;
            }
        }
        auto issue_a = [&](int t, uint32_t ss) {         // (t increasing)
            if constexpr (TMAA) {
                if (ptid == 0) {
                    const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&fullr[t % ST]);
                    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(ABYTES)
                                 : "memory");
                    tma_load_3d(ss, &tmapA, 0, t * KT, r, bar);
                }
            } else {
                if (t / SUB != ecur) {
                    ecur = t / SUB;
                    CUTE_UNROLL
                    for (int q = 0; q < CA; q++) {
                        aptr[q] = params + gA[q] + pnx[q];
                        pnx[q] = ecur + 1 < knt ? ldv_s64(pbase + (long)bA[q] * knt + ecur + 1) : 0;
                    }
                }
                const int hs = (t % SUB) * KT * block_size;
                CUTE_UNROLL
                for (int q = 0; q < CA; q++) if (TA % NP == 0 || oA[q] != ~0u) cpa16(ss + oA[q], aptr[q] + hs);
            }
        };
        // the first stages' parameter tiles while the launch before may still run (none of the pass writes them)
        for (int t = 0; t < min(nk, ST); t++) issue_a(t, sbase + t * STAGE);
        // V's children, the element rows' bases and (run 0) the rows' children, two k-steps ahead
        const int* jg = jpos + (long)g * num_edges;
        const int* cbr = FUSED ? cbase + (long)r * knt * 4 : nullptr;
        const long* ebr = ebase + (long)r * knt;
        int jr[2][CB], jcr[2][CC];
        long br[2][2];
        auto load_j = [&](int t, int (&j)[CB]) {
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) j[q] = oB[q] != ~0u ? ldv_s32(jg + t * KT + eB[q]) : -1;
        };
        auto load_b = [&](int t, long (&bb)[2]) {        // (FUSED: the products' two runs of children; else the row)
            if (FUSED) {
                const int* cq = cbr + (t / SUB) * 4 + (t % SUB) * 2;
                bb[0] = ldv_s32(cq); bb[1] = ldv_s32(cq + 1);
            } else {
                bb[0] = ldv_s64(ebr + t / SUB) + (t % SUB) * KT; bb[1] = 0;
            }
        };
        auto load_jc = [&](int t, int (&j)[CC]) {
            CUTE_UNROLL
            for (int k = 0; k < CC; k++) j[k] = oC[k] != ~0u && cx[k] == 0 ? ldv_s32(jg + t * KT + ce[k]) : 0;
        };
        load_j(0, jr[0]); load_b(0, br[0]); load_jc(0, jcr[0]);
        if (nk > 1) { load_j(1, jr[1]); load_b(1, br[1]); load_jc(1, jcr[1]); }
        cudaGridDependencySynchronize();                 // (the children's values: the launches before's)
        int s = 0, ph = 0;                               // the stage and the parity of its uses so far
        uint32_t sb_r = sbase;
        asm volatile("" : "+r"(sb_r));
        for (int jb = 0; jb < nk; jb += 2) {
            CUTE_UNROLL
            for (int u = 0; u < 2; u++) {
                const int t = jb + u;
                if (t < nk) {
                    uint32_t ss = sb_r + s * STAGE;
                    asm volatile("" : "+r"(ss));
                    if (t >= ST) {
                        mbar_wait(&empty[s], ph ^ 1);
                        issue_a(t, ss);
                    }
                    CUTE_UNROLL
                    for (int q = 0; q < CB; q++) {
                        if (TB % NP != 0 && oB[q] == ~0u) continue;
                        const int jc = jr[u][q];         // (no child: nothing copied, zeros)
                        const bfloat16_t* src = vptr[q] + (long)max(jc, 0) * vrows_l;
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(ss + oB[q]), "l"(src),
                                     "r"(jc >= 0 ? 16u : 0u) : "memory");
                    }
                    CUTE_UNROLL
                    for (int k = 0; k < CC; k++) {
                        if (oC[k] == ~0u) continue;
                        const long base = cx[k] ? br[u][1] : br[u][0];   // (a dummy chunk of a product: -1)
                        const bool ninf = cx[k] == 0 && jcr[u][k] < 0;   // (a row of no child: -inf)
                        const float* src = ninf ? g_ninf : cptr[k] + (max(base, 0L) + ce[k]) * B_i;
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(ss + oC[k]), "l"(src),
                                     "r"(ninf || base >= 0 ? 16u : 0u) : "memory");
                    }
                    cpa_mbar_arrive(&fullr[s]);          // (tracks every copy this thread issued: the stage's A too)
                    if (t < K) prefetch(t);
                    if (t + 2 < nk) { load_j(t + 2, jr[u]); load_b(t + 2, br[u]); load_jc(t + 2, jcr[u]); }
                    if (++s == ST) { s = 0; ph ^= 1; }
                }
            }
        }
        for (int q = nk; q < K; q++) prefetch(q);
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        epilogue_wait();
        epilogue();
        return;
    }
    if (tid >= NC) {
        // ---------------- converters: u = V exp(c - m) beside V, m each sample's running max ----------------
        // A lane = a row of the stage, a warp = the columns [w CPW, (w + 1) CPW): its SPW samples, CPL chunks of 8 a
        // lane; each warp takes its own samples' maxima over the stage's 32 rows (the lanes) and writes their factors
        constexpr unsigned FULLM = 0xffffffffu;
        constexpr int CPW = BN / 4, SPW = CPW / K, CPL = CPW / 8;
        static_assert(SPW >= 1 && CPW % K == 0 && NV == 128, "a converter warp's columns: whole samples");
        const int vtid = tid - NC, lane = vtid % 32, w = vtid / 32;
        uint32_t oB[CPL];
        {
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int j = 0; j < CPL; j++) oB[j] = (uint32_t)((char*)&sBt(w * CPW + j * 8, lane) - smem_raw);
        }
        CUTE_UNROLL
        for (int j = 0; j < CPL; j++) asm volatile("" : "+r"(oB[j]));
        cudaGridDependencySynchronize();                 // (the shifts: the pass's, read once at the end)
        const float shv = lane < CPW ? shift[row0 + col0 + w * CPW + lane] : 0.f;
        float mrun[SPW];                                 // the running max of the warp's samples (every lane)
        CUTE_UNROLL
        for (int i = 0; i < SPW; i++) mrun[i] = -INFINITY;
        int s = 0, ph = 0;
        for (int jj = 0; jj < nk; jj++) {
            mbar_wait(&fullr[s], ph);
            const uint32_t ss = sbase + s * STAGE;
            const float* cs = stC(s);
            float cv[SPW], mx[SPW];                      // row `lane`'s values of the warp's samples (both runs)
            CUTE_UNROLL
            for (int i = 0; i < SPW; i++) {
                const int sl = w * SPW + i;
                float a_ = cs[lane * S + sl];
                if (FUSED) a_ += cs[(KT + lane) * S + sl];
                cv[i] = mx[i] = a_;
            }
            CUTE_UNROLL
            for (int bit = 16; bit >= 1; bit >>= 1)
                CUTE_UNROLL
                for (int i = 0; i < SPW; i++) mx[i] = fmaxf(mx[i], __shfl_xor_sync(FULLM, mx[i], bit));
            float ex[SPW];
            CUTE_UNROLL
            for (int i = 0; i < SPW; i++) {
                const float mold = mrun[i], mnew = fmaxf(mold, mx[i]);
                // (the consumers' factor: the sums so far where the max rose, else exactly 1)
                if (lane == 0) sfac[s * S + w * SPW + i] = mnew > mold && mold > -INFINITY ? __expf(mold - mnew) : 1.f;
                mrun[i] = mnew;
                ex[i] = mnew > -INFINITY ? __expf(cv[i] - mnew) : 0.f;
            }
            CUTE_UNROLL
            for (int j = 0; j < CPL; j++) {
                const float e_ = ex[j * 8 / K];          // (chunk j: 8 columns of one sample)
                uint4 vv;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(vv.x), "=r"(vv.y), "=r"(vv.z), "=r"(vv.w)
                             : "r"(ss + oB[j]));
                unsigned h[4] = {vv.x, vv.y, vv.z, vv.w};
                CUTE_UNROLL
                for (int x = 0; x < 4; x++) {
                    __nv_bfloat162 p2 = *reinterpret_cast<__nv_bfloat162*>(&h[x]);
                    const float2 f = __bfloat1622float2(p2);
                    p2 = __floats2bfloat162_rn(f.x * e_, f.y * e_);
                    h[x] = *reinterpret_cast<unsigned*>(&p2);
                }
                asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ss + oB[j] + UBYTES), "r"(h[0]), "r"(h[1]),
                             "r"(h[2]), "r"(h[3]) : "memory");
            }
            mbar_arrive(&full[s]);
            if (++s == ST) { s = 0; ph ^= 1; }
        }
        {                                                // the columns' shifts: the running max + V's row's
            float m = mrun[0];
            CUTE_UNROLL
            for (int i = 1; i < SPW; i++) if ((lane % CPW) / K == i) m = mrun[i];
            if (lane < CPW) smsh[w * CPW + lane] = m > -INFINITY ? m + shv : -INFINITY;
        }
        asm volatile("bar.arrive 3, %0;" ::"n"(NC + NV) : "memory");   // (smsh: for the consumers' epilogue)
        epilogue_wait();
        epilogue();
        return;
    }
    // ---------------- consumers: the k-steps in order into two fp32 accumulators: S_r (A u) and Z_r (A V) ----------------
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
    Tensor tCrL = thr.partition_fragment_C(cC);         // S_r: the data half (u)
    Tensor tCrZ = thr.partition_fragment_C(cC);         // Z_r: the normalizer (V, as copied)
    clear(tCrL);
    clear(tCrZ);
    constexpr int KB = size<2>(tCrA);
    // tCrL is ((2, 2), MI, NI): n-block ni the columns 16 ni + 8 wn + [0, 8) -- sample ni (rank 16)
    constexpr int MI = BM / (16 * WM), NI = size<2>(tCrL);
    static_assert(NI == S && K == 16, "a fragment n-block a sample");
    int s = 0, ph = 0;
    for (int j = 0; j < nk; j++) {
        mbar_wait(&full[s], ph);
        auto tXsA = tA.partition_S(make_tensor(make_smem_ptr(stA(s)), sAl));
        auto tXsV = tB.partition_S(make_tensor(make_smem_ptr(stB(s)), sBl));
        auto tXsU = tB.partition_S(make_tensor(make_smem_ptr(stU(s)), sBl));
        // the stage's A fragments once, V's into the B fragments for Z, then u's into the same registers for S
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {
            copy(s2rA, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2rB, tXsV(_, _, kb), tXrB(_, _, kb));
        }
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrZ);
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) copy(s2rB, tXsU(_, _, kb), tXrB(_, _, kb));
        mbar_arrive(&empty[s]);                          // the stage is read
        float f[S];                                      // the samples' rescale factors (1: no rise)
        CUTE_UNROLL
        for (int x4 = 0; x4 < S / 4; x4++) {
            const float4 q = *(const float4*)&sfac[s * S + 4 * x4];
            f[4 * x4] = q.x; f[4 * x4 + 1] = q.y; f[4 * x4 + 2] = q.z; f[4 * x4 + 3] = q.w;
        }
        bool rise = false;
        CUTE_UNROLL
        for (int x = 0; x < S; x++) rise |= f[x] != 1.f;
        if (rise) {                                      // (a sample's max rose this stage: the sums so far rescaled)
            CUTE_UNROLL
            for (int i = 0; i < size(tCrL); i++) tCrL(i) *= f[i / (4 * MI)];
        }
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
        if (++s == ST) { s = 0; ph ^= 1; }
    }
    cudaGridDependencySynchronize();                     // (the epilogue's own guarantee: the launch before is done)
    asm volatile("bar.sync 3, %0;" ::"n"(NC + NV) : "memory");   // (smsh: the converters'; every stage read)
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i += 2) {            // (i, i + 1): columns c, c + 1 of one row
        const int m = get<0>(tCcC(i)), c = get<1>(tCcC(i));
        const float2 M = *(const float2*)&smsh[c];
        const float o0 = M.x == -INFINITY ? -INFINITY : (logf(tCrL(i)) + M.x);
        const float o1 = M.y == -INFINITY ? -INFINITY : (logf(tCrL(i + 1)) + M.y);
        *(float2*)&gS[m * GS + c] = make_float2(o0, o1);
        *(float2*)&gZ[m * GS + c] = make_float2(tCrZ(i), tCrZ(i + 1));
    }
    epilogue_wait();
    epilogue();
}

// ======================= the backward's one-launch form: the data half's element flows =======================
// On the virtual batch (its node values taken as 0) the data half's virtual element flows are one product over the
// parents, `G[j, v] = sum_i theta[i, j] U[i, v]`, `U[i, (b, r)] = exp(lf_b[i] - log S_b[i] + A_b[i, r])` -- the plain
// path writes `vnf = log U` and the virtual child values (`lowrank_backward_pre`), runs the stream backward on them
// (pre-pass, GEMM, merge: `vef = vem + log G`) and collapses (`lowrank_collapse_bw`). `lr_om_bw_kernel` does it in one
// launch, as `lr_om_kernel` does the forward: producers copy a k-step's parameter tile (theta^T: a child's 32 parents
// contiguous in its slab), the rows of the pass's `exp(A - shift)` (`lowrank_a_operand`: Ua [M, rows], the rows (copy,
// sample, rank)) of its 32 parents and their flows, values and log normalizers; converters scale Ua in place by
// `exp(lf - log S - m_b)`, m_b each sample's running max over the parents so far (the consumers rescale their sums
// where it rose); the epilogue forms `log G = log(sum) + m_b + shift` and from it, a (child, sample) a thread, the
// element flow `lse_r (c_b[j] + B_b[r, j] + log G)` -- written as the product's two children's node flows where the
// product layer below is fused in (FUSED) -- and dB's data half, `exp(c + B + log G)` (GRAD). A parent of no flow or
// no mass (lf or log S -inf): 0.
// GRAD: and dB's NORMALIZER half in the same launch, `- exp(B_b[r, j] + log Gn)`, `Gn[j, v] = sum_i theta[i, j]
// exp(lf_b[i] - log Z_b[i] + A_b[i, r])` (what the pass's end would form as `theta^T U` and take off with
// `lowrank_normalizer_grads`): the same parameter fragments, a second accumulator -- the converters write `Un = Ua
// exp(lf - log Z - n_b)` beside U, n_b its own running max (a parent of no flow or a log Z of -inf: 0). Scope (`om_bw_serves` and the plan): rank OM_K, B % (OM_BN / OM_K), the tile's
// children of one group of child blocks (equal parents), every parent a real node of the layer's node vector (no
// dummy), parents' blocks a multiple of 32.
// GRAD: and dA, by the grid's last `n_da` CTAs -- the free SMs of its last wave: a node vector's (node, sample) pairs
// strided over them, `dLL / dA_r = f D_r` from the share differences the forward kept (`kd`, `kcols` wide, the twin's
// rows), as the plain path's `lowrank_backward_pre` forms it (the same operations, the same bits): its loads and adds
// beside the data half's GEMM instead of after it.
// MEASURED (toast, HMM 1024 x 31, rank 16, B64, us a layer as the GPU timeline's frontier advances): the plain data
// half 54.5 (backward_pre 20.8, the element stream's pre-pass 5.4, GEMM 14.8, merge 4.2, collapse_bw 9.3); this kernel
// first 39.1 with dA a launch of its own beside (17.1) -- its main loop alone 16.8, the tail epilogue the rest: the
// staged [K, N, B] layout's 16-byte pieces of B and dB, nothing overlapping them --; with the epilogue's pieces
// prefetched into L2 a k-step at a time by the producers and the epilogue on every role's threads 28.8 (+ 16.9); with
// dA in the grid's free SMs 34.1 alone. Backward wall plain -> this: B16 1.83 -> 1.55 ms, B64 4.13 -> 3.57, B256
// 15.09 -> 13.73.
template <int KC>
__device__ __forceinline__ void load_run4f(const float* __restrict__ p, float (&v)[KC]) {
    CUTE_UNROLL
    for (int q = 0; q < KC / 4; q++) {
        const float4 x = reinterpret_cast<const float4*>(p)[q];
        v[4 * q] = x.x; v[4 * q + 1] = x.y; v[4 * q + 2] = x.z; v[4 * q + 3] = x.w;
    }
}

template <int ST, bool FUSED, bool GRAD>
__global__ void __launch_bounds__(OM_THREADS, 1) lr_om_bw_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ ppid, const long* __restrict__ ppar,
        const bfloat16_t* __restrict__ ua, long urows, long row0, const float* sha, const float* nflow,
        const float* nmar, const float* lz, long nshift, long vn0, const long* __restrict__ chids, int npb, int cbs,
        int pbs, int mtiles, int B, const float* vals, const int* __restrict__ table, long first,
        const float* __restrict__ fb, const int* __restrict__ jrow, long e_lo, int N, float* flows, float* grad_b,
        const float* kd, long kcols, float* __restrict__ grad_a, int M, int n_gemm, int n_da) {
    constexpr int BM = OM_BM, BN = OM_BN, K = OM_K, WM = 4, WN = 2;
    if ((int)blockIdx.x >= n_gemm) {
        // ---------------- the dA CTAs ----------------
        cudaTriggerProgrammaticLaunchCompletion();
        if (!GRAD) return;
        cudaGridDependencySynchronize();                 // (the node flows: the launches before's)
        const long P = (long)M * B, stride = (long)n_da * OM_THREADS;
        for (long t = (long)(blockIdx.x - n_gemm) * OM_THREADS + threadIdx.x; t < P; t += stride) {
            const int b = (int)(t % B), i = (int)(t / B);
            const long row = vn0 + nshift + i;           // the node's circuit row
            const long at = (long)i * K * B + b;         // A[i, r, b]: r at a stride of B
            const float lf = nflow[row * B + b];
            const float z = lz[(long)i * B + b];
            const float ls = nmar[row * B + b] + z;      // log S = log n + log Z
            if (!((lf > -INFINITY) && (ls > -INFINITY))) continue;
            const float ef = expf(lf);
            float dv[K], gv[K];
            load_run4f(kd + (vn0 + i) * kcols + (long)b * K, dv);
            CUTE_UNROLL
            for (int r = 0; r < K; r++) gv[r] = grad_a[at + r * B];
            CUTE_UNROLL
            for (int r = 0; r < K; r++) grad_a[at + r * B] = gv[r] + ef * dv[r];
        }
        return;
    }
    constexpr int KT = 32, NC = WM * WN * 32, NP = 128, NV = 128;
    static_assert(NC + NP + NV == OM_THREADS, "the warp roles");
    constexpr int S = BN / K;                            // the tile's samples
    static_assert(S % 4 == 0, "a row's values of the tile's samples in 16-byte runs");
    constexpr int NU = GRAD ? 2 : 1;                     // (U; and Un, GRAD)
    constexpr int ABYTES = KT * BM * 2, UBYTES = KT * BN * 2, CBYTES = 3 * KT * S * 4;   // (flows, values, log Z)
    constexpr int STAGE = (ABYTES + NU * UBYTES + CBYTES + 1023) / 1024 * 1024;
    constexpr int GS = BN + 4;                           // the epilogue's log G (log Gn) tile, a row's stride (padded)
    static_assert(NU * BM * GS * 4 <= ST * STAGE, "the log G tiles in the stages");
    const int cols = K * B, nbt = cols / BN;
    const int pid_b = blockIdx.x % nbt, pid_m = blockIdx.x / nbt;
    // the child tile: one of `mtiles` of a child block, or `bpt` consecutive blocks of one group (equal parents)
    const int bpt = BM > cbs ? BM / cbs : 1;
    const int cb0 = bpt > 1 ? pid_m * bpt : pid_m / mtiles, tile_id = bpt > 1 ? 0 : pid_m % mtiles;
    const int SPB = pbs / KT, nk = npb * SPB;            // k-steps a parent block, all
    float* smsh = (float*)(smem_raw + ST * STAGE);       // [NU][BN] the columns' final shifts (the converters')
    float* sfac = smsh + NU * BN;                        // [NU][ST][S] a stage's rescale factors (the converters')
    uint64_t* fullr = (uint64_t*)(sfac + NU * ST * S);
    uint64_t* full = fullr + ST;
    uint64_t* empty = full + ST;
    const int tid = threadIdx.x;
    auto sAl = tile_to_shape(AtomAK32{}, make_shape(Int<BM>{}, Int<KT>{}));   // (K-major: a child's parents a line)
    auto sBl = tile_to_shape(Atom64{}, make_shape(Int<BN>{}, Int<KT>{}));
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES); };
    auto stN = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES + UBYTES); };   // (GRAD: Un)
    auto stC = [&](int s) { return (const float*)(smem_raw + s * STAGE + ABYTES + NU * UBYTES); };   // [3][KT][S]
    if (tid == 0)
        for (int s = 0; s < ST; s++) { mbar_init(&fullr[s], NP); mbar_init(&full[s], NV); mbar_init(&empty[s], NC); }
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const long col0 = (long)pid_b * BN;
    const int b0 = pid_b * S;
    // The epilogue, every role's (a (child, sample) pair a thread, `BM S` of them): from the consumers' log G tile `gs`,
    // the pair's element flow over the sample's K ranks and dB's data half -- every load (the child's value, B, dB)
    // before the first store; the pieces were prefetched into L2 by the producers during the main loop
    static_assert(BM * S == OM_THREADS, "a thread an epilogue pair");
    float* gs = (float*)smem_raw;
    float* gn = gs + BM * GS;                            // (GRAD: log Gn)
    const long NB = (long)N * B;
    auto child_of = [&](int m, long& e, int& j) {         // a tile row's element row and the node vector's child
        const int bq = bpt > 1 ? m / cbs : 0;
        e = chids[cb0 + bq] + (bpt > 1 ? m - bq * cbs : tile_id * BM + m);
        j = jrow[e - e_lo];
    };
    auto epilogue = [&]() {
        const int m = tid / S, sl = tid % S, b = b0 + sl;
        long e;
        int j;
        child_of(m, e, j);
        if (j < 0) return;                               // (no child of the node vector)
        long ch0 = 0, ch1 = 0;
        float c;
        if (FUSED) {
            const int* chp = table + (e - first) * 2;
            ch0 = chp[0]; ch1 = chp[1];
            c = vals[ch0 * B + b] + vals[ch1 * B + b];
        } else {
            c = vals[e * B + b];
        }
        const float* bsrc = fb + (long)j * B + b;        // B[r, j, b]: r at a stride of N B
        float* g = GRAD ? grad_b + (long)j * B + b : nullptr;
        float x[K], gv[K];
        CUTE_UNROLL
        for (int r = 0; r < K; r++) x[r] = bsrc[r * NB];
        if (GRAD) {
            CUTE_UNROLL
            for (int r = 0; r < K; r++) gv[r] = g[r * NB];
        }
        float nv[K];                                     // (GRAD: the normalizer half, exp(B + log Gn))
        if (GRAD) {
            CUTE_UNROLL
            for (int r = 0; r < K; r++) {
                const float y = x[r] + gn[m * GS + sl * K + r];
                nv[r] = y > -INFINITY ? __expf(y) : 0.f;
            }
        }
        float mx = -INFINITY;
        CUTE_UNROLL
        for (int r = 0; r < K; r++) {
            x[r] = c + x[r] + gs[m * GS + sl * K + r];
            mx = fmaxf(mx, x[r]);
        }
        float sm = 0.f;
        if (mx > -INFINITY) {
            CUTE_UNROLL
            for (int r = 0; r < K; r++) sm += __expf(x[r] - mx);
        }
        const float lef = mx > -INFINITY ? mx + logf(sm) : -INFINITY;
        if (FUSED) { flows[ch0 * B + b] = lef; flows[ch1 * B + b] = lef; }
        else flows[e * B + b] = lef;
        if (GRAD) {
            CUTE_UNROLL
            for (int r = 0; r < K; r++) g[r * NB] = (gv[r] + (x[r] > -INFINITY ? __expf(x[r]) : 0.f)) - nv[r];
        }
    };
    auto epilogue_wait = [&]() { asm volatile("bar.sync 5, %0;" ::"n"(OM_THREADS) : "memory"); };   // (gs written)
    if (tid >= NC + NV) {
        // ---------------- producers: k-step t into stage t % ST once the consumers released it ----------------
        const int ptid = tid - NC - NV;
        constexpr int TA = ABYTES / 16, CA = (TA + NP - 1) / NP;
        constexpr int TB = UBYTES / 16, CB = (TB + NP - 1) / NP;
        constexpr int NC4 = 3 * KT * (S / 4), CC = (NC4 + NP - 1) / NP;
        // the epilogue's pieces of tile row `ptid` (its S samples one 16-byte piece each: the child's value -- or
        // the product's two children's --, B's K ranks, dB's), one prefetched into L2 a k-step
        static_assert(NP == BM, "a producer a tile row's prefetches");
        long pe;
        int pj;
        child_of(ptid, pe, pj);
        const float* pf[2] = {vals, vals};
        int npf = 0;
        if (pj >= 0) {
            if (FUSED) {
                const int* chp = table + (pe - first) * 2;
                pf[0] = vals + (long)chp[0] * B + b0; pf[1] = vals + (long)chp[1] * B + b0;
            } else {
                pf[0] = vals + pe * B + b0; pf[1] = pf[0];
            }
            npf = 2 + K + (GRAD ? K : 0);
        }
        const float* pfb = fb + (long)(pj >= 0 ? pj : 0) * B + b0;
        const float* pdb = GRAD ? grad_b + (long)(pj >= 0 ? pj : 0) * B + b0 : fb;
        auto prefetch = [&](int q) {                     // (q < npf)
            const float* a = q < 2 ? pf[q] : q < 2 + K ? pfb + (q - 2) * NB : pdb + (q - 2 - K) * NB;
            asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(a));
        };
        // A: a chunk is 8 parents of child row m -- its block's slab of (block, parent block), the row in it
        uint32_t oA[CA];
        int cA[CA], mA[CA], kA[CA];
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                const int i = ptid + q * NP, m = i / (KT / 8), kk = (i % (KT / 8)) * 8;
                oA[q] = i < TA ? (uint32_t)((char*)&sAt(m, kk) - smem_raw) : ~0u;
                const int bq = bpt > 1 ? m / cbs : 0;
                cA[q] = cb0 + bq;
                mA[q] = (bpt > 1 ? m - bq * cbs : m + tile_id * BM) * pbs + kk;
                kA[q] = kk;
            }
        }
        // Ua: a chunk is 8 columns of parent row e of the k-step
        uint32_t oB[CB];
        int eB[CB];
        const bfloat16_t* uptr[CB];                      // Ua at the chunk's columns, node row 0
        {
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) {
                const int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - smem_raw) : ~0u;
                eB[q] = e;
                uptr[q] = ua + row0 + col0 + bb;
            }
        }
        // the parents' flows, values and log normalizers: a chunk is the tile's samples of one of them, row e
        uint32_t oC[CC];
        int cx[CC], ce[CC];
        CUTE_UNROLL
        for (int k = 0; k < CC; k++) {
            constexpr int s4 = S / 4;
            const int ci = ptid + k * NP;
            const bool on = ci < NC4;
            const int x = on ? ci / (KT * s4) : 0, rem = ci - x * (KT * s4);
            const int e = on ? rem / s4 : 0, q4 = on ? rem % s4 : 0;
            oC[k] = on ? (uint32_t)(ABYTES + NU * UBYTES + ((x * KT + e) * S + 4 * q4) * 4) : ~0u;
            cx[k] = x; ce[k] = e + (4 * q4 << 8);        // (the row and, above bit 8, the sample offset)
        }
        long urows_l = urows;
        int B_i = B;
        asm volatile("" : "+l"(urows_l), "+r"(B_i));
        CUTE_UNROLL
        for (int q = 0; q < CA; q++) asm volatile("" : "+r"(oA[q]), "+r"(cA[q]), "+r"(mA[q]), "+r"(kA[q]));
        CUTE_UNROLL
        for (int q = 0; q < CB; q++) asm volatile("" : "+r"(oB[q]), "+r"(eB[q]), "+l"(uptr[q]));
        CUTE_UNROLL
        for (int k = 0; k < CC; k++) asm volatile("" : "+r"(oC[k]), "+r"(cx[k]), "+r"(ce[k]));
        // a k-step's tables, a step ahead: its parent block's slab per chunk (A), its first parent's twin row (shared
        // by the tile's blocks: the group's)
        auto slabs = [&](int t, long (&pa)[CA], long& p0) {
            const int pb = t / SPB, sub = t - pb * SPB;
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) pa[q] = oA[q] != ~0u ? ldv_s64(ppid + (long)cA[q] * npb + pb) + sub * KT : 0;
            p0 = ldv_s64(ppar + (long)cb0 * npb + pb) + sub * KT;
        };
        long pa[2][CA], p0[2];
        slabs(0, pa[0], p0[0]);
        if (nk > 1) slabs(1, pa[1], p0[1]);
        cudaGridDependencySynchronize();                 // (the flows above: the launches before's)
        int s = 0, ph = 0;
        uint32_t sb_r = sbase;
        asm volatile("" : "+r"(sb_r));
        for (int jb = 0; jb < nk; jb += 2) {
            CUTE_UNROLL
            for (int u = 0; u < 2; u++) {
                const int t = jb + u;
                if (t < nk) {
                    uint32_t ss = sb_r + s * STAGE;
                    asm volatile("" : "+r"(ss));
                    if (t >= ST) mbar_wait(&empty[s], ph ^ 1);
                    CUTE_UNROLL
                    for (int q = 0; q < CA; q++)
                        if (TA % NP == 0 || oA[q] != ~0u) cpa16(ss + oA[q], params + pa[u][q] + mA[q]);
                    const long i0 = p0[u] - vn0;         // the k-step's first parent, its node index
                    CUTE_UNROLL
                    for (int q = 0; q < CB; q++)
                        if (TB % NP == 0 || oB[q] != ~0u) cpa16(ss + oB[q], uptr[q] + (i0 + eB[q]) * urows_l);
                    CUTE_UNROLL
                    for (int k = 0; k < CC; k++) {
                        if (oC[k] == ~0u) continue;
                        const int e = ce[k] & 0xff, so = ce[k] >> 8;
                        const float* src = cx[k] == 2 ? lz + (i0 + e) * B_i + b0 + so
                                                      : (cx[k] ? nmar : nflow) + (p0[u] + e + nshift) * B_i + b0 + so;
                        cpa16(ss + oC[k], src);
                    }
                    cpa_mbar_arrive(&fullr[s]);
                    if (t < npf) prefetch(t);
                    if (t + 2 < nk) slabs(t + 2, pa[u], p0[u]);
                    if (++s == ST) { s = 0; ph ^= 1; }
                }
            }
        }
        for (int q = nk; q < npf; q++) prefetch(q);
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        epilogue_wait();
        epilogue();
        return;
    }
    if (tid >= NC) {
        // ---------------- converters: U = Ua exp(lf - log S - m), in place, m each sample's running max ----------------
        constexpr unsigned FULLM = 0xffffffffu;
        constexpr int CPW = BN / 4, SPW = CPW / K, CPL = CPW / 8;
        static_assert(SPW >= 1 && CPW % K == 0 && NV == 128, "a converter warp's columns: whole samples");
        const int vtid = tid - NC, lane = vtid % 32, w = vtid / 32;
        uint32_t oB[CPL];
        {
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int j = 0; j < CPL; j++) oB[j] = (uint32_t)((char*)&sBt(w * CPW + j * 8, lane) - smem_raw);
        }
        CUTE_UNROLL
        for (int j = 0; j < CPL; j++) asm volatile("" : "+r"(oB[j]));
        cudaGridDependencySynchronize();
        const float shv = lane < CPW ? sha[row0 + col0 + w * CPW + lane] : 0.f;
        float mrun[NU][SPW];                             // (U's running max; Un's)
        CUTE_UNROLL
        for (int u = 0; u < NU; u++)
            CUTE_UNROLL
            for (int i = 0; i < SPW; i++) mrun[u][i] = -INFINITY;
        int s = 0, ph = 0;
        for (int jj = 0; jj < nk; jj++) {
            mbar_wait(&fullr[s], ph);
            const uint32_t ss = sbase + s * STAGE;
            const float* cs = stC(s);
            // parent row `lane`'s lf - log S (lf - log Z: Un's) at the warp's samples
            float cv[NU][SPW], mx[NU][SPW];
            CUTE_UNROLL
            for (int i = 0; i < SPW; i++) {
                const int sl = w * SPW + i;
                const float lf = cs[lane * S + sl], z = cs[(2 * KT + lane) * S + sl], ls = cs[(KT + lane) * S + sl] + z;
                cv[0][i] = mx[0][i] = (lf > -INFINITY && ls > -INFINITY) ? lf - ls : -INFINITY;
                if (GRAD) cv[NU - 1][i] = mx[NU - 1][i] = (lf > -INFINITY && z > -INFINITY) ? lf - z : -INFINITY;
            }
            CUTE_UNROLL
            for (int bit = 16; bit >= 1; bit >>= 1)
                CUTE_UNROLL
                for (int u = 0; u < NU; u++)
                    CUTE_UNROLL
                    for (int i = 0; i < SPW; i++) mx[u][i] = fmaxf(mx[u][i], __shfl_xor_sync(FULLM, mx[u][i], bit));
            float ex[NU][SPW];
            CUTE_UNROLL
            for (int u = 0; u < NU; u++)
                CUTE_UNROLL
                for (int i = 0; i < SPW; i++) {
                    const float mold = mrun[u][i], mnew = fmaxf(mold, mx[u][i]);
                    if (lane == 0)
                        sfac[(u * ST + s) * S + w * SPW + i] = mnew > mold && mold > -INFINITY ? __expf(mold - mnew) : 1.f;
                    mrun[u][i] = mnew;
                    ex[u][i] = mnew > -INFINITY ? __expf(cv[u][i] - mnew) : 0.f;
                }
            CUTE_UNROLL
            for (int j = 0; j < CPL; j++) {
                const float e_ = ex[0][j * 8 / K], en = ex[NU - 1][j * 8 / K];
                uint4 vv;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(vv.x), "=r"(vv.y), "=r"(vv.z), "=r"(vv.w)
                             : "r"(ss + oB[j]));
                unsigned h[4] = {vv.x, vv.y, vv.z, vv.w}, hn[4];
                CUTE_UNROLL
                for (int x = 0; x < 4; x++) {
                    __nv_bfloat162 p2 = *reinterpret_cast<__nv_bfloat162*>(&h[x]);
                    const float2 f = __bfloat1622float2(p2);
                    p2 = __floats2bfloat162_rn(f.x * e_, f.y * e_);
                    h[x] = *reinterpret_cast<unsigned*>(&p2);
                    if (GRAD) {
                        __nv_bfloat162 q2 = __floats2bfloat162_rn(f.x * en, f.y * en);
                        hn[x] = *reinterpret_cast<unsigned*>(&q2);
                    }
                }
                asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ss + oB[j]), "r"(h[0]), "r"(h[1]), "r"(h[2]),
                             "r"(h[3]) : "memory");
                if (GRAD)
                    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ss + oB[j] + UBYTES), "r"(hn[0]),
                                 "r"(hn[1]), "r"(hn[2]), "r"(hn[3]) : "memory");
            }
            mbar_arrive(&full[s]);
            if (++s == ST) { s = 0; ph ^= 1; }
        }
        CUTE_UNROLL
        for (int u = 0; u < NU; u++) {                   // the columns' shifts: the running max + Ua's row's
            float m = mrun[u][0];
            CUTE_UNROLL
            for (int i = 1; i < SPW; i++) if ((lane % CPW) / K == i) m = mrun[u][i];
            if (lane < CPW) smsh[u * BN + w * CPW + lane] = (m > -INFINITY && shv > -INFINITY) ? m + shv : -INFINITY;
        }
        asm volatile("bar.arrive 3, %0;" ::"n"(NC + NV) : "memory");
        epilogue_wait();
        epilogue();
        return;
    }
    // ---------------- consumers: the k-steps into one fp32 accumulator, then the epilogue ----------------
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    auto tA = s2rA.get_thread_slice(tid);
    auto tB = s2rB.get_thread_slice(tid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    Tensor tCrN = thr.partition_fragment_C(cC);          // (GRAD: Gn's)
    clear(tCrL);
    if (GRAD) clear(tCrN);
    constexpr int KB = size<2>(tCrA);
    constexpr int MI = BM / (16 * WM), NI = size<2>(tCrL);
    static_assert(NI == S && K == 16, "a fragment n-block a sample");
    static_assert(!GRAD || KB >= 2, "Un's first fragments loaded while U's last one is in use");
    auto rescale = [&](auto& acc, int u, int s) {        // (a sample's running max rose: its sums so far scaled)
        float f[S];
        CUTE_UNROLL
        for (int x4 = 0; x4 < S / 4; x4++) {
            const float4 q = *(const float4*)&sfac[(u * ST + s) * S + 4 * x4];
            f[4 * x4] = q.x; f[4 * x4 + 1] = q.y; f[4 * x4 + 2] = q.z; f[4 * x4 + 3] = q.w;
        }
        bool rise = false;
        CUTE_UNROLL
        for (int x = 0; x < S; x++) rise |= f[x] != 1.f;
        if (rise) {
            CUTE_UNROLL
            for (int i = 0; i < size(acc); i++) acc(i) *= f[i / (4 * MI)];
        }
    };
    int s = 0, ph = 0;
    for (int j = 0; j < nk; j++) {
        mbar_wait(&full[s], ph);
        auto tXsA = tA.partition_S(make_tensor(make_smem_ptr(stA(s)), sAl));
        auto tXsB = tB.partition_S(make_tensor(make_smem_ptr(stB(s)), sBl));
        auto tXsN = tB.partition_S(make_tensor(make_smem_ptr(stN(s)), sBl));
        copy(s2rA, tXsA(_, _, 0), tXrA(_, _, 0));
        copy(s2rB, tXsB(_, _, 0), tXrB(_, _, 0));
        rescale(tCrL, 0, s);
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {
            if (kb + 1 < KB) {
                copy(s2rA, tXsA(_, _, kb + 1), tXrA(_, _, kb + 1));
                copy(s2rB, tXsB(_, _, kb + 1), tXrB(_, _, kb + 1));
            } else if (GRAD) {
                copy(s2rB, tXsN(_, _, 0), tXrB(_, _, 0));    // (the parameters' fragments again: Gn's too)
            } else {
                mbar_arrive(&empty[s]);
            }
            cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
        }
        if (GRAD) {
            rescale(tCrN, 1, s);
            CUTE_UNROLL
            for (int kb = 0; kb < KB; kb++) {
                if (kb + 1 < KB) copy(s2rB, tXsN(_, _, kb + 1), tXrB(_, _, kb + 1));
                else mbar_arrive(&empty[s]);
                cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrN);
            }
        }
        if (++s == ST) { s = 0; ph ^= 1; }
    }
    cudaGridDependencySynchronize();                     // (the epilogue's own guarantee: the launch before is done)
    asm volatile("bar.sync 3, %0;" ::"n"(NC + NV) : "memory");   // (smsh: the converters'; every stage read)
    // log G into the free stages, then the epilogue (every role's)
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i += 2) {
        const int m = get<0>(tCcC(i)), c = get<1>(tCcC(i));
        const float2 M = *(const float2*)&smsh[c];
        const float o0 = M.x == -INFINITY ? -INFINITY : logf(tCrL(i)) + M.x;
        const float o1 = M.y == -INFINITY ? -INFINITY : logf(tCrL(i + 1)) + M.y;
        *(float2*)&gs[m * GS + c] = make_float2(o0, o1);
        if (GRAD) {
            const float2 Mn = *(const float2*)&smsh[BN + c];
            const float n0 = Mn.x == -INFINITY ? -INFINITY : logf(tCrN(i)) + Mn.x;
            const float n1 = Mn.y == -INFINITY ? -INFINITY : logf(tCrN(i + 1)) + Mn.y;
            *(float2*)&gn[m * GS + c] = make_float2(n0, n1);
        }
    }
    epilogue_wait();
    epilogue();
}

// ---------------- merge: a thread per 4 columns of a node ----------------
template <int S>
__global__ void __launch_bounds__(256) lr_merge_kernel(
        float* __restrict__ node_mars, const float* part, const float* mp,
        const long* __restrict__ nids, const int* __restrict__ ugrp, int batch, int block_size, long rows_total) {
    cudaGridDependencySynchronize();                     // (the GEMM's partials)
    cudaTriggerProgrammaticLaunchCompletion();           // the launch after may launch: it waits for this one, and reads
                                                         // what this one writes by coherent loads (see the top)
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    const int bq = batch / 4;
    if (idx >= rows_total * bq) return;
    const long n = idx / bq;
    const int b = (int)(idx % bq) * 4;
    const int r = (int)(n / block_size), g = ugrp[r];
    float4 lp[S], xm[S];
    CUTE_UNROLL
    for (int p = 0; p < S; p++) {                        // (every load issued before the arithmetic)
        lp[p] = *(const float4*)&part[((long)p * rows_total + n) * batch + b];
        xm[p] = *(const float4*)&mp[((long)g * S + p) * batch + b];
    }
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
        out[c] = (M == -INFINITY) ? -INFINITY : (logf(L) + M);
    }
    *(float4*)&node_mars[nid * batch + b] = make_float4(out[0], out[1], out[2], out[3]);
}

// ---------------- host ----------------
// `kernel<<<grid, block, smem, st>>>(args...)` as a programmatic dependent of the launch before it (see the top)
template <typename... P, typename... A>
static void launch_pdl(void (*kernel)(P...), dim3 grid, dim3 block, int smem, cudaStream_t st, A... args) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid; lc.blockDim = block; lc.dynamicSmemBytes = smem; lc.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at; lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kernel, static_cast<P>(args)...));
}

template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
static void launch_gemm(const bfloat16_t* params, const long* pbase, const bfloat16_t* u, const int* ugrp, float* part,
                        int knt, int rows, int block_size, int batch, int num_edges, int split, float* node_mars,
                        const float* mp, const long* nids, float* cmax, int ncm, cudaStream_t st) {
    constexpr int STAGE = (KT * BM * 2 + KT * BN * 2 + 1023) / 1024 * 1024;
    constexpr int SMEM = ST * STAGE + 2 * ST * 8;
    constexpr int NTH = (WM * WN + PW) * 32;
    static_assert(SMEM <= 101376, "a config past sm_120's 99 KB of shared memory a block");
    auto kernel = lr_gemm_kernel<BM, BN, WM, WN, PW, ST, KT>;
    // the shared-memory attribute once per device and instantiation (the call costs host time on every launch)
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if (dev < 64) attr_set[dev] = true;
    }
    const int mtiles = BM >= block_size ? 1 : block_size / BM;     // (a tile of several blocks: one a tile)
    const long n_gemm = (long)(batch / BN) * ((long)rows * block_size / BM) * split;
    TORCH_CHECK(n_gemm < (1L << 31), "lowrank stream: a grid past 2^31 blocks");
    launch_pdl(kernel, dim3((unsigned)n_gemm), dim3(NTH), SMEM, st, params, pbase, u, ugrp, part, knt, mtiles, block_size,
               batch, num_edges, split, (long)rows * block_size, node_mars, mp, nids, cmax, ncm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// id -> {BM, BN, WM, WN, PW, ST, KT}: the BlockScale forward stream's table (MEASURED there: the 128 x 128 tiles take
// B >= 128 at most shapes -- 64-row k-steps in 3 stages, or 32-row ones in 6 --, 128 x 64 B 64 and the 512-node blocks,
// 64 x 128 the 256-node blocks at B 512, 64 x 64 the 64-node blocks at a batch that is no multiple of 128); 256 x 128
// (16 consumer warps in two column halves under the 96-register cap, the u tile read once per 256 rows) for blocks of
// 256 and more -- MEASURED HMM-4096: B512 96.6 -> 90.5 us a layer (split 2), B128 loses (26 vs 19 us GEMM: a CTA's
// MMA/ingest overlap, not its bytes, paces it); the tuner picks
#define CFG_TABLE(X)                     \
    X(0, 128, 128, 4, 2, 4, 3, 64)       \
    X(1, 128, 128, 2, 4, 4, 6, 32)       \
    X(2, 128, 64, 4, 2, 4, 4, 64)        \
    X(3, 64, 128, 2, 4, 4, 4, 64)        \
    X(4, 128, 64, 4, 2, 4, 8, 32)        \
    X(5, 64, 64, 2, 2, 4, 6, 64)         \
    X(6, 256, 128, 8, 2, 4, 4, 32)       \
    X(7, 256, 128, 8, 2, 4, 2, 64)

std::vector<std::vector<int64_t>> configs() {
    std::vector<std::vector<int64_t>> out;
#define X(id, bm, bn, wm, wn, pw, st, kt) out.push_back({bm, bn, kt});
    CFG_TABLE(X)
#undef X
    return out;
}

// The config ids that serve a layer: its tile divides the node block -- or, `grouped` (the caller verified that every
// tile's consecutive blocks share a group), the node block divides the tile -- and it divides the batch.
std::vector<int64_t> fitting_configs(int64_t block_size, int64_t batch, bool grouped) {
    std::vector<int64_t> out;
    if (batch % 64 != 0 || block_size % 32 != 0) return out;
    const auto cfgs = configs();
    for (int i = 0; i < (int)cfgs.size(); i++) {
        const int64_t bm = cfgs[i][0];
        const bool tile_ok = block_size % bm == 0 || (grouped && bm % block_size == 0);
        if (tile_ok && batch % cfgs[i][1] == 0) out.push_back(i);
    }
    return out;
}

// The sum_exp forward over the virtual batch into `vnm` [rows, B K] (the twin's node rows): the tables the stream
// kernel's (`nids` [rows] the twin's, `ebase` / `pbase` [rows, knt], `ugrp` int32 [rows], `grep` int32 [groups]);
// `vals` the element values [*, B] -- or, `cbase` given (int32 [rows, knt, 2, 2], `SumLayer._stream_fusion_cbase`),
// the NODE values, each element the sum of its product's two children; `jpos` int32 [groups, edges]: the child of the
// node vector at each edge position of a group's rows (-1: none); `fb` the factors [K, N, B]; `shift` the normalizers'
// row shifts (copy, sample, rank), this copy's from `row0`; `params` bf16; `cfg` = config id * 16 + split.
void lowrank_forward_stream(torch::Tensor vnm, torch::Tensor vals, c10::optional<torch::Tensor> cbase,
                            torch::Tensor ebase, torch::Tensor jpos, torch::Tensor fb, torch::Tensor shift,
                            int64_t row0, torch::Tensor params, torch::Tensor nids, torch::Tensor pbase, torch::Tensor ugrp,
                            torch::Tensor grep, torch::Tensor cmaxs, int64_t block_size, int64_t knt_, int64_t cfg) {
    const int knt = (int)knt_, rows = (int)nids.size(0), ngrp = (int)grep.size(0);
    const int K = (int)fb.size(0), N = (int)fb.size(1), B = (int)fb.size(2), cols = (int)vnm.size(1);
    const int num_edges = knt * BK;
    const int cfg_id = (int)(cfg / 16), split = (int)(cfg % 16);
    const auto cfgs = configs();
    TORCH_CHECK(cfg_id >= 0 && cfg_id < (int)cfgs.size(), "lowrank stream: no config ", cfg_id);
    const int BMc = (int)cfgs[cfg_id][0], BNc = (int)cfgs[cfg_id][1], kt = (int)cfgs[cfg_id][2];
    const int ksteps = (int)(num_edges / kt), kt_sh = kt == 64 ? 6 : 5;
    TORCH_CHECK(params.scalar_type() == torch::kBFloat16, "lowrank stream: bf16 parameters");
    TORCH_CHECK(K >= 1 && K <= KMAX && cols == B * K, "lowrank stream: rank 1 .. ", KMAX, ", vnm [rows, B K]");
    TORCH_CHECK(cols % 64 == 0 && block_size % 32 == 0 && cols % BNc == 0 && B % 16 == 0 &&
                (block_size % BMc == 0 || (BMc % block_size == 0 && ((long)rows * block_size) % BMc == 0)),
                "lowrank stream: B K % 64, B % 16, block_size % 32, the config's tile dividing the node block (or the "
                "block the tile, whole tiles of blocks) and the columns");
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= ksteps, "lowrank stream: split 1 .. ", MAX_SPLIT,
                " and at most the k-steps (", ksteps, "); got ", split);
    TORCH_CHECK(vnm.is_contiguous() && vals.is_contiguous() && nids.is_contiguous() && pbase.is_contiguous() &&
                ebase.is_contiguous() && ugrp.is_contiguous() && grep.is_contiguous() && fb.is_contiguous() &&
                jpos.is_contiguous(), "lowrank stream: contiguous tensors");
    TORCH_CHECK(vnm.scalar_type() == torch::kFloat32 && vals.scalar_type() == torch::kFloat32 &&
                fb.scalar_type() == torch::kFloat32 && jpos.scalar_type() == torch::kInt32 &&
                ebase.scalar_type() == torch::kInt64 && pbase.scalar_type() == torch::kInt64 &&
                nids.scalar_type() == torch::kInt64 && ugrp.scalar_type() == torch::kInt32 &&
                grep.scalar_type() == torch::kInt32, "lowrank stream: dtypes");
    TORCH_CHECK(ebase.numel() == (long)rows * knt && pbase.numel() == ebase.numel() && ugrp.numel() == rows && ngrp >= 1,
                "lowrank stream: ebase / pbase one entry per (node block, edge tile); ugrp [rows]");
    TORCH_CHECK(vals.size(1) == B, "lowrank stream: the values' columns are the batch");
    TORCH_CHECK(jpos.numel() == (long)ngrp * num_edges, "lowrank stream: jpos int32 [groups, edges]");
    TORCH_CHECK(cmaxs.is_contiguous() && cmaxs.scalar_type() == torch::kFloat32 && cmaxs.numel() >= (long)ngrp * split * B,
                "lowrank stream: the maxima float32 [groups, split, B] at least, -inf between calls");
    TORCH_CHECK(shift.is_contiguous() && shift.scalar_type() == torch::kFloat32 && shift.numel() >= row0 + (long)K * B,
                "lowrank stream: the normalizers' shifts (copy, sample, rank) from row0");
    const bool fused = cbase.has_value();
    if (fused) {
        TORCH_CHECK(cbase->is_contiguous() && cbase->scalar_type() == torch::kInt32 &&
                    cbase->numel() == (long)rows * knt * 4, "lowrank stream: cbase int32 [rows, knt, 2, 2]");
    }
    const at::cuda::OptionalCUDAGuard guard(vnm.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    // scratch, ONE allocation: u bf16 [groups, num_edges, cols], Mp [groups, split, cols], the partials
    // [split, rows * bs, cols] (every region 16-byte aligned: cols % 64 == 0)
    const long o_mp = (long)ngrp * num_edges * cols * 2;
    const long o_part = o_mp + (long)ngrp * split * cols * 4;
    const long total = o_part + (split > 1 ? (long)split * rows * block_size * cols * 4 : 0);
    torch::Tensor scratch = torch::empty({total}, vnm.options().dtype(torch::kUInt8));
    char* base = reinterpret_cast<char*>(scratch.data_ptr());
    auto* u = reinterpret_cast<__nv_bfloat16*>(base);
    auto* mp = reinterpret_cast<float*>(base + o_mp);
    auto* part = reinterpret_cast<float*>(base + o_part);
    float* cmax = cmaxs.data_ptr<float>();
    const int ncm = ngrp * split * B;
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    const float* shp = shift.data_ptr<float>();
#define PREPASS(ROWS, PB, F)                                                                                           \
    {                                                                                                                  \
        /* a range is whole k-steps: the longest ceil(ksteps / split) of them */                                     \
        const int nchunk = (int)(((((long)ksteps + split - 1) / split) << kt_sh) + ROWS - 1) / ROWS;                   \
        const long units = (long)ngrp * split * (B / PB) * nchunk;                                                     \
        TORCH_CHECK(units < (1L << 31), "lowrank stream: a pre-pass past 2^31 units");                                 \
        const int* cb = F ? cbase->data_ptr<int>() : (const int*)nullptr;                                              \
        launch_pdl(lr_cmax_kernel<ROWS, PB, F>, dim3((unsigned)units), dim3(ROWS * PB), 0, st, vals.data_ptr<float>(), \
                   ebase.data_ptr<long>(), cb, grep.data_ptr<int>(), jpos.data_ptr<int>(), B, cmax, num_edges, knt,    \
                   ksteps, kt_sh, split, nchunk);                                                                      \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                \
        launch_pdl(lr_exp_kernel<ROWS, PB, F>, dim3((unsigned)units), dim3(ROWS * PB), 0, st, vals.data_ptr<float>(),  \
                   ebase.data_ptr<long>(), cb, grep.data_ptr<int>(), jpos.data_ptr<int>(), fb.data_ptr<float>(), N, B, \
                   K, u, (const float*)cmax, shp, (long)row0, mp, cols, num_edges, knt, ksteps, kt_sh, split, nchunk); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                \
    }
    if (B % 32 == 0) {                                       // a warp one row's 32 samples
        if (fused) PREPASS(4, 32, true) else PREPASS(4, 32, false)
    } else {                                                 // (B 16, 48, ...: two rows a warp)
        if (fused) PREPASS(8, 16, true) else PREPASS(8, 16, false)
    }
#undef PREPASS
    switch (cfg_id) {
#define X(id, bm, bn, wm, wn, pw, stg, ktt) \
        case id: launch_gemm<bm, bn, wm, wn, pw, stg, ktt>(pp, pbase.data_ptr<long>(), (const bfloat16_t*)u, \
                     ugrp.data_ptr<int>(), part, knt, rows, (int)block_size, cols, num_edges, split, \
                     vnm.data_ptr<float>(), mp, nids.data_ptr<long>(), cmax, ncm, st); break;
        CFG_TABLE(X)
#undef X
    }
    if (split > 1) {
        const long rt = (long)rows * block_size, n = rt * (cols / 4);
        const unsigned grid = (unsigned)((n + 255) / 256);
        float* nm = vnm.data_ptr<float>();
        const long* np = nids.data_ptr<long>();
        const int* ug = ugrp.data_ptr<int>();
        switch (split) {
#define M_(S) case S: launch_pdl(lr_merge_kernel<S>, dim3(grid), dim3(256), 0, st, nm, part, mp, np, ug, cols, \
                                 (int)block_size, rt); break;
            M_(2) M_(3) M_(4) M_(5) M_(6) M_(7) M_(8) M_(9) M_(10) M_(11) M_(12)
#undef M_
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// ---------------- the online-max form's launch ----------------
// A's tensor map, encoded once per (the parameters' address, the slabs' geometry): a pure function of those, whatever
// the parameters hold. A few kept, the latest first (a circuit's parameter copies are few).
static CUtensorMap om_tmap(const void* base, long block_size, long knt, long rows, long bstr) {
    struct Entry { const void* base; long block_size, knt, rows, bstr; CUtensorMap map; };
    static std::vector<Entry> kept;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    for (size_t i = 0; i < kept.size(); i++) {
        const Entry& e = kept[i];
        if (e.base == base && e.block_size == block_size && e.knt == knt && e.rows == rows && e.bstr == bstr) {
            const CUtensorMap m = e.map;
            if (i > 0) { const Entry hit = kept[i]; kept.erase(kept.begin() + i); kept.insert(kept.begin(), hit); }
            return m;
        }
    }
    Entry e{base, block_size, knt, rows, bstr, {}};
    // 32 nodes (contiguous) x the k rows of every edge tile (a node block's slab, `block_size` apart) x the node blocks
    // (`bstr` apart); a box the tile's 32 nodes x KT k x 4 blocks, the 64-byte swizzle CuTe's AtomA32 reads
    cuuint64_t gdim[3] = {(cuuint64_t)block_size, (cuuint64_t)knt * BK, (cuuint64_t)rows};
    cuuint64_t gstride[2] = {(cuuint64_t)block_size * 2, (cuuint64_t)bstr * 2};
    cuuint32_t box[3] = {32, 32, (cuuint32_t)(OM_BM / 32)};
    cuuint32_t es[3] = {1, 1, 1};
    const CUresult rr = cuTensorMapEncodeTiled(&e.map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), gdim,
                                               gstride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rr == CUDA_SUCCESS, "lowrank om: cuTensorMapEncodeTiled failed (", (int)rr, ")");
    kept.insert(kept.begin(), e);
    if (kept.size() > 8) kept.pop_back();
    return e.map;
}

template <bool FUSED, bool TMAA>
static void launch_om(const bfloat16_t* params, const long* pbase, const bfloat16_t* vt, long vrows, long row0,
                      const float* shift, const float* vals, const long* ebase, const int* cbase, const int* jpos,
                      const int* ugrp, int knt, int rows, int block_size, int B, float* vnm, const long* nids,
                      int num_edges, const CUtensorMap& tmapA, const float* fa, float* node_mars, float* lz,
                      long nshift, long vn0, cudaStream_t st) {
    constexpr int ST = OM_ST, S = OM_BN / OM_K;
    constexpr int STAGE = (32 * OM_BM * 2 + 2 * 32 * OM_BN * 2 + 2 * 32 * S * 4 + 1023) / 1024 * 1024;
    constexpr int SMEM = ST * STAGE + OM_BN * 4 + ST * S * 4 + 3 * ST * 8;
    static_assert(SMEM <= 101376, "past sm_120's 99 KB of shared memory a block");
    auto kernel = lr_om_kernel<ST, FUSED, TMAA>;
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if (dev < 64) attr_set[dev] = true;
    }
    const int mtiles = OM_BM >= block_size ? 1 : block_size / OM_BM;
    const long n = (long)(OM_K * B / OM_BN) * ((long)rows * block_size / OM_BM);
    TORCH_CHECK(n < (1L << 31), "lowrank om: a grid past 2^31 blocks");
    launch_pdl(kernel, dim3((unsigned)n), dim3(OM_THREADS), SMEM, st, params, pbase, vt, vrows, row0, shift, vals, ebase,
               cbase, jpos, ugrp, knt, mtiles, block_size, B, vnm, nids, num_edges, tmapA, fa, node_mars, lz, nshift,
               vn0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Whether the online-max form serves a layer: rank OM_K, B a multiple of the tile's samples, a node tile of whole
// node blocks or a node block of whole tiles (the stream kernel's tables), the children in edge tiles of 64.
bool om_serves(int64_t K, int64_t B, int64_t block_size, int64_t rows) {
    return K == OM_K && B % (OM_BN / OM_K) == 0 && block_size % 32 == 0 &&
           (block_size % OM_BM == 0 || (OM_BM % block_size == 0 && (rows * block_size) % OM_BM == 0));
}

// The sum_exp forward of a layer over the virtual batch in ONE launch (the online-max form), its normalizer (Z_r =
// theta V under V's shifts) and collapse beside: `vnm` [twin rows, B K] the layer's kept share differences (what the
// backward's dA takes); the tables `lowrank_forward_stream`'s; `vt` the normalizers' V bf16 [N, rows of V]
// (`lowrank_z_operand`'s, the rows (copy, sample, rank)), `shift` their row shifts, this copy's rows from `row0`;
// `node_mars` the node values at the node vector's circuit rows (the twin's + `nshift`), `lz` [M, B] log Z, `fa` the
// factors A [M, K, B]; `vn0` the node vector's first twin row. `a_p0` / `a_bstr`: the
// parameter slabs' bases `a_p0 + block a_bstr + edge tile 64 block_size` (`a_p0` -1: not so, or blocks not of 32 -- A by
// cp.async; else by TMA), checked by the caller once per table.
void lowrank_forward_om(torch::Tensor vnm, torch::Tensor vals, c10::optional<torch::Tensor> cbase, torch::Tensor ebase,
                        torch::Tensor jpos, torch::Tensor vt, int64_t row0, torch::Tensor shift, torch::Tensor params,
                        torch::Tensor nids, torch::Tensor pbase, torch::Tensor ugrp, int64_t block_size, int64_t knt_,
                        int64_t a_p0, int64_t a_bstr, torch::Tensor node_mars, torch::Tensor lz, torch::Tensor fa,
                        int64_t nshift, int64_t vn0) {
    const int knt = (int)knt_, rows = (int)nids.size(0);
    const int B = (int)vals.size(1), cols = (int)vnm.size(1);
    const int num_edges = knt * BK;
    TORCH_CHECK(om_serves(cols / std::max(B, 1), B, block_size, rows) && cols == OM_K * B && knt >= 1,
                "lowrank om: rank ", OM_K, ", B a multiple of ", OM_BN / OM_K, ", block_size % 32 with the ", OM_BM,
                "-node tile dividing the node block or the blocks dividing the tile; got columns ", cols, ", B ", B,
                ", block_size ", block_size);
    TORCH_CHECK(params.scalar_type() == torch::kBFloat16 && vt.scalar_type() == torch::kBFloat16,
                "lowrank om: bf16 parameters and V");
    TORCH_CHECK(vnm.is_contiguous() && vals.is_contiguous() && nids.is_contiguous() && pbase.is_contiguous() &&
                ebase.is_contiguous() && ugrp.is_contiguous() && jpos.is_contiguous() && vt.is_contiguous() &&
                shift.is_contiguous(), "lowrank om: contiguous tensors");
    TORCH_CHECK(vnm.scalar_type() == torch::kFloat32 && vals.scalar_type() == torch::kFloat32 &&
                shift.scalar_type() == torch::kFloat32 && jpos.scalar_type() == torch::kInt32 &&
                ebase.scalar_type() == torch::kInt64 && pbase.scalar_type() == torch::kInt64 &&
                nids.scalar_type() == torch::kInt64 && ugrp.scalar_type() == torch::kInt32, "lowrank om: dtypes");
    TORCH_CHECK(ebase.numel() == (long)rows * knt && pbase.numel() == ebase.numel() && ugrp.numel() == rows,
                "lowrank om: ebase / pbase one entry per (node block, edge tile); ugrp [rows]");
    TORCH_CHECK(jpos.dim() == 2 && jpos.size(1) == num_edges, "lowrank om: jpos int32 [groups, edges]");
    TORCH_CHECK(vt.dim() == 2 && row0 >= 0 && row0 + cols <= vt.size(1) && shift.numel() == vt.size(1),
                "lowrank om: V [N, rows] and the shifts [rows] holding this copy's rows from row0");
    const bool fused = cbase.has_value();
    if (fused)
        TORCH_CHECK(cbase->is_contiguous() && cbase->scalar_type() == torch::kInt32 &&
                    cbase->numel() == (long)rows * knt * 4, "lowrank om: cbase int32 [rows, knt, 2, 2]");
    const long M = lz.size(0);
    for (const torch::Tensor* t : {&node_mars, &lz, &fa})
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat32,
                    "lowrank om: the collapse's tensors contiguous float32 CUDA");
    TORCH_CHECK(lz.dim() == 2 && lz.size(1) == B && fa.numel() == M * OM_K * B && node_mars.dim() == 2 &&
                node_mars.size(1) == B && vn0 >= 0 && node_mars.size(0) >= vn0 + nshift + M && vnm.size(0) >= vn0 + M &&
                (reinterpret_cast<uintptr_t>(vnm.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(shift.data_ptr()) & 15) == 0,
                "lowrank om: lz [M, B], A [M, K, B], the node rows, 16-byte aligned runs");
    const bool tma = a_p0 >= 0 && block_size == 32;
    TORCH_CHECK(!tma || (a_bstr % 8 == 0 && a_p0 % 8 == 0), "lowrank om: the slabs' bases 16-byte aligned");
    const at::cuda::OptionalCUDAGuard guard(vnm.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    const CUtensorMap tm = tma ? om_tmap((const char*)pp + a_p0 * 2, block_size, knt, rows, a_bstr) : CUtensorMap{};
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    const auto* vtp = reinterpret_cast<const bfloat16_t*>(vt.data_ptr());
    const int* cb = fused ? cbase->data_ptr<int>() : nullptr;
#define LOWRANK_OM(F, T)                                                                                              \
    launch_om<F, T>(pp, pbase.data_ptr<long>(), vtp, (long)vt.size(1), (long)row0, shift.data_ptr<float>(),         \
                    vals.data_ptr<float>(), ebase.data_ptr<long>(), cb, jpos.data_ptr<int>(), ugrp.data_ptr<int>(),     \
                    knt, rows, (int)block_size, B, vnm.data_ptr<float>(), nids.data_ptr<long>(), num_edges, tm,      \
                    fa.data_ptr<float>(), node_mars.data_ptr<float>(), lz.data_ptr<float>(), (long)nshift, (long)vn0, st)
    if (fused) { if (tma) LOWRANK_OM(true, true); else LOWRANK_OM(true, false); }
    else { if (tma) LOWRANK_OM(false, true); else LOWRANK_OM(false, false); }
#undef LOWRANK_OM
}

// ---------------- the backward's one-launch form ----------------
// Whether it serves a layer's backward partition: rank OM_K, B a multiple of the tile's samples, the child tile of
// whole child blocks or inside one, the parents' blocks a multiple of the k-step
// The dA CTAs a launch at most (0: as many as the pairs and the free SMs take): the tests make them few, so a CTA's
// threads stride over the pairs more than once (the sizes they run at give a thread one pair otherwise)
static int g_da_ctas_most = 0;
void set_da_ctas_most(int64_t n) { g_da_ctas_most = (int)std::max<int64_t>(n, 0); }

bool om_bw_serves(int64_t K, int64_t B, int64_t cbs, int64_t pbs, int64_t n_cb) {
    return K == OM_K && B % (OM_BN / OM_K) == 0 && pbs % 32 == 0 && cbs > 0 &&
           (cbs % OM_BM == 0 || (OM_BM % cbs == 0 && (n_cb * cbs) % OM_BM == 0));
}

// The data half of a layer's backward over the virtual batch in ONE launch (`lr_om_bw_kernel`): `flows` the element
// flows [rows, B] -- or, `table` given (the product layer below fused in: int32 [products, 2], from element row `first`
// on), the NODE flows the products' two children take --, `vals` the element values or (`table`) the node values;
// `fb` the factors B [K, N, B], `jrow` int32 the node vector's child of each element row from `e_lo` on; `grad_b` dB
// [K, N, B] (None: no gradient); `ua` the pass's exp(A - shift) bf16 [M, rows] (`lowrank_a_operand`), `sha` its rows'
// shifts, this copy's from `row0`; the parents' `node_flows`, `node_mars` (circuit rows: the twin's + `nshift`) and
// `lz` [M, B] (node index: the twin's row - `vn0`); `params` bf16; the twin's backward partition: `chids` [blocks] the
// child blocks' first element rows (`cbs` rows a block), `ppar` / `ppid` int64 [blocks, parent blocks] each parent
// block's first twin row and parameter slab (`pbs` parents a block).
// `grad_a` given (with `grad_b`): dA too, in the grid's free SMs (`lr_om_bw_kernel`'s dA CTAs), from `kd` the share
// differences the forward kept [twin rows, B K]. `grad_b` given: dB's data half AND its normalizer half (the pass's end
// then takes nothing off this copy's dB).
void lowrank_backward_om(torch::Tensor flows, torch::Tensor vals, c10::optional<torch::Tensor> table, int64_t first,
                         torch::Tensor fb, torch::Tensor jrow, int64_t e_lo, c10::optional<torch::Tensor> grad_b,
                         torch::Tensor ua, int64_t row0, torch::Tensor sha, torch::Tensor node_flows,
                         torch::Tensor node_mars, torch::Tensor lz, int64_t nshift, int64_t vn0, torch::Tensor params,
                         torch::Tensor chids, torch::Tensor ppar, torch::Tensor ppid, int64_t cbs, int64_t pbs,
                         c10::optional<torch::Tensor> grad_a, c10::optional<torch::Tensor> kd) {
    const int B = (int)node_flows.size(1), K = (int)fb.size(0), N = (int)fb.size(1);
    const int n_cb = (int)chids.size(0), npb = (int)ppar.size(1);
    TORCH_CHECK(om_bw_serves(K, B, cbs, pbs, n_cb) && fb.size(2) == B, "lowrank om backward: rank ", OM_K, ", B a "
                "multiple of ", OM_BN / OM_K, ", the parents' blocks of 32, the ", OM_BM, "-child tile dividing the child "
                "block or the blocks dividing the tile; got rank ", K, ", B ", B, ", blocks ", cbs, " / ", pbs);
    TORCH_CHECK(params.scalar_type() == torch::kBFloat16 && ua.scalar_type() == torch::kBFloat16,
                "lowrank om backward: bf16 parameters and exp(A)");
    for (const torch::Tensor* t : {&flows, &vals, &fb, &jrow, &ua, &sha, &node_flows, &node_mars, &lz, &chids, &ppar,
                                   &ppid})
        TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "lowrank om backward: contiguous CUDA tensors");
    TORCH_CHECK(flows.scalar_type() == torch::kFloat32 && vals.scalar_type() == torch::kFloat32 &&
                fb.scalar_type() == torch::kFloat32 && jrow.scalar_type() == torch::kInt32 &&
                sha.scalar_type() == torch::kFloat32 && node_flows.scalar_type() == torch::kFloat32 &&
                node_mars.scalar_type() == torch::kFloat32 && lz.scalar_type() == torch::kFloat32 &&
                chids.scalar_type() == torch::kInt64 && ppar.scalar_type() == torch::kInt64 &&
                ppid.scalar_type() == torch::kInt64, "lowrank om backward: dtypes");
    TORCH_CHECK(ppar.dim() == 2 && ppar.size(0) == n_cb && ppid.sizes() == ppar.sizes(),
                "lowrank om backward: ppar / ppid [child blocks, parent blocks]");
    TORCH_CHECK(ua.dim() == 2 && row0 >= 0 && row0 + (long)K * B <= ua.size(1) && sha.numel() == ua.size(1) &&
                ua.size(1) % 8 == 0 && lz.size(1) == B && vals.size(1) == B && flows.size(1) == B,
                "lowrank om backward: exp(A) [M, rows] and its shifts holding this copy's rows from row0");
    const bool fused = table.has_value();
    if (fused)
        TORCH_CHECK(table->is_cuda() && table->is_contiguous() && table->scalar_type() == torch::kInt32 &&
                    table->dim() == 2 && table->size(1) == 2, "lowrank om backward: the products' table int32 [*, 2]");
    if (grad_b.has_value())
        TORCH_CHECK(grad_b->is_cuda() && grad_b->is_contiguous() && grad_b->scalar_type() == torch::kFloat32 &&
                    grad_b->numel() == fb.numel(), "lowrank om backward: dB as B");
    const bool da = grad_a.has_value();
    TORCH_CHECK(da == grad_b.has_value() && da == kd.has_value(), "lowrank om backward: dA's tensors with dB, all or none");
    const int M = (int)lz.size(0);
    if (da) {
        for (const torch::Tensor* t : {&*grad_a, &*kd})
            TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat32 &&
                        (reinterpret_cast<uintptr_t>(t->data_ptr()) & 15) == 0,
                        "lowrank om backward: dA's tensors contiguous float32 CUDA, 16-byte aligned");
        TORCH_CHECK(grad_a->numel() == (long)M * K * B && kd->dim() == 2 && kd->size(1) == (long)K * B &&
                    kd->size(0) >= vn0 + M, "lowrank om backward: dA's tensors do not fit the node vector");
    }
    const at::cuda::OptionalCUDAGuard guard(flows.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    constexpr int ST = OM_ST, S = OM_BN / OM_K;
    const int mtiles = OM_BM >= cbs ? 1 : (int)(cbs / OM_BM);
    const long n = (long)(OM_K * B / OM_BN) * ((long)n_cb * cbs / OM_BM);
    TORCH_CHECK(n < (1L << 31), "lowrank om backward: a grid past 2^31 blocks");
    if (n == 0) return;
    // the dA CTAs: the free SMs of the GEMM's last wave (a CTA an SM), or a wave of their own where few are free
    int sms = 0;
    {
        int dv = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dv));
        static int sms_of[64] = {};
        if (dv >= 64 || sms_of[dv] == 0) {
            C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dv));
            if (dv < 64) sms_of[dv] = sms;
        } else {
            sms = sms_of[dv];
        }
    }
    const int free_sms = (int)((sms - n % sms) % sms);
    const long pairs = (long)M * B;
    int n_da = !da ? 0 : (int)std::min<long>((pairs + OM_THREADS - 1) / OM_THREADS,
                                             free_sms >= sms / 4 ? free_sms : sms);
    if (n_da > 0 && g_da_ctas_most > 0) n_da = std::min(n_da, g_da_ctas_most);
    auto go = [&](auto f, auto g) {
        constexpr bool F = decltype(f)::value, G = decltype(g)::value;
        constexpr int NU = G ? 2 : 1;                    // (as the kernel's: U; and Un with the gradients)
        constexpr int STAGE = (32 * OM_BM * 2 + NU * 32 * OM_BN * 2 + 3 * 32 * S * 4 + 1023) / 1024 * 1024;
        constexpr int SMEM = ST * STAGE + NU * OM_BN * 4 + NU * ST * S * 4 + 3 * ST * 8;
        static_assert(SMEM <= 101376, "past sm_120's 99 KB of shared memory a block");
        auto kernel = lr_om_bw_kernel<ST, F, G>;
        static bool attr_set[64] = {};
        int dev = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        if (dev >= 64 || !attr_set[dev]) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            if (dev < 64) attr_set[dev] = true;
        }
        launch_pdl(kernel, dim3((unsigned)(n + n_da)), dim3(OM_THREADS), SMEM, st, pp, ppid.data_ptr<long>(),
                   ppar.data_ptr<long>(), reinterpret_cast<const bfloat16_t*>(ua.data_ptr()), (long)ua.size(1),
                   (long)row0, sha.data_ptr<float>(), node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
                   lz.data_ptr<float>(), (long)nshift, (long)vn0, chids.data_ptr<long>(), npb, (int)cbs, (int)pbs,
                   mtiles, B, vals.data_ptr<float>(), fused ? table->data_ptr<int>() : (const int*)nullptr,
                   (long)first, fb.data_ptr<float>(), jrow.data_ptr<int>(), (long)e_lo, N, flows.data_ptr<float>(),
                   grad_b.has_value() ? grad_b->data_ptr<float>() : (float*)nullptr,
                   da ? kd->data_ptr<float>() : (const float*)nullptr, da ? (long)kd->size(1) : 0L,
                   da ? grad_a->data_ptr<float>() : (float*)nullptr, M, (int)n, n_da);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (fused) { if (grad_b.has_value()) go(std::true_type{}, std::true_type{}); else go(std::true_type{}, std::false_type{}); }
    else { if (grad_b.has_value()) go(std::false_type{}, std::true_type{}); else go(std::false_type{}, std::false_type{}); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_forward_stream", &lowrank_forward_stream,
          "the sum_exp forward over the virtual batch: the expand fused into the stream kernel's pre-pass");
    m.def("lowrank_forward_om", &lowrank_forward_om,
          "the sum_exp forward over the virtual batch in one launch: the operand formed in the GEMM, the maxima online");
    m.def("om_serves", &om_serves, "whether the one-launch form serves a layer (rank, batch, block size, node blocks)");
    m.def("lowrank_backward_om", &lowrank_backward_om,
          "the data half of the sum_exp backward over the virtual batch in one launch: element flows and dB's data part");
    m.def("set_da_ctas_most", &set_da_ctas_most, "(tests) the backward's dA CTAs a launch at most; 0: no cap");
    m.def("om_bw_serves", &om_bw_serves,
          "whether the backward's one-launch form serves a partition (rank, batch, child and parent block sizes, blocks)");
    m.def("configs", &configs, "(BM, BN, KT) per config id (the stream forward's)");
    m.def("fitting_configs", &fitting_configs, "the config ids that serve a layer (block_size, columns, grouped tiles)");
}
