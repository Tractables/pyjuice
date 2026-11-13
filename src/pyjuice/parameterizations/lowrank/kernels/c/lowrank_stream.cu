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

// A float's order as a signed int (`fkey`): a non-negative float its bits, a negative one its bits with all but the
// sign flipped -- the keys' integer maximum (one `redux` across a warp) is the key of the floats' maximum
__device__ __forceinline__ int fkey(float x) { const int b = __float_as_int(x); return b ^ ((b >> 31) & 0x7fffffff); }
__device__ __forceinline__ float fkey_inv(int k) { return __int_as_float(k ^ ((k >> 31) & 0x7fffffff)); }
// The one-launch kernels' tile `blk` of the grid's `nbt` sample tiles x `nmt` node (child) tiles, in GROUPS of `nbg`
// sample tiles, node-tile-major inside a group: the tiles of one sample tile -- the readers of its operand rows (V and
// the values; Ua and the flows) -- are `nbg` blocks apart, so the rows the first of them drew from DRAM are still in
// the L2 for the others (in the grid's plain order they are `nbt` apart: 512 at a batch of 2048). `nbg >= nbt`: one
// group, the plain order.
__device__ __forceinline__ void tile_of(int blk, int nbt, int nmt, int nbg, int& pid_b, int& pid_m) {
    if (nbg >= nbt) { pid_b = blk % nbt; pid_m = blk / nbt; return; }
    const int per = nbg * nmt, g = blk / per, rem = blk - g * per;
    const int cur = min(nbg, nbt - g * nbg);             // (the last group: the rest)
    pid_m = rem / cur; pid_b = g * nbg + rem % cur;
}
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
__device__ __forceinline__ void tma_load_2d(uint32_t smem, const CUtensorMap* desc, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(smem), "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t smem, const CUtensorMap* desc, int c0, int c1, int c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
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
// collapse takes `log(sum) + m_b + shift` (the sample's final max; shift V's row's). A row of no child (`jpos` -1)
// copies -inf for its value (its V row: zeros). `uaw` given: the collapse writes the one-launch backward's operand
// besides, from the A it reads anyway -- `Ua = exp(A - am)` [M, urows] (this copy's rows from row0, as V's) and `am`
// (`amw`: this copy's [M, B]) each (node, sample)'s largest A over its ranks (`lowrank_mixture.cu`'s
// `a_operand_kernel`, its operations): the pass then makes it in no launch of its own.
//
// PERSISTENT, THE RING CONTINUOUS, THE COLLAPSE IN THE CONSUMERS' REGISTERS: a CTA an SM takes the tiles blockIdx.x -
// n_v, + nct, ... (in `tile_of`'s order); its ring runs on across its tiles (the CTA's k-step g in stage g % ST), so
// the producers and converters never stop at a tile's end; the consumers collapse a tile from their accumulators --
// the stage's columns permuted so that a warp holds whole samples -- and go on to the next: no shared-memory round
// trip, no barrier of every role, no stage drained. MEASURED (toast, HMM 1024 x 31 in blocks of 32, rank 16, B1024, us
// a layer, every variant's MMAs counted in its SASS): the tensor floor 156; the consumers alone 163 (no copies, no
// waits); + the ring's waits 173, + the conversion 182, + the copies 208 (A +3, V +9, the values +5 alone), + A's
// prefetch for the collapse 218; the collapse through shared memory by every role, a tile a CTA (before) 283; this 265
// (its loads 10, stores 19, FP32 11, exponentials 6). The kernel at B16 / 64 / 256 / 1024 / 2048 (ms, 31 layers): 0.684
// / 0.787 / 2.42 / 8.85 / 19.3 before, the grids' groups alone 0.69 / 0.80 / 2.49 / 8.92 / 17.6, this 0.65 / 0.79 /
// 2.35 / 8.23 / 16.2. Not taken (each measured): the collapse by two warps of their own beside the k-steps, the
// consumers' raw sums through a scratch slot in global memory (B1024 258 us a layer, B64 +28%: FP32 work beside the
// MMAs costs them as much on the same SM); a waiting warp asleep (`try_wait` with a suspend-time hint: no change).
//
// MEASURED (croissant, one GPU no one else used, HMM 1024 x 1024 in blocks of 32, rank 16, batch 64, the layer's
// launches replayed in a CUDA graph, us a call; `knt` edge tiles of the layer's 16): the pre-pass chain 9.2 at one edge
// tile, 21.2 at 16 (cmax + exp + GEMM); this 4.2 / 16.6-16.9. Its main loop is the L2's: the A stream alone (TMA)
// reaches ~5.1 TB/s of the ~6.25 a 128-SM grid can draw. Not taken (each measured): the samples' maxima first, by a
// launch of their own (8.8 / 21.4) or by the GEMM's converters walking the range (7.1 / 22.0); 6 stages (no gain);
// V by TMA (16.7 -> 20.7: a 64-column box a k-step half); two-CTA clusters of 128 x 128 tiles summing two k-ranges
// (18.8-19.6: sm_120's distributed shared memory moves 32 KB a CTA in 4-14 us, so the halves went through L2;
// `setmaxnreg` is ignored in cluster kernels); TMA multicast (ptxas: "substantially reduced performance" on sm_120a).
//
// V BY SIDE CTAS (`vnext`; `virtual.PIPELINED_V`): the grid's first n_v CTAs form ANOTHER copy's V rows and shifts --
// the next copy the pass visits -- beside the tiles (`om_v_rows`), instead of every copy's by a launch of their own
// before the pass; n_v by the launch's shape (`virtual._om_vctas`: 16 where the tiles take at most 12 rounds of the
// SMs, else 20; none where V's units would take more than twice the tiles' rounds). The side CTAs come FIRST: last in
// the grid they took the SMs the launch before freed last and ran ~1.5x slower (B2048, 24 of them: the forward 18.8 ->
// 28.5 ms). A side CTA writes a child's run of a sample's ranks at once, 32 bytes (with 8 bytes a child a unit,
// scattered, the step at B256 / B1024 / B2048 -2.7 / -2.6 / -2.0%). Not taken: an L2 prefetch of the units 8 ahead (B64
// / B256 +2%, B1024 / B2048 0).
// MEASURED (croissant, HMM 1024 x 31 in blocks of 32, rank 16, step ms, 3 rounds, two GPUs no one else used, every
// copy's V upfront -> by side CTAs): B2048 55.29 -> 52.98 (-4.2%), B1024 27.51 -> 26.59 (-3.3%), B256 7.88 -> 7.41
// (-5.9%), B64 2.571 -> 2.505 (-2.6%), the other GPU -3.8 / -3.3 / -6.6 / -2.6%; at B1024 the forward's launches
// 8.21 -> 9.35 ms, V's 2.12 -> 0.08 (the pass's first copy).
__device__ __align__(16) float g_ninf[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};   // (a row of no child)


// The tile: OM_BM nodes x OM_BN columns (OM_BN / OM_K samples, every rank), KT = 32 children a k-step in ST stages;
// 8 consumer warps (4 x 2), 4 producer warps, 4 converter warps. TMAA: A by TMA (`tmapA`: 32 nodes x 32 k x the tile's
// 4 node blocks of 32, the slabs affine in (block, edge tile)), else cp.async. VTMA: every k-step's 32 children
// consecutive (`jpos`) and every run of values whole (no dummy, no row of no child) -- V's rows by TMA (`tmapV`: the
// k-step's children x the tile's 64 columns, the stage's 128-byte swizzle) and each run of the children's values
// (`tmapC`: 32 rows x the tile's 4 samples), one producer issuing them; else the producers gather both by cp.async.
constexpr int OM_BM = 128, OM_BN = 64, OM_K = 16, OM_ST = 5, OM_THREADS = 512;

// V's rows of one copy by the side CTAs of the one-launch forward (`lowrank_mixture.cu`'s `inplace_operand_kernel`,
// its operations -- the same bits: the maxima order-free, the same expf and rounding): the copy's factors `f` [B, K, X]
// in place (sample stride `f_sb`), V [X, rows] from `row0` (the rows (sample, rank): a sample's K ranks adjacent), `sh`
// the rows' shifts (each its rank's largest B). Units of OMV_G ranks of a sample -- OMV_G X floats, one bulk copy --,
// the side CTA vi's samples vi, + n_v, ...; OMV_NB units in flight in the shared memory (`omv_smem` bytes): a sample's.
// A thread keeps its children's (at most OMV_MAXX) bf16 values of the sample's ranks in registers and writes each
// child's run of them at once, 32 bytes (MEASURED, B1024, 16 side CTAs: the forward with 8 bytes a child a unit, its
// stores scattered, 16.4 ms; with no stores 9.1).
constexpr int OMV_G = 4, OMV_NB = 4, OMV_MAXX = 3, OMV_NW = OM_THREADS / 32;
// (the one-launch forward's shared memory: its stages, their records, the barriers -- what a side CTA has)
constexpr long OM_SMEM = (long)OM_ST * ((32 * OM_BM * 2 + 2 * 32 * OM_BN * 2 + 2 * 32 * (OM_BN / OM_K) * 4 + 1023) / 1024 *
                                       1024) + 2 * OM_ST * (OM_BN / OM_K) * 4 + 3 * OM_ST * 8;
__host__ __device__ constexpr long omv_smem(int X) { return (long)OMV_NB * OMV_G * X * 4 + 8 * OMV_NB + 4 * 64; }
template <int KC>
__device__ __forceinline__ void om_v_rows(const float* __restrict__ f0, long f_sb, bfloat16_t* __restrict__ out,
                                          float* __restrict__ sh, long rows, long row0, int X, int B, int vi,
                                          int n_v) {
    static_assert(KC % OMV_G == 0 && OMV_NW % OMV_G == 0, "a unit's ranks; its warps a rank");
    constexpr int NG = KC / OMV_G, WPR = OMV_NW / OMV_G;      // (units a sample; warps a rank)
    static_assert(OMV_G == 4 && OMV_NB == NG && KC % 8 == 0, "a unit 2 bf16 pairs; a sample's units the buffers");
    float* buf = (float*)smem_raw;                              // [NB][G X]
    uint64_t* bar = (uint64_t*)(smem_raw + (long)OMV_NB * OMV_G * X * 4);
    float* red = (float*)(bar + OMV_NB);                       // [G][WPR]
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int nsamp = (B - vi + n_v - 1) / n_v, nu = nsamp * NG;
    if (tid == 0) {
        for (int q = 0; q < OMV_NB; q++) mbar_init(&bar[q], 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const uint32_t bytes = (uint32_t)OMV_G * X * 4;
    auto issue = [&](int u) {                                   // (thread 0: unit u into buffer u % NB)
        const int q = u % OMV_NB, b = vi + n_v * (u / NG), g = u % NG;
        const uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[q]);
        const uint32_t da = (uint32_t)__cvta_generic_to_shared(buf + (long)q * OMV_G * X);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(ba), "r"(bytes) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     ::"r"(da), "l"(f0 + (long)b * f_sb + (long)g * OMV_G * X), "r"(bytes), "r"(ba) : "memory");
    };
    if (tid == 0)
        for (int u = 0; u < min(OMV_NB, nu); u++) issue(u);
    for (int s = 0; s < nsamp; s++) {
        const int b = vi + n_v * s;
        uint32_t acc[OMV_MAXX][KC / 2];                         // (the sample's ranks at the thread's children)
        #pragma unroll
        for (int g = 0; g < NG; g++) {                          // (unit s NG + g in buffer g)
            const int u = s * NG + g;
            mbar_wait(&bar[g], s & 1);
            const float* in = buf + (long)g * OMV_G * X;
            {                                                   // rank w / WPR's largest over its children
                const int r = w / WPR, p = w % WPR;
                float m = -INFINITY;
                for (int x = p * 32 + lane; x < X; x += 32 * WPR) m = fmaxf(m, in[(long)r * X + x]);
                for (int off = 16; off >= 1; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
                if (lane == 0) red[r * WPR + p] = m;
            }
            __syncthreads();
            float mr[OMV_G];
            #pragma unroll
            for (int r = 0; r < OMV_G; r++) {
                float m = red[r * WPR];
                for (int i = 1; i < WPR; i++) m = fmaxf(m, red[r * WPR + i]);
                mr[r] = m;
            }
            if (tid < OMV_G) {                                  // (its own rank's max again: no array indexed by tid)
                float m = red[tid * WPR];
                for (int i = 1; i < WPR; i++) m = fmaxf(m, red[tid * WPR + i]);
                sh[row0 + (long)b * KC + g * OMV_G + tid] = m;
            }
            #pragma unroll
            for (int i = 0; i < OMV_MAXX; i++) {
                const int x = tid + i * OM_THREADS;
                if (x < X) {
                    float v[OMV_G];
                    #pragma unroll
                    for (int r = 0; r < OMV_G; r++) {
                        const float y = in[(long)r * X + x];
                        v[r] = y > -INFINITY ? expf(y - mr[r]) : 0.0f;
                    }
                    const __nv_bfloat162 h0 = __floats2bfloat162_rn(v[0], v[1]), h1 = __floats2bfloat162_rn(v[2], v[3]);
                    acc[i][2 * g] = *reinterpret_cast<const uint32_t*>(&h0);
                    acc[i][2 * g + 1] = *reinterpret_cast<const uint32_t*>(&h1);
                }
            }
            __syncthreads();                                    // (buffer g and `red` free again)
            if (tid == 0 && u + OMV_NB < nu) issue(u + OMV_NB);
        }
        #pragma unroll
        for (int i = 0; i < OMV_MAXX; i++) {                    // (a child's run of the sample's ranks: 32 bytes)
            const int x = tid + i * OM_THREADS;
            if (x < X) {
                uint4* dst = reinterpret_cast<uint4*>(out + (long)x * rows + row0 + (long)b * KC);
                #pragma unroll
                for (int k = 0; k < KC / 8; k++)
                    dst[k] = make_uint4(acc[i][4 * k], acc[i][4 * k + 1], acc[i][4 * k + 2], acc[i][4 * k + 3]);
            }
        }
    }
}

// A copy's V rows alone (the pass's first copy, or one no side CTAs formed): the side CTAs' work on a grid of its own
__global__ void __launch_bounds__(OM_THREADS, 1) om_v_kernel(const float* __restrict__ f, long f_sb,
                                                             bfloat16_t* __restrict__ out, float* __restrict__ sh,
                                                             long rows, long row0, int X, int B) {
    om_v_rows<OM_K>(f, f_sb, out, sh, rows, row0, X, B, (int)blockIdx.x, (int)gridDim.x);
}

template <int ST, bool FUSED, bool TMAA, bool VTMA>
__global__ void __launch_bounds__(OM_THREADS, 1) lr_om_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ pbase, const bfloat16_t* __restrict__ vt,
        long vrows, long row0, const float* shift, const float* vals, const long* __restrict__ ebase,
        const int* __restrict__ cbase, const int* __restrict__ jpos, const int* __restrict__ ugrp, int knt, int mtiles,
        int block_size, int B, float* __restrict__ vnm, const long* __restrict__ nids, int num_edges,
        const __grid_constant__ CUtensorMap tmapA, const float* __restrict__ fa, long a_sb, float* node_mars,
        float* __restrict__ lz, long nshift, long vn0, int ntiles, int nbg, bfloat16_t* __restrict__ uaw, long urows,
        float* __restrict__ amw, const __grid_constant__ CUtensorMap tmapV, const __grid_constant__ CUtensorMap tmapC,
        const float* __restrict__ vfb, long vf_sb, bfloat16_t* __restrict__ vout, float* __restrict__ vsh, long vrow0,
        int vX, int n_v, int nct, float* __restrict__ cmo) {
    constexpr int BM = OM_BM, BN = OM_BN, K = OM_K, WM = 4, WN = 2;
    if ((int)blockIdx.x < n_v) {
        // ---------- the side CTAs (the grid's first): another copy's V rows (the launch before done first) ----------
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
        om_v_rows<OM_K>(vfb, vf_sb, vout, vsh, vrows, vrow0, vX, B, (int)blockIdx.x, n_v);
        return;
    }
    constexpr int KT = 32, NC = WM * WN * 32, NP = 128, NV = 128;
    static_assert(NC + NP + NV == OM_THREADS, "the warp roles");
    constexpr int S = BN / K;                            // the tile's samples
    static_assert(S % 4 == 0, "a row's values of the tile's samples in 16-byte runs");
    // a stage: A, V as copied (the normalizer's operand), u = V exp(c - m) (the converters'), the children's values
    constexpr int ABYTES = KT * BM * 2, UBYTES = KT * BN * 2, CBYTES = 2 * KT * S * 4;
    constexpr int STAGE = (ABYTES + 2 * UBYTES + CBYTES + 1023) / 1024 * 1024;
    constexpr int SUB = BK / KT;                         // k-steps an edge tile is
    const int cols = K * B;
    const int nbt = cols / BN;
    const int bpt = BM > block_size ? BM / block_size : 1;
    const int nk = knt * SUB;
    // PERSISTENT: a CTA past the side CTAs takes the tiles blockIdx.x - n_v, + nct, ... of the layer's `ntiles` (the
    // grid: a CTA an SM at most), in `tile_of`'s order, a node tile one of `mtiles` of a node block, or `bpt`
    // consecutive blocks of one group (blocks of 32 or 64).
    struct Tile { int r, tile_id, b0; };                 // (its node block / tile of it, its first sample)
    auto tile_at = [&](int t) {
        int pid_b, pid_m;                                // (the tiles' order: `tile_of`)
        tile_of(t, nbt, ntiles / nbt, nbg, pid_b, pid_m);
        Tile x;
        x.r = bpt > 1 ? pid_m * bpt : pid_m / mtiles; x.tile_id = bpt > 1 ? 0 : pid_m % mtiles;
        x.b0 = pid_b * S;
        return x;
    };
    auto col0_of = [&](const Tile& T) { return (long)T.b0 * K; };   // (its first column: S samples a tile, K each)
    const int gblk = (int)blockIdx.x - n_v, step = nct; // (past the side CTAs: the grid's first n_v)
    // The ring is CONTINUOUS over a CTA's tiles: its k-step g (counted from the CTA's first) in stage g % ST, phase
    // (g / ST) & 1. A stage's record (the converters'): its samples' rescale factors, then their maxima after it.
    float* srec = (float*)(smem_raw + ST * STAGE);       // [ST][2 S]
    uint64_t* fullr = (uint64_t*)(srec + 2 * ST * S);    // a stage's copies landed (the producers')
    // A consumer warp's fragment n-blocks would be the tile's 8-column blocks 2 ni + wn (CuTe's (WM, WN) tiling): half
    // of every sample's ranks. The consumers' MMA permutes its N (`PermN`, below) so that warp wn's are whole samples,
    // wn and wn + 2 -- the collapse then stays in a warp, a (row, sample)'s 16 ranks in the 4 lanes of a group -- and
    // the stage holds V and u in the tile's own column order.
    static_assert(WN == 2 && S == 4 && K == 16, "the MMA's permutation: 2 warps across, 4 samples of 16 ranks");
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
    auto node_of = [&](const Tile& T, int m) {           // a tile row's twin row
        const int bq = bpt > 1 ? m / block_size : 0;
        return nids[T.r + bq] + (bpt > 1 ? m - bq * block_size : T.tile_id * BM + m);
    };
    if (tid >= NC + NV) {
        // ---------------- producers: the CTA's k-step g into stage g % ST once the consumers released it ----------------
        // Every per-chunk address made before the loop (pinned in registers: else the compiler remakes the swizzled
        // offsets from the thread id every stage), the slab bases and the rows' children loaded ahead, and no
        // shared-memory load in the loop (MEASURED: one costs a cp.async loop 34% to 2.1x, even predicated off).
        const int ptid = tid - NC - NV;
        constexpr int TA = ABYTES / 16, CA = (TA + NP - 1) / NP;
        constexpr int TB = UBYTES / 16, CB = (TB + NP - 1) / NP;
        constexpr int NC4 = (FUSED ? 2 : 1) * KT * (S / 4), CC = (NC4 + NP - 1) / NP;
        // (`places`: the chunks' places in a stage -- the same every tile -- and the strides, pinned in registers)
        uint32_t oA[CA], oB[CB], oC[CC];
        int eB[CB], cx[CC], ce[CC];
        long vrows_l = 0;                                // (strides in registers: else reloaded from the constant bank)
        int B_i = 0;
        uint32_t sb_r = 0;
        auto places = [&]() {
            {
                Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
                CUTE_UNROLL
                for (int q = 0; q < CA; q++) {
                    const int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                    oA[q] = i < TA ? (uint32_t)((char*)&sAt(mm, e) - smem_raw) : ~0u;
                }
            }
            {
                Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
                CUTE_UNROLL
                for (int q = 0; q < CB; q++) {
                    const int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                    oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - smem_raw) : ~0u;
                    eB[q] = e;
                }
            }
            CUTE_UNROLL
            for (int k = 0; k < CC; k++) {
                constexpr int s4 = S / 4;
                const int ci = ptid + k * NP;
                const bool on = ci < NC4;
                const int x = on ? ci / (KT * s4) : 0, rem = ci - x * (KT * s4);
                const int e = on ? rem / s4 : 0, q4 = on ? rem % s4 : 0;
                oC[k] = on ? (uint32_t)(ABYTES + 2 * UBYTES + ((x * KT + e) * S + 4 * q4) * 4) : ~0u;   // (after V, u)
                cx[k] = x; ce[k] = e;
            }
            vrows_l = vrows; B_i = B; sb_r = sbase;
            asm volatile("" : "+l"(vrows_l), "+r"(B_i), "+r"(sb_r));
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) asm volatile("" : "+r"(oA[q]));
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) asm volatile("" : "+r"(oB[q]), "+r"(eB[q]));
            CUTE_UNROLL
            for (int k = 0; k < CC; k++) asm volatile("" : "+r"(oC[k]), "+r"(cx[k]), "+r"(ce[k]));
        };
        places();
        // The tile being produced (`pt`) and its bases, `tile_bases`: A's chunks in their slabs, V's row 0 at the chunk's
        // columns, the values at the chunk's samples, the collapse's pieces
        // of A of tile row `ptid` (a sample's K ranks one 64-byte run, the tile's S of them: one prefetched into L2 a
        // k-step) -- and the parameter tiles' pointers (cp.async: each chunk's at the current edge tile's slab, the
        // next one's base loaded an edge tile ahead)
        static_assert(NP == BM, "a producer a tile row's prefetches");
        Tile pt = tile_at(gblk);
        int gA[CA], bA[CA];
        const bfloat16_t* vptr[CB];
        const float* cptr[CC];
        const float* pfa = nullptr;
        int ecur = 0;
        const bfloat16_t* aptr[CA];
        long pnx[CA];
        auto tile_bases = [&]() {
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                const int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                const int bq = bpt > 1 ? mm / block_size : 0;
                gA[q] = e * block_size + (bpt > 1 ? mm - bq * block_size : mm) + pt.tile_id * BM;
                bA[q] = pt.r + bq;
            }
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) vptr[q] = vt + row0 + col0_of(pt) + ((ptid + q * NP) % (BN / 8)) * 8;
            CUTE_UNROLL
            for (int k = 0; k < CC; k++) {
                constexpr int s4 = S / 4;
                const int ci = ptid + k * NP;
                cptr[k] = vals + pt.b0 + 4 * (ci < NC4 ? (ci % (KT * s4)) % s4 : 0);
            }
            pfa = fa + (long)pt.b0 * a_sb + (node_of(pt, ptid) - vn0) * K;
            ecur = 0;
            if constexpr (!TMAA) {
                CUTE_UNROLL
                for (int q = 0; q < CA; q++) {
                    aptr[q] = params + gA[q] + pbase[(long)bA[q] * knt];
                    pnx[q] = knt > 1 ? pbase[(long)bA[q] * knt + 1] : 0;
                }
            }
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) asm volatile("" : "+r"(gA[q]), "+r"(bA[q]));
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) asm volatile("" : "+l"(vptr[q]));
            CUTE_UNROLL
            for (int k = 0; k < CC; k++) asm volatile("" : "+l"(cptr[k]));
        };
        tile_bases();
        auto prefetch = [&](int q) { asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(pfa + (long)q * a_sb)); };
        const uint32_t bar0 = (uint32_t)__cvta_generic_to_shared(&fullr[0]);
        auto issue_a = [&](int t, uint32_t ss, int st) {    // (t increasing; st = t % ST, its stage)
            if constexpr (TMAA) {
                if (ptid == 0) {
                    const uint32_t bar = bar0 + st * 8;
                    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(ABYTES)
                                 : "memory");
                    tma_load_3d(ss, &tmapA, 0, t * KT, pt.r, bar);
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
        // the CTA's first tile's first stages' parameter tiles while the launch before may still run (none of the pass
        // writes them)
        const int npre = min(nk, ST);
        for (int t = 0; t < npre; t++) issue_a(t, sbase + t * STAGE, t);
        // V's children, the element rows' bases and (run 0) the rows' children, two k-steps ahead: a tile's tables
        // (`tile_tables`: its first two k-steps' loaded)
        const int* jg = nullptr;
        const int* cbr = nullptr;
        const long* ebr = nullptr;
        int jr[2][CB], jcr[2][CC];
        long br[2][2];
        const int* jq[CB];                               // (a chunk's row in jpos at k-step 0)
        auto load_j = [&](int t, int (&j)[CB]) {
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) j[q] = TB % NP == 0 || oB[q] != ~0u ? ldv_s32(jq[q] + t * KT) : -1;
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
        auto tile_rows = [&]() {                         // (the tile's rows of the tables)
            jg = jpos + (long)ugrp[pt.r] * num_edges;
            cbr = FUSED ? cbase + (long)pt.r * knt * 4 : nullptr;
            ebr = ebase + (long)pt.r * knt;
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) jq[q] = jg + eB[q];
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) asm volatile("" : "+l"(jq[q]));
        };
        auto tile_tables = [&]() {
            tile_rows();
            if (VTMA && ptid != 0) return;
            load_j(0, jr[0]); load_b(0, br[0]);
            if constexpr (!VTMA) load_jc(0, jcr[0]);
            if (nk > 1) {
                load_j(1, jr[1]); load_b(1, br[1]);
                if constexpr (!VTMA) load_jc(1, jcr[1]);
            }
        };
        tile_tables();
        cudaGridDependencySynchronize();                 // (the children's values: the launches before's)
        int s = 0, g = 0;                                // the ring's stage, the CTA's k-steps so far
        unsigned ph = 0u;                                // (the round's phase)
        auto kstep = [&](int t, int u) {                 // k-step t of the tile, its tables in slot u
            uint32_t ss = sb_r + s * STAGE;
            asm volatile("" : "+r"(ss));
            if (g >= ST) mbar_wait(&empty[s], (int)(ph ^ 1u));   // (used before: once the consumers released it)
            if (g >= npre) issue_a(t, ss, s);            // (the CTA's first stages' A went ahead)
            if constexpr (VTMA) {
                if (ptid == 0) {                         // (the k-step's children from jr[u][0] on: one box each)
                    const uint32_t bar = bar0 + s * 8;
                    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar),
                                 "r"(UBYTES + (FUSED ? 2 : 1) * KT * S * 4) : "memory");
                    tma_load_2d(ss + ABYTES, &tmapV, (int)(row0 + col0_of(pt)), jr[u][0], bar);
                    tma_load_2d(ss + ABYTES + 2 * UBYTES, &tmapC, pt.b0, (int)br[u][0], bar);
                    if (FUSED) tma_load_2d(ss + ABYTES + 2 * UBYTES + KT * S * 4, &tmapC, pt.b0, (int)br[u][1], bar);
                }
            } else {
                CUTE_UNROLL
                for (int q = 0; q < CB; q++) {
                    if (TB % NP != 0 && oB[q] == ~0u) continue;
                    const int jc = jr[u][q];     // (no child: nothing copied, zeros)
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
            }
            cpa_mbar_arrive(&fullr[s]);          // (tracks every copy this thread issued: the stage's A too)
            if (t < S) prefetch(t);
            if (t + 2 < nk && (!VTMA || ptid == 0)) {   // (VTMA: the issuing producer's alone, its first chunk's child)
                load_j(t + 2, jr[u]); load_b(t + 2, br[u]);
                if constexpr (!VTMA) load_jc(t + 2, jcr[u]);
            }
            g++;
            s = s + 1 == ST ? 0 : s + 1;
            if (s == 0) ph ^= 1u;                 // (the ring went round)
        };
        for (int tile = gblk;;) {
            for (int jb = 0; jb < nk; jb += 2) {
                CUTE_UNROLL
                for (int u = 0; u < 2; u++) {
                    const int t = jb + u;
                    if (t < nk) kstep(t, u);
                }
            }
            for (int q = nk; q < S; q++) prefetch(q);
            tile += step;
            if (tile >= ntiles) break;
            pt = tile_at(tile);
            tile_bases();
            tile_tables();
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
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
        float mrun[SPW];                                 // the running max of the warp's samples (every lane)
        int s = 0;
        unsigned ph = 0u;
        auto convert = [&]() {                           // the next stage of the ring
            mbar_wait(&fullr[s], (int)ph);
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
            for (int i = 0; i < SPW; i++) mx[i] = fkey_inv(__reduce_max_sync(FULLM, fkey(mx[i])));
            float ex[SPW];
            CUTE_UNROLL
            for (int i = 0; i < SPW; i++) {
                const float mold = mrun[i], mnew = fmaxf(mold, mx[i]);
                // (the consumers' factor: the sums so far where the max rose, else exactly 1)
                const float fac = mnew > mold && mold > -INFINITY ? __expf(mold - mnew) : 1.f;
                if (lane == 0) { srec[s * 2 * S + w * SPW + i] = fac; srec[s * 2 * S + S + w * SPW + i] = mnew; }
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
                    const float2 f = make_float2(__uint_as_float(h[x] << 16), __uint_as_float(h[x] & 0xffff0000u));
                    __nv_bfloat162 p2 = __floats2bfloat162_rn(f.x * e_, f.y * e_);
                    h[x] = *reinterpret_cast<unsigned*>(&p2);
                }
                asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ss + oB[j] + UBYTES), "r"(h[0]), "r"(h[1]),
                             "r"(h[2]), "r"(h[3]) : "memory");
            }
            mbar_arrive(&full[s]);
            s = s + 1 == ST ? 0 : s + 1;
            if (s == 0) ph ^= 1u;                         // (the ring went round)
        };
        for (int tile = gblk; tile < ntiles; tile += step) {
            CUTE_UNROLL
            for (int i = 0; i < SPW; i++) mrun[i] = -INFINITY;   // (a tile's maxima anew)
            for (int jj = 0; jj < nk; jj++) convert();
            if (cmo != nullptr && lane == 0) {           // (the tile's children's maxima: the pass's end's c_m there)
                int pid_b, pid_m;
                tile_of(tile, nbt, ntiles / nbt, nbg, pid_b, pid_m);
                CUTE_UNROLL
                for (int i = 0; i < SPW; i++) cmo[(long)pid_m * B + pid_b * S + w * SPW + i] = mrun[i];
            }
        }
        return;
    }
    // ---------------- consumers: a tile's k-steps into two fp32 accumulators, S_r (A u) and Z_r (A V); its collapse ----------------
    // (N permuted: logical column c + 8 b0 + 16 b1 + 32 b2 -- b0 the warp across, b1 b2 its n-block -- at the tile's
    // column c + 16 b0 + 8 b1 + 32 b2: warp wn's n-block ni the tile's sample 2 (ni / 2) + wn, its ranks 8 (ni % 2) +
    // [0, 8))
    using PermN = Layout<Shape<_8, _2, _2, _2>, Stride<_1, _16, _8, _32>>;
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{},
                                  cute::Tile<Underscore, PermN, Underscore>{});   // (`Tile` here: the kernel's)
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
    // tCrL is ((2, 2), MI, NI): n-block ni the tile's columns 16 wn + 32 (ni / 2) + 8 (ni % 2) + [0, 8) -- its sample
    // 2 (ni / 2) + wn, its ranks 8 (ni % 2) + [0, 8) (the MMA's N permutation)
    constexpr int MI = BM / (16 * WM), NI = size<2>(tCrL);
    // Each fragment's partition made once, at stage 0 (a stage's: the same layout STAGE bytes on -- the swizzles act on
    // the offsets within 1024 bytes): made in the k-step, the compiler remade their swizzled offsets every k-step
    const auto tXsA0 = tA.partition_S(make_tensor(make_smem_ptr(stA(0)), sAl));
    const auto tXsV0 = tB.partition_S(make_tensor(make_smem_ptr(stB(0)), sBl));
    const auto tXsU0 = tB.partition_S(make_tensor(make_smem_ptr(stU(0)), sBl));
    static_assert(STAGE % 1024 == 0, "a stage a multiple of the swizzles' period");
    // (an A fragment's source: its first element's place in stage 0, for the first 16 k -- the next 16 k's a constant
    // on -- pinned in a register a tile: else remade from the thread id every k-step)
    constexpr int NFA = size<1>(tXrA);
    uint32_t fA[NFA];
    const int dkA = KB > 1 ? (int)((char*)&tXsA0(0, 0, KB - 1) - (char*)&tXsA0(0, 0, 0)) / (KB - 1) : 0;
    auto load_frag = [&](auto& cpy, uint32_t off, auto&& like, auto&& dst) {
        copy(cpy, make_tensor(make_smem_ptr((bfloat16_t*)(smem_raw + off)), make_layout(shape(like))), dst);
    };
    // (the C fragment's places in closed form, the MMA's (WM, WN) warps and N permutation: element pair 2 ro (ro < NR)
    // the tile row 64 (ro / 2) + 16 wm + lane / 4 + 8 (ro % 2); n-block ni's columns 16 wn + 8 (ni % 2) + 32 (ni / 2)
    // + 2 (lane % 4) + [0, 2))
    const int wn = (tid / 32) / WM;
    const int crow = 16 * ((tid / 32) % WM) + (tid % 32) / 4;
    static_assert(BM == 2 * 16 * WM && MI == 2, "the rows: two m16 atoms 16 WM apart");
    // The B fragments (V's for Z, u's for S) by ldmatrix.x4.trans, two a k half: lane l gives the row l % 8 of matrix
    // l / 8 = (n-block 2 p + l / 16, k half (l / 8) % 2) -- the tile's columns 16 wn + 32 p + 8 (l / 16) + [0, 8) (the
    // MMA's N permutation), k 8 ((l / 8) % 2) + l % 8 -- at its place in the 128-byte swizzled stage (Atom64: the
    // 16-byte chunk c of row k at chunk c ^ (k % 8)); the matrices' registers are B's 4 p + 8 kb .. 4 p + 8 kb + 3 (the
    // fragment's (r, ni % 2), ni / 2 = p). The second k half 16 rows -- 2048 bytes -- on.
    static_assert(KB == 2 && size(tCrB) == 32, "B: 16 k x 4 n-blocks a k half, two k halves");
    uint32_t fB[2];                                      // (made at a tile's start: below)
    uint32_t* rb = reinterpret_cast<uint32_t*>(raw_pointer_cast(tCrB.data()));
    auto load_b = [&](uint32_t base, int kb) {           // (base: the stage's V or u, a shared address)
        CUTE_UNROLL
        for (int p = 0; p < 2; p++)
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                         : "=r"(rb[4 * p + 8 * kb]), "=r"(rb[4 * p + 8 * kb + 1]), "=r"(rb[4 * p + 8 * kb + 2]),
                           "=r"(rb[4 * p + 8 * kb + 3])
                         : "r"(base + fB[p] + kb * 2048));
    };
    constexpr int NR = 2 * MI, NE = NR * NI;             // a thread's rows; its (row, n-block)s: 2 adjacent columns each
    static_assert(size(tCrL) == 2 * NE && NI == 4, "a thread: NR rows x 2 samples x 2 halves x 2 columns");
    int s = 0;
    unsigned ph = 0u;
    float mfin[2];                                       // its samples' maxima after the tile's last stage
    int tile = gblk;
    Tile T = tile_at(tile);
    int irow[NR];                                        // its rows' nodes in the node vector (the tile's: loaded in
    auto kbody = [&](const bool last) {                  // its last k-step, under the MMAs)
        mbar_wait(&full[s], (int)ph);
        if (last) {
            T = tile_at(tile);
            CUTE_UNROLL
            for (int ro = 0; ro < NR; ro++) irow[ro] = (int)(node_of(T, crow + 64 * (ro / 2) + 8 * (ro % 2)) - vn0);
        }
        const uint32_t so = (uint32_t)s * STAGE;         // (the stage's place)
        // the stage's A fragments once, V's into the B fragments for Z, then u's into the same registers for S
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {
            CUTE_UNROLL
            for (int i = 0; i < NFA; i++) load_frag(s2rA, so + fA[i] + kb * dkA, tXsA0(_, i, kb), tXrA(_, i, kb));
            load_b(sbase + so + ABYTES, kb);
        }
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrZ);
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) load_b(sbase + so + ABYTES + UBYTES, kb);
        const float* rec = srec + s * 2 * S + wn;        // the stage's record: its two samples' factors (1: no rise)
        const float f0 = rec[0], f1 = rec[2];
        if (last) { mfin[0] = rec[S]; mfin[1] = rec[S + 2]; }
        mbar_arrive(&empty[s]);                          // the stage is read (its record too)
        if (f0 != 1.f || f1 != 1.f) {                    // (a sample's max rose this stage: the sums so far rescaled)
            CUTE_UNROLL
            for (int i = 0; i < size(tCrL); i++) tCrL(i) *= (i / (4 * MI)) / 2 ? f1 : f0;
        }
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
        s = s + 1 == ST ? 0 : s + 1;
        if (s == 0) ph ^= 1u;                            // (the ring went round)
    };
    for (;; tile += step) {
    CUTE_UNROLL
    for (int i = 0; i < NFA; i++) { fA[i] = (uint32_t)((char*)&tXsA0(0, i, 0) - smem_raw); asm volatile("" : "+r"(fA[i])); }
    CUTE_UNROLL
    for (int p = 0; p < 2; p++) {                       // (made anew a tile: none lives across the collapse)
        const int l = tid % 32, m = l / 8, n0 = 16 * wn + 32 * p + 8 * (m / 2), k = 8 * (m % 2) + l % 8;
        fB[p] = (uint32_t)(k * 128 + (((n0 >> 3) ^ (l & 7)) << 4));
        asm volatile("" : "+r"(fB[p]));
    }
    clear(tCrL);
    clear(tCrZ);
    for (int j = 0; j + 1 < nk; j++) kbody(false);
    kbody(true);
    if (tile == gblk)
        cudaGridDependencySynchronize();                 // (the epilogue's own guarantee: the launch before is done)
    // THE COLLAPSE, from the registers: element e = ro + NR ni holds the columns (c, c + 1) of row ro's node in n-block
    // ni; a (row, sample)'s 16 ranks are 2 halves x 2 columns here x the 4 lanes of the group. The node's value
    // lse_r(A + log S_r) - lse_r(A + log Z_r), log Z and the share differences the backward's dA takes (`vnm`: the
    // layer's kept D), as `lowrank_mixture.cu`'s `collapse_sums<16>` forms them, the sums over the ranks as a tree
    // (the pair, the halves, the lanes) and a share as its term over the sum. A warp issues in order -- it waits a
    // cache round trip at the first use of a load not back yet, and a spill is such a load (the stages hold the SM's
    // cache): so nothing on the stack, and each sample's loads issued ahead of work that needs none.
    {
        const long cT = col0_of(T);
        constexpr unsigned FULLW = 0xffffffffu;
        const bool head = (tid & 3) == 0;                // (a group's one writer of the pair's values)
        int c0[2];                                       // a sample's columns here: c0, c0 + 1, c0 + 8, c0 + 9
        float2 A0[2][NR], A1[2][NR], H0[2], H1[2];       // A's and the shifts' there
        float ms[2][NR], mz[2][NR];
        auto load = [&](int sg) {
            c0[sg] = 16 * wn + 32 * sg + 2 * (tid % 4);
            H0[sg] = *(const float2*)&shift[row0 + cT + c0[sg]];
            H1[sg] = *(const float2*)&shift[row0 + cT + c0[sg] + 8];
            const float* base = fa + (long)(T.b0 + c0[sg] / K) * a_sb + c0[sg] % K;   // (A[b, i, r] as the caller gives it)
            CUTE_UNROLL
            for (int ro = 0; ro < NR; ro++) {
                A0[sg][ro] = *(const float2*)(base + (long)irow[ro] * K);
                A1[sg][ro] = *(const float2*)(base + (long)irow[ro] * K + 8);
            }
        };
        auto logs = [&]() {
            CUTE_UNROLL
            for (int i = 0; i < size(tCrL); i++) { tCrL(i) = logf(tCrL(i)); tCrZ(i) = logf(tCrZ(i)); }
        };
        auto part1 = [&](int sg) {                       // the terms' logs (in the sums' registers), their maxima
            const float mm = mfin[sg];
            const float M00 = mm + H0[sg].x, M01 = mm + H0[sg].y, M10 = mm + H1[sg].x, M11 = mm + H1[sg].y;
            CUTE_UNROLL
            for (int ro = 0; ro < NR; ro++) {
                const int e0 = ro + NR * (2 * sg), e1 = e0 + NR;
                tCrL(2 * e0) = A0[sg][ro].x + (tCrL(2 * e0) + M00);
                tCrL(2 * e0 + 1) = A0[sg][ro].y + (tCrL(2 * e0 + 1) + M01);
                tCrL(2 * e1) = A1[sg][ro].x + (tCrL(2 * e1) + M10);
                tCrL(2 * e1 + 1) = A1[sg][ro].y + (tCrL(2 * e1 + 1) + M11);
                tCrZ(2 * e0) = A0[sg][ro].x + (tCrZ(2 * e0) + H0[sg].x);
                tCrZ(2 * e0 + 1) = A0[sg][ro].y + (tCrZ(2 * e0 + 1) + H0[sg].y);
                tCrZ(2 * e1) = A1[sg][ro].x + (tCrZ(2 * e1) + H1[sg].x);
                tCrZ(2 * e1 + 1) = A1[sg][ro].y + (tCrZ(2 * e1 + 1) + H1[sg].y);
                ms[sg][ro] = fmaxf(fmaxf(tCrL(2 * e0), tCrL(2 * e0 + 1)), fmaxf(tCrL(2 * e1), tCrL(2 * e1 + 1)));
                mz[sg][ro] = fmaxf(fmaxf(tCrZ(2 * e0), tCrZ(2 * e0 + 1)), fmaxf(tCrZ(2 * e1), tCrZ(2 * e1 + 1)));
                if (uaw != nullptr) {                    // (the A operand: `lowrank_mixture.cu`'s `a_operand_kernel`'s)
                    const float2 p = A0[sg][ro], q = A1[sg][ro];
                    float a = fmaxf(fmaxf(p.x, p.y), fmaxf(q.x, q.y));
                    a = fmaxf(a, __shfl_xor_sync(FULLW, a, 1));
                    a = fmaxf(a, __shfl_xor_sync(FULLW, a, 2));
                    const bool on = a > -INFINITY;
                    const __nv_bfloat162 u0 = __floats2bfloat162_rn(on && p.x > -INFINITY ? expf(p.x - a) : 0.0f,
                                                                    on && p.y > -INFINITY ? expf(p.y - a) : 0.0f);
                    const __nv_bfloat162 u1 = __floats2bfloat162_rn(on && q.x > -INFINITY ? expf(q.x - a) : 0.0f,
                                                                    on && q.y > -INFINITY ? expf(q.y - a) : 0.0f);
                    const int b = T.b0 + c0[sg] / K;
                    bfloat16_t* u = uaw + (long)irow[ro] * urows + row0 + (long)b * K + c0[sg] % K;
                    *reinterpret_cast<__nv_bfloat162*>(u) = u0;
                    *reinterpret_cast<__nv_bfloat162*>(u + 8) = u1;
                    if (head) amw[(long)irow[ro] * B + b] = a;
                }
            }
        };
        float ss[NR], sz[NR];
        auto part2a = [&](int sg) {
            CUTE_UNROLL
            for (int d = 1; d <= 2; d <<= 1) {
                CUTE_UNROLL
                for (int ro = 0; ro < NR; ro++) {
                    ms[sg][ro] = fmaxf(ms[sg][ro], __shfl_xor_sync(FULLW, ms[sg][ro], d));
                    mz[sg][ro] = fmaxf(mz[sg][ro], __shfl_xor_sync(FULLW, mz[sg][ro], d));
                }
            }
            CUTE_UNROLL
            for (int ro = 0; ro < NR; ro++) {            // the terms (a pair of no mass: nan, caught below)
                const int e0 = ro + NR * (2 * sg), e1 = e0 + NR;
                CUTE_UNROLL
                for (int x = 0; x < 2; x++) {
                    tCrL(2 * e0 + x) = expf(tCrL(2 * e0 + x) - ms[sg][ro]); tCrL(2 * e1 + x) = expf(tCrL(2 * e1 + x) - ms[sg][ro]);
                    tCrZ(2 * e0 + x) = expf(tCrZ(2 * e0 + x) - mz[sg][ro]); tCrZ(2 * e1 + x) = expf(tCrZ(2 * e1 + x) - mz[sg][ro]);
                }
                ss[ro] = (tCrL(2 * e0) + tCrL(2 * e0 + 1)) + (tCrL(2 * e1) + tCrL(2 * e1 + 1));
                sz[ro] = (tCrZ(2 * e0) + tCrZ(2 * e0 + 1)) + (tCrZ(2 * e1) + tCrZ(2 * e1 + 1));
            }
            CUTE_UNROLL
            for (int d = 1; d <= 2; d <<= 1) {
                CUTE_UNROLL
                for (int ro = 0; ro < NR; ro++) {
                    ss[ro] += __shfl_xor_sync(FULLW, ss[ro], d);
                    sz[ro] += __shfl_xor_sync(FULLW, sz[ro], d);
                }
            }
        };
        auto part2b = [&](int sg, auto&& mid) {          // (`mid`: run after the first two rows)
            const int b = T.b0 + c0[sg] / K;
            float nv[NR], zs[NR];                        // the pairs' values and log Z (the group's head stores them)
            CUTE_UNROLL
            for (int ro = 0; ro < NR; ro++) {
                const int e0 = ro + NR * (2 * sg), e1 = e0 + NR;
                const float ls = ms[sg][ro] > -INFINITY ? ms[sg][ro] + logf(ss[ro]) : -INFINITY;
                const float z = mz[sg][ro] > -INFINITY ? mz[sg][ro] + logf(sz[ro]) : -INFINITY;
                const bool live = ls > -INFINITY && z > -INFINITY;
                // (Z = 0, a node whose parameters are all 0: no mass, -inf -- as the collapse kernel)
                nv[ro] = live ? ls - z : -INFINITY; zs[ro] = z;
                const float rs = live ? 1.f / ss[ro] : 0.f, rz = live ? 1.f / sz[ro] : 0.f;
                const float o0 = live ? tCrL(2 * e0) * rs - tCrZ(2 * e0) * rz : 0.f;
                const float o1 = live ? tCrL(2 * e0 + 1) * rs - tCrZ(2 * e0 + 1) * rz : 0.f;
                const float o2 = live ? tCrL(2 * e1) * rs - tCrZ(2 * e1) * rz : 0.f;
                const float o3 = live ? tCrL(2 * e1 + 1) * rs - tCrZ(2 * e1 + 1) * rz : 0.f;
                float* vr = vnm + ((long)irow[ro] + vn0) * cols + cT + c0[sg];
                *(float2*)vr = make_float2(o0, o1);
                *(float2*)(vr + 8) = make_float2(o2, o3);
                if (ro == 1) mid();
            }
            if (head) {
                CUTE_UNROLL
                for (int ro = 0; ro < NR; ro++) {
                    node_mars[((long)irow[ro] + vn0 + nshift) * B + b] = nv[ro];
                    lz[(long)irow[ro] * B + b] = zs[ro];
                }
            }
        };
        load(0);
        logs();
        part1(0); part2a(0); part2b(0, [] {});
        load(1);
        part1(1); part2a(1); part2b(1, [] {});
    }
    if (tile + step >= ntiles) break;
    }
}


// ======================= the backward's one-launch form: the data half's element flows =======================
// On the virtual batch (its node values taken as 0) the data half's virtual element flows are one product over the
// parents, `G[j, v] = sum_i theta[i, j] U[i, v]`, `U[i, (b, r)] = exp(lf_b[i] - log S_b[i] + A_b[i, r])` -- the plain
// path writes `vnf = log U` and the virtual child values (`lowrank_backward_pre`), runs the stream backward on them
// (pre-pass, GEMM, merge: `vef = vem + log G`) and collapses (`lowrank_collapse_bw`). `lr_om_bw_kernel` does it in one
// launch, as `lr_om_kernel` does the forward: producers copy a k-step's parameter tile (theta^T: a child's 32 parents
// contiguous in its slab), the rows of the pass's `exp(A - am)` (`lowrank_a_operand`: Ua [M, rows], the rows (copy,
// sample, rank); `am` each (node, sample)'s largest A over its ranks) of its 32 parents and their flows, values, log
// normalizers and shifts `am`; converters scale Ua in place by `exp(lf - log S + am - m_b)`, m_b each sample's running
// max over the parents so far (the consumers rescale their sums
// where it rose); the epilogue forms `log G = log(sum) + m_b + shift` and from it, a (child, sample) a pair, the
// element flow `lse_r (c_b[j] + B_b[r, j] + log G)` -- written as the product's two children's node flows where the
// product layer below is fused in (FUSED) -- and dB's data half, `exp(c + B + log G)` (GRAD). A parent of no flow or
// no mass (lf or log S -inf): 0.
// GRAD: and dB's NORMALIZER half in the same launch, `- exp(B_b[r, j] + log Gn)`, `Gn[j, v] = sum_i theta[i, j]
// exp(lf_b[i] - log Z_b[i] + A_b[i, r])` (what the pass's end would form as `theta^T U` and take off with
// `lowrank_normalizer_grads`): the same parameter fragments, a second accumulator -- the converters write `Un = Ua
// exp(lf - log Z + am - n_b)` beside U, n_b its own running max (a parent of no flow or a log Z of -inf: 0). Scope (`om_bw_serves` and the plan): rank OM_K, B % (OM_BN / OM_K), the tile's
// children of one group of child blocks (equal parents), every parent a real node of the layer's node vector (no
// dummy), parents' blocks a multiple of 32.
// GRAD: and dA, by the grid's FIRST `n_da` CTAs: a node vector's (node, sample) pairs in tiles strided over them,
// `dLL / dA_r = f D_r` from the share differences the forward kept (`kd`, `kcols` wide, the twin's rows), as the plain
// path's `lowrank_backward_pre` forms it (the same operations, the same bits): its loads and adds beside the data
// half's GEMM from the launch's start -- few CTAs (the launcher's share of the SMs by the work), DRAM-bound, while
// the GEMM tiles leave the DRAM mostly idle. (Last in the grid they ran in the free SMs of its last wave -- at a large
// batch there are none: a wave of their own after the tiles, every SM on DRAM-bound work.)
// MEASURED (toast, HMM 1024 x 31, rank 16, B64, us a layer as the GPU timeline's frontier advances): the plain data
// half 54.5 (backward_pre 20.8, the element stream's pre-pass 5.4, GEMM 14.8, merge 4.2, collapse_bw 9.3); this kernel
// first 39.1 with dA a launch of its own beside (17.1) -- its main loop alone 16.8, the tail epilogue the rest: the
// staged [K, N, B] layout's 16-byte pieces of B and dB, nothing overlapping them --; with the epilogue's pieces
// prefetched into L2 a k-step at a time by the producers and the epilogue on every role's threads 28.8 (+ 16.9); with
// dA in the grid's free SMs 34.1 alone. Backward wall plain -> this: B16 1.83 -> 1.55 ms, B64 4.13 -> 3.57, B256
// 15.09 -> 13.73.
// TMAA (the parameter slabs affine: `virtual._om_bw_slabs`): the parameter tile and Ua's rows one TMA box each a
// k-step, issued by the first producer thread; the parents' four arrays stay the producers' cp.async. MEASURED
// (croissant, HMM 1024 x 31 in blocks of 32, rank 16, step ms, 2 rounds): B2048 61.36 -> 60.89, B1024 31.20 -> 30.89,
// B256 8.39 -> 8.31, B64 2.76 -> 2.70. The k-loop's resources alone (globaltimer stamps, B1024, us a tile, later
// waves): the consumers alone 16.1 (the tensor floor 14.2), the feed alone 13.1, both 18.9; each copy left out: the
// parents' arrays (32 rows' scattered 16-byte pieces each) 1.5, the parameter tile 1.8, Ua 0.8.
// The tile's serial phases (the same stamps by role): the first stage 2.56 us (the CTA's start to the producers' code
// 0.51, their prefetches' chains 0.51, the first copies issued 0.5, landed 0.51, converted 0.26), the epilogue 2.82
// (log G written 0.77, the pairs' loads 1.8, the stores 0.26) -- so the converters prefetch the epilogue's pieces, idle
// before their first stage, and the producers and converters load their pairs right after their own loops, the
// consumers only writing log G: MEASURED (toast, step ms, 2 rounds x 2) B2048 60.12 -> 59.57 / 60.33 -> 59.80, B1024
// 30.70 -> 30.33 / 30.67 -> 30.33, B256 8.49 -> 8.32 / 8.46 -> 8.31, B64 2.761 -> 2.704 / 2.753 -> 2.694. (The dA
// CTAs' next tile prefetched into L2: slower, its streaming lines held in the L2 -- +0.7% B1024, +1.2% B2048.)
template <int KC>
__device__ __forceinline__ void load_run4f(const float* __restrict__ p, float (&v)[KC]) {
    CUTE_UNROLL
    for (int q = 0; q < KC / 4; q++) {
        const float4 x = reinterpret_cast<const float4*>(p)[q];
        v[4 * q] = x.x; v[4 * q + 1] = x.y; v[4 * q + 2] = x.z; v[4 * q + 3] = x.w;
    }
}

template <int ST, bool FUSED, bool GRAD, bool STORE, bool TMAA>
__global__ void __launch_bounds__(OM_THREADS, 1) lr_om_bw_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ ppid, const long* __restrict__ ppar,
        const bfloat16_t* __restrict__ ua, long urows, long row0, const float* am, const float* nflow,
        const float* nmar, const float* lz, long nshift, long vn0, const long* __restrict__ chids, int npb, int cbs,
        int pbs, int mtiles, int B, const float* vals, const int* __restrict__ table, long first,
        const float* __restrict__ fb, long fb_sb, const int* __restrict__ jrow, long e_lo, int N, float* flows,
        float* grad_b, long gb_sb, const float* kd, long kcols, float* __restrict__ grad_a, long ga_sb, int M, int n_gemm,
        int n_da, int nbg, const __grid_constant__ CUtensorMap tmapA, int arow0, int acrow, int aprow,
        const __grid_constant__ CUtensorMap tmapU) {
    constexpr int BM = OM_BM, BN = OM_BN, K = OM_K, WM = 4, WN = 2;
    const int gblk = (int)blockIdx.x - n_da;             // (the grid: the dA CTAs, then the GEMM tiles)
    if (gblk < 0) {
        // ---------------- the dA CTAs: tiles of DTI nodes x DTS samples through shared memory ----------------
        // (the kept share differences `kd` are node-major, [rows, (sample, rank)]; dA as the caller gives it is
        // sample-major, [B, M, K]: a tile's kd read as DTI contiguous rows, dA updated as DTS contiguous runs)
        cudaTriggerProgrammaticLaunchCompletion();
        if (!GRAD) return;
        cudaGridDependencySynchronize();                 // (the node flows: the launches before's)
        constexpr int DTI = 32, DTS = 32, DROW = DTS * K + 4;    // (rows padded: 16-byte aligned, off the banks)
        constexpr int NQ = DTI * DTS * K / 4 / OM_THREADS, NE = DTI * DTS / OM_THREADS;   // (float4s, pairs a thread)
        static_assert(NQ * OM_THREADS * 4 == DTI * DTS * K && NE * OM_THREADS == DTI * DTS, "a tile's work evenly");
        float* sk = (float*)smem_raw;                    // [DTI][DROW]
        constexpr int DES = DTS + 1;                     // (se's rows padded: a warp's reads off one bank)
        float* se = sk + DTI * DROW;                     // [DTI][DES] a pair's exp(lf), -1 where it takes nothing
        const int nti = (M + DTI - 1) / DTI, nts = (B + DTS - 1) / DTS;
        // (every load of a phase issued before its stores: a tile's grad_a loads are not ordered after the stores of
        // another element -- the compiler cannot tell them apart, ga_sb being a run-time stride)
        for (long tile = (long)blockIdx.x; tile < (long)nti * nts; tile += n_da) {
            const int i0 = (int)(tile / nts) * DTI, s0 = (int)(tile % nts) * DTS;
            float4 kv[NQ];
            CUTE_UNROLL
            for (int q = 0; q < NQ; q++) {
                const int x = threadIdx.x + q * OM_THREADS;
                const int ii = x / (DTS * K / 4), c4 = x % (DTS * K / 4), i = i0 + ii, sb = s0 + (c4 * 4) / K;
                kv[q] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (i < M && sb < B) kv[q] = *reinterpret_cast<const float4*>(kd + (vn0 + i) * kcols + (long)s0 * K + c4 * 4);
            }
            float lfv[NE], lsv[NE];
            CUTE_UNROLL
            for (int q = 0; q < NE; q++) {
                const int x = threadIdx.x + q * OM_THREADS, ii = x / DTS, b = s0 + x % DTS, i = i0 + ii;
                lfv[q] = -INFINITY; lsv[q] = -INFINITY;
                if (b < B && i < M) {
                    const long row = vn0 + nshift + i;   // the node's circuit row
                    lfv[q] = nflow[row * B + b];
                    lsv[q] = nmar[row * B + b] + lz[(long)i * B + b];   // log S = log n + log Z
                }
            }
            CUTE_UNROLL
            for (int q = 0; q < NQ; q++) {
                const int x = threadIdx.x + q * OM_THREADS, ii = x / (DTS * K / 4), c4 = x % (DTS * K / 4);
                *reinterpret_cast<float4*>(sk + ii * DROW + c4 * 4) = kv[q];
            }
            CUTE_UNROLL
            for (int q = 0; q < NE; q++) {
                const int x = threadIdx.x + q * OM_THREADS;
                const float lf = lfv[q], ls = lsv[q];
                se[(x / DTS) * DES + x % DTS] = ((lf > -INFINITY) && (ls > -INFINITY)) ? expf(lf) : -1.f;
            }
            __syncthreads();
            float4 gv[NQ];
            float ef[NQ];
            CUTE_UNROLL
            for (int q = 0; q < NQ; q++) {
                const int x = threadIdx.x + q * OM_THREADS, ss = x / (DTI * K / 4), c4 = x % (DTI * K / 4);
                const int b = s0 + ss, ii = (c4 * 4) / K, r0 = (c4 * 4) % K, i = i0 + ii;
                ef[q] = (b < B && i < M) ? se[ii * DES + ss] : -2.f;      // (-2: no such pair; -1: it takes nothing)
                gv[q] = make_float4(0.f, 0.f, 0.f, 0.f);                  // (STORE: dA's first writer, onto +0)
                if (!STORE && ef[q] >= 0.f)
                    gv[q] = *reinterpret_cast<const float4*>(grad_a + (long)b * ga_sb + (long)i * K + r0);
            }
            CUTE_UNROLL
            for (int q = 0; q < NQ; q++) {
                const int x = threadIdx.x + q * OM_THREADS, ss = x / (DTI * K / 4), c4 = x % (DTI * K / 4);
                const int b = s0 + ss, ii = (c4 * 4) / K, r0 = (c4 * 4) % K, i = i0 + ii;
                if (ef[q] < -1.5f) continue;             // (past the node vector or the batch)
                if (!(ef[q] >= 0.f)) {                   // (no flow or no mass: dA takes nothing -- STORE: its +0)
                    if (STORE)
                        *reinterpret_cast<float4*>(grad_a + (long)b * ga_sb + (long)i * K + r0) = gv[q];
                    continue;
                }
                const float4 d = *reinterpret_cast<const float4*>(sk + ii * DROW + ss * K + r0);
                float4 v = gv[q];
                const float e = ef[q];
                v.x = v.x + e * d.x; v.y = v.y + e * d.y; v.z = v.z + e * d.z; v.w = v.w + e * d.w;
                *reinterpret_cast<float4*>(grad_a + (long)b * ga_sb + (long)i * K + r0) = v;
            }
            __syncthreads();
        }
        return;
    }
    constexpr int KT = 32, NC = WM * WN * 32, NP = 128, NV = 128;
    static_assert(NC + NP + NV == OM_THREADS, "the warp roles");
    constexpr int S = BN / K;                            // the tile's samples
    static_assert(S % 4 == 0, "a row's values of the tile's samples in 16-byte runs");
    constexpr int NU = GRAD ? 2 : 1;                     // (U; and Un, GRAD)
    constexpr int ABYTES = KT * BM * 2, UBYTES = KT * BN * 2, CBYTES = 4 * KT * S * 4;   // (flows, values, log Z, am)
    constexpr int STAGE = (ABYTES + NU * UBYTES + CBYTES + 1023) / 1024 * 1024;
    constexpr int GS = BN + 4;                           // the epilogue's log G (log Gn) tile, a row's stride (padded)
    static_assert(NU * BM * GS * 4 <= ST * STAGE, "the log G tiles in the stages");
    const int cols = K * B, nbt = cols / BN;
    int pid_b, pid_m;                                    // (the grid's order: `tile_of`)
    tile_of(gblk, nbt, n_gemm / nbt, nbg, pid_b, pid_m);
    // the child tile: one of `mtiles` of a child block, or `bpt` consecutive blocks of one group (equal parents)
    const int bpt = BM > cbs ? BM / cbs : 1;
    const int cb0 = bpt > 1 ? pid_m * bpt : pid_m / mtiles, tile_id = bpt > 1 ? 0 : pid_m % mtiles;
    const int SPB = pbs / KT, nk = npb * SPB;            // k-steps a parent block, all
    float* smsh = (float*)(smem_raw + ST * STAGE);       // [NU][BN] the columns' final shifts (the converters')
    float* sfac = smsh + NU * BN;                        // [NU][ST][S] a stage's rescale factors (the converters')
    uint64_t* fullr = (uint64_t*)(sfac + NU * ST * S);
    uint64_t* full = fullr + ST;
    uint64_t* empty = full + ST;
    uint32_t* sflg = (uint32_t*)(empty + ST);           // [ST][NU] a stage's "a max rose": a byte a converter warp
    const int tid = threadIdx.x;
    auto sAl = tile_to_shape(AtomAK32{}, make_shape(Int<BM>{}, Int<KT>{}));   // (K-major: a child's parents a line)
    auto sBl = tile_to_shape(Atom64{}, make_shape(Int<BN>{}, Int<KT>{}));
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES); };
    auto stN = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES + UBYTES); };   // (GRAD: Un)
    auto stC = [&](int s) { return (const float*)(smem_raw + s * STAGE + ABYTES + NU * UBYTES); };   // [3][KT][S]
    if (tid == 0) {
        for (int s = 0; s < ST; s++) { mbar_init(&fullr[s], NP); mbar_init(&full[s], NV); mbar_init(&empty[s], NC); }
        if (TMAA) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");   // (visible to the TMA unit)
    }
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const long col0 = (long)pid_b * BN;
    const int b0 = pid_b * S;
    // The epilogue, the producers' and converters' (a (child, sample) pair `P` of the `BM S` two a thread, `P` and
    // `P + NP + NV` for the thread's `P` = tid - NC): from the consumers' log G tile `gs`, the pair's element flow over
    // the sample's K ranks and dB's data half. Its loads (the rows' chains to the child, the child's value, B, dB) by
    // `epi_load` right after the thread's own loop -- the producers' last copies issued ST k-steps before the consumers'
    // end, the converters' last stage one or two -- the rest past the log G tile (`epi_finish`): every load before the
    // first store, as one thread a pair did. (The pieces were prefetched into L2 by the converters at the start.)
    static_assert(BM * S == 2 * (NP + NV), "two epilogue pairs a producer and a converter");
    float* gs = (float*)smem_raw;
    float* gn = gs + BM * GS;                            // (GRAD: log Gn)
    auto child_of = [&](int m, long& e, int& j) {         // a tile row's element row and the node vector's child
        const int bq = bpt > 1 ? m / cbs : 0;
        e = chids[cb0 + bq] + (bpt > 1 ? m - bq * cbs : tile_id * BM + m);
        j = jrow[e - e_lo];
    };
    struct EpiPair {                                     // (a pair's pieces, loaded: `epi_load`)
        long e, ch0, ch1;
        int j;
        float c, x[K], gv[K];
    };
    auto epi_load = [&](int P, EpiPair& p) {
        const int m = P / S, sl = P % S, b = b0 + sl;
        child_of(m, p.e, p.j);
        if (p.j < 0) return;                             // (no child of the node vector)
        p.ch0 = 0; p.ch1 = 0;
        if (FUSED) {
            const int* chp = table + (p.e - first) * 2;
            p.ch0 = chp[0]; p.ch1 = chp[1];
            p.c = vals[p.ch0 * B + b] + vals[p.ch1 * B + b];
        } else {
            p.c = vals[p.e * B + b];
        }
        const float* bsrc = fb + (long)b * fb_sb + p.j;  // B[b, r, j] as the caller gives it: r at a stride of N
        CUTE_UNROLL
        for (int r = 0; r < K; r++) p.x[r] = bsrc[(long)r * N];
        if (GRAD) {
            const float* g = grad_b + (long)b * gb_sb + p.j;
            CUTE_UNROLL
            for (int r = 0; r < K; r++) p.gv[r] = STORE ? 0.f : g[(long)r * N];   // (STORE: dB's first writer)
        }
    };
    auto epi_finish = [&](int P, EpiPair& p) {
        if (p.j < 0) return;
        const int m = P / S, sl = P % S, b = b0 + sl;
        const long e = p.e, ch0 = p.ch0, ch1 = p.ch1;
        const int j = p.j;
        const float c = p.c;
        float* g = GRAD ? grad_b + (long)b * gb_sb + j : nullptr;
        float (&x)[K] = p.x;
        float (&gv)[K] = p.gv;
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
            for (int r = 0; r < K; r++) g[(long)r * N] = (gv[r] + (x[r] > -INFINITY ? __expf(x[r]) : 0.f)) - nv[r];
        }
    };
    auto epilogue_wait = [&]() { asm volatile("bar.sync 5, %0;" ::"n"(OM_THREADS) : "memory"); };   // (gs written)
    if (tid >= NC + NV) {
        // ---------------- producers: k-step t into stage t % ST once the consumers released it ----------------
        const int ptid = tid - NC - NV;
        constexpr int TA = ABYTES / 16, CA = (TA + NP - 1) / NP;
        constexpr int TB = UBYTES / 16, CB = (TB + NP - 1) / NP;
        constexpr int NC4 = 4 * KT * (S / 4), CC = (NC4 + NP - 1) / NP;
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
        // Every chunk's base made once (in registers: else remade every k-step), its k-step's offset added: A's slab
        // table row and its offset in the slab; Ua's chunk at the vector's first node; the flows', values' or log
        // normalizers' chunk at the vector's first node (their rows: the parents' circuit rows, or node indices, lz's)
        const long* ptab[CA];
        const bfloat16_t* abase[CA];
        CUTE_UNROLL
        for (int q = 0; q < CA; q++) { ptab[q] = ppid + (long)cA[q] * npb; abase[q] = params + mA[q]; }
        // (TMAA: the tile's rows of the parameters' [rows, pbs] view at parent block 0; a parent block `aprow` on)
        const int arow = arow0 + cb0 * acrow + (bpt > 1 ? 0 : tile_id * BM);
        const long* ptab0 = ppar + (long)cb0 * npb;
        const bfloat16_t* ubase[CB];
        CUTE_UNROLL
        for (int q = 0; q < CB; q++) ubase[q] = uptr[q] + (long)eB[q] * urows;
        const float* cfirst[CC];
        CUTE_UNROLL
        for (int k = 0; k < CC; k++) {
            const int e = ce[k] & 0xff, so = ce[k] >> 8;
            cfirst[k] = (cx[k] == 3 ? am : cx[k] == 2 ? lz : (cx[k] ? nmar : nflow) + (vn0 + nshift) * B) + (long)e * B +
                        b0 + so;
        }
        long urows_l = urows, B_l = B;
        asm volatile("" : "+l"(urows_l), "+l"(B_l));
        if constexpr (!TMAA) {
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) asm volatile("" : "+r"(oA[q]), "+l"(ptab[q]), "+l"(abase[q]));
        }
        if constexpr (!TMAA) {
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) asm volatile("" : "+r"(oB[q]), "+l"(ubase[q]));
        }
        CUTE_UNROLL
        for (int k = 0; k < CC; k++) asm volatile("" : "+r"(oC[k]), "+l"(cfirst[k]));
        // a k-step's tables, two steps ahead: its parent block's slab per chunk (A), its first parent's node index
        // (from its twin row, shared by the tile's blocks: the group's) -- the parent block and sub-step counted
        int lpb = 0, lsub = 0;
        auto slabs = [&](long (&pa)[CA], long& i0) {
            CUTE_UNROLL
            for (int q = 0; q < CA; q++)
                pa[q] = !TMAA && (TA % NP == 0 || oA[q] != ~0u) ? ldv_s64(ptab[q] + lpb) + lsub * KT : 0;
            i0 = ldv_s64(ptab0 + lpb) + lsub * KT - vn0;
            if (++lsub == SPB) { lsub = 0; lpb++; }
        };
        long pa[2][CA], i0[2];
        slabs(pa[0], i0[0]);
        if (nk > 1) slabs(pa[1], i0[1]);
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
                    if constexpr (TMAA) {
                        if (ptid == 0) {                 // (A: one box, the stage's barrier expecting its bytes)
                            const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&fullr[s]);
                            asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(ABYTES)
                                         : "memory");
                            tma_load_2d(ss, &tmapA, (t % SPB) * KT, arow + (t / SPB) * aprow, bar);
                        }
                    } else {
                        CUTE_UNROLL
                        for (int q = 0; q < CA; q++)
                            if (TA % NP == 0 || oA[q] != ~0u) cpa16(ss + oA[q], abase[q] + pa[u][q]);
                    }
                    if constexpr (TMAA) {                // (Ua: one box at the k-step's first parent i0, A's barrier)
                        if (ptid == 0) {
                            const uint32_t bar = (uint32_t)__cvta_generic_to_shared(&fullr[s]);
                            asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(UBYTES)
                                         : "memory");
                            tma_load_2d(ss + ABYTES, &tmapU, (int)(row0 + col0), (int)i0[u], bar);
                        }
                    } else {
                        CUTE_UNROLL                      // (i0: the k-step's first parent, its node index)
                        for (int q = 0; q < CB; q++)
                            if (TB % NP == 0 || oB[q] != ~0u) cpa16(ss + oB[q], ubase[q] + i0[u] * urows_l);
                    }
                    CUTE_UNROLL
                    for (int k = 0; k < CC; k++)
                        if (oC[k] != ~0u) cpa16(ss + oC[k], cfirst[k] + i0[u] * B_l);
                    cpa_mbar_arrive(&fullr[s]);
                    if (t + 2 < nk) slabs(pa[u], i0[u]);
                    if (++s == ST) { s = 0; ph ^= 1; }
                }
            }
        }
        EpiPair ep0, ep1;                                // (pairs NV + ptid and NV + ptid + NP + NV)
        epi_load(NV + ptid, ep0);
        epi_load(NV + ptid + NP + NV, ep1);
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        epilogue_wait();
        epi_finish(NV + ptid, ep0);
        epi_finish(NV + ptid + NP + NV, ep1);
        return;
    }
    if (tid >= NC) {
        // ---------------- converters: U = Ua exp(lf - log S + am - m), in place, m each sample's running max ----------------
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
        {
            // The epilogue's pieces prefetched into L2, all at once before the PDL wait (the converters idle until the
            // first stage lands; by the producers, the rows' chains held the first stage's copies back): tile row
            // `vtid`'s values (its S samples one 16-byte piece: the child's -- or the product's two children's --),
            // and of B and dB the tile's (sample, rank) rows over its children (a 512-byte run each, the children
            // contiguous from tile row 0's: two lines a converter; dB's read unless STORE)
            static_assert(NV == BM && NV == 2 * S * K, "a converter a tile row's values and two lines of B (and dB)");
            auto l2 = [](const float* a) { asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(a)); };
            long e;
            int j;
            child_of(vtid, e, j);
            if (j >= 0) {
                if (FUSED) {
                    const int* chp = table + (e - first) * 2;
                    l2(vals + (long)chp[0] * B + b0);
                    l2(vals + (long)chp[1] * B + b0);
                } else {
                    l2(vals + e * B + b0);
                }
            }
            child_of(0, e, j);
            if (j >= 0) {
                const int prow = vtid / 2, psl = prow / K, pr = prow % K;
                const long poff = (long)pr * N + j + (vtid % 2) * (BM / 2);
                const float* pfb = fb + (long)(b0 + psl) * fb_sb + poff;
                l2(pfb);
                l2(pfb + 32);
                if (GRAD && !STORE) {
                    const float* pdb = grad_b + (long)(b0 + psl) * gb_sb + poff;
                    l2(pdb);
                    l2(pdb + 32);
                }
            }
        }
        cudaGridDependencySynchronize();
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
                const float a = cs[(3 * KT + lane) * S + sl];   // (Ua's shift; -inf: its row 0)
                cv[0][i] = mx[0][i] = (lf > -INFINITY && ls > -INFINITY && a > -INFINITY) ? lf - ls + a : -INFINITY;
                if (GRAD)
                    cv[NU - 1][i] = mx[NU - 1][i] = (lf > -INFINITY && z > -INFINITY && a > -INFINITY) ? lf - z + a
                                                                                                    : -INFINITY;
            }
            CUTE_UNROLL
            for (int u = 0; u < NU; u++)
                CUTE_UNROLL
                for (int i = 0; i < SPW; i++) mx[u][i] = fkey_inv(__reduce_max_sync(FULLM, fkey(mx[u][i])));
            float ex[NU][SPW];
            bool rose[NU] = {};
            CUTE_UNROLL
            for (int u = 0; u < NU; u++)
                CUTE_UNROLL
                for (int i = 0; i < SPW; i++) {
                    const float mold = mrun[u][i], mnew = fmaxf(mold, mx[u][i]);
                    const float fac = mnew > mold && mold > -INFINITY ? __expf(mold - mnew) : 1.f;
                    if (lane == 0) sfac[(u * ST + s) * S + w * SPW + i] = fac;
                    rose[u] |= fac != 1.f;
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
                    const float2 f = make_float2(__uint_as_float(h[x] << 16), __uint_as_float(h[x] & 0xffff0000u));
                    __nv_bfloat162 p2 = __floats2bfloat162_rn(f.x * e_, f.y * e_);
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
            if (lane == 0) {                             // (the stage's flags: a max of this warp's samples rose)
                CUTE_UNROLL
                for (int u = 0; u < NU; u++) reinterpret_cast<unsigned char*>(sflg)[(s * NU + u) * 4 + w] = rose[u];
            }
            mbar_arrive(&full[s]);
            if (++s == ST) { s = 0; ph ^= 1; }
        }
        CUTE_UNROLL
        for (int u = 0; u < NU; u++) {                   // the columns' shifts: the running max
            float m = mrun[u][0];
            CUTE_UNROLL
            for (int i = 1; i < SPW; i++) if ((lane % CPW) / K == i) m = mrun[u][i];
            if (lane < CPW) smsh[u * BN + w * CPW + lane] = m;   // (Ua's shifts in the factors: none a column)
        }
        asm volatile("bar.arrive 3, %0;" ::"n"(NC + NV) : "memory");
        EpiPair ep0, ep1;                                // (pairs vtid and vtid + NP + NV)
        epi_load(vtid, ep0);
        epi_load(vtid + NP + NV, ep1);
        epilogue_wait();
        epi_finish(vtid, ep0);
        epi_finish(vtid + NP + NV, ep1);
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
    // Each fragment's partition made once, at stage 0 (a stage's: the same layout STAGE bytes on -- the swizzles act on
    // the offsets within 1024 bytes): made in the k-step, the compiler remade their swizzled offsets every k-step
    const auto tXsA0 = tA.partition_S(make_tensor(make_smem_ptr(stA(0)), sAl));
    const auto tXsB0 = tB.partition_S(make_tensor(make_smem_ptr(stB(0)), sBl));
    const auto tXsN0 = tB.partition_S(make_tensor(make_smem_ptr(stN(0)), sBl));
    static_assert(STAGE % 1024 == 0, "a stage a multiple of the swizzles' period");
    // A stage's first fragments (A's and B's of its first 16 k) and its flags, its full wait passed: issued during the
    // k-step before's last MMAs (the stages are converted well ahead), else the k-step's first MMAs wait for them
    auto head = [&](int s, int ph, uint32_t& fl) {
        mbar_wait(&full[s], ph);
        const int so = s * (STAGE / 2);
        copy(s2rA, make_tensor(tXsA0.data() + so, tXsA0.layout())(_, _, 0), tXrA(_, _, 0));
        copy(s2rB, make_tensor(tXsB0.data() + so, tXsB0.layout())(_, _, 0), tXrB(_, _, 0));
        fl = sflg[s * NU];
        if (GRAD) fl |= sflg[s * NU + NU - 1];
    };
    // A stage's k-step (its head loaded); RS: the sums first rescaled where a sample's max rose (`rescale`); the next
    // stage's head (sn, phn into fl) before the k-step's last MMAs where there is one
    auto kstep = [&](int s, auto rs, bool more, int sn, int phn, uint32_t& fl) {
        constexpr bool RS = decltype(rs)::value;
        const int so = s * (STAGE / 2);                  // (the stage's offset in bf16 elements)
        auto tXsA = make_tensor(tXsA0.data() + so, tXsA0.layout());
        auto tXsB = make_tensor(tXsB0.data() + so, tXsB0.layout());
        auto tXsN = make_tensor(tXsN0.data() + so, tXsN0.layout());
        if constexpr (RS) rescale(tCrL, 0, s);
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {
            if (kb + 1 < KB) {
                copy(s2rA, tXsA(_, _, kb + 1), tXrA(_, _, kb + 1));
                copy(s2rB, tXsB(_, _, kb + 1), tXrB(_, _, kb + 1));
            } else if (GRAD) {
                copy(s2rB, tXsN(_, _, 0), tXrB(_, _, 0));    // (the parameters' fragments again: Gn's too)
            } else {
                mbar_arrive(&empty[s]);
                if (more) head(sn, phn, fl);             // (into the first fragments' registers: their MMAs issued)
            }
            cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
        }
        if (GRAD) {
            if constexpr (RS) rescale(tCrN, 1, s);
            CUTE_UNROLL
            for (int kb = 0; kb < KB; kb++) {
                if (kb + 1 < KB) copy(s2rB, tXsN(_, _, kb + 1), tXrB(_, _, kb + 1));
                else {
                    mbar_arrive(&empty[s]);
                    if (more) head(sn, phn, fl);
                }
                cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrN);
            }
        }
    };
    // The k-steps: a stage none of whose maxima rose (the converters' flags, read after its full wait) takes the
    // k-step without the rescales' code, the others (MEASURED: ~30% of them, the HMM's parents in random order) the
    // rescaling one. (MEASURED, B64: the loop of every k-step's rescale checked in line -- its factors compared, its
    // multiplications predicated -- cost the kernel 7%: the checks' register pressure remade the fragments' swizzled
    // offsets every k-step; with the rescale as a branch the same. The next stage's head during the k-step's last MMAs:
    // the kernel 2% more at B64, 3% at B16.)
    int s = 0, ph = 0;
    uint32_t fl = 0u;
    head(0, 0, fl);
    for (int j = 0; j < nk; j++) {
        const int sn = s + 1 == ST ? 0 : s + 1, phn = sn == 0 ? ph ^ 1 : ph;
        const uint32_t f = fl;
        if (f != 0u) kstep(s, cute::true_type{}, j + 1 < nk, sn, phn, fl);
        else kstep(s, cute::false_type{}, j + 1 < nk, sn, phn, fl);
        s = sn; ph = phn;
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
    epilogue_wait();                                     // (the log G tile handed over: the pairs are the others')
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

// A 2D tensor map (`inner` x `outer` elements of `bytes` each, rows `inner` apart; a box `b0` x `b1`), encoded once
// per its arguments: the forward's V rows (`swz`: the stage's 128-byte swizzle) and children's values
static CUtensorMap om_map2d(const void* base, CUtensorMapDataType dt, int bytes, long inner, long outer, int b0, int b1,
                            CUtensorMapSwizzle swz) {
    struct Entry { const void* base; int dt, bytes; long inner, outer; int b0, b1, swz; CUtensorMap map; };
    static std::vector<Entry> kept;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    for (size_t i = 0; i < kept.size(); i++) {
        const Entry& e = kept[i];
        if (e.base == base && e.dt == (int)dt && e.bytes == bytes && e.inner == inner && e.outer == outer && e.b0 == b0 &&
            e.b1 == b1 && e.swz == (int)swz) {
            const CUtensorMap m = e.map;
            if (i > 0) { const Entry hit = kept[i]; kept.erase(kept.begin() + i); kept.insert(kept.begin(), hit); }
            return m;
        }
    }
    Entry e{base, (int)dt, bytes, inner, outer, b0, b1, (int)swz, {}};
    cuuint64_t gdim[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
    cuuint64_t gstride[1] = {(cuuint64_t)inner * bytes};
    cuuint32_t box[2] = {(cuuint32_t)b0, (cuuint32_t)b1};
    cuuint32_t es[2] = {1, 1};
    const CUresult rr = cuTensorMapEncodeTiled(&e.map, dt, 2, const_cast<void*>(base), gdim, gstride, box, es,
                                               CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rr == CUDA_SUCCESS, "lowrank om: cuTensorMapEncodeTiled failed (", (int)rr, ")");
    kept.insert(kept.begin(), e);
    if (kept.size() > 16) kept.pop_back();
    return e.map;
}

// The forward's CTAs a launch at most (0: an SM each): the tests make them few, so a CTA takes several tiles at the
// sizes they run (one or two a CTA otherwise)
static int g_om_ctas_most = 0;
void set_om_ctas_most(int64_t n) { g_om_ctas_most = (int)std::max<int64_t>(n, 0); }
// The one-launch kernels' last grids (for the tests: whether their hooks took): the forward's CTAs and sample tiles a
// group; the backward's dA CTAs and sample tiles a group
static int64_t g_last_grids[6] = {0, 0, 0, 0, 0, 0};
std::vector<int64_t> om_last_grids() { return std::vector<int64_t>(g_last_grids, g_last_grids + 6); }
// The one-launch kernels' TMA boxes (the forward's V rows and values, the backward's parameter tiles and Ua rows):
// 0 where they serve (the forward's where a CTA has more than one tile), 1 wherever they serve, 2 none -- every copy
// the producers' cp.async (the tests: the same bits)
static int g_om_boxes = 0;
void set_om_boxes(int64_t mode) { g_om_boxes = (int)std::min<int64_t>(std::max<int64_t>(mode, 0), 2); }

// The current device's SMs
static int om_sms() {
    static int sms_of[64] = {};
    int dv = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dv));
    if (dv < 64 && sms_of[dv] > 0) return sms_of[dv];
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dv));
    if (dv < 64) sms_of[dv] = sms;
    return sms;
}
// The sample tiles a group of the one-launch grids' order (`tile_of`): a quarter of the SMs -- the node tiles of a
// sample tile then start within a wave or two of each other. MEASURED with the dA CTAs first (toast, HMM 1024 x 31,
// rank 16, ms a step, the grid's plain order -> groups of 47 of 188 SMs): B64 2.93 -> 2.83, B256 9.03 -> 8.87, B512
// 17.63 -> 17.06, B1024 35.5 -> 33.9, B2048 77.1 -> 70.3 (the groups alone at B2048: 77.1 -> 73.9-74.8 at 94 / 188 /
// 376). The tests make the groups small (`set_tile_group`), so the sizes they run at take several.
static int g_tile_group = 0;
void set_tile_group(int64_t n) { g_tile_group = (int)std::max<int64_t>(n, 0); }
static int om_tile_group() { return g_tile_group > 0 ? g_tile_group : std::max(1, om_sms() / 4); }

template <bool FUSED, bool TMAA, bool VTMA>
static void launch_om(const bfloat16_t* params, const long* pbase, const bfloat16_t* vt, long vrows, long row0,
                      const float* shift, const float* vals, const long* ebase, const int* cbase, const int* jpos,
                      const int* ugrp, int knt, int rows, int block_size, int B, float* vnm, const long* nids,
                      int num_edges, const CUtensorMap& tmapA, const float* fa, long a_sb, float* node_mars,
                      float* lz, long nshift, long vn0, bfloat16_t* uaw, long urows, float* amw,
                      const CUtensorMap& tmapV, const CUtensorMap& tmapC, const float* vfb, long vf_sb,
                      bfloat16_t* vout, float* vsh, long vrow0, int vX, int n_v, float* cmo, cudaStream_t st) {
    constexpr int ST = OM_ST, S = OM_BN / OM_K;
    constexpr int STAGE = (32 * OM_BM * 2 + 2 * 32 * OM_BN * 2 + 2 * 32 * S * 4 + 1023) / 1024 * 1024;
    constexpr int SMEM = ST * STAGE + 2 * ST * S * 4 + 3 * ST * 8;   // (the stages, their records, the barriers)
    static_assert(SMEM <= 101376, "past sm_120's 99 KB of shared memory a block");
    auto kernel = lr_om_kernel<ST, FUSED, TMAA, VTMA>;
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if (dev < 64) attr_set[dev] = true;
    }
    const int mtiles = OM_BM >= block_size ? 1 : block_size / OM_BM;
    const long n = (long)(OM_K * B / OM_BN) * ((long)rows * block_size / OM_BM);
    TORCH_CHECK(n < (1L << 31), "lowrank om: past 2^31 tiles");
    // (persistent: a CTA an SM at most -- one fits, by its shared memory --, each taking every grid-th tile)
    int nsm = om_sms() - n_v;                            // (the side CTAs' SMs not the tiles')
    if (g_om_ctas_most > 0) nsm = std::min(nsm, g_om_ctas_most);
    const int nct = (int)std::min<long>(n, nsm);         // (the tiles' CTAs, after the side CTAs)
    g_last_grids[0] = nct; g_last_grids[1] = om_tile_group();
    launch_pdl(kernel, dim3((unsigned)(nct + n_v)), dim3(OM_THREADS), SMEM, st, params, pbase, vt,
               vrows, row0, shift, vals, ebase, cbase, jpos, ugrp, knt, mtiles, block_size, B, vnm, nids, num_edges, tmapA,
               fa, a_sb, node_mars, lz, nshift, vn0, (int)n, om_tile_group(), uaw, urows, amw, tmapV, tmapC, vfb, vf_sb,
               vout, vsh, vrow0, vX, n_v, nct, cmo);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Whether the online-max form serves a layer: rank OM_K, B a multiple of the tile's samples, a node tile of whole
// node blocks or a node block of whole tiles (the stream kernel's tables), the children in edge tiles of 64.
bool om_serves(int64_t K, int64_t B, int64_t block_size, int64_t rows) {
    return K == OM_K && B % (OM_BN / OM_K) == 0 && block_size % 32 == 0 &&
           (block_size % OM_BM == 0 || (OM_BM % block_size == 0 && (rows * block_size) % OM_BM == 0));
}

// V's rows (`exp(B - shift)` bf16, the rows (copy, sample, rank)) and their shifts of ONE copy -- its factors `fb`
// B [B, K, N] fp32 in place, its first row `row0` of `vt` [N, rows] -- by the one-launch forward's side-CTA code
// (`lowrank_mixture.cu`'s `inplace_operand_kernel`'s operations: the same bits). `om_v_serves(N, K)`: whether it (and
// the forward's side CTAs) serve N children at rank K.
bool om_v_serves(int64_t N, int64_t K) {
    return K == OM_K && N >= 1 && N <= OMV_MAXX * OM_THREADS && omv_smem((int)N) <= OM_SMEM;
}
void lowrank_v_rows(torch::Tensor vt, torch::Tensor shift, torch::Tensor fb, int64_t row0) {
    const long X = vt.size(0), rows = vt.size(1), B = fb.size(0);
    TORCH_CHECK(om_v_serves(X, fb.size(1)) && vt.is_cuda() && vt.scalar_type() == torch::kBFloat16 && vt.is_contiguous() &&
                rows % 8 == 0 && (reinterpret_cast<uintptr_t>(vt.data_ptr()) & 15) == 0 && fb.is_cuda() &&
                fb.scalar_type() == torch::kFloat32 && fb.dim() == 3 && fb.size(2) == X && fb.stride(2) == 1 &&
                fb.stride(1) == X && fb.stride(0) >= OM_K * X && fb.stride(0) % 4 == 0 &&
                (reinterpret_cast<uintptr_t>(fb.data_ptr()) & 15) == 0 && shift.is_cuda() && shift.is_contiguous() &&
                shift.scalar_type() == torch::kFloat32 && shift.numel() == rows && row0 >= 0 && row0 % (OM_K * B) == 0 &&
                row0 + OM_K * B <= rows, "lowrank v rows: B fp32 [B, K, N] in place (16-byte runs), V bf16 [N, rows]");
    if (B == 0) return;
    const at::cuda::OptionalCUDAGuard guard(vt.device());
    const int smem = (int)omv_smem((int)X);
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(om_v_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)OM_SMEM));
        if (dev < 64) attr_set[dev] = true;
    }
    const int grid = (int)std::min<long>(B, om_sms());
    om_v_kernel<<<grid, OM_THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
        fb.data_ptr<float>(), (long)fb.stride(0), reinterpret_cast<bfloat16_t*>(vt.data_ptr()), shift.data_ptr<float>(),
        rows, (long)row0, (int)X, (int)B);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// The sum_exp forward of a layer over the virtual batch in ONE launch (the online-max form), its normalizer (Z_r =
// theta V under V's shifts) and collapse beside: `vnm` [twin rows, B K] the layer's kept share differences (what the
// backward's dA takes); the tables `lowrank_forward_stream`'s; `vt` the normalizers' V bf16 [N, rows of V]
// (`lowrank_z_operand`'s, the rows (copy, sample, rank)), `shift` their row shifts, this copy's rows from `row0`;
// `node_mars` the node values at the node vector's circuit rows (the twin's + `nshift`), `lz` [M, B] log Z, `fa` the
// factors A [B, M, K] as the caller gives them (read in place: any batch stride, 16-byte aligned, a sample's block
// contiguous); `vn0` the node vector's first twin row. `a_p0` / `a_bstr`: the
// parameter slabs' bases `a_p0 + block a_bstr + edge tile 64 block_size` (`a_p0` -1: not so, or blocks not of 32 -- A by
// cp.async; else by TMA), checked by the caller once per table. `cmo` (or none): each (node tile, sample)'s largest
// child value over the tile's children, [node tiles, B] -- what the fused pass's end takes c_m from (`pe_tables`).
void lowrank_forward_om(torch::Tensor vnm, torch::Tensor vals, c10::optional<torch::Tensor> cbase, torch::Tensor ebase,
                        torch::Tensor jpos, torch::Tensor vt, int64_t row0, torch::Tensor shift, torch::Tensor params,
                        torch::Tensor nids, torch::Tensor pbase, torch::Tensor ugrp, int64_t block_size, int64_t knt_,
                        int64_t a_p0, int64_t a_bstr, torch::Tensor node_mars, torch::Tensor lz, torch::Tensor fa,
                        int64_t nshift, int64_t vn0, c10::optional<torch::Tensor> ua, c10::optional<torch::Tensor> am,
                        bool vtma, c10::optional<torch::Tensor> vnext, int64_t vrow0, int64_t n_v,
                        c10::optional<torch::Tensor> cmo) {
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
    for (const torch::Tensor* t : {&node_mars, &lz})
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat32,
                    "lowrank om: the collapse's tensors contiguous float32 CUDA");
    TORCH_CHECK(fa.is_cuda() && fa.scalar_type() == torch::kFloat32 && fa.dim() == 3 && fa.size(0) == B &&
                fa.size(1) == M && fa.size(2) == OM_K && fa.stride(2) == 1 && fa.stride(1) == OM_K &&
                fa.stride(0) >= M * OM_K && fa.stride(0) % 4 == 0 && (reinterpret_cast<uintptr_t>(fa.data_ptr()) & 15) == 0,
                "lowrank om: A fp32 [B, M, K], a sample's block contiguous, 16-byte aligned runs");
    TORCH_CHECK(lz.dim() == 2 && lz.size(1) == B && node_mars.dim() == 2 &&
                node_mars.size(1) == B && vn0 >= 0 && node_mars.size(0) >= vn0 + nshift + M && vnm.size(0) >= vn0 + M &&
                (reinterpret_cast<uintptr_t>(vnm.data_ptr()) & 15) == 0 &&
                (reinterpret_cast<uintptr_t>(shift.data_ptr()) & 15) == 0,
                "lowrank om: lz [M, B], the node rows, 16-byte aligned runs");
    TORCH_CHECK(ua.has_value() == am.has_value() &&
                (!ua.has_value() || (ua->is_cuda() && ua->is_contiguous() && ua->scalar_type() == torch::kBFloat16 &&
                                     ua->dim() == 2 && ua->size(0) == M && ua->size(1) == vt.size(1) &&
                                     am->is_cuda() && am->is_contiguous() && am->scalar_type() == torch::kFloat32 &&
                                     am->numel() == vt.size(1) / OM_K * M && row0 % cols == 0)),
                "lowrank om: the A operand's Ua bf16 [M, rows] and am float32 [copies, M, B], with V's rows");
    bfloat16_t* uaw = ua.has_value() ? reinterpret_cast<bfloat16_t*>(ua->data_ptr()) : nullptr;
    float* amw = am.has_value() ? am->data_ptr<float>() + row0 / OM_K * M : nullptr;
    const bool tma = a_p0 >= 0 && block_size == 32;
    // (V's boxes where a CTA takes more than one tile -- MEASURED (croissant, HMM 1024 x 31, rank 16, step ms), one a
    // CTA at B64: 2.76 -> 2.81 with them -- and not where the boxes are off, `set_om_boxes`)
    const long ntiles = (long)(cols / OM_BN) * ((long)rows * block_size / OM_BM);
    if (g_om_boxes == 2 || (g_om_boxes == 0 && ntiles <= om_sms())) vtma = false;
    g_last_grids[4] = vtma;
    // (`vtma`: the caller's word that every k-step's children are consecutive -- `jpos` -- and every run of values whole)
    TORCH_CHECK(!vtma || (vt.size(1) % 8 == 0 && (reinterpret_cast<uintptr_t>(vt.data_ptr()) & 15) == 0 &&
                          (reinterpret_cast<uintptr_t>(vals.data_ptr()) & 15) == 0 && B % 4 == 0),
                "lowrank om: V's and the values' rows for TMA (16-byte aligned)");
    const CUtensorMap tv = vtma ? om_map2d(vt.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, vt.size(1), vt.size(0),
                                           OM_BN, 32, CU_TENSOR_MAP_SWIZZLE_128B) : CUtensorMap{};
    const CUtensorMap tc = vtma ? om_map2d(vals.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, B, vals.size(0),
                                           OM_BN / OM_K, 32, CU_TENSOR_MAP_SWIZZLE_NONE) : CUtensorMap{};
    TORCH_CHECK(!tma || (a_bstr % 8 == 0 && a_p0 % 8 == 0), "lowrank om: the slabs' bases 16-byte aligned");
    const at::cuda::OptionalCUDAGuard guard(vnm.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    const CUtensorMap tm = tma ? om_tmap((const char*)pp + a_p0 * 2, block_size, knt, rows, a_bstr) : CUtensorMap{};
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    const auto* vtp = reinterpret_cast<const bfloat16_t*>(vt.data_ptr());
    const int* cb = fused ? cbase->data_ptr<int>() : nullptr;
    // (side CTAs forming another copy's V rows -- its factors `vnext` B [B, K, N] in place, its first row vrow0 --
    // beside the tiles)
    const int nv = vnext.has_value() && omv_smem((int)vt.size(0)) <= OM_SMEM ? (int)n_v : 0;   // (else: no side CTAs)
    if (nv > 0)
        TORCH_CHECK(nv < om_sms() && vnext->is_cuda() && vnext->scalar_type() == torch::kFloat32 && vnext->dim() == 3 &&
                    vnext->size(0) == B && vnext->size(1) == OM_K && vnext->size(2) == vt.size(0) &&
                    vnext->stride(2) == 1 && vnext->stride(1) == vnext->size(2) &&
                    vnext->stride(0) >= vnext->size(1) * vnext->size(2) && vrow0 >= 0 && vrow0 % ((long)OM_K * B) == 0 &&
                    vrow0 + (long)OM_K * B <= vt.size(1) && vt.size(1) % 8 == 0 &&
                    (reinterpret_cast<uintptr_t>(vt.data_ptr()) & 15) == 0,
                    "lowrank om: the side CTAs' copy: B fp32 [B, K, N] in place, its rows of V");
    const float* vfb = nv > 0 ? vnext->data_ptr<float>() : nullptr;
    const long vf_sb = nv > 0 ? (long)vnext->stride(0) : 0;
    bfloat16_t* vout = nv > 0 ? reinterpret_cast<bfloat16_t*>(vt.data_ptr()) : nullptr;
    if (cmo.has_value())
        TORCH_CHECK(cmo->is_cuda() && cmo->is_contiguous() && cmo->scalar_type() == torch::kFloat32 &&
                    cmo->numel() == ((long)rows * block_size / OM_BM) * B, "lowrank om: cmo float32 [node tiles, B]");
    float* cmw = cmo.has_value() ? cmo->data_ptr<float>() : nullptr;
#define LOWRANK_OM(F, T, VT)                                                                                          \
    launch_om<F, T, VT>(pp, pbase.data_ptr<long>(), vtp, (long)vt.size(1), (long)row0, shift.data_ptr<float>(),     \
                    vals.data_ptr<float>(), ebase.data_ptr<long>(), cb, jpos.data_ptr<int>(), ugrp.data_ptr<int>(),     \
                    knt, rows, (int)block_size, B, vnm.data_ptr<float>(), nids.data_ptr<long>(), num_edges, tm,      \
                    fa.data_ptr<float>(), (long)fa.stride(0), node_mars.data_ptr<float>(), lz.data_ptr<float>(),       \
                    (long)nshift, (long)vn0, uaw, (long)vt.size(1), amw, tv, tc, vfb, vf_sb, vout,           \
                    shift.data_ptr<float>(), (long)vrow0, (int)vt.size(0), nv, cmw, st)
    if (vtma) {
        if (fused) { if (tma) LOWRANK_OM(true, true, true); else LOWRANK_OM(true, false, true); }
        else { if (tma) LOWRANK_OM(false, true, true); else LOWRANK_OM(false, false, true); }
    } else {
        if (fused) { if (tma) LOWRANK_OM(true, true, false); else LOWRANK_OM(true, false, false); }
        else { if (tma) LOWRANK_OM(false, true, false); else LOWRANK_OM(false, false, false); }
    }
#undef LOWRANK_OM
}

// ---------------- the backward's one-launch form ----------------
// Whether it serves a layer's backward partition: rank OM_K, B a multiple of the tile's samples, the child tile of
// whole child blocks or inside one, the parents' blocks a multiple of the k-step
// The dA CTAs a launch at most (0: the launcher's share of the SMs): the tests make them few, so a CTA takes several
// of the pairs' tiles (the sizes they run at give a CTA one otherwise)
constexpr double OM_DA_KSTEPS = 12.0, OM_TILE_KSTEPS = 13.0;
static int g_da_ctas_most = 0;
void set_da_ctas_most(int64_t n) { g_da_ctas_most = (int)std::max<int64_t>(n, 0); }
// The backward's Ua map: [M, urows] bf16, a box OM_BN columns x 32 parents, the stages' 128-byte swizzle (Atom64)
static CUtensorMap om_bw_umap(const void* base, long urows, long M) {
    struct Entry { const void* base; long urows, M; CUtensorMap map; };
    static std::vector<Entry> kept;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    for (size_t i = 0; i < kept.size(); i++) {
        const Entry& e = kept[i];
        if (e.base == base && e.urows == urows && e.M == M) {
            const CUtensorMap m = e.map;
            if (i > 0) { const Entry hit = kept[i]; kept.erase(kept.begin() + i); kept.insert(kept.begin(), hit); }
            return m;
        }
    }
    Entry e{base, urows, M, {}};
    cuuint64_t gdim[2] = {(cuuint64_t)urows, (cuuint64_t)M};
    cuuint64_t gstride[1] = {(cuuint64_t)urows * 2};
    cuuint32_t box[2] = {(cuuint32_t)OM_BN, 32};
    cuuint32_t es[2] = {1, 1};
    const CUresult rr = cuTensorMapEncodeTiled(&e.map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim,
                                               gstride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rr == CUDA_SUCCESS, "lowrank om backward: cuTensorMapEncodeTiled (Ua) failed (", (int)rr, ")");
    kept.insert(kept.begin(), e);
    if (kept.size() > 16) kept.pop_back();
    return e.map;
}
// The backward's parameter tiles' map: the parameters as [rows, pbs] (a slab row's parents contiguous); a box the tile's
// OM_BM rows x 32 parents, the 64-byte swizzle CuTe's AtomAK32 reads. Kept a few (base, pbs, rows) at a time.
static CUtensorMap om_bw_tmap(const void* base, long pbs, long rows) {
    struct Entry { const void* base; long pbs, rows; CUtensorMap map; };
    static std::vector<Entry> kept;
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    for (size_t i = 0; i < kept.size(); i++) {
        const Entry& e = kept[i];
        if (e.base == base && e.pbs == pbs && e.rows == rows) {
            const CUtensorMap m = e.map;
            if (i > 0) { const Entry hit = kept[i]; kept.erase(kept.begin() + i); kept.insert(kept.begin(), hit); }
            return m;
        }
    }
    Entry e{base, pbs, rows, {}};
    cuuint64_t gdim[2] = {(cuuint64_t)pbs, (cuuint64_t)rows};
    cuuint64_t gstride[1] = {(cuuint64_t)pbs * 2};
    cuuint32_t box[2] = {32, (cuuint32_t)OM_BM};
    cuuint32_t es[2] = {1, 1};
    const CUresult rr = cuTensorMapEncodeTiled(&e.map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), gdim,
                                               gstride, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(rr == CUDA_SUCCESS, "lowrank om backward: cuTensorMapEncodeTiled failed (", (int)rr, ")");
    kept.insert(kept.begin(), e);
    if (kept.size() > 8) kept.pop_back();
    return e.map;
}

bool om_bw_serves(int64_t K, int64_t B, int64_t cbs, int64_t pbs, int64_t n_cb) {
    return K == OM_K && B % (OM_BN / OM_K) == 0 && pbs % 32 == 0 && cbs > 0 &&
           (cbs % OM_BM == 0 || (OM_BM % cbs == 0 && (n_cb * cbs) % OM_BM == 0));
}

// The data half of a layer's backward over the virtual batch in ONE launch (`lr_om_bw_kernel`): `flows` the element
// flows [rows, B] -- or, `table` given (the product layer below fused in: int32 [products, 2], from element row `first`
// on), the NODE flows the products' two children take --, `vals` the element values or (`table`) the node values;
// `fb` the factors B [K, N, B], `jrow` int32 the node vector's child of each element row from `e_lo` on; `grad_b` dB
// [K, N, B] (None: no gradient); `ua` the pass's exp(A - am) bf16 [M, rows] and `am` [copies, M, B] its (node, sample)
// shifts (`lowrank_a_operand`), this copy's from `row0`; the parents' `node_flows`, `node_mars` (circuit rows: the
// twin's + `nshift`) and
// `lz` [M, B] (node index: the twin's row - `vn0`); `params` bf16; the twin's backward partition: `chids` [blocks] the
// child blocks' first element rows (`cbs` rows a block), `ppar` / `ppid` int64 [blocks, parent blocks] each parent
// block's first twin row and parameter slab (`pbs` parents a block).
// `grad_a` given (with `grad_b`): dA too, by the grid's first CTAs (`lr_om_bw_kernel`'s dA CTAs), from `kd` the share
// differences the forward kept [twin rows, B K]. `grad_b` given: dB's data half AND its normalizer half (the pass's end
// then takes nothing off this copy's dB).
void lowrank_backward_om(torch::Tensor flows, torch::Tensor vals, c10::optional<torch::Tensor> table, int64_t first,
                         torch::Tensor fb, torch::Tensor jrow, int64_t e_lo, c10::optional<torch::Tensor> grad_b,
                         torch::Tensor ua, int64_t row0, torch::Tensor am, torch::Tensor node_flows,
                         torch::Tensor node_mars, torch::Tensor lz, int64_t nshift, int64_t vn0, torch::Tensor params,
                         torch::Tensor chids, torch::Tensor ppar, torch::Tensor ppid, int64_t cbs, int64_t pbs,
                         c10::optional<torch::Tensor> grad_a, c10::optional<torch::Tensor> kd, int64_t a_p0,
                         int64_t a_cstr, int64_t a_pstr, bool store) {
    const int B = (int)node_flows.size(1), K = (int)fb.size(1), N = (int)fb.size(2);
    const int n_cb = (int)chids.size(0), npb = (int)ppar.size(1);
    TORCH_CHECK(om_bw_serves(K, B, cbs, pbs, n_cb) && fb.dim() == 3 && fb.size(0) == B, "lowrank om backward: rank ", OM_K, ", B a "
                "multiple of ", OM_BN / OM_K, ", the parents' blocks of 32, the ", OM_BM, "-child tile dividing the child "
                "block or the blocks dividing the tile; got rank ", K, ", B ", B, ", blocks ", cbs, " / ", pbs);
    TORCH_CHECK(params.scalar_type() == torch::kBFloat16 && ua.scalar_type() == torch::kBFloat16,
                "lowrank om backward: bf16 parameters and exp(A)");
    for (const torch::Tensor* t : {&flows, &vals, &jrow, &ua, &am, &node_flows, &node_mars, &lz, &chids, &ppar, &ppid})
        TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "lowrank om backward: contiguous CUDA tensors");
    TORCH_CHECK(fb.is_cuda() && fb.stride(2) == 1 && fb.stride(1) == N && fb.stride(0) >= (long)K * N,
                "lowrank om backward: B [B, K, N] as the caller gives it, a sample's block contiguous");
    TORCH_CHECK(flows.scalar_type() == torch::kFloat32 && vals.scalar_type() == torch::kFloat32 &&
                fb.scalar_type() == torch::kFloat32 && jrow.scalar_type() == torch::kInt32 &&
                am.scalar_type() == torch::kFloat32 && node_flows.scalar_type() == torch::kFloat32 &&
                node_mars.scalar_type() == torch::kFloat32 && lz.scalar_type() == torch::kFloat32 &&
                chids.scalar_type() == torch::kInt64 && ppar.scalar_type() == torch::kInt64 &&
                ppid.scalar_type() == torch::kInt64, "lowrank om backward: dtypes");
    TORCH_CHECK(ppar.dim() == 2 && ppar.size(0) == n_cb && ppid.sizes() == ppar.sizes(),
                "lowrank om backward: ppar / ppid [child blocks, parent blocks]");
    TORCH_CHECK(ua.dim() == 2 && row0 >= 0 && row0 % ((long)K * B) == 0 && row0 + (long)K * B <= ua.size(1) &&
                ua.size(1) % 8 == 0 && am.numel() == ua.size(1) / K * ua.size(0) && lz.size(0) == ua.size(0) &&
                lz.size(1) == B && vals.size(1) == B && flows.size(1) == B,
                "lowrank om backward: exp(A) [M, rows] and its shifts [copies, M, B] holding this copy's from row0");
    const bool fused = table.has_value();
    if (fused)
        TORCH_CHECK(table->is_cuda() && table->is_contiguous() && table->scalar_type() == torch::kInt32 &&
                    table->dim() == 2 && table->size(1) == 2, "lowrank om backward: the products' table int32 [*, 2]");
    if (grad_b.has_value())
        TORCH_CHECK(grad_b->is_cuda() && grad_b->scalar_type() == torch::kFloat32 && grad_b->sizes() == fb.sizes() &&
                    grad_b->stride(2) == 1 && grad_b->stride(1) == N && grad_b->stride(0) >= (long)K * N,
                    "lowrank om backward: dB [B, K, N] as B, a sample's block contiguous");
    const bool da = grad_a.has_value();
    TORCH_CHECK(da == grad_b.has_value() && da == kd.has_value(), "lowrank om backward: dA's tensors with dB, all or none");
    const int M = (int)lz.size(0);
    if (da) {
        TORCH_CHECK(kd->is_cuda() && kd->is_contiguous() && kd->scalar_type() == torch::kFloat32 &&
                    (reinterpret_cast<uintptr_t>(kd->data_ptr()) & 15) == 0,
                    "lowrank om backward: the kept share differences contiguous float32 CUDA, 16-byte aligned");
        TORCH_CHECK(grad_a->is_cuda() && grad_a->scalar_type() == torch::kFloat32 && grad_a->dim() == 3 &&
                    grad_a->size(0) == B && grad_a->size(1) == M && grad_a->size(2) == K && grad_a->stride(2) == 1 &&
                    grad_a->stride(1) == K && grad_a->stride(0) >= (long)M * K && grad_a->stride(0) % 4 == 0 &&
                    (reinterpret_cast<uintptr_t>(grad_a->data_ptr()) & 15) == 0,
                    "lowrank om backward: dA [B, M, K] as the caller gives it, 16-byte aligned runs");
        TORCH_CHECK(kd->dim() == 2 && kd->size(1) == (long)K * B && kd->size(0) >= vn0 + M,
                    "lowrank om backward: dA's tensors do not fit the node vector");
    }
    const at::cuda::OptionalCUDAGuard guard(flows.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    constexpr int ST = OM_ST, S = OM_BN / OM_K;
    const int mtiles = OM_BM >= cbs ? 1 : (int)(cbs / OM_BM);
    const long n = (long)(OM_K * B / OM_BN) * ((long)n_cb * cbs / OM_BM);
    TORCH_CHECK(n < (1L << 31), "lowrank om backward: a grid past 2^31 blocks");
    if (n == 0) return;
    // the dA CTAs, first in the grid: the share of the SMs their work is of the launch's -- a dA tile (32 nodes x 32
    // samples: DRAM-bound) costs about OM_DA_KSTEPS k-steps of a GEMM tile, a GEMM tile its k-steps and OM_TILE_KSTEPS
    // more (its setup and epilogue) -- so the pairs are done as the tiles are: too few and the layer waits for them (a
    // cliff), more cost little. MEASURED (brezel, HMM L x 31 in blocks of 32, rank 16, 188 SMs, ms a step, the dA CTAs
    // forced; in brackets the grid's plain order, the dA CTAs in its last wave's free SMs or a wave of their own):
    // L 1024 at B1024 12 / 16 / 20 / 26 / 34 CTAs 35.22 / 33.81 / 33.74 / 33.82 / 33.88 [35.34], at B256 9.29 / 8.97 /
    // 9.01 / 8.99 / 9.16 [9.13]; L 512 at B1024 8 / 16 / 32 / 48 17.93 / 15.20 / 14.09 / 14.27 [14.30]; L 2048 at B512
    // 5 / 9 55.21 / 50.60 [51.12]; L 256 at B2048 24 / 40 / 56 / 80 13.95 / 13.42 / 13.44 / 13.47 [13.30: its last
    // wave's 104 free SMs took dA best], at B512 3.83 / 3.73 / 3.74 / 3.73 [3.75]. The share at L 1024 / 512 / 2048 /
    // 256: 22 / 32 / 14 / 42 CTAs.
    const int sms = om_sms();
    const long da_tiles = (((long)M + 31) / 32) * (((long)B + 31) / 32);
    const double da_w = (double)da_tiles * OM_DA_KSTEPS / ((double)n * (npb * (pbs / 32) + OM_TILE_KSTEPS));
    int n_da = !da ? 0 : (int)std::min<long>(da_tiles, std::max<long>(1, std::lround(sms * da_w / (1.0 + da_w))));
    if (n_da > 0 && g_da_ctas_most > 0) n_da = std::min(n_da, g_da_ctas_most);
    // (store: the gradients' first writer -- dB, and dA where given, every element of the node vector's written, not
    // added into: the destination was not zeroed)
    TORCH_CHECK(!store || grad_b.has_value(), "lowrank om backward: storing the gradients without dB");
    // the parameter tiles by TMA (`a_p0` >= 0: every slab's base `a_p0 + child block a_cstr + parent block a_pstr`, all
    // whole rows of pbs, the tile's child blocks one run where it spans several: `_om_bw_slabs`)
    const bool tma = a_p0 >= 0 && g_om_boxes != 2;
    g_last_grids[5] = tma;
    if (tma)
        TORCH_CHECK(a_p0 % pbs == 0 && a_cstr % pbs == 0 && a_pstr % pbs == 0 && (cbs >= OM_BM || a_cstr == cbs * pbs) &&
                    params.numel() / pbs < (1L << 31), "lowrank om backward: the parameter slabs not one box a k-step");
    const CUtensorMap tm = tma ? om_bw_tmap(pp, pbs, params.numel() / pbs) : CUtensorMap{};
    if (tma)
        TORCH_CHECK((reinterpret_cast<uintptr_t>(ua.data_ptr()) & 15) == 0 && ua.size(0) < (1L << 31),
                    "lowrank om backward: exp(A) 16-byte aligned (its TMA boxes)");
    const CUtensorMap tu = tma ? om_bw_umap(ua.data_ptr(), ua.size(1), ua.size(0)) : CUtensorMap{};
    auto go = [&](auto f, auto g, auto s, auto ta) {
        constexpr bool F = decltype(f)::value, G = decltype(g)::value, SG = decltype(s)::value, TA = decltype(ta)::value;
        constexpr int NU = G ? 2 : 1;                    // (as the kernel's: U; and Un with the gradients)
        constexpr int STAGE = (32 * OM_BM * 2 + NU * 32 * OM_BN * 2 + 4 * 32 * S * 4 + 1023) / 1024 * 1024;
        constexpr int SMEM = ST * STAGE + NU * OM_BN * 4 + NU * ST * S * 4 + 3 * ST * 8 + NU * ST * 4;
        static_assert(SMEM <= 101376, "past sm_120's 99 KB of shared memory a block");
        static_assert(!G || 32 * ((32 * OM_K + 4) + (32 + 1)) * 4 <= SMEM, "the dA CTAs' tiles in the shared memory");
        auto kernel = lr_om_bw_kernel<ST, F, G, SG, TA>;
        static bool attr_set[64] = {};
        int dev = 0;
        C10_CUDA_CHECK(cudaGetDevice(&dev));
        if (dev >= 64 || !attr_set[dev]) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
            if (dev < 64) attr_set[dev] = true;
        }
        if (da) { g_last_grids[2] = n_da; g_last_grids[3] = om_tile_group(); }
        launch_pdl(kernel, dim3((unsigned)(n + n_da)), dim3(OM_THREADS), SMEM, st, pp, ppid.data_ptr<long>(),
                   ppar.data_ptr<long>(), reinterpret_cast<const bfloat16_t*>(ua.data_ptr()), (long)ua.size(1),
                   (long)row0, am.data_ptr<float>() + row0 / K * ua.size(0), node_flows.data_ptr<float>(),
                   node_mars.data_ptr<float>(),
                   lz.data_ptr<float>(), (long)nshift, (long)vn0, chids.data_ptr<long>(), npb, (int)cbs, (int)pbs,
                   mtiles, B, vals.data_ptr<float>(), fused ? table->data_ptr<int>() : (const int*)nullptr,
                   (long)first, fb.data_ptr<float>(), (long)fb.stride(0), jrow.data_ptr<int>(), (long)e_lo, N,
                   flows.data_ptr<float>(), grad_b.has_value() ? grad_b->data_ptr<float>() : (float*)nullptr,
                   grad_b.has_value() ? (long)grad_b->stride(0) : 0L,
                   da ? kd->data_ptr<float>() : (const float*)nullptr, da ? (long)kd->size(1) : 0L,
                   da ? grad_a->data_ptr<float>() : (float*)nullptr, da ? (long)grad_a->stride(0) : 0L, M, (int)n, n_da, om_tile_group(),
                   tm, tma ? (int)(a_p0 / pbs) : 0, tma ? (int)(a_cstr / pbs) : 0, tma ? (int)(a_pstr / pbs) : 0, tu);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    using T_ = std::true_type;
    using F_ = std::false_type;
    auto go2 = [&](auto f, auto g, auto s) { if (tma) go(f, g, s, T_{}); else go(f, g, s, F_{}); };
    if (fused) {
        if (!grad_b.has_value()) go2(T_{}, F_{}, F_{});
        else if (store) go2(T_{}, T_{}, T_{});
        else go2(T_{}, T_{}, F_{});
    } else {
        if (!grad_b.has_value()) go2(F_{}, F_{}, F_{});
        else if (store) go2(F_{}, T_{}, T_{});
        else go2(F_{}, T_{}, F_{});
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_forward_stream", &lowrank_forward_stream,
          "the sum_exp forward over the virtual batch: the expand fused into the stream kernel's pre-pass");
    m.def("lowrank_forward_om", &lowrank_forward_om,
          "the sum_exp forward over the virtual batch in one launch: the operand formed in the GEMM, the maxima online");
    m.def("om_serves", &om_serves, "whether the one-launch form serves a layer (rank, batch, block size, node blocks)");
    m.def("om_v_serves", &om_v_serves, "whether the side CTAs (and lowrank_v_rows) form V's rows of N children at rank K");
    m.def("lowrank_v_rows", &lowrank_v_rows, "V's rows and their shifts of one copy, its factors read in place");
    m.def("lowrank_backward_om", &lowrank_backward_om,
          "the data half of the sum_exp backward over the virtual batch in one launch: element flows and dB's data part");
    m.def("set_da_ctas_most", &set_da_ctas_most, "(tests) the backward's dA CTAs a launch at most; 0: no cap");
    m.def("set_om_ctas_most", &set_om_ctas_most, "(tests) the forward's CTAs a launch at most; 0: an SM each");
    m.def("om_last_grids", &om_last_grids,
          "(tests) the last launches' grids: the forward's CTAs and group, the backward's dA CTAs and group, and "
          "whether the forward took V and the values by TMA, the backward its parameter tiles and Ua");
    m.def("set_om_boxes", &set_om_boxes,
          "(tests) the one-launch kernels' TMA boxes: 0 where they serve, 1 wherever they serve, 2 none");
    m.def("set_tile_group", &set_tile_group,
          "(tests) the sample tiles a group of the one-launch grids' order; 0: a quarter of the SMs");
    m.def("om_bw_serves", &om_bw_serves,
          "whether the backward's one-launch form serves a partition (rank, batch, child and parent block sizes, blocks)");
    m.def("configs", &configs, "(BM, BN, KT) per config id (the stream forward's)");
    m.def("fitting_configs", &fitting_configs, "the config ids that serve a layer (block_size, columns, grouped tiles)");
}
