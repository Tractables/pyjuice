// CUDA (CuTe/CUTLASS) block-sparse sum-layer forward for bf16 parameters as a pre-pass, one split GEMM launch and a
// merge -- what the tlmm kernels compute (tlmm_forward_sum.cu and its pipelined and pre-pass copies), not to their bits:
//
//     node_mars[n, b] = log sum_c params[n, c] exp(element_mars[c, b])
//
// The regular copy of the BlockScale forward stream (nodes/external_params/kernels/c/blockscale_forward_stream.cu),
// without gates and the normalizer. The other kernels walk a node block's edge tiles serially, each tile under its own
// stabilizer, merging the running sums tile by tile (MEASURED HMM-4096 B128: 128 thread blocks on 188 SMs, a tile every
// ~1600 cycles, the forward at a quarter of its compute floor). Here:
//   pre-pass  per (group, split, PB columns): a node block's children are cut into `split` contiguous ranges of
//             k-steps; over a range, the children's values' max Mp and u = bf16 exp(em - Mp) (0 where Mp is -inf), laid
//             out [group, child, b]. Node blocks of equal children (an HMM's blocks, an HCLT's 32 blocks of a variable)
//             form a GROUP and share them. One stabilizer per range makes the GEMM a plain one;
//   GEMM      per (batch tile, node tile, split): L_p = params[tile, range] . u, bf16 MMA in fp32 over the whole range
//             -- producer warps cp.async the parameters and u into mbarrier'd stages from offsets computed once per
//             thread, the consumers ldmatrix and MMA, no per-tile merges -- stored as the range's partial;
//   merge     per 4 columns of a node: L = sum_p L_p exp(Mp - M), M = max_p Mp; node_mars = log L + M (-inf where M is).
// Every sum is in a fixed order: the same bits from run to run.
//
// Scope (`fitting_configs`, and the Python dispatch for the rest): bf16 parameters with 8-aligned runs (the tlmm gate's
// contiguous children per tile and block_size-strided parameters per edge), batch % 64 == 0, block_size % 64 == 0 and a
// multiple of the tile's rows, split 1 .. 8. Scratch: one allocation per call (inside a CUDA graph capture, from its
// pool).

#include <torch/extension.h>
#include "param_load.cuh"
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

#define BK 64           // edge tile: one run of children (ebase / pbase one entry each)
#define CH 32           // the pre-pass's chunk: half an edge tile
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

// ---------------- pre-pass: a thread block per (group, split, PB columns) ----------------
// Thread (part, b) takes the range's 32-child chunks part, part + PARTS, ...: a chunk is one run of `element_mars` rows
// (half an edge tile, the tile's first row from `ebase` of the group's first node block `grep`); the range's max over
// every chunk (shared memory); u = bf16 exp(em - M). A range past PARTS chunks loads its later chunks twice (the max,
// then the exponentials).
template <int NT, int PB>
__global__ void __launch_bounds__(NT) stream_prepass_kernel(
        const float* __restrict__ em, const long* __restrict__ ebase, const int* __restrict__ grep,
        __nv_bfloat16* __restrict__ u, float* __restrict__ mp, int batch, int num_edges, int knt, int ksteps, int kt_sh,
        int split) {
    constexpr int PARTS = NT / PB;
    __shared__ float smx[PARTS][PB];
    const int tid = threadIdx.x, nbc = batch / PB;
    const int g = blockIdx.x / (split * nbc), sp = (blockIdx.x / nbc) % split, b0 = (blockIdx.x % nbc) * PB;
    const int b = tid % PB, part = tid / PB;
    const int q0 = (int)((((long)sp * ksteps / split) << kt_sh) / CH);
    const int q1 = (int)((((long)(sp + 1) * ksteps / split) << kt_sh) / CH);
    const long* et = ebase + (long)grep[g] * knt;
    auto load_chunk = [&](int q, float* v) {
        const int pos0 = q * CH;
        const float* er = em + (et[pos0 / BK] + pos0 % BK) * (long)batch + b0 + b;
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) v[i] = er[(long)i * batch];
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
    __nv_bfloat16* ur = u + ((long)g * num_edges) * batch + b0 + b;
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
    if (part == 0) mp[((long)g * split + sp) * batch + b0 + b] = m;
}

extern __shared__ __align__(1024) char smem_raw[];

// The MN-major smem atom of an operand with 64 rows (m or b) a k: 128-byte lines, swizzled over the 16-byte chunks so
// the transposed ldmatrix's 8 k-rows of one chunk fall in distinct banks.
using Atom64 = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));

// ---------------- the GEMM: (batch tile, node tile, split) blocks ----------------
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
__global__ void __launch_bounds__((WM * WN + PW) * 32) stream_gemm_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ pbase, const bfloat16_t* __restrict__ u,
        const int* __restrict__ ugrp, float* __restrict__ part, int knt, int mtiles, int block_size, int batch,
        int num_edges, int split, long rows_total) {
    constexpr int NC = WM * WN * 32, NP = PW * 32;
    constexpr int ABYTES = KT * BM * 2, BBYTES = KT * BN * 2;
    constexpr int STAGE = (ABYTES + BBYTES + 1023) / 1024 * 1024;
    constexpr int SUB = BK / KT;                         // k-steps an edge tile is
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
        const long* prA[CA];                             // a chunk's node block's parameter table (its slab of the tile)
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
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
                int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - smem_raw) : ~0u;
                gB[q] = e * batch + bb;
            }
        }
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        const bfloat16_t* ug = u + ((long)ugrp[r] * num_edges) * batch + (long)pid_b * BN;
        for (int t = k0; t < k1; t++) {
            const int j = t - k0, s = j % ST;
            if (j >= ST) mbar_wait(&empty[s], ((j / ST) - 1) & 1);
            const uint32_t ss = sbase + s * STAGE;
            const int t64 = t / SUB, sub = t % SUB;
            const long psub = (long)sub * KT * block_size;
            const bfloat16_t* pb = ug + (long)t * KT * batch;
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) if (oA[q] != ~0u) cpa16(ss + oA[q], params + prA[q][t64] + psub + gA[q]);
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
__global__ void __launch_bounds__(256) stream_merge_kernel(
        float* __restrict__ node_mars, const float* __restrict__ part, const float* __restrict__ mp,
        const long* __restrict__ nids, const int* __restrict__ ugrp, int batch, int block_size, long rows_total) {
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
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
static void launch_gemm(const bfloat16_t* params, const long* pbase, const bfloat16_t* u, const int* ugrp, float* part,
                        int knt, int rows, int block_size, int batch, int num_edges, int split, cudaStream_t st) {
    constexpr int STAGE = (KT * BM * 2 + KT * BN * 2 + 1023) / 1024 * 1024;
    constexpr int SMEM = ST * STAGE + 2 * ST * 8;
    constexpr int NTH = (WM * WN + PW) * 32;
    static_assert(SMEM <= 101376, "a config past sm_120's 99 KB of shared memory a block");
    auto kernel = stream_gemm_kernel<BM, BN, WM, WN, PW, ST, KT>;
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
    TORCH_CHECK(n_gemm < (1L << 31), "sum forward stream: a grid past 2^31 blocks");
    kernel<<<(unsigned)n_gemm, NTH, SMEM, st>>>(params, pbase, u, ugrp, part, knt, mtiles, block_size, batch, num_edges,
                                                 split, (long)rows * block_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// id -> {BM, BN, WM, WN, PW, ST, KT}: the BlockScale forward stream's table (MEASURED there: the 128 x 128 tiles take
// B >= 128 at most shapes -- 64-row k-steps in 3 stages, or 32-row ones in 6 --, 128 x 64 B 64 and the 512-node blocks,
// 64 x 128 the 256-node blocks at B 512, 64 x 64 the 64-node blocks at a batch that is no multiple of 128)
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

// The tlmm kernels' tables (`nids` [rows], `ebase` / `pbase` [rows, knt]: an edge tile's first child row and node 0's
// parameter), `ugrp` int32 [rows] (each node block's group: blocks of equal children share the pre-pass), `grep` int32
// [groups] (a group's first node block), `cfg` = config id * 16 + split.
void tlmm_forward_sum_stream(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                             torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase, torch::Tensor ugrp,
                             torch::Tensor grep, int64_t batch_, int64_t block_size, int64_t knt_, int64_t cfg) {
    const int batch = (int)batch_, knt = (int)knt_, rows = (int)nids.size(0), ngrp = (int)grep.size(0);
    const int num_edges = knt * BK;
    const int cfg_id = (int)(cfg / 16), split = (int)(cfg % 16);
    const auto cfgs = configs();
    TORCH_CHECK(cfg_id >= 0 && cfg_id < (int)cfgs.size(), "sum forward stream: no config ", cfg_id);
    const int BMc = (int)cfgs[cfg_id][0], BNc = (int)cfgs[cfg_id][1], kt = (int)cfgs[cfg_id][2];
    const int ksteps = (int)(num_edges / kt), kt_sh = kt == 64 ? 6 : 5;
    TORCH_CHECK(params.scalar_type() == torch::kBFloat16, "sum forward stream: bf16 parameters");
    TORCH_CHECK(batch % 64 == 0 && block_size % 32 == 0 && batch % BNc == 0 &&
                (block_size % BMc == 0 || (BMc % block_size == 0 && ((long)rows * block_size) % BMc == 0)),
                "sum forward stream: batch % 64, block_size % 32, the config's tile dividing the node block (or the "
                "block the tile, whole tiles of blocks) and the batch");
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= ksteps, "sum forward stream: split 1 .. ", MAX_SPLIT,
                " and at most the k-steps (", ksteps, "); got ", split);
    TORCH_CHECK(element_mars.is_contiguous() && node_mars.is_contiguous() && nids.is_contiguous() &&
                pbase.is_contiguous() && ebase.is_contiguous() && ugrp.is_contiguous() && grep.is_contiguous(),
                "sum forward stream: contiguous tensors");
    TORCH_CHECK(ebase.numel() == (long)rows * knt && pbase.numel() == ebase.numel() && ugrp.numel() == rows &&
                ugrp.scalar_type() == torch::kInt32 && grep.scalar_type() == torch::kInt32 && ngrp >= 1,
                "sum forward stream: ebase / pbase one entry per (node block, edge tile); ugrp int32 [rows], grep int32");
    const at::cuda::OptionalCUDAGuard guard(node_mars.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    // scratch, ONE allocation: u bf16 [groups, num_edges, B], Mp [groups, split, B], the partials [split, rows * bs, B]
    // (every region 16-byte aligned: B % 64 == 0)
    const long o_mp = (long)ngrp * num_edges * batch * 2;
    const long o_part = o_mp + (long)ngrp * split * batch * 4;
    const long total = o_part + (long)split * rows * block_size * batch * 4;
    torch::Tensor scratch = torch::empty({total}, element_mars.options().dtype(torch::kUInt8));
    char* base = reinterpret_cast<char*>(scratch.data_ptr());
    auto* u = reinterpret_cast<__nv_bfloat16*>(base);
    auto* mp = reinterpret_cast<float*>(base + o_mp);
    auto* part = reinterpret_cast<float*>(base + o_part);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    {
        constexpr int NT = 512, PB = 16;
        stream_prepass_kernel<NT, PB><<<ngrp * split * (batch / PB), NT, 0, st>>>(
            element_mars.data_ptr<float>(), ebase.data_ptr<long>(), grep.data_ptr<int>(), u, mp, batch, num_edges, knt,
            ksteps, kt_sh, split);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    switch (cfg_id) {
#define X(id, bm, bn, wm, wn, pw, stg, ktt) \
        case id: launch_gemm<bm, bn, wm, wn, pw, stg, ktt>(pp, pbase.data_ptr<long>(), (const bfloat16_t*)u, \
                     ugrp.data_ptr<int>(), part, knt, rows, (int)block_size, batch, num_edges, split, st); break;
        CFG_TABLE(X)
#undef X
    }
    {
        const long rt = (long)rows * block_size, n = rt * (batch / 4);
        const unsigned grid = (unsigned)((n + 255) / 256);
        float* nm = node_mars.data_ptr<float>();
        const long* np = nids.data_ptr<long>();
        const int* ug = ugrp.data_ptr<int>();
        switch (split) {
#define M_(S) case S: stream_merge_kernel<S><<<grid, 256, 0, st>>>(nm, part, mp, np, ug, batch, (int)block_size, rt); \
                      break;
            M_(1) M_(2) M_(3) M_(4) M_(5) M_(6) M_(7) M_(8)
#undef M_
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tlmm_forward_sum_stream", &tlmm_forward_sum_stream,
          "block-sparse sum-layer forward (bf16 params) as a pre-pass, a split GEMM and a merge");
    m.def("configs", &configs, "(BM, BN, KT) per config id");
    m.def("fitting_configs", &fitting_configs, "the config ids that serve a layer (block_size, batch, grouped tiles)");
}
