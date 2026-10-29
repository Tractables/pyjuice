// Small-batch (batch < 64) block-sparse sum-layer ELEMENT-FLOW backward on tensor cores, the element-flow twin of
// smallbatch_forward_sum_mma.cu: a block = (element tile of BM children, batch tile of BN, range of the element block's
// 64-parent tiles).
//
// Per element block eb (children chids[eb] + m) the flow is
//     element_flows[m, b] = element_mars[m, b] + log sum_e params[m, e] * exp(lf[e, b])
//     lf[e, b] = node_flows[par(e), b] - node_mars[par(e), b]        (-inf where node_mars is)
// with the parents in tiles of 64: tile t's parents are the contiguous rows ebase[eb, t] .., child m's parameters
// params[pbase[eb, t * PBT + h] + m * BSK + e'] for the parents e = h * 64 / PBT + e' (BSK the parents' block size; a
// tile over PBT = 1 parameter block, or 2 of 32).
//
// The block's PW producer warps first take each column's maximum of lf over the range (whole rows a thread), then
// stage per tile the bf16 parameter tile, K-MAJOR (cp.async; fp32 parameters through registers, rounded to bf16) and
// the bf16 exponentials exp(lf - M), K-major (8 consecutive parents of one column per 16-byte store); the WM x WN MMA
// warps accumulate in fp32. A range writes its partial (L, M); `merge_kernel` (a programmatic dependent launch: its
// launch overlaps the primary's tail) combines the ranges and adds element_mars; one range writes the flows itself.
// Columns past the batch are computed on -inf (terms 0) and never stored.
//
// Why: the small-batch CUDA element kernel (smallbatch_ele_backward.cu) gives a warp to each child and re-reads every
// parent's flows per child -- fine at a sample or two, not past (HMM-4096 layer: batch 1 37 us, 4 70, 8 189); Triton's
// kernels take 56-67 us a layer from batch 4 to 32; the stream / pre-pass tensor-core kernels tile the batch by 64.
//
// Numerically: fp32 accumulation, a two-pass log-sum-exp per range. bf16 parameters: the stored parameters exact, the
// exponentials one bf16 -- the bf16 stream copy's operands. fp32 parameters: BOTH operands a bf16 hi + lo pair (x = hi +
// lo, lo = bf16(x - hi)), three MMAs (hi.hi + hi.lo + lo.hi): ~16 mantissa bits, past the fp16 / TF32 operands of the
// other fp32-parameter element kernels. One bf16 each (7 bits) was below them: an HMM's flows over 16 steps at
// perturbation 32 drifted past 1e-2 relative (tests/model/flow_numerical_test.py) where Triton's TF32 stays inside.
// The split stage is twice the bytes, so it runs one stage fewer (two at the least) within the 99 KB.
//
// Contract: LL propagation, log-space flows, allow_modify_flows / allow_neg_flows / accumulate_ch_flows off, no
// tempering, tiles of 64 parents that are contiguous node rows, over one parameter block or two of 32 (`pbt`), BSK a
// multiple of 8 and the bases 8-aligned below fp32; cs_block_size a multiple of BM, or BM a multiple of it with every
// aligned run of BM / cs_block_size element blocks sharing its parents (`row_groups`); the caller verifies.

#include <torch/extension.h>
#include "param_load.cuh"
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <type_traits>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;
#define BK 64
#define MAX_SPLIT 16

__device__ __forceinline__ void bar_sync(int id, int count) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int count) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void cpa16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ float log_ratio(float nf, float nm) { return (nm == -INFINITY) ? -INFINITY : nf - nm; }

extern __shared__ char smem_raw[];

// PBT: parameter blocks a 64-parent tile spans (1, or 2 for parents in blocks of 32: an HCLT's) -- `pbase` holds PBT
// bases a tile, the h-th for its parents h * 64 / PBT ..
// the stages a config runs: one fewer (two at the least) for the split fp32 operands, whose stage is twice the bytes
template <int ST, bool SPL> struct EleStages { static constexpr int value = SPL ? (ST >= 4 ? ST - 1 : 2) : ST; };

template <int BM, int BN, int WM, int WN, int PW, int ST_, int PBT, typename TP>
__global__ void __launch_bounds__((WM * WN + PW) * 32) sbmma_ele_kernel(
        float* __restrict__ eflows, const float* __restrict__ emars, const float* __restrict__ nflows,
        const float* __restrict__ nmars, const TP* __restrict__ params, const long* __restrict__ chids,
        const long* __restrict__ ebase, const long* __restrict__ pbase, float* __restrict__ part,
        int batch, int BSK, int CS, int knt_all, int split, long rows_total, int bpad) {
    constexpr int NC = WM * WN * 32, NP = PW * 32, NTH = NC + NP;
    constexpr bool BF16 = std::is_same<TP, __nv_bfloat16>::value;
    constexpr bool SPL = !BF16;                                          // fp32 parameters: hi + lo operands
    constexpr int ST = EleStages<ST_, SPL>::value;
    static_assert(BN % 8 == 0 && BN <= 64 && 2 * ST + 1 <= 15, "columns; named barriers (15 is the producers' own)");
    static_assert(PBT == 1 || PBT == 2, "one or two parameter blocks a tile");
    // An element tile is BM children of one element block (CS >= BM) or, GROUPED, the CS-child blocks eb0 ..
    // eb0 + BM / CS - 1, which share their parents (the caller verifies: an HCLT's 32-element blocks of a variable)
    const bool grouped = CS < BM;
    const int mtiles = grouped ? 1 : CS / BM;
    const int pid_b = blockIdx.x, pid_m = blockIdx.y, sp = blockIdx.z;
    const int eb0 = grouped ? pid_m * (BM / CS) : pid_m / mtiles;
    const int row_off = grouped ? 0 : (pid_m % mtiles) * BM;
    const int b0 = pid_b * BN;
    const int k0 = (int)((long)sp * knt_all / split), k1 = (int)((long)(sp + 1) * knt_all / split);
    const int knt = k1 - k0;
    const long* ebt = ebase + (long)eb0 * knt_all + k0;                 // the range's tiles' first parents (shared)
    const int tid = threadIdx.x;
    auto row_ele = [&](int r) -> long {
        return grouped ? chids[eb0 + r / CS] + r % CS : chids[eb0] + row_off + r;
    };

    // smem: A (BM, BK) and B (BN, BK) both K-major, 64 consecutive k of one row per 128-byte smem row (a global row in
    // one smem row: the cp.async atom that streams at the full rate), Swizzle<3,3,3> -- with SPL the lo halves' A and B
    // after them; then the range maxima and the producer warps' partial maxima. No static shared memory (it would shift
    // this buffer off its alignment).
    auto swzK = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});
    auto sAl = tile_to_shape(swzK, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swzK, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int HALF = BM * BK * 2 + BN * BK * 2;
    constexpr int STAGE = SPL ? 2 * HALF : HALF;
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    auto stA2 = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + HALF); };                  // (SPL: the lo halves)
    auto stB2 = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + HALF + BM * BK * 2); };
    float* sM = (float*)(smem_raw + ST * STAGE);                        // [BN] the range maxima
    float* sMp = sM + BN;                                                // [PW x BN] the warps' partial maxima

    if (tid >= NC) {
        // ---------------- producers ----------------
        const int ptid = tid - NC;
        // the range maximum of lf per column: a thread takes whole parent ROWS (its BN columns, float4s where aligned)
        {
            const int ncols = min(BN, batch - b0);
            float mx[BN];
            CUTE_UNROLL
            for (int j = 0; j < BN; j++) mx[j] = -INFINITY;
            const bool vec = (ncols == BN) && ((batch & 3) == 0) && ((b0 & 3) == 0)
                             && ((((size_t)nflows) & 15) == 0) && ((((size_t)nmars) & 15) == 0);
            for (int i = ptid; i < knt * BK; i += NP) {
                const long row = ebt[i / BK] + (i % BK);
                const float* f = nflows + row * (long)batch + b0;
                const float* n = nmars + row * (long)batch + b0;
                if (vec) {
                    CUTE_UNROLL
                    for (int j4 = 0; j4 < BN / 4; j4++) {
                        const float4 vf = reinterpret_cast<const float4*>(f)[j4];
                        const float4 vn = reinterpret_cast<const float4*>(n)[j4];
                        mx[j4 * 4 + 0] = fmaxf(mx[j4 * 4 + 0], log_ratio(vf.x, vn.x));
                        mx[j4 * 4 + 1] = fmaxf(mx[j4 * 4 + 1], log_ratio(vf.y, vn.y));
                        mx[j4 * 4 + 2] = fmaxf(mx[j4 * 4 + 2], log_ratio(vf.z, vn.z));
                        mx[j4 * 4 + 3] = fmaxf(mx[j4 * 4 + 3], log_ratio(vf.w, vn.w));
                    }
                } else {
                    CUTE_UNROLL
                    for (int j = 0; j < BN; j++) if (j < ncols) mx[j] = fmaxf(mx[j], log_ratio(f[j], n[j]));
                }
            }
            CUTE_UNROLL
            for (int j = 0; j < BN; j++) {
                float v = mx[j];
                CUTE_UNROLL
                for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
                mx[j] = v;
            }
            const int pw = ptid / 32, pl = ptid % 32;
            CUTE_UNROLL
            for (int j0 = 0; j0 < BN; j0 += 32) {
                if (j0 + pl < BN) {
                    float v = -INFINITY;
                    CUTE_UNROLL
                    for (int j = 0; j < BN; j++) if (j == j0 + pl) v = mx[j];
                    sMp[pw * BN + j0 + pl] = v;
                }
            }
            bar_sync(15, NP);
            if (ptid < BN) {
                float m = -INFINITY;
                CUTE_UNROLL
                for (int w = 0; w < PW; w++) m = fmaxf(m, sMp[w * BN + ptid]);
                sM[ptid] = m;
            }
            bar_sync(15, NP);
        }
        constexpr int CH = (BN * BK) / 8;                                // exponential chunks (8 parents of a column)
        constexpr int PBC = (CH + NP - 1) / NP;
        float Mc[PBC];
        CUTE_UNROLL
        for (int q = 0; q < PBC; q++) { const int i = ptid + q * NP; Mc[q] = (i < CH) ? sM[i / (BK / 8)] : 0.f; }
        constexpr int PA = (BM * BK) / 8 / NP;                           // 8-parameter chunks per producer thread
        static_assert((BM * BK) / 8 % NP == 0, "params chunks per producer thread");
        // Each chunk (child row m, parents e8 .. e8 + 7) reads its parameter base from the table once a tile -- the
        // row's block, the tile's h-th base -- through `tq` (its table entry for tile 0 of the range; a tile is PBT
        // entries on), plus a fixed in-block offset `rt`; the next tile's bases are loaded one issue ahead (`pn`).
        const long* tq[PA];
        int rt[PA];
        CUTE_UNROLL
        for (int q = 0; q < PA; q++) {
            const int i = ptid + q * NP, m = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
            const int g = grouped ? m / CS : 0, rr = grouped ? m % CS : row_off + m;
            const int h = PBT > 1 ? e8 / (BK / PBT) : 0, ein = PBT > 1 ? e8 % (BK / PBT) : e8;
            tq[q] = pbase + ((long)(eb0 + g) * knt_all + k0) * PBT + h;
            rt[q] = rr * BSK + ein;
        }
        long pn[PA];
        auto load_bases = [&](int t) {
            CUTE_UNROLL
            for (int q = 0; q < PA; q++) pn[q] = tq[q][(long)t * PBT] + rt[q];
        };
        // fp32 parameters: the tile's 8-float chunks loaded a tile ahead, converted at the issue
        float ra[BF16 ? 1 : PA][8];
        auto load_fp32 = [&]() {
            if constexpr (!BF16) {
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) ParamLoad<TP>::load8(&params[pn[q]], ra[q]);
            }
        };
        // the log ratios of a tile's chunks, loaded a tile ahead
        float xv[PBC][8];
        auto load_x = [&](int t) {
            const long r0 = ebt[t];
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                if (i >= CH) continue;
                const bool live = b0 + c < batch;
                const long off = (r0 + e8) * (long)batch + b0 + c;
                CUTE_UNROLL
                for (int j = 0; j < 8; j++)
                    xv[q][j] = live ? log_ratio(nflows[off + (long)j * batch], nmars[off + (long)j * batch]) : -INFINITY;
            }
        };
        load_bases(0);
        if constexpr (!BF16) { load_fp32(); if (knt > 1) load_bases(1); }
        load_x(0);
        auto issue = [&](int t) {
            const int s = t % ST;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
            if constexpr (BF16) {
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) {
                    const int i = ptid + q * NP, m = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                    cpa16(&sAt(m, e8), &params[pn[q]]);
                }
                if (t + 1 < knt) load_bases(t + 1);
            } else {
                Tensor sAt2 = make_tensor(make_smem_ptr(stA2(s)), sAl);
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) {
                    const int i = ptid + q * NP, m = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                    __align__(16) bfloat16_t r[8], r2[8];
                    CUTE_UNROLL
                    for (int j = 0; j < 8; j++) {
                        r[j] = static_cast<bfloat16_t>(ra[q][j]);
                        r2[j] = static_cast<bfloat16_t>(ra[q][j] - static_cast<float>(r[j]));
                    }
                    *(uint4*)&sAt(m, e8) = *(const uint4*)r;
                    *(uint4*)&sAt2(m, e8) = *(const uint4*)r2;
                }
                if (t + 1 < knt) { load_fp32(); if (t + 2 < knt) load_bases(t + 2); }
            }
            // B[c, e] = bf16 exp(lf - M[c]) (SPL: its hi + lo): 8 consecutive parents of one column, one 16-byte store
            Tensor sBt2 = make_tensor(make_smem_ptr(stB2(s)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                if (i >= CH) continue;
                const float M = Mc[q];
                __align__(16) bfloat16_t u[8], u2[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; j++) {
                    const float x = (M == -INFINITY || xv[q][j] == -INFINITY) ? 0.f : __expf(xv[q][j] - M);
                    u[j] = static_cast<bfloat16_t>(x);
                    if constexpr (SPL) u2[j] = static_cast<bfloat16_t>(x - static_cast<float>(u[j]));
                }
                *(uint4*)&sBt(c, e8) = *(const uint4*)u;
                if constexpr (SPL) *(uint4*)&sBt2(c, e8) = *(const uint4*)u2;
            }
            if (t + 1 < knt) load_x(t + 1);
        };
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();
            __threadfence_block();
            bar_arrive(1 + j % ST, NTH);                                 // FULL: tile j staged
            if (j >= 1) {
                const int t = j - 1 + ST;
                if (t < knt) { bar_sync(1 + ST + (j - 1) % ST, NTH); issue(t); }
                cpa_commit();
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, mma);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor tCrA2 = thr.partition_fragment_A(sAt0);                       // (SPL: the lo halves)
    Tensor tCrB2 = thr.partition_fragment_B(sBt0);
    Tensor tXrA2 = s2rA.get_thread_slice(tid).retile_D(tCrA2);
    Tensor tXrB2 = s2rB.get_thread_slice(tid).retile_D(tCrB2);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);
    for (int kt = 0; kt < knt; kt++) {
        const int s = kt % ST;
        bar_sync(1 + s, NTH);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt), tXrB);
        if constexpr (SPL) {
            Tensor sAt2 = make_tensor(make_smem_ptr(stA2(s)), sAl);
            Tensor sBt2 = make_tensor(make_smem_ptr(stB2(s)), sBl);
            copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt2), tXrA2);
            copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt2), tXrB2);
            cute::gemm(mma, tCrA2, tCrB, tCrL);                          // the small terms first: lo.hi, hi.lo, hi.hi
            cute::gemm(mma, tCrA, tCrB2, tCrL);
        }
        cute::gemm(mma, tCrA, tCrB, tCrL);                               // L[m, b] += sum_e params[m, e] exp(lf[e, b] - M[b])
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        const int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        const float Mf = sM[b];
        if (split == 1) {
            if (b0 + b < batch) {
                const long off = row_ele(m) * (long)batch + b0 + b;
                float r = (Mf == -INFINITY || tCrL(i) <= 0.f) ? -INFINITY : (logf(tCrL(i)) + Mf);
                if (r != -INFINITY) r += emars[off];
                eflows[off] = r;
            }
        } else {
            const long idx = ((long)sp * rows_total + (long)eb0 * CS + row_off + m) * bpad + b0 + b;
            part[idx] = tCrL(i);
            part[(long)split * rows_total * bpad + idx] = Mf;
        }
    }
}

// ---------------- the ranges' merge (split > 1): a thread per (element, column) ----------------
// element_flows = element_mars + log sum_z L_z exp(M_z - M) + M over the ranges (-inf where every M_z is). `part`: L
// [split, rows, bpad] then M [split, rows, bpad]; row = element block x CS + m. A programmatic dependent launch.
__global__ void ele_merge_kernel(float* __restrict__ eflows, const float* __restrict__ emars, const float* __restrict__ part,
                                 const long* __restrict__ chids, int split, long rows_total, int CS, int batch, int bpad) {
    cudaGridDependencySynchronize();
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_total * batch) return;
    const long row = i / batch, stride = rows_total * (long)bpad;
    const int b = (int)(i - row * batch);
    const float* Lp = part + row * bpad + b;
    const float* Mp = Lp + (long)split * stride;
    float M = -INFINITY;
    for (int z = 0; z < split; z++) M = fmaxf(M, Mp[(long)z * stride]);
    const long off = (chids[row / CS] + row % CS) * (long)batch + b;
    float r = -INFINITY;
    if (M != -INFINITY) {
        float L = 0.f;
        for (int z = 0; z < split; z++) {
            const float mz = Mp[(long)z * stride];
            if (mz != -INFINITY) L += Lp[(long)z * stride] * __expf(mz - M);
        }
        if (L > 0.f) r = logf(L) + M + emars[off];
    }
    eflows[off] = r;
}

template <int BM, int BN, int WM, int WN, int PW, int ST, int PBT, typename TP>
static void launch_cfg(torch::Tensor eflows, torch::Tensor emars, torch::Tensor nflows, torch::Tensor nmars,
                       torch::Tensor params, torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase,
                       torch::Tensor scratch, int batch, int bsk, int cs, int knt, int split) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr bool SPL = !std::is_same<TP, __nv_bfloat16>::value;
    constexpr int smem = EleStages<ST, SPL>::value * (BM * BK * 2 + BN * BK * 2) * (SPL ? 2 : 1) + (BN + PW * BN) * 4;
    static_assert(smem <= 99 * 1024, "sm_120's 99 KB of dynamic shared memory");
    auto kernel = sbmma_ele_kernel<BM, BN, WM, WN, PW, ST, PBT, TP>;
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_set[dev] = true;
    }
    const int neb = (int)chids.size(0);
    const int total_m = cs >= BM ? neb * (cs / BM) : neb / (BM / cs);   // (grouped: whole groups)
    const long rows_total = (long)neb * cs;
    const int btiles = (batch + BN - 1) / BN, bpad = btiles * BN;
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    float* part = nullptr;
    if (split > 1) {
        const long need = 2L * split * rows_total * bpad;
        if (scratch.numel() < need) scratch = torch::empty({need}, eflows.options());
        part = scratch.data_ptr<float>();
    }
    dim3 grid(btiles, total_m, split);
    kernel<<<grid, NTH, smem, st>>>(eflows.data_ptr<float>(), emars.data_ptr<float>(), nflows.data_ptr<float>(),
                                    nmars.data_ptr<float>(), param_ptr<TP>(params), chids.data_ptr<long>(),
                                    ebase.data_ptr<long>(), pbase.data_ptr<long>(), part, batch, bsk, cs, knt, split,
                                    rows_total, bpad);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (split > 1) {
        const long n = rows_total * batch;
        cudaLaunchConfig_t cfgl = {};
        cfgl.gridDim = dim3((unsigned)((n + 255) / 256)); cfgl.blockDim = dim3(256); cfgl.dynamicSmemBytes = 0; cfgl.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfgl.attrs = attr; cfgl.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, ele_merge_kernel, eflows.data_ptr<float>(), (const float*)emars.data_ptr<float>(),
                                          (const float*)part, (const long*)chids.data_ptr<long>(), split, rows_total, cs,
                                          batch, bpad));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// cfg -> (BM, BN, WM, WN, PW, ST), within sm_120's 99 KB of shared memory
#define SBMMA_ELE_CONFIGS(X) \
    X(0, 128, 16, 4, 1, 4, 3) X(1, 64, 16, 2, 1, 4, 4) X(2, 128, 32, 4, 1, 4, 3) X(3, 64, 32, 2, 1, 4, 4) \
    X(4, 128, 16, 4, 1, 8, 3) X(5, 128, 64, 4, 2, 4, 3) X(6, 64, 64, 2, 2, 4, 4)

// `cfg`: the tile config's id in the low byte, the split-K range count above it. `ebase` / `pbase`: int64
// [element blocks, num_edges / 64]. `scratch`: a float buffer for a split's partials, reused when large enough.
void smallbatch_ele_backward_sum_mma(torch::Tensor eflows, torch::Tensor emars, torch::Tensor nflows, torch::Tensor nmars,
                                     torch::Tensor params, torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase,
                                     torch::Tensor scratch, int64_t batch, int64_t block_size, int64_t cs_block_size,
                                     int64_t num_edges, int64_t pbt, int64_t cfg) {
    const int split = cfg >> 8 ? (int)(cfg >> 8) : 1;
    cfg &= 255;
    TORCH_CHECK(num_edges % BK == 0, "smallbatch_ele_backward_sum_mma: num_edges must be whole tiles of ", BK);
    const int knt = (int)(num_edges / BK);
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= knt, "smallbatch_ele_backward_sum_mma: split 1 .. ",
                MAX_SPLIT, ", at most the ", knt, " parent tiles; got ", split);
    TORCH_CHECK(batch >= 1 && batch < 64, "smallbatch_ele_backward_sum_mma: batch in [1, 64)");
    TORCH_CHECK(pbt == 1 || (pbt == 2 && block_size * 2 == BK), "smallbatch_ele_backward_sum_mma: pbt 1, or 2 with "
                "parents in blocks of ", BK / 2);
    TORCH_CHECK(ebase.dim() == 2 && ebase.size(0) == chids.size(0) && ebase.size(1) == knt && ebase.is_contiguous()
                && pbase.dim() == 2 && pbase.size(0) == chids.size(0) && pbase.size(1) == knt * pbt && pbase.is_contiguous(),
                "smallbatch_ele_backward_sum_mma: ebase [element blocks, ", knt, "] and pbase [element blocks, ", knt * pbt,
                "] contiguous tables");
    TORCH_CHECK(block_size % 8 == 0, "smallbatch_ele_backward_sum_mma: the parents' block size must be a multiple of 8");
    int b = (int)batch, bsk = (int)block_size, cs = (int)cs_block_size;
    auto run = [&](auto tag) {
        using TP = decltype(tag);
        switch (cfg) {
#define SBMMA_ELE_CASE(ID, BM_, BN_, WM_, WN_, PW_, ST_) case ID: \
            TORCH_CHECK(cs % BM_ == 0 || (BM_ % cs == 0 && chids.size(0) % (BM_ / cs) == 0), \
                        "smallbatch_ele_backward_sum_mma: cs_block_size % BM, or BM % cs_block_size over whole groups"); \
            if (pbt == 2) launch_cfg<BM_, BN_, WM_, WN_, PW_, ST_, 2, TP>(eflows, emars, nflows, nmars, params, chids, ebase, \
                                                                          pbase, scratch, b, bsk, cs, knt, split); \
            else launch_cfg<BM_, BN_, WM_, WN_, PW_, ST_, 1, TP>(eflows, emars, nflows, nmars, params, chids, ebase, pbase, \
                                                                 scratch, b, bsk, cs, knt, split); \
            break;
            SBMMA_ELE_CONFIGS(SBMMA_ELE_CASE)
#undef SBMMA_ELE_CASE
            default: TORCH_CHECK(false, "smallbatch_ele_backward_sum_mma: invalid cfg ", cfg);
        }
    };
    if (params.scalar_type() == torch::kBFloat16) run(__nv_bfloat16{}); else run(float{});
}

std::vector<std::vector<int64_t>> smallbatch_ele_mma_configs() {
    std::vector<std::vector<int64_t>> out;
#define SBMMA_ELE_LIST(ID, BM_, BN_, WM_, WN_, PW_, ST_) out.push_back({BM_, BN_});
    SBMMA_ELE_CONFIGS(SBMMA_ELE_LIST)
#undef SBMMA_ELE_LIST
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallbatch_ele_backward_sum_mma", &smallbatch_ele_backward_sum_mma,
          "small-batch (batch<64) block-sparse sum-layer element-flow backward on tensor cores (CuTe), split-K ranges merged");
    m.def("smallbatch_ele_mma_configs", &smallbatch_ele_mma_configs, "(BM, BN) per config id");
}
