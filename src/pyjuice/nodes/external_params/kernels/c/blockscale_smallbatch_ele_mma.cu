// SMALL-BATCH (batch < 64) ELEMENT-FLOW backward for the per-block multiplicative gate (`BlockScaleSumParams`) on tensor
// cores -- the gated twin of layer/kernels/c/smallbatch_ele_backward_mma.cu. A block = (child tile of BM rows of one
// element block, batch tile of BN, range of 64-parent tiles: whole parent blocks, or a slice of one -- 4 blocks of 1024
// parents would otherwise cap an HMM-4096 layer at 4 ranges, 128 CTAs).
//
// Per child m (its gate g(m) = m >> gate_sh within the element block) and sample b:
//     element_flows[m, b] = element_mars[m, b] + log sum_t phi[j(t), g(m), b] * S_t[m, b] * exp(M_t[b]),
//     S_t[m, b] = sum_{e in 64-parent tile t} params[m, e] * exp(lf[e, b] - M_t[b]),   lf = node_flows - node_mars
// (-inf where node_mars is, as the gated Triton kernel), M_t[b] the tile's own column maximum of lf. The gate is
// constant over a parent block's parents, so S_t is a plain tensor-core product; each tile is folded into a running
// (L, R) per (child, column) by an online log-sum-exp under G_t[g, b] = log phi[j(t), g, b] + M_t[b]. The tile's own
// stabilizer is what the Triton kernel does too: a range-wide one would underflow a block whose flows sit far below the
// range's maximum while its gate lifts it. The gate gradient is the same product:
//     d LL / d log phi[j, g, b] = sum_{m in g} sum_{t in j} S_t[m, b] exp(G_t[g, b] + element_mars[m, b]),
// each (child, UNIT, sample) share stored in `gpart` -- a unit a parent block, or a range where ranges slice the blocks
// (`upb` units a block) -- and `bs_ele_grad_reduce_kernel` summing a gate's children and a block's units in order
// (equal backwards give equal gradients).
//
// The producer warps stage per tile the parameters (K-major; bf16 by cp.async) and the exponentials -- the tile's column
// maxima a 3-step shuffle across the 8 threads that hold a column's 64 parents -- and G_t for the tile's gates; the MMA
// warps keep S, L, R in fp32. Operands as the regular twin: bf16 parameters exact with one-bf16 exponentials; fp32
// parameters both operands a bf16 hi + lo pair, three MMAs (one stage fewer, two at the least). A range writes its
// partial (L, R); `bs_ele_merge_kernel` (a programmatic dependent launch) combines the ranges.
//
// Contract: the layer's k-tiles are 64 parents, each tile's parents contiguous node rows over one parameter block
// (`ele_cuda_ok`, ptr_inc_step 1); parent blocks of block_size a multiple of 64 (a tile in one block); cs_block_size a
// multiple of BM; the gate's ch_block_size a power of two dividing cs_block_size, at most 16 gates a tile; LL, log-space
// flows, no tempering. The caller verifies.

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"
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
#define MAX_GATES 16            // child gates a tile (two a thread of a column's 8)

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

// the stages a config runs: one fewer (two at the least) for the split fp32 operands, whose stage is twice the bytes
template <int ST, bool SPL> struct BsEleStages { static constexpr int value = SPL ? (ST >= 4 ? ST - 1 : 2) : ST; };

template <int BM, int BN, int WM, int WN, int PW, int ST_, bool WGRAD, typename TP>
__global__ void __launch_bounds__((WM * WN + PW) * 32) bs_sbmma_ele_kernel(
        float* __restrict__ eflows, const float* __restrict__ emars, const float* __restrict__ nflows,
        const float* __restrict__ nmars, const TP* __restrict__ params, const float* __restrict__ ext,
        const long* __restrict__ chids, const long* __restrict__ ebase, const long* __restrict__ pbase,
        const long* __restrict__ gate, float* __restrict__ gpart, float* __restrict__ part,
        int batch, int BSK, int CS, int knt_all, int nblk, int split, int upb, long rows_total, int bpad, int gate_sh,
        int ng, long ext_base) {
    constexpr int NC = WM * WN * 32, NP = PW * 32, NTH = NC + NP;
    constexpr bool BF16 = std::is_same<TP, __nv_bfloat16>::value;
    constexpr bool SPL = !BF16;                                          // fp32 parameters: hi + lo operands
    constexpr int ST = BsEleStages<ST_, SPL>::value;
    static_assert(BN % 8 == 0 && BN <= 64 && 2 * ST <= 15, "columns; named barriers");
    const int mtiles = CS / BM;
    const int pid_b = blockIdx.x, pid_m = blockIdx.y, sp = blockIdx.z;
    const int eb = pid_m / mtiles, row_off = (pid_m % mtiles) * BM;
    const int b0 = pid_b * BN;
    const int tpb = BSK / BK;                                            // tiles a parent block
    const int knt = knt_all / split, k0 = sp * knt;                     // the range's tiles (whole blocks or in one)
    const long* ebt = ebase + (long)eb * knt_all + k0;
    const long* pbt = pbase + (long)eb * knt_all + k0;
    const long* gt = gate + (long)eb * nblk;                            // the element block's gate rows
    const int g0 = row_off >> gate_sh;                                   // the tile's first child gate
    const int tid = threadIdx.x;

    // smem per stage: A (BM, BK), B (BN, BK) K-major Swizzle<3,3,3> (SPL: then the lo halves' A and B); after the
    // stages, G_t [ng][BN] per stage
    auto swzK = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});
    auto sAl = tile_to_shape(swzK, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swzK, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int HALF = BM * BK * 2 + BN * BK * 2;
    constexpr int STAGE = SPL ? 2 * HALF : HALF;
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    auto stA2 = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + HALF); };
    auto stB2 = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + HALF + BM * BK * 2); };
    auto stG = [&](int s) { return (float*)(smem_raw + ST * STAGE) + s * ng * BN; };

    if (tid >= NC) {
        // ---------------- producers ----------------
        const int ptid = tid - NC;
        constexpr int CH = (BN * BK) / 8;                    // 8-parent chunks of the exponential tile, column-major
        static_assert(CH % NP == 0 && NP % 8 == 0, "a column's 8 chunks in 8 adjacent lanes of one warp");
        constexpr int PBC = CH / NP;
        constexpr int PA = (BM * BK) / 8 / NP;
        static_assert((BM * BK) / 8 % NP == 0, "params chunks per producer thread");
        float ra[BF16 ? 1 : PA][8];
        auto load_fp32 = [&](int t) {
            if constexpr (!BF16) {
                const TP* pc = params + pbt[t] + (long)row_off * BSK;
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) {
                    const int i = ptid + q * NP, m = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                    ParamLoad<TP>::load8(&pc[(long)m * BSK + e8], ra[q]);
                }
            }
        };
        float xv[PBC][8], lpv[PBC][2];
        auto load_x = [&](int t) {
            const long r0 = ebt[t];
            const long gb = gt[(k0 + t) / tpb];
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), kk = i % (BK / 8), e8 = kk * 8;
                const bool live = b0 + c < batch;
                const long off = (r0 + e8) * (long)batch + b0 + c;
                CUTE_UNROLL
                for (int j = 0; j < 8; j++)
                    xv[q][j] = live ? log_ratio(nflows[off + (long)j * batch], nmars[off + (long)j * batch]) : -INFINITY;
                CUTE_UNROLL
                for (int h = 0; h < 2; h++) {
                    const int g = kk + 8 * h;
                    lpv[q][h] = (live && gb >= 0 && g < ng) ? ext[(gb + ext_base + g0 + g) * (long)batch + b0 + c] : -INFINITY;
                }
            }
        };
        if constexpr (!BF16) load_fp32(0);
        load_x(0);
        auto issue = [&](int t) {
            const int s = t % ST;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
            float* sG = stG(s);
            if constexpr (BF16) {
                const TP* pc = params + pbt[t] + (long)row_off * BSK;
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) {
                    const int i = ptid + q * NP, m = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                    cpa16(&sAt(m, e8), &pc[(long)m * BSK + e8]);
                }
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
                if (t + 1 < knt) load_fp32(t + 1);
            }
            Tensor sBt2 = make_tensor(make_smem_ptr(stB2(s)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), kk = i % (BK / 8), e8 = kk * 8;
                float M = xv[q][0];
                CUTE_UNROLL
                for (int j = 1; j < 8; j++) M = fmaxf(M, xv[q][j]);
                M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 1));      // the column's 8 chunks: 8 adjacent lanes
                M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 2));
                M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 4));
                __align__(16) bfloat16_t u[8], u2[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; j++) {
                    const float x = (M == -INFINITY || xv[q][j] == -INFINITY) ? 0.f : __expf(xv[q][j] - M);
                    u[j] = static_cast<bfloat16_t>(x);
                    if constexpr (SPL) u2[j] = static_cast<bfloat16_t>(x - static_cast<float>(u[j]));
                }
                *(uint4*)&sBt(c, e8) = *(const uint4*)u;
                if constexpr (SPL) *(uint4*)&sBt2(c, e8) = *(const uint4*)u2;
                CUTE_UNROLL
                for (int h = 0; h < 2; h++) {
                    const int g = kk + 8 * h;
                    if (g < ng) sG[g * BN + c] = (M == -INFINITY || lpv[q][h] == -INFINITY) ? -INFINITY : lpv[q][h] + M;
                }
            }
            if (t + 1 < knt) load_x(t + 1);
        };
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();
            __threadfence_block();
            bar_arrive(1 + j % ST, NTH);
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
    Tensor tCrA2 = thr.partition_fragment_A(sAt0);
    Tensor tCrB2 = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, mma);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor tXrA2 = s2rA.get_thread_slice(tid).retile_D(tCrA2);
    Tensor tXrB2 = s2rB.get_thread_slice(tid).retile_D(tCrB2);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    Tensor tCrR = thr.partition_fragment_C(cC);
    constexpr int NE = decltype(size(tCrS))::value;
    float gacc[WGRAD ? NE : 1], em[WGRAD ? NE : 1], Gv[NE];
    int gofs[NE];                                         // this thread's elements' offsets into a stage's G
    const long ele0 = chids[eb] + row_off;
    CUTE_UNROLL
    for (int i = 0; i < NE; i++) {
        const int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        gofs[i] = (((row_off + m) >> gate_sh) - g0) * BN + b;
        tCrL(i) = 0.f; tCrR(i) = -INFINITY;
        if constexpr (WGRAD) {
            gacc[i] = 0.f;
            em[i] = (b0 + b < batch) ? emars[(ele0 + m) * (long)batch + b0 + b] : -INFINITY;
        }
    }
    for (int kt = 0; kt < knt; kt++) {
        const int s = kt % ST;
        bar_sync(1 + s, NTH);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt), tXrB);
        const float* sG = stG(s);
        CUTE_UNROLL
        for (int i = 0; i < NE; i++) Gv[i] = sG[gofs[i]];
        clear(tCrS);
        if constexpr (SPL) {
            Tensor sAt2 = make_tensor(make_smem_ptr(stA2(s)), sAl);
            Tensor sBt2 = make_tensor(make_smem_ptr(stB2(s)), sBl);
            copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt2), tXrA2);
            copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt2), tXrB2);
            cute::gemm(mma, tCrA2, tCrB, tCrS);                          // the small terms first: lo.hi, hi.lo, hi.hi
            cute::gemm(mma, tCrA, tCrB2, tCrS);
        }
        cute::gemm(mma, tCrA, tCrB, tCrS);
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);
        // the tile into the running (L, R) under its gates, and its gradient shares
        CUTE_UNROLL
        for (int i = 0; i < NE; i++) {
            const float G = Gv[i], S = tCrS(i);
            if (G == -INFINITY) continue;
            if (G > tCrR(i)) { tCrL(i) = fmaf(tCrL(i), __expf(tCrR(i) - G), S); tCrR(i) = G; }
            else tCrL(i) = fmaf(S, __expf(G - tCrR(i)), tCrL(i));
            if constexpr (WGRAD) if (em[i] != -INFINITY) gacc[i] = fmaf(S, __expf(G + em[i]), gacc[i]);
        }
        if constexpr (WGRAD) {
            if ((k0 + kt + 1) % tpb == 0 || kt + 1 == knt) {  // a parent block (or the range's slice of one) done
                const int u = upb > 1 ? sp : (k0 + kt) / tpb;   // its unit: the block, or the range slicing it
                const long nu = (long)nblk * upb;
                CUTE_UNROLL
                for (int i = 0; i < NE; i++) {
                    const int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
                    if (b0 + b < batch) gpart[(((long)eb * CS + row_off + m) * nu + u) * batch + b0 + b] = gacc[i];
                    gacc[i] = 0.f;
                }
            }
        }
    }
    CUTE_UNROLL
    for (int i = 0; i < NE; i++) {
        const int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (split == 1) {
            if (b0 + b < batch) {
                const long off = (ele0 + m) * (long)batch + b0 + b;
                float r = (tCrR(i) == -INFINITY || tCrL(i) <= 0.f) ? -INFINITY : logf(tCrL(i)) + tCrR(i);
                if (r != -INFINITY) r += emars[off];
                eflows[off] = r;
            }
        } else {
            const long idx = ((long)sp * rows_total + (long)eb * CS + row_off + m) * bpad + b0 + b;
            part[idx] = tCrL(i);
            part[(long)split * rows_total * bpad + idx] = tCrR(i);
        }
    }
}

// ---------------- the ranges' merge (split > 1): a thread per (element, column); a programmatic dependent launch ------
__global__ void bs_ele_merge_kernel(float* __restrict__ eflows, const float* __restrict__ emars, const float* __restrict__ part,
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
        for (int z = 0; z < split; z++) { const float mz = Mp[(long)z * stride]; if (mz != -INFINITY) L += Lp[(long)z * stride] * __expf(mz - M); }
        if (L > 0.f) r = logf(L) + M + emars[off];
    }
    eflows[off] = r;
}

// the gate gradient from the children's shares: a thread per (element block, parent block, gate, sample), its gate's
// children and the block's `upb` units summed in order, then added into the gate's row (one writer a row unless tied
// gates share storage)
__global__ void bs_ele_grad_reduce_kernel(float* __restrict__ grad_ext, const float* __restrict__ gpart,
                                          const long* __restrict__ gate, int neb, int cs, int nblk, int upb, int gate_cbs,
                                          int batch, long ext_base) {
    cudaGridDependencySynchronize();
    const int ngates = cs / gate_cbs;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)neb * nblk * ngates * batch) return;
    const int b = (int)(i % batch);
    long r = i / batch;
    const int g = (int)(r % ngates); r /= ngates;
    const int j = (int)(r % nblk);
    const int eb = (int)(r / nblk);
    const long gb = gate[(long)eb * nblk + j];
    if (gb < 0) return;
    float s = 0.f;
    const long nu = (long)nblk * upb;
    const float* p = gpart + (((long)eb * cs + (long)g * gate_cbs) * nu + (long)j * upb) * batch + b;
    // (in order, children then units; unrolled so the loads overlap -- one at a time it was ~10 us a layer, the adds'
    // order and so the bits unchanged)
    if (upb == 1) {
#pragma unroll 8
        for (int m = 0; m < gate_cbs; m++) s += p[(long)m * nu * batch];
    } else {
        for (int m = 0; m < gate_cbs; m++)
#pragma unroll 4
            for (int u = 0; u < upb; u++) s += p[((long)m * nu + u) * batch];
    }
    if (s != 0.f) atomicAdd(grad_ext + (gb + ext_base + g) * (long)batch + b, s);
}

#define SMEM_CAP (99 * 1024)                           // sm_120's dynamic shared memory a block
// a config's dynamic shared memory: its stages' tiles, then G_t of each stage for `ng` gates
template <int BM, int BN, int ST, bool SPL> static int cfg_smem(int ng) {
    constexpr int STE = BsEleStages<ST, SPL>::value;
    return STE * (BM * BK * 2 + BN * BK * 2) * (SPL ? 2 : 1) + STE * ng * BN * 4;
}

template <int BM, int BN, int WM, int WN, int PW, int ST, typename TP>
static void launch_cfg(torch::Tensor eflows, torch::Tensor emars, torch::Tensor nflows, torch::Tensor nmars,
                       torch::Tensor params, torch::Tensor ext, torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase,
                       torch::Tensor gate, torch::Tensor grad, torch::Tensor gpart, torch::Tensor scratch, int batch,
                       int bsk, int cs, int knt, int nblk, int split, int upb, int gate_sh, long ext_base) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr bool SPL = !std::is_same<TP, __nv_bfloat16>::value;
    const int ng = (BM >> gate_sh) > 0 ? (BM >> gate_sh) : 1;
    TORCH_CHECK(ng <= MAX_GATES, "blockscale tensor-core ele: at most ", MAX_GATES, " child gates a tile");
    const int smem = cfg_smem<BM, BN, ST, SPL>(ng);
    TORCH_CHECK(smem <= SMEM_CAP, "blockscale tensor-core ele: ", smem, " bytes of shared memory past the ", SMEM_CAP,
                " (`blockscale_sb_ele_mma_fits`)");
    const bool wgrad = grad.numel() > 0;
    auto kernel = wgrad ? bs_sbmma_ele_kernel<BM, BN, WM, WN, PW, ST, true, TP>
                        : bs_sbmma_ele_kernel<BM, BN, WM, WN, PW, ST, false, TP>;
    static int attr_bytes[2][64] = {};                   // the attribute as last set, per device (it grows with ng)
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || attr_bytes[wgrad][dev] < smem) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_bytes[wgrad][dev] = smem;
    }
    const int neb = (int)chids.size(0), total_m = neb * (cs / BM);
    const long rows_total = (long)neb * cs;
    const int btiles = (batch + BN - 1) / BN, bpad = btiles * BN;
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    float* part = nullptr;
    torch::Tensor own;
    if (split > 1) {
        const long need = 2L * split * rows_total * bpad;
        if (scratch.numel() >= need) part = scratch.data_ptr<float>();
        else { own = torch::empty({need}, eflows.options()); part = own.data_ptr<float>(); }
    }
    if (wgrad) TORCH_CHECK(gpart.numel() >= rows_total * (long)nblk * upb * batch, "blockscale tensor-core ele: gpart too small");
    dim3 grid(btiles, total_m, split);
    kernel<<<grid, NTH, smem, st>>>(eflows.data_ptr<float>(), emars.data_ptr<float>(), nflows.data_ptr<float>(),
                                    nmars.data_ptr<float>(), param_ptr<TP>(params), ext.data_ptr<float>(),
                                    chids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(),
                                    gate.data_ptr<long>(), wgrad ? gpart.data_ptr<float>() : nullptr, part, batch, bsk,
                                    cs, knt, nblk, split, upb, rows_total, bpad, gate_sh, ng, ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (split > 1) {
        const long n = rows_total * batch;
        cudaLaunchConfig_t cfgl = {};
        cfgl.gridDim = dim3((unsigned)((n + 255) / 256)); cfgl.blockDim = dim3(256); cfgl.stream = st;
        cfgl.attrs = attr; cfgl.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, bs_ele_merge_kernel, eflows.data_ptr<float>(),
                                          (const float*)emars.data_ptr<float>(), (const float*)part,
                                          (const long*)chids.data_ptr<long>(), split, rows_total, cs, batch, bpad));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (wgrad) {
        const int gcb = 1 << gate_sh;
        const long n = (long)neb * nblk * (cs / gcb) * batch;
        cudaLaunchConfig_t cfgl = {};
        cfgl.gridDim = dim3((unsigned)((n + 255) / 256)); cfgl.blockDim = dim3(256); cfgl.stream = st;
        cfgl.attrs = attr; cfgl.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, bs_ele_grad_reduce_kernel, grad.data_ptr<float>(),
                                          (const float*)gpart.data_ptr<float>(), (const long*)gate.data_ptr<long>(), neb, cs,
                                          nblk, upb, gcb, batch, ext_base));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// cfg -> (BM, BN, WM, WN, PW, ST), the stage count before the fp32 split's reduction
#define BS_ELE_MMA_CONFIGS(X) \
    X(0, 128, 16, 4, 1, 4, 3) X(1, 64, 16, 2, 1, 4, 4) X(2, 128, 32, 4, 1, 4, 3) X(3, 64, 32, 2, 1, 4, 4) \
    X(4, 128, 64, 4, 2, 4, 2) X(5, 64, 64, 2, 2, 4, 3)

// `ebase` / `pbase` int64 [element blocks, num_edges / 64] (a 64-parent tile's first parent / first parameter), `gate`
// int64 [element blocks, parent blocks] (the gate's staging row, -1 unconnected), `grad` the gate gradient (0 elements:
// none) and `gpart` its shares (>= rows x parent blocks x batch floats), `scratch` the ranges' partials (allocated here
// if smaller than 2 x split x rows x padded batch); `cfg` = id | split << 8: `split` equal ranges of the 64-parent tiles,
// each whole parent blocks or inside one (then `gpart` holds split / blocks units a block: >= rows x max(blocks, split)
// x batch floats).
void blockscale_sb_ele_backward_mma(torch::Tensor eflows, torch::Tensor emars, torch::Tensor nflows, torch::Tensor nmars,
                                    torch::Tensor params, torch::Tensor ext, torch::Tensor chids, torch::Tensor ebase,
                                    torch::Tensor pbase, torch::Tensor gate, torch::Tensor grad, torch::Tensor gpart,
                                    torch::Tensor scratch, int64_t batch, int64_t block_size, int64_t cs_block_size,
                                    int64_t num_edges, int64_t gate_cbs, int64_t ext_base, int64_t cfg) {
    const int split = cfg >> 8 ? (int)(cfg >> 8) : 1;
    cfg &= 255;
    const int b = (int)batch, bsk = (int)block_size, cs = (int)cs_block_size, ne = (int)num_edges, gcb = (int)gate_cbs;
    TORCH_CHECK(b >= 1 && b < 64, "blockscale tensor-core ele: batch in [1, 64)");
    TORCH_CHECK(bsk % BK == 0 && ne % bsk == 0, "blockscale tensor-core ele: parent blocks of whole 64-tiles, whole blocks");
    TORCH_CHECK(gcb > 0 && (gcb & (gcb - 1)) == 0 && cs % gcb == 0,
                "blockscale tensor-core ele: a power-of-two gate dividing the child block");
    const int knt = ne / BK, nblk = ne / bsk;
    const int tpb = bsk / BK, rk = knt / (split > 0 ? split : 1);
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && knt % split == 0 && (rk % tpb == 0 || tpb % rk == 0),
                "blockscale tensor-core ele: split 1 .. ", MAX_SPLIT, " equal ranges of whole parent blocks or inside one");
    const int upb = rk < tpb ? tpb / rk : 1;                                // units a parent block
    TORCH_CHECK(ebase.dim() == 2 && ebase.size(1) == knt && ebase.size(0) == chids.size(0) && pbase.sizes() == ebase.sizes()
                && ebase.is_contiguous() && pbase.is_contiguous() && gate.dim() == 2 && gate.size(0) == chids.size(0)
                && gate.size(1) == nblk && gate.is_contiguous(), "blockscale tensor-core ele: table shapes");
    int gate_sh = 0;
    while ((1 << gate_sh) < gcb) ++gate_sh;
    auto run = [&](auto tag) {
        using TP = decltype(tag);
        switch (cfg) {
#define BS_ELE_MMA_CASE(ID, BM_, BN_, WM_, WN_, PW_, ST_) case ID: \
            TORCH_CHECK(cs % BM_ == 0, "blockscale tensor-core ele: cs_block_size % BM"); \
            launch_cfg<BM_, BN_, WM_, WN_, PW_, ST_, TP>(eflows, emars, nflows, nmars, params, ext, chids, ebase, pbase, \
                gate, grad, gpart, scratch, b, bsk, cs, knt, nblk, split, upb, gate_sh, (long)ext_base); \
            break;
            BS_ELE_MMA_CONFIGS(BS_ELE_MMA_CASE)
#undef BS_ELE_MMA_CASE
            default: TORCH_CHECK(false, "blockscale tensor-core ele: invalid cfg ", cfg);
        }
    };
    if (params.scalar_type() == torch::kBFloat16) run(__nv_bfloat16{}); else run(float{});
}

// whether config `cfg` runs a child block of `cs_block_size` under gates of `gate_cbs` (its rows divide the block, its
// gates a tile within MAX_GATES, its shared memory within the cap) for fp32 (`fp32`) or bf16 parameters
bool blockscale_sb_ele_mma_fits(int64_t cfg, bool fp32, int64_t cs_block_size, int64_t gate_cbs) {
    if (gate_cbs <= 0 || (gate_cbs & (gate_cbs - 1)) || cs_block_size % gate_cbs) return false;
    switch (cfg) {
#define BS_ELE_MMA_FITS(ID, BM_, BN_, WM_, WN_, PW_, ST_) case ID: { \
            const int ng = BM_ / gate_cbs > 0 ? (int)(BM_ / gate_cbs) : 1; \
            if (cs_block_size % BM_ || ng > MAX_GATES) return false; \
            return (fp32 ? cfg_smem<BM_, BN_, ST_, true>(ng) : cfg_smem<BM_, BN_, ST_, false>(ng)) <= SMEM_CAP; }
        BS_ELE_MMA_CONFIGS(BS_ELE_MMA_FITS)
#undef BS_ELE_MMA_FITS
        default: return false;
    }
}

std::vector<std::vector<int64_t>> blockscale_sb_ele_mma_configs() {
    std::vector<std::vector<int64_t>> out;
#define BS_ELE_MMA_LIST(ID, BM_, BN_, WM_, WN_, PW_, ST_) out.push_back({BM_, BN_});
    BS_ELE_MMA_CONFIGS(BS_ELE_MMA_LIST)
#undef BS_ELE_MMA_LIST
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_sb_ele_backward_mma", &blockscale_sb_ele_backward_mma,
          "small-batch block-scale element backward on tensor cores (per-tile stabilizers, online merge under the gates)");
    m.def("blockscale_sb_ele_mma_configs", &blockscale_sb_ele_mma_configs, "(BM, BN) per config id");
    m.def("blockscale_sb_ele_mma_fits", &blockscale_sb_ele_mma_fits, "whether a config runs this block / gate / dtype");
}
