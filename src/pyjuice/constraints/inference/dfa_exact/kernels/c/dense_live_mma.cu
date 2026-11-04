// The dfa_exact backend's dense live-column products (`kernels/sum.py`: `_dense_sum_live`, the live branch of
// `dense_sum_top_down`) at a bf16 query, fused on tensor cores for narrow steps -- fewer live columns than a 64-column
// tile, the decode's common case (an HMM-4096 diffusion session: 83% of its dense steps). Per output row m < M and live
// lane k < L (column perm[k] of x, [K, *] fp32 with row stride x_ld):
//     P[m, k] = sum_{e < K} A[m, e] * exp(x[e, perm[k]] - M_k),
// A [M, K] the weights' bf16 copy, K-major (A[m * a_ld + e]) or MN-major (A[e * a_ld + m]), M_k the column's maximum (over
// the block's range of K, or over K); then for every column c < ncols of the output rows (row stride out_ld), k = rank[c]:
//     live (k >= 0):  v = log P[m, k] + M_k, merged over the ranges;  MODE 0 / 1: out = v;  MODE 2: out = logaddexp(out, v)
//     dead (k < 0):   MODE 0: -inf;  MODE 1 (keep) and 2 (add): left as they are
// -- what the chain it replaces computes (`_live_shifted_exp` at bf16, `torch.mm(..., out_dtype = float32)`,
// `_log_scatter_kernel` / `_log_scatter_accumulate_kernel`).
//
// TWO FORMS, both arms of the caller's measured choice (a shape's fastest is the GPU's to say):
//   FORM 0 (`dl_gather_kernel`, one launch and the finalize): a block = (column tile of BN lanes, row tile of BM rows of
//   A, range of the K tiles of 64: grid z) gathers its range's columns of x itself -- its PW producer warps take the
//   range's maxima per lane, stage the bf16 weights (cp.async, zero-filled past M and K) and the bf16 exponentials
//   exp(x - M), both gathers by a warp adjacent lanes of one row (a run of live lanes reads whole sectors) --; the MMA
//   warps accumulate in fp32 and write the range's partial. Every row tile gathers the same columns: cheap at one
//   16-lane tile (MEASURED 16 us a step at 4096 x 4096, the gather 5), dear at 64 lanes (41 us, the gather 18, the
//   exponentials 4).
//   FORM 1 (`dl_max_kernel`, `dl_exp_kernel`, `dl_split_kernel`, the finalize): the lanes' maxima over K (a block 32
//   lanes x a chunk of 128 rows, then over the chunks) and the bf16 exponentials computed ONCE into `ex` [Lpad, Kpad]
//   (K-major: a lane's K contiguous, 0 past K and past the live lanes); the split kernel's producers only copy -- the
//   weights from the start, the exponential tiles once the exp kernel is done (`cudaGridDependencySynchronize`: a
//   programmatic dependent launch, so the weights' first stages stream under the prep) --, its MMA warps load the next
//   tile's fragments while this one multiplies and give a stage back as soon as its fragments are in registers.
//   MEASURED 22 us at 64 lanes (two small prep launches 4 us + 18 of stream and products), 20 at 16 (form 0's 16).
// `dl_finalize_kernel` (a programmatic dependent launch, a thread per output element) merges the ranges' partials
// (L, M) -- form 1 writes one maximum to every range's slot: the same merge -- and writes every column of the output.
// (A merge by the last range's own block, in place of the finalize, MEASURED 14 us of serial tail at 16 lanes: 2048
// cells by 128 threads, four dependent L2 loads each; the finalize's grid does it in 2.4.)
//
// Numerically: bf16 operands, fp32 accumulation -- the chain's class (its exponentials and weights are bf16 too, its
// maximum one over K as form 1's; form 0's per range is at least as tight). Not the chain's bits: the summation order
// differs (no atomics on values either way: equal inputs give equal outputs).
//
// Configs (`DL_CONFIGS`): the form, row tiles of 64 / 128, column tiles of 16 / 32 / 64, warp layouts and stage counts,
// within the DEVICE's shared memory (`dense_live_mma_fits`: the opt-in maximum queried once a device -- more stages
// where there is room, on a GPU with more of it); the caller measures the fitting ones at the covering column tile
// against the chain (`dense_live_arms`, `autotune.choose`), so nothing here is tuned to one GPU.
//
// Contract (the caller verifies): A's base 16-byte aligned and a_ld a multiple of 8 (its 16-byte copies). Any sizes
// otherwise (M, K, L >= 1; L in column tiles, M and K tails masked); the caller takes this path for narrow steps.

#include <torch/extension.h>
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
#define MAXROWS 128                                    // rows a block of the max kernel

__device__ __forceinline__ void bar_sync(int id, int count) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int count) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void cpa16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
// a 16-byte copy of which the first `bytes` come from `gmem`, the rest zero-filled (`bytes` 0: none read)
__device__ __forceinline__ void cpa16z(void* smem, const void* gmem, int bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

extern __shared__ char smem_raw[];

template <int BM, int BN, int PW, int ST> struct DlSmemG {
    static constexpr int STAGE = BM * BK * 2 + BN * BK * 2;
    static constexpr int BYTES = ST * STAGE + (BN + PW * 32) * 4 + BN * 4;     // sM, the producers' partials, sP
};

template <int BM, int BN, int WM, int WN, int PW, int ST, bool AK>
__global__ void __launch_bounds__((WM * WN + PW) * 32) dl_gather_kernel(
        const bfloat16_t* __restrict__ A, long a_ld, const float* __restrict__ x, long x_ld, const int* __restrict__ perm,
        float* __restrict__ partL, float* __restrict__ partM, int Mrows, int K, int L, int Lpad, int knt_all, int split) {
    constexpr int NC = WM * WN * 32, NP = PW * 32, NTH = NC + NP;
    static_assert(BN % 8 == 0 && BN <= 64 && BM % 64 == 0 && 2 * ST + 1 <= 15, "columns; rows; named barriers (15: the producers')");
    const int pid_n = blockIdx.x, pid_m = blockIdx.y, sp = blockIdx.z;
    const int n0 = pid_n * BN, m0 = pid_m * BM;
    const int k0 = (int)((long)sp * knt_all / split), k1 = (int)((long)(sp + 1) * knt_all / split);
    const int knt = k1 - k0;
    const int e_lo = k0 * BK, e_hi = min(k1 * BK, K);                 // the range's rows of x
    const int ncl = min(BN, L - n0);                                     // the tile's live lanes
    const int tid = threadIdx.x;

    // smem: A (BM, BK) swizzled, K-major (8 consecutive k of a row per 16 bytes) or MN-major (8 consecutive m of a k);
    // B (BN, BK) K-major; then the range maxima per column, the producer warps' partial maxima and the tile's lanes'
    // columns of x. No static shared memory: it would shift this buffer off its alignment.
    auto swzK = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});
    auto swzM = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{});
    auto sAl = tile_to_shape(std::conditional_t<AK, decltype(swzK), decltype(swzM)>{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swzK, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int STAGE = DlSmemG<BM, BN, PW, ST>::STAGE;
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    float* sM = (float*)(smem_raw + ST * STAGE);                        // [BN] the range maxima
    float* sMp = sM + BN;                                                // [NP] the producer threads' partial maxima
    int* sP = (int*)(sMp + NP);                                          // [BN] the lanes' columns of x

    if (tid >= NC) {
        // ---------------- producers ----------------
        const int ptid = tid - NC;
        if (ptid < BN) sP[ptid] = ptid < ncl ? perm[n0 + ptid] : 0;
        bar_sync(15, NP);
        // The range maximum per column: a thread takes one lane's column (lane j = ptid % BN) at rows ptid / BN,
        // + NP / BN, ...: a warp's lanes read adjacent lanes of one row -- contiguous where the live lanes are a run
        // (a decode's: a compact row's lanes in order, liveness' runs), one sector per 8 of them; a thread a whole row
        // instead put 32 rows in one instruction, 32 sectors for 32 values (MEASURED 63 us a step at 64 lanes, 4x the
        // chain). The NP / BN partials of a lane are reduced in smem.
        static_assert(NP % BN == 0, "a producer thread one lane");
        {
            const int j = ptid % BN;
            const int col = sP[j];
            float mx = -INFINITY;
            if (j < ncl)
                for (int e = e_lo + ptid / BN; e < e_hi; e += NP / BN) mx = fmaxf(mx, x[(long)e * x_ld + col]);
            sMp[ptid] = mx;
            bar_sync(15, NP);
            if (ptid < BN) {
                float m = -INFINITY;
                CUTE_UNROLL
                for (int w = 0; w < NP / BN; w++) m = fmaxf(m, sMp[w * BN + ptid]);
                sM[ptid] = m;
                if (pid_m == 0 && ptid < ncl) partM[(long)sp * L + n0 + ptid] = m;   // (every row tile's are the same)
            }
            bar_sync(15, NP);
        }
        // exponential chunks (8 rows of a lane's column) a tile: chunk i = (rows (i / BN) * 8 .., lane i % BN), so a
        // warp's loads of one row are adjacent lanes (as the maxima's)
        constexpr int CH = (BN * BK) / 8;
        constexpr int PBC = (CH + NP - 1) / NP;                          // ... per producer thread (some idle when CH < NP)
        static_assert(CH % NP == 0 || NP % CH == 0, "whole chunk rows a pass");
        float Mc[PBC];                                                   // this thread's columns' maxima
        int pcol[PBC];                                                   // ... and columns of x
        CUTE_UNROLL
        for (int q = 0; q < PBC; q++) {
            const int i = ptid + q * NP;
            Mc[q] = (i < CH) ? sM[i % BN] : 0.f;
            pcol[q] = (i < CH) ? sP[i % BN] : 0;
        }
        constexpr int PA = (BM * BK) / 8 / NP;
        static_assert((BM * BK) / 8 % NP == 0, "weight chunks per producer thread");
        // the values of x of a tile's chunks, loaded a tile ahead
        float xv[PBC][8];
        auto load_x = [&](int t) {
            const int eb = (k0 + t) * BK;
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i % BN, e8 = (i / BN) * 8;
                if (i >= CH) continue;
                const bool live = c < ncl;
                const float* xp = x + (long)(eb + e8) * x_ld + pcol[q];
                CUTE_UNROLL
                for (int j = 0; j < 8; j++) xv[q][j] = (live && eb + e8 + j < K) ? xp[(long)j * x_ld] : -INFINITY;
            }
        };
        load_x(0);
        auto issue = [&](int t) {
            const int s = t % ST, eb = (k0 + t) * BK;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
            // the weights: 16-byte chunks of 8 consecutive k (K-major) or m (MN-major), zero-filled past K and M
            CUTE_UNROLL
            for (int q = 0; q < PA; q++) {
                const int i = ptid + q * NP;
                if constexpr (AK) {
                    const int m = i / (BK / 8), e8 = (i % (BK / 8)) * 8, row = m0 + m, col = eb + e8;
                    const int n = (row < Mrows) ? max(0, min(8, K - col)) : 0;
                    cpa16z(&sAt(m, e8), n > 0 ? A + (long)row * a_ld + col : A, n * 2);
                } else {
                    const int e = i / (BM / 8), mm = (i % (BM / 8)) * 8, row = m0 + mm, col = eb + e;
                    const int n = (col < K) ? max(0, min(8, Mrows - row)) : 0;
                    cpa16z(&sAt(mm, e), n > 0 ? A + (long)col * a_ld + row : A, n * 2);
                }
            }
            // B[c, e] = bf16 exp(x - M[c]): 8 consecutive rows of one column, one 16-byte store; past K, past the live
            // lanes and under a column of -inf, exactly 0
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i % BN, e8 = (i / BN) * 8;
                if (i >= CH) continue;
                const float M = Mc[q];
                __align__(16) bfloat16_t u[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; j++)
                    u[j] = static_cast<bfloat16_t>((M == -INFINITY || xv[q][j] == -INFINITY) ? 0.f : __expf(xv[q][j] - M));
                *(uint4*)&sBt(c, e8) = *(const uint4*)u;
            }
            if (t + 1 < knt) load_x(t + 1);
        };
        // prologue: tiles 0 .. ST - 1, a commit group each (empty past the end), so group g is tile g throughout
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();                                          // groups <= j landed
            __threadfence_block();                                       // the B stage's plain stores, too
            bar_arrive(1 + j % ST, NTH);                                 // FULL: tile j staged
            if (j >= 1) {
                const int t = j - 1 + ST;                                // refill the stage tile j - 1 left
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
    using LdA = std::conditional_t<AK, SM75_U32x4_LDSM_N, SM75_U16x8_LDSM_T>;
    auto s2rA = make_tiled_copy_A(Copy_Atom<LdA, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, mma);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);
    for (int kt = 0; kt < knt; kt++) {
        const int s = kt % ST;
        bar_sync(1 + s, NTH);                                            // FULL: tile kt staged
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt), tXrB);
        cute::gemm(mma, tCrA, tCrB, tCrL);                               // L[m, k] += sum_e A[m, e] * exp(x[e, perm k] - M_k)
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);                 // EMPTY: the stage is consumed (where refilled)
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        const int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
        if (m0 + m < Mrows && n < ncl) partL[((long)sp * Mrows + m0 + m) * Lpad + n0 + n] = tCrL(i);
    }
}

// ---------------- the maxima: a block = (32 lanes, a chunk of MAXROWS rows); 8 row groups ----------------
// pmax [chunks, Lpad]: the lanes' maxima over the chunk (-inf past L)
__global__ void __launch_bounds__(256) dl_max_kernel(const float* __restrict__ x, long x_ld, const int* __restrict__ perm,
                                                     float* __restrict__ pmax, int K, int L, int Lpad) {
    __shared__ float red[8][32];
    const int lane = threadIdx.x % 32, g = threadIdx.x / 32, n = blockIdx.x * 32 + lane;
    const int e0 = blockIdx.y * MAXROWS, e1 = min(e0 + MAXROWS, K);
    float mx = -INFINITY;
    if (n < L) {
        const int col = perm[n];
        for (int e = e0 + g; e < e1; e += 8) mx = fmaxf(mx, x[(long)e * x_ld + col]);
    }
    red[g][lane] = mx;
    __syncthreads();
    if (g == 0 && n < Lpad) {
        CUTE_UNROLL
        for (int w = 1; w < 8; w++) mx = fmaxf(mx, red[w][lane]);
        pmax[(long)blockIdx.y * Lpad + n] = mx;
    }
}

// ---------------- the exponentials: a block = (32 lanes, a tile of 64 rows) ----------------
// ex [Lpad, Kpad] bf16; the lanes' maxima into partM [split, L] (the same for every range).
__global__ void __launch_bounds__(256) dl_exp_kernel(
        const float* __restrict__ x, long x_ld, const int* __restrict__ perm, const float* __restrict__ pmax, int nchunks,
        bfloat16_t* __restrict__ ex, float* __restrict__ partM, int split, int K, int L, int Lpad, int Kpad) {
    const int lane = threadIdx.x % 32, g = threadIdx.x / 32, n = blockIdx.x * 32 + lane;
    if (n >= Lpad) return;
    float M = -INFINITY;
    for (int c = 0; c < nchunks; c++) M = fmaxf(M, pmax[(long)c * Lpad + n]);
    if (blockIdx.y == 0 && g == 0 && n < L)
        for (int z = 0; z < split; z++) partM[(long)z * L + n] = M;       // (every range's: the finalize's merge is general)
    const int e8 = blockIdx.y * BK + g * 8;                              // this thread's 8 rows of the lane's column
    const int col = n < L ? perm[n] : 0;
    __align__(16) bfloat16_t u[8];
    CUTE_UNROLL
    for (int j = 0; j < 8; j++) {
        const float v = (n < L && e8 + j < K) ? x[(long)(e8 + j) * x_ld + col] : -INFINITY;
        u[j] = static_cast<bfloat16_t>((M == -INFINITY || v == -INFINITY) ? 0.f : __expf(v - M));
    }
    *(uint4*)&ex[(long)n * Kpad + e8] = *(const uint4*)u;
}

template <int BM, int BN, int ST> struct DlSmemS { static constexpr int STAGE = BM * BK * 2 + BN * BK * 2, BYTES = ST * STAGE; };

template <int BM, int BN, int WM, int WN, int PW, int ST, bool AK>
__global__ void __launch_bounds__((WM * WN + PW) * 32) dl_split_kernel(
        const bfloat16_t* __restrict__ A, long a_ld, const bfloat16_t* __restrict__ ex, int Kpad,
        float* __restrict__ partL, int Mrows, int K, int L, int Lpad, int knt_all, int split) {
    constexpr int NC = WM * WN * 32, NP = PW * 32, NTH = NC + NP;
    static_assert(BN % 8 == 0 && BN <= 64 && BM % 64 == 0 && ST >= 2 && 2 * ST <= 15, "columns; rows; stages; named barriers");
    const int pid_n = blockIdx.x, pid_m = blockIdx.y, sp = blockIdx.z;
    const int n0 = pid_n * BN, m0 = pid_m * BM;
    const int k0 = (int)((long)sp * knt_all / split), k1 = (int)((long)(sp + 1) * knt_all / split);
    const int knt = k1 - k0;
    const int ncl = min(BN, L - n0);                                     // the tile's live lanes
    const int tid = threadIdx.x;

    // smem per stage: A (BM, BK) swizzled, K-major (8 consecutive k of a row per 16 bytes) or MN-major (8 consecutive m
    // of a k); B (BN, BK) K-major. No static shared memory before it: that would shift this buffer off its alignment.
    auto swzK = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});
    auto swzM = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{});
    auto sAl = tile_to_shape(std::conditional_t<AK, decltype(swzK), decltype(swzM)>{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swzK, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int STAGE = DlSmemS<BM, BN, ST>::STAGE;
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };

    if (tid >= NC) {
        // ---------------- producers: copies only ----------------
        const int ptid = tid - NC;
        constexpr int PA = (BM * BK) / 8 / NP, PB = (BN * BK) / 8 / NP;
        static_assert((BM * BK) / 8 % NP == 0 && (BN * BK) / 8 % NP == 0, "16-byte chunks per producer thread");
        const bfloat16_t* exr = ex + (long)n0 * Kpad + (long)k0 * BK;    // the tile's lanes' rows, at the range
        // the weights: 16-byte chunks of 8 consecutive k (K-major) or m (MN-major), zero-filled past K and M
        auto issue_A = [&](int t) {
            Tensor sAt = make_tensor(make_smem_ptr(stA(t % ST)), sAl);
            const int eb = (k0 + t) * BK;
            CUTE_UNROLL
            for (int q = 0; q < PA; q++) {
                const int i = ptid + q * NP;
                if constexpr (AK) {
                    const int m = i / (BK / 8), e8 = (i % (BK / 8)) * 8, row = m0 + m, col = eb + e8;
                    const int n = (row < Mrows) ? max(0, min(8, K - col)) : 0;
                    cpa16z(&sAt(m, e8), n > 0 ? A + (long)row * a_ld + col : A, n * 2);
                } else {
                    const int e = i / (BM / 8), mm = (i % (BM / 8)) * 8, row = m0 + mm, col = eb + e;
                    const int n = (col < K) ? max(0, min(8, Mrows - row)) : 0;
                    cpa16z(&sAt(mm, e), n > 0 ? A + (long)col * a_ld + row : A, n * 2);
                }
            }
        };
        // the exponentials: 8 consecutive rows of a lane, from `ex` (0 past the live lanes and K already)
        auto issue_B = [&](int t) {
            Tensor sBt = make_tensor(make_smem_ptr(stB(t % ST)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < PB; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                cpa16(&sBt(c, e8), exr + (long)c * Kpad + t * BK + e8);
            }
        };
        // prologue: the weights of tiles 0 .. ST - 1 under the prep kernels; then their exponentials, a commit group a
        // tile (group 0 holds every prologue weight tile too: it lands with tile 0 at the latest)
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) if (t < knt) issue_A(t);
        cudaGridDependencySynchronize();
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue_B(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();                                          // groups <= j landed
            bar_arrive(1 + j % ST, NTH);                                 // FULL: tile j staged
            if (j >= 1) {
                const int t = j - 1 + ST;                                // refill the stage tile j - 1 left
                if (t < knt) { bar_sync(1 + ST + (j - 1) % ST, NTH); issue_A(t); issue_B(t); }
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
    // two sets of fragments: the next tile's load under this tile's products
    Tensor tCrA0 = thr.partition_fragment_A(sAt0), tCrA1 = thr.partition_fragment_A(sAt0);
    Tensor tCrB0 = thr.partition_fragment_B(sBt0), tCrB1 = thr.partition_fragment_B(sBt0);
    using LdA = std::conditional_t<AK, SM75_U32x4_LDSM_N, SM75_U16x8_LDSM_T>;
    auto s2rA = make_tiled_copy_A(Copy_Atom<LdA, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, mma);
    auto thrA = s2rA.get_thread_slice(tid);
    auto thrB = s2rB.get_thread_slice(tid);
    Tensor tXrA0 = thrA.retile_D(tCrA0), tXrA1 = thrA.retile_D(tCrA1);
    Tensor tXrB0 = thrB.retile_D(tCrB0), tXrB1 = thrB.retile_D(tCrB1);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);
    // the stage's fragments into registers, the stage given back at once (EMPTY), where it is refilled
    auto load = [&](int kt, auto& xa, auto& xb) {
        const int s = kt % ST;
        bar_sync(1 + s, NTH);                                            // FULL: tile kt staged
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, thrA.partition_S(sAt), xa);
        copy(s2rB, thrB.partition_S(sBt), xb);
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);
    };
    if (knt > 0) load(0, tXrA0, tXrB0);
    for (int kt = 0; kt < knt; kt += 2) {
        if (kt + 1 < knt) load(kt + 1, tXrA1, tXrB1);
        cute::gemm(mma, tCrA0, tCrB0, tCrL);                             // L[m, k] += sum_e A[m, e] * exp(x[e, perm k] - M_k)
        if (kt + 1 < knt) {
            if (kt + 2 < knt) load(kt + 2, tXrA0, tXrB0);
            cute::gemm(mma, tCrA1, tCrB1, tCrL);
        }
    }
    // the range's partial (the finalize merges the ranges)
    float* myL = partL + ((long)sp * Mrows + m0) * Lpad + n0;
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        const int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
        if (m0 + m < Mrows && n < ncl) myL[(long)m * Lpad + n] = tCrL(i);
    }
}

// ---------------- the finalize: a thread per output element (row, column), columns fastest ----------------
// A programmatic dependent launch: it waits here for the product kernel's writes.
template <int MODE>
__global__ void dl_finalize_kernel(float* __restrict__ out, long out_ld, const int* __restrict__ rank,
                                   const float* __restrict__ partL, const float* __restrict__ partM, int Mrows, int ncols,
                                   int L, int Lpad, int split) {
    cudaGridDependencySynchronize();
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)Mrows * ncols) return;
    const int m = (int)(i / ncols), c = (int)(i - (long)m * ncols);
    const int k = rank[c];
    float* o = out + (long)m * out_ld + c;
    if (k < 0) {
        if (MODE == 0) *o = -INFINITY;
        return;
    }
    float mx = -INFINITY;
    for (int z = 0; z < split; z++) mx = fmaxf(mx, partM[(long)z * L + k]);
    float v = -INFINITY;
    if (mx != -INFINITY) {
        float s = 0.f;
        for (int z = 0; z < split; z++) {
            const float mz = partM[(long)z * L + k];
            if (mz != -INFINITY) s += partL[((long)z * Mrows + m) * Lpad + k] * __expf(mz - mx);
        }
        v = (s <= 0.f) ? -INFINITY : logf(s) + mx;
    }
    if (MODE == 2) {                                                     // logaddexp(out, v), -inf + -inf = -inf
        const float a = *o, hi = fmaxf(a, v);
        v = (hi == -INFINITY) ? -INFINITY : logf(__expf(a - hi) + __expf(v - hi)) + hi;
    }
    *o = v;
}

// the device's dynamic shared memory a block may opt into (queried once a device)
static int smem_cap() {
    static int caps[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64) { int v = 0; C10_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)); return v; }
    if (caps[dev] == 0) C10_CUDA_CHECK(cudaDeviceGetAttribute(&caps[dev], cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    return caps[dev];
}

// FORM 0: the gather kernel and the finalize; FORM 1: the maxima, the exponentials, the split kernel (a programmatic
// dependent launch) and the finalize (one too). `partL` [split, Mrows, Lpad], `partM` [split, L].
template <int FORM, int BM, int BN, int WM, int WN, int PW, int ST, bool AK>
static void launch_cfg(torch::Tensor out, long out_ld, torch::Tensor A, long a_ld, torch::Tensor x, long x_ld,
                       torch::Tensor perm, torch::Tensor rank, int Mrows, int K, int L, int ncols, int split, int mode) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr int smem = FORM == 0 ? DlSmemG<BM, BN, PW, ST>::BYTES : DlSmemS<BM, BN, ST>::BYTES;
    TORCH_CHECK(smem <= smem_cap(), "dense live mma: the config's ", smem, " bytes of shared memory are past this device's ",
                smem_cap(), " (`dense_live_mma_fits`)");
    auto kernel = FORM == 0 ? (void*)dl_gather_kernel<BM, BN, WM, WN, PW, ST, AK> : (void*)dl_split_kernel<BM, BN, WM, WN, PW, ST, AK>;
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_set[dev] = true;
    }
    const int knt = (K + BK - 1) / BK, Kpad = knt * BK, nchunks = (K + MAXROWS - 1) / MAXROWS;
    const int ntiles = (L + BN - 1) / BN, mtiles = (Mrows + BM - 1) / BM, Lpad = ntiles * BN, lblocks = (Lpad + 31) / 32;
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    // scratch: partL fp32 [split, Mrows, Lpad]; partM [split, L]; FORM 1: ex bf16 [Lpad, Kpad], pmax [nchunks, Lpad]
    // (each piece's offset a multiple of 4 floats: 16-byte copies and stores from `ex`)
    const long n_pl = (long)split * Mrows * Lpad, n_pm = ((long)split * L + 3) / 4 * 4;
    const long n_ex = FORM == 1 ? (long)Lpad * Kpad : 0, n_px = FORM == 1 ? (long)nchunks * Lpad : 0;
    torch::Tensor scratch = torch::empty({n_pl + n_pm + n_ex / 2 + n_px}, x.options());
    float* partL = scratch.data_ptr<float>();
    float* partM = partL + n_pl;
    bfloat16_t* ex = reinterpret_cast<bfloat16_t*>(partM + n_pm);
    float* pmax = partM + n_pm + n_ex / 2;
    const int* pm = perm.data_ptr<int>();
    const bfloat16_t* Ap = reinterpret_cast<const bfloat16_t*>(A.data_ptr<at::BFloat16>());
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfgl = {};
    cfgl.gridDim = dim3(ntiles, mtiles, split); cfgl.blockDim = dim3(NTH); cfgl.dynamicSmemBytes = smem; cfgl.stream = st;
    cfgl.attrs = attr; cfgl.numAttrs = 1;
    if constexpr (FORM == 0) {
        dl_gather_kernel<BM, BN, WM, WN, PW, ST, AK><<<cfgl.gridDim, NTH, smem, st>>>(
            Ap, a_ld, x.data_ptr<float>(), x_ld, pm, partL, partM, Mrows, K, L, Lpad, knt, split);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        dl_max_kernel<<<dim3(lblocks, nchunks), 256, 0, st>>>(x.data_ptr<float>(), x_ld, pm, pmax, K, L, Lpad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        dl_exp_kernel<<<dim3(lblocks, knt), 256, 0, st>>>(x.data_ptr<float>(), x_ld, pm, pmax, nchunks, ex, partM, split, K, L,
                                                        Lpad, Kpad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, dl_split_kernel<BM, BN, WM, WN, PW, ST, AK>, Ap, a_ld, (const bfloat16_t*)ex,
                                          Kpad, partL, Mrows, K, L, Lpad, knt, split));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const long n = (long)Mrows * ncols;
    cfgl.gridDim = dim3((unsigned)((n + 255) / 256)); cfgl.blockDim = dim3(256); cfgl.dynamicSmemBytes = 0;
    float* o = out.data_ptr<float>();
    const int* rk = rank.data_ptr<int>();
    if (mode == 0)
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, dl_finalize_kernel<0>, o, out_ld, rk, (const float*)partL, (const float*)partM, Mrows, ncols, L, Lpad, split));
    else if (mode == 1)
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, dl_finalize_kernel<1>, o, out_ld, rk, (const float*)partL, (const float*)partM, Mrows, ncols, L, Lpad, split));
    else
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, dl_finalize_kernel<2>, o, out_ld, rk, (const float*)partL, (const float*)partM, Mrows, ncols, L, Lpad, split));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// cfg -> (FORM, BM, BN, WM, WN, PW, ST): the form, row tiles, column tiles, the MMA warps' layout, producer warps, stages
#define DL_CONFIGS(X) \
    X(0, 0, 128, 16, 4, 1, 4, 3) X(1, 0, 64, 16, 2, 1, 4, 4) X(2, 0, 128, 32, 4, 1, 4, 3) X(3, 0, 64, 32, 2, 1, 4, 4) \
    X(4, 0, 128, 64, 4, 2, 4, 2) X(5, 0, 64, 64, 2, 2, 4, 3) \
    X(6, 1, 128, 16, 4, 1, 4, 3) X(7, 1, 128, 16, 4, 1, 4, 5) X(8, 1, 64, 16, 2, 1, 4, 2) X(9, 1, 64, 16, 2, 1, 4, 4) \
    X(10, 1, 128, 32, 4, 1, 4, 3) X(11, 1, 128, 32, 4, 1, 4, 4) X(12, 1, 64, 32, 2, 1, 4, 2) X(13, 1, 64, 32, 2, 1, 4, 4) \
    X(14, 1, 128, 64, 4, 2, 4, 2) X(15, 1, 128, 64, 4, 2, 4, 4) X(16, 1, 64, 64, 2, 2, 4, 2) X(17, 1, 64, 64, 2, 2, 4, 3) \
    X(18, 1, 128, 64, 4, 1, 4, 3)

// `out` the output's first row (fp32, rows of `out_ld`), `A` the bf16 weights' first element ([M, K]: K-major,
// A[m * a_ld + e], or `k_major` false, A[e * a_ld + m]), `x` the children's first row (fp32 [K, *], rows of `x_ld`),
// `perm` / `rank` int32 (the live lanes' columns of x; each output column's lane, -1 dead), `mode` 0 overwrite, 1 keep
// the dead columns, 2 log-add; `cfg` = config id | split << 8.
void dense_live_mma(torch::Tensor out, int64_t out_ld, torch::Tensor A, int64_t a_ld, bool k_major, torch::Tensor x,
                    int64_t x_ld, torch::Tensor perm, torch::Tensor rank, int64_t M, int64_t K, int64_t L, int64_t ncols,
                    int64_t mode, int64_t cfg) {
    const int split = cfg >> 8 ? (int)(cfg >> 8) : 1;
    cfg &= 255;
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32
                && perm.scalar_type() == torch::kInt32 && rank.scalar_type() == torch::kInt32,
                "dense live mma: bf16 weights, fp32 values, int32 lanes");
    TORCH_CHECK(M >= 1 && K >= 1 && L >= 1 && ncols >= 1 && mode >= 0 && mode <= 2, "dense live mma: sizes / mode");
    TORCH_CHECK((((size_t)A.data_ptr()) & 15) == 0 && a_ld % 8 == 0, "dense live mma: the weights' 16-byte copies");
    const int knt = (int)((K + BK - 1) / BK);
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= knt, "dense live mma: split 1 .. ", MAX_SPLIT, ", at most the K tiles");
    switch (cfg) {
#define DL_CASE(ID, F_, BM_, BN_, WM_, WN_, PW_, ST_) case ID: \
        if (k_major) launch_cfg<F_, BM_, BN_, WM_, WN_, PW_, ST_, true>(out, out_ld, A, a_ld, x, x_ld, perm, rank, (int)M, (int)K, \
                                                                        (int)L, (int)ncols, split, (int)mode); \
        else launch_cfg<F_, BM_, BN_, WM_, WN_, PW_, ST_, false>(out, out_ld, A, a_ld, x, x_ld, perm, rank, (int)M, (int)K, \
                                                                 (int)L, (int)ncols, split, (int)mode); \
        break;
        DL_CONFIGS(DL_CASE)
#undef DL_CASE
        default: TORCH_CHECK(false, "dense live mma: invalid cfg ", cfg);
    }
}

// (BM, BN, ST, shared memory bytes, FORM) per config id
std::vector<std::vector<int64_t>> dense_live_mma_configs() {
    std::vector<std::vector<int64_t>> out;
#define DL_LIST(ID, F_, BM_, BN_, WM_, WN_, PW_, ST_) \
    out.push_back({BM_, BN_, ST_, F_ == 0 ? (int64_t)DlSmemG<BM_, BN_, PW_, ST_>::BYTES : (int64_t)DlSmemS<BM_, BN_, ST_>::BYTES, F_});
    DL_CONFIGS(DL_LIST)
#undef DL_LIST
    return out;
}

// whether config `cfg`'s shared memory fits the current device
bool dense_live_mma_fits(int64_t cfg) {
    const auto cs = dense_live_mma_configs();
    return cfg >= 0 && cfg < (int64_t)cs.size() && cs[cfg][3] <= smem_cap();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_live_mma", &dense_live_mma,
          "a dense run's live-column product on tensor cores (the gather fused, or the exponentials once; the log and scatter fused)");
    m.def("dense_live_mma_configs", &dense_live_mma_configs, "(BM, BN, stages, shared memory bytes, form) per config id");
    m.def("dense_live_mma_fits", &dense_live_mma_fits, "whether a config's shared memory fits this device");
}
