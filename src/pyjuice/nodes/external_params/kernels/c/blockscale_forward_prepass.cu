// The `BlockScaleSumParams` forward in three launches -- a pre-pass, the merge factors and a streaming kernel --
// computing what blockscale_forward.cu's `blockscale_tlmm_kernel` computes (its SINGLE_RUN case), to the bit.
//
// The fork walks a row's edge tiles in one thread block per (node tile, batch tile), serially: per tile it folds the
// log-gates into the element tile, takes its max and exponentials, builds the normalizer's operand, converts the
// parameters and runs two MMAs; then it merges the tile into the online log-sum-exp L and the normalizer Z:
//     L = L * sLS + S * sPS          S = A(params) . B(exp(em + log phi - mx))
//     Z = Z * sScale + S2            S2 = A(params) . B2(exp(log phi - mz))
// Everything but the two MMAs depends on the edge tile and the batch column only, and the running maxima the factors
// come from are PREFIX maxima over the tiles (max is exact and order-free). So:
//   pre-pass   per (node block x edge tile, 64 columns): s = em + log phi (the fork's gate_pass), the tile max mx, the
//              bf16 exponentials, the tile's log-gates (one row per gate) and their max gm;
//   factors    per (node block, 32 columns, 8 tiles): the prefix maxima (the fork's chain over the earlier tiles' maxima),
//              then every tile's sLS, sPS, sScale, M, mz and B2 values (bf16 exp(log phi - mz), one row per gate);
//   streaming  producers cp.async each tile's parameters, exponentials, B2 rows and factors into mbarrier'd stages
//              (fp32 parameters: loaded a tile ahead into registers, converted to bf16 as the fork does); G consumer
//              groups take the tiles kt = g (mod G), each tile's two MMAs into a fresh S / S2 (from zero, k-blocks in
//              order: the fork's per-element sums), its merges -- the fork's expressions, rounded as the fork's
//              compiled (`fma(L, sLS, S * sPS)`, `fma(Z, sScale, S2)`) -- in tile order: one group keeps L / Z in
//              registers; several keep them in shared memory and pass a token from the group that merged tile kt - 1.
//              The B2 operand is read straight from its rows per group of 4 children (stride 0 within a group).
// The output, log N - log Z and log Z, is the fork's epilogue on the same values.
//
// MEASURED (RTX PRO 6000, a gated HMM layer, gate width 64, against the fork's best config): HMM-1024 B 64 / 128 bf16
// 51 / 50 -> 10.8 / 11.0 us, fp32 53 / 55 -> 12.6 / 13.5; B 512 52 -> 19.0; HMM-4096 (4 node blocks of 1024, 64 edge
// tiles) B 64 194 -> 33, B 128 199 -> 50, B 512 338 -> 144 (fp32 365 -> 192). The small grids are bound by the L2's
// rate per SM (~31 B / clock at most); the large ones by the bytes a stage ring keeps in flight.
//
// Scope (the Python dispatcher checks it): one gate per node block (n_node_gates 1), node_cbs >= 64 (an edge tile is
// one run of children), a gate width >= 4 (clamped to 64), batch % 64 == 0, block_size % BM == 0. Scratch: one
// allocation per call (inside a CUDA graph capture, from its pool).

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"   // `ParamLoad<TP>`, `param_ptr<TP>`: the stored dtype
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64           // edge tile
#define PB 64           // the pre-pass's batch columns per thread block
#define QG 16           // groups of 4 children per edge tile
#define NF 5            // factors per (tile, column): sLS, sPS, sScale, M, mz

__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void cpa16(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa16_ca(uint32_t smem, const void* gmem) {          // (rows read more than once)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem) : "memory");
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

// ---------------- pre-pass: a thread block per (node block x edge tile, PB batch columns) ----------------
// bx[(tile * BK + e) * batch + b] = bf16 exp(s - mx), s = element + log phi; mx[tile * batch + b] = max_e s;
// lp[(tile * QG + g) * batch + b] = log phi of the tile's gate g (rows 0 .. 64 / gc - 1); gm[tile * batch + b] = their
// max. The gates are staged as the fork stages them (sPh), the add, max, exponential and -inf guard are its own.
template <int NT>
__global__ void __launch_bounds__(NT) bs_prepass_kernel(
        const float* __restrict__ element_mars, const long* __restrict__ cids, const float* __restrict__ ext,
        const long* __restrict__ gate, __nv_bfloat16* __restrict__ bx, float* __restrict__ mx, float* __restrict__ gm,
        float* __restrict__ lp, int batch, int knt, int gate_stride, int node_cbs, int gc, int node_sh, int gate_sh,
        long ext_base) {
    constexpr int PARTS = NT / PB, EP = BK / PARTS, QP = EP / 4;
    static_assert(NT % PB == 0 && BK % PARTS == 0 && EP % 4 == 0, "threads over the batch columns");
    __shared__ float smx[PARTS][PB];
    long tile = blockIdx.x;
    int nblock = (int)(tile / knt), kt = (int)(tile % knt);
    int b0 = blockIdx.y * PB;
    int b = threadIdx.x % PB, part = threadIdx.x / PB;
    long e0 = cids[tile];
    const long* gt = gate + (long)nblock * gate_stride;
    float lq[QP];                                         // the gates of this thread's groups of 4 children
    CUTE_UNROLL
    for (int jj = 0; jj < QP; jj++) {
        int q = part * QP + jj;
        int ge0 = kt * BK + ((q * 4) / gc) * gc;           // the fork's kt * BK + gi * gcbs_eff
        int j = ge0 >> node_sh;
        int d = (ge0 & (node_cbs - 1)) >> gate_sh;
        long gb = (j < gate_stride) ? gt[j] : -1;
        lq[jj] = (gb >= 0) ? ext[(gb + ext_base + d) * (long)batch + b0 + b] : -INFINITY;
    }
    float v[EP];
    CUTE_UNROLL
    for (int j = 0; j < EP; j++) {
        v[j] = element_mars[(e0 + part * EP + j) * (long)batch + b0 + b];
        v[j] += lq[j / 4];
    }
    float m = -INFINITY;
    CUTE_UNROLL
    for (int j = 0; j < EP; j++) m = fmaxf(m, v[j]);
    smx[part][b] = m;
    __syncthreads();
    m = smx[0][b];
    CUTE_UNROLL
    for (int q = 1; q < PARTS; q++) m = fmaxf(m, smx[q][b]);
    CUTE_UNROLL
    for (int j = 0; j < EP; j++) {
        bfloat16_t r = static_cast<bfloat16_t>((m == -INFINITY) ? 0.f : __expf(v[j] - m));
        bx[(tile * BK + part * EP + j) * (long)batch + b0 + b] = *(const __nv_bfloat16*)&r;
    }
    if (part == 0) mx[tile * (long)batch + b0 + b] = m;
    float gq = -INFINITY;                                 // the tile's gate max (over its groups' gates)
    CUTE_UNROLL
    for (int jj = 0; jj < QP; jj++) gq = fmaxf(gq, lq[jj]);
    __syncthreads();
    smx[part][b] = gq;
    __syncthreads();
    if (part == 0) {
        CUTE_UNROLL
        for (int q = 1; q < PARTS; q++) gq = fmaxf(gq, smx[q][b]);
        gm[tile * (long)batch + b0 + b] = gq;
    }
    CUTE_UNROLL
    for (int jj = 0; jj < QP; jj++) {                     // one row per gate: its first group's
        int q = part * QP + jj;
        if ((q * 4) % gc == 0) lp[(tile * QG + (q * 4) / gc) * (long)batch + b0 + b] = lq[jj];
    }
}

// ---------------- factors: a thread block per (node block, 32 columns, CH edge tiles) ----------------
// f[(tile * NF + i) * batch + b] = the fork's sLS, sPS, sScale, M and mz at tile `tile`; zg[(tile * QG + g) * batch + b]
// = bf16 exp(log phi - mz) of gate g. The maxima before the block's tiles are over every earlier tile's (read in
// parallel), its tiles' running maxima the fork's chain itself (warp 0, in shared memory), then the exponentials.
template <int NT, int CH>
__global__ void __launch_bounds__(NT) bs_factors_kernel(const float* __restrict__ mx, const float* __restrict__ gm,
                                                        const float* __restrict__ lp, float* __restrict__ f,
                                                        __nv_bfloat16* __restrict__ zg, int batch, int knt, int ng) {
    constexpr int W = 32, PARTS = NT / W;
    __shared__ float sM[CH + 1][W], sZ[CH + 1][W], sX[CH][W], sC[2][PARTS][W];
    int nblock = blockIdx.x, b0 = blockIdx.y * W, c0 = blockIdx.z * CH;
    int b = threadIdx.x % W, part = threadIdx.x / W;
    const long tile0 = (long)nblock * knt;
    const int cn = min(CH, knt - c0);
    float pm = -INFINITY, pz = -INFINITY;                 // the maxima over the tiles < c0
    for (int t0 = 0; t0 < c0; t0 += 8 * PARTS) {
        float vm[8], vz[8];
        CUTE_UNROLL
        for (int k = 0; k < 8; k++) {
            int t = t0 + part + k * PARTS;
            vm[k] = t < c0 ? mx[(tile0 + t) * batch + b0 + b] : -INFINITY;
            vz[k] = t < c0 ? gm[(tile0 + t) * batch + b0 + b] : -INFINITY;
        }
        CUTE_UNROLL
        for (int k = 0; k < 8; k++) { pm = fmaxf(pm, vm[k]); pz = fmaxf(pz, vz[k]); }
    }
    sC[0][part][b] = pm; sC[1][part][b] = pz;
    for (int t = part; t < cn; t += PARTS) {
        long tile = tile0 + c0 + t;
        sM[t + 1][b] = sX[t][b] = mx[tile * batch + b0 + b];
        sZ[t + 1][b] = gm[tile * batch + b0 + b];
    }
    __syncthreads();
    if (part == 0) {
        float M = sC[0][0][b], mz = sC[1][0][b];
        CUTE_UNROLL
        for (int q = 1; q < PARTS; q++) { M = fmaxf(M, sC[0][q][b]); mz = fmaxf(mz, sC[1][q][b]); }
        sM[0][b] = M; sZ[0][b] = mz;
        for (int t = 0; t < cn; t++) {
            M = fmaxf(M, sM[t + 1][b]); mz = fmaxf(mz, sZ[t + 1][b]);
            sM[t + 1][b] = M; sZ[t + 1][b] = mz;
        }
    }
    __syncthreads();
    for (int t = part; t < cn; t += PARTS) {
        long tile = tile0 + c0 + t;
        const float Mo = sM[t][b], nM = sM[t + 1][b], mzo = sZ[t][b], mzn = sZ[t + 1][b];
        const float mxk = sX[t][b];
        float* ft = f + tile * NF * (long)batch + b0 + b;
        ft[0] = (nM == -INFINITY || Mo == -INFINITY) ? 0.0f : __expf(Mo - nM);
        ft[batch] = (nM == -INFINITY || mxk == -INFINITY) ? 0.0f : __expf(mxk - nM);
        ft[2 * (long)batch] = (mzn == -INFINITY || mzo == -INFINITY) ? 0.0f : __expf(mzo - mzn);
        ft[3 * (long)batch] = nM;
        ft[4 * (long)batch] = mzn;
    }
    for (int i0 = part; i0 < cn * ng; i0 += 8 * PARTS) {    // 8 loads in flight a thread
        float v[8];
        CUTE_UNROLL
        for (int k = 0; k < 8; k++) {
            int i = i0 + k * PARTS, t = i / ng, q = i % ng;
            v[k] = i < cn * ng ? lp[((tile0 + c0 + t) * QG + q) * batch + b0 + b] : 0.f;
        }
        CUTE_UNROLL
        for (int k = 0; k < 8; k++) {
            int i = i0 + k * PARTS, t = i / ng, q = i % ng;
            if (i < cn * ng) {
                const float mzn = sZ[t + 1][b];
                bfloat16_t r = static_cast<bfloat16_t>((mzn == -INFINITY) ? 0.f : __expf(v[k] - mzn));
                zg[((tile0 + c0 + t) * QG + q) * batch + b0 + b] = *(const __nv_bfloat16*)&r;
            }
        }
    }
}

extern __shared__ char smem_raw[];

// The MN-major smem atom of an operand with W rows (m or b) a k: up to 64 rows per 128-byte line, swizzled over the
// 16-byte chunks so the transposed ldmatrix's 8 k-rows of one chunk fall in distinct banks.
template <int W> struct Atom;
template <> struct Atom<32> { using L = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{})); };
template <> struct Atom<64> { using L = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{})); };
template <int W> using atom_of = typename Atom<(W < 64 ? W : 64)>::L;

template <int BM, int BN>
constexpr int stage_bytes() { return BM * BK * 2 + BN * BK * 2 + QG * BN * 2 + 3 * BN * 4; }
template <int BN> constexpr int lrow() { return BN + 8; }   // L / Z rows, padded: the merge's 8-byte accesses conflict-free
// one consumer group keeps L / Z in registers (the fork's tCrL / tCrZ); several pass them in shared memory
template <int BM, int BN, int G, int ST>
constexpr int smem_bytes() { return ST * stage_bytes<BM, BN>() + (G > 1 ? 2 * BM * lrow<BN>() * 4 : 0) + 2 * ST * 8; }

// ---------------- streaming kernel: G groups of WM x WN MMA warps, PW producer warps, ST stages ----------------
template <int BM, int BN, int WM, int WN, int G, int PW, int ST, typename TP>
__global__ void __launch_bounds__((G * WM * WN + PW) * 32) bs_stream_kernel(
        float* __restrict__ node_mars, const TP* __restrict__ mp, const long* __restrict__ nids,
        const long* __restrict__ pids, const __nv_bfloat16* __restrict__ bx, const float* __restrict__ f,
        const __nv_bfloat16* __restrict__ zg, float* __restrict__ log_z_out, int batch, int block_size, int knt,
        int pid_m_offset, int gsh) {
    constexpr int NCG = WM * WN * 32;                // a consumer group's threads
    constexpr int NC = G * NCG;
    constexpr int NP = PW * 32;
    constexpr bool BF16 = std::is_same<TP, __nv_bfloat16>::value;
    constexpr int FULLC = BF16 ? NP : 2 * NP;        // fp32 parameters: their stores arrive too
    static_assert(G + 2 <= 16, "named barriers: a token per group, the end");
    int mtiles = block_size / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int nblock = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_nid = nids[nblock];
    const long* pb = pids + (long)nblock * knt;
    const long tile0 = (long)nblock * knt;
    int tid = threadIdx.x;

    auto sAl = tile_to_shape(atom_of<BM>{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(atom_of<BN>{}, make_shape(Int<BN>{}, Int<BK>{}));
    auto sZl = Layout<Shape<Int<BN>, Shape<_4, Int<QG>>>, Stride<_1, Stride<_0, Int<BN>>>>{};   // [b, e] = z4[e / 4, b]
    constexpr int STAGE = stage_bytes<BM, BN>();
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    auto stZ = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2 + BN * BK * 2); };
    auto stF = [&](int s) { return (float*)(smem_raw + s * STAGE + BM * BK * 2 + BN * BK * 2 + QG * BN * 2); };
    constexpr int LS = lrow<BN>();
    float* sL = (float*)(smem_raw + ST * STAGE);                 // [BM, LS], the running sums (G > 1)
    float* sZs = sL + BM * LS;                                   // the normalizer's
    uint64_t* full = (uint64_t*)(smem_raw + ST * STAGE + (G > 1 ? 2 * BM * LS * 4 : 0));
    uint64_t* empty = full + ST;

    if (tid == 0) {
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], FULLC); mbar_init(&empty[s], NCG); }
    }
    if constexpr (G > 1)
        for (int i = tid; i < BM * LS; i += NC + NP) { sL[i] = 0.f; sZs[i] = 0.f; }
    __syncthreads();

    if (tid >= NC) {
        // ---------------- producers: tile t into stage t % ST once its consumer group released it ----------------
        int ptid = tid - NC;
        // a thread's chunks are the same every tile: their shared offsets (bytes, from a stage) and global offsets
        // (elements, from the tile's bases), computed once; ~0u marks a chunk past the end
        constexpr int TA = (BM * BK) / 8, TB = (BN * BK) / 8, TZ = QG * (BN / 8), TF = 3 * (BN / 4);
        constexpr int CA = (TA + NP - 1) / NP, CB = (TB + NP - 1) / NP, CZ = (TZ + NP - 1) / NP, CF = (TF + NP - 1) / NP;
        constexpr int TA4 = (BM * BK) / 4, CA4 = BF16 ? 1 : (TA4 + NP - 1) / NP;
        uint32_t oA[CA], oB[CB], oZ[CZ], oF[CF], oA4[CA4];
        long gA[CA], gA4[CA4];
        int gB[CB], gZ[CZ], gF[CF];
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                oA[q] = i < TA ? (uint32_t)((char*)&sAt(mm, e) - smem_raw) : ~0u;
                gA[q] = (long)e * block_size + mm;
            }
            CUTE_UNROLL
            for (int q = 0; q < CA4; q++) {
                int i = ptid + q * NP, e = i / (BM / 4), mm = (i % (BM / 4)) * 4;
                oA4[q] = i < TA4 ? (uint32_t)((char*)&sAt(mm, e) - smem_raw) : ~0u;
                gA4[q] = (long)e * block_size + mm;
            }
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) {
                int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - smem_raw) : ~0u;
                gB[q] = e * batch + bb;
            }
            CUTE_UNROLL
            for (int q = 0; q < CZ; q++) {                // group r's B2 row is its gate's
                int i = ptid + q * NP, r = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oZ[q] = i < TZ ? (uint32_t)((char*)&stZ(0)[r * BN + bb] - smem_raw) : ~0u;
                gZ[q] = ((r * 4) >> gsh) * batch + bb;
            }
            CUTE_UNROLL
            for (int q = 0; q < CF; q++) {
                int i = ptid + q * NP, r = i / (BN / 4), bb = (i % (BN / 4)) * 4;
                oF[q] = i < TF ? (uint32_t)((char*)&stF(0)[r * BN + bb] - smem_raw) : ~0u;
                gF[q] = r * batch + bb;
            }
        }
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        long pcn = pb[0];
        float4 ra[CA4];                                  // fp32: the next tile's parameters, loaded a tile ahead
        if constexpr (!BF16) {
            const TP* ga = mp + pcn + (long)tile_id * BM;
            CUTE_UNROLL
            for (int q = 0; q < CA4; q++) if (oA4[q] != ~0u) ra[q] = *(const float4*)(ga + gA4[q]);
        }
        for (int t = 0; t < knt; t++) {
            int s = t % ST;
            if (t >= ST) mbar_wait(&empty[s], ((t / ST) - 1) & 1);
            const long pc = pcn + (long)tile_id * BM;
            if (t + 1 < knt) pcn = pb[t + 1];
            const uint32_t ss = sbase + s * STAGE;
            if constexpr (BF16) {
                const TP* ga = mp + pc;
                CUTE_UNROLL
                for (int q = 0; q < CA; q++) if (oA[q] != ~0u) cpa16(ss + oA[q], ga + gA[q]);
            } else {
                CUTE_UNROLL
                for (int q = 0; q < CA4; q++)
                    if (oA4[q] != ~0u) {
                        bfloat16_t r[4] = {static_cast<bfloat16_t>(ra[q].x), static_cast<bfloat16_t>(ra[q].y),
                                           static_cast<bfloat16_t>(ra[q].z), static_cast<bfloat16_t>(ra[q].w)};
                        *(uint2*)(smem_raw + s * STAGE + oA4[q]) = *(const uint2*)r;
                    }
            }
            const __nv_bfloat16* bt = bx + (tile0 + t) * BK * (long)batch + b0;
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) if (oB[q] != ~0u) cpa16(ss + oB[q], bt + gB[q]);
            const __nv_bfloat16* zt = zg + (tile0 + t) * QG * (long)batch + b0;
            CUTE_UNROLL
            for (int q = 0; q < CZ; q++) if (oZ[q] != ~0u) cpa16_ca(ss + oZ[q], zt + gZ[q]);
            const float* ft = f + (tile0 + t) * NF * (long)batch + b0;
            CUTE_UNROLL
            for (int q = 0; q < CF; q++) if (oF[q] != ~0u) cpa16(ss + oF[q], ft + gF[q]);
            if constexpr (!BF16) {
                mbar_arrive(&full[s]);                   // (the converted parameters' stores, released)
                if (t + 1 < knt) {
                    const TP* ga = mp + pcn + (long)tile_id * BM;
                    CUTE_UNROLL
                    for (int q = 0; q < CA4; q++) if (oA4[q] != ~0u) ra[q] = *(const float4*)(ga + gA4[q]);
                }
            }
            cpa_mbar_arrive(&full[s]);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    // ---------------- consumers: group g, tiles g, g + G, ... ----------------
    const int g = tid / NCG, gtid = tid % NCG;
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(gtid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    Tensor tCrB2 = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    auto tA = s2rA.get_thread_slice(gtid);
    auto tB = s2rB.get_thread_slice(gtid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor tXrB2 = tB.retile_D(tCrB2);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrS2 = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);      // (G == 1)
    Tensor tCrZ = thr.partition_fragment_C(cC);
    clear(tCrL); clear(tCrZ);
    constexpr int KB = size<2>(tCrA);
    constexpr int NE = size(tCrS);
    float al[NE], be[NE], sc[NE];

    for (int kt = g; kt < knt; kt += G) {
        int s = kt % ST;
        mbar_wait(&full[s], (kt / ST) & 1);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        Tensor sZt = make_tensor(make_smem_ptr(stZ(s)), sZl);
        auto tXsA = tA.partition_S(sAt);
        auto tXsB = tB.partition_S(sBt);
        auto tXsZ = tB.partition_S(sZt);
        clear(tCrS); clear(tCrS2);
        copy(s2rA, tXsA(_, _, 0), tXrA(_, _, 0));
        copy(s2rB, tXsB(_, _, 0), tXrB(_, _, 0));
        copy(s2rB, tXsZ(_, _, 0), tXrB2(_, _, 0));
        const float* sf = stF(s);
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {                // block kb + 1's fragments load while block kb multiplies
            if (kb + 1 < KB) {
                copy(s2rA, tXsA(_, _, kb + 1), tXrA(_, _, kb + 1));
                copy(s2rB, tXsB(_, _, kb + 1), tXrB(_, _, kb + 1));
                copy(s2rB, tXsZ(_, _, kb + 1), tXrB2(_, _, kb + 1));
            } else {
                CUTE_UNROLL
                for (int i = 0; i < NE; i++) {
                    int b = get<1>(tCcC(i));
                    al[i] = sf[b]; be[i] = sf[BN + b]; sc[i] = sf[2 * BN + b];
                }
                mbar_arrive(&empty[s]);                  // the stage is read (operands and factors in registers)
            }
            cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrS);
            cute::gemm(mma, tCrA(_, _, kb), tCrB2(_, _, kb), tCrS2);
        }
        // tile kt's merge, after tile kt - 1's: the fork's expressions, as it compiled them
        if constexpr (G == 1) {
            CUTE_UNROLL
            for (int i = 0; i < NE; i++) {
                tCrL(i) = __fmaf_rn(tCrL(i), al[i], __fmul_rn(tCrS(i), be[i]));
                tCrZ(i) = __fmaf_rn(tCrZ(i), sc[i], tCrS2(i));
            }
        } else {
            if (kt > 0) bar_sync(1 + g, 2 * NCG);        // the previous group merged tile kt - 1
            CUTE_UNROLL
            for (int i = 0; i < NE; i += 2) {            // (i, i + 1): columns b, b + 1 of one row
                int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
                int o = m * LS + b;
                float2 l = *(float2*)&sL[o], z = *(float2*)&sZs[o];
                l.x = __fmaf_rn(l.x, al[i], __fmul_rn(tCrS(i), be[i]));
                l.y = __fmaf_rn(l.y, al[i + 1], __fmul_rn(tCrS(i + 1), be[i + 1]));
                z.x = __fmaf_rn(z.x, sc[i], tCrS2(i));
                z.y = __fmaf_rn(z.y, sc[i + 1], tCrS2(i + 1));
                *(float2*)&sL[o] = l; *(float2*)&sZs[o] = z;
            }
            if (kt + 1 < knt) bar_arrive(1 + (g + 1) % G, 2 * NCG);
        }
    }

    // log N - log Z, and log Z for the backward: the fork's epilogue (-inf where Z is 0 or N is -inf)
    const float* fl = f + (tile0 + knt - 1) * NF * (long)batch + b0;
    auto store = [&](int m, int b, float L, float Z) {
        const float mz = fl[4 * (long)batch + b], mn = fl[3 * (long)batch + b];
        const float log_z = (Z <= 0.0f || mz == -INFINITY) ? -INFINITY : mz + logf(Z);
        const float log_n = (mn == -INFINITY) ? -INFINITY : (logf(L) + mn);
        node_mars[(off_nid + (long)tile_id * BM + m) * (long)batch + b0 + b] =
            (log_n == -INFINITY || log_z == -INFINITY) ? -INFINITY : (log_n - log_z);
        if (log_z_out != nullptr)
            log_z_out[((long)nblock * block_size + tile_id * BM + m) * (long)batch + b0 + b] = log_z;
    };
    if constexpr (G == 1) {
        CUTE_UNROLL
        for (int i = 0; i < NE; i++) store(get<0>(tCcC(i)), get<1>(tCcC(i)), tCrL(i), tCrZ(i));
    } else {
        bar_sync(G + 1, NC);                             // every merge done
        for (int i = tid; i < BM * BN; i += NC) store(i / BN, i % BN, sL[(i / BN) * LS + i % BN], sZs[(i / BN) * LS + i % BN]);
    }
}

template <int BM, int BN, int WM, int WN, int G, int PW, int ST, typename TP>
static void launch_cfg(float* node_mars, const TP* params, const long* nids, int n_nblocks, const long* pids,
                       const __nv_bfloat16* bx, const float* f, const __nv_bfloat16* zg, float* log_z, int batch,
                       int block_size, int knt, int gsh, cudaStream_t stream) {
    constexpr int NTH = (G * WM * WN + PW) * 32;
    constexpr int smem = smem_bytes<BM, BN, G, ST>();
    static_assert(smem <= 101376, "a config past sm_120's 99 KB of shared memory a block");
    auto kernel = bs_stream_kernel<BM, BN, WM, WN, G, PW, ST, TP>;
    // the shared-memory attribute once per device and config (the call costs host time on every launch); the flags
    // are this instantiation's own
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_set[dev] = true;
    }
    int total_m = n_nblocks * (block_size / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        kernel<<<grid, NTH, smem, stream>>>(node_mars, params, nids, pids, bx, f, zg, log_z, batch, block_size, knt,
                                            off, gsh);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// id -> {BM, BN, WM, WN, G, PW, ST}. MEASURED (the gated HMM layers above): the four-group tiles take the small grids
// (32 x 32 at HMM-1024 B <= 128 bf16; 8 producer warps for fp32; 64 x 32 / 32 x 64 at HMM-4096 B 64), the one-group
// 64 x 64 tiles the large ones (2 producer warps at bf16, 4 at fp32).
#define CFG_TABLE(X)                     \
    X(0, 32, 32, 2, 1, 4, 4, 8)          \
    X(1, 64, 32, 4, 1, 4, 4, 5)          \
    X(2, 32, 64, 2, 2, 4, 4, 5)          \
    X(3, 32, 32, 2, 1, 4, 8, 8)          \
    X(4, 64, 64, 4, 2, 1, 4, 4)          \
    X(5, 64, 64, 4, 2, 1, 2, 4)

template <typename TP>
static void dispatch_cfg(float* nm, const TP* p, const long* nids, int nb, const long* pp, const __nv_bfloat16* bx,
                         const float* f, const __nv_bfloat16* zg, float* lz, int b, int bs, int k, int gsh, int64_t cfg,
                         cudaStream_t st) {
    switch (cfg) {
#define X(id, bm, bn, wm, wn, gg, pw, stg) \
        case id: launch_cfg<bm, bn, wm, wn, gg, pw, stg, TP>(nm, p, nids, nb, pp, bx, f, zg, lz, b, bs, k, gsh, st); break;
        CFG_TABLE(X)
#undef X
        default: TORCH_CHECK(false, "blockscale forward pre-pass: no config ", cfg);
    }
}

std::vector<std::vector<int64_t>> configs() {
    std::vector<std::vector<int64_t>> out;
#define X(id, bm, bn, wm, wn, gg, pw, stg) out.push_back({bm, bn});
    CFG_TABLE(X)
#undef X
    return out;
}

// The config ids that serve this layer: its tile divides the node block and the batch, and the layer is in scope.
std::vector<int64_t> fitting_configs(int64_t block_size, int64_t batch, int64_t node_cbs, int64_t gate_cbs) {
    std::vector<int64_t> out;
    const int64_t gc = gate_cbs < BK ? gate_cbs : BK;
    if (batch % PB != 0 || node_cbs < BK || gc < 4) return out;
    const auto cfgs = configs();
    for (int i = 0; i < (int)cfgs.size(); i++)
        if (block_size % cfgs[i][0] == 0 && batch % cfgs[i][1] == 0) out.push_back(i);
    return out;
}

// The fork's arguments (`blockscale_forward`), in its SINGLE_RUN case: `cids` / `pids` one entry per edge tile.
void blockscale_forward_prepass(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                                torch::Tensor ext, torch::Tensor nids, torch::Tensor cids, torch::Tensor pids,
                                torch::Tensor gate, torch::Tensor log_z, int64_t block_size, int64_t num_edges,
                                int64_t node_cbs, int64_t gate_cbs, int64_t n_node_gates, int64_t ext_base, int64_t cfg) {
    const int batch = node_mars.size(1);
    const int knt = (int)(num_edges / BK);
    const int gc = (int)(gate_cbs < BK ? gate_cbs : BK);
    TORCH_CHECK(n_node_gates == 1, "blockscale forward pre-pass: one gate per node block; got ", n_node_gates);
    TORCH_CHECK(batch % PB == 0 && num_edges % BK == 0 && node_cbs >= BK && gc >= 4,
                "blockscale forward pre-pass: batch % 64, num_edges % 64, node_cbs >= 64 and a gate width >= 4");
    TORCH_CHECK((node_cbs & (node_cbs - 1)) == 0 && (gate_cbs & (gate_cbs - 1)) == 0 && node_cbs % gate_cbs == 0,
                "blockscale forward pre-pass: power-of-two child blocks, the gate's dividing the node's");
    TORCH_CHECK(element_mars.is_contiguous() && node_mars.is_contiguous() && ext.is_contiguous() &&
                gate.is_contiguous() && nids.is_contiguous() && pids.is_contiguous() && cids.is_contiguous(),
                "blockscale forward pre-pass: contiguous tensors");
    TORCH_CHECK(cids.numel() == nids.numel() * knt && pids.numel() == cids.numel(),
                "blockscale forward pre-pass: cids / pids one entry per (node block, edge tile)");
    const int node_sh = (int)std::log2((double)node_cbs);
    const int gate_sh = (int)std::log2((double)gate_cbs);
    const int gsh = (int)std::log2((double)gc), ng = BK / gc;
    const long tiles = cids.numel();
    // scratch, ONE allocation: bx bf16 [tiles, 64, B], zg bf16 [tiles, QG, B], mx and gm [tiles, B], lp [tiles, QG, B],
    // f [tiles, NF, B] (every region 16-byte aligned: B % 64 == 0)
    const long o_zg = tiles * BK * batch * 2, o_mx = o_zg + tiles * QG * batch * 2, o_gm = o_mx + tiles * batch * 4;
    const long o_lp = o_gm + tiles * batch * 4, o_f = o_lp + tiles * QG * batch * 4, total = o_f + tiles * NF * batch * 4;
    torch::Tensor scratch = torch::empty({total}, element_mars.options().dtype(torch::kUInt8));
    char* base = reinterpret_cast<char*>(scratch.data_ptr());
    auto* bx = reinterpret_cast<__nv_bfloat16*>(base);
    auto* zg = reinterpret_cast<__nv_bfloat16*>(base + o_zg);
    auto* mx = reinterpret_cast<float*>(base + o_mx);
    auto* gm = reinterpret_cast<float*>(base + o_gm);
    auto* lp = reinterpret_cast<float*>(base + o_lp);
    auto* f = reinterpret_cast<float*>(base + o_f);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    bs_prepass_kernel<256><<<dim3(tiles, batch / PB), 256, 0, st>>>(
        element_mars.data_ptr<float>(), cids.data_ptr<long>(), ext.data_ptr<float>(), gate.data_ptr<long>(), bx, mx,
        gm, lp, batch, knt, (int)gate.size(1), (int)node_cbs, gc, node_sh, gate_sh, (long)ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int nb = (int)nids.size(0);
    bs_factors_kernel<256, 8><<<dim3(nb, batch / 32, (knt + 7) / 8), 256, 0, st>>>(mx, gm, lp, f, zg, batch, knt, ng);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    float* lz = log_z.numel() ? log_z.data_ptr<float>() : nullptr;
    const int bs = (int)block_size;
    if (params.scalar_type() == torch::kBFloat16)
        dispatch_cfg<__nv_bfloat16>(node_mars.data_ptr<float>(), param_ptr<__nv_bfloat16>(params), nids.data_ptr<long>(),
                                    nb, pids.data_ptr<long>(), bx, f, zg, lz, batch, bs, knt, gsh, cfg, st);
    else
        dispatch_cfg<float>(node_mars.data_ptr<float>(), param_ptr<float>(params), nids.data_ptr<long>(), nb,
                            pids.data_ptr<long>(), bx, f, zg, lz, batch, bs, knt, gsh, cfg, st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_forward_prepass", &blockscale_forward_prepass,
          "the block-scale forward as a pre-pass, its merge factors and a streaming kernel (log N - log Z)");
    m.def("configs", &configs, "(BM, BN) per config id");
    m.def("fitting_configs", &fitting_configs, "the config ids that serve a layer (block_size, batch, node_cbs, gate_cbs)");
}
