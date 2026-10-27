// The `BlockScaleSumParams` element-flow backward (bf16 parameters) as a pre-pass, a split GEMM and a merge -- what
// the pre-pass element kernel (blockscale_ele_backward_prepass.cu) computes, not to its bits:
//
//     element_flows[m, b] = emars[m, b] + log sum_t sum_e params_t[m, e] exp(lf_t[e, b] + log phi_t[r(m), b])
//     d LL / d log phi[g_t + r, b] += sum_{m in row r} sum_e params_t[m, e] exp(emars[m, b] + lf_t[e, b] + log phi)
//
// with lf = node_flows - node_mars over a k-tile t's 64 parents, g_t its gate (`gate`, -1 without one) and r(m) a
// child's gate row. A RUN is a maximal sequence of k-tiles under one gate (a parent block's); the GEMM's work is cut
// into `split` equal ranges of an element block's gated k-steps, and a PIECE is a range's part inside one run (the
// host's tables, one int32 tensor: per piece its k-steps and gate, per range its pieces). Each piece has its own
// stabilizer and gate, so:
//   pre-pass  per (group, piece, PB columns): M = the piece's max lf, u = exp(lf - M) as bf16 hi + lo halves (the
//             exponentials to ~2^-17), laid out [group, parent position, b]. Element blocks of equal parents and pieces
//             form a GROUP and share them (an HMM's element blocks: one group);
//   GEMM      per (batch tile, child tile, range): L_p = params . u_hi + params . u_lo over each piece, two bf16 MMAs
//             into one fp32 accumulator (the stored bf16 parameters exact, nothing converted), stored at the piece's
//             end -- producers cp.async the parameter tile (K-major, rows contiguous in shared memory: the swizzled
//             row-interleaved atom ran cp.async at half rate) and both halves into mbarrier'd stages;
//   merge     per (element block, a gate row's children or 64, 16 columns): t_p = log L_p + M_p + log phi_p(r),
//             element_flows = emars + LSE_p t_p (-inf where every t is); the gradient exp(emars + t_p) over a warp's
//             children by a fixed butterfly, over the gate row's warps and chunks in order, a run's pieces in order,
//             one atomic add per (run, gate row, column) -- the bits fixed wherever no two runs share a gate row.
//
// MEASURED (RTX PRO 6000, a gated HMM layer of 4 element blocks of 1024 from 4 parent blocks of 1024, gate width 64,
// bf16, each kernel's best tile and split / config): against the pre-pass element kernel B 128 57.8 -> 42.1 us, B 512
// 172.1 -> 122.8; against float64 the element flows' max error 3-5e-6 (the pre-pass kernel's 3.6e-5) and the
// gradient's 2-3e-6 (1.5e-5), and no flow lost where the pre-pass kernel's fp16 parameters flush (children whose
// parameters are all < 6e-8). Three launches: ~2 us more host time a call than the pre-pass kernel's two (7.1 / 5.1).
//
// Scope (`fitting_configs`, and the Python plan for the rest): bf16 parameters with 8-aligned rows, k-tiles of 64
// parents within one parent block (contiguous node ids), a power-of-two gate width from 4 to the element block,
// batch % 64 == 0, the element block a multiple of 64 and of the tile's rows, at most PMAX pieces an element block.
// Scratch: one allocation per call (inside a CUDA graph capture, from its pool).

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

#define BK 64           // k-tile: parents (one run of node ids, one gate)
#define CH 32           // the pre-pass's chunk of parents
#define PMAX 16         // pieces an element block at most
#define PQ 2            // pieces the merge takes at once (MEASURED: 4 needs more registers, slower)

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
    asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 " @!p bra W_%=;\n}\n" ::"r"(a), "r"(phase) : "memory");
}

// ---------------- pre-pass: a block per (group, piece, PB columns) ----------------
// Thread (part, b) takes the piece's 32-parent chunks part, part + PARTS, ...: lf in registers, the piece's max in
// shared memory, the exponentials split into hi + lo. A piece past PARTS chunks loads its later chunks twice.
template <int NT, int PB>
__global__ void __launch_bounds__(NT) bs_eles_prepass_kernel(
        const float* __restrict__ nf, const float* __restrict__ nm, const long* __restrict__ ebase,
        const int* __restrict__ pc_k0, const int* __restrict__ pc_k1, const int* __restrict__ grep,
        __nv_bfloat16* __restrict__ uh, __nv_bfloat16* __restrict__ ul, float* __restrict__ mp, int batch, int knt,
        int kt_sh, int npc) {
    constexpr int PARTS = NT / PB;
    __shared__ float smx[PARTS][PB];
    const int tid = threadIdx.x, nbc = batch / PB;
    const int grp = blockIdx.x / (npc * nbc), p = (blockIdx.x / nbc) % npc, b0 = (blockIdx.x % nbc) * PB;
    const int eb = grep[grp];
    const int b = tid % PB, part = tid / PB;
    const int c0 = pc_k0[eb * PMAX + p] << kt_sh, c1 = pc_k1[eb * PMAX + p] << kt_sh;   // parent positions
    const int q0 = c0 / CH, q1 = c1 / CH;
    const long* et = ebase + (long)eb * knt;
    auto load_chunk = [&](int qc, float* v) {
        const int pos0 = qc * CH;
        const long r0 = (et[pos0 / BK] + pos0 % BK) * (long)batch + b0 + b;
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) {
            const float f = nf[r0 + (long)i * batch], n = nm[r0 + (long)i * batch];
            v[i] = (n == -INFINITY) ? -INFINITY : f - n;
        }
    };
    float v[CH];
    float m = -INFINITY;
    const int qc = q0 + part;
    if (qc < q1) {
        load_chunk(qc, v);
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) m = fmaxf(m, v[i]);
    }
    for (int qq = qc + PARTS; qq < q1; qq += PARTS) {
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
    auto store = [&](int qq, const float* w) {
        const long o = ((long)grp * knt * BK + qq * CH) * batch + b0 + b;
        CUTE_UNROLL
        for (int i = 0; i < CH; i++) {
            const float x = (m == -INFINITY || w[i] == -INFINITY) ? 0.f : __expf(w[i] - m);
            const __nv_bfloat16 h = __float2bfloat16_rn(x);
            uh[o + (long)i * batch] = h;
            ul[o + (long)i * batch] = __float2bfloat16_rn(x - __bfloat162float(h));
        }
    };
    if (qc < q1) store(qc, v);
    for (int qq = qc + PARTS; qq < q1; qq += PARTS) {
        float w[CH];
        load_chunk(qq, w);
        store(qq, w);
    }
    if (part == 0) mp[((long)grp * PMAX + p) * batch + b0 + b] = m;
}

extern __shared__ __align__(1024) char smem_raw[];

// A K-major (the parameters: [child][parent], parents contiguous) with each row's k contiguous in a 128- / 64-byte
// swizzled line: a cp.async chunk of a row lands in that row's line (the row-interleaved atom of the pre-pass element
// kernel ran this stream at half rate, MEASURED: 22 -> 13.4 us of IO alone). B MN-major (u: b contiguous).
template <int KT> struct AtomA;
template <> struct AtomA<64> {
    using L = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
};
template <> struct AtomA<32> {
    using L = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
};
using AtomB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));

// ---------------- split GEMM: a block per (batch tile, child tile, range) ----------------
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
__global__ void __launch_bounds__((WM * WN + PW) * 32) bs_eles_gemm_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ pbase, const bfloat16_t* __restrict__ uh,
        const bfloat16_t* __restrict__ ul, const int* __restrict__ pc_k0, const int* __restrict__ pc_k1,
        const int* __restrict__ sp_lo, const int* __restrict__ sp_hi, const int* __restrict__ ugrp,
        float* __restrict__ part, int knt, int mtiles, int BSK, int batch, int split, long rows_total) {
    constexpr int NC = WM * WN * 32, NP = PW * 32;
    constexpr int ABYTES = KT * BM * 2, BBYTES = KT * BN * 2;
    constexpr int STAGE = (ABYTES + 2 * BBYTES + 1023) / 1024 * 1024;
    constexpr int SUB = BK / KT;                         // k-steps a k-tile
    const int pid_b = blockIdx.x, pid_m = blockIdx.y, q = blockIdx.z;
    const int eb = pid_m / mtiles, tile_id = pid_m % mtiles;
    const int plo = sp_lo[eb * split + q], phi = sp_hi[eb * split + q];
    if (plo >= phi) return;                              // (a range without a gated k-step)
    uint64_t* full = (uint64_t*)(smem_raw + ST * STAGE);
    uint64_t* empty = full + ST;
    const int tid = threadIdx.x;
    auto sAl = tile_to_shape(typename AtomA<KT>::L{}, make_shape(Int<BM>{}, Int<KT>{}));
    auto sBl = tile_to_shape(AtomB{}, make_shape(Int<BN>{}, Int<KT>{}));
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s, int h) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES + h * BBYTES); };
    if (tid == 0)
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], NP); mbar_init(&empty[s], NC); }
    __syncthreads();
    const int* k0p = pc_k0 + eb * PMAX;
    const int* k1p = pc_k1 + eb * PMAX;
    if (tid >= NC) {
        // ---------------- producers: the range's k-steps, piece by piece, into stage j % ST ----------------
        const int ptid = tid - NC;
        // a thread's 16-byte chunks are the same every k-step: their shared offsets and global offsets, once
        constexpr int TA = ABYTES / 16, TB = BBYTES / 16, CA = (TA + NP - 1) / NP, CB = (TB + NP - 1) / NP;
        uint32_t oA[CA], oB[CB];
        long gA[CA];
        int gB[CB];
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(0, 0)), sBl);
            CUTE_UNROLL
            for (int qq = 0; qq < CA; qq++) {
                int i = ptid + qq * NP, m = i / (KT / 8), kk = (i % (KT / 8)) * 8;
                oA[qq] = i < TA ? (uint32_t)((char*)&sAt(m, kk) - smem_raw) : ~0u;
                gA[qq] = (long)m * BSK + kk;
            }
            CUTE_UNROLL
            for (int qq = 0; qq < CB; qq++) {
                int i = ptid + qq * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oB[qq] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - smem_raw) : ~0u;
                gB[qq] = e * batch + bb;
            }
        }
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        const long* pr = pbase + (long)eb * knt;
        const long ub = (long)ugrp[eb] * knt * BK * batch + (long)pid_b * BN;
        int j = 0;
        for (int pc = plo; pc < phi; pc++) {
            for (int t = k0p[pc]; t < k1p[pc]; t++, j++) {
                const int s = j % ST;
                if (j >= ST) mbar_wait(&empty[s], ((j / ST) - 1) & 1);
                const uint32_t ss = sbase + s * STAGE;
                const int t64 = t / SUB, sub = t % SUB;
                const bfloat16_t* pa = params + pr[t64] + (long)tile_id * BM * BSK + sub * KT;
                CUTE_UNROLL
                for (int qq = 0; qq < CA; qq++) if (oA[qq] != ~0u) cpa16(ss + oA[qq], pa + gA[qq]);
                const long ob = ub + (long)t * KT * batch;
                CUTE_UNROLL
                for (int h = 0; h < 2; h++) {
                    const bfloat16_t* pb = (h == 0 ? uh : ul) + ob;
                    CUTE_UNROLL
                    for (int qq = 0; qq < CB; qq++)
                        if (oB[qq] != ~0u) cpa16(ss + h * BBYTES + oB[qq], pb + gB[qq]);
                }
                cpa_mbar_arrive(&full[s]);
            }
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }
    // ---------------- consumers: a piece's k-steps into one accumulator, stored at its end ----------------
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0, 0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    Tensor tCrB2 = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    auto tA = s2rA.get_thread_slice(tid);
    auto tB = s2rB.get_thread_slice(tid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor tXrB2 = tB.retile_D(tCrB2);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    constexpr int KB = size<2>(tCrA);
    int j = 0;
    for (int pc = plo; pc < phi; pc++) {
        clear(tCrL);
        for (int t = k0p[pc]; t < k1p[pc]; t++, j++) {
            const int s = j % ST;
            mbar_wait(&full[s], (j / ST) & 1);
            auto tXsA = tA.partition_S(make_tensor(make_smem_ptr(stA(s)), sAl));
            auto tXsB = tB.partition_S(make_tensor(make_smem_ptr(stB(s, 0)), sBl));
            auto tXsB2 = tB.partition_S(make_tensor(make_smem_ptr(stB(s, 1)), sBl));
            copy(s2rA, tXsA(_, _, 0), tXrA(_, _, 0));
            copy(s2rB, tXsB(_, _, 0), tXrB(_, _, 0));
            copy(s2rB, tXsB2(_, _, 0), tXrB2(_, _, 0));
            CUTE_UNROLL
            for (int kb = 0; kb < KB; kb++) {            // block kb + 1's fragments load while block kb multiplies
                if (kb + 1 < KB) {
                    copy(s2rA, tXsA(_, _, kb + 1), tXrA(_, _, kb + 1));
                    copy(s2rB, tXsB(_, _, kb + 1), tXrB(_, _, kb + 1));
                    copy(s2rB, tXsB2(_, _, kb + 1), tXrB2(_, _, kb + 1));
                } else {
                    mbar_arrive(&empty[s]);              // the stage is read
                }
                cute::gemm(mma, tCrA(_, _, kb), tCrB2(_, _, kb), tCrL);   // the lo half first
                cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrL);
            }
        }
        float* o = part + ((long)pc * rows_total + (long)eb * mtiles * BM + (long)tile_id * BM) * batch + pid_b * BN;
        CUTE_UNROLL
        for (int i = 0; i < size(tCrL); i += 2)          // (i, i + 1): columns b, b + 1 of one row
            *(float2*)&o[(long)get<0>(tCcC(i)) * batch + get<1>(tCcC(i))] = make_float2(tCrL(i), tCrL(i + 1));
    }
}

// one reduce-scatter round over lanes `ofs` apart: the lane with that bit keeps the upper W values, its partner the
// lower W, and each adds the half the other sends
template <int W>
__device__ __forceinline__ void rs_round(float* v, int lane, int ofs) {
    const bool up = (lane & ofs) != 0;
    CUTE_UNROLL
    for (int i = 0; i < W; i++) {
        const float send = up ? v[i] : v[i + W];
        const float keep = up ? v[i + W] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, ofs);
    }
}

// ---------------- merge: a block per (element block, R children, 16 columns), a thread per (child, 4 columns) ------
// R = the larger of 64 and the gate width: a gate row inside one block. The pieces PQ at a time under an online LSE;
// each thread loads its own partials, maxima, gates and log phi (no staging, no barrier before the loads). The
// gradient: per piece exp(emars + t) over a warp's `span` children by a fixed reduce-scatter butterfly -- each lane
// ends with 4 PQ / span of the sums, in shared memory slots of its own, accumulated there over the block's 64-child
// chunks in order; after one barrier, per (piece, gate row, column) the gate row's sums in order, a run's pieces in
// order by its first. MEASURED against a shuffle tree with a filled table (HMM-4096 x4, gate width 64, 4 pieces):
// 24.2 -> 15.7 us at B 512, 8.2 -> 5.9 at B 128 -- the registers for 4 blocks an SM, 3x fewer shuffles.
__global__ void __launch_bounds__(256, 4) bs_eles_merge_kernel(
        float* __restrict__ ef, const float* __restrict__ emars, const float* __restrict__ part,
        const float* __restrict__ mp, const float* __restrict__ ext, const int* __restrict__ pc_gate,
        const long* __restrict__ chids, const int* __restrict__ ugrp, float* __restrict__ grad, int batch, int BSM,
        int gc, int R, long ext_base, int npc, long rows_total, int wgrad) {
    __shared__ float sg[16][PMAX][16];                   // [group of children][piece][column]: summed over chunks
    const int tid = threadIdx.x, mi = tid / 4, cq = tid % 4, lane = tid % 32;
    const int nrb = BSM / R, nbc = batch / 16;
    const int eb = blockIdx.x / (nrb * nbc), mb = ((blockIdx.x / nbc) % nrb) * R, b0 = (blockIdx.x % nbc) * 16;
    const int b = b0 + 4 * cq;
    const long eoff = chids[eb];
    const int grp = ugrp[eb];
    const int* gt = pc_gate + eb * PMAX;
    const int span = gc < 8 ? gc : 8;                    // children a reduction group (lane bits 2.. the child)
    for (int m0 = mb; m0 < mb + R; m0 += 64) {
        const int m = m0 + mi, r = m / gc;
        const long row = (long)eb * BSM + m;
        const float4 E = *(const float4*)&emars[(eoff + m) * (long)batch + b];
        float T[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, S[4] = {0.f, 0.f, 0.f, 0.f};
        for (int p0 = 0; p0 < npc; p0 += PQ) {
            float4 L[PQ], M[PQ], lp[PQ];
            int G[PQ];
            // (a piece without a gate -- padding -- has M = -inf from the pre-pass and its partial never written)
            CUTE_UNROLL
            for (int k = 0; k < PQ; k++) {
                const int p = p0 + k;
                G[k] = p < npc ? gt[p] : -1;
                L[k] = p < npc ? *(const float4*)&part[((long)p * rows_total + row) * batch + b]
                               : make_float4(0.f, 0.f, 0.f, 0.f);
                M[k] = p < npc ? *(const float4*)&mp[((long)grp * PMAX + p) * batch + b]
                               : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            }
            CUTE_UNROLL
            for (int k = 0; k < PQ; k++)
                lp[k] = G[k] >= 0 ? *(const float4*)&ext[((long)G[k] + ext_base + r) * batch + b]
                                  : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            float gv[PQ][4];
            CUTE_UNROLL
            for (int c = 0; c < 4; c++) {
                const float e = (&E.x)[c];
                float t[PQ], Tn = T[c];
                CUTE_UNROLL
                for (int k = 0; k < PQ; k++) {
                    const float Lv = (&L[k].x)[c], Mv = (&M[k].x)[c], lv = (&lp[k].x)[c];
                    t[k] = (Lv > 0.f && Mv != -INFINITY && lv != -INFINITY) ? __logf(Lv) + Mv + lv : -INFINITY;
                    Tn = fmaxf(Tn, t[k]);
                }
                float s = (T[c] == -INFINITY) ? 0.f : S[c] * __expf(T[c] - Tn);
                CUTE_UNROLL
                for (int k = 0; k < PQ; k++) s += (t[k] == -INFINITY) ? 0.f : __expf(t[k] - Tn);
                T[c] = Tn; S[c] = s;
                CUTE_UNROLL
                for (int k = 0; k < PQ; k++) gv[k][c] = (t[k] == -INFINITY || e == -INFINITY) ? 0.f : __expf(e + t[k]);
            }
            if (wgrad) {
                // v[k * 4 + c] = gv[k][c]; after the rounds a lane holds the sums of v[i + NV / span * (mi % span)]
                constexpr int NV = 4 * PQ;
                float* v = &gv[0][0];
                auto keep = [&](auto nvc, int ci) {      // a lane's slots are its own in every chunk: no race
                    constexpr int nv = decltype(nvc)::value;
                    CUTE_UNROLL
                    for (int i = 0; i < nv; i++) {
                        const int idx = i + nv * ci, k = idx / 4, c = idx % 4;
                        if (p0 + k < npc) {
                            float& d = sg[mi / span][p0 + k][4 * cq + c];
                            d = (m0 == mb) ? v[i] : d + v[i];
                        }
                    }
                };
                if (span == 8) {
                    rs_round<NV / 2>(v, lane, 16); rs_round<NV / 4>(v, lane, 8); rs_round<NV / 8>(v, lane, 4);
                    keep(Int<NV / 8>{}, mi % 8);
                } else {                                 // (a gate width of 4)
                    rs_round<NV / 2>(v, lane, 8); rs_round<NV / 4>(v, lane, 4);
                    keep(Int<NV / 4>{}, mi % 4);
                }
            }
        }
        float o[4];
        CUTE_UNROLL
        for (int c = 0; c < 4; c++) o[c] = (T[c] == -INFINITY) ? -INFINITY : (&E.x)[c] + T[c] + __logf(S[c]);
        *(float4*)&ef[(eoff + m) * (long)batch + b] = make_float4(o[0], o[1], o[2], o[3]);
    }
    if (!wgrad) return;
    __syncthreads();
    // per (a run's first piece, gate row jj of the block, column c): the row's groups, then the run's pieces, in order
    const int ngr = R / gc;                              // gate rows in the block
    const int gpr = (gc < 64 ? gc : 64) / span;          // groups of a gate row in a chunk
    for (int pidx = tid; pidx < npc * ngr * 16; pidx += 256) {
        const int p = pidx / (ngr * 16), jj = (pidx / 16) % ngr, c = pidx % 16;
        const int g = gt[p];
        if (g < 0 || (p > 0 && gt[p - 1] == g)) continue;
        float acc = 0.f;
        for (int pp = p; pp < npc && gt[pp] == g; pp++)
            for (int k = 0; k < gpr; k++) acc += sg[(jj * gpr + k) % 16][pp][c];
        atomicAdd(&grad[((long)g + ext_base + mb / gc + jj) * batch + b0 + c], acc);
    }
}

// ---------------- host ----------------
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
static void launch_gemm(const bfloat16_t* params, const long* pbase, const bfloat16_t* uh, const bfloat16_t* ul,
                        const int* pc_k0, const int* pc_k1, const int* sp_lo, const int* sp_hi, const int* ugrp,
                        float* part, int knt, int neb, int BSM, int BSK, int batch, int split, cudaStream_t st) {
    constexpr int STAGE = (KT * BM * 2 + 2 * KT * BN * 2 + 1023) / 1024 * 1024;
    constexpr int SMEM = ST * STAGE + 2 * ST * 8;
    static_assert(SMEM <= 101376, "a config past sm_120's 99 KB of shared memory a block");
    auto kernel = bs_eles_gemm_kernel<BM, BN, WM, WN, PW, ST, KT>;
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if (dev < 64) attr_set[dev] = true;
    }
    const int mtiles = BSM / BM;
    TORCH_CHECK((long)neb * mtiles < 65536, "blockscale element stream: past 65535 child tiles");
    dim3 grid(batch / BN, neb * mtiles, split);
    kernel<<<grid, (WM * WN + PW) * 32, SMEM, st>>>(params, pbase, uh, ul, pc_k0, pc_k1, sp_lo, sp_hi, ugrp, part, knt,
                                                    mtiles, BSK, batch, split, (long)neb * BSM);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// id -> {BM, BN, WM, WN, PW, ST, KT}
#define CFG_TABLE(X)                     \
    X(0, 128, 128, 4, 2, 4, 2, 64)       \
    X(1, 128, 128, 4, 2, 4, 4, 32)       \
    X(2, 128, 64, 4, 2, 4, 3, 64)        \
    X(3, 64, 128, 2, 4, 4, 2, 64)

struct TileCfg { int bm, bn, kt; };
static constexpr TileCfg CFGS[] = {
#define X(id, bm, bn, wm, wn, pw, st, kt) {bm, bn, kt},
    CFG_TABLE(X)
#undef X
};
static constexpr int NCFG = sizeof(CFGS) / sizeof(CFGS[0]);

std::vector<std::vector<int64_t>> configs() {
    std::vector<std::vector<int64_t>> out;
    for (const auto& c : CFGS) out.push_back({c.bm, c.bn, c.kt});
    return out;
}

// The config ids that serve a layer: the tile divides the element block and the batch, and the layer is in scope.
std::vector<int64_t> fitting_configs(int64_t cs_block_size, int64_t batch, int64_t gate_cbs) {
    std::vector<int64_t> out;
    if (batch % 64 != 0 || cs_block_size % 64 != 0 || gate_cbs < 4 || (gate_cbs & (gate_cbs - 1)) != 0 ||
        (gate_cbs < cs_block_size ? cs_block_size % gate_cbs : gate_cbs % cs_block_size) != 0 ||
        gate_cbs > cs_block_size)
        return out;
    for (int i = 0; i < NCFG; i++)
        if (cs_block_size % CFGS[i].bm == 0 && batch % CFGS[i].bn == 0) out.push_back(i);
    return out;
}

int64_t max_pieces() { return PMAX; }

// The pre-pass kernel's arguments (`blockscale_ele_backward_prepass`) without its gate table, then the plan's tables
// as ONE int32 tensor (one argument, not seven: the call's host time is what a host-bound backward pays) -- per element
// block the pieces' first and end k-steps and gates [neb, PMAX] each, the ranges' first and end pieces [neb, split]
// each, its group [neb], each group's first element block [ngrp]; `cfg` = config id * 64 + split.
void blockscale_ele_backward_stream(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                                    torch::Tensor node_mars, torch::Tensor params, torch::Tensor ext,
                                    torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase,
                                    torch::Tensor grad_ext, int64_t batch, int64_t block_size, int64_t cs_block_size,
                                    int64_t knt, int64_t gate_cbs, int64_t ext_base, int64_t write_grad, int64_t cfg,
                                    torch::Tensor tables, int64_t ngrp_, int64_t npc) {
    const int cfg_id = (int)(cfg / 64), split = (int)(cfg % 64);
    TORCH_CHECK(cfg_id >= 0 && cfg_id < NCFG && split >= 1, "blockscale element stream: no config ", cfg);
    const int BMc = CFGS[cfg_id].bm, BNc = CFGS[cfg_id].bn, kt = CFGS[cfg_id].kt;
    const int kt_sh = kt == 64 ? 6 : 5;
    const int neb = (int)chids.size(0), ngrp = (int)ngrp_;
    TORCH_CHECK(batch % 64 == 0 && batch % BNc == 0 && cs_block_size % 64 == 0 && cs_block_size % BMc == 0 &&
                gate_cbs >= 4 && (gate_cbs & (gate_cbs - 1)) == 0 && gate_cbs <= cs_block_size,
                "blockscale element stream: batch % 64, the element block a multiple of 64 and of the tile, a "
                "power-of-two gate width >= 4 within the element block");
    TORCH_CHECK(npc >= 1 && npc <= PMAX, "blockscale element stream: 1 .. ", PMAX, " pieces an element block");
    TORCH_CHECK(chids.scalar_type() == torch::kLong && ebase.scalar_type() == torch::kLong &&
                pbase.scalar_type() == torch::kLong && chids.is_contiguous() && ebase.is_contiguous() &&
                pbase.is_contiguous() && ebase.numel() == (long)neb * knt && pbase.numel() == (long)neb * knt,
                "blockscale element stream: int64 tables [neb, knt]");
    TORCH_CHECK(tables.scalar_type() == torch::kInt && tables.is_contiguous() && ngrp >= 1 &&
                tables.numel() == 3L * neb * PMAX + 2L * neb * split + neb + ngrp,
                "blockscale element stream: the plan's int32 tables");
    const int* tb = tables.data_ptr<int>();
    const int *pc_k0 = tb, *pc_k1 = pc_k0 + neb * PMAX, *pc_gate = pc_k1 + neb * PMAX, *sp_lo = pc_gate + neb * PMAX;
    const int *sp_hi = sp_lo + neb * split, *ugrp = sp_hi + neb * split, *grep = ugrp + neb;
    TORCH_CHECK(element_flows.is_contiguous() && element_mars.is_contiguous() && node_flows.is_contiguous() &&
                node_mars.is_contiguous() && ext.is_contiguous(), "blockscale element stream: contiguous tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(ext.data_ptr()) % 16 == 0, "the gate parameters 16-byte aligned");
    TORCH_CHECK(!write_grad || grad_ext.numel() > 0, "write_grad without a gradient buffer");
    const at::cuda::OptionalCUDAGuard guard(element_flows.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    const long rows_total = (long)neb * cs_block_size;
    // scratch, ONE allocation: u hi and lo bf16 [groups, knt * 64, B], M [groups, PMAX, B], partials [PMAX, rows, B]
    const long n_u = (long)ngrp * knt * BK * batch;
    const long o_mp = 4 * n_u, o_part = o_mp + 4L * ngrp * PMAX * batch;
    torch::Tensor scratch = torch::empty({o_part + 4L * npc * rows_total * batch},
                                         element_mars.options().dtype(torch::kUInt8));
    char* base = reinterpret_cast<char*>(scratch.data_ptr());
    auto* uh = reinterpret_cast<__nv_bfloat16*>(base);
    auto* ul = uh + n_u;
    auto* mp = reinterpret_cast<float*>(base + o_mp);
    auto* part = reinterpret_cast<float*>(base + o_part);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    {
#define P_(NT, PB) bs_eles_prepass_kernel<NT, PB><<<ngrp * (int)npc * ((int)batch / PB), NT, 0, st>>>( \
            node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), ebase.data_ptr<long>(), pc_k0, pc_k1, grep, uh, \
            ul, mp, (int)batch, (int)knt, kt_sh, (int)npc)
        if (batch >= 256) P_(512, 16); else P_(256, 8);
#undef P_
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    switch (cfg_id) {
#define X(id, bm, bn, wm, wn, pw, stg, ktt) \
        case id: launch_gemm<bm, bn, wm, wn, pw, stg, ktt>(pp, pbase.data_ptr<long>(), (const bfloat16_t*)uh, \
                     (const bfloat16_t*)ul, pc_k0, pc_k1, sp_lo, sp_hi, ugrp, part, (int)knt, neb, \
                     (int)cs_block_size, (int)block_size, (int)batch, split, st); break;
        CFG_TABLE(X)
#undef X
    }
    {
        const int R = gate_cbs > 64 ? (int)gate_cbs : 64;
        const long nblk = (long)neb * (cs_block_size / R) * (batch / 16);
        TORCH_CHECK(nblk < (1L << 31), "blockscale element stream: a merge grid past 2^31 blocks");
        bs_eles_merge_kernel<<<(unsigned)nblk, 256, 0, st>>>(
            element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), part, mp, ext.data_ptr<float>(), pc_gate,
            chids.data_ptr<long>(), ugrp, write_grad ? grad_ext.data_ptr<float>() : nullptr, (int)batch,
            (int)cs_block_size, (int)gate_cbs, R, (long)ext_base, (int)npc, rows_total, (int)write_grad);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_ele_backward_stream", &blockscale_ele_backward_stream,
          "the block-scale element-flow backward (and d LL / d log phi) as a pre-pass, a split GEMM and a merge");
    m.def("configs", &configs, "(BM, BN, KT) per config id");
    m.def("fitting_configs", &fitting_configs, "the config ids that serve a layer (cs_block_size, batch, gate_cbs)");
    m.def("max_pieces", &max_pieces, "pieces an element block at most");
}
