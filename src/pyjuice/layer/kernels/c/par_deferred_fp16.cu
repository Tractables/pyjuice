// A tied group's parameter flows, deferred to one contraction (`pyjuice.layer.deferred_par`): a regular sum layer's,
// or a `BlockScaleSumParams` layer's with its gates (`gated`: log phi from the gate table and `ext`; a regular layer's
// log phi is 0 and the two are unread) -- with fp16 operands: the contraction of the TF32-operand kernels it replaced
// (fp32 slots, mma m16n8k8 TF32; until 4f535bfd), its operands at half the bytes and its MMAs at twice the rate. The
// capture and the flush:
//   capture   per step (`capture`): the column maxima M[nb, b] = max_n lf over the node block (lf = node_flows -
//             node_mars, -inf where node_mars is; 0 where every node is -inf), as the TF32 kernels'; then both
//             operands' rows, A = exp(lf - M) and Bop = exp(emars[cid] + log phi[gate row] + M) (0 without a gate), in
//             fp32 and stored as fp16 with a power-of-two scale per (row, slot): the row's max over the slot's columns
//             m = f 2^x (f in [0.5, 1)), the stored values v 2^(15 - x) (the max in [2^14, 2^15): 2^-29 of it at full
//             precision, 2^-39 subnormal), the scale 2^x (fA[nb, slot, n], fB[nb, slot, e]). A slot is at most 512 of a
//             step's columns: a batch past 512 spreads a step over `cps` slots, each with its rows' own scales;
//   flush     (`flush`): per (BN edges, BM nodes, node block), slot by slot, acc = A16 . Bop16^T over the slot's
//             columns (cp.async stages, ldmatrix, mma.sync m16n8k16 fp16, fp32 accumulate), folded into the total as
//             tot += (acc 2^-30) (fA[n] fB[e]) -- 2^-30 first, so the scales' product underflows only where every
//             product of the pair is below 2^-126, as the TF32 operands' own products would -- then
//             pf[pfids[nb, e] + n] += theta[pids[nb, e] + n] * tot[n, e]: theta * tot staged in shared memory and one
//             bulk reduce-add (cp.reduce.async.bulk .add.f32) per edge's run of BM flows, the L2 adding them while
//             the SM goes on (theta prefetched into L2 at the start); where the runs are not 16-byte aligned, the
//             flows' read-add-write by the threads, a chunk's loads first. Each flow is added once either way: the
//             flows are deterministic.
// fp16's 10-bit mantissa is TF32's: against float64 the flows' error is the TF32 kernels' (MEASURED, HMM-4096 in 4
// blocks, gate width 64, 7 steps, bf16 parameters: max relative error over flows >= 1e-6 of their block's max 6.70e-4
// vs 6.80e-4 at B 128, 4.35e-4 vs 4.38e-4 at B 512, none lost). An fp16 tile is half a TF32 one's bytes, and the
// flush is bound by its operands' L2 re-reads -- `tot` beside `acc` caps a tile at 128 x 128 (the registers).
//
// MEASURED (RTX PRO 6000, a tied gated HMM's flush against one TF32 flush of the same steps, bf16, L2 cold):
// HMM-4096 in 4 blocks, 7 steps, B 128 310 -> 183 us, B 512 825 -> 467 (the TF32 slots fill PAR_DEFER_MB at 3
// steps: it flushed 3 times, 999 us); HMM-2048 in 2 blocks, 15 steps, B 256 322 -> 181; HMM-8192 in 8 blocks, B 64
// 744 -> 532 (the flows' read-add-write alone 460 at the DRAM's rate). The training step: HMM-4096 B 128 -3.9%,
// B 512 -11.3%, HMM-8192 B 64 -4.0%.
//
// Scope (the Python side checks it): blk % 64 == 0, num_edges % 64 == 0, a slot's columns W % 32 == 0 and <= 512.

#include <torch/extension.h>
#include "param_load.cuh"                                   // `param_ptr<TP>`, `ParamLoad<TP>`: the stored dtype
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

// ---------------- capture ----------------
// The column maxima M[nb, b] = max_n lf as order-preserving integers (0 below every float, -inf included; the buffer
// zeroed first): a block per (node block, 32 columns, 64 nodes), 8 warps x 8 nodes, lane = column, its partial max
// combined by atomicMax
__device__ __forceinline__ unsigned f2ord(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord2f(unsigned o) {
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}

__global__ void __launch_bounds__(256) capture_m_kernel(const float* __restrict__ nflows, const float* __restrict__ nmars,
                                                        const long* __restrict__ nids, unsigned* __restrict__ Mord,
                                                        int blk, int batch, int Mp) {
    __shared__ float sm[8][32];
    const int nb = blockIdx.x, b = blockIdx.y * 32 + threadIdx.x % 32, w = threadIdx.x / 32, n0 = blockIdx.z * 64 + w * 8;
    const long base = nids[nb];
    float nf[8], nm[8];
#pragma unroll
    for (int u = 0; u < 8; u++) {
        nm[u] = -INFINITY; nf[u] = 0.f;
        if (b < batch && n0 + u < blk) {
            const long off = (base + n0 + u) * (long)batch + b;
            nm[u] = nmars[off]; nf[u] = nflows[off];
        }
    }
    float m = -INFINITY;
#pragma unroll
    for (int u = 0; u < 8; u++) m = fmaxf(m, nm[u] == -INFINITY ? -INFINITY : nf[u] - nm[u]);
    sm[w][threadIdx.x % 32] = m;
    __syncthreads();
    if (w == 0) {
#pragma unroll
        for (int q = 1; q < 8; q++) m = fmaxf(m, sm[q][threadIdx.x]);
        if (m > -INFINITY) atomicMax(Mord + (long)nb * Mp + b, f2ord(m));
    }
}

// A power of two from its exponent, e in [-126, 127] (a normal float)
__device__ __forceinline__ float pow2i(int e) { return __int_as_float((e + 127) << 23); }

// Both operands' rows of one slot (one chunk of a step's columns): a block per (node block, 8 R rows, chunk), a warp
// per R rows -- below blk A's (nodes), the rest Bop's (edges); blk % 64 == 0, so a block is all A or all Bop -- lane
// = columns lane + 32 j (j < W / 32 <= VPL); the R rows' index loads first, then all their values; the rows' maxima
// by warp shuffles, then the fp16 values and the rows' scales (the R the caller's: as many rows a warp as keep two
// blocks an SM)
template <int R, int VPL>
__global__ void __launch_bounds__(256) capture_ab_kernel(
        const float* __restrict__ nflows, const float* __restrict__ nmars, const float* __restrict__ emars,
        const float* __restrict__ ext, const long* __restrict__ nids, const long* __restrict__ cids,
        const long* __restrict__ gate, const unsigned* __restrict__ Mord, __half* __restrict__ A, __half* __restrict__ Bop,
        float* __restrict__ fA, float* __restrict__ fB, int blk, int ne, int batch, int Mp, long Ks, int W, int S,
        int slot0, int node_cbs, int gate_cbs, int gate_stride, long ext_base, bool gated) {
    const int nb = blockIdx.x, lane = threadIdx.x % 32, r0 = (blockIdx.y * 8 + threadIdx.x / 32) * R, c = blockIdx.z;
    const int vpl = W / 32, cb = c * W, slot = slot0 + c;
    float M[VPL];
#pragma unroll
    for (int j = 0; j < VPL; j++) {
        M[j] = 0.f;
        if (j < vpl) {
            const unsigned mo = Mord[(long)nb * Mp + cb + lane + 32 * j];
            M[j] = (mo <= f2ord(-INFINITY)) ? 0.f : ord2f(mo);        // (0 where every node is -inf)
        }
    }
    float v[R][VPL];
    if (r0 < blk) {
        const long base = nids[nb] + r0;
        float nf[R][VPL], nm[R][VPL];
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) {
                const int b = cb + lane + 32 * j;
                nm[u][j] = -INFINITY; nf[u][j] = 0.f;
                if (j < vpl && b < batch) {
                    const long off = (base + u) * (long)batch + b;
                    nm[u][j] = nmars[off]; nf[u][j] = nflows[off];
                }
            }
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) v[u][j] = nm[u][j] == -INFINITY ? 0.f : __expf(nf[u][j] - nm[u][j] - M[j]);
    } else {
        long cid[R], grow[R];                                            // (grow: -2 = no gate table, log phi 0)
#pragma unroll
        for (int u = 0; u < R; u++) {
            const int e = r0 - blk + u, gcol = e / node_cbs;
            cid[u] = cids[(long)nb * ne + e];
            if (gated) {
                const long gb = gcol < gate_stride ? gate[(long)nb * gate_stride + gcol] : -1;
                grow[u] = gb < 0 ? -1 : gb + ext_base + (e % node_cbs) / gate_cbs;
            } else {
                grow[u] = -2;
            }
        }
        float em[R][VPL], lp[R][VPL];
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) {
                const int b = cb + lane + 32 * j;
                em[u][j] = -INFINITY; lp[u][j] = 0.f;
                if (j < vpl && b < batch && grow[u] != -1) {
                    em[u][j] = emars[cid[u] * (long)batch + b];
                    if (grow[u] >= 0) lp[u][j] = ext[grow[u] * (long)batch + b];
                }
            }
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) {
                const float x = em[u][j] + lp[u][j] + M[j];
                v[u][j] = x == -INFINITY ? 0.f : __expf(x);
            }
    }
    int xr[R];
#pragma unroll
    for (int u = 0; u < R; u++) {
        float m = 0.f;
#pragma unroll
        for (int j = 0; j < VPL; j++) m = fmaxf(m, v[u][j]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        // m = f 2^x, f in [0.5, 1) (m is 0 or normal: the exponentials flush to zero)
        xr[u] = m > 0.f ? (int)((__float_as_uint(m) >> 23) & 0xffu) - 126 : 0;
    }
#pragma unroll
    for (int u = 0; u < R; u++) {
        const int r = r0 + u;
        __half* out = (r < blk ? A + ((long)nb * blk + r) * Ks : Bop + ((long)nb * ne + (r - blk)) * Ks) + (long)slot * W;
        // v 2^(15 - x), 15 - x in [-114, 140]: by two normal powers of two, exact (the product at most 2^15)
        const int e = 15 - xr[u], e1 = min(e, 127);
        const float f1 = pow2i(e1), f2 = pow2i(e - e1);
#pragma unroll
        for (int j = 0; j < VPL; j++)
            if (j < vpl) out[lane + 32 * j] = __float2half_rn(v[u][j] * f1 * f2);
        if (lane == 0) {
            const float s = ldexpf(1.f, xr[u]);
            if (r < blk) fA[((long)nb * S + slot) * blk + r] = s;
            else fB[((long)nb * S + slot) * ne + (r - blk)] = s;
        }
    }
}

// ---------------- flush: the contraction, the per-slot fold and the flows' add ----------------
__device__ __forceinline__ void cpa16(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

extern __shared__ char smem_raw[];

// K-major fp16 tiles (8 rows x BK, swizzled: ldmatrix conflict-free)
template <int BK> struct AtomK;
template <> struct AtomK<64> { using L = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{})); };
template <> struct AtomK<32> { using L = decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{})); };

template <int BM, int BN, int BK, int WM, int WN, int ST, int EC, int MINB, bool BULK, typename TP>
__global__ void __launch_bounds__(WM * WN * 32, MINB) flush_kernel(
        float* __restrict__ pf, const __half* __restrict__ A, const __half* __restrict__ Bop,
        const float* __restrict__ fA, const float* __restrict__ fB, const long* __restrict__ pfids,
        const long* __restrict__ pids, const TP* __restrict__ params, int blk, int ne, long Ks, int W, int S,
        int slots) {
    constexpr int NT = WM * WN * 32;
    const int e0 = blockIdx.x * BN, n0 = blockIdx.y * BM, nb = blockIdx.z, tid = threadIdx.x;
    const __half* gA = A + ((long)nb * blk + n0) * Ks;
    const __half* gB = Bop + ((long)nb * ne + e0) * Ks;
    const long* pfb = pfids + (long)nb * ne + e0;
    const long* pib = pids + (long)nb * ne + e0;

    auto sAl = tile_to_shape(typename AtomK<BK>::L{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(typename AtomK<BK>::L{}, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int SA = BM * BK, SB = BN * BK, STAGE = (SA + SB) * 2;
    auto stA = [&](int s) { return (half_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (half_t*)(smem_raw + s * STAGE + SA * 2); };
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);

    // a thread's 16-byte copies of a stage: shared offsets (swizzled) and global offsets, fixed for the whole loop
    constexpr int CPR = BK / 8, CA = (BM * CPR) / NT, CB = (BN * CPR) / NT;
    static_assert((BM * CPR) % NT == 0 && (BN * CPR) % NT == 0, "a stage's copies split over the threads");
    uint32_t oA[CA], oB[CB];
    long gAo[CA], gBo[CB];
    {
        Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
        CUTE_UNROLL
        for (int q = 0; q < CA; q++) {
            int i = tid + q * NT, r = i / CPR, c = (i % CPR) * 8;
            oA[q] = (uint32_t)((char*)&sAt(r, c) - smem_raw);
            gAo[q] = r * Ks + c;
        }
        CUTE_UNROLL
        for (int q = 0; q < CB; q++) {
            int i = tid + q * NT, r = i / CPR, c = (i % CPR) * 8;
            oB[q] = (uint32_t)((char*)&sBt(r, c) - smem_raw);
            gBo[q] = r * Ks + c;
        }
    }
    const int tps = W / BK, nk = slots * tps;
    auto load = [&](int kt) {
        if (kt < nk) {
            const int s = kt % ST, k0 = kt * BK;
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) cpa16(sbase + s * STAGE + oA[q], gA + gAo[q] + k0);
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) cpa16(sbase + s * STAGE + oB[q], gB + gBo[q] + k0);
        }
        cpa_commit();
    };

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, half_t>{}, mma);
    auto tA = s2rA.get_thread_slice(tid);
    auto tB = s2rB.get_thread_slice(tid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor acc = thr.partition_fragment_C(cC);
    Tensor tot = thr.partition_fragment_C(cC);
    clear(acc);
    clear(tot);
    constexpr int KB = size<2>(tCrA), MM = size<1>(acc), MN = size<2>(acc);
    static_assert(size<0>(acc) == 4, "the m16n8 accumulator: (2 columns, 2 row halves)");

    CUTE_UNROLL
    for (int s = 0; s < ST - 1; s++) load(s);
    if constexpr (BULK) {
        // theta's runs into L2 while the main loop runs (the bulk adds read no flows here): per edge a line every 128
        // bytes of its BM parameters, and their last byte
        constexpr int PL = (BM * (int)sizeof(TP) + 127) / 128 + 1;
        for (int i = tid; i < BN * PL; i += NT) {
            const int e = i / PL, j = i % PL;
            const char* p = (const char*)(params + pib[e] + n0) + min(j * 128, BM * (int)sizeof(TP) - 1);
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
        }
    }

    float rs[2][MM], cs[2][MN];                           // the slot's scales of a thread's rows and columns
    auto ktile = [&](int kt) {
        cpa_wait<ST - 2>();
        __syncthreads();
        load(kt + ST - 1);                                // (into the stage read the step before: past the barrier)
        const int s = kt % ST;
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        auto tXsA = tA.partition_S(sAt);
        auto tXsB = tB.partition_S(sBt);
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {
            copy(s2rA, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2rB, tXsB(_, _, kb), tXrB(_, _, kb));
            cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
        }
    };
    int kt = 0;
    for (int slot = 0; slot < slots; slot++) {
        const float* ra = fA + ((long)nb * S + slot) * blk + n0;   // (loaded now, used at the slot's end)
        const float* rb = fB + ((long)nb * S + slot) * ne + e0;
        CUTE_UNROLL
        for (int m = 0; m < MM; m++)
            CUTE_UNROLL
            for (int h = 0; h < 2; h++) rs[h][m] = ra[get<0>(tCcC(make_coord(0, h), m, 0))];
        CUTE_UNROLL
        for (int n = 0; n < MN; n++)
            CUTE_UNROLL
            for (int c = 0; c < 2; c++) cs[c][n] = rb[get<1>(tCcC(make_coord(c, 0), 0, n))];
        for (int t = 0; t < tps - 1; t++) ktile(kt++);
        ktile(kt++);                                      // (peeled: the fold in the last k-tile's basic block)
        CUTE_UNROLL
        for (int n = 0; n < MN; n++)
            CUTE_UNROLL
            for (int m = 0; m < MM; m++)
                CUTE_UNROLL
                for (int h = 0; h < 2; h++)
                    CUTE_UNROLL
                    for (int c = 0; c < 2; c++) {
                        float& a = acc(make_coord(c, h), m, n);
                        float& t = tot(make_coord(c, h), m, n);
                        t = fmaf(a * 0x1p-30f, rs[h][m] * cs[c][n], t);
                        a = 0.f;
                    }
    }
    cpa_wait<0>();
    __syncthreads();

    // EC edges at a time: tot to shared memory as [e][n] (rows padded by 4: the fragments' stores conflict-free), then
    // the flows' adds -- BULK: theta * tot in place and one bulk reduce-add per edge's run (two chunk buffers where the
    // stages' memory holds them, else a chunk waits for the last one's bulk reads); else a thread's flows and
    // parameters loaded 16 at a time, then the adds and stores, consecutive threads on consecutive n
    constexpr int LS = BM + 4;
    constexpr int NBUF = (BULK && 2 * EC * LS * 4 <= ST * STAGE) ? 2 : 1;
    static_assert(EC * LS * 4 <= ST * STAGE && BN % EC == 0 && (BM * EC) % NT == 0, "the epilogue's staging");
    constexpr int PER = BM * EC / NT, SUB = PER > 16 ? 16 : PER;
    static_assert(PER % SUB == 0, "the epilogue's batches");
    float* sC = (float*)smem_raw;
    CUTE_UNROLL
    for (int ec = 0; ec < BN; ec += EC) {
        float* sCc = sC + ((ec / EC) % NBUF) * EC * LS;
        if constexpr (BULK) {
            if (ec / EC >= NBUF) {                        // the buffer's earlier chunk read by its bulk copies
                if (tid < EC) asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(NBUF - 1) : "memory");
                __syncthreads();
            }
        } else if (ec > 0) {
            __syncthreads();
        }
        CUTE_UNROLL
        for (int i = 0; i < size(tot); i++) {
            const int e = get<1>(tCcC(i)) - ec;
            if (e >= 0 && e < EC) sCc[e * LS + get<0>(tCcC(i))] = tot(i);
        }
        __syncthreads();
        CUTE_UNROLL
        for (int u0 = 0; u0 < PER; u0 += SUB) {
            float tv[SUB], pv[SUB];
            CUTE_UNROLL
            for (int u = 0; u < SUB; u++) {
                const int idx = tid + (u0 + u) * NT, e = idx / BM, n = idx % BM;
                tv[u] = ParamLoad<TP>::at(params + pib[ec + e] + n0 + n);
                if constexpr (!BULK) pv[u] = pf[pfb[ec + e] + n0 + n];
            }
            CUTE_UNROLL
            for (int u = 0; u < SUB; u++) {
                const int idx = tid + (u0 + u) * NT, e = idx / BM, n = idx % BM;
                if constexpr (BULK) sCc[e * LS + n] *= tv[u];
                else pf[pfb[ec + e] + n0 + n] = pv[u] + tv[u] * sCc[e * LS + n];
            }
        }
        if constexpr (BULK) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // the writes, seen by the bulk copies
            __syncthreads();
            if (tid < EC) {
                const uint32_t src = (uint32_t)__cvta_generic_to_shared(sCc + tid * LS);
                float* dst = pf + pfb[ec + tid] + n0;
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                             ::"l"(dst), "r"(src), "n"(BM * 4) : "memory");
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
        }
    }
    if constexpr (BULK) {
        if (tid < EC) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // (before the CTA's exit)
    }
}

// id -> {BM, BN, BK, WM, WN, ST, EC, min CTAs an SM}: 128 x 128 (8 and 16 warps), and narrower tiles for node blocks
// or edge counts not a multiple of 128 (also more CTAs for small grids)
#define FLUSH_CFGS(X)                                                                  \
    X(0, 128, 128, 64, 2, 4, 3, 64, 1) X(1, 128, 128, 64, 4, 4, 3, 64, 1)             \
    X(2, 128, 64, 32, 2, 2, 3, 64, 2) X(3, 64, 128, 32, 2, 2, 3, 64, 2)               \
    X(4, 64, 64, 32, 2, 2, 3, 64, 2)

template <int BM, int BN, int BK, int WM, int WN, int ST, int EC, int MINB, bool BULK, typename TP>
static void launch_flush(float* pf, const __half* A, const __half* Bop, const float* fA, const float* fB,
                         const long* pfids, const long* pids, const TP* params, int nbk, int blk, int ne, long Ks,
                         int W, int S, int slots, cudaStream_t st) {
    constexpr int smem = ST * (BM + BN) * BK * 2;
    static_assert(smem <= 101376, "shared memory");
    auto k = flush_kernel<BM, BN, BK, WM, WN, ST, EC, MINB, BULK, TP>;
    static bool set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) set[dev] = true;
    }
    TORCH_CHECK(blk % BM == 0 && ne % BN == 0 && W % BK == 0, "the flush's tile");
    dim3 grid(ne / BN, blk / BM, nbk);
    k<<<grid, WM * WN * 32, smem, st>>>(pf, A, Bop, fA, fB, pfids, pids, params, blk, ne, Ks, W, S, slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<std::vector<int64_t>> flush_configs() {
    std::vector<std::vector<int64_t>> out;
#define X(id, bm, bn, bk, wm, wn, st, ec, minb) out.push_back({bm, bn, bk});
    FLUSH_CFGS(X)
#undef X
    return out;
}

// One step's operands into slots slot0 .. slot0 + cps - 1 (cps = M.size(1) / W chunks of the step's columns, W = a
// slot's columns = A.size(2) / S) and their rows' scales; M: int32 [nb, cps W], scratch; `gated`: log phi from `gate`
// and `ext` (else both unread: any tensors)
void capture(torch::Tensor A, torch::Tensor Bop, torch::Tensor fA, torch::Tensor fB, torch::Tensor M,
             torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor ext,
             torch::Tensor nids, torch::Tensor cids, torch::Tensor gate, int64_t batch, int64_t slot0, int64_t node_cbs,
             int64_t gate_cbs, int64_t ext_base, bool gated) {
    const int nbk = (int)A.size(0), blk = (int)A.size(1), ne = (int)Bop.size(1), S = (int)fA.size(1);
    const long Ks = A.size(2);
    const int W = (int)(Ks / S), Mp = (int)M.size(1), cps = Mp / W;
    TORCH_CHECK(A.scalar_type() == torch::kHalf && Bop.scalar_type() == torch::kHalf && M.scalar_type() == torch::kInt32 &&
                Ks % S == 0 && W % 32 == 0 && W <= 512 && Mp % W == 0 && Mp >= batch && slot0 + cps <= S &&
                blk % 64 == 0 && ne % 64 == 0, "capture: the slot layout");
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(M.data_ptr(), 0, M.numel() * 4, st));
    auto* Mord = reinterpret_cast<unsigned*>(M.data_ptr());
    capture_m_kernel<<<dim3(nbk, Mp / 32, blk / 64), 256, 0, st>>>(node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
                                                                   nids.data_ptr<long>(), Mord, blk, (int)batch, Mp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // rows a warp: the most (up to 32 values a lane) that keep two blocks an SM
    int dev = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int vmax = W <= 128 ? 4 : W <= 256 ? 8 : 16;
    int R = 1;
    while (R < 32 / vmax && R < 8 && (long)nbk * ((blk + ne) / (16 * R)) * cps >= 2L * sms) R *= 2;
#define CAP(R_, VPL)                                                                                                   \
    capture_ab_kernel<R_, VPL><<<dim3(nbk, (blk + ne) / (8 * R_), cps), 256, 0, st>>>(                                \
        node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), ext.data_ptr<float>(), \
        nids.data_ptr<long>(), cids.data_ptr<long>(), gate.data_ptr<long>(), Mord,                                     \
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()), reinterpret_cast<__half*>(Bop.data_ptr<at::Half>()),        \
        fA.data_ptr<float>(), fB.data_ptr<float>(), blk, ne, (int)batch, Mp, Ks, W, S, (int)slot0, (int)node_cbs,      \
        (int)gate_cbs, gated ? (int)gate.size(1) : 0, (long)ext_base, gated)
    if (vmax == 4) {
        if (R == 1) CAP(1, 4); else if (R == 2) CAP(2, 4); else if (R == 4) CAP(4, 4); else CAP(8, 4);
    } else if (vmax == 8) {
        if (R == 1) CAP(1, 8); else if (R == 2) CAP(2, 8); else CAP(4, 8);
    } else {
        if (R == 1) CAP(1, 16); else CAP(2, 16);
    }
#undef CAP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// pf[pfids + n] += theta * the contraction over the first `slots` slots; bulk: each edge's run of flows 16-byte aligned
// (pfids % 4 == 0, the caller checks) -- the L2's adds
void flush(torch::Tensor pf, torch::Tensor A, torch::Tensor Bop, torch::Tensor fA, torch::Tensor fB, torch::Tensor pfids,
           torch::Tensor pids, torch::Tensor params, int64_t slots, int64_t cfg, bool bulk) {
    const int nbk = (int)A.size(0), blk = (int)A.size(1), ne = (int)Bop.size(1), S = (int)fA.size(1);
    const long Ks = A.size(2);
    const int W = (int)(Ks / S);
    TORCH_CHECK(slots >= 1 && slots <= S && (!bulk || reinterpret_cast<uintptr_t>(pf.data_ptr()) % 16 == 0),
                "flush: the slots, the flows' alignment");
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    const __half* a = reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
    const __half* b = reinterpret_cast<const __half*>(Bop.data_ptr<at::Half>());
    const bool bf = params.scalar_type() == torch::kBFloat16;
#define ARGS(TP) pf.data_ptr<float>(), a, b, fA.data_ptr<float>(), fB.data_ptr<float>(), pfids.data_ptr<long>(),        \
                 pids.data_ptr<long>(), param_ptr<TP>(params), nbk, blk, ne, Ks, W, S, (int)slots, st
#define X(id, bm, bn, bk, wm, wn, st_, ec, minb)                                                                       \
    case id:                                                                                                           \
        if (bf) {                                                                                                      \
            if (bulk) launch_flush<bm, bn, bk, wm, wn, st_, ec, minb, true, __nv_bfloat16>(ARGS(__nv_bfloat16));       \
            else launch_flush<bm, bn, bk, wm, wn, st_, ec, minb, false, __nv_bfloat16>(ARGS(__nv_bfloat16));           \
        } else {                                                                                                       \
            if (bulk) launch_flush<bm, bn, bk, wm, wn, st_, ec, minb, true, float>(ARGS(float));                       \
            else launch_flush<bm, bn, bk, wm, wn, st_, ec, minb, false, float>(ARGS(float));                           \
        }                                                                                                              \
        break;
    switch (cfg) {
        FLUSH_CFGS(X)
        default: TORCH_CHECK(false, "no such flush config");
    }
#undef X
#undef ARGS
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("capture", &capture, "one step's operands into its fp16 slots and their rows' scales (M: int32, scratch; gated: log phi read)");
    m.def("flush", &flush, "pf[pfids + n] += theta * sum over the slots of (fA fB 2^-30) (A16 . Bop16^T)");
    m.def("flush_configs", &flush_configs, "(BM, BN, BK) per flush config");
}
