// CUDA (CuTe/CUTLASS) block-sparse sum-layer ELEMENT-FLOW backward for bf16 parameters as a pre-pass, one split GEMM
// launch and a merge -- what the element kernels compute (ele_backward_sum.cu and its pipelined and pre-pass copies),
// not to their bits:
//
//     element_flows[m, b] = emars[m, b] + log sum_e params[m, e] exp(lf[e, b]),   lf = node_flows - node_mars
//
// The regular copy of the BlockScale element stream (nodes/external_params/kernels/c/blockscale_ele_backward_stream.cu),
// without gates, gradients and pieces. The other kernels walk an element block's parent tiles serially under a
// stabilizer each, merging tile by tile. Here:
//   pre-pass  per (group, split, PB columns): an element block's parents are cut into `split` contiguous ranges of
//             k-steps; over a range, the log ratios' max M and u = exp(lf - M) in bf16 (0 where M is -inf; the forward
//             stream's operand precision -- a hi + lo pair of halves, two MMAs, MEASURED the GEMM compute-bound on
//             HCLT-1024 and 1.3x slower on HMM-4096), laid out [group, parent position, b]. Element blocks of equal
//             parents (an HMM's blocks; an HCLT's 32 blocks of a variable) form a GROUP and share them;
//   GEMM      per (batch tile, child tile, split): L_p = params . u over the range, bf16 MMAs into an fp32 accumulator
//             (the stored bf16 parameters exact, nothing converted) -- producers cp.async the parameter tile (K-major,
//             a row's parents contiguous in shared memory) and the u tile into mbarrier'd stages from offsets computed
//             once per thread; a child tile may span `bpt` consecutive child blocks of one group (an HCLT's 32-node
//             blocks), a parent tile's parameters `pbt` blocks (parents' blocks of 32);
//   merge     per 4 columns of an element: t_p = log L_p + M_p, element_flows = emars + LSE_p t_p (-inf where every t is).
//
// Scope (`fitting_configs`, and the Python dispatch for the rest): bf16 parameters with 8-aligned rows, k-tiles of 64
// parents as contiguous node rows, the parents' block size 32 or a multiple of 64, batch % 64 == 0, the child block a
// multiple of the tile's rows or (whole tiles of same-group blocks) dividing it, split 1 .. 8. Scratch: one allocation
// per call (inside a CUDA graph capture, from its pool).

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

#define BK 64           // k-tile: parents (one run of node ids)
#define CH 32           // the pre-pass's chunk of parents
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
template <int NT, int PB>
__global__ void __launch_bounds__(NT) eles_prepass_kernel(
        const float* __restrict__ nf, const float* __restrict__ nm, const long* __restrict__ ebase,
        const int* __restrict__ grep, __nv_bfloat16* __restrict__ u, float* __restrict__ mp, int batch, int knt,
        int ksteps, int kt_sh, int split) {
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
        const long o = ((long)g * knt * BK + qq * CH) * batch + b0 + b;
        CUTE_UNROLL
        for (int i = 0; i < CH; i++)
            u[o + (long)i * batch] = __float2bfloat16_rn((m == -INFINITY || w[i] == -INFINITY) ? 0.f : __expf(w[i] - m));
    };
    if (qc < q1) store(qc, v);
    for (int qq = qc + PARTS; qq < q1; qq += PARTS) {
        float w[CH];
        load_chunk(qq, w);
        store(qq, w);
    }
    if (part == 0) mp[((long)g * split + sp) * batch + b0 + b] = m;
}

extern __shared__ __align__(1024) char smem_raw[];

// A K-major (the parameters: [child][parent], parents contiguous) with each row's k contiguous in a 128- / 64-byte
// swizzled line: a cp.async chunk of a row lands in that row's line. B MN-major (u: b contiguous).
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
__global__ void __launch_bounds__((WM * WN + PW) * 32) eles_gemm_kernel(
        const bfloat16_t* __restrict__ params, const long* __restrict__ pbase, const bfloat16_t* __restrict__ u,
        const int* __restrict__ ugrp, float* __restrict__ part, int knt, int pbt, int BSK, int BSM, int batch, int split,
        long rows_total) {
    constexpr int NC = WM * WN * 32, NP = PW * 32;
    constexpr int ABYTES = KT * BM * 2, BBYTES = KT * BN * 2;
    constexpr int STAGE = (ABYTES + BBYTES + 1023) / 1024 * 1024;
    constexpr int SUB = BK / KT;                         // k-steps a k-tile
    const int pid_b = blockIdx.x, pid_m = blockIdx.y, sp = blockIdx.z;
    // the child tile: one of `mtiles` of a child block, or `bpt` consecutive blocks of one group
    const int mtiles = BM >= BSM ? 1 : BSM / BM, bpt = BM > BSM ? BM / BSM : 1;
    const int eb = bpt > 1 ? pid_m * bpt : pid_m / mtiles, tile_id = bpt > 1 ? 0 : pid_m % mtiles;
    const int ksteps = knt * SUB;
    const int k0 = (int)((long)sp * ksteps / split), k1 = (int)((long)(sp + 1) * ksteps / split);
    uint64_t* full = (uint64_t*)(smem_raw + ST * STAGE);
    uint64_t* empty = full + ST;
    const int tid = threadIdx.x;
    auto sAl = tile_to_shape(typename AtomA<KT>::L{}, make_shape(Int<BM>{}, Int<KT>{}));
    auto sBl = tile_to_shape(AtomB{}, make_shape(Int<BN>{}, Int<KT>{}));
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + ABYTES); };
    if (tid == 0)
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], NP); mbar_init(&empty[s], NC); }
    __syncthreads();
    if (tid >= NC) {
        // ---------------- producers: the range's k-steps into stage j % ST ----------------
        const int ptid = tid - NC;
        // a thread's 16-byte chunks are the same every k-step: their shared offsets, and for the parameters the row's
        // block (`prA`: its parameter table), the row within it and the parents within the k-step
        constexpr int TA = ABYTES / 16, TB = BBYTES / 16, CA = (TA + NP - 1) / NP, CB = (TB + NP - 1) / NP;
        uint32_t oA[CA], oB[CB];
        const long* prA[CA];
        int mA[CA], kA[CA], gB[CB];
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                int i = ptid + q * NP, m = i / (KT / 8), kk = (i % (KT / 8)) * 8;
                oA[q] = i < TA ? (uint32_t)((char*)&sAt(m, kk) - smem_raw) : ~0u;
                const int bq = bpt > 1 ? m / BSM : 0;
                prA[q] = pbase + (long)(eb + bq) * knt * pbt;
                mA[q] = (bpt > 1 ? m - bq * BSM : m) + tile_id * BM;
                kA[q] = kk;
            }
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) {
                int i = ptid + q * NP, e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(bb, e) - (char*)stB(0)) : ~0u;   // within the u tile
                gB[q] = e * batch + bb;
            }
        }
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        const long ub = (long)ugrp[eb] * knt * BK * batch + (long)pid_b * BN;
        for (int t = k0; t < k1; t++) {
            const int j = t - k0, s = j % ST;
            if (j >= ST) mbar_wait(&empty[s], ((j / ST) - 1) & 1);
            const uint32_t ss = sbase + s * STAGE;
            const int t64 = t / SUB, sub = t % SUB;
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                if (oA[q] != ~0u) {
                    // the parents sub * KT + kA .. + 7 of tile t64: parameter block (t64, their block among pbt), the
                    // row's slab of BSK parents, the parents within the block. (The table entry is a dependent load a
                    // step; fetching it a step ahead MEASURED flat on HMM-4096 B128: the stage round trip, not this
                    // latency, paces the pipeline)
                    const int pp = sub * KT + kA[q];
                    const int blk = pbt > 1 ? pp / BSK : 0, pin = pbt > 1 ? pp - blk * BSK : pp;
                    cpa16(ss + oA[q], params + prA[q][(long)t64 * pbt + blk] + (long)mA[q] * BSK + pin);
                }
            }
            const bfloat16_t* pb = u + ub + (long)t * KT * batch;
            CUTE_UNROLL
            for (int q = 0; q < CB; q++)
                if (oB[q] != ~0u) cpa16(ss + ABYTES + oB[q], pb + gB[q]);
            cpa_mbar_arrive(&full[s]);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }
    // ---------------- consumers: the range's k-steps into one accumulator ----------------
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
    clear(tCrL);
    constexpr int KB = size<2>(tCrA);
    for (int t = k0; t < k1; t++) {
        const int j = t - k0, s = j % ST;
        mbar_wait(&full[s], (j / ST) & 1);
        auto tXsA = tA.partition_S(make_tensor(make_smem_ptr(stA(s)), sAl));
        auto tXsB = tB.partition_S(make_tensor(make_smem_ptr(stB(s)), sBl));
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
    float* o = part + ((long)sp * rows_total + (long)eb * BSM + (long)tile_id * BM) * batch + pid_b * BN;
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i += 2)              // (i, i + 1): columns b, b + 1 of one row
        *(float2*)&o[(long)get<0>(tCcC(i)) * batch + get<1>(tCcC(i))] = make_float2(tCrL(i), tCrL(i + 1));
}

// ---------------- merge: a thread per 4 columns of an element ----------------
template <int S>
__global__ void __launch_bounds__(256) eles_merge_kernel(
        float* __restrict__ ef, const float* __restrict__ emars, const float* __restrict__ part,
        const float* __restrict__ mp, const long* __restrict__ chids, const int* __restrict__ ugrp, int batch, int BSM,
        long rows_total) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    const int bq = batch / 4;
    if (idx >= rows_total * bq) return;
    const long n = idx / bq;
    const int b = (int)(idx % bq) * 4;
    const int eb = (int)(n / BSM), g = ugrp[eb];
    float4 lp[S], xm[S];
    CUTE_UNROLL
    for (int p = 0; p < S; p++) {
        lp[p] = *(const float4*)&part[((long)p * rows_total + n) * batch + b];
        xm[p] = *(const float4*)&mp[((long)g * S + p) * batch + b];
    }
    const long row = chids[eb] + n % BSM;
    const float4 E = *(const float4*)&emars[row * batch + b];
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
        out[c] = (M == -INFINITY || L <= 0.f) ? -INFINITY : (logf(L) + M + (&E.x)[c]);
    }
    *(float4*)&ef[row * batch + b] = make_float4(out[0], out[1], out[2], out[3]);
}

// ---------------- host ----------------
template <int BM, int BN, int WM, int WN, int PW, int ST, int KT>
static void launch_gemm(const bfloat16_t* params, const long* pbase, const bfloat16_t* u, const int* ugrp, float* part,
                        int knt, int pbt, int neb, int BSM, int BSK, int batch, int split, cudaStream_t st) {
    constexpr int STAGE = (KT * BM * 2 + KT * BN * 2 + 1023) / 1024 * 1024;
    constexpr int SMEM = ST * STAGE + 2 * ST * 8;
    static_assert(SMEM <= 101376, "a config past sm_120's 99 KB of shared memory a block");
    auto kernel = eles_gemm_kernel<BM, BN, WM, WN, PW, ST, KT>;
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        if (dev < 64) attr_set[dev] = true;
    }
    const long rows_total = (long)neb * BSM, mt = rows_total / BM;
    TORCH_CHECK(mt < 65536, "sum element stream: past 65535 child tiles");
    dim3 grid(batch / BN, (unsigned)mt, split);
    kernel<<<grid, (WM * WN + PW) * 32, SMEM, st>>>(params, pbase, u, ugrp, part, knt, pbt, BSK, BSM, batch, split,
                                                    rows_total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// id -> {BM, BN, WM, WN, PW, ST, KT}: the stages fill the 96 KB of shared memory a block (ST * (BM + BN) * KT * 2)
#define CFG_TABLE(X)                     \
    X(0, 128, 128, 4, 2, 4, 3, 64)       \
    X(1, 128, 128, 4, 2, 4, 6, 32)       \
    X(2, 128, 64, 4, 2, 4, 4, 64)        \
    X(3, 64, 128, 2, 4, 4, 4, 64)        \
    X(4, 64, 64, 2, 2, 4, 6, 64)         \
    X(5, 128, 64, 4, 2, 4, 8, 32)

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

// The config ids that serve a partition: its tile divides the child block -- or, `grouped` (every tile's consecutive
// blocks share a group), the block divides the tile -- and the batch; parents' blocks of 32 take 32-row k-steps only
// (a k-step within one parameter block).
std::vector<int64_t> fitting_configs(int64_t cs_block_size, int64_t batch, int64_t block_size, bool grouped) {
    std::vector<int64_t> out;
    if (batch % 64 != 0 || cs_block_size % 32 != 0 || block_size % 32 != 0) return out;
    for (int i = 0; i < NCFG; i++) {
        const bool tile_ok = cs_block_size % CFGS[i].bm == 0 || (grouped && CFGS[i].bm % cs_block_size == 0);
        const bool k_ok = block_size % 64 == 0 || CFGS[i].kt <= block_size;
        if (tile_ok && k_ok && batch % CFGS[i].bn == 0) out.push_back(i);
    }
    return out;
}

// The pre-pass element kernel's tables: `chids` int64 [neb], `ebase` int64 [neb, knt] (a parent tile's first node row),
// `pbase` int64 [neb, knt x pbt] (a tile's parameter blocks: one, or two where the parents' blocks are 32 wide), `ugrp`
// int32 [neb] (each element block's group), `grep` int32 [groups] (a group's first block); `cfg` = config id * 16 +
// split.
void ele_backward_sum_stream(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                             torch::Tensor node_mars, torch::Tensor params, torch::Tensor chids, torch::Tensor ebase,
                             torch::Tensor pbase, torch::Tensor ugrp, torch::Tensor grep, int64_t batch_,
                             int64_t block_size, int64_t cs_block_size, int64_t knt_, int64_t cfg) {
    const int batch = (int)batch_, knt = (int)knt_, neb = (int)chids.size(0), ngrp = (int)grep.size(0);
    const int BSK = (int)block_size, BSM = (int)cs_block_size;
    const int cfg_id = (int)(cfg / 16), split = (int)(cfg % 16);
    TORCH_CHECK(cfg_id >= 0 && cfg_id < NCFG, "sum element stream: no config ", cfg_id);
    const int BMc = CFGS[cfg_id].bm, BNc = CFGS[cfg_id].bn, kt = CFGS[cfg_id].kt;
    const int kt_sh = kt == 64 ? 6 : 5, ksteps = knt * (BK / kt);
    const int pbt = BSK >= BK ? 1 : BK / BSK;
    TORCH_CHECK(params.scalar_type() == torch::kBFloat16, "sum element stream: bf16 parameters");
    TORCH_CHECK(batch % 64 == 0 && batch % BNc == 0 && BSM % 32 == 0 &&
                (BSM % BMc == 0 || (BMc % BSM == 0 && ((long)neb * BSM) % BMc == 0)),
                "sum element stream: batch % 64 and % the tile's columns; the child block a multiple of the tile's rows "
                "or dividing it (whole tiles of blocks)");
    TORCH_CHECK((BSK >= BK ? BSK % BK == 0 : (BK % BSK == 0 && kt <= BSK)) && pbt <= 2,
                "sum element stream: the parents' block size a multiple of 64, or 32 with 32-row k-steps");
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= ksteps, "sum element stream: split 1 .. ", MAX_SPLIT,
                " and at most the k-steps (", ksteps, "); got ", split);
    TORCH_CHECK(chids.is_contiguous() && ebase.is_contiguous() && pbase.is_contiguous() && ugrp.is_contiguous() &&
                grep.is_contiguous() && element_flows.is_contiguous() && element_mars.is_contiguous() &&
                node_flows.is_contiguous() && node_mars.is_contiguous(), "sum element stream: contiguous tensors");
    TORCH_CHECK(ebase.numel() == (long)neb * knt && pbase.numel() == (long)neb * knt * pbt && ugrp.numel() == neb &&
                ugrp.scalar_type() == torch::kInt32 && grep.scalar_type() == torch::kInt32 && ngrp >= 1,
                "sum element stream: ebase [neb, knt], pbase [neb, knt x pbt], ugrp int32 [neb], grep int32");
    const at::cuda::OptionalCUDAGuard guard(element_flows.device());
    const auto* pp = reinterpret_cast<const bfloat16_t*>(param_ptr<__nv_bfloat16>(params));
    const long rows_total = (long)neb * BSM;
    // scratch, ONE allocation: u bf16 [groups, knt * 64, B], M [groups, split, B], partials [split, rows, B]
    const long n_u = (long)ngrp * knt * BK * batch;
    const long o_mp = 2 * n_u, o_part = o_mp + 4L * ngrp * split * batch;
    torch::Tensor scratch = torch::empty({o_part + 4L * split * rows_total * batch},
                                         element_mars.options().dtype(torch::kUInt8));
    char* base = reinterpret_cast<char*>(scratch.data_ptr());
    auto* u = reinterpret_cast<__nv_bfloat16*>(base);
    auto* mp = reinterpret_cast<float*>(base + o_mp);
    auto* part = reinterpret_cast<float*>(base + o_part);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    {
#define P_(NT, PB) eles_prepass_kernel<NT, PB><<<ngrp * split * (batch / PB), NT, 0, st>>>( \
            node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), ebase.data_ptr<long>(), grep.data_ptr<int>(), u, \
            mp, batch, knt, ksteps, kt_sh, split)
        if (batch >= 256) P_(512, 16); else P_(256, 8);
#undef P_
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    switch (cfg_id) {
#define X(id, bm, bn, wm, wn, pw, stg, ktt) \
        case id: launch_gemm<bm, bn, wm, wn, pw, stg, ktt>(pp, pbase.data_ptr<long>(), (const bfloat16_t*)u, \
                     ugrp.data_ptr<int>(), part, knt, pbt, neb, BSM, BSK, batch, split, st); break;
        CFG_TABLE(X)
#undef X
    }
    {
        const long n = rows_total * (batch / 4);
        const unsigned grid = (unsigned)((n + 255) / 256);
        float* ef = element_flows.data_ptr<float>();
        const float* em = element_mars.data_ptr<float>();
        const long* cp = chids.data_ptr<long>();
        const int* ug = ugrp.data_ptr<int>();
        switch (split) {
#define M_(S) case S: eles_merge_kernel<S><<<grid, 256, 0, st>>>(ef, em, part, mp, cp, ug, batch, BSM, rows_total); break;
            M_(1) M_(2) M_(3) M_(4) M_(5) M_(6) M_(7) M_(8)
#undef M_
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ele_backward_sum_stream", &ele_backward_sum_stream,
          "block-sparse sum-layer element-flow backward (bf16 params) as a pre-pass, a split GEMM and a merge");
    m.def("configs", &configs, "(BM, BN, KT) per config id");
    m.def("fitting_configs", &fitting_configs,
          "the config ids that serve a partition (cs_block_size, batch, parents' block_size, grouped tiles)");
}
