// The deferred parameter flows' flush (`pyjuice.layer.deferred_par`) with its operand tiles brought in by a producer
// warp -- par_deferred_fp16.cu's `flush` (what it computes, its scope and its numerics: see that file's header), the
// same contraction to the bit, as more configs for the flush's tuner (`_Group._pick_cfg`).
//
// There every thread cp.asyncs its share of a k-tile's two operand tiles, and every k-tile ends at a CTA barrier
// (`__syncthreads`) before its stage is refilled. MEASURED (RTX PRO 6000, HMM-4096 in 4 blocks, bf16, the 128 x 128 x 64
// config, timing only): that barrier alone is 11% of the flush at B 128 (554 -> 491 us without it), 13% at B 512 (1458 ->
// 1267); the loads with it 16-18%. Here one producer thread loads each stage with two TMA copies (cp.async.bulk.tensor,
// the operands' 3-D maps [nb, rows, Ks], 128-byte swizzle: the smem layout the consumers' ldmatrix reads -- the K-major
// Swizzle<3,3,3> atom, bit for bit) into an mbarrier'd ring: the consumers wait for a stage's bytes (`full`) and release
// it after their last ldmatrix of it (`empty`); no barrier spans the CTA in the k loop. The consumers -- their MMAs, the
// per-slot fold and the epilogue -- are the flush kernel's code: each flow's sum over the same k16 steps in the same order,
// the same fold, the same add. Scope (the caller checks it): par_deferred_fp16.cu's, with blk % BM == 0 and ne % BN == 0.

#include <torch/extension.h>
#include "param_load.cuh"                                   // `param_ptr<TP>`, `ParamLoad<TP>`: the stored dtype
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <type_traits>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, int bytes) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{\n .reg .pred p;\n W_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@!p bra W_%=;\n}\n" ::"r"(a), "r"(phase) : "memory");
}
// a box of the 3-D map at (k, row, nb) into `smem`, its bytes counted on `bar`
__device__ __forceinline__ void tma_load_3d(void* smem, const CUtensorMap* map, int k, int row, int nb, uint64_t* bar) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem), b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 ::"r"(s), "l"(reinterpret_cast<uint64_t>(map)), "r"(k), "r"(row), "r"(nb), "r"(b) : "memory");
}
// the consumers' own barrier (named barrier 1): the producer warp takes no part
template <int N> __device__ __forceinline__ void consumers_sync() { asm volatile("bar.sync 1, %0;" ::"n"(N) : "memory"); }

extern __shared__ __align__(1024) char smem_raw[];

// K-major fp16 tiles (8 rows x 64, swizzled: ldmatrix conflict-free) -- TMA's 128-byte swizzle of a 64-wide box
using AtomK64 = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

template <int BM, int BN, int WM, int WN, int ST, int EC, bool BULK, typename TP>
__global__ void __launch_bounds__((WM * WN + 1) * 32, 1) flush_ws_kernel(
        float* __restrict__ pf, const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
        const float* __restrict__ fA, const float* __restrict__ fB, const long* __restrict__ pfids,
        const long* __restrict__ pids, const TP* __restrict__ params, int blk, int G, int lgb, int ne, int W, int S,
        int slots) {
    constexpr int BK = 64, NT = WM * WN * 32;            // (NT: the consumers, the flush kernel's threads)
    // (the flush kernel's operand groups: program (edge tile, row tile, g), group g's G blk rows the tile's)
    const int e0 = blockIdx.x * BN, n0 = blockIdx.y * BM, g = blockIdx.z, tid = threadIdx.x;
    const long nbt = (long)g * G + n0 / blk;
    const int nt0 = n0 % blk, run = BM <= blk ? BM : blk, nrun = BM / run;
    auto rnb = [&](int r) -> long { return BM <= blk ? nbt : nbt + (r >> lgb); };
    auto rnd = [&](int r) -> int { return BM <= blk ? nt0 + r : r & (blk - 1); };

    auto sAl = tile_to_shape(AtomK64{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(AtomK64{}, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int SA = BM * BK, SB = BN * BK, STAGE = (SA + SB) * 2;
    static_assert(STAGE % 1024 == 0, "the stages' 128-byte swizzle: 1024-byte aligned tiles");
    auto stA = [&](int s) { return (half_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (half_t*)(smem_raw + s * STAGE + SA * 2); };
    uint64_t* full = (uint64_t*)(smem_raw + ST * STAGE);
    uint64_t* empty = full + ST;
    const int tps = W / BK, nk = slots * tps;
    if (tid == 0) {
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], 1); mbar_init(&empty[s], NT); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (tid >= NT) {
        // ---------------- the producer: k-tile kt into stage kt % ST once the consumers released it ----------------
        if (tid == NT) {
            for (int kt = 0; kt < nk; kt++) {
                const int s = kt % ST;
                if (kt >= ST) mbar_wait(&empty[s], ((kt / ST) - 1) & 1);
                mbar_expect(&full[s], STAGE);
                tma_load_3d(stA(s), &mapA, kt * BK, n0, g, &full[s]);
                tma_load_3d(stB(s), &mapB, kt * BK, e0, g, &full[s]);
            }
        }
        return;
    }

    // ---------------- the consumers: the flush kernel's main loop, fold and epilogue ----------------
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

    if constexpr (BULK) {
        // theta's runs into L2 while the main loop runs (the bulk adds read no flows here): per edge a line every 128
        // bytes of its BM parameters, and their last byte
        const int PL = (run * (int)sizeof(TP) + 127) / 128 + 1;
        for (int i = tid; i < BN * nrun * PL; i += NT) {
            const int e = i / (nrun * PL), k = (i / PL) % nrun, j = i % PL;
            const char* p = (const char*)(params + pids[(rnb(k * run)) * ne + e0 + e] + rnd(k * run)) +
                            min(j * 128, run * (int)sizeof(TP) - 1);
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
        }
    }

    float rs[2][MM], cs[2][MN];                           // the slot's scales of a thread's rows and columns
    auto ktile = [&](int kt) {
        const int s = kt % ST;
        mbar_wait(&full[s], (kt / ST) & 1);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        auto tXsA = tA.partition_S(sAt);
        auto tXsB = tB.partition_S(sBt);
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {
            copy(s2rA, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2rB, tXsB(_, _, kb), tXrB(_, _, kb));
            if (kb + 1 == KB) mbar_arrive(&empty[s]);   // (the stage read: the producer may refill it)
            cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
        }
    };
    int kt = 0;
    for (int slot = 0; slot < slots; slot++) {
        const float* rb = fB + ((long)g * S + slot) * ne + e0;    // (loaded now, used at the slot's end)
        CUTE_UNROLL
        for (int m = 0; m < MM; m++)
            CUTE_UNROLL
            for (int h = 0; h < 2; h++) {
                const int r = get<0>(tCcC(make_coord(0, h), m, 0));
                rs[h][m] = fA[(rnb(r) * S + slot) * blk + rnd(r)];
            }
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
    consumers_sync<NT>();                                 // (every consumer past its last ldmatrix: the stages are free)

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
                if (tid < EC * nrun) asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(NBUF - 1) : "memory");
                consumers_sync<NT>();
            }
        } else if (ec > 0) {
            consumers_sync<NT>();
        }
        CUTE_UNROLL
        for (int i = 0; i < size(tot); i++) {
            const int e = get<1>(tCcC(i)) - ec;
            if (e >= 0 && e < EC) sCc[e * LS + get<0>(tCcC(i))] = tot(i);
        }
        consumers_sync<NT>();
        CUTE_UNROLL
        for (int u0 = 0; u0 < PER; u0 += SUB) {
            float tv[SUB], pv[SUB];
            CUTE_UNROLL
            for (int u = 0; u < SUB; u++) {
                const int idx = tid + (u0 + u) * NT, e = idx / BM, n = idx % BM;
                const long q = rnb(n) * ne + e0 + ec + e;
                tv[u] = ParamLoad<TP>::at(params + pids[q] + rnd(n));
                if constexpr (!BULK) pv[u] = pf[pfids[q] + rnd(n)];
            }
            CUTE_UNROLL
            for (int u = 0; u < SUB; u++) {
                const int idx = tid + (u0 + u) * NT, e = idx / BM, n = idx % BM;
                if constexpr (BULK) sCc[e * LS + n] *= tv[u];
                else pf[pfids[rnb(n) * ne + e0 + ec + e] + rnd(n)] = pv[u] + tv[u] * sCc[e * LS + n];
            }
        }
        if constexpr (BULK) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // the writes, seen by the bulk copies
            consumers_sync<NT>();
            if (tid < EC * nrun) {
                for (int i = tid; i < EC * nrun; i += NT) {   // (an edge's runs: one a node block in the tile)
                    const int e = i / nrun, k = i % nrun;
                    const uint32_t src = (uint32_t)__cvta_generic_to_shared(sCc + e * LS + k * run);
                    float* dst = pf + pfids[rnb(k * run) * ne + e0 + ec + e] + rnd(k * run);
                    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                                 ::"l"(dst), "r"(src), "r"(run * 4) : "memory");
                }
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
        }
    }
    if constexpr (BULK) {
        if (tid < EC * nrun) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // (before the exit)
    }
}

// id -> {BM, BN, WM, WN, ST, EC}: 128 x 128 tiles, 64-wide k-tiles in 3 stages (96 KB), 8 consumer warps as
// par_deferred_fp16.cu's config 0 (64 x 32 a warp) or in 4 x 2 (32 x 64); and 32 x 128 for node blocks of 32
#define FLUSH_WS_CFGS(X) X(0, 128, 128, 2, 4, 3, 64) X(1, 128, 128, 4, 2, 3, 64) X(2, 32, 128, 2, 4, 3, 64)

// The operand's 3-D map: [nb][rows][Ks] fp16, boxes of 64 k x `box_rows` rows, 128-byte swizzle
static CUtensorMap operand_map(const torch::Tensor& t, int box_rows) {
    CUtensorMap map;
    const cuuint64_t gdim[3] = {(cuuint64_t)t.size(2), (cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
    const cuuint64_t gstride[2] = {(cuuint64_t)t.size(2) * 2, (cuuint64_t)t.size(1) * t.size(2) * 2};
    const cuuint32_t box[3] = {64, (cuuint32_t)box_rows, 1};
    const cuuint32_t estride[3] = {1, 1, 1};
    const CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, t.data_ptr(), gdim, gstride, box,
                                              estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                              CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "flush ws: cuTensorMapEncodeTiled failed: ", (int)r);
    return map;
}

template <int BM, int BN, int WM, int WN, int ST, int EC, bool BULK, typename TP>
static void launch_flush_ws(float* pf, const torch::Tensor& A, const torch::Tensor& Bop, const float* fA,
                            const float* fB, const long* pfids, const long* pids, const TP* params, int nbk, int ng,
                            int blk, int ne, int W, int S, int slots, cudaStream_t st) {
    constexpr int smem = ST * (BM + BN) * 64 * 2 + 2 * ST * 8;
    static_assert(smem <= 101376, "shared memory");
    auto k = flush_ws_kernel<BM, BN, WM, WN, ST, EC, BULK, TP>;
    static bool set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) set[dev] = true;
    }
    const int G = nbk / ng, lgb = 31 - __builtin_clz((unsigned)blk);
    TORCH_CHECK(nbk % ng == 0 && ((long)G * blk) % BM == 0 && ne % BN == 0 && W % 64 == 0 &&
                (BM <= blk ? blk % BM == 0 : (blk & (blk - 1)) == 0), "the flush's tile");
    // (A's rows by groups: [ng][G blk][Ks], the same memory)
    const CUtensorMap mA = operand_map(A.view({ng, (long)G * blk, A.size(2)}), BM), mB = operand_map(Bop, BN);
    dim3 grid(ne / BN, G * blk / BM, ng);
    k<<<grid, (WM * WN + 1) * 32, smem, st>>>(pf, mA, mB, fA, fB, pfids, pids, params, blk, G, lgb, ne, W, S, slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<std::vector<int64_t>> flush_configs() {
    std::vector<std::vector<int64_t>> out;
#define X(id, bm, bn, wm, wn, st, ec) out.push_back({bm, bn, 64});
    FLUSH_WS_CFGS(X)
#undef X
    return out;
}

// par_deferred_fp16.cu's `flush` (the same arguments and contract) at config `cfg` of this module
void flush(torch::Tensor pf, torch::Tensor A, torch::Tensor Bop, torch::Tensor fA, torch::Tensor fB, torch::Tensor pfids,
           torch::Tensor pids, torch::Tensor params, int64_t slots, int64_t cfg, bool bulk) {
    const int nbk = (int)A.size(0), blk = (int)A.size(1), ng = (int)Bop.size(0), ne = (int)Bop.size(1),
              S = (int)fA.size(1);
    const long Ks = A.size(2);
    const int W = (int)(Ks / S);
    TORCH_CHECK(A.scalar_type() == torch::kHalf && Bop.scalar_type() == torch::kHalf && A.is_contiguous() &&
                Bop.is_contiguous() && Bop.size(2) == Ks && Ks % S == 0, "flush ws: the operands' slots");
    TORCH_CHECK(ng >= 1 && nbk % ng == 0 && fB.size(0) == ng && fA.size(0) == nbk,
                "flush ws: the operand groups (Bop, fB: one a group of node blocks)");
    TORCH_CHECK(slots >= 1 && slots <= S && (!bulk || reinterpret_cast<uintptr_t>(pf.data_ptr()) % 16 == 0),
                "flush: the slots, the flows' alignment");
    const at::cuda::OptionalCUDAGuard guard(A.device());
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    const bool bf = params.scalar_type() == torch::kBFloat16;
#define ARGS(TP) pf.data_ptr<float>(), A, Bop, fA.data_ptr<float>(), fB.data_ptr<float>(), pfids.data_ptr<long>(),     \
                 pids.data_ptr<long>(), param_ptr<TP>(params), nbk, ng, blk, ne, W, S, (int)slots, st
#define X(id, bm, bn, wm, wn, st_, ec)                                                                                 \
    case id:                                                                                                           \
        if (bf) {                                                                                                      \
            if (bulk) launch_flush_ws<bm, bn, wm, wn, st_, ec, true, __nv_bfloat16>(ARGS(__nv_bfloat16));              \
            else launch_flush_ws<bm, bn, wm, wn, st_, ec, false, __nv_bfloat16>(ARGS(__nv_bfloat16));                  \
        } else {                                                                                                       \
            if (bulk) launch_flush_ws<bm, bn, wm, wn, st_, ec, true, float>(ARGS(float));                              \
            else launch_flush_ws<bm, bn, wm, wn, st_, ec, false, float>(ARGS(float));                                  \
        }                                                                                                              \
        break;
    switch (cfg) {
        FLUSH_WS_CFGS(X)
        default: TORCH_CHECK(false, "no such flush ws config");
    }
#undef X
#undef ARGS
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flush", &flush, "par_deferred_fp16.cu's flush, its tiles loaded by a producer warp (TMA)");
    m.def("flush_configs", &flush_configs, "(BM, BN, BK) per config");
}
