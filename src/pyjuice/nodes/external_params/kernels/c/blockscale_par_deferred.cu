// A tied group's `BlockScaleSumParams` parameter flows, deferred to one contraction. Every tied copy of a gated sum
// node adds `theta * sum_b exp(lf[n, b] + emars[c(e), b] + log phi[g(e), b])` to the SAME flow block; one step at a
// time, each a read-add-write of the whole block (and of the parameters' read), that block's traffic is paid once per
// step. Deferred, each step only captures its two operands into the group's slots, and one contraction over
// (step, sample) adds the group's flows once:
//   capture   per step (`capture`): the column maxima M[nb, b] = max_n lf over the node block (lf = node_flows -
//             node_mars, -inf where node_mars is; 0 where every node is -inf) -- as order-preserving integers, a wide
//             grid of partial maxima combined by atomicMax into a zeroed buffer -- then both operands' rows,
//             A[nb, n, k] = exp(lf - M) and Bop[nb, e, k] = exp(emars[cid] + log phi[gate row] + M) (0 without a gate),
//             k = slot * Bp + b, TF32-rounded (nearest) fp32, 0 in the columns past the batch;
//   flush     (`flush`): per (BN edges, BM nodes, node block) C = A . Bop^T over the captured k (cp.async stages,
//             ldmatrix, mma.sync m16n8k8 TF32), then pf[pfids[nb, e] + n] += theta[pids[nb, e] + n] * C[n, e] --
//             the C tile through shared memory, so the read-add-write streams along a run of one edge's flows.
// The one stabilizer per (node block, column) -- not the per-step kernels' per node tile -- loses nothing measured:
// against float64 the flows' max error 1e-4 .. 2.5e-4 (the per-step Triton kernel's 6.6e-4 .. 9.4e-4, TF32 truncated).
//
// MEASURED (RTX PRO 6000, a tied gated HMM backward, gate width 64, B 128 bf16, against the per-step Triton kernels):
// HMM-4096 (4 blocks of 1024) 7 steps 769 -> 364 us (capture 12.6 us a step, flush 276), HMM-1024 31 steps
// 345 -> 219, 7 steps fp32 B 100 74 -> 56; at 262K flows (HMM-512 in 2 blocks, 15 steps) 67 -> 81. The Python side
// defers from 8M flows a group (`deferred_par.MIN_FLOWS`: a host-bound backward exposes the flush). The flush is bound
// by its fp32 operands' L2 traffic (its MMAs ablated: -5%).
//
// Scope (the Python side checks it): blk % 64 == 0, num_edges % 64 == 0, the slot layout's Bp % 32 == 0.

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"   // `param_ptr<TP>`: the stored dtype
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

__device__ __forceinline__ float tf32_rna(float x) {
    return __uint_as_float((__float_as_uint(x) + 0x1000u) & 0xffffe000u);
}

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
                                                        int blk, int batch, int Bp) {
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
        if (m > -INFINITY) atomicMax(Mord + (long)nb * Bp + b, f2ord(m));
    }
}

// Both operands' rows: a block per (node block, 64 rows, 32 columns) -- rows below blk are A's (nodes), the rest
// Bop's (edges); 8 warps x 8 rows, lane = column, each row's index loads first, then its values (two round trips)
__global__ void __launch_bounds__(256) capture_ab_kernel(const float* __restrict__ nflows, const float* __restrict__ nmars,
                                                         const float* __restrict__ emars, const float* __restrict__ ext,
                                                         const long* __restrict__ nids, const long* __restrict__ cids,
                                                         const long* __restrict__ gate, const unsigned* __restrict__ Mord,
                                                         float* __restrict__ A, float* __restrict__ Bop, int blk, int ne,
                                                         int batch, int Bp, long Ks, long koff, int node_cbs,
                                                         int gate_cbs, int gate_stride, long ext_base) {
    const int nb = blockIdx.x, r0 = blockIdx.y * 64 + (threadIdx.x / 32) * 8, b = blockIdx.z * 32 + threadIdx.x % 32;
    const bool live = b < batch;
    const unsigned mo = Mord[(long)nb * Bp + b];
    const float M = (mo <= f2ord(-INFINITY)) ? 0.f : ord2f(mo);          // (0 where every node is -inf)
    if (r0 < blk) {                                                       // (blk % 64 == 0: a block is all A or all Bop)
        const long base = nids[nb];
        float nf[8], nm[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            nm[u] = -INFINITY; nf[u] = 0.f;
            if (live) {
                const long off = (base + r0 + u) * (long)batch + b;
                nm[u] = nmars[off]; nf[u] = nflows[off];
            }
        }
        float* Ab = A + ((long)nb * blk + r0) * Ks + koff + b;
#pragma unroll
        for (int u = 0; u < 8; u++) Ab[u * Ks] = nm[u] == -INFINITY ? 0.f : tf32_rna(__expf(nf[u] - nm[u] - M));
    } else {
        const int e0 = r0 - blk;
        long cid[8], gb[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int e = e0 + j, gcol = e / node_cbs;
            cid[j] = cids[(long)nb * ne + e];
            gb[j] = gcol < gate_stride ? gate[(long)nb * gate_stride + gcol] : -1;
        }
        float em[8], lp[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int e = e0 + j;
            em[j] = -INFINITY; lp[j] = 0.f;
            if (live && gb[j] >= 0) {
                em[j] = emars[cid[j] * (long)batch + b];
                lp[j] = ext[(gb[j] + ext_base + (e % node_cbs) / gate_cbs) * batch + b];
            }
        }
        float* Bb = Bop + ((long)nb * ne + e0) * Ks + koff + b;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float x = em[j] + lp[j] + M;
            Bb[j * Ks] = x == -INFINITY ? 0.f : tf32_rna(__expf(x));
        }
    }
}

// ---------------- flush: the contraction and the flows' read-add-write ----------------
__device__ __forceinline__ void cpa16(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

extern __shared__ char smem_raw[];

template <int BM, int BN, int BK, int WM, int WN, int ST, int EC, int MINB, typename TP>
__global__ void __launch_bounds__(WM * WN * 32, MINB) flush_kernel(
        float* __restrict__ pf, const float* __restrict__ A, const float* __restrict__ Bop,
        const long* __restrict__ pfids, const long* __restrict__ pids, const TP* __restrict__ params,
        int blk, int ne, long Ks, int Ku) {
    constexpr int NT = WM * WN * 32;
    const int e0 = blockIdx.x * BN, n0 = blockIdx.y * BM, nb = blockIdx.z, tid = threadIdx.x;
    const float* gA = A + ((long)nb * blk + n0) * Ks;
    const float* gB = Bop + ((long)nb * ne + e0) * Ks;

    auto sAl = tile_to_shape(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, Int<BK>>, Stride<Int<BK>, _1>>{}),
                             make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, Int<BK>>, Stride<Int<BK>, _1>>{}),
                             make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int SA = BM * BK, SB = BN * BK, STAGE = (SA + SB) * 4;
    auto stA = [&](int s) { return (tfloat32_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (tfloat32_t*)(smem_raw + s * STAGE + SA * 4); };
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);

    constexpr int CPR = BK / 4, CA = (BM * CPR) / NT, CB = (BN * CPR) / NT;
    uint32_t oA[CA], oB[CB];
    long gAo[CA], gBo[CB];
    {
        Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
        CUTE_UNROLL
        for (int q = 0; q < CA; q++) {
            int i = tid + q * NT, r = i / CPR, c = (i % CPR) * 4;
            oA[q] = (uint32_t)((char*)&sAt(r, c) - smem_raw);
            gAo[q] = r * Ks + c;
        }
        CUTE_UNROLL
        for (int q = 0; q < CB; q++) {
            int i = tid + q * NT, r = i / CPR, c = (i % CPR) * 4;
            oB[q] = (uint32_t)((char*)&sBt(r, c) - smem_raw);
            gBo[q] = r * Ks + c;
        }
    }
    const int nk = Ku / BK;
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

    TiledMMA mma = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, tfloat32_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, tfloat32_t>{}, mma);
    auto tA = s2rA.get_thread_slice(tid);
    auto tB = s2rB.get_thread_slice(tid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor acc = thr.partition_fragment_C(cC);
    clear(acc);
    constexpr int KB = size<2>(tCrA);

    CUTE_UNROLL
    for (int s = 0; s < ST - 1; s++) load(s);
    for (int kt = 0; kt < nk; kt++) {
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
    }
    cpa_wait<0>();
    __syncthreads();

    // EC edges at a time: C to shared memory as [e][n] (rows padded by 4: the fragments' stores conflict-free), then the
    // flows' read-add-write, consecutive threads on consecutive n (a run of one edge's flows)
    constexpr int LS = BM + 4;
    float* sC = (float*)smem_raw;
    const long* pfb = pfids + (long)nb * ne + e0;
    const long* pib = pids + (long)nb * ne + e0;
    CUTE_UNROLL
    for (int ec = 0; ec < BN; ec += EC) {
        if (ec > 0) __syncthreads();
        CUTE_UNROLL
        for (int i = 0; i < size(acc); i++) {
            const int e = get<1>(tCcC(i)) - ec;
            if (e >= 0 && e < EC) sC[e * LS + get<0>(tCcC(i))] = acc(i);
        }
        __syncthreads();
        for (int idx = tid; idx < BM * EC; idx += NT) {
            const int e = idx / BM, n = idx % BM;
            const long f = pfb[ec + e] + n0 + n, p = pib[ec + e] + n0 + n;
            float th;
            if constexpr (std::is_same<TP, __nv_bfloat16>::value) th = __bfloat162float(params[p]);
            else th = params[p];
            pf[f] += th * sC[e * LS + n];
        }
    }
}

// id -> {BM, BN, BK, WM, WN, ST, EC, min CTAs an SM}
#define FLUSH_CFGS(X)                                                                  \
    X(0, 128, 128, 16, 2, 4, 3, 64, 2) X(1, 128, 64, 32, 4, 2, 4, 64, 1)             \
    X(2, 64, 128, 16, 2, 4, 4, 128, 2) X(3, 64, 64, 32, 2, 2, 3, 64, 3)

template <int BM, int BN, int BK, int WM, int WN, int ST, int EC, int MINB, typename TP>
static void launch_flush(float* pf, const float* A, const float* Bop, const long* pfids, const long* pids,
                         const TP* params, int nbk, int blk, int ne, long Ks, int Ku, cudaStream_t st) {
    constexpr int smem = ST * (BM + BN) * BK * 4;
    static_assert(smem <= 101376 && EC * (BM + 4) * 4 <= smem && BN % EC == 0, "shared memory");
    auto k = flush_kernel<BM, BN, BK, WM, WN, ST, EC, MINB, TP>;
    static bool set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) set[dev] = true;
    }
    TORCH_CHECK(blk % BM == 0 && ne % BN == 0 && Ku % BK == 0, "the flush's tile");
    dim3 grid(ne / BN, blk / BM, nbk);
    k<<<grid, WM * WN * 32, smem, st>>>(pf, A, Bop, pfids, pids, params, blk, ne, Ks, Ku);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<std::vector<int64_t>> flush_configs() {
    std::vector<std::vector<int64_t>> out;
#define X(id, bm, bn, bk, wm, wn, st, ec, minb) out.push_back({bm, bn, bk});
    FLUSH_CFGS(X)
#undef X
    return out;
}

void capture(torch::Tensor A, torch::Tensor Bop, torch::Tensor M, torch::Tensor node_flows, torch::Tensor node_mars,
             torch::Tensor element_mars, torch::Tensor ext, torch::Tensor nids, torch::Tensor cids, torch::Tensor gate,
             int64_t batch, int64_t slot, int64_t node_cbs, int64_t gate_cbs, int64_t ext_base) {
    const int nbk = (int)A.size(0), blk = (int)A.size(1), ne = (int)Bop.size(1);
    const long Ks = A.size(2);
    const int Bp = (int)M.size(1);
    TORCH_CHECK(Bp % 32 == 0 && Bp >= batch && (slot + 1) * Bp <= Ks && ne % 64 == 0 && blk % 64 == 0 &&
                M.scalar_type() == torch::kInt32, "capture: the slot layout");
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(M.data_ptr(), 0, M.numel() * 4, st));
    auto* Mord = reinterpret_cast<unsigned*>(M.data_ptr());
    capture_m_kernel<<<dim3(nbk, Bp / 32, blk / 64), 256, 0, st>>>(node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
                                                                   nids.data_ptr<long>(), Mord, blk, (int)batch, Bp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    capture_ab_kernel<<<dim3(nbk, (blk + ne) / 64, Bp / 32), 256, 0, st>>>(
        node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), ext.data_ptr<float>(),
        nids.data_ptr<long>(), cids.data_ptr<long>(), gate.data_ptr<long>(), Mord, A.data_ptr<float>(),
        Bop.data_ptr<float>(), blk, ne, (int)batch, Bp, Ks, slot * Bp, (int)node_cbs, (int)gate_cbs, (int)gate.size(1),
        (long)ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flush(torch::Tensor pf, torch::Tensor A, torch::Tensor Bop, torch::Tensor pfids, torch::Tensor pids,
           torch::Tensor params, int64_t slots, int64_t Bp, int64_t cfg) {
    const int nbk = (int)A.size(0), blk = (int)A.size(1), ne = (int)Bop.size(1);
    const long Ks = A.size(2);
    const int Ku = (int)(slots * Bp);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
#define X(id, bm, bn, bk, wm, wn, st_, ec, minb)                                                                       \
    case id:                                                                                                           \
        if (params.scalar_type() == torch::kBFloat16)                                                                  \
            launch_flush<bm, bn, bk, wm, wn, st_, ec, minb, __nv_bfloat16>(                                            \
                pf.data_ptr<float>(), A.data_ptr<float>(), Bop.data_ptr<float>(), pfids.data_ptr<long>(),               \
                pids.data_ptr<long>(), param_ptr<__nv_bfloat16>(params), nbk, blk, ne, Ks, Ku, st);                    \
        else                                                                                                           \
            launch_flush<bm, bn, bk, wm, wn, st_, ec, minb, float>(                                                    \
                pf.data_ptr<float>(), A.data_ptr<float>(), Bop.data_ptr<float>(), pfids.data_ptr<long>(),               \
                pids.data_ptr<long>(), param_ptr<float>(params), nbk, blk, ne, Ks, Ku, st);                            \
        break;
    switch (cfg) {
        FLUSH_CFGS(X)
        default: TORCH_CHECK(false, "no such config");
    }
#undef X
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("capture", &capture, "one step's operands into its slot (M: int32 [nb, Bp], scratch)");
    m.def("flush", &flush, "pf[pfids + n] += theta * (A . Bop^T) over the captured slots");
    m.def("flush_configs", &flush_configs, "(BM, BN, BK) per flush config");
}
