// CUDA (CuTe/CUTLASS) block-sparse sum-layer forward "tlmm" kernel, warp-specialized: producer warps stream the
// operands while consumer warps compute. The same computation as tlmm_forward_sum.cu, to the bit.
//
// Why: MEASURED by ablation on HCLT-1024's widest layer at B 128 (143 blocks of 1024 x 1024 edges, tile (128, 64)),
// the plain kernel's phases run back to back -- the params' global fp32 load -> bf16 -> shared write is 0.26 of its
// 0.68 ms, everything else 0.42 -- while the parameters alone are 600 MB read once, a 0.41 ms DRAM floor the kernel
// sits at 60% of. Here PW producer warps load the params (fp32 -> bf16 into a double-buffered A stage) and the element
// tile (into a double-buffered fp32 stage) one edge tile ahead, with every load of a thread's share in flight before
// any convert or store, and the consumer warps run the per-batch max, exp into the bf16 B operand, the 16x8x16
// tensor-core MMA and the online log-add-exp on the other stage; hand-off by named barriers (FULL / EMPTY per stage).
// The consumers' phases are latency-bound and want 16 warps per SM, which the plain kernel gets from two blocks per SM
// (42 KB of shared memory each) and this kernel, at 76 KB for its two stages, must hold itself: 8 x 2 MMA warps
// (MEASURED 4 x 2: 0.82 ms, 8 x 2: 0.58 against the plain kernel's 0.68; 0.49 against 0.53 at B 64). No TMA: nothing
// here needs sm_90.
//
// A variant forming the element tile from the children's node values (as tlmm_forward_sum_fused.cu, by the producers)
// was built and measured here too: bit-identical to that kernel and 1.06 ms -- producer-bound by the K predicated
// gathers per element (K the group's largest child count, 8 for 2.29 real children on HCLT) and at its traffic floor
// no better than the product forward + this kernel (0.76) -- so it was not kept: materializing the elements once is
// what wins at B >= 64 on this hardware.
//
// SPECIALIZED fast path under the plain kernel's conditions (LL propagation, bf16 path, no tempering,
// block_size % BM == 0, batch % BN == 0, TILE_SIZE_K == 64, contiguous edge/param layout); the Python dispatcher
// measures it against the plain kernel's configs per shape (`autotune.choose`).

#include <torch/extension.h>
#include "param_load.cuh"
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64

// Named barriers: all `count` threads of the participating warps take part; `arrive` does not wait.
__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
// ids: 1 + s = stage s FULL (producers arrive, consumers sync); 3 + s = stage s EMPTY (consumers arrive, producers
// sync); 5 = the consumers among themselves; 0 is __syncthreads.

extern __shared__ char smem_raw[];

// PW producer warps beside the WM x WN MMA warps.
template <int BM, int BN, int WM, int WN, int PW, typename TP>
__global__ void __launch_bounds__((WM * WN + PW) * 32) tlmm_pipe_kernel(
        float* __restrict__ node_mars, const float* __restrict__ element_mars, const TP* __restrict__ mp,
        const long* __restrict__ nids, const long* __restrict__ ebase, const long* __restrict__ pbase,
        int batch, int block_size, int knt, int pid_m_offset) {
    constexpr int NC = WM * WN * 32;                 // consumer threads
    constexpr int NP = PW * 32;                      // producer threads
    constexpr int NTH = NC + NP;
    int mtiles = block_size / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int nblock = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_nid = nids[nblock];
    const long* eb = ebase + (long)nblock * knt;
    const long* pb = pbase + (long)nblock * knt;
    int tid = threadIdx.x;

    auto swz = composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<Shape<_8, _8>, _8>, Stride<Stride<_1, _64>, _8>>{});
    auto sAl = tile_to_shape(swz, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swz, make_shape(Int<BN>{}, Int<BK>{}));
    bfloat16_t* pA0 = (bfloat16_t*)smem_raw;         // A operand, stage 0 / 1 (producers write, consumers read)
    bfloat16_t* pA1 = pA0 + cosize(sAl);
    bfloat16_t* pBs = pA1 + cosize(sAl);             // B operand (consumers only)
    float* sEm0 = (float*)(pBs + cosize(sBl));       // element tile, stage 0 / 1 (producers write, consumers read)
    float* sEm1 = sEm0 + BK * BN;
    float* sMx = sEm1 + BK * BN;                     // per-batch max (consumers only)
    float* sMx4 = sMx + BN;                          // its NC / BN partials per column

    if (tid >= NC) {
        // ---------------- producers: stage tile kt into (pA, sEm)[kt & 1] ----------------
        int ptid = tid - NC;
        for (int kt = 0; kt < knt; kt++) {
            int s = kt & 1;
            if (kt >= 2) bar_sync(3 + s, NTH);       // the consumers are done with tile kt - 2 on this stage
            bfloat16_t* pA = s ? pA1 : pA0;
            float* sEm = s ? sEm1 : sEm0;
            Tensor sAt = make_tensor(make_smem_ptr(pA), sAl);
            long pc = pb[kt] + (long)tile_id * BM;
            // A operand[m, e] = params, fp32 -> bf16 (8 lanes via two float4): a thread's PA items, all loads first
            // (the loads in flight a DRAM stream needs: MEASURED one load-convert-store at a time ran 1.17x SLOWER
            // than the plain kernel)
            constexpr int PA = (BM * BK) / 8 / NP;
            static_assert((BM * BK) / 8 % NP == 0, "params items per producer thread");
            {
                float v[PA][8];
                CUTE_UNROLL
                for (int t = 0; t < PA; t++) {
                    int i = ptid + t * NP;
                    int e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                    ParamLoad<TP>::load8(&mp[pc + (long)e * block_size + mm], v[t]);
                }
                CUTE_UNROLL
                for (int t = 0; t < PA; t++) {
                    int i = ptid + t * NP;
                    int e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                    bfloat16_t r[8];
                    CUTE_UNROLL
                    for (int j = 0; j < 8; j++) r[j] = static_cast<bfloat16_t>(v[t][j]);
                    *(float4*)&sAt(mm, e) = *(const float4*)r;
                }
            }
            long e0 = eb[kt];
            // the element tile from `element_mars` rows e0 .. e0 + BK - 1 (contiguous: the layout condition)
            constexpr int PE = (BK * BN) / 4 / NP;
            static_assert((BK * BN) / 4 % NP == 0, "element items per producer thread");
            float4 v[PE];
            CUTE_UNROLL
            for (int t = 0; t < PE; t++) {
                int i = ptid + t * NP;
                int e = i / (BN / 4), bb = (i % (BN / 4)) * 4;
                v[t] = *(const float4*)&element_mars[(e0 + e) * (long)batch + b0 + bb];
            }
            CUTE_UNROLL
            for (int t = 0; t < PE; t++) {
                int i = ptid + t * NP;
                int e = i / (BN / 4), bb = (i % (BN / 4)) * 4;
                *(float4*)&sEm[e * BN + bb] = v[t];
            }
            bar_arrive(1 + s, NTH);                  // tile kt is staged
        }
        return;
    }

    // ---------------- consumers ----------------
    Tensor sBt = make_tensor(make_smem_ptr(pBs), sBl);
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                  Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(pA0), sAl);
    Tensor sAt1 = make_tensor(make_smem_ptr(pA1), sAl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    Tensor tXsA0 = s2rA.get_thread_slice(tid).partition_S(sAt0);
    Tensor tXsA1 = s2rA.get_thread_slice(tid).partition_S(sAt1);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXsB = s2rB.get_thread_slice(tid).partition_S(sBt);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    Tensor tCrM = thr.partition_fragment_C(cC);
    clear(tCrL);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrM); i++) tCrM(i) = -INFINITY;

    for (int kt = 0; kt < knt; kt++) {
        int s = kt & 1;
        bar_sync(1 + s, NTH);                        // tile kt staged
        const float* sEm = s ? sEm1 : sEm0;

        // per-batch max over this tile's edges (max-stabilization): every consumer thread a column and a slice of
        // the edges (MEASURED: one thread per column over all 64 edges is a 64-deep chain of shared loads, ~1 us a
        // tile, that nothing hides with one block per SM), then the slices' partials
        {
            constexpr int PARTS = NC / BN;
            static_assert(NC % BN == 0 && BK % PARTS == 0, "consumer threads over the batch columns");
            int b = tid % BN, part = tid / BN;
            float mx = -INFINITY;
            CUTE_UNROLL
            for (int e = part * (BK / PARTS); e < (part + 1) * (BK / PARTS); e++) mx = fmaxf(mx, sEm[e * BN + b]);
            sMx4[part * BN + b] = mx;
        }
        bar_sync(5, NC);
        for (int b = tid; b < BN; b += NC) {
            float mx = sMx4[b];
            CUTE_UNROLL
            for (int q = 1; q < NC / BN; q++) mx = fmaxf(mx, sMx4[q * BN + b]);
            sMx[b] = mx;
        }
        bar_sync(5, NC);

        // B operand[b, e] = exp(element - max) in bf16 (8 lanes via float4)
        for (int i = tid; i < (BN * BK) / 8; i += NC) {
            int e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
            float mx0 = sMx[bb], mx1 = sMx[bb + 1], mx2 = sMx[bb + 2], mx3 = sMx[bb + 3];
            float mx4 = sMx[bb + 4], mx5 = sMx[bb + 5], mx6 = sMx[bb + 6], mx7 = sMx[bb + 7];
            const float* sv = &sEm[e * BN + bb];
            bfloat16_t r[8];
            r[0] = static_cast<bfloat16_t>((mx0 == -INFINITY) ? 0.f : __expf(sv[0] - mx0));
            r[1] = static_cast<bfloat16_t>((mx1 == -INFINITY) ? 0.f : __expf(sv[1] - mx1));
            r[2] = static_cast<bfloat16_t>((mx2 == -INFINITY) ? 0.f : __expf(sv[2] - mx2));
            r[3] = static_cast<bfloat16_t>((mx3 == -INFINITY) ? 0.f : __expf(sv[3] - mx3));
            r[4] = static_cast<bfloat16_t>((mx4 == -INFINITY) ? 0.f : __expf(sv[4] - mx4));
            r[5] = static_cast<bfloat16_t>((mx5 == -INFINITY) ? 0.f : __expf(sv[5] - mx5));
            r[6] = static_cast<bfloat16_t>((mx6 == -INFINITY) ? 0.f : __expf(sv[6] - mx6));
            r[7] = static_cast<bfloat16_t>((mx7 == -INFINITY) ? 0.f : __expf(sv[7] - mx7));
            *(float4*)&sBt(bb, e) = *(const float4*)r;
        }
        bar_sync(5, NC);

        // bf16 tensor-core dot: tCrS[m, b] = sum_e params[m, e] * exp(emar[e, b] - max[b]); fp32 accumulate
        if (s) copy(s2rA, tXsA1, tXrA); else copy(s2rA, tXsA0, tXrA);
        copy(s2rB, tXsB, tXrB);
        bar_arrive(3 + s, NTH);                      // the stage's A operand and element tile are consumed
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);

        // online log-sum-exp: merge this tile's (partial = tCrS, max = mxk) into the running (L = tCrL, M = tCrM)
        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) {
            int b = get<1>(tCcC(i));
            float mxk = sMx[b];
            float partial = tCrS(i), Mo = tCrM(i), Lo = tCrL(i);
            float nM = fmaxf(Mo, mxk);
            if (nM != -INFINITY) {
                float pa = (Mo == -INFINITY) ? 0.f : Lo * __expf(Mo - nM);
                float pcc = (mxk == -INFINITY) ? 0.f : partial * __expf(mxk - nM);
                tCrL(i) = pa + pcc; tCrM(i) = nM;
            }
        }
        bar_sync(5, NC);                             // sMx and sBt are reused by the next tile
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (m < BM && b < BN)
            node_mars[(off_nid + (long)tile_id * BM + m) * (long)batch + b0 + b] =
                (tCrM(i) == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + tCrM(i));
    }
}

template <int BM, int BN, int WM, int WN, int PW, typename TP>
static void launch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params, torch::Tensor nids,
                       torch::Tensor ebase, torch::Tensor pbase, int batch, int block_size, int knt) {
    constexpr int NTH = (WM * WN + PW) * 32;
    int smem = 2 * BM * BK * 2 + BN * BK * 2 + 2 * BK * BN * 4 + (BN + (WM * WN * 32 / BN) * BN + 4) * 4;
    cudaFuncSetAttribute(tlmm_pipe_kernel<BM, BN, WM, WN, PW, TP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int total_m = nids.size(0) * (block_size / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        tlmm_pipe_kernel<BM, BN, WM, WN, PW, TP><<<grid, NTH, smem, c10::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), param_ptr<TP>(params), nids.data_ptr<long>(),
            ebase.data_ptr<long>(), pbase.data_ptr<long>(), batch, block_size, knt, off);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Config table (id -> {BM, BN, WM, WN, PW}): the two stages of each operand bound the node tile to 128 rows within the
// 99 KB of shared memory of sm_120 (2 x 16 KB A, 2 x 16 KB elements, 8 KB B at (128, 64)); 16 MMA warps (MEASURED
// above); 4 producer warps as the alternative (MEASURED 0.63 against 0.58 at B 128, 0.49 against 0.49 at B 64); the
// 64-row tile for blocks of 64.
template <typename TP>
static void dispatch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params, torch::Tensor nids,
                         torch::Tensor ebase, torch::Tensor pbase, int b, int bs, int k, int64_t cfg) {
    switch (cfg) {
        case 0: launch_cfg<128, 64, 8, 2, 8, TP>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
        case 1: launch_cfg<128, 64, 8, 2, 4, TP>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
        case 2: launch_cfg<64, 64, 4, 2, 8, TP>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
        default: launch_cfg<128, 64, 8, 2, 8, TP>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
    }
}

// The parameters in the dtype they are stored in (`TensorCircuit.param_dtype`): fp32 or bf16.
void tlmm_forward_sum_pipelined(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                                torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase, int64_t batch,
                                int64_t block_size, int64_t knt, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, k = (int)knt;
    if (params.scalar_type() == torch::kBFloat16)
        dispatch_cfg<__nv_bfloat16>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k, cfg);
    else
        dispatch_cfg<float>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k, cfg);
}

// (BM, BN) per config id; Python uses this to enumerate the configs valid for a layer shape.
std::vector<std::vector<int64_t>> configs() {
    return {{128, 64}, {128, 64}, {64, 64}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tlmm_forward_sum_pipelined", &tlmm_forward_sum_pipelined,
          "block-sparse sum-layer forward, producer / consumer warps (CuTe)");
    m.def("configs", &configs, "list of (BM, BN) per config id");
}
