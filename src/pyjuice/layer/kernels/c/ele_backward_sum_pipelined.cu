// CUDA (CuTe/CUTLASS) block-sparse sum-layer ELEMENT-FLOW backward, warp-specialized: producer warps stream the
// operands while consumer warps compute. The same computation as ele_backward_sum.cu, to the bit.
//
// Why: MEASURED by ablation on HCLT-1024's widest layer at B 128 (143 child blocks of 1024, 1024 parents each), the
// kernel's phases run back to back -- the params' global fp32 load -> fp16 -> shared write is 0.30 of its 0.83 ms,
// the node flows / values and the log ratio 0.16, the MMA 0.11, the log-add-exp 0.10, the loop itself 0.14 -- while
// the parameters alone are 600 MB read once, a 0.41 ms DRAM floor. Here PW producer warps load the two node tiles
// (contiguous rows: the layout condition) and write their log ratio `node_flows - node_mars` into a double-buffered
// fp32 stage, and the params (fp32 -> fp16) into a double-buffered A stage, one edge tile ahead, with every load of a
// thread's share in flight before any convert or store; the consumer warps run the per-batch max, exp into the fp16
// B operand, the 16x8x16 tensor-core MMA and the online log-add-exp on the other stage; hand-off by named barriers
// (FULL / EMPTY per stage). 8 x 2 MMA warps: the consumers' phases are latency-bound and want 16 warps per SM, which
// the plain kernel gets from two blocks per SM and this kernel, at 74 KB for its two stages, must hold itself (the
// forward's copy, tlmm_forward_sum_pipelined.cu, MEASURED 4 x 2: 0.82 ms, 8 x 2: 0.58). No TMA: nothing here needs
// sm_90.
//
// SPECIALIZED fast path under the plain kernel's conditions (LL, logspace flows, allow_modify_flows / allow_neg_flows
// / accumulate_ch_flows off, no tempering, the fp16 dot regime, TILE_SIZE_K == 64, block_size % 128 == 0,
// batch % 64 == 0, contiguous edge/param layout); the Python dispatcher measures it against the plain kernel per shape.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BM 128
#define BN 64
#define BK 64
#define WM 8
#define WN 2
#define PW 8
#define NC (WM * WN * 32)
#define NP (PW * 32)
#define NTH (NC + NP)

// Named barriers: all `count` threads of the participating warps take part; `arrive` does not wait.
__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
// ids: 1 + s = stage s FULL (producers arrive, consumers sync); 3 + s = stage s EMPTY (consumers arrive, producers
// sync); 5 = the consumers among themselves.

extern __shared__ char smem_raw[];

__global__ void __launch_bounds__(NTH) ele_pipe_kernel(
        float* __restrict__ eflows, const float* __restrict__ emars_g, const float* __restrict__ nflows_g,
        const float* __restrict__ nmars_g, const float* __restrict__ mp,
        const long* __restrict__ chids, const long* __restrict__ ebase, const long* __restrict__ pbase,
        int batch, int BSK, int BSM, int knt, int pid_m_offset) {
    int mtiles = BSM / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int eb = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_ele = chids[eb] + (long)tile_id * BM;
    const long* ebp = ebase + (long)eb * knt;
    const long* pbp = pbase + (long)eb * knt;
    int tid = threadIdx.x;
    // K(edge)-major swizzled smem for both operands (both edge-contiguous) -> LDSM_N.
    auto swz = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
    auto sAl = tile_to_shape(swz, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swz, make_shape(Int<BN>{}, Int<BK>{}));
    half_t* pA0 = (half_t*)smem_raw;                 // A operand (params, fp16), stage 0 / 1: producers write
    half_t* pA1 = pA0 + cosize(sAl);
    half_t* pBs = pA1 + cosize(sAl);                 // B operand (consumers only)
    float* sLf0 = (float*)(pBs + cosize(sBl));       // log ratio tile [BK, BN], stage 0 / 1: producers write
    float* sLf1 = sLf0 + BK * BN;
    float* sMx = sLf1 + BK * BN;                     // per-batch max (consumers only)
    float* sMx4 = sMx + BN;                          // its NC / BN partials per column

    if (tid >= NC) {
        // ---------------- producers: stage tile kt into (pA, sLf)[kt & 1] ----------------
        int ptid = tid - NC;
        for (int kt = 0; kt < knt; kt++) {
            int s = kt & 1;
            if (kt >= 2) bar_sync(3 + s, NTH);       // the consumers are done with tile kt - 2 on this stage
            half_t* pA = s ? pA1 : pA0;
            float* sLf = s ? sLf1 : sLf0;
            Tensor sAt = make_tensor(make_smem_ptr(pA), sAl);
            long ec = ebp[kt]; long pc = pbp[kt] + (long)tile_id * BM * BSK;
            // A[m, e] = params (e-contiguous, 8 per two float4), a thread's PA items, all loads first
            constexpr int PA = (BM * BK) / 8 / NP;
            static_assert((BM * BK) / 8 % NP == 0, "params items per producer thread");
            {
                float4 a[PA], b[PA];
                CUTE_UNROLL
                for (int t = 0; t < PA; t++) {
                    int i = ptid + t * NP;
                    int m = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                    const float* g = &mp[pc + (long)m * BSK + ee];
                    a[t] = *(const float4*)g; b[t] = *(const float4*)(g + 4);
                }
                CUTE_UNROLL
                for (int t = 0; t < PA; t++) {
                    int i = ptid + t * NP;
                    int m = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                    half_t r[8];
                    r[0] = static_cast<half_t>(a[t].x); r[1] = static_cast<half_t>(a[t].y);
                    r[2] = static_cast<half_t>(a[t].z); r[3] = static_cast<half_t>(a[t].w);
                    r[4] = static_cast<half_t>(b[t].x); r[5] = static_cast<half_t>(b[t].y);
                    r[6] = static_cast<half_t>(b[t].z); r[7] = static_cast<half_t>(b[t].w);
                    *(float4*)&sAt(m, ee) = *(const float4*)r;
                }
            }
            // log ratio[e, b] = node_flows[ec + e, b] - node_mars[ec + e, b] (-inf value -> -inf), float4 along the
            // batch, a thread's PE items, all loads first
            constexpr int PE = (BK * BN) / 4 / NP;
            static_assert((BK * BN) / 4 % NP == 0, "node items per producer thread");
            {
                float4 nf[PE], nm[PE];
                CUTE_UNROLL
                for (int t = 0; t < PE; t++) {
                    int i = ptid + t * NP;
                    int e = i / (BN / 4), bb = (i % (BN / 4)) * 4;
                    long off = (ec + e) * (long)batch + b0 + bb;
                    nf[t] = *(const float4*)&nflows_g[off]; nm[t] = *(const float4*)&nmars_g[off];
                }
                CUTE_UNROLL
                for (int t = 0; t < PE; t++) {
                    int i = ptid + t * NP;
                    int e = i / (BN / 4), bb = (i % (BN / 4)) * 4;
                    float4 lf;
                    lf.x = (nm[t].x == -INFINITY) ? -INFINITY : nf[t].x - nm[t].x;
                    lf.y = (nm[t].y == -INFINITY) ? -INFINITY : nf[t].y - nm[t].y;
                    lf.z = (nm[t].z == -INFINITY) ? -INFINITY : nf[t].z - nm[t].z;
                    lf.w = (nm[t].w == -INFINITY) ? -INFINITY : nf[t].w - nm[t].w;
                    *(float4*)&sLf[e * BN + bb] = lf;
                }
            }
            bar_arrive(1 + s, NTH);                  // tile kt is staged
        }
        return;
    }

    // ---------------- consumers ----------------
    Tensor sBt = make_tensor(make_smem_ptr(pBs), sBl);
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(pA0), sAl);
    Tensor sAt1 = make_tensor(make_smem_ptr(pA1), sAl);
    Tensor tCrA = thr.partition_fragment_A(sAt0); Tensor tCrB = thr.partition_fragment_B(sBt);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, half_t>{}, mma);
    Tensor tXsA0 = s2rA.get_thread_slice(tid).partition_S(sAt0);
    Tensor tXsA1 = s2rA.get_thread_slice(tid).partition_S(sAt1);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXsB = s2rB.get_thread_slice(tid).partition_S(sBt); Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{}); Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC); Tensor tCrL = thr.partition_fragment_C(cC); Tensor tCrM = thr.partition_fragment_C(cC);
    clear(tCrL);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrM); i++) tCrM(i) = -INFINITY;

    for (int kt = 0; kt < knt; kt++) {
        int s = kt & 1;
        bar_sync(1 + s, NTH);                        // tile kt staged
        const float* sLf = s ? sLf1 : sLf0;

        // per-batch max over this tile's edges: every consumer thread a column and a slice of the edges, then the
        // slices' partials
        {
            constexpr int PARTS = NC / BN;
            static_assert(NC % BN == 0 && BK % PARTS == 0, "consumer threads over the batch columns");
            int b = tid % BN, part = tid / BN;
            float mx = -INFINITY;
            CUTE_UNROLL
            for (int e = part * (BK / PARTS); e < (part + 1) * (BK / PARTS); e++) mx = fmaxf(mx, sLf[e * BN + b]);
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

        // B[b, e] = exp(log ratio - max) in fp16 (e-contiguous, 8 per float4)
        for (int i = tid; i < (BN * BK) / 8; i += NC) {
            int b = i / (BK / 8), ee = (i % (BK / 8)) * 8; float mx = sMx[b];
            half_t r[8];
            for (int j = 0; j < 8; j++) {
                int e = ee + j; float lf = sLf[e * BN + b];
                r[j] = static_cast<half_t>((mx == -INFINITY || lf == -INFINITY) ? 0.f : __expf(lf - mx));
            }
            *(float4*)&sBt(b, ee) = *(const float4*)r;
        }
        bar_sync(5, NC);

        // fp16 tensor-core dot: partial[m, b] = sum_e params[m, e] * exp(ratio[e, b] - max[b]); fp32 accumulate
        if (s) copy(s2rA, tXsA1, tXrA); else copy(s2rA, tXsA0, tXrA);
        copy(s2rB, tXsB, tXrB);
        bar_arrive(3 + s, NTH);                      // the stage's A operand and log-ratio tile are consumed
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);

        // online log-sum-exp merge of this tile's (partial, max) into the running (L = tCrL, M = tCrM)
        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) { int b = get<1>(tCcC(i));
            float partial = tCrS(i); float pfm = sMx[b];
            float Mo = tCrM(i), Lo = tCrL(i); float nM = fmaxf(Mo, pfm);
            if (nM != -INFINITY) { float pa = (Mo == -INFINITY) ? 0.f : Lo * __expf(Mo - nM); float pc2 = (pfm == -INFINITY) ? 0.f : partial * __expf(pfm - nM); tCrL(i) = pa + pc2; tCrM(i) = nM; } }
        bar_sync(5, NC);                             // sMx and sBt are reused by the next tile
    }
    // finalize: element_flows[m, b] = element_mars[m, b] + (log(L) + M)
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) { int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (m < BM && b < BN) { float r = (tCrM(i) == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + tCrM(i));
            if (r != -INFINITY) r += emars_g[(off_ele + m) * (long)batch + b0 + b];
            eflows[(off_ele + m) * (long)batch + b0 + b] = r; } }
}

// element-flow backward (writes element_flows in-place). chids/ebase/pbase are int64 [neb] / [neb,knt].
void ele_backward_sum_pipelined(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                                torch::Tensor node_mars, torch::Tensor params, torch::Tensor chids,
                                torch::Tensor ebase, torch::Tensor pbase,
                                int64_t batch, int64_t BSK, int64_t BSM, int64_t knt) {
    TORCH_CHECK(BSM % BM == 0, "block_size must be divisible by ", BM);
    TORCH_CHECK(batch % BN == 0, "batch must be divisible by ", BN);
    int n_nblocks = chids.size(0);
    int smem = 2 * BM * BK * 2 + BN * BK * 2 + 2 * BK * BN * 4 + (BN + NC) * 4 + 64;
    cudaFuncSetAttribute(ele_pipe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int total_m = n_nblocks * (BSM / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        ele_pipe_kernel<<<grid, NTH, smem, c10::cuda::getCurrentCUDAStream()>>>(
            element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), params.data_ptr<float>(),
            chids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(),
            (int)batch, (int)BSK, (int)BSM, (int)knt, off);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ele_backward_sum_pipelined", &ele_backward_sum_pipelined,
          "block-sparse sum-layer element-flow backward, producer / consumer warps (CuTe/fp16)");
}
