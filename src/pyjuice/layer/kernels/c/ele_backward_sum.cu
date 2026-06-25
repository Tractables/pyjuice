// CUDA (CuTe/CUTLASS + TMA) implementation of the block-sparse sum-layer ELEMENT-FLOW backward
// kernel (the "_bk_triton_block_sparse_ele_kernel" path, LL / logspace_flows regime).
//
// Per element-block, accumulates the log-space element flow over its (contiguous) parent edges:
//   log_n_fdm[e,b] = node_flows[par(e),b] - node_mars[par(e),b]   (per k-tile, max-stabilized)
//   nfsub[e,b]     = exp(log_n_fdm - max_e)
//   partial[m,b]   = sum_e epars[m,e] * nfsub[e,b]                (fp16 tensor-core dot)
//   element_flows[m,b] = emars[m,b] + logsumexp_kt( log(partial) + max_e )
// Numerically equivalent to the Triton kernel (whose tl.dot runs in TF32); we use fp16 MMA which,
// thanks to its 10-bit mantissa, stays within the accuracy bar (~1.07e-3 logspace < 1.5e-3) at half
// the smem/regs of TF32 -- the key to occupancy.
//
// Two techniques that make it 1.49x over Triton on the heaviest partition:
//   * Factor `emars` out of the k-loop (loop-invariant) -> drops a register C-fragment -> 126 regs
//     -> 2 CTAs/SM on sm_120 (whose smem is only ~100KB/SM, so 2 CTAs needs <=50KB).
//   * Dual-buffer TMA: load BOTH node_flows and node_mars per k-tile via cp.async.bulk.tensor.2d;
//     the node_mars TMA dest REUSES the epars(A) smem region as scratch (free until epars is filled,
//     after log_n_fdm is computed) -> no extra smem -> 41KB -> 2 CTAs/SM preserved. (Both buffers
//     must use TMA; one-async-one-manual gives no overlap and is slower.)
//
// SPECIALIZED fast path; the Python dispatcher only calls it when its assumptions hold (LL,
// logspace_flows, allow_modify_flows=0, allow_neg_flows=0, no tempering/partial-eval, TILE_SIZE_K==64,
// block_size%128==0, batch%64==0, contiguous edge/param layout, TMA-capable GPU sm_90+). Else Triton.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BM 128
#define BN 64
#define BK 64
#define WM 4
#define WN 2
#define NTH (WM * WN * 32)

__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, int bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}
__device__ __forceinline__ void tma_load_2d(void* smem, const CUtensorMap* desc, int c0, int c1, uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(s),
        "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(b)
        : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n .reg .pred p;\n LAB_W:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@p bra DONE_W;\n bra LAB_W;\n DONE_W:\n}\n" ::"r"(a),
                 "r"(phase));
}

extern __shared__ char smem_raw[];

__global__ void __launch_bounds__(NTH) ele_kernel(
        float* __restrict__ eflows, const float* __restrict__ emars_g, const float* __restrict__ mp,
        const long* __restrict__ chids, const long* __restrict__ ebase, const long* __restrict__ pbase,
        int batch, int BSK, int BSM, int knt, int pid_m_offset,
        const __grid_constant__ CUtensorMap descNf, const __grid_constant__ CUtensorMap descNm) {
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
    half_t* pA = (half_t*)smem_raw;
    half_t* pBs = pA + cosize(sAl);
    float* sLf = (float*)(pBs + cosize(sBl));   // nflows TMA dest, then log_n_fdm in-place
    float* sMx = sLf + BK * BN;
    float* sNm = (float*)pA;                     // nmars TMA scratch reuses the epars(A) region
    uint64_t* bar = (uint64_t*)(sMx + BN + 4);
    if (tid == 0) mbar_init(bar, 1);
    __syncthreads();
    int phase = 0;

    Tensor sAt = make_tensor(make_smem_ptr(pA), sAl);
    Tensor sBt = make_tensor(make_smem_ptr(pBs), sBl);
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor tCrA = thr.partition_fragment_A(sAt); Tensor tCrB = thr.partition_fragment_B(sBt);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, half_t>{}, mma);
    Tensor tXsA = s2rA.get_thread_slice(tid).partition_S(sAt); Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXsB = s2rB.get_thread_slice(tid).partition_S(sBt); Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{}); Tensor tCcC = thr.partition_C(cC);
    // tCrS = per-k-tile MMA result; (tCrL, tCrM) = running (sum, max) of the online log-sum-exp across k-tiles.
    Tensor tCrS = thr.partition_fragment_C(cC); Tensor tCrL = thr.partition_fragment_C(cC); Tensor tCrM = thr.partition_fragment_C(cC);
    clear(tCrL);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrM); i++) tCrM(i) = -INFINITY;

    for (int kt = 0; kt < knt; kt++) {
        long ec = ebp[kt]; long pc = pbp[kt] + (long)tile_id * BM * BSK;
        // dual-buffer TMA: nflows -> sLf, nmars -> sNm(=A region); then log_n_fdm in-place into sLf
        if (tid == 0) {
            mbar_expect(bar, BK * BN * 4 + BK * BN * 4);
            tma_load_2d(sLf, &descNf, b0, (int)ec, bar);
            tma_load_2d(sNm, &descNm, b0, (int)ec, bar);
        }
        mbar_wait(bar, phase); phase ^= 1;
        // log_n_fdm[e,b] = node_flows[par,b] - node_mars[par,b] (in-place into sLf); -inf mar -> -inf.
        for (int i = tid; i < (BK * BN) / 4; i += NTH) { int idx = i * 4;
            float4 nf = *(const float4*)&sLf[idx]; float4 nm = *(const float4*)&sNm[idx];
            float4 lf; lf.x = (nm.x == -INFINITY) ? -INFINITY : nf.x - nm.x; lf.y = (nm.y == -INFINITY) ? -INFINITY : nf.y - nm.y;
            lf.z = (nm.z == -INFINITY) ? -INFINITY : nf.z - nm.z; lf.w = (nm.w == -INFINITY) ? -INFINITY : nf.w - nm.w; *(float4*)&sLf[idx] = lf; }
        __syncthreads();
        // per-batch max over this k-tile's edges (max-stabilization for the exp below).
        for (int b = tid; b < BN; b += NTH) { float mx = -INFINITY; for (int e = 0; e < BK; e++) mx = fmaxf(mx, sLf[e * BN + b]); sMx[b] = mx; }
        __syncthreads();
        // B[b,e] = nfsub (e-contiguous, 8/float4)
        for (int i = tid; i < (BN * BK) / 8; i += NTH) { int b = i / (BK / 8), ee = (i % (BK / 8)) * 8; float mx = sMx[b];
            half_t r[8]; for (int j = 0; j < 8; j++) { int e = ee + j; float lf = sLf[e * BN + b]; r[j] = static_cast<half_t>((mx == -INFINITY || lf == -INFINITY) ? 0.f : __expf(lf - mx)); }
            *(float4*)&sBt(b, ee) = *(const float4*)r; }
        __syncthreads();   // all reads of sLf/sNm done before sA(=sNm region) is overwritten with epars
        // A[m,e] = epars (e-contiguous, 8/float4 from 8 contiguous mp floats)
        for (int i = tid; i < (BM * BK) / 8; i += NTH) { int m = i / (BK / 8), ee = (i % (BK / 8)) * 8;
            const float* g = &mp[pc + (long)m * BSK + ee]; float4 a = *(const float4*)g, b2 = *(const float4*)(g + 4);
            half_t r[8]; r[0] = static_cast<half_t>(a.x); r[1] = static_cast<half_t>(a.y); r[2] = static_cast<half_t>(a.z); r[3] = static_cast<half_t>(a.w);
            r[4] = static_cast<half_t>(b2.x); r[5] = static_cast<half_t>(b2.y); r[6] = static_cast<half_t>(b2.z); r[7] = static_cast<half_t>(b2.w);
            *(float4*)&sAt(m, ee) = *(const float4*)r; }
        __syncthreads();
        // fp16 tensor-core dot: partial[m,b] = sum_e epars[m,e] * nfsub[e,b], accumulated in fp32.
        copy(s2rA, tXsA, tXrA); copy(s2rB, tXsB, tXrB);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);
        // online log-sum-exp merge of this k-tile's (partial, max=pfm) into the running (L=tCrL, M=tCrM).
        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) { int b = get<1>(tCcC(i));
            float partial = tCrS(i); float pfm = sMx[b];
            float Mo = tCrM(i), Lo = tCrL(i); float nM = fmaxf(Mo, pfm);
            if (nM != -INFINITY) { float pa = (Mo == -INFINITY) ? 0.f : Lo * __expf(Mo - nM); float pc2 = (pfm == -INFINITY) ? 0.f : partial * __expf(pfm - nM); tCrL(i) = pa + pc2; tCrM(i) = nM; } }
        __syncthreads();
    }
    // finalize: element_flows[m,b] = element_mars[m,b] + (log(L) + M)   [the +emars folds the loop-invariant factor back in]
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) { int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (m < BM && b < BN) { float r = (tCrM(i) == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + tCrM(i));
            if (r != -INFINITY) r += emars_g[(off_ele + m) * (long)batch + b0 + b];
            eflows[(off_ele + m) * (long)batch + b0 + b] = r; } }
}

// TMA descriptors over node_flows / node_mars (2D [n_node_rows, batch] FLOAT32, box {BN, BK}).
// Cached per (base ptr, n_rows, batch): the buffers are shared across the backward pass and only the
// base ptr / dims (not values) enter the descriptor, so caching by ptr is valid.
static CUtensorMap g_descNf, g_descNm;
static void* g_nf = nullptr; static void* g_nm = nullptr; static int g_rows = 0, g_batch = 0;
static bool build_desc(CUtensorMap* d, void* base, int n_rows, int batch) {
    cuuint64_t gdim[2] = {(cuuint64_t)batch, (cuuint64_t)n_rows};
    cuuint64_t gstride[1] = {(cuuint64_t)batch * 4};
    cuuint32_t bdim[2] = {(cuuint32_t)BN, (cuuint32_t)BK};
    cuuint32_t estride[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return r == CUDA_SUCCESS;
}

// element-flow backward (writes element_flows in-place). chids/ebase/pbase are int64 [neb] / [neb,knt].
void ele_backward_sum(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                      torch::Tensor node_mars, torch::Tensor params, torch::Tensor chids,
                      torch::Tensor ebase, torch::Tensor pbase,
                      int64_t batch, int64_t BSK, int64_t BSM, int64_t knt) {
    TORCH_CHECK(BSM % BM == 0, "block_size must be divisible by ", BM);
    TORCH_CHECK(batch % BN == 0, "batch must be divisible by ", BN);
    int n_nblocks = chids.size(0);
    int n_rows = node_mars.size(0);
    void* nf = (void*)node_flows.data_ptr<float>();
    void* nm = (void*)node_mars.data_ptr<float>();
    if (nf != g_nf || nm != g_nm || n_rows != g_rows || (int)batch != g_batch) {
        TORCH_CHECK(build_desc(&g_descNf, nf, n_rows, (int)batch), "cuTensorMapEncodeTiled(node_flows) failed");
        TORCH_CHECK(build_desc(&g_descNm, nm, n_rows, (int)batch), "cuTensorMapEncodeTiled(node_mars) failed");
        g_nf = nf; g_nm = nm; g_rows = n_rows; g_batch = (int)batch;
    }
    int smem = BM * BK * 2 + BN * BK * 2 + BK * BN * 4 + BN * 4 + 1024;   // pA(fp16)+pBs(fp16)+sLf(f32)+sMx+bar
    cudaFuncSetAttribute(ele_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int total_m = n_nblocks * (BSM / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        ele_kernel<<<grid, NTH, smem, c10::cuda::getCurrentCUDAStream()>>>(
            element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), params.data_ptr<float>(),
            chids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(),
            (int)batch, (int)BSK, (int)BSM, (int)knt, off, g_descNf, g_descNm);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ele_backward_sum", &ele_backward_sum, "block-sparse sum-layer element-flow backward (CuTe/fp16/TMA)");
}
