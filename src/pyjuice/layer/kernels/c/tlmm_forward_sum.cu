// CUDA (CuTe/CUTLASS) implementation of the block-sparse sum-layer forward "tlmm" kernel.
//
// Computes, per output node-block, the log-sum-exp over edges:
//     node_mars[m, b] = logsumexp_e( log(params[m, e]) + element_mars[child(e), b] )
// using the same per-k-tile algorithm as the Triton kernel
// (`_fw_triton_block_sparse_tlmm_kernel`): per-tile max -> exp(emars - max) -> bf16 tensor-core
// dot(params, esub) -> online log-add-exp accumulation across k-tiles. Numerically equivalent to
// the Triton kernel (bf16 dot + fp32 accumulate), agreeing to ~1.5e-3 in log-space.
//
// SPECIALIZED fast path; the Python dispatcher only calls it when all assumptions hold (LL
// propagation, bf16 path, no partial-eval / tempering, block_size % BM == 0, batch % BN == 0,
// TILE_SIZE_K == 64, contiguous edge/param layout, TMA-capable GPU sm_90+). Otherwise pyjuice
// falls back to the Triton kernel.
//
// Techniques: TMA (cp.async.bulk.tensor.2d) async bulk-load of each element_mars tile; both MMA
// operands staged MN-major (float4-contiguous writes) + transposed into the SM80 16x8x16 bf16 MMA
// via transposed ldmatrix (LDSM_T); register-resident online (max, sum) accumulator.
//
// The tile shape (BM, BN, WM, WN) is a TEMPLATE parameter so several configs can be compiled and
// the best one selected per layer shape at runtime (autotuned in Python). BK (= TILE_SIZE_K) is
// fixed at 64; KNT (= num_edges / 64) is a runtime argument.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64

// ---- TMA / mbarrier PTX helpers (sm_90+) ----
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
    asm volatile("{\n .reg .pred p;\n LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@p bra DONE;\n bra LAB_WAIT;\n DONE:\n}\n" ::"r"(a),
                 "r"(phase));
}

extern __shared__ char smem_raw[];

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32) tlmm_kernel(
        float* __restrict__ node_mars, const float* __restrict__ mp,
        const long* __restrict__ nids, const long* __restrict__ ebase,
        const long* __restrict__ pbase, int batch, int block_size, int knt, int pid_m_offset,
        const __grid_constant__ CUtensorMap desc) {
    constexpr int NTH = WM * WN * 32;
    // grid = (batch/BN, chunk_of[n_nblocks * block_size/BM]); batch-tile on blockIdx.x (fast) so a
    // node-block's batch-tiles run together and reuse its params from L2. (node-block x m-tile) on
    // blockIdx.y, launched in <=65535 chunks via pid_m_offset.
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
    bfloat16_t* pA = (bfloat16_t*)smem_raw;
    bfloat16_t* pBs = pA + cosize(sAl);
    float* sEm = (float*)(pBs + cosize(sBl));   // emars scratch, TMA destination (128B-aligned)
    float* sMx = sEm + BK * BN;
    uint64_t* bar = (uint64_t*)(sMx + BN + 4);

    Tensor sAt = make_tensor(make_smem_ptr(pA), sAl);
    Tensor sBt = make_tensor(make_smem_ptr(pBs), sBl);
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                  Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor tCrA = thr.partition_fragment_A(sAt);
    Tensor tCrB = thr.partition_fragment_B(sBt);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    Tensor tXsA = s2rA.get_thread_slice(tid).partition_S(sAt);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXsB = s2rB.get_thread_slice(tid).partition_S(sBt);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);              // C coords -> (m, b) per fragment element
    Tensor tCrS = thr.partition_fragment_C(cC);     // per-k-tile MMA result
    Tensor tCrL = thr.partition_fragment_C(cC);     // running sum  L of the online log-sum-exp
    Tensor tCrM = thr.partition_fragment_C(cC);     // running max  M of the online log-sum-exp
    clear(tCrL);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrM); i++) tCrM(i) = -INFINITY;

    if (tid == 0) mbar_init(bar, 1);
    __syncthreads();
    int phase = 0;

    // Contract over edges in BK-sized k-tiles (knt = num_edges/BK). Per tile: TMA-load this tile of
    // element_mars, max-stabilize, exp into the B operand, load params into the A operand, MMA, then
    // online log-sum-exp merge.
    for (int kt = 0; kt < knt; kt++) {
        long pc = pb[kt] + (long)tile_id * BM;   // params base for this edge-tile + m-tile
        // TMA bulk-load element_mars[edge-tile, batch-tile] into the fp32 scratch sEm.
        if (tid == 0) { mbar_expect(bar, BK * BN * 4); tma_load_2d(sEm, &desc, b0, (int)eb[kt], bar); }
        mbar_wait(bar, phase); phase ^= 1;

        // per-batch max over this tile's edges (max-stabilization).
        for (int b = tid; b < BN; b += NTH) {
            float mx = -INFINITY;
            for (int e = 0; e < BK; e++) mx = fmaxf(mx, sEm[e * BN + b]);
            sMx[b] = mx;
        }
        __syncthreads();

        // B operand[b,e] = exp(element_mars - max) in bf16 (8 lanes via float4).
        for (int i = tid; i < (BN * BK) / 8; i += NTH) {
            int e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
            float mx0 = sMx[bb], mx1 = sMx[bb + 1], mx2 = sMx[bb + 2], mx3 = sMx[bb + 3];
            float mx4 = sMx[bb + 4], mx5 = sMx[bb + 5], mx6 = sMx[bb + 6], mx7 = sMx[bb + 7];
            const float* s = &sEm[e * BN + bb];
            bfloat16_t r[8];
            r[0] = static_cast<bfloat16_t>((mx0 == -INFINITY) ? 0.f : __expf(s[0] - mx0));
            r[1] = static_cast<bfloat16_t>((mx1 == -INFINITY) ? 0.f : __expf(s[1] - mx1));
            r[2] = static_cast<bfloat16_t>((mx2 == -INFINITY) ? 0.f : __expf(s[2] - mx2));
            r[3] = static_cast<bfloat16_t>((mx3 == -INFINITY) ? 0.f : __expf(s[3] - mx3));
            r[4] = static_cast<bfloat16_t>((mx4 == -INFINITY) ? 0.f : __expf(s[4] - mx4));
            r[5] = static_cast<bfloat16_t>((mx5 == -INFINITY) ? 0.f : __expf(s[5] - mx5));
            r[6] = static_cast<bfloat16_t>((mx6 == -INFINITY) ? 0.f : __expf(s[6] - mx6));
            r[7] = static_cast<bfloat16_t>((mx7 == -INFINITY) ? 0.f : __expf(s[7] - mx7));
            *(float4*)&sBt(bb, e) = *(const float4*)r;
        }
        // A operand[m,e] = log-params, cast fp32 -> bf16 (the dot computes params . exp(emar-max)).
        for (int i = tid; i < (BM * BK) / 8; i += NTH) {
            int e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
            const float* g = &mp[pc + (long)e * block_size + mm];
            float4 a = *(const float4*)g, b = *(const float4*)(g + 4);
            bfloat16_t r[8];
            r[0] = static_cast<bfloat16_t>(a.x); r[1] = static_cast<bfloat16_t>(a.y);
            r[2] = static_cast<bfloat16_t>(a.z); r[3] = static_cast<bfloat16_t>(a.w);
            r[4] = static_cast<bfloat16_t>(b.x); r[5] = static_cast<bfloat16_t>(b.y);
            r[6] = static_cast<bfloat16_t>(b.z); r[7] = static_cast<bfloat16_t>(b.w);
            *(float4*)&sAt(mm, e) = *(const float4*)r;
        }
        __syncthreads();

        // bf16 tensor-core dot: tCrS[m,b] = sum_e params[m,e] * exp(emar[e,b] - max[b]); fp32 accumulate.
        copy(s2rA, tXsA, tXrA); copy(s2rB, tXsB, tXrB);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);

        // online log-sum-exp: merge this tile's (partial=tCrS, max=mxk) into the running (L=tCrL, M=tCrM).
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
        __syncthreads();
    }
    // finalize: node_mars[m,b] = log(L) + M  (the log-sum-exp over all edges).
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (m < BM && b < BN)
            node_mars[(off_nid + (long)tile_id * BM + m) * (long)batch + b0 + b] =
                (tCrM(i) == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + tCrM(i));
    }
}

// Per-config host launcher. The TMA descriptor box dim depends on BN, so each instantiation caches
// its own descriptor (keyed by element_mars base ptr / dims, constant within a forward pass).
template <int BM, int BN, int WM, int WN>
static void launch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                       torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase,
                       int batch, int block_size, int knt) {
    constexpr int NTH = WM * WN * 32;
    int n_edge_rows = element_mars.size(0);
    void* base = (void*)element_mars.data_ptr<float>();
    static CUtensorMap desc;
    static void* desc_ptr = nullptr;
    static int desc_rows = 0, desc_batch = 0;
    if (base != desc_ptr || n_edge_rows != desc_rows || batch != desc_batch) {
        cuuint64_t gdim[2] = {(cuuint64_t)batch, (cuuint64_t)n_edge_rows};
        cuuint64_t gstride[1] = {(cuuint64_t)batch * 4};
        cuuint32_t bdim[2] = {(cuuint32_t)BN, (cuuint32_t)BK};
        cuuint32_t estride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, gdim, gstride, bdim, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
        desc_ptr = base; desc_rows = n_edge_rows; desc_batch = batch;
    }
    int smem = BM * BK * 2 + BN * BK * 2 + BK * BN * 4 + BN * 4 + 64;
    cudaFuncSetAttribute(tlmm_kernel<BM, BN, WM, WN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int total_m = nids.size(0) * (block_size / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        tlmm_kernel<BM, BN, WM, WN><<<grid, NTH, smem, c10::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), params.data_ptr<float>(), nids.data_ptr<long>(),
            ebase.data_ptr<long>(), pbase.data_ptr<long>(), batch, block_size, knt, off, desc);
    }
}

// Config table (id -> {BM, BN, WM, WN}); also exported to Python (see `configs`). The autotuner
// picks the fastest valid one per layer shape. (Software-pipelined / double-buffered-TMA variants
// were tried and removed: they never beat cfg0 on sm_120 because the per-tile bottleneck is the
// epars smem-write + logaddexp, not the emars load latency that prefetching would hide.)
void tlmm_forward_sum(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                      torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase,
                      int64_t batch, int64_t block_size, int64_t knt, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, k = (int)knt;
    switch (cfg) {
        case 0: launch_cfg<128, 64, 4, 2>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
        case 1: launch_cfg<64,  64, 2, 2>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
        case 2: launch_cfg<256, 64, 8, 2>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
        case 3: launch_cfg<128, 128, 4, 4>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
        default: launch_cfg<128, 64, 4, 2>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, k); break;
    }
}

// (BM, BN) per config id; Python uses this to enumerate the configs valid for a layer shape.
std::vector<std::vector<int64_t>> configs() {
    return {{128, 64}, {64, 64}, {256, 64}, {128, 128}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tlmm_forward_sum", &tlmm_forward_sum, "block-sparse sum-layer forward (CuTe/TMA)");
    m.def("configs", &configs, "list of (BM, BN) per config id");
}
