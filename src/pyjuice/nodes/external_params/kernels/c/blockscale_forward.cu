// CuTe/TMA forward for the per-block multiplicative external parameterization (`BlockScaleSumParams`).
//
// A fork of pyjuice's `tlmm_forward_sum.cu` -- same TMA bulk-load, same MN-major staging, same SM80
// bf16 MMA, same register-resident online log-sum-exp -- with ONE addition in the k-loop and the
// normalizer computed alongside. Forked rather than parameterized because the standard kernel is on the
// hot path of every sum layer and should not grow a branch for this.
//
// The effective parameters are
//
//     theta_tilde[b,n,c] = phi[b, g(n,c)] * theta[n,c] / Z[n,b]
//     Z[n,b]             = sum_c phi[b, g(n,c)] * theta[n,c]
//
// i.e. `phi` reweights the GATE-level mixture weights per sample and leaves each gate's internal
// conditional alone -- an exact no-op when a node has one gate.
//
// WHY THE MAIN KERNEL ONLY NEEDS ONE EXTRA LOOP: `phi` is constant within a gate, so it pulls out of the
// matmul and, in log space, folds into the child values:
//
//     N[n,b] = sum_c theta[n,c] * exp( element_mars[c,b] + log phi[b,g] )
//
// The standard kernel already TMA-loads `element_mars` into an fp32 scratch before max-stabilizing it,
// so the fold is an elementwise add on that scratch between the barrier wait and the max. Nothing else
// about the pipeline changes -- the MMA, the operand staging and the accumulators are untouched.
//
// WHY `Z` IS NOT A SECOND MMA: `Z` does not involve `element_mars` at all. Factored by gate,
//
//     Z[n,b] = sum_g phi[b,g] * sigma[g,n],     sigma[g,n] = sum_{c in g} theta[n,c]
//
// and `sigma` is BATCH-INDEPENDENT. A second MMA would spend another `BN x BK` of shared memory and a
// full second LDSM_T + gemm computing something two small kernels get for a fraction of the work: at
// `block_size = 64` with 8 child gates per edge block there are 8x fewer gate terms than edges, and
// `sigma` costs one pass over the parameters (recomputed each forward, so there is no cache to
// invalidate when EM moves them).
//
// SCOPE: inherits the standard kernel's gate -- TMA-capable sm_90+, LL propagation, bf16, no partial
// eval or tempering, `block_size % BM == 0`, `batch % BN == 0`, `num_edges % BK == 0`. `n_node_gates`
// must be 1: a gate that varies across the nodes of a tile cannot share the staged tile. The Python
// layer raises `NotImplementedError` outside all of that rather than falling back to something slower.

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

// ---- TMA / mbarrier PTX helpers (sm_90+), as in the standard kernel ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, int bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}
__device__ __forceinline__ void tma_load_2d(void* smem, const CUtensorMap* desc, int c0, int c1,
                                            uint64_t* bar) {
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


// ============================================================ main kernel: log N

template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32) blockscale_tlmm_kernel(
        float* __restrict__ node_mars, const float* __restrict__ mp,
        const float* __restrict__ ext, const long* __restrict__ nids,
        const long* __restrict__ ebase, const long* __restrict__ pbase,
        const long* __restrict__ gate,
        int batch, int block_size, int knt, int gate_stride,
        int node_cbs, int gate_cbs, long ext_base, int pid_m_offset,
        const __grid_constant__ CUtensorMap desc) {
    constexpr int NTH = WM * WN * 32;

    int mtiles = block_size / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int nblock = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_nid = nids[nblock];
    const long* eb = ebase + (long)nblock * knt;
    const long* pb = pbase + (long)nblock * knt;
    const long* gt = gate + (long)nblock * gate_stride;
    int tid = threadIdx.x;

    auto swz = composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<Shape<_8, _8>, _8>, Stride<Stride<_1, _64>, _8>>{});
    auto sAl = tile_to_shape(swz, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swz, make_shape(Int<BN>{}, Int<BK>{}));
    bfloat16_t* pA = (bfloat16_t*)smem_raw;
    bfloat16_t* pBs = pA + cosize(sAl);
    float* sEm = (float*)(pBs + cosize(sBl));
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
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    Tensor tCrM = thr.partition_fragment_C(cC);
    clear(tCrL);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrM); i++) tCrM(i) = -INFINITY;

    if (tid == 0) mbar_init(bar, 1);
    __syncthreads();
    int phase = 0;

    for (int kt = 0; kt < knt; kt++) {
        long pc = pb[kt] + (long)tile_id * BM;
        if (tid == 0) { mbar_expect(bar, BK * BN * 4); tma_load_2d(sEm, &desc, b0, (int)eb[kt], bar); }
        mbar_wait(bar, phase); phase ^= 1;

        // ---- THE FOLD: log phi added to the child values, before max-stabilization ----
        // `sEm` is [edge, batch]; the gate of edge `kt*BK + e` is its edge block `j` and, within that,
        // the child gate `d`. `-inf + anything = -inf`, so a padded edge (child 0, the dummy) and a row
        // with fewer edge blocks (`gate == -1`) both stay -inf and contribute exactly nothing.
        for (int i = tid; i < BK * BN; i += NTH) {
            int e = i / BN, b = i % BN;
            int ge = kt * BK + e;
            int j = ge / node_cbs;
            int d = (ge % node_cbs) / gate_cbs;

            long gb = (j < gate_stride) ? gt[j] : -1;
            sEm[i] += (gb >= 0) ? ext[(gb + ext_base + d) * (long)batch + b0 + b] : -INFINITY;
        }
        __syncthreads();

        for (int b = tid; b < BN; b += NTH) {
            float mx = -INFINITY;
            for (int e = 0; e < BK; e++) mx = fmaxf(mx, sEm[e * BN + b]);
            sMx[b] = mx;
        }
        __syncthreads();

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

        copy(s2rA, tXsA, tXrA); copy(s2rB, tXsB, tXrB);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);

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

    // Writes log N; `blockscale_normalize` turns it into log N - log Z.
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (m < BM && b < BN)
            node_mars[(off_nid + (long)tile_id * BM + m) * (long)batch + b0 + b] =
                (tCrM(i) == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + tCrM(i));
    }
}


// ============================================================ sigma: per-gate parameter mass

// `sigma[row, j, d, m] = sum_{c in child gate d of edge block j} theta[row's node m, c]`.
// Batch-independent, so it is the whole reason `Z` needs no second MMA. Recomputed each forward: it is
// one pass over the parameters, which removes any question of staleness after an EM update.
__global__ void blockscale_sigma_kernel(
        const float* __restrict__ mp, const long* __restrict__ pids, float* __restrict__ sigma,
        int rows, int num_edges, int block_size, int node_cbs, int gate_cbs, int n_child_gates) {

    const int n_eblks = num_edges / node_cbs;
    const long total = (long)rows * n_eblks * n_child_gates * block_size;

    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int m = i % block_size;
        const int d = (i / block_size) % n_child_gates;
        const int j = (i / ((long)block_size * n_child_gates)) % n_eblks;
        const int row = i / ((long)block_size * n_child_gates * n_eblks);

        const long e0 = (long)row * num_edges + (long)j * node_cbs + (long)d * gate_cbs;

        float s = 0.0f;
        for (int c = 0; c < gate_cbs; ++c) s += mp[pids[e0 + c] + m];

        sigma[i] = s;
    }
}


// ============================================================ normalize: Z, and log N -> log N - log Z

__global__ void blockscale_normalize_kernel(
        float* __restrict__ node_mars, const float* __restrict__ ext,
        const float* __restrict__ sigma, const long* __restrict__ nids,
        const long* __restrict__ gate, float* __restrict__ log_z_out,
        int rows, int batch, int block_size, int n_eblks, int n_child_gates,
        int gate_stride, long ext_base) {

    const long total = (long)rows * block_size * batch;

    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int b = i % batch;
        const int m = (i / batch) % block_size;
        const int row = i / ((long)batch * block_size);

        // Max-stabilized over the gates: one exp per gate, one log at the end
        float mz = -INFINITY;
        for (int j = 0; j < n_eblks; ++j) {
            const long gb = (j < gate_stride) ? gate[(long)row * gate_stride + j] : -1;
            if (gb < 0) continue;
            for (int d = 0; d < n_child_gates; ++d)
                mz = fmaxf(mz, ext[(gb + ext_base + d) * (long)batch + b]);
        }

        float acc = 0.0f;
        if (mz != -INFINITY) {
            for (int j = 0; j < n_eblks; ++j) {
                const long gb = (j < gate_stride) ? gate[(long)row * gate_stride + j] : -1;
                if (gb < 0) continue;
                for (int d = 0; d < n_child_gates; ++d) {
                    const float lphi = ext[(gb + ext_base + d) * (long)batch + b];
                    const long si = (((long)row * n_eblks + j) * n_child_gates + d) * block_size + m;
                    acc = fmaf(__expf(lphi - mz), sigma[si], acc);
                }
            }
        }

        const float log_z = (acc <= 0.0f) ? -INFINITY : mz + logf(acc);
        const long o = (nids[row] + m) * (long)batch + b;

        node_mars[o] = (node_mars[o] == -INFINITY || log_z == -INFINITY)
                       ? -INFINITY : node_mars[o] - log_z;

        if (log_z_out != nullptr) log_z_out[i] = log_z;
    }
}


// ============================================================ launchers

template <int BM, int BN, int WM, int WN>
static void launch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                       torch::Tensor ext, torch::Tensor nids, torch::Tensor ebase,
                       torch::Tensor pbase, torch::Tensor gate,
                       int batch, int block_size, int knt, int node_cbs, int gate_cbs,
                       long ext_base) {
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
    cudaFuncSetAttribute(blockscale_tlmm_kernel<BM, BN, WM, WN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    int total_m = nids.size(0) * (block_size / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        blockscale_tlmm_kernel<BM, BN, WM, WN><<<grid, NTH, smem, c10::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), params.data_ptr<float>(), ext.data_ptr<float>(),
            nids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(),
            gate.data_ptr<long>(), batch, block_size, knt, (int)gate.size(1),
            node_cbs, gate_cbs, ext_base, off, desc);
    }
}


std::vector<std::vector<int>> configs() {
    return {{128, 64, 2, 2}, {64, 64, 2, 2}, {256, 64, 4, 2}, {128, 128, 2, 4}};
}


void blockscale_forward(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                        torch::Tensor ext, torch::Tensor nids, torch::Tensor ebase,
                        torch::Tensor pbase, torch::Tensor pids, torch::Tensor gate,
                        torch::Tensor sigma, torch::Tensor log_z,
                        int64_t block_size, int64_t num_edges, int64_t node_cbs, int64_t gate_cbs,
                        int64_t n_node_gates, int64_t ext_base, int64_t cfg) {

    const int batch = node_mars.size(1);
    const int rows = nids.size(0);
    const int knt = (int)(num_edges / BK);
    const int n_eblks = (int)(num_edges / node_cbs);
    const int n_child_gates = (int)(node_cbs / gate_cbs);

    TORCH_CHECK(n_node_gates == 1,
                "blockscale forward: a gate varying across the nodes of a block cannot share the staged "
                "tile; got n_node_gates = ", n_node_gates);
    TORCH_CHECK(num_edges % BK == 0, "blockscale forward: num_edges must be a multiple of ", BK);
    TORCH_CHECK(node_cbs % gate_cbs == 0,
                "blockscale forward: the gate's ch_block_size must divide the node's");

    // sigma: one pass over the parameters, then Z is a small batch-independent contraction
    {
        const long total = (long)rows * n_eblks * n_child_gates * block_size;
        const int threads = 256;
        const int blocks = (int)std::min<long>(1024, (total + threads - 1) / threads);
        blockscale_sigma_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            params.data_ptr<float>(), pids.data_ptr<long>(), sigma.data_ptr<float>(),
            rows, (int)num_edges, (int)block_size, (int)node_cbs, (int)gate_cbs, n_child_gates);
    }

    switch (cfg) {
        case 0: launch_cfg<128, 64, 2, 2>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                          gate, batch, (int)block_size, knt, (int)node_cbs,
                                          (int)gate_cbs, (long)ext_base); break;
        case 1: launch_cfg<64, 64, 2, 2>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                         gate, batch, (int)block_size, knt, (int)node_cbs,
                                         (int)gate_cbs, (long)ext_base); break;
        case 2: launch_cfg<256, 64, 4, 2>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                          gate, batch, (int)block_size, knt, (int)node_cbs,
                                          (int)gate_cbs, (long)ext_base); break;
        default: launch_cfg<128, 128, 2, 4>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                            gate, batch, (int)block_size, knt, (int)node_cbs,
                                            (int)gate_cbs, (long)ext_base); break;
    }

    {
        const long total = (long)rows * block_size * batch;
        const int threads = 256;
        const int blocks = (int)std::min<long>(4096, (total + threads - 1) / threads);
        blockscale_normalize_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), ext.data_ptr<float>(), sigma.data_ptr<float>(),
            nids.data_ptr<long>(), gate.data_ptr<long>(),
            log_z.numel() ? log_z.data_ptr<float>() : nullptr,
            rows, batch, (int)block_size, n_eblks, n_child_gates, (int)gate.size(1),
            (long)ext_base);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_forward", &blockscale_forward,
          "CuTe/TMA forward for the per-block multiplicative gate (log N - log Z)");
    m.def("configs", &configs, "Tile shapes {BM, BN, WM, WN} per config id");
}
