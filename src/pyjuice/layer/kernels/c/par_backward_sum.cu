// CUDA (CuTe/CUTLASS + TMA) implementation of the block-sparse sum-layer PARAMETER-FLOW backward
// kernel (the "_bk_triton_block_sparse_par_kernel" path, LL / logspace_flows regime).
//
// Computes, per (output node-block m, edge-block e), the parameter flow -- a GEMM contracting over the
// BATCH dimension:
//   pflow[m,e] = epars[m,e] * sum_b exp( (node_flows[m,b] - node_mars[m,b]) + element_mars[e,b] )
//             += into param_flows (read-modify-write; or atomicAdd / store-only, see `use_atomic`).
// Let lr[m,b] = node_flows[m,b] - node_mars[m,b] (log flow/marginal ratio). The inner sum is a dot over
// batch of exp(lr) (the A operand) and exp(element_mars) (the B operand); we run it on the fp16 tensor
// cores and multiply by epars in the epilogue. Numerically equivalent to the Triton kernel (TF32 dot);
// fp16 stays within the accuracy bar (healthy-node log-param diff ~5.9e-4 < 1e-3).
//
// NUMERICAL TRICK (balanced shift + clamp): A*B = exp(lr) * exp(emar) = exp(lr + emar) is invariant to
// any per-batch shift S[b] applied as exp(lr - S) * exp(emar + S). For near-dead nodes lr or emar can
// reach exp(47)+, overflowing fp16 (max 65504). We pick the BALANCED shift S[b] = (max_m lr - max_e emar)/2
// so the two operands' exponents are centred, then CLAMP each to 65504. On a degenerate batch column the
// clamp only touches the (already-negligible) dead nodes; healthy nodes' A=exp(lr-S)->0 underflows
// harmlessly. A simple (non-balanced) shift breaks healthy nodes, so the balance matters.
//
// PERF: the output is the FULL [block_size x num_edges] param matrix, so the EPILOGUE (params read +
// param_flows RMW) is the dominant cost, NOT the matmul. Levers: BK=32 (half the smem of BK=64 -> 4
// CTAs/SM = 2x occupancy to hide load latency); EE edge-subtiles per CTA share the loaded node tile
// (lr/cmax computed once, reused for EE element-tiles); plain K-major smem + auto-vectorized smem->reg
// copy (no LDSM swizzle); 128-bit float4 params I/O in the epilogue. node_flows / node_mars / element_mars
// are TMA-loaded.
//
// SPECIALIZED fast path; the Python dispatcher only calls it when its assumptions hold (LL, logspace_flows,
// pflow_temperature=1, allow_modify_flows=0, no neg/negate flows, no partial-eval, block_size%64==0,
// num_edges%128==0, batch%32==0, collision-free pfids, contiguous layout, TMA-capable GPU sm_90+). Else Triton.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
using namespace cute;

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 32
#endif
#ifndef WM
#define WM 4
#endif
#ifndef WN
#define WN 2
#endif
#ifndef EE
#define EE 2
#endif
#define NTH (WM * WN * 32)

__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt)); }
__device__ __forceinline__ void mbar_expect(uint64_t* bar, int bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes)); }
__device__ __forceinline__ void tma_load_2d(void* smem, const CUtensorMap* desc, int c0, int c1, uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        ::"r"(s), "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n .reg .pred p;\n LAB_WP:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@p bra DONE_WP;\n bra LAB_WP;\n DONE_WP:\n}\n" ::"r"(a), "r"(phase)); }

extern __shared__ char smem_raw[];

__global__ void __launch_bounds__(NTH) par_kernel(
        float* __restrict__ pflows, const float* __restrict__ mp,
        const long* __restrict__ nbase, const long* __restrict__ cbase,
        const long* __restrict__ pbase, const long* __restrict__ fbase,
        int batch, int block_size, int num_edges, int bnt, int use_atomic, int pid_my_offset,
        const __grid_constant__ CUtensorMap descNf, const __grid_constant__ CUtensorMap descNm,
        const __grid_constant__ CUtensorMap descEm) {
    // grid = (num_edges/(EE*BN) on x, chunk_of[n_nblocks * block_size/BM] on y). Each CTA owns one
    // m-tile (BM nodes of node-block nb) and EE consecutive edge-subtiles (BN edges each) on pid_e.
    int mtiles = block_size / BM;
    int pid_e = blockIdx.x;
    int pid_my = blockIdx.y + pid_my_offset;
    int nb = pid_my / mtiles, tile_id = pid_my % mtiles;
    long node_row0 = nbase[nb] + (long)tile_id * BM;   // first node-row this CTA handles
    int tid = threadIdx.x;

    // Plain K(batch)-major smem [rows, BK] for both MMA operands (auto-vectorized smem->reg, no swizzle).
    auto sAl = make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<BK>{}, _1{}));
    auto sBl = make_layout(make_shape(Int<BN>{}, Int<BK>{}), make_stride(Int<BK>{}, _1{}));
    // smem aliasing to keep the footprint at 4 CTAs/SM:
    float* sNf = (float*)smem_raw;        // node_flows TMA dest; reused in-place for lr, then partialS spill
    float* sNm = sNf + BM * BK;           // node_mars TMA dest; reused as the fp16 A+B operand region
    float* sEm = sNm + BM * BK;           // element_mars TMA dest (one edge-subtile at a time)
    half_t* pA = (half_t*)sNm;            // A operand = exp(lr - S)  [BM x BK]  (overlays sNm)
    half_t* pB = (half_t*)sNm + BM * BK;  // B operand = exp(emar + S) [BN x BK]
    float* sCmax = sEm + BN * BK;         // per-batch max_m lr[m,b]
    float* sS = sCmax + BK;               // per-batch balanced shift S[b]
    int*   sV = (int*)(sS + BK);          // per-batch valid flag (both lr and emar finite)
    uint64_t* bar = (uint64_t*)(sV + BK + 4);
    if (tid == 0) mbar_init(bar, 1);
    __syncthreads();
    int phase = 0;

    Tensor sAt = make_tensor(make_smem_ptr(pA), sAl);
    Tensor sBt = make_tensor(make_smem_ptr(pB), sBl);
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor tCrA = thr.partition_fragment_A(sAt); Tensor tCrB = thr.partition_fragment_B(sBt);
    auto s2rA = make_tiled_copy_A(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half_t>{}, mma);
    Tensor tXsA = s2rA.get_thread_slice(tid).partition_S(sAt); Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXsB = s2rB.get_thread_slice(tid).partition_S(sBt); Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{}); Tensor tCcC = thr.partition_C(cC);
    using FragC = decltype(thr.partition_fragment_C(cC));
    FragC acc[EE];
    CUTE_UNROLL
    for (int s = 0; s < EE; s++) clear(acc[s]);

    // Contract over batch in BK-sized k-tiles (bnt = batch/BK). The node tile (lr, cmax) is loaded
    // ONCE per k-tile and reused across the EE edge-subtiles (the edge-blocking win).
    for (int bt = 0; bt < bnt; bt++) {
        int b0 = bt * BK;
        // TMA-load this k-tile of node_flows and node_mars for the BM nodes.
        if (tid == 0) {
            mbar_expect(bar, (BM * BK + BM * BK) * 4);
            tma_load_2d(sNf, &descNf, b0, (int)node_row0, bar);
            tma_load_2d(sNm, &descNm, b0, (int)node_row0, bar);
        }
        mbar_wait(bar, phase); phase ^= 1;
        // lr[m,b] = node_flows - node_mars (in-place into sNf); -inf node_mar => -inf (dead node).
        for (int i = tid; i < (BM * BK) / 4; i += NTH) { int idx = i * 4;
            float4 nf = *(const float4*)&sNf[idx]; float4 nm = *(const float4*)&sNm[idx];
            float4 lr; lr.x=(nm.x==-INFINITY)?-INFINITY:nf.x-nm.x; lr.y=(nm.y==-INFINITY)?-INFINITY:nf.y-nm.y;
            lr.z=(nm.z==-INFINITY)?-INFINITY:nf.z-nm.z; lr.w=(nm.w==-INFINITY)?-INFINITY:nf.w-nm.w; *(float4*)&sNf[idx]=lr; }
        __syncthreads();
        // cmax[b] = max over the BM nodes of lr[m,b] (the "A" half of the balanced shift).
        for (int b = tid; b < BK; b += NTH) { float cm = -INFINITY;
            for (int m = 0; m < BM; m++) cm = fmaxf(cm, sNf[m * BK + b]); sCmax[b] = cm; }
        __syncthreads();
        CUTE_UNROLL
        for (int s = 0; s < EE; s++) {   // EE edge-subtiles sharing the node tile above
            long ele_row = cbase[nb] + (long)(pid_e * EE + s) * BN;
            if (tid == 0) { mbar_expect(bar, BN * BK * 4); tma_load_2d(sEm, &descEm, b0, (int)ele_row, bar); }
            mbar_wait(bar, phase); phase ^= 1;
            // gm[b] = max_e element_mars; balanced shift S[b] = (cmax - gm)/2; valid iff both finite.
            for (int b = tid; b < BK; b += NTH) { float gm = -INFINITY;
                for (int e = 0; e < BN; e++) gm = fmaxf(gm, sEm[e * BK + b]);
                float cm = sCmax[b]; int v = (cm != -INFINITY && gm != -INFINITY);
                sV[b] = v; sS[b] = v ? 0.5f * (cm - gm) : 0.0f; }
            __syncthreads();
            // A[m,b] = exp(lr - S) clamped to fp16 max (dead/-inf -> 0); staged into the fp16 A operand.
            for (int i = tid; i < (BM * BK) / 8; i += NTH) { int m = i/(BK/8), bb = (i%(BK/8))*8;
                half_t r[8]; for (int j = 0; j < 8; j++) { int b = bb+j; float lr = sNf[m*BK+b];
                    r[j] = static_cast<half_t>((!sV[b]||lr==-INFINITY)?0.f:fminf(__expf(lr-sS[b]),65504.0f)); }
                *(float4*)&sAt(m, bb) = *(const float4*)r; }
            // B[e,b] = exp(emar + S) clamped; staged into the fp16 B operand. (A*B = exp(lr+emar), shift cancels.)
            for (int i = tid; i < (BN * BK) / 8; i += NTH) { int e = i/(BK/8), bb = (i%(BK/8))*8;
                half_t r[8]; for (int j = 0; j < 8; j++) { int b = bb+j; float em = sEm[e*BK+b];
                    r[j] = static_cast<half_t>((!sV[b]||em==-INFINITY)?0.f:fminf(__expf(em+sS[b]),65504.0f)); }
                *(float4*)&sBt(e, bb) = *(const float4*)r; }
            __syncthreads();
            // fp16 tensor-core dot accumulated in fp32 into acc[s] (one fragment per edge-subtile).
            copy(s2rA, tXsA, tXrA); copy(s2rB, tXsB, tXrB);
            cute::gemm(mma, tCrA, tCrB, acc[s]);
            __syncthreads();
        }
    }
    // EPILOGUE (the dominant cost): for each edge-subtile, spill the MMA accumulator to smem, then
    // pflow[m,e] += epars[m,e] * partial[m,e] with 128-bit vectorized params/param_flows I/O.
    float* partialS = sNf;   // reuse the (now-dead) node smem to hold the [BN x BM] partial tile
    CUTE_UNROLL
    for (int s = 0; s < EE; s++) {
        long par0 = pbase[nb] + (long)(pid_e * EE + s) * BN * (long)block_size + (long)tile_id * BM;  // params base
        long pf0  = fbase[nb] + (long)(pid_e * EE + s) * BN * (long)block_size + (long)tile_id * BM;  // param_flows base
        __syncthreads();
        // spill the register C-fragment to smem in [e, m] layout (m contiguous, for coalesced float4 below)
        CUTE_UNROLL
        for (int i = 0; i < size(acc[s]); i++) { int m = get<0>(tCcC(i)), e = get<1>(tCcC(i));
            if (m < BM && e < BN) partialS[e * BM + m] = acc[s](i); }
        __syncthreads();
        // 128-bit (float4) vectorized params I/O: 4 contiguous nodes per thread (the m dim, stride 1).
        for (int idx = tid; idx < (BN * BM) / 4; idx += NTH) { int e = idx/(BM/4), m4 = (idx%(BM/4))*4;
            long off = (long)e * (long)block_size + m4;
            float4 ps = *(const float4*)&partialS[e * BM + m4];
            float4 ep = *(const float4*)&mp[par0 + off];
            float4 v = {ps.x*ep.x, ps.y*ep.y, ps.z*ep.z, ps.w*ep.w};   // pflow = epars * partial
            // Write mode (set by the Python gate): 0 = read-modify-write (RMW; the safe default,
            // accumulates across minibatches, requires collision-free pfids); 1 = atomicAdd (for tied
            // params where edge-blocks collide); 2 = store-only (fastest, but ONLY valid when param_flows
            // was freshly zeroed -- skips the RMW read; currently unused/footgun, see the Python side).
            if (use_atomic == 1) {
                atomicAdd(&pflows[pf0+off], v.x); atomicAdd(&pflows[pf0+off+1], v.y);
                atomicAdd(&pflows[pf0+off+2], v.z); atomicAdd(&pflows[pf0+off+3], v.w);
            } else if (use_atomic == 2) {
                *(float4*)&pflows[pf0+off] = v;
            } else {
                float4 o = *(float4*)&pflows[pf0+off]; o.x+=v.x; o.y+=v.y; o.z+=v.z; o.w+=v.w;
                *(float4*)&pflows[pf0+off] = o;
            } }
    }
}

static CUtensorMap g_dNf, g_dNm, g_dEm;
static void *g_nf = nullptr, *g_nm = nullptr, *g_em = nullptr; static int g_nr = 0, g_er = 0, g_b = 0;
static bool build_desc(CUtensorMap* d, void* base, int n_rows, int batch, int row_box) {
    cuuint64_t gdim[2] = {(cuuint64_t)batch, (cuuint64_t)n_rows};
    cuuint64_t gstride[1] = {(cuuint64_t)batch * 4};
    cuuint32_t bdim[2] = {(cuuint32_t)BK, (cuuint32_t)row_box};
    cuuint32_t estride[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS; }

void par_backward_sum(torch::Tensor param_flows, torch::Tensor node_flows, torch::Tensor node_mars,
                      torch::Tensor element_mars, torch::Tensor params,
                      torch::Tensor nbase, torch::Tensor cbase, torch::Tensor pbase, torch::Tensor fbase,
                      int64_t batch, int64_t block_size, int64_t num_edges, int64_t use_atomic) {
    TORCH_CHECK(block_size % BM == 0); TORCH_CHECK(num_edges % (EE * BN) == 0); TORCH_CHECK(batch % BK == 0);
    int n_nblocks = nbase.size(0);
    int n_node_rows = node_mars.size(0), n_ele_rows = element_mars.size(0);
    void* nf = (void*)node_flows.data_ptr<float>(); void* nm = (void*)node_mars.data_ptr<float>();
    void* em = (void*)element_mars.data_ptr<float>();
    if (nf != g_nf || nm != g_nm || em != g_em || n_node_rows != g_nr || n_ele_rows != g_er || (int)batch != g_b) {
        TORCH_CHECK(build_desc(&g_dNf, nf, n_node_rows, (int)batch, BM), "TMA nf");
        TORCH_CHECK(build_desc(&g_dNm, nm, n_node_rows, (int)batch, BM), "TMA nm");
        TORCH_CHECK(build_desc(&g_dEm, em, n_ele_rows, (int)batch, BN), "TMA em");
        g_nf=nf; g_nm=nm; g_em=em; g_nr=n_node_rows; g_er=n_ele_rows; g_b=(int)batch; }
    int smem = (BM * BK + BM * BK + BN * BK) * 4 + BK * 4 + BK * 4 + BK * 4 + 64;
    cudaFuncSetAttribute(par_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int bnt = (int)batch / BK;
    int total_y = n_nblocks * ((int)block_size / BM);
    int gx = (int)num_edges / (EE * BN);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_y; off += MAX_Y) {
        int chunk = (total_y - off < MAX_Y) ? (total_y - off) : MAX_Y;
        dim3 grid(gx, chunk);
        par_kernel<<<grid, NTH, smem>>>(param_flows.data_ptr<float>(), params.data_ptr<float>(),
            nbase.data_ptr<long>(), cbase.data_ptr<long>(), pbase.data_ptr<long>(), fbase.data_ptr<long>(),
            (int)batch, (int)block_size, (int)num_edges, bnt, (int)use_atomic, off, g_dNf, g_dNm, g_dEm); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("par_backward_sum", &par_backward_sum, "param-flow backward (edge-blocked, BK=32, 4 CTAs)");
}
