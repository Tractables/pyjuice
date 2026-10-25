// CUDA (CuTe/CUTLASS) block-sparse sum-layer ELEMENT-FLOW backward in two launches: a PRE-PASS over the parents' node
// tiles and a STREAMING kernel over the parameter tiles. The same computation as ele_backward_sum.cu (and
// ele_backward_sum_pipelined.cu), to the bit -- the element-flow copy of tlmm_forward_sum_prepass.cu.
//
// Why: per element block, the flow is
//     element_flows[m, b] = element_mars[m, b] + logsumexp over the parent tiles of log(sum_e params[m, e] * nfsub[e, b])
// with nfsub[e, b] = fp16 exp(lf[e, b] - max_e lf[e, b]) and lf = node_flows - node_mars of the tile's 64 parents --
// which depends on the parent tile and the batch column only, never on the element m, yet the other kernels computed it
// (the log ratio, the tile's max, the exponentials) once per element tile (8x over at blocks of 1024, tile 128), and the
// online log-sum-exp's merge factors once per accumulator. Here the pre-pass computes the max and the fp16 exponentials
// once per (element block, parent tile, batch column), their rows e-contiguous as the B operand reads them; the
// streaming kernel's producer warps stage the fp16 parameter tile (from fp32 or bf16 through registers, one issue
// ahead: either rounds to fp16, as the other kernels) and cp.async the pre-pass's tile into 3-4 stages, one of them
// computing the merge factors per column; the consumer warps do the MMA and L = L * alpha + S * beta (fmul, fmul,
// fadd: the other kernels' expression, uncontracted) only. element_flows = element_mars + log(L) + M (-inf where M is).
//
// The smem atoms: the parameters' (written from registers) the other kernels' K-major one; the exponentials'
// (cp.async) 64 consecutive parents of one column per 128-byte row -- the layout that keeps a global row in one smem row
// (see tlmm_forward_sum_prepass.cu: the other one halves a cp.async stream's rate).
//
// MEASURED (RTX PRO 6000, one partition, against the pipelined kernel; bf16 parameters unless said): HMM-1024 (one
// element block of 1024, 16 parent tiles) B 64 46.4 -> 22.5 us; HMM-4096 (4 blocks, 64 tiles) 152 -> 57.2; HCLT-1024's
// largest (143 blocks of 1024, 16 tiles) B 64 375 -> 357, B 128 570 -> 541, fp32 B 128 699 -> 685. On the large layers
// the pre-pass is the reading of the parents' flows and values (150 MB at B 128: 131 us) the other kernels overlap
// with their parameter stream, and the stream kernel itself is at 73% of its floor at B 128 (two 64-wide batch tiles
// each pulling the parameters through L2), 87% at B 64.
//
// Scratch: the exponentials (fp16) and maxima, one allocation per call from the caching allocator (inside a CUDA graph
// capture, from the graph's pool), freed when it returns.
//
// SPECIALIZED fast path under the plain kernel's conditions (LL, logspace flows, allow_modify_flows / allow_neg_flows /
// accumulate_ch_flows off, no tempering, the fp16 dot regime, TILE_SIZE_K == 64, block_size % 128 == 0,
// batch % 64 == 0, contiguous edge/param layout); the Python dispatcher measures it against the other kernels per shape.

#include <torch/extension.h>
#include "param_load.cuh"
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64
#define PB 64

__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void cpa16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ---------------- pre-pass: a thread block per (element block x parent tile, PB batch columns), 256 threads ----------
__global__ void __launch_bounds__(256) ele_prepass_kernel(const float* __restrict__ nflows, const float* __restrict__ nmars,
                                                         const long* __restrict__ ebase, half_t* __restrict__ bx,
                                                         float* __restrict__ mx, int batch) {
    __shared__ float sLf[BK][PB + 1];
    __shared__ float sMx4[4][PB];
    __shared__ float sMx[PB];
    long tile = blockIdx.x;
    int b0 = blockIdx.y * PB;
    int tid = threadIdx.x;
    long e0 = ebase[tile];
    // the log ratio tile, coalesced along the batch: thread (part, b) rows part * 16 ..
    {
        int b = tid % PB, part = tid / PB;
        float m = -INFINITY;
        CUTE_UNROLL
        for (int j = 0; j < BK / 4; j++) {
            int e = part * (BK / 4) + j;
            long off = (e0 + e) * (long)batch + b0 + b;
            float nf = nflows[off], nm = nmars[off];
            float lf = (nm == -INFINITY) ? -INFINITY : nf - nm;
            sLf[e][b] = lf;
            m = fmaxf(m, lf);
        }
        sMx4[part][b] = m;
    }
    __syncthreads();
    if (tid < PB) {
        float m = sMx4[0][tid];
        CUTE_UNROLL
        for (int q = 1; q < 4; q++) m = fmaxf(m, sMx4[q][tid]);
        sMx[tid] = m;
        mx[tile * (long)batch + b0 + tid] = m;
    }
    __syncthreads();
    // bx[tile][b0 + b][e .. e + 7]: 8 consecutive parents a 16-byte store
    for (int i = tid; i < PB * BK / 8; i += 256) {
        int b = i / (BK / 8), ee = (i % (BK / 8)) * 8;
        float m = sMx[b];
        half_t r[8];
        CUTE_UNROLL
        for (int j = 0; j < 8; j++) {
            float lf = sLf[ee + j][b];
            r[j] = static_cast<half_t>((m == -INFINITY || lf == -INFINITY) ? 0.f : __expf(lf - m));
        }
        *(float4*)&bx[((tile * batch) + b0 + b) * (long)BK + ee] = *(const float4*)r;
    }
}

extern __shared__ char smem_raw[];

template <int BM, int BN, int WM, int WN, int PW, int ST, typename TP>
__global__ void __launch_bounds__((WM * WN + PW) * 32) ele_stream_kernel(
        float* __restrict__ eflows, const float* __restrict__ emars, const TP* __restrict__ mp,
        const long* __restrict__ chids, const long* __restrict__ pbase, const half_t* __restrict__ bx,
        const float* __restrict__ mx, int batch, int BSK, int BSM, int knt, int pid_m_offset) {
    constexpr int NC = WM * WN * 32;
    constexpr int NP = PW * 32;
    constexpr int NTH = NC + NP;
    int mtiles = BSM / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int eb = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_ele = chids[eb] + (long)tile_id * BM;
    const long* pbp = pbase + (long)eb * knt;
    int tid = threadIdx.x;

    // A: the plain kernels' K-major atom (written from registers); B: 64 consecutive parents of one column per 128-byte
    // row (cp.async from the pre-pass's rows: a global row in one smem row)
    auto sAl = tile_to_shape(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}),
                             make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}),
                             make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int STAGE = BM * BK * 2 + BN * BK * 2 + 3 * BN * 4;
    auto stA = [&](int s) { return (half_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (half_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    auto stF = [&](int s) { return (float*)(smem_raw + s * STAGE + BM * BK * 2 + BN * BK * 2); };

    if (tid >= NC) {
        int ptid = tid - NC;
        const long tile0 = (long)eb * knt;
        float Mrun[BN / 32], mnext[BN / 32];
        CUTE_UNROLL
        for (int j = 0; j < BN / 32; j++) Mrun[j] = -INFINITY;
        if (ptid < 32) {
            CUTE_UNROLL
            for (int j = 0; j < BN / 32; j++) mnext[j] = mx[tile0 * batch + b0 + ptid + 32 * j];
        }
        long pn1 = pbp[0], pn2 = knt > 1 ? pbp[1] : 0;
        constexpr int PA = (BM * BK) / 8 / NP;            // 8-parameter chunks per producer thread
        static_assert((BM * BK) / 8 % NP == 0, "params chunks per producer thread");
        float ra[PA][8];
        auto loadA = [&](long pcv) {
            CUTE_UNROLL
            for (int q = 0; q < PA; q++) {
                int i = ptid + q * NP;
                int m = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                ParamLoad<TP>::load8(&mp[pcv + (long)m * BSK + ee], ra[q]);
            }
        };
        loadA(pn1 + (long)tile_id * BM * BSK);
        auto issue = [&](int t) {
            int s = t % ST;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
            pn1 = pn2;
            if (t + 2 < knt) pn2 = pbp[t + 2];
            constexpr int PBC = (BN * BK) / 8 / NP;
            static_assert((BN * BK) / 8 % NP == 0, "exponential chunks per producer thread");
            const half_t* bt = bx + ((tile0 + t) * batch + b0) * (long)BK;
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                int i = ptid + q * NP;
                int b = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                cpa16(&sBt(b, ee), &bt[(long)b * BK + ee]);
            }
            CUTE_UNROLL
            for (int q = 0; q < PA; q++) {
                int i = ptid + q * NP;
                int m = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                half_t r[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; j++) r[j] = static_cast<half_t>(ra[q][j]);
                *(float4*)&sAt(m, ee) = *(const float4*)r;
            }
            if (t + 1 < knt) loadA(pn1 + (long)tile_id * BM * BSK);
            if (ptid < 32) {
                float* f = stF(s);
                CUTE_UNROLL
                for (int j = 0; j < BN / 32; j++) {
                    float pfm = mnext[j], Mo = Mrun[j];
                    float nM = fmaxf(Mo, pfm);
                    float al, be;
                    if (nM != -INFINITY) {
                        al = (Mo == -INFINITY) ? 0.f : __expf(Mo - nM);
                        be = (pfm == -INFINITY) ? 0.f : __expf(pfm - nM);
                    } else { al = 1.f; be = 0.f; }
                    Mrun[j] = nM;
                    f[ptid + 32 * j] = al; f[BN + ptid + 32 * j] = be; f[2 * BN + ptid + 32 * j] = nM;
                }
                if (t + 1 < knt) {
                    CUTE_UNROLL
                    for (int j = 0; j < BN / 32; j++) mnext[j] = mx[(tile0 + t + 1) * batch + b0 + ptid + 32 * j];
                }
            }
        };
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();
            bar_arrive(1 + j % ST, NTH);
            if (j >= 1) {
                int t = j - 1 + ST;
                if (t < knt) { bar_sync(1 + ST + (j - 1) % ST, NTH); issue(t); }
                cpa_commit();
            }
        }
        return;
    }

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, half_t>{}, mma);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);
    float al[size(tCrS)], be[size(tCrS)], Mf[size(tCrS)];

    for (int kt = 0; kt < knt; kt++) {
        int s = kt % ST;
        bar_sync(1 + s, NTH);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt), tXrB);
        const float* f = stF(s);
        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) {
            int b = get<1>(tCcC(i));
            al[i] = f[b]; be[i] = f[BN + b];
            if (kt == knt - 1) Mf[i] = f[2 * BN + b];
        }
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);
        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++)
            tCrL(i) = __fadd_rn(__fmul_rn(tCrL(i), al[i]), __fmul_rn(tCrS(i), be[i]));
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        float r = (Mf[i] == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + Mf[i]);
        long off = (off_ele + m) * (long)batch + b0 + b;
        if (r != -INFINITY) r += emars[off];
        eflows[off] = r;
    }
}

template <int BM, int BN, int WM, int WN, int PW, int ST, typename TP>
static void launch_cfg(float* ef, const float* em, const TP* p, const long* chids, int neb, const long* pb,
                       const half_t* bx, const float* mx, int batch, int BSK, int BSM, int knt, cudaStream_t st) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr int smem = ST * (BM * BK * 2 + BN * BK * 2 + 3 * BN * 4);
    auto kernel = ele_stream_kernel<BM, BN, WM, WN, PW, ST, TP>;
    // the shared-memory attribute once per device and config (its host time counts on small layers); this
    // instantiation's own flags
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_set[dev] = true;
    }
    int total_m = neb * (BSM / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        kernel<<<grid, NTH, smem, st>>>(ef, em, p, chids, pb, bx, mx, batch, BSK, BSM, knt, off);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Config table (id -> {BM, BN, WM, WN, PW, ST}): 8 MMA warps on a (128, 64) tile, 3 stages, with 4 or 8 producer warps
// (MEASURED on HCLT-1024's largest partition at B 128: the stream 433 / 421 us, against 627 with 16 MMA warps); the
// (64, 64) tile on 4 stages for few element blocks (HMM-1024 B 64: 22.5 us against 27.5 for the best (128, 64)).
template <typename TP>
static void dispatch_cfg(float* ef, const float* em, const TP* p, const long* chids, int neb, const long* pb,
                         const half_t* bx, const float* mx, int b, int bsk, int bsm, int k, int64_t cfg, cudaStream_t st) {
    switch (cfg) {
        case 0: launch_cfg<128, 64, 4, 2, 4, 3, TP>(ef, em, p, chids, neb, pb, bx, mx, b, bsk, bsm, k, st); break;
        case 1: launch_cfg<128, 64, 4, 2, 8, 3, TP>(ef, em, p, chids, neb, pb, bx, mx, b, bsk, bsm, k, st); break;
        default: launch_cfg<64, 64, 4, 2, 4, 4, TP>(ef, em, p, chids, neb, pb, bx, mx, b, bsk, bsm, k, st); break;
    }
}

// element-flow backward (writes element_flows in place). chids / ebase / pbase are int64 [neb] / [neb, knt]; the parameters
// in the dtype they are stored in (fp32 or bf16).
void ele_backward_sum_prepass(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                              torch::Tensor node_mars, torch::Tensor params, torch::Tensor chids, torch::Tensor ebase,
                              torch::Tensor pbase, int64_t batch, int64_t BSK, int64_t BSM, int64_t knt, int64_t cfg) {
    TORCH_CHECK(batch % PB == 0, "the pre-pass element backward needs a batch that is a multiple of ", PB);
    TORCH_CHECK(ebase.is_contiguous() && ebase.numel() == chids.numel() * knt, "ebase: [element blocks, parent tiles]");
    long tiles = ebase.numel();
    long bx_bytes = tiles * BK * batch * 2;
    torch::Tensor scratch = torch::empty({bx_bytes + tiles * batch * 4}, element_mars.options().dtype(torch::kUInt8));
    auto* bx = reinterpret_cast<half_t*>(scratch.data_ptr());
    auto* mx = reinterpret_cast<float*>(reinterpret_cast<char*>(scratch.data_ptr()) + bx_bytes);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    dim3 grid(tiles, batch / PB);
    ele_prepass_kernel<<<grid, 256, 0, st>>>(node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
                                             ebase.data_ptr<long>(), bx, mx, (int)batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int neb = (int)chids.size(0);
    if (params.scalar_type() == torch::kBFloat16)
        dispatch_cfg<__nv_bfloat16>(element_flows.data_ptr<float>(), element_mars.data_ptr<float>(),
                                    param_ptr<__nv_bfloat16>(params), chids.data_ptr<long>(), neb, pbase.data_ptr<long>(),
                                    bx, mx, (int)batch, (int)BSK, (int)BSM, (int)knt, cfg, st);
    else
        dispatch_cfg<float>(element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), param_ptr<float>(params),
                            chids.data_ptr<long>(), neb, pbase.data_ptr<long>(), bx, mx, (int)batch, (int)BSK, (int)BSM,
                            (int)knt, cfg, st);
}

// (BM, BN) per config id; Python uses this to enumerate the configs valid for a partition.
std::vector<std::vector<int64_t>> configs() {
    return {{128, 64}, {128, 64}, {64, 64}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ele_backward_sum_prepass", &ele_backward_sum_prepass,
          "block-sparse sum-layer element-flow backward: exponentials pre-pass + streaming kernel (CuTe/fp16)");
    m.def("configs", &configs, "list of (BM, BN) per config id");
}
