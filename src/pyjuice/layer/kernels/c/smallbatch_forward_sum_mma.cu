// Small-batch (batch < 64) block-sparse sum-layer FORWARD on tensor cores: a block = (node tile of BM, batch tile of BN,
// edge range of the node block's tiles).
//
// Its PW producer warps first take the range's maximum per column (the range's element rows read once, by all producer
// threads, whole rows per thread), then stage per edge tile of 64 the bf16 parameter tile (cp.async; fp32 parameters
// through registers, rounded to bf16 as the other CUDA forwards do) and the bf16 exponentials exp(x - M) they compute
// from the element rows (K-major, 8 consecutive edges of one column per 16-byte store); the WM x WN MMA warps run the
// bf16 tensor-core dot with fp32 accumulation over the range. A range writes its partial (L, M); `merge_kernel`, a
// thread per (node, column), combines the ranges -- launched as a PROGRAMMATIC DEPENDENT LAUNCH, so its own launch
// overlaps the primary's tail (MEASURED: 3-4 us of gap a layer -> ~1; the result is the same bits). One range writes
// node_mars itself. Columns past the batch are computed on -inf elements (terms 0) and never stored.
//
// Why: the small-batch FMA kernels (smallbatch_forward_sum*.cu) spend BT multiply-adds a parameter a lane and re-read
// the shared element rows once per 32-node group (MEASURED on an HMM-4096 layer at batch 16: 19 of 50 us), and the
// stream / pre-pass tensor-core kernels tile the batch by 64. Here the element tile is shared by a whole node tile and
// the arithmetic is the tensor cores'. MEASURED (RTX PRO 6000, HMM-4096 layer: 4 blocks of 1024 nodes, 4096 edges,
// L2-resident parameters; the FMA kernel / the Triton kernel in brackets), us: bf16 parameters batch 4 16.4 [27 / 115],
// 16 18.4 [53 / 75], 32 26.6 [103 / ~75], 63 ~40 [203]; fp32 parameters 4 25 [27], 16 27.6 [49 / 77], 32 34.8 [94].
// The bare parameter stream of this access pattern measures 13 us (bf16) / 19 (fp32) at 128 CTAs, 10 / 16 at 256;
// the exponentials and the MMA add ~4; the fp32 register conversion is the fp32 path's cost (`param_dtype = bf16`
// removes it).
//
// Numerically: bf16 operands, fp32 accumulation, a two-pass log-sum-exp per range (relative error ~2e-4 on the node
// values against float64; the FMA kernels ~1e-7) -- the class of the other tensor-core forwards.
//
// Contract (the small-batch forward's): a node block's children are CONTIGUOUS (child(e) = ebase[nb] + e), its
// parameters block_size-strided (params[pbase[nb] + e * block_size + m]), num_edges a multiple of 64; block_size a
// multiple of BM, or BM a multiple of block_size with every aligned run of BM / block_size node blocks sharing its
// children (`row_groups`; an HCLT's 32-node blocks) and the runs' parameter offsets within an int of each other; the
// caller verifies (`sb_ok`, `smallbatch_fw_mma_arms`' `grouped`) and keeps the other kernels otherwise.
//
// TRAPS met building it: a static `__shared__` variable shifts the dynamic shared memory off its 128-byte alignment and
// misaligns the swizzled stages (the bare stream 13 -> 17 us); the dynamic shared memory cap here is 99 KB (sm_120);
// ldmatrix for the K-major B operand is the 2-register atom (SM75_U32x2_LDSM_N); a last-CTA merge inside the kernel
// serializes a tail far longer than the merge launch it saves.

#include <torch/extension.h>
#include "param_load.cuh"
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <type_traits>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;
#define BK 64
#define MAX_SPLIT 16

__device__ __forceinline__ void bar_sync(int id, int count) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int count) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void cpa16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

extern __shared__ char smem_raw[];

template <int BM, int BN, int WM, int WN, int PW, int ST, typename TP>
__global__ void __launch_bounds__((WM * WN + PW) * 32) sbmma_kernel(
        float* __restrict__ node_mars, const float* __restrict__ element_mars, const TP* __restrict__ params,
        const long* __restrict__ nids, const long* __restrict__ ebase, const long* __restrict__ pbase,
        float* __restrict__ part, int batch, int block_size, int knt_all, int split, long rows_total, int bpad) {
    constexpr int NC = WM * WN * 32, NP = PW * 32, NTH = NC + NP;
    constexpr bool BF16 = std::is_same<TP, __nv_bfloat16>::value;
    static_assert(BN % 8 == 0 && BN <= 64 && 2 * ST + 1 <= 15, "columns; named barriers (15 is the producers' own)");
    // A node tile is BM rows of one node block (block_size >= BM: block_size / BM tiles a block) or, GROUPED, the
    // block_size-row blocks nb0 .. nb0 + BM / block_size - 1, which share their children (the caller verifies: an
    // HCLT's 32-node blocks of a variable) -- one exponential tile serves them all
    const bool grouped = block_size < BM;
    const int mtiles = grouped ? 1 : block_size / BM;
    const int pid_b = blockIdx.x, pid_m = blockIdx.y, sp = blockIdx.z;
    const int nb0 = grouped ? pid_m * (BM / block_size) : pid_m / mtiles;
    const int row_off = grouped ? 0 : (pid_m % mtiles) * BM;            // the tile's first row in its block
    const int b0 = pid_b * BN;
    const int k0 = (int)((long)sp * knt_all / split), k1 = (int)((long)(sp + 1) * knt_all / split);
    const int knt = k1 - k0;
    const long eb = ebase[nb0] + (long)k0 * BK;                                      // the range's first child
    const long pb_base = pbase[nb0] + (long)k0 * BK * block_size;                   // its first parameter (block nb0)
    const int tid = threadIdx.x;
    // tile row r -> its parameter offset from pb_base at edge 0 (the caller checks it fits an int), and its node
    auto row_delta = [&](int r) -> int {
        return grouped ? (int)(pbase[nb0 + r / block_size] - pbase[nb0]) + r % block_size : row_off + r;
    };
    auto row_node = [&](int r) -> long {
        return grouped ? nids[nb0 + r / block_size] + r % block_size : nids[nb0] + row_off + r;
    };

    // smem: A (BM, BK) MN-major swizzled (64 consecutive m of one k per 128-byte row; 32 for 32-row tiles); B (BN, BK)
    // K-major (64 consecutive k of one column per 128-byte row); then the range maxima per column and the producer
    // warps' partial maxima. No static shared memory: it would shift this buffer off its alignment.
    auto swz = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{});
    auto swz32 = composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{});
    auto sAl = tile_to_shape(std::conditional_t<(BM >= 64), decltype(swz), decltype(swz32)>{}, make_shape(Int<BM>{}, Int<BK>{}));
    auto swzK = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});
    auto sBl = tile_to_shape(swzK, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int STAGE = BM * BK * 2 + BN * BK * 2;
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    float* sM = (float*)(smem_raw + ST * STAGE);                        // [BN] the range maxima
    float* sMp = sM + BN;                                                // [PW x BN] the warps' partial maxima

    if (tid >= NC) {
        // ---------------- producers ----------------
        const int ptid = tid - NC;
        // The range maximum per column: a thread takes whole element ROWS (its BN columns, float4s where the row is
        // 16-byte aligned), lane-consecutive rows, so a warp's load is contiguous; the per-column maxima are reduced
        // across the lanes with shuffles and across the producer warps in smem. (A lane per column, strided down the
        // rows, was latency-bound: MEASURED 8 us of a 20 us kernel.)
        {
            const int ncols = min(BN, batch - b0);
            float mx[BN];
            CUTE_UNROLL
            for (int j = 0; j < BN; j++) mx[j] = -INFINITY;
            const float* x0 = element_mars + eb * (long)batch + b0;
            const bool vec = (ncols == BN) && ((batch & 3) == 0) && ((((size_t)x0) & 15) == 0);
            for (int e = ptid; e < knt * BK; e += NP) {
                const float* r = x0 + (long)e * batch;
                if (vec) {
                    CUTE_UNROLL
                    for (int j4 = 0; j4 < BN / 4; j4++) {
                        const float4 v = reinterpret_cast<const float4*>(r)[j4];
                        mx[j4 * 4 + 0] = fmaxf(mx[j4 * 4 + 0], v.x); mx[j4 * 4 + 1] = fmaxf(mx[j4 * 4 + 1], v.y);
                        mx[j4 * 4 + 2] = fmaxf(mx[j4 * 4 + 2], v.z); mx[j4 * 4 + 3] = fmaxf(mx[j4 * 4 + 3], v.w);
                    }
                } else {
                    CUTE_UNROLL
                    for (int j = 0; j < BN; j++) if (j < ncols) mx[j] = fmaxf(mx[j], r[j]);
                }
            }
            CUTE_UNROLL
            for (int j = 0; j < BN; j++) {
                float v = mx[j];
                CUTE_UNROLL
                for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
                mx[j] = v;
            }
            const int pw = ptid / 32, pl = ptid % 32;
            CUTE_UNROLL
            for (int j0 = 0; j0 < BN; j0 += 32) {                      // lane pl publishes columns pl, pl + 32, ...
                if (j0 + pl < BN) {
                    float v = -INFINITY;
                    CUTE_UNROLL
                    for (int j = 0; j < BN; j++) if (j == j0 + pl) v = mx[j];
                    sMp[pw * BN + j0 + pl] = v;
                }
            }
            bar_sync(15, NP);
            if (ptid < BN) {
                float m = -INFINITY;
                CUTE_UNROLL
                for (int w = 0; w < PW; w++) m = fmaxf(m, sMp[w * BN + ptid]);
                sM[ptid] = m;
            }
            bar_sync(15, NP);
        }
        constexpr int CH = (BN * BK) / 8;                                // exponential chunks (8 edges of a column) a tile
        constexpr int PBC = (CH + NP - 1) / NP;                          // ... per producer thread (some idle when CH < NP)
        float Mc[PBC];                                                   // this thread's columns' maxima
        CUTE_UNROLL
        for (int q = 0; q < PBC; q++) { const int i = ptid + q * NP; Mc[q] = (i < CH) ? sM[i / (BK / 8)] : 0.f; }
        // fp32 parameters: the tile's float4s loaded a tile ahead, converted at the issue (a second tile of lookahead
        // MEASURED slower: the registers)
        constexpr int PA4 = BF16 ? 1 : (BM * BK) / 4 / NP;
        static_assert(BF16 || (BM * BK) / 4 % NP == 0, "fp32 params float4s per producer thread");
        float4 ra[PA4];
        constexpr int PA = BF16 ? (BM * BK) / 8 / NP : 1;
        static_assert(!BF16 || (BM * BK) / 8 % NP == 0, "params chunks per producer thread");
        int rd[BF16 ? PA : PA4];                                         // each chunk's row offset (`row_delta`), once
        CUTE_UNROLL
        for (int q = 0; q < (BF16 ? PA : PA4); q++) {
            const int i = ptid + q * NP;
            rd[q] = row_delta(BF16 ? (i % (BM / 8)) * 8 : (i % (BM / 4)) * 4);
        }
        auto load_fp32 = [&](int t) {
            if constexpr (!BF16) {
                const TP* pc = params + pb_base + (long)t * BK * block_size;
                CUTE_UNROLL
                for (int q = 0; q < PA4; q++) {
                    const int i = ptid + q * NP, e = i / (BM / 4);
                    ra[q] = *(const float4*)&pc[(long)e * block_size + rd[q]];
                }
            }
        };
        // the element values of a tile's chunks, loaded a tile ahead
        float xv[PBC][8];
        auto load_x = [&](int t) {
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                if (i >= CH) continue;
                const bool live = b0 + c < batch;
                const float* x = element_mars + (eb + (long)t * BK + e8) * (long)batch + b0 + c;
                CUTE_UNROLL
                for (int j = 0; j < 8; j++) xv[q][j] = live ? x[(long)j * batch] : -INFINITY;
            }
        };
        if constexpr (!BF16) load_fp32(0);
        load_x(0);
        auto issue = [&](int t) {
            const int s = t % ST;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
            if constexpr (BF16) {
                const TP* pc = params + pb_base + (long)t * BK * block_size;
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) {
                    const int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                    cpa16(&sAt(mm, e), &pc[(long)e * block_size + rd[q]]);
                }
            } else {
                CUTE_UNROLL
                for (int q = 0; q < PA4; q++) {
                    const int i = ptid + q * NP, e = i / (BM / 4), mm = (i % (BM / 4)) * 4;
                    bfloat16_t r[4] = {static_cast<bfloat16_t>(ra[q].x), static_cast<bfloat16_t>(ra[q].y),
                                       static_cast<bfloat16_t>(ra[q].z), static_cast<bfloat16_t>(ra[q].w)};
                    *(uint2*)&sAt(mm, e) = *(const uint2*)r;
                }
                if (t + 1 < knt) load_fp32(t + 1);
            }
            // B[c, e] = bf16 exp(x - M[c]) for this thread's chunks: 8 consecutive edges of one column, one 16-byte store;
            // an all -inf (or tail) column's maximum is -inf and its terms exactly 0
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                if (i >= CH) continue;
                const float M = Mc[q];
                __align__(16) bfloat16_t u[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; j++) u[j] = static_cast<bfloat16_t>((M == -INFINITY) ? 0.f : __expf(xv[q][j] - M));
                *(uint4*)&sBt(c, e8) = *(const uint4*)u;
            }
            if (t + 1 < knt) load_x(t + 1);
        };
        // prologue: tiles 0 .. ST - 1, a commit group each (empty past the end), so group g is tile g throughout
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();                                          // groups <= j landed
            __threadfence_block();                                       // the B stage's plain stores, too
            bar_arrive(1 + j % ST, NTH);                                 // FULL: tile j staged
            if (j >= 1) {
                const int t = j - 1 + ST;                                // refill the stage tile j - 1 left
                if (t < knt) { bar_sync(1 + ST + (j - 1) % ST, NTH); issue(t); }
                cpa_commit();
            }
        }
        return;
    }

    // ---------------- consumers ----------------
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, mma);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);
    for (int kt = 0; kt < knt; kt++) {
        const int s = kt % ST;
        bar_sync(1 + s, NTH);                                            // FULL: tile kt staged
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt), tXrB);
        cute::gemm(mma, tCrA, tCrB, tCrL);                               // L[m, b] += sum_e params[m, e] * exp(x[e, b] - M[b])
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);                 // EMPTY: the stage is consumed (where refilled)
    }
    // the range maxima are final since before the first FULL
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        const int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        const float Mf = sM[b];
        if (split == 1) {
            if (b0 + b < batch)
                node_mars[row_node(m) * (long)batch + b0 + b] =
                    (Mf == -INFINITY || tCrL(i) <= 0.f) ? -INFINITY : (logf(tCrL(i)) + Mf);
        } else {
            // (row = node block x block_size + node: a grouped tile's blocks are consecutive)
            const long idx = ((long)sp * rows_total + (long)nb0 * block_size + row_off + m) * bpad + b0 + b;
            part[idx] = tCrL(i);
            part[(long)split * rows_total * bpad + idx] = Mf;
        }
    }
}

// ---------------- the ranges' merge (split > 1): a thread per (node, column) ----------------
// node_mars = log sum_z L_z exp(M_z - M) + M over the ranges, M their maximum (-inf where every M_z is). `part`: L
// [split, rows, bpad] then M [split, rows, bpad]; row = node block x block_size + node; bpad the batch rounded up to the
// tile. Launched as a programmatic dependent launch: it waits here for the primary grid's writes.
__global__ void merge_kernel(float* __restrict__ node_mars, const float* __restrict__ part, const long* __restrict__ nids,
                             int split, long rows_total, int block_size, int batch, int bpad) {
    cudaGridDependencySynchronize();
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_total * batch) return;
    const long row = i / batch, stride = rows_total * (long)bpad;
    const int b = (int)(i - row * batch);
    const float* Lp = part + row * bpad + b;
    const float* Mp = Lp + (long)split * stride;
    float M = -INFINITY;
    for (int z = 0; z < split; z++) M = fmaxf(M, Mp[(long)z * stride]);
    float r = -INFINITY;
    if (M != -INFINITY) {
        float L = 0.f;
        for (int z = 0; z < split; z++) {
            const float mz = Mp[(long)z * stride];
            if (mz != -INFINITY) L += Lp[(long)z * stride] * __expf(mz - M);
        }
        r = (L <= 0.f) ? -INFINITY : logf(L) + M;
    }
    node_mars[(nids[row / block_size] + row % block_size) * (long)batch + b] = r;
}

template <int BM, int BN, int WM, int WN, int PW, int ST, typename TP>
static void launch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params, torch::Tensor nids,
                       torch::Tensor ebase, torch::Tensor pbase, torch::Tensor scratch, int batch, int block_size, int knt,
                       int split) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr int smem = ST * (BM * BK * 2 + BN * BK * 2) + (BN + PW * BN) * 4;
    auto kernel = sbmma_kernel<BM, BN, WM, WN, PW, ST, TP>;
    // the shared-memory attribute once per device and instantiation
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_set[dev] = true;
    }
    const int nb = (int)nids.size(0);
    const int total_m = block_size >= BM ? nb * (block_size / BM) : nb / (BM / block_size);   // (grouped: whole groups)
    const long rows_total = (long)nb * block_size;
    const int btiles = (batch + BN - 1) / BN, bpad = btiles * BN;
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    float* part = nullptr;
    if (split > 1) {
        // the ranges' partials: the caller's cached buffer where it is large enough (an allocation per call is host
        // time on every launch of a small layer), else one allocation per call
        const long need = 2L * split * rows_total * bpad;
        if (scratch.numel() < need) scratch = torch::empty({need}, node_mars.options());
        part = scratch.data_ptr<float>();
    }
    dim3 grid(btiles, total_m, split);
    kernel<<<grid, NTH, smem, st>>>(node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), param_ptr<TP>(params),
                                    nids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(), part,
                                    batch, block_size, knt, split, rows_total, bpad);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (split > 1) {
        const long n = rows_total * batch;
        cudaLaunchConfig_t cfgl = {};
        cfgl.gridDim = dim3((unsigned)((n + 255) / 256)); cfgl.blockDim = dim3(256); cfgl.dynamicSmemBytes = 0; cfgl.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfgl.attrs = attr; cfgl.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, merge_kernel, node_mars.data_ptr<float>(), (const float*)part,
                                          (const long*)nids.data_ptr<long>(), split, rows_total, block_size, batch, bpad));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// cfg -> (BM, BN, WM, WN, PW, ST). 128-row tiles for layers of few node blocks (an HMM's 4 of 1024: 32 tiles x the split),
// 64-row ones for more CTAs; BN 16 / 32 / 64 by the batch (the pick measures). Within the 99 KB shared memory of sm_120.
#define SBMMA_CONFIGS(X) \
    X(0, 128, 16, 4, 1, 4, 3) X(1, 64, 16, 2, 1, 4, 4) X(2, 128, 32, 4, 1, 4, 3) X(3, 64, 32, 2, 1, 4, 4) \
    X(4, 128, 16, 4, 1, 8, 3) X(5, 128, 64, 4, 2, 4, 3) X(6, 64, 64, 2, 2, 4, 4)

// `cfg`: the tile config's id in the low byte, the split-K range count above it (0 or 1: one range). `scratch`: a float
// buffer for the ranges' partials (2 x split x rows x padded batch), reused when large enough; empty to allocate. The
// parameters in the dtype they are stored in (`TensorCircuit.param_dtype`): fp32 or bf16.
void smallbatch_forward_sum_mma(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params, torch::Tensor nids,
                                torch::Tensor ebase, torch::Tensor pbase, torch::Tensor scratch, int64_t batch,
                                int64_t block_size, int64_t num_edges, int64_t cfg) {
    const int split = cfg >> 8 ? (int)(cfg >> 8) : 1;
    cfg &= 255;
    TORCH_CHECK(num_edges % BK == 0, "smallbatch_forward_sum_mma: num_edges must be whole tiles of ", BK);
    const int knt = (int)(num_edges / BK);
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= knt, "smallbatch_forward_sum_mma: split 1 .. ", MAX_SPLIT,
                ", at most the ", knt, " edge tiles; got ", split);
    TORCH_CHECK(batch >= 1 && batch < 64, "smallbatch_forward_sum_mma: batch in [1, 64)");
    int b = (int)batch, bs = (int)block_size;
    auto run = [&](auto tag) {
        using TP = decltype(tag);
        switch (cfg) {
#define SBMMA_CASE(ID, BM_, BN_, WM_, WN_, PW_, ST_) case ID: \
            TORCH_CHECK(bs % BM_ == 0 || (bs % 8 == 0 && BM_ % bs == 0 && nids.size(0) % (BM_ / bs) == 0), \
                        "smallbatch_forward_sum_mma: block_size % BM, or BM % block_size over whole groups of blocks"); \
            launch_cfg<BM_, BN_, WM_, WN_, PW_, ST_, TP>(node_mars, element_mars, params, nids, ebase, pbase, scratch, b, bs, knt, split); break;
            SBMMA_CONFIGS(SBMMA_CASE)
#undef SBMMA_CASE
            // No silent default: an unknown cfg must not run zero kernels (see the autotune note in sum_layer.py)
            default: TORCH_CHECK(false, "smallbatch_forward_sum_mma: invalid cfg ", cfg);
        }
    };
    if (params.scalar_type() == torch::kBFloat16) run(__nv_bfloat16{}); else run(float{});
}

// (BM, BN) per config id; Python enumerates the configs valid for a layer shape.
std::vector<std::vector<int64_t>> smallbatch_fw_mma_configs() {
    std::vector<std::vector<int64_t>> out;
#define SBMMA_LIST(ID, BM_, BN_, WM_, WN_, PW_, ST_) out.push_back({BM_, BN_});
    SBMMA_CONFIGS(SBMMA_LIST)
#undef SBMMA_LIST
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallbatch_forward_sum_mma", &smallbatch_forward_sum_mma,
          "small-batch (batch<64) block-sparse sum-layer forward on tensor cores (CuTe), split-K ranges merged");
    m.def("smallbatch_fw_mma_configs", &smallbatch_fw_mma_configs, "(BM, BN) per config id");
}
