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
#define MAX_SPLIT 8     // split-K ranges of an element block's parent tiles at most

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

// TIX: the pre-pass tile of (element block, parent tile) is `tix[...]` (element blocks sharing their parents share
// tiles); false where it is the entry's own index (the table is never read -- MEASURED the read, prefetched a tile
// ahead, cost the fp32 kernel 13%)
template <int BM, int BN, int WM, int WN, int PW, int ST, int PBT, bool TIX, typename TP>
__global__ void __launch_bounds__((WM * WN + PW) * 32) ele_stream_kernel(
        float* __restrict__ eflows, const float* __restrict__ emars, const TP* __restrict__ mp,
        const long* __restrict__ chids, const long* __restrict__ pbase, const half_t* __restrict__ bx,
        const float* __restrict__ mx, const int* __restrict__ tix, float* __restrict__ part, int batch, int BSK,
        int BSM, int knt, int split, long rows_total, int pid_m_offset) {
    constexpr int NC = WM * WN * 32;
    constexpr int NP = PW * 32;
    constexpr int NTH = NC + NP;
    int mtiles = BSM / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int eb = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_ele = chids[eb] + (long)tile_id * BM;
    // split-K: this CTA walks the parent tiles k0 .. k1 of its element block (blockIdx.z of `split` ranges; one range
    // is the whole row), everything below over `knt` = the range's tiles; a range's partial sum and maximum go to
    // `part` for `merge_kernel`, a single range writes the flows itself
    const int sp = blockIdx.z, knt_all = knt;
    const int k0 = (int)((long)sp * knt_all / split), k1 = (int)((long)(sp + 1) * knt_all / split);
    knt = k1 - k0;
    // a parent tile's parameters: PBT blocks of BSK parents each (one where BSK >= BK; two where the parents' block
    // size is 32: the halves of a tile lie in different parameter blocks -- a template parameter: the one-block kernel
    // is the kernel it was, MEASURED a runtime count cost it a third)
    const long* pbp = pbase + ((long)eb * knt_all + k0) * PBT;
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
        const long tile0 = (long)eb * knt_all + k0;
        float Mrun[BN / 32], mnext[BN / 32];
        CUTE_UNROLL
        for (int j = 0; j < BN / 32; j++) Mrun[j] = -INFINITY;
        // the pre-pass tiles of tiles t (tx1) and t + 1 (tx2), loaded a tile ahead (a load that waited on them a tile
        // would sit on the producers' path)
        int tx1 = TIX ? tix[tile0] : (int)tile0, tx2 = TIX ? (knt > 1 ? tix[tile0 + 1] : 0) : (int)tile0 + 1;
        if (ptid < 32) {
            CUTE_UNROLL
            for (int j = 0; j < BN / 32; j++) mnext[j] = mx[(long)tx1 * batch + b0 + ptid + 32 * j];
        }
        // the parameter blocks of tiles t (pn1a, and pn1b: the second block of a two-block tile) and t + 1 (pn2a / b),
        // the next tile's fp32/bf16 loads one issue ahead -- scalars, not an array the lambda takes by address (MEASURED
        // that at fp32: local memory, +18% on the kernel)
        long pn1a = pbp[0], pn1b = PBT > 1 ? pbp[1] : 0;
        long pn2a = knt > 1 ? pbp[PBT] : 0, pn2b = (PBT > 1 && knt > 1) ? pbp[PBT + 1] : 0;
        constexpr int PA = (BM * BK) / 8 / NP;            // 8-parameter chunks per producer thread
        static_assert((BM * BK) / 8 % NP == 0, "params chunks per producer thread");
        float ra[PA][8];
        auto loadA = [&](long pa, long pb) {
            CUTE_UNROLL
            for (int q = 0; q < PA; q++) {
                int i = ptid + q * NP;
                int m = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                int h = PBT > 1 ? ee / (BK / PBT) : 0, e = PBT > 1 ? ee % (BK / PBT) : ee;
                ParamLoad<TP>::load8(&mp[(h ? pb : pa) + (long)tile_id * BM * BSK + (long)m * BSK + e], ra[q]);
            }
        };
        loadA(pn1a, pn1b);
        auto issue = [&](int t) {
            int s = t % ST;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
            const int tcur = tx1;
            tx1 = tx2;
            if (t + 2 < knt) tx2 = TIX ? tix[tile0 + t + 2] : (int)tile0 + t + 2;
            pn1a = pn2a; pn1b = pn2b;
            if (t + 2 < knt) { pn2a = pbp[(t + 2) * PBT]; if (PBT > 1) pn2b = pbp[(t + 2) * PBT + 1]; }
            constexpr int PBC = (BN * BK) / 8 / NP;
            static_assert((BN * BK) / 8 % NP == 0, "exponential chunks per producer thread");
            const half_t* bt = bx + ((long)tcur * batch + b0) * (long)BK;
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
            if (t + 1 < knt) loadA(pn1a, pn1b);
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
                    for (int j = 0; j < BN / 32; j++) mnext[j] = mx[(long)tx1 * batch + b0 + ptid + 32 * j];
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

    for (int kt = 0; kt < knt; kt++) {
        int s = kt % ST;
        bar_sync(1 + s, NTH);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt), tXrB);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);
        // the merge factors read from the stage as they are used (per column: a register copy per accumulator element
        // held the kernel to one thread block an SM -- MEASURED 102-108 registers at 384 threads), then the stage
        // released
        const float* f = stF(s);
        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) {
            int b = get<1>(tCcC(i));
            tCrL(i) = __fadd_rn(__fmul_rn(tCrL(i), f[b]), __fmul_rn(tCrS(i), f[BN + b]));
        }
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);
    }
    const float* fM = stF((knt - 1) % ST) + 2 * BN;  // the last tile's stage: the running maxima, never refilled
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        const float Mf = fM[b];
        if (split == 1) {
            float r = (Mf == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + Mf);
            long off = (off_ele + m) * (long)batch + b0 + b;
            if (r != -INFINITY) r += emars[off];
            eflows[off] = r;
        } else {
            const long idx = ((long)sp * rows_total + (long)eb * BSM + (long)tile_id * BM + m) * batch + b0 + b;
            part[idx] = tCrL(i);
            part[(long)split * rows_total * batch + idx] = Mf;
        }
    }
}

// ---------------- the ranges' merge (split > 1): a thread per (element, column) ----------------
// element_flows = emars + log sum_z L_z exp(M_z - M) + M over the ranges, M their maximum (-inf where every M_z is),
// the sum in range order. `part`: L [split, rows, batch] then M [split, rows, batch]; row = element block x BSM + m.
__global__ void merge_kernel(float* __restrict__ eflows, const float* __restrict__ emars, const float* __restrict__ part,
                             const long* __restrict__ chids, int split, long rows_total, int BSM, int batch) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_total * batch) return;
    const long row = i / batch, stride = rows_total * (long)batch;
    const int b = (int)(i - row * batch);
    const float* Lp = part + row * batch + b;
    const float* Mp = Lp + (long)split * stride;
    float M = -INFINITY;
    for (int z = 0; z < split; z++) M = fmaxf(M, Mp[(long)z * stride]);
    const long off = (chids[row / BSM] + row % BSM) * (long)batch + b;
    float r = -INFINITY;
    if (M != -INFINITY) {
        float L = 0.f;
        for (int z = 0; z < split; z++) {
            const float mz = Mp[(long)z * stride];
            if (mz != -INFINITY) L += Lp[(long)z * stride] * __expf(mz - M);
        }
        r = logf(L) + M + emars[off];
    }
    eflows[off] = r;
}

template <int BM, int BN, int WM, int WN, int PW, int ST, int PBT, bool TIX, typename TP>
static void launch_cfg(float* ef, const float* em, const TP* p, const long* chids, int neb, const long* pb,
                       const half_t* bx, const float* mx, const int* tix, float* part, int split, int batch, int BSK,
                       int BSM, int knt, cudaStream_t st) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr int smem = ST * (BM * BK * 2 + BN * BK * 2 + 3 * BN * 4);
    auto kernel = ele_stream_kernel<BM, BN, WM, WN, PW, ST, PBT, TIX, TP>;
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
    const long rows_total = (long)neb * BSM;
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk, split);
        kernel<<<grid, NTH, smem, st>>>(ef, em, p, chids, pb, bx, mx, tix, part, batch, BSK, BSM, knt, split, rows_total,
                                        off);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (split > 1) {
        const long n = rows_total * batch;
        merge_kernel<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(ef, em, part, chids, split, rows_total, BSM, batch);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Config table (id -> {BM, BN, WM, WN, PW, ST}): 8 MMA warps on a (128, 64) tile, 3 stages, with 4 or 8 producer warps
// (MEASURED on HCLT-1024's largest partition at B 128: the stream 433 / 421 us, against 627 with 16 MMA warps); the
// (64, 64) tile on 4 stages for few element blocks (HMM-1024 B 64: 22.5 us against 27.5 for the best (128, 64)); the
// (32, 64) and (32, 128) tiles, 2 x 2 and 2 x 4 MMA warps, for partitions of 32-node child blocks (an HCLT's), which no
// other CUDA kernel takes.
template <int PBT, bool TIX, typename TP>
static void dispatch_cfg(float* ef, const float* em, const TP* p, const long* chids, int neb, const long* pb,
                         const half_t* bx, const float* mx, const int* tix, float* part, int split, int b, int bsk,
                         int bsm, int k, int64_t cfg, cudaStream_t st) {
    switch (cfg) {
        case 0: launch_cfg<128, 64, 4, 2, 4, 3, PBT, TIX, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, st); break;
        case 1: launch_cfg<128, 64, 4, 2, 8, 3, PBT, TIX, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, st); break;
        case 2: launch_cfg<64, 64, 4, 2, 4, 4, PBT, TIX, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, st); break;
        case 3: launch_cfg<32, 64, 2, 2, 4, 4, PBT, TIX, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, st); break;
        default: launch_cfg<32, 128, 2, 4, 4, 3, PBT, TIX, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, st); break;
    }
}

template <typename TP>
static void dispatch_pbt(float* ef, const float* em, const TP* p, const long* chids, int neb, const long* pb,
                         const half_t* bx, const float* mx, const int* tix, bool shared, float* part, int split, int b,
                         int bsk, int bsm, int k, int pbt, int64_t cfg, cudaStream_t st) {
    if (pbt == 1 && !shared) dispatch_cfg<1, false, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, cfg, st);
    else if (pbt == 1) dispatch_cfg<1, true, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, cfg, st);
    else if (!shared) dispatch_cfg<2, false, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, cfg, st);
    else dispatch_cfg<2, true, TP>(ef, em, p, chids, neb, pb, bx, mx, tix, part, split, b, bsk, bsm, k, cfg, st);
}

// element-flow backward (writes element_flows in place). chids is int64 [neb]; pbase int64 [neb, knt * pbt]: a parent
// tile's first parameter offset per parameter block it spans (pbt = BK / BSK blocks where the parents' block size BSK is
// below the tile's BK, else one); `ebase` the DISTINCT parent tiles' first rows (a tile's parents contiguous node rows,
// whatever their blocks), `tix` (int32 [neb, knt]) which of them each element block's parent tile is (element blocks
// sharing their parents -- an HCLT's 32 blocks of a variable -- share the pre-pass); the parameters in the dtype they are
// stored in (fp32 or bf16).
void ele_backward_sum_prepass(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                              torch::Tensor node_mars, torch::Tensor params, torch::Tensor chids, torch::Tensor ebase,
                              torch::Tensor tix, torch::Tensor pbase, int64_t batch, int64_t BSK, int64_t BSM, int64_t knt,
                              int64_t cfg, bool shared) {
    TORCH_CHECK(batch % PB == 0, "the pre-pass element backward needs a batch that is a multiple of ", PB);
    TORCH_CHECK(ebase.is_contiguous() && ebase.dim() == 1, "ebase: [distinct parent tiles]");
    TORCH_CHECK(tix.is_contiguous() && tix.scalar_type() == torch::kInt32 && tix.numel() == chids.numel() * knt,
                "tix: int32 [element blocks, parent tiles]");
    const int pbt = BSK >= BK ? 1 : (int)(BK / BSK);
    TORCH_CHECK((BSK >= BK ? BSK % BK == 0 : BK % BSK == 0) && pbt <= 2 && pbase.is_contiguous()
                && pbase.numel() == chids.numel() * knt * pbt, "pbase: [element blocks, parent tiles x parameter blocks]");
    // `shared`: the blocks share tiles, `tix` read; else `ebase` is every entry's own tile in order and `tix` is
    // unread (a caller's word, not inferred from the sizes: a launch over some element blocks hands a slice of `tix`)
    long tiles = ebase.numel();
    TORCH_CHECK(shared || tiles == tix.numel(), "unshared tiles: one per (element block, parent tile)");
    // `cfg`: the tile config's id in the low byte, the split-K range count above it (0 or 1: one range); the ranges'
    // partials (L and M, [split, rows, batch] each) follow the pre-pass's scratch in the one allocation
    const int split = cfg >> 8 ? (int)(cfg >> 8) : 1;
    cfg &= 255;
    TORCH_CHECK(split <= MAX_SPLIT && split <= knt, "the pre-pass element backward's split: 1 .. ", MAX_SPLIT,
                ", at most the ", knt, " parent tiles; got ", split);
    const long rows_total = chids.numel() * BSM;
    long bx_bytes = tiles * BK * batch * 2, mx_bytes = tiles * batch * 4;
    long part_bytes = split > 1 ? 2L * split * rows_total * batch * 4 : 0;
    torch::Tensor scratch = torch::empty({bx_bytes + mx_bytes + part_bytes}, element_mars.options().dtype(torch::kUInt8));
    auto* bx = reinterpret_cast<half_t*>(scratch.data_ptr());
    auto* mx = reinterpret_cast<float*>(reinterpret_cast<char*>(scratch.data_ptr()) + bx_bytes);
    auto* part = reinterpret_cast<float*>(reinterpret_cast<char*>(scratch.data_ptr()) + bx_bytes + mx_bytes);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    dim3 grid(tiles, batch / PB);
    ele_prepass_kernel<<<grid, 256, 0, st>>>(node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
                                             ebase.data_ptr<long>(), bx, mx, (int)batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int neb = (int)chids.size(0);
    if (params.scalar_type() == torch::kBFloat16)
        dispatch_pbt<__nv_bfloat16>(element_flows.data_ptr<float>(), element_mars.data_ptr<float>(),
                                    param_ptr<__nv_bfloat16>(params), chids.data_ptr<long>(), neb, pbase.data_ptr<long>(),
                                    bx, mx, tix.data_ptr<int>(), shared, part, split, (int)batch, (int)BSK, (int)BSM,
                                    (int)knt, pbt, cfg, st);
    else
        dispatch_pbt<float>(element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), param_ptr<float>(params),
                            chids.data_ptr<long>(), neb, pbase.data_ptr<long>(), bx, mx, tix.data_ptr<int>(), shared,
                            part, split, (int)batch, (int)BSK, (int)BSM, (int)knt, pbt, cfg, st);
}

// (BM, BN) per config id; Python uses this to enumerate the configs valid for a partition.
std::vector<std::vector<int64_t>> configs() {
    return {{128, 64}, {128, 64}, {64, 64}, {32, 64}, {32, 128}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ele_backward_sum_prepass", &ele_backward_sum_prepass,
          "block-sparse sum-layer element-flow backward: exponentials pre-pass + streaming kernel (CuTe/fp16)");
    m.def("configs", &configs, "list of (BM, BN) per config id");
}
