// CUDA (CuTe/CUTLASS) block-sparse sum-layer forward "tlmm" kernel in two launches: a PRE-PASS over the element tiles
// and a STREAMING kernel over the parameter tiles. The same computation as tlmm_forward_sum_pipelined.cu (and so as
// tlmm_forward_sum.cu), to the bit.
//
// Why: in the online log-sum-exp the other kernels run per node tile, everything but the MMA depends on the edge tile
// and the batch column only, never on the node --
//     the tile's max over its 64 edges            mx[b]
//     the B operand                                bf16 exp(element[e, b] - mx[b])
//     the running max                              M[b] = max(M[b], mx[b])
//     both merge factors                           alpha[b] = exp(M_old - M_new), beta[b] = exp(mx - M_new)
// -- yet every node tile recomputed them: the exponentials once per node tile (8x over at block size 1024, tile 128),
// the merge factors once per ACCUMULATOR (two exponentials each, 128x over). Here the pre-pass computes the max and the
// exponentials once per (node block, edge tile, batch column), and the streaming kernel's producer warps stage the
// bf16 parameter tile and the pre-pass's exponential tile with cp.async (several tiles in flight) while one of them
// computes the merge factors per column; the consumer warps do nothing but the MMA and
//     L = L * alpha + S * beta        (fmul, fmul, fadd: the other kernels' expression, uncontracted)
// -- no exponentials, no barriers among themselves. The output is log(L) + M, -inf where M is.
//
// MEASURED (RTX PRO 6000, against the pipelined kernel's best config, one layer, bf16 params):
//     HCLT-1024's widest layer (143 blocks of 1024, 16 edge tiles), B 128   541 -> 385 us
//     HMM-1024 (one block of 1024), B 64                                    35.7 -> 18.3
//     HMM-4096 (4 blocks of 1024, 64 edge tiles), B 64                     126 -> 47.0
// and at fp32 params 593 -> 576, 37.2 -> 21.9, 129 -> 52.1 (the fp32 parameters, 600 MB on HCLT, are a 411 us DRAM
// floor alone: that layer was already near it). Whole forwards, bf16: HCLT-1024 B 128 4.87 -> 3.99 ms, B 64 3.31 ->
// 2.87; HMM-1024 (256 steps) B 128 10.55 -> 6.40, B 64 9.81 -> 6.45 -- the log-likelihoods identical to the bit.
//
// The smem atom: 64 consecutive m (or b) of one k per 128-byte row, Swizzle<3,3,3> over it (MN-major). The other
// kernels' atom ((8,8),8):((1,64),8) spreads the 16-byte chunks of one global row over 8 smem rows; with cp.async
// that streamed the parameters at HALF the DRAM rate (MEASURED: a bare parameter stream 335 vs 202 us, 1.49 TB/s).
// Both are conflict-free for the transposed ldmatrix.
//
// The pipeline: tile t goes to stage t % ST, one cp.async commit group per tile; the producers signal FULL for tile j
// as soon as its group has landed (wait_group<ST - 2>) and only then refill the stage tile j - 1 left (MEASURED: a
// first version that signalled tile j after issuing tile j + ST - 1 put a memory round trip on every tile's path).
// fp32 parameters cannot be cp.async'd into the bf16 operand: the producers load the next tile's into registers one
// issue ahead and convert them at the issue (round to nearest, as the other kernels).
//
// Scratch: the pre-pass's exponentials (bf16, the element tiles' size, half of `element_mars`' bytes for distinct
// tiles) and maxima are one allocation per call from the caching allocator -- inside a CUDA graph capture, from the
// graph's pool -- and freed when it returns (stream-ordered, the kernels queued on the current stream).
//
// SPECIALIZED fast path under the plain kernel's conditions (LL propagation, bf16 path, no tempering, TILE_SIZE_K ==
// 64, contiguous edge/param layout) and block_size % BM == 0, batch % 64 == 0; the Python dispatcher measures it
// against the other kernels' configs per shape (`autotune.choose`).

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
#define PB 64           // the pre-pass's batch columns per thread block (and the streaming kernel's BN)
#define MAX_SPLIT 8     // split-K ranges of a node block's edge tiles at most

// Named barriers: all `count` threads of the participating warps take part; `arrive` does not wait.
// ids: 1 + s = stage s FULL (producers arrive, consumers sync); 1 + ST + s = stage s EMPTY (consumers arrive, producers
// sync); 0 is __syncthreads.
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

// ---------------- pre-pass: a thread block per (node block x edge tile, PB batch columns) ----------------
// bx[(tile * BK + e) * batch + b] = bf16 exp(element[e0 + e, b] - mx), mx[tile * batch + b] the max over the tile's
// edges (exp's argument and the -inf guard as in the other kernels); tile = node block * knt + edge tile.
template <int NT>
__global__ void __launch_bounds__(NT) prepass_kernel(const float* __restrict__ element_mars, const long* __restrict__ ebase,
                                                    __nv_bfloat16* __restrict__ bx, float* __restrict__ mx, int batch) {
    constexpr int PARTS = NT / PB, EP = BK / PARTS;
    static_assert(NT % PB == 0 && BK % PARTS == 0, "threads over the batch columns");
    __shared__ float smx[PARTS][PB];
    long tile = blockIdx.x;
    int b0 = blockIdx.y * PB;
    int b = threadIdx.x % PB, part = threadIdx.x / PB;
    long e0 = ebase[tile];
    float v[EP];
    CUTE_UNROLL
    for (int j = 0; j < EP; j++) v[j] = element_mars[(e0 + part * EP + j) * (long)batch + b0 + b];
    float m = -INFINITY;
    CUTE_UNROLL
    for (int j = 0; j < EP; j++) m = fmaxf(m, v[j]);
    smx[part][b] = m;
    __syncthreads();
    m = smx[0][b];
    CUTE_UNROLL
    for (int q = 1; q < PARTS; q++) m = fmaxf(m, smx[q][b]);
    CUTE_UNROLL
    for (int j = 0; j < EP; j++)
        bx[(tile * BK + part * EP + j) * (long)batch + b0 + b] =
            __float2bfloat16((m == -INFINITY) ? 0.f : __expf(v[j] - m));
    if (part == 0) mx[tile * (long)batch + b0 + b] = m;
}

extern __shared__ char smem_raw[];

// ---------------- streaming kernel: PW producer warps beside the WM x WN MMA warps, ST stages ----------------
// TIX: the pre-pass tile of (node block, edge tile) is `tix[...]` (node blocks sharing their children share tiles); false
// where it is the entry's own index (no block shares: the table is never read -- MEASURED the read, prefetched a tile
// ahead, cost the fp32 kernels 13%)
template <int BM, int BN, int WM, int WN, int PW, int ST, bool TIX, typename TP>
__global__ void __launch_bounds__((WM * WN + PW) * 32) tlmm_prepass_kernel(
        float* __restrict__ node_mars, const TP* __restrict__ mp, const long* __restrict__ nids,
        const long* __restrict__ pbase, const __nv_bfloat16* __restrict__ bx, const float* __restrict__ mx,
        const int* __restrict__ tix, float* __restrict__ part, int batch, int block_size, int knt, int split,
        long rows_total, int pid_m_offset) {
    constexpr int NC = WM * WN * 32;                 // consumer threads
    constexpr int NP = PW * 32;                      // producer threads
    constexpr int NTH = NC + NP;
    constexpr bool BF16 = std::is_same<TP, __nv_bfloat16>::value;
    static_assert(BN % 32 == 0 && 2 * ST + 1 <= 16, "columns per producer lane; named barriers");
    int mtiles = block_size / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int nblock = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_nid = nids[nblock];
    // split-K: this CTA walks the edge tiles k0 .. k1 of its node block (blockIdx.z of `split` ranges; one range is the
    // whole row), everything below over `knt` = the range's tiles; a range's partial sum and its maximum go to `part`
    // for `merge_kernel`, a single range writes the output itself
    const int sp = blockIdx.z, knt_all = knt;
    const int k0 = (int)((long)sp * knt_all / split), k1 = (int)((long)(sp + 1) * knt_all / split);
    knt = k1 - k0;
    const long* pb = pbase + (long)nblock * knt_all + k0;
    int tid = threadIdx.x;

    // 64 consecutive m (or b) of one k per 128-byte row, Swizzle<3, 3, 3> over it; a node tile of 32 rows is half such
    // a row: 32 m of one k per 64-byte row, Swizzle<2, 3, 3> (two k per 128 bytes, so the transposed ldmatrix's 8 k rows
    // still hit 8 distinct 16-byte bank groups)
    auto swz = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{});
    auto swz32 = composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{});
    auto sAl = tile_to_shape(std::conditional_t<(BM >= 64), decltype(swz), decltype(swz32)>{},
                             make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swz, make_shape(Int<BN>{}, Int<BK>{}));
    // a stage: the A operand (params, bf16), the B operand (exponentials, bf16), then alpha, beta, M per column
    constexpr int STAGE = BM * BK * 2 + BN * BK * 2 + 3 * BN * 4;
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    auto stF = [&](int s) { return (float*)(smem_raw + s * STAGE + BM * BK * 2 + BN * BK * 2); };

    if (tid >= NC) {
        // ---------------- producers ----------------
        int ptid = tid - NC;
        const long tile0 = (long)nblock * knt_all + k0;
        // warp 0's running max per column (lane + 32 j), and the next tile's maxima (loaded one issue ahead)
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
        // the parameter offsets two tiles ahead: pn1 tile t's, pn2 tile t + 1's (the next tile's fp32 loads need it)
        long pn1 = pb[0], pn2 = knt > 1 ? pb[1] : 0;
        constexpr int PA4 = BF16 ? 1 : (BM * BK) / 4 / NP;
        static_assert(BF16 || (BM * BK) / 4 % NP == 0, "fp32 params float4s per producer thread");
        float4 ra[PA4];
        auto load_fp32 = [&](long pcv) {
            if constexpr (!BF16) {
                CUTE_UNROLL
                for (int q = 0; q < PA4; q++) {
                    int i = ptid + q * NP;
                    int e = i / (BM / 4), mm = (i % (BM / 4)) * 4;
                    ra[q] = *(const float4*)&mp[pcv + (long)e * block_size + mm];
                }
            }
        };
        if constexpr (!BF16) load_fp32(pn1 + (long)tile_id * BM);
        auto issue = [&](int t) {
            int s = t % ST;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
            long pc = pn1 + (long)tile_id * BM;
            pn1 = pn2;
            if (t + 2 < knt) pn2 = pb[t + 2];
            const int tcur = tx1;
            tx1 = tx2;
            if (t + 2 < knt) tx2 = TIX ? tix[tile0 + t + 2] : (int)tile0 + t + 2;
            if constexpr (BF16) {
                // A operand[m, e] = params, 8 per 16-byte copy
                constexpr int PA = (BM * BK) / 8 / NP;
                static_assert((BM * BK) / 8 % NP == 0, "params chunks per producer thread");
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) {
                    int i = ptid + q * NP;
                    int e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                    cpa16(&sAt(mm, e), &mp[pc + (long)e * block_size + mm]);
                }
            }
            // B operand[b, e] = the pre-pass's exponentials, 8 per 16-byte copy
            constexpr int PBC = (BN * BK) / 8 / NP;
            static_assert((BN * BK) / 8 % NP == 0, "exponential chunks per producer thread");
            const __nv_bfloat16* bt = bx + (long)tcur * BK * (long)batch + b0;
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                int i = ptid + q * NP;
                int e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
                cpa16(&sBt(bb, e), &bt[(long)e * batch + bb]);
            }
            if constexpr (!BF16) {
                // A operand from the registers loaded one issue ago, fp32 -> bf16; then the next tile's
                CUTE_UNROLL
                for (int q = 0; q < PA4; q++) {
                    int i = ptid + q * NP;
                    int e = i / (BM / 4), mm = (i % (BM / 4)) * 4;
                    bfloat16_t r[4] = {static_cast<bfloat16_t>(ra[q].x), static_cast<bfloat16_t>(ra[q].y),
                                       static_cast<bfloat16_t>(ra[q].z), static_cast<bfloat16_t>(ra[q].w)};
                    *(uint2*)&sAt(mm, e) = *(const uint2*)r;
                }
                if (t + 1 < knt) load_fp32(pn1 + (long)tile_id * BM);
            }
            if (ptid < 32) {
                // the online log-sum-exp's merge of tile t, per column: the other kernels' expressions on the same
                // values (M the running max, mxk the tile's), alpha = 1 and beta = 0 where they leave L as it is
                float* f = stF(s);
                CUTE_UNROLL
                for (int j = 0; j < BN / 32; j++) {
                    float mxk = mnext[j], Mo = Mrun[j];
                    float nM = fmaxf(Mo, mxk);
                    float al, be;
                    if (nM != -INFINITY) {
                        al = (Mo == -INFINITY) ? 0.f : __expf(Mo - nM);
                        be = (mxk == -INFINITY) ? 0.f : __expf(mxk - nM);
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
        // prologue: tiles 0 .. ST - 1, an (empty past the end) commit group each, so group g is tile g throughout
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();                      // groups <= j landed (j + ST - 2 or more committed)
            bar_arrive(1 + j % ST, NTH);             // FULL: tile j staged
            if (j >= 1) {
                int t = j - 1 + ST;                  // refill the stage tile j - 1 left
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
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);

    for (int kt = 0; kt < knt; kt++) {
        int s = kt % ST;
        bar_sync(1 + s, NTH);                        // FULL: tile kt staged
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBt), tXrB);
        // bf16 tensor-core dot: tCrS[m, b] = sum_e params[m, e] * exp(emar[e, b] - max[b]); fp32 accumulate
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
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);    // EMPTY: the stage is consumed (only where it is refilled)
    }
    const float* fM = stF((knt - 1) % ST) + 2 * BN;  // the last tile's stage: the running maxima, never refilled
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        const float Mf = fM[b];
        if (split == 1) {
            node_mars[(off_nid + (long)tile_id * BM + m) * (long)batch + b0 + b] =
                (Mf == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + Mf);
        } else {
            const long idx = ((long)sp * rows_total + (long)nblock * block_size + (long)tile_id * BM + m) * batch + b0 + b;
            part[idx] = tCrL(i);
            part[(long)split * rows_total * batch + idx] = Mf;
        }
    }
}

// ---------------- the ranges' merge (split > 1): a thread per (node, column) ----------------
// node_mars = log sum_z L_z exp(M_z - M) + M over the ranges, M their maximum (-inf where every M_z is), the sum in
// range order. `part`: L [split, rows, batch] then M [split, rows, batch]; row = node block x block_size + node.
__global__ void merge_kernel(float* __restrict__ node_mars, const float* __restrict__ part, const long* __restrict__ nids,
                             int split, long rows_total, int block_size, int batch) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_total * batch) return;
    const long row = i / batch, stride = rows_total * (long)batch;
    const int b = (int)(i - row * batch);
    const float* Lp = part + row * batch + b;
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
        r = logf(L) + M;
    }
    node_mars[(nids[row / block_size] + row % block_size) * (long)batch + b] = r;
}

template <int BM, int BN, int WM, int WN, int PW, int ST, bool TIX, typename TP>
static void launch_cfg(float* node_mars, const TP* params, const long* nids, int n_nblocks, const long* pbase,
                       const __nv_bfloat16* bx, const float* mx, const int* tix, float* part, int split, int batch,
                       int block_size, int knt, cudaStream_t stream) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr int smem = ST * (BM * BK * 2 + BN * BK * 2 + 3 * BN * 4);
    auto kernel = tlmm_prepass_kernel<BM, BN, WM, WN, PW, ST, TIX, TP>;
    // the shared-memory attribute once per device and config (the call costs host time on every launch of a small
    // layer); the flags are this instantiation's own
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_set[dev] = true;
    }
    int total_m = n_nblocks * (block_size / BM);
    const long rows_total = (long)n_nblocks * block_size;
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk, split);
        kernel<<<grid, NTH, smem, stream>>>(node_mars, params, nids, pbase, bx, mx, tix, part, batch, block_size, knt,
                                            split, rows_total, off);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (split > 1) {
        const long n = rows_total * batch;
        merge_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(node_mars, part, nids, split, rows_total,
                                                                      block_size, batch);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// Config table (id -> {BM, BN, WM, WN, PW, ST}): 8 MMA warps on a (128, 64) tile, 3 stages (76 KB), the large layers'
// (MEASURED on HCLT-1024's widest layer at B 128, the streaming kernel alone: 306 us, against 322 with 16 MMA warps,
// 323 with 2 producer warps on 4 stages, 416 with 16 MMA warps and 2 producer warps); 16 MMA warps as the alternative;
// the (64, 64) tile on 4 stages for few node blocks (MEASURED HMM-1024 B 64, both launches: 4 producer warps 18.3 us,
// 2 producer warps 19.1); the (32, 64) and (32, 128) tiles, 2 x 2 and 2 x 4 MMA warps, for layers of 32-node blocks
// (an HCLT's), which no other CUDA kernel takes.
template <bool TIX, typename TP>
static void dispatch_cfg(float* nm, const TP* p, const long* nids, int nb, const long* pb, const __nv_bfloat16* bx,
                         const float* mx, const int* tix, float* part, int split, int b, int bs, int k, int64_t cfg,
                         cudaStream_t st) {
    switch (cfg) {
        case 0: launch_cfg<128, 64, 4, 2, 4, 3, TIX, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, st); break;
        case 1: launch_cfg<128, 64, 8, 2, 4, 3, TIX, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, st); break;
        case 2: launch_cfg<64, 64, 4, 2, 4, 4, TIX, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, st); break;
        case 3: launch_cfg<64, 64, 4, 2, 2, 4, TIX, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, st); break;
        case 4: launch_cfg<32, 64, 2, 2, 4, 4, TIX, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, st); break;
        default: launch_cfg<32, 128, 2, 4, 4, 3, TIX, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, st); break;
    }
}

template <typename TP>
static void dispatch_tix(float* nm, const TP* p, const long* nids, int nb, const long* pb, const __nv_bfloat16* bx,
                         const float* mx, const int* tix, bool shared, float* part, int split, int b, int bs, int k,
                         int64_t cfg, cudaStream_t st) {
    if (shared) dispatch_cfg<true, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, cfg, st);
    else dispatch_cfg<false, TP>(nm, p, nids, nb, pb, bx, mx, tix, part, split, b, bs, k, cfg, st);
}

// The parameters in the dtype they are stored in (`TensorCircuit.param_dtype`): fp32 or bf16. The pre-pass's scratch
// is ONE allocation: the exponentials, then the maxima (16-byte aligned: the exponentials are tiles x 64 x batch bf16).
// `ebase`: the DISTINCT element tiles' first rows; `tix` (int32 [node blocks, edge tiles]): which of them each node
// block's edge tile is -- node blocks sharing their children (an HCLT's 32 blocks of a variable) share the pre-pass.
void tlmm_forward_sum_prepass(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                              torch::Tensor nids, torch::Tensor ebase, torch::Tensor tix, torch::Tensor pbase,
                              int64_t batch, int64_t block_size, int64_t knt, int64_t cfg, bool shared) {
    TORCH_CHECK(batch % PB == 0, "the pre-pass forward needs a batch that is a multiple of ", PB);
    TORCH_CHECK(ebase.is_contiguous() && ebase.dim() == 1, "ebase: [distinct element tiles]");
    TORCH_CHECK(tix.is_contiguous() && tix.scalar_type() == torch::kInt32 && tix.numel() == nids.numel() * knt,
                "tix: int32 [node blocks, edge tiles]");
    // `shared`: the blocks share tiles, `tix` read; else `ebase` is every entry's own tile in order and `tix` is
    // unread (a caller's word, not inferred from the sizes: a launch over some node blocks hands a slice of `tix`)
    long tiles = ebase.numel();
    TORCH_CHECK(shared || tiles == tix.numel(), "unshared tiles: one per (node block, edge tile)");
    // `cfg`: the tile config's id in the low byte, the split-K range count above it (0 or 1: one range); the ranges'
    // partials (L and M, [split, rows, batch] each) follow the pre-pass's scratch in the one allocation
    const int split = cfg >> 8 ? (int)(cfg >> 8) : 1;
    cfg &= 255;
    TORCH_CHECK(split <= MAX_SPLIT && split <= knt, "the pre-pass forward's split: 1 .. ", MAX_SPLIT, ", at most the ",
                knt, " edge tiles; got ", split);
    const long rows_total = nids.numel() * block_size;
    long bx_bytes = tiles * BK * batch * 2, mx_bytes = tiles * batch * 4;
    long part_bytes = split > 1 ? 2L * split * rows_total * batch * 4 : 0;
    torch::Tensor scratch = torch::empty({bx_bytes + mx_bytes + part_bytes}, element_mars.options().dtype(torch::kUInt8));
    auto* bx = reinterpret_cast<__nv_bfloat16*>(scratch.data_ptr());
    auto* mx = reinterpret_cast<float*>(reinterpret_cast<char*>(scratch.data_ptr()) + bx_bytes);
    auto* part = reinterpret_cast<float*>(reinterpret_cast<char*>(scratch.data_ptr()) + bx_bytes + mx_bytes);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    dim3 grid(tiles, batch / PB);
    prepass_kernel<256><<<grid, 256, 0, st>>>(element_mars.data_ptr<float>(), ebase.data_ptr<long>(), bx, mx, (int)batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int b = (int)batch, bs = (int)block_size, k = (int)knt, nb = (int)nids.size(0);
    if (params.scalar_type() == torch::kBFloat16)
        dispatch_tix<__nv_bfloat16>(node_mars.data_ptr<float>(), param_ptr<__nv_bfloat16>(params), nids.data_ptr<long>(),
                                    nb, pbase.data_ptr<long>(), bx, mx, tix.data_ptr<int>(), shared, part, split, b, bs,
                                    k, cfg, st);
    else
        dispatch_tix<float>(node_mars.data_ptr<float>(), param_ptr<float>(params), nids.data_ptr<long>(), nb,
                            pbase.data_ptr<long>(), bx, mx, tix.data_ptr<int>(), shared, part, split, b, bs, k, cfg, st);
}

// (BM, BN) per config id; Python uses this to enumerate the configs valid for a layer shape.
std::vector<std::vector<int64_t>> configs() {
    return {{128, 64}, {128, 64}, {64, 64}, {64, 64}, {32, 64}, {32, 128}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tlmm_forward_sum_prepass", &tlmm_forward_sum_prepass,
          "block-sparse sum-layer forward: exponentials pre-pass + streaming kernel (CuTe)");
    m.def("configs", &configs, "list of (BM, BN) per config id");
}
