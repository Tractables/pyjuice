// SMALL-BATCH (batch < 64) forward for the per-block multiplicative external parameterization (`BlockScaleSumParams`)
// on tensor cores -- the gated twin of layer/kernels/c/smallbatch_forward_sum_mma.cu. A block = (node tile of BM rows of
// one node block, batch tile of BN, range of the row's 64-edge tiles).
//
// Per node m and sample b (g(e) the gate covering edge e, log phi[g, b] its log weight, -inf without a gate):
//     N = sum_e theta[m, e] * exp(element_mars[e, b] + log phi[g(e), b])
//     Z = sum_e theta[m, e] * exp(                      log phi[g(e), b])
//     node_mars = log N - log Z,     log_z_out = log Z
// BOTH ARE TENSOR-CORE PRODUCTS OF THE SAME PARAMETER TILE: N = A @ B_N with B_N[e, b] = exp(x + log phi - M_N[b]) and
// Z = A @ (B_Zhi + B_Zlo) with B_Z[e, b] = exp(log phi - M_Z[b]) as two bf16 halves (the high bf16 and the bf16 of the
// remainder: ~16 bits of the gate weight -- one bf16 alone put Z's rounding on top of N's, MEASURED 2.02e-3 on node
// values against the CuTe fork's 2e-3 bar, whose Z comes from fp32 parameter masses) -- the per-gate parameter mass the
// per-sample kernel factors out needs no pass of its own, and the bf16 rounding of the parameters is common to N and Z.
//
// The PW producer warps take the range's maxima per column first (M_N over x + log phi, M_Z over log phi), then stage
// per tile the bf16 parameters (MN-major, cp.async; fp32 through registers, rounded) and the two bf16 exponential tiles
// (K-major); the MMA warps keep two fp32 accumulators. A range writes its partials (L_N, M_N, L_Z, M_Z); `merge_kernel`
// (a programmatic dependent launch) combines the ranges and writes node_mars and log Z; one range writes them itself.
//
// Rows: a node block's edges are one contiguous run (`row_contig`: child(e) = cids[nb, 0] + e, params block_size-
// strided), or edge blocks of node_cbs >= 64 each contiguous (their bases `cids` / `pids` [nb, n_eblks]), so a 64-edge
// tile lies in one; block_size a multiple of BM; num_edges a multiple of 64; the caller verifies. A padded edge block's
// gate is -1: its terms are 0 in N and Z alike (its dummy children and parameters are read and multiplied by 0).
//
// Numerically: bf16 operands (the exponentials and the parameters), fp32 accumulation -- the class of the other
// tensor-core block-scale forwards; N and Z share the parameter rounding.

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"
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
__global__ void __launch_bounds__((WM * WN + PW) * 32) bs_sbmma_kernel(
        float* __restrict__ node_mars, const float* __restrict__ element_mars, const TP* __restrict__ params,
        const float* __restrict__ ext, const long* __restrict__ nids, const long* __restrict__ cids,
        const long* __restrict__ pids, const long* __restrict__ gate, float* __restrict__ log_z_out,
        float* __restrict__ part, int batch, int block_size, int knt_all, int split, long rows_total, int bpad,
        int node_cbs, int node_sh, int gate_sh, int gate_stride, int n_eblks, long ext_base, int row_contig) {
    constexpr int NC = WM * WN * 32, NP = PW * 32, NTH = NC + NP;
    constexpr bool BF16 = std::is_same<TP, __nv_bfloat16>::value;
    static_assert(BN % 8 == 0 && BN <= 64 && 2 * ST + 1 <= 15, "columns; named barriers (15 is the producers' own)");
    const int mtiles = block_size / BM;
    const int pid_b = blockIdx.x, pid_m = blockIdx.y, sp = blockIdx.z;
    const int nb = pid_m / mtiles, row_off = (pid_m % mtiles) * BM;
    const int b0 = pid_b * BN;
    const int k0 = (int)((long)sp * knt_all / split), k1 = (int)((long)(sp + 1) * knt_all / split);
    const int knt = k1 - k0;
    const long* ec = cids + (long)nb * n_eblks;
    const long* ep = pids + (long)nb * n_eblks;
    const long* gt = gate + (long)nb * gate_stride;
    const int tid = threadIdx.x;
    // tile t (of the row) -> its first child and its first parameter (row 0 of the block)
    auto tile_child = [&](int t) -> long {
        const int eg = t * BK;
        return row_contig ? ec[0] + eg : ec[eg >> node_sh] + (eg & (node_cbs - 1));
    };
    auto tile_param = [&](int t) -> long {
        const int eg = t * BK;
        return row_contig ? ep[0] + (long)eg * block_size : ep[eg >> node_sh] + (long)(eg & (node_cbs - 1)) * block_size;
    };
    // edge eg of the row -> its gate's row in `ext` (-1: no gate)
    auto gate_row = [&](int eg) -> long {
        const int j = eg >> node_sh, d = (eg & (node_cbs - 1)) >> gate_sh;
        const long gb = (j < gate_stride) ? gt[j] : -1;
        return gb >= 0 ? gb + ext_base + d : -1;
    };

    // smem: A (BM, BK) MN-major swizzled; B_N and B_Z (BN, BK) K-major; then the range maxima (M_N, M_Z per column) and
    // the producer warps' partial maxima. No static shared memory (it would shift this buffer off its alignment).
    auto swz = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{});
    auto sAl = tile_to_shape(swz, make_shape(Int<BM>{}, Int<BK>{}));
    auto swzK = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});
    auto sBl = tile_to_shape(swzK, make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int STAGE = BM * BK * 2 + 3 * BN * BK * 2;
    auto stA = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE); };
    auto stBn = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    auto stBz = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2 + BN * BK * 2); };
    auto stBz2 = [&](int s) { return (bfloat16_t*)(smem_raw + s * STAGE + BM * BK * 2 + 2 * BN * BK * 2); };
    float* sMn = (float*)(smem_raw + ST * STAGE);                       // [BN]
    float* sMz = sMn + BN;                                               // [BN]
    float* sMp = sMz + BN;                                               // [2 x PW x BN] the warps' partial maxima

    if (tid >= NC) {
        // ---------------- producers ----------------
        const int ptid = tid - NC;
        const int ncols = min(BN, batch - b0);
        // the range maxima per column: a thread takes whole edge rows -- its element row and its gate's row of `ext`
        {
            float mxn[BN], mxz[BN];
            CUTE_UNROLL
            for (int j = 0; j < BN; j++) { mxn[j] = -INFINITY; mxz[j] = -INFINITY; }
            // (float4 loads where the columns are whole and aligned, as the regular twin's prologue)
            const bool vec = (ncols == BN) && ((batch & 3) == 0) && ((b0 & 3) == 0)
                             && ((((size_t)element_mars) & 15) == 0) && ((((size_t)ext) & 15) == 0);
            for (int i = ptid; i < knt * BK; i += NP) {
                const int t = k0 + i / BK, e = i % BK, eg = t * BK + e;
                const long gr = gate_row(eg);
                if (gr < 0) continue;
                const float* xr = element_mars + (tile_child(t) + e) * (long)batch + b0;
                const float* lr = ext + gr * (long)batch + b0;
                if (vec) {
                    CUTE_UNROLL
                    for (int j4 = 0; j4 < BN / 4; j4++) {
                        const float4 vx = reinterpret_cast<const float4*>(xr)[j4];
                        const float4 vl = reinterpret_cast<const float4*>(lr)[j4];
                        mxz[j4 * 4 + 0] = fmaxf(mxz[j4 * 4 + 0], vl.x); mxn[j4 * 4 + 0] = fmaxf(mxn[j4 * 4 + 0], vx.x + vl.x);
                        mxz[j4 * 4 + 1] = fmaxf(mxz[j4 * 4 + 1], vl.y); mxn[j4 * 4 + 1] = fmaxf(mxn[j4 * 4 + 1], vx.y + vl.y);
                        mxz[j4 * 4 + 2] = fmaxf(mxz[j4 * 4 + 2], vl.z); mxn[j4 * 4 + 2] = fmaxf(mxn[j4 * 4 + 2], vx.z + vl.z);
                        mxz[j4 * 4 + 3] = fmaxf(mxz[j4 * 4 + 3], vl.w); mxn[j4 * 4 + 3] = fmaxf(mxn[j4 * 4 + 3], vx.w + vl.w);
                    }
                } else {
                    CUTE_UNROLL
                    for (int j = 0; j < BN; j++) {
                        if (j < ncols) {
                            const float l = lr[j];
                            mxz[j] = fmaxf(mxz[j], l);
                            mxn[j] = fmaxf(mxn[j], xr[j] + l);
                        }
                    }
                }
            }
            CUTE_UNROLL
            for (int j = 0; j < BN; j++) {
                float a = mxn[j], z = mxz[j];
                CUTE_UNROLL
                for (int off = 16; off > 0; off >>= 1) {
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                    z = fmaxf(z, __shfl_xor_sync(0xffffffffu, z, off));
                }
                mxn[j] = a; mxz[j] = z;
            }
            const int pw = ptid / 32, pl = ptid % 32;
            CUTE_UNROLL
            for (int j0 = 0; j0 < BN; j0 += 32) {
                if (j0 + pl < BN) {
                    float a = -INFINITY, z = -INFINITY;
                    CUTE_UNROLL
                    for (int j = 0; j < BN; j++) if (j == j0 + pl) { a = mxn[j]; z = mxz[j]; }
                    sMp[pw * BN + j0 + pl] = a;
                    sMp[(PW + pw) * BN + j0 + pl] = z;
                }
            }
            bar_sync(15, NP);
            if (ptid < BN) {
                float a = -INFINITY, z = -INFINITY;
                CUTE_UNROLL
                for (int w = 0; w < PW; w++) { a = fmaxf(a, sMp[w * BN + ptid]); z = fmaxf(z, sMp[(PW + w) * BN + ptid]); }
                sMn[ptid] = a; sMz[ptid] = z;
            }
            bar_sync(15, NP);
        }
        constexpr int CH = (BN * BK) / 8;                                // exponential chunks (8 edges of a column)
        constexpr int PBC = (CH + NP - 1) / NP;
        float Mnc[PBC], Mzc[PBC];
        CUTE_UNROLL
        for (int q = 0; q < PBC; q++) {
            const int i = ptid + q * NP;
            Mnc[q] = (i < CH) ? sMn[i / (BK / 8)] : 0.f;
            Mzc[q] = (i < CH) ? sMz[i / (BK / 8)] : 0.f;
        }
        constexpr int PA4 = BF16 ? 1 : (BM * BK) / 4 / NP;
        static_assert(BF16 || (BM * BK) / 4 % NP == 0, "fp32 params float4s per producer thread");
        constexpr int PA = BF16 ? (BM * BK) / 8 / NP : 1;
        static_assert(!BF16 || (BM * BK) / 8 % NP == 0, "params chunks per producer thread");
        float4 ra[PA4];
        auto load_fp32 = [&](int t) {
            if constexpr (!BF16) {
                const TP* pc = params + tile_param(k0 + t) + row_off;
                CUTE_UNROLL
                for (int q = 0; q < PA4; q++) {
                    const int i = ptid + q * NP, e = i / (BM / 4), mm = (i % (BM / 4)) * 4;
                    ra[q] = *(const float4*)&pc[(long)e * block_size + mm];
                }
            }
        };
        // the element values and the gates' log weights of a tile's chunks, loaded a tile ahead
        float xv[PBC][8], lv[PBC][8];
        auto load_x = [&](int t) {
            const long ch0 = tile_child(k0 + t);
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                if (i >= CH) continue;
                const bool live = b0 + c < batch;
                const int eg0 = (k0 + t) * BK + e8;
                if (gate_sh >= 3) {
                    // a gate of >= 8 edges (inside one edge block: gates divide it) covers the chunk's 8 edges: one
                    // load -- eight of the same value were ~a quarter of the kernel (MEASURED B16 bf16 34.8 -> 26.6 us)
                    const long gr = gate_row(eg0);
                    const float l = (live && gr >= 0) ? ext[gr * (long)batch + b0 + c] : -INFINITY;
                    CUTE_UNROLL
                    for (int j = 0; j < 8; j++) {
                        lv[q][j] = l;
                        xv[q][j] = live ? element_mars[(ch0 + e8 + j) * (long)batch + b0 + c] : -INFINITY;
                    }
                } else {
                    CUTE_UNROLL
                    for (int j = 0; j < 8; j++) {
                        const long gr = gate_row(eg0 + j);
                        lv[q][j] = (live && gr >= 0) ? ext[gr * (long)batch + b0 + c] : -INFINITY;
                        xv[q][j] = live ? element_mars[(ch0 + e8 + j) * (long)batch + b0 + c] : -INFINITY;
                    }
                }
            }
        };
        if constexpr (!BF16) load_fp32(0);
        load_x(0);
        auto issue = [&](int t) {
            const int s = t % ST;
            Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
            Tensor sBn = make_tensor(make_smem_ptr(stBn(s)), sBl);
            Tensor sBz = make_tensor(make_smem_ptr(stBz(s)), sBl);
            Tensor sBz2 = make_tensor(make_smem_ptr(stBz2(s)), sBl);
            if constexpr (BF16) {
                const TP* pc = params + tile_param(k0 + t) + row_off;
                CUTE_UNROLL
                for (int q = 0; q < PA; q++) {
                    const int i = ptid + q * NP, e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                    cpa16(&sAt(mm, e), &pc[(long)e * block_size + mm]);
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
            CUTE_UNROLL
            for (int q = 0; q < PBC; q++) {
                const int i = ptid + q * NP, c = i / (BK / 8), e8 = (i % (BK / 8)) * 8;
                if (i >= CH) continue;
                const float Mn = Mnc[q], Mz = Mzc[q];
                __align__(16) bfloat16_t un[8], uz[8], uz2[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; j++) {
                    const float l = lv[q][j], a = xv[q][j] + l;
                    un[j] = static_cast<bfloat16_t>((Mn == -INFINITY || a == -INFINITY) ? 0.f : __expf(a - Mn));
                    const float w = (Mz == -INFINITY || l == -INFINITY) ? 0.f : __expf(l - Mz);
                    uz[j] = static_cast<bfloat16_t>(w);
                    uz2[j] = static_cast<bfloat16_t>(w - static_cast<float>(uz[j]));
                }
                *(uint4*)&sBn(c, e8) = *(const uint4*)un;
                *(uint4*)&sBz(c, e8) = *(const uint4*)uz;
                *(uint4*)&sBz2(c, e8) = *(const uint4*)uz2;
            }
            if (t + 1 < knt) load_x(t + 1);
        };
        CUTE_UNROLL
        for (int t = 0; t < ST; t++) { if (t < knt) issue(t); cpa_commit(); }
        for (int j = 0; j < knt; j++) {
            cpa_wait<ST - 2>();
            __threadfence_block();
            bar_arrive(1 + j % ST, NTH);
            if (j >= 1) {
                const int t = j - 1 + ST;
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
    Tensor sBt0 = make_tensor(make_smem_ptr(stBn(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrBn = thr.partition_fragment_B(sBt0);
    Tensor tCrBz = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, bfloat16_t>{}, mma);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXrBn = s2rB.get_thread_slice(tid).retile_D(tCrBn);
    Tensor tXrBz = s2rB.get_thread_slice(tid).retile_D(tCrBz);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrN = thr.partition_fragment_C(cC);
    Tensor tCrZ = thr.partition_fragment_C(cC);
    clear(tCrN); clear(tCrZ);
    for (int kt = 0; kt < knt; kt++) {
        const int s = kt % ST;
        bar_sync(1 + s, NTH);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBn = make_tensor(make_smem_ptr(stBn(s)), sBl);
        Tensor sBz = make_tensor(make_smem_ptr(stBz(s)), sBl);
        Tensor sBz2 = make_tensor(make_smem_ptr(stBz2(s)), sBl);
        copy(s2rA, s2rA.get_thread_slice(tid).partition_S(sAt), tXrA);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBn), tXrBn);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBz), tXrBz);
        cute::gemm(mma, tCrA, tCrBn, tCrN);
        cute::gemm(mma, tCrA, tCrBz, tCrZ);
        copy(s2rB, s2rB.get_thread_slice(tid).partition_S(sBz2), tXrBz);
        cute::gemm(mma, tCrA, tCrBz, tCrZ);
        if (kt + ST < knt) bar_arrive(1 + ST + s, NTH);
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tCrN); i++) {
        const int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        const float Mn = sMn[b], Mz = sMz[b];
        const long row = (long)nb * block_size + row_off + m;
        if (split == 1) {
            if (b0 + b < batch) {
                const float ln = (Mn == -INFINITY || tCrN(i) <= 0.f) ? -INFINITY : logf(tCrN(i)) + Mn;
                const float lz = (Mz == -INFINITY || tCrZ(i) <= 0.f) ? -INFINITY : logf(tCrZ(i)) + Mz;
                node_mars[(nids[nb] + row_off + m) * (long)batch + b0 + b] =
                    (ln == -INFINITY || lz == -INFINITY) ? -INFINITY : ln - lz;
                if (log_z_out != nullptr) log_z_out[row * (long)batch + b0 + b] = lz;
            }
        } else {
            const long stride = (long)split * rows_total * bpad;
            const long idx = ((long)sp * rows_total + row) * bpad + b0 + b;
            part[idx] = tCrN(i);
            part[stride + idx] = Mn;
            part[2 * stride + idx] = tCrZ(i);
            part[3 * stride + idx] = Mz;
        }
    }
}

// ---------------- the ranges' merge (split > 1): a thread per (node, column) ----------------
// `part`: L_N, M_N, L_Z, M_Z, each [split, rows, bpad]; row = node block x block_size + node. A programmatic dependent
// launch: it waits here for the primary grid's writes.
__device__ __forceinline__ float merge_ranges(const float* L, const float* M, long stride, int split) {
    float mx = -INFINITY;
    for (int z = 0; z < split; z++) mx = fmaxf(mx, M[(long)z * stride]);
    if (mx == -INFINITY) return -INFINITY;
    float s = 0.f;
    for (int z = 0; z < split; z++) { const float mz = M[(long)z * stride]; if (mz != -INFINITY) s += L[(long)z * stride] * __expf(mz - mx); }
    return (s <= 0.f) ? -INFINITY : logf(s) + mx;
}

__global__ void bs_merge_kernel(float* __restrict__ node_mars, float* __restrict__ log_z_out, const float* __restrict__ part,
                                const long* __restrict__ nids, int split, long rows_total, int block_size, int batch, int bpad) {
    cudaGridDependencySynchronize();
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows_total * batch) return;
    const long row = i / batch, rs = rows_total * (long)bpad, stride = (long)split * rs;
    const int b = (int)(i - row * batch);
    const float* base = part + row * bpad + b;
    const float ln = merge_ranges(base, base + stride, rs, split);
    const float lz = merge_ranges(base + 2 * stride, base + 3 * stride, rs, split);
    node_mars[(nids[row / block_size] + row % block_size) * (long)batch + b] =
        (ln == -INFINITY || lz == -INFINITY) ? -INFINITY : ln - lz;
    if (log_z_out != nullptr) log_z_out[row * (long)batch + b] = lz;
}

template <int BM, int BN, int WM, int WN, int PW, int ST, typename TP>
static void launch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params, torch::Tensor ext,
                       torch::Tensor nids, torch::Tensor cids, torch::Tensor pids, torch::Tensor gate, torch::Tensor log_z,
                       int batch, int block_size, int knt, int split, int node_cbs, int node_sh, int gate_sh,
                       long ext_base, int row_contig) {
    constexpr int NTH = (WM * WN + PW) * 32;
    constexpr int smem = ST * (BM * BK * 2 + 3 * BN * BK * 2) + (2 * BN + 2 * PW * BN) * 4;
    auto kernel = bs_sbmma_kernel<BM, BN, WM, WN, PW, ST, TP>;
    static bool attr_set[64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[dev]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        if (dev < 64) attr_set[dev] = true;
    }
    const int nb = (int)nids.size(0), total_m = nb * (block_size / BM);
    const long rows_total = (long)nb * block_size;
    const int btiles = (batch + BN - 1) / BN, bpad = btiles * BN;
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    torch::Tensor scratch;
    float* part = nullptr;
    if (split > 1) {
        scratch = torch::empty({4L * split * rows_total * bpad}, node_mars.options());
        part = scratch.data_ptr<float>();
    }
    float* lz = log_z.numel() ? log_z.data_ptr<float>() : nullptr;
    dim3 grid(btiles, total_m, split);
    kernel<<<grid, NTH, smem, st>>>(node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), param_ptr<TP>(params),
                                    ext.data_ptr<float>(), nids.data_ptr<long>(), cids.data_ptr<long>(),
                                    pids.data_ptr<long>(), gate.data_ptr<long>(), lz, part, batch, block_size, knt,
                                    split, rows_total, bpad, node_cbs, node_sh, gate_sh, (int)gate.size(1),
                                    (int)cids.size(1), ext_base, row_contig);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (split > 1) {
        const long n = rows_total * batch;
        cudaLaunchConfig_t cfgl = {};
        cfgl.gridDim = dim3((unsigned)((n + 255) / 256)); cfgl.blockDim = dim3(256); cfgl.dynamicSmemBytes = 0; cfgl.stream = st;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfgl.attrs = attr; cfgl.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfgl, bs_merge_kernel, node_mars.data_ptr<float>(), lz, (const float*)part,
                                          (const long*)nids.data_ptr<long>(), split, rows_total, block_size, batch, bpad));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// cfg -> (BM, BN, WM, WN, PW, ST), within sm_120's 99 KB of shared memory
#define BS_SBMMA_CONFIGS(X) \
    X(0, 128, 16, 4, 1, 4, 3) X(1, 64, 16, 2, 1, 4, 4) X(2, 128, 32, 4, 1, 4, 3) X(3, 64, 32, 2, 1, 4, 4) \
    X(4, 128, 16, 4, 1, 8, 3) X(5, 128, 64, 4, 2, 4, 2) X(6, 64, 64, 2, 2, 4, 3)

// The per-sample kernel's arguments (`blockscale_sb_forward`), `cfg` = config id | split << 8 last.
void blockscale_sb_forward_mma(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                               torch::Tensor ext, torch::Tensor nids, torch::Tensor cids, torch::Tensor pids,
                               torch::Tensor gate, torch::Tensor log_z, int64_t block_size, int64_t num_edges,
                               int64_t node_cbs, int64_t gate_cbs, int64_t ext_base, int64_t row_contig, int64_t cfg) {
    const int batch = node_mars.size(1);
    const int split = cfg >> 8 ? (int)(cfg >> 8) : 1;
    cfg &= 255;
    const int bs = (int)block_size, ne = (int)num_edges, ncbs = (int)node_cbs, gcbs = (int)gate_cbs;
    TORCH_CHECK(batch >= 1 && batch < 64, "blockscale tensor-core small-batch forward: batch in [1, 64)");
    TORCH_CHECK(ne % BK == 0, "blockscale tensor-core small-batch forward: num_edges must be whole tiles of ", BK);
    TORCH_CHECK((ncbs & (ncbs - 1)) == 0 && (gcbs & (gcbs - 1)) == 0 && ncbs % gcbs == 0,
                "blockscale tensor-core small-batch forward: power-of-two child block sizes, the gate's dividing the node's");
    TORCH_CHECK(row_contig || ncbs % BK == 0, "blockscale tensor-core small-batch forward: a row of edge blocks of ",
                BK, " children or more, or one contiguous row");
    const int knt = ne / BK;
    TORCH_CHECK(split >= 1 && split <= MAX_SPLIT && split <= knt, "blockscale tensor-core small-batch forward: split");
    int node_sh = 0, gate_sh = 0;
    while ((1 << node_sh) < ncbs) ++node_sh;
    while ((1 << gate_sh) < gcbs) ++gate_sh;
    auto run = [&](auto tag) {
        using TP = decltype(tag);
        switch (cfg) {
#define BS_SBMMA_CASE(ID, BM_, BN_, WM_, WN_, PW_, ST_) case ID: \
            TORCH_CHECK(bs % BM_ == 0, "blockscale tensor-core small-batch forward: block_size % BM"); \
            launch_cfg<BM_, BN_, WM_, WN_, PW_, ST_, TP>(node_mars, element_mars, params, ext, nids, cids, pids, gate, log_z, \
                                                         batch, bs, knt, split, ncbs, node_sh, gate_sh, (long)ext_base, \
                                                         (int)row_contig); break;
            BS_SBMMA_CONFIGS(BS_SBMMA_CASE)
#undef BS_SBMMA_CASE
            default: TORCH_CHECK(false, "blockscale tensor-core small-batch forward: invalid cfg ", cfg);
        }
    };
    if (params.scalar_type() == torch::kBFloat16) run(__nv_bfloat16{}); else run(float{});
}

std::vector<std::vector<int64_t>> blockscale_sb_mma_configs() {
    std::vector<std::vector<int64_t>> out;
#define BS_SBMMA_LIST(ID, BM_, BN_, WM_, WN_, PW_, ST_) out.push_back({BM_, BN_});
    BS_SBMMA_CONFIGS(BS_SBMMA_LIST)
#undef BS_SBMMA_LIST
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_sb_forward_mma", &blockscale_sb_forward_mma,
          "small-batch block-scale forward on tensor cores (log N - log Z, both as products of the parameter tile)");
    m.def("blockscale_sb_mma_configs", &blockscale_sb_mma_configs, "(BM, BN) per config id");
}
