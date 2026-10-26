// The `BlockScaleSumParams` element-flow backward in two launches -- a pre-pass and a streaming kernel -- computing
// what `_bs_triton_ele_kernel` computes (one parent group per k-tile, `ptr_inc_step == 1`), not to its bits:
//
//     element_flows[m, b] = emars[m, b] + log sum_t sum_e params_t[m, e] exp(lf_t[e, b] + log phi_t[r(m), b])
//     d LL / d log phi[g_t + r, b] += sum_{m in gate row r} sum_e params_t[m, e] exp(emars[m, b] + lf_t[e, b] + log phi)
//
// with lf = node_flows - node_mars over a k-tile t's 64 parents, g_t the tile's gate (`gate`, -1 without one) and
// r(m) a child's gate row. Triton accumulates in log space tile by tile; this sums linearly with one stabilizer per
// (gate row, column) over all the tiles:
//   pre-pass   per (element block x k-tile, 64 columns): the tile's max mx_t over its parents and nsub = fp16
//              exp(lf - mx_t), rows [tile][b][e];
//   streaming  per (element tile, batch tile): the consumer warps first take Mf[r, b] = max_t (mx_t + log phi_t) --
//              a max in shared memory -- while the producer warps start the stages (the parameters through
//              registers, PD tiles ahead, converted to fp16; nsub by cp.async); then per tile S_t = params_t . nsub_t
//              (fp16 MMA, fp32 accumulate), L += w_t S_t with w_t = exp(mx_t + log phi_t - Mf); the output
//              emars + Mf + log L. G groups of consumer warps take the tiles kt = g (mod G) and add their L in order.
//              The gradient: S_t w_t exp(emars + Mf) summed over a gate row's children (warp shuffles), one atomic
//              add per (row, column) and tile -- the buffer arrives zeroed.
// w two ways, a config's choice: CWM, by the consumers from the stage's mx_t and log phi_t rows (cp.async; nothing
// but the consumers waits for Mf -- the small grids' choice); else by the producers once per (gate row, column) into
// the stage, from values loaded with the parameters (the large grids' and narrow gates' choice). SG (a gate at
// least as wide as the tile, one gate row) is a template so every row offset is a compile-time constant.
//
// MEASURED (RTX PRO 6000, one gated HMM layer's launch, a CUDA graph of 50, against the plan's pick today, Triton;
// the CuTe fork `blockscale_ele_backward` is ~2x slower than Triton on all of these): HMM-1024 B 64 / 128 bf16
// 44.1 / 44.2 -> 11.7 / 11.9 us, fp32 B 128 44.4 -> 12.8, B 512 57.2 -> 21.3, gate width 8 50.9 -> 14.1; HMM-4096
// (4 blocks of 1024, 64 k-tiles) B 128 bf16 219.5 -> 60.8, B 512 614.7 -> 180.4, fp32 gate width 8 B 64 199.3 -> 50.6;
// HMM-256 B 128 gate width 16 14.5 -> 8.1. Against float64, the element flows' max error 3e-5 .. 1.2e-4 (Triton's
// TF32 dots: 3e-4 .. 8e-4), the gradient's max relative error <= 4e-5 (Triton's 3e-4 .. 7e-4).
//
// Scope (`fitting_configs`, and the Python plan for the rest): k-tiles of 64 parents within one parent block, the
// parameters' 8-value loads aligned (`_cuda_param_loads_aligned`), a power-of-two gate width >= 4, batch % 64 == 0,
// the element block a multiple of the tile's rows, the element block's gates (knt ints) in shared memory, `ext`
// 16-byte aligned (its rows are 16-byte copies). Scratch: one allocation per call (inside a CUDA graph capture,
// from its pool).

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"   // `param_ptr<TP>`: the stored dtype
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64           // k-tile: parents
#define PB 64           // the pre-pass's batch columns per thread block
#define SMEM_MAX 101376

__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void cpa16(uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
// arrives on `bar` once every cp.async this thread issued before has landed (the barrier's count includes it)
__device__ __forceinline__ void cpa_mbar_arrive(uint64_t* bar) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    unsigned a = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
                 ::"r"(a), "r"(phase) : "memory");
}

// ---------------- pre-pass: a block per (element block x k-tile, PB columns) ----------------
__global__ void __launch_bounds__(256) bs_ele_prepass_kernel(const float* __restrict__ nflows,
                                                             const float* __restrict__ nmars,
                                                             const long* __restrict__ ebase, half_t* __restrict__ bx,
                                                             float* __restrict__ mx, int batch) {
    __shared__ float sLf[BK][PB + 1];
    __shared__ float sMx4[4][PB];
    __shared__ float sMx[PB];
    long tile = blockIdx.x;
    int b0 = blockIdx.y * PB, tid = threadIdx.x;
    long e0 = ebase[tile];
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

// a stage: the parameters [BM x BK] and nsub [BN x BK] (fp16); the v rows (fp32, BN each): mx_t (CWM), then per gate
// row of the tile log phi_t (CWM) or w (else). A multiple of 1 KiB: a stage stride 256 B off it made a 64 x 64
// tile's stream 9% slower (MEASURED).
template <int BM, int BN, bool CWM>
constexpr int stage_bytes() { return (BM * BK * 2 + BN * BK * 2 + (BM / 4 + (CWM ? 1 : 0)) * BN * 4 + 1023) / 1024 * 1024; }
// + the groups' merge rows, the tile's Mf [gate rows x BN], the barriers; then the element block's gates (knt ints)
template <int BM, int BN, int G, int ST, bool CWM>
constexpr int smem_bytes() {
    return ST * stage_bytes<BM, BN, CWM>() + (G > 1 ? BM * (BN + 8) * 4 : 0) + (BM / 4) * BN * 4 + 2 * ST * 8;
}

// ---------------- streaming kernel: G groups of WM x WN MMA warps, PW producer warps, ST stages ----------------
template <int BM, int BN, int WM, int WN, int G, int PW, int ST, int PD, bool CWM, bool WGRAD, bool SG, typename TP>
__global__ void __launch_bounds__((G * WM * WN + PW) * 32) bs_ele_stream_kernel(
        float* __restrict__ eflows, const float* __restrict__ emars, const TP* __restrict__ mp,
        const long* __restrict__ chids, const long* __restrict__ pbase, const long* __restrict__ gate,
        const half_t* __restrict__ bx, const float* __restrict__ mx, const float* __restrict__ ext,
        float* __restrict__ grad_ext, int batch, int BSK, int BSM, int knt, int gc, long ext_base,
        int pid_m_offset) {
    constexpr int NCG = WM * WN * 32, NC = G * NCG, NP = PW * 32;
    constexpr int WR = BM / 4;                         // gate rows a tile spans at most (gate width >= 4)
    int mtiles = BSM / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int eb = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    const long off_ele = chids[eb] + (long)tile_id * BM;
    const long* pbp = pbase + (long)eb * knt;
    const long tile0 = (long)eb * knt;
    const int r0 = (tile_id * BM) / gc;                // the tile's first gate row
    const int nwr = (!SG && BM >= gc) ? BM / gc : 1;   // gate rows the tile spans
    int tid = threadIdx.x;

    auto sAl = tile_to_shape(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}),
                             make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}),
                             make_shape(Int<BN>{}, Int<BK>{}));
    constexpr int STAGE = stage_bytes<BM, BN, CWM>();
    constexpr int VR0 = CWM ? 1 : 0;                   // the v row of gate row 0 (CWM: past mx_t's)
    auto stA = [&](int s) { return (half_t*)(smem_raw + s * STAGE); };
    auto stB = [&](int s) { return (half_t*)(smem_raw + s * STAGE + BM * BK * 2); };
    auto stV = [&](int s) { return (float*)(smem_raw + s * STAGE + BM * BK * 2 + BN * BK * 2); };
    constexpr int LS = BN + 8;
    float* sL = (float*)(smem_raw + ST * STAGE);
    float* sMf = (float*)(smem_raw + ST * STAGE + (G > 1 ? BM * LS * 4 : 0));
    uint64_t* full = (uint64_t*)(sMf + WR * BN);
    uint64_t* empty = full + ST;
    int* sGt = (int*)(empty + ST);
    if (tid == 0)
        for (int s = 0; s < ST; s++) { mbar_init(&full[s], 2 * NP); mbar_init(&empty[s], NCG); }
    __syncthreads();

    if (tid >= NC) {
        // ---------------- producers ----------------
        int ptid = tid - NC;
        constexpr int TA = (BM * BK) / 8, TB = (BN * BK) / 8, TW = (CWM ? WR + 1 : WR) * (BN / 4);
        constexpr int CA = (TA + NP - 1) / NP, CB = (TB + NP - 1) / NP, CW = (TW + NP - 1) / NP;
        uint32_t oA[CA], oB[CB], oW[CW];
        long gA[CA], gW[CW];
        int gB[CB], cW[CW];
        {
            Tensor sAt = make_tensor(make_smem_ptr(stA(0)), sAl);
            Tensor sBt = make_tensor(make_smem_ptr(stB(0)), sBl);
            CUTE_UNROLL
            for (int q = 0; q < CA; q++) {
                int i = ptid + q * NP, m = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                oA[q] = i < TA ? (uint32_t)((char*)&sAt(m, ee) - smem_raw) : ~0u;
                gA[q] = (long)m * BSK + ee;
            }
            CUTE_UNROLL
            for (int q = 0; q < CB; q++) {
                int i = ptid + q * NP, b = i / (BK / 8), ee = (i % (BK / 8)) * 8;
                oB[q] = i < TB ? (uint32_t)((char*)&sBt(b, ee) - smem_raw) : ~0u;
                gB[q] = b * BK + ee;
            }
            CUTE_UNROLL
            for (int q = 0; q < CW; q++) {                // v row r (CWM: 0 mx_t, 1 + j gate row j's) or gate row j
                int i = ptid + q * NP, r = i / (BN / 4), bb = (i % (BN / 4)) * 4;
                const int j = CWM ? r - 1 : r;
                oW[q] = (i < TW && j < nwr) ? (uint32_t)((char*)&stV(0)[(VR0 + j) * BN + bb] - smem_raw) : ~0u;
                gW[q] = j < 0 ? -1 - bb : (long)(r0 + j) * batch + bb;
                cW[q] = j * BN + bb;
            }
        }
        const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
        // the parameters, raw (8 a chunk: one uint4 of bf16, two of fp32), and (not CWM) the w chunks' mx_t and
        // log phi_t, loaded PE tiles ahead into a ring of register buffers; the tile loop unrolled by PE so every
        // buffer index is compile-time
        constexpr bool BF16 = std::is_same<TP, __nv_bfloat16>::value;
        constexpr int RW = BF16 ? 1 : 2;
        constexpr int PE = BF16 ? PD : (PD + 1) / 2;      // the depth (fp32 chunks: twice the registers)
        uint4 ra[PE][CA][RW];
        constexpr int RV = CWM ? 1 : PE;
        float4 rm[RV][CW], rl[RV][CW];
        auto loadP = [&](int t, uint4 (&buf)[CA][RW]) {
            if (t >= knt) return;
            const TP* pa = mp + pbp[t] + (long)tile_id * BM * BSK;
            CUTE_UNROLL
            for (int q = 0; q < CA; q++)
                if (oA[q] != ~0u) {
                    CUTE_UNROLL
                    for (int u = 0; u < RW; u++) buf[q][u] = *(const uint4*)(pa + gA[q] + u * 4);
                }
        };
        auto loadV = [&](int t, float4 (&bm)[CW], float4 (&bl)[CW]) {
            if (CWM || t >= knt) return;
            const int gb = sGt[t];
            const float* mxt = mx + (tile0 + t) * (long)batch + b0;
            const float* lpt = ext + (gb + ext_base) * (long)batch + b0;
            CUTE_UNROLL
            for (int q = 0; q < CW; q++)
                if (oW[q] != ~0u) {
                    bm[q] = *(const float4*)(mxt + (cW[q] % BN));
                    if (gb >= 0) bl[q] = *(const float4*)(lpt + gW[q]);
                }
        };
        // The gates go to shared memory behind the first parameter loads: an address that waited on a global load
        // of its gate inside the prefetch stalled the warp every tile (MEASURED).
        CUTE_UNROLL
        for (int d = 0; d < PE; d++) loadP(d, ra[d]);
        for (int i = ptid; i < knt; i += NP) sGt[i] = (int)gate[tile0 + i];
        bar_sync(3, NC + NP);
        CUTE_UNROLL
        for (int d = 0; d < PE; d++) loadV(d, rm[d % RV], rl[d % RV]);
        float4 mfv[CW];                                   // the chunks' Mf, once the consumers have them
        for (int t0 = 0; t0 < knt; t0 += PE) {
            CUTE_UNROLL
            for (int d = 0; d < PE; d++) {
                const int t = t0 + d;
                if (t >= knt) break;
                int s = t % ST;
                if (t >= ST) mbar_wait(&empty[s], ((t / ST) - 1) & 1);
                const uint32_t ss = sbase + s * STAGE;
                const half_t* bt = bx + ((tile0 + t) * batch + b0) * (long)BK;
                CUTE_UNROLL
                for (int q = 0; q < CB; q++) if (oB[q] != ~0u) cpa16(ss + oB[q], bt + gB[q]);
                if constexpr (CWM) {
                    const int gb = sGt[t];
                    const float* mxt = mx + (tile0 + t) * (long)batch + b0;
                    const float* lpt = ext + (gb + ext_base) * (long)batch + b0;
                    CUTE_UNROLL
                    for (int q = 0; q < CW; q++)
                        if (oW[q] != ~0u) {
                            if (gW[q] < 0) cpa16(ss + oW[q], mxt + (-1 - gW[q]));
                            else if (gb >= 0) cpa16(ss + oW[q], lpt + gW[q]);
                        }
                }
                cpa_mbar_arrive(&full[s]);
                CUTE_UNROLL
                for (int q = 0; q < CA; q++)
                    if (oA[q] != ~0u) {
                        half_t h[8];
                        if constexpr (BF16) {
                            const __nv_bfloat16* r = reinterpret_cast<const __nv_bfloat16*>(&ra[d][q][0]);
                            CUTE_UNROLL
                            for (int j = 0; j < 8; j++) h[j] = static_cast<half_t>(__bfloat162float(r[j]));
                        } else {
                            const float* r = reinterpret_cast<const float*>(&ra[d][q][0]);
                            CUTE_UNROLL
                            for (int j = 0; j < 8; j++) h[j] = static_cast<half_t>(r[j]);
                        }
                        *(float4*)(smem_raw + s * STAGE + oA[q]) = *(const float4*)h;
                    }
                if (!CWM && t == 0) {
                    bar_sync(2, NC + NP);                 // (the consumers' Mf)
                    CUTE_UNROLL
                    for (int q = 0; q < CW; q++) if (oW[q] != ~0u) mfv[q] = *(const float4*)(sMf + cW[q]);
                }
                const bool live = sGt[t] >= 0;
                CUTE_UNROLL
                for (int q = 0; q < CW; q++)
                    if (!CWM && oW[q] != ~0u) {
                        const float* vm = reinterpret_cast<const float*>(&rm[d % RV][q]);
                        const float* vl = reinterpret_cast<const float*>(&rl[d % RV][q]);
                        const float* M = reinterpret_cast<const float*>(&mfv[q]);
                        float wv[4];
                        CUTE_UNROLL
                        for (int k = 0; k < 4; k++) {
                            const float v = vm[k] + vl[k];
                            wv[k] = (!live || v == -INFINITY || M[k] == -INFINITY) ? 0.f : __expf(v - M[k]);
                        }
                        *(float4*)(smem_raw + s * STAGE + oW[q]) = *(const float4*)wv;
                    }
                mbar_arrive(&full[s]);                   // (the parameters' and w's stores, released)
                loadP(t + PE, ra[d]);
                loadV(t + PE, rm[d % RV], rl[d % RV]);
            }
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    // ---------------- consumers ----------------
    const int g = tid / NCG, gtid = tid % NCG;
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(gtid);
    Tensor sAt0 = make_tensor(make_smem_ptr(stA(0)), sAl);
    Tensor sBt0 = make_tensor(make_smem_ptr(stB(0)), sBl);
    Tensor tCrA = thr.partition_fragment_A(sAt0);
    Tensor tCrB = thr.partition_fragment_B(sBt0);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, half_t>{}, mma);
    auto tA = s2rA.get_thread_slice(gtid);
    auto tB = s2rB.get_thread_slice(gtid);
    Tensor tXrA = tA.retile_D(tCrA);
    Tensor tXrB = tB.retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    clear(tCrL);
    constexpr int KB = size<2>(tCrA);
    constexpr int NE = size(tCrS);
    const int lgc = __ffs(gc) - 1;
    // an element's gate row among the tile's
    auto jr = [&](int i) { if constexpr (SG) return 0; else return ((tile_id * BM + get<0>(tCcC(i))) >> lgc) - r0; };
    float wr[NE], e2[NE];
    if constexpr (WGRAD) {
        CUTE_UNROLL
        for (int i = 0; i < NE; i++) e2[i] = emars[(off_ele + get<0>(tCcC(i))) * (long)batch + b0 + get<1>(tCcC(i))];
    }
    // Mf [gate row j, column b] over the tiles: a pair p = (j, b) to NC / P threads (P <= NC) or P / NC pairs to a
    // thread, their max in registers, then into shared memory, the floats ordered as integers (non-negative: as int;
    // negative: reversed, as unsigned). Its loads see the loaded latency of the other CTAs' streams: they are issued
    // U tiles at a time.
    {
        const int P = nwr * BN;
        unsigned* sMu = reinterpret_cast<unsigned*>(sMf);
        for (int i = tid; i < P; i += NC) sMu[i] = 0xff800000u;
        bar_sync(3, NC + NP);                           // (the gates, from the producers)
        const int TS = P >= NC ? 1 : NC / P;
        constexpr int U = 16;
        for (int p = (P >= NC ? tid : tid % P); p < P; p += NC) {
            const int j = p / BN, b = p % BN;
            const float* mxc = mx + tile0 * (long)batch + b0 + b;
            const float* exc = ext + (ext_base + r0 + j) * (long)batch + b0 + b;
            float m = -INFINITY;
            for (int tb = (P >= NC ? 0 : tid / P); tb < knt; tb += U * TS) {
                float v[U];
                CUTE_UNROLL
                for (int u = 0; u < U; u++) {
                    const int t = tb + u * TS;
                    v[u] = -INFINITY;
                    if (t < knt) {
                        const int gb = sGt[t];
                        if (gb >= 0) v[u] = mxc[(long)t * batch] + exc[(long)gb * batch];
                    }
                }
                CUTE_UNROLL
                for (int u = 0; u < U; u++) m = fmaxf(m, v[u]);
            }
            if (m >= 0.f) atomicMax(reinterpret_cast<int*>(sMu + p), __float_as_int(m));
            else if (m > -INFINITY) atomicMin(sMu + p, __float_as_uint(m));
        }
        bar_sync(1, NC);
        if constexpr (!CWM) bar_arrive(2, NC + NP);
    }
    if constexpr (WGRAD) {
        CUTE_UNROLL
        for (int i = 0; i < NE; i++) {
            const float M = sMf[jr(i) * BN + get<1>(tCcC(i))];
            e2[i] = (e2[i] == -INFINITY || M == -INFINITY) ? 0.f : __expf(e2[i] + M);
        }
    }

    for (int kt = g; kt < knt; kt += G) {
        int s = kt % ST;
        mbar_wait(&full[s], (kt / ST) & 1);
        Tensor sAt = make_tensor(make_smem_ptr(stA(s)), sAl);
        Tensor sBt = make_tensor(make_smem_ptr(stB(s)), sBl);
        auto tXsA = tA.partition_S(sAt);
        auto tXsB = tB.partition_S(sBt);
        clear(tCrS);
        copy(s2rA, tXsA(_, _, 0), tXrA(_, _, 0));
        copy(s2rB, tXsB(_, _, 0), tXrB(_, _, 0));
        const float* sv = stV(s);
        int gbk = 0;
        CUTE_UNROLL
        for (int kb = 0; kb < KB; kb++) {
            if (kb + 1 < KB) {
                copy(s2rA, tXsA(_, _, kb + 1), tXrA(_, _, kb + 1));
                copy(s2rB, tXsB(_, _, kb + 1), tXrB(_, _, kb + 1));
            } else {
                if constexpr (CWM) gbk = sGt[kt];
                CUTE_UNROLL
                for (int i = 0; i < NE; i++) {          // (rows m and m + 8 of a 16-row atom share a gate at gc >= 16)
                    if ((SG || gc >= 16) && (i % 4) >= 2) { wr[i] = wr[i - 2]; continue; }
                    const int b = get<1>(tCcC(i)), j = jr(i);
                    if constexpr (CWM) {
                        const float v = sv[b] + sv[(1 + j) * BN + b], M = sMf[j * BN + b];
                        wr[i] = (gbk < 0 || v == -INFINITY || M == -INFINITY) ? 0.f : __expf(v - M);
                    } else {
                        wr[i] = sv[(VR0 + j) * BN + b];
                    }
                }
                mbar_arrive(&empty[s]);
            }
            cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrS);
        }
        CUTE_UNROLL
        for (int i = 0; i < NE; i++) tCrL(i) = fmaf(wr[i], tCrS(i), tCrL(i));
        if constexpr (WGRAD) {
            // S w exp(emars + Mf), summed over a gate's children: across the lanes of a column whose rows share a gate
            // (g8's bits below log2 gc, at most all 8), then rows m and m + 8 of a 16-row atom (gc >= 16), one atomic each
            const int gb = sGt[kt];
            if (gb >= 0) {
                const int lane = gtid % 32;
                float c[NE];
                CUTE_UNROLL
                for (int i = 0; i < NE; i++) c[i] = tCrS(i) * wr[i] * e2[i];
                CUTE_UNROLL
                for (int o = 4; o <= 16; o *= 2)
                    if (SG || o / 4 < gc) {
                        CUTE_UNROLL
                        for (int i = 0; i < NE; i++) c[i] += __shfl_xor_sync(0xffffffffu, c[i], o);
                    }
                const bool leader = SG ? lane < 4 : ((lane / 4) % (gc < 8 ? gc : 8)) == 0;
                CUTE_UNROLL
                for (int i = 0; i < NE; i++) {
                    if ((SG || gc >= 16) && (i % 4) >= 2) continue;  // (added to its row - 8 partner)
                    const float ct = (SG || gc >= 16) ? c[i] + c[i + 2] : c[i];
                    const int b = get<1>(tCcC(i));
                    if (leader) atomicAdd(grad_ext + (gb + ext_base + r0 + jr(i)) * (long)batch + b0 + b, ct);
                }
            }
        }
    }
    // the groups' sums, in order
    if constexpr (G > 1) {
        for (int k = 1; k < G; k++) {
            if (g == k) {
                CUTE_UNROLL
                for (int i = 0; i < NE; i++) {
                    int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
                    sL[m * LS + b] = (k == 1) ? tCrL(i) : sL[m * LS + b] + tCrL(i);
                }
            }
            bar_sync(1, NC);
        }
        CUTE_UNROLL
        for (int i = 0; i < NE; i++) {
            int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
            if (g == 0) tCrL(i) += sL[m * LS + b];
        }
        if (g != 0) return;
    }
    CUTE_UNROLL
    for (int i = 0; i < NE; i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        const long off = (off_ele + m) * (long)batch + b0 + b;
        const float M = sMf[jr(i) * BN + b];
        const float L = tCrL(i);
        eflows[off] = (L > 0.f && M != -INFINITY) ? emars[off] + M + logf(L) : -INFINITY;
    }
}

static int gates_smem(int64_t knt) { return (int)(((knt * 4 + 15) / 16) * 16); }

template <int BM, int BN, int WM, int WN, int G, int PW, int ST, int PD, bool CWM, typename TP>
static void launch_cfg(bool wgrad, float* ef, const float* em, const TP* p, const long* chids, int neb, const long* pb,
                       const long* gate, const half_t* bx, const float* mx, const float* ext, float* grad, int batch,
                       int BSK, int BSM, int knt, int gc, long ext_base, cudaStream_t st) {
    constexpr int NTH = (G * WM * WN + PW) * 32;
    static_assert(smem_bytes<BM, BN, G, ST, CWM>() <= SMEM_MAX, "shared memory");
    const int smem = smem_bytes<BM, BN, G, ST, CWM>() + gates_smem(knt);
    TORCH_CHECK(smem <= SMEM_MAX, "too many k-tiles for the gates' shared memory");
    const bool sg = gc >= BM;
    auto k0 = sg ? bs_ele_stream_kernel<BM, BN, WM, WN, G, PW, ST, PD, CWM, false, true, TP>
                 : bs_ele_stream_kernel<BM, BN, WM, WN, G, PW, ST, PD, CWM, false, false, TP>;
    auto k1 = sg ? bs_ele_stream_kernel<BM, BN, WM, WN, G, PW, ST, PD, CWM, true, true, TP>
                 : bs_ele_stream_kernel<BM, BN, WM, WN, G, PW, ST, PD, CWM, true, false, TP>;
    static bool attr_set[2][64] = {};
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev >= 64 || !attr_set[sg][dev]) {                // (the largest a launch may ask: the gates' share varies)
        C10_CUDA_CHECK(cudaFuncSetAttribute(k0, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        C10_CUDA_CHECK(cudaFuncSetAttribute(k1, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        if (dev < 64) attr_set[sg][dev] = true;
    }
    int total_m = neb * (BSM / BM);
    for (int off = 0; off < total_m; off += 65535) {
        int chunk = (total_m - off < 65535) ? (total_m - off) : 65535;
        dim3 grid(batch / BN, chunk);
        if (wgrad) k1<<<grid, NTH, smem, st>>>(ef, em, p, chids, pb, gate, bx, mx, ext, grad, batch, BSK, BSM, knt, gc,
                                               ext_base, off);
        else k0<<<grid, NTH, smem, st>>>(ef, em, p, chids, pb, gate, bx, mx, ext, grad, batch, BSK, BSM, knt, gc,
                                         ext_base, off);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// id -> {BM, BN, WM, WN, G, PW, ST, PD, CWM}: PD the parameters' register prefetch depth (tiles; at fp32 half),
// CWM w by the consumers
#define CFG_TABLE(X)                         \
    X(0, 32, 32, 2, 1, 4, 4, 8, 3, 0)        \
    X(1, 32, 64, 2, 2, 4, 4, 5, 2, 0)        \
    X(2, 64, 64, 4, 2, 1, 4, 4, 1, 0)        \
    X(3, 64, 64, 4, 2, 2, 4, 3, 1, 0)        \
    X(4, 32, 64, 2, 2, 4, 4, 5, 1, 0)        \
    X(5, 64, 64, 4, 2, 2, 4, 3, 2, 0)        \
    X(6, 32, 32, 2, 1, 4, 4, 8, 1, 0)        \
    X(7, 32, 32, 2, 1, 4, 4, 8, 3, 1)        \
    X(8, 32, 64, 2, 2, 4, 4, 5, 2, 1)        \
    X(9, 32, 32, 2, 1, 4, 4, 8, 1, 1)

template <typename TP>
static void dispatch_cfg(int64_t cfg, bool wg, float* ef, const float* em, const TP* p, const long* chids, int neb,
                         const long* pb, const long* gate, const half_t* bx, const float* mx, const float* ext,
                         float* grad, int batch, int BSK, int BSM, int knt, int gc, long eb, cudaStream_t st) {
    switch (cfg) {
#define X(id, bm, bn, wm, wn, gg, pw, stg, pd, cw)                                                                   \
        case id: launch_cfg<bm, bn, wm, wn, gg, pw, stg, pd, cw != 0, TP>(wg, ef, em, p, chids, neb, pb, gate, bx, mx, \
                                                                          ext, grad, batch, BSK, BSM, knt, gc, eb, st); \
                 break;
        CFG_TABLE(X)
#undef X
        default: TORCH_CHECK(false, "no such config");
    }
}

std::vector<std::vector<int64_t>> configs() {
    std::vector<std::vector<int64_t>> out;
#define X(id, bm, bn, wm, wn, gg, pw, stg, pd, cw) out.push_back({bm, bn});
    CFG_TABLE(X)
#undef X
    return out;
}

// The config ids that serve a layer: the tile divides the element block and the batch, the gates fit in shared
// memory, and the layer is in scope (batch % 64, a power-of-two gate width >= 4).
std::vector<int64_t> fitting_configs(int64_t cs_block_size, int64_t batch, int64_t gate_cbs, int64_t knt) {
    std::vector<int64_t> out;
    if (batch % PB != 0 || gate_cbs < 4 || (gate_cbs & (gate_cbs - 1)) != 0 || knt < 1) return out;
#define X(id, bm, bn, wm, wn, gg, pw, stg, pd, cw)                                                                   \
    if (cs_block_size % bm == 0 && batch % bn == 0 &&                                                                \
        smem_bytes<bm, bn, gg, stg, cw != 0>() + gates_smem(knt) <= SMEM_MAX)                                        \
        out.push_back(id);
    CFG_TABLE(X)
#undef X
    return out;
}

// The CuTe fork's arguments (`blockscale_ele_backward`), and whether to write the gradient and the config.
void blockscale_ele_backward_prepass(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                                     torch::Tensor node_mars, torch::Tensor params, torch::Tensor ext,
                                     torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase, torch::Tensor gate,
                                     torch::Tensor grad_ext, int64_t batch, int64_t block_size, int64_t cs_block_size,
                                     int64_t knt, int64_t gate_cbs, int64_t ext_base, int64_t write_grad, int64_t cfg) {
    TORCH_CHECK(batch % PB == 0 && gate_cbs >= 4 && (gate_cbs & (gate_cbs - 1)) == 0,
                "scope: batch % 64, a power-of-two gate width >= 4");
    TORCH_CHECK(chids.scalar_type() == torch::kLong && ebase.scalar_type() == torch::kLong &&
                pbase.scalar_type() == torch::kLong && gate.scalar_type() == torch::kLong, "int64 tables");
    const int neb = (int)chids.size(0);
    TORCH_CHECK(ebase.is_contiguous() && pbase.is_contiguous() && gate.is_contiguous() && ebase.numel() == neb * knt &&
                pbase.numel() == neb * knt && gate.numel() == neb * knt, "the tables: [element blocks, k-tiles]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(ext.data_ptr()) % 16 == 0, "the gate parameters 16-byte aligned");
    TORCH_CHECK(!write_grad || grad_ext.numel() > 0, "write_grad without a gradient buffer");
    const long tiles = ebase.numel();
    const long o_mx = tiles * BK * batch * 2;
    torch::Tensor scratch = torch::empty({o_mx + tiles * batch * 4}, element_mars.options().dtype(torch::kUInt8));
    char* base = reinterpret_cast<char*>(scratch.data_ptr());
    auto* bx = reinterpret_cast<half_t*>(base);
    auto* mx = reinterpret_cast<float*>(base + o_mx);
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    bs_ele_prepass_kernel<<<dim3(tiles, batch / PB), 256, 0, st>>>(node_flows.data_ptr<float>(),
                                                                   node_mars.data_ptr<float>(), ebase.data_ptr<long>(),
                                                                   bx, mx, (int)batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    float* grad = write_grad ? grad_ext.data_ptr<float>() : nullptr;
    if (params.scalar_type() == torch::kBFloat16)
        dispatch_cfg<__nv_bfloat16>(cfg, write_grad != 0, element_flows.data_ptr<float>(),
                                    element_mars.data_ptr<float>(), param_ptr<__nv_bfloat16>(params),
                                    chids.data_ptr<long>(), neb, pbase.data_ptr<long>(), gate.data_ptr<long>(), bx, mx,
                                    ext.data_ptr<float>(), grad, (int)batch, (int)block_size, (int)cs_block_size,
                                    (int)knt, (int)gate_cbs, (long)ext_base, st);
    else
        dispatch_cfg<float>(cfg, write_grad != 0, element_flows.data_ptr<float>(), element_mars.data_ptr<float>(),
                            param_ptr<float>(params), chids.data_ptr<long>(), neb, pbase.data_ptr<long>(),
                            gate.data_ptr<long>(), bx, mx, ext.data_ptr<float>(), grad, (int)batch, (int)block_size,
                            (int)cs_block_size, (int)knt, (int)gate_cbs, (long)ext_base, st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_ele_backward_prepass", &blockscale_ele_backward_prepass,
          "the block-scale element-flow backward (and d LL / d log phi) as a pre-pass and a streaming kernel");
    m.def("configs", &configs, "(BM, BN) per config id");
    m.def("fitting_configs", &fitting_configs,
          "the config ids that serve a layer (cs_block_size, batch, gate_cbs, K_NUM_TILES)");
}
