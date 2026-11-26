// pyjuice: the tilt products of `LowRank(top_k)` soft-evidence leaves (`lowrank/top_k_leaf_fused.py`) -- a node
// vector's tensors A [Bx, M, re] and Bt [Bx, re, S] (the leaf's U and D) a position t:
//
//   tk_tau_kernel        the forward's tilts, the leaves' kernels' operand tau [T, Bx, S, M] (log2 units):
//                            tau[t, b, s, i] = sum_r (log2 e Bt_t[b, r, s]) A_t[b, i, r]
//   tk_tilt_grad_kernel  their gradients from dtau [T, Bd, S, M] (d / d tau; a row of destinations each, e the row of
//                        tensors destination row b reads):
//                            dA_t[b, i, r] = sum_s D[t, b, s, i] Bt_t[e, r, s]
//                            dBt_t[b, r, s] = sum_i A_t[e, i, r] D[t, b, s, i]
//                        STORED where the launch writes its destinations first (every element of them), else added
//
// A launch takes a run of positions whose tensors share their strides, each tensor's address an argument (up to MAXT
// positions a launch: nothing uploaded, no synchronization). Both kernels are memory-bound -- tau 13 GFLOP against its
// 0.8 GB, the gradients 26 against 1.2 GB at the best Codpula head (48 blocks) --: exact fp32 by FMAs on the CUDA
// cores, every sum in a fixed order (no tensor cores, no atomics). re <= 64 ranks, S <= 65 slots.
//
// MEASURED there (RTX PRO 6000): tau 0.60 ms (cuBLAS's products with their scale 0.97; 0.58 with no FMAs at all), the
// gradients 0.92 ms storing (the earlier kernel's adds 1.92 + 0.28 of zero fill; 0.86 with no FMAs at all). Against
// float64: tau 1.2e-7 of its largest (cuBLAS's 4.3e-7), dA 4e-7, dBt 1.3e-6 (a sum over 1024 nodes).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

constexpr float LOG2E = 1.4426950408889634f;
constexpr int TR = 64, TS = 65;            // ranks, slots (at most)
constexpr int MAXT = 64;                   // positions a launch

__device__ __forceinline__ void cp16(float* dst, const float* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ float4 lds4(const float* p) { return *reinterpret_cast<const float4*>(p); }

// ------------------------------------------------------------------------------------------------ the forward's tilts

// A CTA a (position, row, 128 nodes), two lanes a node: its row of A in their registers (lane h of a pair the rank
// quads 2j + h: the pair's loads one 32-B sector), log2 e Bt's slots in shared memory (a slot's ranks one row: two
// addresses a warp), a slot's dot at a time -- the lanes' halves joined by a shuffle -- stored as made. A lane's ranks'
// FMAs in four chains (r mod 4), joined (r0 + r1) + (r2 + r3), then the two halves.
constexpr int TAU_T = 256;                 // threads a CTA
constexpr int TAU_NB = TAU_T / 2;          // nodes a CTA
constexpr int BTS = TR + 4;                // Bt's staged rows (16-B aligned)

struct TauArgs {
    const float* A[MAXT]; const float* Bt[MAXT];             // a position's tensors
    long sab, sam, sar, sbb, sbr, sbs;
    float* tau;                                              // [T, Bx, S, M]
    int Bx, M, re, S;
};

template <bool VEC>                                          // VEC: A's rows 16-B aligned, ranks contiguous, re % 4 = 0
__global__ void __launch_bounds__(TAU_T, 3) tk_tau_kernel(const TauArgs a) {
    __shared__ __align__(16) float Bs[TS][BTS];
    const int nbk = a.M / TAU_NB, q = blockIdx.x, h = threadIdx.x & 1;
    const int t = q / (a.Bx * nbk), b = q / nbk % a.Bx, i = q % nbk * TAU_NB + (threadIdx.x >> 1);
    const float* A = a.A[t] + (long)b * a.sab + (long)i * a.sam;
    const float* Bt = a.Bt[t] + (long)b * a.sbb;
    const int re = a.re, S = a.S;
    float4 av[TR / 8];
#pragma unroll
    for (int j = 0; j < TR / 8; ++j) {
        const int c = 2 * j + h;
        if constexpr (VEC) {
            av[j] = 4 * c < re ? __ldg(reinterpret_cast<const float4*>(A) + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        } else {
            float x[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) x[u] = 4 * c + u < re ? __ldg(A + (long)(4 * c + u) * a.sar) : 0.f;
            av[j] = make_float4(x[0], x[1], x[2], x[3]);
        }
    }
    if (a.sbs == 1) {                                        // (read in the order Bt lies in)
        for (int k = threadIdx.x; k < S * re; k += TAU_T) {
            const int r = k / S, s = k % S;
            Bs[s][r] = LOG2E * __ldg(Bt + (long)r * a.sbr + s);
        }
    } else {
        for (int k = threadIdx.x; k < S * re; k += TAU_T) {
            const int s = k / re, r = k % re;
            Bs[s][r] = LOG2E * __ldg(Bt + (long)r * a.sbr + (long)s * a.sbs);
        }
    }
    for (int k = threadIdx.x; k < S * (TR - re); k += TAU_T) Bs[k / (TR - re)][re + k % (TR - re)] = 0.f;
    __syncthreads();
    float* out = a.tau + ((long)t * a.Bx + b) * S * a.M + i;
    for (int s = 0; s < S; ++s) {
        float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;
#pragma unroll
        for (int j = 0; j < TR / 8; ++j) {
            const float4 bv = lds4(&Bs[s][4 * (2 * j + h)]);
            x0 = fmaf(av[j].x, bv.x, x0); x1 = fmaf(av[j].y, bv.y, x1);
            x2 = fmaf(av[j].z, bv.z, x2); x3 = fmaf(av[j].w, bv.w, x3);
        }
        float v = (x0 + x1) + (x2 + x3);
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        if (h == 0) out[(long)s * a.M] = v;
    }
}

// ------------------------------------------------------------------------------------------------ their gradients

// A CTA a (position, row): its nodes in chunks of 64, D's chunk [S][64] and A's [64][re] staged by cp.async a chunk
// ahead (two buffers), Bt^T [S][re] once. dA's chunk (nodes x ranks, over the slots) written as made; dBt's sums (ranks
// x slots, over the nodes) kept over the chunks, written at the end. A warp's tiles: dA 16 nodes x 32 ranks (a lane
// 4 x 4), dBt 16 ranks x 32 slots (a lane 4 x 4) with the nodes four at a time, the 65th slot's column a thread a rank
// and a quarter of the nodes. D's node quads XOR-swizzled by (slot / 4) mod 8: the dBt reads (8 slots' quads a node
// quad) and the dA reads (a slot's 4 quads) both conflict-free. FMAs in slot (dA) and node (dBt) order.
constexpr int GC = 64;                     // nodes a chunk
constexpr int GT = 256;                    // threads a CTA
constexpr int GRAD_SMEM = (TS * BTS + 2 * TS * GC + 2 * GC * TR) * 4;   // Bt^T, D's chunks, A's chunks

struct GradArgs {
    const float* D;                                          // [T, Bd, S, M]
    const float* A[MAXT]; const float* Bt[MAXT]; float* dA[MAXT]; float* dB[MAXT];      // a position's tensors
    const int* emap;                                         // [Bd] the row of tensors a row reads (null: its own)
    long sab, sam, sar, sbb, sbr, sbs, sdab, sdam, sdar, sdbb, sdbr, sdbs;
    int Bd, M, re, S, store;
};

__device__ __forceinline__ int swz(int q, int s) { return q ^ ((s >> 2) & 7); }   // node quad q's place in D's row s

template <bool VEC>                                          // VEC: A's and dA's rows 16-B aligned, ranks contiguous
__global__ void __launch_bounds__(GT) tk_tilt_grad_kernel(const GradArgs a) {
    extern __shared__ __align__(16) float sm[];
    float (*BtT)[BTS] = reinterpret_cast<float (*)[BTS]>(sm);                            // [s][r]
    float (*Ds)[TS][GC] = reinterpret_cast<float (*)[TS][GC]>(sm + TS * BTS);            // [stage][s][node] (swizzled)
    float (*As)[GC][TR] = reinterpret_cast<float (*)[GC][TR]>(sm + TS * BTS + 2 * TS * GC);   // [stage][node][r]
    const int tb = blockIdx.x, t = tb / a.Bd, b = tb % a.Bd;
    const int e = a.emap != nullptr ? a.emap[b] : b;
    const float* A = a.A[t] + (long)e * a.sab;
    const float* Bt = a.Bt[t] + (long)e * a.sbb;
    float* dA = a.dA[t] + (long)b * a.sdab;
    float* dB = a.dB[t] + (long)b * a.sdbb;
    const int re = a.re, S = a.S, M = a.M;
    const float* D = a.D + ((long)t * a.Bd + b) * S * M;
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31;
    const bool store = a.store != 0;

    const int nq = re / 4;                                   // (VEC: A's 16-B quads a node)
    auto issue = [&](int st, int i0) {
        for (int k = tid; k < S * (GC / 4); k += GT) {
            const int s = k / (GC / 4), c = k % (GC / 4);
            cp16(&Ds[st][s][swz(c, s) * 4], D + (long)s * M + i0 + 4 * c);
        }
        if constexpr (VEC) {
            for (int k = tid; k < GC * nq; k += GT) {
                const int n = k / nq, c = k % nq;
                cp16(&As[st][n][4 * c], A + (long)(i0 + n) * a.sam + 4 * c);
            }
        }
    };

    const int nA = (w >> 1) * 4 + (l >> 3), rA = (w & 1) * 8 + (l & 7);      // dA: node quad, rank quad
    const int rB = (w >> 1) * 4 + (l >> 3), sB = (w & 1) * 8 + (l & 7);      // dBt: rank quad, slot quad
    const int r64 = tid & 63, p64 = tid >> 6;                                 // the 65th slot: rank, quarter of nodes
    const bool liveA = 4 * rA < re;
    float gB[4][4], g64 = 0.f;
#pragma unroll
    for (int x = 0; x < 4; ++x) gB[x][0] = gB[x][1] = gB[x][2] = gB[x][3] = 0.f;

    const int nck = M / GC;
    issue(0, 0);
    cp_commit();
    // Bt^T (read in the order Bt lies in) and what no copy writes: Bt^T past (S, re), D's rows past S, A's ranks past
    // re (zeros)
    if (a.sbs == 1) {
        for (int k = tid; k < S * re; k += GT) {
            const int r = k / S, s = k % S;
            BtT[s][r] = __ldg(Bt + (long)r * a.sbr + s);
        }
    } else {
        for (int k = tid; k < S * re; k += GT) {
            const int s = k / re, r = k % re;
            BtT[s][r] = __ldg(Bt + (long)r * a.sbr + (long)s * a.sbs);
        }
    }
    for (int k = tid; k < TS * TR; k += GT) {
        const int s = k / TR, r = k % TR;
        if (s >= S || r >= re) BtT[s][r] = 0.f;
    }
    for (int k = tid; k < 2 * (TS - S) * GC; k += GT)
        Ds[k / ((TS - S) * GC)][S + k % ((TS - S) * GC) / GC][k % GC] = 0.f;
    for (int k = tid; k < 2 * GC * (TR - re); k += GT) {
        const int st = k / (GC * (TR - re)), n = k % (GC * (TR - re)) / (TR - re), r = re + k % (TR - re);
        As[st][n][r] = 0.f;
    }
    for (int k = 0; k < nck; ++k) {
        const int st = k & 1, i0 = k * GC;
        if (k + 1 < nck) {
            issue(st ^ 1, i0 + GC);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        if constexpr (!VEC) {
            for (int q = tid; q < GC * re; q += GT) {
                const int n = q / re, r = q % re;
                As[st][n][r] = __ldg(A + (long)(i0 + n) * a.sam + (long)r * a.sar);
            }
        }
        __syncthreads();

        // dA's chunk: nodes 4 nA .. + 3, ranks 4 rA .. + 3
        {
            float* o = dA + (long)(i0 + 4 * nA) * a.sdam + (long)(4 * rA) * a.sdar;
            float g[4][4];
            if (store || !liveA) {
#pragma unroll
                for (int u = 0; u < 4; ++u) g[u][0] = g[u][1] = g[u][2] = g[u][3] = 0.f;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if constexpr (VEC) {
                        const float4 v = *reinterpret_cast<const float4*>(o + (long)u * a.sdam);
                        g[u][0] = v.x; g[u][1] = v.y; g[u][2] = v.z; g[u][3] = v.w;
                    } else {
#pragma unroll
                        for (int v = 0; v < 4; ++v)
                            g[u][v] = 4 * rA + v < re ? o[(long)u * a.sdam + (long)v * a.sdar] : 0.f;
                    }
                }
            }
#pragma unroll 5
            for (int s = 0; s < S; ++s) {
                const float4 d = lds4(&Ds[st][s][swz(nA, s) * 4]);
                const float4 bt = lds4(&BtT[s][4 * rA]);
                const float dv[4] = {d.x, d.y, d.z, d.w}, bv[4] = {bt.x, bt.y, bt.z, bt.w};
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int v = 0; v < 4; ++v) g[u][v] = fmaf(dv[u], bv[v], g[u][v]);
            }
            if (liveA) {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if constexpr (VEC) {
                        *reinterpret_cast<float4*>(o + (long)u * a.sdam) =
                            make_float4(g[u][0], g[u][1], g[u][2], g[u][3]);
                    } else {
#pragma unroll
                        for (int v = 0; v < 4; ++v)
                            if (4 * rA + v < re) o[(long)u * a.sdam + (long)v * a.sdar] = g[u][v];
                    }
                }
            }
        }

        // dBt: ranks 4 rB .. + 3 x slots 4 sB .. + 3 over the chunk's nodes, four at a time
#pragma unroll 2
        for (int iq = 0; iq < GC / 4; ++iq) {
            float av[4][4], dv[4][4];                                         // [node][rank], [slot][node]
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float4 x = lds4(&As[st][4 * iq + v][4 * rB]);
                av[v][0] = x.x; av[v][1] = x.y; av[v][2] = x.z; av[v][3] = x.w;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int s = 4 * sB + u;
                const float4 x = lds4(&Ds[st][s][swz(iq, s) * 4]);
                dv[u][0] = x.x; dv[u][1] = x.y; dv[u][2] = x.z; dv[u][3] = x.w;
            }
#pragma unroll
            for (int v = 0; v < 4; ++v)
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int u = 0; u < 4; ++u) gB[x][u] = fmaf(av[v][x], dv[u][v], gB[x][u]);
        }
        if (S == TS) {                                                        // the 65th slot: rank r64, a node quarter
#pragma unroll 4
            for (int j = 0; j < GC / 4; ++j) {
                const int n = (GC / 4) * p64 + j;
                g64 = fmaf(As[st][n][r64], Ds[st][TS - 1][swz(n >> 2, TS - 1) * 4 + (n & 3)], g64);
            }
        }
        __syncthreads();
    }

    // dBt's tile; the 65th slot's quarters summed through a free buffer
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int r = 4 * rB + x;
        if (r >= re) continue;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int s = 4 * sB + u;
            if (s >= S || s >= TS - 1) continue;
            float* p = dB + (long)r * a.sdbr + (long)s * a.sdbs;
            *p = store ? gB[x][u] : *p + gB[x][u];
        }
    }
    if (S == TS) {
        float* red = &As[0][0][0];
        red[p64 * TR + r64] = g64;
        __syncthreads();
        if (tid < re) {
            const float v = ((red[tid] + red[TR + tid]) + red[2 * TR + tid]) + red[3 * TR + tid];
            float* p = dB + (long)tid * a.sdbr + (long)(TS - 1) * a.sdbs;
            *p = store ? v : *p + v;
        }
    }
}

void check(const torch::Tensor& t, at::ScalarType dt, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), "top_k tilt: ", name,
                " must be a contiguous CUDA tensor of ", c10::toString(dt));
}

}  // namespace

// tau [T, Bx, S, M] = log2 e (A_t Bt_t)^T of a run of positions sharing their strides (their tensors' addresses
// `ptrs` [2 T]: A_t [Bx, M, re] then Bt_t [Bx, re, S]; `strides` [6] A's, Bt's); `vec`: A's rows 16-B aligned with
// their ranks contiguous and re % 4 == 0
void top_k_leaf_tau(torch::Tensor tau, std::vector<int64_t> ptrs, std::vector<int64_t> strides, int64_t re, bool vec) {
    const at::cuda::CUDAGuard guard(tau.device());
    check(tau, at::kFloat, "tau");
    TORCH_CHECK(tau.dim() == 4 && (int64_t)ptrs.size() == 2 * tau.size(0) && strides.size() == 6,
                "top_k tilt: the tilts' operands");
    const int T = tau.size(0), Bx = tau.size(1), S = tau.size(2), M = tau.size(3);
    TORCH_CHECK(re <= TR && S <= TS && M % TAU_NB == 0, "top_k tilt: ranks <= 64, slots <= 65, nodes % 128");
    TauArgs a;
    a.sab = strides[0]; a.sam = strides[1]; a.sar = strides[2];
    a.sbb = strides[3]; a.sbr = strides[4]; a.sbs = strides[5];
    a.Bx = Bx; a.M = M; a.re = (int)re; a.S = S;
    auto stream = at::cuda::getCurrentCUDAStream();
    for (int t0 = 0; t0 < T; t0 += MAXT) {                   // (MAXT positions a launch)
        const int n = std::min(MAXT, T - t0);
        for (int k = 0; k < n; ++k) {
            a.A[k] = reinterpret_cast<const float*>(ptrs[t0 + k]);
            a.Bt[k] = reinterpret_cast<const float*>(ptrs[T + t0 + k]);
        }
        a.tau = tau.data_ptr<float>() + (long)t0 * Bx * S * M;
        const unsigned grid = (unsigned)((long)n * Bx * (M / TAU_NB));
        if (grid == 0) continue;
        if (vec) tk_tau_kernel<true><<<grid, TAU_T, 0, stream>>>(a);
        else tk_tau_kernel<false><<<grid, TAU_T, 0, stream>>>(a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

// The tilt's gradients of a run of positions (`D` [T, Bd, S, M] contiguous; their factors' and destinations' addresses
// `ptrs` [4 T]: A_t, Bt_t, dA_t, dBt_t; every position's strides `strides` [12]: A's, Bt's, dA's, dBt's), STORED into
// the destinations where `store` (their every element), else added; `emap` [Bd] int32 the row of tensors a destination
// row reads (empty: its own); `vec`: A's and dA's rows 16-B aligned with their ranks contiguous, re % 4 == 0
void top_k_leaf_tilt_grads(torch::Tensor D, std::vector<int64_t> ptrs, std::vector<int64_t> strides, torch::Tensor emap,
                           int64_t re, bool store, bool vec) {
    const at::cuda::CUDAGuard guard(D.device());
    check(D, at::kFloat, "D");
    TORCH_CHECK(D.dim() == 4 && (int64_t)ptrs.size() == 4 * D.size(0) && strides.size() == 12,
                "top_k tilt: the tilt's gradients' operands");
    const int T = D.size(0), Bd = D.size(1), S = D.size(2), M = D.size(3);
    TORCH_CHECK(re <= TR && S <= TS && M % GC == 0, "top_k tilt: ranks <= 64, slots <= 65, nodes % 64");
    GradArgs a;
    a.emap = nullptr;
    if (emap.numel() > 0) {
        check(emap, at::kInt, "emap");
        TORCH_CHECK(emap.numel() == Bd, "top_k tilt: emap [Bd]");
        a.emap = emap.data_ptr<int>();
    }
    a.sab = strides[0]; a.sam = strides[1]; a.sar = strides[2];
    a.sbb = strides[3]; a.sbr = strides[4]; a.sbs = strides[5];
    a.sdab = strides[6]; a.sdam = strides[7]; a.sdar = strides[8];
    a.sdbb = strides[9]; a.sdbr = strides[10]; a.sdbs = strides[11];
    a.Bd = Bd; a.M = M; a.re = (int)re; a.S = S; a.store = store ? 1 : 0;
    static bool done[64][2] = {};
    const int dev = D.get_device();
    if (!done[dev][vec]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(vec ? tk_tilt_grad_kernel<true> : tk_tilt_grad_kernel<false>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, GRAD_SMEM));
        done[dev][vec] = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    for (int t0 = 0; t0 < T; t0 += MAXT) {                   // (MAXT positions a launch)
        const int n = std::min(MAXT, T - t0);
        for (int k = 0; k < n; ++k) {
            a.A[k] = reinterpret_cast<const float*>(ptrs[t0 + k]);
            a.Bt[k] = reinterpret_cast<const float*>(ptrs[T + t0 + k]);
            a.dA[k] = reinterpret_cast<float*>(ptrs[2 * T + t0 + k]);
            a.dB[k] = reinterpret_cast<float*>(ptrs[3 * T + t0 + k]);
        }
        a.D = D.data_ptr<float>() + (long)t0 * Bd * S * M;
        if ((long)n * Bd == 0) continue;
        if (vec) tk_tilt_grad_kernel<true><<<n * Bd, GT, GRAD_SMEM, stream>>>(a);
        else tk_tilt_grad_kernel<false><<<n * Bd, GT, GRAD_SMEM, stream>>>(a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top_k_leaf_tau", &top_k_leaf_tau, "the leaves' tilts log2 e (A B)^T, a run of positions");
    m.def("top_k_leaf_tilt_grads", &top_k_leaf_tilt_grads, "the leaves' tilt gradients from dtau, a run of positions");
}
