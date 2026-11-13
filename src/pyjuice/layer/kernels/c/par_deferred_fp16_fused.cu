// The capture of a tied group's deferred parameter flows (par_deferred_fp16.cu, `pyjuice.layer.deferred_par`) with
// the PRODUCT LAYER BELOW FUSED into its B operand: an edge's element value is formed as the sum of that product's two
// child node values (`ctab` int32 [elements past `first_row`, 2], their node rows; -inf below `first_row`, the dummy
// rows) -- what the product layer's forward recomputes into `element_mars` before the sum layer's backward, one fp32
// add -- instead of read from `element_mars`, so the capture needs no recompute (the sum layer's element flows fused
// likewise, ele_backward_sum_stream_fused.cu). The same operands to the bit; the rest is that module's capture, copied
// (a new module: the kernels training runs stay as they are), and its flush adds what this capture stored. What that
// module's header says of the capture holds here.

#include <torch/extension.h>
#include "param_load.cuh"                                   // `param_ptr<TP>`, `ParamLoad<TP>`: the stored dtype
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>
#include <vector>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

// ---------------- capture ----------------
// The column maxima M[nb, b] = max_n lf as order-preserving integers (0 below every float, -inf included; the buffer
// zeroed first): a block per (node block, 32 columns, 64 nodes), 8 warps x 8 nodes, lane = column, its partial max
// combined by atomicMax
__device__ __forceinline__ unsigned f2ord(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord2f(unsigned o) {
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}

__global__ void __launch_bounds__(256) capturef_m_kernel(const float* __restrict__ nflows, const float* __restrict__ nmars,
                                                        const long* __restrict__ nids, unsigned* __restrict__ Mord,
                                                        int blk, int batch, int Mp) {
    __shared__ float sm[8][32];
    const int nb = blockIdx.x, b = blockIdx.y * 32 + threadIdx.x % 32, w = threadIdx.x / 32, n0 = blockIdx.z * 64 + w * 8;
    const long base = nids[nb];
    float nf[8], nm[8];
#pragma unroll
    for (int u = 0; u < 8; u++) {
        nm[u] = -INFINITY; nf[u] = 0.f;
        if (b < batch && n0 + u < blk) {
            const long off = (base + n0 + u) * (long)batch + b;
            nm[u] = nmars[off]; nf[u] = nflows[off];
        }
    }
    float m = -INFINITY;
#pragma unroll
    for (int u = 0; u < 8; u++) m = fmaxf(m, nm[u] == -INFINITY ? -INFINITY : nf[u] - nm[u]);
    sm[w][threadIdx.x % 32] = m;
    __syncthreads();
    if (w == 0) {
#pragma unroll
        for (int q = 1; q < 8; q++) m = fmaxf(m, sm[q][threadIdx.x]);
        if (m > -INFINITY) atomicMax(Mord + (long)nb * Mp + b, f2ord(m));
    }
}

// A power of two from its exponent, e in [-126, 127] (a normal float)
__device__ __forceinline__ float pow2i(int e) { return __int_as_float((e + 127) << 23); }

// Both operands' rows of one slot (one chunk of a step's columns): a block per (node block, 8 R rows, chunk), a warp
// per R rows -- below blk A's (nodes), the rest Bop's (edges); blk % 64 == 0, so a block is all A or all Bop -- lane
// = columns lane + 32 j (j < W / 32 <= VPL); the R rows' index loads first, then all their values; the rows' maxima
// by warp shuffles, then the fp16 values and the rows' scales (the R the caller's: as many rows a warp as keep two
// blocks an SM)
template <int R, int VPL>
__global__ void __launch_bounds__(256) capturef_ab_kernel(
        const float* __restrict__ nflows, const float* __restrict__ nmars, const int* __restrict__ ctab, long first_row,
        const float* __restrict__ ext, const long* __restrict__ nids, const long* __restrict__ cids,
        const long* __restrict__ gate, const unsigned* __restrict__ Mord, __half* __restrict__ A, __half* __restrict__ Bop,
        float* __restrict__ fA, float* __restrict__ fB, int blk, int ne, int batch, int Mp, long Ks, int W, int S,
        int slot0, int node_cbs, int gate_cbs, int gate_stride, long ext_base, bool gated) {
    const int nb = blockIdx.x, lane = threadIdx.x % 32, r0 = (blockIdx.y * 8 + threadIdx.x / 32) * R, c = blockIdx.z;
    const int vpl = W / 32, cb = c * W, slot = slot0 + c;
    float M[VPL];
#pragma unroll
    for (int j = 0; j < VPL; j++) {
        M[j] = 0.f;
        if (j < vpl) {
            const unsigned mo = Mord[(long)nb * Mp + cb + lane + 32 * j];
            M[j] = (mo <= f2ord(-INFINITY)) ? 0.f : ord2f(mo);        // (0 where every node is -inf)
        }
    }
    float v[R][VPL];
    if (r0 < blk) {
        const long base = nids[nb] + r0;
        float nf[R][VPL], nm[R][VPL];
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) {
                const int b = cb + lane + 32 * j;
                nm[u][j] = -INFINITY; nf[u][j] = 0.f;
                if (j < vpl && b < batch) {
                    const long off = (base + u) * (long)batch + b;
                    nm[u][j] = nmars[off]; nf[u][j] = nflows[off];
                }
            }
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) v[u][j] = nm[u][j] == -INFINITY ? 0.f : __expf(nf[u][j] - nm[u][j] - M[j]);
    } else {
        long cid[R], grow[R];                                            // (grow: -2 = no gate table, log phi 0)
#pragma unroll
        for (int u = 0; u < R; u++) {
            const int e = r0 - blk + u, gcol = e / node_cbs;
            cid[u] = cids[(long)nb * ne + e];
            if (gated) {
                const long gb = gcol < gate_stride ? gate[(long)nb * gate_stride + gcol] : -1;
                grow[u] = gb < 0 ? -1 : gb + ext_base + (e % node_cbs) / gate_cbs;
            } else {
                grow[u] = -2;
            }
        }
        float em[R][VPL], lp[R][VPL];
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) {
                const int b = cb + lane + 32 * j;
                em[u][j] = -INFINITY; lp[u][j] = 0.f;
                if (j < vpl && b < batch && grow[u] != -1 && cid[u] >= first_row) {    // (below: dummy rows, -inf)
                    const int* ct = ctab + (cid[u] - first_row) * 2;
                    em[u][j] = nmars[(long)ct[0] * batch + b] + nmars[(long)ct[1] * batch + b];
                    if (grow[u] >= 0) lp[u][j] = ext[grow[u] * (long)batch + b];
                }
            }
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int j = 0; j < VPL; j++) {
                const float x = em[u][j] + lp[u][j] + M[j];
                v[u][j] = x == -INFINITY ? 0.f : __expf(x);
            }
    }
    int xr[R];
#pragma unroll
    for (int u = 0; u < R; u++) {
        float m = 0.f;
#pragma unroll
        for (int j = 0; j < VPL; j++) m = fmaxf(m, v[u][j]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        // m = f 2^x, f in [0.5, 1) (m is 0 or normal: the exponentials flush to zero)
        xr[u] = m > 0.f ? (int)((__float_as_uint(m) >> 23) & 0xffu) - 126 : 0;
    }
#pragma unroll
    for (int u = 0; u < R; u++) {
        const int r = r0 + u;
        __half* out = (r < blk ? A + ((long)nb * blk + r) * Ks : Bop + ((long)nb * ne + (r - blk)) * Ks) + (long)slot * W;
        // v 2^(15 - x), 15 - x in [-114, 140]: by two normal powers of two, exact (the product at most 2^15)
        const int e = 15 - xr[u], e1 = min(e, 127);
        const float f1 = pow2i(e1), f2 = pow2i(e - e1);
#pragma unroll
        for (int j = 0; j < VPL; j++)
            if (j < vpl) out[lane + 32 * j] = __float2half_rn(v[u][j] * f1 * f2);
        if (lane == 0) {
            const float s = ldexpf(1.f, xr[u]);
            if (r < blk) fA[((long)nb * S + slot) * blk + r] = s;
            else fB[((long)nb * S + slot) * ne + (r - blk)] = s;
        }
    }
}

void capture_fused(torch::Tensor A, torch::Tensor Bop, torch::Tensor fA, torch::Tensor fB, torch::Tensor M,
                   torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor ctab, int64_t first_row, torch::Tensor ext,
             torch::Tensor nids, torch::Tensor cids, torch::Tensor gate, int64_t batch, int64_t slot0, int64_t node_cbs,
             int64_t gate_cbs, int64_t ext_base, bool gated) {
    const int nbk = (int)A.size(0), blk = (int)A.size(1), ne = (int)Bop.size(1), S = (int)fA.size(1);
    const long Ks = A.size(2);
    const int W = (int)(Ks / S), Mp = (int)M.size(1), cps = Mp / W;
    TORCH_CHECK(A.scalar_type() == torch::kHalf && Bop.scalar_type() == torch::kHalf && M.scalar_type() == torch::kInt32 &&
                Ks % S == 0 && W % 32 == 0 && W <= 512 && Mp % W == 0 && Mp >= batch && slot0 + cps <= S &&
                blk % 64 == 0 && ne % 64 == 0 && Bop.size(0) == A.size(0) && fB.size(0) == A.size(0),
                "capture: the slot layout (each node block its own operand)");
    TORCH_CHECK(ctab.scalar_type() == torch::kInt32 && ctab.is_contiguous() && ctab.dim() == 2 && ctab.size(1) == 2,
                "capture fused: ctab int32 [rows, 2]");
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(M.data_ptr(), 0, M.numel() * 4, st));
    auto* Mord = reinterpret_cast<unsigned*>(M.data_ptr());
    capturef_m_kernel<<<dim3(nbk, Mp / 32, blk / 64), 256, 0, st>>>(node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
                                                                   nids.data_ptr<long>(), Mord, blk, (int)batch, Mp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // rows a warp: the most (up to 32 values a lane) that keep two blocks an SM
    int dev = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int vmax = W <= 128 ? 4 : W <= 256 ? 8 : 16;
    int R = 1;
    while (R < 32 / vmax && R < 8 && (long)nbk * ((blk + ne) / (16 * R)) * cps >= 2L * sms) R *= 2;
#define CAP(R_, VPL)                                                                                                   \
    capturef_ab_kernel<R_, VPL><<<dim3(nbk, (blk + ne) / (8 * R_), cps), 256, 0, st>>>(                                \
        node_flows.data_ptr<float>(), node_mars.data_ptr<float>(), ctab.data_ptr<int>(), (long)first_row,           \
        ext.data_ptr<float>(),                                                                                         \
        nids.data_ptr<long>(), cids.data_ptr<long>(), gate.data_ptr<long>(), Mord,                                     \
        reinterpret_cast<__half*>(A.data_ptr<at::Half>()), reinterpret_cast<__half*>(Bop.data_ptr<at::Half>()),        \
        fA.data_ptr<float>(), fB.data_ptr<float>(), blk, ne, (int)batch, Mp, Ks, W, S, (int)slot0, (int)node_cbs,      \
        (int)gate_cbs, gated ? (int)gate.size(1) : 0, (long)ext_base, gated)
    if (vmax == 4) {
        if (R == 1) CAP(1, 4); else if (R == 2) CAP(2, 4); else if (R == 4) CAP(4, 4); else CAP(8, 4);
    } else if (vmax == 8) {
        if (R == 1) CAP(1, 8); else if (R == 2) CAP(2, 8); else CAP(4, 8);
    } else {
        if (R == 1) CAP(1, 16); else CAP(2, 16);
    }
#undef CAP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("capture_fused", &capture_fused, "capture, the edges' element values formed from their products' children");
}
