// The capture of a tied group's deferred parameter flows (par_deferred_fp16.cu's `capture`, and
// par_deferred_fp16_fused.cu's `capture_fused`: the edges' element values formed from their products' two children) in
// fewer, shorter launches -- the same operands to the bit (`pyjuice.layer.deferred_par` takes it where it applies):
//   * the column maxima M a STEP of its own, `Ms` [steps, node blocks, Mp] zeroed by the caller before a group's
//     slots fill from the first one -- not a memset before every capture;
//   * the maxima's kernel and the operands' launched as a chain by programmatic dependent launch: the maxima wait for
//     the launch before them and let the operands' kernel launch at once; that one loads its rows' tables and values
//     (nothing either launch writes) and waits only before it reads the maxima;
//   * a lane takes VPL CONSECUTIVE columns of its row (vector loads, one store of its fp16 values) instead of every
//     32nd -- a row's max over its slot's columns is a max (any order), every value computed as those kernels do;
//   * SHARED (every node block's edges the same children: an HMM's blocks), one warp takes an edge's row of every node
//     block, its element values loaded once, not once a node block; the node blocks' rows QB at a time, their maxima's
//     shuffles interleaved.
// The maxima are what the launch before writes: the operands' kernel reads them after its wait through a pointer that
// is NOT __restrict__ -- coherent loads, which the compiler keeps after the wait. Through `const __restrict__` they were
// non-coherent ones (ld.global.nc: data read-only for the kernel's lifetime), and ptxas issued them ABOVE the wait
// (MEASURED at 16 columns a lane and 2 rows a warp: a slot's operands from maxima half made). Not L2-only loads either
// (ld.global.cg: MEASURED the operands' kernel 7.5 -> 15.3 us -- every warp reads the same few lines, all at once).
// A capture may run on a side stream (`side`, `pyjuice.layer.deferred_par`'s choice): the fork is an event recorded on the
// calling stream and waited for on the side one -- here, in C++, a few microseconds of host instead of Python's ~12.
// Scope (the caller checks it): a slot's columns W = 32 VPL with VPL 1, 2, 4, 8 or 16, batch % 4 == 0, blk % 64 == 0,
// num_edges % 64 == 0. What par_deferred_fp16.cu's header says of the operands holds here.
//
// MEASURED (RTX PRO 6000, HMM-4096 in 4 blocks B 128 bf16, one layer's capture as the training step left it, back to
// back): 16.4 us a capture (memset, maxima, operands) -> 5.8.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

__device__ __forceinline__ unsigned f2ord(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ord2f(unsigned o) {
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o);
}
// A power of two from its exponent, e in [-126, 127] (a normal float)
__device__ __forceinline__ float pow2i(int e) { return __int_as_float((e + 127) << 23); }

// ---------------- the column maxima ----------------
// capture_m_kernel's: M[nb, b] = max_n lf over the node block (order-preserving ints, atomicMax into zeros), a block per
// (node block, 32 columns, 64 nodes); waits for the launch before, then lets the operands' kernel launch
__global__ void __launch_bounds__(256) caps_m_kernel(const float* __restrict__ nflows, const float* __restrict__ nmars,
                                                    const long* __restrict__ nids, unsigned* __restrict__ Mord,
                                                    int blk, int batch, int Mp) {
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
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

// ---------------- a lane's VPL consecutive columns ----------------
// loads p[0, VPL) (16-byte pieces from VPL 4 on; `b` the first column, each piece wholly in or past the batch: batch
// % 4 == 0), `fill` past the batch
template <int VPL>
__device__ __forceinline__ void ldv(const float* p, int b, int batch, float fill, float* out) {
    if constexpr (VPL == 1) {
        out[0] = b < batch ? p[0] : fill;
    } else if constexpr (VPL == 2) {
        if (b < batch) { const float2 x = *reinterpret_cast<const float2*>(p); out[0] = x.x; out[1] = x.y; }
        else { out[0] = fill; out[1] = fill; }
    } else {
#pragma unroll
        for (int k = 0; k < VPL / 4; k++) {
            if (b + 4 * k < batch) {
                const float4 x = *reinterpret_cast<const float4*>(p + 4 * k);
                out[4 * k] = x.x; out[4 * k + 1] = x.y; out[4 * k + 2] = x.z; out[4 * k + 3] = x.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; i++) out[4 * k + i] = fill;
            }
        }
    }
}

template <int VPL>
__device__ __forceinline__ void stv(__half* out, const __half* h) {
    if constexpr (VPL == 1) {
        out[0] = h[0];
    } else if constexpr (VPL == 2) {
        *reinterpret_cast<uint32_t*>(out) = *reinterpret_cast<const uint32_t*>(h);
    } else if constexpr (VPL == 4) {
        *reinterpret_cast<uint2*>(out) = *reinterpret_cast<const uint2*>(h);
    } else {
#pragma unroll
        for (int k = 0; k < VPL / 8; k++) reinterpret_cast<uint4*>(out)[k] = reinterpret_cast<const uint4*>(h)[k];
    }
}

// the step's maxima of node block nb at a lane's columns (0 where every node is -inf)
// (coherent loads after the wait: the launch before wrote them -- see the top)
template <int VPL>
__device__ __forceinline__ void ldM(const unsigned* Mord, int nb, int Mp, int b, float* M) {
    unsigned o[VPL];
    const unsigned* p = Mord + (long)nb * Mp + b;
    if constexpr (VPL == 1) {
        o[0] = p[0];
    } else if constexpr (VPL == 2) {
        const uint2 x = *reinterpret_cast<const uint2*>(p); o[0] = x.x; o[1] = x.y;
    } else {
#pragma unroll
        for (int k = 0; k < VPL / 4; k++) {
            const uint4 x = *reinterpret_cast<const uint4*>(p + 4 * k);
            o[4 * k] = x.x; o[4 * k + 1] = x.y; o[4 * k + 2] = x.z; o[4 * k + 3] = x.w;
        }
    }
#pragma unroll
    for (int j = 0; j < VPL; j++) M[j] = (o[j] <= f2ord(-INFINITY)) ? 0.f : ord2f(o[j]);
}

// QB rows (one a node block) of a lane's VPL values: each row's max over the slot's columns (shuffles, the rows'
// interleaved), its fp16 values v 2^(15 - x) and its scale 2^x, as the capture kernels form them
template <int VPL, int QB>
__device__ __forceinline__ void store_rows(float (&v)[QB][VPL], int lane, __half* const* out, float* const* fs) {
    float m[QB];
#pragma unroll
    for (int q = 0; q < QB; q++) {
        m[q] = 0.f;
#pragma unroll
        for (int j = 0; j < VPL; j++) m[q] = fmaxf(m[q], v[q][j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int q = 0; q < QB; q++) m[q] = fmaxf(m[q], __shfl_xor_sync(0xffffffffu, m[q], o));
#pragma unroll
    for (int q = 0; q < QB; q++) {
        // m = f 2^x, f in [0.5, 1) (m is 0 or normal: the exponentials flush to zero)
        const int xr = m[q] > 0.f ? (int)((__float_as_uint(m[q]) >> 23) & 0xffu) - 126 : 0;
        // v 2^(15 - x), 15 - x in [-114, 140]: by two normal powers of two, exact (the product at most 2^15)
        const int e = 15 - xr, e1 = min(e, 127);
        const float f1 = pow2i(e1), f2 = pow2i(e - e1);
        __half h[VPL];
#pragma unroll
        for (int j = 0; j < VPL; j++) h[j] = __float2half_rn(v[q][j] * f1 * f2);
        stv<VPL>(out[q] + VPL * lane, h);
        if (lane == 0) *fs[q] = ldexpf(1.f, xr);
    }
}

// ---------------- the operands ----------------
// Rows [0, nbk blk): A's (node block r / blk), lf of the node; then, SHARED (QB > 0), [.., + ne): an edge's Bop row of
// every node block; else [.., + nbk ne): Bop's (node block (r - nbk blk) / ne). A block's 8 R rows are all of one kind
// (blk % 64 == 0) and one node block. FUSED: an edge's element value is its product's two children's node values
// (`ctab`, rows past `first_row`; below it the dummy rows, -inf), else `emars[cid]`. Gated: log phi from the gate
// table and `ext` (0 without a gate row; no gate for the node block: the edge's -inf), as `capture_ab_kernel`.
template <int R, int VPL, int QB>
__global__ void __launch_bounds__(256) caps_ab_kernel(
        const float* __restrict__ nflows, const float* __restrict__ nmars, const float* __restrict__ esrc,
        const int* __restrict__ ctab, long first_row, const float* __restrict__ ext, const long* __restrict__ nids,
        const long* __restrict__ cids, const long* __restrict__ gate, const unsigned* Mord,
        __half* __restrict__ A, __half* __restrict__ Bop, float* __restrict__ fA, float* __restrict__ fB, int nbk,
        int blk, int ne, int batch, int Mp, long Ks, int W, int S, int slot0, int node_cbs, int gate_cbs,
        int gate_stride, long ext_base, bool gated, bool fused) {
    const int lane = threadIdx.x % 32, c = blockIdx.z;
    const long r0 = ((long)blockIdx.x * 8 + threadIdx.x / 32) * R;
    const int cb = c * W, slot = slot0 + c, b = cb + VPL * lane;
    const long arows = (long)nbk * blk;
    if (r0 < arows) {
        const int nb = (int)(r0 / blk), rr = (int)(r0 % blk);
        const long base = nids[nb] + rr;
        float nf[R][VPL], nm[R][VPL];
#pragma unroll
        for (int u = 0; u < R; u++) {
            const long off = (base + u) * (long)batch + b;
            ldv<VPL>(nmars + off, b, batch, -INFINITY, nm[u]);
            ldv<VPL>(nflows + off, b, batch, 0.f, nf[u]);
        }
        cudaGridDependencySynchronize();                  // (the maxima)
        float M[VPL];
        ldM<VPL>(Mord, nb, Mp, b, M);
#pragma unroll
        for (int u = 0; u < R; u++) {
            float v[1][VPL];
#pragma unroll
            for (int j = 0; j < VPL; j++) v[0][j] = nm[u][j] == -INFINITY ? 0.f : __expf(nf[u][j] - nm[u][j] - M[j]);
            __half* out[1] = {A + ((long)nb * blk + rr + u) * Ks + (long)slot * W};
            float* fs[1] = {fA + ((long)nb * S + slot) * blk + rr + u};
            store_rows<VPL, 1>(v, lane, out, fs);
        }
        return;
    }
    const long er = r0 - arows;
    const int nb0 = QB > 0 ? 0 : (int)(er / ne), e0 = (int)(er % ne);
    // the rows' element values (node block nb0's edges: every node block's where SHARED), -inf for a dummy row
    float em[R][VPL];
    long cid[R];
#pragma unroll
    for (int u = 0; u < R; u++) {
        cid[u] = cids[(long)nb0 * ne + e0 + u];
        if (fused) {
            if (cid[u] >= first_row) {
                const int* ct = ctab + (cid[u] - first_row) * 2;
                const int c0 = ct[0], c1 = ct[1];
                float x0[VPL], x1[VPL];
                ldv<VPL>(nmars + (long)c0 * batch + b, b, batch, -INFINITY, x0);
                ldv<VPL>(nmars + (long)c1 * batch + b, b, batch, -INFINITY, x1);
#pragma unroll
                for (int j = 0; j < VPL; j++) em[u][j] = b + j < batch ? x0[j] + x1[j] : -INFINITY;
            } else {
#pragma unroll
                for (int j = 0; j < VPL; j++) em[u][j] = -INFINITY;
            }
        } else {
            ldv<VPL>(esrc + cid[u] * (long)batch + b, b, batch, -INFINITY, em[u]);
        }
    }
    // a row's gate row in node block nb (-2: no gate table, log phi 0; -1: no gate, the edge -inf)
    auto grow_of = [&](int nb, int e) -> long {
        if (!gated) return -2;
        const int gcol = e / node_cbs;
        const long gb = gcol < gate_stride ? gate[(long)nb * gate_stride + gcol] : -1;
        return gb < 0 ? -1 : gb + ext_base + (e % node_cbs) / gate_cbs;
    };
    cudaGridDependencySynchronize();                      // (the maxima)
    constexpr int Q = QB > 0 ? QB : 1;
    const int ngr = QB > 0 ? nbk / QB : 1;
    for (int g = 0; g < ngr; g++) {
        float M[Q][VPL];
#pragma unroll
        for (int q = 0; q < Q; q++) ldM<VPL>(Mord, nb0 + g * Q + q, Mp, b, M[q]);
#pragma unroll
        for (int u = 0; u < R; u++) {
            float v[Q][VPL];
            __half* out[Q];
            float* fs[Q];
#pragma unroll
            for (int q = 0; q < Q; q++) {
                const int nb = nb0 + g * Q + q;
                const long grow = grow_of(nb, e0 + u);
                float lp[VPL];
#pragma unroll
                for (int j = 0; j < VPL; j++) lp[j] = 0.f;
                if (grow >= 0 && (!fused || cid[u] >= first_row)) ldv<VPL>(ext + grow * (long)batch + b, b, batch, 0.f, lp);
#pragma unroll
                for (int j = 0; j < VPL; j++) {
                    const float x = (grow != -1 ? em[u][j] : -INFINITY) + lp[j] + M[q][j];
                    v[q][j] = x == -INFINITY ? 0.f : __expf(x);
                }
                out[q] = Bop + ((long)nb * ne + e0 + u) * Ks + (long)slot * W;
                fs[q] = fB + ((long)nb * S + slot) * ne + e0 + u;
            }
            store_rows<VPL, Q>(v, lane, out, fs);
        }
    }
}

template <typename... P, typename... Args>
static void launch_pdl(void (*kernel)(P...), dim3 grid, dim3 block, cudaStream_t st, Args... args) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid; lc.blockDim = block; lc.dynamicSmemBytes = 0; lc.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at; lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kernel, static_cast<P>(args)...));
}

// Whether a slot of W columns at this batch takes this capture (the scope above)
bool fits(int64_t W, int64_t batch) {
    const int64_t v = W / 32;
    return W % 32 == 0 && (v == 1 || v == 2 || v == 4 || v == 8 || v == 16) && batch % 4 == 0;
}

// The event a capture on a side stream waits for, recorded on the stream it was called on (one a device, reused: a wait
// takes the event's state when it is queued)
static cudaEvent_t fork_event(int dev) {
    static cudaEvent_t evs[64] = {};
    TORCH_CHECK(dev >= 0 && dev < 64, "capture steps: device ", dev);
    if (evs[dev] == nullptr) C10_CUDA_CHECK(cudaEventCreateWithFlags(&evs[dev], cudaEventDisableTiming));
    return evs[dev];
}

// `capture` / `capture_fused`'s arguments but `Ms` int32 [steps, nbk, Mp] (zeroed before the slots fill from the first;
// this step's at slot0 / cps) for their `M`, `esrc` (element_mars, or node_mars where `fused`) and `ctab` (read where
// `fused`), `shared` (the caller verified that every node block's edges have the same children); `side` a stream
// (`cudaStream_t` as an integer) to launch on after what the current stream has queued, or 0 for the current stream
void capture_steps(torch::Tensor A, torch::Tensor Bop, torch::Tensor fA, torch::Tensor fB, torch::Tensor Ms,
                   torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor esrc, torch::Tensor ctab,
                   int64_t first_row, torch::Tensor ext, torch::Tensor nids, torch::Tensor cids, torch::Tensor gate,
                   int64_t batch, int64_t slot0, int64_t node_cbs, int64_t gate_cbs, int64_t ext_base, bool gated,
                   bool fused, bool shared, int64_t side) {
    const int nbk = (int)A.size(0), blk = (int)A.size(1), ne = (int)Bop.size(1), S = (int)fA.size(1);
    const long Ks = A.size(2);
    const int W = (int)(Ks / S), Mp = (int)Ms.size(2), cps = Mp / W;
    TORCH_CHECK(A.scalar_type() == torch::kHalf && Bop.scalar_type() == torch::kHalf && Ms.scalar_type() == torch::kInt32 &&
                Ms.dim() == 3 && Ms.size(1) == nbk && Ks % S == 0 && fits(W, batch) && Mp % W == 0 && Mp >= batch &&
                slot0 % cps == 0 && slot0 + cps <= S && slot0 / cps < Ms.size(0) && blk % 64 == 0 && ne % 64 == 0,
                "capture steps: the slot layout");
    TORCH_CHECK(!fused || (ctab.scalar_type() == torch::kInt32 && ctab.is_contiguous() && ctab.dim() == 2 &&
                           ctab.size(1) == 2), "capture steps: ctab int32 [rows, 2]");
    const at::cuda::OptionalCUDAGuard guard(A.device());
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    if (side != 0) {                                     // (the fork: an event recorded here, waited for there)
        const cudaEvent_t ev = fork_event(A.device().index());
        C10_CUDA_CHECK(cudaEventRecord(ev, st));
        st = reinterpret_cast<cudaStream_t>(side);
        C10_CUDA_CHECK(cudaStreamWaitEvent(st, ev, 0));
    }
    auto* Mord = reinterpret_cast<unsigned*>(Ms.data_ptr()) + (long)(slot0 / cps) * nbk * Mp;
    launch_pdl(caps_m_kernel, dim3(nbk, Mp / 32, blk / 64), dim3(256), st, node_flows.data_ptr<float>(),
               node_mars.data_ptr<float>(), nids.data_ptr<long>(), Mord, blk, (int)batch, Mp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // two rows a warp where that still leaves two blocks an SM (MEASURED HMM-4096 in 4 blocks B 128: 5.8 us, one row 6.3);
    // the node blocks' rows of an edge 4 at a time (2 from 8 columns a lane, 1 from 16: the registers)
    int dev = 0, sms = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int vpl = W / 32;
    int qb = 0;
    if (shared) {
        const int qmax = vpl <= 4 ? 4 : vpl == 8 ? 2 : 1;
        qb = nbk % 4 == 0 && qmax >= 4 ? 4 : nbk % 2 == 0 && qmax >= 2 ? 2 : 1;
    }
    const long rows = (long)nbk * blk + (shared ? (long)ne : (long)nbk * ne);
    const int R = (rows / 16) * cps >= 2L * sms ? 2 : 1;
    const dim3 grid((unsigned)(rows / (8 * R)), 1, cps);
    const int* ct = fused ? ctab.data_ptr<int>() : nullptr;
#define CAPS(R_, V_, Q_)                                                                                               \
    launch_pdl(caps_ab_kernel<R_, V_, Q_>, grid, dim3(256), st, node_flows.data_ptr<float>(),                       \
               node_mars.data_ptr<float>(), esrc.data_ptr<float>(), ct, (long)first_row, ext.data_ptr<float>(),      \
               nids.data_ptr<long>(), cids.data_ptr<long>(), gate.data_ptr<long>(), (const unsigned*)Mord,           \
               reinterpret_cast<__half*>(A.data_ptr<at::Half>()), reinterpret_cast<__half*>(Bop.data_ptr<at::Half>()), \
               fA.data_ptr<float>(), fB.data_ptr<float>(), nbk, blk, ne, (int)batch, Mp, Ks, W, S, (int)slot0,     \
               (int)node_cbs, (int)gate_cbs, gated ? (int)gate.size(1) : 0, (long)ext_base, gated, fused)
#define CAPS_Q(R_, V_)                                                                                                 \
    if (qb == 0) CAPS(R_, V_, 0); else if (qb == 1) CAPS(R_, V_, 1); else if (qb == 2) CAPS(R_, V_, 2);              \
    else CAPS(R_, V_, 4);
#define CAPS_V(R_)                                                                                                     \
    if (vpl == 1) { CAPS_Q(R_, 1) } else if (vpl == 2) { CAPS_Q(R_, 2) } else if (vpl == 4) { CAPS_Q(R_, 4) }        \
    else if (vpl == 8) { if (qb == 0) CAPS(R_, 8, 0); else if (qb == 1) CAPS(R_, 8, 1); else CAPS(R_, 8, 2); }      \
    else { if (qb == 0) CAPS(R_, 16, 0); else CAPS(R_, 16, 1); }
    if (R == 2) { CAPS_V(2) } else { CAPS_V(1) }
#undef CAPS_V
#undef CAPS_Q
#undef CAPS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("capture_steps", &capture_steps, "the deferred capture: a step's maxima, two chained launches");
    m.def("fits", &fits, "whether a slot of W columns at this batch takes capture_steps");
}
