// The evidence pruning's liveness sweep (`kernels/live.py`, `_liveness_kernel`) in shared memory: `live[b, t, q]`
// (int8 [B, n + 1, W]) the forward reachability of column q at boundary t through the sample's observed classes
// (where a token is missing, the successor lists' classes: every class, or a sample's own lists of the classes its
// inputs leave possible, `ps` / `ss` apart), ANDed with whether acceptance can be reached from it -- a CTA per sample
// for each of the two sweeps, at once (they do not depend on each other), into `live` and `back`, then their AND.
//
// A boundary's step is latency-bound: a column's successors are dependent reads (`entry_ptr`, then `entry_succ`, or,
// for an observed token, the column's state, its successor in `delta`, that one's column) after the live row. Here the rows live in shared memory, and every step's tables --
// the next boundary's successor lists (or observed successors) and, going back, its forward row -- are read into
// registers during the step before and stored to shared memory after it, so a step waits for no global read.
// Columns past a boundary's width are never written, as by the Triton kernel.
//
// `liveness_wide_forward_kernel` / `liveness_wide_backward_kernel`, below: the same sweeps for automata the shared
// memory does not take, the rows in global memory, with shortcuts where the sweep saturates.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#define LT 1024            // threads per CTA
#define LQ 4               // columns a thread owns, at most: W <= LQ * LT
#define LS 16              // successor entries a thread stages per boundary, at most: LS * LT a boundary's

struct LiveArgs {
    signed char* live;
    const int* obs;        // [B, n]
    const int* next_col;   // [n, W, C] every boundary's transitions, or nullptr: through the three tables below
    const int* delta;      // [K, C] the automaton's transitions
    const int* state_id;   // [n + 1, W] every column's state
    const int* col_of_state;   // [n + 1, K] every state's column, -1 if inactive
    const int* ptr;        // [n, W + 1] per sample, ps apart (0: every sample's)
    const int* succ;       // ss apart
    const int* width;      // [n + 1]
    int n, W, C, K, smax;
    long long ps, ss;
};

// The tables of boundary t a step reads, into registers: an observed token's successor of every owned column, or the
// successor lists' offsets and this thread's share of the boundary's entries.
struct Stage {
    int nc[LQ];
    int pq[LQ + 1];
    int sc[LS];
};

__device__ __forceinline__ void stage_load(const LiveArgs& a, const int* ptr, const int* succ, int t, int w, int oc,
                                           int base, int count, int tid, Stage& s) {
    if (oc >= 0) {
#pragma unroll
        for (int k = 0; k < LQ; ++k) {
            int q = tid + k * LT;
            if (a.next_col != nullptr) {
                s.nc[k] = q < w ? a.next_col[((long long)t * a.W + q) * a.C + oc] : -1;
            } else {                                        // the column's state, its successor, that one's column
                int st = q < w ? a.state_id[(long long)t * a.W + q] : -1;
                s.nc[k] = st >= 0 ? a.col_of_state[(long long)(t + 1) * a.K + a.delta[(long long)st * a.C + oc]] : -1;
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k <= LQ; ++k) {
            int q = tid + k * LT;
            s.pq[k] = q <= w ? ptr[(long long)t * (a.W + 1) + q] : 0;
        }
#pragma unroll
        for (int j = 0; j < LS; ++j) {
            int i = tid + j * LT;
            s.sc[j] = i < count ? succ[base + i] : 0;
        }
    }
}

__device__ __forceinline__ void stage_store(int w, int oc, int count, int tid, const Stage& s, int* nc, int* pq,
                                            int* sc) {
    if (oc >= 0) {
#pragma unroll
        for (int k = 0; k < LQ; ++k) {
            int q = tid + k * LT;
            if (q < w) nc[q] = s.nc[k];
        }
    } else {
#pragma unroll
        for (int k = 0; k <= LQ; ++k) {
            int q = tid + k * LT;
            if (q <= w) pq[q] = s.pq[k];
        }
#pragma unroll
        for (int j = 0; j < LS; ++j) {
            int i = tid + j * LT;
            if (i < count) sc[i] = s.sc[j];
        }
    }
}

// One CTA per (sample, direction): blockIdx.y 0 the forward sweep into `live`, 1 the backward one into `back` -- the
// two chains are independent, so they run at once; `liveness_and_kernel` then ANDs them.
__global__ void __launch_bounds__(LT) liveness_kernel(LiveArgs a, signed char* back_all) {
    extern __shared__ int smem[];
    const int n = a.n, W = a.W, tid = threadIdx.x;
    int* nc = smem;                                      // [W]: an observed token's successors
    int* pq = nc + W;                                    // [W + 1]: the successor lists' offsets
    int* sc = pq + W + 1;                                // [smax]: the boundary's successor entries
    int* oc = sc + a.smax;                               // [n]: the sample's observed classes
    int* sbase = oc + n;                                 // [n]: every boundary's first entry
    int* scount = sbase + n;                             // [n]: and its entries
    int* wd = scount + n;                                // [n + 1]: every boundary's columns
    unsigned char* rowa = reinterpret_cast<unsigned char*>(wd + n + 1);         // [W]
    unsigned char* rowb = rowa + W;                                              // [W]
    const long long b = blockIdx.x;
    const int* ptr = a.ptr + b * a.ps;                    // the sample's successor lists
    const int* succ = a.succ + b * a.ss;
    for (int t = tid; t <= n; t += LT) {
        int w = a.width[t];
        wd[t] = w;
        if (t == n) continue;
        oc[t] = a.obs[b * n + t];
        int s0 = ptr[(long long)t * (W + 1)];
        sbase[t] = s0;
        scount[t] = ptr[(long long)t * (W + 1) + w] - s0;
    }
    __syncthreads();
    Stage st;

    if (blockIdx.y == 0) {                               // forward: rowa boundary t, rowb boundary t + 1
        signed char* live = a.live + b * (long long)(n + 1) * W;
        stage_load(a, ptr, succ, 0, wd[0], oc[0], sbase[0], scount[0], tid, st);
        stage_store(wd[0], oc[0], scount[0], tid, st, nc, pq, sc);
        for (int q = tid; q < wd[0]; q += LT) {
            rowa[q] = q == 0;
            live[q] = q == 0;
        }
        __syncthreads();
        for (int t = 0; t < n; ++t) {
            const int w = wd[t], ot = oc[t];
            const int nxt = t + 1 == n ? 1 : wd[t + 1];
            if (t + 1 < n) stage_load(a, ptr, succ, t + 1, wd[t + 1], oc[t + 1], sbase[t + 1], scount[t + 1], tid, st);
            for (int q = tid; q < nxt; q += LT) rowb[q] = 0;
            __syncthreads();
            for (int q = tid; q < w; q += LT) {
                if (!rowa[q]) continue;
                if (ot >= 0) {
                    int s = nc[q];
                    if (t + 1 == n && s >= 0) s = 0;
                    if (s >= 0) rowb[s] = 1;
                } else {
                    const int base = sbase[t];
                    for (int k = pq[q]; k < pq[q + 1]; ++k) rowb[sc[k - base]] = 1;
                }
            }
            __syncthreads();
            if (t + 1 < n) stage_store(wd[t + 1], oc[t + 1], scount[t + 1], tid, st, nc, pq, sc);
            signed char* out = live + (long long)(t + 1) * W;
            for (int q = tid; q < nxt; q += LT) out[q] = rowb[q];
            unsigned char* tmp = rowa; rowa = rowb; rowb = tmp;
            __syncthreads();
        }
        return;
    }

    // backward: rowa acceptance from boundary t + 1, rowb from t, into `back`
    signed char* back = back_all + b * (long long)(n + 1) * W;
    if (tid == 0) rowa[0] = 1;                                                  // the sequence end
    stage_load(a, ptr, succ, n - 1, wd[n - 1], oc[n - 1], sbase[n - 1], scount[n - 1], tid, st);
    stage_store(wd[n - 1], oc[n - 1], scount[n - 1], tid, st, nc, pq, sc);
    __syncthreads();
    for (int t = n - 1; t >= 0; --t) {
        const int w = wd[t], ot = oc[t];
        if (t > 0) stage_load(a, ptr, succ, t - 1, wd[t - 1], oc[t - 1], sbase[t - 1], scount[t - 1], tid, st);
        signed char* out = back + (long long)t * W;
        for (int q = tid; q < w; q += LT) {
            bool g = false;
            if (ot >= 0) {
                int s = nc[q];
                if (t + 1 == n && s >= 0) s = 0;
                g = s >= 0 && rowa[s];
            } else {
                const int base = sbase[t];
                for (int e = pq[q]; e < pq[q + 1] && !g; ++e) g = rowa[sc[e - base]];
            }
            rowb[q] = g;
            out[q] = g;
        }
        __syncthreads();
        if (t > 0) stage_store(wd[t - 1], oc[t - 1], scount[t - 1], tid, st, nc, pq, sc);
        unsigned char* tmp = rowa; rowa = rowb; rowb = tmp;
        __syncthreads();
    }
}

// live[b, t, q] &= back[b, t, q] over every boundary t < n and column q < width[t]: a CTA per (sample, boundary)
__global__ void liveness_and_kernel(signed char* live, const signed char* back, const int* width, int n, int W) {
    const long long row = (long long)blockIdx.x * (n + 1) + blockIdx.y;
    const int w = width[blockIdx.y];
    for (int q = threadIdx.x; q < w; q += blockDim.x)
        live[row * W + q] = live[row * W + q] != 0 && back[row * W + q] != 0;
}

static size_t liveness_smem(int n, int W, int smax) {
    return sizeof(int) * ((size_t)W + W + 1 + smax + 4 * (size_t)n + 1) + 2 * (size_t)W;
}

// Whether the shared-memory sweep takes an automaton of W columns, at most smax successor entries a boundary (a
// sample's lists of fewer classes have no more), over n positions on this device.
bool liveness_fits(int64_t n, int64_t W, int64_t smax) {
    if (W > (int64_t)LQ * LT || smax > (int64_t)LS * LT) return false;
    int dev, optin;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    return liveness_smem((int)n, (int)W, (int)smax) <= (size_t)optin;
}

void liveness_cuda(torch::Tensor live, torch::Tensor back, torch::Tensor obs_class, torch::Tensor next_col,
                   torch::Tensor delta, torch::Tensor state_id, torch::Tensor col_of_state, torch::Tensor entry_ptr,
                   torch::Tensor entry_succ, torch::Tensor width, int64_t n, int64_t W, int64_t C, int64_t K,
                   int64_t smax, int64_t ps, int64_t ss, bool homogeneous) {
    TORCH_CHECK(live.scalar_type() == torch::kInt8 && back.scalar_type() == torch::kInt8 && back.is_contiguous() &&
                back.numel() == live.numel() && obs_class.scalar_type() == torch::kInt32 &&
                next_col.scalar_type() == torch::kInt32 && delta.scalar_type() == torch::kInt32 &&
                state_id.scalar_type() == torch::kInt32 &&
                col_of_state.scalar_type() == torch::kInt32 && entry_ptr.scalar_type() == torch::kInt32 &&
                entry_succ.scalar_type() == torch::kInt32 && width.scalar_type() == torch::kInt32,
                "liveness_cuda: int8 live, int32 tables");
    TORCH_CHECK(live.is_contiguous() && obs_class.is_contiguous() && next_col.is_contiguous() && delta.is_contiguous() &&
                state_id.is_contiguous() && col_of_state.is_contiguous() && entry_ptr.is_contiguous() &&
                entry_succ.is_contiguous() && width.is_contiguous(), "liveness_cuda: contiguous tensors");
    TORCH_CHECK(W <= (int64_t)LQ * LT && smax <= (int64_t)LS * LT, "liveness_cuda: automaton too wide");
    const int B = (int)live.size(0);
    if (B == 0) return;
    TORCH_CHECK(ps == 0 || entry_ptr.numel() >= (int64_t)B * ps, "liveness_cuda: a successor list per sample");
    LiveArgs a{live.data_ptr<signed char>(), obs_class.data_ptr<int>(),
               homogeneous ? nullptr : next_col.data_ptr<int>(), delta.data_ptr<int>(),
               state_id.data_ptr<int>(), col_of_state.data_ptr<int>(), entry_ptr.data_ptr<int>(),
               entry_succ.data_ptr<int>(), width.data_ptr<int>(), (int)n, (int)W, (int)C, (int)K, (int)smax,
               (long long)ps, (long long)ss};
    size_t smem = liveness_smem((int)n, (int)W, (int)smax);
    static size_t allowed[64] = {0};                    // per device, the largest dynamic shared memory set
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (smem > 48 * 1024 && smem > allowed[dev % 64]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(liveness_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        allowed[dev % 64] = smem;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    liveness_kernel<<<dim3(B, 2), LT, smem, stream>>>(a, back.data_ptr<signed char>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    liveness_and_kernel<<<dim3(B, (unsigned)n), 256, 0, stream>>>(live.data_ptr<signed char>(),
                                                                    back.data_ptr<signed char>(),
                                                                    width.data_ptr<int>(), (int)n, (int)W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The sweep for automata too wide for shared memory: the same two sweeps as `liveness_kernel`, a CTA per sample, the
// rows in global memory (`live`, `back`: what the sweeps write anyway) and every boundary's tables read as the step
// needs them -- no bound on the columns or the successor entries. The backward sweep runs after the forward one and
// visits the forward-live columns alone: the final liveness is their AND, and a live column's successors under the
// classes allowed are live, so the backward value of a live column reads live columns' alone.
//
// SATURATION: with every class allowed (the program's lists, not a sample's), a full forward row -- every active column
// live -- stays full through a missing token: a state active at t + 2 is reached from a state reachable at t + 1 that
// can still accept, hence active and live at t + 1. So once the forward row is full and no token is observed at or
// after the boundary, every later row is full: written at once. Likewise, backward, every active column can accept
// after an unobserved suffix: the rows after the last observed position are full, and the sweep starts there.
// With a sample's own lists (its inputs rule classes out) the shortcuts do not hold, and the sweeps step through.

__device__ __forceinline__ int block_count(int v, int* red) {              // the sum of v over the CTA's threads
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    int s = 0;
    for (int k = 0; k < (int)(blockDim.x >> 5); ++k) s += red[k];
    __syncthreads();
    return s;
}

__device__ __forceinline__ int last_observed(const int* obs, int n, int* red) {   // the sample's last observed position
    int last = -1;
    for (int t = threadIdx.x; t < n; t += blockDim.x) if (obs[t] >= 0) last = max(last, t);
    for (int o = 16; o > 0; o >>= 1) last = max(last, __shfl_down_sync(0xffffffffu, last, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = last;
    __syncthreads();
    last = -1;
    for (int k = 0; k < (int)(blockDim.x >> 5); ++k) last = max(last, red[k]);
    __syncthreads();
    return last;
}

// The live columns of a row, 16 bytes at a time from the aligned chunk holding the row's first byte (the tensor's
// base is aligned, so no chunk starts before it; bytes outside the row are masked): `f(q)` for every live `q < w`,
// the CTA's threads over the chunks.
template <typename F>
__device__ __forceinline__ void for_live(const signed char* row, int w, int tid, F f) {
    const int lead = (int)(reinterpret_cast<uintptr_t>(row) & 15);
    const uint4* v = reinterpret_cast<const uint4*>(row - lead);
    const int chunks = (w + lead + 15) >> 4;
    for (int i = tid; i < chunks; i += LT) {
        const uint4 x = v[i];
        if ((x.x | x.y | x.z | x.w) == 0u) continue;
        const unsigned int words[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            if ((words[k >> 2] >> ((k & 3) * 8)) & 0xffu) {
                const int q = i * 16 + k - lead;
                if (q >= 0 && q < w) f(q);
            }
        }
    }
}

__device__ __forceinline__ int observed_successor(const LiveArgs& a, int t, int q, int oc) {
    int s;
    if (a.next_col != nullptr) s = a.next_col[((long long)t * a.W + q) * a.C + oc];
    else {
        int st = a.state_id[(long long)t * a.W + q];
        s = st >= 0 ? a.col_of_state[(long long)(t + 1) * a.K + a.delta[(long long)st * a.C + oc]] : -1;
    }
    if (t + 1 == a.n && s >= 0) s = 0;
    return s;
}

#define LF 2048            // the frontier's columns at most (a step through a few live columns reads them alone)
#define LD 8               // columns a thread stages at once in a dense step (their pointers loaded together)
#define LG 16              // CTAs a sample of the cooperative sweep, at most
#define LN 2048            // boundaries whose width and token the CTA caches (static shared memory: 48 KB)

__global__ void __launch_bounds__(LT) liveness_wide_forward_kernel(LiveArgs a, int every_class) {
    __shared__ int red[LT / 32];
    __shared__ int frontier[2][LF];                      // the live columns of boundary t (fewer than LF), then t + 1
    __shared__ int count[2];
    __shared__ int wd[LN + 1], oc_s[LN];
    const int n = a.n, W = a.W, tid = threadIdx.x;
    const long long b = blockIdx.x;
    const int* ptr = a.ptr + b * a.ps;
    const int* succ = a.succ + b * a.ss;
    const int* obs = a.obs + b * n;
    const bool cached = n < LN;
    if (cached) {
        for (int t = tid; t <= n; t += LT) wd[t] = a.width[t];
        for (int t = tid; t < n; t += LT) oc_s[t] = obs[t];
    }
    const int last = last_observed(obs, n, red);
    signed char* live = a.live + b * (long long)(n + 1) * W;          // (zero: the caller's)
    if (tid == 0) {
        live[0] = 1;
        frontier[0][0] = 0;
        count[0] = 1;
    }
    __syncthreads();
    int cur = 0;                                                      // the frontier in use (-1: dense rows)
    for (int t = 0; t < n; ++t) {
        const int w = cached ? wd[t] : a.width[t], oc = cached ? oc_s[t] : obs[t];
        const int nxt = t + 1 == n ? 1 : (cached ? wd[t + 1] : a.width[t + 1]);
        signed char* out = live + (long long)(t + 1) * W;
        if (cur >= 0) {                                               // sparse: the frontier's columns alone, the
            const int nf = count[cur];                                // successors appended (duplicates allowed) --
            if (tid == 0) count[1 - cur] = 0;                         // unless they overflow: then dense from here
            __syncthreads();
            for (int k = tid; k < nf; k += LT) {
                const int q = frontier[cur][k];
                if (oc >= 0) {
                    const int s = observed_successor(a, t, q, oc);
                    if (s >= 0) {
                        out[s] = 1;
                        const int at = atomicAdd(&count[1 - cur], 1);
                        if (at < LF) frontier[1 - cur][at] = s;
                    }
                } else {
                    const int e0 = ptr[(long long)t * (W + 1) + q], e1 = ptr[(long long)t * (W + 1) + q + 1];
                    for (int e = e0; e < e1; ++e) {
                        const int s = succ[e];
                        out[s] = 1;
                        const int at = atomicAdd(&count[1 - cur], 1);
                        if (at < LF) frontier[1 - cur][at] = s;
                    }
                }
            }
            __syncthreads();
            cur = count[1 - cur] <= LF ? 1 - cur : -1;
        } else {                                                      // dense: every live column of the row
            const signed char* row = live + (long long)t * W;
            if (oc >= 0) {
                for_live(row, w, tid, [&](int q) {
                    const int s = observed_successor(a, t, q, oc);
                    if (s >= 0) out[s] = 1;
                });
            } else {                                                  // LD columns a thread at once: their live
                const int* prow = ptr + (long long)t * (W + 1);       // bytes, then their pointers, then the lists
                for (int q0 = 0; q0 < w; q0 += LD * LT) {
                    int e0[LD], e1[LD];
#pragma unroll
                    for (int k = 0; k < LD; ++k) {
                        const int q = q0 + k * LT + tid;
                        const bool on = q < w && row[q];
                        e0[k] = on ? prow[q] : 0;
                        e1[k] = on ? prow[q + 1] : 0;
                    }
#pragma unroll
                    for (int k = 0; k < LD; ++k)
                        for (int e = e0[k]; e < e1[k]; ++e) out[succ[e]] = 1;
                }
            }
            __syncthreads();
        }
        if (every_class && t + 1 > last && t + 1 < n && cur < 0) {       // saturated, nothing observed ahead: the
            int mine = 0;                                                 // rest of the rows are full
            for_live(out, nxt, tid, [&](int) { ++mine; });
            if (block_count(mine, red) == nxt) {
                for (int u = t + 2; u <= n; ++u) {
                    signed char* r = live + (long long)u * W;
                    const int wu = u == n ? 1 : (cached ? wd[u] : a.width[u]);
                    for (int q = tid; q < wu; q += LT) r[q] = 1;
                }
                return;
            }
        }
    }
}

// After the forward sweep: the backward values of the forward-live columns (the others' are never read: a live
// column's successors under the classes allowed are live, and the AND keeps live columns alone).
__global__ void __launch_bounds__(LT) liveness_wide_backward_kernel(LiveArgs a, signed char* back_all, int every_class) {
    __shared__ int red[LT / 32];
    const int n = a.n, W = a.W, tid = threadIdx.x;
    const long long b = blockIdx.x;
    const int* ptr = a.ptr + b * a.ps;
    const int* succ = a.succ + b * a.ss;
    const int* obs = a.obs + b * n;
    __shared__ int wd[LN + 1], oc_s[LN];
    const bool cached = n < LN;
    if (cached) {
        for (int t = tid; t <= n; t += LT) wd[t] = a.width[t];
        for (int t = tid; t < n; t += LT) oc_s[t] = obs[t];
    }
    const int last = last_observed(obs, n, red);
    const signed char* live = a.live + b * (long long)(n + 1) * W;
    signed char* back = back_all + b * (long long)(n + 1) * W;
    int start = n - 1;
    if (every_class) {                                                    // the unobserved suffix: full rows
        for (int u = max(last + 1, 0); u <= n; ++u) {
            signed char* r = back + (long long)u * W;
            const int wu = u == n ? 1 : (cached ? wd[u] : a.width[u]);
            for (int q = tid; q < wu; q += LT) r[q] = 1;
        }
        start = last;                                                     // (-1: nothing observed, all done)
    } else {
        if (tid == 0) back[(long long)n * W] = 1;
    }
    __syncthreads();
    for (int t = start; t >= 0; --t) {
        const int w = cached ? wd[t] : a.width[t], oc = cached ? oc_s[t] : obs[t];
        const signed char* row = live + (long long)t * W;
        const signed char* nextrow = back + (long long)(t + 1) * W;
        signed char* out = back + (long long)t * W;
        for_live(row, w, tid, [&](int q) {                                // (a column not live: never read)
            bool g = false;
            if (oc >= 0) {
                const int s = observed_successor(a, t, q, oc);
                g = s >= 0 && nextrow[s];
            } else {
                const int e0 = ptr[(long long)t * (W + 1) + q], e1 = ptr[(long long)t * (W + 1) + q + 1];
                for (int e = e0; e < e1 && !g; ++e) g = nextrow[succ[e]];
            }
            out[q] = g;
        });
        __syncthreads();
    }
}

// The same wide sweeps with several CTAs a sample (grid [G, B], cooperative launch: every CTA resident, a grid
// barrier a boundary). The forward sweep's observed prefix -- a frontier of few columns -- is the leader CTA's alone;
// from the first missing token (or a frontier too large) every CTA takes a slice of the row's columns per boundary,
// which is where a single CTA spent its time: 49k columns' lists a boundary for ~100 boundaries on an automaton of
// 16 classes. Saturation is the leader's call, read by all after the next barrier. The backward sweep: the rows
// after the last observed position full (slices), then the boundaries down to the forward's first dense one with
// every CTA, then the sparse prefix with the leader alone (one live column a boundary: a barrier would cost more).
namespace cg = cooperative_groups;

__global__ void __launch_bounds__(LT) liveness_coop_forward_kernel(LiveArgs a, int every_class, int* dense_from,
                                                                 int* done) {
    cg::grid_group grid = cg::this_grid();
    __shared__ int red[LT / 32];
    __shared__ int frontier[2][LF];
    __shared__ int count[2];
    __shared__ int wd[LN + 1], oc_s[LN];
    const int n = a.n, W = a.W, tid = threadIdx.x, G = gridDim.x, g = blockIdx.x;
    const long long b = blockIdx.y;
    const int* ptr = a.ptr + b * a.ps;
    const int* succ = a.succ + b * a.ss;
    const int* obs = a.obs + b * n;
    const bool cached = n < LN;
    if (cached) {
        for (int t = tid; t <= n; t += LT) wd[t] = a.width[t];
        for (int t = tid; t < n; t += LT) oc_s[t] = obs[t];
    }
    const int last = last_observed(obs, n, red);
    signed char* live = a.live + b * (long long)(n + 1) * W;          // (zero: the caller's)
    int t = 0;
    if (g == 0) {                                                     // the leader: the sparse prefix
        if (tid == 0) {
            live[0] = 1;
            frontier[0][0] = 0;
            count[0] = 1;
            done[b] = 0;
        }
        __syncthreads();
        int cur = 0;
        for (; t < n; ++t) {
            const int oc = cached ? oc_s[t] : obs[t];
            if (oc < 0) break;                                        // a missing token: dense from here, all CTAs
            const int nf = count[cur];
            if (tid == 0) count[1 - cur] = 0;
            __syncthreads();
            signed char* out = live + (long long)(t + 1) * W;
            for (int k = tid; k < nf; k += LT) {
                const int s = observed_successor(a, t, frontier[cur][k], oc);
                if (s >= 0) {
                    out[s] = 1;
                    const int at = atomicAdd(&count[1 - cur], 1);
                    if (at < LF) frontier[1 - cur][at] = s;
                }
            }
            __syncthreads();
            if (count[1 - cur] > LF) { ++t; break; }                   // (row t + 1 is written: dense from t + 1)
            cur = 1 - cur;
        }
        if (tid == 0) dense_from[b] = t;
        __threadfence();
    }
    grid.sync();                                                      // (#0: the leaders' prefixes done)
    t = n;                                                            // every CTA steps from the samples' first dense
    for (int k = 0; k < (int)gridDim.y; ++k) t = min(t, dense_from[k]);   // boundary (one barrier count for all)
    const int from = dense_from[b];
    const int lo = (int)((long long)g * W / G), hi = (int)((long long)(g + 1) * W / G);   // this CTA's columns
    bool mine_done = false;
    for (;; ++t) {
        bool all = true;                                              // every sample done, or past its end
        for (int k = 0; k < (int)gridDim.y; ++k) all = all && (done[k] || dense_from[k] >= n);
        if (all || t >= n) break;
        if (!mine_done && t >= from) {
            const int w = cached ? wd[t] : a.width[t], oc = cached ? oc_s[t] : obs[t];
            const signed char* row = live + (long long)t * W;
            signed char* out = live + (long long)(t + 1) * W;
            if (g == 0 && every_class && t > last && t < n) {          // the leader: row t full and nothing observed
                int mine = 0;                                         // ahead -> every later row is full, done
                for_live(row, w, tid, [&](int) { ++mine; });
                if (block_count(mine, red) == w) {
                    for (int u = t + 1; u <= n; ++u) {
                        signed char* r = live + (long long)u * W;
                        const int wu = u == n ? 1 : (cached ? wd[u] : a.width[u]);
                        for (int q = tid; q < wu; q += LT) r[q] = 1;
                    }
                    if (tid == 0) done[b] = 1;
                    __threadfence();
                    mine_done = true;
                }
            }
            if (!mine_done) {
                const int end = min(hi, w);
                for (int q = lo + tid; q < end; q += LT) {
                    if (!row[q]) continue;
                    if (oc >= 0) {
                        const int s = observed_successor(a, t, q, oc);
                        if (s >= 0) out[s] = 1;
                    } else {
                        const int e0 = ptr[(long long)t * (W + 1) + q], e1 = ptr[(long long)t * (W + 1) + q + 1];
                        for (int e = e0; e < e1; ++e) out[succ[e]] = 1;
                    }
                }
            }
        }
        grid.sync();                                                  // (row t + 1 complete; done flags visible)
        if (done[b]) mine_done = true;
    }
}

__global__ void __launch_bounds__(LT) liveness_coop_backward_kernel(LiveArgs a, signed char* back_all, int every_class,
                                                                  const int* dense_from) {
    cg::grid_group grid = cg::this_grid();
    __shared__ int red[LT / 32];
    __shared__ int wd[LN + 1], oc_s[LN];
    const int n = a.n, W = a.W, tid = threadIdx.x, G = gridDim.x, g = blockIdx.x;
    const long long b = blockIdx.y;
    const int* ptr = a.ptr + b * a.ps;
    const int* succ = a.succ + b * a.ss;
    const int* obs = a.obs + b * n;
    const bool cached = n < LN;
    if (cached) {
        for (int t = tid; t <= n; t += LT) wd[t] = a.width[t];
        for (int t = tid; t < n; t += LT) oc_s[t] = obs[t];
    }
    const int last = last_observed(obs, n, red);
    const signed char* live = a.live + b * (long long)(n + 1) * W;
    signed char* back = back_all + b * (long long)(n + 1) * W;
    const int lo = (int)((long long)g * W / G), hi = (int)((long long)(g + 1) * W / G);
    int start = n - 1;
    if (every_class) {                                                // the unobserved suffix: full rows, in slices
        for (int u = max(last + 1, 0); u <= n; ++u) {
            signed char* r = back + (long long)u * W;
            const int wu = u == n ? 1 : (cached ? wd[u] : a.width[u]);
            for (int q = lo + tid; q < min(hi, wu); q += LT) r[q] = 1;
        }
        start = last;
    } else if (g == 0 && tid == 0) {
        back[(long long)n * W] = 1;
    }
    const int sparse_end = dense_from[b];                             // boundaries below it: one live column each
    grid.sync();                                                      // (the full rows written)
    int steps = 0;                                                    // the dense boundaries, every CTA a slice
    for (int t = start; t >= sparse_end; --t, ++steps) {
        const int w = cached ? wd[t] : a.width[t], oc = cached ? oc_s[t] : obs[t];
        const signed char* row = live + (long long)t * W;
        const signed char* nextrow = back + (long long)(t + 1) * W;
        signed char* out = back + (long long)t * W;
        for (int q = lo + tid; q < min(hi, w); q += LT) {
            if (!row[q]) continue;
            bool gq = false;
            if (oc >= 0) {
                const int s = observed_successor(a, t, q, oc);
                gq = s >= 0 && nextrow[s];
            } else {
                const int e0 = ptr[(long long)t * (W + 1) + q], e1 = ptr[(long long)t * (W + 1) + q + 1];
                for (int e = e0; e < e1 && !gq; ++e) gq = nextrow[succ[e]];
            }
            out[q] = gq;
        }
        grid.sync();
    }
    // the other samples' dense boundaries may be more: keep the barrier count equal across the grid
    int longest = 0;
    for (int k = 0; k < (int)gridDim.y; ++k) {
        const int lastk = every_class ? last_observed(a.obs + k * n, n, red) : n - 1;
        longest = max(longest, lastk - dense_from[k] + 1);
    }
    for (; steps < longest; ++steps) grid.sync();
    if (g != 0) return;
    for (int t = min(start, sparse_end - 1); t >= 0; --t) {           // the sparse prefix: the leader alone
        const int w = cached ? wd[t] : a.width[t], oc = cached ? oc_s[t] : obs[t];
        const signed char* row = live + (long long)t * W;
        const signed char* nextrow = back + (long long)(t + 1) * W;
        signed char* out = back + (long long)t * W;
        for_live(row, w, tid, [&](int q) {
            bool gq = false;
            if (oc >= 0) {
                const int s = observed_successor(a, t, q, oc);
                gq = s >= 0 && nextrow[s];
            } else {
                const int e0 = ptr[(long long)t * (W + 1) + q], e1 = ptr[(long long)t * (W + 1) + q + 1];
                for (int e = e0; e < e1 && !gq; ++e) gq = nextrow[succ[e]];
            }
            out[q] = gq;
        });
        __syncthreads();
    }
}

// How many CTAs a sample the cooperative sweep may use (0: not cooperatively, the single-CTA kernels): every CTA of
// the grid [G, B] must be resident at once.
static int coop_ctas_per_sample(int B) {
    static int per_sm[64] = {0}, sms[64] = {0};
    int dev;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    if (sms[dev % 64] == 0) {
        int coop = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms[dev % 64], cudaDevAttrMultiProcessorCount, dev));
        int blocks = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, liveness_coop_forward_kernel, LT, 0));
        int blocks_b = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_b, liveness_coop_backward_kernel, LT, 0));
        per_sm[dev % 64] = coop ? min(blocks, blocks_b) : 0;
    }
    const int resident = per_sm[dev % 64] * sms[dev % 64];
    if (resident < B) return 0;
    return min(LG, resident / B);
}

void liveness_wide_cuda(torch::Tensor live, torch::Tensor back, torch::Tensor obs_class, torch::Tensor next_col,
                        torch::Tensor delta, torch::Tensor state_id, torch::Tensor col_of_state,
                        torch::Tensor entry_ptr, torch::Tensor entry_succ, torch::Tensor width, int64_t n, int64_t W,
                        int64_t C, int64_t K, int64_t ps, int64_t ss, bool homogeneous, bool every_class,
                        torch::Tensor scratch) {
    TORCH_CHECK(live.scalar_type() == torch::kInt8 && back.scalar_type() == torch::kInt8 && back.is_contiguous() &&
                back.numel() == live.numel() && obs_class.scalar_type() == torch::kInt32 &&
                next_col.scalar_type() == torch::kInt32 && delta.scalar_type() == torch::kInt32 &&
                state_id.scalar_type() == torch::kInt32 &&
                col_of_state.scalar_type() == torch::kInt32 && entry_ptr.scalar_type() == torch::kInt32 &&
                entry_succ.scalar_type() == torch::kInt32 && width.scalar_type() == torch::kInt32,
                "liveness_wide_cuda: int8 live, int32 tables");
    TORCH_CHECK(live.is_contiguous() && obs_class.is_contiguous() && next_col.is_contiguous() && delta.is_contiguous() &&
                state_id.is_contiguous() && col_of_state.is_contiguous() && entry_ptr.is_contiguous() &&
                entry_succ.is_contiguous() && width.is_contiguous(), "liveness_wide_cuda: contiguous tensors");
    const int B = (int)live.size(0);
    if (B == 0 || n == 0) return;
    TORCH_CHECK(ps == 0 || entry_ptr.numel() >= (int64_t)B * ps, "liveness_wide_cuda: a successor list per sample");
    LiveArgs a{live.data_ptr<signed char>(), obs_class.data_ptr<int>(),
               homogeneous ? nullptr : next_col.data_ptr<int>(), delta.data_ptr<int>(),
               state_id.data_ptr<int>(), col_of_state.data_ptr<int>(), entry_ptr.data_ptr<int>(),
               entry_succ.data_ptr<int>(), width.data_ptr<int>(), (int)n, (int)W, (int)C, (int)K, 0,
               (long long)ps, (long long)ss};
    auto stream = at::cuda::getCurrentCUDAStream();
    const int G = coop_ctas_per_sample(B);
    if (G >= 2 && scratch.numel() >= 2 * B) {
        TORCH_CHECK(scratch.scalar_type() == torch::kInt32 && scratch.is_contiguous(), "liveness_wide_cuda: int32 scratch");
        int* dense_from = scratch.data_ptr<int>();
        int* done = dense_from + B;
        int ec = every_class ? 1 : 0;
        signed char* back_ptr = back.data_ptr<signed char>();
        void* fargs[] = {&a, &ec, &dense_from, &done};
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)liveness_coop_forward_kernel, dim3(G, B), dim3(LT), fargs, 0,
                                                   stream));
        const int* df = dense_from;
        void* bargs[] = {&a, &back_ptr, &ec, &df};
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)liveness_coop_backward_kernel, dim3(G, B), dim3(LT), bargs, 0,
                                                   stream));
    } else {
        liveness_wide_forward_kernel<<<B, LT, 0, stream>>>(a, every_class ? 1 : 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        liveness_wide_backward_kernel<<<B, LT, 0, stream>>>(a, back.data_ptr<signed char>(), every_class ? 1 : 0);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    liveness_and_kernel<<<dim3(B, (unsigned)n), 256, 0, stream>>>(live.data_ptr<signed char>(),
                                                                    back.data_ptr<signed char>(),
                                                                    width.data_ptr<int>(), (int)n, (int)W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
