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

#include <torch/extension.h>
#include <cuda_runtime.h>
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
