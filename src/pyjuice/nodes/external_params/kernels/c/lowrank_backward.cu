// CUDA implementation of the per-sample low-rank sum-layer BACKWARD correction.
//
// The stock sum backward already produces the right answer for the SHARED parameters, provided it is
// handed `logT` instead of `node_mars`. Writing `S = T/Z` for the effective node value:
//
//     flow(c,n,b) = f[n,b] * theta_tilde[c,n,b] * p[c,b] / S[n,b]
//                 = f * theta_shared[c,n] * p[c,b] / T[n,b]   +   f * Delta[c,n,b] * p[c,b] / T[n,b]
//
// because the `Z` in `theta_tilde` cancels against the `Z` in `S`. The FIRST term is exactly what the
// stock kernel computes on `logT`, so:
//
//   * param flows need NO correction -- theta_shared has no `Delta` component to attribute, which is
//     also why EM stays exact;
//   * element (child) flows need the second term ADDED, which is what this file computes.
//
// Defining two reductions over the parent nodes,
//
//     logP[j,r,b] = logsumexp_n ( log f[n,b] + V[j,n,r,b] - logT[n,b] )
//     logQ[j,r,b] = logsumexp_n ( log f[n,b] + V[j,n,r,b] - logZ[n,b] )
//
// every output follows:
//
//     element_flows[c,b] += sum_r exp( U[j,c,r,b] + element_mars[c,b] + logP[j,r,b] )
//     dLL/dU[j,c,r,b]     = exp(U + element_mars + logP) - exp(U + logQ)
//     dLL/dV[j,n,r,b]     = f[n,b] * ( exp(V + logW - logT) - exp(V + logA - logZ) )
//
// `logP` is shared between the child-flow correction and dLL/dU, so the per-sample gradients the
// caller needs come almost free: they ride the same reduction and the same `U` loads.
//
// FUSION: pass A makes one trip over `V` and emits BOTH dLL/dV and the logP/logQ partials; pass C makes
// one trip over `U` and emits BOTH the child-flow correction and dLL/dU. Splitting either would double
// the traffic on the largest tensors in the problem. The `n`- and `c`-splits exist for occupancy: on a
// dense transition `num_node_blocks` and `num_edge_blocks` are both 1, so those are the only wide axes.
//
// All three are launched from ONE host call, for the reason documented in lowrank_forward.cu.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "lowrank_common.cuh"

// ------------------------------- pass A: dLL/dV, and the logP / logQ partials (one trip over V)

__global__ void lowrank_bw_v_kernel(
        const float* __restrict__ ext, const float* __restrict__ node_flows,
        const float* __restrict__ node_mars_T, const float* __restrict__ log_z,
        const float* __restrict__ log_w, const float* __restrict__ log_a,
        const int64_t* __restrict__ nids, const int64_t* __restrict__ xv,
        float* __restrict__ grad_ext, float* __restrict__ p_lp, float* __restrict__ p_lq,
        int batch_size, int num_eblks, int block_size, int rank,
        int n_ntiles, int tile_n, long ext_base, int TB, int accumulate) {

    const int re = blockIdx.x;                    // row * num_eblks + edge block
    const int nt = blockIdx.y;                    // node tile
    const int b0 = blockIdx.z * TB;

    const int row = re / num_eblks;

    // [3 * tile_n * TB]: log f, logT, logZ
    extern __shared__ float s_nb[];

    const int tid = threadIdx.x;
    const int r = tid / TB;
    const int bb = tid % TB;
    const int b = b0 + bb;

    const long nid0 = nids[row];
    const int n0 = nt * tile_n;
    const int n_end0 = min(block_size, n0 + tile_n);
    const int nn = n_end0 - n0;

    // None of log f, logT or logZ depends on the rank, so all `rank` threads sharing an (n, b) were
    // re-loading the same three values -- three dependent global loads per iteration, against one for
    // `V`. Staged once per block, the loop is bound only by the `V` stream.
    float* s_f = s_nb;
    float* s_t = s_nb + nn * TB;
    float* s_z = s_nb + 2 * nn * TB;

    const int nthreads = rank * TB;
    for (int i = tid; i < nn * TB; i += nthreads) {
        const int ni = i / TB;
        const int bg = b0 + (i % TB);
        if (bg < batch_size) {
            const long o = (nid0 + n0 + ni) * batch_size + bg;
            s_f[i] = node_flows[o];
            s_t[i] = node_mars_T[o];
            s_z[i] = log_z[((long)row * block_size + n0 + ni) * batch_size + bg];
        } else {
            s_f[i] = -INFINITY; s_t[i] = -INFINITY; s_z[i] = 0.0f;
        }
    }
    __syncthreads();                              // before any early exit, or the block deadlocks

    if (b >= batch_size) return;

    const long xv_v = xv[re] + ext_base;
    const long wa = ((long)re * rank + r) * batch_size + b;
    const float lw = log_w[wa];
    const float la = log_a[wa];
    LSE acc_p[LSE_WAYS], acc_q[LSE_WAYS];
    #pragma unroll
    for (int w = 0; w < LSE_WAYS; ++w) {
        acc_p[w] = {-INFINITY, 0.0f};
        acc_q[w] = {-INFINITY, 0.0f};
    }

    for (int ns = n0; ns < n_end0; ns += LSE_WAYS) {
      #pragma unroll
      for (int w = 0; w < LSE_WAYS; ++w) {
        const int n = ns + w;
        if (n >= n_end0) continue;

        const int si = (n - n0) * TB + bb;
        const float lf = s_f[si];                          // log f -- `logspace_flows` is required
        const float lt = s_t[si];                                              // node_mars + logZ
        const float lz = s_z[si];

        const long vo = (xv_v + (long)n * rank + r) * batch_size + b;
        const float v = ext[vo];

        // Kept in the exponent rather than forming `f` first, so a tiny flow does not lose the factor.
        const float lfv = lf + v;
        const float g = safe_exp_diff(lfv + lw, lt) - safe_exp_diff(lfv + la, lz);

        // A plain STORE is correct when this node owns its factors: `nids` partition the nodes, so each
        // (n, r, b) is touched exactly once, and it avoids reading a gradient buffer as large as `V`.
        // When several nodes SHARE one factor pair, every one of them contributes to the same gradient,
        // so it has to accumulate. Layers run sequentially and each element is still written once per
        // layer, so a plain read-modify-write suffices -- no atomics -- and the buffer is zeroed once
        // per backward pass, which is the precondition this relies on.
        if (accumulate) {
            grad_ext[vo] += g;
        } else {
            grad_ext[vo] = g;
        }

        lse_add(acc_p[w], safe_log_diff(lfv, lt));
        lse_add(acc_q[w], safe_log_diff(lfv, lz));
      }
    }

    #pragma unroll
    for (int w = 1; w < LSE_WAYS; ++w) {
        lse_merge(acc_p[0], acc_p[w]);
        lse_merge(acc_q[0], acc_q[w]);
    }

    const long po = (((long)re * rank + r) * n_ntiles + nt) * batch_size + b;
    p_lp[po] = lse_get(acc_p[0]);
    p_lq[po] = lse_get(acc_q[0]);
}

// ------------------------------------------------------- pass B: finish the logP / logQ reduction

__global__ void lowrank_bw_pq_reduce_kernel(
        const float* __restrict__ p_lp, const float* __restrict__ p_lq,
        float* __restrict__ log_p, float* __restrict__ log_q,
        int batch_size, int n_ntiles, int n_outputs) {

    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = gtid >> 5;
    const int lane = gtid & 31;
    if (warp >= n_outputs) return;

    const int slot = warp / batch_size;           // (row * num_eblks + j) * rank + r
    const int b = warp % batch_size;

    LSE ap = {-INFINITY, 0.0f}, aq = {-INFINITY, 0.0f};
    for (int nt = lane; nt < n_ntiles; nt += 32) {
        const long o = ((long)slot * n_ntiles + nt) * batch_size + b;
        lse_add(ap, p_lp[o]);
        lse_add(aq, p_lq[o]);
    }

    lse_warp_reduce(ap);
    lse_warp_reduce(aq);

    if (lane == 0) {
        const long o = (long)slot * batch_size + b;
        log_p[o] = lse_get(ap);
        log_q[o] = lse_get(aq);
    }
}


// --------------------- pass C: child-flow correction and dLL/dU (one trip over U)

__global__ void lowrank_bw_u_kernel(
        const float* __restrict__ ext, const float* __restrict__ emars,
        const float* __restrict__ log_p, const float* __restrict__ log_q,
        const int64_t* __restrict__ cids, const int64_t* __restrict__ xu,
        float* __restrict__ grad_ext, float* __restrict__ element_flows,
        int batch_size, int num_edges, int num_eblks, int ch_block_size, int rank,
        int tile_c, long ext_base, int TB, int accumulate) {

    const int re = blockIdx.x;
    const int ct = blockIdx.y;
    const int b0 = blockIdx.z * TB;

    const int row = re / num_eblks;
    const int j = re % num_eblks;

    // One thread owns one (child, batch) pair and walks `rank` itself, so the sum over rank that the
    // child flow needs happens in a register -- no shared memory and no cross-thread reduction. `b` is
    // the fastest-varying thread index, and the factors are batch-innermost, so each rank step is a
    // fully coalesced read across the warp.
    const int tid = threadIdx.x;
    const int c = ct * tile_c + tid / TB;
    const int b = b0 + (tid % TB);
    if (c >= ch_block_size || b >= batch_size) return;

    const long xu_v = xu[re] + ext_base;
    const long cid_base = (long)row * num_edges + (long)j * ch_block_size;
    const float e = emars[cids[cid_base + c] * batch_size + b];

    LSE log_corr = {-INFINITY, 0.0f};

    for (int r = 0; r < rank; ++r) {
        const long pq = ((long)re * rank + r) * batch_size + b;
        const float lp = log_p[pq];
        const float lq = log_q[pq];

        const long uo = (xu_v + (long)c * rank + r) * batch_size + b;
        const float u = ext[uo];

        // The positive term is both dLL/dU's first part and this child's share of the flow
        const float log_t1 = u + e + lp;                   // LOG space -- `log_corr` accumulates logs
        const float g = expf(log_t1) - expf(u + lq);

        if (accumulate) {                          // see the note in pass A
            grad_ext[uo] += g;
        } else {
            grad_ext[uo] = g;
        }

        lse_add(log_corr, log_t1);
    }

    // Atomic because several parent blocks may share a child block: unlike `node_mars` in the forward,
    // `element_flows` is a scatter here.
    atomic_log_add(&element_flows[cids[cid_base + c] * batch_size + b], lse_get(log_corr));
}

// --------------------------------------- turning node_mars into logT and back again

__global__ void lowrank_shift_logz_kernel(
        float* __restrict__ node_mars, const int64_t* __restrict__ nids,
        const float* __restrict__ log_z, int batch_size, int block_size, int rows, float sign) {

    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)rows * block_size * batch_size;
    if (i >= total) return;

    const int b = i % batch_size;
    const int m = (i / batch_size) % block_size;
    const int row = i / ((long)batch_size * block_size);

    node_mars[(nids[row] + m) * batch_size + b] += sign * log_z[i];
}

void lowrank_shift_logz(torch::Tensor node_mars, torch::Tensor nids, torch::Tensor log_z,
                        int64_t block_size, double sign) {
    const int batch_size = node_mars.size(1);
    const int rows = nids.size(0);
    const long total = (long)rows * block_size * batch_size;

    const int threads = 256;
    lowrank_shift_logz_kernel<<<(int)((total + threads - 1) / threads), threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        node_mars.data_ptr<float>(), nids.data_ptr<int64_t>(), log_z.data_ptr<float>(),
        batch_size, (int)block_size, rows, (float)sign);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --------------------------------------------------------------------------------------- launcher

static inline int bw_cdiv(int a, int b) { return (a + b - 1) / b; }

void lowrank_backward(torch::Tensor node_flows, torch::Tensor element_flows,
                      torch::Tensor node_mars_T, torch::Tensor element_mars,
                      torch::Tensor ext, torch::Tensor grad_ext,
                      torch::Tensor nids, torch::Tensor cids,
                      torch::Tensor xu, torch::Tensor xv,
                      torch::Tensor log_w, torch::Tensor log_a, torch::Tensor log_z,
                      torch::Tensor p_lp, torch::Tensor p_lq,
                      torch::Tensor log_p, torch::Tensor log_q,
                      int64_t block_size, int64_t ch_block_size, int64_t rank, int64_t ext_base,
                      int64_t tile_n, int64_t tile_c, int64_t tb, bool accumulate) {

    const int batch_size = node_flows.size(1);
    const int rows = xu.size(0);
    const int num_eblks = xu.size(1);
    const int num_edges = cids.size(1);
    const int n_ntiles = bw_cdiv((int)block_size, (int)tile_n);

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 gA(rows * num_eblks, n_ntiles, bw_cdiv(batch_size, (int)tb));
    const size_t smemA = (size_t)3 * tile_n * tb * sizeof(float);
    lowrank_bw_v_kernel<<<gA, (int)(rank * tb), smemA, stream>>>(
        ext.data_ptr<float>(), node_flows.data_ptr<float>(),
        node_mars_T.data_ptr<float>(), log_z.data_ptr<float>(),
        log_w.data_ptr<float>(), log_a.data_ptr<float>(),
        nids.data_ptr<int64_t>(), xv.data_ptr<int64_t>(),
        grad_ext.data_ptr<float>(), p_lp.data_ptr<float>(), p_lq.data_ptr<float>(),
        batch_size, num_eblks, (int)block_size, (int)rank,
        n_ntiles, (int)tile_n, (long)ext_base, (int)tb, accumulate ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int n_outputs = rows * num_eblks * (int)rank * batch_size;
    lowrank_bw_pq_reduce_kernel<<<bw_cdiv(n_outputs * 32, 256), 256, 0, stream>>>(
        p_lp.data_ptr<float>(), p_lq.data_ptr<float>(),
        log_p.data_ptr<float>(), log_q.data_ptr<float>(),
        batch_size, n_ntiles, n_outputs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 gC(rows * num_eblks, bw_cdiv((int)ch_block_size, (int)tile_c),
            bw_cdiv(batch_size, (int)tb));
    lowrank_bw_u_kernel<<<gC, (int)(tile_c * tb), 0, stream>>>(
        ext.data_ptr<float>(), element_mars.data_ptr<float>(),
        log_p.data_ptr<float>(), log_q.data_ptr<float>(),
        cids.data_ptr<int64_t>(), xu.data_ptr<int64_t>(),
        grad_ext.data_ptr<float>(), element_flows.data_ptr<float>(),
        batch_size, num_edges, num_eblks, (int)ch_block_size, (int)rank,
        (int)tile_c, (long)ext_base, (int)tb, accumulate ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
