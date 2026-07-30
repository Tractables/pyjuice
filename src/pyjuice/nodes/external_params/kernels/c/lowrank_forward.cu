// CUDA implementation of the per-sample low-rank sum-layer forward correction.
//
// Rewrites `node_mars` from the shared-parameter value `log S1` to the effective value under
// theta_b = normalize_children(theta_shared + exp(U_b) exp(V_b)^T):
//
//     logW[j,r,b] = logsumexp_c ( U[j,c,r,b] + element_mars[cids[j,c], b] )
//     logA[j,r,b] = logsumexp_c   U[j,c,r,b]
//     logS2[m,b]  = logsumexp_{j,r} ( V[j,m,r,b] + logW[j,r,b] )
//     logZ[m,b]   = logaddexp( 0, logsumexp_{j,r} ( V[j,m,r,b] + logA[j,r,b] ) )
//     node_mars   = logaddexp(logS1, logS2) - logZ
//
// The `0` in logZ is the shared parameters' own contribution: pyjuice keeps them child-normalized,
// so their total is exactly 1.
//
// WHY CUDA RATHER THAN TRITON, for a kernel that needs no tensor cores and no TMA:
// this correction is issued once per layer, and on a deep chain (a 32-step HMM = 31 sum layers) the
// binding cost of the launch, not the launch's work, sets the wall clock. MEASURED at batch 64: the
// Triton form spends ~11 us of Python per launch x 2 launches x 31 layers, which pushed the forward
// from 75% CPU-bound to 100% CPU-bound -- the GPU sat idle waiting to be fed, and 62 launches whose
// bodies were replaced by `return` still cost 0.80 ms. Both phases are therefore launched from ONE
// pybind call, with the grid arithmetic done here in C++, so the whole correction for a layer costs
// the host a single ~2-3 us call.
//
// The two phases cannot be one kernel: phase 2 needs logW/logA reduced over ALL children, which is a
// grid-wide reduction. Splitting the child axis across blocks (rather than giving one block the whole
// reduction) is what keeps both phases occupied -- on a dense transition, `num_node_blocks` and
// `num_edge_blocks` are both 1, so the child axis is the only wide one.
//
// Layout note: the factors are stored batch-innermost, `[E, states, rank, B]`, so consecutive threads
// (which differ in `b`) read consecutive addresses in every load below.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "lowrank_common.cuh"

// ---------------------------------------------------------------- phase 1: partial logW / logA

__global__ void lowrank_wa_partial_kernel(
        const float* __restrict__ ext, const float* __restrict__ emars,
        const int64_t* __restrict__ cids, const int64_t* __restrict__ xu,
        float* __restrict__ pw, float* __restrict__ pa,
        int batch_size, int num_edges, int num_eblks, int ch_block_size,
        int rank, int n_ctiles, int tile_c, long ext_base, int TB) {

    const int re = blockIdx.x;                    // row * num_eblks + edge block
    const int ct = blockIdx.y;                    // child tile
    const int b0 = blockIdx.z * TB;

    const int row = re / num_eblks;
    const int j = re % num_eblks;

    // [tile_c] child ids, then [tile_c * TB] staged child values
    extern __shared__ char s_raw[];

    const int tid = threadIdx.x;                  // = r * TB + (b - b0)
    const int r = tid / TB;
    const int bb = tid % TB;
    const int b = b0 + bb;

    const long xu_v = xu[re] + ext_base;
    const long cid_base = (long)row * num_edges + (long)j * ch_block_size;

    const int c0 = ct * tile_c;
    const int c_end0 = min(ch_block_size, c0 + tile_c);
    const int nc = c_end0 - c0;

    // `element_mars[cids[c], b]` does not depend on the rank, so every one of the `rank` threads that
    // shares a (c, b) was re-loading it -- and it sat in the inner loop as a second DEPENDENT global
    // load, next to the `cids` lookup feeding it. Staged once here, the loop keeps one global load (U)
    // and reads the child value from shared memory, which is what the latency chain is bound by.
    const int nthreads = rank * TB;

    int* s_cid = reinterpret_cast<int*>(s_raw);
    float* s_em = reinterpret_cast<float*>(s_raw + ((tile_c * sizeof(int) + 15) & ~15));

    // TWO passes, not one: fusing them makes every staged value a `cids` load followed by a DEPENDENT
    // `element_mars` load, so a thread has one request in flight and the prologue becomes its own serial
    // chain (measured worse than not staging at all). Loading the ids first leaves the second pass with
    // mutually independent global loads, which is what fills the memory pipeline.
    for (int i = tid; i < nc; i += nthreads) {
        s_cid[i] = (int) cids[cid_base + c0 + i];
    }
    __syncthreads();

    for (int i = tid; i < nc * TB; i += nthreads) {
        const int bg = b0 + (i % TB);
        s_em[i] = (bg < batch_size) ? emars[(long)s_cid[i / TB] * batch_size + bg] : -INFINITY;
    }
    __syncthreads();                              // before any early exit, or the block deadlocks

    if (b >= batch_size) return;

    LSE acc_w[LSE_WAYS], acc_a[LSE_WAYS];
    #pragma unroll
    for (int w = 0; w < LSE_WAYS; ++w) {
        acc_w[w] = {-INFINITY, 0.0f};
        acc_a[w] = {-INFINITY, 0.0f};
    }

    for (int c = c0; c < c_end0; c += LSE_WAYS) {
        #pragma unroll
        for (int w = 0; w < LSE_WAYS; ++w) {
            const int cc = c + w;
            if (cc < c_end0) {
                const float u = ext[(xu_v + (long)cc * rank + r) * batch_size + b];
                const float e = s_em[(cc - c0) * TB + bb];
                lse_add(acc_a[w], u);
                lse_add(acc_w[w], u + e);
            }
        }
    }

    #pragma unroll
    for (int w = 1; w < LSE_WAYS; ++w) {
        lse_merge(acc_w[0], acc_w[w]);
        lse_merge(acc_a[0], acc_a[w]);
    }

    const long o = (((long)re * rank + r) * n_ctiles + ct) * batch_size + b;
    pw[o] = lse_get(acc_w[0]);
    pa[o] = lse_get(acc_a[0]);
}

// --------------------------------------------------- phase 1b: finish logW / logA, ONCE per layer
//
// Phase 2 used to fold this reduction into itself, but its grid carries a node dimension and logW/logA
// do not depend on the node index -- so every one of the `block_size / TILE_M` node tiles redid the same
// `n_ctiles`-long reduction. With a small `tile_c` (which is what makes phase 1 fast) that is 64 x 128
// redundant reductions per row, and it made phase 2 the most expensive kernel in the pass (2.7 ms at
// r=16, B=256, 17% of peak). Reduced once here, phase 2 just reads two small vectors.
__global__ void lowrank_wa_reduce_kernel(
        const float* __restrict__ pw, const float* __restrict__ pa,
        float* __restrict__ log_w, float* __restrict__ log_a,
        int batch_size, int n_ctiles, int n_outputs) {

    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = gtid >> 5;
    const int lane = gtid & 31;
    if (warp >= n_outputs) return;

    const int slot = warp / batch_size;           // (row * num_eblks + j) * rank + r
    const int b = warp % batch_size;

    LSE aw = {-INFINITY, 0.0f}, aa = {-INFINITY, 0.0f};
    for (int ct = lane; ct < n_ctiles; ct += 32) {
        const long o = ((long)slot * n_ctiles + ct) * batch_size + b;
        lse_add(aw, pw[o]);
        lse_add(aa, pa[o]);
    }

    lse_warp_reduce(aw);
    lse_warp_reduce(aa);

    if (lane == 0) {
        const long o = (long)slot * batch_size + b;
        log_w[o] = lse_get(aw);
        log_a[o] = lse_get(aa);
    }
}


// ------------------------------------------- phase 2: combine with V and normalize

__global__ void lowrank_combine_kernel(
        float* __restrict__ node_mars, const float* __restrict__ ext,
        const int64_t* __restrict__ nids, const int64_t* __restrict__ xv,
        const float* __restrict__ log_w, const float* __restrict__ log_a,
        float* __restrict__ log_z_out,
        int batch_size, int num_eblks, int block_size, int rank,
        long ext_base, int TM, int TB) {

    // [num_eblks * rank * TB] for logW, then the same for logA
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    const int m0 = blockIdx.y * TM;
    const int b0 = blockIdx.z * TB;

    const int nthreads = TM * TB;
    const int tid = threadIdx.x;

    float* s_w = smem;
    float* s_a = smem + num_eblks * rank * TB;

    // Just a staged COPY now -- phase 1b already reduced these. Kept in shared memory because every
    // node this block owns reads the same `rank` values.
    const int total = num_eblks * rank * TB;
    for (int i = tid; i < total; i += nthreads) {
        const int bb = i % TB;
        const int rr = (i / TB) % rank;
        const int j = i / (TB * rank);
        const int b = b0 + bb;

        if (b < batch_size) {
            const long o = (((long)row * num_eblks + j) * rank + rr) * batch_size + b;
            s_w[i] = log_w[o];
            s_a[i] = log_a[o];
        } else {
            s_w[i] = -INFINITY;
            s_a[i] = -INFINITY;
        }
    }
    __syncthreads();

    const int m = m0 + tid / TB;
    const int bb = tid % TB;
    const int b = b0 + bb;
    if (m >= block_size || b >= batch_size) return;

    float log_s2 = -INFINITY, log_zt = -INFINITY;

    for (int j = 0; j < num_eblks; ++j) {
        const long xv_v = xv[(long)row * num_eblks + j] + ext_base;
        for (int rr = 0; rr < rank; ++rr) {
            const float v = ext[(xv_v + (long)m * rank + rr) * batch_size + b];
            const int si = (j * rank + rr) * TB + bb;
            log_s2 = logaddexp(log_s2, v + s_w[si]);
            log_zt = logaddexp(log_zt, v + s_a[si]);
        }
    }

    const long o = (nids[row] + m) * batch_size + b;
    const float log_s1 = node_mars[o];
    const float log_z = logaddexp(0.0f, log_zt);

    node_mars[o] = logaddexp(log_s1, log_s2) - log_z;

    // `logZ` is what turns `node_mars` back into `logT` for the backward, and it is also the
    // normalizer term of dLL/dV. Node-sized, so far cheaper than recomputing it from `V`.
    if (log_z_out != nullptr) {
        log_z_out[((long)row * block_size + m) * batch_size + b] = log_z;
    }
}

// --------------------------------------------------------------------------------------- launcher

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

void lowrank_forward(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor ext,
                     torch::Tensor nids, torch::Tensor cids, torch::Tensor xu, torch::Tensor xv,
                     torch::Tensor pw, torch::Tensor pa,
                     torch::Tensor log_w, torch::Tensor log_a, torch::Tensor log_z,
                     int64_t block_size, int64_t ch_block_size, int64_t rank, int64_t ext_base,
                     int64_t tile_c, int64_t tile_m, int64_t tb1, int64_t tb2) {

    const int batch_size = node_mars.size(1);
    const int rows = xu.size(0);
    const int num_eblks = xu.size(1);
    const int num_edges = cids.size(1);
    const int n_ctiles = cdiv((int)ch_block_size, (int)tile_c);

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 g1(rows * num_eblks, n_ctiles, cdiv(batch_size, (int)tb1));
    const size_t smem1 = ((tile_c * sizeof(int) + 15) & ~15) + (size_t)tile_c * tb1 * sizeof(float);
    lowrank_wa_partial_kernel<<<g1, (int)(rank * tb1), smem1, stream>>>(
        ext.data_ptr<float>(), element_mars.data_ptr<float>(),
        cids.data_ptr<int64_t>(), xu.data_ptr<int64_t>(),
        pw.data_ptr<float>(), pa.data_ptr<float>(),
        batch_size, num_edges, num_eblks, (int)ch_block_size,
        (int)rank, n_ctiles, (int)tile_c, (long)ext_base, (int)tb1);

    const int n_outputs = rows * num_eblks * (int)rank * batch_size;
    lowrank_wa_reduce_kernel<<<cdiv(n_outputs * 32, 256), 256, 0, stream>>>(
        pw.data_ptr<float>(), pa.data_ptr<float>(),
        log_w.data_ptr<float>(), log_a.data_ptr<float>(),
        batch_size, n_ctiles, n_outputs);

    dim3 g2(rows, cdiv((int)block_size, (int)tile_m), cdiv(batch_size, (int)tb2));
    const size_t smem = 2 * num_eblks * rank * tb2 * sizeof(float);
    lowrank_combine_kernel<<<g2, (int)(tile_m * tb2), smem, stream>>>(
        node_mars.data_ptr<float>(), ext.data_ptr<float>(),
        nids.data_ptr<int64_t>(), xv.data_ptr<int64_t>(),
        log_w.data_ptr<float>(), log_a.data_ptr<float>(),
        log_z.numel() ? log_z.data_ptr<float>() : nullptr,
        batch_size, num_eblks, (int)block_size, (int)rank,
        (long)ext_base, (int)tile_m, (int)tb2);
}

// Defined in lowrank_backward.cu; both sources are compiled into this one extension.
void lowrank_backward(torch::Tensor node_flows, torch::Tensor element_flows,
                      torch::Tensor node_mars_T, torch::Tensor element_mars,
                      torch::Tensor ext, torch::Tensor grad_ext,
                      torch::Tensor nids, torch::Tensor cids,
                      torch::Tensor xu, torch::Tensor xv,
                      torch::Tensor log_w, torch::Tensor log_a, torch::Tensor log_z,
                      torch::Tensor p_lp, torch::Tensor p_lq,
                      torch::Tensor log_p, torch::Tensor log_q,
                      int64_t block_size, int64_t ch_block_size, int64_t rank, int64_t ext_base,
                      int64_t tile_n, int64_t tile_c, int64_t tb);

void lowrank_shift_logz(torch::Tensor node_mars, torch::Tensor nids, torch::Tensor log_z,
                        int64_t block_size, double sign);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_forward", &lowrank_forward,
          "Fused per-sample low-rank sum-layer forward correction (both phases, one host call)");
    m.def("lowrank_backward", &lowrank_backward,
          "Per-sample low-rank sum-layer backward: child-flow correction + dLL/dU, dLL/dV");
    m.def("lowrank_shift_logz", &lowrank_shift_logz,
          "Add (or subtract) logZ over a layer's node range, turning node_mars into logT");
}
