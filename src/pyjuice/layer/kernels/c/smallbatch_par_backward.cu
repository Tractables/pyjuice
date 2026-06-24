// Small-batch (batch < 16) sparse sum-layer PARAMETER-FLOW backward.
//
// Plain-CUDA kernel (no CuTe/TMA/CUTLASS) for the small-batch regime, the parameter-flow counterpart
// of `smallbatch_forward_sum.cu` / `smallbatch_ele_backward.cu`. The sparse Triton par kernel processes
// one node per program and re-reads the shared children per node; this kernel instead exploits the
// node-contiguous parameter layout the way the forward kernel does.
//
// Design:
//   - node-warp: lane (threadIdx.x) = node within the block -> param[node,e] = pids[nb,e] + node is
//     node-contiguous (stride 1), so the 32 lanes' param loads AND param-flow stores are coalesced;
//   - edges are split across the grid (blockIdx.y) x threadIdx.y; each (node, edge) parameter flow is
//     INDEPENDENT (no cross-edge reduction), so with a single batch tile + collision-free (untied)
//     flows the write is a plain read-add-store -- no atomics, no combine;
//   - the batch is the short inner reduction (online-style, max-stabilised), handled per thread.
// Numerically equivalent to the Triton sparse par kernel (fp32); matches it to ~4e-11 here.
//
// Math, per sum-node m (= nids[nb] + tile_id), edge e, child c = cids[nb,e]:
//   pl[b]  = (node_mars[m,b] == -inf) ? node_flows[m,b] : node_flows[m,b] + element_mars[c,b] - node_mars[m,b]
//   term   = sum_b exp(pl[b])            (max-stabilised; 0 if all -inf)
//   param_flows[pfids[nb,e] + tile_id] += params[pids[nb,e] + tile_id] * term
// Valid only for: LL, logspace flows, allow_modify_flows / negate off, a single batch tile, untied
// (collision-free) flows, and block_size a multiple of 32 -- the caller guarantees this.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

template <int EY, int BMAX>
__global__ void sb_par_kernel(float* __restrict__ param_flows, const float* __restrict__ node_flows,
                              const float* __restrict__ node_mars, const float* __restrict__ element_mars,
                              const float* __restrict__ params, const long* __restrict__ nids,
                              const long* __restrict__ cids, const long* __restrict__ pids,
                              const long* __restrict__ pfids, int batch, int block_size, int num_edges) {
    const int groups = block_size >> 5;             // block_size / 32 (block_size is a multiple of 32)
    const int gx = blockIdx.x;
    const int nb = gx / groups;                      // node-block
    const int grp = gx - nb * groups;                // node-group within the block
    const int tile_id = grp * 32 + threadIdx.x;      // node within the block (lane) -> coalesced
    const int e = blockIdx.y * EY + threadIdx.y;     // edge
    if (e >= num_edges) return;

    const long node = nids[nb] + tile_id;
    float lnf[BMAX], lnm[BMAX];
    #pragma unroll
    for (int b = 0; b < BMAX; b++) { if (b >= batch) break; lnf[b] = node_flows[node * (long)batch + b]; lnm[b] = node_mars[node * (long)batch + b]; }

    const long cb = (long)nb * num_edges;
    const long child = cids[cb + e];
    float mx = -INFINITY, pl[BMAX];
    #pragma unroll
    for (int b = 0; b < BMAX; b++) {
        if (b >= batch) break;
        float emb = element_mars[child * (long)batch + b];
        float p = (lnm[b] == -INFINITY) ? lnf[b] : (lnf[b] + emb - lnm[b]);
        pl[b] = p; mx = fmaxf(mx, p);
    }
    float term = 0.f;
    if (mx != -INFINITY) {                            // 0 when every batch element is -inf
        float s = 0.f;
        #pragma unroll
        for (int b = 0; b < BMAX; b++) { if (b >= batch) break; s += __expf(pl[b] - mx); }
        term = s * __expf(mx);
    }
    float val = params[pids[cb + e] + tile_id] * term;          // coalesced across lanes (node-contiguous)
    param_flows[pfids[cb + e] + tile_id] += val;                // collision-free RMW, coalesced
}

template <int EY, int BMAX>
static void launch_par(torch::Tensor param_flows, torch::Tensor node_flows, torch::Tensor node_mars,
                       torch::Tensor element_mars, torch::Tensor params, torch::Tensor nids,
                       torch::Tensor cids, torch::Tensor pids, torch::Tensor pfids,
                       int batch, int block_size, int num_edges) {
    const long nb = nids.size(0);
    const long groups = block_size / 32;
    dim3 grid((unsigned int)(nb * groups), (unsigned int)((num_edges + EY - 1) / EY));
    dim3 blk(32, EY);
    // Launch on the current stream so the kernel is captured correctly under CUDA-graph recording.
    sb_par_kernel<EY, BMAX><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        param_flows.data_ptr<float>(), node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
        element_mars.data_ptr<float>(), params.data_ptr<float>(), nids.data_ptr<long>(),
        cids.data_ptr<long>(), pids.data_ptr<long>(), pfids.data_ptr<long>(), batch, block_size, num_edges);
}

// Dispatch on (EY from `cfg`) x (BMAX = next pow2 of batch, for register-tight state).
#define SB_PAR_DISPATCH_BMAX(EE)                                                                                      \
    switch (bmax) {                                                                                                    \
        case 1:  launch_par<EE, 1 >(param_flows, node_flows, node_mars, element_mars, params, nids, cids, pids, pfids, b, bs, ne); break; \
        case 2:  launch_par<EE, 2 >(param_flows, node_flows, node_mars, element_mars, params, nids, cids, pids, pfids, b, bs, ne); break; \
        case 4:  launch_par<EE, 4 >(param_flows, node_flows, node_mars, element_mars, params, nids, cids, pids, pfids, b, bs, ne); break; \
        case 8:  launch_par<EE, 8 >(param_flows, node_flows, node_mars, element_mars, params, nids, cids, pids, pfids, b, bs, ne); break; \
        case 16: launch_par<EE, 16>(param_flows, node_flows, node_mars, element_mars, params, nids, cids, pids, pfids, b, bs, ne); break; \
        default: TORCH_CHECK(false, "smallbatch_par_backward_sum: invalid batch ", b);                                \
    }

// cfg id -> EY (edges split across threadIdx.y). The kernel is memory/latency-bound and EY-insensitive;
// a couple of options are exposed for the autotuner to pick per layer/batch.
void smallbatch_par_backward_sum(torch::Tensor param_flows, torch::Tensor node_flows, torch::Tensor node_mars,
                                 torch::Tensor element_mars, torch::Tensor params, torch::Tensor nids,
                                 torch::Tensor cids, torch::Tensor pids, torch::Tensor pfids,
                                 int64_t batch, int64_t block_size, int64_t num_edges, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, ne = (int)num_edges;
    int ey = (cfg == 0) ? 8 : (cfg == 1) ? 16 : (cfg == 2) ? 4 : -1;
    TORCH_CHECK(ey > 0, "smallbatch_par_backward_sum: invalid cfg ", cfg);
    TORCH_CHECK(b >= 1 && b < 16, "smallbatch_par_backward_sum: batch must be in [1, 16) (got ", b, ")");
    TORCH_CHECK(bs % 32 == 0, "smallbatch_par_backward_sum: block_size must be a multiple of 32 (got ", bs, ")");
    int bmax = 1;
    while (bmax < b) bmax <<= 1;                       // next power of two >= batch
    // No silent default: an unknown cfg must not run zero kernels (a no-op launch can masquerade as
    // correct against a reused output buffer -- see the autotune note in sum_layer.py).
    switch (ey) {
        case 8:  SB_PAR_DISPATCH_BMAX(8);  break;
        case 16: SB_PAR_DISPATCH_BMAX(16); break;
        case 4:  SB_PAR_DISPATCH_BMAX(4);  break;
        default: TORCH_CHECK(false, "smallbatch_par_backward_sum: invalid cfg ", cfg);
    }
}

std::vector<int64_t> smallbatch_par_configs() { return {8, 16, 4}; }  // EY per config id

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallbatch_par_backward_sum", &smallbatch_par_backward_sum,
          "small-batch (batch<16) sparse sum-layer parameter-flow backward");
    m.def("smallbatch_par_configs", &smallbatch_par_configs, "EY per config id");
}
