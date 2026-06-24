// Small-batch (batch < 16) block-sparse sum-layer FORWARD.
//
// This is a plain-CUDA kernel (no CuTe/TMA/CUTLASS) for the regime the block-sparse Triton kernels
// handle poorly: large block size (>= 128) at tiny batch, where the sparse kernels leave the node
// dimension un-tiled (~1 SM busy) and the block-sparse heuristic collapses to too-few node tiles.
//
// Design (the "v4" structure, after sweeping v1..v9 + 5 pipelining variants on sm_120):
//   - one warp = 32 consecutive nodes (lane = node offset) -> fully-coalesced 128B param loads;
//   - blockDim.y = SPLIT edge-warps that split the edge reduction for SM occupancy;
//   - each (node, edge-warp) runs an online-logsumexp over its edge subset; the SPLIT partials are
//     combined per node in shared memory (cheap, no atomics, no second pass);
//   - grid.x enumerates (node-block, node-group-within-block); grid.y = batch.
// Numerically equivalent to the Triton block-sparse forward (per-tile online-logsumexp, fp32);
// matches it to ~1.5e-6 in log-space, well under pyjuice's 1.5e-3 accuracy bar.
//
// Math, per (node m, batch b), for node-block nb:
//   node_mars[node*batch + b] = log( sum_e params[pbase[nb] + e*block_size + m_local]
//                                      * exp(element_mars[(ebase[nb] + e)*batch + b]) )
// where node = nids[nb] + m_local, m_local in [0, block_size), e in [0, num_edges). This assumes the
// node-block's children are CONTIGUOUS (child(e) = ebase[nb] + e) and its params are block_size-
// strided across edges -- the caller verifies this (cuda_ok) and falls back to Triton otherwise.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <int SPLIT>
__global__ void sb_fwd_kernel(float* __restrict__ node_mars, const float* __restrict__ element_mars,
                              const float* __restrict__ params, const long* __restrict__ nids,
                              const long* __restrict__ ebase, const long* __restrict__ pbase,
                              int batch, int block_size, int num_edges) {
    const int groups = block_size >> 5;          // block_size / 32  (block_size is a multiple of 32)
    const int gx = blockIdx.x;
    const int nb = gx / groups;                   // node-block
    const int grp = gx - nb * groups;             // node-group within the block
    const int b = blockIdx.y;                      // batch element
    const int lane = threadIdx.x;                  // 0..31  -> node offset within the group
    const int ty = threadIdx.y;                    // 0..SPLIT-1  -> edge-warp
    const int m_local = grp * 32 + lane;

    const long eb = ebase[nb], pb = pbase[nb];
    float m = -INFINITY, l = 0.f;                  // running max / linear-sum (online logsumexp)
    for (int e = ty; e < num_edges; e += SPLIT) {
        float emar = element_mars[(eb + e) * batch + b];     // broadcast across the warp's lanes
        float p = params[pb + (long)e * block_size + m_local]; // coalesced: 32 consecutive params
        float nmx = fmaxf(m, emar);
        l = l * __expf(m - nmx) + p * __expf(emar - nmx);
        m = nmx;
    }

    __shared__ float sM[SPLIT][32];
    __shared__ float sL[SPLIT][32];
    sM[ty][lane] = m;
    sL[ty][lane] = l;
    __syncthreads();

    if (ty == 0) {
        float gm = -INFINITY;
        #pragma unroll
        for (int k = 0; k < SPLIT; k++) gm = fmaxf(gm, sM[k][lane]);
        float L = 0.f;
        #pragma unroll
        for (int k = 0; k < SPLIT; k++) L += sL[k][lane] * __expf(sM[k][lane] - gm);
        const long node = nids[nb] + m_local;
        node_mars[node * (long)batch + b] = gm + logf(L);
    }
}

template <int SPLIT>
static void launch_sb(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                      torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase,
                      int batch, int block_size, int num_edges) {
    const long ng = nids.size(0);
    const long groups = block_size / 32;
    dim3 grid((unsigned int)(ng * groups), (unsigned int)batch);   // ng*groups << 2^31 in practice
    dim3 blk(32, SPLIT);
    sb_fwd_kernel<SPLIT><<<grid, blk>>>(
        node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), params.data_ptr<float>(),
        nids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(),
        batch, block_size, num_edges);
}

// cfg id -> SPLIT (edge-warps). SPLIT=32 was the sweep winner; the autotuner picks per layer/batch.
void smallbatch_forward_sum(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                            torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase,
                            int64_t batch, int64_t block_size, int64_t num_edges, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, ne = (int)num_edges;
    TORCH_CHECK(bs % 32 == 0, "smallbatch_forward_sum: block_size must be a multiple of 32 (got ", bs, ")");
    switch (cfg) {
        case 0: launch_sb<32>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, ne); break;
        case 1: launch_sb<16>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, ne); break;
        case 2: launch_sb<24>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, ne); break;
        case 3: launch_sb<8>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, ne); break;
        // No silent default: an unknown cfg must not run zero kernels (a no-op launch can masquerade
        // as correct against a reused output buffer -- see the autotune note in sum_layer.py).
        default: TORCH_CHECK(false, "smallbatch_forward_sum: invalid cfg ", cfg);
    }
}

std::vector<int64_t> smallbatch_fw_configs() { return {32, 16, 24, 8}; }  // SPLIT per config id

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallbatch_forward_sum", &smallbatch_forward_sum,
          "small-batch (batch<16) block-sparse sum-layer forward");
    m.def("smallbatch_fw_configs", &smallbatch_fw_configs, "SPLIT per config id");
}
