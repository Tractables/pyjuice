// Small-batch (batch < 16) block-sparse sum-layer ELEMENT-FLOW backward.
//
// Plain-CUDA kernel (no CuTe/TMA/CUTLASS) for the small-batch regime, the backward counterpart of
// `smallbatch_forward_sum.cu`. The block-sparse Triton kernels under-tile the node dimension at tiny
// batch; this kernel instead gives every child node its own warp.
//
// Design (the "D1a" structure, after profiling v1..v9: the earlier variants STAGED the per-parent
// `log_n_fdm` into shared memory behind a block-wide __syncthreads, which a component ablation showed
// was ~88% of the runtime -- a serializing barrier plus redundant full-array loads, since every
// child re-staged the same parents. D1a deletes the staging and the barrier entirely):
//   - one warp = one child node (lane = a 1/32 stride of that child's edges);
//   - each lane streams ITS OWN edges' node_flows / node_mars directly from global into registers and
//     reduces with an online-logsumexp -- no shared memory, no barrier;
//   - blockDim.y = WARPS child-warps per block; grid.x enumerates (node-block, child-tile-in-block).
// Numerically equivalent to the Triton ele kernel (fp32 online-logsumexp); matches it to ~3e-6 in
// log-space, well under pyjuice's 1.5e-3 accuracy bar.
//
// Math, per child node = chids[eb] + m_local (m_local in [0, cs_block_size)), batch b:
//   log_n_fdm[par, b] = node_flows[par*batch + b] - node_mars[par*batch + b]      (= node_flows when
//                       node_mars is -inf), with par = ebase[eb] + e, e in [0, num_edges);
//   element_flows[node*batch + b] = element_mars[node*batch + b]
//                                   + log( sum_e params[pbase[eb] + m_local*block_size + e]
//                                          * exp(log_n_fdm[par, b]) ).
// This assumes the parents are CONTIGUOUS (par = ebase[eb] + e) and the per-child params are edge-
// contiguous (stride 1 in e) -- the caller verifies this global contiguity (sb_ok) and falls back to
// Triton otherwise. `BMAX` is the power-of-two >= batch, used only to size the per-lane register
// state tightly (e.g. 2 for batch == 2), so tiny batch carries no register waste.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// Combine two online-logsumexp accumulators (max `m`, linear-sum `l`) with a second point/accumulator
// (`om`, `ol`): result represents l*exp(m) + ol*exp(om), kept numerically stable at the joint max.
// The -inf guards avoid (-inf) - (-inf) = NaN when an empty accumulator meets an -inf term.
__device__ __forceinline__ void lse_merge(float& m, float& l, float om, float ol) {
    float nmx = fmaxf(m, om);
    float a = (m == -INFINITY) ? 0.f : l * __expf(m - nmx);
    float c = (om == -INFINITY) ? 0.f : ol * __expf(om - nmx);
    l = a + c;
    m = nmx;
}

template <int WARPS, int BMAX>
__global__ void sb_ele_kernel(float* __restrict__ element_flows, const float* __restrict__ element_mars,
                              const float* __restrict__ node_flows, const float* __restrict__ node_mars,
                              const float* __restrict__ params, const long* __restrict__ chids,
                              const long* __restrict__ ebase, const long* __restrict__ pbase,
                              int batch, int block_size, int cs_block_size, int num_edges) {
    const int lane = threadIdx.x;                       // 0..31  -> a 1/32 stride of the edges
    const int w = threadIdx.y;                          // 0..WARPS-1  -> child within this block's tile
    const int tiles_per_eb = cs_block_size / WARPS;     // child-tiles per node-block
    const int eb = blockIdx.x / tiles_per_eb;           // node-block
    const int tile = blockIdx.x - eb * tiles_per_eb;    // child-tile within the node-block
    const int m_local = tile * WARPS + w;               // child offset within the node-block

    const long eb0 = ebase[eb];                          // first parent node of this node-block
    const long pb0 = pbase[eb] + (long)m_local * block_size;  // first param of this child's row

    float mm[BMAX], ll[BMAX];                            // per-batch running max / linear-sum
    #pragma unroll
    for (int b = 0; b < BMAX; b++) { mm[b] = -INFINITY; ll[b] = 0.f; }

    for (int e = lane; e < num_edges; e += 32) {
        const float wgt = params[pb0 + e];               // epar[m_local, e]; coalesced across lanes
        const float* nf = node_flows + (eb0 + e) * (long)batch;
        const float* nm = node_mars + (eb0 + e) * (long)batch;
        #pragma unroll
        for (int b = 0; b < BMAX; b++) {
            if (b >= batch) break;
            float nmar = nm[b];
            float term = (nmar == -INFINITY) ? nf[b] : (nf[b] - nmar);   // log_n_fdm[par, b]
            lse_merge(mm[b], ll[b], term, wgt);
        }
    }

    const long node = chids[eb] + m_local;
    #pragma unroll
    for (int b = 0; b < BMAX; b++) {
        if (b >= batch) break;
        float m = mm[b], l = ll[b];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)            // warp-reduce the 32 lanes' accumulators
            lse_merge(m, l, __shfl_xor_sync(0xffffffff, m, off), __shfl_xor_sync(0xffffffff, l, off));
        if (lane == 0) {
            float r = (m == -INFINITY) ? -INFINITY : (logf(l) + m);
            if (r != -INFINITY) r += element_mars[node * (long)batch + b];
            element_flows[node * (long)batch + b] = r;
        }
    }
}

template <int WARPS, int BMAX>
static void launch_ele(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                       torch::Tensor node_mars, torch::Tensor params, torch::Tensor chids,
                       torch::Tensor ebase, torch::Tensor pbase,
                       int batch, int block_size, int cs_block_size, int num_edges) {
    const long neb = chids.size(0);
    dim3 grid((unsigned int)(neb * (cs_block_size / WARPS)));   // neb*tiles << 2^31 in practice
    dim3 blk(32, WARPS);
    // Launch on the current stream (not the default stream) so the kernel is captured correctly when
    // pyjuice records a CUDA graph of the backward pass (see cat_backward.cu).
    sb_ele_kernel<WARPS, BMAX><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), node_flows.data_ptr<float>(),
        node_mars.data_ptr<float>(), params.data_ptr<float>(), chids.data_ptr<long>(),
        ebase.data_ptr<long>(), pbase.data_ptr<long>(), batch, block_size, cs_block_size, num_edges);
}

// Dispatch on (WARPS from `cfg`) x (BMAX = next pow2 of batch, for register-tight state).
#define SB_ELE_DISPATCH_BMAX(WW)                                                                       \
    switch (bmax) {                                                                                     \
        case 1:  launch_ele<WW, 1 >(element_flows, element_mars, node_flows, node_mars, params, chids, ebase, pbase, b, bs, cs, ne); break; \
        case 2:  launch_ele<WW, 2 >(element_flows, element_mars, node_flows, node_mars, params, chids, ebase, pbase, b, bs, cs, ne); break; \
        case 4:  launch_ele<WW, 4 >(element_flows, element_mars, node_flows, node_mars, params, chids, ebase, pbase, b, bs, cs, ne); break; \
        case 8:  launch_ele<WW, 8 >(element_flows, element_mars, node_flows, node_mars, params, chids, ebase, pbase, b, bs, cs, ne); break; \
        case 16: launch_ele<WW, 16>(element_flows, element_mars, node_flows, node_mars, params, chids, ebase, pbase, b, bs, cs, ne); break; \
        default: TORCH_CHECK(false, "smallbatch_ele_backward_sum: invalid batch ", b);                  \
    }

// cfg id -> WARPS (child-warps per block). 8/16 were the sweep winners (equivalent); 4 trades a
// smaller block for a larger grid to cover more SMs. The autotuner picks per layer/batch.
void smallbatch_ele_backward_sum(torch::Tensor element_flows, torch::Tensor element_mars,
                                 torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor params,
                                 torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase,
                                 int64_t batch, int64_t block_size, int64_t cs_block_size,
                                 int64_t num_edges, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, cs = (int)cs_block_size, ne = (int)num_edges;
    int warps = (cfg == 0) ? 8 : (cfg == 1) ? 16 : (cfg == 2) ? 4 : -1;
    TORCH_CHECK(warps > 0, "smallbatch_ele_backward_sum: invalid cfg ", cfg);
    TORCH_CHECK(b >= 1 && b < 16, "smallbatch_ele_backward_sum: batch must be in [1, 16) (got ", b, ")");
    TORCH_CHECK(cs % warps == 0, "smallbatch_ele_backward_sum: cs_block_size must be a multiple of WARPS");
    int bmax = 1;
    while (bmax < b) bmax <<= 1;                          // next power of two >= batch
    // No silent default: an unknown cfg must not run zero kernels (a no-op launch can masquerade as
    // correct against a reused output buffer -- see the autotune note in sum_layer.py).
    switch (warps) {
        case 8:  SB_ELE_DISPATCH_BMAX(8);  break;
        case 16: SB_ELE_DISPATCH_BMAX(16); break;
        case 4:  SB_ELE_DISPATCH_BMAX(4);  break;
        default: TORCH_CHECK(false, "smallbatch_ele_backward_sum: invalid cfg ", cfg);
    }
}

std::vector<int64_t> smallbatch_ele_configs() { return {8, 16, 4}; }  // WARPS per config id

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallbatch_ele_backward_sum", &smallbatch_ele_backward_sum,
          "small-batch (batch<16) block-sparse sum-layer element-flow backward");
    m.def("smallbatch_ele_configs", &smallbatch_ele_configs, "WARPS per config id");
}
