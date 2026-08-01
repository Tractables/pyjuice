// Small-batch (batch < 16) BACKWARD for the per-block multiplicative gate (`BlockScaleSumParams`).
//
// Forks of `smallbatch_ele_backward.cu` and `smallbatch_par_backward.cu`, the plain-CUDA kernels the
// standard sum layer uses below batch 16 -- no CuTe, no TMA, no CUTLASS. They serve the batches the
// CuTe/TMA forks cannot tile at all (`batch % 64 != 0`), which is the same split the FORWARD already
// makes between `blockscale_forward.cu` and `blockscale_smallbatch_forward.cu`.
//
// Both kernels live in one translation unit because, unlike the CuTe forks, neither carries a
// file-scope TMA descriptor cache or arch-specific PTX helpers to collide over -- so this costs one
// JIT build instead of two.
//
// WHAT THE GATE CHANGES, in each case exactly one term:
//
//   element flows.  The contribution of parent `par` to child `c` gains a factor `phi[b, g(par, c)]`,
//     which in log space is an ADD on the parent's term inside the online log-sum-exp:
//         lse_merge(m, l, log_n_fdm[par, b] + log phi, params[...])
//     `phi` is constant over the `block_size` parents of one node block and over the `gate_cbs`
//     children of one gate, so the edge loop is restructured to walk PARENT BLOCKS on the outside and
//     the gate is loaded once per (parent block, sample) rather than once per edge.
//
//   parameter flows.  Same factor, but it depends on the CONTRACTED index `b`, so it cannot be pulled
//     out of the batch sum. It folds onto `element_mars` instead:
//         p[b] = node_flows + (element_mars + log phi) - node_mars
//     which is where the CuTe fork puts it too, and which leaves the `node_mars == -inf` branch
//     behaving exactly as the ungated kernel does.
//
// `-inf` is the encoding for "no gate here" (a padded edge block, whose parameters are zero anyway):
// `-inf + anything = -inf` drops the term, matching what the ungated kernel computes for it.
//
// SCOPE: inherits the parents' gate (LL, logspace flows, no partial eval or tempering, contiguous
// parents and edge-contiguous per-child parameters, collision-free param flows). `gate_cbs` and the
// node's `ch_block_size` must be powers of two -- the gate index is computed with shifts.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
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

// --------------------------------------------------------------------------------- element flows

template <int WARPS, int BMAX>
__global__ void bs_sb_ele_kernel(float* __restrict__ element_flows, const float* __restrict__ element_mars,
                                 const float* __restrict__ node_flows, const float* __restrict__ node_mars,
                                 const float* __restrict__ params, const float* __restrict__ ext,
                                 const long* __restrict__ chids, const long* __restrict__ ebase,
                                 const long* __restrict__ pbase, const long* __restrict__ gate,
                                 int batch, int block_size, int cs_block_size, int num_edges,
                                 int gate_sh, int gstride, long ext_base) {
    const int lane = threadIdx.x;                       // 0..31  -> a 1/32 stride of the edges
    const int w = threadIdx.y;                          // 0..WARPS-1  -> child within this block's tile
    const int tiles_per_eb = cs_block_size / WARPS;     // child-tiles per node-block
    const int eb = blockIdx.x / tiles_per_eb;           // node-block
    const int tile = blockIdx.x - eb * tiles_per_eb;    // child-tile within the node-block
    const int m_local = tile * WARPS + w;               // child offset within the node-block

    const long eb0 = ebase[eb];                          // first parent node of this node-block
    const long pb0 = pbase[eb] + (long)m_local * block_size;  // first param of this child's row

    // One gate row per (parent block, this child's gate). The child's gate is fixed for the whole
    // warp, so only the parent block varies below.
    const long* gt = gate + (long)eb * gstride;
    const int gidx = m_local >> gate_sh;

    float mm[BMAX], ll[BMAX];                            // per-batch running max / linear-sum
    #pragma unroll
    for (int b = 0; b < BMAX; b++) { mm[b] = -INFINITY; ll[b] = 0.f; }

    // Parent blocks on the outside: `phi` is constant across a block's `block_size` parents, so it is
    // read once per (block, sample) instead of once per edge. Lanes still stride by 32 WITHIN a block,
    // so the params / node_flows / node_mars loads stay coalesced exactly as in the parent kernel.
    const int nblocks = num_edges / block_size;
    for (int j = 0; j < nblocks; j++) {
        const long gb = gt[j];
        float lphi[BMAX];
        #pragma unroll
        for (int b = 0; b < BMAX; b++) {
            if (b >= batch) break;
            lphi[b] = (gb >= 0) ? ext[(gb + ext_base + gidx) * (long)batch + b] : -INFINITY;
        }

        const int e0 = j * block_size, e1 = e0 + block_size;
        for (int e = e0 + lane; e < e1; e += 32) {
            const float wgt = params[pb0 + e];           // epar[m_local, e]; coalesced across lanes
            const float* nf = node_flows + (eb0 + e) * (long)batch;
            const float* nm = node_mars + (eb0 + e) * (long)batch;
            #pragma unroll
            for (int b = 0; b < BMAX; b++) {
                if (b >= batch) break;
                if (lphi[b] == -INFINITY) continue;      // no gate -> this parent contributes nothing
                float nmar = nm[b];
                float term = (nmar == -INFINITY) ? nf[b] : (nf[b] - nmar);   // log_n_fdm[par, b]
                lse_merge(mm[b], ll[b], term + lphi[b], wgt);
            }
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
static void launch_sb_ele(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                          torch::Tensor node_mars, torch::Tensor params, torch::Tensor ext,
                          torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase, torch::Tensor gate,
                          int batch, int block_size, int cs_block_size, int num_edges,
                          int gate_sh, long ext_base) {
    const long neb = chids.size(0);
    dim3 grid((unsigned int)(neb * (cs_block_size / WARPS)));
    dim3 blk(32, WARPS);
    // Launch on the current stream (not the default stream) so the kernel is captured correctly when
    // pyjuice records a CUDA graph of the backward pass.
    bs_sb_ele_kernel<WARPS, BMAX><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), node_flows.data_ptr<float>(),
        node_mars.data_ptr<float>(), params.data_ptr<float>(), ext.data_ptr<float>(),
        chids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(), gate.data_ptr<long>(),
        batch, block_size, cs_block_size, num_edges, gate_sh, (int)gate.size(1), ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define BS_SB_ELE_DISPATCH_BMAX(WW)                                                                    \
    switch (bmax) {                                                                                     \
        case 1:  launch_sb_ele<WW, 1 >(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, b, bs, cs, ne, gsh, eb_); break; \
        case 2:  launch_sb_ele<WW, 2 >(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, b, bs, cs, ne, gsh, eb_); break; \
        case 4:  launch_sb_ele<WW, 4 >(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, b, bs, cs, ne, gsh, eb_); break; \
        case 8:  launch_sb_ele<WW, 8 >(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, b, bs, cs, ne, gsh, eb_); break; \
        case 16: launch_sb_ele<WW, 16>(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, b, bs, cs, ne, gsh, eb_); break; \
        default: TORCH_CHECK(false, "blockscale_sb_ele_backward: invalid batch ", b);                   \
    }

void blockscale_sb_ele_backward(torch::Tensor element_flows, torch::Tensor element_mars,
                                torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor params,
                                torch::Tensor ext, torch::Tensor chids, torch::Tensor ebase,
                                torch::Tensor pbase, torch::Tensor gate,
                                int64_t batch, int64_t block_size, int64_t cs_block_size,
                                int64_t num_edges, int64_t gate_cbs, int64_t ext_base, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, cs = (int)cs_block_size, ne = (int)num_edges;
    long eb_ = (long)ext_base;
    int warps = (cfg == 0) ? 8 : (cfg == 1) ? 16 : (cfg == 2) ? 4 : -1;
    TORCH_CHECK(warps > 0, "blockscale_sb_ele_backward: invalid cfg ", cfg);
    TORCH_CHECK(b >= 1 && b < 16, "blockscale_sb_ele_backward: batch must be in [1, 16) (got ", b, ")");
    TORCH_CHECK(cs % warps == 0, "blockscale_sb_ele_backward: cs_block_size must be a multiple of WARPS");
    TORCH_CHECK((gate_cbs & (gate_cbs - 1)) == 0 && gate_cbs > 0,
                "blockscale_sb_ele_backward: the gate's ch_block_size must be a power of two");
    TORCH_CHECK(cs % gate_cbs == 0,
                "blockscale_sb_ele_backward: the gate must divide the child block (", cs, ")");
    // The gate is per (parent BLOCK, child gate), so the parent list must be a whole number of blocks.
    TORCH_CHECK(ne % bs == 0,
                "blockscale_sb_ele_backward: num_edges (", ne, ") must be a whole number of parent "
                "blocks of size ", bs);
    TORCH_CHECK(gate.size(1) >= ne / bs,
                "blockscale_sb_ele_backward: the gate table has ", gate.size(1), " columns but the "
                "child block has ", ne / bs, " parent blocks");

    int gsh = 0;
    while ((1 << gsh) < (int)gate_cbs) ++gsh;
    int bmax = 1;
    while (bmax < b) bmax <<= 1;                          // next power of two >= batch
    // No silent default: an unknown cfg must not run zero kernels (a no-op launch can masquerade as
    // correct against a reused output buffer).
    switch (warps) {
        case 8:  BS_SB_ELE_DISPATCH_BMAX(8);  break;
        case 16: BS_SB_ELE_DISPATCH_BMAX(16); break;
        case 4:  BS_SB_ELE_DISPATCH_BMAX(4);  break;
        default: TORCH_CHECK(false, "blockscale_sb_ele_backward: invalid cfg ", cfg);
    }
}

std::vector<int64_t> blockscale_sb_ele_configs() { return {8, 16, 4}; }  // WARPS per config id

// --------------------------------------------------------------------------------- parameter flows

template <int EY, int BMAX>
__global__ void bs_sb_par_kernel(float* __restrict__ param_flows, const float* __restrict__ node_flows,
                                 const float* __restrict__ node_mars, const float* __restrict__ element_mars,
                                 const float* __restrict__ params, const float* __restrict__ ext,
                                 const long* __restrict__ nids, const long* __restrict__ cids,
                                 const long* __restrict__ pids, const long* __restrict__ pfids,
                                 const long* __restrict__ gate, int batch, int block_size, int num_edges,
                                 int node_cbs, int node_sh, int gate_sh, int gstride, long ext_base) {
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

    // This edge's gate: child block `e / node_cbs` of node block `nb`, gate `d` within that block.
    // Identical indexing to the CuTe param-flow fork, which reuses the FORWARD's own table.
    const long gb = gate[(long)nb * gstride + (e >> node_sh)];
    const int d = (e & (node_cbs - 1)) >> gate_sh;

    const long cb = (long)nb * num_edges;
    const long child = cids[cb + e];
    float mx = -INFINITY, pl[BMAX];
    #pragma unroll
    for (int b = 0; b < BMAX; b++) {
        if (b >= batch) break;
        // Folded onto `element_mars`, so the `node_mars == -inf` branch below is untouched -- exactly
        // where the CuTe fork puts it.
        float emb = element_mars[child * (long)batch + b]
                    + ((gb >= 0) ? ext[(gb + ext_base + d) * (long)batch + b] : -INFINITY);
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
static void launch_sb_par(torch::Tensor param_flows, torch::Tensor node_flows, torch::Tensor node_mars,
                          torch::Tensor element_mars, torch::Tensor params, torch::Tensor ext,
                          torch::Tensor nids, torch::Tensor cids, torch::Tensor pids, torch::Tensor pfids,
                          torch::Tensor gate, int batch, int block_size, int num_edges,
                          int node_cbs, int node_sh, int gate_sh, long ext_base) {
    const long nb = nids.size(0);
    const long groups = block_size / 32;
    dim3 grid((unsigned int)(nb * groups), (unsigned int)((num_edges + EY - 1) / EY));
    dim3 blk(32, EY);
    // Launch on the current stream so the kernel is captured correctly under CUDA-graph recording.
    bs_sb_par_kernel<EY, BMAX><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        param_flows.data_ptr<float>(), node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
        element_mars.data_ptr<float>(), params.data_ptr<float>(), ext.data_ptr<float>(),
        nids.data_ptr<long>(), cids.data_ptr<long>(), pids.data_ptr<long>(), pfids.data_ptr<long>(),
        gate.data_ptr<long>(), batch, block_size, num_edges, node_cbs, node_sh, gate_sh,
        (int)gate.size(1), ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define BS_SB_PAR_DISPATCH_BMAX(EE)                                                                    \
    switch (bmax) {                                                                                     \
        case 1:  launch_sb_par<EE, 1 >(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 2:  launch_sb_par<EE, 2 >(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 4:  launch_sb_par<EE, 4 >(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 8:  launch_sb_par<EE, 8 >(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 16: launch_sb_par<EE, 16>(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        default: TORCH_CHECK(false, "blockscale_sb_par_backward: invalid batch ", b);                   \
    }

void blockscale_sb_par_backward(torch::Tensor param_flows, torch::Tensor node_flows, torch::Tensor node_mars,
                                torch::Tensor element_mars, torch::Tensor params, torch::Tensor ext,
                                torch::Tensor nids, torch::Tensor cids, torch::Tensor pids,
                                torch::Tensor pfids, torch::Tensor gate,
                                int64_t batch, int64_t block_size, int64_t num_edges, int64_t node_cbs,
                                int64_t gate_cbs, int64_t ext_base, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, ne = (int)num_edges, ncb = (int)node_cbs;
    long eb_ = (long)ext_base;
    int ey = (cfg == 0) ? 8 : (cfg == 1) ? 16 : (cfg == 2) ? 4 : -1;
    TORCH_CHECK(ey > 0, "blockscale_sb_par_backward: invalid cfg ", cfg);
    TORCH_CHECK(b >= 1 && b < 16, "blockscale_sb_par_backward: batch must be in [1, 16) (got ", b, ")");
    TORCH_CHECK(bs % 32 == 0, "blockscale_sb_par_backward: block_size must be a multiple of 32 (got ", bs, ")");
    TORCH_CHECK((ncb & (ncb - 1)) == 0 && (gate_cbs & (gate_cbs - 1)) == 0 && gate_cbs > 0,
                "blockscale_sb_par_backward: both child block sizes must be powers of two");
    TORCH_CHECK(ncb % gate_cbs == 0,
                "blockscale_sb_par_backward: the gate must divide the child block (", ncb, ")");
    TORCH_CHECK(ne % ncb == 0,
                "blockscale_sb_par_backward: num_edges (", ne, ") must be a whole number of child "
                "blocks of size ", ncb);
    TORCH_CHECK(gate.size(1) >= ne / ncb,
                "blockscale_sb_par_backward: the gate table has ", gate.size(1), " columns but the "
                "row has ", ne / ncb, " edge blocks");

    int nsh = 0, gsh = 0;
    while ((1 << nsh) < ncb) ++nsh;
    while ((1 << gsh) < (int)gate_cbs) ++gsh;
    int bmax = 1;
    while (bmax < b) bmax <<= 1;                       // next power of two >= batch
    switch (ey) {
        case 8:  BS_SB_PAR_DISPATCH_BMAX(8);  break;
        case 16: BS_SB_PAR_DISPATCH_BMAX(16); break;
        case 4:  BS_SB_PAR_DISPATCH_BMAX(4);  break;
        default: TORCH_CHECK(false, "blockscale_sb_par_backward: invalid cfg ", cfg);
    }
}

std::vector<int64_t> blockscale_sb_par_configs() { return {8, 16, 4}; }  // EY per config id

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_sb_ele_backward", &blockscale_sb_ele_backward,
          "small-batch element-flow backward for the per-block multiplicative gate");
    m.def("blockscale_sb_ele_configs", &blockscale_sb_ele_configs, "WARPS per config id");
    m.def("blockscale_sb_par_backward", &blockscale_sb_par_backward,
          "small-batch param-flow backward for the per-block multiplicative gate");
    m.def("blockscale_sb_par_configs", &blockscale_sb_par_configs, "EY per config id");
}
