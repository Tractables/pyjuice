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
#include "../../../../layer/kernels/c/param_load.cuh"   // `ParamLoad<TP>`: the parameters in their stored dtype
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

template <int WARPS, int BMAX, bool WGRAD, typename TP>
__global__ void bs_sb_ele_kernel(float* __restrict__ element_flows, const float* __restrict__ element_mars,
                                 const float* __restrict__ node_flows, const float* __restrict__ node_mars,
                                 const TP* __restrict__ params, const float* __restrict__ ext,
                                 const long* __restrict__ chids, const long* __restrict__ ebase,
                                 const long* __restrict__ pbase, const long* __restrict__ gate,
                                 float* __restrict__ grad_ext, float* __restrict__ gpart,
                                 int batch, int block_size, int cs_block_size, int tk, int ktiles,
                                 int gate_sh, int gstride, long ext_base) {
    const int lane = threadIdx.x;                       // 0..31  -> a 1/32 stride of the edges
    const int w = threadIdx.y;                          // 0..WARPS-1  -> child within this block's tile
    const int tiles_per_eb = cs_block_size / WARPS;     // child-tiles per node-block
    const int eb = blockIdx.x / tiles_per_eb;           // node-block
    const int tile = blockIdx.x - eb * tiles_per_eb;    // child-tile within the node-block
    const int m_local = tile * WARPS + w;               // child offset within the node-block

    // The parents come in `ktiles` tiles of `tk`; each tile's first parent node and first parameter are a table
    // entry (`[node-block, tile]`, the layer's compiled walk): a child's parameters are contiguous WITHIN a tile
    // only -- across parent blocks they jump. (An earlier version took one base per node-block and was gated on
    // the tiles chaining, which no layer of more than one parent block satisfies.)
    const long* ebt = ebase + (long)eb * ktiles;
    const long* pbt = pbase + (long)eb * ktiles;

    // One gate row per (parent block, this child's gate). The child's gate is fixed for the whole
    // warp, so only the parent block varies below.
    const long* gt = gate + (long)eb * gstride;
    const int gidx = m_local >> gate_sh;

    float mm[BMAX], ll[BMAX];                            // per-batch running max / linear-sum
    #pragma unroll
    for (int b = 0; b < BMAX; b++) { mm[b] = -INFINITY; ll[b] = 0.f; }

    // `d LL / d log phi`, first term. Needs this child's own value, which the epilogue reads anyway.
    const long node_g = chids[eb] + m_local;
    float emb[BMAX];
    if (WGRAD) {
        #pragma unroll
        for (int b = 0; b < BMAX; b++) {
            if (b >= batch) break;
            emb[b] = element_mars[node_g * (long)batch + b];
        }
    }

    // The gate is per PARENT BLOCK (`gt[j]`): `phi` is constant across a block's `block_size` parents, so it is read
    // once per (block, sample) instead of once per edge, and the gradient's per-block sum is flushed when the block
    // ends. The parents are walked tile by tile; a tile is cut at parent-block boundaries (a block of several tiles,
    // or a tile of several blocks, both occur), and lanes stride by 32 WITHIN a segment, so the params / node_flows /
    // node_mars loads stay coalesced exactly as in the parent kernel.
    int cur_j = -1;
    long gb = -1;
    float lphi[BMAX];
    #pragma unroll
    for (int b = 0; b < BMAX; b++) lphi[b] = -INFINITY;
    // Per-PARENT-BLOCK accumulator, kept ALONGSIDE the running one rather than replacing it: the gradient needs the
    // block's contribution warp-reduced on its own, while `mm`/`ll` must stay per-lane so the epilogue's reduction
    // below is exactly what it was.
    float mj[BMAX], lj[BMAX];
    #pragma unroll
    for (int b = 0; b < BMAX; b++) { mj[b] = -INFINITY; lj[b] = 0.f; }

    // sum over {parents in the block} x {children in this gate} of the edge flows into the gate's row. Children of
    // one gate live in DIFFERENT warps here (one warp per child), so the sum across parents is the warp reduction and
    // the sum across a gate's children is either atomic (`gpart` null: run-to-run rounding differences) or, with
    // `gpart`, DETERMINISTIC: each child stores its share of each parent block, [element block, child, parent block,
    // sample], and `grad_reduce_kernel` sums a gate's children in order (the gradient the same from run to run, as the
    // other gated kernels keep it).
    const int nblk = tk * ktiles / block_size;                  // parent blocks (the host checks num_edges % block_size)
    auto flush_block = [&]() {
        if (WGRAD && cur_j >= 0) {
            #pragma unroll
            for (int b = 0; b < BMAX; b++) {
                if (b >= batch) break;
                float m = mj[b], l = lj[b];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    lse_merge(m, l, __shfl_xor_sync(0xffffffff, m, off),
                                    __shfl_xor_sync(0xffffffff, l, off));
                if (lane == 0) {
                    const float v = (gb >= 0 && m != -INFINITY) ? l * __expf(m + emb[b]) : 0.f;
                    if (gpart != nullptr)
                        gpart[(((long)eb * cs_block_size + m_local) * nblk + cur_j) * batch + b] = v;
                    else if (v != 0.f)
                        atomicAdd(grad_ext + (gb + ext_base + gidx) * (long)batch + b, v);
                }
                mj[b] = -INFINITY; lj[b] = 0.f;
            }
        }
    };

    for (int t = 0; t < ktiles; t++) {
        const long eb0 = ebt[t];                                  // the tile's first parent node
        const long pb0 = pbt[t] + (long)m_local * block_size;    // the first param of this child's row in the tile
        int s = 0;
        while (s < tk) {
            const int pidx = t * tk + s;                          // the parent's index in the node-block's edge list
            const int j = pidx / block_size;                      // its parent block
            const int seg = min(tk - s, block_size - (pidx - j * block_size));
            if (j != cur_j) {
                flush_block();
                cur_j = j;
                gb = gt[j];
                #pragma unroll
                for (int b = 0; b < BMAX; b++) {
                    if (b >= batch) break;
                    lphi[b] = (gb >= 0) ? ext[(gb + ext_base + gidx) * (long)batch + b] : -INFINITY;
                }
            }
            for (int e = s + lane; e < s + seg; e += 32) {
                const float wgt = ParamLoad<TP>::at(&params[pb0 + e]);   // epar[m_local, e]; coalesced across lanes
                const float* nf = node_flows + (eb0 + e) * (long)batch;
                const float* nm = node_mars + (eb0 + e) * (long)batch;
                #pragma unroll
                for (int b = 0; b < BMAX; b++) {
                    if (b >= batch) break;
                    if (lphi[b] == -INFINITY) continue;      // no gate -> this parent contributes nothing
                    float nmar = nm[b];
                    float term = (nmar == -INFINITY) ? nf[b] : (nf[b] - nmar);   // log_n_fdm[par, b]
                    lse_merge(mm[b], ll[b], term + lphi[b], wgt);
                    if (WGRAD) lse_merge(mj[b], lj[b], term + lphi[b], wgt);
                }
            }
            s += seg;
        }
    }
    flush_block();

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

// The gate gradient from the children's per-parent-block shares (`gpart`): a thread per (element block, parent block,
// gate, sample) sums its gate's children IN ORDER and adds the sum into the gate's row (an atomic add, but one writer a
// row unless two edge blocks share a gate's storage -- tied gates -- which every gated kernel adds atomically).
__global__ void grad_reduce_kernel(float* __restrict__ grad_ext, const float* __restrict__ gpart,
                                   const long* __restrict__ gate, int neb, int cs, int nblk, int gate_cbs,
                                   int batch, long ext_base) {
    const int ngates = cs / gate_cbs;
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)neb * nblk * ngates * batch) return;
    const int b = (int)(i % batch);
    long r = i / batch;
    const int g = (int)(r % ngates); r /= ngates;
    const int j = (int)(r % nblk);
    const int eb = (int)(r / nblk);
    const long gb = gate[(long)eb * nblk + j];
    if (gb < 0) return;
    float s = 0.f;
    const float* p = gpart + (((long)eb * cs + (long)g * gate_cbs) * nblk + j) * batch + b;
#pragma unroll 8
    for (int m = 0; m < gate_cbs; m++) s += p[(long)m * nblk * batch];   // (in order; unrolled so the loads overlap)
    if (s != 0.f) atomicAdd(grad_ext + (gb + ext_base + g) * (long)batch + b, s);
}

template <int WARPS, int BMAX, typename TP>
static void launch_sb_ele(torch::Tensor element_flows, torch::Tensor element_mars, torch::Tensor node_flows,
                          torch::Tensor node_mars, torch::Tensor params, torch::Tensor ext,
                          torch::Tensor chids, torch::Tensor ebase, torch::Tensor pbase, torch::Tensor gate,
                          torch::Tensor grad_ext, torch::Tensor gpart,
                          int batch, int block_size, int cs_block_size, int tk, int ktiles,
                          int gate_sh, long ext_base) {
    const long neb = chids.size(0);
    dim3 grid((unsigned int)(neb * (cs_block_size / WARPS)));
    dim3 blk(32, WARPS);
    // Launch on the current stream (not the default stream) so the kernel is captured correctly when
    // pyjuice records a CUDA graph of the backward pass.
    // WGRAD is a TEMPLATE parameter, so the no-gradient path compiles to exactly the kernel it was
    // before: no extra registers, no occupancy change.
    if (grad_ext.numel() > 0)
        bs_sb_ele_kernel<WARPS, BMAX, true, TP><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
            element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), param_ptr<TP>(params), ext.data_ptr<float>(),
            chids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(), gate.data_ptr<long>(),
            grad_ext.data_ptr<float>(), gpart.numel() ? gpart.data_ptr<float>() : nullptr,
            batch, block_size, cs_block_size, tk, ktiles, gate_sh, (int)gate.size(1), ext_base);
    else
        bs_sb_ele_kernel<WARPS, BMAX, false, TP><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
            element_flows.data_ptr<float>(), element_mars.data_ptr<float>(), node_flows.data_ptr<float>(),
            node_mars.data_ptr<float>(), param_ptr<TP>(params), ext.data_ptr<float>(),
            chids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(), gate.data_ptr<long>(),
            nullptr, nullptr,
            batch, block_size, cs_block_size, tk, ktiles, gate_sh, (int)gate.size(1), ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (grad_ext.numel() > 0 && gpart.numel() > 0) {
        const int nblk = tk * ktiles / block_size, gcb = 1 << gate_sh;
        TORCH_CHECK(gpart.numel() >= neb * cs_block_size * (long)nblk * batch, "blockscale_sb_ele_backward: gpart too small");
        TORCH_CHECK(gate.size(1) >= nblk, "blockscale_sb_ele_backward: the gate table's columns");
        const long n = neb * (long)nblk * (cs_block_size / gcb) * batch;
        grad_reduce_kernel<<<(unsigned)((n + 255) / 256), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            grad_ext.data_ptr<float>(), gpart.data_ptr<float>(), gate.data_ptr<long>(), (int)neb, cs_block_size, nblk,
            gcb, batch, ext_base);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

#define BS_SB_ELE_DISPATCH_BMAX(WW)                                                                    \
    switch (bmax) {                                                                                     \
        case 1:  launch_sb_ele<WW, 1 , TP>(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, grad_ext, gpart, b, bs, cs, tk, kt, gsh, eb_); break; \
        case 2:  launch_sb_ele<WW, 2 , TP>(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, grad_ext, gpart, b, bs, cs, tk, kt, gsh, eb_); break; \
        case 4:  launch_sb_ele<WW, 4 , TP>(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, grad_ext, gpart, b, bs, cs, tk, kt, gsh, eb_); break; \
        case 8:  launch_sb_ele<WW, 8 , TP>(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, grad_ext, gpart, b, bs, cs, tk, kt, gsh, eb_); break; \
        case 16: launch_sb_ele<WW, 16, TP>(element_flows, element_mars, node_flows, node_mars, params, ext, chids, ebase, pbase, gate, grad_ext, gpart, b, bs, cs, tk, kt, gsh, eb_); break; \
        default: TORCH_CHECK(false, "blockscale_sb_ele_backward: invalid batch ", b);                   \
    }

void blockscale_sb_ele_backward(torch::Tensor element_flows, torch::Tensor element_mars,
                                torch::Tensor node_flows, torch::Tensor node_mars, torch::Tensor params,
                                torch::Tensor ext, torch::Tensor chids, torch::Tensor ebase,
                                torch::Tensor pbase, torch::Tensor gate, torch::Tensor grad_ext, torch::Tensor gpart,
                                int64_t batch, int64_t block_size, int64_t cs_block_size,
                                int64_t num_edges, int64_t tile_k,
                                // `cfg` stays LAST: the Python autotuner substitutes the last argument when trying configs.
                                int64_t gate_cbs, int64_t ext_base, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, cs = (int)cs_block_size, ne = (int)num_edges, tk = (int)tile_k;
    long eb_ = (long)ext_base;
    int warps = (cfg == 0) ? 8 : (cfg == 1) ? 16 : (cfg == 2) ? 4 : -1;
    TORCH_CHECK(warps > 0, "blockscale_sb_ele_backward: invalid cfg ", cfg);
    TORCH_CHECK(tk > 0 && tk % 32 == 0 && ne % tk == 0,
                "blockscale_sb_ele_backward: num_edges (", ne, ") must be whole tiles of ", tk, " (a multiple of 32)");
    const int kt = ne / tk;
    TORCH_CHECK(ebase.dim() == 2 && ebase.size(0) == chids.size(0) && ebase.size(1) == kt && ebase.is_contiguous()
                && pbase.sizes() == ebase.sizes() && pbase.is_contiguous(),
                "blockscale_sb_ele_backward: ebase / pbase must be contiguous [node blocks, ", kt, "] tables");
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
    auto run = [&](auto tag) {
        using TP = decltype(tag);
        switch (warps) {
            case 8:  BS_SB_ELE_DISPATCH_BMAX(8);  break;
            case 16: BS_SB_ELE_DISPATCH_BMAX(16); break;
            case 4:  BS_SB_ELE_DISPATCH_BMAX(4);  break;
            default: TORCH_CHECK(false, "blockscale_sb_ele_backward: invalid cfg ", cfg);
        }
    };
    if (params.scalar_type() == torch::kBFloat16) run(__nv_bfloat16{}); else run(float{});
}

std::vector<int64_t> blockscale_sb_ele_configs() { return {8, 16, 4}; }  // WARPS per config id

// --------------------------------------------------------------------------------- parameter flows

// `PADDED` / `PF_ATOMIC` -- the same two hazards `_bs_triton_par_kernel` carries, for the same
// reason. This kernel is now reached even when `pfids` are NOT collision-free, which is what the
// layer's dispatch used to establish on its behalf:
//
//   * `PADDED` -- a padded compiled slot has `cids == 0` / `pids == 0` / `pfids == 0`, and
//     `param_flows` has no dummy prefix, so pfid 0 belongs to a REAL edge block. The padded lane's
//     contribution is already exactly zero, but its read-add-store of `+0.0` racing a real one
//     discards the real update. Masking it out also restores collision-freedom over the slots that
//     ARE written, which is what lets a ragged untied layer keep the non-atomic path.
//   * `PF_ATOMIC` -- the written slots genuinely collide. Parameter tying does this: a tie group
//     shares one `_param_flow_range`, so its members compile to identical `pfids`. A homogeneous HMM
//     ties every transition into one slot, which is an ordinary model, not an edge case.
//
// Both default off, so a dense untied layer generates exactly the code it did before.
template <int EY, int BMAX, bool PADDED, bool PF_ATOMIC, typename TP>
__global__ void bs_sb_par_kernel(float* __restrict__ param_flows, const float* __restrict__ node_flows,
                                 const float* __restrict__ node_mars, const float* __restrict__ element_mars,
                                 const TP* __restrict__ params, const float* __restrict__ ext,
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
    // `cids != 0` is the padding predicate -- exact by construction, since a real child is an element
    // `>= num_dummy_eles > 0`, and unlike the gate it does not depend on the table being indexed right.
    if (PADDED && child == 0) return;

    float val = ParamLoad<TP>::at(&params[pids[cb + e] + tile_id]) * term;   // coalesced across lanes (node-contiguous)
    if (PF_ATOMIC) {
        atomicAdd(&param_flows[pfids[cb + e] + tile_id], val);
    } else {
        param_flows[pfids[cb + e] + tile_id] += val;            // collision-free RMW, coalesced
    }
}

template <int EY, int BMAX, bool PADDED, bool PF_ATOMIC, typename TP>
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
    bs_sb_par_kernel<EY, BMAX, PADDED, PF_ATOMIC, TP><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        param_flows.data_ptr<float>(), node_flows.data_ptr<float>(), node_mars.data_ptr<float>(),
        element_mars.data_ptr<float>(), param_ptr<TP>(params), ext.data_ptr<float>(),
        nids.data_ptr<long>(), cids.data_ptr<long>(), pids.data_ptr<long>(), pfids.data_ptr<long>(),
        gate.data_ptr<long>(), batch, block_size, num_edges, node_cbs, node_sh, gate_sh,
        (int)gate.size(1), ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define BS_SB_PAR_DISPATCH_BMAX(EE, PD, PA)                                                                    \
    switch (bmax) {                                                                                     \
        case 1:  launch_sb_par<EE, 1 , PD, PA, TP>(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 2:  launch_sb_par<EE, 2 , PD, PA, TP>(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 4:  launch_sb_par<EE, 4 , PD, PA, TP>(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 8:  launch_sb_par<EE, 8 , PD, PA, TP>(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        case 16: launch_sb_par<EE, 16, PD, PA, TP>(param_flows, node_flows, node_mars, element_mars, params, ext, nids, cids, pids, pfids, gate, b, bs, ne, ncb, nsh, gsh, eb_); break; \
        default: TORCH_CHECK(false, "blockscale_sb_par_backward: invalid batch ", b);                   \
    }

void blockscale_sb_par_backward(torch::Tensor param_flows, torch::Tensor node_flows, torch::Tensor node_mars,
                                torch::Tensor element_mars, torch::Tensor params, torch::Tensor ext,
                                torch::Tensor nids, torch::Tensor cids, torch::Tensor pids,
                                torch::Tensor pfids, torch::Tensor gate,
                                int64_t batch, int64_t block_size, int64_t num_edges, int64_t node_cbs,
                                // `padded` / `pf_atomic` precede `cfg`: the Python autotuner substitutes the LAST
                                // argument when trying configs, so `cfg` must stay last.
                                int64_t gate_cbs, int64_t ext_base,
                                int64_t padded, int64_t pf_atomic, int64_t cfg) {
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
    // `ey` picks the edge-tiling config; `padded` / `pf_atomic` pick the write specialization (see
    // the note at the kernel). Both are compile-time, so a dense untied layer -- the overwhelmingly
    // common case -- generates exactly the code it did before either flag existed.
#define BS_SB_PAR_BY_EY(PD, PA)                                     \
    switch (ey) {                                                   \
        case 8:  BS_SB_PAR_DISPATCH_BMAX(8,  PD, PA); break;        \
        case 16: BS_SB_PAR_DISPATCH_BMAX(16, PD, PA); break;        \
        case 4:  BS_SB_PAR_DISPATCH_BMAX(4,  PD, PA); break;        \
        default: TORCH_CHECK(false, "blockscale_sb_par_backward: invalid cfg ", cfg); \
    }

    auto run = [&](auto tag) {
        using TP = decltype(tag);
        if (padded) {
            if (pf_atomic) { BS_SB_PAR_BY_EY(true,  true ); }
            else           { BS_SB_PAR_BY_EY(true,  false); }
        } else {
            if (pf_atomic) { BS_SB_PAR_BY_EY(false, true ); }
            else           { BS_SB_PAR_BY_EY(false, false); }
        }
    };
    if (params.scalar_type() == torch::kBFloat16) run(__nv_bfloat16{}); else run(float{});
#undef BS_SB_PAR_BY_EY
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
