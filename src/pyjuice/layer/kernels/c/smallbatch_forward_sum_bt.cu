// Small-batch (batch < 64) block-sparse sum-layer FORWARD, the parameters loaded ONCE for a tile of batch columns.
//
// `smallbatch_forward_sum.cu` runs one block per (node group, SAMPLE): its parameter stream is re-read once per
// sample, and past a sample or two that stream -- not the arithmetic -- is the kernel (MEASURED, HMM-4096, 16.8 M
// parameters a layer, L2-resident: batch 1 21 us, batch 4 60 us, batch 16 205 us; the L2 -> SM ingest wall). This
// kernel keeps that kernel's structure -- a warp = 32 consecutive nodes (lane = node offset, coalesced 128 B parameter
// loads), blockDim.y = SPLIT edge-warps splitting the edge reduction -- but every lane carries BT batch columns, so
// one parameter load serves BT columns and grid.y = ceil(batch / BT).
//
// TWO PASSES PER EDGE-WARP, like the Triton forward's tiles: the maximum of each column over the warp's edges first,
// then `sum_e p_e * exp(x_e - M)` -- one exponential per (edge, column) and no rescaling of the running sum, so the
// edges are independent and the loads of the next U edges are in flight while the current ones compute. (An online
// recurrence, bit-identical to the kernel above, was measured first: 53 us at batch 16 against 20 at batch 1 on the
// layer above -- its exponential chain is serial and the next loads wait behind it.)
//
// THE EXPONENTIALS ARE WARP-UNIFORM. `element_mars[(eb + e) * batch + b]` does not depend on the node, so a column's
// maximum and its per-edge terms are the same in every lane of the warp. Lane j < BT owns column j: it finds the
// maximum and computes the term of each edge once, staged in shared memory; every lane then does BT multiply-adds
// from the staged values. Without this each lane would compute BT exponentials per edge, 32x redundantly across the
// warp, and the MUFU pipe would be the kernel from batch ~8 on.
//
// Columns past the batch (the tile's tail) see `-inf` element values -- their terms are 0 -- and are never stored.
// Numerically equivalent to the kernel above and to the Triton forward (fp32 log-sum-exp with a different
// association), the ~1e-6 the autotuner's arms already differ by.
//
// Math, per (node m, batch b), for node-block nb (children contiguous, parameters block_size-strided, as the caller
// verifies for the kernel above):
//   node_mars[node*batch + b] = log( sum_e params[pbase[nb] + e*block_size + m_local] * exp(element_mars[(ebase[nb] + e)*batch + b]) )

#include <torch/extension.h>
#include "param_load.cuh"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <vector>

// Edges in flight per edge-warp: the parameter and element loads of the NEXT U edges are issued before the current U
// are used, which keeps the parameter stream going while a chunk's exponentials and multiply-adds run.
constexpr int SB_BT_UNROLL = 8;

template <int SPLIT, int BT, typename TP>
__global__ void __launch_bounds__(32 * SPLIT)
sb_fwd_bt_kernel(float* __restrict__ node_mars, const float* __restrict__ element_mars,
                 const TP* __restrict__ params, const long* __restrict__ nids,
                 const long* __restrict__ ebase, const long* __restrict__ pbase,
                 int batch, int block_size, int num_edges) {
    constexpr int U = SB_BT_UNROLL;
    static_assert(BT % 4 == 0 && BT <= 32, "BT: a multiple of 4 up to one per lane");
    static_assert(SPLIT * BT <= 256, "the combine stages SPLIT x BT x 32 partial sums in shared memory");

    const int groups = block_size >> 5;          // block_size / 32  (block_size is a multiple of 32)
    const int gx = blockIdx.x;
    const int nb = gx / groups;                   // node-block
    const int grp = gx - nb * groups;             // node-group within the block
    const int b0 = blockIdx.y * BT;               // first batch column of this tile
    const int lane = threadIdx.x;                 // 0..31  -> node offset within the group
    const int ty = threadIdx.y;                   // 0..SPLIT-1  -> edge-warp
    const int m_local = grp * 32 + lane;

    const long eb = ebase[nb], pb = pbase[nb];
    const bool owner = lane < BT;                                   // lane j owns column j
    const bool live = owner && (b0 + lane < batch);                 // ... and the column is a real one
    const TP* pcol = params + pb + m_local;                         // this lane's node: edge e at pcol[e * block_size]
    const float* ecol = element_mars + eb * (long)batch + b0 + lane; // this lane's column: edge e at ecol[e * batch]

    // The staged terms, [edge-warp][edge in flight][column]; reused for the combine below, which needs SPLIT x BT
    // maxima plus SPLIT x BT x 32 sums.
    constexpr int STAGE = SPLIT * U * BT, COMBINE = SPLIT * BT * 33;
    __shared__ __align__(16) float smem[STAGE > COMBINE ? STAGE : COMBINE];
    float* sE = smem + ty * (U * BT);

    // Pass 1: the column maxima over this edge-warp's edges. The lanes split the EDGES here (lane = edge offset), each
    // loading its edge's BT columns as float4s -- 32 edges' worth of bytes in flight per load instruction instead of
    // one edge's -- and the per-column maxima are reduced across the warp with shuffles. Lane j then holds column j's.
    float mx[BT];
    #pragma unroll
    for (int j = 0; j < BT; j++) mx[j] = -INFINITY;
    {
        const int ncols = min(BT, batch - b0);                      // live columns of this tile
        const float* erow = element_mars + eb * (long)batch + b0;   // edge e's columns at erow[e * batch + j]
        for (int e = ty + lane * SPLIT; e < num_edges; e += SPLIT * 32) {   // edge-warp ty's edges, lane-strided
            const float* r = erow + (long)e * batch;
            if (ncols == BT && (((size_t)r) & 15) == 0) {
                #pragma unroll
                for (int j4 = 0; j4 < BT / 4; j4++) {
                    const float4 v = reinterpret_cast<const float4*>(r)[j4];
                    mx[j4 * 4 + 0] = fmaxf(mx[j4 * 4 + 0], v.x); mx[j4 * 4 + 1] = fmaxf(mx[j4 * 4 + 1], v.y);
                    mx[j4 * 4 + 2] = fmaxf(mx[j4 * 4 + 2], v.z); mx[j4 * 4 + 3] = fmaxf(mx[j4 * 4 + 3], v.w);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < BT; j++) if (j < ncols) mx[j] = fmaxf(mx[j], r[j]);
            }
        }
    }
    float M = -INFINITY;                                            // lane j: column j's maximum (warp-uniform value)
    #pragma unroll
    for (int j = 0; j < BT; j++) {
        float v = mx[j];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        if (lane == j) M = v;
    }
    if (!live) M = -INFINITY;
    // An all-`-inf` column (or a tail column) keeps M = -inf for the combine; its terms exponentiate against 0 and
    // are exactly 0, never -inf - (-inf).
    const float Mf = (M == -INFINITY) ? 0.f : M;

    // Pass 2: the terms, the next chunk's loads in flight while the current chunk computes
    float l[BT];
    #pragma unroll
    for (int j = 0; j < BT; j++) l[j] = 0.f;

    float pe[U], emv[U];
    auto fetch = [&](int e, float* p_out, float* em_out) {
        #pragma unroll
        for (int u = 0; u < U; u++) {
            const int ee = e + u * SPLIT;
            const bool ok = ee < num_edges;
            p_out[u] = ok ? ParamLoad<TP>::at(pcol + (long)ee * block_size) : 0.f;
            em_out[u] = (ok && live) ? ecol[(long)ee * batch] : -INFINITY;
        }
    };
    fetch(ty, pe, emv);
    for (int e = ty; e < num_edges; e += SPLIT * U) {
        float pn[U], emn[U];
        const int en = e + SPLIT * U;
        if (en < num_edges) fetch(en, pn, emn);
        // Lane j: its column's terms of these edges, staged
        if (owner) {
            #pragma unroll
            for (int u = 0; u < U; u++) sE[u * BT + lane] = __expf(emv[u] - Mf);
        }
        __syncwarp();
        // Every lane: BT multiply-adds per edge from the staged terms (broadcast reads)
        #pragma unroll
        for (int u = 0; u < U; u++) {
            const float4* t4 = reinterpret_cast<const float4*>(sE + u * BT);
            #pragma unroll
            for (int j4 = 0; j4 < BT / 4; j4++) {
                const float4 t = t4[j4];
                l[j4 * 4 + 0] = fmaf(pe[u], t.x, l[j4 * 4 + 0]);
                l[j4 * 4 + 1] = fmaf(pe[u], t.y, l[j4 * 4 + 1]);
                l[j4 * 4 + 2] = fmaf(pe[u], t.z, l[j4 * 4 + 2]);
                l[j4 * 4 + 3] = fmaf(pe[u], t.w, l[j4 * 4 + 3]);
            }
        }
        __syncwarp();                                               // before the next edges overwrite the stage
        if (en < num_edges) {
            #pragma unroll
            for (int u = 0; u < U; u++) { pe[u] = pn[u]; emv[u] = emn[u]; }
        }
    }

    // Combine the SPLIT partials per column: L = sum_k L_k * exp(M_k - max_k M_k)
    __syncthreads();                                                // the stage is dead in every warp
    float* sM = smem;                                               // [SPLIT][BT]: a partial's (uniform) maximum
    float* sL = smem + SPLIT * BT;                                  // [SPLIT][BT][32]: its sums per node
    if (owner) sM[ty * BT + lane] = M;
    #pragma unroll
    for (int j = 0; j < BT; j++) sL[(ty * BT + j) * 32 + lane] = l[j];
    __syncthreads();

    for (int j = ty; j < BT; j += SPLIT) {                          // edge-warp ty finishes columns ty, ty + SPLIT, ...
        const int b = b0 + j;
        if (b >= batch) continue;
        float gm = -INFINITY;
        #pragma unroll
        for (int k = 0; k < SPLIT; k++) gm = fmaxf(gm, sM[k * BT + j]);
        float L = 0.f;
        if (gm != -INFINITY) {
            #pragma unroll
            for (int k = 0; k < SPLIT; k++) {
                const float mk = sM[k * BT + j];
                L = fmaf(sL[(k * BT + j) * 32 + lane], (mk == -INFINITY) ? 0.f : __expf(mk - gm), L);
            }
        }
        const long node = nids[nb] + m_local;
        node_mars[node * (long)batch + b] = (gm == -INFINITY || L <= 0.f) ? -INFINITY : gm + logf(L);
    }
}

template <int SPLIT, int BT, typename TP>
static void launch_sb_bt(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                         torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase,
                         int batch, int block_size, int num_edges) {
    const long ng = nids.size(0);
    const long groups = block_size / 32;
    dim3 grid((unsigned int)(ng * groups), (unsigned int)((batch + BT - 1) / BT));
    dim3 blk(32, SPLIT);
    // On the current stream, so a CUDA-graph capture of the forward records it (see cat_backward.cu).
    sb_fwd_bt_kernel<SPLIT, BT, TP><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), param_ptr<TP>(params),
        nids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(),
        batch, block_size, num_edges);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// cfg id -> (SPLIT, BT); `smallbatch_fw_bt_configs` lists them as SPLIT * 100 + BT. The autotuner picks per shape.
// MEASURED (HMM-4096 layer, 4 x 1024 nodes, 4096 edges, fp32, us; the per-sample kernel above in brackets): batch 1
// 25 [20], 4 27 [57], 8 33 [110], 16 49 [199, Triton 77], 32 88 [391], 63 191 [752]. BT = 32 configs and SPLIT = 4
// never won (32 multiply-adds a parameter a lane); past batch ~16 the cost is those multiply-adds, which is where the
// tensor-core tiles take over.
#define SB_BT_CONFIGS(X)      \
    X(0, 32, 4) X(1, 32, 8) X(2, 16, 8) X(3, 16, 16) X(4, 8, 16) X(5, 16, 4)

template <typename TP>
static void dispatch_sb_bt(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                           torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase, int b, int bs, int ne, int64_t cfg) {
    switch (cfg) {
#define SB_BT_CASE(ID, S, T) case ID: launch_sb_bt<S, T, TP>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, ne); break;
        SB_BT_CONFIGS(SB_BT_CASE)
#undef SB_BT_CASE
        // No silent default: an unknown cfg must not run zero kernels (a no-op launch can masquerade as correct
        // against a reused output buffer -- see the autotune note in sum_layer.py).
        default: TORCH_CHECK(false, "smallbatch_forward_sum_bt: invalid cfg ", cfg);
    }
}

// The parameters in the dtype they are stored in (`TensorCircuit.param_dtype`): fp32 or bf16.
void smallbatch_forward_sum_bt(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                               torch::Tensor nids, torch::Tensor ebase, torch::Tensor pbase,
                               int64_t batch, int64_t block_size, int64_t num_edges, int64_t cfg) {
    int b = (int)batch, bs = (int)block_size, ne = (int)num_edges;
    TORCH_CHECK(bs % 32 == 0, "smallbatch_forward_sum_bt: block_size must be a multiple of 32 (got ", bs, ")");
    if (params.scalar_type() == torch::kBFloat16) dispatch_sb_bt<__nv_bfloat16>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, ne, cfg);
    else dispatch_sb_bt<float>(node_mars, element_mars, params, nids, ebase, pbase, b, bs, ne, cfg);
}

std::vector<int64_t> smallbatch_fw_bt_configs() {
    std::vector<int64_t> out;
#define SB_BT_LIST(ID, S, T) out.push_back(S * 100 + T);
    SB_BT_CONFIGS(SB_BT_LIST)
#undef SB_BT_LIST
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smallbatch_forward_sum_bt", &smallbatch_forward_sum_bt,
          "small-batch (batch<64) block-sparse sum-layer forward, parameters loaded once per BT batch columns");
    m.def("smallbatch_fw_bt_configs", &smallbatch_fw_bt_configs, "SPLIT * 100 + BT per config id");
}
