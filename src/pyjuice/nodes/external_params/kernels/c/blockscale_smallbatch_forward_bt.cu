// SMALL-BATCH forward for the per-block multiplicative external parameterization (`BlockScaleSumParams`), the
// parameters loaded ONCE for a tile of BT batch columns.
//
// `blockscale_smallbatch_forward.cu` runs one block per (node group, SAMPLE), so its parameter stream is re-read once
// per sample -- past a sample or two that stream is the kernel (MEASURED, gated HMM-4096: batch 1 16 us a layer,
// batch 4 54; at batch 16 the Triton fallback's 174). This is that kernel's structure -- a warp = 32 consecutive nodes
// (lane = node offset, coalesced 128 B parameter loads), blockDim.y = SPLIT edge-warps walking GATES (each gate's
// edges the inner loop; `log phi` loaded once per gate; `Z` factorized per gate) -- with every lane carrying BT batch
// columns and grid.y = ceil(batch / BT), the same way `smallbatch_forward_sum_bt.cu` tiles the ungated kernel.
//
// TWO PASSES PER EDGE-WARP: the maximum of each column's exponents `x = element_mars + log phi(gate)` over the warp's
// edges first (and of `log phi` alone for Z), then `N = sum_e theta_e * exp(x_e - M)` and `Z = sum_g sigma_g *
// exp(log phi_g - Mz)` -- one exponential per (edge, column), no rescaling, the loads of the next U edges in flight.
//
// THE EXPONENTIALS ARE WARP-UNIFORM: neither `element_mars` nor `log phi` depends on the node, so lane j < BT owns
// column j, computes its terms once per edge (per gate for Z) and stages them in shared memory; every lane then does
// BT multiply-adds from the staged values. Columns past the batch see `-inf` exponents (terms 0) and are never stored.
//
// Numerically equivalent to the kernel above (fp32 log-sum-exp, a different association) -- the ~1e-6 the autotuner's
// arms already differ by. Math, per (node m, sample b), node-block nb, g(e) the gate covering edge e:
//     N = sum_e params[.. e, m] * exp( element_mars[e, b] + logphi[g(e), b] )
//     Z = sum_e params[.. e, m] * exp(                       logphi[g(e), b] )
//     node_mars[node, b] = log N - log Z;    log_z_out[nb, m, b] = log Z (when asked)

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"   // `ParamLoad<TP>`: the parameters in their stored dtype
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <vector>

constexpr int BS_BT_UNROLL = 8;   // edges in flight per edge-warp

template <int SPLIT, int BT, bool ROW_CONTIG, typename TP>
__global__ void __launch_bounds__(32 * SPLIT) bs_sb_fwd_bt_kernel(
        float* __restrict__ node_mars, const float* __restrict__ element_mars,
        const TP* __restrict__ params, const float* __restrict__ ext,
        const long* __restrict__ nids, const long* __restrict__ cids,
        const long* __restrict__ pids, const long* __restrict__ gate,
        float* __restrict__ log_z_out,
        int batch, int block_size, int n_gates, int gate_cbs, int node_cbs, int node_sh,
        int gate_sh, int gate_stride, int n_eblks, long ext_base) {
    constexpr int U = BS_BT_UNROLL;
    static_assert(BT % 4 == 0 && BT <= 32, "BT: a multiple of 4 up to one per lane");
    static_assert(SPLIT * BT <= 256, "the combine stages SPLIT x BT x 32 partial sums in shared memory");

    const int groups = block_size >> 5;            // block_size / 32
    const int gx = blockIdx.x;
    const int nb = gx / groups;                    // node block
    const int grp = gx - nb * groups;              // node group within the block
    const int b0 = blockIdx.y * BT;                // first batch column of this tile
    const int lane = threadIdx.x;                  // node offset within the group
    const int ty = threadIdx.y;                    // edge warp
    const int m_local = grp * 32 + lane;
    const bool owner = lane < BT;                                   // lane j owns column j
    const bool live = owner && (b0 + lane < batch);                 // ... and the column is a real one
    const int ncols = min(BT, batch - b0);

    // Per-edge-block tables (see the per-sample kernel): a gate lies inside one edge block
    const long* ec = cids + (long)nb * n_eblks;
    const long* ep = pids + (long)nb * n_eblks;
    const long* gt = gate + (long)nb * gate_stride;
    const long eb0 = ROW_CONTIG ? ec[0] : 0;
    const long pb0 = ROW_CONTIG ? ep[0] : 0;

    // Where gate gi's parameters / children start, and its `log phi` row (-1: no gate -> -inf)
    auto gate_bases = [&](int gi, long& bp, long& be, long& lprow) {
        const int e0 = gi << gate_sh;
        const int j = e0 >> node_sh;
        const int d = (e0 & (node_cbs - 1)) >> gate_sh;
        const long gb = (j < gate_stride) ? gt[j] : -1;
        lprow = (gb >= 0) ? (gb + ext_base + d) * (long)batch : -1;
        if (ROW_CONTIG) { bp = pb0 + (long)e0 * block_size; be = eb0 + e0; }
        else { const int w = e0 & (node_cbs - 1); bp = ep[j] + (long)w * block_size; be = ec[j] + w; }
    };

    // The stage: [edge in flight][column] terms, then [column] gate weights; reused twice for the combines.
    constexpr int STAGE = SPLIT * (U + 1) * BT, COMBINE = SPLIT * BT * 33;
    __shared__ __align__(16) float smem[STAGE > COMBINE ? STAGE : COMBINE];
    float* sE = smem + ty * ((U + 1) * BT);                         // this edge-warp's stage
    float* sW = sE + U * BT;                                        // its gate weights

    // ---- pass 1: per column, the maxima of x = element_mars + log phi over this edge-warp's gates (N) and of
    //      log phi alone (Z). The lanes split a gate's EDGES (lane-strided), each loading its edge's BT columns;
    //      the maxima are reduced across the warp with shuffles, so lane j ends with column j's.
    float mx[BT];
    #pragma unroll
    for (int j = 0; j < BT; j++) mx[j] = -INFINITY;
    float mz_lane = -INFINITY;                                      // lane j: max_g log phi[g, j]
    // A gate's bases are three table loads; the NEXT gate's are issued before this gate's work in both passes.
    long bp, be, lprow;
    if (ty < n_gates) gate_bases(ty, bp, be, lprow);
    for (int gi = ty; gi < n_gates; gi += SPLIT) {
        long bp_n = 0, be_n = 0, lprow_n = -1;
        if (gi + SPLIT < n_gates) gate_bases(gi + SPLIT, bp_n, be_n, lprow_n);
        float lp[BT];
        #pragma unroll
        for (int j = 0; j < BT; j++) lp[j] = (lprow >= 0 && j < ncols) ? ext[lprow + b0 + j] : -INFINITY;
        if (live && lprow >= 0) mz_lane = fmaxf(mz_lane, lp[lane]);
        for (int c = lane; c < gate_cbs; c += 32) {
            const float* r = element_mars + (be + c) * (long)batch + b0;
            if (ncols == BT && (((size_t)r) & 15) == 0) {
                #pragma unroll
                for (int j4 = 0; j4 < BT / 4; j4++) {
                    const float4 v = reinterpret_cast<const float4*>(r)[j4];
                    mx[j4 * 4 + 0] = fmaxf(mx[j4 * 4 + 0], v.x + lp[j4 * 4 + 0]);
                    mx[j4 * 4 + 1] = fmaxf(mx[j4 * 4 + 1], v.y + lp[j4 * 4 + 1]);
                    mx[j4 * 4 + 2] = fmaxf(mx[j4 * 4 + 2], v.z + lp[j4 * 4 + 2]);
                    mx[j4 * 4 + 3] = fmaxf(mx[j4 * 4 + 3], v.w + lp[j4 * 4 + 3]);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < BT; j++) if (j < ncols) mx[j] = fmaxf(mx[j], r[j] + lp[j]);
            }
        }
        bp = bp_n; be = be_n; lprow = lprow_n;
    }
    float M = -INFINITY;                                            // lane j: column j's maximum of x
    #pragma unroll
    for (int j = 0; j < BT; j++) {
        float v = mx[j];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        if (lane == j) M = v;
    }
    if (!live) M = -INFINITY;
    const float Mz = live ? mz_lane : -INFINITY;
    // -inf maxima (an empty / tail column) exponentiate against 0: the terms are exactly 0, never -inf - (-inf)
    const float Mf = (M == -INFINITY) ? 0.f : M;
    const float Mzf = (Mz == -INFINITY) ? 0.f : Mz;

    // ---- pass 2: N and Z
    float ln[BT], lz[BT];
    #pragma unroll
    for (int j = 0; j < BT; j++) { ln[j] = 0.f; lz[j] = 0.f; }

    if (ty < n_gates) gate_bases(ty, bp, be, lprow);
    for (int gi = ty; gi < n_gates; gi += SPLIT) {
        long bp_n = 0, be_n = 0, lprow_n = -1;
        if (gi + SPLIT < n_gates) gate_bases(gi + SPLIT, bp_n, be_n, lprow_n);
        const TP* pe = params + bp + m_local;                       // this lane's node: edge c at pe[c * block_size]
        const float* ecol = element_mars + be * (long)batch + b0 + lane;   // this lane's column: edge c at ecol[c * batch]
        // lane j: this gate's log phi for column j, and its Z weight staged
        const float lpj = (live && lprow >= 0) ? ext[lprow + b0 + lane] : -INFINITY;
        if (owner) sW[lane] = (lpj == -INFINITY) ? 0.f : __expf(lpj - Mzf);

        float sig = 0.f;                                            // the gate's parameter mass of this node
        float pe_[U], emv[U];
        auto fetch = [&](int c, float* p_out, float* em_out) {
            #pragma unroll
            for (int u = 0; u < U; u++) {
                const int cc = c + u;
                const bool ok = cc < gate_cbs;
                p_out[u] = ok ? ParamLoad<TP>::at(pe + (long)cc * block_size) : 0.f;
                em_out[u] = (ok && live) ? ecol[(long)cc * batch] : -INFINITY;
            }
        };
        fetch(0, pe_, emv);
        for (int c = 0; c < gate_cbs; c += U) {
            float pn[U], emn[U];
            if (c + U < gate_cbs) fetch(c + U, pn, emn);
            if (owner) {
                #pragma unroll
                for (int u = 0; u < U; u++) sE[u * BT + lane] = (emv[u] == -INFINITY) ? 0.f : __expf(emv[u] + lpj - Mf);
            }
            __syncwarp();
            #pragma unroll
            for (int u = 0; u < U; u++) {
                sig += pe_[u];
                const float4* t4 = reinterpret_cast<const float4*>(sE + u * BT);
                #pragma unroll
                for (int j4 = 0; j4 < BT / 4; j4++) {
                    const float4 t = t4[j4];
                    ln[j4 * 4 + 0] = fmaf(pe_[u], t.x, ln[j4 * 4 + 0]);
                    ln[j4 * 4 + 1] = fmaf(pe_[u], t.y, ln[j4 * 4 + 1]);
                    ln[j4 * 4 + 2] = fmaf(pe_[u], t.z, ln[j4 * 4 + 2]);
                    ln[j4 * 4 + 3] = fmaf(pe_[u], t.w, ln[j4 * 4 + 3]);
                }
            }
            __syncwarp();
            if (c + U < gate_cbs) {
                #pragma unroll
                for (int u = 0; u < U; u++) { pe_[u] = pn[u]; emv[u] = emn[u]; }
            }
        }
        // Z: the gate's mass times its weight, per column
        {
            const float4* w4 = reinterpret_cast<const float4*>(sW);
            #pragma unroll
            for (int j4 = 0; j4 < BT / 4; j4++) {
                const float4 w = w4[j4];
                lz[j4 * 4 + 0] = fmaf(sig, w.x, lz[j4 * 4 + 0]);
                lz[j4 * 4 + 1] = fmaf(sig, w.y, lz[j4 * 4 + 1]);
                lz[j4 * 4 + 2] = fmaf(sig, w.z, lz[j4 * 4 + 2]);
                lz[j4 * 4 + 3] = fmaf(sig, w.w, lz[j4 * 4 + 3]);
            }
        }
        __syncwarp();                                               // before the next gate's weights overwrite sW
        bp = bp_n; be = be_n; lprow = lprow_n;
    }

    // ---- combine the SPLIT partials per column: N first, then Z, through the same shared buffer
    float* sM = smem;                                               // [SPLIT][BT]: a partial's (uniform) maximum
    float* sL = smem + SPLIT * BT;                                  // [SPLIT][BT][32]: its sums per node
    float log_n[BT], log_zv[BT];
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        __syncthreads();                                            // the buffer is free (stage / previous combine)
        if (owner) sM[ty * BT + lane] = pass == 0 ? M : Mz;
        #pragma unroll
        for (int j = 0; j < BT; j++) sL[(ty * BT + j) * 32 + lane] = pass == 0 ? ln[j] : lz[j];
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < BT; j++) {
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
            const float v = (gm == -INFINITY || L <= 0.f) ? -INFINITY : gm + logf(L);
            if (pass == 0) log_n[j] = v; else log_zv[j] = v;
        }
    }

    // Every lane holds every column's result; edge-warp ty stores columns ty, ty + SPLIT, ...
    const long node = nids[nb] + m_local;
    #pragma unroll
    for (int j = 0; j < BT; j++) {
        if (j % SPLIT != ty || b0 + j >= batch) continue;
        const int b = b0 + j;
        node_mars[node * (long)batch + b] =
            (log_n[j] == -INFINITY || log_zv[j] == -INFINITY) ? -INFINITY : (log_n[j] - log_zv[j]);
        if (log_z_out != nullptr)
            log_z_out[((long)nb * block_size + m_local) * (long)batch + b] = log_zv[j];
    }
}


template <int SPLIT, int BT, bool ROW_CONTIG, typename TP>
static void launch(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                   torch::Tensor ext, torch::Tensor nids, torch::Tensor cids, torch::Tensor pids,
                   torch::Tensor gate, torch::Tensor log_z,
                   int batch, int block_size, int n_gates, int gate_cbs, int node_cbs, int node_sh,
                   int gate_sh, long ext_base) {
    const long ng = nids.size(0);
    dim3 grid((unsigned)(ng * (block_size / 32)), (unsigned)((batch + BT - 1) / BT));
    dim3 blk(32, SPLIT);

    bs_sb_fwd_bt_kernel<SPLIT, BT, ROW_CONTIG, TP><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), param_ptr<TP>(params),
        ext.data_ptr<float>(), nids.data_ptr<long>(), cids.data_ptr<long>(),
        pids.data_ptr<long>(), gate.data_ptr<long>(),
        log_z.numel() ? log_z.data_ptr<float>() : nullptr,
        batch, block_size, n_gates, gate_cbs, node_cbs, node_sh, gate_sh, (int)gate.size(1),
        (int)cids.size(1), ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// cfg id -> (SPLIT, BT); `blockscale_sb_bt_configs` lists them as SPLIT * 100 + BT. The autotuner picks per layer.
#define BS_BT_CONFIGS(X) X(0, 32, 4) X(1, 32, 8) X(2, 16, 8) X(3, 16, 16) X(4, 8, 16) X(5, 16, 4)

void blockscale_sb_forward_bt(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                              torch::Tensor ext, torch::Tensor nids, torch::Tensor cids,
                              torch::Tensor pids, torch::Tensor gate, torch::Tensor log_z,
                              int64_t block_size, int64_t num_edges, int64_t node_cbs, int64_t gate_cbs,
                              // `row_contig` precedes `cfg` deliberately: the Python autotuner substitutes the LAST
                              // argument when trying configs, so `cfg` must stay last.
                              int64_t ext_base, int64_t row_contig, int64_t cfg) {
    const int batch = node_mars.size(1);
    const int bs = (int)block_size, ne = (int)num_edges;
    const int ncbs = (int)node_cbs, gcbs = (int)gate_cbs;

    TORCH_CHECK(bs % 32 == 0, "blockscale batch-tiled small-batch forward: block_size must be a multiple of 32 (got ", bs, ")");
    TORCH_CHECK(ncbs % gcbs == 0, "blockscale batch-tiled small-batch forward: the gate's ch_block_size must divide the node's");
    TORCH_CHECK((ncbs & (ncbs - 1)) == 0 && (gcbs & (gcbs - 1)) == 0,
                "blockscale batch-tiled small-batch forward: both child block sizes must be powers of two; got node_cbs=",
                ncbs, ", gate_cbs=", gcbs);
    int node_sh = 0, gate_sh = 0;
    while ((1 << node_sh) < ncbs) ++node_sh;
    while ((1 << gate_sh) < gcbs) ++gate_sh;
    TORCH_CHECK(ne % gcbs == 0, "blockscale batch-tiled small-batch forward: the edge count (", ne, ") must be a whole "
                                "number of gates (", gcbs, ")");
    const int n_gates = ne / gcbs;

#define BS_BT_ARGS node_mars, element_mars, params, ext, nids, cids, pids, gate, log_z, \
                   batch, bs, n_gates, gcbs, ncbs, node_sh, gate_sh, (long)ext_base
#define BS_BT_CASE(ID, S, T) case ID: launch<S, T, RC, TP>(BS_BT_ARGS); break;
#define BS_BT_DISPATCH(RC)                                                                           \
    switch (cfg) {                                                                                   \
        BS_BT_CONFIGS(BS_BT_CASE)                                                                    \
        /* No silent default: an unknown cfg must not run zero kernels (see the per-sample kernel) */ \
        default: TORCH_CHECK(false, "blockscale batch-tiled small-batch forward: invalid cfg ", cfg); \
    }

    auto run = [&](auto tag) {
        using TP = decltype(tag);
        if (row_contig) { constexpr bool RC = true; BS_BT_DISPATCH(RC); }
        else { constexpr bool RC = false; BS_BT_DISPATCH(RC); }
    };
    if (params.scalar_type() == torch::kBFloat16) run(__nv_bfloat16{}); else run(float{});

#undef BS_BT_DISPATCH
#undef BS_BT_CASE
#undef BS_BT_ARGS
}


std::vector<int64_t> blockscale_sb_bt_configs() {
    std::vector<int64_t> out;
#define BS_BT_LIST(ID, S, T) out.push_back(S * 100 + T);
    BS_BT_CONFIGS(BS_BT_LIST)
#undef BS_BT_LIST
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_sb_forward_bt", &blockscale_sb_forward_bt,
          "small-batch forward for the per-block multiplicative gate (log N - log Z), parameters loaded once per BT columns");
    m.def("blockscale_sb_bt_configs", &blockscale_sb_bt_configs, "SPLIT * 100 + BT per config id");
}
