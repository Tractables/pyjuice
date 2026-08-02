// SMALL-BATCH forward for the per-block multiplicative external parameterization
// (`BlockScaleSumParams`), for the regime the CuTe/TMA fork cannot serve at all.
//
// That kernel tiles the batch by `BN >= 64`, so it needs `batch % 64 == 0` -- which excludes every
// batch below 64 outright. And it is the wrong shape there anyway: its cost is a serial walk over the
// row's edge tiles, so a batch of 1 would cost what a batch of 64 does. Autoregressive decoding, the
// case a per-sample gate is most obviously for, lands squarely in that gap.
//
// This is a fork of `smallbatch_forward_sum.cu` (the "v4" structure) with the gate folded in:
//   - one warp = 32 consecutive nodes (lane = node offset) -> fully coalesced 128B param loads;
//   - blockDim.y = SPLIT edge-warps splitting the edge reduction for occupancy;
//   - each (node, edge-warp) runs an online log-sum-exp over its edge subset; the SPLIT partials are
//     combined per node in shared memory;
//   - grid.x enumerates (node-block, node-group), grid.y = batch.
//
// WHY THERE IS NO `sigma` PASS HERE. The CuTe fork factors the normalizer by gate,
// `Z[n,b] = sum_g phi[b,g] * sigma[g,n]`, because its inner loop is a tensor-core matmul that `Z` does
// not belong in, so `sigma` is precomputed by a separate kernel. This kernel's inner loop already walks
// every edge and already holds `theta[m,c]`, so `Z = sum_c theta[m,c] * phi[b,g(c)]` is one extra FMA
// per edge into a second online accumulator -- no second kernel, no second pass over the parameters,
// and nothing to invalidate when EM moves them.
//
// Plain CUDA: no CuTe, no TMA, no CUTLASS, no tensor cores. It therefore runs wherever CUDA does, and
// unlike the CuTe fork it needs neither `batch % 64` nor `num_edges % 64` -- only `block_size % 32`.
//
// Math, per (node m, sample b), for node-block nb, writing g(e) for the gate covering edge e:
//     N = sum_e params[pb + e*block_size + m] * exp( element_mars[(eb + e)*batch + b] + logphi[g(e),b] )
//     Z = sum_e params[pb + e*block_size + m] * exp(                                    logphi[g(e),b] )
//     node_mars[node*batch + b] = log N - log Z
// Children are assumed CONTIGUOUS across the WHOLE row (child(e) = eb + e) and parameters
// block_size-strided; the caller verifies both and raises otherwise.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>


// One online (max, linear-sum) update. `-inf` is absorbing: a term whose exponent is -inf contributes
// nothing, and a pair that is still entirely -inf must be left alone -- the difference would be
// inf - inf = NaN, which is why the guard is on the NEW max rather than on the term.
__device__ __forceinline__ void lse_add(float& m, float& l, float x, float w) {
    const float nm = fmaxf(m, x);
    if (nm != -INFINITY) {
        l = l * __expf(m - nm) + w * __expf(x - nm);
        m = nm;
    }
}


// `ROW_CONTIG` -- whether this partition's whole row is ONE contiguous run, which is the dense
// fully-connected case. It is a template parameter, not a runtime test, because the alternative costs
// TWO extra dependent global loads per gate standing in front of the parameter address -- the same
// mistake the note below warns about for `gate[j]`, and measured here at +9.5% on two shapes.
// Specialized, the dense path keeps the single pair of row bases it always had.
template <int SPLIT, bool TWOPASS, bool ROW_CONTIG>
__global__ void __launch_bounds__(32 * SPLIT) bs_sb_fwd_kernel(
        float* __restrict__ node_mars, const float* __restrict__ element_mars,
        const float* __restrict__ params, const float* __restrict__ ext,
        const long* __restrict__ nids, const long* __restrict__ cids,
        const long* __restrict__ pids, const long* __restrict__ gate,
        float* __restrict__ log_z_out,
        int batch, int block_size, int n_gates, int gate_cbs, int node_cbs, int node_sh,
        int gate_sh, int gate_stride, int n_eblks, long ext_base) {

    const int groups = block_size >> 5;            // block_size / 32
    const int gx = blockIdx.x;
    const int nb = gx / groups;                    // node block
    const int grp = gx - nb * groups;              // node group within the block
    const int b = blockIdx.y;                      // sample
    const int lane = threadIdx.x;                  // node offset within the group
    const int ty = threadIdx.y;                    // edge warp
    const int m_local = grp * 32 + lane;

    // ONE ENTRY PER EDGE BLOCK, read from the compiled tables. This kernel used to take a single
    // base for the WHOLE row and walk it, which assumed every edge of the row was one contiguous run
    // -- true only for a dense, fully-connected row, and the reason padded and block-sparse layers
    // were declined here. A gate lies inside one edge block (`gate_cbs` divides `node_cbs`), so the
    // per-gate loop below can take its bases from the edge block it belongs to, which is contiguous
    // by the compile-time invariant. `n_eblks` is the tables' stride; note it is NOT `gate_stride`,
    // which is the layer-wide `ext_max_n_eblks` and can be smaller.
    const long* ec = cids + (long)nb * n_eblks;
    const long* ep = pids + (long)nb * n_eblks;
    const long* gt = gate + (long)nb * gate_stride;
    // Hoisted for the contiguous case: the row's single pair of bases, read once.
    const long eb0 = ROW_CONTIG ? ec[0] : 0;
    const long pb0 = ROW_CONTIG ? ep[0] : 0;

    float mn = -INFINITY, ln = 0.0f;               // the node value N
    float mz = -INFINITY, lz = 0.0f;               // its normalizer Z

    // THE LOOP WALKS GATES, NOT EDGES, and each gate's own edges are the inner loop. Two things fall
    // out, and both matter more than they look:
    //
    //   * `gate[j]` and `phi` are loaded ONCE PER GATE rather than once per edge. Per edge, `gate[j]`
    //     is a dependent global load standing in front of the `phi` address, which is the single
    //     mistake that made an earlier version of the CuTe path's stabilizer kernel 16x slower than
    //     its own bandwidth bound;
    //   * `Z` FACTORIZES. `phi` is constant across a gate's edges, so that gate contributes
    //     `phi * sum_{c in gate} theta[m,c]` -- one online update per gate instead of `gate_cbs` of
    //     them. This is the same factorization the CuTe fork precomputes as `sigma` in a separate
    //     pass; here the inner loop already holds `theta`, so it costs one add per edge and no pass.
    //
    // `N` cannot be factored the same way -- its exponent carries `element_mars`, which varies per
    // edge -- so its cost is what the ungated kernel already pays.
    for (int gi = ty; gi < n_gates; gi += SPLIT) {
        const int e0 = gi << gate_sh;

        // Shifts, not divisions: both block sizes are powers of two (checked by the caller). Hoisted
        // here, this address arithmetic is paid once per gate.
        const int j = e0 >> node_sh;
        const int d = (e0 & (node_cbs - 1)) >> gate_sh;

        const long gb = (j < gate_stride) ? gt[j] : -1;
        const float lp = (gb >= 0) ? ext[(gb + ext_base + d) * (long)batch + b] : -INFINITY;

        // Where this gate's parameters and children start. For a contiguous row that is the row's
        // base plus the gate's own offset -- what this kernel always did. Otherwise it is the base of
        // the EDGE BLOCK the gate sits in, plus the gate's offset inside that block: `j` already
        // indexes it for the gate lookup above, and the inner loop stays within the block because
        // `gate_cbs` divides `node_cbs`, so it never leaves the run the compile step guarantees is
        // contiguous. A padded block's entries are 0 -- the dummy `-inf` element and the dummy zero
        // parameter -- so padding needs no special case here either.
        long bp, be;
        if (ROW_CONTIG) {
            bp = pb0 + (long)e0 * block_size;
            be = eb0 + e0;
        } else {
            const int w = e0 & (node_cbs - 1);
            bp = ep[j] + (long)w * block_size;
            be = ec[j] + w;
        }
        const float* pe = &params[bp + m_local];
        const float* em = &element_mars[be * (long)batch + b];

        // The gate's own log-sum-exp first, WITHOUT the gate: `log phi` is constant across these
        // edges, so adding it per edge is `gate_cbs` adds to achieve what one shift of the merged max
        // does. It is also the better-conditioned order -- the per-edge values never carry the gate's
        // magnitude, so `element_mars` keeps its precision however extreme `phi` is.
        //
        // Both pointers walk by a constant stride; indexing them as `c * stride` would cost a 64-bit
        // multiply per edge for an address the previous iteration already computed.
        float gm = -INFINITY, gl = 0.0f, sig = 0.0f;

        if (!TWOPASS) {
            // ONLINE: one pass, but `(gm, gl)` carries from edge to edge, so each parameter load waits
            // on the previous edge's arithmetic.
            for (int c = 0; c < gate_cbs; c++) {
                const float p = *pe;               // 32 consecutive params, coalesced
                lse_add(gm, gl, *em, p);           // element_mars is warp-uniform
                sig += p;                          // the gate's parameter mass, all Z needs of it
                pe += block_size;
                em += batch;
            }
        } else {
            // TWO-PASS: take the maximum first, then exponentiate into four independent accumulators.
            // That breaks the dependency and halves the exponentials -- no rescaling term -- at the
            // cost of re-reading `element_mars` (warp-uniform and L1-hot; the parameters, which are the
            // bytes actually being streamed, are still read once).
            //
            // It is a config rather than the default because it is NOT uniformly better: measured, it
            // wins by 12% at gate_cbs=16 on a 2048-wide layer and loses by up to 29% at gate_cbs=8,
            // where the chain is too short to be worth a second pass. The autotuner picks per layer.
            {
                const float* e = em;
                int c = 0;
                float x0 = -INFINITY, x1 = -INFINITY;
                for (; c + 1 < gate_cbs; c += 2, e += 2 * batch) {
                    x0 = fmaxf(x0, e[0]);
                    x1 = fmaxf(x1, e[batch]);
                }
                if (c < gate_cbs) x0 = fmaxf(x0, e[0]);
                gm = fmaxf(x0, x1);
            }

            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int c = 0;
            if (gm != -INFINITY) {
                for (; c + 3 < gate_cbs; c += 4) {
                    const float p0 = pe[0], p1 = pe[block_size];
                    const float p2 = pe[2 * block_size], p3 = pe[3 * block_size];
                    g0 = fmaf(p0, __expf(em[0] - gm), g0);
                    g1 = fmaf(p1, __expf(em[batch] - gm), g1);
                    g2 = fmaf(p2, __expf(em[2 * batch] - gm), g2);
                    g3 = fmaf(p3, __expf(em[3 * batch] - gm), g3);
                    s0 += p0; s1 += p1; s2 += p2; s3 += p3;
                    pe += 4 * block_size;
                    em += 4 * batch;
                }
                for (; c < gate_cbs; c++, pe += block_size, em += batch) {
                    const float p = *pe;
                    g0 = fmaf(p, __expf(*em - gm), g0);
                    s0 += p;
                }
            } else {
                // every edge under this gate is -inf; Z still needs the parameter mass
                for (; c < gate_cbs; c++, pe += block_size) s0 += *pe;
            }

            gl = (g0 + g1) + (g2 + g3);
            sig = (s0 + s1) + (s2 + s3);
        }

        lse_add(mn, ln, gm + lp, gl);
        lse_add(mz, lz, lp, sig);
    }

    // BOTH RUNNING MAXIMA ARE WARP-UNIFORM. `element_mars` and `log phi` are broadcast across the
    // warp -- only the parameter varies with the lane -- so every lane of a warp holds the same `mn`
    // and the same `mz`, and one lane per warp is enough to publish them. Storing them per (warp, node)
    // like the sums costs twice the shared memory for `SPLIT` distinct values, and at SPLIT=32 that was
    // 16 KB against the ungated kernel's 8 KB, on a kernel already pinned at the 64-register cap that a
    // 1024-thread block imposes.
    __shared__ float sMn[SPLIT], sMz[SPLIT];
    __shared__ float sLn[SPLIT][32], sLz[SPLIT][32];
    if (lane == 0) { sMn[ty] = mn; sMz[ty] = mz; }
    sLn[ty][lane] = ln; sLz[ty][lane] = lz;
    __syncthreads();

    if (ty != 0) return;

    // COMBINE THE PARTIALS. Both steps below exist because the partial maxima are warp-uniform, which
    // makes the obvious loop -- `L += sL[k][lane] * expf(sM[k] - M)` over k -- quietly quadratic: the
    // exponential does not depend on `lane`, so all 32 lanes compute the same 32 values, 1024
    // exponentials per warp where SPLIT of them would do. At batch 1 the main loop is only a few gates
    // long and this epilogue dominates the kernel.
    static_assert(SPLIT <= 32, "the partial reduction puts one partial per lane");

    const float mnk = (lane < SPLIT) ? sMn[lane] : -INFINITY;
    const float mzk = (lane < SPLIT) ? sMz[lane] : -INFINITY;

    float Mn = mnk, Mz = mzk;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        Mn = fmaxf(Mn, __shfl_xor_sync(0xffffffffu, Mn, off));
        Mz = fmaxf(Mz, __shfl_xor_sync(0xffffffffu, Mz, off));
    }

    // one exponential per partial, not per (partial, node)
    __shared__ float wn[SPLIT], wz[SPLIT];
    if (lane < SPLIT) {
        wn[lane] = (Mn == -INFINITY) ? 0.0f : __expf(mnk - Mn);
        wz[lane] = (Mz == -INFINITY) ? 0.0f : __expf(mzk - Mz);
    }
    __syncwarp();

    float Ln = 0.0f, Lz = 0.0f;
    #pragma unroll
    for (int k = 0; k < SPLIT; k++) {
        Ln = fmaf(sLn[k][lane], wn[k], Ln);
        Lz = fmaf(sLz[k][lane], wz[k], Lz);
    }

    const float log_n = (Mn == -INFINITY || Ln <= 0.0f) ? -INFINITY : Mn + logf(Ln);
    const float log_z = (Mz == -INFINITY || Lz <= 0.0f) ? -INFINITY : Mz + logf(Lz);

    node_mars[(nids[nb] + m_local) * (long)batch + b] =
        (log_n == -INFINITY || log_z == -INFINITY) ? -INFINITY : (log_n - log_z);

    if (log_z_out != nullptr)
        log_z_out[((long)nb * block_size + m_local) * (long)batch + b] = log_z;
}


template <int SPLIT, bool TWOPASS, bool ROW_CONTIG>
static void launch(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                   torch::Tensor ext, torch::Tensor nids, torch::Tensor cids, torch::Tensor pids,
                   torch::Tensor gate, torch::Tensor log_z,
                   int batch, int block_size, int n_gates, int gate_cbs, int node_cbs, int node_sh,
                   int gate_sh, long ext_base) {
    const long ng = nids.size(0);
    dim3 grid((unsigned)(ng * (block_size / 32)), (unsigned)batch);
    dim3 blk(32, SPLIT);

    bs_sb_fwd_kernel<SPLIT, TWOPASS, ROW_CONTIG><<<grid, blk, 0, c10::cuda::getCurrentCUDAStream()>>>(
        node_mars.data_ptr<float>(), element_mars.data_ptr<float>(), params.data_ptr<float>(),
        ext.data_ptr<float>(), nids.data_ptr<long>(), cids.data_ptr<long>(),
        pids.data_ptr<long>(), gate.data_ptr<long>(),
        log_z.numel() ? log_z.data_ptr<float>() : nullptr,
        batch, block_size, n_gates, gate_cbs, node_cbs, node_sh, gate_sh, (int)gate.size(1),
        (int)cids.size(1), ext_base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void blockscale_sb_forward(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                           torch::Tensor ext, torch::Tensor nids, torch::Tensor cids,
                           torch::Tensor pids, torch::Tensor gate, torch::Tensor log_z,
                           int64_t block_size, int64_t num_edges, int64_t node_cbs, int64_t gate_cbs,
                           // `row_contig` precedes `cfg` deliberately: the Python autotuner substitutes the LAST
                           // argument when trying configs, so `cfg` must stay last.
                           int64_t ext_base, int64_t row_contig, int64_t cfg) {

    const int batch = node_mars.size(1);
    const int bs = (int)block_size, ne = (int)num_edges;
    const int ncbs = (int)node_cbs, gcbs = (int)gate_cbs;

    TORCH_CHECK(bs % 32 == 0, "blockscale small-batch forward: block_size must be a multiple of 32 "
                              "(got ", bs, ")");
    TORCH_CHECK(ncbs % gcbs == 0, "blockscale small-batch forward: the gate's ch_block_size must "
                                  "divide the node's");
    TORCH_CHECK((ncbs & (ncbs - 1)) == 0 && (gcbs & (gcbs - 1)) == 0,
                "blockscale small-batch forward: both child block sizes must be powers of two; got "
                "node_cbs=", ncbs, ", gate_cbs=", gcbs);

    int node_sh = 0, gate_sh = 0;
    while ((1 << node_sh) < ncbs) ++node_sh;
    while ((1 << gate_sh) < gcbs) ++gate_sh;

    TORCH_CHECK(ne % gcbs == 0, "blockscale small-batch forward: the edge count (", ne, ") must be a "
                                "whole number of gates (", gcbs, ")");
    const int n_gates = ne / gcbs;

#define BS_SB_ARGS node_mars, element_mars, params, ext, nids, cids, pids, gate, log_z,            \
                   batch, bs, n_gates, gcbs, ncbs, node_sh, gate_sh, (long)ext_base

    // cfg = SPLIT x inner-loop form; see `blockscale_sb_configs`. `row_contig` selects the
    // addressing specialization: see the note on `ROW_CONTIG` at the kernel.
#define BS_SB_DISPATCH(RC)                                                                         \
    switch (cfg) {                                                                                 \
        case 0: launch<32, false, RC>(BS_SB_ARGS); break;                                          \
        case 1: launch<16, false, RC>(BS_SB_ARGS); break;                                          \
        case 2: launch< 8, false, RC>(BS_SB_ARGS); break;                                          \
        case 3: launch< 4, false, RC>(BS_SB_ARGS); break;                                          \
        case 4: launch<32, true , RC>(BS_SB_ARGS); break;                                          \
        case 5: launch<16, true , RC>(BS_SB_ARGS); break;                                          \
        case 6: launch< 8, true , RC>(BS_SB_ARGS); break;                                          \
        case 7: launch< 4, true , RC>(BS_SB_ARGS); break;                                          \
        /* No silent default: an unknown cfg must not run zero kernels, which against a reused   */ \
        /* output buffer can masquerade as a correct -- and very fast -- result to the autotuner.*/ \
        default: TORCH_CHECK(false, "blockscale small-batch forward: invalid cfg ", cfg);           \
    }

    if (row_contig) { BS_SB_DISPATCH(true); } else { BS_SB_DISPATCH(false); }

#undef BS_SB_DISPATCH
#undef BS_SB_ARGS
}


// config id -> SPLIT, and whether the inner loop uses the two-pass form (ids 4..7)
std::vector<int64_t> blockscale_sb_configs() { return {32, 16, 8, 4, 32, 16, 8, 4}; }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_sb_forward", &blockscale_sb_forward,
          "small-batch forward for the per-block multiplicative gate (log N - log Z)");
    m.def("blockscale_sb_configs", &blockscale_sb_configs, "SPLIT per config id");
}
