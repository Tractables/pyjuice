// CuTe/TMA forward for the per-block multiplicative external parameterization (`BlockScaleSumParams`).
//
// A fork of pyjuice's `tlmm_forward_sum.cu` -- same TMA bulk-load, same MN-major staging, same SM80
// bf16 MMA, same register-resident online log-sum-exp -- with ONE addition in the k-loop and the
// normalizer computed alongside. Forked rather than parameterized because the standard kernel is on the
// hot path of every sum layer and should not grow a branch for this.
//
// The effective parameters are
//
//     theta_tilde[b,n,c] = phi[b, g(n,c)] * theta[n,c] / Z[n,b]
//     Z[n,b]             = sum_c phi[b, g(n,c)] * theta[n,c]
//
// i.e. `phi` reweights the GATE-level mixture weights per sample and leaves each gate's internal
// conditional alone -- an exact no-op when a node has one gate.
//
// WHY THE MAIN KERNEL ONLY NEEDS ONE EXTRA LOOP: `phi` is constant within a gate, so it pulls out of the
// matmul and, in log space, folds into the child values:
//
//     N[n,b] = sum_c theta[n,c] * exp( element_mars[c,b] + log phi[b,g] )
//
// The standard kernel already TMA-loads `element_mars` into an fp32 scratch before max-stabilizing it,
// so the fold is an elementwise add on that scratch between the barrier wait and the max. Nothing else
// about the pipeline changes -- the MMA, the operand staging and the accumulators are untouched.
//
// WHY `Z` IS NOT A SECOND MMA: `Z` does not involve `element_mars` at all. Factored by gate,
//
//     Z[n,b] = sum_g phi[b,g] * sigma[g,n],     sigma[g,n] = sum_{c in g} theta[n,c]
//
// and `sigma` is BATCH-INDEPENDENT. A second MMA would spend another `BN x BK` of shared memory and a
// full second LDSM_T + gemm computing something two small kernels get for a fraction of the work: at
// `block_size = 64` with 8 child gates per edge block there are 8x fewer gate terms than edges, and
// `sigma` costs one pass over the parameters (recomputed each forward, so there is no cache to
// invalidate when EM moves them).
//
// SCOPE: inherits the standard kernel's gate -- TMA-capable sm_90+, LL propagation, bf16, no partial
// eval or tempering, `block_size % BM == 0`, `batch % BN == 0`, `num_edges % BK == 0`. `n_node_gates`
// must be 1: a gate that varies across the nodes of a tile cannot share the staged tile. The Python
// layer raises `NotImplementedError` outside all of that rather than falling back to something slower.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

#define BK 64

// ---- TMA / mbarrier PTX helpers (sm_90+), as in the standard kernel ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, int cnt) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar, int bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}
__device__ __forceinline__ void tma_load_2d(void* smem, const CUtensorMap* desc, int c0, int c1,
                                            uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(s),
        "l"(reinterpret_cast<uint64_t>(desc)), "r"(c0), "r"(c1), "r"(b)
        : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n .reg .pred p;\n LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@p bra DONE;\n bra LAB_WAIT;\n DONE:\n}\n" ::"r"(a),
                 "r"(phase));
}

extern __shared__ char smem_raw[];


// ============================================================ main kernel: log N

// One tile's gate work, all of it reading `sPh` (the staged log-gates) and nothing else, so it needs no
// internal barrier:
//
//   * add each log-gate onto every element of `element_mars` it covers, which leaves the max and
//     exponent loops downstream exactly as the standard kernel writes them;
//   * advance `sMz`, the running max log-gate over the row -- the stabilizer for `Z`. Free here: the
//     gates are in shared already and the block visits every edge tile of its row. As its own kernel it
//     cost 27 us flat, since a standalone pass re-reads `phi` from DRAM and chases `gate[j]` before it
//     can issue each load, with too few warps in flight to hide either;
//   * emit `sPe = exp(log phi - mz_new)` for the normalizer's contraction, and `sScale`, the factor
//     that rescales what `Z` has accumulated so far to the new stabilizer.
//
// The stabilizer and the exponentials are computed by the SAME thread per batch column, so `mz_new`
// never has to become visible to anyone else mid-tile.
//
// `GC` -- the gate's child-block width, clamped to the edge tile -- is a template parameter so
// `BK / GC`, `e / GC` and every trip count are compile-time; see the call site.
template <int BN, int NTH, int GC>
__device__ __forceinline__ void gate_pass(float* __restrict__ sEm, const float* __restrict__ sPh,
                                          float* __restrict__ sMz,
                                          float* __restrict__ sScale, int tid) {
    constexpr int NGT = BK / GC;

    CUTE_UNROLL
    for (int i = tid * 4; i < BK * BN; i += NTH * 4) {
        const int e = i / BN, b = i % BN;
        float4 v = *(const float4*)&sEm[i];
        const float4 g = *(const float4*)&sPh[(e / GC) * BN + b];
        v.x += g.x; v.y += g.y; v.z += g.z; v.w += g.w;
        *(float4*)&sEm[i] = v;
    }

    for (int b = tid; b < BN; b += NTH) {
        float gm = -INFINITY;
        CUTE_UNROLL
        for (int gi = 0; gi < NGT; gi++) gm = fmaxf(gm, sPh[gi * BN + b]);

        const float mzo = sMz[b], mzn = fmaxf(mzo, gm);
        sMz[b] = mzn;

        // Both guards are for the all-gates-are-(-inf) column, where `mzn` is -inf and the difference
        // would be inf - inf = NaN. There `Z` must stay 0, which makes `log_z` -inf, which is what the
        // standalone normalizer produced for the same column.
        sScale[b] = (mzn == -INFINITY || mzo == -INFINITY) ? 0.0f : __expf(mzo - mzn);
    }
}


// Fallback for a gate width outside the specialized set; correct, just not unrolled.
template <int BN, int NTH>
__device__ __forceinline__ void gate_pass_rt(float* __restrict__ sEm, const float* __restrict__ sPh,
                                             float* __restrict__ sMz,
                                             float* __restrict__ sScale, int tid, int gc) {
    const int ngt = BK / gc;

    for (int i = tid * 4; i < BK * BN; i += NTH * 4) {
        const int e = i / BN, b = i % BN;
        float4 v = *(const float4*)&sEm[i];
        const float4 g = *(const float4*)&sPh[(e / gc) * BN + b];
        v.x += g.x; v.y += g.y; v.z += g.z; v.w += g.w;
        *(float4*)&sEm[i] = v;
    }

    for (int b = tid; b < BN; b += NTH) {
        float gm = -INFINITY;
        for (int gi = 0; gi < ngt; gi++) gm = fmaxf(gm, sPh[gi * BN + b]);

        const float mzo = sMz[b], mzn = fmaxf(mzo, gm);
        sMz[b] = mzn;
        sScale[b] = (mzn == -INFINITY || mzo == -INFINITY) ? 0.0f : __expf(mzo - mzn);
    }
}


// Gates per edge tile. A gate wider than the tile is clamped to it -- the tile then sits inside a single
// gate -- so this is never zero.
__host__ __device__ __forceinline__ int ngt_of(int gate_cbs) {
    return BK / ((gate_cbs < BK) ? gate_cbs : BK);
}


// Shared memory a tile needs: operands + element_mars scratch + the six per-batch-column scalars +
// the barrier, then the gates (staged and exponentiated) and the tile's parameter mass.
int smem_bytes(int BM, int BN, int gate_cbs) {
    const int ngt = ngt_of(gate_cbs);
    return BM * BK * 2 + 2 * BN * BK * 2 + BK * BN * 4 + 6 * BN * 4 + 64 + ngt * BN * 4;
}


// `SINGLE_RUN` -- whether a k-tile is exactly ONE step-run, i.e. `ch_block_size >= 64`. It is a
// template parameter and not a runtime test because it decides the TRIP COUNTS below: left
// runtime, the operand-staging loop's bound `(BM * step) / 8` stops being compile-time and the
// loop no longer unrolls, which measured +2.6 to +2.9% on dense shapes. Specialized, the common
// case generates exactly the single-transfer, fully-unrolled code it did before this kernel
// learned to read the tables. Same reason `gate_pass` is specialized on the gate width.
template <int BM, int BN, int WM, int WN, bool SINGLE_RUN>
__global__ void __launch_bounds__(WM * WN * 32) blockscale_tlmm_kernel(
        float* __restrict__ node_mars, const float* __restrict__ mp,
        const float* __restrict__ ext, const long* __restrict__ nids,
        const long* __restrict__ cids, const long* __restrict__ pids,
        const long* __restrict__ gate, float* __restrict__ log_z_out,
        int batch, int block_size, int knt, int gate_stride,
        int node_cbs, int gate_cbs, int node_sh, int gate_sh, long ext_base, int pid_m_offset,
        int step_sh,
        const __grid_constant__ CUtensorMap desc) {
    constexpr int NTH = WM * WN * 32;

    int mtiles = block_size / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int nblock = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_nid = nids[nblock];
    int tid = threadIdx.x;

    // ---- THE TABLES ARE READ AT EDGE-BLOCK GRANULARITY, NOT TILE GRANULARITY ----
    //
    // `step = min(node_cbs, BK)` is the widest run of compiled edge slots that is guaranteed to be
    // contiguous in `element_mars` and `block_size`-strided in `params`. The compile step ASSERTS
    // exactly that ("each edge block's children occupy a contiguous run of `cids`"), so a run of this
    // width is safe for any topology -- ragged, block-sparse or dense. So the kernel indexes the
    // COMPILED TABLES at that stride rather than deriving addresses across the whole tile.
    //
    // At `step == BK` (every `ch_block_size >= 64`, the common case) `neb` is 1 and this reduces
    // EXACTLY to one bulk transfer per k-tile off `eb[kt]` -- the code this replaces. Only a child
    // block narrower than the k-tile pays anything, and that is precisely the case that used to be
    // refused outright.
    const int step = SINGLE_RUN ? BK : (1 << step_sh);
    const int neb  = SINGLE_RUN ? 1  : (BK >> step_sh);   // step-runs per k-tile
    // The COMPILED TABLES themselves, at their own row stride (`num_edges == knt * BK`). The kernel
    // reads the first slot of each step-run out of `cids` / `pids`; nothing is precomputed for it and
    // nothing can drift out of step with what the layer compiled.
    const long* ec = cids + (long)nblock * (knt * neb);
    const long* ep = pids + (long)nblock * (knt * neb);
    const long* gt = gate + (long)nblock * gate_stride;

    auto swz = composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<Shape<_8, _8>, _8>, Stride<Stride<_1, _64>, _8>>{});
    auto sAl = tile_to_shape(swz, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swz, make_shape(Int<BN>{}, Int<BK>{}));
    bfloat16_t* pA = (bfloat16_t*)smem_raw;
    bfloat16_t* pBs = pA + cosize(sAl);
    bfloat16_t* pB2 = pBs + cosize(sBl);         // the normalizer's operand: exp(log phi - mz)
    float* sEm = (float*)(pB2 + cosize(sBl));
    float* sMx = sEm + BK * BN;
    float* sMz = sMx + BN;                       // [BN] running max log-gate
    float* sScale = sMz + BN;                    // [BN] rescale for Z when that max moves
    float* sMrun = sScale + BN;                  // [BN] running max of the log-sum-exp
    float* sLS = sMrun + BN;                     // [BN] rescale for the running sum when IT moves
    float* sPS = sLS + BN;                       // [BN] weight of this tile's partial
    uint64_t* bar = (uint64_t*)(sPS + BN + 4);
    float* sPh = (float*)(bar + 4);              // [ngt, BN] staged log-gates

    Tensor sAt = make_tensor(make_smem_ptr(pA), sAl);
    Tensor sBt = make_tensor(make_smem_ptr(pBs), sBl);
    Tensor sB2t = make_tensor(make_smem_ptr(pB2), sBl);
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                  Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor tCrA = thr.partition_fragment_A(sAt);
    Tensor tCrB = thr.partition_fragment_B(sBt);
    Tensor tCrB2 = thr.partition_fragment_B(sB2t);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    Tensor tXsA = s2rA.get_thread_slice(tid).partition_S(sAt);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXsB = s2rB.get_thread_slice(tid).partition_S(sBt);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor tXsB2 = s2rB.get_thread_slice(tid).partition_S(sB2t);
    Tensor tXrB2 = s2rB.get_thread_slice(tid).retile_D(tCrB2);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    Tensor tCrZ = thr.partition_fragment_C(cC);      // the normalizer, in the same fragment layout
    clear(tCrL);
    clear(tCrZ);

    // The running max is NOT a fragment. The standard kernel carries it as one (`tCrM`), but every
    // element of it that shares a batch column holds the same number -- it is only ever merged with
    // `sMx[b]`, which has no `m` dependence -- so a `[BN]` array in shared memory holds the same state
    // in 1/32nd the registers. Those registers are what pay for `Z`'s accumulator, and the two
    // exponentials per element per tile that the fragment form needed collapse to two per COLUMN.

    const int gcbs_eff = (gate_cbs < BK) ? gate_cbs : BK;
    const int ngt = BK / gcbs_eff;

    // A SECOND STAGING BUFFER DOES NOT PAY. Issuing tile `kt+1`'s transfer before waiting on `kt`, so
    // it lands during `kt`'s compute, is the textbook fix for a latency-bound mainloop -- and here it
    // changed nothing at 16 or 64 blocks and cost 65% at 256 blocks, where doubling the staging buffer
    // and keeping two transfers per SM in flight makes every transfer slower. The `element_mars` load
    // is evidently not the latency that is exposed; the gate and sigma staging in front of the wait
    // already covers it.
    if (tid == 0) mbar_init(bar, 1);
    for (int b = tid; b < BN; b += NTH) { sMz[b] = -INFINITY; sMrun[b] = -INFINITY; }
    __syncthreads();

    for (int kt = 0; kt < knt; kt++) {
        // ONE BULK TRANSFER PER STEP-RUN. The box is `[BN, step]`, so `neb` of them fill the same
        // `[BK, BN]` scratch this kernel always used, at the same total byte count -- the barrier
        // accounting is unchanged, and at `neb == 1` this is bit-for-bit the single transfer it
        // replaces. Each run's first child comes from `cids`, so nothing is assumed about how the
        // runs relate to one another: they may be adjacent (dense), separated (block-sparse), or the
        // dummy row 0 (padding).
        if (tid == 0) {
            mbar_expect(bar, BK * BN * 4);
            if (SINGLE_RUN) {
                tma_load_2d(sEm, &desc, b0, (int)ec[kt], bar);
            } else {
                for (int i = 0; i < neb; i++)
                    tma_load_2d(sEm + (long)i * step * BN, &desc, b0, (int)ec[kt * neb + i], bar);
            }
        }


        // ---- STAGE THE GATES for this tile ----
        //
        // Issued BEFORE waiting on the barrier: the gates do not depend on `element_mars`, so these
        // loads overlap the TMA transfer instead of queueing behind it.
        //
        // Per (GATE, batch), not per (edge, batch): a tile spans only `BK / gate_cbs` distinct gates and
        // every edge under one takes the same value. The per-element form loaded each `phi` `gate_cbs`
        // times over and paid two integer divisions per element -- 4096 loads and 8192 divisions per
        // tile at `gate_cbs = 8`, against 512 and 1024 here.
        for (int i = tid; i < ngt * BN; i += NTH) {
            const int gi = i / BN, b = i % BN;
            const int ge0 = kt * BK + gi * gcbs_eff;
            const int j = ge0 >> node_sh;
            const int d = (ge0 & (node_cbs - 1)) >> gate_sh;

            const long gb = (j < gate_stride) ? gt[j] : -1;
            sPh[i] = (gb >= 0) ? ext[(gb + ext_base + d) * (long)batch + b0 + b] : -INFINITY;
        }

        mbar_wait(bar, kt & 1);      // one buffer, so the phase just alternates
        __syncthreads();

        // ---- ADD THE GATES INTO THE STAGED `element_mars` ----
        //
        // Specialized on the gate width so `BK / GC` and `e / GC` are compile-time. Consuming the gates
        // in place in the max and exponent loops instead -- which reads each one rather than writing it
        // back, and looks like strictly less work -- made both loops' trip counts runtime values, so
        // neither unrolled and every shared load paid its full latency. That cost 31 us against the
        // standard kernel, more than the entire normalizer.
        //
        // With the gates folded in, the two loops below are the standard kernel's, unchanged: same
        // compile-time bounds, same unrolling, and `(sEm + lp) - mx` in the same order as before, so the
        // arithmetic is unchanged too. `-inf + anything = -inf`, so a padded edge (child 0, the dummy)
        // and a row with fewer edge blocks (`gate == -1`) both stay -inf and contribute exactly nothing.
        switch (gcbs_eff) {
            case  4: gate_pass<BN, NTH,  4>(sEm, sPh, sMz, sScale, tid); break;
            case  8: gate_pass<BN, NTH,  8>(sEm, sPh, sMz, sScale, tid); break;
            case 16: gate_pass<BN, NTH, 16>(sEm, sPh, sMz, sScale, tid); break;
            case 32: gate_pass<BN, NTH, 32>(sEm, sPh, sMz, sScale, tid); break;
            case 64: gate_pass<BN, NTH, 64>(sEm, sPh, sMz, sScale, tid); break;
            default: gate_pass_rt<BN, NTH>(sEm, sPh, sMz, sScale, tid, gcbs_eff); break;
        }
        __syncthreads();

        // The tile's max, and with it the two factors that merge this tile into the running
        // log-sum-exp. Both depend only on the batch column, so one thread per column computes them
        // once -- see `sMrun`.
        for (int b = tid; b < BN; b += NTH) {
            float mx = -INFINITY;
            for (int e = 0; e < BK; e++) mx = fmaxf(mx, sEm[e * BN + b]);
            sMx[b] = mx;

            const float Mo = sMrun[b], nM = fmaxf(Mo, mx);
            sMrun[b] = nM;
            sLS[b] = (nM == -INFINITY || Mo == -INFINITY) ? 0.0f : __expf(Mo - nM);
            sPS[b] = (nM == -INFINITY || mx == -INFINITY) ? 0.0f : __expf(mx - nM);
        }
        __syncthreads();

        for (int i = tid; i < (BN * BK) / 8; i += NTH) {
            int e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
            float mx0 = sMx[bb], mx1 = sMx[bb + 1], mx2 = sMx[bb + 2], mx3 = sMx[bb + 3];
            float mx4 = sMx[bb + 4], mx5 = sMx[bb + 5], mx6 = sMx[bb + 6], mx7 = sMx[bb + 7];
            const float* s = &sEm[e * BN + bb];
            bfloat16_t r[8];
            r[0] = static_cast<bfloat16_t>((mx0 == -INFINITY) ? 0.f : __expf(s[0] - mx0));
            r[1] = static_cast<bfloat16_t>((mx1 == -INFINITY) ? 0.f : __expf(s[1] - mx1));
            r[2] = static_cast<bfloat16_t>((mx2 == -INFINITY) ? 0.f : __expf(s[2] - mx2));
            r[3] = static_cast<bfloat16_t>((mx3 == -INFINITY) ? 0.f : __expf(s[3] - mx3));
            r[4] = static_cast<bfloat16_t>((mx4 == -INFINITY) ? 0.f : __expf(s[4] - mx4));
            r[5] = static_cast<bfloat16_t>((mx5 == -INFINITY) ? 0.f : __expf(s[5] - mx5));
            r[6] = static_cast<bfloat16_t>((mx6 == -INFINITY) ? 0.f : __expf(s[6] - mx6));
            r[7] = static_cast<bfloat16_t>((mx7 == -INFINITY) ? 0.f : __expf(s[7] - mx7));
            *(float4*)&sBt(bb, e) = *(const float4*)r;
        }

        // The normalizer's operand: `exp(log phi - mz)` over the SAME children the matmul contracts.
        // `Z = sum_c theta * phi` is `N` with `element_mars` dropped, so it reuses the `A` operand and
        // needs only this second `B`. Computed instead as a gate-wise outer product on CUDA cores --
        // which is what this replaces -- every thread re-read shared memory for its own (m, b) pair:
        // 2560 floats per gate where 128 are distinct, a 20x re-read costing ~16 us per block at a
        // gate width of 4. `LDSM` moves each operand once, and the cost stops depending on gate width.
        for (int i = tid; i < (BN * BK) / 8; i += NTH) {
            int e = i / (BN / 8), bb = (i % (BN / 8)) * 8;
            bfloat16_t r[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float mz = sMz[bb + j];
                const float lp = sPh[(e / gcbs_eff) * BN + bb + j];
                r[j] = static_cast<bfloat16_t>((mz == -INFINITY) ? 0.f : __expf(lp - mz));
            }
            *(float4*)&sB2t(bb, e) = *(const float4*)r;
        }
        // The parameter base is per STEP-RUN, so the loop is nested BY RUN with the base hoisted out
        // of the inner body. Writing it the obvious way instead -- one flat loop indexing
        // `sPb[e >> step_sh]` per element -- cost 3.4-5.4% on DENSE shapes, because the shared-memory
        // lookup and the two shifts land in the innermost body of the operand staging and the
        // compiler cannot know that `neb == 1` collapses them. Nested, the common case is one pass
        // with a single hoisted base, i.e. exactly the addressing this replaced.
        //
        // No clamp is needed any more either: a padded run's base is the dummy parameter 0, so the
        // read lands in the dummy region rather than off the end of `params`.
        if (SINGLE_RUN) {
            const long pcr = ep[kt] + (long)tile_id * BM;
            for (int i = tid; i < (BM * BK) / 8; i += NTH) {
                int e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                const float* g = &mp[pcr + (long)e * block_size + mm];
                float4 a = *(const float4*)g, b = *(const float4*)(g + 4);
                bfloat16_t r[8];
                r[0] = static_cast<bfloat16_t>(a.x); r[1] = static_cast<bfloat16_t>(a.y);
                r[2] = static_cast<bfloat16_t>(a.z); r[3] = static_cast<bfloat16_t>(a.w);
                r[4] = static_cast<bfloat16_t>(b.x); r[5] = static_cast<bfloat16_t>(b.y);
                r[6] = static_cast<bfloat16_t>(b.z); r[7] = static_cast<bfloat16_t>(b.w);
                *(float4*)&sAt(mm, e) = *(const float4*)r;
            }
        } else {
            for (int r0 = 0; r0 < neb; r0++) {
                const long pcr = ep[kt * neb + r0] + (long)tile_id * BM;
                const int e_off = r0 * step;
                for (int i = tid; i < (BM * step) / 8; i += NTH) {
                    int e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
                    const float* g = &mp[pcr + (long)e * block_size + mm];
                    float4 a = *(const float4*)g, b = *(const float4*)(g + 4);
                    bfloat16_t r[8];
                    r[0] = static_cast<bfloat16_t>(a.x); r[1] = static_cast<bfloat16_t>(a.y);
                    r[2] = static_cast<bfloat16_t>(a.z); r[3] = static_cast<bfloat16_t>(a.w);
                    r[4] = static_cast<bfloat16_t>(b.x); r[5] = static_cast<bfloat16_t>(b.y);
                    r[6] = static_cast<bfloat16_t>(b.z); r[7] = static_cast<bfloat16_t>(b.w);
                    *(float4*)&sAt(mm, e_off + e) = *(const float4*)r;
                }
            }
        }

        __syncthreads();

        copy(s2rA, tXsA, tXrA); copy(s2rB, tXsB, tXrB);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);

        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) {
            const int b = get<1>(tCcC(i));
            tCrL(i) = tCrL(i) * sLS[b] + tCrS(i) * sPS[b];
        }

        // ...and the same matmul again for `Z`, against the `A` operand already in registers. The
        // partial is already stated in this tile's stabilizer, so merging is one rescale.
        copy(s2rB, tXsB2, tXrB2);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB2, tCrS);
        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) {
            const int b = get<1>(tCcC(i));
            tCrZ(i) = tCrZ(i) * sScale[b] + tCrS(i);
        }
        __syncthreads();
    }

    // Writes log N - log Z, and log Z for the backward. Guarded exactly as the separate normalizer was:
    // a column whose gates are all -inf leaves `Z` at zero, and both outputs are -inf there.
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (m < BM && b < BN) {
            const float mz = sMz[b], mn = sMrun[b];
            const float log_z = (tCrZ(i) <= 0.0f || mz == -INFINITY) ? -INFINITY
                                                                     : mz + logf(tCrZ(i));
            const float log_n = (mn == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + mn);

            node_mars[(off_nid + (long)tile_id * BM + m) * (long)batch + b0 + b] =
                (log_n == -INFINITY || log_z == -INFINITY) ? -INFINITY : (log_n - log_z);

            if (log_z_out != nullptr)
                log_z_out[((long)nblock * block_size + tile_id * BM + m) * (long)batch + b0 + b]
                    = log_z;
        }
    }
}


// ============================================================ launchers

template <int BM, int BN, int WM, int WN>
static void launch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                       torch::Tensor ext, torch::Tensor nids, torch::Tensor cids,
                       torch::Tensor pids, torch::Tensor gate, torch::Tensor log_z,
                       int batch, int block_size, int knt, int node_cbs, int gate_cbs,
                       int node_sh, int gate_sh, long ext_base, int step_sh) {
    // `gcbs_eff` mirrors the kernel: a gate may be wider than one edge tile
    constexpr int NTH = WM * WN * 32;
    int n_edge_rows = element_mars.size(0);
    void* base = (void*)element_mars.data_ptr<float>();

    // The box is `[BN, step]`: one bulk transfer per step-run, so `step` is part of the descriptor's
    // identity and therefore of its cache key.
    const int step = 1 << step_sh;
    static CUtensorMap desc;
    static void* desc_ptr = nullptr;
    static int desc_rows = 0, desc_batch = 0, desc_step = 0;
    if (base != desc_ptr || n_edge_rows != desc_rows || batch != desc_batch || step != desc_step) {
        cuuint64_t gdim[2] = {(cuuint64_t)batch, (cuuint64_t)n_edge_rows};
        cuuint64_t gstride[1] = {(cuuint64_t)batch * 4};
        cuuint32_t bdim[2] = {(cuuint32_t)BN, (cuuint32_t)step};
        cuuint32_t estride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, gdim, gstride, bdim, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
        desc_ptr = base; desc_rows = n_edge_rows; desc_batch = batch; desc_step = step;
    }

    const bool single = (step == BK);
    const int smem = smem_bytes(BM, BN, gate_cbs);
    TORCH_CHECK(cudaFuncSetAttribute(single ? (const void*)blockscale_tlmm_kernel<BM, BN, WM, WN, true>
                                            : (const void*)blockscale_tlmm_kernel<BM, BN, WM, WN, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem)
                    == cudaSuccess,
                "blockscale forward: this tile needs ", smem, " B of shared memory, which this device "
                "will not grant. `fitting_configs` should have excluded it.");

    int total_m = nids.size(0) * (block_size / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
#define BS_LAUNCH(SR)                                                                              \
        blockscale_tlmm_kernel<BM, BN, WM, WN, SR><<<grid, NTH, smem,                              \
                                                     c10::cuda::getCurrentCUDAStream()>>>(         \
            node_mars.data_ptr<float>(), params.data_ptr<float>(), ext.data_ptr<float>(),          \
            nids.data_ptr<long>(), cids.data_ptr<long>(), pids.data_ptr<long>(),                   \
            gate.data_ptr<long>(),                                                                 \
            log_z.numel() ? log_z.data_ptr<float>() : nullptr,                                     \
            batch, block_size, knt, (int)gate.size(1),                                             \
            node_cbs, gate_cbs, node_sh, gate_sh, ext_base, off, step_sh, desc)
        if (single) { BS_LAUNCH(true); } else { BS_LAUNCH(false); }
#undef BS_LAUNCH
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


std::vector<std::vector<int>> configs();


// The tiles this DEVICE can actually run, for this layer's shape. The opt-in shared-memory ceiling is
// not a constant of the architecture family -- it is 48 KB, 64 KB, 100 KB or 227 KB depending on the
// part -- so it is queried rather than assumed. Filtering here keeps a tile that cannot fit out of the
// autotuner, where its launch failure would be an async, sticky CUDA error rather than a skipped
// candidate.
std::vector<int> fitting_configs(int64_t block_size, int64_t batch, int64_t gate_cbs) {
    int dev = 0, smax = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    std::vector<int> out;
    const auto cfgs = configs();
    for (int i = 0; i < (int)cfgs.size(); ++i) {
        const int BM = cfgs[i][0], BN = cfgs[i][1];
        if (block_size % BM != 0 || batch % BN != 0) continue;
        if (smem_bytes(BM, BN, (int)gate_cbs) <= smax) out.push_back(i);
    }
    return out;
}


std::vector<std::vector<int>> configs() {
    // The last two are 8-warp tiles. They cover the same output tile with twice the threads, so each
    // thread holds HALF the accumulator fragment -- which is what the fork's registers are spent on --
    // and the block fits twice the warps per SM.
    return {{128, 64, 2, 2}, {64, 64, 2, 2}, {256, 64, 4, 2}, {128, 128, 2, 4},
            {64, 64, 4, 2}, {128, 64, 4, 2}};
}


void blockscale_forward(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                        torch::Tensor ext, torch::Tensor nids, torch::Tensor cids,
                        torch::Tensor pids, torch::Tensor gate,
                        torch::Tensor log_z,
                        int64_t block_size, int64_t num_edges, int64_t node_cbs, int64_t gate_cbs,
                        int64_t n_node_gates, int64_t ext_base, int64_t cfg) {

    const int batch = node_mars.size(1);
    const int rows = nids.size(0);
    const int knt = (int)(num_edges / BK);
    const int n_eblks = (int)(num_edges / node_cbs);
    const int n_child_gates = (int)(node_cbs / gate_cbs);

    TORCH_CHECK(n_node_gates == 1,
                "blockscale forward: a gate varying across the nodes of a block cannot share the staged "
                "tile; got n_node_gates = ", n_node_gates);
    TORCH_CHECK(num_edges % BK == 0, "blockscale forward: num_edges must be a multiple of ", BK);
    TORCH_CHECK(node_cbs % gate_cbs == 0,
                "blockscale forward: the gate's ch_block_size must divide the node's");

    // Both block sizes are powers of two, so the kernel indexes gates with shifts. Left as runtime
    // divisors they cost ~480 integer instructions per thread per edge tile -- ptxas has to emit the
    // full IABS/MUFU.RCP/IMAD.HI reciprocal sequence -- to produce four floats.
    TORCH_CHECK((node_cbs & (node_cbs - 1)) == 0 && (gate_cbs & (gate_cbs - 1)) == 0,
                "blockscale forward: both child block sizes must be powers of two; got node_cbs=",
                node_cbs, ", gate_cbs=", gate_cbs);
    const int node_sh = (int)std::log2((double)node_cbs);
    const int gate_sh = (int)std::log2((double)gate_cbs);

    // `step` -- the widest run of compiled edge slots guaranteed contiguous in `element_mars` and
    // `block_size`-strided in `params`. That is one edge block, clamped to the k-tile. The kernel
    // reads the first slot of each step-run straight out of the compiled `cids` / `pids`, so it makes
    // no assumption about how the runs relate to one another.
    const int step = ((int)node_cbs < BK) ? (int)node_cbs : BK;
    const int step_sh = (int)std::log2((double)step);
    TORCH_CHECK(cids.size(1) == num_edges / step && pids.size(1) == cids.size(1),
                "blockscale forward: cids/pids must hold one entry per step-run (",
                num_edges / step, "), got ", cids.size(1));

    switch (cfg) {
        case 0: launch_cfg<128, 64, 2, 2>(node_mars, element_mars, params, ext, nids, cids, pids,
                                          gate, log_z, batch, (int)block_size, knt,
                                          (int)node_cbs, (int)gate_cbs, node_sh, gate_sh, (long)ext_base, step_sh); break;
        case 1: launch_cfg<64, 64, 2, 2>(node_mars, element_mars, params, ext, nids, cids, pids,
                                         gate, log_z, batch, (int)block_size, knt,
                                         (int)node_cbs, (int)gate_cbs, node_sh, gate_sh, (long)ext_base, step_sh); break;
        case 2: launch_cfg<256, 64, 4, 2>(node_mars, element_mars, params, ext, nids, cids, pids,
                                          gate, log_z, batch, (int)block_size, knt,
                                          (int)node_cbs, (int)gate_cbs, node_sh, gate_sh, (long)ext_base, step_sh); break;
        case 3: launch_cfg<128, 128, 2, 4>(node_mars, element_mars, params, ext, nids, cids, pids,
                                           gate, log_z, batch, (int)block_size, knt,
                                           (int)node_cbs, (int)gate_cbs, node_sh, gate_sh, (long)ext_base, step_sh); break;
        case 4: launch_cfg<64, 64, 4, 2>(node_mars, element_mars, params, ext, nids, cids, pids,
                                         gate, log_z, batch, (int)block_size, knt,
                                         (int)node_cbs, (int)gate_cbs, node_sh, gate_sh, (long)ext_base, step_sh); break;
        default: launch_cfg<128, 64, 4, 2>(node_mars, element_mars, params, ext, nids, cids, pids,
                                           gate, log_z, batch, (int)block_size, knt,
                                           (int)node_cbs, (int)gate_cbs, node_sh, gate_sh, (long)ext_base, step_sh); break;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_forward", &blockscale_forward,
          "CuTe/TMA forward for the per-block multiplicative gate (log N - log Z)");
    m.def("configs", &configs, "Tile shapes {BM, BN, WM, WN} per config id");
    m.def("fitting_configs", &fitting_configs,
          "Config ids whose tile divides this shape AND whose shared memory this device will grant");
}
