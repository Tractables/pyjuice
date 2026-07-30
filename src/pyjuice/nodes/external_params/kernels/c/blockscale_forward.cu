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
#include <vector>
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

// Add each staged log-gate onto every element of `element_mars` it covers, and fold the gates' own max
// into `sMz`. `GC` -- the gate's child-block width, clamped to the edge tile -- is a template parameter
// so `BK / GC`, `e / GC` and the trip counts are all compile-time; see the call site.
template <int BN, int NTH, int GC>
__device__ __forceinline__ void fold_gates(float* __restrict__ sEm, const float* __restrict__ sPh,
                                           float* __restrict__ sMz, int tid) {
    constexpr int NGT = BK / GC;

    CUTE_UNROLL
    for (int i = tid * 4; i < BK * BN; i += NTH * 4) {
        const int e = i / BN, b = i % BN;
        float4 v = *(const float4*)&sEm[i];
        const float4 g = *(const float4*)&sPh[(e / GC) * BN + b];
        v.x += g.x; v.y += g.y; v.z += g.z; v.w += g.w;
        *(float4*)&sEm[i] = v;
    }

    // The stabilizer for `Z`: the max log-gate over all of the row's gates. Free here -- the gates are
    // in shared already and the block visits every edge tile of its row. In its own kernel it cost 27 us
    // flat, since a standalone pass re-reads `phi` from DRAM and chases `gate[j]` before it can issue
    // each load, with too few warps in flight to hide either.
    for (int b = tid; b < BN; b += NTH) {
        float gm = -INFINITY;
        CUTE_UNROLL
        for (int gi = 0; gi < NGT; gi++) gm = fmaxf(gm, sPh[gi * BN + b]);
        sMz[b] = fmaxf(sMz[b], gm);
    }
}


// Fallback for a gate width outside the specialized set; correct, just not unrolled.
template <int BN, int NTH>
__device__ __forceinline__ void fold_gates_rt(float* __restrict__ sEm, const float* __restrict__ sPh,
                                              float* __restrict__ sMz, int tid, int gc) {
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
        sMz[b] = fmaxf(sMz[b], gm);
    }
}


template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32) blockscale_tlmm_kernel(
        float* __restrict__ node_mars, const float* __restrict__ mp,
        const float* __restrict__ ext, const long* __restrict__ nids,
        const long* __restrict__ ebase, const long* __restrict__ pbase,
        const long* __restrict__ gate, float* __restrict__ mz_out,
        int batch, int block_size, int knt, int gate_stride,
        int node_cbs, int gate_cbs, long ext_base, int pid_m_offset,
        const __grid_constant__ CUtensorMap desc) {
    constexpr int NTH = WM * WN * 32;

    int mtiles = block_size / BM;
    int pid_b = blockIdx.x, pid_m = blockIdx.y + pid_m_offset;
    int nblock = pid_m / mtiles, tile_id = pid_m % mtiles;
    int b0 = pid_b * BN;
    long off_nid = nids[nblock];
    const long* eb = ebase + (long)nblock * knt;
    const long* pb = pbase + (long)nblock * knt;
    const long* gt = gate + (long)nblock * gate_stride;
    int tid = threadIdx.x;

    auto swz = composition(Swizzle<3, 3, 3>{},
                           Layout<Shape<Shape<_8, _8>, _8>, Stride<Stride<_1, _64>, _8>>{});
    auto sAl = tile_to_shape(swz, make_shape(Int<BM>{}, Int<BK>{}));
    auto sBl = tile_to_shape(swz, make_shape(Int<BN>{}, Int<BK>{}));
    bfloat16_t* pA = (bfloat16_t*)smem_raw;
    bfloat16_t* pBs = pA + cosize(sAl);
    float* sEm = (float*)(pBs + cosize(sBl));
    float* sMx = sEm + BK * BN;
    float* sMz = sMx + BN;                       // [BN] running max log-gate
    uint64_t* bar = (uint64_t*)(sMz + BN + 4);
    float* sPh = (float*)(bar + 4);              // [ngt, BN] per tile, or [knt * ngt, BN] preloaded

    Tensor sAt = make_tensor(make_smem_ptr(pA), sAl);
    Tensor sBt = make_tensor(make_smem_ptr(pBs), sBl);
    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                  Layout<Shape<Int<WM>, Int<WN>, _1>>{});
    ThrMMA thr = mma.get_thread_slice(tid);
    Tensor tCrA = thr.partition_fragment_A(sAt);
    Tensor tCrB = thr.partition_fragment_B(sBt);
    auto s2rA = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto s2rB = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bfloat16_t>{}, mma);
    Tensor tXsA = s2rA.get_thread_slice(tid).partition_S(sAt);
    Tensor tXrA = s2rA.get_thread_slice(tid).retile_D(tCrA);
    Tensor tXsB = s2rB.get_thread_slice(tid).partition_S(sBt);
    Tensor tXrB = s2rB.get_thread_slice(tid).retile_D(tCrB);
    Tensor cC = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    Tensor tCcC = thr.partition_C(cC);
    Tensor tCrS = thr.partition_fragment_C(cC);
    Tensor tCrL = thr.partition_fragment_C(cC);
    Tensor tCrM = thr.partition_fragment_C(cC);
    clear(tCrL);
    CUTE_UNROLL
    for (int i = 0; i < size(tCrM); i++) tCrM(i) = -INFINITY;

    const int gcbs_eff = (gate_cbs < BK) ? gate_cbs : BK;
    const int ngt = BK / gcbs_eff;

    if (tid == 0) mbar_init(bar, 1);
    for (int b = tid; b < BN; b += NTH) sMz[b] = -INFINITY;
    __syncthreads();
    int phase = 0;

    for (int kt = 0; kt < knt; kt++) {
        long pc = pb[kt] + (long)tile_id * BM;
        if (tid == 0) { mbar_expect(bar, BK * BN * 4); tma_load_2d(sEm, &desc, b0, (int)eb[kt], bar); }

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
            const int j = ge0 / node_cbs;
            const int d = (ge0 % node_cbs) / gate_cbs;

            const long gb = (j < gate_stride) ? gt[j] : -1;
            sPh[i] = (gb >= 0) ? ext[(gb + ext_base + d) * (long)batch + b0 + b] : -INFINITY;
        }

        mbar_wait(bar, phase); phase ^= 1;
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
            case  4: fold_gates<BN, NTH,  4>(sEm, sPh, sMz, tid); break;
            case  8: fold_gates<BN, NTH,  8>(sEm, sPh, sMz, tid); break;
            case 16: fold_gates<BN, NTH, 16>(sEm, sPh, sMz, tid); break;
            case 32: fold_gates<BN, NTH, 32>(sEm, sPh, sMz, tid); break;
            case 64: fold_gates<BN, NTH, 64>(sEm, sPh, sMz, tid); break;
            default: fold_gates_rt<BN, NTH>(sEm, sPh, sMz, tid, gcbs_eff); break;
        }
        __syncthreads();

        for (int b = tid; b < BN; b += NTH) {
            float mx = -INFINITY;
            for (int e = 0; e < BK; e++) mx = fmaxf(mx, sEm[e * BN + b]);
            sMx[b] = mx;
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
        for (int i = tid; i < (BM * BK) / 8; i += NTH) {
            int e = i / (BM / 8), mm = (i % (BM / 8)) * 8;
            const float* g = &mp[pc + (long)e * block_size + mm];
            float4 a = *(const float4*)g, b = *(const float4*)(g + 4);
            bfloat16_t r[8];
            r[0] = static_cast<bfloat16_t>(a.x); r[1] = static_cast<bfloat16_t>(a.y);
            r[2] = static_cast<bfloat16_t>(a.z); r[3] = static_cast<bfloat16_t>(a.w);
            r[4] = static_cast<bfloat16_t>(b.x); r[5] = static_cast<bfloat16_t>(b.y);
            r[6] = static_cast<bfloat16_t>(b.z); r[7] = static_cast<bfloat16_t>(b.w);
            *(float4*)&sAt(mm, e) = *(const float4*)r;
        }
        __syncthreads();

        copy(s2rA, tXsA, tXrA); copy(s2rB, tXsB, tXrB);
        clear(tCrS); cute::gemm(mma, tCrA, tCrB, tCrS);

        CUTE_UNROLL
        for (int i = 0; i < size(tCrS); i++) {
            int b = get<1>(tCcC(i));
            float mxk = sMx[b];
            float partial = tCrS(i), Mo = tCrM(i), Lo = tCrL(i);
            float nM = fmaxf(Mo, mxk);
            if (nM != -INFINITY) {
                float pa = (Mo == -INFINITY) ? 0.f : Lo * __expf(Mo - nM);
                float pcc = (mxk == -INFINITY) ? 0.f : partial * __expf(mxk - nM);
                tCrL(i) = pa + pcc; tCrM(i) = nM;
            }
        }
        __syncthreads();
    }

    // One m-tile publishes the stabilizer: every tile of a row computed it from the same gates, so the
    // others would only rewrite the same bytes.
    if (tile_id == 0)
        for (int b = tid; b < BN; b += NTH) mz_out[(long)nblock * batch + b0 + b] = sMz[b];

    // Writes log N; `blockscale_normalize` turns it into log N - log Z.
    CUTE_UNROLL
    for (int i = 0; i < size(tCrL); i++) {
        int m = get<0>(tCcC(i)), b = get<1>(tCcC(i));
        if (m < BM && b < BN)
            node_mars[(off_nid + (long)tile_id * BM + m) * (long)batch + b0 + b] =
                (tCrM(i) == -INFINITY) ? -INFINITY : (logf(tCrL(i)) + tCrM(i));
    }
}


// ============================================================ sigma: per-gate parameter mass

// `sigma[row, j, d, m] = sum_{c in child gate d of edge block j} theta[row's node m, c]`.
// Batch-independent, so it is the whole reason `Z` needs no second MMA. Recomputed each forward: it is
// one pass over the parameters, which removes any question of staleness after an EM update.
__global__ void blockscale_sigma_kernel(
        const float* __restrict__ mp, const long* __restrict__ pids, float* __restrict__ sigma,
        int rows, int num_edges, int block_size, int node_cbs, int gate_cbs, int n_child_gates) {

    const int n_eblks = num_edges / node_cbs;
    const long total = (long)rows * n_eblks * n_child_gates * block_size;

    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int m = i % block_size;
        const int d = (i / block_size) % n_child_gates;
        const int j = (i / ((long)block_size * n_child_gates)) % n_eblks;
        const int row = i / ((long)block_size * n_child_gates * n_eblks);

        const long e0 = (long)row * num_edges + (long)j * node_cbs + (long)d * gate_cbs;

        float s = 0.0f;
        for (int c = 0; c < gate_cbs; ++c) s += mp[pids[e0 + c] + m];

        sigma[i] = s;
    }
}


// ============================================================ normalize: Z, and log N -> log N - log Z

// `mz[row, b] = max_g log phi[g, b]`, the stabilizer for `Z`, comes out of the main kernel -- see the
// max loop there.

template <int TM, int TB, int GT, int RM, int RB>
__global__ void __launch_bounds__((TM / RM) * (TB / RB)) blockscale_normalize_kernel(
        float* __restrict__ node_mars, const float* __restrict__ ext,
        const float* __restrict__ sigma, const long* __restrict__ nids,
        const long* __restrict__ gate, const float* __restrict__ mz_in,
        float* __restrict__ log_z_out,
        int batch, int block_size, int n_gates, int n_child_gates, int gate_stride, long ext_base) {

    // `Z[m,b] = sum_g sigma[g,m] * phi[g,b]` is a GEMM: [block_size x n_gates] . [n_gates x batch].
    // Written as a triple loop it re-read every gate's `sigma` and `phi` from global for each output --
    // ~100M load instructions for 2.7 MB of distinct data, and 20x its own IO bound.
    //
    // REGISTER-TILED, `RM x RB` outputs per thread. One output per thread instead made every thread walk
    // all `n_gates` gates as a serial dependent FMA chain, two shared loads per FMA, and needed `TM * TB`
    // threads -- 1024-thread blocks, of which there were barely one wave. Here each staged pair of
    // shared values feeds `RM * RB` FMAs, so the same arithmetic costs 8x fewer shared accesses and the
    // block is small enough to fill the machine.
    constexpr int NTH = (TM / RM) * (TB / RB);

    __shared__ float s_sig[GT][TM];
    __shared__ float s_phe[GT][TB];
    __shared__ float s_mz[TB];

    const int row = blockIdx.x;
    const int m0 = blockIdx.y * TM;
    const int b0 = blockIdx.z * TB;

    const int tid = threadIdx.x;

    // The BATCH index runs along `threadIdx.x` so the epilogue's stores stay coalesced: `node_mars` is
    // batch-innermost, and indexing the other way round would give adjacent threads addresses `batch`
    // floats apart.
    const int bt = (tid % (TB / RB)) * RB;
    const int mt = (tid / (TB / RB)) * RM;

    const long* gt = gate + (long)row * gate_stride;

    for (int i = tid; i < TB; i += NTH)
        s_mz[i] = (b0 + i < batch) ? mz_in[(long)row * batch + b0 + i] : -INFINITY;
    __syncthreads();

    float acc[RM][RB];
    #pragma unroll
    for (int i = 0; i < RM; ++i)
        #pragma unroll
        for (int j = 0; j < RB; ++j) acc[i][j] = 0.0f;

    for (int g0 = 0; g0 < n_gates; g0 += GT) {
        const int ng = min(GT, n_gates - g0);

        // float4: `sigma` is contiguous along the node axis and `phi` along the batch axis, and both
        // tiles are multiples of 4, so each staging load moves 16 bytes instead of 4.
        for (int i = tid; i < ng * (TM / 4); i += NTH) {
            const int gg = i / (TM / 4), mm = (i % (TM / 4)) * 4;
            const long base = ((long)row * n_gates + g0 + gg) * block_size + m0 + mm;

            if (m0 + mm + 3 < block_size) {
                *(float4*)&s_sig[gg][mm] = *(const float4*)&sigma[base];
            } else {
                for (int c = 0; c < 4; ++c)
                    s_sig[gg][mm + c] = (m0 + mm + c < block_size) ? sigma[base + c] : 0.0f;
            }
        }
        for (int i = tid; i < ng * (TB / 4); i += NTH) {
            const int gg = i / (TB / 4), bb = (i % (TB / 4)) * 4;
            const int g = g0 + gg;
            const long gb = gt[g / n_child_gates];

            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (gb >= 0 && b0 + bb + 3 < batch) {
                *(float4*)v = *(const float4*)&ext[(gb + ext_base + g % n_child_gates) * (long)batch
                                                   + b0 + bb];
            } else if (gb >= 0) {
                for (int c = 0; c < 4; ++c)
                    if (b0 + bb + c < batch)
                        v[c] = ext[(gb + ext_base + g % n_child_gates) * (long)batch + b0 + bb + c];
            }

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float mzc = s_mz[bb + c];
                s_phe[gg][bb + c] = (gb < 0 || b0 + bb + c >= batch || mzc == -INFINITY)
                                    ? 0.0f : __expf(v[c] - mzc);
            }
        }
        __syncthreads();

        for (int gg = 0; gg < ng; ++gg) {
            float a[RM], p[RB];
            #pragma unroll
            for (int i = 0; i < RM; ++i) a[i] = s_sig[gg][mt + i];
            #pragma unroll
            for (int j = 0; j < RB; ++j) p[j] = s_phe[gg][bt + j];
            #pragma unroll
            for (int i = 0; i < RM; ++i)
                #pragma unroll
                for (int j = 0; j < RB; ++j) acc[i][j] = fmaf(a[i], p[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        const int m = m0 + mt + i;
        if (m >= block_size) break;

        #pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int b = b0 + bt + j;
            if (b >= batch) break;

            const float mz = s_mz[bt + j];
            const float log_z = (acc[i][j] <= 0.0f || mz == -INFINITY) ? -INFINITY
                                                                       : mz + logf(acc[i][j]);
            const long o = (nids[row] + m) * (long)batch + b;

            node_mars[o] = (node_mars[o] == -INFINITY || log_z == -INFINITY)
                           ? -INFINITY : node_mars[o] - log_z;

            if (log_z_out != nullptr)
                log_z_out[((long)row * block_size + m) * (long)batch + b] = log_z;
        }
    }
}


// ============================================================ launchers

template <int BM, int BN, int WM, int WN>
static void launch_cfg(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                       torch::Tensor ext, torch::Tensor nids, torch::Tensor ebase,
                       torch::Tensor pbase, torch::Tensor gate, torch::Tensor gmax,
                       int batch, int block_size, int knt, int node_cbs, int gate_cbs,
                       long ext_base) {
    // `gcbs_eff` mirrors the kernel: a gate may be wider than one edge tile
    constexpr int NTH = WM * WN * 32;
    int n_edge_rows = element_mars.size(0);
    void* base = (void*)element_mars.data_ptr<float>();

    static CUtensorMap desc;
    static void* desc_ptr = nullptr;
    static int desc_rows = 0, desc_batch = 0;
    if (base != desc_ptr || n_edge_rows != desc_rows || batch != desc_batch) {
        cuuint64_t gdim[2] = {(cuuint64_t)batch, (cuuint64_t)n_edge_rows};
        cuuint64_t gstride[1] = {(cuuint64_t)batch * 4};
        cuuint32_t bdim[2] = {(cuuint32_t)BN, (cuuint32_t)BK};
        cuuint32_t estride[2] = {1, 1};
        CUresult r = cuTensorMapEncodeTiled(
            &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, gdim, gstride, bdim, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
        desc_ptr = base; desc_rows = n_edge_rows; desc_batch = batch;
    }

    const int gcbs_eff = (gate_cbs < BK) ? gate_cbs : BK;
    int smem = BM * BK * 2 + BN * BK * 2 + BK * BN * 4 + BN * 4 + BN * 4 + 64
               + (BK / gcbs_eff) * BN * 4;      // operands, the two maxes, the staged gates
    cudaFuncSetAttribute(blockscale_tlmm_kernel<BM, BN, WM, WN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    int total_m = nids.size(0) * (block_size / BM);
    const int MAX_Y = 65535;
    for (int off = 0; off < total_m; off += MAX_Y) {
        int chunk = (total_m - off < MAX_Y) ? (total_m - off) : MAX_Y;
        dim3 grid(batch / BN, chunk);
        blockscale_tlmm_kernel<BM, BN, WM, WN><<<grid, NTH, smem, c10::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), params.data_ptr<float>(), ext.data_ptr<float>(),
            nids.data_ptr<long>(), ebase.data_ptr<long>(), pbase.data_ptr<long>(),
            gate.data_ptr<long>(), gmax.data_ptr<float>(),
            batch, block_size, knt, (int)gate.size(1),
            node_cbs, gate_cbs, ext_base, off, desc);
    }
}


std::vector<std::vector<int>> configs() {
    return {{128, 64, 2, 2}, {64, 64, 2, 2}, {256, 64, 4, 2}, {128, 128, 2, 4}};
}


void blockscale_forward(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                        torch::Tensor ext, torch::Tensor nids, torch::Tensor ebase,
                        torch::Tensor pbase, torch::Tensor pids, torch::Tensor gate,
                        torch::Tensor sigma, torch::Tensor gmax, torch::Tensor log_z,
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

    // sigma: one pass over the parameters, then Z is a small batch-independent contraction
    {
        const long total = (long)rows * n_eblks * n_child_gates * block_size;
        const int threads = 256;
        const int blocks = (int)std::min<long>(1024, (total + threads - 1) / threads);
        blockscale_sigma_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            params.data_ptr<float>(), pids.data_ptr<long>(), sigma.data_ptr<float>(),
            rows, (int)num_edges, (int)block_size, (int)node_cbs, (int)gate_cbs, n_child_gates);
    }

    switch (cfg) {
        case 0: launch_cfg<128, 64, 2, 2>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                          gate, gmax, batch, (int)block_size, knt, (int)node_cbs,
                                          (int)gate_cbs, (long)ext_base); break;
        case 1: launch_cfg<64, 64, 2, 2>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                         gate, gmax, batch, (int)block_size, knt, (int)node_cbs,
                                         (int)gate_cbs, (long)ext_base); break;
        case 2: launch_cfg<256, 64, 4, 2>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                          gate, gmax, batch, (int)block_size, knt, (int)node_cbs,
                                          (int)gate_cbs, (long)ext_base); break;
        default: launch_cfg<128, 128, 2, 4>(node_mars, element_mars, params, ext, nids, ebase, pbase,
                                            gate, gmax, batch, (int)block_size, knt, (int)node_cbs,
                                            (int)gate_cbs, (long)ext_base); break;
    }

    const int n_gates = n_eblks * n_child_gates;
    {
        constexpr int TM = 64, TB = 64, GT = 32, RM = 4, RB = 4;
        constexpr int NTH = (TM / RM) * (TB / RB);
        dim3 g(rows, (int)((block_size + TM - 1) / TM), (batch + TB - 1) / TB);
        blockscale_normalize_kernel<TM, TB, GT, RM, RB><<<g, NTH, 0,
                                                          c10::cuda::getCurrentCUDAStream()>>>(
            node_mars.data_ptr<float>(), ext.data_ptr<float>(), sigma.data_ptr<float>(),
            nids.data_ptr<long>(), gate.data_ptr<long>(), gmax.data_ptr<float>(),
            log_z.numel() ? log_z.data_ptr<float>() : nullptr,
            batch, (int)block_size, n_gates, n_child_gates, (int)gate.size(1), (long)ext_base);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_forward", &blockscale_forward,
          "CuTe/TMA forward for the per-block multiplicative gate (log N - log Z)");
    m.def("configs", &configs, "Tile shapes {BM, BN, WM, WN} per config id");
}
