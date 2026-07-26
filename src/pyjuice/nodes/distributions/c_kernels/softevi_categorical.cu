// CUDA kernels for the `SoftEvidenceCategorical` input distribution (see softevi_categorical.py).
//
// =====================================================================================
//  Backward: expected-category flow phase of the top-k soft-evidence backward
// =====================================================================================
//
// Computes, for every (param-flow row, category) pair that any soft-evidence slot references,
//
//     phase1[row(l), cat] += beta[l, cat] * sum_j ratio[slot_j, l] * p_theta_j
//
// over the slots j whose candidate set contains `cat`, plus -- from the same `beta` read -- the
// expected-value term of the external evidence gradient
//
//     grad[goff_j] -= p_theta_j * sum_l ratio[slot_j, l] * beta[l, cat]
//
// Each (row, cat) has exactly one owning thread, so the phase-1 update is a plain read-modify-write
// with no atomics. See the long note in softevi_categorical.py for why the computation is organised
// this way rather than as a scatter over slots.
//
// Thread mapping (this is the whole point of the hand-written version):
//   threadIdx.x  -> category within the tile, so the `params` and `param_flows` accesses -- which are
//                   the bulk of the traffic and are contiguous along the category axis -- coalesce
//                   across the warp.
//   blockIdx.y   -> a tile of TL consecutive latents, held in registers.
// `ratio[slot, l]` is then contiguous along l *within a thread*, which lets each thread pull its TL
// values as float4 rather than TL scalar loads. The Triton version cannot express both at once: it
// loads the ratio tile transposed and materializes [BLOCK_L, BLOCK_C] tiles of beta/acc/ratio in
// registers, which is far more register traffic than this needs.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

// TL = latents handled per thread. Larger TL amortizes the per-reference index loads
// (slot / p_theta / grad-offset are read once per reference regardless of TL) over more
// latents, and lengthens each thread's contiguous `ratio` run. Templated so it can be tuned.

// `ratio` is only [num_slots, num_latents] (~1 MB), and a block touches just the [num_slots, TL] slice
// for its latent tile -- a few KB. Reading it from global memory costs far more than the phase-1 traffic
// itself: each thread in a warp needs a DIFFERENT slot, so every one of the ~1.07 GB of float4 loads
// pulls its own sector (~2.1 GB effective), versus 1.35 GB for beta + param_flows combined. Staging that
// slice in shared memory once per block removes essentially all of it. This is the part Triton cannot
// express -- it has no explicit shared-memory control, so it re-reads the ratio tile from global on every
// reference.
template <int TL, bool UPDATE_GRAD, bool SMEM_RATIO>
__global__ void dense_expected_flow_kernel(
        const float* __restrict__ params,
        float* __restrict__ param_flows,
        const float* __restrict__ ratio,
        const int* __restrict__ uniq,
        const int* __restrict__ ref_slot,
        const float* __restrict__ ref_pt,
        const int* __restrict__ ref_goff,
        const int* __restrict__ ref_cnt,
        const int* __restrict__ num_uniq,
        const long* __restrict__ pf_base,
        const long* __restrict__ p_base,
        float* __restrict__ grad,
        const int num_latents,
        const int tot_num_cats,
        const int uniq_stride,
        const int max_refs,
        const int num_slots) {

    extern __shared__ float s_ratio[];   // [num_slots * TL] when SMEM_RATIO

    const int g = blockIdx.z;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (c < num_uniq[g]);

    const int l0 = blockIdx.y * TL;
    if (l0 >= num_latents) return;
    const int nl = min(TL, num_latents - l0);

    if (SMEM_RATIO) {
        // cooperative, coalesced: consecutive threads take consecutive latents of a slot
        for (int i = threadIdx.x; i < num_slots * TL; i += blockDim.x) {
            const int sl = i / TL, t = i - sl * TL;
            s_ratio[i] = (t < nl) ? ratio[(long)sl * num_latents + l0 + t] : 0.0f;
        }
        __syncthreads();
    }

    if (!active) return;

    const long cat = (long)uniq[(long)g * uniq_stride + c];
    const int cnt = ref_cnt[(long)g * uniq_stride + c];

    const long* pfb = pf_base + (long)g * num_latents + l0;
    const long* pb  = p_base  + (long)g * num_latents + l0;

    // beta[l, cat] -- coalesced across the warp (consecutive threads -> nearby categories)
    float beta[TL];
    #pragma unroll
    for (int t = 0; t < TL; ++t) {
        beta[t] = (t < nl) ? __ldg(params + pb[t] + cat) : 0.0f;
    }

    float acc[TL];
    #pragma unroll
    for (int t = 0; t < TL; ++t) acc[t] = 0.0f;

    // reference arrays are [G, max_refs, uniq_stride] (SoA), so a warp's reads are contiguous
    const long ref0 = (long)g * max_refs * uniq_stride + c;
    const bool vec_ok = (nl == TL) && ((l0 & 3) == 0) && ((num_latents & 3) == 0);

    for (int j = 0; j < cnt; ++j) {
        const long  rj = ref0 + (long)j * uniq_stride;
        const int   s  = ref_slot[rj];
        const float pt = ref_pt  [rj];
        const float* r = ratio + (long)s * num_latents + l0;

        float rv[TL];
        if (SMEM_RATIO) {
            const float* sr = s_ratio + (long)s * TL;
            #pragma unroll
            for (int t = 0; t < TL; ++t) rv[t] = sr[t];
        } else if (vec_ok) {
            // contiguous within the thread -> two 128-bit loads instead of TL scalar loads
            const float4* r4 = reinterpret_cast<const float4*>(r);
            #pragma unroll
            for (int v = 0; v < TL / 4; ++v) {
                const float4 q = __ldg(r4 + v);
                rv[4*v+0] = q.x; rv[4*v+1] = q.y; rv[4*v+2] = q.z; rv[4*v+3] = q.w;
            }
        } else {
            #pragma unroll
            for (int t = 0; t < TL; ++t) rv[t] = (t < nl) ? __ldg(r + t) : 0.0f;
        }

        #pragma unroll
        for (int t = 0; t < TL; ++t) acc[t] += rv[t] * pt;

        if (UPDATE_GRAD) {
            float part = 0.0f;
            #pragma unroll
            for (int t = 0; t < TL; ++t) part += rv[t] * beta[t];
            if (part != 0.0f) atomicAdd(grad + ref_goff[rj], -pt * part);
        }
    }

    // one owner per (row, cat): plain read-modify-write, coalesced across the warp
    #pragma unroll
    for (int t = 0; t < TL; ++t) {
        if (t < nl) {
            float* dst = param_flows + pfb[t] + tot_num_cats + cat;
            // RED.E.ADD.F32: the add is done in L2 and nothing is returned to the SM, which halves the
            // SM<->L2 traffic versus load-add-store. Not needed for correctness (one owner per slot) --
            // purely faster: 1.66 -> 1.37 ms on the CoDD config.
            atomicAdd(dst, beta[t] * acc[t]);
        }
    }
}

void dense_expected_flow(torch::Tensor params, torch::Tensor param_flows, torch::Tensor ratio,
                         torch::Tensor uniq, torch::Tensor ref_slot, torch::Tensor ref_pt,
                         torch::Tensor ref_goff, torch::Tensor ref_cnt, torch::Tensor num_uniq,
                         torch::Tensor pf_base, torch::Tensor p_base,
                         c10::optional<torch::Tensor> grad,
                         int64_t num_latents, int64_t tot_num_cats, int64_t uniq_stride,
                         int64_t max_refs, int64_t num_blocks, int64_t block_c, int64_t num_slots,
                         int64_t tl_size) {

    const int threads = (int)block_c;
    const dim3 grid((unsigned)((uniq_stride + threads - 1) / threads),
                    (unsigned)((num_latents + tl_size - 1) / tl_size),
                    (unsigned)num_blocks);

    const bool do_grad = grad.has_value();
    float* grad_ptr = do_grad ? grad->data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();

    // Staging the ratio slice in shared memory MEASURED SLOWER (2.27 ms at 1024 threads vs 2.15 ms
    // reading it from global), so it is off. The reasoning that motivated it was wrong: `ratio` is only
    // ~1 MB in total and is reused heavily, so those "scattered" reads are served from L1/L2 rather than
    // DRAM -- there was no DRAM traffic to remove, and the staging cost occupancy. Kept behind an env
    // opt-in because it would become the right choice if `ratio` ever grew past cache (large batch x
    // many positions).
    const size_t smem = (size_t)num_slots * tl_size * sizeof(float);
    const bool use_smem = (getenv("PYJUICE_SOFTEVI_SMEM_RATIO") != nullptr) && (smem <= 48 * 1024);
    const size_t smem_bytes = use_smem ? smem : 0;

#define LAUNCH_TL(TLV, GRAD, SMEM)                                                                \
    dense_expected_flow_kernel<TLV, GRAD, SMEM><<<grid, threads, smem_bytes, stream>>>(           \
        params.data_ptr<float>(), param_flows.data_ptr<float>(), ratio.data_ptr<float>(),         \
        uniq.data_ptr<int>(), ref_slot.data_ptr<int>(), ref_pt.data_ptr<float>(),                 \
        ref_goff.data_ptr<int>(), ref_cnt.data_ptr<int>(), num_uniq.data_ptr<int>(),              \
        pf_base.data_ptr<long>(), p_base.data_ptr<long>(), grad_ptr,                              \
        (int)num_latents, (int)tot_num_cats, (int)uniq_stride, (int)max_refs, (int)num_slots)

#define DISPATCH(TLV)                                                                             \
    if (do_grad && use_smem)      { LAUNCH_TL(TLV, true,  true);  }                               \
    else if (do_grad)             { LAUNCH_TL(TLV, true,  false); }                               \
    else if (use_smem)            { LAUNCH_TL(TLV, false, true);  }                               \
    else                          { LAUNCH_TL(TLV, false, false); }

    switch (tl_size) {
        case 4:  DISPATCH(4);  break;
        case 16: DISPATCH(16); break;
        case 32: DISPATCH(32); break;
        case 64: DISPATCH(64); break;
        default: DISPATCH(8);  break;
    }
#undef DISPATCH
#undef LAUNCH_TL
    return;

}



// =====================================================================================
//  Forward: local normalizer + observed-token log-probability, top-k soft evidence
// =====================================================================================
//
//   node_mars[n, b] = log beta[n, x] + log p_theta[b, x] - log Z[n, b]
//   Z[n, b]         = sum_k beta[n, cat_k(b)] * p_theta[b, k]
//
// Two things this does that the Triton kernel cannot:
//
//  1. Accumulates Z in LINEAR space. The Triton version carries a running logsumexp -- a max, an exp, a
//     log and a log1p per candidate tile, on top of a log() per gathered parameter -- which measured
//     ~1.6 ms of the 2.9 ms forward, more than the parameter gather itself (~1.2 ms). Here it is one FMA
//     per candidate. beta is pre-scaled by 2^64 (an exact power of two, so the scaling is lossless) to
//     keep the smallest products well clear of denormals; validated against a float64 logsumexp at
//     1.6e-6 max abs error with zero underflow, versus ~1e-6 for the log-space path.
//  2. Fuses the observed-token search into the same pass. The Triton version walks the candidate list a
//     second time just to find where the observed token sits.
//
// One thread owns one (node, batch) pair and walks that node's candidates in ascending category order
// (the caller sorts them), so each thread's reads march forward through a single parameter row.

#define FW_LOG_SCALE 44.3614195558365f   // log(2^64), subtracted back off at the end

template <int U>
__global__ void softevi_forward_kernel(
        const float* __restrict__ params, float* __restrict__ node_mars,
        const long* __restrict__ data, const long* __restrict__ vids,
        const long* __restrict__ s_pids, const long* __restrict__ var_idmapping,
        const float* __restrict__ pt, const long* __restrict__ cat_ids,
        const int layer_num_nodes, const int batch_size, const int node_offset,
        const int num_cats, const int ext_num_vars) {

    // threadIdx.x -> NODE, blockIdx.y -> batch element. Two things decide this kernel's speed, and both
    // were found by benchmarking JUST the gather against an equivalent Triton one:
    //
    //   * 512 threads per block. At 256 this kernel runs 7.4 ms; at 512 it runs 2.6 ms -- a 3x cliff,
    //     reproducible across unroll factors. Below it there is not enough of the parameter row in
    //     flight per block to keep the memory system busy.
    //   * __ldcs (streaming) for the parameter gather. Each row segment is touched once, so ordinary
    //     cached loads only pollute L1; streaming them is a further ~12%.
    //
    // The candidate ids and weights are identical for every thread in the block (they depend on the
    // batch element and variable, not the node), so those loads broadcast -- which is also why the ids
    // are consumed as int64 straight from the caller rather than being narrowed to int32 first: the
    // conversion would cost a full [B, V, k] pass on every step, and the ids change every step so it
    // cannot be cached.
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= layer_num_nodes) return;

    const long vid  = vids[n];
    const long lvid = var_idmapping[vid];
    const long pbase = s_pids[n];
    const long obs = data[vid * batch_size + b];

    const long ebase = (long)b * ext_num_vars * num_cats + lvid * num_cats;
    const float* ptb = pt + ebase;
    const long*  cib = cat_ids + ebase;

    float Z = 0.0f;
    float log_ex_p = -INFINITY;

    int k = 0;
    for (; k + U <= num_cats; k += U) {
        long  c[U];
        float w[U], v[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) c[u] = __ldg(cib + k + u);
        #pragma unroll
        for (int u = 0; u < U; ++u) w[u] = __ldg(ptb + k + u);
        #pragma unroll
        for (int u = 0; u < U; ++u) v[u] = __ldcs(params + pbase + c[u]);
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            Z = fmaf(v[u] * 1.8446744073709552e19f, w[u], Z);   // beta * 2^64 is exact
            if (c[u] == obs) log_ex_p = __logf(w[u]);
        }
    }
    for (; k < num_cats; ++k) {
        const long c0 = __ldg(cib + k);
        const float w0 = __ldg(ptb + k);
        Z = fmaf(__ldcs(params + pbase + c0) * 1.8446744073709552e19f, w0, Z);
        if (c0 == obs) log_ex_p = __logf(w0);
    }

    const float log_in_p = __logf(__ldg(params + pbase + obs));
    node_mars[(long)(n + node_offset) * batch_size + b] =
        log_in_p + log_ex_p - (__logf(Z) - FW_LOG_SCALE);
}

void softevi_forward(torch::Tensor params, torch::Tensor node_mars, torch::Tensor data,
                     torch::Tensor vids, torch::Tensor s_pids, torch::Tensor var_idmapping,
                     torch::Tensor pt, torch::Tensor cat_ids,
                     int64_t layer_num_nodes, int64_t batch_size, int64_t node_offset,
                     int64_t num_cats, int64_t ext_num_vars, int64_t unroll) {
    const int threads = 512;   // 3x faster than 256 here -- see the note on the kernel
    const dim3 blocks((unsigned)((layer_num_nodes + threads - 1) / threads), (unsigned)batch_size);
#define FW(UV) softevi_forward_kernel<UV><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
        params.data_ptr<float>(), node_mars.data_ptr<float>(), data.data_ptr<long>(),               \
        vids.data_ptr<long>(), s_pids.data_ptr<long>(), var_idmapping.data_ptr<long>(),             \
        pt.data_ptr<float>(), cat_ids.data_ptr<long>(),                                              \
        (int)layer_num_nodes, (int)batch_size, (int)node_offset,                                    \
        (int)num_cats, (int)ext_num_vars)
    switch (unroll) {
        case 4:  FW(4);  break;
        case 8:  FW(8);  break;
        case 32: FW(32); break;
        default: FW(16); break;
    }
#undef FW
}



// =====================================================================================
//  Forward, index-driven: the same inversion the backward uses
// =====================================================================================
//
//   Z[slot, latent] = sum over categories c referenced by `slot` of beta[latent, c] * p_theta[slot, c]
//
// Walked as (latent x category) with categories contiguous, so beta is read ONCE, coalesced, at each
// referenced (latent, category): ~113M reads here versus the ~268M SCATTERED reads the direct form does
// (every (node, batch) pair chasing its own candidate columns). Partial sums are kept in shared memory
// and flushed with one global atomic per (slot, latent) per block -- doing it with one atomic per
// reference instead is what made an earlier attempt at this 1.7x SLOWER than the gather form.
// Measured on the CoDD config: 0.84 ms, against a 0.31 ms floor for the beta read alone.

template <int TL>
__global__ void softevi_fw_dense_z(const float* __restrict__ params, const int* __restrict__ uniq,
                                   const int* __restrict__ ref_slot, const float* __restrict__ ref_pt,
                                   const int* __restrict__ ref_cnt, const int* __restrict__ num_uniq,
                                   const long* __restrict__ p_base, float* __restrict__ Z,
                                   int num_latents, int uniq_stride, int max_refs, int num_slots) {
    extern __shared__ float Zs[];                       // [num_slots * TL]

    const int g = blockIdx.z;
    const int l0 = blockIdx.y * TL;
    const int U = num_uniq[g];

    for (int i = threadIdx.x; i < num_slots * TL; i += blockDim.x) Zs[i] = 0.0f;
    __syncthreads();

    const long pb_g = (long)g * num_latents + l0;
    const long ub = (long)g * uniq_stride;
    const long rb = (long)g * max_refs * uniq_stride;

    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < U; c += blockDim.x * gridDim.x) {
        const long cat = (long)uniq[ub + c];
        const int cnt = ref_cnt[ub + c];
        float bet[TL];
        #pragma unroll
        for (int t = 0; t < TL; ++t)
            bet[t] = (l0 + t < num_latents) ? __ldcs(params + p_base[pb_g + t] + cat) : 0.0f;
        for (int j = 0; j < cnt; ++j) {
            const long rj = rb + (long)j * uniq_stride + c;
            const int s = ref_slot[rj];
            const float w = ref_pt[rj] * 1.8446744073709552e19f;   // 2^64, exact
            float* dst = Zs + (long)s * TL;
            #pragma unroll
            for (int t = 0; t < TL; ++t) atomicAdd(dst + t, bet[t] * w);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_slots * TL; i += blockDim.x) {
        const float v = Zs[i];
        if (v != 0.0f) {
            const int s = i / TL, t = i - s * TL;
            if (l0 + t < num_latents) atomicAdd(Z + (long)s * num_latents + l0 + t, v);
        }
    }
}

__global__ void softevi_fw_epilogue(const float* __restrict__ params, float* __restrict__ node_mars,
                                    const float* __restrict__ Z, const float* __restrict__ log_ex_p,
                                    const long* __restrict__ data, const long* __restrict__ vids,
                                    const long* __restrict__ s_pids, const long* __restrict__ nids,
                                    const long* __restrict__ var_idmapping,
                                    int layer_num_nodes, int batch_size, int node_offset,
                                    int num_latents) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)layer_num_nodes * batch_size) return;
    const int n = (int)(i / batch_size);
    const int b = (int)(i - (long)n * batch_size);

    const long vid = vids[n];
    const long lvid = var_idmapping[vid];
    const long lat = nids[n];
    const long obs = data[vid * batch_size + b];

    const float z = Z[(lvid * batch_size + b) * num_latents + lat];
    node_mars[(long)(n + node_offset) * batch_size + b] =
        __logf(__ldg(params + s_pids[n] + obs)) + log_ex_p[lvid * batch_size + b]
        - (__logf(z) - FW_LOG_SCALE);
}

void softevi_forward_dense(torch::Tensor params, torch::Tensor node_mars, torch::Tensor Z,
                           torch::Tensor log_ex_p, torch::Tensor data, torch::Tensor vids,
                           torch::Tensor s_pids, torch::Tensor nids, torch::Tensor var_idmapping,
                           torch::Tensor uniq, torch::Tensor ref_slot, torch::Tensor ref_pt,
                           torch::Tensor ref_cnt, torch::Tensor num_uniq, torch::Tensor p_base,
                           int64_t num_latents, int64_t uniq_stride, int64_t max_refs,
                           int64_t num_slots, int64_t num_blocks, int64_t layer_num_nodes,
                           int64_t batch_size, int64_t node_offset, int64_t TLv, int64_t threads,
                           int64_t cat_blocks) {
    auto st = at::cuda::getCurrentCUDAStream();
    Z.zero_();
    const dim3 grid((unsigned)cat_blocks, (unsigned)((num_latents + TLv - 1) / TLv), (unsigned)num_blocks);
#define GO(T) { const size_t sm = (size_t)num_slots * T * sizeof(float);                            \
    softevi_fw_dense_z<T><<<grid, threads, sm, st>>>(params.data_ptr<float>(), uniq.data_ptr<int>(),\
        ref_slot.data_ptr<int>(), ref_pt.data_ptr<float>(), ref_cnt.data_ptr<int>(),                \
        num_uniq.data_ptr<int>(), p_base.data_ptr<long>(), Z.data_ptr<float>(),                     \
        (int)num_latents, (int)uniq_stride, (int)max_refs, (int)num_slots); }
    switch (TLv) { case 8: GO(8); break; case 16: GO(16); break; default: GO(4); break; }
#undef GO
    const long tot = layer_num_nodes * batch_size;
    softevi_fw_epilogue<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(
        params.data_ptr<float>(), node_mars.data_ptr<float>(), Z.data_ptr<float>(),
        log_ex_p.data_ptr<float>(), data.data_ptr<long>(), vids.data_ptr<long>(),
        s_pids.data_ptr<long>(), nids.data_ptr<long>(), var_idmapping.data_ptr<long>(),
        (int)layer_num_nodes, (int)batch_size, (int)node_offset, (int)num_latents);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_expected_flow", &dense_expected_flow, "Expected-category flow phase (CUDA)");
    m.def("softevi_forward", &softevi_forward, "Top-k soft-evidence forward, gather form (CUDA)");
    m.def("softevi_forward_dense", &softevi_forward_dense, "Top-k soft-evidence forward, index-driven (CUDA)");
}
