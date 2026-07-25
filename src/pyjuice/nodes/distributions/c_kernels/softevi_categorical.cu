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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_expected_flow", &dense_expected_flow, "Expected-category flow phase (CUDA)");
}
