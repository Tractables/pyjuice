// CUDA kernels for `SoftEvidenceCategorical`'s top-k forward in ROW form (see softevi/forward.py).
//
//   node_mars[n, b] = log beta[n, x] + log p_theta[b, x] - log Z[n, b]
//   Z[slot, l]      = sum_k p_theta[slot, k] * beta[l, cat_k(slot)]          (slot = (variable, batch))
//
// `softevi_categorical.cu`'s index-driven forward walks Z as (latent x category) and accumulates every reference
// into a [slots, TL] shared-memory tile with an atomic -- slots x latents atomics in all, and a tile that stops
// fitting past ~6300 slots (where the forward fell back to the Triton gather). Here Z is a row-wise sparse-dense
// product instead:
//
//   T[u, l] = beta[l, cat(u)] * 2^64        the referenced (block, category) keys' parameters, TRANSPOSED: a key's
//                                           latents contiguous (`softevi_rows_table`)
//   Z[slot, l] = sum_k p_theta[slot, k] * T[key(slot, k), l]                (`softevi_rows_z`)
//
// A block owns one slot and a run of latents, each thread four of them: every candidate is one coalesced float4 row
// read of T (broadcast index and weight), one FMA a latent, no atomics, and the sum runs in candidate order (the
// same Z every launch). Common tokens -- in most candidate lists -- make T's rows L2-resident; measured on real
// LLaDA top-k 1024 evidence it streams them at the L2 rate (~7 TB/s here). beta is pre-scaled by 2^64 (exact) as
// in the index-driven form, to keep the smallest products clear of denormals.
//
// The latents can be taken in chunks (`l0`, `lc`): T then holds `lc` of them a key, bounding its memory.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#define ROWS_SCALE 1.8446744073709552e19f   // 2^64
#define ROWS_LOG_SCALE 44.3614195558365f    // log(2^64)

// T[u, j] = beta[p_base[g(u), l0 + j] + cat(u)] * 2^64 for j < lc, through a 32 x 32 shared tile: the reads
// coalesce along the keys (ascending categories within a block), the writes along the latents.
__global__ void softevi_rows_table_kernel(const float* __restrict__ params, const long* __restrict__ keys,
                                          const long* __restrict__ p_base, float* __restrict__ T,
                                          int num_keys, int num_latents, long num_cats, int l0, int lc) {
    __shared__ float tile[32][33];
    const int u0 = blockIdx.x * 32, j0 = blockIdx.y * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;            // 32 x 8
    const int u = u0 + tx;
    long cat = 0, row = 0;
    if (u < num_keys) {
        const long key = keys[u];
        row = (key / num_cats) * num_latents;                // the key's parameter block's first row
        cat = key - (key / num_cats) * num_cats;
    }
    for (int i = ty; i < 32; i += 8) {
        const int j = j0 + i;
        tile[i][tx] = (u < num_keys && j < lc) ? __ldcs(params + p_base[row + l0 + j] + cat) * ROWS_SCALE : 0.0f;
    }
    __syncthreads();
    for (int i = ty; i < 32; i += 8) {
        const int uu = u0 + i, j = j0 + tx;
        if (uu < num_keys && j < lc) T[(long)uu * lc + j] = tile[tx][i];
    }
}

// Z[v * B + b, l0 + 4t .. +3] for the block's slot (b, v): its K candidates' rows of T, in candidate order.
template <int UN>
__global__ void softevi_rows_z_kernel(const float* __restrict__ T, const int* __restrict__ elem_key,
                                      const float* __restrict__ pt, float* __restrict__ Z,
                                      int B, int V, int K, int num_latents, int l0, int lc) {
    const int bv = blockIdx.x;                              // b * V + v: the elements' order
    const int j = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (j >= lc) return;
    const int b = bv / V, v = bv - (bv / V) * V;
    const int* ek = elem_key + (long)bv * K;
    const float* pw = pt + (long)bv * K;
    const float* Tj = T + j;
    // UN partial sums, candidate k into sum k % UN, added pairwise at the end: one running sum over all K candidates
    // measured 3x the error against float64 of the index-driven form's (which adds in shared-memory partials)
    float4 acc[UN];
    #pragma unroll
    for (int i = 0; i < UN; ++i) acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    int k = 0;
    for (; k + UN <= K; k += UN) {
        int r[UN]; float w[UN]; float4 x[UN];
        #pragma unroll
        for (int i = 0; i < UN; ++i) { r[i] = __ldg(ek + k + i); w[i] = __ldg(pw + k + i); }
        #pragma unroll
        for (int i = 0; i < UN; ++i) x[i] = __ldg(reinterpret_cast<const float4*>(Tj + (long)r[i] * lc));
        #pragma unroll
        for (int i = 0; i < UN; ++i) {
            acc[i].x = fmaf(w[i], x[i].x, acc[i].x); acc[i].y = fmaf(w[i], x[i].y, acc[i].y);
            acc[i].z = fmaf(w[i], x[i].z, acc[i].z); acc[i].w = fmaf(w[i], x[i].w, acc[i].w);
        }
    }
    for (int i = 0; k < K; ++k, ++i) {
        const int r0 = __ldg(ek + k);
        const float w0 = __ldg(pw + k);
        const float4 x0 = __ldg(reinterpret_cast<const float4*>(Tj + (long)r0 * lc));
        acc[i].x = fmaf(w0, x0.x, acc[i].x); acc[i].y = fmaf(w0, x0.y, acc[i].y);
        acc[i].z = fmaf(w0, x0.z, acc[i].z); acc[i].w = fmaf(w0, x0.w, acc[i].w);
    }
    #pragma unroll
    for (int h = UN / 2; h > 0; h /= 2) {
        #pragma unroll
        for (int i = 0; i < h; ++i) {
            acc[i].x += acc[i + h].x; acc[i].y += acc[i + h].y; acc[i].z += acc[i + h].z; acc[i].w += acc[i + h].w;
        }
    }
    *reinterpret_cast<float4*>(Z + ((long)v * B + b) * num_latents + l0 + j) = acc[0];
}

// node_mars from Z, as the index-driven form's epilogue (softevi_categorical.cu, `softevi_fw_epilogue`) computes it.
__global__ void softevi_rows_epilogue_kernel(const float* __restrict__ params, float* __restrict__ node_mars,
                                             const float* __restrict__ Z, const float* __restrict__ log_ex_p,
                                             const long* __restrict__ data, const long* __restrict__ vids,
                                             const long* __restrict__ s_pids, const long* __restrict__ nids,
                                             const long* __restrict__ var_idmapping,
                                             int layer_num_nodes, int batch_size, int node_offset, int num_latents) {
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
        __logf(__ldg(params + s_pids[n] + obs)) + log_ex_p[lvid * batch_size + b] - (__logf(z) - ROWS_LOG_SCALE);
}

// One forward: for each chunk of `chunk` latents, the chunk's table into `T` (>= num_keys * chunk floats) and its
// Z; then the epilogue into node_mars.
void softevi_forward_rows(torch::Tensor params, torch::Tensor node_mars, torch::Tensor Z, torch::Tensor T,
                          torch::Tensor log_ex_p, torch::Tensor data, torch::Tensor vids, torch::Tensor s_pids,
                          torch::Tensor nids, torch::Tensor var_idmapping, torch::Tensor keys, torch::Tensor p_base,
                          torch::Tensor elem_key, torch::Tensor pt, int64_t num_keys, int64_t num_cats,
                          int64_t num_latents, int64_t chunk, int64_t B, int64_t V, int64_t K,
                          int64_t layer_num_nodes, int64_t node_offset, int64_t threads, int64_t unroll) {
    TORCH_CHECK(num_latents % 4 == 0 && chunk % 4 == 0, "softevi_forward_rows: latents in fours");
    TORCH_CHECK(T.numel() >= num_keys * chunk, "softevi_forward_rows: table too small");
    auto st = at::cuda::getCurrentCUDAStream();
    for (int64_t l0 = 0; l0 < num_latents; l0 += chunk) {
        const int lc = (int)std::min(chunk, num_latents - l0);
        const dim3 tg((unsigned)((num_keys + 31) / 32), (unsigned)((lc + 31) / 32)), tb(32, 8);
        softevi_rows_table_kernel<<<tg, tb, 0, st>>>(params.data_ptr<float>(), keys.data_ptr<long>(),
            p_base.data_ptr<long>(), T.data_ptr<float>(), (int)num_keys, (int)num_latents, (long)num_cats,
            (int)l0, lc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const dim3 zg((unsigned)(B * V), (unsigned)((lc / 4 + threads - 1) / threads));
#define RZ(UV) softevi_rows_z_kernel<UV><<<zg, (unsigned)threads, 0, st>>>(T.data_ptr<float>(),             \
            elem_key.data_ptr<int>(), pt.data_ptr<float>(), Z.data_ptr<float>(), (int)B, (int)V, (int)K,    \
            (int)num_latents, (int)l0, lc)
        if (unroll == 8) RZ(8); else RZ(4);
#undef RZ
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const long tot = layer_num_nodes * B;
    softevi_rows_epilogue_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(
        params.data_ptr<float>(), node_mars.data_ptr<float>(), Z.data_ptr<float>(), log_ex_p.data_ptr<float>(),
        data.data_ptr<long>(), vids.data_ptr<long>(), s_pids.data_ptr<long>(), nids.data_ptr<long>(),
        var_idmapping.data_ptr<long>(), (int)layer_num_nodes, (int)B, (int)node_offset, (int)num_latents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The hybrid's forward (`softevi/forward.py`'s `_fw_launch_hybrid`): the hot categories' part of Z a dense product,
// Zh = P T_hot (a GEMM, made per chunk of latents beside this), and the cold candidates' rows summed onto it. As
// `softevi_rows_z_kernel` with `elem_key` the element's COLD key, -1 for a hot one (the product's: no row read, no add),
// the sum starting from the product's value.
template <int UN>
__global__ void softevi_rows_z_cold_kernel(const float* __restrict__ T, const int* __restrict__ elem_key,
                                           const float* __restrict__ pt, const float* __restrict__ Zh,
                                           float* __restrict__ Z, int B, int V, int K, int num_latents, int l0, int lc) {
    const int bv = blockIdx.x;
    const int j = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (j >= lc) return;
    const int b = bv / V, v = bv - (bv / V) * V;
    const long slot = (long)v * B + b;
    const int* ek = elem_key + (long)bv * K;
    const float* pw = pt + (long)bv * K;
    const float* Tj = T + j;
    float4 acc[UN];
    acc[0] = __ldg(reinterpret_cast<const float4*>(Zh + slot * lc + j));
    #pragma unroll
    for (int i = 1; i < UN; ++i) acc[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    int k = 0;
    for (; k + UN <= K; k += UN) {
        int r[UN]; float w[UN]; float4 x[UN];
        #pragma unroll
        for (int i = 0; i < UN; ++i) { r[i] = __ldg(ek + k + i); w[i] = __ldg(pw + k + i); }
        #pragma unroll
        for (int i = 0; i < UN; ++i)
            x[i] = r[i] >= 0 ? __ldg(reinterpret_cast<const float4*>(Tj + (long)r[i] * lc)) : make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int i = 0; i < UN; ++i) {
            acc[i].x = fmaf(w[i], x[i].x, acc[i].x); acc[i].y = fmaf(w[i], x[i].y, acc[i].y);
            acc[i].z = fmaf(w[i], x[i].z, acc[i].z); acc[i].w = fmaf(w[i], x[i].w, acc[i].w);
        }
    }
    for (int i = 0; k < K; ++k, ++i) {
        const int r0 = __ldg(ek + k);
        if (r0 < 0) continue;
        const float w0 = __ldg(pw + k);
        const float4 x0 = __ldg(reinterpret_cast<const float4*>(Tj + (long)r0 * lc));
        acc[i].x = fmaf(w0, x0.x, acc[i].x); acc[i].y = fmaf(w0, x0.y, acc[i].y);
        acc[i].z = fmaf(w0, x0.z, acc[i].z); acc[i].w = fmaf(w0, x0.w, acc[i].w);
    }
    #pragma unroll
    for (int h = UN / 2; h > 0; h /= 2) {
        #pragma unroll
        for (int i = 0; i < h; ++i) {
            acc[i].x += acc[i + h].x; acc[i].y += acc[i + h].y; acc[i].z += acc[i + h].z; acc[i].w += acc[i + h].w;
        }
    }
    *reinterpret_cast<float4*>(Z + slot * num_latents + l0 + j) = acc[0];
}

// The hybrid forward's pieces, launched one at a time between the products (`_fw_launch_hybrid`)
void softevi_rows_table(torch::Tensor params, torch::Tensor keys, torch::Tensor p_base, torch::Tensor T,
                        int64_t num_keys, int64_t num_latents, int64_t num_cats, int64_t l0, int64_t lc) {
    TORCH_CHECK(T.numel() >= num_keys * lc, "softevi_rows_table: table too small");
    if (num_keys == 0) return;
    const dim3 tg((unsigned)((num_keys + 31) / 32), (unsigned)((lc + 31) / 32)), tb(32, 8);
    softevi_rows_table_kernel<<<tg, tb, 0, at::cuda::getCurrentCUDAStream()>>>(params.data_ptr<float>(),
        keys.data_ptr<long>(), p_base.data_ptr<long>(), T.data_ptr<float>(), (int)num_keys, (int)num_latents,
        (long)num_cats, (int)l0, (int)lc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softevi_rows_z_cold(torch::Tensor T, torch::Tensor elem_key, torch::Tensor pt, torch::Tensor Zh, torch::Tensor Z,
                         int64_t B, int64_t V, int64_t K, int64_t num_latents, int64_t l0, int64_t lc) {
    TORCH_CHECK(num_latents % 4 == 0 && lc % 4 == 0, "softevi_rows_z_cold: latents in fours");
    TORCH_CHECK(Zh.is_contiguous() && Zh.numel() >= B * V * lc, "softevi_rows_z_cold: the product is [slots, chunk]");
    const int threads = 128;
    const dim3 zg((unsigned)(B * V), (unsigned)((lc / 4 + threads - 1) / threads));
    softevi_rows_z_cold_kernel<4><<<zg, threads, 0, at::cuda::getCurrentCUDAStream()>>>(T.data_ptr<float>(),
        elem_key.data_ptr<int>(), pt.data_ptr<float>(), Zh.data_ptr<float>(), Z.data_ptr<float>(), (int)B, (int)V,
        (int)K, (int)num_latents, (int)l0, (int)lc);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softevi_rows_epilogue(torch::Tensor params, torch::Tensor node_mars, torch::Tensor Z, torch::Tensor log_ex_p,
                           torch::Tensor data, torch::Tensor vids, torch::Tensor s_pids, torch::Tensor nids,
                           torch::Tensor var_idmapping, int64_t layer_num_nodes, int64_t B, int64_t node_offset,
                           int64_t num_latents) {
    const long tot = layer_num_nodes * B;
    softevi_rows_epilogue_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        params.data_ptr<float>(), node_mars.data_ptr<float>(), Z.data_ptr<float>(), log_ex_p.data_ptr<float>(),
        data.data_ptr<long>(), vids.data_ptr<long>(), s_pids.data_ptr<long>(), nids.data_ptr<long>(),
        var_idmapping.data_ptr<long>(), (int)layer_num_nodes, (int)B, (int)node_offset, (int)num_latents);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The hybrid backward's hot categories (`softevi/backward.py`'s `_hot_expected_flow`): their expected-category flow, from
// the dense product G[u, l] = sum_slots P[slot, u] ratio[slot, l] made by a GEMM,
//
//   param_flows[pf_base[g(u), l] + num_cats + cat(u)] += beta[l, cat(u)] * G[u, l]
//
// a thread a hot category (consecutive threads: ascending categories, so the parameter reads and the flows' updates
// coalesce where the hot categories are dense, as the list walk's own update), TL latents a thread from its row of G.
// Each (row, category) once: the reduction in L2 (an atomic add, result unused) never collides.
template <int TL>
__global__ void softevi_hot_flows_kernel(const float* __restrict__ params, float* __restrict__ param_flows,
                                         const float* __restrict__ G, const long* __restrict__ keys,
                                         const long* __restrict__ p_base, const long* __restrict__ pf_base,
                                         int num_hot, int num_latents, long num_cats) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int l0 = blockIdx.y * TL;
    if (u >= num_hot) return;
    const long key = keys[u];
    const long g = key / num_cats;
    const long cat = key - g * num_cats;
    const long* pb = p_base + g * num_latents + l0;
    const long* pfb = pf_base + g * num_latents + l0;
    const float* gu = G + (long)u * num_latents + l0;
    #pragma unroll
    for (int t = 0; t < TL; ++t) {
        if (l0 + t < num_latents) {
            const float v = __ldg(gu + t);
            if (v != 0.0f) atomicAdd(param_flows + pfb[t] + num_cats + cat, __ldg(params + pb[t] + cat) * v);
        }
    }
}

void softevi_hot_flows(torch::Tensor params, torch::Tensor param_flows, torch::Tensor G, torch::Tensor keys,
                       torch::Tensor p_base, torch::Tensor pf_base, int64_t num_hot, int64_t num_latents,
                       int64_t num_cats) {
    TORCH_CHECK(G.is_contiguous() && G.numel() == num_hot * num_latents, "softevi_hot_flows: G is [hot, latents]");
    if (num_hot == 0) return;
    const dim3 grid((unsigned)((num_hot + 127) / 128), (unsigned)((num_latents + 31) / 32));
    softevi_hot_flows_kernel<32><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        params.data_ptr<float>(), param_flows.data_ptr<float>(), G.data_ptr<float>(), keys.data_ptr<long>(),
        p_base.data_ptr<long>(), pf_base.data_ptr<long>(), (int)num_hot, (int)num_latents, (long)num_cats);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softevi_hot_flows", &softevi_hot_flows, "The hybrid backward's hot categories' expected flow (CUDA)");
    m.def("softevi_rows_table", &softevi_rows_table, "The row form's table of a set of keys (CUDA)");
    m.def("softevi_rows_z_cold", &softevi_rows_z_cold, "The hybrid forward's cold candidates' sums (CUDA)");
    m.def("softevi_rows_epilogue", &softevi_rows_epilogue, "The row form's epilogue (CUDA)");
    m.def("softevi_forward_rows", &softevi_forward_rows, "Top-k soft-evidence forward, row form (CUDA)");
}
