// Categorical input-layer backward (param-flow accumulation) via shared-memory privatized histogram.
// Replaces the Triton scattered global atomic_add. One CUDA block owns one node's FULL batch:
//   1. accumulate batch flows into smem hist[num_cats] (fast smem atomics, low contention),
//   2. flush hist -> param_flows[s_pfid + cat] += hist[cat], COALESCED float4 RMW.
// The flush is NON-ATOMIC: it is correct only when each node's param_flow region is written by exactly
// one block, i.e. s_pfids are distinct (untied) — the caller's gate guarantees this (else Triton).
// Computes the same as Categorical.bk_flow_fn: param_flows[node,cat] += sum_b (exp if logspace)flow[node,b]
// over batch elements b with data[vid[node],b]==cat.
//
// `data` is integer category ids; pyjuice uses a compact dtype (uint8 for num_cats<=256, else
// int16/int32/int64), so the kernel is templated on the data type and dispatched at launch.
// float4-vectorized over node_flows/param_flows; requires num_cats%4==0, batch_size%4==0, and 16B-aligned
// bases (s_pfids multiples of 4, node_offset & batch_size multiples of 4) — enforced by the input_layer gate.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#ifndef NTH
#define NTH 128     // blockDim: 64/128 fastest (measured), 256 ok, 512 bad
#endif

template <typename DataT>
__global__ void cat_backward_kernel(float* __restrict__ param_flows, const float* __restrict__ node_flows,
                                    const DataT* __restrict__ data, const long* __restrict__ vids,
                                    const long* __restrict__ s_pfids, int layer_num_nodes, int batch_size,
                                    long node_offset, int num_cats, int logspace) {
    int n = blockIdx.x;
    if (n >= layer_num_nodes) return;
    extern __shared__ float hist[];
    int nc4 = num_cats >> 2, B4 = batch_size >> 2;
    for (int c4 = threadIdx.x; c4 < nc4; c4 += blockDim.x) reinterpret_cast<float4*>(hist)[c4] = make_float4(0, 0, 0, 0);
    __syncthreads();

    long vid = vids[n];
    long nf_base = (node_offset + (long)n) * (long)batch_size;
    const DataT* dptr = data + vid * (long)batch_size;
    const float4* nf4 = reinterpret_cast<const float4*>(node_flows + nf_base);
    for (int b4 = threadIdx.x; b4 < B4; b4 += blockDim.x) {
        float4 f = nf4[b4]; int b = b4 << 2;
        if (logspace) { f.x = __expf(f.x); f.y = __expf(f.y); f.z = __expf(f.z); f.w = __expf(f.w); }
        atomicAdd(&hist[(int)dptr[b]],     f.x); atomicAdd(&hist[(int)dptr[b + 1]], f.y);
        atomicAdd(&hist[(int)dptr[b + 2]], f.z); atomicAdd(&hist[(int)dptr[b + 3]], f.w);
    }
    __syncthreads();

    float4* pf4 = reinterpret_cast<float4*>(param_flows + s_pfids[n]);
    for (int c4 = threadIdx.x; c4 < nc4; c4 += blockDim.x) {
        float4 h = reinterpret_cast<float4*>(hist)[c4], p = pf4[c4];
        p.x += h.x; p.y += h.y; p.z += h.z; p.w += h.w;
        pf4[c4] = p;
    }
}

void cat_backward(torch::Tensor param_flows, torch::Tensor node_flows, torch::Tensor data,
                  torch::Tensor vids, torch::Tensor s_pfids, int layer_num_nodes, int batch_size,
                  long node_offset, int num_cats, int logspace) {
    int smem = num_cats * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    float* pf = param_flows.data_ptr<float>();
    const float* nf = node_flows.data_ptr<float>();
    const long* vp = vids.data_ptr<long>();
    const long* sp = s_pfids.data_ptr<long>();
#define LAUNCH(T) cat_backward_kernel<T><<<layer_num_nodes, NTH, smem, stream>>>( \
        pf, nf, data.data_ptr<T>(), vp, sp, layer_num_nodes, batch_size, node_offset, num_cats, logspace)
    switch (data.scalar_type()) {
        case torch::kByte:  LAUNCH(uint8_t);  break;
        case torch::kChar:  LAUNCH(int8_t);   break;
        case torch::kShort: LAUNCH(int16_t);  break;
        case torch::kInt:   LAUNCH(int32_t);  break;
        case torch::kLong:  LAUNCH(int64_t);  break;
        default: TORCH_CHECK(false, "cat_backward: unsupported data dtype ", data.scalar_type());
    }
#undef LAUNCH
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cat_backward", &cat_backward, "Categorical input-layer backward (smem-histogram)");
}
