// Loads of the sum layers' parameters for kernels templated on the dtype they are STORED in (`TensorCircuit.param_dtype`):
// `float`, or `__nv_bfloat16` -- 8 consecutive values as floats from one 16-byte load (the operand tiles), 4 as floats
// (the param-flow epilogue), and one. The fp32 forms are the loads the kernels had before they were templated
// (two float4 / one float4 / a scalar), so the fp32 instantiations compute what they always did, to the bit.
#pragma once
#include <cuda_bf16.h>
#include <type_traits>

template <typename TP> struct ParamLoad;

template <> struct ParamLoad<float> {
    static __device__ __forceinline__ void load8(const float* __restrict__ g, float* r) {
        float4 a = *(const float4*)g, b = *(const float4*)(g + 4);
        r[0] = a.x; r[1] = a.y; r[2] = a.z; r[3] = a.w; r[4] = b.x; r[5] = b.y; r[6] = b.z; r[7] = b.w;
    }
    static __device__ __forceinline__ void load4(const float* __restrict__ g, float* r) {
        float4 a = *(const float4*)g;
        r[0] = a.x; r[1] = a.y; r[2] = a.z; r[3] = a.w;
    }
    static __device__ __forceinline__ float at(const float* __restrict__ g) { return *g; }
};

template <> struct ParamLoad<__nv_bfloat16> {
    // 8 bf16 = 16 bytes: the operand tiles' rows start at multiples of 8 parameters (a block's parameters start at a
    // multiple of its block size, the tiles step by 8), so the load is aligned as the fp32 float4 pair was
    static __device__ __forceinline__ void load8(const __nv_bfloat16* __restrict__ g, float* r) {
        uint4 u = *(const uint4*)g;
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); r[2 * i] = f.x; r[2 * i + 1] = f.y; }
    }
    static __device__ __forceinline__ void load4(const __nv_bfloat16* __restrict__ g, float* r) {
        uint2 u = *(const uint2*)g;
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
        float2 f0 = __bfloat1622float2(h[0]), f1 = __bfloat1622float2(h[1]);
        r[0] = f0.x; r[1] = f0.y; r[2] = f1.x; r[3] = f1.y;
    }
    static __device__ __forceinline__ float at(const __nv_bfloat16* __restrict__ g) { return __bfloat162float(*g); }
};

// The parameters' device pointer in the dtype `TP` the kernel was instantiated for -- checked, as the `data_ptr<float>()`
// it replaced was: the entries dispatch bf16 to one instantiation and everything else to the fp32 one, which would
// otherwise reinterpret, say, fp16 parameters silently.
template <typename TP> static inline const TP* param_ptr(const torch::Tensor& params) {
    constexpr bool is_fp32 = std::is_same<TP, float>::value;
    TORCH_CHECK(params.scalar_type() == (is_fp32 ? torch::kFloat : torch::kBFloat16),
                "the sum layers' parameters are ", params.scalar_type(), "; these kernels take Float or BFloat16");
    return reinterpret_cast<const TP*>(params.data_ptr());
}
