// Shared by the dfa_exact CUDA kernels: index tables of either width, and Triton's fp32 math bit for bit -- `tl.exp`
// (ex2.approx of x log2(e)), `tl.log` (libdevice's __nv_logf, ftz, as Triton inlines it), `prod._lse_step_one_exp`
// and `common._logaddexp_tile` as they compile -- so a CUDA kernel's values are the Triton kernel's own.
#pragma once

#include <torch/extension.h>
#include <math_constants.h>
#include <cstdint>

namespace tri {

struct WIdx {              // an index table of either width
    const void* p;
    int wide;
    __device__ __forceinline__ long long operator[](long long i) const {
        return wide ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
    }
};

inline WIdx widx(const torch::Tensor& t) {
    TORCH_CHECK(t.scalar_type() == torch::kInt64 || t.scalar_type() == torch::kInt32, "index tables: int32 or int64");
    return WIdx{t.data_ptr(), t.scalar_type() == torch::kInt64 ? 1 : 0};
}

inline const int* i32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.is_contiguous(), name, ": int32, contiguous");
    return t.data_ptr<int>();
}

// Triton's `tl.exp` on fp32: ex2.approx of x log2(e)
__device__ __forceinline__ float tri_exp(float x) {
    float r;
    asm("{\n\t.reg .f32 s;\n\tmul.f32 s, %1, 0f3FB8AA3B;\n\tex2.approx.f32 %0, s;\n\t}" : "=f"(r) : "f"(x));
    return r;
}

// Triton's `tl.log` on fp32: libdevice's `__nv_logf` (ftz), as Triton inlines it
__device__ __forceinline__ float tri_log(float a) {
    float r;
    asm("{\n\t"
        ".reg .pred p1, p2, p3;\n\t"
        ".reg .b32 x, j, k, m, e, adj, f, q, w;\n\t"
        "setp.lt.f32 p1, %1, 0f00800000;\n\t"
        "mul.f32 x, %1, 0f4B000000;\n\t"
        "selp.f32 x, x, %1, p1;\n\t"
        "selp.f32 adj, 0fC1B80000, 0f00000000, p1;\n\t"
        "add.s32 j, x, -1059760811;\n\t"
        "and.b32 k, j, -8388608;\n\t"
        "sub.s32 m, x, k;\n\t"
        "cvt.rn.f32.s32 e, k;\n\t"
        "fma.rn.ftz.f32 e, e, 0f34000000, adj;\n\t"
        "add.f32 f, m, 0fBF800000;\n\t"
        "fma.rn.ftz.f32 q, 0fBE055027, f, 0f3E1039F6;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBDF8CDCC;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0f3E0F2955;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBE2AD8B9;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0f3E4CED0B;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBE7FFF22;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0f3EAAAA78;\n\t"
        "fma.rn.ftz.f32 q, q, f, 0fBF000000;\n\t"
        "mul.f32 q, f, q;\n\t"
        "fma.rn.ftz.f32 q, q, f, f;\n\t"
        "fma.rn.ftz.f32 w, e, 0f3F317218, q;\n\t"
        "setp.lt.u32 p2, x, 2139095040;\n\t"
        "@!p2 fma.rn.ftz.f32 w, x, 0f7F800000, 0f7F800000;\n\t"
        "setp.eq.f32 p3, x, 0f00000000;\n\t"
        "selp.f32 %0, 0fFF800000, w, p3;\n\t"
        "}" : "=f"(r) : "f"(a));
    return r;
}

// `prod._lse_step_one_exp`, as Triton compiles it
__device__ __forceinline__ void lse_one_exp(float& m, float& acc, float v) {
    float e = tri_exp(0.0f - fabsf(v - m));
    float up;
    asm("fma.rn.f32 %0, %1, %2, 0f3F800000;" : "=f"(up) : "f"(acc), "f"(e));
    float next = v > m ? up : acc + e;
    acc = v == -CUDART_INF_F ? acc : next;
    m = fmaxf(m, v);
}

// `common._logaddexp_tile`
__device__ __forceinline__ float log_add(float a, float b) {
    float hi = fmaxf(a, b);
    float s = hi == -CUDART_INF_F ? 0.0f : hi;
    return hi == -CUDART_INF_F ? -CUDART_INF_F : tri_log(tri_exp(a - s) + tri_exp(b - s)) + s;
}

}  // namespace tri
