// Tiled transpose for staging external parameters.
//
// Staging copies the caller's tensor into the PC's buffer, and for any parameterization that stores
// batch-innermost -- which the kernels want, so that adjacent threads read adjacent samples -- that
// copy is a TRANSPOSE: `[B, N] -> [N, B]`, with N the product of everything but the batch axis.
//
// `Tensor.copy_` on a transposed view goes through TensorIterator, which handles arbitrary strides but
// does not tile, so one of the two sides is uncoalesced -- each 4-byte element costs its own sector.
// Measured on sm_120, against this kernel on identical data:
//
//     B x N          torch    tiled
//     256 x 2048     8.2 us   2.1 us     3.9x
//     1024 x 2048   20.5 us   4.1 us     5.0x
//     1024 x 8192   69.7 us  12.3 us     5.7x
//
// That matters because staging is not a rounding error: at batch 256 it was 37-59% of the entire cost
// of applying a gate, more than the sum kernel's own overhead.
//
// Plain CUDA -- no CuTe, no CUTLASS, no tensor cores -- so it is available wherever CUDA is.

#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>


// The classic 32x32 staging tile: read a tile coalesced along N, write it coalesced along B. The row
// is padded to 33 so that the transposed read hits 32 distinct banks instead of one.
__global__ void staging_transpose_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                         int B, int N) {
    __shared__ float tile[32][33];

    const int n0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;

    for (int i = 0; i < 32; i += 8) {
        const int b = b0 + ty + i, n = n0 + tx;
        if (b < B && n < N) tile[ty + i][tx] = src[(long)b * N + n];
    }
    __syncthreads();

    for (int i = 0; i < 32; i += 8) {
        const int n = n0 + ty + i, b = b0 + tx;
        if (n < N && b < B) dst[(long)n * B + b] = tile[tx][ty + i];
    }
}


// `dst[n, b] = src[b, n]`, both contiguous. The caller guarantees the shapes; anything it cannot
// express this way keeps using `Tensor.copy_`.
void staging_transpose(torch::Tensor dst, torch::Tensor src, int64_t B, int64_t N) {
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(),
                "staging_transpose: both tensors must be contiguous");
    TORCH_CHECK(src.numel() == B * N && dst.numel() == B * N,
                "staging_transpose: element count does not match B * N");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32 && dst.scalar_type() == torch::kFloat32,
                "staging_transpose: float32 only");

    if (B == 0 || N == 0) return;

    dim3 grid((unsigned)((N + 31) / 32), (unsigned)((B + 31) / 32));
    dim3 block(32, 8);

    staging_transpose_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), (int)B, (int)N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Several transposes in ONE launch -- a forward's staging of every node's tensors, a backward's gradients written
// back --: pair `blockIdx.z` reads its `(src address, dst address, B, N)` from the launch's PARAMETERS (by value: no
// table on the device, so nothing to upload or to cache by the pairs' addresses -- a caller handing fresh tensors
// every call, a router head's output or new gradient buffers, cost an upload a call), a block past its pair's tiles
// returns (the grid spans the largest pair). At most PAIRS_PER_LAUNCH pairs a launch (2 KB of parameters), more in
// as many launches. MEASURED (a tied gated HMM-2048 x4: 15 nodes' gates a forward): one call per pair was most of
// the staging's 83 us of host time.
constexpr int PAIRS_PER_LAUNCH = 64;
struct TransposePairs { long long rows[PAIRS_PER_LAUNCH * 4]; };

__global__ void staging_transpose_batched_kernel(const TransposePairs pairs) {
    const long long* t = pairs.rows + 4 * blockIdx.z;
    const float* src = reinterpret_cast<const float*>(t[0]);
    float* dst = reinterpret_cast<float*>(t[1]);
    const int B = (int)t[2], N = (int)t[3];
    const int n0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    if (n0 >= N || b0 >= B) return;                     // (the whole block: no barrier is left waiting)
    __shared__ float tile[32][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    for (int i = 0; i < 32; i += 8) {
        const int b = b0 + ty + i, n = n0 + tx;
        if (b < B && n < N) tile[ty + i][tx] = src[(long)b * N + n];
    }
    __syncthreads();
    for (int i = 0; i < 32; i += 8) {
        const int n = n0 + ty + i, b = b0 + tx;
        if (n < N && b < B) dst[(long)n * B + b] = tile[tx][ty + i];
    }
}

// `dst_p[n, b] = src_p[b, n]` for every pair p of `rows` (flat: src address, dst address, B, N per pair -- contiguous
// float32 tensors of B * N elements each, which the caller guarantees) -- `max_b` / `max_n` the largest B and N.
void staging_transpose_batched(const std::vector<int64_t>& rows, int64_t max_b, int64_t max_n) {
    TORCH_CHECK(rows.size() % 4 == 0, "staging_transpose_batched: 4 entries a pair (src, dst, B, N)");
    const long P = (long)rows.size() / 4;
    if (P == 0 || max_b == 0 || max_n == 0) return;
    dim3 block(32, 8);
    for (long p0 = 0; p0 < P; p0 += PAIRS_PER_LAUNCH) {
        const int n = (int)std::min<long>(PAIRS_PER_LAUNCH, P - p0);
        TransposePairs pairs;
        for (int i = 0; i < 4 * n; i++) pairs.rows[i] = (long long)rows[4 * p0 + i];
        dim3 grid((unsigned)((max_n + 31) / 32), (unsigned)((max_b + 31) / 32), (unsigned)n);
        staging_transpose_batched_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(pairs);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
