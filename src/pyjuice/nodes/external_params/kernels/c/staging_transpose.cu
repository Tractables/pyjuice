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
