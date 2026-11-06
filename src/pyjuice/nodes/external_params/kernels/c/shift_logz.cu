// `node_mars[nids[r] + m, b] += sign * log_z[r, m, b]` over a layer's node rows.
//
// A parameterization whose effective parameters are the shared ones scaled and renormalized
// (`BlockScaleSumParams`) keeps its nodes' `log Z` beside the node values: adding it turns the node values into
// the unnormalized ones the shared-parameter backward kernels expect, and subtracting it afterwards turns them back.
// One thread an entry; every (node, sample) belongs to one thread, so the read-modify-write cannot race.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

__global__ void shift_logz_kernel(
        float* __restrict__ node_mars, const int64_t* __restrict__ nids,
        const float* __restrict__ log_z, int batch_size, int block_size, int rows, float sign) {

    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)rows * block_size * batch_size;
    if (i >= total) return;

    const int b = i % batch_size;
    const int m = (i / batch_size) % block_size;
    const int row = i / ((long)batch_size * block_size);

    node_mars[(nids[row] + m) * batch_size + b] += sign * log_z[i];
}

void shift_logz(torch::Tensor node_mars, torch::Tensor nids, torch::Tensor log_z,
                        int64_t block_size, double sign) {
    const int batch_size = node_mars.size(1);
    const int rows = nids.size(0);
    const long total = (long)rows * block_size * batch_size;

    const int threads = 256;
    shift_logz_kernel<<<(int)((total + threads - 1) / threads), threads, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        node_mars.data_ptr<float>(), nids.data_ptr<int64_t>(), log_z.data_ptr<float>(),
        batch_size, (int)block_size, rows, (float)sign);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
