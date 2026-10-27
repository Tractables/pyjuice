// The per-gate parameter sums of a `BlockScaleSumParams` partition (`_DenominatorsMixin._sigma`):
//
//     sigma[(r * block_size + n) * n_gates + g] = sum over gate g's edges e of params[pids[r, e] + n]    (fp32)
//
// one thread per (row r, gate g, node n), the node fastest: an edge's parameters for a node block are contiguous, so
// each of a gate's loads is coalesced over the nodes, and UNR of them are in flight at once (UNR partial sums, added in
// order at the end). One read of the parameters, nothing staged. What `_sigma` did with torch -- an int64 index tensor
// the size of the partition's parameters, their fp32 copy, then a sum -- MEASURED (RTX PRO 6000, a gated HMM layer's
// partition at a parameter change): 4 node blocks of 1024 x 4096 edges, gate width 64, 316 -> 12.5 us and ~300 MiB
// of transient memory -> none; gate width 16 329 -> 17.2 us; 1024 x 1024 edges 46 -> 9.9 us. On those layers' bf16
// parameters the sums came out equal to the bit (in general the order differs from torch's: within fp32 rounding).
//
// Scope (checked): parameters fp32 or bf16 (`param_ptr`), `pids` int64 [rows, num_edges] and contiguous, a gate width
// dividing num_edges, `sigma` fp32 [rows * block_size, n_gates] and contiguous.

#include <torch/extension.h>
#include "../../../../layer/kernels/c/param_load.cuh"   // `ParamLoad<TP>`, `param_ptr<TP>`: the stored dtype
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#define NT 256          // threads a block
#define UNR 16          // loads in flight a thread

template <typename TP>
__global__ void __launch_bounds__(NT) bs_sigma_kernel(const TP* __restrict__ params, const long* __restrict__ pids,
                                                      float* __restrict__ sigma, long total, int block_size,
                                                      int num_edges, int gate_cbs, int n_gates) {
    const long i = (long)blockIdx.x * NT + threadIdx.x;
    if (i >= total) return;
    const int n = (int)(i % block_size);
    const long rg = i / block_size;
    const int g = (int)(rg % n_gates);
    const long r = rg / n_gates;
    const long* pe = pids + r * num_edges + (long)g * gate_cbs;
    float acc[UNR];
#pragma unroll
    for (int k = 0; k < UNR; k++) acc[k] = 0.f;
    for (int e0 = 0; e0 < gate_cbs; e0 += UNR) {
        float v[UNR];
#pragma unroll
        for (int k = 0; k < UNR; k++) v[k] = (e0 + k < gate_cbs) ? ParamLoad<TP>::at(params + pe[e0 + k] + n) : 0.f;
#pragma unroll
        for (int k = 0; k < UNR; k++) acc[k] += v[k];
    }
    float s = acc[0];
#pragma unroll
    for (int k = 1; k < UNR; k++) s += acc[k];
    sigma[(r * block_size + n) * n_gates + g] = s;
}

void blockscale_sigma(torch::Tensor params, torch::Tensor pids, torch::Tensor sigma, int64_t block_size,
                      int64_t gate_cbs) {
    TORCH_CHECK(pids.scalar_type() == torch::kLong && pids.dim() == 2 && pids.is_contiguous(),
                "blockscale sigma: pids int64 [rows, num_edges], contiguous");
    TORCH_CHECK(params.is_contiguous() && params.device() == pids.device() && sigma.device() == pids.device(),
                "blockscale sigma: contiguous parameters, every tensor on one device");
    const long rows = pids.size(0), num_edges = pids.size(1);
    TORCH_CHECK(block_size > 0 && gate_cbs > 0 && num_edges % gate_cbs == 0,
                "blockscale sigma: a gate width dividing the edge count");
    const long n_gates = num_edges / gate_cbs;
    TORCH_CHECK(sigma.scalar_type() == torch::kFloat && sigma.is_contiguous() &&
                sigma.numel() == rows * block_size * n_gates,
                "blockscale sigma: sigma fp32 [rows * block_size, n_gates]");
    const long total = rows * n_gates * block_size;
    if (total == 0) return;
    const at::cuda::OptionalCUDAGuard guard(pids.device());
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    const long blocks = (total + NT - 1) / NT;
    if (params.scalar_type() == torch::kBFloat16)
        bs_sigma_kernel<__nv_bfloat16><<<blocks, NT, 0, st>>>(param_ptr<__nv_bfloat16>(params), pids.data_ptr<long>(),
                                                               sigma.data_ptr<float>(), total, (int)block_size,
                                                               (int)num_edges, (int)gate_cbs, (int)n_gates);
    else
        bs_sigma_kernel<float><<<blocks, NT, 0, st>>>(param_ptr<float>(params), pids.data_ptr<long>(),
                                                      sigma.data_ptr<float>(), total, (int)block_size, (int)num_edges,
                                                      (int)gate_cbs, (int)n_gates);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockscale_sigma", &blockscale_sigma,
          "a partition's per-gate parameter sums sigma[(r * block_size + n) * n_gates + g] (fp32)");
}
