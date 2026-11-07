// The pybind module for the PLAIN-CUDA external-parameter kernels.
//
// Kept apart from the kernel sources so each of those stays a self-contained translation unit: the
// module macro may appear only once per extension, and hanging every new kernel's declaration off
// whichever file happens to own it makes the ownership arbitrary.
//
// The CuTe/TMA block-scale forward is NOT here -- it is its own extension, because it needs CUTLASS
// headers, an arch-specific flag and the driver API, and because a CUTLASS failure should not take these
// kernels down with it.

#include <torch/extension.h>
#include <vector>

// ---- shift_logz.cu ----
void shift_logz(torch::Tensor node_mars, torch::Tensor nids, torch::Tensor log_z,
                int64_t block_size, double sign);

// ---- staging_transpose.cu ----
void staging_transpose(torch::Tensor dst, torch::Tensor src, int64_t B, int64_t N);
void staging_transpose_batched(const std::vector<int64_t>& rows, int64_t max_b, int64_t max_n);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shift_logz", &shift_logz,
          "Add (or subtract) logZ over a layer's node range, turning node_mars into logT");
    m.def("staging_transpose", &staging_transpose,
          "Tiled [B, N] -> [N, B] copy for staging batch-innermost external parameters");
    m.def("staging_transpose_batched", &staging_transpose_batched,
          "Several staging transposes in one launch, the pairs (src, dst, B, N) flat, by value");
}
