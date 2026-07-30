// The single pybind module for the external-parameter CUDA kernels.
//
// Kept apart from the kernel sources so that each of those stays a self-contained translation unit --
// `PYBIND11_MODULE` may appear only once per extension, and hanging every new kernel's declaration off
// whichever file happens to own it makes the ownership arbitrary.

#include <torch/extension.h>

// ---- lowrank_forward.cu ----
void lowrank_forward(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor ext,
                     torch::Tensor nids, torch::Tensor cids, torch::Tensor xu, torch::Tensor xv,
                     torch::Tensor pw, torch::Tensor pa,
                     torch::Tensor log_w, torch::Tensor log_a, torch::Tensor log_z,
                     int64_t block_size, int64_t ch_block_size, int64_t rank, int64_t ext_base,
                     int64_t tile_c, int64_t tile_m, int64_t tb1, int64_t tb2);

// ---- lowrank_backward.cu ----
void lowrank_backward(torch::Tensor node_flows, torch::Tensor element_flows,
                      torch::Tensor node_mars_T, torch::Tensor element_mars,
                      torch::Tensor ext, torch::Tensor grad_ext,
                      torch::Tensor nids, torch::Tensor cids,
                      torch::Tensor xu, torch::Tensor xv,
                      torch::Tensor log_w, torch::Tensor log_a, torch::Tensor log_z,
                      torch::Tensor p_lp, torch::Tensor p_lq,
                      torch::Tensor log_p, torch::Tensor log_q,
                      int64_t block_size, int64_t ch_block_size, int64_t rank, int64_t ext_base,
                      int64_t tile_n, int64_t tile_c, int64_t tb, bool accumulate);

void lowrank_shift_logz(torch::Tensor node_mars, torch::Tensor nids, torch::Tensor log_z,
                        int64_t block_size, double sign);

// ---- blockscale_forward.cu ----
void blockscale_forward(torch::Tensor node_mars, torch::Tensor element_mars, torch::Tensor params,
                        torch::Tensor ext, torch::Tensor nids, torch::Tensor cids,
                        torch::Tensor pids, torch::Tensor gate, torch::Tensor log_z,
                        int64_t block_size, int64_t node_ch_block_size, int64_t gate_ch_block_size,
                        int64_t n_node_gates, int64_t ext_base,
                        int64_t tm, int64_t tb, int64_t tk);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lowrank_forward", &lowrank_forward,
          "Fused per-sample low-rank sum-layer forward correction (both phases, one host call)");
    m.def("lowrank_backward", &lowrank_backward,
          "Per-sample low-rank sum-layer backward: child-flow correction + dLL/dU, dLL/dV");
    m.def("lowrank_shift_logz", &lowrank_shift_logz,
          "Add (or subtract) logZ over a layer's node range, turning node_mars into logT");
    m.def("blockscale_forward", &blockscale_forward,
          "Per-block multiplicative gate: computes node_mars = log N - log Z for a whole partition");
}
