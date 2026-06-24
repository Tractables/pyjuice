
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import warnings

try:
    from triton.runtime.errors import OutOfResources as _TritonOutOfResources
except Exception:  # pragma: no cover - guards across triton versions
    class _TritonOutOfResources(Exception):
        pass
from copy import deepcopy
from typing import Sequence, List, Tuple, Optional

from pyjuice.nodes import SumNodes
from pyjuice.utils import BitSet
from pyjuice.utils.parameter_list import FastParamList
from .kernels import sum_forward_block_sparse as fw_bsparse
from .kernels import sum_forward_sparse as fw_sparse
from .kernels import sum_backward_node_flows as bk_nflows
from .kernels import sum_backward_element_block_sparse as bk_ele_bsparse
from .kernels import sum_backward_element_sparse as bk_ele_sparse
from .kernels import sum_backward_param_block_sparse as bk_par_bsparse
from .kernels import sum_backward_param_sparse as bk_par_sparse
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges
from .backend.index_set import batched_index_set, index_cum
from .compilation import get_sum_layer_forward_stats, sum_layer_forward_compilation, \
                         get_sum_layer_backward_stats, \
                         sum_layer_backward_compilation, next_power_of_2


# Launch tuning for the block-sparse parameter-flow backward kernel (kernel body unchanged).
# When enabled, the dominant LL block-sparse-dot regime uses a larger `TILE_SIZE_K`, a
# smaller `TILE_SIZE_B`, and `num_warps = 8`. `TILE_SIZE_M` -- which sets the node group over
# which the `log_n_fdm_max` stabilizer is taken -- is left unchanged, so per-element results
# match the original up to floating-point reduction-order noise (the same ~1e-7 order as the
# atomic-add nondeterminism already present in this kernel). These constants are tuned for an
# RTX PRO 6000 (Blackwell); set to False to recover the exact original launch configuration.
BACKWARD_PAR_FLOW_TUNED = True

# Analogous tuning for the block-sparse element-flow backward kernel: a larger TILE_SIZE_M
# (element-output tiling only, so bit-exact). Tuned for an RTX PRO 6000; set False to disable.
BACKWARD_ELE_FLOW_TUNED = True

# Optional CUDA (CuTe/CUTLASS + fp16 + TMA) fast path for the block-sparse element-flow backward
# `_bk_triton_block_sparse_ele_kernel`. When enabled AND all dispatch conditions hold (LL, logspace
# flows, allow_modify_flows / allow_neg_flows / accumulate_ch_flows all off, no tempering / partial
# eval, TILE_SIZE_K == 64, ptr_inc_step == 1, block_size % 128 == 0, batch % 64 == 0, and a
# TMA-capable GPU sm_90+ with CUTLASS), it replaces the Triton ele launch. Numerically equivalent
# (fp16 dot + fp32 accumulate; ~1.07e-3 in log-space, under the 1.5e-3 bound). The best of {CUDA,
# Triton} is autotuned once per layer signature INTO A SCRATCH BUFFER (never the live element_flows,
# which is read-accumulate-write when accumulate_ch_flows=True) and cached. Falls back to Triton on
# any unsupported shape / missing toolchain. Set False to disable.
BACKWARD_ELE_FLOW_CUDA = True

# Optional CUDA (CuTe/CUTLASS + fp16 + TMA) fast path for the block-sparse parameter-flow backward
# `_bk_triton_block_sparse_par_kernel(_rmw)`. When enabled AND all dispatch conditions hold (LL,
# logspace flows, allow_modify_flows / allow_neg_flows / negate_pflows all off, temperature 1,
# collision-free (untied) param-flow slots, contiguous cids / block_size-strided pids&pfids,
# block_size % 64 == 0, num_edges % 128 == 0, batch % 32 == 0, and a TMA-capable GPU sm_90+ with
# CUTLASS), it replaces the Triton param launch. Numerically equivalent (fp16 dot + fp32 accumulate;
# unbiased, < 1e-3 in log-param space). It is a batch-contraction GEMM whose epilogue is the dominant
# (params-I/O) cost; the kernel uses BK=32 for 4-CTA occupancy, edge-blocking to share node loads, and
# a 128-bit vectorized RMW epilogue. The best of {CUDA, Triton} is autotuned once per layer signature
# INTO A SCRATCH BUFFER (never the live param_flows, which is read-accumulate-write) and cached. Falls
# back to Triton on any unsupported shape / missing toolchain. Set False to disable.
BACKWARD_PAR_FLOW_CUDA = True

# Launch tuning for the block-sparse sum-layer FORWARD kernel (kernel body unchanged). In the LL
# block-sparse-dot regime, `emars_max` / `exp(emars - max)` / the `emars` load depend only on
# (edge, batch), not on the output node, so the standard `TILE_SIZE_M = 32` recomputes them
# `block_size / TILE_SIZE_M` times across the m-tile programs of a block. `TILE_SIZE_M` and
# `BLOCK_B` are pure output- / batch-tiling (the max stabilizer is over `TILE_SIZE_K` edges,
# left unchanged), so enlarging them is BIT-EXACT; doing so amortizes the redundant max/exp/load
# (`TILE_SIZE_M`) and the `epars` load (`BLOCK_B`). Raises the tile cap 2048 -> 4096 (=> 64x64 for
# `TILE_SIZE_K = 64`). Auto-falls-back on `OutOfResources` (smaller GPUs). Tuned for an RTX PRO
# 6000 (Blackwell); set False to recover the exact original launch configuration.
FORWARD_SUM_TUNED = True

# Optional CUDA (CuTe/CUTLASS + TMA) fast path for the block-sparse sum-layer FORWARD `tlmm`
# kernel. When enabled AND all dispatch conditions hold (LL propagation, bf16 dot path, no
# partial-eval / tempering, block_size % 128 == 0, batch % 64 == 0, TILE_SIZE_K == 64, a
# contiguous edge/param layout, and a TMA-capable GPU sm_90+ with CUTLASS available), it replaces
# the Triton `tlmm` launch. The kernel is JIT-compiled on first use and is numerically equivalent
# to the Triton kernel (bf16 dot + fp32 accumulate; agrees to ~1.5e-3 in log-space, well within the
# accuracy bar). On ANY of: incompatible GPU, missing nvcc/CUTLASS, compile failure, or an
# unsupported layer shape, it transparently falls back to the Triton kernel. Set False to disable.
FORWARD_SUM_CUDA = True
from .kernels import c as cuda_kernels


class SumLayer(Layer, nn.Module):

    BLOCK_SPARSE = 0
    SPARSE = 1
    PYTORCH = 2
    STR2MODE = {"block_sparse": 0, "sparse": 1, "pytorch": 2}

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int, 
                 global_pid_start: int, global_pfid_start: int, node2tiednodes: dict(),
                 layer_sparsity_tol: Optional[float] = None, 
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_block: int = 8,
                 disable_gpu_compilation: bool = False,
                 force_gpu_compilation: bool = False) -> None:

        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."
        assert len(nodes) == len(set(nodes)), "Input node list contains duplicates."

        layer_nid_start = global_nid_start
        layer_pid_start = global_pid_start
        layer_pfid_start = global_pfid_start

        ## Get layer statistics & prepare for compilation ##

        # n_chs:       [num_node_blocks]          stores the number of child nodes of each node
        # Note: to allow different nodes to have different `ch_block_size`s, we record the number of 
        #       child **nodes** (instead of # node blocks) in `n_chs`
        layer_num_nblocks, layer_num_edges, n_chs = get_sum_layer_forward_stats(self.nodes, global_nid_start)

        self.num_nodes = layer_num_nblocks * self.block_size # Total number of nodes
        self.num_edges = layer_num_edges # Total number of edges

        # Find a good strategy to partition the node blocks according to their number of children 
        # to minimize total computation cost
        fw_partition_max_chs = partition_nodes_by_n_edges(
            n_chs, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
        )

        # Since the triton kernels require the maximum number children for each block to be a power of 2,
        # we postprocess the partition sizes
        fw_partition_max_chs = torch.unique(next_power_of_2(fw_partition_max_chs))

        self.num_fw_partitions = len(fw_partition_max_chs) # Number of blocks

        # fw_n_partition_ids:      [num_nblocks]           stores the partition id for each node block
        # fw_n_id_in_partition:    [num_nblocks]           stores the index of the node blocks in the partition
        # fw_num_ngs_in_partition: [num_fw_partitions]     number of node blocks in each partition
        fw_n_partition_ids = torch.zeros([layer_num_nblocks], dtype = torch.long)
        fw_n_id_in_partition = torch.zeros([layer_num_nblocks], dtype = torch.long)
        fw_num_ngs_in_partition = torch.zeros([self.num_fw_partitions], dtype = torch.long)

        min_n_chs = 1
        for partition_id, max_n_chs in enumerate(fw_partition_max_chs):
            criterion = (n_chs >= min_n_chs) & (n_chs <= max_n_chs)
            partition_size = criterion.sum().item()

            fw_n_partition_ids[criterion] = partition_id
            fw_n_id_in_partition[criterion] = torch.arange(partition_size)
            fw_num_ngs_in_partition[partition_id] = partition_size

            min_n_chs = max_n_chs + 1

        ## Initialize forward pass ##

        # nids:      List[[partition_size]]                      stores node block ids
        # cids:      List[[partition_size, partition_max_n_chs]] stores indices of child node blocks
        # pids:      List[[partition_size, partition_max_n_chs]] stores indices of edge parameters (1st parameter of every block)
        # pfids:     List[[partition_size, partition_max_n_chs]] stores indices of edge parameter flows (1st parameter flow of every block)
        nids, cids, pids, pfids, layer_pid_end, layer_pfid_end = sum_layer_forward_compilation(
            self.nodes, fw_partition_max_chs, fw_n_partition_ids, fw_n_id_in_partition, 
            fw_num_ngs_in_partition, n_chs, global_nid_start, global_pid_start, global_pfid_start, node2tiednodes,
            max_tied_ns_per_parflow_block = max_tied_ns_per_parflow_block,
            # GPU compilation is slightly slower for small layer due to the kernel jit compilation time
            use_cuda = force_gpu_compilation or (not disable_gpu_compilation and (self.num_edges > 1000))
        )

        # Store buffers for the forward pass
        self.partitioned_nids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in nids])
        self.partitioned_cids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in cids])
        self.partitioned_pids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in pids])
        self.partitioned_pfids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in pfids])

        # Store pre-compiled indices from `cids` and `pids` in the following buffer
        self._cached_fw_pcids = dict()

        # Per-signature cache for the optional CUDA fast path: (ebase, pbase, cuda_ok) where ebase /
        # pbase are the per-tile first child-node index / first param offset, and cuda_ok records
        # whether this layer's edge/param layout satisfies the kernel's contiguity assumptions.
        self._cached_fw_cuda = dict()
        # Per-signature autotuned dispatch choice: ("cuda", cfg_id) or ("triton", -1). Decided once
        # by benchmarking the valid CUDA tile configs against Triton for this layer's shape.
        self._cached_fw_cuda_choice = dict()

        # Layer info
        self._layer_nid_range = (layer_nid_start, layer_nid_start + self.num_nodes)
        self._layer_pid_range = (layer_pid_start, layer_pid_end)
        self._layer_pfid_range = (layer_pfid_start, layer_pfid_end)

        ## Initialize backward pass ##

        # A sum layer could have children of different block sizes
        # We separate and partition them into different backward kernels
        ch_gsize2cs, ch_gsize2num_nblocks, ch_gsize2n_pargs, cs2parns = get_sum_layer_backward_stats(nodes)

        # For every possible child block size, we first compute the best partition strategy.
        # We then move on to do the actual compilation
        chids = []
        parids = []
        parpids = []
        cs_block_sizes = []
        for ch_gsize in ch_gsize2n_pargs:

            num_nblocks = ch_gsize2num_nblocks[ch_gsize]
            n_pargs = ch_gsize2n_pargs[ch_gsize]

            # Find a good strategy to partition the node blocks according to their number of children 
            # to minimize total computation cost
            bk_partition_max_pars = partition_nodes_by_n_edges(
                n_pargs, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
            )

            # Since the triton kernels require the maximum number children for each block to be a power of 2,
            # we postprocess the partition sizes
            bk_partition_max_pars = torch.unique(next_power_of_2(bk_partition_max_pars))
            num_bk_partitions = bk_partition_max_pars.size(0)

            # bk_n_partition_ids:      [num_nblocks]           stores the partition id for each node block
            # bk_n_id_in_partition:    [num_nblocks]           stores the index of the node blocks in the partition
            # bk_num_ngs_in_partition: [num_bk_partitions]     number of node blocks in each partition
            bk_n_partition_ids = torch.zeros([num_nblocks], dtype = torch.long)
            bk_n_id_in_partition = torch.zeros([num_nblocks], dtype = torch.long)
            bk_num_ngs_in_partition = torch.zeros([num_bk_partitions], dtype = torch.long)

            min_n_pars = 1
            for partition_id, max_n_pars in enumerate(bk_partition_max_pars):
                criterion = (n_pargs >= min_n_pars) & (n_pargs <= max_n_pars)
                partition_size = criterion.sum().item()

                bk_n_partition_ids[criterion] = partition_id
                bk_n_id_in_partition[criterion] = torch.arange(partition_size)
                bk_num_ngs_in_partition[partition_id] = partition_size

                min_n_pars = max_n_pars + 1

            # chids:      List[[partition_num_chs]]                         stores child block ids
            # parids:     List[[partition_num_chs, partition_max_n_pargs]]  stores parent node blocks' ids for each child node
            # parpids:    List[[partition_num_chs, partition_max_n_pargs]]  param id for the edges to parent (correspond to `parids`)
            curr_chids, curr_parids, curr_parpids = sum_layer_backward_compilation(
                nodes = ch_gsize2cs[ch_gsize], 
                cs2parns = cs2parns,
                n_partition_ids = bk_n_partition_ids, 
                n_id_in_partition = bk_n_id_in_partition, 
                num_ngs_in_partition = bk_num_ngs_in_partition,
                partition_max_pars = bk_partition_max_pars,
                # GPU compilation is slightly slower for small layer due to the kernel jit compilation time
                use_cuda = force_gpu_compilation or (not disable_gpu_compilation and (self.num_edges > 1000))
            )

            chids.extend(curr_chids)
            parids.extend(curr_parids)
            parpids.extend(curr_parpids)
            cs_block_sizes.extend([ch_gsize] * num_bk_partitions)

        # Store buffers for the forward pass
        self.partitioned_chids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in chids])
        self.partitioned_parids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in parids])
        self.partitioned_parpids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in parpids])
        self.cs_block_sizes = cs_block_sizes

        self.num_bk_partitions = len(chids)

        # Store pre-compiled indices from `parids` and `parpids` in the following buffer
        self._cached_bk_parids = dict()

        # Optional CUDA ele-backward fast path: per-signature (ebase, pbase, cuda_ok), per-(signature,
        # batch) autotuned choice, and a reused scratch output buffer for corruption-safe autotuning.
        self._cached_bk_ele_cuda = dict()
        self._cached_bk_ele_choice = dict()
        self._bk_ele_scratch = None

        # Optional CUDA param-backward fast path: per-partition (nbase, cbase, pbase, fbase, cuda_ok),
        # per-(partition, batch) autotuned choice, and a reused scratch param_flows buffer for
        # corruption-safe autotuning (the kernel is read-accumulate-write).
        self._cached_bk_par_cuda = dict()
        self._cached_bk_par_choice = dict()
        self._bk_par_scratch = None

    def to(self, device):
        super(SumLayer, self).to(device)

        # Move cached fw pcids to the new device
        for k, v in self._cached_fw_pcids.items():
            new_v = [tensor.to(device) for tensor in v]
            self._cached_fw_compiled_pcids[k] = new_v

    @property
    def num_parameters(self):
        return self._layer_pid_range[1] - self._layer_pid_range[0]

    @property
    def num_param_flows(self):
        return self._layer_pfid_range[1] - self._layer_pfid_range[0]

    @property
    def nid_range(self):
        return self._layer_nid_range[0], self._layer_nid_range[1]

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                force_use_bf16: bool = False, force_use_fp32: bool = False,
                propagation_alg: str = "LL", pflow_temperature: float = 1.0, **kwargs) -> None:
        """
        Computes the forward pass of a sum layer.

        Parameters:
        `node_mars`:    [num_nodes, B]
        `element_mars`: [max_num_els, B]
        `params`:       [num_params, B] or [num_params]
        """
        assert not (propagation_alg != "LL" and abs(pflow_temperature - 1.0) > 1e-6), "`pflow_temperature` can only be 1 if `propagation_alg` is not 'LL'."

        if not self.provided("fw_partition_local_ids"):
            # Evaluate the whole layer
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]
                cids = self.partitioned_cids[partition_id]
                pids = self.partitioned_pids[partition_id]

                self._forward(
                    node_mars, element_mars, params, nids, cids, pids, 
                    partition_id = partition_id, force_use_bf16 = force_use_bf16,
                    force_use_fp32 = force_use_fp32, 
                    propagation_alg = propagation_alg, 
                    pflow_temperature = pflow_temperature, **kwargs
                )

        else:
            # Partial evaluation
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]
                cids = self.partitioned_cids[partition_id]
                pids = self.partitioned_pids[partition_id]
                local_ids = self.fw_partition_local_ids[partition_id]

                self._forward(
                    node_mars, element_mars, params, 
                    nids, cids, pids, local_ids = local_ids,
                    partition_id = partition_id, force_use_bf16 = force_use_bf16,
                    force_use_fp32 = force_use_fp32,
                    propagation_alg = propagation_alg, 
                    pflow_temperature = pflow_temperature, **kwargs
                )

        return None

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                 node_mars: torch.Tensor, element_mars: torch.Tensor, 
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 allow_modify_flows: bool = False, propagation_alg: str = "LL", 
                 logspace_flows: bool = False, negate_pflows: bool = False, 
                 accumulate_ch_flows: bool = False, allow_neg_flows: bool = False,
                 force_use_fp32: bool = False, pflow_temperature: float = 1.0, 
                 temper_eflow: bool = False, **kwargs) -> None:
        """
        Computes the forward pass of a sum layer:
        ```
        element_flows[chids] = (node_flows[parids] * params[parpids] * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)
        ```
        Optionally, cumulate parameter flows:
        ```
        param_flows[seq_parpids] += (node_flows[parids] * params[parpids] * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 2)[seq_ids0, seq_ids1]
        ```

        Parameters:
        `node_flows`:    [num_nodes, B]
        `element_flows`: [max_num_els, B]
        `node_mars`:     [num_nodes, B]
        `element_mars`:  [max_num_els, B]
        `params`:        [num_params, B] or [num_params]
        """

        assert not (allow_modify_flows and logspace_flows), "`allow_modify_flows` should be set to `False` when using `logspace_flows`."
        assert not (accumulate_ch_flows and logspace_flows), "`accumulate_ch_flows` should be set to `False` when using `logspace_flows`."
        assert not (accumulate_ch_flows and allow_modify_flows), "`accumulate_ch_flows` should be set to `False` when `allow_modify_flows=True`."
        assert not (allow_neg_flows and logspace_flows), "`allow_neg_flows` should be set to `False` when using `logspace_flows`."
        assert not (allow_neg_flows and allow_modify_flows), "`allow_neg_flows` should be set to `False` when `allow_modify_flows=True`."
        assert not (propagation_alg != "LL" and abs(pflow_temperature - 1.0) > 1e-6), "`pflow_temperature` can only be 1 if `propagation_alg` is not 'LL'."
        assert logspace_flows or abs(pflow_temperature - 1.0) < 1e-6, "`pflow_temperature` can only be enabled when `logspace_flows = True`."

        # Disallow modifications of `node_flows` in case of partial evaluation
        if self.provided("bk_partition_local_ids") and allow_modify_flows:
            allow_modify_flows = False

        ## Pre-compute `nflows.log() - nmars` if needed ##
        if allow_modify_flows:
            assert not self.provided("bk_partition_local_ids"), "Must set `allow_modify_flows = False` for partial evaluation."
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]

                self._bk_triton_modify_flow(
                    node_flows, node_mars, nids, local_ids = None,
                    propagation_alg = propagation_alg, **kwargs
                )
        
        ## Compute flows w.r.t. elements (i.e., product nodes) ##
        if not self.provided("bk_partition_local_ids"):
            # Evaluate the whole layer
            for partition_id in range(self.num_bk_partitions):
                chids = self.partitioned_chids[partition_id]
                parids = self.partitioned_parids[partition_id]
                parpids = self.partitioned_parpids[partition_id]
                cs_block_size = self.cs_block_sizes[partition_id]

                self._backward(
                    node_flows, element_flows, params, node_mars, 
                    element_mars, param_flows, 
                    chids = chids, parids = parids, parpids = parpids,
                    cs_block_size = cs_block_size,
                    partition_id = partition_id,
                    allow_modify_flows = allow_modify_flows,
                    propagation_alg = propagation_alg,
                    logspace_flows = logspace_flows,
                    negate_pflows = negate_pflows, 
                    accumulate_ch_flows = accumulate_ch_flows,
                    allow_neg_flows = allow_neg_flows,
                    force_use_fp32 = force_use_fp32,
                    pflow_temperature = pflow_temperature,
                    temper_eflow = temper_eflow,
                    **kwargs
                )

        else:
            # Partial evaluation
            for partition_id in range(self.num_bk_partitions):
                chids = self.partitioned_chids[partition_id]
                parids = self.partitioned_parids[partition_id]
                parpids = self.partitioned_parpids[partition_id]
                cs_block_size = self.cs_block_sizes[partition_id]
                local_ids = self.bk_partition_local_ids[partition_id]

                self._backward(
                    node_flows, element_flows, params, node_mars,
                    element_mars, param_flows, 
                    chids = chids, parids = parids, parpids = parpids,
                    cs_block_size = cs_block_size, local_ids = local_ids,
                    partition_id = partition_id,
                    allow_modify_flows = allow_modify_flows,
                    propagation_alg = propagation_alg,
                    logspace_flows = logspace_flows,
                    negate_pflows = negate_pflows, 
                    accumulate_ch_flows = accumulate_ch_flows, 
                    allow_neg_flows = allow_neg_flows,
                    force_use_fp32 = force_use_fp32,
                    pflow_temperature = pflow_temperature,
                    temper_eflow = temper_eflow,
                    **kwargs
                )

        ## Compute flows w.r.t. sum parameters ##
        if param_flows is not None:
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]
                cids = self.partitioned_cids[partition_id]
                pids = self.partitioned_pids[partition_id]
                pfids = self.partitioned_pfids[partition_id]

                self._backward(
                    node_flows, element_flows, params, node_mars, 
                    element_mars, param_flows, nids = nids, 
                    cids = cids, pids = pids, pfids = pfids, 
                    partition_id = partition_id,
                    allow_modify_flows = allow_modify_flows,
                    propagation_alg = propagation_alg,
                    logspace_flows = logspace_flows,
                    negate_pflows = negate_pflows, 
                    allow_neg_flows = allow_neg_flows, 
                    force_use_fp32 = force_use_fp32,
                    pflow_temperature = pflow_temperature,
                    temper_eflow = temper_eflow,
                    **kwargs
                )

        return None

    def is_sum(self):
        return True

    def __repr__(self):
        return f"SumLayer(nid_range=({self._layer_nid_range[0]}, {self._layer_nid_range[1]}), num_nodes={self.num_nodes}, num_edges={self.num_edges})"

    def _forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                 pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                 partition_id: int = -1, mode: Optional[str] = None,
                 force_use_bf16: bool = False, force_use_fp32: bool = False,
                 propagation_alg: str = "LL", pflow_temperature: float = 1.0, **kwargs) -> None:
        """
        Forward pass of sum layers.
        
        Parameters:
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `params`:       [E]
        `nids`:         [ng]
        `cids`:         [ng, c]
        `pids`:         [ng, c]
        """

        num_edges = cids.size(1)
        batch_size = node_mars.size(1)

        if mode is not None:
            assert mode in STR2MODE
            mode = self.STR2MODE[mode]

        elif params.dim() == 1 and self.block_size >= 16 and num_edges >= 16 and batch_size >= 16:
            # In this case, we should definitely use the block-sparse implementation
            mode = self.BLOCK_SPARSE
        elif self.block_size == 1 or num_edges < 4:
            # In this case, we should definitely use the sparse implementation
            mode = self.SPARSE
        elif self.block_size * batch_size < 32 or batch_size < 4:
            # Advantage of block-sparse processing is diminishing
            mode = self.SPARSE
        else:
            mode = self.BLOCK_SPARSE

        if mode == self.BLOCK_SPARSE:
            self._forward_block_sparse(
                node_mars, element_mars, params, nids, cids, pids, local_ids,
                partition_id = partition_id, force_use_bf16 = force_use_bf16,
                force_use_fp32 = force_use_fp32, propagation_alg = propagation_alg, 
                pflow_temperature = pflow_temperature, **kwargs
            )

        elif mode == self.SPARSE:
            self._forward_sparse(
                node_mars, element_mars, params, nids, cids, pids, local_ids,
                partition_id = partition_id, propagation_alg = propagation_alg, 
                pflow_temperature = pflow_temperature, **kwargs
            )

        elif mode == self.PYTORCH:
            assert abs(pflow_temperature - 1.0) < 1e-6, "`pflow_temperature != 1.0` not supported by the PyTorch backend."

            self._forward_pytorch(
                node_mars, element_mars, params, nids, cids, pids, local_ids,
                propagation_alg = propagation_alg, **kwargs
            )
        
        else:
            raise ValueError(f"Unexpected mode `{mode}`.")

    def _forward_block_sparse(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                              params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                              pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                              partition_id: int = -1, force_use_bf16: bool = False,
                              force_use_fp32: bool = False, propagation_alg: str = "LL", 
                              pflow_temperature: float = 1.0, **kwargs) -> None:
        """
        Forward pass of sum layers with the block-sparse processing kernel.
        
        Parameters:
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `params`:       [E]
        `nids`:         [ng]
        `cids`:         [ng, c]
        `pids`:         [ng, c]
        """

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_nblocks = nids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * self.block_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        # Tempered pflow
        if abs(pflow_temperature - 1.0) < 1e-6:
            pflow_tempered_enabled = False
            pflow_tempered_kwargs = {}
        else:
            assert "node_mars_tempered" in kwargs
            pflow_tempered_enabled = True
            pflow_tempered_kwargs = {
                "pflow_temperature": pflow_temperature,
                "node_mars_tempered": kwargs["node_mars_tempered"]
            }

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.block_size, num_edges, BATCH_SIZE_NP2, 128)
        if base_size >= 64:
            TILE_SIZE_K = min(2048 // 32, num_edges)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_K = min(2048 // remainder, base_size * remainder, num_edges)
        # Larger TILE_SIZE_M / BLOCK_B (bit-exact) amortize the per-m-tile recomputation of
        # `emars_max` / `exp` / the `emars` load and the per-batch-tile `epars` load. Gated to the
        # LL (non-tempered) regime that was validated bit-exact; auto-disabled on a prior OOM.
        _fw_tile_cap = 4096 if (FORWARD_SUM_TUNED and not getattr(self, "_fw_tuning_oom", False)
                                and propagation_alg_id == 0 and not pflow_tempered_enabled) else 2048
        TILE_SIZE_M = min(_fw_tile_cap // TILE_SIZE_K, self.block_size)
        BLOCK_B = min(_fw_tile_cap // TILE_SIZE_K, BATCH_SIZE_NP2)
        K_NUM_TILES = num_edges // TILE_SIZE_K

        assert TILE_SIZE_K >= 4, f"`TILE_SIZE_K` should be greater than 4 (but got {TILE_SIZE_K}) in order to use the block-sparse kernel. " \
                                  "This is an internal error of PyJuice. Please consider checking the kernel dispatching criterions and use the " \
                                  "corresponding sparse kernel instead."

        signature = ("block_sparse", partition_id, TILE_SIZE_K)
        if signature not in self._cached_fw_pcids:
            # Pre-compute pointer increments for `cids` and `pids`

            cids = cids.clone().reshape(cids.size(0), K_NUM_TILES, TILE_SIZE_K)
            cids_start = cids[:,0,:].contiguous()
            cids_increment = torch.cat(
                (cids[:,1:,:] - cids[:,:-1,:], cids[:,0:1,:] * 0), 
                dim = 1
            ).contiguous()

            pids = pids.clone().reshape(pids.size(0), K_NUM_TILES, TILE_SIZE_K)
            pids_start = pids[:,0,:].contiguous()
            pids_increment = torch.cat(
                (pids[:,1:,:] - pids[:,:-1,:], pids[:,0:1,:] * 0),
                dim = 1
            ).contiguous()

            self._cached_fw_pcids[signature] = [cids_start, cids_increment, pids_start, pids_increment]

            # Pre-compute the CUDA fast-path operands + validate its layout assumptions (once).
            # `ebase` / `pbase`: per-tile first child-node index / first param offset. The kernel
            # reads element_mars[ebase + e, b] and params[pbase + e * block_size + m], i.e. it
            # assumes the tile's children are CONTIGUOUS node rows and its params stride by
            # `block_size` across edges. Record whether both hold so the dispatcher can fall back.
            ebase = cids[:, :, 0].contiguous().to(torch.int64)   # [ng, K_NUM_TILES]
            pbase = pids[:, :, 0].contiguous().to(torch.int64)   # [ng, K_NUM_TILES]
            _ar = torch.arange(TILE_SIZE_K, device=cids.device, dtype=torch.int64)
            cids_contig = torch.equal(cids.to(torch.int64), ebase.unsqueeze(-1) + _ar.view(1, 1, -1))
            pids_strided = torch.equal(pids.to(torch.int64),
                                       pbase.unsqueeze(-1) + _ar.view(1, 1, -1) * self.block_size)
            cuda_ok = bool(cids_contig and pids_strided)
            self._cached_fw_cuda[signature] = [ebase, pbase, cuda_ok]
        else:
            cids_start, cids_increment, pids_start, pids_increment = self._cached_fw_pcids[signature]

        partial_eval = 1 if local_ids is not None else 0
        BLOCK_SIZE_M = self.block_size

        if force_use_bf16:
            assert not force_use_fp32
            use_bf16 = True
        elif force_use_fp32:
            use_bf16 = False
        else:
            if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
                use_bf16 = True
            else:
                use_bf16 = False

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

        # Optional CUDA (CuTe/TMA) fast path for the `tlmm` regime. It is numerically equivalent to
        # the Triton tlmm kernel and only valid here: LL propagation (`propagation_alg_id == 0`), the
        # bf16 dot path (`use_bf16`), no partial eval (`local_ids is None` <=> `partial_eval == 0`),
        # no pflow tempering, TILE_SIZE_K == 64 with num_edges a multiple of it, and a contiguous
        # edge/param layout (`cuda_ok`). `is_available()` JIT-compiles on first call and self-disables
        # (-> Triton) on an unsuitable GPU/toolchain/CUTLASS, so this is a no-op without the CUDA
        # prerequisites. The best tile config (or Triton) is autotuned once per layer signature.
        if (FORWARD_SUM_CUDA and use_bf16 and propagation_alg_id == 0
                and not pflow_tempered_enabled and local_ids is None
                and TILE_SIZE_K == 64 and num_edges % TILE_SIZE_K == 0
                and node_mars.is_cuda and cuda_kernels.is_available()):
            ebase, pbase, cuda_ok = self._cached_fw_cuda[signature]
            valid_cfgs = cuda_kernels.valid_configs(self.block_size, batch_size) if cuda_ok else []
            if valid_cfgs:
                # The best config depends on batch_size (which tile shapes are valid + their speed),
                # so the autotuned choice is keyed by (signature, batch_size), not signature alone.
                choice_key = (signature, batch_size)
                choice = self._cached_fw_cuda_choice.get(choice_key)
                if choice is None:
                    # Autotune once: fastest of {valid CUDA tile configs} vs Triton. Every candidate
                    # computes the same result into `node_mars`, so it stays correct afterwards.
                    cands = [(("cuda", c),
                              (lambda c=c: cuda_kernels.tlmm_forward_sum(
                                  node_mars, element_mars, params, nids, ebase, pbase,
                                  batch_size, self.block_size, K_NUM_TILES, c)))
                             for c in valid_cfgs]

                    def _triton_tlmm_cand():
                        for s in range(0, grid[1], 32768):
                            cg = (grid[0], min(s + 32768, grid[1]) - s)
                            fw_bsparse._fw_triton_block_sparse_tlmm_kernel[cg](
                                node_mars, element_mars, params, nids, cids_start, cids_increment,
                                pids_start, pids_increment, local_ids, batch_size,
                                partial_eval = partial_eval, BLOCK_B = BLOCK_B,
                                TILE_SIZE_K = TILE_SIZE_K, K_NUM_TILES = K_NUM_TILES,
                                TILE_SIZE_M = TILE_SIZE_M, BLOCK_SIZE_M = BLOCK_SIZE_M,
                                use_bf16 = use_bf16, propagation_alg_id = propagation_alg_id,
                                pflow_tempered_enabled = pflow_tempered_enabled, pid_m_offset = s,
                                **propagation_alg_kwargs, **pflow_tempered_kwargs, num_stages = 1)
                    cands.append((("triton", -1), _triton_tlmm_cand))
                    choice = cuda_kernels.autotune(cands) or ("triton", -1)
                    self._cached_fw_cuda_choice[choice_key] = choice

                if choice[0] == "cuda":
                    cuda_kernels.tlmm_forward_sum(
                        node_mars, element_mars, params, nids, ebase, pbase,
                        batch_size, self.block_size, K_NUM_TILES, choice[1])
                    return None
                # choice == ("triton", -1): fall through to the Triton launch below
        
        # OOM-safe tuned launch: if the larger tuned tiles exceed this GPU's shared-memory/
        # register budget, fall back to the default configuration (recompiled untuned).
        try:
            for pid_m_start in range(0, grid[1], 32768):
                pid_m_end = min(pid_m_start + 32768, grid[1])
                block_m_size = pid_m_end - pid_m_start

                curr_grid = (grid[0], block_m_size)
        
                if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
                    fw_bsparse._fw_triton_block_sparse_tlmm_kernel[curr_grid](
                        node_mars, 
                        element_mars, 
                        params, 
                        nids, 
                        cids_start,
                        cids_increment, 
                        pids_start,
                        pids_increment,
                        local_ids,
                        batch_size,
                        partial_eval = partial_eval,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_K = TILE_SIZE_K,
                        K_NUM_TILES = K_NUM_TILES,
                        TILE_SIZE_M = TILE_SIZE_M,
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        use_bf16 = use_bf16,
                        propagation_alg_id = propagation_alg_id,
                        pflow_tempered_enabled = pflow_tempered_enabled,
                        pid_m_offset = pid_m_start,
                        **propagation_alg_kwargs,
                        **pflow_tempered_kwargs,
                        num_stages = 1
                    )
                
                elif TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and BLOCK_B >= 8:
                    fw_bsparse._fw_triton_block_sparse_csmm1_kernel[curr_grid](
                        node_mars, 
                        element_mars, 
                        params, 
                        nids, 
                        cids_start,
                        cids_increment, 
                        pids_start,
                        pids_increment,
                        local_ids,
                        batch_size,
                        partial_eval = partial_eval,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_K = TILE_SIZE_K,
                        K_NUM_TILES = K_NUM_TILES,
                        TILE_SIZE_M = TILE_SIZE_M,
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        use_bf16 = use_bf16,
                        propagation_alg_id = propagation_alg_id,
                        pflow_tempered_enabled = pflow_tempered_enabled,
                        pid_m_offset = pid_m_start,
                        **propagation_alg_kwargs,
                        **pflow_tempered_kwargs,
                        num_stages = 1
                    )

                else:
                    fw_bsparse._fw_triton_block_sparse_csmm2_kernel[curr_grid](
                        node_mars, 
                        element_mars, 
                        params, 
                        nids, 
                        cids_start,
                        cids_increment, 
                        pids_start,
                        pids_increment,
                        local_ids,
                        batch_size,
                        partial_eval = partial_eval,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_K = TILE_SIZE_K,
                        K_NUM_TILES = K_NUM_TILES,
                        TILE_SIZE_M = TILE_SIZE_M,
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        use_bf16 = use_bf16,
                        propagation_alg_id = propagation_alg_id,
                        pflow_tempered_enabled = pflow_tempered_enabled,
                        pid_m_offset = pid_m_start,
                        **propagation_alg_kwargs,
                        **pflow_tempered_kwargs,
                        num_stages = 1
                    )
        except _TritonOutOfResources:
            # `OutOfResources` is raised at compile time before any write, so retry is safe.
            if not (FORWARD_SUM_TUNED and not getattr(self, "_fw_tuning_oom", False)
                    and propagation_alg_id == 0 and not pflow_tempered_enabled):
                raise
            self._fw_tuning_oom = True
            warnings.warn("Forward sum-layer tile tuning (FORWARD_SUM_TUNED) exceeded shared "
                          "memory on this GPU; falling back to the default launch config.", RuntimeWarning)
            return self._forward_block_sparse(
                node_mars, element_mars, params, nids, cids, pids, local_ids=local_ids,
                partition_id=partition_id, force_use_bf16=force_use_bf16,
                force_use_fp32=force_use_fp32, propagation_alg=propagation_alg,
                pflow_temperature=pflow_temperature, **kwargs)
        return None

    def _forward_sparse(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                        params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                        pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                        partition_id: int = -1, propagation_alg: str = "LL",
                        pflow_temperature: float = 1.0, **kwargs) -> None:
        """
        Forward pass of sum layers with the sparse processing kernel.
        
        Parameters:
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `params`:       [E]
        `nids`:         [ng]
        `cids`:         [ng, c]
        `pids`:         [ng, c]
        """

        num_nblocks = nids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * self.block_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        # Tempered pflow
        if abs(pflow_temperature - 1.0) < 1e-6:
            pflow_tempered_enabled = False
            pflow_tempered_kwargs = {}
        else:
            assert "node_mars_tempered" in kwargs
            pflow_tempered_enabled = True
            pflow_tempered_kwargs = {
                "pflow_temperature": pflow_temperature,
                "node_mars_tempered": kwargs["node_mars_tempered"]
            }

        if triton.cdiv(layer_n_nodes, self.block_size) <= 2048:
            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)

            partial_eval = 1 if local_ids is not None else 0
            BLOCK_SIZE_M = self.block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, BLOCK_SIZE_M))

            fw_sparse._fw_triton_sparse_kernel[grid](
                node_mars = node_mars, 
                element_mars = element_mars, 
                mparams = params, 
                nids = nids, 
                cids = cids,
                pids = pids,
                local_ids = local_ids, 
                batch_size = batch_size, 
                partial_eval = partial_eval, 
                num_edges = num_edges, 
                BLOCK_B = BLOCK_B, 
                BLOCK_SIZE_M = BLOCK_SIZE_M,
                propagation_alg_id = propagation_alg_id,
                pflow_tempered_enabled = pflow_tempered_enabled,
                **propagation_alg_kwargs,
                **pflow_tempered_kwargs
            )

        else:
            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            TILE_SIZE_M = max(min(4096 // num_edges // BLOCK_B, triton.next_power_of_2(layer_n_nodes)), 1)

            partial_eval = 1 if local_ids is not None else 0
            BLOCK_SIZE_M = self.block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            if grid[1] <= 32768:
                fw_sparse._fw_triton_large_sparse_kernel[grid](
                    node_mars = node_mars,
                    element_mars = element_mars,
                    mparams = params,
                    nids = nids,
                    cids = cids,
                    pids = pids,
                    local_ids = local_ids,
                    batch_size = batch_size,
                    num_nodes = layer_n_nodes,
                    pid_m_offset = 0,
                    partial_eval = partial_eval,
                    num_edges = num_edges,
                    BLOCK_B = BLOCK_B,
                    TILE_SIZE_M = TILE_SIZE_M,
                    BLOCK_SIZE_M = BLOCK_SIZE_M,
                    propagation_alg_id = propagation_alg_id,
                    pflow_tempered_enabled = pflow_tempered_enabled,
                    **propagation_alg_kwargs,
                    **pflow_tempered_kwargs
                )
            else:
                for pid_m_start in range(0, grid[1], 32768):

                    pid_m_end = min(pid_m_start + 32768, grid[1])
                    small_grid = (grid[0], pid_m_end - pid_m_start)

                    fw_sparse._fw_triton_large_sparse_kernel[small_grid](
                        node_mars = node_mars,
                        element_mars = element_mars,
                        mparams = params,
                        nids = nids,
                        cids = cids,
                        pids = pids,
                        local_ids = local_ids,
                        batch_size = batch_size,
                        num_nodes = layer_n_nodes,
                        pid_m_offset = pid_m_start,
                        partial_eval = partial_eval,
                        num_edges = num_edges,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_M = TILE_SIZE_M,
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        propagation_alg_id = propagation_alg_id,
                        pflow_tempered_enabled = pflow_tempered_enabled,
                        **propagation_alg_kwargs,
                        **pflow_tempered_kwargs
                    )

        return None

    @staticmethod
    @torch.compile
    def _forward_pytorch_kernel(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, 
                                nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor,
                                local_ids: torch.Tensor, propagation_alg_id: int, alpha: float = 0.0):

        if local_ids is not None:
            nids = nids[local_ids]
            cids = cids[local_ids]
            pids = pids[local_ids]

        num_nblocks = nids.size(0)
        num_edges = cids.size(1)
        nids = (nids[:,None].repeat(1, self.block_size) + \
            torch.arange(0, self.block_size, device = nids.device)[None,:]).reshape(num_nblocks * self.block_size)
        cids = cids[:,None,:].repeat(1, self.block_size, 1).reshape(num_nblocks * self.block_size, num_edges)
        pids = (pids[:,None,:].repeat(1, self.block_size, 1) + \
            torch.arange(0, self.block_size, device = cids.device)[None,:,None]).reshape(num_nblocks * self.block_size, num_edges)

        ch_mars = element_mars[cids]

        if propagation_alg_id == 0:
            maxval = ch_mars.max(dim = 1, keepdim = True).values
            node_mars[nids] = (((ch_mars - maxval).exp() * params[pids].unsqueeze(-1)).sum(
                dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

        elif propagation_alg_id == 1:
            node_mars[nids] = (ch_mars + params[pids].log().unsqueeze(-1)).max(dim = 1).values

        elif propagation_alg_id == 2:
            maxval = ch_mars.max(dim = 1, keepdim = True).values
            node_mars[nids] = ((((ch_mars - maxval).exp() * params[pids].unsqueeze(-1)) ** alpha).sum(
                dim = 1).clamp(min = 1e-10)).log() ** (1.0 / alpha) + maxval.squeeze(1)

        return None

    def _forward_pytorch(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, 
                         nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor,
                         local_ids: torch.Tensor, propagation_alg: str = "LL", **kwargs):

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        self._forward_pytorch_kernel(
            node_mars, element_mars, params, nids, cids, pids, local_ids,
            propagation_alg_id = propagation_alg_id, **propagation_alg_kwargs
        )

    def _backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                  params: torch.Tensor, node_mars: torch.Tensor, 
                  element_mars: torch.Tensor, param_flows: torch.Tensor,
                  nids: Optional[torch.Tensor] = None, cids: Optional[torch.Tensor] = None, 
                  pids: Optional[torch.Tensor] = None, pfids: Optional[torch.Tensor] = None, 
                  chids: Optional[torch.Tensor] = None, parids: Optional[torch.Tensor] = None, 
                  parpids: Optional[torch.Tensor] = None, 
                  cs_block_size: int = 0, local_ids: Optional[torch.Tensor] = None, 
                  partition_id: int = -1, mode: Optional[str] = None,
                  allow_modify_flows: bool = False,
                  propagation_alg: str = "LL", 
                  logspace_flows: bool = False, 
                  negate_pflows: bool = False, 
                  accumulate_ch_flows: bool = False, 
                  allow_neg_flows: bool = False, 
                  force_use_fp32: bool = False, 
                  pflow_temperature: float = 1.0, 
                  temper_eflow: bool = False, **kwargs) -> None:
        """
        Back pass of sum layers.
        
        Parameters:
        `node_flows`:   [N, B]
        `element_flows: [M, B]
        `params`:       [E]
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `param_flows`:  [E]
        `chids`:        [ng]
        `parids`:       [ng, c]
        `parpids`:      [ng, c]
        """

        if cids is not None:
            num_edges = cids.size(1) * self.block_size
        else:
            num_edges = parids.size(1) * self.block_size
        batch_size = node_flows.size(1)

        if mode is not None:
            assert mode in STR2MODE
            mode = self.STR2MODE[mode]

        elif params.dim() == 1 and self.block_size >= 16 and num_edges >= 16 and batch_size >= 16:
            # In this case, we should definitely use the block-sparse implementation
            mode = self.BLOCK_SPARSE
        elif cs_block_size == 1 or self.block_size == 1:
            # In this case, we should definitely use the sparse implementation
            mode = self.SPARSE
        elif self.block_size * batch_size < 32 or batch_size < 4:
            # Advantage of block-sparse processing is diminishing
            mode = self.SPARSE
        elif num_edges <= 32768:
            mode = self.BLOCK_SPARSE
        else:
            mode = self.BLOCK_SPARSE

        if mode == self.BLOCK_SPARSE:
            self._backward_block_sparse(
                node_flows, element_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, chids, parids, parpids, cs_block_size, local_ids, 
                partition_id = partition_id, allow_modify_flows = allow_modify_flows,
                propagation_alg = propagation_alg, logspace_flows = logspace_flows, 
                negate_pflows = negate_pflows, accumulate_ch_flows = accumulate_ch_flows, 
                allow_neg_flows = allow_neg_flows, force_use_fp32 = force_use_fp32, 
                pflow_temperature = pflow_temperature, temper_eflow = temper_eflow, **kwargs
            )

        elif mode == self.SPARSE:
            self._backward_sparse(
                node_flows, element_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, chids, parids, parpids, cs_block_size, local_ids, 
                partition_id = partition_id, allow_modify_flows = allow_modify_flows,
                propagation_alg = propagation_alg, logspace_flows = logspace_flows, 
                negate_pflows = negate_pflows, accumulate_ch_flows = accumulate_ch_flows, 
                pflow_temperature = pflow_temperature, temper_eflow = temper_eflow, **kwargs
            )

        elif mode == self.PYTORCH:
            assert not allow_modify_flows, "Please set `allow_modify_flows` to False when " \
                                           "using the native PyTorch backward."
            assert abs(pflow_temperature - 1.0) < 1e-6, "`pflow_temperature != 0` not supported with PyTorch backend."
            assert not temper_eflow, "`temper_eflow == True` not supported with PyTorch backend."
            self._backward_pytorch(
                node_flows, element_flows, params, node_mars, 
                element_mars, param_flows, nids, cids, pids, pfids, 
                chids, parids, parpids, cs_block_size,
                propagation_alg = propagation_alg, 
                negate_pflows = negate_pflows, accumulate_ch_flows = accumulate_ch_flows, **kwargs
            )
        else:
            raise ValueError(f"Not supported mode `{mode}`.")

        return None

    def _bk_triton_modify_flow(self, node_flows: torch.Tensor, node_mars: torch.Tensor,
                               nids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                               propagation_alg: str = "LL", **kwargs):
        """
        Replace `node_flows[nids]` with `node_flows[nids].log() - node_mars[nids]`
        """

        num_nblocks = nids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * self.block_size
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        if triton.cdiv(layer_n_nodes, self.block_size) <= 4096:

            if BATCH_SIZE_NP2 >= 64 and self.block_size >= 64:
                BLOCK_B = min(2048 // 64, BATCH_SIZE_NP2)
                BLOCK_M = min(4096 // BLOCK_B, self.block_size)
            else:
                BLOCK_B = min(2048, BATCH_SIZE_NP2)
                BLOCK_M = min(2048 // BLOCK_B, self.block_size)

            partial_eval = 1 if local_ids is not None else 0
            BLOCK_SIZE_M = self.block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, BLOCK_M))

            bk_nflows._bk_triton_modify_flow_kernel[grid](
                node_flows = node_flows, 
                node_mars = node_mars, 
                local_ids = local_ids, 
                nids = nids, 
                batch_size = batch_size, 
                partial_eval = partial_eval, 
                BLOCK_B = BLOCK_B, 
                BLOCK_M = BLOCK_M, 
                BLOCK_SIZE_M = BLOCK_SIZE_M,
                propagation_alg_id = propagation_alg_id,
                **propagation_alg_kwargs
            )

        else:

            BLOCK_B = min(2048, BATCH_SIZE_NP2)
            TILE_SIZE_M = min(4096 // BLOCK_B, triton.next_power_of_2(layer_n_nodes))

            partial_eval = 1 if local_ids is not None else 0
            BLOCK_SIZE_M = self.block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            for pid_m_start in range(0, grid[1], 32768):
                pid_m_end = min(pid_m_start + 32768, grid[1])
                block_m_size = pid_m_end - pid_m_start

                curr_grid = (grid[0], block_m_size)

                bk_nflows._bk_triton_large_modify_flow_kernel[grid](
                    node_flows = node_flows,
                    node_mars = node_mars,
                    local_ids = local_ids,
                    nids = nids,
                    num_nodes = layer_n_nodes,
                    batch_size = batch_size,
                    partial_eval = partial_eval,
                    BLOCK_B = BLOCK_B,
                    TILE_SIZE_M = TILE_SIZE_M,
                    BLOCK_SIZE_M = BLOCK_SIZE_M,
                    propagation_alg_id = propagation_alg_id,
                    pid_m_offset = pid_m_start,
                    **propagation_alg_kwargs
                )

        return None

    def _backward_block_sparse(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                               params: torch.Tensor, node_mars: torch.Tensor, 
                               element_mars: torch.Tensor, param_flows: torch.Tensor, 
                               nids: Optional[torch.Tensor], cids: Optional[torch.Tensor], pids: Optional[torch.Tensor], pfids: Optional[torch.Tensor],
                               chids: Optional[torch.Tensor], parids: Optional[torch.Tensor], parpids: Optional[torch.Tensor], 
                               cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                               partition_id: int = -1, allow_modify_flows: bool = False, propagation_alg: str = "LL", 
                               logspace_flows: bool = False, negate_pflows: bool = False, accumulate_ch_flows: bool = False, 
                               allow_neg_flows: bool = False, force_use_fp32 = False, pflow_temperature: float = 1.0, 
                               temper_eflow: bool = False, **kwargs) -> None:
        """
        Back pass of sum layers with block-sparse processing kernel.
        
        Parameters:
        `node_flows`:   [N, B]
        `element_flows: [M, B]
        `params`:       [E]
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `param_flows`:  [E]
        `chids`:        [ng]
        `parids`:       [ng, c]
        `parpids`:      [ng, c]
        """

        # Flows w.r.t. input elements (product nodes)
        if chids is not None:
            self._backward_block_sparse_ele_flows(
                node_flows, element_flows, params, node_mars, element_mars,
                chids = chids, parids = parids, parpids = parpids, 
                cs_block_size = cs_block_size, local_ids = local_ids, 
                partition_id = partition_id, allow_modify_flows = allow_modify_flows,
                propagation_alg = propagation_alg, 
                logspace_flows = logspace_flows, 
                accumulate_ch_flows = accumulate_ch_flows, 
                allow_neg_flows = allow_neg_flows, 
                force_use_fp32 = force_use_fp32, 
                eflow_temperature = pflow_temperature if temper_eflow else 1.0, 
                **kwargs
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_block_sparse_par_flows(
                node_flows, params, node_mars, element_mars, param_flows,
                nids = nids, cids = cids, pids = pids, pfids = pfids,
                allow_modify_flows = allow_modify_flows,
                propagation_alg = propagation_alg, 
                logspace_flows = logspace_flows, 
                negate_pflows = negate_pflows, 
                allow_neg_flows = allow_neg_flows, 
                pflow_temperature = pflow_temperature, **kwargs
            )

        return None

    def _backward_block_sparse_ele_flows(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                                         params: torch.Tensor, node_mars: torch.Tensor,
                                         element_mars: torch.Tensor, chids: torch.Tensor, parids: torch.Tensor,
                                         parpids: torch.Tensor, cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                                         partition_id: int = -1, allow_modify_flows: bool = False,
                                         propagation_alg: str = "LL", logspace_flows: bool = False, 
                                         accumulate_ch_flows: bool = False, allow_neg_flows: bool = False, 
                                         force_use_fp32: bool = False, eflow_temperature: float = 1.0, **kwargs) -> None:

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_nblocks = chids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * cs_block_size
        num_edges = parids.size(1) * self.block_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.block_size, num_edges, BATCH_SIZE_NP2, 64)
        if base_size >= 64:
            TILE_SIZE_K = min(2048 // 32, num_edges)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_K = min(512, base_size * remainder, num_edges)
        TILE_SIZE_M = min(2048 // TILE_SIZE_K, cs_block_size)
        BLOCK_B = min(2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        K_NUM_TILES = num_edges // TILE_SIZE_K

        assert TILE_SIZE_K >= 4, f"`TILE_SIZE_K` should be greater than 4 (but got {TILE_SIZE_K}) in order to use the block-sparse kernel. " \
                                  "This is an internal error of PyJuice. Please consider checking the kernel dispatching criterions and use the " \
                                  "corresponding sparse kernel instead."

        signature = ("block_sparse", partition_id, TILE_SIZE_K)
        if signature not in self._cached_bk_parids:
            # Pre-compute pointer increments for `parids` and `parpids`

            if TILE_SIZE_K < self.block_size:
                ptr_inc_step = 1

                num_rep = self.block_size // TILE_SIZE_K
                parids = (parids[:,:,None].repeat(1, 1, num_rep) + \
                    torch.arange(0, self.block_size, TILE_SIZE_K, device = parids.device)[None,None,:]).reshape(
                        parids.size(0), K_NUM_TILES, 1)
                parpids = (parpids[:,:,None].repeat(1, 1, num_rep) + \
                    torch.arange(0, self.block_size, TILE_SIZE_K, device = parpids.device)[None,None,:]).reshape(
                        parpids.size(0), K_NUM_TILES, 1)

            else:
                ptr_inc_step = TILE_SIZE_K // self.block_size

                parids = parids.reshape(parids.size(0), K_NUM_TILES, ptr_inc_step)
                parpids = parpids.reshape(parpids.size(0), K_NUM_TILES, ptr_inc_step)

            parids_start = parids[:,0,:].contiguous()
            parids_increment = torch.cat(
                (parids[:,1:,:] - parids[:,:-1,:], parids[:,0:1,:] * 0),
                dim = 1
            ).contiguous()

            parpids_start = parpids[:,0,:].contiguous()
            parpids_increment = torch.cat(
                (parpids[:,1:,:] - parpids[:,:-1,:], parpids[:,0:1,:] * 0),
                dim = 1
            ).contiguous()

            self._cached_bk_parids[signature] = [parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step]

            # Pre-compute the CUDA fast-path operands: per-tile first parent-node index (`ebase`) and
            # first param offset (`pbase`) = cumsum-reconstruction of the parids / parpids starts +
            # increments. The kernel reads node_*[ebase + e] and mp[pbase + m*BLOCK_SIZE_K + e] over
            # the tile's BK contiguous edges; that contiguity is structural when ptr_inc_step == 1
            # (single param-block group per tile), which is the only layout the CUDA kernel supports.
            def _cumbase(start, incr):
                c = torch.cumsum(incr.to(torch.int64), 1)
                sh = torch.zeros_like(c); sh[:, 1:] = c[:, :-1]
                return (start[:, None, :].to(torch.int64) + sh)[:, :, 0].contiguous()
            ele_ebase = _cumbase(parids_start, parids_increment)    # [n_eleblocks, K_NUM_TILES]
            ele_pbase = _cumbase(parpids_start, parpids_increment)
            ele_cuda_ok = (ptr_inc_step == 1)
            self._cached_bk_ele_cuda[signature] = [ele_ebase, ele_pbase, ele_cuda_ok]
        else:
            parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step = self._cached_bk_parids[signature]

        partial_eval = 1 if local_ids is not None else 0
        BLOCK_SIZE_M = cs_block_size
        BLOCK_SIZE_K = self.block_size
        allow_modify_flows = 1 if allow_modify_flows else 0

        # Bit-exact tuning: a larger TILE_SIZE_M (element-output tiling only -> identical results)
        # improves throughput in the LL block-sparse-dot regime. See BACKWARD_ELE_FLOW_TUNED.
        if BACKWARD_ELE_FLOW_TUNED and propagation_alg_id == 0 and abs(eflow_temperature - 1.0) < 1e-6 \
                and TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16 \
                and 2 * TILE_SIZE_M <= cs_block_size:
            TILE_SIZE_M = 2 * TILE_SIZE_M

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16 and not force_use_fp32:
            TL_DOT = 1
        else:
            TL_DOT = 0

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

        # Optional CUDA (CuTe/fp16/TMA) fast path for the element-flow backward `tlmm` regime. It is
        # numerically equivalent to the Triton ele kernel (fp16 dot + fp32 accumulate; ~1.07e-3
        # log-space) and only valid here: LL, logspace flows, allow_modify_flows / allow_neg_flows /
        # accumulate_ch_flows all off, no tempering / partial eval, the bf16/fp16 dot regime
        # (TL_DOT, not force_use_fp32), TILE_SIZE_K == 64, ptr_inc_step == 1 (contiguous layout),
        # block_size % 128 == 0, batch % 64 == 0, and a TMA-capable GPU with CUTLASS. The best of
        # {CUDA, Triton} is autotuned once per (signature, batch) -- INTO A SCRATCH buffer, never the
        # live element_flows (which is read-accumulate-write when accumulate_ch_flows is on) -- and
        # cached. Otherwise it falls through to the Triton launch below.
        if (BACKWARD_ELE_FLOW_CUDA and propagation_alg_id == 0 and abs(eflow_temperature - 1.0) < 1e-6
                and allow_modify_flows == 0 and logspace_flows and not allow_neg_flows
                and not accumulate_ch_flows and local_ids is None and not force_use_fp32
                and TILE_SIZE_K == 64 and num_edges % TILE_SIZE_K == 0
                and cs_block_size % 128 == 0 and batch_size % 64 == 0
                and node_flows.is_cuda and cuda_kernels.ele_is_available()):
            ele_ebase, ele_pbase, ele_cuda_ok = self._cached_bk_ele_cuda[signature]
            if ele_cuda_ok:
                def _cuda_ele(tgt):
                    cuda_kernels.ele_backward_sum(
                        tgt, element_mars, node_flows, node_mars, params, chids, ele_ebase, ele_pbase,
                        batch_size, self.block_size, cs_block_size, K_NUM_TILES)

                def _triton_ele(tgt):
                    for s in range(0, grid[1], 32768):
                        cg = (grid[0], min(s + 32768, grid[1]) - s)
                        bk_ele_bsparse._bk_triton_block_sparse_ele_kernel[cg](
                            node_flows = node_flows, element_flows = tgt, node_mars = node_mars,
                            element_mars = element_mars, mparams = params, chids = chids,
                            parids_start = parids_start, parids_increment = parids_increment,
                            parpids_start = parpids_start, parpids_increment = parpids_increment,
                            local_ids = local_ids, batch_size = batch_size, partial_eval = partial_eval,
                            ptr_inc_step = ptr_inc_step, allow_modify_flows = allow_modify_flows,
                            logspace_flows = logspace_flows, BLOCK_B = BLOCK_B, TILE_SIZE_K = TILE_SIZE_K,
                            K_NUM_TILES = K_NUM_TILES, TILE_SIZE_M = TILE_SIZE_M,
                            BLOCK_SIZE_M = BLOCK_SIZE_M, BLOCK_SIZE_K = BLOCK_SIZE_K, TL_DOT = TL_DOT,
                            num_stages = 1, propagation_alg_id = propagation_alg_id,
                            accumulate_ch_flows = accumulate_ch_flows, allow_neg_flows = allow_neg_flows,
                            pid_m_offset = s, **propagation_alg_kwargs)

                choice_key = (signature, batch_size)
                choice = self._cached_bk_ele_choice.get(choice_key)
                if choice is None:
                    # autotune into a SCRATCH clone (corruption-safe); the live element_flows is
                    # touched only by the single real run below.
                    if (self._bk_ele_scratch is None or self._bk_ele_scratch.shape != element_flows.shape
                            or self._bk_ele_scratch.dtype != element_flows.dtype):
                        self._bk_ele_scratch = torch.empty_like(element_flows)
                    scr = self._bk_ele_scratch
                    choice = cuda_kernels.autotune(
                        [(("cuda", 0), (lambda: _cuda_ele(scr))),
                         (("triton", -1), (lambda: _triton_ele(scr)))]) or ("triton", -1)
                    self._cached_bk_ele_choice[choice_key] = choice
                if choice[0] == "cuda":
                    _cuda_ele(element_flows)
                    return None
                # choice == ("triton", -1): fall through to the Triton launch below

        for pid_m_start in range(0, grid[1], 32768):
            pid_m_end = min(pid_m_start + 32768, grid[1])
            block_m_size = pid_m_end - pid_m_start

            curr_grid = (grid[0], block_m_size)

            if abs(eflow_temperature - 1.0) < 1e-6:

                if TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and BLOCK_B >= 8:
                    bk_ele_bsparse._bk_triton_block_sparse_ele_kernel[curr_grid](
                        node_flows = node_flows, 
                        element_flows = element_flows, 
                        node_mars = node_mars, 
                        element_mars = element_mars, 
                        mparams = params, 
                        chids = chids, 
                        parids_start = parids_start,
                        parids_increment = parids_increment,
                        parpids_start = parpids_start,
                        parpids_increment = parpids_increment, 
                        local_ids = local_ids, 
                        batch_size = batch_size, 
                        partial_eval = partial_eval,
                        ptr_inc_step = ptr_inc_step,
                        allow_modify_flows = allow_modify_flows,
                        logspace_flows = logspace_flows,
                        BLOCK_B = BLOCK_B, 
                        TILE_SIZE_K = TILE_SIZE_K, 
                        K_NUM_TILES = K_NUM_TILES,
                        TILE_SIZE_M = TILE_SIZE_M, 
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        BLOCK_SIZE_K = BLOCK_SIZE_K,
                        TL_DOT = TL_DOT,
                        num_stages = 1,
                        propagation_alg_id = propagation_alg_id,
                        accumulate_ch_flows = accumulate_ch_flows,
                        allow_neg_flows = allow_neg_flows,
                        pid_m_offset = pid_m_start,
                        **propagation_alg_kwargs
                    )
                else:
                    bk_ele_bsparse._bk_triton_block_sparse_ele_csmm2_kernel[curr_grid](
                        node_flows = node_flows, 
                        element_flows = element_flows, 
                        node_mars = node_mars, 
                        element_mars = element_mars, 
                        mparams = params, 
                        chids = chids, 
                        parids_start = parids_start,
                        parids_increment = parids_increment,
                        parpids_start = parpids_start,
                        parpids_increment = parpids_increment, 
                        local_ids = local_ids, 
                        batch_size = batch_size, 
                        partial_eval = partial_eval,
                        ptr_inc_step = ptr_inc_step,
                        allow_modify_flows = allow_modify_flows,
                        logspace_flows = logspace_flows,
                        BLOCK_B = BLOCK_B, 
                        TILE_SIZE_K = TILE_SIZE_K, 
                        K_NUM_TILES = K_NUM_TILES,
                        TILE_SIZE_M = TILE_SIZE_M, 
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        BLOCK_SIZE_K = BLOCK_SIZE_K,
                        TL_DOT = TL_DOT,
                        num_stages = 1,
                        propagation_alg_id = propagation_alg_id,
                        accumulate_ch_flows = accumulate_ch_flows,
                        allow_neg_flows = allow_neg_flows,
                        pid_m_offset = pid_m_start,
                        **propagation_alg_kwargs
                    )

            else:

                if TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and BLOCK_B >= 8:
                    bk_ele_bsparse._bk_triton_block_sparse_tempered_ele_kernel[curr_grid](
                        node_flows = node_flows, 
                        element_flows = element_flows, 
                        node_mars_tempered = kwargs["node_mars_tempered"], 
                        element_mars = element_mars, 
                        mparams = params, 
                        chids = chids, 
                        parids_start = parids_start,
                        parids_increment = parids_increment,
                        parpids_start = parpids_start,
                        parpids_increment = parpids_increment, 
                        local_ids = local_ids, 
                        batch_size = batch_size, 
                        partial_eval = partial_eval,
                        ptr_inc_step = ptr_inc_step,
                        BLOCK_B = BLOCK_B, 
                        TILE_SIZE_K = TILE_SIZE_K, 
                        K_NUM_TILES = K_NUM_TILES,
                        TILE_SIZE_M = TILE_SIZE_M, 
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        BLOCK_SIZE_K = BLOCK_SIZE_K,
                        TL_DOT = TL_DOT,
                        accumulate_ch_flows = accumulate_ch_flows,
                        pid_m_offset = pid_m_start,
                        eflow_temperature = eflow_temperature,
                        num_stages = 1,
                    )
                else:
                    bk_ele_bsparse._bk_triton_block_sparse_tempered_ele_csmm2_kernel[curr_grid](
                        node_flows = node_flows, 
                        element_flows = element_flows, 
                        node_mars_tempered = kwargs["node_mars_tempered"],
                        element_mars = element_mars, 
                        mparams = params, 
                        chids = chids, 
                        parids_start = parids_start,
                        parids_increment = parids_increment,
                        parpids_start = parpids_start,
                        parpids_increment = parpids_increment, 
                        local_ids = local_ids, 
                        batch_size = batch_size, 
                        partial_eval = partial_eval,
                        ptr_inc_step = ptr_inc_step,
                        BLOCK_B = BLOCK_B, 
                        TILE_SIZE_K = TILE_SIZE_K, 
                        K_NUM_TILES = K_NUM_TILES,
                        TILE_SIZE_M = TILE_SIZE_M, 
                        BLOCK_SIZE_M = BLOCK_SIZE_M,
                        BLOCK_SIZE_K = BLOCK_SIZE_K,
                        TL_DOT = TL_DOT,
                        accumulate_ch_flows = accumulate_ch_flows,
                        pid_m_offset = pid_m_start,
                        eflow_temperature = eflow_temperature,
                        num_stages = 1,
                    )

        return None

    def _par_flow_collision_free(self, pfids: torch.Tensor) -> bool:
        """
        Whether the parameter-flow writes of this partition are collision-free, i.e. no two
        programs accumulate into the same `param_flows` slot. The kernel writes the block of
        `block_size` slots `[pfid, pfid + block_size)` for each `pfid` in `pfids`; these are
        disjoint iff all `pfids` are distinct and spaced at least `block_size` apart. When true,
        the non-atomic read-add-store kernel variant is bit-exact; otherwise the atomic kernel is
        required (e.g. tied parameter flows). Computed once per `pfids` tensor and cached.
        """
        if not hasattr(self, "_par_collision_free_cache"):
            self._par_collision_free_cache = dict()
        key = id(pfids)
        if key not in self._par_collision_free_cache:
            flat = pfids.reshape(-1)
            if flat.numel() <= 1:
                collision_free = True
            else:
                sorted_ids = torch.unique(flat, sorted = True)
                collision_free = bool(sorted_ids.numel() == flat.numel()) and \
                    bool((sorted_ids[1:] - sorted_ids[:-1] >= self.block_size).all().item())
            self._par_collision_free_cache[key] = collision_free
        return self._par_collision_free_cache[key]

    def _backward_block_sparse_par_flows(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor,
                                         element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                         cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor,
                                         allow_modify_flows: bool = False, propagation_alg: str = "LL", 
                                         logspace_flows: bool = False, negate_pflows: bool = False, 
                                         allow_neg_flows: bool = False, pflow_temperature: float = 1.0, **kwargs) -> None:
        """
        Backward pass of sum layers w.r.t. sum parameters with the block-sparse processing kernel.
        
        Parameters:
        `node_flows`:    [N, B]
        `element_flows`: [M, B]
        `params`:        [E]
        `node_mars`:     [N, B]
        `element_mars`:  [M, B]
        `param_flows`:   [E]
        `nids`:          [ng]
        `cids`:          [ng, c]
        `pids`:          [ng, c]
        """

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_nblocks = nids.size(0)
        layer_n_nodes = num_nblocks * self.block_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.block_size, num_edges, BATCH_SIZE_NP2)
        if base_size >= 64:
            TILE_SIZE_B = min(2048 // 32, BATCH_SIZE_NP2)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_B = min(2048 // remainder, base_size * remainder, BATCH_SIZE_NP2)
        TILE_SIZE_M = min(2048 // TILE_SIZE_B, self.block_size)
        TILE_SIZE_K = min(2048 // TILE_SIZE_B, num_edges)
        
        if propagation_alg_id == 1:
            # The kernel will stall if the tile sizes are too large
            TILE_SIZE_M = min(TILE_SIZE_M, 16)
            TILE_SIZE_K = min(TILE_SIZE_K, 16)
            TILE_SIZE_B = min(TILE_SIZE_B, 16)

        B_NUM_TILES = batch_size // TILE_SIZE_B

        allow_modify_flows = 1 if allow_modify_flows else 0

        assert TILE_SIZE_B >= 4, f"`TILE_SIZE_B` should be greater than 4 (but got {TILE_SIZE_B}) in order to use the block-sparse kernel. " \
                                  "This is an internal error of PyJuice. Please consider checking the kernel dispatching criterions and use the " \
                                  "corresponding sparse kernel instead."

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and TILE_SIZE_B >= 16:
            TL_DOT = 1
        else:
            TL_DOT = 0

        # Launch tuning for the LL block-sparse-dot regime (see `BACKWARD_PAR_FLOW_TUNED`).
        # Bit-exactness note: results depend only on `TILE_SIZE_M` (it sets the node group
        # over which `log_n_fdm_max` is taken), which is left unchanged; `TILE_SIZE_K`
        # (output-column tiling) and `TILE_SIZE_B` (batch-reduction tiling) do not change the
        # per-element result. Doubling K halves redundant `node_mars`/`node_flows` reads, a
        # smaller batch tile improves occupancy, and `num_warps=8` pairs with both. These
        # constants are tuned for an RTX PRO 6000 (Blackwell) and may differ on other GPUs.
        par_kernel_extra = {}
        if BACKWARD_PAR_FLOW_TUNED and not getattr(self, "_par_tuning_oom", False) \
                and TL_DOT == 1 and propagation_alg_id == 0 \
                and abs(pflow_temperature - 1.0) < 1e-6 and num_edges >= 2 * TILE_SIZE_K:
            TILE_SIZE_K = 2 * TILE_SIZE_K
            if TILE_SIZE_B > 32 and batch_size % 32 == 0:
                TILE_SIZE_B = 32
                B_NUM_TILES = batch_size // TILE_SIZE_B
            par_kernel_extra["num_warps"] = 8

        grid = (triton.cdiv(num_edges, TILE_SIZE_K), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

        # Optional CUDA fast path (CuTe/fp16/TMA), autotuned vs Triton INTO A SCRATCH buffer so the
        # autotune timing runs never corrupt the live param_flows (the kernel is read-accumulate-write).
        # Only intercepts when CUDA wins; Triton / unsupported shapes fall through to the dispatch below
        # (which keeps the OOM-retry path). See BACKWARD_PAR_FLOW_CUDA.
        if (BACKWARD_PAR_FLOW_CUDA and propagation_alg_id == 0 and abs(pflow_temperature - 1.0) < 1e-6
                and allow_modify_flows == 0 and logspace_flows and not allow_neg_flows and not negate_pflows
                and self.block_size % 64 == 0 and num_edges % 128 == 0 and batch_size % 32 == 0
                and node_flows.is_cuda and cuda_kernels.par_is_available()):
            par_sig = id(pfids)
            cache = self._cached_bk_par_cuda.get(par_sig)
            if cache is None:
                # collision-free (untied) + contiguous cids / block_size-strided pids&pfids -> the CUDA
                # kernel's index assumptions hold. Computed once per partition and cached.
                contig = bool((cids[:, 1:] - cids[:, :-1] == 1).all()
                              and (pids[:, 1:] - pids[:, :-1] == self.block_size).all()
                              and (pfids[:, 1:] - pfids[:, :-1] == self.block_size).all())
                if contig and self._par_flow_collision_free(pfids):
                    cache = (nids.contiguous(), cids[:, 0].contiguous(), pids[:, 0].contiguous(),
                             pfids[:, 0].contiguous(), True)
                else:
                    cache = (None, None, None, None, False)
                self._cached_bk_par_cuda[par_sig] = cache
            nbase, cbase, pbase, fbase, par_ok = cache
            if par_ok:
                def _cuda_par(tgt):
                    # mode 0 = read-accumulate-write (RMW): always correct (accumulates onto prior
                    # param_flows, e.g. multi-batch). The kernel also has a store-only mode that skips
                    # the RMW read, but that is only valid when param_flows is freshly zeroed; not used
                    # here to keep the path unconditionally correct.
                    cuda_kernels.par_backward_sum(tgt, node_flows, node_mars, element_mars, params,
                        nbase, cbase, pbase, fbase, batch_size, self.block_size, num_edges, 0)

                def _triton_par(tgt):
                    for s in range(0, grid[1], 32768):
                        cg = (grid[0], min(s + 32768, grid[1]) - s)
                        bk_par_bsparse._bk_triton_block_sparse_par_kernel_rmw[cg](
                            node_flows = node_flows, node_mars = node_mars, element_mars = element_mars,
                            mparams = params, param_flows = tgt, nids = nids, cids = cids, pids = pids,
                            pfids = pfids, batch_size = batch_size, num_edges = num_edges,
                            allow_modify_flows = allow_modify_flows, logspace_flows = logspace_flows,
                            TILE_SIZE_B = TILE_SIZE_B, B_NUM_TILES = B_NUM_TILES, TILE_SIZE_K = TILE_SIZE_K,
                            TILE_SIZE_M = TILE_SIZE_M, BLOCK_SIZE_M = self.block_size, TL_DOT = TL_DOT,
                            propagation_alg_id = propagation_alg_id, negate_pflows = negate_pflows,
                            allow_neg_flows = allow_neg_flows, pid_m_offset = s,
                            **propagation_alg_kwargs, **par_kernel_extra, num_stages = 1)

                choice_key = (par_sig, batch_size)
                choice = self._cached_bk_par_choice.get(choice_key)
                if choice is None:
                    # autotune into a scratch clone (corruption-safe; the kernel is read-accumulate-write).
                    # `param_flows` is the full param array (can be GBs), so the scratch is LOCAL and freed
                    # right after the autotune (run once per signature; the choice is then cached). If the
                    # scratch can't be allocated (memory-constrained GPU), fall back to Triton.
                    try:
                        scr = torch.empty_like(param_flows)
                        choice = cuda_kernels.autotune(
                            [("cuda", (lambda: _cuda_par(scr))), ("triton", (lambda: _triton_par(scr)))]) or "triton"
                        del scr
                    except torch.cuda.OutOfMemoryError:
                        choice = "triton"
                    self._cached_bk_par_choice[choice_key] = choice
                if choice == "cuda":
                    _cuda_par(param_flows)
                    return None
                # choice == "triton": fall through to the Triton dispatch below

        for pid_m_start in range(0, grid[1], 32768):
            pid_m_end = min(pid_m_start + 32768, grid[1])
            block_m_size = pid_m_end - pid_m_start

            curr_grid = (grid[0], block_m_size)

            if abs(pflow_temperature - 1.0) < 1e-6:

                if TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and TILE_SIZE_B >= 8:
                    # Use the non-atomic read-add-store variant when this partition's param-flow
                    # slots are provably collision-free (untied); otherwise the atomic kernel.
                    # The check is computed once per partition and cached (see the helper).
                    if self._par_flow_collision_free(pfids):
                        par_kernel = bk_par_bsparse._bk_triton_block_sparse_par_kernel_rmw
                    else:
                        par_kernel = bk_par_bsparse._bk_triton_block_sparse_par_kernel
                    try:
                        par_kernel[curr_grid](
                            node_flows = node_flows,
                            node_mars = node_mars,
                            element_mars = element_mars,
                            mparams = params,
                            param_flows = param_flows,
                            nids = nids,
                            cids = cids,
                            pids = pids,
                            pfids = pfids,
                            batch_size = batch_size,
                            num_edges = num_edges,
                            allow_modify_flows = allow_modify_flows,
                            logspace_flows = logspace_flows,
                            TILE_SIZE_B = TILE_SIZE_B,
                            B_NUM_TILES = B_NUM_TILES,
                            TILE_SIZE_K = TILE_SIZE_K,
                            TILE_SIZE_M = TILE_SIZE_M,
                            BLOCK_SIZE_M = self.block_size,
                            TL_DOT = TL_DOT,
                            propagation_alg_id = propagation_alg_id,
                            negate_pflows = negate_pflows,
                            allow_neg_flows = allow_neg_flows,
                            pid_m_offset = pid_m_start,
                            **propagation_alg_kwargs,
                            **par_kernel_extra,
                            num_stages = 1
                        )
                    except _TritonOutOfResources:
                        # The tuned launch config exceeds this GPU's shared memory. Disable the
                        # tuning for this layer (cached) and retry with the default heuristic.
                        # `OutOfResources` is raised at compile time, before any `param_flows`
                        # write, so re-running from scratch is safe (no partial accumulation).
                        if "num_warps" not in par_kernel_extra:
                            raise
                        self._par_tuning_oom = True
                        warnings.warn("pyjuice: tuned parameter-flow backward launch exceeds GPU "
                                      "shared memory; falling back to the default configuration.")
                        return self._backward_block_sparse_par_flows(
                            node_flows, params, node_mars, element_mars, param_flows,
                            nids, cids, pids, pfids, allow_modify_flows = allow_modify_flows,
                            propagation_alg = propagation_alg, logspace_flows = logspace_flows,
                            negate_pflows = negate_pflows, allow_neg_flows = allow_neg_flows,
                            pflow_temperature = pflow_temperature, **kwargs)

                else:
                    bk_par_bsparse._bk_triton_block_sparse_par_csmm2_kernel[curr_grid](
                        node_flows = node_flows, 
                        node_mars = node_mars, 
                        element_mars = element_mars, 
                        mparams = params, 
                        param_flows = param_flows, 
                        nids = nids, 
                        cids = cids, 
                        pids = pids,
                        pfids = pfids,
                        batch_size = batch_size, 
                        num_edges = num_edges, 
                        allow_modify_flows = allow_modify_flows,
                        logspace_flows = logspace_flows,
                        TILE_SIZE_B = TILE_SIZE_B, 
                        B_NUM_TILES = B_NUM_TILES, 
                        TILE_SIZE_K = TILE_SIZE_K, 
                        TILE_SIZE_M = TILE_SIZE_M, 
                        BLOCK_SIZE_M = self.block_size,
                        TL_DOT = TL_DOT,
                        propagation_alg_id = propagation_alg_id,
                        negate_pflows = negate_pflows,
                        allow_neg_flows = allow_neg_flows,
                        pid_m_offset = pid_m_start,
                        **propagation_alg_kwargs,
                        num_stages = 1
                    )

            else:

                if TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and TILE_SIZE_B >= 8:
                    bk_par_bsparse._bk_triton_block_sparse_tempered_par_kernel[curr_grid](
                        node_flows = node_flows, 
                        node_mars_tempered = kwargs["node_mars_tempered"], 
                        element_mars = element_mars, 
                        mparams = params, 
                        param_flows = param_flows, 
                        nids = nids, 
                        cids = cids, 
                        pids = pids,
                        pfids = pfids,
                        batch_size = batch_size, 
                        num_edges = num_edges, 
                        TILE_SIZE_B = TILE_SIZE_B, 
                        B_NUM_TILES = B_NUM_TILES, 
                        TILE_SIZE_K = TILE_SIZE_K, 
                        TILE_SIZE_M = TILE_SIZE_M, 
                        BLOCK_SIZE_M = self.block_size,
                        TL_DOT = TL_DOT,
                        negate_pflows = negate_pflows,
                        pid_m_offset = pid_m_start,
                        pflow_temperature = pflow_temperature,
                        num_stages = 1
                    )

                else:
                    bk_par_bsparse._bk_triton_block_sparse_tempered_par_csmm2_kernel[curr_grid](
                        node_flows = node_flows, 
                        node_mars_tempered = kwargs["node_mars_tempered"], 
                        element_mars = element_mars, 
                        mparams = params, 
                        param_flows = param_flows, 
                        nids = nids, 
                        cids = cids, 
                        pids = pids,
                        pfids = pfids,
                        batch_size = batch_size, 
                        num_edges = num_edges, 
                        TILE_SIZE_B = TILE_SIZE_B, 
                        B_NUM_TILES = B_NUM_TILES, 
                        TILE_SIZE_K = TILE_SIZE_K, 
                        TILE_SIZE_M = TILE_SIZE_M, 
                        BLOCK_SIZE_M = self.block_size,
                        TL_DOT = TL_DOT,
                        negate_pflows = negate_pflows,
                        allow_neg_flows = allow_neg_flows,
                        pid_m_offset = pid_m_start,
                        pflow_temperature = pflow_temperature,
                        num_stages = 1
                    )

        return None

    def _backward_sparse(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                         params: torch.Tensor, node_mars: torch.Tensor, 
                         element_mars: torch.Tensor, param_flows: torch.Tensor, 
                         nids: Optional[torch.Tensor], cids: Optional[torch.Tensor], pids: Optional[torch.Tensor], pfids: Optional[torch.Tensor],
                         chids: Optional[torch.Tensor], parids: Optional[torch.Tensor], parpids: Optional[torch.Tensor], 
                         cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                         partition_id: int = -1, allow_modify_flows: bool = False, 
                         propagation_alg: str = "LL", logspace_flows: bool = False, 
                         negate_pflows: bool = False, accumulate_ch_flows: bool = False, 
                         pflow_temperature: float = 1.0, temper_eflow: bool = False, **kwargs) -> None:
        """
        Back pass of sum layers with sparse processing kernel.
        
        Parameters:
        `node_flows`:   [N, B]
        `element_flows: [M, B]
        `params`:       [E]
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `param_flows`:  [E]
        `chids`:        [ng]
        `parids`:       [ng, c]
        `parpids`:      [ng, c]
        """

        # Flows w.r.t. input elements (product nodes)
        if chids is not None:
            self._backward_sparse_ele_flows(
                node_flows, element_flows, params, node_mars, element_mars,
                chids = chids, parids = parids, parpids = parpids, 
                cs_block_size = cs_block_size, local_ids = local_ids,
                allow_modify_flows = allow_modify_flows,
                propagation_alg = propagation_alg, 
                logspace_flows = logspace_flows, 
                accumulate_ch_flows = accumulate_ch_flows, 
                eflow_temperature = pflow_temperature if temper_eflow else 1.0, 
                **kwargs
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_sparse_par_flows(
                node_flows, params, node_mars, element_mars, param_flows,
                nids = nids, cids = cids, pids = pids, pfids = pfids,
                allow_modify_flows = allow_modify_flows,
                propagation_alg = propagation_alg, 
                logspace_flows = logspace_flows, 
                negate_pflows = negate_pflows, 
                pflow_temperature = pflow_temperature, **kwargs
            )

        return None

    def _backward_sparse_ele_flows(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                                   params: torch.Tensor, node_mars: torch.Tensor,
                                   element_mars: torch.Tensor, chids: torch.Tensor, parids: torch.Tensor,
                                   parpids: torch.Tensor, cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                                   allow_modify_flows: bool = False, propagation_alg: str = "LL", 
                                   logspace_flows: bool = False, accumulate_ch_flows: bool = False, 
                                   eflow_temperature: float = 1.0, **kwargs) -> None:

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_nblocks = chids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * cs_block_size
        n_edge_blocks = parids.size(1)
        num_edges = n_edge_blocks * self.block_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        assert num_edges <= 16384, "The sparse backward kernel only support nodes with # edges smaller than 16384."

        if triton.cdiv(layer_n_nodes, cs_block_size) <= 32768:

            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            BLOCK_M = cs_block_size

            allow_modify_flows = 1 if allow_modify_flows else 0

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, BLOCK_M))

            if abs(eflow_temperature - 1.0) < 1e-6:

                bk_ele_sparse._bk_triton_sparse_ele_kernel[grid](
                    node_flows = node_flows, 
                    element_flows = element_flows, 
                    node_mars = node_mars, 
                    element_mars = element_mars, 
                    mparams = params, 
                    chids = chids, 
                    parids = parids,
                    parpids = parpids,
                    local_ids = local_ids,
                    batch_size = batch_size,
                    partial_eval = 1 if local_ids is not None else 0,
                    n_edge_blocks = n_edge_blocks,
                    allow_modify_flows = allow_modify_flows,
                    logspace_flows = logspace_flows,
                    BLOCK_B = BLOCK_B,
                    BLOCK_M = BLOCK_M,
                    BLOCK_SIZE_K = self.block_size,
                    propagation_alg_id = propagation_alg_id,
                    accumulate_ch_flows = accumulate_ch_flows,
                    **propagation_alg_kwargs
                )

            else:

                bk_ele_sparse._bk_triton_sparse_tempered_ele_kernel[grid](
                    node_flows = node_flows, 
                    element_flows = element_flows, 
                    node_mars_tempered = kwargs["node_mars_tempered"],
                    element_mars = element_mars, 
                    mparams = params, 
                    chids = chids, 
                    parids = parids,
                    parpids = parpids,
                    local_ids = local_ids,
                    batch_size = batch_size,
                    partial_eval = 1 if local_ids is not None else 0,
                    n_edge_blocks = n_edge_blocks,
                    BLOCK_B = BLOCK_B,
                    BLOCK_M = BLOCK_M,
                    BLOCK_SIZE_K = self.block_size,
                    accumulate_ch_flows = accumulate_ch_flows,
                    eflow_temperature = eflow_temperature
                )

        else:

            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            TILE_SIZE_M = max(min(4096 // num_edges // BLOCK_B, triton.next_power_of_2(layer_n_nodes)), 1)

            allow_modify_flows = 1 if allow_modify_flows else 0

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            if abs(eflow_temperature - 1.0) < 1e-6:

                if grid[1] <= 32768:
                    bk_ele_sparse._bk_triton_large_sparse_ele_kernel[grid](
                        node_flows = node_flows,
                        element_flows = element_flows,
                        node_mars = node_mars,
                        element_mars = element_mars,
                        mparams = params,
                        chids = chids,
                        parids = parids,
                        parpids = parpids,
                        local_ids = local_ids,
                        num_eles = layer_n_nodes,
                        pid_m_offset = 0,
                        batch_size = batch_size,
                        partial_eval = 1 if local_ids is not None else 0,
                        n_edge_blocks = n_edge_blocks,
                        allow_modify_flows = allow_modify_flows,
                        logspace_flows = logspace_flows,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_M = TILE_SIZE_M,
                        BLOCK_SIZE_M = cs_block_size,
                        BLOCK_SIZE_K = self.block_size,
                        propagation_alg_id = propagation_alg_id,
                        accumulate_ch_flows = accumulate_ch_flows,
                        **propagation_alg_kwargs
                    )

                else:
                    for pid_m_start in range(0, grid[1], 32768):

                        pid_m_end = min(pid_m_start + 32768, grid[1])
                        small_grid = (grid[0], pid_m_end - pid_m_start)

                        bk_ele_sparse._bk_triton_large_sparse_ele_kernel[small_grid](
                            node_flows = node_flows,
                            element_flows = element_flows,
                            node_mars = node_mars,
                            element_mars = element_mars,
                            mparams = params,
                            chids = chids,
                            parids = parids,
                            parpids = parpids,
                            local_ids = local_ids,
                            num_eles = layer_n_nodes,
                            pid_m_offset = pid_m_start,
                            batch_size = batch_size,
                            partial_eval = 1 if local_ids is not None else 0,
                            n_edge_blocks = n_edge_blocks,
                            allow_modify_flows = allow_modify_flows,
                            logspace_flows = logspace_flows,
                            BLOCK_B = BLOCK_B,
                            TILE_SIZE_M = TILE_SIZE_M,
                            BLOCK_SIZE_M = cs_block_size,
                            BLOCK_SIZE_K = self.block_size,
                            propagation_alg_id = propagation_alg_id,
                            accumulate_ch_flows = accumulate_ch_flows,
                            **propagation_alg_kwargs
                        )

            else:

                if grid[1] <= 32768:
                    bk_ele_sparse._bk_triton_large_sparse_tempered_ele_kernel[grid](
                        node_flows = node_flows,
                        element_flows = element_flows,
                        node_mars_tempered = kwargs["node_mars_tempered"], 
                        element_mars = element_mars,
                        mparams = params,
                        chids = chids,
                        parids = parids,
                        parpids = parpids,
                        local_ids = local_ids,
                        num_eles = layer_n_nodes,
                        pid_m_offset = 0,
                        batch_size = batch_size,
                        partial_eval = 1 if local_ids is not None else 0,
                        n_edge_blocks = n_edge_blocks,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_M = TILE_SIZE_M,
                        BLOCK_SIZE_M = cs_block_size,
                        BLOCK_SIZE_K = self.block_size,
                        accumulate_ch_flows = accumulate_ch_flows,
                        eflow_temperature = eflow_temperature
                    )

                else:
                    for pid_m_start in range(0, grid[1], 32768):

                        pid_m_end = min(pid_m_start + 32768, grid[1])
                        small_grid = (grid[0], pid_m_end - pid_m_start)

                        bk_ele_sparse._bk_triton_large_sparse_tempered_ele_kernel[small_grid](
                            node_flows = node_flows,
                            element_flows = element_flows,
                            node_mars_tempered = kwargs["node_mars_tempered"], 
                            element_mars = element_mars,
                            mparams = params,
                            chids = chids,
                            parids = parids,
                            parpids = parpids,
                            local_ids = local_ids,
                            num_eles = layer_n_nodes,
                            pid_m_offset = pid_m_start,
                            batch_size = batch_size,
                            partial_eval = 1 if local_ids is not None else 0,
                            n_edge_blocks = n_edge_blocks,
                            BLOCK_B = BLOCK_B,
                            TILE_SIZE_M = TILE_SIZE_M,
                            BLOCK_SIZE_M = cs_block_size,
                            BLOCK_SIZE_K = self.block_size,
                            accumulate_ch_flows = accumulate_ch_flows,
                            eflow_temperature = eflow_temperature
                        )

        return None

    def _backward_sparse_par_flows(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                   element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                   cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor,
                                   allow_modify_flows: bool = False, propagation_alg: str = "LL", 
                                   logspace_flows: bool = False, negate_pflows: bool = False, 
                                   pflow_temperature: float = 1.0, **kwargs) -> None:
        """
        Backward pass of sum layers w.r.t. sum parameters with the block-sparse processing kernel.
        
        Parameters:
        `node_flows`:    [N, B]
        `element_flows`: [M, B]
        `params`:        [E]
        `node_mars`:     [N, B]
        `element_mars`:  [M, B]
        `param_flows`:   [E]
        `nids`:          [ng]
        `cids`:          [ng, c]
        `pids`:          [ng, c]
        """

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_nblocks = nids.size(0)
        layer_n_nodes = num_nblocks * self.block_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Propagation algorithm
        propagation_alg_id = self.propagation_alg_mapping[propagation_alg]
        propagation_alg_kwargs = self._get_propagation_alg_kwargs(propagation_alg, **kwargs)

        if num_edges <= 1024:
            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            BLOCK_K = num_edges
            BLOCK_M = self.block_size # The kernel recovers the node block via `pid_m // BLOCK_M`, so this must equal `block_size`
        else:
            BLOCK_B = min(512, BATCH_SIZE_NP2)
            BLOCK_K = min(2048 // BLOCK_B, num_edges)
            BLOCK_M = self.block_size # The kernel recovers the node block via `pid_m // BLOCK_M`, so this must equal `block_size`
        B_NUM_BLOCKS = triton.cdiv(batch_size, BLOCK_B)
        K_NUM_BLOCKS = triton.cdiv(num_edges, BLOCK_K)

        # When a thread-block is allocated for too much work, the overhead 
        # outweigh that incurred by `atomic_add`. Add more thread-blocks 
        # for parallel processing in this case.
        if B_NUM_BLOCKS >= 4:
            TILE_SIZE_B = 4 * BLOCK_B
            B_NUM_BLOCKS = 4
        else:
            TILE_SIZE_B = BATCH_SIZE_NP2
        B_NUM_TILES = triton.cdiv(batch_size, TILE_SIZE_B)

        allow_modify_flows = 1 if allow_modify_flows else 0

        grid = (B_NUM_TILES, K_NUM_BLOCKS, layer_n_nodes)

        # Triton seems to produce wrong results when using a (1, 1, 1) grid with BLOCK_B = 1 or 2...
        # if grid[0] == 1 and grid[1] == 1 and grid[2] == 1 and BLOCK_B < 4:
        #     BLOCK_B = 4

        if abs(pflow_temperature - 1.0) < 1e-6:
            if grid[2] <= 32768:
                bk_par_sparse._bk_triton_sparse_par_kernel[grid](
                    node_flows = node_flows, 
                    node_mars = node_mars, 
                    element_mars = element_mars, 
                    mparams = params, 
                    param_flows = param_flows, 
                    nids = nids, 
                    cids = cids, 
                    pids = pids,
                    pfids = pfids,
                    pid_m_offset = 0,
                    num_edges = num_edges,
                    batch_size = batch_size,
                    allow_modify_flows = allow_modify_flows,
                    logspace_flows = logspace_flows,
                    BLOCK_M = BLOCK_M,
                    BLOCK_K = BLOCK_K,
                    BLOCK_B = BLOCK_B,
                    TILE_SIZE_B = TILE_SIZE_B,
                    B_NUM_BLOCKS = B_NUM_BLOCKS,
                    propagation_alg_id = propagation_alg_id,
                    negate_pflows = negate_pflows,
                    **propagation_alg_kwargs
                )
            
            else:
                # TODO: This is a temporal fix...
                for pid_m_start in range(0, grid[2], 32768):

                    pid_m_end = min(pid_m_start + 32768, grid[2])
                    small_grid = (grid[0], grid[1], pid_m_end - pid_m_start)

                    bk_par_sparse._bk_triton_sparse_par_kernel[small_grid](
                        node_flows = node_flows, 
                        node_mars = node_mars, 
                        element_mars = element_mars, 
                        mparams = params, 
                        param_flows = param_flows, 
                        nids = nids, 
                        cids = cids, 
                        pids = pids,
                        pfids = pfids,
                        pid_m_offset = pid_m_start,
                        num_edges = num_edges,
                        batch_size = batch_size,
                        allow_modify_flows = allow_modify_flows,
                        logspace_flows = logspace_flows,
                        BLOCK_M = BLOCK_M,
                        BLOCK_K = BLOCK_K,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_B = TILE_SIZE_B,
                        B_NUM_BLOCKS = B_NUM_BLOCKS,
                        propagation_alg_id = propagation_alg_id,
                        negate_pflows = negate_pflows,
                        **propagation_alg_kwargs
                    )

        else:
            if grid[2] <= 32768:
                bk_par_sparse._bk_triton_sparse_tempered_par_kernel[grid](
                    node_flows = node_flows, 
                    node_mars_tempered = kwargs["node_mars_tempered"], 
                    element_mars = element_mars, 
                    mparams = params, 
                    param_flows = param_flows, 
                    nids = nids, 
                    cids = cids, 
                    pids = pids,
                    pfids = pfids,
                    pid_m_offset = 0,
                    num_edges = num_edges,
                    batch_size = batch_size,
                    BLOCK_M = BLOCK_M,
                    BLOCK_K = BLOCK_K,
                    BLOCK_B = BLOCK_B,
                    TILE_SIZE_B = TILE_SIZE_B,
                    B_NUM_BLOCKS = B_NUM_BLOCKS,
                    negate_pflows = negate_pflows,
                    pflow_temperature = pflow_temperature
                )
            
            else:
                # TODO: This is a temporal fix...
                for pid_m_start in range(0, grid[2], 32768):

                    pid_m_end = min(pid_m_start + 32768, grid[2])
                    small_grid = (grid[0], grid[1], pid_m_end - pid_m_start)

                    bk_par_sparse._bk_triton_sparse_tempered_par_kernel[small_grid](
                        node_flows = node_flows, 
                        node_mars_tempered = kwargs["node_mars_tempered"], 
                        element_mars = element_mars, 
                        mparams = params, 
                        param_flows = param_flows, 
                        nids = nids, 
                        cids = cids, 
                        pids = pids,
                        pfids = pfids,
                        pid_m_offset = pid_m_start,
                        num_edges = num_edges,
                        batch_size = batch_size,
                        BLOCK_M = BLOCK_M,
                        BLOCK_K = BLOCK_K,
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_B = TILE_SIZE_B,
                        B_NUM_BLOCKS = B_NUM_BLOCKS,
                        negate_pflows = negate_pflows,
                        pflow_temperature = pflow_temperature
                    )

        return None

    def _backward_pytorch(self, node_flows, element_flows, params, node_mars, 
                          element_mars, param_flows, nids, cids, pids, pfids, 
                          chids, parids, parpids, cs_block_size, propagation_alg: str = "LL", 
                          logspace_flows: bool = False, negate_pflows: bool = False, 
                          accumulate_ch_flows: bool = False):
        """
        Back pass of sum layers with native pytorch.
        
        Parameters:
        `node_flows`:   [N, B]
        `element_flows: [M, B]
        `params`:       [E]
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `param_flows`:  [E]
        `chids`:        [ng]
        `parids`:       [ng, c]
        `parpids`:      [ng, c]
        """

        assert propagation_alg == "LL"

        # Flows w.r.t. input elements (product nodes)
        if chids is not None:
            self._backward_pytorch_ele_kernel(
                node_flows, element_flows, params, node_mars, element_mars, 
                param_flows, chids, parids, parpids, cs_block_size, logspace_flows,
                accumulate_ch_flows = accumulate_ch_flows
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_pytorch_par_kernel(
                node_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, self.block_size, logspace_flows,
                negate_pflows
            )

    @torch.compile
    def _backward_pytorch_ele_kernel(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                                     params: torch.Tensor, node_mars: torch.Tensor, 
                                     element_mars: torch.Tensor, param_flows: Optional[torch.Tensor], 
                                     chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor,
                                     cs_block_size: int, logspace_flows: bool, accumulate_ch_flows: bool):

        num_nblocks = chids.size(0)
        num_eblocks = parids.size(1)
        parids = (parids[:,:,None].repeat(1, 1, self.block_size) + torch.arange(0, self.block_size, device = parids.device)).reshape(num_nblocks, num_eblocks * self.block_size)
        parpids = (parpids[:,:,None] + torch.arange(0, self.block_size, device = parids.device)).reshape(
            num_nblocks, num_eblocks * self.block_size)

        chids = (chids[:,None].repeat(1, cs_block_size) + torch.arange(0, cs_block_size, device = chids.device)).reshape(num_nblocks * cs_block_size)
        parids = parids[:,None,:].repeat(1, cs_block_size, 1).reshape(num_nblocks * cs_block_size, num_eblocks * self.block_size)
        parpids = (parpids[:,None,:].repeat(1, cs_block_size, 1) + torch.arange(0, cs_block_size * self.block_size, self.block_size, device = parpids.device)[None,:,None]).reshape(
            num_nblocks * cs_block_size, num_eblocks * self.block_size
        )
        
        if logspace_flows:
            if accumulate_ch_flows:
                element_flows[chids] += (node_flows[parids] + params[parpids].log().unsqueeze(-1) + \
                    element_mars[chids].unsqueeze(1) - node_mars[parids]).logsumexp(dim = 1)
            else:
                element_flows[chids] = (node_flows[parids] + params[parpids].log().unsqueeze(-1) + \
                    element_mars[chids].unsqueeze(1) - node_mars[parids]).logsumexp(dim = 1)
        else:
            if accumulate_ch_flows:
                element_flows[chids] += (node_flows[parids] * params[parpids].unsqueeze(-1) * \
                    (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)
            else:
                element_flows[chids] = (node_flows[parids] * params[parpids].unsqueeze(-1) * \
                    (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

        return None

    @torch.compile
    def _backward_pytorch_par_kernel(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                     element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                     cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor, ns_block_size: int,
                                     logspace_flows: bool, negate_pflows: bool):

        num_nblocks = nids.size(0)
        num_edges = cids.size(1)
        nids = (nids[:,None].repeat(1, self.block_size) + \
            torch.arange(0, self.block_size, device = nids.device)[None,:]).reshape(num_nblocks * self.block_size)
        cids = cids[:,None,:].repeat(1, self.block_size, 1).reshape(num_nblocks * self.block_size, num_edges)
        pids = (pids[:,None,:].repeat(1, self.block_size, 1) + \
            torch.arange(0, self.block_size, device = cids.device)[None,:,None]).reshape(num_nblocks * self.block_size, num_edges)
        pfids = (pfids[:,None,:].repeat(1, self.block_size, 1) + \
            torch.arange(0, self.block_size, device = cids.device)[None,:,None]).reshape(num_nblocks * self.block_size, num_edges)

        if logspace_flows:
            parflows = (node_flows[nids].exp().unsqueeze(1) * params[pids].unsqueeze(-1) * (element_mars[cids] - node_mars[nids].unsqueeze(1)).exp()).sum(dim = 2)
        else:
            parflows = (node_flows[nids].unsqueeze(1) * params[pids].unsqueeze(-1) * (element_mars[cids] - node_mars[nids].unsqueeze(1)).exp()).sum(dim = 2)

        for i in range(num_nblocks):
            sid, eid = ns_block_size * i, ns_block_size * (i + 1)
            if negate_pflows:
                param_flows[pfids[sid:eid,:]] -= parflows[sid:eid,:]
            else:
                param_flows[pfids[sid:eid,:]] += parflows[sid:eid,:]

        return None

    def _prepare_scope2nids(self, prod_scope_eleids: Sequence[Tuple[BitSet, torch.Tensor]]):
        if not (hasattr(self, "fw_scope2localids") and hasattr(self, "bk_scope2localids")):
            fw_scope2localids = dict()
            bk_scope2localids = dict()

            # Forward local indices
            global_nid = self._layer_nid_range[0]
            for ns in self.nodes:
                scope = ns.scope

                s_nid = global_nid
                e_nid = global_nid + ns.num_nodes

                with torch.no_grad():
                    if scope not in fw_scope2localids:
                        fw_scope2localids[scope] = [
                            torch.zeros([0], dtype = torch.long).to(self.partitioned_nids[0].device) for _ in range(self.num_fw_partitions)
                        ]

                    for partition_id in range(self.num_fw_partitions):
                        nids = self.partitioned_nids[partition_id]
                        partition_local_ids = torch.where((nids >= s_nid) & (nids < e_nid))[0]

                        fw_scope2localids[scope][partition_id] = torch.cat(
                            (fw_scope2localids[scope][partition_id], partition_local_ids), dim = 0
                        )

                global_nid += ns.num_nodes

            # Backward local indices
            for scope, ele_id_range in prod_scope_eleids:
                s_eid, e_eid = ele_id_range

                with torch.no_grad():
                    if scope not in bk_scope2localids:
                        bk_scope2localids[scope] = [
                            torch.zeros([0], dtype = torch.long).to(self.partitioned_chids[0].device) for _ in range(self.num_bk_partitions)
                        ]

                    for partition_id in range(self.num_bk_partitions):
                        chids = self.partitioned_chids[partition_id]
                        partition_local_ids = torch.where((chids >= s_eid) & (chids < e_eid))[0]

                        bk_scope2localids[scope][partition_id] = torch.cat(
                            (bk_scope2localids[scope][partition_id], partition_local_ids), dim = 0
                        )

            self.fw_scope2localids = fw_scope2localids
            self.bk_scope2localids = bk_scope2localids