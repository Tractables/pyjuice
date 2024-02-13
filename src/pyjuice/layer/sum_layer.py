
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
from copy import deepcopy
from typing import Sequence, List, Tuple, Optional

from pyjuice.nodes import SumNodes
from pyjuice.utils import BitSet
from pyjuice.utils.parameter_list import FastParamList
from pyjuice.utils.kernel_launcher import FastJITFunction
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges
from .backend.index_set import batched_index_set, index_cum
from .compilation import get_sum_layer_forward_stats, sum_layer_forward_compilation, \
                         get_sum_layer_backward_stats, \
                         sum_layer_backward_compilation, next_power_of_2


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

        self.nodes = nodes

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

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                force_use_fp16: bool = False, force_use_fp32: bool = False) -> None:
        """
        Computes the forward pass of a sum layer.

        Parameters:
        `node_mars`:    [num_nodes, B]
        `element_mars`: [max_num_els, B]
        `params`:       [num_params, B] or [num_params]
        """

        if not self.provided("fw_partition_local_ids"):
            # Evaluate the whole layer
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]
                cids = self.partitioned_cids[partition_id]
                pids = self.partitioned_pids[partition_id]

                self._forward(
                    node_mars, element_mars, params, nids, cids, pids, 
                    partition_id = partition_id, force_use_fp16 = force_use_fp16,
                    force_use_fp32 = force_use_fp32
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
                    partition_id = partition_id, force_use_fp16 = force_use_fp16,
                    force_use_fp32 = force_use_fp32
                )

        return None

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                 node_mars: torch.Tensor, element_mars: torch.Tensor, 
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 allow_modify_flows: bool = False) -> None:
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

        # Disallow modifications of `node_flows` in case of partial evaluation
        if self.provided("bk_partition_local_ids") and allow_modify_flows:
            allow_modify_flows = False

        ## Pre-compute `nflows.log() - nmars` if needed ##
        if allow_modify_flows:
            assert not self.provided("bk_partition_local_ids"), "Must set `allow_modify_flows = False` for partial evaluation."
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]

                self._bk_triton_modify_flow(
                    node_flows, node_mars, nids, local_ids = None
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
                    allow_modify_flows = allow_modify_flows
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
                    allow_modify_flows = allow_modify_flows
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
                    allow_modify_flows = allow_modify_flows
                )

        return None

    def _forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                 pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                 partition_id: int = -1, mode: Optional[str] = None,
                 force_use_fp16: bool = False, force_use_fp32: bool = False) -> None:
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
        elif self.block_size * batch_size < 32:
            # Advantage of block-sparse processing is diminishing
            mode = self.SPARSE
        else:
            mode = self.BLOCK_SPARSE

        if mode == self.BLOCK_SPARSE:
            self._forward_block_sparse(
                node_mars, element_mars, params, nids, cids, pids, local_ids,
                partition_id = partition_id, force_use_fp16 = force_use_fp16,
                force_use_fp32 = force_use_fp32
            )

        elif mode == self.SPARSE:
            self._forward_sparse(
                node_mars, element_mars, params, nids, cids, pids, local_ids,
                partition_id = partition_id
            )

        elif mode == self.PYTORCH:
            self._forward_pytorch(
                node_mars, element_mars, params, nids, cids, pids, local_ids
            )
        
        else:
            raise ValueError(f"Unexpected mode `{mode}`.")

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _fw_triton_block_sparse_tlmm_kernel(node_mars, element_mars, params, nids, cids_start, cids_increment,
                                            pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                            BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                            TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, use_fp16: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            nblock_id = tl.load(local_ids + nblock_id)

        # Node offsets
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

        # Edge offsets
        offs_edge = tl.arange(0, TILE_SIZE_K)

        # Initialize pointers to `params`
        offs_estart = nblock_id * TILE_SIZE_K + offs_edge
        offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
        par_start = tl.load(pids_start + offs_estart)
        epars_ptr = params + \
            offs_node[:,None] + \
            par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        edge_start = tl.load(cids_start + offs_estart)
        emars_ptr = element_mars + \
            edge_start[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

        # Batch increment pointers
        pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
        cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr)
            emars = tl.load(emars_ptr, mask = mask_batch[None,:])

            emars_max = tl.max(emars, axis = 0)[None,:]
            emars_sub = tl.where(emars_max != -float("inf"), tl.exp(emars - emars_max), 0.0)

            if use_fp16 == 1:
                # Built-in matmul kernel of triton + float16
                epars_fp16 = (epars * (2**12)).to(tl.float16)
                emars_fp16 = emars_sub.to(tl.float16)
                nmars = tl.dot(epars_fp16, emars_fp16).to(tl.float32) / (2**12)
            else:
                # Built-in matmul kernel of triton + float32
                nmars = tl.dot(epars, emars_sub)

            acc = tl.where(emars_max > acc,
                tl.log(nmars + tl.exp(acc - emars_max)) + emars_max,
                tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc
            )

            # Increment `epars_ptr`
            pids_inc = tl.load(pids_inc_ptr)
            epars_ptr += pids_inc[None,:]
            pids_inc_ptr += TILE_SIZE_K

            # Increment `emars_ptr`
            cids_inc = tl.load(cids_inc_ptr)
            emars_ptr += cids_inc[:,None] * batch_size
            cids_inc_ptr += TILE_SIZE_K

        # Write back
        off_nids = tl.load(nids + nblock_id)
        offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
        tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _fw_triton_block_sparse_csmm1_kernel(node_mars, element_mars, params, nids, cids_start, cids_increment,
                                            pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                            BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                            TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, use_fp16: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            nblock_id = tl.load(local_ids + nblock_id)

        # Node offsets
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

        # Edge offsets
        offs_edge = tl.arange(0, TILE_SIZE_K)

        # Initialize pointers to `params`
        offs_estart = nblock_id * TILE_SIZE_K + offs_edge
        offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
        par_start = tl.load(pids_start + offs_estart)
        epars_ptr = params + \
            offs_node[:,None] + \
            par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        edge_start = tl.load(cids_start + offs_estart)
        emars_ptr = element_mars + \
            edge_start[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

        # Batch increment pointers
        pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
        cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr)
            emars = tl.load(emars_ptr, mask = mask_batch[None,:])

            emars_max = tl.max(emars, axis = 0)[None,:]
            emars_sub = tl.where(emars_max != -float("inf"), tl.exp(emars - emars_max), 0.0)

            if use_fp16 == 1:
                # Simulated matmul kernel + float16
                epars = (epars * (2**4)).to(tl.float16)
                emars_sub = emars_sub.to(tl.float16)
                nmars = tl.sum(epars[:,:,None] * emars_sub[None,:,:], axis = 1).to(tl.float32) / (2**4)
            else:
                # Simulated matmul kernel + float32
                nmars = tl.sum(epars[:,:,None] * emars_sub[None,:,:], axis = 1)

            acc = tl.where(emars_max > acc,
                tl.log(nmars + tl.exp(acc - emars_max)) + emars_max,
                tl.log(tl.exp(emars_max - acc) * nmars + 1.0) + acc
            )

            # Increment `epars_ptr`
            pids_inc = tl.load(pids_inc_ptr)
            epars_ptr += pids_inc[None,:]
            pids_inc_ptr += TILE_SIZE_K

            # Increment `emars_ptr`
            cids_inc = tl.load(cids_inc_ptr)
            emars_ptr += cids_inc[:,None] * batch_size
            cids_inc_ptr += TILE_SIZE_K

        # Write back
        off_nids = tl.load(nids + nblock_id)
        offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
        tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _fw_triton_block_sparse_csmm2_kernel(node_mars, element_mars, params, nids, cids_start, cids_increment,
                                             pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                             BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                             TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, use_fp16: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            nblock_id = tl.load(local_ids + nblock_id)

        # Node offsets
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

        # Edge offsets
        offs_edge = tl.arange(0, TILE_SIZE_K)

        # Initialize pointers to `params`
        offs_estart = nblock_id * TILE_SIZE_K + offs_edge
        offs_estart = tl.max_contiguous(offs_estart, TILE_SIZE_K)
        par_start = tl.load(pids_start + offs_estart)
        epars_ptr = params + \
            offs_node[:,None] + \
            par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        offs_batch = tl.max_contiguous(offs_batch, BLOCK_B)
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        edge_start = tl.load(cids_start + offs_estart)
        emars_ptr = element_mars + \
            edge_start[None,:] * batch_size + \
            offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]

        # Batch increment pointers
        pids_inc_ptr = pids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
        cids_inc_ptr = cids_increment + nblock_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr)
            emars = tl.load(emars_ptr, mask = mask_batch[:,None])

            emars_max = tl.max(emars, axis = 1)
            emars_sub = tl.where(emars_max[:,None] != -float("inf"), tl.exp(emars - emars_max[:,None]), 0.0)

            # Simulated matmul kernel + float32
            nmars = tl.sum(epars[:,:,None] * tl.trans(emars_sub)[None,:,:], axis = 1)

            acc = tl.where(emars_max[None,:] > acc,
                tl.log(nmars + tl.exp(acc - emars_max[None,:])) + emars_max[None,:],
                tl.log(tl.exp(emars_max[None,:] - acc) * nmars + 1.0) + acc
            )

            # Increment `epars_ptr`
            pids_inc = tl.load(pids_inc_ptr)
            epars_ptr += pids_inc[None,:]
            pids_inc_ptr += TILE_SIZE_K

            # Increment `emars_ptr`
            cids_inc = tl.load(cids_inc_ptr)
            emars_ptr += cids_inc[None,:] * batch_size
            cids_inc_ptr += TILE_SIZE_K

        # Write back
        off_nids = tl.load(nids + nblock_id)
        offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
        tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])

    def _forward_block_sparse(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                              params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                              pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                              partition_id: int = -1, force_use_fp16: bool = False,
                              force_use_fp32: bool = False) -> None:
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

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.block_size, num_edges, BATCH_SIZE_NP2, 128)
        if base_size >= 64:
            TILE_SIZE_K = min(2048 // 32, num_edges)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_K = min(2048 // remainder, base_size * remainder, num_edges)
        TILE_SIZE_M = min(2048 // TILE_SIZE_K, self.block_size)
        BLOCK_B = min(2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
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
        else:
            cids_start, cids_increment, pids_start, pids_increment = self._cached_fw_pcids[signature]

        partial_eval = 1 if local_ids is not None else 0
        BLOCK_SIZE_M = self.block_size

        if force_use_fp16:
            assert not force_use_fp32
            use_fp16 = True
        elif force_use_fp32:
            use_fp16 = False
        else:
            if TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and BLOCK_B >= 8:
                use_fp16 = True
            else:
                use_fp16 = False

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))
        
        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
            self._fw_triton_block_sparse_tlmm_kernel[grid](
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
                use_fp16 = use_fp16
            )
            
        elif TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and BLOCK_B >= 8:
            self._fw_triton_block_sparse_csmm1_kernel[grid](
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
                use_fp16 = use_fp16
            )
        else:
            self._fw_triton_block_sparse_csmm2_kernel[grid](
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
                use_fp16 = use_fp16
            )
        
        return None

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _fw_triton_sparse_kernel(node_mars, element_mars, params, nids, cids, pids,
                                 local_ids, batch_size, partial_eval: tl.constexpr, num_edges: tl.constexpr, 
                                 BLOCK_B: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        
        pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(axis = 1) # ID of size-`BLOCK_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            nblock_id = tl.load(local_ids + nblock_id)

        # Initialize pointers to `params`
        offs_edge = tl.arange(0, num_edges)
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_ptr = params + par_start # [num_edges]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize and load edge mars
        edge_ids = tl.load(cids + nblock_id * num_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_ids[:,None] * batch_size + \
            offs_batch[None,:]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

        # Compute max and subtract
        emars_max = tl.max(emars, axis = 0)
        emars = tl.exp(emars - emars_max[None,:])

        # Initialize pointers to `node_mars`
        off_nids = tl.load(nids + nblock_id)
        nmars_ptr = node_mars + \
            off_nids * batch_size + \
            offs_batch

        # Inner loop
        for i in range(0, BLOCK_SIZE_M):
            epars = tl.load(epars_ptr)

            nmars = tl.log(tl.sum(emars * epars[:,None], axis = 0)) + emars_max

            tl.store(nmars_ptr, nmars, mask = mask_batch)

            # Increment `epars_ptr`
            epars_ptr += 1

            # Increment `nmars_ptr`
            nmars_ptr += batch_size

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _fw_triton_large_sparse_kernel(node_mars, element_mars, params, nids, cids, pids,
                                       local_ids, batch_size, num_nodes, pid_m_offset, partial_eval: tl.constexpr, num_edges: tl.constexpr, 
                                       BLOCK_B: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                       BLOCK_SIZE_M: tl.constexpr):

        pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(axis = 1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

        offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
        mask_m = offs_m < num_nodes

        # Get inferred node block id from `pid_m`
        nblock_ids = offs_m // BLOCK_SIZE_M
        tile_ids = offs_m % BLOCK_SIZE_M

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            nblock_ids = tl.load(local_ids + nblock_ids, mask = mask_m)

        # Initialize pointers to `params`
        offs_edge = tl.arange(0, num_edges)
        par_start = tl.load(pids + nblock_ids[:,None] * num_edges + offs_edge[None,:], mask = mask_m[:,None]) # [TILE_SIZE_M, num_edges]
        epars = tl.load(params + tile_ids[:,None] * BLOCK_SIZE_M + par_start, mask = mask_m[:,None], other = 0.0) # [TILE_SIZE_M, num_edges]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize and load edge mars
        edge_ids = tl.load(cids + nblock_ids[:,None] * num_edges + offs_edge[None,:]) # [TILE_SIZE_M, num_edges]
        emars_ptr = element_mars + \
            edge_ids[:,:,None] * batch_size + \
            offs_batch[None,None,:] # [TILE_SIZE_M, num_edges, BLOCK_B]
        emars = tl.load(emars_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:]), other = 0.0) # [TILE_SIZE_M, num_edges, BLOCK_B]

        # Compute max and subtract
        emars_max = tl.max(emars, axis = 1)
        emars = tl.exp(emars - emars_max[:,None,:])
        nmars = tl.log(tl.sum(emars * epars[:,:,None], axis = 1)) + emars_max

        # Initialize pointers to `node_mars`
        off_nids = tl.load(nids + nblock_ids) # [TILE_SIZE_M]
        nmars_ptr = node_mars + \
            (off_nids + tile_ids)[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]

        tl.store(nmars_ptr, nmars, mask = (mask_m[:,None] & mask_batch[None,:]))

    def _forward_sparse(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                        params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                        pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                        partition_id: int = -1) -> None:
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

        assert num_edges <= 16384, "The sparse forward kernel only support nodes with # edges smaller than 16384."

        if triton.cdiv(layer_n_nodes, self.block_size) <= 2048:
            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)

            partial_eval = 1 if local_ids is not None else 0
            BLOCK_SIZE_M = self.block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, BLOCK_SIZE_M))

            self._fw_triton_sparse_kernel[grid](
                node_mars = node_mars, 
                element_mars = element_mars, 
                params = params, 
                nids = nids, 
                cids = cids,
                pids = pids,
                local_ids = local_ids, 
                batch_size = batch_size, 
                partial_eval = partial_eval, 
                num_edges = num_edges, 
                BLOCK_B = BLOCK_B, 
                BLOCK_SIZE_M = BLOCK_SIZE_M
            )

        else:
            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            TILE_SIZE_M = max(min(4096 // num_edges // BLOCK_B, triton.next_power_of_2(layer_n_nodes)), 1)

            partial_eval = 1 if local_ids is not None else 0
            BLOCK_SIZE_M = self.block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            if grid[1] <= 32768:
                self._fw_triton_large_sparse_kernel[grid](
                    node_mars = node_mars,
                    element_mars = element_mars,
                    params = params,
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
                    BLOCK_SIZE_M = BLOCK_SIZE_M
                )
            else:
                for pid_m_start in range(0, grid[1], 32768):

                    pid_m_end = min(pid_m_start + 32768, grid[1])
                    small_grid = (grid[0], pid_m_end - pid_m_start)

                    self._fw_triton_large_sparse_kernel[small_grid](
                        node_mars = node_mars,
                        element_mars = element_mars,
                        params = params,
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
                        BLOCK_SIZE_M = BLOCK_SIZE_M
                    )

        return None

    @staticmethod
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _forward_pytorch_kernel(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, 
                                nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor,
                                local_ids: torch.Tensor):

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
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        node_mars[nids] = (((ch_mars - maxval).exp() * params[pids].unsqueeze(-1)).sum(
            dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

        return None

    def _forward_pytorch(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, 
                         nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor,
                         local_ids: torch.Tensor):

        self._forward_pytorch_kernel(
            node_mars, element_mars, params, nids, cids, pids, local_ids
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
                  allow_modify_flows: bool = False) -> None:
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
        elif (cs_block_size == 1 or self.block_size == 1) and num_edges <= 32768:
            # In this case, we should definitely use the sparse implementation
            mode = self.SPARSE
        elif self.block_size * batch_size < 32:
            # Advantage of block-sparse processing is diminishing
            mode = self.SPARSE
        else:
            mode = self.BLOCK_SPARSE

        if mode == self.BLOCK_SPARSE:
            self._backward_block_sparse(
                node_flows, element_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, chids, parids, parpids, cs_block_size, local_ids, 
                partition_id = partition_id, allow_modify_flows = allow_modify_flows
            )

        elif mode == self.SPARSE:
            self._backward_sparse(
                node_flows, element_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, chids, parids, parpids, cs_block_size, local_ids, 
                partition_id = partition_id, allow_modify_flows = allow_modify_flows
            )

        elif mode == self.PYTORCH:
            assert not allow_modify_flows, "Please set `allow_modify_flows` to False when " \
                                           "using the native PyTorch backward."
            self._backward_pytorch(
                node_flows, element_flows, params, node_mars, 
                element_mars, param_flows, nids, cids, pids, pfids, 
                chids, parids, parpids, cs_block_size
            )
        else:
            raise ValueError(f"Not supported mode `{mode}`.")

        return None

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_modify_flow_kernel(node_flows, node_mars, local_ids, nids, batch_size: tl.constexpr, partial_eval: tl.constexpr, 
                                      BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` examples
        pid_m = tl.program_id(1) # ID of size-`BLOCK_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (BLOCK_SIZE_M // BLOCK_M)
        tile_id = pid_m % (BLOCK_SIZE_M // BLOCK_M)

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            nblock_id = tl.load(local_ids + nblock_id)

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `node_flows` and `node_mars`
        offs_node = tl.arange(0, BLOCK_M) + tile_id * BLOCK_M
        off_nids = tl.load(nids + nblock_id)
        offs_nmfs = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

        nmars = tl.load(node_mars + offs_nmfs, mask = mask_batch[None,:])
        nflows = tl.load(node_flows + offs_nmfs, mask = mask_batch[None,:])

        uflows = tl.where(nmars != -float("inf"), tl.log(nflows) - nmars, -float("inf"))

        tl.store(node_flows + offs_nmfs, uflows, mask = mask_batch[None,:])

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_large_modify_flow_kernel(node_flows, node_mars, local_ids, nids, num_nodes, batch_size: tl.constexpr, partial_eval: tl.constexpr, 
                                            BLOCK_B: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` examples
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
        mask_m = offs_m < num_nodes

        # Get inferred node block id from `pid_m`
        nblock_ids = offs_m // BLOCK_SIZE_M
        tile_ids = offs_m % BLOCK_SIZE_M

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            nblock_ids = tl.load(local_ids + nblock_ids)

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `node_flows` and `node_mars`
        off_nids = tl.load(nids + nblock_ids, mask = mask_m) # [TILE_SIZE_M]
        offs_nmfs = (off_nids + tile_ids)[:,None] * batch_size + offs_batch[None,:]

        nmars = tl.load(node_mars + offs_nmfs, mask = (mask_m[:,None] & mask_batch[None,:]))
        nflows = tl.load(node_flows + offs_nmfs, mask = (mask_m[:,None] & mask_batch[None,:]))

        uflows = tl.where(nmars != -float("inf"), tl.log(nflows) - nmars, -float("inf"))

        tl.store(node_flows + offs_nmfs, uflows, mask = (mask_m[:,None] & mask_batch[None,:]))

    def _bk_triton_modify_flow(self, node_flows: torch.Tensor, node_mars: torch.Tensor,
                               nids: torch.Tensor, local_ids: Optional[torch.Tensor] = None):
        """
        Replace `node_flows[nids]` with `node_flows[nids].log() - node_mars[nids]`
        """

        num_nblocks = nids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * self.block_size
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

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

            self._bk_triton_modify_flow_kernel[grid](
                node_flows = node_flows, 
                node_mars = node_mars, 
                local_ids = local_ids, 
                nids = nids, 
                batch_size = batch_size, 
                partial_eval = partial_eval, 
                BLOCK_B = BLOCK_B, 
                BLOCK_M = BLOCK_M, 
                BLOCK_SIZE_M = BLOCK_SIZE_M
            )

        else:

            BLOCK_B = min(2048, BATCH_SIZE_NP2)
            TILE_SIZE_M = min(4096 // BLOCK_B, triton.next_power_of_2(layer_n_nodes))

            partial_eval = 1 if local_ids is not None else 0
            BLOCK_SIZE_M = self.block_size

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            self._bk_triton_large_modify_flow_kernel[grid](
                node_flows = node_flows,
                node_mars = node_mars,
                local_ids = local_ids,
                nids = nids,
                num_nodes = layer_n_nodes,
                batch_size = batch_size,
                partial_eval = partial_eval,
                BLOCK_B = BLOCK_B,
                TILE_SIZE_M = TILE_SIZE_M,
                BLOCK_SIZE_M = BLOCK_SIZE_M
            )

        return None

    def _backward_block_sparse(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                               params: torch.Tensor, node_mars: torch.Tensor, 
                               element_mars: torch.Tensor, param_flows: torch.Tensor, 
                               nids: Optional[torch.Tensor], cids: Optional[torch.Tensor], pids: Optional[torch.Tensor], pfids: Optional[torch.Tensor],
                               chids: Optional[torch.Tensor], parids: Optional[torch.Tensor], parpids: Optional[torch.Tensor], 
                               cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                               partition_id: int = -1, allow_modify_flows: bool = False) -> None:
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
                partition_id = partition_id, allow_modify_flows = allow_modify_flows
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_block_sparse_par_flows(
                node_flows, params, node_mars, element_mars, param_flows,
                nids = nids, cids = cids, pids = pids, pfids = pfids,
                allow_modify_flows = allow_modify_flows
            )

        return None

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_block_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                           chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                           local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                           allow_modify_flows: tl.constexpr, BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                           K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, 
                                           BLOCK_SIZE_K: tl.constexpr, TL_DOT: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            eleblock_id = tl.load(local_ids + eleblock_id)

        # Initialize pointers to `params`
        offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_edge = tl.arange(0, TILE_SIZE_K)
        offs_edge_gid = offs_edge // BLOCK_SIZE_K
        offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
        par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
        epars_ptr = params + \
            offs_ele[:,None] * BLOCK_SIZE_K + \
            (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `node_mars`
        edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
        nmars_ptr = node_mars + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
        nflows_ptr = node_flows + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

        # Batch increment pointers
        parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
        parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
                nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
                log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None,:]
            n_fdm_sub = tl.where(log_n_fdm_max != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max), 0.0)

            if TL_DOT == 1:
                partial_flows = tl.dot(epars, n_fdm_sub)
            else:
                partial_flows = tl.sum(epars[:,:,None] * n_fdm_sub[None,:,:], axis = 1)

            acc = tl.where(log_n_fdm_max == acc,
                acc + 0.69314718056, # log(2)
                tl.where(log_n_fdm_max > acc,
                    tl.log(partial_flows + tl.exp(acc - log_n_fdm_max)) + log_n_fdm_max,
                    tl.log(tl.exp(log_n_fdm_max - acc) * partial_flows + 1.0) + acc
                )
            )
            # neginf_flag = (log_n_fdm_max == -float("inf")) & (acc == -float("inf"))
            # acc = tl.where(log_n_fdm_max > acc,
            #     tl.log(partial_flows + tl.exp(acc - log_n_fdm_max)) + log_n_fdm_max,
            #     tl.log(tl.exp(log_n_fdm_max - acc) * partial_flows + 1.0) + acc
            # )
            # acc = tl.where(neginf_flag, -float("inf"), acc)

            # Increment `epars_ptr`
            parpids_inc = tl.load(parpids_inc_ptr)
            epars_ptr += parpids_inc[None,:]
            parpids_inc_ptr += ptr_inc_step

            # Increment `nmars_ptr`
            parids_inc = tl.load(parids_inc_ptr)
            if allow_modify_flows == 0:
                nmars_ptr += parids_inc[:,None] * batch_size
            nflows_ptr += parids_inc[:,None] * batch_size
            parids_inc_ptr += ptr_inc_step

        # Initialize pointers to `element_mars`
        off_eleids = tl.load(chids + eleblock_id)
        emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        eflows = tl.exp(acc + emars)

        # Write back
        offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
        tl.store(element_flows + offs_elemfs, eflows, mask = mask_batch[None,:])

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_block_sparse_ele_csmm2_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                                 chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                                 local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                                 allow_modify_flows: tl.constexpr, BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                                 K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, 
                                                 BLOCK_SIZE_K: tl.constexpr, TL_DOT: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            eleblock_id = tl.load(local_ids + eleblock_id)

        # Initialize pointers to `params`
        offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_edge = tl.arange(0, TILE_SIZE_K)
        offs_edge_gid = offs_edge // BLOCK_SIZE_K
        offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
        par_start = tl.load(parpids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
        epars_ptr = params + \
            offs_ele[:,None] * BLOCK_SIZE_K + \
            (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `node_mars`
        edge_start = tl.load(parids_start + eleblock_id * ptr_inc_step + offs_edge_gid)
        nmars_ptr = node_mars + \
            (edge_start + offs_edge_nid)[None,:] * batch_size + \
            offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]
        nflows_ptr = node_flows + \
            (edge_start + offs_edge_nid)[None,:] * batch_size + \
            offs_batch[:,None] # [BLOCK_B, TILE_SIZE_K]

        # Batch increment pointers
        parids_inc_ptr = parids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
        parpids_inc_ptr = parpids_increment + eleblock_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
                nmars = tl.load(nmars_ptr, mask = mask_batch[:,None]) # [BLOCK_B, TILE_SIZE_K]
                log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 1)
            n_fdm_sub = tl.where(log_n_fdm_max[:,None] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[:,None]), 0.0)

            partial_flows = tl.sum(epars[:,:,None] * tl.trans(n_fdm_sub)[None,:,:], axis = 1)

            acc = tl.where(log_n_fdm_max[None,:] == acc,
                acc + 0.69314718056, # log(2)
                tl.where(log_n_fdm_max[None,:] > acc,
                    tl.log(partial_flows + tl.exp(acc - log_n_fdm_max[None,:])) + log_n_fdm_max[None,:],
                    tl.log(tl.exp(log_n_fdm_max[None,:] - acc) * partial_flows + 1.0) + acc
                )
            )
            # neginf_flag = (log_n_fdm_max[None,:] == -float("inf")) & (acc == -float("inf"))
            # acc = tl.where(log_n_fdm_max[None,:] > acc,
            #     tl.log(partial_flows + tl.exp(acc - log_n_fdm_max[None,:])) + log_n_fdm_max[None,:],
            #     tl.log(tl.exp(log_n_fdm_max[None,:] - acc) * partial_flows + 1.0) + acc
            # )
            # acc = tl.where(neginf_flag, -float("inf"), acc)

            # Increment `epars_ptr`
            parpids_inc = tl.load(parpids_inc_ptr)
            epars_ptr += parpids_inc[None,:]
            parpids_inc_ptr += ptr_inc_step

            # Increment `nmars_ptr`
            parids_inc = tl.load(parids_inc_ptr)
            if allow_modify_flows == 0:
                nmars_ptr += parids_inc[None,:] * batch_size
            nflows_ptr += parids_inc[None,:] * batch_size
            parids_inc_ptr += ptr_inc_step

        # Initialize pointers to `element_mars`
        off_eleids = tl.load(chids + eleblock_id)
        emars_ptr = element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        eflows = tl.exp(acc + emars)

        # Write back
        offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
        tl.store(element_flows + offs_elemfs, eflows, mask = mask_batch[None,:])

    def _backward_block_sparse_ele_flows(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                                         params: torch.Tensor, node_mars: torch.Tensor,
                                         element_mars: torch.Tensor, chids: torch.Tensor, parids: torch.Tensor,
                                         parpids: torch.Tensor, cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                                         partition_id: int = -1, allow_modify_flows: bool = False) -> None:

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_nblocks = chids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * cs_block_size
        num_edges = parids.size(1) * self.block_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

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
        else:
            parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step = self._cached_bk_parids[signature]

        partial_eval = 1 if local_ids is not None else 0
        BLOCK_SIZE_M = cs_block_size
        BLOCK_SIZE_K = self.block_size
        allow_modify_flows = 1 if allow_modify_flows else 0

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
            TL_DOT = 1
        else:
            TL_DOT = 0

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

        if TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and BLOCK_B >= 8:
            self._bk_triton_block_sparse_ele_kernel[grid](
                node_flows = node_flows, 
                element_flows = element_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                params = params, 
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
                BLOCK_B = BLOCK_B, 
                TILE_SIZE_K = TILE_SIZE_K, 
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M, 
                BLOCK_SIZE_M = BLOCK_SIZE_M,
                BLOCK_SIZE_K = BLOCK_SIZE_K,
                TL_DOT = TL_DOT,
                num_warps = 2, # TODO: test for different devices
                num_stages = 1
            )
        else:
            self._bk_triton_block_sparse_ele_csmm2_kernel[grid](
                node_flows = node_flows, 
                element_flows = element_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                params = params, 
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
                BLOCK_B = BLOCK_B, 
                TILE_SIZE_K = TILE_SIZE_K, 
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M, 
                BLOCK_SIZE_M = BLOCK_SIZE_M,
                BLOCK_SIZE_K = BLOCK_SIZE_K,
                TL_DOT = TL_DOT,
                num_warps = 2, # TODO: test for different devices
                num_stages = 1
            )

        return None

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_block_sparse_par_kernel(node_flows, node_mars, element_mars, params, param_flows, nids, cids, pids, pfids,
                                           batch_size: tl.constexpr, num_edges: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                           TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                           TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, TL_DOT: tl.constexpr):

        pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

        # Batch offsets and mask
        offs_batch = tl.arange(0, TILE_SIZE_B)
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
        edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_start[None,:] * batch_size + \
            offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

        # Initialize pointers to `node_flows` and `node_mars`
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        off_nids = tl.load(nids + nblock_id)
        nmars_ptr = node_mars + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
        nflows_ptr = node_flows + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)
        
        for b in range(0, B_NUM_TILES):
            emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[None,:], other = -float("inf")) # [TILE_SIZE_M, TILE_SIZE_B]
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]
                nmars = tl.load(nmars_ptr, mask = mask_batch[None,:], other = 0.0) # [TILE_SIZE_M, TILE_SIZE_B]
                log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 0)
            n_fdm_sub = tl.where(log_n_fdm_max[None,:] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[None,:]), 0.0)

            scaled_emars = tl.exp(emars + log_n_fdm_max[:,None])

            if TL_DOT == 1:
                partial_flows = tl.dot(n_fdm_sub, scaled_emars)
            else:
                partial_flows = tl.sum(n_fdm_sub[:,:,None] * scaled_emars[None,:,:], axis = 1)

            acc += partial_flows

            # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
            emars_ptr += TILE_SIZE_B
            if allow_modify_flows == 0:
                nmars_ptr += TILE_SIZE_B
            nflows_ptr += TILE_SIZE_B

            # Update batch mask
            offs_batch += TILE_SIZE_B
            mask_batch = offs_batch < batch_size

        # Initialize `params`
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tl.load(params + epars_offsets)

        pflows = acc * epars

        parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
        eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        tl.atomic_add(param_flows + eparflows_offsets, pflows)

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_block_sparse_par_csmm2_kernel(node_flows, node_mars, element_mars, params, param_flows, nids, cids, pids, pfids,
                                                 batch_size: tl.constexpr, num_edges: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                                 TILE_SIZE_B: tl.constexpr, B_NUM_TILES: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                                 TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, TL_DOT: tl.constexpr):

        pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

        # Batch offsets and mask
        offs_batch = tl.arange(0, TILE_SIZE_B)
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
        edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_start[None,:] * batch_size + \
            offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

        # Initialize pointers to `node_flows` and `node_mars`
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        off_nids = tl.load(nids + nblock_id)
        nmars_ptr = node_mars + (off_nids + offs_node[None,:]) * batch_size + offs_batch[:,None]
        nflows_ptr = node_flows + (off_nids + offs_node[None,:]) * batch_size + offs_batch[:,None]

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)
        
        for b in range(0, B_NUM_TILES):
            emars = tl.load(emars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_K]

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[:,None], other = -float("inf")) # [TILE_SIZE_B, TILE_SIZE_M]
            else:
                nflows = tl.load(nflows_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_M]
                nmars = tl.load(nmars_ptr, mask = mask_batch[:,None], other = 0.0) # [TILE_SIZE_B, TILE_SIZE_M]
                log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), tl.log(nflows) - nmars)

            log_n_fdm_max = tl.max(log_n_fdm, axis = 1)
            n_fdm_sub = tl.where(log_n_fdm_max[:,None] != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max[:,None]), 0.0)

            scaled_emars = tl.exp(emars + log_n_fdm_max[:,None])

            partial_flows = tl.sum(tl.trans(n_fdm_sub)[:,:,None] * scaled_emars[None,:,:], axis = 1)

            acc += partial_flows

            # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
            emars_ptr += TILE_SIZE_B
            if allow_modify_flows == 0:
                nmars_ptr += TILE_SIZE_B
            nflows_ptr += TILE_SIZE_B

            # Update batch mask
            offs_batch += TILE_SIZE_B
            mask_batch = offs_batch < batch_size

        # Initialize `params`
        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tl.load(params + epars_offsets)

        pflows = acc * epars

        parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
        eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        tl.atomic_add(param_flows + eparflows_offsets, pflows)

    def _backward_block_sparse_par_flows(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                         element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                         cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor,
                                         allow_modify_flows: bool = False) -> None:
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

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.block_size, num_edges, BATCH_SIZE_NP2)
        if base_size >= 64:
            TILE_SIZE_B = min(2048 // 32, BATCH_SIZE_NP2)
        else:
            remainder = 2048 // (base_size ** 2)
            TILE_SIZE_B = min(2048 // remainder, base_size * remainder, BATCH_SIZE_NP2)
        TILE_SIZE_M = min(2048 // TILE_SIZE_B, self.block_size)
        TILE_SIZE_K = min(2048 // TILE_SIZE_B, num_edges)
        B_NUM_TILES = batch_size // TILE_SIZE_B

        allow_modify_flows = 1 if allow_modify_flows else 0

        assert TILE_SIZE_B >= 4, f"`TILE_SIZE_B` should be greater than 4 (but got {TILE_SIZE_B}) in order to use the block-sparse kernel. " \
                                  "This is an internal error of PyJuice. Please consider checking the kernel dispatching criterions and use the " \
                                  "corresponding sparse kernel instead."

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and TILE_SIZE_B >= 16:
            TL_DOT = 1
        else:
            TL_DOT = 0

        grid = (triton.cdiv(num_edges, TILE_SIZE_K), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

        if TILE_SIZE_M >= 8 and TILE_SIZE_K >= 8 and TILE_SIZE_B >= 8:
            self._bk_triton_block_sparse_par_kernel[grid](
                node_flows = node_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                params = params, 
                param_flows = param_flows, 
                nids = nids, 
                cids = cids, 
                pids = pids,
                pfids = pfids,
                batch_size = batch_size, 
                num_edges = num_edges, 
                allow_modify_flows = allow_modify_flows, 
                TILE_SIZE_B = TILE_SIZE_B, 
                B_NUM_TILES = B_NUM_TILES, 
                TILE_SIZE_K = TILE_SIZE_K, 
                TILE_SIZE_M = TILE_SIZE_M, 
                BLOCK_SIZE_M = self.block_size,
                TL_DOT = TL_DOT
            )
        else:
            self._bk_triton_block_sparse_par_csmm2_kernel[grid](
                node_flows = node_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                params = params, 
                param_flows = param_flows, 
                nids = nids, 
                cids = cids, 
                pids = pids,
                pfids = pfids,
                batch_size = batch_size, 
                num_edges = num_edges, 
                allow_modify_flows = allow_modify_flows, 
                TILE_SIZE_B = TILE_SIZE_B, 
                B_NUM_TILES = B_NUM_TILES, 
                TILE_SIZE_K = TILE_SIZE_K, 
                TILE_SIZE_M = TILE_SIZE_M, 
                BLOCK_SIZE_M = self.block_size,
                TL_DOT = TL_DOT
            )

    def _backward_sparse(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                         params: torch.Tensor, node_mars: torch.Tensor, 
                         element_mars: torch.Tensor, param_flows: torch.Tensor, 
                         nids: Optional[torch.Tensor], cids: Optional[torch.Tensor], pids: Optional[torch.Tensor], pfids: Optional[torch.Tensor],
                         chids: Optional[torch.Tensor], parids: Optional[torch.Tensor], parpids: Optional[torch.Tensor], 
                         cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                         partition_id: int = -1, allow_modify_flows: bool = False) -> None:
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
                allow_modify_flows = allow_modify_flows
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_sparse_par_flows(
                node_flows, params, node_mars, element_mars, param_flows,
                nids = nids, cids = cids, pids = pids, pfids = pfids,
                allow_modify_flows = allow_modify_flows
            )

        return None

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                     chids, parids, parpids, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                     n_edge_blocks: tl.constexpr, allow_modify_flows: tl.constexpr, BLOCK_B: tl.constexpr, 
                                     BLOCK_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`BLOCK_M` nodes

        # Get inferred node block id from `pid_m`
        eleblock_id = pid_m

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            eleblock_id = tl.load(local_ids + eleblock_id)

        # Initialize pointers to `params`
        offs_edge = tl.arange(0, n_edge_blocks * BLOCK_SIZE_K) # I.e., [0, num_edges)
        offs_edge_gid = offs_edge // BLOCK_SIZE_K
        offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
        par_start = tl.load(parpids + eleblock_id * n_edge_blocks + offs_edge_gid)
        epars_ptr = params + par_start + offs_edge_nid # [num_edges]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize `node_flows` and `node_mars`
        edge_start = tl.load(parids + eleblock_id * n_edge_blocks + offs_edge_gid)
        nmars_ptr = node_mars + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]
        nflows_ptr = node_flows + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:]
        if allow_modify_flows == 1:
            log_n_fdm = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]
        else:
            nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

        # Initialize pointers to `element_flows` and `element_mars`
        off_eleids = tl.load(chids + eleblock_id)
        eflows_ptr = element_flows + off_eleids * batch_size + offs_batch # [BLOCK_B]
        emars_ptr = element_mars + off_eleids * batch_size + offs_batch # [BLOCK_B]

        # Inner loop
        for i in range(0, BLOCK_M):
            epars = tl.load(epars_ptr) # [num_edges]
            emars = tl.load(emars_ptr, mask = mask_batch) # [BLOCK_B]

            if allow_modify_flows == 1:
                eflows = tl.sum(epars[:,None] * tl.exp(emars[None,:] + log_n_fdm), axis = 0)
            else:
                eflows = tl.sum(nflows * epars[:,None] * tl.exp(emars[None,:] - nmars), axis = 0)

            tl.store(eflows_ptr, eflows, mask = mask_batch)

            # Increment `epars_ptr`
            epars_ptr += BLOCK_SIZE_K

            # Increment `emars_ptr` and `eflows_ptr`
            emars_ptr += batch_size
            eflows_ptr += batch_size

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_large_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                           chids, parids, parpids, local_ids, num_eles, pid_m_offset, 
                                           batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                           n_edge_blocks: tl.constexpr, allow_modify_flows: tl.constexpr, BLOCK_B: tl.constexpr, 
                                           TILE_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) + pid_m_offset # ID of size-`TILE_SIZE_M` nodes

        offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
        mask_m = offs_m < num_eles

        # Get inferred node block id from `pid_m`
        eleblock_ids = offs_m // BLOCK_SIZE_M
        tile_ids = offs_m % BLOCK_SIZE_M

        # Get the real node block id in the case of partial evaluation
        if partial_eval == 1:
            eleblock_ids = tl.load(local_ids + eleblock_ids)

        # Initialize pointers to `params`
        offs_edge = tl.arange(0, n_edge_blocks * BLOCK_SIZE_K) # I.e., [0, num_edges)
        offs_edge_gid = offs_edge // BLOCK_SIZE_K
        offs_edge_nid = (offs_edge % BLOCK_SIZE_K)
        par_start = tl.load(parpids + eleblock_ids[:,None] * n_edge_blocks + offs_edge_gid[None,:])
        epars = tl.load(params + par_start + offs_edge_nid[None,:], mask = mask_m[:,None]) # [TILE_SIZE_M, num_edges]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize `node_flows` and `node_mars`
        edge_start = tl.load(parids + eleblock_ids[:,None] * n_edge_blocks + offs_edge_gid[None,:]) # [TILE_SIZE_M, num_edges]
        nmars_ptr = node_mars + \
            (edge_start + offs_edge_nid[None,:])[:,:,None] * batch_size + \
            offs_batch[None,None,:]
        nmars = tl.load(nmars_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]
        nflows_ptr = node_flows + \
            (edge_start + offs_edge_nid[None,:])[:,:,None] * batch_size + \
            offs_batch[None,None,:]
        if allow_modify_flows == 1:
            log_n_fdm = tl.load(nflows_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]
        else:
            nflows = tl.load(nflows_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:])) # [TILE_SIZE_M, num_edges, BLOCK_B]

        # Initialize pointers to `element_flows` and values `emars`
        off_eleids = tl.load(chids + eleblock_ids) # TILE_SIZE_M
        eflows_ptr = element_flows + off_eleids[:,None] * batch_size + offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]
        emars = tl.load(element_mars + off_eleids[:,None] * batch_size + offs_batch[None,:], 
                        mask = (mask_m[:,None] & mask_batch[None,:])) # [TILE_SIZE_M, BLOCK_B]

        # Compute eflows
        if allow_modify_flows == 1:
            eflows = tl.sum(epars[:,:,None] * tl.exp(emars[:,None,:] + log_n_fdm), axis = 1)
        else:
            eflows = tl.sum(nflows * epars[:,:,None] * tl.exp(emars[:,None,:] - nmars), axis = 1)

        tl.store(eflows_ptr, eflows, mask = (mask_m[:,None] & mask_batch[None,:]))

    def _backward_sparse_ele_flows(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                                   params: torch.Tensor, node_mars: torch.Tensor,
                                   element_mars: torch.Tensor, chids: torch.Tensor, parids: torch.Tensor,
                                   parpids: torch.Tensor, cs_block_size: int, local_ids: Optional[torch.Tensor] = None,
                                   allow_modify_flows: bool = False) -> None:

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_nblocks = chids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_nblocks * cs_block_size
        n_edge_blocks = parids.size(1)
        num_edges = n_edge_blocks * self.block_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        assert num_edges <= 16384, "The sparse backward kernel only support nodes with # edges smaller than 16384."

        if triton.cdiv(layer_n_nodes, cs_block_size) <= 32768:

            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            BLOCK_M = cs_block_size

            allow_modify_flows = 1 if allow_modify_flows else 0

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, BLOCK_M))

            self._bk_triton_sparse_ele_kernel[grid](
                node_flows = node_flows, 
                element_flows = element_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                params = params, 
                chids = chids, 
                parids = parids,
                parpids = parpids,
                local_ids = local_ids,
                batch_size = batch_size,
                partial_eval = 1 if local_ids is not None else 0,
                n_edge_blocks = n_edge_blocks,
                allow_modify_flows = allow_modify_flows,
                BLOCK_B = BLOCK_B,
                BLOCK_M = BLOCK_M,
                BLOCK_SIZE_K = self.block_size
            )

        else:

            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            TILE_SIZE_M = max(min(4096 // num_edges // BLOCK_B, triton.next_power_of_2(layer_n_nodes)), 1)

            allow_modify_flows = 1 if allow_modify_flows else 0

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

            if grid[1] <= 32768:
                self._bk_triton_large_sparse_ele_kernel[grid](
                    node_flows = node_flows,
                    element_flows = element_flows,
                    node_mars = node_mars,
                    element_mars = element_mars,
                    params = params,
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
                    BLOCK_B = BLOCK_B,
                    TILE_SIZE_M = TILE_SIZE_M,
                    BLOCK_SIZE_M = cs_block_size,
                    BLOCK_SIZE_K = self.block_size
                )

            else:
                for pid_m_start in range(0, grid[1], 32768):

                    pid_m_end = min(pid_m_start + 32768, grid[1])
                    small_grid = (grid[0], pid_m_end - pid_m_start)

                    self._bk_triton_large_sparse_ele_kernel[small_grid](
                        node_flows = node_flows,
                        element_flows = element_flows,
                        node_mars = node_mars,
                        element_mars = element_mars,
                        params = params,
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
                        BLOCK_B = BLOCK_B,
                        TILE_SIZE_M = TILE_SIZE_M,
                        BLOCK_SIZE_M = cs_block_size,
                        BLOCK_SIZE_K = self.block_size
                    )

        return None

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_sparse_par_kernel(node_flows, node_mars, element_mars, params, param_flows, nids, cids, pids, pfids,
                                     pid_m_offset, num_edges: tl.constexpr, batch_size: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                     BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr, 
                                     TILE_SIZE_B: tl.constexpr, B_NUM_BLOCKS: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` samples
        pid_e = tl.program_id(1) # ID of size-`BLOCK_K` edges
        pid_m = tl.program_id(2) + pid_m_offset # ID of size-`BLOCK_M` nodes

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // BLOCK_M
        tile_id = pid_m % BLOCK_M

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        offs_edge = tl.arange(0, BLOCK_K) + pid_e * BLOCK_K
        edge_start = tl.load(cids + nblock_id * num_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_start[:,None] * batch_size + \
            offs_batch[None,:] # [BLOCK_K, BLOCK_B]

        # Initialize pointers to `node_flows` and `node_mars`
        off_nids = tl.load(nids + nblock_id)
        nmars_ptr = node_mars + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]
        nflows_ptr = node_flows + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]

        # Inner loop
        acc = tl.zeros([BLOCK_K], dtype = tl.float32)

        for b in range(0, B_NUM_BLOCKS):
            # Batch offsets and mask
            offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B + b * BLOCK_B
            mask_batch = offs_batch < batch_size

            emars = tl.load(emars_ptr, mask = mask_batch[None,:], other = -float("inf")) # [BLOCK_K, BLOCK_B]

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = mask_batch, other = -float("inf")) # [BLOCK_B]
                pflows = tl.sum(tl.exp(emars + log_n_fdm[None,:]), axis = 1)
            else:
                nmars = tl.load(nmars_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]
                nflows = tl.load(nflows_ptr, mask = mask_batch, other = 0.0) # [BLOCK_B]
                pflows = tl.sum(nflows[None,:] * tl.exp(emars - nmars[None,:]), axis = 1)

            acc += pflows

            # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
            emars_ptr += BLOCK_B
            nmars_ptr += BLOCK_B
            nflows_ptr += BLOCK_B

        par_start = tl.load(pids + nblock_id * num_edges + offs_edge)
        epars_ptr = params + par_start + tile_id
        epars = tl.load(epars_ptr) # [BLOCK_K]

        parflow_start = tl.load(pfids + nblock_id * num_edges + offs_edge)
        eparflows_ptr = param_flows + parflow_start + tile_id
        
        curr_pflows = acc * epars

        tl.atomic_add(eparflows_ptr, curr_pflows)

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _bk_triton_large_sparse_par_kernel(node_flows, node_mars, element_mars, params, param_flows, nids, cids, pids, pfids,
                                           num_nodes, num_edges: tl.constexpr, batch_size: tl.constexpr, allow_modify_flows: tl.constexpr, 
                                           TILE_SIZE_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr, 
                                           TILE_SIZE_B: tl.constexpr, B_NUM_BLOCKS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` samples
        pid_e = tl.program_id(1) # ID of size-`BLOCK_K` edges
        pid_m = tl.program_id(2) # ID of size-`TILE_SIZE_M` nodes

        offs_m = tl.arange(0, TILE_SIZE_M) + pid_m * TILE_SIZE_M
        mask_m = offs_m < num_nodes

        # Get inferred node block id from `pid_m`
        nblock_ids = offs_m // BLOCK_SIZE_M
        tile_ids = offs_m % BLOCK_SIZE_M

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        offs_edge = tl.arange(0, BLOCK_K) + pid_e * BLOCK_K
        edge_start = tl.load(cids + nblock_ids[:,None] * num_edges + offs_edge[None,:], mask = mask_m[:,None])
        emars_ptr = element_mars + \
            edge_start[:,:,None] * batch_size + \
            offs_batch[None,None,:] # [TILE_SIZE_M, BLOCK_K, BLOCK_B]

        # Initialize pointers to `node_flows` and `node_mars`
        off_nids = tl.load(nids + nblock_ids, mask = mask_m)
        nmars_ptr = node_mars + (off_nids + tile_ids)[:,None] * batch_size + offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]
        nflows_ptr = node_flows + (off_nids + tile_ids)[:,None] * batch_size + offs_batch[None,:] # [TILE_SIZE_M, BLOCK_B]

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_K], dtype = tl.float32) + 0.1

        for b in range(0, B_NUM_BLOCKS):
            # Batch offsets and mask
            offs_batch = tl.arange(0, BLOCK_B) + pid_b * TILE_SIZE_B + b * BLOCK_B
            mask_batch = offs_batch < batch_size

            emars = tl.load(emars_ptr, mask = (mask_m[:,None,None] & mask_batch[None,None,:]), other = -float("inf")) # [TILE_SIZE_M, BLOCK_K, BLOCK_B]

            if allow_modify_flows == 1:
                log_n_fdm = tl.load(nflows_ptr, mask = (mask_m[:,None] & mask_batch[None,:]), other = -float("inf")) # [TILE_SIZE_M, BLOCK_B]
                pflows = tl.sum(tl.exp(emars + log_n_fdm[:,None,:]), axis = 2)
            else:
                nmars = tl.load(nmars_ptr, mask = (mask_m[:,None] & mask_batch[None,:]), other = 0.0) # [TILE_SIZE_M, BLOCK_B]
                nflows = tl.load(nflows_ptr, mask = (mask_m[:,None] & mask_batch[None,:]), other = 0.0) # [TILE_SIZE_M, BLOCK_B]
                pflows = tl.sum(nflows[:,None,:] * tl.exp(emars - nmars[:,None,:]), axis = 2)

            acc += pflows

            # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
            emars_ptr += BLOCK_B
            nmars_ptr += BLOCK_B
            nflows_ptr += BLOCK_B

        par_start = tl.load(pids + nblock_ids[:,None] * num_edges + offs_edge[None,:])
        epars_ptr = params + par_start + tile_ids[:,None]
        epars = tl.load(epars_ptr, mask = mask_m[:,None]) # [TILE_SIZE_M, BLOCK_K]

        parflow_start = tl.load(pfids + nblock_ids[:,None] * num_edges + offs_edge[None,:])
        eparflows_ptr = param_flows + parflow_start + tile_ids[:,None]
        
        curr_pflows = acc * epars

        tl.atomic_add(eparflows_ptr, curr_pflows, mask = mask_m[:,None])

    def _backward_sparse_par_flows(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                   element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                   cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor,
                                   allow_modify_flows: bool = False) -> None:
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

        assert num_edges <= 16384, "The sparse backward kernel only support nodes with # edges smaller than 16384."


        if num_edges <= 1024:
            BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
            BLOCK_K = num_edges
            BLOCK_M = max(min(2048 // num_edges, self.block_size), 1)
        else:
            BLOCK_B = min(512, BATCH_SIZE_NP2)
            BLOCK_K = min(2048 // BLOCK_B, num_edges)
            BLOCK_M = max(min(2048 // num_edges, self.block_size), 1)
        B_NUM_BLOCKS = triton.cdiv(batch_size, BLOCK_B)
        K_NUM_BLOCKS = triton.cdiv(num_edges, BLOCK_K)

        # When a thread-block is allocated for too much work, the overhead 
        # outweigh that incurred by `atomic_add`. Add more thread-blocks 
        # for parallel processing in this case.
        if B_NUM_BLOCKS >= 4:
            TILE_SIZE_B = 4 * BLOCK_B
            B_NUM_BLOCKS = 4
        else:
            TILE_SIZE_B = batch_size
        B_NUM_TILES = triton.cdiv(batch_size, TILE_SIZE_B)

        allow_modify_flows = 1 if allow_modify_flows else 0

        grid = (B_NUM_TILES, K_NUM_BLOCKS, layer_n_nodes)

        if grid[2] <= 32768:
            self._bk_triton_sparse_par_kernel[grid](
                node_flows = node_flows, 
                node_mars = node_mars, 
                element_mars = element_mars, 
                params = params, 
                param_flows = param_flows, 
                nids = nids, 
                cids = cids, 
                pids = pids,
                pfids = pfids,
                pid_m_offset = 0,
                num_edges = num_edges,
                batch_size = batch_size,
                allow_modify_flows = allow_modify_flows,
                BLOCK_M = BLOCK_M,
                BLOCK_K = BLOCK_K,
                BLOCK_B = BLOCK_B,
                TILE_SIZE_B = TILE_SIZE_B,
                B_NUM_BLOCKS = B_NUM_BLOCKS
            )
        
        else:
            # TODO: This is a temporal fix...
            for pid_m_start in range(0, grid[2], 32768):

                pid_m_end = min(pid_m_start + 32768, grid[2])
                small_grid = (grid[0], grid[1], pid_m_end - pid_m_start)

                self._bk_triton_sparse_par_kernel[small_grid](
                    node_flows = node_flows, 
                    node_mars = node_mars, 
                    element_mars = element_mars, 
                    params = params, 
                    param_flows = param_flows, 
                    nids = nids, 
                    cids = cids, 
                    pids = pids,
                    pfids = pfids,
                    pid_m_offset = pid_m_start,
                    num_edges = num_edges,
                    batch_size = batch_size,
                    allow_modify_flows = allow_modify_flows,
                    BLOCK_M = BLOCK_M,
                    BLOCK_K = BLOCK_K,
                    BLOCK_B = BLOCK_B,
                    TILE_SIZE_B = TILE_SIZE_B,
                    B_NUM_BLOCKS = B_NUM_BLOCKS
                )

        # else:

        #     if num_edges <= 1024:
        #         BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
        #         BLOCK_K = num_edges
        #         TILE_SIZE_M = max(min(4096 // num_edges, triton.next_power_of_2(layer_n_nodes)), 1)
        #     else:
        #         BLOCK_B = min(512, BATCH_SIZE_NP2)
        #         BLOCK_K = min(2048 // BLOCK_B, num_edges)
        #         TILE_SIZE_M = max(min(2048 // num_edges, triton.next_power_of_2(layer_n_nodes)), 1)
        #     B_NUM_BLOCKS = triton.cdiv(batch_size, BLOCK_B)
        #     K_NUM_BLOCKS = triton.cdiv(num_edges, BLOCK_K)

        #     # When a thread-block is allocated for too much work, the overhead 
        #     # outweigh that incurred by `atomic_add`. Add more thread-blocks 
        #     # for parallel processing in this case.
        #     if B_NUM_BLOCKS >= 4:
        #         TILE_SIZE_B = 4 * BLOCK_B
        #         B_NUM_BLOCKS = 4
        #     else:
        #         TILE_SIZE_B = batch_size
        #     B_NUM_TILES = triton.cdiv(batch_size, TILE_SIZE_B)

        #     allow_modify_flows = 1 if allow_modify_flows else 0

        #     grid = (B_NUM_TILES, K_NUM_BLOCKS, triton.cdiv(layer_n_nodes, TILE_SIZE_M))

        #     print(">>>G", grid, "in")
        #     # TODO: This kernel gets stuck for some input configurations. Fix it.
        #     if grid[0] == 2 and grid[1] == 1 and grid[2] == 308:
        #         import pdb; pdb.set_trace()
        #     self._bk_triton_large_sparse_par_kernel[grid](
        #         node_flows = node_flows, 
        #         node_mars = node_mars, 
        #         element_mars = element_mars, 
        #         params = params, 
        #         param_flows = param_flows, 
        #         nids = nids, 
        #         cids = cids, 
        #         pids = pids, 
        #         pfids = pfids,
        #         num_nodes = layer_n_nodes, 
        #         num_edges = num_edges, 
        #         batch_size = batch_size, 
        #         allow_modify_flows = allow_modify_flows, 
        #         TILE_SIZE_M = TILE_SIZE_M, 
        #         BLOCK_K = BLOCK_K,
        #         BLOCK_B = BLOCK_B,
        #         TILE_SIZE_B = TILE_SIZE_B,
        #         B_NUM_BLOCKS = B_NUM_BLOCKS,
        #         BLOCK_SIZE_M = self.block_size
        #     )
        #     print(">>>G", grid, "out")

        return None

    def _backward_pytorch(self, node_flows, element_flows, params, node_mars, 
                          element_mars, param_flows, nids, cids, pids, pfids, 
                          chids, parids, parpids, cs_block_size):
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

        # Flows w.r.t. input elements (product nodes)
        if chids is not None:
            self._backward_pytorch_ele_kernel(
                node_flows, element_flows, params, node_mars, element_mars, 
                param_flows, chids, parids, parpids, cs_block_size
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_pytorch_par_kernel(
                node_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, self.block_size
            )

    @torch.compile(mode = "reduce-overhead")
    def _backward_pytorch_ele_kernel(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                                     params: torch.Tensor, node_mars: torch.Tensor, 
                                     element_mars: torch.Tensor, param_flows: Optional[torch.Tensor], 
                                     chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor,
                                     cs_block_size: int):

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
        
        element_flows[chids] = (node_flows[parids] * params[parpids].unsqueeze(-1) * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

        return None

    @torch.compile(mode = "reduce-overhead")
    def _backward_pytorch_par_kernel(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                     element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                     cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor, ns_block_size: int):

        num_nblocks = nids.size(0)
        num_edges = cids.size(1)
        nids = (nids[:,None].repeat(1, self.block_size) + \
            torch.arange(0, self.block_size, device = nids.device)[None,:]).reshape(num_nblocks * self.block_size)
        cids = cids[:,None,:].repeat(1, self.block_size, 1).reshape(num_nblocks * self.block_size, num_edges)
        pids = (pids[:,None,:].repeat(1, self.block_size, 1) + \
            torch.arange(0, self.block_size, device = cids.device)[None,:,None]).reshape(num_nblocks * self.block_size, num_edges)
        pfids = (pfids[:,None,:].repeat(1, self.block_size, 1) + \
            torch.arange(0, self.block_size, device = cids.device)[None,:,None]).reshape(num_nblocks * self.block_size, num_edges)

        parflows = (node_flows[nids].unsqueeze(1) * params[pids].unsqueeze(-1) * (element_mars[cids] - node_mars[nids].unsqueeze(1)).exp()).sum(dim = 2)

        for i in range(num_nblocks):
            sid, eid = ns_block_size * i, ns_block_size * (i + 1)
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