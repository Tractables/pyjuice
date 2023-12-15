
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
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges
from .backend.index_set import batched_index_set, index_cum
from .compilation import get_sum_layer_forward_stats, sum_layer_forward_compilation, \
                         get_sum_layer_backward_stats, \
                         sum_layer_backward_compilation, next_power_of_2


class SumLayer(Layer, nn.Module):

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int, 
                 global_pid_start: int, global_pfid_start: int, node2tiednodes: dict(),
                 layer_sparsity_tol: Optional[float] = None, 
                 max_num_partitions: Optional[int] = None,
                 max_tied_ns_per_parflow_group: int = 8,
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

        # n_chs:       [num_node_groups]          stores the number of child nodes of each node
        # Note: to allow different nodes to have different `ch_group_size`s, we record the number of 
        #       child **nodes** (instead of # node groups) in `n_chs`
        layer_num_ngroups, layer_num_edges, n_chs = get_sum_layer_forward_stats(self.nodes, global_nid_start)

        self.num_nodes = layer_num_ngroups * self.group_size # Total number of nodes
        self.num_edges = layer_num_edges # Total number of edges

        # Find a good strategy to partition the node groups according to their number of children 
        # to minimize total computation cost
        fw_partition_max_chs = partition_nodes_by_n_edges(
            n_chs, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
        )

        # Since the triton kernels require the maximum number children for each group to be a power of 2,
        # we postprocess the partition sizes
        fw_partition_max_chs = torch.unique(next_power_of_2(fw_partition_max_chs))

        self.num_fw_partitions = len(fw_partition_max_chs) # Number of groups

        # fw_n_partition_ids:      [num_ngroups]           stores the partition id for each node node
        # fw_n_id_in_partition:    [num_ngroups]           stores the index of the node groups in the partition
        # fw_num_ngs_in_partition: [num_fw_partitions]     number of node groups in each partition
        fw_n_partition_ids = torch.zeros([layer_num_ngroups], dtype = torch.long)
        fw_n_id_in_partition = torch.zeros([layer_num_ngroups], dtype = torch.long)
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

        # nids:      List[[partition_size]]                      stores node group ids
        # cids:      List[[partition_size, partition_max_n_chs]] stores indices of child node groups
        # pids:      List[[partition_size, partition_max_n_chs]] stores indices of edge parameters (1st parameter of every group)
        # pfids:     List[[partition_size, partition_max_n_chs]] stores indices of edge parameter flows (1st parameter flow of every group)
        nids, cids, pids, pfids, layer_pid_end, layer_pfid_end = sum_layer_forward_compilation(
            self.nodes, fw_partition_max_chs, fw_n_partition_ids, fw_n_id_in_partition, 
            fw_num_ngs_in_partition, n_chs, global_nid_start, global_pid_start, global_pfid_start, node2tiednodes,
            max_tied_ns_per_parflow_group = max_tied_ns_per_parflow_group,
            # GPU compilation is slightly slower for small layer due to the kernel jit compilation time
            use_cuda = force_gpu_compilation or (not disable_gpu_compilation and (self.num_edges > 1000))
        )

        # Store buffers for the forward pass
        self.partitioned_nids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in nids])
        self.partitioned_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in cids])
        self.partitioned_pids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in pids])
        self.partitioned_pfids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in pfids])

        # Store pre-compiled indices from `cids` and `pids` in the following buffer
        self._cached_fw_pcids = dict()

        # Layer info
        self._layer_nid_range = (layer_nid_start, layer_nid_start + self.num_nodes)
        self._layer_pid_range = (layer_pid_start, layer_pid_end)
        self._layer_pfid_range = (layer_pfid_start, layer_pfid_end)

        ## Initialize backward pass ##

        # A sum layer could have children of different group sizes
        # We separate and partition them into different backward kernels
        ch_gsize2cs, ch_gsize2num_ngroups, ch_gsize2n_pargs, cs2parns = get_sum_layer_backward_stats(nodes)

        # For every possible child group size, we first compute the best partition strategy.
        # We then move on to do the actual compilation
        chids = []
        parids = []
        parpids = []
        cs_group_sizes = []
        for ch_gsize in ch_gsize2n_pargs:

            num_ngroups = ch_gsize2num_ngroups[ch_gsize]
            n_pargs = ch_gsize2n_pargs[ch_gsize]

            # Find a good strategy to partition the node groups according to their number of children 
            # to minimize total computation cost
            bk_partition_max_pars = partition_nodes_by_n_edges(
                n_pargs, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
            )

            # Since the triton kernels require the maximum number children for each group to be a power of 2,
            # we postprocess the partition sizes
            bk_partition_max_pars = torch.unique(next_power_of_2(bk_partition_max_pars))
            num_bk_partitions = bk_partition_max_pars.size(0)

            # bk_n_partition_ids:      [num_ngroups]           stores the partition id for each node group
            # bk_n_id_in_partition:    [num_ngroups]           stores the index of the node groups in the partition
            # bk_num_ngs_in_partition: [num_bk_partitions]     number of node groups in each partition
            bk_n_partition_ids = torch.zeros([num_ngroups], dtype = torch.long)
            bk_n_id_in_partition = torch.zeros([num_ngroups], dtype = torch.long)
            bk_num_ngs_in_partition = torch.zeros([num_bk_partitions], dtype = torch.long)

            min_n_pars = 1
            for partition_id, max_n_pars in enumerate(bk_partition_max_pars):
                criterion = (n_pargs >= min_n_pars) & (n_pargs <= max_n_pars)
                partition_size = criterion.sum().item()

                bk_n_partition_ids[criterion] = partition_id
                bk_n_id_in_partition[criterion] = torch.arange(partition_size)
                bk_num_ngs_in_partition[partition_id] = partition_size

                min_n_pars = max_n_pars + 1

            # chids:      List[[partition_num_chs]]                         stores child group ids
            # parids:     List[[partition_num_chs, partition_max_n_pargs]]  stores parent node groups' ids for each child node
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
            cs_group_sizes.extend([ch_gsize] * num_bk_partitions)

        # Store buffers for the forward pass
        self.partitioned_chids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in chids])
        self.partitioned_parids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in parids])
        self.partitioned_parpids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in parpids])
        self.cs_group_sizes = cs_group_sizes

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

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor) -> None:
        """
        Computes the forward pass of a sum layer:
        ```
        ch_mars = element_mars[cids]
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        node_mars[nids] = (((ch_mars - maxval).exp() * params[pids]).sum(
            dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)
        ```

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
                    node_mars, element_mars, params, nids, cids, pids, partition_id = partition_id
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
                    partition_id = partition_id
                )

        return None

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                 node_mars: torch.Tensor, element_mars: torch.Tensor, 
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None) -> None:
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
        
        ## Compute flows w.r.t. elements (i.e., product nodes) ##
        if not self.provided("bk_group_local_ids"):
            # Evaluate the whole layer
            for partition_id in range(self.num_bk_partitions):
                chids = self.partitioned_chids[partition_id]
                parids = self.partitioned_parids[partition_id]
                parpids = self.partitioned_parpids[partition_id]
                cs_group_size = self.cs_group_sizes[partition_id]

                self._backward(
                    node_flows, element_flows, params, node_mars, 
                    element_mars, param_flows, 
                    chids = chids, parids = parids, parpids = parpids,
                    cs_group_size = cs_group_size
                )

        else:
            # Partial evaluation
            for partition_id in range(self.num_bk_partitions):
                chids = self.grouped_chids[partition_id]
                parids = self.grouped_parids[partition_id]
                parpids = self.grouped_parpids[partition_id]
                cs_group_size = self.cs_group_sizes[partition_id]
                local_ids = self.bk_group_local_ids[partition_id]

                self._backward(
                    node_flows, element_flows, params, node_mars,
                    element_mars, param_flows, 
                    chids = chids, parids = parids, parpids = parpids,
                    cs_group_size = cs_group_size, local_ids = local_ids
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
                    cids = cids, pids = pids, pfids = pfids, partition_id = partition_id
                )

        return None

    def _forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                 pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                 partition_id: int = -1, mode: Optional[str] = None) -> None:
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
            assert mode in ["block_sparse", "sparse"]

        elif params.dim() == 1 and self.group_size >= 16 and num_edges >= 16 and batch_size >= 16:
            # In this case, we should definitely use the block-sparse implementation
            mode = "block_sparse"
        elif self.group_size == 1 and num_edges < 16384:
            # In this case, we should definitely use the sparse implementation
            mode = "sparse"
        elif num_edges < 4:
            # In this case, the block-sparse kernel will have compilation issues
            mode = "sparse"
        else:
            mode = "block_sparse"

        # mode = "sparse" #### debug

        if mode == "block_sparse":
            self._forward_block_sparse(
                node_mars, element_mars, params, nids, cids, pids, local_ids,
                partition_id = partition_id
            )

        elif mode == "sparse":
            self._forward_sparse(
                node_mars, element_mars, params, nids, cids, pids, local_ids,
                partition_id = partition_id
            )

        elif mode == "pytorch":
            self._forward_pytorch_kernel(
                node_mars, element_mars, params, nids, cids, pids, local_ids
            )
        
        else:
            raise ValueError(f"Unexpected mode `{mode}`.")

    @staticmethod
    @triton.jit
    def _fw_triton_block_sparse_tlmm_kernel(node_mars, element_mars, params, nids, cids_start, cids_increment,
                                            pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                            BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                            TILE_SIZE_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr, use_fp16: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node group id from `pid_m`
        ngroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

        # Get the real node group id in the case of partial evaluation
        if partial_eval == 1:
            ngroup_id = tl.load(local_ids + ngroup_id)

        # Node offsets
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

        # Edge offsets
        offs_edge = tl.arange(0, TILE_SIZE_K)

        # Initialize pointers to `params`
        offs_estart = ngroup_id * TILE_SIZE_K + offs_edge
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
        pids_inc_ptr = pids_increment + ngroup_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
        cids_inc_ptr = cids_increment + ngroup_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr)
            emars = tl.load(emars_ptr, mask = mask_batch[None,:])

            emars_max = tl.max(emars, axis = 0)[None,:]
            emars_sub = tl.exp(emars - emars_max)

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
        off_nids = tl.load(nids + ngroup_id)
        offs_nmars = (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
        tl.store(node_mars + offs_nmars, acc, mask = mask_batch[None,:])

    @staticmethod
    @triton.jit
    def _fw_triton_block_sparse_csmm_kernel(node_mars, element_mars, params, nids, cids_start, cids_increment,
                                            pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                            BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                            TILE_SIZE_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr, use_fp16: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node group id from `pid_m`
        ngroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

        # Get the real node group id in the case of partial evaluation
        if partial_eval == 1:
            ngroup_id = tl.load(local_ids + ngroup_id)

        # Node offsets
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_node = tl.max_contiguous(offs_node, TILE_SIZE_M)

        # Edge offsets
        offs_edge = tl.arange(0, TILE_SIZE_K)

        # Initialize pointers to `params`
        offs_estart = ngroup_id * TILE_SIZE_K + offs_edge
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
        pids_inc_ptr = pids_increment + ngroup_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge
        cids_inc_ptr = cids_increment + ngroup_id * (K_NUM_TILES * TILE_SIZE_K) + offs_edge

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr)
            emars = tl.load(emars_ptr, mask = mask_batch[:,None])

            emars_max = tl.max(emars, axis = 1)
            emars_sub = tl.exp(emars - emars_max[:,None])

            if use_fp16 == 1:
                # Simulated matmul kernel + float16
                epars = (epars * (2**12)).to(tl.float16)
                emars = emars.to(tl.float16)
                nmars = tl.sum(epars[:,:,None] * tl.trans(emars_sub)[None,:,:], axis = 1).to(tl.float32) / (2**12)
            else:
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
        off_nids = tl.load(nids + ngroup_id)
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

        num_ngroups = nids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_ngroups * self.group_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.group_size, num_edges, BATCH_SIZE_NP2, 128)
        if base_size >= 64:
            TILE_SIZE_K = base_size
            TILE_SIZE_M = 2048 // base_size
            BLOCK_B = 2048 // base_size
        else:
            remainder = 2048 // (base_size ** 2)

            TILE_SIZE_K = min(2048 // remainder, base_size * remainder, num_edges)
            TILE_SIZE_M = min(2048 // TILE_SIZE_K, self.group_size)
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

        if force_use_fp16:
            assert not force_use_fp32
            use_fp16 = True
        elif force_use_fp32:
            use_fp16 = False
        else:
            if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
                use_fp16 = True
            else:
                use_fp16 = False

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

        # print("========")
        # print(layer_n_nodes)
        # print(grid, grid[0] * grid[1])
        # print(TILE_SIZE_M, TILE_SIZE_K, BLOCK_B)

        # import time
        # torch.cuda.synchronize()
        # t0 = time.time()
        
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
                partial_eval = 1 if local_ids is not None else 0,
                BLOCK_B = BLOCK_B,
                TILE_SIZE_K = TILE_SIZE_K,
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M,
                GROUP_SIZE_M = self.group_size,
                use_fp16 = use_fp16
            )
        else:
            self._fw_triton_block_sparse_csmm_kernel[grid](
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
                partial_eval = 1 if local_ids is not None else 0,
                BLOCK_B = BLOCK_B,
                TILE_SIZE_K = TILE_SIZE_K,
                K_NUM_TILES = K_NUM_TILES,
                TILE_SIZE_M = TILE_SIZE_M,
                GROUP_SIZE_M = self.group_size,
                use_fp16 = use_fp16
            )

        # torch.cuda.synchronize()
        # t1 = time.time()

        # print(f"kernel time: {(t1-t0)*1000:.3f}ms")
        
        return None

    @staticmethod
    @triton.jit
    def _fw_triton_sparse_kernel(node_mars, element_mars, params, nids, cids, pids,
                                 local_ids, batch_size, partial_eval: tl.constexpr, num_edges: tl.constexpr, 
                                 BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        
        pid_b = tl.program_id(axis = 0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(axis = 1) # ID of size-`BLOCK_M` nodes

        # Get inferred node group id from `pid_m`
        ngroup_id = pid_m // (GROUP_SIZE_M // BLOCK_M)
        tile_id = pid_m % (GROUP_SIZE_M // BLOCK_M)

        # Get the real node group id in the case of partial evaluation
        if partial_eval == 1:
            ngroup_id = tl.load(local_ids + ngroup_id)

        # Initialize pointers to `params`
        offs_edge = tl.arange(0, num_edges)
        par_start = tl.load(pids + ngroup_id * num_edges + offs_edge)
        epars_ptr = params + tile_id * BLOCK_M + par_start # [num_edges]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize and load edge mars
        edge_ids = tl.load(cids + ngroup_id * num_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_ids[:,None] * batch_size + \
            offs_batch[None,:]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

        # Compute max and subtract
        emars_max = tl.max(emars, axis = 0)
        emars = tl.exp(emars - emars_max[None,:])

        # Initialize pointers to `node_mars`
        off_nids = tl.load(nids + ngroup_id)
        nmars_ptr = node_mars + \
            (off_nids + tile_id * BLOCK_M) * batch_size + \
            offs_batch

        # Inner loop
        for i in range(0, BLOCK_M):
            epars = tl.load(epars_ptr)

            nmars = tl.log(tl.sum(emars * epars[:,None], axis = 0)) + emars_max

            tl.store(nmars_ptr, nmars, mask = mask_batch)

            # Increment `epars_ptr`
            epars_ptr += 1

            # Increment `nmars_ptr`
            nmars_ptr += batch_size

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

        num_ngroups = nids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_ngroups * self.group_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        assert num_edges <= 16384, "The sparse forward kernel only support nodes with # edges smaller than 16384."

        BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
        BLOCK_M = self.group_size

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, BLOCK_M))

        self._fw_triton_sparse_kernel[grid](
            node_mars = node_mars, 
            element_mars = element_mars, 
            params = params, 
            nids = nids, 
            cids = cids,
            pids = pids,
            local_ids = local_ids, 
            batch_size = batch_size, 
            partial_eval = 1 if local_ids is not None else 0, 
            num_edges = num_edges, 
            BLOCK_B = BLOCK_B, 
            BLOCK_M = BLOCK_M, 
            GROUP_SIZE_M = self.group_size
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

        num_ngroups = nids.size(0)
        num_edges = cids.size(1)
        nids = (nids[:,None].repeat(1, self.group_size) + \
            torch.arange(0, self.group_size, device = nids.device)[None,:]).reshape(num_ngroups * self.group_size)
        cids = cids[:,None,:].repeat(1, self.group_size, 1).reshape(num_ngroups * self.group_size, num_edges)
        pids = (pids[:,None,:].repeat(1, self.group_size, 1) + \
            torch.arange(0, self.group_size, device = cids.device)[None,:,None]).reshape(num_ngroups * self.group_size, num_edges)

        ch_mars = element_mars[cids]
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        node_mars[nids] = (((ch_mars - maxval).exp() * params[pids].unsqueeze(-1)).sum(
            dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

        return None

    def _backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                  params: torch.Tensor, node_mars: torch.Tensor, 
                  element_mars: torch.Tensor, param_flows: torch.Tensor,
                  nids: Optional[torch.Tensor] = None, cids: Optional[torch.Tensor] = None, 
                  pids: Optional[torch.Tensor] = None, pfids: Optional[torch.Tensor] = None, 
                  chids: Optional[torch.Tensor] = None, parids: Optional[torch.Tensor] = None, 
                  parpids: Optional[torch.Tensor] = None, 
                  cs_group_size: int = 0, local_ids: Optional[torch.Tensor] = None, 
                  partition_id: int = -1, mode: Optional[str] = None) -> None:
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
            num_edges = cids.size(1) * self.group_size
        else:
            num_edges = parids.size(1) * self.group_size
        batch_size = node_flows.size(1)

        if mode is not None:
            assert mode in ["block_sparse", "sparse", "pytorch"]
        elif params.dim() == 1 and self.group_size >= 16 and num_edges >= 16 and batch_size >= 16:
            # In this case, we should definitely use the block-sparse implementation
            mode = "block_sparse"
        elif (cs_group_size == 1 or self.group_size == 1) and num_edges < 16384:
            # In this case, we should definitely use the sparse implementation
            mode = "sparse"
        elif num_edges < 4 or batch_size < 4:
            # In this case, the block-sparse kernel will have compilation issues
            mode = "sparse"
        else:
            mode = "block_sparse"

        if mode == "block_sparse":
            self._backward_block_sparse(
                node_flows, element_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, chids, parids, parpids, cs_group_size, local_ids, 
                partition_id = partition_id
            )
        elif mode == "sparse":
            self._backward_sparse(
                node_flows, element_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, chids, parids, parpids, cs_group_size, local_ids, 
                partition_id = partition_id
            )
        elif mode == "pytorch":
            self._backward_pytorch(
                node_flows, element_flows, params, node_mars, 
                element_mars, param_flows, nids, cids, pids, pfids, 
                chids, parids, parpids, cs_group_size
            )
        else:
            raise ValueError(f"Not supported mode `{mode}`.")

    def _backward_block_sparse(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                               params: torch.Tensor, node_mars: torch.Tensor, 
                               element_mars: torch.Tensor, param_flows: torch.Tensor, 
                               nids: Optional[torch.Tensor], cids: Optional[torch.Tensor], pids: Optional[torch.Tensor], pfids: Optional[torch.Tensor],
                               chids: Optional[torch.Tensor], parids: Optional[torch.Tensor], parpids: Optional[torch.Tensor], 
                               cs_group_size: int, local_ids: Optional[torch.Tensor] = None,
                               partition_id: int = -1) -> None:
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
                cs_group_size = cs_group_size, local_ids = local_ids, 
                partition_id = partition_id
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_block_sparse_par_flows(
                node_flows, params, node_mars, element_mars, param_flows,
                nids = nids, cids = cids, pids = pids, pfids = pfids
            )

        return None

    @staticmethod
    @triton.jit
    def _bk_triton_block_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                           chids, parids_start, parids_increment, parpids_start, parpids_increment, 
                                           local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr, ptr_inc_step: tl.constexpr, 
                                           BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, TILE_SIZE_M: tl.constexpr, 
                                           GROUP_SIZE_M: tl.constexpr, GROUP_SIZE_K: tl.constexpr, OP_MODE: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node group id from `pid_m`
        elegroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

        # Get the real node group id in the case of partial evaluation
        if partial_eval == 1:
            elegroup_id = tl.load(local_ids + elegroup_id)

        # Initialize pointers to `params`
        offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        offs_edge = tl.arange(0, TILE_SIZE_K)
        offs_edge_gid = offs_edge // GROUP_SIZE_K
        offs_edge_nid = (offs_edge % GROUP_SIZE_K)
        par_start = tl.load(parpids_start + elegroup_id * ptr_inc_step + offs_edge_gid)
        epars_ptr = params + \
            offs_ele[:,None] * GROUP_SIZE_K + \
            (par_start + offs_edge_nid)[None,:] # [TILE_SIZE_M, TILE_SIZE_K]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `node_mars`
        edge_start = tl.load(parids_start + elegroup_id * ptr_inc_step + offs_edge_gid)
        nmars_ptr = node_mars + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]
        nflows_ptr = node_flows + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

        # Initialize pointers to `element_mars`
        off_eleids = tl.load(chids + elegroup_id)
        offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
        tmp_emars = tl.load(element_mars + offs_elemfs, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
        emars_max = tl.max(tmp_emars, axis = 0) # [BLOCK_B]

        # Batch increment pointers
        parids_inc_ptr = parids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid
        parpids_inc_ptr = parpids_increment + elegroup_id * (K_NUM_TILES * ptr_inc_step) + offs_edge_gid

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)

        for k in range(0, K_NUM_TILES):
            epars = tl.load(epars_ptr) # [TILE_SIZE_M, TILE_SIZE_K]
            nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]
            nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

            nflows_div_mars = nflows * tl.exp(emars_max[None,:] - nmars)
            
            if OP_MODE == 0:
                # Built-in matmul kernel of triton + float16
                epars = epars.to(tl.float16)
                nflows_div_mars = nflows_div_mars.to(tl.float16)
                eflows = tl.dot(epars, nflows_div_mars).to(tl.float32)
            if OP_MODE == 1:
                # Built-in matmul kernel of triton + float32
                eflows = tl.dot(epars, nflows_div_mars)
            if OP_MODE == 2:
                # Simulated matmul kernel + float16
                epars = epars.to(tl.float16)
                nflows_div_mars = nflows_div_mars.to(tl.float16)
                eflows = tl.sum(epars[:,:,None] * nflows_div_mars[None,:,:], axis = 1).to(tl.float32)
            if OP_MODE == 3:
                # Simulated matmul kernel + float32
                eflows = tl.sum(epars[:,:,None] * nflows_div_mars[None,:,:], axis = 1)

            acc += eflows

            # Increment `epars_ptr`
            parpids_inc = tl.load(parpids_inc_ptr)
            epars_ptr += parpids_inc[None,:]
            parpids_inc_ptr += ptr_inc_step

            # Increment `nmars_ptr`
            parids_inc = tl.load(parids_inc_ptr)
            nmars_ptr += parids_inc[:,None] * batch_size
            nflows_ptr += parids_inc[:,None] * batch_size
            parids_inc += ptr_inc_step

        # Initialize pointers to `element_mars`
        off_eleids = tl.load(chids + elegroup_id)
        offs_elemfs = (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:]
        emars = tl.load(element_mars + offs_elemfs, mask = mask_batch[None,:]) # [TILE_SIZE_K, BLOCK_B]

        eflows = acc * tl.exp(emars - emars_max[None,:])
        tl.store(element_flows + offs_elemfs, eflows, mask = mask_batch[None,:])

    def _backward_block_sparse_ele_flows(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                                         params: torch.Tensor, node_mars: torch.Tensor,
                                         element_mars: torch.Tensor, chids: torch.Tensor, parids: torch.Tensor,
                                         parpids: torch.Tensor, cs_group_size: int, local_ids: Optional[torch.Tensor] = None,
                                         partition_id: int = -1, force_use_fp16: bool = False, force_use_fp32: bool = False) -> None:

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_ngroups = chids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_ngroups * cs_group_size
        num_edges = parids.size(1) * self.group_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.group_size, num_edges, BATCH_SIZE_NP2, 128)
        if base_size >= 64:
            TILE_SIZE_K = base_size
            TILE_SIZE_M = 2048 // base_size
            BLOCK_B = 2048 // base_size
        else:
            remainder = 2048 // (base_size ** 2)

            TILE_SIZE_K = min(2048 // remainder, base_size * remainder, num_edges)
            TILE_SIZE_M = min(2048 // TILE_SIZE_K, cs_group_size)
            BLOCK_B = min(2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        K_NUM_TILES = num_edges // TILE_SIZE_K

        assert TILE_SIZE_K >= 4, f"`TILE_SIZE_K` should be greater than 4 (but got {TILE_SIZE_K}) in order to use the block-sparse kernel. " \
                                  "This is an internal error of PyJuice. Please consider checking the kernel dispatching criterions and use the " \
                                  "corresponding sparse kernel instead."

        signature = ("block_sparse", partition_id, TILE_SIZE_K)
        if signature not in self._cached_bk_parids:
            # Pre-compute pointer increments for `parids` and `parpids`

            if TILE_SIZE_K <= self.group_size:
                ptr_inc_step = 1

                num_rep = self.group_size // TILE_SIZE_K
                parids = (parids[:,:,None].repeat(1, 1, num_rep) + \
                    torch.arange(0, self.group_size, TILE_SIZE_K, device = parids.device)[None,None,:]).reshape(
                        parids.size(0), K_NUM_TILES, 1)
                parpids = (parpids[:,:,None].repeat(1, 1, num_rep) + \
                    torch.arange(0, self.group_size * cs_group_size, TILE_SIZE_K * cs_group_size, device = parpids.device)[None,None,:]).reshape(
                        parpids.size(0), K_NUM_TILES, 1)

            else:
                ptr_inc_step = TILE_SIZE_K // self.group_size

                parids = parids.reshape(parids.size(0), K_NUM_TILES, ptr_inc_step)
                parpids = parpids.reshape(parpids.size(0), K_NUM_TILES, ptr_inc_step)

            parids_start = parids[:,0,:].contiguous()
            parids_increment = torch.cat(
                (parids[:,1:,:] - parids[:,:-1,:], parids[:,0:1,:] * 0),
                dim = 1
            ).contiguous()

            parpids_start = parpids[:,0,:].contiguous()
            parpids_increment = torch.cat(
                (parpids[:,1:,:] - parpids[:,:-1], parpids[:,0:1,:] * 0),
                dim = 1
            ).contiguous()

            self._cached_bk_parids[signature] = [parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step]
        else:
            parids_start, parids_increment, parpids_start, parpids_increment, ptr_inc_step = self._cached_bk_parids[signature]

        if force_use_fp16:
            assert not force_use_fp32
            use_fp16 = True
        elif force_use_fp32:
            use_fp16 = False
        else:
            if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
                use_fp16 = True
            else:
                use_fp16 = False

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and BLOCK_B >= 16:
            if use_fp16:
                OP_MODE = 0
            else:
                OP_MODE = 1
        else:
            if use_fp16:
                OP_MODE = 2
            else:
                OP_MODE = 3

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

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
            partial_eval = 1 if local_ids is not None else 0,
            ptr_inc_step = ptr_inc_step,
            BLOCK_B = BLOCK_B, 
            TILE_SIZE_K = TILE_SIZE_K, 
            K_NUM_TILES = K_NUM_TILES,
            TILE_SIZE_M = TILE_SIZE_M, 
            GROUP_SIZE_M = cs_group_size,
            GROUP_SIZE_K = self.group_size,
            OP_MODE = OP_MODE
        )

        return None

    @staticmethod
    @triton.jit
    def _bk_triton_block_sparse_par_kernel(node_flows, node_mars, element_mars, params, param_flows, nids, cids, pids, pfids,
                                           batch_size: tl.constexpr, num_edges: tl.constexpr, TILE_SIZE_B: tl.constexpr, 
                                           B_NUM_TILES: tl.constexpr, TILE_SIZE_K: tl.constexpr, 
                                           TILE_SIZE_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr, OP_MODE: tl.constexpr):

        pid_k = tl.program_id(0) # ID of size-`TILE_SIZE_K` edges
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node group id from `pid_m`
        ngroup_id = pid_m // (GROUP_SIZE_M // TILE_SIZE_M)
        tile_id = pid_m % (GROUP_SIZE_M // TILE_SIZE_M)

        # Batch offsets and mask
        offs_batch = tl.arange(0, TILE_SIZE_B)
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        offs_edge = tl.arange(0, TILE_SIZE_K) + pid_k * TILE_SIZE_K
        edge_start = tl.load(cids + ngroup_id * num_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_start[None,:] * batch_size + \
            offs_batch[:,None] # [TILE_SIZE_B, TILE_SIZE_K]

        # Initialize pointers to `node_flows` and `node_mars`
        offs_node = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
        off_nids = tl.load(nids + ngroup_id)
        nmars_ptr = node_mars + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]
        nflows_ptr = node_flows + (off_nids + offs_node[:,None]) * batch_size + offs_batch[None,:]

        # Inner loop
        acc = tl.zeros([TILE_SIZE_M, TILE_SIZE_K], dtype = tl.float32)
        
        for b in range(0, B_NUM_TILES):
            emars = tl.load(emars_ptr, mask = mask_batch[:,None]) # [TILE_SIZE_B, TILE_SIZE_K]
            nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, TILE_SIZE_B]
            nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [TILE_SIZE_M, TILE_SIZE_B]

            emars_max = tl.max(emars, axis = 1)
            nflows_div_mars = nflows * tl.exp(emars_max[None,:] - nmars)

            emars = tl.exp(emars - emars_max[:,None])

            if OP_MODE == 0:
                # Built-in matmul kernel of triton + float16
                nflows_div_mars = nflows_div_mars.to(tl.float16)
                emars = emars.to(tl.float16)
                pflows = tl.dot(nflows_div_mars, emars).to(tl.float32)
            if OP_MODE == 1:
                # Built-in matmul kernel of triton + float32
                pflows = tl.dot(nflows_div_mars, emars).to(tl.float32)
            if OP_MODE == 2:
                # Simulated matmul kernel + float16
                nflows_div_mars = nflows_div_mars.to(tl.float16)
                emars = emars.to(tl.float16)
                pflows = tl.sum(nflows_div_mars[:,:,None] * emars[None,:,:], axis = 1).to(tl.float32)
            if OP_MODE == 3:
                # Simulated matmul kernel + float32
                pflows = tl.sum(nflows_div_mars[:,:,None] * emars[None,:,:], axis = 1)

            acc += pflows

            # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
            emars_ptr += TILE_SIZE_B
            nmars_ptr += TILE_SIZE_B
            nflows_ptr += TILE_SIZE_B

            # Update batch mask
            offs_batch += TILE_SIZE_B
            mask_batch = offs_batch < batch_size

        par_start = tl.load(pids + ngroup_id * num_edges + offs_edge)
        epars_offsets = offs_node[:,None] + par_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        epars = tl.load(params + epars_offsets)

        parflow_start = tl.load(pfids + ngroup_id * num_edges + offs_edge)
        eparflows_offsets = offs_node[:,None] + parflow_start[None,:] # [TILE_SIZE_M, TILE_SIZE_K]
        
        curr_pflows = acc * epars

        tl.atomic_add(param_flows + eparflows_offsets, curr_pflows)

    def _backward_block_sparse_par_flows(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                         element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                         cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor,
                                         force_use_fp16: bool = False, force_use_fp32: bool = False) -> None:
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

        num_ngroups = nids.size(0)
        layer_n_nodes = num_ngroups * self.group_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        # Heuristic to set `TILE_SIZE_M`, `TILE_SIZE_K`, and `BLOCK_B`
        base_size = min(self.group_size, num_edges, BATCH_SIZE_NP2, 64)
        if base_size >= 64:
            TILE_SIZE_B = base_size
            TILE_SIZE_M = 2048 // base_size
            TILE_SIZE_K = 2048 // base_size
        else:
            remainder = 2048 // (base_size ** 2)

            TILE_SIZE_B = min(2048 // remainder, base_size * remainder, BATCH_SIZE_NP2)
            TILE_SIZE_M = min(2048 // TILE_SIZE_B, self.group_size)
            TILE_SIZE_K = min(2048 // TILE_SIZE_B, num_edges)
        B_NUM_TILES = batch_size // TILE_SIZE_B

        assert TILE_SIZE_B >= 4, f"`TILE_SIZE_B` should be greater than 4 (but got {TILE_SIZE_B}) in order to use the block-sparse kernel. " \
                                  "This is an internal error of PyJuice. Please consider checking the kernel dispatching criterions and use the " \
                                  "corresponding sparse kernel instead."

        if force_use_fp16:
            assert not force_use_fp32
            use_fp16 = True
        elif force_use_fp32:
            use_fp16 = False
        else:
            if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and TILE_SIZE_B >= 16:
                use_fp16 = True
            else:
                use_fp16 = False

        if TILE_SIZE_M >= 16 and TILE_SIZE_K >= 16 and TILE_SIZE_B >= 16:
            if use_fp16:
                OP_MODE = 0
            else:
                OP_MODE = 1
        else:
            if use_fp16:
                OP_MODE = 2
            else:
                OP_MODE = 3

        grid = (triton.cdiv(num_edges, TILE_SIZE_K), triton.cdiv(layer_n_nodes, TILE_SIZE_M))

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
            TILE_SIZE_B = TILE_SIZE_B, 
            B_NUM_TILES = B_NUM_TILES, 
            TILE_SIZE_K = TILE_SIZE_K, 
            TILE_SIZE_M = TILE_SIZE_M, 
            GROUP_SIZE_M = self.group_size,
            OP_MODE = OP_MODE
        )

    def _backward_sparse(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                         params: torch.Tensor, node_mars: torch.Tensor, 
                         element_mars: torch.Tensor, param_flows: torch.Tensor, 
                         nids: Optional[torch.Tensor], cids: Optional[torch.Tensor], pids: Optional[torch.Tensor], pfids: Optional[torch.Tensor],
                         chids: Optional[torch.Tensor], parids: Optional[torch.Tensor], parpids: Optional[torch.Tensor], 
                         cs_group_size: int, local_ids: Optional[torch.Tensor] = None,
                         partition_id: int = -1) -> None:
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
                cs_group_size = cs_group_size, local_ids = local_ids
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_sparse_par_flows(
                node_flows, params, node_mars, element_mars, param_flows,
                nids = nids, cids = cids, pids = pids, pfids = pfids
            )

        return None

    @staticmethod
    @triton.jit
    def _bk_triton_sparse_ele_kernel(node_flows, element_flows, node_mars, element_mars, params, 
                                     chids, parids, parpids, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                     n_edge_groups: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, 
                                     GROUP_SIZE_K: tl.constexpr):

        pid_b = tl.program_id(0) # ID of size-`BLOCK_B` batches
        pid_m = tl.program_id(1) # ID of size-`TILE_SIZE_M` nodes

        # Get inferred node group id from `pid_m`
        elegroup_id = pid_m

        # Get the real node group id in the case of partial evaluation
        if partial_eval == 1:
            elegroup_id = tl.load(local_ids + elegroup_id)

        # Initialize pointers to `params`
        offs_edge = tl.arange(0, n_edge_groups * GROUP_SIZE_K) # I.e., [0, num_edges)
        offs_edge_gid = offs_edge // GROUP_SIZE_K
        offs_edge_nid = (offs_edge % GROUP_SIZE_K)
        par_start = tl.load(parpids + elegroup_id * n_edge_groups + offs_edge_gid)
        epars_ptr = params + par_start + offs_edge_nid # [num_edges]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize `node_flows` and `node_mars`
        edge_start = tl.load(parids + elegroup_id * n_edge_groups + offs_edge_gid)
        nmars_ptr = node_mars + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:]
        nmars = tl.load(nmars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]
        nflows_ptr = node_flows + \
            (edge_start + offs_edge_nid)[:,None] * batch_size + \
            offs_batch[None,:]
        nflows = tl.load(nflows_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]

        # Initialize pointers to `element_flows` and `element_mars`
        off_eleids = tl.load(chids + elegroup_id)
        eflows_ptr = element_flows + off_eleids * batch_size + offs_batch # [BLOCK_B]
        emars_ptr = element_mars + off_eleids * batch_size + offs_batch # [BLOCK_B]

        # Inner loop
        for i in range(0, BLOCK_M):
            epars = tl.load(epars_ptr) # [num_edges]
            emars = tl.load(emars_ptr, mask = mask_batch) # [BLOCK_B]

            eflows = tl.sum(nflows * epars[:,None] * tl.exp(emars[None,:] - nmars), axis = 0)

            tl.store(eflows_ptr, eflows, mask = mask_batch)

            # Increment `epars_ptr`
            epars_ptr += GROUP_SIZE_K

            # Increment `emars_ptr` and `eflows_ptr`
            emars_ptr += batch_size
            eflows_ptr += batch_size

    def _backward_sparse_ele_flows(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                                   params: torch.Tensor, node_mars: torch.Tensor,
                                   element_mars: torch.Tensor, chids: torch.Tensor, parids: torch.Tensor,
                                   parpids: torch.Tensor, cs_group_size: int, local_ids: Optional[torch.Tensor] = None) -> None:

        assert params.dim() == 1, "Expecting a 1D `params`."

        num_ngroups = chids.size(0) if local_ids is None else local_ids.size(0)
        layer_n_nodes = num_ngroups * cs_group_size
        n_edge_groups = parids.size(1)
        num_edges = n_edge_groups * self.group_size
        batch_size = node_flows.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        assert num_edges <= 16384, "The sparse backward kernel only support nodes with # edges smaller than 16384."

        BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
        BLOCK_M = cs_group_size

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
            n_edge_groups = n_edge_groups,
            BLOCK_B = BLOCK_B, 
            BLOCK_M = BLOCK_M,
            GROUP_SIZE_K = self.group_size
        )

        return None

    @staticmethod
    @triton.jit
    def _bk_triton_sparse_par_kernel(node_flows, node_mars, element_mars, params, param_flows, nids, cids, pids, pfids,
                                     batch_size: tl.constexpr, num_edges: tl.constexpr, BLOCK_M: tl.constexpr, 
                                     BLOCK_B: tl.constexpr, B_NUM_BLOCKS: tl.constexpr):

        pid_m = tl.program_id(0) # ID of size-`BLOCK_M` nodes

        # Get inferred node group id from `pid_m`
        ngroup_id = pid_m // BLOCK_M
        tile_id = pid_m % BLOCK_M

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B)
        mask_batch = offs_batch < batch_size

        # Initialize pointers to `element_mars`
        offs_edge = tl.arange(0, num_edges)
        edge_start = tl.load(cids + ngroup_id * num_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_start[:,None] * batch_size + \
            offs_batch[None,:] # [num_edges, BLOCK_B]

        # Initialize pointers to `node_flows` and `node_mars`
        off_nids = tl.load(nids + ngroup_id)
        nmars_ptr = node_mars + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]
        nflows_ptr = node_flows + (off_nids + tile_id) * batch_size + offs_batch # [BLOCK_B]

        # Inner loop
        acc = tl.zeros([num_edges], dtype = tl.float32)

        for b in range(0, B_NUM_BLOCKS):
            emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [num_edges, BLOCK_B]
            nmars = tl.load(nmars_ptr, mask = mask_batch) # [BLOCK_B]
            nflows = tl.load(nflows_ptr, mask = mask_batch) # [BLOCK_B]

            pflows = tl.sum(nflows[None,:] * tl.exp(emars - nmars[None,:]), axis = 1)

            acc += pflows

            # Increment `emars_ptr`, `nmars_ptr`, and `nmars_ptr`
            emars_ptr += BLOCK_B
            nmars_ptr += BLOCK_B
            nflows_ptr += BLOCK_B

            # Update batch mask
            offs_batch += BLOCK_B
            mask_batch = offs_batch < batch_size

        par_start = tl.load(pids + ngroup_id * num_edges + offs_edge)
        epars_ptr = params + par_start + tile_id
        epars = tl.load(epars_ptr) # [num_edges]

        parflow_start = tl.load(pfids + ngroup_id * num_edges + offs_edge)
        eparflows_ptr = param_flows + parflow_start + tile_id
        
        curr_pflows = acc * epars

        tl.atomic_add(eparflows_ptr, curr_pflows)

    def _backward_sparse_par_flows(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                   element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                   cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor) -> None:
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

        num_ngroups = nids.size(0)
        layer_n_nodes = num_ngroups * self.group_size
        num_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        assert num_edges <= 16384, "The sparse backward kernel only support nodes with # edges smaller than 16384."

        BLOCK_B = max(min(2048 // num_edges, BATCH_SIZE_NP2), 1)
        BLOCK_M = self.group_size

        grid = (layer_n_nodes,)

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
            batch_size = batch_size, 
            num_edges = num_edges, 
            BLOCK_M = BLOCK_M,
            BLOCK_B = BLOCK_B,
            B_NUM_BLOCKS = triton.cdiv(batch_size, BLOCK_B)
        )

    def _backward_pytorch(self, node_flows, element_flows, params, node_mars, 
                          element_mars, param_flows, nids, cids, pids, pfids, 
                          chids, parids, parpids, cs_group_size):
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
                param_flows, chids, parids, parpids, cs_group_size
            )

        # Flows w.r.t. parameters
        if param_flows is not None and nids is not None:
            self._backward_pytorch_par_kernel(
                node_flows, params, node_mars, element_mars, param_flows, 
                nids, cids, pids, pfids, self.group_size
            )

    @torch.compile(mode = "reduce-overhead")
    def _backward_pytorch_ele_kernel(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                                     params: torch.Tensor, node_mars: torch.Tensor, 
                                     element_mars: torch.Tensor, param_flows: Optional[torch.Tensor], 
                                     chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor,
                                     cs_group_size: int):

        num_ngroups = chids.size(0)
        num_egroups = parids.size(1)
        parids = (parids[:,:,None].repeat(1, 1, self.group_size) + torch.arange(0, self.group_size, device = parids.device)).reshape(num_ngroups, num_egroups * self.group_size)
        parpids = (parpids[:,:,None] + torch.arange(0, self.group_size, device = parids.device)).reshape(
            num_ngroups, num_egroups * self.group_size)

        chids = (chids[:,None].repeat(1, cs_group_size) + torch.arange(0, cs_group_size, device = chids.device)).reshape(num_ngroups * cs_group_size)
        parids = parids[:,None,:].repeat(1, cs_group_size, 1).reshape(num_ngroups * cs_group_size, num_egroups * self.group_size)
        parpids = (parpids[:,None,:].repeat(1, cs_group_size, 1) + torch.arange(0, cs_group_size * self.group_size, self.group_size, device = parpids.device)[None,:,None]).reshape(
            num_ngroups * cs_group_size, num_egroups * self.group_size
        )
        
        element_flows[chids] = (node_flows[parids] * params[parpids].unsqueeze(-1) * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

        return None

    @torch.compile(mode = "reduce-overhead")
    def _backward_pytorch_par_kernel(self, node_flows: torch.Tensor, params: torch.Tensor, node_mars: torch.Tensor, 
                                     element_mars: torch.Tensor, param_flows: torch.Tensor, nids: torch.Tensor, 
                                     cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor, ns_group_size: int):

        num_ngroups = nids.size(0)
        num_edges = cids.size(1)
        nids = (nids[:,None].repeat(1, self.group_size) + \
            torch.arange(0, self.group_size, device = nids.device)[None,:]).reshape(num_ngroups * self.group_size)
        cids = cids[:,None,:].repeat(1, self.group_size, 1).reshape(num_ngroups * self.group_size, num_edges)
        pids = (pids[:,None,:].repeat(1, self.group_size, 1) + \
            torch.arange(0, self.group_size, device = cids.device)[None,:,None]).reshape(num_ngroups * self.group_size, num_edges)
        pfids = (pfids[:,None,:].repeat(1, self.group_size, 1) + \
            torch.arange(0, self.group_size, device = cids.device)[None,:,None]).reshape(num_ngroups * self.group_size, num_edges)

        parflows = (node_flows[nids].unsqueeze(1) * params[pids].unsqueeze(-1) * (element_mars[cids] - node_mars[nids].unsqueeze(1)).exp()).sum(dim = 2)

        for i in range(num_ngroups):
            sid, eid = ns_group_size * i, ns_group_size * (i + 1)
            param_flows[pfids[sid:eid,:]] += parflows[sid:eid,:]

        return None

    def _prepare_scope2nids(self, prod_scope_eleids: Sequence[Tuple[BitSet, torch.Tensor]]):
        if not (hasattr(self, "fw_scope2localids") and hasattr(self, "bk_scope2localids")):
            fw_scope2localids = dict()
            bk_scope2localids = dict()

            # Forward local indices
            global_nid = self.global_nid_range[0]
            for ns in self.nodes:
                scope = ns.scope

                s_nid = global_nid
                e_nid = global_nid + ns.num_nodes

                with torch.no_grad():
                    if scope not in fw_scope2localids:
                        fw_scope2localids[scope] = [
                            torch.zeros([0], dtype = torch.long).to(self.grouped_nids[0].device) for _ in range(self.num_fw_groups)
                        ]

                    for group_id in range(self.num_fw_groups):
                        nids = self.grouped_nids[group_id]
                        group_local_ids = torch.where((nids >= s_nid) & (nids < e_nid))[0]

                        fw_scope2localids[scope][group_id] = torch.cat(
                            (fw_scope2localids[scope][group_id], group_local_ids), dim = 0
                        )

                global_nid += ns.num_nodes

            # Backward local indices
            for scope, ele_id_range in prod_scope_eleids:
                s_eid, e_eid = ele_id_range

                with torch.no_grad():
                    if scope not in bk_scope2localids:
                        bk_scope2localids[scope] = [
                            torch.zeros([0], dtype = torch.long).to(self.grouped_nids[0].device) for _ in range(self.num_bk_groups)
                        ]

                    for group_id in range(self.num_bk_groups):
                        chids = self.grouped_chids[group_id]
                        group_local_ids = torch.where((chids >= s_eid) & (chids < e_eid))[0]

                        bk_scope2localids[scope][group_id] = torch.cat(
                            (bk_scope2localids[scope][group_id], group_local_ids), dim = 0
                        )

            self.fw_scope2localids = fw_scope2localids
            self.bk_scope2localids = bk_scope2localids
