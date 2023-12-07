
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
                 param_ends: Sequence, tied_param_ids: Sequence,
                 tied_param_group_ids: Sequence, tied_param_ends: Sequence,
                 ch_prod_layer_size: int, layer_sparsity_tol: Optional[float] = None, 
                 max_num_partitions: Optional[int] = None,
                 disable_gpu_compilation: bool = False) -> None:

        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."

        self.nodes = nodes
        self.ch_prod_layer_size = ch_prod_layer_size

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

        min_n_chs = 0
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
        nids, cids, pids, param_ends = sum_layer_forward_compilation(
            self.nodes, fw_partition_max_chs, fw_n_partition_ids, fw_n_id_in_partition, fw_num_ngs_in_partition, 
            n_chs, global_nid_start, ch_prod_layer_size, param_ends = param_ends,
            # GPU compilation is slightly slower for small layer due to the kernel jit compilation time
            use_cuda = True # not disable_gpu_compilation and (self.num_edges > 1000)
        )

        # Store buffers for the forward pass
        self.partitioned_nids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in nids])
        self.partitioned_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in cids])
        self.partitioned_pids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in pids])

        # Store pre-compiled indices from `cids` and `pids` in the following buffer
        self._cached_fw_pcids = dict()

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

            min_n_pars = 0
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
                use_cuda = not disable_gpu_compilation and (self.num_edges > 1000)
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

    def __init__old(self, nodes: Sequence[SumNodes], global_nid_start: int, 
                 param_ends: Sequence, tied_param_ids: Sequence,
                 tied_param_group_ids: Sequence, tied_param_ends: Sequence,
                 ch_prod_layer_size: int, layer_sparsity_tol: float = 0.0, 
                 max_num_partitions: Optional[int] = None,
                 disable_gpu_compilation: bool = False) -> None:

        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."

        self.nodes = nodes
        self.ch_prod_layer_size = ch_prod_layer_size

        ## Get layer statistics & prepare for compilation ##

        # n_chs:       [num_nodes]            stores the number of child nodes of each node
        layer_num_nodes, layer_num_edges, n_chs = get_sum_layer_stats(self.nodes, global_nid_start)

        self.num_nodes = layer_num_nodes # Total number of nodes
        self.num_edges = layer_num_edges # Total number of edges

        # Find a good strategy to partition the nodes into groups according to their number of children 
        # to minimize total computation cost
        fw_group_max_chs = partition_nodes_by_n_edges(
            n_chs, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
        )

        # Since the triton kernels require the maximum number children for each group to be a power of 2,
        # we postprocess the group sizes
        fw_group_max_chs = torch.unique(next_power_of_2(fw_group_max_chs))
        
        self.num_fw_groups = len(fw_group_max_chs) # Number of groups

        # fw_n_group_ids:     [num_nodes]          stores the group id for each node
        # fw_n_id_in_group:   [num_nodes]          stores the index of the nodes in the group
        # fw_num_ns_in_group: [num_fw_groups]      number of nodes in each group
        fw_n_group_ids = torch.zeros([self.num_nodes], dtype = torch.long)
        fw_n_id_in_group = torch.zeros([self.num_nodes], dtype = torch.long)
        fw_num_ns_in_group = torch.zeros([self.num_fw_groups], dtype = torch.long)

        min_n_chs = 0
        for group_id, max_n_chs in enumerate(fw_group_max_chs):
            criterion = (n_chs >= min_n_chs) & (n_chs <= max_n_chs)
            group_size = criterion.sum().item()

            fw_n_group_ids[criterion] = group_id
            fw_n_id_in_group[criterion] = torch.arange(group_size)
            fw_num_ns_in_group[group_id] = group_size

            min_n_chs = max_n_chs + 1

        ## Initialize forward pass ##

        # nids:      List[[group_size]]                  stores node ids
        # cids:      List[[group_size, group_max_n_chs]] stores indices of child nodes
        # pids:      List[[group_size, group_max_n_chs]] stores indices of edge parameters
        # ch_n_pars: [ch_prod_layer_size]                stores the number of parents for each child node
        nids, cids, pids, ch_n_pars, param_ends = sum_layer_forward_compilation(
            self.nodes, fw_group_max_chs, fw_n_group_ids, fw_n_id_in_group, fw_num_ns_in_group, 
            n_chs, global_nid_start, ch_prod_layer_size, param_ends = param_ends,
            # GPU compilation is slightly slower for small layer due to the kernel jit compilation time
            use_cuda = not disable_gpu_compilation and (self.num_edges > 1000)
        )

        # Store buffers for the forward pass
        self.grouped_nids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in nids])
        self.grouped_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in cids])
        self.grouped_pids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in pids])

        ## Initialize backward pass ##

        # Find a good strategy to partition the child nodes into groups according to their number of parents 
        # to minimize total computation cost
        ch_n_pars = ch_n_pars[1:] # Strip away the dummy node. We will never use it in the following
        bk_group_max_pars = partition_nodes_by_n_edges(
            ch_n_pars, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
        )

        # Since the triton kernels require the maximum number children for each group to be a power of 2,
        # we postprocess the group sizes
        bk_group_max_pars = torch.unique(next_power_of_2(bk_group_max_pars))

        self.num_bk_groups = len(bk_group_max_pars) # Number of groups

        # bk_n_group_ids:     [ch_prod_layer_size]  stores the group id for each (child) node
        # bk_n_id_in_group:   [ch_prod_layer_size]  stores the index of the (child) nodes in the group
        # bk_num_ns_in_group: [num_bk_groups]       number of nodes in each group
        # chids:              List[[group_size]]    stores child ids
        bk_n_group_ids = torch.zeros([self.ch_prod_layer_size], dtype = torch.long)
        bk_n_id_in_group = torch.zeros([self.ch_prod_layer_size], dtype = torch.long)
        bk_num_ns_in_group = torch.zeros([self.num_bk_groups], dtype = torch.long)
        chids = []

        min_n_pars = 0
        for group_id, max_n_pars in enumerate(bk_group_max_pars):
            criterion = (ch_n_pars >= min_n_pars) & (ch_n_pars <= max_n_pars)
            filtered_idxs = torch.where(criterion)[0] + 1 # plus one to offset the dummy node since it is removed from `ch_n_pars`
            group_size = criterion.sum().item()

            bk_n_group_ids[filtered_idxs] = group_id
            bk_n_id_in_group[filtered_idxs] = torch.arange(group_size)
            bk_num_ns_in_group[group_id] = group_size
            chids.append(filtered_idxs)

            min_n_pars = max_n_pars + 1

        # parids:     List[[group_ch_size, group_max_n_pars]]  stores parameter ids for each child node
        # parpids:    List[[group_ch_size, group_max_n_pars]]  param id for the edges to parent (correspond to `parids`)
        parids, parpids = sum_layer_backward_compilation(
            self.nodes, pids, fw_n_group_ids, fw_n_id_in_group, self.num_bk_groups, bk_n_group_ids, bk_n_id_in_group,
            fw_group_max_chs, bk_group_max_pars, fw_num_ns_in_group, bk_num_ns_in_group, ch_prod_layer_size, global_nid_start,
            # GPU compilation is slightly slower for small layer due to the kernel jit compilation time
            use_cuda = not disable_gpu_compilation and (self.num_edges > 1000)
        )

        # Store buffers for the backward pass
        self.grouped_chids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in chids])
        self.grouped_parids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in parids])
        self.grouped_parpids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in parpids])

        # Store the range of global node indices belonging to this layer
        # This is used to implement partial evaluation
        self.global_nid_range = (global_nid_start, global_nid_start + self.num_nodes)

    def to(self, device):
        super(SumLayer, self).to(device)

        # Move cached fw pcids to the new device
        for k, v in self._cached_fw_pcids.items():
            new_v = [tensor.to(device) for tensor in v]
            self._cached_fw_compiled_pcids[k] = new_v

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
        
        if not self.provided("bk_group_local_ids"):
            # Evaluate the whole layer
            for group_id in range(self.num_bk_groups):
                chids = self.grouped_chids[group_id]
                parids = self.grouped_parids[group_id]
                parpids = self.grouped_parpids[group_id]

                self._backward(
                    node_flows, element_flows, params, node_mars, 
                    element_mars, param_flows, chids, parids, parpids
                )

        else:
            # Partial evaluation
            for group_id in range(self.num_bk_groups):
                chids = self.grouped_chids[group_id]
                parids = self.grouped_parids[group_id]
                parpids = self.grouped_parpids[group_id]
                local_ids = self.bk_group_local_ids[group_id]

                self._backward(
                    node_flows, element_flows, params, node_mars,
                    element_mars, param_flows, chids, parids, parpids,
                    local_ids = local_ids
                )

        return None
        
    @staticmethod
    @triton.jit
    def _forward_triton_kernel_old(node_mars_ptr, element_mars_ptr, params_ptr, 
                               nids_ptr, cids_ptr, pids_ptr, tot_n_nodes, 
                               tot_n_eles, n_nodes, n_edges: tl.constexpr, 
                               batch_size, n_nodes_per_block_m: tl.constexpr,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):

        # We use BLOCK_M to index over edges, and BLOCK_N to index over batches
        pid0 = tl.program_id(axis = 0)
        pid1 = tl.program_id(axis = 1)
        ne_start = pid0 * BLOCK_M
        b_start = pid1 * BLOCK_N

        # Id of edges processed by the current block
        ne_offsets = ne_start + tl.arange(0, BLOCK_M)
        # Batch ids processed by the current block
        b_offsets = b_start + tl.arange(0, BLOCK_N)
        b_mask = b_offsets < batch_size

        # Get node ids from `nids`
        n_start = ne_start // n_edges
        nid_offsets = n_start + tl.arange(0, n_nodes_per_block_m)
        nid_mask = nid_offsets < n_nodes
        n_ids = tl.load(nids_ptr + nid_offsets, mask = nid_mask, other = 0)

        # Get edge ids from `cids`
        cid_offsets = tl.view(ne_offsets, (n_edges, n_nodes_per_block_m))
        cid_mask = tl.broadcast_to(nid_mask[None,:], (n_edges, n_nodes_per_block_m))
        ch_ids = tl.load(cids_ptr + cid_offsets, mask = cid_mask, other = 0)

        # Use `ch_ids` to retrieve the corresponding element mars
        ele_offsets = tl.broadcast_to(ch_ids[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) * batch_size + \
            tl.broadcast_to(b_offsets[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
        ele_mask = tl.broadcast_to(nid_mask[None,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) & \
            tl.broadcast_to(b_mask[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
        ch_logps = tl.load(element_mars_ptr + ele_offsets, mask = ele_mask, other = 0) # `element_mars[cids]`

        # Take the max of the child mars
        ch_max_logp = tl.max(ch_logps, axis = 1) # `maxval`

        # Subtract the max from child mars
        ch_logps_sub_max = ch_logps - tl.broadcast_to(ch_max_logp[:,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m))

        # Take exp
        ch_ps_sub_max = tl.exp(ch_logps_sub_max)

        # Get param ids from `pids`
        # Here we reuse `cid_offsets` and `cid_mask` thank to their similar structure
        par_ids = tl.load(pids_ptr + cid_offsets, mask = cid_mask, other = 0)

        # Use `par_ids` to retrieve the corresponding parameters
        par_mask = tl.broadcast_to(nid_mask[None,:], (n_edges, n_nodes_per_block_m))
        ch_pars = tl.load(params_ptr + par_ids, mask = par_mask, other = 0) # `params[pids]`
        ch_pars = tl.broadcast_to(ch_pars[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m))

        # Sum node marginals (unnormalized)
        n_ps = tl.sum(ch_ps_sub_max * ch_pars, axis = 1)

        # Take log and subtract max vals
        n_logps = tl.log(tl.maximum(n_ps, 1e-10)) + ch_max_logp

        # Read out the target indices for `node_mars`
        nmar_offsets = tl.broadcast_to(n_ids[None,:], (BLOCK_N, n_nodes_per_block_m)) * batch_size + \
            tl.broadcast_to(b_offsets[:,None], (BLOCK_N, n_nodes_per_block_m))
        nmar_mask = tl.broadcast_to(nid_mask[None,:], (BLOCK_N, n_nodes_per_block_m)) & \
            tl.broadcast_to(b_mask[:,None], (BLOCK_N, n_nodes_per_block_m))
        
        # Reshape seems to be necessary for certain combinations of (BLOCK_N, n_nodes_per_block_m)
        nmar_offsets = tl.view(nmar_offsets, (BLOCK_N * n_nodes_per_block_m,))
        nmar_mask = tl.view(nmar_mask, (BLOCK_N * n_nodes_per_block_m,))
        n_logps = tl.view(n_logps, (BLOCK_N * n_nodes_per_block_m,))
        tl.store(node_mars_ptr + nmar_offsets, n_logps, mask = nmar_mask)

    @staticmethod
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _forward_pytorch_kernel(node_mars: torch.Tensor, element_mars: torch.Tensor, 
                                params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor, pids: torch.Tensor):

        ch_mars = element_mars[cids]
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        node_mars[nids] = (((ch_mars - maxval).exp() * params[pids].unsqueeze(-1)).sum(
            dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

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
        elif self.group_size * num_edges < 16 and num_edges * batch_size < 16:
            # In this case, we should definitely use the sparse implementation
            mode = "sparse"
        else:
            mode = "sparse"

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
        
        else:
            raise ValueError(f"Unexpected mode `{mode}`.")

    @staticmethod
    @triton.jit
    def _fw_triton_block_sparse_kernel(node_mars, element_mars, params, nids, cids_start, cids_increment,
                                       pids_start, pids_increment, local_ids, batch_size: tl.constexpr, partial_eval: tl.constexpr,
                                       BLOCK_B: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                       TILE_SIZE_M: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

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
            emars = tl.exp(emars - emars_max)
            epars = epars.to(tl.float16)
            emars = emars.to(tl.float16)
            nmars = tl.dot(epars, emars).to(tl.float32)

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

    def _forward_block_sparse(self, node_mars: torch.Tensor, element_mars: torch.Tensor,
                              params: torch.Tensor, nids: torch.Tensor, cids: torch.Tensor,
                              pids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                              partition_id: int = -1) -> None:
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

        signature = ("block_sparse", partition_id, TILE_SIZE_K)
        if signature not in self._cached_fw_pcids:
            # Pre-compute pointer increments for `cids` and `pids`

            cids = cids.clone().reshape(num_ngroups, K_NUM_TILES, TILE_SIZE_K)
            cids_start = cids[:,0,:].contiguous()
            cids_increment = torch.cat(
                (cids[:,1:,:] - cids[:,:-1,:], cids[:,0:1,:] * 0), 
                dim = 1
            ).contiguous()

            pids = pids.clone().reshape(num_ngroups, K_NUM_TILES, TILE_SIZE_K)
            pids_start = pids[:,0,:].contiguous()
            pids_increment = torch.cat(
                (pids[:,1:,:] - pids[:,:-1,:], pids[:,0:1,:] * 0),
                dim = 1
            ).contiguous()

            self._cached_fw_pcids[signature] = [cids_start, cids_increment, pids_start, pids_increment]
        else:
            cids_start, cids_increment, pids_start, pids_increment = self._cached_fw_pcids[signature]

        grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(layer_n_nodes, TILE_SIZE_M))
        
        self._fw_triton_block_sparse_kernel[grid](
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
            GROUP_SIZE_M = self.group_size
        )
        
        return None

    @staticmethod
    @triton.jit
    def _fw_triton_sparse_kernel(node_mars, element_mars, params, nids, cids, pids,
                                 local_ids, batch_size, partial_eval: tl.constexpr, n_edges: tl.constexpr, 
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
        offs_edge = tl.arange(0, n_edges)
        par_start = tl.load(pids + ngroup_id * n_edges + offs_edge)
        epars_ptr = params + tile_id * BLOCK_M + par_start # [n_edges]

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
        mask_batch = offs_batch < batch_size

        # Initialize and load edge mars
        edge_ids = tl.load(cids + ngroup_id * n_edges + offs_edge)
        emars_ptr = element_mars + \
            edge_ids[:,None] * batch_size + \
            offs_batch[None,:]
        emars = tl.load(emars_ptr, mask = mask_batch[None,:]) # [n_edges, BLOCK_B]

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
        n_edges = cids.size(1)
        batch_size = node_mars.size(1)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)

        assert n_edges <= 16384

        BLOCK_B = max(min(2048 // n_edges, BATCH_SIZE_NP2), 1)
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
            n_edges = n_edges, 
            BLOCK_B = BLOCK_B, 
            BLOCK_M = BLOCK_M, 
            GROUP_SIZE_M = self.group_size
        )

        return None

    @staticmethod
    @triton.jit
    def _backward_kernel(node_flows_ptr, element_flows_ptr, params_ptr, 
                         node_mars_ptr, element_mars_ptr, param_flows_ptr,
                         chids_ptr, parids_ptr, parpids_ptr, tot_n_nodes, 
                         tot_n_eles, n_nodes, n_edges: tl.constexpr, batch_size,
                         n_nodes_per_block_m: tl.constexpr,
                         accumulate_param_flows: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # We use BLOCK_M to index over edges, and BLOCK_N to index over batches
        pid0 = tl.program_id(axis = 0)
        pid1 = tl.program_id(axis = 1)
        ne_start = pid0 * BLOCK_M
        b_start = pid1 * BLOCK_N

        # Id of edges processed by the current block
        ne_offsets = ne_start + tl.arange(0, BLOCK_M)
        # Batch ids processed by the current block
        b_offsets = b_start + tl.arange(0, BLOCK_N)
        b_mask = b_offsets < batch_size

        # Node mask for future reuse
        n_start = ne_start // n_edges
        n_offsets = n_start + tl.arange(0, n_nodes_per_block_m)
        n_mask = n_offsets < n_nodes

        # Reusable ids for index tensors
        par_offsets = tl.view(ne_offsets, (n_edges, n_nodes_per_block_m))
        par_mask = tl.broadcast_to(n_mask[None,:], (n_edges, n_nodes_per_block_m)) 
        bpar_mask = tl.broadcast_to(n_mask[None,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) & \
            tl.broadcast_to(b_mask[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))

        # Get node ids from `parids` and retrieve the corresponding node flows and node mars
        node_ids = tl.load(parids_ptr + par_offsets, mask = par_mask, other = 0)
        node_offsets = tl.broadcast_to(node_ids[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) * batch_size + \
            tl.broadcast_to(b_offsets[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
        nflows = tl.load(node_flows_ptr + node_offsets, mask = bpar_mask, other = 0) # node_flows[parids]
        nmars = tl.load(node_mars_ptr + node_offsets, mask = bpar_mask, other = 0) # node_mars[parids]

        # Get param ids from `parpids` and retrieve the corresponding node params
        eparam_ids = tl.load(parpids_ptr + par_offsets, mask = par_mask, other = 0)
        eparams = tl.load(params_ptr + eparam_ids, mask = par_mask, other = 0)
        eparams = tl.broadcast_to(eparams[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) # params[parpids]

        # Compute edge flows (partially)
        cum_flow = nflows * eparams

        # Get element ids from `cids` and retrieve the corresponding element mars
        ele_ids = tl.load(chids_ptr + n_offsets, mask = n_mask, other = 0)
        ele_offsets = tl.broadcast_to(ele_ids[None,:], (BLOCK_N, n_nodes_per_block_m)) * batch_size + \
            tl.broadcast_to(b_offsets[:,None], (BLOCK_N, n_nodes_per_block_m))
        ele_mask = tl.broadcast_to(n_mask[None,:], (BLOCK_N, n_nodes_per_block_m)) & \
            tl.broadcast_to(b_mask[:,None], (BLOCK_N, n_nodes_per_block_m))
        emars = tl.load(element_mars_ptr + ele_offsets, mask = ele_mask, other = 0) # element_mars[chids]
        emars = tl.broadcast_to(emars[:,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) # element_mars[chids].unsqueeze(1)

        # Compute edge flows
        emars_log_diff = emars - nmars
        emars_diff = tl.exp(emars_log_diff)
        eflows = cum_flow * emars_diff

        # Store to `element_flows[chids]`
        cum_eflows = tl.sum(eflows, axis = 1) # [BLOCK_N, n_nodes_per_block_m]
        tl.store(element_flows_ptr + ele_offsets, cum_eflows, mask = ele_mask)

        # Compute parameter flows
        if accumulate_param_flows:
            parflows = tl.sum(eflows, axis = 0) # [n_edges, n_nodes_per_block_m]
            # Here the `eparam_ids > 0` term masks out dummy edges
            parflow_mask = (eparam_ids > 0) & tl.broadcast_to(n_mask[None,:], (n_edges, n_nodes_per_block_m))
            tl.atomic_add(param_flows_ptr + eparam_ids, parflows, mask = parflow_mask)

    @staticmethod
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _backward_pytorch_kernel(node_flows: torch.Tensor, element_flows: torch.Tensor, 
                                 params: torch.Tensor, node_mars: torch.Tensor, 
                                 element_mars: torch.Tensor, param_flows: Optional[torch.Tensor], 
                                 chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor):
        
        element_flows[chids] = (node_flows[parids] * params[parpids].unsqueeze(-1) * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

        return None

    def _backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                  params: torch.Tensor, node_mars: torch.Tensor, 
                  element_mars: torch.Tensor, param_flows: torch.Tensor, 
                  chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor, 
                  local_ids: Optional[torch.Tensor] = None, 
                  BLOCK_M_HARD_LIMIT = 2**16, BLOCK_SIZE = 2**12, MAX_BLOCK_M = 2**11, 
                  MAX_BLOCK_N = 64) -> None:
        """
        This function is equivalent to running:
        ``` 
        element_flows[chids] = (node_flows[parids] * params[parpids] * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

        param_flows[seq_parpids] += (node_flows[parids] * params[parpids] * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 2)[seq_ids0, seq_ids1]
        ```
        
        Parameters:
        `node_flows`:    [N, B]
        `element_flows`: [M, B]
        `params`:        [E]
        `node_mars`:     [N, B]
        `element_mars`:  [M, B]
        `param_flows`:   [E]
        `chids`:         [n]
        `parids`:        [n, p]
        `parpids`:       [n, p]
        """

        if local_ids is not None and local_ids.size(0) == 0:
            # Nothing need to be evaluated in the current group
            return None
        elif local_ids is not None:
            # Select nodes
            chids = chids[local_ids].contiguous()
            parids = parids[local_ids,:].contiguous()
            parpids = parpids[local_ids,:].contiguous()

        tot_n_nodes = node_mars.size(0)
        tot_n_eles = element_mars.size(0)
        n_nodes = chids.size(0)
        n_edges = parids.size(1)
        batch_size = node_mars.size(1)

        if params.dim() == 2 and params.size(1) == 1:
            params = params.squeeze(1)

        # If child nodes in the current group have no parent, we set the corresponding element flows to 0
        if n_edges == 0:
            element_flows[chids] = 0.0

            return None

        # Fall back to the `torch.compile` kernel in the case where we cannot store child edges within a single block
        if n_edges > BLOCK_M_HARD_LIMIT or not node_mars.is_cuda:
            assert param_flows is None
            self._backward_pytorch_kernel(
                node_flows, element_flows, params, node_mars, 
                element_mars, param_flows, chids, parids, parpids
            )

            return None

        assert n_edges <= BLOCK_M_HARD_LIMIT, f"Number of edges should be smaller than or equal to {BLOCK_M_HARD_LIMIT}."
        assert params.dim() == 1, "Expecting a 1D `params`."

        if n_edges <= MAX_BLOCK_M:
            # In this case, we can find a better thread-block balance
            MIN_BLOCK_M = min(triton.next_power_of_2(n_edges), MAX_BLOCK_M)
            BLOCK_N = min(BLOCK_SIZE // MIN_BLOCK_M, MAX_BLOCK_N, triton.next_power_of_2(batch_size))
            BLOCK_M = min(BLOCK_SIZE // BLOCK_N, MAX_BLOCK_M)
        else:
            # Try to fit all edges of a node in a single thread-block
            BLOCK_M = triton.next_power_of_2(n_edges)
            BLOCK_N = max(BLOCK_SIZE // BLOCK_M, 1)

        grid = (triton.cdiv(n_nodes * n_edges, BLOCK_M), triton.cdiv(batch_size, BLOCK_N), 1)

        self._backward_kernel[grid](
            node_flows_ptr = node_flows,
            element_flows_ptr = element_flows,
            params_ptr = params,
            node_mars_ptr = node_mars, 
            element_mars_ptr = element_mars,
            param_flows_ptr = param_flows, 
            chids_ptr = chids, 
            parids_ptr = parids, 
            parpids_ptr = parpids,
            tot_n_nodes = tot_n_nodes,
            tot_n_eles = tot_n_eles,
            n_nodes = n_nodes,
            n_edges = n_edges,
            batch_size = batch_size,
            n_nodes_per_block_m = BLOCK_M // n_edges,
            accumulate_param_flows = (param_flows is not None),
            BLOCK_M = BLOCK_M, 
            BLOCK_N = BLOCK_N
        )

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
