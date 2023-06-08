from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
from typing import Sequence, Optional

from pyjuice.nodes import ProdNodes
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges
from .backend.index_set import batched_index_set, batched_index_cum


class ProdLayer(Layer, nn.Module):

    def __init__(self, nodes: Sequence[ProdNodes], layer_sparsity_tol: float = 0.0, max_num_groups: Optional[int] = None) -> None:
        Layer.__init__(self)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."

        self.nodes = nodes

        max_n_chs = 0
        layer_num_nodes = 0
        cum_nodes = 1 # id 0 is reserved for the dummy node
        for ns in self.nodes:
            if ns.num_child_regions > max_n_chs:
                max_n_chs = ns.num_child_regions
            ns._output_ind_range = (cum_nodes, cum_nodes + ns.num_nodes)
            cum_nodes += ns.num_nodes
            layer_num_nodes += ns.num_nodes

        self.num_nodes = layer_num_nodes

        ## Initialize forward pass ##

        n_edges = triton.next_power_of_2(max_n_chs)
        cids = torch.zeros([self.num_nodes, n_edges], dtype = torch.long) # cids[i,:] are the child node ids for node i
        n_chs = torch.zeros([self.num_nodes], dtype = torch.long) # Number of children for each node
        node_start = 0
        for ns in self.nodes:
            node_end = node_start + ns.num_nodes
            for i, c in enumerate(ns.chs):
                cids[node_start:node_end, i] = ns.edge_ids[:,i] + c._output_ind_range[0]
                
            n_chs[node_start:node_end] = ns.num_child_regions

            node_start = node_end

        # Find a good strategy to partition the nodes into groups according to their number of children 
        # to minimize total computation cost
        ch_group_sizes = partition_nodes_by_n_edges(n_chs, max_num_groups = max_num_groups)

        grouped_nids = []
        grouped_cids = []
        min_n_chs = 0
        for max_n_chs in ch_group_sizes:
            filter = (n_chs >= min_n_chs) & (n_chs <= max_n_chs)
            n_edges = triton.next_power_of_2(max_n_chs)

            curr_nids = (torch.where(filter)[0] + 1).clone()
            curr_cids = cids[filter,:n_edges].contiguous()

            grouped_nids.append(curr_nids)
            grouped_cids.append(curr_cids)

            min_n_chs = max_n_chs + 1

        self.num_forward_groups = ch_group_sizes.shape[0]
        self.grouped_nids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_nids])
        self.grouped_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_cids])

        ## Initialize backward pass ##

        u_cids, par_counts = torch.unique(cids, sorted = True, return_counts = True)

        if u_cids[0] != 0:
            u_cids = torch.cat(
                (torch.zeros([1], dtype = torch.long), u_cids),
                dim = 0
            )
            par_counts = torch.cat(
                (torch.zeros([1], dtype = torch.long), par_counts),
                dim = 0
            )

        max_n_pars = torch.max(par_counts[1:])
        n_edges = triton.next_power_of_2(max_n_pars)
        parids = torch.zeros([u_cids.size(0), n_edges], dtype = torch.long)
        for idx in range(u_cids.size(0)):
            cid = u_cids[idx]
            if cid == 0:
                continue # Skip the dummy node
            b_nid = torch.where(cids == cid)[0]
            parids[idx,:b_nid.size(0)] = b_nid + 1 # 1 for the dummy node

        # Find a good strategy to partition the child nodes into groups according to their number of parents 
        # to minimize total computation cost
        par_counts = par_counts[1:] # Strip away the dummy node. We will never use it in the following
        par_group_sizes = partition_nodes_by_n_edges(par_counts, max_num_groups = max_num_groups)

        grouped_parids = []
        grouped_u_cids = []
        min_n_pars = 0
        for max_n_pars in par_group_sizes:
            filter = (par_counts >= min_n_pars) & (par_counts <= max_n_pars)
            filtered_idxs = torch.where(filter)[0]
            n_edges = triton.next_power_of_2(max_n_pars)

            curr_parids = parids[filtered_idxs+1,:n_edges].contiguous()
            curr_u_cids = u_cids[filtered_idxs+1].contiguous()

            grouped_parids.append(curr_parids)
            grouped_u_cids.append(curr_u_cids)

            min_n_pars = max_n_pars + 1

        self.num_backward_groups = par_group_sizes.shape[0]
        self.grouped_parids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_parids])
        self.grouped_u_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_u_cids])

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor) -> None:
        """
        Computes the forward pass of a product layer:
        ```
        element_mars[nids] = node_mars[cids].sum(dim = 1)
        ```

        Parameters:
        `node_mars`:    [num_nodes, B]
        `element_mars`: [max_num_els, B]
        """

        for group_id in range(self.num_forward_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]

            self._forward_backward_triton(element_mars, node_mars, nids, cids)

        return None

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor) -> None:
        """
        Computes the backward pass of a product layer:
        ```
        node_flows[u_cids] = element_flows[parids].sum(dim = 1)
        ```

        Parameters:
        `node_flows`:    [num_nodes, B]
        `element_flows`: [max_num_els, B]
        """
        
        for group_id in range(self.num_backward_groups):
            u_cids = self.grouped_u_cids[group_id]
            parids = self.grouped_parids[group_id]

            self._forward_backward_triton(node_flows, element_flows, u_cids, parids)
        
        return None

    @staticmethod
    @triton.jit
    def _forward_backward_kernel(node_vals_ptr, element_vals_ptr, nids_ptr, cids_ptr, 
                                 tot_n_nodes: tl.constexpr, tot_n_eles: tl.constexpr,
                                 n_nodes: tl.constexpr, n_edges: tl.constexpr, 
                                 batch_size: tl.constexpr, n_nodes_per_block_m: tl.constexpr,
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
        cid_offsets = tl.reshape(ne_offsets, (n_edges, n_nodes_per_block_m))
        cid_mask = tl.broadcast_to(nid_mask[None,:], (n_edges, n_nodes_per_block_m))
        ch_ids = tl.load(cids_ptr + cid_offsets, mask = cid_mask, other = 0)

        # Use `ch_ids` to retrieve the corresponding element mars
        ele_offsets = tl.broadcast_to(ch_ids[None,:,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) * batch_size + \
            tl.broadcast_to(b_offsets[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
        ele_mask = tl.broadcast_to(nid_mask[None,None,:], (BLOCK_N, n_edges, n_nodes_per_block_m)) & \
            tl.broadcast_to(b_mask[:,None,None], (BLOCK_N, n_edges, n_nodes_per_block_m))
        ch_logps = tl.load(element_vals_ptr + ele_offsets, mask = ele_mask, other = 0)

        # Take the sum of the child mars
        n_logps = tl.sum(ch_logps, axis = 1)

        # Read out the target indices for `node_vals`
        nmar_offsets = tl.broadcast_to(n_ids[None,:], (BLOCK_N, n_nodes_per_block_m)) * batch_size + \
            tl.broadcast_to(b_offsets[:,None], (BLOCK_N, n_nodes_per_block_m))
        nmar_mask = tl.broadcast_to(nid_mask[None,:], (BLOCK_N, n_nodes_per_block_m)) & \
            tl.broadcast_to(b_mask[:,None], (BLOCK_N, n_nodes_per_block_m))
        
        tl.store(node_vals_ptr + nmar_offsets, n_logps, mask = nmar_mask)

    def _forward_backward_triton(self, node_vals: torch.Tensor, element_vals: torch.Tensor, 
                                 nids: torch.Tensor, cids: torch.Tensor, 
                                 BLOCK_SIZE = 2**12, MAX_BLOCK_M = 512, MAX_BLOCK_N = 64) -> None:
        """
        This function is equivalent to running:
        ``` node_vals[nids] = element_vals[cids].sum(dim = 1) ```
        
        Parameters:
        `node_vals`:    [N, B]
        `element_vals`: [M, B]
        `nids`:         [n]
        `cids`:         [n, c]
        """
        tot_n_nodes = node_vals.size(0)
        tot_n_eles = element_vals.size(0)
        n_nodes = nids.size(0)
        n_edges = cids.size(1)
        batch_size = node_vals.size(1)

        assert n_edges <= MAX_BLOCK_M, "Number of edges should be smaller than or equal to MAX_BLOCK_M."
        assert n_edges & (n_edges - 1) == 0, "`n_edges` must be power of 2."

        MIN_BLOCK_M = min(triton.next_power_of_2(n_edges), MAX_BLOCK_M)
        BLOCK_N = min(BLOCK_SIZE // MIN_BLOCK_M, MAX_BLOCK_N, triton.next_power_of_2(batch_size))
        BLOCK_M = min(BLOCK_SIZE // BLOCK_N, MAX_BLOCK_M)

        grid = (triton.cdiv(n_nodes * n_edges, BLOCK_M), triton.cdiv(batch_size, BLOCK_N), 1)

        self._forward_backward_kernel[grid](
            node_vals_ptr = node_vals, 
            element_vals_ptr = element_vals, 
            nids_ptr = nids, 
            cids_ptr = cids, 
            tot_n_nodes = tot_n_nodes,
            tot_n_eles = tot_n_eles,
            n_nodes = n_nodes, 
            n_edges = n_edges, 
            batch_size = batch_size, 
            n_nodes_per_block_m = BLOCK_M // n_edges,
            BLOCK_M = BLOCK_M, 
            BLOCK_N = BLOCK_N
        )

        return None
