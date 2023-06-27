from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
import time
from typing import Sequence, Optional

from pyjuice.nodes import ProdNodes
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges
from .backend.index_set import batched_index_set, batched_index_cum
from .compilation import next_power_of_2, get_prod_layer_stats, prod_layer_forward_compilation, \
                         flatten_c_ids, get_prod_layer_parstats, prod_layer_backward_compilation


class ProdLayer(Layer, nn.Module):

    def __init__(self, nodes: Sequence[ProdNodes], layer_sparsity_tol: float = 0.0, 
                 max_num_groups: Optional[int] = None, disable_gpu_compilation: bool = False) -> None:
        Layer.__init__(self)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."

        self.nodes = nodes

        ## Get layer statistics & prepare for compilation ##

        layer_num_nodes, layer_num_edges, n_chs = get_prod_layer_stats(self.nodes)

        self.num_nodes = layer_num_nodes
        self.num_edges = layer_num_edges

        # Find a good strategy to partition the nodes into groups according to their number of children 
        # to minimize total computation cost
        fw_group_max_chs = partition_nodes_by_n_edges(
            n_chs, sparsity_tolerance = layer_sparsity_tol, max_num_groups = max_num_groups
        )

        # Since the triton kernels require the maximum number children for each group to be a power of 2,
        # we postprocess the group sizes
        fw_group_max_chs = torch.unique(next_power_of_2(fw_group_max_chs))

        self.num_fw_groups = len(fw_group_max_chs) # Number of groups

        # fw_n_group_ids:     [num_ns]           stores the group id for each `ns` in `nodes`
        # fw_n_id_in_group:   [num_ns]           stores the start index of each `ns` in the group
        # fw_num_ns_in_group: [num_fw_groups]    number of nodes in each group
        num_ns = len(self.nodes)
        fw_n_group_ids = torch.zeros([num_ns], dtype = torch.long)
        fw_n_id_in_group = torch.zeros([num_ns], dtype = torch.long)
        fw_num_ns_in_group = torch.zeros([self.num_fw_groups], dtype = torch.long)

        for ns_id, ns in enumerate(self.nodes):
            group_id = (ns.num_chs > fw_group_max_chs).sum().item()

            fw_n_group_ids[ns_id] = group_id
            fw_n_id_in_group[ns_id] = fw_num_ns_in_group[group_id]
            fw_num_ns_in_group[group_id] += ns.num_nodes

        ## Initialize forward pass ##

        # nids:      List[[group_size]]                  stores node ids
        # cids:      List[[group_size, group_max_n_chs]] stores indices of child nodes
        nids, cids = prod_layer_forward_compilation(
            self.nodes, fw_group_max_chs, fw_n_group_ids, fw_n_id_in_group, fw_num_ns_in_group
        )

        # Store buffers for the forward pass
        self.grouped_nids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in nids])
        self.grouped_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in cids])

        ## Initialize backward pass ##

        # flat_cids:   flattened version of `cids`
        # flat_cid2id: mapping from every `flat_cids` to its corresponding `nids`
        flat_cids, flat_cid2nid = flatten_c_ids(nids, cids)

        # flat_u_cids:        [num_used_ch_nodes]    child node ids that have at least one parent
        # par_counts:         [num_used_ch_nodes]    the number of parents for each child node
        # Note: the dummy node has been removed from `flat_u_cids` and `par_counts`
        flat_u_cids, par_counts = get_prod_layer_parstats(flat_cids)

        # Find a good strategy to partition the child nodes into groups according to their number of parents 
        # to minimize total computation cost
        bk_group_max_pars = partition_nodes_by_n_edges(
            par_counts, sparsity_tolerance = layer_sparsity_tol, max_num_groups = max_num_groups
        )

        # Since the triton kernels require the maximum number children for each group to be a power of 2,
        # we postprocess the group sizes
        bk_group_max_pars = torch.unique(next_power_of_2(bk_group_max_pars))

        self.num_bk_groups = len(bk_group_max_pars) # Number of groups

        # bk_n_group_ids:     [num_ch_nodes]       stores the group id for each `ns` in `nodes`
        # bk_n_id_in_group:   [num_ch_nodes]       stores the start index of each `ns` in the group
        # bk_num_ns_in_group: [num_bk_groups]      number of nodes in each group
        num_ch_nodes = flat_u_cids.size(0)
        bk_n_group_ids = torch.zeros([num_ch_nodes], dtype = torch.long)
        bk_n_id_in_group = torch.zeros([num_ch_nodes], dtype = torch.long)
        bk_num_ns_in_group = torch.zeros([self.num_bk_groups], dtype = torch.long)

        min_n_pars = 0
        for group_id, max_n_pars in enumerate(bk_group_max_pars):
            criterion = (par_counts >= min_n_pars) & (par_counts <= max_n_pars)
            filtered_idxs = torch.where(criterion)[0]
            group_size = criterion.sum().item()

            bk_n_group_ids[criterion] = group_id
            bk_n_id_in_group[criterion] = torch.arange(group_size)
            bk_num_ns_in_group[group_id] = group_size

            min_n_pars = max_n_pars + 1

        # u_cids:    List[[group_size]]                   stores child node ids
        # parids:    List[[group_size, group_max_n_pars]] stores indices of parent nodes
        u_cids, parids = prod_layer_backward_compilation(
            flat_u_cids, flat_cids, flat_cid2nid, 
            bk_group_max_pars, bk_n_group_ids, bk_n_id_in_group, bk_num_ns_in_group
        )

        # Store buffers for the backward pass
        self.grouped_u_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in u_cids])
        self.grouped_parids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in parids])

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

        for group_id in range(self.num_fw_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]

            self._forward_backward(element_mars, node_mars, nids, cids)

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
        
        for group_id in range(self.num_bk_groups):
            u_cids = self.grouped_u_cids[group_id]
            parids = self.grouped_parids[group_id]

            self._forward_backward(node_flows, element_flows, u_cids, parids)
        
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

    @staticmethod
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _forward_backward_pytorch(node_vals, element_vals, nids, cids):
        node_vals[nids] = element_vals[cids].sum(dim = 1)

        return None

    def _forward_backward(self, node_vals: torch.Tensor, element_vals: torch.Tensor, 
                          nids: torch.Tensor, cids: torch.Tensor, 
                          BLOCK_M_HARD_LIMIT = 2**16,
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

        # Fall back to the `torch.compile` kernel in the case where we cannot store child edges within a single block
        if n_edges > BLOCK_M_HARD_LIMIT or not node_vals.is_cuda:
            self._forward_backward_pytorch(node_vals, element_vals, nids, cids)

            return None

        assert n_edges <= BLOCK_M_HARD_LIMIT, "Number of edges should be smaller than or equal to MAX_BLOCK_M."
        assert n_edges & (n_edges - 1) == 0, "`n_edges` must be power of 2."

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
