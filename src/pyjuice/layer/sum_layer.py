
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
from copy import deepcopy
from typing import Sequence, List, Optional

from pyjuice.nodes import SumNodes
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges
from .backend.index_set import batched_index_set, index_cum
from .compilation import get_sum_layer_stats, sum_layer_forward_compilation, \
                         sum_layer_backward_compilation, next_power_of_2


class SumLayer(Layer, nn.Module):

    def __init__(self, nodes: Sequence[SumNodes], global_nid_start: int, 
                 param_ends: Sequence, tied_param_ids: Sequence,
                 tied_param_group_ids: Sequence, tied_param_ends: Sequence,
                 ch_prod_layer_size: int, layer_sparsity_tol: float = 0.0, 
                 max_num_groups: Optional[int] = None,
                 disable_gpu_compilation: bool = False) -> None:

        Layer.__init__(self)
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
            n_chs, sparsity_tolerance = layer_sparsity_tol, max_num_groups = max_num_groups
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
            use_cuda = not disable_gpu_compilation and (self.num_edges > 250000) # Consider tuning this
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
            ch_n_pars, sparsity_tolerance = layer_sparsity_tol, max_num_groups = max_num_groups
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

        # parids:     List[[group_size, group_max_n_pars]]  stores parameter ids for each child node
        # parpids:    List[[group_size, max_n_pars]]        param id for the edges to parent (correspond to `parids`)
        parids, parpids = sum_layer_backward_compilation(
            self.nodes, pids, fw_n_group_ids, fw_n_id_in_group, self.num_bk_groups, bk_n_group_ids, bk_n_id_in_group,
            bk_group_max_pars, bk_num_ns_in_group, ch_prod_layer_size, global_nid_start,
            use_cuda = not disable_gpu_compilation and (self.num_edges > 250000), # Consider tuning this
            debug = (bk_group_max_pars[0] == 1).item()
        )

        # Store buffers for the backward pass
        self.grouped_chids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in chids])
        self.grouped_parids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in parids])
        self.grouped_parpids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in parpids])

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

        for group_id in range(self.num_fw_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]
            pids = self.grouped_pids[group_id]

            self._forward(
                node_mars, element_mars, params, nids, cids, pids
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
        
        for group_id in range(self.num_bk_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parpids = self.grouped_parpids[group_id]

            self._backward(
                node_flows, element_flows, params, node_mars, 
                element_mars, param_flows, chids, parids, parpids
            )

        return None

    def sample(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
               node_mars: torch.Tensor, element_mars: torch.Tensor, 
               params: torch.Tensor, node_mask: torch.Tensor) -> None:
        """
        Compute sampling flow.

        Parameters:
        `node_flows`:    [num_nodes, B]
        `element_flows`: [max_num_els, B]
        `node_mars`:     [num_nodes, B]
        `element_mars`:  [max_num_els, B]
        `params`:        [num_params] or [num_params, B]
        `node_mask`:     [num_nodes, B]
        """
        if params.dim() == 1:
            params = params.unsqueeze(1)

        self._sample_mask_generation(node_mars, element_mars, params, node_mask)
        self._sample_backward_pass(node_flows, element_flows, node_mars, element_mars, params, node_mask)

        return None
        
    @staticmethod
    @triton.jit
    def _forward_triton_kernel(node_mars_ptr, element_mars_ptr, params_ptr, 
                               nids_ptr, cids_ptr, pids_ptr,
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
                 pids: torch.Tensor, BLOCK_M_HARD_LIMIT = 2**16, BLOCK_SIZE = 2**12, 
                 MAX_BLOCK_M = 2**12, MAX_BLOCK_N = 64) -> None:
        """
        This function is equivalent to running:
        ``` 
        ch_mars = element_mars[cids]
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        node_mars[nids] = (((ch_mars - maxval).exp() * params[pids]).sum(
            dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)
        ```
        
        Parameters:
        `node_mars`:    [N, B]
        `element_mars`: [M, B]
        `params`:       [E]
        `nids`:         [n]
        `cids`:         [n, c]
        `pids`:         [n, c]
        """
        tot_n_nodes = node_mars.size(0)
        tot_n_eles = element_mars.size(0)
        n_nodes = nids.size(0)
        n_edges = cids.size(1)
        batch_size = node_mars.size(1)

        if params.dim() == 2 and params.size(1) == 1:
            params = params.squeeze(1)

        # Fall back to the `torch.compile` kernel in the case where we cannot store child edges within a single block
        if n_edges > BLOCK_M_HARD_LIMIT or not node_mars.is_cuda:
            self._forward_pytorch_kernel(node_mars, element_mars, params, nids, cids, pids)

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

        self._forward_triton_kernel[grid](
            node_mars_ptr = node_mars, 
            element_mars_ptr = element_mars, 
            params_ptr = params,
            nids_ptr = nids, 
            cids_ptr = cids, 
            pids_ptr = pids,
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

    @staticmethod
    @triton.jit
    def _backward_kernel(node_flows_ptr, element_flows_ptr, params_ptr, 
                         node_mars_ptr, element_mars_ptr, param_flows_ptr,
                         chids_ptr, parids_ptr, parpids_ptr,
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

        # Node mask for future reuse
        n_start = ne_start // n_edges
        n_offsets = n_start + tl.arange(0, n_nodes_per_block_m)
        n_mask = n_offsets < n_nodes

        # Reusable ids for index tensors
        par_offsets = tl.reshape(ne_offsets, (n_edges, n_nodes_per_block_m))
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
        parflows = tl.sum(eflows, axis = 0) # [n_edges, n_nodes_per_block_m]
        # Here the `eparam_ids > 0` term masks out dummy edges
        parflow_mask = (eparam_ids > 0) & tl.broadcast_to(n_mask[None,:], (n_edges, n_nodes_per_block_m))
        tl.atomic_add(param_flows_ptr + eparam_ids, parflows, mask = parflow_mask)

    @staticmethod
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _backward_pytorch_kernel(node_flows: torch.Tensor, element_flows: torch.Tensor, 
                                 params: torch.Tensor, node_mars: torch.Tensor, 
                                 element_mars: torch.Tensor, param_flows: torch.Tensor, 
                                 chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor):
        
        element_flows[chids] = (node_flows[parids] * params[parpids].unsqueeze(-1) * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

        param_flows[seq_parpids] += (node_flows[parids] * params[parpids].unsqueeze(-1) * \
            (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 2)[seq_ids0, seq_ids1]

        return None

    def _backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                  params: torch.Tensor, node_mars: torch.Tensor, 
                  element_mars: torch.Tensor, param_flows: torch.Tensor, 
                  chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor, 
                  BLOCK_M_HARD_LIMIT = 2**16, BLOCK_SIZE = 2**12, MAX_BLOCK_M = 2**12, 
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
            raise NotImplementedError("This fallback workaround needs to be fixed..")
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
            BLOCK_M = BLOCK_M, 
            BLOCK_N = BLOCK_N
        )

        return None

    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _sample_mask_generation(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                                node_mask: torch.Tensor):
        for group_id in range(self.num_fw_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]
            pids = self.grouped_pids[group_id]

            N, C = cids.size()
            B = node_mars.size(1)

            ch_mars = element_mars[cids]
            maxval = ch_mars.max(dim = 1, keepdim = True).values
            unnorm_probs = (ch_mars - maxval).exp() * params[pids]
            dist = torch.distributions.Categorical(probs = unnorm_probs.permute(0, 2, 1))
            node_mask[nids] = cids.unsqueeze(2).expand(N, C, B).gather(1, dist.sample().unsqueeze(1)).squeeze(1) # [num nodes, batch_size]

        return None

    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _sample_backward_pass(self, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                              element_mars: torch.Tensor, params: torch.Tensor, node_mask: torch.Tensor):
        for group_id in range(self.num_backward_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parcids = self.grouped_parcids[group_id]

            element_flows[chids] = (node_flows[parids] * (node_mask[parids] == chids.unsqueeze(-1))).any(dim = 1)

        return None