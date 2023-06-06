
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


class SumLayer(Layer, nn.Module):

    def __init__(self, nodes: Sequence[SumNodes], cum_nodes: int, 
                 param_ends: Sequence, tied_param_ids: Sequence,
                 tied_param_group_ids: Sequence, tied_param_ends: Sequence,
                 ch_prod_layer_size: int, sum_region_start_id: int = 0,
                 max_num_groups: int = 1) -> None:

        Layer.__init__(self)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."

        self.nodes = nodes

        layer_num_nodes = 0
        total_num_edges = 0
        max_n_chs = 0
        for ns_idx, ns in enumerate(self.nodes):
            n_chs = torch.max(torch.bincount(ns.edge_ids[0,:])).item()
            if n_chs > max_n_chs:
                max_n_chs = n_chs
            ns._output_ind_range = (cum_nodes, cum_nodes + ns.num_nodes)
            cum_nodes += ns.num_nodes
            layer_num_nodes += ns.num_nodes
            total_num_edges += ns.edge_ids.size(1)

        self.num_nodes = layer_num_nodes
        self.num_params = total_num_edges

        self.ch_prod_layer_size = ch_prod_layer_size

        ## Initialize forward pass ##

        n_start_id = cum_nodes - self.num_nodes
        nids = torch.arange(cum_nodes - self.num_nodes, cum_nodes) # Node id
        cids = torch.zeros([self.num_nodes, max_n_chs], dtype = torch.long) # Child id
        pids = torch.zeros([self.num_nodes, max_n_chs], dtype = torch.long) # Parameter id
        n_chs = torch.zeros([self.num_nodes], dtype = torch.long) # Number of children
        ch_n_pars = torch.zeros([self.ch_prod_layer_size], dtype = torch.long) # Number of parents for each child node
        node_start = 0
        param_start = param_ends[-1]
        for ns_idx, ns in enumerate(self.nodes):

            param_ids = torch.zeros([ns.edge_ids.size(1)], dtype = torch.long)
            ns_param_start = param_start

            node_end = node_start + ns.num_nodes

            for nid in range(ns.num_nodes):
                ch_start = 0
                local_cumchs = 0
                for c in ns.chs:
                    criterion = (ns.edge_ids[1,:] >= local_cumchs) & (ns.edge_ids[1,:] < local_cumchs + c.num_nodes) & \
                                (ns.edge_ids[0,:] == nid)
                    local_cumchs += c.num_nodes

                    ch_ids = ns.edge_ids[1,criterion] + c._output_ind_range[0]
                    cids[node_start+nid,ch_start:ch_start+ch_ids.size(0)] = ch_ids
                    ch_start += ch_ids.size(0)

                    ch_n_pars[ch_ids] += 1

                    ids = torch.where(criterion)
                    param_ids[ids] = torch.arange(param_start + ch_start - ch_ids.size(0), param_start + ch_start)

                pids[node_start+nid,:ch_start] = torch.arange(param_start, param_start + ch_start, dtype = torch.long)
                param_start += ch_start
                param_ends.append(param_start)

                n_chs[node_start+nid] = ch_start

            node_start = node_end

            ns_param_end = param_start
            ns._param_range = (ns_param_start, ns_param_end)
            ns._param_ids = param_ids
            ns._inverse_param_ids = torch.argsort(param_ids)

            # For tied nodes
            if ns.is_tied():
                source_ns = ns.get_source_ns()
                if source_ns._tied_param_group_ids is None:
                    num_tied_param_groups = tied_param_ends[-1] if len(tied_param_ends) > 0 else 0
                    ns_param_ends = filter(lambda x: (x > ns_param_start) and (x <= ns_param_end), param_ends)
                    ns_param_ends = map(lambda x: x - ns_param_start + num_tied_param_groups, ns_param_ends)
                    tied_param_ends.extend(ns_param_ends)

                    num_source_params = source_ns._param_range[1] - source_ns._param_range[0] + 1
                    source_ns._tied_param_group_ids = [i for i in range(num_tied_param_groups, num_tied_param_groups + num_source_params - 1)]

                    tied_param_ids.extend([i for i in range(source_ns._param_range[0], source_ns._param_range[1])])
                    tied_param_group_ids.extend(source_ns._tied_param_group_ids)

                ns._tied_param_group_ids = deepcopy(source_ns._tied_param_group_ids)
                tied_param_ids.extend([i for i in range(ns._param_range[0], ns._param_range[1])])
                tied_param_group_ids.extend(ns._tied_param_group_ids)

        # Find a good strategy to partition the nodes into groups according to their number of children 
        # to minimize total computation cost
        ch_group_sizes = partition_nodes_by_n_edges(n_chs, max_num_groups = max_num_groups)

        grouped_nids = []
        grouped_cids = []
        grouped_pids = []
        min_n_chs = 0
        for max_n_chs in ch_group_sizes:
            id_filter = (n_chs >= min_n_chs) & (n_chs <= max_n_chs)
            curr_nids = nids[id_filter].contiguous()
            curr_cids = cids[id_filter,:max_n_chs].contiguous()
            curr_pids = pids[id_filter,:max_n_chs].contiguous()

            grouped_nids.append(curr_nids)
            grouped_cids.append(curr_cids)
            grouped_pids.append(curr_pids)

            min_n_chs = max_n_chs + 1

        self.num_forward_groups = ch_group_sizes.shape[0]
        self.grouped_nids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_nids])
        self.grouped_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_cids])
        self.grouped_pids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_pids])

        ## Initialize backward pass ##

        ch_n_pars[0] = 0 # We do not need to compute flows for the dummy node

        max_n_pars = torch.max(ch_n_pars)
        tot_n_pars = torch.sum(ch_n_pars[1:]) # Exclude the dummy node

        parids = torch.zeros([self.ch_prod_layer_size, max_n_pars], dtype = torch.long) # Indices of parent nodes for each child node
        parpids = torch.zeros([self.ch_prod_layer_size, max_n_pars], dtype = torch.long) # Parameter indices for these edges
        # For each edge, this matrix stores the index of the edge for the parent
        parcids = torch.zeros([self.ch_prod_layer_size, max_n_pars], dtype = torch.long) 
        par_counts = torch.zeros([self.ch_prod_layer_size], dtype = torch.long)
        node_start = 0
        for ns in self.nodes:
            node_end = node_start + ns.num_nodes
            for nid in range(ns.num_nodes):
                ch_start = 0
                local_cumchs = 0
                for cnode_id, c in enumerate(ns.chs):
                    criterion = (ns.edge_ids[1,:] >= local_cumchs) & (ns.edge_ids[1,:] < local_cumchs + c.num_nodes) & \
                                (ns.edge_ids[0,:] == nid)
                    local_cumchs += c.num_nodes

                    ch_ids = ns.edge_ids[1,criterion] + c._output_ind_range[0]
                    parids[ch_ids, par_counts[ch_ids]] = n_start_id + node_start + nid
                    parpids[ch_ids, par_counts[ch_ids]] = pids[node_start+nid,:len(ch_ids)]
                    parcids[ch_ids, par_counts[ch_ids]] = torch.arange(ch_start, ch_start + ch_ids.size(0))

                    par_counts[ch_ids] += 1
                    ch_start += criterion.size(0)

            node_start = node_end

        # Find a good strategy to partition the child nodes into groups according to their number of parents 
        # to minimize total computation cost
        ch_n_pars = ch_n_pars[1:] # Strip away the dummy node. We will never use it in the following
        par_group_sizes = partition_nodes_by_n_edges(ch_n_pars, max_num_groups = max_num_groups)

        grouped_chids = []
        grouped_parids = []
        grouped_parpids = []
        grouped_parcids = []
        grouped_seq_ids0 = []
        grouped_seq_ids1 = []
        grouped_seq_parpids = []
        min_n_pars = 0
        for max_n_pars in par_group_sizes:
            id_filter = (ch_n_pars >= min_n_pars) & (ch_n_pars <= max_n_pars)
            filtered_idxs = torch.where(id_filter)[0]
            curr_chids = (filtered_idxs + 1).clone()
            curr_parids = parids[filtered_idxs+1,:max_n_pars].contiguous()
            curr_parpids = parpids[filtered_idxs+1,:max_n_pars].contiguous()
            curr_parcids = parcids[filtered_idxs+1,:max_n_pars].contiguous()

            curr_tot_npar = ch_n_pars[id_filter].sum()
            curr_seq_ids0 = torch.zeros([curr_tot_npar], dtype = torch.long)
            curr_seq_ids1 = torch.zeros([curr_tot_npar], dtype = torch.long)
            curr_seq_parpids = torch.zeros([curr_tot_npar], dtype = torch.long)
            s_idx = 0
            for i in range(filtered_idxs.size(0)):
                nid = filtered_idxs[i]
                num_pars = ch_n_pars[nid] # No need to add 1 here since the dummy node is already stripped away
                e_idx = s_idx + num_pars

                curr_seq_ids0[s_idx:e_idx] = i
                curr_seq_ids1[s_idx:e_idx] = torch.arange(num_pars)
                curr_seq_parpids[s_idx:e_idx] = parpids[nid+1,:num_pars]

                s_idx = e_idx

            grouped_chids.append(curr_chids)
            grouped_parids.append(curr_parids)
            grouped_parpids.append(curr_parpids)
            grouped_parcids.append(curr_parcids)
            grouped_seq_ids0.append(curr_seq_ids0)
            grouped_seq_ids1.append(curr_seq_ids1)
            grouped_seq_parpids.append(curr_seq_parpids)

            min_n_pars = max_n_pars + 1

        self.num_backward_groups = par_group_sizes.shape[0]
        self.grouped_chids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_chids])
        self.grouped_parids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_parids])
        self.grouped_parpids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_parpids])
        self.grouped_parcids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_parcids])
        self.grouped_seq_ids0 = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_seq_ids0])
        self.grouped_seq_ids1 = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_seq_ids1])
        self.grouped_seq_parpids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_seq_parpids])

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

        for group_id in range(self.num_forward_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]
            pids = self.grouped_pids[group_id]

            self._forward_triton(node_mars, element_mars, params, nids, cids, pids)

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
        
        for group_id in range(self.num_backward_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parpids = self.grouped_parpids[group_id]
            seq_ids0 = self.grouped_seq_ids0[group_id]
            seq_ids1 = self.grouped_seq_ids1[group_id]
            seq_parpids = self.grouped_seq_parpids[group_id]

            self._backward_triton(node_flows, element_flows, params, node_mars, 
                                  element_mars, param_flows, chids, parids, parpids)

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
                               tot_n_pars: tl.constexpr, 
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

    def _forward_triton(self, node_mars: torch.Tensor, element_mars: torch.Tensor, 
                        params: torch.Tensor,
                        nids: torch.Tensor, cids: torch.Tensor,
                        pids: torch.Tensor, BLOCK_SIZE = 2**12, MAX_BLOCK_M = 512, MAX_BLOCK_N = 64) -> None:
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
        tot_n_pars = params.size(0)
        n_nodes = nids.size(0)
        n_edges = cids.size(1)
        batch_size = node_mars.size(1)

        if params.dim() == 2 and params.size(1) == 1:
            params = params.squeeze(1)

        assert n_edges <= MAX_BLOCK_M, "Number of edges should be smaller than or equal to MAX_BLOCK_M."
        assert params.dim() == 1, "Expecting a 1D `params`."

        MIN_BLOCK_M = min(triton.next_power_of_2(n_edges), MAX_BLOCK_M)
        BLOCK_N = min(BLOCK_SIZE // MIN_BLOCK_M, MAX_BLOCK_N, triton.next_power_of_2(batch_size))
        BLOCK_M = min(BLOCK_SIZE // BLOCK_N, MAX_BLOCK_M)

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
            tot_n_pars = tot_n_pars,
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
        ele_mask = tl.broadcast_to(n_mask[None,:], (BLOCK_N, n_nodes_per_block_m))
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

    def _backward_triton(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                         params: torch.Tensor, node_mars: torch.Tensor, 
                         element_mars: torch.Tensor, param_flows: torch.Tensor, 
                         chids: torch.Tensor, parids: torch.Tensor, parpids: torch.Tensor, 
                         BLOCK_SIZE = 2**12, MAX_BLOCK_M = 512, MAX_BLOCK_N = 64) -> None:
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

        assert n_edges <= MAX_BLOCK_M, "Number of edges should be smaller than or equal to MAX_BLOCK_M."
        assert params.dim() == 1, "Expecting a 1D `params`."

        MIN_BLOCK_M = min(triton.next_power_of_2(n_edges), MAX_BLOCK_M)
        BLOCK_N = min(BLOCK_SIZE // MIN_BLOCK_M, MAX_BLOCK_N, triton.next_power_of_2(batch_size))
        BLOCK_M = min(BLOCK_SIZE // BLOCK_N, MAX_BLOCK_M)

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
        for group_id in range(self.num_forward_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]
            pids = self.grouped_pids[group_id]

            ch_mars = element_mars[cids]
            maxval = ch_mars.max(dim = 1, keepdim = True).values
            unnorm_probs = (ch_mars - maxval).exp() * params[pids]
            dist = torch.distributions.Categorical(probs = unnorm_probs.permute(0, 2, 1))
            node_mask[nids] = dist.sample() # [num nodes, batch_size]

        return None

    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _sample_backward_pass(self, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                              element_mars: torch.Tensor, params: torch.Tensor, node_mask: torch.Tensor):
        for group_id in range(self.num_backward_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parcids = self.grouped_parcids[group_id]

            element_flows[chids] = (node_flows[parids] * (node_mask[parids] == parcids.unsqueeze(-1))).any(dim = 1)

        return None