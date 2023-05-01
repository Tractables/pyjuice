
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
from typing import Tuple, List, Optional

from pyjuice.graph.region_graph import RegionGraph, InnerRegionNode
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges

# Try to enable tensor cores
torch.set_float32_matmul_precision('high')


class SumLayer(Layer,nn.Module):

    def __init__(self, layer_id: int, region_nodes: List[RegionGraph], 
                 cum_nodes: int, 
                 param_ends: List, 
                 ch_prod_layer_size: int,
                 sum_region_start_id: int = 0,
                 max_num_groups: int = 1) -> None:
        Layer.__init__(self, layer_id)
        nn.Module.__init__(self)

        assert len(region_nodes) > 0, "No input region node."
        for rnode in region_nodes:
            assert isinstance(rnode, InnerRegionNode), "SumLayer must respect to InnerRegionNode."

        self.region_nodes = region_nodes

        layer_num_nodes = 0
        total_num_edges = 0
        max_n_chs = 0
        for rnode_idx, rnode in enumerate(self.region_nodes):
            n_chs = torch.max(torch.bincount(rnode.edge_ids[0,:])).item()
            if n_chs > max_n_chs:
                max_n_chs = n_chs
            rnode._output_ind_range = (cum_nodes, cum_nodes + rnode.num_nodes)
            cum_nodes += rnode.num_nodes
            layer_num_nodes += rnode.num_nodes
            total_num_edges += rnode.edge_ids.size(1)

        self.num_nodes = layer_num_nodes
        self.num_params = total_num_edges

        self.ch_prod_layer_size = ch_prod_layer_size

        ## Initialize forward pass ##

        n_start_id = cum_nodes - self.num_nodes
        nids = torch.arange(cum_nodes - self.num_nodes, cum_nodes) # Node id
        cids = torch.zeros([self.num_nodes, max_n_chs], dtype = torch.long) # Child id
        pids = torch.zeros([self.num_nodes, max_n_chs], dtype = torch.long) # Parameter id
        srids = torch.zeros([self.num_nodes], dtype = torch.long) # Sum region id
        n_chs = torch.zeros([self.num_nodes], dtype = torch.long) # Number of children
        ch_n_pars = torch.zeros([self.ch_prod_layer_size], dtype = torch.long) # Number of parents for each child node
        node_start = 0
        param_start = param_ends[-1]
        for rnode_idx, rnode in enumerate(self.region_nodes):

            param_ids = torch.zeros([rnode.edge_ids.size(1)], dtype = torch.long)
            rnode_param_start = param_start

            node_end = node_start + rnode.num_nodes

            srids[node_start:node_end] = sum_region_start_id + rnode_idx

            for nid in range(rnode.num_nodes):
                ch_start = 0
                local_cumchs = 0
                for c in rnode.children:
                    criterion = (rnode.edge_ids[1,:] >= local_cumchs) & (rnode.edge_ids[1,:] < local_cumchs + c.num_nodes) & \
                                (rnode.edge_ids[0,:] == nid)
                    local_cumchs += c.num_nodes

                    ch_ids = rnode.edge_ids[1,criterion] + c._output_ind_range[0]
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

            rnode_param_end = param_start
            rnode._param_range = (rnode_param_start, rnode_param_end)
            rnode._param_ids = param_ids
            rnode._inverse_param_ids = torch.argsort(param_ids)

        # Find a good strategy to partition the nodes into groups according to their number of children 
        # to minimize total computation cost
        ch_group_sizes = partition_nodes_by_n_edges(n_chs, max_num_groups = max_num_groups)

        grouped_nids = []
        grouped_cids = []
        grouped_pids = []
        grouped_srids = []
        min_n_chs = 0
        for max_n_chs in ch_group_sizes:
            filter = (n_chs >= min_n_chs) & (n_chs <= max_n_chs)
            curr_nids = nids[filter].contiguous()
            curr_cids = cids[filter,:max_n_chs].contiguous()
            curr_pids = pids[filter,:max_n_chs].contiguous()
            curr_srids = srids[filter].contiguous()

            grouped_nids.append(curr_nids)
            grouped_cids.append(curr_cids)
            grouped_pids.append(curr_pids)
            grouped_srids.append(curr_srids)

            min_n_chs = max_n_chs + 1

        self.num_forward_groups = ch_group_sizes.shape[0]
        self.grouped_nids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_nids])
        self.grouped_cids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_cids])
        self.grouped_pids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_pids])
        self.grouped_srids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_srids])

        ## Initialize backward pass ##

        ch_n_pars[0] = 0 # We do not need to compute flows for the dummy node

        max_n_pars = torch.max(ch_n_pars)
        tot_n_pars = torch.sum(ch_n_pars[1:]) # Exclude the dummy node

        parids = torch.zeros([self.ch_prod_layer_size, max_n_pars], dtype = torch.long) # Indices of parent nodes for each child node
        parpids = torch.zeros([self.ch_prod_layer_size, max_n_pars], dtype = torch.long) # Parameter indices for these edges
        par_counts = torch.zeros([self.ch_prod_layer_size], dtype = torch.long)
        node_start = 0
        for rnode in self.region_nodes:
            node_end = node_start + rnode.num_nodes
            for nid in range(rnode.num_nodes):
                ch_start = 0
                local_cumchs = 0
                for c in rnode.children:
                    criterion = (rnode.edge_ids[1,:] >= local_cumchs) & (rnode.edge_ids[1,:] < local_cumchs + c.num_nodes) & \
                                (rnode.edge_ids[0,:] == nid)
                    local_cumchs += c.num_nodes

                    ch_ids = rnode.edge_ids[1,criterion] + c._output_ind_range[0]
                    parids[ch_ids, par_counts[ch_ids]] = n_start_id + node_start + nid
                    parpids[ch_ids, par_counts[ch_ids]] = pids[node_start+nid,:len(ch_ids)]

                    par_counts[ch_ids] += 1

            node_start = node_end

        # Find a good strategy to partition the child nodes into groups according to their number of parents 
        # to minimize total computation cost
        ch_n_pars = ch_n_pars[1:] # Strip away the dummy node. We will never use it in the following
        par_group_sizes = partition_nodes_by_n_edges(ch_n_pars, max_num_groups = max_num_groups)

        grouped_chids = []
        grouped_parids = []
        grouped_parpids = []
        grouped_seq_ids0 = []
        grouped_seq_ids1 = []
        grouped_seq_parpids = []
        min_n_pars = 0
        for max_n_pars in par_group_sizes:
            filter = (ch_n_pars >= min_n_pars) & (ch_n_pars <= max_n_pars)
            filtered_idxs = torch.where(filter)[0]
            curr_chids = (filtered_idxs + 1).clone()
            curr_parids = parids[filtered_idxs+1,:max_n_pars].contiguous()
            curr_parpids = parpids[filtered_idxs+1,:max_n_pars].contiguous()

            curr_tot_npar = ch_n_pars[filter].sum()
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
            grouped_seq_ids0.append(curr_seq_ids0)
            grouped_seq_ids1.append(curr_seq_ids1)
            grouped_seq_parpids.append(curr_seq_parpids)

            min_n_pars = max_n_pars + 1

        self.num_backward_groups = par_group_sizes.shape[0]
        self.grouped_chids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_chids])
        self.grouped_parids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_parids])
        self.grouped_parpids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_parpids])
        self.grouped_seq_ids0 = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_seq_ids0])
        self.grouped_seq_ids1 = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_seq_ids1])
        self.grouped_seq_parpids = nn.ParameterList([nn.Parameter(tensor, requires_grad = False) for tensor in grouped_seq_parpids])

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, 
                sum_region_mars: Optional[torch.tensor] = None, skip_logsumexp: bool = False):
        """
        node_mars: [num_nodes, B]
        element_mars: [max_num_els, B]
        params: [num_params, B] or [num_params]
        """
        if params.dim() == 1:
            params = params.unsqueeze(1)

        if skip_logsumexp:
            self._dense_forward_pass_nolog(node_mars, element_mars, params, sum_region_mars)
        else:
            self._dense_forward_pass(node_mars, element_mars, params)

    # @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def sample(self, node_flows: torch.Tensor, 
                        element_flows: torch.Tensor, 
                        node_mars: torch.Tensor, 
                        element_mars: torch.Tensor, 
                        params: torch.Tensor):
        """
        node_flows:         [num_nodes, B]
        element_flows:      [max_num_els, B]
        node_mars:          [num_nodes, B]
        element_mars:       [max_num_els, B]
        params:             [num_params] or [num_params, B]
        """
        if params.dim() == 1:
            params = params.unsqueeze(1)

        for group_id in range(self.num_backward_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]   # (num_sum_nodes, max_sum_children)
            pids = self.grouped_pids[group_id]

            chids = self.grouped_chids[group_id]

            # For each sum node `n` we need to sample a child proportional to theta_{c|n} * pr_c
            # That child c* will have all the flow_c (and flow_c was either 0 or 1), i.e:
            #       flow_c* = flow_n
            #       flow_c =  0 for every other c != c*
            probs = params[pids] * element_mars[cids].exp()                                      # (num_sum_nodes, max_sum_children, batch_size)
            cummul_probs = torch.cumsum(probs[:, :, :], 1)                                       # (num_sum_nodes, max_sum_children, batch_size)
            cummul_probs_last = cummul_probs[:, -1:, :]                                          # (num_sum_nodes, 1, batch_size)
            
            rand = torch.rand((probs.size(0), 1, probs.size(2)))#.cuda()                         # (num_sum_nodes, 1, batch_size)
            rand = cummul_probs_last * rand                                                      # (num_sum_nodes, 1, batch_size)   

            sampled_idx = (torch.sum(rand > cummul_probs, dim=1).long())                         # (num_sum_nodes, batch_size)             
            sampled_child_ids = torch.gather(cids, 1, sampled_idx) - 1                           # (num_sum_nodes, batch_size)
            
            # print("sampled_idx\n", sampled_idx)
            # print("sampled_child_ids\n", sampled_child_ids)
            # print("bad stuff in sampled_child_ids", (sampled_child_ids >= element_flows[chids].size(0)).sum())
            
            # print("element_flows", element_flows.size())
            # print("element_flows[chids]", element_flows[chids].size())
            # print("node_flows", node_flows.size())
            # print("node_flows[nids]", node_flows[nids].size())
            # print("node_mars", node_mars.size())
            # print("chids", chids.size())
            # print("cids", cids.size())
            element_flows[chids] = torch.scatter_add(element_flows[chids], dim=0, index=sampled_child_ids, src=node_flows[nids])
            # element_flows[chids].scatter_add_(dim=0, index=sampled_child_ids, src=node_flows[nids])



    def backward(self, node_flows: torch.Tensor, 
                        element_flows: torch.Tensor, 
                        node_mars: torch.Tensor, 
                        element_mars: torch.Tensor, 
                        params: torch.Tensor, 
                        param_flows: Optional[torch.Tensor] = None,
                        sum_region_mars: Optional[torch.Tensor] = None,
                        skip_logsumexp: bool = False):
        """
        node_flows: [num_nodes, B]
        element_flows: [max_num_els, B]
        node_mars: [num_nodes, B]
        element_mars: [max_num_els, B]
        params: [num_params, B] or [num_params]
        """
        if params.dim() == 1:
            params = params.unsqueeze(1)

        if skip_logsumexp:
            self._dense_backward_pass_nolog(node_flows, element_flows, node_mars, element_mars, params, param_flows = param_flows, sum_region_mars=sum_region_mars)
        else:
            if param_flows is None:
                self._dense_backward_pass1(node_flows, element_flows, node_mars, element_mars, params, param_flows = param_flows)
            elif params.size(1) == 1:
                self._dense_backward_pass2(node_flows, element_flows, node_mars, element_mars, params, param_flows = param_flows)
            else:
                self._dense_backward_pass3(node_flows, element_flows, node_mars, element_mars, params, param_flows = param_flows)

    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _dense_forward_pass(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor):
        for group_id in range(self.num_forward_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]
            pids = self.grouped_pids[group_id]

            ch_mars = element_mars[cids]
            maxval = ch_mars.max(dim = 1, keepdim = True).values
            node_mars[nids] = (((ch_mars - maxval).exp() * params[pids]).sum(
                dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

        return None
    
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _dense_forward_pass_nolog(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, sum_region_mars: torch.tensor):
        for group_id in range(self.num_forward_groups):
            nids = self.grouped_nids[group_id]
            cids = self.grouped_cids[group_id]
            pids = self.grouped_pids[group_id]
            srids = self.grouped_srids[group_id]

            node_mars[nids] = (element_mars[cids] * params[pids]).sum(dim = 1)
            sum_region_mars.index_add_(0, srids, node_mars[nids], alpha = 1.0)
            node_mars[nids] = node_mars[nids].div(sum_region_mars[srids])

        return None

    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _dense_backward_pass1(self, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                              element_mars: torch.Tensor, params: torch.Tensor):
        for group_id in range(self.num_backward_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parpids = self.grouped_parpids[group_id]

            element_flows[chids] = (node_flows[parids] * params[parpids] * \
                (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

        return None

    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _dense_backward_pass2(self, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                              element_mars: torch.Tensor, params: torch.Tensor, param_flows: torch.Tensor):
        for group_id in range(self.num_backward_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parpids = self.grouped_parpids[group_id]
            seq_ids0 = self.grouped_seq_ids0[group_id]
            seq_ids1 = self.grouped_seq_ids1[group_id]
            seq_parpids = self.grouped_seq_parpids[group_id]

            element_flows[chids] = (node_flows[parids] * params[parpids] * \
                (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

            param_flows[seq_parpids] += (node_flows[parids] * params[parpids] * \
                (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 2)[seq_ids0, seq_ids1]

        return None

    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _dense_backward_pass3(self, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                              element_mars: torch.Tensor, params: torch.Tensor, param_flows: torch.Tensor):
        for group_id in range(self.num_backward_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parpids = self.grouped_parpids[group_id]
            seq_ids0 = self.grouped_seq_ids0[group_id]
            seq_ids1 = self.grouped_seq_ids1[group_id]
            seq_parpids = self.grouped_seq_parpids[group_id]

            element_flows[chids] = (node_flows[parids] * params[parpids] * \
                (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp()).sum(dim = 1)

            param_flows[self.seq_parpids] += (node_flows[parids] * params[parpids] * \
                (element_mars[chids].unsqueeze(1) - node_mars[parids]).exp())[seq_ids0, seq_ids1]

        return None
    
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _dense_backward_pass_nolog(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                                   node_mars: torch.Tensor, 
                                   element_mars: torch.Tensor, 
                                   params: torch.Tensor, 
                                   sum_region_mars: torch.tensor,
                                   param_flows: Optional[torch.Tensor] = None):
        for group_id in range(self.num_backward_groups):
            chids = self.grouped_chids[group_id]
            parids = self.grouped_parids[group_id]
            parpids = self.grouped_parpids[group_id]
            seq_ids0 = self.grouped_seq_ids0[group_id]
            seq_ids1 = self.grouped_seq_ids1[group_id]
            seq_parpids = self.grouped_seq_parpids[group_id]
        
            element_flows[chids] = (node_flows[parids] * params[parpids] * \
                (element_mars[chids].unsqueeze(1) / node_mars[parids] )).sum(dim = 1)

            if param_flows is not None:
                param_flows[self.seq_parpids] += (node_flows[parids] * params[parpids] * \
                    (element_mars[chids].unsqueeze(1) / node_mars[parids] )).sum(dim = 2)[seq_ids0, seq_ids1]

        return None
