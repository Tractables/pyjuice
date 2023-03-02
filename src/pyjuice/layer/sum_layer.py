
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
from typing import Tuple, List, Optional

from pyjuice.graph.region_graph import RegionGraph, InnerRegionNode
from .layer import Layer

# Try to enable tensor cores
torch.set_float32_matmul_precision('high')


class SumLayer(Layer,nn.Module):

    def __init__(self, layer_id: int, region_nodes: List[RegionGraph], 
                                        cum_nodes: int, 
                                        param_ends: List, 
                                        ch_prod_layer_size: int,
                                        sum_region_start_id: int = 0) -> None:
        Layer.__init__(self, layer_id)
        nn.Module.__init__(self)

        assert len(region_nodes) > 0, "No input region node."
        for rnode in region_nodes:
            assert isinstance(rnode, InnerRegionNode), "SumLayer must respect to InnerRegionNode."

        self.region_nodes = region_nodes
        sum_region_ids = []

        layer_num_nodes = 0
        total_num_edges = 0
        max_n_chs = 0
        for (rnode_idx, rnode) in enumerate(self.region_nodes):
            n_chs = torch.max(torch.bincount(rnode.edge_ids[0,:])).item()
            if n_chs > max_n_chs:
                max_n_chs = n_chs
            rnode._output_ind_range = (cum_nodes, cum_nodes + rnode.num_nodes)
            cum_nodes += rnode.num_nodes
            layer_num_nodes += rnode.num_nodes
            total_num_edges += rnode.edge_ids.size(1)

            sum_region_ids.extend([(sum_region_start_id + rnode_idx) for i in range(rnode.num_nodes)])

        self.num_nodes = layer_num_nodes
        self.num_params = total_num_edges

        self.ch_prod_layer_size = ch_prod_layer_size

        ## Initialize forward pass ##

        n_start_id = cum_nodes - self.num_nodes
        self.nrange = (cum_nodes - self.num_nodes, cum_nodes) # Range of node ids
        cids = torch.zeros([self.num_nodes, max_n_chs], dtype = torch.long) # Child id
        pids = torch.zeros([self.num_nodes, max_n_chs], dtype = torch.long) # Parameter id
        ch_n_pars = torch.zeros([self.ch_prod_layer_size], dtype = torch.long) # Number of parents for each child node
        node_start = 0
        param_start = param_ends[-1]
        for rnode in self.region_nodes:

            param_ids = torch.zeros([rnode.edge_ids.size(1)], dtype = torch.long)
            rnode_param_start = param_start

            node_end = node_start + rnode.num_nodes
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
                
            node_start = node_end

            rnode_param_end = param_start
            rnode._param_range = (rnode_param_start, rnode_param_end)
            rnode._param_ids = param_ids
            rnode._inverse_param_ids = torch.argsort(param_ids)

        self.register_buffer("cids", cids)
        self.register_buffer("pids", pids)
        self.register_buffer("sum_region_ids", torch.tensor(sum_region_ids, dtype=torch.long))

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

        self.register_buffer("parids", parids[1:,:]) # Strip away the dummy node
        self.register_buffer("parpids", parpids[1:,:])

        seq_ids0 = torch.zeros([tot_n_pars], dtype = torch.long)
        seq_ids1 = torch.zeros([tot_n_pars], dtype = torch.long)
        seq_parpids = torch.zeros([tot_n_pars], dtype = torch.long)
        s_idx = 0
        for idx in range(self.ch_prod_layer_size - 1):
            num_pars = ch_n_pars[idx+1]
            e_idx = s_idx + num_pars

            seq_ids0[s_idx:e_idx] = torch.ones([num_pars], dtype = torch.long) * idx
            seq_ids1[s_idx:e_idx] = torch.arange(num_pars)
            seq_parpids[s_idx:e_idx] = parpids[idx+1, :num_pars]

            s_idx = e_idx

        self.register_buffer("seq_ids0", seq_ids0)
        self.register_buffer("seq_ids1", seq_ids1)
        self.register_buffer("seq_parpids", seq_parpids)

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, sum_region_mars: Optional[torch.tensor] = None, skip_logsumexp: bool = False):
        """
        node_mars: [num_nodes, B]
        element_mars: [max_num_els, B]
        params: [num_params, B]
        """
        if skip_logsumexp:
            self._dense_forward_pass_nolog(node_mars, element_mars, params, sum_region_mars)
        else:
            self._dense_forward_pass(node_mars, element_mars, params)

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                    element_mars: torch.Tensor, params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                    sum_region_mars: Optional[torch.Tensor] = None,
                    skip_logsumexp: bool = False):
        """
        node_flows: [num_nodes, B]
        element_flows: [max_num_els, B]
        node_mars: [num_nodes, B]
        element_mars: [max_num_els, B]
        params: [num_params, B]
        """
        if skip_logsumexp:
            self._dense_backward_pass_nolog(node_flows, element_flows, node_mars, element_mars, params, param_flows = param_flows, sum_region_mars=sum_region_mars)
        else:
            self._dense_backward_pass(node_flows, element_flows, node_mars, element_mars, params, param_flows = param_flows)

    @torch.compile(mode = "reduce-overhead")
    def _dense_forward_pass(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor):
        ch_mars = element_mars[self.cids]
        maxval = ch_mars.max(dim = 1, keepdim = True).values
        node_mars[self.nrange[0]:self.nrange[1]] = (((ch_mars - maxval).exp() * params[self.pids].unsqueeze(-1)).sum(
            dim = 1).clamp(min=1e-10)).log() + maxval.squeeze()

        return None
    
    @torch.compile(mode = "reduce-overhead")
    def _dense_forward_pass_nolog(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor, sum_region_mars: torch.tensor):
        node_mars[self.nrange[0]:self.nrange[1]] = (element_mars[self.cids] * params[self.pids].unsqueeze(-1)).sum(dim = 1)
        sum_region_mars.index_add_(0, self.sum_region_ids, node_mars[self.nrange[0]:self.nrange[1]], alpha=1.0)
        node_mars[self.nrange[0]:self.nrange[1]] = node_mars[self.nrange[0]:self.nrange[1]].div(sum_region_mars[self.sum_region_ids])

        return None


    @torch.compile(mode = "reduce-overhead")
    def _dense_backward_pass(self, node_flows: torch.Tensor, element_flows: torch.Tensor, node_mars: torch.Tensor, 
                             element_mars: torch.Tensor, params: torch.Tensor, param_flows: Optional[torch.Tensor] = None):
        element_flows[1:self.ch_prod_layer_size] = (node_flows[self.parids] * params[self.parpids].unsqueeze(2) * \
            (element_mars[1:self.ch_prod_layer_size].unsqueeze(1) - node_mars[self.parids]).exp()).sum(dim = 1)

        if param_flows is not None:
            param_flows[self.seq_parpids] += (node_flows[self.parids] * params[self.parpids].unsqueeze(2) * \
                (element_mars[1:self.ch_prod_layer_size].unsqueeze(1) - node_mars[self.parids]).exp()).sum(dim = 2)[self.seq_ids0, self.seq_ids1]

        return None
    
    @torch.compile(mode = "reduce-overhead")
    def _dense_backward_pass_nolog(self, node_flows: torch.Tensor, element_flows: torch.Tensor, 
                                        node_mars: torch.Tensor, 
                                        element_mars: torch.Tensor, 
                                        params: torch.Tensor, 
                                        sum_region_mars: torch.tensor,
                                        param_flows: Optional[torch.Tensor] = None):
        
        element_flows[1:self.ch_prod_layer_size] = (node_flows[self.parids] * params[self.parpids].unsqueeze(2) * \
            (element_mars[1:self.ch_prod_layer_size].unsqueeze(1) / node_mars[self.parids] )).sum(dim = 1)

        if param_flows is not None:
            param_flows[self.seq_parpids] += (node_flows[self.parids] * params[self.parpids].unsqueeze(2) * \
                (element_mars[1:self.ch_prod_layer_size].unsqueeze(1) / node_mars[self.parids] )).sum(dim = 2)[self.seq_ids0, self.seq_ids1]

        return None
