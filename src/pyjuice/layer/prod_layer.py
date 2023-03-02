from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
from pyjuice.graph.region_graph import RegionGraph, PartitionNode
from typing import List
from .layer import Layer

# Try to enable tensor cores
torch.set_float32_matmul_precision('high')


class ProdLayer(Layer,nn.Module):

    def __init__(self, layer_id: int, region_nodes: List[RegionGraph]) -> None:
        Layer.__init__(self, layer_id)
        nn.Module.__init__(self)

        assert len(region_nodes) > 0, "No input region node."
        for rnode in region_nodes:
            assert isinstance(rnode, PartitionNode), "ProdLayer must respect to PartitionNode."

        self.region_nodes = region_nodes

        max_n_chs = 0
        layer_num_nodes = 0
        cum_nodes = 1 # id 0 is reserved for the dummy node
        for rnode in self.region_nodes:
            if rnode.num_chs > max_n_chs:
                max_n_chs = rnode.num_chs
            rnode._output_ind_range = (cum_nodes, cum_nodes + rnode.num_nodes)
            cum_nodes += rnode.num_nodes
            layer_num_nodes += rnode.num_nodes

        self.num_nodes = layer_num_nodes

        ## Initialize forward pass ##

        cids = torch.zeros([self.num_nodes, max_n_chs], dtype = torch.long) # cids[i,:] are the child node ids for node i
        node_start = 0
        for rnode in self.region_nodes:
            node_end = node_start + rnode.num_nodes
            for i, c in enumerate(rnode.children):
                cids[node_start:node_end, i] = rnode.edge_ids[:,i] + c._output_ind_range[0]

            node_start = node_end

        self.register_buffer("cids", cids)

        ## Initialize backward pass ##

        u_cids, par_counts = torch.unique(cids, sorted = True, return_counts = True)

        max_n_pars = torch.max(par_counts[1:])
        parids = torch.zeros([u_cids.size(0), max_n_pars], dtype = torch.long)
        for idx in range(u_cids.size(0)):
            cid = u_cids[idx]
            if cid == 0:
                continue # Skip the dummy node
            b_nid = torch.where(cids == cid)[0]
            parids[idx,:b_nid.size(0)] = b_nid + 1 # 1 for the dummy node

        self.register_buffer("parids", parids)
        self.register_buffer("u_cids", u_cids)

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, skip_logsumexp: bool = False):
        """
        node_mars: [num_nodes, B]
        element_mars: [max_num_els, B]
        """
        if skip_logsumexp:
            self._dense_forward_pass_nolog(node_mars, element_mars)
        else:
            self._dense_forward_pass(node_mars, element_mars)

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, skip_logsumexp: bool = False):
        """
        node_flows: [num_nodes, B]
        element_flows: [max_num_els, B]
        """
        if skip_logsumexp:
            self._dense_forward_pass_nolog(node_flows, element_flows)
        else:
            self._dense_backward_pass(node_flows, element_flows)

    @torch.compile(mode = "reduce-overhead")
    def _dense_forward_pass(self, node_mars: torch.Tensor, element_mars: torch.Tensor):
        element_mars[1:self.num_nodes+1,:] = node_mars[self.cids].sum(dim = 1)
        return None

    @torch.compile(mode = "reduce-overhead")
    def _dense_forward_pass_nolog(self, node_mars: torch.Tensor, element_mars: torch.Tensor):
        element_mars[1:self.num_nodes+1,:] = node_mars[self.cids].prod(dim = 1)
        return None

    @torch.compile(mode = "reduce-overhead")
    def _dense_backward_pass(self, node_flows: torch.Tensor, element_flows: torch.Tensor):
        node_flows[self.u_cids] += element_flows[self.parids].sum(dim = 1)
        
        return None
