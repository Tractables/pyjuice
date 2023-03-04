from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import List, Dict

from pyjuice.graph.region_graph import RegionGraph, InputRegionNode
from pyjuice.layer.input_layer import InputLayer


class CategoricalLayer(InputLayer, nn.Module):
    def __init__(self, layer_id: int, region_nodes: List[RegionGraph], cum_nodes: int = 0) -> None:
        nn.Module.__init__(self)

        num_nodes = sum(map(lambda r: r.num_nodes, region_nodes))

        InputLayer.__init__(self, layer_id, region_nodes, num_nodes)

        self.vars = []
        self.rnode_num_nodes = []
        self.rnode_num_cats = []
        self.param_ends = []
        layer_num_nodes = 0
        cum_params = 0
        num_nodes = 0
        for rnode in self.region_nodes:
            assert len(rnode.scope) == 1, "CategoricalLayer only support uni-variate categorical distributions."

            self.vars.append(next(iter(rnode.scope)))
            self.rnode_num_nodes.append(rnode.num_nodes)
            self.rnode_num_cats.append(rnode.extra_params["num_cats"])

            num_nodes += rnode.num_nodes

            rnode._output_ind_range = (cum_nodes, cum_nodes + rnode.num_nodes)
            cum_nodes += rnode.num_nodes
            layer_num_nodes += rnode.num_nodes

            for nid in range(1, rnode.num_nodes + 1):
                self.param_ends.append(cum_params + rnode.extra_params["num_cats"] * nid)
            cum_params += rnode.num_nodes * rnode.extra_params["num_cats"]

        self._output_ind_range = (cum_nodes - layer_num_nodes, cum_nodes)
        self.param_ends = torch.tensor(self.param_ends)

        self.num_params = cum_params
        self.num_nodes = num_nodes

        vids = torch.empty([layer_num_nodes], dtype = torch.long)
        psids = torch.empty([layer_num_nodes], dtype = torch.long)
        n_start = 0
        for idx, rnode in enumerate(self.region_nodes):
            n_end = n_start + rnode.num_nodes
            vids[n_start:n_end] = self.vars[idx]

            if idx == 0:
                psids[0] = 0
                psids[1:n_end] = torch.tensor(self.param_ends[0:n_end-1])
            else:
                psids[n_start:n_end] = torch.tensor(self.param_ends[n_start-1:n_end-1])

            n_start = n_end

        self.register_buffer("vids", vids)
        self.register_buffer("psids", psids)

        # For parameter normalization
        node_ids = torch.empty([self.num_params], dtype = torch.long)
        node_nchs = torch.empty([self.num_nodes], dtype = torch.long)
        node_ids[:self.param_ends[0]] = 0
        node_nchs[0] = self.param_ends[0]
        for i in range(1, len(self.param_ends)):
            node_ids[self.param_ends[i-1]:self.param_ends[i]] = i
            node_nchs[i] = self.param_ends[i] - self.param_ends[i-1]
        
        self.register_buffer("node_ids", node_ids)
        self.register_buffer("node_nchs", node_nchs)

        # Initialize parameters
        self._init_params()

        self.param_flows_size = self.params.size(0)

    @torch.compile(mode = "reduce-overhead")
    def forward(self, data: torch.Tensor, node_mars: torch.Tensor, skip_logsumexp: bool = False):
        """
        data: [num_vars, B]
        node_mars: [num_nodes, B]
        """
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]
        if skip_logsumexp:
            node_mars[sid:eid,:] = ((self.params[data[self.vids] + self.psids.unsqueeze(1)] + 1e-8).clamp(min=1e-10))
        else:
            node_mars[sid:eid,:] = ((self.params[data[self.vids] + self.psids.unsqueeze(1)] + 1e-8).clamp(min=1e-10)).log()

        return None

    def backward(self, data: torch.Tensor, node_flows: torch.Tensor):
        """
        data: [num_vars, B]
        node_flows: [num_nodes, B]
        """
        layer_num_nodes = self._output_ind_range[1] - self._output_ind_range[0]
        tot_num_nodes = node_flows.size(0)
        batch_size = node_flows.size(1)
        node_offset = self._output_ind_range[0]

        param_ids = data[self.vids] + self.psids.unsqueeze(1)
        
        grid = lambda meta: (triton.cdiv(layer_num_nodes * batch_size, meta['BLOCK_SIZE']),)
        self._flows_kernel[grid](self.param_flows, node_flows, param_ids, layer_num_nodes, tot_num_nodes, batch_size, node_offset, BLOCK_SIZE = 1024)
        
        return None

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        with torch.no_grad():
            flows = self.param_flows
            self._normalize_parameters(flows, pseudocount = pseudocount)
            self.params.data = (1.0 - step_size) * self.params.data + step_size * flows

    def _init_params(self, perturbation: float = 4.0):
        """
        Initialize the parameters with random values
        """
        params = torch.exp(torch.rand([self.num_params]) * -perturbation)
        
        n_start = 0
        for idx, rnode in enumerate(self.region_nodes):
            n_end = n_start + rnode.num_nodes

            if hasattr(rnode, "_params"):
                if idx == 0:
                    par_start = 0
                    par_end = self.param_ends[n_end-1]
                else:
                    par_start = self.param_ends[n_start-1]
                    par_end = self.param_ends[n_end-1]

                params[par_start:par_end] = rnode._params["params"].to(params.device)

            n_start = n_end

        self._normalize_parameters(params)
        self.params = nn.Parameter(params)

        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.params.requires_grad = False

    def _extract_params_to_rnodes(self):
        raise NotImplementedError()

    @staticmethod
    def _prune_nodes(_params: Dict, node_keep_flag: torch.Tensor):
        raise NotImplementedError()

    @staticmethod
    def _duplicate_nodes(_params: Dict):
        raise NotImplementedError()