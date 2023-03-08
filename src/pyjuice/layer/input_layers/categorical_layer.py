from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import List, Dict, Optional

from pyjuice.graph.region_graph import RegionGraph, InputRegionNode
from pyjuice.layer.input_layer import InputLayer
from pyjuice.functional import normalize_parameters

# Try to enable tensor cores
torch.set_float32_matmul_precision('high')


class CategoricalLayer(InputLayer):
    def __init__(self, layer_id: int, region_nodes: List[RegionGraph], cum_nodes: int = 0) -> None:

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

        # Batch size of parameters in the previous forward pass
        self._param_batch_size = 1

    def forward(self, data: torch.Tensor, node_mars: torch.Tensor, params: Optional[Dict] = None, missing_mask: Optional[torch.Tensor]=None,  skip_logsumexp: bool = False):
        """
        data: [num_vars, B]
        node_mars: [num_nodes, B]
        """
        super(CategoricalLayer, self).forward(params is not None)

        if params is None:
            params = self.params
        else:
            params = params["params"]

        assert params.dim() == 1
        
        if skip_logsumexp:
            self._dense_forward_pass_nolog(data, node_mars, params)
        else:
            self._dense_forward_pass(data, node_mars, params, missing_mask=missing_mask)

        return None

    def backward(self, data: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor, params: Optional[Dict] = None):
        """
        data: [num_vars, B]
        node_flows: [num_nodes, B]
        node_mars: [num_nodes, B]
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
        if not self._used_external_params:
            with torch.no_grad():
                flows = self.param_flows
                self._normalize_parameters(flows, pseudocount = pseudocount)
                self.params.data = (1.0 - step_size) * self.params.data + step_size * flows

    def get_param_specs(self):
        return {"params": torch.Size([self.params.size(0)])}

    @torch.compile(mode = "reduce-overhead")
    def _dense_forward_pass(self, data: torch.Tensor, node_mars: torch.Tensor, params: torch.Tensor, missing_mask: Optional[torch.Tensor]=None):
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]
        param_idxs = data[self.vids] + self.psids.unsqueeze(1)
        if missing_mask is not None:
            not_missing_mask = ~missing_mask[self.vids]
            node_mars[sid:eid,:][not_missing_mask] = ((params[param_idxs][not_missing_mask]).clamp(min=1e-10)).log()
        else:
            node_mars[sid:eid,:] = ((params[param_idxs]).clamp(min=1e-10)).log()
        return None

    @torch.compile(mode = "reduce-overhead")
    def _dense_forward_pass_nolog(self, data: torch.Tensor, node_mars: torch.Tensor, params: torch.Tensor):
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]
        node_mars[sid:eid,:] = ((params[data[self.vids] + self.psids.unsqueeze(1)] + 1e-8).clamp(min=1e-10))

        return None

    def _normalize_parameters(self, params, pseudocount: float = 0.0):
        normalize_parameters(params, self.node_ids, self.node_nchs, pseudocount)

    @staticmethod
    @triton.jit
    def _cum_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, tot_num_params, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tot_num_params

        n_offsets = tl.load(node_ids_ptr + offsets, mask = mask, other = 0)

        params = tl.load(params_ptr + offsets, mask = mask, other = 0)

        tl.atomic_add(cum_params_ptr + n_offsets, params, mask = mask)

    @staticmethod
    @triton.jit
    def _norm_params_kernel(params_ptr, cum_params_ptr, node_ids_ptr, node_nchs_ptr, tot_num_params, 
                            pseudocount, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tot_num_params

        n_offsets = tl.load(node_ids_ptr + offsets, mask = mask, other = 0)

        params = tl.load(params_ptr + offsets, mask = mask, other = 0)
        cum_params = tl.load(cum_params_ptr + n_offsets, mask = mask, other = 1)
        nchs = tl.load(node_nchs_ptr + n_offsets, mask = mask, other = 1)

        normed_params = (params + pseudocount / nchs) / (cum_params + pseudocount)
        tl.store(params_ptr + offsets, normed_params, mask = mask)

    @staticmethod
    @triton.jit
    def _flows_kernel(param_flows_ptr, node_flows_ptr, param_ids_ptr, layer_num_nodes, tot_num_nodes, batch_size, node_offset, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_nodes * batch_size

        nf_offsets = batch_size * node_offset + offsets
        pr_offsets = tl.load(param_ids_ptr + offsets, mask = mask, other = 0)

        nflow = tl.load(node_flows_ptr + nf_offsets, mask = mask, other = 0)
        tl.atomic_add(param_flows_ptr + pr_offsets, nflow, mask = mask)

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
        n_start = 0
        for idx, rnode in enumerate(self.region_nodes):
            n_end = n_start + rnode.num_nodes

            if idx == 0:
                par_start = 0
                par_end = self.param_ends[n_end-1]

                param_ends = self.param_ends[0:n_end]
            else:
                par_start = self.param_ends[n_start-1]
                par_end = self.param_ends[n_end-1]

                param_ends = self.param_ends[n_start:n_end] - par_start

            rnode._params = {
                "params": self.params.data[par_start:par_end].detach().cpu().clone(),
                "param_ends": param_ends.detach().cpu().clone()
            }

            n_start = n_end

    @staticmethod
    def _prune_nodes(_params: Dict, node_keep_flag: torch.Tensor):
        kept_nodes = torch.where(node_keep_flag)[0]
        num_nodes = node_keep_flag.sum().item()

        old_n_params = torch.zeros_like(_params["param_ends"])
        old_n_params[0] = _params["param_ends"][0]
        old_n_params[1:] = _params["param_ends"][1:] - _params["param_ends"][:-1]

        param_ends = torch.cumsum(old_n_params[node_keep_flag], dim = 0)

        params = torch.zeros([param_ends[-1]], dtype = torch.float32)
        for idx in range(num_nodes):
            oidx = kept_nodes[idx]
            if oidx == 0:
                ops, ope = 0, _params["param_ends"][0]
            else:
                ops, ope = _params["param_ends"][oidx-1], _params["param_ends"][oidx]

            if idx == 0:
                ps, pe = 0, param_ends[0]
            else:
                ps, pe = param_ends[idx-1], param_ends[idx]

            params[ps:pe] = _params["params"][ops:ope]

        return {"params": params, "param_ends": param_ends}

    @staticmethod
    def _duplicate_nodes(_params: Dict):
        num_nodes = _params["param_ends"].size(0)
        num_params = _params["params"].size(0)

        params = torch.zeros([num_params * 2])
        par_start = 0
        for idx in range(num_nodes):
            if idx == 0:
                np = _params["param_ends"][0]
            else:
                np = _params["param_ends"][idx] - _params["param_ends"][idx-1]

            params[par_start*2:par_start*2+np] = _params["params"][par_start:par_start+np]
            params[par_start*2+np:par_start*2+np*2] = _params["params"][par_start:par_start+np]

            par_start += np

        param_ends = _params["param_ends"].view(-1, 1).repeat(1, 2).reshape(-1) * 2
        param_ends[2::2] -= _params["param_ends"][1:] - _params["param_ends"][:-1]
        param_ends[0] -= _params["param_ends"][0]

        return {"params": params, "param_ends": param_ends}