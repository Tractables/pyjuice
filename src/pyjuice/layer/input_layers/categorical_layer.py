from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Sequence, Dict, Optional

from pyjuice.nodes import InputNodes
from pyjuice.nodes.distributions import Categorical
from pyjuice.layer.input_layer import InputLayer
from pyjuice.functional import normalize_parameters, tie_param_flows

# Try to enable tensor cores
torch.set_float32_matmul_precision('high')


class CategoricalLayer(InputLayer, nn.Module):
    
    def __init__(self, nodes: Sequence[InputNodes], cum_nodes: int = 0) -> None:
        nn.Module.__init__(self)
        InputLayer.__init__(self, nodes)

        # Parse input `nodes`
        self.vars = []
        self.node_sizes = []
        self.node_num_cats = []
        self.param_ends = []
        self.tied_param_ids = []
        self.tied_param_group_ids = []
        self.num_tied_params = 0
        layer_num_nodes = 0
        cum_params = 0
        for ns in self.nodes:
            assert len(ns.scope) == 1, "CategoricalLayer only support uni-variate categorical distributions."
            assert isinstance(ns.dist, Categorical), f"Adding a `{type(ns.dist)}` node to a `CategoricalLayer`."

            self.vars.append(next(iter(ns.scope)))
            self.node_sizes.append(ns.num_nodes)
            self.node_num_cats.append(ns.dist.num_cats)

            ns._output_ind_range = (cum_nodes, cum_nodes + ns.num_nodes)
            cum_nodes += ns.num_nodes
            layer_num_nodes += ns.num_nodes

            for nid in range(1, ns.num_nodes + 1):
                self.param_ends.append(cum_params + ns.dist.num_cats * nid)
            cum_params += ns.num_nodes * ns.dist.num_cats
            ns._param_range = (cum_params - ns.num_nodes * ns.dist.num_cats, cum_params)

            if ns._source_node is not None:
                source_ns = ns._source_node
                if source_ns._tied_param_group_ids is None:

                    num_source_params = source_ns._param_range[1] - source_ns._param_range[0] + 1
                    source_ns._tied_param_group_ids = [i for i in range(self.num_tied_params, self.num_tied_params + num_source_params)]

                    tied_param_ids.extend([i for i in range(source_ns._param_range[0], source_ns._param_range[1])])
                    tied_param_group_ids.extend(source_ns._tied_param_group_ids)

                    self.num_tied_params += ns_param_end - ns_param_start + 1

                ns._tied_param_group_ids = deepcopy(source_ns._tied_param_group_ids)
                self.tied_param_ids.extend([i for i in range(ns._param_range[0], ns._param_range[1])])
                self.tied_param_group_ids.extend(ns._tied_param_group_ids)

        self._output_ind_range = (cum_nodes - layer_num_nodes, cum_nodes)
        self.param_ends = torch.tensor(self.param_ends)

        if self.num_tied_params > 0:
            self.tied_param_ids = torch.tensor(tied_param_ids)
            self.tied_param_group_ids = torch.tensor(tied_param_group_ids)

        self.num_params = cum_params
        self.num_nodes = layer_num_nodes

        # Construct layered
        vids = torch.empty([self.num_nodes], dtype = torch.long)
        psids = torch.empty([self.num_nodes], dtype = torch.long)
        n_start = 0
        for idx, ns in enumerate(self.nodes):
            n_end = n_start + ns.num_nodes
            vids[n_start:n_end] = self.vars[idx]

            if idx == 0:
                psids[0] = 0
                psids[1:n_end] = self.param_ends[0:n_end-1].clone().detach()
            else:
                psids[n_start:n_end] = self.param_ends[n_start-1:n_end-1].clone().detach()

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

    def forward(self, data: torch.Tensor, node_mars: torch.Tensor, 
                params: Optional[Dict] = None, 
                missing_mask: Optional[torch.Tensor] = None,  
                alphas:Optional[torch.Torch] = None,
                skip_logsumexp: bool = False):
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
            self._dense_forward_pass(data, node_mars, params, missing_mask=missing_mask, alphas=alphas)

        return None

    def backward(self, data: torch.Tensor, node_flows: torch.Tensor, 
                 node_mars: torch.Tensor, params: Optional[Dict] = None):
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
    
    def sample(self, samples: torch.Tensor, missing_mask:torch.tensor, node_flows: torch.tensor):
        """
        samples:       [num_vars, B]
        missing_mask:  [num_vars, B]
        node_flows:    [num_nodes, B]    
        
         - Note: it does not return anything, will update the samples in-place
         - node_flows[sid:eid].size() == (num_input_nodes, B)
        """
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]
        # print(f"Input layer w/+ flows (should be {samples.size(0)})", node_flows[sid:eid, :].sum(dim=0))   
    
        node_idxs, batch_idxs = node_flows[sid:eid, :].nonzero(as_tuple=True)   # (num_vars * B, 2)
        var_idxs = self.vids[node_idxs]                                         # (num_vars * B) 
        
        sampled_node_idxs = torch.zeros(samples.shape, dtype=torch.long, device=samples.device)
        sampled_node_idxs[var_idxs, batch_idxs] = node_idxs

        # TODO fix later, for now assume constant number of cats 
        # need a custome kernel probably
        ps = self.params.view(-1, self.psids[1])                        # (num_nodes, num_cats)
        cummulp = ps.cumsum(-1)
        replace_cummul_ps = cummulp[sampled_node_idxs]                 # (var, B, num_cats)
        rands = torch.rand(samples.size(0), samples.size(1), 1, device=samples.device)  # (vars, B, 1)
    
        sampled_cats = torch.sum(rands > replace_cummul_ps, dim=2).to(samples.dtype)
        samples[missing_mask] = sampled_cats[missing_mask]

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        if not self._used_external_params:
            # Tie parameter flows if necessary
            if self.num_tied_params > 0:
                with torch.no_grad():
                    self._tie_param_flows(self.param_flows)

            # Normalize and update parameters
            with torch.no_grad():
                flows = self.param_flows
                self._normalize_parameters(flows, pseudocount = pseudocount)
                self.params.data = (1.0 - step_size) * self.params.data + step_size * flows

    def _tie_param_flows(self, param_flows):
        if param_flows is not None:
            tie_param_flows(
                param_flows = param_flows, 
                num_tied_params = self.num_tied_params, 
                tied_param_ids = self.tied_param_ids, 
                tied_param_group_ids = self.tied_param_group_ids
            )

    def get_param_specs(self):
        return {"params": torch.Size([self.params.size(0)])}

    @torch.compile(mode = "default")
    def _dense_forward_pass(self, data: torch.Tensor, node_mars: torch.Tensor,
                                                            params: torch.Tensor, 
                                                            missing_mask: Optional[torch.Tensor]=None, 
                                                            alphas:Optional[torch.Tensor]=None):
        
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]
        param_idxs = data[self.vids] + self.psids.unsqueeze(1)
        if missing_mask is not None:
            mask = missing_mask[self.vids]
            node_mars[sid:eid,:][~mask] = ((params[param_idxs][~mask]).clamp(min=1e-10)).log()
            node_mars[sid:eid,:][mask] = 0.0
        else:
            node_mars[sid:eid,:] = ((params[param_idxs]).clamp(min=1e-10)).log()            

        if alphas is not None:
            node_mars[sid:eid,:] = ((node_mars[sid:eid,:].exp() * alphas[self.vids]) + (1 - alphas[self.vids])).log()

        return None

    @torch.compile(mode = "default")
    def _dense_forward_pass_nolog(self, data: torch.Tensor, node_mars: torch.Tensor, params: torch.Tensor):
        sid, eid = self._output_ind_range[0], self._output_ind_range[1]
        node_mars[sid:eid,:] = ((params[data[self.vids] + self.psids.unsqueeze(1)] + 1e-8).clamp(min=1e-10))

        return None

    def _normalize_parameters(self, params, pseudocount: float = 0.0):
        normalize_parameters(params, self.node_ids, self.node_nchs, pseudocount)

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
        for idx, ns in enumerate(self.nodes):
            n_end = n_start + ns.num_nodes

            if hasattr(ns, "_params"):
                if idx == 0:
                    par_start = 0
                    par_end = self.param_ends[n_end-1]
                else:
                    par_start = self.param_ends[n_start-1]
                    par_end = self.param_ends[n_end-1]

                params[par_start:par_end] = ns._params["params"].to(params.device)

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