from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from copy import deepcopy
from typing import Sequence, Dict, Optional

from pyjuice.nodes import InputNodes
from pyjuice.nodes.distributions import Categorical
from pyjuice.layer.input_layer import InputLayer
from pyjuice.functional import normalize_parameters


class CategoricalLayer(InputLayer, nn.Module):
    
    def __init__(self, nodes: Sequence[InputNodes], cum_nodes: int = 0) -> None:
        nn.Module.__init__(self)

        # Reorder input nodes such that for any tied nodes, its source nodes appear before them
        nodes = self._reorder_nodes(nodes)
        InputLayer.__init__(self, nodes)

        # Parse input `nodes`
        self.vars = []
        self.node_sizes = []
        self.node_num_cats = []
        self.param_ends = []
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

            if not ns.is_tied():
                for nid in range(1, ns.num_nodes + 1):
                    self.param_ends.append(cum_params + ns.dist.num_cats * nid)
                cum_params += ns.num_nodes * ns.dist.num_cats
                ns._param_range = (cum_params - ns.num_nodes * ns.dist.num_cats, cum_params)
            else:
                source_ns = ns.get_source_ns()
                ns._param_range = deepcopy(source_ns._param_range)

        self._output_ind_range = (cum_nodes - layer_num_nodes, cum_nodes)
        self.param_ends = torch.tensor(self.param_ends)

        self.num_params = cum_params
        self.num_nodes = layer_num_nodes

        # Construct layer
        vids = torch.empty([self.num_nodes], dtype = torch.long)
        psids = torch.empty([self.num_nodes], dtype = torch.long)
        node_ids = torch.empty([self.num_params], dtype = torch.long)
        node_nchs = torch.empty([self.num_nodes], dtype = torch.long)
        n_start, pn_start, p_start = 0, 0, 0
        ns2pnrange = dict()
        for idx, ns in enumerate(self.nodes):
            n_end = n_start + ns.num_nodes
            vids[n_start:n_end] = self.vars[idx]

            node_nchs[n_start:n_end] = ns.dist.num_cats

            if not ns.is_tied():
                pn_end = pn_start + ns.num_nodes
                p_end = p_start + ns.num_nodes * ns.dist.num_cats

                node_ids[p_start:p_end] = torch.arange(pn_start, pn_end)[:,None].repeat(1, ns.dist.num_cats).reshape(-1)

                if idx == 0:
                    psids[0] = 0
                    psids[1:n_end] = self.param_ends[0:pn_end-1].clone().detach()
                    ns2pnrange[ns] = (-1, pn_end)
                else:
                    psids[n_start:n_end] = self.param_ends[pn_start-1:pn_end-1].clone().detach()
                    ns2pnrange[ns] = (pn_start, pn_end)

                pn_start = pn_end
                p_start = p_end
            else:
                source_ns = ns.get_source_ns()
                pns, pne = ns2pnrange[source_ns]
                if pns == -1:
                    psids[n_start] = 0
                    psids[n_start+1:n_end] = self.param_ends[0:pne-1].clone().detach()
                else:
                    psids[n_start:n_end] = self.param_ends[pns-1:pne-1].clone().detach()

            n_start = n_end

        self.register_buffer("vids", vids)
        self.register_buffer("psids", psids)
        self.register_buffer("node_ids", node_ids)
        self.register_buffer("node_nchs", node_nchs)

        # Initialize parameters
        self._init_params()

        self.param_flows_size = self.params.size(0)

        # Batch size of parameters in the previous forward pass
        self._param_batch_size = 1

    def forward(self, data: torch.Tensor, node_mars: torch.Tensor, 
                params: Optional[Dict] = None, 
                missing_mask: Optional[torch.Tensor] = None):
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
        
        if not self.provided("fw_local_ids"):
            # Evaluate the whole layer
            self._forward(data, node_mars, params, missing_mask = missing_mask)

        else:
            # Partial evaluation
            self._forward(data, node_mars, params, missing_mask = missing_mask, local_ids = self.fw_local_ids)

        return None

    def backward(self, data: torch.Tensor, node_flows: torch.Tensor, 
                 node_mars: torch.Tensor, params: Optional[Dict] = None):
        """
        data: [num_vars, B]
        node_flows: [num_nodes, B]
        node_mars: [num_nodes, B]
        """

        if not self.provided("bk_local_ids"):
            # Evaluate the whole layer
            layer_num_nodes = self._output_ind_range[1] - self._output_ind_range[0]
            tot_num_nodes = node_flows.size(0)
            batch_size = node_flows.size(1)
            node_offset = self._output_ind_range[0]

            param_ids = data[self.vids] + self.psids.unsqueeze(1)
            
            grid = lambda meta: (triton.cdiv(layer_num_nodes * batch_size, meta['BLOCK_SIZE']),)
            
            self._flows_kernel[grid](self.param_flows, node_flows, param_ids, layer_num_nodes, tot_num_nodes, batch_size, node_offset, BLOCK_SIZE = 1024)

        else:
            # Partial evaluation
            local_ids = self.bk_local_ids
            layer_num_nodes = local_ids.size(0)
            tot_num_nodes = node_flows.size(0)
            batch_size = node_flows.size(1)
            node_offset = self._output_ind_range[0]

            param_ids = data[self.vids[local_ids]] + self.psids[local_ids].unsqueeze(1)

            grid = lambda meta: (triton.cdiv(layer_num_nodes * batch_size, meta["BLOCK_SIZE"]),)

            self._partial_flows_kernel[grid](self.param_flows, node_flows, param_ids, local_ids, layer_num_nodes, tot_num_nodes, batch_size, node_offset, BLOCK_SIZE = 1024)
        
        return None

    def sample(self, samples: torch.Tensor, node_flows: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, 
               params: Optional[torch.Tensor] = None):
        """
        samples:       [num_vars, B]
        missing_mask:  [num_vars, B] or [num_vars] or None
        node_flows:    [num_nodes, B]
        """
        sid, eid = self._output_ind_range
        num_nodes = eid - sid
        num_vars = self.vids.max().item() + 1
        num_cats = self.node_nchs.max().item()
        batch_size = node_flows.size(1)
        all_vars = torch.unique(self.vids)

        probs = torch.zeros([num_vars * num_cats * batch_size], dtype = torch.float32, device = node_flows.device)

        if params is None:
            params = self.params
        else:
            params = params["params"]

        # Get all node ids with non-zero flow
        nflow_xids, nflow_yids = torch.where(node_flows[sid:eid,:])
        nflow_xids += sid
        num_activ_nodes = nflow_xids.size(0)

        # Prepare and run the sample kernel
        grid = lambda meta: (triton.cdiv(num_activ_nodes * num_cats, meta['BLOCK_SIZE']),)
        self._sample_kernel[grid](
            probs, node_flows, params, nflow_xids, nflow_yids, self.vids, self.psids, self.node_nchs,
            sid, num_activ_nodes, num_cats, batch_size, BLOCK_SIZE = 2048
        )

        # Reshape the cumulated (unnormalized) probabilities
        probs = probs.reshape(num_vars, num_cats, batch_size)

        if missing_mask is None:
            dist = torch.distributions.Categorical(probs = probs.permute(0, 2, 1).clip(min = 1e-6))
            samples[all_vars,:] = dist.sample()[all_vars,:]

        elif missing_mask.dim() == 1:
            mask = torch.zeros([missing_mask.size(0)], dtype = torch.bool, device = missing_mask.device)
            mask[all_vars] = True
            missing_mask = missing_mask & mask
            dist = torch.distributions.Categorical(probs = probs[missing_mask,:,:].permute(0, 2, 1).clip(min = 1e-6))
            samples[missing_mask,:] = dist.sample()

        else:
            assert missing_mask.dim() == 2

            mask = torch.zeros([missing_mask.size(0), 1], dtype = torch.bool, device = missing_mask.device)
            mask[all_vars,:] = True
            missing_mask = missing_mask & mask

            dist = torch.distributions.Categorical(probs = probs.permute(0, 2, 1).clip(min = 1e-6))
            samples = samples.view(-1)
            missing_mask = missing_mask.view(-1)
            samples[missing_mask] = dist.sample()[missing_mask]
            samples = samples.reshape(num_vars, batch_size)

        return samples

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        if not self._used_external_params:
            # Normalize and update parameters
            with torch.no_grad():
                flows = self.param_flows
                self._normalize_parameters(flows, pseudocount = pseudocount)
                self.params.data = (1.0 - step_size) * self.params.data + step_size * flows

    def get_param_specs(self):
        return {"params": torch.Size([self.params.size(0)])}

    @torch.compile(mode = "default", fullgraph = False)
    def _forward(self, data: torch.Tensor, node_mars: torch.Tensor, params: torch.Tensor, 
                 missing_mask: Optional[torch.Tensor] = None, local_ids: Optional[torch.Tensor] = None):

        sid, eid = self._output_ind_range[0], self._output_ind_range[1]

        if local_ids is None:
            param_idxs = data[self.vids] + self.psids.unsqueeze(1)
            node_mars[sid:eid,:] = ((params[param_idxs]).clamp(min=1e-10)).log()
        else:
            param_idxs = data[self.vids[local_ids]] + self.psids[local_ids].unsqueeze(1)
            node_mars[local_ids+sid,:] = ((params[param_idxs]).clamp(min=1e-10)).log()

        if missing_mask is not None:
            if missing_mask.dim() == 1:
                mask = torch.where(missing_mask[self.vids])[0] + sid
                node_mars[mask,:] = 0.0
            elif missing_mask.dim() == 2:
                maskx, masky = torch.where(missing_mask[self.vids])
                maskx = maskx + sid
                node_mars[maskx,masky] = 0.0
            else:
                raise ValueError()

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

    @staticmethod
    @triton.jit
    def _partial_flows_kernel(param_flows_ptr, node_flows_ptr, param_ids_ptr, local_ids_ptr, layer_num_nodes, tot_num_nodes, batch_size, node_offset, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_nodes * batch_size

        batch_offsets = (offsets % batch_size)
        local_offsets = (offsets // batch_size)

        nf_offsets = tl.load(local_ids_ptr + local_offsets, mask = mask, other = 0) + node_offset
        pr_offsets = tl.load(param_ids_ptr + offsets, mask = mask, other = 0)

        nflow = tl.load(node_flows_ptr + nf_offsets, mask = mask, other = 0)
        tl.atomic_add(param_flows_ptr + pr_offsets, nflow, mask = mask)

    def _init_params(self, perturbation: float = 4.0):
        """
        Initialize the parameters with random values
        """
        params = torch.exp(torch.rand([self.num_params]) * -perturbation)
        
        pn_start = 0
        for idx, ns in enumerate(self.nodes):
            if not ns.is_tied():
                pn_end = pn_start + ns.num_nodes

                if ns.has_params():
                    if idx == 0:
                        par_start = 0
                        par_end = self.param_ends[pn_end-1]
                    else:
                        par_start = self.param_ends[pn_start-1]
                        par_end = self.param_ends[pn_end-1]

                    params[par_start:par_end] = ns._params.to(params.device)

                pn_start = pn_end

        self._normalize_parameters(params)
        self.params = nn.Parameter(params)

        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.params.requires_grad = False

    def update_parameters(self):
        n_start = 0
        for idx, ns in enumerate(self.nodes):
            if ns.is_tied():
                continue

            n_end = n_start + ns.num_nodes

            if idx == 0:
                par_start = 0
                par_end = self.param_ends[n_end-1]
            else:
                par_start = self.param_ends[n_start-1]
                par_end = self.param_ends[n_end-1]

            ns._params = self.params.data[par_start:par_end].detach().cpu().clone()

            n_start = n_end

    @staticmethod
    @triton.jit
    def _sample_kernel(probs_ptr, node_flows_ptr, params_ptr, nflow_xids_ptr, nflow_yids_ptr, vids_ptr, 
                       psids_ptr, node_nchs_ptr, sid: tl.constexpr, num_activ_nodes: tl.constexpr, 
                       num_cats: tl.constexpr, batch_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_activ_nodes * num_cats

        # Get node ID, category ID, and batch ID
        nf_offsets = offsets // num_cats
        cat_offsets = (offsets % num_cats)
        node_offsets = tl.load(nflow_xids_ptr + nf_offsets, mask = mask, other = 0)
        batch_offsets = tl.load(nflow_yids_ptr + nf_offsets, mask = mask, other = 0)

        local_n_offsets = node_offsets - sid

        # Number of chs for every node
        node_nch = tl.load(node_nchs_ptr + local_n_offsets, mask = mask, other = 0)
        mask = mask & (cat_offsets < node_nch)

        # Get variable ID
        vid = tl.load(vids_ptr + local_n_offsets, mask = mask, other = 0)

        # Get param
        psid = tl.load(psids_ptr + local_n_offsets, mask = mask, other = 0)
        param = tl.load(params_ptr + psid + cat_offsets, mask = mask, other = 0)

        # Get flow
        nflow_offsets = node_offsets * batch_size + batch_offsets
        nflow = tl.load(node_flows_ptr + nflow_offsets, mask = mask, other = 0)

        # Compute edge flow and add to output
        eflow = param * nflow

        o_offsets = vid * (num_cats * batch_size) + cat_offsets * batch_size + batch_offsets
        tl.atomic_add(probs_ptr + o_offsets, eflow, mask = mask)

    def _reorder_nodes(self, nodes):
        node_set = set(nodes)
        reordered_untied_nodes = []
        reordered_tied_nodes = []
        added_node_set = set()
        for ns in nodes:
            if ns in added_node_set:
                continue
            if not ns.is_tied():
                reordered_untied_nodes.append(ns)
                added_node_set.add(ns)
            else:
                source_ns = ns.get_source_ns()
                if source_ns in added_node_set:
                    reordered_tied_nodes.append(ns)
                    added_node_set.add(ns)
                elif source_ns in node_set:
                    reordered_untied_nodes.append(source_ns)
                    reordered_tied_nodes.append(ns)
                    added_node_set.add(ns)
                    added_node_set.add(source_ns)
                else:
                    raise ValueError("A tied `InputNodes` should be in the same layer with its source nodes.")

        reordered_nodes = reordered_untied_nodes + reordered_tied_nodes

        assert len(reordered_nodes) == len(nodes), "Total node length should not change after reordering."

        return reordered_nodes