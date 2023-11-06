from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Union, Callable
from functools import partial

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit
from pyjuice.nodes.methods import get_subsumed_scopes
from pyjuice.utils import BitSet
from .base import query


## Categorical layer ##

@triton.jit
def _soft_evi_categorical_fw_kernel(data_ptr, node_mars_ptr, params_ptr, vids_ptr, psids_ptr, node_nchs_ptr, local_ids,
                                    sid: tl.constexpr, num_nodes: tl.constexpr, num_cats: tl.constexpr, 
                                    batch_size: tl.constexpr, partial: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_nodes * batch_size

    # Get node ID and category ID
    ns_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    # Get number of children (categories)
    node_nch = tl.load(node_nchs_ptr + ns_offsets, mask = mask, other = 0)

    # Get variable ID
    vid = tl.load(vids_ptr + ns_offsets, mask = mask, other = 0)

    # Get param start ID
    psid = tl.load(psids_ptr + ns_offsets, mask = mask, other = 0)

    # Compute soft evidence per category
    node_vals = tl.zeros((BLOCK_SIZE,), tl.float32)
    for cat_id in range(num_cats):

        cmask = mask & (cat_id < node_nch)

        # Get data (soft evidence)
        data_offsets = vid * (num_cats * batch_size) + cat_id * batch_size + batch_offsets
        d_soft_evi = tl.load(data_ptr + data_offsets, mask = cmask, other = 0)

        # Get param
        param = tl.load(params_ptr + psid + cat_id, mask = cmask, other = 0)

        # Compute current likelihood and accumulate
        node_vals += d_soft_evi * param

    # Write back
    if not partial:
        tl.store(node_mars_ptr + offsets + (sid * batch_size), tl.log(node_vals), mask = mask)
    else:
        global_nid = tl.load(local_ids + ns_offsets, mask = mask, other = 0) + sid
        tl.store(node_mars_ptr + global_nid * batch_size + batch_offsets, tl.log(node_vals), mask = mask)


@torch.compile(mode = "default", fullgraph = False)
def _cat_forward_gpu(self, data: torch.Tensor, node_mars: torch.Tensor, params: torch.Tensor, 
                    missing_mask: Optional[torch.Tensor] = None, local_ids: Optional[torch.Tensor] = None):

    sid, eid = self._output_ind_range[0], self._output_ind_range[1]

    if local_ids is None:
        param_idxs = data[self.vids[:,0]] + self.s_pids.unsqueeze(1)
        node_mars[sid:eid,:] = ((params[param_idxs]).clamp(min=1e-10)).log()
    else:
        param_idxs = data[self.vids[local_ids,0]] + self.s_pids[local_ids].unsqueeze(1)
        node_mars[local_ids+sid,:] = ((params[param_idxs]).clamp(min=1e-10)).log()

    if missing_mask is not None:
        if missing_mask.dim() == 1:
            mask = torch.where(missing_mask[self.vids])[0] + sid
            node_mars[mask,:] = 0.0
        elif missing_mask.dim() == 2:
            maskx, masky = torch.where(missing_mask[self.vids[:,0]])
            maskx = maskx + sid
            node_mars[maskx,masky] = 0.0
        else:
            raise ValueError()

    return None

def _cat_forward(self, data: torch.Tensor, node_mars: torch.Tensor, params: torch.Tensor, 
                 missing_mask: Optional[torch.Tensor] = None, local_ids: Optional[torch.Tensor] = None):
    if self.device.type == "cuda":
        _cat_forward_gpu(self, data, node_mars, params, missing_mask = missing_mask, local_ids = local_ids)
        return None

    sid, eid = self._output_ind_range[0], self._output_ind_range[1]

    data = data.cpu()
    device = node_mars.device

    if local_ids is None:
        param_idxs = data[self.vids] + self.s_pids.unsqueeze(1)
        node_mars[sid:eid,:] = ((params[param_idxs]).clamp(min=1e-10)).log().to(device)
    else:
        param_idxs = data[self.vids[local_ids]] + self.s_pids[local_ids].unsqueeze(1)
        node_mars[local_ids+sid,:] = ((params[param_idxs]).clamp(min=1e-10)).log().to(device)

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



def _categorical_forward(layer, node_mars: torch.Tensor,
                         params: Optional[torch.Tensor] = None, 
                         missing_mask: Optional[torch.Tensor] = None, **kwargs):

    if params is None:
        params = layer.params
        params = params.clip(min = 1e-8)

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]
    if "tokens" in kwargs:
        data = kwargs["tokens"]
        assert data.dim() == 2 and data.dtype == torch.long
        data = data.permute(1, 0)

        if not layer.provided("fw_local_ids"):
            _cat_forward(layer, data, node_mars, params, missing_mask)
        else:
            _cat_forward(layer, data, node_mars, params, missing_mask, local_ids = layer.fw_local_ids)

    elif "soft_evidence" in kwargs:

        data = kwargs["soft_evidence"]
        assert data.dim() == 3 and data.dtype == torch.float32 and data.min() >= 0.0 and data.max() <= 1.0
        
        if missing_mask is not None:
            data = data.flatten(0, 1)
            data[missing_mask.permute(1, 0).flatten(),:] = 1.0
            data = data.reshape(missing_mask.size(1), missing_mask.size(0), -1).permute(1, 2, 0)
        else:
            data = data.permute(1, 2, 0)

        num_cats = data.size(1)
        batch_size = data.size(2)
        
        if not layer.provided("fw_local_ids"):
            num_nodes = eid - sid

            node_nchs = layer.metadata[layer.s_mids]

            grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)

            _soft_evi_categorical_fw_kernel[grid](
                data.reshape(-1).contiguous(), node_mars, params, layer.vids.reshape(-1), layer.s_pids, node_nchs,
                None, sid, num_nodes, num_cats, batch_size, partial = False, BLOCK_SIZE = 512
            )

        else:
            local_ids = layer.fw_local_ids
            num_nodes = local_ids.size(0)

            vids = layer.vids[local_ids,0]
            s_pids = layer.s_pids[local_ids]
            node_nchs = layer.metadata[layer.s_mids[local_ids]]
            
            grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)

            _soft_evi_categorical_fw_kernel[grid](
                data.reshape(-1).contiguous(), node_mars, params, vids, s_pids, node_nchs,
                local_ids, sid, num_nodes, num_cats, batch_size, partial = True, BLOCK_SIZE = 512
            )

        node_mars[sid:eid,:] = node_mars[sid:eid,:].clip(max = 0.0)

    else:
        raise NotImplementedError("Unknown method to compute the forward pass for `Categorical`.")

    return None


@triton.jit
def _categorical_backward_kernel(cat_probs_ptr, node_flows_ptr, local_ids_ptr, rev_vars_mapping_ptr, vids_ptr, psids_ptr, 
                                 node_nchs_ptr, params_ptr, sid, eid, num_target_nodes, batch_size: tl.constexpr, 
                                 num_cats: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < num_target_nodes * batch_size)

    # Get node offsets and batch offsets
    local_offsets = (offsets // batch_size)
    local_node_offsets = tl.load(local_ids_ptr + local_offsets, mask = mask, other = 0)
    batch_offsets = (offsets % batch_size)

    global_node_offsets = local_node_offsets + sid

    # Get variable ID
    origin_vid = tl.load(vids_ptr + local_node_offsets, mask = mask, other = 0)
    vid = tl.load(rev_vars_mapping_ptr + origin_vid, mask = mask, other = 0)

    # Get number of children per node
    node_nch = tl.load(node_nchs_ptr + local_node_offsets, mask = mask, other = 0)

    # Get param start ID
    psid = tl.load(psids_ptr + local_node_offsets, mask = mask, other = 0)

    # Get flow
    nflow_offsets = global_node_offsets * batch_size + batch_offsets
    nflow = tl.load(node_flows_ptr + nflow_offsets, mask = mask, other = 0)

    # Compute edge flows and accumulate
    for cat_id in range(num_cats):
        cmask = mask & (cat_id < node_nch)

        param = tl.load(params_ptr + psid + cat_id, mask = cmask, other = 0)
        eflow = nflow * param

        p_offsets = vid * num_cats * batch_size + cat_id * batch_size + batch_offsets
        tl.atomic_add(cat_probs_ptr + p_offsets, eflow, mask = cmask)


def _categorical_backward(layer, node_flows: torch.Tensor, node_mars: torch.Tensor,
                          params: Optional[torch.Tensor] = None, 
                          mode: str = "full_distribution", **kwargs):

    if params is None:
        params = layer.params

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]
    if mode == "full_distribution":

        num_nodes = eid - sid
        num_vars = layer.vids.max().item() + 1
        num_cats = int(layer.metadata[layer.s_mids].max().item())
        batch_size = node_flows.size(1)

        if "target_vars" in kwargs and kwargs["target_vars"] is not None:
            target_vars = kwargs["target_vars"]

            rev_vars_mapping = torch.zeros([num_vars], dtype = torch.long)
            for i, var in enumerate(target_vars):
                rev_vars_mapping[var] = i
            rev_vars_mapping = rev_vars_mapping.to(node_flows.device)
        else:
            target_vars = [var for var in range(num_vars)]

            rev_vars_mapping = torch.arange(0, num_vars, device = node_flows.device)

        num_target_vars = len(target_vars)

        cat_probs = torch.zeros([num_target_vars * num_cats * batch_size], dtype = torch.float32, device = node_flows.device)

        local_ids = layer.enable_partial_evaluation(bk_scopes = target_vars, return_ids = True).to(node_flows.device)
        num_target_nodes = local_ids.size(0)

        node_nchs = layer.metadata[layer.s_mids]

        grid = lambda meta: (triton.cdiv(num_target_nodes * batch_size, meta['BLOCK_SIZE']),)
        _categorical_backward_kernel[grid](
            cat_probs, node_flows, local_ids, rev_vars_mapping, layer.vids, layer.s_pids, node_nchs, layer.params,
            sid, eid, num_target_nodes, batch_size, num_cats, BLOCK_SIZE = 512
        )

        cat_probs = cat_probs.reshape(num_target_vars, num_cats, batch_size)

        cat_probs /= cat_probs.sum(dim = 1, keepdim = True)
        cat_probs = cat_probs.permute(2, 0, 1)

    else:
        raise ValueError(f"Unknown mode {mode}.")

    return cat_probs


## General API ##


def _conditional_fw_input_fn(layer, inputs, node_mars, params, **kwargs):
    if layer.dist_signature == "Categorical":
        _categorical_forward(layer, node_mars, params, **kwargs)

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)}.")


def _conditional_bk_input_fn(layer, inputs, node_flows, node_mars, params = None, outputs = None, **kwargs):
    if layer.dist_signature == "Categorical":
        outputs.append(
            _categorical_backward(layer, node_flows, node_mars, params, **kwargs)
        )

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)}.")


def conditional(pc: TensorCircuit, target_vars: Optional[Sequence[int]] = None,
                missing_mask: Optional[torch.Tensor] = None,
                fw_input_fn: Optional[Union[str,Callable]] = None, 
                bk_input_fn: Optional[Union[str,Callable]] = None, 
                fw_delta_vars: Optional[Sequence[int]] = None,
                fw_scopes: Optional[Sequence[BitSet]] = None,
                bk_scopes: Optional[Sequence[BitSet]] = None,
                overwrite_partial_eval: bool = True,
                cache: Optional[dict] = None, **kwargs):

    if missing_mask is not None:
        missing_mask = missing_mask.permute(1, 0)
        B = missing_mask.size(1)
    elif "soft_evidence" in kwargs:
        B = kwargs["soft_evidence"].size(0)
    else:
        raise ValueError("Either `missing_mask` or `soft_evidence` should be provided.")

    outputs = []

    _wrapped_bk_input_fn = partial(_conditional_bk_input_fn, outputs = outputs)

    kwargs["target_vars"] = target_vars

    if cache is None:
        query(pc, inputs = torch.zeros([B, 1]), run_backward = True, 
            fw_input_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn, 
            bk_input_fn = _wrapped_bk_input_fn if bk_input_fn is None else bk_input_fn, 
            missing_mask = missing_mask, **kwargs)

        return outputs[0]

    else:
        if fw_delta_vars is not None and fw_scopes is None:
            fw_scopes = get_subsumed_scopes(pc, fw_delta_vars, type = "any")
        
        if fw_scopes is not None and overwrite_partial_eval:
            pc.enable_partial_evaluation(scopes = fw_scopes, forward = True)

        if target_vars is not None and bk_scopes is None:
            bk_scopes = get_subsumed_scopes(pc, target_vars, type = "any")
        
        if bk_scopes is not None and overwrite_partial_eval:
            pc.enable_partial_evaluation(scopes = bk_scopes, backward = True)

        kwargs["missing_mask"] = missing_mask

        # Forward
        lls, cache = pc.forward(
            inputs = torch.zeros([B, 1]), 
            input_layer_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn, 
            cache = cache, return_cache = True, **kwargs
        )

        # Backward
        cache = pc.backward(
            input_layer_fn = _wrapped_bk_input_fn if bk_input_fn is None else bk_input_fn, 
            compute_param_flows = False, cache = cache, return_cache = True, **kwargs
        )

        # pc.disable_partial_evaluation(forward = True, backward = True)

        return outputs[0], cache
