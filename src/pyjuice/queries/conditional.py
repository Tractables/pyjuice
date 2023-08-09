from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Union, Callable
from functools import partial

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit
from pyjuice.layer.input_layers import CategoricalLayer
from .base import query


## Categorical layer ##

@triton.jit
def _soft_evi_categorical_fw_kernel(data_ptr, node_mars_ptr, params_ptr, vids_ptr, psids_ptr, node_nchs_ptr,
                                    sid: tl.constexpr, num_nodes: tl.constexpr, num_cats: tl.constexpr, 
                                    batch_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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
    tl.store(node_mars_ptr + offsets + (sid * batch_size), tl.log(node_vals), mask = mask)


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

        param_idxs = data[layer.vids] + layer.psids.unsqueeze(1)

        if missing_mask is not None:
            mask = missing_mask[layer.vids]
            node_mars[sid:eid,:][~mask] = ((params[param_idxs][~mask]).clamp(min=1e-10)).log()
            node_mars[sid:eid,:][mask] = 0.0

    elif "soft_evidence" in kwargs:

        data = kwargs["soft_evidence"]
        assert data.dim() == 3 and data.dtype == torch.float32 and data.min() >= 0.0 and data.max() <= 1.0
        
        if missing_mask is not None:
            data = data.flatten(0, 1)
            data[missing_mask.permute(1, 0).flatten(),:] = 1.0
            data = data.reshape(missing_mask.size(1), missing_mask.size(0), -1).permute(1, 2, 0)
        else:
            data = data.permute(1, 2, 0)
        
        num_nodes = eid - sid
        num_cats = data.size(1)
        batch_size = data.size(2)

        grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)

        _soft_evi_categorical_fw_kernel[grid](
            data.reshape(-1).contiguous(), node_mars, params, layer.vids, layer.psids, layer.node_nchs,
            sid, num_nodes, num_cats, batch_size, BLOCK_SIZE = 512
        )

        node_mars[sid:eid,:] = node_mars[sid:eid,:].clip(max = 0.0)

    else:
        raise NotImplementedError("Unknown method to compute the forward pass for `CategoricalLayer`.")

    return None


@triton.jit
def _categorical_backward_kernel(cat_probs_ptr, node_flows_ptr, vids_ptr, psids_ptr, node_nchs_ptr, params_ptr,
                                 sid: tl.constexpr, eid: tl.constexpr,
                                 batch_size: tl.constexpr, num_cats: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE
    num_nodes = eid - sid

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < num_nodes * batch_size)

    # Get node offsets and batch offsets
    local_node_offsets = (offsets // batch_size)
    batch_offsets = (offsets % batch_size)

    global_node_offsets = local_node_offsets + sid

    # Get variable ID
    vid = tl.load(vids_ptr + local_node_offsets, mask = mask, other = 0)

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
                          missing_mask: Optional[torch.Tensor] = None, 
                          mode: str = "full_distribution", **kwargs):

    mask = missing_mask[layer.vids]

    if params is None:
        params = layer.params

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]
    if mode == "full_distribution":

        num_nodes = eid - sid
        num_vars = layer.vids.max().item() + 1
        num_cats = layer.node_nchs.max().item()
        batch_size = node_flows.size(1)

        cat_probs = torch.zeros([num_vars * num_cats * batch_size], dtype = torch.float32, device = node_flows.device)

        grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)
        _categorical_backward_kernel[grid](
            cat_probs, node_flows, layer.vids, layer.psids, layer.node_nchs, layer.params,
            sid, eid, batch_size, num_cats, BLOCK_SIZE = 512
        )

        cat_probs = cat_probs.reshape(num_vars, num_cats, batch_size)

        cat_probs /= cat_probs.sum(dim = 1, keepdim = True)
        cat_probs = cat_probs.permute(2, 0, 1)

    else:
        raise ValueError(f"Unknown mode {mode}.")

    return cat_probs


## General API ##


def _conditional_fw_input_fn(layer, inputs, node_mars, params, **kwargs):
    if isinstance(layer, CategoricalLayer):
        _categorical_forward(layer, node_mars, params, **kwargs)

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)}.")


def _conditional_bk_input_fn(layer, inputs, node_flows, node_mars, params = None, outputs = None, **kwargs):
    if isinstance(layer, CategoricalLayer):
        assert "missing_mask" in kwargs
        outputs.append(
            _categorical_backward(layer, node_flows, node_mars, params, **kwargs)
        )

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)}.")


def conditional(pc: TensorCircuit, missing_mask: torch.Tensor,
                fw_input_fn: Optional[Union[str,Callable]] = None, 
                bk_input_fn: Optional[Union[str,Callable]] = None, **kwargs):

    missing_mask = missing_mask.permute(1, 0)

    outputs = []

    _wrapped_bk_input_fn = partial(_conditional_bk_input_fn, outputs = outputs)
    
    query(pc, inputs = torch.zeros([missing_mask.size(1), 1]), run_backward = True, 
          fw_input_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn, 
          bk_input_fn = _wrapped_bk_input_fn if bk_input_fn is None else bk_input_fn, 
          missing_mask = missing_mask, **kwargs)

    return outputs[0]
