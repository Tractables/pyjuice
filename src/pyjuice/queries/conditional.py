from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Union, Callable, Optional, Sequence
from functools import partial

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit
from pyjuice.nodes.methods import get_subsumed_scopes
from pyjuice.utils import BitSet
from pyjuice.utils.kernel_launcher import FastJITFunction
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


def _categorical_forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
                         missing_mask: Optional[torch.Tensor] = None, **kwargs):

    batch_size, num_vars = inputs.size(0), inputs.size(1)

    if inputs.dim() == 2:
        # Hard evidence
        assert inputs.dtype == torch.long

        inputs = inputs.permute(1, 0).contiguous()

        layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask)

    elif inputs.dim() == 3:
        # Soft evidence
        assert inputs.dtype == torch.float32 and inputs.min() >= 0.0 and inputs.max() <= 1.0

        if missing_mask is not None:
            if missing_mask.dim() == 1:
                inputs[:,missing_mask,:] = 1.0
            else:
                assert missing_mask.dim() == 2
                inputs = inputs.flatten(0, 1)
                inputs[missing_mask.flatten(),:] = 1.0
                inputs = inputs.reshape(batch_size, num_vars, -1)

        inputs = inputs.permute(1, 2, 0) # [num_vars, num_cats, B]
        num_cats = inputs.size(1)

        sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]
        num_nodes = eid - sid

        node_nchs = layer.metadata[layer.s_mids]

        grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)

        _soft_evi_categorical_fw_kernel[grid](
            inputs.reshape(-1).contiguous(), node_mars, layer.params, layer.vids.reshape(-1), layer.s_pids, node_nchs,
            None, sid, num_nodes, num_cats, batch_size, partial = False, BLOCK_SIZE = 512
        )

        node_mars[sid:eid,:] = node_mars[sid:eid,:].clip(max = 0.0)

    else:
        raise NotImplementedError("Unknown method to compute the forward pass for `Categorical`.")

    return None


@triton.jit
def _soft_evi_discrete_logistic_fw_kernel(data_ptr, node_mars_ptr, params_ptr, vids_ptr, psids_ptr, s_mids_ptr, metadata_ptr, 
                                          local_ids, sid: tl.constexpr, num_nodes: tl.constexpr, num_cats: tl.constexpr, 
                                          batch_size: tl.constexpr, partial: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_nodes * batch_size

    # Get node ID and category ID
    ns_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    # Get variable ID
    vid = tl.load(vids_ptr + ns_offsets, mask = mask, other = 0)

    # Get params
    psid = tl.load(psids_ptr + ns_offsets, mask = mask, other = 0)
    mu = tl.load(params_ptr + psid, mask = mask, other = 0)
    s = tl.load(params_ptr + psid + 1, mask = mask, other = 0)

    # Get metadata
    s_mids = tl.load(s_mids_ptr + ns_offsets, mask = mask, other = 0)
    range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
    range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
    node_nch = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

    # Compute soft evidence per category
    node_vals = tl.zeros((BLOCK_SIZE,), tl.float32)
    for cat_id in range(num_cats):

        cmask = mask & (cat_id < node_nch)

        # Get data (soft evidence)
        data_offsets = vid * (num_cats * batch_size) + cat_id * batch_size + batch_offsets
        d_soft_evi = tl.load(data_ptr + data_offsets, mask = cmask, other = 0)

        # Get param
        interval = (range_high - range_low) / node_nch
        vlow = cat_id * interval + range_low
        vhigh = vlow + interval

        cdfhigh = tl.where(cat_id == node_nch - 1, 1.0, 1.0 / (1.0 + tl.exp((mu - vhigh) / s)))
        cdflow = tl.where(cat_id == 0, 0.0, 1.0 / (1.0 + tl.exp((mu - vlow) / s)))

        param = tl.maximum(cdfhigh - cdflow, 0.0)

        # Compute current likelihood and accumulate
        node_vals += d_soft_evi * param

    # Write back
    if not partial:
        tl.store(node_mars_ptr + offsets + (sid * batch_size), tl.log(node_vals), mask = mask) # debug
    else:
        global_nid = tl.load(local_ids + ns_offsets, mask = mask, other = 0) + sid
        tl.store(node_mars_ptr + global_nid * batch_size + batch_offsets, tl.log(node_vals), mask = mask)


def _discrete_logistic_forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
                               missing_mask: Optional[torch.Tensor] = None, **kwargs):
    
    batch_size, num_vars = inputs.size(0), inputs.size(1)

    if inputs.dim() == 2:
        # Hard evidence
        if layer.nodes[0].dist.input_type == "discrete":
            assert inputs.dtype == torch.long, "Input dtype should be `torch.float32`."
        else: 
            assert layer.nodes[0].dist.input_type == "continuous"
            assert inputs.dtype == torch.float32, "Input dtype should be `torch.float32`."

        inputs = inputs.permute(1, 0).contiguous()

        layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask)

    elif inputs.dim() == 3:
        # Soft evidence
        assert inputs.dtype == torch.float32 and inputs.min() >= 0.0 and inputs.max() <= 1.0

        if missing_mask is not None:
            if missing_mask.dim() == 1:
                inputs[:,missing_mask,:] = 1.0
            else:
                assert missing_mask.dim() == 2
                inputs = inputs.flatten(0, 1)
                inputs[missing_mask.flatten(),:] = 1.0
                inputs = inputs.reshape(batch_size, num_vars, -1)

        inputs = inputs.permute(1, 2, 0) # [num_vars, num_cats, B]
        num_cats = inputs.size(1)

        sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]
        num_nodes = eid - sid

        grid = lambda meta: (triton.cdiv(num_nodes * batch_size, meta['BLOCK_SIZE']),)

        _soft_evi_discrete_logistic_fw_kernel[grid](
            inputs.reshape(-1).contiguous(), node_mars, layer.params, layer.vids.reshape(-1), layer.s_pids, 
            layer.s_mids, layer.metadata, None, sid, num_nodes, num_cats, batch_size, 
            partial = False, BLOCK_SIZE = 512
        )

        node_mars[sid:eid,:] = node_mars[sid:eid,:].clip(max = 0.0)

    else:
        raise NotImplementedError("Unknown method to compute the forward pass for `DiscreteLogistic`.")

    return None


@triton.jit
def _categorical_backward_kernel(cat_probs_ptr, node_flows_ptr, local_ids_ptr, rev_vars_mapping_ptr, vids_ptr, psids_ptr, 
                                 node_nchs_ptr, params_ptr, sid, eid, num_target_nodes, batch_size: tl.constexpr, 
                                 num_cats: tl.constexpr, partial_eval: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < num_target_nodes * batch_size)

    # Get node offsets and batch offsets
    local_offsets = (offsets // batch_size)
    if partial_eval == 1: 
        local_node_offsets = tl.load(local_ids_ptr + local_offsets, mask = mask, other = 0)
    else:
        local_node_offsets = local_offsets
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


def _categorical_backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
                          params: Optional[torch.Tensor] = None, **kwargs):

    if params is None:
        params = layer.params

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]

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

    if len(target_vars) < num_vars:
        local_ids = layer.enable_partial_evaluation(bk_scopes = target_vars, return_ids = True).to(node_flows.device)
        num_target_nodes = local_ids.size(0)
        partial_eval = 1
    else:
        local_ids = None
        num_target_nodes = eid - sid
        partial_eval = 0

    node_nchs = layer.metadata[layer.s_mids]

    grid = lambda meta: (triton.cdiv(num_target_nodes * batch_size, meta['BLOCK_SIZE']),)

    _categorical_backward_kernel[grid](
        cat_probs, node_flows, local_ids, rev_vars_mapping, layer.vids, layer.s_pids, node_nchs, layer.params,
        sid, eid, num_target_nodes, batch_size, num_cats, partial_eval = partial_eval, BLOCK_SIZE = 512
    )

    cat_probs = cat_probs.reshape(num_target_vars, num_cats, batch_size)

    cat_probs /= (cat_probs.sum(dim = 1, keepdim = True) + 1e-12)
    cat_probs = cat_probs.permute(2, 0, 1)

    return cat_probs


@triton.jit
def _discrete_logistic_backward_kernel(cat_probs_ptr, node_flows_ptr, local_ids_ptr, rev_vars_mapping_ptr, vids_ptr, psids_ptr, 
                                       msids_ptr, metadata_ptr, params_ptr, sid, eid, num_target_nodes, 
                                       batch_size: tl.constexpr, num_cats: tl.constexpr, partial_eval: tl.constexpr, 
                                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < num_target_nodes * batch_size)

    # Get node offsets and batch offsets
    local_offsets = (offsets // batch_size)
    if partial_eval == 1: 
        local_node_offsets = tl.load(local_ids_ptr + local_offsets, mask = mask, other = 0)
    else:
        local_node_offsets = local_offsets
    batch_offsets = (offsets % batch_size)

    global_node_offsets = local_node_offsets + sid

    # Get variable ID
    origin_vid = tl.load(vids_ptr + local_node_offsets, mask = mask, other = 0)
    vid = tl.load(rev_vars_mapping_ptr + origin_vid, mask = mask, other = 0)

    # Get params
    psid = tl.load(psids_ptr + local_offsets, mask = mask, other = 0)
    mu = tl.load(params_ptr + psid, mask = mask, other = 0)
    s = tl.load(params_ptr + psid + 1, mask = mask, other = 0)

    # Get metadata
    s_mids = tl.load(msids_ptr + local_offsets, mask = mask, other = 0)
    range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
    range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
    node_nch = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

    # Get flow
    nflow_offsets = global_node_offsets * batch_size + batch_offsets
    nflow = tl.load(node_flows_ptr + nflow_offsets, mask = mask, other = 0)

    # Compute edge flows and accumulate
    for cat_id in range(num_cats):
        cmask = mask & (cat_id < node_nch)

        # Get param
        interval = (range_high - range_low) / node_nch
        vlow = cat_id * interval + range_low
        vhigh = vlow + interval

        cdfhigh = tl.where(cat_id == node_nch - 1, 1.0, 1.0 / (1.0 + tl.exp((mu - vhigh) / s)))
        cdflow = tl.where(cat_id == 0, 0.0, 1.0 / (1.0 + tl.exp((mu - vlow) / s)))

        param = tl.maximum(cdfhigh - cdflow, 0.0)
        eflow = nflow * param

        p_offsets = vid * num_cats * batch_size + cat_id * batch_size + batch_offsets
        tl.atomic_add(cat_probs_ptr + p_offsets, eflow, mask = cmask)


def _discrete_logistic_backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
                                params: Optional[torch.Tensor] = None, **kwargs):
    
    if params is None:
        params = layer.params

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]

    num_nodes = eid - sid
    num_vars = layer.vids.max().item() + 1
    num_cats = int(layer.metadata[layer.s_mids + 2].max().item())
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

    if len(target_vars) < num_vars:
        local_ids = layer.enable_partial_evaluation(bk_scopes = target_vars, return_ids = True).to(node_flows.device)
        num_target_nodes = local_ids.size(0)
        partial_eval = 1
    else:
        local_ids = None
        num_target_nodes = eid - sid
        partial_eval = 0

    grid = lambda meta: (triton.cdiv(num_target_nodes * batch_size, meta['BLOCK_SIZE']),)

    _discrete_logistic_backward_kernel[grid](
        cat_probs, node_flows, local_ids, rev_vars_mapping, layer.vids, layer.s_pids, 
        layer.s_mids, layer.metadata, layer.params,
        sid, eid, num_target_nodes, batch_size, num_cats, partial_eval = partial_eval, BLOCK_SIZE = 512
    )

    cat_probs = cat_probs.reshape(num_target_vars, num_cats, batch_size)

    cat_probs /= (cat_probs.sum(dim = 1, keepdim = True) + 1e-12)
    cat_probs = cat_probs.permute(2, 0, 1)

    return cat_probs


def _conditional_fw_input_fn(layer, inputs, node_mars, **kwargs):
    if layer.dist_signature == "Categorical":
        _categorical_forward(layer, inputs, node_mars, **kwargs)

    elif layer.dist_signature == "DiscreteLogistic":
        _discrete_logistic_forward(layer, inputs, node_mars, **kwargs)

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)} for the forward pass. Please implement and provide your own `fw_input_fn`.")


def _conditional_bk_input_fn(layer, inputs, node_flows, node_mars, outputs = None, **kwargs):
    if layer.dist_signature == "Categorical":
        outputs.append(
            _categorical_backward(layer, inputs, node_flows, node_mars, layer.params, **kwargs)
        )

    elif layer.dist_signature == "DiscreteLogistic":
        outputs.append(
            _discrete_logistic_backward(layer, inputs, node_flows, node_mars, layer.params, **kwargs)
        )

    else:
        raise TypeError(f"Unknown/unsupported layer type {type(layer)} for the backward pass. Please implement and provide your own `bk_input_fn`.")


## Main API ##


def conditional(pc: TensorCircuit, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                target_vars: Optional[Sequence[int]] = None,
                fw_input_fn: Optional[Union[str,Callable]] = None, 
                bk_input_fn: Optional[Union[str,Callable]] = None, **kwargs):
    """
    Compute the conditional probability given hard or soft evidence, i.e., P(o|e).

    :param pc: the input PC
    :type pc: TensorCircuit

    :param data: data of size [B, num_vars] (hard evidence) or a custom shape paired with `fw_input_fn`
    :type data: torch.Tensor

    :param missing_mask: a boolean mask indicating marginalized variables; the size can be [num_vars] or [B, num_vars]
    :type missing_mask: torch.Tensor

    :param fw_input_fn: an optional custom function for the forward pass of input layers
    :type fw_input_fn: Optional[Union[str,Callable]]

    :param bk_input_fn: an optional custom function for the backward pass of input layers
    :type bk_input_fn: Optional[Union[str,Callable]]
    """

    outputs = []

    _wrapped_bk_input_fn = partial(_conditional_bk_input_fn, outputs = outputs)

    kwargs["target_vars"] = target_vars

    query(pc, inputs = data, run_backward = True, 
          fw_input_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn, 
          bk_input_fn = _wrapped_bk_input_fn if bk_input_fn is None else bk_input_fn, 
          missing_mask = missing_mask, **kwargs)

    return outputs[0]
