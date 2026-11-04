from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Optional

from .common import emission_sums, layer_facts, runs as _runs, target_weights, targets


# From this many categories (from `MATMUL_MIN_CATS_ONE_TARGET` for a query of one target) the conditional's input-layer
# backward takes `emission_sums` rather than `_categorical_backward_kernel`, which loops over the categories in every
# program and adds atomically per (node, sample, category), every target's outputs taking one add per node of it.
# MEASURED (the input-layer step alone, both arms without their host syncs; tied and untied HMM emissions, 16 / 128 /
# 1024 nodes per target, B = 1 / 16 / 128, 1 or 32 targets): from 256 categories the matmuls win every configuration
# (0.03-0.81x of the kernel); from 128, every one of one target (0.21-0.94x), while with 32 targets they lose up to
# 1.39x; at 64 and below the kernel wins or ties wherever there are several targets.
MATMUL_MIN_CATS = 256
MATMUL_MIN_CATS_ONE_TARGET = 128

# `forward` takes `node_runs` (see `queries.distributions`)
NODE_RUNS = True


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


def forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
            missing_mask: Optional[torch.Tensor] = None, node_runs = None, **kwargs):

    batch_size, num_vars = inputs.size(0), inputs.size(1)

    if inputs.dim() == 2:
        # Hard evidence
        assert inputs.dtype == torch.long

        inputs = inputs.permute(1, 0).contiguous()

        layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask, node_runs = node_runs)

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
        soft = inputs.reshape(-1).contiguous()
        vids = layer.vids.reshape(-1)

        # Every run of nodes (`node_runs`) through views that start at it, as `InputLayer._forward_runs` does
        for start, n in _runs(node_runs, num_nodes):
            grid = lambda meta: (triton.cdiv(n * batch_size, meta['BLOCK_SIZE']),)
            _soft_evi_categorical_fw_kernel[grid](
                soft, node_mars[start:], layer.params, vids[start:], layer.s_pids[start:], node_nchs[start:],
                sid, n, num_cats, batch_size, BLOCK_SIZE = 512
            )
            node_mars[sid + start:sid + start + n,:] = node_mars[sid + start:sid + start + n,:].clip(max = 0.0)

    else:
        raise NotImplementedError("Unknown method to compute the forward pass for `Categorical`.")

    return None


@triton.jit
def _categorical_backward_kernel(cat_probs_ptr, node_flows_ptr, node_ids_ptr, rev_vars_mapping_ptr, vids_ptr, psids_ptr,
                                 node_nchs_ptr, params_ptr, sid, eid, num_target_nodes, batch_size: tl.constexpr,
                                 num_cats: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < num_target_nodes * batch_size)

    # Get node offsets and batch offsets
    local_offsets = (offsets // batch_size)
    local_node_offsets = tl.load(node_ids_ptr + local_offsets, mask = mask, other = 0)
    batch_offsets = (offsets % batch_size)

    global_node_offsets = local_node_offsets + sid

    # Get variable ID
    origin_vid = tl.load(vids_ptr + local_node_offsets, mask = mask, other = 0)
    vid = tl.load(rev_vars_mapping_ptr + origin_vid, mask = mask, other = 0)

    # Get number of children per node
    node_nch = tl.load(node_nchs_ptr + local_node_offsets, mask = mask, other = 0)

    # Get param start ID
    psid = tl.load(psids_ptr + local_node_offsets, mask = mask, other = 0)

    # Get flow (log-space flows, see `backward`)
    nflow_offsets = global_node_offsets * batch_size + batch_offsets
    nflow = tl.exp(tl.load(node_flows_ptr + nflow_offsets, mask = mask, other = -float("inf")))

    # Compute edge flows and accumulate
    for cat_id in range(num_cats):
        cmask = mask & (cat_id < node_nch)

        param = tl.load(params_ptr + psid + cat_id, mask = cmask, other = 0)
        eflow = nflow * param

        p_offsets = vid * num_cats * batch_size + cat_id * batch_size + batch_offsets
        tl.atomic_add(cat_probs_ptr + p_offsets, eflow, mask = cmask)


def _facts(layer):
    """``(num_cats, uniform)`` of ``layer``: the most categories any of its nodes has, and whether every node has that
    many -- kept with its :class:`~.common.LayerFacts`."""
    f = layer_facts(layer)
    c = f.extra.get("categorical")
    if c is None:
        num_cats = int(layer.metadata[layer.s_mids].max().item())
        c = f.extra["categorical"] = (num_cats, all(ns.dist.num_cats == num_cats for ns in layer.nodes))
    return c


def backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
             params: Optional[torch.Tensor] = None, **kwargs):

    assert kwargs.get("logspace_flows", False), "The conditional backward reads log-space flows (`logspace_flows = True`)."

    if params is None:
        params = layer.params

    sid, eid = layer._output_ind_range[0], layer._output_ind_range[1]

    num_cats, uniform = _facts(layer)
    batch_size = node_flows.size(1)

    t = targets(layer, kwargs.get("target_vars"), node_flows.device)
    num_target_nodes = t.nodes.size(0)

    # Large vocabularies: sum_n F(n) p_n(:) as matmuls with the emission table (`emission_sums`) instead of the kernel's
    # atomic add per (node, sample, category). Needs every node of the layer to have the same `num_cats`.
    if uniform and (num_cats >= MATMUL_MIN_CATS or (t.num_targets == 1 and num_cats >= MATMUL_MIN_CATS_ONE_TARGET)):
        weights = target_weights(node_flows[sid + t.nodes], t.rows_of, t.num_targets)
        s_pids, rows_of = t.host if t.host is not None else (layer.s_pids[t.nodes], t.rows_of)
        sums = emission_sums(layer.params, s_pids, rows_of, weights, t.num_targets, num_cats)
        return sums / (sums.sum(dim = 2, keepdim = True) + 1e-12)

    cat_probs = torch.zeros([t.num_targets * num_cats * batch_size], dtype = torch.float32, device = node_flows.device)

    node_nchs = layer.metadata[layer.s_mids]

    grid = lambda meta: (triton.cdiv(num_target_nodes * batch_size, meta['BLOCK_SIZE']),)

    _categorical_backward_kernel[grid](
        cat_probs, node_flows, t.nodes, t.rev_vars_mapping, layer.vids, layer.s_pids, node_nchs, layer.params,
        sid, eid, num_target_nodes, batch_size, num_cats, BLOCK_SIZE = 512
    )

    cat_probs = cat_probs.reshape(t.num_targets, num_cats, batch_size)

    cat_probs /= (cat_probs.sum(dim = 1, keepdim = True) + 1e-12)
    cat_probs = cat_probs.permute(2, 0, 1)

    return cat_probs
