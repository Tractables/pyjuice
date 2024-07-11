from __future__ import annotations

import torch
import numpy as np
import triton
import triton.language as tl
import random
from typing import Union, Callable, Optional
from functools import partial
from numba import njit

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit


@njit
def _assign_cids_ind_target(ind_target, element_pointers, ind_b, num_samples):
    for i in range(ind_target.shape[0]):
        bid = ind_b[i]
        ind_t = element_pointers[bid]
        ind_target[i] = ind_t * num_samples + bid
        element_pointers[bid] = ind_t + 1


@triton.jit
def sample_sum_layer_kernel(nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples, 
                            ind_target, ind_n, ind_b, seed, block_size: tl.constexpr, batch_size: tl.constexpr, 
                            num_edges: tl.constexpr, num_samples: tl.constexpr, num_nblocks: tl.constexpr, BLOCK_S: tl.constexpr, 
                            BLOCK_M: tl.constexpr, TILE_SIZE_M: tl.constexpr, BLOCK_K: tl.constexpr, TILE_SIZE_K: tl.constexpr,
                            conditional: tl.constexpr):
    
    pid_s = tl.program_id(0) # ID of size-`BLOCK_S` batches

    # Sample offsets and mask
    offs_sample = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    mask_sample = offs_sample < num_samples

    # Load node and batch ids
    node_sample_id = tl.load(ind_n + offs_sample, mask = mask_sample, other = 0)
    batch_id = tl.load(ind_b + offs_sample, mask = mask_sample, other = 0)
    node_id = tl.load(node_samples + node_sample_id * batch_size)

    # Locate node ids in `nids`
    offs_nids = tl.arange(0, BLOCK_M)
    local_nids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    local_nid_offs = tl.zeros([BLOCK_S], dtype = tl.int64)
    for i in range(TILE_SIZE_M):
        mask_nids = offs_nids < num_nblocks

        ref_nid = tl.load(nids + offs_nids, mask = mask_nids, other = 0)
        is_match = (node_id[:,None] >= ref_nid[None,:]) & (node_id[:,None] < ref_nid[None,:] + block_size)

        match_local_id = tl.sum(is_match * (offs_nids[None,:] + 1), axis = 1)
        match_local_offset = tl.sum(is_match * (node_id[:,None] - ref_nid[None,:]), axis = 1)

        local_nids = tl.where(match_local_id > 0, match_local_id - 1, local_nids)
        local_nid_offs = tl.where(match_local_id > 0, match_local_offset, local_nid_offs)

        offs_nids += BLOCK_M

    # Update sample mask to filter out inactive ones
    mask_sample = mask_sample & (local_nids >= 0)

    # Sample random probabilities uniform between 0 and 1
    rnd_val = tl.rand(seed, tl.arange(0, BLOCK_S))

    # Offset for children
    offs_child = tl.arange(0, BLOCK_K)
    mask_child = offs_child < num_edges

    if conditional:
        nmars = tl.load(node_mars + node_id, mask = mask_sample, other = 0.0)

    # Main loop over blocks of child nodes
    chids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    for i in range(TILE_SIZE_K):

        # Load parameters
        param_id = tl.load(pids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
        epars = tl.load(params + param_id + local_nid_offs[None,:], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0)

        if conditional:
            # In this case, we use `param * cmar / nmar` as the "parameter"
            emars_id = tl.load(cids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
            emars = tl.load(params + emars_id, mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0)

            epars = epars * tl.exp(emars - nmars[None,:])
        
        cum_probs = tl.cumsum(epars, axis = 0) # [BLOCK_K, BLOCK_S]
        local_chids = tl.sum((rnd_val[None,:] >= cum_probs).to(tl.int64), axis = 0) # [BLOCK_S]

        is_overflow = (local_chids == BLOCK_K)
        rnd_val = tl.where(is_overflow, rnd_val - tl.sum(epars, axis = 0), rnd_val)

        chids = tl.where(is_overflow | (chids > -1), chids, local_chids + i * BLOCK_K)

        offs_child += BLOCK_K
        mask_child = offs_child < num_edges

    # Retrieve the global child ids and save them to `element_samples`
    global_chids = tl.load(cids + local_nids * num_edges + chids, mask = mask_sample, other = 0)
    target_id = tl.load(ind_target + offs_sample, mask = mask_sample, other = 0)

    tl.store(element_samples + target_id, global_chids, mask = mask_sample)


def sample_sum_layer(layer, nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples, 
                     ind_target, ind_n, ind_b, block_size, conditional):
    
    num_samples = ind_n.size(0)
    num_nblocks = nids.size(0)
    num_edges = cids.size(1)
    batch_size = node_samples.size(1)
    seed = random.randint(0, 2**31)

    BLOCK_S = min(256, triton.next_power_of_2(num_samples))
    BLOCK_M = min(1024 // BLOCK_S, triton.next_power_of_2(num_nblocks))
    BLOCK_K = min(1024 // BLOCK_S, triton.next_power_of_2(num_edges))

    TILE_SIZE_M = triton.cdiv(num_nblocks, BLOCK_M)
    TILE_SIZE_K = triton.cdiv(num_edges, BLOCK_K)

    grid = (triton.cdiv(num_samples, BLOCK_S),)

    sample_sum_layer_kernel[grid](
        nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples, 
        ind_target, ind_n, ind_b, seed, block_size, batch_size, num_edges, num_samples, num_nblocks, 
        BLOCK_S, BLOCK_M, TILE_SIZE_M, BLOCK_K, TILE_SIZE_K, conditional
    )

    return None


def push_non_neg_ones_to_front(matrix):

    result = torch.full_like(matrix, -1)

    s_mask = (matrix != -1)
    d_mask = torch.sum(s_mask, dim = 0, keepdims = True) > torch.arange(matrix.size(0)).to(matrix.device)[:,None]

    result[d_mask] = matrix[s_mask]
    matrix[:] = result[:]

    return s_mask.long().sum(dim = 0)


def sample(pc: TensorCircuit, num_samples: Optional[int] = None, conditional: bool = False):
    if not conditional:
        assert num_samples is not None, "`num_samples` should be specified when doing unconditioned sampling."
    else:
        num_samples = pc.node_mars.size(0) # Reuse the batch size

    root_ns = pc.root_ns
    assert root_ns._output_ind_range[1] - root_ns._output_ind_range[0] == 1, "It is ambiguous to sample from multi-head PCs."

    num_nscopes = 0
    num_escopes = 0
    for layer_group in pc.layers(ret_layer_groups = True):
        if layer_group.is_input() or layer_group.is_sum():
            for layer in layer_group:
                num_nscopes += len(layer.scopes)
        else:
            assert layer_group.is_prod()
            curr_escopes = 0
            for layer in layer_group:
                curr_escopes += len(layer.scopes)
            num_escopes = max(num_escopes, curr_escopes)

    node_samples = torch.zeros([num_nscopes * 2, num_samples], dtype = torch.long, device = pc.device)
    element_samples = torch.zeros([num_escopes, num_samples], dtype = torch.long, device = pc.device)
    element_pointers = np.zeros([num_samples], dtype = np.int64)

    # Initialize pointers to the root node
    node_samples[:,:] = -1
    node_samples[0,:] = root_ns._output_ind_range[0]

    # Iterate (backward) through layers
    for layer_id in range(len(pc.inner_layer_groups)-1, -1, -1):
        layer_group = pc.inner_layer_groups[layer_id]
        if layer_group.is_sum():
            # Initialize `element_samples` and `element_pointers`
            element_samples[:,:] = -1
            element_pointers[:] = 0

            # Iterate over sum layers in the current layer group
            for layer in layer_group:

                # Gather the indices to be processed
                lsid, leid = layer._layer_nid_range
                ind_n, ind_b = torch.where((node_samples >= lsid) & (node_samples < leid))

                # Pre-compute the target indices in `element_samples`
                ind_target = np.zeros([ind_n.size(0)], dtype = np.int64)
                _assign_cids_ind_target(ind_target, element_pointers, ind_b.detach().cpu().numpy(), num_samples)
                ind_target = torch.from_numpy(ind_target).to(pc.device)

                # In the case of conditional sampling, recompute to get the `element_mars`
                if conditional:
                    pc.inner_layer_groups[layer_id-1](pc.node_mars, pc.element_mars)

                # Sample child indices
                for partition_id in range(layer.num_fw_partitions):
                    nids = layer.partitioned_nids[partition_id]
                    cids = layer.partitioned_cids[partition_id]
                    pids = layer.partitioned_pids[partition_id]
                    
                    sample_sum_layer(layer, nids, cids, pids, pc.node_mars, pc.element_mars, pc.params, 
                                     node_samples, element_samples, ind_target, ind_n, ind_b, 
                                     layer.block_size, conditional)

                # Clear completed nodes
                node_samples[ind_n, ind_b] = -1

        else:
            assert layer_group.is_prod()

            # Iterate over product layers in the current layer group
            for layer in layer_group:
                # Re-align `node_samples` by pushing all values to the front
                node_pointers = push_non_neg_ones_to_front(node_samples)

                # Gather the indices to be processed
                lsid, leid = layer._layer_nid_range
                mask = (element_samples >= lsid) & (element_samples < leid)
                ind_n, ind_b = torch.where(mask)
                global_cids = element_samples[ind_n, ind_b]

                # Get child indices
                for partition_id in range(layer.num_fw_partitions):
                    nids = layer.partitioned_nids[partition_id]
                    cids = layer.partitioned_cids[partition_id]

                    is_match = (global_cids[:,None] >= nids[None,:]) & (global_cids[:,None] < nids[None,:] + layer.block_size)
                    local_nids = (is_match * torch.arange(1, nids.size(0) + 1, device = pc.device)[None,:] - 1).sum(dim = 1)
                    local_nid_offsets = ((global_cids[:,None] - nids[None,:]) * is_match.long()).sum(dim = 1)

                    target_nids = cids[local_nids,:] + local_nid_offsets[:,None]
                    target_cids = cids[local_nids,:]

                    target_idx = node_pointers[ind_b] + (torch.cumsum(mask, dim = 0)[ind_n, ind_b] - 1) * cids.size(1)

                    if target_idx.max() + cids.size(1) > node_samples.size(0):

                        node_samples_new = torch.zeros([target_idx.max() + cids.size(1), num_samples], dtype = torch.long, device = pc.device)
                        node_samples_new[:,:] = -1

                        node_samples_new[:node_samples.size(0),:] = node_samples
                        node_samples = node_samples_new

                    match_filter = is_match.any(dim = 1)
                    target_nids = target_nids[match_filter,:]
                    target_cids = target_cids[match_filter,:]
                    target_idx = target_idx[match_filter]
                    target_b = ind_b[match_filter]

                    for i in range(cids.size(1)):
                        cmask = target_cids[:,i] != 0
                        node_samples[target_idx[cmask]+i, target_b[cmask]] = target_nids[cmask,i]

                    node_pointers = (node_samples != -1).sum(dim = 0)

    # Create tensor for the samples
    data_dtype = pc.input_layer_group[0].get_data_dtype()
    samples = torch.zeros([pc.num_vars, num_samples], dtype = data_dtype, device = pc.device)

    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, num_samples), set_value = 0.0)
    ind_n, ind_b = torch.where(node_samples != -1)
    ind_node = node_samples[ind_n, ind_b]
    pc.node_flows[ind_node, ind_b] = 1.0

    for layer in pc.input_layer_group:
        seed = random.randint(0, 2**31)
        layer.sample(samples, pc.node_flows, seed = seed)

    return samples.permute(1, 0).contiguous()
