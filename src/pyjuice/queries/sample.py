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
from pyjuice.utils.kernel_launcher import FastJITFunction


@njit
def _assign_cids_ind_target(ind_target, element_pointers, ind_b, num_samples):
    for i in range(ind_target.shape[0]):
        bid = ind_b[i]
        ind_t = element_pointers[bid]
        ind_target[i] = ind_t * num_samples + bid
        element_pointers[bid] = ind_t + 1


@triton.jit
def _assign_nids_ind_target_kernel(ind_target_ptr, ind_ch_count_ptr, node_pointers_ptr, ind_b_ptr, 
                                   num_samples, num_nodes, BLOCK_SIZE: tl.constexpr, NUM_BLKS: tl.constexpr):
    bid = tl.program_id(0) # The batch ID for this node block

    target_val_sid = tl.load(node_pointers_ptr + bid)

    offsets = tl.arange(0, BLOCK_SIZE)
    offset_first = 0

    for i in range(NUM_BLKS):
        mask = (offsets < num_nodes)

        inds_b = tl.load(ind_b_ptr + offsets, mask = mask)
        mask_b = (inds_b == bid)

        count_c = tl.load(ind_ch_count_ptr + offsets, mask = mask & mask_b, other = 0)
        
        cumcount_c = tl.cumsum(count_c, axis = 0) - count_c + target_val_sid

        tl.store(ind_target_ptr + offsets, cumcount_c * num_samples + bid, mask = mask & mask_b)

        last_onehot = ((offsets + 1) == tl.max((offsets + 1) * mask_b.to(tl.int64))).to(tl.int64)
        target_val_sid = tl.max(cumcount_c) + tl.sum(count_c * last_onehot)

        offsets += BLOCK_SIZE


def _assign_nids_ind_target(ind_target, ind_ch_count, node_pointers, ind_b, num_samples):
    """
    A GPU implementation of the following:

    @njit
    def _assign_nids_ind_target(ind_target, ind_ch_count, node_pointers, ind_b, num_samples):
        for nid in range(ind_target.shape[0]):
            bid = ind_b[nid]
            ind_t = node_pointers[bid]
            ind_target[i] = ind_t * num_samples + bid
            node_pointers[bid] = ind_t + ind_ch_count[nid]
    """

    num_nodes = ind_b.size(0)

    BLOCK_SIZE = min(512, triton.next_power_of_2(num_nodes))
    NUM_BLKS = triton.cdiv(num_nodes, BLOCK_SIZE)

    grid = (num_samples,)

    _assign_nids_ind_target_kernel[grid](
        ind_target,
        ind_ch_count,
        node_pointers,
        ind_b,
        num_samples,
        num_nodes,
        BLOCK_SIZE = BLOCK_SIZE,
        NUM_BLKS = NUM_BLKS
    )


@triton.jit
# @FastJITFunction
def sample_sum_layer_kernel(nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples, 
                            ind_target, ind_n, ind_b, seed, block_size: tl.constexpr, batch_size: tl.constexpr, 
                            num_edges: tl.constexpr, num_samples: tl.constexpr, num_nblocks: tl.constexpr, BLOCK_S: tl.constexpr, 
                            BLOCK_M: tl.constexpr, M_NUM_BLKS: tl.constexpr, BLOCK_K: tl.constexpr, K_NUM_BLKS: tl.constexpr,
                            conditional: tl.constexpr):
    
    pid_s = tl.program_id(0) # ID of size-`BLOCK_S` batches

    # Sample offsets and mask
    offs_sample = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    mask_sample = offs_sample < num_samples

    # Load node and batch ids
    node_sample_id = tl.load(ind_n + offs_sample, mask = mask_sample, other = 0)
    batch_id = tl.load(ind_b + offs_sample, mask = mask_sample, other = 0)
    node_id = tl.load(node_samples + node_sample_id * batch_size + batch_id)

    # Locate node ids in `nids`
    offs_nids = tl.arange(0, BLOCK_M)
    local_nids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    local_nid_offs = tl.zeros([BLOCK_S], dtype = tl.int64)
    for i in range(M_NUM_BLKS):
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
        nmars = tl.load(node_mars + node_id * batch_size + batch_id, mask = mask_sample, other = 0.0) # [Block_B]

    # Main loop over blocks of child nodes
    chids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    for i in range(K_NUM_BLKS):

        # Load parameters
        param_id = tl.load(pids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
        epars = tl.load(params + param_id + local_nid_offs[None,:], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0) # [BLOCK_K, BLOCK_B]

        if conditional:
            # In this case, we use `param * cmar / nmar` as the "parameter"
            emars_id = tl.load(cids + local_nids[None,:] * num_edges + offs_child[:,None], mask = (mask_sample[None,:] & mask_child[:,None]), other = 0)
            emars = tl.load(element_mars + emars_id * batch_size + batch_id, mask = (mask_sample[None,:] & mask_child[:,None]), other = 0.0)

            epars = epars * tl.exp(emars - nmars[None,:]) # [BLOCK_K, BLOCK_B]
        
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

    M_NUM_BLKS = triton.cdiv(num_nblocks, BLOCK_M)
    K_NUM_BLKS = triton.cdiv(num_edges, BLOCK_K)

    grid = (triton.cdiv(num_samples, BLOCK_S),)

    sample_sum_layer_kernel[grid](
        nids, cids, pids, node_mars, element_mars, params, node_samples, element_samples, 
        ind_target, ind_n, ind_b, seed, block_size, batch_size, num_edges, num_samples, num_nblocks, 
        BLOCK_S, BLOCK_M, M_NUM_BLKS, BLOCK_K, K_NUM_BLKS, conditional
    )

    return None


@triton.jit
# @FastJITFunction
def push_non_neg_ones_to_front_kernel(matrix_ptr, counts_ptr, row_count, col_count, 
                                      BLOCK_SIZE: tl.constexpr, NUM_BLKS: tl.constexpr):
    off_col = tl.program_id(0)

    offs_row = tl.arange(0, BLOCK_SIZE)
    
    # Target row id
    target_row_id = -1
    target_row_id = target_row_id.to(tl.int64)
    
    for i in range(NUM_BLKS):
        mask_row = (offs_row < row_count)

        value = tl.load(matrix_ptr + offs_row * col_count + off_col, mask = mask_row, other = -1)

        mask_val = (value != -1)
        
        offs_target = tl.cumsum(mask_val.to(tl.int64), axis = 0) + target_row_id

        tl.store(matrix_ptr + offs_target * col_count + off_col, value, mask = mask_row & mask_val)

        offs_row += BLOCK_SIZE
        target_row_id += tl.sum(mask_val.to(tl.int64))

    tl.store(counts_ptr + off_col, target_row_id + 1)

    target_row_id = (target_row_id + 1).to(tl.int32)

    while target_row_id < row_count:
        offs_row = tl.arange(0, BLOCK_SIZE) + target_row_id
        mask_row = (offs_row < row_count)

        tl.store(matrix_ptr + offs_row * col_count + off_col, -1, mask = mask_row)

        target_row_id += BLOCK_SIZE


def push_non_neg_ones_to_front(matrix):
    """
    An efficient implementation of the following:

    def push_non_neg_ones_to_front(matrix):
        result = torch.full_like(matrix, -1)

        s_mask = (matrix != -1)
        d_mask = torch.sum(s_mask, dim = 0, keepdims = True) > torch.arange(matrix.size(0)).to(matrix.device)[:,None]

        result[d_mask] = matrix[s_mask]
        matrix[:] = result[:]

        return s_mask.long().sum(dim = 0)
    """
    row_count, col_count = matrix.size()

    counts = torch.zeros([col_count], dtype = torch.long, device = matrix.device)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(row_count))
    NUM_BLKS = triton.cdiv(row_count, BLOCK_SIZE)
    
    grid = lambda meta: (col_count,)
    
    push_non_neg_ones_to_front_kernel[grid](
        matrix, 
        counts, 
        row_count, 
        col_count, 
        BLOCK_SIZE = BLOCK_SIZE,
        NUM_BLKS = NUM_BLKS
    )
    
    return counts


@triton.jit
# @FastJITFunction
def count_prod_nch_kernel(nids, cids, element_samples, ind_ch_count, ind_nids, ind_nid_offs, ind_mask, ind_n, ind_b, partition_id,
                          block_size: tl.constexpr, num_samples: tl.constexpr, num_nblocks: tl.constexpr, 
                          batch_size: tl.constexpr, num_edges: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, 
                          BLOCK_S: tl.constexpr, M_NUM_BLKS: tl.constexpr, C_NUM_BLKS: tl.constexpr):
    
    pid_s = tl.program_id(0) # ID of size-`BLOCK_S` batches

    # Sample offsets and mask
    offs_sample = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    mask_sample = offs_sample < num_samples

    # Load node and batch ids
    node_sample_id = tl.load(ind_n + offs_sample, mask = mask_sample, other = 0)
    batch_id = tl.load(ind_b + offs_sample, mask = mask_sample, other = 0)
    ele_id = tl.load(element_samples + node_sample_id * batch_size + batch_id)

    # Locate node ids in `nids`
    offs_nids = tl.arange(0, BLOCK_M)
    local_nids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
    local_nid_offs = tl.zeros([BLOCK_S], dtype = tl.int64)
    for i in range(M_NUM_BLKS):
        mask_nids = offs_nids < num_nblocks

        ref_nid = tl.load(nids + offs_nids, mask = mask_nids, other = 0)
        is_match = (ele_id[:,None] >= ref_nid[None,:]) & (ele_id[:,None] < ref_nid[None,:] + block_size)

        match_local_id = tl.sum(is_match * (offs_nids[None,:] + 1), axis = 1)
        match_local_offset = tl.sum(is_match * (ele_id[:,None] - ref_nid[None,:]), axis = 1)

        local_nids = tl.where(match_local_id > 0, match_local_id - 1, local_nids)
        local_nid_offs = tl.where(match_local_id > 0, match_local_offset, local_nid_offs)

        offs_nids += BLOCK_M

    # Store `local_nids` and `local_nid_offs` for future reuse
    mask_sample = mask_sample & (local_nids >= 0)
    tl.store(ind_nids + offs_sample, local_nids, mask = mask_sample)
    tl.store(ind_nid_offs + offs_sample, local_nid_offs, mask = mask_sample)
    tl.store(ind_mask + offs_sample, partition_id, mask = mask_sample)

    # Offset for children
    offs_child = tl.arange(0, BLOCK_C)
    mask_child = offs_child < num_edges

    # Main loop over blocks of child nodes
    ch_count = tl.zeros([BLOCK_S], dtype = tl.int64)
    for i in range(C_NUM_BLKS):

        c_ids = tl.load(cids + local_nids[:,None] * num_edges + offs_child[None,:], mask = (mask_sample[:,None] & mask_child[None,:]), other = 0)
        ch_count += tl.sum((c_ids > 0).to(tl.int64), axis = 1)

        offs_child += BLOCK_C
        mask_child = offs_child < num_edges

    # Store `ch_count`
    tl.store(ind_ch_count + offs_sample, ch_count, mask = mask_sample)


def count_prod_nch(layer, nids, cids, element_samples, ind_ch_count, ind_nids, ind_nid_offs, ind_mask, ind_n, ind_b, block_size, partition_id):

    num_samples = ind_n.size(0)
    num_nblocks = nids.size(0)
    batch_size = element_samples.size(1)
    num_edges = cids.size(1)

    BLOCK_C = min(1024, triton.next_power_of_2(num_edges))
    BLOCK_S = min(1024 // BLOCK_C, triton.next_power_of_2(num_samples))
    BLOCK_M = min(1024 // BLOCK_S, triton.next_power_of_2(num_nblocks))

    M_NUM_BLKS = triton.cdiv(num_nblocks, BLOCK_M)
    C_NUM_BLKS = triton.cdiv(num_edges, BLOCK_C)

    grid = (triton.cdiv(num_samples, BLOCK_S),)

    count_prod_nch_kernel[grid](
        nids, cids, element_samples, ind_ch_count, ind_nids, ind_nid_offs, ind_mask, ind_n, ind_b, partition_id, 
        block_size, num_samples, num_nblocks, batch_size, num_edges, BLOCK_M, BLOCK_C, BLOCK_S, M_NUM_BLKS, C_NUM_BLKS
    )

    return None


@triton.jit
# @FastJITFunction
def sample_prod_layer_kernel(nids, cids, node_samples, element_samples, ind_target, ind_n, ind_b, 
                             ind_nids, ind_nid_offs, ind_mask, partition_id, block_size: tl.constexpr, 
                             num_samples: tl.constexpr, num_nblocks: tl.constexpr, batch_size: tl.constexpr, num_edges: tl.constexpr,
                             BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr, C_NUM_BLKS: tl.constexpr):

    pid_s = tl.program_id(0) # ID of size-`BLOCK_S` batches

    # Sample offsets and mask
    offs_sample = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    mask_sample = offs_sample < num_samples

    # Load node and batch ids
    node_sample_id = tl.load(ind_n + offs_sample, mask = mask_sample, other = 0)
    batch_id = tl.load(ind_b + offs_sample, mask = mask_sample, other = 0)
    ele_id = tl.load(element_samples + node_sample_id * batch_size + batch_id)

    # Load offsets of `nids` and the node offsets
    local_nids = tl.load(ind_nids + offs_sample, mask = mask_sample, other = 0)
    local_nid_offs = tl.load(ind_nid_offs + offs_sample, mask = mask_sample, other = 0)
    local_partition_id = tl.load(ind_mask + offs_sample, mask = mask_sample, other = 0)

    # Update sample mask
    mask_sample = mask_sample & (local_partition_id == partition_id)

    # Offset for children
    offs_child = tl.arange(0, BLOCK_C)
    mask_child = offs_child < num_edges

    # Main loop over blocks of child nodes
    target_id_base = tl.load(ind_target + offs_sample, mask = mask_sample, other = 0)
    for i in range(C_NUM_BLKS):

        c_ids = tl.load(cids + local_nids[:,None] * num_edges + offs_child[None,:], mask = (mask_sample[:,None] & mask_child[None,:]), other = 0)

        target_id = target_id_base[:,None] + offs_child[None,:] * num_samples

        tl.store(node_samples + target_id, c_ids + local_nid_offs[:,None], mask = (mask_sample[:,None] & mask_child[None,:] & (c_ids > 0)))

        offs_child += BLOCK_C
        mask_child = offs_child < num_edges


def sample_prod_layer(layer, nids, cids, node_samples, element_samples, ind_target, 
                      ind_n, ind_b, ind_nids, ind_nid_offs, ind_mask, block_size, partition_id):
    
    num_samples = ind_n.size(0)
    num_nblocks = nids.size(0)
    num_edges = cids.size(1)
    batch_size = node_samples.size(1)

    BLOCK_C = min(128, triton.next_power_of_2(num_edges))
    BLOCK_S = min(128 // BLOCK_C, triton.next_power_of_2(num_samples))

    C_NUM_BLKS = triton.cdiv(num_edges, BLOCK_C)

    grid = (triton.cdiv(num_samples, BLOCK_S),)

    sample_prod_layer_kernel[grid](
        nids, cids, node_samples, element_samples, ind_target, ind_n, ind_b, 
        ind_nids, ind_nid_offs, ind_mask, partition_id, block_size, num_samples, 
        num_nblocks, batch_size, num_edges, BLOCK_S, BLOCK_C, C_NUM_BLKS
    )


def sample(pc: TensorCircuit, num_samples: Optional[int] = None, conditional: bool = False, _sample_input_ns: bool = True):
    if not conditional:
        assert num_samples is not None, "`num_samples` should be specified when doing unconditioned sampling."
    else:
        num_samples = pc.node_mars.size(1) # Reuse the batch size

    root_ns = pc.root_ns
    assert root_ns._output_ind_range[1] - root_ns._output_ind_range[0] == 1, "It is ambiguous to sample from multi-head PCs."

    num_nscopes = 0
    num_escopes = 0
    for layer_group in pc.layers(ret_layer_groups = True):
        curr_scopes = 0
        for layer in layer_group:
            curr_scopes += len(layer.scopes)

        if layer_group.is_input() or layer_group.is_sum():
            num_nscopes += curr_scopes
        else:
            assert layer_group.is_prod()
            num_escopes = max(num_escopes, curr_scopes)

    # Stores selected node indices by the sampler
    node_samples = torch.zeros([num_nscopes, num_samples], dtype = torch.long, device = pc.device)
    # Stores selected element indices by the sampler
    element_samples = torch.zeros([num_escopes, num_samples], dtype = torch.long, device = pc.device)
    # Pointers indicating how many elements are used in each column of `element_samples`
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

                    if ind_n.size(0) == 0:
                        import pdb; pdb.set_trace()
                    
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
                ind_n, ind_b = torch.where((element_samples >= lsid) & (element_samples < leid))

                # Get the number of children for the selected sample indices
                ind_ch_count = torch.zeros_like(ind_n)
                ind_nids = torch.zeros_like(ind_n)
                ind_nid_offs = torch.zeros_like(ind_n)
                ind_mask = torch.zeros_like(ind_n)
                for partition_id in range(layer.num_fw_partitions):
                    nids = layer.partitioned_nids[partition_id]
                    cids = layer.partitioned_cids[partition_id]

                    count_prod_nch(layer, nids, cids, element_samples, ind_ch_count, ind_nids, 
                                   ind_nid_offs, ind_mask, ind_n, ind_b, layer.block_size, partition_id)

                # Pre-compute the target indices in `node_samples`
                ind_target = torch.zeros_like(ind_n)
                _assign_nids_ind_target(ind_target, ind_ch_count, node_pointers, ind_b, num_samples)

                # Store child indices
                for partition_id in range(layer.num_fw_partitions):
                    nids = layer.partitioned_nids[partition_id]
                    cids = layer.partitioned_cids[partition_id]

                    sample_prod_layer(layer, nids, cids, node_samples, element_samples, ind_target, 
                                      ind_n, ind_b, ind_nids, ind_nid_offs, ind_mask, layer.block_size, partition_id)

    if _sample_input_ns:
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
    else:
        # In this case, we do not explicitly sample input nodes
        return node_samples
