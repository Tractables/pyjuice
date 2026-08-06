"""
Ancestral sampling through a product layer.

A product node is deterministic -- every child is taken -- so there is nothing to draw here; the work
is entirely addressing. It takes two passes because the number of children varies per node and the
destination slots have to be allocated before anything is written:

* :func:`count_prod_nch` counts each selected element's children and, on the way, caches the
  `(row, offset)` location it found in the compiled layout, along with the partition that owns it;
* :func:`sample_prod_layer` reuses that location to scatter the children into `node_samples`.

Product layers are untouched by an external parameterization -- gates live on sum edges -- so these
serve every layer type.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
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

    # Handle triton bug.. (otherwise `local_nids` will be wrong)
    local_nids = tl.load(ind_nids + offs_sample, mask = mask_sample, other = 0)

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

    BLOCK_C = min(128, triton.next_power_of_2(num_edges))
    BLOCK_M = min(512, triton.next_power_of_2(num_nblocks))
    BLOCK_S = min(2048 // BLOCK_C, 2048 // BLOCK_M, max(triton.next_power_of_2(num_samples // 128), 1))

    M_NUM_BLKS = triton.cdiv(num_nblocks, BLOCK_M)
    C_NUM_BLKS = triton.cdiv(num_edges, BLOCK_C)

    grid = (triton.cdiv(num_samples, BLOCK_S),)

    count_prod_nch_kernel[grid](
        nids, cids, element_samples, ind_ch_count, ind_nids, ind_nid_offs, ind_mask, ind_n, ind_b, partition_id,
        block_size, num_samples, num_nblocks, batch_size, num_edges, BLOCK_M, BLOCK_C, BLOCK_S, M_NUM_BLKS, C_NUM_BLKS
    )

    return None


@triton.jit
def sample_prod_layer_kernel(nids, cids, node_samples, element_samples, ind_target, ind_target_sid, ind_n, ind_b,
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
    target_sid = tl.load(ind_target_sid + offs_sample, mask = mask_sample, other = 0)
    for i in range(C_NUM_BLKS):

        c_ids = tl.load(cids + local_nids[:,None] * num_edges + offs_child[None,:], mask = (mask_sample[:,None] & mask_child[None,:]), other = 0)
        target_id = tl.load(ind_target + target_sid[:,None] + offs_child[None,:], mask = (mask_sample[:,None] & mask_child[None,:] & (c_ids > 0)), other = 0)

        tl.store(node_samples + target_id, c_ids + local_nid_offs[:,None], mask = (mask_sample[:,None] & mask_child[None,:] & (c_ids > 0)))

        offs_child += BLOCK_C
        mask_child = offs_child < num_edges


def sample_prod_layer(layer, nids, cids, node_samples, element_samples, ind_target, ind_target_sid,
                      ind_n, ind_b, ind_nids, ind_nid_offs, ind_mask, block_size, partition_id):

    num_samples = ind_n.size(0)
    num_nblocks = nids.size(0)
    num_edges = cids.size(1)
    batch_size = node_samples.size(1)

    BLOCK_C = min(1024, triton.next_power_of_2(num_edges))
    BLOCK_S = min(1024 // BLOCK_C, max(triton.next_power_of_2(num_samples // 128), 1))

    C_NUM_BLKS = triton.cdiv(num_edges, BLOCK_C)

    grid = (triton.cdiv(num_samples, BLOCK_S),)

    sample_prod_layer_kernel[grid](
        nids, cids, node_samples, element_samples, ind_target, ind_target_sid, ind_n, ind_b,
        ind_nids, ind_nid_offs, ind_mask, partition_id, block_size, num_samples,
        num_nblocks, batch_size, num_edges, BLOCK_S, BLOCK_C, C_NUM_BLKS
    )

    return None
