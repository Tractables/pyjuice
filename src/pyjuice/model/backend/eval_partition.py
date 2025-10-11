from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.layer import SumLayer, ProdLayer, InputLayer
from pyjuice.utils.kernel_launcher import triton_jit


@triton_jit
def prod_layer_partition_fn_kernel(node_mars, element_mars, nids, cids, n_num_nblocks: tl.constexpr, n_num_nodes: tl.constexpr,
                                   c_num_nblocks: tl.constexpr, block_size: tl.constexpr, TILE_SIZE_N: tl.constexpr, TILE_SIZE_C: tl.constexpr):
    pid_n = tl.program_id(0)

    n_ids = tl.arange(0, TILE_SIZE_N) + pid_n * TILE_SIZE_N
    n_mask = (n_ids < n_num_nodes)
    group_ids = n_ids // block_size
    offs_node = n_ids % block_size

    c_ids = tl.arange(0, TILE_SIZE_C)
    c_mask = (c_ids < c_num_nblocks)

    nc_mask = n_mask[:,None] & c_mask[None,:]

    cids_base = tl.load(cids + group_ids[:,None] * c_num_nblocks + c_ids[None,:], mask = nc_mask)
    cids = cids_base + offs_node[:,None]
    cmars = tl.load(node_mars + cids, mask = nc_mask, other = 0.0)

    nmars = tl.sum(cmars, axis = 1)

    nids_base = tl.load(nids + group_ids, mask = n_mask)
    nids = nids_base + offs_node
    tl.store(element_mars + nids, nmars, mask = n_mask)


def prod_layer_partition_fn(layer, node_mars, element_mars, params):
    block_size = layer.block_size
    for partition_id in range(layer.num_fw_partitions):
        nids = layer.partitioned_nids[partition_id]
        cids = layer.partitioned_cids[partition_id]

        n_num_nblocks = nids.size(0)
        n_num_nodes = n_num_nblocks * block_size
        c_num_nblocks = cids.size(1)

        if c_num_nblocks <= 2048:
            TILE_SIZE_C = triton.next_power_of_2(c_num_nblocks)
            TILE_SIZE_N = min(2048 // TILE_SIZE_C, triton.next_power_of_2(n_num_nodes))

            grid = (triton.cdiv(n_num_nodes, TILE_SIZE_N),)

            prod_layer_partition_fn_kernel[grid](
                node_mars,
                element_mars,
                nids,
                cids,
                n_num_nblocks = n_num_nblocks,
                n_num_nodes = n_num_nodes,
                c_num_nblocks = c_num_nblocks,
                block_size = block_size,
                TILE_SIZE_N = TILE_SIZE_N,
                TILE_SIZE_C = TILE_SIZE_C
            )

        else:
            raise NotImplementedError("Need to implement a new kernel for this case.")


@triton_jit
def sum_layer_partition_fn_block_kernel(node_mars, element_mars, mparams, nids, cids, pids, n_num_nodes: tl.constexpr,
                                        c_num_nodes: tl.constexpr, block_size: tl.constexpr, TILE_SIZE_N: tl.constexpr, 
                                        TILE_SIZE_C: tl.constexpr, C_NUM_TILES: tl.constexpr):
    pid_n = tl.program_id(0)

    ngroup_id = (pid_n * TILE_SIZE_N) // block_size
    offs_nnode = tl.arange(0, TILE_SIZE_N) + ((pid_n * TILE_SIZE_N) % block_size)

    acc = tl.zeros([TILE_SIZE_N], dtype = tl.float32) - float("inf")
    for pid_c in range(C_NUM_TILES):

        offs_cnode = tl.arange(0, TILE_SIZE_C) + pid_c * TILE_SIZE_C
        mask_cnode = (offs_cnode < c_num_nodes)
        cnode_ids = tl.load(cids + ngroup_id * c_num_nodes + offs_cnode, mask = mask_cnode) # [TILE_SIZE_C]
        cmars = tl.load(element_mars + cnode_ids, mask = mask_cnode, other = 0.0)

        par_ids = tl.load(pids + ngroup_id * c_num_nodes + offs_cnode, mask = mask_cnode) # [TILE_SIZE_C]
        pars = tl.load(mparams + par_ids[None,:] + offs_nnode[:,None], mask = mask_cnode[None,:]) # [TILE_SIZE_N, TILE_SIZE_C]

        cmars_max = tl.max(cmars)
        cmars_sub = tl.exp(cmars - cmars_max)

        nmars = tl.sum(pars * cmars_sub[None,:], axis = 1)

        acc = tl.where(cmars_max > acc,
            tl.log(nmars + tl.exp(acc - cmars_max) + 1e-24) + cmars_max,
            tl.log(tl.exp(cmars_max - acc) * nmars + 1.0) + acc
        )

    nid_base = tl.load(nids + ngroup_id)
    nnode_ids = nid_base + offs_nnode # [TILE_SIZE_N]
    tl.store(node_mars + nnode_ids, acc)


@triton_jit
def sum_layer_partition_fn_node_kernel(node_mars, element_mars, mparams, nids, cids, pids, n_num_nodes: tl.constexpr,
                                       c_num_nodes: tl.constexpr, block_size: tl.constexpr, TILE_SIZE_N: tl.constexpr, 
                                       TILE_SIZE_C: tl.constexpr, C_NUM_TILES: tl.constexpr):
    pid_n = tl.program_id(0)

    n_ids = tl.arange(0, TILE_SIZE_N) + pid_n * TILE_SIZE_N
    n_mask = (n_ids < n_num_nodes)
    ngroup_ids = n_ids // block_size
    offs_nnode = n_ids % block_size

    acc = tl.zeros([TILE_SIZE_N], dtype = tl.float32) - float("inf")
    for pid_c in range(C_NUM_TILES):

        offs_cnode = tl.arange(0, TILE_SIZE_C) + pid_c * TILE_SIZE_C
        mask_cnode = (offs_cnode < c_num_nodes)
        cnode_ids = tl.load(cids + ngroup_ids[:,None] * c_num_nodes + offs_cnode[None,:], mask = (n_mask[:,None] & mask_cnode[None,:])) # [TILE_SIZE_N, TILE_SIZE_C]
        cmars = tl.load(element_mars + cnode_ids, mask = (n_mask[:,None] & mask_cnode[None,:]), other = 0.0)

        par_ids = tl.load(pids + ngroup_ids[:,None] * c_num_nodes + offs_cnode[None,:], mask = (n_mask[:,None] & mask_cnode[None,:])) # [TILE_SIZE_N, TILE_SIZE_C]
        pars = tl.load(mparams + par_ids + offs_nnode[:,None], mask = (n_mask[:,None] & mask_cnode[None,:])) # [TILE_SIZE_N, TILE_SIZE_C]

        cmars_max = tl.max(cmars, axis = 1)
        cmars_sub = tl.exp(cmars - cmars_max[:,None])

        nmars = tl.sum(pars * cmars_sub, axis = 1)

        acc = tl.where(cmars_max > acc,
            tl.log(nmars + tl.exp(acc - cmars_max) + 1e-24) + cmars_max,
            tl.log(tl.exp(cmars_max - acc) * nmars + 1.0) + acc
        )

    nid_base = tl.load(nids + ngroup_ids, mask = n_mask) # [TILE_SIZE_N]
    nnode_ids = nid_base + offs_nnode # [TILE_SIZE_N]
    tl.store(node_mars + nnode_ids, acc, mask = n_mask)


def sum_layer_partition_fn(layer, node_mars, element_mars, params):
    block_size = layer.block_size
    for partition_id in range(layer.num_fw_partitions):
        nids = layer.partitioned_nids[partition_id]
        cids = layer.partitioned_cids[partition_id]
        pids = layer.partitioned_pids[partition_id]

        n_num_nodes = nids.size(0) * block_size
        c_num_nodes = cids.size(1)

        if block_size >= 8:
            TILE_SIZE_C = min(1024, triton.next_power_of_2(c_num_nodes))
            TILE_SIZE_N = min(1024 // TILE_SIZE_C, block_size)

            C_NUM_TILES = triton.cdiv(c_num_nodes, TILE_SIZE_C)
            
            grid = (triton.cdiv(n_num_nodes, TILE_SIZE_N),)

            sum_layer_partition_fn_block_kernel[grid](
                node_mars,
                element_mars,
                params,
                nids,
                cids,
                pids,
                n_num_nodes = n_num_nodes,
                c_num_nodes = c_num_nodes,
                block_size = block_size,
                TILE_SIZE_N = TILE_SIZE_N,
                TILE_SIZE_C = TILE_SIZE_C,
                C_NUM_TILES = C_NUM_TILES
            )

        else:
            TILE_SIZE_C = min(1024, triton.next_power_of_2(c_num_nodes))
            TILE_SIZE_N = min(1024 // TILE_SIZE_C, n_num_nodes)

            C_NUM_TILES = triton.cdiv(c_num_nodes, TILE_SIZE_C)

            grid = (triton.cdiv(n_num_nodes, TILE_SIZE_N),)

            sum_layer_partition_fn_node_kernel[grid](
                node_mars,
                element_mars,
                params,
                nids,
                cids,
                pids,
                n_num_nodes = n_num_nodes,
                c_num_nodes = c_num_nodes,
                block_size = block_size,
                TILE_SIZE_N = TILE_SIZE_N,
                TILE_SIZE_C = TILE_SIZE_C,
                C_NUM_TILES = C_NUM_TILES
            )


def eval_partition_fn(pc):
    """
    Computes the partition function of every node.
    We use the first # nodes entries in `pc.node_mars` and the first # element entries in `pc.element_mars`.
    """
    assert hasattr(pc, "node_mars") and hasattr(pc, "element_mars")
    assert pc.node_mars.numel() >= pc.num_nodes
    assert pc.element_mars.numel() >= pc.num_elements

    node_mars = pc.node_mars.view(-1) # Reuse the allocated memory
    element_mars = pc.element_mars.view(-1)

    node_mars[:pc.num_nodes] = 0.0
    element_mars[:pc.num_elements] = -float("inf")

    # Forward pass over the input layers
    for layer in pc.input_layer_group:
        layer.eval_partition_fn(node_mars)

    # Forward pass over the inner layers
    for layer_id, layer_group in enumerate(pc.inner_layer_groups):
        layer_group = pc.inner_layer_groups[layer_id]

        for layer in layer_group:
            if layer.is_prod():
                prod_layer_partition_fn(layer, node_mars, element_mars, pc.params)

            elif layer.is_sum():
                sum_layer_partition_fn(layer, node_mars, element_mars, pc.params)

    partition_fn = node_mars[pc._root_node_range[0]:pc._root_node_range[1]]
    return partition_fn