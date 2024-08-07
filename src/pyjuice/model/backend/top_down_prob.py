from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.layer import SumLayer, ProdLayer, InputLayer
from pyjuice.utils.kernel_launcher import FastJITFunction


@FastJITFunction
def prod_layer_td_backward_kernel(node_flows, element_flows, u_cids, parids, c_num_nblocks: tl.constexpr,
                                  c_num_nodes: tl.constexpr, n_num_nblocks: tl.constexpr, block_size: tl.constexpr,
                                  TILE_SIZE_N: tl.constexpr, TILE_SIZE_C: tl.constexpr):
    pid_c = tl.program_id(0)

    c_ids = tl.arange(0, TILE_SIZE_C) + pid_c * TILE_SIZE_C
    c_mask = (c_ids < c_num_nodes)
    group_ids = c_ids // block_size
    offs_node = c_ids % block_size

    n_ids = tl.arange(0, TILE_SIZE_N)
    n_mask = (n_ids < n_num_nblocks)

    cn_mask = c_mask[:,None] & n_mask[None,:]

    nids_base = tl.load(parids + group_ids[:,None] * n_num_nblocks + n_ids[None,:], mask = cn_mask)
    nids = nids_base + offs_node[:,None]
    nflows = tl.load(element_flows + nids, mask = cn_mask, other = 0.0)

    cflows = tl.sum(nflows, axis = 1)

    cids_base = tl.load(u_cids + group_ids, mask = c_mask)
    cids = cids_base + offs_node
    tl.atomic_add(node_flows + cids, cflows, mask = c_mask)


def prod_layer_td_backward(layer, node_flows, element_flows):
    block_size = layer.block_size
    for partition_id in range(layer.num_bk_partitions):
        u_cids = layer.partitioned_u_cids[partition_id]
        parids = layer.partitioned_parids[partition_id]

        c_num_nblocks = u_cids.size(0)
        c_num_nodes = c_num_nblocks * block_size
        n_num_nblocks = parids.size(1)

        if n_num_nblocks <= 2048:
            TILE_SIZE_N = triton.next_power_of_2(n_num_nblocks)
            TILE_SIZE_C = min(2048 // TILE_SIZE_N, triton.next_power_of_2(c_num_nodes))

            grid = (triton.cdiv(c_num_nodes, TILE_SIZE_C),)

            prod_layer_td_backward_kernel[grid](
                node_flows,
                element_flows,
                u_cids,
                parids,
                c_num_nblocks = c_num_nblocks,
                c_num_nodes = c_num_nodes,
                n_num_nblocks = n_num_nblocks,
                block_size = block_size,
                TILE_SIZE_N = TILE_SIZE_N,
                TILE_SIZE_C = TILE_SIZE_C
            )

        else:
            raise NotImplementedError("Need to implement a new kernel for this case.")


@FastJITFunction
def sum_layer_td_backward_block_kernel(node_flows, element_flows, params, chids, parids, parpids, n_num_nodes: tl.constexpr,
                                       c_num_nodes: tl.constexpr, TILE_SIZE_N: tl.constexpr, TILE_SIZE_C: tl.constexpr,
                                       n_block_size: tl.constexpr, c_block_size: tl.constexpr, NUM_N_BLKS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    cgroup_id = (pid_c * TILE_SIZE_C) // c_block_size
    offs_cnode = tl.arange(0, TILE_SIZE_C) + ((pid_c * TILE_SIZE_C) % c_block_size)

    cid_base = tl.load(chids + cgroup_id)
    cids = cid_base + offs_cnode # [TILE_SIZE_C]

    ngroup_id = (pid_n * TILE_SIZE_N) // n_block_size
    offs_nnode = tl.arange(0, TILE_SIZE_N) + ((pid_n * TILE_SIZE_N) % n_block_size)

    nid_base = tl.load(parids + cgroup_id * NUM_N_BLKS + ngroup_id)
    nids = nid_base + offs_nnode # [TILE_SIZE_N]
    nflows = tl.load(node_flows + nids)

    pid_base = tl.load(parpids + cgroup_id * NUM_N_BLKS + ngroup_id)
    pids = pid_base + offs_cnode[:,None] * n_block_size + offs_nnode[None,:] # [TILE_SIZE_C, TILE_SIZE_N]
    epars = tl.load(params + pids)

    cflows = tl.sum(epars * nflows[None,:], axis = 1)
    tl.atomic_add(element_flows + cids, cflows)


@FastJITFunction
def sum_layer_td_backward_node_kernel(node_flows, element_flows, params, chids, parids, parpids, n_num_nodes: tl.constexpr,
                                      c_num_nodes: tl.constexpr, TILE_SIZE_N: tl.constexpr, TILE_SIZE_C: tl.constexpr,
                                      n_block_size: tl.constexpr, c_block_size: tl.constexpr, NUM_N_BLKS: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_ids = tl.arange(0, TILE_SIZE_C) + pid_c * TILE_SIZE_C
    c_mask = (c_ids < c_num_nodes)
    cgroup_ids = c_ids // c_block_size
    offs_cnode = c_ids % c_block_size

    cids_base = tl.load(chids + cgroup_ids)
    cids = cids_base + offs_cnode # [TILE_SIZE_C]
    
    n_ids = tl.arange(0, TILE_SIZE_N) + pid_n * TILE_SIZE_N
    n_mask = (n_ids < n_num_nodes)
    ngroup_ids = n_ids // n_block_size
    offs_nnode = n_ids % n_block_size

    cn_mask = c_mask[:,None] & n_mask[None,:]

    nids_base = tl.load(parids + cgroup_ids[:,None] * NUM_N_BLKS + ngroup_ids[None,:], mask = cn_mask)
    nids = nids_base + offs_nnode[None,:] # [TILE_SIZE_C, TILE_SIZE_N]
    nflows = tl.load(node_flows + nids, mask = cn_mask, other = 0.0)

    pids_base = tl.load(parpids + cgroup_ids[:,None] * NUM_N_BLKS + ngroup_ids[None,:], mask = cn_mask)
    pids = pids_base + offs_cnode[:,None] * n_block_size + offs_nnode[None,:] # [TILE_SIZE_C, TILE_SIZE_N]
    epars = tl.load(params + pids, mask = cn_mask, other = 0.0)

    cflows = tl.sum(epars * nflows, axis = 1)
    tl.atomic_add(element_flows + cids, cflows)


def sum_layer_td_backward(layer, node_flows, element_flows, params):
    block_size = layer.block_size
    for partition_id in range(layer.num_bk_partitions):
        chids = layer.partitioned_chids[partition_id]
        parids = layer.partitioned_parids[partition_id]
        parpids = layer.partitioned_parpids[partition_id]
        cs_block_size = layer.cs_block_sizes[partition_id]

        n_num_nodes = parids.size(1) * block_size # max number of parents per c node
        c_num_nodes = chids.size(0) * cs_block_size
        
        if block_size >= 8 and cs_block_size >= 8:
            TILE_SIZE_C = min(64, cs_block_size)
            TILE_SIZE_N = min(64, block_size)

            grid = (triton.cdiv(n_num_nodes, TILE_SIZE_N), triton.cdiv(c_num_nodes, TILE_SIZE_C))

            sum_layer_td_backward_block_kernel[grid](
                node_flows,
                element_flows,
                params,
                chids,
                parids,
                parpids,
                n_num_nodes = n_num_nodes,
                c_num_nodes = c_num_nodes,
                TILE_SIZE_N = TILE_SIZE_N,
                TILE_SIZE_C = TILE_SIZE_C,
                n_block_size = block_size,
                c_block_size = cs_block_size,
                NUM_N_BLKS = parids.size(1)
            )

        else:
            TILE_SIZE_C = min(32, triton.next_power_of_2(c_num_nodes))
            TILE_SIZE_N = min(32, triton.next_power_of_2(n_num_nodes))

            grid = (triton.cdiv(n_num_nodes, TILE_SIZE_N), triton.cdiv(c_num_nodes, TILE_SIZE_C))

            sum_layer_td_backward_node_kernel[grid](
                node_flows,
                element_flows,
                params,
                chids,
                parids,
                parpids,
                n_num_nodes = n_num_nodes,
                c_num_nodes = c_num_nodes,
                TILE_SIZE_N = TILE_SIZE_N,
                TILE_SIZE_C = TILE_SIZE_C,
                n_block_size = block_size,
                c_block_size = cs_block_size,
                NUM_N_BLKS = parids.size(1)
            )


@FastJITFunction
def sum_layer_td_pflow_kernel(node_flows, params, param_flows, nids, cids, pids, pfids, scale,
                              n_num_nodes: tl.constexpr, c_num_nodes: tl.constexpr,
                              TILE_SIZE_N: tl.constexpr, TILE_SIZE_C: tl.constexpr,
                              n_block_size: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_ids = tl.arange(0, TILE_SIZE_N) + pid_n * TILE_SIZE_N
    n_mask = (n_ids < n_num_nodes)
    ngroup_ids = n_ids // n_block_size
    offs_nnode = n_ids % n_block_size

    nids_base = tl.load(nids + ngroup_ids, mask = n_mask)
    snids = nids_base + offs_nnode # [TILE_SIZE_N]
    nflows = tl.load(node_flows + snids, mask = n_mask, other = 0.0)

    offs_cnode = tl.arange(0, TILE_SIZE_C) + pid_c * TILE_SIZE_C
    c_mask = (offs_cnode < c_num_nodes)

    nc_mask = n_mask[:,None] & c_mask[None,:]

    pids_base = tl.load(pids + ngroup_ids[:,None] * c_num_nodes + offs_cnode[None,:], mask = nc_mask)
    epids = pids_base + offs_nnode[:,None] # [TILE_SIZE_N, TILE_SIZE_C]
    epars = tl.load(params + epids, mask = nc_mask, other = 0.0)

    eflows = epars * nflows[:,None] * scale

    pfids_base = tl.load(pfids + ngroup_ids[:,None] * c_num_nodes + offs_cnode[None,:], mask = nc_mask)
    epfids = pfids_base + offs_nnode[:,None] # [TILE_SIZE_N, TILE_SIZE_C]
    tl.atomic_add(param_flows + epfids, eflows, mask = nc_mask)


def sum_layer_td_pflow(layer, node_flows, params, param_flows, scale):
    block_size = layer.block_size
    for partition_id in range(layer.num_fw_partitions):
        nids = layer.partitioned_nids[partition_id]
        cids = layer.partitioned_cids[partition_id]
        pids = layer.partitioned_pids[partition_id]
        pfids = layer.partitioned_pfids[partition_id]

        n_num_nodes = nids.size(0) * block_size
        c_num_nodes = cids.size(1)

        TILE_SIZE_N = min(32, triton.next_power_of_2(n_num_nodes))
        TILE_SIZE_C = min(32, triton.next_power_of_2(c_num_nodes))

        grid = (triton.cdiv(c_num_nodes, TILE_SIZE_C), triton.cdiv(n_num_nodes, TILE_SIZE_N))

        sum_layer_td_pflow_kernel[grid](
            node_flows,
            params,
            param_flows,
            nids,
            cids,
            pids,
            pfids,
            scale,
            n_num_nodes = n_num_nodes,
            c_num_nodes = c_num_nodes,
            TILE_SIZE_N = TILE_SIZE_N,
            TILE_SIZE_C = TILE_SIZE_C,
            n_block_size = block_size
        )


def eval_top_down_probs(pc, update_pflow: bool = True, scale: float = 1.0):
    """
    Computes the top-down probabilities of every node.
    We use the first # nodes entries in `pc.node_flows` and the first # element entries in `pc.element_flows`.
    """

    node_flows = pc.node_flows.view(-1) # Reuse the allocated memory
    element_flows = pc.element_flows.view(-1)

    node_flows[:pc.num_nodes] = 0.0
    element_flows[:pc.num_elements] = 0.0

    node_flows[pc._root_node_range[0]:pc._root_node_range[1]] = 1.0

    # Backward pass over the inner layers
    for layer_id in range(len(pc.inner_layer_groups) - 1, -1, -1):
        layer_group = pc.inner_layer_groups[layer_id]

        for layer in layer_group:
            if layer.is_prod():
                prod_layer_td_backward(layer, node_flows, element_flows)

            elif layer.is_sum():
                element_flows[:pc.num_nodes] = 0.0
                sum_layer_td_backward(layer, node_flows, element_flows, pc.params)

                if update_pflow:
                    sum_layer_td_pflow(layer, node_flows, pc.params, pc.param_flows, scale)

    # Backward pass over the input layers
    for layer in pc.input_layer_group:
        layer.add_missing_flows(node_flows, scale = scale)

    return None
