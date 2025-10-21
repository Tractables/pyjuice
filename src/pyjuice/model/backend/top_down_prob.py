from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from pyjuice.layer import SumLayer, ProdLayer, InputLayer
from pyjuice.utils.kernel_launcher import triton_jit

from .eval_partition import eval_partition_fn, prod_layer_partition_fn


@triton_jit
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


@triton_jit
def sum_layer_td_backward_block_kernel(node_flows, element_flows, node_mars, element_mars, mparams, chids, parids, parpids, 
                                       n_num_nodes: tl.constexpr, c_num_nodes: tl.constexpr, TILE_SIZE_N: tl.constexpr, 
                                       TILE_SIZE_C: tl.constexpr, n_block_size: tl.constexpr, c_block_size: tl.constexpr, 
                                       NUM_N_BLKS: tl.constexpr, pc_is_normalized: tl.constexpr):
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
    epars = tl.load(mparams + pids)

    if not pc_is_normalized:
        cmars = tl.load(element_mars + cids) # [TILE_SIZE_C]
        nmars = tl.load(node_mars + nids) # [TILE_SIZE_N]

        epars = tl.exp(tl.log(epars) + cmars[:,None] - nmars[None,:])

    cflows = tl.sum(epars * nflows[None,:], axis = 1)
    tl.atomic_add(element_flows + cids, cflows)


@triton_jit
def sum_layer_td_backward_node_kernel(node_flows, element_flows, node_mars, element_mars, mparams, chids, parids, parpids, 
                                      n_num_nodes: tl.constexpr, c_num_nodes: tl.constexpr, TILE_SIZE_N: tl.constexpr, 
                                      TILE_SIZE_C: tl.constexpr, n_block_size: tl.constexpr, c_block_size: tl.constexpr, 
                                      NUM_N_BLKS: tl.constexpr, pc_is_normalized: tl.constexpr):
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
    epars = tl.load(mparams + pids, mask = cn_mask, other = 0.0)

    if not pc_is_normalized:
        cmars = tl.load(element_mars + cids) # [TILE_SIZE_C]
        nmars = tl.load(node_mars + nids, mask = cn_mask, other = 0.0) # [TILE_SIZE_C, TILE_SIZE_N]

        epars = tl.exp(tl.log(epars) + cmars[:,None] - nmars)

    cflows = tl.sum(epars * nflows, axis = 1)
    tl.atomic_add(element_flows + cids, cflows)


def sum_layer_td_backward(layer, node_flows, element_flows, node_mars, element_mars, params, pc_is_normalized = True):
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
                node_mars,
                element_mars,
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
                NUM_N_BLKS = parids.size(1),
                pc_is_normalized = pc_is_normalized,
            )

        else:
            TILE_SIZE_C = min(32, triton.next_power_of_2(c_num_nodes))
            TILE_SIZE_N = min(32, triton.next_power_of_2(n_num_nodes))

            grid = (triton.cdiv(n_num_nodes, TILE_SIZE_N), triton.cdiv(c_num_nodes, TILE_SIZE_C))

            sum_layer_td_backward_node_kernel[grid](
                node_flows,
                element_flows,
                node_mars,
                element_mars,
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
                NUM_N_BLKS = parids.size(1),
                pc_is_normalized = pc_is_normalized,
            )


@triton_jit
def sum_layer_td_pflow_kernel(node_flows, node_mars, element_mars, mparams, param_flows, nids, cids, pids, pfids, hyperparameters,
                              n_num_nodes: tl.constexpr, c_num_nodes: tl.constexpr, TILE_SIZE_N: tl.constexpr, TILE_SIZE_C: tl.constexpr,
                              n_block_size: tl.constexpr, pc_is_normalized: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    scale = tl.load(hyperparameters)

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
    epars = tl.load(mparams + epids, mask = nc_mask, other = 0.0)

    if not pc_is_normalized:
        nmars = tl.load(node_mars + snids, mask = n_mask, other = 0.0)

        cids_base = tl.load(cids + ngroup_ids[:,None] * c_num_nodes + offs_cnode[None,:], mask = c_mask[None,:])
        cmars = tl.load(element_mars + cids_base, mask = c_mask[None,:], other = 0.0) # [TILE_SIZE_N, TILE_SIZE_C]

        epars = tl.exp(tl.log(epars) + cmars - nmars[:,None])

    eflows = epars * nflows[:,None] * scale

    pfids_base = tl.load(pfids + ngroup_ids[:,None] * c_num_nodes + offs_cnode[None,:], mask = nc_mask)
    epfids = pfids_base + offs_nnode[:,None] # [TILE_SIZE_N, TILE_SIZE_C]
    tl.atomic_add(param_flows + epfids, eflows, mask = nc_mask)


def sum_layer_td_pflow(layer, node_flows, node_mars, element_mars, params, param_flows, hyperparameters, pc_is_normalized = True):
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
            node_mars,
            element_mars,
            params,
            param_flows,
            nids,
            cids,
            pids,
            pfids,
            hyperparameters,
            n_num_nodes = n_num_nodes,
            c_num_nodes = c_num_nodes,
            TILE_SIZE_N = TILE_SIZE_N,
            TILE_SIZE_C = TILE_SIZE_C,
            n_block_size = block_size,
            pc_is_normalized = pc_is_normalized
        )


def eval_top_down_probs(pc, update_pflow: bool = True, scale: float = 1.0, pc_is_normalized: bool = True, use_cudagraph: bool = True):
    """
    Computes the top-down probabilities of every node.
    We use the first # nodes entries in `pc.node_flows` and the first # element entries in `pc.element_flows`.
    """
    assert hasattr(pc, "node_flows") and hasattr(pc, "element_flows")
    assert pc.node_flows.numel() >= pc.num_nodes
    assert pc.element_flows.numel() >= pc.num_elements

    if not hasattr(pc, "_single_node_flows"):
        pc._single_node_flows = torch.zeros([pc.num_nodes], dtype = torch.float32, device = pc.node_flows.device)

    if not hasattr(pc, "_single_element_flows"):
        pc._single_element_flows = torch.zeros([pc.num_elements], dtype = torch.float32, device = pc.element_flows.device)

    if not hasattr(pc, "_tdp_hyperparams"):
        pc._tdp_hyperparams = torch.zeros([1], dtype = torch.float32, device = pc.node_flows.device)

    node_flows = pc._single_node_flows
    element_flows = pc._single_element_flows
    hyperparameters = pc._tdp_hyperparams

    node_flows[:] = 0.0
    element_flows[:] = 0.0

    node_flows[pc._root_node_range[0]:pc._root_node_range[1]] = 1.0

    # Load hyperparameter(s)
    hyperparameters[0] = scale

    if not pc_is_normalized:
        # Run a forward pass to evaluate the partition function of every node
        eval_partition_fn(pc)
        node_mars = pc.node_mars.view(-1)
        element_mars = pc.element_mars.view(-1)
    else:
        node_mars, element_mars = None, None

    def run_tdp_backward():
        # Backward pass over the inner layers
        for layer_id in range(len(pc.inner_layer_groups) - 1, -1, -1):
            layer_group = pc.inner_layer_groups[layer_id]

            if not pc_is_normalized and layer_group.is_sum():
                prod_layer_group = pc.inner_layer_groups[layer_id]
                for layer in prod_layer_group:
                    prod_layer_partition_fn(layer, node_mars, element_mars, pc.params)

            for layer in layer_group:
                if layer.is_prod():
                    prod_layer_td_backward(layer, node_flows, element_flows)

                elif layer.is_sum():
                    element_flows[:] = 0.0
                    sum_layer_td_backward(layer, node_flows, element_flows, node_mars, element_mars, pc.params, 
                                          pc_is_normalized = pc_is_normalized)

                    if update_pflow:
                        sum_layer_td_pflow(layer, node_flows, node_mars, element_mars, pc.params, pc.param_flows, hyperparameters,
                                           pc_is_normalized = pc_is_normalized)

    if not hasattr(pc, "_tdp_cudagraph"):
        pc._tdp_cudagraph = dict()

    key = (update_pflow, pc_is_normalized, (None if node_mars is None else id(node_mars)), 
           (None if element_mars is None else id(element_mars)), id(node_flows), id(pc.params), 
           id(pc.param_flows), id(hyperparameters))
    if use_cudagraph and key in pc._tdp_cudagraph:
        g = pc._tdp_cudagraph[key]
        g.replay()
    elif not use_cudagraph:
        run_tdp_backward()
    else:
        # Backup param flows
        backup_node_flows = node_flows.detach().cpu().clone()
        backup_element_flows = element_flows.detach().cpu().clone()
        if update_pflow:
            backup_param_flows = pc.param_flows.detach().cpu().clone()
            backup_input_param_flows = []
            for layer in pc.input_layer_group:
                backup_input_param_flows.append(layer.param_flows.detach().cpu().clone())

        # Record CUDAGraph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run_tdp_backward()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_tdp_backward()

        pc._tdp_cudagraph[key] = g

        # Restore param_flows (no extra GPU allocation)
        node_flows.copy_(backup_node_flows, non_blocking = True)
        element_flows.copy_(backup_element_flows, non_blocking = True)
        if update_pflow:
            pc.param_flows.copy_(backup_param_flows, non_blocking = True)
            for layer, backup_pfs in zip(pc.input_layer_group, backup_input_param_flows):
                layer.param_flows.copy_(backup_pfs, non_blocking = True)
        torch.cuda.synchronize()
        
        g.replay()

    # Backward pass over the input layers
    if update_pflow:
        for layer in pc.input_layer_group:
            layer.add_missing_flows(node_flows, scale = scale)

    return None
