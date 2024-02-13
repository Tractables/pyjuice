from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl
from numba import njit
from typing import Sequence

from pyjuice.nodes import CircuitNodes


@njit
def _record_par_blks(par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs,
                     num_edges_per_ng, ns_num_node_blocks, ns_block_size, cs_block_size, pid, 
                     global_nid, par_start, pflow_start, BLOCK_SIZE):
    for local_ngid in range(ns_num_node_blocks):
        num_edges = num_edges_per_ng[local_ngid]
        num_chs = num_edges * cs_block_size

        for sid in range(0, num_chs, BLOCK_SIZE):
            eid = min(sid + BLOCK_SIZE, num_chs)
            blk_size = eid - sid

            for gid in range(ns_block_size):
                psid = par_start + sid * ns_block_size + gid
                pfsid = pflow_start + sid * ns_block_size + gid
                global_ind = global_nid + gid

                par_start_ids[pid] = par_start + sid * ns_block_size + gid
                pflow_start_ids[pid] = pflow_start + sid * ns_block_size + gid
                blk_sizes[pid] = blk_size
                blk_intervals[pid] = ns_block_size
                global_nids[pid] = global_nid + gid
                nchs[pid] = num_edges * cs_block_size

                pid += 1

        pflow_start += ns_block_size * num_edges * cs_block_size
        par_start += ns_block_size * num_edges * cs_block_size
        global_nid += ns_block_size

    return global_nid, pid


@torch.no_grad()
def compile_par_update_fn(root_ns: CircuitNodes, BLOCK_SIZE: int = 32, buffer_inc_interval: int = 10000, use_numba: bool = True):

    assert BLOCK_SIZE & (BLOCK_SIZE - 1) == 0, "`BLOCK_SIZE` must be a power of 2."

    par_start_ids = np.zeros([buffer_inc_interval], dtype = np.int64)
    pflow_start_ids = np.zeros([buffer_inc_interval], dtype = np.int64)
    blk_sizes = np.zeros([buffer_inc_interval], dtype = np.int64)
    blk_intervals = np.zeros([buffer_inc_interval], dtype = np.int64)
    global_nids = np.zeros([buffer_inc_interval], dtype = np.int64)
    nchs = np.zeros([buffer_inc_interval], dtype = np.int64)
    pid = 0

    global_nid = 0
    for i, ns in enumerate(root_ns):
        if not ns.is_sum() or ns.is_tied():
            continue

        par_start = ns._param_range[0]
        pflow_start = ns._param_flow_range[0]
        tot_n_pars = ns._param_range[1] - ns._param_range[0]

        num_edges_per_ng = torch.bincount(ns.edge_ids[0,:], minlength = ns.num_node_blocks).contiguous().numpy()

        # Enlarge the buffer if needed
        est_num_slots = triton.cdiv(ns.edge_ids.size(1) * ns.block_size * ns.ch_block_size, BLOCK_SIZE) + ns.num_nodes
        if pid + est_num_slots > par_start_ids.shape[0]:
            curr_size = par_start_ids.shape[0]
            inc_shape = triton.cdiv(pid + est_num_slots - curr_size, buffer_inc_interval) * buffer_inc_interval

            par_start_ids_new = np.zeros([curr_size + inc_shape], dtype = np.int64)
            par_start_ids_new[:curr_size] = par_start_ids[:curr_size]
            par_start_ids = par_start_ids_new

            pflow_start_ids_new = np.zeros([curr_size + inc_shape], dtype = np.int64)
            pflow_start_ids_new[:curr_size] = pflow_start_ids[:curr_size]
            pflow_start_ids = pflow_start_ids_new

            blk_sizes_new = np.zeros([curr_size + inc_shape], dtype = np.int64)
            blk_sizes_new[:curr_size] = blk_sizes[:curr_size]
            blk_sizes = blk_sizes_new

            blk_intervals_new = np.zeros([curr_size + inc_shape], dtype = np.int64)
            blk_intervals_new[:curr_size] = blk_intervals[:curr_size]
            blk_intervals = blk_intervals_new

            global_nids_new = np.zeros([curr_size + inc_shape], dtype = np.int64)
            global_nids_new[:curr_size] = global_nids[:curr_size]
            global_nids = global_nids_new

            nchs_new = np.zeros([curr_size + inc_shape], dtype = np.int64)
            nchs_new[:curr_size] = nchs[:curr_size]
            nchs = nchs_new

            buffer_inc_interval *= 2

        if use_numba:
            ns_num_node_blocks = ns.num_node_blocks
            ns_block_size = ns.block_size
            cs_block_size = ns.ch_block_size

            global_nid, pid = _record_par_blks(
                par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs,
                num_edges_per_ng, ns_num_node_blocks, ns_block_size, cs_block_size, pid, 
                global_nid, par_start, pflow_start, BLOCK_SIZE
            )

        else:
            ns_gid_range = torch.arange(0, ns.block_size)

            for local_ngid in range(ns.num_node_blocks):
                num_edges = num_edges_per_ng[local_ngid]
                num_chs = num_edges * ns.ch_block_size

                for sid in range(0, num_chs, BLOCK_SIZE):
                    eid = min(sid + BLOCK_SIZE, num_chs)
                    blk_size = eid - sid

                    curr_psids = par_start + sid * ns.block_size + ns_gid_range
                    curr_pfsids = pflow_start + sid * ns.block_size + ns_gid_range
                    curr_global_nids = global_nid + ns_gid_range

                    par_start_ids[pid:pid+ns.block_size] = curr_psids
                    pflow_start_ids[pid:pid+ns.block_size] = curr_pfsids
                    blk_sizes[pid:pid+ns.block_size] = blk_size
                    blk_intervals[pid:pid+ns.block_size] = ns.block_size
                    global_nids[pid:pid+ns.block_size] = curr_global_nids
                    nchs[pid:pid+ns.block_size] = num_edges * ns.ch_block_size

                    pid += ns.block_size

                par_start += ns.block_size * num_edges * ns.ch_block_size
                pflow_start += ns.block_size * num_edges * ns.ch_block_size
                global_nid += ns.block_size

    par_start_ids = torch.from_numpy(par_start_ids[:pid]).contiguous()
    pflow_start_ids = torch.from_numpy(pflow_start_ids[:pid]).contiguous()
    blk_sizes = torch.from_numpy(blk_sizes[:pid]).contiguous()
    blk_intervals = torch.from_numpy(blk_intervals[:pid]).contiguous()
    global_nids = torch.from_numpy(global_nids[:pid]).contiguous()
    nchs = torch.from_numpy(nchs[:pid]).contiguous()

    cum_pflows = torch.zeros([global_nids[-1] + 1], dtype = torch.float32)
    
    metadata = {"tot_num_nodes": global_nids[-1].item() + 1, "BLOCK_SIZE": BLOCK_SIZE}

    return [par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata]


def par_update_to_device(par_update_kwargs, device):

    par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata = par_update_kwargs

    return [
        par_start_ids.to(device),
        pflow_start_ids.to(device),
        blk_sizes.to(device),
        blk_intervals.to(device),
        global_nids.to(device),
        nchs.to(device),
        cum_pflows.to(device),
        metadata
    ]


@triton.jit
def cum_pflow_kernel(cum_pflows, params, param_flows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, 
                     global_nids, constexprs, num_blocks, keep_zero_params: tl.constexpr, BLOCK_ID: tl.constexpr, 
                     BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)

    # Retrieve the constants
    pseudocount = tl.load(constexprs + 1)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    pflow_start = tl.load(pflow_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)
    global_nid = tl.load(global_nids + offs_m, mask = mask_m, other = 0)

    offs_pflow = pflow_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_pflow = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    pflows = tl.load(param_flows + offs_pflow, mask = mask_pflow, other = 0)

    if keep_zero_params == 1:
        par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
        offs_par = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
        old_params = tl.load(params + offs_par, mask = mask_pflow, other = 0)

        nch = tl.load(nchs + global_nid, mask = mask_m, other = 1)
        pflows += (pseudocount / nch[:,None])

        nflows = tl.sum(tl.where(old_params < 1e-12, 0.0, pflows), axis = 1)
    else:
        nflows = tl.sum(pflows, axis = 1)

    tl.atomic_add(cum_pflows + global_nid, nflows, mask = mask_m)


@triton.jit
def par_update_kernel(params, param_flows, cum_pflows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals,
                      global_nids, constexprs, num_blocks, keep_zero_params: tl.constexpr, BLOCK_ID: tl.constexpr, 
                      BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)

    # Retrieve the constants
    step_size = tl.load(constexprs)
    pseudocount = tl.load(constexprs + 1)

    offs_m = pid * BLOCK_ID + tl.arange(0, BLOCK_ID)
    mask_m = offs_m < num_blocks

    offs_blk = tl.arange(0, BLOCK_SIZE)

    par_start = tl.load(par_start_ids + offs_m, mask = mask_m, other = 0)
    pflow_start = tl.load(pflow_start_ids + offs_m, mask = mask_m, other = 0)
    blk_size = tl.load(blk_sizes + offs_m, mask = mask_m, other = 0)
    blk_interval = tl.load(blk_intervals + offs_m, mask = mask_m, other = 0)
    global_nid = tl.load(global_nids + offs_m, mask = mask_m, other = 0)

    offs_pflow = pflow_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    mask_pflow = mask_m[:,None] & (offs_blk[None,:] < blk_size[:,None])
    pflows = tl.load(param_flows + offs_pflow, mask = mask_pflow, other = 0)

    nflows = tl.load(cum_pflows + global_nid, mask = mask_m, other = 1)
    nch = tl.load(nchs + global_nid, mask = mask_m, other = 1)

    if keep_zero_params == 1:
        new_param = (pflows + pseudocount / nch[:,None]) / nflows[:,None]
    else:
        new_param = (pflows + pseudocount / nch[:,None]) / (nflows[:,None] + pseudocount)

    offs_par = par_start[:,None] + offs_blk[None,:] * blk_interval[:,None]
    old_param = tl.load(params + offs_par, mask = mask_pflow, other = 0)

    updated_param = (1.0 - step_size) * old_param + step_size * new_param

    if keep_zero_params == 1:
        updated_params = tl.where(old_param < 1e-12, 0.0, updated_param)

    tl.store(params + offs_par, updated_param, mask = mask_pflow)


def em_par_update(params: torch.Tensor, param_flows: torch.Tensor, par_update_kwargs: Sequence, 
                  step_size: float, pseudocount: float = 0.0, keep_zero_params: bool = True):

    par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, global_nids, nchs, cum_pflows, metadata = par_update_kwargs

    tot_num_nodes = metadata["tot_num_nodes"]
    BLOCK_SIZE = metadata["BLOCK_SIZE"]

    if cum_pflows is None:
        cum_pflows = torch.zeros([tot_num_nodes], dtype = torch.float32, device = params.device)
    else:
        cum_pflows[:] = 0.0

    num_blocks = par_start_ids.size(0)
    BLOCK_ID = 2048 // BLOCK_SIZE

    grid = (triton.cdiv(num_blocks, BLOCK_ID),)

    constexprs = torch.tensor([step_size, pseudocount]).to(params.device)

    keep_zero_params = 1 if keep_zero_params else 0

    cum_pflow_kernel[grid](
        cum_pflows, params, param_flows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals, 
        global_nids, constexprs, num_blocks, keep_zero_params, BLOCK_ID, BLOCK_SIZE
    )

    par_update_kernel[grid](
        params, param_flows, cum_pflows, nchs, par_start_ids, pflow_start_ids, blk_sizes, blk_intervals,
        global_nids, constexprs, num_blocks, keep_zero_params, BLOCK_ID, BLOCK_SIZE
    )
