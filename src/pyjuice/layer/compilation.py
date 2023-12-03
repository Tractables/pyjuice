from __future__ import annotations

import numpy as np
import torch
import threading
import functools
import os
import warnings
import time
import triton
import triton.language as tl
from numba import njit
from copy import deepcopy
from typing import Optional, Sequence

from pyjuice.nodes import CircuitNodes, SumNodes


## Helper functions ##


def flatten_sum_nodes(ns: SumNodes, *args, use_cuda: bool = False):
    edge_ids = ns.edge_ids
    if use_cuda:
        edge_ids = edge_ids.cuda()

    if not ns.is_tied():
        return (ns.num_nodes, edge_ids, ns._param_range, [(c.num_nodes, c._output_ind_range) for c in ns.chs], *args)
    else:
        source_ns = ns.get_source_ns()
        return (ns.num_nodes, edge_ids, source_ns._param_range, [(c.num_nodes, c._output_ind_range) for c in ns.chs], *args)


def get_chunk_ids(n, k):
    chunk_size = n // k
    remainder = n % k

    chunks = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + chunk_size + 1
        else:
            end = start + chunk_size

        chunks.append((start, end))
        start = end

    return chunks


def next_power_of_2(x: torch.Tensor):
    return torch.where(
        x == 1,
        1,
        2 ** torch.ceil(torch.log2(x.float())).long()
    )


## Compilation for SumLayer ##


def get_sum_layer_stats(nodes: Sequence[SumNodes], global_nid_start: int):
    layer_num_nodes = sum(map(lambda ns: ns.num_nodes, nodes))
    layer_num_edges = 0

    n_sid = 0
    n_chs = torch.zeros([layer_num_nodes], dtype = torch.long)
    for ns_idx, ns in enumerate(nodes):
        n_eid = n_sid + ns.num_nodes

        curr_n_chs = torch.bincount(ns.edge_ids[0,:])
        n_chs[n_sid:n_eid] = curr_n_chs

        ns._output_ind_range = (global_nid_start, global_nid_start + ns.num_nodes)
        global_nid_start += ns.num_nodes
        layer_num_edges += ns.edge_ids.size(1)

        n_sid = n_eid

    return layer_num_nodes, layer_num_edges, n_chs


@torch.no_grad()
def sum_layer_forward_compilation_job(flat_nodes, nids, cids, pids, fw_group_max_chs, n_group_ids, n_id_in_group,
                                      global_nid_start, ch_prod_layer_size, job_start, job_end, return_dict = None, 
                                      idx = 0, use_cuda: bool = False):
    """
    Note: Only process jobs in [job_start, job_end).
    """
    all_ns_param_ids = dict()

    node_start = 0
    for ns_idx, flat_ns in enumerate(flat_nodes):
        # Outer iteration over `ns` in this layer
        ns_num_nodes = flat_ns[0]
        if node_start + ns_num_nodes < job_start:
            node_start += ns_num_nodes
            continue # Move on to the next ns
        elif node_start >= job_end:
            break # All jobs completed

        edge_ids = flat_ns[1] # Edge indices of this `ns`
        ns_num_edges = edge_ids.size(1)

        add_params_flag = flat_ns[4]
        if add_params_flag:
            ns_param_ids = torch.zeros([edge_ids.size(1)], dtype = torch.long, device = edge_ids.device)

        # Pre-compute cid flags for future reuse
        num_chs = len(flat_ns[3])
        cid_starts = torch.zeros([num_chs], dtype = torch.long)
        cid_ends = torch.zeros([num_chs], dtype = torch.long)
        cid_start = 0
        for cnode_id, flat_cs in enumerate(flat_ns[3]):
            cs_num_nodes = flat_cs[0]
            cid_end = cid_start + cs_num_nodes
            cid_starts[cnode_id] = cid_start
            cid_ends[cnode_id] = cid_end

            cid_start = cid_end
        
        if use_cuda:
            cid_starts = cid_starts.cuda()
            cid_ends = cid_ends.cuda()

        # Shape: [num_chs, num_edges]
        cs_criterion = (edge_ids[1,:].unsqueeze(0) >= cid_starts[:,None]) & \
                       (edge_ids[1,:].unsqueeze(0) < cid_ends[:,None])

        # Loop over the nodes assigned to the current thread
        nid_start = 0 if node_start >= job_start else job_start - node_start
        nid_end = ns_num_nodes if node_start + ns_num_nodes <= job_end else job_end - node_start
        ns_pid_start = flat_ns[2][0] # Start param id
        ns_local_pid = (edge_ids[0,:] < nid_start).sum().item()
        for nid in range(nid_start, nid_end):
            # Global node idx
            global_nid = global_nid_start + node_start + nid

            # `group_id`:   which group the current node belongs to
            # `local_id`:   the index of the node within the current group
            # `group_nchs`: maximum number of child nodes in the current group
            group_id = n_group_ids[node_start + nid]
            local_id = n_id_in_group[node_start + nid]
            group_nchs = fw_group_max_chs[group_id]

            ns_criterion = (edge_ids[0,:] == nid)

            # assign node id
            nids[group_id][local_id] = global_nid

            ch_start = 0
            cid_start = 0
            for cnode_id, flat_cs in enumerate(flat_ns[3]):
                cs_num_nodes = flat_cs[0]
                cs_out_ind_range = flat_cs[1]
                cid_end = cid_start + cs_num_nodes

                criterion = cs_criterion[cnode_id,:] & ns_criterion

                # assign child ids
                ch_ids = edge_ids[1,criterion] + (cs_out_ind_range[0] - cid_start)
                cids[group_id][local_id,ch_start:ch_start+ch_ids.size(0)] = ch_ids

                # mapping from the current params to global params
                if add_params_flag:
                    curr_ids = torch.where(criterion)[0]
                    curr_param_ids = torch.arange(curr_ids.size(0), device = edge_ids.device) + (ns_pid_start + ns_local_pid + ch_start)
                    ns_param_ids[curr_ids] = curr_param_ids

                ch_start += ch_ids.size(0)
                cid_start = cid_end

            # assign parameter ids
            parids = torch.arange(ch_start, device = edge_ids.device) + (ns_pid_start + ns_local_pid)
            pids[group_id][local_id,:ch_start] = parids

            ns_local_pid += ch_start

        node_start += ns_num_nodes
        ns_pid_start += ns_num_edges

        if add_params_flag:
            all_ns_param_ids[ns_idx] = ns_param_ids

    if return_dict is not None:
        return_dict[idx] = all_ns_param_ids
    else:
        return all_ns_param_ids


@torch.no_grad()
def sum_layer_forward_compilation_legacy(nodes, fw_group_max_chs, n_group_ids, n_id_in_group, num_ns_in_group, n_chs,
                                         global_nid_start, ch_prod_layer_size, param_ends, 
                                         num_threads: int = 1, use_cuda: bool = False):

    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    total_num_jobs = sum(map(lambda ns: ns.num_nodes, nodes))

    # Construct flattened_nodes
    global_pid_start = param_ends[-1]
    flat_nodes = []
    add_ns_params_flag = []
    for ns in nodes:
        if ns.is_tied():
            source_ns = ns.get_source_ns()
            if not hasattr(source_ns, "_param_range") or source_ns._param_range is None:
                global_pid_end = global_pid_start + source_ns.num_edges
                source_ns._param_range = (global_pid_start, global_pid_end)
                global_pid_start = global_pid_end

                add_params_flag = True
            else:
                add_params_flag = False
        else:
            if not hasattr(ns, "_param_range") or ns._param_range is None:
                global_pid_end = global_pid_start + ns.num_edges
                ns._param_range = (global_pid_start, global_pid_end)
                global_pid_start = global_pid_end

                add_params_flag = True
            else:
                add_params_flag = False

        add_ns_params_flag.append(add_params_flag)
        flat_nodes.append(flatten_sum_nodes(ns, add_params_flag, use_cuda = use_cuda))

    # Allocate target buffers
    nids = [torch.zeros([group_size], dtype = torch.long) for group_size in num_ns_in_group] # Node id
    cids = [torch.zeros([group_size, max_chs], dtype = torch.long) for group_size, max_chs in zip(num_ns_in_group, fw_group_max_chs)] # Child id
    pids = [torch.zeros([group_size, max_chs], dtype = torch.long) for group_size, max_chs in zip(num_ns_in_group, fw_group_max_chs)] # Parameter id

    if use_cuda:
        nids = [tensor.cuda() for tensor in nids]
        cids = [tensor.cuda() for tensor in cids]
        pids = [tensor.cuda() for tensor in pids]

    if num_threads == 1:
        curr_ns_param_ids = sum_layer_forward_compilation_job(
            flat_nodes, nids, cids, pids, fw_group_max_chs, n_group_ids, n_id_in_group,
            global_nid_start, ch_prod_layer_size, 0, total_num_jobs, use_cuda = use_cuda
        )
        all_ns_param_ids = [curr_ns_param_ids]

    else:
        job_indices = get_chunk_ids(total_num_jobs, num_threads)

        threads = []
        return_dict = dict()
        for idx, (job_start, job_end) in enumerate(job_indices):
            th = threading.Thread(
                target = sum_layer_forward_compilation_job, 
                args = (flat_nodes, nids, cids, pids, fw_group_max_chs, n_group_ids, n_id_in_group,
                        global_nid_start, ch_prod_layer_size, job_start, job_end, return_dict, idx, 
                        use_cuda)
            )
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

        all_ns_param_ids = []
        for idx in range(num_threads):
            curr_ns_param_ids = return_dict[idx]
            all_ns_param_ids.append(curr_ns_param_ids)

    # Compute the number of (sum) parents for each (prod) input node
    ch_n_pars = torch.zeros([ch_prod_layer_size], dtype = torch.long) # Number of parents for each child node
    for ns in nodes:
        ch_start = 0
        for cs in ns.chs:
            ch_end = ch_start + cs.num_nodes
            criterion = (ns.edge_ids[1,:] >= ch_start) & (ns.edge_ids[1,:] < ch_end)
            
            cs_s_oind = cs._output_ind_range[0]
            cs_e_oind = cs._output_ind_range[1]
            c_ns_counts = torch.bincount(ns.edge_ids[1,criterion] - ch_start, minlength = cs.num_nodes)
            ch_n_pars[cs_s_oind:cs_e_oind] = c_ns_counts

            ch_start = ch_end

    # Store local -> global parameter id mapping in `ns`
    for ns_param_ids in all_ns_param_ids:
        for ns_idx, param_ids in ns_param_ids.items():
            if use_cuda:
                param_ids = param_ids.cpu()
            ns = nodes[ns_idx]
            if not hasattr(ns, "_param_ids") or ns._param_ids is None:
                ns._param_ids = param_ids
            else:
                mask = (param_ids > 0)
                ns._param_ids[mask] = param_ids[mask]

    # Store global -> local parameter id mapping in `ns`
    for ns, add_params_flag in zip(nodes, add_ns_params_flag):
        if add_params_flag:
            ns._param_range = (ns._param_ids.min().item(), ns._param_ids.max().item() + 1)
            ns._inverse_param_ids = torch.argsort(ns._param_ids)

    # Update `param_ends`
    npars = param_ends[-1]
    nid = 0
    for ns, add_params_flag in zip(nodes, add_ns_params_flag):
        if add_params_flag:
            for i in range(ns.num_nodes):
                npars += n_chs[nid+i].item()
                param_ends.append(npars)
        
        nid += ns.num_nodes

    if use_cuda:
        # Move buffers back to CPU
        nids = [tensor.cpu() for tensor in nids]
        cids = [tensor.cpu() for tensor in cids]
        pids = [tensor.cpu() for tensor in pids]

    return nids, cids, pids, ch_n_pars, param_ends


@njit
def _assign_chid_kernel(chs_offsets, ns_nchs, edge_ids):
    for i in range(edge_ids.shape[1]):
        nid = edge_ids[0,i]
        idx = ns_nchs[nid]
        chs_offsets[i] = idx
        ns_nchs[nid] = idx + 1


@triton.jit
def _assign_target_ncpids_kernel(target_nids_ptr, nids_group_start_ptr, target_cids_ptr, pcids_group_start_ptr,
                                 target_pids_ptr, edge_ids_ptr, chs_offsets_ptr, n_group_ids_ptr, n_id_in_group_ptr, 
                                 cs_ele_id_start_ptr, cs_node_cum_ids_ptr, fw_group_max_chs_ptr, cum_n_chs_ptr, 
                                 ns_param_ids_ptr, ch_n_pars_ptr, constexprs_ptr, num_chs: tl.constexpr, 
                                 num_chs_np2: tl.constexpr, add_params_flag: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    global_nid_start = tl.load(constexprs_ptr)
    ns_pid_start = tl.load(constexprs_ptr + 1)
    node_start = tl.load(constexprs_ptr + 2)
    num_edges = tl.load(constexprs_ptr + 3)

    # Get edge indices to be processed by the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Get `nid` and `cid`
    nid = tl.load(edge_ids_ptr + offsets, mask = mask, other = 0)
    cid = tl.load(edge_ids_ptr + offsets + num_edges, mask = mask, other = 0)

    # Get `group_id` and `local_id`
    group_id = tl.load(n_group_ids_ptr + nid + node_start, mask = mask, other = 0)
    local_id = tl.load(n_id_in_group_ptr + nid + node_start, mask = mask, other = 0)

    # Get the child ns index every `cid` belongs to and the cum nodes & global sid
    cs_offsets = tl.arange(0, num_chs_np2)
    cs_node_cum_ids = tl.load(cs_node_cum_ids_ptr + cs_offsets, mask = (cs_offsets < num_chs), other = 0)
    
    cid_node_id = tl.sum(tl.broadcast_to(cid[:,None], (BLOCK_SIZE, num_chs_np2)) >= \
        tl.broadcast_to(cs_node_cum_ids[None,:], (BLOCK_SIZE, num_chs_np2)), axis = 1) - \
        (1 + num_chs_np2 - num_chs)

    cs_cum_num = tl.load(cs_node_cum_ids_ptr + cid_node_id, mask = mask, other = 0)
    cs_ele_ind = tl.load(cs_ele_id_start_ptr + cid_node_id, mask = mask, other = 0)

    # Get child offsets
    chs_offset = tl.load(chs_offsets_ptr + offsets, mask = mask, other = 0)

    # Store to `target_nids`
    nids_start = tl.load(nids_group_start_ptr + group_id, mask = mask, other = 0)
    global_nid = global_nid_start + node_start + nid
    tl.store(target_nids_ptr + nids_start + local_id, global_nid, mask = mask)

    # Store to `target_cids`
    group_max_n_chs = tl.load(fw_group_max_chs_ptr + group_id, mask = mask, other = 0)
    pcids_start = tl.load(pcids_group_start_ptr + group_id, mask = mask, other = 0)
    pcids_offsets = pcids_start + local_id * group_max_n_chs + chs_offset
    global_cid = cid + cs_ele_ind - cs_cum_num
    tl.store(target_cids_ptr + pcids_offsets, global_cid, mask = mask)

    # Cumulate number of parents for every child node
    tl.atomic_add(ch_n_pars_ptr + global_cid, 1, mask = mask)

    # Store to `target_pids`
    ns_local_pid = tl.load(cum_n_chs_ptr + nid, mask = mask, other = 0)
    global_pid = chs_offset + ns_pid_start + ns_local_pid
    tl.store(target_pids_ptr + pcids_offsets, global_pid, mask = mask)

    # Global parameter indices for all edges
    if add_params_flag:
        tl.store(ns_param_ids_ptr + offsets, global_pid, mask = mask)


@torch.no_grad()
def sum_layer_forward_compilation(nodes, fw_group_max_chs, n_group_ids, n_id_in_group, num_ns_in_group, n_chs,
                                  global_nid_start, ch_prod_layer_size, param_ends,
                                  num_threads: int = 1, use_cuda: bool = True, legacy: bool = False):

    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    # Also use the legacy code if we compile with CPU
    if not use_cuda or legacy:
        return sum_layer_forward_compilation_legacy(
            nodes, fw_group_max_chs, n_group_ids, n_id_in_group, num_ns_in_group, n_chs,
            global_nid_start, ch_prod_layer_size, param_ends, num_threads = num_threads,
            use_cuda = use_cuda
        )

    # We construct a flattened version of `nids` where the vectors of every group is concatenated
    # into a single vector. `nids_group_start` is used to indicate the start index of every group's
    # `nids`. That is, `target_nids[nids_group_start[gid]:nids_group_start[gid+1]] == nids[gid]`
    nids_group_start = torch.zeros_like(num_ns_in_group)
    nids_group_start[1:] = torch.cumsum(num_ns_in_group[:-1], dim = 0)
    target_nids = torch.zeros([num_ns_in_group.sum()], dtype = torch.long).cuda()

    # Similarly, we flatten `cids`...
    pcids_group_start = torch.zeros_like(num_ns_in_group)
    pcids_group_start[1:] = torch.cumsum((num_ns_in_group * fw_group_max_chs)[:-1], dim = 0)
    target_cids = torch.zeros([(num_ns_in_group * fw_group_max_chs).sum()], dtype = torch.long).cuda()

    # ...and `pids`
    target_pids = torch.zeros([(num_ns_in_group * fw_group_max_chs).sum()], dtype = torch.long).cuda()

    # This tensor is to be filled with number of parents for every child node
    ch_n_pars = torch.zeros([ch_prod_layer_size], dtype = torch.int32).cuda()

    # Move necessary tensors to GPU
    n_group_ids = n_group_ids.cuda()
    n_id_in_group = n_id_in_group.cuda()
    fw_group_max_chs = fw_group_max_chs.cuda()

    all_ns_param_ids = dict()
    original_param_nids = []

    # This is the main loop: iterate over `ns` in the layer
    global_pid_start = param_ends[-1]
    node_start = 0 # The start index of nodes in the current `ns`
    for ns_idx, ns in enumerate(nodes):
        if ns.is_tied():
            target_ns = ns.get_source_ns()
        else:
            target_ns = ns

        # If the parameters have not been instantiated, do it :)
        if not hasattr(target_ns, "_param_range") or target_ns._param_range is None:
            global_pid_end = global_pid_start + target_ns.num_edges
            target_ns._param_range = (global_pid_start, global_pid_end)
            global_pid_start = global_pid_end

            add_params_flag = True
            original_param_nids.append(ns_idx)
        else:
            add_params_flag = False

        # Global pid start index for `ns`
        ns_pid_start = target_ns._param_range[0]

        # number of nodes
        ns_num_nodes = ns.num_nodes

        # Edge indices of size [2, ns_num_edges]
        edge_ids = ns.edge_ids
        ns_num_edges = edge_ids.size(1)

        # Precompute the child offset ids for every edge. That is, the `?` 
        # mark in `cids[group_id][local_id,?]`
        chs_offsets = np.zeros([ns_num_edges], dtype = np.int64)
        ns_nchs = np.zeros([ns_num_nodes], dtype = np.int64)

        _assign_chid_kernel(chs_offsets, ns_nchs, edge_ids.numpy())
        chs_offsets = torch.from_numpy(chs_offsets)

        # Construct helper indices for child nodes
        # `cs_ele_id_start` contains the global start indices of the child nodes
        # `cs_node_cum_ids` contains the local cumulative number of child nodes
        cs_ele_id_start = torch.zeros([ns.num_chs], dtype = torch.long)
        cs_node_cum_ids = torch.zeros([ns.num_chs], dtype = torch.long)
        for i, cs in enumerate(ns.chs):
            cs_ele_id_start[i] = cs._output_ind_range[0]
            if i < ns.num_chs - 1:
                cs_node_cum_ids[i+1] = cs_node_cum_ids[i] + cs.num_nodes

        # Cumulative nchs
        ns_nchs = torch.from_numpy(ns_nchs)
        cum_n_chs = torch.zeros([ns_num_nodes], dtype = torch.long)
        cum_n_chs[1:] = torch.cumsum(ns_nchs[:-1], dim = 0)

        if add_params_flag:
            ns_param_ids = torch.zeros([ns_num_edges], dtype = torch.long).cuda()
        else:
            ns_param_ids = None

        # The following kernel assigns the corresponding indices to `nids`, `cids`, and `pids`
        # We first move necessary buffers to GPU
        nids_group_start = nids_group_start.cuda()
        edge_ids = edge_ids.cuda()
        chs_offsets = chs_offsets.cuda()
        cs_ele_id_start = cs_ele_id_start.cuda()
        cs_node_cum_ids = cs_node_cum_ids.cuda()
        cum_n_chs = cum_n_chs.cuda()
        pcids_group_start = pcids_group_start.cuda()

        # We store these constants in a tensor and retrieve them in the kernel
        # This is to avoid `triton` from compiling separate kernels for every layer configuration
        # Saves 99.9% compilation time :)
        constexprs = torch.tensor([global_nid_start, ns_pid_start, node_start, ns_num_edges]).long().cuda()

        # Make the grid and launch kernel
        grid = lambda meta: (triton.cdiv(ns_num_edges, meta["BLOCK_SIZE"]),)

        num_chs_np2 = triton.next_power_of_2(ns.num_chs)
        _assign_target_ncpids_kernel[grid](
            target_nids, nids_group_start, target_cids, pcids_group_start,
            target_pids, edge_ids, chs_offsets, n_group_ids, n_id_in_group, 
            cs_ele_id_start, cs_node_cum_ids, fw_group_max_chs, cum_n_chs, 
            ns_param_ids, ch_n_pars, constexprs, ns.num_chs, num_chs_np2, 
            add_params_flag, BLOCK_SIZE = min(2048, 2**20 // num_chs_np2)
        )

        node_start += ns_num_nodes

        if add_params_flag:
            all_ns_param_ids[ns_idx] = ns_param_ids

    # Store local -> global parameter id mapping in `ns`
    for ns_idx, param_ids in all_ns_param_ids.items():
        ns = nodes[ns_idx]
        ns._param_ids = param_ids.cpu()

    # Store global -> local parameter id mapping in `ns`
    for ns_idx in original_param_nids:
        ns = nodes[ns_idx]
        ns._param_range = (ns._param_ids.min().item(), ns._param_ids.max().item() + 1)
        ns._inverse_param_ids = torch.argsort(ns._param_ids)

    # Update `param_ends`
    npars = param_ends[-1]
    nid = 0
    for ns_idx in original_param_nids:
        ns = nodes[ns_idx]
        for i in range(ns.num_nodes):
            npars += n_chs[nid+i].item()
            param_ends.append(npars)
        
        nid += ns.num_nodes

    # Restore `nids`
    target_nids = target_nids.cpu()
    nids = []
    for group_id in range(num_ns_in_group.size(0)):
        sid = nids_group_start[group_id]
        eid = sid + num_ns_in_group[group_id]
        nids.append(target_nids[sid:eid].contiguous())

    # Restore `cids` and `pids`
    target_cids = target_cids.cpu()
    target_pids = target_pids.cpu()
    cids = []
    pids = []
    for group_id in range(num_ns_in_group.size(0)):
        sid = pcids_group_start[group_id]
        gsize = num_ns_in_group[group_id]
        gnchs = fw_group_max_chs[group_id]
        eid = sid + gsize * gnchs
        cids.append(target_cids[sid:eid].reshape(gsize, gnchs).contiguous())
        pids.append(target_pids[sid:eid].reshape(gsize, gnchs).contiguous())

    # Convert `ch_n_pars` to `torch.long` type
    ch_n_pars = ch_n_pars.cpu().long()

    return nids, cids, pids, ch_n_pars, param_ends


@torch.no_grad()
def sum_layer_backward_compilation_legacy(nodes, pids, fw_n_group_ids, fw_n_id_in_group, 
                                          num_bk_groups, bk_n_group_ids, bk_n_id_in_group, 
                                          bk_group_max_pars, bk_num_ns_in_group,
                                          ch_prod_layer_size, global_nid_start, use_cuda: bool = False):

    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    # Since we will be iterating over parent nodes, we want to create a flattened scratch space for the 
    # buffers. In the following, `flat_parids` and `flat_parpids` are the flattened version of 
    # `parids` and `parpids`, respectively. Also, we create `ch2flatidx` which points to the start 
    # location of the scratch space (`flat_parids` and `flat_parpids``) for every child node.
    group2flatidx = torch.zeros([num_bk_groups], dtype = torch.long)
    flatidx = 0
    for group_id in range(num_bk_groups):
        group_size = bk_num_ns_in_group[group_id]
        max_n_par = bk_group_max_pars[group_id]

        group2flatidx[group_id] = flatidx

        flatidx += group_size * max_n_par
    num_slots = flatidx

    # parids:  indices of parent nodes for each child node
    # parpids: parameter indices for these edges
    flat_parids = torch.zeros([num_slots], dtype = torch.long)
    flat_parpids = torch.zeros([num_slots], dtype = torch.long)

    # The indexing vector pointing to the start position in the scratch space
    ch2flatidx = group2flatidx[bk_n_group_ids] + bk_n_id_in_group * bk_group_max_pars[bk_n_group_ids]

    # This vector maintains the count of parents that have been processed for every child node
    par_counts = torch.zeros([ch_prod_layer_size], dtype = torch.long)

    if use_cuda:
        # Move buffers to GPU
        flat_parids = flat_parids.cuda()
        flat_parpids = flat_parpids.cuda()
        ch2flatidx = ch2flatidx.cuda()
        par_counts = par_counts.cuda()

        fw_n_group_ids = fw_n_group_ids.cuda()
        fw_n_id_in_group = fw_n_id_in_group.cuda()

    node_start = 0
    for ns in nodes:
        node_end = node_start + ns.num_nodes
        if use_cuda:
            edge_ids = ns.edge_ids.cuda()
        else:
            edge_ids = ns.edge_ids

        # Pre-compute cid flags for future reuse
        cid_starts = torch.zeros([ns.num_chs], dtype = torch.long)
        cid_ends = torch.zeros([ns.num_chs], dtype = torch.long)
        cid_start = 0
        for cnode_id, cs in enumerate(ns.chs):
            cid_end = cid_start + cs.num_nodes
            cid_starts[cnode_id] = cid_start
            cid_ends[cnode_id] = cid_end
            cid_start = cid_end
        
        if use_cuda:
            cid_starts = cid_starts.cuda()
            cid_ends = cid_ends.cuda()

        # Shape: [ns.num_chs, num_edges]
        cs_criterion = (edge_ids[1,:].unsqueeze(0) >= cid_starts[:,None]) & \
                       (edge_ids[1,:].unsqueeze(0) < cid_ends[:,None])

        for nid in range(ns.num_nodes):
            # `group_id`: which group the current node belongs to
            # `local_id`: the index of the node within the current group
            group_id = fw_n_group_ids[node_start + nid]
            local_id = fw_n_id_in_group[node_start + nid]
            curr_pids = pids[group_id][local_id,:]
            if use_cuda:
                curr_pids = curr_pids.cuda()

            ns_criterion = (edge_ids[0,:] == nid)

            cid_start = 0
            pid_start = 0
            for cnode_id, cs in enumerate(ns.chs):
                cid_end = cid_start + cs.num_nodes
                criterion = cs_criterion[cnode_id,:] & ns_criterion
                pid_end = pid_start + criterion.sum().item()

                ch_ids = edge_ids[1,criterion] + (cs._output_ind_range[0] - cid_start)
                flat_cids = ch2flatidx[ch_ids] + par_counts[ch_ids] # start position specified by `ch2flatidx` + offset specified by `par_counts`
                flat_parids[flat_cids] = global_nid_start + node_start + nid
                flat_parpids[flat_cids] = curr_pids[pid_start:pid_end]

                par_counts[ch_ids] += 1
                cid_start = cid_end
                pid_start = pid_end

        node_start = node_end

    if use_cuda:
        flat_parids = flat_parids.cpu()
        flat_parpids = flat_parpids.cpu()

    # Restore the original `parids` and `parpids`
    parids = []
    parpids = []
    flatid_start = 0
    for group_id in range(num_bk_groups):
        group_size = bk_num_ns_in_group[group_id]
        max_n_par = bk_group_max_pars[group_id]
        flatid_end = flatid_start + group_size * max_n_par

        parids.append(flat_parids[flatid_start:flatid_end].reshape(group_size, max_n_par).contiguous())
        parpids.append(flat_parpids[flatid_start:flatid_end].reshape(group_size, max_n_par).contiguous())

        flatid_start = flatid_end

    return parids, parpids


@triton.jit
def _assign_global_eleids_kernel(global_ele_ids_ptr, cs_ele_id_start_ptr, cs_node_cum_ids_ptr, edge_ids_ptr, 
                                 constexprs_ptr, num_chs: tl.constexpr, num_chs_np2: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    num_edges = tl.load(constexprs_ptr)

    # Get edge indices to be processed by the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Get `cid`
    cid = tl.load(edge_ids_ptr + offsets + num_edges, mask = mask, other = 0)

    # Get the child ns index every `cid` belongs to and the cum nodes & global sid
    cs_offsets = tl.arange(0, num_chs_np2)
    cs_node_cum_ids = tl.load(cs_node_cum_ids_ptr + cs_offsets, mask = (cs_offsets < num_chs), other = 0)
    
    cid_node_id = tl.sum(tl.broadcast_to(cid[:,None], (BLOCK_SIZE, num_chs_np2)) >= \
        tl.broadcast_to(cs_node_cum_ids[None,:], (BLOCK_SIZE, num_chs_np2)), axis = 1) - \
        (1 + num_chs_np2 - num_chs)

    cs_cum_num = tl.load(cs_node_cum_ids_ptr + cid_node_id, mask = mask, other = 0)
    cs_ele_ind = tl.load(cs_ele_id_start_ptr + cid_node_id, mask = mask, other = 0)

    # Compute global cids and store them
    global_cid = cid + cs_ele_ind - cs_cum_num
    tl.store(global_ele_ids_ptr + offsets, global_cid, mask = mask)


@njit
def _assign_parid_kernel(par_offsets, par_counts, global_ele_ids):
    for i in range(par_offsets.shape[0]):
        global_cid = global_ele_ids[i]
        idx = par_counts[global_cid]
        par_offsets[i] = idx
        par_counts[global_cid] = idx + 1


@triton.jit
def _assign_target_parids_kernel(target_parids_ptr, target_parpids_ptr, parids_group_start_ptr, flat_pids_ptr, pids_group_start_ptr,
                                 edge_ids_ptr, global_ele_ids_ptr, chs_offsets_ptr, par_offsets_ptr,
                                 fw_n_group_ids_ptr, fw_n_id_in_group_ptr, bk_n_group_ids_ptr, bk_n_id_in_group_ptr,
                                 fw_group_max_chs_ptr, bk_group_max_pars_ptr, constexprs_ptr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    global_nid_start = tl.load(constexprs_ptr)
    node_start = tl.load(constexprs_ptr + 1)
    num_edges = tl.load(constexprs_ptr + 2)

    # Get edge indices to be processed by the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Get `nid` and `global_cid`
    nid = tl.load(edge_ids_ptr + offsets, mask = mask, other = 0)
    global_cid = tl.load(global_ele_ids_ptr + offsets, mask = mask, other = 0)

    # Get `fw_group_id` and `fw_local_id` (for indexing `pids`)
    fw_group_id = tl.load(fw_n_group_ids_ptr + nid + node_start, mask = mask, other = 0)
    fw_local_id = tl.load(fw_n_id_in_group_ptr + nid + node_start, mask = mask, other = 0)

    # Get `bk_group_id` and `bk_local_id` (for indexing `parids` and `parpids`)
    bk_group_id = tl.load(bk_n_group_ids_ptr + global_cid, mask = mask, other = 0)
    bk_local_id = tl.load(bk_n_id_in_group_ptr + global_cid, mask = mask, other = 0)

    # Get child offsets (for indexing `pids`) and parent offsets (for indexing `parids` and `parpids`)
    chs_offset = tl.load(chs_offsets_ptr + offsets, mask = mask, other = 0)
    par_offset = tl.load(par_offsets_ptr + offsets, mask = mask, other = 0)
    
    # Store to `target_parids`
    group_max_n_pars = tl.load(bk_group_max_pars_ptr + bk_group_id, mask = mask, other = 0)
    parids_start = tl.load(parids_group_start_ptr + bk_group_id, mask = mask, other = 0)
    parids_offsets = parids_start + bk_local_id * group_max_n_pars + par_offset
    global_nid = global_nid_start + node_start + nid
    tl.store(target_parids_ptr + parids_offsets, global_nid, mask = mask)

    # Get the parameter ids of the edges...
    group_max_n_chs = tl.load(fw_group_max_chs_ptr + fw_group_id, mask = mask, other = 0)
    pids_start = tl.load(pids_group_start_ptr + fw_group_id, mask = mask, other = 0)
    pids_offsets = pids_start + fw_local_id * group_max_n_chs + chs_offset
    pid = tl.load(flat_pids_ptr + pids_offsets, mask = mask, other = 0)

    # ...and store them to `target_parpids`
    tl.store(target_parpids_ptr + parids_offsets, pid, mask = mask)


@torch.no_grad()
def sum_layer_backward_compilation(nodes, pids, fw_n_group_ids, fw_n_id_in_group, 
                                   num_bk_groups, bk_n_group_ids, bk_n_id_in_group, 
                                   fw_group_max_chs, bk_group_max_pars, 
                                   fw_num_ns_in_group, bk_num_ns_in_group,
                                   ch_prod_layer_size, global_nid_start, use_cuda: bool = False,
                                   legacy: bool = False):

    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    # Also use the legacy code if we compile with CPU
    if not use_cuda or legacy:
        return sum_layer_backward_compilation_legacy(
            nodes, pids, fw_n_group_ids, fw_n_id_in_group, 
            num_bk_groups, bk_n_group_ids, bk_n_id_in_group, 
            bk_group_max_pars, bk_num_ns_in_group,
            ch_prod_layer_size, global_nid_start, use_cuda = use_cuda
        )

    # We construct a flattened version of `parids` where the vectors of every group is concatenated
    # into a single vector. `parids_group_start` is used to indicate the start index of every group's
    # `parids`. That is, `target_parids[parids_group_start[gid]:parids_group_start[gid+1]] == parids[gid]`
    parids_group_start = torch.zeros_like(bk_num_ns_in_group)
    parids_group_start[1:] = torch.cumsum((bk_num_ns_in_group * bk_group_max_pars)[:-1], dim = 0)
    target_parids = torch.zeros([(bk_num_ns_in_group * bk_group_max_pars).sum()], dtype = torch.long).cuda()

    # Do the same to `parpids`
    target_parpids = torch.zeros([(bk_num_ns_in_group * bk_group_max_pars).sum()], dtype = torch.long).cuda()

    parids_group_start = parids_group_start.cuda()

    # We also re-create `flat_pids` to be used to fill `parpids`
    pids_group_start = torch.zeros_like(fw_num_ns_in_group)
    pids_group_start[1:] = torch.cumsum((fw_num_ns_in_group * fw_group_max_chs)[:-1], dim = 0)
    flat_pids = torch.zeros([(fw_num_ns_in_group * fw_group_max_chs).sum()], dtype = torch.long)
    sid = 0
    for group_id, (gsize, gnchs) in enumerate(zip(fw_num_ns_in_group, fw_group_max_chs)):
        eid = sid + (gsize * gnchs)
        flat_pids[sid:eid] = pids[group_id].reshape(gsize * gnchs)
        sid = eid

    flat_pids = flat_pids.cuda()
    pids_group_start = pids_group_start.cuda()

    # This vector maintains the "current" count of parents that have been processed for every child node
    par_counts = torch.zeros([ch_prod_layer_size], dtype = torch.long)

    # Move tensors to GPU
    fw_n_group_ids = fw_n_group_ids.cuda()
    fw_n_id_in_group = fw_n_id_in_group.cuda()
    bk_n_group_ids = bk_n_group_ids.cuda()
    bk_n_id_in_group = bk_n_id_in_group.cuda()
    fw_group_max_chs = fw_group_max_chs.cuda()
    bk_group_max_pars = bk_group_max_pars.cuda()

    # This is the main loop: iterate over `ns` in the layer
    node_start = 0 # The start index of nodes in the current `ns`
    for ns in nodes:
        node_end = node_start + ns.num_nodes

        # number of nodes
        ns_num_nodes = ns.num_nodes

        # Edge indices of size [2, ns_num_edges]
        edge_ids = ns.edge_ids.cuda()
        ns_num_edges = edge_ids.size(1)

        # Construct helper indices for child nodes
        # `cs_ele_id_start` contains the global start indices of the child nodes
        # `cs_node_cum_ids` contains the local cumulative number of child nodes
        cs_ele_id_start = torch.zeros([ns.num_chs], dtype = torch.long)
        cs_node_cum_ids = torch.zeros([ns.num_chs], dtype = torch.long)
        for i, cs in enumerate(ns.chs):
            cs_ele_id_start[i] = cs._output_ind_range[0]
            if i < ns.num_chs - 1:
                cs_node_cum_ids[i+1] = cs_node_cum_ids[i] + cs.num_nodes

        cs_ele_id_start = cs_ele_id_start.cuda()
        cs_node_cum_ids = cs_node_cum_ids.cuda()

        # Get the global element ids for the child node of all edges
        global_ele_ids = torch.zeros([ns_num_edges], dtype = torch.long).cuda()

        # We store these constants in a tensor and retrieve them in the kernel
        constexprs = torch.tensor([ns_num_edges]).long().cuda()

        # Make the grid and launch kernel
        grid = lambda meta: (triton.cdiv(ns_num_edges, meta["BLOCK_SIZE"]),)

        num_chs_np2 = triton.next_power_of_2(ns.num_chs)
        _assign_global_eleids_kernel[grid](
            global_ele_ids, cs_ele_id_start, cs_node_cum_ids, edge_ids, 
            constexprs, ns.num_chs, num_chs_np2, BLOCK_SIZE = 2048
        )

        # [Recomputed] the child offset ids for every edge. That is, the `?` 
        # mark in `pids[fw_group_id][fw_local_id,?]`
        chs_offsets = np.zeros([ns_num_edges], dtype = np.int64)
        ns_nchs = np.zeros([ns_num_nodes], dtype = np.int64)
        edge_ids_np = ns.edge_ids.numpy()

        _assign_chid_kernel(chs_offsets, ns_nchs, edge_ids_np)
        chs_offsets = torch.from_numpy(chs_offsets).cuda()

        # Compute the parent offset ids for every edge. That is, the `?`
        # mark in `parids[bk_group_id][bk_local_id,?]`
        par_offsets = np.zeros([ns_num_edges], dtype = np.int64)
        par_counts_np = par_counts.numpy()
        global_ele_ids_np = global_ele_ids.cpu().numpy()

        _assign_parid_kernel(par_offsets, par_counts_np, global_ele_ids_np)

        par_counts = torch.from_numpy(par_counts_np)
        par_offsets = torch.from_numpy(par_offsets).cuda()

        # The following kernel assigns the corresponding indices to `pids` and `psids`

        # We store these constants in a tensor and retrieve them in the kernel
        # This is to avoid `triton` from compiling separate kernels for every layer configuration
        # Saves 99.9% compilation time :)
        constexprs = torch.tensor([global_nid_start, node_start, ns_num_edges]).long().cuda()

        # Make the grid and launch kernel
        grid = lambda meta: (triton.cdiv(ns_num_edges, meta["BLOCK_SIZE"]),)

        _assign_target_parids_kernel[grid](
            target_parids, target_parpids, parids_group_start, flat_pids, pids_group_start,
            edge_ids, global_ele_ids, chs_offsets, par_offsets,
            fw_n_group_ids, fw_n_id_in_group, bk_n_group_ids, bk_n_id_in_group,
            fw_group_max_chs, bk_group_max_pars, constexprs, BLOCK_SIZE = min(2048, 2**20 // num_chs_np2)
        )

        node_start = node_end

    # Restore `parids` and `parpids`
    target_parids = target_parids.cpu()
    target_parpids = target_parpids.cpu()
    parids = []
    parpids = []
    for group_id in range(bk_num_ns_in_group.size(0)):
        sid = parids_group_start[group_id]
        gsize = bk_num_ns_in_group[group_id]
        gnchs = bk_group_max_pars[group_id]
        eid = sid + gsize * gnchs
        parids.append(target_parids[sid:eid].reshape(gsize, gnchs).contiguous())
        parpids.append(target_parpids[sid:eid].reshape(gsize, gnchs).contiguous())

    return parids, parpids


## Compilation for ProdLayer ##


def get_prod_layer_stats(nodes: Sequence[SumNodes], group_size: int):
    layer_num_ngroup = sum(map(lambda ns: ns.num_node_groups, nodes))
    layer_num_edges = 0
    
    global_nid_start = group_size # indices `0`` to `group_size - 1`` is reserved for the dummy node

    ng_sid = 0
    n_chgs = torch.zeros([layer_num_ngroup], dtype = torch.long)
    for ns_idx, ns in enumerate(nodes):
        ng_eid = ng_sid + ns.num_node_groups

        n_chgs[ng_sid:ng_eid] = ns.num_chs

        layer_num_edges += ns.num_nodes * ns.num_chs

        ns._output_ind_range = (global_nid_start, global_nid_start + ns.num_nodes)
        global_nid_start += ns.num_nodes

        ng_sid = ng_eid

    return layer_num_ngroup, layer_num_edges, n_chgs


@torch.no_grad()
def prod_layer_forward_compilation(nodes, fw_partition_max_chs, n_partition_ids, n_id_in_partition, num_ngs_in_partition, group_size, use_cuda: bool = False):
    
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    nids = [torch.zeros([partition_size], dtype = torch.long) for partition_size in num_ngs_in_partition] # Node group start id
    cids = [torch.zeros([partition_size, max_chs] , dtype = torch.long) for partition_size, max_chs in zip(num_ngs_in_partition, fw_partition_max_chs)] # Child group start id

    for ns_id, ns in enumerate(nodes):

        # `partition_id`:   which partition the current node belongs to
        # `local_sid`:      the start index of the node within the current partition
        # `partition_nchs`: maximum number of child nodes in the current partition
        partition_id = n_partition_ids[ns_id]
        local_sid = n_id_in_partition[ns_id]
        local_eid = local_sid + ns.num_node_groups
        partition_nchs = fw_partition_max_chs[partition_id]

        n_sid = ns._output_ind_range[0]
        nids[partition_id][local_sid:local_eid] = torch.arange(0, ns.num_nodes, group_size) + n_sid
        for cs_id, cs in enumerate(ns.chs):
            cids[partition_id][local_sid:local_eid,cs_id] = ns.edge_ids[:,cs_id] * group_size + cs._output_ind_range[0]

    return nids, cids


@torch.no_grad()
def flatten_c_ids(nids, cids):

    num_cid_slots = sum(map(lambda x: x.size(0) * x.size(1), cids))
    flat_cids = torch.zeros([num_cid_slots], dtype = torch.long)
    flat_cid2nid = torch.zeros([num_cid_slots], dtype = torch.long)

    n_sid = 0
    c_sid = 0
    for curr_nids, curr_cids in zip(nids, cids):
        n_eid = n_sid + curr_nids.size(0)
        c_eid = c_sid + curr_cids.size(0) * curr_cids.size(1)

        flat_cids[c_sid:c_eid] = curr_cids.reshape(-1)
        flat_cid2nid[c_sid:c_eid] = curr_nids.unsqueeze(1).repeat(1, curr_cids.size(1)).reshape(-1)

        n_sid = n_eid
        c_sid = c_eid

    return flat_cids, flat_cid2nid


@torch.no_grad()
def get_prod_layer_parstats(flat_cids):

    u_cids, par_counts = torch.unique(flat_cids, sorted = True, return_counts = True)

    if u_cids[0] == 0:
        # Strip away the dummy node
        u_cids = u_cids[1:]
        par_counts = par_counts[1:]

    return u_cids, par_counts


@njit
def _assign_c_idx_kernel(flat_cids, flat_par_offsets, cum_c_nodes):
    for i in range(flat_cids.shape[0]):
        cid = flat_cids[i]
        idx = cum_c_nodes[cid]
        flat_par_offsets[i] = idx
        cum_c_nodes[cid] = idx + 1


@njit
def _assign_cid2_group_local_id(flat_u_cids, n_group_ids, n_id_in_group, cid2group_id, cid2local_id):
    for i in range(flat_u_cids.shape[0]):
        cid = flat_u_cids[i]
        cid2group_id[cid] = n_group_ids[i]
        cid2local_id[cid] = n_id_in_group[i]


@triton.jit
def _assign_target_ucids_kernel(target_u_cids_ptr, flat_u_cids_ptr, n_partition_ids_ptr, n_id_in_partition_ptr, 
                                u_cids_partition_start_ptr, constexprs_ptr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    num_nodes = tl.load(constexprs_ptr)

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_nodes

    # Get `cid`
    cid = tl.load(flat_u_cids_ptr + offsets, mask = mask, other = 0)

    # Get `partition_id` and `local_id`
    partition_id = tl.load(n_partition_ids_ptr + offsets, mask = mask, other = 0)
    local_id = tl.load(n_id_in_partition_ptr + offsets, mask = mask, other = 0)

    # Get the corresponding start id in the target tensors
    u_cids_start = tl.load(u_cids_partition_start_ptr + partition_id, mask = mask, other = 0)

    # Assign to `target_u_cids`
    tl.store(target_u_cids_ptr + u_cids_start + local_id, cid, mask = mask)


@triton.jit
def _assign_prod_target_parids_kernel(target_parids_ptr, flat_cid2nid_ptr, flat_cids_ptr, 
                                      cid2partition_id_ptr, cid2local_id_ptr, parids_partition_start_ptr,
                                      flat_par_offsets_ptr, bk_partition_max_pars_ptr, constexprs_ptr, 
                                      BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    num_edges = tl.load(constexprs_ptr)

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Get `nid` and `cid` of the edges
    nid = tl.load(flat_cid2nid_ptr + offsets, mask = mask, other = 0)
    cid = tl.load(flat_cids_ptr + offsets, mask = mask, other = 0)

    # Mask out edges that point to the dummy node
    mask = mask & (cid != 0)

    # Get `partition_id` and `local_id` using `cid`
    partition_id = tl.load(cid2partition_id_ptr + cid, mask = mask, other = 0)
    local_id = tl.load(cid2local_id_ptr + cid, mask = mask, other = 0)

    # Get the corresponding start id in the target tensors
    parids_start = tl.load(parids_partition_start_ptr + partition_id, mask = mask, other = 0)

    # Get `par_offset` of the edges
    par_offset = tl.load(flat_par_offsets_ptr + offsets, mask = mask, other = 0)

    # Assign to `target_parids`
    partition_max_n_pars = tl.load(bk_partition_max_pars_ptr + partition_id, mask = mask, other = 0)
    parid_offsets = parids_start + local_id * partition_max_n_pars + par_offset
    tl.store(target_parids_ptr + parid_offsets, nid, mask = mask)


@torch.no_grad()
def prod_layer_backward_compilation(flat_u_cids, flat_cids, flat_cid2nid, 
                                    bk_partition_max_pars, n_partition_ids, n_id_in_partition, num_ns_in_partition, 
                                    use_cuda: bool = False):
    
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    if use_cuda:

        # We construct a flattened version of `u_cids` where the vectors of every partition is concatenated
        # into a single vector. `u_cids_partition_start` is used to indicate the start index of every partition's
        # `u_cids`. That is, `target_u_cids[u_cids_partition_start[gid]:u_cids_partition_start[gid+1]] == u_cids[gid]`
        u_cids_partition_start = torch.zeros_like(num_ns_in_partition)
        u_cids_partition_start[1:] = torch.cumsum(num_ns_in_partition[:-1], dim = 0)
        target_u_cids = torch.zeros([num_ns_in_partition.sum()], dtype = torch.long)

        # Similar to `target_u_cids`, we construct a flattened version of `parids` and use `parids_partition_start`
        # for indexing
        parids_partition_start = torch.zeros_like(num_ns_in_partition)
        parids_partition_start[1:] = torch.cumsum((num_ns_in_partition * bk_partition_max_pars)[:-1], dim = 0)
        target_parids = torch.zeros([(num_ns_in_partition * bk_partition_max_pars).sum()], dtype = torch.long)

        # Precompute the parent offset ids for every edge. That is, the `?` mark in `parids[partition_id][local_id,?]`
        flat_par_offsets = np.zeros([flat_cids.size(0)], dtype = np.int64)
        num_c_ngroups = flat_u_cids.max().item() + 1
        cum_c_ngroups = np.zeros([num_c_ngroups], dtype = np.int64)

        _assign_c_idx_kernel(flat_cids.numpy(), flat_par_offsets, cum_c_ngroups)
        flat_par_offsets = torch.from_numpy(flat_par_offsets).cuda()

        # Direct mapping from `cid` to `group_id` and `local_id`
        cid2partition_id = np.zeros([num_c_ngroups], dtype = np.int64)
        cid2local_id = np.zeros([num_c_ngroups], dtype = np.int64)

        _assign_cid2_group_local_id(flat_u_cids.numpy(), n_partition_ids.numpy(), n_id_in_partition.numpy(), cid2partition_id, cid2local_id)
        cid2partition_id = torch.from_numpy(cid2partition_id).cuda()
        cid2local_id = torch.from_numpy(cid2local_id).cuda()

        # The following kernel assigns the indices to `target_u_cids` and `target_parids`. This is equivalent
        # to the easier-to-read CPU version enabled by setting `use_cuda = False`
        num_ngroups = flat_u_cids.size(0)
        num_edges = flat_cids.size(0)
        flat_u_cids = flat_u_cids.cuda()
        n_partition_ids = n_partition_ids.cuda()
        n_id_in_partition = n_id_in_partition.cuda()
        target_u_cids = target_u_cids.cuda()
        target_parids = target_parids.cuda()
        flat_cid2nid = flat_cid2nid.cuda()
        flat_cids = flat_cids.cuda()
        u_cids_partition_start = u_cids_partition_start.cuda()
        parids_partition_start = parids_partition_start.cuda()
        bk_partition_max_pars = bk_partition_max_pars.cuda()

        # We store these constants in a tensor and retrieve them in the kernel
        constexprs1 = torch.tensor([num_ngroups]).long().cuda()
        constexprs2 = torch.tensor([num_edges]).long().cuda()

        grid1 = lambda meta: (triton.cdiv(num_ngroups, meta["BLOCK_SIZE"]),)

        _assign_target_ucids_kernel[grid1](
            target_u_cids, flat_u_cids, n_partition_ids, n_id_in_partition, 
            u_cids_partition_start, constexprs1, BLOCK_SIZE = 2048
        )

        grid2 = lambda meta: (triton.cdiv(num_edges, meta["BLOCK_SIZE"]),)

        _assign_prod_target_parids_kernel[grid2](
            target_parids, flat_cid2nid, flat_cids, 
            cid2partition_id, cid2local_id, parids_partition_start,
            flat_par_offsets, bk_partition_max_pars, constexprs2, BLOCK_SIZE = 2048
        )

        target_u_cids = target_u_cids.cpu()
        u_cids = []
        for partition_id in range(num_ns_in_partition.size(0)):
            sid = u_cids_partition_start[partition_id]
            eid = sid + num_ns_in_partition[partition_id]
            u_cids.append(target_u_cids[sid:eid].contiguous())

        target_parids = target_parids.cpu()
        parids = []
        for partition_id in range(num_ns_in_partition.size(0)):
            sid = parids_partition_start[partition_id]
            psize = num_ns_in_partition[partition_id]
            pnpar = bk_partition_max_pars[partition_id]
            eid = sid + psize * pnpar
            parids.append(target_parids[sid:eid].reshape(psize, pnpar).contiguous())

    else:

        u_cids = [torch.zeros([partition_size], dtype = torch.long) for partition_size in num_ns_in_partition] # Node group id
        parids = [torch.zeros([partition_size, max_n_pars], dtype = torch.long) for partition_size, max_n_pars in zip(num_ns_in_partition, bk_partition_max_pars)] # Parent group id

        for idx in range(flat_u_cids.size(0)):
            cid = flat_u_cids[idx]

            # `partition_id`:   which partition the current node group belongs to
            # `local_id`:       the index of the node group within the current partition
            partition_id = n_partition_ids[idx]
            local_id = n_id_in_partition[idx]

            criterion = (flat_cids == cid)
            npar = criterion.sum()

            u_cids[partition_id][local_id] = cid
            parids[partition_id][local_id,:npar] = flat_cid2nid[criterion]

    return u_cids, parids
