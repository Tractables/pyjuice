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

    # Only allocate once for future reuse
    range_vec = torch.arange(fw_group_max_chs.max().item())
    if use_cuda:
        range_vec = range_vec.cuda()

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
            pids[group_id][local_id,:ch_start] = range_vec[:ch_start]
            pids[group_id][local_id,:ch_start] += ns_pid_start + ns_local_pid

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
def sum_layer_forward_compilation(nodes, fw_group_max_chs, n_group_ids, n_id_in_group, num_ns_in_group, n_chs,
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


@torch.no_grad()
def sum_layer_backward_compilation(nodes, pids, fw_n_group_ids, fw_n_id_in_group, 
                                   num_bk_groups, bk_n_group_ids, bk_n_id_in_group, 
                                   bk_group_max_pars, bk_num_ns_in_group,
                                   ch_prod_layer_size, global_nid_start, use_cuda: bool = False,
                                   debug = False):

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

        parids.append(flat_parids[flatid_start:flatid_end].reshape(group_size, max_n_par))
        parpids.append(flat_parpids[flatid_start:flatid_end].reshape(group_size, max_n_par))

        flatid_start = flatid_end

    return parids, parpids


## Compilation for ProdLayer ##


def get_prod_layer_stats(nodes: Sequence[SumNodes]):
    layer_num_nodes = sum(map(lambda ns: ns.num_nodes, nodes))
    layer_num_edges = 0
    
    global_nid_start = 1 # idx 0 is reserved for the dummy node

    n_sid = 0
    n_chs = torch.zeros([layer_num_nodes], dtype = torch.long)
    for ns_idx, ns in enumerate(nodes):
        n_eid = n_sid + ns.num_nodes

        n_chs[n_sid:n_eid] = ns.num_chs

        layer_num_edges += ns.num_nodes * ns.num_chs

        ns._output_ind_range = (global_nid_start, global_nid_start + ns.num_nodes)
        global_nid_start += ns.num_nodes

        n_sid = n_eid

    return layer_num_nodes, layer_num_edges, n_chs


@torch.no_grad()
def prod_layer_forward_compilation(nodes, fw_group_max_chs, n_group_ids, n_id_in_group, num_ns_in_group, use_cuda: bool = False):

    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    nids = [torch.zeros([group_size], dtype = torch.long) for group_size in num_ns_in_group] # Node id
    cids = [torch.zeros([group_size, max_chs], dtype = torch.long) for group_size, max_chs in zip(num_ns_in_group, fw_group_max_chs)] # Child id

    n_sid = 1 # offset the dummy node
    for ns_id, ns in enumerate(nodes):
        n_eid = n_sid + ns.num_nodes

        # `group_id`:   which group the current node belongs to
        # `local_sid`:  the start index of the node within the current group
        # `group_nchs`: maximum number of child nodes in the current group
        group_id = n_group_ids[ns_id]
        local_sid = n_id_in_group[ns_id]
        local_eid = local_sid + ns.num_nodes
        group_nchs = fw_group_max_chs[group_id]

        nids[group_id][local_sid:local_eid] = torch.arange(ns.num_nodes) + n_sid
        for cs_id, cs in enumerate(ns.chs):
            cids[group_id][local_sid:local_eid,cs_id] = ns.edge_ids[:,cs_id] + cs._output_ind_range[0]

        n_sid = n_eid

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
def _assign_target_ucids_parids_kernel(target_u_cids_ptr, target_parids_ptr, flat_cid2nid_ptr, flat_cids_ptr, 
                                       cid2group_id_ptr, cid2local_id_ptr, u_cids_group_start_ptr, parids_group_start_ptr,
                                       flat_par_offsets_ptr, bk_group_max_pars_ptr, 
                                       num_edges: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Get `nid` and `cid` of the edges
    nid = tl.load(flat_cid2nid_ptr + offsets, mask = mask, other = 0)
    cid = tl.load(flat_cids_ptr + offsets, mask = mask, other = 0)

    # Get `group_id` and `local_id` using `cid`
    group_id = tl.load(cid2group_id_ptr + cid, mask = mask, other = 0)
    local_id = tl.load(cid2local_id_ptr + cid, mask = mask, other = 0)

    # Get the corresponding start id in the target tensors
    u_cids_start = tl.load(u_cids_group_start_ptr + group_id, mask = mask, other = 0)
    parids_start = tl.load(parids_group_start_ptr + group_id, mask = mask, other = 0)

    # Get `par_offset` of the edges
    par_offset = tl.load(flat_par_offsets_ptr + offsets, mask = mask, other = 0)

    # Assign to `target_u_cids`
    tl.store(target_u_cids_ptr + u_cids_start + local_id, cid, mask = mask)

    # Assign to `target_parids`
    group_max_n_pars = tl.load(bk_group_max_pars_ptr + group_id, mask = mask, other = 0)
    parid_offsets = parids_start + local_id * group_max_n_pars + par_offset
    tl.store(target_parids_ptr + parid_offsets, nid, mask = mask)


@torch.no_grad()
def prod_layer_backward_compilation(flat_u_cids, flat_cids, flat_cid2nid, 
                                    bk_group_max_pars, n_group_ids, n_id_in_group, num_ns_in_group, 
                                    use_cuda: bool = False):
    
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    if use_cuda:

        # We construct a flattened version of `u_cids` where the vectors of every group is concatenated
        # into a single vector. `u_cids_group_start` is used to indicate the start index of every group's
        # `u_cids`. That is, `target_u_cids[u_cids_group_start[gid]:u_cids_group_start[gid+1]] == u_cids[gid]`
        u_cids_group_start = torch.zeros_like(num_ns_in_group)
        u_cids_group_start[1:] = torch.cumsum(num_ns_in_group[:-1], dim = 0)
        target_u_cids = torch.zeros([num_ns_in_group.sum()], dtype = torch.long)

        # Similar to `target_u_cids`, we construct a flattened version of `parids` and use `parids_group_start`
        # for indexing
        parids_group_start = torch.zeros_like(num_ns_in_group)
        parids_group_start[1:] = torch.cumsum((num_ns_in_group * bk_group_max_pars)[:-1], dim = 0)
        target_parids = torch.zeros([(num_ns_in_group * bk_group_max_pars).sum()], dtype = torch.long)

        # Precompute the parent offset ids for every edge. That is, the `?` mark in `parids[group_id][local_id,?]`
        flat_par_offsets = np.zeros([flat_cids.size(0)], dtype = np.int64)
        num_c_nodes = flat_u_cids.max().item() + 1
        cum_c_nodes = np.zeros([num_c_nodes], dtype = np.int64)

        _assign_c_idx_kernel(flat_cids.numpy(), flat_par_offsets, cum_c_nodes)
        flat_par_offsets = torch.from_numpy(flat_par_offsets).cuda()

        # Direct mapping from `cid` to `group_id` and `local_id`
        cid2group_id = np.zeros([num_c_nodes], dtype = np.int64)
        cid2local_id = np.zeros([num_c_nodes], dtype = np.int64)

        _assign_cid2_group_local_id(flat_u_cids.numpy(), n_group_ids.numpy(), n_id_in_group.numpy(), cid2group_id, cid2local_id)
        cid2group_id = torch.from_numpy(cid2group_id).cuda()
        cid2local_id = torch.from_numpy(cid2local_id).cuda()

        # The following kernel assigns the indices to `target_u_cids` and `target_parids`. This is equivalent
        # to the easier-to-read CPU version enabled by setting `use_cuda = False`
        num_edges = flat_cids.size(0)
        target_u_cids = target_u_cids.cuda()
        target_parids = target_parids.cuda()

        grid = lambda meta: (triton.cdiv(num_edges, meta["BLOCK_SIZE"]),)

        _assign_target_ucids_parids_kernel[grid](
            target_u_cids, target_parids, flat_cid2nid.cuda(), flat_cids.cuda(), 
            cid2group_id, cid2local_id, u_cids_group_start.cuda(), parids_group_start.cuda(),
            flat_par_offsets, bk_group_max_pars.cuda(), 
            num_edges = num_edges, BLOCK_SIZE = 2048
        )

        target_u_cids = target_u_cids.cpu()
        u_cids = []
        for group_id in range(num_ns_in_group.size(0)):
            sid = u_cids_group_start[group_id]
            eid = sid + num_ns_in_group[group_id]
            u_cids.append(target_u_cids[sid:eid].contiguous())

        target_parids = target_parids.cpu()
        parids = []
        for group_id in range(num_ns_in_group.size(0)):
            sid = parids_group_start[group_id]
            gsize = num_ns_in_group[group_id]
            gnpar = bk_group_max_pars[group_id]
            eid = sid + gsize * gnpar
            parids.append(target_parids[sid:eid].reshape(gsize, gnpar).contiguous())

    else:

        u_cids = [torch.zeros([group_size], dtype = torch.long) for group_size in num_ns_in_group] # Node id
        parids = [torch.zeros([group_size, max_n_pars], dtype = torch.long) for group_size, max_n_pars in zip(num_ns_in_group, bk_group_max_pars)] # Parent id

        for idx in range(flat_u_cids.size(0)):
            cid = flat_u_cids[idx]

            # `group_id`:   which group the current node belongs to
            # `local_id`:   the index of the node within the current group
            group_id = n_group_ids[idx]
            local_id = n_id_in_group[idx]

            criterion = (flat_cids == cid)
            npar = criterion.sum()

            u_cids[group_id][local_id] = cid
            parids[group_id][local_id,:npar] = flat_cid2nid[criterion]

    return u_cids, parids
