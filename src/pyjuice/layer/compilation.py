from __future__ import annotations

import torch
import threading
import functools
import os
import warnings
import time
import numba
from copy import deepcopy
from typing import Optional, Sequence

from pyjuice.nodes import CircuitNodes, SumNodes


## Helper functions ##


def flatten_sum_nodes(ns: SumNodes, *args):
    if not ns.is_tied():
        return (ns.num_nodes, ns.edge_ids, ns._param_range, [(c.num_nodes, c._output_ind_range) for c in ns.chs], *args)
    else:
        source_ns = ns.get_source_ns()
        return (ns.num_nodes, ns.edge_ids, source_ns._param_range, [(c.num_nodes, c._output_ind_range) for c in ns.chs], *args)


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


def sum_layer_forward_compilation_job(flat_nodes, nids, cids, pids, fw_group_max_chs, n_group_ids, n_id_in_group,
                                      global_nid_start, ch_prod_layer_size, job_start, job_end, return_dict = None, idx = 0):
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
            ns_param_ids = torch.zeros([edge_ids.size(1)], dtype = torch.long)

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

            ch_start = 0
            cum_tot_chs = 0
            for flat_cs in flat_ns[3]:
                cs_num_nodes = flat_cs[0]
                cs_out_ind_range = flat_cs[1]
                criterion = (edge_ids[1,:] >= cum_tot_chs) & \
                            (edge_ids[1,:] < cum_tot_chs + cs_num_nodes) & \
                            (edge_ids[0,:] == nid)
                cum_tot_chs += cs_num_nodes

                # assign node id
                nids[group_id][local_id] = global_nid

                # assign child ids
                ch_ids = edge_ids[1,criterion] + cs_out_ind_range[0]
                cids[group_id][local_id,ch_start:ch_start+ch_ids.size(0)] = ch_ids

                # mapping from the current params to global params
                if add_params_flag:
                    curr_ids = torch.where(criterion)[0]
                    curr_param_ids = torch.arange(curr_ids.size(0)) + (ns_pid_start + ns_local_pid + ch_start)
                    ns_param_ids[curr_ids] = curr_param_ids

                ch_start += ch_ids.size(0)

            # assign parameter ids
            parids = torch.arange(ch_start) + (ns_pid_start + ns_local_pid)
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


def sum_layer_forward_compilation(nodes, fw_group_max_chs, n_group_ids, n_id_in_group, num_ns_in_group, n_chs,
                                  global_nid_start, ch_prod_layer_size, param_ends, num_threads: Optional[int] = None):
    total_num_jobs = sum(map(lambda ns: ns.num_nodes, nodes))

    # Decide number of threads
    if num_threads is None:
        num_threads = max(min(os.cpu_count(), total_num_jobs // 128), 1)

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
        flat_nodes.append(flatten_sum_nodes(ns, add_params_flag))

    # Allocate target buffers
    nids = [torch.zeros([group_size], dtype = torch.long) for group_size in num_ns_in_group] # Node id
    cids = [torch.zeros([group_size, max_chs], dtype = torch.long) for group_size, max_chs in zip(num_ns_in_group, fw_group_max_chs)] # Child id
    pids = [torch.zeros([group_size, max_chs], dtype = torch.long) for group_size, max_chs in zip(num_ns_in_group, fw_group_max_chs)] # Parameter id

    if num_threads == 1:
        curr_ns_param_ids = sum_layer_forward_compilation_job(
            flat_nodes, nids, cids, pids, fw_group_max_chs, n_group_ids, n_id_in_group,
            global_nid_start, ch_prod_layer_size, 0, total_num_jobs
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
                        global_nid_start, ch_prod_layer_size, job_start, job_end, return_dict, idx)
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
            c_ns_counts = torch.bincount(ns.edge_ids[1,criterion] - ch_start)
            ch_n_pars[cs_s_oind:cs_e_oind] = c_ns_counts

    # Store local -> global parameter id mapping in `ns`
    for ns_param_ids in all_ns_param_ids:
        for ns_idx, param_ids in ns_param_ids.items():
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

    return nids, cids, pids, ch_n_pars, param_ends


def sum_layer_backward_compilation(nodes, pids, fw_n_group_ids, fw_n_id_in_group, 
                                   num_bk_groups, bk_n_group_ids, bk_n_id_in_group, 
                                   bk_group_max_pars, bk_num_ns_in_group,
                                   ch_prod_layer_size, global_nid_start):
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

    node_start = 0
    for ns in nodes:
        node_end = node_start + ns.num_nodes
        for nid in range(ns.num_nodes):
            # `group_id`: which group the current node belongs to
            # `local_id`: the index of the node within the current group
            group_id = fw_n_group_ids[node_start + nid]
            local_id = fw_n_id_in_group[node_start + nid]

            cid_start = 0
            pid_start = 0
            for cnode_id, cs in enumerate(ns.chs):
                cid_end = cid_start + cs.num_nodes
                criterion = (ns.edge_ids[1,:] >= cid_start) & \
                            (ns.edge_ids[1,:] < cid_end) & \
                            (ns.edge_ids[0,:] == nid)
                pid_end = pid_start + criterion.sum()

                ch_ids = ns.edge_ids[1,criterion] - cid_start + cs._output_ind_range[0]
                flat_cids = ch2flatidx[ch_ids] + par_counts[ch_ids] # start position specified by `ch2flatidx` + offset specified by `par_counts`
                flat_parids[flat_cids] = global_nid_start + node_start + nid
                flat_parpids[flat_cids] = pids[group_id][local_id, pid_start:pid_end]

                par_counts[ch_ids] += 1
                cid_start = cid_end

        node_start = node_end

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

    return parids, parpids