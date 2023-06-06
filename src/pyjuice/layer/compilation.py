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


def flatten_sum_nodes(ns: SumNodes):
    return (ns.num_nodes, ns.edge_ids, [(c.num_nodes, c._output_ind_range) for c in ns.chs])


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


def get_sum_layer_stats(nodes: Sequence[SumNodes], global_node_start: int):
    layer_num_nodes = 0
    layer_num_edges = 0
    max_n_chs = 0
    for ns_idx, ns in enumerate(nodes):
        n_chs = torch.max(torch.bincount(ns.edge_ids[0,:])).item()
        if n_chs > max_n_chs:
            max_n_chs = n_chs
        ns._output_ind_range = (global_node_start, global_node_start + ns.num_nodes)
        global_node_start += ns.num_nodes
        layer_num_nodes += ns.num_nodes
        layer_num_edges += ns.edge_ids.size(1)

    return layer_num_nodes, layer_num_edges, max_n_chs


def sum_layer_forward_compilation_job(cids, pids, n_chs, flat_nodes, max_n_chs, ch_prod_layer_size, global_pid_start, 
                                      job_start, job_end, return_dict = None, idx = 0):
    """
    Note: Only process jobs in [job_start, job_end).
    """
    num_jobs = job_end - job_start
    # cids = torch.zeros([num_jobs, max_n_chs], dtype = torch.long) # Child id
    # pids = torch.zeros([num_jobs, max_n_chs], dtype = torch.long) # Parameter id
    # n_chs = torch.zeros([num_jobs], dtype = torch.long) # Number of children
    ch_n_pars = torch.zeros([ch_prod_layer_size], dtype = torch.long) # Number of parents for each child node
    all_ns_param_ids = dict()

    node_start = 0
    ns_pid_start = global_pid_start
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

        ns_param_ids = torch.zeros([edge_ids.size(1)], dtype = torch.long)

        nid_start = 0 if node_start >= job_start else job_start - node_start
        nid_end = ns_num_nodes if node_start + ns_num_nodes <= job_end else job_end - node_start
        ns_local_pid = (edge_ids[0,:] < nid_start).sum().item()
        for nid in range(nid_start, nid_end):
            # job_nid = node_start + nid - job_start
            global_nid = node_start + nid

            ch_start = 0
            cum_tot_chs = 0
            for flat_cs in flat_ns[2]:
                cs_num_nodes = flat_cs[0]
                cs_out_ind_range = flat_cs[1]
                criterion = (edge_ids[1,:] >= cum_tot_chs) & \
                            (edge_ids[1,:] < cum_tot_chs + cs_num_nodes) & \
                            (edge_ids[0,:] == nid)
                cum_tot_chs += cs_num_nodes

                ch_ids = edge_ids[1,criterion] + cs_out_ind_range[0]
                cids[global_nid,ch_start:ch_start+ch_ids.size(0)] = ch_ids

                curr_ids = torch.where(criterion)[0]
                curr_param_ids = torch.arange(curr_ids.size(0)) + (ns_pid_start + ns_local_pid + ch_start)
                ns_param_ids[curr_ids] = curr_param_ids

                ch_start += ch_ids.size(0)
                ch_n_pars[ch_ids] += 1

            parids = torch.arange(ch_start) + (ns_pid_start + ns_local_pid)
            pids[global_nid,:ch_start] = parids

            ns_local_pid += ch_start

            n_chs[global_nid] = ch_start

        node_start += ns_num_nodes
        ns_pid_start += ns_num_edges

        all_ns_param_ids[ns_idx] = ns_param_ids

    if return_dict is not None:
        return_dict[idx] = (ch_n_pars, all_ns_param_ids)
    else:
        return ch_n_pars, all_ns_param_ids


def sum_layer_forward_compilation(nodes, max_n_chs, ch_prod_layer_size, param_ends, num_threads: Optional[int] = None):
    total_num_jobs = sum(map(lambda ns: ns.num_nodes, nodes))
    flat_nodes = [flatten_sum_nodes(ns) for ns in nodes]

    if num_threads is None:
        num_threads = max(min(os.cpu_count(), total_num_jobs // 128), 1)

    global_pid_start = param_ends[-1]

    cids = torch.zeros([total_num_jobs, max_n_chs], dtype = torch.long) # Child id
    pids = torch.zeros([total_num_jobs, max_n_chs], dtype = torch.long) # Parameter id
    n_chs = torch.zeros([total_num_jobs], dtype = torch.long) # Number of children

    if num_threads == 1:
        ch_n_pars, curr_ns_param_ids = sum_layer_forward_compilation_job(
            cids, pids, n_chs, flat_nodes, max_n_chs, ch_prod_layer_size, global_pid_start, 0, total_num_jobs
        )
        all_ns_param_ids = [curr_ns_param_ids]

    else:
        job_indices = get_chunk_ids(total_num_jobs, num_threads)

        threads = []
        return_dict = dict()
        for idx, (job_start, job_end) in enumerate(job_indices):
            th = threading.Thread(
                target = sum_layer_forward_compilation_job, 
                args = (cids, pids, n_chs, flat_nodes, max_n_chs, ch_prod_layer_size, 
                        global_pid_start, job_start, job_end, return_dict, idx)
            )
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

        all_ch_n_pars = []
        all_ns_param_ids = []

        for idx in range(num_threads):
            curr_ch_n_pars, curr_ns_param_ids = return_dict[idx]
            all_ch_n_pars.append(curr_ch_n_pars)
            all_ns_param_ids.append(curr_ns_param_ids)

        ch_n_pars = functools.reduce(lambda x, y: x + y, all_ch_n_pars)

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
    for ns in nodes:
        ns._param_range = (ns._param_ids.min().item(), ns._param_ids.max().item() + 1)
        ns._inverse_param_ids = torch.argsort(ns._param_ids)

    # Update `param_ends`
    npars = param_ends[-1]
    for nch in n_chs:
        npars += nch.item()
        param_ends.append(npars)

    return cids, pids, n_chs, ch_n_pars, param_ends


def sum_layer_backward_compilation(nodes, pids, ch_prod_layer_size, max_n_pars, global_node_start):
    parids = torch.zeros([ch_prod_layer_size, max_n_pars], dtype = torch.long) # Indices of parent nodes for each child node
    parpids = torch.zeros([ch_prod_layer_size, max_n_pars], dtype = torch.long) # Parameter indices for these edges
    
    # For each edge, this matrix stores the index of the edge for the parent
    parcids = torch.zeros([ch_prod_layer_size, max_n_pars], dtype = torch.long) 
    par_counts = torch.zeros([ch_prod_layer_size], dtype = torch.long)

    node_start = 0
    for ns in nodes:
        node_end = node_start + ns.num_nodes
        for nid in range(ns.num_nodes):
            ch_start = 0
            local_cumchs = 0
            for cnode_id, cs in enumerate(ns.chs):
                criterion = (ns.edge_ids[1,:] >= local_cumchs) & \
                            (ns.edge_ids[1,:] < local_cumchs + cs.num_nodes) & \
                            (ns.edge_ids[0,:] == nid)
                local_cumchs += cs.num_nodes

                ch_ids = ns.edge_ids[1,criterion] + cs._output_ind_range[0]
                parids[ch_ids, par_counts[ch_ids]] = global_node_start + node_start + nid
                parpids[ch_ids, par_counts[ch_ids]] = pids[node_start+nid,:len(ch_ids)]
                parcids[ch_ids, par_counts[ch_ids]] = torch.arange(ch_start, ch_start + ch_ids.size(0))

                par_counts[ch_ids] += 1
                ch_start += criterion.size(0)

        node_start = node_end

    return parids, parpids, parcids, par_counts