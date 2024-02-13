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
from collections import OrderedDict

from pyjuice.nodes import CircuitNodes, SumNodes


## Helper functions ##


class OrderedSet():
    def __init__(self):
        self.item_set = set()
        self.item_list = list()

        self.index = 0

    def append(self, item):
        if item in self.item_set:
            return None
        
        self.item_set.add(item)
        self.item_list.append(item)

    def index(self, item):
        if item not in self.item_set:
            raise ValueError("Item not found.")

        return self.item_list.index(item)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.item_list):
            item = self.item_list[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration  # To signal the end of iteration

    def __getitem__(self, idx):
        if idx >= len(self.item_list):
            raise ValueError()

        return self.item_list[idx]

    def __contains__(self, item):
        return item in self.item_set


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


def get_sum_layer_forward_stats(nodes: Sequence[SumNodes], global_nid_start: int):
    layer_num_nblocks = sum(map(lambda ns: ns.num_node_blocks, nodes))
    layer_num_edges = 0

    n_sid = 0
    n_chs = torch.zeros([layer_num_nblocks], dtype = torch.long)
    for ns_idx, ns in enumerate(nodes):
        n_eid = n_sid + ns.num_node_blocks

        curr_n_chs = torch.bincount(ns.edge_ids[0,:])
        # To maximize flexibility, we point to individual child nodes instead of a node block
        n_chs[n_sid:n_eid] = curr_n_chs * ns.ch_block_size

        ns._output_ind_range = (global_nid_start, global_nid_start + ns.num_nodes)
        global_nid_start += ns.num_nodes
        layer_num_edges += ns.num_edges

        n_sid = n_eid

    return layer_num_nblocks, layer_num_edges, n_chs


def get_sum_layer_backward_stats(nodes: Sequence[SumNodes]):
    ch_gsize2cs = dict()
    ch_gsize2num_nblocks = dict()
    cs2parns = dict()

    for ns in nodes:
        for cs in ns.chs:
            ch_gsize = cs.block_size

            if ch_gsize not in ch_gsize2cs:
                ch_gsize2cs[ch_gsize] = OrderedSet()
                ch_gsize2num_nblocks[ch_gsize] = 0

            if cs not in ch_gsize2cs[ch_gsize]:
                ch_gsize2cs[ch_gsize].append(cs)
                ch_gsize2num_nblocks[ch_gsize] += cs.num_node_blocks

            if cs not in cs2parns:
                cs2parns[cs] = OrderedSet()

            cs2parns[cs].append(ns)

    # Iterate over all child nodes to get the parent (# node blocks) counts
    ch_gsize2n_pargs = dict()
    for ch_gsize, ch_nodes in ch_gsize2cs.items():
        n_sid = 0
        n_pargs = torch.zeros([ch_gsize2num_nblocks[ch_gsize]], dtype = torch.long)
        for cs in ch_nodes:
            n_eid = n_sid + cs.num_node_blocks

            pargcounts = torch.zeros([cs.num_node_blocks], dtype = torch.long)
            for ns in cs2parns[cs]:
                cs_id = ns.chs.index(cs)
                edge_sid = sum([c.num_node_blocks for c in ns.chs[:cs_id]])
                edge_eid = edge_sid + cs.num_node_blocks

                criterion = (ns.edge_ids[1,:] >= edge_sid) & (ns.edge_ids[1,:] < edge_eid)
                pargcounts += torch.bincount(ns.edge_ids[1,criterion] - edge_sid, minlength = cs.num_node_blocks)

            n_pargs[n_sid:n_eid] = pargcounts

            n_sid = n_eid

        ch_gsize2n_pargs[ch_gsize] = n_pargs

    return ch_gsize2cs, ch_gsize2num_nblocks, ch_gsize2n_pargs, cs2parns


@njit
def _assign_chid_kernel(chs_offsets, ns_nchs, edge_ids):
    for i in range(edge_ids.shape[1]):
        nid = edge_ids[0,i]
        idx = ns_nchs[nid]
        chs_offsets[i] = idx
        ns_nchs[nid] = idx + 1


@triton.jit
def _assign_target_ncpids_kernel(target_nids_ptr, nids_partition_start_ptr, target_cids_ptr, pcids_partition_start_ptr,
                                 target_pids_ptr, target_pfids_ptr, edge_ids_ptr, chs_offsets_ptr, n_partition_ids_ptr, 
                                 n_id_in_partition_ptr, cs_ele_id_start_ptr, cs_node_cum_ids_ptr, fw_partition_max_chs_ptr, 
                                 cum_n_chs_ptr, ns_param_ids_ptr, ns_param_flow_ids_ptr, cid_node_id_ptr, constexprs_ptr, 
                                 num_chs: tl.constexpr, num_chs_np2: tl.constexpr, add_params_flag: tl.constexpr, 
                                 add_param_flows_flag: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    global_nid_start = tl.load(constexprs_ptr)
    ns_pid_start = tl.load(constexprs_ptr + 1)
    ns_pfid_start = tl.load(constexprs_ptr + 2)
    nblock_start = tl.load(constexprs_ptr + 3)
    num_edges = tl.load(constexprs_ptr + 4)
    block_size = tl.load(constexprs_ptr + 5)

    # Get edge indices to be processed by the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Get `nid` and `cid` (size of `edge_ids` is [2, num_edges])
    nid = tl.load(edge_ids_ptr + offsets, mask = mask, other = 0)
    cid = tl.load(edge_ids_ptr + offsets + num_edges, mask = mask, other = 0)

    # Get `partition_id` and `local_id`
    partition_id = tl.load(n_partition_ids_ptr + nid + nblock_start, mask = mask, other = 0)
    local_id = tl.load(n_id_in_partition_ptr + nid + nblock_start, mask = mask, other = 0)

    # Get the child ns index every `cid` belongs to and the cum nodes & global sid
    cs_offsets = tl.arange(0, num_chs_np2)
    cs_node_cum_ids = tl.load(cs_node_cum_ids_ptr + cs_offsets, mask = (cs_offsets < num_chs), other = 0)
    
    # Get the `cs` indices the edges belong to
    cid_node_id = tl.load(cid_node_id_ptr + offsets, mask = mask, other = 0)

    cs_cum_num = tl.load(cs_node_cum_ids_ptr + cid_node_id, mask = mask, other = 0)
    cs_ele_ind = tl.load(cs_ele_id_start_ptr + cid_node_id, mask = mask, other = 0)

    # Get child offsets
    # Note: this is the `?` mark in `cids[partition_id][local_id,?]`
    chs_offset = tl.load(chs_offsets_ptr + offsets, mask = mask, other = 0)

    # Store to `target_nids`
    nids_start = tl.load(nids_partition_start_ptr + partition_id, mask = mask, other = 0)
    global_nid = global_nid_start + (nblock_start + nid) * block_size
    tl.store(target_nids_ptr + nids_start + local_id, global_nid, mask = mask)

    # Store to `target_cids`
    partition_max_n_chs = tl.load(fw_partition_max_chs_ptr + partition_id, mask = mask, other = 0)
    pcids_start = tl.load(pcids_partition_start_ptr + partition_id, mask = mask, other = 0)
    pcids_offsets = pcids_start + local_id * partition_max_n_chs + chs_offset
    global_cid = cs_ele_ind + cid - cs_cum_num
    tl.store(target_cids_ptr + pcids_offsets, global_cid, mask = mask)

    # Store to `target_pids`
    ns_local_pid = tl.load(cum_n_chs_ptr + nid, mask = mask, other = 0)
    global_pid = ns_pid_start + (ns_local_pid + chs_offset) * block_size
    tl.store(target_pids_ptr + pcids_offsets, global_pid, mask = mask)

    # Store to `target_pfids`
    global_pfid = ns_pfid_start + (ns_local_pid + chs_offset) * block_size
    tl.store(target_pfids_ptr + pcids_offsets, global_pfid, mask = mask)

    # Global parameter indices for all edges
    if add_params_flag:
        tl.store(ns_param_ids_ptr + offsets, global_pid, mask = mask)

    # Global parameter flow indices for all edges
    if add_param_flows_flag:
        tl.store(ns_param_flow_ids_ptr + offsets, global_pfid, mask = mask)


@torch.no_grad()
def sum_layer_forward_compilation(nodes, fw_partition_max_chs, n_partition_ids, n_id_in_partition, num_ngs_in_partition, n_chs,
                                  global_nid_start: int, global_pid_start: int, global_pfid_start: int, node2tiednodes: dict, 
                                  max_tied_ns_per_parflow_block: int = 4, use_cuda: bool = True):

    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if use_cuda:
        # We construct a flattened version of `nids` where the vectors of every partition is concatenated
        # into a single vector. `nids_block_start` is used to indicate the start index of every block's
        # `nids`. That is, `target_nids[nids_partition_start[i]:nids_partition_start[i+1]] == nids[i]`
        nids_partition_start = torch.zeros_like(num_ngs_in_partition)
        nids_partition_start[1:] = torch.cumsum(num_ngs_in_partition[:-1], dim = 0)
        target_nids = torch.zeros([num_ngs_in_partition.sum()], dtype = torch.long).to(device)

        # Similarly, we flatten `cids`...
        # Note: we call it `pcids...` because it is shared with `target_pids`
        pcids_partition_start = torch.zeros_like(num_ngs_in_partition)
        pcids_partition_start[1:] = torch.cumsum((num_ngs_in_partition * fw_partition_max_chs)[:-1], dim = 0)
        target_cids = torch.zeros([(num_ngs_in_partition * fw_partition_max_chs).sum()], dtype = torch.long).to(device)

        # ...and `pids`
        target_pids = torch.zeros([(num_ngs_in_partition * fw_partition_max_chs).sum()], dtype = torch.long).to(device)

        # ...and `pfids`
        target_pfids = torch.zeros([(num_ngs_in_partition * fw_partition_max_chs).sum()], dtype = torch.long).to(device)

        # Move necessary tensors to GPU
        n_partition_ids = n_partition_ids.to(device)
        n_id_in_partition = n_id_in_partition.to(device)
        fw_partition_max_chs = fw_partition_max_chs.to(device)

    else:
        nids = [torch.zeros([num_ngs_in_partition[i]], dtype = torch.long) for i in range(len(num_ngs_in_partition))]
        cids = [torch.zeros([num_ngs_in_partition[i], fw_partition_max_chs[i]], dtype = torch.long) for i in range(len(num_ngs_in_partition))]
        pids = [torch.zeros([num_ngs_in_partition[i], fw_partition_max_chs[i]], dtype = torch.long) for i in range(len(num_ngs_in_partition))]
        pfids = [torch.zeros([num_ngs_in_partition[i], fw_partition_max_chs[i]], dtype = torch.long) for i in range(len(num_ngs_in_partition))]

        ngid_in_partition = torch.zeros([len(num_ngs_in_partition)], dtype = torch.long)

    all_ns_param_ids = dict()
    all_ns_param_flow_ids = dict()
    original_param_nids = [] # `ns` with their original parameters (i.e., not tied)

    # This is the main loop: iterate over `ns` in the layer
    nblock_start = 0 # The start index of the node blocks in the current `ns`
    node2tiedcounts = dict() # Locally accumulate the occupation count
    for ns_idx, ns in enumerate(nodes):

        if not ns.is_tied():
            if not ns.provided("_param_range"):
                global_pid_end = global_pid_start + ns.num_edges
                ns._param_range = (global_pid_start, global_pid_end)
                global_pid_start = global_pid_end

                global_pfid_end = global_pfid_start + ns.num_edges
                ns._param_flow_range = (global_pfid_start, global_pfid_end)
                global_pfid_start = global_pfid_end

                add_params_flag = True
                add_param_flows_flag = True
            else:
                assert ns.provided("_param_flow_range")

                add_params_flag = False
                add_param_flows_flag = False

            original_param_nids.append(ns_idx)
                
            # Global pid and pfid start index for `ns`
            ns_pid_start = ns._param_range[0]
            ns_pfid_start = ns._param_flow_range[0]
        else:
            source_ns = ns.get_source_ns()

            # Initialize parameters
            if not source_ns.provided("_param_range"):
                global_pid_end = global_pid_start + ns.num_edges
                ns._param_range = (global_pid_start, global_pid_end)

                global_pfid_end = global_pfid_start + ns.num_edges
                ns._param_flow_range = (global_pfid_start, global_pfid_end)

                source_ns._param_range = (global_pid_start, global_pid_end)
                source_ns._param_flow_range = (global_pfid_start, global_pfid_end)

                global_pid_start = global_pid_end
                global_pfid_start = global_pfid_end

                add_params_flag = True
                add_param_flows_flag = True
            else:
                ns._param_range = deepcopy(source_ns._param_range)

                add_params_flag = False
                add_param_flows_flag = False

            if source_ns not in node2tiednodes:
                node2tiednodes[source_ns] = [[source_ns], [source_ns._param_flow_range]]
                node2tiedcounts[source_ns] = [1]
            elif source_ns not in node2tiedcounts:
                node2tiedcounts[source_ns] = [0 for _ in range(len(node2tiednodes[source_ns][0]))]
            
            if not ns.provided("_param_flow_range"):
                if all([dup_count >= max_tied_ns_per_parflow_block for dup_count in node2tiedcounts[source_ns]]):
                    global_pfid_end = global_pfid_start + ns.num_edges
                    ns._param_flow_range = (global_pfid_start, global_pfid_end)
                    global_pfid_start = global_pfid_end
                    node2tiednodes[source_ns][1].append(ns._param_flow_range)

                    node2tiednodes[source_ns][0].append(ns)
                    node2tiedcounts[source_ns].append(1)

                    add_param_flows_flag = True
                else:
                    target_id = min(range(len(node2tiedcounts[source_ns])), key = lambda i: node2tiedcounts[source_ns][i])
                    ns._param_flow_range = deepcopy(node2tiednodes[source_ns][1][target_id])

                    node2tiedcounts[source_ns][target_id] += 1

                    add_param_flows_flag = False

            # Global pid and pfid start index for `ns`
            ns_pid_start = source_ns._param_range[0]
            ns_pfid_start = ns._param_flow_range[0]

        # number of node blocks
        ns_num_nblocks = ns.num_node_blocks

        # Edge indices of size [2, ns_num_edges]
        # Here child ids of the edges are flattened out, i.e., every edge points to 
        # an actual "node" instead of a node block
        edge_ids = ns.edge_ids.clone()
        edge_ids = edge_ids[:,:,None].repeat(1, 1, ns.ch_block_size)
        edge_ids[1,:,:] *= ns.ch_block_size
        edge_ids[1,:,:] += torch.arange(0, ns.ch_block_size)[None,:]
        edge_ids = edge_ids.reshape(2, ns.edge_ids.size(1) * ns.ch_block_size).contiguous()
        ns_num_edges = edge_ids.size(1)

        if use_cuda:
            ## GPU mode ##

            # Precompute the child offset ids for every edge. That is, the `?` 
            # mark in `cids[partition][local_id,?]`
            chs_offsets = np.zeros([ns_num_edges], dtype = np.int64)
            ns_nchs = np.zeros([ns_num_nblocks], dtype = np.int64)

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
            cum_n_chs = torch.zeros([ns_num_nblocks], dtype = torch.long)
            cum_n_chs[1:] = torch.cumsum(ns_nchs[:-1], dim = 0)

            if add_params_flag:
                ns_param_ids = torch.zeros([ns_num_edges], dtype = torch.long).to(device)
            else:
                ns_param_ids = None

            if add_param_flows_flag:
                ns_param_flow_ids = torch.zeros([ns_num_edges], dtype = torch.long).to(device)
            else:
                ns_param_flow_ids = None

            # The following kernel assigns the corresponding indices to `nids`, `cids`, and `pids`
            # We first move necessary buffers to GPU
            nids_partition_start = nids_partition_start.to(device)
            edge_ids = edge_ids.to(device)
            chs_offsets = chs_offsets.to(device)
            cs_ele_id_start = cs_ele_id_start.to(device)
            cs_node_cum_ids = cs_node_cum_ids.to(device)
            cum_n_chs = cum_n_chs.to(device)
            pcids_partition_start = pcids_partition_start.to(device)

            # Which `cs` are the edges pointing to
            cid_node_id = (edge_ids[1,:].unsqueeze(1) >= cs_node_cum_ids[None,:]).sum(dim = 1) - 1

            # We store these constants in a tensor and retrieve them in the kernel
            # This is to avoid `triton` from compiling separate kernels for every layer configuration
            # Saves 99.9% compilation time :)
            constexprs = torch.tensor([global_nid_start, ns_pid_start, ns_pfid_start, nblock_start, ns_num_edges, ns.block_size]).long().to(device)

            num_chs_np2 = triton.next_power_of_2(ns.num_chs)

            # Make the grid and launch kernel
            grid = lambda meta: (triton.cdiv(ns_num_edges, meta["BLOCK_SIZE"]),)

            _assign_target_ncpids_kernel[grid](
                target_nids, nids_partition_start, target_cids, pcids_partition_start,
                target_pids, target_pfids, edge_ids, chs_offsets, n_partition_ids, 
                n_id_in_partition, cs_ele_id_start, cs_node_cum_ids, fw_partition_max_chs, 
                cum_n_chs, ns_param_ids, ns_param_flow_ids, cid_node_id, constexprs, ns.num_chs, num_chs_np2, 
                add_params_flag, add_param_flows_flag, BLOCK_SIZE = min(2048, 2**20 // num_chs_np2)
            )

            nblock_start += ns_num_nblocks

        else:
            ## CPU mode ##

            # Get number of child nodes for all nodes
            ns_nchs = torch.bincount(edge_ids[0,:], minlength = ns_num_nblocks)

            cs_node_cum_nodes = torch.zeros([ns.num_chs], dtype = torch.long)
            cs_node_cum_nodes[0] = ns.chs[0].num_nodes
            for i in range(1, ns.num_chs):
                cs_node_cum_nodes[i] = cs_node_cum_nodes[i-1] + ns.chs[i].num_nodes

            if add_params_flag:
                ns_param_ids = torch.zeros([ns_num_edges], dtype = torch.long)
            else:
                ns_param_ids = None

            if add_param_flows_flag:
                ns_param_flow_ids = torch.zeros([ns_num_edges], dtype = torch.long).to(device)
            else:
                ns_param_flow_ids = None

            # Iterate over node blocks
            cum_n_chs = 0
            for ng_id in range(ns_num_nblocks):
                partition_id = (ns_nchs[ng_id] > fw_partition_max_chs).sum()
                local_id = ngid_in_partition[partition_id]

                global_nid = ns._output_ind_range[0] + ng_id * ns.block_size

                # Assign `nids`
                nids[partition_id][local_id] = global_nid

                # Assign `cids`
                criterion = (edge_ids[0,:] == ng_id)
                local_cids = edge_ids[1,criterion]
                cids_gid = (local_cids[:,None] >= cs_node_cum_nodes[None,:]).sum(dim = 1)
                for ch_id in range(local_cids.size(0)):
                    local_base = cs_node_cum_nodes[cids_gid[ch_id]-1] if cids_gid[ch_id] >= 1 else 0
                    global_cid = ns.chs[cids_gid[ch_id]]._output_ind_range[0] + local_cids[ch_id] - local_base
                    cids[partition_id][local_id, ch_id] = global_cid

                # Assign `pids`
                global_pids = ns_pid_start + cum_n_chs + torch.arange(0, ns.block_size * criterion.sum(), ns.block_size)
                pids[partition_id][local_id, 0:global_pids.size(0)] = global_pids

                # Assign `pfids`
                global_pfids = ns_pfid_start + cum_n_chs + torch.arange(0, ns.block_size * criterion.sum(), ns.block_size)
                pfids[partition_id][local_id, 0:global_pfids.size(0)] = global_pfids

                cum_n_chs += ns.block_size * criterion.sum()

                if add_params_flag:
                    ns_param_ids[criterion] = global_pids

                if add_param_flows_flag:
                    ns_param_flow_ids[criterion] = global_pfids

                ngid_in_partition[partition_id] = local_id + 1

        if add_params_flag:
            all_ns_param_ids[ns_idx] = ns_param_ids

        if add_param_flows_flag:
            all_ns_param_flow_ids[ns_idx] = ns_param_flow_ids

    # Store global -> local parameter id mapping in `ns`
    for ns_idx, param_ids in all_ns_param_ids.items():
        ns = nodes[ns_idx]
        if ns.is_tied():
            ns = ns.get_source_ns()
        # Every edge specify the start id of [ch_block_size, block_size] parameters
        ns._param_ids = param_ids.cpu()[0::ns.ch_block_size]

    # Store global -> local parameter flow id mapping in `ns`
    for ns_idx, param_flow_ids in all_ns_param_flow_ids.items():
        ns = nodes[ns_idx]
        # Every edge specify the start id of [ch_block_size, block_size] parameter flows
        ns._param_flow_ids = param_flow_ids.cpu()[0::ns.ch_block_size]

    # Store local -> global parameter id mapping in `ns`
    for ns_idx in original_param_nids:
        ns = nodes[ns_idx]
        ns._inverse_param_ids = torch.argsort(ns._param_ids)

    if use_cuda:
        # Restore `nids`
        target_nids = target_nids.cpu()
        nids = []
        for partition_id in range(num_ngs_in_partition.size(0)):
            sid = nids_partition_start[partition_id]
            eid = sid + num_ngs_in_partition[partition_id]
            nids.append(target_nids[sid:eid].contiguous())

        # Restore `cids` and `pids`
        target_cids = target_cids.cpu()
        target_pids = target_pids.cpu()
        target_pfids = target_pfids.cpu()
        cids = []
        pids = []
        pfids = []
        for partition_id in range(num_ngs_in_partition.size(0)):
            sid = pcids_partition_start[partition_id]
            gsize = num_ngs_in_partition[partition_id]
            gnchs = fw_partition_max_chs[partition_id]
            eid = sid + gsize * gnchs
            cids.append(target_cids[sid:eid].reshape(gsize, gnchs).contiguous())
            pids.append(target_pids[sid:eid].reshape(gsize, gnchs).contiguous())
            pfids.append(target_pfids[sid:eid].reshape(gsize, gnchs).contiguous())

    return nids, cids, pids, pfids, global_pid_start, global_pfid_start


@njit
def _assign_chid_kernel(chs_offsets, ns_nchs, edge_ids):
    for i in range(edge_ids.shape[1]):
        nid = edge_ids[0,i]
        idx = ns_nchs[nid]
        chs_offsets[i] = idx
        ns_nchs[nid] = idx + 1


@njit
def _assign_parid_kernel(pars_offsets, cs_npars, edge_ids, edge_sid):
    for i in range(edge_ids.shape[1]):
        cid = edge_ids[1,i]
        idx = cs_npars[cid]
        pars_offsets[edge_sid+i] = idx
        cs_npars[cid] = idx + 1


@triton.jit
def _assign_target_chpapids_kernel(target_chids_ptr, chids_partition_start_ptr, target_parids_ptr, target_parpids_ptr, 
                                   parids_partition_start_ptr, edge_ids_ptr, pars_offsets_ptr, n_partition_ids_ptr, 
                                   n_id_in_partition_ptr, num_ngs_in_partition_ptr, partition_max_pars_ptr, cum_n_chs_ptr, 
                                   chs_offsets_ptr, constexprs_ptr, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    # Retrieve all constexprs
    ns_global_node_start = tl.load(constexprs_ptr)
    cs_global_ele_start = tl.load(constexprs_ptr + 1)
    ns_block_size = tl.load(constexprs_ptr + 2)
    cs_block_size = tl.load(constexprs_ptr + 3)
    ns_pid_start = tl.load(constexprs_ptr + 4)
    num_edges = tl.load(constexprs_ptr + 5)
    cs_nblock_start = tl.load(constexprs_ptr + 6)
    pars_offsets_start = tl.load(constexprs_ptr + 7)

    # Get edge indices to be processed by the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_edges

    # Get `cid` and `nid` (size of `edge_ids` is [2, num_edges])
    cid = tl.load(edge_ids_ptr + offsets + num_edges, mask = mask, other = 0)
    nid = tl.load(edge_ids_ptr + offsets, mask = mask, other = 0)

    # Get `partition_id` and `local_id`
    partition_id = tl.load(n_partition_ids_ptr + cid + cs_nblock_start, mask = mask, other = 0)
    local_id = tl.load(n_id_in_partition_ptr + cid + cs_nblock_start, mask = mask, other = 0)

    # Get parent offsets
    # Note: this is the `?` mark in `parids[partition_id][local_id,?]`
    pars_offset = tl.load(pars_offsets_ptr + pars_offsets_start + offsets, mask = mask, other = 0)

    # Store to `target_chids`
    chids_start = tl.load(chids_partition_start_ptr + partition_id, mask = mask, other = 0)
    global_chid = cs_global_ele_start + cid * cs_block_size
    tl.store(target_chids_ptr + chids_start + local_id, global_chid, mask = mask)

    # Store to `target_parids`
    partition_max_n_pargs = tl.load(partition_max_pars_ptr + partition_id, mask = mask, other = 0)
    parids_start = tl.load(parids_partition_start_ptr + partition_id, mask = mask, other = 0)
    parids_offsets = parids_start + local_id * partition_max_n_pargs + pars_offset
    global_parid = ns_global_node_start + nid * ns_block_size
    tl.store(target_parids_ptr + parids_offsets, global_parid, mask = mask)

    # Store to `target_parpids`
    ns_local_pid = tl.load(cum_n_chs_ptr + nid, mask = mask, other = 0)
    chs_offset = tl.load(chs_offsets_ptr + offsets, mask = mask, other = 0)
    global_pid = ns_pid_start + (ns_local_pid + chs_offset) * ns_block_size * cs_block_size
    tl.store(target_parpids_ptr + parids_offsets, global_pid, mask = mask)


def _assign_target_chpapids_cpu(target_chids, chids_partition_start, target_parids, target_parpids, parids_partition_start,
                                edge_ids, pars_offsets, n_partition_ids, n_id_in_partition, num_ngs_in_partition,
                                partition_max_pars, cum_n_chs, chs_offsets, ns_global_node_start, cs_global_ele_start, 
                                ns_block_size, cs_block_size, ns_pid_start, ns_num_edges, cs_nblock_start, pars_offsets_start):

    for edge_id in range(ns_num_edges):
        # Get `cid` and `nid` (size of `edge_ids` is [2, num_edges])
        cid = edge_ids[1, edge_id]
        nid = edge_ids[0, edge_id]

        # Get `partition_id` and `local_id`
        partition_id = n_partition_ids[cid + cs_nblock_start]
        local_id = n_id_in_partition[cid + cs_nblock_start]

        # Get parent offsets
        # Note: this is the `?` mark in `parids[partition_id][local_id,?]`
        pars_offset = pars_offsets[pars_offsets_start + edge_id]

        # Store to `target_chids`
        chids_start = chids_partition_start[partition_id]
        global_chid = cs_global_ele_start + cid * cs_block_size
        target_chids[chids_start + local_id] = global_chid

        # Store to `target_parids`
        partition_max_n_pargs = partition_max_pars[partition_id]
        parids_start = parids_partition_start[partition_id]
        parids_offsets = parids_start + local_id * partition_max_n_pargs + pars_offset
        global_parid = ns_global_node_start + nid * ns_block_size
        target_parids[parids_offsets] = global_parid

        # Store to `target_parpids`
        ns_local_pid = cum_n_chs[nid]
        chs_offset = chs_offsets[edge_id]
        global_pid = ns_pid_start + (ns_local_pid + chs_offset) * ns_block_size * cs_block_size
        target_parpids[parids_offsets] = global_pid


@torch.no_grad()
def sum_layer_backward_compilation(nodes, cs2parns, n_partition_ids, n_id_in_partition, num_ngs_in_partition, partition_max_pars,
                                   use_cuda: bool = False):

    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # We construct a flattened version of `chids` where the vectors of every partition is concatenated
    # into a single vector. `chids_partition_start` is used to indicate the start index of every partition's
    # `chids`. That is, `target_chids[chids_partition_start[i]:chids_partition_start[i+1]] == chids[i]`
    chids_partition_start = torch.zeros_like(num_ngs_in_partition)
    chids_partition_start[1:] = torch.cumsum(num_ngs_in_partition[:-1], dim = 0)
    target_chids = torch.zeros([num_ngs_in_partition.sum()], dtype = torch.long).to(device)

    # Similarly, we flatten `parids`...
    # Note: we call it `pcids...` because it is shared with `target_pids`
    parids_partition_start = torch.zeros_like(num_ngs_in_partition)
    parids_partition_start[1:] = torch.cumsum((num_ngs_in_partition * partition_max_pars)[:-1], dim = 0)
    target_parids = torch.zeros([(num_ngs_in_partition * partition_max_pars).sum()], dtype = torch.long).to(device)

    # ...and `parpids`
    target_parpids = torch.zeros([(num_ngs_in_partition * partition_max_pars).sum()], dtype = torch.long).to(device)

    # Move tensors to GPU
    n_partition_ids = n_partition_ids.to(device)
    n_id_in_partition = n_id_in_partition.to(device)
    num_ngs_in_partition = num_ngs_in_partition.to(device)
    partition_max_pars = partition_max_pars.to(device)

    # This is the main loop: iterate over `cs` in the layer
    cs_nblock_start = 0 # The start index of nodes in the current `cs`
    ns2cum_n_chs = dict()
    ns2chs_offsets = dict()
    for cs in nodes:

        # Collect all edge ids that point to `cs` in every parent `ns`
        par_edge_ids = []
        local_ns2chs_offsets = dict()
        for ns in cs2parns[cs]:
            cs_id = ns.chs.index(cs)
            edge_sid = sum([c.num_node_blocks for c in ns.chs[:cs_id]])
            edge_eid = edge_sid + cs.num_node_blocks

            criterion = (ns.edge_ids[1,:] >= edge_sid) & (ns.edge_ids[1,:] < edge_eid)
            extracted_edge_ids = ns.edge_ids[:,criterion].clone()
            extracted_edge_ids[1,:] -= edge_sid

            par_edge_ids.append(extracted_edge_ids)

            # Recreate `chs_offsets` and `cum_n_chs` to get compute the parameter ids
            if not ns in ns2cum_n_chs:
                chs_offsets = np.zeros([ns.edge_ids.size(1)], dtype = np.int64)
                ns_nchs = np.zeros([ns.num_node_blocks], dtype = np.int64)

                _assign_chid_kernel(chs_offsets, ns_nchs, ns.edge_ids.numpy())
                chs_offsets = torch.from_numpy(chs_offsets)

                ns_nchs = torch.from_numpy(ns_nchs)
                cum_n_chs = torch.zeros([ns.num_node_blocks], dtype = torch.long)
                cum_n_chs[1:] = torch.cumsum(ns_nchs[:-1], dim = 0)

                ns2cum_n_chs[ns] = cum_n_chs
                ns2chs_offsets[ns] = chs_offsets

            local_ns2chs_offsets[ns] = ns2chs_offsets[ns][criterion]

        cs_num_nblocks = cs.num_node_blocks
        cs_num_edges = sum([edge_ids.size(1) for edge_ids in par_edge_ids])
        
        # Precompute the parent offset ids for every. That is, the `?`
        # mark in `parids[partition_id][local_id,?]`
        pars_offsets = np.zeros([cs_num_edges], dtype = np.int64)
        cs_npars = np.zeros([cs_num_nblocks], dtype = np.int64)

        edge_sid = 0
        for edge_ids in par_edge_ids:
            edge_eid = edge_sid + edge_ids.size(1)

            _assign_parid_kernel(pars_offsets, cs_npars, edge_ids.numpy(), edge_sid)

            edge_sid = edge_eid

        pars_offsets = torch.from_numpy(pars_offsets)

        # Move necessary buffers to GPU
        chids_partition_start = chids_partition_start.to(device)
        parids_partition_start = parids_partition_start.to(device)
        pars_offsets = pars_offsets.to(device)

        pars_offsets_start = 0
        for ns, edge_ids in zip(cs2parns[cs], par_edge_ids):

            ns_num_edges = edge_ids.size(1)
            edge_ids = edge_ids.to(device)

            if ns.is_tied():
                ns_pid_start = ns.get_source_ns()._param_range[0]
            else:
                ns_pid_start = ns._param_range[0]

            # Get `cum_n_chs` and `chs_offsets`, which are used to get the parameter indices
            cum_n_chs = ns2cum_n_chs[ns].to(device)
            chs_offsets = local_ns2chs_offsets[ns].to(device)

            # We store these constants in a tensor and retrieve them in the kernel
            # This is to avoid `triton` from compiling separate kernels for every layer configuration
            # Saves 99.9% compilation time :)
            cs_global_ele_start = cs._output_ind_range[0]
            ns_global_node_start = ns._output_ind_range[0]
            ns_block_size = ns.block_size
            cs_block_size = cs.block_size
            
            if use_cuda:
                constexprs = torch.tensor([ns_global_node_start, cs_global_ele_start, ns_block_size, cs_block_size, 
                                           ns_pid_start, ns_num_edges, cs_nblock_start, pars_offsets_start]).long().to(device)

                # Make the grid and launch kernel
                grid = lambda meta: (triton.cdiv(ns_num_edges, meta["BLOCK_SIZE"]),)

                _assign_target_chpapids_kernel[grid](
                    target_chids, chids_partition_start, target_parids, target_parpids, parids_partition_start,
                    edge_ids, pars_offsets, n_partition_ids, n_id_in_partition, num_ngs_in_partition,
                    partition_max_pars, cum_n_chs, chs_offsets, constexprs, BLOCK_SIZE = 1024
                )
            else:
                _assign_target_chpapids_cpu(
                    target_chids, chids_partition_start, target_parids, target_parpids, parids_partition_start,
                    edge_ids, pars_offsets, n_partition_ids, n_id_in_partition, num_ngs_in_partition,
                    partition_max_pars, cum_n_chs, chs_offsets, ns_global_node_start, cs_global_ele_start, 
                    ns_block_size, cs_block_size, ns_pid_start, ns_num_edges, cs_nblock_start, pars_offsets_start
                )

            pars_offsets_start += ns_num_edges

        cs_nblock_start += cs.num_node_blocks

    # Restore `chids`
    target_chids = target_chids.cpu()
    chids = []
    for partition_id in range(num_ngs_in_partition.size(0)):
        sid = chids_partition_start[partition_id]
        eid = sid + num_ngs_in_partition[partition_id]
        chids.append(target_chids[sid:eid].contiguous())

    # Restore `parids` and `parpids`
    target_parids = target_parids.cpu()
    target_parpids = target_parpids.cpu()
    parids = []
    parpids = []
    for partition_id in range(num_ngs_in_partition.size(0)):
        sid = parids_partition_start[partition_id]
        gsize = num_ngs_in_partition[partition_id]
        gnchs = partition_max_pars[partition_id]
        eid = sid + gsize * gnchs
        parids.append(target_parids[sid:eid].reshape(gsize, gnchs).contiguous())
        parpids.append(target_parpids[sid:eid].reshape(gsize, gnchs).contiguous())

    return chids, parids, parpids


## Compilation for ProdLayer ##

def get_prod_layer_stats(nodes: Sequence[SumNodes], block_size: int, global_nid_start: int, use_block_sparse_edges: bool):
    if use_block_sparse_edges:
        layer_num_nblock = sum(map(lambda ns: ns.num_node_blocks, nodes))
        layer_num_edges = 0

        ng_sid = 0
        n_chgs = torch.zeros([layer_num_nblock], dtype = torch.long)
        for ns_idx, ns in enumerate(nodes):
            ng_eid = ng_sid + ns.num_node_blocks

            n_chgs[ng_sid:ng_eid] = ns.num_chs

            layer_num_edges += ns.num_nodes * ns.num_chs

            ns._output_ind_range = (global_nid_start, global_nid_start + ns.num_nodes)
            global_nid_start += ns.num_nodes

            ng_sid = ng_eid
    else:
        layer_num_nblock = sum(map(lambda ns: ns.num_nodes, nodes))
        layer_num_edges = 0

        ng_sid = 0
        n_chgs = torch.zeros([layer_num_nblock], dtype = torch.long)
        for ns_idx, ns in enumerate(nodes):
            ng_eid = ng_sid + ns.num_nodes

            n_chgs[ng_sid:ng_eid] = ns.num_chs

            layer_num_edges += ns.num_nodes * ns.num_chs

            ns._output_ind_range = (global_nid_start, global_nid_start + ns.num_nodes)
            global_nid_start += ns.num_nodes

            ng_sid = ng_eid

    return layer_num_nblock, layer_num_edges, n_chgs


@torch.no_grad()
def prod_layer_forward_compilation(nodes, fw_partition_max_chs, n_partition_ids, n_id_in_partition, num_ngs_in_partition, 
                                   block_size, use_block_sparse_edges: bool, use_cuda: bool = False):
    
    if use_cuda and not torch.cuda.is_available():
        use_cuda = False

    if not use_block_sparse_edges:
        assert block_size == 1

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    nids = [torch.zeros([partition_size], dtype = torch.long, device = device) for partition_size in num_ngs_in_partition] # Node block start id
    cids = [torch.zeros([partition_size, max_chs] , dtype = torch.long, device = device) for partition_size, max_chs in zip(num_ngs_in_partition, fw_partition_max_chs)] # Child block start id

    for ns_id, ns in enumerate(nodes):

        # `partition_id`:   which partition the current node belongs to
        # `local_sid`:      the start index of the node within the current partition
        # `partition_nchs`: maximum number of child nodes in the current partition
        partition_id = n_partition_ids[ns_id]
        local_sid = n_id_in_partition[ns_id]
        if use_block_sparse_edges:
            local_eid = local_sid + ns.num_node_blocks
        else:
            local_eid = local_sid + ns.num_nodes
        partition_nchs = fw_partition_max_chs[partition_id]

        if use_block_sparse_edges:
            n_sid = ns._output_ind_range[0]
            nids[partition_id][local_sid:local_eid] = torch.arange(0, ns.num_nodes, block_size, device = device) + n_sid
            for cs_id, cs in enumerate(ns.chs):
                cids[partition_id][local_sid:local_eid,cs_id] = ns.edge_ids[:,cs_id].to(device) * block_size + cs._output_ind_range[0]
        else:
            n_sid = ns._output_ind_range[0]
            nids[partition_id][local_sid:local_eid] = torch.arange(0, ns.num_nodes, device = device) + n_sid
            if ns.is_sparse():
                for cs_id, cs in enumerate(ns.chs):
                    cids[partition_id][local_sid:local_eid,cs_id] = ns.edge_ids[:,cs_id].to(device) + cs._output_ind_range[0]
            else:
                assert ns.is_block_sparse()
                edge_ids = ns.edge_ids.clone()
                edge_ids = (edge_ids[:,None,:].repeat(1, ns.block_size, 1) * ns.block_size + torch.arange(0, ns.block_size)[None,:,None]).flatten(0, 1)
                for cs_id, cs in enumerate(ns.chs):
                    cids[partition_id][local_sid:local_eid,cs_id] = edge_ids[:,cs_id].to(device) + cs._output_ind_range[0]

    if use_cuda:
        nids = [tensor.cpu() for tensor in nids]
        cids = [tensor.cpu() for tensor in cids]

    return nids, cids


@torch.no_grad()
def flatten_c_ids(nids: Sequence[torch.Tensor], cids: Sequence[torch.Tensor]):

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
def get_prod_layer_parstats(flat_cids: torch.Tensor, global_nid_start: int):

    u_cids, par_counts = torch.unique(flat_cids, sorted = True, return_counts = True)

    c_sids = torch.arange(0, u_cids.size(0))[u_cids < global_nid_start]

    if c_sids.numel() > 0:
        # Strip away dummy nodes
        c_sid = c_sids.max() + 1
        u_cids = u_cids[c_sid:]
        par_counts = par_counts[c_sid:]

    return u_cids, par_counts


@njit
def _assign_c_idx_kernel(flat_cids, flat_par_offsets, cum_c_nodes):
    for i in range(flat_cids.shape[0]):
        cid = flat_cids[i]
        idx = cum_c_nodes[cid]
        flat_par_offsets[i] = idx
        cum_c_nodes[cid] = idx + 1


@njit
def _assign_cid2_block_local_id(flat_u_cids, n_block_ids, n_id_in_block, cid2block_id, cid2local_id):
    for i in range(flat_u_cids.shape[0]):
        cid = flat_u_cids[i]
        cid2block_id[cid] = n_block_ids[i]
        cid2local_id[cid] = n_id_in_block[i]


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
def prod_layer_backward_compilation(flat_u_cids, flat_cids, flat_cid2nid, bk_partition_max_pars, n_partition_ids, 
                                    n_id_in_partition, num_ns_in_partition, use_cuda: bool = False):
    
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
        num_c_nblocks = flat_u_cids.max().item() + 1
        cum_c_nblocks = np.zeros([num_c_nblocks], dtype = np.int64)

        _assign_c_idx_kernel(flat_cids.numpy(), flat_par_offsets, cum_c_nblocks)
        flat_par_offsets = torch.from_numpy(flat_par_offsets).cuda()

        # Direct mapping from `cid` to `block_id` and `local_id`
        cid2partition_id = np.zeros([num_c_nblocks], dtype = np.int64)
        cid2local_id = np.zeros([num_c_nblocks], dtype = np.int64)

        _assign_cid2_block_local_id(flat_u_cids.numpy(), n_partition_ids.numpy(), n_id_in_partition.numpy(), cid2partition_id, cid2local_id)
        cid2partition_id = torch.from_numpy(cid2partition_id).cuda()
        cid2local_id = torch.from_numpy(cid2local_id).cuda()

        # The following kernel assigns the indices to `target_u_cids` and `target_parids`. This is equivalent
        # to the easier-to-read CPU version enabled by setting `use_cuda = False`
        num_nblocks = flat_u_cids.size(0)
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
        constexprs1 = torch.tensor([num_nblocks]).long().cuda()
        constexprs2 = torch.tensor([num_edges]).long().cuda()

        grid1 = lambda meta: (triton.cdiv(num_nblocks, meta["BLOCK_SIZE"]),)

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

        u_cids = [torch.zeros([partition_size], dtype = torch.long) for partition_size in num_ns_in_partition] # Node block id
        parids = [torch.zeros([partition_size, max_n_pars], dtype = torch.long) for partition_size, max_n_pars in zip(num_ns_in_partition, bk_partition_max_pars)] # Parent block id

        for idx in range(flat_u_cids.size(0)):
            cid = flat_u_cids[idx]

            # `partition_id`:   which partition the current node block belongs to
            # `local_id`:       the index of the node block within the current partition
            partition_id = n_partition_ids[idx]
            local_id = n_id_in_partition[idx]

            criterion = (flat_cids == cid)
            npar = criterion.sum()

            u_cids[partition_id][local_id] = cid
            parids[partition_id][local_id,:npar] = flat_cid2nid[criterion]

    return u_cids, parids
