from __future__ import annotations

import numpy as np
import torch
import networkx as nx
from typing import Type, Optional
from copy import deepcopy

from pyjuice.model import ProbCircuit
from pyjuice.graph import RegionGraph, InputRegionNode, InnerRegionNode, PartitionNode


def prune(pc: ProbCircuit, threshold: float, threshold_type: str = "fraction", params: Optional[torch.Tensor] = None):
    assert pc.param_flows is not None

    # Move PC to cpu for now
    pc_device = pc.device
    pc.to(torch.device("cpu"))

    # Get parameters back to the region graph
    pc._extract_params_to_rnodes()

    if params is None:
        params = pc.params.data

    num_params = pc.param_flows.size(0)
    param_flows = pc.param_flows.detach().cpu()
    param_flows[0] = 1.0e8 # Never prune the dummy node
    param_keep_flag = torch.zeros(param_flows.size(), dtype = torch.bool)

    if threshold_type == "fraction":
        k = int(np.round(threshold * num_params))
        param_keep_flag[torch.topk(param_flows, k, sorted = False).indices] = True
    else:
        raise ValueError(f"Unknown threshold type {threshold_type}.")

    # Generate rnode sequence
    rnode_sequence = []
    visited = set()
    def dfs(n: RegionGraph):
        if n in visited:
            return
        if not isinstance(n, InputRegionNode):
            for c in n.children:
                dfs(c)
                
        rnode_sequence.append(n)
        visited.add(n)

    dfs(pc.region_graph)

    with torch.no_grad():

        # Track nodes and edges to be kept
        node_keep_flags = dict()
        edge_keep_flags = dict()
        for rnode in rnode_sequence:
            if rnode == pc.region_graph:
                node_keep_flags[rnode] = torch.ones([rnode.num_nodes], dtype = torch.bool)
            else:
                node_keep_flags[rnode] = torch.zeros([rnode.num_nodes], dtype = torch.bool)

        # Do a top-down pass to mark nodes/edges to be kept
        for i in range(len(rnode_sequence) - 1, -1, -1):
            rnode = rnode_sequence[i]

            if isinstance(rnode, InnerRegionNode):
                node_keep_flag = node_keep_flags[rnode]
                edge_keep_flag = param_keep_flag[rnode._param_ids]
                edge_keep_flag &= (rnode.edge_ids[0:1,:] == torch.where(node_keep_flag)[0].unsqueeze(1)).any(dim = 0)

                edge_keep_flags[rnode] = edge_keep_flag

                kept_chs = torch.unique(rnode.edge_ids[1,edge_keep_flag])

                local_cumchs = 0
                for c in rnode.children:
                    criterion = (kept_chs >= local_cumchs) & (kept_chs < local_cumchs + c.num_nodes)
                    node_keep_flags[c][kept_chs[criterion]-local_cumchs] = True

                    local_cumchs += c.num_nodes

            elif isinstance(rnode, PartitionNode):
                node_keep_flag = node_keep_flags[rnode]

                for idx, c in enumerate(rnode.children):
                    kept_chs = torch.unique(rnode.edge_ids[node_keep_flag,idx])
                    node_keep_flags[c][kept_chs] = True

        # Do a bottom-up pass to generate the pruned region graph
        old2new = dict()
        for rnode in rnode_sequence:

            if isinstance(rnode, InputRegionNode):
                num_nodes = node_keep_flags[rnode].sum().item()

                scope = deepcopy(rnode.scope)
                node_type = deepcopy(rnode.node_type)
                extra_params = deepcopy(rnode.extra_params)

                pruned_params = node_type._prune_nodes(rnode._params, node_keep_flags[rnode])

                if num_nodes == 0:
                    old2new[rnode] = None
                else:
                    n = InputRegionNode(scope, num_nodes, node_type, **extra_params)
                    n._params = pruned_params
                    old2new[rnode] = n

            elif isinstance(rnode, PartitionNode):
                # Bottom-up pruning
                node_keep_flag = node_keep_flags[rnode]
                edge_ids = rnode.edge_ids
                for idx, c in enumerate(rnode.children):
                    c_kept_nodes = torch.where(node_keep_flags[c])[0]
                    node_keep_flag &= (edge_ids[:,idx:idx+1] == c_kept_nodes.unsqueeze(0)).any(dim = 1)
                node_keep_flags[rnode] = node_keep_flag

                # Map indices
                num_nodes = node_keep_flags[rnode].sum().item()
                edge_ids = rnode.edge_ids[node_keep_flags[rnode],:]

                if num_nodes == 0:
                    old2new[rnode] = None
                else:
                    for idx, c in enumerate(rnode.children):
                        idx_mapping = torch.zeros([c.num_nodes], dtype = torch.long) - 1
                        idx_mapping[node_keep_flags[c]] = torch.arange(node_keep_flags[c].sum())
                        edge_ids[:,idx] = idx_mapping[edge_ids[:,idx]]

                    assert torch.where(edge_ids == -1)[0].size(0) == 0

                    children = [old2new[c] for c in rnode.children]
                    n = PartitionNode(children, num_nodes, edge_ids.clone())
                    old2new[rnode] = n

            else:
                assert isinstance(rnode, InnerRegionNode)

                # Bottom-up pruning
                node_keep_flag = node_keep_flags[rnode]
                edge_keep_flag = edge_keep_flags[rnode]
                local_cumchs = 0
                cids = torch.zeros([0], dtype = torch.long)
                for c in rnode.children:
                    cids = torch.cat(
                        (cids, torch.where(node_keep_flags[c])[0] + local_cumchs), dim = 0
                    )
                    local_cumchs += c.num_nodes
                
                edge_keep_flag &= (rnode.edge_ids[1:2,:] == cids.unsqueeze(1)).any(dim = 0)

                node_keep_flag[:] = False
                kept_node_ids = torch.unique(rnode.edge_ids[0,edge_keep_flag])
                node_keep_flag[kept_node_ids] = True

                node_keep_flags[rnode] = node_keep_flag
                edge_keep_flags[rnode] = edge_keep_flag

                # Map indices
                num_nodes = node_keep_flag.sum().item()

                assert num_nodes > 0

                edge_ids = rnode.edge_ids[:,edge_keep_flag]
                rnode_params = rnode._params[edge_keep_flag]

                # Index mapping for `parent nodes` (nodes in `rnode`)
                idx_mapping = torch.zeros([rnode.num_nodes], dtype = torch.long) - 1
                idx_mapping[node_keep_flag] = torch.arange(node_keep_flag.sum())
                edge_ids[0,:] = idx_mapping[edge_ids[0,:]]

                # Index mapping for `child nodes`
                tot_num_ch_nodes = sum([c.num_nodes for c in rnode.children])
                idx_mapping = torch.zeros([tot_num_ch_nodes], dtype = torch.long) - 1
                local_cumchs = 0
                cum_new_ids = 0
                for c in rnode.children:
                    cids = torch.where(node_keep_flags[c])[0] + local_cumchs
                    idx_mapping[cids] = torch.arange(node_keep_flags[c].sum()) + cum_new_ids
                    local_cumchs += c.num_nodes
                    cum_new_ids += node_keep_flags[c].sum()
                edge_ids[1,:] = idx_mapping[edge_ids[1,:]]

                assert torch.where(edge_ids == -1)[0].size(0) == 0

                children = []
                for c in rnode.children:
                    nc = old2new[c]
                    if nc is not None:
                        children.append(nc)
                n = InnerRegionNode(children, num_nodes, edge_ids.clone())
                n._params = rnode_params.clone()
                old2new[rnode] = n

    pruned_pc = ProbCircuit(old2new[pc.region_graph])

    # Move PC back to its original device
    pruned_pc.to(pc_device)

    return pruned_pc
