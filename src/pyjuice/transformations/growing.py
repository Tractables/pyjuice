from __future__ import annotations

import numpy as np
import torch
import networkx as nx
from typing import Type, Optional
from copy import deepcopy

from pyjuice.model import TensorCircuit
from pyjuice.graph import RegionGraph, InputRegionNode, InnerRegionNode, PartitionNode


def duplicate(pc: TensorCircuit, sigma: float = 0.1):
    # Move PC to cpu for now
    pc_device = pc.device
    pc.to(torch.device("cpu"))

    # Get parameters back to the region graph
    pc._extract_params_to_rnodes()

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

        old2new = dict()
        for rnode in rnode_sequence:

            if isinstance(rnode, InputRegionNode):

                scope = deepcopy(rnode.scope)
                node_type = deepcopy(rnode.node_type)
                extra_params = deepcopy(rnode.extra_params)

                pruned_params = node_type._duplicate_nodes(rnode._params)

                n = InputRegionNode(scope, rnode.num_nodes * 2, node_type, **extra_params)
                n._params = pruned_params
                old2new[rnode] = n

            elif isinstance(rnode, PartitionNode):
                edge_ids = torch.zeros([rnode.num_nodes * 2, rnode.num_chs], dtype = torch.long)
                edge_ids[0::2,:] = rnode.edge_ids * 2
                edge_ids[1::2,:] = rnode.edge_ids * 2 + 1

                children = [old2new[c] for c in rnode.children]
                n = PartitionNode(children, rnode.num_nodes * 2, edge_ids)
                old2new[rnode] = n

            else:
                assert isinstance(rnode, InnerRegionNode)

                origin_num_edges = rnode.edge_ids.size(1)
                origin_num_ch_nodes = sum([c.num_nodes for c in rnode.children])

                idx_mapping = torch.zeros([origin_num_ch_nodes], dtype = torch.long)
                local_cumchs = 0
                for c in rnode.children:
                    idx_mapping[local_cumchs:local_cumchs+c.num_nodes] = local_cumchs * 2 + c.num_nodes + torch.arange(c.num_nodes)
                    local_cumchs += c.num_nodes

                edge_ids = torch.zeros([2, origin_num_edges * 4], dtype = torch.long)
                edge_ids[:,0::4] = rnode.edge_ids
                edge_ids[:,1::4] = rnode.edge_ids
                edge_ids[:,2::4] = rnode.edge_ids
                edge_ids[:,3::4] = rnode.edge_ids

                edge_ids[1,0::4] = edge_ids[1,0::4] * 2
                edge_ids[1,1::4] = edge_ids[1,1::4] * 2 + 1
                edge_ids[1,2::4] = edge_ids[1,2::4] * 2
                edge_ids[1,3::4] = edge_ids[1,3::4] * 2 + 1

                edge_ids[0,0::4] = edge_ids[0,0::4] * 2
                edge_ids[0,1::4] = edge_ids[0,1::4] * 2
                edge_ids[0,2::4] = edge_ids[0,2::4] * 2 + 1
                edge_ids[0,3::4] = edge_ids[0,3::4] * 2 + 1

                params = torch.zeros([origin_num_edges * 4], dtype = torch.float32)
                params[0::4] = rnode._params
                params[1::4] = rnode._params * torch.normal(mean = torch.ones([origin_num_edges]), std = sigma)
                params[2::4] = rnode._params * torch.normal(mean = torch.ones([origin_num_edges]), std = sigma)
                params[3::4] = rnode._params * torch.normal(mean = torch.ones([origin_num_edges]), std = sigma)

                children = [old2new[c] for c in rnode.children]
                n = InnerRegionNode(children, rnode.num_nodes * 2, edge_ids)
                n._params = params
                old2new[rnode] = n

    grown_pc = ProbCircuit(old2new[pc.region_graph])

    # Move PC back to its original device
    grown_pc.to(pc_device)

    return grown_pc
