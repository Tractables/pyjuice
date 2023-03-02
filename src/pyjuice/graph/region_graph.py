
from __future__ import annotations

import numpy as np
import torch
from typing import List, Union, Type
from copy import deepcopy
from pyjuice.utils import BitSet

Tensor = Union[np.ndarray,torch.Tensor]


class RegionGraph():
    def __init__(self, scope: BitSet, children: List[RegionGraph], num_nodes: int) -> None:
        self.scope = scope
        self.children = children
        self.num_nodes = num_nodes
        self.num_chs = len(children)

        self.parents = []
        for region_node in self.children:
            region_node.parents.append(self)

        self._output_ind_range = None
        self._param_ids = None

    
class PartitionNode(RegionGraph):
    def __init__(self, children: List[Union[InnerRegionNode,InputRegionNode]], num_nodes: int, edge_ids: Tensor) -> None:

        assert len(children) > 0, "PartitionNode receives no child."

        scope = BitSet()
        for n in children:
            assert len(scope & n.scope) == 0, "Children of a PartitionNode have overlapping scopes."
            scope |= n.scope

        super().__init__(scope, children, num_nodes)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)
        
        # Sanity checks
        assert edge_ids.size(0) == self.num_nodes and edge_ids.size(1) == self.num_chs, f"Expect edge_ids.size() == ({self.num_nodes}, {self.num_chs})."
        for cid in range(self.num_chs):
            assert torch.all(edge_ids[:,cid] >= 0), "Edge index underflow."
            assert torch.all(edge_ids[:,cid] < self.children[cid].num_nodes), "Edge index overflow."
        
        self.edge_ids = edge_ids


class InnerRegionNode(RegionGraph):
    def __init__(self, children: List[PartitionNode], num_nodes: int, edge_ids: Tensor) -> None:

        assert len(children) > 0, "InnerRegionNode receives no child."

        scope = deepcopy(children[0].scope)
        for n in children[1:]:
            assert scope == n.scope, "Children of an InnerRegionNode must have the same scope."

        super().__init__(scope, children, num_nodes)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        # Sanity checks
        num_ch_nodes = sum(map(lambda r: r.num_nodes, self.children))
        assert edge_ids.size(0) == 2, "Expect edge_ids.size(0) == 2."
        assert torch.all(edge_ids[0,:] >= 0) and torch.all(edge_ids[1,:] >= 0), "Edge index underflow."
        assert torch.all(edge_ids[0,:] < self.num_nodes) and torch.all(edge_ids[1,:] < num_ch_nodes), "Edge index overflow."

        self.edge_ids = edge_ids


class InputRegionNode(RegionGraph):
    def __init__(self, scope: Union[List,BitSet], num_nodes: int, node_type: Type, **kwargs) -> None:
        if isinstance(scope, List):
            scope = BitSet.from_array(scope)

        super().__init__(scope, [], num_nodes)

        self.node_type = node_type
        self.extra_params = deepcopy(kwargs)


def truncate_npartition(region_graph: RegionGraph, max_npartitions: int):
    old2new = dict()

    def divide_rnode(chs, num_nodes, edge_ids):
        if len(chs) <= max_npartitions:
            return PartitionNode(chs, num_nodes, edge_ids)

        grouped_chs = []
        eids = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
        for i in range(max_npartitions):
            s_idx = i * (len(chs) // max_npartitions)
            e_idx = (i + 1) * (len(chs) // max_npartitions) if i < max_npartitions - 1 else len(chs)

            rn = divide_rnode(chs[s_idx:e_idx], num_nodes, edge_ids[:,s_idx:e_idx].clone())
            rn = InnerRegionNode([rn], num_nodes, eids.clone())
            grouped_chs.append(rn)

        eids = torch.arange(num_nodes).unsqueeze(1).repeat(1, max_npartitions)
        return PartitionNode(grouped_chs, num_nodes, eids)

    def traverse(rnode: RegionGraph):
        if rnode in old2new:
            return old2new[rnode]

        if isinstance(rnode, InputRegionNode):
            new_rnode = rnode

        elif isinstance(rnode, PartitionNode):
            chs = [traverse(c) for c in rnode.children]
            new_rnode = divide_rnode(chs, rnode.num_nodes, rnode.edge_ids)

        elif isinstance(rnode, InnerRegionNode):
            chs = [traverse(c) for c in rnode.children]
            new_rnode = InnerRegionNode(chs, rnode.num_nodes, rnode.edge_ids)

        else:
            raise ValueError()

        old2new[rnode] = new_rnode
        return new_rnode

    return traverse(region_graph)