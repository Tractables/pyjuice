from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type, Optional
from copy import deepcopy
from functools import reduce

from pyjuice.graph import PartitionNode
from .nodes import CircuitNodes

Tensor = Union[np.ndarray,torch.Tensor]


class ProdNodes(CircuitNodes):
    def __init__(self, num_nodes: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Tensor] = None, **kwargs) -> None:

        rg_node = PartitionNode([ch.region_node for ch in chs])
        super(ProdNodes, self).__init__(num_nodes, rg_node, **kwargs)

        # Child layers
        self.chs = chs

        # Construct sum edges
        self._construct_edges(edge_ids)

        # Callbacks
        self._run_init_callbacks(**kwargs)

    def _construct_edges(self, edge_ids: Optional[Tensor]):
        if edge_ids is None:
            for c in self.chs:
                assert self.num_nodes == c.num_nodes, "Cannot create edges implicitly since # nodes do not match."

            edge_ids = torch.arange(self.num_nodes).unsqueeze(1).repeat(1, self.num_child_regions)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        # Sanity checks
        assert edge_ids.size(0) == self.num_nodes and edge_ids.size(1) == self.num_child_regions, f"Expect edge_ids.size() == ({self.num_nodes}, {self.num_child_regions})."
        for cid in range(self.num_child_regions):
            assert torch.all(edge_ids[:,cid] >= 0), "Edge index underflow."
            assert torch.all(edge_ids[:,cid] < self.chs[cid].num_nodes), "Edge index overflow."

        self.edge_ids = edge_ids

    def duplicate(self, *args, tie_params: bool = True):
        chs = []
        for ns in args:
            assert isinstance(ns, CircuitNodes)
            chs.append(ns)

        if len(chs) == 0:
            chs = self.chs
        else:
            assert self.num_child_regions == len(chs), f"Number of new children ({len(chs)}) must match the number of original children ({self.num_child_regions})."
            for old_c, new_c in zip(self.chs, chs):
                assert type(old_c) == type(new_c), f"Child type not match: ({type(new_c)} != {type(old_c)})."
                assert old_c.num_nodes == new_c.num_nodes, f"Child node size not match: ({new_c.num_nodes} != {old_c.num_nodes})."

        edge_ids = self.edge_ids.clone()

        return ProdNodes(self.num_nodes, chs, edge_ids, source_node = self if tie_params else None)

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, **kwargs):
        super(ProdNodes, self).init_parameters(
            perturbation = perturbation, recursive = recursive, **kwargs
        )