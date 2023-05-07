from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type
from copy import deepcopy
from functools import reduce

from pyjuice.graph import PartitionNode
from .nodes import CircuitNodes

Tensor = Union[np.ndarray,torch.Tensor]


class ProdNodes(CircuitNodes):
    def __init__(self, num_nodes: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Tensor] = None) -> None:

        rg_node = PartitionNode([ch.region_node for ch in chs])
        super(ProdNodes, self).__init__(num_nodes, rg_node)

        # Child layers
        self.chs = chs

        # Construct sum edges
        self._construct_edges(edge_ids)

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