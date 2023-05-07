from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type
from copy import deepcopy
from functools import reduce

from pyjuice.graph import InnerRegionNode
from .nodes import CircuitNodes

Tensor = Union[np.ndarray,torch.Tensor]


class SumNodes(CircuitNodes):
    def __init__(self, num_nodes: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Tensor] = None) -> None:

        rg_node = InnerRegionNode([ch.region_node for ch in chs])
        super(SumNodes, self).__init__(num_nodes, rg_node)

        # Child layers
        self.chs = chs

        # Total number of child circuit nodes
        self.num_ch_nodes = reduce(lambda m, n: m + n, map(lambda n: n.num_nodes, chs))

        # Construct sum edges
        self._construct_edges(edge_ids)

    def _construct_edges(self, edge_ids: Optional[Tensor]):
        if edge_ids is None:
            edge_ids = torch.cat(
                (torch.arange(self.num_nodes).unsqueeze(1).repeat(1, self.num_ch_nodes).reshape(1, -1),
                 torch.arange(self.num_ch_nodes).unsqueeze(0).repeat(self.num_nodes, 1).reshape(1, -1)),
                dim = 0
            )

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        # Sanity checks
        assert edge_ids.size(0) == 2, "Expect edge_ids.size(0) == 2."
        assert torch.all(edge_ids[0,:] >= 0) and torch.all(edge_ids[1,:] >= 0), "Edge index underflow."
        assert torch.all(edge_ids[0,:] < self.num_nodes) and torch.all(edge_ids[1,:] < self.num_ch_nodes), "Edge index overflow."

        self.edge_ids = edge_ids
