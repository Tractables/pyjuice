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
    def __init__(self, num_nodes: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Tensor] = None, 
                 params: Optional[Tensor] = None, **kwargs) -> None:

        rg_node = InnerRegionNode([ch.region_node for ch in chs])
        super(SumNodes, self).__init__(num_nodes, rg_node, **kwargs)

        # Child layers
        self.chs = chs

        # Total number of child circuit nodes
        self.num_ch_nodes = reduce(lambda m, n: m + n, map(lambda n: n.num_nodes, chs))

        # Construct sum edges
        self._construct_edges(edge_ids)

        # Set parameters
        if params is not None:
            self.set_params(params)

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

    def duplicate(self, *args):
        chs = []
        for ns in args:
            assert isinstance(ns, CircuitNodes)
            chs.append(ns)

        if chs is None:
            chs = self.chs
        else:
            assert self.num_child_regions == len(chs), f"Number of new children ({len(chs)}) must match the number of original children ({self.num_child_regions})."
            for old_c, new_c in zip(self.chs, chs):
                assert type(old_c) == type(new_c), f"Child type not match: ({type(new_c)} != {type(old_c)})."
                assert old_c.num_nodes == new_c.num_nodes, f"Child node size not match: ({new_c.num_nodes} != {old_c.num_nodes})."

        edge_ids = self.edge_ids.clone()

        return SumNodes(self.num_nodes, chs, edge_ids, source_node = self)

    def get_params(self):
        return self._params

    def set_params(self, params: torch.Tensor):
        assert params.dim() == 1
        assert self.edge_ids.size(1) == params.size(0)

        self._params = params.clone()