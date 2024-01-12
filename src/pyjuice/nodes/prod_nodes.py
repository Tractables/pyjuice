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

    SPARSE = 0
    BLOCK_SPARSE = 1

    def __init__(self, num_node_groups: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Tensor] = None, group_size: int = 0, **kwargs) -> None:

        rg_node = PartitionNode([ch.region_node for ch in chs])
        super(ProdNodes, self).__init__(num_node_groups, rg_node, group_size = group_size, **kwargs)

        # Child layers
        self.chs = chs

        # Construct product edges
        self._construct_edges(edge_ids)

        # Callbacks
        self._run_init_callbacks(**kwargs)

    @property
    def num_edges(self):
        return self.num_nodes * self.num_chs

    @property
    def edge_type(self):
        if self.edge_ids.size(0) == self.num_node_groups:
            return self.BLOCK_SPARSE
        elif self.edge_ids.size(0) == self.num_nodes:
            return self.SPARSE
        else:
            raise RuntimeError(f"Unexpected shape of `edge_ids`: ({self.edge_ids.size(0)}, {self.edge_ids.size(1)})")

    def is_block_sparse(self):
        return self.edge_type == self.BLOCK_SPARSE

    def is_sparse(self):
        return self.edge_type == self.SPARSE

    def duplicate(self, *args, tie_params: bool = False, allow_type_mismatch: bool = False):
        chs = []
        for ns in args:
            assert isinstance(ns, CircuitNodes)
            chs.append(ns)

        if len(chs) == 0:
            chs = self.chs
        else:
            assert self.num_chs == len(chs), f"Number of new children ({len(chs)}) must match the number of original children ({self.num_chs})."
            for old_c, new_c in zip(self.chs, chs):
                if not allow_type_mismatch:
                    assert type(old_c) == type(new_c), f"Child type not match: ({type(new_c)} != {type(old_c)})."
                else:
                    assert not new_c.is_prod(), f"Cannot connect a product node to another."
                assert old_c.num_node_groups == new_c.num_node_groups, f"Child node size not match: (`num_node_groups`: {new_c.num_node_groups} != {old_c.num_node_groups})."
                assert old_c.group_size == new_c.group_size, f"Child node size not match: (`group_size`: {new_c.group_size} != {old_c.group_size})."

        edge_ids = self.edge_ids.clone()

        return ProdNodes(self.num_node_groups, chs, edge_ids, group_size = self.group_size, source_node = self if tie_params else None)

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, is_root: bool = True, **kwargs):
        super(ProdNodes, self).init_parameters(
            perturbation = perturbation, 
            recursive = recursive, 
            is_root = is_root,
            **kwargs
        )

    def _construct_edges(self, edge_ids: Optional[Tensor]):
        if edge_ids is None:
            for c in self.chs:
                assert self.num_node_groups == c.num_node_groups and self.group_size == c.group_size, \
                    "Cannot create edges implicitly since # nodes do not match."

            edge_ids = torch.arange(self.num_node_groups).unsqueeze(1).repeat(1, self.num_chs)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        # Sanity checks
        if edge_ids.size(0) == self.num_node_groups:
            assert edge_ids.size(0) == self.num_node_groups and edge_ids.size(1) == self.num_chs, f"Expect edge_ids.size() == ({self.num_node_groups}, {self.num_chs})."
            for cid in range(self.num_chs):
                assert torch.all(edge_ids[:,cid] >= 0), "Edge index underflow."
                assert torch.all(edge_ids[:,cid] < self.chs[cid].num_node_groups), "Edge index overflow."
        elif edge_ids.size(0) == self.num_nodes:
            assert edge_ids.size(0) == self.num_nodes and edge_ids.size(1) == self.num_chs, f"Expect edge_ids.size() == ({self.num_nodes}, {self.num_chs})."
            for cid in range(self.num_chs):
                assert torch.all(edge_ids[:,cid] >= 0), "Edge index underflow."
                assert torch.all(edge_ids[:,cid] < self.chs[cid].num_nodes), "Edge index overflow."
        else:
            raise RuntimeError(f"Unexpected shape of `edge_ids`: ({edge_ids.size(0)}, {edge_ids.size(1)})")

        self.edge_ids = edge_ids

    def __repr__(self):
        edge_type = "sparse" if self.edge_type == self.SPARSE else "block_sparse"
        scope_size = len(self.scope)
        return f"ProdNodes(num_node_groups={self.num_node_groups}, group_size={self.group_size}, num_chs={self.num_chs}, edge_type='{edge_type}', scope_size={scope_size})"
