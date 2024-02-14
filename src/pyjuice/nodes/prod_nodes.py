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
    """
    A class representing vectors of product nodes. It is created by `pyjuice.multiply`.

    :param num_node_blocks: number of node blocks
    :type num_node_blocks: int

    :param chs: sequence of child nodes
    :type chs: Sequence[CircuitNodes]

    :param edge_ids: a matrix of size [# product node blocks, # children] - the ith product node block is connected to the `edge_ids[i,j]`th node block in the jth child
    :type edge_ids: Optional[Tensor]

    :param block_size: block size
    :type block_size: int
    """

    SPARSE = 0
    BLOCK_SPARSE = 1

    def __init__(self, num_node_blocks: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Tensor] = None, block_size: int = 0, **kwargs) -> None:

        rg_node = PartitionNode([ch.region_node for ch in chs])
        super(ProdNodes, self).__init__(num_node_blocks, rg_node, block_size = block_size, **kwargs)

        # Child layers
        self.chs = chs

        # Construct product edges
        self._construct_edges(edge_ids)

        # Callbacks
        self._run_init_callbacks(**kwargs)

    @property
    def num_edges(self):
        """
        Number of edges within the current node.
        """
        return self.num_nodes * self.num_chs

    @property
    def edge_type(self):
        """
        Type of the product edge. Either `BLOCK_SPARSE` or `SPARSE`.
        """
        if self.edge_ids.size(0) == self.num_node_blocks:
            return self.BLOCK_SPARSE
        elif self.edge_ids.size(0) == self.num_nodes:
            return self.SPARSE
        else:
            raise RuntimeError(f"Unexpected shape of `edge_ids`: ({self.edge_ids.size(0)}, {self.edge_ids.size(1)})")

    def is_block_sparse(self):
        """
        Whether the edge type is `BLOCK_SPARSE`.
        """
        return self.edge_type == self.BLOCK_SPARSE

    def is_sparse(self):
        """
        Whether the edge type is `SPARSE`.
        """
        return self.edge_type == self.SPARSE

    def duplicate(self, *args, tie_params: bool = False, allow_type_mismatch: bool = False):
        """
        Create a duplication of the current node with the same specification (i.e., number of nodes, block size).

        :note: The child nodes should have the same specifications compared to the original child nodes.

        :param args: a sequence of new child nodes
        :type args: CircuitNodes

        :param tie_params: whether to tie the parameters of the current node and the duplicated node
        :type tie_params: bool

        :returns: a duplicated `ProdNodes`
        """
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
                assert old_c.num_node_blocks == new_c.num_node_blocks, f"Child node size not match: (`num_node_blocks`: {new_c.num_node_blocks} != {old_c.num_node_blocks})."
                assert old_c.block_size == new_c.block_size, f"Child node size not match: (`block_size`: {new_c.block_size} != {old_c.block_size})."

        edge_ids = self.edge_ids.clone()

        return ProdNodes(self.num_node_blocks, chs, edge_ids, block_size = self.block_size, source_node = self if tie_params else None)

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, is_root: bool = True, **kwargs):
        """
        Randomly initialize node parameters.

        :param perturbation: "amount of perturbation" added to the parameters (should be greater than 0)
        :type perturbation: float

        :param recursive: whether to recursively apply the function to child nodes
        :type recursive: bool
        """
        super(ProdNodes, self).init_parameters(
            perturbation = perturbation, 
            recursive = recursive, 
            is_root = is_root,
            **kwargs
        )

    def _construct_edges(self, edge_ids: Optional[Tensor]):
        if edge_ids is None:
            for c in self.chs:
                assert self.num_node_blocks == c.num_node_blocks and self.block_size == c.block_size, \
                    "Cannot create edges implicitly since # nodes do not match."

            edge_ids = torch.arange(self.num_node_blocks).unsqueeze(1).repeat(1, self.num_chs)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        # Sanity checks
        if edge_ids.size(0) == self.num_node_blocks:
            assert edge_ids.size(0) == self.num_node_blocks and edge_ids.size(1) == self.num_chs, f"Expect edge_ids.size() == ({self.num_node_blocks}, {self.num_chs})."
            for cid in range(self.num_chs):
                assert torch.all(edge_ids[:,cid] >= 0), "Edge index underflow."
                assert torch.all(edge_ids[:,cid] < self.chs[cid].num_node_blocks), "Edge index overflow."
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
        return f"ProdNodes(num_node_blocks={self.num_node_blocks}, block_size={self.block_size}, num_chs={self.num_chs}, edge_type='{edge_type}', scope_size={scope_size})"
