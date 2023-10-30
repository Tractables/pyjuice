from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type
from copy import deepcopy
from functools import reduce

from pyjuice.graph import InnerRegionNode
from pyjuice.functional import normalize_parameters
from .nodes import CircuitNodes
from .prod_nodes import ProdNodes

Tensor = Union[np.ndarray,torch.Tensor]


class SumNodes(CircuitNodes):
    def __init__(self, num_nodes: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Union[Tensor,Sequence[Tensor]]] = None, 
                 params: Optional[Tensor] = None, **kwargs) -> None:

        rg_node = InnerRegionNode([ch.region_node for ch in chs])
        super(SumNodes, self).__init__(num_nodes, rg_node, **kwargs)

        # Child layers
        self.chs = self._standardize_chs(chs)

        # Total number of child circuit nodes
        self.num_ch_nodes = reduce(lambda m, n: m + n, map(lambda n: n.num_nodes, chs))

        # Construct sum edges
        self._construct_edges(edge_ids)

        # Set parameters
        if params is not None:
            self.set_params(params)

        # Callbacks
        self._run_init_callbacks(**kwargs)

    @property
    def num_edges(self):
        return self.edge_ids.size(1)

    def duplicate(self, *args, tie_params: bool = False):
        chs = []
        for ns in args:
            assert isinstance(ns, CircuitNodes)
            chs.append(ns)

        if len(chs) == 0:
            chs = self.chs
        else:
            assert self.num_chs == len(chs), f"Number of new children ({len(chs)}) must match the number of original children ({self.num_chs})."
            for old_c, new_c in zip(self.chs, chs):
                assert type(old_c) == type(new_c), f"Child type not match: ({type(new_c)} != {type(old_c)})."
                assert old_c.num_nodes == new_c.num_nodes, f"Child node size not match: ({new_c.num_nodes} != {old_c.num_nodes})."

        edge_ids = self.edge_ids.clone()

        if hasattr(self, "_params") and self._params is not None and not tie_params:
            params = self._params.clone()
        else:
            # We also do not copy parameters explicitly if this is a tied node
            params = None

        return SumNodes(self.num_nodes, chs, edge_ids, params = params, source_node = self if tie_params else None)

    def get_params(self):
        if not hasattr(self, "_params"):
            return None
        return self._params

    def set_params(self, params: torch.Tensor, normalize: bool = True, pseudocount: float = 0.1):
        if self._source_node is not None:
            ns_source = self._source_node
            ns_source.set_params(params, normalize = normalize, pseudocount = pseudocount)

            return None

        if params.dim() == 1:
            assert self.edge_ids.size(1) == params.size(0)

            self._params = params.clone()

        elif params.dim() == 2:
            assert params.size(0) == self.num_nodes and params.size(1) == self.num_ch_nodes

            self._params = params[self.edge_ids[0,:],self.edge_ids[1,:]].clone().contiguous()

        if normalize:
            normalize_parameters(self._params, self.edge_ids[0,:], pseudocount = pseudocount)

    def set_edges(self, edge_ids: Union[Tensor,Sequence[Tensor]]):
        self._construct_edges(edge_ids)

        self._params = None # Clear parameters

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, is_root: bool = True, **kwargs):
        if self._source_node is None:
            self._params = torch.exp(torch.rand([self.edge_ids.size(1)]) * -perturbation)

            normalize_parameters(self._params, self.edge_ids[0,:], pseudocount = 0.0)

        super(SumNodes, self).init_parameters(
            perturbation = perturbation, 
            recursive = recursive, 
            is_root = is_root, 
            **kwargs
        )

    def _get_edges_as_mask(self):
        mask = torch.zeros([self.num_nodes, self.num_ch_nodes], dtype = torch.bool)
        mask[self.edge_ids[0,:], self.edge_ids[1,:]] = True

        return mask

    def _standardize_chs(self, chs):
        new_chs = []
        for cs in chs:
            if cs.is_input():
                new_cs = ProdNodes(
                    num_nodes = cs.num_nodes,
                    chs = [cs],
                    edge_ids = torch.arange(0, cs.num_nodes).reshape(-1, 1)
                )
                new_chs.append(new_cs)
            else:
                new_chs.append(cs)

        return new_chs

    def _construct_edges(self, edge_ids: Optional[Union[Tensor,Sequence[Tensor]]]):
        if edge_ids is None:
            edge_ids = torch.cat(
                (torch.arange(self.num_nodes).unsqueeze(1).repeat(1, self.num_ch_nodes).reshape(1, -1),
                 torch.arange(self.num_ch_nodes).unsqueeze(0).repeat(self.num_nodes, 1).reshape(1, -1)),
                dim = 0
            )
        elif isinstance(edge_ids, Sequence):
            assert len(edge_ids) == len(self.chs)

            per_ns_edge_ids = edge_ids
            ch_nid_start = 0
            edge_ids = []
            for cs_id in range(len(self.chs)):
                curr_edge_ids = per_ns_edge_ids[cs_id]
                curr_edge_ids[1,:] += ch_nid_start
                edge_ids.append(curr_edge_ids)

                ch_nid_start += self.chs[cs_id].num_nodes

            edge_ids = torch.cat(edge_ids, dim = 1)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        if edge_ids.dim() == 2 and edge_ids.type() == torch.bool:
            assert edge_ids.size(0) == self.num_nodes and edge_ids.size(1) == self.num_ch_nodes
            x_ids, y_ids = torch.where(edge_ids)
            edge_ids = torch.stack((x_ids, y_ids), dim = 0)

        # Sanity checks
        assert edge_ids.size(0) == 2, "Expect `edge_ids.size(0) == 2`."
        assert torch.all(edge_ids[0,:] >= 0) and torch.all(edge_ids[1,:] >= 0), "Edge index underflow."
        assert torch.all(edge_ids[0,:] < self.num_nodes) and torch.all(edge_ids[1,:] < self.num_ch_nodes), "Edge index overflow."
        par_ns = torch.unique(edge_ids[0,:])
        assert par_ns.size(0) == self.num_nodes and par_ns.max() == self.num_nodes - 1, "Some node has no edge."

        self.edge_ids = edge_ids
