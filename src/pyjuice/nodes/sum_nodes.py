from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union, Type
from copy import deepcopy
from functools import reduce

from pyjuice.graph import InnerRegionNode
from .backend import normalize_ns_parameters
from .nodes import CircuitNodes
from .prod_nodes import ProdNodes

Tensor = Union[np.ndarray,torch.Tensor]


class SumNodes(CircuitNodes):
    def __init__(self, num_node_groups: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Union[Tensor,Sequence[Tensor]]] = None, 
                 params: Optional[Tensor] = None, zero_param_mask: Optional[Tensor] = None, group_size: int = 0, **kwargs) -> None:

        assert len(chs) > 0, "`SumNodes` must have at least one child."
        for i in range(1, len(chs)):
            assert chs[0].group_size == chs[i].group_size, "`group_size` of the children of a `SumNodes` should be the same."

        rg_node = InnerRegionNode([ch.region_node for ch in chs])
        super(SumNodes, self).__init__(num_node_groups, rg_node, group_size = group_size, **kwargs)

        # Child layers
        self.chs = self._standardize_chs(chs)

        # Total number of child circuit node groups
        self.num_ch_node_groups = reduce(lambda m, n: m + n, map(lambda n: n.num_node_groups, chs))

        # Group size of the children
        self.ch_group_size = self.chs[0].group_size

        # Construct sum edges
        self._construct_edges(edge_ids)

        # Set zero parameter mask
        if zero_param_mask is not None:
            self.set_zero_param_mask(zero_param_mask)

        # Set parameters
        if params is not None:
            self.set_params(params, pseudocount = 0.0)

        # Callbacks
        self._run_init_callbacks(**kwargs)

    @property
    def num_edges(self):
        return self.edge_ids.size(1) * self.group_size * self.ch_group_size

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
                assert old_c.num_node_groups == new_c.num_node_groups, f"Child node size not match: (`num_node_groups`: {new_c.num_node_groups} != {old_c.num_node_groups})."
                assert old_c.group_size == new_c.group_size, f"Child node size not match: (`group_size`: {new_c.group_size} != {old_c.group_size})."

        edge_ids = self.edge_ids.clone()

        if hasattr(self, "_params") and self._params is not None and not tie_params:
            params = self._params.clone()
        else:
            # We also do not copy parameters explicitly if this is a tied node
            params = None

        return SumNodes(self.num_node_groups, chs, edge_ids, params = params, group_size = self.group_size, source_node = self if tie_params else None)

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
            assert self.group_size == 1 and self.ch_group_size == 1
            assert self.edge_ids.size(1) == params.size(0)

            self._params = params.clone().view(-1, 1, 1)

        elif params.dim() == 3:
            assert self.edge_ids.size(1) == params.size(0) and self.group_size == params.size(1) and self.ch_group_size == params.size(2)

            self._params = params.clone()

        elif params.dim() == 4:
            assert params.size(0) == self.num_node_groups and params.size(1) == self.num_ch_node_groups and \
                self.group_size == params.size(2) and self.ch_group_size == params.size(3)

            self._params = params[self.edge_ids[0,:],self.edge_ids[1,:],:,:].clone().contiguous()

        else:
            raise ValueError("Unsupported parameter input.")

        if self.provided("zero_param_mask"):
            self._params[self.zero_param_mask] = 0.0

        if normalize:
            normalize_ns_parameters(self._params, self.edge_ids[0,:], group_size = self.group_size, 
                                    ch_group_size = self.ch_group_size, pseudocount = pseudocount)

    def set_zero_param_mask(self, zero_param_mask: Optional[Tensor] = None):
        if zero_param_mask is None:
            return None

        if self._source_node is not None:
            ns_source = self._source_node
            ns_source.set_zero_param_mask(zero_param_mask)

            return None

        assert zero_param_mask.dim() == 3
        assert zero_param_mask.size(0) == self.edge_ids.size(1)
        assert zero_param_mask.size(1) == self.group_size
        assert zero_param_mask.size(2) == self.ch_group_size
        assert zero_param_mask.dtype == torch.bool

        self.zero_param_mask = zero_param_mask

    def set_edges(self, edge_ids: Union[Tensor,Sequence[Tensor]]):
        self._construct_edges(edge_ids)

        self._params = None # Clear parameters

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, is_root: bool = True, **kwargs):
        if self._source_node is None:
            self._params = torch.exp(torch.rand([self.edge_ids.size(1), self.group_size, self.ch_group_size]) * -perturbation)

            if self.provided("zero_param_mask"):
                self._params[self.zero_param_mask] = 0.0

            normalize_ns_parameters(self._params, self.edge_ids[0,:], group_size = self.group_size, 
                                    ch_group_size = self.ch_group_size, pseudocount = 0.0)

        super(SumNodes, self).init_parameters(
            perturbation = perturbation, 
            recursive = recursive, 
            is_root = is_root, 
            **kwargs
        )

    def update_parameters(self, params: torch.Tensor, clone: bool = True):
        assert self.provided("_param_range"), "The `SumNodes` has not been compiled into a `TensorCircuit`."

        if self.is_tied():
            # Do not update parameters for tied nodes
            return None

        psid, peid = self._param_range
        if clone:
            ns_params = params[psid:peid].cpu().clone()
        else:
            ns_params = params[psid:peid].cpu()

        local_parids = (self._param_ids - psid) // (self.group_size * self.ch_group_size)
        num_pargroups = local_parids.size(0)
        ns_params = ns_params.reshape(num_pargroups, self.ch_group_size, self.group_size)
        self._params = ns_params[local_parids,:,:].permute(0, 2, 1)

    def update_param_flows(self, param_flows: torch.Tensor, origin_ns_only: bool = True, clone: bool = True):
        assert self.provided("_param_flow_range"), "The `SumNodes` has not been compiled into a `TensorCircuit`."

        if origin_ns_only and self.is_tied():
            return None

        pfsid, pfeid = self._param_flow_range
        if clone:
            ns_param_flows = param_flows[pfsid:pfeid].cpu().clone()
        else:
            ns_param_flows = param_flows[pfsid:pfeid].cpu()

        local_parfids = (self._param_flow_ids - pfsid) // (self.group_size * self.ch_group_size)
        num_parfgroups = local_parfids.size(0)
        ns_param_flows = ns_param_flows.reshape(num_parfgroups, self.ch_group_size, self.group_size)
        self._param_flows = ns_param_flows[local_parfids,:,:].permute(0, 2, 1)

    def gather_parameters(self, params: torch.Tensor):
        assert self.provided("_param_range"), "The `SumNodes` has not been compiled into a `TensorCircuit`."

        if self.is_tied() or not self.has_params():
            return None

        psid, peid = self._param_range
        ns_params = self._params[self._inverse_param_ids,:,:].permute(0, 2, 1).reshape(-1)
        params[psid:peid] = ns_params.to(params.device)

    def _get_edges_as_mask(self):
        mask = torch.zeros([self.num_node_groups, self.num_ch_nodes], dtype = torch.bool)
        mask[self.edge_ids[0,:], self.edge_ids[1,:]] = True

        return mask

    def _standardize_chs(self, chs):
        new_chs = []
        for cs in chs:
            if cs.is_input():
                new_cs = ProdNodes(
                    num_node_groups = cs.num_node_groups,
                    chs = [cs],
                    edge_ids = torch.arange(0, cs.num_node_groups).reshape(-1, 1),
                    group_size = cs.group_size
                )
                new_chs.append(new_cs)
            else:
                new_chs.append(cs)

        return new_chs

    def _construct_edges(self, edge_ids: Optional[Union[Tensor,Sequence[Tensor]]]):
        if edge_ids is None:
            edge_ids = torch.cat(
                (torch.arange(self.num_node_groups).unsqueeze(1).repeat(1, self.num_ch_node_groups).reshape(1, -1),
                 torch.arange(self.num_ch_node_groups).unsqueeze(0).repeat(self.num_node_groups, 1).reshape(1, -1)),
                dim = 0
            )
        elif isinstance(edge_ids, Sequence):
            assert len(edge_ids) == len(self.chs)

            per_ns_edge_ids = edge_ids
            ch_gid_start = 0
            edge_ids = []
            for cs_id in range(len(self.chs)):
                curr_edge_ids = per_ns_edge_ids[cs_id]
                curr_edge_ids[1,:] += ch_gid_start
                edge_ids.append(curr_edge_ids)

                ch_nid_start += self.chs[cs_id].num_node_groups

            edge_ids = torch.cat(edge_ids, dim = 1)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        if edge_ids.dim() == 2 and edge_ids.type() == torch.bool:
            assert edge_ids.size(0) == self.num_node_groups and edge_ids.size(1) == self.num_ch_node_groups
            x_ids, y_ids = torch.where(edge_ids)
            edge_ids = torch.stack((x_ids, y_ids), dim = 0)

        # Sanity checks
        assert edge_ids.size(0) == 2, "Expect `edge_ids.size(0) == 2`."
        assert torch.all(edge_ids[0,:] >= 0) and torch.all(edge_ids[1,:] >= 0), "Edge index underflow."
        assert torch.all(edge_ids[0,:] < self.num_node_groups) and torch.all(edge_ids[1,:] < self.num_ch_node_groups), "Edge index overflow."
        par_ns = torch.unique(edge_ids[0,:])
        assert par_ns.size(0) == self.num_node_groups and par_ns.max() == self.num_node_groups - 1, "Some node has no edge."

        self.edge_ids = edge_ids

    def __repr__(self):
        return f"SumNodes(num_node_groups={self.num_node_groups}, group_size={self.group_size}, num_chs={self.num_chs}, num_edges={self.num_edges})"
