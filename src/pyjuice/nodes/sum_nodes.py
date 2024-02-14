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
    """
    A class representing vectors of sum nodes. It is created by `pyjuice.summate`.

    :param num_node_blocks: number of node blocks
    :type num_node_blocks: int

    :param chs: sequence of child nodes
    :type chs: Sequence[CircuitNodes]

    :param edge_ids: a matrix of size [2, # edges] - every size-2 column vector [i,j] defines a set of edges that fully connect the ith sum node block and the jth child node block
    :type edge_ids: Optional[Tensor]

    :param block_size: block size
    :type block_size: int
    """

    def __init__(self, num_node_blocks: int, chs: Sequence[CircuitNodes], edge_ids: Optional[Union[Tensor,Sequence[Tensor]]] = None, 
                 params: Optional[Tensor] = None, zero_param_mask: Optional[Tensor] = None, block_size: int = 0, **kwargs) -> None:

        assert len(chs) > 0, "`SumNodes` must have at least one child."
        for i in range(1, len(chs)):
            assert chs[0].block_size == chs[i].block_size, "`block_size` of the children of a `SumNodes` should be the same."

        rg_node = InnerRegionNode([ch.region_node for ch in chs])
        super(SumNodes, self).__init__(num_node_blocks, rg_node, block_size = block_size, **kwargs)

        # Child layers
        self.chs = self._standardize_chs(chs)

        # Total number of child circuit node blocks
        self.num_ch_node_blocks = reduce(lambda m, n: m + n, map(lambda n: n.num_node_blocks, chs))

        # Block size of the children
        self.ch_block_size = self.chs[0].block_size

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
        """
        Number of edges within the current node.
        """
        return self.edge_ids.size(1) * self.block_size * self.ch_block_size

    def duplicate(self, *args, tie_params: bool = False) -> SumNodes:
        """
        Create a duplication of the current node with the same specification (i.e., number of nodes, block size).

        :note: The child nodes should have the same specifications compared to the original child nodes.

        :param args: a sequence of new child nodes
        :type args: CircuitNodes

        :param tie_params: whether to tie the parameters of the current node and the duplicated node
        :type tie_params: bool

        :returns: a duplicated `SumNodes`
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
                assert type(old_c) == type(new_c), f"Child type not match: ({type(new_c)} != {type(old_c)})."
                assert old_c.num_node_blocks == new_c.num_node_blocks, f"Child node size not match: (`num_node_blocks`: {new_c.num_node_blocks} != {old_c.num_node_blocks})."
                assert old_c.block_size == new_c.block_size, f"Child node size not match: (`block_size`: {new_c.block_size} != {old_c.block_size})."

        edge_ids = self.edge_ids.clone()

        if hasattr(self, "_params") and self._params is not None and not tie_params:
            params = self._params.clone()
        else:
            # We also do not copy parameters explicitly if this is a tied node
            params = None

        return SumNodes(self.num_node_blocks, chs, edge_ids, params = params, block_size = self.block_size, source_node = self if tie_params else None)

    def get_params(self):
        """
        Get the sum node parameters.
        """
        if not hasattr(self, "_params"):
            return None
        return self._params

    def set_params(self, params: torch.Tensor, normalize: bool = True, pseudocount: float = 0.0):
        """
        Set the sum node parameters.

        :param params: parameters to be set
        :type params: Union[torch.Tensor,Dict]

        :param normalize: whether to normalize the parameters
        :type normalize: bool

        :param pseudocount: pseudo count added to the parameters
        :type pseudocount: float
        """
        if self._source_node is not None:
            ns_source = self._source_node
            ns_source.set_params(params, normalize = normalize, pseudocount = pseudocount)

            return None

        if params.dim() == 1:
            assert self.block_size == 1 and self.ch_block_size == 1
            assert self.edge_ids.size(1) == params.size(0)

            self._params = params.clone().view(-1, 1, 1)

        elif params.dim() == 2:
            ch_num_nblocks = sum([cs.num_node_blocks for cs in self.chs])
            assert params.size(0) == self.num_nodes
            assert params.size(1) == self.ch_block_size * ch_num_nblocks

            self._params = params.reshape(self.num_node_blocks, self.block_size, ch_num_nblocks, self.ch_block_size).permute(0, 2, 1, 3).flatten(0, 1).contiguous()

        elif params.dim() == 3:
            assert self.edge_ids.size(1) == params.size(0) and self.block_size == params.size(1) and self.ch_block_size == params.size(2)

            self._params = params.clone()

        elif params.dim() == 4:
            assert params.size(0) == self.num_node_blocks and params.size(1) == self.num_ch_node_blocks and \
                self.block_size == params.size(2) and self.ch_block_size == params.size(3)

            self._params = params[self.edge_ids[0,:],self.edge_ids[1,:],:,:].clone().contiguous()

        else:
            raise ValueError("Unsupported parameter input.")

        if self.provided("zero_param_mask"):
            self._params[self._zero_param_mask] = 0.0

        if normalize:
            normalize_ns_parameters(self._params, self.edge_ids[0,:], block_size = self.block_size, 
                                    ch_block_size = self.ch_block_size, pseudocount = pseudocount)

    def set_zero_param_mask(self, zero_param_mask: Optional[Tensor] = None):
        if zero_param_mask is None:
            return None

        if self._source_node is not None:
            ns_source = self._source_node
            ns_source.set_zero_param_mask(zero_param_mask)

            return None

        assert zero_param_mask.dim() == 3
        assert zero_param_mask.size(0) == self.edge_ids.size(1)
        assert zero_param_mask.size(1) == self.block_size
        assert zero_param_mask.size(2) == self.ch_block_size
        assert zero_param_mask.dtype == torch.bool

        self._zero_param_mask = zero_param_mask

    def get_zero_param_mask(self):
        if not self.provided("_zero_param_mask"):
            return None
        else:
            return self._zero_param_mask

    def set_edges(self, edge_ids: Union[Tensor,Sequence[Tensor]]):
        self._construct_edges(edge_ids)

        self._params = None # Clear parameters

    def init_parameters(self, perturbation: float = 2.0, recursive: bool = True, is_root: bool = True, **kwargs):
        """
        Randomly initialize node parameters.

        :param perturbation: "amount of perturbation" added to the parameters (should be greater than 0)
        :type perturbation: float

        :param recursive: whether to recursively apply the function to child nodes
        :type recursive: bool
        """
        if self._source_node is None:
            self._params = torch.exp(torch.rand([self.edge_ids.size(1), self.block_size, self.ch_block_size]) * -perturbation)

            if self.provided("zero_param_mask"):
                self._params[self._zero_param_mask] = 0.0

            normalize_ns_parameters(self._params, self.edge_ids[0,:], block_size = self.block_size, 
                                    ch_block_size = self.ch_block_size, pseudocount = 0.0)

        super(SumNodes, self).init_parameters(
            perturbation = perturbation, 
            recursive = recursive, 
            is_root = is_root, 
            **kwargs
        )

    def update_parameters(self, params: torch.Tensor, clone: bool = True):
        """
        Update parameters from `pyjuice.TensorCircuit` to the current node.

        :param params: the parameter tensor in the `TensorCircuit`
        :type params: torch.Tensor

        :param clone: whether to clone the parameters
        :type clone: bool
        """
        assert self.provided("_param_range"), "The `SumNodes` has not been compiled into a `TensorCircuit`."

        if self.is_tied():
            # Do not update parameters for tied nodes
            return None

        psid, peid = self._param_range
        if clone:
            ns_params = params[psid:peid].cpu().clone()
        else:
            ns_params = params[psid:peid].cpu()

        local_parids = (self._param_ids - psid) // (self.block_size * self.ch_block_size)
        num_parblocks = local_parids.size(0)
        ns_params = ns_params.reshape(num_parblocks, self.ch_block_size, self.block_size)
        self._params = ns_params[local_parids,:,:].permute(0, 2, 1)

    def update_param_flows(self, param_flows: torch.Tensor, origin_ns_only: bool = True, clone: bool = True):
        """
        Update parameter flows from `pyjuice.TensorCircuit` to the current node.

        :param params_flows: the parameter flow tensor in the `TensorCircuit`
        :type params_flows: torch.Tensor

        :param clone: whether to clone the parameters
        :type clone: bool
        """
        assert self.provided("_param_flow_range"), "The `SumNodes` has not been compiled into a `TensorCircuit`."

        if origin_ns_only and self.is_tied():
            return None

        pfsid, pfeid = self._param_flow_range
        if clone:
            ns_param_flows = param_flows[pfsid:pfeid].cpu().clone()
        else:
            ns_param_flows = param_flows[pfsid:pfeid].cpu()

        local_parfids = (self._param_flow_ids - pfsid) // (self.block_size * self.ch_block_size)
        num_parfblocks = local_parfids.size(0)
        ns_param_flows = ns_param_flows.reshape(num_parfblocks, self.ch_block_size, self.block_size)
        self._param_flows = ns_param_flows[local_parfids,:,:].permute(0, 2, 1)

    def gather_parameters(self, params: torch.Tensor):
        """
        Update parameters from the current node to the compiled `pyjuice.TensorCircuit`.

        :param params: the parameter tensor in the `TensorCircuit`
        :type params: torch.Tensor
        """
        assert self.provided("_param_range"), "The `SumNodes` has not been compiled into a `TensorCircuit`."

        if self.is_tied() or not self.has_params():
            return None

        psid, peid = self._param_range
        ns_params = self._params[self._inverse_param_ids,:,:].permute(0, 2, 1).reshape(-1)
        params[psid:peid] = ns_params.to(params.device)

    def _get_edges_as_mask(self):
        mask = torch.zeros([self.num_node_blocks, self.num_ch_nodes], dtype = torch.bool)
        mask[self.edge_ids[0,:], self.edge_ids[1,:]] = True

        return mask

    def _standardize_chs(self, chs):
        new_chs = []
        for cs in chs:
            if cs.is_input():
                new_cs = ProdNodes(
                    num_node_blocks = cs.num_node_blocks,
                    chs = [cs],
                    edge_ids = torch.arange(0, cs.num_node_blocks).reshape(-1, 1),
                    block_size = cs.block_size
                )
                new_chs.append(new_cs)
            else:
                new_chs.append(cs)

        return new_chs

    def _construct_edges(self, edge_ids: Optional[Union[Tensor,Sequence[Tensor]]], reorder: bool = False):
        if edge_ids is None:
            edge_ids = torch.cat(
                (torch.arange(self.num_node_blocks).unsqueeze(1).repeat(1, self.num_ch_node_blocks).reshape(1, -1),
                 torch.arange(self.num_ch_node_blocks).unsqueeze(0).repeat(self.num_node_blocks, 1).reshape(1, -1)),
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

                ch_nid_start += self.chs[cs_id].num_node_blocks

            edge_ids = torch.cat(edge_ids, dim = 1)

        if reorder:
            edge_ids = self._reorder_edges(edge_ids)

        if isinstance(edge_ids, np.ndarray):
            edge_ids = torch.from_numpy(edge_ids)

        if edge_ids.dim() == 2 and edge_ids.type() == torch.bool:
            assert edge_ids.size(0) == self.num_node_blocks and edge_ids.size(1) == self.num_ch_node_blocks
            x_ids, y_ids = torch.where(edge_ids)
            edge_ids = torch.stack((x_ids, y_ids), dim = 0)

        # Sanity checks
        assert edge_ids.size(0) == 2, "Expect `edge_ids.size(0) == 2`."
        assert torch.all(edge_ids[0,:] >= 0) and torch.all(edge_ids[1,:] >= 0), "Edge index underflow."
        assert torch.all(edge_ids[0,:] < self.num_node_blocks) and torch.all(edge_ids[1,:] < self.num_ch_node_blocks), "Edge index overflow."
        par_ns = torch.unique(edge_ids[0,:])
        assert par_ns.size(0) == self.num_node_blocks and par_ns.max() == self.num_node_blocks - 1, "Some node has no edge."

        self.edge_ids = edge_ids

    def _reorder_edges(self, edge_ids: Tensor):
        ids = torch.argsort(edge_ids[0,:] * self.num_ch_node_blocks + edge_ids[1,:])
        return edge_ids[:,ids].contiguous()

    def __repr__(self):
        scope_size = len(self.scope)
        return f"SumNodes(num_node_blocks={self.num_node_blocks}, block_size={self.block_size}, num_chs={self.num_chs}, num_edges={self.num_edges}, scope_size={scope_size})"
