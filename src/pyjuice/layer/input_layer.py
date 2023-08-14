from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Sequence, Dict

from pyjuice.nodes import InputNodes
from pyjuice.utils.grad_fns import ReverseGrad
from pyjuice.utils import BitSet
from .layer import Layer


class InputLayer(Layer, nn.Module):
    def __init__(self, nodes: Sequence[InputNodes]) -> None:
        nn.Module.__init__(self)
        Layer.__init__(self)

        self.nodes = nodes

        self.param_flows = None

        self.device = torch.device("cpu")

        self._used_external_params = False
    
    def to(self, device):
        nn.Module.to(self, device = device)

        self.device = device

    def init_param_flows(self, flows_memory: float = 0.0):
        batch_size = self._param_batch_size
        if self.param_flows is None \
                or (self.param_flows.dim() == 1 and batch_size > 1) \
                or (self.param_flows.dim() == 2 and batch_size != self.param_flows.size(1)):
            if batch_size == 1:
                shape = [self.param_flows_size]
            else:
                shape = [self.param_flows_size, batch_size]
            self.param_flows = torch.zeros(shape, device = self.device)
        else:
            assert self.param_flows.size(0) == self.param_flows_size
            self.param_flows[:] *= flows_memory

        return None

    def forward(self, used_external_params: bool):
        self._used_external_params = used_external_params

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        raise NotImplementedError()

    def mini_batch_em(self):
        raise NotImplementedError()

    def get_param_specs(self):
        raise NotImplementedError()

    def enable_partial_evaluation(self, fw_scopes: Optional[Union[Sequence[BitSet],Sequence[int]]] = None, 
                                  bk_scopes: Optional[Union[Sequence[BitSet],Sequence[int]]] = None, return_ids: bool = False):
        # Create cache if needed
        if not self.provided("scope2localids"):
            self._prepare_scope2nids()

        # Filter forward nodes
        if fw_scopes is not None:
            fw_local_ids = []
            for scope in fw_scopes:
                if isinstance(scope, int):
                    scope = BitSet.from_array([scope])

                if scope not in self.scope2localids:
                    continue

                fw_local_ids.append(self.scope2localids[scope])

            if return_ids:
                return torch.cat(fw_local_ids, dim = 0)
            else:
                self.fw_local_ids = torch.cat(fw_local_ids, dim = 0)

        # Filter backward nodes
        if bk_scopes is not None:
            bk_local_ids = []
            for scope in bk_scopes:
                if isinstance(scope, int):
                    scope = BitSet.from_array([scope])

                if scope not in self.scope2localids:
                    continue

                bk_local_ids.append(self.scope2localids[scope])

            if return_ids:
                return torch.cat(bk_local_ids, dim = 0)
            else:
                self.bk_local_ids = torch.cat(bk_local_ids, dim = 0)

    def disable_partial_evaluation(self, forward: bool = True, backward: bool = True):
        if forward:
            self.fw_local_ids = None

        if backward:
            self.bk_local_ids = None

    @staticmethod
    def _hook_params(grad_hook_idx: int, _inputs: List, layer_params: Dict):
        raise NotImplementedError()

    def _hook_param_grads(self, grad_hook_idx: int, _inputs: List, _inputs_grad: List):
        raise NotImplementedError()

    def _hook_input_grads(self, _inputs: List, _inputs_grad: List):
        pass

    def _prepare_scope2nids(self):
        if not hasattr(self, "scope2localids"):
            scope2localids = dict()

            local_nid = 0
            for ns in self.nodes:
                scope = ns.scope

                s_nid = local_nid
                e_nid = local_nid + ns.num_nodes

                with torch.no_grad():
                    if scope not in scope2localids:
                        scope2localids[scope] = [torch.zeros([0], dtype = torch.long)]

                    group_local_ids = torch.arange(s_nid, e_nid)
                    scope2localids[scope].append(group_local_ids)

                local_nid += ns.num_nodes

            self.scope2localids = {
                scope: torch.cat(ids, dim = 0).to(self.params.device) for scope, ids in scope2localids.items()
            }
