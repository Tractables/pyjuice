from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Sequence, Dict

from pyjuice.nodes import InputNodes
from pyjuice.utils.grad_fns import ReverseGrad
from .layer import Layer

# Try to enable tensor cores
torch.set_float32_matmul_precision('high')


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

    def backward(self):
        raise NotImplementedError()

    def mini_batch_em(self):
        raise NotImplementedError()

    def get_param_specs(self):
        raise NotImplementedError()

    @staticmethod
    def _hook_params(grad_hook_idx: int, _inputs: List, layer_params: Dict):
        raise NotImplementedError()

    def _hook_param_grads(self, grad_hook_idx: int, _inputs: List, _inputs_grad: List):
        raise NotImplementedError()

    def _hook_input_grads(self, _inputs: List, _inputs_grad: List):
        pass
