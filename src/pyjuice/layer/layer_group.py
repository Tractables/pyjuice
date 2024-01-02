from __future__ import annotations

import torch
import torch.nn as nn

from typing import Sequence

from .layer import Layer
from .input_layer import InputLayer
from .prod_layer import ProdLayer
from .sum_layer import SumLayer


class LayerGroup(nn.Module):
    def __init__(self, layers: Sequence[Layer]):
        super(LayerGroup, self).__init__()

        assert len(layers) >= 1, "A `LayerGroup` must contains at least 1 layer."

        for i in range(1, len(layers)):
            assert type(layers[i]) == type(layers[0])

        if isinstance(layers[0], InputLayer):
            self.layer_type = "input"
        elif isinstance(layers[0], ProdLayer):
            self.layer_type = "prod"
        else:
            assert isinstance(layers[0], SumLayer)
            self.layer_type = "sum"

        self.num_layers = len(layers)

        self.layers = []
        for i, layer in enumerate(layers):
            self.add_module(f"layer_{i}", layer)
            self.layers.append(layer)

    def to(self, device):
        super(LayerGroup, self).to(device)

        for layer in self.layers:
            layer.to(device)

    def forward(self, *args, **kwargs):

        for layer in self.layers:
            layer.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):

        for layer in self.layers:
            layer.backward(*args, **kwargs)

    def enable_partial_evaluation(self, *args, **kwargs):

        for layer in self.layers:
            layer.enable_partial_evaluation(*args, **kwargs)

    def disable_partial_evaluation(self, *args, **kwargs):

        for layer in self.layers:
            layer.disable_partial_evaluation(*args, **kwargs)

    def is_input(self):
        return self.layer_type == "input"
        
    def is_prod(self):
        return self.layer_type == "prod"

    def is_sum(self):
        return self.layer_type == "sum"

    def __len__(self):
        self.num_layers

    def __getitem__(self, idx):
        return self.layers[idx]

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.num_layers:
            layer = self.layers[self.iter_idx]
            self.iter_idx += 1
            return layer
        else:
            raise StopIteration

    def _prepare_scope2nids(self, *args, **kwargs):
        
        if self.is_prod():
            prod_scope_eleids = list()
            for layer in self.layers:
                prod_scope_eleids.extend(layer._prepare_scope2nids(*args, **kwargs))

            return prod_scope_eleids
        
        else:
            for layer in self.layers:
                layer._prepare_scope2nids(*args, **kwargs)
