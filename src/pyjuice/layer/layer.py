from __future__ import annotations

import torch
from typing import Union
from pyjuice.graph import PartitionNode, InnerRegionNode


class Layer():
    def __init__(self, layer_id) -> None:
        self.layer_id = layer_id

    def init_layer(self, params: Union[torch.Tensor,None]):
        raise NotImplementedError()