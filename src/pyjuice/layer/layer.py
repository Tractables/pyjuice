from __future__ import annotations

import torch
from typing import Union


class Layer():
    def __init__(self) -> None:
        pass

    def init_layer(self, params: Union[torch.Tensor,None]):
        raise NotImplementedError()