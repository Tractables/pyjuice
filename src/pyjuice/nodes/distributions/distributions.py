from __future__ import annotations

import torch


class Distribution():
    def __init__(self):
        pass

    def raw2processed_params(self, params: torch.Tensor):
        raise NotImplementedError()

    def processed2raw_params(self, params: torch.Tensor):
        raise NotImplementedError()
