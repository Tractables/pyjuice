from __future__ import annotations

import torch

from .distributions import Distribution


class Categorical(Distribution):
    def __init__(self, num_cats: int):
        super(Categorical, self).__init__()

        self.num_cats = num_cats

    def raw2processed_params(self, params: torch.Tensor):
        return params.reshape(-1, self.num_cats)

    def processed2raw_params(self, params: torch.Tensor):
        return params.reshape(-1)