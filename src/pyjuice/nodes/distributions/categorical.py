from __future__ import annotations

import torch

from .distributions import Distribution


class Categorical(Distribution):
    def __init__(self, num_cats: int):
        super(Categorical, self).__init__()

        self.num_cats = num_cats

    def raw2processed_params(self, num_nodes: int, params: torch.Tensor):
        assert params.dim() == 2 and params.size(0) == num_nodes and params.size(1) == self.num_cats, \
            f"Input parameters should have size ({num_nodes}, {self.num_cats})"
        params = self.normalize_parameters(params)
        return params.reshape(-1, self.num_cats)

    def processed2raw_params(self, num_nodes: int, params: torch.Tensor):
        return params.reshape(-1)

    def init_parameters(self, num_nodes: int, perturbation: float, **kwargs):
        params = torch.exp(torch.rand([num_nodes, self.num_cats]) * -perturbation)
        params = self.normalize_parameters(params)
        return params.reshape(num_nodes * self.num_cats)

    def normalize_parameters(self, params, pseudocount: float = 1.0):
        params = params + pseudocount / self.num_cats
        params /= params.sum(dim = 1, keepdim = True)
        return params

    def __getstate__(self):
        state = {"num_cats": self.num_cats}
        return state

    def __setstate__(self, state):
        self.num_cats = state["num_cats"]