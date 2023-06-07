from __future__ import annotations

from typing import Tuple, Optional

from .distributions import Distribution


class DiscreteLogistic(Distribution):
    def __init__(self, input_range: Tuple[float,float], bin_count: int):
        super(DiscreteLogistic, self).__init__()

        self.input_range = input_range
        self.bin_count = bin_count

        self.bin_size = (self.input_range[1] - self.input_range[0]) / self.bin_count

    def raw2processed_params(self, num_nodes: int, params: torch.Tensor):
        return params

    def processed2raw_params(self, num_nodes: int, params: torch.Tensor):
        return params

    def init_parameters(num_nodes: int, perturbation: float, **kwargs):
        raise NotImplementedError()

    def __getstate__(self):
        state = {
            "input_range": self.input_range,
            "bin_count": self.bin_count
        }
        return state

    def __setstate__(self, state):
        self.input_range = state["input_range"]
        self.bin_count = state["bin_count"]