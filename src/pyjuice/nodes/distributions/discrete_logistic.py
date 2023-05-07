from __future__ import annotations

from typing import Tuple, Optional

from .distributions import Distribution


class DiscreteLogistic(Distribution):
    def __init__(self, input_range: Tuple[float,float], bin_count: int):
        super(DiscreteLogistic, self).__init__()

        self.input_range = input_range
        self.bin_count = bin_count

        self.bin_size = (self.input_range[1] - self.input_range[0]) / self.bin_count