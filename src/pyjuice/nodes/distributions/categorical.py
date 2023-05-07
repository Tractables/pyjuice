from __future__ import annotations

from .distributions import Distribution


class Categorical(Distribution):
    def __init__(self, num_cats: int):
        super(Categorical, self).__init__()

        self.num_cats = num_cats