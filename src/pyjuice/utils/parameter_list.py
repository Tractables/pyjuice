import torch
import torch.nn as nn
from typing import Iterator, Any


class FastParamList(nn.ParameterList):

    def __getitem__(self, idx) -> Any:
        return getattr(self, str(idx))

    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))
