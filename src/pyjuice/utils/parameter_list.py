import torch
import torch.nn as nn
from typing import Iterator, Any


class FastParamList(nn.ParameterList):

    def __getitem__(self, idx) -> Any:
        # The registered parameter straight from `_parameters`, which is where `nn.Module.__getattr__`
        # finds it -- without that call's overhead (this runs ~500 times a step). Anything not there
        # takes the old route, so an invalid index fails exactly as before.
        try:
            return self._parameters[str(idx)]
        except KeyError:
            return getattr(self, str(idx))

    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))
