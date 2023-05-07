from __future__ import annotations

from pyjuice.nodes.distributions import *
from .categorical_layer import CategoricalLayer
from .discrete_logistic_layer import DiscreteLogisticLayer


def layerize(ltype):
    if isinstance(ltype, Categorical):
        return CategoricalLayer
    elif isinstance(ltype, DiscreteLogistic):
        return DiscreteLogisticLayer
    else:
        raise NotImplementedError()
