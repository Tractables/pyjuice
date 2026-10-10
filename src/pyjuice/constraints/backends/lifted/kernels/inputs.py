"""
Input nodes under the lifted plan. An observed token keeps pyjuice's own log-probability (the input layers
write it, as in a :class:`TensorCircuit`); a missing one is summed per token class, by the input distribution's
own function in :mod:`pyjuice.constraints.distributions`.
"""

import torch

from ....distributions import lookup


def class_masses(pc, token_class: torch.Tensor, num_classes: int, class_mars: torch.Tensor, input_start: int):
    """Write the log-mass of every token class of every input node into ``class_mars`` [num_input_rows, C]."""
    for layer in pc.input_layer_group:
        first, end = layer._output_ind_range
        class_mars[first - input_start:end - input_start] = lookup(layer.dist).class_masses(layer, token_class,
                                                                                            num_classes)
