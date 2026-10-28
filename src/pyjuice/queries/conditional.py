from __future__ import annotations

import torch
from typing import Union, Callable, Optional, Sequence
from functools import partial

from pyjuice.model import TensorCircuit
from .base import query
from . import distributions


def _conditional_fw_input_fn(layer, inputs, node_mars, **kwargs):
    distributions.forward(layer, inputs, node_mars, **kwargs)


def _conditional_bk_input_fn(layer, inputs, node_flows, node_mars, outputs = None, **kwargs):
    outputs.append(
        distributions.backward(layer, inputs, node_flows, node_mars, **kwargs)
    )


## Main API ##


def conditional(pc: TensorCircuit, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                target_vars: Optional[Sequence[int]] = None,
                fw_input_fn: Optional[Union[str,Callable]] = None,
                bk_input_fn: Optional[Union[str,Callable]] = None, **kwargs):
    """
    Compute the conditional probability given hard or soft evidence, i.e., P(o|e).

    :param pc: the input PC
    :type pc: TensorCircuit

    :param data: data of size [B, num_vars] (hard evidence) or a custom shape paired with `fw_input_fn`
    :type data: torch.Tensor

    :param missing_mask: a boolean mask indicating marginalized variables; the size can be [num_vars] or [B, num_vars]
    :type missing_mask: torch.Tensor

    :param fw_input_fn: an optional custom function for the forward pass of input layers
    :type fw_input_fn: Optional[Union[str,Callable]]

    :param bk_input_fn: an optional custom function for the backward pass of input layers
    :type bk_input_fn: Optional[Union[str,Callable]]

    :note: the backward pass runs with log-space flows and without flow modification (`logspace_flows = True`,
           `allow_modify_flows = False`) -- the regime external sum parameters need -- so a custom `bk_input_fn`
           receives log-space flows.
    """

    assert kwargs.setdefault("logspace_flows", True) and not kwargs.setdefault("allow_modify_flows", False), \
        "`conditional` runs the backward pass with `logspace_flows = True` and `allow_modify_flows = False`."

    outputs = []

    _wrapped_bk_input_fn = partial(_conditional_bk_input_fn, outputs = outputs)

    kwargs["target_vars"] = target_vars

    query(pc, inputs = data, run_backward = True,
          fw_input_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn,
          bk_input_fn = _wrapped_bk_input_fn if bk_input_fn is None else bk_input_fn,
          missing_mask = missing_mask, **kwargs)

    return outputs[0]
