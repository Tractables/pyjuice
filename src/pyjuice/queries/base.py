from __future__ import annotations

import torch
from typing import Optional, Union, Callable

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit


def query(pc: TensorCircuit, inputs: torch.Tensor,
          run_backward: bool = True,
          fw_input_fn: Optional[Union[str,Callable]] = None,
          bk_input_fn: Optional[Union[str,Callable]] = None,
          fw_output_fn: Optional[Callable] = None, **kwargs):
    """
    A general-purpose entry point for running queries on a PC. It runs a forward pass (and optionally a
    backward pass) of the PC, optionally with custom input-layer functions, and is the common backend of
    :func:`marginal`, :func:`conditional`, and :func:`sample`.

    :param pc: the input PC
    :type pc: TensorCircuit

    :param inputs: input tensor of size [B, num_vars], or a custom shape paired with `fw_input_fn`
    :type inputs: torch.Tensor

    :param run_backward: whether to run the backward pass after the forward pass
    :type run_backward: bool

    :param fw_input_fn: an optional custom function (or the name of an input-layer method) for the forward pass of input layers
    :type fw_input_fn: Optional[Union[str,Callable]]

    :param bk_input_fn: an optional custom function (or the name of an input-layer method) for the backward pass of input layers
    :type bk_input_fn: Optional[Union[str,Callable]]

    :param fw_output_fn: an optional function applied to `pc` right after the forward pass; if provided, its return value is returned immediately and no backward pass is run
    :type fw_output_fn: Optional[Callable]

    :returns: the log-likelihoods from the forward pass (when `run_backward` is `False`), the output of `fw_output_fn` (when provided), or `None` (after a backward pass)
    """

    # Run forward pass
    lls = pc.forward(inputs, input_layer_fn = fw_input_fn, **kwargs)

    if fw_output_fn is not None:
        return fw_output_fn(pc)
    elif not run_backward:
        return lls

    # (Optionally) run backward pass
    assert bk_input_fn is not None
    pc.backward(inputs, input_layer_fn = bk_input_fn, compute_param_flows = False, **kwargs)

    return None