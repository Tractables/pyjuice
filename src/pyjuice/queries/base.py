from __future__ import annotations

import torch
from typing import Union, Callable

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit


def query(pc: TensorCircuit, inputs: torch.Tensor, 
          run_backward: bool = True,
          fw_input_fn: Optional[Union[str,Callable]] = None, 
          bk_input_fn: Optional[Union[str,Callable]] = None, 
          fw_output_fn: Optional[Callable] = None, **kwargs):

    # Run forward pass
    lls = pc.forward(inputs, input_layer_fn = fw_input_fn, **kwargs)

    if fw_output_fn is not None:
        return fw_output_fn(pc)
    elif not run_backward:
        return lls

    # (Optionally) run backward pass
    assert bk_input_fn is not None
    pc.backward(input_layer_fn = bk_input_fn, compute_param_flows = False, **kwargs)

    return None