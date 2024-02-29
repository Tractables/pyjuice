from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Union, Callable
from functools import partial

from pyjuice.nodes import CircuitNodes
from pyjuice.model import TensorCircuit
from .base import query

from .conditional import _conditional_fw_input_fn


def marginal(pc: TensorCircuit, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, 
             fw_input_fn: Optional[Union[str,Callable]] = None, **kwargs):
    """
    Compute the marginal probability given the assignment of a subset of variables, i.e., P(e).

    :param pc: the input PC
    :type pc: TensorCircuit

    :param data: data of size [B, num_vars] (hard evidence) or a custom shape paired with `fw_input_fn`
    :type data: torch.Tensor

    :param missing_mask: a boolean mask indicating marginalized variables; the size can be [num_vars] or [B, num_vars]
    :type missing_mask: torch.Tensor

    :param fw_input_fn: an optional custom function for the forward pass of input layers
    :type fw_input_fn: Optional[Union[str,Callable]]
    """
    
    lls = query(pc, inputs = data, run_backward = False, 
                fw_input_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn,
                missing_mask = missing_mask, **kwargs)

    return lls