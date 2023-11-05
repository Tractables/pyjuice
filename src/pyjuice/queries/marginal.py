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


def marginal(pc: TensorCircuit, missing_mask: Optional[torch.Tensor] = None,
             fw_input_fn: Optional[Union[str,Callable]] = None, **kwargs):

    if missing_mask is not None:
        missing_mask = missing_mask.permute(1, 0)
        B = missing_mask.size(1)
    elif "soft_evidence" in kwargs:
        B = kwargs["soft_evidence"].size(0)
    else:
        raise ValueError("Either `missing_mask` or `soft_evidence` should be provided.")
    
    lls = query(pc, inputs = torch.zeros([B, 1]), run_backward = False, 
                fw_input_fn = _conditional_fw_input_fn if fw_input_fn is None else fw_input_fn,
                missing_mask = missing_mask, **kwargs)

    return lls