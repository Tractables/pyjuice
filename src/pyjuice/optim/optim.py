import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional

from pyjuice.model import TensorCircuit


class CircuitOptimizer():
    """
    A PyTorch-style optimizer for PCs that wraps PyJuice's parameter-update routines (e.g., EM).

    It mirrors the :class:`torch.optim.Optimizer` API (:func:`zero_grad`, :func:`step`,
    :func:`state_dict`, :func:`load_state_dict`), so a training loop looks the same as a standard
    PyTorch loop. An optional `base_optimizer` can be supplied to additionally update any non-PC
    (e.g., neural network) parameters in the same `step`.

    :param pc: the PC to optimize
    :type pc: TensorCircuit

    :param base_optimizer: an optional PyTorch optimizer for non-PC parameters, stepped alongside the PC update
    :type base_optimizer: Optional[torch.optim.Optimizer]

    :param method: the parameter-update method; one of `"EM"`, `"Viterbi"`, or `"GeneralEM"`
    :type method: str

    :param lr: the step size (learning rate) of the PC parameter update
    :type lr: float

    :param pseudocount: the Laplace smoothing pseudocount added during the update
    :type pseudocount: float
    """

    SUPPORTED_OPTIM_METHODS = ["EM", "Viterbi", "GeneralEM"]

    def __init__(self, pc: TensorCircuit, base_optimizer: Optional[Optimizer] = None, method: str = "EM", lr: float = 0.1,
                 pseudocount: float = 0.1, **kwargs):

        self.pc = pc

        self.base_optimizer = base_optimizer

        assert method in self.SUPPORTED_OPTIM_METHODS, f"Unsupported optimization method {method} for PCs."
        self.method = method

        self.lr = lr
        self.pseudocount = pseudocount

    def zero_grad(self):
        if self.base_optimizer is not None:
            self.base_optimizer.zero_grad()

        self.pc.init_param_flows(flows_memory = 0.0)

    def step(self, closure = None):
        if self.base_optimizer is not None:
            self.base_optimizer.step()

        if self.method == "EM":
            self.pc.mini_batch_em(
                step_size = self.lr,
                pseudocount = self.pseudocount
            )
        else:
            raise ValueError(f"Unknown PC optimization method {self.method}.")

    def state_dict(self):
        if self.base_optimizer is not None:
            state_dict = self.base_optimizer.state_dict()
        else:
            state_dict = dict()

        state_dict["pc_states"] = {
            "method": self.method, 
            "lr": self.lr, 
            "pseudocount": self.pseudocount
        }

    def load_state_dict(self, state_dict: Dict):
        pc_states = state_dict["pc_states"]
        
        self.method = pc_states["method"]
        self.lr = pc_states["lr"]
        self.pseudocount = pc_states["pseudocount"]

        del state_dict["pc_states"]

        if self.base_optimizer is not None:
            self.base_optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        assert self.base_optimizer is not None
        return self.base_optimizer.param_groups