import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional

from pyjuice.model import TensorCircuit


class CircuitOptimizer():

    SUPPORTED_OPTIM_METHODS = ["EM"]

    def __init__(self, pc: TensorCircuit, base_optimizer: Optional[Optimizer] = None, method: str = "EM", lr: float = 0.1,
                 pseudocount: float = 0.1):
        
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