import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, Optional

from pyjuice.model import ProbCircuit


class CircuitOptimizer():

    SUPPORTED_OPTIM_METHODS = ["EM"]

    def __init__(self, circuit: ProbCircuit, base_optimizer: Optional[Optimizer] = None, method: str = "EM", lr: float = 0.1,
                 pseudocount: float = 0.1):
        
        self.circuit = circuit

        self.base_optimizer = base_optimizer

        assert method in self.SUPPORTED_OPTIM_METHODS, f"Unsupported optimization method {method} for circuits."
        self.method = method

        self.lr = lr
        self.pseudocount = pseudocount

    def zero_grad(self, flows_memory: float = 0.0):
        if self.base_optimizer is not None:
            self.base_optimizer.zero_grad()

        self.circuit._optim_hyperparams["flows_memory"] = flows_memory

    def step(self, closure = None):
        if self.base_optimizer is not None:
            self.base_optimizer.step()

        if self.method == "EM":
            self.circuit.mini_batch_em(
                step_size = self.lr,
                pseudocount = self.pseudocount
            )
        else:
            raise ValueError(f"Unknown circuit optimization method {self.method}.")

    def state_dict(self):
        if self.base_optimizer is not None:
            state_dict = self.base_optimizer.state_dict()
        else:
            state_dict = dict()

        state_dict["circuit_states"] = {
            "method": self.method, 
            "lr": self.lr, 
            "pseudocount": self.pseudocount
        }

    def load_state_dict(self, state_dict: Dict):
        circuit_states = state_dict["circuit_states"]
        
        self.method = circuit_states["method"]
        self.lr = circuit_states["lr"]
        self.pseudocount = circuit_states["pseudocount"]

        del state_dict["circuit_states"]

        if self.base_optimizer is not None:
            self.base_optimizer.load_state_dict(state_dict)
        