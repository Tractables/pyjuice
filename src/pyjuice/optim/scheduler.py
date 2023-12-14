import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, Optional, List

from .optim import CircuitOptimizer


class CircuitScheduler():

    SUPPORTED_SCHEDULER_METHODS = ["constant", "multi_linear"]

    def __init__(self, optimizer: CircuitOptimizer, base_scheduler: Optional[LRScheduler] = None, 
                 method: str = "constant", **kwargs):

        self.optimizer = optimizer
        
        self.base_scheduler = base_scheduler

        assert method in self.SUPPORTED_SCHEDULER_METHODS, f"Unsupported optimization method {method} for circuits."
        self.method = method

        for key, value in kwargs.items():
            self.__dict__[key] = value

        self.step_count = 0

        if self.method == "constant":
            self.lr = self.optimizer.lr
        elif self.method == "multi_linear":
            assert "lrs" in self.__dict__
            assert "milestone_steps" in self.__dict__

            if isinstance(self.lrs, List):
                self.lrs = np.array(self.lrs)
            else:
                try:
                    self.lrs = np.array(list(self.lrs))
                except Exceptions:
                    pass

            if isinstance(self.milestone_steps, List):
                self.milestone_steps = np.array(self.milestone_steps)
            else:
                try:
                    self.milestone_steps = np.array(list(self.milestone_steps))
                except Exceptions:
                    pass
            
            assert isinstance(self.lrs, np.ndarray)
            assert isinstance(self.milestone_steps, np.ndarray)
        else:
            raise ValueError(f"Unknown method {self.method}.")

    def step(self):
        if self.base_scheduler is not None:
            self.base_scheduler.step()

        if self.method == "constant":
            self.optimizer.lr = self.lr
        elif self.method == "multi_linear":
            if self.step_count >= self.milestone_steps[-1]:
                self.optimizer.lr = self.lrs[-1]
            else:
                idx = np.sum(self.milestone_steps < self.step_count)
                if idx == 0:
                    self.optimizer.lr = self.lrs[0]
                else:
                    self.optimizer.lr = self.lrs[idx-1] + (self.lrs[idx] - self.lrs[idx-1]) * \
                        (self.step_count - self.milestone_steps[idx-1]) / \
                        (self.milestone_steps[idx] - self.milestone_steps[idx-1])

        self.step_count += 1

    def state_dict(self):
        if self.base_optimizer is not None:
            state_dict = self.base_optimizer.state_dict()
        else:
            state_dict = dict()

        state_dict["pc_states"] = dict()
        for key, value in self.__dict__.items():
            if key != "optimizer" and key != "base_scheduler":
                state_dict["pc_states"][key] = value

    def load_state_dict(self, state_dict: Dict):

        for key, value in state_dict["circuit_states"].items():
            self.__dict__[key] = value

        del state_dict["pc_states"]

        if self.base_optimizer is not None:
            self.base_optimizer.load_state_dict(state_dict)