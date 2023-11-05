from __future__ import annotations

import torch


class Distribution():
    def __init__(self):
        pass

    def get_signature(self):
        raise NotImplementedError()

    def get_metadata(self):
        return [] # no metadata

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        raise NotImplementedError()

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        raise NotImplementedError()

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        raise NotImplementedError()

    @staticmethod
    def fw_mar_fn(*args, **kwargs):
        """
        """
        raise NotImplementedError()

    @staticmethod
    def bk_flow_fn(*args, **kwargs):
        """
        """
        raise NotImplementedError()

    @staticmethod
    def em_fn(*args, **kwargs):
        """
        """
        raise NotImplementedError()
