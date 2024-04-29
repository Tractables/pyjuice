from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Any, Union

from .distributions import Distribution


class Literal(Distribution):
    """
    A class representing Literal (indicator) distributions.
    """
    def __init__(self, lit: Union[bool,int], p: float = 1.0):
        super(Literal, self).__init__()

        self.lit = int(lit) # Convert True/False to 1/0
        self.p = p

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "Literal"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return [self.lit, self.p]

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        return 0

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        return 0

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[Any] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """

        return torch.zeros(0)

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        lit = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)
        prob = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0).to(tl.int64)

        probs = tl.where(data == lit, prob, 1.0 - prob)
        log_probs = tl.log(probs)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        pass

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        pass

    @staticmethod
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):
        pass

    def _get_constructor(self):
        return Literal, {"lit": self.lit, "p": self.p}