from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Any, Union

from .distributions import Distribution


class Indicator(Distribution):
    """
    A class representing Indicator distributions. It is a special case of `Literal`, and handles
    the case where we want to have k nodes on X representing X = 0, X = 1, ..., X = k-1.
    """
    def __init__(self, num_states: int):
        super(Indicator, self).__init__()

        self.num_states = num_states

        self.requires_individual_node_counts = True

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return f"Indicator"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return []

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        return 1

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
        assert "individual_node_counts" in kwargs

        params = torch.zeros([num_nodes], dtype = torch.float32)

        sid = 0
        for node_count in kwargs["individual_node_counts"]:
            params[sid:sid+node_count] = torch.arange(node_count).float()
            sid += node_count

        return params

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        lits = tl.load(params_ptr + s_pids, mask = mask, other = 0).to(tl.int64)
        probs = tl.where(data == lit, 1.0, 0.0)
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
        return Literal, {"lit": self.lit}
