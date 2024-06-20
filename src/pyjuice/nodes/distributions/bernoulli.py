from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Any

from .distributions import Distribution


class Bernoulli(Distribution):
    """
    A class representing Bernoulli distributions.
    """
    def __init__(self):
        super(Bernoulli, self).__init__()

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "Bernoulli"

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
        return 2

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[Any] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        if params is not None:
            assert isinstance(params, torch.Tensor)
            assert params.numel() == num_nodes * self.num_parameters()
            return params

        params = torch.exp(torch.rand([num_nodes, 2]) * -perturbation)
        params /= params.sum(dim = 1, keepdim = True)

        return params[:,0].contiguous()

    def get_data_dtype(self):
        """
        Get the data dtype for the distribution.
        """
        return torch.bool

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        probs = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        probs = tl.where(data > 0, probs, 1.0 - probs)
        log_probs = tl.log(probs)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        pf_offsets = s_pfids + tl.where(data > 0, 0, 1)
        tl.atomic_add(param_flows_ptr + pf_offsets, flows, mask = mask)

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):

        rnd_val = tl.rand(seed, tl.arange(0, BLOCK_SIZE))

        param = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sampled_id = tl.where(rnd_val < param, 1, 0)

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_id, mask = mask)

    @staticmethod
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):

        flow1 = tl.load(param_flows_ptr + s_pfids, mask = mask, other = 0)
        flow0 = tl.load(param_flows_ptr + s_pfids + 1, mask = mask, other = 0)

        param = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        updated_param = (flow1 + 0.5 * pseudocount) / (flow1 + flow0 + pseudocount)

        new_param = (1.0 - step_size) * param + step_size * updated_param
        tl.store(params_ptr + s_pids, new_param, mask = mask)

    def _get_constructor(self):
        return Bernoulli, {}