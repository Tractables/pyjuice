from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Any

from .distributions import Distribution


class Gaussian(Distribution):
    """
    A class representing Gaussian distributions.

    :note: `mu` and `sigma` are used to specify (approximately) the mean and std of the data. This is used for parameter initialization.

    :note: The parameters will NOT be initialized directly using the values of `mu` and `sigma`, perturbations will be added. You can specify the initialization behavior by passing `perturbation`, `mu`, and `sigma` to the `init_parameters` function.

    :param mu: mean of the Gaussian
    :type mu: float

    :param sigma: standard deviation of the Gaussian
    :type sigma: float
    """

    def __init__(self, mu: Optional[float] = None, sigma: Optional[float] = None, min_sigma: float = 0.01):
        super(Gaussian, self).__init__()

        self.mu = mu
        self.sigma = sigma
        self.min_sigma = min_sigma

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "Gaussian"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return [self.min_sigma]

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        return 2

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        return 3

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[Any] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        if params is not None:
            assert isinstance(params, torch.Tensor)
            assert params.numel() == num_nodes * self.num_parameters()
            return params.contiguous()

        mu = kwargs["mu"] if "mu" in kwargs else self.mu
        sigma = kwargs["sigma"] if "sigma" in kwargs else self.sigma
        assert (mu is not None), "`mu` should be provided either during initialization or when calling `init_parameters`."
        assert (sigma is not None), "`sigma` should be provided either during initialization or when calling `init_parameters`."

        mus = torch.normal(mean = torch.ones([num_nodes]) * mu, std = sigma)
        sigmas = (torch.exp(torch.rand([num_nodes]) * -0.04 * perturbation) * sigma).clamp(min = self.min_sigma)

        return torch.stack((mus, sigmas), dim = 1).reshape(-1).contiguous()

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sigma = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)

        centered_val = (data - mu) / sigma
        log_probs = -0.5 * centered_val * centered_val - 0.5 * 1.8378770664093453 - tl.log(sigma) # the constant is `math.log(2 * math.pi)`

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        stat1 = data * flows
        stat2 = data * data * flows
        stat3 = flows

        tl.atomic_add(param_flows_ptr + s_pfids, stat1, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 1, stat2, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 2, stat3, mask = mask)

    @staticmethod
    def bk_flow_mask_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                        s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE, TILE_SIZE_K):
        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sigma = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)

        stat1 = mu * flows
        stat2 = (sigma * sigma + mu * mu) * flows
        stat3 = flows

        tl.atomic_add(param_flows_ptr + s_pfids, stat1, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 1, stat2, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 2, stat3, mask = mask)

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):

        rnd_val = tl.randn(seed, tl.arange(0, BLOCK_SIZE))

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sigma = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)
        
        sample = mu + rnd_val * sigma

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sample, mask = mask)

    @staticmethod
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):
        # Get `min_sigma` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        min_sigma = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sigma = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)
        ori_theta1 = mu
        ori_theta2 = sigma * sigma + mu * mu

        stat1 = tl.load(param_flows_ptr + s_pfids, mask = mask)
        stat2 = tl.load(param_flows_ptr + s_pfids + 1, mask = mask)
        stat3 = tl.load(param_flows_ptr + s_pfids + 2, mask = mask)

        new_theta1 = stat1 / (stat3 + 1e-10)
        new_theta2 = stat2 / (stat3 + 1e-10)

        # Get the updated natural parameters
        updated_theta1 = (1.0 - step_size) * ori_theta1 + step_size * new_theta1
        updated_theta2 = (1.0 - step_size) * ori_theta2 + step_size * new_theta2

        # Reconstruct `mu` and `sigma` from the expectation parameters (moment matching)
        updated_mu = updated_theta1
        updated_sigma2 = updated_theta2 - updated_mu * updated_mu
        updated_sigma = tl.where(updated_sigma2 < min_sigma * min_sigma, min_sigma, tl.sqrt(updated_sigma2))

        tl.store(params_ptr + s_pids, updated_mu, mask = mask)
        tl.store(params_ptr + s_pids + 1, updated_sigma, mask = mask)

    def _get_constructor(self):
        return Gaussian, {"mu": self.mu, "sigma": self.sigma, "min_sigma": self.min_sigma}
