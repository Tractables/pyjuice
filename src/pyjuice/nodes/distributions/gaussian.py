from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Optional

from .distributions import Distribution


class Gaussian(Distribution):
    def __init__(self, mu: Optional[float] = None, sigma: Optional[float] = None):
        """
        `mu` and `sigma` are used to specify (approximately) the mean and std of the data.
        This is used for parameter initialization.
        Note: the parameters will NOT be initialized directly using the values of `mu` and `sigma`,
              perturbations will be added. You can specify the initialization behavior by passing
              `perturbation`, `mu`, and `sigma` to the `init_parameters` function.
        """
        super(Gaussian, self).__init__()

        self.mu = mu
        self.sigma = sigma

    def get_signature(self):
        return "Gaussian"

    def get_metadata(self):
        return []

    def num_parameters(self):
        return 2

    def num_param_flows(self):
        return 3

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        mu = kwargs["mu"] if "mu" in kwargs else self.mu
        sigma = kwargs["sigma"] if "sigma" in kwargs else self.sigma
        assert (mu is not None), "`mu` should be provided either during initialization or when calling `init_parameters`."
        assert (sigma is not None), "`sigma` should be provided either during initialization or when calling `init_parameters`."

        mus = torch.normal(mean = torch.ones([num_nodes]) * mu, std = sigma)
        sigmas = torch.exp(torch.rand([num_nodes]) * -perturbation) * sigma

        return torch.stack((mus, sigmas), dim = 1).reshape(-1).contiguous()

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, metadata, mask, num_vars_per_node, BLOCK_SIZE):
        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sigma = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)

        centered_val = (data - mu) / sigma
        log_probs = -0.5 * centered_val * centered_val - 0.5 * 1.8378770664093453 - tl.log(sigma) # the constant is `math.log(2 * math.pi)`

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, data, flows, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   metadata, mask, num_vars_per_node, BLOCK_SIZE):
        stat1 = data * flows
        stat2 = data * data * flows
        stat3 = flows

        tl.atomic_add(param_flows_ptr + s_pfids, stat1, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 1, stat2, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 2, stat3, mask = mask)

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, batch_size, BLOCK_SIZE, seed):

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

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sigma = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)

        stat1 = tl.load(param_flows_ptr + s_pfids, mask = mask)
        stat2 = tl.load(param_flows_ptr + s_pfids + 1, mask = mask)
        stat3 = tl.load(param_flows_ptr + s_pfids + 2, mask = mask)

        updated_mu = stat1 / (stat3 + pseudocount)
        updated_sigma2 = (stat2 - updated_mu * updated_mu * stat3) / (stat3 + pseudocount)

        # Treating Gaussians as exponential distributions, we can do EM updates by 
        # linear interpolation of `mu` and `sigma^2 - mu^2`
        new_mu = (1.0 - step_size) * mu + step_size * updated_mu
        new_sigma2_min_mu2 = (1.0 - step_size) * (sigma * sigma - mu * mu) + \
            step_size * (updated_sigma2 - updated_mu * updated_mu)
        new_sigma = tl.sqrt(new_sigma2_min_mu2 + new_mu * new_mu)

        tl.store(params_ptr + s_pids, new_mu, mask = mask)
        tl.store(params_ptr + s_pids + 1, new_sigma, mask = mask)