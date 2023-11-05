from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Tuple

from .distributions import Distribution


class DiscreteLogistic(Distribution):
    def __init__(self, val_range: Tuple[float,float], num_cats: int):
        super(DiscreteLogistic, self).__init__()

        self.val_range = val_range
        self.num_cats = num_cats

    def get_signature(self):
        return "DiscreteLogistic"

    def get_metadata(self):
        return [self.val_range[0], self.val_range[1], self.num_cats]

    def num_parameters(self):
        return 2

    def num_param_flows(self):
        return 3

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        mu = torch.normal(mean = torch.ones([num_nodes]) * kwargs["mu"], std = kwargs["sigma"])
        s = torch.exp(torch.rand([num_nodes]) * -perturbation) * kwargs["sigma"] * math.sqrt(3) / math.pi

        return torch.stack((mu, s), dim = 1).reshape(-1).contiguous()

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, metadata, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `val_range` and `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
        range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        s = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)

        # 0: (-inf, range_low]; ...; num_cats - 1: [range_high, inf)
        interval = (range_high - range_low) / (num_cats - 2)
        vhigh = data * interval + range_low
        vlow = vhigh - interval

        cdfhigh = tl.where(data == num_cats - 1, 1.0, 1.0 / (1.0 + tl.exp((mu - vhigh) / s)))
        cdflow = tl.where(data == 0, 0.0, 1.0 / (1.0 + tl.exp((mu - vlow) / s)))

        log_probs = tl.log(cdfhigh - cdflow)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, data, flows, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   metadata, mask, num_vars_per_node, BLOCK_SIZE):
        stat1 = data * flows
        stat2 = tl.pow(data, 2) * flows
        stat3 = flows

        tl.atomic_add(param_flows_ptr + s_pfids, stat1, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 1, stat2, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 2, stat3, mask = mask)

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, batch_size, BLOCK_SIZE, seed):
        # Get `val_range` and `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
        range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

        rnd_val = tl.rand(seed, tl.arange(0, BLOCK_SIZE))

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        s = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)
        
        # Get sample using the quantile function
        sample = mu + s * tl.log(rnd_val / (1.0 - rnd_val))

        # Discretize
        interval = (range_high - range_low) / (num_cats - 2)
        bin_val = (sample - range_low) / interval
        sampled_id = tl.where(bin_val < 0, 0, tl.where(bin_val > num_cats - 1, num_cats, tl.floor(bin_val).to(tl.int64) + 1)) # TODO check

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_id, mask = mask)

    @staticmethod
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        sigma = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0) * math.pi / math.sqrt(3.0)

        stat1 = tl.load(param_flows_ptr + s_pfids, mask = mask)
        stat2 = tl.load(param_flows_ptr + s_pfids + 1, mask = mask)
        stat3 = tl.load(param_flows_ptr + s_pfids + 2, mask = mask)

        updated_mu = stat1 / (stat3 + pseudocount)
        updated_sigma2 = (stat2 - tl.pow(updated_mu, 2) * stat3) / (stat3 + pseudocount)

        # Treating Gaussians as exponential distributions, we can do EM updates by 
        # linear interpolation of `mu` and `sigma^2 - mu^2`
        new_mu = (1.0 - step_size) * mu + step_size * updated_mu
        new_sigma2_min_mu2 = (1.0 - step_size) * (tl.pow(sigma, 2) - tl.pow(mu, 2)) + \
            step_size * (updated_sigma2 - tl.pow(updated_mu, 2))
        new_sigma = tl.sqrt(new_sigma2_min_mu2 + tl.pow(new_mu, 2))
        new_s = new_sigma * math.sqrt(3.0) / math.pi

        tl.store(params_ptr + s_pids, new_mu, mask = mask)
        tl.store(params_ptr + s_pids + 1, new_s, mask = mask)