from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional, Any

from .distributions import Distribution


class DiscreteLogistic(Distribution):
    """
    A class representing Discrete Logistic distributions.

    :param val_range: range of the values represented by the distribution
    :type val_range: Tuple[float,float]

    :param num_cats: number of categories
    :type num_cats: int

    :param min_std: minimum standard deviation
    :type min_std: float
    """

    def __init__(self, val_range: Tuple[float,float], num_cats: int, min_std: float = 0.01):
        super(DiscreteLogistic, self).__init__()

        self.val_range = val_range
        self.num_cats = num_cats
        self.min_std = min_std

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "DiscreteLogistic"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return [self.val_range[0], self.val_range[1], self.num_cats, self.min_std]

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
            return params

        mu = kwargs["mu"] if "mu" in kwargs else (self.val_range[0] + self.val_range[1]) / 2.0
        std = kwargs["std"] if "std" in kwargs else (self.val_range[1] - self.val_range[0]) / 4.0 # 2 sigma range

        mus = torch.normal(mean = torch.ones([num_nodes]) * mu, std = std)
        ss = torch.exp(torch.rand([num_nodes]) * -0.04 * perturbation) * std * math.sqrt(3) / math.pi

        return torch.stack((mus, ss), dim = 1).reshape(-1).contiguous()

    def get_data_dtype(self):
        """
        Get the data dtype for the distribution.
        """
        return torch.float32

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `val_range` and `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
        range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        s = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)

        # 0: (-inf, range_low + interval]; 
        # 1: (range_low + interval, range_low + 2 * interval]; 
        # ...; 
        # num_cats - 1: [range_high - interval, inf)
        interval = (range_high - range_low) / num_cats
        vlow = data * interval + range_low
        vhigh = vlow + interval

        cdfhigh = tl.where(data == num_cats - 1, 1.0, 1.0 / (1.0 + tl.exp((mu - vhigh) / s)))
        cdflow = tl.where(data == 0, 0.0, 1.0 / (1.0 + tl.exp((mu - vlow) / s)))

        log_probs = tl.maximum(tl.log(cdfhigh - cdflow), -1000.0)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `val_range` and `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        range_low = tl.load(metadata_ptr + s_mids, mask = mask, other = 0)
        range_high = tl.load(metadata_ptr + s_mids + 1, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids + 2, mask = mask, other = 0).to(tl.int64)

        interval = (range_high - range_low) / num_cats
        vmid = data * interval + range_low + 0.5 * interval # (vlow + vhigh) / 2

        stat1 = vmid * flows
        stat2 = vmid * vmid * flows
        stat3 = flows

        tl.atomic_add(param_flows_ptr + s_pfids, stat1, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 1, stat2, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 2, stat3, mask = mask)

    @staticmethod
    def bk_flow_mask_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                        s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE, TILE_SIZE_K):
        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        s = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0)

        stat1 = mu * flows
        stat2 = (s * s * 3.2898681337 + mu * mu) * flows # the constant is pi^2 / 3
        stat3 = flows

        tl.atomic_add(param_flows_ptr + s_pfids, stat1, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 1, stat2, mask = mask)
        tl.atomic_add(param_flows_ptr + s_pfids + 2, stat3, mask = mask)

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
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
        interval = (range_high - range_low) / num_cats
        bin_val = (sample - range_low) / interval
        sampled_id = tl.where(bin_val < 0, 0, tl.where(bin_val > num_cats - 1, num_cats - 1, tl.floor(bin_val).to(tl.int64)))

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_id, mask = mask)

    @staticmethod
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):
        # Get `min_std`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        min_std = tl.load(metadata_ptr + s_mids + 3, mask = mask, other = 0)

        mu = tl.load(params_ptr + s_pids, mask = mask, other = 0)
        std = tl.load(params_ptr + s_pids + 1, mask = mask, other = 0) * 1.8137993642342178 # The constant is `math.pi / math.sqrt(3.0)`
        ori_theta1 = mu
        ori_theta2 = std * std + mu * mu

        stat1 = tl.load(param_flows_ptr + s_pfids, mask = mask)
        stat2 = tl.load(param_flows_ptr + s_pfids + 1, mask = mask)
        stat3 = tl.load(param_flows_ptr + s_pfids + 2, mask = mask)

        new_theta1 = stat1 / (stat3 + 1e-10)
        new_theta2 = stat2 / (stat3 + 1e-10)

        # Get the updated natural parameters
        updated_theta1 = (1.0 - step_size) * ori_theta1 + step_size * new_theta1
        updated_theta2 = (1.0 - step_size) * ori_theta2 + step_size * new_theta2

        # Reconstruct `mu` and `std` from the expectation parameters (moment matching)
        updated_mu = updated_theta1
        updated_std2 = updated_theta2 - updated_mu * updated_mu
        updated_std = tl.where(updated_std2 < min_std * min_std, min_std, tl.sqrt(updated_std2))
        updated_s = updated_std * 0.5513288954217921 # The constant is `math.sqrt(3.0) / math.pi`

        tl.store(params_ptr + s_pids, updated_mu, mask = mask)
        tl.store(params_ptr + s_pids + 1, updated_s, mask = mask)

    def _get_constructor(self):
        return DiscreteLogistic, {"val_range": self.val_range, "num_cats": self.num_cats, "min_std": self.min_std}
