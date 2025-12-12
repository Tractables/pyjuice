from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Any

from .distributions import Distribution


class Categorical(Distribution):
    """
    A class representing Categorical distributions.

    :param num_cats: number of categories
    :type num_cats: int
    """
    def __init__(self, num_cats: int):
        super(Categorical, self).__init__()

        self.num_cats = num_cats

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "Categorical"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return [self.num_cats]

    def normalize_parameters(self, params: torch.Tensor):
        params = params.reshape(-1, self.num_cats)
        params /= params.sum(dim = 1, keepdim = True)

        return params.reshape(-1)

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        return self.num_cats

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        return self.num_cats

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[Any] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        if params is not None:
            assert isinstance(params, torch.Tensor)
            assert params.numel() == num_nodes * self.num_parameters()
            return params

        params = torch.exp(torch.rand([num_nodes, self.num_cats]) * -perturbation)
        params /= params.sum(dim = 1, keepdim = True)

        return params.reshape(-1)

    def get_data_dtype(self):
        """
        Get the data dtype for the distribution.
        """
        return torch.long

    def get_em_fn(self):
        if self.num_cats <= 256:
            return self.small_ncats_em_fn
        else:
            self.em_block_size = 8
            return self.large_ncats_em_fn

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # I am not sure why, but the following code will not work...
        # probs = tl.load(params_ptr + s_pids + data, mask = mask, other = 0)
        # Seems like a bug of triton.
        param_idx = s_pids + data
        probs = tl.load(params_ptr + param_idx, mask = mask, other = 0)
        log_probs = tl.log(probs)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # I am not sure why, but the following code will not work...
        # tl.atomic_add(param_flows_ptr + s_pfids + data, flows, mask = mask)
        # Seems like a bug of triton.
        pf_offsets = s_pfids + data
        tl.atomic_add(param_flows_ptr + pf_offsets, flows, mask = mask)

    @staticmethod
    def bk_flow_mask_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                        s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE, TILE_SIZE_K):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        if TILE_SIZE_K > 1:
            num_iters = tlmath.ceil(max_num_cats / TILE_SIZE_K).to(tl.int64)

            cat_ids = tl.arange(0, TILE_SIZE_K)

            for i in range(num_iters):
                cat_mask = mask[:,None] & missing_mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

                p_offsets = s_pids[:,None] + cat_ids[None,:]
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                pf_offsets = s_pfids[:,None] + cat_ids[None,:]
                tl.atomic_add(param_flows_ptr + pf_offsets, flows[:,None] * param, mask = cat_mask)

                cat_ids += TILE_SIZE_K
        else:
            for cat_id in range(max_num_cats):
                cat_mask = mask & missing_mask & (cat_id < num_cats)

                p_offsets = s_pids + cat_id
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                pf_offsets = s_pfids + cat_id
                tl.atomic_add(param_flows_ptr + pf_offsets, flows * param, mask = cat_mask)

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        rnd_val = tl.rand(seed, tl.arange(0, BLOCK_SIZE) + block_start)
        sampled_id = tl.zeros([BLOCK_SIZE], dtype = tl.int64) - 1

        # Sample by computing cumulative probability
        cum_param = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            cum_param += param

            sampled_id = tl.where((cum_param >= rnd_val) & (sampled_id == -1), cat_id, sampled_id)

        sampled_id = tl.where((sampled_id == -1), 0, sampled_id)

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_id, mask = mask)

    @staticmethod
    def small_ncats_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                          step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        numerate_pseudocount = pseudocount / num_cats
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            if keep_zero_params:
                param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
                cum_flow += tl.where(param < 1e-12, 0.0, flow + numerate_pseudocount)
            else:
                cum_flow += flow

        # Parameter update
        cum_flow += pseudocount
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            if keep_zero_params:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / (cum_flow - pseudocount)
                new_param = tl.where(param < 1e-12, 0.0, new_param)
            else:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / cum_flow

            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)

    @staticmethod
    def large_ncats_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        numerate_pseudocount = pseudocount / num_cats
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        cat_ids = tl.arange(0, 128)
        for cat_sid in range(0, max_num_cats, 128):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            flow = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            if keep_zero_params:
                param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
                cum_flow += tl.sum(tl.where(param < 1e-12, 0.0, flow + numerate_pseudocount[:,None]))
            else:
                cum_flow += tl.sum(flow, axis = 1)

            cat_ids += 128

        # Parameter update
        cum_flow += pseudocount
        cat_ids = tl.arange(0, 128)
        for cat_sid in range(0, max_num_cats, 128):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            if keep_zero_params:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount[:,None]) / (cum_flow[:,None] - pseudocount)
                new_param = tl.where(param < 1e-12, 0.0, new_param)
            else:
                new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount[:,None]) / cum_flow[:,None]
            tl.store(params_ptr + s_pids[:,None] + cat_ids[None,:], new_param, mask = cat_mask)

            cat_ids += 128

    @staticmethod
    def partition_fn(local_offsets, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, BLOCK_SIZE, TILE_SIZE_K):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        num_iters = tlmath.ceil(max_num_cats / TILE_SIZE_K).to(tl.int64)

        cat_ids = tl.arange(0, TILE_SIZE_K)
        mars = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

        for i in range(num_iters):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            p_offsets = s_pids[:,None] + cat_ids[None,:]
            param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

            mars += tl.sum(param, axis = 1)

            cat_ids += TILE_SIZE_K

        mars = tl.log(mars)
        return mars

    def _get_constructor(self):
        return Categorical, {"num_cats": self.num_cats}

    def _need_2nd_kernel_dim(self):
        return True
