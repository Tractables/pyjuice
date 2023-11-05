from __future__ import annotations

import torch
import triton
import triton.language as tl

from .distributions import Distribution


class Categorical(Distribution):
    def __init__(self, num_cats: int):
        super(Categorical, self).__init__()

        self.num_cats = num_cats

    def get_signature(self):
        return "Categorical"

    def get_metadata(self):
        return [self.num_cats]

    def num_parameters(self):
        return self.num_cats

    def num_param_flows(self):
        return self.num_cats

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        params = torch.exp(torch.rand([num_nodes, self.num_cats]) * -perturbation)
        params /= params.sum(dim = 1, keepdim = True)

        return params.reshape(-1)

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, metadata, mask, num_vars_per_node, BLOCK_SIZE):
        # I am not sure why, but the following code will not work...
        # probs = tl.load(params_ptr + s_pids + data, mask = mask, other = 0)
        # Seems like a bug of triton.
        param_idx = s_pids + data
        probs = tl.load(params_ptr + param_idx, mask = mask, other = 0)
        log_probs = tl.log(probs)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, data, flows, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   metadata, mask, num_vars_per_node, BLOCK_SIZE):
        # I am not sure why, but the following code will not work...
        # tl.atomic_add(param_flows_ptr + s_pfids + data, flows, mask = mask)
        # Seems like a bug of triton.
        pf_offsets = s_pfids + data
        tl.atomic_add(param_flows_ptr + pf_offsets, flows, mask = mask)

    @staticmethod
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            cum_flow += flow

        # Parameter update
        numerate_pseudocount = pseudocount / num_cats
        cum_flow += pseudocount
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / cum_flow
            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)