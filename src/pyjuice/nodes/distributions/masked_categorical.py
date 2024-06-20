from __future__ import annotations

import torch
import triton
import triton.language as tl

from typing import Optional, Any

from .distributions import Distribution


class MaskedCategorical(Distribution):
    """
    A class representing Categorical distributions with masks.

    :param num_cats: number of categories
    :type num_cats: int

    :param mask_mode: type of mask; should be in ["range", "full_mask", "rev_range"]
    :type num_cats: str
    """

    def __init__(self, num_cats: int, mask_mode: str):
        super(MaskedCategorical, self).__init__()

        # `range``: only values in [lo, hi) are accepted
        # `full_mask`: specify a full mask of size Size([num_cats]); 
        #              1 for allowed values and 0 for disallowed
        # `rev_range`: values in [lo, hi) are masked out
        assert mask_mode in ["range", "full_mask", "rev_range"]

        self.num_cats = num_cats
        self.mask_mode = mask_mode

        if self.mask_mode == "range":
            self.fw_mar_fn = self.fw_mar_fn_range
            self.bk_flow_fn = self.bk_flow_fn_range
            self.sample_fn = self.sample_fn_range
            self.em_fn = self.em_fn_range
        elif self.mask_mode == "full_mask":
            self.fw_mar_fn = self.fw_mar_fn_full_mask
            self.bk_flow_fn = self.bk_flow_fn_full_mask
            self.sample_fn = self.sample_fn_full_mask
            self.em_fn = self.em_fn_full_mask
        elif self.mask_mode == "rev_range":
            self.fw_mar_fn = self.fw_mar_fn_rev_range
            self.bk_flow_fn = self.bk_flow_fn_rev_range
            self.sample_fn = self.sample_fn_rev_range
            self.em_fn = self.em_fn_rev_range

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return f"MaskedCategorical-{self.mask_mode}"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return [self.num_cats]

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        if self.mask_mode == "range":
            return self.num_cats + 3
        elif self.mask_mode == "full_mask":
            return self.num_cats * 2 + 1
        elif self.mask_mode == "rev_range":
            return self.num_cats + 3
        else:
            raise ValueError(f"Unknown mask mode {self.mask_mode}.")

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        return self.num_cats

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[torch.Tensor] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        assert params is not None, "Parameters should be provided to get meta-parameters."
        params = params.reshape(-1, self.num_parameters())
        assert params.size(0) == num_nodes

        num_nodes = params.size(0)

        if self.mask_mode == "range":
            mask_tensor = params[:,self.num_cats:self.num_cats+2]
        elif self.mask_mode == "full_mask":
            mask_tensor = params[:,self.num_cats:self.num_cats*2]
        elif self.mask_mode == "rev_range":
            mask_tensor = params[:,self.num_cats:self.num_cats+2]

        cat_params = torch.exp(torch.rand([num_nodes, self.num_cats]) * -perturbation)
        
        # Apply mask
        self._apply_mask(cat_params, num_nodes, mask_tensor)
        
        cat_params /= cat_params.sum(dim = 1, keepdim = True)

        params = params.clone()
        params[:,:self.num_cats] = cat_params

        return params.reshape(-1)

    def normalize_parameters(self, params: torch.Tensor, mask_tensor: Optional[torch.Tensor] = None):
        params = params.reshape(-1, self.num_parameters())
        num_nodes = params.size(0)

        cat_params = params[:,:self.num_cats]

        if mask_tensor is None:
            if self.mask_mode == "range":
                mask_tensor = params[:,self.num_cats:self.num_cats+2]
            elif self.mask_mode == "full_mask":
                mask_tensor = params[:,self.num_cats:self.num_cats*2]
            elif self.mask_mode == "rev_range":
                mask_tensor = params[:,self.num_cats:self.num_cats+2]

        # Apply mask
        self._apply_mask(cat_params, num_nodes, mask_tensor)

        cat_params /= cat_params.sum(dim = 1, keepdim = True)
        params[:,:self.num_cats] = cat_params

        return params.reshape(-1)

    def set_meta_parameters(self, num_nodes: int, **kwargs):
        assert "mask" in kwargs, "`MaskedCategorical` requires an input argument `mask`."
        mask_tensor = kwargs["mask"]
        assert mask_tensor.size(0) == num_nodes

        if self.mask_mode == "range":
            assert mask_tensor.size(1) == 2
            num_free_cats = mask_tensor[:,1:2] - mask_tensor[:,0:1]
        elif self.mask_mode == "full_mask":
            assert mask_tensor.size(1) == self.num_cats
            num_free_cats = mask_tensor.sum(dim = 1).unsqueeze(1)
        elif self.mask_mode == "rev_range":
            assert mask_tensor.size(1) == 2
            num_free_cats = self.num_cats - (mask_tensor[:,1:2] - mask_tensor[:,0:1])

        cat_params = torch.zeros([num_nodes, self.num_cats])

        params = torch.cat(
            (cat_params, mask_tensor, num_free_cats), 
            dim = 1
        ).contiguous()

        return params.reshape(-1)

    @property
    def need_meta_parameters(self):
        """
        A flag indicating whether users need to pass in meta-parameters to the 
        constructor of InputNodes. In this case, we need to provide information
        regarding the categorical mask.
        """
        return True

    def get_data_dtype(self):
        """
        Get the data dtype for the distribution.
        """
        return torch.long

    def _apply_mask(self, cat_params: torch.Tensor, num_nodes: int, mask_tensor: torch.Tensor):
        if self.mask_mode == "range":
            mask = torch.arange(self.num_cats).unsqueeze(0).expand(num_nodes, -1)
            cat_params[(mask < mask_tensor[:,:1]) | (mask >= mask_tensor[:,1:])] = 0.0
        elif self.mask_mode == "full_mask":
            cat_params[(mask_tensor < 0.5)] = 0.0
        elif self.mask_mode == "rev_range":
            mask = torch.arange(self.num_cats).unsqueeze(0).expand(num_nodes, -1)
            cat_params[(mask >= mask_tensor[:,:1]) & (mask < mask_tensor[:,1:])] = 0.0
        else:
            raise ValueError(f"Unknown mask mode {self.mask_mode}.")

    @staticmethod
    def fw_mar_fn_range(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        # Range mode
        lo_idx = s_pids + num_cats
        lo = tl.load(params_ptr + lo_idx, mask = mask, other = 0)
        hi_idx = lo_idx + 1
        hi = tl.load(params_ptr + hi_idx, mask = mask, other = 0)

        param_idx = s_pids + data
        probs = tl.load(params_ptr + param_idx, mask = mask, other = 0)
        log_probs = tl.where((data >= lo) & (data < hi), tl.log(probs), -23.0258509299) # log(1e-10)

        return log_probs

    @staticmethod
    def fw_mar_fn_full_mask(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `num_cats` and from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        # Full mask mode
        prob_mask_idx = s_pids + num_cats + data
        prob_mask = tl.load(params_ptr + prob_mask_idx, mask = mask, other = 0)

        param_idx = s_pids + data
        probs = tl.load(params_ptr + param_idx, mask = mask, other = 0)

        log_probs = tl.where(prob_mask > 0.5, tl.log(probs), -23.0258509299) # log(1e-10)

        return log_probs

    @staticmethod
    def fw_mar_fn_rev_range(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        # Rev range mode
        lo_idx = s_pids + num_cats
        lo = tl.load(params_ptr + lo_idx, mask = mask, other = 0)
        hi_idx = lo_idx + 1
        hi = tl.load(params_ptr + hi_idx, mask = mask, other = 0)

        param_idx = s_pids + data
        probs = tl.load(params_ptr + param_idx, mask = mask, other = 0)
        log_probs = tl.where((data < lo) | (data >= hi), tl.log(probs), -23.0258509299) # log(1e-10)

        return log_probs

    @staticmethod
    def bk_flow_fn_range(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                         s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `num_cats` and from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        # Range mode
        lo_idx = s_pids + num_cats
        lo = tl.load(params_ptr + lo_idx, mask = mask, other = 0)
        hi_idx = lo_idx + 1
        hi = tl.load(params_ptr + hi_idx, mask = mask, other = 0)

        mask = mask & (data >= lo) & (data < hi)

        pf_offsets = s_pfids + data
        tl.atomic_add(param_flows_ptr + pf_offsets, flows, mask = mask)

    @staticmethod
    def bk_flow_fn_full_mask(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                             s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        # Full mask mode
        prob_mask_idx = s_pids + num_cats + data
        prob_mask = tl.load(params_ptr + prob_mask_idx, mask = mask, other = 0)

        mask = mask & (prob_mask > 0.5)

        pf_offsets = s_pfids + data
        tl.atomic_add(param_flows_ptr + pf_offsets, flows, mask = mask)

    @staticmethod
    def bk_flow_fn_rev_range(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                             s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Get `num_cats` and from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        # Rev range mode
        lo_idx = s_pids + num_cats
        lo = tl.load(params_ptr + lo_idx, mask = mask, other = 0)
        hi_idx = lo_idx + 1
        hi = tl.load(params_ptr + hi_idx, mask = mask, other = 0)

        mask = mask & ((data < lo) | (data >= hi))

        pf_offsets = s_pfids + data
        tl.atomic_add(param_flows_ptr + pf_offsets, flows, mask = mask)

    @staticmethod
    def sample_fn_range(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        rnd_val = tl.rand(seed, tl.arange(0, BLOCK_SIZE))
        sampled_id = tl.zeros([BLOCK_SIZE], dtype = tl.int64) - 1

        # Range mode
        lo_idx = s_pids + num_cats
        lo = tl.load(params_ptr + lo_idx, mask = mask, other = 0)
        hi_idx = lo_idx + 1
        hi = tl.load(params_ptr + hi_idx, mask = mask, other = 0)

        # Sample by computing cumulative probability
        cum_param = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id >= lo) & (cat_id < hi)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            cum_param += param

            sampled_id = tl.where((cum_param >= rnd_val) & (sampled_id == -1), cat_id, sampled_id)

        sampled_id = tl.where((sampled_id == -1), 0, sampled_id)

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_id, mask = mask)

    @staticmethod
    def sample_fn_full_mask(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        rnd_val = tl.rand(seed, tl.arange(0, BLOCK_SIZE))
        sampled_id = tl.zeros([BLOCK_SIZE], dtype = tl.int64) - 1

        # Sample by computing cumulative probability
        cum_param = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            # Full mask mode
            prob_mask_idx = s_pids + num_cats + cat_id
            prob_mask = tl.load(params_ptr + prob_mask_idx, mask = mask, other = 0)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            cum_param += tl.where(prob_mask > 0.5, param, 0.0)

            sampled_id = tl.where((cum_param >= rnd_val) & (sampled_id == -1), cat_id, sampled_id)

        sampled_id = tl.where((sampled_id == -1), 0, sampled_id)

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_id, mask = mask)

    @staticmethod
    def sample_fn_rev_range(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        rnd_val = tl.rand(seed, tl.arange(0, BLOCK_SIZE))
        sampled_id = tl.zeros([BLOCK_SIZE], dtype = tl.int64) - 1

        # Range mode
        lo_idx = s_pids + num_cats
        lo = tl.load(params_ptr + lo_idx, mask = mask, other = 0)
        hi_idx = lo_idx + 1
        hi = tl.load(params_ptr + hi_idx, mask = mask, other = 0)

        # Sample by computing cumulative probability
        cum_param = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & ((cat_id < lo) | (cat_id >= hi))

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            cum_param += param

            sampled_id = tl.where((cum_param >= rnd_val) & (sampled_id == -1), cat_id, sampled_id)

        sampled_id = tl.where((sampled_id == -1), 0, sampled_id)

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_id, mask = mask)

    @staticmethod
    def em_fn_range(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                    step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        eff_num_cats = tl.load(params_ptr + s_pids + num_cats + 2, mask = mask, other = 1)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            cum_flow += flow

        # Parameter update
        numerate_pseudocount = pseudocount / eff_num_cats
        cum_flow += pseudocount
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / cum_flow
            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)

    @staticmethod
    def em_fn_full_mask(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                        step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        eff_num_cats = tl.load(params_ptr + s_pids + num_cats * 2, mask = mask, other = 1)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            cum_flow += flow

        # Parameter update
        numerate_pseudocount = pseudocount / eff_num_cats
        cum_flow += pseudocount
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / cum_flow
            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)

    @staticmethod
    def em_fn_rev_range(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
                        step_size, pseudocount, BLOCK_SIZE):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        eff_num_cats = tl.load(params_ptr + s_pids + num_cats + 2, mask = mask, other = 1)

        max_num_cats = tl.max(num_cats, axis = 0)

        # Compute cumulative flows
        cum_flow = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            cum_flow += flow

        # Parameter update
        numerate_pseudocount = pseudocount / eff_num_cats
        cum_flow += pseudocount
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)

            new_param = (1.0 - step_size) * param + step_size * (flow + numerate_pseudocount) / cum_flow
            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)

    def _get_constructor(self):
        return MaskedCategorical, {"num_cats": self.num_cats, "mask_mode": self.mask_mode}
