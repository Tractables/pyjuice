from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional, Any

from .distributions import Distribution
from pyjuice.utils.kernel_launcher import triton_jit

# In the latest triton, math functions were shuffled around into different modules:
# https://github.com/openai/triton/pull/3172
if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math


def _condition_apply_ll_kernel(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs


def _prep_args_apply_ll_kernel(layer, kwargs):
    target_kwargs = dict()

    if kwargs["extern_product_categorical_mode"] == "normalized_ll":
        compute_unnorm_logp = True
        compute_logz = True
    elif kwargs["extern_product_categorical_mode"] == "unnormalized_ll":
        compute_unnorm_logp = True
        compute_logz = False
    elif kwargs["extern_product_categorical_mode"] == "normalizing_constant":
        compute_unnorm_logp = False
        compute_logz = True
    else:
        raise ValueError("Unexpected `extern_product_categorical_mode`. Should be 'normalized_ll', 'unnormalized_ll', or 'normalizing_constant'.")

    kwargs["compute_unnorm_logp"] = compute_unnorm_logp
    kwargs["compute_logz"] = compute_logz

    assert "extern_product_categorical_logps" in kwargs
    extern_product_categorical_logps = kwargs["extern_product_categorical_logps"]
    assert extern_product_categorical_logps.dim() == 3

    target_kwargs["extern_product_categorical_logps_ptr"] = extern_product_categorical_logps

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = extern_product_categorical_logps.size(1)

    for ns in layer.nodes:
        assert ns.dist.num_cats <= extern_product_categorical_logps.size(2)

    target_kwargs["max_num_cats"] = extern_product_categorical_logps.size(2)

    # prepare BLOCK_SIZE and TILE_SIZE_K
    target_kwargs["TILE_SIZE_K"] = min(128, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["BLOCK_SIZE"] = 1024 // target_kwargs["TILE_SIZE_K"]

    return target_kwargs


def _condition_apply_ll_bp_kernel(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs


def _prep_args_apply_ll_bp_kernel(layer, kwargs):
    target_kwargs = dict()

    if kwargs["extern_product_categorical_mode"] == "normalized_ll":
        compute_unnorm_logp = True
        compute_logz = True
    elif kwargs["extern_product_categorical_mode"] == "unnormalized_ll":
        compute_unnorm_logp = True
        compute_logz = False
    elif kwargs["extern_product_categorical_mode"] == "normalizing_constant":
        compute_unnorm_logp = False
        compute_logz = True
    else:
        raise ValueError("Unexpected `extern_product_categorical_mode`. Should be 'normalized_ll', 'unnormalized_ll', or 'normalizing_constant'.")

    kwargs["compute_unnorm_logp"] = compute_unnorm_logp
    kwargs["compute_logz"] = compute_logz

    assert "extern_product_categorical_logps" in kwargs
    extern_product_categorical_logps = kwargs["extern_product_categorical_logps"]
    assert extern_product_categorical_logps.dim() == 3

    target_kwargs["extern_product_categorical_logps_ptr"] = extern_product_categorical_logps

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = extern_product_categorical_logps.size(1)

    target_kwargs["max_num_cats"] = extern_product_categorical_logps.size(2)

    if "extern_product_categorical_logps_grad" in kwargs:
        target_kwargs["extern_product_categorical_logps_grad_ptr"] = kwargs["extern_product_categorical_logps_grad"]
        target_kwargs["compute_extern_grad"] = True
    else:
        target_kwargs["extern_product_categorical_logps_grad_ptr"] = None
        target_kwargs["compute_extern_grad"] = False

    # prepare BLOCK_SIZE and TILE_SIZE_K
    target_kwargs["TILE_SIZE_K"] = min(128, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["BLOCK_SIZE"] = 1024 // target_kwargs["TILE_SIZE_K"]

    return target_kwargs


class ExternProductCategorical(Distribution):
    """
    A class representing a product distribution of two Categorical distributions: Pr_{A} * Pr_{B}.
    Pr_{A} is the the distribution parameterized by the internal parameters, similar to `juice.distributions.Categorical`.
    Pr_{B} is an external Categorical distribution provided on the fly (at training/inference time).

    :param num_cats: number of categories
    :type num_cats: int
    """
    def __init__(self, num_cats: int):
        super(ExternProductCategorical, self).__init__()

        self.num_cats = num_cats

        self.post_fw_fns = [
            (self.ll_kernel, _condition_apply_ll_kernel, _prep_args_apply_ll_kernel)
        ]

        self.post_bp_fns = [
            (self.ll_bp_kernel, _condition_apply_ll_bp_kernel, _prep_args_apply_ll_bp_kernel)
        ]

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "ExternProductCategorical"

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
    @triton_jit
    def ll_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, 
                  fw_local_ids_ptr, partial_eval: tl.constexpr, layer_num_nodes: tl.constexpr, batch_size: tl.constexpr, 
                  num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                  TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, compute_unnorm_logp: tl.constexpr, compute_logz: tl.constexpr,
                  extern_product_categorical_logps_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_nodes * batch_size

        # Raw batch and (local) node id
        batch_offsets = (offsets % batch_size)
        local_offsets = (offsets // batch_size)

        if partial_eval > 0:
            local_offsets = tl.load(fw_local_ids_ptr + local_offsets, mask = mask, other = 0)

        # Get all variable ids
        vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)
        lvids = tl.load(var_idmapping_ptr + vids, mask = mask, other = 0)

        # Get all latent offsets
        nids = tl.load(nids_ptr + local_offsets, mask = mask, other = 0)

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64) # [BLOCK_SIZE]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)

        # Ptrs pointing to internal parameters
        inpars_ptr = s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE, TILE_SIZE_K]

        # Ptrs pointing to external parameters
        expars_ptr = extern_product_categorical_logps_ptr + \
            batch_offsets[:,None] * (ext_num_vars * max_num_cats) + \
            lvids[:,None] * max_num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:]  # [BLOCK_SIZE, TILE_SIZE_K]

        # Compute logZ
        if compute_logz:
            logZ = tl.zeros([BLOCK_SIZE], dtype = tl.float32) - float("inf")
            for i in range(K_NUM_TILES):
                cat_mask = mask[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats[:,None])

                inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)
                expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)

                lpar = (inpar * expar).sum(axis = 1).log()

                # Compute log-add-exp(logZ, lpar)
                maxval = tl.maximum(logZ, lpar)
                minval = tl.minimum(logZ, lpar)
                diff = minval - maxval

                logZ = tl.where(logZ == -float("inf"),
                    lpar,
                    maxval + tlmath.log1p(tl.exp(diff))
                )
        else:
            logZ = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

        # Compute unnormalized log-probabilities
        if compute_unnorm_logp:
            data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = mask, other = 0)
            log_in_p = tl.load(params_ptr + s_pids + data, mask = mask, other = 0.0).log()
            
            ex_p_ptr = extern_product_categorical_logps_ptr + \
                batch_offsets * (ext_num_vars * max_num_cats) + \
                lvids * max_num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask, other = 0.0).log()
        else:
            log_in_p = tl.zeros([BLOCK_SIZE], dtype = tl.float32)
            log_ex_p = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

        # Compute the final output
        if compute_logz and not compute_unnorm_logp:
            log_p = logZ
        else:
            log_p = log_in_p + log_ex_p - logZ

        # Store the logprob
        node_offsets = local_offsets + node_offset
        tl.store(node_mars_ptr + node_offsets * batch_size + batch_offsets, log_p, mask = mask)

    @staticmethod
    @triton_jit
    def ll_bp_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                     metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                     batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                     BLOCK_SIZE: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, compute_unnorm_logp: tl.constexpr, compute_logz: tl.constexpr,
                     extern_product_categorical_logps_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr,
                     extern_product_categorical_logps_grad_ptr, compute_extern_grad: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_nodes * batch_size

        # Raw batch and (local) node id
        batch_offsets = (offsets % batch_size)
        local_offsets = (offsets // batch_size)

        if partial_eval > 0:
            local_offsets = tl.load(bk_local_ids_ptr + local_offsets, mask = mask, other = 0)

        # Get all variable ids
        vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)
        lvids = tl.load(var_idmapping_ptr + vids, mask = mask, other = 0)

        # Get all latent offsets
        nids = tl.load(nids_ptr + local_offsets, mask = mask, other = 0)

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64) # [BLOCK_SIZE]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)

        # Get data
        data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = mask, other = 0)

        if compute_extern_grad:
            # Compute unnormalized log-probabilities
            log_in_p = tl.load(params_ptr + s_pids + data, mask = mask, other = 0.0).log()
            
            ex_p_ptr = extern_product_categorical_logps_ptr + \
                batch_offsets * (ext_num_vars * max_num_cats) + \
                lvids * max_num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask, other = 0.0).log()

            # Load the forward log-probabilities
            node_offsets = local_offsets + node_offset
            logp = tl.load(node_mars_ptr + node_offsets * batch_size + batch_offsets, mask = mask)

            # Get logZ
            logZ = log_in_p + log_ex_p - logp

        # Load flows
        flows = tl.load(node_flows_ptr + node_offsets * batch_size + batch_offsets, mask = mask, other = 0)

        if compute_unnorm_logp:
            pf_offsets = s_pfids + data
            tl.atomic_add(param_flows_ptr + pf_offsets, flows, mask = mask)

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        log_probs = tl.where(data == 0, 0.0, 0.0)

        return log_probs

    @staticmethod
    def bk_flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                   s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        pass

    @staticmethod
    def bk_flow_mask_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                        s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE, TILE_SIZE_K):
        pass

    @staticmethod
    def sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed):
        pass

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

    def _get_constructor(self):
        return ExternProductCategorical, {"num_cats": self.num_cats}

    def __reduce__(self):
        return (self.__class__, (self.num_cats,))
