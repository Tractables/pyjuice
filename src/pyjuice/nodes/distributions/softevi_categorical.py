from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional, Any

from .distributions import Distribution
from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import max_power_of_2_factor

# In the latest triton, math functions were shuffled around into different modules:
# https://github.com/openai/triton/pull/3172
if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math


def _condition_apply_fw_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs


def _prep_args_apply_fw_kernel(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]

    categorical_evidence_logp = kwargs["categorical_evidence_logp"]
    assert categorical_evidence_logp.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."

    ext_num_vars = categorical_evidence_logp.size(1)
    target_kwargs["ext_num_vars"] = ext_num_vars

    num_cats = categorical_evidence_logp.size(2)
    for ns in layer.nodes:
        assert num_cats <= ns.dist.num_cats
    target_kwargs["num_cats"] = num_cats

    target_kwargs["categorical_evidence_logp_ptr"] = categorical_evidence_logp
    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    # (Optional) soft_evidence_cat_ids
    if "soft_evidence_cat_ids" in kwargs:
        soft_evidence_cat_ids = kwargs["soft_evidence_cat_ids"]
        assert categorical_evidence_logp.size() == soft_evidence_cat_ids.size()

        target_kwargs["soft_evidence_cat_ids_ptr"] = soft_evidence_cat_ids
        target_kwargs["has_ext_ids"] = True
    else:
        target_kwargs["soft_evidence_cat_ids_ptr"] = None
        target_kwargs["has_ext_ids"] = False

    # Prepare block/grid size
    assert not layer.provided("fw_local_ids")
    n_block_size = max_power_of_2_factor(layer.n_block_size)

    # prepare BLOCK_SIZE and TILE_SIZE_K
    if target_kwargs["has_ext_ids"]:
        TILE_SIZE_K = min(16, triton.next_power_of_2(num_cats))
        K_NUM_TILES = triton.cdiv(num_cats, TILE_SIZE_K)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        BLOCK_SIZE_B = min(128, 1024 // TILE_SIZE_K, BATCH_SIZE_NP2)
        BLOCK_SIZE_N = min(n_block_size, max(4096 // BLOCK_SIZE_B // TILE_SIZE_K, 1))
    else:
        TILE_SIZE_K = min(64, triton.next_power_of_2(num_cats))
        K_NUM_TILES = triton.cdiv(num_cats, TILE_SIZE_K)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        BLOCK_SIZE_B = min(128, 2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        BLOCK_SIZE_N = max(min(n_block_size, 2048 // TILE_SIZE_K, 2048 // BLOCK_SIZE_B), 1)

    use_tensor_core = (TILE_SIZE_K >= 16) and (BLOCK_SIZE_B >= 16) and (BLOCK_SIZE_N >= 16) and not target_kwargs["has_ext_ids"]

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    target_kwargs["TILE_SIZE_K"] = TILE_SIZE_K
    target_kwargs["K_NUM_TILES"] = K_NUM_TILES
    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N
    target_kwargs["use_tensor_core"] = use_tensor_core

    return target_kwargs, grid


def _condition_apply_bk_params_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs


def _prep_args_apply_bk_params_kernel(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]

    categorical_evidence_logp = kwargs["categorical_evidence_logp"]
    assert categorical_evidence_logp.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."

    ext_num_vars = categorical_evidence_logp.size(1)
    target_kwargs["ext_num_vars"] = ext_num_vars

    num_cats = categorical_evidence_logp.size(2)
    target_kwargs["num_cats"] = num_cats

    target_kwargs["categorical_evidence_logp_ptr"] = categorical_evidence_logp
    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    # Prepare block/grid size
    assert not layer.provided("fw_local_ids")
    n_block_size = max_power_of_2_factor(layer.n_block_size)

    # prepare BLOCK_SIZE
    BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
    BLOCK_SIZE_B = min(1024, BATCH_SIZE_NP2)
    BLOCK_SIZE_N = min(n_block_size, 2048 // BLOCK_SIZE_B)

    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    return target_kwargs, grid


def _condition_apply_bk_softevi_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and "categorical_evidence_logp_grad" in kwargs


def _prep_args_apply_bk_softevi_kernel(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]

    categorical_evidence_logp = kwargs["categorical_evidence_logp"]
    assert categorical_evidence_logp.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."

    ext_num_vars = categorical_evidence_logp.size(1)
    target_kwargs["ext_num_vars"] = ext_num_vars

    num_cats = categorical_evidence_logp.size(2)
    target_kwargs["num_cats"] = num_cats

    categorical_evidence_logp_grad = kwargs["categorical_evidence_logp_grad"]
    assert categorical_evidence_logp_grad.size(0) == batch_size
    assert categorical_evidence_logp_grad.size(1) == ext_num_vars
    assert categorical_evidence_logp_grad.size(2) == num_cats

    target_kwargs["categorical_evidence_logp_ptr"] = categorical_evidence_logp
    target_kwargs["categorical_evidence_logp_grad_ptr"] = categorical_evidence_logp_grad
    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    # (Optional) soft_evidence_cat_ids
    if "soft_evidence_cat_ids" in kwargs:
        soft_evidence_cat_ids = kwargs["soft_evidence_cat_ids"]
        assert categorical_evidence_logp.size() == soft_evidence_cat_ids.size()

        target_kwargs["soft_evidence_cat_ids_ptr"] = soft_evidence_cat_ids
        target_kwargs["has_ext_ids"] = True
    else:
        target_kwargs["soft_evidence_cat_ids_ptr"] = None
        target_kwargs["has_ext_ids"] = False

    # Prepare block/grid size
    assert not layer.provided("fw_local_ids")
    n_block_size = max_power_of_2_factor(layer.n_block_size)

    # prepare BLOCK_SIZE and TILE_SIZE_K
    if target_kwargs["has_ext_ids"]:
        TILE_SIZE_K = min(16, triton.next_power_of_2(num_cats))
        K_NUM_TILES = triton.cdiv(num_cats, TILE_SIZE_K)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        BLOCK_SIZE_B = min(128, 1024 // TILE_SIZE_K, BATCH_SIZE_NP2)
        BLOCK_SIZE_N = min(n_block_size, max(4096 // BLOCK_SIZE_B // TILE_SIZE_K, 1))
    else:
        TILE_SIZE_K = min(64, triton.next_power_of_2(num_cats))
        K_NUM_TILES = triton.cdiv(num_cats, TILE_SIZE_K)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        BLOCK_SIZE_B = min(128, 2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        BLOCK_SIZE_N = min(n_block_size, 2048 // TILE_SIZE_K, 2048 // BLOCK_SIZE_B)

    use_tensor_core = (TILE_SIZE_K >= 16) and (BLOCK_SIZE_B >= 16) and (BLOCK_SIZE_N >= 16) and not target_kwargs["has_ext_ids"]

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    target_kwargs["TILE_SIZE_K"] = TILE_SIZE_K
    target_kwargs["K_NUM_TILES"] = K_NUM_TILES
    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N
    target_kwargs["use_tensor_core"] = use_tensor_core

    return target_kwargs, grid


def _condition_sample_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs


def _prep_args_sample_kernel(layer, kwargs):
    target_kwargs = dict()

    categorical_evidence_logp = kwargs["categorical_evidence_logp"]

    assert kwargs["batch_size"] == categorical_evidence_logp.size(0), "Batch size doesn't match."

    target_kwargs["categorical_evidence_logp_ptr"] = categorical_evidence_logp

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = categorical_evidence_logp.size(1)
    target_kwargs["max_num_cats"] = categorical_evidence_logp.size(2)

    num_activ_nodes = kwargs["num_activ_nodes"]

    target_kwargs["TILE_SIZE_K"] = min(64, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    target_kwargs["BLOCK_S"] = min(64, 1024 // target_kwargs["TILE_SIZE_K"], triton.next_power_of_2(num_activ_nodes))

    grid = (triton.cdiv(num_activ_nodes, target_kwargs["BLOCK_S"]),)

    return target_kwargs, grid


class SoftEvidenceCategorical(Distribution):
    """
    A class representing a Categorical distribution that allows external soft evidence.

    :param num_cats: number of categories
    :type num_cats: int
    """
    def __init__(self, num_cats: int):
        super(SoftEvidenceCategorical, self).__init__()

        self.num_cats = num_cats

        self.post_fw_fns = [
            (self.fw_kernel, _condition_apply_fw_kernel, _prep_args_apply_fw_kernel)
        ]

        self.post_bp_fns = [
            (self.bk_params_kernel, _condition_apply_bk_params_kernel, _prep_args_apply_bk_params_kernel),
            (self.bk_softevi_kernel, _condition_apply_bk_softevi_kernel, _prep_args_apply_bk_softevi_kernel)
        ]

        self.sampling_fns = [
            (self.sample_kernel, _condition_sample_kernel, _prep_args_sample_kernel)
        ]

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "ExternSoftEvidenceCategorical"

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
    def fw_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, fw_local_ids_ptr, layer_num_nodes,
                  batch_size, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset, partial_eval: tl.constexpr,
                  TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, use_tensor_core: tl.constexpr,
                  categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr, var_idmapping_ptr, num_cats: tl.constexpr, ext_num_vars: tl.constexpr, has_ext_ids: tl.constexpr):

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vid = tl.load(vids_ptr + offset_n) # Global variable ID
        lvid = tl.load(var_idmapping_ptr + vid) # Variable ID for "this type of inputs"

        # Get latent offset of all nodes
        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Ptrs pointing to external parameters
        expars_ptr = categorical_evidence_logp_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + \
            lvid * num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        # Compute logZ
        logZ = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_N], dtype = tl.float32) - float("inf")

        if has_ext_ids:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + s_pids # [BLOCK_SIZE_N]

            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Load the internal parameters
                in_catpars_ptr = inpars_ptr[None,None,:] + catids[:,:,None] # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]
                inpars = tl.load(in_catpars_ptr, mask = (mask_b[:,None,None] & mask_c[None,:,None] & mask_n[None,None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                params = expars[:,:,None] + tl.log(inpars)
                params_max = tl.max(params, axis = 1)
                cum_params = tl.log(tl.sum(tl.exp(params - params_max[:,None,:]), axis = 1)) + params_max # [BLOCK_SIZE_B, BLOCK_SIZE_N]

                # Compute logaddexp(logZ, cum_params)
                maxval = tl.maximum(logZ, cum_params)
                minval = tl.minimum(logZ, cum_params)
                diff = minval - maxval

                logZ = tl.where(logZ == -float("inf"),
                    cum_params,
                    maxval + tlmath.log1p(tl.exp(diff))
                )

        else:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + \
                tl.arange(0, TILE_SIZE_K)[:,None] + \
                s_pids[None,:] # [TILE_SIZE_K, BLOCK_SIZE_N]

            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the internal parameters
                inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[:,None] & mask_n[None,:]), other = 0.0) # [TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                expars_max = tl.max(expars, axis = 1)[:,None]
                expars_sub = tl.exp(expars - expars_max)

                if use_tensor_core:
                    params = tl.dot(expars_sub, inpars).log() + expars_max
                else:
                    params = tl.sum(expars_sub[:,:,None] * inpars[None,:,:], axis = 1).log() + expars_max

                # Compute logaddexp(logZ, params)
                maxval = tl.maximum(logZ, params)
                minval = tl.minimum(logZ, params)
                diff = minval - maxval

                logZ = tl.where(logZ == -float("inf"),
                    params,
                    maxval + tlmath.log1p(tl.exp(diff))
                )

        # Compute unnormalized logprobs
        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0) # [BLOCK_SIZE_B]

        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0).log() # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        if has_ext_ids:
            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            # Ptrs pointing to external parameters
            expar_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats # [BLOCK_SIZE_B]

            log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Find matching ids
                is_match = (catids == data[:,None]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
                match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1) # [BLOCK_SIZE_B]
                has_match = (tl.sum(is_match, axis = 1) > 0) # [BLOCK_SIZE_B]

                # Load parameters if found
                expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0) # [BLOCK_SIZE_B]
                log_ex_p = tl.where(has_match, expar, log_ex_p)

        else:
            ex_p_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask_b, other = 0.0) # [BLOCK_SIZE_B]

        # Final output logprob
        log_p = log_in_p + log_ex_p[:,None] - logZ

        # Store results
        node_offsets = offsets_n + node_offset
        tl.store(node_mars_ptr + node_offsets[None,:] * batch_size + offsets_b[:,None], log_p, mask = (mask_b[:,None] & mask_n[None,:]))

    @staticmethod
    @triton_jit
    def bk_params_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr, nids_ptr,
                         bk_local_ids_ptr, layer_num_nodes, batch_size, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                         node_offset, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                         categorical_evidence_logp_ptr, var_idmapping_ptr, num_cats: tl.constexpr, ext_num_vars: tl.constexpr):

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes
        mask_nb = mask_n[:,None] & mask_b[None,:]

        offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vids = tl.load(vids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get data
        data = tl.load(data_ptr + vids[:,None] * batch_size + offsets_b[None,:], mask = mask_nb, other = 0) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        # Load node flows
        node_offsets = offsets_n + node_offset
        nflows = tl.load(node_flows_ptr + node_offsets[:,None] * batch_size + offsets_b[None,:], mask = mask_nb, other = 0.0) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        if logspace_flows:
            nflows = nflows.exp()

        # Cumulate parameter flows
        tl.atomic_add(param_flows_ptr + s_pfids[:,None] + data, nflows, mask = mask_nb)

    @staticmethod
    @triton_jit
    def bk_softevi_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr, nids_ptr,
                          bk_local_ids_ptr, layer_num_nodes, batch_size, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                          node_offset, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                          TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, use_tensor_core: tl.constexpr,
                          categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr, categorical_evidence_logp_grad_ptr, var_idmapping_ptr, 
                          num_cats: tl.constexpr, ext_num_vars: tl.constexpr, has_ext_ids: tl.constexpr):
        
        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vid = tl.load(vids_ptr + offset_n) # Global variable ID
        lvid = tl.load(var_idmapping_ptr + vid) # Variable ID for "this type of inputs"

        # Get latent offset of all nodes
        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Ptrs pointing to external parameters
        expars_ptr = categorical_evidence_logp_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + \
            lvid * num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        # Load nmars
        nmars_ptr = node_mars_ptr + \
            (offsets_n + node_offset)[None,:] * batch_size + \
            offsets_b[:,None] # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        nmars = tl.load(nmars_ptr, mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Load nflows
        nflows_ptr = node_flows_ptr + \
            (offsets_n + node_offset)[None,:] * batch_size + \
            offsets_b[:,None] # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        nflows = tl.load(nflows_ptr, mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        if logspace_flows:
            nflows = nflows.exp()

        # Compute unnormalized logprobs & backprop the "nominator" parts of the gradients
        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0) # [BLOCK_SIZE_B]

        if has_ext_ids:
            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            # Ptrs pointing to external parameters
            expar_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats # [BLOCK_SIZE_B]

            # Ptrs pointing to external parameter gradients
            expar_grad_ptr = categorical_evidence_logp_grad_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats # [BLOCK_SIZE_B]

            log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Find matching ids
                is_match = (catids == data[:,None]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
                match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1) # [BLOCK_SIZE_B]
                has_match = (tl.sum(is_match, axis = 1) > 0) # [BLOCK_SIZE_B]

                # Load parameters if found
                expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0) # [BLOCK_SIZE_B]
                log_ex_p = tl.where(has_match, expar, log_ex_p)

                # Accumulate gradients
                tl.atomic_add(expar_grad_ptr + i * TILE_SIZE_K + match_ids, tl.sum(nflows, axis = 1), mask = (mask_b & has_match)) # [BLOCK_SIZE_B]

        else:
            ex_p_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask_b, other = 0.0) # [BLOCK_SIZE_B]

            # Accumulate gradients
            unnorm_ll_grad_ptr = categorical_evidence_logp_grad_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                data
            tl.atomic_add(unnorm_ll_grad_ptr, tl.sum(nflows, axis = 1), mask = mask_b)

        # Retrieve logZ
        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0).log() # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        logZ = log_in_p + log_ex_p[:,None] - nmars # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Ptrs pointing to external parameter gradients
        expars_grad_ptr = categorical_evidence_logp_grad_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + \
            lvid * num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        # Backprop the "denominator" parts of the gradients
        if has_ext_ids:
            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + s_pids # [BLOCK_SIZE_N]

            # Ptrs pointing to external parameter indices
            catids_ptr = soft_evidence_cat_ids_ptr + \
                offsets_b[:,None] * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Load the internal parameters
                in_catpars_ptr = inpars_ptr[None,None,:] + catids[:,:,None] # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]
                inpars = tl.load(in_catpars_ptr, mask = (mask_b[:,None,None] & mask_c[None,:,None] & mask_n[None,None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                ve_grads = (nflows.log() - logZ)[:,None,:] + inpars.log() # [BLOCK_SIZE_B, TILE_SIZE_K, BLOCK_SIZE_N]
                ve_grads_max = tl.max(ve_grads, axis = 2)
                ve_grads_sub = tl.exp(ve_grads - ve_grads_max[:,:,None])
                cum_ve_grads = tl.sum(ve_grads_sub, axis = 2).log() + ve_grads_max # [BLOCK_SIZE_B, TILE_SIZE_K]

                expars_grad = cum_ve_grads + expars

                tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, -tl.exp(expars_grad), mask = (mask_b[:,None] & mask_c[None,:]))

        else:
            # Ptrs pointing to internal parameters
            if use_tensor_core:
                inpars_ptr = params_ptr + \
                    tl.arange(0, TILE_SIZE_K)[None,:] + \
                    s_pids[:,None] # [BLOCK_SIZE_N, TILE_SIZE_K]
            else:
                inpars_ptr = params_ptr + \
                    tl.arange(0, TILE_SIZE_K)[:,None] + \
                    s_pids[None,:] # [TILE_SIZE_K, BLOCK_SIZE_N]

            nflow_sub_logz = nflows.log() - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]
            nflow_sub_logz_max = tl.max(nflow_sub_logz, axis = 1)[:,None]
            nflow_sub_logz_sub = tl.exp(nflow_sub_logz - nflow_sub_logz_max)

            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the internal parameters
                if use_tensor_core:
                    inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[None,:] & mask_n[:,None]), other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
                else:
                    inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[:,None] & mask_n[None,:]), other = 0.0) # [TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                if use_tensor_core:
                    expars_grad = tl.dot(nflow_sub_logz_sub, inpars).log() + nflow_sub_logz_max + expars
                else:
                    expars_grad = tl.sum(nflow_sub_logz_sub[:,None,:] * inpars[None,:,:], axis = 2).log() + nflow_sub_logz_max + expars

                tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, -tl.exp(expars_grad), mask = (mask_b[:,None] & mask_c[None,:]))

    @staticmethod
    @triton_jit
    def sample_kernel(samples_ptr, params_ptr, nflow_xids_ptr, nflow_yids_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr,
                      num_activ_nodes, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, batch_size: tl.constexpr, seed, 
                      categorical_evidence_logp_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr,
                      TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_S: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_S

        offsets = block_start + tl.arange(0, BLOCK_S) # [BLOCK_S]
        mask = offsets < num_activ_nodes

        # Raw batch and (local) node id
        local_offsets = tl.load(nflow_xids_ptr + offsets, mask = mask, other = 0)
        batch_offsets = tl.load(nflow_yids_ptr + offsets, mask = mask, other = 0)

        # Load variable ids from `vids_ptr`
        vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)
        lvids = tl.load(var_idmapping_ptr + vids, mask = mask, other = 0)

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64) # [BLOCK_SIZE]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)

        # Ptrs pointing to internal parameters
        inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_S, TILE_SIZE_K]

        # Ptrs pointing to external parameters
        expars_ptr = categorical_evidence_logp + \
            batch_offsets[:,None] * (ext_num_vars * max_num_cats) + \
            lvids[:,None] * max_num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:]  # [BLOCK_S, TILE_SIZE_K]

        # Compute logZ
        logZ = tl.zeros([BLOCK_S], dtype = tl.float32) - float("inf")
        for i in range(K_NUM_TILES):
            cat_mask = mask[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats[:,None])

            inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)
            expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)

            addlpar = inpar.log() + expar
            addlpar_max = tl.max(addlpar, axis = 1)
            lpar = (addlpar - addlpar_max[:,None]).exp().sum(axis = 1).log() + addlpar_max

            # Compute log-add-exp(logZ, lpar)
            maxval = tl.maximum(logZ, lpar)
            minval = tl.minimum(logZ, lpar)
            diff = minval - maxval

            logZ = tl.where(logZ == -float("inf"),
                lpar,
                maxval + tlmath.log1p(tl.exp(diff))
            )

        # Generate random number
        rnd_val = tl.rand(seed, offsets)

        # Draw samples
        sampled_ids = tl.zeros([BLOCK_S], dtype = tl.int64) - 1
        for i in range(K_NUM_TILES):
            cat_mask = mask[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats[:,None])

            inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)
            expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask, other = 0.0)

            probs = tl.exp(tl.log(inpar) + expar - logZ[:,None]) # [BLOCK_S, TILE_SIZE_K]
            cum_probs = tl.cumsum(probs, axis = 1) # [BLOCK_S, TILE_SIZE_K]

            local_catids = tl.sum((rnd_val[:,None] >= cum_probs).to(tl.int64), axis = 1) # [BLOCK_S]

            is_overflow = (local_catids == TILE_SIZE_K)
            rnd_val = tl.where(is_overflow, rnd_val - tl.sum(probs, axis = 1), rnd_val)
            sampled_ids = tl.where(is_overflow | (sampled_ids > -1), sampled_ids, local_catids + i * TILE_SIZE_K)

        # Write back to `samples`
        sample_offsets = vids * batch_size + batch_offsets
        tl.store(samples_ptr + sample_offsets, sampled_ids, mask = mask)

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
        return SoftEvidenceCategorical, {"num_cats": self.num_cats}

    def __reduce__(self):
        return (self.__class__, (self.num_cats,))
