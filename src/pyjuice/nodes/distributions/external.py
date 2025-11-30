from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional, Any

from .distributions import Distribution
from pyjuice.utils.kernel_launcher import triton_jit


def _condition_apply_soft_evi_kernel(layer, kwargs):
    return "external_soft_evi" in kwargs


def _prep_args_apply_soft_evi_kernel(layer, kwargs):
    target_kwargs = dict()

    assert "external_soft_evi" in kwargs
    external_soft_evi = kwargs["external_soft_evi"]
    assert external_soft_evi.dim() == 3

    target_kwargs["external_soft_evi_ptr"] = external_soft_evi

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_soft_evi.size(1)

    target_kwargs["max_num_latents"] = external_soft_evi.size(2)

    target_kwargs["BLOCK_SIZE"] = 1024

    return target_kwargs, None


def _condition_soft_evi_grad_kernel(layer, kwargs):
    return "external_soft_evi_grad" in kwargs


def _prep_args_soft_evi_grad_kernel(layer, kwargs):
    target_kwargs = dict()

    assert "external_soft_evi_grad" in kwargs
    external_soft_evi_grad = kwargs["external_soft_evi_grad"]
    assert external_soft_evi_grad.dim() == 3

    target_kwargs["external_soft_evi_grad_ptr"] = external_soft_evi_grad

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_soft_evi_grad.size(1)

    target_kwargs["max_num_latents"] = external_soft_evi_grad.size(2)

    target_kwargs["BLOCK_SIZE"] = 1024

    return target_kwargs, None


class External(Distribution):
    """
    A class representing user-define distributions (PyJuice only processes the incoming log-probabilities).
    """

    def __init__(self):
        super(External, self).__init__()

        self.post_fw_fns = [
            (self.apply_soft_evi_kernel, _condition_apply_soft_evi_kernel, _prep_args_apply_soft_evi_kernel)
        ]

        self.post_bp_fns = [
            (self.soft_evi_grad_kernel, _condition_soft_evi_grad_kernel, _prep_args_soft_evi_grad_kernel)
        ]

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return "External"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return []

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        return 0

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        return 0

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[Any] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        return torch.zeros([0], dtype = torch.float32)

    def get_data_dtype(self):
        """
        Get the data dtype for the distribution.
        """
        return torch.float32

    @staticmethod
    @triton_jit
    def apply_soft_evi_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, 
                              fw_local_ids_ptr, partial_eval: tl.constexpr, layer_num_nodes: tl.constexpr, batch_size: tl.constexpr, 
                              num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                              external_soft_evi_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_latents: tl.constexpr):
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

        # Load the corresponding data
        soft_evi = tl.load(external_soft_evi_ptr + batch_offsets * (ext_num_vars * max_num_latents) + lvids * max_num_latents + nids, mask = mask, other = 0.0)

        node_offsets = local_offsets + node_offset
        tl.store(node_mars_ptr + node_offsets * batch_size + batch_offsets, soft_evi, mask = mask)

    @staticmethod
    @triton_jit
    def soft_evi_grad_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                             metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                             batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                             BLOCK_SIZE: tl.constexpr, external_soft_evi_grad_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_latents: tl.constexpr):
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

        # Load the flows
        ns_offsets = (local_offsets + node_offset) * batch_size + batch_offsets
        flows = tl.load(node_flows_ptr + ns_offsets, mask = mask, other = 0)

        # Store the corresponding data
        tl.store(external_soft_evi_grad_ptr + batch_offsets * (ext_num_vars * max_num_latents) + lvids * max_num_latents + nids, flows, mask = mask)

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
    def em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE):
        pass

    def _get_constructor(self):
        return External, {}

    def __reduce__(self):
        return (self.__class__, ())
