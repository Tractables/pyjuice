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
    return "indicator_evidence_logp" in kwargs


def _prep_args_apply_fw_kernel(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]

    indicator_evidence_logp = kwargs["indicator_evidence_logp"]
    assert indicator_evidence_logp.size(0) == batch_size, "Batch size doesn't match in `indicator_evidence_logp`."

    ext_num_vars = indicator_evidence_logp.size(1)
    target_kwargs["ext_num_vars"] = ext_num_vars

    num_states = indicator_evidence_logp.size(2)
    for ns in layer.nodes:
        assert num_states <= ns.dist.num_states
    target_kwargs["num_states"] = num_states

    target_kwargs["indicator_evidence_logp_ptr"] = indicator_evidence_logp
    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    # prepare BLOCK_SIZE
    BLOCK_SIZE_B = min(2048, triton.next_power_of_2(batch_size))
    BLOCK_SIZE_N = max(1, 2048 // BLOCK_SIZE_B)

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N

    return target_kwargs, grid


def _condition_apply_bk_kernel(layer, kwargs):
    return "indicator_evidence_logp" in kwargs and "indicator_evidence_logp_grad" in kwargs


def _prep_args_apply_bk_kernel(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]

    indicator_evidence_logp = kwargs["indicator_evidence_logp"]
    assert indicator_evidence_logp.size(0) == batch_size, "Batch size doesn't match in `indicator_evidence_logp`."

    ext_num_vars = indicator_evidence_logp.size(1)
    target_kwargs["ext_num_vars"] = ext_num_vars

    num_states = indicator_evidence_logp.size(2)
    for ns in layer.nodes:
        assert num_states <= ns.dist.num_states
    target_kwargs["num_states"] = num_states

    indicator_evidence_logp_grad = kwargs["indicator_evidence_logp_grad"]
    assert indicator_evidence_logp_grad.size(0) == batch_size
    assert indicator_evidence_logp_grad.size(1) == ext_num_vars
    assert indicator_evidence_logp_grad.size(2) == num_states

    target_kwargs["indicator_evidence_logp_ptr"] = indicator_evidence_logp
    target_kwargs["indicator_evidence_logp_grad_ptr"] = indicator_evidence_logp_grad
    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    # prepare BLOCK_SIZE
    BLOCK_SIZE_B = min(2048, triton.next_power_of_2(batch_size))
    BLOCK_SIZE_N = max(1, 2048 // BLOCK_SIZE_B)

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N

    return target_kwargs, grid


class SoftEvidenceIndicator(Distribution):
    """
    A class representing an Indicator distribution that allows external soft evidence.

    :param num_cats: number of categories
    :type num_cats: int
    """
    def __init__(self, num_states: int):
        super(SoftEvidenceIndicator, self).__init__()

        self.num_states = num_states

        self.post_fw_fns = [
            (self.fw_kernel, _condition_apply_fw_kernel, _prep_args_apply_fw_kernel)
        ]

        self.post_bp_fns = [
            (self.bk_kernel, _condition_apply_bk_kernel, _prep_args_apply_bk_kernel)
        ]
        
    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        return f"ExternSoftEvidenceIndicator"

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return []

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        return 1

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
        assert num_nodes == self.num_states, "The number of nodes should match the pre-defined number of states."

        params = torch.arange(num_nodes).float()

        return params

    @staticmethod
    @triton_jit
    def fw_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, fw_local_ids_ptr, layer_num_nodes,
                  batch_size, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset, partial_eval: tl.constexpr,
                  BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, indicator_evidence_logp_ptr, var_idmapping_ptr, 
                  num_states: tl.constexpr, ext_num_vars: tl.constexpr):
        
        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        # Get all variable ids
        vids = tl.load(vids_ptr + offsets_n) # [BLOCK_SIZE_N]
        lvids = tl.load(var_idmapping_ptr + vids) # Variable ID for "this type of inputs"

        # Get latent offset of all nodes
        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)

        # Get start parameter indices and the literals
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        lits = tl.load(params_ptr + s_pids, mask = mask_n, other = 0).to(tl.int64)

        # Load external soft evidence
        ext_evi_ptr = indicator_evidence_logp_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_states) + \
            lvids[None,:] * num_states + \
            lits[None,:] # [BLOCK_SIZE_B, TILE_SIZE_N]
        ext_evi = tl.load(ext_evi_ptr, mask = (mask_b[:,None] & mask_n[None,:]), other = 0)

        # Store results
        node_offsets = offsets_n + node_offset
        tl.store(node_mars_ptr + node_offsets[None,:] * batch_size + offsets_b[:,None], ext_evi, mask = (mask_b[:,None] & mask_n[None,:]))

    @staticmethod
    @triton_jit
    def bk_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr, nids_ptr,
                  bk_local_ids_ptr, layer_num_nodes, batch_size, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                  node_offset, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                  indicator_evidence_logp_ptr, indicator_evidence_logp_grad_ptr, var_idmapping_ptr, num_states: tl.constexpr, ext_num_vars: tl.constexpr):

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        # Get all variable ids
        vids = tl.load(vids_ptr + offsets_n) # [BLOCK_SIZE_B]
        lvids = tl.load(var_idmapping_ptr + vids) # Variable ID for "this type of inputs"

        # Get latent offset of all nodes
        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)

        # Get start parameter indices and the literals
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        lits = tl.load(params_ptr + s_pids, mask = mask_n, other = 0).to(tl.int64)

        # Load nflows
        nflows_ptr = node_flows_ptr + \
            (offsets_n + node_offset)[None,:] * batch_size + \
            offsets_b[:,None] # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        nflows = tl.load(nflows_ptr, mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Store external soft evidence
        ext_evi_ptr = indicator_evidence_logp_grad_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_states) + \
            lvids[None,:] * num_states + \
            lits[None,:] # [BLOCK_SIZE_B, TILE_SIZE_N]
        tl.store(ext_evi_ptr, nflows, mask = (mask_b[:,None] & mask_n[None,:]))
