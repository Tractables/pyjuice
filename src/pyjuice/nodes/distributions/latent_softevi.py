from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional, Any

from .distributions import Distribution
from pyjuice.utils.kernel_launcher import triton_jit


def _layer_latent_dims(layer):
    """
    Number of latent-evidence variables in `layer` and the per-variable node count (= max(nids) + 1).

    Both are static properties of the compiled layer, so they are computed once and cached to avoid a
    device synchronization on every forward/backward call.
    """
    dims = getattr(layer, "_latent_softevi_dims", None)
    if dims is None:
        num_vars = int(layer.var_idmapping.max().item()) + 1
        num_latents = int(layer.nids.max().item()) + 1
        dims = (num_vars, num_latents)
        layer._latent_softevi_dims = dims

    return dims


def _check_evidence_tensor(layer, tensor, name, batch_size):
    assert tensor.dim() == 3, f"`{name}` should be of shape [batch_size, ext_num_vars, num_latents]."
    assert tensor.size(0) == batch_size, f"Batch size doesn't match in `{name}`."
    assert tensor.dtype == torch.float32, f"`{name}` should be of dtype `torch.float32`."
    assert tensor.is_contiguous(), f"`{name}` should be contiguous."

    num_vars, num_latents = _layer_latent_dims(layer)
    assert tensor.size(1) >= num_vars, \
        f"`{name}` has {tensor.size(1)} variables, but the layer defines {num_vars}."
    assert tensor.size(2) >= num_latents, \
        f"`{name}` has {tensor.size(2)} latent states, but the layer defines {num_latents} nodes per variable."


def _condition_apply_fw_kernel(layer, kwargs):
    return "latent_evidence_logp" in kwargs


def _prep_args_apply_fw_kernel(layer, kwargs):
    target_kwargs = dict()

    latent_evidence_logp = kwargs["latent_evidence_logp"]
    _check_evidence_tensor(layer, latent_evidence_logp, "latent_evidence_logp", kwargs["batch_size"])

    target_kwargs["latent_evidence_logp_ptr"] = latent_evidence_logp

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = latent_evidence_logp.size(1)

    target_kwargs["num_latents"] = latent_evidence_logp.size(2)

    target_kwargs["BLOCK_SIZE"] = 1024

    return target_kwargs, None


def _condition_apply_bk_kernel(layer, kwargs):
    return "latent_evidence_logp_grad" in kwargs


def _prep_args_apply_bk_kernel(layer, kwargs):
    target_kwargs = dict()

    latent_evidence_logp_grad = kwargs["latent_evidence_logp_grad"]
    _check_evidence_tensor(layer, latent_evidence_logp_grad, "latent_evidence_logp_grad", kwargs["batch_size"])

    # Every `(b, lvid, nid)` slot is written by exactly one node, so the kernel can use a plain store;
    # zeroing here keeps the slots no node writes to (padding latents/variables) well-defined.
    latent_evidence_logp_grad.zero_()

    target_kwargs["latent_evidence_logp_grad_ptr"] = latent_evidence_logp_grad

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = latent_evidence_logp_grad.size(1)

    target_kwargs["num_latents"] = latent_evidence_logp_grad.size(2)

    target_kwargs["BLOCK_SIZE"] = 1024

    return target_kwargs, None


class LatentSoftEvidence(Distribution):
    """
    A parameter-free input distribution that injects an externally supplied, per-latent additive
    log-potential into the PC.

    Node `(v, i)` -- the node for latent state `i` of latent-evidence variable `v` -- takes value

        node_mars[(v,i), b] = latent_evidence_logp[b, v, i]

    when `latent_evidence_logp` is supplied to the forward pass, and `0.0` (the additive identity, i.e.
    the latent channel is off) otherwise. The potential is raw: no normalization over `i` is assumed or
    imposed -- normalization is the job of the sum node above and of the global partition function.

    Multiplying such a node into the latent branch of a block position, state-aligned with the sum node,
    yields a product of experts at the latent level:

        q(x) prop_to  sum_z p_omega(x, z) * prod_v p_theta(z_v)

    The backward pass writes the node's *linear* flow (the marginal responsibility of state `i` at `v`)
    into `latent_evidence_logp_grad`, which is exactly `d (sum_b log f(x_b)) / d latent_evidence_logp`
    -- regardless of `logspace_flows`. A minimizing trainer therefore backprops `-grad`.

    The distribution holds no parameters and produces no parameter flows, so it never participates in
    EM; it only reweights the latent posterior that drives the sum-node and emission EM updates.
    """

    def __init__(self):
        super(LatentSoftEvidence, self).__init__()

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
        return "LatentSoftEvidence"

    def requires_external_inputs(self):
        """
        The kernels index the evidence buffer by layer-local variable id, which requires the
        `var_idmapping` buffer to be built for this layer.

        :note: the base implementation keys off an "Extern" substring in the signature; overridden here
               so the signature can stay descriptive.
        """
        return True

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

        :note: the data is ignored by this distribution; `long` is chosen so the `[B, num_vars]` data
               tensor can be shared with the categorical (emission) input layers without friction.
        """
        return torch.long

    @staticmethod
    @triton_jit
    def fw_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr,
                  fw_local_ids_ptr, partial_eval: tl.constexpr, layer_num_nodes: tl.constexpr, batch_size: tl.constexpr,
                  num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                  latent_evidence_logp_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, num_latents: tl.constexpr):
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

        # Get all latent offsets (the node's structural latent state index)
        nids = tl.load(nids_ptr + local_offsets, mask = mask, other = 0)

        # Load the corresponding log-potential
        latent_evi = tl.load(latent_evidence_logp_ptr + batch_offsets * (ext_num_vars * num_latents) + lvids * num_latents + nids, mask = mask, other = 0.0)

        node_offsets = local_offsets + node_offset
        tl.store(node_mars_ptr + node_offsets * batch_size + batch_offsets, latent_evi, mask = mask)

    @staticmethod
    @triton_jit
    def bk_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                  metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr,
                  batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr, latent_evidence_logp_grad_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, num_latents: tl.constexpr):
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

        # Get all latent offsets (the node's structural latent state index)
        nids = tl.load(nids_ptr + local_offsets, mask = mask, other = 0)

        # Load the flows
        ns_offsets = (local_offsets + node_offset) * batch_size + batch_offsets
        flows = tl.load(node_flows_ptr + ns_offsets, mask = mask, other = 0.0)

        # `d log f / d latent_evidence_logp` is the LINEAR-space flow, so undo the log storage when
        # `node_flows` holds `log phi`.
        if logspace_flows:
            flows = tl.exp(flows)

        # Store the corresponding gradient
        tl.store(latent_evidence_logp_grad_ptr + batch_offsets * (ext_num_vars * num_latents) + lvids * num_latents + nids, flows, mask = mask)

    @staticmethod
    def fw_mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE):
        # Additive identity: with no evidence supplied, the node does not affect the PC
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
        return LatentSoftEvidence, {}

    def __reduce__(self):
        return (self.__class__, ())
