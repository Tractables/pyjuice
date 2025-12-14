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


def max_power_of_2_factor(n):
    if n == 0:
        return 0
    if n % 2 != 0:
        return 1

    power_of_2 = 1
    while n % 2 == 0:
        power_of_2 *= 2
        n //= 2  # Use integer division

    return power_of_2


def _condition_apply_ll_kernel(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs and \
        kwargs["extern_product_categorical_mode"] in ("normalized_ll", "unnormalized_ll", "normalizing_constant") and \
        ("external_categorical_value_mask" not in kwargs or kwargs["external_categorical_value_mask"] is None)


def _prep_args_apply_ll_kernel(layer, kwargs):
    target_kwargs = dict()

    assert "external_categorical_logps" in kwargs
    external_categorical_logps = kwargs["external_categorical_logps"]

    if kwargs["extern_product_categorical_mode"] == "normalized_ll":
        assert external_categorical_logps.dim() == 3
        compute_unnorm_logp = True
        compute_logz = True
        ext_softevi_indexing = True
    elif kwargs["extern_product_categorical_mode"] == "unnormalized_ll":
        compute_unnorm_logp = True
        compute_logz = False
        if external_categorical_logps.dim() == 2:
            ext_softevi_indexing = False
        elif external_categorical_logps.dim() == 3:
            ext_softevi_indexing = True
        else:
            raise ValueError()
    elif kwargs["extern_product_categorical_mode"] == "normalizing_constant":
        assert external_categorical_logps.dim() == 3
        compute_unnorm_logp = False
        compute_logz = True
        ext_softevi_indexing = True
    else:
        raise ValueError("Unexpected `extern_product_categorical_mode`. Should be 'normalized_ll', 'unnormalized_ll', or 'normalizing_constant'.")

    assert kwargs["batch_size"] == external_categorical_logps.size(0), "Batch size doesn't match in `external_categorical_logps`."

    target_kwargs["compute_unnorm_logp"] = compute_unnorm_logp
    target_kwargs["compute_logz"] = compute_logz
    target_kwargs["ext_softevi_indexing"] = ext_softevi_indexing

    target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    if ext_softevi_indexing:
        for ns in layer.nodes:
            assert ns.dist.num_cats <= external_categorical_logps.size(2)

    if external_categorical_logps.dim() == 3:
        target_kwargs["max_num_cats"] = external_categorical_logps.size(2)
    else:
        target_kwargs["max_num_cats"] = 1

    # prepare BLOCK_SIZE and TILE_SIZE_K
    target_kwargs["TILE_SIZE_K"] = min(128, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    target_kwargs["BLOCK_SIZE"] = 1024 // target_kwargs["TILE_SIZE_K"]

    return target_kwargs, None


def _condition_apply_ll_w_mask_kernel(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs and \
        kwargs["extern_product_categorical_mode"] in ("normalized_ll", "unnormalized_ll") and \
        ("external_categorical_value_mask" in kwargs and kwargs["external_categorical_value_mask"] is not None)


def _prep_args_apply_ll_w_mask_kernel(layer, kwargs):
    target_kwargs = dict()

    assert "external_categorical_logps" in kwargs
    external_categorical_logps = kwargs["external_categorical_logps"]
    assert external_categorical_logps.dim() == 3

    if kwargs["extern_product_categorical_mode"] == "normalized_ll":
        use_normalized = True
    elif kwargs["extern_product_categorical_mode"] == "unnormalized_ll":
        use_normalized = False
    else:
        raise ValueError("Unexpected `extern_product_categorical_mode`. Should be 'normalized_ll' or 'unnormalized_ll'.")

    batch_size = kwargs["batch_size"]

    external_categorical_value_mask = kwargs["external_categorical_value_mask"]
    assert external_categorical_value_mask.dim() == 2

    assert external_categorical_logps.size(0) == batch_size, "Batch size doesn't match in `external_categorical_logps`."
    assert external_categorical_value_mask.size(0) == batch_size, "Batch size doesn't match in `external_categorical_value_mask`."

    target_kwargs["use_normalized"] = use_normalized

    target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps
    target_kwargs["external_categorical_value_mask_ptr"] = external_categorical_value_mask

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    for ns in layer.nodes:
        assert ns.dist.num_cats <= external_categorical_logps.size(2)

    if external_categorical_logps.dim() == 3:
        target_kwargs["max_num_cats"] = external_categorical_logps.size(2)
    else:
        target_kwargs["max_num_cats"] = 1

    # prepare BLOCK_SIZE and TILE_SIZE_K
    target_kwargs["TILE_SIZE_K"] = min(128, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    target_kwargs["BLOCK_SIZE"] = 1024 // target_kwargs["TILE_SIZE_K"]

    return target_kwargs, None


def _condition_apply_ll_bp_kernel(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs and \
        kwargs["extern_product_categorical_mode"] in ("normalized_ll", "unnormalized_ll", "normalizing_constant") and \
        ("external_categorical_value_mask" not in kwargs or kwargs["external_categorical_value_mask"] is None)


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

    target_kwargs["compute_unnorm_logp"] = compute_unnorm_logp
    target_kwargs["compute_logz"] = compute_logz

    batch_size = kwargs["batch_size"]

    if kwargs["extern_product_categorical_mode"] == "normalizing_constant":
        assert "external_categorical_logps" in kwargs
        external_categorical_logps = kwargs["external_categorical_logps"]
        assert external_categorical_logps.dim() == 3

        assert kwargs["batch_size"] == external_categorical_logps.size(0), "Batch size doesn't match in `external_categorical_logps`."

        target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps

        target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

        target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

        target_kwargs["max_num_cats"] = external_categorical_logps.size(2)
    else:
        target_kwargs["external_categorical_logps_ptr"] = None

        target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

        target_kwargs["ext_num_vars"] = 1

        target_kwargs["max_num_cats"] = 1

    if not layer.provided("fw_local_ids"):
        layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    else:
        layer_num_nodes = layer.fw_local_ids.size(0)

    # prepare BLOCK_SIZE and TILE_SIZE_K
    if kwargs["extern_product_categorical_mode"] == "normalizing_constant":
        target_kwargs["TILE_SIZE_K"] = min(64, triton.next_power_of_2(target_kwargs["max_num_cats"]))
        target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        target_kwargs["BLOCK_SIZE_B"] = min(128, 1024 // target_kwargs["TILE_SIZE_K"], BATCH_SIZE_NP2)
        target_kwargs["BLOCK_SIZE_N"] = min(128, 1024 // target_kwargs["BLOCK_SIZE_B"] // target_kwargs["TILE_SIZE_K"])
    else:
        target_kwargs["TILE_SIZE_K"] = 1
        target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        target_kwargs["BLOCK_SIZE_B"] = min(128, BATCH_SIZE_NP2)
        target_kwargs["BLOCK_SIZE_N"] = min(128, 1024 // target_kwargs["BLOCK_SIZE_B"])

    grid = (triton.cdiv(batch_size, target_kwargs["BLOCK_SIZE_B"]),
            triton.cdiv(layer_num_nodes, target_kwargs["BLOCK_SIZE_N"]))

    return target_kwargs, grid


def _condition_apply_ll_bp_w_mask_kernel1(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs and \
        kwargs["extern_product_categorical_mode"] in ("normalized_ll", "unnormalized_ll") and \
        ("external_categorical_value_mask" in kwargs and kwargs["external_categorical_value_mask"] is not None)


def _prep_args_apply_ll_bp_w_mask_kernel1(layer, kwargs):
    target_kwargs = dict()

    if kwargs["extern_product_categorical_mode"] == "normalized_ll":
        use_normalized = True
    elif kwargs["extern_product_categorical_mode"] == "unnormalized_ll":
        use_normalized = False
    else:
        raise ValueError("Unexpected `extern_product_categorical_mode`. Should be 'normalized_ll' or 'unnormalized_ll'.")

    batch_size = kwargs["batch_size"]

    external_categorical_value_mask = kwargs["external_categorical_value_mask"]
    assert external_categorical_value_mask.dim() == 2

    assert "external_categorical_logps" in kwargs
    external_categorical_logps = kwargs["external_categorical_logps"]
    assert external_categorical_logps.dim() == 3

    assert external_categorical_logps.size(0) == batch_size, "Batch size doesn't match in `external_categorical_logps`."
    assert external_categorical_value_mask.size(0) == batch_size, "Batch size doesn't match in `external_categorical_value_mask`."

    target_kwargs["use_normalized"] = use_normalized

    assert kwargs["batch_size"] == external_categorical_logps.size(0), "Batch size doesn't match in `external_categorical_logps`."

    target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps
    target_kwargs["external_categorical_value_mask_ptr"] = external_categorical_value_mask

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    target_kwargs["max_num_cats"] = 1

    if not layer.provided("fw_local_ids"):
        layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    else:
        layer_num_nodes = layer.fw_local_ids.size(0)

    target_kwargs["TILE_SIZE_K"] = 1
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
    target_kwargs["BLOCK_SIZE_B"] = min(128, BATCH_SIZE_NP2)
    target_kwargs["BLOCK_SIZE_N"] = min(128, 1024 // target_kwargs["BLOCK_SIZE_B"])

    grid = (triton.cdiv(batch_size, target_kwargs["BLOCK_SIZE_B"]),
            triton.cdiv(layer_num_nodes, target_kwargs["BLOCK_SIZE_N"]))

    return target_kwargs, grid


def _condition_apply_ll_bp_w_mask_kernel2(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs and \
        kwargs["extern_product_categorical_mode"] in ("normalized_ll", "unnormalized_ll") and \
        ("external_categorical_value_mask" in kwargs and kwargs["external_categorical_value_mask"] is not None)


def _prep_args_apply_ll_bp_w_mask_kernel2(layer, kwargs):
    target_kwargs = dict()

    if kwargs["extern_product_categorical_mode"] == "normalized_ll":
        use_normalized = True
    elif kwargs["extern_product_categorical_mode"] == "unnormalized_ll":
        use_normalized = False
    else:
        raise ValueError("Unexpected `extern_product_categorical_mode`. Should be 'normalized_ll' or 'unnormalized_ll'.")

    batch_size = kwargs["batch_size"]

    external_categorical_value_mask = kwargs["external_categorical_value_mask"]
    assert external_categorical_value_mask.dim() == 2

    assert "external_categorical_logps" in kwargs
    external_categorical_logps = kwargs["external_categorical_logps"]
    assert external_categorical_logps.dim() == 3

    assert external_categorical_logps.size(0) == batch_size, "Batch size doesn't match in `external_categorical_logps`."
    assert external_categorical_value_mask.size(0) == batch_size, "Batch size doesn't match in `external_categorical_value_mask`."

    target_kwargs["use_normalized"] = use_normalized

    assert kwargs["batch_size"] == external_categorical_logps.size(0), "Batch size doesn't match in `external_categorical_logps`."

    target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps
    target_kwargs["external_categorical_value_mask_ptr"] = external_categorical_value_mask

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    target_kwargs["max_num_cats"] = external_categorical_logps.size(2)

    assert external_categorical_value_mask.size(0) == kwargs["batch_size"]
    assert external_categorical_value_mask.size(1) == target_kwargs["ext_num_vars"]

    if not layer.provided("fw_local_ids"):
        layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    else:
        layer_num_nodes = layer.fw_local_ids.size(0)

    target_kwargs["TILE_SIZE_K"] = min(64, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
    target_kwargs["BLOCK_SIZE_B"] = min(128, 1024 // target_kwargs["TILE_SIZE_K"], BATCH_SIZE_NP2)
    target_kwargs["BLOCK_SIZE_N"] = min(128, 1024 // target_kwargs["BLOCK_SIZE_B"] // target_kwargs["TILE_SIZE_K"])

    grid = (triton.cdiv(batch_size, target_kwargs["BLOCK_SIZE_B"]),
            triton.cdiv(layer_num_nodes, target_kwargs["BLOCK_SIZE_N"]))

    return target_kwargs, grid


def _condition_apply_ll_bp_extern_grad_kernel(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs and \
        "external_categorical_logps_grad" in kwargs and \
        kwargs["extern_product_categorical_mode"] in ("normalized_ll", "unnormalized_ll", "normalizing_constant") and \
        ("external_categorical_value_mask" not in kwargs or kwargs["external_categorical_value_mask"] is None)


def _prep_args_apply_ll_bp_extern_grad_kernel(layer, kwargs):
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

    target_kwargs["compute_unnorm_logp"] = compute_unnorm_logp
    target_kwargs["compute_logz"] = compute_logz

    assert "external_categorical_logps" in kwargs
    external_categorical_logps = kwargs["external_categorical_logps"]
    assert external_categorical_logps.dim() == 3

    assert kwargs["batch_size"] == external_categorical_logps.size(0), "Batch size doesn't match in `external_categorical_logps`."

    assert "external_categorical_logps_grad" in kwargs
    external_categorical_logps_grad = kwargs["external_categorical_logps_grad"]
    assert external_categorical_logps_grad.shape == external_categorical_logps.shape
    target_kwargs["external_categorical_logps_grad_ptr"] = external_categorical_logps_grad

    external_categorical_logps_grad.zero_()

    target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    target_kwargs["max_num_cats"] = external_categorical_logps.size(2)

    n_block_size = max_power_of_2_factor(layer.n_block_size)

    if not layer.provided("fw_local_ids"):
        layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    else:
        layer_num_nodes = layer.fw_local_ids.size(0)

    target_kwargs["TILE_SIZE_K"] = min(64, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    BATCH_SIZE_NP2 = triton.next_power_of_2(external_categorical_logps.size(0))
    target_kwargs["BLOCK_SIZE_B"] = min(128, 1024 // target_kwargs["TILE_SIZE_K"], BATCH_SIZE_NP2)
    target_kwargs["BLOCK_SIZE_N"] = min(n_block_size, 1024 // target_kwargs["BLOCK_SIZE_B"], 1024 // target_kwargs["TILE_SIZE_K"])

    target_kwargs["use_tensor_core"] = (target_kwargs["TILE_SIZE_K"] >= 16) and \
                                       (target_kwargs["BLOCK_SIZE_B"] >= 16) and \
                                       (target_kwargs["BLOCK_SIZE_N"] >= 16)

    grid = (triton.cdiv(external_categorical_logps.size(0), target_kwargs["BLOCK_SIZE_B"]),
            triton.cdiv(layer_num_nodes, target_kwargs["BLOCK_SIZE_N"]))

    return target_kwargs, grid


def _condition_apply_ll_bp_extern_grad_w_mask_kernel(layer, kwargs):
    return "extern_product_categorical_mode" in kwargs and \
        "external_categorical_logps_grad" in kwargs and \
        kwargs["extern_product_categorical_mode"] in ("normalized_ll", "unnormalized_ll") and \
        ("external_categorical_value_mask" in kwargs and kwargs["external_categorical_value_mask"] is not None)


def _prep_args_apply_ll_bp_extern_grad_w_mask_kernel(layer, kwargs):
    target_kwargs = dict()

    if kwargs["extern_product_categorical_mode"] == "normalized_ll":
        use_normalized = True
    elif kwargs["extern_product_categorical_mode"] == "unnormalized_ll":
        use_normalized = False
    else:
        raise ValueError("Unexpected `extern_product_categorical_mode`. Should be 'normalized_ll', 'unnormalized_ll'.")

    target_kwargs["use_normalized"] = use_normalized

    assert "external_categorical_logps" in kwargs
    external_categorical_logps = kwargs["external_categorical_logps"]
    assert external_categorical_logps.dim() == 3

    assert kwargs["batch_size"] == external_categorical_logps.size(0), "Batch size doesn't match in `external_categorical_logps`."

    assert "external_categorical_logps_grad" in kwargs
    external_categorical_logps_grad = kwargs["external_categorical_logps_grad"]
    assert external_categorical_logps_grad.shape == external_categorical_logps.shape
    target_kwargs["external_categorical_logps_grad_ptr"] = external_categorical_logps_grad

    external_categorical_logps_grad.zero_()

    external_categorical_value_mask = kwargs["external_categorical_value_mask"]

    assert external_categorical_value_mask.size(0) == kwargs["batch_size"]
    assert external_categorical_value_mask.size(1) == external_categorical_logps.size(1)

    target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps
    target_kwargs["external_categorical_value_mask_ptr"] = external_categorical_value_mask

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    target_kwargs["max_num_cats"] = external_categorical_logps.size(2)

    n_block_size = max_power_of_2_factor(layer.n_block_size)

    if not layer.provided("fw_local_ids"):
        layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    else:
        layer_num_nodes = layer.fw_local_ids.size(0)

    target_kwargs["TILE_SIZE_K"] = min(64, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    BATCH_SIZE_NP2 = triton.next_power_of_2(external_categorical_logps.size(0))
    target_kwargs["BLOCK_SIZE_B"] = min(128, 1024 // target_kwargs["TILE_SIZE_K"], BATCH_SIZE_NP2)
    target_kwargs["BLOCK_SIZE_N"] = min(n_block_size, 1024 // target_kwargs["BLOCK_SIZE_B"], 1024 // target_kwargs["TILE_SIZE_K"])

    target_kwargs["use_tensor_core"] = (target_kwargs["TILE_SIZE_K"] >= 16) and \
                                       (target_kwargs["BLOCK_SIZE_B"] >= 16) and \
                                       (target_kwargs["BLOCK_SIZE_N"] >= 16)

    grid = (triton.cdiv(external_categorical_logps.size(0), target_kwargs["BLOCK_SIZE_B"]),
            triton.cdiv(layer_num_nodes, target_kwargs["BLOCK_SIZE_N"]))

    return target_kwargs, grid


def _condition_sample_kernel(layer, kwargs):
    assert "external_categorical_logps" in kwargs, "`external_categorical_logps` must be provided to sample from `ExternProductCategorical` distributions."
    return "external_categorical_logps" in kwargs and kwargs["external_categorical_logps"].dim() == 3


def _prep_args_sample_kernel(layer, kwargs):
    target_kwargs = dict()

    external_categorical_logps = kwargs["external_categorical_logps"]

    assert kwargs["batch_size"] == external_categorical_logps.size(0), "Batch size doesn't match."

    target_kwargs["external_categorical_logps_ptr"] = external_categorical_logps

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    target_kwargs["max_num_cats"] = external_categorical_logps.size(2)

    num_activ_nodes = kwargs["num_activ_nodes"]

    target_kwargs["TILE_SIZE_K"] = min(64, triton.next_power_of_2(target_kwargs["max_num_cats"]))
    target_kwargs["K_NUM_TILES"] = triton.cdiv(target_kwargs["max_num_cats"], target_kwargs["TILE_SIZE_K"])
    target_kwargs["BLOCK_S"] = min(64, 1024 // target_kwargs["TILE_SIZE_K"], triton.next_power_of_2(num_activ_nodes))

    grid = (triton.cdiv(num_activ_nodes, target_kwargs["BLOCK_S"]),)

    return target_kwargs, grid


def _condition_sample_set_hard_context_kernel(layer, kwargs):
    return "inputs" in kwargs and "external_categorical_value_mask" in kwargs


def _prep_args_sample_set_hard_context_kernel(layer, kwargs):
    target_kwargs = dict()

    inputs = kwargs["inputs"]
    external_categorical_value_mask = kwargs["external_categorical_value_mask"]
    external_categorical_logps = kwargs["external_categorical_logps"]

    batch_size = kwargs["batch_size"]
    assert external_categorical_value_mask.size(0) == batch_size, "Batch size doesn't match."
    assert inputs.size(0) == batch_size, "Batch size doesn't match."

    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping
    target_kwargs["ext_num_vars"] = external_categorical_logps.size(1)

    target_kwargs["num_vars"] = layer.pc_num_vars

    target_kwargs["inputs_ptr"] = inputs
    target_kwargs["external_categorical_value_mask_ptr"] = external_categorical_value_mask

    target_kwargs["BLOCK_SIZE"] = min(512, triton.next_power_of_2(batch_size * target_kwargs["ext_num_vars"]))

    grid = (triton.cdiv(batch_size * target_kwargs["ext_num_vars"], target_kwargs["BLOCK_SIZE"]),)

    return target_kwargs, grid


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
            (self.ll_kernel, _condition_apply_ll_kernel, _prep_args_apply_ll_kernel),
            (self.ll_w_mask_kernel, _condition_apply_ll_w_mask_kernel, _prep_args_apply_ll_w_mask_kernel)
        ]

        self.post_bp_fns = [
            (self.ll_bp_kernel, _condition_apply_ll_bp_kernel, _prep_args_apply_ll_bp_kernel),
            (self.ll_bp_w_mask_kernel1, _condition_apply_ll_bp_w_mask_kernel1, _prep_args_apply_ll_bp_w_mask_kernel1),
            (self.ll_bp_w_mask_kernel2, _condition_apply_ll_bp_w_mask_kernel2, _prep_args_apply_ll_bp_w_mask_kernel2),
            (self.ll_bp_extern_grad_kernel, _condition_apply_ll_bp_extern_grad_kernel, _prep_args_apply_ll_bp_extern_grad_kernel),
            (self.ll_bp_extern_grad_w_mask_kernel, _condition_apply_ll_bp_extern_grad_w_mask_kernel, _prep_args_apply_ll_bp_extern_grad_w_mask_kernel)
        ]

        self.sampling_fns = [
            (self.sample_kernel, _condition_sample_kernel, _prep_args_sample_kernel),
            (self.sample_set_hard_context_kernel, _condition_sample_set_hard_context_kernel, _prep_args_sample_set_hard_context_kernel)
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
                  ext_softevi_indexing: tl.constexpr, external_categorical_logps_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr):
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
        inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE, TILE_SIZE_K]

        # Ptrs pointing to external parameters
        expars_ptr = external_categorical_logps_ptr + \
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
        else:
            logZ = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

        # Compute unnormalized log-probabilities
        if compute_unnorm_logp:
            data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = mask, other = 0)
            log_in_p = tl.load(params_ptr + s_pids + data, mask = mask, other = 0.0).log()
            
            if ext_softevi_indexing:
                ex_p_ptr = external_categorical_logps_ptr + \
                    batch_offsets * (ext_num_vars * max_num_cats) + \
                    lvids * max_num_cats + \
                    data
            else:
                ex_p_ptr = external_categorical_logps_ptr + \
                    batch_offsets * (ext_num_vars * max_num_cats) + \
                    lvids * max_num_cats
            log_ex_p = tl.load(ex_p_ptr, mask = mask, other = 0.0)
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
    def ll_w_mask_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, 
                         fw_local_ids_ptr, partial_eval: tl.constexpr, layer_num_nodes: tl.constexpr, batch_size: tl.constexpr, 
                         num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                         TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, use_normalized: tl.constexpr,
                         external_categorical_logps_ptr, external_categorical_value_mask_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, 
                         max_num_cats: tl.constexpr):
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
        inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE, TILE_SIZE_K]

        # Ptrs pointing to external parameters
        expars_ptr = external_categorical_logps_ptr + \
            batch_offsets[:,None] * (ext_num_vars * max_num_cats) + \
            lvids[:,None] * max_num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:]  # [BLOCK_SIZE, TILE_SIZE_K]

        # Compute logZ
        logZ = tl.zeros([BLOCK_SIZE], dtype = tl.float32) - float("inf")
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

        # Compute unnormalized log-probabilities
        data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = mask, other = 0)
        log_in_p = tl.load(params_ptr + s_pids + data, mask = mask, other = 0.0).log()
        
        ex_p_ptr = external_categorical_logps_ptr + \
            batch_offsets * (ext_num_vars * max_num_cats) + \
            lvids * max_num_cats + \
            data
        log_ex_p = tl.load(ex_p_ptr, mask = mask, other = 0.0)

        if use_normalized:
            log_p = log_in_p + log_ex_p - logZ
        else:
            log_p = log_in_p + log_ex_p

        # Get value masks (`True` to use value, `False` to use logZ)
        ext_value_mask = tl.load(external_categorical_value_mask_ptr + batch_offsets * ext_num_vars + lvids, mask = mask, other = False) # [BLOCK_SIZE]

        val = tl.where(ext_value_mask, log_p, logZ)

        # Store the logprob
        node_offsets = local_offsets + node_offset
        tl.store(node_mars_ptr + node_offsets * batch_size + batch_offsets, val, mask = mask)

    @staticmethod
    @triton_jit
    def ll_bp_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                     metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                     batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                     BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, compute_unnorm_logp: tl.constexpr, 
                     compute_logz: tl.constexpr, external_categorical_logps_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr):
        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        if partial_eval > 0:
            offsets_n = tl.load(bk_local_ids_ptr + offsets_n, mask = mask, other = 0)

        # Get all variable ids
        vids = tl.load(vids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        lvids = tl.load(var_idmapping_ptr + vids, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask_n, other = 0).to(tl.int64) # [BLOCK_SIZE_N]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get data
        mask = mask_n[:,None] & mask_b[None,:]
        data = tl.load(data_ptr + vids[:,None] * batch_size + offsets_b[None,:], mask = mask, other = 0) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        # Load flows
        flows = tl.load(
            node_flows_ptr + (offsets_n[:,None] + node_offset) * batch_size + offsets_b[None,:], 
            mask = mask, 
            other = 0
        ) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        if logspace_flows:
            flows = flows.exp()

        if compute_unnorm_logp:
            tl.atomic_add(param_flows_ptr + s_pfids[:,None] + data, flows, mask = mask)

        else:
            # Load logZ from `node_mars`
            logZ = tl.load(
                node_mars_ptr + (offsets_n[:,None] + node_offset) * batch_size + offsets_b[None,:], 
                mask = mask, 
                other = 0
            ) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_N, TILE_SIZE_K]

            # Ptrs pointing to external parameters
            expars_ptr = external_categorical_logps_ptr + \
                offsets_b[None,:,None] * (ext_num_vars * max_num_cats) + \
                lvids[:,None,None] * max_num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,None,:]  # [BLOCK_SIZE_N, BLOCK_SIZE_B, TILE_SIZE_K]

            parflow_ptrs = param_flows_ptr + s_pfids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_N, TILE_SIZE_K]
            for i in range(K_NUM_TILES):
                cat_mask_in = mask_n[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats[:,None]) # [BLOCK_SIZE_N, TILE_SIZE_K]
                cat_mask_ex = mask[:,:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,None,:] < num_cats[:,None,None]) # [BLOCK_SIZE_N, BLOCK_SIZE_B, TILE_SIZE_K]

                inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask_in, other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
                expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask_ex, other = 0.0) # [BLOCK_SIZE_N, BLOCK_SIZE_B, TILE_SIZE_K]

                par = inpar[:,None,:].log() + expar
                parflows = flows[:,:,None] * tl.exp(par - logZ[:,:,None])

                tl.atomic_add(parflow_ptrs, tl.sum(parflows, axis = 1), mask = cat_mask_in)

                parflow_ptrs += TILE_SIZE_K

    @staticmethod
    @triton_jit
    def ll_bp_w_mask_kernel1(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                             metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                             batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                             BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, use_normalized: tl.constexpr, 
                             external_categorical_logps_ptr, external_categorical_value_mask_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr):
        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        if partial_eval > 0:
            offsets_n = tl.load(bk_local_ids_ptr + offsets_n, mask = mask, other = 0)

        # Get all variable ids
        vids = tl.load(vids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        lvids = tl.load(var_idmapping_ptr + vids, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask_n, other = 0).to(tl.int64) # [BLOCK_SIZE_N]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get data
        mask = mask_n[:,None] & mask_b[None,:]
        data = tl.load(data_ptr + vids[:,None] * batch_size + offsets_b[None,:], mask = mask, other = 0) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        # Load flows
        flows = tl.load(
            node_flows_ptr + (offsets_n[:,None] + node_offset) * batch_size + offsets_b[None,:], 
            mask = mask, 
            other = 0
        ) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        if logspace_flows:
            flows = flows.exp()

        # Load mask
        val_mask = tl.load(external_categorical_value_mask_ptr + offsets_b[None,:] * ext_num_vars + lvids[:,None], mask = mask, other = False) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        ## Handle cases where `val_mask = True` (data is provided) ##

        tl.atomic_add(param_flows_ptr + s_pfids[:,None] + data, flows, mask = (mask & val_mask))

        ## Handle cases where `val_mask = False` (data is not provided) ##

        # This is handled by `ll_bp_w_mask_kernel2`

    @staticmethod
    @triton_jit
    def ll_bp_w_mask_kernel2(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                             metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                             batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                             BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, use_normalized: tl.constexpr, 
                             external_categorical_logps_ptr, external_categorical_value_mask_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr):
        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        if partial_eval > 0:
            offsets_n = tl.load(bk_local_ids_ptr + offsets_n, mask = mask, other = 0)

        # Get all variable ids
        vids = tl.load(vids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        lvids = tl.load(var_idmapping_ptr + vids, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask_n, other = 0).to(tl.int64) # [BLOCK_SIZE_N]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get data
        mask = mask_n[:,None] & mask_b[None,:]
        data = tl.load(data_ptr + vids[:,None] * batch_size + offsets_b[None,:], mask = mask, other = 0) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        # Load flows
        flows = tl.load(
            node_flows_ptr + (offsets_n[:,None] + node_offset) * batch_size + offsets_b[None,:], 
            mask = mask, 
            other = 0
        ) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        if logspace_flows:
            flows = flows.exp()

        # Load mask
        val_mask = tl.load(external_categorical_value_mask_ptr + offsets_b[None,:] * ext_num_vars + lvids[:,None], mask = mask, other = False) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        ## Handle cases where `val_mask = True` (data is provided) ##

        # This is handled by `ll_bp_w_mask_kernel1`

        ## Handle cases where `val_mask = False` (data is not provided) ##

        # mask_n = (mask_n & (~tl.min(val_mask, axis = 1))) # Use `tl.min` to simulate `all`
        mask = (mask & (~val_mask))

        # Load logZ from `node_mars`
        logZ = tl.load(
            node_mars_ptr + (offsets_n[:,None] + node_offset) * batch_size + offsets_b[None,:], 
            mask = mask, 
            other = 0
        ) # [BLOCK_SIZE_N, BLOCK_SIZE_B]

        # Ptrs pointing to internal parameters
        inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_N, TILE_SIZE_K]

        # Ptrs pointing to external parameters
        expars_ptr = external_categorical_logps_ptr + \
            offsets_b[None,:,None] * (ext_num_vars * max_num_cats) + \
            lvids[:,None,None] * max_num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,None,:]  # [BLOCK_SIZE_N, BLOCK_SIZE_B, TILE_SIZE_K]

        parflow_ptrs = param_flows_ptr + s_pfids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_N, TILE_SIZE_K]
        for i in range(K_NUM_TILES):
            cat_mask_in = mask_n[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats[:,None]) # [BLOCK_SIZE_N, TILE_SIZE_K]
            cat_mask_ex = mask[:,:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,None,:] < num_cats[:,None,None]) # [BLOCK_SIZE_N, BLOCK_SIZE_B, TILE_SIZE_K]

            inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask_in, other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
            expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask_ex, other = -float("inf")) # [BLOCK_SIZE_N, BLOCK_SIZE_B, TILE_SIZE_K]

            par = inpar[:,None,:].log() + expar
            parflows = flows[:,:,None] * tl.exp(par - logZ[:,:,None])

            tl.atomic_add(parflow_ptrs, tl.sum(parflows, axis = 1), mask = cat_mask_in)

            parflow_ptrs += TILE_SIZE_K

    @staticmethod
    @triton_jit
    def ll_bp_extern_grad_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                                 metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                                 batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                                 BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, compute_unnorm_logp: tl.constexpr, 
                                 compute_logz: tl.constexpr, external_categorical_logps_ptr, external_categorical_logps_grad_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, 
                                 max_num_cats: tl.constexpr, use_tensor_core: tl.constexpr):
        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        if partial_eval > 0:
            offsets_n = tl.load(bk_local_ids_ptr + offsets_n, mask = mask, other = 0)
            offset_n = tl.load(bk_local_ids_ptr + pid_n * BLOCK_SIZE_N, mask = mask, other = 0)
        else:
            offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vid = tl.load(vids_ptr + offset_n) # [1]
        lvid = tl.load(var_idmapping_ptr + vid) # [1]

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + offset_n) # [1]
        num_cats = tl.load(metadata_ptr + s_mids).to(tl.int64) # [1]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get data
        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0) # [BLOCK_SIZE_B]

        # Load flows
        mask = mask_n[None,:] & mask_b[:,None]
        flows = tl.load(
            node_flows_ptr + (offsets_n[None,:] + node_offset) * batch_size + offsets_b[:,None], 
            mask = mask, 
            other = 0
        ) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        if logspace_flows:
            flows = flows.exp()

        if compute_unnorm_logp:
            expars_grad_ptr = external_categorical_logps_grad_ptr + \
                offsets_b * ext_num_vars * max_num_cats + \
                lvid * max_num_cats # [BLOCK_SIZE_B]

            tl.atomic_add(expars_grad_ptr + data, tl.sum(flows, axis = 1), mask = mask_b)

        if compute_logz:
            # Load mars from `node_mars`
            mars = tl.load(
                node_mars_ptr + (offsets_n[None,:] + node_offset) * batch_size + offsets_b[:,None], 
                mask = mask, 
                other = 0
            ) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

            if compute_unnorm_logp:
                target_expars_ptr = external_categorical_logps_ptr + \
                    offsets_b * (ext_num_vars * max_num_cats) + \
                    lvid * max_num_cats + \
                    data # [BLOCK_SIZE_B]
                target_expar = tl.load(target_expars_ptr, mask = mask_b, other = 0.0)

                target_inpars_ptr = params_ptr + s_pids[None,:] + data[:,None]
                target_inpar = tl.load(target_inpars_ptr, mask = mask, other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

                logZ = target_inpar.log() + target_expar[:,None] - mars
            else:
                logZ = mars
            
            fsublogZ = tl.log(flows) - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]
            fsublogZ_max = tl.max(fsublogZ, axis = 1)[:,None]
            fsublogZ_sub = (fsublogZ - fsublogZ_max).exp()

            # Ptrs pointing to internal parameters
            inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_N, TILE_SIZE_K]

            # Ptrs pointing to external parameters
            expars_ptr = external_categorical_logps_ptr + \
                offsets_b[:,None] * (ext_num_vars * max_num_cats) + \
                lvid * max_num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:]  # [BLOCK_SIZE_B, TILE_SIZE_K]

            expars_grad_ptr = external_categorical_logps_grad_ptr + \
                offsets_b[:,None] * ext_num_vars * max_num_cats + \
                lvid * max_num_cats + \
                tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

            for i in range(K_NUM_TILES):
                cat_mask_in = mask_n[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats) # [BLOCK_SIZE_N, TILE_SIZE_K]
                cat_mask_ex = mask_b[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats) # [BLOCK_SIZE_B, TILE_SIZE_K]

                inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask_in, other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
                expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask_ex, other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                if use_tensor_core:
                    acc = tl.dot(fsublogZ_sub, inpar).log() + fsublogZ_max
                else:
                    acc = tl.sum(fsublogZ_sub[:,:,None] * inpar[None,:,:], axis = 1).log() + fsublogZ_max

                if compute_unnorm_logp:
                    parflows = -1.0 * tl.exp(acc + expar) # [BLOCK_SIZE_B, TILE_SIZE_K]
                else:
                    parflows = tl.exp(acc + expar) # [BLOCK_SIZE_B, TILE_SIZE_K]

                tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, parflows, mask = cat_mask_ex)

    @staticmethod
    @triton_jit
    def ll_bp_extern_grad_w_mask_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                                        metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, logspace_flows: tl.constexpr, layer_num_nodes: tl.constexpr, 
                                        batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                                        BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, use_normalized: tl.constexpr, 
                                        external_categorical_logps_ptr, external_categorical_logps_grad_ptr, external_categorical_value_mask_ptr, 
                                        var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr, use_tensor_core: tl.constexpr):
        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes

        if partial_eval > 0:
            offsets_n = tl.load(bk_local_ids_ptr + offsets_n, mask = mask, other = 0)
            offset_n = tl.load(bk_local_ids_ptr + pid_n * BLOCK_SIZE_N, mask = mask, other = 0)
        else:
            offset_n = pid_n * BLOCK_SIZE_N

        # Get all variable ids
        vid = tl.load(vids_ptr + offset_n) # [1]
        lvid = tl.load(var_idmapping_ptr + vid) # [1]

        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + offset_n) # [1]
        num_cats = tl.load(metadata_ptr + s_mids).to(tl.int64) # [1]

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get start parameter flow indices
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

        # Get data
        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0) # [BLOCK_SIZE_B]

        # Load flows
        mask = mask_n[None,:] & mask_b[:,None]
        flows = tl.load(
            node_flows_ptr + (offsets_n[None,:] + node_offset) * batch_size + offsets_b[:,None], 
            mask = mask, 
            other = 0
        ) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        if logspace_flows:
            flows = flows.exp()

        # Load mask
        val_mask = tl.load(external_categorical_value_mask_ptr + offsets_b * ext_num_vars + lvid, mask = mask_b, other = False) # [BLOCK_SIZE_B]

        ## Gradients from the unnormalized part ##

        expars_grad_ptr = external_categorical_logps_grad_ptr + \
            offsets_b * ext_num_vars * max_num_cats + \
            lvid * max_num_cats # [BLOCK_SIZE_B]

        tl.atomic_add(expars_grad_ptr + data, tl.sum(flows, axis = 1), mask = (mask_b & val_mask))

        ## Gradients from the normalized part ##

        # Load mars from `node_mars`
        mars = tl.load(
            node_mars_ptr + (offsets_n[None,:] + node_offset) * batch_size + offsets_b[:,None], 
            mask = mask, 
            other = 0
        ) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Ptrs pointing to internal parameters
        inpars_ptr = params_ptr + s_pids[:,None] + tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_N, TILE_SIZE_K]

        # Ptrs pointing to external parameters
        expars_ptr = external_categorical_logps_ptr + \
            offsets_b[:,None] * (ext_num_vars * max_num_cats) + \
            lvid * max_num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:]  # [BLOCK_SIZE_B, TILE_SIZE_K]

        expars_grad_ptr = external_categorical_logps_grad_ptr + \
            offsets_b[:,None] * ext_num_vars * max_num_cats + \
            lvid * max_num_cats + \
            tl.arange(0, TILE_SIZE_K)[None,:] # [BLOCK_SIZE_B, TILE_SIZE_K]

        ## Case 1: the `val_mask = False` case ##

        logZ = mars
        fsublogZ = tl.log(flows) - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        fsublogZ_max = tl.max(fsublogZ, axis = 1)[:,None]
        fsublogZ_sub = (fsublogZ - fsublogZ_max).exp()

        for i in range(K_NUM_TILES):
            cat_mask_in = mask_n[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats) # [BLOCK_SIZE_N, TILE_SIZE_K]
            cat_mask_ex = mask_b[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats) # [BLOCK_SIZE_B, TILE_SIZE_K]

            inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask_in, other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
            expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask_ex, other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

            if use_tensor_core:
                acc = tl.dot(fsublogZ_sub, inpar).log() + fsublogZ_max
            else:
                acc = tl.sum(fsublogZ_sub[:,:,None] * inpar[None,:,:], axis = 1).log() + fsublogZ_max

            parflows = tl.exp(acc + expar) # [BLOCK_SIZE_B, TILE_SIZE_K]

            tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, parflows, mask = (cat_mask_ex & (~val_mask[:,None])))

        ## Case 2: the `use_normalized = True` and `val_mask = True` case ##

        if use_normalized:
            target_expars_ptr = external_categorical_logps_ptr + \
                offsets_b * (ext_num_vars * max_num_cats) + \
                lvid * max_num_cats + \
                data # [BLOCK_SIZE_B]
            target_expar = tl.load(target_expars_ptr, mask = mask_b, other = 0.0)

            target_inpars_ptr = params_ptr + s_pids[None,:] + data[:,None]
            target_inpar = tl.load(target_inpars_ptr, mask = mask, other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N]

            logZ = target_inpar.log() + target_expar[:,None] - mars

            fsublogZ = tl.log(flows) - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]
            fsublogZ_max = tl.max(fsublogZ, axis = 1)[:,None]
            fsublogZ_sub = (fsublogZ - fsublogZ_max).exp()

            for i in range(K_NUM_TILES):
                cat_mask_in = mask_n[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats) # [BLOCK_SIZE_N, TILE_SIZE_K]
                cat_mask_ex = mask_b[:,None] & (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[None,:] < num_cats) # [BLOCK_SIZE_B, TILE_SIZE_K]

                inpar = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = cat_mask_in, other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
                expar = tl.load(expars_ptr + i * TILE_SIZE_K, mask = cat_mask_ex, other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                if use_tensor_core:
                    acc = tl.dot(fsublogZ_sub, inpar).log() + fsublogZ_max
                else:
                    acc = tl.sum(fsublogZ_sub[:,:,None] * inpar[None,:,:], axis = 1).log() + fsublogZ_max

                parflows = -tl.exp(acc + expar) # [BLOCK_SIZE_B, TILE_SIZE_K]

                tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, parflows, mask = (cat_mask_ex & val_mask[:,None]))

    @staticmethod
    @triton_jit
    def sample_kernel(samples_ptr, params_ptr, nflow_xids_ptr, nflow_yids_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr,
                      num_activ_nodes, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, batch_size: tl.constexpr, seed, 
                      external_categorical_logps_ptr, var_idmapping_ptr, ext_num_vars: tl.constexpr, max_num_cats: tl.constexpr,
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
        expars_ptr = external_categorical_logps_ptr + \
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
    @triton_jit
    def sample_set_hard_context_kernel(samples_ptr, params_ptr, nflow_xids_ptr, nflow_yids_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr,
                                       num_activ_nodes, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, batch_size: tl.constexpr, seed,
                                       var_idmapping_ptr, ext_num_vars: tl.constexpr, num_vars: tl.constexpr, inputs_ptr, external_categorical_value_mask_ptr, 
                                       BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE) # [BLOCK_SIZE]
        mask = (offsets < batch_size * ext_num_vars)

        offsets_b = (offsets // ext_num_vars)
        offsets_v = (offsets % ext_num_vars)

        # Variable indices
        lvids = tl.load(var_idmapping_ptr + offsets_v, mask = mask, other = 0)

        # Load mask value
        val_mask = tl.load(external_categorical_value_mask_ptr + offsets_b * ext_num_vars + offsets_v, mask = mask, other = False)

        # Load data
        inputs = tl.load(inputs_ptr + offsets_b * num_vars + lvids, mask = mask, other = 0)

        # Overwrite conditioned values to `samples`
        tl.store(samples_ptr + offsets_v * batch_size + offsets_b, inputs, mask = (mask & val_mask))

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
