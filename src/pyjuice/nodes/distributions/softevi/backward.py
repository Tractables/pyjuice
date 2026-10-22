"""
The backward and the sampling of :class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical`'s
layers, the launch side: the parameter-flow and soft-evidence kernels' hooks, the dense backward's prologue and
expected-category flow (CUDA or Triton), the fold of the float32 evidence-gradient scratch into the caller's buffer,
and the sampler's hooks.
"""

from __future__ import annotations

import torch
import triton
import os

from pyjuice.utils.util import max_power_of_2_factor

from .common import _grad_scratch
from .dense_index import _build_dense_index, _dense_scratch, _dense_topk_applicable, _dense_update_pflows, \
                         _missing_mask_kwargs


class _GradCastEpilogue:
    """Fold the float32 evidence-gradient scratch back into the caller's half-precision buffer.

    Runs last in `post_bp_fns`, after every kernel that touches the gradient. Leaves the scratch zeroed
    so the next step starts clean."""

    def __getitem__(self, grid):
        def launch(**kw):
            dst, src = kw["grad_dst"], kw["grad_src"]
            dst += src.to(dst.dtype)
            src.zero_()
        return launch


def _condition_grad_cast(layer, kwargs):
    grad = kwargs.get("categorical_evidence_logp_grad", None)
    if grad is None or grad.dtype == torch.float32:
        return False
    buf = getattr(layer, "_grad_f32_buf", None)
    return buf is not None and buf.shape == grad.shape and buf.device == grad.device


def _prep_args_grad_cast(layer, kwargs):
    return dict(grad_dst = kwargs["categorical_evidence_logp_grad"],
                grad_src = layer._grad_f32_buf), (1,)


def _condition_apply_bk_params_kernel(layer, kwargs):
    # `layer.param_flows is not None` -- this kernel exists ONLY to accumulate param flows, so with no
    # buffer to accumulate into there is nothing for it to do. Declining here rather than guarding
    # inside the kernel, because unlike `bk_softevi_kernel` it has no other work to contribute.
    # A caller that asks for `backward(compute_param_flows = False)` -- which is what
    # `pyjuice.queries.base.query` does, hence every `marginal`/`conditional` query -- otherwise
    # reaches `tl.atomic_add(param_flows_ptr + ...)` with a null pointer and dies at Triton COMPILE
    # time with a bare `AttributeError("'NoneType' object has no attribute 'type'")`.
    return ("categorical_evidence_logp" in kwargs and not kwargs["dual_flow_backward"]
            and layer.param_flows is not None)


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


def _condition_bk_dense_prologue(layer, kwargs):
    return _dense_topk_applicable(layer, kwargs)


def _prep_args_bk_dense_prologue(layer, kwargs):
    """Everything the scattered kernel does except the expected-category flow phase: the observed-category
    flow, the observed-token term of the external evidence gradient, and the per-(slot, latent) ratio that
    the expected-flow kernel consumes."""
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]
    evidence = kwargs["categorical_evidence_logp"]
    assert evidence.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."
    index = _build_dense_index(layer, kwargs)

    ext_num_vars = evidence.size(1)
    num_cats = evidence.size(2)

    target_kwargs["categorical_evidence_logp_ptr"] = evidence
    target_kwargs["soft_evidence_cat_ids_ptr"] = kwargs["soft_evidence_cat_ids"]
    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping
    target_kwargs["ext_num_vars"] = ext_num_vars
    target_kwargs["num_cats"] = num_cats
    target_kwargs["num_latents"] = index["num_latents"]

    grad = kwargs.get("categorical_evidence_logp_grad", None)
    target_kwargs["categorical_evidence_logp_grad_ptr"] = _grad_scratch(layer, grad)
    target_kwargs["update_extflows"] = grad is not None
    # Without dual flows the observed-category flow belongs to `bk_params_kernel`, which still runs (see
    # `_condition_apply_bk_params_kernel`), so writing it here as well would count it twice.
    target_kwargs["update_pflows"] = _dense_update_pflows(layer, kwargs)

    target_kwargs["ratio_ptr"] = _dense_scratch(layer, ext_num_vars, batch_size, index["num_latents"])
    target_kwargs.update(_missing_mask_kwargs(kwargs))

    n_block_size = max_power_of_2_factor(layer.n_block_size)
    TILE_SIZE_K = min(16, triton.next_power_of_2(num_cats))
    BLOCK_SIZE_B = min(128, 1024 // TILE_SIZE_K, triton.next_power_of_2(batch_size))
    BLOCK_SIZE_N = min(n_block_size, max(8192 // BLOCK_SIZE_B // TILE_SIZE_K, 1))

    target_kwargs["TILE_SIZE_K"] = TILE_SIZE_K
    target_kwargs["K_NUM_TILES"] = triton.cdiv(num_cats, TILE_SIZE_K)
    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    return target_kwargs, grid


class _DenseDenomDispatch:
    """Routes the expected-category flow phase to the CUDA kernel when enabled, else Triton.

    `post_bp_fns` entries are launched as ``kernel[grid](**kwargs)``, so this mimics that protocol: the
    CUDA path ignores the Triton launch geometry and reads what it needs out of the keyword arguments.
    Falls back transparently if the extension will not compile (no nvcc / no ninja), so it is never
    required for correctness.

    :note: ON by default (disable with PYJUICE_SOFTEVI_DENSE_CUDA=0), and it wins for exactly one
           reason. This phase is dominated by the `param_flows` read-modify-write (0.75 of 1.68 ms in
           an ablation, against a 0.62 ms traffic floor). Expressed as `atomicAdd` with the result
           unused, that lowers to RED.E.ADD.F32 -- the add happens in L2 and nothing returns to the SM,
           halving the SM<->L2 traffic -- which takes the kernel 1.66 -> 1.37 ms. Triton cannot express
           it: `tl.atomic_add` there is 1.5x SLOWER than load-add-store. Net on the input-layer
           backward: ~1.85 ms CUDA vs ~2.09 ms Triton. Phase-1 output is bit-identical between the two.
           Everything else tried made no difference (float4 ratio loads, shared-memory staging of the
           ratio slice, streaming cache hints, thread counts 64-1024).
    """

    def __init__(self, triton_kernel):
        self.triton_kernel = triton_kernel
        self._use_cuda = None

    def _cuda_ok(self):
        if self._use_cuda is None:
            if os.environ.get("PYJUICE_SOFTEVI_DENSE_CUDA", "1") == "0":
                self._use_cuda = False
            else:
                try:
                    from ..c_kernels import dense_expected_flow_available
                    self._use_cuda = dense_expected_flow_available()
                except Exception:
                    self._use_cuda = False
        return self._use_cuda

    # Launch geometry that only the CUDA kernel understands. `_prep_args_bk_dense_denom` always puts
    # these in `target_kwargs`, so the Triton fallback has to drop them -- Triton rejects unknown
    # keywords outright (`KeyError: Keyword argument CUDA_TL was specified but unrecognised`).
    _CUDA_ONLY_KWARGS = ("CUDA_TL", "CUDA_THREADS")

    # Per-shard tensors and dimensions; everything else in `target_kwargs` is common to all shards.
    _SHARD_KWARGS = (("uniq_ptr", "uniq"), ("ref_slot_ptr", "ref_slot"), ("ref_pt_ptr", "ref_pt"),
                     ("ref_goff_ptr", "ref_goff"), ("ref_cnt_ptr", "ref_cnt"),
                     ("num_uniq_ptr", "num_uniq"), ("UNIQ_STRIDE", "Umax"), ("MAX_REFS", "max_refs"))

    def __getitem__(self, grid):
        # The shards partition the lists, and every update is an accumulation (atomic wherever a category
        # can have more than one list), so the shards may run in any order.
        def launch(**kw):
            shards = kw.pop("shards")
            for sh in shards:
                skw = dict(kw)
                for key, field in self._SHARD_KWARGS:
                    skw[key] = sh[field]
                sgrid = (grid[0], triton.cdiv(sh["Umax"], kw["BLOCK_C"]), grid[2])

                if not self._cuda_ok():
                    for k in self._CUDA_ONLY_KWARGS:
                        skw.pop(k, None)
                    self.triton_kernel[sgrid](**skw)
                    continue

                from ..c_kernels import dense_expected_flow
                dense_expected_flow(
                    skw["params_ptr"], skw["param_flows_ptr"] if skw["update_pflows"] else None, skw["ratio_ptr"],
                    skw["uniq_ptr"], skw["ref_slot_ptr"], skw["ref_pt_ptr"], skw["ref_goff_ptr"],
                    skw["ref_cnt_ptr"], skw["num_uniq_ptr"], skw["pf_base_ptr"], skw["p_base_ptr"],
                    skw["categorical_evidence_logp_grad_ptr"] if skw["update_extflows"] else None,
                    skw["num_latents"], skw["tot_num_cats"], skw["UNIQ_STRIDE"], skw["MAX_REFS"],
                    sgrid[2], skw.get("CUDA_THREADS", 256), skw["ratio_ptr"].size(0),
                    skw.get("CUDA_TL", 32))

        return launch


def _condition_bk_dense_denom(layer, kwargs):
    return _dense_topk_applicable(layer, kwargs)


def _prep_args_bk_dense_denom(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]
    evidence = kwargs["categorical_evidence_logp"]
    assert evidence.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."
    index = _build_dense_index(layer, kwargs)

    num_latents = index["num_latents"]

    target_kwargs["shards"] = index["shards"]
    target_kwargs["pf_base_ptr"] = index["pf_base"]
    target_kwargs["p_base_ptr"] = index["p_base"]
    target_kwargs["ratio_ptr"] = _dense_scratch(layer, evidence.size(1), batch_size, num_latents)

    grad = kwargs.get("categorical_evidence_logp_grad", None)
    target_kwargs["categorical_evidence_logp_grad_ptr"] = _grad_scratch(layer, grad)
    target_kwargs["update_extflows"] = grad is not None
    target_kwargs["update_pflows"] = _dense_update_pflows(layer, kwargs)
    # The CUDA kernel's update is always atomic; the Triton one only needs to be when lists were cut.
    target_kwargs["PF_ATOMIC"] = index["split"]

    target_kwargs["num_latents"] = num_latents
    target_kwargs["tot_num_cats"] = layer.nodes[0].dist.num_cats
    target_kwargs["pf_row_stride"] = 2 * layer.nodes[0].dist.num_cats
    # CUDA-path launch geometry, kept separate from the Triton tile so each can be tuned on its own.
    # TL = latents per thread: swept 4/8/16/32/64 -> 2.17/1.83/1.72/1.58/1.67 ms (32 is the optimum;
    # 64 regresses on register pressure). Threads 64-256 are within noise, 256 marginally best.
    target_kwargs["CUDA_TL"] = 32
    target_kwargs["CUDA_THREADS"] = 256
    # Swept on the CoDD config with the evidence-gradient term folded in; the optimum differs from the
    # fold-off optimum (the fold adds live temporaries), so re-sweep if that term ever moves out.
    target_kwargs["BLOCK_L"] = 64
    target_kwargs["BLOCK_C"] = 128

    # `UNIQ_STRIDE` and `MAX_REFS` vary per shard, so the dispatch fills them in per launch; only the
    # category-axis extent of the grid changes with them.
    grid = (triton.cdiv(num_latents, target_kwargs["BLOCK_L"]), 1, index["num_blocks"])

    return target_kwargs, grid


def _condition_apply_bk_softevi_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and \
        ("categorical_evidence_logp_grad" in kwargs or kwargs["dual_flow_backward"]) and \
        not _dense_topk_applicable(layer, kwargs)


def _prep_args_apply_bk_softevi_kernel(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]

    categorical_evidence_logp = kwargs["categorical_evidence_logp"]
    assert categorical_evidence_logp.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."

    ext_num_vars = categorical_evidence_logp.size(1)
    target_kwargs["ext_num_vars"] = ext_num_vars

    num_cats = categorical_evidence_logp.size(2)
    target_kwargs["num_cats"] = num_cats

    # Full (distribution) num_cats = the width of one param-flow phase. With top-k soft
    # evidence, `num_cats` above is the top-k tile width (< V_full); the F- (denominator)
    # phase of the dual-flow buffer starts at offset V_full, NOT at the top-k width.
    target_kwargs["tot_num_cats"] = layer.nodes[0].dist.num_cats

    if "categorical_evidence_logp_grad" in kwargs:
        categorical_evidence_logp_grad = kwargs["categorical_evidence_logp_grad"]
        assert categorical_evidence_logp_grad.size(0) == batch_size
        assert categorical_evidence_logp_grad.size(1) == ext_num_vars
        assert categorical_evidence_logp_grad.size(2) == num_cats
    else:
        categorical_evidence_logp_grad = None

    target_kwargs["categorical_evidence_logp_ptr"] = categorical_evidence_logp
    target_kwargs["categorical_evidence_logp_grad_ptr"] = _grad_scratch(layer, categorical_evidence_logp_grad)
    target_kwargs["var_idmapping_ptr"] = layer.var_idmapping

    # (Optional) soft_evidence_cat_ids
    if "soft_evidence_cat_ids" in kwargs and kwargs["soft_evidence_cat_ids"] is not None:
        soft_evidence_cat_ids = kwargs["soft_evidence_cat_ids"]
        assert categorical_evidence_logp.size() == soft_evidence_cat_ids.size()

        target_kwargs["soft_evidence_cat_ids_ptr"] = soft_evidence_cat_ids
        target_kwargs["has_ext_ids"] = True
    else:
        target_kwargs["soft_evidence_cat_ids_ptr"] = None
        target_kwargs["has_ext_ids"] = False

    # Prepare block/grid size
    n_block_size = max_power_of_2_factor(layer.n_block_size)

    # prepare BLOCK_SIZE and TILE_SIZE_K
    if target_kwargs["has_ext_ids"]:
        TILE_SIZE_K = min(16, triton.next_power_of_2(num_cats))
        K_NUM_TILES = triton.cdiv(num_cats, TILE_SIZE_K)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        BLOCK_SIZE_B = min(128, 1024 // TILE_SIZE_K, BATCH_SIZE_NP2)
        # A LARGER [B, N, K] tile wins here: every CTA pays a full K_NUM_TILES prologue (the local
        # normalizer + the observed-token search) before the expected-category flow phase, so shrinking
        # BLOCK_SIZE_N multiplies that fixed cost by the extra CTA count -- more than it saves on the
        # scatter. Budget 8192 tile elements rather than 4096. Measured on the CoDD config (T=32,
        # L=1024, C=126464, top-k 1024, B=8) with sorted candidate ids: backward 20.0 -> 14.8 ms.
        BLOCK_SIZE_N = min(n_block_size, max(8192 // BLOCK_SIZE_B // TILE_SIZE_K, 1))
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

    # Whether to update `pflow` and `extflow`.
    #
    # `layer.param_flows is not None` is NOT redundant with `dual_flow_backward`. A caller that asks
    # for `backward(compute_param_flows = False)` -- which is exactly what `pyjuice.queries.base.query`
    # does, and therefore every `marginal`/`conditional` query -- gets no param-flow buffer, but
    # `dual_flow_backward` is the DISTRIBUTION's own flag and knows nothing about that. Without this
    # guard the kernel still takes its `if update_pflows` branch and dereferences a null
    # `param_flows_ptr`, which fails at Triton COMPILE time with a bare
    # `AttributeError("'NoneType' object has no attribute 'type'")` -- no mention of param flows, and
    # nothing to connect it to the flag the caller actually passed.
    target_kwargs["update_pflows"] = kwargs["dual_flow_backward"] and (layer.param_flows is not None)
    target_kwargs["update_extflows"] = ("categorical_evidence_logp_grad" in kwargs)

    target_kwargs.update(_missing_mask_kwargs(kwargs))

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
