"""
The forward of :class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical`'s layers, the launch
side: the Triton kernel's hooks; the CUDA forward, in gather form or index-driven over :mod:`.dense_index`'s index,
autotuned; the dense (full-vocabulary) forward as one GEMM; and the value-mask kernel's hooks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import os

from pyjuice.utils.util import max_power_of_2_factor

from .common import _evidence_cache_hit, _evidence_cache_key, _row_index
from .dense_index import _build_dense_index, _fw_smem_cap, _FW_TL_CANDIDATES, _fw_use_dense, _params_exceed_l2


def _fw_transposed_args(layer, target_kwargs, num_cats):
    """
    Point the forward at the transposed emission table when `pyjuice.fast_inference` built one.

    Only for the top-k path: it is the `cat_ids` gather that suffers, because there the innermost
    tile axis is the NODE axis and two consecutive nodes are `num_cats` floats apart. The dense
    branch streams whole rows and is a different regime, so it keeps the ordinary layout.

    Returns whether the transposed path was taken; the caller re-tiles if so, because the tile that
    is right for a scattered gather is not the one that is right for a coalesced one.
    """
    table = getattr(layer, "_fast_inference_params", None)
    if table is None or not target_kwargs["has_ext_ids"]:
        # Unused under `TRANSPOSED = 0`, but Triton still needs real pointers for the arguments.
        target_kwargs["params_t_ptr"] = layer.params
        target_kwargs["row_idx_ptr"] = layer.s_pids
        target_kwargs["num_rows"] = 1
        target_kwargs["TRANSPOSED"] = 0
        return False

    target_kwargs["params_t_ptr"] = table
    target_kwargs["row_idx_ptr"] = _row_index(layer, num_cats)
    target_kwargs["num_rows"] = layer.params.numel() // num_cats
    target_kwargs["TRANSPOSED"] = 1
    return True


def _fw_cuda_structural(layer, kwargs):
    """The structural half of `_fw_cuda_applicable`: top-k evidence, no value mask, no partial
    evaluation, extension built."""
    if os.environ.get("PYJUICE_SOFTEVI_FW_CUDA", "1") == "0":
        return False
    if kwargs.get("soft_evidence_cat_ids", None) is None or "categorical_evidence_logp" not in kwargs:
        return False
    if kwargs.get("soft_evidence_value_mask", None) is not None:
        return False
    try:
        from ..c_kernels import dense_expected_flow_available
        return dense_expected_flow_available()
    except Exception:
        return False


def _fw_cuda_applicable(layer, kwargs):
    """Whether to take a CUDA forward at all.

    Structural requirements first (top-k evidence, no value mask, extension built), then the same
    "is it big enough" test the backward uses. Both CUDA forwards were tuned on a configuration whose
    parameter table is far larger than L2 -- the gather form's 512-thread cliff and
    its streaming loads are both about DRAM behaviour. When the table is L2-resident none of that
    applies and the well-tuned Triton kernel is the safer choice, so fall back rather than extrapolate
    from a regime that was never measured.
    """
    if not _fw_cuda_structural(layer, kwargs):
        return False

    if _fw_use_dense(layer, kwargs):
        return True

    # Otherwise only the gather form is available, and it is NOT auto-selected: it has never been
    # measured to be the best choice anywhere. Where the dense form applies (tied parameters, table
    # larger than L2) dense beats it 0.95 vs 2.35 ms; where the dense form declines -- untied
    # parameters, so 32768 distinct parameter rows instead of 1024 -- it LOSES to Triton, 17.2 vs
    # 15.0 ms, because its whole design assumes few, heavily reused rows. Triton is the safe default
    # in that regime. Kept reachable for experimentation on shapes not covered here.
    return os.environ.get("PYJUICE_SOFTEVI_FW_GATHER", "0") == "1" \
        and _params_exceed_l2(layer, kwargs)


def _fw_linear_evidence(layer, kwargs):
    """exp(categorical_evidence_logp), cached per step: the gather form accumulates the normalizer in
    linear space and would otherwise redo this exp 268M times inside the kernel."""
    evidence = kwargs["categorical_evidence_logp"]
    key = _evidence_cache_key(evidence)
    cached = getattr(layer, "_fw_pt_cache", None)
    if cached is not None and _evidence_cache_hit(cached[0], (evidence,)):
        return cached[1]
    # The CUDA kernels read this as float32, and half-precision evidence is a reasonable thing for a
    # caller to hand over (it comes straight off a language model), so widen rather than reject. `.float()`
    # is a no-op when the evidence is already float32.
    pt = evidence.float().exp().contiguous()
    layer._fw_pt_cache = (key, pt)
    return pt


def _fw_scratch(layer, num_slots, num_latents):
    """[num_slots, num_latents] accumulator for the local normalizer (~1 MB)."""
    buf = getattr(layer, "_fw_z_buf", None)
    if buf is None or tuple(buf.shape) != (num_slots, num_latents) or buf.device != layer.params.device:
        buf = torch.zeros([num_slots, num_latents], dtype = torch.float32, device = layer.params.device)
        layer._fw_z_buf = buf
    return buf


def _fw_observed_evidence(layer, kwargs):
    """log p_theta at the observed token, per (variable, batch) -> [ext_num_vars * batch_size].

    Done once per (variable, batch) rather than per node, which is why it is far cheaper here than the
    per-node candidate re-scan the Triton kernel performs."""
    evidence = kwargs["categorical_evidence_logp"]
    cat_ids = kwargs["soft_evidence_cat_ids"]
    B, V, K = cat_ids.shape

    data = kwargs["_fw_data"].view(-1, B)                      # [pc_num_vars, batch]
    gv = layer.layer_var_list
    obs = torch.empty([V, B], dtype = data.dtype, device = data.device)
    obs[layer.var_idmapping[gv]] = data[gv]

    match = (cat_ids == obs.permute(1, 0).unsqueeze(2))         # [B, V, K]
    kstar = match.float().argmax(dim = 2, keepdim = True)
    lex = torch.gather(evidence, 2, kstar).squeeze(2)           # [B, V]
    # `argmax` yields 0 when the observed token is absent from the candidate set, which would silently
    # pick the wrong entry; the Triton kernel leaves the term at -inf in that case, so match it.
    lex = torch.where(match.any(dim = 2), lex, torch.full_like(lex, -float("inf")))
    # float32 regardless of the evidence dtype: the epilogue reads this as float (see `_fw_linear_evidence`)
    return lex.float().permute(1, 0).contiguous().view(-1)


def _condition_fw_cuda_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and _fw_cuda_applicable(layer, kwargs)


def _prep_args_fw_cuda_kernel(layer, kwargs):
    evidence = kwargs["categorical_evidence_logp"]
    batch_size = kwargs["batch_size"]
    assert evidence.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."

    if not _fw_use_dense(layer, kwargs):
        return dict(use_dense = False,
                    pt_ptr = _fw_linear_evidence(layer, kwargs),
                    cat_ids_ptr = kwargs["soft_evidence_cat_ids"],
                    var_idmapping_ptr = layer.var_idmapping,
                    ext_num_vars = evidence.size(1),
                    num_cats = evidence.size(2),
                    fw_unroll = 4), (1,)

    return _fw_dense_target(layer, kwargs, _build_dense_index(layer, kwargs)), (1,)


def _fw_dense_target(layer, kwargs, index):
    """The dense forward's launch arguments for a given `index` (also used to time it, see
    `_tune_dense_choice`)."""
    num_latents = index["num_latents"]
    num_slots = kwargs["categorical_evidence_logp"].size(1) * kwargs["batch_size"]

    return dict(use_dense = True,
                Z_ptr = _fw_scratch(layer, num_slots, num_latents),
                log_ex_p_ptr = _fw_observed_evidence(layer, kwargs),
                var_idmapping_ptr = layer.var_idmapping,
                index = index,
                num_slots = num_slots,
                num_latents = num_latents,
                # `FW_TL` and the swizzle are chosen by `_fw_autotune`, once per shape.
                _fw_layer = layer,
                FW_THREADS = 256, FW_CAT_BLOCKS = 64)


def _fw_launch_dense(kw, tl, swizzle):
    """One forward normalizer pass: every shard, then the epilogue once."""
    from ..c_kernels import softevi_forward_dense
    ix = kw["index"]
    shards = ix["shards"]
    for si, sh in enumerate(shards):
        # Every block zeroes and then flushes a [num_slots, TL] shared tile regardless of how many
        # categories it gets, so a shard holding only the few hundred long lists must not be launched
        # over the full category grid -- that flush would cost more than the work it saves.
        blocks = max(1, min(kw["FW_CAT_BLOCKS"], -(-sh["Umax"] // kw["FW_THREADS"])))
        softevi_forward_dense(
            kw["params_ptr"], kw["node_mars_ptr"], kw["Z_ptr"], kw["log_ex_p_ptr"],
            kw["data_ptr"], kw["vids_ptr"], kw["s_pids_ptr"], kw["nids_ptr"],
            kw["var_idmapping_ptr"], sh["uniq"], sh["ref_slot"], sh["ref_pt"], sh["ref_cnt"],
            sh["num_uniq"], ix["p_base"], kw["num_latents"], sh["Umax"], sh["max_refs"],
            kw["num_slots"], ix["num_blocks"], kw["layer_num_nodes"], kw["batch_size"],
            kw["node_offset"], tl, kw["FW_THREADS"], blocks,
            si == 0, si == len(shards) - 1, swizzle)


def _fw_autotune(kw):
    """Pick (latents-per-thread, XOR swizzle) for the normalizer by timing them once per shape.

    These two interact and neither has a threshold that survived contact with more than a couple of
    configurations. A larger `TL` amortizes the per-reference index loads but grows the shared tile; the
    swizzle removes a shared-memory bank conflict (`s * TL + t` reaches only 32/TL banks, so 32 distinct
    slots collapse onto 8 at TL=4) that is worth 1.95x on this kernel at a 4 KB tile and is a wash by
    16 KB. Hand-picking a cutoff from three measured points produced a few-percent regression at the
    fourth, so measure instead.

    Runs ONCE per (shape, shard layout) and is cached on the layer; every later step reads the cache.
    Timing repeats the real launch, which is idempotent -- `Z` is re-zeroed by the first shard of each
    pass and the epilogue recomputes `node_mars` from it -- so the state left behind is the correct one
    either way."""
    ix = kw["index"]
    shards = ix["shards"]
    # Bucket `Umax` to a power of two: it drifts a little every step with the candidate data, and
    # re-tuning on each drift would cost far more than it could recover.
    key = (kw["num_slots"], kw["num_latents"], ix["num_blocks"],
           tuple(s["max_refs"] for s in shards),
           tuple(1 << max(0, s["Umax"] - 1).bit_length() for s in shards))

    layer = kw["_fw_layer"]
    cache = getattr(layer, "_fw_tune_cache", None)
    if cache is None:
        cache = layer._fw_tune_cache = {}
    hit = cache.get(key)
    if hit is not None:
        return hit

    cands = [(tl, swz) for tl in _FW_TL_CANDIDATES for swz in (0, 1)
             if kw["num_slots"] * tl * 4 <= _fw_smem_cap(layer)]
    assert cands, "the dense forward was taken with no `Zs` tile that fits (see `_fw_dense_fits`)"
    best, best_t = None, None
    for tl, swz in cands:
        for _ in range(2):                                   # warm up this instantiation
            _fw_launch_dense(kw, tl, swz)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing = True)
        t1 = torch.cuda.Event(enable_timing = True)
        t0.record()
        for _ in range(3):
            _fw_launch_dense(kw, tl, swz)
        t1.record()
        torch.cuda.synchronize()
        el = t0.elapsed_time(t1)
        if best_t is None or el < best_t:
            best, best_t = (tl, swz), el

    cache[key] = best
    return best


class _FwCudaDispatch:
    """Adapter so the CUDA forward can live in `post_fw_fns`, which launches ``kernel[grid](**kwargs)``.

    Two implementations, chosen by `_fw_use_dense`:

      * INDEX-DRIVEN (preferred). Inverts (variable, batch, candidate) -> category and walks
        (latent x category), so `params` is read ONCE, coalesced, at each referenced position: ~113M
        reads rather than the 268M SCATTERED ones the direct form does. Needs tied parameters to pay
        off, exactly like the backward. The normalizer kernel measures 0.86 ms against a 0.31 ms floor
        for the parameter read alone, versus 2.90 ms for the whole Triton forward.
      * GATHER. One thread per (node, batch) walking its own candidate list; applies to any top-k
        configuration. 2.35 ms versus Triton's 2.90 ms on the tied config, and its speed hinges on two
        things found by benchmarking the gather in isolation against an equivalent Triton one: 512
        threads per block (at 256 it runs 7.4 ms -- a 3x cliff) and `__ldcs` for the parameter read.
        NOT auto-selected -- see `_fw_cuda_applicable` -- because it is never the best option in any
        regime measured: dense wins where dense applies, and Triton wins where it does not.
    """

    def __getitem__(self, grid):
        def launch(**kw):
            if kw["use_dense"]:
                tl, swz = _fw_autotune(kw)
                _fw_launch_dense(kw, tl, swz)
            else:
                from ..c_kernels import softevi_forward
                softevi_forward(
                    kw["params_ptr"], kw["node_mars_ptr"], kw["data_ptr"], kw["vids_ptr"],
                    kw["s_pids_ptr"], kw["var_idmapping_ptr"], kw["pt_ptr"], kw["cat_ids_ptr"],
                    kw["layer_num_nodes"], kw["batch_size"], kw["node_offset"],
                    kw["num_cats"], kw["ext_num_vars"], kw["fw_unroll"])
        return launch


def _condition_apply_fw_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and \
        kwargs.get("soft_evidence_value_mask", None) is None and \
        not _fw_cuda_applicable(layer, kwargs) and \
        not _fw_gemm_applicable(layer, kwargs)


def _prep_args_apply_fw_kernel(layer, kwargs):
    target_kwargs = dict()

    batch_size = kwargs["batch_size"]

    categorical_evidence_logp = kwargs["categorical_evidence_logp"]
    assert categorical_evidence_logp.size(0) == batch_size, "Batch size doesn't match in `categorical_evidence_logp`."

    # SAFETY NET for batch-broadcast evidence. The GEMM path (see `_fw_gemm_applicable`) wants callers
    # to hand it `evi.expand(B, -1, -1)` UNMATERIALISED, because batch stride 0 is how it detects that
    # every row shares one evidence tensor. These kernels cannot take that: they address it
    # `ptr + b * (ext_num_vars * num_cats)`, so on a stride-0 tensor every row past the first reads
    # beyond the single row that actually exists. Materialise here rather than forbid the caller --
    # the copy only happens when the fast path declined, which is exactly when it is needed.
    if batch_size > 1 and categorical_evidence_logp.stride(0) == 0:
        categorical_evidence_logp = categorical_evidence_logp.contiguous()
        kwargs["categorical_evidence_logp"] = categorical_evidence_logp

    ext_num_vars = categorical_evidence_logp.size(1)
    target_kwargs["ext_num_vars"] = ext_num_vars

    num_cats = categorical_evidence_logp.size(2)
    for ns in layer.nodes:
        assert num_cats <= ns.dist.num_cats
    target_kwargs["num_cats"] = num_cats

    target_kwargs["categorical_evidence_logp_ptr"] = categorical_evidence_logp
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
        # :note: the backward's expected-category flow phase wants a bigger tile (see
        #        `_prep_args_apply_bk_softevi_kernel`), but the forward measured best at this
        #        4096-element budget -- do not "fix" it to match.
        BLOCK_SIZE_N = min(n_block_size, max(4096 // BLOCK_SIZE_B // TILE_SIZE_K, 1))
    else:
        TILE_SIZE_K = min(64, triton.next_power_of_2(num_cats))
        K_NUM_TILES = triton.cdiv(num_cats, TILE_SIZE_K)
        BATCH_SIZE_NP2 = triton.next_power_of_2(batch_size)
        BLOCK_SIZE_B = min(128, 2048 // TILE_SIZE_K, BATCH_SIZE_NP2)
        BLOCK_SIZE_N = max(min(n_block_size, 2048 // TILE_SIZE_K, 2048 // BLOCK_SIZE_B), 1)

    use_tensor_core = (TILE_SIZE_K >= 16) and (BLOCK_SIZE_B >= 16) and (BLOCK_SIZE_N >= 16) and not target_kwargs["has_ext_ids"]

    if _fw_transposed_args(layer, target_kwargs, num_cats = layer.dist.num_cats):
        # RE-TILE. The node axis is innermost, so under the ordinary layout consecutive nodes are
        # `num_cats` floats apart, and the tile shape then barely matters -- MEASURED flat across all
        # 16 shapes tried, because every useful float costs its own 32-byte sector regardless.
        # Transposed they are adjacent, and the shape matters a great deal: the same loop went
        # 1.77 ms at BLOCK_SIZE_N = 256 (what the scattered layout wants) to 0.14 ms at 32 -- 19.8x
        # over the ordinary path, against the 1.68x the untuned tile gives. Shipping the transpose
        # without re-tiling would capture under a tenth of it.
        BLOCK_SIZE_N = min(n_block_size, 32)

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    target_kwargs["TILE_SIZE_K"] = TILE_SIZE_K
    target_kwargs["K_NUM_TILES"] = K_NUM_TILES
    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N
    target_kwargs["use_tensor_core"] = use_tensor_core

    return target_kwargs, grid


# ---------------------------------------------------------------------------------------------
#  Dense (full-vocab) forward as a single GEMM
# ---------------------------------------------------------------------------------------------
#
# For a MARGINALISED variable the forward wants, over every emission row `r`,
#
#     logZ[v, r] = logsumexp_c ( log p[r, c] + evi[v, c] )
#                = log( sum_c p[r, c] * exp(evi[v, c] - m_v) ) + m_v
#
# which is a matrix product between the emission table and the evidence of the marginalised
# variables. The Triton kernel does NOT compute it as one: it contracts over categories with the
# BATCH on the other axis, so the 494 MiB table is re-read once per marginalised VARIABLE. Measured
# on the CoDD circuit: `1.01 ms + 0.18 ms * n_masked`, and 0.18 ms is exactly one L2-served pass over
# the table.
#
# Training never noticed, because at batch >= 16 the batch axis supplies that reuse. Decode runs at
# batch 1 with 16-32 marginalised positions and gets none -- which is why the PC costs ~25-45% of a
# dLLM step at decode against ~3% in training. Same kernel; wrong axis.
#
# As one GEMM the table is read ONCE, so cost is FLAT in `n_masked` and lands on the memory roofline:
# 0.66 ms against a 0.70 ms streaming read of the same tensor.
#
# Two things that were measured, not assumed:
#   * ORIENTATION. `Pm @ El.t()` -> [n_rows, n_pairs] is 1.4x faster than `El @ Pm.t()` and is the one
#     on the floor, because it walks `params` in its natural row-major order.
#   * NOT tf32: 0.03 ms for 8e-4 of relative error. See [[tf32-fork-nondeterminism]].


# Below this the Triton kernel wins and the GEMM is a REGRESSION. 4096 was a guess and it was wrong:
# measured against the Triton dense forward on this GPU, batch 1, two masked variables --
#
#     num_cats    32 nodes   128 nodes   512 nodes
#        4096       0.49x      0.60x       0.56x      <- GEMM loses everywhere
#        8192       0.76x      1.30x       0.89x      <- near-tie, and not monotone in node count
#       16384       0.76x      1.01x       1.79x      <- near-tie
#       65536       5.71x          -           -
#      126464    4-7x (the CoDD circuit, 32768 nodes)
#
# so the threshold sits above the near-tie band rather than inside it. The GEMM only pays once the
# emission-table read dominates, and below that its fixed costs -- the max, the exp, the scatter back
# into node_mars -- are the whole of it.
#
# DELIBERATELY A FIXED THRESHOLD, NOT AUTOTUNED, even though pyjuice has `c.autotune` and the softevi
# CUDA fork uses it for its own launch config. The two paths here are not numerically identical
# (3e-7 to 9e-7 relative, growing with num_cats), so a timing-chosen fork would make the ANSWER depend
# on which path happened to be faster on that run -- exactly the defect recorded in
# `tf32-fork-nondeterminism`, where a fork picked by measurement silently changed precision and the
# failure rate tracked machine load. Autotune launch configs, which are bit-equivalent; do not
# autotune a choice between paths that disagree.
_GEMM_MIN_CATS = 32768


def _fw_gemm_applicable(layer, kwargs):
    """Whether the dense forward can be served by the GEMM path.

    Deliberately narrow: it has to reproduce exactly what the Triton kernel writes into `node_mars`,
    so every layout assumption it makes is checked rather than hoped for.
    """
    if kwargs.get("soft_evidence_value_mask", None) is None:
        # SCOPED TO THE GENERATION PATH ON PURPOSE. Without a value mask every variable is observed,
        # there is no logZ to amortise, and the ordinary likelihood forward -- the one training runs --
        # keeps its Triton kernel untouched. This is a decode optimisation; it should not be able to
        # move a training step at all.
        return False
    if kwargs.get("soft_evidence_cat_ids", None) is not None:
        return False                                     # top-k: no dense table pass to amortise
    if kwargs.get("categorical_evidence_logp", None) is None:
        return False
    # NO `missing_mask` CHECK HERE, deliberately. An earlier version had one and it was DEAD:
    # `missing_mask` is a named parameter of `InputLayer.forward`, not part of `**kwargs`, so
    # `kwargs.get("missing_mask")` is always None and the guard never fired. It is also unnecessary --
    # `_fw_missing_mask_kernel` runs AFTER the `post_fw_fns` loop and overwrites `node_mars` at
    # marginalised positions whatever wrote them first. VERIFIED: a forward with a 1-D or 2-D
    # `missing_mask` matches the Triton path to 1.5e-07 relative. Do not re-add the guard believing it
    # protects something; if a real one is ever needed it has to read the named argument.
    num_cats = getattr(layer.dist, "num_cats", None)
    if num_cats is None or num_cats < _GEMM_MIN_CATS:
        return False
    # `get_signature` excludes num_cats, so one layer can mix them; the table view would be wrong.
    for ns in layer.nodes:
        if getattr(ns.dist, "num_cats", num_cats) != num_cats:
            return False
    params = layer.params
    if params is None or params.numel() % num_cats != 0:
        return False
    if layer.num_vars_per_node != 1:
        return False
    cache = getattr(layer, "_gemm_rows_ok", None)
    if cache is None:
        cache = bool((layer.s_pids % num_cats == 0).all())   # else s_pids//num_cats is not a row id
        layer._gemm_rows_ok = cache
    return cache


def _gemm_emission_table(layer, params, n_rows, num_cats):
    """The emission table as the GEMM's left operand -- in bf16 when the scope asked for it.

    This forward reads the whole table once and is AT the memory roofline in fp32 (0.73 ms against a
    0.70 ms `params.sum()` on the same tensor). At the roofline the only remaining lever is to read
    fewer BYTES, which is what bf16 buys: 0.30 ms, i.e. below the fp32 streaming floor, for a maximum
    relative error of 8.9e-03 on logZ.

    That is an ACCURACY trade, not a memory one, so it is gated on `allow_bf16_params` and is OFF
    unless a caller explicitly asks. The copy is built lazily, cached on the layer, and freed by
    exactly the same hooks as the transposed table -- including every in-repo path that writes
    `params`, since a stale copy would be silently wrong.
    """
    from pyjuice.utils.fast_inference import bf16_params_allowed, register_layer

    if not bf16_params_allowed():
        return params.view(n_rows, num_cats)

    cached = getattr(layer, "_gemm_bf16_params", None)
    if cached is not None and cached.shape == (n_rows, num_cats):
        return cached

    cached = params.view(n_rows, num_cats).to(torch.bfloat16)
    layer._gemm_bf16_params = cached
    register_layer(layer)          # so scope exit frees it
    return cached


def _fw_dense_gemm(layer, kwargs, kw):
    """Write this layer's `node_mars` from ONE GEMM plus one gather.

    NO host-side branching anywhere in here. An earlier version asked `bool(need.any())` and
    `bool(sel.all())` to skip halves it did not need; each of those is a device-to-host SYNC, and on a
    loaded GPU three syncs per forward cost more than the work they save -- the profiler read 0.92 ms
    of GPU work against 15 ms of wall. So both halves are computed unconditionally and combined with
    `torch.where`.

    The logZ is taken over EVERY (batch, variable) pair rather than only the marginalised ones. That
    is not waste: the GEMM is bandwidth-bound on reading `params` once, and the pair count only widens
    the skinny side. It costs the same and removes the last reason to look at the mask on the host.
    """
    import torch as _torch

    num_cats = layer.dist.num_cats
    evi = kwargs["categorical_evidence_logp"]                    # [B, ext_num_vars, num_cats]
    batch_size = kw["batch_size"]
    # Addressed `data_ptr + vid * batch_size + b`, i.e. [num_vars, B], but it arrives FLAT.
    data = kw["data_ptr"]
    data = data.view(-1, batch_size) if data.dim() == 1 else data
    node_mars = kw["node_mars_ptr"]
    vmask = kwargs["soft_evidence_value_mask"]                    # [B, ext_num_vars]; see the gate

    sid, eid = layer._output_ind_range
    params = layer.params
    n_rows = params.numel() // num_cats
    Pm = _gemm_emission_table(layer, params, n_rows, num_cats)
    n_ext = evi.size(1)

    vids = layer.vids[:, 0]                                      # global variable id per node
    rows = layer.s_pids // num_cats                              # emission row per node
    lvids = vids if layer.var_idmapping is None else layer.var_idmapping[vids]

    # ---- marginalised half: ONE GEMM, `params` read exactly once ----
    #
    # BATCH-BROADCAST EVIDENCE. logZ depends on (variable, emission row) and NOT on the batch row, so
    # when every row shares one evidence tensor the [B * n_ext] columns hold only n_ext distinct ones.
    # `_pc_group_deps` is exactly this case -- its 2C+2 rows differ ONLY in the value mask -- so at
    # C=4 that is 320 columns for 32 distinct answers, and the GEMM goes from bandwidth-bound to
    # compute-bound computing the other 288 for nothing.
    #
    # Detected by STRIDE, which costs nothing and needs no device sync: a tensor built with
    # `.expand(B, -1, -1)` has batch stride 0. A caller that materialises it with `.contiguous()`
    # loses the signal and simply gets the general path -- correct either way, just slower.
    bcast = (batch_size == 1) or (evi.stride(0) == 0)
    e = evi[0] if bcast else evi.reshape(batch_size * n_ext, num_cats)
    m = e.max(dim = -1, keepdim = True).values
    # `Pm @ ...` and NOT `... @ Pm.t()`: measured 1.4x faster and the orientation that sits on the
    # memory roofline, because it walks `params` in its natural row-major order.
    El = _torch.exp(e - m)
    if Pm.dtype != El.dtype:
        El = El.to(Pm.dtype)                                     # bf16 table -> bf16 operand
    logZ = _torch.log((Pm @ El.t()).float()) + m.t()             # accumulate/log in fp32 regardless

    bstep = 0 if bcast else n_ext
    bcol = _torch.arange(batch_size, device = evi.device)[None, :] * bstep
    col = bcol + lvids[:, None]                                  # [num_nodes, B] -> pair column
    marg = logZ[rows[:, None].expand(-1, batch_size), col]       # [num_nodes, B]

    # ---- observed half: two gathers, no intermediate ----
    tok = data[vids, :]                                          # [num_nodes, B]
    vexp = lvids[:, None].expand(-1, batch_size)
    if bcast:
        # Index the SINGLE row, never the expanded batch axis. Advanced-indexing a stride-0 tensor
        # makes PyTorch materialise it first -- 550 MB at batch 34 on this circuit, and it turned a
        # 1.3 ms call into 90 ms. Found only by driving the real `_pc_group_deps`, not the harness.
        lp = evi[0][vexp, tok]                                   # [num_nodes, B]
    else:
        bidx = _torch.arange(batch_size, device = evi.device)[None, :].expand(tok.size(0), batch_size)
        lp = evi[bidx, vexp, tok]                                # evidence at the token
    inp = _torch.log(params[rows[:, None] * num_cats + tok])     # emission at the token

    out = _torch.where(vmask.t()[lvids], lp + inp, marg)

    node_mars[sid:eid, :batch_size] = out


class _FwDenseGemmDispatch:
    """Adapter so the GEMM forward can live in `post_fw_fns`, which calls ``kernel[grid](**kw)``."""

    def __getitem__(self, grid):
        def launch(**kw):
            gk = kw["_gemm_kwargs"]
            _fw_dense_gemm(kw["_gemm_layer"], gk, kw)
        return launch


def _condition_apply_fw_gemm(layer, kwargs):
    return _fw_gemm_applicable(layer, kwargs)


def _prep_args_apply_fw_gemm(layer, kwargs):
    # Tensors, not pointers: `node_mars_ptr`/`data_ptr`/`batch_size` already come from the launcher.
    return {"_gemm_kwargs": kwargs, "_gemm_layer": layer, "BLOCK_SIZE": 1}, (1,)


def _condition_apply_fw_w_value_mask_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and \
        kwargs.get("soft_evidence_value_mask", None) is not None and \
        not _fw_gemm_applicable(layer, kwargs)


def _prep_args_apply_fw_w_value_mask_kernel(layer, kwargs):
    # Identical setup to the standard forward, plus the value mask. Delegating keeps the two
    # kernels' launch configurations in lockstep.
    target_kwargs, grid = _prep_args_apply_fw_kernel(layer, kwargs)

    soft_evidence_value_mask = kwargs["soft_evidence_value_mask"]
    assert soft_evidence_value_mask.dim() == 2
    assert soft_evidence_value_mask.size(0) == kwargs["batch_size"], \
        "Batch size doesn't match in `soft_evidence_value_mask`."
    assert soft_evidence_value_mask.size(1) == target_kwargs["ext_num_vars"], \
        "Number of variables doesn't match in `soft_evidence_value_mask`."

    target_kwargs["soft_evidence_value_mask_ptr"] = soft_evidence_value_mask.contiguous()

    return target_kwargs, grid
