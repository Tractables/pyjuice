from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
import os
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


def sort_soft_evidence_candidates(categorical_evidence_logp: torch.Tensor, soft_evidence_cat_ids: torch.Tensor,
                                  return_inverse: bool = False):
    """
    Sort top-k soft evidence by category id along the candidate axis.

    With top-k soft evidence (`soft_evidence_cat_ids` provided) the kernels address both `params` and
    `param_flows` as `<row base>(node) + cat_ids(batch, k)`. `torch.topk` returns candidates in
    descending *probability* order, i.e. uniformly random w.r.t. category id, so consecutive `k` land
    ~`num_cats/k` rows apart and every access is its own memory sector. Sorting by category id makes
    consecutive `k` adjacent (~`num_cats/k` categories apart), which is a large win -- and it is what
    makes the kernels' candidate-innermost tile layout pay off.

Measured on the CoDD/latent config (homogeneous HMM, seq 32, 1024 latents, 126464 cats, top-k 1024,
    batch 8, dual-flow backward), the FORWARD goes from 17.1 ms to 3.9 ms (4.3x) for a sort cost of
    0.06 ms per step, with a bit-identical log-likelihood.

    :note: this matters for the forward only. The backward's expensive phase now uses an inverted index
           (see the note above `_dense_topk_applicable`) which is order-independent by construction --
           it measured 5.34 ms unsorted vs 5.33 ms sorted. So sorting is worth it for the forward, and
           harmless for the backward.

    Permuting the candidate axis is semantically a no-op: `k` is only ever used as an index into the
    paired (`soft_evidence_cat_ids`, `categorical_evidence_logp`) lists. Three things to keep in mind:

    * `categorical_evidence_logp_grad` is indexed by `k` as well, so it comes back in the SAME order as
      the arrays you passed in -- automatic if you sort once at top-k time and use the sorted arrays
      throughout. Pass `return_inverse = True` if you need to map a gradient back to the original
      candidate order (`grad_orig = torch.gather(grad_sorted, -1, inverse)`).
    * Do not assume the observed token sits at a fixed slot (e.g. `k = 0`, or the last slot where a
      top-k builder forced it in). Sorting moves it; derive the observed values independently.
    * The candidate ids within a `(batch, var)` row must be UNIQUE. This is a pre-existing requirement,
      not one sorting introduces -- the kernels locate the observed token with
      `sum((cat_ids == data) * arange(...))`, which silently returns a bogus slot if a tile holds the
      same id twice -- but it is worth restating, since `torch.topk` guarantees it and hand-built
      candidate sets may not.

    Sorting per step is cheap enough not to bother caching (candidates change every step anyway); best
    of all is to have the producer emit them sorted.

    :param categorical_evidence_logp: [B, num_vars, k] external log-probabilities
    :param soft_evidence_cat_ids: [B, num_vars, k] the category id of each candidate

    :returns: `(logp_sorted, cat_ids_sorted)`, or `(logp_sorted, cat_ids_sorted, inverse)`
    """
    assert categorical_evidence_logp.shape == soft_evidence_cat_ids.shape, \
        "`categorical_evidence_logp` and `soft_evidence_cat_ids` must have the same shape."

    perm = soft_evidence_cat_ids.argsort(dim = -1)

    cat_ids_sorted = torch.gather(soft_evidence_cat_ids, -1, perm).contiguous()
    logp_sorted = torch.gather(categorical_evidence_logp, -1, perm).contiguous()

    if return_inverse:
        return logp_sorted, cat_ids_sorted, perm.argsort(dim = -1)

    return logp_sorted, cat_ids_sorted


def _condition_apply_fw_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and \
        kwargs.get("soft_evidence_value_mask", None) is None


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
    if "soft_evidence_cat_ids" in kwargs and kwargs["soft_evidence_cat_ids"] is not None:
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

    layer_num_nodes = layer._output_ind_range[1] - layer._output_ind_range[0]
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B), triton.cdiv(layer_num_nodes, BLOCK_SIZE_N))

    target_kwargs["TILE_SIZE_K"] = TILE_SIZE_K
    target_kwargs["K_NUM_TILES"] = K_NUM_TILES
    target_kwargs["BLOCK_SIZE_B"] = BLOCK_SIZE_B
    target_kwargs["BLOCK_SIZE_N"] = BLOCK_SIZE_N
    target_kwargs["use_tensor_core"] = use_tensor_core

    return target_kwargs, grid


def _condition_apply_fw_w_value_mask_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and \
        kwargs.get("soft_evidence_value_mask", None) is not None


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


def _condition_apply_bk_params_kernel(layer, kwargs):
    return "categorical_evidence_logp" in kwargs and not kwargs["dual_flow_backward"]


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


###############################################################################################
##  The two parameter-flow phases, and a dense kernel for the expensive one                  ##
###############################################################################################
#
# WHAT THE TWO PHASES ARE
#
# With `_dual_flow_backward`, `num_param_flows()` is `2 * num_cats`, so each node's slice of
# `param_flows` holds two concatenated accumulators of width `num_cats`:
#
#   phase 0, at offset 0            -- the OBSERVED-category flow. Exactly what a plain `Categorical`
#                                      accumulates: the node flow added at the observed category, so it
#                                      is nonzero in at most one category per (node, sample).
#   phase 1, at offset `num_cats`   -- the EXPECTED-category flow. The same node flow, but spread across
#                                      every category the leaf considers possible, in proportion to the
#                                      leaf's own posterior over categories,
#                                      `beta[node, c] * p_theta(c) / Z`. It is the model's expected
#                                      sufficient statistic rather than an observed count, so it is
#                                      dense over the candidate set.
#
# The M-step then forms roughly `beta * (phase0 + pseudocount/K) / (phase1 + pseudocount*beta)` -- an
# observed count divided by an expected count. Dividing by the expected count is what makes the update
# a NORMALIZED (conditional) one instead of the plain joint-likelihood update a single flow gives.
# Elsewhere in this file the two phases are written `F+` / `F-`, or "numerator" / "denominator".
#
# WHY PHASE 1 IS THE EXPENSIVE ONE, AND HOW THIS KERNEL FIXES IT
#
# Being dense over the candidate set, phase 1 must be touched once per (node, candidate) rather than
# once per (node, sample). The straightforward way -- what `bk_softevi_kernel` does -- is to walk the
# soft-evidence slots and scatter:
#     phase1[row(position, latent), cat] += ratio[position, batch, latent] * beta[latent, cat] * p_theta
# one `atomic_add` per (position, batch, candidate) slot. On the CoDD config that is 268M random atomics
# into a multi-GB buffer. But it is a NEARLY INJECTIVE scatter executed randomly: it writes only ~113M
# distinct (row, cat) slots, i.e. ~1.3 slots reference each category. And the cost is the address
# pattern, not the atomic -- those same 268M `atomic_add`s take 1.0 ms coalesced versus 44 ms scattered.
#
# So invert it. Group the slots by category once per step, then walk (latent x category) with the
# CATEGORY axis innermost -- the only axis of `param_flows` that is contiguous -- which gives every
# (row, cat) a single owner and needs no atomic at all:
#     phase1[row, c] = beta[l, c] * sum over the slots referencing c of ratio[slot, l] * p_theta[slot]
# The per-category reference lists are tiny, so they are padded to `MAX_REFS` and masked. The same
# `beta` read also serves the external-evidence gradient's expected-value term, so that is folded into
# the same walk rather than paying for the reads twice.
#
# Measured on the CoDD config (seq 32, 1024 latents, 126464 cats, top-k 1024, batch 8, one param-flow
# block), whole-PC backward: 65.7 ms -> 5.3 ms. Being index-driven it is also insensitive to the order of
# the candidate axis, unlike the scattered kernel it replaces.

_DENSE_TOPK_BACKWARD = os.environ.get("PYJUICE_SOFTEVI_DENSE_BACKWARD", "1") != "0"

# Cap on how many soft-evidence slots may reference one category. ~1.3 is typical; the index build
# checks the real maximum and falls back to the scattered kernel if it is exceeded, so this only bounds
# how much padding is wasted.
_DENSE_MAX_REFS = 16


def _dense_topk_applicable(layer, kwargs):
    """Whether to use the dense kernels for the expected-category flow phase (see the note above).

    Only that phase changes; the observed-category flow, the evidence gradient and the forward are
    computed exactly as before, and any case this does not cover falls back to `bk_softevi_kernel`."""
    if not _DENSE_TOPK_BACKWARD:
        return False
    if "categorical_evidence_logp" not in kwargs:
        return False
    if kwargs.get("soft_evidence_cat_ids", None) is None:
        return False
    if not kwargs["dual_flow_backward"]:
        # Without the dual-flow denominator there is no scatter to eliminate.
        return False
    if layer.provided("bk_local_ids"):
        return False
    if not _dense_worth_it(layer, kwargs):
        return False
    return _build_dense_index(layer, kwargs) is not None


def _dense_worth_it(layer, kwargs):
    """
    Is the dense path actually faster than just scattering?

    The dense path carries a fixed cost (the inverted-index build plus the prologue that materializes the
    per-slot ratio) of order 1 ms, independent of the top-k width. It buys that back only when the
    scattered alternative is going to DRAM -- if `params` and the flow buffer fit in L2, random atomics
    are already cheap and there is nothing to win. Measured backward, dense vs scattered:

        8 vars,   64 latents,   2k cats, k=48   ->  0.96 vs 0.56 ms   (dense 1.7x SLOWER)
        16 vars, 128 latents,   8k cats, k=64   ->  0.99 vs 0.57 ms   (dense 1.7x slower)
        32 vars, 256 latents,  32k cats, k=128  ->  1.00 vs 0.70 ms   (dense 1.4x slower, ~67 MB, L2-resident)
        32 vars, 1024 latents,126k cats, k=32   ->  1.68 vs 2.32 ms   (dense 1.4x faster, ~1 GB)
        32 vars, 1024 latents,126k cats, k=1024 ->  2.71 vs 11.34 ms  (dense 4.2x faster)

    The middle two have the SAME number of scatter operations (8.4M) and opposite verdicts, so the
    footprint -- not the operation count -- is what decides it.
    """
    tot_num_cats = layer.nodes[0].dist.num_cats
    lnn = layer._output_ind_range[1] - layer._output_ind_range[0]
    ext_num_vars = kwargs["categorical_evidence_logp"].size(1)
    if ext_num_vars == 0 or lnn % ext_num_vars != 0:
        return False
    num_latents = lnn // ext_num_vars

    # What the scattered path touches at random: the emission parameters plus the phase-1 flow half
    footprint = 2 * num_latents * tot_num_cats * 4

    l2 = getattr(_dense_worth_it, "_l2", None)
    if l2 is None:
        try:
            l2 = torch.cuda.get_device_properties(layer.params.device).L2_cache_size
        except Exception:
            l2 = 64 * 1024 * 1024
        _dense_worth_it._l2 = l2

    return footprint > l2


def _dense_scratch(layer, ext_num_vars, batch_size, num_latents):
    """[ext_num_vars * batch_size, num_latents] scratch for the per-(slot, latent) flow/Z ratio (~1 MB)."""
    shape = (ext_num_vars * batch_size, num_latents)
    buf = getattr(layer, "_dense_ratio_buf", None)
    if buf is None or tuple(buf.shape) != shape or buf.device != layer.params.device:
        buf = torch.zeros(shape, dtype = torch.float32, device = layer.params.device)
        layer._dense_ratio_buf = buf
    return buf


def _dense_layer_layout(layer, num_ext_vars):
    """
    How the layer's nodes map onto variables and param-flow blocks.

    This depends only on the compiled layer, never on the step's data, so it is computed once and cached.
    Doing it per step is what made the index build look expensive: grouping the variables by param-flow
    block with `int(s_pfids[i])` costs one device->host sync PER VARIABLE (32 of them here, ~0.65 ms),
    which dwarfed the ~0.2 ms of actual index work.

    Returns `(num_latents, groups, lvid_of_head, pf_base, p_base)`, or None if the layout is not one
    the dense kernels
    support (the caller then falls back to the scattered kernel).
    """
    lnn = layer._output_ind_range[1] - layer._output_ind_range[0]
    cached = getattr(layer, "_dense_layout_cache", None)
    if cached is not None and cached[0] == (lnn, num_ext_vars):
        return cached[1]

    dev = layer.s_pfids.device
    if num_ext_vars == 0 or lnn % num_ext_vars != 0:
        return None
    num_latents = lnn // num_ext_vars

    # Every variable's nodes must be one contiguous run of `num_latents` slots sharing one param-flow
    # block; that is what lets a block be described by a single (pf_base, p_base) pair per latent.
    if not torch.equal(layer.nids.view(-1),
                       torch.arange(num_latents, device = dev).repeat(num_ext_vars)):
        layer._dense_layout_cache = ((lnn, num_ext_vars), None)
        return None

    head_slots = torch.arange(0, lnn, num_latents, device = dev)
    lvid_of_head = layer.var_idmapping[layer.vids.view(-1)][head_slots]

    # One host transfer for the whole thing, rather than one per variable
    pf_of_head = layer.s_pfids[head_slots].tolist()

    blocks = {}
    for i, pf in enumerate(pf_of_head):
        blocks.setdefault(int(pf), []).append(i)
    groups = [sorted(v) for v in blocks.values()]

    # The per-block parameter / parameter-flow row bases are static too
    heads = [int(head_slots[g[0]]) for g in groups]
    pf_base = torch.stack([layer.s_pfids[h : h + num_latents] for h in heads]).contiguous()
    p_base = torch.stack([layer.s_pids[h : h + num_latents] for h in heads]).contiguous()

    layout = (num_latents, groups, lvid_of_head, pf_base, p_base)
    layer._dense_layout_cache = ((lnn, num_ext_vars), layout)
    return layout


def _build_dense_index(layer, kwargs):
    """
    Invert (position, batch, candidate) -> category, once per step, per param-flow block.

    Returns a dict of device tensors (padded over blocks so one launch covers all of them), or None if
    the layout is not supported -- in which case the caller falls back to the scattered kernel.

    Cached on the layer and keyed by the identity of `soft_evidence_cat_ids`, so the forward and the
    backward of a step share one build.
    """
    cat_ids = kwargs["soft_evidence_cat_ids"]
    evidence = kwargs["categorical_evidence_logp"]

    # `batch_size` is injected into `kwargs` only after the condition check, so take it from the tensor
    # (the layer asserts these agree in the prep functions).
    key = (cat_ids.data_ptr(), cat_ids._version, evidence.data_ptr(), evidence._version,
           tuple(cat_ids.shape))
    cached = getattr(layer, "_dense_index_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    B, V, K = cat_ids.shape
    dev = cat_ids.device

    layout = _dense_layer_layout(layer, V)
    if layout is None:
        return None
    num_latents, groups, lvid_of_head, pf_base, p_base = layout
    G = len(groups)

    # ---- per block: sort the slots by category, then pad to [U, MAX_REFS] ----
    uniq_l, slot_l, pt_l, goff_l, cnt_l, n_uniq = [], [], [], [], [], []
    pt_all = evidence.exp()
    for g in groups:
        lv = lvid_of_head[torch.tensor(g, device = dev)]                     # [Gp] layer-local var ids

        ids_g = cat_ids[:, lv, :]                                            # [B, Gp, K]
        pt_g = pt_all[:, lv, :]
        Gp = lv.numel()

        # slot id indexes the [V*B, num_latents] ratio scratch
        slot = (lv.view(1, Gp, 1) * B + torch.arange(B, device = dev).view(B, 1, 1)).expand(B, Gp, K)
        # gradient offset into `categorical_evidence_logp_grad` [B, V, K]
        goff = (torch.arange(B, device = dev).view(B, 1, 1) * (V * K) +
                lv.view(1, Gp, 1) * K + torch.arange(K, device = dev).view(1, 1, K)).expand(B, Gp, K)

        # Sort 32-bit keys (category ids are < num_cats) rather than 64-bit -- this is the single most
        # expensive step of the build.
        cat_f = ids_g.reshape(-1).int()
        order = cat_f.argsort()
        cat_s = cat_f[order]

        # `return_inverse` gives the per-element row directly, which avoids two `repeat_interleave`s
        uniq, inverse, counts = torch.unique_consecutive(cat_s, return_inverse = True, return_counts = True)
        if int(counts.max().item()) > _DENSE_MAX_REFS:
            return None                                                      # -> scattered fallback

        U = uniq.numel()
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        row = inverse
        within = torch.arange(cat_s.numel(), device = dev) - starts[row]

        rc = counts.int().contiguous()
        # [MAX_REFS, U] rather than [U, MAX_REFS]: the kernel assigns one CATEGORY per thread, so with
        # the reference index innermost, lane-adjacent threads would read MAX_REFS apart and every lane
        # would fetch its own sector. Transposed, a warp's reads of reference `j` are contiguous.
        rs = torch.zeros([_DENSE_MAX_REFS, U], dtype = torch.int32, device = dev)
        rp = torch.zeros([_DENSE_MAX_REFS, U], dtype = torch.float32, device = dev)
        # offsets are bounded by B * V * K, so 32 bits is plenty
        rg = torch.zeros([_DENSE_MAX_REFS, U], dtype = torch.int32, device = dev)
        rs[within, row] = slot.reshape(-1)[order].int()
        rp[within, row] = pt_g.reshape(-1)[order]
        rg[within, row] = goff.reshape(-1)[order].int()

        uniq_l.append(uniq.int()); slot_l.append(rs); pt_l.append(rp); goff_l.append(rg)
        cnt_l.append(rc)
        n_uniq.append(U)

    Umax = max(n_uniq)

    def _pad(ts, dim = 0):
        out = []
        for t in ts:
            if t.size(dim) < Umax:
                shape = list(t.shape)
                shape[dim] = Umax - t.size(dim)
                t = torch.cat([t, t.new_zeros(shape)], dim = dim)
            out.append(t)
        return torch.stack(out).contiguous()

    index = dict(
        uniq = _pad(uniq_l),                                              # [G, Umax]
        ref_slot = _pad(slot_l, dim = 1),                            # [G, Umax, MAX_REFS]
        ref_pt = _pad(pt_l, dim = 1),
        ref_goff = _pad(goff_l, dim = 1),
        ref_cnt = _pad(cnt_l),                                               # [G, Umax]
        num_uniq = torch.tensor(n_uniq, dtype = torch.int32, device = dev),   # [G]
        pf_base = pf_base,
        p_base = p_base,
        num_blocks = G,
        num_latents = num_latents,
        Umax = Umax,
    )

    layer._dense_index_cache = (key, index)
    return index


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
    target_kwargs["categorical_evidence_logp_grad_ptr"] = grad
    target_kwargs["update_extflows"] = grad is not None

    target_kwargs["ratio_ptr"] = _dense_scratch(layer, ext_num_vars, batch_size, index["num_latents"])

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
                    from .c_kernels import dense_expected_flow_available
                    self._use_cuda = dense_expected_flow_available()
                except Exception:
                    self._use_cuda = False
        return self._use_cuda

    def __getitem__(self, grid):
        if not self._cuda_ok():
            return self.triton_kernel[grid]

        def launch(**kw):
            from .c_kernels import dense_expected_flow
            dense_expected_flow(
                kw["params_ptr"], kw["param_flows_ptr"], kw["ratio_ptr"],
                kw["uniq_ptr"], kw["ref_slot_ptr"], kw["ref_pt_ptr"], kw["ref_goff_ptr"],
                kw["ref_cnt_ptr"], kw["num_uniq_ptr"], kw["pf_base_ptr"], kw["p_base_ptr"],
                kw["categorical_evidence_logp_grad_ptr"] if kw["update_extflows"] else None,
                kw["num_latents"], kw["tot_num_cats"], kw["UNIQ_STRIDE"], kw["MAX_REFS"],
                grid[2], kw.get("CUDA_THREADS", 256), kw["ratio_ptr"].size(0), kw.get("CUDA_TL", 32))

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

    target_kwargs["uniq_ptr"] = index["uniq"]
    target_kwargs["ref_slot_ptr"] = index["ref_slot"]
    target_kwargs["ref_pt_ptr"] = index["ref_pt"]
    target_kwargs["ref_goff_ptr"] = index["ref_goff"]
    target_kwargs["ref_cnt_ptr"] = index["ref_cnt"]
    target_kwargs["num_uniq_ptr"] = index["num_uniq"]
    target_kwargs["pf_base_ptr"] = index["pf_base"]
    target_kwargs["p_base_ptr"] = index["p_base"]
    target_kwargs["ratio_ptr"] = _dense_scratch(layer, evidence.size(1), batch_size, num_latents)

    grad = kwargs.get("categorical_evidence_logp_grad", None)
    target_kwargs["categorical_evidence_logp_grad_ptr"] = grad
    target_kwargs["update_extflows"] = grad is not None

    target_kwargs["num_latents"] = num_latents
    target_kwargs["tot_num_cats"] = layer.nodes[0].dist.num_cats
    target_kwargs["pf_row_stride"] = 2 * layer.nodes[0].dist.num_cats
    target_kwargs["MAX_REFS"] = _DENSE_MAX_REFS
    # CUDA-path launch geometry, kept separate from the Triton tile so each can be tuned on its own.
    # TL = latents per thread: swept 4/8/16/32/64 -> 2.17/1.83/1.72/1.58/1.67 ms (32 is the optimum;
    # 64 regresses on register pressure). Threads 64-256 are within noise, 256 marginally best.
    target_kwargs["CUDA_TL"] = 32
    target_kwargs["CUDA_THREADS"] = 256
    target_kwargs["UNIQ_STRIDE"] = index["Umax"]
    # Swept on the CoDD config with the evidence-gradient term folded in; the optimum differs from the
    # fold-off optimum (the fold adds live temporaries), so re-sweep if that term ever moves out.
    target_kwargs["BLOCK_L"] = 64
    target_kwargs["BLOCK_C"] = 128

    grid = (triton.cdiv(num_latents, target_kwargs["BLOCK_L"]),
            triton.cdiv(index["Umax"], target_kwargs["BLOCK_C"]),
            index["num_blocks"])

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
    target_kwargs["categorical_evidence_logp_grad_ptr"] = categorical_evidence_logp_grad
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
    assert not layer.provided("fw_local_ids")
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

    # Whether to update `pflow` and `extflow`
    target_kwargs["update_pflows"] = kwargs["dual_flow_backward"]
    target_kwargs["update_extflows"] = ("categorical_evidence_logp_grad" in kwargs)

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

    :note: with top-k soft evidence (i.e. when `soft_evidence_cat_ids` is supplied), pass the candidate
           axis SORTED BY CATEGORY ID -- see :func:`sort_soft_evidence_candidates`. The forward is bound
           by the `params` gather addressed through `cat_ids`, so the candidate order dominates its cost:
           4.3x on the CoDD/latent config for a 0.06 ms sort. Sorting is semantically a no-op.

    :param num_cats: number of categories
    :type num_cats: int
    """
    def __init__(self, num_cats: int, _dual_flow_backward: bool = True):
        super(SoftEvidenceCategorical, self).__init__()

        self.num_cats = num_cats

        self.post_fw_fns = [
            (self.fw_kernel, _condition_apply_fw_kernel, _prep_args_apply_fw_kernel),
            # Opt-in generation forward; the two conditions are mutually exclusive on
            # `soft_evidence_value_mask`, so the default path is untouched.
            (self.fw_w_value_mask_kernel, _condition_apply_fw_w_value_mask_kernel, _prep_args_apply_fw_w_value_mask_kernel)
        ]

        self.post_bp_fns = [
            (self.bk_params_kernel, _condition_apply_bk_params_kernel, _prep_args_apply_bk_params_kernel),
            (self.bk_softevi_kernel, _condition_apply_bk_softevi_kernel, _prep_args_apply_bk_softevi_kernel),
            # Dense top-k denominator path (dedicated kernels; mutually exclusive with the scattered
            # `bk_softevi_kernel` above, which stays the fallback for every other case). Order matters:
            # the prologue writes the ratio scratch that the denominator kernel reads.
            (self.bk_dense_prologue_kernel, _condition_bk_dense_prologue, _prep_args_bk_dense_prologue),
            (_DenseDenomDispatch(self.bk_dense_denom_kernel), _condition_bk_dense_denom, _prep_args_bk_dense_denom)
        ]

        self.sampling_fns = [
            (self.sample_kernel, _condition_sample_kernel, _prep_args_sample_kernel)
        ]

        self._dual_flow_backward = _dual_flow_backward

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
        return self.num_cats * 2 if self._dual_flow_backward else self.num_cats

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
        if self._dual_flow_backward:
            if self.num_cats <= 256:
                return self.small_ncats_dual_em_fn
            else:
                self.em_block_size = 8
                return self.large_ncats_dual_em_fn
        else:
            if self.num_cats <= 256:
                return self.small_ncats_em_fn
            else:
                self.em_block_size = 8
                return self.large_ncats_em_fn

    def get_flow_mask_fn(self):
        if self._dual_flow_backward:
            return self.bk_dual_flow_mask_fn
        else:
            return self.bk_flow_mask_fn

    def set_custom_kernel_kwargs(self, kwargs):
        kwargs["dual_flow_backward"] = self._dual_flow_backward

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
                # :note: unlike the backward's expected-category flow phase, the [B, K, N] tile (node
                #        axis innermost) measured FASTER here -- the forward only gathers, it does not
                #        also scatter, and the logsumexp then reduces over the innermost axis. Flipping
                #        this to [B, N, K] cost 43% on the sorted-candidate forward (3.3 -> 4.7 ms).
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

                # Find matching ids (mask out padding categories so they can't spuriously match `data == 0`)
                is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
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
    def fw_w_value_mask_kernel(params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, nids_ptr, fw_local_ids_ptr, layer_num_nodes,
                               batch_size, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset, partial_eval: tl.constexpr,
                               TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, use_tensor_core: tl.constexpr,
                               categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr, soft_evidence_value_mask_ptr, var_idmapping_ptr,
                               num_cats: tl.constexpr, ext_num_vars: tl.constexpr, has_ext_ids: tl.constexpr):
        """Forward pass with a per-variable value mask (used for generation/conditional queries).

        A mirror of `fw_kernel` -- everything up to and including `logZ`, `log_in_p` and `log_ex_p` is
        identical; only the final output differs. Per variable, `soft_evidence_value_mask` selects

            observed (`True`)  -> log beta_z(d) + log p_theta(d)      [unnormalized conditional, no -logZ]
            masked   (`False`) -> logZ_z = log sum_c beta_z(c) p_theta(c)   [evidence-weighted marginal]

        which is exactly what `ExternProductCategorical` computes in mode "unnormalized_ll" with a
        value mask. `fw_kernel` is deliberately left untouched so the training/LL forward keeps its
        code path, kernel and performance unchanged.
        """

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
                # :note: unlike the backward's expected-category flow phase, the [B, K, N] tile (node
                #        axis innermost) measured FASTER here -- the forward only gathers, it does not
                #        also scatter, and the logsumexp then reduces over the innermost axis. Flipping
                #        this to [B, N, K] cost 43% on the sorted-candidate forward (3.3 -> 4.7 ms).
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

                # Find matching ids (mask out padding categories so they can't spuriously match `data == 0`)
                is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
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

        # Get the value mask (`True` to condition on the observed value, `False` to marginalize)
        value_mask = tl.load(soft_evidence_value_mask_ptr + offsets_b * ext_num_vars + lvid, mask = mask_b, other = False) # [BLOCK_SIZE_B]

        # Final output logprob: unnormalized conditional where observed, logZ where masked
        log_p = tl.where(value_mask[:,None], log_in_p + log_ex_p[:,None], logZ)

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
                          num_cats: tl.constexpr, tot_num_cats: tl.constexpr, ext_num_vars: tl.constexpr, has_ext_ids: tl.constexpr, update_pflows: tl.constexpr, update_extflows: tl.constexpr):
        
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

        # Update the numerater part of `pflow` if required
        if update_pflows:
            # Get start parameter flow indices
            s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0) # [BLOCK_SIZE_N]

            tl.atomic_add(param_flows_ptr + s_pfids[None,:] + data[:,None], nflows, mask = (mask_b[:,None] & mask_n[None,:]))

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
            if update_extflows:
                expar_grad_ptr = categorical_evidence_logp_grad_ptr + \
                    offsets_b * (ext_num_vars * num_cats) + \
                    lvid * num_cats # [BLOCK_SIZE_B]

            log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Find matching ids (mask out padding categories so they can't spuriously match `data == 0`)
                is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64) # [BLOCK_SIZE_B, TILE_SIZE_K]
                match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1) # [BLOCK_SIZE_B]
                has_match = (tl.sum(is_match, axis = 1) > 0) # [BLOCK_SIZE_B]

                # Load parameters if found
                expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0) # [BLOCK_SIZE_B]
                log_ex_p = tl.where(has_match, expar, log_ex_p)

                # Accumulate gradients for `extflow`
                if update_extflows:
                    tl.atomic_add(expar_grad_ptr + i * TILE_SIZE_K + match_ids, tl.sum(nflows, axis = 1), mask = (mask_b & has_match)) # [BLOCK_SIZE_B]

        else:
            ex_p_ptr = categorical_evidence_logp_ptr + \
                offsets_b * (ext_num_vars * num_cats) + \
                lvid * num_cats + \
                data
            log_ex_p = tl.load(ex_p_ptr, mask = mask_b, other = 0.0) # [BLOCK_SIZE_B]

            # Accumulate gradients
            if update_extflows:
                unnorm_ll_grad_ptr = categorical_evidence_logp_grad_ptr + \
                    offsets_b * (ext_num_vars * num_cats) + \
                    lvid * num_cats + \
                    data
                tl.atomic_add(unnorm_ll_grad_ptr, tl.sum(nflows, axis = 1), mask = mask_b)

        # Retrieve logZ
        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = (mask_b[:,None] & mask_n[None,:]), other = 0.0).log() # [BLOCK_SIZE_B, BLOCK_SIZE_N]
        logZ = log_in_p + log_ex_p[:,None] - nmars # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Loop-invariant, so hoisted out of the K loop below (which otherwise recomputes it once per
        # tile, per phase)
        log_nflow_sub_logz = nflows.log() - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]

        # Ptrs pointing to external parameter gradients
        if update_extflows:
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

            # The working tile is laid out [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K] -- i.e. the CANDIDATE
            # axis is innermost -- rather than [B, K, N]. Both the `params` gather and the `param_flows`
            # scatter address `<row base>(n) + catids(b,k)`, so with N innermost each lane of a warp lands
            # in a different node row (~0.5-1 MB apart) and every access is its own sector; with K
            # innermost a warp stays inside ONE row. Measured on the CoDD config: this phase 25.4 -> 16.6
            # ms (and this is what makes sorted candidate ids pay off -- see the class docstring).
            for i in range(K_NUM_TILES):
                mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats) # [TILE_SIZE_K]

                # Load the category IDs from `soft_evidence_cat_ids`
                catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Load the internal parameters
                in_catpars_ptr = inpars_ptr[None,:,None] + catids[:,None,:] # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]
                inpars = tl.load(in_catpars_ptr, mask = (mask_b[:,None,None] & mask_n[None,:,None] & mask_c[None,None,:]), other = 0.0) # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                # Shared by both phases below
                log_inpars = inpars.log() # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]

                if update_pflows:
                    vp_grads = log_nflow_sub_logz[:,:,None] + log_inpars + expars[:,None,:] # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]

                    tl.atomic_add(param_flows_ptr + tot_num_cats + s_pfids[None,:,None] + catids[:,None,:], tl.exp(vp_grads), mask = (mask_b[:,None,None] & mask_n[None,:,None]))

                if update_extflows:
                    ve_grads = log_nflow_sub_logz[:,:,None] + log_inpars # [BLOCK_SIZE_B, BLOCK_SIZE_N, TILE_SIZE_K]
                    ve_grads_max = tl.max(ve_grads, axis = 1)
                    ve_grads_sub = tl.exp(ve_grads - ve_grads_max[:,None,:])
                    cum_ve_grads = tl.sum(ve_grads_sub, axis = 1).log() + ve_grads_max # [BLOCK_SIZE_B, TILE_SIZE_K]

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

            if update_pflows:
                nflow_sub_logz_p = tl.trans(nflows.log() - logZ) # [BLOCK_SIZE_N, BLOCK_SIZE_B]
                nflow_sub_logz_p_max = tl.max(nflow_sub_logz_p, axis = 1)[:,None]
                nflow_sub_logz_p_sub = tl.exp(nflow_sub_logz_p - nflow_sub_logz_p_max)

            if update_extflows:
                nflow_sub_logz = nflows.log() - logZ # [BLOCK_SIZE_B, BLOCK_SIZE_N]
                nflow_sub_logz_max = tl.max(nflow_sub_logz, axis = 1)[:,None]
                nflow_sub_logz_sub = tl.exp(nflow_sub_logz - nflow_sub_logz_max)

            for i in range(K_NUM_TILES):
                offsets_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K))
                mask_c = (offsets_c < num_cats) # [TILE_SIZE_K]

                # Load the internal parameters
                if use_tensor_core:
                    inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[None,:] & mask_n[:,None]), other = 0.0) # [BLOCK_SIZE_N, TILE_SIZE_K]
                else:
                    inpars = tl.load(inpars_ptr + i * TILE_SIZE_K, mask = (mask_c[:,None] & mask_n[None,:]), other = 0.0) # [TILE_SIZE_K, BLOCK_SIZE_N]

                # Load the external parameters
                expars = tl.load(expars_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0.0) # [BLOCK_SIZE_B, TILE_SIZE_K]

                if update_pflows:
                    if use_tensor_core:
                        expars_max = tl.max(expars, axis = 0)[None,:]
                        expars_sub = tl.exp(expars - expars_max)
                        pars_grad = tl.dot(nflow_sub_logz_p_sub, expars_sub) + inpars.log() + nflow_sub_logz_p_max + expars_max
                    else:
                        expars_max = tl.max(expars, axis = 0)
                        expars_sub = tl.exp(tl.trans(expars) - expars_max[:,None]) # [TILE_SIZE_K, BLOCK_SIZE_B]
                        pars_grad = tl.sum(nflow_sub_logz_p_sub[:,None,:] * expars_sub[None,:,:], axis = 2).log() + tl.trans(inpars).log() + nflow_sub_logz_p_max + expars_max[None,:]

                    tl.atomic_add(param_flows_ptr + tot_num_cats + s_pfids[:,None] + offsets_c[None,:], tl.exp(pars_grad), mask = (mask_n[:,None] & mask_c[None,:]))

                if update_extflows:
                    if use_tensor_core:
                        expars_grad = tl.dot(nflow_sub_logz_sub, inpars).log() + nflow_sub_logz_max + expars
                    else:
                        expars_grad = tl.sum(nflow_sub_logz_sub[:,None,:] * inpars[None,:,:], axis = 2).log() + nflow_sub_logz_max + expars

                    tl.atomic_add(expars_grad_ptr + i * TILE_SIZE_K, -tl.exp(expars_grad), mask = (mask_b[:,None] & mask_c[None,:]))

    @staticmethod
    @triton_jit
    def bk_dense_prologue_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                                 metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, layer_num_nodes, batch_size,
                                 num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                                 node_offset, partial_eval: tl.constexpr, logspace_flows: tl.constexpr,
                                 BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                 TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                 categorical_evidence_logp_ptr, soft_evidence_cat_ids_ptr,
                                 categorical_evidence_logp_grad_ptr, var_idmapping_ptr, ratio_ptr,
                                 num_cats: tl.constexpr, ext_num_vars: tl.constexpr, num_latents: tl.constexpr,
                                 update_extflows: tl.constexpr):
        """First half of the dense top-k backward: everything except the expected-category flow phase.

        Accumulates the observed-category flow, adds the observed-token term of the external evidence
        gradient, and stores `ratio[slot, latent] = node_flow / Z` for the second kernel to consume.
        Identical to `bk_softevi_kernel` up to and including the local normalizer `logZ`; it just stores
        that ratio instead of running the scattered expected-flow loop."""

        pid_b = tl.program_id(axis = 0)
        pid_n = tl.program_id(axis = 1)

        offsets_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        mask_b = offsets_b < batch_size
        mask_n = offsets_n < layer_num_nodes
        mask_bn = mask_b[:,None] & mask_n[None,:]

        offset_n = pid_n * BLOCK_SIZE_N

        vid = tl.load(vids_ptr + offset_n)
        lvid = tl.load(var_idmapping_ptr + vid)

        nids = tl.load(nids_ptr + offsets_n, mask = mask_n, other = 0)
        s_pids = tl.load(s_pids_ptr + offsets_n, mask = mask_n, other = 0)
        s_pfids = tl.load(s_pfids_ptr + offsets_n, mask = mask_n, other = 0)

        # `node_mars` / `node_flows` are laid out [node, batch], so load them with the BATCH axis
        # innermost and transpose in registers. Loading them directly as [batch, node] -- which is the
        # orientation the rest of this kernel wants -- puts a stride of `batch_size` floats between
        # adjacent lanes, so each lane pulls its own 32-byte sector for 4 useful bytes. That alone was
        # 0.50 of the prologue's 0.63 ms.
        mask_nb = mask_n[:,None] & mask_b[None,:]
        nmars = tl.trans(tl.load(
            node_mars_ptr + (offsets_n + node_offset)[:,None] * batch_size + offsets_b[None,:],
            mask = mask_nb, other = 0.0))
        nflows = tl.trans(tl.load(
            node_flows_ptr + (offsets_n + node_offset)[:,None] * batch_size + offsets_b[None,:],
            mask = mask_nb, other = 0.0))

        # Keep the log form for the ratio and the linear form for the observed-category flow, rather
        # than the exp -> log round trip the scattered kernel does.
        if logspace_flows:
            log_nflows = nflows
            nflows = nflows.exp()
        else:
            log_nflows = nflows.log()

        data = tl.load(data_ptr + vid * batch_size + offsets_b, mask = mask_b, other = 0)

        # Observed-category flow (phase 0)
        tl.atomic_add(param_flows_ptr + s_pfids[None,:] + data[:,None], nflows, mask = mask_bn)

        catids_ptr = soft_evidence_cat_ids_ptr + \
            offsets_b[:,None] * (ext_num_vars * num_cats) + lvid * num_cats + tl.arange(0, TILE_SIZE_K)[None,:]
        expar_ptr = categorical_evidence_logp_ptr + offsets_b * (ext_num_vars * num_cats) + lvid * num_cats
        if update_extflows:
            expar_grad_ptr = categorical_evidence_logp_grad_ptr + offsets_b * (ext_num_vars * num_cats) + lvid * num_cats

        # Locate the observed token among the candidates, and accumulate the numerator half of the
        # external gradient onto its slot
        log_ex_p = tl.zeros([BLOCK_SIZE_B], dtype = tl.float32) - float("inf")
        for i in range(K_NUM_TILES):
            mask_c = (i * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K) < num_cats)
            catids = tl.load(catids_ptr + i * TILE_SIZE_K, mask = (mask_b[:,None] & mask_c[None,:]), other = 0)

            is_match = ((catids == data[:,None]) & mask_c[None,:]).to(tl.int64)
            match_ids = tl.sum(is_match * tl.arange(0, TILE_SIZE_K), axis = 1)
            has_match = (tl.sum(is_match, axis = 1) > 0)

            expar = tl.load(expar_ptr + i * TILE_SIZE_K + match_ids, mask = (mask_b & has_match), other = 0.0)
            log_ex_p = tl.where(has_match, expar, log_ex_p)

            if update_extflows:
                tl.atomic_add(expar_grad_ptr + i * TILE_SIZE_K + match_ids, tl.sum(nflows, axis = 1),
                              mask = (mask_b & has_match))

        log_in_p = tl.load(params_ptr + s_pids[None,:] + data[:,None], mask = mask_bn, other = 0.0).log()
        logZ = log_in_p + log_ex_p[:,None] - nmars

        # ratio[slot, latent], slot = lvid * batch_size + b; `nids` is the latent index, so the store is
        # contiguous along the innermost axis
        ratio = tl.exp(log_nflows - logZ)
        tl.store(ratio_ptr + (lvid * batch_size + offsets_b)[:,None] * num_latents + nids[None,:],
                 ratio, mask = mask_bn)

    @staticmethod
    @triton_jit
    def bk_dense_denom_kernel(params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                              metadata_ptr, s_mids_ptr, nids_ptr, bk_local_ids_ptr, layer_num_nodes, batch_size,
                              num_vars_per_node: tl.constexpr, num_vars: tl.constexpr, nv_block_size: tl.constexpr,
                              node_offset, partial_eval: tl.constexpr, logspace_flows: tl.constexpr,
                              uniq_ptr, ref_slot_ptr, ref_pt_ptr, ref_goff_ptr, ref_cnt_ptr, num_uniq_ptr,
                              pf_base_ptr, p_base_ptr, ratio_ptr, categorical_evidence_logp_grad_ptr,
                              num_latents: tl.constexpr, tot_num_cats: tl.constexpr, pf_row_stride: tl.constexpr,
                              MAX_REFS: tl.constexpr, UNIQ_STRIDE: tl.constexpr,
                              BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr,
                              update_extflows: tl.constexpr):
        """Second half of the dense top-k backward: the expected-category flow phase (phase 1).

        Walks (latent x category) so that every (param-flow row, category) has a single owner and needs
        no atomic:
            phase1[row, c] = beta[l, c] * sum_j ratio[slot_j, l] * p_theta_j
        over the soft-evidence slots j whose candidate set contains category c. The same `beta` read
        also gives the expected-value term of the external evidence gradient, so it is folded in here:
            evidence_grad[slot_j, k_j] -= p_theta_j * sum_l ratio[slot_j, l] * beta[l, c]
        """
        pid_l = tl.program_id(axis = 0)
        pid_c = tl.program_id(axis = 1)
        pid_g = tl.program_id(axis = 2)

        num_uniq = tl.load(num_uniq_ptr + pid_g)

        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_l = offs_l < num_latents
        mask_c = offs_c < num_uniq
        m = mask_l[:,None] & mask_c[None,:]

        cats = tl.load(uniq_ptr + pid_g * UNIQ_STRIDE + offs_c, mask = mask_c, other = 0).to(tl.int64)

        # beta[l, cat]: `uniq` is sorted, so this is a near-contiguous run inside each params row
        p_base = tl.load(p_base_ptr + pid_g * num_latents + offs_l, mask = mask_l, other = 0)
        beta = tl.load(params_ptr + p_base[:,None] + cats[None,:], mask = m, other = 0.0) # [BLOCK_L, BLOCK_C]

        ref_base = pid_g * (MAX_REFS * UNIQ_STRIDE) + offs_c   # + j * UNIQ_STRIDE below

        # Reference lists are padded to MAX_REFS but average ~2 entries, and the evidence-gradient term
        # below costs real work on every iteration (it scaled +1.1 ms from MAX_REFS=1 to 16). So bound the
        # loop by the largest list actually present in THIS tile rather than by the global maximum.
        cnt = tl.load(ref_cnt_ptr + pid_g * UNIQ_STRIDE + offs_c, mask = mask_c, other = 0)
        num_refs = tl.max(cnt)

        acc = tl.zeros([BLOCK_L, BLOCK_C], dtype = tl.float32)
        for j in range(num_refs):
            s = tl.load(ref_slot_ptr + ref_base + j * UNIQ_STRIDE, mask = mask_c, other = 0).to(tl.int64)
            p = tl.load(ref_pt_ptr + ref_base + j * UNIQ_STRIDE, mask = mask_c, other = 0.0)

            # ratio[slot, latent] out of a ~1 MB (L2-resident) scratch. Load it [C, L] so the LATENT axis
            # is innermost and each (cat, ref) reads a contiguous run, then transpose in registers --
            # loading it [L, C] puts `slot` innermost, making every lane its own sector (3x slower).
            r = tl.load(ratio_ptr + s[:,None] * num_latents + offs_l[None,:],
                        mask = mask_c[:,None] & mask_l[None,:], other = 0.0) # [BLOCK_C, BLOCK_L]
            r_t = tl.trans(r)                                                # [BLOCK_L, BLOCK_C]

            acc += r_t * p[None,:]

            if update_extflows:
                # expected-value term of the external evidence gradient, from the beta already in registers
                part = tl.sum(r_t * beta, axis = 0)                          # [BLOCK_C]
                goff = tl.load(ref_goff_ptr + ref_base + j * UNIQ_STRIDE, mask = mask_c, other = 0)
                tl.atomic_add(categorical_evidence_logp_grad_ptr + goff, -p * part,
                              mask = mask_c & (p != 0.0))

        pf_base = tl.load(pf_base_ptr + pid_g * num_latents + offs_l, mask = mask_l, other = 0)
        optr = param_flows_ptr + pf_base[:,None] + tot_num_cats + cats[None,:]
        # One owner per (row, category), so this is a plain read-modify-write.
        # :note: `tl.atomic_add` here is 1.5x SLOWER (2.09 -> 3.22 ms) even though the equivalent
        #        `atomicAdd` in the CUDA kernel is 1.2x FASTER -- Triton does not lower it to a bare
        #        RED (reduction) instruction. That asymmetry is the reason the CUDA path exists.
        tl.store(optr, tl.load(optr, mask = m, other = 0.0) + beta * acc, mask = m)

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
        expars_ptr = categorical_evidence_logp_ptr + \
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
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        if TILE_SIZE_K > 1:
            num_iters = tlmath.ceil(max_num_cats / TILE_SIZE_K).to(tl.int64)

            cat_ids = tl.arange(0, TILE_SIZE_K)

            for i in range(num_iters):
                cat_mask = mask[:,None] & missing_mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

                p_offsets = s_pids[:,None] + cat_ids[None,:]
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                pf_offsets = s_pfids[:,None] + cat_ids[None,:]
                tl.atomic_add(param_flows_ptr + pf_offsets, flows[:,None] * param, mask = cat_mask)

                cat_ids += TILE_SIZE_K
        else:
            for cat_id in range(max_num_cats):
                cat_mask = mask & missing_mask & (cat_id < num_cats)

                p_offsets = s_pids + cat_id
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                pf_offsets = s_pfids + cat_id
                tl.atomic_add(param_flows_ptr + pf_offsets, flows * param, mask = cat_mask)

    @staticmethod
    def bk_dual_flow_mask_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr,
                             s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE, TILE_SIZE_K):
        # Get `num_cats` from `metadata`
        s_mids = tl.load(s_mids_ptr + local_offsets, mask = mask, other = 0)
        num_cats = tl.load(metadata_ptr + s_mids, mask = mask, other = 0).to(tl.int64)

        max_num_cats = tl.max(num_cats, axis = 0)

        if TILE_SIZE_K > 1:
            num_iters = tlmath.ceil(max_num_cats / TILE_SIZE_K).to(tl.int64)

            cat_ids = tl.arange(0, TILE_SIZE_K)

            for i in range(num_iters):
                cat_mask = mask[:,None] & missing_mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

                p_offsets = s_pids[:,None] + cat_ids[None,:]
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                # Anchor = (scale·)TD·β, already carrying `scale=(1-step_size)` via `flows`.
                # Add the SAME anchor to both phases so ratio -> 1 at s=0 (no update).
                anchor = flows[:,None] * param

                pf_offsets = s_pfids[:,None] + cat_ids[None,:]
                tl.atomic_add(param_flows_ptr + pf_offsets,                    anchor, mask = cat_mask)  # F⁺
                tl.atomic_add(param_flows_ptr + pf_offsets + num_cats[:,None], anchor, mask = cat_mask)  # F⁻

                cat_ids += TILE_SIZE_K
        else:
            for cat_id in range(max_num_cats):
                cat_mask = mask & missing_mask & (cat_id < num_cats)

                p_offsets = s_pids + cat_id
                param = tl.load(params_ptr + p_offsets, mask = cat_mask, other = 0)

                anchor = flows * param

                pf_offsets = s_pfids + cat_id
                tl.atomic_add(param_flows_ptr + pf_offsets,            anchor, mask = cat_mask)  # F⁺
                tl.atomic_add(param_flows_ptr + pf_offsets + num_cats, anchor, mask = cat_mask)  # F⁻

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

    @staticmethod
    def small_ncats_dual_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
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

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids + num_cats + cat_id, mask = cat_mask, other = 0)

            # MAP M-step (Dirichlet prior): denominator is F- + pseudocount*beta, NOT F- + pseudocount.
            # This is the multiplicative form of beta = (F+ + pc/K)/(lambda + G-); the pc*beta term
            # floors never-observed categories (F+=0 => beta settles at (pc/K)/(G-+pc), independent
            # of beta) so they can't underflow, and reduces exactly to (F+ + pc/K)/(sum F+ + pc) when
            # p_theta is uniform (F- = beta*Gamma).
            flow = param * (flow_num + numerate_pseudocount) / (flow_denom + pseudocount * param)
            # Padding lanes (cat_id >= num_cats) load param/flows as 0 => 0/0 = NaN; zero them out.
            flow = tl.where(cat_mask, flow, 0.0)

            if keep_zero_params:
                cum_flow += tl.where(param < 1e-12, 0.0, flow)
            else:
                cum_flow += flow

        cum_flow = (1.0 - step_size) + step_size * cum_flow

        # Parameter update
        for cat_id in range(max_num_cats):
            cat_mask = mask & (cat_id < num_cats)

            param = tl.load(params_ptr + s_pids + cat_id, mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids + cat_id, mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids + num_cats + cat_id, mask = cat_mask, other = 0)

            new_param = param * ((1.0 - step_size) + step_size * (flow_num + numerate_pseudocount) / (flow_denom + pseudocount * param)) / cum_flow

            # Numerical guard: the MAP denominator self-floors only when F- is computed fresh from
            # the current beta. Under flow momentum, a stale (larger) F- is divided against a fast-
            # collapsing beta, so dying categories underflow float32 -> NaN. Clamp far below any real
            # token probability (~1e-30, ~8 orders above float32 underflow); LL-neutral.
            new_param = tl.maximum(new_param, 1e-30)

            if keep_zero_params:
                new_param = tl.where(param < 1e-12, 0.0, new_param)

            tl.store(params_ptr + s_pids + cat_id, new_param, mask = cat_mask)

    @staticmethod
    def large_ncats_dual_em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
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

            param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids[:,None] + num_cats[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            # MAP M-step (Dirichlet prior): denominator F- + pseudocount*beta (see small_ncats variant).
            flow = param * (flow_num + numerate_pseudocount[:,None]) / (flow_denom + pseudocount * param)
            # Padding lanes (cat_id >= num_cats) load param/flows as 0 => 0/0 = NaN; zero them out.
            flow = tl.where(cat_mask, flow, 0.0)

            if keep_zero_params:
                cum_flow += tl.sum(tl.where(param < 1e-12, 0.0, flow), axis = 1)
            else:
                cum_flow += tl.sum(flow, axis = 1)

            cat_ids += 128

        cum_flow = (1.0 - step_size) + step_size * cum_flow

        # Parameter update
        cat_ids = tl.arange(0, 128)
        for cat_sid in range(0, max_num_cats, 128):
            cat_mask = mask[:,None] & (cat_ids[None,:] < num_cats[:,None])

            param = tl.load(params_ptr + s_pids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_num = tl.load(param_flows_ptr + s_pfids[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)
            flow_denom = tl.load(param_flows_ptr + s_pfids[:,None] + num_cats[:,None] + cat_ids[None,:], mask = cat_mask, other = 0)

            new_param = param * ((1.0 - step_size) + step_size * (flow_num + numerate_pseudocount[:,None]) / (flow_denom + pseudocount * param)) / cum_flow[:,None]

            # Numerical guard against momentum-induced underflow (see small_ncats variant).
            new_param = tl.maximum(new_param, 1e-30)

            if keep_zero_params:
                new_param = tl.where(param < 1e-12, 0.0, new_param)

            tl.store(params_ptr + s_pids[:,None] + cat_ids[None,:], new_param, mask = cat_mask)

            cat_ids += 128

    def _get_constructor(self):
        return SoftEvidenceCategorical, {"num_cats": self.num_cats, "_dual_flow_backward": self._dual_flow_backward}

    def __reduce__(self):
        return (self.__class__, (self.num_cats, self._dual_flow_backward))
