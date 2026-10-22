"""
The dense, index-driven path of :class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical`:
whether it applies -- to the backward's expected-category flow and to the CUDA forward -- and is worth it (measured),
and the inverted index it runs on, the (position, batch, candidate) slots grouped by category into length-bucketed
shards. The note below explains the two flow phases and why the dense kernel inverts the scatter.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import math
import os

from .common import _evidence_cache_hit, _evidence_cache_key, _grad_scratch, tlmath


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
# Elsewhere (here, the kernels, the backward) the two phases are written `F+` / `F-`, or "numerator" /
# "denominator".
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
# (row, cat) a single owner (or a few, once a long list is cut into pieces, see `_DENSE_LIST_SPLIT`):
#     phase1[row, c] = beta[l, c] * sum over the slots referencing c of ratio[slot, l] * p_theta[slot]
# The per-category reference lists are padded to a common `MAX_REFS` and masked, and both kernels bound
# their inner loop by the longest list actually present rather than by that width. The same
# `beta` read also serves the external-evidence gradient's expected-value term, so that is folded into
# the same walk rather than paying for the reads twice.
#
# Measured on the CoDD config (seq 32, 1024 latents, 126464 cats, top-k 1024, batch 8, one param-flow
# block), whole-PC backward: 65.7 ms -> 5.3 ms. Being index-driven it is also insensitive to the order of
# the candidate axis, unlike the scattered kernel it replaces.

_DENSE_TOPK_BACKWARD = os.environ.get("PYJUICE_SOFTEVI_DENSE_BACKWARD", "1") != "0"


# The longest reference list either kernel walks: a category referenced by more soft-evidence slots than
# this is cut into pieces (`_build_dense_index`), each its own list, accumulated atomically. A top-k
# drawn from a language model puts common tokens near the top at nearly every position, so one category
# can be referenced by `batch * seq` slots -- unbounded in the batch -- while the median list is ~2 long.
# Walked by one thread each, those few long lists were pure latency; cut, they spread over threads like
# any other. It also bounds the reference tables' width (`MAX_REFS`, the longest piece rounded up to a
# power of two), so lists of any length are served: there used to be a ceiling past which the build gave
# up and the scattered kernel ran instead.
#
# MEASURED, CoDD PC fwd+bwd per step (two interleaved runs, tuned), for W = 128 / 64 / 32 / 16 / 8:
# batch 12: 9.0 / 7.5-7.7 / 7.4-7.7 / 7.5-7.6 / 7.7 ms; batch 32: 12.0 / 10.4-10.7 / 9.4-9.5 / 9.5 / 10.2.
# Batch 1-8, where no list is that long, are unchanged by it. Bounded this short, the pieces also make
# the warps nearly even on their own, and the tuner stopped picking the length-sharded layout.
_DENSE_LIST_SPLIT = 32


# The padded reference arrays are `12 * max_refs * Umax * num_blocks` bytes (three [max_refs, Umax]
# tables: slot, p_theta, grad offset). Heavily-overlapping candidate lists at a large batch can push
# that into the hundreds of MB, which is worth it -- but not unboundedly, and not at the cost of an OOM
# in a path that has a working fallback. Over budget, decline and let the scattered kernel run.
_DENSE_INDEX_BUDGET_FRAC = 1.0 / 16.0


# Length buckets used to pick a shard split: 1, 2, 4, ... 4096 -- covering every list length up to
# `_DENSE_LIST_SPLIT`.
_LADDER_N = 13


assert _DENSE_LIST_SPLIT <= (1 << (_LADDER_N - 1))


_WARP = 32


# Only split when the model says the inner loop drops to at most this fraction of its un-split cost.
# A shard costs an extra launch of both kernels and an extra reference table, so the split has to remove
# a large multiple of the work, not a marginal one. The bimodal case this exists for lands at 0.06-0.07;
# candidate lists that are all the same length -- where the shards would just be "the tail" and "almost
# everything" -- land at 0.50 and measured 3% SLOWER than leaving them alone, which is what sets the bar.
_DENSE_SHARD_GAIN = 0.4


# Length buckets per shard: lists within one shard differ in length by at most 2 ** this.
_SHARD_BUCKETS = 2


def _shard_sizes(st, shards, gi):
    """Categories in each shard, for block `gi`."""
    return [sum(st[gi][lo:hi + 1]) for lo, hi in shards]


def _choose_shards(st, ladder = None):
    """Partition the categories into shards by reference-list length: a list of `(lo, hi)` length-bucket
    ranges (bucket `b` holds lists of length `(2 ** (b - 1), 2 ** b]`), one per shard, or a single
    shard spanning every bucket.

    Both kernels give one category to one thread and loop over its list, so a warp runs to the longest
    list among its 32 lanes, and the lists are anything but uniform: a language model proposes its
    common tokens at nearly every position, so their lists grow like `batch * positions`, while the tail
    of the vocabulary is proposed once or twice. Ordered by category id the two interleave, and nearly
    every warp catches a long list. MEASURED on the CoDD PC (32 positions, top-k 1024): the warp-level
    inner loop was 11x the references actually present at batch 8 and 14x at batch 32 -- growing with
    the batch, which is what made both kernels super-linear in it.

    A LADDER of shards, `_SHARD_BUCKETS` length buckets each, makes every warp nearly homogeneous. The
    lengths are spread continuously (median 2, 99th percentile ~7.5 per batch element, maximum 32 per
    batch element at batch 32), so a single light/heavy split -- what this used to try -- cannot pay: one
    side always keeps the spread. Each shard keeps ASCENDING category order, so the `params` reads and
    `param_flows` writes stay coalesced within it. Cost: one launch of each kernel per shard.

    The figures above predate cutting lists into pieces of at most `_DENSE_LIST_SPLIT`, which bounds
    the spread by itself: since then the tuner has picked the single layout on the CoDD PC at every batch
    measured (1-32). The ladder stays a timed candidate for shapes where that is not enough.

    `ladder` True / False forces the layout (it is TIMED where it can be, see `_dense_choice`); None
    takes the ladder only when its inner loop, bounded above (every warp in a shard charged that shard's
    full width), is at most `_DENSE_SHARD_GAIN` of the exact single-shard one. That gate is a statement
    about the forward, whose inner loop dominates it; the backward, dominated by per-category memory
    traffic that sharding makes less coalesced, can prefer one shard at small batch."""
    full = [(0, _LADDER_N - 1)]
    if ladder is False or os.environ.get("PYJUICE_SOFTEVI_DENSE_SHARD", "1") == "0":
        return full

    # Plain Python arithmetic over the already-transferred counts: this runs on every build, and at
    # tens of microseconds a tensor op per candidate it was measurably denting the small configs.
    hist = [sum(row[i] for row in st) for i in range(_LADDER_N)]
    shards = []
    for lo in range(0, _LADDER_N, _SHARD_BUCKETS):
        present = [b for b in range(lo, min(lo + _SHARD_BUCKETS, _LADDER_N)) if hist[b] > 0]
        if present:
            shards.append((lo, max(present)))
    if len(shards) <= 1:
        return full
    if ladder:
        return shards

    work_single = sum(row[2 * _LADDER_N] for row in st)
    est = sum(-(-n // _WARP) * _WARP * (1 << hi)
              for row in st for n, (lo, hi) in zip(_shard_sizes([row], shards, 0), shards))
    if est >= _DENSE_SHARD_GAIN * work_single:
        return full
    return shards


def _dense_index_budget(dev):
    override = os.environ.get("PYJUICE_SOFTEVI_DENSE_INDEX_BYTES", None)
    if override is not None:
        return int(override)
    try:
        total = torch.cuda.get_device_properties(dev).total_memory
    except Exception:
        return 1 << 30
    return int(total * _DENSE_INDEX_BUDGET_FRAC)


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
    if not kwargs["dual_flow_backward"] and kwargs.get("categorical_evidence_logp_grad", None) is None:
        # Without the dual-flow denominator the walk's only remaining output is the evidence gradient's
        # expected-value term, so with no gradient requested there is nothing for it to do. WITH one, the
        # scattered kernel pays the same [batch, latent, candidate] gather of `params` for it that the
        # denominator did, and the same inversion removes it: the dense kernels run with `update_pflows`
        # off. CoDD shape, fwd+bwd, dual-flow off: 29.8 -> 12.7 ms at batch 14, 64.7 -> 27.4 at 32.
        return False
    if not _dense_worth_it(layer, kwargs):
        return False
    return _build_dense_index(layer, kwargs) is not None


def _dense_update_pflows(layer, kwargs):
    """Whether the dense kernels write param flows (both phases), or only the evidence gradient.

    `layer.param_flows is not None` for the same reason as in `_prep_args_apply_bk_softevi_kernel`: a
    `backward(compute_param_flows = False)` allocates no buffer to write into."""
    return kwargs["dual_flow_backward"] and (layer.param_flows is not None)


def _l2_bytes(layer):
    l2 = getattr(_l2_bytes, "_cached", None)
    if l2 is None:
        try:
            l2 = torch.cuda.get_device_properties(layer.params.device).L2_cache_size
        except Exception:
            l2 = 64 * 1024 * 1024
        _l2_bytes._cached = l2
    return l2


def _params_exceed_l2(layer, kwargs):
    """Whether the emission parameter table is too big to sit in L2 -- the regime every CUDA path here
    was tuned for. Below it, scattered reads are already cheap and Triton is the safer default."""
    lnn = layer._output_ind_range[1] - layer._output_ind_range[0]
    ext_num_vars = kwargs["categorical_evidence_logp"].size(1)
    if ext_num_vars == 0 or lnn % ext_num_vars != 0:
        return False
    num_latents = lnn // ext_num_vars
    return num_latents * layer.nodes[0].dist.num_cats * 4 > _l2_bytes(layer)


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

    The second condition is that the PARAMETERS MUST BE TIED across variables, which is the property the
    dense form really runs on. The scattered form always does `num_nodes * batch * k` operations. The
    dense form does `sum over param-flow groups of (latents * |unique categories in that group|)`, and
    because the unique-category count saturates (coupon collector), merging many variables into one
    group makes the dense form win big -- while untied parameters put every variable in its own group
    and the two converge. At 32 vars / 1024 latents / 126464 cats / k=1024 / batch 8:

        fully tied (1 group of 32 vars) -> ~113M dense slots vs 268M scattered ops -> dense wins
        untied    (32 groups of 1 var)  -> ~260M dense slots vs 268M scattered ops -> no gain, and the
                                           dense form would still pay to build its index

    All of this is host-side arithmetic: no device sync, and no index build for cases we then reject.

    That work ratio is only the DEFAULT, though -- the answer before a shape has been measured, and with
    autotuning off. It prices a random scattered atomic the same as a coalesced dense op, while the
    first costs ~9x more here, and it has no term for the index build's fixed cost per group. Both
    errors are large. MEASURED fwd+bwd, CoDD PC (tied, 32 vars, 1024 latents, 126464 cats, k=1024): at
    batch 1-4 it picks scattered, at 10.7-30.9 ms, while dense takes 5.5-5.9 ms. Yet for an untied
    8-var model at batch 1, dense was SLOWER (8.0 vs 4.6 ms: an index build per group, since vectorized
    over the groups -- 2.5 -> 0.5 ms of build at 8 groups). So the candidates
    are timed once per shape, by the first forward or backward that can, and the choice is cached
    (`_dense_choice`).
    """
    tot_num_cats = layer.nodes[0].dist.num_cats
    lnn = layer._output_ind_range[1] - layer._output_ind_range[0]
    evidence = kwargs["categorical_evidence_logp"]
    ext_num_vars = evidence.size(1)
    num_k = evidence.size(2)
    batch_size = evidence.size(0)
    if ext_num_vars == 0 or lnn % ext_num_vars != 0:
        return False
    num_latents = lnn // ext_num_vars

    # (1) What the scattered path touches at random: the emission parameters plus the phase-1 flow half
    footprint = 2 * num_latents * tot_num_cats * 4

    if footprint <= _l2_bytes(layer):
        return False

    # (2) Does parameter tying give the dense form a real work advantage?
    layout = _dense_layer_layout(layer, ext_num_vars)
    if layout is None:
        return False
    num_groups = len(layout[1])
    vars_per_group = max(1, ext_num_vars // num_groups)

    slots_per_group = vars_per_group * batch_size * num_k
    uniq_est = tot_num_cats * (1.0 - math.exp(-slots_per_group / tot_num_cats))
    dense_ops = num_groups * num_latents * uniq_est
    scattered_ops = lnn * batch_size * num_k

    return _dense_choice(layer, kwargs, dense_ops < 0.6 * scattered_ops)


def _dense_choice_key(layer, kwargs):
    """The shape key of the dense-vs-scattered choice, or None where the dense form cannot apply."""
    lnn = layer._output_ind_range[1] - layer._output_ind_range[0]
    evidence = kwargs["categorical_evidence_logp"]
    B, V, K = evidence.shape
    if V == 0 or lnn % V != 0:
        return None
    layout = _dense_layer_layout(layer, V)
    if layout is None:
        return None
    shape = (lnn, V, B, K, layer.nodes[0].dist.num_cats, len(layout[1]))
    # The flag comes from the distribution, not from `kwargs`: `set_custom_kernel_kwargs` copies it in
    # only AFTER `preprocess_fw_kwargs`, which consults this choice too, and a key that changed between
    # the two calls built the index twice per forward.
    return ("softevi_dense_vs_scattered",) + shape + (bool(layer.dist._dual_flow_backward),)


def _dense_choice(layer, kwargs, default):
    """Scattered, or dense with which index layout -- by measurement where possible (see the end of
    `_dense_worth_it`). Returns whether to take the dense form; the layout is read by `_index_layout`.

    Three candidates: "scattered", "single" (one category order, coalesced) and "ladder" (shards by
    list length, see `_choose_shards`). The two dense layouts trade against each other: the ladder makes
    the forward's inner loop balanced (3.2x on the CoDD forward at batch 32, before lists were cut into
    pieces -- see `_DENSE_LIST_SPLIT`), but splitting the categories costs the backward coalescing, so at
    small batch it is slower there. That crossover moves with the shape, hence timing.

    Two choices are kept, both keyed by SHAPE and cached process-wide in the launch autotuner:
      * the JOINT one (forward + backward), measured in a backward. Once it exists both passes follow
        it, because a forward that is followed by a backward is what it was measured on.
      * a FORWARD-ONLY one, measured in a forward while no joint one exists -- what a caller that never
        runs a backward (evaluation, conditional sampling) gets.
    With neither -- autotuning off, CUDA-graph capture -- the work-ratio `default` decides dense vs
    scattered, and `_choose_shards`' gate the layout. Like every tuned choice, the candidates agree only
    to rounding (~6e-5 relative here), so which one won is part of the numerics."""
    if getattr(layer, "_dense_choice_override", None) is None:
        from pyjuice.layer.kernels import autotune
        key = _dense_choice_key(layer, kwargs)
        if kwargs.get("_bk_launch_common", None) is not None:
            if autotune.should_tune(key, 3):
                _tune_dense_choice(layer, kwargs, key, default, backward = True)
        elif kwargs.get("_fw_launch_common", None) is not None and autotune.cached(key) is None:
            if autotune.should_tune(key + ("forward_only",), 3):
                _tune_dense_choice(layer, kwargs, key + ("forward_only",), default, backward = False)

    choice = _cached_dense_choice(layer, kwargs)
    return default if choice is None else (choice != "scattered")


def _cached_dense_choice(layer, kwargs):
    """The choice in force: the one being timed, else the joint one, else the forward-only one; None
    while there is none."""
    override = getattr(layer, "_dense_choice_override", None)
    if override is not None:
        return override
    key = _dense_choice_key(layer, kwargs)
    if key is None:
        return None
    from pyjuice.layer.kernels import autotune
    choice = autotune.cached(key)
    return choice if choice is not None else autotune.cached(key + ("forward_only",))


def _index_layout(layer, kwargs):
    """The dense index layout to build: True (ladder), False (single) or None (let `_choose_shards`'
    gate decide), per `_cached_dense_choice`."""
    return {"ladder": True, "single": False}.get(_cached_dense_choice(layer, kwargs), None)


def _tune_dense_choice(layer, kwargs, key, default, backward):
    """Time each candidate and cache the winner under `key`; returns it.

    A candidate is timed by REPLAYING the input layer's own post-processing lists with the choice forced
    (`_dense_choice_override`) and every output redirected to scratch: the forward's into a scratch
    `node_mars`, and in a backward the backward's too, into scratch parameter flows and evidence
    gradient. So what is timed is exactly what a step runs under that choice -- the candidate sort the
    scattered forward takes, the Triton fallbacks without a CUDA toolchain, a `missing_mask`, a
    `backward(compute_param_flows = False)` with no parameter flows at all. Re-implementing the passes
    here instead got all three of those wrong at one time or another."""
    from pyjuice.layer.kernels import autotune

    # Only the dense layouts whose index can actually be built here: one that cannot would time as the
    # scattered path it falls back to, and could then win on noise and pay a failed build every step.
    cands = ["scattered"] + [name for name, ladder in (("single", False), ("ladder", True))
                             if _build_dense_index(layer, kwargs, ladder = ladder) is not None]
    if len(cands) == 1:
        return "scattered"
    # The reference -- kept unless beaten by the autotuner's margin -- is what would run otherwise: a
    # choice already in force (the forward-only one, when the joint one is being timed), else the
    # untuned default with the gate's layout.
    current = _cached_dense_choice(layer, kwargs)
    if current in cands:
        reference = current
    elif default:
        gate = _build_dense_index(layer, kwargs, ladder = None)
        reference = "ladder" if (gate is not None and len(gate["shards"]) > 1 and "ladder" in cands) else cands[1]
    else:
        reference = "scattered"
    candidates = [reference] + [c for c in cands if c != reference]

    if backward:
        bk = kwargs["_bk_launch_common"]
        nm_scr = autotune.scratch_like(bk["node_mars_ptr"])
        pf_scr = autotune.scratch_like(layer.param_flows) if layer.param_flows is not None else None
        grad = kwargs.get("categorical_evidence_logp_grad", None)
        # float32, so the kernels accumulate straight into it and `_GradCastEpilogue` stays out of it
        grad_scr = autotune.scratch_like(_grad_scratch(layer, grad)) if grad is not None else None
        if nm_scr is None or (pf_scr is None and layer.param_flows is not None) or \
                (grad_scr is None and grad is not None):
            return reference                           # no room to benchmark: leave it untuned
        fw_common = layer._fw_launch_common(bk["data_ptr"], nm_scr)
        bk_common = dict(bk, param_flows_ptr = pf_scr)
    else:
        fw = kwargs["_fw_launch_common"]
        nm_scr = autotune.scratch_like(fw["node_mars_ptr"])
        if nm_scr is None:
            return reference
        fw_common = dict(fw, node_mars_ptr = nm_scr)

    def run(choice):
        layer._dense_choice_override = choice
        try:
            # Charge each candidate the per-step work it causes: a step sorts the candidates (when its
            # forward does) and builds the index once, shared by the forward and the backward.
            layer._dense_index_cache = None
            layer._fw_sorted_cache = None
            fkw = dict(kwargs, _fw_launch_common = fw_common)
            fkw.pop("_bk_launch_common", None)
            layer.dist.preprocess_fw_kwargs(layer, fkw)
            layer.dist.set_custom_kernel_kwargs(fkw)
            fkw["_fw_data"] = fw_common["data_ptr"]
            layer._run_post(layer.post_fw_fns, fkw, fw_common)
            if backward:
                bkw = dict(kwargs, _bk_launch_common = bk_common)
                if grad is not None:
                    bkw["categorical_evidence_logp_grad"] = grad_scr
                layer._run_post(layer.post_bp_fns, bkw, bk_common)
        finally:
            layer._dense_choice_override = None

    try:
        return autotune.pick(key, candidates, run)
    finally:
        # The live call rebuilds for the winner.
        layer._dense_index_cache = None
        layer._fw_sorted_cache = None


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

    Returns `(num_latents, groups, lvid_of_head, pf_base, p_base, var_group)`, or None if the layout is
    not one the dense kernels support (the caller then falls back to the scattered kernel). `var_group`
    maps each evidence variable (the middle axis of `categorical_evidence_logp`) to its block in
    `groups`.
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

    lvid_host = lvid_of_head.tolist()
    # One run per evidence variable: the ratio scratch is indexed by (variable, batch) slot, so two runs
    # of one variable would share a slot. Decline rather than mix them.
    if sorted(lvid_host) != list(range(num_ext_vars)):
        layer._dense_layout_cache = ((lnn, num_ext_vars), None)
        return None
    var_group = [0] * num_ext_vars
    for gi, g in enumerate(groups):
        for i in g:
            var_group[lvid_host[i]] = gi
    var_group = torch.tensor(var_group, dtype = torch.long, device = dev)

    layout = (num_latents, groups, lvid_of_head, pf_base, p_base, var_group)
    layer._dense_layout_cache = ((lnn, num_ext_vars), layout)
    return layout


@triton.jit
def _dense_index_scatter_kernel(order_ptr, list_ptr, list_start_ptr, first_piece_ptr, piece_dst_ptr, piece_us_ptr,
                                piece_us, evidence_ptr, ref_slot_ptr, ref_pt_ptr, ref_goff_ptr, num_refs,
                                B, V, K, W: tl.constexpr, SPLIT: tl.constexpr, US_PER_PIECE: tl.constexpr,
                                BLOCK: tl.constexpr):
    """Pass 2 of `_build_dense_index`, one reference per lane: where the `i`-th reference in (block,
    category) order goes in its shard's [G, width, Us] tables, and what it holds there.

    `order[i]` is the reference's element index e = (b, v, k) in row-major [B, V, K] order; `list[i]` its
    category list, which starts at `list_start`. Its position j in that list picks the piece -- the
    `j // W`-th of the list, starting at `first_piece` -- and the row within it, `j % W`. A piece's row 0
    sits at `piece_dst`, and rows are `us` apart (the tables are [width, Us]: see the build).

    Stored: the slot `v * B + b` of the ratio scratch, `exp(evidence[e])` widened to float32, and e itself,
    which IS the offset into `categorical_evidence_logp_grad`."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_refs

    e = tl.load(order_ptr + offs, mask = mask, other = 0)
    lst = tl.load(list_ptr + offs, mask = mask, other = 0)
    j = offs - tl.load(list_start_ptr + lst, mask = mask, other = 0)
    if SPLIT:
        piece = tl.load(first_piece_ptr + lst, mask = mask, other = 0) + j // W
        j = j % W
    else:
        piece = lst
    if US_PER_PIECE:
        us = tl.load(piece_us_ptr + piece, mask = mask, other = 0)
    else:
        us = piece_us
    dst = tl.load(piece_dst_ptr + piece, mask = mask, other = 0) + j * us

    t = e // K
    tl.store(ref_slot_ptr + dst, ((t % V) * B + t // V).to(tl.int32), mask = mask)
    # libdevice's exp, the accurate one torch uses: `tl.exp` is the fast approximation, off by a few ulp
    pt = tlmath.exp(tl.load(evidence_ptr + e, mask = mask, other = 0.0).to(tl.float32))
    tl.store(ref_pt_ptr + dst, pt, mask = mask)
    tl.store(ref_goff_ptr + dst, e.to(tl.int32), mask = mask)


def _build_dense_index(layer, kwargs, ladder = None):
    """
    Invert (position, batch, candidate) -> category, once per step, per param-flow block.

    Returns a dict of device tensors (padded over blocks so one launch covers all of them), or None if
    the layout is not supported -- in which case the caller falls back to the scattered kernel.

    `ladder` picks the layout: True shards the categories by list length (`_choose_shards`), False keeps
    them in one piece, None defers to `_index_layout` (the tuned choice, else `_choose_shards`' gate).

    Cached on the layer and keyed by the identity of `soft_evidence_cat_ids` (and the layout), so the
    forward and the backward of a step share one build.
    """
    cat_ids = kwargs["soft_evidence_cat_ids"]
    evidence = kwargs["categorical_evidence_logp"]
    if ladder is None:
        ladder = _index_layout(layer, kwargs)

    # `batch_size` is injected into `kwargs` only after the condition check, so take it from the tensor
    # (the layer asserts these agree in the prep functions).
    #
    # Keyed by tensor IDENTITY, not by address -- see `_evidence_cache_key` for why a `data_ptr()` key
    # silently serves the previous call's index once the allocator recycles a block.
    key = _evidence_cache_key(cat_ids, evidence)
    cached = getattr(layer, "_dense_index_cache", None)
    if cached is not None and _evidence_cache_hit(cached[0], (cat_ids, evidence)):
        if ladder in cached[1]:
            return cached[1][ladder]
    else:
        cached = layer._dense_index_cache = (key, {})

    B, V, K = cat_ids.shape
    dev = cat_ids.device

    layout = _dense_layer_layout(layer, V)
    if layout is None:
        return None
    num_latents, groups, lvid_of_head, pf_base, p_base, var_group = layout
    G = len(groups)
    num_cats = layer.nodes[0].dist.num_cats

    # ---- pass 1: sort every block's references by category, all blocks at once ----
    # One sort on a (block, category) key rather than one sort per block. The per-block form cost a
    # `unique_consecutive` device sync and a dozen small launches PER BLOCK, so an untied model (a block
    # per variable) paid ~`V` times the host time of a tied one for the same references -- 2.5 ms at 8
    # blocks, 5.0 ms at 16, against 0.6 ms now -- and every shard of the ladder added another round of
    # them. The whole build is a fixed number of launches and three host round trips.
    #
    # Element e = (b, v, k), in row-major [B, V, K] order. Its offset into
    # `categorical_evidence_logp_grad` IS e, and its slot in the [V * B, num_latents] ratio scratch is
    # `v * B + b`.
    E = B * V * K
    # 32-bit keys whenever they fit: the sort is the single most expensive step of the build.
    kdt = torch.int32 if G * num_cats < 2 ** 31 else torch.long
    key = cat_ids.to(kdt)
    if G > 1:
        key = key + (var_group.to(kdt) * num_cats).view(1, V, 1)
    # Stable, so each list holds its references in element order and the build is reproducible.
    key_s, order = torch.sort(key.reshape(-1), stable = True)

    # `return_inverse` gives the per-element list directly, which avoids two `repeat_interleave`s
    ukey, row, counts = torch.unique_consecutive(key_s, return_inverse = True, return_counts = True)
    list_start = counts.cumsum(0) - counts                                       # first reference of each list

    # A category's reference list, cut into PIECES of at most `_DENSE_LIST_SPLIT` references; each piece
    # is a "list" from here on, so one category may own several consecutive ones. Both kernels give a list
    # to one thread and walk it serially, and a token proposed at nearly every position has a list of
    # `batch * positions` references: unsplit, those few lists were a shard of their own whose cost was
    # all latency. MEASURED on the CoDD PC at batch 17: 11 lists of 513-544 references, 1.05 ms of a
    # 9.2 ms step for 11 categories -- the batch-17 cliff. Pieces are accumulated atomically everywhere
    # (see `PF_ATOMIC`).
    #
    # The pieces are derived arithmetically (a list of length c makes c // W full pieces and one of c % W),
    # so their histograms come straight from the lists' lengths and the build still costs two host round
    # trips: this one, for the number of lists, and the statistics below, which also bring back the
    # number of pieces.
    W = _DENSE_LIST_SPLIT
    full = counts // W
    rem = counts - full * W
    n_pieces = full + (rem > 0).long()
    g_r = ukey.long() // num_cats                                                # block of each list

    # Everything the shard decision needs, in one transfer: per block, how many pieces and how many
    # references fall in each power-of-two length bucket; the longest list; and the number of pieces.
    gb_full = g_r * _LADDER_N + (W - 1).bit_length()                             # a full piece: length W
    gb_rem = g_r * _LADDER_N + torch.ceil(torch.log2(rem.clamp(min = 1).to(torch.float64))).long()
    hist = torch.zeros(G * _LADDER_N, dtype = torch.long, device = dev)
    hist.scatter_add_(0, gb_full, full).scatter_add_(0, gb_rem, (rem > 0).long())
    rhist = torch.zeros(G * _LADDER_N, dtype = torch.long, device = dev)
    rhist.scatter_add_(0, gb_full, full * W).scatter_add_(0, gb_rem, rem)
    flat = torch.cat([hist, rhist, counts.max().view(1), n_pieces.sum().view(1)]).tolist()
    st = [flat[gi * _LADDER_N:(gi + 1) * _LADDER_N] + flat[(G + gi) * _LADDER_N:(G + gi + 1) * _LADDER_N]
          for gi in range(G)]
    longest, U = flat[2 * G * _LADDER_N], flat[2 * G * _LADDER_N + 1]
    split = longest > W                                                          # some category owns several pieces
    max_refs = 1 << max(0, min(longest, W) - 1).bit_length()                    # round up to a power of two

    first_piece = None
    if split:
        lst = torch.repeat_interleave(torch.arange(ukey.numel(), device = dev), n_pieces, output_size = U)
        first_piece = n_pieces.cumsum(0) - n_pieces
        k = torch.arange(U, device = dev) - first_piece[lst]                     # which piece of its list
        ukey = ukey[lst]
        counts = (counts[lst] - k * W).clamp_(max = W)
    g_l = ukey.long() // num_cats                                                # block of each piece
    # The pieces are in (block, category) order, so a block's first piece is where its id first appears.
    g_start = torch.searchsorted(g_l, g_l)
    pos = torch.arange(U, device = dev) - g_start                               # position in its block
    bidx = torch.ceil(torch.log2(counts.to(torch.float64))).long().clamp_(0, _LADDER_N - 1)

    if ladder is None:
        # Only the untuned gate needs this, and it needs the pieces in order: the inner-loop cost of leaving
        # them in one shard. That cost is per WARP -- both kernels give a piece to one thread and loop over
        # it, so 32 lanes run to the longest piece among them. A block's warps start at its first piece;
        # numbering them from `g_start // 32 + g` keeps every block's range disjoint from the next one's in
        # at most `U / 32 + G` ids.
        wid = g_start // _WARP + g_l + pos // _WARP
        n_w = U // _WARP + G + 1
        w_max = torch.zeros(n_w, dtype = torch.long, device = dev).scatter_reduce_(0, wid, counts, "amax")
        w_blk = torch.zeros(n_w, dtype = torch.long, device = dev).scatter_(0, wid, g_l)
        work = torch.zeros(G, dtype = torch.long, device = dev).scatter_add_(0, w_blk, w_max * _WARP).tolist()
        st = [r + [w] for r, w in zip(st, work)]

    shard_ranges = _choose_shards(st, ladder)
    S = len(shard_ranges)
    # A shard is as wide as its longest possible list, capped at the longest list actually present
    shard_widths = [min(1 << hi, max_refs) for lo, hi in shard_ranges]
    shard_n = [_shard_sizes(st, shard_ranges, gi) for gi in range(G)]      # [G][S] lists per shard
    shard_us = [max(shard_n[gi][si] for gi in range(G)) for si in range(S)]

    if sum(12 * w * u * G for w, u in zip(shard_widths, shard_us)) > _dense_index_budget(dev):
        return None                                                          # -> scattered fallback

    # ---- pass 2: scatter every reference into its shard's [G, width, Us] tables, all at once ----
    # Every shard's tables are views into one flat buffer per field, so this is one indexed store per
    # field however many shards and blocks there are. Each shard starts on a 128-byte boundary, as its
    # own allocation used to.
    #
    # Filled IN PLACE and NOT zeroed. Every reader stops at a category's own count (`ref_cnt`): the CUDA
    # kernels loop `j < cnt`, the Triton fallback masks `j < cnt`. So the padding past a list's end is
    # never read. Zeroing it, plus the copy a per-block `torch.stack` made, scaled with `width` (the
    # LONGEST list, rounded up to a power of two -- a token proposed at every position makes that
    # `batch * positions`), not with the references actually present: on the CoDD PC it was ~1.4 ms
    # of a 10.9 ms fwd+bwd at batch 16, and 2.9 ms at batch 17, where `width` jumps from 512 to 1024.
    def _bases(sizes):
        out, acc = [], 0
        for n in sizes:
            out.append(acc)
            acc += -(-n // 32) * 32
        return out, acc
    rbase, r_tot = _bases([G * w * u for w, u in zip(shard_widths, shard_us)])
    ubase, u_tot = _bases([G * u for u in shard_us])

    # The small per-shard tables, in one host-to-device copy: lists per (shard, block), and for the
    # ladder the bucket -> shard map and each shard's table geometry.
    host = [shard_n[gi][si] for si in range(S) for gi in range(G)]
    if S > 1:
        b2s = [0] * _LADDER_N
        for si, (lo, hi) in enumerate(shard_ranges):
            for b in range(lo, hi + 1):
                b2s[b] = si
        host += b2s + rbase + shard_widths + shard_us + ubase
    host = torch.tensor(host, dtype = torch.long, device = dev)
    num_uniq = host[:(S * G)].view(S, G).int()

    if S == 1:
        rb, wd, us, ub = rbase[0], shard_widths[0], shard_us[0], ubase[0]
        rank = pos                                    # the lists are already in (block, category) order
    else:
        sid = host[(S * G):(S * G + _LADDER_N)][bidx]                        # shard of each list
        rb, wd, us, ub = host[(S * G + _LADDER_N):].view(4, S)[:, sid]
        # Rank within (shard, block). A stable sort on that pair keeps the ASCENDING category order the
        # lists already have inside each piece -- which is what makes the `params` reads and
        # `param_flows` writes coalesce.
        seg_s, perm = torch.sort(sid * G + g_l, stable = True)
        rank = torch.empty_like(perm)
        rank[perm] = torch.arange(U, device = dev) - torch.searchsorted(seg_s, seg_s)

    # [width, Us] rather than [Us, width]: the kernels assign one CATEGORY per thread, so with the
    # reference index innermost, lane-adjacent threads would read `width` apart and every lane would
    # fetch its own sector. Transposed, a warp's reads of reference `j` are contiguous. So a piece's row
    # 0 is at `piece_dst` and its rows are `us` apart.
    piece_dst = rb + g_l * (wd * us) + rank
    ref_slot = torch.empty(r_tot, dtype = torch.int32, device = dev)
    ref_pt = torch.empty(r_tot, dtype = torch.float32, device = dev)
    ref_goff = torch.empty(r_tot, dtype = torch.int32, device = dev)         # < B * V * K: 32 bits is plenty
    # One pass over the references (`_dense_index_scatter_kernel`), where a chain of torch ops took two
    # dozen launches. `ref_pt` is a float32 table both kernels read as float, so half-precision evidence
    # is widened here rather than in the kernels (see `_fw_linear_evidence`).
    _dense_index_scatter_kernel[(triton.cdiv(E, 1024),)](
        order, row, list_start, first_piece if split else list_start, piece_dst,
        us if S > 1 else piece_dst, us if S == 1 else 0, evidence.contiguous(), ref_slot, ref_pt, ref_goff,
        E, B, V, K, W = W, SPLIT = split, US_PER_PIECE = S > 1, BLOCK = 1024)

    u_dst = ub + g_l * us + rank
    uniq = torch.zeros(u_tot, dtype = torch.int32, device = dev)
    ref_cnt = torch.zeros(u_tot, dtype = torch.int32, device = dev)
    uniq[u_dst] = (ukey % num_cats).int() if G > 1 else ukey.int()
    ref_cnt[u_dst] = counts.int()

    shards = []
    for si, (w, u) in enumerate(zip(shard_widths, shard_us)):
        r0, u0 = rbase[si], ubase[si]
        shards.append(dict(
            uniq = uniq[u0:(u0 + G * u)].view(G, u),
            ref_slot = ref_slot[r0:(r0 + G * w * u)].view(G, w, u),
            ref_pt = ref_pt[r0:(r0 + G * w * u)].view(G, w, u),
            ref_goff = ref_goff[r0:(r0 + G * w * u)].view(G, w, u),
            ref_cnt = ref_cnt[u0:(u0 + G * u)].view(G, u),
            num_uniq = num_uniq[si],                                         # [G]
            Umax = u,
            max_refs = w,
        ))

    index = dict(
        shards = shards,
        pf_base = pf_base,
        p_base = p_base,
        num_blocks = G,
        num_latents = num_latents,
        # A category may own several lists: its parameter-flow update must then be atomic.
        split = split,
    )

    cached[1][ladder] = index
    return index


def _missing_mask_kwargs(kwargs):
    """Hand `backward`'s `missing_mask` to a post-processing kernel.

    `missing_mask_mode` is a `tl.constexpr`, and mode 0 means "no mask", so the unmasked case specializes
    back to exactly the code it compiled to before -- no extra load, no extra branch."""
    return {
        "missing_mask_ptr": kwargs.get("_missing_mask", None),
        "missing_mask_mode": kwargs.get("_missing_mask_mode", 0),
        "missing_mask_num_vars": kwargs.get("_missing_mask_num_vars", 0),
    }


def _fw_use_dense(layer, kwargs):
    """Whether the forward can use the index-driven form. It needs its shared tile to fit, and the same
    inverted index and tying / L2 criterion as the backward; otherwise the gather form still applies."""
    return _fw_dense_fits(layer, kwargs) and _dense_worth_it(layer, kwargs) and \
        _build_dense_index(layer, kwargs) is not None


# Latents per thread for the normalizer. Larger amortizes the per-reference index loads over more
# latents but grows the shared `Zs` tile, `num_slots * TL` floats; every one that fits the device's
# opt-in shared memory (`_fw_smem_cap`) is a candidate, and `_fw_autotune` times them.
_FW_TL_CANDIDATES = (4, 8, 16)


def _fw_smem_cap(layer):
    """Shared memory one block of the dense forward may use: the device's opt-in limit (99 KB on the
    RTX PRO 6000, 227 KB on an H100). It used to be a fixed 48 KB -- the limit without opting in -- and
    at batch * positions above 3072 slots no tile fit, so the forward crashed."""
    cap = getattr(_fw_smem_cap, "_cached", None)
    if cap is None:
        try:
            cap = torch.cuda.get_device_properties(layer.params.device).shared_memory_per_block_optin
        except Exception:
            cap = 48 * 1024
        _fw_smem_cap._cached = cap
    return cap


def _fw_dense_fits(layer, kwargs):
    """Whether the dense forward's smallest `Zs` tile fits. Past that it declines and the forward takes
    the Triton kernel -- while the BACKWARD's dense kernels, which keep no such tile, still apply."""
    evidence = kwargs["categorical_evidence_logp"]
    num_slots = evidence.size(0) * evidence.size(1)
    return num_slots * min(_FW_TL_CANDIDATES) * 4 <= _fw_smem_cap(layer)
