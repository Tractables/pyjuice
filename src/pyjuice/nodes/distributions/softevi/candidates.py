"""
The candidate order of top-k soft evidence for
:class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical`: the public
:func:`sort_soft_evidence_candidates`, and the copy in category-id order the kernels read, cached per step.
"""

from __future__ import annotations

import torch

from .common import _evidence_cache_hit, _evidence_cache_key


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
    0.06 ms per step.

    Semantically a no-op, but NOT bit-exact: permuting the candidate axis permutes the order the
    logsumexp accumulates in, so float32 non-associativity shows. Measured over 180 configurations,
    the input layer moves by up to 1.4e-5 in log space and the root log-likelihood by up to 6.1e-5 on
    an |LL| of ~675 (relative ~1e-7) -- below the run-to-run spread the inner layers already have.
    Do not write an exact-equality assertion against an unsorted reference.

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


def _sorted_soft_evidence(layer, kwargs):
    """
    This step's candidate arrays in category-id order, cached on the layer by input identity.

    The cache is not (only) about avoiding a repeated sort -- it is about returning the SAME tensor
    objects every time. `_build_dense_index` and `_fw_linear_evidence` are keyed by tensor identity, so
    handing them a freshly sorted pair on each call would silently cost a rebuild per call.
    """
    evidence = kwargs["categorical_evidence_logp"]
    cat_ids = kwargs["soft_evidence_cat_ids"]

    cached = getattr(layer, "_fw_sorted_cache", None)
    if cached is not None and _evidence_cache_hit(cached[0], (evidence, cat_ids)):
        return cached[1]

    key = _evidence_cache_key(evidence, cat_ids)
    out = sort_soft_evidence_candidates(evidence, cat_ids)
    layer._fw_sorted_cache = (key, out)
    return out
