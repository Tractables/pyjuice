"""
:class:`~pyjuice.nodes.distributions.SoftEvidenceCategorical` input layers. Its per-sample evidence weighs token ``v`` of
variable ``i`` by ``w(v) = exp(E[b, i, v])`` (``categorical_evidence_logp``: over the first ``K`` tokens, or with
``soft_evidence_cat_ids`` over those candidates; the other tokens weigh nothing), which makes a node's emission the
product of experts ``q_n(v) = p_n(v) w(v) / Z_n``, ``Z_n = sum_v p_n(v) w(v)``.

``forward`` is the layer's own. ``backward`` gives every target the posterior of the measure the forward marginalized
there -- ``q_n`` where the token is missing, ``p_n w`` where a value mask frees it (its value is ``log Z_n``) -- which
from the node flows ``F`` is, in both cases,

    P(x_t = v | e)  proportional to  sum_n F(n) p_n(v) w(v) / Z_n

over the input nodes ``n`` of ``x_t``. ``log Z_n`` is what the layer's own forward writes at a free position (its
value-mask forward), and the sum over the nodes is one matmul with the emission table per group of targets whose nodes
read the same emission rows.
"""

from __future__ import annotations

import torch
from typing import Optional

from .common import emission_sums, layer_facts, target_weights, targets


def forward(layer, inputs: torch.Tensor, node_mars: torch.Tensor,
            missing_mask: Optional[torch.Tensor] = None, **kwargs):

    assert inputs.dim() == 2, "`SoftEvidenceCategorical` takes token ids, [B, num_vars]; its soft evidence is " \
                              "`categorical_evidence_logp`."
    inputs = inputs.permute(1, 0).contiguous()

    layer.forward(data = inputs, node_mars = node_mars, missing_mask = missing_mask, **kwargs)


def _log_normalizers(layer, inputs: torch.Tensor, node_mars: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """``log Z_n`` of every node of ``layer`` for every sample, ``[layer nodes, B]``: the layer's own forward with every
    position free (value mask False), where the token is ignored -- so it reads zeros for data."""
    sid, eid = layer._output_ind_range
    batch_size = node_mars.size(1)
    evidence = {k: kwargs[k] for k in ("categorical_evidence_logp", "soft_evidence_cat_ids", "sort_soft_evidence")
                if kwargs.get(k, None) is not None}
    ext_num_vars = evidence["categorical_evidence_logp"].size(1)

    scratch = torch.empty([eid, batch_size], dtype = node_mars.dtype, device = node_mars.device)
    data = torch.zeros([inputs.size(1), batch_size], dtype = torch.long, device = node_mars.device)
    free = torch.zeros([batch_size, ext_num_vars], dtype = torch.bool, device = node_mars.device)
    layer.forward(data = data, node_mars = scratch, soft_evidence_value_mask = free, **evidence)

    return scratch[sid:eid]


def backward(layer, inputs: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
             params: Optional[torch.Tensor] = None, **kwargs):

    assert kwargs.get("logspace_flows", False), "The conditional backward reads log-space flows (`logspace_flows = True`)."
    if kwargs.get("categorical_evidence_logp", None) is None:
        raise ValueError("A conditional query of a `SoftEvidenceCategorical` layer needs its `categorical_evidence_logp`.")
    if kwargs.get("categorical_evidence_logp_grad", None) is not None:
        raise ValueError("A conditional query does not compute `categorical_evidence_logp_grad`.")

    if params is None:
        params = layer.params

    num_cats = layer.dist.num_cats
    f = layer_facts(layer)
    if "softevi_uniform" not in f.extra:
        f.extra["softevi_uniform"] = all(ns.dist.num_cats == num_cats for ns in layer.nodes)
    if not f.extra["softevi_uniform"]:
        raise NotImplementedError("A conditional query of a `SoftEvidenceCategorical` layer whose nodes have "
                                  "different `num_cats`.")

    sid, eid = layer._output_ind_range
    batch_size = node_flows.size(1)
    device = node_flows.device

    t = targets(layer, kwargs.get("target_vars"), device)
    nodes, rows_of, num_target_vars = t.nodes, t.rows_of, t.num_targets   # rows_of: every target node's output row
    vids = layer.vids[nodes, 0]
    lvids = vids if layer.var_idmapping is None else layer.var_idmapping[vids]

    # Every target node's weight F(n) / Z_n -- a node no token weighs (Z_n = 0) contributes nothing
    log_z = _log_normalizers(layer, inputs, node_mars, kwargs)[nodes]       # [target nodes, B]
    a = node_flows[sid + nodes] - log_z
    a = torch.where(torch.isneginf(log_z), -float("inf"), a)
    weights = target_weights(a, rows_of, num_target_vars)

    s_pids, rows = t.host if t.host is not None else (layer.s_pids[nodes], rows_of)
    sums = emission_sums(params, s_pids, rows, weights, num_target_vars, num_cats)

    # Every target's evidence row (a target the layer has no node of reads row 0, and its sums are zero)
    lvid_of = torch.zeros([num_target_vars], dtype = torch.long, device = device).scatter_(0, rows_of, lvids)
    evidence = kwargs["categorical_evidence_logp"][:, lvid_of, :]           # [B, targets, K]
    cat_ids = kwargs.get("soft_evidence_cat_ids", None)

    if cat_ids is None:
        K = evidence.size(2)
        logits = torch.full([batch_size, num_target_vars, num_cats], -float("inf"), dtype = sums.dtype, device = device)
        logits[:, :, :K] = sums[:, :, :K].log() + evidence
    else:
        ids = cat_ids[:, lvid_of, :].long()                                 # [B, targets, K]
        logits = sums.gather(2, ids).log() + evidence

    m = logits.amax(dim = 2, keepdim = True)
    probs = torch.exp(logits - torch.where(torch.isfinite(m), m, 0.0))
    probs /= (probs.sum(dim = 2, keepdim = True) + 1e-12)

    if cat_ids is not None:
        probs = torch.zeros([batch_size, num_target_vars, num_cats], dtype = probs.dtype, device = device) \
                     .scatter_(2, ids, probs)

    return probs
