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

import numpy as np
import torch
from typing import Optional

from .common import target_nodes


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


def _emission_sums(params: torch.Tensor, s_pids: torch.Tensor, rows_of: torch.Tensor, weights: torch.Tensor,
                   num_targets: int, num_cats: int) -> torch.Tensor:
    """``[B, num_targets, num_cats]``: ``sum_n weights[n, b] p_n(:)`` over the nodes ``n`` of every target
    (``rows_of[n]`` its output row, ``s_pids[n]`` its parameters). One matmul per group of targets whose nodes read the
    same emission rows -- all of them when the emissions are tied, one per target when they are not -- so every row is
    read once per group and no product with a row a target does not read is formed."""
    batch_size = weights.size(1)
    device = weights.device

    # The emission table: `params` itself, viewed as rows, when every node's parameters start a row of it
    if params.numel() % num_cats == 0 and bool((s_pids % num_cats == 0).all()):
        table = params.view(-1, num_cats)
        rows = s_pids // num_cats
    else:
        uniq, rows = torch.unique(s_pids, return_inverse = True)
        table = params[uniq[:, None] + torch.arange(num_cats, device = device)[None, :]]

    rows_h, t_h = rows.cpu().numpy(), rows_of.cpu().numpy()
    groups = dict()
    for t in np.unique(t_h):
        groups.setdefault(tuple(np.unique(rows_h[t_h == t]).tolist()), []).append(int(t))

    sums = torch.zeros([batch_size, num_targets, num_cats], dtype = weights.dtype, device = device)
    for key, targets in groups.items():
        if key[-1] - key[0] + 1 == len(key):
            tab = table[key[0]:key[-1] + 1]                                  # a contiguous run of rows: a view
        else:
            tab = table[torch.tensor(key, device = device)]

        sel = np.nonzero(np.isin(t_h, targets))[0]
        sel_d = torch.from_numpy(sel).to(device)
        pos = torch.from_numpy(np.searchsorted(targets, t_h[sel])).to(device)            # target within the group
        col = torch.from_numpy(np.searchsorted(key, rows_h[sel])).to(device)             # row within the group

        R = torch.zeros([batch_size, len(targets), len(key)], dtype = weights.dtype, device = device)
        R.index_put_((torch.arange(batch_size, device = device)[:, None], pos[None, :], col[None, :]),
                     weights[sel_d].t(), accumulate = True)
        sums[:, targets, :] = (R.view(batch_size * len(targets), len(key)) @ tab).view(batch_size, len(targets), num_cats)

    return sums


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
    for ns in layer.nodes:
        if ns.dist.num_cats != num_cats:
            raise NotImplementedError("A conditional query of a `SoftEvidenceCategorical` layer whose nodes have "
                                      "different `num_cats`.")

    sid, eid = layer._output_ind_range
    batch_size = node_flows.size(1)
    device = node_flows.device
    num_vars = layer.vids.max().item() + 1

    nodes, rev_vars_mapping, num_target_vars = target_nodes(layer, kwargs.get("target_vars"), num_vars, device)
    vids = layer.vids[nodes, 0]
    rows_of = rev_vars_mapping[vids]                                         # every target node's output row
    lvids = vids if layer.var_idmapping is None else layer.var_idmapping[vids]

    # Every target node's weight F(n) / Z_n, shifted per (target, sample) by its largest -- a node no token weighs
    # (Z_n = 0) contributes nothing
    log_z = _log_normalizers(layer, inputs, node_mars, kwargs)[nodes]       # [target nodes, B]
    a = node_flows[sid + nodes] - log_z
    a = torch.where(torch.isneginf(log_z), -float("inf"), a)
    shift = torch.full([num_target_vars, batch_size], -float("inf"), dtype = a.dtype, device = device)
    shift = shift.scatter_reduce(0, rows_of[:, None].expand(-1, batch_size), a, reduce = "amax")
    shift = torch.where(torch.isfinite(shift), shift, 0.0)
    weights = torch.exp(a - shift[rows_of])

    sums = _emission_sums(params, layer.s_pids[nodes], rows_of, weights, num_target_vars, num_cats)

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
