from __future__ import annotations

import torch


def target_nodes(layer, target_vars, num_vars, device):
    """
    The input nodes a conditional query reads, and where each variable's result goes.

    Returns ``node_ids`` (the layer-local ids of the nodes whose variable is a target, in node order),
    ``rev_vars_mapping`` (variable id -> output row) and the number of target variables. Without
    ``target_vars`` every node and every variable is a target. The kernels always go through
    ``node_ids``, so a node's parameters, metadata and flow are all read at its own id.
    """
    if target_vars is None:
        target_vars = list(range(num_vars))
    rev_vars_mapping = torch.zeros([num_vars], dtype = torch.long)
    for i, var in enumerate(target_vars):
        rev_vars_mapping[var] = i
    is_target = torch.isin(layer.vids[:, 0], torch.as_tensor(list(target_vars), dtype = torch.long, device = layer.vids.device))
    node_ids = torch.nonzero(is_target).flatten().to(device)
    return node_ids, rev_vars_mapping.to(device), len(target_vars)


def target_weights(log_w: torch.Tensor, rows_of: torch.Tensor, num_targets: int) -> torch.Tensor:
    """``exp(log_w)`` ([nodes, B], node ``n`` of target ``rows_of[n]``), every (target, sample) shifted by its largest
    -- a scale its answer is normalized out of -- so that no target's weights underflow; a target that nothing weighs
    keeps zeros."""
    batch_size = log_w.size(1)
    shift = torch.full([num_targets, batch_size], -float("inf"), dtype = log_w.dtype, device = log_w.device)
    shift = shift.scatter_reduce(0, rows_of[:, None].expand(-1, batch_size), log_w, reduce = "amax")
    shift = torch.where(torch.isfinite(shift), shift, 0.0)
    return torch.exp(log_w - shift[rows_of])


def emission_sums(params: torch.Tensor, s_pids: torch.Tensor, rows_of: torch.Tensor, weights: torch.Tensor,
                  num_targets: int, num_cats: int) -> torch.Tensor:
    """``[B, num_targets, num_cats]``: ``sum_n weights[n, b] p_n(:)`` over the nodes ``n`` of every target
    (``rows_of[n]`` its output row, ``s_pids[n]`` the start of its ``num_cats`` parameters in ``params``), as matmuls
    with the emission table:

    * rows the targets share (tied emissions): ONE matmul, ``[B * targets, rows] @ [rows, num_cats]``, every row read
      once;
    * rows each target has to itself (untied): one batched matmul over the targets, each against its own rows,
      gathered in chunks of at most ``_SUMS_CHUNK`` floats -- the dense form would multiply every target's weights
      with every other target's rows, all of them zero.

    Which: by how full the dense form's weight matrix would be. The indexing is built on the device; the host reads
    only sizes."""
    batch_size = weights.size(1)
    device = weights.device
    if s_pids.numel() == 0:                                                  # no node of any target
        return torch.zeros([batch_size, num_targets, num_cats], dtype = weights.dtype, device = device)

    # The emission table: `params` itself, viewed as rows, when every node's parameters start a row of it
    if params.numel() % num_cats == 0 and bool((s_pids % num_cats == 0).all()):
        table = params.view(-1, num_cats)
        rows = s_pids // num_cats
    else:
        uniq, rows = torch.unique(s_pids, return_inverse = True)
        table = params[uniq[:, None] + torch.arange(num_cats, device = device)[None, :]]

    sums = torch.empty([batch_size, num_targets, num_cats], dtype = weights.dtype, device = device)
    uniq_rows, col = torch.unique(rows, return_inverse = True)
    pairs, pair_of = torch.unique(rows_of * table.size(0) + rows, return_inverse = True)    # distinct (target, row)
    n_rows, n_pairs = uniq_rows.numel(), pairs.numel()

    if 2 * n_pairs >= num_targets * n_rows:
        # Shared rows: R[b, t, j] = the weight target t gives row uniq_rows[j]
        R = torch.zeros([batch_size, num_targets, n_rows], dtype = weights.dtype, device = device)
        R.index_put_((torch.arange(batch_size, device = device)[:, None], rows_of[None, :], col[None, :]),
                     weights.t(), accumulate = True)
        if int(uniq_rows[-1] - uniq_rows[0]) + 1 == n_rows:
            tab = table[int(uniq_rows[0]):int(uniq_rows[0]) + n_rows]         # a contiguous run of rows: a view
        else:
            tab = table[uniq_rows]
        torch.matmul(R.view(batch_size * num_targets, n_rows), tab, out = sums.view(batch_size * num_targets, num_cats))
        return sums

    # Private rows: target t's rows, padded to the most any target has (the padding weighs nothing)
    pair_t = pairs // table.size(0)
    counts = torch.bincount(pair_t, minlength = num_targets)
    k_max = int(counts.max())
    pos = torch.arange(n_pairs, device = device) - (torch.cumsum(counts, 0) - counts)[pair_t]
    idx = torch.zeros([num_targets, k_max], dtype = torch.long, device = device)
    idx[pair_t, pos] = pairs % table.size(0)
    R = torch.zeros([num_targets, batch_size, k_max], dtype = weights.dtype, device = device)
    R.index_put_((rows_of[:, None], torch.arange(batch_size, device = device)[None, :], pos[pair_of][:, None]),
                 weights, accumulate = True)

    chunk = max(1, _SUMS_CHUNK // (k_max * num_cats))
    for t0 in range(0, num_targets, chunk):
        t1 = min(t0 + chunk, num_targets)
        sums[:, t0:t1, :] = torch.bmm(R[t0:t1], table[idx[t0:t1]]).permute(1, 0, 2)

    return sums


# The most floats the private-rows form of `emission_sums` gathers from the emission table at a time
_SUMS_CHUNK = 1 << 26
