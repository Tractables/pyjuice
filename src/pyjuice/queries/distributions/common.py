from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class LayerFacts:
    """What the conditional queries read of an input layer's structure, kept on the host (``layer_facts``): the number
    of variables; every variable's nodes (layer-local ids, ascending), as a CSR over each node's first variable; every
    node's first parameter. Structure only -- it does not change once the circuit is compiled, while parameters are
    read where they are used. ``extra``: what a distribution's module keeps of the same kind."""
    num_vars: int
    var_ptr: np.ndarray
    var_nodes: np.ndarray
    s_pids: np.ndarray
    extra: dict = field(default_factory = dict)


def layer_facts(layer) -> LayerFacts:
    """``layer``'s :class:`LayerFacts`, computed on its first query (reading its tables back from the device once)."""
    facts = layer.__dict__.get("_query_facts")
    if facts is None:
        vids = layer.vids.cpu().numpy().astype(np.int64)
        first = vids[:, 0]
        num_vars = int(vids.max()) + 1
        var_nodes = np.argsort(first, kind = "stable")
        var_ptr = np.zeros([num_vars + 1], dtype = np.int64)
        var_ptr[1:] = np.cumsum(np.bincount(first, minlength = num_vars))
        facts = LayerFacts(num_vars, var_ptr, var_nodes, layer.s_pids.cpu().numpy().astype(np.int64))
        layer.__dict__["_query_facts"] = facts
    return facts


def runs(node_runs, num_nodes: int):
    """``[(start, count)]`` of a layer's nodes to compute: every ``(start, end)`` of ``node_runs`` rounded up to a power
    of two of nodes within the layer, as :meth:`~pyjuice.layer.InputLayer._forward_runs` rounds them; all of them when
    ``node_runs`` is None."""
    if node_runs is None:
        return [(0, num_nodes)]
    out = []
    for start, end in node_runs:
        n = min(1 << max(end - start - 1, 0).bit_length(), num_nodes)
        out.append((min(start, num_nodes - n), n))
    return out


@dataclass
class Targets:
    """The input nodes of a layer that a conditional query reads -- those whose (first) variable is a target -- and
    where each one's answer goes, on the device: ``nodes`` (layer-local ids; ascending within each target),
    ``rows_of`` (every node's output row), ``rev_vars_mapping`` (variable -> output row). ``host``: ``(first parameters,
    rows_of)`` of the nodes as numpy arrays, for indexing built on the host (`emission_sums`) -- None for target sets
    of more than ``_HOST_INDEX_MAX`` nodes, whose indexing is built on the device. ``max_nodes``: the most nodes any
    target has."""
    nodes: torch.Tensor
    rows_of: torch.Tensor
    rev_vars_mapping: torch.Tensor
    num_targets: int
    max_nodes: int
    host: Optional[tuple] = None


def targets(layer, target_vars, device, num_vars: Optional[int] = None) -> Targets:
    """The :class:`Targets` of ``target_vars`` (every variable if None) in ``layer``. A variable listed more than once
    answers in the row of its last listing. At most one upload (none for one target on a run of nodes) and no read
    back from the device -- but past ``_HOST_INDEX_MAX`` nodes, found on the device. ``num_vars``: the size of
    ``rev_vars_mapping`` (the layer's number of variables if None)."""
    f = layer_facts(layer)
    if num_vars is None:
        num_vars = f.num_vars
    tv = np.arange(f.num_vars, dtype = np.int64) if target_vars is None else \
         np.asarray(list(target_vars), dtype = np.int64).reshape(-1)

    # One target whose nodes are a run of consecutive ids (a variable's input group): built on the device as it is,
    # with nothing to upload -- every row is 0, as is every variable's
    if tv.size == 1 and 0 <= tv[0] < f.num_vars:
        a, b = f.var_ptr[tv[0]], f.var_ptr[tv[0] + 1]
        run = f.var_nodes[a:b]
        n = int(b - a)
        if n > 0 and int(run[-1] - run[0]) == n - 1:
            zeros = torch.zeros([n + num_vars], dtype = torch.long, device = device)
            return Targets(torch.arange(int(run[0]), int(run[0]) + n, device = device), zeros[:n], zeros[n:], 1, n,
                           host = (f.s_pids[run], np.zeros([n], dtype = np.int64)) if n <= _HOST_INDEX_MAX else None)

    rev = np.zeros([num_vars], dtype = np.int64)
    uniq, last = np.unique(tv[::-1], return_index = True)
    rev[uniq] = tv.size - 1 - last
    inside = uniq[(uniq >= 0) & (uniq < f.num_vars)]                    # variables the layer has nodes of
    lens = f.var_ptr[inside + 1] - f.var_ptr[inside]
    total = int(lens.sum())
    max_nodes = int(lens.max()) if lens.size > 0 else 0

    if total <= _HOST_INDEX_MAX:
        starts = f.var_ptr[inside]
        offsets = np.cumsum(lens) - lens
        nodes = f.var_nodes[np.arange(total, dtype = np.int64) + np.repeat(starts - offsets, lens)]
        rows_of = np.repeat(rev[inside], lens)
        packed = torch.from_numpy(np.concatenate([nodes, rows_of, rev])).to(device)
        return Targets(packed[:total], packed[total:2 * total], packed[2 * total:], tv.size, max_nodes,
                       host = (f.s_pids[nodes], rows_of))

    rev_d = torch.from_numpy(rev).to(device)
    is_target = torch.zeros([f.num_vars], dtype = torch.bool)
    is_target[torch.from_numpy(inside)] = True
    nodes = torch.nonzero(is_target.to(device)[layer.vids[:, 0].long()]).flatten()
    return Targets(nodes, rev_d[layer.vids[nodes, 0].long()], rev_d, tv.size, max_nodes)


def target_nodes(layer, target_vars, num_vars, device):
    """
    The input nodes a conditional query reads, and where each variable's result goes.

    Returns ``node_ids`` (the layer-local ids of the nodes whose variable is a target; ascending within each target),
    ``rev_vars_mapping`` (variable id -> output row) and the number of target variables. Without ``target_vars`` every
    node and every variable is a target. The kernels always go through ``node_ids``, so a node's parameters, metadata
    and flow are all read at its own id.
    """
    t = targets(layer, target_vars, device, num_vars = num_vars)
    return t.nodes, t.rev_vars_mapping, t.num_targets


def target_weights(log_w: torch.Tensor, rows_of: torch.Tensor, num_targets: int) -> torch.Tensor:
    """``exp(log_w)`` ([nodes, B], node ``n`` of target ``rows_of[n]``), every (target, sample) shifted by its largest
    -- a scale its answer is normalized out of -- so that no target's weights underflow; a target that nothing weighs
    keeps zeros."""
    batch_size = log_w.size(1)
    if num_targets == 1 and log_w.size(0) > 0:                   # the one row's maximum, as the scatter would take it
        shift = log_w.amax(dim = 0, keepdim = True)
        return torch.exp(log_w - torch.where(torch.isfinite(shift), shift, 0.0))
    shift = torch.full([num_targets, batch_size], -float("inf"), dtype = log_w.dtype, device = log_w.device)
    shift = shift.scatter_reduce(0, rows_of[:, None].expand(-1, batch_size), log_w, reduce = "amax")
    shift = torch.where(torch.isfinite(shift), shift, 0.0)
    return torch.exp(log_w - shift[rows_of])


def emission_sums(params: torch.Tensor, s_pids, rows_of, weights: torch.Tensor,
                  num_targets: int, num_cats: int) -> torch.Tensor:
    """``[B, num_targets, num_cats]``: ``sum_n weights[n, b] p_n(:)`` over the nodes ``n`` of every target
    (``rows_of[n]`` its output row, ``s_pids[n]`` the start of its ``num_cats`` parameters in ``params``), as matmuls
    with the emission table:

    * rows the targets share (tied emissions): ONE matmul, ``[B * targets, rows] @ [rows, num_cats]``, every row read
      once;
    * rows each target has to itself (untied): one batched matmul over the targets, each against its own rows,
      gathered in chunks of at most ``_SUMS_CHUNK`` floats -- the dense form would multiply every target's weights
      with every other target's rows, all of them zero.

    Which: by how full the dense form's weight matrix would be. ``s_pids`` and ``rows_of`` are numpy arrays (the
    indexing is then built on the host and uploaded at once: no read back from the device) or device tensors (built on
    the device; the host reads only sizes)."""
    if isinstance(s_pids, np.ndarray):
        return _emission_sums_host(params, s_pids, rows_of, weights, num_targets, num_cats)

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


def _emission_sums_host(params, s_pids, rows_of, weights, num_targets, num_cats):
    """`emission_sums` with its indexing built on the host from numpy ``s_pids`` / ``rows_of`` -- the same steps as on
    the device, and the same operands for the device's scatter and matmuls -- and uploaded in one copy."""
    batch_size = weights.size(1)
    device = weights.device
    if s_pids.size == 0:
        return torch.zeros([batch_size, num_targets, num_cats], dtype = weights.dtype, device = device)

    gather = None
    if params.numel() % num_cats == 0 and bool((s_pids % num_cats == 0).all()):
        table_rows = params.numel() // num_cats
        rows = s_pids // num_cats
    else:
        gather, rows = np.unique(s_pids, return_inverse = True)                  # rows of a gathered table
        table_rows = gather.size

    # One target whose nodes read distinct rows in ascending order (a variable's emissions, laid out by node): the
    # scatter below would write the weights as they are, so they are handed over as they are
    ascending = num_targets == 1 and (rows.size == 1 or bool((np.diff(rows) > 0).all()))
    if ascending:
        uniq_rows, col = rows, None
        n_rows = n_pairs = rows.size
    else:
        uniq_rows, col = np.unique(rows, return_inverse = True)
        n_rows = n_pairs = uniq_rows.size                                        # one target: every pair is (0, row)
        if num_targets > 1:
            pairs, pair_of = np.unique(rows_of * table_rows + rows, return_inverse = True)
            n_pairs = pairs.size
    shared = 2 * n_pairs >= num_targets * n_rows
    contiguous = shared and int(uniq_rows[-1] - uniq_rows[0]) + 1 == n_rows
    # Nodes of one target reading one row add up in the scatter; without any, its writes are the values themselves
    accumulate = n_pairs < rows.size

    # Every index the device needs, in one upload (none at all for an ascending target on a run of rows)
    if ascending:
        parts = [] if contiguous else [uniq_rows]
    elif shared:
        parts = [rows_of, col] + ([] if contiguous else [uniq_rows])
    else:
        pair_t = pairs // table_rows
        counts = np.bincount(pair_t, minlength = num_targets)
        k_max = int(counts.max())
        pos = np.arange(n_pairs) - (np.cumsum(counts) - counts)[pair_t]
        idx = np.zeros([num_targets, k_max], dtype = np.int64)
        idx[pair_t, pos] = pairs % table_rows
        parts = [rows_of, pos[pair_of], idx.reshape(-1)]
    if gather is not None:
        parts.append(gather)
    pieces = []
    if parts:
        sizes = [p.size for p in parts]
        dev = torch.from_numpy(np.concatenate([np.asarray(p, dtype = np.int64).reshape(-1) for p in parts])).to(device)
        pieces = list(torch.split(dev, sizes))

    if gather is not None:
        table = params[pieces.pop()[:, None] + torch.arange(num_cats, device = device)[None, :]]
    else:
        table = params.view(-1, num_cats)

    sums = torch.empty([batch_size, num_targets, num_cats], dtype = weights.dtype, device = device)
    if shared:
        if ascending:
            R = weights.t().contiguous()                                         # = the scatter's result, [B, rows]
        else:
            rows_of_d, col_d = pieces[0], pieces[1]
            R = torch.zeros([batch_size, num_targets, n_rows], dtype = weights.dtype, device = device)
            R.index_put_((torch.arange(batch_size, device = device)[:, None], rows_of_d[None, :], col_d[None, :]),
                         weights.t(), accumulate = accumulate)
        if contiguous:
            tab = table[int(uniq_rows[0]):int(uniq_rows[0]) + n_rows]
        else:
            tab = table[pieces[-1]]
        torch.matmul(R.view(batch_size * num_targets, n_rows), tab, out = sums.view(batch_size * num_targets, num_cats))
        return sums

    rows_of_d, pos_d, idx_d = pieces[0], pieces[1], pieces[2].view(num_targets, k_max)
    R = torch.zeros([num_targets, batch_size, k_max], dtype = weights.dtype, device = device)
    R.index_put_((rows_of_d[:, None], torch.arange(batch_size, device = device)[None, :], pos_d[:, None]),
                 weights, accumulate = accumulate)

    chunk = max(1, _SUMS_CHUNK // (k_max * num_cats))
    for t0 in range(0, num_targets, chunk):
        t1 = min(t0 + chunk, num_targets)
        sums[:, t0:t1, :] = torch.bmm(R[t0:t1], table[idx_d[t0:t1]]).permute(1, 0, 2)

    return sums


# The most floats the private-rows form of `emission_sums` gathers from the emission table at a time
_SUMS_CHUNK = 1 << 26

# The most target nodes whose indexing (`targets`, `emission_sums`) is built on the host: beyond, numpy's sorts and the
# upload cost more than the device's indexing and its syncs. MEASURED (matmul arm, HMM emissions): the host's wins up to
# 4096 nodes (0.51-0.68 ms against 0.80-1.05 per call) and loses at 32768 (1.17-1.37 against 0.84-1.11)
_HOST_INDEX_MAX = 1 << 13
