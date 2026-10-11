"""
Conditionals under a constraint on the lifted plan: ``log p(x_t = v | e, C)`` for every target (sample, position) and
every token ``v``.

A query runs the forward pass (:mod:`.forward`) over the inside values its targets need -- those of the nodes whose
scope misses a target -- and the top-down pass (:mod:`.backward`) over the top-down values they need -- those of the
nodes whose scope meets one -- which leaves the CLASS WEIGHTS of the target positions' input nodes, every sample's, in
``dfa_class_flow`` (see :mod:`.kernels.score`). Then every ROW GROUP (the positions whose input nodes read the same rows
of their parameter tables, :class:`~.kernels.score.ScoreTables`) is scored in one pass, all its targets at once, against
a token-major copy of those rows that the constrained circuit keeps inside :func:`pyjuice.fast_inference`
(:meth:`~pyjuice.constraints.ConstrainedCircuit._token_table`). The normalizer ``p(e, C)`` comes from the class weights
and the class masses: the root's inside value is never needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from . import forward as F
from .backward import run_backward
from .kernels.live import prune
from .kernels.score import score_group, score_tables
from .query_plan import prior_regions, query_plan, top_down_prior


def _host_mask(mask: torch.Tensor, B: int, n: int, name: str) -> np.ndarray:
    m = mask.detach().cpu().numpy().astype(bool)
    if m.ndim == 1:
        m = m[None, :].repeat(B, axis = 0)
    if m.shape != (B, n):
        raise ValueError(f"`{name}` must be [{n}] or [{B}, {n}], got {list(mask.shape)}.")
    return m


def _masks(missing_mask: Optional[torch.Tensor], targets: Optional[torch.Tensor], B: int, n: int):
    """``(missing, targets)``, bool [B, n] on the host: every missing position targeted by default; each target must
    be missing."""
    missing = np.zeros((B, n), dtype = bool) if missing_mask is None else _host_mask(missing_mask, B, n, "missing_mask")
    if targets is None:
        return missing, missing
    tgt = _host_mask(targets, B, n, "targets")
    if (tgt & ~missing).any():
        raise ValueError("Every target must be a missing position (marginalized in `missing_mask`).")
    return missing, tgt


def _layout(cc, positions: np.ndarray, B: int):
    """Where the class weights of a query with the target ``positions`` go in ``dfa_class_flow``: every position's
    block ``[B, classes, rows of its row group]``, a row group's consecutive: ``(groups, base, K, size)`` -- per row group
    with a target position ``(group, its target positions, start)``, and per position its block's start (-1: no
    target) and its row group's rows (int64 [n], host)."""
    score, n, C = score_tables(cc._lifted_program()), cc.n, cc.num_classes
    union = np.nonzero(positions)[0]
    assert (score.group_of[union] >= 0).all(), "a target position without an input node in any product"
    base, K = np.full(n, -1, dtype = np.int64), np.zeros(n, dtype = np.int64)
    groups, size = [], 0
    for g in np.unique(score.group_of[union]).tolist():
        pos = union[score.group_of[union] == g]
        k = len(score.rows[g])
        K[pos] = k
        base[pos] = size + np.arange(len(pos)) * B * C * k
        groups.append((g, pos, size))
        size += len(pos) * B * C * k
    return groups, base, K, size


def _check(cc, data: torch.Tensor, precision: str):
    if precision not in F.PRECISIONS:
        raise ValueError(f"`precision` must be one of {sorted(F.PRECISIONS)}, got {precision!r}.")
    if cc._device.type != "cuda":
        raise NotImplementedError(f"The lifted backend runs on CUDA only, but the constrained circuit is on "
                                  f"{cc._device}. Move it with `cc.to('cuda')`.")
    if data.dim() != 2 or data.size(1) != cc.n:
        raise ValueError(f"`data` must be [batch, {cc.n}], got {list(data.shape)}.")


def class_weights(cc, data: torch.Tensor, missing: torch.Tensor, positions: np.ndarray, precision: str = "fp32",
                  held: Optional[dict] = None, observed: Optional[np.ndarray] = None, layout = None):
    """
    The forward and top-down passes of a conditional query: the log class weights of every input node over the
    target ``positions`` (bool [n], host), every sample's, summed per row of the class masses (see
    :mod:`.kernels.score`), in ``dfa_class_flow``.

    :param missing: [B, n] bool on the device, True where a token is missing
    :param held: see :func:`conditional`
    :param observed: None, or bool [n] (host): the positions some sample observes (read off ``missing`` if None)
    :param layout: None, or ``(groups, size, tables)``: :func:`_layout`'s groups and size, and its ``base`` and ``K``
        already on the device, concatenated (int64 [2n]) -- copied before the query launches anything, as copies from
        the host wait for the stream
    :returns: ``(groups, class_mars)``: per row group with a target position, ``(group, its target positions (host),
        weights [positions x B, classes, rows of the group])`` (sample ``b`` of the ``p``-th position: column ``p * B +
        b``), and the class masses the query read
    """
    prog, td, n, dev = cc._lifted_program(), cc._top_down_program(), cc.n, cc._device
    score = score_tables(prog)
    B = data.size(0)
    if layout is None:
        groups, base, K, size = _layout(cc, positions, B)
        layout = (groups, size, torch.from_numpy(np.concatenate([base, K])).to(dev))
    groups, size, tables = layout
    if observed is None:
        observed = (~missing).any(dim = 0).cpu().numpy()
    x = torch.where(missing, 0, data.to(dev).long())                          # missing tokens: any valid id
    bufs = cc._buffers(B)
    flows = cc._flows(B, bufs)
    held = held or {}
    obs_class = torch.where(missing, -1, cc.token_classes.token_class[x]).contiguous()
    pruning = prune(prog, obs_class, B) if observed.any() else None
    valid, kept = cc._prior_validity()
    prior = prior_regions(prog, observed, B, kept, pruning)
    pf = cc._prior_flows(np.zeros(len(cc.sum_regions), dtype = bool))
    top = dict(program = td, once = top_down_prior(prog, td, observed, B, kept, pruning), valid = pf["valid"].copy(),
               held = held.get("top_down"), scoring = True)
    plan = query_plan(prog, observed, positions, valid.numpy(), prior, held = held.get("inside"), top_down = top)
    class_mars = F.run_forward(cc, bufs, plan, x, missing, obs_class, pruning, precision)
    valid |= plan.todo

    C = cc.num_classes
    buf = cc._storage_view("dfa_class_flow", size)
    buf.fill_(-float("inf"))
    cw = dict(buf = buf, base = tables[:n], K = tables[n:2 * n])
    run_backward(cc, bufs, flows, plan, x, missing, obs_class, pruning, precision, class_weights = cw,
                 class_mars = class_mars)
    return [(g, pos, buf[s:s + len(pos) * B * C * len(score.rows[g])].view(len(pos) * B, C, len(score.rows[g])))
            for g, pos, s in groups], class_mars


def conditional(cc, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None, precision: str = "fp32", return_log_marginal: bool = False,
                held: Optional[dict] = None):
    """
    ``log p(x_t = v | e, C)`` for every target ``(b, t)`` and token ``v``: one row per target, in the order
    ``dense[targets]`` lists them (sample by sample, positions in order); -inf rows where ``p(e, C) = 0``.

    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True where a token is marginalized
    :param targets: None (every missing position of every sample), or a bool mask [n] or [B, n] of missing positions
    :param precision: "fp32" (fp32-level, the default) or "tf32", for the sum layers' and the scoring's matrix
        products; a token table in bf16 (inside a ``pyjuice.fast_inference`` scope allowing bf16 parameter copies) is
        read in tf32
    :param return_log_marginal: also return ``log p(e, C)`` of every row's sample, [rows]
    :param held: None, or what the buffers hold for this query, as a caching engine decides (see :mod:`.query_plan`):
        a dict with ``inside`` and ``top_down``, each None or bool [sum regions]
    :returns: [rows, V]
    """
    _check(cc, data, precision)
    n, dev = cc.n, cc._device
    B = data.size(0)
    miss, tgt = _masks(missing_mask, targets, B, n)                          # on the host: the query's one wait
    V = cc.token_classes.token_class.numel()
    rank = np.cumsum(tgt.ravel()).reshape(B, n) - 1                          # a target's row
    M = int(tgt.sum())
    if M == 0 or not cc.satisfiable:
        out = torch.full((M, V), -float("inf"), device = dev)
        return (out, torch.full((M,), -float("inf"), device = dev)) if return_log_marginal else out

    score = score_tables(cc._lifted_program())
    positions = tgt.any(axis = 0)
    groups, base, K, size = _layout(cc, positions, B)
    # every row group's columns' rows (-1: a sample not targeting the position), with the layout, in one copy
    out_rows = [np.where(tgt[:, pos], rank[:, pos], -1).T.ravel() for _, pos, _ in groups]     # column p * B + b
    tables = torch.from_numpy(np.concatenate([base, K, *out_rows])).to(dev)
    missing = F._missing(missing_mask, B, n, dev) if missing_mask is None or missing_mask.is_cuda else \
        torch.from_numpy(miss).to(dev)
    with torch.no_grad():
        groups, class_mars = class_weights(cc, data, missing, positions, precision, held,
                                           observed = (~miss).any(axis = 0), layout = (groups, size, tables))
        out = torch.empty(M, V, device = dev)
        log_marginal = torch.empty(M + 1, device = dev) if return_log_marginal else None
        start = 2 * n
        for g, pos, w in groups:
            out_row = tables[start:start + len(pos) * B]
            start += len(pos) * B
            logz = score_group(out, cc._token_table(g, score.rows[g]), w, score.rows_t[g], class_mars, out_row,
                               cc.token_classes, F.PRECISIONS[precision])
            if log_marginal is not None:
                log_marginal[torch.where(out_row >= 0, out_row, M)] = logz
    return (out, log_marginal[:M]) if return_log_marginal else out
