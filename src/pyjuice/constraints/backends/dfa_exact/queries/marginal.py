"""
The marginal query: ``log p(C, e)`` for a batch of evidence, one forward pass over every inside value
(:func:`marginal`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..kernels.compact import STORAGES, compact_buffers
from ..kernels.live import liveness
from .forward import PRECISIONS, missing_tensor, run_forward
from .plans import per_sample_intervals, prior_regions, query_plan



def marginal(backend, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, precision: str = "fp32",
             held: Optional[np.ndarray] = None, report: Optional[dict] = None, storage: str = "dense",
             carry: Optional[dict] = None):
    """
    ``log p(C, e)`` for every sample: the probability that the PC generates a string that satisfies the
    constraint and agrees with the observed tokens.

    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True where a token is marginalized
    :param precision: "fp32" (default) or "tf32" for the sum layers' matrix products
    :param held: None, or bool [sum regions]: the per-sample inside values ``node_mars`` holds for this query, as a
        decoder decides (see :mod:`.plans`)
    :param report: None, or a dict where the query leaves its :class:`~.plans.QueryPlan` (``"plan"``), its
        :class:`~..kernels.live.Liveness` (``"liveness"``; None when it observes nothing, dense) and its
        :class:`~..kernels.compact.CompactLayout` (``"compact"``; None when dense)
    :param storage: "dense" or "compact": how a query keeps its per-sample values (see
        :func:`~.conditional.conditional`); a marginal has no top-down value
    :param carry: compact, where the inside values ``held`` marks are (see :func:`~.conditional.conditional`)
    :returns: [B, number of root nodes], as :func:`pyjuice.queries.marginal`
    """
    if precision not in PRECISIONS:
        raise ValueError(f"`precision` must be one of {sorted(PRECISIONS)}, got {precision!r}.")
    if storage not in STORAGES:
        raise ValueError(f"`storage` must be one of {list(STORAGES)}, got {storage!r}.")
    pc, n, dev = backend.pc, backend.n, backend.device
    if dev.type != "cuda":
        raise NotImplementedError(f"The dfa_exact backend runs on CUDA only, but the constrained circuit is on {dev}. "
                                  f"Move it with `cc.to('cuda')`.")
    if data.dim() != 2 or data.size(1) != n:
        raise ValueError(f"`data` must be [batch, {n}], got {list(data.shape)}.")
    B = data.size(0)
    root_first, root_end = pc._root_node_range
    if not backend.satisfiable:
        return torch.full((B, root_end - root_first), -float("inf"), device = dev)

    prog = backend.program()
    missing = missing_tensor(missing_mask, B, n, dev)
    observed = np.ones(n, dtype = bool) if missing_mask is None else \
        (~missing).any(dim = 0).cpu().numpy()                           # the positions some sample observes
    x = torch.where(missing, 0, data.to(dev).long())                     # missing tokens: any valid id

    with torch.no_grad():
        obs_class = torch.where(missing, -1, backend.token_classes.token_class[x]).contiguous()
        compact = storage == "compact"
        # nothing observed: nothing to prune, but compact values are laid out by the live columns (every one)
        lv = liveness(prog, obs_class, B) if observed.any() or compact else None
        valid, kept = backend.state.prior_validity()
        prior = prior_regions(prog, observed, B, kept, lv)
        plan = query_plan(prog, observed, None, valid.numpy(), prior, held = held)   # every inside value
        inputs_only = compact and not plan.sample.any() and (held is None or not held.any())
        if compact:                                     # rows for the per-sample values only, the held ones
            compact, bufs = compact_buffers(backend, lv, plan, B, inputs_only,   # re-gathered into them
                                            np.zeros(len(backend.sum_regions), dtype = bool), held, carry)
        else:
            if lv is not None:
                lv.order(per_sample_intervals(prog, plan))
            compact, bufs = None, backend.state.buffers(B)
        if report is not None:
            report.update(plan = plan, liveness = lv, inputs_only = inputs_only, compact = compact)
        run_forward(backend, bufs, plan, x, missing, obs_class, lv, precision, compact)
        valid |= plan.todo

        r = prog.root_region
        if plan.root_prior:                                             # nothing observed: every sample's value
            prior = backend.state.prior_buffer()
            base = prior["offset"][r] + (root_first - backend.sum_regions[r][0]) * prior["width"][r]
            rows = base + torch.arange(root_end - root_first, device = dev) * prior["width"][r]
            return prior["mars"][rows][None, :].expand(B, -1).contiguous()
        lay = bufs["layout"]
        base = lay["sum_offsets"][r] + (root_first - backend.sum_regions[r][0]) * lay["sum_widths"][r]
        width = lay["sum_widths"][r]
        rows = base + torch.arange(root_end - root_first, device = dev)[:, None] * width
        if compact is None:
            return bufs["node_mars"][rows + torch.arange(B, device = dev)[None, :]].t().contiguous()
        k = prog.reg_iv_all[r]                         # compact: a sample's one lane if both its ends are live, else
        if compact.total[k] == 0:                      # its evidence breaks the constraint
            return torch.full((B, root_end - root_first), -float("inf"), device = dev)
        off = compact.off[k]
        lane = off[:B] < off[1:]
        out = bufs["node_mars"][rows + torch.where(lane, off[:B], 0)[None, :]]
        return torch.where(lane[None, :], out, -float("inf")).t().contiguous()
