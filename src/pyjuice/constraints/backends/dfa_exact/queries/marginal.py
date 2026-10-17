"""
The marginal query: ``log p(C, e)`` for a batch of evidence, one forward pass over every inside value
(:func:`marginal`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..kernels.compact import (STORAGES, compact_buffers, held_elsewhere, held_moves, query_layout,
                              resolve_storage, worth_parking)
from ..kernels.live import live_successors, liveness
from .forward import PRECISIONS, gate_context, missing_tensor, run_forward
from .plans import per_sample_intervals, prior_regions, query_plan, read_regions



def marginal(backend, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, precision: str = "fp32",
             held: Optional[np.ndarray] = None, report: Optional[dict] = None, storage: str = "auto",
             carry: Optional[dict] = None, external = None):
    """
    ``log p(C, e)`` for every sample: the probability that the PC generates a string that satisfies the
    constraint and agrees with the observed tokens.

    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True where a token is marginalized
    :param precision: "fp32" (default), "tf32" or "bf16" for the sum layers' matrix products (see
        :data:`~.forward.PRECISIONS`)
    :param held: None, or bool [sum regions]: the per-sample inside values ``node_mars`` holds for this query, as a
        decoder decides (see :mod:`.plans`)
    :param report: None, or a dict where the query leaves what :func:`~.conditional.class_weights` says
    :param storage: "auto" (the default), "dense" or "compact": how a query keeps its per-sample values (see
        :func:`~.conditional.conditional`); a marginal has no top-down value
    :param carry: compact, where the inside values ``held`` marks are (see :func:`~.conditional.conditional`)
    :param external: None, or a decoder session's per-sample inputs (:class:`~..state.ExternalInputs`): their
        variables (soft evidence) and scopes (gates) count as observed when they differ per row (a batch of 1: the
        values over them are every sample's, kept once for the session); the positions they leave free are missing
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
    if external is not None and external.batch not in (1, B):
        raise ValueError(f"The decoder's per-sample inputs have a batch of {external.batch}: a query's must be the "
                         f"same (or theirs 1), got {B}.")
    root_first, root_end = pc._root_node_range
    if not backend.satisfiable:
        return torch.full((B, root_end - root_first), -float("inf"), device = dev)

    prog = backend.program()
    missing = missing_tensor(missing_mask, B, n, dev)
    if external is not None and external.free is not None:             # (a decoder's queries have them already)
        missing, missing_mask = missing | external.free, True
    observed = np.ones(n, dtype = bool) if missing_mask is None else \
        (~missing).any(dim = 0).cpu().numpy()                           # the positions some sample observes
    if external is not None:                                            # values that differ per sample
        observed |= external.observed
    x = torch.where(missing, 0, data.to(dev).long())                     # missing tokens: any valid id

    with torch.no_grad():
        obs_class = torch.where(missing, -1, backend.token_classes.token_class[x]).contiguous()
        masses = backend.state.masses(external)
        succ = live_successors(prog, masses)            # (per-sample inputs that rule classes out: theirs)
        # nothing observed: nothing to prune, but compact values are laid out by the live columns (every one)
        lv = liveness(prog, obs_class, B, succ) if observed.any() or storage != "dense" or succ is not None else None
        valid, kept = backend.state.prior_validity(masses.version)
        prior = prior_regions(prog, observed, B, kept, lv)
        no_top_down = np.zeros(len(backend.sum_regions), dtype = bool)
        plan = query_plan(prog, observed, None, valid.numpy(), prior, held = held)   # every inside value
        move = held                                     # compact: the held values it reads (the root's too) get rows
        if move is not None and carry is not None and worth_parking(prog, carry):
            move = held_moves(prog, carry, move & read_regions(prog, plan, root = True))
        storage, compact = resolve_storage(backend, storage, B, False, lv, plan, no_top_down, move)
        dropped = held_elsewhere(storage, [held], carry)
        if dropped:                                     # held in the other storage: computed again
            held, carry, move = None, None, None
            plan = query_plan(prog, observed, None, valid.numpy(), prior)
            if compact is not None:
                compact = query_layout(backend, lv, plan, no_top_down, None)
        inputs_only = compact is not None and not plan.sample.any() and (move is None or not move.any())
        parked = []
        if compact is not None:                         # rows for the per-sample values only, the held ones it
            bufs, parked = compact_buffers(backend, compact, B, inputs_only, carry, move)   # reads re-gathered there
        else:
            if lv is not None and not observed.any() and succ is None:   # (laid out for "auto", but dense:
                lv = None                                                 # nothing to prune)
            if lv is not None:
                lv.order(per_sample_intervals(prog, plan))
            bufs = backend.state.buffers(B)
        if report is not None:
            report.update(plan = plan, liveness = lv, inputs_only = inputs_only, compact = compact, storage = storage,
                          dropped = dropped, parked = parked)
        run_forward(backend, bufs, plan, x, missing, obs_class, lv, precision, compact, masses,
                    gates = gate_context(backend, external))
        valid |= plan.todo

        r = prog.root_region
        if plan.root_prior:                                             # nothing observed: every sample's value
            prior = backend.state.prior_buffer()
            base = prior["offset"][r] + (root_first - backend.sum_regions[r][0]) * prior["width"][r]
            rows = base + torch.arange(root_end - root_first, device = dev) * prior["width"][r]
            return prior["mars"][rows][None, :].expand(B, -1).contiguous()
        lay = bufs["layout"]
        width = int(lay["sum_widths"][r])
        base = int(lay["sum_offsets"][r]) + (root_first - backend.sum_regions[r][0]) * width
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
