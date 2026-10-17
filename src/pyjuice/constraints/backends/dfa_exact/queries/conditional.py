"""
Conditionals under a constraint on the lifted plan: ``log p(x_t = v | e, C)`` for every target (sample, position) and
every token ``v``.

A query runs the forward pass (:mod:`.forward`) over the inside values its targets need -- those of the nodes whose
scope misses a target -- and the top-down pass (:mod:`.queries.top_down`) over the top-down values they need -- those of
the nodes whose scope meets one -- which leaves the CLASS WEIGHTS of the target positions' input nodes, every sample's,
in ``dfa_class_flow`` (see :mod:`.score`). Then every ROW GROUP (the positions whose input nodes read the same rows of
their parameter tables, :class:`~.score.ScoreTables`) is scored in one pass, all its targets at once, against a
token-major copy of those rows that the backend's state keeps inside :func:`pyjuice.fast_inference`
(:meth:`~.state.State.token_table`). The normalizer ``p(e, C)`` comes from the class weights and the class masses: the
root's inside value is never needed.
"""

from __future__ import annotations

import weakref
from typing import Optional

import numpy as np
import torch

from . import forward as F
from ..kernels.common import pack_aligned
from ..kernels.compact import (STORAGES, compact_buffers, held_elsewhere, held_moves, query_layout, regather_into,
                               resolve_storage, worth_parking)
from ..kernels.live import live_successors, liveness
from ..kernels.score import score_group, score_tables
from .plans import (PLANS_PER_POSITION, PLAN_CACHE_SIZE, per_sample_intervals, prior_regions, query_plan,
                    read_regions, scope_intervals, top_down_prior)
from .top_down import run_backward



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


def _layout(backend, positions: np.ndarray, B: int):
    """Where the class weights of a query with the target ``positions`` go in ``dfa_class_flow``: every position's
    block ``[B, classes, rows of its row group]``, a row group's consecutive: ``(groups, base, K, size)`` -- per row group
    with a target position ``(group, its target positions, start)``, and per position its block's start (-1: no
    target) and its row group's rows (int64 [n], host). Kept on the Program, by its arguments."""
    prog = backend.program()
    key = (positions.tobytes(), B)
    if key in prog.layouts:
        prog.layouts.move_to_end(key)
        return prog.layouts[key]
    score, n, C = score_tables(prog), backend.n, backend.num_classes
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
    prog.layouts[key] = (groups, base, K, size)
    if len(prog.layouts) > max(PLAN_CACHE_SIZE, PLANS_PER_POSITION * n):   # bounded as the plans are
        prog.layouts.popitem(last = False)
    return groups, base, K, size


def _sample_columns(masses, score, g: int, pos: np.ndarray, B: int, prog) -> Optional[tuple]:
    """What :func:`~..kernels.score.score_group`'s ``sample`` needs for row group ``g``'s target positions ``pos``
    (int64, host), besides the class masses: None unless per-sample inputs reweight their input nodes. Every node over
    a position reweighted by the same weights (or none of them reweighted), the column of sample ``b`` of the ``p``-th
    position reads its nodes' rows and weights: ``(nodes, positions, offsets)`` -- every position's input nodes on the
    device (int64 [n, rows of the group], kept on the program), and the positions and every column's first weight
    (int64, host: the query uploads them with its other tables, as a copy from the host waits for the device). Every
    position's weights' row is kept per session (its class masses, by weak reference)."""
    if masses.lognorm is None:
        return None
    rows = prog.__dict__.setdefault("sample_rows", {})
    kept = rows.get(g)
    if kept is None or kept[0]() is not masses:
        nodes = score.nodes[g]
        at = np.where(nodes >= 0, nodes, 0)
        tr = np.where(nodes >= 0, masses.token_row_host[at], -2)           # -2: no node
        real = tr != -2
        first = np.where(real.any(axis = 1), tr[np.arange(len(tr)), real.argmax(axis = 1)], -1)
        mixed = ~(np.where(real, tr, first[:, None]) == first[:, None]).all(axis = 1)
        kept = rows[g] = (weakref.ref(masses), first, mixed)
    first, mixed = kept[1][pos], kept[2][pos]
    if mixed.any() or (masses.token_ids is not None and (first < 0).any() and (first >= 0).any()):
        raise NotImplementedError("Conditionals at a position whose input nodes are reweighted by different per-sample "
                                  "inputs (or some only) are not supported.")
    if (first < 0).all():
        return None
    nodes = prog.__dict__.setdefault("sample_nodes", {})
    if g not in nodes:
        nodes[g] = torch.from_numpy(np.maximum(score.nodes[g], 0)).to(prog.dev)
    off = first[:, None]
    offset = np.where(off >= 0, (off + np.arange(B)[None, :] * masses.token_stride) * masses.token_width, -1)
    return nodes[g], np.asarray(pos, dtype = np.int64), offset.reshape(-1)


def _as_int64(a: np.ndarray) -> np.ndarray:
    """``a``'s bytes as int64 (zero-padded to whole ones): packed with int64 tables, viewed back as ``a``'s dtype on the
    device."""
    b = np.ascontiguousarray(a).view(np.uint8).ravel()
    return np.concatenate([b, np.zeros((-b.size) % 8, dtype = np.uint8)]).view(np.int64)


def _scope_ivs(td, external) -> np.ndarray:
    """:func:`~.plans.scope_intervals` of a session's gated scopes, kept on its inputs (for the program ``td``)."""
    kept = getattr(external, "_scope_ivs", None)
    if kept is None or kept[0] is not td:
        kept = external._scope_ivs = (td, scope_intervals(td, external.scopes))
    return kept[1]


def _check(backend, data: torch.Tensor, precision: str, storage: str = "dense"):
    if precision not in F.PRECISIONS:
        raise ValueError(f"`precision` must be one of {sorted(F.PRECISIONS)}, got {precision!r}.")
    if storage not in STORAGES:
        raise ValueError(f"`storage` must be one of {list(STORAGES)}, got {storage!r}.")
    if backend.device.type != "cuda":
        raise NotImplementedError(f"The dfa_exact backend runs on CUDA only, but the constrained circuit is on "
                                  f"{backend.device}. Move it with `cc.to('cuda')`.")
    if data.dim() != 2 or data.size(1) != backend.n:
        raise ValueError(f"`data` must be [batch, {backend.n}], got {list(data.shape)}.")


def class_weights(backend, data: torch.Tensor, missing: torch.Tensor, positions: np.ndarray, precision: str = "fp32",
                  held: Optional[dict] = None, observed: Optional[np.ndarray] = None, layout = None,
                  report: Optional[dict] = None, storage: str = "auto", carry: Optional[dict] = None,
                  external = None, masses = None, evidence: Optional[tuple] = None):
    """
    The forward and top-down passes of a conditional query: the log class weights of every input node over the
    target ``positions`` (bool [n], host), every sample's, summed per row of the class masses (see
    :mod:`.score`), in ``dfa_class_flow``.

    :param missing: [B, n] bool on the device, True where a token is missing
    :param held: see :func:`conditional`
    :param observed: None, or bool [n] (host): the positions some sample observes (read off ``missing`` if None)
    :param report: None, or a dict where the query leaves its :class:`~.plans.QueryPlan` (``"plan"``), its
        :class:`~..kernels.live.Liveness` (``"liveness"``; None when it observes nothing, dense), its
        :class:`~..kernels.compact.CompactLayout` (``"compact"``; None when dense), the storage it used (``"storage"``,
        "dense" or "compact"), whether it computed again the values held in the other one (``"dropped"``) and the
        storages holding the held values it did not read (``"parked"``, see :func:`~..kernels.compact.compact_buffers`)
    :param layout: None, or ``(groups, size, (base, K))``: :func:`_layout`'s groups and size, and its ``base`` and
        ``K`` already on the device (int64 [n] each) -- copied before the query launches anything, as copies from the
        host wait for the stream
    :param storage: see :func:`conditional`
    :param carry: see :func:`conditional`
    :param external: see :func:`conditional`
    :param masses: None, or the query's class masses (:meth:`~..state.State.masses` of ``external``) if already
        taken
    :param evidence: None, or ``(x, obs_class)`` on the device if already built: the tokens (int64 [B, n], 0 where
        missing) and their classes (int32 [B, n], -1 where missing)
    :returns: ``(groups, masses)``: per row group with a target position, ``(group, its target positions (host),
        weights [positions x B, classes, rows of the group])`` (sample ``b`` of the ``p``-th position: column ``p * B +
        b``), and the class masses the query read (:class:`~..state.ClassMasses`)
    """
    prog, td, dev = backend.program(), backend.top_down_program(), backend.device
    score = score_tables(prog)
    B = data.size(0)
    if layout is None:
        groups, base, K, size = _layout(backend, positions, B)
        layout = (groups, size, (torch.from_numpy(base).to(dev), torch.from_numpy(K).to(dev)))
    groups, size, (base, K) = layout
    if observed is None:
        observed = (~missing).any(dim = 0).cpu().numpy()
    if external is not None:                                                # values that differ per sample
        observed = observed | external.observed
    if evidence is None:
        x = torch.where(missing, 0, data.to(dev).long())                      # missing tokens: any valid id
        obs_class = torch.where(missing, -1, backend.token_classes.token_class[x]).contiguous()
    else:
        x, obs_class = evidence
    held = held or {}
    masses = backend.state.masses(external) if masses is None else masses
    succ = live_successors(prog, masses)                # (per-sample inputs that rule classes out: theirs)
    # compact, a query needs its liveness even with nothing observed: every column is live, but only the regions it
    # computes per sample get rows (none, every value input-independent, inside fast_inference)
    lv = liveness(prog, obs_class, B, succ) if observed.any() or storage != "dense" or succ is not None else None
    valid, kept = backend.state.prior_validity(masses.version)
    prior = prior_regions(prog, observed, B, kept, lv)
    pf = backend.state.prior_flows(np.zeros(len(backend.sum_regions), dtype = bool))
    once, pf_valid = top_down_prior(prog, td, observed, B, kept, lv), pf["valid"].copy()
    if external is not None and external.scopes:                            # under gates of a batch: their nodes'
        once = once & ~_scope_ivs(td, external)                             # children's values are every sample's own

    def plan_for(held):
        top = dict(program = td, once = once, valid = pf_valid, held = held.get("top_down"), scoring = True)
        return query_plan(prog, observed, positions, valid.numpy(), prior, held = held.get("inside"), top_down = top)

    plan = plan_for(held)
    move = held.get("inside")                           # compact: the held inside values it reads get its rows
    if move is not None and carry is not None and worth_parking(prog, carry):
        move = held_moves(prog, carry, move & read_regions(prog, plan))
    storage, compact = resolve_storage(backend, storage, B, True, lv, plan, plan.top_down.per_sample, move)
    dropped = held_elsewhere(storage, held.values(), carry)
    if dropped:                                         # held in the other storage: computed again
        held, carry, move = {}, None, None
        plan = plan_for(held)
        if compact is not None:
            compact = query_layout(backend, lv, plan, plan.top_down.per_sample, None)
    inputs_only = compact is not None and not plan.sample.any() and (move is None or not move.any())
    parked = []
    if compact is not None:                             # rows for the per-sample values only, walked in their own
        bufs, parked = compact_buffers(backend, compact, B, inputs_only, carry, move)   # order, the held ones it
    else:                                                                              # reads re-gathered into them
        if lv is not None and not observed.any() and succ is None:   # (laid out for "auto", but dense: nothing to
            lv = None                                                 # prune)
        if lv is not None:
            lv.order(per_sample_intervals(prog, plan))
        bufs = backend.state.buffers(B)
    if compact is not None and carry is not None and carry["regions"].any():
        flows = regather_into(backend.state, prog, carry["layout"], compact, carry["regions"], carry["parents"],
                              lambda: backend.state.flows(B, bufs, compact))
    else:
        flows = backend.state.flows(B, bufs, compact)
    if report is not None:
        report.update(plan = plan, liveness = lv, compact = compact, inputs_only = inputs_only, storage = storage,
                      dropped = dropped, parked = parked)
    gates = F.gate_context(backend, external)
    F.run_forward(backend, bufs, plan, x, missing, obs_class, lv, precision, compact, masses, gates = gates)
    valid |= plan.todo

    C = backend.num_classes
    buf = backend.state.storage_view("dfa_class_flow", size)
    if plan.top_down.cw_fill:                           # (every weight stored by the launch writing it first: no fill)
        buf.fill_(-float("inf"))
    cw = dict(buf = buf, base = base, K = K)
    run_backward(backend, bufs, flows, plan, x, missing, obs_class, lv, precision, class_weights = cw,
                 masses = masses, compact = compact, gates = gates)
    return [(g, pos, buf[s:s + len(pos) * B * C * len(score.rows[g])].view(len(pos) * B, C, len(score.rows[g])))
            for g, pos, s in groups], masses


def conditional(backend, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None, precision: str = "bf16", return_log_marginal: bool = False,
                held: Optional[dict] = None, candidates: Optional[torch.Tensor] = None, report: Optional[dict] = None,
                storage: str = "auto", carry: Optional[dict] = None, external = None):
    """
    ``log p(x_t = v | e, C)`` for every target ``(b, t)`` and token ``v`` (or candidate ``v``): one row per target, in
    the order ``dense[targets]`` lists them (sample by sample, positions in order); -inf rows where ``p(e, C) = 0``.

    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True where a token is marginalized
    :param targets: None (every missing position of every sample), or a bool mask [n] or [B, n] of missing positions
    :param precision: "bf16" (the default: the conditionals as accurate as at "tf32"), "fp32" (fp32-level) or
        "tf32" (see :data:`~.forward.PRECISIONS`), for the sum layers' and the scoring's matrix products -- the log
        marginals too (``return_log_marginal``); a token table in bf16 (inside a ``pyjuice.fast_inference`` scope
        allowing bf16 parameter copies) is read in tf32
    :param return_log_marginal: also return ``log p(e, C)`` of every row's sample, [rows]
    :param held: None, or what the buffers hold for this query, as a caching engine decides (see :mod:`.plans`):
        a dict with ``inside`` and ``top_down``, each None or bool [sum regions]
    :param candidates: None (every token), or [B, n, k] token ids (see :mod:`pyjuice.constraints.evidence`): a target
        ``(b, t)``'s row scores ``candidates[b, t]`` only (:func:`~.score.score_group`); ids outside the vocabulary
        pad, and get -inf
    :param report: None, or a dict where the query leaves its :class:`~.plans.QueryPlan`,
        :class:`~..kernels.live.Liveness` and :class:`~..kernels.compact.CompactLayout` (see :func:`class_weights`;
        none when it computes nothing)
    :param storage: "auto" (the default), "dense" or "compact": how the query keeps its per-sample values -- every
        lane of every block, or only the pairs of each sample's live columns, and only of the regions the query needs
        per sample (see :mod:`~..kernels.compact`) -- and, compact, no per-sample inside value at all where it computes
        and holds none (a prefix's next token on an HMM: every per-sample scope holds the prefix's last position).
        Compact takes far less memory when the evidence leaves few live columns, and skips the dead lanes' work; but
        every pair of a sample's live columns has a lane, joined or not, so it takes more where the automaton joins
        few of them (sparse evidence on a wide automaton). "auto" picks the one whose buffers are smaller
        (:func:`~..kernels.compact.resolve_storage`); values held in the other storage are computed again
    :param carry: compact, the values ``held`` marks are in ``node_mars`` and ``node_flows`` as a previous query laid
        them out, not yet as this one does: a dict with that query's ``layout`` (its
        :class:`~..kernels.compact.CompactLayout`), the regions whose inside (``inside``) and top-down (``regions``)
        values it holds (bool [sum regions] each) and every row's ``parents`` there (int64 [B] on the device), and the
        inside values earlier queries left in storages of their own (``parked``, see
        :func:`~..kernels.compact.compact_buffers`); those the query reads are re-gathered
        (:func:`~..kernels.compact.regather`), the other inside values stay where they are
    :param external: None, or a decoder session's per-sample inputs (:class:`~..state.ExternalInputs`), e.g. soft
        evidence: every input node's tokens then weigh ``q(v) = p(v) w(v) / Z`` -- a target's too (or ``p(v) w(v)`` at
        a position its value mask leaves free: missing too, so a target may be there)
    :returns: [rows, V], or [rows, k] with ``candidates``
    """
    _check(backend, data, precision, storage)
    n, dev = backend.n, backend.device
    B = data.size(0)
    if external is not None and external.batch not in (1, B):
        raise ValueError(f"The decoder's per-sample inputs have a batch of {external.batch}: a query's must be the "
                         f"same (or theirs 1), got {B}.")
    if external is not None and external.free is not None:                 # (a decoder's queries have them already)
        miss = np.zeros((B, n), dtype = bool) if missing_mask is None else _host_mask(missing_mask, B, n, "missing_mask")
        missing_mask = torch.from_numpy(miss | external.free_host)
    miss, tgt = _masks(missing_mask, targets, B, n)                          # on the host: the query's one wait
    V = backend.token_classes.token_class.numel()
    width = V if candidates is None else candidates.size(2)
    rank = np.cumsum(tgt.ravel()).reshape(B, n) - 1                          # a target's row
    M = int(tgt.sum())
    if M == 0 or not backend.satisfiable:
        out = torch.full((M, width), -float("inf"), device = dev)
        return (out, torch.full((M,), -float("inf"), device = dev)) if return_log_marginal else out

    prog = backend.program()
    score = score_tables(prog)
    positions = tgt.any(axis = 0)
    groups, base, K, size = _layout(backend, positions, B)
    masses = backend.state.masses(external)                                  # (the query's, read twice)
    cols = [_sample_columns(masses, score, g, pos, B, prog) for g, pos, _ in groups]
    # every row group's columns' rows (-1: a sample not targeting the position), with the layout, every row's (sample,
    # position) under candidates, and the per-sample columns' positions and weights' offsets: in one copy
    out_rows = [np.where(tgt[:, pos], rank[:, pos], -1).T.ravel() for _, pos, _ in groups]     # column p * B + b
    rows_bt = list(np.nonzero(tgt)) if candidates is not None else []          # every row's (sample, position)
    G = len(groups)
    ev = []
    if data.device.type == "cpu":                       # (a decoder's evidence, kept on the host): every token, and
        x_h = np.where(miss, 0, data.numpy()).astype(np.int64)                 # its class, built here, in the copy
        ev = [x_h.ravel(), _as_int64(np.where(miss, -1, backend.token_classes.host()[x_h]).astype(np.int32)),
              _as_int64(miss)]
    tables, starts = pack_aligned([base, K, *out_rows, *rows_bt, *(a for c in cols if c is not None for a in c[1:]),
                                   *ev], np.int64)
    tables = torch.from_numpy(tables).to(dev)
    starts = starts.tolist()
    view = lambda i: tables[starts[i]:starts[i + 1]]
    cand = None
    if candidates is not None:                                                # every row's candidates, [M, k]
        sb, st = starts[2 + G], starts[3 + G]
        cand = candidates.to(dev)[tables[sb:sb + M], tables[st:st + M]].to(torch.int32).contiguous()
    samples, i = [], 2 + G + len(rows_bt)
    for c in cols:
        samples.append(None if c is None else
                       dict(nodes = c[0], positions = view(i)[:len(c[1])], row = masses.row, stride = masses.stride,
                            columns = B, lognorm = masses.lognorm, logw = masses.token_logw,
                            offset = view(i + 1)[:len(c[2])], width = masses.token_width, ids = masses.token_ids))
        i += 0 if c is None else 2
    evidence = None
    if ev:                                              # (the last three arrays)
        j = len(starts) - 4
        x, missing = view(j)[:B * n].view(B, n), view(j + 2).view(torch.bool)[:B * n].view(B, n)
        evidence = (x, view(j + 1).view(torch.int32)[:B * n].view(B, n))
    else:
        missing = F.missing_tensor(missing_mask, B, n, dev) if missing_mask is None or missing_mask.is_cuda else \
            torch.from_numpy(miss).to(dev)
    with torch.no_grad():
        layout = (groups, size, (tables[starts[0]:starts[0] + n], tables[starts[1]:starts[1] + n]))
        groups, _ = class_weights(backend, data, missing, positions, precision, held,
                                  observed = (~miss).any(axis = 0), layout = layout, report = report,
                                  storage = storage, carry = carry, external = external, masses = masses,
                                  evidence = evidence)
        if masses.token_ids is not None and cand is None:                   # top-k weights: only theirs are written
            out = torch.full((M, width), -float("inf"), device = dev)
        else:
            out = torch.empty(M, width, device = dev)
        log_marginal = torch.empty(M + 1, device = dev) if return_log_marginal else None
        for (g, pos, w), start, sample in zip(groups, starts[2:], samples):
            out_row = tables[start:start + len(pos) * B]
            table = backend.state.token_table(g, score.rows[g], precision)
            logz = score_group(out, table, w, score.rows_t[g], masses.table, out_row, backend.token_classes,
                               F.PRECISIONS[precision], candidates = cand, sample = sample)
            if log_marginal is not None:
                log_marginal[torch.where(out_row >= 0, out_row, M)] = logz
    return (out, log_marginal[:M]) if return_log_marginal else out
