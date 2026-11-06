"""
The class weights through the pairs' class sets (:attr:`~..transitions.tables.BoundaryGroups.class_sets`).

A class's weights at a target column are a sum over the (column, successor) pairs it leads by that are live for the sample. A pair is led by a SET of classes, and the
sets are few however many the pairs, so the same pair is summed once a class of its set: 5 to 14 times on the wide automata of JSON Schemas. Here every live pair is summed
once, into its set's row (:func:`_set_weight_kernel`: the class weights' terms, :func:`~.class_weight_kernels._cw_input_block_term` and
:func:`~.class_weight_kernels._cw_block_input_term`, over a set's live pairs), and a class's row is then the sum of the rows of the sets it is in
(:func:`_set_combine_kernel`). The pairs partition by set, so a class's sum has the same terms, grouped by set: the same value up to rounding, not to the bit.

The lists are a query's own (:func:`set_weight_lists`): only the pairs live at both ends for the sample are in them, found from its live columns, so a walk reads
no dead pair -- what makes the walk over a class's whole list lose under evidence. The classes' lists (:meth:`~...compiled.program.Program.class_lists`) are not read.

Where it applies is a rule on the automaton alone (:func:`set_weights_apply`), so a query's values do not depend on timings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ..common import IDENTITY, NODE_TILE, _block_ptr, _exit_width, _interval, _logaddexp_tile, _tile_rows, grid_chunks, pdl_launch, tensor
from ..prod.launches import _lse_step_one_exp, _lse_value
from ..prod_top_down.launches import _once, _out_flows_ptr, _parent_base
from ..transitions.tables import BoundaryGroups, _ranges, _sort_key, boundary_groups
from .class_weight_kernels import CLASS_WEIGHT_CUDA, CLASS_WEIGHT_CUDA_CLASSES, CLASS_WEIGHT_CUDA_SMEM, _cw_block_input_term, _cw_input_block_term


#: Whether a query's class weights go through its live pairs' class sets where :func:`set_weights_apply` says so; "force": wherever they can (tests)
SET_WEIGHTS = True
#: ... on an automaton whose classes' lists hold at least ``SET_WEIGHT_MIN_RATIO`` times its pairs, and at least ``SET_WEIGHT_MIN_ENTRIES`` entries a boundary
SET_WEIGHT_MIN_RATIO = 2.0
SET_WEIGHT_MIN_ENTRIES = 1 << 16
#: The most pairs a query's lists may be built from (8 bytes each kept, several times that while they are built), and the most (list group, set) cells of their
#: rank table: past either, the classes' own walks; and the fewest pairs worth the lists: below, the classes' walks are short too (a prefix's next token on K 286,841:
#: 28 pairs, 6.9 ms by the classes' walks, 9.3 through the sets). MEASURED GPT-2's vocabulary, HMM-64, 64 positions, batch 2, settled conditionals with a prefix, scattered
#: tokens, one target or nothing observed, on the six automata of the corpus this rule takes: from 2^12 to 2^14 pairs on, the sets faster on 17 of 17 queries,
#: 40.4 -> 15.1 s in all; below, slower on 6 of 6
SET_WEIGHT_PAIRS_MOST = 1 << 27
SET_WEIGHT_RANKS_MOST = 1 << 27
SET_WEIGHT_PAIRS_LEAST = 1 << 14


@dataclass
class SetWeights:
    """
    A query's live pairs by class set (:func:`set_weight_lists`). A LIST GROUP is a target column (position, sample), or under no pruning a position, whose samples then
    walk the same lists; its live sets are its ROWS, the longest list first, so that a program's lanes walk lists of a length.

    :ivar count: int32 [N], every target column's rows
    :ivar first: int64 [N], its first list
    :ivar off: int64 [N + 1], its first float in the set weights' buffer, a row ``K`` floats as the classes' are (the last: the size)
    :ivar lptr: int64 [lists + 1], every list's pairs in ``pair_i`` / ``pair_q`` (int32: the column, its successor's column; a sequence end's: its one column)
    :ivar grp: int64 [N], every target column's list group
    :ivar rank: int32 [groups, S], every set's row among its group's, -1 without a live pair
    :ivar most, size: the most rows of a column, and the buffer's floats
    :ivar pairs: the pairs of the live columns' states the lists were found among
    """

    count: torch.Tensor
    first: torch.Tensor
    off: torch.Tensor
    lptr: torch.Tensor
    pair_i: torch.Tensor
    pair_q: torch.Tensor
    grp: torch.Tensor
    rank: torch.Tensor
    most: int
    size: int
    S: int
    pairs: int = 0


def set_weights_apply(prog, liveness, compact) -> bool:
    """Whether a query of ``prog`` takes its class weights through the class sets: never while a CUDA graph is captured (the lists are the query's own), nor on tables made
    on the host (a small automaton), nor a query the CUDA kernels take (:func:`~.class_weight_kernels._class_weight_cuda`: compact values under pruning, few classes, and
    boundaries whose rows fit a CTA's shared memory: left as they are; past that width they decline, and the walk is Triton's); else by what the automaton's lists hold
    (:data:`SET_WEIGHT_MIN_RATIO`, :data:`SET_WEIGHT_MIN_ENTRIES`). Host values alone.

    MEASURED GPT-2's vocabulary, HMM-64, 64 positions, batch 2, every automaton of the corpus on the device's tables forced through the sets, a prefix, scattered tokens, one
    target or nothing observed: of the queries this takes, 41 of 41 faster, 49.5 -> 20.2 s in all, and no faster query left out; with the width at 8,192, where the CUDA kernels
    still fit, 6 of 47 slower (K 11,845 with nothing observed: 65 -> 156 ms)."""
    if not SET_WEIGHTS or torch.cuda.is_current_stream_capturing() or not isinstance(boundary_groups(prog._tr), BoundaryGroups):
        return False
    if SET_WEIGHTS == "force":
        return True
    if CLASS_WEIGHT_CUDA and liveness is not None and compact is not None and prog.C <= CLASS_WEIGHT_CUDA_CLASSES and 8 * prog.W <= CLASS_WEIGHT_CUDA_SMEM:
        return False                                                 # (a destination's rows at both boundaries, 4 bytes a column: what a CTA stages)
    entries, pairs = prog.transition_stats[0], int(prog.trans_pairs.sum())
    return entries >= SET_WEIGHT_MIN_RATIO * pairs and entries >= SET_WEIGHT_MIN_ENTRIES * prog.n


def _class_sets_of(prog) -> tuple:
    """``(ptr, sets, most)``: every class's sets, ``sets[ptr[c] .. ptr[c + 1]]`` (int32 on the device), and the most of any class; kept with the program."""
    kept = getattr(prog, "_class_set_tables", None)
    if kept is None:
        _, member, in_set, _, _ = boundary_groups(prog._tr).class_sets
        member = member.long()
        per = torch.bincount(member, minlength = prog.C)
        ptr = torch.zeros(prog.C + 1, dtype = torch.int64, device = prog.dev)
        ptr[1:] = torch.cumsum(per, dim = 0)
        sets = torch.cat([in_set.index_select(0, torch.argsort(member, stable = True)), torch.zeros(1, dtype = in_set.dtype, device = prog.dev)])
        kept = prog._class_set_tables = (ptr.to(torch.int32).contiguous(), sets.to(torch.int32).contiguous(), int(per.max()) if per.numel() else 0)
    return kept


def set_weight_lists(prog, liveness, col_pos: torch.Tensor, col_K: torch.Tensor, B: int, most_columns: Optional[list] = None) -> Optional[SetWeights]:
    """
    The :class:`SetWeights` of a query's target columns (``col_pos`` int64 [N / B], every block of ``B`` columns' position; ``col_K`` int64 [N], the floats of a row),
    from its :class:`~..live.Liveness` (None: every column live), or None where they are too many (:data:`SET_WEIGHT_PAIRS_MOST`, :data:`SET_WEIGHT_RANKS_MOST`) or
    there are none, or too few to be worth the lists (:data:`SET_WEIGHT_PAIRS_LEAST`): first by the host's count, where it has one -- ``most_columns``, per target
    position the most live columns of any sample there, each of at most the tables' most pairs --, so that a query of few live columns asks nothing of the device. A
    group's live columns, their states' pairs (:attr:`~..transitions.tables.BoundaryGroups.pairs`), those whose successor is a live column of the next boundary; by
    group, set, then column: a stable sort. It waits for the device, twice: for how many pairs there are.
    """
    if most_columns is not None and B * sum(most_columns) * max(prog.succ.max_pairs, 1) < SET_WEIGHT_PAIRS_LEAST:
        return None
    groups = boundary_groups(prog._tr)
    n, W, dev = prog.n, prog.W, prog.dev
    p_count, p_start, p_succ, _ = groups.pairs
    of_pair, _, _, _, in_each = groups.class_sets
    S, P = in_each.numel(), col_pos.numel()
    N = P * B
    shared = liveness is None
    G = P if shared else N
    if N == 0 or S == 0 or G * S > SET_WEIGHT_RANKS_MOST:
        return None
    if shared:                                                       # every column of the position
        cnt = prog.width_t.index_select(0, col_pos).long()
        t_g = col_pos
        columns = int(cnt.sum())
        icol = _ranges(torch.zeros_like(cnt), cnt, columns)
    else:                                                            # the sample's live columns there, ascending
        count, start, live_columns = liveness.lists()
        rows = (col_pos[:, None] + torch.arange(B, device = dev)[None, :] * (n + 1)).reshape(-1)
        cnt = count.index_select(0, rows).long()
        t_g = col_pos.repeat_interleave(B)
        columns = int(cnt.sum())
        icol = live_columns.index_select(0, _ranges(start.index_select(0, rows), cnt, columns)).long()
    gl = torch.repeat_interleave(torch.arange(G, device = dev), cnt)
    t_l = t_g.index_select(0, gl)
    state = prog.state_id.reshape(-1).index_select(0, t_l * W + icol).long()
    pc = p_count.index_select(0, state)
    total = int(pc.sum())
    if total == 0 or total > SET_WEIGHT_PAIRS_MOST or total < SET_WEIGHT_PAIRS_LEAST:
        return None
    of_l = torch.repeat_interleave(torch.arange(columns, device = dev), pc)
    p = _ranges(p_start.index_select(0, state), pc, total)
    del state, pc
    t_next = t_l.index_select(0, of_l) + 1
    q = prog.col_of_state.reshape(-1).index_select(0, t_next * prog.K + p_succ.index_select(0, p)).long()
    q = torch.where((t_next == n) & (q >= 0), torch.zeros_like(q), q)      # (a sequence end: its one column)
    g = gl.index_select(0, of_l)
    keep = q >= 0
    if not shared:
        keep &= liveness.live.reshape(-1).index_select(0, ((g % B) * (n + 1) + t_next) * W + q.clamp(min = 0)) != 0
    del t_next
    g, q, i, x = g[keep], q[keep], icol.index_select(0, of_l)[keep], of_pair.index_select(0, p[keep]).long()
    del of_l, p, keep, gl, t_l, icol
    if g.numel() == 0:
        return None
    counts = torch.bincount(g * S + x, minlength = G * S).view(G, S)
    order = torch.argsort(-counts, dim = 1, stable = True)           # a group's sets, the longest list first
    rows_of = (counts > 0).sum(dim = 1)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(S, device = dev).expand(G, S))
    by = torch.argsort(_sort_key(g * S + rank.reshape(-1).index_select(0, g * S + x), G * S), stable = True)
    lens = torch.gather(counts, 1, order)[torch.arange(S, device = dev)[None, :] < rows_of[:, None]]
    lptr = torch.zeros(lens.numel() + 1, dtype = torch.int64, device = dev)
    lptr[1:] = torch.cumsum(lens, dim = 0)
    grp = torch.arange(N, device = dev) // B if shared else torch.arange(N, device = dev)
    count = rows_of.index_select(0, grp)
    off = torch.zeros(N + 1, dtype = torch.int64, device = dev)
    off[1:] = torch.cumsum(count * col_K, dim = 0)
    return SetWeights(count = count.to(torch.int32).contiguous(), first = (torch.cumsum(rows_of, dim = 0) - rows_of).index_select(0, grp).contiguous(), off = off, lptr = lptr,
                      pair_i = i.index_select(0, by).to(torch.int32).contiguous(), pair_q = q.index_select(0, by).to(torch.int32).contiguous(), grp = grp.contiguous(),
                      rank = torch.where(counts > 0, rank, -1).to(torch.int32).contiguous(), most = int(rows_of.max()), size = int(off[-1]), S = S, pairs = total)


@triton_jit
def _set_weight_kernel(set_flow, cw_col, cw_K, s_count, s_first, s_off, lptr, pair_i, pair_q, element_flows, temp_flows, node_mars, prior_mars, temp, width, iv_of,
                       iv_info, iv_table, once_iv, count, coff, crank, dests, tiles, ptr, parents, reg_offset, reg_width, reg_first, prior_offset, prior_width,
                       reg_prior, out_first, out_width, temp_width, tflow_width, B, n, W, pid0, BLOCK_INPUT: tl.constexpr, S_KIND: tl.constexpr,
                       OUT_TEMP: tl.constexpr, OUT_FULL: tl.constexpr, S_FULL: tl.constexpr, COMPACT: tl.constexpr, MARS: tl.constexpr, STORE: tl.constexpr,
                       TN: tl.constexpr, SUBS: tl.constexpr, TB: tl.constexpr, TC: tl.constexpr, TJ: tl.constexpr, PDL: tl.constexpr = False):
    """:func:`~.class_weight_kernels._class_weight_kernel`'s walk over a class's columns, a lane a (sample, row of its target column's live sets) and its list the row's
    live pairs (:class:`SetWeights`): the same programs, destinations and terms, every pair of the list a term (none is dead), log-added to the lane's row of ``set_flow``
    or stored there (``STORE``)."""
    if PDL:                                             # (the launch before it, done)
        gdc_wait()
    d, real, d0 = _tile_rows(tiles, tl.program_id(0), TN, SUBS)
    lane = tl.arange(0, TB * TC)
    b = ((pid0 + tl.program_id(1)) * TB + lane // TC).to(tl.int64)
    t = tl.load(dests + d0 * 2)
    rank = tl.program_id(2) * TC + lane % TC
    cm = b < B
    nb = tl.load(cw_col + t) + b                                               # the lanes' columns: their rows
    cm = cm & (rank < tl.load(s_count + nb, mask = cm, other = 0))
    row = tl.load(s_first + nb, mask = cm, other = 0) + rank
    p0c = tl.load(lptr + row, mask = cm, other = 0)                            # the lanes' lists
    p1c = tl.load(lptr + row + 1, mask = cm, other = 0)
    L = tl.max(p1c - p0c, axis = 0)
    vm = real[:, None, None] & cm[None, :, None]                               # (destination, lane, column)
    k = tl.load(dests + d * 2 + 1)
    end = t + 1 == n
    M = tl.load(width + t).to(tl.int32)
    mx = tl.full([TN, TB * TC, TJ], float("-inf"), dtype = tl.float32)
    acc = tl.zeros([TN, TB * TC, TJ], dtype = tl.float32)
    p0 = tl.load(ptr + d)
    f0 = tl.load(ptr + d0)
    for r in range(0, tl.load(ptr + d0 + 1) - f0):
        p = p0 + r
        f = f0 + r                                                             # (the tile's first destination's step)
        op = _out_flows_ptr(tl.load(parents + p * 7 + 0), element_flows, temp_flows, out_first, out_width, tflow_width, OUT_TEMP)[:, None, None]
        if BLOCK_INPUT:                                                        # a block over [t0, t), then the input
            t0 = tl.load(parents + f * 7 + 3)
            E = tl.load(width + t0)
            X = _exit_width(width, t + 1, n)
            OS, opos, _ = _interval(iv_of, iv_info, t0, t + 1, n)
            LS, lpos, _ = _interval(iv_of, iv_info, t0, t, n)
            once = _once(once_iv, iv_of, t0, t + 1, n)
            ostride = tl.where(once, 0, OS)
            kp = tl.load(iv_of + t0 * (n + 1) + t + 1)
            op = op + _parent_base(b, ostride, COMPACT)[None, :, None]
            lp, l_own = _block_ptr(tl.load(parents + p * 7 + 1), tl.load(parents + p * 7 + 2), node_mars, prior_mars, temp, reg_offset, reg_width, reg_first,
                                   prior_offset, prior_width, reg_prior, temp_width, S_KIND)
            lp = lp[:, None, None] + _parent_base(b[None, :], tl.where(l_own, LS, 0)[:, None], MARS)[:, :, None]
            kl = tl.load(iv_of + t0 * (n + 1) + t)
            for ii in range(0, L):
                ok = cm & (p0c + ii < p1c)
                i = tl.load(pair_i + p0c + ii, mask = ok, other = 0)
                q = tl.load(pair_q + p0c + ii, mask = ok, other = -1)
                mx, acc = _cw_block_input_term(mx, acc, i, ok, q, op, lp, iv_table, lpos, opos, coff, count, crank, kl, kp, b, vm, once, l_own, M, E, X, t0, t, end,
                                               B, n, W, TJ, S_FULL, OUT_FULL, COMPACT, MARS, False)
        else:                                                                  # the input, then a block over [t + 1, t2)
            t2 = tl.load(parents + f * 7 + 5)
            X = _exit_width(width, t2, n)
            OS, opos, _ = _interval(iv_of, iv_info, t, t2, n)
            once = _once(once_iv, iv_of, t, t2, n)
            ostride = tl.where(once, 0, OS)
            kp = tl.load(iv_of + t * (n + 1) + t2)
            op = op + _parent_base(b, ostride, COMPACT)[None, :, None]
            if S_KIND != 2:
                RS, rpos, _ = _interval(iv_of, iv_info, t + 1, t2, n)
                rp, r_own = _block_ptr(tl.load(parents + p * 7 + 3), tl.load(parents + p * 7 + 4), node_mars, prior_mars, temp, reg_offset, reg_width, reg_first,
                                       prior_offset, prior_width, reg_prior, temp_width, S_KIND)
                rp = rp[:, None, None] + _parent_base(b[None, :], tl.where(r_own, RS, 0)[:, None], MARS)[:, :, None]
                kr = tl.load(iv_of + (t + 1) * (n + 1) + t2)
            else:                                                              # (unread)
                rpos = opos
                rp = op
                r_own = once
                kr = kp
            for ii in range(0, L):
                ok = cm & (p0c + ii < p1c)
                i = tl.load(pair_i + p0c + ii, mask = ok, other = 0)
                q = tl.load(pair_q + p0c + ii, mask = ok, other = -1)
                mx, acc = _cw_input_block_term(mx, acc, i, ok, q, op, rp, iv_table, opos, rpos, coff, count, crank, kp, kr, b, vm, once, r_own, X, t, t2, end, B, n,
                                               W, TJ, S_KIND, S_FULL, OUT_FULL, COMPACT, MARS, False)
    top = tl.max(mx, axis = 2)
    shift = tl.where(top == float("-inf"), 0.0, top)
    val = tl.log(tl.sum(acc * tl.exp(mx - shift[:, :, None]), axis = 2)) + shift
    dest = set_flow + (tl.load(s_off + nb, mask = cm, other = 0) + rank * tl.load(cw_K + t))[None, :] + k[:, None]
    dm = real[:, None] & cm[None, :]
    if STORE:
        tl.store(dest, val, mask = dm)
    else:
        tl.store(dest, _logaddexp_tile(tl.load(dest, mask = dm, other = float("-inf")), val), mask = dm)


@triton_jit
def _set_combine_kernel(class_flow, set_flow, cw_count, cw_cls, cw_off, s_off, s_grp, s_rank, cs_ptr, cs_set, col_K, C, S, J, TR: tl.constexpr, TK: tl.constexpr):
    """One program per target column ``m`` and ``TR`` of its class rows (``cw_cls[m, :cw_count[m]]``, see :class:`~.class_rows.ClassWeightRows`): each row the sum, in
    log, of the rows of the sets its class is in (``cs_ptr`` / ``cs_set``, at most ``J`` a class) that have a row at the column's list group (``s_rank``), in the sets'
    order, ``TK`` floats a step; stored: every float of a class's row is written."""
    m = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1) * TR + tl.arange(0, TR)
    rm = r < tl.load(cw_count + m)
    c = tl.load(cw_cls + m * C + r, mask = rm, other = 0)
    a0 = tl.load(cs_ptr + c, mask = rm, other = 0)
    a1 = tl.load(cs_ptr + c + 1, mask = rm, other = 0)
    K = tl.load(col_K + m)
    g = tl.load(s_grp + m)
    src = set_flow + tl.load(s_off + m)
    dst = class_flow + tl.load(cw_off + m)
    for k0 in range(0, K, TK):
        k = k0 + tl.arange(0, TK)
        km = k < K
        mx = tl.full([TR, TK], float("-inf"), dtype = tl.float32)
        acc = tl.zeros([TR, TK], dtype = tl.float32)
        for j in range(0, J):
            jm = rm & (a0 + j < a1)
            x = tl.load(cs_set + a0 + j, mask = jm, other = 0)
            rk = tl.load(s_rank + g * S + x, mask = jm, other = -1)
            on = (jm & (rk >= 0))[:, None] & km[None, :]
            v = tl.load(src + (rk.to(tl.int64) * K)[:, None] + k[None, :], mask = on, other = float("-inf"))
            mx, acc = _lse_step_one_exp(mx, acc, v)
        tl.store(dst + (r.to(tl.int64) * K)[:, None] + k[None, :], _lse_value(mx, acc), mask = rm[:, None] & km[None, :])


def run_set_weights(launches, cw: dict, prog, bufs, element_flows, temp_flows, node_mars, liveness, elem_first: int, elem_width: int, B: int, prior: dict,
                    once_iv: torch.Tensor, regions: dict, compact = None, tflow_width: Optional[int] = None, inside: bool = False) -> None:
    """:func:`~.class_weights.run_class_weights`' launches into the sets' rows (``cw["sets"]``, ``cw["set_buf"]``): the same tiles and lanes, a lane a row of sets."""
    from . import class_weights as CW
    sw = cw["sets"]
    temp, temp_width = bufs
    tflow_width = temp_width if tflow_width is None else tflow_width
    n, W = prog.n, prog.W
    ivs = prog.intervals
    count = liveness.lists()[0] if liveness is not None else cw["K"]
    ctables = (compact.off, compact.rank) if compact is not None else (cw["K"],) * 2
    TC = min(64, host_next_power_of_2(max(sw.most, 1)))
    for la, item, store in launches:
        idx = tensor(item)
        tiles, height = (la.tiles, la.tile_height) if idx is None else (idx, item.height)
        s_kind, out_temp, (out_full, *s_full) = la.kinds
        identity = la.form == "input_block" and s_kind == IDENTITY
        TJ = 1 if identity else min(64, host_next_power_of_2(prog.cw_most[la.number]))
        TN = min(height, CW.CLASS_WEIGHT_NODES, max(1, CW.CLASS_WEIGHT_LANES // (TC * TJ)))   # (powers of 2)
        TB = max(1, min(host_next_power_of_2(B), 128 // TC, CW.CLASS_WEIGHT_LANES // (TC * TJ * TN)))
        lanes = TN * TB * TC * TJ
        warps = 1 if lanes <= 128 else 2 if lanes <= 512 else 4
        pdl = pdl_launch(element_flows.device)
        for first, size in grid_chunks(host_cdiv(B, TB), 1):
            _set_weight_kernel[(tiles.numel() // NODE_TILE * (height // TN), size, host_cdiv(max(sw.most, 1), TC))](
                cw["set_buf"], cw["col"], cw["K"], sw.count, sw.first, sw.off, sw.lptr, sw.pair_i, sw.pair_q, element_flows, temp_flows, node_mars, prior["mars"], temp,
                prog.width_t, ivs.of, ivs.info, ivs.table, once_iv, count, *ctables, la.dests, tiles, la.ptr, la.parents, regions["offset"], regions["width"],
                regions["first"], prior["offset"], prior["width"], prior["reg_prior"], elem_first, elem_width, temp_width, tflow_width, B, n, W, first,
                BLOCK_INPUT = la.form == "block_input", S_KIND = s_kind, OUT_TEMP = out_temp, OUT_FULL = out_full, S_FULL = all(s_full), COMPACT = compact is not None,
                MARS = inside, STORE = store, TN = TN, SUBS = height // TN, TB = TB, TC = TC, TJ = TJ, PDL = pdl, launch_pdl = pdl, num_warps = warps)


def combine_set_weights(prog, cw: dict, col_K: torch.Tensor) -> None:
    """The classes' rows of ``cw["buf"]`` from the sets' (``cw["set_buf"]``), once a query's launches wrote those: every class row of every target column."""
    sw, rows = cw["sets"], cw["rows"]
    cs_ptr, cs_set, most = _class_sets_of(prog)
    N = sw.count.numel()
    TR = min(16, host_next_power_of_2(max(rows.most, 1)))
    _set_combine_kernel[(N, host_cdiv(max(rows.most, 1), TR))](cw["buf"], cw["set_buf"], rows.count, rows.cls, rows.off, sw.off, sw.grp, sw.rank, cs_ptr, cs_set, col_K,
                                                              prog.C, sw.S, most, TR = TR, TK = 64, num_warps = 4)
