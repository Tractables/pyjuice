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

import numpy as np
import torch
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv, host_next_power_of_2

from ..common import IDENTITY, NODE_TILE, _block_ptr, _exit_width, _interval, _logaddexp_tile, _tile_rows, available_memory, grid_chunks, pdl_launch, tensor
from ..prod.launches import _lse_step_one_exp, _lse_value
from ..prod_top_down.launches import _once, _out_flows_ptr, _parent_base
from ..transitions import tables as TT
from ..transitions.tables import BoundaryGroups, _ranges, _sort_key, boundary_groups
from .class_weight_kernels import CLASS_WEIGHT_CUDA, CLASS_WEIGHT_CUDA_CLASSES, CLASS_WEIGHT_CUDA_SMEM, _cw_block_input_term, _cw_input_block_term


#: Whether a query's class weights go through its live pairs' class sets where :func:`set_weights_apply` says so; "force": wherever they can (tests)
SET_WEIGHTS = True
#: ... on an automaton whose classes' lists hold at least ``SET_WEIGHT_MIN_RATIO`` times its pairs, and at least ``SET_WEIGHT_MIN_ENTRIES`` entries a boundary
SET_WEIGHT_MIN_RATIO = 2.0
SET_WEIGHT_MIN_ENTRIES = 1 << 16
#: The most pairs a query's lists may be built from whatever the device holds (8 bytes each kept) -- past it, as many as a share of the memory this process can still
#: take there holds (``SET_WEIGHT_MEMORY_SHARE``, as the transition masses' budget: a batch of 32 with a prefix observed on K 371,046 has 140 M) --, and the most
#: (list group, set) cells of their rank table: past either, the classes' own walks; the lists are built a chunk of list groups at a time, of at most
#: ``SET_WEIGHT_CHUNK`` pairs (or one group's), some 90 bytes a pair while a chunk is built; and the fewest pairs worth the lists: below, the classes' walks are short too (a prefix's next token on K 286,841:
#: 28 pairs, 6.9 ms by the classes' walks, 9.3 through the sets). MEASURED GPT-2's vocabulary, HMM-64, 64 positions, batch 2, settled conditionals with a prefix, scattered
#: tokens, one target or nothing observed, on the six automata of the corpus this rule takes: from 2^12 to 2^14 pairs on, the sets faster on 17 of 17 queries,
#: 40.4 -> 15.1 s in all; below, slower on 6 of 6
SET_WEIGHT_PAIRS_MOST = 1 << 27
SET_WEIGHT_MEMORY_SHARE = 0.125
SET_WEIGHT_CHUNK = 1 << 24
SET_WEIGHT_RANKS_MOST = 1 << 27
SET_WEIGHT_PAIRS_LEAST = 1 << 14
#: The most cells of what a query's rows are found in when they come from the sets (:func:`set_weight_rows`): (target column, class), (target column, entry of a class's
#: sets) and (set, class); past it, the rows from the classes' own lists
SET_WEIGHT_ROW_CELLS = 1 << 26


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


def no_lists_by_class(prog) -> bool:
    """Whether the lists by class of ``prog``'s tables would have more entries than their int32 pointers reach
    (:data:`~..transitions.tables.TABLE_ENTRIES_MOST`: :meth:`~..transitions.tables.BoundaryGroups.class_lists` raises): no kernel can walk a class's list then, so
    the class sets are the one way to the class weights, for every query, however small. Counted once (it asks the device), kept with the program."""
    kept = getattr(prog, "_no_lists_by_class", None)
    if kept is None:
        kept = prog._no_lists_by_class = int(boundary_groups(prog._tr).class_columns.sum()) > TT.TABLE_ENTRIES_MOST
    return kept


def set_weights_apply(prog, liveness, compact) -> bool:
    """Whether a query of ``prog`` takes its class weights through the class sets: never while a CUDA graph is captured (the lists are the query's own), nor on tables made
    on the host (a small automaton); always where the lists by class cannot be made (:func:`no_lists_by_class`); else not a query the CUDA kernels take
    (:func:`~.class_weight_kernels._class_weight_cuda`: compact values under pruning, few classes, and boundaries whose rows fit a CTA's shared memory: left as they
    are; past that width they decline, and the walk is Triton's), and by what the automaton's lists hold (:data:`SET_WEIGHT_MIN_RATIO`,
    :data:`SET_WEIGHT_MIN_ENTRIES`). Host values alone, but for that count, made once.

    MEASURED GPT-2's vocabulary, HMM-64, 64 positions, batch 2, every automaton of the corpus on the device's tables forced through the sets, a prefix, scattered tokens, one
    target or nothing observed: of the queries this takes, 41 of 41 faster, 49.5 -> 20.2 s in all, and no faster query left out; with the width at 8,192, where the CUDA kernels
    still fit, 6 of 47 slower (K 11,845 with nothing observed: 65 -> 156 ms)."""
    if not SET_WEIGHTS or torch.cuda.is_current_stream_capturing() or not isinstance(boundary_groups(prog._tr), BoundaryGroups):
        return False
    if SET_WEIGHTS == "force" or no_lists_by_class(prog):
        return True
    if CLASS_WEIGHT_CUDA and liveness is not None and compact is not None and prog.C <= CLASS_WEIGHT_CUDA_CLASSES and 8 * prog.W <= CLASS_WEIGHT_CUDA_SMEM:
        return False                                                 # (a destination's rows at both boundaries, 4 bytes a column: what a CTA stages)
    entries, pairs = prog.transition_stats[0], int(prog.trans_pairs.sum())
    return entries >= SET_WEIGHT_MIN_RATIO * pairs and entries >= SET_WEIGHT_MIN_ENTRIES * prog.n


def set_weights_always(prog) -> bool:
    """Whether every query of ``prog`` large enough takes the class sets, whatever its storage and liveness (:func:`set_weights_apply`): its rows then come from the sets
    too (:func:`set_weight_rows`), and the lists by class are not read."""
    return set_weights_apply(prog, True, True)


def most_live_columns(prog, liveness, positions) -> list:
    """Per target position (``positions``, bool [n] on the host), the most live columns of any sample there, every column without a liveness: what the host's count of
    :func:`set_weight_lists` takes. Read off the liveness' lists: it waits for those alone."""
    targets = np.nonzero(positions)[0].tolist()
    if liveness is None:
        return [prog.width[t] for t in targets]
    walk = liveness.lists()[0].view(liveness.B, prog.n + 1).amax(0).tolist()
    return [walk[t] for t in targets]


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


def _row_tables(prog) -> tuple:
    """``(of_entry, sets, member, weights)``: the classes of the (class, set) entries, by class, and their sets (as :func:`_class_sets_of` has them); which classes a set
    has, bool [S, C]; what a hash of a collection of sets adds up of them, int64 [S]. Kept with the program."""
    kept = getattr(prog, "_class_set_row_tables", None)
    if kept is None:
        _, member, in_set, _, in_each = boundary_groups(prog._tr).class_sets
        ptr, sets, _ = _class_sets_of(prog)
        S, C, dev = in_each.numel(), prog.C, prog.dev
        of_entry = torch.repeat_interleave(torch.arange(C, device = dev), (ptr[1:] - ptr[:-1]).long())
        table = torch.zeros(S, C, dtype = torch.bool, device = dev)
        table[in_set, member.long()] = True
        weights = torch.from_numpy(np.random.default_rng(7).integers(1, 1 << 62, S) | 1).to(dev)
        kept = prog._class_set_row_tables = (of_entry, sets[:of_entry.numel()].long(), table, weights)
    return kept


def _collection_keys(live: torch.Tensor, sets_of: torch.Tensor, kind: torch.Tensor, of_entry: torch.Tensor, sets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """int64 [N, C], below 2^61: a hash of every class's sets with rows at every target column (``live`` bool [N, entries], ``sets_of`` how many), and of its kind. It
    only proposes which classes share a row: :func:`set_weight_rows` checks every class against the first of its key."""
    key = torch.zeros_like(sets_of).index_add_(1, of_entry, torch.where(live, weights.index_select(0, sets)[None, :], 0))
    return ((key + (sets_of * 2 + kind[None, :].long()) * -7046029254386353131) >> 2) & ((1 << 61) - 1)


def set_weight_rows(prog, sw: SetWeights, liveness, col_pos: torch.Tensor, col_K: torch.Tensor, col_first: torch.Tensor, group_first: torch.Tensor, B: int,
                    stored: Optional[torch.Tensor] = None):
    """
    :func:`~.class_rows.class_weight_rows` of a query that has its lists (``sw``), without the classes' own lists: the same :class:`~.class_rows.ClassWeightRows`, field
    for field. A class is live at a target column where a set it is in has a row there (and, under a session's per-sample classes, the sample allows it: ``liveness``,
    read for nothing else). The pairs partition by set, so two classes have the same live pairs exactly where the same sets with rows have them: they share a row then, if
    of a kind (both stored, or both enumerated: ``stored``, and :data:`~.class_rows.CLASS_ROWS_SHARED`), that of the first of them in the boundary's class order. A hash of
    a class's sets with rows proposes, in that order; every class is then checked against the first of its hash, set by set. None where the tables would be large
    (:data:`SET_WEIGHT_ROW_CELLS`): the classes' lists then.

    One difference, at the sequence end: every accepting successor is one column there, so the classes' lists give one row to classes that reach different accepting
    states from the same columns; their pairs are in different sets, and here each keeps a row of its own. The same weights in more rows, at that position alone.
    """
    from .class_rows import CLASS_ROWS_SHARED, ClassWeightRows
    n, C, S, dev = prog.n, prog.C, sw.S, prog.dev
    N = sw.count.numel()
    if S * C > SET_WEIGHT_ROW_CELLS:
        return None
    of_entry, sets, member, weights = _row_tables(prog)
    if N * max(C, of_entry.numel(), S) > SET_WEIGHT_ROW_CELLS:
        return None
    live = (sw.rank.index_select(0, sw.grp) >= 0).index_select(1, sets)     # [N, entries]: the entry's set has a row at the column
    sets_of = torch.zeros(N, C, dtype = torch.int64, device = dev).index_add_(1, of_entry, live.long())
    t = col_pos.repeat_interleave(B)
    on = sets_of > 0
    if liveness is not None and liveness.allowed is not None:                # (a session's per-sample classes: one set, or a sample's own)
        allowed = liveness.allowed.view(-1, n, C)
        b = torch.arange(N, device = dev) % B if allowed.size(0) > 1 else torch.zeros(N, dtype = torch.long, device = dev)
        on &= allowed[b, t] != 0
    order = prog.succ.cls_order.view(n, C).index_select(0, t).long()         # [N, C]: the boundary's classes in its order
    every = torch.arange(C, device = dev)[None, :]
    rep = None
    if CLASS_ROWS_SHARED and stored is not None:
        kind = stored.long() >= 0
        key = torch.where(on, _collection_keys(live, sets_of, kind, of_entry, sets, weights), 1 << 61)    # (a dead class: after every live one)
        keys, by = torch.sort(key.gather(1, order), dim = 1, stable = True)  # a key's classes, in class order
        classes = order.gather(1, by)
        first = torch.ones_like(keys, dtype = torch.bool)
        first[:, 1:] = keys[:, 1:] != keys[:, :-1]
        start = torch.cummax(torch.where(first, every, 0), dim = 1).values
        cand = torch.empty_like(classes).scatter_(1, classes, classes.gather(1, start))    # every class's candidate: the first of its key
        # exactly: as many sets with rows, of a kind, and every such set of the class has the candidate too
        missing = live & ~member.view(-1).index_select(0, (sets[None, :] * C + cand.index_select(1, of_entry)).view(-1)).view(N, -1)
        differs = torch.zeros(N, C, dtype = torch.int64, device = dev).index_add_(1, of_entry, missing.long()) > 0
        differs |= (sets_of != sets_of.gather(1, cand)) | (kind[None, :] != kind[cand])
        rep = torch.where(on, torch.where(differs, every, cand), -1)
        own = on & (rep == every)
    else:
        own = on
    own_o = own.gather(1, order)                                             # (in class order)
    place = torch.cumsum(own_o, dim = 1) - 1
    count = own_o.sum(dim = 1)
    off = torch.zeros(N + 1, dtype = torch.int64, device = dev)
    off[1:] = torch.cumsum(count * col_K, dim = 0)
    local = (off[:N] - off.index_select(0, col_first)) // col_K              # a column's first row among its row group's
    m, r = torch.nonzero(own_o, as_tuple = True)                             # every row: its column, its class's place in the order
    cls = torch.zeros(N, C, dtype = torch.int32, device = dev)
    cls[m, place[m, r]] = order[m, r].to(torch.int32)
    row = torch.full((N, C), -1, dtype = torch.int32, device = dev)
    row[m, order[m, r]] = (local[m] + place[m, r]).to(torch.int32)
    if rep is not None:                                                      # (a class's row is its representative's)
        row = torch.where(on, row.gather(1, rep.clamp(min = 0)), -1)
        rep = rep.to(torch.int32).contiguous()
    return ClassWeightRows(count = count.to(torch.int32), cls = cls, off = off, row = row.contiguous(), _host = None, _done = None, _groups = group_first.numel() - 1,
                           rep = rep, _read = (off.index_select(0, group_first).tolist(), int(count.max())))


def set_weight_lists(prog, liveness, col_pos: torch.Tensor, col_K: torch.Tensor, B: int, most_columns: Optional[list] = None) -> Optional[SetWeights]:
    """
    The :class:`SetWeights` of a query's target columns (``col_pos`` int64 [N / B], every block of ``B`` columns' position; ``col_K`` int64 [N], the floats of a row),
    from its :class:`~..live.Liveness` (None: every column live), or None where they are too many (:data:`SET_WEIGHT_PAIRS_MOST` and the memory left,
    :data:`SET_WEIGHT_RANKS_MOST`) or
    there are none, or too few to be worth the lists (:data:`SET_WEIGHT_PAIRS_LEAST`; none too few where there are no lists by class to fall back to:
    :func:`no_lists_by_class`): first by the host's count, where it has one -- ``most_columns``, per target
    position the most live columns of any sample there, each of at most the tables' most pairs --, so that a query of few live columns asks nothing of the device. A
    group's live columns, their states' pairs (:attr:`~..transitions.tables.BoundaryGroups.pairs`), those whose successor is a live column of the next boundary; by
    group, set, then column: a stable sort, a chunk of groups at a time (:data:`SET_WEIGHT_CHUNK`). It waits for the device, for how many pairs there are.
    """
    least = 0 if no_lists_by_class(prog) else SET_WEIGHT_PAIRS_LEAST
    if most_columns is not None and B * sum(most_columns) * max(prog.succ.max_pairs, 1) < least:
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
    if total == 0 or total < least:
        return None
    if total > SET_WEIGHT_PAIRS_MOST and (dev.type != "cuda" or 8 * total > SET_WEIGHT_MEMORY_SHARE * available_memory(dev)):
        return None
    # a chunk of list groups at a time, of at most SET_WEIGHT_CHUNK pairs (or one group's): what is kept of a chunk is its lists, 8 bytes a live pair
    pairs_to = torch.cumsum(torch.zeros(G, dtype = torch.int64, device = dev).index_add_(0, gl, pc), dim = 0).tolist()
    columns_to = torch.cumsum(cnt, dim = 0).tolist()
    every_set = torch.arange(S, device = dev)
    pair_i, pair_q, lens, rows_of, ranks = [], [], [], [], []
    g0 = 0
    while g0 < G:
        before = pairs_to[g0 - 1] if g0 else 0
        g1 = g0 + 1
        while g1 < G and pairs_to[g1] - before <= SET_WEIGHT_CHUNK:
            g1 += 1
        c0, c1, here, Gc = (columns_to[g0 - 1] if g0 else 0), columns_to[g1 - 1], pairs_to[g1 - 1] - before, g1 - g0
        of_l = torch.repeat_interleave(torch.arange(c1 - c0, device = dev), pc[c0:c1])
        p = _ranges(p_start.index_select(0, state[c0:c1]), pc[c0:c1], here)
        t_next = t_l[c0:c1].index_select(0, of_l) + 1
        q = prog.col_of_state.reshape(-1).index_select(0, t_next * prog.K + p_succ.index_select(0, p)).long()
        q = torch.where((t_next == n) & (q >= 0), torch.zeros_like(q), q)      # (a sequence end: its one column)
        g = gl[c0:c1].index_select(0, of_l)
        keep = q >= 0
        if not shared:
            keep &= liveness.live.reshape(-1).index_select(0, ((g % B) * (n + 1) + t_next) * W + q.clamp(min = 0)) != 0
        g, q, i, x = g[keep] - g0, q[keep], icol[c0:c1].index_select(0, of_l)[keep], of_pair.index_select(0, p[keep]).long()
        del of_l, p, keep, t_next
        counts = torch.bincount(g * S + x, minlength = Gc * S).view(Gc, S)
        order = torch.argsort(-counts, dim = 1, stable = True)       # a group's sets, the longest list first
        live_sets = (counts > 0).sum(dim = 1)
        rank = torch.empty_like(order).scatter_(1, order, every_set.expand(Gc, S))
        by = torch.argsort(_sort_key(g * S + rank.reshape(-1).index_select(0, g * S + x), Gc * S), stable = True)
        pair_i.append(i.index_select(0, by).to(torch.int32))
        pair_q.append(q.index_select(0, by).to(torch.int32))
        lens.append(torch.gather(counts, 1, order)[every_set[None, :] < live_sets[:, None]])
        rows_of.append(live_sets)
        ranks.append(torch.where(counts > 0, rank, -1).to(torch.int32))
        del g, q, i, x, counts, order, rank, by
        g0 = g1
    del state, pc, gl, t_l, icol
    pair_i, pair_q, lens, rows_of = torch.cat(pair_i), torch.cat(pair_q), torch.cat(lens), torch.cat(rows_of)
    if pair_i.numel() == 0:
        return None
    lptr = torch.zeros(lens.numel() + 1, dtype = torch.int64, device = dev)
    lptr[1:] = torch.cumsum(lens, dim = 0)
    grp = torch.arange(N, device = dev) // B if shared else torch.arange(N, device = dev)
    count = rows_of.index_select(0, grp)
    off = torch.zeros(N + 1, dtype = torch.int64, device = dev)
    off[1:] = torch.cumsum(count * col_K, dim = 0)
    return SetWeights(count = count.to(torch.int32).contiguous(), first = (torch.cumsum(rows_of, dim = 0) - rows_of).index_select(0, grp).contiguous(), off = off, lptr = lptr,
                      pair_i = pair_i.contiguous(), pair_q = pair_q.contiguous(), grp = grp.contiguous(), rank = torch.cat(ranks).contiguous(), most = int(rows_of.max()),
                      size = int(off[-1]), S = S, pairs = total)


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
