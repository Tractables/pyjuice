"""
The constraint's side of the lifted values: which automaton states can occur at each boundary of the PC's positions,
and how they move from one position to the next (:func:`build_layout`), with the tables the kernels read them through
(:class:`Reachability`, :class:`IntervalTables`).

Boundary ``t`` (``t = 0, ..., n``) is the point just before the PC reads ``x_t``. A state is ACTIVE at
boundary ``t`` if it is reachable from the initial state in exactly ``t`` tokens and can still reach an
accepting state in exactly ``n - t`` tokens; every other state contributes nothing to any accepted
string of length ``n`` and gets no column. Columns at a boundary are its active states, farthest from
acceptance first (the automaton's shortest number of tokens to an accepting state), ties by state id: any order
is valid, and this one gathers the (column, column) pairs the automaton cannot join into whole tiles, which the
product kernels skip (:class:`Reachability`).

The layout depends only on the automaton and ``n`` -- never on the PC, its parameters or evidence -- so it
is built once and cached by the automaton's fingerprint.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional, Tuple

import torch

from ....language.dfa import DFA



@dataclass(frozen = True)
class BoundaryLayout:
    """
    Active automaton states per boundary, and the column-to-column transitions between boundaries.

    :ivar n: number of positions
    :ivar token_class: [V] the automaton's token classes
    :ivar width: [n+1] number of active states (columns) per boundary
    :ivar state_id: [n+1, W] automaton state of every column, ``-1`` for padding (``W`` = max width); a
        boundary's columns run from the state farthest from acceptance to the nearest, ties by state id
    :ivar col_of_state: [n+1, K] column of every automaton state, ``-1`` if inactive
    :ivar delta: [K, C] the automaton's transitions: the successor state of every (state, class)
    :ivar satisfiable: whether any string of length ``n`` is accepted. If not, every boundary is empty.

    Boundary 0 holds only the initial state (column 0), and every active state at boundary ``n`` is
    accepting, so neither needs a separate table. The column-to-column transitions between boundaries are the
    automaton's through the two boundaries' column maps -- :meth:`next_col_at` gives one boundary's table; the table
    over every boundary, ``[n, W, C]``, is never kept (2.4e9 entries for a JSON schema's automaton over 1024
    positions).
    """

    n: int
    token_class: torch.Tensor
    width: torch.Tensor
    state_id: torch.Tensor
    col_of_state: torch.Tensor
    delta: torch.Tensor
    satisfiable: bool

    @property
    def max_width(self) -> int:
        return self.state_id.size(1)

    @cached_property
    def reachability(self) -> "Reachability":
        """The pairs of columns the automaton joins, per interval (:class:`Reachability`, built on first use and kept
        with the layout)."""
        return Reachability(self)

    @property
    def num_classes(self) -> int:
        return self.delta.size(1)

    @property
    def num_states(self) -> int:
        return self.delta.size(0)

    def next_col_at(self, t: int) -> torch.Tensor:
        """[width[t], C]: the column at boundary ``t + 1`` reached from every column at ``t`` by a token of each class,
        ``-1`` if that successor is inactive (pruned or dead) -- :func:`next_columns` of this layout's tables."""
        return next_columns(self.delta, self.state_id, self.col_of_state, t, int(self.width[t]))

    def next_col_table(self) -> torch.Tensor:
        """[n, W, C]: every boundary's :meth:`next_col_at`, ``-1`` past its columns -- for inspection of small
        automata (the backend never builds it)."""
        out = torch.full((self.n, self.max_width, self.num_classes), -1, dtype = torch.long)
        for t in range(self.n):
            out[t, :int(self.width[t])] = self.next_col_at(t)
        return out

    def transitions(self, device) -> "Transitions":
        """The layout's transitions on ``device`` (:class:`Transitions`)."""
        return Transitions(self.delta.to(device), self.state_id.to(device), self.col_of_state.to(device),
                           [int(w) for w in self.width.tolist()])

    @cached_property
    def successors(self):
        """``(ptr, states)``: every automaton state's distinct successor states, as a CSR (int64, host): those of
        state ``s`` are ``states[ptr[s] .. ptr[s + 1]]``, ascending."""
        K = self.delta.size(0)
        keys = torch.unique(torch.arange(K)[:, None] * K + self.delta)             # (state, successor), ascending
        ptr = torch.zeros(K + 1, dtype = torch.long)
        ptr[1:] = torch.cumsum(torch.bincount(keys // K, minlength = K), dim = 0)
        return ptr, keys % K


def next_columns(delta: torch.Tensor, state_id: torch.Tensor, col_of_state: torch.Tensor, t: int,
                 w: int) -> torch.Tensor:
    """[w, C]: the column at boundary ``t + 1`` reached from each of the ``w`` columns at ``t`` by a token of each
    class -- ``col_of_state[t + 1][delta[state_id[t, k], c]]`` --, ``-1`` where that successor is inactive there
    (pruned or dead). The tables on one device, of one integer type."""
    return col_of_state[t + 1][delta[state_id[t, :w]]]


@dataclass(frozen = True)
class Transitions:
    """
    A layout's transitions on one device (:meth:`BoundaryLayout.transitions`): :meth:`at` is a boundary's table
    (:meth:`BoundaryLayout.next_col_at`) -- what the program's transition tables are built from, one boundary at a
    time.

    :ivar delta, state_id, col_of_state: the layout's, on the device
    :ivar width: the boundaries' columns (Python ints)
    """

    delta: torch.Tensor
    state_id: torch.Tensor
    col_of_state: torch.Tensor
    width: list

    @property
    def n(self) -> int:
        return len(self.width) - 1

    @property
    def K(self) -> int:
        return self.delta.size(0)

    @property
    def C(self) -> int:
        return self.delta.size(1)

    @property
    def W(self) -> int:
        return self.state_id.size(1)

    @property
    def device(self):
        return self.delta.device

    def at(self, t: int) -> torch.Tensor:
        """[width[t], C]: boundary ``t``'s table (see :func:`next_columns`)."""
        return next_columns(self.delta, self.state_id, self.col_of_state, t, self.width[t])


_CACHE: "OrderedDict[tuple, BoundaryLayout]" = OrderedDict()
_CACHE_SIZE = 64


def build_layout(automaton: DFA, n: int, device = None, report = None) -> BoundaryLayout:
    """
    The :class:`BoundaryLayout` of ``automaton`` over ``n`` positions (cached by fingerprint and ``n``; on the host).

    :param automaton: a complete DFA (what the ``automaton`` capability returns)
    :type automaton: DFA

    :param n: number of positions (the PC's number of variables)
    :type n: int

    :param device: where the sweeps over the automaton run (None: the host) -- the same layout either way
    :param report: None, or the compilation's progress (:class:`~...compiler.Progress`), told of the sweeps
    """
    if not isinstance(automaton, DFA):
        raise TypeError(f"Expected a DFA, got {type(automaton).__name__}.")
    if n < 0:
        raise ValueError(f"`n` must be non-negative, got {n}.")
    key = (automaton.fingerprint(), int(n))
    layout = _CACHE.get(key)
    if layout is not None:
        _CACHE.move_to_end(key)
        return layout
    layout = _build(automaton, int(n), device, report)
    _CACHE[key] = layout
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last = False)
    return layout


def _sweep(first: torch.Tensor, step, n: int, report, stage: str) -> torch.Tensor:
    """bool [n + 1, K]: ``first``, then ``step`` of the row before, ``n`` times -- to a FIXED POINT: a row equal to
    the one before stays (the step is a function of the row), so the rest are copies. Decoding at a length well past
    the shortest accepted string, the sweeps settle after a few dozen boundaries of a thousand."""
    out = torch.empty(n + 1, first.numel(), dtype = torch.bool, device = first.device)
    out[0] = first
    if report is not None:
        report.stage(stage, n)
    for t in range(n):
        out[t + 1] = step(out[t])
        if torch.equal(out[t + 1], out[t]):
            out[t + 2:] = out[t + 1]
            if report is not None:
                report.advance(n - t)
            break
        if report is not None:
            report.advance()
    return out


def _build(dfa: DFA, n: int, device = None, report = None) -> BoundaryLayout:
    K, C = dfa.num_states, dfa.num_classes
    dev = torch.device("cpu") if device is None else torch.device(device)
    _, delta, accept = dfa._tables_on(dev)                                  # [K, C], [K]
    delta = delta.long()

    def forward(row):                                                       # the successors of a set of states
        nxt = torch.zeros(K, dtype = torch.bool, device = dev)
        nxt[delta[row].reshape(-1)] = True
        return nxt

    initial = torch.zeros(K, dtype = torch.bool, device = dev)
    initial[dfa.initial] = True
    # reachable from the initial state in exactly t tokens ...
    reach = _sweep(initial, forward, n, report, "layout: reachable states")
    # ... and able to reach acceptance in exactly n - t more tokens
    accept_in = _sweep(accept.bool(), lambda row: row[delta].any(dim = 1), n, report, "layout: accepting states")
    active = (reach & accept_in.flip(0)).cpu()                              # [n+1, K]
    # the column order: the fewest tokens to acceptance (within n), descending, then the state id
    to_accept = torch.where(accept_in.any(dim = 0), accept_in.to(torch.uint8).argmax(dim = 0), n + 1).cpu()

    width = active.sum(dim = 1)
    W = max(int(width.max()), 1)
    state_id = torch.full((n + 1, W), -1, dtype = torch.long)
    col_of_state = torch.full((n + 1, K), -1, dtype = torch.long)
    if report is not None:
        report.stage("layout: columns", n + 1)
    for t in range(n + 1):
        if t > 0 and torch.equal(active[t], active[t - 1]):                 # (the same states: the same columns)
            state_id[t], col_of_state[t] = state_id[t - 1], col_of_state[t - 1]
        else:
            ids = torch.nonzero(active[t]).flatten()
            ids = ids[torch.argsort((n + 1 - to_accept[ids]) * K + ids)]
            state_id[t, :ids.numel()] = ids
            col_of_state[t, ids] = torch.arange(ids.numel())
        if report is not None:
            report.advance()

    return BoundaryLayout(n = n, token_class = dfa.token_class, width = width, state_id = state_id,
                          col_of_state = col_of_state, delta = dfa.delta.to(torch.long).contiguous(),
                          satisfiable = bool(width[0] > 0))


class Reachability:
    """
    Which (column at boundary ``a``, column at boundary ``b``) pairs the automaton joins in exactly ``b - a``
    tokens -- the entries of a block over ``[a, b)`` that are not structurally ``-inf``, and the only ones a block
    keeps: its SLOTS, the joined pairs in row-major order. A sequence end (``b == n``) is one column, joined from
    every column that can still accept. Per interval, computed on demand by one forward sweep per entry boundary
    (the pair matrices dropped as the sweep moves on):

    * :meth:`slots`: how many pairs are joined. A prefix (``a == 0``) or a suffix (``b == n``) joins every pair --
      every column is reachable from the initial state and can still accept -- so neither is swept;
    * :meth:`positions`: every pair's slot, or None when every pair is joined (pair ``(i, j)`` in slot ``i * X +
      j``, ``X`` the exit width);
    * :meth:`occupancy`: which ``tile x tile`` tiles of columns hold a joined pair.

    :param layout: the constraint's layout
    :param tile: the side of an occupancy tile, in columns
    """

    def __init__(self, layout: BoundaryLayout, tile: int = 16):
        self.n = layout.n
        self.width = [int(w) for w in layout.width.tolist()]
        self.tile = tile
        self._layout = layout
        self._occupancy: Dict[Tuple[int, int], torch.Tensor] = {}
        self._slots: Dict[Tuple[int, int], int] = {}
        self._positions: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}

    def exit_width(self, b: int) -> int:
        return 1 if b == self.n else self.width[b]

    def _every_pair(self, a: int, b: int) -> bool:
        return a == 0 or b == self.n

    def slots(self, a: int, b: int) -> int:
        """The joined pairs of the interval ``[a, b)``: the slots of its blocks."""
        if self._every_pair(a, b):
            return self.width[a] * self.exit_width(b)
        if (a, b) not in self._slots:
            self._sweep(a)
        return self._slots[a, b]

    def positions(self, a: int, b: int) -> Optional[torch.Tensor]:
        """int32 [width[a], exit width at b]: the slot of every pair, -1 where not joined; None if every pair is
        joined."""
        if self._every_pair(a, b):
            return None
        if (a, b) not in self._positions:
            self._sweep(a)
        return self._positions[a, b]

    def _advance(self, cur: torch.Tensor, t: int) -> torch.Tensor:
        """[R, width[t]] 0/1 -> [R, width[t + 1]]: the columns one token on (a scatter through every column's state's
        distinct successors, :attr:`BoundaryLayout.successors`, those active at ``t + 1``)."""
        layout = self._layout
        ids = layout.state_id[t, :self.width[t]]
        ptr, states = layout.successors
        counts = ptr[ids + 1] - ptr[ids]
        k = torch.repeat_interleave(torch.arange(ids.numel()), counts)
        at = ptr[ids][k] + torch.arange(k.numel()) - (torch.cumsum(counts, 0) - counts)[k]
        cols = layout.col_of_state[t + 1][states[at]]
        keep = cols >= 0
        nxt = torch.zeros(cur.size(0), self.width[t + 1])
        nxt.index_add_(1, cols[keep], cur[:, k[keep]])
        return (nxt > 0).to(cur.dtype)

    def _sweep(self, a: int):
        """Every interval starting at ``a``: pairs (a, b) for b = a, ..., n, kept as occupancy, slots and
        positions."""
        cur = torch.eye(self.width[a])
        for b in range(a, self.n + 1):
            if b > a:
                cur = self._advance(cur, b - 1)
            pairs = (cur.sum(dim = 1, keepdim = True) if b == self.n else cur) > 0
            self._occupancy[a, b] = _tile_any(pairs, self.tile, self.tile)
            count = int(pairs.sum())
            self._slots[a, b] = count
            self._positions[a, b] = None if count == pairs.numel() else \
                torch.where(pairs, torch.cumsum(pairs.flatten(), 0).view_as(pairs) - 1, -1).to(torch.int32)

    def pairs(self, a: int, b: int) -> torch.Tensor:
        """bool [width[a], exit width at b]: the joined pairs (computed afresh, not cached)."""
        cur = torch.eye(self.width[a])
        for t in range(a, b):
            cur = self._advance(cur, t)
        return (cur.sum(dim = 1, keepdim = True) if b == self.n else cur) > 0

    def occupancy(self, a: int, b: int) -> torch.Tensor:
        """bool [ceil(width[a] / tile), ceil(exit width at b / tile)]."""
        if (a, b) not in self._occupancy:
            self._sweep(a)
        return self._occupancy[a, b]


def _tile_any(m: torch.Tensor, tr: int, tc: int) -> torch.Tensor:
    """[ceil(R / tr), ceil(C / tc)]: whether each tr x tc tile of the bool matrix ``m`` holds a True."""
    R, C = m.shape
    I, J = math.ceil(R / tr), math.ceil(C / tc)
    p = torch.zeros(I * tr, J * tc, dtype = torch.bool)
    p[:R, :C] = m
    return p.view(I, tr, J, tc).any(dim = 3).any(dim = 1)


@dataclass(frozen = True)
class IntervalTables:
    """
    Where the product kernels find the slots of the blocks over every interval they read or write (see
    :class:`Reachability`).

    :ivar of: int32 [(n + 1) * (n + 1)], the number of interval ``[a, b)`` at ``a * (n + 1) + b`` (-1 if unused)
    :ivar info: int64 [intervals, 3], per interval: its slots per sample, and where its positions and its pairs
        start in ``table`` -- both -1 when every pair is joined (pair ``(i, j)`` is then in slot ``i * X + j``)
    :ivar table: int32, the positions (``[entry columns, exit columns]``, every pair's slot or -1) and the pairs
        (``[slots]``, ``i * X + j`` of every slot's pair) of the intervals where not every pair is joined
    """

    of: torch.Tensor
    info: torch.Tensor
    table: torch.Tensor


def interval_tables(reach: Reachability, intervals, device) -> IntervalTables:
    """The :class:`IntervalTables` of ``intervals`` (``(a, b)`` boundary pairs), on ``device``."""
    n = reach.n
    of = torch.full(((n + 1) * (n + 1),), -1, dtype = torch.int32)
    info, parts, size = [], [], 0
    for k, (a, b) in enumerate(intervals):
        of[a * (n + 1) + b] = k
        positions = reach.positions(a, b)
        if positions is None:
            info.append((reach.slots(a, b), -1, -1))
        else:
            pairs = torch.nonzero(positions.flatten() >= 0).flatten().to(torch.int32)
            info.append((pairs.numel(), size, size + positions.numel()))
            parts += [positions.flatten(), pairs]
            size += positions.numel() + pairs.numel()
    table = torch.cat(parts) if parts else torch.zeros(1, dtype = torch.int32)
    return IntervalTables(of.to(device), torch.tensor(info, dtype = torch.int64).reshape(-1, 3).to(device),
                          table.to(device))
