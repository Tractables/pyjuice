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
is built once and kept with the automaton object (hashing the automaton's tables would take longer than building it).
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import torch
from numba import njit

from ....language.dfa import DFA
from .reachability import Reachability


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
    :ivar delta: [K, C] the automaton's transitions: the successor state of every (state, class), on the host, made if it is to come
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
    table: object                                     # the automaton's: a tensor, or one to come (it holds the automaton's moves, not the automaton)
    satisfiable: bool
    on_device: tuple                                  # (device, delta, state_id, col_of_state, order), as they were built

    @cached_property
    def delta(self) -> torch.Tensor:
        return self.table if isinstance(self.table, torch.Tensor) else self.table.dense()

    @cached_property
    def state_id(self) -> torch.Tensor:
        return self.on_device[2].cpu()

    @cached_property
    def col_of_state(self) -> torch.Tensor:
        return self.on_device[3].cpu()

    @property
    def max_width(self) -> int:
        return self.on_device[2].size(1)

    @cached_property
    def reachability(self) -> "Reachability":
        """The pairs of columns the automaton joins, per interval (:class:`Reachability`, built on first use and kept
        with the layout)."""
        return Reachability(weakref.proxy(self))     # (it keeps it: no cycle to wait for a collection)

    @property
    def num_classes(self) -> int:
        return self.on_device[1].size(1)

    @property
    def num_states(self) -> int:
        return self.on_device[1].size(0)

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
        device = torch.device(device)
        kept = self.__dict__.setdefault("_transitions", {})
        tr = kept.get(device)
        if tr is None:
            widths = [int(w) for w in self.width.tolist()]
            dev, delta, state_id, col_of_state, order = self.on_device
            host = None                                       # (built on the host: at hand)
            if "state_id" in self.__dict__ and self.num_states * self.num_classes <= LAYOUT_HOST_ENTRIES:
                host = (self.delta.numpy(), self.state_id.numpy(), self.col_of_state.numpy())
            if dev == device:
                tr = Transitions(delta, state_id, col_of_state, widths, order, host)
            else:
                tr = Transitions(delta.to(device), state_id.to(device), col_of_state.to(device), widths, order.to(device), host)
            kept[device] = tr
        return tr

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
    :ivar order: int64 [K] the states in the order of the columns: every boundary's columns are those of its states in this order
    :ivar host: ``(delta, state_id, col_of_state)`` as arrays, for a small automaton (its tables are then built on the host,
        :mod:`~..kernels.transitions.host`), else None
    """

    delta: torch.Tensor
    state_id: torch.Tensor
    col_of_state: torch.Tensor
    width: list
    order: torch.Tensor
    host: Optional[tuple] = None

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


_CACHE: "OrderedDict[tuple, Tuple[weakref.ref, BoundaryLayout]]" = OrderedDict()
_CACHE_SIZE = 64


def build_layout(automaton: DFA, n: int, device = None, report = None) -> BoundaryLayout:
    """
    The :class:`BoundaryLayout` of ``automaton`` over ``n`` positions (kept for the automaton object and ``n``; on the
    host).

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
    key = (id(automaton), int(n))
    kept = _CACHE.get(key)
    if kept is not None and kept[0]() is automaton:                       # (an id is only unique among live objects)
        _CACHE.move_to_end(key)
        return kept[1]
    layout = _build(automaton, int(n), device, report)

    def forget(ref):                                                      # (a dead automaton's layout: freed with it)
        kept = _CACHE.get(key)
        if kept is not None and kept[0] is ref:
            del _CACHE[key]

    _CACHE[key] = (weakref.ref(automaton, forget), layout)
    _CACHE.move_to_end(key)
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last = False)
    return layout


@njit(cache = True)
def _state_edges(delta, accept):
    """The automaton's transitions but those into a state that never accepts and never leaves itself: most of its [K, C]
    table, and nothing can be accepted through them. ``(ptr, succ)``: every state's distinct successors, CSR."""
    K, C = delta.shape
    dead = np.empty(K, dtype = np.bool_)
    for s in range(K):
        loops = not accept[s]
        if loops:
            for c in range(C):
                if delta[s, c] != s:
                    loops = False
                    break
        dead[s] = loops
    ptr = np.zeros(K + 1, dtype = np.int64)
    stamp = np.full(K, -1, dtype = np.int64)
    succ = np.empty(4 * K + 16, dtype = np.int64)
    num = 0
    for s in range(K):
        for c in range(C):
            t = delta[s, c]
            if stamp[t] != s and not dead[t]:
                stamp[t] = s
                if num == succ.shape[0]:
                    more = np.empty(2 * num, dtype = np.int64)
                    more[:num] = succ[:num]
                    succ = more
                succ[num] = t
                num += 1
        ptr[s + 1] = num
    return ptr, succ[:num]


@njit(cache = True)
def _reachable(ptr, succ, initial, K, n):
    """bool [n + 1, K]: the states reachable from ``initial`` in exactly ``t`` tokens. A row equal to the one before stays
    (a step is a function of the row), so the rest are copies: decoding at a length well past the shortest accepted
    string, a sweep settles after a few dozen boundaries of a thousand."""
    reach = np.zeros((n + 1, K), dtype = np.bool_)
    reach[0, initial] = True
    for t in range(n):
        for s in range(K):
            if reach[t, s]:
                for k in range(ptr[s], ptr[s + 1]):
                    reach[t + 1, succ[k]] = True
        same = True
        for s in range(K):
            if reach[t + 1, s] != reach[t, s]:
                same = False
                break
        if same:
            reach[t + 2:] = reach[t + 1]
            break
    return reach


@njit(cache = True)
def _accepting(ptr, succ, accept, n):
    """``(accept_in, to_accept)``: bool [n + 1, K], the states from which acceptance is reachable in exactly ``t`` tokens
    (a row equal to the one before stays), and every state's fewest tokens to acceptance (within ``n``, else ``n + 1``)."""
    K = accept.shape[0]
    accept_in = np.zeros((n + 1, K), dtype = np.bool_)
    to_accept = np.full(K, n + 1, dtype = np.int64)
    for s in range(K):
        if accept[s]:
            accept_in[0, s] = True
            to_accept[s] = 0
    for t in range(n):
        same = True
        for s in range(K):
            found = False
            for k in range(ptr[s], ptr[s + 1]):
                if accept_in[t, succ[k]]:
                    found = True
                    break
            accept_in[t + 1, s] = found
            if found != accept_in[t, s]:
                same = False
            if found and to_accept[s] > t + 1:
                to_accept[s] = t + 1
        if same:
            accept_in[t + 2:] = accept_in[t + 1]
            break
    return accept_in, to_accept


@njit(cache = True)
def _layout_columns(reach, accept_in, to_accept):
    """``(width, state_id, col_of_state)`` (:class:`BoundaryLayout`, int64), from the sweeps: :func:`_reachable`,
    :func:`_accepting`."""
    rows, K = reach.shape
    n = rows - 1
    active = np.empty((rows, K), dtype = np.bool_)
    width = np.zeros(rows, dtype = np.int64)
    W = 1
    for t in range(rows):
        count = 0
        for s in range(K):
            on = reach[t, s] and accept_in[n - t, s]
            active[t, s] = on
            if on:
                count += 1
        width[t] = count
        W = max(W, count)
    # the column order: the fewest tokens to acceptance (within n), descending, then the state id
    count_of = np.zeros(n + 3, dtype = np.int64)
    for s in range(K):
        count_of[to_accept[s]] += 1
    place = np.empty(n + 3, dtype = np.int64)
    at = 0
    for v in range(n + 2, -1, -1):
        place[v] = at
        at += count_of[v]
    order = np.empty(K, dtype = np.int64)
    for s in range(K):
        order[place[to_accept[s]]] = s
        place[to_accept[s]] += 1
    state_id = np.full((rows, W), -1, dtype = np.int64)
    col_of_state = np.full((rows, K), -1, dtype = np.int64)
    for t in range(rows):
        col = 0
        for i in range(K):
            s = order[i]
            if active[t, s]:
                state_id[t, col] = s
                col_of_state[t, s] = col
                col += 1
    return width, state_id, col_of_state, order


#: an automaton of at most so many table entries has its layout built on the host (a few array operations a boundary on
#: the device are milliseconds, the same sweeps in compiled loops microseconds); larger, on the device, where they scale
LAYOUT_HOST_ENTRIES = 1 << 19


def _build(dfa: DFA, n: int, device = None, report = None) -> BoundaryLayout:
    dev = torch.device("cpu") if device is None else torch.device(device)
    if dev.type != "cpu" and dfa.num_states * dfa.num_classes > LAYOUT_HOST_ENTRIES:
        return _build_on_device(dfa, n, dev, report)
    delta = dfa._tables_on(dev)[1].long()
    accept = dfa._accept.numpy()
    ptr, succ = _state_edges(dfa._delta.numpy(), accept)
    if report is not None:
        report.stage("layout: reachable states", 1)
    reach = _reachable(ptr, succ, dfa.initial, accept.size, n)
    if report is not None:
        report.advance()
        report.stage("layout: accepting states", 1)
    accept_in, to_accept = _accepting(ptr, succ, accept, n)
    if report is not None:
        report.advance()
        report.stage("layout: columns", 1)
    width, state_id, col_of_state, order = _layout_columns(reach, accept_in, to_accept)
    host = torch.from_numpy(state_id), torch.from_numpy(col_of_state)
    # (the automaton's tables are not copied: nothing writes them)
    layout = BoundaryLayout(n = n, token_class = dfa._token_class, width = torch.from_numpy(width),
                            table = dfa._table, satisfiable = bool(width[0] > 0),
                            on_device = (dev, delta, host[0].to(dev), host[1].to(dev), torch.from_numpy(order).to(dev)))
    layout.__dict__.update(state_id = host[0], col_of_state = host[1])        # (the host's, as they were built)
    if report is not None:
        report.advance()
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


def _build_on_device(dfa: DFA, n: int, dev: torch.device, report = None) -> BoundaryLayout:
    """:func:`_build`, on the device."""
    K = dfa.num_states
    _, delta, accept = dfa._tables_on(dev)                                  # [K, C], [K]
    accept = accept.bool()
    # the sweeps walk the pairs of states the automaton joins, each once (a state's classes lead to some ten states), but those into a state that never
    # accepts and never leaves itself: most of its [K, C] table, and nothing can be accepted through them. At every boundary: it is most of a long layout's cost
    if isinstance(dfa._table, torch.Tensor):
        dead = (delta == torch.arange(K, device = dev)[:, None]).all(dim = 1) & ~accept
        state, cls = torch.nonzero(~dead[delta], as_tuple = True)
        keys = torch.unique(state * K + delta[state, cls])
        state, succ = torch.div(keys, K, rounding_mode = "floor"), torch.remainder(keys, K)
    else:
        state, succ = dfa._table.pairs_on(dev)

    def forward(row):                                                       # the successors of a set of states
        nxt = torch.zeros(K, dtype = torch.bool, device = dev)
        nxt[succ[row[state]]] = True
        return nxt

    def backward(row):                                                      # the states with a successor in a set
        nxt = torch.zeros(K, dtype = torch.bool, device = dev)
        nxt[state[row[succ]]] = True
        return nxt

    initial = torch.zeros(K, dtype = torch.bool, device = dev)
    initial[dfa.initial] = True
    # reachable from the initial state in exactly t tokens ...
    reach = _sweep(initial, forward, n, report, "layout: reachable states")
    # ... and able to reach acceptance in exactly n - t more tokens
    accept_in = _sweep(accept, backward, n, report, "layout: accepting states")
    active = reach & accept_in.flip(0)                                      # [n+1, K]
    # the column order: the fewest tokens to acceptance (within n), descending, then the state id
    to_accept = torch.where(accept_in.any(dim = 0), accept_in.to(torch.uint8).argmax(dim = 0), n + 1)
    order = torch.argsort((n + 1 - to_accept) * K + torch.arange(K, device = dev))     # the states in that order

    if report is not None:
        report.stage("layout: columns", 1)
    width = active.sum(dim = 1)
    W = max(int(width.max()), 1)
    in_order = active[:, order]
    cols = torch.cumsum(in_order, dim = 1) - 1                              # a state's column, were it active
    rows, at = torch.nonzero(in_order, as_tuple = True)
    state_id = torch.full((n + 1, W), -1, dtype = torch.long, device = dev)
    state_id[rows, cols[rows, at]] = order[at]
    col_of_state = torch.full((n + 1, K), -1, dtype = torch.long, device = dev)
    col_of_state[rows, order[at]] = cols[rows, at]
    # (the automaton's tables are not copied: nothing writes them)
    layout = BoundaryLayout(n = n, token_class = dfa._token_class, width = width.cpu(), table = dfa._table,
                            satisfiable = bool(width[0] > 0), on_device = (dev, delta, state_id, col_of_state, order))
    if report is not None:
        report.advance()
    return layout
