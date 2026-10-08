"""
The constraint side of the lifted plan: which automaton states can occur at each position, and how they
move from one position to the next.

Boundary ``t`` (``t = 0, ..., n``) is the point just before the PC reads ``x_t``. A state is ACTIVE at
boundary ``t`` if it is reachable from the initial state in exactly ``t`` tokens and can still reach an
accepting state in exactly ``n - t`` tokens; every other state contributes nothing to any accepted
string of length ``n`` and gets no column. Columns at a boundary are its active states in increasing
state id.

The layout depends only on the automaton and ``n`` -- never on the PC, its parameters or evidence -- so it
is built once and cached by the automaton's fingerprint.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from ...language.dfa import DFA


@dataclass(frozen = True)
class BoundaryLayout:
    """
    Active automaton states per boundary, and the column-to-column transitions between boundaries.

    :ivar n: number of positions
    :ivar token_class: [V] the automaton's token classes
    :ivar width: [n+1] number of active states (columns) per boundary
    :ivar state_id: [n+1, W] automaton state of every column, ``-1`` for padding (``W`` = max width)
    :ivar col_of_state: [n+1, K] column of every automaton state, ``-1`` if inactive
    :ivar next_col: [n, W, C] column at boundary ``t+1`` reached from column ``k`` at ``t`` by a token of
        class ``c``; ``-1`` if that successor is inactive (pruned or dead) or ``k`` is padding
    :ivar satisfiable: whether any string of length ``n`` is accepted. If not, every boundary is empty.

    Boundary 0 holds only the initial state (column 0), and every active state at boundary ``n`` is
    accepting, so neither needs a separate table.
    """

    n: int
    token_class: torch.Tensor
    width: torch.Tensor
    state_id: torch.Tensor
    col_of_state: torch.Tensor
    next_col: torch.Tensor
    satisfiable: bool

    @property
    def max_width(self) -> int:
        return self.state_id.size(1)

    @property
    def num_classes(self) -> int:
        return self.next_col.size(2)


_CACHE: "OrderedDict[tuple, BoundaryLayout]" = OrderedDict()
_CACHE_SIZE = 64


def build_layout(automaton: DFA, n: int) -> BoundaryLayout:
    """
    The :class:`BoundaryLayout` of ``automaton`` over ``n`` positions (cached by fingerprint and ``n``).

    :param automaton: a complete DFA (what the ``automaton`` capability returns)
    :type automaton: DFA

    :param n: number of positions (the PC's number of variables)
    :type n: int
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
    layout = _build(automaton, int(n))
    _CACHE[key] = layout
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last = False)
    return layout


def _build(dfa: DFA, n: int) -> BoundaryLayout:
    K, C = dfa.num_states, dfa.num_classes
    delta = dfa.delta                                                       # [K, C]

    # reachable from the initial state in exactly t tokens
    reach = torch.zeros(n + 1, K, dtype = torch.bool)
    reach[0, dfa.initial] = True
    for t in range(n):
        reach[t + 1, delta[reach[t]].flatten()] = True
    # ... and able to reach acceptance in exactly n - t more tokens
    co_reach = torch.stack([dfa.accept_within(n - t) for t in range(n + 1)], dim = 0)
    active = reach & co_reach                                               # [n+1, K]

    width = active.sum(dim = 1)
    W = max(int(width.max()), 1)
    state_id = torch.full((n + 1, W), -1, dtype = torch.long)
    col_of_state = torch.full((n + 1, K), -1, dtype = torch.long)
    for t in range(n + 1):
        ids = torch.nonzero(active[t]).flatten()                            # increasing state id
        state_id[t, :ids.numel()] = ids
        col_of_state[t, ids] = torch.arange(ids.numel())

    if n > 0:
        src = state_id[:n]                                                  # [n, W]
        succ = delta[src.clamp(min = 0)]                                    # [n, W, C] successor states
        next_col = torch.gather(col_of_state[1:], 1, succ.reshape(n, W * C)).reshape(n, W, C)
        next_col[src < 0] = -1
    else:
        next_col = torch.full((0, W, C), -1, dtype = torch.long)

    return BoundaryLayout(n = n, token_class = dfa.token_class, width = width, state_id = state_id,
                          col_of_state = col_of_state, next_col = next_col,
                          satisfiable = bool(width[0] > 0))
