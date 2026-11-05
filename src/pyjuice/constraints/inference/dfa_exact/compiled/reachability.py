"""
Which (column, column) pairs an interval's blocks keep, and the tables the kernels read them through: the slots of a block, its positions and its
occupancy per interval (:class:`Reachability`), the intervals of a program at once (:class:`IntervalTables`).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch


if TYPE_CHECKING:
    from .layout import BoundaryLayout


class Reachability:
    """
    Which (column at boundary ``a``, column at boundary ``b``) pairs the automaton joins in exactly ``b - a``
    tokens -- the entries of a block over ``[a, b)`` that are not structurally ``-inf``, and the only ones a block
    keeps: its SLOTS, the joined pairs in row-major order. A sequence end (``b == n``) is one column, joined from
    every column that can still accept. Per interval, computed when first asked, by a forward sweep from its entry
    boundary as far as it asks: on the device its layout was built on (bitsets, :mod:`~..kernels.reachability`), on the
    host for one built there:

    * :meth:`slots`: how many pairs are joined. A prefix (``a == 0``) or a suffix (``b == n``) joins every pair --
      every column is reachable from the initial state and can still accept -- so neither is swept;
    * :meth:`positions`: every pair's slot, or None when every pair is joined (pair ``(i, j)`` in slot ``i * X +
      j``, ``X`` the exit width);
    * :meth:`occupancy`: which ``tile x tile`` tiles of columns hold a joined pair.

    :param layout: the constraint's layout
    :param tile: the side of an occupancy tile, in columns
    """

    #: the words of the sweeps kept to go on from (on the device), the last used kept
    SWEEP_WORDS = 1 << 28
    #: the pairs :meth:`prepare` takes in a device call at most (padding included)
    PREPARE_PAIRS = 1 << 28

    def __init__(self, layout: BoundaryLayout, tile: int = 16):
        self.n = layout.n
        self.width = [int(w) for w in layout.width.tolist()]
        self.tile = tile
        self._layout = layout
        self._dev = layout.on_device[0] if layout.on_device[0].type == "cuda" else None
        self._occupancy: Dict[Tuple[int, int], torch.Tensor] = {}
        self._slots: Dict[Tuple[int, int], int] = {}
        self._positions: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
        self._sweeps = None
        self._by_path: Dict[tuple, tuple] = {}             # the boundaries' groups an interval passes -> what it keeps
        self._bits: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()   # ... -> where a sweep along them is
        self._bits_words = 0
        # what :meth:`prepare` swept, by path (the groups an interval passes) and number: an interval's number at
        # ``a * (n + 1) + b`` (-1: not swept), per number its slots, columns, stack and place there and where its positions
        # and pairs are among its stack's flat ones (:meth:`_stack_tables`; -1: every pair joined); per stack its occupancy
        # and positions, and the path numbers it holds from
        self._path_no: Optional[np.ndarray] = None
        self._numbers: Dict[tuple, int] = {}
        self._path_slots: List[int] = []
        self._path_columns: List[Tuple[int, int]] = []
        self._path_place: List[Tuple[int, int]] = []
        self._path_flat: List[Tuple[int, int]] = []
        self._path_views: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
        self._stacks: List[Tuple[torch.Tensor, torch.Tensor, int, int]] = []

    def exit_width(self, b: int) -> int:
        return 1 if b == self.n else self.width[b]

    def _every_pair(self, a: int, b: int) -> bool:
        return a == 0 or b == self.n

    def prepare(self, intervals) -> None:
        """Keep the slots, positions and occupancy of ``intervals`` (``(a, b)`` pairs): those whose pairs are to be swept,
        all of them in a few device calls (one at a time, they take some twenty each, a wait for the device among them,
        and a circuit may ask thousands). On a layout on the host, nothing: each is swept when first asked."""
        if self._dev is None:
            return
        side = self.n + 1
        if self._path_no is None:
            self._path_no = np.full(side * side, -1, dtype = np.int32)
        last, numbers, no, todo = self.n, self._numbers, self._path_no, {}
        group = None
        for a, b in intervals:
            if a == 0 or b == last or no[a * side + b] >= 0:
                continue
            if group is None:
                group = self._tables().group
            path = tuple(group[a:b]) if b > a else ("identity", self.width[a])    # (:meth:`_path`, thousands of times)
            known = numbers.get(path)
            if known is not None:
                no[a * side + b] = known
            elif path in todo:
                todo[path][3].append((a, b))
            else:
                todo[path] = (self.width[a], self.width[b], None, [(a, b)])
        swept = self._sweep_together(todo)
        items = sorted(((path, (Wa, X, swept[path], ivs)) for path, (Wa, X, _, ivs) in todo.items()),
                       key = lambda item: item[1][0] * item[1][1])                   # (blocks of a size together)
        k = 0
        while k < len(items):                                # as many as a device call takes, padded to the largest
            end, entries, exits = k, 0, 0
            while end < len(items):
                wider, longer = max(entries, items[end][1][0]), max(exits, items[end][1][1])
                if end > k and (end - k + 1) * wider * longer > self.PREPARE_PAIRS:
                    break
                entries, exits, end = wider, longer, end + 1
            self._keep_stacked(items[k:end], entries, exits)
            k = end

    def _sweep_together(self, todo: dict) -> dict:
        """What each of the paths ``todo`` (:meth:`prepare`'s, by an interval of it) joins: ``(columns' bits of several
        sweeps, where its are, its words)``. A boundary at a time, every sweep there side by side: a launch a boundary,
        where one sweep at a time takes a launch a boundary and sweep. Every table of them stays as it is written: a path
        reads the one at its end, and a sweep starts from a part of one table of a column per bit."""
        from ..kernels.reachability import advance, column_bits, words_of    # (they import this module's)
        starts, ends, last = set(), {}, {}
        for path, (_, _, _, ivs) in todo.items():
            a, b = ivs[0]
            starts.add(a)
            ends.setdefault(b, []).append((a, path))
            last[a] = max(last.get(a, a), b)
        out = {}
        if not starts:
            return out
        tables = self._tables()
        every_column = column_bits(max(self.width[a] for a in starts), self._dev)
        cur, live = None, {}                                 # [columns here, words], a sweep's start -> (first word, words)
        for t in range(min(starts), max(last.values()) + 1):
            if t in starts:
                bits = every_column[:self.width[t], :words_of(self.width[t])]
                live[t] = (0 if cur is None else cur.size(1), bits.size(1))
                cur = bits.contiguous() if cur is None else torch.cat([cur, bits], dim = 1)     # (a kernel reads rows of its words)
            for a, path in ends.get(t, ()):                  # (the paths ending here: their words, read when stacked)
                out[path] = (cur, *live[a])
            done = [a for a in live if last[a] <= t]         # (what no path reads any more goes)
            if done:
                for a in done:
                    del live[a]
                at, kept = 0, []
                for a, (first, words) in live.items():
                    kept.append(np.arange(first, first + words))
                    live[a] = (at, words)
                    at += words
                cur = cur[:, torch.from_numpy(np.concatenate(kept)).to(self._dev)] if kept else None
            if cur is not None:
                cur = advance(cur, t, self.width[t + 1], tables)
        return out

    def _evict_bits(self) -> None:
        while self._bits_words > self.SWEEP_WORDS and len(self._bits) > 1:
            _, old = self._bits.popitem(last = False)
            self._bits_words -= old.numel()

    def occupancy_places(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(stack, place)`` int32: where the occupancy of every interval ``(a[k], b[k])`` is in :attr:`_stacks`, -1 for one
        not in a stack (:meth:`device_occupancy` keeps it): not swept yet, or every pair joined."""
        if not self._path_place:
            return np.full(a.size, -1, dtype = np.int32), np.full(a.size, -1, dtype = np.int32)
        no = self._path_no[a * (self.n + 1) + b]
        places = np.asarray(self._path_place, dtype = np.int32)
        return np.where(no >= 0, places[no, 0], -1), np.where(no >= 0, places[no, 1], -1)

    def slots_of(self, ends: np.ndarray) -> np.ndarray:
        """int64: :meth:`slots` of every interval of ``ends`` (int64 [intervals, 2]), at once where they are swept."""
        a, b = ends[:, 0], ends[:, 1]
        widths = np.asarray(self.width + [1], dtype = np.int64)
        out = widths[a] * np.where(b == self.n, 1, widths[b])
        every = (a == 0) | (b == self.n)
        if self._path_no is not None and self._path_slots:
            no = self._path_no[a * (self.n + 1) + b]
            found = (no >= 0) & ~every
            out[found] = np.asarray(self._path_slots, dtype = np.int64)[no[found]]
            every = every | found
        for k in np.flatnonzero(~every).tolist():            # (not swept yet)
            out[k] = self.slots(int(a[k]), int(b[k]))
        return out

    def interval_table(self, ends: np.ndarray) -> Optional[tuple]:
        """``(info, table parts)`` of :class:`IntervalTables`, if every interval of ``ends`` (int64 [intervals, 2]) was swept
        or joins every pair, else None; the tables of positions go (read once, and kept here no more)."""
        if self._path_no is None:
            return None
        a, b = ends[:, 0], ends[:, 1]
        every = (a == 0) | (b == self.n)
        no = self._path_no[a * (self.n + 1) + b]
        if ((no < 0) & ~every).any():
            return None
        info = np.full((a.size, 3), -1, dtype = np.int64)
        info[:, 0] = self.slots_of(ends)
        parts, size, first_flat = [], 0, []
        for stack in range(len(self._stacks)):                # a stack's positions, then its pairs
            flats, pairs = self._stack_tables(stack)
            first_flat.append((size, size + flats.numel()))
            parts += [flats, pairs]
            size += flats.numel() + pairs.numel()
        if self._path_flat:
            flats = np.asarray(self._path_flat, dtype = np.int64)
            stacks = np.asarray(self._path_place, dtype = np.int64)[:, 0]
            bases = np.asarray(first_flat, dtype = np.int64)[stacks]
            at = np.flatnonzero(~every & (flats[no, 0] >= 0))
            info[at, 1:] = bases[no[at]] + flats[no[at]]
        return info, parts

    def _stack_tables(self, stack: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """int32 ``(positions, pairs)``: of the paths of a stack that do not join every pair, flat, one after the other, in
        a few device calls: every pair's slot or -1, the pair ``i * X + j`` of every slot (``X`` the path's own)."""
        _, positions, first, end = self._stacks[stack]
        dev = self._dev
        columns = np.asarray(self._path_columns[first:end], dtype = np.int64).reshape(-1, 2)
        sizes = columns[:, 0] * columns[:, 1]
        some = torch.from_numpy(np.asarray(self._path_slots[first:end], dtype = np.int64) != sizes).to(dev)
        if not bool(some.any()):
            empty = torch.zeros(0, dtype = torch.int32, device = dev)
            return empty, empty
        entries, exits = positions.size(1), positions.size(2)
        rows = torch.arange(entries, device = dev)[None, :] < torch.from_numpy(columns[:, 0]).to(dev)[:, None]
        cols = torch.arange(exits, device = dev)[None, :] < torch.from_numpy(columns[:, 1]).to(dev)[:, None]
        flats = positions[rows[:, :, None] & cols[:, None, :] & some[:, None, None]]
        path_of, at = torch.nonzero(positions.flatten(1) >= 0, as_tuple = True)
        keep = some[path_of]
        path_of, at = path_of[keep], at[keep]
        X = torch.from_numpy(columns[:, 1]).to(dev)[path_of]
        return flats, ((at // max(exits, 1)) * X + at % max(exits, 1)).to(torch.int32)

    def _keep_stacked(self, items: list, entries: int, exits: int) -> None:
        """Keep what the paths ``items`` (``(path, (width at a, width at b, their bits, their intervals))``, the bits as
        :meth:`_sweep_together` has them) join, every block padded to ``entries`` x ``exits`` pairs that no column joins,
        which count for nothing in row-major order."""
        from ..kernels.reachability import pairs_of_bits, stack_bits             # (they import this module's)
        widths = np.asarray([(Wa, X) for _, (Wa, X, _, _) in items], dtype = np.int64)
        pairs = pairs_of_bits(stack_bits([bits for _, (_, _, bits, _) in items], exits, entries), entries)
        occupancy = _tile_any_stacked(pairs, self.tile, self.tile)
        flat = pairs.flatten(1)
        slots = flat.sum(dim = 1).tolist()                   # (the one wait)
        positions = torch.where(pairs, torch.cumsum(flat, dim = 1).view_as(pairs) - 1, -1).to(torch.int32)
        counts = np.asarray(slots, dtype = np.int64)
        sizes = widths[:, 0] * widths[:, 1]
        full = counts == sizes
        stack, first = len(self._stacks), len(self._path_slots)
        self._stacks.append((occupancy, positions, first, first + len(items)))
        at_position = np.cumsum(np.where(full, 0, sizes)) - np.where(full, 0, sizes)
        at_pair = np.cumsum(np.where(full, 0, counts)) - np.where(full, 0, counts)
        no, side = self._path_no, self.n + 1
        for i, (path, (Wa, X, _, ivs)) in enumerate(items):
            number = len(self._path_slots)
            self._numbers[path] = number
            self._path_slots.append(slots[i])
            self._path_columns.append((Wa, X))
            self._path_place.append((stack, i))
            self._path_flat.append((-1, -1) if full[i] else (at_position[i], at_pair[i]))
            for a, b in ivs:
                no[a * side + b] = number

    def _path_view(self, number: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``(occupancy, positions or None)`` of a swept path (its intervals share them)."""
        kept = self._path_views.get(number)
        if kept is None:
            stack, i = self._path_place[number]
            Wa, X = self._path_columns[number]
            occupancy, positions, _, _ = self._stacks[stack]
            tiles = lambda columns: (columns + self.tile - 1) // self.tile
            kept = self._path_views[number] = (occupancy[i, :tiles(Wa), :tiles(X)],
                                               None if self._path_slots[number] == Wa * X else positions[i, :Wa, :X])
        return kept

    def _swept(self, a: int, b: int) -> int:
        """An interval's number among the swept paths (found by its path if another of it was), or -1."""
        if self._path_no is None:
            return -1
        at = a * (self.n + 1) + b
        number = int(self._path_no[at])
        if number < 0:
            number = self._numbers.get(self._path(a, b), -1) if self._dev is not None else -1
            if number >= 0:
                self._path_no[at] = number
        return number

    def _compute(self, a: int, b: int):
        """Keep the interval's slots, positions and occupancy."""
        if self._dev is None:
            self._sweep(a)
            return
        # intervals through the same alike boundaries join the same pairs: what they keep is one, and tables of them
        # are too (past the shortest accepted string, most intervals of a length are alike)
        path = self._path(a, b)
        kept = self._by_path.get(path)
        if kept is None:
            pairs = self._device_pairs(a, b)
            self._keep(a, b, pairs)
            self._by_path[path] = (self._slots[a, b], self._occupancy[a, b], self._positions[a, b])
        else:
            self._slots[a, b], self._occupancy[a, b], self._positions[a, b] = kept

    def slots(self, a: int, b: int) -> int:
        """The joined pairs of the interval ``[a, b)``: the slots of its blocks."""
        if self._every_pair(a, b):
            return self.width[a] * self.exit_width(b)
        number = self._swept(a, b)
        if number >= 0:
            return self._path_slots[number]
        if (a, b) not in self._slots:
            self._compute(a, b)
        return self._slots[a, b]

    def keeps_every_pair(self, a: int, b: int) -> bool:
        """Whether the automaton joins every pair of the interval: :meth:`positions` is None."""
        return self._every_pair(a, b) or self.slots(a, b) == self.width[a] * self.exit_width(b)

    def positions(self, a: int, b: int) -> Optional[torch.Tensor]:
        """int32 [width[a], exit width at b] (host): the slot of every pair, -1 where not joined; None if every pair is
        joined."""
        out = self.device_positions(a, b)
        return None if out is None else out.cpu()

    def device_positions(self, a: int, b: int) -> Optional[torch.Tensor]:
        """:meth:`positions`, where they are kept."""
        if self._every_pair(a, b):
            return None
        number = self._swept(a, b)
        if number >= 0:
            return self._path_view(number)[1]
        if (a, b) not in self._positions:
            self._compute(a, b)
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

    def _keep(self, a: int, b: int, pairs: torch.Tensor):
        self._occupancy[a, b] = _tile_any(pairs, self.tile, self.tile)
        count = int(pairs.sum())
        self._slots[a, b] = count
        self._positions[a, b] = None if count == pairs.numel() else \
            torch.where(pairs, torch.cumsum(pairs.flatten(), 0).view_as(pairs) - 1, -1).to(torch.int32)

    def _sweep(self, a: int):
        """Every interval starting at ``a``: pairs (a, b) for b = a, ..., n, kept as occupancy, slots and
        positions (on the host)."""
        cur = torch.eye(self.width[a])
        for b in range(a, self.n + 1):
            if b > a:
                cur = self._advance(cur, b - 1)
            self._keep(a, b, (cur.sum(dim = 1, keepdim = True) if b == self.n else cur) > 0)

    def _tables(self):
        from ..kernels.reachability import sweep_tables
        if self._sweeps is None:
            self._sweeps = sweep_tables(self._layout.transitions(self._dev))
        return self._sweeps

    def _path(self, a: int, b: int) -> tuple:
        """What the pairs of ``[a, b)`` depend on: the boundaries' groups it passes (a group fixes a boundary's states and
        the next's), or for no boundary, its width."""
        return tuple(self._tables().group[a:b]) if b > a else ("identity", self.width[a])

    def _device_pairs(self, a: int, b: int) -> torch.Tensor:
        """bool [width[a], exit width at b] on the device: a prefix's and a suffix's every pair, else by a sweep along
        the interval's boundaries (:meth:`_device_bits`)."""
        from ..kernels.reachability import pairs_of_bits
        Wa, X = self.width[a], self.exit_width(b)
        if self._every_pair(a, b):
            return torch.ones(Wa, X, dtype = torch.bool, device = self._dev)
        return pairs_of_bits(self._device_bits(a, b), Wa)

    def _device_bits(self, a: int, b: int) -> torch.Tensor:
        """int32 [width[b], words] on the device: the columns of ``a`` that reach every column of ``b``, by a sweep along
        the interval's boundaries, going on from the longest part of them swept already."""
        from ..kernels.reachability import advance, column_bits
        Wa = self.width[a]
        tables = self._tables()
        t, bits = b, None
        while t > a:
            bits = self._bits.get(tuple(tables.group[a:t]))
            if bits is not None:
                break
            t -= 1
        if bits is None:
            t, bits = a, column_bits(Wa, self._dev)
        for t in range(t, b):
            bits = advance(bits, t, self.width[t + 1], tables)
            key = tuple(tables.group[a:t + 1])
            if key not in self._bits:
                self._bits[key] = bits
                self._bits_words += bits.numel()
            self._bits.move_to_end(key)
        self._evict_bits()
        return bits

    def slot_pairs(self, positions: torch.Tensor) -> torch.Tensor:
        """int32 [slots]: the pair ``i * X + j`` of every slot of a table of :meth:`device_positions`."""
        return torch.nonzero(positions.flatten() >= 0).flatten().to(torch.int32)

    def flat_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """A table of :meth:`device_positions`, flat."""
        return positions.flatten()

    def pairs(self, a: int, b: int) -> torch.Tensor:
        """bool [width[a], exit width at b] (host; computed afresh, not cached)."""
        if self._dev is not None:
            return self._device_pairs(a, b).cpu()
        cur = torch.eye(self.width[a])
        for t in range(a, b):
            cur = self._advance(cur, t)
        return (cur.sum(dim = 1, keepdim = True) if b == self.n else cur) > 0

    def occupancy(self, a: int, b: int) -> torch.Tensor:
        """bool [ceil(width[a] / tile), ceil(exit width at b / tile)] (host)."""
        return self.device_occupancy(a, b).cpu()

    def device_occupancy(self, a: int, b: int) -> torch.Tensor:
        """:meth:`occupancy`, where it is kept."""
        number = -1 if self._every_pair(a, b) else self._swept(a, b)
        if number >= 0:
            return self._path_view(number)[0]
        if (a, b) not in self._occupancy:
            if self._dev is None or not self._every_pair(a, b):
                self._compute(a, b)
            else:                                            # (every pair: one matrix per shape)
                shape = ("every pair", self.width[a], self.exit_width(b))
                full = self._by_path.get(shape)
                if full is None:
                    full = self._by_path[shape] = _tile_any(self._device_pairs(a, b), self.tile, self.tile)
                self._occupancy[a, b] = full
        return self._occupancy[a, b]


def _tile_any(m: torch.Tensor, tr: int, tc: int) -> torch.Tensor:
    """[ceil(R / tr), ceil(C / tc)]: whether each tr x tc tile of the bool matrix ``m`` holds a True."""
    if tr == 1 and tc == 1:
        return m
    R, C = m.shape
    I, J = math.ceil(R / tr), math.ceil(C / tc)
    p = torch.zeros(I * tr, J * tc, dtype = torch.bool, device = m.device)
    p[:R, :C] = m
    return p.view(I, tr, J, tc).any(dim = 3).any(dim = 1)


def _tile_any_stacked(m: torch.Tensor, tr: int, tc: int) -> torch.Tensor:
    """:func:`_tile_any` of every matrix of ``m`` [matrices, R, C]."""
    if tr == 1 and tc == 1:
        return m
    P, R, C = m.shape
    I, J = math.ceil(R / tr), math.ceil(C / tc)
    p = torch.zeros(P, I * tr, J * tc, dtype = torch.bool, device = m.device)
    p[:, :R, :C] = m
    return p.view(P, I, tr, J, tc).any(dim = 4).any(dim = 2)


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


def interval_tables(reach: Reachability, intervals, device, of: Optional[torch.Tensor] = None,
                    ends: Optional[np.ndarray] = None) -> IntervalTables:
    """The :class:`IntervalTables` of ``intervals`` (``(a, b)`` boundary pairs), on ``device``; ``of``: their numbers if
    the caller has them (as :attr:`IntervalTables.of`, on the device), ``ends``: themselves as int64 [intervals, 2]."""
    n = reach.n
    if of is None:
        of = torch.full(((n + 1) * (n + 1),), -1, dtype = torch.int32)
        for k, (a, b) in enumerate(intervals):
            of[a * (n + 1) + b] = k
    ends = np.asarray(intervals, dtype = np.int64).reshape(-1, 2) if ends is None else ends
    swept = reach.interval_table(ends)
    if swept is not None:                                       # (read off the stacks, all of them at once)
        info, parts = swept
        info = torch.from_numpy(info)
    else:
        info, parts, size, seen = [], [], 0, {}
        for a, b in intervals:
            positions = reach.device_positions(a, b)
            if positions is None:
                info.append((reach.slots(a, b), -1, -1))
                continue
            at = seen.get(id(positions))                        # (alike intervals keep one table of positions)
            if at is None:
                pairs = reach.slot_pairs(positions)
                at = seen[id(positions)] = (pairs.numel(), size, size + positions.numel())
                parts += [reach.flat_positions(positions), pairs]
                size += positions.numel() + pairs.numel()
            info.append(at)
        info = torch.tensor(info, dtype = torch.int64).reshape(-1, 3)
    table = torch.cat([t.to(device) for t in parts]) if parts else torch.zeros(1, dtype = torch.int32)
    return IntervalTables(of.to(device), info.to(device), table.to(device))
