"""
The storage of a state's buffers, and the pool holding the inside values kept once under a memory budget (:class:`PriorPool`).
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, Optional

import numpy as np
import torch

from .compiled.tables import ALIGN, _align
from .kernels.common import available_memory


#: What a storage holds when it is allocated (:meth:`State.storage_view`): None, nothing in particular -- every slot a
#: query reads is written first --, or a value it is filled with (the tests fill NaN, so that a read of a slot nothing
#: wrote shows).
FRESH_FILL: Optional[float] = None


def fresh_storage(numel: int, device) -> torch.Tensor:
    """``numel`` floats of new storage on ``device``, as :data:`FRESH_FILL` says."""
    out = torch.empty(max(int(numel), 1), device = device)
    if FRESH_FILL is not None:
        out.fill_(FRESH_FILL)
    return out


#: Of the device's free memory when the values kept once are first stored, the share they may take when the layout
#: of every region kept once does not fit that memory (then :class:`PriorPool` segments them; when it fits, they take
#: it all, as they did without a pool); a decoder session's ``memory_budget`` sets the bytes instead.
PRIOR_MEMORY_SHARE = 0.6


def _bytes(n: int) -> str:
    """``n`` bytes, in the unit that reads best."""
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} bytes"


class PriorPool:
    """
    The storage of the inside values kept once (``prior_mars``, see :mod:`~.queries.backend.plans`), under a memory
    budget. Every sum region kept once has an EXTENT of the pool while its values are held: a query gets extents for
    the values it computes (:meth:`allocate`), and when the pool is full, the values of an interval no query needs
    now are EVICTED -- their validity withdrawn (:meth:`State.prior_validity`), so that a later query reading them
    computes them again from the values still held below them. Which interval: the one read longest ago, among those
    that are no CHECKPOINT first -- the intervals whose depth (the longest chain of product steps to the inputs,
    ``Program.iv_depth``) is a multiple of the ``stride``, the smallest for which the checkpoints and a segment (the
    values of ``stride`` consecutive depths) fit the pool together: no value is then recomputed from more than
    ``stride`` steps below (left to right, a segment is computed once more as decoding enters it: about one forward
    pass more a session). Holes are merged as extents are freed, and the live extents moved to the front when none
    fits.

    A query whose values do not fit beside the checkpoints computes them in PIECES (:func:`~.queries.backend.forward.
    kept_values`), the lowest depths first (:meth:`piece`). Decoding that reads every value at every call (every
    position a target) would evict and recompute endlessly under a smaller pool: the pool refuses a query whose
    values do not fit it at all.

    The budget is the full layout when that fits the device's free memory (else :data:`PRIOR_MEMORY_SHARE` of it):
    then every region kept once has its extent, nothing is ever evicted, and the values are as without a pool.

    :param state: the :class:`State`
    :param budget: bytes, or None: by :data:`PRIOR_MEMORY_SHARE`
    """

    def __init__(self, state: "State", budget: Optional[int] = None):
        self.state = weakref.proxy(state)             # (it owns the pool: no cycle to wait for a collection)
        self.budget = budget
        self.buffer: Optional[torch.Tensor] = None
        self.extents: Dict[int, tuple] = {}           # region -> (offset, floats)
        self.free: list = []                          # (offset, floats) of the holes, by offset
        self.stamp: Dict[int, int] = {}               # interval -> the query that last read or wrote its values
        self.queries = 0
        self.stride = 1
        self.evictions = 0
        self.reg_floats = None                        # a region's extent (floats, aligned)
        self.iv_floats = None                         # an interval's regions' extents together
        self.iv_regions = None                        # an interval's regions
        self._version = 0                             # counts changes of the extents
        self._table = None                            # (version, offsets host, offsets device)

    def reset(self):
        """Every value dropped (their validity withdrawn by the caller); the buffer kept."""
        self.extents, self.stamp = {}, {}
        self.free = [(0, self.buffer.numel())] if self.buffer is not None else []
        self._version += 1

    # ---------------------------------------------------------------------------------------------

    def _open(self):
        """The buffer, allocated on first use within the budget, and the checkpoint stride."""
        if self.buffer is not None:
            return
        prog, layout = self.state._backend.program(), self.state.batch_layout(1)
        full = layout["node_size"]
        self.reg_floats = (prog.reg_rows * layout["sum_widths_host"] + ALIGN - 1) // ALIGN * ALIGN
        K = len(prog.sum_ivs)
        self.iv_floats = np.zeros(K, dtype = np.int64)
        np.add.at(self.iv_floats, prog.reg_iv_np, self.reg_floats)
        self.iv_regions = prog.iv_regions
        dev = self.state._device
        if self.budget is not None:
            budget = self.budget
        else:                                          # the memory this process can take: the device's free memory
            if dev.type == "cuda":                     # and the blocks its caching allocator holds unused
                free = available_memory(dev)
            else:
                free = full * 4
            budget = full * 4 if full * 4 <= free else int(PRIOR_MEMORY_SHARE * free)
        floats = min(int(_align(max(budget // 4, 1))), int(_align(full)))
        self.buffer = fresh_storage(floats, dev)
        self.state._storage["prior_mars"] = self.buffer
        self.free = [(0, floats)]
        self.stride = 1
        if floats < full:                              # segmented: the stride by what the pool holds
            self.stride = self._stride(prog.iv_depth, floats)
        self._version += 1

    def _stride(self, depth: np.ndarray, floats: int) -> int:
        """
        The smallest stride whose checkpoints, a segment (any ``stride`` consecutive depths) and two more values fit
        ``floats`` together: a session walking the chain recomputes every value once at most. When none does (fewer
        than about 2 sqrt(D) values fit), the stride of the least estimated recomputation: every segment entered
        costs its values, and a segment the room beside the checkpoints does not hold costs about (S - room)^2 / 2
        room more as the walk steps through it -- against no checkpoints at all, D^2 / 2 P (the walk recomputes a
        pool's worth of the chain every P steps); D + 1 then (the leaves the checkpoints).
        """
        D = int(depth.max(initial = 0))
        level = np.zeros(D + 1, dtype = np.int64)       # the values' floats per depth
        np.add.at(level, depth, self.iv_floats)
        value = max(1, int(self.iv_floats.max()))
        best, best_cost = D + 1, D * D / (2 * max(1, floats / value))
        for S in range(1, D + 1):
            padded = np.concatenate([[0], np.cumsum(np.concatenate([level, np.zeros(S, dtype = np.int64)]))])
            segment = int((padded[S:] - padded[:-S]).max())
            room = floats - int(level[::S].sum()) - 2 * value
            if room >= segment:
                return S
            if room > 0:
                r = room / value
                cost = (D / S) * (S + max(0.0, S - r) ** 2 / (2 * r))
                if cost < best_cost:
                    best, best_cost = S, cost
        return best

    def _fit(self, floats: int) -> Optional[int]:
        """The offset of the first hole of ``floats`` or more, taken (first fit), or None."""
        for k, (off, size) in enumerate(self.free):
            if size >= floats:
                if size == floats:
                    del self.free[k]
                else:
                    self.free[k] = (off + floats, size - floats)
                return off
        return None

    def _release(self, off: int, floats: int):
        """A hole, merged with its neighbours."""
        self.free.append((off, floats))
        self.free.sort()
        merged = []
        for o, s in self.free:
            if merged and merged[-1][0] + merged[-1][1] == o:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((o, s))
        self.free = merged

    def _compact(self):
        """The live extents moved to the front, in order, one hole after them."""
        at = 0
        for r, (off, size) in sorted(self.extents.items(), key = lambda kv: kv[1][0]):
            if off != at:
                self.buffer[at:at + size] = self.buffer[off:off + size].clone()
                self.extents[r] = (at, size)
            at += size
        self.free = [(at, self.buffer.numel() - at)] if at < self.buffer.numel() else []
        self._version += 1

    def _take(self, floats: int) -> Optional[int]:
        """An extent of ``floats``: a hole, or the live extents compacted first when the holes hold it together."""
        off = self._fit(floats)
        if off is None and sum(size for _, size in self.free) >= floats:
            self._compact()
            off = self._fit(floats)
        return off

    def _evict(self, iv: int):
        """Interval ``iv``'s values dropped: its regions' extents freed, its validity withdrawn."""
        for r in self.iv_regions[iv]:
            if int(r) in self.extents:
                self._release(*self.extents.pop(int(r)))
        self.state._prior_valid[iv] = False
        self.stamp.pop(iv, None)
        self.evictions += 1
        self._version += 1

    def _held(self) -> np.ndarray:
        """bool [sum intervals]: those with extents."""
        out = np.zeros(len(self.iv_floats), dtype = bool)
        if self.extents:
            out[self.state._backend.program().reg_iv_np[list(self.extents)]] = True
        return out

    def _checkpoints(self) -> np.ndarray:
        return self.state._backend.program().iv_depth % self.stride == 0

    def _too_small(self, floats: int) -> RuntimeError:
        return RuntimeError(f"The values kept once that this query reads take {_bytes(floats * 4)}, more than the "
                            f"{_bytes(self.buffer.numel() * 4)} kept for them (the decoder's `memory_budget`, or "
                            f"{PRIOR_MEMORY_SHARE:.0%} of the free device memory when they were first stored, their "
                            f"total not fitting it). A query that reads them all (every position a target) needs them "
                            f"all held.")

    # ---------------------------------------------------------------------------------------------

    def holds(self, ivs: np.ndarray, checkpoints: bool = True) -> bool:
        """Whether the values of the intervals ``ivs`` (bool [sum intervals]) fit the pool -- beside the checkpoints
        it holds, unless ``checkpoints`` is False (then they may be evicted)."""
        self._open()
        keep = ivs | (self._held() & self._checkpoints()) if checkpoints else ivs
        return int(self.iv_floats[keep].sum()) <= self.buffer.numel()

    def piece(self, plan) -> Optional[np.ndarray]:
        """bool [sum intervals]: of the values ``plan`` computes, the ones to compute first when they do not all fit
        beside the ones it reads: the intervals of the lowest depths such that they, the held values they read, the
        held values the plan reads (kept through every piece: computed again otherwise) and the checkpoints fit the
        pool together -- without the checkpoints when nothing fits beside them; None when nothing fits at all."""
        self._open()
        prog, valid = self.state._backend.program(), self.state._prior_valid.numpy()
        todo = plan.todo.numpy() & ~valid
        keep, depth = plan.prior_read & valid, prog.iv_depth
        for checkpoints in (True, False):
            best = None
            for d in np.unique(depth[todo]):
                piece = todo & (depth <= d)
                below = np.zeros_like(piece)
                for k in np.nonzero(piece)[0]:
                    below[prog.iv_children[k]] = True
                if not self.holds(piece | (below & valid) | keep, checkpoints):
                    break
                best = piece
            if best is not None:
                return best
        return None

    def allocate(self, plan) -> Dict[str, Any]:
        """Extents for the values ``plan`` computes (:attr:`~.queries.backend.plans.QueryPlan.todo`), the ones it
        reads (``prior_read``) kept meanwhile and the others evicted as needed; then :meth:`table`."""
        self._open()
        prog, valid = self.state._backend.program(), self.state._prior_valid
        self.queries += 1
        pinned = plan.prior_read
        for iv in np.nonzero(pinned)[0]:
            self.stamp[int(iv)] = self.queries
        todo = plan.todo.numpy()
        checkpoint = self._checkpoints()
        new = [r for r in np.nonzero(todo[prog.reg_iv_np])[0].tolist() if r not in self.extents]
        if new and self.free:                          # (all of them in the first hole, as first fit puts them, at once)
            sizes = self.reg_floats[new]
            ends = np.cumsum(sizes)
            off, room = self.free[0]
            if int(ends[-1]) <= room:
                self.extents.update(zip(new, zip((off + ends - sizes).tolist(), sizes.tolist())))
                if int(ends[-1]) == room:
                    del self.free[0]
                else:
                    self.free[0] = (off + int(ends[-1]), room - int(ends[-1]))
                self._version += 1
                new = []
        for r in new:
            floats = int(self.reg_floats[r])
            off = self._take(floats)
            while off is None:
                held = [iv for iv in np.nonzero(self._held() & ~pinned)[0]]
                if not held:
                    raise self._too_small(int(self.iv_floats[pinned].sum()))
                held.sort(key = lambda iv: (bool(checkpoint[iv]), self.stamp.get(int(iv), -1)))
                self._evict(int(held[0]))
                off = self._take(floats)
            self.extents[r] = (off, floats)
            self._version += 1
        return self.table()

    def table(self) -> Dict[str, Any]:
        """Where the values kept once are: ``mars`` (the pool), every sum region's ``offset`` there (-1 for one
        without an extent) and ``width`` (Python ints), ``offset_t`` / ``width_t`` on the device."""
        self._open()
        layout = self.state.batch_layout(1)
        if self._table is None or self._table[0] != self._version:
            offsets = np.full(len(layout["sum_widths"]), -1, dtype = np.int64)
            if self.extents:
                offsets[list(self.extents)] = [extent[0] for extent in self.extents.values()]
            self._table = (self._version, offsets.tolist(), torch.from_numpy(offsets).to(self.state._device))
        return dict(mars = self.buffer, offset = self._table[1], width = layout["sum_widths"], offset_t = self._table[2],
                    width_t = layout["sum_widths_t"])
