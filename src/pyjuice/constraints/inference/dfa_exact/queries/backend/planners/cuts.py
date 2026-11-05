"""
What a plan runs of a launch: index lists, in one copy, and a launch cut to some of its steps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ....kernels.common import IndexList, pack_aligned
from ....kernels.prod import Cut, Launch


class IndexLists:
    """The index lists of one query plan, copied to the device together (a copy from the host waits for the stream:
    one per plan)."""

    def __init__(self):
        self.items = []

    def add(self, idx, height: Optional[int] = None) -> IndexList:
        item = IndexList(idx, height)
        self.items.append(item)
        return item

    def upload(self, dev):
        if self.items:
            buf, starts = pack_aligned([i.host for i in self.items], np.int32)
            buf = torch.from_numpy(buf).to(dev)
            for i, s in zip(self.items, starts.tolist()):
                i.tensor, i.host = buf[s:s + len(i.host)], None


def cut_launch(launch: Launch, sel: np.ndarray, lists: IndexLists, parts: list) -> Optional[Cut]:
    """The :class:`Cut` of ``launch`` running the steps ``sel`` marks (bool [S], host), None if none; its index lists
    go into ``lists``, and into ``parts`` what its sizes are the most of (``(launch number, its parts or None for all)``:
    :func:`cut_sizes`, by the place there)."""
    count = int(sel.sum())
    if count == 0:
        return None
    every = count == sel.size or launch.part_of is None
    at = 3 * len(parts)
    parts.append((launch.number, None if every else np.unique(launch.part_of.numpy()[sel])))
    chunks, groups = [(0, sel.size, None)], None
    if launch.form in ("input_block", "block_input"):
        _, chunks, groups = launch.steps
    cuts = []
    for s0, s1, job in chunks:
        part = sel[s0:s1]
        c = int(part.sum())
        if c:
            cuts.append((s0, s1, job, None if c == s1 - s0 else lists.add(np.nonzero(part)[0]), c))
    gcuts = None
    if groups is not None:
        gcuts = {}
        for G, firsts in launch.groups_host.items():
            gsel = sel[firsts[:-1]]
            c = int(gsel.sum())
            gcuts[G] = (None if c == gsel.size else lists.add(np.nonzero(gsel)[0]), c)
    return Cut(cuts, gcuts, at, count)


def cut_parts(assembly, parts: list) -> tuple:
    """``(part numbers, ptr)``: :func:`cut_launch`'s ``parts`` as int64 arrays, by the numbers of the launches' parts."""
    first = assembly.part_ptr
    runs = [np.arange(first[L], first[L + 1]) if own is None else first[L] + own for L, own in parts]
    return (np.concatenate(runs) if runs else np.zeros(0, dtype = np.int64),
            np.cumsum([0] + [r.size for r in runs], dtype = np.int64))


def cut_sizes(prog, part: np.ndarray, ptr: np.ndarray) -> list:
    """What this program's automaton says of cuts (:func:`cut_parts`): every one's most entry columns, exit columns and
    slots, flat."""
    return np.maximum.reduceat(prog.launch_parts[part], ptr[:-1], axis = 0).ravel().tolist() if part.size else []


class _Packed:
    """Int64 arrays packed into one (``data``, ``offsets``: a numba call pays per argument), readable by name."""

    def __init__(self, names, arrays: dict):
        parts = [np.asarray(arrays[k], dtype = np.int64).reshape(-1) for k in names]
        self.offsets = np.cumsum([0] + [p.size for p in parts]).astype(np.int64)
        self.data = np.concatenate(parts)
        for k, a, b in zip(names, self.offsets[:-1], self.offsets[1:]):
            setattr(self, k, self.data[a:b])


class _Flat:
    """Arrays built by appending, then concatenated (see :func:`_top_down_tables`)."""

    def __init__(self, *names):
        self._parts = {name: [] for name in names}

    def add(self, name, values):
        self._parts[name].append(np.asarray(values, dtype = np.int64).reshape(-1))

    def count(self, name) -> int:
        return sum(a.size for a in self._parts[name])

    def done(self) -> dict:
        return {k: np.concatenate(v) if v else np.zeros(0, dtype = np.int64) for k, v in self._parts.items()}
