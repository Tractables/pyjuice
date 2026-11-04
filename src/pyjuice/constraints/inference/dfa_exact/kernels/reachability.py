"""
Which columns of a boundary are reached from which columns of an earlier one, on the device
(:class:`~..compiled.layout.Reachability`), as bitsets: for every column of the boundary a sweep has reached, a bit per
column of the boundary it started from. A step is an OR over every column's predecessors, a word at a time, through the
distinct (column, successor) pairs of the automaton's alike boundaries
(:class:`~.transitions.BoundaryGroups`).
"""

from __future__ import annotations

import numpy as np
import torch
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from pyjuice.utils.util import host_cdiv

from ....evidence import current_device
from .transitions import boundary_groups

#: the words of a program
WORD_TILE = 16


@triton_jit
def _or_predecessors_kernel(out, cur, ptr, cols, base, words, TILE: tl.constexpr):
    """One program per (column ``j`` of the next boundary, tile of its words): ``out[j] = OR of cur[i]`` over ``j``'s
    predecessor columns ``i``, ``cols[ptr[base + j] .. ptr[base + j + 1]]``."""
    j = tl.program_id(0)
    w = tl.program_id(1) * TILE + tl.arange(0, TILE)
    m = w < words
    acc = tl.zeros([TILE], dtype = tl.int32)
    for e in range(tl.load(ptr + base + j), tl.load(ptr + base + j + 1)):
        acc = acc | tl.load(cur + tl.load(cols + e).to(tl.int64) * words + w, mask = m, other = 0)
    tl.store(out + j.to(tl.int64) * words + w, acc, mask = m)


class SweepTables:
    """Every group of alike boundaries' predecessors: those of column ``j`` at the next boundary of group ``g`` are
    ``cols[ptr[g * (W + 1) + j] .. ptr[g * (W + 1) + j + 1]]``."""

    def __init__(self, tr):
        groups = boundary_groups(tr)
        W, dev = tr.W, tr.device
        self.group, self.W = groups.group, W
        parts, rows = [], []
        total = 0
        for k, end in groups.batches:
            Gb = end - k
            g, q, _, s = groups.entries(k, end)
            key = torch.unique((g * (W + 1) + s) * (W + 1) + q)         # by group, successor, column
            at = key // (W + 1)
            counts = torch.bincount(at, minlength = Gb * (W + 1)).view(Gb, W + 1)
            before = torch.cumsum(counts, dim = 1) - counts             # in a group, the columns' first
            base = torch.cumsum(counts.sum(dim = 1), dim = 0) - counts.sum(dim = 1)
            rows.append((before + base[:, None] + total).view(-1))
            parts.append((key % (W + 1)).to(torch.int32))
            total += key.numel()
        self.cols = torch.cat(parts) if parts else torch.zeros(1, dtype = torch.int32, device = dev)
        ptr = torch.cat(rows) if rows else torch.zeros(0, dtype = torch.int64, device = dev)
        self.ptr = torch.cat([ptr, torch.tensor([total], dtype = torch.int64, device = dev)])


def sweep_tables(tr) -> SweepTables:
    """The :class:`SweepTables` of a layout's transitions."""
    return SweepTables(tr)


def column_bits(width: int, dev) -> torch.Tensor:
    """int32 [width, words]: every column reached from itself alone."""
    words = max(host_cdiv(width, 32), 1)
    out = torch.zeros(width, words, dtype = torch.int64, device = dev)
    at = torch.arange(width, device = dev)
    out[at, at // 32] = 1 << (at % 32)
    return out.to(torch.int32)


def advance(bits: torch.Tensor, t: int, next_width: int, tables: SweepTables) -> torch.Tensor:
    """int32 [columns at ``t``, words] -> [``next_width``, words]: one token on from boundary ``t``."""
    words = bits.size(1)
    out = torch.empty(next_width, words, dtype = torch.int32, device = bits.device)
    if next_width > 0 and words > 0:
        with current_device(bits.device):
            _or_predecessors_kernel[(next_width, host_cdiv(words, WORD_TILE))](
                out, bits, tables.ptr, tables.cols, tables.group[t] * (tables.W + 1), words, TILE = WORD_TILE, num_warps = 1)
    return out


def stack_bits(swept: list, most_columns: int, entries: int) -> torch.Tensor:
    """int32 [sweeps, ``most_columns``, words of ``entries``]: the bits of ``swept``, ``(table, first word, words)``: of
    every sweep the ``words`` of ``table`` [its columns, ...] from its first, one above the other, zeros past their columns
    and words -- a few device calls a table."""
    dev = swept[0][0].device
    words = max(host_cdiv(entries, 32), 1)
    out = torch.zeros(len(swept), most_columns, words, dtype = torch.int32, device = dev)
    tables = {}
    for k, (table, first, own) in enumerate(swept):
        tables.setdefault((id(table), own), (table, own, [], []))
        tables[id(table), own][2].append(k)
        tables[id(table), own][3].append(first)
    for table, own, ks, firsts in tables.values():
        at = np.asarray(firsts, dtype = np.int64)[:, None] + np.arange(own, dtype = np.int64)[None, :]
        out[torch.tensor(ks, device = dev), :table.size(0), :own] = table[:, torch.from_numpy(at).to(dev)].transpose(0, 1)
    return out


def pairs_of_bits(bits: torch.Tensor, entries: int) -> torch.Tensor:
    """bool [..., ``entries``, columns]: which column each entry column reaches (``bits`` int32 [..., columns, words]:
    several sweeps of one shape, stacked, in as many device calls as one)."""
    *batch, columns, words = bits.shape
    out = torch.zeros(*batch, entries, columns, dtype = torch.bool, device = bits.device)
    shifts = torch.arange(32, device = bits.device, dtype = torch.int32)
    for c0 in range(0, columns, 1024):                             # (a word a bit, 32 times: not all at once)
        chunk = (bits[..., c0:c0 + 1024, :, None] >> shifts) & 1
        out[..., c0:c0 + 1024] = chunk.view(*batch, chunk.size(-3), words * 32)[..., :entries].transpose(-1, -2) != 0
    return out
