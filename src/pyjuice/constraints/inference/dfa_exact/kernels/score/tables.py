"""
What every scoring kernel reads: a program's tables by class (:class:`ScoreTables`).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ScoreTables:
    """
    Where every position's conditionals are scored. The input nodes over a position read rows of the class masses
    (:meth:`~..state.State.class_mass_rows`; tied nodes one row), the rows of their
    parameter tables: positions whose input nodes read the same rows form a ROW GROUP, scored together against one
    token-major table of those rows (an HMM with tied emissions: one group).

    :ivar group_of: int64 [n] (host): every position's row group
    :ivar rows: per row group, the sorted rows its positions' input nodes read (host int64)
    :ivar rows_t: the same on the device
    :ivar col_of: int64 [input nodes] (host, by ``node_mars`` row from the inputs' first): every input node's column
        in its position's row group, -1 for an input node no product step reads
    :ivar nodes: per row group, int64 [n, rows of the group] (host): the input node (from the inputs' first) at every
        position of the group and column, -1 for none
    """

    group_of: np.ndarray
    rows: list
    rows_t: list
    col_of: np.ndarray
    nodes: list


def score_tables(prog) -> ScoreTables:
    """The :class:`ScoreTables` of a lifted :class:`~.program.Program` (built on first use, kept on it): the input
    operands of its product steps, by position."""
    if getattr(prog, "score", None) is not None:
        return prog.score
    mass_row = prog.mass_row.cpu().numpy().astype(np.int64)
    u, t = [], []
    for kind, _, layers in prog.steps:
        if kind != "prod":
            continue
        for stages, _, _ in layers:
            for launches in stages:
                for la in launches:
                    cols = {"input_block": (1, 2), "block_input": (4, 5)}.get(la.form)
                    if cols is not None:
                        u.append(la.rows[:, cols[0]].numpy())
                        t.append(la.rows[:, cols[1]].numpy())
    u = np.concatenate(u) - prog.input_start if u else np.zeros(0, dtype = np.int64)
    t = np.concatenate(t) if t else np.zeros(0, dtype = np.int64)
    group_of = np.full(prog.n, -1, dtype = np.int64)
    col_of = np.full(prog.input_end - prog.input_start, -1, dtype = np.int64)
    groups, rows, nodes = {}, [], []
    order = np.argsort(t, kind = "stable")
    u, t = u[order], t[order]
    bounds = np.searchsorted(t, np.arange(prog.n + 1))
    for pos in range(prog.n):
        at = u[bounds[pos]:bounds[pos + 1]]
        if at.size == 0:
            continue
        r = np.unique(mass_row[at])
        g = groups.setdefault(r.tobytes(), len(rows))
        if g == len(rows):
            rows.append(r)
            nodes.append(np.full((prog.n, len(r)), -1, dtype = np.int64))
        group_of[pos] = g
        col_of[at] = np.searchsorted(r, mass_row[at])
        nodes[g][pos, col_of[at]] = at
    prog.score = prog.assembly.score = ScoreTables(group_of = group_of, rows = rows, col_of = col_of, nodes = nodes,
                                                   rows_t = [torch.from_numpy(r).to(prog.dev) for r in rows])
    return prog.score
