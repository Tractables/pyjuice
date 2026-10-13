"""
The dfa_exact backend's decoding session (:class:`DFAExactDecoder`): the circuit's queries, each reusing the
per-sample values its previous one left in the buffers wherever the evidence did not change them.

A sum region's inside values depend only on the evidence on its scope, and its top-down (log outside) values only on
the evidence outside it. Every row of a call continues a row of the previous call
(:func:`~pyjuice.constraints.decoder.match_rows`): it observes all that row observed, maybe more. So where no row
newly observes a position of a region's scope, the region's inside values are the ones the previous call left, row by
row. Where no row newly observes a position outside the scope, its top-down values are the previous call's too. The
call holds the ones it needs (``held``, see :mod:`.plans`) and computes none of them. They stay in place when every row
continues the row at its own index. Otherwise they are first moved into the call's rows
(:func:`~..kernels.blocks.move_blocks`: only the lanes live in the previous call are read), and the others are given
up.

They are exact under the call's evidence pruning (:mod:`..kernels.live`). Every slot holds its true value if live and
at most its true value if dead. More evidence leaves fewer slots live, never more, so that stays true: a slot live
before and dead now holds its true value, as the slots of a block kept once do. Every kernel reads such a slot only as
a term of a product that is -inf anyway.

Nothing is reused after a call where a row continues no row of the previous one (a new prompt, a position observed and
then missing again). Nothing is reused when anything else used the buffers in between (:attr:`~..state.State.epoch`:
another query, inside the session or not), or when the precision changed.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ....decoder import Decoder
from ..kernels.blocks import move_blocks, move_table
from ..kernels.common import pack_aligned
from ..kernels.live import Liveness
from .conditional import conditional
from .marginal import marginal


def _empty(numel: int, device) -> torch.Tensor:
    """New storage for moved values (see :meth:`~..state.State.replace_storage`): not filled."""
    return torch.empty(numel, device = device)


@dataclass
class _Kept:
    """What the buffers hold after a decoder's call: per sum region, whether ``node_mars`` holds its per-sample inside
    values for the call's rows, and whether ``node_flows`` holds its top-down values. They are valid while the state
    is the same (a weak reference: a backend moved to another device frees its old state), its buffers were not used
    since (``epoch``), and the next call has the same ``precision``. ``liveness``: the call's liveness (None: every
    lane), whose live lanes hold their true values in every region valid here."""
    state: weakref.ref
    epoch: int
    precision: str
    B: int
    inside: np.ndarray
    top_down: np.ndarray
    liveness: Optional[Liveness]


class DFAExactDecoder(Decoder):
    """
    A :class:`~pyjuice.constraints.decoder.Decoder` over a constrained circuit with the dfa_exact backend
    (:meth:`~..backend.DFAExactBackend.decoder`): a query reuses what the previous one computed where the evidence did
    not change it (see the module docstring). It keeps its own copy of the last call's evidence.
    """

    remembers = True

    def __init__(self, cc, **options):
        super().__init__(cc, **options)
        self._kept = None                           # what the buffers hold for the next call (`_Kept`)
        self._regions = None                        # see `_region_tables`
        self._layouts = {}                          # batch size -> (sum offsets, sum widths) as int64 arrays

    def __exit__(self, *exc) -> bool:
        self._kept, self._layouts = None, {}
        return super().__exit__(*exc)

    def _marginal(self, data: torch.Tensor, missing: torch.Tensor, precision: str = "fp32") -> torch.Tensor:
        be = self.cc.backend
        held, keep = self._reuse(data, missing, None, precision, marginal = True)
        report = {}
        out = marginal(be, data, missing, precision, held = None if held is None else held[0], report = report)
        self._after(data.size(0), precision, keep, report)
        return out

    def _conditional(self, data: torch.Tensor, missing: torch.Tensor, targets: Optional[torch.Tensor],
                     return_log_marginal: bool, candidates: Optional[torch.Tensor], precision: str = "fp32"):
        be = self.cc.backend
        held, keep = self._reuse(data, missing, targets, precision, marginal = False)
        report = {}
        out = conditional(be, data, missing, targets, precision, return_log_marginal = return_log_marginal,
                          held = None if held is None else dict(inside = held[0], top_down = held[1]),
                          candidates = candidates, report = report)
        self._after(data.size(0), precision, keep, report)
        return out

    # ---------------------------------------------------------------------------------------------

    def _reuse(self, data: torch.Tensor, missing: torch.Tensor, targets: Optional[torch.Tensor], precision: str,
               marginal: bool):
        """
        ``(held, keep)`` for a call: ``held`` None (nothing reused) or ``(inside, top_down)``, the values the call
        holds (bool per sum region, see :mod:`.plans`), in place or moved there; ``keep`` None or ``(inside,
        top_down)``, those the buffers still hold for the call's rows besides what it computes.
        """
        kept, self._kept = self._kept, None         # given up until the call returns (if it raises: for good)
        be = self.cc.backend
        state, B = be.state, data.size(0)
        if kept is None or kept.state() is not state or kept.epoch != state.epoch or kept.precision != precision:
            return None, None
        if not marginal:
            asked = (missing if targets is None else targets).cpu().numpy()  # every missing position by default
            asked = asked.any(axis = 0) if asked.ndim == 2 else asked
            if asked.shape != (be.n,):
                return None, None                                       # the query says what is wrong
        parents, new = self._match_rows(data, missing)                  # on the host (see `remembers`)
        if (parents < 0).any():
            return None, None

        _, cover, _, _ = self._region_tables(be.program())
        on_new = cover[new.any(axis = 0)]                               # [newly observed positions, regions]
        inside = kept.inside & ~on_new.any(axis = 0)                    # nothing new on the scope
        top_down = kept.top_down & on_new.all(axis = 0)                 # nothing new outside it
        if marginal:                                                    # every inside value; no top-down one
            held = (inside, np.zeros_like(top_down))
        else:                                                           # those of a scope missing a target; meeting one
            on_asked = cover[asked]
            held = (inside & ~on_asked.all(axis = 0), top_down & on_asked.any(axis = 0))
        if B == kept.B and (parents == np.arange(B)).all():
            return held, (inside, top_down)                             # in place: all of them stay
        self._move(state, kept.B, B, parents, *held, kept.liveness)
        return held, held

    def _move(self, state, B_old: int, B: int, parents: np.ndarray, inside: np.ndarray, top_down: np.ndarray,
              liveness: Optional[Liveness]):
        """The ``inside`` and ``top_down`` values of the previous call's rows (``B_old`` of them) moved into this
        call's: row ``b`` gets row ``parents[b]``'s, in new storage laid out for a batch of ``B``, which replaces the
        old (in one pass: the old and the new layout of a storage overlap); only the lanes live under the previous
        call's ``liveness``, -inf at the others."""
        regs = np.concatenate([np.nonzero(inside)[0], np.nonzero(top_down)[0]])
        if regs.size == 0:
            return
        be = self.cc.backend
        _, _, rows, slots = self._region_tables(be.program())
        rows, slots = rows[regs], slots[regs]
        which = np.repeat(np.array([0, 1]), [int(inside.sum()), int(top_down.sum())])
        old_offset, old_width = self._layout(state, B_old)
        new_offset, new_width = self._layout(state, B)
        prog = be.program()
        bounds = np.asarray(prog.sum_ivs, dtype = np.int64).reshape(-1, 2)[prog.reg_iv_np[regs]]
        table = move_table(old_offset[regs], old_width[regs], new_offset[regs], new_width[regs], slots, rows, which,
                           bounds)
        tables, starts = pack_aligned([table.ravel(), parents], np.int64)
        tables = torch.from_numpy(tables).to(be.device)
        size = state.batch_layout(B)["node_size"]
        srcs, dsts = [], []
        for name, any_ in (("node_mars", inside.any()), ("node_flows", top_down.any())):
            if any_:
                dsts.append(_empty(size, be.device))
                srcs.append(state.replace_storage(name, dsts[-1]))
        move_blocks(srcs, dsts, tables[:starts[1]], B, int(rows.max()), int(slots.max()),
                    parents = tables[starts[1]:starts[1] + B], live = None if liveness is None else liveness.live,
                    prog = prog)

    def _after(self, B: int, precision: str, keep, report: dict):
        """What the buffers hold after a call: what it kept, and what it computed per sample."""
        be = self.cc.backend
        R = len(be.sum_regions)
        inside, top_down = (np.zeros(R, dtype = bool),) * 2 if keep is None else keep
        inside, top_down = inside.copy(), top_down.copy()
        plan = report.get("plan")
        if plan is not None:
            inside |= plan.sample[be.program().reg_iv_np]
            if plan.top_down is not None:
                top_down |= plan.top_down.sample
        self._kept = _Kept(weakref.ref(be.state), be.state.epoch, precision, B, inside, top_down,
                           report.get("liveness"))

    def _region_tables(self, prog):
        """``(prog, cover, rows, slots)``: whether every sum region's scope holds every position (bool [n, R]), and
        every region's rows and slots per sample (int64 [R]); read once per Program."""
        if self._regions is None or self._regions[0] is not prog:
            regions = np.asarray(self.cc.backend.sum_regions, dtype = np.int64).reshape(-1, 3)
            bounds = np.asarray([prog.sum_ivs[k] for k in prog.reg_iv_host], dtype = np.int64).reshape(-1, 2)
            at = np.arange(self.cc.backend.n)[:, None]
            cover = (bounds[None, :, 0] <= at) & (at < bounds[None, :, 1])
            self._regions = (prog, cover, regions[:, 1] - regions[:, 0], regions[:, 2])
        return self._regions

    def _layout(self, state, B: int):
        """Every sum region's offset and row width (floats) in the buffers of a batch of ``B`` (int64 arrays)."""
        if B not in self._layouts:
            lay = state.batch_layout(B)
            self._layouts[B] = (np.asarray(lay["sum_offsets"], dtype = np.int64),
                                np.asarray(lay["sum_widths"], dtype = np.int64))
        return self._layouts[B]
