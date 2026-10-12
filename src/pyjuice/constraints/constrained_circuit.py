"""
The result of :func:`pyjuice.constraints.compile`: a PC under a constraint, ready for queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch

from .backends.base import Backend
from .evidence import candidate_tensor, evidence, target_mask
from .language.base import Constraint
from .structure import PCStructure, analyze_structure


class ConstrainedCircuit:
    """
    A PC under a constraint: the PC, the constraint, the PC's structure, and the :class:`~.backends.base.Backend`
    compiled from the pair, which answers the queries -- its methods.

    Everything the backend compiled depends only on the constraint and the PC's STRUCTURE, never on parameter values
    or evidence, so it stays valid while the PC's parameters change (e.g. CoDD's external parameters at every step).
    The PC itself is fixed: :attr:`pc` cannot be reassigned (:meth:`with_pc` puts the same constraint on another PC
    with the same structure), and it moves to another device together with the backend, through :meth:`to`. A PC
    moved on its own makes every query raise.

    Created by :func:`pyjuice.constraints.compile`; not meant to be constructed directly.

    :ivar pc: the PC queries run on
    :ivar constraint: the constraint
    :ivar structure: the PC's :class:`~pyjuice.constraints.structure.PCStructure`
    :ivar backend: the :class:`~.backends.base.Backend` that answers the queries (what it compiled, and its reports,
        are its own: e.g. ``cc.backend.num_states`` for the ``dfa_exact`` backend)
    :ivar compile_time_s: wall-clock seconds :func:`~pyjuice.constraints.compile` (or :meth:`with_pc`) took
    """

    def __init__(self, pc, constraint: Constraint, structure: PCStructure, backend: Backend, compile_time_s: float):
        self._pc = pc
        self._device = pc.params.device             # where the backend's tables live
        self.constraint = constraint
        self.structure = structure
        self.backend = backend
        self.compile_time_s = compile_time_s

    @property
    def pc(self):
        """The PC queries run on (fixed; see the class docstring)."""
        return self._pc

    @property
    def n(self) -> int:
        """Number of positions: the PC's number of variables."""
        return self.structure.num_vars

    def _check_pc_unchanged(self):
        """Refuse a query on a PC that moved on its own (one attribute comparison per query)."""
        if self._pc.params.device != self._device:
            raise RuntimeError(f"The PC moved from {self._device} to {self._pc.params.device} on its own, but the "
                               f"constrained circuit's tables are still on {self._device}. Move both with "
                               f"`cc.to(device)`, or compile the constraint again.")

    def to(self, device) -> "ConstrainedCircuit":
        """
        Move the PC and everything compiled from the pair to ``device`` (in place, like
        :meth:`TensorCircuit.to`), and return this constrained circuit.

        :param device: an int ordinal, a string such as ``"cuda:1"`` or ``"cpu"``, or a ``torch.device``
        """
        self._pc.to(device)
        self._device = self._pc.params.device
        self.backend.to(self._device)
        return self

    def with_pc(self, pc) -> "ConstrainedCircuit":
        """
        The same constraint on another PC with the same structure (e.g. a drafter and a verifier, or a
        copy of the PC on another device). The structure analysis is checked by signature and reused, and the
        backend rebinds what it compiled from the constraint (see :meth:`~.backends.base.Backend.rebind`).

        :param pc: a compiled PC whose structure equals this one's (parameters may differ)
        :type pc: TensorCircuit
        """
        from .compiler import _check_pc
        from .errors import ConstraintCompileError

        t0 = time.perf_counter()
        _check_pc(pc)
        structure = analyze_structure(pc)
        if structure.signature != self.structure.signature:
            raise ConstraintCompileError("Cannot rebind the constrained circuit: the new PC's structure differs "
                                         "from the one the constraint was compiled against. Compile the "
                                         "constraint against the new PC with `pyjuice.constraints.compile` "
                                         "instead.")
        backend = self.backend.rebind(pc, structure)
        return ConstrainedCircuit(pc, self.constraint, structure, backend, compile_time_s = time.perf_counter() - t0)

    # ---------------------------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """A summary of the constrained circuit: its own entries, then the backend's (:meth:`Backend.info`)."""
        return dict(backend = self.backend.name, exact = self.backend.exact, n = self.n,
                    shape_counts = self.structure.shape_counts, compile_time_s = self.compile_time_s,
                    constraint = self.constraint.info(), **self.backend.info())

    def __repr__(self) -> str:
        return f"ConstrainedCircuit(n={self.n}, backend={self.backend!r})"

    # ---------------------------------------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------------------------------------

    def marginal(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, *, mask_id: int = -1,
                 **options) -> torch.Tensor:
        """
        ``log p(C, e)`` for every sample: the log-probability that the PC generates a string that satisfies the
        constraint and agrees with the observed tokens.

        :param data: [B, n] token ids, or a prefix [B, L] (``L <= n``) whose later positions are missing; a token equal
            to ``mask_id`` is missing
        :type data: torch.Tensor

        :param missing_mask: None, [n] or [B, n]; True where a token is marginalized (as well as where ``data`` holds
            ``mask_id``)
        :type missing_mask: Optional[torch.Tensor]

        :param mask_id: the token id marking a missing position, never a real token (default -1)
        :type mask_id: int

        :param options: the backend's options (e.g. ``precision`` for ``dfa_exact``, see its ``marginal``)

        :returns: [B, number of root nodes], as :func:`pyjuice.queries.marginal`
        """
        self._check_pc_unchanged()
        data, missing = evidence(data, missing_mask, mask_id, self.n)
        return self.backend.marginal(data, missing, **options)

    def conditional(self, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None,
                    targets: Optional[torch.Tensor] = None, return_log_marginal: bool = False, *, mask_id: int = -1,
                    candidates: Optional[torch.Tensor] = None, **options):
        """
        ``log p(x_t = v | e, C)``: the distribution of every target position's token given the observed tokens and the
        constraint, for every target (sample, position).

        :param data: [B, n] token ids, or a prefix [B, L] (``L <= n``) whose later positions are missing; a token equal
            to ``mask_id`` is missing
        :type data: torch.Tensor

        :param missing_mask: None, [n] or [B, n]; True where a token is marginalized (as well as where ``data`` holds
            ``mask_id``)
        :type missing_mask: Optional[torch.Tensor]

        :param targets: None (every missing position of every sample); a bool mask [n] or [B, n] of missing
            positions; or the positions themselves, [k] for every sample or [B, k] per sample (-1 pads)
        :type targets: Optional[torch.Tensor]

        :param return_log_marginal: also return ``log p(C, e)`` of every row's sample
        :type return_log_marginal: bool

        :param mask_id: the token id marking a missing position, never a real token (default -1)
        :type mask_id: int

        :param candidates: None (every token), or the tokens whose log-probabilities to return: [k] for every
            target, [B, k] for every target of sample ``b``, or [B, n, k] per sample and position; an id outside the
            vocabulary (-1) pads, and gets -inf. Exact, and cheaper than every token when ``k`` is small
        :type candidates: Optional[torch.Tensor]

        :param options: the backend's options (e.g. ``precision`` for ``dfa_exact``, see its ``conditional``)

        :returns: [number of targets, V] (or [number of targets, k] with ``candidates``), one row per target in the
            order ``dense[targets]`` lists them (sample by sample, positions in order), -inf rows where ``p(C, e) =
            0``; with ``return_log_marginal``, also [number of targets]
        """
        self._check_pc_unchanged()
        data, missing = evidence(data, missing_mask, mask_id, self.n)
        B = data.size(0)
        return self.backend.conditional(data, missing, target_mask(targets, B, self.n),
                                        return_log_marginal = return_log_marginal,
                                        candidates = candidate_tensor(candidates, B, self.n), **options)

    def sample(self, *args, **kwargs):
        """Samples from the PC conditioned on the constraint (and evidence)."""
        self._check_pc_unchanged()
        return self.backend.sample(*args, **kwargs)

    def decoder(self, *args, **kwargs):
        """Incremental (token-by-token) constrained decoding."""
        self._check_pc_unchanged()
        return self.backend.decoder(*args, **kwargs)
