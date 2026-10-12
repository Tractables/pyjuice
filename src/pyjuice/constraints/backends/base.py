"""
The interface every inference backend for constrained queries implements.

A backend serves some pairs of a constraint and a PC: :meth:`Backend.refusals` says why it cannot serve one (from
the constraint's capabilities and the PC's :class:`~pyjuice.constraints.structure.PCStructure` alone), and
:meth:`Backend.build` compiles one it can serve -- once, from the structure, so the result stays valid while the
PC's parameters change. A built backend answers the queries of the
:class:`~pyjuice.constraints.ConstrainedCircuit` that holds it; a query it does not serve raises
:class:`NotImplementedError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

import torch


class Backend(ABC):
    """
    An inference backend: what :func:`pyjuice.constraints.compile` built for one constraint and one PC.

    Options of a query that only some backends have (e.g. a numerical precision) are keyword arguments of its
    query methods, passed through by the :class:`~pyjuice.constraints.ConstrainedCircuit`. The circuit gives the
    queries their inputs in one form (see :mod:`pyjuice.constraints.evidence`): ``data`` [B, n] token ids and
    ``missing_mask`` [B, n] bool on ``data``'s device, ``targets`` None or a bool mask ([n] or [B, n]), and
    ``candidates`` None or [B, n, k] token ids (a view where shared; an id outside the vocabulary pads).
    """

    #: the name ``pyjuice.constraints.compile(..., backend = name)`` selects it by
    name: ClassVar[str]
    #: whether its queries are exact (up to floating point)
    exact: ClassVar[bool]
    #: the options its decoders start from (:meth:`decoder`), under the caller's
    decoder_defaults: ClassVar[Dict[str, Any]] = {}

    @classmethod
    @abstractmethod
    def refusals(cls, constraint, structure) -> List[str]:
        """Every reason this backend cannot serve the constraint on a PC of this structure (empty if it can)."""

    @classmethod
    @abstractmethod
    def build(cls, constraint, pc, structure) -> "Backend":
        """Compile the pair (one :meth:`refusals` accepts) for queries."""

    @abstractmethod
    def rebind(self, pc, structure) -> "Backend":
        """The same constraint on another PC whose structure equals this one's (parameters may differ)."""

    @abstractmethod
    def to(self, device) -> None:
        """Move everything compiled from the pair to ``device`` (the PC itself is moved by its circuit)."""

    def info(self) -> Dict[str, Any]:
        """A summary of what the backend compiled (merged into :meth:`ConstrainedCircuit.info`)."""
        return {}

    def marginal(self, data: torch.Tensor, missing_mask: torch.Tensor, **options) -> torch.Tensor:
        """``log p(C, e)`` for every sample (see :meth:`~pyjuice.constraints.ConstrainedCircuit.marginal`)."""
        raise NotImplementedError(f"The {self.name} backend does not serve `marginal`.")

    def conditional(self, data: torch.Tensor, missing_mask: torch.Tensor, targets: Optional[torch.Tensor] = None,
                    return_log_marginal: bool = False, candidates: Optional[torch.Tensor] = None, **options):
        """``log p(x_t = v | e, C)`` for every target, of every token or of the ``candidates`` (see
        :meth:`~pyjuice.constraints.ConstrainedCircuit.conditional`)."""
        raise NotImplementedError(f"The {self.name} backend does not serve `conditional`.")

    def sample(self, *args, **kwargs):
        """Samples from the PC conditioned on the constraint (and evidence)."""
        raise NotImplementedError(f"The {self.name} backend does not serve `sample`.")

    def decoder(self, cc, **options):
        """A decoding session over ``cc`` (see :meth:`~pyjuice.constraints.ConstrainedCircuit.decoder`): this one
        reuses nothing between queries; a backend that can returns its own
        :class:`~pyjuice.constraints.decoder.Decoder`."""
        from ..decoder import Decoder
        return Decoder(cc, **{**self.decoder_defaults, **options})
