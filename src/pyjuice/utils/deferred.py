"""
Work a layer defers to the end of a backward's inner layers (:func:`deferred_scope`, :func:`current_scope`): the
backward opens a scope around its inner layers and runs what they deferred after the last of them -- inside the same
pass, so a CUDA graph recorded from it replays the deferred launches too. A tied group's parameter flows use it: each
tied layer captures its step's operands, and one contraction after the pass adds the whole group's flows (see
`pyjuice.nodes.external_params.blockscale.deferred_par`).

Outside a scope (a layer's backward called directly) nothing can be deferred: :func:`current_scope` is None and the
caller runs its work at once. A scope that ends in an exception drops what it held -- state a layer keeps for a scope
must therefore be checked against the scope it belongs to (the identity of :func:`current_scope`) before reuse.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, List, Optional


class DeferredScope:
    """What the inner layers of one backward deferred, run in order by :meth:`run`."""

    def __init__(self):
        self.pending: List[Callable[[], None]] = []

    def defer(self, fn: Callable[[], None]) -> None:
        self.pending.append(fn)

    def run(self) -> None:
        pending, self.pending = self.pending, []
        for fn in pending:
            fn()


_SCOPES: List[DeferredScope] = []


def current_scope() -> Optional[DeferredScope]:
    """The innermost open scope, or None outside every one."""
    return _SCOPES[-1] if _SCOPES else None


@contextmanager
def deferred_scope():
    """A scope for the inner layers of one backward: the caller runs ``scope.run()`` after the last layer."""
    scope = DeferredScope()
    _SCOPES.append(scope)
    try:
        yield scope
    finally:
        _SCOPES.pop()
