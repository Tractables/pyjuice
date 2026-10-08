"""
Base class for constraints, their capabilities, and lazy composition.

A :class:`Constraint` is a set of token sequences (a formal language) over the vocabulary
``{0, ..., vocab_size - 1}``. It knows nothing about any PC: when it is compiled against a PC with
``n`` variables, the event it denotes is "the sequence x_0, ..., x_{n-1} of the PC's variables, read in
variable-index order, is in the language". Which positions matter, and any notion of length, are part
of the language itself (e.g. a constraint that accepts anything at its first ``m`` positions).

CAPABILITIES. Inference backends need different things from a constraint, so a constraint is not
required to be convertible into one canonical form. Membership (:meth:`Constraint.accepts`) is the only
requirement; everything else is an optional capability, reported by :meth:`Constraint.capabilities`:

==============  ================================================================  ==========================
capability      method                                                            used by
==============  ================================================================  ==========================
``accepts``     :meth:`~Constraint.accepts`: membership of a whole sequence        tests, rejection sampling
``matcher``     :meth:`~Constraint.matcher`: incremental prefix tracking           support masks (SMC, decoding)
``wmc``         :meth:`~Constraint.wmc`: weighted model count, factorised weights  mean-field proposals
``sample``      :meth:`~Constraint.sample`: exact draws under factorised weights   mean-field / collapsed
``automaton``   :meth:`~Constraint.automaton`: finite-state form                   exact lifting
``relax``       :meth:`~Constraint.relax`: finite-state over-approximation         coarsened proposals
==============  ================================================================  ==========================

The compiler picks a backend from what is available and reports its choice. A capability that is
listed must work; one that is not raises :class:`NotImplementedError`.

Constraints are immutable. Combining them with ``&``, ``|``, ``~`` and :meth:`Constraint.concat` builds
a small expression tree and does no automaton construction; products are built at compile time, when
``n`` is known and only the reachable part is needed. Every constraint has a
:meth:`~Constraint.fingerprint`, a stable hash of its structure, which also defines equality, so
identical constraints in a batch (or reused across calls) can share compiled tables.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple, Union

import torch


Tokens = Union[Sequence[int], torch.Tensor]

#: The optional capabilities, in addition to the required ``accepts``.
OPTIONAL_CAPABILITIES = ("matcher", "wmc", "sample", "automaton", "relax")


def _as_tokens(tokens: Tokens) -> Tuple[int, ...]:
    if isinstance(tokens, torch.Tensor):
        if tokens.dim() != 1:
            raise ValueError(f"Expected a 1-D token sequence, got a tensor of shape {tuple(tokens.shape)}.")
        return tuple(int(t) for t in tokens.tolist())
    return tuple(int(t) for t in tokens)


def _hash_into(h, obj: Any):
    """Feed a structural description into a hashlib object, deterministically across processes."""
    if isinstance(obj, Constraint):
        h.update(b"C"); h.update(obj.fingerprint().encode())
    elif isinstance(obj, torch.Tensor):
        t = obj.detach().to("cpu").contiguous()
        h.update(b"T"); h.update(str(t.dtype).encode()); h.update(repr(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    elif isinstance(obj, (tuple, list)):
        h.update(b"(" if isinstance(obj, tuple) else b"[")
        for x in obj:
            _hash_into(h, x)
        h.update(b")")
    elif isinstance(obj, bytes):
        h.update(b"B"); h.update(len(obj).to_bytes(8, "little")); h.update(obj)
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        h.update(type(obj).__name__.encode()); h.update(repr(obj).encode())
    else:
        raise TypeError(f"Cannot fingerprint an object of type {type(obj).__name__}.")


class Matcher(ABC):
    """
    Incremental tracking of a token prefix against a constraint (the ``matcher`` capability).

    A matcher starts at the empty prefix. It is what an SMC sampler or a decoding loop uses as the hard
    support mask, so :meth:`allowed_next` must be SOUND: it may never exclude a token from which an
    accepted sequence is still reachable. (Masking with anything coarser destroys absolute continuity,
    and no importance weighting repairs that.) Exactness -- excluding every dead token -- is preferred
    but not required.
    """

    @property
    @abstractmethod
    def num_consumed(self) -> int:
        """Number of tokens consumed so far."""

    @abstractmethod
    def allowed_next(self, remaining: Optional[int] = None) -> torch.Tensor:
        """
        Tokens that may come next.

        :param remaining: if given, the sequence will have exactly ``remaining`` more tokens (counting the
            next one), and a token is allowed only if some continuation of that length is accepted.
            ``None`` means any length.
        :type remaining: Optional[int]

        :returns: a bool tensor of size [vocab_size]
        """

    @abstractmethod
    def advance(self, token: int) -> bool:
        """Consume ``token`` if ``allowed_next()`` (no length limit) allows it and return True;
        otherwise leave the state unchanged and return False."""

    @abstractmethod
    def is_accepting(self) -> bool:
        """Whether the prefix consumed so far is itself in the language."""

    @abstractmethod
    def rollback(self, num_tokens: int = 1):
        """Undo the last ``num_tokens`` advances."""

    @abstractmethod
    def clone(self) -> "Matcher":
        """An independent copy with the same state (e.g. for SMC particles)."""


class Constraint(ABC):
    """
    A set of token sequences over ``{0, ..., vocab_size - 1}``.

    Subclasses implement :meth:`accepts` and :meth:`_fingerprint_payload`, and any of the optional
    capabilities they support (see the module docstring). A leaf class advertises a capability simply
    by overriding its method; composition nodes compute theirs from their children.

    :param vocab_size: size of the token vocabulary the constraint reads
    :type vocab_size: int
    """

    def __init__(self, vocab_size: int):
        vocab_size = int(vocab_size)
        if vocab_size < 1:
            raise ValueError(f"`vocab_size` must be positive, got {vocab_size}.")
        self._vocab_size = vocab_size
        self._fingerprint = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ---------------------------------------------------------------------------------------------
    # Required interface
    # ---------------------------------------------------------------------------------------------

    @abstractmethod
    def accepts(self, tokens: Tokens) -> bool:
        """
        Whether a token sequence is in the language. This is the ground truth every other capability
        must agree with.

        :param tokens: a sequence of token ids (list, tuple or 1-D tensor)
        :type tokens: Union[Sequence[int], torch.Tensor]
        """

    @abstractmethod
    def _fingerprint_payload(self) -> Any:
        """A structural description of the constraint (ints, strings, bytes, tensors, nested
        tuples/lists and child constraints). Two constraints with equal payloads and the same class
        and vocabulary size are considered equal."""

    # ---------------------------------------------------------------------------------------------
    # Optional capabilities
    # ---------------------------------------------------------------------------------------------

    def capabilities(self) -> FrozenSet[str]:
        """The capabilities this constraint supports: always ``accepts``, plus every optional one whose
        method the class overrides. ``relax`` comes for free with ``automaton`` (an exact automaton is
        a valid over-approximation)."""
        caps = {"accepts"}
        for name in OPTIONAL_CAPABILITIES:
            if getattr(type(self), name) is not getattr(Constraint, name):
                caps.add(name)
        if "automaton" in caps:
            caps.add("relax")
        return frozenset(caps)

    def matcher(self) -> Matcher:
        """A fresh :class:`Matcher` at the empty prefix."""
        raise NotImplementedError(f"{type(self).__name__} does not support `matcher`.")

    def wmc(self, log_weights: torch.Tensor) -> torch.Tensor:
        """
        Weighted model count under fully factorised weights:
        ``log sum_{x in V^n, x in L} prod_t exp(log_weights[b, t, x_t])`` for every batch element ``b``.

        :param log_weights: log weights of size [B, n, vocab_size]
        :type log_weights: torch.Tensor

        :returns: a tensor of size [B]
        """
        raise NotImplementedError(f"{type(self).__name__} does not support `wmc`.")

    def sample(self, log_weights: torch.Tensor, num_samples: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Draw ``x ~ prod_t exp(log_weights[b, t, x_t]) * 1[x in L] / Z_b`` for every batch element ``b``.

        :param log_weights: log weights of size [B, n, vocab_size]
        :type log_weights: torch.Tensor

        :param num_samples: number of samples per batch element
        :type num_samples: int

        :returns: a tensor of token ids of size [B, num_samples, n]
        """
        raise NotImplementedError(f"{type(self).__name__} does not support `sample`.")

    def automaton(self):
        """The finite-state form exact lifting needs (states, transitions over token classes,
        accepting states). Lifting is exact as long as every accepted sequence has exactly one
        accepting run, so an unambiguous automaton suffices; a DFA is the common case."""
        raise NotImplementedError(f"{type(self).__name__} does not support `automaton`.")

    def relax(self, budget: Optional[int] = None) -> "Constraint":
        """
        A constraint with the ``automaton`` capability whose language CONTAINS this one (at most
        ``budget`` states, if given). Used as a proposal, never as a support mask. A constraint that
        already has an automaton returns itself.
        """
        if "automaton" in self.capabilities():
            return self
        raise NotImplementedError(f"{type(self).__name__} does not support `relax`.")

    # ---------------------------------------------------------------------------------------------
    # Lazy composition
    # ---------------------------------------------------------------------------------------------

    def __and__(self, other: "Constraint") -> "Constraint":
        return And(self, other)

    def __or__(self, other: "Constraint") -> "Constraint":
        return Or(self, other)

    def __invert__(self) -> "Constraint":
        return Not(self)

    def concat(self, other: "Constraint") -> "Constraint":
        """The concatenation: sequences ``u + v`` with ``u`` in this language and ``v`` in ``other``'s."""
        return Concat(self, other)

    # ---------------------------------------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------------------------------------

    def fingerprint(self) -> str:
        """A stable hex digest of the constraint's structure (same across processes)."""
        if self._fingerprint is None:
            h = hashlib.sha256()
            _hash_into(h, (type(self).__name__, self._vocab_size, self._fingerprint_payload()))
            self._fingerprint = h.hexdigest()
        return self._fingerprint

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constraint) and self.fingerprint() == other.fingerprint()

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def info(self) -> Dict[str, Any]:
        """A summary of the constraint (subclasses add their own fields)."""
        return dict(kind = type(self).__name__, vocab_size = self._vocab_size,
                    capabilities = sorted(self.capabilities()))

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in self.info().items() if k != "kind")
        return f"{type(self).__name__}({fields})"

    def _check_tokens(self, tokens: Tokens) -> Tuple[int, ...]:
        toks = _as_tokens(tokens)
        for t in toks:
            if not 0 <= t < self._vocab_size:
                raise ValueError(f"Token id {t} is outside the vocabulary [0, {self._vocab_size}).")
        return toks


# -------------------------------------------------------------------------------------------------
# Composition nodes
# -------------------------------------------------------------------------------------------------
#
# A composition node only advertises a capability once it is implemented for that node. What each
# can gain, given its children's capabilities:
#   automaton / relax: And, Or, Concat when every child has it (product / union / concatenation
#                      automata); Not when its child's automaton is deterministic.
#   matcher:           And, Or when every child has one (run the children's matchers side by side);
#                      Concat needs a set of split points; Not needs a deterministic child.
#   wmc / sample:      via the composite automaton when one exists; a mixed DFA & CFG needs the
#                      grammar-automaton intersection and is left out until something needs it.

class _Composite(Constraint):
    """Shared plumbing for composition nodes: vocabulary check, flattening, capabilities."""

    #: whether the operation is associative (nested nodes of the same type are flattened)
    _associative = True
    #: whether the operation is commutative (children are ordered by fingerprint for equality)
    _commutative = False

    def __init__(self, *children: Constraint):
        if len(children) < 2:
            raise ValueError(f"{type(self).__name__} needs at least two constraints, got {len(children)}.")
        for c in children:
            if not isinstance(c, Constraint):
                raise TypeError(f"{type(self).__name__} expects Constraint objects, got {type(c).__name__}.")
        vocab = children[0].vocab_size
        if any(c.vocab_size != vocab for c in children):
            raise ValueError(f"Cannot combine constraints over different vocabularies: "
                             f"{[c.vocab_size for c in children]}.")
        flat = []
        for c in children:
            if self._associative and type(c) is type(self):
                flat.extend(c.children)
            else:
                flat.append(c)
        super().__init__(vocab)
        self._children = tuple(flat)

    @property
    def children(self) -> Tuple[Constraint, ...]:
        return self._children

    def capabilities(self) -> FrozenSet[str]:
        # Membership composes for every node; the rest is added as each composition is implemented
        # (see the table above).
        return frozenset({"accepts"})

    def _fingerprint_payload(self):
        fps = [c.fingerprint() for c in self._children]
        return tuple(sorted(fps)) if self._commutative else tuple(fps)

    def info(self):
        info = super().info()
        info["children"] = [c.info()["kind"] for c in self._children]
        return info


class And(_Composite):
    """Intersection: sequences accepted by every child."""

    _commutative = True

    def accepts(self, tokens: Tokens) -> bool:
        toks = self._check_tokens(tokens)
        return all(c.accepts(toks) for c in self._children)



class Or(_Composite):
    """Union: sequences accepted by at least one child."""

    _commutative = True

    def accepts(self, tokens: Tokens) -> bool:
        toks = self._check_tokens(tokens)
        return any(c.accepts(toks) for c in self._children)



class Concat(_Composite):
    """Concatenation: sequences ``u_1 + u_2 + ... + u_k`` with ``u_i`` accepted by the i-th child."""

    def accepts(self, tokens: Tokens) -> bool:
        toks = self._check_tokens(tokens)
        return self._accepts_from(toks, 0)

    def _accepts_from(self, toks: Tuple[int, ...], i: int) -> bool:
        child = self._children[i]
        if i == len(self._children) - 1:
            return child.accepts(toks)
        return any(child.accepts(toks[:k]) and self._accepts_from(toks[k:], i + 1)
                   for k in range(len(toks) + 1))



class Not(Constraint):
    """Complement: sequences not accepted by the child (relative to all sequences over the
    vocabulary)."""

    def __init__(self, child: Constraint):
        if not isinstance(child, Constraint):
            raise TypeError(f"Not expects a Constraint, got {type(child).__name__}.")
        super().__init__(child.vocab_size)
        self._child = child

    @property
    def child(self) -> Constraint:
        return self._child

    def capabilities(self) -> FrozenSet[str]:
        # see the table above _Composite
        return frozenset({"accepts"})

    def accepts(self, tokens: Tokens) -> bool:
        return not self._child.accepts(self._check_tokens(tokens))

    def _fingerprint_payload(self):
        return (self._child,)

    def __invert__(self) -> Constraint:
        return self._child

    def info(self):
        info = super().info()
        info["child"] = self._child.info()["kind"]
        return info
