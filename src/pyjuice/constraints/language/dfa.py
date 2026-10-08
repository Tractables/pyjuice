"""
Deterministic finite automata over token classes.

A :class:`DFA` reads token ids. Most states treat most tokens identically, so transitions are stored
over TOKEN CLASSES rather than tokens:

* ``token_class[v]``  -- the class of token ``v`` (``int64[vocab_size]``, values in ``[0, C)``);
* ``delta[q, c]``     -- the successor of state ``q`` on any token of class ``c`` (``int64[K, C]``);
* ``initial``, ``accept[q]``.

Tokens share a class exactly when they lead to the same successor from every state (the constructor
merges identical columns of ``delta``), so the representation is as small as the automaton allows: a
keyword DFA has one class per keyword token plus "everything else". The DFA is complete: every
``(q, c)`` has a successor, with rejection represented by an explicit dead state.

The DFA has every optional capability of :class:`~pyjuice.constraints.language.base.Constraint`: an exact
:class:`DFAMatcher`, ``wmc`` and ``sample`` under factorised weights (a forward pass over positions),
and ``automaton`` (itself).

Two DFAs are ``==`` when their tables are identical. Use :meth:`DFA.minimize` (canonical: minimal,
with states and classes renumbered deterministically) or :meth:`DFA.equivalent` to compare languages.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .base import Constraint, Matcher, Tokens


class DFA(Constraint):
    """
    A complete DFA over token classes.

    :param vocab_size: size of the token vocabulary
    :type vocab_size: int

    :param token_class: class of every token, size [vocab_size]
    :type token_class: Union[torch.Tensor, Sequence[int]]

    :param delta: successor of every (state, class), size [num_states, num_classes]
    :type delta: Union[torch.Tensor, Sequence[Sequence[int]]]

    :param initial: the initial state
    :type initial: int

    :param accept: accepting states, as a bool mask of size [num_states] or a collection of state ids
    :type accept: Union[torch.Tensor, Iterable[int]]
    """

    def __init__(self, vocab_size: int, token_class, delta, initial: int, accept):
        super().__init__(vocab_size)
        token_class = torch.as_tensor(token_class, dtype = torch.long).cpu()
        delta = torch.as_tensor(delta, dtype = torch.long).cpu()
        if token_class.shape != (vocab_size,):
            raise ValueError(f"`token_class` must have shape [{vocab_size}], got {tuple(token_class.shape)}.")
        if delta.dim() != 2 or delta.size(0) < 1:
            raise ValueError(f"`delta` must be [num_states, num_classes], got {tuple(delta.shape)}.")
        K, C = delta.shape
        if token_class.min() < 0 or token_class.max() >= C:
            raise ValueError(f"`token_class` values must lie in [0, {C}).")
        if delta.min() < 0 or delta.max() >= K:
            raise ValueError(f"`delta` successors must lie in [0, {K}) (the DFA must be complete).")
        if not 0 <= int(initial) < K:
            raise ValueError(f"`initial` must lie in [0, {K}), got {initial}.")
        if isinstance(accept, torch.Tensor) and accept.dtype == torch.bool:
            accept = accept.cpu()
            if accept.shape != (K,):
                raise ValueError(f"`accept` mask must have shape [{K}], got {tuple(accept.shape)}.")
        else:
            ids = torch.as_tensor(sorted(set(int(q) for q in accept)), dtype = torch.long)
            if ids.numel() > 0 and (ids.min() < 0 or ids.max() >= K):
                raise ValueError(f"Accepting state ids must lie in [0, {K}).")
            mask = torch.zeros(K, dtype = torch.bool)
            mask[ids] = True
            accept = mask

        self._token_class, self._delta = _compress_classes(token_class, delta)
        self._initial = int(initial)
        self._accept = accept.clone()
        self._live = None              # [K] bool: an accepting state is reachable
        self._accept_in = None         # list: _accept_in[r][q] = acceptance reachable in exactly r steps
        self._device_tables = {}

    # ---------------------------------------------------------------------------------------------
    # Tables
    # ---------------------------------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._delta.size(0)

    @property
    def num_classes(self) -> int:
        return self._delta.size(1)

    @property
    def initial(self) -> int:
        return self._initial

    @property
    def accept(self) -> torch.Tensor:
        return self._accept.clone()

    @property
    def token_class(self) -> torch.Tensor:
        return self._token_class.clone()

    @property
    def delta(self) -> torch.Tensor:
        return self._delta.clone()

    def class_sizes(self) -> torch.Tensor:
        """Number of tokens in every class, size [num_classes]."""
        return torch.bincount(self._token_class, minlength = self.num_classes)

    def to_dense(self) -> torch.Tensor:
        """The token-level transition table, size [num_states, vocab_size] (for tests and inspection)."""
        return self._delta[:, self._token_class]

    def live_states(self) -> torch.Tensor:
        """Bool mask [num_states] of states from which some accepting state is reachable."""
        if self._live is None:
            live = self._accept.clone()
            while True:
                nxt = live | live[self._delta].any(dim = 1)
                if torch.equal(nxt, live):
                    break
                live = nxt
            self._live = live
        return self._live

    def accept_within(self, r: int) -> torch.Tensor:
        """Bool mask [num_states]: an accepting state is reachable in EXACTLY ``r`` more tokens."""
        if self._accept_in is None:
            self._accept_in = [self._accept.clone()]
        while len(self._accept_in) <= r:
            self._accept_in.append(self._accept_in[-1][self._delta].any(dim = 1))
        return self._accept_in[r]

    def _tables_on(self, device):
        device = torch.device(device)
        if device not in self._device_tables:
            self._device_tables[device] = (self._token_class.to(device), self._delta.to(device),
                                           self._accept.to(device))
        return self._device_tables[device]

    # ---------------------------------------------------------------------------------------------
    # Running the automaton
    # ---------------------------------------------------------------------------------------------

    def step(self, state: int, token: int) -> int:
        """The successor of ``state`` on ``token``."""
        return int(self._delta[state, self._token_class[token]])

    def run(self, tokens: Tokens, state: Optional[int] = None) -> int:
        """The state reached from ``state`` (default: the initial state) after reading ``tokens``."""
        q = self._initial if state is None else int(state)
        for t in self._check_tokens(tokens):
            q = int(self._delta[q, self._token_class[t]])
        return q

    def accepts(self, tokens: Tokens) -> bool:
        return bool(self._accept[self.run(tokens)])

    # ---------------------------------------------------------------------------------------------
    # Capabilities
    # ---------------------------------------------------------------------------------------------

    def matcher(self) -> "DFAMatcher":
        return DFAMatcher(self)

    def automaton(self) -> "DFA":
        return self

    def _class_log_weights(self, log_weights: torch.Tensor) -> torch.Tensor:
        """log sum of exp(log_weights) over the tokens of every class: [B, n, V] -> [B, n, C]."""
        token_class, _, _ = self._tables_on(log_weights.device)
        B, n, V = log_weights.shape
        m = torch.amax(log_weights, dim = 2, keepdim = True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        w = torch.exp(log_weights - m).reshape(B * n, V)
        out = torch.zeros(B * n, self.num_classes, dtype = w.dtype, device = w.device)
        out.index_add_(1, token_class, w)
        return torch.log(out).reshape(B, n, self.num_classes) + m

    def _check_weights(self, log_weights: torch.Tensor):
        if log_weights.dim() != 3 or log_weights.size(2) != self.vocab_size:
            raise ValueError(f"`log_weights` must be [B, n, {self.vocab_size}], got {tuple(log_weights.shape)}.")

    def _backward_messages(self, lwc: torch.Tensor) -> List[torch.Tensor]:
        """beta[t][b, q] = log sum over completions x_t..x_{n-1} from state q of their weight,
        restricted to completions ending in an accepting state; beta[n] = log 1[q accepting]."""
        _, delta, accept = self._tables_on(lwc.device)
        B, n, C = lwc.shape
        beta = [None] * (n + 1)
        beta[n] = torch.where(accept, 0.0, float("-inf")).to(lwc.dtype)[None, :].expand(B, -1)
        for t in range(n - 1, -1, -1):
            # [B, K, C]: take class c from state q, then continue from delta[q, c]
            terms = lwc[:, t, None, :] + beta[t + 1][:, delta]
            beta[t] = torch.logsumexp(terms, dim = 2)
        return beta

    def wmc(self, log_weights: torch.Tensor) -> torch.Tensor:
        self._check_weights(log_weights)
        beta = self._backward_messages(self._class_log_weights(log_weights))
        return beta[0][:, self._initial]

    def sample(self, log_weights: torch.Tensor, num_samples: int,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        self._check_weights(log_weights)
        token_class, delta, _ = self._tables_on(log_weights.device)
        B, n, V = log_weights.shape
        lwc = self._class_log_weights(log_weights)
        beta = self._backward_messages(lwc)
        if not torch.isfinite(beta[0][:, self._initial]).all():
            raise ValueError("No sequence of this length is accepted (the weighted model count is zero) "
                             "for at least one batch element.")
        S = num_samples
        rows = torch.arange(B, device = log_weights.device)[:, None].expand(B, S)
        q = torch.full((B, S), self._initial, dtype = torch.long, device = log_weights.device)
        out = torch.empty(B, S, n, dtype = torch.long, device = log_weights.device)
        for t in range(n):
            # class: p(c) ∝ exp(lwc[b, t, c] + beta[t+1][b, delta[q, c]])
            logits = lwc[rows, t] + beta[t + 1][rows[:, :, None], delta[q]]
            c = torch.multinomial(torch.softmax(logits.reshape(B * S, -1), dim = 1), 1,
                                  generator = generator).reshape(B, S)
            # token within the class: p(v | c) ∝ exp(log_weights[b, t, v]) over tokens of class c
            tl = log_weights[rows, t]
            tl = torch.where(token_class[None, None, :] == c[:, :, None], tl, float("-inf"))
            v = torch.multinomial(torch.softmax(tl.reshape(B * S, V), dim = 1), 1,
                                  generator = generator).reshape(B, S)
            out[:, :, t] = v
            q = delta[q, c]
        return out

    # ---------------------------------------------------------------------------------------------
    # Transformations
    # ---------------------------------------------------------------------------------------------

    def minimize(self) -> "DFA":
        """The minimal equivalent DFA, in canonical form: unreachable states removed, equivalent states
        merged (Moore partition refinement), states numbered in BFS order from the initial state over
        classes, and classes numbered by their smallest token. Two DFAs accept the same language iff
        their minimized forms are ``==``."""
        reach = self._reachable_order()
        delta = self._delta
        # Moore refinement over the reachable states
        idx = torch.as_tensor(reach, dtype = torch.long)
        remap = torch.full((self.num_states,), -1, dtype = torch.long)
        remap[idx] = torch.arange(len(reach))
        d = remap[delta[idx]]                                   # [R, C], all reachable by construction
        acc = self._accept[idx]
        part = acc.long()
        while True:
            sig = torch.cat([part[:, None], part[d]], dim = 1)
            _, new = torch.unique(sig, dim = 0, return_inverse = True)
            if int(new.max()) == int(part.max()) and _same_partition(new, part):
                break
            part = new
        num_blocks = int(part.max()) + 1
        rep = torch.full((num_blocks,), -1, dtype = torch.long)
        for s in range(len(reach) - 1, -1, -1):                 # any member represents its block
            rep[part[s]] = s
        block_delta = part[d[rep]]                              # [num_blocks, C]
        block_accept = acc[rep]
        merged = DFA(self.vocab_size, self._token_class, block_delta, int(part[0]), block_accept)
        return merged._renumbered()

    def equivalent(self, other: "DFA") -> bool:
        """Whether two DFAs accept the same language."""
        return self.minimize() == other.minimize()

    def _reachable_order(self) -> List[int]:
        order, seen, queue = [], {self._initial}, deque([self._initial])
        delta = self._delta.tolist()
        while queue:
            q = queue.popleft()
            order.append(q)
            for nxt in delta[q]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return order

    def _renumbered(self) -> "DFA":
        """Canonical numbering: states in BFS order (classes visited in increasing order)."""
        order = self._reachable_order()
        remap = torch.full((self.num_states,), -1, dtype = torch.long)
        remap[torch.as_tensor(order, dtype = torch.long)] = torch.arange(len(order))
        idx = torch.as_tensor(order, dtype = torch.long)
        return DFA(self.vocab_size, self._token_class, remap[self._delta[idx]], 0, self._accept[idx])

    # ---------------------------------------------------------------------------------------------
    # Builders
    # ---------------------------------------------------------------------------------------------

    @classmethod
    def anything(cls, vocab_size: int) -> "DFA":
        """Every sequence (one accepting state)."""
        return cls(vocab_size, torch.zeros(vocab_size, dtype = torch.long), [[0]], 0, [0])

    @classmethod
    def nothing(cls, vocab_size: int) -> "DFA":
        """No sequence (one rejecting state)."""
        return cls(vocab_size, torch.zeros(vocab_size, dtype = torch.long), [[0]], 0, [])

    @classmethod
    def from_transitions(cls, vocab_size: int, transitions: Mapping[int, Mapping[int, int]],
                         initial: int, accept: Iterable[int]) -> "DFA":
        """
        From a PARTIAL token-level transition map ``{state: {token: next_state}}`` with arbitrary
        integer state ids (the shape outlines-core's ``Index.get_transitions()`` returns). Missing
        (state, token) pairs go to an added dead state.
        """
        states = set(transitions) | {initial} | set(int(a) for a in accept)
        for outs in transitions.values():
            states.update(int(s) for s in outs.values())
        ids = {s: i for i, s in enumerate(sorted(states))}
        K = len(ids) + 1                                        # + dead state
        dead = K - 1
        dense = np.full((K, vocab_size), dead, dtype = np.int64)
        for s, outs in transitions.items():
            if outs:
                toks = np.fromiter((int(t) for t in outs.keys()), dtype = np.int64, count = len(outs))
                nxt = np.fromiter((ids[int(n)] for n in outs.values()), dtype = np.int64, count = len(outs))
                if toks.min() < 0 or toks.max() >= vocab_size:
                    raise ValueError(f"Token ids must lie in [0, {vocab_size}).")
                dense[ids[s], toks] = nxt
        return cls.from_dense(vocab_size, dense, ids[initial], [ids[int(a)] for a in accept])

    @classmethod
    def from_dense(cls, vocab_size: int, dense, initial: int, accept) -> "DFA":
        """From a complete token-level table ``dense[q, v]`` of size [num_states, vocab_size]."""
        dense = torch.as_tensor(dense, dtype = torch.long)
        cols, token_class = torch.unique(dense, dim = 1, return_inverse = True)
        return cls(vocab_size, token_class, cols, initial, accept)

    @classmethod
    def contains(cls, patterns: Sequence[Sequence[int]], vocab_size: int) -> "DFA":
        """
        Sequences containing at least one of ``patterns`` (each a sequence of token ids) as a
        contiguous substring. This is the semantics of Ctrl-G's ``AhoCorasickBuilder`` (an Aho-Corasick
        automaton whose accepting states absorb).
        """
        patterns = [tuple(int(t) for t in p) for p in patterns]
        if not patterns or any(len(p) == 0 for p in patterns):
            raise ValueError("`patterns` must be a non-empty list of non-empty token sequences.")
        alphabet = sorted({t for p in patterns for t in p})
        if alphabet[0] < 0 or alphabet[-1] >= vocab_size:
            raise ValueError(f"Pattern tokens must lie in [0, {vocab_size}).")
        # trie
        goto: List[Dict[int, int]] = [{}]
        final = [False]
        for p in patterns:
            s = 0
            for t in p:
                if t not in goto[s]:
                    goto.append({}); final.append(False)
                    goto[s][t] = len(goto) - 1
                s = goto[s][t]
            final[s] = True
        # failure links and completed transitions (BFS); a node is final if any suffix is a pattern
        fail = [0] * len(goto)
        delta = [[0] * (len(alphabet) + 1) for _ in goto]       # last class = every other token
        queue = deque()
        for ci, t in enumerate(alphabet):
            if t in goto[0]:
                fail[goto[0][t]] = 0
                queue.append(goto[0][t])
                delta[0][ci] = goto[0][t]
        while queue:
            s = queue.popleft()
            final[s] = final[s] or final[fail[s]]
            for ci, t in enumerate(alphabet):
                if t in goto[s]:
                    c = goto[s][t]
                    fail[c] = delta[fail[s]][ci]
                    queue.append(c)
                    delta[s][ci] = c
                else:
                    delta[s][ci] = delta[fail[s]][ci]
        # accepting states absorb (once a pattern has occurred, the sequence is accepted)
        for s in range(len(goto)):
            if final[s]:
                delta[s] = [s] * (len(alphabet) + 1)
        token_class = torch.full((vocab_size,), len(alphabet), dtype = torch.long)
        token_class[torch.as_tensor(alphabet)] = torch.arange(len(alphabet))
        accept = [s for s in range(len(goto)) if final[s]]
        return cls(vocab_size, token_class, delta, 0, accept).minimize()

    @classmethod
    def word_count(cls, lo: int, hi: int, token_kind) -> "DFA":
        """
        Sequences containing between ``lo`` and ``hi`` words (inclusive), where word boundaries are
        read off a per-token kind (see :func:`pyjuice.constraints.language.text.word_token_kinds`)::

            token_kind[v] = 2 * (v starts with a separator) + (v contains a letter or digit)

        A word starts at a token with a letter or digit that either begins with a separator or follows
        a separator. This is the definition of Ctrl-G's ``WordCountBuilder``; the sequence is treated as
        if preceded by a separator.

        :param token_kind: int tensor of size [vocab_size] with values in {0, 1, 2, 3}
        """
        token_kind = torch.as_tensor(token_kind, dtype = torch.long)
        if not 0 <= lo <= hi:
            raise ValueError(f"Need 0 <= lo <= hi, got lo={lo}, hi={hi}.")
        if token_kind.dim() != 1 or token_kind.min() < 0 or token_kind.max() > 3:
            raise ValueError("`token_kind` must be a 1-D tensor with values in {0, 1, 2, 3}.")
        # state 2k + s: k words so far, s = 1 if the previous token ended at a separator; 2(hi+1) = too many
        sink = 2 * (hi + 1)
        word = lambda k: 2 * (k + 1) if k + 1 <= hi else sink
        delta = []
        for k in range(hi + 1):
            # kinds: 0 = no sep / no word char, 1 = no sep / word char, 2 = sep / no word char, 3 = sep / word char
            delta.append([2 * k, 2 * k, 2 * k + 1, word(k)])            # s = 0: continue unless a separator
            delta.append([2 * k + 1, word(k), 2 * k + 1, word(k)])      # s = 1: any word char starts a word
        delta.append([sink] * 4)
        accept = [2 * k + s for k in range(lo, hi + 1) for s in (0, 1)]
        return cls(token_kind.numel(), token_kind, delta, 1, accept).minimize()

    @classmethod
    def from_ctrlg(cls, graph: Mapping, vocab_size: int) -> "DFA":
        """
        From a Ctrl-G DFA graph ``{"edges": [(u, v, token_bitset), ...], "initial_state": ...,
        "accept_states": ...}`` (state ids may be any hashable values; each bitset is a bool array of
        size [vocab_size]). The graph must be complete, as Ctrl-G's minimised graphs are.
        """
        states = {}
        for u, v, _ in graph["edges"]:
            states.setdefault(u, len(states)); states.setdefault(v, len(states))
        states.setdefault(graph["initial_state"], len(states))
        K = len(states)
        dense = np.full((K, vocab_size), -1, dtype = np.int64)
        for u, v, bits in graph["edges"]:
            bits = np.asarray(bits, dtype = bool)
            if bits.shape != (vocab_size,):
                raise ValueError(f"Edge bitsets must have shape [{vocab_size}], got {bits.shape}.")
            if (dense[states[u], bits] >= 0).any():
                raise ValueError(f"Ctrl-G graph is not deterministic at state {u!r}.")
            dense[states[u], bits] = states[v]
        if (dense < 0).any():
            raise ValueError("Ctrl-G graph is not complete (some state has no successor for some token).")
        accept = [states[a] for a in graph["accept_states"] if a in states]
        return cls.from_dense(vocab_size, dense, states[graph["initial_state"]], accept)

    # ---------------------------------------------------------------------------------------------
    # Identity
    # ---------------------------------------------------------------------------------------------

    def _fingerprint_payload(self):
        return (self._initial, self._accept, self._token_class, self._delta)

    def info(self):
        info = super().info()
        info.update(num_states = self.num_states, num_classes = self.num_classes)
        return info


class DFAMatcher(Matcher):
    """An exact matcher for a :class:`DFA`: the state is the DFA state after the consumed prefix."""

    def __init__(self, dfa: DFA, states: Optional[List[int]] = None):
        self._dfa = dfa
        self._states = [dfa.initial] if states is None else list(states)

    @property
    def state(self) -> int:
        return self._states[-1]

    @property
    def num_consumed(self) -> int:
        return len(self._states) - 1

    def allowed_next(self, remaining: Optional[int] = None) -> torch.Tensor:
        dfa = self._dfa
        nxt = dfa._delta[self.state]                            # [C]
        if remaining is None:
            ok = dfa.live_states()[nxt]
        else:
            if remaining < 1:
                return torch.zeros(dfa.vocab_size, dtype = torch.bool)
            ok = dfa.accept_within(remaining - 1)[nxt]
        return ok[dfa._token_class]

    def advance(self, token: int) -> bool:
        token = int(token)
        if not 0 <= token < self._dfa.vocab_size:
            raise ValueError(f"Token id {token} is outside the vocabulary [0, {self._dfa.vocab_size}).")
        nxt = self._dfa.step(self.state, token)
        if not bool(self._dfa.live_states()[nxt]):
            return False
        self._states.append(nxt)
        return True

    def is_accepting(self) -> bool:
        return bool(self._dfa._accept[self.state])

    def rollback(self, num_tokens: int = 1):
        if not 0 <= num_tokens <= self.num_consumed:
            raise ValueError(f"Cannot roll back {num_tokens} tokens; {self.num_consumed} consumed.")
        if num_tokens:
            del self._states[-num_tokens:]

    def clone(self) -> "DFAMatcher":
        return DFAMatcher(self._dfa, self._states)


# -------------------------------------------------------------------------------------------------
# Composition of DFAs
# -------------------------------------------------------------------------------------------------

def _joint_classes(dfas: Sequence[DFA]):
    """The coarsest common refinement of several DFAs' token classes. Returns ``token_class`` [V] for
    the joint classes and ``child_class`` [C_joint, k]: the class each joint class has in each DFA."""
    vocab = dfas[0].vocab_size
    if any(d.vocab_size != vocab for d in dfas):
        raise ValueError("Cannot combine DFAs over different vocabularies.")
    sizes = [d.num_classes for d in dfas]
    if float(np.prod(sizes, dtype = np.float64)) <= 2 ** 24:
        # mixed-radix key per token, then a counting pass: O(V + prod(C_i)), no sort over V
        key = torch.zeros(vocab, dtype = torch.long)
        for d, c in zip(dfas, sizes):
            key = key * c + d._token_class
        present = torch.bincount(key, minlength = int(np.prod(sizes))) > 0
        keys = torch.nonzero(present).flatten()                            # sorted joint keys
        remap = torch.cumsum(present.long(), dim = 0) - 1
        token_class = remap[key]
        child_class = torch.empty(keys.numel(), len(dfas), dtype = torch.long)
        rest = keys.clone()
        for i in range(len(dfas) - 1, -1, -1):
            child_class[:, i] = rest % sizes[i]
            rest = rest // sizes[i]
        return token_class, child_class
    stacked = torch.stack([d._token_class for d in dfas], dim = 1)          # [V, k]
    child_class, token_class = torch.unique(stacked, dim = 0, return_inverse = True)
    return token_class, child_class


def product(dfas: Sequence[DFA], mode: str) -> DFA:
    """
    The product automaton of ``dfas`` over their joint token classes, built over reachable state
    tuples only, then minimised. ``mode`` is ``"and"`` (intersection: every DFA accepts) or ``"or"``
    (union: at least one accepts).
    """
    if mode not in ("and", "or"):
        raise ValueError(f"`mode` must be 'and' or 'or', got {mode!r}.")
    token_class, child_class = _joint_classes(dfas)
    deltas = [d._delta.tolist() for d in dfas]
    accepts = [d._accept.tolist() for d in dfas]
    cc = child_class.tolist()                                               # [C_joint][k]
    start = tuple(d.initial for d in dfas)
    ids, order, queue = {start: 0}, [start], deque([start])
    rows = []
    while queue:
        s = queue.popleft()
        row = []
        for joint in cc:
            nxt = tuple(deltas[i][s[i]][joint[i]] for i in range(len(dfas)))
            if nxt not in ids:
                ids[nxt] = len(order); order.append(nxt); queue.append(nxt)
            row.append(ids[nxt])
        rows.append(row)
    combine = all if mode == "and" else any
    accept = [i for i, s in enumerate(order) if combine(accepts[k][s[k]] for k in range(len(dfas)))]
    return DFA(dfas[0].vocab_size, token_class, rows, 0, accept).minimize()


def complement(dfa: DFA) -> DFA:
    """Sequences the DFA rejects (a complete DFA: flip the accepting states)."""
    return DFA(dfa.vocab_size, dfa._token_class, dfa._delta, dfa.initial, ~dfa._accept).minimize()


def concatenate(dfas: Sequence[DFA]) -> DFA:
    """
    Sequences ``u_1 + ... + u_k`` with ``u_i`` accepted by the i-th DFA, by subset construction (folded
    left). A state is (state of the left part, set of live states of the right part that some split
    point has reached).
    """
    out = dfas[0]
    for right in dfas[1:]:
        out = _concatenate_pair(out, right)
    return out


def _concatenate_pair(a: DFA, b: DFA) -> DFA:
    token_class, child_class = _joint_classes([a, b])
    da, db = a._delta.tolist(), b._delta.tolist()
    acc_a, acc_b = a._accept.tolist(), b._accept.tolist()
    live_b = b.live_states().tolist()
    cc = child_class.tolist()

    def enter(qa, bset):
        # every time the left part accepts, the right part may start
        return frozenset(bset | {b.initial}) if acc_a[qa] and live_b[b.initial] else frozenset(bset)

    start = (a.initial, enter(a.initial, set()))
    ids, order, queue = {start: 0}, [start], deque([start])
    rows = []
    while queue:
        qa, bset = queue.popleft()
        row = []
        for ca, cb in cc:
            na = da[qa][ca]
            nb = {db[s][cb] for s in bset}
            nb = {s for s in nb if live_b[s]}
            nxt = (na, enter(na, nb))
            if nxt not in ids:
                ids[nxt] = len(order); order.append(nxt); queue.append(nxt)
            row.append(ids[nxt])
        rows.append(row)
    accept = [i for i, (_, bset) in enumerate(order) if any(acc_b[s] for s in bset)]
    return DFA(a.vocab_size, token_class, rows, 0, accept).minimize()


def _compress_classes(token_class: torch.Tensor, delta: torch.Tensor):
    """Merge classes with identical successor columns, drop classes no token belongs to, and number
    the remaining classes by their smallest token (a canonical form for a given transition table)."""
    used = torch.nonzero(torch.bincount(token_class, minlength = delta.size(1))).flatten()   # O(V), no sort
    cols, inv = torch.unique(delta[:, used], dim = 1, return_inverse = True)   # merge identical columns
    old_to_new = torch.full((delta.size(1),), -1, dtype = torch.long)
    old_to_new[used] = inv
    tc = old_to_new[token_class]
    # number classes by first (smallest) token
    C = cols.size(1)
    first = torch.full((C,), token_class.numel(), dtype = torch.long)
    first.scatter_reduce_(0, tc, torch.arange(tc.numel()), reduce = "amin")
    order = torch.argsort(first)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(C)
    return rank[tc].contiguous(), cols[:, order].contiguous()


def _same_partition(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two labelings induce the same partition (equal number of blocks and a consistent map)."""
    pairs = torch.unique(torch.stack([a, b], dim = 1), dim = 0)
    return pairs.size(0) == int(a.max()) + 1 == int(b.max()) + 1
