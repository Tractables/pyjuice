"""
Strong correctness checks of the dfa_exact backend on Codpula circuits (a GeneralizedHMM with SoftEvidenceCategorical
inputs under top-k or dense soft evidence, and BlockScale external parameters) and PD circuits: every output of a
decoder session against an independent float64 reference, and invariants that need no reference.

THE REFERENCE (:class:`Float64Reference`) computes ``log p(C, e)`` and the conditionals of every missing token from the
definitions alone -- the automaton's own ``token_class``, ``delta``, ``initial`` and ``accept``, the PC's structure and
parameters (read from ``pc.params`` and its input layers), and the inputs' semantics (soft evidence: a node's token
probabilities times the evidence's weights, over its candidates under top-k, normalized per node; external
parameters: a gated sum's weights times its gates, normalized per node) -- and none of the backend's own tables (its
layout and columns, token classes, liveness, plans, kernels). In float64 throughout.

Every node over the variables ``[a, b)`` holds the block ``V[i, j]``: the log of its mass over the strings
``x_a .. x_(b-1)`` that take the automaton from state ``i`` to state ``j``, its rows the states reachable at ``a`` and
able to accept from there, its columns the same at ``b`` -- the initial state alone at ``a = 0``, and at ``b = n`` one
column, every accepting state's sum. An input node is a block only when it must be: a product applies it to the block
after it, ``sum_c mass(c) R[delta(i, c), j]``, so an HMM's blocks are vectors. The outside pass gives every node its
outside block, and every input node its outside per (state, class) -- a missing token's conditional, from the input
nodes over its position.

THE INVARIANTS (:func:`check_invariants`): a session's outputs do not change with the samples' order, nor when a
sample runs alone; dense and compact storage, and pruning by the inputs' classes on and off, give the same values;
every conditional normalizes; the chain rule ties a marginal, a conditional and the marginal of the evidence one token
longer. (The CUDA and Triton sums against each other: ``cuda_sum_test.py``.)

TIERS: the default tests (seconds) on small circuits; ``@pytest.mark.slow`` at Codpula's shapes and PD16's; ``python
strong_check_test.py --help`` for the full shapes (4096 latents).
TOLERANCES (log space, :data:`TOLERANCE`): fp32 ``1e-4``; TF32 ``2.5e-3`` for the marginals and ``5e-3`` for the
conditionals; bf16 ``1.5e-2`` and ``5e-3``; -inf exactly where the reference has it.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import BlockScaleSumParams
from pyjuice.nodes.distributions import Categorical, SoftEvidenceCategorical

#: the largest |difference| from the reference (log space) per precision, of the marginals (and the conditional rows'
#: log marginals) and of the conditionals. TF32 rounds both operands of every product to the nearest TF32 value (2^-11
#: relative at most): the marginals err upward where a sum's children are nearly equal -- a chunk's maximum is 1, the
#: terms just below it round up to it -- by up to ~1.5e-3 over Codpula's and PD16's depth; the conditionals, normalized,
#: by unbiased noise, up to 3.5e-3 on a small PD (sums of few children: little to average over). bf16 rounds the CUDA
#: fused sums' operands to 2^-9: the marginals err up to ~1e-2 at Codpula's shapes (the same upward rounding near a
#: chunk's maximum), the conditionals as at TF32 -- the error cancels in their normalization
TOLERANCE = {"fp32": dict(marginal = 1e-4, conditional = 1e-4), "tf32": dict(marginal = 2.5e-3, conditional = 5e-3),
             "bf16": dict(marginal = 1.5e-2, conditional = 5e-3)}
DEV = torch.device("cuda:0")
F64 = torch.float64
NEG = -float("inf")
#: the reference's work arrays: at most this many floats at once (operators and reductions run in node chunks)
CHUNK = 1 << 25


# -------------------------------------------------------------------------------------------------
# Log-space helpers (max-shifted; an all -inf slice stays -inf, never NaN)
# -------------------------------------------------------------------------------------------------

def _shift(x, dim):
    m = x.amax(dim = dim, keepdim = True)
    return torch.where(torch.isfinite(m), m, torch.zeros_like(m))


def _logsumexp(x, dim):
    m = _shift(x, dim)
    return (torch.log(torch.exp(x - m).sum(dim = dim, keepdim = True)) + m).squeeze(dim)


def _log_matmul(A, B):
    """log(exp(A) @ exp(B)) over the last two dims."""
    sa, sb = _shift(A, -1), _shift(B, -2)
    return torch.log(torch.exp(A - sa) @ torch.exp(B - sb)) + sa + sb


def _logaddexp_into(cur, idx, x):
    """``cur[idx] = log(exp(cur[idx]) + exp(x))``, rows of ``idx`` repeated or not (in place, returns ``cur``)."""
    if idx is None:
        return torch.logaddexp(cur, x, out = cur)
    if torch.unique(idx).numel() == idx.numel():
        cur[idx] = torch.logaddexp(cur[idx], x)
        return cur
    s = _shift(torch.cat([cur[idx], x], dim = 0), 0)                          # (a shift over every row involved)
    acc = torch.exp(cur - s)
    acc.index_add_(0, idx, torch.exp(x - s))
    return cur.copy_(torch.log(acc) + s)


def _node_chunks(N, per_node):
    step = max(1, CHUNK // max(1, per_node))
    return [slice(i, min(N, i + step)) for i in range(0, N, step)]


# -------------------------------------------------------------------------------------------------
# The float64 reference
# -------------------------------------------------------------------------------------------------

class Float64Reference:
    """
    ``log p(C, e)`` and the missing tokens' conditionals of ``pc`` under the automaton ``dfa`` over ``n`` positions,
    in float64, from the definitions (see the module docstring).

    :ivar width: per boundary ``t`` (0..n), how many automaton states a block's rows (at ``t``) or columns are over --
        those reachable from the initial state in ``t`` tokens and able to accept in ``n - t``; boundary ``n``: one
    :ivar nxt: per position ``t``, long [rows at t, classes]: the row at ``t + 1`` class ``c`` takes row ``i`` to (at
        ``t = n - 1``: 0 if it accepts), -1 where it leads out
    """

    def __init__(self, pc, dfa, n: int, device = DEV):
        self.pc, self.dfa, self.n, self.dev = pc, dfa, n, device
        delta = dfa.delta.cpu().long()                                        # [K, C]
        K, C = delta.shape
        self.C = C
        self.token_class = dfa.token_class.to(device).long()                  # [V]
        accept = dfa.accept.cpu().bool()
        reach = [torch.zeros(K, dtype = torch.bool) for _ in range(n + 1)]
        reach[0][int(dfa.initial)] = True
        for t in range(n):
            reach[t + 1][delta[reach[t]].reshape(-1)] = True
        able = [torch.zeros(K, dtype = torch.bool) for _ in range(n + 1)]
        able[n] = accept.clone()
        for t in range(n - 1, -1, -1):
            able[t] = able[t + 1][delta].any(dim = 1)
        live = [reach[t] & able[t] for t in range(n + 1)]
        self.satisfiable = bool(live[0].any())
        states = [torch.nonzero(live[t]).view(-1) for t in range(n)]
        self.width = [int(s.numel()) for s in states] + [1]
        self.nxt = []
        for t in range(n):
            if t + 1 == n:
                nx = torch.where(accept[delta[states[t]]], 0, -1)
            else:
                at = torch.full((K,), -1, dtype = torch.long)
                at[states[t + 1]] = torch.arange(self.width[t + 1])
                nx = at[delta[states[t]]]
            self.nxt.append(nx.to(device))
        self.order = list(pc.root_ns)                                         # children before parents
        self.scope = {}
        for ns in self.order:
            v = sorted(ns.scope.to_list())
            assert v == list(range(v[0], v[-1] + 1)), "the reference needs every node group's scope contiguous"
            self.scope[ns] = (v[0], v[-1] + 1)
        assert self.order[-1].num_nodes == 1, "the reference needs one root node"

    # ---------------------------------------------------------------------------------------------
    # The PC's parameters and the inputs
    # ---------------------------------------------------------------------------------------------

    def _layer_of(self, ns):
        src = ns.get_source_ns()
        return next(layer for layer in self.pc.input_layer_group if src in layer.nodes)

    def token_probs(self, ns):
        """float64 [nodes, V]: an input node group's token probabilities (a Categorical's table; soft evidence's)."""
        if not isinstance(ns.dist, (Categorical, SoftEvidenceCategorical)):
            raise NotImplementedError(f"the reference reads Categorical inputs, got {type(ns.dist).__name__}")
        src = ns.get_source_ns()
        ps, pe = src._param_range
        return self._layer_of(ns).params[ps:pe].to(self.dev, F64).reshape(ns.num_nodes, ns.dist.num_cats)

    def sum_weights(self, ns):
        """float64 [nodes, children's nodes]: a sum node group's weights from ``pc.params`` (an edge block listed
        twice counts twice, as the forward adds both)."""
        src = ns.get_source_ns()
        bs, cbs = ns.block_size, ns.ch_block_size
        psid, peid = src._param_range
        local = ((src._param_ids - psid) // (bs * cbs)).to(self.dev)
        blocks = self.pc.params[psid:peid].to(self.dev, F64).reshape(-1, cbs, bs)[local].permute(0, 2, 1)
        num_ch_blocks = sum(cs.num_node_blocks for cs in ns.chs)
        dense = torch.zeros(ns.num_node_blocks * num_ch_blocks, bs, cbs, dtype = F64, device = self.dev)
        e = ns.edge_ids.to(self.dev)
        dense.index_put_((e[0] * num_ch_blocks + e[1],), blocks, accumulate = True)
        return dense.reshape(ns.num_node_blocks, num_ch_blocks, bs, cbs).permute(0, 2, 1, 3).reshape(
            ns.num_nodes, num_ch_blocks * cbs)

    def weights(self, ns, B):
        """float64 [B or 1, nodes, children's nodes]: a sum node group's weights, under external parameters sample
        ``b``'s weights times its gates ``phi_b[n // node gate, c // child gate]``, normalized per node."""
        W = self.sum_weights(ns)
        ext = getattr(ns, "external_params", None)
        if self.gates is None or ext is None:
            return W[None]
        lphi = self.gates.get(ns, self.gates.get(ext.storage_owner(ns)))
        if lphi is None:
            return W[None]
        lphi = (lphi[0] if isinstance(lphi, (tuple, list)) else lphi).to(self.dev, F64)
        if not isinstance(ext, BlockScaleSumParams):
            raise NotImplementedError(f"the reference reads BlockScale external parameters, got {type(ext).__name__}")
        ng, cg = ext.gate_sizes(ns)
        rows = torch.arange(ns.num_nodes, device = self.dev) // ng
        cols = torch.arange(ns.num_ch_nodes, device = self.dev) // cg
        g = torch.exp(lphi - lphi.amax(dim = (1, 2), keepdim = True))[:, rows][:, :, cols]
        Wb = W[None] * g
        return Wb / Wb.sum(dim = 2, keepdim = True)

    def product_child_index(self, ns, k):
        """long [nodes]: the node of child ``k`` every product node multiplies."""
        if ns.is_block_sparse():
            cs = ns.chs[k]
            off = torch.arange(ns.num_nodes) % ns.block_size
            blocks = ns.edge_ids[torch.arange(ns.num_nodes) // ns.block_size, k].cpu()
            return (blocks * cs.block_size + off).to(self.dev)
        return ns.edge_ids[:, k].to(self.dev).long()

    def emission(self, ns, B):
        """``(q, pw)``: float64 [nodes, B, V] every sample's token probabilities at an input node group -- its own,
        or under soft evidence ``q(v) = p(v) w(v) / Z`` (top-k: ``w`` 0 off the candidates) -- and, with soft
        evidence, the unnormalized ``p(v) w(v)`` (None without)."""
        p = self.token_probs(ns)
        ev = None if self.inputs is None else self.inputs.get("categorical_evidence_logp")
        if not isinstance(ns.dist, SoftEvidenceCategorical) or ev is None:
            return p[:, None, :].expand(-1, B, -1), None
        lv = int(self._layer_of(ns).var_idmapping[ns.scope.to_list()[0]])
        e = ev[:, lv].to(self.dev, F64).expand(B, -1)
        w = torch.zeros(B, ns.dist.num_cats, dtype = F64, device = self.dev)
        ids = self.inputs.get("soft_evidence_cat_ids")
        if ids is None:
            w[:, :e.size(1)] = torch.exp(e)
        else:
            w.scatter_(1, ids[:, lv].to(self.dev).long().expand(B, -1), torch.exp(e))
        pw = p[:, None, :] * w[None]
        return pw / pw.sum(dim = 2, keepdim = True), pw

    def value_mask(self, ns, B):
        """bool [B] (host): soft evidence's value mask at the node group's variable, or None."""
        vm = None if self.inputs is None else self.inputs.get("soft_evidence_value_mask")
        if vm is None or not isinstance(ns.dist, SoftEvidenceCategorical):
            return None
        lv = int(self._layer_of(ns).var_idmapping[ns.scope.to_list()[0]])
        return vm[:, lv].cpu().bool().expand(B)

    def class_masses(self, ns, data, missing):
        """float64 [nodes, B, classes], log: a missing token's mass per class, an observed one's at its class alone
        (value mask False: the position free, ``p w`` per class; True and observed: ``p(d) w(d)``)."""
        B = data.size(0)
        t = ns.scope.to_list()[0]
        q, pw = self.emission(ns, B)
        vm = self.value_mask(ns, B)
        free = missing[:, t].clone()
        if vm is not None:
            free |= ~vm
        src = q if vm is None else torch.where((~vm).to(self.dev)[None, :, None], pw, q)
        mass = torch.zeros(ns.num_nodes, B, self.C, dtype = F64, device = self.dev)
        mass.index_add_(2, self.token_class, src)
        for b in torch.nonzero(~free).view(-1).tolist():                     # an observed token: its class alone
            x = int(data[b, t])
            val = (pw if vm is not None and bool(vm[b]) else q)[:, b, x]
            mass[:, b] = 0.0
            mass[:, b, int(self.token_class[x])] = val
        return torch.log(mass)

    # ---------------------------------------------------------------------------------------------
    # Blocks and operators
    # ---------------------------------------------------------------------------------------------

    def input_block(self, lm, t):
        """[nodes, B, rows at t, columns at t + 1], log: an input node group's block from its class masses ``lm``."""
        N, B, C = lm.shape
        nx = self.nxt[t]
        out = torch.zeros(N, B, self.width[t], self.width[t + 1], dtype = F64, device = self.dev)
        s = _shift(lm, 2)
        p = torch.exp(lm - s)
        for c in range(C):
            ii = torch.nonzero(nx[:, c] >= 0).view(-1)
            out[:, :, ii, nx[ii, c]] += p[:, :, c:c + 1]
        return torch.log(out) + s[..., None]

    def apply_input(self, lm, t, R):
        """``log sum_c mass(c) R[delta(i, c), j]``: an input node's masses ``lm`` [nodes, B, C] applied to the block
        after it ``R`` [nodes, B, rows at t + 1, columns] (nodes paired) -> [nodes, B, rows at t, columns]."""
        nx = self.nxt[t]
        N, B, _, J = R.shape
        Wi, C = nx.shape
        ok = (nx >= 0)[None, None, :, :, None]
        out = torch.empty(N, B, Wi, J, dtype = F64, device = self.dev)
        for sl in _node_chunks(N, B * Wi * C * J):
            g = torch.where(ok, R[sl][:, :, nx.clamp(min = 0)], NEG)         # [n, B, Wi, C, J]
            out[sl] = _logsumexp(g + lm[sl][:, :, None, :, None], dim = 3)
        return out

    def _input_op_outside(self, M, lm, t, R):
        """A product's input ``u`` (masses ``lm``) applied to the block ``R`` after it, the pair's outside ``M``
        [N, B, rows at t, columns]: ``(u's outside per (state, class) [N, B, rows at t, C], R's outside [N, B, rows
        at t + 1, columns])``."""
        nx = self.nxt[t]
        N, B, Wi, J = M.shape
        C, Wn = nx.size(1), self.width[t + 1]
        ok = (nx >= 0)[None, None, :, :, None]
        ou = torch.empty(N, B, Wi, C, dtype = F64, device = self.dev)
        orr = torch.empty(N, B, Wn, J, dtype = F64, device = self.dev)
        tgt = nx.clamp(min = 0).reshape(-1)
        for sl in _node_chunks(N, B * Wi * C * J):
            g = torch.where(ok, R[sl][:, :, nx.clamp(min = 0)], NEG)         # [n, B, Wi, C, J]
            ou[sl] = _logsumexp(M[sl][:, :, :, None, :] + g, dim = 4)
            x = torch.where(ok, M[sl][:, :, :, None, :] + lm[sl][:, :, None, :, None], NEG)
            flat = x.reshape(x.size(0), B, Wi * C, J)
            s = _shift(flat, 2)
            acc = torch.zeros(x.size(0), B, Wn, J, dtype = F64, device = self.dev)
            acc.index_add_(2, tgt, torch.exp(flat - s))
            orr[sl] = torch.log(acc) + s
        return ou, orr

    def _input_block_outside(self, M, t):
        """An input read as a block, its outside block ``M`` [N, B, rows at t, columns at t + 1] per (state, class):
        ``M[i, delta(i, c)]``, -inf where class ``c`` leads out."""
        nx = self.nxt[t]
        g = M[:, :, torch.arange(nx.size(0), device = self.dev)[:, None], nx.clamp(min = 0)]
        return torch.where((nx >= 0)[None, None], g, NEG)

    def _parts(self, ns):
        """A product's children in scope order as parts: ``("op", (k, k + 1))`` an input then a block (applied as an
        operator), ``("blk", (k,))`` a block (an input read as one)."""
        order = sorted(range(len(ns.chs)), key = lambda k: self.scope[ns.chs[k]][0])
        parts, k = [], 0
        while k < len(order):
            if ns.chs[order[k]].is_input() and k + 1 < len(order) and not ns.chs[order[k + 1]].is_input():
                parts.append(("op", (order[k], order[k + 1])))
                k += 2
            else:
                parts.append(("blk", (order[k],)))
                k += 1
        return parts

    def _block_of(self, cs):
        if self.vals[cs] is None:                                            # an input group read as a block
            self.vals[cs] = self.input_block(self.masses[cs], self.scope[cs][0])
        return self.vals[cs]

    def _part_value(self, ns, kind, ks):
        idx = [self.product_child_index(ns, k) for k in ks]
        if kind == "op":
            u, r = ns.chs[ks[0]], ns.chs[ks[1]]
            return self.apply_input(self.masses[u][idx[0]], self.scope[u][0], self._block_of(r)[idx[1]])
        return self._block_of(ns.chs[ks[0]])[idx[0]]

    # ---------------------------------------------------------------------------------------------
    # The inside pass
    # ---------------------------------------------------------------------------------------------

    def inside(self, data, missing, inputs = None):
        """``log p(C, e)`` [B] for the tokens ``data`` [B, n] (where ``missing`` [B, n] is False); every node group's
        inside block kept for :meth:`conditionals`."""
        self.data, self.missing = data.cpu().long(), missing.cpu().bool()
        self.inputs = inputs
        B = data.size(0)
        gates = None if inputs is None else inputs.get("sum_external_params")
        if gates is not None and any(isinstance(k, str) for k in gates):
            first = next(iter(gates.values()))
            gates_B = (first[0] if isinstance(first, tuple) else first).size(0)
            gates = self.pc._expand_external_params_groups(dict(gates), gates_B)
        self.gates = gates
        self.vals, self.masses = {}, {}
        for ns in self.order:
            if ns.is_input():
                self.masses[ns] = self.class_masses(ns, self.data, self.missing)
                self.vals[ns] = None
            elif ns.is_prod():
                v = None
                for kind, ks in self._parts(ns):
                    part = self._part_value(ns, kind, ks)
                    v = part if v is None else _log_matmul(v, part)
                self.vals[ns] = v
            else:
                X = torch.cat([self._block_of(cs) for cs in ns.chs], dim = 0)   # [K, B, I, J]
                Wb = self.weights(ns, B)
                s = _shift(X, 0)
                E = torch.exp(X - s)
                K, _, I, J = E.shape
                if Wb.size(0) == 1:
                    out = (Wb[0] @ E.reshape(K, -1)).reshape(-1, B, I, J)
                else:
                    out = torch.einsum("bnk,kbr->nbr", Wb, E.reshape(K, B, -1)).reshape(-1, B, I, J)
                self.vals[ns] = torch.log(out) + s
        self.log_marginal = self.vals[self.order[-1]][0, :, 0, 0]
        return self.log_marginal

    # ---------------------------------------------------------------------------------------------
    # The outside pass and the conditionals
    # ---------------------------------------------------------------------------------------------

    def conditionals(self, data, missing, inputs = None, tokens = None):
        """``(log p(C, e) [B], conditionals)``: the missing tokens' ``log p(x_t = v | C, e)``, float64 [rows, V] (the
        missing ``(b, t)`` in row-major order, as the decoder lists them), -inf where impossible (a sample of no mass:
        its rows) -- ``tokens`` (None: every token) a long tensor [rows, k] of the tokens to compute (-1: none), the
        others -inf."""
        lm = self.inside(data, missing, inputs)
        B = data.size(0)
        out_blk, out_in = {}, {}
        root = self.order[-1]
        out_blk[root] = torch.zeros_like(self.vals[root])

        def add(d, ns, idx, x):
            if ns not in d:
                d[ns] = torch.full((ns.num_nodes,) + tuple(x.shape[1:]), NEG, dtype = F64, device = self.dev)
            _logaddexp_into(d[ns], idx, x)

        for ns in reversed(self.order):
            O = out_blk.get(ns)
            if O is None or ns.is_input():
                continue
            if ns.is_sum():
                Wb = self.weights(ns, B)
                s = _shift(O, 0)
                E = torch.exp(O - s)                                        # [N, B, I, J]
                N, _, I, J = E.shape
                if Wb.size(0) == 1:
                    Oc = (Wb[0].T @ E.reshape(N, -1)).reshape(-1, B, I, J)
                else:
                    Oc = torch.einsum("bnk,nbr->kbr", Wb, E.reshape(N, B, -1)).reshape(-1, B, I, J)
                Oc = torch.log(Oc) + s
                at = 0
                for cs in ns.chs:
                    part = Oc[at:at + cs.num_nodes]
                    at += cs.num_nodes
                    if cs.is_input():
                        add(out_in, cs, None, self._input_block_outside(part, self.scope[cs][0]))
                    else:
                        add(out_blk, cs, None, part)
                continue
            parts = self._parts(ns)
            values = [self._part_value(ns, kind, ks) for kind, ks in parts]
            for p, (kind, ks) in enumerate(parts):
                M = O
                Rr = None
                for q in range(p + 1, len(parts)):
                    Rr = values[q] if Rr is None else _log_matmul(Rr, values[q])
                if Rr is not None:
                    M = _log_matmul(M, Rr.transpose(-1, -2))
                L = None
                for q in range(p):
                    L = values[q] if L is None else _log_matmul(L, values[q])
                if L is not None:
                    M = _log_matmul(L.transpose(-1, -2), M)                    # the part's outside
                idx = [self.product_child_index(ns, k) for k in ks]
                if kind == "blk":
                    cs = ns.chs[ks[0]]
                    if cs.is_input():
                        add(out_in, cs, idx[0], self._input_block_outside(M, self.scope[cs][0]))
                    else:
                        add(out_blk, cs, idx[0], M)
                else:
                    u, r = ns.chs[ks[0]], ns.chs[ks[1]]
                    ou, orr = self._input_op_outside(M, self.masses[u][idx[0]], self.scope[u][0],
                                                     self._block_of(r)[idx[1]])
                    add(out_in, u, idx[0], ou)
                    add(out_blk, r, idx[1], orr)

        rows = torch.nonzero(self.missing)                                    # [R, 2]: (b, t)
        V = next(ns.dist.num_cats for ns in self.order if ns.is_input())
        joint = torch.full((rows.size(0), V), NEG, dtype = F64, device = self.dev)
        for ns in self.order:
            if not ns.is_input() or ns not in out_in:
                continue
            t = self.scope[ns][0]
            q, _ = self.emission(ns, B)                                       # [N, B, V]
            w = _logsumexp(out_in[ns], dim = 2)                               # [N, B, C]: outside per class
            for r in torch.nonzero(rows[:, 1] == t).view(-1).tolist():
                b = int(rows[r, 0])
                tok = torch.arange(V, device = self.dev) if tokens is None else tokens[r].to(self.dev).long()
                tok = tok[(tok >= 0) & (tok < V)]
                j = _logsumexp(torch.log(q[:, b, tok]) + w[:, b, self.token_class[tok]], dim = 0)
                joint[r, tok] = torch.logaddexp(joint[r, tok], j)
        lmr = lm[rows[:, 0].to(self.dev)][:, None]
        return lm, torch.where(torch.isfinite(lmr), joint - lmr, NEG)     # (no mass: -inf rows, as the decoder's)


# -------------------------------------------------------------------------------------------------
# Circuits, evidence, sessions
# -------------------------------------------------------------------------------------------------

def codpula_pc(n, latents, V, gated = True, seed = 0, block = 32):
    """A Codpula circuit: a GeneralizedHMM of ``latents`` in blocks of ``block`` over ``V`` tokens, its inputs
    SoftEvidenceCategorical, its transition sums under BlockScale external parameters (one gate per pair of blocks)."""
    from pyjuice.nodes.external_sum_nodes import ExternalParamsSumNodes
    torch.manual_seed(seed)
    ns = juice.structures.GeneralizedHMM(seq_length = n, num_latents = latents, block_size = block,
                                         input_dist = dists.SoftEvidenceCategorical(num_cats = V))
    if gated:
        for s in ns:
            if s.is_sum() and s.num_nodes == latents:
                s.__class__ = ExternalParamsSumNodes
                s.external_params = BlockScaleSumParams(ch_block_size = block)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False).to(DEV)


def pd_pc(n, latents, V, seed = 0):
    """A PD circuit over ``n`` variables of ``V`` categories, ``latents`` nodes a region."""
    torch.manual_seed(seed)
    ns = juice.structures.PD(data_shape = (n,), num_latents = latents, split_intervals = 1,
                             input_node_params = {"num_cats": V})
    return juice.compile(ns, verbose = False).to(DEV)


def keyword_dfa(V, phrases, forbidden = None):
    """Every one of ``phrases`` and, if given, never ``forbidden``."""
    dfa = jc.DFA.contains([phrases[0]], V)
    for ph in phrases[1:]:
        dfa = dfa & jc.DFA.contains([ph], V)
    if forbidden is not None:
        dfa = dfa & jc.Not(jc.DFA.contains([[forbidden]], V))
    return dfa.automaton() if hasattr(dfa, "automaton") else dfa


def soft_inputs(pc, B, n, V, K = None, gates = True, seed = 0, data = None, boost = 6.0):
    """Per-sample inputs as a decoder takes them: soft evidence over ``V`` tokens (top-``K`` when given; the data's
    tokens boosted, when given, so some evidence agrees with them) and, if the PC has them, external parameters."""
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(B, n, V, generator = g) * 2
    if data is not None:
        E.scatter_add_(2, data.cpu()[:, :, None], torch.full((B, n, 1), boost))
    kw = {}
    if K is None:
        kw["categorical_evidence_logp"] = torch.log_softmax(E, dim = -1).to(DEV)
    else:
        top = torch.topk(E, K, dim = 2)
        ids, order = torch.sort(top.indices, dim = 2)
        kw["categorical_evidence_logp"] = torch.log_softmax(top.values, dim = -1).gather(2, order).to(DEV)
        kw["soft_evidence_cat_ids"] = ids.to(DEV)
    if gates and pc.external_params_nodes:
        kw["sum_external_params"] = {s: (torch.randn(s.external_params.tensor_shapes(s, B)[0], generator = g) * 2)
                                     .to(DEV) for s in pc.external_params_nodes}
    return kw


def schedule(mode, B, n, seed = 0, calls = None):
    """The masks of a session: ``diffusion`` (two more positions observed a call, in a random order per sample) or
    ``l2r`` (left to right); ``calls`` (None: every call) the calls kept."""
    rng = np.random.default_rng(seed)
    order = np.stack([rng.permutation(n) for _ in range(B)])
    masks = []
    for t in range(n - 1):
        if mode == "l2r":
            m = np.broadcast_to(np.arange(n)[None, :] >= t, (B, n)).copy()
        else:
            m = np.ones((B, n), dtype = bool)
            for b in range(B):
                m[b, order[b, :min(n - 1, 2 * t)]] = False
        masks.append(torch.tensor(m))
    keep = range(len(masks)) if calls is None else calls
    return [(i, masks[i]) for i in keep if i < len(masks)]


def run_session(cc, data, masks, inputs, storage = "auto", precision = "fp32"):
    """A decoder session over ``masks`` (``(call, mask)``, the calls in order): per call ``(marginal [B], log
    marginal of every conditional row, conditionals [rows, V])``. The calls before the first kept one run too (so the
    session's held values are those of the calls in between)."""
    out = {}
    wanted = {i for i, _ in masks}
    last = max(wanted)
    all_masks = dict(masks)
    with cc.decoder(storage = storage, precision = precision, **(inputs or {})) as d:
        for i in range(last + 1):
            m = all_masks.get(i)
            if m is None:
                continue
            x, mm = data.to(DEV), m.to(DEV)
            marg = d.marginal(x, mm)
            cond, lm = d.conditional(x, mm, return_log_marginal = True)
            out[i] = (marg.detach().double().view(-1), lm.detach().double().view(-1), cond.detach().double())
    return out


# -------------------------------------------------------------------------------------------------
# Comparisons
# -------------------------------------------------------------------------------------------------

def compare(got, want, tol, what = ""):
    """Max |got - want| over the finite entries; asserts the -inf patterns agree and every finite entry is within
    ``tol`` (log space)."""
    got, want = got.double().cpu(), want.double().cpu()
    assert got.shape == want.shape, (what, got.shape, want.shape)
    gi, wi = torch.isneginf(got), torch.isneginf(want)
    assert torch.equal(gi, wi), f"{what}: -inf patterns differ at {torch.nonzero(gi != wi)[:5].tolist()}"
    assert not torch.isnan(got).any(), f"{what}: NaN"
    fin = torch.isfinite(want)
    d = float((got[fin] - want[fin]).abs().max()) if fin.any() else 0.0
    assert d <= tol, f"{what}: max |diff| {d:.3g} > {tol}"
    return d


def candidates_of(inputs, rows):
    """The tokens a top-k evidence's rows have (long [rows, k]), or None (dense evidence, no evidence)."""
    ids = None if inputs is None else inputs.get("soft_evidence_cat_ids")
    if ids is None:
        return None
    ids = ids.cpu().long()
    b = rows[:, 0] if ids.size(0) > 1 else torch.zeros_like(rows[:, 0])
    return ids[b, rows[:, 1]]


def check_against_reference(cc, ref, data, masks, inputs, storage, precision):
    """Every kept call of a session against the reference -- the marginals, the conditional rows' log marginals, and
    the conditionals (under top-k evidence: every candidate, and -inf at every other token); the worst difference."""
    got = run_session(cc, data, masks, inputs, storage, precision)
    tm, tc = TOLERANCE[precision]["marginal"], TOLERANCE[precision]["conditional"]
    worst = 0.0
    for i, m in masks:
        marg, lmg, cond = got[i]
        rows = torch.nonzero(m)
        tok = candidates_of(inputs, rows)
        lm, want = ref.conditionals(data, m, inputs, tokens = tok)
        worst = max(worst, compare(marg, lm, tm, f"call {i} marginal"))
        worst = max(worst, compare(lmg, lm[rows[:, 0].to(lm.device)], tm, f"call {i} log marginal"))
        if tok is not None:
            off = torch.ones_like(cond, dtype = torch.bool)
            off.scatter_(1, tok.to(cond.device), False)
            assert torch.isneginf(cond[off]).all(), f"call {i}: a token that is no candidate is not -inf"
        worst = max(worst, compare(cond, want, tc, f"call {i} conditionals"))
    return worst


def check_invariants(cc, data, masks, inputs, precision = "fp32"):
    """The invariants of the module docstring on one session (the marginals to the marginals' tolerance, the
    conditionals and the identities on them to the conditionals'); returns the worst difference."""
    from pyjuice.constraints.backends.dfa_exact.kernels import live
    tm, tc = TOLERANCE[precision]["marginal"], TOLERANCE[precision]["conditional"]
    outputs = (("marginal", tm), ("log marginal", tm), ("conditionals", tc))
    worst = 0.0
    base = run_session(cc, data, masks, inputs, "auto", precision)
    B = data.size(0)
    # storage and pruning
    for storage in ("dense", "compact"):
        other = run_session(cc, data, masks, inputs, storage, precision)
        for i, _ in masks:
            for a, b, (w, tol) in zip(base[i], other[i], outputs):
                worst = max(worst, compare(b, a, tol, f"{storage} call {i} {w}"))
    old = live.LIVE_SUPPORT
    try:
        live.LIVE_SUPPORT = not old
        other = run_session(cc, data, masks, inputs, "auto", precision)
    finally:
        live.LIVE_SUPPORT = old
    for i, _ in masks:
        for a, b, (w, tol) in zip(base[i], other[i], outputs):
            worst = max(worst, compare(b, a, tol, f"pruning switched call {i} {w}"))
    # the samples permuted, and one alone
    perm = torch.randperm(B, generator = torch.Generator().manual_seed(1))
    pinputs = permute_inputs(inputs, perm, B)
    other = run_session(cc, data[perm], [(i, m[perm]) for i, m in masks], pinputs, "auto", precision)
    for i, m in masks:
        worst = max(worst, compare(other[i][0], base[i][0][perm.to(base[i][0].device)], tm, f"permuted call {i}"))
        rows = torch.nonzero(m)
        prow = torch.nonzero(m[perm])
        key = {(int(b), int(t)): r for r, (b, t) in enumerate(rows.tolist())}
        sel = torch.tensor([key[(int(perm[b]), int(t))] for b, t in prow.tolist()], dtype = torch.long)
        worst = max(worst, compare(other[i][2], base[i][2][sel.to(base[i][2].device)], tc, f"permuted call {i} cond"))
    b0 = int(B - 1)
    one = run_session(cc, data[b0:b0 + 1], [(i, m[b0:b0 + 1]) for i, m in masks],
                      permute_inputs(inputs, torch.tensor([b0]), B), "auto", precision)
    for i, m in masks:
        worst = max(worst, compare(one[i][0], base[i][0][b0:b0 + 1], tm, f"alone call {i}"))
        rows = torch.nonzero(m)
        sel = (rows[:, 0] == b0).to(base[i][2].device)
        worst = max(worst, compare(one[i][2], base[i][2][sel], tc, f"alone call {i} cond"))
    # identities: conditionals normalize; the chain rule one token on
    for i, m in masks:
        marg, lmg, cond = base[i]
        fin = torch.isfinite(lmg)
        if fin.any():
            z = torch.logsumexp(cond[fin], dim = 1)
            worst = max(worst, compare(z, torch.zeros_like(z), tc, f"call {i} normalization"))
        rows = torch.nonzero(m)
        if rows.size(0) and fin.any():
            r = int(torch.nonzero(fin)[0])
            b, t = (int(v) for v in rows[r])
            v = int(torch.argmax(cond[r]))
            x2, m2 = data.clone(), m.clone()
            x2[b, t], m2[b, t] = v, False
            with cc.decoder(precision = precision, **(inputs or {})) as d:
                after = d.marginal(x2.to(DEV), m2.to(DEV)).double().view(-1)[b]
            worst = max(worst, compare(after.view(1), (marg[b] + cond[r, v]).view(1), tc, f"call {i} chain rule"))
    return worst


def permute_inputs(inputs, perm, B):
    """``inputs`` with the samples ``perm`` (rows of a batch of one stay)."""
    if inputs is None:
        return None
    out = {}
    for k, v in inputs.items():
        if isinstance(v, dict):
            out[k] = {ns: (t[perm.to(t.device)] if t.size(0) == B else t) for ns, t in v.items()}
        else:
            out[k] = v[perm.to(v.device)] if v.size(0) == B else v
    return out


# -------------------------------------------------------------------------------------------------
# The tests: small circuits (the default tier)
# -------------------------------------------------------------------------------------------------

def brute_force(pc, dfa, data, missing, inputs):
    """``(log p(C, e) [B], conditionals [rows, V])`` by enumerating every completion of the missing tokens, each
    string's value pyjuice's own forward (soft evidence and gates as ``pc(x, ...)`` takes them), in float64 sums."""
    B, n = data.shape
    V = next(ns.dist.num_cats for ns in pc.root_ns if ns.is_input())
    lms, rows = [], []
    for b in range(B):
        miss = torch.nonzero(missing[b]).view(-1).tolist()
        fills = torch.tensor(list(itertools.product(range(V), repeat = len(miss))), dtype = torch.long)
        xs = data[b].repeat(max(1, fills.size(0)), 1)
        if miss:
            xs[:, miss] = fills
        kw = {}
        for key, val in (inputs or {}).items():
            if isinstance(val, dict):
                kw[key] = {ns: t[b if t.size(0) > 1 else 0][None].expand(xs.size(0), *t.shape[1:]).contiguous()
                           for ns, t in val.items()}
            else:
                kw[key] = val[b if val.size(0) > 1 else 0][None].expand(xs.size(0), *val.shape[1:]).contiguous()
        with torch.no_grad():
            lp = pc(xs.to(DEV), **kw).view(-1).double().cpu()
        ok = torch.tensor([dfa.accepts(x) for x in xs.tolist()])
        lp = torch.where(ok, lp, torch.tensor(NEG, dtype = F64))
        lm = torch.logsumexp(lp, dim = 0)
        lms.append(lm)
        for t in miss:
            row = torch.full((V,), NEG, dtype = F64)
            for v in range(V):
                sel = xs[:, t] == v
                if sel.any():
                    row[v] = torch.logsumexp(lp[sel], dim = 0)
            rows.append(row - lm if torch.isfinite(lm) else torch.full((V,), NEG, dtype = F64))
    return torch.stack(lms), (torch.stack(rows) if rows else torch.zeros(0, V, dtype = F64))


TINY = {"codpula": dict(n = 4, latents = 8, V = 4), "pd": dict(n = 4, latents = 2, V = 3)}


def tiny_case(kind, seed = 0):
    """A tiny circuit, automaton, evidence and inputs (every completion enumerable)."""
    cfg = TINY[kind]
    n, V = cfg["n"], cfg["V"]
    if kind == "codpula":
        pc = codpula_pc(n, cfg["latents"], V, block = 4, seed = seed)
        dfa = keyword_dfa(V, [[1, 2]], forbidden = 3)
    else:
        pc = pd_pc(n, cfg["latents"], V, seed = seed)
        dfa = keyword_dfa(V, [[0, 1], [2]])
    g = torch.Generator().manual_seed(seed)
    B = 4
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.6
    missing[0] = True
    inputs = soft_inputs(pc, B, n, V, K = 3, seed = seed) if kind == "codpula" else None
    return pc, dfa, data, missing, inputs


@pytest.mark.parametrize("kind", ["codpula", "pd"])
def test_the_reference_matches_a_brute_force(kind):
    """The reference itself, against every completion enumerated through pyjuice's own forward (float32 there, so
    to 1e-5): marginals and every missing token's conditional -- top-k soft evidence and gates for Codpula."""
    pc, dfa, data, missing, inputs = tiny_case(kind)
    ref = Float64Reference(pc, dfa, data.size(1))
    lm, cond = ref.conditionals(data, missing, inputs)
    want_lm, want_cond = brute_force(pc, dfa, data, missing, inputs)
    compare(lm, want_lm, 1e-5, "marginal")
    compare(cond, want_cond, 1e-5, "conditionals")


SMALL = {"codpula": dict(n = 8, latents = 64, V = 40, K = 6, B = 5), "pd": dict(n = 8, latents = 4, V = 6, B = 5)}


def small_case(kind, seed = 0):
    """A small circuit (Codpula: blocks of 32 under gates, top-k evidence; PD: Categorical inputs), an automaton,
    data whose observed tokens keep the constraint possible, and the inputs."""
    cfg = SMALL[kind]
    n, V, B = cfg["n"], cfg["V"], cfg["B"]
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(4, V, (B, n), generator = g)
    if kind == "codpula":
        pc = codpula_pc(n, cfg["latents"], V, seed = seed)
        phrases = [[1, 2], [3]]
        dfa = keyword_dfa(V, phrases, forbidden = 0)
    else:
        pc = pd_pc(n, cfg["latents"], V, seed = seed)
        phrases = [[0, 1], [2]]
        dfa = keyword_dfa(V, phrases)
    data[:, 2:4], data[:, 5] = torch.tensor(phrases[0]), phrases[1][0]
    inputs = soft_inputs(pc, B, n, V, K = cfg["K"], seed = seed, data = data) if kind == "codpula" else None
    return pc, dfa, data, inputs


@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("mode", ["diffusion", "l2r"])
@pytest.mark.parametrize("kind", ["codpula", "pd"])
def test_a_session_against_float64(kind, mode, storage, precision):
    """Every call of a decoder session -- marginals, log marginals, every missing token's conditional -- against the
    float64 reference (to :data:`TOLERANCE`; -inf exactly where it has it)."""
    pc, dfa, data, inputs = small_case(kind)
    cc = jc.compile(dfa, pc)
    ref = Float64Reference(pc, dfa, data.size(1))
    masks = schedule(mode, data.size(0), data.size(1))
    check_against_reference(cc, ref, data, masks, inputs, storage, precision)


@pytest.mark.parametrize("kind", ["codpula", "pd"])
def test_the_invariants(kind):
    """The invariants of the module docstring on a diffusion session's calls 0, 1 and 3."""
    pc, dfa, data, inputs = small_case(kind)
    cc = jc.compile(dfa, pc)
    check_invariants(cc, data, schedule("diffusion", data.size(0), data.size(1), calls = [0, 1, 3]), inputs)


# -------------------------------------------------------------------------------------------------
# The tests at the benchmarks' shapes (slow)
# -------------------------------------------------------------------------------------------------

def shape_case(kind, latents = 1024, B = 2, keys = 8, seed = 0):
    """Codpula at its shape (32 positions, ``latents`` in blocks of 32, 126464 tokens, top-1024 evidence, gates) under
    a keyword automaton of ``keys`` phrases; or PD16 (32 positions, 16 latents a region) over 50257 tokens under an
    automaton of 3 phrases."""
    n = 32
    krng = np.random.default_rng(7)
    if kind == "codpula":
        V = 126464
        phrases = [[int(v) for v in krng.integers(1000, 48000, size = int(krng.integers(2, 4)))] for _ in range(keys)]
        pc = codpula_pc(n, latents, V, seed = seed)
        dfa = keyword_dfa(V, phrases, forbidden = 49000)
    else:
        V = 50257
        phrases = [[int(v) for v in krng.integers(1000, 48000, size = 2)] for _ in range(3)]
        pc = pd_pc(n, 16, V, seed = seed)
        dfa = keyword_dfa(V, phrases)
    rng = np.random.default_rng(B)
    data = torch.tensor(rng.integers(0, 900, size = (B, n)))
    flat = [v for ph in phrases for v in ph]
    data[:, 2:2 + len(flat)] = torch.tensor(flat)
    inputs = soft_inputs(pc, B, n, V, K = 1024, seed = seed, data = data, boost = 25.0) if kind == "codpula" else None
    return pc, dfa, data, inputs


@pytest.mark.slow
@pytest.mark.parametrize("precision", ["tf32", "fp32", "bf16"])
@pytest.mark.parametrize("kind", ["codpula", "pd"])
def test_the_shapes_against_float64(kind, precision):
    """Codpula (1024 latents, 126464 tokens, top-1024, gates, 8 keyword phrases) and PD16 (50257 tokens): diffusion
    sessions' calls 0, 1, 4, 10, 20 and 30 against the float64 reference."""
    pc, dfa, data, inputs = shape_case(kind)
    cc = jc.compile(dfa, pc)
    ref = Float64Reference(pc, dfa, data.size(1))
    masks = schedule("diffusion", data.size(0), data.size(1), calls = [0, 1, 4, 10, 20, 30])
    check_against_reference(cc, ref, data, masks, inputs, "auto", precision)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["codpula", "pd"])
def test_the_invariants_at_the_shapes(kind):
    pc, dfa, data, inputs = shape_case(kind, B = 3)
    cc = jc.compile(dfa, pc)
    check_invariants(cc, data, schedule("diffusion", data.size(0), data.size(1), calls = [0, 4, 20]), inputs, "tf32")


# -------------------------------------------------------------------------------------------------
# The full shapes from the command line
# -------------------------------------------------------------------------------------------------

def main(argv = None):
    ap = argparse.ArgumentParser(description = "The strong checks at a chosen shape (default: Codpula's full one).")
    ap.add_argument("--kind", choices = ["codpula", "pd"], default = "codpula")
    ap.add_argument("--latents", type = int, default = 4096)
    ap.add_argument("--keys", type = int, default = 8)
    ap.add_argument("--batch", type = int, default = 2)
    ap.add_argument("--mode", choices = ["diffusion", "l2r"], default = "diffusion")
    ap.add_argument("--calls", default = "0,1,4,10,20,30")
    ap.add_argument("--precision", choices = ["fp32", "tf32", "bf16"], default = "tf32")
    ap.add_argument("--storage", choices = ["auto", "dense", "compact"], default = "auto")
    ap.add_argument("--invariants", action = "store_true", help = "also the invariants")
    args = ap.parse_args(argv)
    pc, dfa, data, inputs = shape_case(args.kind, latents = args.latents, B = args.batch, keys = args.keys)
    cc = jc.compile(dfa, pc)
    ref = Float64Reference(pc, dfa, data.size(1))
    masks = schedule(args.mode, data.size(0), data.size(1), calls = [int(c) for c in args.calls.split(",")])
    worst = check_against_reference(cc, ref, data, masks, inputs, args.storage, args.precision)
    print(f"{args.kind} latents {args.latents} B {args.batch} {args.mode} {args.precision} {args.storage}: "
          f"worst |diff| vs float64 {worst:.3g} (tolerances {TOLERANCE[args.precision]})")
    if args.invariants:
        print(f"invariants: worst |diff| {check_invariants(cc, data, masks, inputs, args.precision):.3g}")


if __name__ == "__main__":
    main()
