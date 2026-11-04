"""
The decoder against a hand computation: small circuits built from explicit parameters -- an HMM, and a tree whose root
has two groups of children -- their polynomials written out in float64 numpy, every completion of the free positions
enumerated and filtered by the constraint. Gated sums (:class:`~pyjuice.nodes.external_params.BlockScaleSumParams`,
node and child gates of a block or finer) and :class:`~pyjuice.nodes.distributions.SoftEvidenceCategorical` inputs
(dense, top-k, value mask), per row or one row for all.

Sample ``b``'s gated sum weighs child ``c`` of node ``n`` by ``theta[n, c] exp(lphi_b[n // ng, c // cg])``,
normalized per node. Input node ``n`` at position ``i`` weighs token ``v`` by ``p_n(v) w_i(v) / Z_ni``, ``w_i(v) =
exp(E_i(v))`` (top-k: for a candidate, else 0) and ``Z_ni = sum_u p_n(u) w_i(u)`` -- observed and completed tokens
alike; under a value mask an observed token weighs ``p_n(v) w_i(v)`` if True, a position False is free and completed
with ``p_n(v) w_i(v)``, a missing one with ``p_n(v) w_i(v) / Z_ni`` (the library's forward, softevi_categorical.py).
Every test checks the hand polynomial against the library's own forward first -- the unconstrained marginals, and
without a value mask every full assignment -- then the decoder's marginals and conditionals under two constraints,
dense and compact storage, and a prefix-decoding session call by call; the fused sum kernels' reductions split by
the library's rule, or as far as they go; the top-down's parent chunks precomputed by the library's rule, or always.
"""
import itertools

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import BlockScaleSumParams, inputs, multiply, summate

V, N_VARS, B = 3, 4, 3
DEV = torch.device("cuda:0")
CONSTRAINTS = {
    "anything": lambda: jc.DFA.anything(V),
    "contains_either_and_not": lambda: jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)),
}
#: (latents, block size, (node gate, child gate) of the inner sums -- the root's child gate is theirs -- or None)
SHAPES = {"b2_node1_child1": (4, 2, (1, 1)), "b2_ungated": (4, 2, None),
          "b32_block_child1": (64, 32, (32, 1)),           # a gate per block, a gate per child: per-edge gates
          "b32_node16_child32": (64, 32, (16, 32)),        # a tile of rows within one node gate
          "b32_node8_child8": (64, 32, (8, 8))}            # a tile of rows of several, chunks of several cells


def _softmax_rows(a):
    a = np.exp(a - a.max(1, keepdims = True))
    return a / a.sum(1, keepdims = True)


class _Sum:
    """A sum node group of the hand circuit: its nodes, ``theta`` [N, C] and gate sizes (None: plain)."""

    def __init__(self, rng, chs, N, bs, gate):
        self.C = sum(c.num_nodes for c in chs)
        self.theta = _softmax_rows(rng.normal(size = (N, self.C)) * 1.5)
        self.gate = gate
        ext = None if gate is None else BlockScaleSumParams(block_size = None if gate[0] == bs else gate[0],
                                                             ch_block_size = gate[1])
        self.ns = summate(*chs, num_node_blocks = N // bs, block_size = bs, external_params = ext)
        self.ns.set_params(torch.tensor(self.theta, dtype = torch.float32))

    def weights(self, lphi):
        """[N, C] the weights under one sample's log gates (None: ``theta``)."""
        if self.gate is None or lphi is None:
            return self.theta
        ng, cg = self.gate
        g = lphi[np.arange(self.theta.shape[0])[:, None] // ng, np.arange(self.C)[None, :] // cg]
        w = self.theta * np.exp(g - g.max(1, keepdims = True))
        return w / w.sum(1, keepdims = True)


def build(rng, circuit, H, bs, gate, soft):
    """``(root, sums, P, evaluate)``: ``P[i]`` [H, V] position ``i``'s input probabilities; ``evaluate(F, lphi)`` the
    root's values [C] from the input factors ``F[i]`` [C, H] of C assignments and one sample's log gates ``lphi``
    (sum index -> [node gates, child gates])."""
    xs, P = [], []
    for i in range(N_VARS):
        dist = dists.SoftEvidenceCategorical(num_cats = V) if soft else dists.Categorical(num_cats = V)
        xs.append(inputs(i, num_node_blocks = H // bs, block_size = bs, dist = dist))
        P.append(_softmax_rows(rng.normal(size = (H, V)) * 1.5))
        xs[-1].set_params(torch.tensor(P[-1], dtype = torch.float32).reshape(-1))
    root_gate = None if gate is None else (1, gate[1])
    if circuit == "hmm":                                   # s_t[k] = sum_j T_t[k, j] h_{t+1}[j], h_t = x_t s_t
        sums, h = [], xs[-1]
        for t in range(N_VARS - 2, -1, -1):
            sums.insert(0, _Sum(rng, [h], H, bs, gate))
            h = multiply(xs[t], sums[0].ns)
        sums.append(_Sum(rng, [h], 1, 1, root_gate))

        def evaluate(F, lphi):
            h = F[-1]
            for t in range(N_VARS - 2, -1, -1):
                h = F[t] * (h @ sums[t].weights(lphi.get(t)).T)
            return (h @ sums[-1].weights(lphi.get(len(sums) - 1)).T)[:, 0]
    else:                                                  # the root's children: a b and a2 b
        a, a2, b = (_Sum(rng, [multiply(xs[u], xs[u + 1])], H, bs, gate) for u in (0, 0, 2))
        sums = [a, a2, b, _Sum(rng, [multiply(a.ns, b.ns), multiply(a2.ns, b.ns)], 1, 1, root_gate)]

        def evaluate(F, lphi):
            A, A2, Bv = ((F[u] * F[u + 1]) @ s.weights(lphi.get(k)).T for k, (s, u) in enumerate(zip(sums, (0, 0, 2))))
            return (np.concatenate([A * Bv, A2 * Bv], 1) @ sums[-1].weights(lphi.get(3)).T)[:, 0]
    return sums[-1].ns, sums, P, evaluate


def factors(P, x, b, ev, missing, vmask):
    """``F[i]`` [C, H]: every input node's factor at position ``i`` for the assignments ``x`` [C, n] of sample ``b``."""
    F = []
    for i in range(N_VARS):
        v = x[:, i]
        if ev is None:
            F.append(P[i][:, v].T)
            continue
        E, ids = ev
        be = b if E.shape[0] > 1 else 0
        w = np.zeros(V)
        if ids is None:
            w[:] = np.exp(E[be, i])
        else:
            w[ids[be, i]] = np.exp(E[be, i])
        Z = P[i] @ w
        unnormalized = vmask is not None and not missing[b, i]
        F.append((P[i][:, v] * w[v][None, :] / (1.0 if unnormalized else Z[:, None])).T)
    return F


def enumerate_truth(evaluate, P, data, missing, vmask, ev, lphi, constraint):
    """Per sample: the log marginal under the constraint, the unconstrained one, the conditional rows of its free
    positions (missing, or value mask False; no mass under the constraint: -inf rows), the assignments enumerated and
    their values."""
    out = []
    for b in range(B):
        free = np.nonzero(missing[b] | (False if vmask is None else ~vmask[b]))[0]
        x = np.repeat(data[b][None], V ** len(free), 0)
        if len(free):
            x[:, free] = np.array(list(itertools.product(range(V), repeat = len(free))))
        values = evaluate(factors(P, x, b, ev, missing, vmask),
                          {k: g[b if g.shape[0] > 1 else 0] for k, g in lphi.items()})
        accepted = np.array([constraint.accepts(row) for row in x.tolist()])
        total = values[accepted].sum()
        with np.errstate(divide = "ignore"):
            rows = [np.log([values[accepted & (x[:, t] == v)].sum() for v in range(V)]) -
                    (np.log(total) if total > 0 else 0.0) for t in free]
            out.append((np.log(total), np.log(values.sum()), rows, x, values))
    return out


def assert_close(got, want, tol = 2e-5):
    got = np.asarray(got.detach().double().cpu() if torch.is_tensor(got) else got, dtype = np.float64).reshape(-1)
    want = np.asarray(want, dtype = np.float64).reshape(-1)
    ninf = np.isneginf(want)
    assert np.array_equal(np.isneginf(got), ninf), (got, want)
    if (~ninf).any():
        d = np.abs(got[~ninf] - want[~ninf])
        assert (d <= tol * (1 + np.abs(want[~ninf]))).all(), d.max()


@pytest.mark.parametrize("rows", ["per_row", "shared"])
@pytest.mark.parametrize("evidence", ["none", "dense", "topk", "value_mask"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("circuit", ["hmm", "tree"])
def test_the_hand_computation(circuit, shape, evidence, rows, split, pre):
    H, bs, gate = SHAPES[shape]
    if gate is None and evidence == "none":
        pytest.skip("nothing per sample")
    rng = np.random.default_rng(0)
    root, sums, P, evaluate = build(rng, circuit, H, bs, gate, evidence != "none")
    pc = juice.compile(root, verbose = False).to(DEV)
    data = rng.integers(0, V, size = (B, N_VARS))
    missing = rng.random((B, N_VARS)) < 0.5
    missing[0], missing[1, 0] = True, False
    Br = B if rows == "per_row" else 1
    kw, lphi, ev, vmask = {}, {}, None, None
    if gate is not None:
        for k, s in enumerate(sums):
            lphi[k] = rng.normal(size = (Br, s.theta.shape[0] // s.gate[0], s.C // s.gate[1])) * 1.5
        assert {sums[k].ns for k in lphi} == set(pc.external_params_nodes)
        kw["sum_external_params"] = {sums[k].ns: torch.tensor(g, dtype = torch.float32, device = DEV)
                                     for k, g in lphi.items()}
    if evidence != "none":
        E = rng.normal(size = (Br, N_VARS, V)) * 2.0
        if evidence == "topk":                                 # 2 of 3 candidates
            ids = np.argsort(-E, axis = 2)[:, :, :2].copy()
            E = np.take_along_axis(E, ids, 2)
            kw["soft_evidence_cat_ids"] = torch.tensor(ids, device = DEV)
        ev = (E, ids if evidence == "topk" else None)
        kw["categorical_evidence_logp"] = torch.tensor(E, dtype = torch.float32, device = DEV)
        if evidence == "value_mask":
            vmask = (rng.random((B, N_VARS)) < 0.6) | missing     # missing and False is refused
            kw["soft_evidence_value_mask"] = torch.tensor(vmask, device = DEV)
    x, m = torch.tensor(data, device = DEV), torch.tensor(missing, device = DEV)

    # The hand polynomial is the library's (whose sums run TF32 from blocks of 16: ~1e-3 a layer; a wrong hand model
    # is off by 0.1 or more)
    tol = 2e-5 if bs < 16 else 1e-2
    truth = enumerate_truth(evaluate, P, data, missing, vmask, ev, lphi, CONSTRAINTS["anything"]())
    full = lambda t, k: t.expand(k, *t.shape[1:]).contiguous() if t.size(0) == 1 else t
    lib_kw = {key: ({ns: full(t, B) for ns, t in val.items()} if isinstance(val, dict) else full(val, B))
              for key, val in kw.items()}
    with torch.no_grad():
        assert_close(pc(x, missing_mask = m, **lib_kw), [t[1] for t in truth], tol)
        for b in range(B if vmask is None else 0):
            xs = truth[b][3]
            row = lambda t: t[b if t.size(0) > 1 else 0][None].expand(len(xs), *t.shape[1:]).contiguous()
            kb = {key: ({ns: row(t) for ns, t in val.items()} if isinstance(val, dict) else row(val))
                  for key, val in kw.items()}
            with np.errstate(divide = "ignore"):
                assert_close(pc(torch.tensor(xs, device = DEV), **kb), np.log(truth[b][4]), tol)

    # The decoder
    free = missing if vmask is None else missing | ~vmask
    for name, make in CONSTRAINTS.items():
        constraint = make()
        cc = jc.compile(constraint, pc)
        truth = enumerate_truth(evaluate, P, data, missing, vmask, ev, lphi, constraint)
        for storage in ("dense", "compact"):
            with cc.decoder(storage = storage, precision = "fp32", **kw) as decoder:
                marg = decoder.marginal(x, m)
                cond, lm = decoder.conditional(x, m, return_log_marginal = True)
            assert_close(marg, [t[0] for t in truth])
            assert_close(cond, np.concatenate([np.stack(t[2]) for t in truth if len(t[2])]))
            assert_close(lm, [truth[b][0] for b in np.nonzero(free)[0]])
        with cc.decoder(precision = "fp32", **kw) as decoder:                      # prefix decoding, positions t.. missing
            for t in range(N_VARS):
                pm = np.broadcast_to(np.arange(N_VARS)[None, :] >= t, (B, N_VARS)).copy()
                if vmask is not None:
                    pm &= vmask                                # the session's value mask False: free anyway
                truth = enumerate_truth(evaluate, P, data, pm, vmask, ev, lphi, constraint)
                assert_close(decoder.marginal(x, torch.tensor(pm, device = DEV)), [r[0] for r in truth])
                want = [np.stack(r[2]) for r in truth if len(r[2])]
                if want:
                    assert_close(decoder.conditional(x, torch.tensor(pm, device = DEV)), np.concatenate(want))
