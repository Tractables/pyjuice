"""
A slow but obviously correct reference for constrained queries on the lifted plan, for tests only.

``reference_marginal(cc, data, missing_mask)`` computes ``log p(C, e)`` for a
:class:`~pyjuice.constraints.ConstrainedCircuit` in plain PyTorch, straight from the definition, for
every (PC, constraint) pair that compiles. The library's own implementation is tested against it; it is
exposed to the tests in this directory through the ``reference`` fixture (the test suite runs with
``--import-mode=importlib``, so test modules cannot import a helper module directly). The PCs these tests
share come from the ``build_pc`` fixture.

INPUT NODES: CATEGORICAL, AND SOFTEVIDENCECATEGORICAL. The reference turns a node's token probabilities into
per-token-class masses: a Categorical's own, or under soft evidence (``inputs``, as ``pc(x, ...)`` takes them) every
sample's product of experts ``q(v) = p(v) w(v) / Z`` -- under a value mask the library's unnormalized ``p(v) w(v)``
at an observed token and where the mask is False, a free position (marginalized whatever the data holds); any other
input node raises ``NotImplementedError``. Other distributions need their own class-mass rule here.

PRECISION: float32, the precision the library runs at. Every log-space step is max-shifted, so the
reference's own rounding stays around 1e-7 relative -- far below the tolerances it is compared at.

Every node over the contiguous scope ``[a, b]`` gets a block of log-values

    V[node, sample, i, j] = log sum over the evidence-consistent x_a..x_b of
                            p_node(x_a..x_b) * 1[the automaton goes from column i to column j reading them]

where ``i`` is a column at boundary ``a`` and ``j`` a column at boundary ``b + 1`` (columns are the
active automaton states, see :mod:`pyjuice.constraints.backends.dfa_exact.compiled.layout`). Only active columns are
kept: an accepted string of length ``n`` passes through active states only. Then

* an input node at variable ``t`` sums its token probabilities per token class, and class ``c`` moves column
  ``i`` to ``next_col[t, i, c]``;
* a product chains its children's blocks in scope order (a matrix product per node and sample);
* a sum takes its weighted sum over children, block by block;
* the root is ``log sum_j V[root, sample, 0, j]``: boundary 0 holds only the initial state, and every
  active state at boundary ``n`` is accepting.

``reference_top_down`` gives every node's TOP-DOWN block, of the same shape, from the root: ``O[node, sample, i, j]``
is the log of everything ``p(C, e)`` multiplies the node's entry ``(i, j)`` by, over the positions outside its scope
(its log outside value; pyjuice's top-down probability, with the evidence and the automaton's columns).
``reference_conditional`` reads the conditionals of target positions off the top-down blocks of their input nodes.

Parameters are read from the PC's live parameter tensors (never from the node objects, which are not
kept in sync), so in-place parameter updates are seen by the next call.
"""

import random
from functools import partial

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.constraints.backends.dfa_exact.kernels import sum as _lifted_sum
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes.distributions import Categorical, SoftEvidenceCategorical
from pyjuice.nodes.methods.edge_constructors import block_sparse_rnd_blk_edge_constructor


DTYPE = torch.float32


# -------------------------------------------------------------------------------------------------
# Parameters, read from the compiled PC
# -------------------------------------------------------------------------------------------------

def sum_weights(pc, ns) -> torch.Tensor:
    """[num_nodes, num_ch_nodes] weights of a sum node group (children concatenated in `ns.chs` order),
    from `pc.params`. Mirrors `SumNodes.update_parameters` followed by `get_params(as_matrix = True)`."""
    src = ns.get_source_ns()
    bs, cbs = ns.block_size, ns.ch_block_size
    psid, peid = src._param_range
    local = ((src._param_ids - psid) // (bs * cbs)).to(pc.params.device)
    blocks = pc.params[psid:peid].to(DTYPE).reshape(-1, cbs, bs)[local].permute(0, 2, 1)   # [edge blocks, bs, cbs]
    num_ch_nblocks = sum(cs.num_node_blocks for cs in ns.chs)
    dense = torch.zeros(ns.num_node_blocks * num_ch_nblocks, bs, cbs, dtype = DTYPE, device = pc.params.device)
    edge_ids = ns.edge_ids.to(pc.params.device)
    # accumulate: an edge block may appear twice in `edge_ids`, and the forward adds both copies
    dense.index_put_((edge_ids[0] * num_ch_nblocks + edge_ids[1],), blocks, accumulate = True)
    return dense.reshape(ns.num_node_blocks, num_ch_nblocks, bs, cbs).permute(0, 2, 1, 3).reshape(
        ns.num_nodes, num_ch_nblocks * cbs)


def categorical_probs(pc, ns) -> torch.Tensor:
    """[num_nodes, num_cats] probabilities of a Categorical input node group, from its layer's params."""
    src = ns.get_source_ns()
    for layer in pc.input_layer_group:
        if src in layer.nodes:
            ps, pe = src._param_range
            return layer.params[ps:pe].to(DTYPE).reshape(ns.num_nodes, ns.dist.num_cats)
    raise KeyError("input node group not found in any input layer")


def product_child_index(ns, k) -> torch.Tensor:
    """[num_nodes] the node of child `k` that each product node multiplies."""
    if ns.is_block_sparse():
        cs = ns.chs[k]
        assert cs.block_size == ns.block_size
        off = torch.arange(ns.num_nodes) % ns.block_size
        return ns.edge_ids[torch.arange(ns.num_nodes) // ns.block_size, k].cpu() * cs.block_size + off
    return ns.edge_ids[:, k].cpu()


# -------------------------------------------------------------------------------------------------
# Log-space block operations (max-shifted; an all -inf slice stays -inf, never NaN)
# -------------------------------------------------------------------------------------------------

def _shift(x, dim):
    m = x.amax(dim = dim, keepdim = True)
    return torch.where(torch.isfinite(m), m, torch.zeros_like(m))


def log_matmul(A, B):
    """log(exp(A) @ exp(B)) over the last two dims: A [..., I, M], B [..., M, J]."""
    sa, sb = _shift(A, -1), _shift(B, -2)                          # [..., I, 1], [..., 1, J]
    return torch.log(torch.exp(A - sa) @ torch.exp(B - sb)) + sa + sb


def log_weighted_sum(Wt, X):
    """log(Wt @ exp(X)) over the first dim of X: Wt [N, K], X [K, ...] -> [N, ...]; or every sample's own weights,
    Wt [B, N, K] and X [K, B, ...]."""
    s = _shift(X, 0)
    if Wt.dim() == 3:
        out = torch.einsum("bnk,kbr->nbr", Wt, torch.exp(X - s).reshape(X.size(0), X.size(1), -1))
        return torch.log(out).reshape(Wt.size(1), *X.shape[1:]) + s
    out = Wt @ torch.exp(X - s).reshape(X.size(0), -1)
    return torch.log(out).reshape(Wt.size(0), *X.shape[1:]) + s


def node_weights(cc, ns, inputs, B):
    """A sum node group's weights [num_nodes, num_ch_nodes] -- or, a gated one under ``sum_external_params``
    (``inputs``, as ``pc(x, ...)`` takes them), every sample's [B, num_nodes, num_ch_nodes]: ``theta[n, c] phi_b[n //
    node gate size, c // child gate size] / Z_b[n]``, from the gate tensor of the node group (or of its storage owner,
    under tied gates)."""
    W = sum_weights(cc.pc, ns)
    gates = None if inputs is None else inputs.get("sum_external_params")
    ext = getattr(ns, "external_params", None)
    if gates is None or ext is None:
        return W
    if any(isinstance(k, str) for k in gates):
        first = next(iter(gates.values()))
        gates = cc.pc._expand_external_params_groups(gates, (first[0] if isinstance(first, tuple) else first).size(0))
    lphi = gates.get(ns, gates.get(ext.storage_owner(ns)))
    if lphi is None:
        return W
    lphi = (lphi[0] if isinstance(lphi, (tuple, list)) else lphi).double()
    gbs, gcbs = ext.gate_sizes(ns)
    rows = torch.arange(ns.num_nodes, device = W.device) // gbs
    cols = torch.arange(ns.num_ch_nodes, device = W.device) // gcbs
    Wb = W.double()[None] * lphi.exp()[:, rows][:, :, cols].expand(B, -1, -1)
    return (Wb / Wb.sum(dim = 2, keepdim = True)).to(DTYPE)


# -------------------------------------------------------------------------------------------------
# The reference
# -------------------------------------------------------------------------------------------------

def _evidence(data, missing_mask, n, cc = None, inputs = None):
    """``(data, missing)`` on the host; with soft evidence's value mask (``inputs``), the positions it marks False
    missing too -- free: marginalized under the evidence's weights, whatever ``data`` holds there."""
    B = data.size(0)
    if missing_mask is None:
        missing = torch.zeros(B, n, dtype = torch.bool)
    else:
        missing = missing_mask.cpu().bool()
        missing = missing.expand(B, n) if missing.dim() == 1 else missing
    if cc is not None:
        for ns in cc.pc.root_ns:
            vm = value_mask(cc, ns, inputs, B) if ns.is_input() else None
            if vm is not None:
                missing = missing.clone()
                missing[:, ns.scope.to_list()[0]] |= ~vm.cpu()
    return data.cpu().long(), missing


def value_mask(cc, ns, inputs, B):
    """[B] bool: soft evidence's value mask at the variable of ``ns`` (a SoftEvidenceCategorical node group) for every
    sample, from ``inputs``; None without one."""
    vm = None if inputs is None else inputs.get("soft_evidence_value_mask")
    if vm is None or not isinstance(ns.dist, SoftEvidenceCategorical):
        return None
    src = ns.get_source_ns()
    layer = next(layer for layer in cc.pc.input_layer_group if src in layer.nodes)
    return vm[:, int(layer.var_idmapping[ns.scope.to_list()[0]])].to(cc.pc.params.device).expand(B)


def soft_weights(cc, ns, inputs, B):
    """[B, V] float64: the soft evidence's weight of every token at the variable of ``ns`` (a SoftEvidenceCategorical
    node group) for every sample, from ``inputs`` (as ``pc(x, ...)`` takes them): dense log-weights over the first
    K tokens (the others 0), or top-k (``soft_evidence_cat_ids``: the candidates' ids, the rest 0); None without."""
    ev = None if inputs is None else inputs.get("categorical_evidence_logp")
    if ev is None:
        return None
    src = ns.get_source_ns()
    layer = next(layer for layer in cc.pc.input_layer_group if src in layer.nodes)
    lv = int(layer.var_idmapping[ns.scope.to_list()[0]])
    V, dev = ns.dist.num_cats, cc.pc.params.device
    e = ev[:, lv].to(dev, torch.float64).expand(B, -1)                         # [B, K] log-weights
    w = torch.zeros(B, V, dtype = torch.float64, device = dev)
    ids = inputs.get("soft_evidence_cat_ids")
    if ids is None:
        w[:, :e.size(1)] = torch.exp(e)
    else:
        w.scatter_(1, ids[:, lv].to(dev).long().expand(B, -1), torch.exp(e))
    return w


def emission(cc, ns, inputs, B, observed = False):
    """[num_nodes, B, V]: every sample's token probabilities at the input nodes ``ns`` -- a Categorical's, or under
    soft evidence (``inputs``) the product of experts ``q(v) = p(v) w(v) / Z``, in float64 (products of tiny weights are
    subnormal in float32, with too few digits: a class of one unlikely token off by nats at a 126k vocabulary). Under
    a value mask, the library's unnormalized ``p(v) w(v)`` for an ``observed`` token (the mask True there) and where
    the mask is False (a free position)."""
    probs = categorical_probs(cc.pc, ns)                                        # [N, V]
    w = soft_weights(cc, ns, inputs, B) if isinstance(ns.dist, SoftEvidenceCategorical) else None
    if w is None:
        return probs[:, None, :].expand(-1, B, -1)
    pw = probs[:, None, :].double() * w[None, :, :]
    q = pw / pw.sum(dim = 2, keepdim = True)
    vm = value_mask(cc, ns, inputs, B)
    if vm is None:
        return q
    return torch.where((torch.ones_like(vm) if observed else ~vm)[None, :, None], pw, q)


def input_block(cc, ns, data, missing, inputs = None):
    """[num_nodes, B, W_t, W_{t+1}] block of an input node at variable t: Categorical, or SoftEvidenceCategorical whose
    every sample's emission is the product of experts ``q(v) = p(v) w(v) / Z`` under the evidence in ``inputs`` (as
    ``pc(x, ...)`` takes it). The per-class masses below are sums of those token probabilities."""
    if not isinstance(ns.dist, (Categorical, SoftEvidenceCategorical)):
        raise NotImplementedError(f"The reference supports Categorical input nodes only, got {type(ns.dist).__name__} "
                                  f"over variable {ns.scope.to_list()}.")
    layout, dev = cc.backend.layout, cc.pc.params.device
    t = ns.scope.to_list()[0]
    token_class = layout.token_class.to(dev)
    C = layout.num_classes
    N, B = ns.num_nodes, data.size(0)
    q = emission(cc, ns, inputs, B)                                             # [N, B, V]

    # log mass per token class: sum over the tokens of the class, or only the observed token
    mass = torch.zeros(N, B, C, dtype = q.dtype, device = dev).index_add_(2, token_class, q)
    obs = ~missing[:, t]
    if obs.any():
        x = data[obs, t].to(dev)
        o = obs.to(dev)
        qo = emission(cc, ns, inputs, B, observed = True)
        m_obs = torch.zeros(N, int(obs.sum()), C, dtype = q.dtype, device = dev)
        m_obs.scatter_(2, token_class[x][None, :, None].expand(N, -1, 1),
                       qo[:, o].gather(2, x[None, :, None].expand(N, -1, 1)))
        mass[:, o] = m_obs
    lmass = torch.log(mass).to(DTYPE)                                           # [N, B, C]

    # class c moves column i (boundary t) to column next_col[t, i, c] (boundary t + 1)
    Wi, Wj = int(layout.width[t]), int(layout.width[t + 1])
    T = torch.zeros(Wi, C, Wj, dtype = DTYPE, device = dev)
    nc = layout.next_col[t, :Wi].to(dev)                                        # [Wi, C]
    ii, cc_ = torch.nonzero(nc >= 0, as_tuple = True)
    T[ii, cc_, nc[ii, cc_]] = 1.0
    s = _shift(lmass, -1)
    return torch.log(torch.einsum("nbc,icj->nbij", torch.exp(lmass - s), T)) + s[..., None]


def reference_inside(cc, data, missing_mask = None, inputs = None) -> dict:
    """
    Every node group's block ``[num_nodes, B, W_a, W_{b+1}]`` of log inside values (see the module docstring), by
    node group. The constraint must be satisfiable (every boundary is empty otherwise).

    :param cc: a constrained circuit
    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True = marginalized
    :param inputs: None, or per-sample inputs as ``pc(x, ...)`` takes them (soft evidence, see :func:`input_block`)
    """
    pc, layout = cc.pc, cc.backend.layout
    assert layout.satisfiable
    data, missing = _evidence(data, missing_mask, cc.n, cc, inputs)
    B = data.size(0)
    dev = pc.params.device
    vals = {}
    for ns in pc.root_ns:                                                       # children before parents
        info = cc.structure.node(ns)
        (a, b), = info.scope_runs
        if ns.is_input():
            v = input_block(cc, ns, data, missing, inputs)
        elif ns.is_prod():
            order = sorted(range(len(ns.chs)), key = lambda k: cc.structure.node(ns.chs[k]).scope_runs[0][0])
            v = None
            for k in order:
                child = vals[ns.chs[k]][product_child_index(ns, k).to(dev)]
                v = child if v is None else log_matmul(v, child)
        else:
            X = torch.cat([vals[cs] for cs in ns.chs], dim = 0)
            v = log_weighted_sum(node_weights(cc, ns, inputs, B), X)
        assert v.shape == (ns.num_nodes, B, int(layout.width[a]), int(layout.width[b + 1])), (v.shape, a, b)
        vals[ns] = v
    return vals


def reference_marginal(cc, data, missing_mask = None, inputs = None) -> torch.Tensor:
    """
    log p(C, e) for every sample, in float32. The PC's input nodes must be Categorical or SoftEvidenceCategorical.

    :param cc: a constrained circuit
    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True = marginalized
    :returns: [B, num_root_nodes]
    """
    pc = cc.pc
    if not cc.backend.layout.satisfiable:
        return torch.full((data.size(0), pc.root_ns.num_nodes), -float("inf"), dtype = DTYPE, device = pc.params.device)
    root = reference_inside(cc, data, missing_mask, inputs)[pc.root_ns]                 # [N_root, B, 1, W_n]
    return torch.logsumexp(root[:, :, 0, :], dim = -1).t()


def _log_index_add(dest, idx, src):
    """``dest[idx[k]] = log(exp(dest[idx[k]]) + exp(src[k]))`` for every ``k``, repeated indices adding up."""
    m = dest.scatter_reduce(0, idx.view(-1, *[1] * (src.dim() - 1)).expand_as(src), src, "amax")   # at least dest
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    acc = torch.exp(dest - m).index_add_(0, idx, torch.exp(src - m[idx]))
    return torch.log(acc) + m


def reference_top_down(cc, data, missing_mask = None, inside = None, inputs = None) -> dict:
    """
    Every node group's block of log OUTSIDE values, of the same shape as its inside block, by node group: ``O[node,
    sample, i, j]`` is the log of the derivative of ``p(C, e)`` with respect to the node's (linear) inside value at
    ``(i, j)``. Top-down from the root, 0 on every column (its value is summed over the accepting columns):

    * a sum passes ``W^T`` times its outside to its children;
    * a product passes child ``m`` the product ``L^T @ O @ R^T``, ``L`` and ``R`` the products of its children
      before and after ``m`` in scope order;
    * the contributions of every parent, and of every product node reading the same child node, add up.

    Single-root PCs only (each root has its own outside values otherwise).

    :param inside: :func:`reference_inside` of the same arguments, if already computed
    :param inputs: per-sample inputs, as :func:`reference_inside` takes them
    """
    pc = cc.pc
    if pc.root_ns.num_nodes != 1:
        raise NotImplementedError(f"The reference computes outside values of single-root PCs only, got "
                                  f"{pc.root_ns.num_nodes} root nodes.")
    if inside is None:
        inside = reference_inside(cc, data, missing_mask, inputs)
    dev = pc.params.device
    outs = {ns: torch.full_like(v, -float("inf")) for ns, v in inside.items()}
    outs[pc.root_ns] = torch.zeros_like(inside[pc.root_ns])
    T = lambda x: x.transpose(-1, -2)
    for ns in reversed(list(pc.root_ns)):                                       # parents before children
        O = outs[ns]
        if ns.is_sum():
            W = node_weights(cc, ns, inputs, O.size(1))
            O_ch = log_weighted_sum(W.transpose(-1, -2), O)                     # children's nodes, concatenated
            k = 0
            for cs in ns.chs:
                outs[cs] = torch.logaddexp(outs[cs], O_ch[k:k + cs.num_nodes])
                k += cs.num_nodes
        elif ns.is_prod():
            order = sorted(range(len(ns.chs)), key = lambda k: cc.structure.node(ns.chs[k]).scope_runs[0][0])
            idx = [product_child_index(ns, k).to(dev) for k in order]
            blocks = [inside[ns.chs[k]][i] for k, i in zip(order, idx)]
            left = [None]                                                       # left[m]: blocks[0] @ .. @ blocks[m-1]
            for b in blocks[:-1]:
                left.append(b if left[-1] is None else log_matmul(left[-1], b))
            right = [None]                                                      # right[m]: blocks[m+1] @ .. (reversed)
            for b in reversed(blocks[1:]):
                right.append(b if right[-1] is None else log_matmul(b, right[-1]))
            right.reverse()
            for m, (k, i) in enumerate(zip(order, idx)):
                o = O if left[m] is None else log_matmul(T(left[m]), O)
                o = o if right[m] is None else log_matmul(o, T(right[m]))
                outs[ns.chs[k]] = _log_index_add(outs[ns.chs[k]], i, o)
    return outs


def _targets(targets, missing):
    """[B, n] bool: the target (sample, position) pairs, every missing one by default; each must be missing."""
    if targets is None:
        return missing.clone()
    targets = targets.cpu().bool()
    targets = targets.expand(missing.shape) if targets.dim() == 1 else targets
    if (targets & ~missing).any():
        raise ValueError("Every target must be a missing position.")
    return targets


def reference_conditional(cc, data, missing_mask = None, targets = None, inputs = None) -> torch.Tensor:
    """
    ``log p(x_t = v | e, C)`` for every target ``(b, t)`` and token ``v``, in float32: one row per target, in the
    order ``dense[targets]`` lists them (sample by sample, positions in order). Rows are -inf where ``p(e, C) = 0``.
    The PC's input nodes must be Categorical or SoftEvidenceCategorical (``inputs``: its soft evidence, see
    :func:`input_block`; ``p_u`` below is then every sample's ``q``).

    Token ``v`` at ``t`` moves column ``i`` to ``next_col[t, i, class(v)]`` with probability ``p_u(v)`` at every input
    node ``u`` over ``t``, so ``p(x_t = v, e, C) = sum_u p_u(v) A_u[class(v)]``, with the class weights
    ``A_u[c] = sum_i exp(O_u[i, next_col[t, i, c]])`` of the input node's outside block; normalized over ``v``.

    :param cc: a constrained circuit
    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True = marginalized
    :param targets: None (every missing position of every sample), or a bool mask [n] or [B, n] of missing positions
    :returns: [number of targets, V]
    """
    pc, layout = cc.pc, cc.backend.layout
    n, dev = cc.n, pc.params.device
    _, missing = _evidence(data, missing_mask, n, cc, inputs)
    targets = _targets(targets, missing)
    V = layout.token_class.numel()
    if not layout.satisfiable:
        return torch.full((int(targets.sum()), V), -float("inf"), dtype = DTYPE, device = dev)
    inside = reference_inside(cc, data, missing_mask, inputs)
    top_down = reference_top_down(cc, data, missing_mask, inside, inputs)

    token_class = layout.token_class.to(dev)
    scores = torch.full((missing.size(0), n, V), -float("inf"), dtype = DTYPE, device = dev)   # log p(x_t = v, e, C)
    wanted = targets.any(dim = 0)
    for ns, O in top_down.items():
        if not ns.is_input():
            continue
        t = ns.scope.to_list()[0]
        if not wanted[t]:
            continue
        if not isinstance(ns.dist, (Categorical, SoftEvidenceCategorical)):
            raise NotImplementedError(f"The reference supports Categorical input nodes only, got "
                                      f"{type(ns.dist).__name__} over variable {t}.")
        Wi = int(layout.width[t])
        nc = layout.next_col[t, :Wi].to(dev)                                    # [Wi, C]
        G = O[:, :, torch.arange(Wi, device = dev)[:, None], nc.clamp(min = 0)] # [N, B, Wi, C]
        A = torch.logsumexp(G.masked_fill(nc < 0, -float("inf")), dim = 2)      # [N, B, C] log class weights
        logp = torch.log(emission(cc, ns, inputs, missing.size(0))).to(DTYPE)  # [N, B, V]
        s = torch.logsumexp(A[:, :, token_class] + logp, dim = 0)               # [B, V]
        scores[:, t] = torch.logaddexp(scores[:, t], s)

    rows = scores[targets.to(dev)]
    norm = torch.logsumexp(rows, dim = -1, keepdim = True)                      # log p(e, C)
    return torch.where(torch.isfinite(norm), rows - norm, torch.full_like(rows, -float("inf")))


@pytest.fixture
def reference():
    """The reference implementation (this module's functions), for tests in this directory."""
    import types
    return types.SimpleNamespace(marginal = reference_marginal, inside = reference_inside,
                                 top_down = reference_top_down, conditional = reference_conditional,
                                 sum_weights = sum_weights, categorical_probs = categorical_probs,
                                 input_block = input_block, product_child_index = product_child_index,
                                 log_matmul = log_matmul, log_weighted_sum = log_weighted_sum)


# -------------------------------------------------------------------------------------------------
# PCs shared by the tests in this directory
# -------------------------------------------------------------------------------------------------

def hand_built(V):
    """Every construct a contiguous PC can have, over 5 variables."""
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = dists.Categorical(num_cats = V)) for v in range(5)]
    s12 = summate(multiply(x[1], x[2]), num_node_blocks = 2, block_size = 2)          # interval [1, 2]
    s01 = summate(multiply(x[0], x[1]), num_node_blocks = 2, block_size = 2)          # prefix [0, 1]
    s3 = summate(x[3], num_node_blocks = 2, block_size = 2)                           # a sum over an input node
    p34 = multiply(s3, x[4], edge_ids = torch.tensor([[0, 3], [1, 2], [2, 1], [3, 0]]),
                   sparse_edges = True)                                               # node-level edges
    s34 = summate(p34, num_node_blocks = 2, block_size = 2)                           # suffix [3, 4]
    p_a = multiply(s34, x[0], s12)                                                    # three children, listed
    p_b = multiply(x[2], s34, s01)                                                    # out of scope order
    s02 = summate(multiply(x[0], s12), num_node_blocks = 2, block_size = 2)          # an input node next to an
    p_c = multiply(s02, s34)                                                          # interval node, [0, 2]
    return summate(p_a, p_b, p_c, num_node_blocks = 1, block_size = 1)               # a sum over three products


def hand_permuted(V):
    """Block-level product edges that permute blocks or reuse a child's only block, explicit block-sparse
    sum edges with an edge block listed twice, and a four-child product, over 5 variables."""
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = cat()) for v in range(5)]
    one = inputs(1, num_node_blocks = 1, block_size = 2, dist = cat())             # a single block
    p12 = multiply(one, x[2], edge_ids = torch.tensor([[0, 1], [0, 0]]))            # reuses it, permutes x2's
    s12 = summate(p12, num_node_blocks = 2, block_size = 2,
                  edge_ids = torch.tensor([[0, 0, 1], [1, 1, 0]]))                  # (0, 1) twice
    p34 = multiply(x[3], x[4], edge_ids = torch.tensor([[1, 0], [0, 1]]))           # permuted blocks
    s34 = summate(p34, num_node_blocks = 2, block_size = 2)
    p_a = multiply(x[0], s12, s34)
    p_b = multiply(s34, x[2], x[1], x[0], edge_ids = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]]))   # four children
    return summate(p_a, p_b, num_node_blocks = 1, block_size = 1)


def hand_unit(V):
    """Block size 1 everywhere, with node-level product edges, over 4 variables."""
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 3, block_size = 1, dist = cat()) for v in range(4)]
    p01 = multiply(x[0], x[1], edge_ids = torch.tensor([[0, 2], [1, 1], [2, 0]]), sparse_edges = True)
    p23 = multiply(x[2], x[3], edge_ids = torch.tensor([[2, 1], [0, 0], [1, 2]]), sparse_edges = True)
    s01 = summate(p01, num_node_blocks = 3, block_size = 1)
    s23 = summate(p23, num_node_blocks = 3, block_size = 1)
    return summate(multiply(s01, s23), num_node_blocks = 1, block_size = 1)


def hand_left(V):
    """Left-linear chains, over 5 variables. Prefix sums times the next input (`block @ input` before and at the
    sequence end), and a three-child product with inputs at both ends, whose first step writes scratch."""
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = cat()) for v in range(5)]
    pre = summate(x[0], num_node_blocks = 2, block_size = 2)                          # [0, 0]
    for v in (1, 2, 3):
        pre = summate(multiply(pre, x[v]), num_node_blocks = 2, block_size = 2)       # [0, v]: a prefix, then an input
    s12 = summate(multiply(x[1], x[2]), num_node_blocks = 2, block_size = 2)          # [1, 2]
    s03 = summate(multiply(x[0], s12, x[3]), num_node_blocks = 2, block_size = 2)     # [0, 3]: inputs at both ends
    return summate(multiply(pre, x[4]), multiply(s03, x[4]), num_node_blocks = 1, block_size = 1)


def hand_shared(V):
    """Sums of one layer over overlapping sets of products of sums, over 4 variables: ``a01`` over ``p01``, ``c01``
    over ``r01`` and ``b01`` over ``p01``, ``q01`` and ``r01``, whose children are two runs of rows (``r01``'s come
    first, ``c01`` being first), the first starting in rows ``a01`` reads too. Where the dense path is taken, ``a01``
    and ``c01`` are dense and ``b01`` is not."""
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = cat()) for v in range(4)]
    s0, s0b, s1 = (summate(x[v], num_node_blocks = 2, block_size = 2) for v in (0, 0, 1))
    p01 = multiply(s0, s1)
    q01 = multiply(s0b, s1)
    r01 = multiply(s0, s1, edge_ids = torch.tensor([[1, 1], [0, 0]]))
    c01 = summate(r01, num_node_blocks = 1, block_size = 2)
    b01 = summate(p01, q01, r01, num_node_blocks = 1, block_size = 2)
    a01 = summate(p01, num_node_blocks = 2, block_size = 2)
    s23 = summate(multiply(x[2], x[3]), num_node_blocks = 2, block_size = 2)
    one = lambda s: multiply(s, s23, edge_ids = torch.tensor([[0, 0], [0, 1]]))     # a one-block sum next to s23
    return summate(one(c01), one(b01), multiply(a01, s23), num_node_blocks = 1, block_size = 1)


def hand_fold(V):
    """A product of three sums, ``s01 @ (s2 @ s3)``, over 4 variables, and nothing else: no other sum covers the fold
    of ``s2`` and ``s3``, which ``s01``'s top-down value reads -- computed for a batch of one when that value is
    input-independent (nothing observed after position 1), reading ``s2`` and ``s3`` once."""
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = cat()) for v in range(4)]
    s01 = summate(multiply(x[0], x[1]), num_node_blocks = 2, block_size = 2)
    s2, s3 = (summate(x[v], num_node_blocks = 2, block_size = 2) for v in (2, 3))
    return summate(multiply(s01, s2, s3), num_node_blocks = 1, block_size = 1)


def hand_dead(V):
    """``x0`` times a sum over [1, 3), over 3 variables, under a root whose block-sparse edges read the products' first
    block only: the second block's steps have no top-down value, so the class weights of ``x0``'s second block are
    written by no launch (their rows of ``dfa_class_flow`` must hold -inf)."""
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 2, block_size = 2, dist = cat()) for v in range(3)]
    s12 = summate(multiply(x[1], x[2]), num_node_blocks = 2, block_size = 2)
    return summate(multiply(x[0], s12), num_node_blocks = 1, block_size = 1, edge_ids = torch.tensor([[0], [0]]))


def walks(V, unit = False):
    """``input @ block`` steps over 6 variables that reach the CUDA walk kernels (``kernels/c/walk.cu``) from every
    side: ``s3`` under two products of different inputs at position 2 (two parents per child), ``x2`` under two
    products of different sums over [3, 4) (two steps per class weight destination), over intervals ending before the
    sequence ([3, 4) and [2, 4): pairs the automaton does not join), in a three-child product (folds); ``unit``: blocks
    of one node, every node tile one row."""
    nb, bs = (1, 1) if unit else (2, 2)
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = nb, block_size = bs, dist = cat()) for v in range(6)]
    x2 = inputs(2, num_node_blocks = nb, block_size = bs, dist = cat())
    s3, s3b = (summate(x[3], num_node_blocks = nb, block_size = bs) for _ in range(2))
    s24 = summate(multiply(x[2], s3), multiply(x2, s3), multiply(x2, s3b), num_node_blocks = nb, block_size = bs)
    s01 = summate(multiply(x[0], x[1]), num_node_blocks = nb, block_size = bs)
    s45 = summate(multiply(x[4], x[5]), num_node_blocks = nb, block_size = bs)
    return summate(multiply(s01, s24, s45), num_node_blocks = 1, block_size = 1)


def left_linear(n, V, num_latents = 8):
    """The mirror of an HMM: a sum over every prefix, each over the previous prefix times the next input (`block @
    input`, whose left blocks have the initial state as their only entry column), every input tied to the first."""
    x0 = inputs(0, num_node_blocks = 1, block_size = num_latents, dist = dists.Categorical(num_cats = V))
    pre = summate(x0, num_node_blocks = 1, block_size = num_latents)                  # [0, 0]
    for v in range(1, n):
        x = x0.duplicate(v, tie_params = True)
        pre = summate(multiply(pre, x), num_node_blocks = 1, block_size = num_latents if v < n - 1 else 1)
    return pre


#: Every kind :func:`build_pc` makes, with the number of variables its circuit needs (None: any).
PC_KINDS = {
    "hmm": None,                 # tied HMM, one node block per position
    "hmm_untied": None,
    "hmm_block_sparse": None,    # 4 node blocks per position, random block-sparse transitions, tied
    "left_linear": None,         # :func:`left_linear`: prefix sums, inputs tied
    "pd": None,                  # 1-D PD
    "pd_prod_dominated": None,   # sums shared by several products, block-sparse sum edges
    "pd_blockified": None,       # `juice.blockify` of a block-size-1 PD
    "hand": 5,                   # :func:`hand_built`
    "hand_permuted": 5,          # :func:`hand_permuted`
    "hand_unit": 4,              # :func:`hand_unit`
    "hand_left": 5,              # :func:`hand_left`
    "hand_shared": 4,            # :func:`hand_shared`
    "hand_fold": 4,              # :func:`hand_fold`
    "hand_dead": 3,              # :func:`hand_dead`
    "walks": 6,                  # :func:`walks`
    "walks_unit": 6,             # :func:`walks`, blocks of one node
}


def gate_sums(root, child_gate = lambda ns: ns.ch_block_size, tie_external = False, which = lambda ns: True,
              node_gate = lambda ns: ns.block_size):
    """Every sum node group of the circuit under ``root`` that ``which`` picks made gated, before it is compiled: a
    :class:`BlockScaleSumParams` with ``node_gate(ns)`` nodes per node gate (default: one per node block) and
    ``child_gate(ns)`` children per child gate."""
    from pyjuice.nodes import BlockScaleSumParams
    from pyjuice.nodes.external_sum_nodes import ExternalParamsSumNodes
    for ns in root:
        if ns.is_sum() and which(ns):
            ns.__class__ = ExternalParamsSumNodes
            node = node_gate(ns)
            ns.external_params = BlockScaleSumParams(block_size = None if node == ns.block_size else node,
                                                     ch_block_size = child_gate(ns), tie_external = tie_external)
            ns.external_params.validate_ns(ns)
    return root


def build_pc(kind, n, V, seed = 0, device = torch.device("cuda:0"), perturbation = 2.0, input_dist = None,
             gates = None, **compile_kwargs):
    """A compiled PC of one of the :data:`PC_KINDS` over ``n`` variables with ``V`` categories (``perturbation``: of
    its parameters' initialization, see ``init_parameters``; ``input_dist``: None, or another distribution class of
    ``V`` categories every input node takes, e.g. ``SoftEvidenceCategorical``; ``gates``: None, or keywords of
    :func:`gate_sums` -- its sums gated)."""
    assert PC_KINDS[kind] in (None, n), (kind, n)
    torch.manual_seed(seed); random.seed(seed)
    if kind == "hmm":
        ns = juice.structures.HMM(seq_length = n, num_latents = 8, num_emits = V)
    elif kind == "hmm_untied":
        ns = juice.structures.GeneralizedHMM(seq_length = n, num_latents = 8, homogeneous = False,
                                             input_dist = dists.Categorical(num_cats = V))
    elif kind == "hmm_block_sparse":
        ns = juice.structures.HMM(seq_length = n, num_latents = 16, num_emits = V, block_size = 4,
                                  sum_edge_ids_constructor = partial(block_sparse_rnd_blk_edge_constructor,
                                                                     num_chs_per_block = 2))
    elif kind == "left_linear":
        ns = left_linear(n, V)
    elif kind == "pd":
        ns = juice.structures.PD(data_shape = (n,), num_latents = 4, split_intervals = 1,
                                 input_node_params = {"num_cats": V})
    elif kind == "pd_prod_dominated":
        ns = juice.structures.PD(data_shape = (n,), num_latents = 8, split_intervals = 1, block_size = 2,
                                 structure_type = "prod_dominated", max_prod_block_conns = 2,
                                 input_node_params = {"num_cats": V})
    elif kind == "pd_blockified":
        base = juice.structures.PD(data_shape = (n,), num_latents = 8, split_intervals = 1, block_size = 1,
                                   input_node_params = {"num_cats": V})
        base.init_parameters(perturbation = 2.0)
        ns = juice.blockify(base, sparsity_tolerance = 0.5, max_target_block_size = 4)
    else:
        ns = {"hand": hand_built, "hand_permuted": hand_permuted, "hand_unit": hand_unit, "hand_left": hand_left,
              "hand_shared": hand_shared, "hand_fold": hand_fold, "hand_dead": hand_dead, "walks": walks,
              "walks_unit": lambda V: walks(V, unit = True)}[kind](V)
    if input_dist is not None:                          # the same circuit over another input distribution
        for node in ns:
            if node.is_input():
                node.dist = input_dist(num_cats = V)
    if gates is not None:
        gate_sums(ns, **gates)
    ns.init_parameters(perturbation = perturbation)
    return juice.compile(ns, verbose = False, **compile_kwargs).to(device)


@pytest.fixture(name = "build_pc")
def build_pc_fixture():
    """:func:`build_pc`, for tests in this directory."""
    return build_pc


@pytest.fixture(name = "gate_sums")
def gate_sums_fixture():
    """:func:`gate_sums`, for tests in this directory."""
    return gate_sums


_SPLIT_DEFAULTS = (_lifted_sum.SPLIT_PROGRAMS_PER_SM, _lifted_sum.SPLIT_MIN_SHARE, _lifted_sum.SPLIT_MAX)


def set_split(monkeypatch, mode: str):
    """The fused sum kernels' split reductions (``kernels/sum.py``): ``"default"`` the library's rule, ``"max"``
    every launch split as far as its reductions allow (shares of one chunk, whatever the launch's size), ``"off"``
    none; any mode after any other."""
    per_sm, min_share, most = _SPLIT_DEFAULTS
    values = {"default": (per_sm, min_share, most), "max": (1 << 20, 1, most), "off": (per_sm, min_share, 1)}[mode]
    for name, value in zip(("SPLIT_PROGRAMS_PER_SM", "SPLIT_MIN_SHARE", "SPLIT_MAX"), values):
        monkeypatch.setattr(_lifted_sum, name, value)


@pytest.fixture(params = ["default", "max"])
def split(request, monkeypatch):
    """:func:`set_split`: the library's rule, or every launch split as far as its reductions allow."""
    set_split(monkeypatch, request.param)
    return request.param


@pytest.fixture(name = "set_split")
def set_split_fixture(monkeypatch):
    """:func:`set_split` for a test that switches modes."""
    return lambda mode: set_split(monkeypatch, mode)


@pytest.fixture(params = ["default", "always"])
def pre(request, monkeypatch):
    """The fused top-down kernel's pre-pass (``kernels/sum.py``): the library's rule, or every launch that may take
    it (any number of tiles of children)."""
    if request.param == "always":
        monkeypatch.setattr(_lifted_sum, "PRE_MIN_TILES", 1)
    return request.param
