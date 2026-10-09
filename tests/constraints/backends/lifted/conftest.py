"""
A slow but obviously correct reference for constrained queries on the lifted plan, for tests only.

``reference_marginal(cc, data, missing_mask)`` computes ``log p(C, e)`` for a
:class:`~pyjuice.constraints.ConstrainedCircuit` in plain PyTorch, straight from the definition, for
every (PC, constraint) pair that compiles. The library's own implementation is tested against it; it is
exposed to the tests in this directory through the ``reference`` fixture (the test suite runs with
``--import-mode=importlib``, so test modules cannot import a helper module directly).

INPUT NODES: CATEGORICAL ONLY. Like the first version of the compiler (which refuses every other input
distribution), the reference only knows how to turn a Categorical input node into per-token-class
masses; any other input node raises ``NotImplementedError``. Other distributions need their own
class-mass rule here.

PRECISION: float32, the precision the library runs at. Every log-space step is max-shifted, so the
reference's own rounding stays around 1e-7 relative -- far below the tolerances it is compared at.

Every node over the contiguous scope ``[a, b]`` gets a block of log-values

    V[node, sample, i, j] = log sum over the evidence-consistent x_a..x_b of
                            p_node(x_a..x_b) * 1[the automaton goes from column i to column j reading them]

where ``i`` is a column at boundary ``a`` and ``j`` a column at boundary ``b + 1`` (columns are the
active automaton states, see :mod:`pyjuice.constraints.backends.lifted.plan`). Only active columns are
kept: an accepted string of length ``n`` passes through active states only. Then

* an input node at variable ``t`` sums its token probabilities per token class, and class ``c`` moves column
  ``i`` to ``next_col[t, i, c]``;
* a product chains its children's blocks in scope order (a matrix product per node and sample);
* a sum takes its weighted sum over children, block by block;
* the root is ``log sum_j V[root, sample, 0, j]``: boundary 0 holds only the initial state, and every
  active state at boundary ``n`` is accepting.

Parameters are read from the PC's live parameter tensors (never from the node objects, which are not
kept in sync), so in-place parameter updates are seen by the next call.
"""

import pytest
import torch

from pyjuice.nodes.distributions import Categorical


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
    dense[edge_ids[0] * num_ch_nblocks + edge_ids[1]] = blocks
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
    """log(Wt @ exp(X)) over the first dim of X: Wt [N, K], X [K, ...] -> [N, ...]."""
    s = _shift(X, 0)
    out = Wt @ torch.exp(X - s).reshape(X.size(0), -1)
    return torch.log(out).reshape(Wt.size(0), *X.shape[1:]) + s


# -------------------------------------------------------------------------------------------------
# The reference
# -------------------------------------------------------------------------------------------------

def _evidence(data, missing_mask, n):
    B = data.size(0)
    if missing_mask is None:
        missing = torch.zeros(B, n, dtype = torch.bool)
    else:
        missing = missing_mask.cpu().bool()
        missing = missing.expand(B, n) if missing.dim() == 1 else missing
    return data.cpu().long(), missing


def input_block(cc, ns, data, missing):
    """[num_nodes, B, W_t, W_{t+1}] block of an input node at variable t. CATEGORICAL ONLY: the per-class
    masses below are sums of Categorical token probabilities."""
    if not isinstance(ns.dist, Categorical):
        raise NotImplementedError(f"The reference supports Categorical input nodes only, got {type(ns.dist).__name__} "
                                  f"over variable {ns.scope.to_list()}.")
    layout, dev = cc.layout, cc.pc.params.device
    t = ns.scope.to_list()[0]
    probs = categorical_probs(cc.pc, ns)                                        # [N, V]
    token_class = layout.token_class.to(dev)
    C = layout.num_classes
    N, B = ns.num_nodes, data.size(0)

    # log mass per token class: sum over the tokens of the class, or only the observed token
    free = torch.zeros(N, C, dtype = DTYPE, device = dev).index_add_(1, token_class, probs)
    mass = free[:, None, :].expand(N, B, C).clone()
    obs = ~missing[:, t]
    if obs.any():
        x = data[obs, t].to(dev)
        m_obs = torch.zeros(N, int(obs.sum()), C, dtype = DTYPE, device = dev)
        m_obs.scatter_(2, token_class[x][None, :, None].expand(N, -1, 1), probs[:, x][:, :, None])
        mass[:, obs.to(dev)] = m_obs
    lmass = torch.log(mass)                                                     # [N, B, C]

    # class c moves column i (boundary t) to column next_col[t, i, c] (boundary t + 1)
    Wi, Wj = int(layout.width[t]), int(layout.width[t + 1])
    T = torch.zeros(Wi, C, Wj, dtype = DTYPE, device = dev)
    nc = layout.next_col[t, :Wi].to(dev)                                        # [Wi, C]
    ii, cc_ = torch.nonzero(nc >= 0, as_tuple = True)
    T[ii, cc_, nc[ii, cc_]] = 1.0
    s = _shift(lmass, -1)
    return torch.log(torch.einsum("nbc,icj->nbij", torch.exp(lmass - s), T)) + s[..., None]


def reference_marginal(cc, data, missing_mask = None) -> torch.Tensor:
    """
    log p(C, e) for every sample, in float32. The PC's input nodes must be Categorical.

    :param cc: a constrained circuit
    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True = marginalized
    :returns: [B, num_root_nodes]
    """
    pc, layout = cc.pc, cc.layout
    n = cc.n
    data, missing = _evidence(data, missing_mask, n)
    B = data.size(0)
    dev = pc.params.device
    if not layout.satisfiable:
        return torch.full((B, pc.root_ns.num_nodes), -float("inf"), dtype = DTYPE, device = dev)

    vals = {}
    for ns in pc.root_ns:                                                       # children before parents
        info = cc.structure.node(ns)
        (a, b), = info.scope_runs
        if ns.is_input():
            v = input_block(cc, ns, data, missing)
        elif ns.is_prod():
            order = sorted(range(len(ns.chs)), key = lambda k: cc.structure.node(ns.chs[k]).scope_runs[0][0])
            v = None
            for k in order:
                child = vals[ns.chs[k]][product_child_index(ns, k).to(dev)]
                v = child if v is None else log_matmul(v, child)
        else:
            X = torch.cat([vals[cs] for cs in ns.chs], dim = 0)
            v = log_weighted_sum(sum_weights(pc, ns), X)
        assert v.shape == (ns.num_nodes, B, int(layout.width[a]), int(layout.width[b + 1])), (v.shape, a, b)
        vals[ns] = v

    root = vals[pc.root_ns]                                                     # [N_root, B, 1, W_n]
    return torch.logsumexp(root[:, :, 0, :], dim = -1).t()


@pytest.fixture
def reference():
    """The reference implementation (this module's functions), for tests in this directory."""
    import types
    return types.SimpleNamespace(marginal = reference_marginal, sum_weights = sum_weights,
                                 categorical_probs = categorical_probs, input_block = input_block,
                                 log_matmul = log_matmul, log_weighted_sum = log_weighted_sum)
