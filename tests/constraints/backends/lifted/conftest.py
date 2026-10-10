"""
A slow but obviously correct reference for constrained queries on the lifted plan, for tests only.

``reference_marginal(cc, data, missing_mask)`` computes ``log p(C, e)`` for a
:class:`~pyjuice.constraints.ConstrainedCircuit` in plain PyTorch, straight from the definition, for
every (PC, constraint) pair that compiles. The library's own implementation is tested against it; it is
exposed to the tests in this directory through the ``reference`` fixture (the test suite runs with
``--import-mode=importlib``, so test modules cannot import a helper module directly). The PCs these tests
share come from the ``build_pc`` fixture.

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

import random
from functools import partial

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes.distributions import Categorical
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
                                 product_child_index = product_child_index,
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


#: Every kind :func:`build_pc` makes, with the number of variables its circuit needs (None: any).
PC_KINDS = {
    "hmm": None,                 # tied HMM, one node block per position
    "hmm_untied": None,
    "hmm_block_sparse": None,    # 4 node blocks per position, random block-sparse transitions, tied
    "pd": None,                  # 1-D PD
    "pd_prod_dominated": None,   # sums shared by several products, block-sparse sum edges
    "pd_blockified": None,       # `juice.blockify` of a block-size-1 PD
    "hand": 5,                   # :func:`hand_built`
    "hand_permuted": 5,          # :func:`hand_permuted`
    "hand_unit": 4,              # :func:`hand_unit`
    "hand_left": 5,              # :func:`hand_left`
}


def build_pc(kind, n, V, seed = 0, device = torch.device("cuda:0"), **compile_kwargs):
    """A compiled PC of one of the :data:`PC_KINDS` over ``n`` variables with ``V`` categories."""
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
        ns = {"hand": hand_built, "hand_permuted": hand_permuted, "hand_unit": hand_unit, "hand_left": hand_left}[kind](V)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, **compile_kwargs).to(device)


@pytest.fixture(name = "build_pc")
def build_pc_fixture():
    """:func:`build_pc`, for tests in this directory."""
    return build_pc
