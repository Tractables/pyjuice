import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, LowRankSumParams
from pyjuice.layer import ExternalParamsSumLayer, SumLayer

import pytest


def _build_single_layer(num_node_blocks, num_ch_blocks, block_size, rank, num_cats = 5, seed = 0):
    """
    The smallest PC containing one external low-rank sum layer: an input layer, one product layer, the
    external sum layer, and a root over it.

    With `edge_ids` left to the default the sum node is fully connected, so it has
    `num_node_blocks * num_ch_blocks` edge blocks.
    """

    torch.manual_seed(seed)

    with juice.set_block_size(block_size):

        ni = inputs(0, num_node_blocks = num_ch_blocks, dist = dists.Categorical(num_cats = num_cats))

        ns = summate(multiply(ni), num_node_blocks = num_node_blocks,
                     external_params = LowRankSumParams(rank = rank))

        root_ns = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root_ns.init_parameters(perturbation = 2.0)

    pc = juice.compile(root_ns)
    pc.to(torch.device("cuda:0"))

    layer = [layer for layer_group in pc.inner_layer_groups if layer_group.is_sum()
             for layer in layer_group.layers if isinstance(layer, ExternalParamsSumLayer)][0]

    return pc, root_ns, ns, layer


def _dense_reference(pc, ns, U, V, element_mars, log_s1 = None):
    """
    What the layer must compute, written out directly and densely.

    For every node `n` and sample `b`, with `theta_shared[c, n]` the shared parameters (indexed
    `[child, parent]`, normalized over children):

        theta[c, n]  =  ( theta_shared[c, n] + sum_r exp(U[b, e, c, r]) * exp(V[b, e, n, r]) )  /  Z
        Z            =  sum_c ( theta_shared[c, n] + correction[c, n] )
        node_mars    =  log sum_c theta[c, n] * exp(element_mars[c, b])

    i.e. a plain matmul of the parameters against the child probabilities, plus a matmul of the
    low-rank correction against the same child probabilities, renormalized. The last line is done with
    the max subtracted so that arbitrarily small child probabilities do not underflow, and the whole
    thing runs in float64 so the reference is not itself the source of any disagreement.

    Everything here is derived from `ns.edge_ids` and the child's own id range -- deliberately not from
    the layer's compiled index tensors, so this checks those too.

    :param log_s1: when given, the shared term `log sum_c theta_shared[c,n] * exp(element_mars[c,b])`
                   is taken from it (i.e. from what the PC's own shared kernel produced) instead of
                   being recomputed. Only the CORRECTION is then under test, so the comparison is not
                   limited by the shared kernel's float32 / bfloat16 precision:

                       node_mars = logaddexp(log_s1, log_s2) - logaddexp(0, log_zt)
                       log_s2 = log sum_c correction[c,n] * exp(element_mars[c,b])
                       log_zt = log sum_c correction[c,n]
    """
    device = element_mars.device

    edge_ids = ns.edge_ids
    block_size, ch_block_size = ns.block_size, ns.ch_block_size
    batch_size = element_mars.size(1)

    # [edge block, parent, child] -> the shared parameter tile of each edge block. A tied `ns` holds no
    # parameters of its own (`get_node_params` returns None for it by design), so resolve to the source
    # it shares them with -- that is the whole point of tying, and it is what the kernel reads via `pids`.
    theta_blocks = pc.get_node_params(ns.get_source_ns() if ns.is_tied() else ns).double().to(device)

    ch_start = ns.chs[0]._output_ind_range[0]

    node_mars = torch.zeros([ns.num_nodes, batch_size], dtype = torch.float64, device = device)

    for nblock_id in range(ns.num_node_blocks):
        eblk_ids = torch.nonzero(edge_ids[0,:] == nblock_id, as_tuple = False).flatten().tolist()

        # Stack this node block's children, edge block by edge block
        theta_shared, child_ids, factors = [], [], []
        for eblk_id in eblk_ids:
            ch_block_id = int(edge_ids[1, eblk_id])

            theta_shared.append(theta_blocks[eblk_id].t())                          # [Kc, K]
            child_ids.append(ch_start + ch_block_id * ch_block_size
                             + torch.arange(0, ch_block_size, device = device))
            factors.append((U[:,eblk_id,:,:].double(), V[:,eblk_id,:,:].double()))

        theta_shared = torch.cat(theta_shared, dim = 0)                             # [C, K]
        emars = element_mars[torch.cat(child_ids),:].double()                       # [C, B]

        for b in range(batch_size):
            # The rank-r correction of every edge block, stacked the same way
            correction = torch.cat([torch.exp(u[b]) @ torch.exp(v[b]).t() for u, v in factors], dim = 0)

            # Max-subtracted, so arbitrarily small child probabilities cannot underflow
            shift = emars[:,b].max()
            shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
            child = torch.exp(emars[:,b] - shift)                                   # [C]

            rows = slice(nblock_id * block_size, (nblock_id + 1) * block_size)

            if log_s1 is None:
                theta = theta_shared + correction
                theta = theta / theta.sum(dim = 0, keepdim = True)                  # normalize over children

                probs = (theta * child[:,None]).sum(dim = 0)                        # [K]

                node_mars[rows, b] = torch.log(probs) + shift

            else:
                # The correction's own contribution, combined with the shared term the PC produced.
                # `theta_shared` sums to 1 over children, so it contributes exactly 0 in log space.
                log_s2 = torch.log((correction * child[:,None]).sum(dim = 0)) + shift
                log_zt = torch.log(correction.sum(dim = 0))

                node_mars[rows, b] = torch.logaddexp(log_s1[rows, b].double(), log_s2) \
                                     - torch.logaddexp(torch.zeros_like(log_zt), log_zt)

    return node_mars


def _run_layer(pc, layer, node_mars, element_mars, staged, use_kernel):
    """
    Run just this layer's forward over the given buffers, with or without the Triton kernel.
    """
    from pyjuice.layer import EXTERNAL_PARAMS_BUFFER_KWARG, EXTERNAL_PARAMS_KWARG

    applicable = LowRankSumParams._kernel_applicable
    try:
        if not use_kernel:
            LowRankSumParams._kernel_applicable = lambda *args, **kwargs: False

        layer.forward(node_mars, element_mars, pc.params,
                      **{EXTERNAL_PARAMS_KWARG: staged,
                         EXTERNAL_PARAMS_BUFFER_KWARG: pc.external_params})
    finally:
        LowRankSumParams._kernel_applicable = applicable


def _stage(pc, ns, U, V, data):
    """Stage `(U, V)` for `ns` by running a forward, and hand back the staged views."""
    pc(data, sum_external_params = {ns: (U, V)})
    return pc._staged_external_params


@pytest.mark.parametrize("use_kernel", [True, False])
@pytest.mark.parametrize("num_node_blocks,num_ch_blocks,block_size,rank",
                         [(1, 1, 16, 4),      # one edge block -- a dense transition
                          (1, 3, 16, 4),      # several child blocks feeding one node block
                          (2, 3, 16, 8),      # several node blocks, several edge blocks each
                          (1, 1, 32, 16)])    # larger block, larger rank
def test_lowrank_layer_matches_dense_reference(num_node_blocks, num_ch_blocks, block_size, rank, use_kernel):
    """
    One external layer, checked against the dense formula it stands for.
    """

    device = torch.device("cuda:0")

    batch_size = 32

    pc, root_ns, ns, layer = _build_single_layer(num_node_blocks, num_ch_blocks, block_size, rank)

    num_edge_blocks = ns.edge_ids.size(1)

    torch.manual_seed(1)
    data = torch.randint(0, 5, [batch_size, 1]).to(device)

    U = torch.randn([batch_size, num_edge_blocks, ns.ch_block_size, rank], device = device) - 1.0
    V = torch.randn([batch_size, num_edge_blocks, block_size, rank], device = device) - 1.0

    staged = _stage(pc, ns, U, V, data)

    element_mars = pc.element_mars.clone()

    _run_layer(pc, layer, pc.node_mars, element_mars, staged, use_kernel)

    got = pc.node_mars[ns._output_ind_range[0]:ns._output_ind_range[1],:].double()

    # Against the fully dense float64 formula. The bound is the SHARED kernel's float32 / bfloat16
    # precision, which this comparison necessarily includes -- see the isolated test below for the
    # correction on its own.
    expected = _dense_reference(pc, ns, U, V, element_mars)

    assert torch.all(torch.isfinite(got))
    assert torch.all(torch.abs(got - expected) < 2e-3)

    # Against the same formula, but taking the shared term from the PC instead of recomputing it, so
    # that only the correction is under test
    shared_only = _shared_only_node_mars(pc, ns, layer, element_mars)
    expected_isolated = _dense_reference(pc, ns, U, V, element_mars, log_s1 = shared_only)

    assert torch.all(torch.abs(got - expected_isolated) < 1e-5)


def _shared_only_node_mars(pc, ns, layer, element_mars):
    """`log S1` -- what the standard sum layer alone writes for `ns`, on these child values."""
    node_mars = pc.node_mars.clone()

    SumLayer.forward(layer, node_mars, element_mars, pc.params)

    return node_mars[ns._output_ind_range[0]:ns._output_ind_range[1],:].clone()


@pytest.mark.parametrize("use_kernel", [True, False])
def test_lowrank_layer_underflow(use_kernel):
    """
    Child probabilities spanning many orders of magnitude, including exact zeros.

    `exp(-1e4)` is 0 in float32 and `exp(-inf)` is 0 exactly, so a kernel that exponentiated the child
    values before combining them would lose everything; only the max-subtracted form survives. The
    all-`-inf` node is the degenerate case where there is nothing to subtract.
    """

    device = torch.device("cuda:0")

    batch_size, block_size, rank = 32, 16, 4

    pc, root_ns, ns, layer = _build_single_layer(1, 3, block_size, rank)

    num_edge_blocks = ns.edge_ids.size(1)

    torch.manual_seed(2)
    data = torch.randint(0, 5, [batch_size, 1]).to(device)

    U = torch.randn([batch_size, num_edge_blocks, ns.ch_block_size, rank], device = device) - 1.0
    V = torch.randn([batch_size, num_edge_blocks, block_size, rank], device = device) - 1.0

    staged = _stage(pc, ns, U, V, data)

    ## Craft child values that underflow ##

    element_mars = pc.element_mars.clone()

    ch_sid = ns.chs[0]._output_ind_range[0]
    ch_eid = ns.chs[0]._output_ind_range[1]

    torch.manual_seed(3)
    crafted = torch.rand([ch_eid - ch_sid, batch_size], device = device) * 20.0 - 30.0

    crafted[0::3,:] = -1e4                     # underflows float32 exp
    crafted[1::3,:] = -float("inf")            # exactly zero probability

    element_mars[ch_sid:ch_eid,:] = crafted

    _run_layer(pc, layer, pc.node_mars, element_mars, staged, use_kernel)

    got = pc.node_mars[ns._output_ind_range[0]:ns._output_ind_range[1],:].double()

    shared_only = _shared_only_node_mars(pc, ns, layer, element_mars)
    expected = _dense_reference(pc, ns, U, V, element_mars, log_s1 = shared_only)

    assert torch.all(torch.isfinite(got))
    assert torch.all(torch.abs(got - expected) < 1e-5)

    ## Every child of a node at -inf: the node itself is -inf, and nothing becomes NaN ##

    element_mars[ch_sid:ch_eid,:] = -float("inf")

    _run_layer(pc, layer, pc.node_mars, element_mars, staged, use_kernel)

    got = pc.node_mars[ns._output_ind_range[0]:ns._output_ind_range[1],:]

    assert not torch.any(torch.isnan(got))
    assert torch.all(got == -float("inf"))


@pytest.mark.parametrize("use_kernel", [True, False])
def test_lowrank_layer_vanishing_correction(use_kernel):
    """
    With `-inf` factors the correction is identically zero, so the layer must reproduce the plain
    shared sum layer exactly -- and must not produce a NaN doing it, since the normalizer's inner term
    is then `-inf` too.
    """

    device = torch.device("cuda:0")

    batch_size, block_size, rank = 32, 16, 4

    pc, root_ns, ns, layer = _build_single_layer(2, 3, block_size, rank)

    num_edge_blocks = ns.edge_ids.size(1)

    torch.manual_seed(4)
    data = torch.randint(0, 5, [batch_size, 1]).to(device)

    neg_inf = torch.full([batch_size, num_edge_blocks, ns.ch_block_size, rank], -float("inf"), device = device)
    neg_inf_v = torch.full([batch_size, num_edge_blocks, block_size, rank], -float("inf"), device = device)

    staged = _stage(pc, ns, neg_inf, neg_inf_v, data)

    element_mars = pc.element_mars.clone()

    # The shared layer alone, on the SAME child values (a full `pc(data)` would recompute them)
    shared_only = _shared_only_node_mars(pc, ns, layer, element_mars)

    _run_layer(pc, layer, pc.node_mars, element_mars, staged, use_kernel)

    got = pc.node_mars[ns._output_ind_range[0]:ns._output_ind_range[1],:]

    assert not torch.any(torch.isnan(got))
    assert torch.equal(got, shared_only)


def test_lowrank_layer_kernel_matches_torch():
    """
    The two paths must agree with each other, not merely each with the reference.
    """

    device = torch.device("cuda:0")

    batch_size, block_size, rank = 64, 32, 8

    pc, root_ns, ns, layer = _build_single_layer(2, 2, block_size, rank)

    num_edge_blocks = ns.edge_ids.size(1)

    torch.manual_seed(5)
    data = torch.randint(0, 5, [batch_size, 1]).to(device)

    U = torch.randn([batch_size, num_edge_blocks, ns.ch_block_size, rank], device = device) - 1.0
    V = torch.randn([batch_size, num_edge_blocks, block_size, rank], device = device) - 1.0

    staged = _stage(pc, ns, U, V, data)
    element_mars = pc.element_mars.clone()

    results = dict()
    for use_kernel in [True, False]:
        _run_layer(pc, layer, pc.node_mars, element_mars, staged, use_kernel)
        results[use_kernel] = pc.node_mars[ns._output_ind_range[0]:ns._output_ind_range[1],:].clone()

    assert torch.all(torch.abs(results[True] - results[False]) < 1e-4)


if __name__ == "__main__":
    for use_kernel in [True, False]:
        for config in [(1, 1, 16, 4), (1, 3, 16, 4), (2, 3, 16, 8), (1, 1, 32, 16)]:
            test_lowrank_layer_matches_dense_reference(*config, use_kernel)
        test_lowrank_layer_underflow(use_kernel)
        test_lowrank_layer_vanishing_correction(use_kernel)
    test_lowrank_layer_kernel_matches_torch()
