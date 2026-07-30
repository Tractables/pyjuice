"""
Cross-validation of the low-rank forward and backward against a MATERIALIZED plain PC.

The idea is to avoid hand-written references entirely. Two HMMs with the same topology:

  A -- tied transition carrying `LowRankSumParams`, given shared parameters and external factors (U, V)
  B -- plain, UNTIED transitions, whose parameters are set to the effective per-sample matrix

        theta_tilde[n, c] = ( theta_shared[n, c] + sum_r exp(U[c, r]) exp(V[n, r]) ) / Z[n]

      for ONE chosen sample of A's batch.

`theta_tilde` is a legitimate child-normalized parameter matrix, so B is an ordinary PC and its forward
and backward are pyjuice's own trusted code. A's output on that sample must therefore equal B's output,
which validates the whole low-rank path -- staging, the forward correction, the `logT` shift, and the
child-flow correction -- against an oracle that shares none of its code.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import LowRankSumParams


NUM_EMITS = 5


def _build(seq_length, num_latents, rank, external, seed = 0):
    """An HMM whose transition is either external+tied (A) or plain and untied per timestep (B)."""

    torch.manual_seed(seed)

    with juice.set_block_size(num_latents):
        ni = inputs(seq_length - 1, num_node_blocks = 1,
                    dist = dists.Categorical(num_cats = NUM_EMITS))

        transitions, cur = [], ni
        for var in range(seq_length - 2, -1, -1):
            cx = ni.duplicate(var, tie_params = True)

            if external:
                # One tied transition, as the real model would have
                ns = summate(cur, num_node_blocks = 1,
                             external_params = LowRankSumParams(rank = rank)) \
                     if not transitions else transitions[0].duplicate(cur, tie_params = True)
            else:
                # Untied: every timestep needs its own materialized matrix
                ns = summate(cur, num_node_blocks = 1)

            transitions.append(ns)
            cur = multiply(cx, ns)

        root = summate(cur, num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)

    return root, transitions


def _set_node_params(pc, ns, vals):
    """Inverse of :func:`TensorCircuit.get_node_params` -- writes `[E, block_size, ch_block_size]`."""
    import math

    psid, peid = ns._param_range
    local_parids = (ns._param_ids - psid) // (ns.block_size * ns.ch_block_size)

    buf = pc.params[psid:peid].reshape(-1, ns.ch_block_size, ns.block_size)
    buf[local_parids, :, :] = vals.permute(0, 2, 1).to(buf.dtype)


def _effective_params(theta, U, V):
    """
    `theta_tilde` for one sample: `[E, K, Kc]` from shared `[E, K, Kc]` and factors `[E, Kc, r]`,
    `[E, K, r]`, normalized over children exactly as pyjuice keeps its own parameters.
    """
    delta = torch.einsum("enr,ecr->enc", V.exp().double(), U.exp().double())
    eff = theta.double() + delta

    return eff / eff.sum(dim = 2, keepdim = True)


@pytest.mark.parametrize("seq_length,num_latents,rank", [(4, 64, 4), (5, 32, 8)])
def test_lowrank_crossvalidates_materialized_pc(seq_length, num_latents, rank):
    device = torch.device("cuda:0")
    batch_size, sample = 32, 7

    root_a, trans_a = _build(seq_length, num_latents, rank, external = True)
    root_b, trans_b = _build(seq_length, num_latents, rank, external = False)

    pc_a = juice.compile(root_a, verbose = False).to(device)
    pc_b = juice.compile(root_b, verbose = False).to(device)

    # Same emissions in both, so only the transitions differ
    pc_b.input_layer_group.layers[0].params.copy_(pc_a.input_layer_group.layers[0].params)
    _set_node_params(pc_b, root_b, pc_a.get_node_params(root_a))

    torch.manual_seed(3)
    data = torch.randint(0, NUM_EMITS, [batch_size, seq_length], device = device)

    ext = {}
    for ns in trans_a:
        num_edge_blocks = ns.edge_ids.size(1)
        ext[ns] = (torch.randn([batch_size, num_edge_blocks, ns.ch_block_size, rank],
                               device = device) - 1.0,
                   torch.randn([batch_size, num_edge_blocks, ns.block_size, rank],
                               device = device) - 1.0)

    # A's transitions are tied, so they all share one parameter matrix
    theta = pc_a.get_node_params(trans_a[0])

    for ns_a, ns_b in zip(trans_a, trans_b):
        U, V = ext[ns_a]
        _set_node_params(pc_b, ns_b, _effective_params(theta, U[sample], V[sample]))

    # ------------------------------------------------------------------ forward
    lls_a = pc_a(data, sum_external_params = ext)
    lls_b = pc_b(data[sample:sample + 1, :])

    assert torch.abs(lls_a[sample] - lls_b[0]) < 2e-3, \
        f"forward: A {lls_a[sample].item()} vs materialized B {lls_b[0].item()}"

    # ------------------------------------------------------------------ backward
    pc_a.init_param_flows(flows_memory = 0.0)
    pc_b.init_param_flows(flows_memory = 0.0)
    pc_a.backward(data, allow_modify_flows = False)
    pc_b.backward(data[sample:sample + 1, :], allow_modify_flows = False)

    # Node values and flows of every transition must agree on that sample. Located through each model's
    # own id range, so the comparison does not assume the two compile to the same layout.
    for ns_a, ns_b in zip(trans_a, trans_b):
        sl_a = slice(*ns_a._output_ind_range)
        sl_b = slice(*ns_b._output_ind_range)

        ma = pc_a.node_mars[sl_a, sample]
        mb = pc_b.node_mars[sl_b, 0]
        assert torch.all(torch.abs(ma - mb) < 2e-3), \
            f"node_mars mismatch, max {float((ma - mb).abs().max())}"

        # logspace flows -> compare as probabilities so -inf entries are simply 0
        fa = pc_a.node_flows[sl_a, sample].exp()
        fb = pc_b.node_flows[sl_b, 0].exp()
        assert torch.all(torch.abs(fa - fb) < 2e-3), \
            f"node_flows mismatch, max {float((fa - fb).abs().max())}"


if __name__ == "__main__":
    test_lowrank_crossvalidates_materialized_pc(4, 64, 4)
    test_lowrank_crossvalidates_materialized_pc(5, 32, 8)
    print("cross-validation OK")
