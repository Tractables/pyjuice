"""Regression test for the InputLayer partial-evaluation block-id bug.

``InputLayer._prepare_scope2nids`` builds ``scope2localgids`` by incrementing a
running counter by ``ns.num_node_blocks`` (i.e. in *block* units), but every
InputLayer kernel uses the resulting ids as *per-node* indices into per-node
tensors like ``layer.vids`` and ``layer.s_pids``. When ``block_size > 1`` (so
``num_node_blocks < num_nodes``), partial evaluation silently processes only
``num_node_blocks`` out of ``num_nodes`` nodes per ``InputNodes`` group.

These tests build an HMM with ``block_size == num_latents`` (so each input
``InputNodes`` has ``num_node_blocks == 1`` but ``num_nodes == num_latents``) and
check two paths that hit the bug:

1. ``juice.queries.conditional(pc, data, target_vars=subset)`` — the atomic-add
   backward kernel reads partial ids as per-node offsets. Expected: output rows
   for the requested target vars match the sliced full-output; actual (buggy):
   rows 1+ are zero because the partial lookup lands all threads within the
   first target variable's node block.

2. ``pc.enable_partial_evaluation(scopes=..., forward=True)`` followed by a
   forward pass with a cache — should recompute node_mars for all nodes in the
   target scope. Expected: the result matches a full forward on the new data.
   Actual (buggy): only the first node of each matching ``InputNodes`` is
   recomputed, so the log-likelihood is far from the correct value.

Both tests should PASS after the fix to ``_prepare_scope2nids`` (change
``ns.num_node_blocks`` -> ``ns.num_nodes``).
"""
import torch
import pytest

import pyjuice as juice
from pyjuice.nodes.methods import get_subsumed_scopes


@pytest.fixture
def hmm_pc():
    """Small HMM with ``block_size == num_latents`` so ``num_node_blocks == 1``
    per ``InputNodes`` (triggers the bug). Homogeneous=False so each variable
    has its own params — otherwise tied params can mask some of the damage.
    """
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    T, K, V, B = 8, 32, 16, 4
    ns = juice.structures.HMM(
        seq_length=T, num_latents=K, num_emits=V, homogeneous=False,
    )
    pc = juice.TensorCircuit(ns, verbose=False).to(device)
    data = torch.randint(0, V, (B, T), device=device)

    # Sanity: confirm we're actually in the block_size > 1 regime.
    layer = pc.input_layer_group[0]
    assert all(
        ns_.num_node_blocks < ns_.num_nodes for ns_ in layer.nodes
    ), "test precondition: at least one InputNodes must have num_node_blocks < num_nodes"

    return pc, ns, data, (T, K, V, B)


def test_conditional_target_vars_matches_full(hmm_pc):
    """Conditional output for a subset of ``target_vars`` must match the
    full-output tensor sliced to those same vars.
    """
    pc, ns, data, (T, K, V, B) = hmm_pc
    target_vars = [0, 2, 4, 6]

    out_full = juice.queries.conditional(pc, data=data, target_vars=None)
    out_sub  = juice.queries.conditional(pc, data=data, target_vars=target_vars)

    ref = out_full[:, target_vars, :].contiguous()
    assert out_sub.shape == ref.shape
    torch.testing.assert_close(out_sub, ref, rtol=1e-4, atol=1e-5)


def test_partial_forward_matches_full_recompute(hmm_pc):
    """A partial forward that re-evaluates nodes whose scope contains a changed
    variable must produce the same log-likelihood as a full forward on the
    changed data.
    """
    pc, ns, data, (T, K, V, B) = hmm_pc

    # Full forward to populate the cache on the original data.
    _, cache = pc(data, return_cache=True)

    # Change data for variable 1 only, then do a partial forward.
    data2 = data.clone()
    data2[:, 1] = (data2[:, 1] + 1) % V

    scopes = get_subsumed_scopes(ns, [1])
    pc.enable_partial_evaluation(scopes=scopes, forward=True)
    lls_partial = pc(data2, cache=cache).clone()
    pc.disable_partial_evaluation()

    lls_full = pc(data2).clone()

    torch.testing.assert_close(lls_partial, lls_full, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    T, K, V, B = 8, 32, 16, 4
    ns = juice.structures.HMM(
        seq_length=T, num_latents=K, num_emits=V, homogeneous=False,
    )
    pc = juice.TensorCircuit(ns, verbose=False).to(device)
    data = torch.randint(0, V, (B, T), device=device)

    class _Stub:
        pass
    stub = (pc, ns, data, (T, K, V, B))
    test_conditional_target_vars_matches_full(stub)
    test_partial_forward_matches_full_recompute(stub)
    print("unexpectedly passed — bug may be fixed")
