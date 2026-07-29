from __future__ import annotations

import torch
from typing import Any, Optional, Tuple

from .external_params import ExternalSumParams


class LowRankSumParams(ExternalSumParams):
    """
    A per-sample, **nonnegative rank-k additive correction** to the sum node's parameters, supplied
    externally at call time.

    The correction is defined **per parameter block**, matching PyJuice's block-sparse layout: the
    `e`-th column of `ns.edge_ids` defines one `block_size x ch_block_size` tile of shared parameters,
    and that tile gets its own rank-`k` correction built from its own factors. Writing `U_e` for the
    child-side factor of edge block `e` and `V_e` for its node-side factor, the effective parameters
    for batch element `b` are

    .. code-block:: text

        theta_b[e, n, c] = ( theta_shared[e, n, c] + sum_r exp(U[b,e,c,r]) * exp(V[b,e,n,r]) ) / Z_b[n]

    normalized over all children `c` of node `n`. Two properties make this cheap and safe:

    * the correction is **nonnegative**, so `theta_b > 0` always and no clamping is required; and
    * it is **additive and rank-k**, so the forward pass factors into the existing shared block-sparse
      matmul (untouched, still on tensor cores) plus two `O(num_edge_blocks * block_size * k * B)`
      terms. The dense per-sample correction is never materialized.

    `Z_b` is computed on the fly. The shared parameters are child-normalized by PyJuice at all times,
    so their contribution to the normalizer is exactly 1:

    .. code-block:: text

        Z_b[n] = 1 + sum_{e incident to n} sum_r exp(V[b,e,n,r]) * ( sum_c exp(U[b,e,c,r]) )

    **Tensor layout.** Two `float32` tensors, indexed by edge block:

    * `U`: `[B, num_edge_blocks, ch_block_size, k]` -- child-side factors
    * `V`: `[B, num_edge_blocks, block_size, k]` -- node-side factors

    where `num_edge_blocks == ns.edge_ids.size(1)`, and axis 1 follows the column order of
    `ns.edge_ids` as stored on the node. They are supplied as a `(U, V)` pair and validated during the
    forward pass.

    :note: for a single-block node -- e.g. an HMM transition with `block_size = num_latents` -- there
           is exactly one edge block, and the layout collapses to the familiar `[B, S, k]` (with a
           singleton edge-block axis).

    :note: the tensors are keyed by the `ns` **instance**. Tied duplicates (such as the per-timestep
           copies of a homogeneous HMM transition) each need their own entry; passing the *same* `U`,
           `V` for several of them is how one per-sample correction is shared across timesteps, and
           passing the same gradient buffers accumulates their gradients.

    :note: initializing the caller's factors to a large negative constant shrinks the correction, so
           training starts near the shared-parameter baseline. It is a *near*-identity, not an exact
           one: at `-6` with `ch_block_size = 1024` and `k = 16` the correction still carries roughly
           10% of each column's mass, near-uniformly, which acts as mild smoothing. `-inf` makes the
           correction identically zero.

    :param rank: the rank `k` of the correction. It is part of the signature, so nodes of different
                 rank compile into different layers and the kernels can specialize on it.
    :type rank: int
    """

    def __init__(self, rank: int):
        super(LowRankSumParams, self).__init__()

        assert isinstance(rank, int) and rank >= 1, f"`rank` must be a positive integer, got {rank}."

        self.rank = rank

    def get_signature(self) -> str:
        return f"LowRank_r{self.rank}"

    def tensor_shapes(self, ns, batch_size: int):
        num_edge_blocks = ns.edge_ids.size(1)

        return (
            (batch_size, num_edge_blocks, ns.ch_block_size, self.rank),  # U
            (batch_size, num_edge_blocks, ns.block_size, self.rank),     # V
        )

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Rewrite `node_mars` from the shared-parameter value `log S1` to the effective value.

        Writing `S1[n,b]` for what the standard kernel produced, `emars` for the child values, and
        with the correction factored so the dense `[B, num_children, num_nodes]` tile is never built:

        .. code-block:: text

            logW[b,e,r] = logsumexp_c ( U[b,e,c,r] + emars[ch_eids[e] + c, b] )
            logA[b,e,r] = logsumexp_c   U[b,e,c,r]

            logS2[n,b]  = logsumexp_{e incident to n, r} ( V[b,e,n,r] + logW[b,e,r] )
            logZ[n,b]   = logaddexp( 0, logsumexp_{e incident to n, r} ( V[b,e,n,r] + logA[b,e,r] ) )

            node_mars[n,b] = logaddexp( logS1[n,b], logS2[n,b] ) - logZ[n,b]

        The `0` in `logZ` is the shared parameters' own contribution: PyJuice keeps them normalized
        over children at all times, so it is exactly 1.

        Both terms cost `O(num_edge_blocks * block_size * rank * B)`; the shared block-sparse matmul
        is untouched.
        """
        raise NotImplementedError("The `LowRankSumParams` forward kernel has not been implemented yet.")

    def pre_backward(self, layer, ns_info, tensors, node_flows, element_flows, node_mars,
                     element_mars, params, **kwargs) -> None:
        """
        Put `node_mars` back into its UNNORMALIZED form, `logT = logaddexp(logS1, logS2)`, by adding
        `logZ` to this `ns`'s slice.

        This is what lets the standard kernels compute the shared component's flows with no change at
        all. They evaluate `param * exp(emars - node_mars)`; feeding them `theta_shared` and `logT`
        yields `theta_shared * ch / T`, which is exactly the shared part's share of the flow, since
        the effective parameter is `theta_shared / Z` and the node value is `T / Z`. The parameter
        flows that come out are therefore the ordinary sum-node flows of `theta_shared`, so EM /
        Anemone keep training it unchanged.
        """
        raise NotImplementedError("The `LowRankSumParams` backward kernels have not been implemented yet.")

    def post_backward(self, layer, ns_info, tensors, grad_tensors, node_flows, element_flows,
                      node_mars, element_mars, params, param_flows = None, **kwargs) -> None:
        """
        Add the correction's share of the child flows, write `dLL/dU` and `dLL/dV`, and restore
        `node_mars` to the normalized value the forward left.

        With `f[n,b]` the node flow and every term below a bounded ratio times a flow (so nothing
        overflows and no `[S,S]` object is formed):

        .. code-block:: text

            P[b,e,r] = sum_{n in par(e)} f[n,b] * exp( V[b,e,n,r] + logW[b,e,r] - logT[n,b] )
            Q[b,e,r] = sum_{n in par(e)} f[n,b] * exp( V[b,e,n,r] + logA[b,e,r] - logZ[n,b] )

            child flow  += sum_r P[b,e,r] * exp( U[b,e,c,r] + emars[c,b] - logW[b,e,r] )
            dLL/dU      =  sum over incident e of the same term, minus Q[b,e,r] * exp( U - logA )
            dLL/dV      =  f[n,b] * ( exp( V + logW - logT ) - exp( V + logA - logZ ) )

        The first term of `dLL/dU` is the child-flow contribution itself, so one pass produces both.

        :note: `U`, `V` of `-inf` (an exactly-zero correction) make `logW` and `logA` `-inf` too, so
               the `U - logA` and `V - logA` differences are `-inf - -inf`. The kernels must mask
               those to `0` rather than let them become `NaN`.
        """
        raise NotImplementedError("The `LowRankSumParams` backward kernels have not been implemented yet.")

    def _get_constructor(self):
        return LowRankSumParams, {"rank": self.rank}

    def __reduce__(self):
        return (self.__class__, (self.rank,))
