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

    def storage_shapes(self, ns, batch_size: int):
        """
        Stored BATCH-INNERMOST, `[E, states, rank, B]`.

        The kernels tile the child / node axis against the batch axis, and `element_mars` / `node_mars`
        are themselves `[·, batch]`. Storing the factors batch-innermost gives them the same access
        shape -- runs of consecutive batch elements -- rather than striding a whole `E*K*rank` block
        between batch elements, which the caller's `[B, E, K, rank]` order would force.

        It also collapses the addressing to one compiled index per edge block, because the per-node
        base and the state / rank offsets fold together:

        .. code-block:: text

            U[e, c, rr, b] = external_params[ (xu[ng, j] + c * rank + rr) * B + b ]
            V[e, m, rr, b] = external_params[ (xv[ng, j] + m * rank + rr) * B + b ]
        """
        num_edge_blocks = ns.edge_ids.size(1)

        return (
            (num_edge_blocks, ns.ch_block_size, self.rank, batch_size),   # U
            (num_edge_blocks, ns.block_size, self.rank, batch_size),      # V
        )

    def storage_perm(self):
        return (1, 2, 3, 0)      # (B, E, states, rank) -> (E, states, rank, B)

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
        self.forward_torch(ns_info, tensors, node_mars, element_mars)

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Apply the correction to the whole layer.

        The compiled `xu` / `xv` tables are laid out per forward partition and already carry each
        row's node identity in their offsets, so one launch per partition covers every node -- no
        per-node arguments, and the kernel's signature does not grow with the number of nodes.

        Falls back to the per-node torch reference when the kernel's assumptions do not hold.
        """
        if len(ns_tensors) == 0:
            return None

        # Imported here rather than at module load: `pyjuice.layer` imports `pyjuice.nodes`, so a
        # top-level import would close a cycle
        from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_BUFFER_KWARG
        from .kernels import fw_lowrank

        external_params = kwargs.get(EXTERNAL_PARAMS_BUFFER_KWARG, None)

        if external_params is None or not self._kernel_applicable(layer, ns_tensors, node_mars):
            for ns_info, tensors in ns_tensors:
                self.forward_torch(ns_info, tensors, node_mars, element_mars)
            return None

        block_size = layer.block_size
        ch_block_size = layer.external_node_infos[0].ch_block_size

        for partition_id in range(layer.num_fw_partitions):
            fw_lowrank(
                node_mars = node_mars,
                element_mars = element_mars,
                external_params = external_params,
                nids = layer.partitioned_nids[partition_id],
                cids = layer.partitioned_cids[partition_id],
                xu = layer.ext_xu[partition_id],
                xv = layer.ext_xv[partition_id],
                block_size = block_size,
                ch_block_size = ch_block_size,
                rank = self.rank,
            )

    def _kernel_applicable(self, layer, ns_tensors, node_mars) -> bool:
        """
        Whether the Triton kernel covers this layer's shape.

        It is written for the regime the feature targets -- a large batch with a large per-node
        workload and a moderate rank -- so anything outside that falls back to the torch reference
        rather than being served by a kernel that was not tuned for it.
        """
        # The compiled index tables are only built for a batch-innermost storage layout
        if getattr(layer, "ext_xu", None) is None:
            return False

        # Every node of the layer must be corrected: the kernel walks whole partitions
        if len(ns_tensors) != len(layer.external_node_infos):
            return False

        # One `ch_block_size` and one `max_n_eblks` for the whole layer, since both are constexprs
        ch_block_sizes = set([ns_info.ch_block_size for ns_info in layer.external_node_infos])
        if len(ch_block_sizes) != 1:
            return False

        ch_block_size = ch_block_sizes.pop()

        return (
            self.rank <= 64 and (self.rank & (self.rank - 1)) == 0   # power-of-2 rank, tile-friendly
            and layer.block_size >= 16 and ch_block_size >= 16
            and node_mars.size(1) >= 16                              # tuned for large batch
        )

    def forward_torch(self, ns_info, tensors, node_mars, element_mars) -> None:
        """
        Reference implementation of :func:`forward`, in plain torch.

        Correctness-first and deliberately unfused -- it is the oracle the kernels are validated
        against, and the fallback for shapes they do not cover.
        """
        U, V = tensors                      # [E, Kc, r, B] and [E, K, r, B], batch-innermost

        block_size, ch_block_size = ns_info.block_size, ns_info.ch_block_size

        par_ptr = ns_info.par_ptr
        child_offsets = torch.arange(0, ch_block_size, device = U.device)

        for nblock_id in range(ns_info.num_node_blocks):
            eblk_ids = ns_info.eblk_ids[par_ptr[nblock_id]:par_ptr[nblock_id + 1]]

            log_s2, log_zt = None, None
            for eblk_id in eblk_ids.tolist():
                # Children of this edge block, and their values
                cids = ns_info.ch_eids[eblk_id] + child_offsets
                emars = element_mars[cids,:]                                   # [Kc, B]

                u = U[eblk_id]                                                 # [Kc, r, B]
                log_w = torch.logsumexp(u + emars[:,None,:], dim = 0)          # [r, B]
                log_a = torch.logsumexp(u, dim = 0)                            # [r, B]

                v = V[eblk_id]                                                 # [K, r, B]
                s2 = torch.logsumexp(v + log_w[None,:,:], dim = 1)             # [K, B]
                zt = torch.logsumexp(v + log_a[None,:,:], dim = 1)             # [K, B]

                log_s2 = s2 if log_s2 is None else torch.logaddexp(log_s2, s2)
                log_zt = zt if log_zt is None else torch.logaddexp(log_zt, zt)

            nid_sid = ns_info.nid_start + nblock_id * block_size
            nid_eid = nid_sid + block_size

            log_s1 = node_mars[nid_sid:nid_eid,:]                              # written by the shared kernel

            # `log Z` -- the shared parameters contribute exactly 1, since PyJuice keeps them
            # child-normalized, so their log-contribution is 0
            log_z = torch.logaddexp(torch.zeros_like(log_zt), log_zt)

            node_mars[nid_sid:nid_eid,:] = torch.logaddexp(log_s1, log_s2) - log_z

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
