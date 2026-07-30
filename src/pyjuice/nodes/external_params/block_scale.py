from __future__ import annotations

import warnings

import torch
from typing import Optional, Tuple

from .external_params import ExternalSumParams


_BUFFER_KWARG = None


def _buffer_kwarg() -> str:
    """The staging-buffer kwarg name, resolved once (a per-call import shows up in profiles)."""
    global _BUFFER_KWARG
    if _BUFFER_KWARG is None:
        from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_BUFFER_KWARG
        _BUFFER_KWARG = EXTERNAL_PARAMS_BUFFER_KWARG
    return _BUFFER_KWARG


class BlockScaleSumParams(ExternalSumParams):
    """
    A per-sample **multiplicative gate** on each parameter block, supplied externally at call time.

    Every gate owns one scalar per sample. Writing `g(n,c)` for the gate covering the edge from node `n`
    to child `c`, the effective parameters for batch element `b` are

    .. code-block:: text

        theta_b[n, c] = phi[b, g(n,c)] * theta_shared[n, c] / Z_b[n]
        Z_b[n]        = sum_c phi[b, g(n,c)] * theta_shared[n, c]

    normalized over the children of `n`. Factoring out each gate's share of the node's mass,
    `sigma[g,n] = sum_{c in g} theta_shared[n,c]`, shows what the gate actually does:

    .. code-block:: text

        theta_b[n,c] = [ phi[b,g] sigma[g,n] / sum_{g'} phi[b,g'] sigma[g',n] ] * ( theta_shared[n,c] / sigma[g,n] )

    -- it reweights the **gate-level mixture weights** per sample and leaves each gate's internal
    conditional untouched. Two consequences follow directly:

    * it is an **exact no-op when a node has only one gate**, since the reweighting then cancels against
      the normalizer. Whether a node has several gates is entirely the caller's choice of gate sizes; a
      dense transition with one edge block and the default gate size does nothing at all.
    * a gate that does not depend on the parent node is equivalent to multiplying the child's value, i.e.
      to soft evidence on the children, which PyJuice already provides. The capability that is new here
      is a gate that depends on the parent group **and** the child group.

    **Why it is cheap.** Because `phi` is constant within a gate it pulls out of the block matmul, so in
    log space it folds into the child values as an elementwise add -- the dense per-sample parameter
    matrix is never materialized, and the tensor is `[B, num_gates]` rather than anything shaped like the
    parameters. For a 1024-state layer at `block_size = 32` that is 1 MB at batch 256.

    **Tensor layout.** One `float32` tensor of LOG gates:

    * `phi`: `[B, num_edge_blocks, n_node_gates, n_child_gates]`, where a parameter block is tiled by
      `n_node_gates = ns.block_size / block_size` gates along the node axis and
      `n_child_gates = ns.ch_block_size / ch_block_size` along the child axis.

    Axis 1 follows the column order of `ns.edge_ids`. Log space keeps the gate positive for free and makes
    the fold an add rather than a multiply.

    :note: the gate tile is deliberately NOT forced to equal the parameter block. Refining along the CHILD axis
           is free (the fold is elementwise in the child index), while refining along the NODE axis costs
           one reduction per subgroup, because a single staged tile can only serve nodes that share a
           gate. Shrinking `ns.block_size` to obtain a finer gate would instead tax the whole layer: the
           node tile of the standard kernels collapses with it, below the 16 rows an efficient MMA wants,
           and the compiled index tensors grow as `K^2 / block_size`.

    :param block_size: the GATE's block size -- how many nodes one gate spans, NOT the node's own
                       `block_size`. `None` means the node's, i.e. one gate per parameter block, and it
                       must divide the node's. Only the full block size is supported so far; splitting
                       the node axis needs a separate kernel.
    :type block_size: Optional[int]

    :param ch_block_size: the GATE's child block size -- how many children one gate spans. `None` means
                          the node's own. Any divisor is allowed and costs nothing, which is why this is
                          the axis to refine.
    :type ch_block_size: Optional[int]

    :param apply_z_correction: whether the parameter flows include the term coming from `Z`'s own
                               dependence on `theta`. See :func:`post_backward_layer`.
    :type apply_z_correction: bool

    :param tie_external: share one gate tensor across every copy of a tied node, instead of one per copy.
    :type tie_external: bool
    """

    #: This parameterization reweights the per-edge-block partial sums, and those are gone once the
    #: standard kernel has summed them -- `sum_e phi_e M_e` cannot be recovered from `sum_e M_e`. So it
    #: computes the node values itself instead of correcting them afterwards.
    replaces_shared_forward = True

    def __init__(self, block_size: Optional[int] = None, ch_block_size: Optional[int] = None,
                 apply_z_correction: bool = False, tie_external: bool = False):
        super(BlockScaleSumParams, self).__init__()

        for name, value in (("block_size", block_size), ("ch_block_size", ch_block_size)):
            assert value is None or (isinstance(value, int) and value >= 1), \
                f"`{name}` must be a positive integer or None, got {value}."

        # These are the GATE tile's dimensions, not the node's. Both exist in this class's scope, so
        # every other method goes through `gate_sizes(ns)` rather than reading these directly -- that
        # keeps the one place where `self.block_size` and `ns.block_size` meet down to a single method.
        self.block_size = block_size
        self.ch_block_size = ch_block_size

        # `Z = sum_c phi * theta` depends on `theta`, unlike the low-rank parameterization where the
        # shared parameters' contribution to the normalizer was the constant 1. So the exact derivative
        # w.r.t. `log theta` carries a second term, `- theta * sum_b f phi / Z`. Off by default: the first
        # term alone still sums to `sum_b f` per node, so it drops straight into PyJuice's EM optimizers,
        # whereas the corrected quantity sums to zero and is a gradient, not an expected count.
        self.apply_z_correction = bool(apply_z_correction)

        # Share one gate tensor across the copies of a tied node (see `storage_owner`).
        self.tie_external = bool(tie_external)

        self._auto_ok = None

    # ------------------------------------------------------------------ identity

    def get_signature(self) -> str:
        # Granularity changes the tensor layout and the kernel specialization, and `tie_external` changes
        # which nodes share storage, so nodes that disagree on any of them must not share a layer.
        node = "full" if self.block_size is None else str(self.block_size)
        child = "full" if self.ch_block_size is None else str(self.ch_block_size)

        return f"BlockScale_n{node}_c{child}" + ("_tied" if self.tie_external else "")

    def _get_constructor(self):
        return BlockScaleSumParams, {"block_size": self.block_size,
                                     "ch_block_size": self.ch_block_size,
                                     "apply_z_correction": self.apply_z_correction,
                                     "tie_external": self.tie_external}

    def __reduce__(self):
        return (self.__class__, (self.block_size, self.ch_block_size,
                                 self.apply_z_correction, self.tie_external))

    def storage_owner(self, ns):
        """With `tie_external`, every copy of a tied node reads one shared gate tensor -- the source's."""
        if self.tie_external and ns.is_tied():
            return ns.get_source_ns()

        return ns

    # ------------------------------------------------------------------ layout

    def gate_sizes(self, ns) -> Tuple[int, int]:
        """
        The gate tile's dimensions in nodes and children, resolved against `ns`.

        `None` means "the node's own block size", so the default is one gate per parameter block.
        """
        return (self.block_size if self.block_size is not None else ns.block_size,
                self.ch_block_size if self.ch_block_size is not None else ns.ch_block_size)

    def gate_counts(self, ns) -> Tuple[int, int]:
        """How many gates tile one parameter block, along the node and the child axis respectively."""
        gate_bs, gate_cbs = self.gate_sizes(ns)

        return ns.block_size // gate_bs, ns.ch_block_size // gate_cbs

    def validate_ns(self, ns) -> None:
        gate_bs, gate_cbs = self.gate_sizes(ns)

        assert ns.block_size % gate_bs == 0, \
            f"the gate's `block_size` ({gate_bs}) must divide the node's ({ns.block_size})."
        assert ns.ch_block_size % gate_cbs == 0, \
            f"the gate's `ch_block_size` ({gate_cbs}) must divide the node's ({ns.ch_block_size})."

        # A gate that varies across the nodes of a block cannot share one staged tile, so it needs one
        # reduction per subgroup. Rejected here rather than silently mis-computed in the kernel.
        if gate_bs != ns.block_size:
            raise NotImplementedError(
                f"a gate `block_size` smaller than the node's is not supported yet (got {gate_bs} vs "
                f"{ns.block_size}). Refining the gate's `ch_block_size` is free and is usually what is "
                f"wanted; splitting the node axis needs a separate kernel."
            )

        if ns.edge_ids.size(1) == 1 and gate_cbs == ns.ch_block_size:
            warnings.warn(
                f"`BlockScaleSumParams` on a node with a single gate is an exact no-op: the gate cancels "
                f"against the normalizer. Use a smaller gate `ch_block_size` (or more edge blocks) for "
                f"it to have any effect.", RuntimeWarning)

    def tensor_shapes(self, ns, batch_size: int):
        num_edge_blocks = ns.edge_ids.size(1)
        n_node_gates, n_child_gates = self.gate_counts(ns)

        return ((batch_size, num_edge_blocks, n_node_gates, n_child_gates),)

    def storage_shapes(self, ns, batch_size: int):
        """
        Stored BATCH-INNERMOST, `[E, A, D, B]`.

        The kernel reads a gate for a `(child, batch)` tile, so consecutive threads -- which differ in
        the batch index -- must read consecutive addresses. It also collapses the addressing to one
        compiled index per edge block, since the subgroup offsets fold in:

        .. code-block:: text

            phi[e, a, d, b] = external_params[ (gate[ng, j] + a * D + d) * B + b ]
        """
        num_edge_blocks = ns.edge_ids.size(1)
        n_node_gates, n_child_gates = self.gate_counts(ns)

        return ((num_edge_blocks, n_node_gates, n_child_gates, batch_size),)

    def storage_perm(self):
        return (1, 2, 3, 0)      # (B, E, A, D) -> (E, A, D, B)

    # ------------------------------------------------------------------ execution

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs) -> None:
        raise NotImplementedError(
            "`BlockScaleSumParams` has no per-node reference path; it is served by `forward_layer`."
        )

    def _build_plan(self, layer, ns_tensors, node_mars, element_mars, params, external_params):
        """
        Resolve every per-layer launch argument once, and check the CuTe kernel's assumptions.

        This parameterization has no fallback: the forward is a fork of pyjuice's CuTe/TMA sum kernel,
        so wherever that kernel does not apply, neither does this. Everything unsupported raises rather
        than silently running something slower or, worse, something wrong.
        """
        from .kernels.c import get_cute_module
        import pyjuice.layer.kernels.c as ck

        mod = get_cute_module()
        if mod is None:
            raise NotImplementedError(
                "`BlockScaleSumParams` needs the CuTe/TMA CUDA extension, which is unavailable here "
                "(it requires nvcc, CUTLASS headers and a TMA-capable GPU, sm_90+). There is no Triton "
                "fallback for this parameterization -- see the compile warning above."
            )

        if getattr(layer, "ext_slots", None) is None:
            raise NotImplementedError(
                "`BlockScaleSumParams` needs the compiled edge-block tables, which require a "
                "batch-innermost storage layout."
            )

        ns0 = ns_tensors[0][0].ns
        batch_size = node_mars.size(1)
        block_size = layer.block_size

        n_node_gates, n_child_gates = self.gate_counts(ns0)
        _, gate_cbs = self.gate_sizes(ns0)
        node_cbs = ns0.ch_block_size

        if n_node_gates != 1:
            raise NotImplementedError(
                f"a gate spanning fewer nodes than the block cannot share the staged tile; got "
                f"{n_node_gates} node gates per block."
            )

        # Same tile shapes as the kernel's `configs()`; the gate is the standard kernel's own.
        cfgs = [tuple(c) for c in mod.configs()]
        valid = [i for i, c in enumerate(cfgs)
                 if block_size % c[0] == 0 and batch_size % c[1] == 0]
        if not valid:
            raise NotImplementedError(
                f"no CuTe tile fits this layer: block_size={block_size}, batch={batch_size}, tiles="
                f"{[(c[0], c[1]) for c in cfgs]}. The gate needs `block_size % BM == 0` and "
                f"`batch % BN == 0` -- a larger node `block_size` (>= 64) is usually the fix; the gate "
                f"can still be made fine through its own `ch_block_size`, which costs nothing."
            )
        cfg = valid[0]      # provisional; the real one is measured once the launch args exist

        first_view = ns_tensors[0][1][0]
        ext_base = ((first_view.data_ptr() - external_params.data_ptr())
                    // external_params.element_size()) // batch_size

        dev = node_mars.device
        calls = []
        for partition_id in range(layer.num_fw_partitions):
            nids = layer.partitioned_nids[partition_id]
            cids = layer.partitioned_cids[partition_id]
            pids = layer.partitioned_pids[partition_id]
            gate = layer.ext_slots[0][partition_id]

            num_edges = cids.size(1)
            if num_edges % 64 != 0:
                raise NotImplementedError(
                    f"the CuTe forward iterates the edge dimension in tiles of 64; this partition has "
                    f"{num_edges} edges."
                )
            if num_edges % node_cbs != 0:
                raise NotImplementedError(
                    f"the compiled edge count ({num_edges}) must be a whole number of child blocks "
                    f"({node_cbs})."
                )

            # The kernel reads `element_mars[ebase + e]` and `params[pbase + e * block_size + m]`, so
            # the tile's children must be contiguous rows and its parameters must stride by
            # `block_size`. Checked rather than assumed -- exactly as the standard kernel checks it.
            knt = num_edges // 64
            c3 = cids.view(-1, knt, 64).to(torch.int64)
            p3 = pids.view(-1, knt, 64).to(torch.int64)
            ebase = c3[:, :, 0].contiguous()
            pbase = p3[:, :, 0].contiguous()
            ar = torch.arange(64, device = cids.device, dtype = torch.int64)

            if not (torch.equal(c3, ebase.unsqueeze(-1) + ar.view(1, 1, -1))
                    and torch.equal(p3, pbase.unsqueeze(-1) + ar.view(1, 1, -1) * block_size)):
                raise NotImplementedError(
                    "`BlockScaleSumParams` needs the CuTe layout: each edge tile's children contiguous "
                    "and its parameters strided by `block_size`."
                )

            rows = nids.size(0)
            n_eblks = num_edges // node_cbs

            sigma = torch.empty([rows * n_eblks * n_child_gates * block_size], dtype = torch.float32,
                                device = dev)
            log_z = torch.empty([rows * block_size * batch_size], dtype = torch.float32, device = dev)

            calls.append((nids, ebase, pbase, pids, gate, sigma, log_z,
                          block_size, num_edges, node_cbs, gate_cbs, n_node_gates, ext_base, cfg))

        # ---- PICK THE TILE BY MEASURING IT ----
        #
        # There is no rule that gets this right. The kernel is latency bound at these grid sizes, so a
        # narrower tile -- more blocks, each doing less -- usually wins; but the accumulator fragment is
        # `BM * BN / threads` registers, and the fork carries three of them, so a tile with twice the
        # WARPS halves the fragment and doubles the warps that fit per SM. The two pull in opposite
        # directions and which wins flips with the batch size: at K=2048 batch=256 the narrow 8-warp
        # tile is fastest, at batch=512 the wide one is, by 11%.
        #
        # Safe to run: every config computes bit-identical values (they differ only in how the same
        # contraction is tiled), and `forward_layer` runs the real launch after this returns, so
        # whatever the trial launches leave in `node_mars` is overwritten with the same numbers.
        def with_cfg(i):
            return [tuple(a[:-1]) + (i,) for a in calls]

        def runner(i):
            tuned = with_cfg(i)

            def run():
                for args in tuned:
                    mod.blockscale_forward(node_mars, element_mars, params, external_params, *args)

            return run

        best = ck.autotune([(i, runner(i)) for i in valid])
        calls = with_cfg(best if best is not None else valid[0])

        layer._bs_bw_state = (block_size, batch_size, calls)

        return mod, calls

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Compute this layer's node values under the gate, REPLACING the standard sum forward.

        Unlike a corrective parameterization this owns the whole computation: it reweights the
        per-edge-block partial sums, and those do not survive the standard kernel.
        """
        if len(ns_tensors) == 0:
            return None

        external_params = kwargs.get(_buffer_kwarg(), None)
        if external_params is None:
            return None

        ptrs = (node_mars.data_ptr(), element_mars.data_ptr(), params.data_ptr(),
                external_params.data_ptr(), node_mars.size(1))

        entry = getattr(layer, "_bs_fw_plan", None)
        if entry is None or entry[0] != ptrs:
            entry = (ptrs, self._build_plan(layer, ns_tensors, node_mars, element_mars, params,
                                            external_params))
            layer._bs_fw_plan = entry

        mod, calls = entry[1]
        for args in calls:
            mod.blockscale_forward(node_mars, element_mars, params, external_params, *args)

        return None

    def pre_backward_layer(self, layer, ns_tensors, node_flows, element_flows, node_mars,
                           element_mars, params, **kwargs) -> None:
        raise NotImplementedError("BlockScaleSumParams backward: not implemented yet (milestone B).")

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                            node_mars, element_mars, params, param_flows = None, **kwargs) -> None:
        raise NotImplementedError("BlockScaleSumParams backward: not implemented yet (milestone B).")
