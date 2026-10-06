from __future__ import annotations

import os
import torch
from typing import Any, Optional, Sequence, Tuple


# Elements per tile for the shared dual-EM kernels' HEURISTIC default -- `candidates[0]` in
# `_em_tile_candidates`, i.e. what runs with `PYJUICE_AUTOTUNE=0`. The autotuner spans a wider range,
# so it is a seed rather than a cap; 1024 because these kernels keep ~3x the live tiles of a plain
# scatter and 4096 spills. The older BlockScale-specific name is still honoured.
_EM_TILE_BUDGET = int(os.environ.get("PYJUICE_DUAL_EM_TILE",
                                     os.environ.get("PYJUICE_BLOCKSCALE_EM_TILE", "1024")))

_SM_COUNT = {}


def _sm_count(device):
    """SM count, cached: the dual-EM launcher sizes its grid against it once per partition per call."""
    import torch
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    n = _SM_COUNT.get(idx)
    if n is None:
        n = _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return n


class ExternalSumParams():
    """
    Base class of the *external parameterizations* of a sum node.

    An `ExternalSumParams` describes how per-sample, externally supplied tensors modify the effective
    parameters of an :class:`~pyjuice.nodes.ExternalParamsSumNodes`. It is the sum-node counterpart of
    :class:`~pyjuice.nodes.distributions.Distribution` for input nodes: the node holds the descriptor,
    the descriptor owns the semantics (the tensor layout and the math the layer applies), and the
    compiled circuit groups nodes into layers by :func:`get_signature`, so a mode never shares a
    compiled layer -- or a kernel -- with a different mode.

    The descriptor is pure configuration: it holds no tensors and no per-call state. The external
    tensors are owned by the caller, are **not** EM-trained by the node, and are supplied per call

    .. code-block:: python

        lls = pc(x, sum_external_params = {ns: tensors})

    with the matching per-sample gradients written back through

    .. code-block:: python

        pc.backward(x, sum_external_params_grad = {ns: grad_tensors})

    They are validated against :func:`tensor_shapes` during the forward pass. The shared parameters
    keep training through the ordinary EM / gradient path, unchanged.
    """

    def get_signature(self) -> str:
        """
        Get the signature of the current parameterization.

        Sum nodes are grouped into layers by (block size, signature), and one layer compiles a single
        set of kernels, so two nodes may only share a layer if their signatures match. Any setting
        that the kernels specialize on must therefore appear in the signature.
        """
        raise NotImplementedError()

    #: Whether this parameterization computes the layer's node values ITSELF, making the standard
    #: sum-layer forward redundant. `False` -- the default -- means the standard forward runs first and
    #: the descriptor then corrects what it wrote, which is possible whenever the correction can be
    #: recovered from the shared total. A parameterization that reweights the per-edge-block partial
    #: sums cannot do that (the partials are gone once the standard kernel has summed them), so it sets
    #: this and takes over the whole computation.
    replaces_shared_forward: bool = False

    #: The backward counterpart. `False` -- the default -- means the standard element-flow and
    #: param-flow kernels run and the descriptor corrects what they wrote. A parameterization whose
    #: effective parameters vary PER EDGE BLOCK cannot be served that way: the standard kernels sum
    #: every parent of a child before the correction could be applied, and `sum_e w_e X_e` is not
    #: recoverable from `sum_e X_e`. Such a parameterization sets this and owns both flows, exactly as
    #: `replaces_shared_forward` makes it own the node values.
    replaces_shared_backward: bool = False

    #: Whether the backward writes `d LL / d(external parameters)` into the gradient buffer. `True` --
    #: the default -- is the complete parameterization. A descriptor that contributes the element and
    #: parameter flows but not yet its own gradient sets this to `False`, which leaves `pc.backward()`
    #: and the EM optimizers fully working and makes `pc.get_external_params_grad` say plainly that the
    #: gradient is unavailable, rather than returning the zeroed buffer as though it were an answer.
    computes_external_grads: bool = True

    #: Whether this parameterization's M-step needs the DENOMINATOR flow buffer -- a second
    #: accumulator, alongside the ordinary observed flow `F+` in `param_flows`, holding whatever
    #: per-mini-batch statistic the conditional M-step divides by. `False` -- the default -- trains
    #: through the plain single-flow M-step `theta <- normalize(F+)`. A parameterization whose
    #: per-sample effective parameters carry a normalizer that itself depends on `theta` (so
    #: `normalize(F+)` is not exact EM) sets this: the PC then allocates `pc.denom_param_flows`, the
    #: layer accumulates into it, and the M-step becomes the conditional
    #: `theta <- normalize(theta * F+ / F-)`. A plain sum layer -- and any PC with no such
    #: parameterization -- never allocates the buffer and pays nothing.
    #:
    #: The buffer's LAYOUT belongs to the parameterization, not to the PC: it is sized by
    #: :func:`denom_flow_sizes` and is NOT addressed by `pfids`. That is what keeps it cheap -- `F-` is
    #: usually far more compressible than `F+`. `BlockScaleSumParams` stores the gate-space contraction
    #: `W[node, gate]`, `gate_cbs` times smaller than a param-flow mirror (MEASURED 64x at a 1024-state
    #: gated HMM, 128x at 2048, where the mirror was 13.6% / 18.5% of peak training memory), and
    #: reconstructs `F-[n,c] = theta[n,c] * W[n, g(c)]` at M-step time.
    requests_denom_param_flows: bool = False

    def denom_flow_sizes(self, layer) -> "list":
        """
        How many floats of `pc.denom_param_flows` this layer needs, one entry per FORWARD PARTITION.

        Called once, at compile time, for every layer whose parameterization
        :attr:`requests_denom_param_flows`. The PC concatenates the sizes over layers, allocates one
        flat buffer, and hands each layer where its slices sit (`layer.denom_flow_slices`, one
        `(offset, size)` per forward partition); the
        parameterization decides what lives inside. Sizes must be derivable from the compiled tables
        alone -- `layer.partitioned_nids[pid].size(0)`, `layer.partitioned_cids[pid].size(1)`,
        `layer.block_size` -- because the buffer is allocated before the first forward pass.
        """
        raise NotImplementedError(
            f"{self.get_signature()} requests the denominator flow buffer but does not implement "
            "`denom_flow_sizes`, so the PC cannot size it."
        )

    def storage_owner(self, ns):
        """
        The node whose staging slots `ns` uses.

        Identity by default: every node gets its own slots. A parameterization may return a DIFFERENT
        node to make several nodes share one set of external tensors -- e.g. along a parameter-tying
        relation, so one factor pair serves every copy of a tied layer. Sharing changes two things for
        the caller: tensors are supplied once, for the owner, and the gradient returned for the owner is
        the SUM over every node that shares it.
        """
        return ns

    def validate_ns(self, ns) -> None:
        """
        Check that `ns` is compatible with this parameterization. Called once, at node construction.

        :param ns: the sum nodes carrying this parameterization
        :type ns: ExternalParamsSumNodes
        """
        pass

    def tensor_shapes(self, ns, batch_size: int) -> Tuple[Tuple[int,...],...]:
        """
        Shapes of the external tensors this parameterization consumes, for `ns` at `batch_size`, in
        the order the tensors are supplied. This is the layout the caller must produce and against
        which the forward pass validates.
        """
        raise NotImplementedError()

    def storage_shapes(self, ns, batch_size) -> Tuple[Tuple[int,...],...]:
        """
        Shapes the external tensors are *stored* in inside the PC's staging buffer.

        The caller's layout and the kernels' preferred layout need not agree: staging copies, so it
        can transpose for free-ish while the caller still hands over whatever their own head produced.
        Defaults to storing exactly what the caller supplies.

        :note: every axis except the batch axis must be batch-independent, so that a slot's size is
               proportional to the batch size. That is what lets the compiled index tensors express
               offsets in per-batch units and stay valid across batch sizes.
        """
        return self.tensor_shapes(ns, batch_size)

    def storage_perm(self) -> Optional[Tuple[int,...]]:
        """
        Permutation taking the caller's axis order to :func:`storage_shapes`' order, or `None` when
        they agree. Applied to the caller's tensor during staging.
        """
        return None

    def storage_offsets(self, ns):
        """
        Where each edge block's entry begins within each storage slot, in per-batch units, as one
        `long` tensor per slot indexed by edge-block id -- or `None` for the default layout.

        The default is `[E, ...rest, B]`: one contiguous entry per edge block, in edge-block order, so
        the offset is `edge_block * prod(rest)` and the layer computes it itself. Override when storage
        is indexed by something other than the edge block, so that the compiled tables point at the
        right place and staging stays a copy rather than a gather.
        """
        return None

    def to_storage(self, ns, tensors: Tuple) -> Tuple:
        """
        Map the caller's tensors into :func:`storage_shapes`' layout, ready to be copied.

        The default applies :func:`storage_perm`, which covers any parameterization whose two layouts
        differ only in axis order. Override when they differ in SHAPE -- when the caller's layout is
        the one that reads naturally for the model and the storage layout is the one the kernels index,
        and getting from one to the other needs a gather rather than a transpose.

        :param tensors: the caller's tensors, already validated against :func:`tensor_shapes`.
        :returns: one tensor per storage slot, each matching :func:`storage_shapes`. They are copied,
                  so arbitrary strides are fine and a contiguous result is not required.
        """
        perm = self.storage_perm()
        if perm is None:
            return tuple(tensors)

        return tuple(tensor.permute(perm) for tensor in tensors)

    def from_storage(self, ns, tensors: Tuple) -> Tuple:
        """
        The inverse of :func:`to_storage`, used to hand gradients back in the caller's layout.

        Must invert `to_storage` exactly, so that a gradient lines up element-for-element with the
        tensor the caller supplied. Where `to_storage` gathers, this scatters -- and any entry of the
        caller's layout that storage has no slot for takes a zero gradient, which is correct: nothing
        in the model read it.
        """
        perm = self.storage_perm()
        if perm is None:
            return tuple(tensors)

        inverse = tuple(perm.index(axis) for axis in range(len(perm)))
        return tuple(tensor.permute(inverse) for tensor in tensors)

    def compile(self, layer) -> None:
        """
        Compile the indices and other tensors this parameterization's kernels need, called once when
        `layer` is built.

        By this point the layer has built the generic per-`ns` metadata -- one
        :class:`~pyjuice.layer.external_sum_layer.ExternalNodeInfo` per `ns` in
        `layer.external_node_infos`, giving the mapping from the node's `edge_ids` columns to global
        node and element ids -- which is usually the starting point for anything further.

        Register whatever is derived through the layer, not on `self`:

        .. code-block:: python

            def compile(self, layer):
                tables = [build_table(ns_info) for ns_info in layer.external_node_infos]
                layer.register_external_buffers("edge_table", tables)   # -> ns_info.edge_table

        :func:`~pyjuice.layer.ExternalParamsSumLayer.register_external_buffers` takes one tensor per
        `ns` and exposes it as an attribute of that `ns_info`;
        :func:`~pyjuice.layer.ExternalParamsSumLayer.register_external_buffer` takes a single
        layer-wide tensor. Either way the layer owns the storage, so `.to(device)` moves it and
        `state_dict` sees it. Non-tensor state (caches, autotuned choices) can be set as a plain
        attribute on `layer`.

        :note: do NOT keep compiled state on the descriptor. One descriptor instance is shared by
               every node constructed with it -- including tied duplicates, which live in *different*
               layers -- so per-layer state stored on `self` would be overwritten by whichever layer
               compiles last. The descriptor is stateless configuration.

        :param layer: the layer that compiled these nodes
        :type layer: ExternalParamsSumLayer
        """
        pass

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Turn the shared-parameter node values into the effective ones, in place.

        Called once per `ns` after the standard sum-layer forward has run, so on entry
        `node_mars[ns_info.nid_start:ns_info.nid_end]` holds the value each node takes under the
        SHARED parameters alone. On exit it must hold the value under the effective parameters. The
        shared kernels are not re-run and not modified, so a parameterization is responsible for
        expressing its effect as a correction to what they produced.

        :param layer: the layer being evaluated
        :type layer: ExternalParamsSumLayer

        :param ns_info: compiled metadata for the `ns` these tensors belong to
        :type ns_info: ExternalNodeInfo

        :param tensors: the validated external tensors supplied for `ns_info.ns`
        :type tensors: Tuple[torch.Tensor,...]
        """
        raise NotImplementedError()

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Apply the parameterization to a whole layer, once per forward pass.

        The default loops the layer's nodes and calls :func:`forward` per node, which is the simplest
        thing to implement. Override it when the kernels can span several nodes in one launch -- the
        compiled index tensors are laid out per FORWARD PARTITION, covering every node of the layer,
        so a partition-level launch needs no per-node arguments at all.

        :param ns_tensors: `[(ns_info, tensors), ...]` for the nodes that were given external tensors
        """
        for ns_info, tensors in ns_tensors:
            self.forward(layer, ns_info, tensors, node_mars, element_mars, params, **kwargs)

    def pre_backward(self, layer, ns_info, tensors, node_flows, element_flows, node_mars,
                     element_mars, params, **kwargs) -> None:
        """
        Prepare the buffers so that the *standard* sum-layer backward, run immediately afterwards and
        unmodified, computes the flows of the SHARED component of the parameters.

        Called once per `ns` before the standard backward. Anything changed here must be undone in
        :func:`post_backward`, since the buffers are shared with the rest of the circuit.
        """
        raise NotImplementedError()

    def pre_backward_layer(self, layer, ns_tensors, node_flows, element_flows, node_mars,
                           element_mars, params, **kwargs) -> None:
        """
        Layer-level counterpart of :func:`pre_backward`, mirroring :func:`forward_layer`.

        Defaults to looping over the layer's nodes; override when the whole layer can be prepared in
        one shot (the compiled tables span every node in it, so a per-node loop repeats work).
        """
        for ns_info, tensors in ns_tensors:
            self.pre_backward(layer, ns_info, tensors, node_flows, element_flows, node_mars,
                              element_mars, params, **kwargs)

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                            node_mars, element_mars, params, param_flows = None,
                            denom_param_flows = None, **kwargs) -> None:
        """Layer-level counterpart of :func:`post_backward`."""
        for (ns_info, tensors), grad_tensors in zip(ns_tensors, ns_grad_tensors):
            self.post_backward(layer, ns_info, tensors, grad_tensors, node_flows, element_flows,
                               node_mars, element_mars, params, param_flows = param_flows,
                               denom_param_flows = denom_param_flows, **kwargs)

    def post_backward(self, layer, ns_info, tensors, grad_tensors, node_flows, element_flows,
                      node_mars, element_mars, params, param_flows = None,
                      denom_param_flows = None, **kwargs) -> None:
        """
        Add the external contribution to the child flows, write the per-sample gradients of the
        external tensors, and undo whatever :func:`pre_backward` changed.

        Called once per `ns` after the standard backward.

        :param grad_tensors: buffers to ACCUMULATE the per-sample gradients into, laid out exactly
                             like the external tensors, or `None` if the caller did not request
                             gradients for this `ns`. They are zeroed once per `pc.backward` before
                             any layer runs, so several nodes may share one buffer and have their
                             gradients summed into it.
        :type grad_tensors: Optional[Tuple[torch.Tensor,...]]

        :param denom_param_flows: the PC's denominator flow buffer, or `None` when no layer requested
                             it (see :attr:`requests_denom_param_flows`). A parameterization that
                             requested it accumulates its expected/normalizer statistic into its own
                             slices -- `denom_param_flows[off : off + size]` for each `(off, size)`
                             in `layer.denom_flow_slices` -- here, alongside the standard backward's
                             `F+` into `param_flows[pfid]`. The layout is the parameterization's own;
                             :func:`compute_em_correction` turns it back into `F-`.
        :type denom_param_flows: Optional[torch.Tensor]
        """
        raise NotImplementedError()

    def accumulate_denom_top_down(self, layer, node_flows, denom_param_flows, scale: float) -> None:
        """
        Add the mini-batch-EM top-down term to this layer's denominator slices.

        Called only under `step_size_rescaling` (Anemone), from `eval_top_down_probs`, and only for
        layers that :attr:`requests_denom_param_flows`. The top-down pass adds
        `scale * P_td[n] * theta[n,c]` to `param_flows[pfid(n,c)]`; the numerator and the denominator
        of the conditional M-step must be built the same way, so the SAME term has to reach `F-`.
        `node_flows[n, 0]` holds `P_td[n]` (the pass runs one "sample").

        A parameterization whose denominator is not a param-flow mirror cannot let the generic
        param-flow kernel do this -- it has to add the term in its own layout, which is why this is a
        hook rather than a second call into the standard kernel.
        """
        raise NotImplementedError(
            f"{self.get_signature()} requests the denominator flow buffer but does not implement "
            "`accumulate_denom_top_down`, so `mini_batch_em(step_size_rescaling = True)` cannot build "
            "a denominator that matches the numerator."
        )

    def materialize_denom_flows(self, ns, params, denom_param_flows, denom_sources, out = None):
        """
        Rebuild `F-` for `ns`'s parameter-flow range, laid out exactly like `param_flows[pfs:pfe]`.

        THE ONE HOOK the generic conditional M-step needs. `denom_param_flows` holds whatever
        :func:`denom_flow_sizes` asked for, which is deliberately NOT required to be `F-` itself --
        a parameterization is free to accumulate any sufficient statistic and expand it here. If its
        statistic already IS `F-` in param-flow layout, this is a view:

        .. code-block:: python

            off, size = layer.denom_flow_slices[0]
            return denom_param_flows[off : off + size]

        `BlockScaleSumParams` instead keeps the gate-space contraction `W[node, gate]` and expands
        `F-[n,c] = theta[n,c] * W[n, g(c)]` with a scatter kernel, which is `ch_block_size` times less
        memory to carry between backward calls.

        :param denom_sources: every `(layer, member_ns)` whose slices hold flow for THESE parameters
            -- `ns` plus every tied copy of it. Their contributions must be SUMMED here: the PC does
            not run `compute_cum_par_flows` on the denominator, because its layout is yours and not
            `pfid`-indexed. A parameterization that does store it in `pfid` space may simply fuse the
            members the same way the numerator does.
        :param out: a `pfe - pfs` tensor to write into, or `None` to return a fresh/cached one.
        """
        raise NotImplementedError(
            f"{self.get_signature()} requests the denominator flow buffer but does not implement "
            "`materialize_denom_flows`, so the conditional M-step cannot reconstruct `F-`."
        )

    def denom_kernel_spec(self, layer, denom_param_flows, pid):
        """
        How the SHARED dual-EM kernels should form `F-` for one forward partition, or `None` to
        decline the fast path (the generic torch M-step then runs).

        Returns `(mode, tensor, n_groups, group_cbs, denom_base)`:

        * `mode` -- `DENOM_DENSE` if `tensor[pfid - denom_base]` IS `F-`, or `DENOM_ROWGROUP` if
          `F-[n,c] = theta[n,c] * tensor[n, c // group_cbs]` (a per-(node, group-of-children) factor,
          laid out `[rows * block_size, n_groups]`). See `kernels/dual_em.py`.
        * `n_groups` / `group_cbs` -- the row-group geometry; `(0, 1)` when unused.
        * `denom_base` -- the `pfid` origin of `tensor` in dense mode; `0` when unused.

        Implementing this is what buys a parameterization the FAST M-step. It is worth doing for
        MEMORY and not only for speed: the generic torch path has to materialize `F-` and then builds
        `ratio` / `flow` / `new_theta` on top of it, MEASURED at ~5x the parameter range in transients
        against ~1x here (20.1 MB vs 4.1 MB at a 4 MB `ns`), and 2.2x the time.

        Under parameter tying the base calls this once per member and SUMS the tensors, so every
        member's must be elementwise-addable onto the source's -- same mode, same geometry, same
        shape. It declines the fast path rather than guessing when they are not.
        """
        return None

    def _em_plan(self, ns, layer, device):
        """
        Everything the fused M-step needs that depends only on the COMPILED structure, or `None` if
        this shape is not served.

        Cached on the layer per parameter range and device. Everything in it is STRUCTURAL -- which
        rows of each partition belong to this `ns`, how many children each of their nodes has, and the
        span of `cum` -- so nothing that varies call to call belongs here.

        Declines an `ns` whose parameter and param-flow ranges differ in length, which would break the
        caller's reshape contract.
        """
        ps, pe = ns._param_range
        pfs, pfe = ns._param_flow_range
        if pfe - pfs != pe - ps:
            return None

        block_size = layer.block_size
        cache = layer.__dict__.setdefault("_dual_em_plan_cache", {})
        key = (ps, pe, str(device))
        plan = cache.get(key)
        if plan is not None:
            return plan

        # ROWS OF THIS `ns` ONLY. A partition may hold node blocks of several `ns`; selecting rows
        # here, rather than masking writes afterwards, is what keeps them separate.
        parts = []
        for pid in range(layer.num_fw_partitions):
            pids, cids = layer.partitioned_pids[pid], layer.partitioned_cids[pid]
            mine = ((pids[:, 0] >= ps) & (pids[:, 0] < pe)).nonzero().flatten()
            if mine.numel() == 0:
                parts.append(None)
                continue
            parts.append({
                "row_map": mine.to(torch.int32).contiguous().to(device),
                # children per node of each row, i.e. the torch path's `eblk_per_nb[nb] * cbs`
                "kcount": (cids != 0).sum(dim = 1).to(torch.int32).contiguous().to(device),
            })
        if all(x is None for x in parts):
            return None

        # `cum` is indexed by GLOBAL node id so it can be shared across partitions, which it must be:
        # a node's children may be split across them.
        live = [pid for pid in range(layer.num_fw_partitions) if parts[pid] is not None]
        nids = [layer.partitioned_nids[pid][parts[pid]["row_map"].long()] for pid in live]
        nid_min = min(int(t.min()) for t in nids)
        nid_max = max(int(t.max()) for t in nids)
        plan = cache[key] = {"parts": parts, "cum_base": nid_min,
                             "cum_size": nid_max + block_size - nid_min}
        return plan

    @staticmethod
    def _em_tile_candidates(block_size, num_edges, n_rows, dev):
        """
        `(TILE_SIZE_M, TILE_SIZE_K)` candidates for the shared dual-EM kernels, heuristic default first.

        `TILE_SIZE_M` MUST divide `block_size`: the kernels derive `M_TILES = BLOCK_SIZE_M //
        TILE_SIZE_M` and address `pid_y // M_TILES`, so a remainder would mis-map rows.

        Spans tile BUDGETS 1024-8192 as well as the TM/TK split, because neither alone is right
        everywhere: MEASURED, HMM 2048 prefers a 1024-element tile (3315 us against 4123 at 4096)
        while HMM 512 prefers 4096 (901 us against 1086 at 1024) -- a 20% swing in OPPOSITE
        directions, and picking one budget globally is what made 512 regress. The small tiles earn
        their place the same way: at HMM 512 a 64x64 tile leaves only 64 programs against a
        ~752-program target, so there the GRID binds rather than the tile.

        `TILE_SIZE_K` groups the normalizer's REDUCTION, so varying it reassociates
        `sum_c theta*ratio`. That is benign here and nowhere else would be: the summands are
        non-negative (no cancellation) and carry no max-stabilization, and the cross-tile combine is
        already `tl.atomic_add`, i.e. order-nondeterministic from run to run -- so TK only reorders a
        sum that was never deterministic.
        """
        import triton

        target = 4 * _sm_count(dev)
        ne_pow2 = triton.next_power_of_2(num_edges)

        def legal(tm, tk):
            tm = max(1, min(tm, block_size))
            if block_size % tm != 0:
                return None
            return (tm, max(1, min(tk, ne_pow2)))

        tk0 = max(1, min(ne_pow2, 32))
        tm0 = max(1, min(block_size, max(1, _EM_TILE_BUDGET // tk0)))
        floor = 16 if block_size >= 16 else 1
        while tm0 > floor and triton.cdiv(num_edges, tk0) * n_rows * (block_size // tm0) < target:
            tm0 //= 2
        while tk0 > 8 and triton.cdiv(num_edges, tk0) * n_rows * (block_size // tm0) < target:
            tk0 //= 2

        out, seen = [], set()
        for cand in [(tm0, tk0), (64, 64), (128, 32), (64, 32), (128, 16), (64, 16), (32, 32),
                     (32, 16), (16, 16), (16, 32), (256, 16), (32, 64)]:
            c = legal(*cand)
            if c is not None and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _fused_em_correction(self, ns, params, param_flows, denom_param_flows, step_size,
                             pseudocount, keep_zero_params, denom_sources, out = None):
        """
        The conditional M-step as two shared Triton kernels, or `None` when this shape is not served
        (:func:`compute_em_correction` then runs the generic torch path, which is also what this is
        validated against).

        Generic in the parameterization: how `F-` is formed is the `DENOM_MODE` constexpr that
        :func:`denom_kernel_spec` selects, and `F-` is never materialized -- it is computed inside the
        kernels from whatever statistic was stored. A parameterization gets all of this by
        implementing that one method.
        """
        import triton
        from .kernels.dual_em import _dual_em_cum_kernel, _dual_em_update_kernel

        sources = list(denom_sources)
        if len(sources) == 0:
            return None
        layer = sources[0][0]
        dev = params.device
        block_size = layer.block_size

        # ---- gather the per-partition `F-` recipes, and the tie group's addends ----
        specs = []
        for pid in range(layer.num_fw_partitions):
            s0 = self.denom_kernel_spec(layer, denom_param_flows, pid)
            if s0 is None:
                return None
            mode, tensor, n_groups, group_cbs, denom_base = s0
            addends = [tensor]
            for other, _ in sources[1:]:
                if (other.num_fw_partitions != layer.num_fw_partitions
                        or other.block_size != block_size):
                    return None
                so = self.denom_kernel_spec(other, denom_param_flows, pid)
                if so is None or so[0] != mode or so[2] != n_groups or so[3] != group_cbs \
                        or so[1].shape != tensor.shape:
                    return None                  # not elementwise-addable onto the source's
                addends.append(so[1])
            specs.append((mode, addends, n_groups, group_cbs, denom_base))

        plan = self._em_plan(ns, layer, dev)
        if plan is None:
            return None

        ps, pe = ns._param_range
        pfs, pfe = ns._param_flow_range
        size = pfe - pfs

        # The caller may hand us `params[ps:pe]` to write in place, which is what removes BOTH the
        # separate write-back and the discarded `em_par_update` work on these ranges. It is NOT zeroed:
        # every element of a node's parameter range is covered by exactly one real edge slot, so each
        # is written exactly once.
        by_par = out is not None
        if by_par:
            assert out.numel() == pe - ps, (out.numel(), pe - ps)
            out_base, out_size_ = ps, pe - ps
        else:
            out = torch.zeros(size, device = dev, dtype = torch.float32)
            out_base, out_size_ = pfs, size
        cum = torch.zeros(plan["cum_size"], device = dev, dtype = torch.float32)

        # EVERY scalar the kernels use, fp32, on the DEVICE -- including the clamp thresholds. Triton
        # types a Python float KERNEL ARGUMENT as fp64, and a float LITERAL inside the kernel promotes
        # too, so either one silently turns the whole `den`/`ratio`/`new` chain into double precision.
        # Order must match the `tl.load(consts + i)` offsets in `kernels/dual_em.py`.
        # CACHED by value: every `ns` of one EM step uses the same step size and pseudocount, so
        # building this per `ns` was 15 host-to-device copies per step -- MEASURED 1007 -> 1185 us.
        ckey = (float(pseudocount), float(step_size), str(dev))
        cache_c = self.__dict__.setdefault("_dual_em_consts", {})
        consts = cache_c.get(ckey)
        if consts is None:
            consts = cache_c[ckey] = torch.tensor(
                [float(pseudocount), float(step_size), 1.0 - float(step_size),
                 1e-38, 1e-30, 1e-12, 0.0], dtype = torch.float32, device = dev)
            if len(cache_c) > 8:                  # a schedule sweeping step sizes must not grow it
                for k in list(cache_c)[:-4]:
                    del cache_c[k]

        from pyjuice.layer.kernels import autotune

        # ---- pass 1: the per-node normalizer, into `cum` ----
        # What each partition's pass-2 launch needs, built here so the tile search and the member sum
        # are not repeated. It is a LOCAL: `plan["parts"]` is cached on the layer, structure only.
        staged = []
        for pid in range(layer.num_fw_partitions):
            part = plan["parts"][pid]
            if part is None:
                continue
            mode, addends, n_groups, group_cbs, denom_base = specs[pid]
            cids = layer.partitioned_cids[pid]
            num_edges = cids.size(1)
            n_rows = part["row_map"].numel()      # rows of THIS `ns`, not of the whole partition

            denom = addends[0]
            if len(addends) > 1:
                denom = denom.clone()
                for extra in addends[1:]:
                    denom = denom + extra

            common = dict(mparams = params, param_flows = param_flows, denom = denom,
                          cids = cids, pids = layer.partitioned_pids[pid],
                          pfids = layer.partitioned_pfids[pid],
                          nids = layer.partitioned_nids[pid], row_map = part["row_map"],
                          kcount = part["kcount"], num_edges = num_edges, n_groups = n_groups,
                          BLOCK_SIZE_M = block_size, GROUP_CBS = group_cbs,
                          consts = consts, cum_base = plan["cum_base"], denom_base = denom_base,
                          DENOM_MODE = mode, num_stages = 1)
            cands = self._em_tile_candidates(block_size, num_edges, n_rows, dev)
            akey = (num_edges, n_groups, block_size, group_cbs, n_rows, mode)

            def _cum(cfg, target_cum):
                tm, tk = cfg
                _dual_em_cum_kernel[(triton.cdiv(num_edges, tk), n_rows * (block_size // tm))](
                    cum = target_cum, TILE_SIZE_K = tk, TILE_SIZE_M = tm, **common)

            # BENCHMARK INTO A SCRATCH. This kernel is read-accumulate-write (`tl.atomic_add` into
            # `cum`), so timing it on the live buffer would add its contribution once per trial and
            # leave the normalizer several times too large -- a silent wrong answer, not a crash. No
            # scratch (OOM) means no tuning, never a tainted `cum`.
            cfg = autotune.cached(("dual_em_cum",) + akey)
            if cfg is None:
                # Only allocate the benchmark scratch when tuning will actually happen: `pick`
                # declines when `autotune.ENABLED` is false, so allocating first would have cost a
                # buffer the size of the output for nothing. MEASURED as 4 MB of an 8 MB transient at
                # a 4 MB `ns` with `PYJUICE_AUTOTUNE=0`, i.e. half the M-step's peak.
                sc = (autotune.scratch_like(cum)
                      if autotune.should_tune(("dual_em_cum",) + akey, len(cands)) else None)
                cfg = cands[0] if sc is None else autotune.pick(
                    ("dual_em_cum",) + akey, cands, lambda c: _cum(c, sc))
            _cum(cfg, cum)
            staged.append((common, cands, akey, n_rows))

        # ---- pass 2: the update, into `out`. Runs only once every partition's `cum` is complete,
        #      because a node's children may be split across partitions. ----
        for common, cands, akey, n_rows in staged:
            num_edges = common["num_edges"]

            def _upd(cfg, target_out):
                tm, tk = cfg
                _dual_em_update_kernel[(triton.cdiv(num_edges, tk),
                                        n_rows * (block_size // tm))](
                    cum = cum, out = target_out, out_base = out_base,
                    out_size = out_size_, KEEP_ZERO = 1 if keep_zero_params else 0,
                    OUT_BY_PAR = 1 if by_par else 0,
                    TILE_SIZE_K = tk, TILE_SIZE_M = tm, **common)

            # A pure overwrite, so re-running it is harmless -- but it is still benchmarked into a
            # scratch, because it reads `cum` and writes the buffer this function RETURNS, and a trial
            # left in `out` for a partition the real launch then skips would be returned as a result.
            key = ("dual_em_upd",) + akey + (bool(keep_zero_params), by_par)
            cfg = autotune.cached(key)
            if cfg is None:
                so = autotune.scratch_like(out) if autotune.should_tune(key, len(cands)) else None
                cfg = cands[0] if so is None else autotune.pick(key, cands,
                                                                lambda c: _upd(c, so))
            _upd(cfg, out)

        return out

    def compute_em_correction(self, ns, params, param_flows, denom_param_flows, step_size,
                              pseudocount, keep_zero_params, denom_sources = (), out = None):
        """
        The conditional dual-flow EM update for `ns`: `theta <- normalize(theta * F+ / F-)`, in the
        multiplicative MAP form. Returns the new parameters for `ns._param_range` as a flat tensor --
        or `None` when this parameterization does not request the denominator flow, which leaves the
        standard `normalize(F+)` update in place.

        IMPLEMENTED HERE, GENERICALLY, because it is the definition of dual-flow EM rather than a
        property of any one parameterization: it needs only `ns`'s compiled geometry and `F-`. So a
        parameterization that sets :attr:`requests_denom_param_flows` gets a correct M-step by
        implementing :func:`denom_flow_sizes`, accumulating its statistic in :func:`post_backward`,
        and expanding it in :func:`materialize_denom_flows` -- no M-step code of its own. Override
        :func:`_fused_em_correction` to go faster.

        Called once per EM step, after `param_flows` (F+, tied-fused) and `denom_param_flows` have been
        accumulated over the mini-batch. `params` still holds the PRE-update parameters: the PC SKIPS
        the standard M-step on `ns._param_range` and takes what is returned here instead, so the
        returned tensor replaces -- not adds to -- the standard update.

        With a single gate `F- = theta * sum(F+)`, so this collapses to `normalize(F+)` and the
        correction is an exact no-op -- a useful check when implementing a new parameterization.

        :param denom_sources: see :func:`materialize_denom_flows`.
        :param out: a `pe - ps` tensor to write the result into, in place of allocating one. The PC
            passes `params[ps:pe]`, i.e. the very range being updated, which is safe because every
            element of it is written exactly once and this is the only writer.
        """
        if not self.requests_denom_param_flows:
            return None

        fused = self._fused_em_correction(ns, params, param_flows, denom_param_flows, step_size,
                                          pseudocount, keep_zero_params, denom_sources, out = out)
        if fused is not None:
            return fused

        import torch

        ps, pe = ns._param_range
        pfs, pfe = ns._param_flow_range
        bs, cbs = ns.block_size, ns.ch_block_size
        E = ns.edge_ids.size(1)

        # Node-level layout `[E, ch_block_size, block_size]` (edge block, child-in-block,
        # node-in-block) -- the same one `params` uses.
        theta = params[ps:pe].reshape(E, cbs, bs)
        Fp = param_flows[pfs:pfe].reshape(E, cbs, bs)
        Fm = self.materialize_denom_flows(ns, params, denom_param_flows,
                                          denom_sources).reshape(E, cbs, bs)

        # A node's children are all edge blocks of its node block, times `ch_block_size`; the per-node
        # normalizer sums over exactly those. `K` (children per node) sets the pseudocount split.
        nb = ns.edge_ids[0].to(device = theta.device, dtype = torch.long)
        eblk_per_nb = torch.bincount(nb, minlength = ns.num_node_blocks)
        K = (eblk_per_nb[nb].to(theta.dtype) * cbs)[:, None, None]          # [E,1,1]

        # ratio = (F+ + pc/K) / (F- + pc*theta) -- the MAP form (denominator `pc*theta`, not `pc`), so
        # a never-observed child floors instead of underflowing.
        ratio = (Fp + pseudocount / K) / (Fm + pseudocount * theta).clamp_min(1e-38)
        flow = theta * ratio                                                # [E, cbs, bs]

        # cum[node block, m] = (1-s) + s * sum over the node's children of theta*ratio
        cum = torch.zeros(ns.num_node_blocks, bs, device = theta.device, dtype = theta.dtype)
        cum.index_add_(0, nb, flow.sum(dim = 1))
        cum = ((1.0 - step_size) + step_size * cum).clamp_min(1e-38)

        new_theta = theta * ((1.0 - step_size) + step_size * ratio) / cum[nb][:, None, :]
        new_theta = new_theta.clamp_min(1e-30)                              # momentum-underflow guard
        if keep_zero_params:
            new_theta = torch.where(theta < 1e-12, torch.zeros_like(new_theta), new_theta)
        new_theta = new_theta.reshape(-1)
        if out is not None:
            out.copy_(new_theta)
            return out
        return new_theta

    def sample_layer(self, layer, ns_tensors, node_mars, element_mars, params, node_samples,
                     element_samples, rows, erows, seed_ptr, conditional: bool = False,
                     **kwargs) -> None:
        """
        Draw one child per live sample of every frontier ROW this layer owns, under the EFFECTIVE
        parameters.

        The rows come from the circuit's structural frontier layout
        (:mod:`pyjuice.queries.sampling.scope_plan`): `rows` are the `node_samples` rows this layer
        owns and `erows` where each one's drawn child is written. A row holds `-1` where its scope is
        not on that sample's path, so liveness is a mask rather than a shape.

        Everything else matches :func:`sample_layer_pairs`, which this replaces -- see there for what
        the draw has to be, and why the normalizer cancels out of it.
        """
        raise NotImplementedError(
            f"`{self.get_signature()}` does not implement ancestral sampling against the structural "
            f"frontier layout. Sampling it with the shared-parameter kernel would ignore the "
            f"per-sample parameters and quietly return samples from a different distribution than "
            f"the forward pass scores, so it is refused instead."
        )

    def sample_layer_pairs(self, layer, ns_tensors, node_mars, element_mars, params, node_samples,
                           element_samples, ind_target, ind_n, ind_b, conditional: bool = False,
                           rnd = None, rnd_offset = 0, **kwargs) -> None:
        """
        Draw one child per selected node of this layer, under the EFFECTIVE parameters.

        The top-down ancestral pass (:func:`pyjuice.queries.sample`) reaches this once per sum layer
        that was given external tensors, in place of the shared-parameter kernel -- which would
        otherwise draw from `theta_shared` and return samples from a different distribution than the
        forward pass scores. So this is the sampling counterpart of :func:`forward_layer`, and it
        owns the whole draw rather than correcting one: a normalized categorical distribution is not
        recoverable from a draw already made under different weights.

        The distribution to draw from is the node's effective conditional. Note that the normalizer
        cancels: for `theta_b[n,c] = w_b[n,c] / Z_b[n]`, drawing `c` in proportion to `w_b[n,c]`
        (times `exp(element_mars[c,b])` when conditioning) is the same draw, so a parameterization
        needs no normalizer from its forward pass -- only its own per-sample weights.

        Not implemented by default. A parameterization that leaves it that way makes
        :func:`pyjuice.queries.sample` raise rather than silently sample the shared parameters.

        :param ns_tensors: `[(ns_info, tensors), ...]` for the nodes that were given external
                           tensors, as in :func:`forward_layer`

        :param node_samples: `[scopes, num_samples]`, the sampler's frontier of selected node ids

        :param element_samples: `[scopes, num_samples]`, where the drawn child ids are written

        :param ind_target: flat index into `element_samples` at which each selected node's drawn
                           child belongs

        :param ind_n: index into `node_samples`' first axis of each selected node
        :param ind_b: sample (column) index of each selected node

        :param conditional: whether to condition on the evidence a forward pass left in
                            `element_mars`. Unconditionally the child of `n` is drawn in proportion
                            to the effective parameters alone; conditionally, to those times
                            `exp(element_mars[c,b])`.
        """
        raise NotImplementedError(
            f"`{self.get_signature()}` does not implement ancestral sampling. Sampling it with the "
            f"shared-parameter kernel would ignore the per-sample parameters entirely and quietly "
            f"return samples from a different distribution than the forward pass scores, so it is "
            f"refused instead. Implement `sample_layer`, or draw samples without supplying external "
            f"parameters, which samples the shared parameters and is what an ungated forward pass "
            f"also computes."
        )

    def _get_constructor(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, ExternalSumParams) and self.get_signature() == other.get_signature()

    def __hash__(self):
        return hash(self.get_signature())

    def __repr__(self):
        return self.get_signature()
