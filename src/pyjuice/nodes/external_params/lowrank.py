from __future__ import annotations

import warnings

import torch
from typing import Any, Optional, Tuple

from .external_params import ExternalSumParams


_BUFFER_KWARG = None
_GRAD_BUFFER_KWARG = None


def _grad_buffer_kwarg() -> str:
    """The gradient-buffer kwarg name, resolved once (see :func:`_buffer_kwarg`)."""
    global _GRAD_BUFFER_KWARG
    if _GRAD_BUFFER_KWARG is None:
        from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_GRAD_BUFFER_KWARG
        _GRAD_BUFFER_KWARG = EXTERNAL_PARAMS_GRAD_BUFFER_KWARG
    return _GRAD_BUFFER_KWARG


def _buffer_kwarg() -> str:
    """The staging-buffer kwarg name, resolved once (a per-call import showed up in the profile)."""
    global _BUFFER_KWARG
    if _BUFFER_KWARG is None:
        from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_BUFFER_KWARG
        _BUFFER_KWARG = EXTERNAL_PARAMS_BUFFER_KWARG
    return _BUFFER_KWARG


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

    def __init__(self, rank: int, tile_m: Optional[int] = None, tile_b: Optional[int] = None,
                 tile_c: Optional[int] = None, tile_bc: Optional[int] = None,
                 tile_n: Optional[int] = None, variant: Optional[str] = None,
                 tie_external: bool = False):
        super(LowRankSumParams, self).__init__()

        assert isinstance(rank, int) and rank >= 1, f"`rank` must be a positive integer, got {rank}."

        self.rank = rank

        # Kernel launch tiles. `tile_m` is the one that matters: `logW` / `logA` do not depend on the
        # node index, so they are recomputed once per node tile and the arithmetic overhead relative to
        # the shared sum kernel is `2 * rank / tile_m + rank / ch_block_size`. Large `tile_m` is cheap
        # but yields fewer programs, so it trades against occupancy. `None` picks a heuristic.
        self.tile_m, self.tile_b, self.tile_c, self.tile_bc = tile_m, tile_b, tile_c, tile_bc

        # Node tile of the backward's V pass. Like `tile_c` in the forward, this is really a BLOCK-COUNT
        # knob: on a dense transition the reduction axis is the only wide one, so the tile size sets how
        # many blocks the kernel gets. MEASURED: too large and the grid is a few dozen blocks on 188 SMs.
        self.tile_n = tile_n

        # Share ONE factor pair across every copy of a tied node, instead of one per copy. For a chain
        # like an HMM that divides the unique factor data by the number of timesteps -- which also brings
        # it inside L2, turning the per-timestep re-reads into cache hits -- at the cost of accumulating
        # the gradient across copies rather than storing it.
        self.tie_external = bool(tie_external)

        # Scratch for the multi-launch variants, keyed by name and reused across calls. One descriptor is
        # shared by every node that uses it (all 31 timesteps of a tied HMM chain), and the launches are
        # sequential, so a single set of buffers serves them all -- which also keeps allocation out of
        # the per-layer path, where 4 allocations x 31 layers per forward is pure overhead, and keeps
        # `torch.empty` out of a CUDA-graph capture.
        self._scratch = dict()

        # Per-call bookkeeping is cached because it is re-derived identically on every step: the launch
        # tiles, the staging-buffer base offset, and the scratch views. MEASURED at 31 layers x 2
        # launches, rebuilding them cost ~16 us per layer -- more than the two kernel launches
        # themselves (~11 us) -- so on a deep chain it, not the kernels, dominated the correction.
        self._scratch_views = dict()
        self._tiles_cache = dict()
        self._base_cache = dict()
        self._alloc_device = None

        # Fully-resolved CUDA launch arguments per layer. Everything in them is derived from the layer's
        # compiled tables and the PC's persistent buffers, so it is identical on every step -- but
        # re-deriving it cost ~27 us per layer against a ~4 us kernel call (nn.Module attribute lookups,
        # per-call imports, the applicability check, tile selection), i.e. the bookkeeping dominated the
        # work by 7x. Keyed on the buffer identities it was built for, so a reallocation rebuilds it.
        self._plan = dict()
        self._bw_plan = dict()
        self._auto_variant = None

        # Which forward launch structure to use. `logW` / `logA` do not depend on the node index, so the
        # three differ in whether they recompute them per node tile (`"grid"`), share them in registers
        # by walking the node tiles inside one program (`"hoist"`, single edge block only), or stage them
        # through a small global scratch (`"scratch"`, two launches). They compute the SAME quantity;
        # this only trades redundancy against parallelism, and the best choice is shape-dependent and
        # still being measured -- so the default stays the structure that is currently validated.
        assert variant is None or variant in ("cuda", "split2"), \
            f"Unknown variant {variant}; expected None (auto), \"cuda\" or \"split2\"."

        # `None` means auto: the CUDA path when its toolchain is available, else the best Triton form.
        # Only the CUDA path stages logW / logA / logZ, so it is also what the backward requires.
        self.variant = variant

    def get_signature(self) -> str:
        # `tie_external` changes the storage layout, so nodes that disagree about it must not share a
        # layer -- putting it in the signature is what keeps them apart.
        return f"LowRank_r{self.rank}_tied" if self.tie_external else f"LowRank_r{self.rank}"

    def storage_owner(self, ns):
        """With `tie_external`, every copy of a tied node reads one shared factor pair -- the source's."""
        if self.tie_external and ns.is_tied():
            return ns.get_source_ns()

        return ns

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

    def _build_cuda_plan(self, layer, ns_tensors, node_mars, element_mars, external_params):
        """Resolve every per-layer CUDA launch argument once. Returns `(module, [args, ...])`."""
        from .kernels.c import get_module

        mod = get_module()
        if mod is None or not self._kernel_applicable(layer, ns_tensors, node_mars):
            return None

        block_size = layer.block_size
        ch_block_size = layer.external_node_infos[0].ch_block_size
        batch_size = node_mars.size(1)

        first_view = ns_tensors[0][1][0]
        ext_base = ((first_view.data_ptr() - external_params.data_ptr())
                    // external_params.element_size()) // batch_size

        tiles = self._tiles(block_size, ch_block_size, batch_size)
        tile_m = min(self.tile_m or 8, block_size)
        # 16 rather than the Triton default: these tiles set the BLOCK COUNT, and 64 left the grid at a
        # few dozen blocks on 188 SMs. MEASURED best over {4, 8, 16, 64} on the HMM.
        tile_c = min(self.tile_c or 16, ch_block_size)

        # Thread blocks are `rank x tb1` and `tile_m x tb2`; keep both within the 1024-thread limit
        tb1 = max(1, min(tiles["TILE_B"], batch_size))
        while self.rank * tb1 > 1024:
            tb1 //= 2
        tb2 = max(1, min(tiles["TILE_BC"] or tiles["TILE_B"], batch_size))
        while tile_m * tb2 > 1024:
            tb2 //= 2

        self._alloc_device = node_mars.device
        n_ctiles = -(-ch_block_size // tile_c)
        dev = node_mars.device

        calls, state = [], []
        for partition_id in range(layer.num_fw_partitions):
            nids = layer.partitioned_nids[partition_id]
            cids = layer.partitioned_cids[partition_id]
            xu, xv = layer.ext_xu[partition_id], layer.ext_xv[partition_id]

            rows, num_eblks = xu.size(0), xu.size(1)
            numel = rows * num_eblks * self.rank * n_ctiles * batch_size

            # PER-LAYER (not shared scratch): the backward reads these after every layer's forward has
            # run, so they must survive the rest of the forward pass. logW/logA are tiny; logZ is
            # node-sized, which is still far cheaper than recomputing it from `V` in the backward.
            log_w = torch.empty([rows * num_eblks * self.rank * batch_size], dtype = torch.float32,
                                device = dev)
            log_a = torch.empty_like(log_w)
            log_z = torch.empty([rows * block_size * batch_size], dtype = torch.float32, device = dev)
            state.append((nids, cids, xu, xv, log_w, log_a, log_z))

            calls.append((nids, cids, xu, xv,
                          self._alloc(f"pw{partition_id}", numel),
                          self._alloc(f"pa{partition_id}", numel),
                          log_w, log_a, log_z,
                          block_size, ch_block_size, self.rank, ext_base, tile_c, tile_m, tb1, tb2))

        self._bw_plan[id(layer)] = (block_size, ch_block_size, ext_base, batch_size, state)

        return mod, calls

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

        if self._resolved_variant() == "cuda":
            # Fast path: one resolved host call per partition, no per-step bookkeeping
            key = id(layer)
            ptrs = (node_mars.data_ptr(), element_mars.data_ptr(), node_mars.size(1))

            entry = self._plan.get(key, None)
            if entry is None or entry[0] != ptrs:
                buf = kwargs.get(_buffer_kwarg(), None)
                entry = (ptrs, None if buf is None else
                         self._build_cuda_plan(layer, ns_tensors, node_mars, element_mars, buf))
                self._plan[key] = entry

            if entry[1] is not None:
                mod, calls = entry[1]
                buf = kwargs[_buffer_kwarg()]
                for args in calls:
                    mod.lowrank_forward(node_mars, element_mars, buf, *args)
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

        # `xu` / `xv` are offsets WITHIN this layer's block of the staging buffer, so they need the
        # global offset of that block added. The layer's nodes occupy a contiguous range of the buffer
        # (the PC lays nodes out in layer order), and `ns_tensors` is in `layer.nodes` order, so the
        # first staged view marks where the block starts.
        batch_size = node_mars.size(1)
        first_view = ns_tensors[0][1][0]

        base_key = (id(layer), batch_size, external_params.data_ptr(), first_view.data_ptr())
        ext_base = self._base_cache.get(base_key, None)
        if ext_base is None:
            ext_base = ((first_view.data_ptr() - external_params.data_ptr())
                        // external_params.element_size()) // batch_size
            self._base_cache[base_key] = ext_base

        self._alloc_device = node_mars.device

        tiles_key = (block_size, ch_block_size, batch_size)
        tiles = self._tiles_cache.get(tiles_key, None)
        if tiles is None:
            tiles = self._tiles(block_size, ch_block_size, batch_size)
            self._tiles_cache[tiles_key] = tiles

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
                ext_base = ext_base,
                variant = self._resolved_variant(),
                alloc = self._alloc,
                **tiles,
            )

    def _resolved_variant(self) -> str:
        """The variant to actually use, resolving `None` once."""
        if self.variant is not None:
            return self.variant

        if self._auto_variant is None:
            from .kernels.c import is_available

            if is_available():
                self._auto_variant = "cuda"
            else:
                # The forward degrades silently to Triton, but the BACKWARD has no Triton implementation
                # -- and it is the CUDA forward that stages logW / logA / logZ for it. Say so now rather
                # than at the first `backward()`, which is a long way from the cause.
                warnings.warn(
                    "pyjuice: the low-rank CUDA extension is unavailable, so the forward will use the "
                    "Triton fallback and the BACKWARD will not be available for external low-rank "
                    "parameters. This usually means `nvcc` or `ninja` is not on PATH -- check the "
                    "compile warning above.", RuntimeWarning)
                self._auto_variant = "split2"

        return self._auto_variant

    def _alloc(self, name: str, numel: int):
        """
        A scratch buffer of at least `numel` elements, as a CACHED view.

        Slicing (`buf[:numel]`) is an ATen call, so re-slicing per launch per layer showed up as real
        per-step cost; the view is memoized and only rebuilt when the backing buffer is reallocated.
        """
        device = self._alloc_device
        key = (name, numel, device)

        view = self._scratch_views.get(key, None)
        if view is not None:
            return view

        buf = self._scratch.get(name, None)
        if buf is None or buf.numel() < numel or buf.device != device:
            buf = torch.empty([numel], dtype = torch.float32, device = device)
            self._scratch[name] = buf
            # Views into the old buffer are now stale
            self._scratch_views = {k: v for k, v in self._scratch_views.items() if k[0] != name}

        view = buf[:numel]
        self._scratch_views[key] = view

        return view

    def pre_backward_layer(self, layer, ns_tensors, node_flows, element_flows, node_mars,
                           element_mars, params, **kwargs) -> None:
        """
        Turn `node_mars` into `logT` over this layer's nodes, so the STOCK backward -- run next,
        unmodified -- produces exactly the shared parameters' element and parameter flows.

        `theta_tilde = (theta_shared + Delta)/Z` and `S = T/Z`, so the `Z` cancels and the shared
        term's flow is `f * theta_shared * p / T`. That is what the stock kernel computes when it reads
        `logT` where it expects `log S`. Undone in :func:`post_backward_layer`.
        """
        if not kwargs.get("logspace_flows", False):
            raise NotImplementedError(
                "The low-rank backward requires `logspace_flows = True` (the default): the child-flow "
                "correction is combined in log space."
            )

        entry = self._bw_plan.get(id(layer), None)
        if entry is None:
            from .kernels.c import is_available

            if not is_available():
                raise NotImplementedError(
                    "The low-rank backward requires the CUDA extension, which is unavailable on this "
                    "system (usually `nvcc` or `ninja` not on PATH). The forward runs on the Triton "
                    "fallback, but only the CUDA forward stages logW / logA / logZ, and there is no "
                    "Triton backward."
                )

            raise NotImplementedError(
                f"The low-rank backward did not run on this layer because the CUDA forward kernel does "
                f"not apply to its shape. It requires a single child block size, a power-of-two rank "
                f"<= 64 (got {self.rank}), block_size >= 16, ch_block_size >= 16, and batch >= 16, with "
                f"external parameters supplied for every node of the layer."
            )
        from .kernels.c import get_module
        mod = get_module()

        block_size, _, _, _, state = entry
        for nids, _, _, _, _, _, log_z in state:
            mod.lowrank_shift_logz(node_mars, nids, log_z, block_size, 1.0)

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                            node_mars, element_mars, params, param_flows = None, **kwargs) -> None:
        """
        Add the external contribution to the child flows, write `dLL/dU` and `dLL/dV`, and restore
        `node_mars`.

        The restore runs in a `finally`: `node_mars` is shared with the rest of the circuit, so leaving
        it in the `logT` form would silently corrupt every layer evaluated afterwards.
        """
        from .kernels.c import get_module
        mod = get_module()

        entry = self._bw_plan[id(layer)]
        block_size, ch_block_size, ext_base, batch_size, state = entry

        try:
            ext = kwargs.get(_buffer_kwarg(), None)
            grad_ext = kwargs.get(_grad_buffer_kwarg(), None)

            if ext is not None and grad_ext is not None:
                self._alloc_device = node_mars.device

                tile_n = min(self.tile_n or 16, block_size)
                tb = max(1, min(32, batch_size))
                while self.rank * tb > 1024:
                    tb //= 2
                # Pass C's block is `tile_c x tb` threads (one per child/batch pair)
                tile_c = min(64, ch_block_size)
                while tile_c * tb > 1024:
                    tile_c //= 2
                n_ntiles = -(-block_size // tile_n)

                for nids, cids, xu, xv, log_w, log_a, log_z in state:
                    slots = xu.size(0) * xu.size(1) * self.rank
                    mod.lowrank_backward(
                        node_flows, element_flows, node_mars, element_mars, ext, grad_ext,
                        nids, cids, xu, xv, log_w, log_a, log_z,
                        self._alloc("p_lp", slots * n_ntiles * batch_size),
                        self._alloc("p_lq", slots * n_ntiles * batch_size),
                        self._alloc("log_p", slots * batch_size),
                        self._alloc("log_q", slots * batch_size),
                        block_size, ch_block_size, self.rank, ext_base, tile_n, tile_c, tb,
                        self.tie_external,
                    )
        finally:
            for nids, _, _, _, _, _, log_z in state:
                mod.lowrank_shift_logz(node_mars, nids, log_z, block_size, -1.0)

    def _tiles(self, block_size, ch_block_size, batch_size) -> dict:
        """
        Launch tiles, either as configured or from a heuristic.

        The heuristic keeps the node tile as large as it can while still filling the GPU: it grows
        `tile_m` until the program count would drop below a target, which bounds the redundant
        recomputation of `logW` / `logA` without starving occupancy.
        """
        # MEASURED (HMM seq32 / nl1024 / rank16 / batch64, RTX PRO 6000): `tile_b` dominates, not
        # `tile_m`. The kernel's working tile is `[tile_c, rank, tile_b]`, so register pressure grows
        # with the product and a large `tile_b` spills badly -- tile_b 8 vs 32 was 2.95x vs 9.30x on the
        # same shape, and raising tile_m past 32-64 also cost (128 -> 3.92x). So keep both modest.
        tile_b = self.tile_b if self.tile_b is not None else min(8, batch_size)
        tile_m = self.tile_m if self.tile_m is not None else min(32, block_size)
        tile_c = self.tile_c if self.tile_c is not None else min(64, ch_block_size)

        tile_bc = self.tile_bc if self.tile_bc is not None else 0

        return {"TILE_M": tile_m, "TILE_B": tile_b, "TILE_C": tile_c, "TILE_BC": tile_bc}

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
