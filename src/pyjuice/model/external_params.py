"""
External parameters in :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (:class:`_ExternalParamsMixin`, one of its
bases): the caller's per-sample tensors -- per node, or one per named group of nodes -- checked and staged into the
one buffer the layers read (a tiled transpose where the layouts differ, :func:`_staged_copy`), and their gradients
written back in the layout they came in.
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Sequence

from pyjuice.nodes import CircuitNodes
from pyjuice.layer import StagedExternalParams, EXTERNAL_PARAMS_KWARG, EXTERNAL_PARAMS_GRAD_KWARG, \
                          EXTERNAL_PARAMS_BUFFER_KWARG, EXTERNAL_PARAMS_GRAD_BUFFER_KWARG
from pyjuice.layer.external_sum_layer import validate_external_tensors
from pyjuice.nodes.external_params.external_params import ExternalSumParams


def _staging_transpose_fn():
    """
    The tiled-transpose backend for staging: the CUDA extension if it built, Triton otherwise, or None
    to fall back to a generic strided `copy_`.

    Behind ONE function so a test can force each tier. The fallbacks are otherwise unreachable on any
    machine where the extension compiled -- which is how a shape bug lived in one of them: staging a
    whole GROUP hands this a flat destination, the per-node path hands it a storage-shaped one, and the
    fallback only ever handled the second.
    """
    from pyjuice.nodes.external_params.kernels.c import get_module
    mod = get_module()
    if mod is not None:
        return mod.staging_transpose

    try:
        from pyjuice.nodes.external_params.kernels.staging import staging_transpose_triton
        return staging_transpose_triton
    except Exception:
        return None


def _staged_copy(plain_d: list, plain_s: list, fast: list) -> None:
    """
    Issue the staging copies: a tiled transpose for the pairs that are one, a batched `copy_` for the
    rest.

    `Tensor.copy_` on a transposed view goes through TensorIterator, which handles arbitrary strides but
    does not tile, so one side of the transpose is uncoalesced -- 8.2 us against 2.1 us for the same
    4 MB. Staging is worth optimizing because it was 37-59% of the whole cost of applying a gate at
    batch 256.

    Which pairs qualify is decided by the CALLER from the parameterization's declared layout, not by
    inspecting strides here: this runs on every forward, and building a permuted view per tensor just to
    test it cost more Python than the kernel saved in GPU time.
    """
    if fast:
        fn = _staging_transpose_fn()
        for dst, src in fast:
            batch = src.size(0)
            n = src.numel() // max(batch, 1)
            if fn is not None:
                fn(dst, src, batch, n)
            else:
                # NORMALIZE both shapes rather than assuming either. `dst` is flat for a whole-group
                # stage (`_group_fast_stage` returns a `narrow` of the buffer) and storage-shaped for a
                # per-node one, so the old `src.permute(...)` matched only the second -- a group stage
                # raised a shape mismatch here. It never showed up because this branch is unreachable
                # wherever the CUDA extension built, which is everywhere it was tested.
                #
                # `view`, not `reshape`: a non-contiguous destination has to raise rather than quietly
                # copy into a temporary that is then discarded.
                plain_d.append(dst.view(n, batch))
                plain_s.append(src.view(batch, n).permute(1, 0))

    if plain_d:
        torch._foreach_copy_(plain_d, plain_s)


class _ExternalParamsMixin:
    """External parameters, for :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (one of its bases)."""

    def _write_external_params_grads(self) -> None:
        """
        Copy the gradients into the buffers the caller supplied to `sum_external_params_grad`.

        The kernels write the PC's internal buffer, which is in STORAGE layout (batch innermost); the
        caller's tensors are in the layout they supplied the gates in. `from_storage` is what maps
        between the two, so this is the exact inverse of what staging does on the forward, batched into
        one `_foreach_copy_` rather than one op per node.
        """
        dsts = getattr(self, "_external_grad_dsts", None)
        if not dsts:
            return None

        d, s = [], []
        # A node's views of the gradient buffer are the same objects every call at a batch size
        # (`_external_params_views` caches them), and `from_storage` is a layout mapping of them: kept per node
        # against those views' identity (~3 us a node, every backward).
        src_cache = self.__dict__.setdefault("_external_grad_src_cache", {})
        for ns, tensors in dsts.items():
            views = self._staged_external_params_grad[ns]
            hit = src_cache.get(ns)
            if hit is not None and hit[0] is views:
                src = hit[1]
            else:
                src = ns.external_params.from_storage(ns, views)
                src_cache[ns] = (views, src)
            if torch.is_tensor(tensors):
                tensors = (tensors,)
            for dst, one in zip(tensors, src):
                d.append(dst)
                s.append(one)

        if d:
            torch._foreach_copy_(d, s)

        self._external_grad_dsts = None

    def _external_params_layout(self, batch_size: int):
        """
        Slot layout of the external-parameter staging buffer at `batch_size`.

        Every node that takes external parameters gets its own contiguous slot for each tensor its
        parameterization declares, laid out in `external_params_nodes` order. The shapes come from the
        descriptor, so the layout is whatever the parameterization says it is; the buffer just gives
        every tensor a known, contiguous home that the kernels can address and that keeps a stable
        pointer across calls (which is what lets the caller pass freshly allocated tensors, and lets
        the layers run under CUDA-graph capture).

        Cached per batch size, since that is the only thing the layout depends on.

        :returns: `(total_numel, {ns: [(offset, shape), ...]})`
        """
        layout = self._external_params_layouts.get(batch_size, None)
        if layout is not None:
            return layout

        # Two passes so that a node may share another node's slots (see
        # `ExternalSumParams.storage_owner`): allocate for the owners first, then alias the rest onto
        # them. Aliased nodes still appear in the map, so the layers and kernels need no special case --
        # they look up their own `ns` and simply find the same memory.
        offset = 0
        ns2slots = dict()
        for ns in self.external_params_nodes:
            if ns.external_params.storage_owner(ns) is not ns:
                continue

            slots = []
            for shape in ns.external_params.storage_shapes(ns, batch_size):
                slots.append((offset, tuple(shape)))
                offset += math.prod(shape)

            ns2slots[ns] = slots

        for ns in self.external_params_nodes:
            owner = ns.external_params.storage_owner(ns)
            if owner is not ns:
                assert owner in ns2slots, \
                    f"`storage_owner` of {ns} is not itself an external-parameter node of this circuit."
                ns2slots[ns] = ns2slots[owner]

        layout = (offset, ns2slots)
        self._external_params_layouts[batch_size] = layout

        return layout

    def _external_params_views(self, name: str, batch_size: int, set_value: Optional[float] = None):
        """
        (Re)allocate the staging buffer `name` for `batch_size` and return one contiguous view per
        tensor slot, as `{ns: (view, ...)}`.

        The views are what the layers and kernels consume, so a parameterization never has to know
        that a shared buffer exists -- it just receives correctly shaped, contiguous, correctly placed
        tensors.
        """
        total_numel, ns2slots = self._external_params_layout(batch_size)

        self._init_buffer(name = name, shape = (total_numel,), set_value = set_value)

        buffer = self.__dict__[name]

        # Building these is pure Python -- one slice and one view per node -- and it runs on every
        # forward. On a 32-timestep gated HMM it was 67 us of the 111 us the whole staging cost, six
        # times what the copy itself takes, so it is cached. The views depend only on the layout (fixed
        # per batch size) and the buffer's identity, so the check is `is` against the tensor the
        # allocation just returned: a reallocation -- a new batch size, a `.to(device)` -- hands back a
        # different object and rebuilds. `_init_buffer` still runs every time, since it is also what
        # zeroes the gradient buffer.
        #
        # The returned dict is SHARED between calls. Both callers copy it into a `StagedExternalParams`
        # before deleting entries, so nothing mutates it; a new caller must do the same.
        key = (name, batch_size)
        cached = self._external_params_views_cache.get(key)
        if cached is not None and cached[0] is buffer:
            return cached[1]

        views = {ns: tuple(buffer[offset:offset + math.prod(shape)].view(shape)
                           for offset, shape in slots)
                 for ns, slots in ns2slots.items()}
        self._external_params_views_cache[key] = (buffer, views)
        return views

    def register_external_params_group(self, name: str, nodes: Sequence[CircuitNodes],
                                       dim: int = 1) -> None:
        """
        Address several nodes' external parameters with ONE concatenated tensor, under a name.

        Nodes that share a parameterization usually share a shape too -- an HMM's per-timestep
        transitions being the case this exists for -- and a head producing them naturally emits one
        tensor already. Without a group the caller has to slice that tensor apart before every call,
        once per node, only for the PC to copy each piece separately.

        Once registered, the name is usable wherever an `ns` key is:

        .. code-block:: python

            pc.register_external_params_group("router", [t0, t1, t2])
            pc(x, sum_external_params = {"router": phi})        # phi: [B, 3 * Nk, Ck]

        and may be mixed freely with per-node entries in the same dict.

        Members are split along `dim` by their OWN extents, so they need not have equal shapes there --
        only on every other axis. Nothing is copied: each member receives a view.

        :note: order is significant. `nodes[i]` receives the i-th slice, so the list is the contract.

        :note: for a parameterization with several tensors per node (low-rank factors, say) the value
               is a tuple of concatenated tensors, one per slot, each split the same way.

        :param name: the key to use in `sum_external_params`
        :type name: str

        :param nodes: the nodes this group addresses, in slice order
        :type nodes: Sequence[CircuitNodes]

        :param dim: axis of the concatenation. Defaults to 1, the first non-batch axis; `0` is the
                    batch and is rejected.
        :type dim: int
        """
        assert isinstance(name, str) and name != "", "An external-parameter group needs a non-empty name."
        assert name not in self.external_params_groups, \
            f"External-parameter group '{name}' is already registered."
        assert len(nodes) > 0, f"External-parameter group '{name}' is empty."
        assert dim != 0, \
            "`dim = 0` is the batch axis; a group concatenates along a PARAMETER axis (default 1)."

        seen = set()
        for ns in nodes:
            assert ns in self.external_params_nodes, \
                f"External-parameter group '{name}' contains a node this PC did not compile with an " \
                f"external parameterization. Construct it with `pyjuice.summate(..., external_params = ...)`."
            if id(ns) in seen:
                raise AssertionError(f"External-parameter group '{name}' lists the same node twice.")
            seen.add(id(ns))

        sigs = {ns.external_params.get_signature() for ns in nodes}
        assert len(sigs) == 1, \
            f"External-parameter group '{name}' mixes parameterizations {sigs}; one concatenated " \
            f"tensor cannot serve nodes whose layouts differ."

        # Members that SHARE storage (a tied node and its copies) are fed once, by the owner. Handing
        # such a pair two different slices is a contradiction, and silently letting one win would apply
        # the wrong gate to the other -- so it is rejected here rather than resolved arbitrarily.
        owners = [ns.external_params.storage_owner(ns) for ns in nodes]
        if len(set(id(o) for o in owners)) != len(owners):
            raise AssertionError(
                f"External-parameter group '{name}' contains nodes that share one external tensor "
                f"(`tie_external`). They are supplied once, through the storage owner, so they cannot "
                f"also take separate slices of a concatenated tensor."
            )

        # Shapes must agree everywhere except along `dim`. Checked once, at registration, on a nominal
        # batch of 1: the shapes are affine in the batch, so the comparison holds for every batch.
        shapes = [ns.external_params.tensor_shapes(ns, 1) for ns in nodes]
        n_slots = len(shapes[0])
        assert all(len(sh) == n_slots for sh in shapes), \
            f"External-parameter group '{name}' mixes nodes with different numbers of tensors."
        for slot in range(n_slots):
            ref = list(shapes[0][slot])
            nd = len(ref)
            assert -nd <= dim < nd, \
                f"`dim = {dim}` is out of range for slot {slot} of group '{name}', whose tensors " \
                f"have {nd} axes."
            axis = dim % nd
            for i, sh in enumerate(shapes[1:], start = 1):
                got = list(sh[slot])
                if [d for a, d in enumerate(got) if a != axis] != [d for a, d in enumerate(ref) if a != axis]:
                    raise AssertionError(
                        f"External-parameter group '{name}': node {i}'s slot-{slot} shape {tuple(got)} "
                        f"disagrees with node 0's {tuple(ref)} on an axis other than {axis}."
                    )

        self.external_params_groups[name] = (list(nodes), int(dim))

    def unregister_external_params_group(self, name: str) -> None:
        """Remove a group registered by :func:`register_external_params_group`."""
        assert name in self.external_params_groups, \
            f"No external-parameter group named '{name}'."
        del self.external_params_groups[name]

    def _group_fast_stage(self, name: str, tensors, views: dict, batch_size: int,
                          buffer: Optional[torch.Tensor] = None):
        """
        `(destination, source)` staging a WHOLE group in one transpose, or None if it cannot.

        Splitting a group into per-node views is correct but costs: each slice of a batch-first
        concatenated tensor is strided, so every member falls off the tiled-transpose path onto the
        generic copy, and the group ends up SLOWER to stage than the per-node form it replaces.

        It does not have to. When the members' slots happen to be adjacent in the staging buffer and
        in the group's own order -- which is what compiling an HMM's transitions in depth order
        produces -- the whole group is one contiguous destination, and `[B, sum Nk, Ck]` maps onto it
        by exactly the transpose the fast kernel already does. One copy instead of T.

        Every precondition is checked rather than assumed; anything unmet just returns None and the
        per-member path runs.
        """
        nodes, dim = self.external_params_groups[name]

        # `dim = 1` is what makes the concatenated axis the OUTERMOST storage axis, so that
        # concatenating along it is the same thing as laying the members out back to back.
        if dim != 1 or len(nodes) < 2:
            return None

        ep = nodes[0].external_params
        perm = ep.storage_perm()
        if not (perm is not None
                and type(ep).to_storage is ExternalSumParams.to_storage
                and tuple(perm) == tuple(range(1, len(perm))) + (0,)):
            return None

        if torch.is_tensor(tensors):
            tensors = (tensors,)
        if len(tensors) != 1:                       # one slot per node; the multi-slot case splits
            return None
        cat = tensors[0]
        if not (cat.is_contiguous() and cat.dtype == torch.float32 and cat.dim() >= 3):
            return None

        base = views[nodes[0]][0]
        # The buffer being staged INTO, which is not always the forward's: an unconditional
        # `queries.sample()` stages into a private slab so it cannot disturb what the forward left
        # for its backward, and `self.external_params` is then `None` on a circuit that has only
        # ever been sampled from.
        buf = buffer if buffer is not None else self.external_params
        elem = buf.element_size()
        total = 0
        for ns in nodes:
            d = views[ns][0]
            if d.data_ptr() != base.data_ptr() + total * elem:
                return None                          # not adjacent, or not in the group's order
            total += d.numel()

        if total != cat.numel():
            return None
        start = (base.data_ptr() - buf.data_ptr()) // elem
        return buf.narrow(0, start, total), cat

    def _expand_external_params_groups(self, ns2tensors: dict, batch_size: int) -> dict:
        """
        Replace every group-name key with one entry per member, holding a VIEW of the caller's tensor.

        Splitting rather than copying is the point: the per-node staging below already accepts
        arbitrary strides, so the views cost nothing here and the single batched copy absorbs them.
        """
        if not any(isinstance(k, str) for k in ns2tensors):
            return ns2tensors

        # Collected up front, because dict order is the caller's: a node supplied both directly and
        # through a group must be caught whichever key comes first.
        bare = {k for k in ns2tensors if not isinstance(k, str)}
        claimed = {}

        out = {}
        for key, tensors in ns2tensors.items():
            if not isinstance(key, str):
                out[key] = tensors
                continue

            assert key in self.external_params_groups, \
                f"`{EXTERNAL_PARAMS_KWARG}` names an unregistered external-parameter group: '{key}'. " \
                f"Register it with `pc.register_external_params_group('{key}', [...])`."
            nodes, dim = self.external_params_groups[key]

            if torch.is_tensor(tensors):
                tensors = (tensors,)
            n_slots = len(nodes[0].external_params.tensor_shapes(nodes[0], batch_size))
            assert len(tensors) == n_slots, \
                f"External-parameter group '{key}' expects {n_slots} concatenated tensor(s), got " \
                f"{len(tensors)}."

            splits = []
            for slot, cat in enumerate(tensors):
                assert torch.is_tensor(cat), \
                    f"External-parameter group '{key}', slot {slot}: expected a tensor, got {type(cat)}."
                sizes = [ns.external_params.tensor_shapes(ns, batch_size)[slot] for ns in nodes]
                axis = dim % len(sizes[0])
                want = list(sizes[0])
                want[axis] = sum(int(s[axis]) for s in sizes)
                assert tuple(cat.size()) == tuple(want), \
                    f"External-parameter group '{key}', slot {slot}: expected the concatenation of " \
                    f"{len(nodes)} tensors along axis {axis}, i.e. shape {tuple(want)}, got " \
                    f"{tuple(cat.size())}."
                splits.append(torch.split(cat, [int(s[axis]) for s in sizes], dim = axis))

            for i, ns in enumerate(nodes):
                assert ns not in bare, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node both directly and through group '{key}'."
                assert ns not in claimed, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node through two groups, "\
                    f"'{claimed[ns]}' and '{key}'."
                claimed[ns] = key
                out[ns] = tuple(sp[i] for sp in splits)

        return out

    def _stage_external_params(self, kwargs: dict, batch_size: int,
                               name: str = "external_params") -> None:
        """
        Copy the caller's external tensors into the staging buffer and replace the `sum_external_params`
        entry with views into it, so the layers only ever see the buffer.

        This is what decouples the caller's memory layout from the kernels'. The caller may hand over
        slices of whatever their own head produced -- strided, freshly allocated each step, in any
        arrangement -- and the layers still receive contiguous tensors at a stable address. The copy is
        issued as ONE batched op rather than one per tensor: a per-tensor `copy_` would cost a kernel
        launch per node, which for a model with many tied copies would dominate the correction itself.

        Called from the forward pass. The backward reuses what was staged here rather than asking for
        the tensors again, which also makes it impossible to run a backward whose external parameters
        disagree with the forward that produced `node_mars`.

        :note: what is staged always describes the MOST RECENT forward, including when that forward
               supplied nothing -- see below.

        :param name: which staging buffer to use. A caller that is NOT a forward pass -- an
                     unconditional `queries.sample()` -- must pass its own name and gets a private
                     slab, leaving `_staged_external_params` and the forward's views untouched.
                     Sharing the default buffer is not merely untidy: `_external_params_views`
                     reallocates it whenever the batch size differs, so a draw between a forward and
                     its backward would invalidate the forward's views IN PLACE, and clearing the
                     field would make the backward run ungated against gated `node_mars`.
        """
        owns_forward_state = (name == "external_params")
        ns2tensors = kwargs.get(EXTERNAL_PARAMS_KWARG, None)

        if ns2tensors is None:
            # This forward runs the gated layers as plain sum layers, so whatever was staged earlier
            # no longer describes the state `node_mars` was computed in. DROPPING it is what keeps the
            # staged tensors a description of the latest forward rather than of some earlier one.
            #
            # Leaving them in place is a quiet trap, because the two consumers that read this field --
            # `pc.backward()` and a conditional `queries.sample()` -- take it unconditionally. An
            # ungated forward between a gated one and its consumer then applied a stale gate: at a
            # matching batch size silently, and at a differing one as a confusing shape error from
            # deep inside the sampler. `lls.backward()` was never exposed to it, since the hook
            # carries the kwargs of the forward that registered it.
            #
            # The cost is that a FORGOTTEN gate is now silent rather than loud: the backward or draw
            # simply runs ungated, consistent with the forward it follows. That is the honest reading
            # of "no external parameters were supplied", and alternating gated and ungated forwards is
            # legitimate (an ungated baseline likelihood, a marginal query), so it is not warned about.
            if owns_forward_state:
                self._staged_external_params = None
            return None

        if isinstance(ns2tensors, StagedExternalParams):
            # Already staged: the caller handed back a dict this method produced. Not hypothetical --
            # `forward` mutates `kwargs` in place and then registers the backward hook with `**kwargs`,
            # so a staged dict really does come back round, and clearing here would break it.
            return None

        views = StagedExternalParams(self._external_params_views(
            name = name, batch_size = batch_size
        ))

        # A group whose members land on a contiguous run of the buffer stages as ONE transpose; the
        # rest resolve to per-node views and go through the loop below, which never learns that groups
        # exist. Tried before the expansion because it is the concatenated tensor that is contiguous,
        # not the slices of it.
        # Conflicts are checked HERE, on the caller's dict, before either staging path touches it: a
        # group taken wholesale is dropped before the expansion below, so the expansion's own check
        # would never see it.
        _bare = {k for k in ns2tensors if not isinstance(k, str)}
        _claimed = {}
        for key in ns2tensors:
            if not isinstance(key, str):
                continue
            assert key in self.external_params_groups, \
                f"`{EXTERNAL_PARAMS_KWARG}` names an unregistered external-parameter group: '{key}'. " \
                f"Register it with `pc.register_external_params_group('{key}', [...])`."
            for ns in self.external_params_groups[key][0]:
                assert ns not in _bare, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node both directly and through group '{key}'."
                assert ns not in _claimed, \
                    f"`{EXTERNAL_PARAMS_KWARG}` supplies a node through two groups, " \
                    f"'{_claimed[ns]}' and '{key}'."
                _claimed[ns] = key

        group_fast, group_done, fast_keys = [], set(), set()
        for key, tensors in ns2tensors.items():
            if not isinstance(key, str):
                continue
            assert key in self.external_params_groups, \
                f"`{EXTERNAL_PARAMS_KWARG}` names an unregistered external-parameter group: '{key}'. " \
                f"Register it with `pc.register_external_params_group('{key}', [...])`."
            whole = self._group_fast_stage(key, tensors, views, batch_size,
                                           buffer = self.__dict__[name])
            if whole is not None:
                group_fast.append(whole)
                group_done.update(self.external_params_groups[key][0])
                fast_keys.add(key)

        # A group staged wholesale is dropped BEFORE the expansion: splitting it into T views and
        # validating each, only for the loop below to skip them, was most of what was left of the cost.
        if fast_keys:
            ns2tensors = {k: v for k, v in ns2tensors.items() if k not in fast_keys}

        ns2tensors = self._expand_external_params_groups(ns2tensors, batch_size)

        # Validate against the buffer's own device rather than `self.device`, which may be index-less
        # (`torch.device("cuda")`) and so compare unequal to an otherwise identical `cuda:0`
        device = self.__dict__[name].device

        dsts, srcs, fast = [], [], list(group_fast)
        for ns, tensors in ns2tensors.items():
            tensors = validate_external_tensors(
                ns, ns.external_params, tensors, batch_size, device, require_contiguous = False
            )

            # The staging buffer may hold the tensors in a different LAYOUT than the caller uses --
            # a different axis order (batch-innermost, so the kernels read them like `element_mars`),
            # or a different shape entirely, where the caller's layout is the one that reads naturally
            # for the model and storage keeps only the entries the kernels index. The copy absorbs
            # whichever it is, so the caller never has to know about it.
            #
            # When that layout change is exactly "rotate the batch axis to the end", the copy is a
            # transpose and gets the tiled kernel. Recognized from the DECLARED permutation rather than from
            # the tensors, and kept per node (`_ext_ns_static`).
            rotates_batch, _ = self._ext_ns_static(ns)

            if rotates_batch:
                for dst, tensor in zip(views[ns], tensors):
                    if tensor.is_contiguous() and tensor.dtype == torch.float32 and tensor.dim() >= 2:
                        fast.append((dst, tensor))
                    else:
                        dsts.append(dst)
                        srcs.append(tensor.permute(*range(1, tensor.dim()), 0))
                continue

            tensors = ns.external_params.to_storage(ns, tensors)

            dsts.extend(views[ns])
            srcs.extend(tensors)

        _staged_copy(dsts, srcs, fast)

        # Nodes the caller did NOT supply keep whatever the buffer held; drop them so a layer sees
        # exactly the set that was staged.
        #
        # Membership is by STORAGE GROUP, not by node: when several nodes share one set of tensors (see
        # `ExternalSumParams.storage_owner`) the caller supplies them once, and every node in that group
        # is staged. Comparing node-by-node would keep only the one that was named and silently turn the
        # rest into plain sum layers -- correct-looking output with the correction applied to one
        # timestep out of many.
        supplied_groups = {self._ext_ns_static(ns)[1] for ns in (set(ns2tensors) | group_done)}
        for ns in list(views.keys()):
            if self._ext_ns_static(ns)[1] not in supplied_groups:
                del views[ns]

        kwargs[EXTERNAL_PARAMS_KWARG] = views
        kwargs[EXTERNAL_PARAMS_BUFFER_KWARG] = self.__dict__[name]
        if owns_forward_state:
            self._staged_external_params = views

    def _ext_ns_static(self, ns) -> tuple:
        """`(rotates_batch, storage owner)` of an external-parameter node, both fixed by its parameterization: whether
        its storage layout is the default `to_storage` moving the batch to the end (the staging copy's fast form) and
        whose slots it uses. Kept per node (two method calls and a tuple comparison a node, every forward)."""
        memo = self.__dict__.setdefault("_ext_ns_static_memo", {})
        hit = memo.get(ns)
        if hit is None:
            ep = ns.external_params
            perm = ep.storage_perm()
            rotates = (perm is not None and type(ep).to_storage is ExternalSumParams.to_storage
                       and tuple(perm) == tuple(range(1, len(perm))) + (0,))
            hit = memo[ns] = (rotates, ep.storage_owner(ns))
        return hit

    def _resolve_backward_external_params(self, kwargs: dict) -> None:
        """
        Point the backward pass at the external parameters the forward actually staged.

        The forward leaves `node_mars` in a form that only the matching external backward interprets
        correctly, so running a backward against *different* external parameters -- or none -- would
        silently produce wrong flows. Taking them from the staging buffer makes that impossible: the
        values used are, by construction, the ones the forward used, and they are pyjuice's own
        snapshot, so nothing the caller does to their tensors in between can perturb them.

        A caller may still pass `sum_external_params` (the autograd hook forwards it), but it only has
        to name the same set of nodes; the staged values are what gets used.
        """
        staged = self._staged_external_params

        ns2tensors = kwargs.get(EXTERNAL_PARAMS_KWARG, None)
        if ns2tensors is not None and not isinstance(ns2tensors, StagedExternalParams):
            assert staged is not None, \
                f"`{EXTERNAL_PARAMS_KWARG}` was given to the backward pass, but the forward pass did " \
                f"not receive any external parameters."
            named = set()
            for k in ns2tensors:
                named.update(self.external_params_groups[k][0] if isinstance(k, str) else [k])
            assert named == set(staged.keys()), \
                f"`{EXTERNAL_PARAMS_KWARG}` names a different set of nodes than the forward pass did."

        if staged is not None:
            kwargs[EXTERNAL_PARAMS_KWARG] = staged
            kwargs[EXTERNAL_PARAMS_BUFFER_KWARG] = self.external_params

    def _check_external_params_kwargs(self, kwargs: dict) -> None:
        """
        Check that every `sum_external_params` / `sum_external_params_grad` entry names a node this PC
        actually compiled with an external parameterization.

        Without this, a stale or mistyped `ns` key is silently ignored -- the PC runs on its shared
        parameters and quietly returns a different answer than intended.
        """
        for kwarg_name in (EXTERNAL_PARAMS_KWARG,):
            ns2tensors = kwargs.get(kwarg_name, None)
            if ns2tensors is None:
                continue

            assert isinstance(ns2tensors, dict), \
                f"`{kwarg_name}` should be a dict mapping nodes to tensors, got {type(ns2tensors)}."

            for ns in ns2tensors:
                if isinstance(ns, str):
                    assert ns in self.external_params_groups, \
                        f"`{kwarg_name}` names an unregistered external-parameter group: '{ns}'. " \
                        f"Register it with `pc.register_external_params_group('{ns}', [...])`."
                    continue
                assert ns in self.external_params_nodes, \
                    f"`{kwarg_name}` contains a node that this PC did not compile with an external " \
                    f"parameterization: {ns}. Construct it with `pyjuice.summate(..., external_params = ...)`."

    def _init_external_params_grads(self, kwargs: dict, batch_size: int, compute_external_grads: bool) -> None:
        """
        Allocate and zero the gradient buffer for the staged external parameters, and hand the layers
        views into it.

        It mirrors the value buffer slot for slot, so the gradient of a node's tensor lives at the
        same offset as the tensor itself and :func:`get_external_params_grad` can hand back a view
        rather than copying anything out. Zeroing happens once, here, because the layers ACCUMULATE --
        which is what lets a node appear in several layers (a tied transition) and sum its
        contributions.
        """
        # The caller may supply DESTINATIONS -- `{ns: buffer}` or `{group: buffer}`, exactly the
        # forward's shape -- to have the gradients written into their own tensors. The kernels still
        # write the PC's internal buffer (they address it by compiled offset), so this records the
        # destinations and the copy-out happens once the backward is done; `get_external_params_grad`
        # keeps working either way.
        dsts = kwargs.pop(EXTERNAL_PARAMS_GRAD_KWARG, None)
        self._external_grad_dsts = None
        if dsts is not None:
            assert isinstance(dsts, dict), \
                f"`{EXTERNAL_PARAMS_GRAD_KWARG}` should be a dict mapping nodes (or group names) to " \
                f"the buffers to fill, got {type(dsts)}."
            assert self._staged_external_params is not None, \
                f"`{EXTERNAL_PARAMS_GRAD_KWARG}` was given, but the forward pass received no external " \
                f"parameters."
            dsts = self._expand_external_params_groups(dsts, batch_size)
            for ns, tensors in dsts.items():
                assert ns in self._staged_external_params, \
                    f"`{EXTERNAL_PARAMS_GRAD_KWARG}` names {ns}, which was not given external " \
                    f"parameters in the forward pass."
                validate_external_tensors(ns, ns.external_params, tensors, batch_size,
                                          self.external_params.device, require_contiguous = False)
            self._external_grad_dsts = dsts
            compute_external_grads = True          # asking for them is asking for them

        if self._staged_external_params is None or not compute_external_grads:
            self._staged_external_params_grad = None
            return None

        grad_views = StagedExternalParams(self._external_params_views(
            name = "external_params_grad", batch_size = batch_size, set_value = 0.0
        ))

        # Only the nodes that actually got external parameters this pass
        for ns in list(grad_views.keys()):
            if ns not in self._staged_external_params:
                del grad_views[ns]

        kwargs[EXTERNAL_PARAMS_GRAD_KWARG] = grad_views
        # The flat buffer too: a kernel that writes gradients addresses it with the same compiled
        # offsets as the forward uses for the tensors themselves (the two buffers share a layout).
        kwargs[EXTERNAL_PARAMS_GRAD_BUFFER_KWARG] = self.external_params_grad
        self._staged_external_params_grad = grad_views

    def get_external_params_grad(self, ns: CircuitNodes):
        """
        Per-sample gradients of the external parameters of `ns`, as computed by the last backward pass.

        Returned as views into the PC's gradient buffer, laid out exactly like the tensors that were
        supplied for `ns` -- so there is nothing to allocate and nothing to copy out. They are valid
        until the next backward pass overwrites them; clone if you need to keep them.

        :param ns: a node that was given external parameters in the last forward pass
        :type ns: CircuitNodes

        A group name registered with :func:`register_external_params_group` is also accepted, and
        returns the members' gradients concatenated along the group's axis -- laid out exactly like the
        tensor that was supplied. Unlike the forward, this one COPIES: the per-node gradients are views
        into a buffer in which a group's members are not generally adjacent.

        :returns: a tuple of gradient tensors, matching the tensors supplied for `ns`
        """
        if isinstance(ns, str):
            assert ns in self.external_params_groups, f"No external-parameter group named '{ns}'."
            nodes, dim = self.external_params_groups[ns]
            per_node = [self.get_external_params_grad(m) for m in nodes]
            n_slots = len(per_node[0])
            axis = dim % per_node[0][0].dim()
            return tuple(torch.cat([g[slot] for g in per_node], dim = axis) for slot in range(n_slots))

        assert self._staged_external_params_grad is not None, \
            "No external-parameter gradients are available. Run a forward pass with " \
            f"`{EXTERNAL_PARAMS_KWARG}` and then a backward pass with `compute_external_grads = True`."
        assert ns in self._staged_external_params_grad, \
            f"No external-parameter gradients for {ns}; it was not given external parameters in the " \
            f"last forward pass."

        # Refused HERE rather than in the backward, because the buffer is allocated and zeroed by
        # default (`compute_external_grads = True`): a parameterization that computes the element and
        # parameter flows but not its own gradient would otherwise either break every backward or hand
        # back a plausible-looking tensor of zeros.
        if not ns.external_params.computes_external_grads:
            raise NotImplementedError(
                f"`{ns.external_params.get_signature()}` does not compute gradients with respect to "
                f"its own parameters yet. The element and parameter flows it contributes ARE computed, "
                f"so `pc.backward()` and the EM optimizers work; only this read is unavailable."
            )

        grad_tensors = self._staged_external_params_grad[ns]

        # Stored in the kernels' layout; hand them back in the caller's, so they line up
        # element-for-element with the tensors that were supplied.
        return ns.external_params.from_storage(ns, grad_tensors)
