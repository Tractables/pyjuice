"""
External parameters in :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (:class:`_ExternalParamsMixin`, one of its
bases): the caller's per-sample tensors -- per node, or one per named group of nodes -- checked and staged into the
one buffer the layers read (a tiled transpose where the layouts differ, :func:`_staged_copy`), and their gradients
written back in the layout they came in.
"""

from __future__ import annotations

import itertools
import math
import torch
from typing import Optional, Sequence

from pyjuice.nodes import CircuitNodes
from pyjuice.layer import StagedExternalParams, EXTERNAL_PARAMS_KWARG, EXTERNAL_PARAMS_GRAD_KWARG, \
                          EXTERNAL_PARAMS_BUFFER_KWARG, EXTERNAL_PARAMS_GRAD_BUFFER_KWARG
from pyjuice.layer.parameterized_sum_layer import validate_external_tensors
from pyjuice.parameterizations.sum_parameterization import SumParameterization


def _inner_contiguous(t: torch.Tensor) -> bool:
    """Whether every axis of `t` past the first (the batch) is laid out contiguously -- a sample's block one run --,
    whatever the batch stride (checked from the strides: no view made)."""
    expect = 1
    for d in range(t.dim() - 1, 0, -1):
        if t.size(d) != 1 and t.stride(d) != expect:
            return False
        expect *= t.size(d)
    return t.dim() >= 2


def _staging_transpose_fn():
    """
    The tiled-transpose backend for staging: the CUDA extension if it built, Triton otherwise, or None
    to fall back to a generic strided `copy_`.

    Behind ONE function so a test can force each tier. The fallbacks are otherwise unreachable on any
    machine where the extension compiled -- which is how a shape bug lived in one of them: staging a
    whole GROUP hands this a flat destination, the per-node path hands it a storage-shaped one, and the
    fallback only ever handled the second.
    """
    from pyjuice.parameterizations.kernels.c import get_module
    mod = get_module()
    if mod is not None:
        return mod.staging_transpose

    try:
        from pyjuice.parameterizations.kernels.staging import staging_transpose_triton
        return staging_transpose_triton
    except Exception:
        return None


def _staged_transposes_batched(fn, fast: list) -> bool:
    """
    Every `fast` pair's transpose in ONE launch of the CUDA extension's batched sibling, where `fn` is the
    extension's own transpose (a test forcing another tier keeps it) and every pair is contiguous float32 of equal
    sizes on one device; False (nothing launched) otherwise. One call per pair was most of a forward's staging
    host time: MEASURED 83 us at a tied gated HMM-2048 x4 (15 nodes). The pairs' addresses and sizes go to the
    launch by value -- nothing on the device to upload or to keep by address, so a caller handing fresh tensors
    every call (a head's output, new gradient buffers) costs no more than one with stable ones, and what the
    backward launches never depends on where the allocator put them.
    """
    from pyjuice.parameterizations.kernels.c import get_module
    mod = get_module()
    batched = getattr(mod, "staging_transpose_batched", None) if mod is not None else None
    if batched is None or fn is not mod.staging_transpose:
        return False
    device = fast[0][0].device
    rows, max_b, max_n = [], 0, 0
    for dst, src in fast:
        if not (dst.device == device and src.device == device and dst.dtype == torch.float32
                and src.dtype == torch.float32 and dst.is_contiguous() and src.is_contiguous()
                and dst.numel() == src.numel()):
            return False
        batch = src.size(0)
        n = src.numel() // max(batch, 1)
        rows += (src.data_ptr(), dst.data_ptr(), batch | n << 32, n | batch << 32)
        max_b, max_n = max(max_b, batch), max(max_n, n)
    batched(rows, max_b, max_n)
    return True


class _GroupRows:
    """
    What staging a whole group takes where it is not one transpose -- several tensors a node (their slots alternate in
    the buffer), or members that are not adjacent: a transpose a member's slot, all of them in one launch of the
    extension's batched transpose. A member's slot is one run in every sample of the caller's concatenation, and that
    is all the transpose needs of it, so the group is neither split into views nor validated a node at a time, which
    with the generic copies of those strided views was the staging's cost (MEASURED, LowRank, a tied HMM of 1024
    latents and 31 layers, batch 16: 0.36 ms of the forward's 1.3 ms of host time). Sizes and offsets only, for one
    batch size; the addresses are the call's.
    """
    __slots__ = ("shapes", "to_buffer", "to_caller", "max_n")

    def __init__(self, shapes, to_buffer, to_caller, max_n):
        self.shapes = shapes                # per slot: the shape of the concatenated tensor
        self.to_buffer = to_buffer          # per slot: per member `(offset in it, offset in the buffer, sizes, strides)`
        self.to_caller = to_caller          # ... the other way round (sizes and strides swapped)
        self.max_n = max_n                  # the most elements a member's slot holds in a sample

    def launch(self, batched, tensors, buffer, to_buffer: bool) -> None:
        """The transposes between the concatenated `tensors` and `buffer`, whichever way, in one launch."""
        rows, at = [], buffer.data_ptr()
        if to_buffer:
            for cat, slot in zip(tensors, self.to_buffer):
                base = cat.data_ptr()
                for offset, slot_offset, sizes, strides in slot:
                    rows += (base + offset, at + slot_offset, sizes, strides)
            batched(rows, self.shapes[0][0], self.max_n)
        else:
            for cat, slot in zip(tensors, self.to_caller):
                base = cat.data_ptr()
                for offset, slot_offset, sizes, strides in slot:
                    rows += (at + slot_offset, base + offset, sizes, strides)
            batched(rows, self.max_n, self.shapes[0][0])


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
        if len(fast) > 1 and _staged_transposes_batched(fn, fast):
            fast = []
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
        groups = self.__dict__.get("_external_grad_groups")
        if groups:
            for batched, rows, cats in groups:
                if batched == "strided":                       # (`_group_strided`: a copy a slot, the other way round)
                    for src, dst in self._group_strided(rows, cats, cats[0].size(0) if not torch.is_tensor(cats)
                                                        else cats.size(0), self.external_params_grad):
                        dst.copy_(src)
                    continue
                rows.launch(batched, cats, self.external_params_grad, to_buffer = False)
            self._external_grad_groups = None

        dsts = getattr(self, "_external_grad_dsts", None)
        if not dsts:
            return None

        d, s, fast = [], [], []
        # A node's views of the gradient buffer are the same objects every call at a batch size
        # (`_external_params_views` caches them), and `from_storage` is a layout mapping of them: kept per node
        # against those views' identity (~3 us a node, every backward). Where the storage layout rotates the batch
        # axis to the end (`_ext_ns_static`) and the caller's tensor is contiguous fp32, the copy is the staging
        # transpose the other way round -- `[N, B] -> [B, N]`, the same kernel with the sides swapped -- and every
        # such pair goes in ONE launch: `_foreach_copy_` on the transposed views issued one copy kernel per tensor
        # (MEASURED, a tied gated HMM-2048 x4, 15 nodes: 15 launches a backward, ~10 us of host each).
        src_cache = self.__dict__.setdefault("_external_grad_src_cache", {})
        for ns, tensors in dsts.items():
            views = self._staged_external_params_grad[ns]
            hit = src_cache.get(ns)
            if hit is not None and hit[0] is views:
                src, flat = hit[1], hit[2]
            else:
                src = ns.parameterization.from_storage(ns, views)
                rotates, _ = self._ext_ns_static(ns)
                # the storage views as `[N, B]`, the transpose's source (None where the layout is not that rotation)
                flat = tuple(v.view(-1, v.size(-1)) for v in views) if rotates else None
                src_cache[ns] = (views, src, flat)
            if torch.is_tensor(tensors):
                tensors = (tensors,)
            for i, (dst, one) in enumerate(zip(tensors, src)):
                if flat is not None and dst.is_contiguous() and dst.dtype == torch.float32 and dst.dim() >= 2:
                    fast.append((dst, flat[i]))
                else:
                    d.append(dst)
                    s.append(one)

        if fast:
            fn = _staging_transpose_fn()
            if len(fast) > 1 and _staged_transposes_batched(fn, fast):
                fast = []
            for dst, src in fast:
                if fn is not None:
                    fn(dst, src, src.size(0), src.size(1))   # dst[b, n] = src[n, b]: the transpose, sides swapped
                else:
                    d.append(dst)
                    s.append(src.permute(1, 0).view(dst.shape))
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
        # `SumParameterization.storage_owner`): allocate for the owners first, then alias the rest onto
        # them. Aliased nodes still appear in the map, so the layers and kernels need no special case --
        # they look up their own `ns` and simply find the same memory.
        offset = 0
        ns2slots = dict()
        for ns in self.external_params_nodes:
            if ns.parameterization.storage_owner(ns) is not ns:
                continue

            slots = []
            for shape in ns.parameterization.storage_shapes(ns, batch_size):
                slots.append((offset, tuple(shape)))
                offset += math.prod(shape)

            ns2slots[ns] = slots

        for ns in self.external_params_nodes:
            owner = ns.parameterization.storage_owner(ns)
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
                f"external parameterization. Construct it with `pyjuice.summate(..., parameterization = ...)`."
            if id(ns) in seen:
                raise AssertionError(f"External-parameter group '{name}' lists the same node twice.")
            seen.add(id(ns))

        sigs = {ns.parameterization.get_signature() for ns in nodes}
        assert len(sigs) == 1, \
            f"External-parameter group '{name}' mixes parameterizations {sigs}; one concatenated " \
            f"tensor cannot serve nodes whose layouts differ."

        # Members that SHARE storage (a tied node and its copies) are fed once, by the owner. Handing
        # such a pair two different slices is a contradiction, and silently letting one win would apply
        # the wrong gate to the other -- so it is rejected here rather than resolved arbitrarily.
        owners = [ns.parameterization.storage_owner(ns) for ns in nodes]
        if len(set(id(o) for o in owners)) != len(owners):
            raise AssertionError(
                f"External-parameter group '{name}' contains nodes that share one external tensor "
                f"(`tie_external`). They are supplied once, through the storage owner, so they cannot "
                f"also take separate slices of a concatenated tensor."
            )

        # Shapes must agree everywhere except along `dim`. Checked once, at registration, on a nominal
        # batch of 1: the shapes are affine in the batch, so the comparison holds for every batch.
        shapes = [ns.parameterization.tensor_shapes(ns, 1) for ns in nodes]
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
        self._forget_group_plans()

    def unregister_external_params_group(self, name: str) -> None:
        """Remove a group registered by :func:`register_external_params_group`."""
        assert name in self.external_params_groups, \
            f"No external-parameter group named '{name}'."
        del self.external_params_groups[name]
        self._forget_group_plans()

    def _forget_group_plans(self) -> None:
        """Drop what is kept per group name (its rows, its in-place split, its storage owners) and per set of names
        supplied (the in-place staging's plan): a name registered again may name other nodes."""
        for memo in ("_ext_group_rows", "_ext_group_in_place", "_ext_owner_sets", "_ext_in_place_plans"):
            self.__dict__.pop(memo, None)

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

        ep = nodes[0].parameterization
        perm = ep.storage_perm()
        if not (perm is not None
                and type(ep).to_storage is SumParameterization.to_storage
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

    def _group_rows(self, name: str, batch_size: int) -> Optional[_GroupRows]:
        """The group's :class:`_GroupRows` at `batch_size`, or None where its members do not lend themselves: another
        axis than the first after the batch, storage that is not the caller's with the batch moved to the end, slots
        shared with another node. Kept per group and batch size (the layout depends on nothing else)."""
        memo = self.__dict__.setdefault("_ext_group_rows", {})
        key = (name, batch_size)
        if key in memo:
            return memo[key]
        memo[key] = None
        nodes, dim = self.external_params_groups[name]
        if dim != 1 or not all(self._ext_ns_static(ns) == (True, ns) for ns in nodes):
            return None
        _, ns2slots = self._external_params_layout(batch_size)
        shapes_of = [ns.parameterization.tensor_shapes(ns, batch_size) for ns in nodes]
        cat_shapes, to_buffer, to_caller, max_n = [], [], [], 0
        elem = 4                                    # (float32: the buffer's, and what `launch`'s callers check)
        for slot in range(len(shapes_of[0])):
            sizes = [tuple(shapes[slot]) for shapes in shapes_of]
            if not all(len(sz) >= 2 and sz[0] == batch_size and sz[2:] == sizes[0][2:] for sz in sizes):
                return None
            inner = math.prod(sizes[0][2:])
            total = sum(sz[1] for sz in sizes) * inner
            if batch_size == 0 or total >= 1 << 31:
                return None
            rows_in, rows_out, run = [], [], 0
            for ns, sz in zip(nodes, sizes):
                offset, shape = ns2slots[ns][slot]
                n = sz[1] * inner
                if math.prod(shape) != batch_size * n:
                    return None
                rows_in.append((run * elem, offset * elem, batch_size | n << 32, total | batch_size << 32))
                rows_out.append((run * elem, offset * elem, n | batch_size << 32, batch_size | total << 32))
                run += n
                max_n = max(max_n, n)
            cat_shapes.append((batch_size, sum(sz[1] for sz in sizes)) + sizes[0][2:])
            to_buffer.append(rows_in)
            to_caller.append(rows_out)
        memo[key] = _GroupRows(cat_shapes, to_buffer, to_caller, max_n)
        return memo[key]

    def _group_batched(self, name: str, tensors, batch_size: int, buffer: torch.Tensor):
        """`(the batched transpose, the group's rows, its tensors)` where the group's concatenated `tensors` can go
        between the caller and `buffer` that way -- contiguous fp32 of exactly the concatenation's shape, the extension
        built --, else None: the per-node path takes it, and what is wrong with it, if anything, is said there."""
        fn = _staging_transpose_fn()
        from pyjuice.parameterizations.kernels.c import get_module
        mod = get_module()
        batched = getattr(mod, "staging_transpose_batched", None) if mod is not None else None
        if batched is None or fn is not mod.staging_transpose or not buffer.is_cuda:
            return None
        rows = self._group_rows(name, batch_size)
        if rows is None:
            return None
        if torch.is_tensor(tensors):
            tensors = (tensors,)
        if not isinstance(tensors, (tuple, list)) or len(tensors) != len(rows.shapes):
            return None
        device = buffer.device
        for cat, shape in zip(tensors, rows.shapes):
            if not (torch.is_tensor(cat) and cat.dtype == torch.float32 and cat.device == device
                    and cat.is_contiguous() and tuple(cat.size()) == shape):
                return None
        return batched, rows, tensors

    def _group_strided(self, name: str, tensors, batch_size: int, buffer: torch.Tensor):
        """`[(the buffer's view, the caller's view)]`, a pair a slot, where the group's concatenated `tensors` go between
        the caller and `buffer` as ONE strided copy a slot: its members stored as the caller gives them (no
        `storage_perm`, the default `to_storage`), each its own slots, of one shape, their slots evenly spaced in the
        buffer in the group's order (ascending) -- then a slot's members are `[T, B, n]` of the buffer (a stride a
        member) and the concatenation's `[B, T, n]` permuted. Else None: the per-node path takes it."""
        nodes, dim = self.external_params_groups[name]
        if dim != 1 or len(nodes) < 2:
            return None
        for ns in nodes:
            ep = ns.parameterization
            if not (ep.storage_perm() is None and type(ep).to_storage is SumParameterization.to_storage
                    and ep.storage_owner(ns) is ns):
                return None
        if torch.is_tensor(tensors):
            tensors = (tensors,)
        _, ns2slots = self._external_params_layout(batch_size)
        shapes = ns2slots[nodes[0]]
        if not isinstance(tensors, (tuple, list)) or len(tensors) != len(shapes):
            return None
        T, pairs = len(nodes), []
        for slot, cat in enumerate(tensors):
            off0, shape = shapes[slot]
            if len(shape) < 2 or shape[0] != batch_size:
                return None
            if not all(tuple(ns2slots[ns][slot][1]) == tuple(shape) for ns in nodes):
                return None
            step = ns2slots[nodes[1]][slot][0] - off0
            if step <= 0 or any(ns2slots[ns][slot][0] != off0 + k * step for k, ns in enumerate(nodes)):
                return None                                    # (members not in the buffer's order, or not evenly)
            want = (batch_size, T * shape[1]) + tuple(shape[2:])
            if not (torch.is_tensor(cat) and cat.dtype == buffer.dtype and cat.device == buffer.device
                    and cat.is_contiguous() and tuple(cat.size()) == want):
                return None
            per = math.prod(shape[1:])
            pairs.append((buffer.as_strided((T, batch_size, per), (step, per, 1), off0),
                          cat.view(batch_size, T, per).permute(1, 0, 2)))
        return pairs

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
            n_slots = len(nodes[0].parameterization.tensor_shapes(nodes[0], batch_size))
            assert len(tensors) == n_slots, \
                f"External-parameter group '{key}' expects {n_slots} concatenated tensor(s), got " \
                f"{len(tensors)}."

            splits = []
            for slot, cat in enumerate(tensors):
                assert torch.is_tensor(cat), \
                    f"External-parameter group '{key}', slot {slot}: expected a tensor, got {type(cat)}."
                sizes = [ns.parameterization.tensor_shapes(ns, batch_size)[slot] for ns in nodes]
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

        # The nodes whose parameterization reads its tensors IN PLACE (`SumParameterization.reads_in_place`; the
        # forward's own staging alone): the layers are handed views of the caller's tensors -- a group's split along
        # its axis -- and nothing is copied; the caller's tensors' versions are noted for the backward.
        in_place, in_place_versions = {}, []
        if owns_forward_state:
            keys = frozenset(ns2tensors)
            ns2tensors, in_place, in_place_versions = self._take_in_place(
                ns2tensors, batch_size, self.__dict__[name].device)
            if in_place and not ns2tensors:
                # (every entry read in place: what the steps below make of it -- the nodes whose storage group was
                # supplied, each handed its group's tensors, in the buffer's order -- from a plan kept per (keys,
                # batch size), no node looked at one by one)
                memo = self.__dict__.setdefault("_ext_in_place_plans", {})
                plan = memo.get((keys, batch_size))
                if plan is None:
                    member = {}
                    for ns in in_place:
                        member[self._ext_ns_static(ns)[1]] = ns
                    pairs = tuple((ns, member[self._ext_ns_static(ns)[1]]) for ns in views
                                  if self._ext_ns_static(ns)[1] in member)
                    plan = memo[(keys, batch_size)] = (pairs, frozenset(member))
                views = StagedExternalParams({ns: in_place[m] for ns, m in plan[0]})
                views.in_place_versions = in_place_versions
                views.in_place_owners = set(plan[1])
                views.in_place_plan = plan[0]
                kwargs[EXTERNAL_PARAMS_KWARG] = views
                kwargs[EXTERNAL_PARAMS_BUFFER_KWARG] = self.__dict__[name]
                self._staged_external_params = views
                return None

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
                continue
            taken = self._group_batched(key, tensors, batch_size, self.__dict__[name])
            if taken is not None:
                batched, rows, cats = taken
                rows.launch(batched, cats, self.__dict__[name], to_buffer = True)
                group_done.update(self.external_params_groups[key][0])
                fast_keys.add(key)
                continue
            pairs = self._group_strided(key, tensors, batch_size, self.__dict__[name])
            if pairs is not None:
                for dst, src in pairs:
                    dst.copy_(src)
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
                ns, ns.parameterization, tensors, batch_size, device, require_contiguous = False
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

            tensors = ns.parameterization.to_storage(ns, tensors)

            dsts.extend(views[ns])
            srcs.extend(tensors)

        _staged_copy(dsts, srcs, fast)

        # Nodes the caller did NOT supply keep whatever the buffer held; drop them so a layer sees
        # exactly the set that was staged.
        #
        # Membership is by STORAGE GROUP, not by node: when several nodes share one set of tensors (see
        # `SumParameterization.storage_owner`) the caller supplies them once, and every node in that group
        # is staged. Comparing node-by-node would keep only the one that was named and silently turn the
        # rest into plain sum layers -- correct-looking output with the correction applied to one
        # timestep out of many.
        supplied_groups = {self._ext_ns_static(ns)[1] for ns in (set(ns2tensors) | group_done | set(in_place))}
        for ns in list(views.keys()):
            if self._ext_ns_static(ns)[1] not in supplied_groups:
                del views[ns]

        if in_place:
            # (every node of an in-place node's storage group reads that node's tensors: one set for the group)
            owned = {self._ext_ns_static(ns)[1]: tensors for ns, tensors in in_place.items()}
            for ns in views:
                hit = owned.get(self._ext_ns_static(ns)[1])
                if hit is not None:
                    views[ns] = hit
            views.in_place_versions = in_place_versions
            views.in_place_owners = set(owned)

        kwargs[EXTERNAL_PARAMS_KWARG] = views
        kwargs[EXTERNAL_PARAMS_BUFFER_KWARG] = self.__dict__[name]
        if owns_forward_state:
            self._staged_external_params = views

    def _group_in_place_views(self, key: str, tensors, batch_size: int, device, aligned16: bool = False) -> Optional[dict]:
        """`{ns: its views}` of a group's concatenated `tensors`, split along the group's axis -- where that is the first
        after the batch and each tensor is fp32 on `device`, of the concatenation's exact shape, every axis past the
        batch contiguous (so is each member's view) --, else None. The members' extents are kept per group and batch
        size: a call checks one tensor a slot and splits it (no node's tensors validated one by one). `aligned16`: and
        every member's view 16-byte aligned with a batch stride of a multiple of 4 floats (the concatenation's base
        and batch stride so, each member's offset a multiple of 4 floats: kept with the extents)."""
        memo = self.__dict__.setdefault("_ext_group_in_place", {})
        plan = memo.get((key, batch_size))
        if plan is None:
            nodes, dim = self.external_params_groups[key]
            shapes = [tuple(tuple(sh) for sh in ns.parameterization.tensor_shapes(ns, batch_size)) for ns in nodes]
            slots = []
            for slot in range(len(shapes[0])):
                sizes = [sh[slot] for sh in shapes]
                if len(sizes[0]) < 3 or dim % len(sizes[0]) != 1 or \
                        any(sz[:1] + sz[2:] != sizes[0][:1] + sizes[0][2:] for sz in sizes):
                    slots = None
                    break
                inner = math.prod(sizes[0][2:])
                offs = itertools.accumulate([0] + [sz[1] * inner for sz in sizes[:-1]])
                slots.append(((sizes[0][0], sum(sz[1] for sz in sizes)) + sizes[0][2:], [sz[1] for sz in sizes],
                              all(o % 4 == 0 for o in offs)))
            plan = memo[(key, batch_size)] = (nodes, slots) if slots else False
        if plan is False:
            return None
        nodes, slots = plan
        cats = (tensors,) if torch.is_tensor(tensors) else tuple(tensors) if isinstance(tensors, (tuple, list)) else ()
        if len(cats) != len(slots):
            return None
        splits = []
        for cat, (want, sizes, starts4) in zip(cats, slots):
            if not (torch.is_tensor(cat) and cat.dtype == torch.float32 and cat.device == device and
                    tuple(cat.shape) == want and cat.size(0) > 0 and _inner_contiguous(cat)):
                return None
            if aligned16 and not (starts4 and cat.data_ptr() % 16 == 0 and cat.stride(0) % 4 == 0):
                return None
            splits.append(torch.split(cat, sizes, dim = 1))
        return dict(zip(nodes, zip(*splits)))

    def _take_in_place(self, ns2tensors: dict, batch_size: int, device) -> tuple:
        """`(the rest of ns2tensors, {ns: its tensors}, [(tensor, version)])`: the entries whose nodes all read their
        tensors in place taken out of `ns2tensors` -- a group's split along its axis into views, one set a node --,
        each node's tensors validated and fp32 with every axis past the batch contiguous (an entry with a tensor that
        is not stays in the rest: it is copied), and the caller's tensors as handed over, with their versions now (a
        view shares its base's version: an in-place write to either shows)."""
        rest, taken, versions = {}, {}, []
        for key, tensors in ns2tensors.items():
            nodes = self.external_params_groups[key][0] if isinstance(key, str) else (key,)
            # (never a graph-safe parameterization's: a replay reads the addresses the capture saw)
            if not all(ns.parameterization.reads_in_place(ns) and not ns.parameterization.cuda_graph_safe
                       for ns in nodes):
                rest[key] = tensors
                continue
            if isinstance(key, str):
                mine = self._group_in_place_views(key, tensors, batch_size, device)
            else:
                ts = validate_external_tensors(key, key.parameterization, tensors, batch_size, device,
                                               require_contiguous = False)
                mine = {key: ts} if all(t.size(0) > 0 and _inner_contiguous(t) for t in ts) else None
            if mine is None:
                rest[key] = tensors
                continue
            taken.update(mine)
            for t in ((tensors,) if torch.is_tensor(tensors) else tensors):
                versions.append((t, t._version))
        return rest, taken, versions

    def _ext_ns_static(self, ns) -> tuple:
        """`(rotates_batch, storage owner)` of an external-parameter node, both fixed by its parameterization: whether
        its storage layout is the default `to_storage` moving the batch to the end (the staging copy's fast form) and
        whose slots it uses. Kept per node (two method calls and a tuple comparison a node, every forward)."""
        memo = self.__dict__.setdefault("_ext_ns_static_memo", {})
        hit = memo.get(ns)
        if hit is None:
            ep = ns.parameterization
            perm = ep.storage_perm()
            rotates = (perm is not None and type(ep).to_storage is SumParameterization.to_storage
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
            # (the tensors the forward read in place are the caller's own: a backward after one of them was written
            # in place would interpret `node_mars` with other parameters than made it)
            for t, version in getattr(staged, "in_place_versions", ()):
                if t._version != version:
                    raise RuntimeError(
                        "An external-parameter tensor that the forward pass read in place was modified in place "
                        "before the backward pass. Run the forward again, or hand the forward a copy.")
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
                    f"parameterization: {ns}. Construct it with `pyjuice.summate(..., parameterization = ...)`."

    def _take_in_place_grads(self, dsts: dict, batch_size: int) -> tuple:
        """`(the rest of dsts, {ns: its destinations}, [the caller's tensors to zero])`: the destinations the layers add
        into themselves, so no copy-out -- an entry every node of which read its tensors in place in the forward
        (`reads_in_place`), its tensors fp32 on the buffer's device with every axis past the batch contiguous and
        16-byte aligned runs (a group's split along its axis) --, each zeroed once before the backward. Nothing is taken
        where a node is named twice (the paths after this say so)."""
        owners = getattr(self._staged_external_params, "in_place_owners", None)
        if not owners:
            return dsts, {}, []
        named = []
        for key in dsts:
            if isinstance(key, str):
                named.extend(self.external_params_groups[key][0] if key in self.external_params_groups else ())
            else:
                named.append(key)
        if len(named) != len(set(named)):
            return dsts, {}, []
        device = self.external_params.device
        # (a destination sharing memory with what the forward read in place would be zeroed under the backward's feet)
        read = {t.untyped_storage().data_ptr() for t, _ in getattr(self._staged_external_params, "in_place_versions", ())}
        rest, taken, zero = {}, {}, []
        for key, tensors in dsts.items():
            if isinstance(key, str) and key not in self.external_params_groups:
                rest[key] = tensors
                continue
            gown = self.__dict__.setdefault("_ext_owner_sets", {})
            mine_owners = gown.get(key)
            if mine_owners is None:
                nodes = self.external_params_groups[key][0] if isinstance(key, str) else (key,)
                mine_owners = gown[key] = frozenset(self._ext_ns_static(ns)[1] for ns in nodes)
            cats = (tensors,) if torch.is_tensor(tensors) else tuple(tensors) if isinstance(tensors, (tuple, list)) else ()
            if not (cats and mine_owners <= owners and
                    all(torch.is_tensor(t) and t.dtype == torch.float32 and t.device == device and
                        t.untyped_storage().data_ptr() not in read for t in cats)):
                rest[key] = tensors
                continue
            if isinstance(key, str):
                mine = self._group_in_place_views(key, tensors, batch_size, device, aligned16 = True)
            else:
                ts = validate_external_tensors(key, key.parameterization, tensors, batch_size, device,
                                               require_contiguous = False)
                mine = {key: ts} if all(t.size(0) > 0 and _inner_contiguous(t) and t.data_ptr() % 16 == 0 and
                                        t.stride(0) % 4 == 0 for t in ts) else None
            if mine is None:
                rest[key] = tensors
                continue
            taken.update(mine)
            zero.extend(cats)
        return rest, taken, zero

    def _take_whole_grad_groups(self, dsts: dict, batch_size: int) -> dict:
        """The destinations that are whole groups going back in one launch (`_group_batched`) noted for the end of the
        backward (`_external_grad_groups`); what is left of `dsts` returned for the per-node path. Nothing is taken where
        a node is named twice, or a group's member got no external parameters: the per-node path says so."""
        if not any(isinstance(k, str) for k in dsts):
            return dsts
        seen = {k for k in dsts if not isinstance(k, str)}
        for key in dsts:
            if isinstance(key, str):
                members = self.external_params_groups[key][0] if key in self.external_params_groups else ()
                if any(ns in seen or ns not in self._staged_external_params for ns in members):
                    return dsts
                seen.update(members)
        taken, rest = [], dict()
        for key, tensors in dsts.items():
            whole = None
            if isinstance(key, str) and key in self.external_params_groups:
                whole = self._group_batched(key, tensors, batch_size, self.external_params)    # (its device: the gradient
                                                                                                # buffer is allocated next)
                if whole is None and self._group_strided(key, tensors, batch_size, self.external_params) is not None:
                    whole = ("strided", key, tensors)                                           # (the views: once the
                                                                                                # buffer exists)
            if whole is None:
                rest[key] = tensors
            else:
                taken.append(whole)
        self._external_grad_groups = taken or None
        return rest

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
        self._external_grad_groups = None
        in_place_grads, zero = {}, []
        if dsts is not None:
            assert isinstance(dsts, dict), \
                f"`{EXTERNAL_PARAMS_GRAD_KWARG}` should be a dict mapping nodes (or group names) to " \
                f"the buffers to fill, got {type(dsts)}."
            assert self._staged_external_params is not None, \
                f"`{EXTERNAL_PARAMS_GRAD_KWARG}` was given, but the forward pass received no external " \
                f"parameters."
            dsts, in_place_grads, zero = self._take_in_place_grads(dsts, batch_size)
            dsts = self._take_whole_grad_groups(dsts, batch_size)
            dsts = self._expand_external_params_groups(dsts, batch_size)
            for ns, tensors in dsts.items():
                assert ns in self._staged_external_params, \
                    f"`{EXTERNAL_PARAMS_GRAD_KWARG}` names {ns}, which was not given external " \
                    f"parameters in the forward pass."
                validate_external_tensors(ns, ns.parameterization, tensors, batch_size,
                                          self.external_params.device, require_contiguous = False)
            self._external_grad_dsts = dsts
            compute_external_grads = True          # asking for them is asking for them

        if self._staged_external_params is None or not compute_external_grads:
            self._staged_external_params_grad = None
            return None

        # (the destinations the layers write themselves: their storage groups need nothing of the buffer -- which is
        # not filled at all where that is every node; where the forward read every node in place and the destinations
        # are its plan's, the views straight from them)
        plan = getattr(self._staged_external_params, "in_place_plan", None)
        if plan is not None and in_place_grads and all(m in in_place_grads for _, m in plan):
            self._external_params_views(name = "external_params_grad", batch_size = batch_size, set_value = None)
            grad_views = StagedExternalParams({ns: in_place_grads[m] for ns, m in plan})
            seen = {}
            for t in zero:
                seen.setdefault((t.data_ptr(), tuple(t.shape), tuple(t.stride())), t)
            torch._foreach_zero_(list(seen.values()))
            kwargs[EXTERNAL_PARAMS_GRAD_KWARG] = grad_views
            kwargs[EXTERNAL_PARAMS_GRAD_BUFFER_KWARG] = self.external_params_grad
            self._staged_external_params_grad = grad_views
            return None
        owned = {self._ext_ns_static(ns)[1]: tensors for ns, tensors in in_place_grads.items()}
        whole = bool(owned) and all(self._ext_ns_static(ns)[1] in owned for ns in self._staged_external_params)
        grad_views = StagedExternalParams(self._external_params_views(
            name = "external_params_grad", batch_size = batch_size, set_value = None if whole else 0.0
        ))

        # Only the nodes that actually got external parameters this pass
        for ns in list(grad_views.keys()):
            if ns not in self._staged_external_params:
                del grad_views[ns]

        if owned:
            for ns in grad_views:
                hit = owned.get(self._ext_ns_static(ns)[1])
                if hit is not None:
                    grad_views[ns] = hit
            seen = {}
            for t in zero:
                seen.setdefault((t.data_ptr(), tuple(t.shape), tuple(t.stride())), t)
            torch._foreach_zero_(list(seen.values()))

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
        if not ns.parameterization.computes_external_grads:
            raise NotImplementedError(
                f"`{ns.parameterization.get_signature()}` does not compute gradients with respect to "
                f"its own parameters yet. The element and parameter flows it contributes ARE computed, "
                f"so `pc.backward()` and the EM optimizers work; only this read is unavailable."
            )

        grad_tensors = self._staged_external_params_grad[ns]

        # Stored in the kernels' layout; hand them back in the caller's, so they line up
        # element-for-element with the tensors that were supplied.
        return ns.parameterization.from_storage(ns, grad_tensors)
