from __future__ import annotations

import torch
import numpy as np
import random
from collections import OrderedDict
from typing import Optional

from pyjuice.layer.external_sum_layer import ExternalParamsSumLayer
from pyjuice.model import TensorCircuit

from .sampling import assign_cids_ind_target, assign_nids_ind_target, push_non_neg_ones_to_front, \
                      count_prod_nch, sample_prod_layer, sample_sum_layer
from .sampling.scope_plan import build_scope_plan
from .sampling.scoped import scoped_prod_layer, scoped_sum_layer


#: How many recorded sampling plans to keep per circuit, keyed by (num_samples, conditional).
#: Each holds a handful of index tensors per layer, so this bounds the memory a plan cache can hold.
_PLAN_CACHE_SIZE = 8


class _PlanTape():
    """
    Records the sampler's per-layer index tensors on one pass and replays them on the next.

    The top-down pass spends most of its time deriving *where* things go rather than drawing them:
    per layer a `torch.where`, a device-to-host copy, a serial slot allocation and a copy back, all
    to compute indices that -- on a structured-decomposable circuit -- are the same on every call.
    That is the point of :attr:`TensorCircuit.is_structured_decomposable`: with one vtree, the shape
    of the frontier after each layer is a function of the scopes alone, never of which node a draw
    happened to select, so the indices repeat exactly and only the node ids inside them differ.

    The tape keeps the driver a single code path: each step asks for its indices and gets either the
    freshly computed ones (recording) or the cached ones (replaying), so the two cannot drift.

    :note: what is NOT cached, because it is genuinely data-dependent: `ind_nids` / `ind_nid_offs`
           from `count_prod_nch`, which say which compiled row a *sampled* element fell in. That
           kernel therefore still runs on every pass; the cache removes the host round-trips around
           it, not the kernel.
    """

    __slots__ = ("steps", "cursor", "replaying", "scratch", "graph", "bindings", "uniforms",
                 "_frontier")

    def __init__(self, replaying: bool = False):
        self.steps = []
        self.cursor = 0
        self.replaying = replaying
        # Set only on the opt-in CUDA-graph path; see `sample(use_cudagraph = True)`
        self.graph = None
        # The device pointers `graph` was captured with; see `_graph_bindings`
        self.bindings = None
        self.uniforms = None
        self._frontier = None
        # Buffers the pass would otherwise reallocate per layer per call. Keyed by name, and by
        # length where the shape varies with the layer.
        self.scratch = {"compact": None}

    def next(self):
        """The next recorded value. Only valid while replaying."""
        value = self.steps[self.cursor]
        self.cursor += 1
        return value

    def record(self, value):
        self.steps.append(value)

    def step(self, compute):
        """The next step's cached value, or `compute()`'s result when recording."""
        if self.replaying:
            return self.next()

        value = compute()
        self.steps.append(value)
        return value

    def frontier(self, num_nscopes, num_escopes, num_samples, device):
        """
        The `node_samples` / `element_samples` buffers, held across calls.

        A captured launch keeps the pointers it was recorded with, so the graph path needs these at
        fixed addresses. Everything else allocates them per call and lets them go.
        """
        if self._frontier is None:
            self._frontier = (
                torch.zeros([num_nscopes, num_samples], dtype = torch.long, device = device),
                torch.zeros([num_escopes, num_samples], dtype = torch.long, device = device),
            )
        return self._frontier

    def buffers(self, key, n, device, count = 1):
        """`count` zeroed `[n]` int64 scratch tensors under `key`, allocated once and reused."""
        entry = self.scratch.get(key)
        if entry is None or entry.size(1) != n:
            entry = torch.zeros([count, n], dtype = torch.long, device = device)
            self.scratch[key] = entry
        else:
            entry.zero_()
        return entry


def _plan_tape(pc: TensorCircuit, num_samples: int, conditional: bool) -> _PlanTape:
    """
    The recorded plan for this shape, or a fresh recorder.

    Only structured-decomposable circuits get one. Elsewhere a sum node can choose between products
    that decompose its scope differently, and then how many nodes land on the frontier depends on the
    draw -- the plan genuinely changes call to call, and replaying it would be wrong rather than
    merely stale. `PD` is the everyday example. Circuits without the property simply take the
    original path, at the original cost.
    """
    if not getattr(pc, "is_structured_decomposable", False):
        return _PlanTape()

    plans = pc.__dict__.setdefault("_sample_plans", OrderedDict())
    key = (int(num_samples), bool(conditional))

    tape = plans.get(key)
    if tape is None:
        # A plan holds a few index tensors per layer, sized by the frontier, so it is proportional to
        # `num_samples`. LRU-bounded rather than unbounded: a caller sweeping batch sizes would
        # otherwise accumulate one plan per size for the lifetime of the circuit.
        while len(plans) >= _PLAN_CACHE_SIZE:
            plans.popitem(last = False)
        tape = plans[key] = _PlanTape()
    else:
        plans.move_to_end(key)          # LRU: a hit has to protect the entry, or it cannot help
        tape.cursor = 0
        tape.replaying = True

    return tape


def _assert_graphable(pc, do_calibration, requires_sd: bool) -> None:
    """
    Refuse a CUDA-graph capture that would be wrong, rather than capturing it.

    :param requires_sd: whether the pass being captured replays a recorded index plan. The pair-list
                        pass does, so capturing it is only correct where that plan repeats -- i.e. on
                        a structured-decomposable circuit. The scoped pass derives its layout from the
                        circuit instead, so its shapes are static for ANY circuit and the requirement
                        does not apply.
    """
    assert torch.cuda.is_available(), "`use_cudagraph` needs a CUDA device."
    assert not do_calibration, "`use_cudagraph` is not supported together with `_do_calibration`."
    if requires_sd:
        assert getattr(pc, "is_structured_decomposable", False), (
            "`use_cudagraph = True` with `_use_scope_plan = False` needs a structured-decomposable "
            "circuit: that pass replays a recorded index plan, and without one vtree the plan changes "
            "from call to call. The default (scoped) pass has no such restriction."
        )


def _drop_caches_if_moved(pc: TensorCircuit) -> None:
    """
    Discard everything the sampler caches on the circuit when the circuit has changed device.

    All three caches hold DEVICE TENSORS -- the scope plan's row tables, the persistent frontier
    buffers, the recorded index plans -- and none of them is consulted for where it lives. After a
    `pc.to(other_device)` they would be handed to kernels launched against the new one, which is a
    hard failure at best and reads another device's memory through peer access at worst.

    Cheap to get right and cheap to check: one comparison per call, and a rebuild only when the
    circuit actually moved.
    """
    device = pc.device
    if pc.__dict__.get("_sample_cache_device") == device:
        return

    pc.__dict__["_sample_cache_device"] = device
    for name in ("_sample_scope_plan", "_sample_scoped_states", "_sample_plans"):
        pc.__dict__.pop(name, None)


def _scope_plan(pc: TensorCircuit):
    """The circuit's frontier layout, derived once and cached on it."""
    plan = pc.__dict__.get("_sample_scope_plan")
    if plan is None:
        plan = pc.__dict__["_sample_scope_plan"] = build_scope_plan(pc).to(pc.device)
    return plan


def _graph_bindings(pc, conditional: bool, staged = None):
    """
    Everything about the circuit that a captured pass BAKES IN, as a comparable signature.

    A CUDA graph records the device addresses its kernels were launched with. The buffers this pass
    reads from the circuit -- `node_mars`, `element_mars`, the external-parameter staging buffer --
    are all `_init_buffer`-backed, so a forward pass at a different batch size REALLOCATES them and a
    graph captured beforehand goes on reading memory the allocator has since given to somebody else.

    That fails silently. The draw is merely wrong, by however much the recycled pages happen to
    differ from what was there at capture, so it can look fine on one run and not the next; MEASURED
    on a conditional draw whose evidence had changed, the replay came back 10.5 sigma from the live
    answer with `node_mars` sitting at a different address. Interleaving two batch sizes on one
    circuit is an ordinary thing for a decode loop to do, which is how this is reached.

    The staged external parameters are in the signature by IDENTITY, not just by address: a pass
    captured with no gates has no gate kernels in it at all, so replaying it for a gated draw would
    drop the gate entirely rather than read it from the wrong place.

    :returns: a hashable signature; when it changes, the graph has to be recaptured.
    """
    # An unconditional draw stages into its OWN buffer and deliberately does not touch
    # `_staged_external_params` (see `sample`), so reading only that field would leave this signature
    # blind to the draw's own gates -- and therefore unable to notice when their buffer moves. The
    # caller passes what IT staged; the field is the fallback, which is what a conditional draw uses.
    if staged is None:
        staged = getattr(pc, "_staged_external_params", None)
    staged = staged or {}
    external = tuple(sorted(
        (id(ns), tuple(t.data_ptr() for t in tensors))
        for ns, tensors in staged.items()
    ))

    return (pc.node_mars.data_ptr() if pc.node_mars is not None else 0,
            pc.element_mars.data_ptr() if pc.element_mars is not None else 0,
            pc.params.data_ptr(), bool(conditional), external)


def _scoped_state(pc, plan, num_samples: int, conditional: bool, persistent: bool,
                  gated: bool = False):
    """
    The scoped pass's buffers, and its CUDA graph when one was asked for.

    Held across calls only when capturing: a captured launch keeps the pointers it was recorded with,
    so those buffers must not move. An ordinary draw allocates and lets go, which keeps `sample()`
    from pinning memory it no longer needs.

    **One buffer serves every batch size, for an UNCONDITIONAL draw.** A graph captured at width `W`
    always computes `W` lanes, so a request for fewer can replay it and read the first `num_samples`
    columns; only a request for MORE has to reallocate and recapture. A decode loop whose batch
    shrinks as sequences finish would otherwise capture a graph per size -- each capture costing
    three live passes -- and then evict them through an 8-entry cache, which is slower than not using
    a graph at all.

    **A conditional draw cannot share.** Its kernels address `pc.node_mars` as
    `node_id * batch_size + b`, and `batch_size` is frozen at capture; replaying a width-`W` graph
    after a forward pass run at some other batch would read that tensor with the wrong stride -- not
    a truncated answer but a wrong one. Those stay keyed by batch size, as before.
    """
    def allocate(width):
        return {"node": torch.empty([plan.num_node_rows, width], dtype = torch.long,
                                    device = pc.device),
                "elem": torch.empty([plan.num_elem_rows, width], dtype = torch.long,
                                    device = pc.device),
                "seed": torch.zeros([1], dtype = torch.int32, device = pc.device),
                "graph": None, "bindings": None}

    if not persistent:
        return allocate(num_samples)

    states = pc.__dict__.setdefault("_sample_scoped_states", OrderedDict())
    key = (int(num_samples), True) if conditional else ((int(num_samples), False) if gated
                                                       else (None, False))

    state = states.get(key)
    if state is None:
        while len(states) >= _PLAN_CACHE_SIZE:
            states.popitem(last = False)
        state = states[key] = allocate(num_samples)
    elif state["node"].size(1) < num_samples:
        # Grow. The old graph held pointers into buffers that are about to go, so it cannot survive.
        state = states[key] = allocate(num_samples)
    else:
        states.move_to_end(key)
    return state


def _scoped_top_down(pc, plan, node_samples, element_samples, seed_ptr, conditional, kwargs,
                     do_calibration = False):
    """
    The top-down pass over a STRUCTURAL frontier layout.

    Every index it needs comes from `plan`, so there is no `torch.where`, no compaction and no slot
    cursor -- the three things that account for ~92% of the ordinary pass's GPU time. What is left is
    the sampling kernels themselves, one launch per layer per partition.
    """
    node_samples.fill_(-1)
    node_samples[plan.root_row, :] = pc.root_ns._output_ind_range[0]

    for layer_id in range(len(pc.inner_layer_groups) - 1, -1, -1):
        layer_group = pc.inner_layer_groups[layer_id]

        if layer_group.is_sum():
            element_samples.fill_(-1)

            for layer in layer_group:
                rows, erows = plan.sum_rows[id(layer)], plan.sum_erows[id(layer)]

                if conditional:
                    pc.inner_layer_groups[layer_id - 1](pc.node_mars, pc.element_mars)

                # A layer whose parameters are modified per sample owns its own kernel, exactly as
                # it owns its forward and its backward. It declines (returning `False`) when no
                # external tensors were supplied, in which case it IS a plain sum layer.
                handled = False
                if isinstance(layer, ExternalParamsSumLayer):
                    handled = layer.sample_layer(
                        pc.node_mars, pc.element_mars, pc.params, node_samples, element_samples,
                        rows, erows, seed_ptr, conditional = conditional, **kwargs
                    )

                if not handled:
                    for partition_id in range(layer.num_fw_partitions):
                        scoped_sum_layer(
                            layer, layer.partitioned_nids[partition_id],
                            layer.partitioned_cids[partition_id],
                            layer.partitioned_pids[partition_id],
                            pc.node_mars, pc.element_mars, pc.params,
                            node_samples, element_samples, rows, erows, seed_ptr, conditional,
                            do_calibration
                        )

                # Cleared here rather than inside the kernel: a row may be served by any partition,
                # and clearing it in one launch would hide it from the next.
                node_samples.index_fill_(0, rows, -1)

        else:
            for layer in layer_group:
                rows = plan.prod_rows[id(layer)]
                for partition_id in range(layer.num_fw_partitions):
                    scoped_prod_layer(
                        layer, layer.partitioned_nids[partition_id],
                        layer.partitioned_cids[partition_id],
                        plan.prod_crows[id(layer)][partition_id],
                        node_samples, element_samples, rows
                    )


def sample(pc: TensorCircuit, num_samples: Optional[int] = None, conditional: bool = False,
           use_cudagraph: bool = False, _use_scope_plan: bool = True,
           _sample_input_ns: bool = True, _do_calibration: bool = False, **kwargs):
    """
    Draw samples from a PC by performing a top-down ancestral sampling pass.

    For unconditional sampling, `num_samples` must be specified. For conditional sampling
    (`conditional = True`), run a forward pass on the evidence first (e.g., via :func:`marginal`); the
    sampler then reuses the cached `pc.node_mars` and draws one sample per example in that batch, so
    `num_samples` is ignored.

    :param pc: the input PC
    :type pc: TensorCircuit

    :param num_samples: number of samples to draw; required for unconditional sampling, ignored when `conditional = True`
    :type num_samples: Optional[int]

    :param conditional: whether to sample conditioned on the evidence cached by a preceding forward pass
    :type conditional: bool

    Per-sample external sum parameters are supplied through `sum_external_params`, in `kwargs`,
    exactly as they are to :func:`TensorCircuit.forward`

    .. code-block:: python

        samples = juice.queries.sample(pc, num_samples = 1024, sum_external_params = {ns: phi})

    with one gate per DRAWN SAMPLE, so their batch axis is `num_samples`. A sum layer that is given
    none is sampled from its shared parameters, which is what an ungated forward pass computes for
    it too. Under `conditional = True` they are instead taken from the forward pass that produced
    `pc.node_mars` -- `element_mars` was built under those gates, so the draw has to use them -- and
    passing them here only has to name the same nodes.

    :param use_cudagraph: capture the top-down pass into a CUDA graph and replay it. OPT-IN, because
                          a graph owns a private memory pool and the frontier buffers it was captured
                          with, held for the circuit's lifetime -- worth it for a sampling loop, not
                          for a one-off draw. Needs `pc.is_structured_decomposable` (the index plan
                          must repeat, or a replay is simply wrong) and is refused otherwise. One
                          graph is kept per `(num_samples, conditional)`, bounded with the plans.
    :type use_cudagraph: bool

    :param _sample_input_ns: whether to finish the draw by emitting a value from each selected input
                             node. Default `True`, which is what makes the return a tensor of
                             VALUES. Setting it `False` stops one step earlier and returns the
                             FRONTIER instead -- see the note below, which is the whole contract.
    :type _sample_input_ns: bool

    :returns: with `_sample_input_ns = True` (the default), samples of size `[num_samples, num_vars]`
              in the input distributions' own dtype.

              With `_sample_input_ns = False`, the frontier: an int64 tensor of shape
              `[rows, num_samples]` holding the NODE ID each sample selected, padded with `-1`.
              `rows` is the frontier's height, which is at least `num_vars` and usually more; each
              column is a dense prefix of live ids followed by nothing but `-1`, so `>= 0` selects
              exactly the live entries, and every column of one draw holds exactly `num_vars` of
              them -- one per variable, never two.
    :rtype: torch.Tensor

    :note: **the frontier's ids are GLOBAL node ids**, and a circuit may have more than one input
           layer -- it has one per distribution TYPE, so a model mixing, say, `Categorical` and
           `Bernoulli` variables has two. Decoding therefore has to attribute each id to its owning
           layer BEFORE subtracting a start index:

           .. code-block:: python

               for node_id in frontier[frontier[:, j] >= 0, j].tolist():
                   for layer in pc.input_layer_group:
                       start, end = layer._output_ind_range
                       if start <= node_id < end:
                           var = int(layer.vids[node_id - start, 0])
                           params_start = int(layer.s_pids[node_id - start])
                           break

           The shortcut of subtracting `pc.input_layer_group[0]._output_ind_range[0]` and indexing
           that layer's `vids` is correct only while the circuit has ONE input layer. It does not
           fail gracefully when it stops being true: ids from the second layer index past the end of
           the first layer's `vids`, which is a device-side assert that poisons the CUDA context.
    """
    if not conditional:
        assert num_samples is not None, "`num_samples` should be specified when doing unconditioned sampling."
    else:
        num_samples = pc.node_mars.size(1) # Reuse the batch size

    root_ns = pc.root_ns
    assert root_ns._output_ind_range[1] - root_ns._output_ind_range[0] == 1, "It is ambiguous to sample from multi-head PCs."

    # Per-sample external sum parameters, if any -- handled by the PC's own staging, so that they are
    # fed exactly as they are to a forward or a backward pass and end up in the same buffer.
    #
    # Which of the two applies is not a matter of taste. An unconditional pass is a forward: nothing
    # has been staged, and the caller's tensors are copied in at `batch_size = num_samples`, one gate
    # per drawn sample. A conditional pass is a backward: it runs against the `node_mars` /
    # `element_mars` a forward pass left behind, which were built under THAT pass's gates, so the
    # draw has to use the same ones -- and taking them from the staging buffer makes using any
    # others impossible rather than merely discouraged.
    #
    # Both are no-ops when no external parameters were supplied, so an ordinary PC pays nothing.
    _drop_caches_if_moved(pc)

    pc._check_external_params_kwargs(kwargs)
    if conditional:
        pc._resolve_backward_external_params(kwargs)
    else:
        # Into a PRIVATE buffer, not the forward's. An unconditional draw is a forward in its own
        # right, but it does not recompute `node_mars`, so it must not be mistaken for the forward
        # that did. Staging into the shared slab made an ordinary `forward -> sample -> backward`
        # loop compute its flows against the wrong gates: the draw either cleared the staging (an
        # ungated draw) or overwrote it (a gated one), and the backward then ran against a
        # `node_mars` built under gates that were no longer there.
        pc._stage_external_params(kwargs, num_samples, name = "sample_external_params")

    if hasattr(pc, "_num_nscopes") and hasattr(pc, "_num_escopes"):
        num_nscopes = pc._num_nscopes
        num_escopes = pc._num_escopes
    else:
        num_nscopes = 0
        num_escopes = 0
        for layer_group in pc.layers(ret_layer_groups = True):
            curr_scopes = 0
            for layer in layer_group:
                curr_scopes += len(layer.scopes)

            if layer_group.is_input() or layer_group.is_sum():
                num_nscopes += curr_scopes
            else:
                assert layer_group.is_prod()
                num_escopes = max(num_escopes, curr_scopes)

        pc._num_nscopes = num_nscopes
        pc._num_escopes = num_escopes

    # The index plan repeats across calls exactly when the circuit respects one vtree (see
    # `_PlanTape`), so it is recorded on the first pass at a given shape and replayed afterwards.
    # `_do_calibration` is excluded only because it is a debugging switch not worth a second cache.
    # The scoped path derives its indices from the circuit, so it neither records nor replays a
    # plan. It must not take one from the cache either: leaving a tape marked "replaying" with no
    # steps in it would make the NEXT ordinary call index off the end of an empty list.
    tape = (_PlanTape() if (_do_calibration or _use_scope_plan)
            else _plan_tape(pc, num_samples, conditional))
    if not _use_scope_plan and tape.scratch["compact"] is None:
        tape.scratch["compact"] = torch.empty([num_nscopes + 1, num_samples], dtype = torch.long,
                                              device = pc.device)

    if use_cudagraph:
        _assert_graphable(pc, _do_calibration, requires_sd = not _use_scope_plan)

    # Under CUDA-graph capture the frontier buffers have to live at fixed addresses across calls,
    # since a captured launch holds the pointers it was recorded with. Elsewhere they are per-call,
    # which keeps an ordinary `sample()` from holding memory after it returns.
    if use_cudagraph:
        node_samples, element_samples = tape.frontier(num_nscopes, num_escopes, num_samples, pc.device)
    else:
        node_samples = torch.zeros([num_nscopes, num_samples], dtype = torch.long, device = pc.device)
        element_samples = torch.zeros([num_escopes, num_samples], dtype = torch.long, device = pc.device)

    # Pointers indicating how many elements are used in each column of `element_samples`
    element_pointers = np.zeros([num_samples], dtype = np.int64)

    # Uniforms for the sum layers' inverse-CDF draws. Only the graph path uses a buffer -- see the
    # note in `sample_sum_layer_kernel` -- so an ordinary call keeps drawing inside the kernel and is
    # bit-for-bit what it was.
    rnd_buf, rnd_cursor = (tape.uniforms, 0) if use_cudagraph else (None, 0)

    def _top_down_pass():
        nonlocal rnd_cursor
        rnd_cursor = 0

        # Initialize pointers to the root node. Inside the pass, so that a captured graph resets the
        # frontier itself rather than depending on the caller having done it.
        node_samples[:,:] = -1
        node_samples[0,:] = root_ns._output_ind_range[0]

        # Iterate (backward) through layers
        for layer_id in range(len(pc.inner_layer_groups)-1, -1, -1):
            layer_group = pc.inner_layer_groups[layer_id]
            if layer_group.is_sum():
                # Initialize `element_samples` and `element_pointers`
                element_samples[:,:] = -1
                element_pointers[:] = 0

                # Iterate over sum layers in the current layer group
                for layer in layer_group:

                    # Gather the indices to be processed, and pre-compute the target indices in
                    # `element_samples` -- the sampled child node indices are put where `ind_target` says
                    def _sum_indices(layer = layer):
                        lsid, leid = layer._layer_nid_range
                        ind_n, ind_b = torch.where((node_samples >= lsid) & (node_samples < leid))

                        ind_target = np.zeros([ind_n.size(0)], dtype = np.int64)
                        assign_cids_ind_target(ind_target, element_pointers,
                                               ind_b.detach().cpu().numpy(), num_samples)
                        # Flat positions of this layer's frontier entries, for the clear below.
                        # Part of the plan because it is derived from indices that are.
                        clear_at = (ind_n * num_samples + ind_b).contiguous()
                        return ind_n, ind_b, torch.from_numpy(ind_target).to(pc.device), clear_at

                    ind_n, ind_b, ind_target, clear_at = tape.step(_sum_indices)

                    # This layer's slice of the uniforms buffer (graph path only). Every partition of
                    # the layer shares it: a lane belongs to exactly one partition, so each uniform is
                    # consumed once.
                    rnd_offset, rnd_cursor = rnd_cursor, rnd_cursor + ind_n.size(0)

                    # In the case of conditional sampling, recompute to get the `element_mars`
                    if conditional:
                        pc.inner_layer_groups[layer_id-1](pc.node_mars, pc.element_mars)

                    # A layer whose parameters are modified per sample owns its own sampler, exactly as it
                    # owns its forward and its backward -- the shared-parameter kernel below would draw
                    # from a different distribution. It declines (returning `False`) when no external
                    # tensors were supplied for it, in which case it IS a plain sum layer.
                    handled = False
                    if isinstance(layer, ExternalParamsSumLayer):
                        handled = layer.sample_layer_pairs(
                            pc.node_mars, pc.element_mars, pc.params, node_samples, element_samples,
                            ind_target, ind_n, ind_b, conditional = conditional,
                            rnd = rnd_buf, rnd_offset = rnd_offset, **kwargs
                        )

                    # Sample child indices
                    if not handled:
                        for partition_id in range(layer.num_fw_partitions):
                            nids = layer.partitioned_nids[partition_id]
                            cids = layer.partitioned_cids[partition_id]
                            pids = layer.partitioned_pids[partition_id]

                            sample_sum_layer(pc, layer, nids, cids, pids, pc.node_mars, pc.element_mars, pc.params,
                                             node_samples, element_samples, ind_target, ind_n, ind_b,
                                             layer.block_size, conditional, do_calibration = _do_calibration,
                                             rnd = rnd_buf, rnd_offset = rnd_offset)

                    # Clear completed nodes. A flat `scatter_` rather than `node_samples[ind_n,
                    # ind_b] = -1`: advanced-index assignment is NOT capturable into a CUDA graph
                    # (it invalidates the capture), while a scatter is, and the two write exactly the
                    # same positions since the indices are unique.
                    node_samples.view(-1).scatter_(0, clear_at, -1)

            else:
                assert layer_group.is_prod()

                # Iterate over product layers in the current layer group
                for layer in layer_group:
                    # Re-align `node_samples` by pushing all values to the front. Unlike the other steps
                    # this one has to RUN either way -- it moves the frontier -- so only the destination
                    # map is cached, which is four of the routine's seven operations.
                    if tape.replaying:
                        node_pointers = None
                        push_non_neg_ones_to_front(node_samples, dst = tape.next(),
                                                   buffer = tape.scratch["compact"])
                    else:
                        node_pointers, compact_dst = push_non_neg_ones_to_front(
                            node_samples, buffer = tape.scratch["compact"])
                        tape.record(compact_dst)

                    # Indices to process
                    def _prod_indices(layer = layer):
                        lsid, leid = layer._layer_nid_range
                        return torch.where((element_samples >= lsid) & (element_samples < leid))

                    ind_n, ind_b = tape.step(_prod_indices)

                    # `ind_nids` / `ind_nid_offs` / `ind_mask` say which compiled row each SAMPLED element
                    # fell in, so they are data-dependent and this kernel runs on every pass. The buffers
                    # are reused rather than reallocated per layer.
                    ind_ch_count, ind_nids, ind_nid_offs, ind_mask = tape.buffers(
                        ("prod", layer_id, id(layer)), ind_n.size(0), pc.device, count = 4)
                    for partition_id in range(layer.num_fw_partitions):
                        nids = layer.partitioned_nids[partition_id]
                        cids = layer.partitioned_cids[partition_id]

                        count_prod_nch(layer, nids, cids, element_samples, ind_ch_count, ind_nids,
                                       ind_nid_offs, ind_mask, ind_n, ind_b, layer.block_size, partition_id)

                    # Pre-compute the target indices in `node_samples`
                    def _prod_targets(node_pointers = node_pointers):
                        ind_target_sid = np.zeros([ind_n.size(0)], dtype = np.int64)
                        ind_target_sid[1:] = ind_ch_count[:-1].cumsum(dim = 0).detach().cpu().numpy()
                        ind_target = np.zeros([ind_ch_count.sum()], dtype = np.int64)
                        assign_nids_ind_target(ind_target, ind_target_sid,
                                               node_pointers.detach().cpu().numpy(),
                                               ind_b.detach().cpu().numpy(), num_samples)
                        return (torch.from_numpy(ind_target_sid).to(pc.device),
                                torch.from_numpy(ind_target).to(pc.device))

                    ind_target_sid, ind_target = tape.step(_prod_targets)

                    # Store child indices
                    for partition_id in range(layer.num_fw_partitions):
                        nids = layer.partitioned_nids[partition_id]
                        cids = layer.partitioned_cids[partition_id]

                        sample_prod_layer(layer, nids, cids, node_samples, element_samples, ind_target, ind_target_sid,
                                          ind_n, ind_b, ind_nids, ind_nid_offs, ind_mask, layer.block_size, partition_id)

    
    if _use_scope_plan:
        plan = _scope_plan(pc)
        # A gated draw cannot use a workspace WIDER than it asked for: the gated kernels take the
        # frontier's width as the sample count and use it both to validate the staged tensors and to
        # stride the gate table, so a width-64 replay of an 8-sample draw is rejected (and, were it
        # not, would read past the gate buffer). Gated draws therefore keep a buffer per batch size;
        # only ungated ones share. See `_scoped_state`.
        gated = bool(getattr(pc, "external_params_nodes", ()) and kwargs.get("sum_external_params"))
        state = _scoped_state(pc, plan, num_samples, conditional, use_cudagraph, gated = gated)
        node_samples, element_samples, seed_ptr = state["node"], state["elem"], state["seed"]

        # A fresh seed per call, written OUTSIDE any capture -- the kernels load it rather than
        # taking it as a scalar argument, precisely so a replay redraws
        seed_ptr.fill_(random.randint(0, 2**31 - 1))

        if not use_cudagraph:
            _scoped_top_down(pc, plan, node_samples, element_samples, seed_ptr, conditional, kwargs,
                             _do_calibration)
        else:
            # A graph is only replayable while the buffers it baked in are still where they were.
            # Recapture rather than refuse: whether they moved is a consequence of what the CALLER
            # did with the circuit in between (a forward at another batch size), which is not
            # something they should have to track to keep a draw correct.
            bindings = _graph_bindings(pc, conditional, kwargs.get("sum_external_params"))
            if state["graph"] is None or state["bindings"] != bindings:
                state["graph"] = None                       # release the old pool before capturing

                # One live pass so every buffer exists and every kernel is compiled -- capture
                # tolerates neither an allocation nor a JIT compile -- then capture on a side stream.
                _scoped_top_down(pc, plan, node_samples, element_samples, seed_ptr, conditional, kwargs,
                             _do_calibration)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    _scoped_top_down(pc, plan, node_samples, element_samples, seed_ptr,
                                     conditional, kwargs, _do_calibration)
                torch.cuda.current_stream().wait_stream(stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _scoped_top_down(pc, plan, node_samples, element_samples, seed_ptr,
                                     conditional, kwargs, _do_calibration)

                # Recorded only after a capture that completed: a half-captured graph must not be
                # taken for a valid one on the next call.
                state["graph"], state["bindings"] = graph, bindings
            state["graph"].replay()
    elif not use_cudagraph:
        _top_down_pass()
    else:
        graph = tape.graph
        bindings = _graph_bindings(pc, conditional, kwargs.get("sum_external_params"))
        if graph is not None and tape.bindings != bindings:
            # The buffers this graph baked in have moved -- see `_graph_bindings`. Recapture rather
            # than replay against memory the allocator has handed to somebody else.
            graph = tape.graph = None
        if graph is None:
            # Two live passes before capturing: the first records the index plan, the second runs the
            # replay path so that every buffer and every Triton kernel it needs already exists --
            # capture tolerates neither an allocation nor a JIT compile.
            _top_down_pass()
            tape.cursor, tape.replaying = 0, True
            rnd_buf = tape.uniforms = torch.empty([max(rnd_cursor, 1)], dtype = torch.float32,
                                                  device = pc.device)
            rnd_buf.uniform_()
            _top_down_pass()

            tape.cursor = 0
            graph = tape.graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _top_down_pass()
                tape.cursor = 0
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                _top_down_pass()
            tape.bindings = bindings

        # A captured launch holds the scalar arguments it was recorded with, so the draw cannot come
        # from a seed inside the kernels -- it comes from this buffer, refilled outside the graph.
        tape.uniforms.uniform_()
        graph.replay()
        tape.cursor = 0

    # The frontier buffer may be WIDER than this call asked for -- a captured unconditional pass is
    # shared across batch sizes (see `_scoped_state`), so only its first `num_samples` columns belong
    # to this draw. Everything downstream is sized by `num_samples`, so take that view here.
    oversized = node_samples.size(1) != num_samples
    frontier = node_samples[:, :num_samples] if oversized else node_samples

    # Create tensor for the samples
    data_dtype = pc.input_layer_group[0].get_data_dtype()
    samples = torch.zeros([pc.num_vars, num_samples], dtype = data_dtype, device = pc.device)

    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, num_samples), set_value = 0.0)
    ind_n, ind_b = torch.where(frontier != -1)
    ind_node = frontier[ind_n, ind_b]
    pc.node_flows[ind_node, ind_b] = 1.0

    if _sample_input_ns:
        for layer in pc.input_layer_group:
            seed = random.randint(0, 2**31)
            layer.sample(samples, pc.node_flows, seed = seed, **kwargs)

        return samples.permute(1, 0).contiguous()
    else:
        # In this case, we do not explicitly sample input nodes. The compaction runs per column, so
        # it is applied to the whole buffer and the caller's share sliced out afterwards -- as a COPY
        # when the buffer is shared, since a view of it would alias the next draw's workspace.
        push_non_neg_ones_to_front(node_samples)

        # A COPY whenever the buffer outlives the call. `use_cudagraph` pins the frontier for the
        # circuit's lifetime, so returning a view of it hands the caller a tensor the next draw
        # overwrites in place -- two "different" frontiers that share storage. `.contiguous()` does
        # NOT do this: on an already-contiguous slice it returns the same object, which is why the
        # first attempt at this fix changed nothing. An ordinary draw allocates fresh each call and
        # can hand its buffer over directly.
        frontier = node_samples[:, :num_samples]
        return frontier.clone() if use_cudagraph else frontier
