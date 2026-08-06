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

    __slots__ = ("steps", "cursor", "replaying", "scratch")

    def __init__(self, replaying: bool = False):
        self.steps = []
        self.cursor = 0
        self.replaying = replaying
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


def sample(pc: TensorCircuit, num_samples: Optional[int] = None, conditional: bool = False,
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

    :returns: a tensor of samples of size [num_samples, num_vars]
    :rtype: torch.Tensor
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
    pc._check_external_params_kwargs(kwargs)
    if conditional:
        pc._resolve_backward_external_params(kwargs)
    else:
        pc._stage_external_params(kwargs, num_samples)

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

    # Stores selected node indices by the sampler
    node_samples = torch.zeros([num_nscopes, num_samples], dtype = torch.long, device = pc.device)
    # Stores selected element indices by the sampler
    element_samples = torch.zeros([num_escopes, num_samples], dtype = torch.long, device = pc.device)
    # Pointers indicating how many elements are used in each column of `element_samples`
    element_pointers = np.zeros([num_samples], dtype = np.int64)

    # Initialize pointers to the root node
    node_samples[:,:] = -1
    node_samples[0,:] = root_ns._output_ind_range[0]

    # The index plan repeats across calls exactly when the circuit respects one vtree (see
    # `_PlanTape`), so it is recorded on the first pass at a given shape and replayed afterwards.
    # `_do_calibration` is excluded only because it is a debugging switch not worth a second cache.
    tape = _plan_tape(pc, num_samples, conditional) if not _do_calibration else _PlanTape()
    if tape.scratch["compact"] is None:
        tape.scratch["compact"] = torch.empty([num_nscopes + 1, num_samples], dtype = torch.long,
                                              device = pc.device)

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
                    return ind_n, ind_b, torch.from_numpy(ind_target).to(pc.device)

                ind_n, ind_b, ind_target = tape.step(_sum_indices)

                # In the case of conditional sampling, recompute to get the `element_mars`
                if conditional:
                    pc.inner_layer_groups[layer_id-1](pc.node_mars, pc.element_mars)

                # A layer whose parameters are modified per sample owns its own sampler, exactly as it
                # owns its forward and its backward -- the shared-parameter kernel below would draw
                # from a different distribution. It declines (returning `False`) when no external
                # tensors were supplied for it, in which case it IS a plain sum layer.
                handled = False
                if isinstance(layer, ExternalParamsSumLayer):
                    handled = layer.sample_layer(
                        pc.node_mars, pc.element_mars, pc.params, node_samples, element_samples,
                        ind_target, ind_n, ind_b, conditional = conditional, **kwargs
                    )

                # Sample child indices
                if not handled:
                    for partition_id in range(layer.num_fw_partitions):
                        nids = layer.partitioned_nids[partition_id]
                        cids = layer.partitioned_cids[partition_id]
                        pids = layer.partitioned_pids[partition_id]

                        sample_sum_layer(pc, layer, nids, cids, pids, pc.node_mars, pc.element_mars, pc.params,
                                         node_samples, element_samples, ind_target, ind_n, ind_b,
                                         layer.block_size, conditional, do_calibration = _do_calibration)

                # Clear completed nodes
                node_samples[ind_n, ind_b] = -1

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

    # Create tensor for the samples
    data_dtype = pc.input_layer_group[0].get_data_dtype()
    samples = torch.zeros([pc.num_vars, num_samples], dtype = data_dtype, device = pc.device)

    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, num_samples), set_value = 0.0)
    ind_n, ind_b = torch.where(node_samples != -1)
    ind_node = node_samples[ind_n, ind_b]
    pc.node_flows[ind_node, ind_b] = 1.0

    if _sample_input_ns:
        for layer in pc.input_layer_group:
            seed = random.randint(0, 2**31)
            layer.sample(samples, pc.node_flows, seed = seed, **kwargs)

        return samples.permute(1, 0).contiguous()
    else:
        # In this case, we do not explicitly sample input nodes
        push_non_neg_ones_to_front(node_samples)
        return node_samples
