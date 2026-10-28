"""
The activation buffers of :class:`~pyjuice.model.tensorcircuit.TensorCircuit` and the CUDA graphs recorded against
them (:class:`_CudaGraphsMixin`, one of its bases): a pass's buffers cut from one storage per name, reallocated only
when it is far too large (:data:`_BUFFER_SHRINK_RATIO`), and the inner layers' passes recorded into graphs under a
signature of the buffers and options they bake in (:func:`_graph_signature`), replayed by later identical calls.
"""

from __future__ import annotations

import math
import torch
import warnings
from collections import OrderedDict
from typing import Optional, Callable, Tuple

from pyjuice.layer import Layer, InputLayer
from pyjuice.utils.util import cuda_graph_key


# A buffer's backing storage is reallocated at the needed size once it is more than this many times larger
# (see `TensorCircuit._init_buffer`)
_BUFFER_SHRINK_RATIO = 4


def _graph_option(value):
    """An option of a captured pass as a hashable signature entry: tensors by the memory a capture bakes
    in (`cuda_graph_key`), mappings and sequences elementwise, anything else as it is."""
    if isinstance(value, torch.Tensor):
        return cuda_graph_key(value)
    if isinstance(value, dict):
        return tuple((k, _graph_option(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_graph_option(v) for v in value)
    return value


def _graph_signature(pass_id, buffers, options, states = ()):
    """
    The key a recorded inner-layer pass is stored and looked up under: the buffers it bakes in, by
    address (`cuda_graph_key`); every option the inner layers are called with -- built from the very dict
    they are called with, so nothing reaches a captured pass without being part of its key; and, for a
    backward, the identity of the state the forward left on the layers (see `Layer.cuda_graph_state`).

    None when an option cannot be part of a key (it is not hashable), and the pass then runs eagerly.
    """
    try:
        signature = (pass_id, tuple(cuda_graph_key(b) for b in buffers),
                     tuple(sorted((name, _graph_option(v)) for name, v in options.items())),
                     tuple(id(s) for s in states))
        hash(signature)
    except TypeError:
        return None
    return signature


class _RecordedPass:
    """A captured inner-layer pass, with what has to outlive or follow its replays."""

    __slots__ = ("graph", "layer_states", "keepalive")

    def __init__(self, graph, layer_states, keepalive):
        self.graph = graph
        # `(layer, state)` as the recording left them, put back after every replay (the Python that set
        # them does not run)
        self.layer_states = layer_states
        # Everything the inner layers owned at capture time, so no pointer the graph baked in can be freed
        # while it lives -- e.g. by a layer evicting a cached plan -- and the forward states its key names
        self.keepalive = keepalive


class _CudaGraphsMixin:
    """Buffers and CUDA graphs, for :class:`~pyjuice.model.tensorcircuit.TensorCircuit` (one of its bases)."""

    def _init_buffer(self, name: str, shape: Tuple, set_value: Optional[float] = None, check_device: bool = True):
        """Make `self.<name>` a contiguous `shape` buffer on the circuit's device, filled with `set_value`
        if one is given; a buffer that had to change shape (or device) comes back zeroed.

        A shape change -- the batch size, typically -- re-cuts the buffer from ONE backing storage per
        name, grown only when too small, instead of reallocating it. A captured CUDA graph bakes in buffer
        addresses (`cuda_graph_key`), so with a fresh allocation per shape a loop whose batch size changes
        every step put its buffers at new addresses, captured a new graph on nearly every step and kept
        all of them: MEASURED on the CoDD PC (batch 10-26 at random, graphs on), 182 graphs after 80
        steps and 21.7 ms per step, against 7.0 eager. Re-cut, a buffer keeps its address and there is
        one graph per batch size. The views of different shapes share memory, which nothing holds across
        calls (outputs are cloned). Growing a storage moves its buffer, so the graphs recorded against it
        are dropped.

        A storage more than `_BUFFER_SHRINK_RATIO` times what the shape needs is reallocated at the needed
        size, unless graphs of the inner layers are recorded (their replays need the addresses to stay).
        Grow-only, one large batch -- an evaluation pass, say -- held its buffers for the rest of the
        process: MEASURED on an HMM with 1024 latents, 3.1 GB still allocated back at batch 16 after one
        batch-4096 step, against 1.0 GB when every shape got its own allocation. Batch sizes that vary
        within the ratio (CoDD's 10-26) keep one storage, as before.
        """
        tensor = self.__dict__.get(name)
        on_device = lambda t: not check_device or self.device.index is None or t.device == self.device

        if not (isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(shape) and on_device(tensor)):
            numel = math.prod(shape)
            storage = self.__dict__.setdefault("_buffer_storage", dict()).get(name)
            oversized = (storage is not None and storage.numel() > _BUFFER_SHRINK_RATIO * numel
                         and not self.__dict__.get("_recorded_cuda_graphs"))
            if storage is None or storage.numel() < numel or not on_device(storage) or oversized:
                if storage is not None:
                    self._drop_cuda_graphs()    # the old storage moves; a first allocation strands nothing
                storage = torch.empty(numel, device = self.device)
                self._buffer_storage[name] = storage
            tensor = storage[:numel].view(shape)
            tensor.zero_()
            self.__dict__[name] = tensor

        if set_value is not None:
            tensor.fill_(set_value)

    def free_activation_buffers(self):
        """
        Release the buffers a forward or backward pass fills -- `node_mars`, `element_mars`,
        `node_mars_tempered`, `node_flows` and `element_flows` -- with their backing storage, and forget the
        CUDA graphs recorded against them. The circuit is then as it was before its first pass: the next
        pass allocates the buffers again, and whatever a query reads from the last pass (e.g. conditional
        sampling) needs a new forward pass first.

        The parameter flows (`param_flows`, `denom_param_flows`) are kept: they accumulate statistics across
        batches for the next parameter update. A buffer that is still referenced elsewhere -- e.g. through the
        autograd graph of a forward pass's output the caller keeps without `.detach()` -- is released when that
        reference goes.
        """
        storage = self.__dict__.get("_buffer_storage", {})
        for name in ("node_mars", "element_mars", "node_mars_tempered", "node_flows", "element_flows"):
            self.__dict__.pop(name, None)
            storage.pop(name, None)
        self._drop_cuda_graphs()

    def _drop_cuda_graphs(self):
        """Forget every recorded CUDA graph (inner layers and top-down): after a buffer they baked in has
        moved, or when what the passes compute has changed in a way no key sees."""
        self._clear_inner_graphs()
        if hasattr(self, "_tdp_cudagraph"):
            self._tdp_cudagraph.clear()

    def _clear_inner_graphs(self):
        """Forget the recorded inner-layer graphs, and how many were recorded per buffer layout."""
        graphs = self.__dict__.get("_recorded_cuda_graphs")
        if graphs is not None:
            graphs.clear()
        self.__dict__["_cuda_graph_records"] = dict()
        self.__dict__["_cuda_graph_seen"] = OrderedDict()
        self.__dict__["_cuda_graph_failed"] = OrderedDict()

    def _graph_key_options(self, sum_opts: dict, **extra) -> dict:
        """
        What a graph of the inner layers is keyed on besides the buffers: the options the sum layers are
        called with (`sum_opts`, plus `extra`), except the call kwargs only the input layers read
        (`Distribution.call_kwargs`, `InputLayer.call_kwargs`) -- per-call evidence, a fresh tensor on
        every call, which would otherwise stop every replay. The inner layers are still handed them, as
        they always were; that no inner layer READS one is what makes leaving them out sound, and is
        checked over the inner layers' source by a test.
        """
        names = self.__dict__.get("_input_call_kwargs")
        if names is None:
            names = set(InputLayer.call_kwargs)
            for layer in self.input_layer_group:
                for ns in layer.nodes:
                    names.update(ns.dist.call_kwargs)
            names = self.__dict__["_input_call_kwargs"] = frozenset(names)
        return dict({k: v for k, v in sum_opts.items() if k not in names}, **extra)

    def _run_inner_pass(self, run: Callable, pass_id: int, buffers: Tuple, options: dict, record: bool,
                        apply: bool, record_on_repeat: bool = False, reads_layer_states: bool = False) -> None:
        """
        Run the inner layers once (`run`): by replaying a CUDA graph recorded under exactly this
        signature (`_graph_signature`) when there is one and `apply`, and otherwise eagerly -- capturing
        a graph as well when `record` (only from the second time this signature is seen if
        `record_on_repeat`, so a pass that never repeats never pays a capture).

        `options` must be what the inner layers are called with, besides `buffers`. `reads_layer_states`
        marks a pass that consumes what an earlier pass left on the layers (the backward reads the
        forward's, see `Layer.cuda_graph_state`); their identity is then part of its key.
        """
        graphs = self._recorded_cuda_graphs
        if not (record or (apply and graphs)) or not self._cuda_graphs_usable(record):
            run()
            return

        states = [layer.cuda_graph_state() for layer in self._cuda_graph_stateful_layers()] \
            if reads_layer_states else []
        signature = _graph_signature(pass_id, buffers, options, states)
        entry = graphs.get(signature) if signature is not None else None

        if entry is not None and apply:
            graphs.move_to_end(signature)
            entry.graph.replay()
            for layer, state in entry.layer_states:
                layer.restore_cuda_graph_state(state)
        elif entry is None and record and self._may_record_cuda_graph(signature, record_on_repeat):
            self._record_inner_pass(signature, run, states)
        else:
            run()

    def _cuda_graphs_usable(self, record: bool) -> bool:
        """
        Whether this pass may replay or record a graph at all; if not, it simply runs eagerly.

        Not on the CPU, where there is nothing to capture. Not while the current stream is ALREADY being
        captured -- the caller's own `torch.cuda.graph`, say -- where the inner layers must run eagerly
        so that they become part of the caller's graph (a nested capture fails). And not when an inner
        layer's external parameterization does not declare itself graph-safe
        (`ExternalSumParams.cuda_graph_safe`): its per-call Python would run twice on a recording call
        and never on a replay.
        """
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return False
        unsafe = self.__dict__.get("_cuda_graph_unsafe")
        if unsafe is None:
            unsafe = self.__dict__["_cuda_graph_unsafe"] = sorted({
                type(layer.external_params).__name__ for layer_group in self.inner_layer_groups
                for layer in layer_group if not layer.cuda_graph_safe})
        if unsafe:
            if record:
                self._warn_cuda_graph_once("unsafe",
                    f"not recording CUDA graphs: the external parameterization(s) {unsafe} do not set "
                    f"`cuda_graph_safe = True`, so the inner layers always run eagerly.")
            return False
        return True

    def _record_inner_pass(self, signature: tuple, run: Callable, states: list) -> None:
        """
        Run the inner layers for this call, then capture them into a graph stored under `signature`.

        The run is the call's own work, done for real, and the capture after it executes nothing -- so
        recording leaves exactly what an eager call would, with nothing to undo. Earlier this ran the pass
        three times to warm up and once more to replay: the warm-ups accumulated into the flows (a
        backward that recorded came out with ~4x its parameter flows) and later had to be undone by a
        round trip of the flows through host memory, 0.6 s per recording on an HCLT with 256 latents. One
        run is enough warm-up: it compiles the kernels, settles the autotuner (which decides on its first
        call and never during capture) and builds the layers' caches, so the capture records the launches
        of the steady state.

        Captures with the bare `capture_begin`/`capture_end` rather than `torch.cuda.graph`, which
        synchronizes the device and empties the allocator's cache first -- releasing memory every other
        model in the process holds cached, on every recording.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        failure = None
        with torch.cuda.stream(side):
            run()

            # "thread_local": only this thread's CUDA calls are checked against the capture. In the
            # default "global" mode another thread's -- a DataLoader pinning host memory, MEASURED --
            # failed both that thread and the capture.
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(capture_error_mode = "thread_local")
            try:
                run()
                graph.capture_end()
            except BaseException as e:
                try:
                    graph.capture_end()
                except Exception:
                    pass
                if not isinstance(e, Exception):
                    raise
                failure = e
        torch.cuda.current_stream().wait_stream(side)

        if failure is not None:
            # Some code on this path cannot be captured (a host sync, typically). The call's own work is
            # done already, so it only has to stay eager -- and not try again on every call.
            failed = self.__dict__.setdefault("_cuda_graph_failed", OrderedDict())
            failed[signature] = None
            while len(failed) > 4 * max(self.max_cuda_graphs, 1):
                failed.popitem(last = False)
            self._warn_cuda_graph_once(("failed",) + signature[:2],
                f"could not capture a {'backward' if signature[0] else 'forward'} pass into a CUDA graph "
                f"({type(failure).__name__}: {str(failure).splitlines()[0][:200]}); it runs eagerly.")
            return

        layer_states = [(layer, layer.cuda_graph_state()) for layer in self._cuda_graph_stateful_layers()]
        graphs = self._recorded_cuda_graphs
        graphs[signature] = _RecordedPass(graph, layer_states, (states, self._inner_layer_refs()))
        while len(graphs) > max(self.max_cuda_graphs, 1):
            graphs.popitem(last = False)

    def _may_record_cuda_graph(self, signature: Optional[tuple], on_repeat: bool = False) -> bool:
        """Whether to record a graph under `signature`: not when it has no key or could not be captured
        before; with `on_repeat`, not the first time it is seen; and not when its buffer layout has already
        had `max_cuda_graphs_per_layout` recordings (then some option changes on every call, or graphs are
        being evicted as fast as they are made). Says why, once."""
        if signature is None:
            self._warn_cuda_graph_once(None,
                "an option of this pass cannot be part of a CUDA-graph key (it is not hashable), so the "
                "pass runs eagerly and no graph is recorded for it.")
            return False
        if signature in self.__dict__.get("_cuda_graph_failed", ()):
            return False
        if on_repeat:
            seen = self.__dict__.setdefault("_cuda_graph_seen", OrderedDict())
            if signature not in seen:
                seen[signature] = None
                while len(seen) > 4 * max(self.max_cuda_graphs, 1):
                    seen.popitem(last = False)
                return False

        layout, options = signature[:2], signature[2]
        recorded = self._cuda_graph_records.setdefault(layout, [])
        if len(recorded) < self.max_cuda_graphs_per_layout:
            recorded.append(options)
            return True

        missing = object()
        varying = sorted({name for opts in recorded for name, value in opts
                          if any(dict(other).get(name, missing) != value for other in recorded)})
        why = (f"the option(s) {varying} took a new value on most calls -- if they are per-call inputs of "
               f"an input distribution, list them in its `call_kwargs`" if varying else
               f"its graphs kept being evicted -- raise `max_cuda_graphs` (now {self.max_cuda_graphs})")
        self._warn_cuda_graph_once(layout,
            f"stopped recording CUDA graphs for one {'backward' if layout[0] else 'forward'} pass shape "
            f"after {len(recorded)} recordings, each of which costs a capture: {why}. The pass runs eagerly.")
        return False

    def _warn_cuda_graph_once(self, key, message: str) -> None:
        warned = self.__dict__.setdefault("_cuda_graph_warned", set())
        if key not in warned:
            warned.add(key)
            warnings.warn(f"pyjuice: {message}", RuntimeWarning)

    def _cuda_graph_stateful_layers(self) -> list:
        """The inner layers that hand host-side state from one pass to a later one (`Layer.cuda_graph_state`)."""
        layers = self.__dict__.get("_cuda_graph_stateful")
        if layers is None:
            layers = [layer for layer_group in self.inner_layer_groups for layer in layer_group
                      if type(layer).cuda_graph_state is not Layer.cuda_graph_state]
            self.__dict__["_cuda_graph_stateful"] = layers
        return layers

    def _inner_layer_refs(self) -> list:
        """
        References to everything the inner layers (and their external-parameter descriptors) own right
        now, expanded through containers.

        Held by a recorded graph so that nothing its capture baked a pointer to can be freed while it may
        still be replayed. A layer may drop a tensor from a cache after the capture -- BlockScale and
        LowRank evict per-batch plans, each with its own `log Z` buffers -- and a replay would then write
        into memory since handed to something else.
        """
        refs = []

        def collect(obj, depth):
            refs.append(obj)
            if depth > 0:
                if isinstance(obj, dict):
                    for value in obj.values():
                        collect(value, depth - 1)
                elif isinstance(obj, (list, tuple)):
                    for value in obj:
                        collect(value, depth - 1)

        for layer_group in self.inner_layer_groups:
            for layer in layer_group:
                collect(layer.__dict__, 4)
                descriptor = layer.__dict__.get("external_params")
                if descriptor is not None:
                    collect(descriptor.__dict__, 4)

        return refs

    def _buffer_matches(self, name: str, cache: Optional[dict], check_device: bool = True):
        if cache is None:
            return False

        assert name in self.__dict__

        tensor = self.__dict__[name]
        
        if name not in cache:
            return False

        if tensor.size() != cache[name].size():
            return False

        if check_device and tensor.device != cache[name].device:
            return False

        return True
