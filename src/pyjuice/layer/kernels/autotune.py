"""One-shot launch-config autotuning for the sum- / product-layer Triton kernels.

Every Triton launch site in `sum_layer.py` / `prod_layer.py` picks its tile sizes from a budget
heuristic plus a handful of hand-measured constants ("cap the node tile below batch 64", "double
the edge tile in the LL regime", ...). Those constants were measured on ONE GPU and one set of
layer shapes, so they are a guess everywhere else. This module lets a site hand over a SHORT list
of candidate configs instead: the first call benchmarks them, the winner is cached, and every
later call is a dict lookup.

Contract at every call site:

  * ``candidates[0]`` is the heuristic default. It is what gets used whenever tuning is off, is
    impossible (CUDA-graph capture), or every candidate fails to launch -- so with tuning disabled
    the behaviour is exactly what it was before the autotuner existed.
  * All candidates must compute the same values up to floating-point reduction order. Only tile
    sizes that partition the *output* (or the batch) are eligible; a tile size that sets a
    reduction / max-stabilization group changes the result materially and must stay fixed. Each
    call site carries a note saying which of its knobs is which. Even an output-only tile size is
    not BIT-identical -- its shape changes how Triton lays the tile out and reduces it, measured at
    ~1e-7, the same order as the atomic-add nondeterminism these kernels already have -- which is
    why `pick` caches by shape rather than per layer (see there).
  * ``bench(cfg)`` must not corrupt live buffers. That is automatic when the kernel's output is a
    pure overwrite (re-running it recomputes the same values); a read-accumulate-write output must
    be redirected to a scratch buffer (see `scratch_like`).

Cost: the benchmark runs once per key and pays one Triton compile per candidate config (cached on
disk by Triton across processes), i.e. it lands in the first iteration's warmup. The winners are kept
on disk too, per GPU model and version of the code (`_Store`): the next process of a machine starts
from them, so it neither measures them again nor can come to another answer.
"""

import ast
import hashlib
import json
import os
import warnings
import torch
import triton
from triton.runtime.errors import OutOfResources
from typing import Optional


# Master switch, settable in code with `pyjuice.set_autotune(...)` or via PYJUICE_AUTOTUNE=0. When
# off, every site uses its heuristic default and nothing is ever benchmarked -- the behaviour from
# before this module existed. Worth turning off for A/B, for debugging, and for test suites, where
# many short-lived models would each pay a warmup they never amortize.
ENABLED = os.environ.get("PYJUICE_AUTOTUNE", "1") != "0"


# How much faster than the reference a candidate must measure before it is adopted. These kernels
# run in tens of microseconds, where event-timing noise is several percent even at a median of 7
# reps: measured on a large HCLT, the {CUDA, Triton} element-flow comparisons cluster in 0.90-1.05
# and land on either side from run to run, while the comparisons that genuinely favour CUDA sit at
# 1.4-1.6. A 10% margin cleanly separates the two, so a real win is still taken while a tie always
# resolves to the reference -- which matters because the arms of a {CUDA, Triton} comparison are
# numerically equivalent but not bit-identical, so a coin-flip there changes a run's output.
MARGIN = float(os.environ.get("PYJUICE_AUTOTUNE_MARGIN", 1.10))


# A candidate whose first timed run is this many times slower than the fastest candidate's is dropped
# before the full measurement (see `best_of`): it cannot clear `MARGIN`, and its remaining runs were
# where a slow candidate's tuning time went. MEASURED on the CoDD PC: the softevi dense-vs-scattered
# choice took 1.1-2.7 s at every new batch size, nearly all of it running the scattered candidate --
# 10-30x slower than the dense one -- ten times over.
PRUNE = float(os.environ.get("PYJUICE_AUTOTUNE_PRUNE", 2.0))

# The host time of one more kernel launch issued from inside an extension call (seconds): what a candidate with an
# extra launch costs a host-bound step, which a trial's event time does not see when the GPU work outlasts the host
# (`charged`). MEASURED in the gated plans (three launches against two): 7.1 against 5.1 us a call; a host-bound
# HMM-1024 B64 step took split arms in 17 of 31 layers at 2 us and lost 2%, none at 3.
LAUNCH_S = 3e-6


def charged(run, extra: float):
    """`run` behind a host wait of `extra` seconds -- between a trial's first event and the launch, so the trial's event
    time counts the host time the candidate costs beyond its rivals' (a split-K arm's merge launch: `LAUNCH_S`)."""
    if extra <= 0:
        return run
    import time

    def go():
        end = time.perf_counter() + extra
        while time.perf_counter() < end:
            pass
        run()

    return go


def _capturing() -> bool:
    """True while a CUDA graph is being captured. Benchmarking synchronizes (illegal during
    capture) and would bake the warmup launches into the graph, so tuning is skipped there."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


class KernelCandidateWarning(UserWarning):
    """A candidate dropped for something other than this device's limits: a kernel that does not compile, say. The
    query still runs on the others, but that is a bug to fix; the test suite turns this into an error."""


def _median_time(run, warmup: int, reps: int):
    """Median wall time of `run` in ms, or None if it cannot be launched: a tile config that exceeds this GPU's shared
    memory raises `OutOfResources` at COMPILE time, before any write, and one that takes too much of its memory
    `OutOfMemoryError`. Anything else also returns None, with a :class:`KernelCandidateWarning`."""
    ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    try:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            ev0.record(); run(); ev1.record(); torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
    except (OutOfResources, torch.cuda.OutOfMemoryError):
        return None
    except Exception as e:
        warnings.warn(f"A kernel candidate failed and is dropped: {type(e).__name__}: {str(e)[:200]}",
                      KernelCandidateWarning)
        return None
    ts.sort()
    return ts[len(ts) // 2]


def best_of(candidates: list, warmup: int = 3, reps: int = 7, margin: Optional[float] = None):
    """Benchmark each ``(key, run)`` candidate and return the winning key (None if none can run).

    ``candidates[0]`` is the REFERENCE -- the heuristic tile config, or the plain Triton kernel a
    CUDA fast path is competing with -- and it wins unless some other candidate measures at least
    `MARGIN` times faster. That tie-break is what makes the choice reproducible: several of these
    comparisons sit within a percent of each other, and the arms of a {CUDA, Triton} comparison are
    numerically equivalent but NOT bit-identical, so letting noise settle them makes a run's output
    depend on how warm the GPU happened to be. `run` may write into scratch; only timing matters.

    Measured in two stages. Every candidate first runs once untimed -- compilation and any nested
    first-call tuning land there -- and once timed; one that comes out `PRUNE` times slower than the
    fastest cannot clear the margin against it, so it is dropped. The others get the remaining
    `warmup - 2` warm-up runs and `reps` timed ones: the same number of runs as a single stage, so a
    close race is measured exactly as before.

    ``margin`` replaces `MARGIN` for this choice: 1.0 where the candidates compute the same bits -- nothing to keep
    reproducible -- and the faster is to be taken.
    """
    first = [_median_time(run, 1, 1) for _, run in candidates]
    if all(t is None for t in first):
        return None
    cutoff = PRUNE * min(t for t in first if t is not None)
    kept = [t is not None and t <= cutoff for t in first]
    times = [_median_time(run, max(0, warmup - 2), reps) if keep else t
             for (_, run), t, keep in zip(candidates, first, kept)]

    ref_key, ref_t = candidates[0][0], times[0]
    best_key, best_t = None, None
    for (key, _), t, keep in zip(candidates[1:], times[1:], kept[1:]):
        if keep and t is not None and (best_t is None or t < best_t):
            best_key, best_t = key, t

    if ref_t is None:                                    # the reference cannot run on this GPU
        return best_key
    margin = MARGIN if margin is None else margin
    return best_key if (best_t is not None and best_t * margin < ref_t) else ref_key


# Process-wide cache of tuned configs, keyed by SHAPE -- see `pick`.
_CACHE = dict()

# Whether winners are kept on disk (`_Store`), and where; settable with PYJUICE_AUTOTUNE_PERSIST=0 and
# PYJUICE_AUTOTUNE_DIR. Read when first needed.
PERSIST_ENV, DIR_ENV = "PYJUICE_AUTOTUNE_PERSIST", "PYJUICE_AUTOTUNE_DIR"

# The files of other versions of the code kept beside a new one, the newest (every edit makes a version).
KEPT_VERSIONS = 16


def _source_digest() -> str:
    """A digest of the package's sources: what candidates a key offers, and what they run, follows from them."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    digest = hashlib.blake2b(digest_size = 16)
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = sorted(d for d in subfolders if d != "__pycache__")
        for name in sorted(files):
            if name.endswith((".py", ".cu", ".cuh", ".h", ".cpp")):
                path = os.path.join(folder, name)
                digest.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as f:
                    digest.update(f.read())
    return digest.hexdigest()


def _key_text(key) -> Optional[str]:
    """`key` as text that reads the same in any process, or None if it holds something that has no such text: a tuple
    of plain values, dtypes, and kernels and functions by their names."""
    if isinstance(key, tuple):
        parts = [_key_text(k) for k in key]
        return None if None in parts else "(" + ",".join(parts) + ")"
    if key is None or isinstance(key, (bool, int, float, str, torch.dtype)):
        return repr(key)
    fn = getattr(key, "fn", getattr(getattr(key, "jit_fn", None), "fn", key))    # (a Triton kernel's, or a plain function)
    module, name = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    return None if module is None or name is None else f"{module}.{name}"


def _is_literal(value) -> bool:
    """Whether `value` reads back from its `repr` as itself."""
    if isinstance(value, (tuple, list)):
        return all(_is_literal(v) for v in value)
    return value is None or isinstance(value, (bool, int, float, str))


class _Store:
    """
    The winners of earlier processes for one GPU model and version of the code: a file of one JSON line per winner,
    ``[key text, repr of the choice]``, only ever appended to (one short write, so that processes side by side can share
    it: a line cut short by a crash is passed over, the last line of a key counts). Nothing in it is run: choices are read
    as Python literals. What a choice depends on but its key's shape names the file: the sources (:func:`_source_digest`:
    a candidate of another version may not exist any more), the GPU, torch and Triton. A file that cannot be read or
    written is no error: its winners are measured again.
    """

    def __init__(self, directory: str, device: int):
        name = "|".join((_source_digest(), torch.cuda.get_device_name(device), torch.__version__, triton.__version__))
        self.directory = directory
        self.path = os.path.join(directory, hashlib.blake2b(name.encode(), digest_size = 16).hexdigest() + ".jsonl")
        self.choices = dict()
        try:
            with open(self.path) as f:
                for line in f:
                    try:
                        text, choice = json.loads(line)
                    except ValueError:
                        continue
                    self.choices[text] = choice
        except OSError:
            pass

    def get(self, key):
        """The choice kept for `key` (without the device), or None."""
        text = _key_text(key)
        choice = None if text is None else self.choices.get(text)
        if choice is None:
            return None
        try:
            return ast.literal_eval(choice)
        except (ValueError, SyntaxError):
            return None

    def add(self, key, choice) -> None:
        text = _key_text(key)
        if text is None or not _is_literal(choice):
            return
        self.choices[text] = repr(choice)
        try:
            fresh = not os.path.exists(self.path)
            if fresh:
                os.makedirs(self.directory, exist_ok = True)
            with open(self.path, "a") as f:
                f.write(json.dumps([text, repr(choice)]) + "\n")
            if fresh:
                self._prune()
        except OSError:
            pass

    def _prune(self) -> None:
        """Only the newest files of other versions stay."""
        paths = [os.path.join(self.directory, name) for name in os.listdir(self.directory) if name.endswith(".jsonl")]
        for path in sorted(paths, key = os.path.getmtime)[:-KEPT_VERSIONS]:
            os.remove(path)

    def forget(self) -> None:
        self.choices.clear()
        try:
            os.remove(self.path)
        except OSError:
            pass


# Per device number; the keys of this process already looked up in them; whether a store opened from now on is to forget
# what it has (`set_autotune`).
_STORES = dict()
_LOOKED_UP = set()
_FORGET = False


def _store(device: int) -> Optional[_Store]:
    """The :class:`_Store` of a device, or None if keeping winners is off."""
    if os.environ.get(PERSIST_ENV, "1") == "0":
        return None
    store = _STORES.get(device)
    if store is None:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        store = _STORES[device] = _Store(os.environ.get(DIR_ENV) or os.path.join(cache, "pyjuice", "autotune"), device)
        if _FORGET:
            store.forget()
    return store


def _known(key):
    """The choice for `key` (with its device) of this process or, from now on this process's too, of an earlier one. The
    file is looked at once a key, however often a launch that never tunes asks."""
    choice = _CACHE.get(key)
    if choice is None and key not in _LOOKED_UP:
        _LOOKED_UP.add(key)
        store = _store(key[0])
        if store is not None:
            choice = store.get(key[1])
            if choice is not None:
                _CACHE[key] = choice
    return choice


def _from_store(key, offered: list):
    """:func:`_known` for a key not in this process's choices, if the choice kept is one of those `offered`."""
    choice = _known(key)
    if choice is not None and choice not in offered:
        del _CACHE[key]
        return None
    return choice


def _keep(key, choice) -> None:
    """`choice`, just measured for `key`, for this process and the next."""
    _CACHE[key] = choice
    store = _store(key[0])
    if store is not None:
        store.add(key[1], choice)


def _full_key(key):
    return (torch.cuda.current_device(), key)


def set_autotune(enabled: bool = True, clear_cache: bool = False):
    """Enable or disable launch-config autotuning process-wide; returns the previous setting.

    Choices already measured stay cached (and keep being used) unless `clear_cache` is set, which also forgets what is
    kept on disk for this process's devices: they are measured again. Off, a choice kept on disk is not read either, but
    for the gates of `choose`, which always measure.
    """
    global ENABLED, _FORGET
    was, ENABLED = ENABLED, bool(enabled)
    if clear_cache:
        _CACHE.clear()
        _LOOKED_UP.clear()
        _FORGET = True
        for store in _STORES.values():
            store.forget()
    return was


def cached(key):
    """The config already chosen for `key`, or None if this key still has to go through `pick`.
    Lets a call site skip setting up for a benchmark (allocating a scratch output) on the steady
    state path, where the answer is already known."""
    key = _full_key(key)
    return _known(key) if ENABLED else _CACHE.get(key)


def should_tune(key, num_candidates: int = 2) -> bool:
    """Whether `pick` would actually BENCHMARK this key -- i.e. whether a scratch output is worth
    allocating for it.

    The complement of `cached`, and the distinction matters. `cached` answers "is the choice already
    known"; this also covers the cases where `pick` declines WITHOUT caching: autotuning switched off,
    nothing to choose between, and CUDA-graph capture (deliberately uncached, since capture is
    transient and a later ordinary call should still tune). A caller that only consulted `cached`
    therefore allocated its benchmark buffer on EVERY call in those cases -- and at the param-flow and
    element-flow sites that buffer is the whole flow array.

    MEASURED on an HCLT with 256 latents at batch 512, `PYJUICE_AUTOTUNE=0`: the backward's transient
    peak was 696 MB with 102 allocations per iteration, against 3.06 MB and 3 with tuning on, i.e.
    227x the peak for buffers that were never benchmarked into. Wall time was unchanged (21.4 vs
    21.3 ms -- the caching allocator just reuses the block), so this never showed up in a profile.
    """
    key = _full_key(key)
    if key in _CACHE or (ENABLED and _known(key) is not None):
        return False
    return ENABLED and num_candidates >= 2 and not _capturing()


def pick(key, candidates: list, bench, warmup: int = 3, reps: int = 7, margin: Optional[float] = None):
    """Return the best of `candidates` (config values), benchmarking them at most ONCE per `key`.

    `candidates[0]` is the heuristic default, kept unless `best_of`'s margin is cleared. Never
    raises: a candidate that fails to launch is simply skipped.

    `key` must describe the SHAPE of the launch -- the kernel, the tile/block/edge/batch counts and
    every constexpr flag -- and must NOT identify a particular layer object. Two knock-on reasons:

      * a config picked for one layer is equally good for any other launch of the same kernel at the
        same shape, so keying on shape both cuts the tuning cost and gets a cache hit far more often;
      * more importantly, these candidates are NOT bit-identical. They agree to ~1e-7 (changing the
        tile shape changes how Triton lays out and reduces it), so two structurally identical models
        that tuned independently would disagree in the last ulp -- and with the winner decided by
        measurement, they sometimes would. Keying on shape makes them share one answer instead.

    A later process on the machine shares the answer too (`_Store`). On another machine, or
    after an edit, the choice can still differ, exactly as the existing atomic-add
    nondeterminism in these kernels already does, and at the same ~1e-7 magnitude.
    """
    key = _full_key(key)
    cfg = _CACHE.get(key)
    if cfg is not None:
        return cfg
    if ENABLED and _from_store(key, candidates) is not None:
        return _CACHE[key]

    if not ENABLED or len(candidates) < 2 or _capturing():
        # Not cached on purpose: capture is transient, so a later ordinary call still tunes.
        return candidates[0]

    best = best_of([(c, (lambda c = c: bench(c))) for c in candidates], warmup, reps, margin)
    cfg = candidates[0] if best is None else best
    if best is None:
        _CACHE[key] = cfg                                # (nothing ran: nothing for another process to go by)
    else:
        _keep(key, cfg)
    return cfg


def choose(key, candidates: list):
    """The winner of a fast-path gate's comparison -- `candidates` as `best_of` takes them, ``(choice, run)``
    pairs with the reference (the plain Triton launch) first --, measured at most ONCE per `key` and cached
    process-wide as `pick` caches: `key` describes the launch's SHAPE, so structurally identical layers share one
    answer. The arms of a {CUDA, Triton} gate are numerically equivalent but NOT bit-identical (~1e-6 apart),
    so two identical models whose layers each measured their own could disagree in the last bits.

    Unlike `pick`, it measures whatever `ENABLED` says, as these gates always have. During a CUDA graph capture
    an unmeasured key takes the reference, uncached (measuring synchronizes). The reference too when no
    candidate can run.
    """
    key = _full_key(key)
    choice = _CACHE.get(key)
    if choice is not None:
        return choice
    if _from_store(key, [c for c, _ in candidates]) is not None:
        return _CACHE[key]
    if _capturing():
        return candidates[0][0]
    choice = best_of(candidates)
    if choice is None:
        _CACHE[key] = candidates[0][0]
        return candidates[0][0]
    _keep(key, choice)
    return choice


def scratch_like(tensor: torch.Tensor):
    """A throwaway buffer to benchmark a read-accumulate-write kernel into, or None if it cannot
    be allocated (in which case the caller must skip tuning rather than touch the live output)."""
    try:
        return torch.empty_like(tensor)
    except torch.cuda.OutOfMemoryError:
        return None
