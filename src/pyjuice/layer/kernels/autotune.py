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
disk by Triton across processes), i.e. it lands in the first iteration's warmup.
"""

import os
import torch


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


def _capturing() -> bool:
    """True while a CUDA graph is being captured. Benchmarking synchronizes (illegal during
    capture) and would bake the warmup launches into the graph, so tuning is skipped there."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _median_time(run, warmup: int, reps: int):
    """Median wall time of `run` in ms, or None if it cannot be launched (e.g. a tile config that
    exceeds this GPU's shared memory raises `OutOfResources` at COMPILE time, before any write)."""
    ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    try:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            ev0.record(); run(); ev1.record(); torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
    except Exception:
        return None
    ts.sort()
    return ts[len(ts) // 2]


def best_of(candidates: list, warmup: int = 3, reps: int = 7):
    """Benchmark each ``(key, run)`` candidate and return the winning key (None if none can run).

    ``candidates[0]`` is the REFERENCE -- the heuristic tile config, or the plain Triton kernel a
    CUDA fast path is competing with -- and it wins unless some other candidate measures at least
    `MARGIN` times faster. That tie-break is what makes the choice reproducible: several of these
    comparisons sit within a percent of each other, and the arms of a {CUDA, Triton} comparison are
    numerically equivalent but NOT bit-identical, so letting noise settle them makes a run's output
    depend on how warm the GPU happened to be. `run` may write into scratch; only timing matters.
    """
    ref_key, ref_run = candidates[0]
    ref_t = _median_time(ref_run, warmup, reps)

    best_key, best_t = None, None
    for key, run in candidates[1:]:
        t = _median_time(run, warmup, reps)
        if t is not None and (best_t is None or t < best_t):
            best_key, best_t = key, t

    if ref_t is None:                                    # the reference cannot run on this GPU
        return best_key
    return best_key if (best_t is not None and best_t * MARGIN < ref_t) else ref_key


# Process-wide cache of tuned configs, keyed by SHAPE -- see `pick`.
_CACHE = dict()


def _full_key(key):
    return (torch.cuda.current_device(), key)


def set_autotune(enabled: bool = True, clear_cache: bool = False):
    """Enable or disable launch-config autotuning process-wide; returns the previous setting.

    Choices already measured stay cached (and keep being used) unless `clear_cache` is set.
    """
    global ENABLED
    was, ENABLED = ENABLED, bool(enabled)
    if clear_cache:
        _CACHE.clear()
    return was


def cached(key):
    """The config already chosen for `key`, or None if this key still has to go through `pick`.
    Lets a call site skip setting up for a benchmark (allocating a scratch output) on the steady
    state path, where the answer is already known."""
    return _CACHE.get(_full_key(key))


def pick(key, candidates: list, bench, warmup: int = 3, reps: int = 7):
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

    Across processes the choice can still differ, exactly as the existing atomic-add
    nondeterminism in these kernels already does, and at the same ~1e-7 magnitude.
    """
    key = _full_key(key)
    cfg = _CACHE.get(key)
    if cfg is not None:
        return cfg

    if not ENABLED or len(candidates) < 2 or _capturing():
        # Not cached on purpose: capture is transient, so a later ordinary call still tunes.
        return candidates[0]

    best = best_of([(c, (lambda c = c: bench(c))) for c in candidates], warmup, reps)
    cfg = candidates[0] if best is None else best
    _CACHE[key] = cfg
    return cfg


def scratch_like(tensor: torch.Tensor):
    """A throwaway buffer to benchmark a read-accumulate-write kernel into, or None if it cannot
    be allocated (in which case the caller must skip tuning rather than touch the live output)."""
    try:
        return torch.empty_like(tensor)
    except torch.cuda.OutOfMemoryError:
        return None
