"""
`pyjuice.fast_inference()` -- a scope in which parameters are promised not to change, so layers may
hold derived copies of them that would otherwise be unsafe to cache.

The motivating case is the soft-evidence forward. It gathers `params[s_pids[n] + cat]`, and `params`
is laid out `[rows, cats]`, so consecutive NODES -- the innermost axis of the kernel's tile -- are
`num_cats` floats apart. On the CoDD circuit that is 505 KB, one 32-byte sector per useful float, and
the kernel sits at ~50 GB/s of useful bandwidth however its tiles are shaped. A transposed copy
`[cats, rows]` puts consecutive nodes 1 float apart; MEASURED bit-identical and ~19.8x on that loop.

WHY THIS NEEDS A SCOPE, rather than a cache with an invalidation rule. There is no reliable way to
notice that `params` changed: `mini_batch_em` writes through a Triton kernel and `tensor._version`
does NOT move (MEASURED), and neither does `params.data[...] = x`. A derived copy keyed on any
implicit signal would silently serve stale emissions. So the lifetime is made explicit instead: the
copy exists only inside the `with` block, and the caller's side of the bargain is not to change
parameters inside it.

THE COPY IS REBUILT ON EVERY ENTRY, deliberately -- there is no `persistent` option. Rebuilding is a
~1 GB read+write (~1-2 ms) against the ~25-50 ms it saves over a generation, so paying it per entry
buys unconditional correctness for a few percent of the benefit. A copy that outlived the block would
put the staleness problem straight back.

    with pyjuice.fast_inference():
        for step in range(num_steps):
            pc(data, ...)                       # builds the copy on the first forward
            juice.queries.sample(pc, ...)
    # copies freed here

:note: NOT thread-safe, and the scope is process-global: it is a mode, like `torch.no_grad()`. The
       depth is a counter, so nested blocks compose and the inner exit does not free what the outer
       one is still using.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List, Optional


class _FastInferenceState():
    __slots__ = ("depth", "allow_param_copy", "allow_bf16_params", "layers")

    def __init__(self):
        self.depth = 0
        self.allow_param_copy = False
        self.allow_bf16_params = False
        # The layers that actually built something, so exit frees exactly those. Held by strong
        # reference only for the duration of the block, which is bounded by construction.
        self.layers: List = []


_STATE = _FastInferenceState()

#: Fill a discarded copy with NaN and KEEP the allocation, so anything still reading it produces NaN
#: rather than whatever the allocator recycled into that memory. Costs the copy's memory for the
#: process's lifetime, so it is for debugging a suspected stale read, not for production.
_POISON = os.environ.get("PYJUICE_FAST_INFERENCE_POISON", "0") == "1"


def is_active() -> bool:
    """Whether a :func:`fast_inference` scope is currently open."""
    return _STATE.depth > 0


def param_copies_allowed() -> bool:
    """Whether layers inside the current scope may build derived parameter copies."""
    return _STATE.depth > 0 and _STATE.allow_param_copy


def bf16_params_allowed() -> bool:
    """Whether layers inside the current scope may hold a bf16 copy of their parameters.

    Separate from :func:`param_copies_allowed` because it is a different KIND of trade: that one
    spends memory, this one spends ACCURACY. Off unless asked for.
    """
    return _STATE.depth > 0 and _STATE.allow_param_copy and _STATE.allow_bf16_params


def register_layer(layer) -> None:
    """Record that `layer` built a derived copy, so the scope's exit can free it."""
    if _STATE.depth > 0:
        _STATE.layers.append(layer)


def _release_all() -> None:
    layers, _STATE.layers = _STATE.layers, []
    for layer in layers:
        release = getattr(layer, "_release_fast_inference_params", None)
        if release is not None:
            release(poison = _POISON)


@contextmanager
def fast_inference(allow_param_copy: bool = True, allow_bf16_params: Optional[bool] = None):
    """
    Open a scope in which the circuit's parameters will not change.

    Inside it, layers may trade memory for speed by holding derived copies of their parameters --
    today a transposed emission table for the top-k soft-evidence forward, and optionally a bfloat16
    emission table for the dense one. Copies are built LAZILY, on the first forward inside the scope
    that would use one, so a circuit that is never run never pays; and they are freed on exit.

    :param allow_param_copy: permit copies that cost extra device memory. The transposed emission
                             table is the size of the emission table itself (494 MB on the CoDD
                             circuit). Set `False` to enter the scope without paying that, which
                             leaves the ordinary kernels in place.
    :type allow_param_copy: bool

    :param allow_bf16_params: permit a bfloat16 copy of the emission table for the dense soft-evidence
                              forward. **OFF BY DEFAULT, and unlike `allow_param_copy` this one costs
                              ACCURACY, not memory** -- it halves the bytes the forward reads, which
                              is the only lever left once that forward is at the memory roofline.
                              MEASURED on the CoDD circuit: the logZ GEMM 0.73 -> 0.30 ms (2.4x, and
                              below the fp32 streaming floor because it reads half the table), at a
                              maximum RELATIVE error of 8.9e-03 on logZ. For calibration that is ~100x
                              looser than the tf32 `tl.dot` path already in the Triton kernels
                              (8.6e-05). It also SAVES memory rather than costing it -- the bf16 copy
                              is half the table, against the transposed copy's full size.
                              Run an eval before turning this on; it changes answers.
                              `None` (the default) means INHERIT: off at the outermost scope, and
                              whatever the enclosing scope permitted for a nested one. That is what
                              makes it behave like `allow_param_copy`, whose `True` default already
                              leaves an enclosing scope's choice alone -- without the sentinel, a
                              nested plain `fast_inference()` would silently switch bf16 off for its
                              duration, having never been asked to.
                              (float16 was measured and is UNUSABLE here: emission probabilities over
                              a 126k vocabulary underflow its exponent range and the forward returns
                              `inf`. bfloat16 keeps fp32's exponent range, which is the whole reason
                              it works.)
    :type allow_bf16_params: bool

    :note: THE CONTRACT: parameters must not change inside the block. Nothing can enforce this --
           an EM step writes through a raw pointer that PyTorch's version counter never sees -- so a
           derived copy would go stale silently. The in-repo parameter-writing paths
           (`mini_batch_em`, `_init_parameters`, `to`) drop the copies defensively, but a direct
           write to `params.data` is undetectable. Do not train inside this scope.
    """
    _STATE.depth += 1
    outer_allow = _STATE.allow_param_copy
    # An inner scope cannot widen what an outer one permitted: a caller that opened the outer scope
    # with `allow_param_copy = False` did so to bound memory, and a nested call should not overrule
    # that. It can, however, narrow it.
    _STATE.allow_param_copy = (allow_param_copy and outer_allow) if _STATE.depth > 1 else allow_param_copy
    outer_bf16 = _STATE.allow_bf16_params
    # Same narrowing rule as above: an inner scope may turn bf16 OFF but never on. A caller that
    # opened the outer scope in fp32 did so for accuracy, and a nested call must not quietly spend it.
    if allow_bf16_params is None:
        _STATE.allow_bf16_params = outer_bf16 if _STATE.depth > 1 else False
    else:
        _STATE.allow_bf16_params = (allow_bf16_params and outer_bf16) if _STATE.depth > 1 \
            else allow_bf16_params
    try:
        yield
    finally:
        _STATE.depth -= 1
        _STATE.allow_param_copy = outer_allow
        _STATE.allow_bf16_params = outer_bf16
        if _STATE.depth == 0:
            _release_all()
