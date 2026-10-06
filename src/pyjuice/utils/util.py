from __future__ import annotations

import math


def max_cdf_power_of_2(val: int):
    count = 0
    while True:
        halfval = val // 2

        if halfval * 2 != val:
            break

        val = halfval
        count += 1

    return 2 ** count


def max_power_of_2_factor(n):
    if n == 0:
        return 0
    if n % 2 != 0:
        return 1

    power_of_2 = 1
    while n % 2 == 0:
        power_of_2 *= 2
        n //= 2  # Use integer division

    return power_of_2


def cuda_graph_key(t):
    """Identify a buffer the way a CAPTURED GRAPH does -- by the memory it baked in, not by the
    Python object that happens to wrap it.

    `id()` was used for this and is wrong in BOTH directions.

    Too STRICT: reallocating `node_mars` for a different batch size produces a new Python object, so
    the old graph is never reused and a batch-alternating loop re-captures on nearly every call.
    Measured on the CoDD circuit alternating batch 1 and batch 10: **31 graphs recorded over 40
    steps** where 2 would do -- and each capture is 3 warm-up runs plus the capture itself, so the
    "graphed" path can end up SLOWER than eager. A temporary such as `x.view(-1)` is a new object on
    every call, so an `id()` of one identifies nothing at all.

    Too LOOSE: CPython recycles ids. Measured in the same run, **7 of 24 observed `id(node_mars)`
    values came back for a DIFFERENT batch size**. A recycled id paired with the same batch size
    matches a graph captured against memory that has since been freed and handed to something else --
    an illegal access, or silent corruption when the allocator kept it mapped. This is exactly the
    pattern adaptive decoding produces, since it alternates a batch-1 refine with a batch-(2C+2)
    dependence call every step.

    The address, shape, strides and dtype are what a graph actually depends on: if they all match, the
    pointers and layouts baked into the capture still describe the buffer we mean.
    """
    if t is None:
        return None
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
