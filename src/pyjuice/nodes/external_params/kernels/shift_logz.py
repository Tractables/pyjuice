"""
The +-log-Z normalizer shift, in Triton.

A port of `lowrank_shift_logz` from `c/lowrank_backward.cu`, with the same contract:

    node_mars[nids[r] + m, b] += sign * log_z[r, m, b]

`BlockScaleSumParams` stores `log N - log Z` in `node_mars` during the forward, and its backward needs
`log N`; it shifts by `+log Z` on the way in and by `-log Z` on the way out. That made a CUDA toolchain
a hard requirement for the gated backward -- every OTHER kernel on that path already has a Triton fork,
so this one elementwise add was the whole of what an nvcc-less machine was missing.

Bit-identical to the CUDA kernel: `sign` is exactly +-1, so `sign * log_z` is a negation or a no-op and
the remaining add rounds once either way.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton_jit
def _shift_logz_kernel(node_mars, nids, log_z, batch_size, num_rows,
                       block_size: tl.constexpr, sign, BLOCK_M: tl.constexpr, BLOCK_B: tl.constexpr):
    # `offs_m` walks the [rows x block_size] node rows, which is exactly `log_z`'s leading dimension.
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_m = offs_m < num_rows * block_size
    mask = mask_m[:,None] & (offs_b < batch_size)[None,:]

    # `block_size` is a constexpr power of two, so these are a shift and a mask.
    row = offs_m // block_size
    m = offs_m % block_size

    # int64 for the same reason the staging transpose needs it: `num_rows * block_size * batch_size`
    # is a whole activation buffer and passes 2^31 on a large model at a large batch.
    ob = offs_b.to(tl.int64)
    lz = tl.load(log_z + offs_m.to(tl.int64)[:,None] * batch_size + ob[None,:], mask = mask, other = 0.0)

    nid = tl.load(nids + row, mask = mask_m, other = 0)
    ptr = node_mars + (nid + m)[:,None] * batch_size + ob[None,:]
    # Each (node, sample) belongs to exactly one program, so the read-modify-write cannot race.
    tl.store(ptr, tl.load(ptr, mask = mask, other = 0.0) + sign * lz, mask = mask)


def shift_logz_triton(node_mars, nids, log_z, block_size: int, sign: float) -> None:
    """`node_mars[nids[r] + m, b] += sign * log_z[r, m, b]`, matching `lowrank_shift_logz` from the
    CUDA extension argument for argument so the two are drop-in for each other."""
    batch_size = node_mars.size(1)
    num_rows = nids.size(0)

    assert log_z.is_contiguous(), "shift_logz: `log_z` must be contiguous"
    assert log_z.numel() == num_rows * block_size * batch_size, \
        "shift_logz: `log_z` must hold num_rows * block_size * batch_size entries"

    if num_rows == 0 or block_size == 0 or batch_size == 0:
        return None

    BLOCK_B = min(triton.next_power_of_2(batch_size), 128)
    BLOCK_M = max(1024 // BLOCK_B, 1)

    _shift_logz_kernel[(triton.cdiv(num_rows * block_size, BLOCK_M), triton.cdiv(batch_size, BLOCK_B))](
        node_mars = node_mars, nids = nids, log_z = log_z, batch_size = batch_size,
        num_rows = num_rows, block_size = block_size, sign = sign,
        BLOCK_M = BLOCK_M, BLOCK_B = BLOCK_B)


def shift_logz(node_mars, nids, log_z, block_size: int, sign: float) -> None:
    """The same shift, taking the CUDA extension when it is built and the Triton port otherwise."""
    from .c import get_module

    mod = get_module()
    if mod is not None:
        mod.lowrank_shift_logz(node_mars, nids, log_z, block_size, sign)
    else:
        shift_logz_triton(node_mars, nids, log_z, block_size, sign)
