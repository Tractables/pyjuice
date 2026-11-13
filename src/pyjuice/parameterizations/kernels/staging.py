"""
Triton tiled transpose for external-parameter staging.

A port of `c/staging_transpose.cu`, with the same contract -- `dst[n, b] = src[b, n]` over flat
contiguous memory -- so the two are interchangeable. It exists because the C++ extension is an optional
build: without it, staging fell back to a strided `Tensor.copy_`, which goes through TensorIterator and
leaves one side of the transpose uncoalesced (8.2 us against 2.1 us for the same 4 MB). Triton is
already a hard dependency, so there is no reason for the fast path to hinge on a compiler being present.
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton_jit
def _staging_transpose_kernel(src, dst, B, N, BLOCK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    offs_b = pid_b * BLOCK + tl.arange(0, BLOCK)

    # int64 because `B * N` is a whole staging buffer -- a large model at a large batch passes 2^31,
    # and `tl.arange` is int32.
    rn = offs_n.to(tl.int64)
    rb = offs_b.to(tl.int64)

    # Read coalesced along N, write coalesced along B. The register transpose is what buys the second
    # one; indexing the store directly with the untransposed tile would stride by B per lane.
    x = tl.load(src + rb[:,None] * N + rn[None,:],
                mask = (offs_b[:,None] < B) & (offs_n[None,:] < N), other = 0.0)
    tl.store(dst + rn[:,None] * B + rb[None,:], tl.trans(x),
             mask = (offs_n[:,None] < N) & (offs_b[None,:] < B))


def staging_transpose_triton(dst, src, B: int, N: int) -> None:
    """`dst[n, b] = src[b, n]`, matching `staging_transpose` from the CUDA extension argument for
    argument so the two are drop-in for each other."""
    assert src.is_contiguous() and dst.is_contiguous(), \
        "staging_transpose: both tensors must be contiguous"
    assert src.numel() == B * N and dst.numel() == B * N, \
        "staging_transpose: element count does not match B * N"

    if B == 0 or N == 0:
        return None

    _staging_transpose_kernel[(triton.cdiv(N, 32), triton.cdiv(B, 32))](
        src = src, dst = dst, B = B, N = N, BLOCK = 32)
