"""Triton kernels for the circuit layers, split by computation phase and sparsity pattern."""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


@triton.jit
def _round_to_tf32(x):
    # Round-to-nearest onto the TF32 grid (10 mantissa bits), for the operands of an fp32 `tl.dot`.
    #
    # A TF32 `tl.dot` hands fp32 registers to the tensor core, which ignores the low 13 mantissa bits,
    # i.e. TRUNCATES every operand toward zero. For the all-positive operands of the sum layers that is
    # a BIAS of ~6e-4 low per product, not noise -- and since a layer's output is the next layer's
    # input, it compounds with depth (an HMM lost ~2% of its flow by layer 32). Rounding first leaves the
    # truncation nothing to drop: same tensor-core speed, zero-mean error.
    #
    # Every fp32 dot in a chain needs it, not just some: a forward that truncates while the backward
    # does not makes the flows come out HIGH by the forward's bias (the two used to cancel by accident).
    #
    # The bit trick rounds the magnitude (sign-agnostic). Infinities pass through unchanged; a finite
    # value within half a TF32 ulp of FLT_MAX rounds up to inf -- far outside anything stored here.
    b = x.to(tl.uint32, bitcast = True)
    return ((b + 0x1000) & 0xFFFFE000).to(tl.float32, bitcast = True)
