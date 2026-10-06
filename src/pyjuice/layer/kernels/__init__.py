"""Triton kernels for the circuit layers, split by computation phase and sparsity pattern."""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit


_BROADCAST_SUM_NOTE = """
Never write a small matmul as `tl.sum(X[:,:,None] * Y[None,:,:], axis = 1)` with fp32 operands.

Triton (3.7) pattern-matches exactly that spelling -- operand order and reduction axis -- and rewrites
it into a tensor-core `tl.dot`. For fp32 operands that is wrong twice over, MEASURED with X: [M, K],
Y: [K, N], M and N >= 16:
  * K in {1, 2, 4}: the result is multiplied by 8 / K (8x, 4x, 2x). The MMA's reduction is 8 wide and
    the short dimension is padded by REPEATING it. In pyjuice this gave a log-likelihood off by ln 2
    per sum layer with fewer than 8 children per block and by ln 2 per variable for a soft-evidence
    leaf with fewer than 8 categories (both at batch >= 16), and it would have doubled the block-sparse
    parameter flows at batch 4.
  * K >= 8: a TF32 dot, which truncates the operands -- a ~6e-4 LOW bias per product that compounds
    through the layers (see `_round_to_tf32`).
These are exactly the small-tile fallback branches of kernels whose large-tile branch is a `tl.dot`,
i.e. code written to be exact fp32.

The equivalent `tl.sum(tl.trans(X)[:,:,None] * Y[:,None,:], axis = 0)` is not matched and stays in fp32
FMAs; it measured exact at every shape and costs nothing measurable. (Reducing over axis 2, swapping the
operands, or a three-operand product are not matched either -- but use the axis-0 form, so there is one
spelling to recognise.) bf16 operands are fine: the rewrite is then a bf16 dot, exact for bf16 inputs;
the one such site is marked `# broadcast-sum ok: bf16`.

`tests/misc/broadcast_sum_rewrite_test.py` forbids the spelling in the source tree and re-checks, against
the installed Triton, that the axis-0 form is still exact -- so an upgrade that starts matching it too
fails loudly instead of silently.
"""

# The forbidden spelling, on one line, as the guard test searches for it.
_BROADCAST_SUM_FORBIDDEN = r"tl\.sum\(.*\[\s*:\s*,\s*:\s*,\s*None\s*\]\s*\*.*\[\s*None\s*,\s*:\s*,\s*:\s*\].*axis\s*=\s*1\s*\)"


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
