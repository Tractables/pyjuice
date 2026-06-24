import pytest
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import FastJITFunction3x


# Raw kernel function (NOT @triton.jit -- FastJITFunction3x / triton.jit wrap it themselves).
def _scale_kernel(x_ptr, y_ptr, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask = mask) * scale, mask = mask)


@pytest.mark.skipif(not (torch.cuda.is_available() and int(triton.__version__.split(".")[0]) >= 3),
                    reason="FastJITFunction3x targets triton >= 3 on CUDA")
def test_fast_jit3x_matches_reference_no_silent_miscache():
    """
    The triton>=3 fast launcher caches the compiled kernel under a cheap signature and relaunches it
    directly. Its signature must be a SUPERSET of everything triton specializes on -- otherwise a
    cache hit would silently reuse a kernel compiled for a DIFFERENT alignment / dtype / constexpr.
    This test hammers exactly those axes and checks bit-identical results vs the stock triton.jit
    launch on every call (a too-coarse signature would silently corrupt at least one).
    """
    dev = torch.device("cuda:0")
    ref = triton.jit(_scale_kernel)
    fast = FastJITFunction3x(_scale_kernel)

    torch.manual_seed(0)
    # Vary: size, dtype, constexpr BLOCK, scalar `scale`, AND pointer alignment (offset slices) --
    # interleaved and repeated so cached entries are exercised, including the aligned/misaligned pair
    # that share constexprs (the case a coarse signature would mis-cache).
    configs = []
    for dtype in (torch.float32, torch.float16):
        for BLOCK in (64, 128):
            for n in (1000, 4096):
                for off in (0, 1, 3):           # 0 -> 16B aligned; 1/3 -> misaligned
                    configs.append((dtype, BLOCK, n, off))
    configs = configs * 3                        # repeat -> hit the cached fast path many times

    for dtype, BLOCK, n, off in configs:
        base = torch.randn(n + off, device = dev, dtype = dtype)
        x = base[off:]
        scale = 2.5
        grid = (triton.cdiv(n, BLOCK),)
        y_ref = torch.zeros(n, device = dev, dtype = dtype)
        y_fast = torch.zeros(n, device = dev, dtype = dtype)
        ref[grid](x, y_ref, n, scale, BLOCK = BLOCK)
        fast[grid](x, y_fast, n, scale, BLOCK = BLOCK)
        torch.cuda.synchronize()
        assert torch.equal(y_ref, y_fast), \
            f"fast launcher diverged from triton.jit (dtype={dtype}, BLOCK={BLOCK}, n={n}, off={off})"
        # and both must equal the actual math
        assert torch.allclose(y_fast.float(), (x * scale).float(), atol = 1e-2), "wrong result"


if __name__ == "__main__":
    test_fast_jit3x_matches_reference_no_silent_miscache()
    print("ok")
