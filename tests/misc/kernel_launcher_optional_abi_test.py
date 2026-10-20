"""Optional fast-launch ABI discovery must not parse a kernel twice."""

import builtins
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton.language as tl

import pyjuice.utils.kernel_launcher as launcher


def _kernel():
    pass


def test_missing_triton_knobs_constructs_only_the_stock_fallback():
    stock = SimpleNamespace(params=())
    original_import = builtins.__import__
    def import_without_knobs(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "triton" and "knobs" in fromlist:
            raise ImportError("this Triton release has no knobs ABI")
        return original_import(name, globals, locals, fromlist, level)

    with patch.object(launcher.triton, "__version__", "3.0.0"), \
            patch.object(launcher, "_FAST_LAUNCH", True), \
            patch.object(launcher.triton, "jit", return_value=stock) as compile_jit, \
            patch("builtins.__import__", side_effect=import_without_knobs):
        actual = launcher.triton_jit(_kernel)
    assert actual is stock
    compile_jit.assert_called_once_with(_kernel)


def test_available_knobs_keeps_the_same_fast_wrapper_and_one_jit():
    stock = SimpleNamespace(params=())
    knobs = SimpleNamespace()
    with patch.object(launcher.triton, "__version__", "3.5.0"), \
            patch.object(launcher, "_FAST_LAUNCH", True), \
            patch.object(launcher.triton, "knobs", knobs, create=True), \
            patch.object(launcher.triton, "jit", return_value=stock) as compile_jit:
        actual = launcher.triton_jit(_kernel)
    assert isinstance(actual, launcher.FastJITFunction3x)
    assert actual.jit_fn is stock and actual._knobs is knobs
    assert actual.n_params == 0 and actual.cache == {}
    compile_jit.assert_called_once_with(_kernel)


def test_fast_launch_opt_out_does_not_probe_the_optional_abi():
    stock = SimpleNamespace(params=())
    original_import = builtins.__import__
    def reject_knobs_probe(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "triton" and "knobs" in fromlist:
            raise AssertionError("fast launch is disabled")
        return original_import(name, globals, locals, fromlist, level)

    with patch.object(launcher.triton, "__version__", "3.0.0"), \
            patch.object(launcher, "_FAST_LAUNCH", False), \
            patch.object(launcher.triton, "jit", return_value=stock) as compile_jit, \
            patch("builtins.__import__", side_effect=reject_knobs_probe):
        assert launcher.triton_jit(_kernel) is stock
    compile_jit.assert_called_once_with(_kernel)


def _scale_kernel(x, y, n, scale, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + offsets, tl.load(x + offsets, mask=offsets < n) * scale, mask=offsets < n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA launch")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("offset", [0, 1])
def test_actual_factory_launch_matches_stock_triton(dtype, offset):
    # This checks the installed release's real selected path; the mocked ABI
    # tests above do not claim compatibility with other Triton releases.
    with patch.object(launcher, "_FAST_LAUNCH", True):
        actual = launcher.triton_jit(_scale_kernel)
    reference = launcher.triton.jit(_scale_kernel)
    for n, block in ((17, 128), (1024, 256), (17, 128)):
        x = torch.arange(n + offset, dtype=dtype, device="cuda:0")[offset:] * .125
        result, expected = torch.empty_like(x), torch.empty_like(x)
        grid = ((n + block - 1) // block,)
        actual[grid](x, result, n, 2., BLOCK=block)
        reference[grid](x, expected, n, 2., BLOCK=block)
        torch.cuda.synchronize()
        assert torch.equal(result, expected)
        assert torch.equal(result, x * 2.)
