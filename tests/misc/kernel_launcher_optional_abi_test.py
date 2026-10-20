"""Optional fast-launch ABI discovery must not parse a kernel twice."""

import builtins
from types import SimpleNamespace
from unittest.mock import patch

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
