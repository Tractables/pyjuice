"""
Lazy loaders for `LowRank`'s CUDA kernels, the sources beside this file: JIT-compiled on first use, and if
anything about the toolchain is missing (no `nvcc`, no compiler, compile error) the loader disables itself
and the caller falls back to its Triton or torch kernels. The switch and the compile flags are the
parameterizations' (`pyjuice.parameterizations.kernels.c`).
"""

import os
import warnings

from pyjuice.parameterizations.kernels.c import ENABLE_CUDA_KERNELS, _compile_flags

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


_lowrank_module = None
_lowrank_attempted = False


def get_lowrank_module():
    """The low-rank mixture's elementwise kernels around the plain sum-layer kernels (`lowrank_mixture.cu`), or None
    if unavailable (warns once): `LowRank`' "sum_exp" form then keeps its torch passes. Plain CUDA."""
    global _lowrank_module, _lowrank_attempted

    if _lowrank_attempted:
        return _lowrank_module

    _lowrank_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _lowrank_module = jit_load(
            "pyjuice_lowrank_mixture_cuda",
            sources = [os.path.join(_THIS_DIR, "lowrank_mixture.cu")],
            extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_lowrank_mixture_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The low-rank mixture keeps its torch passes.", RuntimeWarning)
        _lowrank_module = None
    return _lowrank_module


_exp_sum_module = None
_exp_sum_attempted = False


def get_exp_sum_module():
    """The "exp_sum" form's fused passes (`lowrank_exp_sum.cu`: the scale made tile by tile, never stored), or None if
    unavailable (warns once): the form then keeps its torch passes. Plain CUDA."""
    global _exp_sum_module, _exp_sum_attempted

    if _exp_sum_attempted:
        return _exp_sum_module

    _exp_sum_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _exp_sum_module = jit_load(
            "pyjuice_lowrank_exp_sum_cuda",
            sources = [os.path.join(_THIS_DIR, "lowrank_exp_sum.cu")],
            extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_lowrank_exp_sum_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The low-rank exp_sum form keeps its torch passes.", RuntimeWarning)
        _exp_sum_module = None
    return _exp_sum_module


_exp_sum_h_module = None
_exp_sum_h_attempted = False


def get_exp_sum_h_module():
    """The "exp_sum" form's fused passes at the default precision from fp16-rounded factors (`lowrank_exp_sum_h.cu`:
    phi one fp16 MMA a tile), or None if unavailable (warns once; silently below compute capability 9.0, whose bulk
    copies and tensor-map reduces they take): the bf16 three-term passes of `lowrank_exp_sum.cu` then serve. Plain
    CUDA."""
    global _exp_sum_h_module, _exp_sum_h_attempted

    if _exp_sum_h_attempted:
        return _exp_sum_h_module

    _exp_sum_h_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _exp_sum_h_module = jit_load(
            "pyjuice_lowrank_exp_sum_h_cuda",
            sources = [os.path.join(_THIS_DIR, "lowrank_exp_sum_h.cu")],
            extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_lowrank_exp_sum_h_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The exp_sum form keeps its bf16 three-term passes.", RuntimeWarning)
        _exp_sum_h_module = None
    return _exp_sum_h_module


_fp8_module = None
_fp8_attempted = False


def get_pass_end_fp8_module():
    """The "sum_exp" form's pass's end operands in e4m3 (`lowrank_pass_end_fp8.cu`: `LowRank(flow_precision =
    "fp8")`), or None if unavailable (warns once): the pass's end then keeps its bf16 operands. Plain CUDA."""
    global _fp8_module, _fp8_attempted

    if _fp8_attempted:
        return _fp8_module

    _fp8_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _fp8_module = jit_load(
            "pyjuice_lowrank_pass_end_fp8_cuda",
            sources = [os.path.join(_THIS_DIR, "lowrank_pass_end_fp8.cu")],
            extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_lowrank_pass_end_fp8_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The low-rank pass's end keeps its bf16 operands.", RuntimeWarning)
        _fp8_module = None
    return _fp8_module


_stream_module = None
_stream_attempted = False


def get_lowrank_stream_module():
    """The "sum_exp" form's forward over the virtual batch, the virtual child values never stored -- the `expand` fused
    into the stream kernel's pre-pass, or the whole layer one launch (`lowrank_stream.cu`: CuTe and TMA, the sum-layer
    kernels' toolchain) --, or None if unavailable (warns once): the virtual batch then keeps the plain path
    (`lowrank_expand`, the twin's forward)."""
    global _stream_module, _stream_attempted

    if _stream_attempted:
        return _stream_module

    _stream_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags as _cute_flags
    flags = _cute_flags()
    if flags is None:
        return None
    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _stream_module = jit_load(
            "pyjuice_lowrank_stream_cuda",
            sources = [os.path.join(_THIS_DIR, "lowrank_stream.cu")],
            extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_lowrank_stream_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The low-rank mixture's virtual batch keeps its plain path.", RuntimeWarning)
        _stream_module = None
    return _stream_module


_fused_end_module = None
_fused_end_attempted = False


def get_pass_end_fused_module():
    """The pass's end's two products as split-K GEMMs that form their operands from the kept exponentials in their own
    stages (`lowrank_pass_end_fused.cu`: CuTe and TMA, the sum-layer kernels' toolchain), or None if unavailable (warns
    once): the pass's end then forms the operands (`lowrank_kept_operands`) for cuBLAS."""
    global _fused_end_module, _fused_end_attempted

    if _fused_end_attempted:
        return _fused_end_module

    _fused_end_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags as _cute_flags
    flags = _cute_flags()
    if flags is None:
        return None
    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _fused_end_module = jit_load(
            "pyjuice_lowrank_pass_end_fused_cuda",
            sources = [os.path.join(_THIS_DIR, "lowrank_pass_end_fused.cu")],
            extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_lowrank_pass_end_fused_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The low-rank pass's end forms its operands for cuBLAS.", RuntimeWarning)
        _fused_end_module = None
    return _fused_end_module
