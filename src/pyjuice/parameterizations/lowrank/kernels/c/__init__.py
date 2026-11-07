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
