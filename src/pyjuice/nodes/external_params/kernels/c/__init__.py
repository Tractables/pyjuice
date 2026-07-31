"""
Lazy loader for the CUDA low-rank forward correction.

Mirrors `pyjuice.layer.kernels.c`: the extension is JIT-compiled on first use, and if anything about
the toolchain is missing (no `nvcc`, no compiler, compile error) the loader disables itself and the
caller falls back to the Triton kernels. Nothing here is imported at `import pyjuice` time.
"""

import os
import warnings

import torch


# Same escape hatch as the layer kernels, so both can be turned off together.
ENABLE_CUDA_KERNELS = os.environ.get("PYJUICE_DISABLE_CUDA_KERNELS", "0") != "1"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# None = not attempted yet, module = loaded, False = attempted and unavailable
_module = None
_attempted = False


def _compile_flags():
    if not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        return None

    cc = torch.cuda.get_device_capability()

    # Plain arch (no `a` suffix): only ordinary CUDA is used -- no TMA, no arch-specific MMA.
    return ["-O3", f"-arch=sm_{cc[0]}{cc[1]}", "--use_fast_math", "-DNDEBUG"]


_sb_module = None
_sb_attempted = False


def get_sb_module():
    """The SMALL-BATCH block-scale extension, or None if unavailable (warns once).

    Deliberately its own extension, and plain CUDA: it uses no CuTe, no TMA and no CUTLASS, so unlike
    `get_cute_module` it compiles on any CUDA GPU and must not be taken down by a CUTLASS failure. It
    serves the batches the CuTe fork cannot tile at all (`batch % 64 != 0`).
    """
    global _sb_module, _sb_attempted

    if _sb_attempted:
        return _sb_module

    _sb_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _sb_module = jit_load(
            "pyjuice_blockscale_sb_cuda",
            sources = [os.path.join(_THIS_DIR, "blockscale_smallbatch_forward.cu")],
            extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_sb_cuda' failed to compile "
            f"({type(e).__name__}: {e}). The small-batch block-scale path is unavailable.",
            RuntimeWarning)
        _sb_module = None

    return _sb_module


_bw_modules = {}


def _get_bw_module(name: str, source: str):
    """
    Load one of the block-scale BACKWARD kernels, or None if unavailable (warns once).

    Each is its OWN extension rather than one module holding both: they are forks of two different
    standard kernels and each carries that kernel's file-scope TMA descriptor cache and PTX helpers, so
    compiling them into a single translation unit would collide at link time. They need the same
    CuTe/CUTLASS/TMA toolchain as the forward.
    """
    if name in _bw_modules:
        return _bw_modules[name]

    _bw_modules[name] = None
    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags as _cute_flags
    flags = _cute_flags()
    if flags is None:
        return None
    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _bw_modules[name] = jit_load(name, sources = [os.path.join(_THIS_DIR, source)],
                                     extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags,
                                     verbose = False)
    except Exception as e:
        warnings.warn(f"pyjuice CUDA kernel '{name}' failed to compile ({type(e).__name__}: {e}). "
                      f"The block-scale backward is unavailable.", RuntimeWarning)
        _bw_modules[name] = None

    return _bw_modules[name]


_sb_bw_module = None
_sb_bw_attempted = False


def get_sb_bw_module():
    """
    The SMALL-BATCH block-scale backward extension, or None if unavailable (warns once).

    Plain CUDA, like `get_sb_module`: forks of the standard small-batch backward kernels, which use no
    CuTe, TMA or CUTLASS, so this compiles on any CUDA GPU and a CUTLASS failure cannot take it down.
    Both kernels share one extension -- unlike the CuTe forks neither carries a file-scope TMA
    descriptor cache to collide over, so one build serves both.
    """
    global _sb_bw_module, _sb_bw_attempted

    if _sb_bw_attempted:
        return _sb_bw_module

    _sb_bw_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _sb_bw_module = jit_load(
            "pyjuice_blockscale_sb_bw_cuda",
            sources = [os.path.join(_THIS_DIR, "blockscale_smallbatch_backward.cu")],
            extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_sb_bw_cuda' failed to compile "
            f"({type(e).__name__}: {e}). The small-batch block-scale backward is unavailable.",
            RuntimeWarning)
        _sb_bw_module = None

    return _sb_bw_module


def get_ele_bw_module():
    """Element-flow backward for the per-block multiplicative gate."""
    return _get_bw_module("pyjuice_blockscale_ele_bw_cuda", "blockscale_ele_backward.cu")


def get_par_bw_module():
    """Parameter-flow backward for the per-block multiplicative gate."""
    return _get_bw_module("pyjuice_blockscale_par_bw_cuda", "blockscale_par_backward.cu")


def get_module():
    """The loaded extension, or None if it is unavailable (warns once)."""
    global _module, _attempted

    if _attempted:
        return _module

    _attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _module = jit_load("pyjuice_external_params_cuda",
                           sources = [os.path.join(_THIS_DIR, "lowrank_forward.cu"),
                                      os.path.join(_THIS_DIR, "lowrank_backward.cu"),
                                      os.path.join(_THIS_DIR, "staging_transpose.cu"),
                                      os.path.join(_THIS_DIR, "bindings.cu")],
                           extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_external_params_cuda' failed to compile "
            f"({type(e).__name__}: {e}). Falling back to the Triton kernels.", RuntimeWarning)
        _module = None

    return _module


def is_available() -> bool:
    return get_module() is not None


# --------------------------------------------------------------------------- CuTe / TMA extension

# Separate from the plain one: the block-scale forward is a fork of pyjuice's CuTe sum kernel and needs
# CUTLASS headers, an arch-specific `sm_XXa` flag and the driver API, none of which the plain kernels
# want. Two extensions also means a CUTLASS-related failure cannot disable the low-rank kernels.
_cute_module = None
_cute_attempted = False


def get_cute_module():
    """The CuTe/TMA extension, or None if its toolchain is unavailable (warns once)."""
    global _cute_module, _cute_attempted

    if _cute_attempted:
        return _cute_module

    _cute_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    # Reuse the sum layer's toolchain probe: same requirements, already cached, and it warns once with
    # a message that names the actual missing piece (CUTLASS path, compute capability, nvcc).
    from pyjuice.layer.kernels.c import _compile_flags

    flags = _compile_flags()
    if flags is None:
        return None

    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _cute_module = jit_load("pyjuice_blockscale_cuda",
                                sources = [os.path.join(_THIS_DIR, "blockscale_forward.cu")],
                                extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags,
                                verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_cuda' failed to compile "
            f"({type(e).__name__}: {e}). `BlockScaleSumParams` will be unavailable.", RuntimeWarning)
        _cute_module = None

    return _cute_module


def cute_is_available() -> bool:
    return get_cute_module() is not None
