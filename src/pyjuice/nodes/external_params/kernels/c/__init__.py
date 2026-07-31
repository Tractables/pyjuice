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
