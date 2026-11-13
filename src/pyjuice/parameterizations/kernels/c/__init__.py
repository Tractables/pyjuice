"""
Lazy loaders for the external parameterizations' CUDA kernels.

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
                           sources = [os.path.join(_THIS_DIR, "shift_logz.cu"),
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
