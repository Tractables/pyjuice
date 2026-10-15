"""
The dfa_exact backend's CUDA kernels (``fused_sum.cu``, ``liveness.cu``), JIT-compiled on first use like pyjuice's other extensions
(:mod:`pyjuice.nodes.external_params.kernels.c`): if anything about the toolchain is missing -- no ``nvcc``, a GPU
without TF32 tensor cores, a compile error -- the loader disables itself (warning once) and the callers run their
Triton kernels. ``PYJUICE_DISABLE_CUDA_KERNELS=1`` turns it off with the others. Nothing here runs at import time.
"""

import os
import warnings

import torch

ENABLE_CUDA_KERNELS = os.environ.get("PYJUICE_DISABLE_CUDA_KERNELS", "0") != "1"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_module = None
_attempted = False


def get_module():
    """The extension (``fused_sum_cuda``, ``ctas_per_sm``, ``liveness_fits``, ``liveness_cuda``), or None if it is
    unavailable."""
    global _module, _attempted
    if _attempted:
        return _module
    _attempted = True
    if not ENABLE_CUDA_KERNELS or not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import CUDA_HOME
    cc = torch.cuda.get_device_capability()
    if CUDA_HOME is None or cc < (8, 0):                                   # mma.sync on TF32: sm_80 and later
        return None
    from pyjuice.utils.cuda_ext import jit_load
    try:
        _module = jit_load("pyjuice_dfa_exact_cuda", sources = [os.path.join(_THIS_DIR, "fused_sum.cu"),
                                                                os.path.join(_THIS_DIR, "liveness.cu")],
                           extra_cuda_cflags = ["-O3", f"-arch=sm_{cc[0]}{cc[1]}", "--use_fast_math", "-DNDEBUG"],
                           verbose = False)
    except Exception as e:
        warnings.warn(f"pyjuice CUDA kernel 'pyjuice_dfa_exact_cuda' failed to compile ({type(e).__name__}: {e}). "
                      f"The dfa_exact backend runs its Triton sum kernels instead.", RuntimeWarning)
        _module = None
    return _module
