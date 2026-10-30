"""
The dfa_exact backend's CUDA kernels (``fused_sum.cu``, ``liveness.cu``, ``walk.cu``; ``dense_live_mma.cu`` its own
extension), JIT-compiled on first use like pyjuice's other extensions
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
    """The extension (``fused_sum_cuda``, ``ctas_per_sm``, ``liveness_fits``, ``liveness_cuda``,
    ``ibtd_cuda``), or None if it is unavailable."""
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
                                                                os.path.join(_THIS_DIR, "liveness.cu"),
                                                                os.path.join(_THIS_DIR, "walk.cu")],
                           extra_cuda_cflags = ["-O3", f"-arch=sm_{cc[0]}{cc[1]}", "--use_fast_math", "-DNDEBUG"],
                           verbose = False)
    except Exception as e:
        warnings.warn(f"pyjuice CUDA kernel 'pyjuice_dfa_exact_cuda' failed to compile ({type(e).__name__}: {e}). "
                      f"The dfa_exact backend runs its Triton sum kernels instead.", RuntimeWarning)
        _module = None
    return _module


_dl_module = None
_dl_attempted = False


def get_dense_live_module():
    """The tensor-core dense live-column product (``dense_live_mma.cu``: ``dense_live_mma``, ``dense_live_mma_configs``),
    its own extension -- it builds with pyjuice's CUTLASS flags (`pyjuice.layer.kernels.c._compile_flags`: sm_90 and
    later) --, or None if it is unavailable (warns once): the dense sums then run their chain of kernels."""
    global _dl_module, _dl_attempted
    if _dl_attempted:
        return _dl_module
    _dl_attempted = True
    if not ENABLE_CUDA_KERNELS:
        return None
    from pyjuice.layer.kernels.c import _compile_flags
    flags = _compile_flags()
    if flags is None:
        return None
    cuda_cflags, ldflags = flags
    from pyjuice.utils.cuda_ext import jit_load
    try:
        _dl_module = jit_load("pyjuice_dfa_exact_dense_live_cuda", sources = [os.path.join(_THIS_DIR, "dense_live_mma.cu")],
                              extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags, verbose = False)
    except Exception as e:
        warnings.warn(f"pyjuice CUDA kernel 'pyjuice_dfa_exact_dense_live_cuda' failed to compile ({type(e).__name__}: "
                      f"{e}). The dense sums run their chain of kernels instead.", RuntimeWarning)
        _dl_module = None
    return _dl_module


_dl_arms_cache = {}


def dense_live_arms(M: int, K: int, L: int):
    """The tensor-core dense live-column product's codes (config id | split << 8) for a product of M rows over K at L live
    lanes: the configs whose column tile is the narrowest that covers L, either row tile, each at the split of K that
    brings the grid nearest two waves of the SMs and at half of it (at most 16, at most the K tiles) -- a few to measure
    against the chain. Empty from 64 lanes on (the chain's products are cuBLAS's strength there) or without the
    module."""
    if not 1 <= L < 64 or M < 1 or K < 1:
        return []
    mod = get_dense_live_module()
    if mod is None:
        return []
    key = (M, K, L, torch.cuda.current_device())
    arms = _dl_arms_cache.get(key)
    if arms is not None:
        return arms
    from pyjuice.layer.kernels.c import _sms
    sms, knt = _sms(), -(-K // 64)
    cover = min(bn for bn in (16, 32, 64) if bn >= L)
    arms = []
    for c, (bm, bn) in enumerate([(int(v[0]), int(v[1])) for v in mod.dense_live_mma_configs()]):
        if bn != cover:
            continue
        tiles = -(-L // bn) * -(-M // bm)
        split = 1
        while split * 2 <= min(16, knt) and tiles * split * 2 <= 2 * sms:
            split *= 2
        for sp in sorted({split, max(1, split // 2)}):
            arms.append(c | (sp << 8))
    _dl_arms_cache[key] = arms
    return arms
