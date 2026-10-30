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


_sb_bt_module = None
_sb_bt_attempted = False


def get_sb_bt_module():
    """The BATCH-TILED small-batch block-scale forward extension (`blockscale_smallbatch_forward_bt.cu`: the
    parameters loaded once per BT batch columns), or None if unavailable (warns once). Plain CUDA, like
    `get_sb_module`; its configs are arms of the same pick as that kernel's."""
    global _sb_bt_module, _sb_bt_attempted

    if _sb_bt_attempted:
        return _sb_bt_module

    _sb_bt_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    flags = _compile_flags()
    if flags is None:
        return None

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _sb_bt_module = jit_load(
            "pyjuice_blockscale_sb_bt_cuda",
            sources = [os.path.join(_THIS_DIR, "blockscale_smallbatch_forward_bt.cu")],
            extra_cuda_cflags = flags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_sb_bt_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The per-sample small-batch kernel and Triton remain.", RuntimeWarning)
        _sb_bt_module = None
    return _sb_bt_module


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


# The pre-pass forward (blockscale_forward_prepass.cu): CuTe (MMA atoms, ldmatrix), no TMA, the same toolchain gate.
# Its own extension, so a failure leaves the fork in place.
_prepass_module = None
_prepass_attempted = False


_sb_mma_module = None
_sb_mma_attempted = False


def get_sb_mma_module():
    """The tensor-core small-batch block-scale forward extension (`blockscale_smallbatch_forward_mma.cu`: N and Z both
    products of the parameter tile, split-K ranges merged), or None if its toolchain is unavailable (warns once)."""
    global _sb_mma_module, _sb_mma_attempted
    if _sb_mma_attempted:
        return _sb_mma_module
    _sb_mma_attempted = True
    if not ENABLE_CUDA_KERNELS:
        return None
    from pyjuice.layer.kernels.c import _compile_flags
    flags = _compile_flags()
    if flags is None:
        return None
    cuda_cflags, ldflags = flags
    from pyjuice.utils.cuda_ext import jit_load
    try:
        _sb_mma_module = jit_load("pyjuice_blockscale_sb_mma_cuda",
                                  sources = [os.path.join(_THIS_DIR, "blockscale_smallbatch_forward_mma.cu")],
                                  extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_sb_mma_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The block-scale forward keeps its other kernels.", RuntimeWarning)
        _sb_mma_module = None
    return _sb_mma_module


_bs_sb_mma_arms_cache = {}


def blockscale_sb_mma_arms(block_size: int, batch_size: int, num_edges: int, n_blocks: int, node_cbs: int,
                           row_contig: bool, fp32: bool):
    """The tensor-core small-batch block-scale forward's codes (config id | HP << 7 | split << 8) for a partition shape:
    node tiles dividing the block, batch tiles no wider than the narrowest that covers the batch, splits 1/2/4/8 within
    four waves of the SMs (`pyjuice.layer.kernels.c.smallbatch_fw_mma_arms`' rule). Below batch 16 only the
    high-precision form (every operand a bf16 hi + lo pair, ~2^-16 a term): the gated path's other forks there are fp32
    (the per-sample and batch-tiled kernels), from 16 the other gated forwards already use tensor-core precision. Empty
    where it does not apply: batch 64 or more, edges not whole 64-tiles, a row of edge blocks narrower than 64 that is
    not one contiguous run."""
    if batch_size >= 64 or num_edges % 64 or not (row_contig or node_cbs % 64 == 0):
        return []
    mod = get_sb_mma_module()
    if mod is None:
        return []
    key = (block_size, batch_size, num_edges, n_blocks, bool(fp32), torch.cuda.current_device())
    arms = _bs_sb_mma_arms_cache.get(key)
    if arms is not None:
        return arms
    from pyjuice.layer.kernels.c import _sms
    sms, knt = _sms(), num_edges // 64
    cover = min(bn for bn in (16, 32, 64) if bn >= batch_size)
    hp = batch_size < 16
    arms = []
    for c, (bm, bn) in enumerate([(int(v[0]), int(v[1])) for v in mod.blockscale_sb_mma_configs()]):
        if block_size % bm or bn > cover or (hp and not mod.blockscale_sb_mma_hp_fits(c, bool(fp32))):
            continue
        tiles = n_blocks * (block_size // bm) * ((batch_size + bn - 1) // bn)
        for sp in (1, 2, 4, 8):
            if sp > knt or (sp > 1 and tiles * sp > 4 * sms):
                continue
            arms.append(c | (128 if hp else 0) | (sp << 8))
    _bs_sb_mma_arms_cache[key] = arms
    return arms


_sb_ele_mma_module = None
_sb_ele_mma_attempted = False


def get_sb_ele_mma_module():
    """The tensor-core small-batch block-scale element backward (`blockscale_smallbatch_ele_mma.cu`: per-tile stabilizers
    folded under the gates, split ranges of whole parent blocks merged), or None if its toolchain is unavailable."""
    global _sb_ele_mma_module, _sb_ele_mma_attempted
    if _sb_ele_mma_attempted:
        return _sb_ele_mma_module
    _sb_ele_mma_attempted = True
    if not ENABLE_CUDA_KERNELS:
        return None
    from pyjuice.layer.kernels.c import _compile_flags
    flags = _compile_flags()
    if flags is None:
        return None
    cuda_cflags, ldflags = flags
    from pyjuice.utils.cuda_ext import jit_load
    try:
        _sb_ele_mma_module = jit_load("pyjuice_blockscale_sb_ele_mma_cuda",
                                      sources = [os.path.join(_THIS_DIR, "blockscale_smallbatch_ele_mma.cu")],
                                      extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_sb_ele_mma_cuda' failed to compile ({type(e).__name__}: {e}). "
            "The block-scale element backward keeps its other kernels.", RuntimeWarning)
        _sb_ele_mma_module = None
    return _sb_ele_mma_module


_bs_sb_ele_mma_arms_cache = {}


def blockscale_sb_ele_mma_arms(cs_block_size: int, block_size: int, batch_size: int, num_edges: int, n_blocks: int,
                               gate_cbs: int, fp32: bool):
    """The tensor-core small-batch block-scale element backward's codes (config id | HP << 7 | split << 8, `split` equal
    ranges of the 64-parent tiles, each whole parent blocks or inside one): child tiles dividing the child block under
    at most 16 gates within the shared memory (`blockscale_sb_ele_mma_fits`), batch tiles no wider than the narrowest
    covering the batch, splits 1/2/4/8/16 within four waves of the SMs. Below batch 16 only the high-precision form
    (bf16 parameters' exponentials hi + lo too), as the gated forward's. Empty where it does not apply: batch 64 or more,
    parent blocks not whole 64-tiles."""
    if batch_size >= 64 or block_size % 64 or num_edges % block_size:
        return []
    mod = get_sb_ele_mma_module()
    if mod is None:
        return []
    key = (cs_block_size, block_size, batch_size, num_edges, n_blocks, gate_cbs, bool(fp32), torch.cuda.current_device())
    arms = _bs_sb_ele_mma_arms_cache.get(key)
    if arms is not None:
        return arms
    from pyjuice.layer.kernels.c import _sms
    sms, knt, tpb = _sms(), num_edges // 64, block_size // 64
    cover = min(bn for bn in (16, 32, 64) if bn >= batch_size)
    hp = batch_size < 16
    arms = []
    for c, (bm, bn) in enumerate([(int(v[0]), int(v[1])) for v in mod.blockscale_sb_ele_mma_configs()]):
        if bn > cover or not mod.blockscale_sb_ele_mma_fits(c, bool(fp32), cs_block_size, gate_cbs, hp):
            continue
        tiles = n_blocks * (cs_block_size // bm) * ((batch_size + bn - 1) // bn)
        for sp in (1, 2, 4, 8, 16):
            rk = knt // sp
            if knt % sp or not (rk % tpb == 0 or tpb % rk == 0) or (sp > 1 and tiles * sp > 4 * sms):
                continue
            arms.append(c | (128 if hp else 0) | (sp << 8))
    _bs_sb_ele_mma_arms_cache[key] = arms
    return arms


def get_prepass_module():
    """The pre-pass block-scale forward extension, or None if its toolchain is unavailable (warns once)."""
    global _prepass_module, _prepass_attempted

    if _prepass_attempted:
        return _prepass_module

    _prepass_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags

    flags = _compile_flags()
    if flags is None:
        return None

    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _prepass_module = jit_load("pyjuice_blockscale_prepass_cuda",
                                   sources = [os.path.join(_THIS_DIR, "blockscale_forward_prepass.cu")],
                                   extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags,
                                   verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_prepass_cuda' failed to compile "
            f"({type(e).__name__}: {e}). The block-scale forward keeps its other kernels.", RuntimeWarning)
        _prepass_module = None

    return _prepass_module


# The pre-pass element-flow backward (blockscale_ele_backward_prepass.cu): CuTe (MMA atoms, ldmatrix), no TMA, the
# same toolchain gate. Its own extension, so a failure leaves the other element forks in place.
_ele_prepass_module = None
_ele_prepass_attempted = False


def get_ele_prepass_module():
    """The pre-pass block-scale element-flow backward extension, or None if its toolchain is unavailable (warns once)."""
    global _ele_prepass_module, _ele_prepass_attempted

    if _ele_prepass_attempted:
        return _ele_prepass_module

    _ele_prepass_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags

    flags = _compile_flags()
    if flags is None:
        return None

    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _ele_prepass_module = jit_load("pyjuice_blockscale_ele_prepass_cuda",
                                       sources = [os.path.join(_THIS_DIR, "blockscale_ele_backward_prepass.cu")],
                                       extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags,
                                       verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_ele_prepass_cuda' failed to compile "
            f"({type(e).__name__}: {e}). The block-scale element-flow backward keeps its other kernels.",
            RuntimeWarning)
        _ele_prepass_module = None

    return _ele_prepass_module


# The per-gate parameter sums sigma (blockscale_sigma.cu): plain CUDA, no CuTe. Its own extension, so a failure leaves
# `_sigma` with its torch gather.
_sigma_module = None
_sigma_attempted = False


def get_sigma_module():
    """The block-scale per-gate parameter sum extension, or None if its toolchain is unavailable (warns once)."""
    global _sigma_module, _sigma_attempted

    if _sigma_attempted:
        return _sigma_module

    _sigma_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags

    flags = _compile_flags()
    if flags is None:
        return None

    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _sigma_module = jit_load("pyjuice_blockscale_sigma_cuda",
                                 sources = [os.path.join(_THIS_DIR, "blockscale_sigma.cu")],
                                 extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags,
                                 verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_sigma_cuda' failed to compile "
            f"({type(e).__name__}: {e}). The block-scale gate sums keep their torch gather.", RuntimeWarning)
        _sigma_module = None

    return _sigma_module


# The split forward (blockscale_forward_stream.cu): CuTe (MMA atoms, ldmatrix), no TMA, the same toolchain gate as the
# pre-pass copy. Its own extension, so a failure leaves the other forward forks in place.
_stream_module = None
_stream_attempted = False


def get_stream_module():
    """The split block-scale forward extension, or None if its toolchain is unavailable (warns once)."""
    global _stream_module, _stream_attempted

    if _stream_attempted:
        return _stream_module

    _stream_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags

    flags = _compile_flags()
    if flags is None:
        return None

    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _stream_module = jit_load("pyjuice_blockscale_stream_cuda",
                                  sources = [os.path.join(_THIS_DIR, "blockscale_forward_stream.cu")],
                                  extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags,
                                  verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_stream_cuda' failed to compile "
            f"({type(e).__name__}: {e}). The block-scale forward keeps its other kernels.", RuntimeWarning)
        _stream_module = None

    return _stream_module


_ele_stream_module = None
_ele_stream_attempted = False


def get_ele_stream_module():
    """The split block-scale element backward extension, or None if its toolchain is unavailable (warns once)."""
    global _ele_stream_module, _ele_stream_attempted

    if _ele_stream_attempted:
        return _ele_stream_module

    _ele_stream_attempted = True

    if not ENABLE_CUDA_KERNELS:
        return None

    from pyjuice.layer.kernels.c import _compile_flags

    flags = _compile_flags()
    if flags is None:
        return None

    cuda_cflags, ldflags = flags

    from pyjuice.utils.cuda_ext import jit_load
    try:
        _ele_stream_module = jit_load("pyjuice_blockscale_ele_stream_cuda",
                                      sources = [os.path.join(_THIS_DIR, "blockscale_ele_backward_stream.cu")],
                                      extra_cuda_cflags = cuda_cflags, extra_ldflags = ldflags,
                                      verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel 'pyjuice_blockscale_ele_stream_cuda' failed to compile "
            f"({type(e).__name__}: {e}). The block-scale element backward keeps its other kernels.", RuntimeWarning)
        _ele_stream_module = None

    return _ele_stream_module
