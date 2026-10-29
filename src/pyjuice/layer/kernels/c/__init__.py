"""Optional CUDA (CuTe/CUTLASS + TMA) fast-path kernels for pyjuice sum layers.

These kernels are JIT-compiled on first use via ``torch.utils.cpp_extension.load`` and compiled for
the GPU actually present (``sm_<cc>a``). They are entirely OPTIONAL: if anything required is missing
(a TMA-capable GPU sm_90+, ``nvcc``, or the header-only CUTLASS library) the loader disables itself
and the caller transparently falls back to the Triton kernels. So a plain ``pip install pyjuice``
never needs a CUDA toolchain or CUTLASS — those only gate the extra-fast path.

Locating CUTLASS (header-only) — in priority order:
  1. ``$PYJUICE_CUTLASS_PATH`` (or ``$CUTLASS_PATH``) pointing at a CUTLASS checkout
     (the dir that contains ``include/cute`` or the ``include`` dir itself).
  2. The ``nvidia-cutlass`` pip package, if installed (ships the headers).
  3. A vendored checkout at ``<this dir>/cutlass`` (e.g. a git submodule).
Set the env var to enable the fast path, e.g. ``export PYJUICE_CUTLASS_PATH=/path/to/cutlass``.
"""

import os
import warnings
from typing import Optional

import torch

# Module-level toggle (mirrors the FORWARD_SUM_* flags in sum_layer.py). Can also be turned off via
# the env var PYJUICE_DISABLE_CUDA_KERNELS=1 without touching code.
ENABLE_CUDA_KERNELS = os.environ.get("PYJUICE_DISABLE_CUDA_KERNELS", "0") != "1"

# Minimum compute capability: TMA (cp.async.bulk.tensor) requires sm_90+ (Hopper / Blackwell).
_MIN_CC = (9, 0)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy singletons: None = not attempted yet, module = loaded, False = attempted and unavailable.
_module = None          # forward (tlmm)
_attempted = False
_ele_module = None      # element-flow backward
_ele_attempted = False
_par_module = None      # parameter-flow backward
_par_attempted = False
_sbfw_module = None     # small-batch (batch<16) forward
_sbfw_attempted = False
_sbele_module = None    # small-batch (batch<16) element-flow backward
_sbele_attempted = False
_sbpar_module = None    # small-batch (batch<16) parameter-flow backward
_sbpar_attempted = False
_pipe_module = None     # forward (tlmm), warp-specialized
_pipe_attempted = False
_elepipe_module = None  # element-flow backward, warp-specialized
_elepipe_attempted = False
_prepass_module = None  # forward (tlmm), exponentials pre-pass + streaming kernel
_prepass_attempted = False
_eleprepass_module = None  # element-flow backward, exponentials pre-pass + streaming kernel
_eleprepass_attempted = False
_stream_module = None   # forward (tlmm, bf16 params): pre-pass per split range, a split GEMM, a merge
_stream_attempted = False
_elestream_module = None  # element-flow backward (bf16 params): the same three launches
_elestream_attempted = False
_pardef_module = None   # a tied group's deferred parameter flows (fp16 operands): capture + flush
_pardef_attempted = False
_flags_cache = "unset"  # cached (cuda_cflags, ldflags) or None; computed once
_plain_flags_cache = "unset"  # cached plain (no-CUTLASS) cuda_cflags or None


def _find_cutlass_include() -> Optional[str]:
    """Return a CUTLASS include directory containing ``cute/tensor.hpp``, or None."""
    def _ok(inc):
        return inc is not None and os.path.isfile(os.path.join(inc, "cute", "tensor.hpp"))

    # 1. explicit env var (accept either the repo root or its include/ dir)
    for var in ("PYJUICE_CUTLASS_PATH", "CUTLASS_PATH"):
        p = os.environ.get(var)
        if p:
            for cand in (os.path.join(p, "include"), p):
                if _ok(cand):
                    return cand

    # 2. nvidia-cutlass pip package (ships headers under the package dir). The layout has varied across
    # releases: nvidia-cutlass >= 4.x puts them at `cutlass_library/source/include`; older wheels used
    # `<base>/include` or a sibling `include/`. Try all known locations.
    try:
        import cutlass_library  # noqa: F401  (provided by the nvidia-cutlass package)
        base = os.path.dirname(os.path.abspath(cutlass_library.__file__))
        for cand in (os.path.join(base, "source", "include"),  # nvidia-cutlass >= 4.x
                     os.path.join(base, "..", "include"),
                     os.path.join(base, "include")):
            cand = os.path.normpath(cand)
            if _ok(cand):
                return cand
    except Exception:
        pass

    # 3. vendored checkout / git submodule next to this file
    for cand in (os.path.join(_THIS_DIR, "cutlass", "include"),
                 os.path.join(_THIS_DIR, "cutlass")):
        if _ok(cand):
            return cand

    return None


def _compile_flags():
    """Return (cuda_cflags, ldflags) for nvcc, or None if a prerequisite is missing. Computed once
    (warns once) and cached, so multiple kernel modules share the same toolchain probe."""
    global _flags_cache
    if _flags_cache != "unset":
        return _flags_cache
    _flags_cache = None
    if not ENABLE_CUDA_KERNELS or not torch.cuda.is_available():
        return None
    cc = torch.cuda.get_device_capability()
    if cc < _MIN_CC:
        warnings.warn(
            f"pyjuice CUDA sum-layer kernels need compute capability >= {_MIN_CC[0]}.{_MIN_CC[1]} "
            f"(TMA); this GPU is sm_{cc[0]}{cc[1]}. Using the Triton kernels.", RuntimeWarning)
        return None
    cutlass_inc = _find_cutlass_include()
    if cutlass_inc is None:
        warnings.warn(
            "pyjuice CUDA sum-layer fast path is disabled: CUTLASS headers not found. Set "
            "$PYJUICE_CUTLASS_PATH to a CUTLASS checkout (>= 3.5) or `pip install nvidia-cutlass` "
            "to enable it. Falling back to the Triton kernels.", RuntimeWarning)
        return None
    from torch.utils.cpp_extension import CUDA_HOME
    arch = f"sm_{cc[0]}{cc[1]}a"  # arch-specific (the `a` suffix enables TMA & family features)
    cuda_cflags = ["-O3", f"-arch={arch}", "--use_fast_math",
                   "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG", f"-I{cutlass_inc}"]
    # `-lcuda` for the driver API (cuTensorMapEncodeTiled). The stubs dir guarantees the linker finds
    # libcuda even on build hosts without a driver lib on the default search path.
    ldflags = ["-lcuda"]
    if CUDA_HOME:
        ldflags = [f"-L{os.path.join(CUDA_HOME, 'lib64', 'stubs')}"] + ldflags
    _flags_cache = (cuda_cflags, ldflags)
    return _flags_cache


def _plain_compile_flags():
    """Flags for a PLAIN CUDA kernel (no CuTe/TMA/CUTLASS, no driver API). Unlike `_compile_flags`
    this needs neither CUTLASS nor sm_90 -- it works on any CUDA GPU -- so the small-batch forward
    fast path is usable far more broadly than the CuTe kernels. Computed once and cached."""
    global _plain_flags_cache
    if _plain_flags_cache != "unset":
        return _plain_flags_cache
    _plain_flags_cache = None
    if not ENABLE_CUDA_KERNELS or not torch.cuda.is_available():
        return None
    cc = torch.cuda.get_device_capability()
    # Plain arch (no `a` suffix): the kernel uses only ordinary CUDA, so no arch-specific features.
    _plain_flags_cache = ["-O3", f"-arch=sm_{cc[0]}{cc[1]}", "--use_fast_math", "-DNDEBUG"]
    return _plain_flags_cache


def _jit_plain(name: str, source_file: str):
    """JIT-compile a plain-CUDA .cu (no CUTLASS) into a loaded module, or None (warns once)."""
    flags = _plain_compile_flags()
    if flags is None:
        return None
    from pyjuice.utils.cuda_ext import jit_load
    try:
        return jit_load(name, sources=[os.path.join(_THIS_DIR, source_file)],
                    extra_cuda_cflags=flags, verbose=False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA kernel '{name}' failed to compile ({type(e).__name__}: {e}). "
            "Falling back to the Triton kernels.", RuntimeWarning)
        return None


def _jit(name: str, source_file: str):
    """JIT-compile one .cu into a loaded module, or None (warns once on failure)."""
    fl = _compile_flags()
    if fl is None:
        return None
    cuda_cflags, ldflags = fl
    from pyjuice.utils.cuda_ext import jit_load
    try:
        return jit_load(name, sources=[os.path.join(_THIS_DIR, source_file)],
                    extra_cuda_cflags=cuda_cflags, extra_ldflags=ldflags, verbose=False)
    except Exception as e:  # nvcc missing, compile error, CUTLASS too old, etc.
        warnings.warn(
            f"pyjuice CUDA kernel '{name}' failed to compile ({type(e).__name__}: {e}). "
            "Falling back to the Triton kernels.", RuntimeWarning)
        return None


def is_available() -> bool:
    """Whether the CUDA forward fast-path kernel is usable (lazily JIT-compiles once)."""
    global _module, _attempted
    if not _attempted:
        _attempted = True
        _module = _jit("pyjuice_sum_forward_cuda", "tlmm_forward_sum.cu")
    return _module is not None


def ele_is_available() -> bool:
    """Whether the CUDA element-flow backward fast-path kernel is usable (lazily JIT-compiles once)."""
    global _ele_module, _ele_attempted
    if not _ele_attempted:
        _ele_attempted = True
        _ele_module = _jit("pyjuice_sum_backward_ele_cuda", "ele_backward_sum.cu")
    return _ele_module is not None


def par_is_available() -> bool:
    """Whether the CUDA parameter-flow backward fast-path kernel is usable (lazily JIT-compiles once)."""
    global _par_module, _par_attempted
    if not _par_attempted:
        _par_attempted = True
        _par_module = _jit("pyjuice_sum_backward_par_cuda", "par_backward_sum.cu")
    return _par_module is not None


def get_par_deferred_module():
    """The deferred parameter-flow extension (`par_deferred_fp16.cu`: a tied group's capture and flush, see
    `pyjuice.layer.deferred_par`), or None if its toolchain is unavailable (warns once)."""
    global _pardef_module, _pardef_attempted
    if not _pardef_attempted:
        _pardef_attempted = True
        _pardef_module = _jit("pyjuice_sum_par_deferred_cuda", "par_deferred_fp16.cu")
    return _pardef_module


def smallbatch_fw_is_available() -> bool:
    """Whether the small-batch (batch<16) CUDA forward kernel is usable (lazily JIT-compiles once).
    Plain CUDA, so it needs no CUTLASS and no sm_90 -- only nvcc + a CUDA GPU."""
    global _sbfw_module, _sbfw_attempted
    if not _sbfw_attempted:
        _sbfw_attempted = True
        _sbfw_module = _jit_plain("pyjuice_sum_forward_smallbatch_cuda", "smallbatch_forward_sum.cu")
    return _sbfw_module is not None


def smallbatch_forward_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                           nids: torch.Tensor, ebase: torch.Tensor, pbase: torch.Tensor,
                           batch_size: int, block_size: int, num_edges: int, cfg: int = 0) -> None:
    """Small-batch block-sparse sum-layer forward (in-place into ``node_mars``) using SPLIT config
    ``cfg``. Caller must guarantee the dispatch conditions hold (see ``smallbatch_fw_is_available``
    + the small-batch gate in ``sum_layer._forward_block_sparse``): block_size a multiple of 32,
    contiguous children + block_size-strided params, LL propagation."""
    _sbfw_module.smallbatch_forward_sum(node_mars, element_mars, params, nids, ebase, pbase,
                                        int(batch_size), int(block_size), int(num_edges), int(cfg))


_sbfwbt_module = None
_sbfwbt_attempted = False
_sbfwbt_cfg_cache = None


def smallbatch_fw_bt_is_available() -> bool:
    """Whether the batch-tiled small-batch CUDA forward (`smallbatch_forward_sum_bt.cu`: the parameters loaded once
    per BT batch columns, for batch < 64) is usable (lazily JIT-compiles once). Plain CUDA, like the kernel above."""
    global _sbfwbt_module, _sbfwbt_attempted
    if not _sbfwbt_attempted:
        _sbfwbt_attempted = True
        _sbfwbt_module = _jit_plain("pyjuice_sum_forward_smallbatch_bt_cuda", "smallbatch_forward_sum_bt.cu")
    return _sbfwbt_module is not None


def smallbatch_forward_sum_bt(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                              nids: torch.Tensor, ebase: torch.Tensor, pbase: torch.Tensor,
                              batch_size: int, block_size: int, num_edges: int, cfg: int = 0) -> None:
    """The batch-tiled small-batch forward (in-place into ``node_mars``) with config ``cfg`` (an index into
    `smallbatch_fw_bt_configs`). The same dispatch conditions as `smallbatch_forward_sum`; bit-identical to it at the
    same SPLIT, column by column."""
    _sbfwbt_module.smallbatch_forward_sum_bt(node_mars, element_mars, params, nids, ebase, pbase,
                                             int(batch_size), int(block_size), int(num_edges), int(cfg))


def smallbatch_fw_bt_configs():
    """(SPLIT, BT) per config id (index = the ``cfg`` arg). Memoized (constant)."""
    global _sbfwbt_cfg_cache
    if _sbfwbt_cfg_cache is None:
        _sbfwbt_cfg_cache = [(int(c) // 100, int(c) % 100) for c in _sbfwbt_module.smallbatch_fw_bt_configs()] \
            if smallbatch_fw_bt_is_available() else []
    return _sbfwbt_cfg_cache


_sbmma_module = None
_sbmma_attempted = False
_sbmma_cfg_cache = None


def smallbatch_fw_mma_is_available() -> bool:
    """Whether the tensor-core small-batch forward (`smallbatch_forward_sum_mma.cu`, CuTe: a node tile x a batch tile of
    16-64 x an edge range per block, the ranges merged by a dependent launch) is usable (lazily JIT-compiles once)."""
    global _sbmma_module, _sbmma_attempted
    if not _sbmma_attempted:
        _sbmma_attempted = True
        _sbmma_module = _jit("pyjuice_sum_forward_smallbatch_mma_cuda", "smallbatch_forward_sum_mma.cu")
    return _sbmma_module is not None


def smallbatch_forward_sum_mma(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                               nids: torch.Tensor, ebase: torch.Tensor, pbase: torch.Tensor,
                               batch_size: int, block_size: int, num_edges: int, code: int = 0,
                               scratch: torch.Tensor = None) -> None:
    """The tensor-core small-batch forward (in-place into ``node_mars``); ``code`` = config id | split << 8 (an arm of
    `smallbatch_fw_mma_arms`); ``scratch`` a float buffer for a split's partials (`smallbatch_fw_mma_scratch`), reused when
    it is large enough. The small-batch forward's contract (`smallbatch_forward_sum`) plus num_edges % 64 == 0."""
    if scratch is None:
        scratch = _EMPTY_F32.get(node_mars.device)
        if scratch is None:                                  # (never `tensor or ...`: an empty tensor's truth value raises)
            scratch = _EMPTY_F32[node_mars.device] = torch.empty(0, device = node_mars.device)
    _sbmma_module.smallbatch_forward_sum_mma(node_mars, element_mars, params, nids, ebase, pbase, scratch,
                                             int(batch_size), int(block_size), int(num_edges), int(code))


_EMPTY_F32 = {}


def smallbatch_fw_mma_scratch(code: int, batch_size: int, block_size: int, n_blocks: int) -> int:
    """The partials buffer an arm needs, in floats (0 for one range): 2 x split x rows x the batch padded to the tile."""
    split = code >> 8
    if split <= 1:
        return 0
    bn = smallbatch_fw_mma_configs()[code & 255][1]
    return 2 * split * n_blocks * block_size * (((batch_size + bn - 1) // bn) * bn)


def smallbatch_fw_mma_configs():
    """(BM, BN) per config id. Memoized (constant)."""
    global _sbmma_cfg_cache
    if _sbmma_cfg_cache is None:
        _sbmma_cfg_cache = [(int(c[0]), int(c[1])) for c in _sbmma_module.smallbatch_fw_mma_configs()] \
            if smallbatch_fw_mma_is_available() else []
    return _sbmma_cfg_cache


def smallbatch_fw_mma_arms(block_size: int, batch_size: int, num_edges: int, n_blocks: int, grouped: int = 1):
    """The tensor-core small-batch forward's `code`s for a layer shape: every config whose node tile divides the block
    size -- or, ``grouped`` (`row_groups`: aligned runs of that many node blocks share their children), spans at most
    that many whole blocks -- and whose batch tile is not wider than the batch needs (the narrowest tile that covers
    it), with the split-K range counts 1, 2, 4, 8 (at most the edge tiles) whose grid -- node tiles x batch tiles x
    ranges -- stays within four waves of the SMs: a CTA walking its whole row alone is latency-bound (MEASURED HMM-4096
    batch 16: 128 CTAs 14 us of bare stream, 256 CTAs 10). Empty where the kernel does not apply."""
    if num_edges % 64 or block_size % 8 or batch_size >= 64 or not smallbatch_fw_mma_is_available():
        return []
    key = (block_size, batch_size, num_edges, n_blocks, grouped, torch.cuda.current_device())
    arms = _sbmma_arms_cache.get(key)
    if arms is not None:
        return arms
    knt = num_edges // 64
    sms = _sms()                                                         # (cached: a device-properties query per call is host time)
    cover = min(bn for bn in (16, 32, 64) if bn >= batch_size)           # the narrowest tile that takes the batch whole
    arms = []
    for c, (bm, bn) in enumerate(smallbatch_fw_mma_configs()):
        if bn > cover:
            continue
        if block_size % bm:
            g = bm // block_size
            if bm % block_size or g > grouped or n_blocks % g:
                continue
        tiles = n_blocks * block_size // bm * ((batch_size + bn - 1) // bn)
        for s in (1, 2, 4, 8):
            if s > knt or (s > 1 and tiles * s > 4 * sms):
                continue
            arms.append(c | (s << 8))
    _sbmma_arms_cache[key] = arms
    return arms


def smallbatch_fw_mma_rows(code: int, block_size: int) -> int:
    """The node blocks one tile of an arm spans (1 unless grouped): a launch over some rows (`runs`) must cover whole
    multiples of it."""
    bm = smallbatch_fw_mma_configs()[code & 255][0]
    return bm // block_size if block_size < bm else 1


_sbmma_arms_cache = {}


_sbmmae_module = None
_sbmmae_attempted = False
_sbmmae_cfg_cache = None
_sbmmae_arms_cache = {}


def smallbatch_ele_mma_is_available() -> bool:
    """Whether the tensor-core small-batch element backward (`smallbatch_ele_backward_mma.cu`) is usable (lazily
    JIT-compiles once)."""
    global _sbmmae_module, _sbmmae_attempted
    if not _sbmmae_attempted:
        _sbmmae_attempted = True
        _sbmmae_module = _jit("pyjuice_sum_backward_ele_smallbatch_mma_cuda", "smallbatch_ele_backward_mma.cu")
    return _sbmmae_module is not None


def smallbatch_ele_backward_sum_mma(element_flows: torch.Tensor, element_mars: torch.Tensor, node_flows: torch.Tensor,
                                    node_mars: torch.Tensor, params: torch.Tensor, chids: torch.Tensor,
                                    ebase: torch.Tensor, pbase: torch.Tensor, batch_size: int, block_size: int,
                                    cs_block_size: int, num_edges: int, code: int = 0,
                                    scratch: torch.Tensor = None, pbt: int = 1) -> None:
    """The tensor-core small-batch element-flow backward (overwrites ``element_flows``); ``ebase`` the layer's
    [element blocks, num_edges / 64] first parent per tile, ``pbase`` its [element blocks, num_edges / 64 * pbt] parameter
    bases (``pbt`` 2: a tile over two parameter blocks of 32); ``code`` = config id | split << 8
    (`smallbatch_ele_mma_arms`); ``scratch`` a float buffer for a split's partials (`smallbatch_ele_mma_scratch`)."""
    if scratch is None:
        scratch = _EMPTY_F32.get(element_flows.device)
        if scratch is None:
            scratch = _EMPTY_F32[element_flows.device] = torch.empty(0, device = element_flows.device)
    _sbmmae_module.smallbatch_ele_backward_sum_mma(element_flows, element_mars, node_flows, node_mars, params, chids,
                                                   ebase, pbase, scratch, int(batch_size), int(block_size),
                                                   int(cs_block_size), int(num_edges), int(pbt), int(code))


def smallbatch_ele_mma_configs():
    """(BM, BN) per config id. Memoized (constant)."""
    global _sbmmae_cfg_cache
    if _sbmmae_cfg_cache is None:
        _sbmmae_cfg_cache = [(int(c[0]), int(c[1])) for c in _sbmmae_module.smallbatch_ele_mma_configs()] \
            if smallbatch_ele_mma_is_available() else []
    return _sbmmae_cfg_cache


def smallbatch_ele_mma_arms(cs_block_size: int, block_size: int, batch_size: int, num_edges: int, n_blocks: int,
                            grouped: int = 1):
    """The tensor-core small-batch element backward's `code`s for a partition shape: `smallbatch_fw_mma_arms`' rule over
    the element blocks (child tiles dividing the child block size, or -- ``grouped``: aligned runs of that many element
    blocks share their parents -- spanning at most that many whole blocks). Empty where it does not apply."""
    if (num_edges % 64 or cs_block_size % 8 or block_size % 8 or batch_size >= 64
            or not smallbatch_ele_mma_is_available()):
        return []
    key = (cs_block_size, block_size, batch_size, num_edges, n_blocks, grouped, torch.cuda.current_device())
    arms = _sbmmae_arms_cache.get(key)
    if arms is not None:
        return arms
    knt = num_edges // 64
    sms = _sms()
    cover = min(bn for bn in (16, 32, 64) if bn >= batch_size)
    arms = []
    for c, (bm, bn) in enumerate(smallbatch_ele_mma_configs()):
        if bn > cover:
            continue
        if cs_block_size % bm:
            g = bm // cs_block_size
            if bm % cs_block_size or g > grouped or n_blocks % g:
                continue
        tiles = n_blocks * cs_block_size // bm * ((batch_size + bn - 1) // bn)
        for s in (1, 2, 4, 8):
            if s > knt or (s > 1 and tiles * s > 4 * sms):
                continue
            arms.append(c | (s << 8))
    _sbmmae_arms_cache[key] = arms
    return arms


def smallbatch_ele_mma_rows(code: int, cs_block_size: int) -> int:
    """The element blocks one tile of an arm spans (1 unless grouped)."""
    bm = smallbatch_ele_mma_configs()[code & 255][0]
    return bm // cs_block_size if cs_block_size < bm else 1


def smallbatch_ele_mma_scratch(code: int, batch_size: int, cs_block_size: int, n_blocks: int) -> int:
    """The partials buffer an element arm needs, in floats (0 for one range)."""
    split = code >> 8
    if split <= 1:
        return 0
    bn = smallbatch_ele_mma_configs()[code & 255][1]
    return 2 * split * n_blocks * cs_block_size * (((batch_size + bn - 1) // bn) * bn)


_sbfw_cfg_cache = None
def smallbatch_fw_configs():
    """List of SPLIT values per config id (index = the ``cfg`` arg). Memoized (constant) -- the value
    is queried per layer per step in the dispatch, so it must not re-enter the pybind module."""
    global _sbfw_cfg_cache
    if _sbfw_cfg_cache is None:
        _sbfw_cfg_cache = [int(c) for c in _sbfw_module.smallbatch_fw_configs()] if smallbatch_fw_is_available() else []
    return _sbfw_cfg_cache


def smallbatch_ele_is_available() -> bool:
    """Whether the small-batch (batch<16) CUDA element-flow backward kernel is usable (lazily
    JIT-compiles once). Plain CUDA, so it needs no CUTLASS and no sm_90 -- only nvcc + a CUDA GPU."""
    global _sbele_module, _sbele_attempted
    if not _sbele_attempted:
        _sbele_attempted = True
        _sbele_module = _jit_plain("pyjuice_sum_backward_ele_smallbatch_cuda", "smallbatch_ele_backward.cu")
    return _sbele_module is not None


def smallbatch_ele_backward_sum(element_flows: torch.Tensor, element_mars: torch.Tensor,
                                node_flows: torch.Tensor, node_mars: torch.Tensor, params: torch.Tensor,
                                chids: torch.Tensor, ebase: torch.Tensor, pbase: torch.Tensor,
                                batch_size: int, block_size: int, cs_block_size: int, num_edges: int,
                                tile_k: int, cfg: int = 0) -> None:
    """Small-batch block-sparse sum-layer element-flow backward (overwrites ``element_flows``) using
    WARPS config ``cfg``. ``ebase`` / ``pbase`` are the layer's per-tile tables, ``[node blocks, num_edges //
    tile_k]``: a tile's first parent node and first parameter (the CUDA ele kernels' operands). Caller must
    guarantee the dispatch conditions hold (see ``smallbatch_ele_is_available`` + the small-batch gate in
    ``sum_layer._backward_block_sparse_ele_flows``): one parameter-block group per tile (``ptr_inc_step == 1``), LL
    propagation, ``allow_modify_flows`` off."""
    _sbele_module.smallbatch_ele_backward_sum(element_flows, element_mars, node_flows, node_mars, params,
                                              chids, ebase, pbase, int(batch_size), int(block_size),
                                              int(cs_block_size), int(num_edges), int(tile_k), int(cfg))


_sbele_cfg_cache = None
def smallbatch_ele_configs():
    """List of WARPS values per config id (index = the ``cfg`` arg). Memoized (queried per layer/step)."""
    global _sbele_cfg_cache
    if _sbele_cfg_cache is None:
        _sbele_cfg_cache = [int(c) for c in _sbele_module.smallbatch_ele_configs()] if smallbatch_ele_is_available() else []
    return _sbele_cfg_cache


def smallbatch_par_is_available() -> bool:
    """Whether the small-batch (batch<16) CUDA parameter-flow backward kernel is usable (lazily
    JIT-compiles once). Plain CUDA, so it needs no CUTLASS and no sm_90 -- only nvcc + a CUDA GPU."""
    global _sbpar_module, _sbpar_attempted
    if not _sbpar_attempted:
        _sbpar_attempted = True
        _sbpar_module = _jit_plain("pyjuice_sum_backward_par_smallbatch_cuda", "smallbatch_par_backward.cu")
    return _sbpar_module is not None


def smallbatch_par_backward_sum(param_flows: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
                                element_mars: torch.Tensor, params: torch.Tensor, nids: torch.Tensor,
                                cids: torch.Tensor, pids: torch.Tensor, pfids: torch.Tensor,
                                batch_size: int, block_size: int, num_edges: int, cfg: int = 0) -> None:
    """Small-batch sparse sum-layer parameter-flow backward (accumulates into ``param_flows`` via a
    collision-free read-add-store) using EY config ``cfg``. Caller must guarantee the dispatch
    conditions hold (see ``smallbatch_par_is_available`` + the small-batch gate in
    ``sum_layer._backward_sparse_par_flows``): LL, logspace flows, allow_modify_flows / negate off,
    a single batch tile, collision-free (untied) flows, block_size a multiple of 32."""
    _sbpar_module.smallbatch_par_backward_sum(param_flows, node_flows, node_mars, element_mars, params,
                                              nids, cids, pids, pfids, int(batch_size), int(block_size),
                                              int(num_edges), int(cfg))


_sbpar_cfg_cache = None
def smallbatch_par_configs():
    """List of EY values per config id (index = the ``cfg`` arg). Memoized (queried per layer/step)."""
    global _sbpar_cfg_cache
    if _sbpar_cfg_cache is None:
        _sbpar_cfg_cache = [int(c) for c in _sbpar_module.smallbatch_par_configs()] if smallbatch_par_is_available() else []
    return _sbpar_cfg_cache


def par_backward_sum(param_flows: torch.Tensor, node_flows: torch.Tensor, node_mars: torch.Tensor,
                     element_mars: torch.Tensor, params: torch.Tensor, nbase: torch.Tensor,
                     cbase: torch.Tensor, pbase: torch.Tensor, fbase: torch.Tensor,
                     batch_size: int, block_size: int, num_edges: int, mode: int = 0) -> None:
    """Block-sparse sum-layer parameter-flow backward (accumulates into ``param_flows``). ``mode``:
    0 = read-add-store (RMW, collision-free); 1 = atomicAdd (tied params); 2 = store-only (valid only
    when ``param_flows`` is freshly zeroed). Caller guarantees the dispatch conditions (see
    ``par_is_available`` + the gate in ``sum_layer._backward_block_sparse_par_flows``)."""
    _par_module.par_backward_sum(param_flows, node_flows, node_mars, element_mars, params,
                                 nbase, cbase, pbase, fbase, int(batch_size), int(block_size),
                                 int(num_edges), int(mode))


def ele_backward_sum(element_flows: torch.Tensor, element_mars: torch.Tensor, node_flows: torch.Tensor,
                     node_mars: torch.Tensor, params: torch.Tensor, chids: torch.Tensor,
                     ebase: torch.Tensor, pbase: torch.Tensor,
                     batch_size: int, block_size_k: int, block_size_m: int, k_num_tiles: int) -> None:
    """Block-sparse sum-layer element-flow backward (in-place into ``element_flows``). Caller must
    guarantee the dispatch conditions hold (see ``ele_is_available`` + the gate in
    ``sum_layer._backward_block_sparse``)."""
    _ele_module.ele_backward_sum(element_flows, element_mars, node_flows, node_mars, params, chids,
                                 ebase, pbase, int(batch_size), int(block_size_k), int(block_size_m),
                                 int(k_num_tiles))


def tlmm_forward_sum(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                     nids: torch.Tensor, ebase: torch.Tensor, pbase: torch.Tensor,
                     batch_size: int, block_size: int, k_num_tiles: int, cfg: int = 0) -> None:
    """Block-sparse sum-layer forward (in-place into ``node_mars``) using tile config ``cfg``.
    Caller must guarantee the dispatch conditions hold (see ``is_available`` + the gate in
    ``sum_layer._forward_block_sparse``)."""
    _module.tlmm_forward_sum(node_mars, element_mars, params, nids, ebase, pbase,
                             int(batch_size), int(block_size), int(k_num_tiles), int(cfg))


def configs():
    """List of (BM, BN) tile shapes per config id (index = the ``cfg`` arg)."""
    return [tuple(c) for c in _module.configs()] if is_available() else []


def valid_configs(block_size: int, batch_size: int):
    """Config ids whose tile shape divides this layer's (block_size, batch)."""
    return [i for i, (bm, bn) in enumerate(configs())
            if block_size % bm == 0 and batch_size % bn == 0]


def pipe_is_available() -> bool:
    """Whether the warp-specialized CUDA tlmm forward (tlmm_forward_sum_pipelined.cu) is usable (lazily JIT-compiles
    once). CuTe, so it needs CUTLASS; no TMA, but it shares the CuTe kernels' toolchain gate."""
    global _pipe_module, _pipe_attempted
    if not _pipe_attempted:
        _pipe_attempted = True
        _pipe_module = _jit("pyjuice_sum_forward_pipelined_cuda", "tlmm_forward_sum_pipelined.cu")
    return _pipe_module is not None


def tlmm_forward_sum_pipelined(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                               nids: torch.Tensor, ebase: torch.Tensor, pbase: torch.Tensor,
                               batch_size: int, block_size: int, k_num_tiles: int, cfg: int = 0) -> None:
    """The tlmm forward with producer warps streaming the operands and consumer warps computing (in-place into
    ``node_mars``, bit-identical to `tlmm_forward_sum`), with tile config ``cfg`` (`pipe_configs`). Caller must
    guarantee the plain kernel's dispatch conditions (the gate in ``sum_layer._forward_block_sparse``)."""
    _pipe_module.tlmm_forward_sum_pipelined(node_mars, element_mars, params, nids, ebase, pbase,
                                            int(batch_size), int(block_size), int(k_num_tiles), int(cfg))


def pipe_configs():
    """List of (BM, BN) tile shapes per config id of the warp-specialized forward kernel."""
    return [tuple(c) for c in _pipe_module.configs()] if pipe_is_available() else []


def pipe_valid_configs(block_size: int, batch_size: int):
    """The warp-specialized forward kernel's config ids whose tile shape divides this layer's (block_size, batch)."""
    return [i for i, (bm, bn) in enumerate(pipe_configs())
            if block_size % bm == 0 and batch_size % bn == 0]


def prepass_is_available() -> bool:
    """Whether the pre-pass tlmm forward (tlmm_forward_sum_prepass.cu) is usable (lazily JIT-compiles once). CuTe, so it
    needs CUTLASS; no TMA, but it shares the CuTe kernels' toolchain gate."""
    global _prepass_module, _prepass_attempted
    if not _prepass_attempted:
        _prepass_attempted = True
        _prepass_module = _jit("pyjuice_sum_forward_prepass_cuda", "tlmm_forward_sum_prepass.cu")
    return _prepass_module is not None


def tlmm_forward_sum_prepass(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                             nids: torch.Tensor, ebase: torch.Tensor, tix: torch.Tensor, pbase: torch.Tensor,
                             batch_size: int, block_size: int, k_num_tiles: int, cfg: int = 0,
                             shared: bool = False) -> None:
    """The tlmm forward as a pre-pass over the element tiles (``ebase``: their first rows; their maxima and exponentials
    once each) and a streaming kernel over the parameter tiles (in-place into ``node_mars``, bit-identical to
    `tlmm_forward_sum`), with tile config ``cfg`` (`prepass_configs`). ``shared``: node blocks share tiles, ``tix``
    (int32 [node blocks, edge tiles]) saying which each entry is; else ``ebase`` holds every entry's own tile in order
    (`tile_index` gives all three). Caller must guarantee the plain kernel's dispatch conditions but the tile shape
    (the gate in ``sum_layer._forward_block_sparse``)."""
    _prepass_module.tlmm_forward_sum_prepass(node_mars, element_mars, params, nids, ebase, tix, pbase,
                                             int(batch_size), int(block_size), int(k_num_tiles), int(cfg), bool(shared))


#: Blocks sharing a tile before a pre-pass kernel shares it among them (`tile_index`). MEASURED (HMM-4096 B128, 4 blocks
#: of 1024 sharing their children): the stream kernels' read of the table -- prefetched a tile ahead, the same register
#: count -- costs the fp32 element kernel 45 -> 51.5 us a layer, the bf16 one ~1%, against ~2 us of pre-pass saved at
#: 4:1; at 32:1 (an HCLT of 32-node blocks) the pre-pass over every entry would be 369 MB of exponentials a layer.
TILE_SHARING_MIN = 8


def tile_index(ebase: torch.Tensor):
    """``(tiles, tix, shared)`` for a pre-pass kernel: the tiles its pre-pass covers (a tile's first row each); int32 of
    ``ebase``'s shape ([blocks, tiles]: a tile's first row per block and tile), which of them each entry is; and
    whether the blocks share tiles. Where at least `TILE_SHARING_MIN` blocks share a tile (an HCLT's 32 node blocks of
    a variable share their children), the tiles are ``ebase``'s distinct values and the blocks share their pre-pass;
    else every entry is its own tile, in order, and the kernel never reads the table."""
    distinct, inverse = torch.unique(ebase.reshape(-1), return_inverse = True)
    if distinct.numel() * TILE_SHARING_MIN > ebase.numel():
        return (ebase.reshape(-1).contiguous(),
                torch.arange(ebase.numel(), dtype = torch.int32, device = ebase.device).view(ebase.shape), False)
    return distinct.contiguous(), inverse.to(torch.int32).view(ebase.shape).contiguous(), True


def prepass_configs():
    """List of (BM, BN) tile shapes per config id of the pre-pass forward kernel."""
    return [tuple(c) for c in _prepass_module.configs()] if prepass_is_available() else []


def prepass_valid_configs(block_size: int, batch_size: int):
    """The pre-pass forward kernel's config ids whose tile shape divides this layer's (block_size, batch)."""
    return [i for i, (bm, bn) in enumerate(prepass_configs())
            if block_size % bm == 0 and batch_size % bn == 0]


#: Split-K ranges a pre-pass kernel may cut a row's edge tiles into (`prepass_arms`): the kernels' MAX_SPLIT
MAX_SPLIT = 8
_SMS = {}


def _sms() -> int:
    """The current device's SM count (cached)."""
    dev = torch.cuda.current_device()
    n = _SMS.get(dev)
    if n is None:
        n = _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


def _split_arms(configs, valid, rows, batch_size, k_num_tiles):
    """`valid`'s config ids, then `id | split << 8` for the splits 2, 4, 8 (at most the tiles) where a config's grid --
    `rows` tile rows x batch tiles x split -- stays under three waves of the SMs: a CTA walking its whole row of tiles
    alone on an SM is latency-bound (MEASURED HMM-4096 B128 bf16: 128 CTAs on 188 SMs, the forward at a quarter of its
    compute floor); more ranges, resident together, fill the machine, the merge is a thread per (row, column)."""
    arms = list(valid)
    for s in (2, 4, 8):
        if s > k_num_tiles:
            break
        for c in valid:
            bm, bn = configs[c]
            if (rows // bm) * (batch_size // bn) * s <= 3 * _sms():
                arms.append(c | (s << 8))
    return arms


def prepass_arms(block_size: int, batch_size: int, k_num_tiles: int, n_blocks: int):
    """The pre-pass forward's tuner arms for a layer: its valid configs, and their splits where the grid is small
    (`_split_arms`); an arm is the kernel's `cfg` code."""
    return _split_arms(prepass_configs(), prepass_valid_configs(block_size, batch_size), n_blocks * block_size,
                       batch_size, k_num_tiles)


def ele_pipe_is_available() -> bool:
    """Whether the warp-specialized CUDA element-flow backward (ele_backward_sum_pipelined.cu) is usable (lazily
    JIT-compiles once). CuTe, so it needs CUTLASS; no TMA, but it shares the CuTe kernels' toolchain gate."""
    global _elepipe_module, _elepipe_attempted
    if not _elepipe_attempted:
        _elepipe_attempted = True
        _elepipe_module = _jit("pyjuice_sum_backward_ele_pipelined_cuda", "ele_backward_sum_pipelined.cu")
    return _elepipe_module is not None


def ele_backward_sum_pipelined(element_flows: torch.Tensor, element_mars: torch.Tensor, node_flows: torch.Tensor,
                               node_mars: torch.Tensor, params: torch.Tensor, chids: torch.Tensor,
                               ebase: torch.Tensor, pbase: torch.Tensor,
                               batch_size: int, block_size: int, cs_block_size: int, k_num_tiles: int) -> None:
    """The element-flow backward with producer warps streaming the operands and consumer warps computing (in-place
    into ``element_flows``, bit-identical to `ele_backward_sum`). Caller must guarantee the plain kernel's dispatch
    conditions (the gate in ``sum_layer._backward_block_sparse_ele_flows``)."""
    _elepipe_module.ele_backward_sum_pipelined(element_flows, element_mars, node_flows, node_mars, params, chids,
                                               ebase, pbase, int(batch_size), int(block_size), int(cs_block_size),
                                               int(k_num_tiles))


def ele_prepass_is_available() -> bool:
    """Whether the pre-pass element-flow backward (ele_backward_sum_prepass.cu) is usable (lazily JIT-compiles once). CuTe,
    so it needs CUTLASS; no TMA, but it shares the CuTe kernels' toolchain gate."""
    global _eleprepass_module, _eleprepass_attempted
    if not _eleprepass_attempted:
        _eleprepass_attempted = True
        _eleprepass_module = _jit("pyjuice_sum_backward_ele_prepass_cuda", "ele_backward_sum_prepass.cu")
    return _eleprepass_module is not None


def ele_backward_sum_prepass(element_flows: torch.Tensor, element_mars: torch.Tensor, node_flows: torch.Tensor,
                             node_mars: torch.Tensor, params: torch.Tensor, chids: torch.Tensor,
                             ebase: torch.Tensor, tix: torch.Tensor, pbase: torch.Tensor,
                             batch_size: int, block_size: int, cs_block_size: int, k_num_tiles: int, cfg: int = 0,
                             shared: bool = False) -> None:
    """The element-flow backward as a pre-pass over the parent tiles (``ebase``: their first rows; their log ratios'
    maxima and exponentials once each) and a streaming kernel over the parameter tiles (in-place into
    ``element_flows``, bit-identical to `ele_backward_sum`), with tile config ``cfg`` (`ele_prepass_configs`).
    ``shared``: element blocks share tiles, ``tix`` (int32 [element blocks, parent tiles]) saying which each entry is;
    else ``ebase`` holds every entry's own tile in order (`tile_index` gives all three). ``pbase``: int64 [element
    blocks, parent tiles x parameter blocks a tile] (two where the parents' block size is 32). Caller must guarantee
    the plain kernel's dispatch conditions but the tile shape (the gate in
    ``sum_layer._backward_block_sparse_ele_flows``)."""
    _eleprepass_module.ele_backward_sum_prepass(element_flows, element_mars, node_flows, node_mars, params, chids,
                                                ebase, tix, pbase, int(batch_size), int(block_size), int(cs_block_size),
                                                int(k_num_tiles), int(cfg), bool(shared))


def ele_prepass_configs():
    """List of (BM, BN) tile shapes per config id of the pre-pass element-flow backward."""
    return [tuple(c) for c in _eleprepass_module.configs()] if ele_prepass_is_available() else []


def ele_prepass_valid_configs(cs_block_size: int, batch_size: int):
    """The pre-pass element-flow backward's config ids whose tile shape divides a partition's (child block size,
    batch)."""
    return [i for i, (bm, bn) in enumerate(ele_prepass_configs())
            if cs_block_size % bm == 0 and batch_size % bn == 0]


def ele_prepass_arms(cs_block_size: int, batch_size: int, k_num_tiles: int, n_blocks: int):
    """The pre-pass element backward's tuner arms for a partition: its valid configs, and their splits where the grid
    is small (`_split_arms`); an arm is the kernel's `cfg` code."""
    return _split_arms(ele_prepass_configs(), ele_prepass_valid_configs(cs_block_size, batch_size),
                       n_blocks * cs_block_size, batch_size, k_num_tiles)


def stream_is_available() -> bool:
    """Whether the stream tlmm forward (tlmm_forward_sum_stream.cu: a pre-pass per split range, one split GEMM of
    mbarrier'd cp.async stages, a merge; bf16 parameters) is usable (lazily JIT-compiles once)."""
    global _stream_module, _stream_attempted
    if not _stream_attempted:
        _stream_attempted = True
        _stream_module = _jit("pyjuice_sum_forward_stream_cuda", "tlmm_forward_sum_stream.cu")
    return _stream_module is not None


def tlmm_forward_sum_stream(node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                            nids: torch.Tensor, ebase: torch.Tensor, pbase: torch.Tensor, groups,
                            batch_size: int, block_size: int, k_num_tiles: int, cfg: int) -> None:
    """The tlmm forward of a layer with bf16 parameters as a pre-pass per split range (the range's children's maxima
    and bf16 exponentials, shared by the node blocks of equal children: ``groups`` = `row_groups(ebase)`), a split GEMM
    and a merge, in-place into ``node_mars``; ``cfg`` = config id * 16 + split (`stream_arms`). Numerically equivalent
    to `tlmm_forward_sum` (the same bf16 operands under another stabilizer), not to its bits. Caller must guarantee the
    plain kernel's dispatch conditions but the tile shape (the gate in ``sum_layer._forward_block_sparse``)."""
    _stream_module.tlmm_forward_sum_stream(node_mars, element_mars, params, nids, ebase, pbase, groups[0], groups[1],
                                           int(batch_size), int(block_size), int(k_num_tiles), int(cfg))


def stream_configs():
    """List of (BM, BN, KT) per config id of the stream forward."""
    return [tuple(c) for c in _stream_module.configs()] if stream_is_available() else []


def stream_arms(block_size: int, batch_size: int, k_num_tiles: int, n_blocks: int, grouped: int = 1):
    """The stream forward's tuner arms for a layer (`cfg` codes = config id * 16 + split): every fitting config at
    split 1, and at the splits (2 .. 8, within the k-steps) whose GEMM grid fills one wave of the SMs -- half to all of
    them. ``grouped`` (`row_groups`): the node tiles may span that many consecutive blocks of one group, so tiles up
    to ``grouped`` x the block size fit (an HCLT's 32-node blocks: 4 of a variable in a 128-row tile).
    MEASURED (HMM-4096 B128 bf16, 32 node tiles of 128 x 128): splits 4 and 5 (128 / 160 blocks) 36 / 33 us, split 6
    (192 blocks: a second wave) 44, split 2 55; the GEMM runs one block an SM."""
    if not stream_is_available():
        return []
    cfgs = stream_configs()
    fits = [int(c) for c in _stream_module.fitting_configs(int(block_size), int(batch_size), grouped > 1)
            if cfgs[int(c)][0] <= block_size * grouped]
    arms = [c * 16 + 1 for c in fits]
    sms = _sms()
    for c in fits:
        bm, bn, kt = cfgs[c]
        ctas = (n_blocks * block_size // bm) * (batch_size // bn)
        for s in range(2, STREAM_MAX_SPLIT + 1):
            if s <= k_num_tiles * (64 // kt) and sms // 2 <= ctas * s <= sms:
                arms.append(c * 16 + s)
    return arms


#: Split-K ranges the stream forward may cut a row's k-steps into (its kernel's MAX_SPLIT; the 256-row tile needs more
#: ranges than the 128-row one to fill a wave)
STREAM_MAX_SPLIT = 12


def ele_stream_is_available() -> bool:
    """Whether the stream element-flow backward (ele_backward_sum_stream.cu; bf16 parameters) is usable (lazily
    JIT-compiles once)."""
    global _elestream_module, _elestream_attempted
    if not _elestream_attempted:
        _elestream_attempted = True
        _elestream_module = _jit("pyjuice_sum_backward_ele_stream_cuda", "ele_backward_sum_stream.cu")
    return _elestream_module is not None


def ele_backward_sum_stream(element_flows: torch.Tensor, element_mars: torch.Tensor, node_flows: torch.Tensor,
                            node_mars: torch.Tensor, params: torch.Tensor, chids: torch.Tensor, ebase: torch.Tensor,
                            pbase: torch.Tensor, groups, batch_size: int, block_size: int, cs_block_size: int,
                            k_num_tiles: int, cfg: int) -> None:
    """The element-flow backward of a partition with bf16 parameters as a pre-pass per split range (the range's
    parents' log ratios' maxima and bf16 exponentials, shared by the element blocks of equal parents: ``groups`` =
    `row_groups(ebase)`), a split GEMM and a merge, in-place into ``element_flows``; ``pbase`` the per-tile parameter
    offsets of `_cached_bk_ele_pre` ([neb, knt x blocks a tile]); ``cfg`` = config id * 16 + split (`ele_stream_arms`).
    Numerically equivalent to `ele_backward_sum` (the stream forward's bf16 exponentials, under another stabilizer),
    not to its bits. Caller must guarantee the plain kernel's dispatch conditions but the tile shape."""
    _elestream_module.ele_backward_sum_stream(element_flows, element_mars, node_flows, node_mars, params, chids, ebase,
                                              pbase, groups[0], groups[1], int(batch_size), int(block_size),
                                              int(cs_block_size), int(k_num_tiles), int(cfg))


def ele_stream_configs():
    """List of (BM, BN, KT) per config id of the stream element backward."""
    return [tuple(c) for c in _elestream_module.configs()] if ele_stream_is_available() else []


def ele_stream_arms(cs_block_size: int, batch_size: int, block_size: int, k_num_tiles: int, n_blocks: int,
                    grouped: int = 1):
    """The stream element backward's tuner arms for a partition (`cfg` codes = config id * 16 + split): every fitting
    config at split 1, and at the splits whose GEMM grid fills one wave of the SMs (`stream_arms`' rule); ``grouped``
    (`row_groups`): the child tiles may span that many consecutive blocks of one group."""
    if not ele_stream_is_available():
        return []
    cfgs = ele_stream_configs()
    fits = [int(c) for c in _elestream_module.fitting_configs(int(cs_block_size), int(batch_size), int(block_size),
                                                              grouped > 1) if cfgs[int(c)][0] <= cs_block_size * grouped]
    arms = [c * 16 + 1 for c in fits]
    sms = _sms()
    for c in fits:
        bm, bn, kt = cfgs[c]
        ctas = (n_blocks * cs_block_size // bm) * (batch_size // bn)
        for s in range(2, 9):
            if s <= k_num_tiles * (64 // kt) and sms // 2 <= ctas * s <= sms:
                arms.append(c * 16 + s)
    return arms


def _stream_fits(cfgs, code: int, block_rows: int, n_blocks: int, grouped: int) -> bool:
    bm = cfgs[code // 16][0]
    return block_rows % bm == 0 or (bm % block_rows == 0 and bm <= block_rows * grouped
                                    and (n_blocks * block_rows) % bm == 0)


def stream_run_fits(code: int, block_size: int, n_blocks: int, grouped: int) -> bool:
    """Whether the stream forward's arm ``code`` serves a launch over ``n_blocks`` node blocks whose tiles may span
    ``grouped`` consecutive blocks (`row_groups` of the launch's own rows of ebase): the tile's rows divide the block,
    or whole tiles of same-group blocks fill the launch. A run of `partition_runs` may hold fewer blocks than the
    partition's grouped tiles need (an HCLT's 32-node blocks under a 128-row tile): such a run takes another kernel."""
    return _stream_fits(stream_configs(), code, block_size, n_blocks, grouped)


def ele_stream_run_fits(code: int, cs_block_size: int, n_blocks: int, grouped: int) -> bool:
    """`stream_run_fits` for the stream element backward: ``n_blocks`` child blocks of ``cs_block_size`` rows."""
    return _stream_fits(ele_stream_configs(), code, cs_block_size, n_blocks, grouped)


def row_groups(ebase: torch.Tensor):
    """``(ugrp, grep, grouped)`` for the stream forward: int32 per node block its group -- node blocks whose rows of
    ``ebase`` (an edge tile's first child row per tile) are equal share the pre-pass --, int32 per group its first
    block, and how many consecutive blocks a node tile may span: the largest power of two g (1, 2, 4) such that every
    aligned run of g blocks is one group (an HCLT's blocks of a variable are consecutive)."""
    distinct, inverse = torch.unique(ebase, dim = 0, return_inverse = True)
    first = torch.full((distinct.size(0),), ebase.size(0), dtype = torch.long, device = ebase.device)
    first.scatter_reduce_(0, inverse, torch.arange(ebase.size(0), device = ebase.device), reduce = "amin")
    grouped = 1
    for g in (2, 4):
        n = inverse.numel()
        if n % g or not bool((inverse.view(-1, g) == inverse.view(-1, g)[:, :1]).all()):
            break
        grouped = g
    return inverse.to(torch.int32).contiguous(), first.to(torch.int32).contiguous(), grouped


def autotune(candidates, warmup: int = 3, reps: int = 7) -> int:
    """Benchmark each candidate ``(key, run_fn)`` (a no-arg callable that launches one variant) and
    return the ``key`` of the fastest. ``run_fn`` may write into scratch buffers; only timing matters.
    A candidate whose ``run_fn`` raises is skipped. Used to pick per-layer the best of {CUDA configs}
    ∪ {Triton}. Run once per layer signature; the choice is cached by the caller."""
    best_key, best_t = None, float("inf")
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for key, run_fn in candidates:
        try:
            for _ in range(warmup):
                run_fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                ev0.record(); run_fn(); ev1.record(); torch.cuda.synchronize()
                ts.append(ev0.elapsed_time(ev1))
            ts.sort()
            t = ts[len(ts) // 2]
        except Exception:
            continue
        if t < best_t:
            best_key, best_t = key, t
    return best_key
