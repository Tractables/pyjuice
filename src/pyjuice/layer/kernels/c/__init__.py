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

# Lazy singletons: None = not attempted yet, False = attempted and unavailable, module = loaded.
_module = None
_attempted = False


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

    # 2. nvidia-cutlass pip package (ships headers under the package dir)
    try:
        import cutlass_library  # noqa: F401  (provided by the nvidia-cutlass package)
        base = os.path.dirname(os.path.abspath(cutlass_library.__file__))
        for cand in (os.path.join(base, "..", "include"), os.path.join(base, "include")):
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


def _build():
    """Attempt the JIT build. Returns the loaded module or None (and warns once on failure)."""
    if not ENABLE_CUDA_KERNELS:
        return None
    if not torch.cuda.is_available():
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

    from torch.utils.cpp_extension import load, CUDA_HOME

    arch = f"sm_{cc[0]}{cc[1]}a"  # arch-specific (the `a` suffix enables TMA & family features)
    cuda_cflags = [
        "-O3", f"-arch={arch}", "--use_fast_math",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "-DNDEBUG",
        f"-I{cutlass_inc}",
    ]
    # `-lcuda` for the driver API (cuTensorMapEncodeTiled). The stubs dir guarantees the linker
    # finds libcuda even on build hosts without a driver lib on the default search path.
    ldflags = ["-lcuda"]
    if CUDA_HOME:
        ldflags = [f"-L{os.path.join(CUDA_HOME, 'lib64', 'stubs')}"] + ldflags

    try:
        mod = load(
            name = "pyjuice_sum_forward_cuda",
            sources = [os.path.join(_THIS_DIR, "tlmm_forward_sum.cu")],
            extra_cuda_cflags = cuda_cflags,
            extra_ldflags = ldflags,
            verbose = False,
        )
        return mod
    except Exception as e:  # nvcc missing, compile error, CUTLASS too old, etc.
        warnings.warn(
            f"pyjuice CUDA sum-layer kernel failed to compile ({type(e).__name__}: {e}). "
            "Falling back to the Triton kernels.", RuntimeWarning)
        return None


def is_available() -> bool:
    """Whether the CUDA fast-path kernels are usable (lazily attempts the JIT build once)."""
    global _module, _attempted
    if not _attempted:
        _attempted = True
        _module = _build()
    return _module is not None


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
