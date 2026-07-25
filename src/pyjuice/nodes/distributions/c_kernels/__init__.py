"""Optional plain-CUDA fast-path kernels for input distributions.

Mirrors ``pyjuice.layer.kernels.c.input_layers``: each kernel JIT-compiles on first use and disables
itself -- falling back to the Triton implementation -- if anything required is missing (no CUDA, no
compiler, no ninja). Nothing here is required for correctness; it is purely a faster path for shapes
where it has been measured to win.
"""

import os
import warnings

import torch

ENABLE_CUDA_KERNELS = os.environ.get("PYJUICE_DISABLE_CUDA_KERNELS", "0") != "1"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_dense_module = None
_dense_attempted = False


def _jit_plain(name: str, source_file: str):
    """JIT-compile a plain-CUDA kernel (no CUTLASS) for the GPU present, or None (warns once)."""
    if not ENABLE_CUDA_KERNELS or not torch.cuda.is_available():
        return None
    cc = torch.cuda.get_device_capability()
    cuda_cflags = ["-O3", f"-arch=sm_{cc[0]}{cc[1]}", "--use_fast_math", "-DNDEBUG"]
    from torch.utils.cpp_extension import load
    try:
        return load(name = name, sources = [os.path.join(_THIS_DIR, source_file)],
                    extra_cuda_cflags = cuda_cflags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA input-distribution kernel '{name}' failed to compile "
            f"({type(e).__name__}: {e}). Falling back to the Triton kernel.", RuntimeWarning)
        return None


def dense_expected_flow_available() -> bool:
    """Whether the CUDA expected-category-flow kernel is usable (lazily JIT-compiles once)."""
    global _dense_module, _dense_attempted
    if not _dense_attempted:
        _dense_attempted = True
        _dense_module = _jit_plain("pyjuice_dense_expected_flow", "dense_expected_flow.cu")
    return _dense_module is not None


def dense_expected_flow(params, param_flows, ratio, uniq, ref_slot, ref_pt, ref_goff, ref_cnt,
                        num_uniq, pf_base, p_base, grad, num_latents, tot_num_cats, uniq_stride,
                        max_refs, num_blocks, block_c, num_slots, tl_size = 8):
    _dense_module.dense_expected_flow(
        params, param_flows, ratio, uniq, ref_slot, ref_pt, ref_goff, ref_cnt, num_uniq,
        pf_base, p_base, grad, int(num_latents), int(tot_num_cats), int(uniq_stride),
        int(max_refs), int(num_blocks), int(block_c), int(num_slots), int(tl_size))
