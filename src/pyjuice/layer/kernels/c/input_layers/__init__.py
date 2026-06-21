"""Optional plain-CUDA fast-path kernels for input layers (per-distribution). Like the sum-layer
kernels in the parent package, these JIT-compile on first use and disable themselves (falling back to
the Triton kernels) if anything required is missing. Unlike the sum kernels these need NO CUTLASS/TMA —
they are plain float4 bandwidth code, so they compile for any CUDA GPU.

Each distribution that has a CUDA fast path exposes it via ``Distribution.get_bk_flow_cuda_fn`` (etc.);
the input layer calls it only when its dispatch gate holds, else uses Triton.
"""

import os
import warnings

import torch

ENABLE_CUDA_KERNELS = os.environ.get("PYJUICE_DISABLE_CUDA_KERNELS", "0") != "1"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_cat_module = None       # Categorical backward (smem-histogram)
_cat_attempted = False


def _jit_plain(name: str, source_file: str):
    """JIT-compile a plain-CUDA kernel (no CUTLASS) for the GPU present, or None (warns once)."""
    if not ENABLE_CUDA_KERNELS or not torch.cuda.is_available():
        return None
    cc = torch.cuda.get_device_capability()
    cuda_cflags = ["-O3", f"-arch=sm_{cc[0]}{cc[1]}a", "--use_fast_math", "-DNDEBUG"]
    from torch.utils.cpp_extension import load
    try:
        return load(name=name, sources=[os.path.join(_THIS_DIR, source_file)],
                    extra_cuda_cflags=cuda_cflags, verbose=False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA input-layer kernel '{name}' failed to compile ({type(e).__name__}: {e}). "
            "Falling back to the Triton kernels.", RuntimeWarning)
        return None


def cat_backward_is_available() -> bool:
    """Whether the CUDA Categorical-backward fast-path kernel is usable (lazily JIT-compiles once)."""
    global _cat_module, _cat_attempted
    if not _cat_attempted:
        _cat_attempted = True
        _cat_module = _jit_plain("pyjuice_cat_backward_cuda", "cat_backward.cu")
    return _cat_module is not None


def cat_backward(param_flows: torch.Tensor, node_flows: torch.Tensor, data: torch.Tensor,
                 vids: torch.Tensor, s_pfids: torch.Tensor, layer_num_nodes: int, batch_size: int,
                 node_offset: int, num_cats: int, logspace: bool) -> None:
    """Categorical input-layer backward: param_flows[node, cat] += sum_b (exp if logspace) node_flows
    where data[vid[node], b] == cat. The flush is NON-ATOMIC, so the caller must guarantee distinct
    (untied), 4-aligned ``s_pfids`` and num_cats/batch divisible by 4 (see the gate in
    ``input_layer.backward``)."""
    _cat_module.cat_backward(param_flows, node_flows, data, vids, s_pfids, int(layer_num_nodes),
                             int(batch_size), int(node_offset), int(num_cats), int(logspace))
