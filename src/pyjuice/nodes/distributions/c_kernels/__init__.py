"""Optional plain-CUDA fast-path kernels for input distributions.

One source file per distribution, named after its Python module (e.g. `softevi_categorical.cu` holds
every CUDA kernel for `SoftEvidenceCategorical` in `softevi_categorical.py`).

Mirrors ``pyjuice.layer.kernels.c.input_layers``: each kernel JIT-compiles on first use and disables
itself -- falling back to the Triton implementation -- if anything required is missing (no CUDA, no
compiler, no ninja). Nothing here is required for correctness; it is purely a faster path for shapes
where it has been measured to win.
"""

import os
import warnings

import torch

# Deliberately a DIFFERENT switch from the layer kernels' PYJUICE_DISABLE_CUDA_KERNELS: those need a
# CUTLASS checkout for part of their set, so a user without one may well want them off while keeping
# these (which are plain CUDA and compile anywhere).
ENABLE_CUDA_KERNELS = os.environ.get("PYJUICE_DISABLE_DIST_CUDA_KERNELS", "0") != "1"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_softevi_module = None
_softevi_attempted = False


def _jit_plain(name: str, source_file: str):
    """JIT-compile a plain-CUDA kernel (no CUTLASS) for the GPU present, or None (warns once)."""
    if not ENABLE_CUDA_KERNELS or not torch.cuda.is_available():
        return None
    cc = torch.cuda.get_device_capability()
    cuda_cflags = ["-O3", f"-arch=sm_{cc[0]}{cc[1]}", "--use_fast_math", "-DNDEBUG"]
    from pyjuice.utils.cuda_ext import jit_load
    try:
        return jit_load(name, [os.path.join(_THIS_DIR, source_file)],
                        extra_cuda_cflags = cuda_cflags, verbose = False)
    except Exception as e:
        warnings.warn(
            f"pyjuice CUDA input-distribution kernel '{name}' failed to compile "
            f"({type(e).__name__}: {e}). Falling back to the Triton kernel.", RuntimeWarning)
        return None


def dense_expected_flow_available() -> bool:
    """Whether the CUDA expected-category-flow kernel is usable (lazily JIT-compiles once)."""
    global _softevi_module, _softevi_attempted
    if not _softevi_attempted:
        _softevi_attempted = True
        _softevi_module = _jit_plain("pyjuice_softevi_categorical", "softevi_categorical.cu")
    return _softevi_module is not None


def dense_expected_flow(params, param_flows, ratio, uniq, ref_slot, ref_pt, ref_goff, ref_cnt,
                        num_uniq, pf_base, p_base, grad, num_latents, tot_num_cats, uniq_stride,
                        max_refs, num_blocks, block_c, num_slots, tl_size = 8):
    _softevi_module.dense_expected_flow(
        params, param_flows, ratio, uniq, ref_slot, ref_pt, ref_goff, ref_cnt, num_uniq,
        pf_base, p_base, grad, int(num_latents), int(tot_num_cats), int(uniq_stride),
        int(max_refs), int(num_blocks), int(block_c), int(num_slots), int(tl_size))


def softevi_forward(params, node_mars, data, vids, s_pids, var_idmapping, pt, cat_ids,
                    layer_num_nodes, batch_size, node_offset, num_cats, ext_num_vars,
                    unroll):
    _softevi_module.softevi_forward(
        params, node_mars, data, vids, s_pids, var_idmapping, pt, cat_ids,
        int(layer_num_nodes), int(batch_size), int(node_offset), int(num_cats),
        int(ext_num_vars), int(unroll))


def softevi_forward_dense(params, node_mars, Z, log_ex_p, data, vids, s_pids, nids, var_idmapping,
                          uniq, ref_slot, ref_pt, ref_cnt, num_uniq, p_base, num_latents,
                          uniq_stride, max_refs, num_slots, num_blocks, layer_num_nodes, batch_size,
                          node_offset, TL, threads, cat_blocks):
    _softevi_module.softevi_forward_dense(
        params, node_mars, Z, log_ex_p, data, vids, s_pids, nids, var_idmapping,
        uniq, ref_slot, ref_pt, ref_cnt, num_uniq, p_base, int(num_latents), int(uniq_stride),
        int(max_refs), int(num_slots), int(num_blocks), int(layer_num_nodes), int(batch_size),
        int(node_offset), int(TL), int(threads), int(cat_blocks))
