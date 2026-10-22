"""
What the :mod:`pyjuice.nodes.distributions.softevi` modules share: ``tlmath``; the identity key a step's evidence is
cached under (:func:`_evidence_cache_key` -- not a data pointer); a layer's nodes as emission rows; the float32 scratch
the external evidence gradient accumulates into; and the Triton test of a marginalized slot
(:func:`_missing_slot_mask`).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import weakref


# In the latest triton, math functions were shuffled around into different modules:
# https://github.com/openai/triton/pull/3172
if hasattr(tl.extra.cuda, "libdevice"):
    tlmath = tl.extra.cuda.libdevice
else:
    tlmath = tl.math


def _row_index(layer, num_cats):
    """
    `s_pids` expressed as emission-ROW indices, for addressing the transposed table.

    Depends only on the compiled layout, which never changes, so unlike the transposed table itself
    this is cached for the layer's lifetime rather than for the `fast_inference` scope.
    """
    # Keyed by DEVICE, not merely cached. It is derived from `s_pids`, which `InputLayer.to()` moves;
    # `to()` drops the transposed table but has no reason to know about this one, so a bare cache
    # would hand a Triton launch a tensor still sitting on the old device.
    cached = getattr(layer, "_softevi_row_index", None)
    if cached is not None and cached[0] == layer.s_pids.device:
        return cached[1]

    idx = (layer.s_pids // num_cats).contiguous()
    layer._softevi_row_index = (idx.device, idx)
    return idx


def _evidence_cache_key(*tensors):
    """An identity key for caller-supplied tensors that survives allocator address reuse.

    `data_ptr()` IS NOT AN IDENTITY, and keying on it is a correctness bug rather than a stale-cache
    inefficiency. The caching allocator hands a freed block straight back out, so under
    `torch.no_grad()` -- where a step's evidence dies the moment the forward returns -- the next call's
    tensors routinely land on the same addresses with the same shapes and a fresh `_version` of 0. The
    key then matches BYTE FOR BYTE while the data is entirely different, and the cache serves the
    previous call's answer: the circuit is evaluated against the previous step's evidence, which can
    return a POSITIVE log-likelihood. Measured on a 32-step gated HMM, 256 of 616 hits were stale.

    Under autograd the graph keeps the evidence alive until the backward, so the next step's tensors
    get fresh addresses and the collision does not arise -- which is why this only ever showed up in
    evaluation, never in training.

    A weak reference distinguishes the two cases exactly. While the tensor is alive its address cannot
    be reused, so `ref() is tensor` means the same tensor and nothing else; once it dies the reference
    reads back as `None` and the entry simply misses. `_version` is still carried, to catch in-place
    mutation of a tensor that IS the same object.
    """
    return tuple((weakref.ref(t), t._version, tuple(t.shape)) for t in tensors)


def _evidence_cache_hit(key, tensors):
    """Whether `key` (from :func:`_evidence_cache_key`) still refers to exactly `tensors`."""
    return key is not None and len(key) == len(tensors) and all(
        ref() is t and version == t._version and shape == tuple(t.shape)
        for (ref, version, shape), t in zip(key, tensors)
    )


def _grad_scratch(layer, grad):
    """A float32 buffer for the kernels to accumulate the external evidence gradient into.

    The caller's gradient buffer is frequently half precision -- it goes back to a language model that
    runs in bf16 -- but every kernel here accumulates into it with an atomic add and reads it as float.
    Accumulating in half precision would be both lossy (thousands of small atomic adds land on the same
    slot, which is exactly where bf16's 8 mantissa bits give out) and slower. So accumulate in float32
    and fold the result back once, in `_GradCastEpilogue`.

    Returns the caller's own tensor when it is already float32, so the common path allocates nothing."""
    if grad is None or grad.dtype == torch.float32:
        return grad
    buf = getattr(layer, "_grad_f32_buf", None)
    if buf is None or buf.shape != grad.shape or buf.device != grad.device:
        buf = torch.zeros(grad.shape, dtype = torch.float32, device = grad.device)
        layer._grad_f32_buf = buf
    return buf


@triton.jit
def _missing_slot_mask(missing_mask_ptr, vid, offsets_b, batch_size, num_vars: tl.constexpr,
                       missing_mask_mode: tl.constexpr):
    """`True` where this leaf's variable is marginalized for that batch element.

    Decodes the same three `missing_mask` layouts as `_flows_kernel_template`. Every kernel here runs one
    VARIABLE per program (`vid` is a scalar), so the result is a [BLOCK_SIZE_B] vector; the `offsets_b * 0`
    in mode 1 is just how a scalar is broadcast to that shape."""
    if missing_mask_mode == 1:
        return tl.load(missing_mask_ptr + vid + offsets_b * 0).to(tl.int1)
    elif missing_mask_mode == 2:
        return tl.load(missing_mask_ptr + offsets_b * num_vars + vid).to(tl.int1)
    else:
        return tl.load(missing_mask_ptr + vid * batch_size + offsets_b).to(tl.int1)
