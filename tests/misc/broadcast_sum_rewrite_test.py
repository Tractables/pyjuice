import io
import os
import re
import tokenize

import torch
import triton
import triton.language as tl

import pytest

import pyjuice
from pyjuice.layer.kernels import _BROADCAST_SUM_FORBIDDEN


# See `_BROADCAST_SUM_NOTE` in `pyjuice/layer/kernels/__init__.py`: Triton rewrites the fp32 spelling
# `tl.sum(X[:,:,None] * Y[None,:,:], axis = 1)` into a tensor-core dot that multiplies the result by
# 8 / K when the reduced dimension K is 1, 2 or 4, and truncates to TF32 otherwise.

_MARKER = "# broadcast-sum ok:"


def _code_only(source):
    """`source` with every string literal and comment blanked out, line structure preserved, so the
    explanation of the forbidden spelling (and quotes of it) cannot match."""
    lines = source.splitlines(keepends = True)
    out = [list(l) for l in lines]
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            (sr, sc), (er, ec) = tok.start, tok.end
            for r in range(sr, er + 1):
                row = out[r - 1]
                lo = sc if r == sr else 0
                hi = ec if r == er else len(row)
                for c in range(lo, hi):
                    if row[c] != "\n":
                        row[c] = " "
    return ["".join(l) for l in out]


def test_no_kernel_uses_the_miscompiled_broadcast_sum():
    root = os.path.dirname(pyjuice.__file__)
    pattern = re.compile(_BROADCAST_SUM_FORBIDDEN)
    offenders = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path) as f:
                source = f.read()
            raw = source.splitlines()
            for i, line in enumerate(_code_only(source)):
                if pattern.search(line) and _MARKER not in raw[i]:
                    offenders.append(f"{os.path.relpath(path, root)}:{i + 1}: {raw[i].strip()}")
    assert not offenders, (
        "`tl.sum(X[:,:,None] * Y[None,:,:], axis = 1)` is miscompiled by Triton for fp32 operands "
        "(x8/K for a reduced dimension K < 8, TF32-truncated otherwise). Write "
        "`tl.sum(tl.trans(X)[:,:,None] * Y[:,None,:], axis = 0)` instead, or mark a bf16 site with "
        f"`{_MARKER} <reason>`. See `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py.\n"
        + "\n".join(offenders))


def test_the_guard_itself_catches_the_spelling():
    """Positive control for the scan above: it must flag the bad spelling in code and ignore it in
    comments, strings and marked lines."""
    pattern = re.compile(_BROADCAST_SUM_FORBIDDEN)
    src = (
        "acc = tl.sum(epars[:,:,None] * tl.trans(n_fdm_sub)[None,:,:], axis = 1)\n"
        "# acc = tl.sum(a[:,:,None] * b[None,:,:], axis = 1)\n"
        "s = 'tl.sum(a[:,:,None] * b[None,:,:], axis = 1)'\n"
        "acc = tl.sum(tl.trans(epars)[:,:,None] * n_fdm_sub[:,None,:], axis = 0)\n"
    )
    hits = [i for i, l in enumerate(_code_only(src)) if pattern.search(l)]
    assert hits == [0]


@triton.jit
def _bsum_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, AXIS0: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    if AXIS0:
        c = tl.sum(tl.trans(a)[:, :, None] * b[:, None, :], axis = 0)
    else:
        c = tl.sum(a[:, :, None] * b[None, :, :], axis = 1)  # broadcast-sum ok: the bug under test
    tl.store(c_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)


@pytest.mark.parametrize("K", [1, 2, 4, 8, 16])
def test_the_axis0_broadcast_sum_is_exact_fp32(K):
    """Every fp32 small-tile fallback in pyjuice relies on this; if a Triton upgrade starts rewriting
    the axis-0 spelling too, this fails here rather than as wrong likelihoods somewhere else."""
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    M, N = 32, 32
    a = torch.rand(M, K, device = device) + 0.1
    b = torch.rand(K, N, device = device) + 0.1
    c = torch.empty(M, N, device = device)
    _bsum_kernel[(1,)](a, b, c, M, K, N, True)
    ref = a.double() @ b.double()
    rel = ((c.double() - ref).abs() / ref).max().item()
    assert rel < 1e-5, f"the axis-0 broadcast sum is no longer exact fp32 at K={K} (max rel err {rel:.2e})"
