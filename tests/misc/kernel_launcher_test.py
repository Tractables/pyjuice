import os
import subprocess
import sys

import pytest
import torch
import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import FastJITFunction3x


# Raw kernel function (NOT @triton.jit -- FastJITFunction3x / triton.jit wrap it themselves).
def _scale_kernel(x_ptr, y_ptr, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask = mask) * scale, mask = mask)


@pytest.mark.skipif(not (torch.cuda.is_available() and int(triton.__version__.split(".")[0]) >= 3),
                    reason="FastJITFunction3x targets triton >= 3 on CUDA")
def test_fast_jit3x_matches_reference_no_silent_miscache():
    """
    The triton>=3 fast launcher caches the compiled kernel under a cheap signature and relaunches it
    directly. Its signature must be a SUPERSET of everything triton specializes on -- otherwise a
    cache hit would silently reuse a kernel compiled for a DIFFERENT alignment / dtype / constexpr.
    This test hammers exactly those axes and checks bit-identical results vs the stock triton.jit
    launch on every call (a too-coarse signature would silently corrupt at least one).
    """
    dev = torch.device("cuda:0")
    ref = triton.jit(_scale_kernel)
    fast = FastJITFunction3x(_scale_kernel)

    torch.manual_seed(0)
    # Vary: size, dtype, constexpr BLOCK, scalar `scale`, AND pointer alignment (offset slices) --
    # interleaved and repeated so cached entries are exercised, including the aligned/misaligned pair
    # that share constexprs (the case a coarse signature would mis-cache).
    configs = []
    for dtype in (torch.float32, torch.float16):
        for BLOCK in (64, 128):
            for n in (1000, 4096):
                for off in (0, 1, 3):           # 0 -> 16B aligned; 1/3 -> misaligned
                    configs.append((dtype, BLOCK, n, off))
    configs = configs * 3                        # repeat -> hit the cached fast path many times

    for dtype, BLOCK, n, off in configs:
        base = torch.randn(n + off, device = dev, dtype = dtype)
        x = base[off:]
        scale = 2.5
        grid = (triton.cdiv(n, BLOCK),)
        y_ref = torch.zeros(n, device = dev, dtype = dtype)
        y_fast = torch.zeros(n, device = dev, dtype = dtype)
        ref[grid](x, y_ref, n, scale, BLOCK = BLOCK)
        fast[grid](x, y_fast, n, scale, BLOCK = BLOCK)
        torch.cuda.synchronize()
        assert torch.equal(y_ref, y_fast), \
            f"fast launcher diverged from triton.jit (dtype={dtype}, BLOCK={BLOCK}, n={n}, off={off})"
        # and both must equal the actual math
        assert torch.allclose(y_fast.float(), (x * scale).float(), atol = 1e-2), "wrong result"


# Every kind of parameter: tensors, ints, a float, a pointer that may be None, constexprs, defaults.
def _mixed_kernel(x_ptr, y_ptr, n, scale, z_ptr, offset = 0, A: tl.constexpr = 4, FLAG: tl.constexpr = False,
                  B: tl.constexpr = 32):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask = mask) * scale + offset + A
    if FLAG:
        v = v + tl.load(z_ptr + offs, mask = mask)
    tl.store(y_ptr + offs, v, mask = mask)


@pytest.mark.skipif(not (torch.cuda.is_available() and int(triton.__version__.split(".")[0]) >= 3),
                    reason="FastJITFunction3x targets triton >= 3 on CUDA")
def test_fast_jit3x_binders_build_the_generic_signature():
    """
    A call pattern's generated binder (positional count, keyword names in order) builds exactly the
    signature the generic binding (`_gather` + `_signature`) does, and the same values but the tensors on
    the current device as their addresses (a CPU tensor, or one on another device, as it is); a pattern
    the generic binding does not bind plainly (a name repeated or unknown, a required parameter missing)
    gets none.
    """
    fast = FastJITFunction3x(_mixed_kernel)
    dev = torch.device("cuda:0")
    x, y = torch.randn(100, device = dev), torch.zeros(100, device = dev)
    xc = torch.randn(100)                                        # a CPU tensor: passed as it is
    names = [p.name for p in fast.params]
    full = dict(x_ptr = x, y_ptr = y, n = 100, scale = 0.5, z_ptr = None, offset = 3, A = 8, FLAG = True, B = 64)
    patterns = []
    for npos in range(len(names) + 1):
        pos = names[:npos]
        rest = [k for k in names[npos:]]
        for drop in ([], ["offset"], ["A", "FLAG", "B"], ["offset", "B"]):
            kw = [k for k in rest if k not in drop]
            if any(k in pos for k in drop):
                continue
            for meta in ({}, {"num_warps": 2}, {"num_stages": 1, "num_warps": 4}):
                for order in (kw, kw[::-1]):
                    patterns.append((pos, order, meta))
    assert len(patterns) > 100
    for pos, order, meta in patterns:
        for xv in (x, x[1:], xc):
            vals_in = dict(full, x_ptr = xv)
            args = tuple(vals_in[k] for k in pos)
            kwargs = {**{k: vals_in[k] for k in order}, **meta}
            bind = fast._binder(len(args), tuple(kwargs))
            assert bind is not None, (pos, order, meta)
            for current in (0, 1):                               # (1: the tensors' device is not the current one)
                got_vals, got_sig = bind(current, args, kwargs)
                vals, meta_items = fast._gather(args, kwargs)
                assert got_sig == fast._signature(vals, current, meta_items), (pos, order, meta)
                want = [v.data_ptr() if isinstance(v, torch.Tensor) and v.get_device() == current else v for v in vals]
                assert len(got_vals) == len(want)
                for g, w in zip(got_vals, want):
                    assert (g is w) or (type(g) is int and g == w), (pos, order, meta, current, g, w)
                assert sum(type(g) is int and isinstance(v, torch.Tensor) for g, v in zip(got_vals, vals)) == \
                    (sum(isinstance(v, torch.Tensor) and v.is_cuda for v in vals) if current == 0 else 0)
    # not bound plainly: the generic binding's (unknown names fall back, repeated ones keep the keyword's)
    assert fast._binder(2, ("x_ptr", "n", "scale", "z_ptr")) is None                 # x_ptr twice
    assert fast._binder(0, ("x_ptr", "y_ptr", "n", "scale", "z_ptr", "bogus")) is None
    assert fast._binder(0, ("x_ptr", "y_ptr", "n", "scale")) is None                  # z_ptr missing
    assert fast._binder(len(names) + 1, ()) is None


@pytest.mark.skipif(not (torch.cuda.is_available() and int(triton.__version__.split(".")[0]) >= 3),
                    reason="FastJITFunction3x targets triton >= 3 on CUDA")
def test_fast_jit3x_runs_with_addresses_and_rejects_cpu_tensors():
    """
    The fast launch hands Triton a tensor on the current device as its address: the same results as the
    stock launch, every parameter kind, the cached patterns relaunched; a CPU tensor of the same dtype and
    alignment as a cached launch's is passed as it is, and Triton's launcher rejects it as it does on the
    stock path.
    """
    dev = torch.device("cuda:0")
    ref = triton.jit(_mixed_kernel)
    fast = FastJITFunction3x(_mixed_kernel)
    torch.manual_seed(0)
    n = 1000
    x, z = torch.randn(n + 1, device = dev), torch.randn(n, device = dev)
    for _ in range(3):
        for xv in (x[:n], x[1:]):
            for flag in (False, True):
                y_ref, y_fast = torch.zeros(n, device = dev), torch.zeros(n, device = dev)
                ref[(triton.cdiv(n, 64),)](xv, y_ref, n, 0.5, z, offset = 2, FLAG = flag, B = 64)
                fast[(triton.cdiv(n, 64),)](xv, y_fast, n, 0.5, z, offset = 2, FLAG = flag, B = 64)
                fast[(triton.cdiv(n, 64),)](xv, y_fast, n = n, scale = 0.5, z_ptr = z, offset = 2, FLAG = flag,
                                            B = 64, num_warps = 4)
                torch.cuda.synchronize()
                assert torch.equal(y_ref, y_fast)
    base = torch.empty(n + 64)                                  # a CPU tensor aligned as the CUDA ones
    xc = base[((-base.data_ptr()) % 16) // 4:][:n]
    assert xc.data_ptr() & 15 == x[:n].data_ptr() & 15

    def outcome(kernel):
        try:
            kernel[(triton.cdiv(n, 64),)](xc, torch.zeros(n, device = dev), n, 0.5, z, offset = 2, FLAG = True, B = 64)
            torch.cuda.synchronize()
        except Exception as e:
            return type(e), "cpu tensor" in str(e)
        return None

    stock = outcome(ref)
    if stock is None:
        pytest.skip("this driver lets kernels read pageable host memory: no rejection to keep")
    assert outcome(fast) == stock


def _two_gpus():
    """
    Two GPU ids for a real cross-device launch, or None: the conftest pins every test process to ONE GPU,
    so `device_count()` is always 1 here -- read the pool it published and hand the pair to a child process
    (as tests/queries/sample/sample_scope_plan_test.py does, which explains the trap at length).
    """
    pool = os.environ.get("PYJUICE_TEST_GPU_POOL")
    if pool:
        gpus = [g for g in pool.split(",") if g]
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output = True, text = True)
            gpus = [str(i) for i in range(len(out.stdout.strip().splitlines()))]
        except (FileNotFoundError, OSError):
            gpus = []
    if len(gpus) < 2:
        return None
    idle = gpus[int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")):]
    return idle[:2] if len(idle) >= 2 else gpus[-2:]


_CROSS_DEVICE_CHILD = r"""
import torch, triton, triton.language as tl
from pyjuice.utils.kernel_launcher import FastJITFunction3x


def _double_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask = mask) * 2.0, mask = mask)


ref, fast = triton.jit(_double_kernel), FastJITFunction3x(_double_kernel)
n = 1000


def outcome(kernel, on, current):
    with torch.cuda.device(current):
        x = torch.arange(n, dtype = torch.float32, device = f"cuda:{on}")
        y = torch.zeros(n, device = f"cuda:{on}")
        try:
            kernel[(triton.cdiv(n, 128),)](x, y, n, BLOCK = 128)
            torch.cuda.synchronize(on)
        except Exception as e:
            return "raises", type(e).__name__, "cannot be accessed" in str(e)
        return "runs", torch.equal(y.cpu(), x.cpu() * 2)


for d in (0, 1):                       # the fast launcher's entries for either device, from its own tensors
    for kernel in (ref, fast, fast):
        assert outcome(kernel, d, d) == ("runs", True), d
for on, current in ((1, 0), (0, 1)):   # the other device's tensors: a cached signature, launched as the stock one
    want = outcome(ref, on, current)
    got = outcome(fast, on, current)
    assert got == want, (on, current, got, want)
    print("tensors on", on, "current", current, "->", want)
print("CHILD_OK")
"""


@pytest.mark.skipif(not (torch.cuda.is_available() and int(triton.__version__.split(".")[0]) >= 3),
                    reason="FastJITFunction3x targets triton >= 3 on CUDA")
@pytest.mark.skipif(_two_gpus() is None, reason = "requires >= 2 GPUs")
def test_fast_jit3x_cross_device_launches_as_the_stock_one(tmp_path):
    """
    REGRESSION. A tensor on a GPU other than the current one keeps going to Triton's launcher as a tensor,
    whose driver check refuses it when that GPU's memory is not mapped in the current context (the error
    tests/queries/sample/sample_scope_plan_test.py's cross-device move once hit): handed over as an address
    it would reach the kernel unchecked. In a CHILD with two GPUs visible: each device's launch from its
    own tensors, then a cached signature launched with the other device's tensors, which must end as the
    stock launch's does -- refused, or run alike where the peer's memory is mapped.
    """
    gpus = _two_gpus()
    script = tmp_path / "cross_device_child.py"              # (a file: Triton reads a kernel's source)
    script.write_text(_CROSS_DEVICE_CHILD)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env.pop("PYJUICE_TEST_GPU_POOL", None)
    proc = subprocess.run([sys.executable, str(script)], capture_output = True, text = True, timeout = 900, env = env)
    assert proc.returncode == 0 and "CHILD_OK" in proc.stdout, (
        f"the cross-device launches differ from the stock ones on GPUs {gpus} (exit {proc.returncode})\n"
        f"--- child stdout ---\n{proc.stdout}\n--- child stderr ---\n{proc.stderr}")


if __name__ == "__main__":
    test_fast_jit3x_matches_reference_no_silent_miscache()
    test_fast_jit3x_binders_build_the_generic_signature()
    test_fast_jit3x_runs_with_addresses_and_rejects_cpu_tensors()
    print("ok")
