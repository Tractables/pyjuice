import inspect
import torch
import triton
from typing import Callable, Tuple, Union


# Triton launch meta-parameters: they affect COMPILATION (so they belong in the cache signature) but
# are NOT kernel arguments (they are not passed to `CompiledKernel.run`).
_TRITON_META_PARAMS = frozenset({
    "num_warps", "num_stages", "num_ctas", "maxnregs", "enable_fp_fusion",
    "launch_cooperative_grid", "launch_pdl", "grid", "warmup", "debug", "instrumentation_mode",
})


class FastJITFunction3x:
    """Low-overhead Triton relaunch for triton >= 3.0.

    Triton's ``JITFunction.run`` does a fair amount of per-launch Python (the "binder": unwrapping
    constexprs + computing the alignment/divisibility specialization for every argument, building a
    cache key, etc.). For PyJuice's hot loop -- the same kernels launched thousands of times on
    persistent buffers -- that overhead dominates at small batch. This class caches the *compiled*
    kernel under a cheap signature and, on a cache hit, calls ``CompiledKernel.run`` directly.

    SAFETY (no silent mis-launch): the signature is a SUPERSET of everything Triton specializes on --
    device, every constexpr value, every tensor's (dtype, 16-byte alignment class), every other arg's
    value, and the launch meta-params -- so a cache hit provably corresponds to the identical compiled
    kernel (a misaligned pointer, a changed constexpr, a different dtype, ... all change the signature
    and force a fresh compile via the standard path). Anything unexpected (callable grid, unknown
    kwarg, a missing required arg, any exception) falls back to the standard ``triton.jit`` launch, so
    the result is always correct -- at worst it is merely the un-accelerated path.
    """

    def __init__(self, fn: Callable):
        self.jit_fn = triton.jit(fn)
        self.params = self.jit_fn.params
        self.n_params = len(self.params)
        self.is_constexpr = tuple(p.is_constexpr for p in self.params)
        self.defaults = tuple(p.default for p in self.params)   # inspect._empty when required
        self.name_to_idx = {p.name: i for i, p in enumerate(self.params)}
        self.cache = dict()
        # Imported here (not at module load) so plain `import pyjuice` never depends on these triton
        # internals on triton versions where this fast path is unused.
        from triton.runtime import driver as _driver
        from triton import knobs as _knobs
        self._driver = _driver
        self._knobs = _knobs

    def _gather(self, args, kernel_kwargs):
        # Bind (*args, **kernel_kwargs) to the kernel parameters in declared order -- matching
        # `bound_args.values()` that the standard path passes to `CompiledKernel.run`.
        vals = list(self.defaults)
        for i, a in enumerate(args):
            vals[i] = a
        n2i = self.name_to_idx
        for name, v in kernel_kwargs.items():
            vals[n2i[name]] = v                     # KeyError on unknown kwarg -> caught -> fallback
        return vals

    def _signature(self, vals, device, meta_items):
        sig = [device]
        is_cexpr = self.is_constexpr
        for i in range(self.n_params):
            v = vals[i]
            if v is inspect._empty:                 # a required arg was not supplied -> force fallback
                raise ValueError("missing required kernel argument")
            if is_cexpr[i]:
                sig.append(v)                       # constexpr value (baked into the compiled kernel)
            elif isinstance(v, torch.Tensor):
                sig.append((v.dtype, v.data_ptr() & 15))   # dtype + 16-byte alignment class
            else:
                sig.append(v)                       # int (divisibility) / float / bool / None
        sig.append(meta_items)
        return tuple(sig)

    def __getitem__(self, grid: Union[Tuple, Callable]):
        jit_fn = self.jit_fn

        def wrapper(*args, **kwargs):
            try:
                if type(grid) is not tuple:          # callable grids: not fast-pathed
                    return jit_fn[grid](*args, **kwargs)

                if kwargs:
                    meta_items = tuple(sorted((k, v) for k, v in kwargs.items() if k in _TRITON_META_PARAMS))
                    kernel_kwargs = {k: v for k, v in kwargs.items() if k not in _TRITON_META_PARAMS} \
                        if meta_items else kwargs
                else:
                    meta_items = ()
                    kernel_kwargs = kwargs

                vals = self._gather(args, kernel_kwargs)
                device = self._driver.active.get_current_device()
                sig = self._signature(vals, device, meta_items)

                kernel = self.cache.get(sig)
                if kernel is None:
                    kernel = jit_fn[grid](*args, **kwargs)     # standard launch: compile + run + return
                    if kernel is not None and hasattr(kernel, "result"):
                        kernel = kernel.result()
                    self.cache[sig] = kernel
                    return

                stream = self._driver.active.get_current_stream(device)
                ng = len(grid)
                g0 = grid[0]
                g1 = grid[1] if ng > 1 else 1
                g2 = grid[2] if ng > 2 else 1
                # launch_metadata is only consumed by the (optional) launch hooks, never by the kernel
                # itself, so None is correctness-safe: with no hooks (the default) it is unused, and a
                # registered profiling hook would fail loudly rather than silently corrupt results.
                kernel.run(g0, g1, g2, stream, kernel.function, kernel.packed_metadata, None,
                           self._knobs.runtime.launch_enter_hook, self._knobs.runtime.launch_exit_hook, *vals)
            except Exception:
                return jit_fn[grid](*args, **kwargs)           # safe fallback: always correct

        return wrapper


class FastJITFunction2:
    def __init__(self, fn: Callable, device_check: bool = False):
        self.jit_fn = triton.JITFunction(fn)

        self.device_check = device_check

        try:
            self.constexpr_ids = [p.num for p in self.jit_fn.params if p.is_constexpr]
            self.constexpr_names = {
                p.name: p.num for p in self.jit_fn.params if p.is_constexpr
            }
            self.nonconstexpr_names = [
                p.name for p in self.jit_fn.params if not p.is_constexpr
            ]
        except AttributeError:
            self.constexpr_ids = self.jit_fn.constexprs
            self.constexpr_names = {
                self.jit_fn.arg_names[i]: i for i in self.jit_fn.constexprs
            }
            self.nonconstexpr_names = [
                self.jit_fn.arg_names[i]
                for i in range(len(self.jit_fn.arg_names))
                if i not in self.jit_fn.constexprs
            ]

        self.constexpr_ids_set = set(self.constexpr_ids)

        self.cache = dict()

    def __getitem__(self, grid: Union[Tuple, Callable]):

        def wrapper(*args, **kwargs):

            nonlocal grid

            signature_list = list()

            # Get device ID
            if self.device_check:
                device_id = -1
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        device_id = arg.device.index
                        break

                if device_id == -1:
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            device_id = v.device.index
                            break

                signature_list.append(device_id)

            for i in self.constexpr_ids:
                if i >= len(args):
                    break
                signature_list.append(args[i])

            for k, v in kwargs.items():
                if k in self.constexpr_names:
                    signature_list.append((self.constexpr_names[k], v))

            if "batch_size" in kwargs:
                signature_list.append(("batch_size", kwargs["batch_size"]))

            if isinstance(grid, Callable):
                grid = grid(kwargs)

            grid_length = len(grid)
            grid0, grid1, grid2 = (
                grid[0],
                grid[1] if grid_length > 1 else 1,
                grid[2] if grid_length > 2 else 1,
            )

            signature = tuple(signature_list)
            if signature in self.cache:
                kernel = self.cache[signature]

                aligned_args = list()
                for i, arg in enumerate(args):
                    if i not in self.constexpr_ids_set:
                        aligned_args.append(arg)

                for k in self.nonconstexpr_names:
                    if k in kwargs:
                        aligned_args.append(kwargs[k])

                if self.device_check:
                    with torch.cuda.device(device_id):
                        kernel[(grid0, grid1, grid2)](*aligned_args)
                else:
                    kernel[(grid0, grid1, grid2)](*aligned_args)
            else:
                if self.device_check:
                    with torch.cuda.device(device_id):
                        kernel = self.jit_fn[grid](*args, **kwargs)
                        self.cache[signature] = kernel
                else:
                    kernel = self.jit_fn[grid](*args, **kwargs)
                    self.cache[signature] = kernel

        return wrapper


import os

# Backward-compat alias for the renamed original launcher (imported by a few modules; never used
# directly now that `triton_jit` dispatches by version).
FastJITFunction = FastJITFunction2

# Low-overhead launch wrappers reduce Triton's per-launch Python overhead, which dominates PyJuice's
# small-batch step. Opt out with PYJUICE_FAST_LAUNCH=0 (falls back to the stock triton.jit launch).
_FAST_LAUNCH = os.environ.get("PYJUICE_FAST_LAUNCH", "1") != "0"


def triton_jit(fn: Callable, device_check: bool = False):
    # The launch ABI changed across triton major versions, so the wrapper is version-specific:
    #   - triton >= 3.0: FastJITFunction3x (cache compiled kernel + CompiledKernel.run, superset
    #     signature + fallback -> correctness-safe).
    #   - triton 2.x + torch <= 2.2: FastJITFunction2 (the original manual arg-packer).
    #   - otherwise: stock triton.jit.
    try:
        _triton_major = int(triton.__version__.split(".")[0])
    except Exception:
        _triton_major = 3  # assume modern triton if version can't be parsed

    if _triton_major >= 3:
        if _FAST_LAUNCH:
            try:
                return FastJITFunction3x(fn)
            except Exception:
                return triton.jit(fn)
        return triton.jit(fn)

    # For triton 2.x with older torch, use the FastJITFunction2 optimisation
    try:
        _torch_minor = int(torch.__version__.split(".")[1])
    except Exception:
        _torch_minor = 99

    if torch.__version__.startswith("2.") and _torch_minor <= 2:
        return FastJITFunction2(fn, device_check=device_check)
    else:
        return triton.jit(fn)
