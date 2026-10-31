import inspect
import torch
import triton
from typing import Callable, Tuple, Union

from pyjuice.utils.c import launch_bind


# Triton launch meta-parameters: they affect COMPILATION (so they belong in the cache signature) but
# are NOT kernel arguments (they are not passed to `CompiledKernel.run`).
_TRITON_META_PARAMS = frozenset({
    "num_warps", "num_stages", "num_ctas", "maxnregs", "enable_fp_fusion",
    "launch_cooperative_grid", "launch_pdl", "grid", "warmup", "debug", "instrumentation_mode",
})

_UNSEEN = object()                      # (a call pattern no launch has had yet)


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

    A launch binds its arguments with a function generated for its call pattern -- how many arguments
    it passes by position, and the names it passes by keyword, in order (``_binder``) --, which builds
    the same signature as the generic binding (``_gather`` + ``_signature``) without a loop over the
    parameters. It hands ``CompiledKernel.run`` each tensor on the current CUDA device as its address
    (the int its signature read), which Triton's launcher takes as it is; for a tensor, it would call
    ``data_ptr`` and ask the driver for the pointer's device address in the current context
    (``cuPointerGetAttribute``) -- for memory of the current device, that address itself -- for every
    tensor argument of every launch. Any other tensor (a CPU tensor, PyTorch's pinned memory among them;
    a tensor on another GPU, which the driver refuses when that GPU's memory is not mapped here) is passed
    as it is, and Triton's launcher handles it as before. Where the C++ binding builds (:mod:`pyjuice.utils.c`,
    on first use; ``PYJUICE_NATIVE_BIND=0`` switches it off), the binder hands the values it unpacked to it, which
    builds the same values and signature in one call -- a call with a value outside its domain (a tensor subclass,
    a sparse tensor, ... -- see ``launch_bind.cpp``) bound in Python as before.
    """

    def __init__(self, fn: Callable):
        self.jit_fn = triton.jit(fn)
        self.params = self.jit_fn.params
        self.n_params = len(self.params)
        self.is_constexpr = tuple(p.is_constexpr for p in self.params)
        self.constexpr_mask = bytes(self.is_constexpr)              # (the C++ binding's)
        self.defaults = tuple(p.default for p in self.params)   # inspect._empty when required
        self.name_to_idx = {p.name: i for i, p in enumerate(self.params)}
        self.cache = dict()
        self.binders = dict()                                    # call pattern -> its binder (None: generic)
        # Imported here (not at module load) so plain `import pyjuice` never depends on these triton
        # internals on triton versions where this fast path is unused.
        from triton.runtime import driver as _driver
        from triton import knobs as _knobs
        self._driver = _driver
        self._knobs = _knobs

    def _gather(self, args, kwargs):
        # Bind (*args, **kwargs) to the kernel parameters in declared order -- matching
        # `bound_args.values()` that the standard path passes to `CompiledKernel.run` -- and set the
        # launch meta-parameters aside, sorted. One pass over `kwargs`: this runs on every launch.
        vals = list(self.defaults)
        for i, a in enumerate(args):
            vals[i] = a
        n2i = self.name_to_idx
        meta = []
        for name, v in kwargs.items():
            if name in _TRITON_META_PARAMS:
                meta.append((name, v))
            else:
                vals[n2i[name]] = v                 # KeyError on unknown kwarg -> caught -> fallback
        return vals, (tuple(sorted(meta)) if meta else ())

    def _signature(self, vals, device, meta_items):
        # Constexprs by value (baked into the compiled kernel), tensors by dtype + 16-byte alignment class,
        # every other argument by value (int divisibility / float / bool / None). One comprehension rather
        # than a loop of appends: this runs on every launch. A required argument that was not supplied
        # stays `inspect._empty`, under which no successful launch is ever cached, so such a call goes to
        # the standard launch and fails there.
        Tensor = torch.Tensor
        return (device, *[(v.dtype, v.data_ptr() & 15) if not c and isinstance(v, Tensor) else v
                          for v, c in zip(vals, self.is_constexpr)], meta_items)

    def _binder(self, npos: int, knames: tuple):
        # The binding of the calls passing the first `npos` parameters by position and `knames` by keyword,
        # in this order, as a function `(device, args, kwargs) -> (values to run with, signature)`: the
        # positional arguments and the keyword values unpacked into locals named as the parameters, the
        # missing ones their defaults, then the signature `_signature` would build (constexprs and non-tensor
        # values by value, a tensor's (dtype, data_ptr() & 15), the meta-parameters given, sorted), and the
        # values in declared order -- a tensor on the current device as the int its signature read. None
        # (the call bound by `_gather`, as before) when the generic binding would not be a plain one: a name
        # it does not know (`_gather` raises, the launch falls back), one given twice (`_gather` keeps the
        # keyword's), a required parameter not given, or a parameter name the generated code reserves
        # (`_fj_...`).
        names = [p.name for p in self.params]
        if npos > len(names) or any(name.startswith("_fj_") for name in names):
            return None
        given, meta = set(names[:npos]), []
        local = []                                              # every keyword value's local
        for k in knames:
            if k in _TRITON_META_PARAMS:
                meta.append(k)
                local.append(f"_fj_m{len(meta) - 1}")
            elif k in self.name_to_idx and k not in given:
                given.add(k)
                local.append(k)
            else:
                return None
        lines = ["def _fj_bind(_fj_dev, _fj_args, _fj_kwargs):"]
        if npos:
            lines.append(f"    {', '.join(names[:npos])}, = _fj_args")
        if knames:
            lines.append(f"    {', '.join(local)}, = _fj_kwargs.values()")
        defaults = {}
        for i, name in enumerate(names):
            if name not in given:
                if self.defaults[i] is inspect.Parameter.empty:
                    return None
                defaults[f"_fj_d{i}"] = self.defaults[i]
                lines.append(f"    {name} = _fj_d{i}")
        order = sorted(range(len(meta)), key = lambda j: meta[j])
        meta_items = "".join(f"({meta[j]!r}, _fj_m{j}), " for j in order)
        values = f"({', '.join(names)},)"
        tail, sig = [], []
        for i, (name, c) in enumerate(zip(names, self.is_constexpr)):
            if c:
                sig.append(name)
                continue
            tail += [f"    if _fj_isinstance({name}, _fj_Tensor):",
                     f"        _fj_p = {name}.data_ptr()",
                     f"        _fj_s{i} = ({name}.dtype, _fj_p & 15)",
                     f"        if {name}.get_device() == _fj_dev:",
                     f"            {name} = _fj_p",
                     f"    else:",
                     f"        _fj_s{i} = {name}"]
            sig.append(f"_fj_s{i}")
        tail.append(f"    return {values}, (_fj_dev, {', '.join(sig)}, ({meta_items}))")
        python = self._compile(lines + tail, dict(_fj_isinstance = isinstance, _fj_Tensor = torch.Tensor, **defaults))
        native = launch_bind() if python is not None else None
        if native is None:
            return python
        # The C++ binding (`pyjuice.utils.c`): the same unpacking, then one call building the same values and
        # signature -- or None for a value outside its domain (a tensor subclass, ...), that call bound in Python
        tail = [f"    _fj_r = _fj_bind_c(_fj_dev, {values}, _fj_constexpr, ({meta_items}))",
                f"    return _fj_python(_fj_dev, _fj_args, _fj_kwargs) if _fj_r is None else _fj_r"]
        return self._compile(lines + tail, dict(_fj_bind_c = native.bind, _fj_constexpr = self.constexpr_mask,
                                                _fj_python = python, **defaults)) or python

    @staticmethod
    def _compile(lines, namespace):
        # The generated `_fj_bind`, or None if it does not compile (e.g. no parameter at all: the generic binding)
        try:
            exec("\n".join(lines), namespace)
        except Exception:
            return None
        return namespace["_fj_bind"]

    def __getitem__(self, grid: Union[Tuple, Callable]):
        jit_fn = self.jit_fn

        def wrapper(*args, **kwargs):
            try:
                if type(grid) is not tuple:          # callable grids: not fast-pathed
                    return jit_fn[grid](*args, **kwargs)

                driver = self._driver.active
                device = driver.get_current_device()
                key = (len(args), *kwargs)
                bind = self.binders.get(key, _UNSEEN)
                if bind is _UNSEEN:
                    bind = self.binders[key] = self._binder(len(args), key[1:])
                if bind is not None:
                    vals, sig = bind(device, args, kwargs)
                else:
                    vals, meta_items = self._gather(args, kwargs)
                    sig = self._signature(vals, device, meta_items)

                kernel = self.cache.get(sig)
                if kernel is None:
                    kernel = jit_fn[grid](*args, **kwargs)     # standard launch: compile + run + return
                    if kernel is not None and hasattr(kernel, "result"):
                        kernel = kernel.result()
                    self.cache[sig] = kernel
                    return

                stream = driver.get_current_stream(device)
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


class FastJITFunction2x:
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
FastJITFunction = FastJITFunction2x

# Low-overhead launch wrappers reduce Triton's per-launch Python overhead, which dominates PyJuice's
# small-batch step. Opt out with PYJUICE_FAST_LAUNCH=0 (falls back to the stock triton.jit launch).
_FAST_LAUNCH = os.environ.get("PYJUICE_FAST_LAUNCH", "1") != "0"


def triton_jit(fn: Callable, device_check: bool = False):
    # The launch ABI changed across triton major versions, so the wrapper is version-specific:
    #   - triton >= 3.0: FastJITFunction3x (cache compiled kernel + CompiledKernel.run, superset
    #     signature + fallback -> correctness-safe).
    #   - triton 2.x + torch <= 2.2: FastJITFunction2x (the original manual arg-packer).
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

    # For triton 2.x with older torch, use the FastJITFunction2x optimisation
    try:
        _torch_minor = int(torch.__version__.split(".")[1])
    except Exception:
        _torch_minor = 99

    if torch.__version__.startswith("2.") and _torch_minor <= 2:
        return FastJITFunction2x(fn, device_check=device_check)
    else:
        return triton.jit(fn)
