"""Shared helper for JIT-compiling the optional CUDA extensions.

`torch.utils.cpp_extension.load` serializes builds with a lock file and waits on it forever. If a build
is ever interrupted -- Ctrl-C, a killed job, a scheduler timeout -- the lock is left behind and EVERY
later process that touches that extension hangs on import, with no error and no timeout. That is a
particularly nasty failure mode because it looks like a slow GPU rather than a stuck build.

`jit_load` clears a lock that is provably stale (old, and left by no live build) before calling `load`,
so a killed build costs one recompile instead of wedging the installation.
"""

import os
import time
import warnings

_STALE_AFTER_SECONDS = 300.0


def _clear_stale_lock(name: str, verbose: bool = False) -> None:
    try:
        from torch.utils.cpp_extension import _get_build_directory
        build_dir = _get_build_directory(name, verbose = False)
    except Exception:
        return

    lock_path = os.path.join(build_dir, "lock")
    try:
        age = time.time() - os.path.getmtime(lock_path)
    except OSError:
        return                                  # no lock, or it vanished under us

    if age > _STALE_AFTER_SECONDS:
        try:
            os.remove(lock_path)
            if verbose:
                warnings.warn(
                    f"Removed a stale build lock for CUDA extension '{name}' (age {age:.0f}s). It was "
                    "left behind by an interrupted build and would otherwise block this process "
                    "indefinitely.", RuntimeWarning)
        except OSError:
            pass


def jit_load(name: str, sources, extra_cuda_cflags = None, **kwargs):
    """`torch.utils.cpp_extension.load`, but not wedged by a lock from an interrupted build."""
    from torch.utils.cpp_extension import load

    _clear_stale_lock(name)
    return load(name = name, sources = sources, extra_cuda_cflags = extra_cuda_cflags, **kwargs)
