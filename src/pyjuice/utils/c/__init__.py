"""The fast launcher's per-argument binding in C++ (``launch_bind.cpp``, for
:class:`pyjuice.utils.kernel_launcher.FastJITFunction3x`).

Built on first use -- plain host C++ against PyTorch's headers, no CUDA compiler -- and None where it cannot be (no
compiler or ninja, a failed build) or is switched off (``PYJUICE_NATIVE_BIND=0``): the launcher then binds in Python, as
before. It changes nothing a launch computes: it builds the same values and signature the Python binding does.
"""

import os
import warnings

#: Whether the launcher binds through the C++ module (``PYJUICE_NATIVE_BIND=0``: the Python binding always)
ENABLED = os.environ.get("PYJUICE_NATIVE_BIND", "1") != "0"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_module = None
_attempted = False


def launch_bind():
    """The C++ binding module (its ``bind``), built and loaded once, or None (warns once if the build fails)."""
    global _module, _attempted
    if not _attempted:
        _attempted = True
        if ENABLED:
            from pyjuice.utils.cuda_ext import jit_load
            try:
                _module = jit_load("pyjuice_launch_bind", [os.path.join(_THIS_DIR, "launch_bind.cpp")],
                                   extra_cflags = ["-O3"], verbose = False)
            except Exception as e:
                warnings.warn(f"pyjuice's C++ launch binding failed to build ({type(e).__name__}: {e}); Triton "
                              f"launches bind in Python.", RuntimeWarning)
    return _module
