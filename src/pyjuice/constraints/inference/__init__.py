"""
Inference backends for constrained queries (see :mod:`.base` for the interface). Each backend depends only on
constraint capabilities and the PC's structure.
"""

from .base import Backend
from .dfa_exact import DFAExactBackend

#: every backend :func:`pyjuice.constraints.compile` can use, by name, in the order it tries them when none is named
BACKENDS = {cls.name: cls for cls in (DFAExactBackend,)}
