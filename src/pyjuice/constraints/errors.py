"""
Errors of :mod:`pyjuice.constraints`, in a module of their own: the compiler and every backend raise them.
"""


class ConstraintCompileError(ValueError):
    """Raised when a constraint cannot be compiled against a PC. The message lists every reason."""
