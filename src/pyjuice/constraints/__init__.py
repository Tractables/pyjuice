"""
Constraints on a PC's variables, for exact (and controlled approximate) constrained inference.

A constraint is a formal language over token ids; see :mod:`pyjuice.constraints.language.base`.
The package is conventionally imported as ``jc``::

    import pyjuice as juice
    import pyjuice.constraints as jc

    pc = juice.compile(juice.structures.HMM(seq_length = 12, num_latents = 64, num_emits = V))

    # the tokens 3, 7 appear in a row, and token 5 never appears
    c = jc.DFA.contains([[3, 7]], vocab_size = V) & ~jc.DFA.contains([[5]], vocab_size = V)
    cc = jc.compile(c, pc)      # depends only on pc's structure, so it survives parameter updates
"""

from .language import Constraint, Matcher, And, Or, Not, Concat, OPTIONAL_CAPABILITIES, DFA, DFAMatcher
from .compiler import compile, ConstraintCompileError
from .compiled import CompiledConstraint
