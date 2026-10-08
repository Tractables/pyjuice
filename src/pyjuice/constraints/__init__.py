"""
Constraints on a PC's variables, for exact (and controlled approximate) constrained inference.

A constraint is a formal language over token ids; see :mod:`pyjuice.constraints.base`.
"""

from .base import Constraint, Matcher, And, Or, Not, Concat, OPTIONAL_CAPABILITIES
from .dfa import DFA, DFAMatcher
