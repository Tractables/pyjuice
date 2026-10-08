"""
What a constraint is: formal languages over token ids, independent of any PC.
"""

from .base import Constraint, Matcher, And, Or, Not, Concat, OPTIONAL_CAPABILITIES
from .dfa import DFA, DFAMatcher
