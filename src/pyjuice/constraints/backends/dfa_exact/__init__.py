"""
The ``dfa_exact`` backend: exact inference under a constraint with a finite-state form, by lifting -- the
automaton's states become extra columns of the PC's node values.
"""

from .backend import DFAExactBackend
