"""
Transition masses: for a missing token at a position, a column and one of its successors, the log-mass of the token
classes that lead there, from the class masses (:func:`transition_masses`). The products' ``input @ block`` and
``block @ input`` steps read them, so a missing token costs a gather per successor, not per class (see
:mod:`.prod`). Their tables (:class:`TransitionTables`) are built once per backend.

ENUMERATED CLASSES (see :class:`~pyjuice.constraints.distributions.TokenClasses`) have no class mass: where a kernel
read the mass of class ``c`` at row ``r`` it reads a CODE instead (:class:`EnumeratedTokens`): the class's stored
column, or ``-1 - e`` for the ``e``-th enumerated token live at the boundary, whose mass is its token's own
probability under the row's parameter row (``eprobs[prow[r], index]``, the enumerated tokens' probabilities, by
parameter row: a step's lanes read along one row, see :meth:`~..state.State.enumerated_probs`) times its per-sample
weight (``enum_logw``, see :class:`~..state.ClassMasses`) over the row's ``Z``.
"""

from . import tables, masses, signatures
from .tables import (EnumeratedTokens, TransitionTables, BoundaryGroups, boundary_groups, enumerated_tokens,
                     next_col_table, transition_stats, transition_tables)
from .masses import (transition_job, job_tile, job_tiles, enumerated_args, enum_logw_stride, transition_masses,
                     SampleMasses, sample_masses, sample_classes, TRANSITION_TILES, TRANSITION_PAIR_TILE)
