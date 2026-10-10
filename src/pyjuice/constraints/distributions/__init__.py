"""
Input distributions under a constraint. A constrained backend needs two things from the distribution of an input
node, besides what its input layer already computes (the log-probability of an observed token):

* how many values it ranges over -- the constraint's tokens must be exactly those;
* for a token that is marginalized, the mass of every token class of the constraint's automaton:
  ``log sum_{v in c} p_n(v)``.

Both are computed here, one module per distribution, named as in :mod:`pyjuice.nodes.distributions`. Each module
provides

* ``num_values(dist) -> int``: the values (tokens) the distribution ranges over;
* ``class_masses(layer, token_class, num_classes) -> torch.Tensor``: ``[number of nodes of layer, num_classes]``,
  the log-mass of every class for every node of the input layer, the same for every sample (``token_class``:
  ``[num_values]``, each value's class).

Only the distributions in :data:`SUPPORTED` can be compiled under a constraint, matched by exact type: a subclass
may compute different probabilities, so it is refused until it has its own module.
"""

from typing import Optional

from pyjuice.nodes.distributions import Categorical

from . import categorical

#: the module of every supported input distribution, by exact type
SUPPORTED = {Categorical: categorical}


def lookup(dist) -> Optional[object]:
    """The module of ``dist``'s exact type in :data:`SUPPORTED`, or None."""
    return SUPPORTED.get(type(dist))
