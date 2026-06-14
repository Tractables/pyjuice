pyjuice.queries
===============

.. currentmodule:: pyjuice.queries

Probabilistic queries over a compiled :doc:`TensorCircuit <tensorcircuit>`. These cover the common
inference tasks: marginals, conditionals, and sampling. All of them are thin wrappers around
:func:`query`, which exposes the underlying forward/backward machinery for custom queries.

.. autosummary::
    :toctree: generated
    :nosignatures:

    marginal
    conditional
    sample
    query
