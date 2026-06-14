pyjuice.nodes
=============

.. currentmodule:: pyjuice.nodes

Nodes
-----

.. autosummary::
    :toctree: generated
    :nosignatures:

    CircuitNodes
    InputNodes
    ProdNodes
    SumNodes

Methods
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    foreach
    foldup_aggregate

Edge Constructors
-----------------

Helpers that build structured (block-diagonal or block-sparse) edge patterns, passed as the
``edge_ids`` argument of :func:`~pyjuice.summate`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    edge_constructors.block_diagonal_edge_constructor
    edge_constructors.block_sparse_rnd_blk_edge_constructor

Input Distributions
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    distributions.Bernoulli
    distributions.Categorical
    distributions.DiscreteLogistic
    distributions.Gaussian
    distributions.MaskedCategorical
    distributions.Literal
    distributions.Indicator
    distributions.SoftEvidenceCategorical
    distributions.SoftEvidenceIndicator
    distributions.External
    distributions.ExternProductCategorical
