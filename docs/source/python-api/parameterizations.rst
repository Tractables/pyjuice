pyjuice.parameterizations
=========================

.. currentmodule:: pyjuice.parameterizations

A *parameterization* is the rule a sum node's effective parameters follow: the shared, EM-trained parameters modified
by per-sample tensors -- the **external parameters** -- supplied at call time through
``pc(x, sum_external_params = {ns: tensors})``. A sum node is given one with
``pyjuice.summate(..., parameterization = BlockScale(...))``.

.. autosummary::
    :toctree: generated
    :nosignatures:

    SumParameterization
    BlockScale
    LowRank
