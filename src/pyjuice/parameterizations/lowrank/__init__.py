"""
:class:`~pyjuice.parameterizations.low_rank.LowRank` in parts: the class itself -- its definition, layout and
geometry -- stays in :mod:`..low_rank`, and its passes are here: the torch passes as mixins, one module
each (:mod:`.forward`, :mod:`.backward`, :mod:`.denominators`), and the virtual batch through the plain
sum-layer kernels (:mod:`.virtual`), its kernels in :mod:`.kernels`.
"""
