"""
:class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams` in parts: the class itself -- its definition,
layout and staging geometry -- stays in :mod:`..block_scale`, and its forward, backward, sampling and denominator flow
are mixins here, one module each (:mod:`.forward`, :mod:`.backward`, :mod:`.sample`, :mod:`.denominators`), as its
kernels are in :mod:`..kernels`.
"""
