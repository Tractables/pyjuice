"""
:class:`~pyjuice.nodes.distributions.softevi_categorical.SoftEvidenceCategorical` in parts: the class stays in
:mod:`..softevi_categorical`, its Triton kernels are a mixin here (:mod:`.kernels`), and the host code that launches
them is split by what it serves -- :mod:`.forward`, :mod:`.backward`, the dense index-driven path both use
(:mod:`.dense_index`), the candidate order (:mod:`.candidates`) and what they all share (:mod:`.common`).
"""
