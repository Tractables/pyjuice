"""
Kernels of the dfa_exact backend. Every node's values live in the region of the lifted buffers its node group
owns (see :func:`~pyjuice.constraints.backends.dfa_exact.plan.buffer_layout`); a kernel finds a node's row from
the region's offset, row width and first row, and slot ``s`` of sample ``b`` in column ``b * slots + s``.
"""
