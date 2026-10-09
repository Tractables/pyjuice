"""
Kernels of the lifted backend. Every node's values live in the region of the lifted buffers its node group
owns (see :func:`~pyjuice.constraints.backends.lifted.plan.buffer_layout`); a kernel finds a node's row from
the region's offset, row width and first row, and slot ``s`` of sample ``b`` in column ``s * B + b``.
"""
