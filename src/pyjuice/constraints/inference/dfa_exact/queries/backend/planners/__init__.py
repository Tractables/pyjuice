"""
How a query plan (:mod:`.plans`) is built fast. The Program's launch tables are flattened once into packed int64
arrays (:func:`_forward_tables`, :func:`_top_down_tables`), and numba planners decide, in one pass over runs of steps
of one interval or source, which values every launch, dense run and node block computes and in which mode
(:func:`_forward_modes`, :func:`_top_down_modes`). Their output becomes the plan's steps (:func:`_forward_steps`,
:func:`_top_down_steps`): index lists into the Program's fixed launch tables (:class:`IndexLists`,
:func:`cut_launch`), copied to the device together.
"""

from . import cuts, forward, top_down
