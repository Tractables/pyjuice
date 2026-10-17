"""
The language layer's algorithms (no constraint classes): the DFA kernels :mod:`.dfa` (breadth-first orders,
minimization, products), and the byte-level automata text-level constraints compile through (:mod:`.byte_level`).
Nothing here is imported eagerly (:mod:`.byte_level` imports the DFA class, which imports :mod:`.dfa`).
"""
