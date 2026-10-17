"""
The language layer's algorithms (no constraint classes): the DFA kernels :mod:`.dfa` (breadth-first orders,
minimization, products), the byte-level automata text-level constraints compile through (:mod:`.byte_level`), and
LTLf formulas to DFAs (:mod:`.ltlf`). Nothing here is imported eagerly (:mod:`.byte_level` and :mod:`.ltlf` import the
DFA class, which imports :mod:`.dfa`).
"""
