"""
What the ``dfa_exact`` backend builds once per (constraint, PC) pair, from the constraint's automaton and the PC's
structure alone: the automaton's layout over the positions (:mod:`.layout`), the PC's rows (:mod:`.tables`), and the
tables the forward and top-down passes read (:mod:`.program`, :mod:`.top_down_program`).
"""
