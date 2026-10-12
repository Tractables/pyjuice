"""
What the ``dfa_exact`` backend runs for every query (named as :mod:`pyjuice.queries`): the queries (:mod:`.marginal`,
:mod:`.conditional`), the plan of which values a query computes (:mod:`.plans`, built by :mod:`.planners`), and the
passes that compute them (:mod:`.forward`, :mod:`.top_down`).
"""
