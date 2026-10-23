"""
Decoder sessions: queries of a circuit that reuse, between calls, the values the evidence left unchanged. The planner
(:mod:`.plan`) decides, from which variables a call changed, which node groups it recomputes and which cached values
it reads.
"""

from .plan import ConditionalPlan, MarginalPlan, Planner, Structure
