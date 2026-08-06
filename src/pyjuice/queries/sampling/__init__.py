"""
The per-layer machinery behind :func:`pyjuice.queries.sample`.

`sample.py` owns the top-down ancestral pass -- which layer runs when, and where each drawn id is
stored -- and this package owns everything below that: the frontier bookkeeping and the kernels that
draw a child of a sum node or expand the children of a product node.

Everything here works on the SHARED parameters. A sum layer whose parameters are modified per sample
is served by its own parameterization instead (see
:func:`pyjuice.nodes.external_params.ExternalSumParams.sample_layer`), which keeps this package free
of any knowledge of what parameterizations exist.
"""

from .frontier import assign_cids_ind_target, assign_nids_ind_target, push_non_neg_ones_to_front
from .sum_layer import sample_sum_layer, sample_sum_layer_kernel
from .prod_layer import count_prod_nch, sample_prod_layer
