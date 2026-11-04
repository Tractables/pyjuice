"""
External sum parameterizations under a constraint, in the dfa_exact backend (:mod:`pyjuice.nodes.external_params`):
sum node groups whose weights a query's per-sample inputs change (``pc(x, sum_external_params = ...)``), as
:mod:`pyjuice.constraints.distributions` are the input nodes'. A decoder takes those inputs for its whole session (see
:meth:`~pyjuice.constraints.ConstrainedCircuit.decoder`), with the library's own rules: keyed by node group, or by a
group registered with ``pc.register_external_params_group``; a sum layer given none is a plain sum layer.

One module per parameterization, named as in :mod:`pyjuice.nodes.external_params`. Each provides

* ``refusal(ns) -> Optional[str]``: why the backend cannot serve one of its node groups (None if it can);
* ``cells(pc)``: where every node group's inputs are in a session's row of them, one row per sample;
* ``edge_tables(cells, layer, nids, cids, elem_first, dev)``: what every compiled edge of one of its sum layers reads
  of them (the layer's node blocks ``nids`` and children ``cids``, as the backend's sum kernels take them);
* ``session_inputs(pc, cells, value)``: a session's ``sum_external_params``, checked against the PC, in its rows;
* ``log_normalizers(pc, cells, inputs)``: every node's ``log Z`` of every sample;

and the helpers the backend's lifted sum kernels call where a layer has inputs (see
:mod:`..kernels.sum`).

Only the parameterizations in :data:`SUPPORTED` can be compiled under a constraint, matched by exact type.
"""

from typing import Optional

from pyjuice.nodes.external_params import BlockScaleSumParams

from . import block_scale

#: the module of every supported parameterization, by exact type
SUPPORTED = {BlockScaleSumParams: block_scale}


def lookup(external_params) -> Optional[object]:
    """The module of ``external_params``' exact type in :data:`SUPPORTED`, or None."""
    return SUPPORTED.get(type(external_params))
