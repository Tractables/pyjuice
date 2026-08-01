from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Sequence, Union

from .sum_nodes import SumNodes
from .external_params import ExternalSumParams

Tensor = Union[np.ndarray,torch.Tensor]


class ExternalParamsSumNodes(SumNodes):
    """
    A vector of sum nodes whose effective parameters are the shared (EM-trained) parameters modified
    by **per-sample tensors supplied externally** at call time. It is created by
    :code:`pyjuice.summate(..., external_params = <descriptor>)`.

    This is the single node class for every external parameterization; *how* the external tensors
    modify the parameters is owned by the `external_params` descriptor -- an
    :class:`~pyjuice.nodes.external_params.ExternalSumParams` such as
    :class:`~pyjuice.nodes.external_params.LowRankSumParams`. The arrangement mirrors input nodes,
    where one :class:`~pyjuice.nodes.InputNodes` class carries a `dist` descriptor, and it means a new
    mode adds a descriptor rather than a node class.

    The external tensors are passed per call and are never owned or EM-trained by the node:

    .. code-block:: python

        ns = juice.summate(ch, num_node_blocks = 1,
                           external_params = LowRankSumParams(rank = 16))
        pc = juice.compile(root_ns)

        lls = pc(x, sum_external_params = {ns: (U, V)})
        pc.backward(x, sum_external_params_grad = {ns: (dU, dV)})

    The shared parameters continue to train through the ordinary EM / gradient path, unchanged.

    :note: at compile time these nodes are grouped into their own layer, keyed by
           `external_params.get_signature()`, so the standard sum-layer kernels used by every other
           node are never branched on.

    :note: the external tensors are keyed by the `ns` **instance**. A tied duplicate needs its own
           entry; passing the same tensors for several of them shares one per-sample modification
           across all of them.

    :param num_node_blocks: number of node blocks
    :type num_node_blocks: int

    :param chs: sequence of child nodes
    :type chs: Sequence[CircuitNodes]

    :param edge_ids: a matrix of size [2, # edges] - every size-2 column vector [i,j] defines a set of
                     edges that fully connect the ith sum node block and the jth child node block
    :type edge_ids: Optional[Tensor]

    :param external_params: the external parameterization descriptor
    :type external_params: ExternalSumParams

    :param block_size: block size
    :type block_size: int
    """

    def __init__(self, num_node_blocks: int, chs: Sequence, edge_ids: Optional[Union[Tensor,Sequence[Tensor]]] = None,
                 params: Optional[Tensor] = None, zero_param_mask: Optional[Tensor] = None, block_size: int = 0,
                 external_params: Optional[ExternalSumParams] = None, **kwargs) -> None:

        assert external_params is not None, \
            "`ExternalParamsSumNodes` requires an `external_params` descriptor; use `pyjuice.summate(...)` " \
            "without it to construct a plain `SumNodes`."
        assert isinstance(external_params, ExternalSumParams), \
            f"`external_params` should be an `ExternalSumParams`, got {type(external_params)}."

        self.external_params = external_params

        super(ExternalParamsSumNodes, self).__init__(
            num_node_blocks, chs, edge_ids, params = params, zero_param_mask = zero_param_mask,
            block_size = block_size, **kwargs
        )

        self.external_params.validate_ns(self)

    def get_external_signature(self) -> str:
        """
        Signature of the external parameterization. Sum nodes are grouped into layers by
        (block size, external signature).
        """
        return self.external_params.get_signature()

    def set_source_ns(self, source_ns):
        super(ExternalParamsSumNodes, self).set_source_ns(source_ns)

        assert self.get_external_signature() == source_ns.get_external_signature(), \
            f"External parameterization of the source ns ({source_ns.get_external_signature()}) does " \
            f"not match that of self ({self.get_external_signature()})."

    def _construction_kwargs(self):
        # The descriptor is stateless configuration, so rebuilt nodes share it
        return {"external_params": self.external_params}

    def __repr__(self):
        scope_size = len(self.scope)
        return f"ExternalParamsSumNodes(num_node_blocks={self.num_node_blocks}, block_size={self.block_size}, " \
               f"num_chs={self.num_chs}, num_edges={self.num_edges}, scope_size={scope_size}, " \
               f"external_params={self.external_params})"
