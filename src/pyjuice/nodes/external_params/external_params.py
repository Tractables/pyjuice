from __future__ import annotations

import torch
from typing import Any, Optional, Sequence, Tuple


class ExternalSumParams():
    """
    Base class of the *external parameterizations* of a sum node.

    An `ExternalSumParams` describes how per-sample, externally supplied tensors modify the effective
    parameters of an :class:`~pyjuice.nodes.ExternalParamsSumNodes`. It is the sum-node counterpart of
    :class:`~pyjuice.nodes.distributions.Distribution` for input nodes: the node holds the descriptor,
    the descriptor owns the semantics (the tensor layout and the math the layer applies), and the
    compiled circuit groups nodes into layers by :func:`get_signature`, so a mode never shares a
    compiled layer -- or a kernel -- with a different mode.

    The descriptor is pure configuration: it holds no tensors and no per-call state. The external
    tensors are owned by the caller, are **not** EM-trained by the node, and are supplied per call

    .. code-block:: python

        lls = pc(x, sum_external_params = {ns: tensors})

    with the matching per-sample gradients written back through

    .. code-block:: python

        pc.backward(x, sum_external_params_grad = {ns: grad_tensors})

    They are validated against :func:`tensor_shapes` during the forward pass. The shared parameters
    keep training through the ordinary EM / gradient path, unchanged.
    """

    def get_signature(self) -> str:
        """
        Get the signature of the current parameterization.

        Sum nodes are grouped into layers by (block size, signature), and one layer compiles a single
        set of kernels, so two nodes may only share a layer if their signatures match. Any setting
        that the kernels specialize on must therefore appear in the signature.
        """
        raise NotImplementedError()

    def validate_ns(self, ns) -> None:
        """
        Check that `ns` is compatible with this parameterization. Called once, at node construction.

        :param ns: the sum nodes carrying this parameterization
        :type ns: ExternalParamsSumNodes
        """
        pass

    def tensor_shapes(self, ns, batch_size: int) -> Tuple[Tuple[int,...],...]:
        """
        Shapes of the external tensors this parameterization consumes, for `ns` at `batch_size`, in
        the order the tensors are supplied. This is the layout the caller must produce and against
        which the forward pass validates.
        """
        raise NotImplementedError()

    def _get_constructor(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, ExternalSumParams) and self.get_signature() == other.get_signature()

    def __hash__(self):
        return hash(self.get_signature())

    def __repr__(self):
        return self.get_signature()
