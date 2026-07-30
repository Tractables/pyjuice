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

    def storage_shapes(self, ns, batch_size) -> Tuple[Tuple[int,...],...]:
        """
        Shapes the external tensors are *stored* in inside the PC's staging buffer.

        The caller's layout and the kernels' preferred layout need not agree: staging copies, so it
        can transpose for free-ish while the caller still hands over whatever their own head produced.
        Defaults to storing exactly what the caller supplies.

        :note: every axis except the batch axis must be batch-independent, so that a slot's size is
               proportional to the batch size. That is what lets the compiled index tensors express
               offsets in per-batch units and stay valid across batch sizes.
        """
        return self.tensor_shapes(ns, batch_size)

    def storage_perm(self) -> Optional[Tuple[int,...]]:
        """
        Permutation taking the caller's axis order to :func:`storage_shapes`' order, or `None` when
        they agree. Applied to the caller's tensor during staging.
        """
        return None

    def compile(self, layer) -> None:
        """
        Compile the indices and other tensors this parameterization's kernels need, called once when
        `layer` is built.

        By this point the layer has built the generic per-`ns` metadata -- one
        :class:`~pyjuice.layer.external_sum_layer.ExternalNodeInfo` per `ns` in
        `layer.external_node_infos`, giving the mapping from the node's `edge_ids` columns to global
        node and element ids -- which is usually the starting point for anything further.

        Register whatever is derived through the layer, not on `self`:

        .. code-block:: python

            def compile(self, layer):
                tables = [build_table(ns_info) for ns_info in layer.external_node_infos]
                layer.register_external_buffers("edge_table", tables)   # -> ns_info.edge_table

        :func:`~pyjuice.layer.ExternalParamsSumLayer.register_external_buffers` takes one tensor per
        `ns` and exposes it as an attribute of that `ns_info`;
        :func:`~pyjuice.layer.ExternalParamsSumLayer.register_external_buffer` takes a single
        layer-wide tensor. Either way the layer owns the storage, so `.to(device)` moves it and
        `state_dict` sees it. Non-tensor state (caches, autotuned choices) can be set as a plain
        attribute on `layer`.

        :note: do NOT keep compiled state on the descriptor. One descriptor instance is shared by
               every node constructed with it -- including tied duplicates, which live in *different*
               layers -- so per-layer state stored on `self` would be overwritten by whichever layer
               compiles last. The descriptor is stateless configuration.

        :param layer: the layer that compiled these nodes
        :type layer: ExternalParamsSumLayer
        """
        pass

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Turn the shared-parameter node values into the effective ones, in place.

        Called once per `ns` after the standard sum-layer forward has run, so on entry
        `node_mars[ns_info.nid_start:ns_info.nid_end]` holds the value each node takes under the
        SHARED parameters alone. On exit it must hold the value under the effective parameters. The
        shared kernels are not re-run and not modified, so a parameterization is responsible for
        expressing its effect as a correction to what they produced.

        :param layer: the layer being evaluated
        :type layer: ExternalParamsSumLayer

        :param ns_info: compiled metadata for the `ns` these tensors belong to
        :type ns_info: ExternalNodeInfo

        :param tensors: the validated external tensors supplied for `ns_info.ns`
        :type tensors: Tuple[torch.Tensor,...]
        """
        raise NotImplementedError()

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Apply the parameterization to a whole layer, once per forward pass.

        The default loops the layer's nodes and calls :func:`forward` per node, which is the simplest
        thing to implement. Override it when the kernels can span several nodes in one launch -- the
        compiled index tensors are laid out per FORWARD PARTITION, covering every node of the layer,
        so a partition-level launch needs no per-node arguments at all.

        :param ns_tensors: `[(ns_info, tensors), ...]` for the nodes that were given external tensors
        """
        for ns_info, tensors in ns_tensors:
            self.forward(layer, ns_info, tensors, node_mars, element_mars, params, **kwargs)

    def pre_backward(self, layer, ns_info, tensors, node_flows, element_flows, node_mars,
                     element_mars, params, **kwargs) -> None:
        """
        Prepare the buffers so that the *standard* sum-layer backward, run immediately afterwards and
        unmodified, computes the flows of the SHARED component of the parameters.

        Called once per `ns` before the standard backward. Anything changed here must be undone in
        :func:`post_backward`, since the buffers are shared with the rest of the circuit.
        """
        raise NotImplementedError()

    def pre_backward_layer(self, layer, ns_tensors, node_flows, element_flows, node_mars,
                           element_mars, params, **kwargs) -> None:
        """
        Layer-level counterpart of :func:`pre_backward`, mirroring :func:`forward_layer`.

        Defaults to looping over the layer's nodes; override when the whole layer can be prepared in
        one shot (the compiled tables span every node in it, so a per-node loop repeats work).
        """
        for ns_info, tensors in ns_tensors:
            self.pre_backward(layer, ns_info, tensors, node_flows, element_flows, node_mars,
                              element_mars, params, **kwargs)

    def post_backward_layer(self, layer, ns_tensors, ns_grad_tensors, node_flows, element_flows,
                            node_mars, element_mars, params, param_flows = None, **kwargs) -> None:
        """Layer-level counterpart of :func:`post_backward`."""
        for (ns_info, tensors), grad_tensors in zip(ns_tensors, ns_grad_tensors):
            self.post_backward(layer, ns_info, tensors, grad_tensors, node_flows, element_flows,
                               node_mars, element_mars, params, param_flows = param_flows, **kwargs)

    def post_backward(self, layer, ns_info, tensors, grad_tensors, node_flows, element_flows,
                      node_mars, element_mars, params, param_flows = None, **kwargs) -> None:
        """
        Add the external contribution to the child flows, write the per-sample gradients of the
        external tensors, and undo whatever :func:`pre_backward` changed.

        Called once per `ns` after the standard backward.

        :param grad_tensors: buffers to ACCUMULATE the per-sample gradients into, laid out exactly
                             like the external tensors, or `None` if the caller did not request
                             gradients for this `ns`. They are zeroed once per `pc.backward` before
                             any layer runs, so several nodes may share one buffer and have their
                             gradients summed into it.
        :type grad_tensors: Optional[Tuple[torch.Tensor,...]]
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
