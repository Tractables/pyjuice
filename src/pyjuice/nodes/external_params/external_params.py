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

    #: Whether this parameterization computes the layer's node values ITSELF, making the standard
    #: sum-layer forward redundant. `False` -- the default -- means the standard forward runs first and
    #: the descriptor then corrects what it wrote, which is possible whenever the correction can be
    #: recovered from the shared total. A parameterization that reweights the per-edge-block partial
    #: sums cannot do that (the partials are gone once the standard kernel has summed them), so it sets
    #: this and takes over the whole computation.
    replaces_shared_forward: bool = False

    #: The backward counterpart. `False` -- the default -- means the standard element-flow and
    #: param-flow kernels run and the descriptor corrects what they wrote. A parameterization whose
    #: effective parameters vary PER EDGE BLOCK cannot be served that way: the standard kernels sum
    #: every parent of a child before the correction could be applied, and `sum_e w_e X_e` is not
    #: recoverable from `sum_e X_e`. Such a parameterization sets this and owns both flows, exactly as
    #: `replaces_shared_forward` makes it own the node values.
    replaces_shared_backward: bool = False

    #: Whether the backward writes `d LL / d(external parameters)` into the gradient buffer. `True` --
    #: the default -- is the complete parameterization. A descriptor that contributes the element and
    #: parameter flows but not yet its own gradient sets this to `False`, which leaves `pc.backward()`
    #: and the EM optimizers fully working and makes `pc.get_external_params_grad` say plainly that the
    #: gradient is unavailable, rather than returning the zeroed buffer as though it were an answer.
    computes_external_grads: bool = True

    def storage_owner(self, ns):
        """
        The node whose staging slots `ns` uses.

        Identity by default: every node gets its own slots. A parameterization may return a DIFFERENT
        node to make several nodes share one set of external tensors -- e.g. along a parameter-tying
        relation, so one factor pair serves every copy of a tied layer. Sharing changes two things for
        the caller: tensors are supplied once, for the owner, and the gradient returned for the owner is
        the SUM over every node that shares it.
        """
        return ns

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

    def storage_offsets(self, ns):
        """
        Where each edge block's entry begins within each storage slot, in per-batch units, as one
        `long` tensor per slot indexed by edge-block id -- or `None` for the default layout.

        The default is `[E, ...rest, B]`: one contiguous entry per edge block, in edge-block order, so
        the offset is `edge_block * prod(rest)` and the layer computes it itself. Override when storage
        is indexed by something other than the edge block, so that the compiled tables point at the
        right place and staging stays a copy rather than a gather.
        """
        return None

    def to_storage(self, ns, tensors: Tuple) -> Tuple:
        """
        Map the caller's tensors into :func:`storage_shapes`' layout, ready to be copied.

        The default applies :func:`storage_perm`, which covers any parameterization whose two layouts
        differ only in axis order. Override when they differ in SHAPE -- when the caller's layout is
        the one that reads naturally for the model and the storage layout is the one the kernels index,
        and getting from one to the other needs a gather rather than a transpose.

        :param tensors: the caller's tensors, already validated against :func:`tensor_shapes`.
        :returns: one tensor per storage slot, each matching :func:`storage_shapes`. They are copied,
                  so arbitrary strides are fine and a contiguous result is not required.
        """
        perm = self.storage_perm()
        if perm is None:
            return tuple(tensors)

        return tuple(tensor.permute(perm) for tensor in tensors)

    def from_storage(self, ns, tensors: Tuple) -> Tuple:
        """
        The inverse of :func:`to_storage`, used to hand gradients back in the caller's layout.

        Must invert `to_storage` exactly, so that a gradient lines up element-for-element with the
        tensor the caller supplied. Where `to_storage` gathers, this scatters -- and any entry of the
        caller's layout that storage has no slot for takes a zero gradient, which is correct: nothing
        in the model read it.
        """
        perm = self.storage_perm()
        if perm is None:
            return tuple(tensors)

        inverse = tuple(perm.index(axis) for axis in range(len(perm)))
        return tuple(tensor.permute(inverse) for tensor in tensors)

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

    def sample_layer(self, layer, ns_tensors, node_mars, element_mars, params, node_samples,
                     element_samples, rows, erows, seed_ptr, conditional: bool = False,
                     **kwargs) -> None:
        """
        Draw one child per live sample of every frontier ROW this layer owns, under the EFFECTIVE
        parameters.

        The rows come from the circuit's structural frontier layout
        (:mod:`pyjuice.queries.sampling.scope_plan`): `rows` are the `node_samples` rows this layer
        owns and `erows` where each one's drawn child is written. A row holds `-1` where its scope is
        not on that sample's path, so liveness is a mask rather than a shape.

        Everything else matches :func:`sample_layer_pairs`, which this replaces -- see there for what
        the draw has to be, and why the normalizer cancels out of it.
        """
        raise NotImplementedError(
            f"`{self.get_signature()}` does not implement ancestral sampling against the structural "
            f"frontier layout. Sampling it with the shared-parameter kernel would ignore the "
            f"per-sample parameters and quietly return samples from a different distribution than "
            f"the forward pass scores, so it is refused instead."
        )

    def sample_layer_pairs(self, layer, ns_tensors, node_mars, element_mars, params, node_samples,
                           element_samples, ind_target, ind_n, ind_b, conditional: bool = False,
                           rnd = None, rnd_offset = 0, **kwargs) -> None:
        """
        Draw one child per selected node of this layer, under the EFFECTIVE parameters.

        The top-down ancestral pass (:func:`pyjuice.queries.sample`) reaches this once per sum layer
        that was given external tensors, in place of the shared-parameter kernel -- which would
        otherwise draw from `theta_shared` and return samples from a different distribution than the
        forward pass scores. So this is the sampling counterpart of :func:`forward_layer`, and it
        owns the whole draw rather than correcting one: a normalized categorical distribution is not
        recoverable from a draw already made under different weights.

        The distribution to draw from is the node's effective conditional. Note that the normalizer
        cancels: for `theta_b[n,c] = w_b[n,c] / Z_b[n]`, drawing `c` in proportion to `w_b[n,c]`
        (times `exp(element_mars[c,b])` when conditioning) is the same draw, so a parameterization
        needs no normalizer from its forward pass -- only its own per-sample weights.

        Not implemented by default. A parameterization that leaves it that way makes
        :func:`pyjuice.queries.sample` raise rather than silently sample the shared parameters.

        :param ns_tensors: `[(ns_info, tensors), ...]` for the nodes that were given external
                           tensors, as in :func:`forward_layer`

        :param node_samples: `[scopes, num_samples]`, the sampler's frontier of selected node ids

        :param element_samples: `[scopes, num_samples]`, where the drawn child ids are written

        :param ind_target: flat index into `element_samples` at which each selected node's drawn
                           child belongs

        :param ind_n: index into `node_samples`' first axis of each selected node
        :param ind_b: sample (column) index of each selected node

        :param conditional: whether to condition on the evidence a forward pass left in
                            `element_mars`. Unconditionally the child of `n` is drawn in proportion
                            to the effective parameters alone; conditionally, to those times
                            `exp(element_mars[c,b])`.
        """
        raise NotImplementedError(
            f"`{self.get_signature()}` does not implement ancestral sampling. Sampling it with the "
            f"shared-parameter kernel would ignore the per-sample parameters entirely and quietly "
            f"return samples from a different distribution than the forward pass scores, so it is "
            f"refused instead. Implement `sample_layer`, or draw samples without supplying external "
            f"parameters, which samples the shared parameters and is what an ungated forward pass "
            f"also computes."
        )

    def _get_constructor(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, ExternalSumParams) and self.get_signature() == other.get_signature()

    def __hash__(self):
        return hash(self.get_signature())

    def __repr__(self):
        return self.get_signature()
