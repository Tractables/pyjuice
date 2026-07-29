from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pyjuice.nodes import ExternalParamsSumNodes
from pyjuice.utils.parameter_list import FastParamList
from .sum_layer import SumLayer


# Forward kwarg carrying the per-sample external tensors, keyed by `ns` instance:
#   `pc(x, sum_external_params = {ns: tensors})`
EXTERNAL_PARAMS_KWARG = "sum_external_params"

# Backward kwarg carrying the buffers the per-sample gradients are accumulated into:
#   `pc.backward(x, sum_external_params_grad = {ns: grad_tensors})`
EXTERNAL_PARAMS_GRAD_KWARG = "sum_external_params_grad"


class ExternalNodeInfo():
    """
    Compiled, per-`ns` metadata for the external-parameter kernels.

    A `SumLayer` compiles its nodes into *partitions* keyed by edge count, which is the layout the
    shared-parameter kernels want but not the one an external parameterization is indexed by: the
    caller supplies tensors laid out against the node's own `edge_ids`. This carries the translation.
    An edge block here is one column of `ns.edge_ids`, i.e. one `block_size x ch_block_size` tile of
    shared parameters, and axis order matches `ns.edge_ids` exactly, so external tensor axis `e`
    always refers to `ns.edge_ids[:, e]`.

    This is also where a parameterization's *own* compile-time tensors are reached from: whatever it
    registers through :func:`ExternalParamsSumLayer.register_external_buffers` becomes an attribute
    here alongside the generic ones, so a kernel only ever needs the `ns_info` it is handed.

    Attributes:
        `ns`:               the nodes this describes
        `ns_idx`:           index of `ns` within `layer.nodes`
        `nid_start`:        global node id of the node block `ns.edge_ids[0] == 0`
        `nid_end`:          one past the global node id of this `ns`'s last node
        `num_node_blocks`:  number of node blocks of `ns`
        `num_edge_blocks`:  number of edge blocks of `ns` (`ns.edge_ids.size(1)`)
        `block_size`:       number of nodes per node block
        `ch_block_size`:    number of child nodes per child block
        `max_n_eblks`:      largest number of edge blocks incident to one node block

    Registered buffers (generic, always present):
        `par_nids`:         [num_edge_blocks] global node id of the first node of each edge block's
                            parent block
        `ch_eids`:          [num_edge_blocks] global element id of the first element of each edge
                            block's child block
        `eblk_ids`:         [num_edge_blocks] edge block ids, grouped by parent block
        `par_ptr`:          [num_node_blocks + 1] offsets into `eblk_ids`, so the edge blocks of node
                            block `i` are `eblk_ids[par_ptr[i]:par_ptr[i+1]]`
    """

    def __init__(self, ns, ns_idx: int) -> None:

        self.ns = ns
        self.ns_idx = ns_idx

        self.nid_start = ns._output_ind_range[0]
        self.nid_end = ns._output_ind_range[1]

        self.num_node_blocks = ns.num_node_blocks
        self.num_edge_blocks = ns.edge_ids.size(1)
        self.block_size = ns.block_size
        self.ch_block_size = ns.ch_block_size

        self.max_n_eblks = int(torch.bincount(ns.edge_ids[0,:], minlength = ns.num_node_blocks).max())

        # Names of the buffers attached by `ExternalParamsSumLayer.register_external_buffers`
        self.buffer_names = []

    def __repr__(self):
        return f"ExternalNodeInfo(nid_range=({self.nid_start}, {self.nid_end}), " \
               f"num_node_blocks={self.num_node_blocks}, num_edge_blocks={self.num_edge_blocks}, " \
               f"block_size={self.block_size}, ch_block_size={self.ch_block_size}, " \
               f"buffers={self.buffer_names})"


def validate_external_tensors(ns, external_params, tensors: Any, batch_size: int, device) -> Tuple:
    """
    Check externally supplied tensors against the layout the parameterization declares.

    Driven entirely by `external_params.tensor_shapes`, so it holds for any parameterization without
    the layer knowing what the tensors mean.
    """
    shapes = external_params.tensor_shapes(ns, batch_size)

    if torch.is_tensor(tensors):
        tensors = (tensors,)

    assert isinstance(tensors, (tuple, list)), \
        f"External parameters of `{external_params.get_signature()}` should be given as a tuple of " \
        f"{len(shapes)} tensors, got {type(tensors)}."
    assert len(tensors) == len(shapes), \
        f"`{external_params.get_signature()}` expects {len(shapes)} external tensors, got {len(tensors)}."

    for idx, (tensor, shape) in enumerate(zip(tensors, shapes)):
        assert torch.is_tensor(tensor), f"External tensor {idx} should be a `torch.Tensor`."
        assert tuple(tensor.size()) == tuple(shape), \
            f"External tensor {idx} should be of shape {tuple(shape)}, got {tuple(tensor.size())}."
        assert tensor.dtype == torch.float32, \
            f"External tensor {idx} should be of dtype `torch.float32`, got {tensor.dtype}."
        assert tensor.is_contiguous(), f"External tensor {idx} should be contiguous."
        assert tensor.device == device, \
            f"External tensor {idx} is on {tensor.device}, but the PC's buffers are on {device}."

    return tuple(tensors)


class ExternalParamsSumLayer(SumLayer):
    """
    A sum layer whose effective parameters are the shared parameters modified by per-sample tensors
    supplied externally at call time.

    Compilation is inherited from :class:`~pyjuice.layer.SumLayer` unchanged -- the edge partitions,
    `nids` / `cids` / `pids` and the shared-parameter kernels are exactly those of a standard sum
    layer. What this class adds is the translation between the compiled layout and the caller's
    per-`ns` layout (:class:`ExternalNodeInfo`), and the hooks that hand control to the node's
    :class:`~pyjuice.nodes.external_params.ExternalSumParams` descriptor at the three points where an
    external parameterization has to act:

    * after the shared forward, to turn the shared-parameter node values into the effective ones;
    * before the shared backward, to put `node_mars` into whatever form makes the *standard* kernels
      compute the shared component's flows; and
    * after it, to add the external contribution to the child flows, write the per-sample gradients,
      and undo anything the pre-hook changed.

    All the nodes of one layer share an external signature (that is what compilation groups them by),
    so exactly one descriptor -- and one set of kernels -- applies to the whole layer. The standard
    sum-layer kernels are never branched on.

    When no external tensors are supplied for any of this layer's nodes, the layer behaves exactly as
    a plain `SumLayer`: the hooks are skipped entirely.
    """

    def __init__(self, nodes: Sequence[ExternalParamsSumNodes], *args, **kwargs) -> None:

        super(ExternalParamsSumLayer, self).__init__(nodes, *args, **kwargs)

        for ns in self.nodes:
            assert isinstance(ns, ExternalParamsSumNodes), \
                f"`ExternalParamsSumLayer` only accepts `ExternalParamsSumNodes`, got {type(ns)}."

        signatures = set([ns.get_external_signature() for ns in self.nodes])
        assert len(signatures) == 1, \
            f"All nodes of an `ExternalParamsSumLayer` must share one external signature, got {signatures}."

        # One descriptor governs the whole layer. It is stateless configuration, and the signature it
        # is grouped by covers everything the kernels specialize on, so any node's descriptor will do.
        self.external_params = self.nodes[0].external_params
        self.external_signature = signatures.pop()

        self._compile_external_node_info()

        # Compile-time preparation specific to the parameterization. It registers whatever tensors its
        # kernels need through `register_external_buffers` / `register_external_buffer`, so they are
        # owned and moved by this layer rather than by the (shared, stateless) descriptor.
        self.external_params.compile(self)

    def register_external_buffers(self, name: str, tensors: Sequence[torch.Tensor]) -> None:
        """
        Register one compile-time tensor per `ns`, in `self.external_node_infos` order, under `name`.

        The tensors are stored on the layer, so `nn.Module.to(device)` moves them and `state_dict`
        sees them, and each is exposed as `ns_info.<name>` -- which is where a kernel reads it from.

        A parameterization must register its compiled tensors this way rather than keeping them on
        its descriptor: one descriptor instance is shared by every node built with it, tied duplicates
        across *different layers* included (see `CircuitNodes._construction_kwargs`), so per-layer
        state on the descriptor would be overwritten by the next layer that compiles. The descriptor
        stays stateless configuration; the layer owns the storage.

        :param name: attribute name to expose on each `ExternalNodeInfo`
        :type name: str

        :param tensors: one tensor per `ns`, in `self.nodes` order
        :type tensors: Sequence[torch.Tensor]
        """
        assert len(tensors) == len(self.external_node_infos), \
            f"`register_external_buffers` expects one tensor per ns ({len(self.external_node_infos)}), " \
            f"got {len(tensors)}."
        assert not hasattr(self, f"ext_{name}"), f"External buffer `{name}` is already registered."

        param_list = FastParamList([nn.Parameter(tensor.contiguous(), requires_grad = False) for tensor in tensors])
        setattr(self, f"ext_{name}", param_list)

        # `nn.Module._apply` rewrites `param.data` in place, keeping the `Parameter` object identity,
        # so the reference held by `ns_info` follows the layer across devices.
        for ns_info in self.external_node_infos:
            setattr(ns_info, name, param_list[ns_info.ns_idx])
            ns_info.buffer_names.append(name)

    def register_external_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """
        Register a single layer-wide compile-time tensor, exposed as `layer.ext_<name>`.

        Use this for data that spans the whole layer rather than one `ns`; per-`ns` data belongs in
        :func:`register_external_buffers`. Non-tensor state (caches, autotuned choices) can simply be
        set as a plain attribute on the layer.
        """
        assert not hasattr(self, f"ext_{name}"), f"External buffer `{name}` is already registered."

        setattr(self, f"ext_{name}", nn.Parameter(tensor.contiguous(), requires_grad = False))

    def _compile_external_node_info(self) -> None:
        """
        Build, for every `ns` in the layer, the mapping from its `edge_ids` columns to global node and
        element ids, plus the grouping of edge blocks by parent node block.
        """

        self.external_node_infos = [ExternalNodeInfo(ns, ns_idx) for ns_idx, ns in enumerate(self.nodes)]

        par_nids, ch_eids, eblk_ids, par_ptr = [], [], [], []

        for ns in self.nodes:
            edge_ids = ns.edge_ids

            # Global element id of the first element of every child block, indexed by the child block
            # id used in `edge_ids[1]` (children are concatenated in `ns.chs` order)
            ch_blk2eid = torch.zeros([ns.num_ch_node_blocks], dtype = torch.long)
            cum_blocks = 0
            for cs in ns.chs:
                assert cs.provided("_output_ind_range"), \
                    "Child nodes should have been compiled before the sum layer that consumes them."
                ch_blk2eid[cum_blocks:cum_blocks + cs.num_node_blocks] = \
                    cs._output_ind_range[0] + torch.arange(0, cs.num_node_blocks) * ns.ch_block_size
                cum_blocks += cs.num_node_blocks

            curr_par_nids = ns._output_ind_range[0] + edge_ids[0,:] * ns.block_size
            curr_ch_eids = ch_blk2eid[edge_ids[1,:]]

            # Group the edge blocks by parent node block, so a kernel handling node block `i` can walk
            # its incident edge blocks contiguously. `edge_ids` is not required to be sorted.
            curr_eblk_ids = torch.argsort(edge_ids[0,:], stable = True)
            counts = torch.bincount(edge_ids[0,:], minlength = ns.num_node_blocks)
            curr_par_ptr = torch.zeros([ns.num_node_blocks + 1], dtype = torch.long)
            curr_par_ptr[1:] = torch.cumsum(counts, dim = 0)

            par_nids.append(curr_par_nids)
            ch_eids.append(curr_ch_eids)
            eblk_ids.append(curr_eblk_ids)
            par_ptr.append(curr_par_ptr)

        self.register_external_buffers("par_nids", par_nids)
        self.register_external_buffers("ch_eids", ch_eids)
        self.register_external_buffers("eblk_ids", eblk_ids)
        self.register_external_buffers("par_ptr", par_ptr)

    @property
    def external_nodes(self) -> List[ExternalParamsSumNodes]:
        """
        The nodes of this layer that take external parameters, i.e. all of them.
        """
        return list(self.nodes)

    def _resolve_external_tensors(self, kwargs: dict, batch_size: int, device) -> List[Tuple[ExternalNodeInfo,Tuple]]:
        """
        Pick out and validate the external tensors supplied for this layer's nodes.

        Returns an empty list when none were given, in which case the layer runs as a plain sum layer.

        :note: `device` is taken from the buffers being evaluated rather than `self.device`, which
               `Layer` sets once at construction and never updates.
        """
        ns2tensors = kwargs.get(EXTERNAL_PARAMS_KWARG, None)
        if ns2tensors is None:
            return []

        assert isinstance(ns2tensors, dict), \
            f"`{EXTERNAL_PARAMS_KWARG}` should be a dict mapping nodes to their external tensors, " \
            f"got {type(ns2tensors)}."

        resolved = []
        for ns_info in self.external_node_infos:
            tensors = ns2tensors.get(ns_info.ns, None)
            if tensors is None:
                continue

            resolved.append((
                ns_info,
                validate_external_tensors(ns_info.ns, self.external_params, tensors, batch_size, device)
            ))

        return resolved

    def _resolve_external_grad_tensors(self, kwargs: dict, ns_info: ExternalNodeInfo,
                                       batch_size: int, device) -> Optional[Tuple]:
        """
        Pick out and validate the gradient buffers the per-sample gradients of `ns_info` are written
        into. Returns `None` when the caller did not ask for gradients for this `ns`.
        """
        ns2grads = kwargs.get(EXTERNAL_PARAMS_GRAD_KWARG, None)
        if ns2grads is None:
            return None

        assert isinstance(ns2grads, dict), \
            f"`{EXTERNAL_PARAMS_GRAD_KWARG}` should be a dict mapping nodes to their gradient " \
            f"buffers, got {type(ns2grads)}."

        grad_tensors = ns2grads.get(ns_info.ns, None)
        if grad_tensors is None:
            return None

        return validate_external_tensors(ns_info.ns, self.external_params, grad_tensors, batch_size, device)

    def _assert_supported(self, propagation_alg: str, is_backward: bool, **kwargs) -> None:
        """
        Guard the settings an external parameterization cannot currently be combined with. These are
        only checked when external tensors are actually supplied -- without them the layer is a plain
        sum layer and every setting is fair game.
        """
        assert propagation_alg == "LL", \
            f"External sum parameters are only supported for the 'LL' propagation algorithm, got " \
            f"'{propagation_alg}'."

        partition_ids = "bk_partition_local_ids" if is_backward else "fw_partition_local_ids"
        assert not self.provided(partition_ids), \
            "External sum parameters are not supported under partial evaluation."

        if is_backward:
            # `allow_modify_flows` overwrites `node_flows` with `log(flow) - node_mars` in place, so the
            # node flows the external gradients are built from would be gone by `post_backward`.
            assert not kwargs.get("allow_modify_flows", False), \
                "External sum parameters are not supported with `allow_modify_flows = True`, which " \
                "consumes `node_flows` in place."

            # `negate_pflows` belongs to the unnormalized partition pass of the gradient-based
            # optimizers; the external correction is defined against a normalized sum node.
            assert not kwargs.get("negate_pflows", False), \
                "External sum parameters are not supported with `negate_pflows = True`."

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                propagation_alg: str = "LL", **kwargs) -> None:
        """
        Forward pass. The shared-parameter term is the standard sum-layer forward, run unchanged; the
        descriptor then turns the values it wrote into the effective ones.
        """

        # Shared-parameter term -- the standard kernels, untouched
        super(ExternalParamsSumLayer, self).forward(
            node_mars, element_mars, params, propagation_alg = propagation_alg, **kwargs
        )

        ns_tensors = self._resolve_external_tensors(kwargs, node_mars.size(1), node_mars.device)
        if len(ns_tensors) == 0:
            # Nothing supplied for this layer -> it is a plain sum layer
            return None

        self._assert_supported(propagation_alg, is_backward = False, **kwargs)

        for ns_info, tensors in ns_tensors:
            self.external_params.forward(
                self, ns_info, tensors, node_mars, element_mars, params,
                propagation_alg = propagation_alg, **kwargs
            )

        return None

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor,
                 node_mars: torch.Tensor, element_mars: torch.Tensor,
                 params: torch.Tensor, param_flows: Optional[torch.Tensor] = None,
                 propagation_alg: str = "LL", **kwargs) -> None:
        """
        Backward pass. The descriptor is given a chance to prepare the buffers so that the *standard*
        sum-layer backward computes the shared component's element and parameter flows, and then to
        add the external contribution to the child flows and write the per-sample gradients.
        """

        ns_tensors = self._resolve_external_tensors(kwargs, node_mars.size(1), node_mars.device)

        if len(ns_tensors) == 0:
            # Nothing supplied for this layer -> it is a plain sum layer
            return super(ExternalParamsSumLayer, self).backward(
                node_flows, element_flows, node_mars, element_mars, params,
                param_flows = param_flows, propagation_alg = propagation_alg, **kwargs
            )

        self._assert_supported(propagation_alg, is_backward = True, **kwargs)

        # Resolve the gradient buffers BEFORE anything runs: `pre_backward` may put the shared buffers
        # into an intermediate form that only `post_backward` undoes, so a validation failure raised
        # between the two would leave the circuit's state inconsistent.
        ns_grad_tensors = [
            self._resolve_external_grad_tensors(kwargs, ns_info, node_mars.size(1), node_mars.device)
            for ns_info, _ in ns_tensors
        ]

        for ns_info, tensors in ns_tensors:
            self.external_params.pre_backward(
                self, ns_info, tensors, node_flows, element_flows, node_mars, element_mars, params,
                param_flows = param_flows, propagation_alg = propagation_alg, **kwargs
            )

        # Shared-parameter flows -- the standard kernels, untouched
        super(ExternalParamsSumLayer, self).backward(
            node_flows, element_flows, node_mars, element_mars, params,
            param_flows = param_flows, propagation_alg = propagation_alg, **kwargs
        )

        for (ns_info, tensors), grad_tensors in zip(ns_tensors, ns_grad_tensors):
            self.external_params.post_backward(
                self, ns_info, tensors, grad_tensors,
                node_flows, element_flows, node_mars, element_mars, params,
                param_flows = param_flows, propagation_alg = propagation_alg, **kwargs
            )

        return None

    def __repr__(self):
        return f"ExternalParamsSumLayer(nid_range=({self._layer_nid_range[0]}, {self._layer_nid_range[1]}), " \
               f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, " \
               f"external_params={self.external_signature})"
