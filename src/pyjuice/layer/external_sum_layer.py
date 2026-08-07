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

# The flat staging buffer the views point into, supplied by the PC so that kernels spanning several
# nodes can address it directly instead of taking one pointer per node
EXTERNAL_PARAMS_BUFFER_KWARG = "sum_external_params_buffer"
EXTERNAL_PARAMS_GRAD_BUFFER_KWARG = "sum_external_params_grad_buffer"


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


class StagedExternalParams(dict):
    """
    An `{ns: tensors}` mapping whose tensors are views into the PC's external-parameter staging
    buffer, i.e. already validated, contiguous and correctly placed.

    Layers use the type as the marker that re-validation is unnecessary -- the per-call checks would
    otherwise run over every tensor of every node on every forward and backward.
    """
    pass


def validate_external_tensors(ns, external_params, tensors: Any, batch_size: int, device,
                              require_contiguous: bool = True) -> Tuple:
    """
    Check externally supplied tensors against the layout the parameterization declares.

    Driven entirely by `external_params.tensor_shapes`, so it holds for any parameterization without
    the layer knowing what the tensors mean.

    :param require_contiguous: whether the tensors must already be contiguous. False when they are
                               about to be *copied* into the staging buffer, which handles arbitrary
                               strides -- so the caller may hand over slices of whatever their own
                               head produced without paying for a `.contiguous()` per node.
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
        if require_contiguous:
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

    Ancestral sampling is handed over the same way, through :func:`sample_layer`, with the one
    difference that the descriptor REPLACES the standard draw rather than correcting it -- a
    categorical draw already made under the shared parameters cannot be corrected after the fact.

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

        self._compile_partition_edge_block_ids()

    def _compile_partition_edge_block_ids(self) -> None:
        """
        Build, per FORWARD PARTITION, the table that ties the external tensors to `nids` / `cids`.

        The kernels work in the compiled partition layout: a row `ng` of `nids` is a node block, and
        `cids[ng, e]` are its child nodes. The external tensors are indexed by edge block instead, so
        each row needs to know which edge blocks it owns and where their factors live:

            `ext_slots[s][pid][ng, j]` -- offset of storage slot `s` of row `ng`'s j-th incident edge
            block, in PER-BATCH units (multiply by the batch size at launch), `-1` where a row has
            fewer than `max_n_eblks` edge blocks.

        One table per slot the parameterization declares, so a descriptor with one tensor per edge
        block (a scalar gate) and one with two (low-rank factors) are served by the same machinery.

        Offsets are per-batch units precisely so this table is independent of the batch size, and can
        therefore be compiled once rather than rebuilt whenever the batch changes.

        Edge block `j` occupies compiled edge slots `[j * ch_block_size, (j+1) * ch_block_size)` of
        the row, which is what makes `cids` usable directly by the external kernels; that contiguity
        is asserted below rather than assumed.
        """

        # The tables address the buffer as `(index + within-slot offset) * batch_size + b`, which is
        # only meaningful when the storage layout is batch-innermost. A parameterization that keeps
        # the caller's order simply does not get them -- it is served by its own (torch) path.
        self.ext_slots, self.ext_max_n_eblks, self.ext_unit_bases = None, None, None

        for ns in self.nodes:
            if any([shape[-1] != 1 for shape in ns.external_params.storage_shapes(ns, 1)]):
                return None

        # Per-batch base of every node's storage slots, in the staging buffer's node order. Taking the
        # shapes at `batch_size = 1` gives sizes in per-batch units directly.
        # Keyed by STORAGE OWNER, not by node -- and not by "is the owner one of this layer's nodes",
        # which is the subtly wrong test: an owner may live in a DIFFERENT layer (a tied chain puts the
        # source at one depth and its copies at others), and then two copies sharing this layer would
        # each advance the cursor and address different slots while the circuit gave them the same one.
        # That read past the end of the buffer, and the symptom was whatever the allocator happened to
        # be holding there -- correct-looking whenever a previous run had left the same factors behind.
        unit_base, ns2unit, owner2unit = 0, dict(), dict()
        for ns in self.nodes:
            owner = ns.external_params.storage_owner(ns)

            if owner in owner2unit:
                ns2unit[ns] = owner2unit[owner]
                continue

            slots = []
            for shape in ns.external_params.storage_shapes(ns, 1):
                slots.append(unit_base)
                unit_base += int(torch.tensor(shape).prod())

            owner2unit[owner] = slots
            ns2unit[ns] = slots

        # Kept because a BACKWARD table is indexed by (parent block, child block) rather than by the
        # forward's (row, edge slot), so it cannot be read off `ext_slots`. With these bases a
        # parameterization can pair `storage_offsets(ns)` with `par_nids` / `ch_eids` directly and get
        # the same per-batch offsets `ext_slots` holds -- one number per ns per slot, in `self.nodes` order.
        self.ext_unit_bases = [ns2unit[ns] for ns in self.nodes]

        max_n_eblks = max([ns_info.max_n_eblks for ns_info in self.external_node_infos])

        # NOT `n_slots`: that name is taken below for the child-block count within `cids`, and shadowing
        # it here silently truncated the table list.
        n_ext_slots = len(self.nodes[0].external_params.storage_shapes(self.nodes[0], 1))
        assert all(len(ns.external_params.storage_shapes(ns, 1)) == n_ext_slots for ns in self.nodes), \
            "All nodes of a layer must declare the same number of external storage slots."

        tables = [[] for _ in range(n_ext_slots)]
        for partition_id in range(self.num_fw_partitions):
            nids = self.partitioned_nids[partition_id].cpu()
            cids = self.partitioned_cids[partition_id].cpu()

            curr = [torch.full([nids.size(0), max_n_eblks], -1, dtype = torch.long)
                    for _ in range(n_ext_slots)]

            for ns_info in self.external_node_infos:
                ns = ns_info.ns
                block_size, ch_block_size = ns.block_size, ns.ch_block_size

                rows = torch.nonzero((nids >= ns_info.nid_start) & (nids < ns_info.nid_end),
                                     as_tuple = False).flatten()
                if rows.size(0) == 0:
                    continue

                nblock_ids = (nids[rows] - ns_info.nid_start) // block_size

                par_ptr = ns_info.par_ptr.cpu()
                starts = par_ptr[nblock_ids]
                counts = par_ptr[nblock_ids + 1] - starts

                # Edge block ids of each row, padded out to `max_n_eblks`
                all_eblks = ns_info.eblk_ids.cpu()
                slot_ids = torch.arange(0, max_n_eblks)[None,:]
                valid = slot_ids < counts[:,None]
                eblks = all_eblks[(starts[:,None] + slot_ids).clamp(max = all_eblks.size(0) - 1)]

                # `cids` must lay each edge block's children out contiguously, in edge-block order, so
                # that the kernel can derive the child of (j, c) as `cids[ng, j*ch_block_size + c]`
                n_slots = cids.size(1) // ch_block_size
                blocked = cids[rows,:n_slots * ch_block_size].reshape(-1, n_slots, ch_block_size)
                blk_valid = torch.arange(0, n_slots)[None,:] < counts[:,None]

                expected = blocked[:,:,:1] + torch.arange(0, ch_block_size)[None,None,:]
                assert torch.all(blocked[blk_valid] == expected[blk_valid]), \
                    "External sum parameters require each edge block's children to occupy a " \
                    "contiguous run of `cids`."
                # CLIPPED to this partition's width. `valid` is sized by the LAYER-wide `max_n_eblks`
                # while `blocked` is only as wide as the partition, so when the partitioner splits a
                # gated layer into partitions of differing width the mask and the tensor disagree and
                # this raises `IndexError` at compile time -- refusing shapes it was meant to validate.
                keep = min(n_slots, max_n_eblks)
                assert torch.all(blocked[:,:keep,0][valid[:,:keep]]
                                 == ns_info.ch_eids.cpu()[eblks][:,:keep][valid[:,:keep]]), \
                    "External sum parameters require `cids` to list edge blocks in `par_ptr` order."

                # Where each edge block's entry starts within its slot, in per-batch units. The
                # parameterization decides: the default layout is `[E, ...rest, B]`, one entry per edge
                # block in edge-block order, so the offset is just `eblk * stride`. A layout that stores
                # something OTHER than one contiguous entry per edge block -- a dense grid indexed by
                # (node block, child block), say -- supplies its own map, which is what keeps staging a
                # permute instead of a gather.
                offsets = ns.external_params.storage_offsets(ns)
                for slot in range(n_ext_slots):
                    off = offsets[slot].cpu()[eblks] if offsets is not None else None
                    if off is None:
                        shape = ns.external_params.storage_shapes(ns, 1)[slot]
                        stride = 1
                        for d in shape[1:-1]:
                            stride *= int(d)
                        off = eblks * stride

                    curr[slot][rows] = torch.where(valid, ns2unit[ns][slot] + off, -1)

            for slot in range(n_ext_slots):
                tables[slot].append(curr[slot])

        for slot in range(n_ext_slots):
            self.register_external_partition_buffers(f"slot{slot}", tables[slot])

        self.ext_slots = [getattr(self, f"ext_slot{slot}") for slot in range(n_ext_slots)]
        self.ext_max_n_eblks = max_n_eblks

    def register_external_partition_buffers(self, name: str, tensors: Sequence[torch.Tensor]) -> None:
        """
        Register one compile-time tensor per FORWARD PARTITION, exposed as `layer.ext_<name>[pid]`.

        The per-partition counterpart of :func:`register_external_buffers`: for tensors laid out like
        `nids` / `cids` rather than per `ns`.
        """
        assert getattr(self, f"ext_{name}", None) is None, \
            f"External buffer `{name}` is already registered."

        setattr(self, f"ext_{name}",
                FastParamList([nn.Parameter(tensor.contiguous(), requires_grad = False) for tensor in tensors]))

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

        # Tensors staged by the PC are views into its own buffer: already validated, contiguous and
        # correctly shaped, so re-checking them on every call is pure overhead
        pre_validated = isinstance(ns2tensors, StagedExternalParams)

        resolved = []
        for ns_info in self.external_node_infos:
            tensors = ns2tensors.get(ns_info.ns, None)
            if tensors is None:
                continue

            if not pre_validated:
                tensors = validate_external_tensors(ns_info.ns, self.external_params, tensors,
                                                    batch_size, device)

            resolved.append((ns_info, tensors))

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

        if isinstance(ns2grads, StagedExternalParams):
            return grad_tensors

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

        ns_tensors = self._resolve_external_tensors(kwargs, node_mars.size(1), node_mars.device)

        # Some parameterizations compute the node values themselves, in which case the standard forward
        # would be redundant work whose result is thrown away. It still has to run when nothing was
        # supplied for this layer, since then the layer IS a plain sum layer.
        if len(ns_tensors) == 0 or not self.external_params.replaces_shared_forward:
            super(ExternalParamsSumLayer, self).forward(
                node_mars, element_mars, params, propagation_alg = propagation_alg, **kwargs
            )

        if len(ns_tensors) == 0:
            return None

        self._assert_supported(propagation_alg, is_backward = False, **kwargs)

        self.external_params.forward_layer(
            self, ns_tensors, node_mars, element_mars, params,
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

        try:
            self.external_params.pre_backward_layer(
                self, ns_tensors, node_flows, element_flows, node_mars, element_mars, params,
                param_flows = param_flows, propagation_alg = propagation_alg, **kwargs
            )

            # Shared-parameter flows -- the standard kernels, untouched. Skipped entirely when the
            # parameterization owns the flows: its effective parameters vary per edge block, and the
            # standard kernels would sum the parents together before any correction could separate them.
            if not self.external_params.replaces_shared_backward:
                super(ExternalParamsSumLayer, self).backward(
                    node_flows, element_flows, node_mars, element_mars, params,
                    param_flows = param_flows, propagation_alg = propagation_alg, **kwargs
                )

            self.external_params.post_backward_layer(
                self, ns_tensors, ns_grad_tensors,
                node_flows, element_flows, node_mars, element_mars, params,
                param_flows = param_flows, propagation_alg = propagation_alg, **kwargs
            )
        finally:
            # A parameterization may redirect the standard backward's CUDA kernels at its own for the
            # duration of this call (see `SumLayer._ext_bw_ele_hook`). Left installed after a raise
            # they would make the NEXT backward -- possibly an ungated one -- fail or, worse, run a
            # gated kernel against a stale plan.
            self._ext_bw_ele_hook = None
            self._ext_bw_par_hook = None
            self._ext_bw_par_sb_hook = None
            self._ext_bw_par_triton_hook = None

        return None

    def sample_layer(self, node_mars: torch.Tensor, element_mars: torch.Tensor, params: torch.Tensor,
                     node_samples: torch.Tensor, element_samples: torch.Tensor,
                     ind_target: torch.Tensor, ind_n: torch.Tensor, ind_b: torch.Tensor,
                     conditional: bool = False, rnd = None, rnd_offset = 0,
                     **kwargs) -> bool:
        """
        Draw one child for every node of this layer the ancestral sampler has selected.

        Returns whether this layer handled the draw. `False` means no external tensors were supplied
        for it, in which case it IS a plain sum layer and the caller falls back to the
        shared-parameter kernel -- the same rule the forward and backward passes follow, so a layer
        is sampled under exactly the parameterization it would have been scored under.

        :note: `num_samples` plays the role the batch size plays elsewhere, and the external tensors
               are validated against it. Unconditionally that is the number of samples requested, so
               there is one gate per drawn sample; conditionally it is the batch of the forward pass
               being conditioned on.
        """

        num_samples = node_samples.size(1)
        ns_tensors = self._resolve_external_tensors(kwargs, num_samples, node_samples.device)

        if len(ns_tensors) == 0:
            return False

        # The compiled tables express offsets in PER-BATCH units and the kernels multiply by the
        # sample count, so tensors staged for a different batch would be read at addresses scaled by
        # the wrong stride -- in bounds, plausible, and someone else's gate. `_resolve_external_tensors`
        # skips validation for already-staged views, which is exactly the case this can arise in:
        # `pc._staged_external_params` is never cleared, so a conditional draw can reach views a much
        # older forward pass left behind.
        for ns_info, tensors in ns_tensors:
            shapes = self.external_params.storage_shapes(ns_info.ns, num_samples)
            for tensor, shape in zip(tensors, shapes):
                assert tuple(tensor.size()) == tuple(shape), \
                    f"External tensors staged for {ns_info.ns} are shaped {tuple(tensor.size())}, but " \
                    f"drawing {num_samples} samples needs {tuple(shape)}. They were staged by a " \
                    f"forward pass at a different batch size."

        self.external_params.sample_layer(
            self, ns_tensors, node_mars, element_mars, params, node_samples, element_samples,
            ind_target, ind_n, ind_b, conditional = conditional,
            rnd = rnd, rnd_offset = rnd_offset, **kwargs
        )

        return True

    def __repr__(self):
        return f"ExternalParamsSumLayer(nid_range=({self._layer_nid_range[0]}, {self._layer_nid_range[1]}), " \
               f"num_nodes={self.num_nodes}, num_edges={self.num_edges}, " \
               f"external_params={self.external_signature})"
