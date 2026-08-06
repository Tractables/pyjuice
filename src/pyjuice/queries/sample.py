from __future__ import annotations

import torch
import numpy as np
import random
from typing import Optional

from pyjuice.layer.external_sum_layer import ExternalParamsSumLayer
from pyjuice.model import TensorCircuit

from .sampling import assign_cids_ind_target, assign_nids_ind_target, push_non_neg_ones_to_front, \
                      count_prod_nch, sample_prod_layer, sample_sum_layer


def sample(pc: TensorCircuit, num_samples: Optional[int] = None, conditional: bool = False,
           _sample_input_ns: bool = True, _do_calibration: bool = False, **kwargs):
    """
    Draw samples from a PC by performing a top-down ancestral sampling pass.

    For unconditional sampling, `num_samples` must be specified. For conditional sampling
    (`conditional = True`), run a forward pass on the evidence first (e.g., via :func:`marginal`); the
    sampler then reuses the cached `pc.node_mars` and draws one sample per example in that batch, so
    `num_samples` is ignored.

    :param pc: the input PC
    :type pc: TensorCircuit

    :param num_samples: number of samples to draw; required for unconditional sampling, ignored when `conditional = True`
    :type num_samples: Optional[int]

    :param conditional: whether to sample conditioned on the evidence cached by a preceding forward pass
    :type conditional: bool

    Per-sample external sum parameters are supplied through `sum_external_params`, in `kwargs`,
    exactly as they are to :func:`TensorCircuit.forward`

    .. code-block:: python

        samples = juice.queries.sample(pc, num_samples = 1024, sum_external_params = {ns: phi})

    with one gate per DRAWN SAMPLE, so their batch axis is `num_samples`. A sum layer that is given
    none is sampled from its shared parameters, which is what an ungated forward pass computes for
    it too. Under `conditional = True` they are instead taken from the forward pass that produced
    `pc.node_mars` -- `element_mars` was built under those gates, so the draw has to use them -- and
    passing them here only has to name the same nodes.

    :returns: a tensor of samples of size [num_samples, num_vars]
    :rtype: torch.Tensor
    """
    if not conditional:
        assert num_samples is not None, "`num_samples` should be specified when doing unconditioned sampling."
    else:
        num_samples = pc.node_mars.size(1) # Reuse the batch size

    root_ns = pc.root_ns
    assert root_ns._output_ind_range[1] - root_ns._output_ind_range[0] == 1, "It is ambiguous to sample from multi-head PCs."

    # Per-sample external sum parameters, if any -- handled by the PC's own staging, so that they are
    # fed exactly as they are to a forward or a backward pass and end up in the same buffer.
    #
    # Which of the two applies is not a matter of taste. An unconditional pass is a forward: nothing
    # has been staged, and the caller's tensors are copied in at `batch_size = num_samples`, one gate
    # per drawn sample. A conditional pass is a backward: it runs against the `node_mars` /
    # `element_mars` a forward pass left behind, which were built under THAT pass's gates, so the
    # draw has to use the same ones -- and taking them from the staging buffer makes using any
    # others impossible rather than merely discouraged.
    #
    # Both are no-ops when no external parameters were supplied, so an ordinary PC pays nothing.
    pc._check_external_params_kwargs(kwargs)
    if conditional:
        pc._resolve_backward_external_params(kwargs)
    else:
        pc._stage_external_params(kwargs, num_samples)

    if hasattr(pc, "_num_nscopes") and hasattr(pc, "_num_escopes"):
        num_nscopes = pc._num_nscopes
        num_escopes = pc._num_escopes
    else:
        num_nscopes = 0
        num_escopes = 0
        for layer_group in pc.layers(ret_layer_groups = True):
            curr_scopes = 0
            for layer in layer_group:
                curr_scopes += len(layer.scopes)

            if layer_group.is_input() or layer_group.is_sum():
                num_nscopes += curr_scopes
            else:
                assert layer_group.is_prod()
                num_escopes = max(num_escopes, curr_scopes)

        pc._num_nscopes = num_nscopes
        pc._num_escopes = num_escopes

    # Stores selected node indices by the sampler
    node_samples = torch.zeros([num_nscopes, num_samples], dtype = torch.long, device = pc.device)
    # Stores selected element indices by the sampler
    element_samples = torch.zeros([num_escopes, num_samples], dtype = torch.long, device = pc.device)
    # Pointers indicating how many elements are used in each column of `element_samples`
    element_pointers = np.zeros([num_samples], dtype = np.int64)

    # Initialize pointers to the root node
    node_samples[:,:] = -1
    node_samples[0,:] = root_ns._output_ind_range[0]

    # Iterate (backward) through layers
    for layer_id in range(len(pc.inner_layer_groups)-1, -1, -1):
        layer_group = pc.inner_layer_groups[layer_id]
        if layer_group.is_sum():
            # Initialize `element_samples` and `element_pointers`
            element_samples[:,:] = -1
            element_pointers[:] = 0

            # Iterate over sum layers in the current layer group
            for layer in layer_group:

                # Gather the indices to be processed
                lsid, leid = layer._layer_nid_range
                ind_n, ind_b = torch.where((node_samples >= lsid) & (node_samples < leid))

                # Pre-compute the target indices in `element_samples`
                # The sampled child node indices will be put into the indices presented in `ind_target`
                ind_target = np.zeros([ind_n.size(0)], dtype = np.int64)
                assign_cids_ind_target(ind_target, element_pointers, ind_b.detach().cpu().numpy(), num_samples)
                ind_target = torch.from_numpy(ind_target).to(pc.device)

                # In the case of conditional sampling, recompute to get the `element_mars`
                if conditional:
                    pc.inner_layer_groups[layer_id-1](pc.node_mars, pc.element_mars)

                # A layer whose parameters are modified per sample owns its own sampler, exactly as it
                # owns its forward and its backward -- the shared-parameter kernel below would draw
                # from a different distribution. It declines (returning `False`) when no external
                # tensors were supplied for it, in which case it IS a plain sum layer.
                handled = False
                if isinstance(layer, ExternalParamsSumLayer):
                    handled = layer.sample_layer(
                        pc.node_mars, pc.element_mars, pc.params, node_samples, element_samples,
                        ind_target, ind_n, ind_b, conditional = conditional, **kwargs
                    )

                # Sample child indices
                if not handled:
                    for partition_id in range(layer.num_fw_partitions):
                        nids = layer.partitioned_nids[partition_id]
                        cids = layer.partitioned_cids[partition_id]
                        pids = layer.partitioned_pids[partition_id]

                        sample_sum_layer(pc, layer, nids, cids, pids, pc.node_mars, pc.element_mars, pc.params,
                                         node_samples, element_samples, ind_target, ind_n, ind_b,
                                         layer.block_size, conditional, do_calibration = _do_calibration)

                # Clear completed nodes
                node_samples[ind_n, ind_b] = -1

        else:
            assert layer_group.is_prod()

            # Iterate over product layers in the current layer group
            for layer in layer_group:
                # Re-align `node_samples` by pushing all values to the front
                node_pointers = push_non_neg_ones_to_front(node_samples)

                # Gather the indices to be processed
                lsid, leid = layer._layer_nid_range
                ind_n, ind_b = torch.where((element_samples >= lsid) & (element_samples < leid))

                # Get the number of children for the selected sample indices
                ind_ch_count = torch.zeros_like(ind_n)
                ind_nids = torch.zeros_like(ind_n)
                ind_nid_offs = torch.zeros_like(ind_n)
                ind_mask = torch.zeros_like(ind_n)
                for partition_id in range(layer.num_fw_partitions):
                    nids = layer.partitioned_nids[partition_id]
                    cids = layer.partitioned_cids[partition_id]

                    count_prod_nch(layer, nids, cids, element_samples, ind_ch_count, ind_nids,
                                   ind_nid_offs, ind_mask, ind_n, ind_b, layer.block_size, partition_id)

                # Pre-compute the target indices in `node_samples`
                ind_target_sid = np.zeros([ind_n.size(0)], dtype = np.int64)
                ind_target_sid[1:] = ind_ch_count[:-1].cumsum(dim = 0).detach().cpu().numpy()
                ind_target = np.zeros([ind_ch_count.sum()], dtype = np.int64)
                assign_nids_ind_target(ind_target, ind_target_sid,
                                       node_pointers.detach().cpu().numpy(),
                                       ind_b.detach().cpu().numpy(), num_samples)
                ind_target_sid = torch.from_numpy(ind_target_sid).to(pc.device)
                ind_target = torch.from_numpy(ind_target).to(pc.device)

                # Store child indices
                for partition_id in range(layer.num_fw_partitions):
                    nids = layer.partitioned_nids[partition_id]
                    cids = layer.partitioned_cids[partition_id]

                    sample_prod_layer(layer, nids, cids, node_samples, element_samples, ind_target, ind_target_sid,
                                      ind_n, ind_b, ind_nids, ind_nid_offs, ind_mask, layer.block_size, partition_id)

    # Create tensor for the samples
    data_dtype = pc.input_layer_group[0].get_data_dtype()
    samples = torch.zeros([pc.num_vars, num_samples], dtype = data_dtype, device = pc.device)

    pc._init_buffer(name = "node_flows", shape = (pc.num_nodes, num_samples), set_value = 0.0)
    ind_n, ind_b = torch.where(node_samples != -1)
    ind_node = node_samples[ind_n, ind_b]
    pc.node_flows[ind_node, ind_b] = 1.0

    if _sample_input_ns:
        for layer in pc.input_layer_group:
            seed = random.randint(0, 2**31)
            layer.sample(samples, pc.node_flows, seed = seed, **kwargs)

        return samples.permute(1, 0).contiguous()
    else:
        # In this case, we do not explicitly sample input nodes
        push_non_neg_ones_to_front(node_samples)
        return node_samples
