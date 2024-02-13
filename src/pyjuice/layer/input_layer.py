from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import textwrap
import inspect
import random
from typing import Sequence, Dict
from triton.runtime.jit import JITFunction
from copy import deepcopy

from pyjuice.nodes import InputNodes
from pyjuice.utils.grad_fns import ReverseGrad
from pyjuice.utils import BitSet
from pyjuice.utils.source2fn import make_function_from_src
from pyjuice.utils.kernel_launcher import FastJITFunction
from .layer import Layer


class InputLayer(Layer, nn.Module):
    def __init__(self, nodes: Sequence[InputNodes], cum_nodes: int = 0, pc_num_vars: int = 0, max_tied_ns_per_parflow_block: int = 4) -> None:
        """
        Compiler flags:
        - `max_tied_ns_per_parflow_block`: the maximum number of tied nodes allowed in the backward pass. Setting to a larger value will
                                           lead to reduced memory overhead but might lead to additional computational burden due to conflicts
                                           in gradient accumulation.
        """

        nn.Module.__init__(self)
        Layer.__init__(self, nodes, disable_block_size_check = True)

        # Reorder input nodes such that for any tied nodes, its source nodes appear before them
        self.nodes = self._reorder_nodes(nodes)

        # Total number of variables
        self.pc_num_vars = pc_num_vars

        ## Parse input `nodes` ##
        node_vars = []
        node_sizes = []
        node_metadata = []
        layer_num_nodes = 0
        cum_params = 0
        cum_param_flows = 0
        cum_source_ns = 0
        dist_signature = None
        node2tiednodes = dict()
        for node_id, ns in enumerate(self.nodes):
            if dist_signature is None:
                dist_signature = ns.dist.get_signature()
            else:
                assert dist_signature == ns.dist.get_signature(), "Nodes of an InputLayer must have the same distribution type."

            node_vars.append(ns.scope.to_list())
            node_sizes.append(ns.num_nodes)
            node_metadata.append(ns.dist.get_metadata())

            ns._output_ind_range = (cum_nodes, cum_nodes + ns.num_nodes)
            cum_nodes += ns.num_nodes
            layer_num_nodes += ns.num_nodes

            if not ns.is_tied():
                cum_params += ns.num_nodes * ns.dist.num_parameters()
                ns._param_range = (cum_params - ns.num_nodes * ns.dist.num_parameters(), cum_params)

                cum_param_flows += ns.num_nodes * ns.dist.num_param_flows()
                ns._param_flow_range = (cum_param_flows - ns.num_nodes * ns.dist.num_param_flows(), cum_param_flows)

                cum_source_ns += ns.num_nodes
            else:
                source_ns = ns.get_source_ns()
                ns._param_range = deepcopy(source_ns._param_range)

                if source_ns not in node2tiednodes:
                    node2tiednodes[source_ns] = [[source_ns], 1, source_ns._param_flow_range]
                
                dup_count = node2tiednodes[source_ns][1]
                if dup_count >= max_tied_ns_per_parflow_block:
                    cum_param_flows += ns.num_nodes * ns.dist.num_param_flows()
                    ns._param_flow_range = (cum_param_flows - ns.num_nodes * ns.dist.num_param_flows(), cum_param_flows)
                    node2tiednodes[source_ns][2] = ns._param_flow_range

                    node2tiednodes[source_ns][0].append(ns)
                    node2tiednodes[source_ns][1] = 1
                else:
                    ns._param_flow_range = deepcopy(node2tiednodes[source_ns][2])

                    node2tiednodes[source_ns][1] += 1

        self._output_ind_range = (cum_nodes - layer_num_nodes, cum_nodes)
        self.num_parameters = cum_params
        self.num_param_flows = cum_param_flows
        self.num_nodes = layer_num_nodes
        self.dist_signature = dist_signature

        # Store the triton kernel functions implemented by the target `Distribution`
        self.fw_mar_fn = self.nodes[0].dist.fw_mar_fn
        self.bk_flow_fn = self.nodes[0].dist.bk_flow_fn
        self.sample_fn = self.nodes[0].dist.sample_fn
        self.em_fn = self.nodes[0].dist.em_fn

        ## Prepair and compile the layer ##
        num_vars = len(node_vars[0])
        # Start variable index: vids[i,:] are the variables of the ith node
        vids = torch.empty([self.num_nodes, num_vars], dtype = torch.long)
        # Start parameter index: params[s_pids[i]] is the first parameter of the ith node
        s_pids = torch.empty([self.num_nodes], dtype = torch.long)
        # Start parameter flow index: param_flows[s_pfids[i]] is the first parameter flow of the ith node
        s_pfids = torch.empty([self.num_nodes], dtype = torch.long)
        # Start metadata index: metadata[s_mids[i]] is the first metadata of the ith node
        metadata = []
        s_mids = torch.empty([self.num_nodes], dtype = torch.long)
        # source node ids (nodes with their original parameters)
        source_nids = torch.empty([cum_source_ns], dtype = torch.long)

        # Parameters of this layer
        params = torch.empty([self.num_parameters], dtype = torch.float32)
        
        n_start = 0
        source_n_start = 0
        for ns_id, ns in enumerate(self.nodes):
            n_end = n_start + ns.num_nodes

            # `vids`
            assert len(node_vars[ns_id]) == num_vars
            vids[n_start:n_end,:] = torch.tensor(node_vars[ns_id]).view(1, -1)

            # `s_pids` and `s_pfids`
            pid_offsets = torch.arange(0, ns.num_nodes * ns.dist.num_parameters(), ns.dist.num_parameters())
            s_pids[n_start:n_end] = ns._param_range[0] + pid_offsets

            pfid_offsets = torch.arange(0, ns.num_nodes * ns.dist.num_param_flows(), ns.dist.num_param_flows())
            s_pfids[n_start:n_end] = ns._param_flow_range[0] + pfid_offsets

            # `source_nids`
            if not ns.is_tied():
                source_n_end = source_n_start + ns.num_nodes
                source_nids[source_n_start:source_n_end] = torch.arange(n_start, n_end)
                source_n_start = source_n_end

            # `metadata` and `s_mids`
            s_mids[n_start:n_end] = len(metadata)
            metadata.extend(node_metadata[ns_id])

            n_start = n_end

        self.register_buffer("vids", vids)
        self.register_buffer("s_pids", s_pids)
        self.register_buffer("s_pfids", s_pfids)
        self.register_buffer("metadata", torch.tensor(metadata).float())
        self.register_buffer("s_mids", s_mids)
        self.register_buffer("source_nids", source_nids)

        ## Prepare info buffers for tied nodes ##
        self.tied2source_nids = []
        for source_ns, item in node2tiednodes.items():
            if len(item[0]) > 1: # If the length is 1, then everything is already accumulated in the source node's parflow
                num_par_flows = source_ns._param_flow_range[1] - source_ns._param_flow_range[0]
                pfid_start = source_ns._param_flow_range[0]
                ch_nodes = item[0]

                ch_pfids = torch.empty([len(ch_nodes)], dtype = torch.long)
                for ch_id, ch_ns in enumerate(ch_nodes):
                    ch_pfids[ch_id] = ch_ns._param_flow_range[0]

                self.tied2source_nids.append([pfid_start, num_par_flows, ch_pfids])

        self.params = nn.Parameter(params)
        # Due to the custom inplace backward pass implementation, we do not track 
        # gradient of PC parameters by PyTorch.
        self.params.requires_grad = False

        self.num_vars_per_node = num_vars

        self.param_flows = None

        self.device = torch.device("cpu")

        self._used_external_params = False

        # Batch size of parameters in the previous forward pass
        self._param_batch_size = 1
    
    def to(self, device):
        nn.Module.to(self, device = device)

        # Take special care to `tied2source_nids`
        for i in range(len(self.tied2source_nids)):
            self.tied2source_nids[i][2] = self.tied2source_nids[i][2].to(device)

        self.device = device

    def init_param_flows(self, flows_memory: float = 1.0):
        batch_size = self._param_batch_size
        if self.param_flows is None \
                or (self.param_flows.dim() == 1 and batch_size > 1) \
                or (self.param_flows.dim() == 2 and batch_size != self.param_flows.size(1)):
            if batch_size == 1:
                shape = [self.num_param_flows]
            else:
                shape = [self.num_param_flows, batch_size]
            self.param_flows = torch.zeros(shape, device = self.device)
        else:
            assert self.param_flows.size(0) == self.num_param_flows
            if flows_memory < 1.0:
                self.param_flows[:] *= flows_memory

        return None

    def forward(self, data: torch.Tensor, node_mars: torch.Tensor, params: Optional[Dict] = None,
                missing_mask: Optional[torch.Tensor] = None, _batch_first: bool = True, 
                _apply_missing_mask_only: bool = False):
        self._used_external_params = (params is not None)

        if params is None:
            params = self.params
        else:
            params = params["params"]

        assert params.dim() == 1

        if "cuda" in self.device.type:
            # Need to flatten data to ensure the memory is aligned following [num_vars, batch_size]
            data = data.reshape(-1).contiguous()

            tot_num_nodes = node_mars.size(0)
            batch_size = node_mars.size(1)
            node_offset = self._output_ind_range[0]

            if not self.provided("fw_local_ids"):
                layer_num_nodes = self._output_ind_range[1] - self._output_ind_range[0]
                fw_local_ids = None
            else:
                layer_num_nodes = self.fw_local_ids.size(0)
                fw_local_ids = self.fw_local_ids

            if not self.provided("_mars_kernel"):
                self._mars_kernel = self._compile_triton_kernel(self._mars_kernel_template, mar_fn = self.fw_mar_fn)

            BLOCK_SIZE = 1024

            grid = (triton.cdiv(layer_num_nodes * batch_size, BLOCK_SIZE),)

            if not _apply_missing_mask_only:
                self._mars_kernel[grid](
                    params_ptr = self.params, 
                    node_mars_ptr = node_mars, 
                    data_ptr = data, 
                    vids_ptr = self.vids, 
                    s_pids_ptr = self.s_pids, 
                    metadata_ptr = self.metadata, 
                    s_mids_ptr = self.s_mids, 
                    fw_local_ids_ptr = fw_local_ids,
                    layer_num_nodes = layer_num_nodes, 
                    batch_size = batch_size, 
                    num_vars_per_node = self.num_vars_per_node, 
                    nv_block_size = triton.next_power_of_2(self.num_vars_per_node),
                    node_offset = node_offset, 
                    BLOCK_SIZE = BLOCK_SIZE, 
                    partial_eval = 1 if fw_local_ids is not None else 0,
                    num_warps = 8
                )
            else:
                assert missing_mask is not None, "`missing_mask` should be provided when `_apply_missing_mask_only = True`."

            # Apply missing mask if required
            if missing_mask is not None:
                assert self.num_vars_per_node == 1, "`missing_mask` only supported for univariate distributions."

                mask_dim = missing_mask.dim()
                if mask_dim == 1:
                    mode = 0
                elif _batch_first or num_vars == missing_mask.size(1):
                    mode = 1
                else:
                    mode = 2

                grid = (triton.cdiv(layer_num_nodes * batch_size, BLOCK_SIZE),)

                self._fw_missing_mask_kernel[grid](
                    missing_mask_ptr = missing_mask,
                    node_mars_ptr = node_mars, 
                    vids_ptr = self.vids, 
                    fw_local_ids_ptr = fw_local_ids,
                    num_vars = self.pc_num_vars,
                    layer_num_nodes = layer_num_nodes, 
                    batch_size = batch_size, 
                    node_offset = node_offset, 
                    BLOCK_SIZE = 1024, 
                    partial_eval = 1 if fw_local_ids is not None else 0,
                    mode = mode,
                    num_warps = 8
                )

        else:
            raise NotImplementedError("CPU forward fn for input nodes is not implemented.")

    def backward(self, data: torch.Tensor, node_flows: torch.Tensor, 
                 node_mars: torch.Tensor, params: Optional[Dict] = None):
        """
        data: [num_vars, B]
        node_flows: [num_nodes, B]
        node_mars: [num_nodes, B]
        """

        if params is None:
            params = self.params
        else:
            params = params["params"]

        assert params.dim() == 1

        if "cuda" in self.device.type:
            # Need to flatten data to ensure the memory is aligned following [num_vars, batch_size]
            data = data.reshape(-1).contiguous()

            tot_num_nodes = node_flows.size(0)
            batch_size = node_flows.size(1)
            node_offset = self._output_ind_range[0]

            if not self.provided("bk_local_ids"):
                layer_num_nodes = self._output_ind_range[1] - self._output_ind_range[0]
                bk_local_ids = None
            else:
                layer_num_nodes = self.bk_local_ids.size(0)
                bk_local_ids = self.bk_local_ids

            if not self.provided("_flows_kernel"):
                self._flows_kernel = self._compile_triton_kernel(self._flows_kernel_template, flow_fn = self.bk_flow_fn)

            BLOCK_SIZE = 1024

            grid = (triton.cdiv(layer_num_nodes * batch_size, BLOCK_SIZE),)

            self._flows_kernel[grid](
                params_ptr = self.params,
                param_flows_ptr = self.param_flows,
                node_flows_ptr = node_flows, 
                node_mars_ptr = node_mars,
                data_ptr = data, 
                vids_ptr = self.vids, 
                s_pids_ptr = self.s_pids,
                s_pfids_ptr = self.s_pfids,
                metadata_ptr = self.metadata, 
                s_mids_ptr = self.s_mids, 
                bk_local_ids_ptr = bk_local_ids,
                layer_num_nodes = layer_num_nodes, 
                batch_size = batch_size, 
                num_vars_per_node = self.num_vars_per_node, 
                nv_block_size = triton.next_power_of_2(self.num_vars_per_node),
                node_offset = node_offset, 
                BLOCK_SIZE = BLOCK_SIZE, 
                partial_eval = 1 if bk_local_ids is not None else 0,
                num_warps = 8
            )

        else:
            raise NotImplementedError("CPU backward fn for input nodes is not implemented.")

    def sample(self, samples: torch.Tensor, node_flows: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, 
               params: Optional[torch.Tensor] = None, seed: Optional[int] = None):
        """
        samples:       [num_vars, B]
        missing_mask:  [num_vars, B] or [num_vars] or None
        node_flows:    [num_nodes, B]
        """

        if params is None:
            params = self.params
        else:
            params = params["params"]

        assert params.dim() == 1

        if "cuda" in self.device.type:

            sid, eid = self._output_ind_range
            tot_num_nodes = node_flows.size(0)
            batch_size = node_flows.size(1)
            node_offset = self._output_ind_range[0]

            # Get all node ids with non-zero flow
            nflow_xids, nflow_yids = torch.where(node_flows[sid:eid,:])
            num_activ_nodes = nflow_xids.size(0)

            if not self.provided("_sample_kernel"):
                self._sample_kernel = self._compile_triton_kernel(self._sample_kernel_template, sample_fn = self.sample_fn)

            BLOCK_SIZE = 1024

            grid = (triton.cdiv(num_activ_nodes, BLOCK_SIZE),)

            self._sample_kernel[grid](
                samples_ptr = samples, 
                params_ptr = params,
                nflow_xids_ptr = nflow_xids, 
                nflow_yids_ptr = nflow_yids, 
                vids_ptr = self.vids, 
                s_pids_ptr = self.s_pids, 
                metadata_ptr = self.metadata,
                s_mids_ptr = self.s_mids,
                num_activ_nodes = num_activ_nodes, 
                num_vars_per_node = self.num_vars_per_node, 
                nv_block_size = triton.next_power_of_2(self.num_vars_per_node),
                batch_size = batch_size, 
                BLOCK_SIZE = BLOCK_SIZE,
                seed = seed if seed is not None else random.randint(0, 1e8)
            )

        else:
            raise NotImplementedError("CPU sample fn for input nodes is not implemented.")

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        if not self._used_external_params:
            # Normalize and update parameters
            with torch.no_grad():

                if "cuda" in self.device.type:

                    # Accumulate parameter flows of tied nodes
                    for i in range(len(self.tied2source_nids)):
                        pfid_start, num_par_flows, ch_pfids = self.tied2source_nids[i]
                        num_coalesced_blocks = ch_pfids.size(0)

                        if num_coalesced_blocks <= 1024:
                            BLOCK_N = triton.next_power_of_2(num_coalesced_blocks)
                            BLOCK_M = min(1024 // BLOCK_N, num_par_flows)

                            grid = (triton.cdiv(num_par_flows, BLOCK_M),)

                            self._pflow_accum_kernel[grid](
                                param_flows_ptr = self.param_flows,
                                pfid_start = pfid_start,
                                ch_pfids_ptr = ch_pfids,
                                num_coalesced_blocks = num_coalesced_blocks,
                                num_par_flows = num_par_flows,
                                BLOCK_M = BLOCK_M,
                                BLOCK_N = BLOCK_N
                            )
                        else:
                            raise NotImplementedError("Unsupported number of coalesced parameter flows.")


                    layer_num_source_nodes = self.source_nids.size(0)

                    if not self.provided("_em_kernel"):
                        self._em_kernel = self._compile_triton_kernel(self._em_kernel_template, em_fn = self.em_fn)

                    constexprs = torch.tensor([step_size, pseudocount], dtype = torch.float32, device = self.device)

                    BLOCK_SIZE = 1024

                    grid = (triton.cdiv(layer_num_source_nodes, BLOCK_SIZE),)

                    self._em_kernel[grid](
                        params_ptr = self.params,
                        param_flows_ptr = self.param_flows,
                        s_pids_ptr = self.s_pids,
                        s_pfids_ptr = self.s_pfids,
                        metadata_ptr = self.metadata,
                        s_mids_ptr = self.s_mids,
                        source_nids_ptr = self.source_nids,
                        constexprs_ptr = constexprs,
                        layer_num_source_nodes = layer_num_source_nodes,
                        BLOCK_SIZE = BLOCK_SIZE,
                        num_warps = 8
                    )

                else:
                    raise NotImplementedError("CPU minibatch em fn for input nodes is not implemented.")

    def get_param_specs(self):
        return {"params": torch.Size([self.num_parameters])}

    def enable_partial_evaluation(self, fw_scopes: Optional[Union[Sequence[BitSet],Sequence[int]]] = None, 
                                  bk_scopes: Optional[Union[Sequence[BitSet],Sequence[int]]] = None, return_ids: bool = False):
        # Create cache if needed
        if not self.provided("scope2localgids"):
            self._prepare_scope2nids()

        # Filter forward nodes
        if fw_scopes is not None:
            fw_local_ids = []
            for scope in fw_scopes:
                if isinstance(scope, int):
                    scope = BitSet.from_array([scope])

                if scope not in self.scope2localgids:
                    continue

                fw_local_ids.append(self.scope2localgids[scope])

            if return_ids:
                return torch.cat(fw_local_ids, dim = 0)
            else:
                self.fw_local_ids = torch.cat(fw_local_ids, dim = 0)

        # Filter backward nodes
        if bk_scopes is not None:
            bk_local_ids = []
            for scope in bk_scopes:
                if isinstance(scope, int):
                    scope = BitSet.from_array([scope])

                if scope not in self.scope2localgids:
                    continue

                bk_local_ids.append(self.scope2localgids[scope])

            if return_ids:
                return torch.cat(bk_local_ids, dim = 0)
            else:
                self.bk_local_ids = torch.cat(bk_local_ids, dim = 0)

    def disable_partial_evaluation(self, forward: bool = True, backward: bool = True):
        if forward:
            self.fw_local_ids = None

        if backward:
            self.bk_local_ids = None

    def update_parameters(self):
        for idx, ns in enumerate(self.nodes):
            if ns.is_tied():
                continue

            par_start, par_end = ns._param_range
            ns._params = self.params.data[par_start:par_end].detach().cpu().clone()

    def _prepare_scope2nids(self):
        if not hasattr(self, "scope2localgids"):
            scope2localgids = dict()

            local_ngid = 0
            for ns in self.nodes:
                scope = ns.scope

                s_ngid = local_ngid
                e_ngid = local_ngid + ns.num_node_blocks

                with torch.no_grad():
                    if scope not in scope2localgids:
                        scope2localgids[scope] = [torch.zeros([0], dtype = torch.long)]

                    scope2localgids[scope].append(torch.arange(s_ngid, e_ngid))

                local_ngid += ns.num_node_blocks

            self.scope2localgids = {
                scope: torch.cat(ids, dim = 0).to(self.params.device) for scope, ids in scope2localgids.items()
            }

    def _reorder_nodes(self, nodes):
        node_set = set(nodes)
        reordered_untied_nodes = []
        reordered_tied_nodes = []
        added_node_set = set()
        for ns in nodes:
            if ns in added_node_set:
                continue
            if not ns.is_tied():
                reordered_untied_nodes.append(ns)
                added_node_set.add(ns)
            else:
                source_ns = ns.get_source_ns()
                if source_ns in added_node_set:
                    reordered_tied_nodes.append(ns)
                    added_node_set.add(ns)
                elif source_ns in node_set:
                    reordered_untied_nodes.append(source_ns)
                    reordered_tied_nodes.append(ns)
                    added_node_set.add(ns)
                    added_node_set.add(source_ns)
                else:
                    raise ValueError("A tied `InputNodes` should be in the same layer with its source nodes.")

        reordered_nodes = reordered_untied_nodes + reordered_tied_nodes

        assert len(reordered_nodes) == len(nodes), "Total node length should not change after reordering."

        return reordered_nodes

    def _init_parameters(self, perturbation):

        p_start, p_end = 0, 0
        for ns_id, ns in enumerate(self.nodes):
            # `params` (init/copy parameters)
            if not ns.is_tied():
                p_end = p_start + ns.num_nodes * ns.dist.num_parameters()
                if ns.has_params():
                    self.params[p_start:p_end] = ns._params.to(self.device)
                else:
                    self.params[p_start:p_end] = ns.init_parameters(ret_params = True).to(self.device)

                p_start = p_end

    @staticmethod
    def _mars_kernel_template(mar_fn, params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr, 
                              fw_local_ids_ptr, partial_eval: tl.constexpr, layer_num_nodes: tl.constexpr, batch_size: tl.constexpr, 
                              num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_nodes * batch_size

        # Raw batch and (local) node id
        batch_offsets = (offsets % batch_size)
        local_offsets = (offsets // batch_size)

        if partial_eval > 0:
            local_offsets = tl.load(fw_local_ids_ptr + local_offsets, mask = mask, other = 0)

        if num_vars_per_node == 1:
            # Get all variable ids
            vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)

            # Load the corresponding data
            data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = mask, other = 0)
        else:
            # Get all variable ids
            vids_offsets = tl.broadcast_to(local_offsets[:,None], (BLOCK_SIZE, nv_block_size)) * num_vars_per_node + \
                tl.broadcast_to(tl.arange(0, nv_block_size)[None,:], (BLOCK_SIZE, nv_block_size))
            vids_mask = tl.broadcast_to(mask[:,None], (BLOCK_SIZE, nv_block_size)) & \
                tl.broadcast_to((tl.arange(0, nv_block_size) < num_vars_per_node)[None,:], (BLOCK_SIZE, nv_block_size))
            vids = tl.load(vids_ptr + vids_offsets, mask = vids_mask, other = 0)

            # Load the corresponding data
            data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = vids_mask, other = 0)

        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)

        mars = mar_fn(local_offsets, data, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE)

        node_offsets = local_offsets + node_offset
        tl.store(node_mars_ptr + node_offsets * batch_size + batch_offsets, mars, mask = mask)

    @staticmethod
    @triton.jit
    def _fw_missing_mask_kernel(missing_mask_ptr, node_mars_ptr, vids_ptr, fw_local_ids_ptr, num_vars,
                                layer_num_nodes: tl.constexpr, batch_size: tl.constexpr, node_offset: tl.constexpr, 
                                BLOCK_SIZE: tl.constexpr, partial_eval: tl.constexpr, mode: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_nodes * batch_size

        # Raw batch and (local) node id
        batch_offsets = (offsets % batch_size)
        local_offsets = (offsets // batch_size)

        if partial_eval > 0:
            local_offsets = tl.load(fw_local_ids_ptr + local_offsets, mask = mask, other = 0)

        # Get all variable ids
        vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)

        # Fetch mask
        if mode == 0:
            # `mask_dim == 1`
            missing_mask = tl.load(missing_mask_ptr + vids, mask = mask, other = False)
        elif mode == 1:
            # `mask_dim == 1` and first dimension is batch
            mask_offsets = vids + batch_offsets * num_vars
            missing_mask = tl.load(missing_mask_ptr + mask_offsets, mask = mask, other = False)
        elif mode == 2:
            # `mask_dim == 1` and second dimension is batch
            mask_offsets = vids * batch_size + batch_offsets
            missing_mask = tl.load(missing_mask_ptr + mask_offsets, mask = mask, other = False)

        # Apply mask
        node_offsets = (local_offsets + node_offset) * batch_size + batch_offsets
        mars = tl.load(node_mars_ptr + node_offsets, mask = mask, other = 0.0)
        mars = tl.where(missing_mask, 0.0, mars)
        tl.store(node_mars_ptr + node_offsets, mars, mask = mask)

    @staticmethod
    def _flows_kernel_template(flow_fn, params_ptr, param_flows_ptr, node_flows_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, s_pfids_ptr,
                               metadata_ptr, s_mids_ptr, bk_local_ids_ptr, partial_eval: tl.constexpr, layer_num_nodes: tl.constexpr, 
                               batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                               BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_nodes * batch_size

        # Raw batch and (local) node id
        batch_offsets = (offsets % batch_size)
        local_offsets = (offsets // batch_size)

        if partial_eval > 0:
            local_offsets = tl.load(bk_local_ids_ptr + local_offsets, mask = mask, other = 0)

        if num_vars_per_node == 1:
            # Get all variable ids
            vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)

            # Load the corresponding data
            data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = mask, other = 0)
        else:
            # Get all variable ids
            vids_offsets = tl.broadcast_to(local_offsets[:,None], (BLOCK_SIZE, nv_block_size)) * num_vars_per_node + \
                tl.broadcast_to(tl.arange(0, nv_block_size)[None,:], (BLOCK_SIZE, nv_block_size))
            vids_mask = tl.broadcast_to(mask[:,None], (BLOCK_SIZE, nv_block_size)) & \
                tl.broadcast_to((tl.arange(0, nv_block_size) < num_vars_per_node)[None,:], (BLOCK_SIZE, nv_block_size))
            vids = tl.load(vids_ptr + vids_offsets, mask = vids_mask, other = 0)

            # Load the corresponding data
            data = tl.load(data_ptr + vids * batch_size + batch_offsets, mask = vids_mask, other = 0)

        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)
        s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)

        ns_offsets = (local_offsets + node_offset) * batch_size + batch_offsets
        flows = tl.load(node_flows_ptr + ns_offsets, mask = mask, other = 0)

        flow_fn(local_offsets, ns_offsets, data, flows, node_mars_ptr, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, 
                s_mids_ptr, mask, num_vars_per_node, BLOCK_SIZE)

    @staticmethod
    def _sample_kernel_template(sample_fn, samples_ptr, params_ptr, nflow_xids_ptr, nflow_yids_ptr, vids_ptr, s_pids_ptr, metadata_ptr, s_mids_ptr,
                                seed, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, num_activ_nodes: tl.constexpr, 
                                batch_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_activ_nodes

        # Raw batch and (local) node id
        local_offsets = tl.load(nflow_xids_ptr + offsets, mask = mask, other = 0)
        batch_offsets = tl.load(nflow_yids_ptr + offsets, mask = mask, other = 0)

        # Get all variable ids
        if num_vars_per_node == 1:
            vids = tl.load(vids_ptr + local_offsets, mask = mask, other = 0)
        else:
            # Get all variable ids
            vids_offsets = tl.broadcast_to(local_offsets[:,None], (BLOCK_SIZE, nv_block_size)) * num_vars_per_node + \
                tl.broadcast_to(tl.arange(0, nv_block_size)[None,:], (BLOCK_SIZE, nv_block_size))
            vids_mask = tl.broadcast_to(mask[:,None], (BLOCK_SIZE, nv_block_size)) & \
                tl.broadcast_to((tl.arange(0, nv_block_size) < num_vars_per_node)[None,:], (BLOCK_SIZE, nv_block_size))
            vids = tl.load(vids_ptr + vids_offsets, mask = vids_mask, other = 0)

        # Get start parameter indices
        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)

        sample_fn(samples_ptr, local_offsets, batch_offsets, vids, s_pids, params_ptr, metadata_ptr, s_mids_ptr, mask, batch_size, BLOCK_SIZE, seed)

    @staticmethod
    @triton.jit
    def _pflow_accum_kernel(param_flows_ptr, pfid_start, ch_pfids_ptr, num_coalesced_blocks, num_par_flows, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(axis = 0)

        offs_pflow = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_pflow = offs_pflow < num_par_flows

        offs_ch = tl.arange(0, BLOCK_N)
        mask_ch = offs_ch < num_coalesced_blocks

        # Start id for all ch parflows
        ch_pstart = tl.load(ch_pfids_ptr + offs_ch, mask = mask_ch)

        offs_ch_pflow = offs_pflow[:,None] + ch_pstart[None,:]
        mask_ch_pflow = mask_pflow[:,None] & mask_ch[None,:]
        ch_pflows = tl.load(param_flows_ptr + offs_ch_pflow, mask = mask_ch_pflow, other = 0)

        tar_pflows = tl.sum(ch_pflows, axis = 1)

        tl.store(param_flows_ptr + pfid_start + offs_pflow, tar_pflows, mask = mask_pflow)

    @staticmethod
    def _em_kernel_template(em_fn, params_ptr, param_flows_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr,
                            source_nids_ptr, constexprs_ptr, layer_num_source_nodes: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        # Retrieve all constexprs
        step_size = tl.load(constexprs_ptr)
        pseudocount = tl.load(constexprs_ptr + 1)

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_source_nodes

        # Get the local node ids
        local_offsets = tl.load(source_nids_ptr + offsets, mask = mask, other = 0)

        # Get the corresponding start id for `params` and `param_flows`
        s_pids = tl.load(s_pids_ptr + local_offsets, mask = mask, other = 0)
        s_pfids = tl.load(s_pfids_ptr + local_offsets, mask = mask, other = 0)

        em_fn(local_offsets, params_ptr, param_flows_ptr, s_pids, s_pfids, metadata_ptr, s_mids_ptr, mask,
              step_size, pseudocount, BLOCK_SIZE)

    @staticmethod
    def _compile_triton_kernel(main_fn, **sub_fns):

        def parse_source(src, get_signature = False):

            if not isinstance(src, str):
                src = textwrap.dedent(inspect.getsource(src))
                src = src[src.find("def"):]

            fn_header = ""
            fn_body = []
            curr_body = ""
            is_header = True
            for line in src.split("\n"):
                if is_header:
                    fn_header += line.strip(" ")
                    if ")" in line:
                        is_header = False
                else:
                    if line.strip(" ").startswith("#") or len(line.strip(" ")) == 0:
                        continue

                    if len(curr_body) > 0:
                        line = line.strip(" ")
                    else:
                        line = line.rstrip(" ")
                    curr_body += line
                    if curr_body[-1] == "\\":
                        curr_body = curr_body[:-1]
                    else:
                        parenthesis_count = curr_body.count("(") - curr_body.count(")")
                        if parenthesis_count == 0:
                            fn_body.append(curr_body)
                            curr_body = ""
                            parenthesis_count = 0

            if get_signature:
                args = fn_header.split("(")[1].split(")")[0].split(",")
                signature = list(map(lambda arg: arg.split(":")[0].split("=")[0].strip(" "), args))

                return fn_header, fn_body, signature

            return fn_header, fn_body

        main_fn_header, main_fn_body = parse_source(main_fn)

        # Map to new function header
        fn_name = main_fn_header.split("(")[0]
        fn_args = main_fn_header.split("(")[1].split(")")[0].split(",")
        fn_args = list(map(
            lambda str: str.strip(" "),
            filter(lambda arg: arg.split(":")[0].strip(" ") not in sub_fns, fn_args)
        ))
        seed_str = f"_{random.randint(0,1e8)}"
        new_fn_header = fn_name + seed_str + "(" + ",".join(fn_args) + "):"
        global_key = fn_name.split("def")[1].strip(" ") + seed_str

        # Map to new function body
        new_fn_body = []
        sub_fns = {k: parse_source(v, get_signature = True) for k, v in sub_fns.items()}
        for line in main_fn_body:
            substituted = False
            for sub_fn_name in sub_fns:
                if sub_fn_name in line:

                    # Get the target variable name
                    target_var = line.split(sub_fn_name)[0].split("=")[0].strip(" ")

                    # Get the variable name mapping
                    target_args = line.split(sub_fn_name)[1].split("(")[1].split(")")[0].split(",")
                    var_mapping = {k: v.strip(" ") for k, v in zip(sub_fns[sub_fn_name][2], target_args)}

                    indent = " " * (len(line) - len(line.lstrip()))
                    base_indent_len = len(sub_fns[sub_fn_name][1][0]) - len(sub_fns[sub_fn_name][1][0].lstrip())
                    for fn_line in sub_fns[sub_fn_name][1]:
                        fn_line = indent + fn_line[base_indent_len:]
                        if fn_line.split("#")[0].strip(" ") == "pass":
                            continue
                        for k, v in var_mapping.items():
                            fn_line = fn_line.replace(k, v)
                        if "return" in fn_line and len(target_var) > 0:
                            fn_line = fn_line.replace("return", f"{target_var} =")
                        new_fn_body.append(fn_line)

                    substituted = True
                    break

            if not substituted:
                new_fn_body.append(line)

        # Updated source code
        new_src = new_fn_header + "\n" + "\n".join(new_fn_body)

        # Add import commands
        new_src = "import triton\nimport triton.language as tl\n\n" + new_src

        # Make a pseudo-function from the source code
        new_fn = make_function_from_src(new_src)

        return FastJITFunction(new_fn)