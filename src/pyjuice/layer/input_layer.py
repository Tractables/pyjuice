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
from .layer import Layer


class InputLayer(Layer, nn.Module):
    def __init__(self, nodes: Sequence[InputNodes], cum_nodes: int = 0, maximize_group_size: bool = True) -> None:
        nn.Module.__init__(self)
        Layer.__init__(self, nodes)

        # Reorder input nodes such that for any tied nodes, its source nodes appear before them
        self.nodes = self._reorder_nodes(nodes)

        # Group size of the nodes in the current layer
        self.group_size = self.nodes[0].group_size
        if maximize_group_size:
            min_num_groups = min([node.num_node_groups for node in self.nodes])
            self.group_size *= 2 ** (min_num_groups.bit_length() - 1)
            self.group_size = min(self.group_size, 512)

        ## Parse input `nodes` ##
        node_vars = []
        node_sizes = []
        node_metadata = []
        layer_num_nodes = 0
        cum_params = 0
        cum_param_flows = 0
        cum_source_ngroups = 0
        dist_signature = None
        for ns in self.nodes:
            if dist_signature is None:
                dist_signature = ns.dist.get_signature()
            else:
                assert dist_signature == ns.dist.get_signature(), f"Nodes of an InputLayer must have the same distribution type, but got `{dist_signature}` and `{ns.dist.get_signature()}`."

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

                cum_source_ngroups += ns.num_nodes // self.group_size
            else:
                source_ns = ns.get_source_ns()
                ns._param_range = deepcopy(source_ns._param_range)

        self._output_ind_range = (cum_nodes - layer_num_nodes, cum_nodes)
        self.num_params = cum_params
        self.num_param_flows = cum_param_flows
        self.num_nodes = layer_num_nodes
        self.num_node_groups = self.num_nodes // self.group_size
        self.dist_signature = dist_signature

        # Store the triton kernel functions implemented by the target `Distribution`
        self.fw_mar_fn = self.nodes[0].dist.fw_mar_fn
        self.bk_flow_fn = self.nodes[0].dist.bk_flow_fn
        self.sample_fn = self.nodes[0].dist.sample_fn
        self.em_fn = self.nodes[0].dist.em_fn

        ## Prepair and compile the layer ##
        num_vars = len(node_vars[0])
        # Start variable index: vids[i,:] are the variables of the ith node group
        vids = torch.empty([self.num_node_groups, num_vars], dtype = torch.long)
        # Start parameter index: params[s_pids[i]] is the first parameter of the 1st node in the ith node group
        s_pids = torch.empty([self.num_node_groups], dtype = torch.long)
        # Pointer increment of the parameters: params[s_pids[i]+j*inc_pids[i]] is the first parameter 
        # of the (j+1)th node in the ith node group
        inc_pids = torch.empty([self.num_node_groups], dtype = torch.long)
        # Start parameter flow index: param_flows[s_pfids[i]] is the first parameter flow of the 1st node in the ith node group
        s_pfids = torch.empty([self.num_node_groups], dtype = torch.long)
        # Pointer increment of the parameters: param_flows[s_pfids[i]+j*inc_pfids[i]] is the first parameter flow
        # of the (j+1)th node in the ith node group
        inc_pfids = torch.empty([self.num_node_groups], dtype = torch.long)
        # Start metadata index: metadata[s_mids[i]] is the first metadata of the 1th node in the ith node group
        metadata = []
        s_mids = torch.empty([self.num_node_groups], dtype = torch.long)
        # source node group ids (nodes with their original parameters)
        source_ngids = torch.empty([cum_source_ngroups], dtype = torch.long)

        # Parameters of this layer
        params = torch.empty([self.num_params], dtype = torch.float32)
        
        ng_start = 0
        source_ng_start = 0
        param_start = 0
        for ns_id, ns in enumerate(self.nodes):
            ng_end = ng_start + ns.num_nodes // self.group_size

            # `vids`
            assert len(node_vars[ns_id]) == num_vars, f"Input nodes in the same layer should define on the same " \
                                                      f"number of variables, but got {len(node_vars[ns_id])} and {num_vars}."
            vids[ng_start:ng_end,:] = torch.tensor(node_vars[ns_id]).view(1, -1)

            # `s_pids` and `s_pfids`
            if not ns.is_tied():
                source_ns = ns
            else:
                source_ns = ns.get_source_ns()

            num_node_groups = ns.num_nodes // self.group_size

            n_params_per_group = self.group_size * ns.dist.num_parameters()
            gpid_offsets = torch.arange(0, num_node_groups * n_params_per_group, n_params_per_group)
            s_pids[ng_start:ng_end] = source_ns._param_range[0] + gpid_offsets
            inc_pids[ng_start:ng_end] = ns.dist.num_parameters()

            n_pflows_per_group = self.group_size * ns.dist.num_param_flows()
            gpfid_offsets = torch.arange(0, num_node_groups * n_pflows_per_group, n_pflows_per_group)
            s_pfids[ng_start:ng_end] = source_ns._param_flow_range[0] + gpfid_offsets
            inc_pfids[ng_start:ng_end] = ns.dist.num_param_flows()

            # `source_ngids`
            if not ns.is_tied():
                source_ng_end = source_ng_start + num_node_groups
                source_ngids[source_ng_start:source_ng_end] = torch.arange(ng_start, ng_end)
                source_ng_start = source_ng_end

            # `metadata` and `s_mids`
            s_mids[ng_start:ng_end] = len(metadata)
            metadata.extend(node_metadata[ns_id])

            ng_start = ng_end

        self.register_buffer("vids", vids)
        self.register_buffer("s_pids", s_pids)
        self.register_buffer("inc_pids", inc_pids)
        self.register_buffer("s_pfids", s_pfids)
        self.register_buffer("inc_pfids", inc_pfids)
        self.register_buffer("metadata", torch.tensor(metadata).float())
        self.register_buffer("s_mids", s_mids)
        self.register_buffer("source_ngids", source_ngids)

        self.params = nn.Parameter(params) # Parameters will be set later in `self._init_parameters()`
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

        self.device = device

    def init_param_flows(self, flows_memory: float = 0.0):
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
            self.param_flows[:] *= flows_memory

        return None

    def forward(self, data: torch.Tensor, node_mars: torch.Tensor, params: Optional[Dict] = None,
                missing_mask: Optional[torch.Tensor] = None):
        self._used_external_params = (params is not None)

        if params is None:
            params = self.params
        else:
            params = params["params"]

        assert params.dim() == 1

        if "cuda" in self.device.type:
            # Need to flatten data to ensure the memory is aligned following [num_vars, batch_size]
            data = data.reshape(-1).contiguous()

            batch_size = node_mars.size(1)
            node_offset = self._output_ind_range[0]

            if not self.provided("fw_local_group_ids"):
                fw_local_group_ids = None
            else:
                fw_local_group_ids = self.fw_local_group_ids

            if not self.provided("_mars_kernel"):
                self._mars_kernel = self._compile_triton_kernel(self._mars_kernel_template, mar_fn = self.fw_mar_fn)

            eval_num_groups = self.num_node_groups if not self.provided("fw_local_group_ids") else self.fw_local_group_ids.size(0)
            BLOCK_B = min(batch_size, 1024)
            TILE_SIZE_K = min(1024 // BLOCK_B, self.group_size)
            BLOCK_M = 1

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(eval_num_groups, BLOCK_M))

            self._mars_kernel[grid](
                params_ptr = params, 
                node_mars_ptr = node_mars, 
                data_ptr = data, 
                vids_ptr = self.vids, 
                s_pids_ptr = self.s_pids,
                inc_pids_ptr = self.inc_pids,
                metadata_ptr = self.metadata, 
                s_mids_ptr = self.s_mids, 
                fw_local_group_ids_ptr = fw_local_group_ids,
                batch_size = batch_size, 
                num_vars_per_node = self.num_vars_per_node, 
                nv_block_size = triton.next_power_of_2(self.num_vars_per_node),
                node_offset = node_offset, 
                group_size = self.group_size,
                TILE_SIZE_K = TILE_SIZE_K,
                K_NUM_TILES = self.group_size // TILE_SIZE_K,
                BLOCK_B = BLOCK_B,
                partial_eval = 1 if fw_local_group_ids is not None else 0
            )

            # Apply missing mask if required
            if missing_mask is not None:
                assert self.num_vars_per_node == 1, "`missing_mask` only supported for univariate distributions."
                assert missing_mask.dtype == torch.bool, "`missing_mask` must be boolean."

                mask_dim = missing_mask.dim()

                self._fw_missing_mask_kernel[grid](
                    missing_mask_ptr = missing_mask,
                    node_mars_ptr = node_mars, 
                    vids_ptr = self.vids, 
                    fw_local_group_ids_ptr = fw_local_group_ids,
                    batch_size = batch_size, 
                    node_offset = node_offset, 
                    group_size = self.group_size,
                    TILE_SIZE_K = TILE_SIZE_K,
                    K_NUM_TILES = self.group_size // TILE_SIZE_K,
                    BLOCK_B = BLOCK_B,
                    partial_eval = 1 if fw_local_group_ids is not None else 0,
                    mask_dim = mask_dim
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

            batch_size = node_flows.size(1)
            node_offset = self._output_ind_range[0]

            if not self.provided("bk_local_group_ids"):
                bk_local_group_ids = None
            else:
                bk_local_group_ids = self.bk_local_group_ids

            if not self.provided("_flows_kernel"):
                self._flows_kernel = self._compile_triton_kernel(self._flows_kernel_template, flow_fn = self.bk_flow_fn)

            eval_num_groups = self.num_node_groups if not self.provided("bk_local_group_ids") else self.bk_local_group_ids.size(0)
            BLOCK_B = min(batch_size, 1024)
            TILE_SIZE_K = min(1024 // BLOCK_B, self.group_size)
            BLOCK_M = 1

            grid = (triton.cdiv(batch_size, BLOCK_B), triton.cdiv(eval_num_groups, BLOCK_M))

            self._flows_kernel[grid](
                params_ptr = params,
                param_flows_ptr = self.param_flows,
                node_flows_ptr = node_flows,
                data_ptr = data, 
                vids_ptr = self.vids, 
                s_pids_ptr = self.s_pids,
                inc_pids_ptr = self.inc_pids,
                s_pfids_ptr = self.s_pfids,
                inc_pfids_ptr = self.inc_pfids,
                metadata_ptr = self.metadata, 
                s_mids_ptr = self.s_mids, 
                bk_local_group_ids_ptr = bk_local_group_ids,
                batch_size = batch_size, 
                num_vars_per_node = self.num_vars_per_node, 
                nv_block_size = triton.next_power_of_2(self.num_vars_per_node),
                node_offset = node_offset, 
                group_size = self.group_size,
                TILE_SIZE_K = TILE_SIZE_K,
                K_NUM_TILES = self.group_size // TILE_SIZE_K,
                BLOCK_B = BLOCK_B,
                partial_eval = 1 if bk_local_group_ids is not None else 0
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

            grid = lambda meta: (triton.cdiv(num_activ_nodes, meta['BLOCK_SIZE']),)
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
                BLOCK_SIZE = 2048,
                seed = seed if seed is not None else random.randint(0, 1e8)
            )

        else:
            raise NotImplementedError("CPU sample fn for input nodes is not implemented.")

    def mini_batch_em(self, step_size: float, pseudocount: float = 0.0):
        if not self._used_external_params:
            # Normalize and update parameters
            with torch.no_grad():

                if "cuda" in self.device.type:
                    layer_num_source_nodes = self.source_ngids.size(0)

                    if not self.provided("_em_kernel"):
                        self._em_kernel = self._compile_triton_kernel(self._em_kernel_template, em_fn = self.em_fn)

                    constexprs = torch.tensor([step_size, pseudocount], dtype = torch.float32, device = self.device)

                    grid = lambda meta: (triton.cdiv(layer_num_source_nodes, meta['BLOCK_SIZE']),)
                    self._em_kernel[grid](
                        params_ptr = self.params,
                        param_flows_ptr = self.param_flows,
                        s_pids_ptr = self.s_pids,
                        s_pfids_ptr = self.s_pfids,
                        metadata_ptr = self.metadata,
                        s_mids_ptr = self.s_mids,
                        source_ngids_ptr = self.source_ngids,
                        constexprs_ptr = constexprs,
                        layer_num_source_nodes = layer_num_source_nodes,
                        BLOCK_SIZE = 1024
                    )

                else:
                    raise NotImplementedError("CPU minibatch em fn for input nodes is not implemented.")

    def get_param_specs(self):
        return {"params": torch.Size([self.num_params])}

    def enable_partial_evaluation(self, fw_scopes: Optional[Union[Sequence[BitSet],Sequence[int]]] = None, 
                                  bk_scopes: Optional[Union[Sequence[BitSet],Sequence[int]]] = None, return_ids: bool = False):
        # Create cache if needed
        if not self.provided("scope2localgids"):
            self._prepare_scope2nids()

        # Filter forward nodes
        if fw_scopes is not None:
            fw_local_group_ids = []
            for scope in fw_scopes:
                if isinstance(scope, int):
                    scope = BitSet.from_array([scope])

                if scope not in self.scope2localgids:
                    continue

                fw_local_group_ids.append(self.scope2localgids[scope])

            if return_ids:
                return torch.cat(fw_local_group_ids, dim = 0)
            else:
                self.fw_local_group_ids = torch.cat(fw_local_group_ids, dim = 0)

        # Filter backward nodes
        if bk_scopes is not None:
            bk_local_group_ids = []
            for scope in bk_scopes:
                if isinstance(scope, int):
                    scope = BitSet.from_array([scope])

                if scope not in self.scope2localgids:
                    continue

                bk_local_group_ids.append(self.scope2localgids[scope])

            if return_ids:
                return torch.cat(bk_local_group_ids, dim = 0)
            else:
                self.bk_local_group_ids = torch.cat(bk_local_group_ids, dim = 0)

    def disable_partial_evaluation(self, forward: bool = True, backward: bool = True):
        if forward:
            self.fw_local_group_ids = None

        if backward:
            self.bk_local_group_ids = None

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
                e_ngid = local_ngid + ns.num_node_groups

                with torch.no_grad():
                    if scope not in scope2localgids:
                        scope2localgids[scope] = [torch.zeros([0], dtype = torch.long)]

                    scope2localgids[scope].append(torch.arange(s_nid, e_nid))

                local_nid += ns.num_nodes

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
    def _mars_kernel_template(mar_fn, params_ptr, node_mars_ptr, data_ptr, vids_ptr, s_pids_ptr, inc_pids_ptr, metadata_ptr, s_mids_ptr,
                              fw_local_group_ids_ptr, partial_eval: tl.constexpr, batch_size: tl.constexpr, 
                              num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, group_size: tl.constexpr,
                              TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_B: tl.constexpr):
        bid = tl.program_id(axis = 0)
        ngroup_id = tl.program_id(axis = 1)

        # Batch ids to process
        offs_batch = bid * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_batch = offs_batch < batch_size

        if partial_eval > 0:
            ngroup_id = tl.load(fw_local_group_ids_ptr + ngroup_id)

        if num_vars_per_node == 1:
            # Get variable id
            vid = tl.load(vids_ptr + ngroup_id)

            # Load the corresponding data
            offs_data = vid * batch_size + offs_batch
            data = tl.load(data_ptr + offs_data, mask = mask_batch, other = 0) # [BLOCK_B]
        else:
            # Get all variable ids
            offs_vs = tl.arange(0, nv_block_size)
            mask_vs = offs_vs < num_vars_per_node
            offs_vids = ngroup_id * num_vars_per_node + offs_vs
            mask_vids = mask_vs
            vids = tl.load(vids_ptr + offs_vids, mask = mask_vids, other = 0)

            # Load the corresponding data
            offs_data = vids[:,None] * batch_size + offs_batch[None,:]
            data = tl.load(data_ptr + offs_data, mask = (mask_vids[:,None] & mask_batch[None,:]), other = 0)

        # Initialize pointers to `params`
        off_params = tl.load(s_pids_ptr + ngroup_id)
        inc_params = tl.load(inc_pids_ptr + ngroup_id)
        offs_node = tl.arange(0, TILE_SIZE_K)
        p_params = params_ptr + off_params + inc_params * offs_node # [TILE_SIZE_K]

        # Initialize pointers to `metadata`
        offs_metadata = tl.load(s_mids_ptr + ngroup_id)
        p_metadata = metadata_ptr + offs_metadata # [1]

        # Initialize pointers to `node_mars`
        p_nmars = node_mars_ptr + \
            (ngroup_id * group_size + offs_node[:,None] + node_offset) * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

        # Inner loop to process everything in the node group
        mask = mask_batch[None,:]
        for i in range(K_NUM_TILES):

            mars = mar_fn(data, p_params, p_metadata, mask, num_vars_per_node)

            tl.store(p_nmars, mars, mask = mask)

            # Increment pointers
            p_params += inc_params * TILE_SIZE_K
            p_nmars += TILE_SIZE_K * batch_size

    @staticmethod
    @triton.jit
    def _fw_missing_mask_kernel(missing_mask_ptr, node_mars_ptr, vids_ptr, fw_local_group_ids_ptr, group_size: tl.constexpr,
                                batch_size: tl.constexpr, node_offset: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr,
                                BLOCK_B: tl.constexpr, partial_eval: tl.constexpr, mask_dim: tl.constexpr):

        bid = tl.program_id(axis = 0)
        ngroup_id = tl.program_id(axis = 1)

        # Batch ids to process
        offs_batch = bid * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_batch = offs_batch < batch_size

        if partial_eval > 0:
            ngroup_id = tl.load(fw_local_group_ids_ptr + ngroup_id)

        # Get variable id
        vid = tl.load(vids_ptr + ngroup_id)

        # Fetch mask
        if mask_dim == 1:
            missing_mask = tl.load(missing_mask_ptr + vid)
        else:
            offs_mmask = vid * batch_size + offs_batch
            missing_mask = tl.load(missing_mask_ptr + offs_mmask, mask = mask_batch, other = False)

        # Initialize pointers to `node_mars`
        offs_node = tl.arange(0, TILE_SIZE_K)
        p_nmars = node_mars_ptr + \
            (ngroup_id * group_size + offs_node[:,None] + node_offset) * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

        # Apply mask
        mask = mask_batch[None,:]
        if mask_dim == 1:
            if missing_mask:
                for i in range(K_NUM_TILES):

                    # mars = tl.load(p_nmars, mask = mask, other = 0.0)
                    tl.store(p_nmars, 0.0, mask = mask)

                    # Increment pointers
                    p_nmars += TILE_SIZE_K * batch_size
        else:
            for i in range(K_NUM_TILES):

                mars = tl.load(p_nmars, mask = mask, other = 0.0)
                mars = tl.where(missing_mask[None,:], 0.0, mars)
                tl.store(p_nmars, mars, mask = mask)

                # Increment pointers
                p_nmars += TILE_SIZE_K * batch_size

    @staticmethod
    def _flows_kernel_template(flow_fn, params_ptr, param_flows_ptr, node_flows_ptr, data_ptr, vids_ptr, s_pids_ptr, inc_pids_ptr, 
                               s_pfids_ptr, inc_pfids_ptr, metadata_ptr, s_mids_ptr, bk_local_group_ids_ptr, partial_eval: tl.constexpr, 
                               batch_size: tl.constexpr, num_vars_per_node: tl.constexpr, nv_block_size: tl.constexpr, node_offset: tl.constexpr, 
                               group_size: tl.constexpr, TILE_SIZE_K: tl.constexpr, K_NUM_TILES: tl.constexpr, BLOCK_B: tl.constexpr):

        bid = tl.program_id(axis = 0)
        ngroup_id = tl.program_id(axis = 1)

        # Batch ids to process
        offs_batch = bid * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_batch = offs_batch < batch_size

        if partial_eval > 0:
            ngroup_id = tl.load(bk_local_group_ids_ptr + ngroup_id)

        if num_vars_per_node == 1:
            # Get variable id
            vid = tl.load(vids_ptr + ngroup_id)

            # Load the corresponding data
            offs_data = vid * batch_size + offs_batch
            data = tl.load(data_ptr + offs_data, mask = mask_batch, other = 0) # [BLOCK_B]
        else:
            # Get all variable ids
            offs_vs = tl.arange(0, nv_block_size)
            mask_vs = offs_vs < num_vars_per_node
            offs_vids = ngroup_id * num_vars_per_node + offs_vs
            mask_vids = mask_vs
            vids = tl.load(vids_ptr + offs_vids, mask = mask_vids, other = 0)

            # Load the corresponding data
            offs_data = vids[:,None] * batch_size + offs_batch[None,:]
            data = tl.load(data_ptr + offs_data, mask = (mask_vids[:,None] & mask_batch[None,:]), other = 0)

        # Initialize pointers to `params`
        off_params = tl.load(s_pids_ptr + ngroup_id)
        inc_params = tl.load(inc_pids_ptr + ngroup_id)
        offs_node = tl.arange(0, TILE_SIZE_K)
        p_params = params_ptr + off_params + inc_params * offs_node # [TILE_SIZE_K]

        # Initialize pointers to `param_flows`
        off_parflows = tl.load(s_pfids_ptr + ngroup_id)
        inc_parflows = tl.load(inc_pfids_ptr + ngroup_id)
        p_parflows = param_flows_ptr + off_parflows + inc_parflows * offs_node # [TILE_SIZE_K]

        # Initialize pointers to `metadata`
        offs_metadata = tl.load(s_mids_ptr + ngroup_id)
        p_metadata = metadata_ptr + offs_metadata # [1]

        # Initialize pointers to `node_mars`
        p_nflows = node_flows_ptr + \
            (ngroup_id * group_size + offs_node[:,None] + node_offset) * batch_size + \
            offs_batch[None,:] # [TILE_SIZE_K, BLOCK_B]

        # Inner loop to process everything in the node group
        mask = mask_batch[None,:]
        for i in range(K_NUM_TILES):

            # Read out the flows
            flows = tl.load(p_nflows, mask = mask, other = 0)

            flow_fn(flows, data, p_parflows, p_params, p_metadata, mask, num_vars_per_node)

            # Increment pointers
            p_params += inc_params * TILE_SIZE_K
            p_parflows += inc_parflows * TILE_SIZE_K
            p_nflows += TILE_SIZE_K * batch_size

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
    def _em_kernel_template(em_fn, params_ptr, param_flows_ptr, s_pids_ptr, s_pfids_ptr, metadata_ptr, s_mids_ptr,
                            source_ngids_ptr, constexprs_ptr, layer_num_source_nodes: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis = 0)
        block_start = pid * BLOCK_SIZE

        # Retrieve all constexprs
        step_size = tl.load(constexprs_ptr)
        pseudocount = tl.load(constexprs_ptr + 1)

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < layer_num_source_nodes

        # Get the local node ids
        local_offsets = tl.load(source_ngids_ptr + offsets, mask = mask, other = 0)

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

        return JITFunction(new_fn)