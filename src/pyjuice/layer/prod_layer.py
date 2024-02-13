from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import warnings
import time
from typing import Sequence, Optional

from pyjuice.nodes import ProdNodes
from pyjuice.utils.parameter_list import FastParamList
from pyjuice.utils.kernel_launcher import FastJITFunction
from .layer import Layer
from .backend.node_partition import partition_nodes_by_n_edges
from .backend.index_set import batched_index_set, batched_index_cum
from .compilation import next_power_of_2, get_prod_layer_stats, prod_layer_forward_compilation, \
                         flatten_c_ids, get_prod_layer_parstats, prod_layer_backward_compilation


class ProdLayer(Layer, nn.Module):

    def __init__(self, nodes: Sequence[ProdNodes], global_nid_start: Optional[int] = None, 
                 layer_sparsity_tol: Optional[float] = None, max_num_partitions: Optional[int] = None, 
                 disable_gpu_compilation: bool = False, force_gpu_compilation: bool = False) -> None:
        Layer.__init__(self, nodes)
        nn.Module.__init__(self)

        assert len(nodes) > 0, "No input node."

        use_block_sparse_edges = True
        for nid in range(0, len(nodes)):
            if nodes[nid].is_sparse() or nodes[0].block_size != nodes[nid].block_size:
                use_block_sparse_edges = False
                break
        self.use_block_sparse_edges = use_block_sparse_edges

        self.nodes = nodes
        self.block_size = nodes[0].block_size if self.use_block_sparse_edges else 1

        if global_nid_start is None:
            global_nid_start = self.block_size

        ## Get layer statistics & prepare for compilation ##

        layer_num_nblocks, layer_num_edges, n_chgs = get_prod_layer_stats(
            self.nodes, self.block_size, global_nid_start = global_nid_start, 
            use_block_sparse_edges = self.use_block_sparse_edges
        )

        self.num_nodes = layer_num_nblocks * self.block_size
        self.num_edges = layer_num_edges

        # Find a good strategy to partition the nodes into partitions according to their number of children 
        # to minimize total computation cost
        fw_partition_max_chs = partition_nodes_by_n_edges(
            n_chgs, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
        )

        # Since the triton kernels require the maximum number children for each partition to be a power of 2,
        # we postprocess the block sizes
        fw_partition_max_chs = torch.unique(next_power_of_2(fw_partition_max_chs))

        self.num_fw_partitions = len(fw_partition_max_chs) # Number of partitions

        # fw_n_partition_ids:      [num_ns]             stores the partition id for each `ns` in `nodes`
        # fw_n_id_in_partition:    [num_ns]             stores the start index of each `ns` in the corresponding partition
        # fw_num_ngs_in_partition: [num_fw_partitions]  number of node blocks in each partition
        num_ns = len(self.nodes)
        fw_n_partition_ids = torch.zeros([num_ns], dtype = torch.long)
        fw_n_id_in_partition = torch.zeros([num_ns], dtype = torch.long)
        fw_num_ngs_in_partition = torch.zeros([self.num_fw_partitions], dtype = torch.long)

        for ns_id, ns in enumerate(self.nodes):
            partition_id = (ns.num_chs > fw_partition_max_chs).sum().item()

            fw_n_partition_ids[ns_id] = partition_id
            fw_n_id_in_partition[ns_id] = fw_num_ngs_in_partition[partition_id]
            if self.use_block_sparse_edges:
                fw_num_ngs_in_partition[partition_id] += ns.num_node_blocks
            else:
                fw_num_ngs_in_partition[partition_id] += ns.num_nodes

        ## Initialize forward pass ##

        # nids:      List[[partition_size]]                      stores node ids
        # cids:      List[[partition_size, partition_max_n_chs]] stores indices of child nodes
        nids, cids = prod_layer_forward_compilation(
            self.nodes, fw_partition_max_chs, fw_n_partition_ids, fw_n_id_in_partition, fw_num_ngs_in_partition, 
            self.block_size, self.use_block_sparse_edges
        )

        # Store buffers for the forward pass
        self.partitioned_nids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in nids])
        self.partitioned_cids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in cids])

        ## Initialize backward pass ##

        # flat_cids:   flattened version of `cids`
        # flat_cid2id: mapping from every `flat_cids` to its corresponding `nids`
        flat_cids, flat_cid2nid = flatten_c_ids(nids, cids)

        # flat_u_cids:        [num_used_ch_nblocks]    child block ids that have at least one parent
        # par_counts:         [num_used_ch_nblocks]    the number of parents for each child node block
        # Note: the dummy node has been removed from `flat_u_cids` and `par_counts`
        flat_u_cids, par_counts = get_prod_layer_parstats(flat_cids, global_nid_start = global_nid_start)

        # Find a good strategy to partition the child nodes into blocks according to their number of parents 
        # to minimize total computation cost
        bk_partition_max_pars = partition_nodes_by_n_edges(
            par_counts, sparsity_tolerance = layer_sparsity_tol, max_num_partitions = max_num_partitions
        )

        # Since the triton kernels require the maximum number children for each block to be a power of 2,
        # we postprocess the block sizes
        bk_partition_max_pars = torch.unique(next_power_of_2(bk_partition_max_pars))

        self.num_bk_partitions = len(bk_partition_max_pars) # Number of partitions

        # bk_n_partition_ids:     [num_ch_nblocks]       stores the block id for each `ns` in `nodes`
        # bk_n_id_in_partition:   [num_ch_nblocks]       stores the start index of each `ns` in the partition
        # bk_num_ns_in_partition: [num_bk_partitions]    number of node blocks in each partition
        num_ch_nblocks = flat_u_cids.size(0)
        bk_n_partition_ids = torch.zeros([num_ch_nblocks], dtype = torch.long)
        bk_n_id_in_partition = torch.zeros([num_ch_nblocks], dtype = torch.long)
        bk_num_ns_in_partition = torch.zeros([self.num_bk_partitions], dtype = torch.long)

        min_n_pars = 0
        for partition_id, max_n_pars in enumerate(bk_partition_max_pars):
            criterion = (par_counts >= min_n_pars) & (par_counts <= max_n_pars)
            filtered_idxs = torch.where(criterion)[0]
            partition_size = criterion.sum().item()

            bk_n_partition_ids[criterion] = partition_id
            bk_n_id_in_partition[criterion] = torch.arange(partition_size)
            bk_num_ns_in_partition[partition_id] = partition_size

            min_n_pars = max_n_pars + 1

        # u_cids:    List[[partition_ch_size]]                       stores child node block ids
        # parids:    List[[partition_ch_size, partition_max_n_pars]] stores indices of parent node blocks
        u_cids, parids = prod_layer_backward_compilation(
            flat_u_cids, flat_cids, flat_cid2nid, 
            bk_partition_max_pars, bk_n_partition_ids, bk_n_id_in_partition, bk_num_ns_in_partition,
            use_cuda = force_gpu_compilation or (not disable_gpu_compilation and (flat_cids.size(0) > 4000))
        )

        # Store buffers for the backward pass
        self.partitioned_u_cids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in u_cids])
        self.partitioned_parids = FastParamList([nn.Parameter(tensor, requires_grad = False) for tensor in parids])

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, _for_backward: bool = False) -> None:
        """
        Computes the forward pass of a product layer. If `block_size == 1`, it is equivalent to the following:
        ```
        element_mars[nids] = node_mars[cids].sum(dim = 1)
        ```

        Parameters:
        `node_mars`:    [num_nodes, B]
        `element_mars`: [max_num_els, B]
        """

        if not _for_backward and self.provided("fw_partition_local_ids"):
            # Partial evaluation (for forward pass)
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]
                cids = self.partitioned_cids[partition_id]
                local_ids = self.fw_partition_local_ids[partition_id]

                self._forward_backward(
                    element_mars, node_mars, nids, cids, local_ids = local_ids, accum = False
                )

        elif _for_backward and self.provided("bk_fw_partition_local_ids"):
            # Partial evaluation (for backward pass)
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]
                cids = self.partitioned_cids[partition_id]
                local_ids = self.bk_fw_partition_local_ids[partition_id]

                self._forward_backward(
                    element_mars, node_mars, nids, cids, local_ids = local_ids, accum = False
                )

        else:
            # Evaluate the whole layer
            for partition_id in range(self.num_fw_partitions):
                nids = self.partitioned_nids[partition_id]
                cids = self.partitioned_cids[partition_id]

                self._forward_backward(
                    element_mars, node_mars, nids, cids, accum = False
                )

        return None

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor) -> None:
        """
        Computes the backward pass of a product layer:
        ```
        node_flows[u_cids] = element_flows[parids].sum(dim = 1)
        ```

        Parameters:
        `node_flows`:    [num_nodes, B]
        `element_flows`: [max_num_els, B]
        """
        
        if self.provided("bk_partition_local_ids"):
            # Partial evaluation
            for partition_id in range(self.num_bk_partitions):
                u_cids = self.partitioned_u_cids[partition_id]
                parids = self.partitioned_parids[partition_id]
                local_ids = self.bk_partition_local_ids[partition_id]

                self._forward_backward(node_flows, element_flows, u_cids, parids, local_ids = local_ids, accum = True)
        
        else:
            # Evaluate the whole layer
            for partition_id in range(self.num_bk_partitions):
                u_cids = self.partitioned_u_cids[partition_id]
                parids = self.partitioned_parids[partition_id]

                self._forward_backward(node_flows, element_flows, u_cids, parids, accum = True)
        
        return None

    def enable_partial_evaluation(self, fw_scopes: Optional[Sequence[BitSet]] = None, bk_scopes: Optional[Sequence[BitSet]] = None):
        super(ProdLayer, self).enable_partial_evaluation(fw_scopes = fw_scopes, bk_scopes = bk_scopes)

        # For product layers, we need a special forward pass during the backward process of the circuit
        if bk_scopes is not None:
            bk_fw_partition_local_ids = [[] for _ in range(self.num_fw_partitions)]
            for scope in bk_scopes:
                if scope not in self.fw_scope2localids:
                    continue

                for partition_id, ids in enumerate(self.fw_scope2localids[scope]):
                    bk_fw_partition_local_ids[partition_id].append(self.fw_scope2localids[scope][partition_id])

            self.bk_fw_partition_local_ids = [
                torch.cat(ids, dim = 0) if len(ids) > 0 else torch.zeros([0], dtype = torch.long) for ids in bk_fw_partition_local_ids
            ]

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _forward_backward_kernel_3d(node_vals_ptr, element_vals_ptr, local_ids_ptr, nids_ptr, cids_ptr, tot_n_nodes, tot_n_eles, n_nblocks,
                                    num_edges: tl.constexpr, batch_size, BLOCK_M: tl.constexpr, BLOCK_B: tl.constexpr, 
                                    block_size: tl.constexpr, accum: tl.constexpr, partial_eval: tl.constexpr):
        """
        This kernel implements the function with 3d tensors. However, it only work with `triton==2.0.0`.
        """
        
        pid_m = tl.program_id(axis = 0) # ID of size-`BLOCK_M` nodes
        pid_b = tl.program_id(axis = 1) # ID of size-`BLOCK_B` batches

        if block_size >= BLOCK_M:

            # Get inferred node block id from `pid_m`
            nblock_id = pid_m // (block_size // BLOCK_M)
            ntile_id = pid_m % (block_size // BLOCK_M)

            # For partial evaluation
            if partial_eval == 1:
                nblock_id = tl.load(local_ids_ptr + nblock_id)

            # Batch offsets and mask
            offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B 
            mask_batch = offs_batch < batch_size

            # Get the block start ids for the children
            # To make the triton compiler happy, we reload every index `BLOCK_M` times
            offs_ne = tl.arange(0, num_edges * BLOCK_M) // BLOCK_M
            offs_ne = tl.view(offs_ne, (BLOCK_M, num_edges))
            offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_ne) # [BLOCK_M, num_edges]

            # Get the edge values from child nodes
            block_nids = tl.arange(0, BLOCK_M) + ntile_id * BLOCK_M
            offs_evals = offs_egstart + block_nids[:,None]
            evals = tl.load(element_vals_ptr + offs_evals[None,:,:] * batch_size + offs_batch[:,None,None], mask = mask_batch[:,None,None])

            # Take the sum of the child nodes' log-probabilities
            nvals = tl.sum(evals, axis = 2)

            # Node ids to `node_vals_ptr`
            nblock_start = tl.load(nids_ptr + nblock_id)
            offs_nvals = (nblock_start + block_nids[None,:]) * batch_size + offs_batch[:,None]

            # Accumulate the `node_vals` if required
            if accum == 1:
                node_vals = tl.load(node_vals_ptr + offs_nvals, mask = mask_batch[:,None], other = 0)
                nvals += node_vals

            tl.store(node_vals_ptr + offs_nvals, nvals, mask = mask_batch[:,None])

        else:

            # Node offsets and mask
            offs_node = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
            mask_node = offs_node < n_nblocks * block_size

            # Inferred block ids
            nblock_ids = offs_node // block_size

            # For partial evaluation
            if partial_eval == 1:
                nblock_ids = tl.load(local_ids_ptr + nblock_ids, mask = mask_node)

            # Batch offsets and mask
            offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B 
            mask_batch = offs_batch < batch_size

            # Get the block start ids for the children
            offs_ne = tl.arange(0, num_edges * BLOCK_M) // BLOCK_M
            offs_ne = tl.view(offs_ne, (BLOCK_M, num_edges))
            offs_egstart = tl.load(cids_ptr + nblock_ids[:,None] * num_edges + offs_ne, mask = mask_node[:,None]) # [BLOCK_M, num_edges]

            # Get the edge values from child nodes
            block_nids = (offs_node % block_size)
            offs_evals = offs_egstart + block_nids[:,None]
            evals = tl.load(element_vals_ptr + offs_evals[None,:,:] * batch_size + offs_batch[:,None,None], mask = (mask_batch[:,None,None] & mask_node[None,:,None]))

            # Take the sum of the child nodes' log-probabilities
            nvals = tl.sum(evals, axis = 2)

            # Node ids to `node_vals_ptr`
            nblock_start = tl.load(nids_ptr + nblock_ids[None,:])
            offs_nvals = (nblock_start + block_nids[None,:]) * batch_size + offs_batch[:,None]

            # Accumulate the `node_vals` if required
            if accum == 1:
                node_vals = tl.load(node_vals_ptr + offs_nvals, mask = mask_batch[:,None], other = 0)
                nvals += node_vals

            tl.store(node_vals_ptr + offs_nvals, nvals, mask = mask_batch[:,None])

    @staticmethod
    # @triton.jit
    @FastJITFunction
    def _forward_backward_kernel_2d(node_vals_ptr, element_vals_ptr, local_ids_ptr, nids_ptr, cids_ptr, tot_n_nodes, tot_n_eles, n_nblocks,
                                    num_edges: tl.constexpr, batch_size, BLOCK_M: tl.constexpr, BLOCK_B: tl.constexpr, 
                                    block_size: tl.constexpr, accum: tl.constexpr, partial_eval: tl.constexpr):
        """
        This kernel implements the function with 2d tensors. It works for all `triton` versions.
        """

        pid_m = tl.program_id(axis = 0) # ID of size-`BLOCK_M` nodes
        pid_b = tl.program_id(axis = 1) # ID of size-`BLOCK_B` batches

        # Get inferred node block id from `pid_m`
        nblock_id = pid_m // (block_size // BLOCK_M)
        ntile_id = pid_m % (block_size // BLOCK_M)

        # For partial evaluation
        if partial_eval == 1:
            nblock_id = tl.load(local_ids_ptr + nblock_id)

        # Batch offsets and mask
        offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B # [BLOCK_B]
        mask_batch = offs_batch < batch_size

        # Get the block start ids for the children
        offs_edge = tl.arange(0, num_edges)
        offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_edge) # [num_edges]

        # Base ptr for ch values
        offs_evals = (offs_egstart[:,None] + ntile_id * BLOCK_M) * batch_size + offs_batch[None,:] # [num_edges, BLOCK_B]

        # Base ptr for par values
        nblock_start = tl.load(nids_ptr + nblock_id)
        offs_nvals = (nblock_start + ntile_id * BLOCK_M) * batch_size + offs_batch # [BLOCK_B]

        # Inner loop
        for i in range(0, BLOCK_M):
            evals = tl.load(element_vals_ptr + offs_evals, mask = mask_batch[None,:], other = 0)
            nvals = tl.sum(evals, axis = 0)

            # Accumulate the `node_vals` if required
            if accum == 1:
                node_vals = tl.load(node_vals_ptr + offs_nvals, mask = mask_batch)
                nvals += node_vals

            tl.store(node_vals_ptr + offs_nvals, nvals, mask = mask_batch)

            offs_nvals += batch_size
            offs_evals += batch_size

    @staticmethod
    @torch.compile(mode = "reduce-overhead", fullgraph = True)
    def _forward_backward_pytorch(node_vals, element_vals, nids, cids, accum: bool = False):
        nids = nids[:,None] + torch.arange(0, self.block_size, device = node_vals.device)[None,:]
        cids = cids[:,None,:] + torch.arange(0, self.block_size, device = node_vals.device)[None,:,None]
        if accum:
            node_vals[nids] += element_vals[cids].sum(dim = 2)
        else:
            node_vals[nids] = element_vals[cids].sum(dim = 2)

        return None

    def _forward_backward(self, node_vals: torch.Tensor, element_vals: torch.Tensor,
                          nids: torch.Tensor, cids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                          accum: bool = False) -> None:
        tot_n_nodes = node_vals.size(0)
        tot_n_eles = element_vals.size(0)
        n_nblocks = nids.size(0) if local_ids is None else local_ids.size(0)
        num_edges = cids.size(1)
        batch_size = node_vals.size(1)

        block_size = self.block_size
        accum = 1 if accum else 0
        partial_eval = 1 if local_ids is not None else 0

        assert num_edges & (num_edges - 1) == 0, "`num_edges` must be a power of 2."

        # Fall back to the `torch.compile` kernel in the case where we cannot store child edges within a single block
        if num_edges > 1024:
            self._forward_backward_pytorch(node_vals, element_vals, nids, cids, accum = accum)

            return None

        if not triton.__version__ == "2.0.0":

            BLOCK_B = min(2048 // num_edges, triton.next_power_of_2(batch_size))
            BLOCK_M = min(max(2048 // (BLOCK_B * num_edges), 1), self.block_size)

            grid = (triton.cdiv(n_nblocks * self.block_size, BLOCK_M), triton.cdiv(batch_size, BLOCK_B))

            self._forward_backward_kernel_2d[grid](
                node_vals_ptr = node_vals, 
                element_vals_ptr = element_vals,
                local_ids_ptr = local_ids,
                nids_ptr = nids, 
                cids_ptr = cids, 
                tot_n_nodes = tot_n_nodes,
                tot_n_eles = tot_n_eles,
                n_nblocks = n_nblocks,
                num_edges = num_edges,
                batch_size = batch_size,
                BLOCK_M = BLOCK_M, 
                BLOCK_B = BLOCK_B,
                block_size = block_size,
                accum = accum,
                partial_eval = partial_eval
            )

        else:

            BLOCK_B = min(1024 // num_edges, triton.next_power_of_2(batch_size))
            BLOCK_M = min(max(1024 // (BLOCK_B * num_edges), 1), triton.next_power_of_2(n_nblocks) * self.block_size)

            grid = (triton.cdiv(n_nblocks * self.block_size, BLOCK_M), triton.cdiv(batch_size, BLOCK_B))

            self._forward_backward_kernel_3d[grid](
                node_vals_ptr = node_vals, 
                element_vals_ptr = element_vals,
                local_ids_ptr = local_ids,
                nids_ptr = nids, 
                cids_ptr = cids, 
                tot_n_nodes = tot_n_nodes,
                tot_n_eles = tot_n_eles,
                n_nblocks = n_nblocks,
                num_edges = num_edges,
                batch_size = batch_size,
                BLOCK_M = BLOCK_M, 
                BLOCK_B = BLOCK_B,
                block_size = block_size,
                accum = accum,
                partial_eval = partial_eval
            )

        return None

    def _prepare_scope2nids(self):

        # Saved for the next sum layer
        prod_scope_eleids = list()
        global_eid = 1
        for ns in self.nodes:
            s_eid = global_eid
            e_eid = global_eid + ns.num_nodes

            prod_scope_eleids.append((ns.scope, (s_eid, e_eid)))

            global_eid += ns.num_nodes

        if not (hasattr(self, "fw_scope2localids") and hasattr(self, "bk_scope2localids")):
            fw_scope2localids = dict()
            bk_scope2localids = dict()

            # Forward local indices
            global_eid = 1
            for ns in self.nodes:
                scope = ns.scope

                s_eid = global_eid
                e_eid = global_eid + ns.num_nodes

                with torch.no_grad():
                    if scope not in fw_scope2localids:
                        fw_scope2localids[scope] = [
                            torch.zeros([0], dtype = torch.long).to(self.partitioned_nids[0].device) for _ in range(self.num_fw_partitions)
                        ]

                    for partition_id in range(self.num_fw_partitions):
                        nids = self.partitioned_nids[partition_id]
                        partition_local_ids = torch.where((nids >= s_eid) & (nids < e_eid))[0]

                        fw_scope2localids[scope][partition_id] = torch.cat(
                            (fw_scope2localids[scope][partition_id], partition_local_ids), dim = 0
                        )

                global_eid += ns.num_nodes

            # Backward local indices
            for ns in self.nodes:
                for cs in ns.chs:
                    scope = cs.scope

                    s_nid = cs._output_ind_range[0]
                    e_nid = cs._output_ind_range[1]

                    if scope not in bk_scope2localids:
                        bk_scope2localids[scope] = [
                            torch.zeros([0], dtype = torch.long).to(self.partitioned_nids[0].device) for _ in range(self.num_bk_partitions)
                        ]

                    for partition_id in range(self.num_bk_partitions):
                        u_cids = self.partitioned_u_cids[partition_id]
                        partition_local_ids = torch.where((u_cids >= s_nid) & (u_cids < e_nid))[0]

                        bk_scope2localids[scope][partition_id] = torch.cat(
                            (bk_scope2localids[scope][partition_id], partition_local_ids), dim = 0
                        )

            self.fw_scope2localids = fw_scope2localids
            self.bk_scope2localids = bk_scope2localids

        return prod_scope_eleids
