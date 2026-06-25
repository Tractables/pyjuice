from __future__ import annotations

import os
import torch
import torch.nn as nn
import triton
import warnings
import time
from typing import Sequence, Optional

# Small-batch (batch < 16) node-tile cap for the 2D product kernel. The default BLOCK_M heuristic
# targets a fixed ~2048-element tile, which at tiny batch balloons BLOCK_M up to `block_size` -- one
# serial program per node-block (~1 SM busy). Capping BLOCK_M fans the node dimension across many
# programs to fill the SMs (the kernel walks BLOCK_M nodes serially, so this is pure tiling and leaves
# results bit-identical). Env-overridable for tuning.
_SMALL_BATCH_PROD_TILE_M = int(os.environ.get("PYJUICE_SB_PROD_TM", 8))

from pyjuice.nodes import ProdNodes
from pyjuice.utils.parameter_list import FastParamList
from .kernels import prod as kernels
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
        assert len(nodes) == len(set(nodes)), "Input node list contains duplicates."

        use_block_sparse_edges = True
        for nid in range(0, len(nodes)):
            if nodes[nid].is_sparse() or nodes[0].block_size != nodes[nid].block_size:
                use_block_sparse_edges = False
                break
        self.use_block_sparse_edges = use_block_sparse_edges

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
        self._layer_nid_range = (global_nid_start, global_nid_start + self.num_nodes)

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

    def forward(self, node_mars: torch.Tensor, element_mars: torch.Tensor, _for_backward: bool = False, **kwargs) -> None:
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

    def backward(self, node_flows: torch.Tensor, element_flows: torch.Tensor, logspace_flows: bool = False, **kwargs) -> None:
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

                self._forward_backward(node_flows, element_flows, u_cids, parids, local_ids = local_ids, accum = True,
                                       prop_logsumexp = logspace_flows)
        
        else:
            # Evaluate the whole layer
            for partition_id in range(self.num_bk_partitions):
                u_cids = self.partitioned_u_cids[partition_id]
                parids = self.partitioned_parids[partition_id]

                self._forward_backward(node_flows, element_flows, u_cids, parids, accum = True,
                                       prop_logsumexp = logspace_flows)
        
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

    def is_prod(self):
        return True

    def __repr__(self):
        return f"ProdLayer(nid_range=({self._layer_nid_range[0]}, {self._layer_nid_range[1]}), num_nodes={self.num_nodes}, num_edges={self.num_edges})"

    def _forward_backward(self, node_vals: torch.Tensor, element_vals: torch.Tensor,
                          nids: torch.Tensor, cids: torch.Tensor, local_ids: Optional[torch.Tensor] = None,
                          accum: bool = False, prop_logsumexp: bool = False) -> None:
        tot_n_nodes = node_vals.size(0)
        tot_n_eles = element_vals.size(0)
        n_nblocks = nids.size(0) if local_ids is None else local_ids.size(0)
        num_edges = cids.size(1)
        batch_size = node_vals.size(1)

        block_size = self.block_size
        partial_eval = local_ids is not None

        assert num_edges & (num_edges - 1) == 0, "`num_edges` must be a power of 2."

        # Special case: every node have > 2048 edges
        if num_edges > 2048:

            BLOCK_N = 2048
            BLOCK_B = 1

            grid = (n_nblocks * self.block_size, batch_size)
            
            kernels._forward_backward_kernel_large[grid](
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
                BLOCK_N = BLOCK_N, 
                BLOCK_B = BLOCK_B, 
                N_NUM_BLKS = triton.cdiv(num_edges, BLOCK_B), 
                block_size = block_size, 
                accum = accum, 
                partial_eval = partial_eval,
                prop_logsumexp = prop_logsumexp
            )

            return None

        if not triton.__version__ == "2.0.0":

            BLOCK_B = min(2048 // num_edges, triton.next_power_of_2(batch_size))
            BLOCK_M = min(max(2048 // (BLOCK_B * num_edges), 1), self.block_size)

            # Small/gap-batch: cap BLOCK_M so the node dimension fans out across many programs (one
            # serial program per node-block otherwise leaves ~1 SM busy). The budget heuristic above
            # under-tiles up to batch ~128, well past the small-batch (<16) regime, so the cap is
            # applied through the gap (< 64). Pure tiling -> bit-identical (BLOCK_M only tiles nodes).
            if batch_size < 64:
                BLOCK_M = min(BLOCK_M, _SMALL_BATCH_PROD_TILE_M)

            grid = (triton.cdiv(n_nblocks * self.block_size, BLOCK_M), triton.cdiv(batch_size, BLOCK_B))

            kernels._forward_backward_kernel_2d[grid](
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
                partial_eval = partial_eval,
                prop_logsumexp = prop_logsumexp
            )

        else:

            BLOCK_B = min(1024 // num_edges, triton.next_power_of_2(batch_size))
            BLOCK_M = min(max(1024 // (BLOCK_B * num_edges), 1), triton.next_power_of_2(n_nblocks) * self.block_size)

            grid = (triton.cdiv(n_nblocks * self.block_size, BLOCK_M), triton.cdiv(batch_size, BLOCK_B))

            kernels._forward_backward_kernel_3d[grid](
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
                partial_eval = partial_eval,
                prop_logsumexp = prop_logsumexp
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
