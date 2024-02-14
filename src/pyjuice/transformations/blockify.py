from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from copy import deepcopy as pydeepcopy
from numba import njit
from typing import Optional, Dict, Sequence

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes, foldup_aggregate
from pyjuice.utils import BitSet
from pyjuice.utils.util import max_cdf_power_of_2


@njit
def _compute_param_target_ids_kernel(target_id0, target_id1, target_id2, edge_ids, new_edge_ids, 
                                     block_mul_size, ch_block_mul_size, block_size, ch_block_size):
    for i in range(edge_ids.shape[1]):
        old_ngid = edge_ids[0,i]
        old_cgid = edge_ids[1,i]

        for j in range(new_edge_ids.shape[1]):
            new_ngid = new_edge_ids[0,j]
            new_cgid = new_edge_ids[1,j]
            
            ng_sid = new_ngid * block_mul_size
            ng_eid = ng_sid + block_mul_size
            cg_sid = new_cgid * ch_block_mul_size
            cg_eid = cg_sid + ch_block_mul_size

            if (old_ngid >= ng_sid) and (old_ngid < ng_eid) and (old_cgid >= cg_sid) and (old_cgid < cg_eid):
                target_id0[i] = j

                target_id1[0,i] = (old_ngid - ng_sid) * block_size
                target_id1[1,i] = (old_ngid - ng_sid + 1) * block_size

                target_id2[0,i] = (old_cgid - cg_sid) * ch_block_size
                target_id2[1,i] = (old_cgid - cg_sid + 1) * ch_block_size
                break


@triton.jit
def _copy_params_kernel(new_params, params, target_id0, target_id1, target_id2, 
                        old_block_size: tl.constexpr, old_ch_block_size: tl.constexpr, 
                        new_block_size: tl.constexpr, new_ch_block_size: tl.constexpr, 
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N

    offs_pars = pid_b * (old_block_size * old_ch_block_size) + offs_m[:,None] * old_ch_block_size + offs_n[None,:]
    pars = tl.load(params + offs_pars)

    id0 = tl.load(target_id0 + pid_b)
    id1 = tl.load(target_id1 + pid_b)
    id2 = tl.load(target_id2 + pid_b)

    offs_npars = id0 * (new_block_size * new_ch_block_size) + (id1 + offs_m)[:,None] * new_ch_block_size + (id2 + offs_n)[None,:]
    tl.store(new_params + offs_npars, pars)


def blockify(root_ns: CircuitNodes, sparsity_tolerance: float = 0.25, max_target_block_size: int = 32, use_cuda: bool = True) -> CircuitNodes:
    """
    Generate an equivalent PC with potentially high block sizes.

    :param root_ns: the input PC
    :type root_ns: CircuitNodes

    :param sparsity_tolerance: allowed fraction of zero parameters to be added (should be in the range (0, 1])
    :type sparsity_tolerance: float

    :param max_target_block_size: the maximum block size to search for
    :type max_target_block_size: int

    :param use_cuda: use GPU when possible
    :type use_cuda: bool

    :returns: An equivalent `CircuitNodes`
    """

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    ## Do an initial pass to compute the maximum block size of every `ns` ##

    ns2block_size = dict()
    for ns in root_ns:
        if ns.is_input():
            ns2block_size[ns] = min(max_cdf_power_of_2(ns.num_nodes), max(ns.block_size, max_target_block_size))

        elif ns.is_prod():
            ns2block_size[ns] = min(max_cdf_power_of_2(ns.num_nodes), max(ns.block_size, max_target_block_size))

        else:
            assert ns.is_sum()

            old_block_size = ns.block_size
            old_ch_block_size = ns.ch_block_size
            edge_ids = ns.edge_ids

            old_ns_num_nblocks = ns.num_node_blocks
            old_cs_num_nblocks = sum([cs.num_node_blocks for cs in ns.chs])

            flag = False
            plausible_combinations = list()

            block_size = min(max_cdf_power_of_2(ns.num_nodes), max(ns.block_size, max_target_block_size))
            while block_size >= old_block_size:
                block_mul_size = block_size // old_block_size

                ns_num_nblocks = old_ns_num_nblocks // block_mul_size
                
                ch_block_size = ns2block_size[ns.chs[0]]
                while ch_block_size >= old_ch_block_size:
                    ch_block_mul_size = ch_block_size // old_ch_block_size

                    cs_num_nblocks = old_cs_num_nblocks // block_mul_size

                    n_edge_ids = edge_ids[0,:] // block_mul_size
                    c_edge_ids = edge_ids[1,:] // ch_block_mul_size
                    _, counts = torch.unique(n_edge_ids * cs_num_nblocks + c_edge_ids, return_counts = True)

                    if counts.float().mean() >= (1.0 - sparsity_tolerance) * block_mul_size * ch_block_mul_size:
                        plausible_combinations.append((block_size, ch_block_size))

                    ch_block_size = ch_block_size // 2

                block_size = block_size // 2

            # Find the best block size combination
            best_block_size = 0
            best_ch_block_size = 0
            for block_size, ch_block_size in plausible_combinations:
                if block_size >= 16 and ch_block_size >= 16:
                    best_block_size = block_size
                    best_ch_block_size = ch_block_size
                    break

            if best_block_size == 0:
                best_val = 0
                best_frac = 0
                for block_size, ch_block_size in plausible_combinations:
                    cond1 = block_size * ch_block_size > best_val
                    cond2 = (block_size * ch_block_size > best_val) and \
                        (max(block_size, ch_block_size) // min(block_size, ch_block_size) < best_frac)
                    if cond1 or cond2:
                        best_block_size = block_size
                        best_ch_block_size = ch_block_size
                        best_val = block_size * ch_block_size
                        best_frac = max(block_size, ch_block_size) // min(block_size, ch_block_size)

            if best_block_size == 0:
                best_block_size = old_block_size
                best_ch_block_size = old_ch_block_size

            ns2block_size[ns] = best_block_size
            for cs in ns.chs:
                ns2block_size[cs] = best_ch_block_size

    ## Do a second pass to finalize the block sizes ##

    for ns in root_ns:
        if ns.is_prod():
            block_size = ns2block_size[ns]
            for cs in ns.chs:
                block_size = min(block_size, ns2block_size[cs])

            ns2block_size[ns] = block_size
            for cs in ns.chs:
                ns2block_size[cs] = block_size

    ## Apply the new block sizes ##

    def update_ns(ns: CircuitNodes, ns_chs: Sequence[CircuitNodes]):
        new_block_size = ns2block_size[ns]
        block_mul_size = new_block_size // ns.block_size

        new_num_nblocks = ns.num_node_blocks // block_mul_size

        assert new_num_nblocks * new_block_size == ns.num_node_blocks * ns.block_size

        if ns.is_input():
            new_ns = InputNodes(
                num_node_blocks = new_num_nblocks,
                scope = pydeepcopy(ns.scope),
                dist = pydeepcopy(ns.dist),
                block_size = new_block_size
            )

            if not ns.is_tied():
                params = ns.get_params()
                if params is not None:
                    new_ns.set_params(params.clone(), normalize = False)

        elif ns.is_prod():
            edge_ids = ns.edge_ids.clone()
            edge_ids = edge_ids.reshape(new_num_nblocks, block_mul_size, ns.num_chs)
            if torch.all(edge_ids[:,1:,:] - edge_ids[:,:-1,:]) == 1:
                # Block-sparse mode
                edge_ids = edge_ids[:,0,:].contiguous() // block_mul_size
                mode = "block_sparse"
            else:
                # Sparse mode
                edge_ids = (edge_ids.reshape(ns.num_node_blocks, ns.num_chs)[:,None,:] * ns.block_size + \
                    torch.arange(0, ns.block_size)[None,:,None]).flatten(0, 1)
                mode = "sparse"

            new_ns = ProdNodes(
                num_node_blocks = new_num_nblocks,
                chs = ns_chs,
                edge_ids = edge_ids,
                block_size = new_block_size
            )

            if mode == "block_sparse":
                assert new_ns.is_block_sparse()
            elif mode == "sparse":
                assert new_ns.is_sparse()

        else:
            assert ns.is_sum()

            old_num_nblocks = ns.num_node_blocks
            old_num_cblocks = sum([cs.num_node_blocks for cs in ns.chs])

            new_ch_block_size = ns2block_size[ns.chs[0]]
            ch_block_mul_size = new_ch_block_size // ns.chs[0].block_size

            new_num_cblocks = old_num_cblocks // ch_block_mul_size

            edge_ids = ns.edge_ids.clone()
            grid_edge_ids = torch.zeros([old_num_nblocks, old_num_cblocks], dtype = torch.bool)
            grid_edge_ids[edge_ids[0,:],edge_ids[1,:]] = True

            grid_edge_ids = grid_edge_ids.reshape(new_num_nblocks, block_mul_size, new_num_cblocks, ch_block_mul_size)
            new_edge_ids = torch.nonzero(grid_edge_ids.any(dim = 3).any(dim = 1), as_tuple = False).permute(1, 0)

            new_ns = SumNodes(
                num_node_blocks = new_num_nblocks,
                chs = ns_chs,
                edge_ids = new_edge_ids,
                block_size = new_block_size
            )
            
            if not ns.is_tied():
                # Collect selected blocks
                grid_edge_ids = grid_edge_ids.permute(0, 2, 1, 3).flatten(0, 1)
                block_ids = new_edge_ids[0,:] * new_num_cblocks + new_edge_ids[1,:]
                param_indicator = grid_edge_ids[block_ids,:,:]
                if not torch.all(param_indicator):
                    param_indicator = param_indicator[:,:,None,:,None].repeat(1, 1, ns.block_size, 1, ns.chs[0].block_size)
                    param_indicator = param_indicator.flatten(3, 4).flatten(1, 2)
                    zero_param_mask = ~param_indicator

                    new_ns.set_zero_param_mask(zero_param_mask)

                params = ns.get_params()
                if params is not None:
                    old_block_size = ns.block_size
                    old_ch_block_size = ns.chs[0].block_size
                    if new_block_size > old_block_size or new_ch_block_size > old_ch_block_size:
                        new_params = torch.zeros([new_edge_ids.size(1), new_block_size, new_ch_block_size], device = device)
                        if new_edge_ids.size(1) == new_num_nblocks * new_num_cblocks and params.numel() == new_params.numel():
                            # Fully-connected parameters
                            new_params = params.reshape(
                                new_num_nblocks, block_mul_size, new_num_cblocks, ch_block_mul_size, old_block_size, old_ch_block_size
                            ).permute(0, 2, 1, 4, 3, 5).reshape(new_params.size()).contiguous()
                        elif use_cuda:
                            edge_ids_np = edge_ids.numpy()
                            new_edge_ids_np = new_edge_ids.numpy()

                            target_id0 = np.zeros([edge_ids.size(1)], dtype = np.int64) - 1
                            target_id1 = np.zeros([2, edge_ids.size(1)], dtype = np.int64) - 1
                            target_id2 = np.zeros([2, edge_ids.size(1)], dtype = np.int64) - 1
                            
                            _compute_param_target_ids_kernel(
                                target_id0, target_id1, target_id2, edge_ids_np, new_edge_ids_np, 
                                block_mul_size, ch_block_mul_size, old_block_size, old_ch_block_size
                            )

                            target_id0 = torch.from_numpy(target_id0).to(device)
                            target_id1 = torch.from_numpy(target_id1).to(device)
                            target_id2 = torch.from_numpy(target_id2).to(device)

                            params = params.to(device)

                            BLOCK_M = min(32, old_block_size)
                            BLOCK_N = min(32, old_ch_block_size)

                            grid = (old_ch_block_size // BLOCK_N, old_block_size // BLOCK_M, edge_ids.size(1))

                            _copy_params_kernel[grid](
                                new_params, params, target_id0, target_id1, target_id2, 
                                old_block_size = old_block_size, 
                                old_ch_block_size = old_ch_block_size, 
                                new_block_size = new_block_size, 
                                new_ch_block_size = new_ch_block_size, 
                                BLOCK_M = BLOCK_M, 
                                BLOCK_N = BLOCK_N
                            )

                        else:
                            for par_block_id in range(new_edge_ids.size(1)):
                                nsid = new_edge_ids[0,par_block_id] * block_mul_size
                                neid = nsid + block_mul_size
                                csid = new_edge_ids[1,par_block_id] * ch_block_mul_size
                                ceid = csid + ch_block_mul_size

                                blk_ids = torch.where((edge_ids[0,:] >= nsid) & (edge_ids[0,:] < neid) & (edge_ids[1,:] >= csid) & (edge_ids[1,:] < ceid))[0]
                                for blk_id in blk_ids:
                                    nid0, nid1 = (edge_ids[0,blk_id] - nsid) * ns.block_size, (edge_ids[0,blk_id] - nsid + 1) * ns.block_size
                                    cid0, cid1 = (edge_ids[1,blk_id] - csid) * ns.chs[0].block_size, (edge_ids[1,blk_id] - csid + 1) * ns.chs[0].block_size
                                    new_params[par_block_id,nid0:nid1,cid0:cid1] = params[blk_id,:,:]

                        new_ns.set_params(new_params.cpu(), normalize = False)
                    else:
                        new_ns.set_params(params, normalize = False)

        return new_ns

    old2new = dict()
    new_root_ns = foldup_aggregate(update_ns, root_ns, cache = old2new)

    # Re-link tied nodes to their source
    for ns in root_ns:
        if ns.is_tied():
            new_source_ns = old2new[ns.get_source_ns()]
            new_ns = old2new[ns]
            new_ns.set_source_ns(new_source_ns)

    return new_root_ns


def unblockify(root_ns: CircuitNodes, block_size: int = 1, recursive: bool = True, keys_to_copy: Optional[Sequence[str]] = None):
    """
    Decrease the block size of a PC.

    :param root_ns: the input PC
    :type root_ns: CircuitNodes

    :param block_size: the target block size
    :type block_size: int

    :param recursive: whether to do it recursively or just for the current node
    :type recursive: bool

    :param keys_to_copy: an optional dictionary of properties to copy
    :type keys_to_copy: Optional[Sequence[str]]

    :returns: An equivalent `CircuitNodes`
    """
    
    def update_ns(ns: CircuitNodes, ns_chs: Sequence[CircuitNodes]):
        new_block_size = min(block_size, ns.block_size)
        new_num_nblocks = ns.num_nodes // new_block_size

        if ns.is_input():
            new_ns = InputNodes(
                num_node_blocks = new_num_nblocks,
                scope = pydeepcopy(ns.scope),
                dist = pydeepcopy(ns.dist),
                block_size = new_block_size
            )

            if not ns.is_tied():
                params = ns.get_params()
                if params is not None:
                    new_ns.set_params(params.clone(), normalize = False)

        elif ns.is_prod():
            if ns.is_block_sparse():
                gsize_redu = ns.block_size // new_block_size
                edge_ids = ns.edge_ids.clone()
                edge_ids = (edge_ids[:,None,:].repeat(1, gsize_redu, 1) * gsize_redu + \
                    torch.arange(0, gsize_redu)[None,:,None]).reshape(ns.num_node_blocks * gsize_redu, ns.num_chs)

            else:
                edge_ids = ns.edge_ids.clone()

            new_ns = ProdNodes(
                num_node_blocks = new_num_nblocks,
                chs = ns_chs,
                edge_ids = edge_ids,
                block_size = new_block_size
            )

        else:
            assert ns.is_sum()

            new_ch_block_size = ns_chs[0].block_size
            gsize_redu = ns.block_size // new_block_size
            ch_gsize_redu = ns.ch_block_size // new_ch_block_size

            grid_x, grid_y = torch.meshgrid(
                torch.arange(0, gsize_redu), 
                torch.arange(0, ch_gsize_redu), 
                indexing = 'ij'
            )

            edge_ids = ns.edge_ids.clone()
            edge_ids = edge_ids[:,None,:].repeat(1, gsize_redu * ch_gsize_redu, 1)
            edge_ids[0,:,:] = edge_ids[0,:,:] * gsize_redu + grid_x.reshape(-1)[:,None]
            edge_ids[1,:,:] = edge_ids[1,:,:] * ch_gsize_redu + grid_y.reshape(-1)[:,None]
            edge_ids = edge_ids.reshape(2, ns.edge_ids.size(1) * gsize_redu * ch_gsize_redu)

            new_ns = SumNodes(
                num_node_blocks = new_num_nblocks,
                chs = ns_chs,
                edge_ids = edge_ids,
                block_size = new_block_size
            )
            
            if not ns.is_tied() and ns.has_params():
                params = ns._params.clone()
                params = params.reshape(
                    ns.edge_ids.size(1), gsize_redu, new_block_size, ch_gsize_redu, new_ch_block_size
                ).permute(0, 1, 3, 2, 4).reshape(
                    ns.edge_ids.size(1) * gsize_redu * ch_gsize_redu, new_block_size, new_ch_block_size
                )

                new_ns.set_params(params)

                if keys_to_copy is not None:
                    for key in keys_to_copy:
                        if hasattr(ns, key):
                            vals = getattr(ns, key)

                            if vals.dim() == 3:
                                vals = vals.reshape(
                                    ns.edge_ids.size(1), gsize_redu, new_block_size, ch_gsize_redu, new_ch_block_size
                                ).permute(0, 1, 3, 2, 4).reshape(
                                    ns.edge_ids.size(1) * gsize_redu * ch_gsize_redu, new_block_size, new_ch_block_size
                                )
                                setattr(new_ns, key, vals)
                            else:
                                raise NotImplementedError()

            assert new_ns._params.size(0) == new_ns.edge_ids.size(1)

        return new_ns

    old2new = dict()
    new_root_ns = foldup_aggregate(update_ns, root_ns, cache = old2new)

    # Re-link tied nodes to their source
    for ns in root_ns:
        if ns.is_tied():
            new_source_ns = old2new[ns.get_source_ns()]
            new_ns = old2new[ns]
            new_ns.set_source_ns(new_source_ns)

    return new_root_ns


def bump_block_size(ns: CircuitNodes, block_size: int, use_cuda: bool = True):
    assert block_size > ns.block_size, f"`block_size` already greater than {block_size}."
    assert ns.num_nodes % block_size == 0, f"`num_nodes` not divicible by the target block size."

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    block_mul_size = block_size // ns.block_size

    if ns.is_input():
        new_ns = InputNodes(
            num_node_blocks = new_num_nblocks,
            scope = pydeepcopy(ns.scope),
            dist = pydeepcopy(ns.dist),
            block_size = block_size
        )

        if not ns.is_tied():
            params = ns.get_params()
            if params is not None:
                new_ns.set_params(params.clone(), normalize = False)

    elif ns.is_prod():
        edge_ids = ns.edge_ids.clone()
        edge_ids = edge_ids.reshape(new_num_nblocks, block_mul_size, ns.num_chs)
        if torch.all(edge_ids[:,1:,:] - edge_ids[:,:-1,:]) == 1:
            # Block-sparse mode
            edge_ids = edge_ids[:,0,:].contiguous()
            mode = "block_sparse"
        else:
            # Sparse mode
            edge_ids = (edge_ids.reshape(ns.num_node_blocks, ns.num_chs)[:,None,:] * ns.block_size + \
                torch.arange(0, ns.block_size)[None,:,None]).flatten(0, 1)
            mode = "sparse"
        
        new_ns = ProdNodes(
            num_node_blocks = new_num_nblocks,
            chs = ns_chs,
            edge_ids = edge_ids,
            block_size = block_size
        )

        if mode == "block_sparse":
            assert new_ns.is_block_sparse()
        elif mode == "sparse":
            assert new_ns.is_sparse()
    if ns.is_sum():
        old_num_nblocks = ns.num_node_blocks
        old_block_size = ns.block_size
        num_cblocks = sum([cs.num_node_blocks for cs in ns.chs])

        new_num_nblocks = ns.num_nodes // block_size

        ns_chs = ns.chs
        ch_block_size = ns.chs[0].block_size

        edge_ids = ns.edge_ids.clone()
        grid_edge_ids = torch.zeros([old_num_nblocks, num_cblocks], dtype = torch.bool)
        grid_edge_ids[edge_ids[0,:],edge_ids[1,:]] = True

        grid_edge_ids = grid_edge_ids.reshape(new_num_nblocks, block_mul_size, num_cblocks, 1)
        new_edge_ids = torch.nonzero(grid_edge_ids.any(dim = 3).any(dim = 1), as_tuple = False).permute(1, 0)

        new_ns = SumNodes(
            num_node_blocks = new_num_nblocks,
            chs = ns_chs,
            edge_ids = new_edge_ids,
            block_size = block_size
        )

        if not ns.is_tied():
            # Collect selected blocks
            grid_edge_ids = grid_edge_ids.permute(0, 2, 1, 3).flatten(0, 1)
            block_ids = new_edge_ids[0,:] * num_cblocks + new_edge_ids[1,:]
            param_indicator = grid_edge_ids[block_ids,:,:]
            if not torch.all(param_indicator):
                param_indicator = param_indicator[:,:,None,:,None].repeat(1, 1, ns.block_size, 1, ns.chs[0].block_size)
                param_indicator = param_indicator.flatten(3, 4).flatten(1, 2)
                zero_param_mask = ~param_indicator

                new_ns.set_zero_param_mask(zero_param_mask)

            params = ns.get_params()
            if params is not None:
                new_params = torch.zeros([new_edge_ids.size(1), block_size, ch_block_size], device = device)
                if new_edge_ids.size(1) == new_num_nblocks * num_cblocks:
                    # Fully-connected parameters
                    new_params = params.reshape(
                        new_num_nblocks, block_size // old_block_size, num_cblocks, 1, old_block_size, ch_block_size
                    ).permute(0, 2, 1, 4, 3, 5).reshape(new_params.size()).contiguous()
                elif use_cuda:
                    edge_ids_np = edge_ids.numpy()
                    new_edge_ids_np = new_edge_ids.numpy()

                    target_id0 = np.zeros([edge_ids.size(1)], dtype = np.int64) - 1
                    target_id1 = np.zeros([2, edge_ids.size(1)], dtype = np.int64) - 1
                    target_id2 = np.zeros([2, edge_ids.size(1)], dtype = np.int64) - 1
                    
                    _compute_param_target_ids_kernel(
                        target_id0, target_id1, target_id2, edge_ids_np, new_edge_ids_np, 
                        block_mul_size, 1, old_block_size, ch_block_size
                    )

                    target_id0 = torch.from_numpy(target_id0).to(device)
                    target_id1 = torch.from_numpy(target_id1).to(device)
                    target_id2 = torch.from_numpy(target_id2).to(device)

                    params = params.to(device)

                    BLOCK_M = min(32, old_block_size)
                    BLOCK_N = min(32, ch_block_size)

                    grid = (ch_block_size // BLOCK_N, old_block_size // BLOCK_M, edge_ids.size(1))

                    _copy_params_kernel[grid](
                        new_params, params, target_id0, target_id1, target_id2, 
                        old_block_size = old_block_size, 
                        old_ch_block_size = ch_block_size, 
                        new_block_size = block_size, 
                        new_ch_block_size = ch_block_size, 
                        BLOCK_M = BLOCK_M, 
                        BLOCK_N = BLOCK_N
                    )

                else:
                    for par_block_id in range(new_edge_ids.size(1)):
                        nsid = new_edge_ids[0,par_block_id] * block_mul_size
                        neid = nsid + block_mul_size
                        csid = new_edge_ids[1,par_block_id]
                        ceid = csid + 1

                        blk_ids = torch.where((edge_ids[0,:] >= nsid) & (edge_ids[0,:] < neid) & (edge_ids[1,:] >= csid) & (edge_ids[1,:] < ceid))[0]
                        for blk_id in blk_ids:
                            nid0, nid1 = (edge_ids[0,blk_id] - nsid) * ns.block_size, (edge_ids[0,blk_id] - nsid + 1) * ns.block_size
                            cid0, cid1 = (edge_ids[1,blk_id] - csid) * ns.chs[0].block_size, (edge_ids[1,blk_id] - csid + 1) * ns.chs[0].block_size
                            new_params[par_block_id,nid0:nid1,cid0:cid1] = params[blk_id,:,:]

                new_ns.set_params(new_params.cpu(), normalize = False)

    return new_ns
