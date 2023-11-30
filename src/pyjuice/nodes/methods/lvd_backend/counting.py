from __future__ import annotations

import torch
import triton
import triton.language as tl
import math

from typing import Optional

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from pyjuice.nodes.distributions import Categorical


@triton.jit
def _pairwise_count_kernel(data1_ptr, data2_ptr, pairwise_count_ptr, num_samples: tl.constexpr,
                           n_cls1: tl.constexpr, n_cls2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_samples

    cid1 = tl.load(data1_ptr + offsets, mask = mask, other = 0)
    cid2 = tl.load(data2_ptr + offsets, mask = mask, other = 0)
    cid = cid1 * n_cls2 + cid2

    tl.atomic_add(pairwise_count_ptr + cid, 1, mask = mask)


def get_pairwise_count(data1: torch.Tensor, data2: torch.Tensor, n_cls1: int, n_cls2: int, 
                       device: Optional[torch.device] = None, BLOCK_SIZE = 2048):
    assert data1.min() >= 0 and data1.max() < n_cls1, f"Value range of `data1` exceeds limit: [Min: {data1.min().item()}, Max: {data1.max().item()}]."
    assert data2.min() >= 0 and data2.max() < n_cls2, f"Value range of `data2` exceeds limit: [Min: {data2.min().item()}, Max: {data2.max().item()}]."
    assert data1.size(0) == data2.size(0), "`data1` and `data2` must have the same number of examples."

    if device is not None:
        data1 = data1.to(device)
        data2 = data2.to(device)

    if data1.is_cuda:

        data1 = data1.long()
        data2 = data2.long()

        num_samples = data1.size(0)
        pairwise_count = torch.zeros([n_cls1, n_cls2], dtype = torch.float32, device = data1.device)

        grid = lambda meta: (triton.cdiv(num_samples, meta['BLOCK_SIZE']),)

        _pairwise_count_kernel[grid](
            data1_ptr = data1, 
            data2_ptr = data2,
            pairwise_count_ptr = pairwise_count,
            num_samples = num_samples,
            n_cls1 = n_cls1,
            n_cls2 = n_cls2,
            BLOCK_SIZE = BLOCK_SIZE
        )

    else:
        pairwise_count = torch.bincount(data1 * n_cls2 + data2, minlength = n_cls1 * n_cls2)
        pairwise_count = pairwise_count.reshape(n_cls1, n_cls2)

    return pairwise_count


def lvd_for_input_nodes(lvdistiller, ns, lv_dataset: torch.Tensor, obs_dataset: torch.Tensor):
    if isinstance(ns.dist, Categorical):
        pairwise_count = get_pairwise_count(lv_dataset, obs_dataset, ns.num_nodes, ns.dist.num_cats)
        pairwise_count = pairwise_count + lvdistiller.pseudocount
        params = pairwise_count / (pairwise_count.sum(dim = 1, keepdim = True) + 1e-8)
        ns.set_params(params)
    else:
        raise NotImplementedError(f"Input LVD function not implemented for distribution type {type(ns.dist)}.")


def lvd_by_counting(lvdistiller, ns: CircuitNodes):
    lv_dataset_id = lvdistiller.ns2lv_dataset_id[ns]
    if ns.is_prod():
        for cs in ns.chs:
            if cs not in lvdistiller.ns2lv_dataset_id:
                lvdistiller.ns2lv_dataset_id[cs] = lv_dataset_id

    # Get candidate LVD nodes
    nodes_for_lvd = []
    if ns.is_sum():
        if any([cs in lvdistiller.ns2lv_dataset_id for cs in ns.chs]):
            nodes_for_lvd.append(ns)
    elif ns.is_prod():
        for cs in ns.chs:
            if any([ccs in lvdistiller.ns2lv_dataset_id for ccs in cs.chs]):
                nodes_for_lvd.append(cs)
            elif cs.is_input() and cs in lvdistiller.ns2lv_dataset_id and cs in lvdistiller.ns2obs_dataset_id:
                nodes_for_lvd.append(cs)
    elif ns.is_input() and ns in lvdistiller.ns2lv_dataset_id and ns in lvdistiller.ns2obs_dataset_id:
        nodes_for_lvd.append(ns)

    # Run LVD by counting
    for ns in nodes_for_lvd:
        if ns.is_input():
            ns_lv_dataset = lvdistiller.lv_datasets[lvdistiller.ns2lv_dataset_id[ns]]
            ns_obs_dataset = lvdistiller.obs_datasets[lvdistiller.ns2obs_dataset_id[ns]]
            lvd_for_input_nodes(lvdistiller, ns, ns_lv_dataset, ns_obs_dataset)
        else:
            assert ns.is_sum(), "Product nodes cannot apply LVD."
            
            ns_dataset = lvdistiller.lv_datasets[lvdistiller.ns2lv_dataset_id[ns]]
            num_ch_nodes = sum([cs.num_nodes for cs in ns.chs])
            edge_params = torch.empty([ns.num_nodes, num_ch_nodes], dtype = torch.float32, device = ns_dataset.device)
            sid = 0
            for cs in ns.chs:
                eid = sid + cs.num_nodes
                if cs in lvdistiller.ns2lv_dataset_id:
                    cs_dataset = lvdistiller.lv_datasets[lvdistiller.ns2lv_dataset_id[cs]]
                    pairwise_count = get_pairwise_count(ns_dataset, cs_dataset, ns.num_nodes, cs.num_nodes)
                else:
                    # Randomly initialize parameters
                    pairwise_count = torch.exp(-torch.rand([ns.num_nodes, cs.num_nodes]) * 2.0)

                edge_params[:,sid:eid] = pairwise_count

                sid = eid

            # Pruning
            if "prune_frac" in lvdistiller.kwargs and lvdistiller.kwargs["prune_frac"] != 0.0:
                assert 0.0 <= lvdistiller.kwargs["prune_frac"] < 1.0
                edge_mask = ns._get_edges_as_mask()
                num_edges_to_keep = int(math.ceil(ns.num_edges() * (1.0 - lvdistiller.kwargs["prune_frac"])))
                edge_params[~edge_mask] = 0
                _, indices = torch.topk(edge_params.flatten(), k = num_edges_to_keep)
                edge_mask = torch.zeros([ns.num_nodes * ns.num_ch_nodes], dtype = torch.bool)
                edge_mask[indices] = True
                edge_mask = edge_mask.reshape(ns.num_nodes, ns.num_ch_nodes)

                # Make sure that every node has at least one child
                dnids = torch.where(edge_mask.sum(dim = 1) == 0)
                amaxes = torch.argmax(edge_params[dnids,:], dim = 1)
                edge_mask[dnids,amaxes] = True

                # Update edges
                ns._construct_edges(edge_mask)

            edge_params /= edge_params.sum(dim = 1, keepdim = True) + 1e-8
            edge_params = edge_params[:,:,None,None]

            ns.set_params(edge_params, pseudocount = lvdistiller.pseudocount)