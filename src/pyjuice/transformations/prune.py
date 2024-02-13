from __future__ import annotations

import numpy as np
import torch

from typing import Optional, Dict, Union, Sequence, Tuple

from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes, foreach, foldup_aggregate, summate, multiply

Tensor = Union[np.ndarray,torch.Tensor]


def prune_by_score(root_nodes: CircuitNodes, key: str = "_scores", scores: Optional[Dict[CircuitNodes,Tensor]] = None, 
                   keep_frac: Optional[float] = None, score_threshold: Optional[float] = None, block_reduction: str = "sum"):
    
    # Traverse all nodes to collect scores
    score_ranges = dict()
    score_start = 0
    flat_scores = []

    def _get_scores(ns: CircuitNodes):

        nonlocal score_start

        if scores is not None:
            if ns in scores:
                curr_scores = scores[ns]
            else:
                curr_scores = None
        else:
            if hasattr(ns, key):
                curr_scores = ns.__dict__[key]
            else:
                curr_scores = None

        if curr_scores is None or ns._source_node is not None:
            # We do not consider nodes with no specified scores and parameter-tied nodes
            return None

        if isinstance(curr_scores, np.ndarray):
            curr_scores = torch.from_numpy(curr_scores)

        if curr_scores.dim() == 3:
            if block_reduction == "sum":
                curr_scores = curr_scores.sum(dim = 2).sum(dim = 1)
            elif block_reduction == "mean":
                curr_scores = curr_scores.mean(dim = 2).mean(dim = 1)
            elif block_reduction == "max":
                curr_scores = curr_scores.max(dim = 2).max(dim = 1)
            else:
                raise ValueError(f"Unknown block reduction method `{block_reduction}`.")
        else:
            assert curr_scores.dim() == 1

        flat_scores.append(curr_scores)
        num_scores = curr_scores.size(0)

        score_ranges[ns] = (score_start, score_start + num_scores)
        score_start += num_scores

    for ns in root_nodes:
        _get_scores(ns)

    # Indices to keep
    flat_scores = torch.cat(flat_scores, dim = 0)
    if flat_scores.dim() == 3:
        flat_scores = flat_scores.sum(dim = 2).sum(dim = 1)
    if keep_frac is not None:
        assert score_threshold is None, "Only one of `keep_frac` and `score_threshold` should be set."
        score_threshold = torch.quantile(flat_scores, 1.0 - keep_frac, dim = 0)
        selected_edges = (flat_scores >= score_threshold)
    else:
        assert score_threshold is not None, "At least one of `keep_frac` and `score_threshold` should be set."
        selected_edges = (flat_scores >= score_threshold)

    # Prune circuit

    def _construct_pruned_circuit(ns: CircuitNodes, ch_outputs: Sequence[CircuitNodes], dup2source: Dict[CircuitNodes,CircuitNodes] = dict()):
        if isinstance(ns, InputNodes):
            new_ns = ns.duplicate(tie_params = False)

        elif isinstance(ns, ProdNodes):
            new_ns = ns.duplicate(*ch_outputs, tie_params = False)

        else:
            assert isinstance(ns, SumNodes)

            if ns in score_ranges:
                # Prune away edge parameters
                edge_ids = ns.edge_ids.clone()
                edge_filter = selected_edges[score_ranges[ns][0]:score_ranges[ns][1]]
                copied_edges = []
                copied_params = []
                for node_id in range(ns.num_node_blocks):
                    curr_eids = (edge_ids[0,:] == node_id) * edge_filter
                    if curr_eids.sum().item() == 0:
                        maxid = torch.argmax(
                            flat_scores[score_ranges[ns][0]:score_ranges[ns][1]] * (edge_ids[0,:] == node_id) + 
                            (edge_ids[0,:] == node_id) * 1e-8
                        )
                        copied_edges.append(edge_ids[:,maxid].unsqueeze(1))
                        copied_params.append(ns._params[maxid].reshape(1, ns.block_size, ns.ch_block_size))
                    else:
                        copied_edges.append(edge_ids[:,curr_eids])
                        copied_params.append(ns._params[curr_eids,:,:])

                edge_ids = torch.cat(copied_edges, dim = 1)
                params = torch.cat(copied_params, dim = 0)

                assert edge_ids.size(1) == params.size(0)

                new_ns = SumNodes(
                    num_node_blocks = ns.num_node_blocks, 
                    chs = ch_outputs, 
                    edge_ids = edge_ids,
                    params = params,
                    block_size = ns.block_size
                )
            else:
                # Keep the node as-is
                new_ns = ns.duplicate(*ch_outputs, tie_params = False)

        if ns._source_node is not None:
            dup2source[new_ns] = ns._source_node

        return new_ns

    old2new = dict()
    dup2source = dict()
    new_root_ns = foldup_aggregate(_construct_pruned_circuit, root_nodes, cache = old2new, dup2source = dup2source)

    for ns in dup2source:
        ns_source = old2new[dup2source[ns]]
        ns._source_node = ns_source

        assert ns.num_node_blocks == ns_source.num_node_blocks and ns.block_size == ns_source.block_size

        if hasattr(ns_source, "edge_ids"):
            ns.edge_ids = ns_source.edge_ids.clone()

    return new_root_ns