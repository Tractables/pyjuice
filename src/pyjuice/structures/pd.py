from __future__ import annotations

import torch
import random
from copy import deepcopy
from functools import reduce

import pyjuice.transformations as jtf
from typing import Tuple, Sequence, Optional, Type, Dict
from pyjuice.nodes import multiply, summate, inputs, set_block_size
from pyjuice.nodes.distributions import *
from pyjuice.structures.hclt import HCLT
from pyjuice.utils import BitSet
from pyjuice.utils.util import max_cdf_power_of_2


def PD(data_shape: Tuple, num_latents: int, 
       split_intervals: Optional[Union[int, Tuple[int]]] = None, 
       split_points: Optional[Sequence[Sequence[int]]] = None,
       max_split_depth: Optional[int] = None,
       max_prod_block_conns: int = 4,
       structure_type: str = "sum_dominated",
       input_layer_fn: Optional[Callable] = None,
       input_dists: Optional[Distribution] = None,
       input_layer_type: Type[Distribution] = Categorical, 
       input_layer_params: Dict = {"num_cats": 256},
       use_linear_mixing: bool = False,
       block_size: Optional[int] = None):
    """
    Generate PCs with the PD structure (https://arxiv.org/pdf/1202.3732.pdf).

    :param data_shape: shape of the data (e.g., [H, W, 3] for images and [S] for sequences)
    :type data_shape: Tuple

    :param num_latents: size of the latent space
    :type num_latents: int

    :param split_intervals: a sequence of integers specifying the interval between split points in every dimension; either this or `split_points` needs to be specified
    :type split_intervals: Optional[Union[int, Tuple[int]]]

    :param split_points: a sequence of split points in each dimension; either this or `split_intervals` needs to be specified
    :type split_points: Optional[Sequence[Sequence[int]]]

    :param max_split_depth: maximum depth of the splits
    :type max_split_depth: Optional[int]

    :param max_prod_block_conns: the maximum number of product nodes connected to every sum node
    :type max_prod_block_conns: int

    :param structure_type: whether to reuse sum nodes; no reuse: "sum_dominated", reuse: "prod_dominated"
    :type structure_type: str

    :param input_dist: input distribution
    :type input_dist: Distribution

    :param block_size: block size
    :type block_size: int
    """
    assert structure_type in ["sum_dominated", "prod_dominated"]

    if input_dist is not None:
        input_node_type, input_node_params = input_dist._get_constructor()

    # Specify block size
    if block_size is None:
        if num_latents <= 32:
            block_size = min(16, max_cdf_power_of_2(num_latents))
        else:
            block_size = min(32, max_cdf_power_of_2(num_latents))

    num_node_blocks = num_latents // block_size

    num_axes = len(data_shape)

    # Construct split points
    if split_intervals is not None:
        if isinstance(split_intervals, int):
            split_intervals = (split_intervals,) * num_axes

        split_points = [
            [j for j in range(split_intervals[axis], data_shape[axis], split_intervals[axis])] 
            for axis in range(num_axes)
        ]
    else:
        assert split_points is not None
        assert len(split_points) == num_axes
        for axis in range(num_axes):
            assert all(map(lambda x: 0 < x < data_shape[axis] - 1, split_points[axis]))

    global_split_points = split_points

    # Cache
    hypercube2ns = dict()

    def hypercube2scope(hypercube):
        vars = None
        cum_id = 1
        for axis in range(num_axes - 1, -1, -1):
            if vars is None:
                vars = torch.arange(hypercube[0][axis], hypercube[1][axis])
            else:
                s, e = hypercube[0][axis], hypercube[1][axis]
                vars = vars[None,:].repeat(e - s, 1) + (torch.arange(s, e) * cum_id)[:,None]
                vars = vars.reshape(-1)

            cum_id *= data_shape[axis]

        return BitSet.from_array(vars.tolist())

    def updated_hypercube(hypercube, axis, s = None, e = None):
        hypercube = [list(hypercube[0]), list(hypercube[1])]
        if s is not None:
            hypercube[0][axis] = s
        if e is not None:
            hypercube[1][axis] = e

        hypercube = (tuple(hypercube[0]), tuple(hypercube[1]))
        return hypercube

    def create_input_ns(hypercube):
        scope = hypercube2scope(hypercube)
        if input_layer_fn is not None:
            return input_layer_fn(scope, num_latents, block_size)
        else:
            input_nodes = []
            for var in scope:
                ns = inputs(var, num_node_blocks = num_node_blocks, dist = input_layer_type(**input_layer_params))
                input_nodes.append(ns)

            edge_ids = torch.arange(0, num_node_blocks)[None,:].repeat(2, 1)
            return summate(multiply(*input_nodes), num_node_blocks = num_node_blocks, edge_ids = edge_ids)

    def recursive_construct(hypercube, depth = 1):
        if hypercube in hypercube2ns:
            return hypercube2ns[hypercube]

        if max_split_depth is not None and depth > max_split_depth:
            ns = create_input_ns(hypercube)
            hypercube2ns[hypercube] = ns
            return ns

        # Try to split over every axis
        pns = []
        for axis in range(num_axes):
            s, e = hypercube[0][axis], hypercube[1][axis]
            split_points = [sp for sp in global_split_points[axis] if s < sp < e]

            for sp in split_points:
                ns1 = recursive_construct(updated_hypercube(hypercube, axis, e = sp))
                ns2 = recursive_construct(updated_hypercube(hypercube, axis, s = sp))

                if structure_type == "sum_dominated":
                    pn = multiply(ns1, ns2)
                elif structure_type == "prod_dominated":
                    pn = multiply(ns1.duplicate(), ns2.duplicate())

                pns.append(pn)

        if len(pns) == 0:
            # No split point found. Create input nodes instead
            ns = create_input_ns(hypercube)
        elif hypercube == root_hypercube:
            ns = summate(*pns, num_node_blocks = 1, block_size = 1)
        elif not use_linear_mixing:
            if len(pns) <= max_prod_block_conns:
                ns = summate(*pns, num_node_blocks = num_node_blocks)
            else:
                block_ids = torch.topk(torch.rand([num_node_blocks, len(pns)]), k = max_prod_block_conns, dim = 1).indices
                par_ids = torch.arange(0, num_node_blocks)[:,None,None].repeat(1, max_prod_block_conns, num_node_blocks)
                chs_ids = block_ids[:,:,None] * num_node_blocks + torch.arange(0, num_node_blocks)[None,None,:]
                edge_ids = torch.stack((par_ids.reshape(-1), chs_ids.reshape(-1)), dim = 0)
                ns = summate(*pns, num_node_blocks = num_node_blocks, edge_ids = edge_ids)
        else:
            # Linear mixing as implemented in EiNet's Mixing layer
            if len(pns) <= max_prod_block_conns:
                ns = summate(*pns, num_node_blocks = num_node_blocks)
            else:
                ch_ns = [multiply(summate(pn, num_node_blocks = num_node_blocks)) for pn in pns]
                ns = summate(*ch_ns, num_node_blocks = num_node_blocks, edge_ids = torch.arange(0, num_node_blocks)[None,:].repeat(2, 1))

        hypercube2ns[hypercube] = ns
        return ns

    with set_block_size(block_size = block_size):
        root_hypercube = ((0,) * num_axes, deepcopy(data_shape))
        root_ns = recursive_construct(root_hypercube)

    return root_ns


def PDHCLT(data: torch.Tensor, data_shape: Tuple, num_latents: int, 
           split_intervals: Optional[Union[int, Tuple[int]]] = None, 
           split_points: Optional[Sequence[Sequence[int]]] = None,
           max_split_depth: Optional[int] = None,
           max_prod_block_conns: int = 4,
           structure_type: str = "sum_dominated",
           input_layer_type: Type[Distribution] = Categorical, 
           input_layer_params: Dict = {"num_cats": 256},
           hclt_kwargs: Dict = {"num_bins": 32, "sigma": 0.5 / 32, "chunk_size": 32},
           block_size: Optional[int] = None):

    assert data.dim() == 2
    assert data.size(1) == reduce(lambda x, y: x * y, data_shape)

    def input_layer_fn(scope, num_latents, block_size):
        vars = torch.tensor(scope.to_list()).sort().values
        ns = HCLT(
            x = data[:,vars], 
            num_latents = num_latents, 
            input_layer_type = input_layer_type,
            input_layer_params = input_layer_params,
            num_root_ns = num_latents,
            block_size = block_size,
            **hclt_kwargs
        )

        # Map input variables
        var_mapping = {v: vars[v].item() for v in range(vars.size(0))}
        ns = jtf.deepcopy(ns, var_mapping = var_mapping)

        return ns

    ns = PD(data_shape = data_shape, num_latents = num_latents, 
            split_intervals = split_intervals, split_points = split_points,
            max_split_depth = max_split_depth, max_prod_block_conns = max_prod_block_conns,
            structure_type = structure_type, input_layer_fn = input_layer_fn,
            input_layer_type = input_layer_type, input_layer_params = input_layer_params,
            block_size = block_size)

    if ns.num_node_blocks > 1:
        ns = summate(*ns.chs, num_node_blocks = 1, block_size = 1)

    return ns
