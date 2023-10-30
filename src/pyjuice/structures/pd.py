from __future__ import annotations

import torch
import random
from copy import deepcopy
from functools import reduce

import pyjuice.transformations as jtf
from typing import Tuple, Sequence, Optional, Type, Dict
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.nodes.distributions import *
from pyjuice.structures.hclt import HCLT
from pyjuice.utils import BitSet


def PD(data_shape: Tuple, num_latents: int, 
       split_intervals: Optional[Union[int, Tuple[int]]] = None, 
       split_points: Optional[Sequence[Sequence[int]]] = None,
       max_split_depth: Optional[int] = None,
       max_prod_group_conns: int = 4,
       structure_type: str = "sum_dominated",
       input_layer_fn: Optional[Callable] = None,
       input_layer_type: Type[Distribution] = Categorical, 
       input_layer_params: Dict = {"num_cats": 256}):
    """
    The PD structure was proposed in
        Sum-Product Networks: A New Deep Architecture
        Hoifung Poon, Pedro Domingos
        UAI 2011
    and generates a PC structure for random variables which can be naturally arranged on discrete grids, like images.
    """
    assert structure_type in ["sum_dominated", "prod_dominated"]

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
            return input_layer_fn(scope, num_latents)
        else:
            input_nodes = []
            for var in scope:
                ns = inputs(var, num_nodes = num_latents, dist = input_layer_type(**input_layer_params))
                input_nodes.append(ns)

            edge_ids = torch.arange(0, num_latents)[None,:].repeat(2, 1)
            return summate(multiply(*input_nodes), num_nodes = num_latents, edge_ids = edge_ids)

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
            ns = summate(*pns, num_nodes = 1)
        elif len(pns) <= max_prod_group_conns:
            ns = summate(*pns, num_nodes = num_latents)
        else:
            group_ids = torch.topk(torch.rand([num_latents, len(pns)]), k = max_prod_group_conns, dim = 1).indices
            par_ids = torch.arange(0, num_latents)[:,None,None].repeat(1, max_prod_group_conns, num_latents)
            chs_ids = group_ids[:,:,None] * num_latents + torch.arange(0, num_latents)[None,None,:]
            edge_ids = torch.stack((par_ids.reshape(-1), chs_ids.reshape(-1)), dim = 0)
            ns = summate(*pns, num_nodes = num_latents, edge_ids = edge_ids)

        hypercube2ns[hypercube] = ns
        return ns

    root_hypercube = ((0,) * num_axes, deepcopy(data_shape))
    root_ns = recursive_construct(root_hypercube)

    return root_ns


def PDHCLT(data: torch.Tensor, data_shape: Tuple, num_latents: int, 
           split_intervals: Optional[Union[int, Tuple[int]]] = None, 
           split_points: Optional[Sequence[Sequence[int]]] = None,
           max_split_depth: Optional[int] = None,
           max_prod_group_conns: int = 4,
           structure_type: str = "sum_dominated",
           input_layer_type: Type[Distribution] = Categorical, 
           input_layer_params: Dict = {"num_cats": 256},
           hclt_kwargs: Dict = {"num_bins": 32, "sigma": 0.5 / 32, "chunk_size": 32}):

    assert data.dim() == 2
    assert data.size(1) == reduce(lambda x, y: x * y, data_shape)

    def input_layer_fn(scope, num_latents):
        vars = torch.tensor(scope.to_list()).sort().values
        ns = HCLT(
            x = data[:,vars], 
            num_latents = num_latents, 
            input_layer_type = input_layer_type,
            input_layer_params = input_layer_params,
            num_root_ns = num_latents,
            **hclt_kwargs
        )

        # Map input variables
        var_mapping = {v: vars[v].item() for v in range(vars.size(0))}
        ns = jtf.deepcopy(ns, var_mapping = var_mapping)

        return ns

    ns = PD(data_shape = data_shape, num_latents = num_latents, 
            split_intervals = split_intervals, split_points = split_points,
            max_split_depth = max_split_depth, max_prod_group_conns = max_prod_group_conns,
            structure_type = structure_type, input_layer_fn = input_layer_fn,
            input_layer_type = input_layer_type, input_layer_params = input_layer_params)

    if ns.num_nodes > 1:
        ns = summate(*ns.chs, num_nodes = 1)

    return ns
