"""
Input nodes under the lifted plan. An observed token keeps pyjuice's own log-probability (the input layers
write it, as in a :class:`TensorCircuit`); a missing one is summed per token class. CATEGORICAL ONLY, like
the compiler: a class's mass is the sum of the probabilities of its tokens.
"""

import torch
import triton
import triton.language as tl

from pyjuice.nodes.distributions import Categorical


@triton.jit
def _class_mass_kernel(probs, starts, token_class, out, V, C, C_PAD: tl.constexpr, TILE_V: tl.constexpr):
    u = tl.program_id(0)
    start = tl.load(starts + u).to(tl.int64)
    classes = tl.arange(0, C_PAD)
    acc = tl.zeros([C_PAD], dtype = tl.float32)
    for v0 in range(0, V, TILE_V):
        offs = v0 + tl.arange(0, TILE_V)
        vmask = offs < V
        p = tl.load(probs + start + offs, mask = vmask, other = 0.0)
        cls = tl.load(token_class + offs, mask = vmask, other = -1)
        acc += tl.sum(tl.where(classes[:, None] == cls[None, :], p[None, :], 0.0), axis = 1)
    tl.store(out + u * C + classes, tl.log(acc), mask = classes < C)


def input_class_tables(pc, num_cats: int):
    """
    Per input layer: the distinct parameter starts of its nodes (tied nodes share them) and, for every node,
    which of them it reads, so that class masses are computed once per distinct node.

    :returns: a list of ``(layer, starts [U], inverse [num_nodes])`` on the PC's device
    """
    tables = []
    for layer in pc.input_layer_group:
        for ns in layer.nodes:
            if not isinstance(ns.dist, Categorical) or ns.dist.num_cats != num_cats:
                raise NotImplementedError(f"The lifted backend supports Categorical input nodes over {num_cats} "
                                          f"tokens only, got {ns.dist} over variables {ns.scope.to_list()}.")
        starts, inverse = torch.unique(layer.s_pids, return_inverse = True)
        tables.append((layer, starts, inverse))
    return tables


def class_masses(tables, token_class: torch.Tensor, num_classes: int, class_mars: torch.Tensor, input_start: int):
    """Write the log-mass of every token class of every input node into ``class_mars`` [num_input_rows, C]."""
    V = token_class.numel()
    for layer, starts, inverse in tables:
        masses = torch.empty(starts.numel(), num_classes, device = class_mars.device)
        _class_mass_kernel[(starts.numel(),)](layer.params, starts, token_class, masses, V, num_classes,
                                              C_PAD = max(16, triton.next_power_of_2(num_classes)),
                                              TILE_V = 1024, num_warps = 4)
        first, end = layer._output_ind_range
        class_mars[first - input_start:end - input_start] = masses[inverse]
