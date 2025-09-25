import torch
import pyjuice as juice
from ..nodes.distributions.masked_categorical import MaskedCategorical


def create_leaves(v, num_cats = 2):
    ns = juice.inputs(
        v, 
        num_nodes = num_cats, 
        dist = MaskedCategorical(
            num_cats = num_cats, mask_mode = "full_mask"), 
            mask = torch.eye(num_cats, dtype = torch.long
        )
    )
    return ns


def markov(seq_len, num_cats = 2):

    ns = juice.multiply(create_leaves(0))

    for v in range(1, seq_len):
        ns = juice.summate(ns, num_nodes = num_cats)
        ni = create_leaves(v)

        ns = juice.multiply(ns, ni)

    ns = juice.summate(ns, num_nodes = 1)

    return ns