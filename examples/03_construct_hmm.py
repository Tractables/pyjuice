"""
Construct an HMM
================

This tutorial demonstrates how to construct an HMM with :code:`pyjuice.inputs`, :code:`pyjuice.multiply`, and :code:`pyjuice.summate`.
"""

# sphinx_gallery_thumbnail_path = 'imgs/juice.png'

import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists

# %%
# We start with specifying the structural parameters of the HMM

seq_length = 32
num_latents = 2048
num_emits = 4023

# %%
# An important parameter to be determined is the block size, which is crucial for PyJuice to compile efficient models.
# Specifically, we want the block size to be large enough so that PyJuice can leverage block-based parallelization.

block_size = min(juice.utils.util.max_cdf_power_of_2(num_latents), 1024)

# %%
# The number of node blocks is derived accordingly

num_node_blocks = num_latents // block_size

# %%
# We use the context manager `set_block_size` to set the block size of all PC nodes.
# In the following we assume `T = seq_length` and `K = num_latents`

with juice.set_block_size(block_size):
    # We begin by defining p(X_{T-1}|Z_{T-1}) for all k = 0...K-1
    ns_input = juice.inputs(seq_length - 1, num_node_blocks = num_node_blocks,
                            dist = dists.Categorical(num_cats = num_emits))
    
    ns_sum = None
    curr_zs = ns_input
    for var in range(seq_length - 2, -1, -1):
        # The emission probabilities p(X_{var}|Z_{var}=k) for all k = 0...K-1
        curr_xs = ns_input.duplicate(var, tie_params = True)
        
        # The transition probabilities p(Z_{var+1}|Z_{var})
        if ns_sum is None:
            # Create both the structure and the transition probabilities
            ns = juice.summate(curr_zs, num_node_blocks = num_node_blocks)
            ns_sum = ns
        else:
            # Create only the structure and reuse the transition probabilities from `ns_sum`
            ns = ns_sum.duplicate(curr_zs, tie_params=True)

        curr_zs = juice.multiply(curr_xs, ns)
        
    # The Initial probabilities p(Z_{0})
    ns = juice.summate(curr_zs, num_node_blocks = 1, block_size = 1)

# %%
# Note that :code:`ns.duplication` is a handy function to create duplications of existing node vectors. 
# For input nodes (e.g., :code:`ns_input.duplicate(var, tie_params = True)`) we can specify it to define on a new variable. 
# The argument :code:`tie_params = True` means we want to use the same set of parameters in the original and the duplicated node vector.
# The parameters will remain tied after parameter learning, structural transformation, etc.
#
# For sum node vectors, :code:`ns.duplicate` allows us to specify a new list of children. However, the child nodes must have the same size (same number of node blocks and block size).
