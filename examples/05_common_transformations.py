"""
PC Structural Transformation Functions
======================================

In this tutorial, you will learn how to use the built-in structural transformation algorithms to easily create and manipulate PCs.
"""

# sphinx_gallery_thumbnail_path = 'imgs/juice.png'

# %% 
# Let's start by importing the necessary packages.

import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists

# %%
# Clone a PC
# ----------

# %%
# :code:`pyjuice.deepcopy` allows us to clone a PC, with some options to manipulate the copy. 
# Let's start by defining a PC.

with juice.set_block_size(block_size = 4):

    i00 = juice.inputs(0, num_node_blocks = 8, dist = dists.Categorical(num_cats = 5))
    i10 = juice.inputs(1, num_node_blocks = 8, dist = dists.Categorical(num_cats = 5))
    i11 = juice.inputs(1, num_node_blocks = 8, dist = dists.Categorical(num_cats = 5))
    
    ms0 = juice.multiply(i00, i10)
    ms1 = juice.multiply(i00, i11)

    ns = juice.summate(ms0, ms1, num_node_blocks = 1)

ns.init_parameters()

# %%
# To create an independent copy of :code:`ns`, we can run:

new_ns1 = juice.deepcopy(ns)

# %%
# By setting :code:`tie_params` to :code:`True`, we can tie the parameters of the original PC with that of the copied PC. Note that tied parameters will remain the same during all transformation/learning procedures implemented in PyJuice.

new_ns2 = juice.deepcopy(ns, tie_params = True)

# %%
# By providing a :code:`var_mapping`, we can define the copied PC on another set of variables.

var_mapping = {0: 2, 1: 3}
new_ns3 = juice.deepcopy(ns, var_mapping = var_mapping)

# %%
# Note that :code:`tie_params` and :code:`var_mapping` can be used simultaneously:

new_ns3 = juice.deepcopy(ns, tie_params = True, var_mapping = var_mapping)

# %%
# Merge PCs
# ---------

# %%
# :code:`juice.merge` can be used to collapse vectors of nodes defined on the same variable scope into a single vector of nodes.
# Take the following PC as an example:

with juice.set_block_size(block_size = 4):

    i00 = juice.inputs(0, num_node_blocks = 8, dist = dists.Categorical(num_cats = 5))
    i01 = juice.inputs(0, num_node_blocks = 8, dist = dists.Categorical(num_cats = 5))
    i10 = juice.inputs(1, num_node_blocks = 8, dist = dists.Categorical(num_cats = 5))
    i11 = juice.inputs(1, num_node_blocks = 8, dist = dists.Categorical(num_cats = 5))
    
    ms00 = juice.multiply(i00, i10)
    ms01 = juice.multiply(i01, i11)

    ns = juice.summate(ms00, ms01, num_node_blocks = 1, block_size = 1)

# %%
# In the above PC (i.e., :code:`ns`), :code:`i00` and :code:`i01` (also :code:`i10` and :code:`i11`) can be merged into a single object since they define on the same variable and has the same distribution.
# :code:`juice.merge` outputs an equivalent PC with the objects properly merged:

new_ns1 = juice.merge(ns)

# %%
# Another usage of the merge function is to "concatenate" nodes defined by multiple PyJuice objects, if they are defined on the same set of variables.

ns0 = juice.summate(ms00, num_node_blocks = 8, block_size = 4)
ns1 = juice.summate(ms01, num_node_blocks = 8, block_size = 4)

new_ns2 = juice.merge(ns0, ns1)

# %%
# :code:`new_ns2` will also be equivalent to :code:`ns`.

# %%
# Adjust block sizes
# ------------------

# %%
# In the second tutorial (i.e., "Construct Simple PCs"), we mentioned that defining PCs with large :code:`block_size` is crucial to their efficiency.
# While the best practice is to set high block sizes manually whenever possible, we provide :code:`pyjuice.blockify` to try bumping the group size of all vectors of nodes within a PC.

with juice.set_block_size(block_size = 2):

    ni0 = juice.inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
    ni1 = juice.inputs(1, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
    ni2 = juice.inputs(2, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))
    ni3 = juice.inputs(3, num_node_blocks = 2, dist = dists.Categorical(num_cats = 2))

    ms1 = juice.multiply(ni0, ni1, edge_ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = torch.long))
    ns1 = juice.summate(ms1, edge_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 0, 1, 2, 3]], dtype = torch.long))

    ms2 = juice.multiply(ni2, ni3, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    ns2 = juice.summate(ms2, edge_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype = torch.long))

    ms = juice.multiply(ns1, ns2, edge_ids = torch.tensor([[0, 0], [1, 1]], dtype = torch.long))
    ns = juice.summate(ms, edge_ids = torch.tensor([[0, 0], [0, 1]], dtype = torch.long), block_size = 1)

ns.init_parameters()

# %%
# While the block sizes of the above-defined PC is 2 (except for the root node), :code:`ns1` could have block size 4 since every pair of 4 aligned sum nodes in :code:`ns1` and 4 aligned child nodes are fully connected.
# To apply such change, we can use:

new_ns = juice.blockify(ns, sparsity_tolerance = 0.25, max_target_block_size = 32)

# %%
# There are two parameters to the function. :code:`max_target_block_size` specifies the maximum block size to be considered; :code:`sparsity_tolerance` specifies what fraction of pseudo-edges do we allow to add in order to increase the block size even if some pairs of parent and child node blocks are neither fully-connected nor unconnected.

