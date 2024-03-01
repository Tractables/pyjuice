"""
Construct Simple PCs
====================

In this tutorial, you will learn about the basic APIs to construct PCs.
"""

# sphinx_gallery_thumbnail_path = 'imgs/juice.png'

import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists

# %%
# A PC can be created using :code:`pyjuice.inputs`, :code:`pyjuice.multiply`, and :code:`pyjuice.summate`, which create input node vectors, product node vectors, and sum node vectors, respectively.
# The goal of this tutorial is to get you familiar with these functions.

# %%
# Input nodes
# -----------

# %%
# Let us start with :code:`pyjuice.inputs`.

ni0 = juice.inputs(var = 0, num_node_blocks = 4, block_size = 2, dist = dists.Categorical(num_cats = 4))

# %%
# The above line defines :code:`num_node_blocks * block_size = 8` input nodes on variable ID :code:`var = 0` featuring Categorical distributions with 4 categories (i.e., :code:`dists.Categorical(num_cats = 4)`). The set of input distributions are defined under :code:`pyjuice.nodes.distributions`.
#
# A seemingly redundant pair of parameters are :code:`num_node_blocks` and :code:`block_size`, since we can alternatively only specify the number of nodes in the node vector :code:`ni0`.
# In fact, ensuring large :code:`block_size` is crucial to the efficiency of the PC. So always try to use large block sizes when defining PCs.
# Although using larger block sizes when defining input node vectors do not have any negative effects, it poses restrictions on the edge connection pattern as we shall proceed to show.
# 
# Note that :code:`block_size` has to be a power of 2.

# %%
# Product nodes
# -------------

# %%
# Let us define another input node vector and a product node vector that take the input nodes as children.

ni1 = juice.inputs(var = 1, num_node_blocks = 4, block_size = 2, dist = dists.Categorical(num_cats = 4))

edge_ids = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
ms = juice.multiply(ni0, ni1, edge_ids = edge_ids)

# %%
# :code:`ms` defines a product node vector where every node have two children: the first child is a node in :code:`ni0` and the second child is a node in :code:`ni1`.
# :code:`edge_ids` specifies which child nodes do every node in :code:`ms` connects to. Specifically, :code:`edge_ids` has size :code:`[# product node blocks, # child node vectors]`, so in this case, it should has size :code:`[4, 2]`.
# The semantic of :code:`edge_ids[i,j]` is: the :math:`i`th product node block connects to the :code:`edge_ids[i,j]`th node block in the :math:`j`th child node vector (assume we always count from 0).
# For example, :code:`edge_ids[1,0] = 1` means that the 1th product node block connects to the 1th node block in :code:`ni0`.
# 
# We require the node vectors fed to :code:`pyjuice.multiply` have the same :code:`block_size`. And the block size of the output product node vector is also the same with that of the inputs.
# We do not need to specify the number of node blocks (e.g., using :code:`num_node_blocks`) since it is equal to :code:`edge_ids.size(0)`.
# For :code:`pyjuice.multiply`, if two node blocks are connected (as defined by :code:`edge_ids`), we assume the :math:`i`th node in the (parent) product node block is connected to the :math:`i`th node in the child node block.
# 
# When we do not provide the :code:`edge_ids`, PyJuice assumes it to be the following (we can only use this shortcut when the child node vectors have the same :code:`num_node_blocks` and :code:`block_size`):

num_node_blocks = 4
num_child_node_vectors = 2
edge_ids = torch.arange(0, num_node_blocks)[:,None].repeat(1, num_child_node_vectors)

# %%
# Sum nodes
# ---------

# %%
# Finally, we introduce :code:`pyjuice.summate`, which is used to define sum node vectors.

edge_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 5], [0, 1, 0, 2, 1, 2, 2, 3, 2, 0]])
ns = juice.summate(ms, num_node_blocks = 6, block_size = 2, edge_ids = edge_ids)

# %%
# Similar to :code:`pyjuice.multiply`, the positional arguments to :code:`pyjuice.summate` are the list of input node vectors (here we only provide :code:`ms`).
# :code:`num_node_blocks` and :code:`block_size` are the number of node blocks and the size of each node block, respectively.
# Therefore, :code:`ns` defines a vector of :code:`num_node_blocks * block_size = 12` sum nodes that are (partially) connected to the nodes in :code:`ms`.
# If there are multiple child node vectors, :code:`pyjuice.summate` assumes they have the same block size. However, the sum node vector can have a different block size compared to its children.
# 
# The connection pattern is specified by the keyword argument :code:`edge_ids`, which have shape :code:`[2, # edge blocks]`. 
# Every size-2 column vector :math:`[m, n]^T` in :code:`edge_ids` indicates the existance of fully-connected edges between the :math:`m`th sum node block and the :math:`n`th product node block.
# That is, in the case where both :code:`ns` and :code:`ms` have block size 2, every column in :code:`edge_ids` specifies :math:`2 \times 2 = 4` edges.
# 
# If :code:`edge_ids` is not provided, we assume that all nodes in the sum node vector are connected to all child nodes:

ns = juice.summate(ms, num_node_blocks = 6, block_size = 2)