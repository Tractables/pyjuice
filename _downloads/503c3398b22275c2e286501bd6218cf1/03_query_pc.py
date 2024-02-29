"""
Query a PC
==========

Assume we have constructed and trained a PC following the previous tutorials. This tutorial demonstrates how to query the PC, i.e., ask probabilistic queries about the distribution encoded by the PC.

We will cover how to compute marginal and conditional probabilities.
"""

# sphinx_gallery_thumbnail_path = 'imgs/juice.png'

# %%
# Generate a PC
# -------------

# %%
# We create a simple PC consisting of two variables :math:`X_1` and :math:`X_2`:

import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists

ni0 = juice.inputs(0, num_nodes = 2, dist = dists.Categorical(num_cats = 2))
ni1 = juice.inputs(1, num_nodes = 2, dist = dists.Categorical(num_cats = 4))

ms = juice.multiply(ni0, ni1)
ns = juice.summate(ms, num_nodes = 1)

ns.init_parameters()

pc = juice.compile(ns)

# %%
# Move the PC to a GPU:

device = torch.device("cuda:0")
pc.to(device)

# %%
# Compute marginal probabilities
# ------------------------------

# %%
# Assume we want to compute the probabilities :math:`P(X_1 = 0)` and :math:`P(X_1 = 1)`. We need to create two tensors: a "data" tensor consisting the values of the observed variables (:math:`X_1` in this case) and another "mask" tensor indicating which variables are missing.

data = torch.tensor([[0, 0], [1, 0]]).to(device)
missing_mask = torch.tensor([[False, True], [False, True]]).to(device) # True for variables to be conditioned on/are missing

# %%
# In the data tensor, entries corresponding missing variables will be dismissed by PyJuice and will not influence the output.
# The `missing_mask` can have have shape [batch_size, num_vars] or [num_vars] if for all samples we marginalize out the same subset of variables.
# 
# We proceed to compute the marginal probabilities using `pyjuice.queries.marginal`:

lls = juice.queries.marginal(
    pc, data = data, missing_mask = missing_mask
)

# %%
# For PCs defined on categorical variables, we can alternatively query for marginal probabilities given *soft* evidence, e.g., :math:`P(X_1 = 0 \text{~w.p.~} 0.3 \text{~and~} 1 \text{~w.p.~} 0.7)`.
# This can be done by defining `date` as a 3D tensor of size [batch_size, num_vars, num_cats]:

data = torch.tensor([[[0.4, 0.6, 0, 0], [0, 0, 0, 0]], [[0.3, 0.7, 0, 0], [0, 0, 0, 0]]]).to(device)

# %%
# Since :math:`X_1` has two categories and $X_2$ has four categories, the size of the last dimension of `data` should be 4.
#
# The soft marginal probabilities can be similarly computed by:

lls = juice.queries.marginal(
    pc, data = data, missing_mask = missing_mask
)

# %%
# Compute conditional probabilities
# ---------------------------------

# %%
# Since every conditional probability can be represented as the quotient of two marginal probabilities, one may wonder why do we need a separate function for computing conditional probabilities.
# In fact, with `pyjuice.queries.conditional`, we can simultaneously compute a *set of* conditional probabilities. Specifically, given evidence $\mathbf{E} = \mathbf{e}$, we can compute $\forall X \not\in \mathbf{E}, x \in \mathrm{val}(X), P(X = x | \mathbf{e})$.

# %%
# Say we want to compute the conditional probability of $X_2$ given evidence $X_1 = 0$ and $X_1 = 1$, respectively. We prepare the data and the mask similarly.

data = torch.tensor([[0, 0], [1, 0]]).to(device)
missing_mask = torch.tensor([[False, True], [False, True]]).to(device) # True for variables to be conditioned on/are missing

# %%
# The conditional probabilities are computed as follows:

outputs = juice.queries.conditional(
    pc, data = data, missing_mask = missing_mask, target_vars = [1]
)

# %%
# The parameter `target_vars` is used to indicate the subset of variables which we want to compute their conditional probabilities. Probabilities of all variables will be returned if we do not specify `target_vars`.
# 
# The shape of `outputs` is [B, num_target_vars, num_categories]. For example, `outputs[1,0,3]` is the conditional probability $P(X_2 = 3 | X_1 = 1)$.

# %%
# Similar to the marginal query, for categorical data, we can also feed *soft* evidence:

data = torch.tensor([[[0.4, 0.6, 0, 0], [0, 0, 0, 0]], [[0.3, 0.7, 0, 0], [0, 0, 0, 0]]]).to(device)
missing_mask = torch.tensor([[False, True], [False, True]]).to(device)

outputs = juice.queries.conditional(
    pc, data = data, missing_mask = missing_mask, target_vars = [1]
)