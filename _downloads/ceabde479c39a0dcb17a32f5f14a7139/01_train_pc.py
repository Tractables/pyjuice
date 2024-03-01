"""
Train a PC
==========

This tutorial demonstrates how to create a Hidden Chow-Liu Tree (https://arxiv.org/pdf/2106.02264.pdf) using :code:`pyjuice.structures` and train the model with mini-batch EM and full-batch EM.
For simplicity, we use the MNIST dataset as an example. 

Note that the goal of this tutorial is just to quickly demonstrate the basic training pipeline using PyJuice without covering additional details such as ways to construct a PC, which will be covered in the following tutorials.
"""

# sphinx_gallery_thumbnail_path = 'imgs/juice.png'

# %%
# Load the MNIST Dataset
# ----------------------

import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists

train_dataset = torchvision.datasets.MNIST(root = "../data", train = True, download = True)
valid_dataset = torchvision.datasets.MNIST(root = "../data", train = False, download = True)

train_data = train_dataset.data.reshape(60000, 28*28)
valid_data = valid_dataset.data.reshape(10000, 28*28)

train_loader = DataLoader(
    dataset = TensorDataset(train_data),
    batch_size = 512,
    shuffle = True,
    drop_last = True
)
valid_loader = DataLoader(
    dataset = TensorDataset(valid_data),
    batch_size = 512,
    shuffle = False,
    drop_last = True
)

# %%
# Create the PC
# -------------

# %%
# Let's create a HCLT PC with latent size 128.

device = torch.device("cuda:0")

# The data is required to construct the backbone Chow-Liu Tree structure for the HCLT
ns = juice.structures.HCLT(
    train_data.float().to(device), 
    num_latents = 128
)

# %%
# :code:`ns` is a Directed Acyclic Graph (DAG) representation of the PC. 
# Specifically, we use :code:`pyjuice.nodes.InputNodes`, :code:`pyjuice.nodes.ProdNodes`, and :code:`pyjuice.nodes.SumNodes` to define vectors of input nodes, product nodes, and sum nodes, respectively.
# By also storing the topological structure of the node vectors (with pointers to the child node vectors), we create the PC as a DAG-based structure. :code:`ns` is also just a node vector defining the root node of the PC.
# 
# While being user-friendly, the DAG-based representation is not amenable to efficient computation. 
# Therefore, before doing any computation, we need to compile the PC with :code:`pyjuice.compile`, which creates a compact and equivalent representation of the PC.

pc = juice.compile(ns)

# %%
# The :code:`pc` is an instance of :code:`torch.nn.Module`. So we can safely assume it is just a neural network with the variable assignments :math:`\mathbf{x}` as input and its log-likelihood :math:`\log p(\mathbf{x})` as output. 
# We proceed to move it to the GPU specified by :code:`device`.

pc.to(device)

# %%
# Train the PC
# ------------

# %%
# We start by defining the optimizer and scheduler.

optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1, method = "EM")
scheduler = juice.optim.CircuitScheduler(
    optimizer, 
    method = "multi_linear", 
    lrs = [0.9, 0.1, 0.05], 
    milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]
)

# %%
# Optionally, we can leverage CUDA Graphs to hide the kernel launching overhead by doing a dry run.

for batch in train_loader:
    x = batch[0].to(device)

    lls = pc(x, record_cudagraph = True)
    lls.mean().backward()
    break

# %%
# We are now ready for the training. Below is an example training loop for mini-batch EM.

for epoch in range(1, 350+1):
    t0 = time.time()
    train_ll = 0.0
    for batch in train_loader:
        x = batch[0].to(device)

        # Similar to PyTorch optimizers zeroling out the gradients, we zero out the parameter flows
        optimizer.zero_grad()

        # Forward pass
        lls = pc(x)

        # Backward pass
        lls.mean().backward()

        train_ll += lls.mean().detach().cpu().numpy().item()

        # Perform a mini-batch EM step
        optimizer.step()
        scheduler.step()

    train_ll /= len(train_loader)

    t1 = time.time()
    test_ll = 0.0
    for batch in valid_loader:
        x = batch[0].to(pc.device)
        lls = pc(x)
        test_ll += lls.mean().detach().cpu().numpy().item()
    
    test_ll /= len(valid_loader)
    t2 = time.time()

    print(f"[Epoch {epoch}/{350}][train LL: {train_ll:.2f}; val LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; val forward {t2-t1:.2f}] ")

# %%
# Similarly, an example training loop for full-batch EM is given as follows.

for epoch in range(1, 1+1):
    t0 = time.time()

    # Manually zeroling out the flows
    pc.init_param_flows(flows_memory = 0.0)

    train_ll = 0.0
    for batch in train_loader:
        x = batch[0].to(device)

        # We only run the forward and the backward pass, and accumulate the flows throughout the epoch
        lls = pc(x)
        lls.mean().backward()

        train_ll += lls.mean().detach().cpu().numpy().item()

    # Set step size to 1.0 for full-batch EM
    pc.mini_batch_em(step_size = 1.0, pseudocount = 0.01)

    train_ll /= len(train_loader)

    t1 = time.time()
    test_ll = 0.0
    for batch in valid_loader:
        x = batch[0].to(pc.device)
        lls = pc(x)
        test_ll += lls.mean().detach().cpu().numpy().item()
    
    test_ll /= len(valid_loader)
    t2 = time.time()
    print(f"[Epoch {epoch}/{1}][train LL: {train_ll:.2f}; val LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; val forward {t2-t1:.2f}] ")
