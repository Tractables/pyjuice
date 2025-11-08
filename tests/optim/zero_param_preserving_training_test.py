import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

import pytest


def load_penn_treebank(seq_length = 32):

    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?,;:-'\"()[]{}"
    vocab = {char: idx for idx, char in enumerate(CHARS)}

    # Load the Penn Treebank dataset
    try:
        dataset = load_dataset('ptb_text_only')
    except ConnectionError:
        return None # Skip the test if the dataset fails to load
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    test_dataset = dataset['test']

    train_data = []
    for sample in tqdm(train_dataset["sentence"]):
        train_data.extend([vocab[token] if token in vocab else len(CHARS) for token in sample])

    valid_data = []
    for sample in tqdm(valid_dataset["sentence"]):
        valid_data.extend([vocab[token] if token in vocab else len(CHARS) for token in sample])

    test_data = []
    for sample in tqdm(test_dataset["sentence"]):
        test_data.extend([vocab[token] if token in vocab else len(CHARS) for token in sample])

    # Convert to PyTorch tensors
    train_data = torch.tensor(train_data)
    valid_data = torch.tensor(valid_data)
    test_data = torch.tensor(test_data)

    nsamples = train_data.size(0) // seq_length * seq_length
    train_data = train_data[:nsamples].reshape(-1, seq_length)

    nsamples = valid_data.size(0) // seq_length * seq_length
    valid_data = valid_data[:nsamples].reshape(-1, seq_length)

    nsamples = test_data.size(0) // seq_length * seq_length
    test_data = test_data[:nsamples].reshape(-1, seq_length)

    return train_data, valid_data, test_data


def evaluate(pc, loader):
    lls_total = 0.0
    for batch in loader:
        x = batch[0].to(pc.device)

        lls = pc(x)

        lls_total += lls.mean().detach().cpu().numpy().item()
    
    lls_total /= len(loader)
    return lls_total


def full_batch_em_epoch(pc, train_loader, test_loader, device):
    t0 = time.time()
    train_ll = 0.0
    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x)
        pc.backward(x, allow_modify_flows = False, logspace_flows = True)

        train_ll += lls.mean().detach().cpu().numpy().item()

    pc.mini_batch_em(step_size = 1.0, pseudocount = 0.001, keep_zero_params = True)

    train_ll /= len(train_loader)

    t1 = time.time()
    test_ll = evaluate(pc, loader = test_loader)
    t2 = time.time()
    print(f"[train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def test_hclt_zero_preserving():

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28)
    test_data = test_dataset.data.reshape(10000, 28*28)

    num_features = train_data.size(1)

    train_loader = DataLoader(
        dataset = TensorDataset(train_data),
        batch_size = 512,
        shuffle = True,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = TensorDataset(test_data),
        batch_size = 512,
        shuffle = False,
        drop_last = True
    )

    num_latents = 128

    root_ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = num_latents, 
        chunk_size = 32
    )
    root_ns.init_parameters(perturbation = 8.0)

    ns2mask = dict()

    for ns in root_ns:
        if ns.is_sum() and ns.has_params():
            num_node_blocks = ns.num_node_blocks
            num_ch_blocks = ns.chs[0].num_node_blocks
            block_size = ns.block_size
            ch_block_size = ns.chs[0].block_size

            num_sum_nodes = num_node_blocks * block_size
            num_ch_nodes = num_ch_blocks * ch_block_size

            params = ns.get_params().reshape(
                num_node_blocks, num_ch_blocks, block_size, ch_block_size).permute(0, 2, 1, 3).reshape(
                    num_sum_nodes, num_ch_nodes
                )

            while True:
                mask = torch.rand(num_sum_nodes, num_ch_nodes) < 0.1
                if torch.all(mask.long().sum(dim = 1) > 0):
                    break

            params[mask] = 0.0
            ns.set_params(params)
            ns2mask[ns] = mask

    pc = juice.compile(root_ns)
    pc.to(device)

    for epoch in range(1, 10 + 1):
        full_batch_em_epoch(pc, train_loader, test_loader, device)

    test_ll = evaluate(pc, test_loader)
    assert test_ll > -780

    for ns in root_ns:
        if ns.is_sum() and ns.has_params():
            num_node_blocks = ns.num_node_blocks
            num_ch_blocks = ns.chs[0].num_node_blocks
            block_size = ns.block_size
            ch_block_size = ns.chs[0].block_size

            num_sum_nodes = num_node_blocks * block_size
            num_ch_nodes = num_ch_blocks * ch_block_size

            params = ns.get_params().reshape(
                num_node_blocks, num_ch_blocks, block_size, ch_block_size).permute(0, 2, 1, 3).reshape(
                    num_sum_nodes, num_ch_nodes
                )

            mask = ns2mask[ns]

            assert torch.all(~mask | (params < 1e-12))


def test_hmm_zero_preserving():

    device = torch.device("cuda:0")

    seq_length = 32

    data = load_penn_treebank(seq_length = seq_length)
    if data is None:
        return None
    train_data, valid_data, test_data = data

    vocab_size = train_data.max().item() + 1

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

    root_ns = juice.structures.HMM(
        seq_length = seq_length,
        num_latents = 512,
        num_emits = vocab_size,
        homogeneous = True
    )
    root_ns.init_parameters(perturbation = 8.0)

    ns2mask = dict()

    for ns in root_ns:
        if ns.is_sum() and ns.has_params() and not ns.is_tied():
            num_node_blocks = ns.num_node_blocks
            num_ch_blocks = ns.chs[0].num_node_blocks
            block_size = ns.block_size
            ch_block_size = ns.chs[0].block_size

            num_sum_nodes = num_node_blocks * block_size
            num_ch_nodes = num_ch_blocks * ch_block_size

            params = ns.get_params().reshape(
                num_node_blocks, num_ch_blocks, block_size, ch_block_size).permute(0, 2, 1, 3).reshape(
                    num_sum_nodes, num_ch_nodes
                )

            while True:
                mask = torch.rand(num_sum_nodes, num_ch_nodes) < 0.1
                if torch.all(mask.long().sum(dim = 1) > 0):
                    break

            params[mask] = 0.0
            ns.set_params(params)
            ns2mask[ns] = mask

    pc = juice.compile(root_ns)
    pc.to(device)

    for epoch in range(1, 10 + 1):
        full_batch_em_epoch(pc, train_loader, valid_loader, device)

    test_ll = evaluate(pc, valid_loader)
    assert test_ll > -90

    for ns in root_ns:
        if ns.is_sum() and ns.has_params() and not ns.is_tied():
            num_node_blocks = ns.num_node_blocks
            num_ch_blocks = ns.chs[0].num_node_blocks
            block_size = ns.block_size
            ch_block_size = ns.chs[0].block_size

            num_sum_nodes = num_node_blocks * block_size
            num_ch_nodes = num_ch_blocks * ch_block_size

            params = ns.get_params().reshape(
                num_node_blocks, num_ch_blocks, block_size, ch_block_size).permute(0, 2, 1, 3).reshape(
                    num_sum_nodes, num_ch_nodes
                )

            mask = ns2mask[ns]

            assert torch.all(~mask | (params < 1e-12))


if __name__ == "__main__":
    test_hclt_zero_preserving()
    test_hmm_zero_preserving()
