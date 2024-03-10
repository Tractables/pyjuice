import pyjuice as juice
import torch
import torchvision
import time
from tqdm import tqdm
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.distributions as dists

import pytest


def load_penn_treebank(seq_length = 32):

    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?,;:-'\"()[]{}"
    vocab = {char: idx for idx, char in enumerate(CHARS)}

    # Define a tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Load the Penn Treebank dataset
    train_dataset, valid_dataset, test_dataset = PennTreebank(root = "./examples/data")

    train_data = []
    for sample in tqdm(train_dataset):
        train_data.extend([vocab[token] if token in vocab else len(CHARS) for token in sample])

    valid_data = []
    for sample in tqdm(valid_dataset):
        valid_data.extend([vocab[token] if token in vocab else len(CHARS) for token in sample])

    test_data = []
    for sample in tqdm(test_dataset):
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


@pytest.mark.slow
def test_hmm_viterbi():
    
    device = torch.device("cuda:0")

    seq_length = 32

    train_data, valid_data, test_data = load_penn_treebank(seq_length = seq_length)

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

    print(f"> Number of training samples: {train_data.size(0)}")
    print(f"> Number of validation samples: {valid_data.size(0)}")

    root_ns = juice.structures.HMM(
        seq_length = seq_length,
        num_latents = 256,
        num_emits = vocab_size,
        homogeneous = True
    )

    pc = juice.compile(root_ns)
    pc.to(device)

    best_valid_ll = -10000.0
    for epoch in range(1, 20 + 1):
        t0 = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            lls = pc(x, propagation_alg = "MPE")
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

        train_ll /= len(train_loader)

        pc.mini_batch_em(step_size = 1.0, pseudocount = 0.01)

        t1 = time.time()

        with torch.no_grad():
            valid_ll = 0.0
            for batch in valid_loader:
                x = batch[0].to(device)

                lls = pc(x, propagation_alg = "LL")

                valid_ll += lls.mean().detach().cpu().numpy().item()

            valid_ll /= len(valid_loader)

        t2 = time.time()

        print(f"[epoch {epoch:3d}][train LL: {train_ll:.2f}; valid LL: {valid_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}]")

        if valid_ll > best_valid_ll:
            best_valid_ll = valid_ll

    assert best_valid_ll > -90.0


if __name__ == "__main__":
    test_hmm_viterbi()