import sys
import pyjuice as juice
import torch
import torch._dynamo as dynamo
import time
import torchvision
import numpy as np
import sys
import logging
import warnings
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.compile_fx").setLevel(logging.ERROR)

def mini_batch_em_epoch(num_epochs, pc, optimizer, scheduler, train_loader, test_loader, device):
    for epoch in range(num_epochs):
        t0 = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            lls = pc(x)
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

            optimizer.step()
            scheduler.step()

        t1 = time.time()

        train_ll /= len(train_loader)

        test_ll = 0.0
        for batch in test_loader:
            x = batch[0].to(device)

            lls = pc(x)
            test_ll += lls.mean().detach().cpu().numpy().item()
        
        t2 = time.time()

        test_ll /= len(test_loader)
        print(f"[Epoch {epoch}][train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def full_batch_em_epoch(epoch, pc, train_loader, test_loader, device):
    with torch.no_grad():
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            lls = pc(x)
            pc.backward(x, flows_memory = 1.0)

            train_ll += lls.mean().detach().cpu().numpy().item()

        pc.mini_batch_em(step_size = 1.0, pseudocount = 0.1)

        train_ll /= len(train_loader)

        test_ll = 0.0
        for batch in test_loader:
            x = batch[0].to(device)

            lls = pc(x)
            test_ll += lls.mean().detach().cpu().numpy().item()
        
        test_ll /= len(test_loader)
        print(f"Epoch {epoch} - train LL: {train_ll:.2f} - test LL: {test_ll:.2f}")


def main():
    NUM_LATENTS = 32
    BATCH_SIZE = 512
    device = torch.device("cuda:0")
    
    tr_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    ts_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)

    tr_data = tr_dataset.data.reshape(60000, 28*28)
    ts_data = ts_dataset.data.reshape(10000, 28*28)

    train_loader = DataLoader(
        dataset = TensorDataset(tr_data),
        batch_size = BATCH_SIZE,
        shuffle = True,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = TensorDataset(ts_data),
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = True
    )
    
    pc = juice.structures.HCLT(tr_data.float().to(device), num_bins = 32, sigma = 0.5 / 32, num_latents = NUM_LATENTS, chunk_size = 32)
    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1)
    scheduler = juice.optim.CircuitScheduler(optimizer, method = "multi_linear", lrs = [0.9, 0.1, 0.05], 
                                             milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350])

    mini_batch_em_epoch(350, pc, optimizer, scheduler, train_loader, test_loader, device)

    for epoch in range(1):
        full_batch_em_epoch(epoch, pc, train_loader, test_loader, device)

    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024:.1f}GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024:.1f}GB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved(device) / 1024 / 1024 / 1024:.1f}GB")


if __name__ == "__main__":
    main()