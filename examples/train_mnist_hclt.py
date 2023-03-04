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
import argparse

warnings.filterwarnings("ignore")
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.compile_fx").setLevel(logging.ERROR)

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--cuda', type=int, default=0, help='cuda idx')
    parser.add_argument('--num_latents', type=int, default=32, help='num_latents')
    parser.add_argument("--mode", type=str, default="train", help="options: 'train', 'load'")
    parser.add_argument("--output_dir", type=str, default="examples", help="output directory")
    args = parser.parse_args()
    return args

def evaluate(pc: juice.ProbCircuit, loader: DataLoader):
    lls_total = 0.0
    for batch in loader:
        x = batch[0].to(pc.device)
        lls = pc(x)
        lls_total += lls.mean().detach().cpu().numpy().item()
    
    lls_total /= len(loader)
    return lls_total


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

        train_ll /= len(train_loader)

        t1 = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t2 = time.time()

        print(f"[Epoch {epoch}][train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def full_batch_em_epoch(pc, train_loader, test_loader, device):
    with torch.no_grad():
        t0 = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            lls = pc(x)
            pc.backward(x, flows_memory = 1.0)

            train_ll += lls.mean().detach().cpu().numpy().item()

        pc.mini_batch_em(step_size = 1.0, pseudocount = 0.1)

        train_ll /= len(train_loader)

        t1 = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t2 = time.time()
        print(f"[train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def main(args):
    torch.cuda.set_device(args.cuda)
    device = torch.device(f"cuda:{args.cuda}")
    filename = f"{args.output_dir}/mnist_{args.num_latents}.torch"
    
    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)
    train_data = train_dataset.data.reshape(60000, 28*28)
    test_data = test_dataset.data.reshape(10000, 28*28)

    train_loader = DataLoader(
        dataset = TensorDataset(train_data),
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = TensorDataset(test_data),
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True
    )

    if args.mode == "train":
        print("===========================Train===============================")
        pc = juice.structures.HCLT(train_data.float().to(device), num_bins = 32, 
                                                                    sigma = 0.5 / 32, 
                                                                    num_latents = args.num_latents, 
                                                                    chunk_size = 32)
        pc.to(device)

        optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1)
        scheduler = juice.optim.CircuitScheduler(optimizer, method = "multi_linear", 
                                                            lrs = [0.9, 0.1, 0.05], 
                                                            milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350])

        mini_batch_em_epoch(350, pc, optimizer, scheduler, train_loader, test_loader, device)
        full_batch_em_epoch(pc, train_loader, test_loader, device)
        
        print(f"Saving pc into {filename}.....", end="")
        t0_save = time.time()
        torch.save(pc, filename)
        t1_save = time.time()
        print(f"took {t1_save - t0_save:2.f} (s)")

    elif args.mode == "load":
        print("===========================LOAD===============================")
        t0 = time.time()
        print(f"Loading {filename} into {device}.......", end="")
        pc = torch.load(filename)
        pc.to(device)
        t1 = time.time()
        print(f"Took {t1-t0:.2f} (s)")

        t_compile = time.time()
        test_ll = evaluate(pc, loader=test_loader) # force compilation

        t0 = time.time()
        train_ll = evaluate(pc, loader=train_loader)
        t1 = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t2 = time.time()

        train_bpd = train_ll / (28*28 * np.log(2))
        test_bpd = test_ll / (28*28 * np.log(2))

        print(f"Compilation+test took {t0-t_compile:.2f} (s); train_ll {t1-t0:.2f} (s); test_ll {t2-t1:.2f} (s)")
        print(f"train_ll: {train_ll:.2f}, test_ll: {test_ll:.2f}")
        print(f"train_bpd: {train_bpd:.2f}, test_bpd: {test_bpd:.2f}")

    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024:.1f}GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024:.1f}GB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved(device) / 1024 / 1024 / 1024:.1f}GB")


if __name__ == "__main__":
    args = process_args()
    print(args)
    main(args)