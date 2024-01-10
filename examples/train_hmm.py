import pyjuice as juice
import torch
import torchvision
import time
import tqdm
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists


def evaluate(pc, loader):
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

        print(f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def full_batch_em_epoch(pc, train_loader, test_loader, device):
    t0 = time.time()
    train_ll = 0.0
    for batch in tqdm.tqdm(train_loader):
        x = batch[0].to(device)

        lls = pc(x)
        lls.mean().backward()

        train_ll += lls.mean().detach().cpu().numpy().item()

    pc.mini_batch_em(step_size = 1.0, pseudocount = 0.1)

    train_ll /= len(train_loader)

    t1 = time.time()
    test_ll = evaluate(pc, loader=test_loader)
    t2 = time.time()
    print(f"[train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def homogenes_hmm(seq_length, num_latents, vocab_size):
    
    
    # with juice.set_group_size(group_size = 32):
    ns_input = juice.inputs(seq_length - 1, num_latents,
                            dists.Categorical(num_cats=vocab_size))
    
    ns_sum = None
    curr_zs = ns_input
    for var in range(seq_length - 2, -1, -1):
        curr_xs = ns_input.duplicate(var, tie_params = True)
        
        if ns_sum is None:
            ns = juice.summate(
                curr_zs, num_nodes = num_latents)
            ns_sum = ns
        else:
            ns = ns_sum.duplicate(curr_zs, tie_params=True)

        curr_zs = juice.multiply(curr_xs, ns)
        
    ns = juice.summate(curr_zs, num_nodes=1)
    
    ns.init_parameters()
    
    return ns


def train_hmm(enable_cudagrph = True):

    device = torch.device("cuda:0")
    # ns = juice.structures.HMM(
    #     16, # T = 16
    #     num_latents = 16384, #8192, 
    #     input_layer_params = {"num_cats": 50257})
    T = 5
    ns = homogenes_hmm(T, 16384, 50257)
    
    pc = juice.TensorCircuit(ns, max_tied_ns_per_parflow_group = 9999999)

    print("finish compilation")
    pc.to(device)


if __name__ == "__main__":
    train_hmm()
