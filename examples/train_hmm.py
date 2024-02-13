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
            if scheduler is not None:
                scheduler.step()

        train_ll /= len(train_loader)

        t1 = time.time()
        test_ll = evaluate(pc, loader=test_loader)
        t2 = time.time()

        print(f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; test LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; test forward {t2-t1:.2f}] ")


def full_batch_em_epoch(pc, train_loader, test_loader, device):

    pc.init_param_flows(flows_memory = 0.0)

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
    
    group_size = min(juice.utils.util.max_cdf_power_of_2(num_latents), 1024)
    num_node_groups = num_latents // group_size
    
    with juice.set_group_size(group_size = group_size):
        ns_input = juice.inputs(seq_length - 1, num_node_groups = num_node_groups,
                                dist = dists.Categorical(num_cats = vocab_size))
        
        ns_sum = None
        curr_zs = ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = True)
            
            if ns_sum is None:
                ns = juice.summate(
                    curr_zs, num_node_groups = num_node_groups)
                ns_sum = ns
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params=True)

            curr_zs = juice.multiply(curr_xs, ns)
            
        ns = juice.summate(curr_zs, num_node_groups = 1, group_size = 1)
    
    ns.init_parameters()
    
    return ns


def train_hmm(enable_cudagrph = True):

    device = torch.device("cuda:0")

    T = 32
    ns = homogenes_hmm(T, 8192, 4023)
    
    pc = juice.TensorCircuit(ns, max_tied_ns_per_parflow_group = 2)
    pc.print_statistics()

    pc.to(device)

    data = torch.randint(0, 10000, (6400, T))

    data_loader = DataLoader(
        dataset = TensorDataset(data),
        batch_size = 64,
        shuffle = True,
        drop_last = True
    )

    optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.0001)

    for batch in tqdm.tqdm(data_loader):
        x = batch[0].to(device)

        lls = pc(x)
        lls.mean().backward()

        break

    torch.cuda.synchronize()
    t0 = time.time()

    for batch in tqdm.tqdm(data_loader):
        x = batch[0].to(device)

        lls = pc(x)
        lls.mean().backward()

    torch.cuda.synchronize()
    t1 = time.time()

    print((t1-t0)/100*1000, "ms")

    # mini_batch_em_epoch(350, pc, optimizer, None, data_loader, data_loader, device)


if __name__ == "__main__":
    train_hmm()
