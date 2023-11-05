import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader


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


def pd_test():

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

    ns = juice.structures.PD(
        data_shape = (28, 28),
        num_latents = 32,
        split_intervals = (4, 4),
        structure_type = "sum_dominated"
    )
    pc = juice.TensorCircuit(ns)

    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.0001)

    mini_batch_em_epoch(350, pc, optimizer, None, train_loader, test_loader, device)
    full_batch_em_epoch(pc, train_loader, test_loader, device)


if __name__ == "__main__":
    torch.manual_seed(2391)
    pd_test()