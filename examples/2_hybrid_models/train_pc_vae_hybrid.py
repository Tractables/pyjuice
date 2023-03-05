import torch
import torch.utils.data
from torch import nn, optim
from torchvision import *
import socket
import os
import time
from datetime import datetime
import numpy as np
import argparse
import sys
import pyjuice as juice
import warnings
import logging
from pyjuice.layer import DiscreteLogisticLayer

sys.path.append(os.path.dirname(__file__))

import VAE.modules as modules
import VAE.rand as random
from VAE.model import Model as VAE

warnings.filterwarnings("ignore")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.compile_fx").setLevel(logging.ERROR)


def warmup(model, device, data_loader, warmup_batches):
    # convert model to evaluation mode (no Dropout etc.)
    model.eval()

    # prepare initialization batch
    for batch_idx, (image, _) in enumerate(data_loader):
        # stack image with to current stack
        warmup_images = torch.cat((warmup_images, image), dim=0) \
            if batch_idx != 0 else image

        # stop stacking batches if reaching limit
        if batch_idx + 1 == warmup_batches:
            break

    # set the stack to current device
    warmup_images = warmup_images.to(device)

    # do one 'special' forward pass to initialize parameters
    with modules.init_mode():
        logrecon, logdec, logenc, _ = model.loss(warmup_images)

    # log
    logdec = torch.sum(logdec, dim=1)
    logenc = torch.sum(logenc, dim=1)

    elbo = -logrecon + torch.sum(-logdec + logenc)

    elbo = elbo.detach().cpu().numpy() * model.perdimsscale
    entrecon = -logrecon.detach().cpu().numpy() * model.perdimsscale
    entdec = -logdec.detach().cpu().numpy() * model.perdimsscale
    entenc = -logenc.detach().cpu().numpy() * model.perdimsscale

    kl = entdec - entenc

    print(f'====> Epoch: {0} Average loss: {elbo:.4f}')


def train(model, pc, device, epoch, data_loader, optimizer, log_interval, schedule=True, decay=0.99995):
    # convert model to train mode (activate Dropout etc.)
    model.train()

    # get number of batches
    nbatches = data_loader.batch_sampler.sampler.num_samples // data_loader.batch_size

    # setup training metrics
    elbos = torch.zeros((nbatches), device=device)
    logrecons = torch.zeros((nbatches), device=device)
    logdecs = torch.zeros((nbatches, model.nz), device=device)
    logencs = torch.zeros((nbatches, model.nz), device=device)

    start_time = time.time()

    # allocate memory for data
    data = torch.zeros((data_loader.batch_size,) + model.xs, device=device)

    # enumerate over the batches
    for batch_idx, (batch, _) in enumerate(data_loader):
        # keep track of the global step
        global_step = (epoch - 1) * len(data_loader) + (batch_idx + 1)

        # update the learning rate according to schedule
        if schedule:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr = lr_step(global_step, lr, decay=decay)
                param_group['lr'] = lr

        # empty all the gradients stored
        optimizer.zero_grad()

        # copy the mini-batch in the pre-allocated data-variable
        data.copy_(batch)

        # evaluate the data under the model and calculate ELBO components
        logrecon, logdec, logenc, zsamples = model.loss(data, pc)

        # free bits technique, in order to prevent posterior collapse
        bits_pc = 1.
        kl = torch.sum(torch.max(-logdec + logenc, bits_pc * torch.ones((model.nz, model.zdim[0]), device=device)))

        # compute the inference- and generative-model loss
        logdec = torch.sum(logdec, dim=1)
        logenc = torch.sum(logenc, dim=1)

        # construct ELBO
        elbo = -logrecon + kl

        # scale by image dimensions to get "bits/dim"
        elbo *= model.perdimsscale
        logrecon *= model.perdimsscale
        logdec *= model.perdimsscale
        logenc *= model.perdimsscale

        # calculate gradients
        elbo.backward()

        # take gradient step
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
        optimizer.step()

        # log
        elbos[batch_idx] += elbo
        logrecons[batch_idx] += logrecon
        logdecs[batch_idx] += logdec
        logencs[batch_idx] += logenc

        # log and save parameters
        if batch_idx % log_interval == 0 and log_interval < nbatches:
            # print metrics to console
            print(f'Train Epoch: {epoch} [{batch_idx}/{nbatches} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {elbo.item():.6f}\tGnorm: {total_norm:.2f}\tSteps/sec: {(time.time() - start_time) / (batch_idx + 1):.3f}')

            entrecon = -logrecon
            entdec = -logdec
            entenc = -logenc
            kl = entdec - entenc

    # print the average loss of the epoch to the console
    elbo = torch.mean(elbos).detach().cpu().numpy()
    print(f'====> Epoch: {epoch} Average loss: {elbo:.4f}')


def test(model, pc, device, epoch, data_loader, tag):
    # convert model to evaluation mode (no Dropout etc.)
    model.eval()

    # setup the reconstruction dataset
    recon_dataset = None
    nbatches = data_loader.batch_sampler.sampler.num_samples // data_loader.batch_size
    recon_batch_idx = int(torch.Tensor(1).random_(0, nbatches - 1))

    # setup testing metrics
    logrecons = torch.zeros((nbatches), device=device)
    logdecs = torch.zeros((nbatches, model.nz), device=device)
    logencs = torch.zeros((nbatches, model.nz), device=device)

    elbos = []

    # allocate memory for the input data
    data = torch.zeros((data_loader.batch_size,) + model.xs, device=device)

    # enumerate over the batches
    for batch_idx, (batch, _) in enumerate(data_loader):
        # save batch for reconstruction
        if batch_idx == recon_batch_idx:
            recon_dataset = data

        # copy the mini-batch in the pre-allocated data-variable
        data.copy_(batch)

        with torch.no_grad():
            # evaluate the data under the model and calculate ELBO components
            logrecon, logdec, logenc, _ = model.loss(data, pc)

            # construct the ELBO
            elbo = -logrecon + torch.sum(-logdec + logenc)

            # compute the inference- and generative-model loss
            logdec = torch.sum(logdec, dim=1)
            logenc = torch.sum(logenc, dim=1)

        # scale by image dimensions to get "bits/dim"
        elbo *= model.perdimsscale
        logrecon *= model.perdimsscale
        logdec *= model.perdimsscale
        logenc *= model.perdimsscale

        elbos.append(elbo.item())

        # log
        logrecons[batch_idx] += logrecon
        logdecs[batch_idx] += logdec
        logencs[batch_idx] += logenc

    elbo = np.mean(elbos)

    entrecon = -torch.mean(logrecons).detach().cpu().numpy()
    entdec = -torch.mean(logdecs, dim=0).detach().cpu().numpy()
    entenc = -torch.mean(logencs, dim=0).detach().cpu().numpy()
    kl = entdec - entenc

    # print metrics to console and Tensorboard
    print(f'\nEpoch: {epoch}\tTest loss: {elbo:.6f}')

# learning rate schedule
def lr_step(step, curr_lr, decay=0.99995, min_lr=5e-4):
    # only decay after certain point
    # and decay down until minimal value
    if curr_lr > min_lr:
        curr_lr *= decay
        return curr_lr
    return curr_lr


if __name__ == '__main__':
    # hyperparameters, input from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=99, type=int, help="seed for experiment reproducibility")
    parser.add_argument('--nz', default=8, type=int, help="number of latent variables, greater or equal to 1")
    parser.add_argument('--zchannels', default=1, type=int, help="number of channels for the latent variables")
    parser.add_argument('--nprocessing', default=4, type=int, help='number of processing layers')
    parser.add_argument('--gpu', default=0, type=int, help="number of gpu's to distribute optimization over")
    parser.add_argument('--interval', default=100, type=int, help="interval for logging/printing of relevant values")
    parser.add_argument('--epochs', default=10000000000, type=int, help="number of sweeps over the dataset (epochs)")
    parser.add_argument('--blocks', default=8, type=int, help="number of ResNet blocks")
    parser.add_argument('--width', default=64, type=int, help="number of channels in the convolutions in the ResNet blocks")
    parser.add_argument('--dropout', default=0.2, type=float, help="dropout rate of the hidden units")
    parser.add_argument('--kernel', default=3, type=int, help="size of the convolutional filter (kernel) in the ResNet blocks")
    parser.add_argument('--batch', default=128, type=int, help="size of the mini-batch for gradient descent")
    parser.add_argument('--lr', default=2e-3, type=float, help="learning rate gradient descent")
    parser.add_argument('--schedule', default=1, type=float, help="learning rate schedule: yes (1) or no (0)")
    parser.add_argument('--decay', default=0.9995, type=float, help="decay of the learning rate when using learning rate schedule")
    parser.add_argument('--num-latents', default=1, type=int, help="")

    args = parser.parse_args()
    print(args) # print all the hyperparameters

    # store hyperparameters in variables
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch
    nz = args.nz
    zchannels = args.zchannels
    nprocessing = args.nprocessing
    gpu = args.gpu
    blocks = args.blocks
    width = args.width
    log_interval = args.interval
    dropout = args.dropout
    kernel = args.kernel
    lr = args.lr
    schedule = True if args.schedule == 1 else False
    decay = args.decay
    assert nz > 0

    # setup seeds to maintain experiment reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # set GPU/CPU options
    use_cuda = torch.cuda.is_available()
    cudastring = f"cuda:{gpu}"
    device = torch.device(cudastring if use_cuda else "cpu")

    # set number of workers and pin the memory if we distribute over multiple gpu's
    # (see Dataloader docs of PyTorch)
    kwargs = {}

    # create class that scales up the data to [0,255] if called
    class ToInt:
        def __call__(self, pic):
            return pic * 255

    # set data pre-processing transforms
    transform_ops = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])

    train_set = datasets.MNIST(root="examples/data", train=True, transform=transform_ops, download=True)
    test_set = datasets.MNIST(root="examples/data", train=False, transform=transform_ops, download=True)

    # setup mini-batch enumerator for both train-set and test-set
    train_loader = torch.utils.data.DataLoader(
        dataset = train_set,
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_set, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = True, 
    )

    # store MNIST data shape
    xs = (1, 32, 32)

    # build model from hyperparameters
    model = VAE(xs=xs,
                  kernel_size=kernel,
                  nprocessing=nprocessing,
                  nz=nz,
                  zchannels=zchannels,
                  resdepth=blocks,
                  reswidth=width,
                  dropout_p=dropout,
                  tag="tag",
                  num_latents=args.num_latents).to(device)

    inputs = [juice.graph.InputRegionNode(
        scope = [i], num_nodes = args.num_latents, node_type = DiscreteLogisticLayer, input_range = [0, 1], bin_count = 256
    ) for i in range(32*32)]
    prods = [juice.graph.PartitionNode(
        children = [inputs[i]], num_nodes = args.num_latents, edge_ids = torch.arange(0, args.num_latents).unsqueeze(1)
    ) for i in range(32*32)]
    sums = [juice.graph.InnerRegionNode(
        children = [prods[i]], num_nodes = 1, edge_ids = torch.stack((torch.zeros([args.num_latents], dtype = torch.long), torch.arange(0, args.num_latents)), dim = 0)
    ) for i in range(32*32)]
    rnode = juice.graph.InnerRegionNode(
        children = [juice.graph.PartitionNode(
            children = sums, num_nodes = 1, edge_ids = torch.zeros([32*32]).unsqueeze(0)
        )],
        num_nodes = 1, edge_ids = torch.zeros([2, 1], dtype = torch.long)
    )
    pc = juice.ProbCircuit(rnode)
    pc.to(device)

    # set up Adam optimizer
    nn_optim = optim.Adam(model.parameters(), lr = lr)
    optimizer = juice.optim.CircuitOptimizer(pc, base_optimizer = nn_optim, lr = 0.1, pseudocount = 0.1)

    print("Data Dependent Initialization")
    # data-dependent initialization
    warmup(model, device, train_loader, 25)

    # do the training loop and run over the test-set 1/5 epochs.
    print("Training")
    for epoch in range(1, epochs + 1):
        train(model, pc, device, epoch, train_loader, optimizer, log_interval, schedule, decay)
        if epoch % 5 == 0:
            test(model, pc, device, epoch, test_loader, tag)