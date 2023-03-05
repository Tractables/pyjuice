import sys
import pyjuice as juice
import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
import time
import torchvision
import numpy as np
import sys
import logging
import warnings
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Function
from torchvision.transforms import Resize

import argparse
import os

sys.path.append(os.path.dirname(__file__))

# from vae import IWAE
from IDF.idf import Model as IDF

from pyjuice.layer import DiscreteLogisticLayer

warnings.filterwarnings("ignore")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.compile_fx").setLevel(logging.ERROR)


def _log_min_exp(a: torch.Tensor, b: torch.Tensor, epsilon = 1e-8):
    return a + torch.log(1 - torch.exp(b - a) + epsilon)


def discrete_logistic_ll(x, mean, logscale):

    scale = torch.exp(logscale)

    logp = _log_min_exp(
        F.logsigmoid((x + 0.5 / 256.0 - mean) / scale),
        F.logsigmoid((x - 0.5 / 256.0 - mean) / scale))

    return logp


def get_args():
    parser = argparse.ArgumentParser(description='PC+IDF')

    parser.add_argument('--variable_type', type=str, default='discrete',
                        help='variable type of data distribution: discrete/continuous',
                        choices=['discrete', 'continuous'])
    parser.add_argument('--distribution_type', type=str, default='logistic',
                        choices=['logistic', 'normal', 'steplogistic'],
                        help='distribution type: logistic/normal')
    parser.add_argument('--n_flows', type=int, default=8,
                        help='number of flows per level')
    parser.add_argument('--n_levels', type=int, default=3,
                        help='number of levels')

    parser.add_argument('--n_bits', type=int, default=8,
                        help='')

    # ---------------- SETTINGS CONCERNING NETWORKS -------------
    parser.add_argument('--densenet_depth', type=int, default=2,
                        help='Depth of densenets')
    parser.add_argument('--n_channels', type=int, default=512,
                        help='number of channels in coupling and splitprior')
    # ---------------- ----------------------------- -------------


    # ---------------- SETTINGS CONCERNING COUPLING LAYERS -------------
    parser.add_argument('--coupling_type', type=str, default='shallow',
                        choices=['shallow', 'resnet', 'densenet', 'densenet++'],
                        help='Type of coupling layer')
    parser.add_argument('--splitfactor', default=0, type=int,
                        help='Split factor for coupling layers.')

    parser.add_argument('--split_quarter', dest='split_quarter', action='store_true',
                        help='Split coupling layer on quarter')
    parser.add_argument('--no_split_quarter', dest='split_quarter', action='store_false')
    parser.set_defaults(split_quarter=True)
    # ---------------- ----------------------------------- -------------


    # ---------------- SETTINGS CONCERNING SPLITPRIORS -------------
    parser.add_argument('--splitprior_type', type=str, default='densenet',
                        choices=['none', 'shallow', 'resnet', 'densenet', 'densenet++'],
                        help='Type of splitprior. Use \'none\' for no splitprior')
    # ---------------- ------------------------------- -------------


    # ---------------- SETTINGS CONCERNING PRIORS -------------
    parser.add_argument('--n_mixtures', type=int, default=4,
                        help='number of mixtures')
    # ---------------- ------------------------------- -------------

    parser.add_argument('--hard_round', dest='hard_round', action='store_true',
                        help='Rounding of translation in discrete models. Weird '
                        'probabilistic implications, only for experimental phase')
    parser.add_argument('--no_hard_round', dest='hard_round', action='store_false')
    parser.set_defaults(hard_round=True)

    parser.add_argument('--round_approx', type=str, default='smooth',
                        choices=['smooth', 'stochastic'])

    parser.add_argument('--lr_decay', default=0.999, type=float,
                        help='Learning rate')

    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Temperature used for BackRound. It is used in '
                        'the the SmoothRound module. '
                        '(default=1.0')

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    BATCH_SIZE = 64
    NUM_LATENTS = 4

    device = torch.device(f"cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)

    train_data = train_dataset.data.reshape(60000, 28, 28).float()
    test_data = test_dataset.data.reshape(10000, 28, 28).float()

    resize = Resize((32, 32))

    train_data = (resize(train_data) * 256).floor().type(torch.uint8).unsqueeze(1)
    test_data = (resize(test_data) * 256).floor().type(torch.uint8).unsqueeze(1)

    num_features = train_data.size(1)

    train_loader = DataLoader(
        dataset = TensorDataset(train_data),
        batch_size = BATCH_SIZE,
        shuffle = True,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = TensorDataset(test_data),
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = True
    )

    pc = juice.structures.HCLT(
        train_data[:,0,::2,::2].reshape(-1, 16*16).float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = args.n_mixtures, 
        chunk_size = 32,
        input_layer_type = DiscreteLogisticLayer, 
        input_layer_params = {"input_range": [0, 1], "bin_count": 256}
    )
    pc.to(device)

    param_specs = pc.get_param_specs()
    print(param_specs)

    args.input_size = (1, 32, 32)

    idf = IDF(args).to(device)
    idf.set_temperature(args.temperature)
    idf.enable_hard_round(args.hard_round)

    nn_optim = torch.optim.Adam(idf.parameters(), lr = 0.001)
    optimizer = juice.optim.CircuitOptimizer(pc, base_optimizer = nn_optim, lr = 0.1, pseudocount = 0.1)
    scheduler = juice.optim.CircuitScheduler(optimizer, method = "multi_linear", lrs = [0.1, 0.01], 
                                             milestone_steps = [0, len(train_loader) * 200])

    for epoch in range(1, 100+1):
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            pz, z, pys, ys, _ = idf.forward(x, forward_only = True)
    
            ## Latent codes ##
            latent_codes = [
                [z, pz[0], pz[1]], [ys[0], pys[0][0], pys[0][1]], [ys[1], pys[1][0], pys[1][1]]
            ]
            
            ## Pre-process latent codes ##
            for idx in range(len(latent_codes)):
                zs = latent_codes[idx]
                
                n, k = zs[0].size(0), zs[0].size(1) * zs[0].size(2) * zs[0].size(3)
                h = zs[0].view(n, -1) + 0.5
                h_mean = zs[1].reshape(n, -1).reshape(n, -1, k).permute(0, 2, 1)
                h_logscale = zs[2].reshape(n, -1).reshape(n, -1, k).permute(0, 2, 1)
                
                latent_codes[idx] = [h, h_mean, h_logscale]

            input_params = {"input_0": {"mus": latent_codes[0][1].reshape(-1, 16 * 16 * args.n_mixtures).permute(1, 0),
                                        "log_scales": latent_codes[0][2].reshape(-1, 16 * 16 * args.n_mixtures).permute(1, 0)}}

            # Use a PC as the distribution of the first set of latent variables
            lls = pc(latent_codes[0][0], input_params = input_params) + \
                discrete_logistic_ll(latent_codes[1][0], latent_codes[1][1][:,:,0], latent_codes[1][2][:,:,0]).sum(dim = 1) + \
                discrete_logistic_ll(latent_codes[2][0], latent_codes[2][1][:,:,0], latent_codes[2][2][:,:,0]).sum(dim = 1)

            # Original LL
            lls = discrete_logistic_ll(latent_codes[0][0], latent_codes[0][1][:,:,0], latent_codes[0][2][:,:,0]).sum(dim = 1) + \
                discrete_logistic_ll(latent_codes[1][0], latent_codes[1][1][:,:,0], latent_codes[1][2][:,:,0]).sum(dim = 1) + \
                discrete_logistic_ll(latent_codes[2][0], latent_codes[2][1][:,:,0], latent_codes[2][2][:,:,0]).sum(dim = 1)

            loss = -lls.mean()
            loss.backward()
            print(lls.mean().item())

            train_ll += lls.mean().item()

            optimizer.step()
            scheduler.step()

        train_ll /= len(train_loader)
        print(train_ll)


if __name__ == "__main__":
    main()