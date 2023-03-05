"""
Collection of flow strategies
"""

from __future__ import print_function

import torch
import numpy as np

from .utils import Base
from .backround import BackRound
from .networks import NN


UNIT_TESTING = False


class SplitFactorCoupling(Base):
    def __init__(self, c_in, factor, height, width, args):
        super().__init__()
        self.n_channels = args.n_channels
        self.kernel = 3
        self.input_channel = c_in
        self.round_approx = args.round_approx
        
        if args.variable_type == 'discrete':
            self.round = BackRound(
                args, inverse_bin_width=2**args.n_bits)
        else:
            self.round = None

        self.split_idx = c_in - (c_in // factor)

        self.nn = NN(
            args=args,
            c_in=self.split_idx,
            c_out=c_in - self.split_idx,
            height=height,
            width=width,
            kernel=self.kernel,
            nn_type=args.coupling_type)

    def forward(self, z, ldj, reverse=False):
        z1 = z[:, :self.split_idx, :, :]
        z2 = z[:, self.split_idx:, :, :]

        t = self.nn(z1)

        if self.round is not None:
            # print("before rounding", t[0, :, 0, 0])
            t = self.round(t)
            # print("after rounding", t[0, :, 0, 0])

        if not reverse:
            z2 = z2 + t
        else:
            z2 = z2 - t

        z = torch.cat([z1, z2], dim=1)

        return z, ldj


class Coupling(Base):
    def __init__(self, c_in, height, width, args):
        super().__init__()

        if args.split_quarter:
            factor = 4
        elif args.splitfactor > 1:
            factor = args.splitfactor
        else:
            factor = 2

        self.coupling = SplitFactorCoupling(
            c_in, factor, height, width, args=args)

    def forward(self, z, ldj, reverse=False):
        return self.coupling(z, ldj, reverse)


def test_generative_flow():
    import models.networks as networks
    global UNIT_TESTING

    networks.UNIT_TESTING = True
    UNIT_TESTING = True

    batch_size = 17

    input_size = [12, 16, 16]

    class Args():
        def __init__(self):
            self.input_size = input_size
            self.learn_split = False
            self.variable_type = 'continuous'
            self.distribution_type = 'logistic'
            self.round_approx = 'smooth'
            self.coupling_type = 'shallow'
            self.conv_type = 'standard'
            self.densenet_depth = 8
            self.bottleneck = False
            self.n_channels = 512
            self.network1x1 = 'standard'
            self.auxilary_freq = -1
            self.actnorm = False
            self.LU = False
            self.coupling_lifting_L = True
            self.splitprior = True
            self.split_quarter = True
            self.n_levels = 2
            self.n_flows = 2
            self.cond_L = True
            self.n_bits = True

    args = Args()

    x = (torch.randint(256, size=[batch_size] + input_size).float() - 128.) / 256.
    ldj = torch.zeros_like(x[:, 0, 0, 0])

    model = Coupling(c_in=12, height=16, width=16, args=args)

    print(model)

    model.set_temperature(1.)
    model.enable_hard_round()

    model.eval()

    z, ldj = model(x, ldj, reverse=False)

    # Check if gradient computation works
    loss = torch.sum(z**2)
    loss.backward()

    recon, ldj = model(z, ldj, reverse=True)

    sse = torch.sum(torch.pow(x - recon, 2)).item()
    ae = torch.abs(x - recon).sum()
    print('Error in recon: sse {} ae {}'.format(sse / np.prod(input_size), ae))


if __name__ == '__main__':
    test_generative_flow()