"""
Collection of flow strategies
"""

from __future__ import print_function

import torch
import numpy as np

from .utils import Base
from .priors import SplitPrior
from .coupling import Coupling


UNIT_TESTING = False


def space_to_depth(x):
    xs = x.size()
    # Pick off every second element
    x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
    # Transpose picked elements next to channels.
    x = x.permute((0, 1, 3, 5, 2, 4)).contiguous()
    # Combine with channels.
    x = x.view(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
    return x


def depth_to_space(x):
    xs = x.size()
    # Pick off elements from channels
    x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
    # Transpose picked elements next to HW dimensions.
    x = x.permute((0, 1, 4, 2, 5, 3)).contiguous()
    # Combine with HW dimensions.
    x = x.view(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
    return x


def int_shape(x):
    return list(map(int, x.size()))


class Flatten(Base):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(Base):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Reverse(Base):
    def __init__(self):
        super().__init__()

    def forward(self, z, reverse=False):
        flip_idx = torch.arange(z.size(1) - 1, -1, -1).long()
        z = z[:, flip_idx, :, :]
        return z


class Permute(Base):
    def __init__(self, n_channels):
        super().__init__()

        permutation = np.arange(n_channels, dtype='int')
        np.random.shuffle(permutation)

        permutation_inv = np.zeros(n_channels, dtype='int')
        permutation_inv[permutation] = np.arange(n_channels, dtype='int')

        self.permutation = torch.from_numpy(permutation)
        self.permutation_inv = torch.from_numpy(permutation_inv)

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z = z[:, self.permutation, :, :]
        else:
            z = z[:, self.permutation_inv, :, :]

        return z, ldj

    def InversePermute(self):
        inv_permute = Permute(len(self.permutation))
        inv_permute.permutation = self.permutation_inv
        inv_permute.permutation_inv = self.permutation
        return inv_permute


class Squeeze(Base):
    def __init__(self):
        super().__init__()

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z = space_to_depth(z)
        else:
            z = depth_to_space(z)
        return z, ldj


class GenerativeFlow(Base):
    def __init__(self, n_channels, height, width, args):
        super().__init__()
        layers = []
        layers.append(Squeeze())
        n_channels *= 4
        height //= 2
        width //= 2

        for level in range(args.n_levels):

            for i in range(args.n_flows):
                perm_layer = Permute(n_channels)
                layers.append(perm_layer)

                layers.append(
                    Coupling(n_channels, height, width, args))
                
                ## IDF++ ##
                inv_perm_layer = perm_layer.InversePermute()
                layers.append(inv_perm_layer)

            if level < args.n_levels - 1:
                if args.splitprior_type != 'none':
                    # Standard splitprior
                    factor_out = n_channels // 2
                    layers.append(SplitPrior(n_channels, factor_out, height, width, args))
                    n_channels = n_channels - factor_out

                layers.append(Squeeze())
                n_channels *= 4
                height //= 2
                width //= 2

        self.layers = torch.nn.ModuleList(layers)
        self.z_size = (n_channels, height, width)

    def forward(self, z, ldj, pys=(), ys=(), reverse=False):
        if not reverse:
            for l, layer in enumerate(self.layers):
                if isinstance(layer, (SplitPrior)):
                    py, y, z, ldj = layer(z, ldj)
                    pys += (py,)
                    ys += (y,)

                else:
                    z, ldj = layer(z, ldj)

        else:
            for l, layer in reversed(list(enumerate(self.layers))):
                if isinstance(layer, (SplitPrior)):
                    if len(ys) > 0:
                        z, ldj = layer.inverse(z, ldj, y=ys[-1])
                        # Pop last element
                        ys = ys[:-1]
                    else:
                        z, ldj = layer.inverse(z, ldj, y=None)

                else:
                    z, ldj = layer(z, ldj, reverse=True)

        return z, ldj, pys, ys

    def decode(self, z, ldj, state, decode_fn):

        for l, layer in reversed(list(enumerate(self.layers))):
            if isinstance(layer, SplitPrior):
                z, ldj, state = layer.decode(z, ldj, state, decode_fn)

            else:
                z, ldj = layer(z, ldj, reverse=True)

        return z, ldj