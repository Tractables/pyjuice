"""
Collection of flow strategies
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Base


UNIT_TESTING = False


class Conv2dReLU(Base):
    def __init__(
            self, n_inputs, n_outputs, kernel_size=3, stride=1, padding=0,
            bias=True):
        super().__init__()

        self.nn = nn.Conv2d(n_inputs, n_outputs, kernel_size, padding=padding)

    def forward(self, x):
        h = self.nn(x)

        y = F.relu(h)

        return y
    
    
class Conv2dGnSwish(Base):
    def __init__(self, n_inputs, n_outputs, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()
        
        self.nn = nn.Conv2d(n_inputs, n_outputs, kernel_size, padding=padding)
        
        if n_outputs % 3 == 0:
            num_groups = 3
        elif n_outputs % 2 == 0:
            num_groups = 2
        else:
            num_groups = 1
        
        self.groupnorm = nn.GroupNorm(num_groups, n_outputs)
        self.swish = nn.SiLU()
        
    def forward(self, x):
        h = self.nn(x)
        h = self.groupnorm(h)
        y = self.swish(h)
        return y


class ResidualBlock(Base):
    def __init__(self, n_channels, kernel, Conv2dAct):
        super().__init__()

        self.nn = torch.nn.Sequential(
            Conv2dAct(n_channels, n_channels, kernel, padding=1),
            torch.nn.Conv2d(n_channels, n_channels, kernel, padding=1),
            )

    def forward(self, x):
        h = self.nn(x)
        h = F.relu(h + x)
        return h


class DenseLayer(Base):
    def __init__(self, args, n_inputs, growth, Conv2dAct):
        super().__init__()

        conv1x1 = Conv2dAct(
                n_inputs, n_inputs, kernel_size=1, stride=1,
                padding=0, bias=True)

        self.nn = torch.nn.Sequential(
            conv1x1,
            Conv2dAct(
                n_inputs, growth, kernel_size=3, stride=1,
                padding=1, bias=True),
            )

    def forward(self, x):
        h = self.nn(x)

        h = torch.cat([x, h], dim=1)
        return h


class DenseBlock(Base):
    def __init__(
            self, args, n_inputs, n_outputs, kernel, Conv2dAct):
        super().__init__()
        depth = args.densenet_depth

        future_growth = n_outputs - n_inputs

        layers = []

        for d in range(depth):
            growth = future_growth // (depth - d)

            layers.append(DenseLayer(args, n_inputs, growth, Conv2dAct))
            n_inputs += growth
            future_growth -= growth

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        y = self.nn(x)
        return y


class Identity(Base):
    def __init__(self):
        super.__init__()

    def forward(self, x):
        return x


class NN(Base):
    def __init__(
            self, args, c_in, c_out, height, width, nn_type, kernel=3):
        super().__init__()

        Conv2dAct = Conv2dReLU
        n_channels = args.n_channels

        if nn_type == 'shallow':

            if True or args.network1x1 == 'standard':
                conv1x1 = Conv2dAct(
                    n_channels, n_channels, kernel_size=1, stride=1,
                    padding=0, bias=False)

            layers = [
                Conv2dAct(c_in, n_channels, kernel, padding=1),
                conv1x1]

            layers += [torch.nn.Conv2d(n_channels, c_out, kernel, padding=1)]

        elif nn_type == 'resnet':
            layers = [
                Conv2dAct(c_in, n_channels, kernel, padding=1),
                ResidualBlock(n_channels, kernel, Conv2dAct),
                ResidualBlock(n_channels, kernel, Conv2dAct)]

            layers += [
                torch.nn.Conv2d(n_channels, c_out, kernel, padding=1)
                ]

        elif nn_type == 'densenet':
            layers = [
                DenseBlock(
                    args=args,
                    n_inputs=c_in,
                    n_outputs=n_channels + c_in,
                    kernel=kernel,
                    Conv2dAct=Conv2dAct)]

            layers += [
                torch.nn.Conv2d(n_channels + c_in, c_out, kernel, padding=1)
                ]
        elif nn_type == 'densenet++':
            Conv2dAct = Conv2dGnSwish
            
            layers = [
                DenseBlock(
                    args=args,
                    n_inputs=c_in,
                    n_outputs=n_channels + c_in,
                    kernel=kernel,
                    Conv2dAct=Conv2dAct)]

            layers += [
                torch.nn.Conv2d(n_channels + c_in, c_out, kernel, padding=1)
                ]
        else:
            raise ValueError

        self.nn = torch.nn.Sequential(*layers)

        # Set parameters of last conv-layer to zero.
        if not UNIT_TESTING:
            self.nn[-1].weight.data.zero_()
            self.nn[-1].bias.data.zero_()

    def forward(self, x):
        y = self.nn(x)
        return y