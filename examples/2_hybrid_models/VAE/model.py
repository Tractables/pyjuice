import torch
import torch.utils.data
from torch import nn, optim
from torchvision import *
import socket
import os
import time
from datetime import datetime
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

import modules as modules
import rand as random


class Model(nn.Module):
    def __init__(self, xs=(3, 32, 32), nz=1, zchannels=16, nprocessing=1, kernel_size=3, resdepth=2,
                 reswidth=256, dropout_p=0., tag='', num_latents = 4):
        super().__init__()
        # default: disable compressing mode
        # if activated, tensors will be flattened
        self.compressing = False

        # hyperparameters
        self.xs = xs
        self.nz = nz
        self.zchannels = zchannels
        self.nprocessing = nprocessing
        # latent height/width is always 16,
        # the number of channels depends on the dataset
        self.zdim = (self.zchannels, 16, 16)
        self.resdepth = resdepth
        self.reswidth = reswidth
        self.kernel_size = kernel_size

        # apply these two factors (i.e. on the ELBO) in sequence and it results in "bits/dim"
        # factor to convert "nats" to bits
        self.bitsscale = np.log2(np.e)
        # factor to divide by the data dimension
        self.perdimsscale = 1. / np.prod(self.xs)

        # calculate processing layers convolutions options
        # kernel/filter is 5, so in order to ensure same-size outputs, we have to pad by 2
        padding_proc = (5 - 1) / 2
        assert padding_proc.is_integer()
        padding_proc = int(padding_proc)

        # calculate other convolutions options
        padding = (self.kernel_size - 1) / 2
        assert padding.is_integer()
        padding = int(padding)

        # set-up current "best elbo"
        self.best_elbo = np.inf

        # distribute ResNet blocks over latent layers
        resdepth = [0] * (self.nz)
        i = 0
        for _ in range(self.resdepth):
            i = 0 if i == (self.nz) else i
            resdepth[i] += 1
            i += 1

        # reduce initial variance of distributions corresponding
        # to latent layers if latent nz increases
        scale = 1.0 / (self.nz ** 0.5)

        # store activations
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ELU()
        self.actresnet = nn.ELU()

        # Below we build up the main model architecture of the inference- and generative-models
        # All the architecure components are built up from different custom are existing PyTorch modules

        # <===== INFERENCE MODEL =====>
        # the bottom (zi=1) inference model
        self.infer_in = nn.Sequential(
            # shape: [1,32,32] -> [4,16,16]
            modules.Squeeze2d(factor=2),

            # shape: [4,16,16] -> [32,16,16]
            modules.WnConv2d(4 * xs[0],
                             self.reswidth,
                             5,
                             1,
                             padding_proc,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )
        self.infer_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.infer_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        # shape: [32,16,16] -> [1,16,16]
        self.infer_mu = modules.WnConv2d(self.reswidth,
                                         self.zchannels,
                                         self.kernel_size,
                                         1,
                                         padding,
                                         init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # shape: [32,16,16] -> [1,16,16]
        self.infer_std = modules.WnConv2d(self.reswidth,
                                          self.zchannels,
                                          self.kernel_size,
                                          1,
                                          padding,
                                          init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # <===== DEEP INFERENCE MODEL =====>
        # the deeper (zi > 1) inference models
        self.deepinfer_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepinfer_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepinfer_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        self.deepinfer_std = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        # <===== DEEP GENERATIVE MODEL =====>
        # the deeper (zi > 1) generative models
        self.deepgen_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepgen_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepgen_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        self.deepgen_std = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding, init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        # <===== GENERATIVE MODEL =====>
        # the bottom (zi = 1) inference model
        self.gen_in = nn.Sequential(
            # shape: [1,16,16] -> [32,16,16]
            modules.WnConv2d(self.zchannels,
                             self.reswidth,
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )

        self.gen_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        self.gen_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.gen_mu = nn.Sequential(
            # shape: [32,16,16] -> [4,16,16]
            modules.WnConv2d(self.reswidth,
                             4 * xs[0] * num_latents,
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=0.1),
            # shape: [4,16,16] -> [1,32,23]
            modules.UnSqueeze2d(factor=2)
        )

        # the scale parameter of the bottom (zi = 1) generative model is modelled unconditional
        self.gen_std = nn.Parameter(torch.zeros([self.xs[0] * num_latents, self.xs[1], self.xs[2]]))
        nn.init.zeros_(self.gen_std)

        self.num_latents = num_latents

    # function to set the model to compression mode
    def compress(self, compress=True):
        self.compressing = compress

    # function that only takes in the layer number and returns a distribution based on that
    def infer(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            h = given

            # if compressing, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()

            # bottom latent layer
            if i == 0:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.xs)
                # also, when NOT compressing, the input is not scaled from [0,255] to [-1,1]
                else:
                    h = (h - 127.5) / 127.5

                # input convolution
                h = self.infer_in(h)

                # processing ResNet blocks
                h = self.infer_res0(h)

                # other ResNet blocks
                h = self.infer_res1(h)

                # mu parameter of the conditional Logistic distribution
                mu = self.infer_mu(h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.infer_std(h) + 2.)

            # deeper latent layers
            else:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.zdim)

                # input convolution
                h = self.deepinfer_in[i - 1](h)

                # other ResNet blocks
                h = self.deepinfer_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepinfer_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.deepinfer_std[i - 1](h) + 2.)

            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.zdim)).type(type)
                scale = scale.view(np.prod(self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that only takes in the layer number and returns a distribution based on that
    def generate(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            h = given

            # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
            # also, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()
                h = h.view((-1,) + self.zdim)

            # bottom latent layer
            if i == 0:
                # input convolution
                h = self.gen_in(h)

                # processing ResNet blocks
                h = self.gen_res1(h)

                # other ResNet blocks
                h = self.gen_res0(h)

                # mu parameter of the conditional Logistic distribution
                mu = self.gen_mu(h)

                # scale parameter of the conditional Logistic distribution
                # set a minimal value for the scale parameter of the bottom generative model
                scale = ((2. / 255.) / 8.) + modules.softplus(self.gen_std)

            # deeper latent layers
            else:
                # input convolution
                h = self.deepgen_in[i - 1](h)

                # other ResNet blocks
                h = self.deepgen_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepgen_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * modules.softplus(self.deepgen_std[i - 1](h) + np.log(np.exp(1.) - 1.))


            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.xs if i == 0 else self.zdim)).type(type)
                scale = scale.view(np.prod(self.xs if i == 0 else self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that takes as input the data and outputs all the components of the ELBO + the latent samples
    def loss(self, x, pc = None):
        # tensor to store inference model losses
        logenc = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the generative model losses
        logdec = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the latent samples
        zsamples = torch.zeros((self.nz, x.shape[0], np.prod(self.zdim)), device=x.device)

        for i in range(self.nz):
            # inference model
            # get the parameters of inference distribution i given x (if i == 0) or z (otherwise)
            mu, scale = self.infer(i)(given=x if i == 0 else z)

            # sample untransformed sample from Logistic distribution (mu=0, scale=1)
            eps = random.logistic_eps(mu.shape, device=mu.device)
            # reparameterization trick: transform using obtained parameters
            z_next = random.transform(eps, mu, scale)

            # store the inference model loss
            zsamples[i] = z_next.flatten(1)
            logq = torch.sum(random.logistic_logp(mu, scale, z_next), dim=2)
            logenc[i] += logq

            # generative model
            # get the parameters of inference distribution i given z
            mu, scale = self.generate(i)(given=z_next)

            import torch.nn.functional as F
            def _log_min_exp(a: torch.Tensor, b: torch.Tensor, epsilon = 1e-8):
                return a + torch.log(1 - torch.exp(b - a) + epsilon)

            def discrete_logistic_ll(x, mean, logscale):
                scale = torch.exp(logscale)
                logp = _log_min_exp(
                    F.logsigmoid((x + 0.5 / 256.0 - mean) / scale),
                    F.logsigmoid((x - 0.5 / 256.0 - mean) / scale))

                return logp

            # store the generative model loss
            if i == 0:
                # if bottom (zi = 1) generative model, evaluate loss using discretized Logistic distribution
                scale = scale.unsqueeze(0).expand_as(mu)
                input_params = {"input_0": {"mus": mu.permute(0, 2, 3, 1).reshape(-1, 32 * 32 * self.num_latents),
                                            "log_scales": scale.log().permute(0, 2, 3, 1).reshape(-1, 32 * 32 * self.num_latents)}}

                if pc is not None:
                    logp = pc(x.reshape(-1, 32 * 32) / 255.0, input_params = input_params)
                else:
                    logp = torch.sum(random.discretized_logistic_logp(mu, scale, x), dim=1)
                
                logrecon = logp

            else:
                logp = torch.sum(random.logistic_logp(mu, scale, x if i == 0 else z), dim=2)
                logdec[i - 1] += logp

            z = z_next

        # store the prior loss
        logp = torch.sum(random.logistic_logp(torch.zeros(1, device=x.device), torch.ones(1, device=x.device), z), dim=2)
        logdec[self.nz - 1] += logp

        # convert from "nats" to bits
        logenc = torch.mean(logenc, dim=1) * self.bitsscale
        logdec = torch.mean(logdec, dim=1) * self.bitsscale
        logrecon = torch.mean(logrecon) * self.bitsscale
        return logrecon, logdec, logenc, zsamples

    # function to sample from the model (using the generative model)
    def sample(self, device, epoch, num=64):
        # sample "num" latent variables from the prior
        z = random.logistic_eps(((num,) + self.zdim), device=device)

        # sample from the generative distribution(s)
        for i in reversed(range(self.nz)):
            mu, scale = self.generate(i)(given=z)
            eps = random.logistic_eps(mu.shape, device=device)
            z_prev = random.transform(eps, mu, scale)
            z = z_prev

        # scale up from [-1,1] to [0,255]
        x_cont = (z * 127.5) + 127.5

        # ensure that [0,255]
        x = torch.clamp(x_cont, 0, 255)

        # scale from [0,255] to [0,1] and convert to right shape
        x_sample = x.float() / 255.
        x_sample = x_sample.view((num,) + self.xs)

        # make grid out of "num" samples
        x_grid = utils.make_grid(x_sample)

    # function to sample a reconstruction of input data
    def reconstruct(self, x_orig, device, epoch):
        # take only first 32 datapoints of the input
        # otherwise the output image grid may be too big for visualization
        x_orig = x_orig[:32, :, :, :].to(device)

        # sample from the bottom (zi = 1) inference model
        mu, scale = self.infer(0)(given=x_orig)
        eps = random.logistic_eps(mu.shape, device=device)
        z = random.transform(eps, mu, scale)  # sample zs

        # sample from the bottom (zi = 1) generative model
        mu, scale = self.generate(0)(given=z)
        x_eps = random.logistic_eps(mu.shape, device=device)
        x_cont = random.transform(x_eps, mu, scale)

        # scale up from [-1.1] to [0,255]
        x_cont = (x_cont * 127.5) + 127.5

        # esnure that [0,255]
        x_sample = torch.clamp(x_cont, 0, 255)

        # scale from [0,255] to [0,1] and convert to right shape
        x_sample = x_sample.float() / 255.
        x_orig = x_orig.float() / 255.

        # concatenate the input data and the sampled reconstructions for comparison
        x_with_recon = torch.cat((x_orig, x_sample))

        # make a grid out of the original data and the reconstruction samples
        x_with_recon = x_with_recon.view((2 * x_orig.shape[0],) + self.xs)
        x_grid = utils.make_grid(x_with_recon)