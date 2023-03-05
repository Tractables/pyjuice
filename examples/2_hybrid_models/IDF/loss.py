from __future__ import print_function

import numpy as np
import torch

from .distributions import log_discretized_logistic, \
    log_mixture_discretized_logistic, log_normal, log_discretized_normal, \
    log_logistic, log_mixture_normal
from .backround import _round_straightthrough


def compute_log_ps(pxs, xs, args):
    # Add likelihoods of intermediate representations.
    inverse_bin_width = 2.**args.n_bits

    log_pxs = []
    for px, x in zip(pxs, xs):

        if args.variable_type == 'discrete':
            if args.distribution_type == 'logistic':
                log_px = log_discretized_logistic(
                    x, *px, inverse_bin_width=inverse_bin_width)
            elif args.distribution_type == 'normal':
                log_px = log_discretized_normal(
                    x, *px, inverse_bin_width=inverse_bin_width)
        elif args.variable_type == 'continuous':
            if args.distribution_type == 'logistic':
                log_px = log_logistic(x, *px)
            elif args.distribution_type == 'normal':
                log_px = log_normal(x, *px)
            elif args.distribution_type == 'steplogistic':
                x = _round_straightthrough(x * inverse_bin_width) / inverse_bin_width
                log_px = log_discretized_logistic(
                    x, *px, inverse_bin_width=inverse_bin_width)

        log_pxs.append(
            torch.sum(log_px, dim=[1, 2, 3]))

    return log_pxs


def compute_log_pz(pz, z, args):
    inverse_bin_width = 2.**args.n_bits

    if args.variable_type == 'discrete':
        if args.distribution_type == 'logistic':
            if args.n_mixtures == 1:
                log_pz = log_discretized_logistic(
                    z, pz[0], pz[1], inverse_bin_width=inverse_bin_width)
            else:
                log_pz = log_mixture_discretized_logistic(
                    z, pz[0], pz[1], pz[2],
                    inverse_bin_width=inverse_bin_width)
        elif args.distribution_type == 'normal':
            log_pz = log_discretized_normal(
                z, *pz, inverse_bin_width=inverse_bin_width)

    elif args.variable_type == 'continuous':
        if args.distribution_type == 'logistic':
            log_pz = log_logistic(z, *pz)
        elif args.distribution_type == 'normal':
            if args.n_mixtures == 1:
                log_pz = log_normal(z, *pz)
            else:
                log_pz = log_mixture_normal(z, *pz)
        elif args.distribution_type == 'steplogistic':
            z = _round_straightthrough(z * 256.) / 256.
            log_pz = log_discretized_logistic(z, *pz)

    log_pz = torch.sum(
        log_pz,
        dim=[1, 2, 3])

    return log_pz


def compute_loss_function(pz, z, pys, ys, ldj, args):
    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """
    batch_size = z.size(0)

    # Get array loss, sum over batch
    loss_array, bpd_array, bpd_per_prior_array = \
        compute_loss_array(pz, z, pys, ys, ldj, args)

    loss = torch.mean(loss_array)
    bpd = torch.mean(bpd_array).item()
    bpd_per_prior = [torch.mean(x) for x in bpd_per_prior_array]

    return loss, bpd, bpd_per_prior


def convert_bpd(log_p, input_size):
    return -log_p / (np.prod(input_size) * np.log(2.))


def compute_loss_array(pz, z, pys, ys, ldj, args):
    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """
    bpd_per_prior = []

    # Likelihood of final representation.
    log_pz = compute_log_pz(pz, z, args)

    bpd_per_prior.append(convert_bpd(log_pz.detach(), args.input_size))

    log_p = log_pz

    # Add likelihoods of intermediate representations.
    if ys:
        log_pys = compute_log_ps(pys, ys, args)

        for log_py in log_pys:
            log_p += log_py

            bpd_per_prior.append(convert_bpd(log_py.detach(), args.input_size))

    log_p += ldj

    loss = -log_p
    bpd = convert_bpd(log_p.detach(), args.input_size)

    return loss, bpd, bpd_per_prior


def calculate_loss(pz, z, pys, ys, ldj, loss_aux, args):
    return compute_loss_function(pz, z, pys, ys, ldj, loss_aux, args)