from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import math

MIN_EPSILON = 1e-5
MAX_EPSILON = 1.-1e-5


PI = math.pi


def log_min_exp(a, b, epsilon=1e-8):
    """
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
     log(exp(a) - exp(b))
     c + log(exp(a-c) - exp(b-c))
     a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y


def log_normal(x, mean, logvar):
    logp = -0.5 * logvar
    logp += -0.5 * np.log(2 * PI)
    logp += -0.5 * (x - mean) * (x - mean) / torch.exp(logvar)
    return logp


def log_mixture_normal(x, mean, logvar, pi):
    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3), 1)

    logp_mixtures = log_normal(x, mean, logvar)

    logp = torch.log(torch.sum(pi * torch.exp(logp_mixtures), dim=-1) + 1e-8)

    return logp


def sample_normal(mean, logvar):
    y = torch.randn_like(mean)

    x = torch.exp(0.5 * logvar) * y + mean

    return x


def sample_mixture_normal(mean, logvar, pi):
    b, c, h, w, n_mixtures = tuple(map(int, pi.size()))
    pi = pi.view(b * c * h * w, n_mixtures)
    sampled_pi = torch.multinomial(pi, num_samples=1).view(-1)

    # Select mixture params
    mean = mean.view(b * c * h * w, n_mixtures)
    mean = mean[torch.arange(b*c*h*w), sampled_pi].view(b, c, h, w)
    logvar = logvar.view(b * c * h * w, n_mixtures)
    logvar = logvar[torch.arange(b*c*h*w), sampled_pi].view(b, c, h, w)

    y = sample_normal(mean, logvar)

    return y


def log_logistic(x, mean, logscale):
    """
       pdf = sigma([x - mean] / scale) * [1 - sigma(...)] * 1/scale
    """
    scale = torch.exp(logscale)

    u = (x - mean) / scale

    logp = F.logsigmoid(u) + F.logsigmoid(-u) - logscale

    return logp


def sample_logistic(mean, logscale):
    y = torch.rand_like(mean)

    x = torch.exp(logscale) * torch.log(y / (1 - y)) + mean

    return x


def log_discretized_logistic(x, mean, logscale, inverse_bin_width):
    scale = torch.exp(logscale)

    logp = log_min_exp(
        F.logsigmoid((x + 0.5 / inverse_bin_width - mean) / scale),
        F.logsigmoid((x - 0.5 / inverse_bin_width - mean) / scale))

    return logp


def discretized_logistic_cdf(x, mean, logscale, inverse_bin_width):
    scale = torch.exp(logscale)

    cdf = torch.sigmoid((x + 0.5 / inverse_bin_width - mean) / scale)

    return cdf


def sample_discretized_logistic(mean, logscale, inverse_bin_width):
    x = sample_logistic(mean, logscale)

    x = torch.round(x * inverse_bin_width) / inverse_bin_width
    return x


def normal_cdf(value, loc, std):
        return 0.5 * (1 + torch.erf((value - loc) * std.reciprocal() / math.sqrt(2)))


def log_discretized_normal(x, mean, logvar, inverse_bin_width):
    std = torch.exp(0.5 * logvar)
    log_p = torch.log(normal_cdf(x + 0.5 / inverse_bin_width, mean, std) - normal_cdf(x - 0.5 / inverse_bin_width, mean, std) + 1e-7)

    return log_p


def log_mixture_discretized_normal(x, mean, logvar, pi, inverse_bin_width):
    std = torch.exp(0.5 * logvar)

    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3), 1)

    p = normal_cdf(x + 0.5 / inverse_bin_width, mean, std) - normal_cdf(x - 0.5 / inverse_bin_width, mean, std)

    p = torch.sum(p * pi, dim=-1)

    logp = torch.log(p + 1e-8)

    return logp


def sample_discretized_normal(mean, logvar, inverse_bin_width):
    y = torch.randn_like(mean)

    x = torch.exp(0.5 * logvar) * y + mean

    x = torch.round(x * inverse_bin_width) / inverse_bin_width

    return x


def log_mixture_discretized_logistic(x, mean, logscale, pi, inverse_bin_width):
    scale = torch.exp(logscale)

    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3), 1)

    p = torch.sigmoid((x + 0.5 / inverse_bin_width - mean) / scale) \
        - torch.sigmoid((x - 0.5 / inverse_bin_width - mean) / scale)

    p = torch.sum(p * pi, dim=-1)

    logp = torch.log(p + 1e-8)

    return logp


def mixture_discretized_logistic_cdf(x, mean, logscale, pi, inverse_bin_width):
    scale = torch.exp(logscale)

    x = x[..., None]

    cdfs = torch.sigmoid((x + 0.5 / inverse_bin_width - mean) / scale)

    cdf = torch.sum(cdfs * pi, dim=-1)

    return cdf


def sample_mixture_discretized_logistic(mean, logs, pi, inverse_bin_width):
    # Sample mixtures
    b, c, h, w, n_mixtures = tuple(map(int, pi.size()))
    pi = pi.view(b * c * h * w, n_mixtures)
    sampled_pi = torch.multinomial(pi, num_samples=1).view(-1)

    # Select mixture params
    mean = mean.view(b * c * h * w, n_mixtures)
    mean = mean[torch.arange(b*c*h*w), sampled_pi].view(b, c, h, w)
    logs = logs.view(b * c * h * w, n_mixtures)
    logs = logs[torch.arange(b*c*h*w), sampled_pi].view(b, c, h, w)

    y = torch.rand_like(mean)
    x = torch.exp(logs) * torch.log(y / (1 - y)) + mean

    x = torch.round(x * inverse_bin_width) / inverse_bin_width

    return x


def log_multinomial(logits, targets):
    return -F.cross_entropy(logits, targets, reduction='none')


def sample_multinomial(logits):
    b, n_categories, c, h, w = logits.size()
    logits = logits.permute(0, 2, 3, 4, 1)
    p = F.softmax(logits, dim=-1)
    p = p.view(b * c * h * w, n_categories)
    x = torch.multinomial(p, num_samples=1).view(b, c, h, w)
    return x