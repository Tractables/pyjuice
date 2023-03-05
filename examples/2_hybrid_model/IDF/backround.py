import torch
import torch.nn.functional as F
import numpy as np

from .utils import Base


class RoundStraightThrough(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input, out=None)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


_round_straightthrough = RoundStraightThrough().apply


def _stacked_sigmoid(x, temperature, n_approx=3):

    x_ = x - 0.5
    rounded = torch.round(x_)
    x_remainder = x_ - rounded

    size = x_.size()
    x_remainder = x_remainder.view(size + (1,))

    translation = torch.arange(n_approx) - n_approx // 2
    translation = translation.to(device=x.device, dtype=x.dtype)
    translation = translation.view([1] * len(size) + [len(translation)])
    out = torch.sigmoid((x_remainder - translation) / temperature).sum(dim=-1)

    return out + rounded - (n_approx // 2)


class SmoothRound(Base):
    def __init__(self):
        self._temperature = None
        self._n_approx = None
        super().__init__()
        self.hard_round = None

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

        if self._temperature <= 0.05:
            self._n_approx = 1
        elif 0.05 < self._temperature < 0.13:
            self._n_approx = 3
        else:
            self._n_approx = 5

    def forward(self, x):
        assert self._temperature is not None
        assert self._n_approx is not None
        assert self.hard_round is not None

        if self.temperature <= 0.25:
            h = _stacked_sigmoid(x, self.temperature, n_approx=self._n_approx)
        else:
            h = x

        if self.hard_round:
            h = _round_straightthrough(h)

        return h


class StochasticRound(Base):
    def __init__(self):
        super().__init__()
        self.hard_round = None

    def forward(self, x):
        u = torch.rand_like(x)

        h = x + u - 0.5

        if self.hard_round:
            h = _round_straightthrough(h)

        return h


class BackRound(Base):

    def __init__(self, args, inverse_bin_width):
        """
        BackRound is an approximation to Round that allows for Backpropagation.
        Approximate the round function using a sum of translated sigmoids.
        The temperature determines how well the round function is approximated,
        i.e., a lower temperature corresponds to a better approximation, at
        the cost of more vanishing gradients.
        BackRound supports the following settings:
        * By setting hard to True and temperature > 0.25, BackRound
          reduces to a round function with a straight through gradient
          estimator
        * When using 0 < temperature <= 0.25 and hard = True, the
          output in the forward pass is equivalent to a round function, but the
          gradient is approximated by the gradient of a sum of sigmoids.
        * When using hard = False, the output is not constrained to integers.
        * When temperature > 0.25 and hard = False, BackRound reduces to
          the identity function.
        Arguments
        ---------
        temperature: float
            Temperature used for stacked sigmoid approximated. If temperature
            is greater than 0.25, the approximation reduces to the indentiy
            function.
        hard: bool
            If hard is True, a (hard) round is applied before returning. The
            gradient for this is approximated using the straight-through
            estimator.
        """
        super().__init__()
        self.inverse_bin_width = inverse_bin_width
        self.round_approx = args.round_approx

        if args.round_approx == 'smooth':
            self.round = SmoothRound()
        elif args.round_approx == 'stochastic':
            self.round = StochasticRound()
        else:
            raise ValueError

    def forward(self, x):
        if self.round_approx == 'smooth' or self.round_approx == 'stochastic':
            h = x * self.inverse_bin_width

            h = self.round(h)

            return h / self.inverse_bin_width

        else:
            raise ValueError