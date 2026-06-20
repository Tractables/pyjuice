import torch
import pytest

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.model.backend import eval_partition_fn


def _build_pc(device, seed = 0):
    torch.manual_seed(seed)
    xb = torch.randint(0, 8, (600, 16), device = device)
    ns = juice.structures.HCLT(xb.float(), num_latents = 32, num_bins = 8,
                               input_dist = dists.Categorical(num_cats = 8))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)
    return pc


def _data(device, n = 512, seed = 1):
    torch.manual_seed(seed)
    return torch.randint(0, 8, (n, 16), device = device)


def _log_z(pc):
    z = eval_partition_fn(pc)
    return float(z.abs().max()) if torch.is_tensor(z) else abs(float(z))


def _train(pc, opt, data, epochs = 6, chunk = 128):
    for _ in range(epochs):
        for i in range(0, data.size(0), chunk):
            x = data[i:i + chunk]
            pc(x)
            pc.backward(x)
            opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_adam_increases_ll():
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    data = _data(device)
    opt = juice.optim.Adam(pc, lr = 0.05)
    ll0 = pc(data).mean().item()
    _train(pc, opt, data)
    ll1 = pc(data).mean().item()
    assert ll1 > ll0, f"Adam did not increase LL: {ll0:.4f} -> {ll1:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_sgd_increases_ll():
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    data = _data(device)
    opt = juice.optim.SGD(pc, lr = 0.1)
    ll0 = pc(data).mean().item()
    _train(pc, opt, data)
    ll1 = pc(data).mean().item()
    assert ll1 > ll0, f"SGD did not increase LL: {ll0:.4f} -> {ll1:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_sgd_momentum_increases_ll():
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    data = _data(device)
    opt = juice.optim.SGD(pc, lr = 0.1, momentum = 0.9)
    ll0 = pc(data).mean().item()
    _train(pc, opt, data)
    ll1 = pc(data).mean().item()
    assert ll1 > ll0, f"SGD+momentum did not increase LL: {ll0:.4f} -> {ll1:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_grad_optimizers_keep_normalized():
    # After every step the circuit must be renormalized, i.e. the partition function is ~1 (log Z ~ 0),
    # so pc(x) is a valid normalized log-likelihood.
    device = torch.device("cuda:0")
    for make_opt in (lambda pc: juice.optim.Adam(pc, lr = 0.05),
                     lambda pc: juice.optim.SGD(pc, lr = 0.1, momentum = 0.9)):
        pc = _build_pc(device)
        data = _data(device)
        opt = make_opt(pc)
        _train(pc, opt, data, epochs = 3)
        assert _log_z(pc) < 1e-3, f"parameters not normalized after training: |log Z| = {_log_z(pc):.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_grad_niters_per_update():
    # With niters_per_update = k, the params only change on every k-th step.
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    data = _data(device, n = 6 * 64)
    opt = juice.optim.Adam(pc, lr = 0.05, niters_per_update = 3)

    params0 = pc.params.clone()
    for i in range(2):
        x = data[i * 64:(i + 1) * 64]
        pc(x); pc.backward(x); opt.step()
    assert torch.equal(pc.params, params0), "params changed before the update window completed"
    x = data[2 * 64:3 * 64]
    pc(x); pc.backward(x); opt.step()
    assert not torch.equal(pc.params, params0), "params did not change after the update window"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_grad_ddp_noop_without_dist():
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    data = _data(device, n = 64)
    opt = juice.optim.Adam(pc, lr = 0.05, ddp = True)   # ddp sync must be a no-op without a process group
    params0 = pc.params.clone()
    pc(data); pc.backward(data); opt.step()
    assert not torch.equal(pc.params, params0), "ddp=True Adam step did not update params"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_grad_requires_categorical_inputs():
    # Gradient optimizers only support categorical input layers; constructing on a non-categorical
    # input PC must raise a clear error.
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    xb = torch.rand(400, 8, device = device)
    ns = juice.structures.HCLT(xb, num_latents = 16, num_bins = 8,
                               input_dist = dists.Gaussian(mu = 0.0, sigma = 1.0))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns); pc.to(device)
    with pytest.raises(AssertionError):
        juice.optim.Adam(pc, lr = 0.05)
