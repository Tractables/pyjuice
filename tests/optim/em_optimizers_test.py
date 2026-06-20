import torch
import pytest

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def _build_pc(device, seed = 0):
    # Deterministic small PC: HCLT structure from seeded data + seeded parameter init, so two calls
    # with the same seed produce bit-identical circuits (needed for the equivalence tests below).
    torch.manual_seed(seed)
    xb = torch.randint(0, 8, (600, 16), device = device)
    ns = juice.structures.HCLT(xb.float(), num_latents = 32, num_bins = 8,
                               input_dist = dists.Categorical(num_cats = 8))
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns)
    pc.to(device)
    return pc


def _accumulate(pc, x):
    pc(x)
    pc.backward(x)


def _fixed_batches(device, n_batches = 12, batch_size = 64, seed = 7):
    torch.manual_seed(seed)
    return [torch.randint(0, 8, (batch_size, 16), device = device) for _ in range(n_batches)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_full_batch_em_increases_ll():
    # Full-batch EM is guaranteed not to decrease the (training) log-likelihood.
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    torch.manual_seed(1)
    data = torch.randint(0, 8, (512, 16), device = device)

    opt = juice.optim.FullBatchEM(pc, pseudocount = 0.01)

    ll0 = pc(data).mean().item()
    for _ in range(5):
        for i in range(0, data.size(0), 128):
            _accumulate(pc, data[i:i + 128])
        opt.step()
    ll1 = pc(data).mean().item()

    assert ll1 > ll0, f"full-batch EM did not increase LL: {ll0:.4f} -> {ll1:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_mini_batch_em_increases_ll():
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    torch.manual_seed(1)
    data = torch.randint(0, 8, (512, 16), device = device)

    opt = juice.optim.MiniBatchEM(pc, step_size = 0.5, pseudocount = 0.01)

    ll0 = pc(data).mean().item()
    for _ in range(5):
        for i in range(0, data.size(0), 128):
            x = data[i:i + 128]
            _accumulate(pc, x)
            opt.step()
    ll1 = pc(data).mean().item()

    assert ll1 > ll0, f"mini-batch EM did not increase LL: {ll0:.4f} -> {ll1:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_mini_batch_em_niters_per_update():
    # With niters_per_update = k, an update only fires every k-th step; the others just accumulate.
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    batches = _fixed_batches(device, n_batches = 6)

    opt = juice.optim.MiniBatchEM(pc, step_size = 0.5, niters_per_update = 3)

    params0 = pc.params.clone()
    _accumulate(pc, batches[0]); opt.step()           # iter 1: no update
    _accumulate(pc, batches[1]); opt.step()           # iter 2: no update
    assert torch.equal(pc.params, params0), "params changed before the update window completed"
    _accumulate(pc, batches[2]); opt.step()           # iter 3: update fires
    assert not torch.equal(pc.params, params0), "params did not change after the update window"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_anemone_matches_manual_loop():
    # Anemone must reproduce the hand-written scaled-mini-batch-EM-with-momentum loop exactly.
    device = torch.device("cuda:0")
    step_size, momentum, niters = 0.4, 0.9, 3
    batches = _fixed_batches(device, n_batches = 12)

    pc1 = _build_pc(device, seed = 0)
    opt = juice.optim.Anemone(pc1, step_size = step_size, momentum = momentum,
                              niters_per_update = niters, pseudocount = 1e-6)
    for x in batches:
        _accumulate(pc1, x)
        opt.step()

    # Manual replication on an identical circuit.
    pc2 = _build_pc(device, seed = 0)
    m = momentum
    pc2.init_param_flows(flows_memory = 0.0)
    buf_sum = torch.zeros_like(pc2.param_flows)
    buf_inp = [torch.zeros_like(layer.param_flows) for layer in pc2.input_layer_group]
    upd = 0
    for i, x in enumerate(batches):
        _accumulate(pc2, x)
        if (i + 1) % niters == 0:
            bias = 1.0 - m ** (upd + 1)

            def _ema(flows, buffer):
                flows.mul_(1.0 - m)
                buffer.mul_(m).add_(flows)
                flows.copy_(buffer).div_(bias)

            _ema(pc2.param_flows, buf_sum)
            for buffer, layer in zip(buf_inp, pc2.input_layer_group):
                _ema(layer.param_flows, buffer)

            pc2.mini_batch_em(step_size = step_size, pseudocount = 1e-6, step_size_rescaling = True)
            pc2.init_param_flows(flows_memory = 0.0)
            upd += 1

    assert torch.allclose(pc1.params, pc2.params, rtol = 1e-4, atol = 1e-6), \
        "Anemone sum-layer params diverge from the manual loop"
    for l1, l2 in zip(pc1.input_layer_group, pc2.input_layer_group):
        assert torch.allclose(l1.params, l2.params, rtol = 1e-4, atol = 1e-6), \
            "Anemone input-layer params diverge from the manual loop"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_anemone_first_update_is_unbiased():
    # Bias-corrected momentum: the FIRST update must equal a no-momentum update, since the
    # bias-corrected EMA of a single value reproduces that value exactly.
    device = torch.device("cuda:0")
    batches = _fixed_batches(device, n_batches = 3)

    pc_m = _build_pc(device, seed = 0)
    pc_0 = _build_pc(device, seed = 0)
    opt_m = juice.optim.Anemone(pc_m, step_size = 0.4, momentum = 0.9, niters_per_update = 3)
    opt_0 = juice.optim.Anemone(pc_0, step_size = 0.4, momentum = 0.0, niters_per_update = 3)

    for x in batches:
        _accumulate(pc_m, x); opt_m.step()
    for x in batches:
        _accumulate(pc_0, x); opt_0.step()

    assert torch.allclose(pc_m.params, pc_0.params, rtol = 1e-4, atol = 1e-6), \
        "Anemone first update is not unbiased w.r.t. momentum"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")
def test_optimizer_ddp_is_noop_without_dist():
    # ddp = True must be a safe no-op when torch.distributed is not initialized (single-process use).
    device = torch.device("cuda:0")
    pc = _build_pc(device)
    x = _fixed_batches(device, n_batches = 1)[0]

    opt = juice.optim.MiniBatchEM(pc, step_size = 0.5, ddp = True)
    params0 = pc.params.clone()
    _accumulate(pc, x)
    opt.step()
    assert not torch.equal(pc.params, params0), "ddp=True optimizer step did not update params"
