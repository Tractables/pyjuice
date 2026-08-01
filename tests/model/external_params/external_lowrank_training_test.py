"""
End-to-end training with per-sample low-rank external parameters.

Unit tests pin down each piece in isolation; this one checks they work together over a training run,
which is where a mistake that every static check tolerates would show up. The data is a mixture of two
HMMs whose transition matrices differ by a rank-1 nonnegative term -- inside the model class -- so a
router that sees which mode a sequence came from can fit both, and a single shared transition cannot.

Three properties are asserted, in increasing strength:

  1. a VANISHING correction reproduces the plain HMM's whole training trajectory. If the low-rank path
     perturbed the shared model -- staging, the `logT` shift, the flows -- this drifts.
  2. `theta_shared` still trains monotonically by EM while a live correction is applied. This is the
     exact-EM property, exercised rather than derived.
  3. a router trained ONLY on the gradients this feature returns beats an identically-initialized
     frozen router. Numerically correct gradients are not enough for this: they have to point uphill.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, LowRankSumParams


NUM_STATES = 32
NUM_EMITS = 6
SEQ_LEN = 6
RANK = 4


def _ground_truth(seed):
    g = torch.Generator().manual_seed(seed)

    base = torch.rand([NUM_STATES, NUM_STATES], generator = g) + 0.05
    base = base / base.sum(dim = 1, keepdim = True)

    trans = []
    for _ in range(2):
        a = torch.rand([NUM_STATES, 1], generator = g) * 8.0
        b = torch.rand([1, NUM_STATES], generator = g) * 8.0
        t = base + a @ b
        trans.append(t / t.sum(dim = 1, keepdim = True))

    emit = torch.rand([NUM_STATES, NUM_EMITS], generator = g) + 0.05

    return trans, emit / emit.sum(dim = 1, keepdim = True), g


def _sample(n, trans, emit, g):
    mode = torch.randint(0, 2, [n], generator = g)
    data = torch.zeros([n, SEQ_LEN], dtype = torch.long)

    state = torch.randint(0, NUM_STATES, [n], generator = g)
    for t in range(SEQ_LEN):
        data[:, t] = torch.multinomial(emit[state], 1, generator = g).squeeze(1)
        if t + 1 < SEQ_LEN:
            probs = torch.stack([trans[int(m)][int(s)] for m, s in zip(mode, state)])
            state = torch.multinomial(probs, 1, generator = g).squeeze(1)

    return data, mode


def _build(external, seed):
    torch.manual_seed(seed)

    with juice.set_block_size(NUM_STATES):
        ni = inputs(SEQ_LEN - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = NUM_EMITS))

        src, cur = None, ni
        for var in range(SEQ_LEN - 2, -1, -1):
            cx = ni.duplicate(var, tie_params = True)
            if src is None:
                kw = dict(external_params = LowRankSumParams(rank = RANK, tie_external = True)) \
                     if external else dict()
                src = ns = summate(cur, num_node_blocks = 1, **kw)
            else:
                ns = src.duplicate(cur, tie_params = True)
            cur = multiply(cx, ns)

        root = summate(cur, num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)

    return root, src


class _Router(torch.nn.Module):
    """Per-sample feature -> the factors of one shared transition correction."""

    def __init__(self, n_feat, device):
        super().__init__()
        self.head = torch.nn.Linear(n_feat, 2 * NUM_STATES * RANK)
        torch.nn.init.normal_(self.head.weight, std = 0.05)
        torch.nn.init.constant_(self.head.bias, -3.0)      # start with a small correction
        self.to(device)

    def forward(self, x):
        out = self.head(x)
        half = NUM_STATES * RANK
        b = x.size(0)
        return (out[:, :half].view(b, 1, NUM_STATES, RANK).contiguous(),
                out[:, half:].view(b, 1, NUM_STATES, RANK).contiguous())


def _factors(arm, router, feats, device):
    if arm == "plain":
        return None
    if arm == "vanish":
        t = torch.full((feats.size(0), 1, NUM_STATES, RANK), -float("inf"), device = device)
        return t, t.clone()

    return router(feats)


def _mean_ll(pc, data, feats, src, arm, router, device, batch = 250):
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, data.size(0), batch):
            d = data[i:i + batch]
            if d.size(0) < 16:                       # the CUDA kernel requires batch >= 16
                continue
            uv = _factors(arm, router, feats[i:i + batch], device)
            lls = pc(d) if uv is None else pc(d, sum_external_params = {src: uv})
            tot += float(lls.sum()); n += d.size(0)

    return tot / max(n, 1)


def _train(arm, data, feats, device, epochs, batch = 125, seed = 0):
    root, src = _build(arm != "plain", seed)
    pc = juice.compile(root, verbose = False).to(device)

    torch.manual_seed(seed + 100)
    router = _Router(feats.size(1), device) if arm in ("router", "frozen") else None
    opt = torch.optim.Adam(router.parameters(), lr = 5e-2) if arm == "router" else None

    history = [_mean_ll(pc, data, feats, src, arm, router, device)]

    for epoch in range(epochs):
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)      # same order in every arm
        perm = torch.randperm(data.size(0), generator = gen).to(device)

        for i in range(0, data.size(0) - batch + 1, batch):
            idx = perm[i:i + batch]
            d = data[idx]

            pc.init_param_flows(flows_memory = 0.0)
            uv = _factors(arm, router, feats[idx], device)

            if uv is None:
                pc(d)
                pc.backward(d, allow_modify_flows = False)
            else:
                U, V = uv
                pc(d, sum_external_params = {src: (U.detach(), V.detach())})
                pc.backward(d, allow_modify_flows = False)

                if opt is not None:
                    gU, gV = (g.clone() for g in pc.get_external_params_grad(src))
                    # dLL/dU is returned; the likelihood is ASCENDED, so the loss gradient is its
                    # negation. This is the only thing that updates the router.
                    opt.zero_grad(set_to_none = True)
                    torch.autograd.backward([U, V], [-gU, -gV])
                    opt.step()

            pc.mini_batch_em(step_size = 0.25, pseudocount = 0.05)

        history.append(_mean_ll(pc, data, feats, src, arm, router, device))

    del pc
    torch.cuda.empty_cache()

    return history


def test_lowrank_trains_end_to_end():
    device = torch.device("cuda:0")
    epochs = 8

    trans, emit, g = _ground_truth(0)
    data, mode = _sample(1000, trans, emit, g)
    data = data.to(device)

    feats = torch.nn.functional.one_hot(mode, 2).float().to(device)
    feats = feats + 0.1 * torch.randn(feats.shape, generator = torch.Generator(device = device)
                                      .manual_seed(1), device = device)

    hist = {arm: _train(arm, data, feats, device, epochs)
            for arm in ("plain", "vanish", "frozen", "router")}

    # 1. a vanishing correction must not disturb the shared model, at any point in training
    drift = max(abs(a - b) for a, b in zip(hist["plain"], hist["vanish"]))
    assert drift < 5e-2, f"vanishing correction changed the trajectory by {drift}"

    # 2. EM still improves the shared parameters with a live correction applied
    for arm in ("vanish", "router"):
        h = hist[arm]
        assert h[-1] > h[0], f"{arm}: no improvement, {h[0]} -> {h[-1]}"
        assert all(h[i + 1] >= h[i] - 5e-2 for i in range(len(h) - 1)), \
            f"{arm}: log-likelihood went backwards: {h}"

    # 3. the returned gradients point uphill -- a trained router must beat the same router frozen
    gain = hist["router"][-1] - hist["frozen"][-1]
    assert gain > 5e-3, \
        f"training the router on the returned gradients gained only {gain:+.5f} nats/sample"


if __name__ == "__main__":
    test_lowrank_trains_end_to_end()
    print("training smoke test OK")
