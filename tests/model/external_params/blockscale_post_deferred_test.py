"""
BlockScale's post-backward deferred to the end of the pass (`pyjuice.nodes.external_params.blockscale.post_deferred`):
each gated layer of a circuit's backward leaves its `_bs_post_backward_kernel` launch to the end of the inner layers,
where the tuners choose between those launches and the deferred kernels over every layer at once
(`kernels/blockscale_post_backward_deferred.py`). The deferred kernels compute what the per-layer ones compute: on
the inputs of every layer of real backwards, the shifted node values and the gradient's log-Z term to the bit, `W` to
the bit where a layer is one batch slice, else to its float64 value as the per-layer kernel does (whose slices then add
in no fixed order). A circuit's backward that defers: node values and flows to the bit of one that does not, its
gradients and denominator flows to their summation order; replayed from a CUDA graph, its eager values.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
import pyjuice.layer.kernels.autotune as autotune
import pyjuice.nodes.external_params.blockscale.post_deferred as pdm

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
NUM_CATS = 16


def _chain(L = 256, nb = 4, gcb = 32, T = 5, tied = True, z = True):
    """A chain of BlockScale-gated sum layers (an HMM's transitions: `nb` node blocks of `L // nb`, gates of `gcb`
    children), tied or not, and its gated nodes."""
    torch.manual_seed(0)
    gated = []
    with juice.set_block_size(L // nb):
        ch = inputs(0, num_node_blocks = nb, dist = dists.Categorical(num_cats = NUM_CATS))
        first = None
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = nb, dist = dists.Categorical(num_cats = NUM_CATS))
            if first is None or not tied:
                ch = summate(multiply(ch, emit), num_node_blocks = nb,
                             external_params = BlockScaleSumParams(ch_block_size = gcb, apply_z_correction = z))
                first = first or ch
            else:
                ch = first.duplicate(multiply(ch, emit), tie_params = True)
            gated.append(ch)
        root = summate(multiply(ch), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, gated, T


def _circuit(B, **kw):
    root, gated, T = _chain(**kw)
    pc = juice.compile(root, verbose = False).to(DEV)
    g = torch.Generator().manual_seed(1)
    phi = {n: (torch.randn(n.external_params.tensor_shapes(n, B)[0], generator = g) * 2).to(DEV) for n in gated}
    x = torch.randint(0, NUM_CATS, (B, T), device = DEV, generator = torch.Generator(device = DEV).manual_seed(2))
    return pc, gated, phi, x


def _step(pc, phi, x, grad = True):
    grads = {n: torch.zeros_like(p) for n, p in phi.items()} if grad else None
    pc.init_param_flows(flows_memory = 0.0)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 1.0, **(dict(sum_external_params_grad = grads) if grad else
                                          dict(compute_external_grads = False)))
    return grads


def _members(pc, phi, x, monkeypatch, grad = True):
    """Every partition's post-backward of a backward, as `_post_fused` hands it on (launched per layer)."""
    got = []
    with monkeypatch.context() as m:
        m.setattr(pdm, "try_defer", lambda owner, layer, member: got.append((owner, member)) and False)
        for _ in range(2):
            got.clear()
            _step(pc, phi, x, grad)
    return got[0][0], [mem for _, mem in got]


def _run(owner, members, mode, nm0):
    nm = nm0.clone()
    g0 = members[0]["grad_ext"]
    gr = None if g0 is None else torch.zeros_like(g0)
    ws = {pdm._ptr(m["W"]): torch.zeros_like(m["W"]) for m in members if m["W"] is not None}
    if mode == "layers":
        for m in members:
            pdm._launch_layer(owner, m, nm, gr, ws.get(pdm._ptr(m["W"])))
    else:
        pdm._launch_deferred(members, pdm._tables(members), nm, gr, ws)
    torch.cuda.synchronize()
    return nm, gr, ws


def _w_float64(members):
    """`W` per buffer in float64: the sum over the members and samples of exp(node flow - log Z) phi."""
    out = {}
    for m in members:
        c = m["plan"][1]
        bs, rows, ng, ncg, B = c["BLOCK_SIZE_M"], m["rows"], c["n_gates"], c["N_CHILD_GATES"], c["batch_size"]
        nrows = (m["nids"].long()[:, None] + torch.arange(bs, device = DEV)[None, :]).reshape(-1)
        v = torch.exp(m["node_flows"][nrows].double() - m["log_z"].view(rows * bs, B).double()).view(rows, bs, B)
        gb = m["gb"].long()
        gi = torch.arange(ng, device = DEV)
        j = gi // ncg
        base = torch.where(j[None, :] < gb.size(1), gb[:, j.clamp(max = gb.size(1) - 1)], torch.full_like(gb[:, :1], -1))
        grow = base + int(c["ext_base"]) + (gi % ncg)[None, :]
        ph = torch.where((base >= 0)[..., None], torch.exp(m["ext"].view(-1, B)[grow.clamp(min = 0)].double()),
                         torch.zeros((), dtype = torch.float64, device = DEV))
        p = pdm._ptr(m["W"])
        out[p] = out.get(p, 0) + torch.einsum("rnb,rgb->rng", v, ph).reshape(rows * bs, ng)
    return out


# (latents, node blocks, gate width, batch, tied, z-correction, gradient): one batch slice and several, a batch off the
# batch tile, a node block of 8 (no `tl.dot`), untied layers, the gradient or `W` alone (each layer has a `W` of its own,
# tied or not: the deferred `W` kernel walks one layer per `W`)
CASES = [(256, 4, 32, 32, True, True, True), (256, 4, 32, 128, True, True, True), (256, 4, 32, 48, True, True, True),
         (32, 4, 4, 32, True, True, True), (256, 4, 32, 64, False, True, True), (256, 4, 32, 64, True, False, True),
         (256, 4, 32, 64, True, True, False)]


@cuda_only
@pytest.mark.parametrize("L, nb, gcb, B, tied, z, grad", CASES)
def test_the_deferred_kernels_compute_the_layers_bits(L, nb, gcb, B, tied, z, grad, monkeypatch):
    """On every layer's inputs of real backwards: the shifted node values and the gradient's log-Z term to the bit of the
    layers' own launches, `W` to the bit where a layer is one batch slice, else to float64 as the layers' own."""
    pc, gated, phi, x = _circuit(B, L = L, nb = nb, gcb = gcb, tied = tied, z = z)
    owner, members = _members(pc, phi, x, monkeypatch, grad)
    assert len(members) == len(gated)
    assert (members[0]["grad_ext"] is not None) == grad and (members[0]["W"] is not None) == z
    nm0 = members[0]["node_mars"].clone() + 0.25
    a, b = _run(owner, members, "layers", nm0), _run(owner, members, "deferred", nm0)
    assert torch.equal(a[0], b[0]) and not torch.equal(a[0], nm0)
    if grad:
        assert torch.equal(a[1], b[1]) and a[1].abs().max() > 0
    assert len(a[2]) == (len({pdm._ptr(m["W"]) for m in members}) if z else 0)
    ref = _w_float64(members) if z else {}
    for p in a[2]:
        if members[0]["plan"][1]["W_ATOMIC"] == 0:
            assert torch.equal(a[2][p], b[2][p])
        scale = ref[p].abs().max()
        err_a = ((a[2][p].double() - ref[p]).abs().max() / scale).item()
        err_b = ((b[2][p].double() - ref[p]).abs().max() / scale).item()
        assert err_b <= max(2 * err_a, 1e-6), (err_a, err_b)


def _forced(monkeypatch, choice):
    real = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, *a, **k: choice if key[0] == "bs_post_deferred"
                        else real(key, cands, bench, *a, **k))


@cuda_only
@pytest.mark.parametrize("choice", ["deferred", "layers"])
def test_a_deferring_backward_computes_what_one_launching_per_layer_computes(choice, monkeypatch):
    """One circuit's backwards in turn launching every layer's post-backward at once (the deferral off) and deferring it
    (either choice at the end), on the same forward: node values and node flows to the bit, the gradients, the
    denominator and the parameter flows to their summation order (the element kernel adds its gradient term
    atomically)."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    B = 64
    pc, gated, phi, x = _circuit(B)
    _forced(monkeypatch, choice)
    deferred, real = [], pdm.try_defer
    _step(pc, phi, x)                                          # (the kernels' choices settle)
    for it in range(2):
        out = []
        for on in (False, True):
            with monkeypatch.context() as m:
                m.setattr(pdm, "try_defer", (lambda *a, **kw: deferred.append(1) or real(*a, **kw)) if on
                          else (lambda *a, **kw: False))
                grads = _step(pc, phi, x)
            out.append((pc.node_mars.clone(), pc.node_flows.clone(), [grads[n] for n in gated], pc.param_flows.clone(),
                        pc.denom_param_flows.clone()))
        (nm0, nf0, g0, pf0, w0), (nm1, nf1, g1, pf1, w1) = out
        assert torch.equal(nm0, nm1) and torch.equal(nf0, nf1), it
        for a, b in zip(g0, g1):
            assert torch.allclose(a, b, rtol = 1e-5, atol = 1e-5 * float(a.abs().max())), it
        assert torch.allclose(pf0, pf1, rtol = 1e-5, atol = 1e-6 * float(pf0.abs().max())), it
        assert torch.allclose(w0, w1, rtol = 1e-5, atol = 1e-6 * float(w0.abs().max())), it
    assert deferred


@cuda_only
def test_the_choice_is_the_tuners(monkeypatch):
    """At the end of the pass, `autotune.pick` keyed by the group's shape chooses between the layers' own launches and the
    deferred ones (margin 1.0: the same results), timed once."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    asked, real = [], autotune.pick

    def pick(key, cands, bench, *a, **k):
        if key[0] == "bs_post_deferred":
            asked.append((key, list(cands), k.get("margin")))
        return real(key, cands, bench, *a, **k)
    monkeypatch.setattr(autotune, "pick", pick)
    pc, gated, phi, x = _circuit(64)
    for _ in range(3):
        _step(pc, phi, x)
    assert len(asked) == 1 and asked[0][1] == ["layers", "deferred"] and asked[0][2] == 1.0
    assert asked[0][0][1] == len(gated)


@cuda_only
def test_nothing_deferred_with_a_layer_callback(monkeypatch):
    """A backward with a layer callback (which sees the node values between layers) launches each layer's own: the
    callback after a gated layer sees its node values shifted back."""
    pc, gated, phi, x = _circuit(64)
    _step(pc, phi, x)
    seen = []
    rows = [l._layer_nid_range for g in pc.inner_layer_groups if g.is_sum() for l in g if hasattr(l, "external_node_infos")]

    def cb(layer, node_mars, **kwargs):
        if hasattr(layer, "external_node_infos"):
            a, b = layer._layer_nid_range
            seen.append(((a, b), node_mars[a:b].clone()))
    pc.init_param_flows(flows_memory = 0.0)
    pc(x, sum_external_params = phi)
    pc.backward(x, flows_memory = 1.0, sum_external_params_grad = {n: torch.zeros_like(p) for n, p in phi.items()},
                sum_layer_post_backward_callback = cb)
    assert len(seen) == len(rows)
    for (a, b), nm in seen:
        assert torch.equal(nm, pc.node_mars[a:b])


@cuda_only
def test_a_deferring_backward_replayed_from_a_graph_computes_the_eager_values(monkeypatch):
    """A backward recorded into a CUDA graph with the deferred kernels (chosen before the capture) replays its eager
    node values, gradient log-Z term and denominator flows."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    _forced(monkeypatch, "deferred")
    B = 64
    pc, gated, phi, x = _circuit(B)
    grads = {n: torch.zeros_like(p) for n, p in phi.items()}
    outs = []
    for rec in (False, True, None):
        for g in grads.values():
            g.zero_()
        pc.init_param_flows(flows_memory = 0.0)
        pc(x, sum_external_params = phi)
        kw = dict(record_cudagraph = True) if rec else {}
        pc.backward(x, flows_memory = 1.0, sum_external_params_grad = grads, **kw)
        outs.append((pc.node_mars.clone(), [grads[n].clone() for n in gated], pc.denom_param_flows.clone()))
    (nm0, g0, w0), (nm2, g2, w2) = outs[0], outs[2]
    assert torch.equal(nm0, nm2)
    for a, b in zip(g0, g2):
        assert torch.allclose(a, b, rtol = 1e-5, atol = 1e-5 * float(a.abs().max()))
    assert torch.allclose(w0, w2, rtol = 1e-5, atol = 1e-6 * float(w0.abs().max()))
