"""
BlockScale's post-backward deferred to the end of the pass (`pyjuice.parameterizations.blockscale.post_deferred`):
each gated layer of a circuit's backward leaves its `_bs_post_backward_kernel` launch to the end of the inner layers,
where the tuners choose between those launches and the deferred kernels over every layer at once
(`kernels/blockscale_post_backward_deferred.py`). The deferred kernels compute what the per-layer ones compute: on
the inputs of every layer of real backwards, the shifted node values and the gradient's log-Z term to the bit, `W` to
the bit where a layer is one batch slice, else to its float64 value as the per-layer kernel does (whose slices then add
in no fixed order); their other forms (`_forms`: `W` on a side stream at a node tile of its own, the gradient over
several gate tiles a program) the deferred kernels' bits, each timed in the backwards after the tuner's choice and the
fastest kept. A circuit's backward that defers: node values and flows to the bit of one that does not, its gradients and
denominator flows to their summation order; replayed from a CUDA graph, its eager values (a capture amid the trials
too); its side stream held back on either end, the unhurried run's.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.parameterizations import BlockScale
import pyjuice.layer.kernels.autotune as autotune
import pyjuice.parameterizations.blockscale.post_deferred as pdm

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
                             parameterization = BlockScale(ch_block_size = gcb, apply_z_correction = z))
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
    phi = {n: (torch.randn(n.parameterization.tensor_shapes(n, B)[0], generator = g) * 2).to(DEV) for n in gated}
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
    else:                                                      # ("deferred", or one of `_forms`)
        pdm._launch_deferred(members, pdm._tables(members), nm, gr, ws, None if mode == "deferred" else mode)
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
# tied or not: the deferred `W` kernel walks one layer per `W`), 4 gate tiles a node block, 3 (the wide gradient's
# programs then one tile short) and 8 (two programs a node block and batch slice; and with the plan's batch tile 32, the
# wide gradient's 16 then twice its batch slices)
CASES = [(256, 4, 32, 32, True, True, True), (256, 4, 32, 128, True, True, True), (256, 4, 32, 48, True, True, True),
         (32, 4, 4, 32, True, True, True), (256, 4, 32, 64, False, True, True), (256, 4, 32, 64, True, False, True),
         (256, 4, 32, 64, True, True, False), (256, 4, 2, 64, True, True, True), (192, 3, 2, 48, True, True, True),
         (512, 4, 2, 32, True, True, True), (1024, 4, 4, 64, True, True, True)]


@cuda_only
@pytest.mark.parametrize("L, nb, gcb, B, tied, z, grad", CASES)
def test_the_deferred_kernels_compute_the_layers_bits(L, nb, gcb, B, tied, z, grad, monkeypatch):
    """On every layer's inputs of real backwards: the shifted node values and the gradient's log-Z term to the bit of the
    layers' own launches, `W` to the bit where a layer is one batch slice, else to float64 as the layers' own; every
    other form of the deferred kernels (`_forms`, none without `tl.dot`) all three to the deferred kernels' bits."""
    pc, gated, phi, x = _circuit(B, L = L, nb = nb, gcb = gcb, tied = tied, z = z)
    owner, members = _members(pc, phi, x, monkeypatch, grad)
    assert len(members) == len(gated)
    assert (members[0]["grad_ext"] is not None) == grad and (members[0]["W"] is not None) == z
    # (every layer's own gates: one writer of each gradient row, so the stores -- no atomic adds -- in both)
    assert all(m["plan"][1]["GRAD_ATOMIC"] == 0 for m in members) and pdm._tables(members)["grad_atomic"] == 0
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
    forms, c0 = pdm._forms(members[0]), members[0]["plan"][1]
    g_tiles = -(-c0["n_gates"] // c0["TILE_G"])
    assert L != 1024 or (c0["TILE_B"] == 32 and g_tiles == 8), c0
    assert len(forms) == (0 if not c0["USE_DOT"] else int(z) + (2 if grad and g_tiles > 1 else 0)), forms
    assert all(f[0] == min(4, 1 << (g_tiles - 1).bit_length()) for f in forms if f[0])
    for form in forms:
        c = _run(owner, members, form, nm0)
        assert torch.equal(c[0], b[0]), form
        if grad:
            assert torch.equal(c[1].view(torch.int32), b[1].view(torch.int32)), form
        assert c[2].keys() == b[2].keys() and all(torch.equal(c[2][p].view(torch.int32), b[2][p].view(torch.int32))
                                                  for p in b[2]), form


def _forced(monkeypatch, choice, bench_all = False):
    """The tuners' choice at the end of the pass forced: a name, or the index of a candidate (one of `_forms`'s) -- no
    candidate benched, so no form timed in the backwards after (`_FormTrials`), unless ``bench_all``: each benched once
    (one that raises skipped, as the tuner's timing skips it). A deferred choice is cached as `pick` caches it -- a
    capture, which tunes nothing, then takes it --; the layers' own launches are not, so the layers keep deferring
    and launch them at the end of the pass. Returns the candidate lists the tuner was asked to choose from."""
    real, asked = autotune.pick, []

    def pick(key, cands, bench, *a, **k):
        if key[0] != "bs_post_deferred":
            return real(key, cands, bench, *a, **k)
        asked.append(list(cands))
        for c in cands if bench_all else ():
            try:
                bench(c)
            except RuntimeError:
                pass
        got = cands[choice] if isinstance(choice, int) else choice
        if got != "layers":
            autotune._CACHE[autotune._full_key(key)] = got
        return got
    monkeypatch.setattr(autotune, "pick", pick)
    return asked


# the choices: the deferred kernels in their own form, the layers' own launches, `_forms`'s (`W` beside the gradient; that
# with the gradient over the gate tiles, batch tiles of 16 and of 32) -- the circuit's node blocks 4 gate tiles (`GCB`)
CHOICES = ["deferred", "layers", 2, 3, 4]
CHOICE_IDS = ["deferred", "layers", "w_beside", "wide16_w_beside", "wide32_w_beside"]
GCB = 2


@cuda_only
@pytest.mark.parametrize("choice", CHOICES, ids = CHOICE_IDS)
def test_a_deferring_backward_computes_what_one_launching_per_layer_computes(choice, monkeypatch):
    """One circuit's backwards in turn launching every layer's post-backward at once (the deferral off) and deferring it
    (either choice at the end), on the same forward: node values and node flows to the bit, the gradients, the
    denominator and the parameter flows to their summation order (the element kernel adds its gradient term
    atomically)."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    B = 64
    pc, gated, phi, x = _circuit(B, gcb = GCB)
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
    deferred ones in their forms (margin 1.0: the same results), timed once."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    asked, real = [], autotune.pick
    forms, real_forms = [], pdm._forms
    monkeypatch.setattr(pdm, "_forms", lambda m0: forms.append(real_forms(m0)) or forms[-1])

    def pick(key, cands, bench, *a, **k):
        if key[0] == "bs_post_deferred":
            asked.append((key, list(cands), k.get("margin")))
        return real(key, cands, bench, *a, **k)
    monkeypatch.setattr(autotune, "pick", pick)
    monkeypatch.setattr(pdm, "_FORM_TRIALS", {})
    pc, gated, phi, x = _circuit(64, gcb = GCB)
    for _ in range(3):
        _step(pc, phi, x)
    assert len(asked) == 1 and len(forms) == 1 and len(forms[0]) == 3
    assert asked[0][1] == ["layers", "deferred"] + forms[0] and asked[0][2] == 1.0
    assert asked[0][0][1] == len(gated)
    # (a deferred one chosen, its forms are timed in the backwards that follow; the layers' own, nothing to time)
    trials = pdm._FORM_TRIALS.get(asked[0][0])
    assert (trials is None) == (autotune.cached(asked[0][0]) == "layers")
    assert trials is None or trials.forms == [None] + forms[0]


@cuda_only
def test_the_trials_keep_the_form_of_the_shortest_periods(monkeypatch):
    """The choice (`_FormTrials`, timings scripted): the tuner's form for `FORM_WARM` backwards, then each form in turn
    -- the deferred kernels' own first -- timed by its backward's period (its launch to the next timed one), the best of
    each form's after `FORM_TRIALS` each: the fastest kept where it is at least `FORM_MARGIN` faster than the deferred
    kernels' own, else their own; kept after, untimed. In a capture the tuner's form, nothing timed. The warm-up spans
    deferred_par's side-stream trials, which change a backward's period."""
    from pyjuice.layer import deferred_par as dpar
    assert pdm.FORM_WARM >= dpar.SIDE_WARM + 2 * dpar.SIDE_TRIALS
    clock = [0.0]

    class Event:
        def __init__(self, **kw):
            self.t = None

        def record(self):
            self.t = clock[0]

        def query(self):
            return True

        def elapsed_time(self, b):
            return b.t - self.t

    monkeypatch.setattr(pdm.torch.cuda, "Event", Event)
    forms = [None, (0, 32, 64), (4, 16, 64), (4, 32, 64)]

    def chosen(periods, capture_at = None):
        tr = pdm._FormTrials(forms)
        for i in range(pdm.FORM_WARM):
            assert tr.next("pick") == "pick" and not tr.marks
            clock[0] += 50.0                                        # (untimed: any period)
        for k in range(len(forms) * pdm.FORM_TRIALS):
            if k == capture_at:
                with monkeypatch.context() as m:
                    m.setattr(pdm.torch.cuda, "is_current_stream_capturing", lambda: True)
                    assert tr.next("pick") == "pick" and len(tr.marks) == k
            f = tr.next("pick")
            assert f == forms[k % len(forms)] and tr.chosen is None, (k, f)
            clock[0] += periods[k % len(forms)]                     # (this backward's period)
        got = tr.next("pick")
        assert tr.chosen == (got,) and tr.next("pick") == got and tr.marks is None
        return got

    assert chosen([10.0, 9.9, 9.0, 9.5]) == forms[2]
    assert chosen([10.0, 9.0, 9.0, 9.5]) == forms[1]                # (a tie: the first)
    assert chosen([9.0, 10.0, 9.5, 9.9]) is None
    assert chosen([10.0, 10.0 * (1 - pdm.FORM_MARGIN) + 0.01, 11.0, 11.0]) is None
    assert chosen([10.0, 10.0 * (1 - pdm.FORM_MARGIN) - 0.01, 11.0, 11.0]) == forms[1]
    assert chosen([10.0, 11.0, 9.0, 12.0], capture_at = 5) == forms[2]


@cuda_only
def test_the_backwards_after_the_choice_time_each_form_and_keep_the_fastest(monkeypatch):
    """A deferred form chosen, the backwards that follow run in the tuner's choice for `FORM_WARM`, then in each form
    the tuner's bench launched -- the deferred kernels' own first --, `FORM_TRIALS` times each in turn, timed by their
    periods (one more in turn while the last period's end has not run yet: a GPU behind its host); then keep the
    fastest: here the form every other one is held back against (`torch.cuda._sleep` before it)."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    monkeypatch.setattr(pdm, "_FORM_TRIALS", {})
    pc, gated, phi, x = _circuit(64, gcb = GCB)
    real, used, forms = pdm._launch_deferred, [], []

    def launch(members, tabs, node_mars, grad_ext, ws, form = None):
        if not forms:
            forms.extend(pdm._forms(members[0]))
        if form != forms[-1]:
            torch.cuda._sleep(int(2e6))
        if node_mars is members[0]["node_mars"]:                   # (the backward's own launch, not the tuner's)
            used.append(form)
        return real(members, tabs, node_mars, grad_ext, ws, form)
    monkeypatch.setattr(pdm, "_launch_deferred", launch)
    _forced(monkeypatch, 2, bench_all = True)                     # (every candidate benched: each form launched)
    for _ in range(pdm.FORM_WARM + 3 * (1 + 3) + 8):
        _step(pc, phi, x)
        torch.cuda.synchronize()                                   # (each backward's events read by the next)
    assert len(forms) == 3
    n_trials = (len(forms) + 1) * pdm.FORM_TRIALS
    warm, trial = used[:pdm.FORM_WARM], used[pdm.FORM_WARM:pdm.FORM_WARM + n_trials]
    assert warm == [forms[0]] * pdm.FORM_WARM, used                # (the tuner's: candidate 2, the first form)
    assert trial == ([None] + forms) * pdm.FORM_TRIALS, used
    (trials,) = pdm._FORM_TRIALS.values()
    assert trials.chosen == (forms[-1],) and used[-4:] == [forms[-1]] * 4, used


@cuda_only
def test_a_form_that_did_not_launch_in_the_bench_is_not_timed_after(monkeypatch):
    """A form the tuner's bench could not launch (a compile out of shared memory, say) is left out of the forms timed in
    the backwards that follow: none of them raises, the others are timed."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    monkeypatch.setattr(pdm, "_FORM_TRIALS", {})
    pc, gated, phi, x = _circuit(64, gcb = GCB)
    real, bad, used = pdm._launch_deferred, [], []

    def launch(members, tabs, node_mars, grad_ext, ws, form = None):
        if not bad:
            bad.append(pdm._forms(members[0])[1])
        if form == bad[0]:
            raise RuntimeError("out of resource: shared memory")
        if node_mars is members[0]["node_mars"]:
            used.append(form)
        return real(members, tabs, node_mars, grad_ext, ws, form)
    monkeypatch.setattr(pdm, "_launch_deferred", launch)
    _forced(monkeypatch, 2, bench_all = True)
    for _ in range(pdm.FORM_WARM + 3 * pdm.FORM_TRIALS + 6):
        _step(pc, phi, x)
        torch.cuda.synchronize()
    (trials,) = pdm._FORM_TRIALS.values()
    assert bad[0] not in trials.forms and len(trials.forms) == 3 and trials.chosen is not None
    assert set(used) == set(trials.forms)


@cuda_only
def test_nothing_deferred_with_a_layer_callback(monkeypatch):
    """A backward with a layer callback (which sees the node values between layers) launches each layer's own: the
    callback after a gated layer sees its node values shifted back (the deferred kernels the choice, so a deferral
    would show)."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    _forced(monkeypatch, "deferred")
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
@pytest.mark.parametrize("choice", ["deferred", 2, 3, 4, "trials"], ids = ["deferred"] + CHOICE_IDS[2:] + ["trials"])
def test_a_deferring_backward_replayed_from_a_graph_computes_the_eager_values(choice, monkeypatch):
    """A backward recorded into a CUDA graph with the deferred kernels (chosen before the capture), in their own form or
    `_forms`'s (`W` on a side stream: a fork and a join in the graph), replays its eager node values, gradient log-Z
    term and denominator flows -- also captured while the forms are timed (`_FormTrials`: the tuner's choice in the
    capture, nothing timed there), the eager backward before it in a timed form."""
    monkeypatch.setattr(autotune, "_CACHE", {})
    monkeypatch.setattr(pdm, "_FORM_TRIALS", {})
    used, real = [], pdm._launch_deferred

    def launch(members, tabs, node_mars, grad_ext, ws, form = None):
        if node_mars is members[0]["node_mars"]:                   # (the backward's own launch, not the tuner's)
            used.append(form)
        return real(members, tabs, node_mars, grad_ext, ws, form)
    monkeypatch.setattr(pdm, "_launch_deferred", launch)
    if choice == "trials":
        monkeypatch.setattr(pdm, "FORM_WARM", 0)
        asked = _forced(monkeypatch, 4, bench_all = True)
    else:
        asked = _forced(monkeypatch, choice)
    B = 64
    pc, gated, phi, x = _circuit(B, gcb = GCB)
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
    # (recording runs the pass eagerly, then captures it: the deferred kernels in the chosen form in the graph; amid the
    # trials, the first backward in the first form timed -- the deferred kernels' own --, the recording's eager run in
    # the next, its capture in the tuner's)
    if choice == "trials":
        (trials,) = pdm._FORM_TRIALS.values()
        assert used[:3] == [None, trials.forms[1], trials.forms[-1]], used
    else:
        assert used[:3] == [None if choice == "deferred" else asked[0][choice]] * 3, used


@cuda_only
def test_the_side_streams_W_waits_for_the_pass_and_the_pass_for_it(monkeypatch):
    """`_forms`'s `W` runs on a side stream, forked before the gradient's launch and joined after both. Held back on
    either end (`torch.cuda._sleep`) -- the main stream just before the fork, the node flows `W` reads written there
    last (poisoned before the sleep, put back after it); the side stream before `W` -- the backward's denominator flows,
    read on the main stream with nothing synchronized, are the unhurried run's to the bit."""
    import pyjuice.parameterizations.kernels.blockscale_post_backward_deferred as kd
    monkeypatch.setattr(autotune, "_CACHE", {})
    _forced(monkeypatch, 2)
    pc, gated, phi, x = _circuit(64)
    _step(pc, phi, x)
    _step(pc, phi, x)
    ref = pc.denom_param_flows.clone()
    torch.cuda.synchronize()
    assert ref.abs().max() > 0
    sleep = int(5e7)                                           # (~25 ms)

    real_launch = pdm._launch_deferred

    def main_late(members, tabs, node_mars, grad_ext, ws, form = None):
        nf = members[0]["node_flows"]
        keep = nf.clone()
        nf.fill_(float("nan"))
        torch.cuda._sleep(sleep)
        nf.copy_(keep)
        return real_launch(members, tabs, node_mars, grad_ext, ws, form)

    class SideLate:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                torch.cuda._sleep(sleep)                       # (on the stream `W` is launched on)
                return self.kernel[grid](*args, **kwargs)
            return launch

    for late in ("main", "side"):
        with monkeypatch.context() as m:
            if late == "main":
                m.setattr(pdm, "_launch_deferred", main_late)
            else:
                m.setattr(kd, "_bs_post_deferred_w_kernel", SideLate(kd._bs_post_deferred_w_kernel))
            _step(pc, phi, x)
            got = pc.denom_param_flows.clone()
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), late
