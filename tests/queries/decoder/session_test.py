"""
Decoder sessions (`juice.decoder`) against the queries. Every sequence of calls is answered by `juice.queries` first and
then by one session -- never interleaved, since any other pass of the circuit makes a session start over (counted:
each sequence must run on one start). Without early stop, marginals are the queries', to the bit, and conditionals
within 1e-6: the backward kernels accumulate with atomics (the input layers' conditionals, the external-parameter
backwards), so a conditional varies from run to run by a few 1e-8, the queries' own included. With early stop, both
to float rounding. Circuits: an HMM, an HCLT and a PD over Categorical inputs (hard data, and soft data for the HMM); a chain
over SoftEvidenceCategorical with its evidence held by the session (dense, and top-k); chains whose transitions carry
BlockScale and LowRank external parameters. Sequences: left to right, diffusion-like (several positions per step,
different per sample), and random (values changed, observed, marginalized again), marginals and conditionals mixed.
Besides: other passes of the circuit between a session's calls, a batch size that changes, repeated conditionals
through external-parameter layers (whose backward shifts `node_mars` in place), the layers a left-to-right step
launches, and the refusals.
"""

import numpy as np
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.queries.decoder.session as session
from pyjuice.layer import InputLayer, SumLayer
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams, LowRankSumParams


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEV = torch.device("cuda:0")
C = 5


def _chain(dist_fn, V, num_latents, block_size, external = None):
    with juice.set_block_size(block_size):
        chain = inputs(0, num_node_blocks = num_latents // block_size, dist = dist_fn())
        transitions = []
        for t in range(1, V):
            emit = inputs(t, num_node_blocks = num_latents // block_size, dist = dist_fn())
            ext = dict()
            if external == "blockscale":
                ext = dict(external_params = BlockScaleSumParams(ch_block_size = block_size))
            elif external == "lowrank":
                ext = dict(external_params = LowRankSumParams(rank = 4))
            chain = summate(multiply(chain, emit), num_node_blocks = num_latents // block_size, **ext)
            transitions.append(chain)
        root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False).to(DEV), transitions


def _model(kind):
    """(pc, number of variables, batch size, the per-sample inputs, soft data or not)."""
    torch.manual_seed(0)
    precision = "fp32" if kind.endswith("_fp32") else "auto"     # fp32: where marginals take cuts
    kind = kind.replace("_fp32", "")
    if kind in ("hmm", "hmm_soft"):
        pc = juice.compile(juice.structures.HMM(seq_length = 8, num_latents = 16, num_emits = C), verbose = False,
                           precision = precision)
        return pc.to(DEV), 8, 4, {}, kind == "hmm_soft"
    if kind == "hclt":
        x = torch.randint(0, C, (512, 8))
        ns = juice.structures.HCLT(x.float(), num_latents = 8, input_dist = dists.Categorical(num_cats = C))
        ns.init_parameters(perturbation = 2.0)
        return juice.compile(ns, verbose = False, precision = precision).to(DEV), 8, 4, {}, False
    if kind == "pd":
        ns = juice.structures.PD(data_shape = (4, 4), num_latents = 4, split_intervals = 2,
                                 input_dist = dists.Categorical(num_cats = C))
        ns.init_parameters(perturbation = 2.0)
        return juice.compile(ns, verbose = False, precision = precision).to(DEV), 16, 4, {}, False
    if kind in ("softevi", "softevi_topk"):
        pc, _ = _chain(lambda: dists.SoftEvidenceCategorical(num_cats = C), 6, 8, 4)
        g = torch.Generator(device = DEV).manual_seed(1)
        if kind == "softevi":
            E = torch.randn(4, 6, C, device = DEV, generator = g) * 2
            return pc, 6, 4, dict(categorical_evidence_logp = E), False
        ids = torch.argsort(torch.rand(4, 6, C, device = DEV, generator = g), dim = 2)[..., :3]
        E = torch.randn(4, 6, 3, device = DEV, generator = g) * 2
        return pc, 6, 4, dict(categorical_evidence_logp = E, soft_evidence_cat_ids = ids), False
    pc, transitions = _chain(lambda: dists.Categorical(num_cats = C), 5, 32, 16, external = kind)
    g = torch.Generator(device = DEV).manual_seed(2)
    params = dict()
    for ns in transitions:
        shapes = ns.external_params.tensor_shapes(ns, 16)
        tensors = tuple(torch.randn(s, device = DEV, generator = g) * 0.5 - 0.5 for s in shapes)
        params[ns] = tensors if len(tensors) > 1 else tensors[0]
    return pc, 5, 16, dict(sum_external_params = params), False


def _data(B, V, soft, rng, inputs):
    """Tokens every position's evidence weighs (a candidate under top-k evidence), or soft data."""
    if soft:
        d = torch.from_numpy(rng.random((B, V, C))).float().to(DEV)
        return d / d.sum(dim = 2, keepdim = True)
    if "soft_evidence_cat_ids" in inputs:
        return inputs["soft_evidence_cat_ids"][..., 0].clone()
    return torch.from_numpy(rng.integers(0, C, (B, V))).to(DEV)


def _steps(sequence, B, V, soft, inputs, rng):
    """[(data, missing [B, V], query, target_vars)]."""
    x = _data(B, V, soft, rng, inputs)
    miss = np.ones((B, V), dtype = bool)
    steps = []

    def add(query, targets = None):
        steps.append((x.clone(), torch.from_numpy(miss.copy()).to(DEV), query, targets))

    if sequence == "left_to_right":
        for t in range(V):
            add("conditional", [t])
            add("marginal")
            miss[:, t] = False
        add("marginal")
    elif sequence == "diffusion":
        while miss.any():
            targets = np.flatnonzero(miss.all(axis = 0)).tolist()
            if targets:
                add("conditional", targets)
            add("marginal")
            for b in range(B):
                masked = np.flatnonzero(miss[b])
                miss[b, rng.choice(masked, size = min(2, masked.size), replace = False)] = False
        add("marginal")
    elif sequence == "targets_in_turn":
        # One evidence, the conditionals of the variables it leaves missing one after another -- nothing changes
        # between those calls, so seeds hold valid insides -- and a marginal; then one of them observed, and again
        miss[:] = rng.random((B, V)) < 0.3
        miss[:, rng.choice(V, size = max(2, V // 2), replace = False)] = True
        for _ in range(2):
            free = np.flatnonzero(miss.all(axis = 0))
            for t in free.tolist():
                add("conditional", [t])
            add("marginal")
            miss[:, int(rng.choice(free))] = False
    else:
        miss[:] = rng.random((B, V)) < 0.5
        for _ in range(20):
            for _ in range(rng.integers(0, 3)):
                v = int(rng.integers(0, V))
                op = rng.integers(0, 3)
                if op == 0 and not soft and "soft_evidence_cat_ids" not in inputs:
                    x[:, v] = torch.from_numpy(rng.integers(0, C, B)).to(DEV)
                elif op == 0 and soft:
                    x[:, v] = _data(B, 1, True, rng, inputs)[:, 0]
                elif op == 1:
                    miss[int(rng.integers(0, B)), v] = True
                else:
                    miss[:, v] = False
            if rng.random() < 0.5:
                add("marginal")
            else:
                targets = sorted(rng.choice(V, size = int(rng.integers(1, 3)), replace = False).tolist())
                keep = miss.copy()
                miss[:, targets] = True
                add("conditional", targets)
                miss[:] = keep
    return steps


def _queries(pc, steps, inputs):
    out = []
    for x, miss, query, targets in steps:
        if query == "marginal":
            out.append(juice.queries.marginal(pc, data = x.clone(), missing_mask = miss, **inputs).detach())
        else:
            out.append(juice.queries.conditional(pc, data = x.clone(), missing_mask = miss, target_vars = targets,
                                                 **inputs))
    return out


def _session(pc, steps, inputs, early_stop, monkeypatch):
    """The session's answers, and how many times it started (over)."""
    starts = []
    reset = session.Planner.reset
    monkeypatch.setattr(session.Planner, "reset", lambda self: starts.append(1) or reset(self))
    out = []
    with juice.decoder(pc, early_stop = early_stop, **inputs) as decoder:
        for x, miss, query, targets in steps:
            if query == "marginal":
                out.append(decoder.marginal(x, missing_mask = miss))
            else:
                out.append(decoder.conditional(x, missing_mask = miss, target_vars = targets))
    return out, len(starts)


MODELS = ["hmm", "hmm_fp32", "hmm_soft", "hclt", "hclt_fp32", "pd", "pd_fp32", "softevi", "softevi_topk",
          "blockscale", "lowrank"]


def _restrict_always(monkeypatch):
    """Input runs and restricted flow fills at any size and coverage (the test circuits are below the sizes they start
    at, and their calls often fill most of a buffer)."""
    monkeypatch.setattr(session, "_RUNS_MIN_VALUES", 0)
    monkeypatch.setattr(session, "_FILL_MIN_VALUES", 0)
    monkeypatch.setattr(session, "_FILL_WHOLE_FRACTION", 2.0)        # never the whole buffer


@pytest.fixture(params = ["default", "always"])
def restrict(request, monkeypatch):
    """The session's input runs and restricted fills at their default sizes -- whole layers and whole fills here -- or
    always."""
    if request.param == "always":
        _restrict_always(monkeypatch)
    return request.param


@cuda_only
@pytest.mark.parametrize("sequence", ["left_to_right", "diffusion", "random", "targets_in_turn"])
@pytest.mark.parametrize("kind", MODELS)
@pytest.mark.parametrize("early_stop", [False, True])
def test_a_session_answers_as_the_queries(kind, sequence, early_stop, restrict, monkeypatch):
    pc, V, B, inputs, soft = _model(kind)
    steps = _steps(sequence, B, V, soft, inputs, np.random.default_rng(3))
    want = _queries(pc, steps, inputs)
    got, starts = _session(pc, steps, inputs, early_stop, monkeypatch)
    assert starts == 1
    for i, ((x, miss, query, targets), g, w) in enumerate(zip(steps, got, want)):
        if not early_stop:
            if query == "marginal":
                assert torch.equal(g, w), (i, float((g - w).abs().max()))
            else:
                assert torch.allclose(g, w, atol = 1e-6, rtol = 0.0), (i, float((g - w).abs().max()))
        elif query == "marginal":
            assert torch.allclose(g, w, atol = 1e-4, rtol = 1e-5), (i, float((g - w).abs().max()))
        else:
            assert torch.allclose(g, w, atol = 1e-5), (i, float((g - w).abs().max()))


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "hmm_fp32", "pd", "pd_fp32", "blockscale"])
def test_marginals_take_cuts_only_where_outsides_are_exact(kind, monkeypatch):
    """Left to right with early stop: marginals sum over cuts on a circuit compiled at fp32 without external
    parameters -- after the first steps, every one -- and never otherwise."""
    pc, V, B, inputs, soft = _model(kind)
    steps = _steps("left_to_right", B, V, soft, inputs, np.random.default_rng(6))
    plans = []
    marginal = session.Planner.marginal
    monkeypatch.setattr(session.Planner, "marginal",
                        lambda self, cuts = True: plans.append(marginal(self, cuts = cuts)) or plans[-1])
    _session(pc, steps, inputs, True, monkeypatch)
    taken = [plan.cut.size > 0 for plan in plans]
    if kind.endswith("_fp32"):
        assert all(taken[2:-1]), taken
    else:
        assert not any(taken), taken


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "blockscale"])
def test_other_passes_between_calls(kind):
    """A `juice.queries` call -- a forward and a backward of the circuit, into the buffers the session keeps its values
    in -- between every two of the session's calls: the session starts over, and answers right."""
    pc, V, B, inputs, soft = _model(kind)
    steps = _steps("left_to_right", B, V, soft, inputs, np.random.default_rng(4))
    with juice.decoder(pc, early_stop = True, **inputs) as decoder:
        for x, miss, query, targets in steps:
            if query == "marginal":
                got = decoder.marginal(x, missing_mask = miss)
                want = juice.queries.marginal(pc, data = x, missing_mask = miss, **inputs).detach()
                assert torch.allclose(got, want, atol = 1e-4, rtol = 1e-5)
            else:
                got = decoder.conditional(x, missing_mask = miss, target_vars = targets)
                want = juice.queries.conditional(pc, data = x, missing_mask = miss, target_vars = targets, **inputs)
                assert torch.allclose(got, want, atol = 1e-5)


@cuda_only
def test_a_batch_size_that_changes(monkeypatch):
    pc, V, B, inputs, soft = _model("hmm")
    rng = np.random.default_rng(5)
    steps = _steps("left_to_right", 4, V, False, {}, rng)[:6] + _steps("left_to_right", 6, V, False, {}, rng)[:6]
    want = _queries(pc, steps, {})
    got, starts = _session(pc, steps, {}, False, monkeypatch)
    assert starts == 2
    for (x, miss, query, targets), g, w in zip(steps, got, want):
        if query == "marginal":
            assert torch.equal(g, w)
        else:
            assert torch.allclose(g, w, atol = 1e-6, rtol = 0.0)


@cuda_only
def test_repeated_conditionals_through_external_parameter_layers():
    """BlockScale's backward adds log Z to its layer's `node_mars` and subtracts it again; the session puts the rows
    back bit for bit, so after 30 conditionals of unchanged evidence -- each a backward through every gated layer --
    the marginal, read from the cached insides, is still the queries' to the bit."""
    pc, V, B, inputs, soft = _model("blockscale")
    x = torch.randint(0, C, (B, V), device = DEV)
    miss = torch.zeros(B, V, dtype = torch.bool, device = DEV)
    miss[:, 2] = True
    want_m = juice.queries.marginal(pc, data = x, missing_mask = miss, **inputs).detach()
    with juice.decoder(pc, early_stop = False, **inputs) as decoder:
        first = decoder.conditional(x, missing_mask = miss, target_vars = [2])
        for _ in range(30):
            # the backward's atomics vary its answer by a few 1e-8; drift would grow with the calls
            assert torch.allclose(decoder.conditional(x, missing_mask = miss, target_vars = [2]), first,
                                  atol = 1e-6, rtol = 0.0)
        assert torch.equal(decoder.marginal(x, missing_mask = miss), want_m)        # the cached insides, exactly


@cuda_only
def test_a_left_to_right_step_launches_a_few_layers(monkeypatch):
    """Left to right over an HMM of 24 steps (at fp32, where marginals take cuts): past the first steps, a
    conditional runs at most two sum layers' backwards and two sum layers' forwards (the newly observed variable's
    emission into its transition), the input layer's forward over the one input group of that variable, a pass that
    fills a fraction of the flows, and the marginal after it none."""
    _restrict_always(monkeypatch)
    torch.manual_seed(0)
    V = 24
    pc = juice.compile(juice.structures.HMM(seq_length = V, num_latents = 16, num_emits = C), verbose = False,
                       precision = "fp32").to(DEV)
    counts = {"forward": 0, "backward": 0}
    fw, bw = SumLayer.forward, SumLayer.backward

    def count(name, fn):
        def run(self, *args, **kwargs):
            counts[name] += 1
            return fn(self, *args, **kwargs)
        return run

    monkeypatch.setattr(SumLayer, "forward", count("forward", fw))
    monkeypatch.setattr(SumLayer, "backward", count("backward", bw))
    input_runs = []
    forward_runs = InputLayer._forward_runs
    monkeypatch.setattr(InputLayer, "_forward_runs",
                        lambda self, data, node_mars, node_runs, **kw: input_runs.append(list(node_runs)) or
                        forward_runs(self, data, node_mars, node_runs, **kw))
    x = torch.randint(0, C, (2, V), device = DEV)
    with juice.decoder(pc) as decoder:
        for t in range(V):
            miss = torch.zeros(2, V, dtype = torch.bool, device = DEV)
            miss[:, t:] = True
            before = dict(counts)
            n_runs = len(input_runs)
            if t >= 2:
                pc.node_flows.fill_(float("nan"))
            decoder.conditional(x, missing_mask = miss, target_vars = [t])
            step = {k: counts[k] - before[k] for k in counts}
            if t >= 2:
                assert step["backward"] <= 2 and step["forward"] <= 2, (t, step)
                assert len(input_runs) == n_runs + 1 and len(input_runs[-1]) == 1, (t, input_runs[n_runs:])
                assert input_runs[-1][0][1] - input_runs[-1][0][0] == 16                 # one group, 16 latents
                untouched = torch.isnan(pc.node_flows).all(dim = 1).float().mean()
                assert untouched > 0.5, (t, float(untouched))
            before = dict(counts)
            decoder.marginal(x, missing_mask = miss)
            if t >= 2:
                assert counts == before, (t, {k: counts[k] - before[k] for k in counts})


@cuda_only
@pytest.mark.parametrize("sequence", ["left_to_right", "diffusion", "random", "targets_in_turn"])
@pytest.mark.parametrize("kind", ["hmm", "hmm_soft", "hclt", "pd_fp32", "softevi", "blockscale", "lowrank"])
@pytest.mark.parametrize("early_stop", [False, True])
def test_a_session_reads_only_the_flows_it_fills(kind, sequence, early_stop, monkeypatch):
    """A conditional's pass fills (-inf) only the flows it reads before writing them: with both flow buffers poisoned
    with NaN before every call, the session still answers as the queries."""
    _restrict_always(monkeypatch)
    pc, V, B, inputs, soft = _model(kind)
    steps = _steps(sequence, B, V, soft, inputs, np.random.default_rng(9))
    want = _queries(pc, steps, inputs)
    with juice.decoder(pc, early_stop = early_stop, **inputs) as decoder:
        for i, ((x, miss, query, targets), w) in enumerate(zip(steps, want)):
            pc.node_flows.fill_(float("nan"))
            pc.element_flows.fill_(float("nan"))
            if query == "marginal":
                g = decoder.marginal(x, missing_mask = miss)
            else:
                g = decoder.conditional(x, missing_mask = miss, target_vars = targets)
            if query == "marginal" and not early_stop:
                assert torch.equal(g, w), (i, float((g - w).abs().max()))
            else:
                assert torch.allclose(g, w, atol = 1e-4 if query == "marginal" else 1e-5, rtol = 1e-5), \
                    (i, query, float((g - w).abs().max()))


@cuda_only
@pytest.mark.parametrize("sequence", ["left_to_right", "diffusion", "random", "targets_in_turn"])
@pytest.mark.parametrize("kind", ["hmm", "hclt", "pd_fp32"])
@pytest.mark.parametrize("early_stop", [False, True])
def test_the_insides_a_session_holds_valid_are_a_forwards(kind, sequence, early_stop, restrict):
    """After every call, every inside the planner holds valid is, to the bit, what a forward of the call's evidence
    computes -- a twin circuit's (compiled the same, parameters the same) through `juice.queries.marginal`: the
    rows a pass seeds or an external-parameter layer shifts are put back, input runs write what the whole layer
    would, and nothing stale is held valid."""
    pc, V, B, inputs, soft = _model(kind)
    twin = _model(kind)[0]
    steps = _steps(sequence, B, V, soft, inputs, np.random.default_rng(8))
    with juice.decoder(pc, early_stop = early_stop, **inputs) as decoder:
        for i, (x, miss, query, targets) in enumerate(steps):
            if query == "marginal":
                decoder.marginal(x, missing_mask = miss)
            else:
                decoder.conditional(x, missing_mask = miss, target_vars = targets)
            planner = decoder._planner
            valid = np.flatnonzero(planner.in_ok)
            if valid.size == 0:
                continue
            rows = torch.cat([torch.arange(*planner.structure.groups[g]._output_ind_range) for g in valid]).to(DEV)
            juice.queries.marginal(twin, data = x.clone(), missing_mask = miss, **inputs)
            assert torch.equal(pc.node_mars[rows], twin.node_mars[rows]), \
                (i, query, float((pc.node_mars[rows] - twin.node_mars[rows]).abs().max()))


@cuda_only
def test_a_conditional_after_evidence_of_low_probability():
    """With early stop a conditional's pass is scaled by the last call's log p: after nearly every variable of a
    64-category chain is observed at once, its flows at the target are ~e^-40 of a full backward's -- far below the
    input layer's `1e-12` guard -- until normalized over the target's nodes. The session answers as the queries."""
    torch.manual_seed(0)
    V, C64, B = 12, 64, 4
    pc = juice.compile(juice.structures.HMM(seq_length = V, num_latents = 16, num_emits = C64), verbose = False,
                       precision = "fp32").to(DEV)
    x = torch.randint(0, C64, (B, V), device = DEV)
    everything = torch.ones(B, V, dtype = torch.bool, device = DEV)
    most = torch.zeros(B, V, dtype = torch.bool, device = DEV)
    most[:, [3, 7]] = True
    with juice.decoder(pc) as decoder:
        decoder.conditional(x, missing_mask = everything, target_vars = [3])
        got = decoder.conditional(x, missing_mask = most, target_vars = [3])
    want = juice.queries.conditional(pc, data = x, missing_mask = most, target_vars = [3])
    assert juice.queries.marginal(pc, data = x, missing_mask = most).max() < -28        # below the guard: e^-28 = 7e-13
    assert torch.allclose(got, want, atol = 1e-5), float((got - want).abs().max())


@cuda_only
@pytest.mark.parametrize("expand", ["host", "device"])
def test_row_sets(expand, monkeypatch):
    """`_Rows`: a set's rows in the order of its groups, ranges that continue one another merged -- one run a slice,
    more an index tensor (expanded on the host, or past `_HOST_ROWS` rows on the device) -- with every row's segment
    where asked; an empty set absent."""
    from types import SimpleNamespace
    if expand == "device":
        monkeypatch.setattr(session, "_HOST_ROWS", 0)
    lo = np.array([0, 4, 10, 12, 30, 40, 48, 60])
    hi = np.array([4, 8, 12, 20, 31, 48, 50, 64])
    rows = session._Rows(SimpleNamespace(lo = lo, hi = hi))
    rows.add("one_run", np.array([0, 1]))
    rows.add("two_runs", np.array([2, 3, 0]))
    rows.add("merged_and_single", np.array([5, 6, 4]))
    rows.add("not_continuing", np.array([1, 0]))
    rows.add("segments", np.array([0, 2, 5]), segments = np.array([0, 1, 1]))
    rows.add("empty", np.array([], dtype = np.int64))
    rows.upload(DEV)
    assert rows.get("one_run") == slice(0, 8)
    want = {"two_runs": [*range(10, 20), *range(0, 4)], "merged_and_single": [*range(40, 50), 30],
            "not_continuing": [*range(4, 8), *range(0, 4)]}
    for name, w in want.items():
        assert rows.get(name).tolist() == w, name
    r, seg = rows.get("segments")
    assert r.tolist() == [*range(0, 4), *range(10, 12), *range(40, 48)]
    assert seg.tolist() == [0] * 4 + [1] * 2 + [1] * 8
    assert rows.get("empty") is None


@cuda_only
def test_refusals():
    pc, V, B, inputs, soft = _model("hmm")
    with pytest.raises(TypeError, match = "per-sample input"):
        juice.decoder(pc, categorical_evidence_logp = torch.zeros(1))
    decoder = juice.decoder(pc)
    x = torch.zeros(B, V, dtype = torch.long, device = DEV)
    with pytest.raises(RuntimeError, match = "with"):
        decoder.marginal(x)
    with decoder:
        with pytest.raises(RuntimeError, match = "already has an open decoder"):
            with juice.decoder(pc):
                pass
    with pytest.raises(RuntimeError, match = "opened once"):
        with decoder:
            pass
