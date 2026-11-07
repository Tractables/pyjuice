"""
The deferred parameter flows' capture in two chained launches (layer/kernels/c/par_deferred_fp16_capture.cu,
`capture_steps`) against the captures it stands in for -- par_deferred_fp16.cu's `capture` (the element values from
`element_mars`) and par_deferred_fp16_fused.cu's `capture_fused` (from the products' two children): every operand
value, row scale and column maximum to the bit, over the slot layouts it takes (1, 2, 4, 8 and 16 columns a lane, a
batch past the last full lane, two slots a step), node blocks that share their edges' children (one warp an edge's rows
of every block: 1, 2 and 4 at a time) and blocks that do not, with gates (missing ones too) and without, nodes and
columns at -inf, dummy element rows; two steps into one group's maxima; two rows a warp at 16 columns a lane, repeated
(the maxima read only after the wait). Where it does not apply (`fits`): the other widths. And `deferred_par`'s side
stream: the same flows from captures there (eager -- the side stream held back, the slots poisoned -- and replayed from
a CUDA graph), captures that wait for the main stream (it held back, the buffers another batch's), joins as a backward
that raised closes and before a gated layer below shifts its node values, the place chosen from the backwards'
periods, and the captures that never go there (from `element_mars`). Skipped where the CUDA kernels cannot build.
"""
import pytest
import torch

DEV = torch.device("cuda:0")
NEG = -float("inf")


def _mods():
    if not torch.cuda.is_available():
        return None
    import pyjuice.layer.kernels.c as ck
    try:
        mods = (ck.get_par_deferred_module(), ck.get_par_deferred_fused_module(), ck.get_par_deferred_capture_module())
    except Exception:
        return None
    return mods if all(m is not None for m in mods) else None


needs_kernels = pytest.mark.skipif(_mods() is None, reason = "needs the deferred parameter-flow kernels (nvcc + CUTLASS)")


def _layout(batch):
    """`deferred_par._Group`'s slot layout at this batch: slots a step, a slot's columns."""
    bp = (batch + 31) // 32 * 32
    cps = (bp + 511) // 512
    return cps, (bp // 32 + cps - 1) // cps * 32


def _case(nbk, blk, ne, batch, gated, fused, shared, seed):
    g = torch.Generator(device = DEV).manual_seed(seed)
    cps, W = _layout(batch)
    S = 3 * cps
    rows = nbk * blk + 2 * ne + 64                               # this layer's nodes, then the children's
    nm = torch.randn(rows, batch, device = DEV, generator = g) * 3
    nm[torch.rand(rows, batch, device = DEV, generator = g) < 0.05] = NEG
    nm[: blk, 3] = NEG                                           # a column of node block 0 at -inf (M 0)
    nm[blk + 5] = NEG                                            # a whole node at -inf
    nf = torch.randn(rows, batch, device = DEV, generator = g) * 3
    nids = torch.arange(nbk, device = DEV) * blk
    # edges: element rows (dummy rows below `first_row`), their two children among the node rows past the layer's
    first_row, ne_rows = 16, ne + 16
    if shared:
        cid = torch.randint(0, ne_rows, (ne,), device = DEV, generator = g)
        cids = cid.view(1, -1).repeat(nbk, 1)
    else:
        cids = torch.randint(0, ne_rows, (nbk, ne), device = DEV, generator = g)
    ctab = torch.randint(nbk * blk, rows, (ne_rows - first_row, 2), device = DEV, generator = g).int()
    emars = torch.randn(ne_rows, batch, device = DEV, generator = g) * 2
    emars[torch.rand(ne_rows, batch, device = DEV, generator = g) < 0.05] = NEG
    node_cbs, gate_cbs = 64, 32
    gate_stride = ne // node_cbs
    gate = torch.randint(0, 10, (nbk, gate_stride), device = DEV, generator = g) * 2
    gate[torch.rand(nbk, gate_stride, device = DEV, generator = g) < 0.2] = -1
    ext = torch.randn(32, batch, device = DEV, generator = g)
    ext_base = 3
    bufs = lambda: dict(A = torch.zeros(nbk, blk, S * W, device = DEV, dtype = torch.half),
                        Bop = torch.zeros(nbk, ne, S * W, device = DEV, dtype = torch.half),
                        fA = torch.ones(nbk, S, blk, device = DEV), fB = torch.ones(nbk, S, ne, device = DEV))
    return dict(cps = cps, W = W, S = S, nm = nm, nf = nf, nids = nids, cids = cids.contiguous(), ctab = ctab,
                first_row = first_row, emars = emars, gate = gate if gated else nids, ext = ext if gated else nm,
                node_cbs = node_cbs, gate_cbs = gate_cbs, ext_base = ext_base, gated = gated, fused = fused,
                shared = shared, batch = batch, bufs = bufs)


def _old(c, b, M, slot0):
    pd, pdf, _ = _mods()
    if c["fused"]:
        pdf.capture_fused(b["A"], b["Bop"], b["fA"], b["fB"], M, c["nf"], c["nm"], c["ctab"], c["first_row"], c["ext"],
                          c["nids"], c["cids"], c["gate"], c["batch"], slot0, c["node_cbs"], c["gate_cbs"],
                          c["ext_base"], c["gated"])
    else:
        pd.capture(b["A"], b["Bop"], b["fA"], b["fB"], M, c["nf"], c["nm"], c["emars"], c["ext"], c["nids"], c["cids"],
                   c["gate"], c["batch"], slot0, c["node_cbs"], c["gate_cbs"], c["ext_base"], c["gated"])


def _new(c, b, Ms, slot0):
    _, _, cap = _mods()
    cap.capture_steps(b["A"], b["Bop"], b["fA"], b["fB"], Ms, c["nf"], c["nm"], c["nm"] if c["fused"] else c["emars"],
                      c["ctab"] if c["fused"] else Ms, c["first_row"] if c["fused"] else 0, c["ext"], c["nids"],
                      c["cids"], c["gate"], c["batch"], slot0, c["node_cbs"], c["gate_cbs"], c["ext_base"], c["gated"],
                      c["fused"], c["shared"], 0)


def _bits(x):
    return x.view(torch.int16 if x.dtype == torch.half else torch.int32)


# (node blocks, block size, edges, batch): 4 / 2 / 3 node blocks (an edge's rows 4, 2, 1 at a time), every lane width
SHAPES = [(4, 64, 128, 128), (2, 128, 64, 32), (3, 64, 64, 64), (4, 64, 64, 100), (2, 64, 128, 256), (8, 64, 64, 256),
          (2, 64, 64, 512), (1, 64, 64, 1024), (4, 64, 192, 96)]


@needs_kernels
@pytest.mark.parametrize("fused", [False, True], ids = ["element_mars", "children"])
@pytest.mark.parametrize("gated", [False, True], ids = ["plain", "gated"])
@pytest.mark.parametrize("shared", [True, False], ids = ["shared", "per_block"])
@pytest.mark.parametrize("nbk,blk,ne,batch", SHAPES, ids = [f"nb{s[0]}_blk{s[1]}_e{s[2]}_B{s[3]}" for s in SHAPES])
def test_the_operands_are_the_captures_bits(nbk, blk, ne, batch, gated, fused, shared):
    c = _case(nbk, blk, ne, batch, gated, fused, shared, seed = nbk * 1000 + batch)
    _, _, cap = _mods()
    if not cap.fits(c["W"], batch):
        pytest.skip(f"a slot of {c['W']} columns at batch {batch}: the other captures' (`fits`)")
    cps, W = c["cps"], c["W"]
    ref, got = c["bufs"](), c["bufs"]()
    Ms = torch.zeros(c["S"] // cps, nbk, cps * W, device = DEV, dtype = torch.int32)
    for step in (0, 1):                                          # two steps into one group's slots and maxima
        M = torch.zeros(nbk, cps * W, device = DEV, dtype = torch.int32)
        if step == 1:                                            # (another step's values)
            c["nf"] = c["nf"].roll(1, 0) * 0.5
        _old(c, ref, M, step * cps)
        _new(c, got, Ms, step * cps)
        torch.cuda.synchronize()
        assert _bits(Ms[step]).equal(_bits(M)), f"step {step}: the column maxima"
    for k in ("A", "Bop", "fA", "fB"):
        assert _bits(got[k]).equal(_bits(ref[k])), f"{k} differs"



@needs_kernels
@pytest.mark.parametrize("fused", [False, True], ids = ["element_mars", "children"])
@pytest.mark.parametrize("shared", [True, False], ids = ["shared", "per_block"])
def test_two_rows_a_warp_read_the_finished_maxima(shared, fused):
    """Enough rows for the operands' kernel to take two a warp (`capture_steps`: two blocks an SM still) at 16 columns a
    lane -- an HMM-4096's capture at batch 512 --, repeated: there a `const __restrict__` pointer to the maxima let the
    compiler load them above the wait for the launch that writes them (a slot's operands from maxima half made, in
    most runs). Bits against the old capture, as above."""
    sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    nbk, blk, batch = 4, 1024, 512
    rows = lambda ne: nbk * blk + (ne if shared else nbk * ne)
    ne = 128
    while rows(ne) // 16 < 2 * sms:                              # (the kernel's rule for two rows a warp)
        ne += 64
    _, _, cap = _mods()
    for rep in range(4):
        c = _case(nbk, blk, ne, batch, False, fused, shared, seed = rep)
        cps, W = c["cps"], c["W"]
        assert W == 512 and cap.fits(W, batch)
        ref, got = c["bufs"](), c["bufs"]()
        Ms = torch.zeros(c["S"] // cps, nbk, cps * W, device = DEV, dtype = torch.int32)
        for step in (0, 1):
            M = torch.zeros(nbk, cps * W, device = DEV, dtype = torch.int32)
            if step == 1:
                c["nf"] = c["nf"].roll(1, 0) * 0.5
            _old(c, ref, M, step * cps)
            _new(c, got, Ms, step * cps)
            torch.cuda.synchronize()
            assert _bits(Ms[step]).equal(_bits(M)), f"run {rep} step {step}: the column maxima"
        for k in ("A", "Bop", "fA", "fB"):
            assert _bits(got[k]).equal(_bits(ref[k])), f"run {rep}: {k} differs"

def _groups():
    from pyjuice.layer import deferred_par as dpar
    return [g for _, (_, gs) in list(dpar._GROUPS.items()) for g in gs.values()]


def _block_sparse_chain(T, V):
    """An HMM's chain of tied steps whose sum layer connects each of its 4 node blocks (512 nodes) to 2 consecutive of
    the 4 product blocks below (the CUDA parameter flows' contiguous children) -- node blocks that do NOT all share
    their edges' children."""
    import pyjuice as juice
    import pyjuice.nodes.distributions as dists
    from pyjuice.nodes import inputs, multiply, summate
    with juice.set_block_size(512):
        ns = inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = V))
        src = esrc = None
        for t in range(1, T):
            if esrc is None:
                emit = esrc = inputs(t, num_node_blocks = 4, dist = dists.Categorical(num_cats = V))
            else:
                emit = esrc.duplicate(t, tie_params = True)
            prod = multiply(ns, emit)
            if src is None:
                edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 2, 3]])   # (runs of rows)
                ns = src = summate(prod, num_node_blocks = 4, edge_ids = edges)
            else:
                ns = src.duplicate(prod, tie_params = True)
        return summate(multiply(ns), num_node_blocks = 1, block_size = 1)


@needs_kernels
@pytest.mark.parametrize("case", ["one_flush", "flushed_mid_pass", "block_sparse"])
def test_a_backward_defers_the_same_flows_through_either_capture(case, monkeypatch):
    """`deferred_par.try_defer` through `capture_steps` -- its node blocks sharing their children or taken one by one
    -- and through the old capture: the group's flows of two accumulating backwards to the bit; `flushed_mid_pass`:
    slots for two steps, so the group flushes during the pass and fills its slots and maxima from the first again;
    `block_sparse`: node blocks of different children (found so: `shared` False)."""
    import pyjuice as juice
    from pyjuice.layer import deferred_par as dpar
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    if case == "flushed_mid_pass":
        monkeypatch.setattr(dpar, "PAR_DEFER_MB", 1.6)          # (2 slots: blocks of 512, 1024 edges, W 128)
    torch.manual_seed(0)
    if case == "block_sparse":
        ns = _block_sparse_chain(8, 32)
    else:
        ns = juice.structures.HMM(seq_length = 8, num_latents = 1024, num_emits = 32, homogeneous = True,
                                  block_size = 512)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = torch.bfloat16).to(DEV)
    x = torch.randint(0, 32, (128, 8), device = DEV)

    def backwards():
        pc.init_param_flows(flows_memory = 0.0)
        for _ in range(2):
            pc(x)
            pc.backward(x, flows_memory = 1.0)
        torch.cuda.synchronize()

    backwards()
    (group,) = _groups()
    assert group.S == (2 if case == "flushed_mid_pass" else 64) and group.steps_ok
    members = [m for mod in pc.modules() for m in mod.__dict__.get("_bs_pd_member", {}).values() if m]
    assert members and all(m[4] == (case != "block_sparse") for m in members), "which node blocks share children"
    lo, hi = int(group.pfids.min()), int(group.pfids.max()) + group.A.size(1)
    out = {}
    arms = ("old", "per_block") if case == "block_sparse" else ("old", "shared", "per_block")
    for arm in arms:
        group.steps_ok = arm != "old"
        for mod in pc.modules():
            cache = mod.__dict__.get("_bs_pd_member", {})
            for k, m in list(cache.items()):
                if m:
                    cache[k] = m[:4] + (arm == "shared",)
        backwards()
        out[arm] = pc.param_flows[lo:hi].clone()
    for arm in arms[1:]:
        assert _bits(out[arm]).equal(_bits(out["old"])), f"{arm}: the group's flows differ from the old capture's"


def _steer_fused(monkeypatch):
    """The forward's and the element backward's CUDA choices steered to the stream kernels and the product layer's
    fusion into them, as stream_fusion_test steers them (the tuners' timing decides them otherwise, and a busy GPU may
    decline the fusion: then no capture is the side stream's to take); nothing kept from an earlier test."""
    from pyjuice.layer.kernels import autotune
    real_choose, real_best_of, real_pick = autotune.choose, autotune.best_of, autotune.pick

    def choose(key, cands):
        if key[0] == "sum_fw_tlmm_cuda":
            stream = [(c, run) for c, run in cands if c[0] == "cuda_stream"]
            if stream:
                return real_choose(key, stream[:1])
        return real_choose(key, cands)

    def pick(key, cands, bench, *a, **k):
        if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk"):
            for c in cands:                                      # (the trials run, as the tuner runs them)
                bench(c)
            return "fused"
        return real_pick(key, cands, bench, *a, **k)

    monkeypatch.setattr(autotune, "_CACHE", {})
    monkeypatch.setattr(autotune, "choose", choose)
    monkeypatch.setattr(autotune, "pick", pick)
    monkeypatch.setattr(autotune, "best_of", lambda cands, *a, **k: next(
        (key for key, _ in cands if isinstance(key, tuple) and key[:1] == ("cuda_stream",)), None)
        or real_best_of(cands, *a, **k))


def _fused_hmm(T = 6, B = 128):
    """A homogeneous HMM of 2048 states in 2 blocks of 1024: the stream kernels with the product layer fused into them,
    so the deferral captures from the children's node values (what a side stream may take)."""
    import pyjuice as juice
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = 2048, num_emits = 32, homogeneous = True, block_size = 1024)
    ns.init_parameters(perturbation = 2.0)
    pc = juice.compile(ns, verbose = False, param_dtype = torch.bfloat16).to(DEV)
    return pc, torch.randint(0, 32, (B, T), device = DEV)


def _place(group, side):
    """The group's captures forced onto the side stream (True) or the main one, its measuring dropped."""
    group.side, group.mode, group.trials, group.warm = side, None, [], 0


#: GPU cycles the side stream sleeps before each capture there (`_watch_side`; ~2 ms), and the main stream before a
#: backward (~80 ms: longer than the host takes to queue it)
SIDE_SLEEP, MAIN_SLEEP = 5_000_000, 200_000_000


def _watch_side(monkeypatch, sleep = 0):
    """The captures' fetches of `deferred_par`'s side stream, recorded (the list returned) -- and with `sleep`, the
    side stream held back behind a sleep of that many cycles before each: a capture there then runs after the main
    stream has gone past it, so a launch that does not wait for it (a join left out) runs first, every time."""
    from pyjuice.layer import deferred_par as dpar
    taken, real = [], dpar._side_stream

    def side_stream(dev):
        entry = real(dev)
        taken.append(dev)
        if sleep:
            with torch.cuda.stream(entry[0]):
                torch.cuda._sleep(sleep)
        return entry

    monkeypatch.setattr(dpar, "_side_stream", side_stream)
    return taken


def _mixed_hmm(T = 6, gated_steps = 3, B = 128):
    """A homogeneous HMM's chain of 4096 states in 4 blocks of 1024 whose first `gated_steps` transitions are
    BlockScale-gated (tied too, gates of 32): the lowest regular layer's captures read the top gated layer's node values
    -- which that layer's backward shifts by log Z."""
    import pyjuice as juice
    import pyjuice.nodes.distributions as dists
    from pyjuice.nodes import inputs, multiply, summate
    from pyjuice.parameterizations import BlockScale
    torch.manual_seed(0)
    gated, esrc, gsrc, rsrc = [], None, None, None
    with juice.set_block_size(1024):
        ns = inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = 32))
        for t in range(1, T):
            if esrc is None:
                emit = esrc = inputs(t, num_node_blocks = 4, dist = dists.Categorical(num_cats = 32))
            else:
                emit = esrc.duplicate(t, tie_params = True)
            prod = multiply(ns, emit)
            if t <= gated_steps:
                if gsrc is None:
                    ns = gsrc = summate(prod, num_node_blocks = 4,
                                        parameterization = BlockScale(ch_block_size = 32))
                else:
                    ns = gsrc.duplicate(prod, tie_params = True)
                gated.append(ns)
            elif rsrc is None:
                ns = rsrc = summate(prod, num_node_blocks = 4)
            else:
                ns = rsrc.duplicate(prod, tie_params = True)
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(DEV)
    phi = {n: torch.randn(n.parameterization.tensor_shapes(n, B)[0], device = DEV) for n in gated}
    return pc, phi, torch.randint(0, 32, (B, T), device = DEV)


@needs_kernels
@pytest.mark.parametrize("graphs", [False, True], ids = ["eager", "cuda_graph"])
def test_captures_on_the_side_stream_give_the_same_flows(graphs, monkeypatch):
    """`deferred_par`'s side stream: the group's flows of a backward whose captures ran there, to the bit as on the
    main stream -- eager, and from a replayed CUDA graph recorded with them there (the fork and the join in the graph);
    the side stream really taken, and every capture joined by the time the backward returns."""
    from pyjuice.layer import deferred_par as dpar
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    _steer_fused(monkeypatch)
    pc, x = _fused_hmm()
    # (eager: the side stream held back -- a flush that did not wait for its captures would read the poisoned slots)
    taken = _watch_side(monkeypatch, sleep = 0 if graphs else SIDE_SLEEP)

    def backward(**kw):
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0, **kw)
        assert not any(e[1] for e in dpar._SIDE.values()), "a capture left on the side stream after the backward"
        torch.cuda.synchronize()

    backward()
    backward()                                                   # (the fusion decided, the group made)
    (group,) = _groups()
    assert group.steps_ok
    lo, hi = int(group.pfids.min()), int(group.pfids.max()) + group.A.size(1)
    _place(group, False)
    backward()
    ref = pc.param_flows[lo:hi].clone()
    _place(group, True)
    taken.clear()
    if graphs:
        for _ in range(3):                                       # record, then replay
            backward(record_cudagraph = True)
    else:
        for t in (group.A, group.Bop, group.fA, group.fB):
            t.fill_(float("nan"))
        backward()
        assert taken, "the captures did not take the side stream"
    assert _bits(pc.param_flows[lo:hi]).equal(_bits(ref)), "the side stream's flows differ"


@needs_kernels
def test_captures_on_the_side_stream_wait_for_the_main_one(monkeypatch):
    """The fork: a capture on the side stream starts after what the main stream queued before it -- the main stream
    held back by a sleep before the backward, its buffers still another batch's values: the flows the main stream's to
    the bit, and the side stream still waiting as the backward returns (its captures there)."""
    from pyjuice.layer import deferred_par as dpar
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    _steer_fused(monkeypatch)
    pc, x = _fused_hmm()
    other = torch.randint(0, 32, x.shape, device = DEV, generator = torch.Generator(device = DEV).manual_seed(7))
    taken = _watch_side(monkeypatch)

    def backward(x, hold = 0):
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        if hold:
            torch.cuda._sleep(hold)
        pc.backward(x, flows_memory = 1.0)

    backward(x)
    backward(x)
    torch.cuda.synchronize()
    (group,) = _groups()
    lo, hi = int(group.pfids.min()), int(group.pfids.max()) + group.A.size(1)
    _place(group, False)
    backward(x)
    torch.cuda.synchronize()
    ref = pc.param_flows[lo:hi].clone()
    _place(group, True)
    backward(other)
    torch.cuda.synchronize()
    taken.clear()
    backward(x, hold = MAIN_SLEEP)
    waiting = not dpar._SIDE[DEV.index][0].query()
    torch.cuda.synchronize()
    assert taken and waiting, "the captures did not wait on the side stream"
    assert _bits(pc.param_flows[lo:hi]).equal(_bits(ref)), "the side stream's flows differ"


@needs_kernels
def test_a_backward_that_raised_joined_its_side_captures(monkeypatch):
    """A backward that raised after captures went to the side stream, before its flush: its scope's close joined them
    -- no later launch, nor a buffer freed after it, overlaps them --, and the next backward's flows are the main
    stream's."""
    import pyjuice.layer.kernels.c as ck
    from pyjuice.layer import deferred_par as dpar
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    _steer_fused(monkeypatch)
    pc, x = _fused_hmm()
    taken = _watch_side(monkeypatch, sleep = SIDE_SLEEP)

    def backward():
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)
        torch.cuda.synchronize()

    backward()
    backward()
    (group,) = _groups()
    lo, hi = int(group.pfids.min()), int(group.pfids.max()) + group.A.size(1)
    _place(group, False)
    backward()
    ref = pc.param_flows[lo:hi].clone()
    _place(group, True)
    cmod = ck.get_par_deferred_capture_module()
    real_cap, calls = cmod.capture_steps, []

    def capture_steps(*a):
        calls.append(a[-1])
        if len(calls) == 3:
            raise RuntimeError("injected")
        return real_cap(*a)

    monkeypatch.setattr(cmod, "capture_steps", capture_steps)
    taken.clear()
    with pytest.raises(RuntimeError, match = "injected"):
        backward()
    assert taken and calls[0] != 0, "the captures did not take the side stream"
    assert not any(e[1] for e in dpar._SIDE.values()), "the raised backward left captures on the side stream"
    monkeypatch.setattr(cmod, "capture_steps", real_cap)
    backward()
    assert _bits(pc.param_flows[lo:hi]).equal(_bits(ref)), "the next backward's flows differ"


@needs_kernels
def test_a_gated_layer_below_joins_the_side_stream_first(monkeypatch):
    """A regular tied layer above a gated one captures from the gated layer's node values, which that layer's backward
    shifts by log Z in place (its parameterization's `pre_backward_layer`): no capture is left on the side stream at
    any such shift (`ParameterizedSumLayer.backward` joins it first), the gated layers' own captures (their products
    fused too: BlockScale's split kernels steered) stay on the main stream, and the groups' flows are the main stream's
    to the bit -- the side stream held back."""
    import pyjuice.layer.kernels.c as ck
    import pyjuice.parameterizations.kernels.shift_logz as sl
    from pyjuice.layer import deferred_par as dpar
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    _steer_fused(monkeypatch)
    real_tune = ck.autotune
    monkeypatch.setattr(ck, "autotune", lambda cands, *a, **k: next(
        (key for key, _ in cands if isinstance(key, tuple) and key[:1] in (("stream",), ("eles",))), None)
        or real_tune(cands, *a, **k))
    pc, phi, x = _mixed_hmm()
    _watch_side(monkeypatch, sleep = SIDE_SLEEP)
    left, real_shift = [], sl.shift_logz

    def shift_logz(node_mars, nids, log_z, block_size, sign):
        if sign > 0:
            left.append(any(e[1] for e in dpar._SIDE.values()))
        return real_shift(node_mars, nids, log_z, block_size, sign)

    monkeypatch.setattr(sl, "shift_logz", shift_logz)
    cmod = ck.get_par_deferred_capture_module()
    real_cap, places = cmod.capture_steps, []                 # (gated, fused, on the side stream) a capture
    monkeypatch.setattr(cmod, "capture_steps", lambda *a: (places.append((a[19], a[20], a[-1] != 0)), real_cap(*a))[1])

    def backward():
        pc.init_param_flows(flows_memory = 0.0)
        pc(x, sum_external_params = phi)
        pc.backward(x, flows_memory = 1.0)
        torch.cuda.synchronize()

    backward()
    backward()
    groups = _groups()
    # (the groups' flows only: elsewhere a backward's atomics may round in another order)
    ranges = [(int(g.pfids.min()), int(g.pfids.max()) + g.A.size(1)) for g in groups]
    for g in groups:
        _place(g, False)
    backward()
    ref = [pc.param_flows[a:b].clone() for a, b in ranges]
    for g in groups:
        _place(g, True)
    left.clear()
    places.clear()
    backward()
    assert any(side for _, _, side in places), "no capture went to the side stream"
    assert any(g and f for g, f, _ in places), "expected the gated layers' captures from their children"
    assert not any(g and side for g, _, side in places), "a gated capture went to the side stream"
    assert left and not any(left), "a capture was left on the side stream as a gated layer shifted its node values"
    for (a, b), r in zip(ranges, ref):
        assert _bits(pc.param_flows[a:b]).equal(_bits(r)), "the side stream's flows differ"


def test_the_group_keeps_the_faster_place(monkeypatch):
    """The choice (`_Group.on_side`, timings scripted): a backward's period its first capture to the next's, the best
    of each place's after `SIDE_WARM` untimed -- the side stream where it is at least `SIDE_MARGIN` faster, else the
    main one; kept after, untimed; a capture not eligible always on the main stream, no trial spent."""
    from types import SimpleNamespace
    from pyjuice.layer import deferred_par as dpar
    from pyjuice.layer.kernels import autotune
    monkeypatch.setattr(autotune, "ENABLED", True)
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

    monkeypatch.setattr(dpar.torch.cuda, "Event", Event)

    def chosen(main, side):
        g = SimpleNamespace(side = None, mode = None, trials = [], warm = dpar.SIDE_WARM)
        for i in range(dpar.SIDE_WARM + 2 * dpar.SIDE_TRIALS + 1):
            g.mode = None
            assert dpar._Group.on_side(g, False) is False and g.mode is False
            g.mode = None
            clock[0] += side if dpar._Group.on_side(g, True) else main       # (this backward's period)
            assert (g.side is None) == (i < dpar.SIDE_WARM + 2 * dpar.SIDE_TRIALS), "decided after the trials"
        g.mode = None
        assert dpar._Group.on_side(g, True) is g.side and not g.trials
        return g.side

    assert chosen(10.0, 9.0) is True
    assert chosen(9.0, 10.0) is False
    assert chosen(10.0, 10.0 * (1 - dpar.SIDE_MARGIN) + 0.01) is False


@needs_kernels
def test_the_group_measures_where_its_captures_run(monkeypatch):
    """With autotuning on, a group times its backwards in each place (`SIDE_WARM` untimed, then `SIDE_TRIALS` in each,
    a backward's first capture to the next one's) and keeps one -- the flows the same bits all along; with it off, the
    main stream, untimed."""
    from pyjuice.layer import deferred_par as dpar
    from pyjuice.layer.kernels import autotune
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    monkeypatch.setattr(autotune, "ENABLED", True)
    _steer_fused(monkeypatch)
    pc, x = _fused_hmm()
    outs = []
    for i in range(2 + dpar.SIDE_WARM + 2 * dpar.SIDE_TRIALS + 4):
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)
        torch.cuda.synchronize()
        (group,) = _groups()
        lo, hi = int(group.pfids.min()), int(group.pfids.max()) + group.A.size(1)
        outs.append(pc.param_flows[lo:hi].clone())
    assert group.side is not None, "no place chosen after the trials"
    assert all(_bits(o).equal(_bits(outs[0])) for o in outs[1:]), "the flows changed with the place"
    monkeypatch.setattr(autotune, "ENABLED", False)
    _place(group, None)
    pc.init_param_flows(flows_memory = 0.0)
    pc(x)
    pc.backward(x, flows_memory = 1.0)
    assert group.side is None and group.mode is False and not group.trials


@needs_kernels
def test_captures_from_element_mars_stay_on_the_main_stream(monkeypatch):
    """A capture that reads `element_mars` (the product layer not fused into the stream kernels: the circuit's fusion
    decisions switched off) never goes to the side stream -- the next layers' products rewrite those rows --, whatever
    the group chose: the flows the main stream's."""
    from pyjuice.layer import deferred_par as dpar
    import pyjuice.layer.kernels.c as ck
    monkeypatch.setattr(dpar, "MIN_FLOWS", 1 << 20)
    taken = _watch_side(monkeypatch)
    _steer_fused(monkeypatch)
    pc, x = _fused_hmm()
    pc._stream_fusion_groups = lambda *a, **k: frozenset()
    pc._stream_fusion_bk_groups = lambda *a, **k: frozenset()
    fused_calls = []
    cmod = ck.get_par_deferred_capture_module()
    real_cap = cmod.capture_steps
    monkeypatch.setattr(cmod, "capture_steps", lambda *a: (fused_calls.append(a[20]), real_cap(*a))[1])

    def backward():
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)
        torch.cuda.synchronize()

    backward()
    (group,) = _groups()
    lo, hi = int(group.pfids.min()), int(group.pfids.max()) + group.A.size(1)
    _place(group, False)
    backward()
    ref = pc.param_flows[lo:hi].clone()
    _place(group, True)
    fused_calls.clear()
    backward()
    assert fused_calls and not any(fused_calls), "expected captures from element_mars here"
    assert not taken, "a capture from element_mars went to the side stream"
    assert _bits(pc.param_flows[lo:hi]).equal(_bits(ref))
