"""
The deferred parameter flows' capture in two chained launches (layer/kernels/c/par_deferred_fp16_capture.cu,
`capture_steps`) against the captures it stands in for -- par_deferred_fp16.cu's `capture` (the element values from
`element_mars`) and par_deferred_fp16_fused.cu's `capture_fused` (from the products' two children): every operand
value, row scale and column maximum to the bit, over the slot layouts it takes (1, 2, 4, 8 and 16 columns a lane, a
batch past the last full lane, two slots a step), node blocks that share their edges' children (one warp an edge's rows
of every block: 1, 2 and 4 at a time) and blocks that do not, with gates (missing ones too) and without, nodes and
columns at -inf, dummy element rows; two steps into one group's maxima; two rows a warp at 16 columns a lane, repeated
(the maxima read only after the wait). Where it does not apply (`fits`): the other widths. Skipped where the CUDA
kernels cannot build.
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
                      c["fused"], c["shared"])


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
