"""
The bf16 copy of shared fp32 parameters (`TensorCircuit._param_operands`): under "auto" from batch 16, the sum layers
that read only parameters several sum nodes share (a tied HMM's transitions), at offsets the bf16 kernels' 16-byte loads
take, read that copy in their kernels whose products are bf16. The CUDA kernels that round the parameters to bf16 anyway
compute the same bits on it (each forced in turn, against the circuit without it), and the stream kernels, which read
bf16 parameters only, join them -- the forward's and the element flows': their bits are a bf16-stored twin's. Every
other kernel reads the fp32 parameters, as without the copy -- the fp16 and fp32 element kernels, the fp32 forward forks,
the Triton kernels, the parameter flows --, so a circuit taking no stream kernel computes the bits it computed without
it. Tied BlockScale-gated layers too (`SumParameterization.reads_param_operands`): its bf16 forks on the copy, its split
forward and split element backward joining them, `sigma` from the copy where the split forward summed `log Z` from it
(the forward and the gate gradient). Nowhere else: untied layers, a parameterization that does not declare it, batches
below 16, the "tf32" / "fp32" precisions, parameters stored in bf16. The copy is made inside each pass: a write through
`params.data` is in the next pass's values, eager or replayed from a recorded CUDA graph. Whether a pass takes the copy
is a rule on the work per layer (`TensorCircuit._param_operand_groups`): forced on here but for the rule's own tests.
"""

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.layer.kernels.autotune as autotune
import pyjuice.layer.kernels.c as cuda_kernels
import pyjuice.model.tensorcircuit as tcm
from pyjuice.layer.sum_layer import SumLayer

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


@pytest.fixture(autouse = True)
def _copy_forced(monkeypatch):
    """The copy taken wherever it applies, its rule overridden: these are the copy's own tests (the rule's set it back
    to None)."""
    monkeypatch.setattr(tcm.TensorCircuit, "_param_operands_force", True)


def _hmm(L, T = 8, E = 64, dtype = torch.float32, **kw):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = L, num_emits = E)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = dtype, **kw).to(DEV), T, E


def _marked(pc):
    return [layer for group in pc.inner_layer_groups if group.is_sum() for layer in group
            if layer.__dict__.get("_reads_param_operands", False)]


def _no_copy(pc):
    pc._param_operands = lambda *a, **k: None
    return pc


def _spy(monkeypatch, mod, name, pos, seen):
    """``mod.name`` records, with its name, the dtype of its positional argument ``pos`` -- the parameters it is given
    -- into ``seen``."""
    real = getattr(mod, name)

    def f(*a, **k):
        seen.add((name, a[pos].dtype))
        return real(*a, **k)
    monkeypatch.setattr(mod, name, f)


@cuda_only
def test_the_tied_layers_take_it_and_nothing_else():
    pc, T, E = _hmm(1024)
    x = torch.randint(0, E, (64, T), device = DEV)
    with torch.no_grad():
        pc(x)
    groups, ranges = pc._param_operand_plan()
    assert len(groups) == T - 1 and len(_marked(pc)) == T - 1          # the transitions, not the root
    (a, b), = ranges
    assert b - a == 1024 * 1024
    op = pc._param_operands(64)
    assert op.dtype == torch.bfloat16 and torch.equal(op[a:b], pc.params.detach()[a:b].to(torch.bfloat16))

    # not below batch 16, under another precision, for parameters stored in bf16
    assert pc._param_operands(8) is None
    for precision in ("tf32", "fp32"):
        pc.precision = precision
        assert pc._param_operands(64) is None
    pc.precision = "auto"
    pc16, _, _ = _hmm(1024, dtype = torch.bfloat16)
    assert pc16._param_operands(64) is None

    # untied: an HCLT has no shared parameters
    xs = torch.randint(0, 16, (256, 12))
    hclt = juice.compile(juice.structures.HCLT(xs.float(), num_latents = 32, input_dist = dists.Categorical(num_cats = 16)),
                         verbose = False).to(DEV)
    assert len(hclt._param_operand_plan()[0]) == 0 and hclt._param_operands(64) is None and not _marked(hclt)


@cuda_only
def test_a_layer_whose_parameter_bases_are_not_multiples_of_8_is_not_marked():
    """The bf16 kernels load 8 parameters as one 16-byte load: a tied layer whose parameter bases -- its forward's
    (`partitioned_pids`) or its backward's (`partitioned_parpids`) -- are not multiples of 8 is left on the fp32
    parameters (the tables shifted by 4 for the plan's build, as packing a 4-parameter node ahead of them would)."""
    for attr in ("partitioned_pids", "partitioned_parpids"):
        pc, T, E = _hmm(256)
        layer = [l for group in pc.inner_layer_groups if group.is_sum() for l in group][1]
        real = getattr(layer, attr)
        layer.__dict__[attr] = [t + 4 for t in real]
        try:
            groups, _ = pc._param_operand_plan()
        finally:
            layer.__dict__.pop(attr)
        assert getattr(layer, attr) is real
        assert not layer.__dict__["_reads_param_operands"] and len(groups) == T - 2, attr
        assert len(_marked(pc)) == T - 2


def _gated_chain(T = 5, L = 256, nb = 4, gcb = 32, C = 16):
    """A tied chain of BlockScale-gated sum layers (an HMM's transitions, `nb` node blocks of `L // nb`, gates of `gcb`
    children); its gated layers' nodes."""
    from pyjuice.nodes import inputs, multiply, summate
    from pyjuice.parameterizations import BlockScale
    torch.manual_seed(0)
    gated = []
    with juice.set_block_size(L // nb):
        chain = inputs(0, num_node_blocks = nb, dist = dists.Categorical(num_cats = C))
        first = None
        for t in range(1, T):
            emit = inputs(t, num_node_blocks = nb, dist = dists.Categorical(num_cats = C))
            if first is None:
                chain = first = summate(multiply(chain, emit), num_node_blocks = nb,
                                        parameterization = BlockScale(ch_block_size = gcb))
            else:
                chain = first.duplicate(multiply(chain, emit), tie_params = True)
            gated.append(chain)
        root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, gated


def _gates(gated, B):
    g = torch.Generator(device = DEV).manual_seed(2)
    return {n: torch.randn(n.parameterization.tensor_shapes(n, B)[0], device = DEV, generator = g) * 0.5 - 0.5
            for n in gated}


def _gated_rows(pc):
    from pyjuice.layer.external_sum_layer import ExternalParamsSumLayer
    return [layer._layer_nid_range for group in pc.inner_layer_groups if group.is_sum() for layer in group
            if isinstance(layer, ExternalParamsSumLayer)]


@cuda_only
def test_external_parameter_layers_are_marked_if_their_parameterization_takes_it(monkeypatch):
    """A tied gated chain: its layers take the copy (`SumParameterization.reads_param_operands`, BlockScale's); a
    parameterization that does not declare it leaves its layers unmarked."""
    from pyjuice.layer.external_sum_layer import ExternalParamsSumLayer
    from pyjuice.parameterizations import BlockScale
    root, _ = _gated_chain(L = 128)
    pc = juice.compile(root, verbose = False).to(DEV)
    pc._param_operand_plan()
    gated = [layer for group in pc.inner_layer_groups if group.is_sum() for layer in group
             if isinstance(layer, ExternalParamsSumLayer)]
    assert gated and all(layer.__dict__.get("_reads_param_operands") for layer in gated)
    monkeypatch.setattr(BlockScale, "reads_param_operands", False)
    pc2 = juice.compile(root, verbose = False).to(DEV)
    pc2._param_operand_plan()
    assert not any(isinstance(layer, ExternalParamsSumLayer) for layer in _marked(pc2))


# ---------------------------------------------------------------------------------------------------------------------
# The forward
# ---------------------------------------------------------------------------------------------------------------------

@cuda_only
def test_the_dense_layers_take_the_stream_kernels_on_the_copy(monkeypatch):
    """Blocks of 1024 at batch 512 (`_DENSE_MIN_WORK`: edges x batch from 2^19): on fp32 parameters the dense cuBLAS
    path; on the copy the CUDA arms, the stream kernel's among them."""
    pc, T, E = _hmm(1024, T = 4)
    x = torch.randint(0, E, (512, T), device = DEV)
    dense, offered = [], set()
    real_dense, real_choose, real_cached = SumLayer._forward_dense, autotune.choose, autotune.cached
    monkeypatch.setattr(SumLayer, "_forward_dense", lambda self, *a, **k: dense.append(1) or real_dense(self, *a, **k))
    monkeypatch.setattr(autotune, "cached", lambda key: None if key[0] == "sum_fw_tlmm_cuda" else real_cached(key))
    monkeypatch.setattr(autotune, "choose", lambda key, cands: offered.update(c[0] for c, _ in cands) or
                        real_choose(key, cands))
    real_operands = pc._param_operands
    with torch.no_grad():
        _no_copy(pc)
        pc(x)
        assert len(dense) == T - 1 and not offered                         # (the root layer: one block)
        dense.clear()
        pc._param_operands = real_operands
        pc(x)
    assert not dense and "cuda_stream" in offered, (dense, offered)


@cuda_only
@pytest.mark.parametrize("arm", ["cuda", "cuda_pipe", "cuda_prepass", "cuda_stream"])
def test_every_cuda_arm_reads_the_copy_and_computes_the_same_bits(arm, monkeypatch):
    """A tied HMM of 512-node blocks (below the dense path: its fp32 parameters take the CUDA arms too), each arm forced
    with the copy and without it: every arm rounds the parameters to bf16 for its products, so it reads the copy, and
    computes the same node values to the bit; the stream kernel, bf16 parameters only, is offered on the copy alone."""
    pc, T, E = _hmm(512)
    x = torch.randint(0, E, (128, T), device = DEV)
    taken, seen = [], set()

    def choose(key, cands):
        for choice, _ in cands:
            if choice[0] == arm:
                taken.append((key[-1] == "operands", choice))
                return choice
        taken.append((key[-1] == "operands", None))
        return ("triton", -1)
    real_cached = autotune.cached
    monkeypatch.setattr(autotune, "cached", lambda key: None if key[0] == "sum_fw_tlmm_cuda" else real_cached(key))
    monkeypatch.setattr(autotune, "choose", choose)
    monkeypatch.setattr(SumLayer, "_fw_fast_launch", lambda self, *a, **k: False)      # (every call dispatched)
    name = {"cuda": "tlmm_forward_sum", "cuda_pipe": "tlmm_forward_sum_pipelined",
            "cuda_prepass": "tlmm_forward_sum_prepass", "cuda_stream": "tlmm_forward_sum_stream"}[arm]
    _spy(monkeypatch, cuda_kernels, name, 2, seen)

    values = {}
    real_operands = pc._param_operands
    for copy in (False, True):
        pc._param_operands = real_operands if copy else (lambda *a, **k: None)
        with torch.no_grad():
            pc(x)
        values[copy] = pc.node_mars.clone()
    ran = {c for c, choice in taken if choice is not None}
    if arm == "cuda_stream":
        assert ran == {True} and seen == {(name, torch.bfloat16)}, (taken, seen)
        return
    if ran != {False, True}:
        pytest.skip(f"no {arm} arm at this shape")
    assert seen == {(name, torch.float32), (name, torch.bfloat16)}, seen
    assert torch.equal(values[True], values[False])


@cuda_only
@pytest.mark.parametrize("arm", ["cuda_mma", "cuda_bt"])
def test_the_small_batch_arms_read_what_their_products_take(arm, monkeypatch):
    """Batch 32, the small-batch kernels forced (the tlmm gate falling through): the tensor-core arm rounds the
    parameters to bf16, so it reads the copy, the batch-tiled fp32 one the fp32 parameters -- each the same node values
    to the bit with the copy and without it."""
    pc, T, E = _hmm(1024)
    x = torch.randint(0, E, (32, T), device = DEV)
    taken, seen = [], set()
    real_pick, real_choose = autotune.pick, autotune.choose

    def pick(key, cands, bench, *a, **k):
        if key[0] != "sum_fw_smallbatch_cuda":
            return real_pick(key, cands, bench, *a, **k)
        c = next((c for c in cands if c[0] == arm), None)
        taken.append((key[-1] == "operands", c))
        return c or ("triton", -1)
    monkeypatch.setattr(autotune, "pick", pick)
    monkeypatch.setattr(autotune, "choose", lambda key, cands: ("triton", -1) if key[0] == "sum_fw_tlmm_cuda"
                        else real_choose(key, cands))
    real_cached = autotune.cached
    monkeypatch.setattr(autotune, "cached", lambda key: None if key[0] in ("sum_fw_tlmm_cuda", "sum_fw_smallbatch_cuda")
                        else real_cached(key))
    name = {"cuda_mma": "smallbatch_forward_sum_mma", "cuda_bt": "smallbatch_forward_sum_bt"}[arm]
    _spy(monkeypatch, cuda_kernels, name, 2, seen)

    values = {}
    real_operands = pc._param_operands
    for copy in (False, True):
        pc._param_operands = real_operands if copy else (lambda *a, **k: None)
        with torch.no_grad():
            pc(x)
        values[copy] = pc.node_mars.clone()
    if {c for c, choice in taken if choice is not None} != {False, True}:
        pytest.skip(f"no {arm} arm at this shape")
    assert seen == ({(name, torch.float32), (name, torch.bfloat16)} if arm == "cuda_mma"
                    else {(name, torch.float32)}), seen
    assert torch.equal(values[True], values[False])


def _force_fork(monkeypatch, kind, offered):
    """BlockScale's forward tuner (`pyjuice.layer.kernels.c.autotune`) records each call's candidates and takes
    ``kind``'s first config where it is offered."""
    import pyjuice.layer.kernels.c as ck
    real = ck.autotune

    def pick(cands, warmup = 3, reps = 7):
        offered.append([key for key, _ in cands])
        for key, _ in cands:
            if key[0] == kind:
                return key
        return real(cands, warmup = warmup, reps = reps)
    monkeypatch.setattr(ck, "autotune", pick)


_FORKS = {"cute": ("get_cute_module", "blockscale_forward"),
          "pre": ("get_prepass_module", "blockscale_forward_prepass"),
          "sbmma": ("get_sb_mma_module", "blockscale_sb_forward_mma"),
          "sbbt": ("get_sb_bt_module", "blockscale_sb_forward_bt"),
          "sb": ("get_sb_module", "blockscale_sb_forward")}


@cuda_only
@pytest.mark.parametrize("kind", ["cute", "pre", "sbmma", "sbbt", "sb"])
def test_every_gated_fork_reads_what_its_products_take(kind, monkeypatch):
    """The tied gated chain, each forward fork forced with the copy and without it: the forks whose products round the
    parameters to bf16 (the CuTe fork, its pre-pass copy, the tensor-core small-batch one) read the copy, the fp32 ones
    (the per-sample and batch-tiled small-batch forks) the fp32 parameters -- every one the same node values to the
    bit. Batch 64 for the first two, 32 for the small-batch forks."""
    import pyjuice.parameterizations.kernels.c as bk
    getter, name = _FORKS[kind]
    mod = getattr(bk, getter)()
    if mod is None:
        pytest.skip(f"no {kind} fork on this toolchain")
    seen = set()
    _spy(monkeypatch, mod, name, 2, seen)
    root, gated = _gated_chain()
    B = 64 if kind in ("cute", "pre") else 32
    x = torch.randint(0, 16, (B, 5), device = DEV)
    gates = _gates(gated, B)
    values, offered = {}, []
    _force_fork(monkeypatch, kind, offered)
    for copy in (False, True):
        pc = juice.compile(root, verbose = False).to(DEV)
        if not copy:
            _no_copy(pc)
        with torch.no_grad():
            pc(x, sum_external_params = gates)
        values[copy] = pc.node_mars.clone()
    if not any(key[0] == kind for call in offered for key in call):
        pytest.skip(f"no {kind} fork at this shape")
    bf16 = kind in ("cute", "pre", "sbmma")
    assert seen == ({(name, torch.float32), (name, torch.bfloat16)} if bf16 else {(name, torch.float32)}), seen
    for a, b in _gated_rows(pc):
        assert torch.equal(values[True][a:b], values[False][a:b]), (a, b)


@cuda_only
def test_the_gated_layers_take_the_split_forward_on_the_copy(monkeypatch):
    """On fp32 parameters BlockScale's split forward (bf16 parameters only) is not among the forward's candidates; on
    the copy it is."""
    import pyjuice.parameterizations.kernels.c as bk
    if bk.get_stream_module() is None:
        pytest.skip("no split forward on this toolchain")
    root, gated = _gated_chain()
    B = 64
    x = torch.randint(0, 16, (B, 5), device = DEV)
    offered = {}
    for copy in (False, True):
        pc = juice.compile(root, verbose = False).to(DEV)
        if not copy:
            _no_copy(pc)
        offered[copy] = []
        with monkeypatch.context() as mp:
            _force_fork(mp, "none", offered[copy])
            with torch.no_grad():
                pc(x, sum_external_params = _gates(gated, B))
    kinds = {c: {key[0] for call in offered[c] for key in call} for c in offered}
    assert offered[True] and "stream" in kinds[True] and "stream" not in kinds[False], kinds


@cuda_only
@pytest.mark.parametrize("kind", ["stream", "pre"])
def test_a_tied_gated_layer_computes_what_its_bf16_twin_computes(kind, monkeypatch):
    """The fp32 circuit's gated layers on the copy and a circuit storing the same parameters' rounding in bf16, each fork
    forced in both: their node values to the bit -- the split forward's `sigma` summed from the copy, as from the stored
    values."""
    import pyjuice.parameterizations.kernels.c as bk
    if kind == "stream" and bk.get_stream_module() is None:
        pytest.skip("no split forward on this toolchain")
    root, gated = _gated_chain()
    pc32 = juice.compile(root, verbose = False).to(DEV)
    pc16 = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(DEV)
    assert torch.equal(pc16.params, pc32.params.detach().to(torch.bfloat16))
    B = 64
    x = torch.randint(0, 16, (B, 5), device = DEV)
    gates = _gates(gated, B)
    offered = []
    _force_fork(monkeypatch, kind, offered)
    with torch.no_grad():
        pc32(x, sum_external_params = gates)
        pc16(x, sum_external_params = gates)
    if not any(key[0] == kind for call in offered for key in call):
        pytest.skip(f"no {kind} fork at this shape")
    rows = _gated_rows(pc32)
    assert len(rows) == len(gated)
    for a, b in rows:
        assert torch.equal(pc32.node_mars[a:b], pc16.node_mars[a:b]), (a, b)


@cuda_only
def test_a_tied_gated_circuit_trains_on_the_copy():
    """Training steps of a tied gated fp32 circuit (the forward on the copy; the backward, its gate gradient and the EM
    on fp32): finite, and with its log-likelihood within the forward's bf16 rounding of the same circuit without the
    copy, step after step."""
    root, gated = _gated_chain()
    B = 64
    x = torch.randint(0, 16, (B, 5), device = DEV)
    gates = _gates(gated, B)
    pcs = [juice.compile(root, verbose = False).to(DEV) for _ in range(2)]
    _no_copy(pcs[1])
    for step in range(3):
        lls = []
        for pc in pcs:
            grads = {n: torch.zeros_like(t) for n, t in gates.items()}
            pc.init_param_flows(flows_memory = 0.0)
            lls.append(pc(x, sum_external_params = gates).detach().clone())
            pc.backward(x, flows_memory = 1.0, sum_external_params = gates, sum_external_params_grad = grads)
            assert all(torch.isfinite(g).all() for g in grads.values())
            pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
            assert torch.isfinite(pc.params).all()
        assert (lls[0] - lls[1]).abs().max() < 1e-2, (step, float((lls[0] - lls[1]).abs().max()))


@cuda_only
def test_a_write_to_the_parameters_is_in_the_next_forward():
    """`pc.params.data[...] = x` moves no version counter: the copy is made in every forward, and the values follow --
    against a second circuit given the same parameters."""
    pc, T, E = _hmm(1024)
    ref, _, _ = _hmm(1024)
    x = torch.randint(0, E, (64, T), device = DEV)
    (a, b), = pc._param_operand_plan()[1]
    with torch.no_grad():
        pc(x)
        g = torch.Generator(device = DEV).manual_seed(1)
        pc.params.data[a:b] *= torch.rand(b - a, device = DEV, generator = g) + 0.5
        ref.params.data.copy_(pc.params.data)
        assert torch.equal(pc(x), ref(x))


@cuda_only
def test_the_backward_reads_the_copy_its_forward_made():
    """The backward's kernels read the copy their forward made -- the parameters its node values were computed with --:
    parameters written in between (through `params.data`) are copied by the next forward, not by the backward; a
    buffer made anew in between (as moving the circuit makes it) is filled before the backward's pass."""
    pc, T, E = _hmm(1024)
    x = torch.randint(0, E, (128, T), device = DEV)
    (a, b), = pc._param_operand_plan()[1]
    pc.init_param_flows(flows_memory = 0.0)
    pc(x)
    old = pc.params.detach()[a:b].to(torch.bfloat16)
    with torch.no_grad():
        pc.params.data[a:b] *= 1.5
    pc.backward(x, flows_memory = 1.0)
    assert torch.equal(pc._param_operands(x.size(0), refresh = False)[a:b], old)
    pc(x)
    new = pc.params.detach()[a:b].to(torch.bfloat16)
    assert not torch.equal(new, old) and torch.equal(pc._param_operands(x.size(0), refresh = False)[a:b], new)
    pc.__dict__.pop("_param_operands_")
    pc.backward(x, flows_memory = 1.0)
    assert torch.equal(pc._param_operands(x.size(0), refresh = False)[a:b], new)


@cuda_only
def test_parameters_given_a_new_storage_are_copied_from_it():
    """The copy's views of the parameters are kept between passes: `params.data` given a new storage, the next
    forward copies from that one -- its values a circuit holding the same parameters computes."""
    pc, T, E = _hmm(1024)
    ref, _, _ = _hmm(1024)
    x = torch.randint(0, E, (64, T), device = DEV)
    with torch.no_grad():
        pc(x)
        pc.params.data = pc.params.data * 1.25
        ref.params.data.copy_(pc.params.data)
        assert torch.equal(pc(x), ref(x))


@cuda_only
def test_training_steps_replayed_from_graphs_read_their_forwards_copy(monkeypatch):
    """Training steps whose forward and backward passes replay recorded CUDA graphs, the EM step changing the
    parameters in between: the forward's graph makes the copy, the backward's reads it -- each step's node flows those
    of the same step run eagerly from the same parameters, to the bit, the stream kernels taken in both. (Every
    parameter given to the replayed circuit in place before each step, so its graphs stay valid: the parameter flows'
    atomic sums leave two circuits' EM updates ulps apart -- MEASURED, node flows 1.3e-5 apart after three steps between
    two eager circuits, as between these two --, which a run of steps would compound.)"""
    offered = set()
    _prefer(monkeypatch, {"cuda_stream"}, offered)
    pcs = [_hmm(1024)[0] for _ in range(2)]
    T, E = 8, 64
    x = torch.randint(0, E, (128, T), device = DEV)
    for step in range(3):
        with torch.no_grad():
            for p0, p1 in zip(pcs[0].parameters(), pcs[1].parameters()):     # (the sum layers' and the inputs')
                if p0.is_floating_point():
                    p0.data.copy_(p1.data)
        for graphs, pc in zip((True, False), pcs):
            rec = graphs if step == 0 else None
            pc.init_param_flows(flows_memory = 0.0)
            pc(x, record_cudagraph = rec, apply_cudagraph = graphs)
            pc.backward(x, flows_memory = 1.0, record_cudagraph = rec, apply_cudagraph = graphs)
        assert "cuda_stream" in offered
        assert torch.equal(pcs[0].node_flows, pcs[1].node_flows), step
        assert torch.allclose(pcs[0].param_flows, pcs[1].param_flows, rtol = 1e-6, atol = 0), step
        for pc in pcs:
            pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    assert len(pcs[0]._recorded_cuda_graphs) == 2 and not pcs[1]._recorded_cuda_graphs


@cuda_only
def test_a_recorded_graph_makes_the_copy_too():
    """The copy is part of the pass: a graph recorded from it replays it, so a write through `params.data` between
    replays is in the replayed values -- as eager computes them."""
    pc, T, E = _hmm(1024)
    x = torch.randint(0, E, (64, T), device = DEV)
    with torch.no_grad():
        pc(x, record_cudagraph = True)
        (a, b), = pc._param_operand_plan()[1]
        pc.params.data[a:b] *= 1.25
        replayed = pc(x).clone()
        eager = pc(x, apply_cudagraph = False).clone()
    assert torch.equal(replayed, eager)


# ---------------------------------------------------------------------------------------------------------------------
# The backward: the stream element kernels read the copy, every other kernel the fp32 parameters
# ---------------------------------------------------------------------------------------------------------------------

def _prefer(monkeypatch, kinds, offered):
    """Every tuner (`autotune.best_of`, through which `pick` and `choose` measure, and BlockScale's
    `pyjuice.layer.kernels.c.autotune`) takes its first candidate of a kind in ``kinds`` where one is offered, else its
    first -- the same choices in two circuits offered the same candidates, nothing cached between calls or kept from
    an earlier test; each call's offered kinds into ``offered``."""
    import pyjuice.layer.kernels.c as ck

    def kind(key):
        return key[0] if isinstance(key, tuple) else key

    def pick(cands, *a, **k):
        keys = [key for key, _ in cands]
        offered.update(kind(key) for key in keys)
        return next((key for key in keys if kind(key) in kinds), keys[0])
    monkeypatch.setattr(autotune, "best_of", pick)
    monkeypatch.setattr(ck, "autotune", pick)
    monkeypatch.setattr(autotune, "cached", lambda key: None)
    monkeypatch.setattr(autotune, "_CACHE", {})


def _step(pc, x, **kw):
    pc.init_param_flows(flows_memory = 0.0)
    pc(x, **{k: v for k, v in kw.items() if k == "sum_external_params"})
    pc.backward(x, flows_memory = 1.0, **kw)


_ELE = {"cuda": "ele_backward_sum", "cuda_pipe": "ele_backward_sum_pipelined", "cuda_prepass": "ele_backward_sum_prepass",
        "cuda_stream": "ele_backward_sum_stream"}


@cuda_only
@pytest.mark.parametrize("arm", ["cuda_stream", "cuda_prepass", "cuda_pipe", "cuda", "triton"])
def test_only_the_stream_element_kernel_reads_the_copy(arm, monkeypatch):
    """A tied layer's backward with the copy, each element arm taken in turn: the stream kernel (bf16 parameters only)
    is given the copy; the fp16 CUDA arms and the Triton kernel the fp32 parameters, as without it, and so do the
    parameter-flow kernels (the flows' last factor the parameter itself)."""
    pc, T, E = _hmm(1024)
    offered, seen = set(), set()
    _prefer(monkeypatch, {arm}, offered)
    for a, name in _ELE.items():
        _spy(monkeypatch, cuda_kernels, name, 4, seen)
    inner = {"ele": set(), "par": set()}
    real_ele, real_par = SumLayer._backward_block_sparse_ele_flows, SumLayer._backward_block_sparse_par_flows

    def ele(self, node_flows, element_flows, params, *a, **k):
        if self.__dict__.get("_reads_param_operands"):
            inner["ele"].add(params.dtype)
        return real_ele(self, node_flows, element_flows, params, *a, **k)

    def par(self, node_flows, params, *a, **k):
        if self.__dict__.get("_reads_param_operands"):
            inner["par"].add(params.dtype)
        return real_par(self, node_flows, params, *a, **k)
    monkeypatch.setattr(SumLayer, "_backward_block_sparse_ele_flows", ele)
    monkeypatch.setattr(SumLayer, "_backward_block_sparse_par_flows", par)
    _step(pc, torch.randint(0, E, (128, T), device = DEV))
    if arm not in offered:
        pytest.skip(f"no {arm} arm at this shape")
    assert inner == {"ele": {torch.float32}, "par": {torch.float32}}, inner
    if arm == "triton":
        assert not seen, seen
    else:
        assert seen == {(_ELE[arm], torch.bfloat16 if arm == "cuda_stream" else torch.float32)}, seen


_GATED_ELE = {"eles": ("get_ele_stream_module", "blockscale_ele_backward_stream"),
              "elep": ("get_ele_prepass_module", "blockscale_ele_backward_prepass"),
              "cute": ("get_ele_bw_module", "blockscale_ele_backward"),
              "sbm": ("get_sb_ele_mma_module", "blockscale_sb_ele_backward_mma")}


@cuda_only
@pytest.mark.parametrize("kind", ["eles", "elep", "cute", "sbm"])
def test_the_gated_element_kinds_read_what_their_products_take(kind, monkeypatch):
    """The tied gated chain's backward with the copy, each element kind taken in turn: the split element backward
    (bf16 parameters only) is given the copy; the fp16 kinds (the CuTe fork, its pre-pass copy) and the tensor-core
    small-batch one (fp32 parameters as bf16 hi + lo halves) the fp32 parameters. Batch 64, 32 for the small-batch
    kind."""
    import pyjuice.parameterizations.kernels.c as bk
    getter, name = _GATED_ELE[kind]
    mod = getattr(bk, getter)()
    if mod is None:
        pytest.skip(f"no {kind} kind on this toolchain")
    seen, offered = set(), set()
    _spy(monkeypatch, mod, name, 4, seen)
    _prefer(monkeypatch, {kind, "stream"}, offered)
    root, gated = _gated_chain(L = 512)
    pc = juice.compile(root, verbose = False).to(DEV)
    B = 32 if kind == "sbm" else 64
    gates = _gates(gated, B)
    _step(pc, torch.randint(0, 16, (B, 5), device = DEV), sum_external_params = gates,
          sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
    if kind not in offered:
        pytest.skip(f"no {kind} kind at this shape")
    assert seen == {(name, torch.bfloat16 if kind == "eles" else torch.float32)}, seen


@cuda_only
def test_the_split_element_backwards_tuning_times_it_on_the_copy(monkeypatch):
    """The split element backward's (tile, split) choice (`_eles_code`: `autotune.pick`, its cache emptied, its trials
    run) times the kernel on the copy, as it will run: every call, the trials' among them, given bf16 parameters."""
    import pyjuice.parameterizations.kernels.c as bk
    import pyjuice.layer.kernels.c as ck
    from pyjuice.parameterizations import BlockScale
    mod = bk.get_ele_stream_module()
    if mod is None:
        pytest.skip("no split element backward on this toolchain")
    calls, in_trials = [], []
    real_kernel, real_code, real_kind = mod.blockscale_ele_backward_stream, BlockScale._eles_code, ck.autotune

    def kernel(*a, **k):
        calls.append((bool(in_trials), a[4].dtype))
        return real_kernel(*a, **k)

    def code(self, *a, **k):
        in_trials.append(True)
        try:
            return real_code(self, *a, **k)
        finally:
            in_trials.pop()

    def kind(cands, *a, **k):               # (the element plan takes the split element backward where offered)
        key = next((key for key, _ in cands if isinstance(key, tuple) and key[0] == "eles"), None)
        return key if key is not None else real_kind(cands, *a, **k)
    monkeypatch.setattr(mod, "blockscale_ele_backward_stream", kernel)
    monkeypatch.setattr(BlockScale, "_eles_code", code)
    monkeypatch.setattr(ck, "autotune", kind)
    monkeypatch.setattr(autotune, "_CACHE", {})
    root, gated = _gated_chain(L = 512)
    pc = juice.compile(root, verbose = False).to(DEV)
    B = 64
    gates = _gates(gated, B)
    _step(pc, torch.randint(0, 16, (B, 5), device = DEV), sum_external_params = gates,
          sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
    if not any(t for t, _ in calls):
        pytest.skip("one (tile, split) candidate at this shape: no trial")
    assert any(not t for t, _ in calls) and {d for _, d in calls} == {torch.bfloat16}, calls


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "hmm_b32", "gated", "gated_b32"])
def test_a_circuit_taking_no_stream_kernel_computes_the_bits_it_computed_without_it(kind, monkeypatch):
    """Every tuner steered to the same non-stream kinds in a circuit with the copy and one without it (the bf16 kernels
    that round the parameters anyway, the fp16 and fp32 ones): node values, node flows, parameter flows and the gate
    gradient as one circuit's twice over (the parameter flows' and the gate gradient's sums take atomics: MEASURED, the
    gate gradient of the gated chain without the copy 2.2e-8 apart from itself, 1.3e-6 of its largest entry; with the
    copy 1.1e-8 and 1.5e-8 from it) -- the copy changes no kernel's arithmetic but the stream kernels'."""
    kinds = {"cuda", "cuda_pipe", "cuda_prepass", "cuda_mma", "cute", "pre", "elep", "sbmma", "sbm"}
    offered = set()
    _prefer(monkeypatch, kinds, offered)
    B = 32 if kind.endswith("b32") else (128 if kind.startswith("hmm") else 64)
    out = []
    for copy in (False, True):
        if kind.startswith("hmm"):
            pc, T, E = _hmm(1024)
            x, kw = torch.randint(0, E, (B, T), device = DEV, generator = torch.Generator(device = DEV).manual_seed(3)), {}
        else:
            root, gated = _gated_chain(L = 512)
            pc = juice.compile(root, verbose = False).to(DEV)
            x = torch.randint(0, 16, (B, 5), device = DEV, generator = torch.Generator(device = DEV).manual_seed(3))
            gates = _gates(gated, B)
            kw = dict(sum_external_params = gates,
                      sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
        if not copy:
            _no_copy(pc)
        _step(pc, x, **kw)
        out.append((pc.node_mars.clone(), pc.node_flows.clone(), pc.param_flows.clone(),
                    [g.clone() for g in kw.get("sum_external_params_grad", {}).values()]))
        assert copy or not offered & {"cuda_stream", "stream", "eles"}
    assert offered & kinds, offered
    (m0, f0, p0, g0), (m1, f1, p1, g1) = out
    assert torch.equal(m0, m1) and torch.equal(f0, f1)
    assert torch.allclose(p0, p1, rtol = 1e-6, atol = 0)
    assert all(torch.allclose(a, b, rtol = 1e-5, atol = 1e-5 * float(a.abs().max())) for a, b in zip(g0, g1))


@cuda_only
@pytest.mark.parametrize("kind", ["stream", "pre"])
def test_the_gate_gradient_sums_sigma_from_what_the_forward_summed(kind, monkeypatch):
    """BlockScale's `sigma` (the parameters' per-gate sums: the split forward's `log Z`, the gate gradient's `log Z`
    term) in the backward from the copy where the forward took the split forward on it -- the gradient of the `log Z`
    it computed -- else, as without the copy, from the fp32 parameters (the pre-pass fork's `log Z` its own
    products')."""
    from pyjuice.parameterizations import BlockScale
    import pyjuice.parameterizations.kernels.c as bk
    if kind == "stream" and bk.get_stream_module() is None:
        pytest.skip("no split forward on this toolchain")
    root, gated = _gated_chain(L = 512)
    pc = juice.compile(root, verbose = False).to(DEV)
    dtypes = []
    real = BlockScale._sigma
    monkeypatch.setattr(BlockScale, "_sigma",
                        lambda self, layer, params, *a, **k: dtypes.append(params.dtype) or real(self, layer, params, *a, **k))
    offered = []
    _force_fork(monkeypatch, kind, offered)
    B = 64
    gates = _gates(gated, B)
    x = torch.randint(0, 16, (B, 5), device = DEV)
    pc.init_param_flows(flows_memory = 0.0)
    pc(x, sum_external_params = gates)
    if not any(key[0] == kind for call in offered for key in call):
        pytest.skip(f"no {kind} fork at this shape")
    assert (kind != "stream") or set(dtypes) == {torch.bfloat16}, dtypes          # (the split forward's own)
    dtypes.clear()
    pc.backward(x, flows_memory = 1.0, sum_external_params = gates,
                sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
    assert dtypes and set(dtypes) == {torch.bfloat16 if kind == "stream" else torch.float32}, dtypes


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "gated"])
def test_a_backward_with_and_without_the_copy_in_turn(kind, monkeypatch):
    """Calls with the copy and without it (another precision) in turn, each planned on its own (the plans keyed on it),
    and a backward whose forward made no copy making it (the parameters written through `params.data` after that
    forward): each call's flows are those of a fresh circuit making the same call (the tuners steered alike in both,
    `_prefer`: two circuits tuning on their own may take arms a few ulps apart)."""
    _prefer(monkeypatch, {"cuda_stream", "stream", "eles"}, set())

    def make():
        if kind == "hmm":
            pc, T, E = _hmm(1024)
            return pc, torch.randint(0, E, (128, T), device = DEV), {}
        root, gated = _gated_chain(L = 512)
        pc = juice.compile(root, verbose = False).to(DEV)
        gates = _gates(gated, 64)
        return pc, torch.randint(0, 16, (64, 5), device = DEV), dict(
            sum_external_params = gates, sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
    torch.manual_seed(0)
    pc, x, kw = make()
    for precision in ("auto", "auto", "tf32", "tf32", "auto", "auto"):
        pc.precision = precision
        _step(pc, x, **kw)
        torch.manual_seed(0)
        ref, _, rkw = make()
        with torch.no_grad():
            ref.params.data.copy_(pc.params.data)
        ref.precision = precision
        _step(ref, x, **rkw)
        assert torch.equal(pc.node_flows, ref.node_flows), precision
    # a forward without the copy (tf32), the parameters written, the backward with it (auto): the copy made again there
    pc.precision = "tf32"
    pc.init_param_flows(flows_memory = 0.0)
    with torch.no_grad():
        pc(x, **{k: v for k, v in kw.items() if k == "sum_external_params"})
        (a, b), = pc._param_operand_plan()[1]
        pc.params.data[a:b] *= 1.5
    pc.precision = "auto"
    pc.backward(x, flows_memory = 1.0, **kw)
    assert torch.equal(pc._param_operands(x.size(0), refresh = False)[a:b], pc.params.detach()[a:b].to(torch.bfloat16))


@cuda_only
def test_a_tied_layers_backward_is_its_bf16_twins(monkeypatch):
    """Parameters on the bf16 grid (a bf16-stored twin then holds the same values), the stream kernels taken where
    offered and every other choice the first candidate in both: the fp32 circuit's forward and backward on the copy and
    the twin's -- node values, node flows and parameter flows to the bit."""
    pc16, T, E = _hmm(1024, dtype = torch.bfloat16)
    pc32, _, _ = _hmm(1024)
    with torch.no_grad():
        pc32.params.data.copy_(pc16.params.data.float())
    offered = set()
    _prefer(monkeypatch, {"cuda_stream"}, offered)
    x = torch.randint(0, E, (128, T), device = DEV)
    for pc in (pc32, pc16):
        _step(pc, x)
    assert "cuda_stream" in offered, offered
    assert torch.equal(pc32.node_mars, pc16.node_mars)
    assert torch.equal(pc32.node_flows, pc16.node_flows)
    # (the parameter flows: the fp32 parameters' kernels, the twin's the bf16 ones -- on the same values, summed in
    # another order: MEASURED 0.03% of them 1-2 ulps apart)
    assert torch.allclose(pc32.param_flows, pc16.param_flows, rtol = 1e-6, atol = 0)


@cuda_only
def test_a_tied_gated_layers_backward_is_its_bf16_twins(monkeypatch):
    """The same for the tied gated chain: BlockScale's split forward and split element backward (`eles`) taken where
    offered -- node values, node flows and the gate gradient (its `log Z` term's `sigma` summed from the copy, as from
    the twin's stored values) to the bit, the parameter flows to their summation order. Blocks of 128 (`L = 512`): the
    split element backward's scope."""
    root, gated = _gated_chain(L = 512)
    pc32 = juice.compile(root, verbose = False).to(DEV)
    pc16 = juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(DEV)
    with torch.no_grad():
        pc32.params.data.copy_(pc16.params.data.float())
    offered = set()
    _prefer(monkeypatch, {"cuda_stream", "stream", "eles"}, offered)
    B = 64
    x = torch.randint(0, 16, (B, 5), device = DEV)
    gates = _gates(gated, B)
    grads = []
    for pc in (pc32, pc16):
        g = {n: torch.zeros_like(t) for n, t in gates.items()}
        _step(pc, x, sum_external_params = gates, sum_external_params_grad = g)
        grads.append(g)
    assert {"stream", "eles"} <= offered, offered
    assert torch.equal(pc32.node_mars, pc16.node_mars)
    assert torch.equal(pc32.node_flows, pc16.node_flows)
    assert torch.allclose(pc32.param_flows, pc16.param_flows, rtol = 1e-6, atol = 0)     # (as the plain layers')
    for n in gates:
        assert torch.equal(grads[0][n], grads[1][n])


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "gated"])
def test_a_replayed_backward_reads_the_copy_as_the_full_path_does(kind, monkeypatch):
    """A backward's later calls replay its launch plan (`pyjuice.utils.launch_plans`), whose element launchers are given
    `params` and read the copy (`SumLayer._bk_copy`): the flows of a replayed call are the full path's, the stream
    kernels taken (a bf16 kernel given the fp32 parameters would read garbage)."""
    offered = set()
    _prefer(monkeypatch, {"cuda_stream", "stream", "eles"}, offered)
    if kind == "hmm":
        pc, T, E = _hmm(1024)
        x, kw = torch.randint(0, E, (128, T), device = DEV), {}
    else:
        root, gated = _gated_chain(L = 512)
        pc = juice.compile(root, verbose = False).to(DEV)
        x = torch.randint(0, 16, (64, 5), device = DEV)
        gates = _gates(gated, 64)
        kw = dict(sum_external_params = gates, sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
    flows = []
    for _ in range(3):
        _step(pc, x, **kw)
        flows.append((pc.node_flows.clone(), pc.param_flows.clone()))
    assert offered & {"cuda_stream", "eles"}, offered
    for nf, pf in flows[1:]:
        assert torch.equal(nf, flows[0][0])
        assert torch.allclose(pf, flows[0][1], rtol = 1e-6, atol = 0)


@cuda_only
@pytest.mark.parametrize("kind", ["hmm", "gated"])
def test_a_backward_without_the_copy_at_the_same_signature_is_planned_apart(kind, monkeypatch):
    """The circuit's backward planned with the copy, then the same calls -- batch, precision, dtypes -- without it (as a
    caller driving the layers without the circuit's copy would make them): the copy is in the plan keys, so they are
    planned on their own and compute on the fp32 parameters, as a circuit that never made the copy does (a plan made
    with the copy, replayed without it, would hand its stream kernels the fp32 parameters)."""
    _prefer(monkeypatch, {"cuda_stream", "stream", "eles"}, set())

    def make():
        torch.manual_seed(0)
        if kind == "hmm":
            pc, T, E = _hmm(1024)
            return pc, torch.randint(0, E, (128, T), device = DEV), {}
        root, gated = _gated_chain(L = 512)
        pc = juice.compile(root, verbose = False).to(DEV)
        gates = _gates(gated, 64)
        return pc, torch.randint(0, 16, (64, 5), device = DEV), dict(
            sum_external_params = gates, sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
    pc, x, kw = make()
    for _ in range(2):
        _step(pc, x, **kw)                                   # planned with the copy
    _no_copy(pc)
    ref, _, rkw = make()
    _no_copy(ref)
    for _ in range(2):
        _step(pc, x, **kw)
        _step(ref, x, **rkw)
        assert torch.equal(pc.node_flows, ref.node_flows)


# ---------------------------------------------------------------------------------------------------------------------
# The rule: the copy taken from a work per layer on
# ---------------------------------------------------------------------------------------------------------------------

@cuda_only
def test_the_copy_is_taken_from_a_work_per_layer_on(monkeypatch):
    """By the rule (`_param_operand_groups`, not overridden), a pass takes the copy where the parameters each layer
    given it reads a sample, times the batch, reach `_COPY_MIN_WORK` -- a tied HMM-1024 at half that batch not, at it
    yes, the backward as its forward, every call the same --, an untied circuit never."""
    monkeypatch.setattr(tcm.TensorCircuit, "_param_operands_force", None)
    pc, T, E = _hmm(1024)
    pc._param_operand_plan()
    layer = _marked(pc)[0]
    at = tcm._COPY_MIN_WORK // layer.num_edges                       # (the batch at the threshold)
    assert at // 2 >= 16 and at * layer.num_edges == tcm._COPY_MIN_WORK
    for B, taken in ((at // 2, False), (at, True), (at // 2, False), (at, True)):
        _step(pc, torch.randint(0, E, (B, T), device = DEV))
        assert pc.__dict__["_param_operands_made"] is taken, B
        assert (layer.__dict__["_bk_op_params"] is not None) is taken, B
    # (an untied circuit: no layer given the copy, at any batch)
    xs = torch.randint(0, 16, (256, 12))
    hclt = juice.compile(juice.structures.HCLT(xs.float(), num_latents = 32, input_dist = dists.Categorical(num_cats = 16)),
                         verbose = False).to(DEV)
    assert not hclt._param_operand_groups(1 << 20)


def _two_size_chain(C = 16):
    """A chain of 128 latents (tied transitions) feeding, through one untied transition, a chain of 512 (tied): tied
    groups of two sizes, 2^14 and 2^18 parameters a layer."""
    from pyjuice.nodes import inputs, multiply, summate
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ch, first = inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = C)), None
        for t in range(1, 4):
            emit = inputs(t, num_node_blocks = 4, dist = dists.Categorical(num_cats = C))
            ch = first = (summate(multiply(ch, emit), num_node_blocks = 4) if first is None
                          else first.duplicate(multiply(ch, emit), tie_params = True))
        ch = summate(multiply(ch, inputs(4, num_node_blocks = 4, dist = dists.Categorical(num_cats = C))),
                     num_node_blocks = 16)
        first = None
        for t in range(5, 9):
            emit = inputs(t, num_node_blocks = 16, dist = dists.Categorical(num_cats = C))
            ch = first = (summate(multiply(ch, emit), num_node_blocks = 16) if first is None
                          else first.duplicate(multiply(ch, emit), tie_params = True))
        root = summate(multiply(ch), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root


@cuda_only
def test_the_rule_gives_the_copy_group_by_group(monkeypatch):
    """Tied groups of two sizes, at a batch where the larger ones' work reaches `_COPY_MIN_WORK` (the threshold set
    there) and the smaller ones' does not: only the larger groups' layers are given the copy, in the forward and the
    backward alike."""
    monkeypatch.setattr(tcm.TensorCircuit, "_param_operands_force", None)
    pc = juice.compile(_two_size_chain(), verbose = False).to(DEV)
    pc._param_operand_plan()
    sizes = sorted({layer.num_edges for layer in _marked(pc)})
    assert sizes == [1 << 14, 1 << 18], sizes
    B = 64
    monkeypatch.setattr(tcm, "_COPY_MIN_WORK", (1 << 18) * B)
    given = {}
    real_fw, real_bw = SumLayer.forward, SumLayer.backward

    def fw(self, *a, op_params = None, **k):
        given.setdefault(("forward", self.num_edges), set()).add(op_params is not None)
        return real_fw(self, *a, op_params = op_params, **k)

    def bw(self, *a, op_params = None, **k):
        given.setdefault(("backward", self.num_edges), set()).add(op_params is not None)
        return real_bw(self, *a, op_params = op_params, **k)
    monkeypatch.setattr(SumLayer, "forward", fw)
    monkeypatch.setattr(SumLayer, "backward", bw)
    _step(pc, torch.randint(0, 16, (B, 9), device = DEV))
    for p in ("forward", "backward"):
        assert given[(p, 1 << 18)] == {True} and given[(p, 1 << 14)] == {False}, (p, given)


@cuda_only
@pytest.mark.parametrize("least", ["blockscale", "none"])
def test_a_parameterization_sets_the_least_layer_that_takes_the_copy(least, monkeypatch):
    """A tied gated chain whose work per layer reaches `_COPY_MIN_WORK` at batch 64 (the rule's threshold set there):
    its layers, smaller than BlockScale's `param_operands_min_edges`, are given no copy; without that minimum they
    are."""
    from pyjuice.parameterizations import BlockScale
    monkeypatch.setattr(tcm.TensorCircuit, "_param_operands_force", None)
    if least == "none":
        monkeypatch.setattr(BlockScale, "param_operands_min_edges", 0)
    root, gated = _gated_chain(L = 512)
    pc = juice.compile(root, verbose = False).to(DEV)
    pc._param_operand_plan()
    layer = _marked(pc)[0]
    B = 64
    monkeypatch.setattr(tcm, "_COPY_MIN_WORK", layer.num_edges * B)
    assert layer.num_edges < BlockScale.param_operands_min_edges or least == "none"
    gates = _gates(gated, B)
    _step(pc, torch.randint(0, 16, (B, 5), device = DEV), sum_external_params = gates,
          sum_external_params_grad = {n: torch.zeros_like(t) for n, t in gates.items()})
    taken = least == "none"
    assert pc.__dict__["_param_operands_made"] is taken and (layer.__dict__["_bk_op_params"] is not None) is taken


@cuda_only
def test_below_the_work_the_forward_makes_no_copy(monkeypatch):
    """Below `_COPY_MIN_WORK` neither pass makes the copy or hands it to a layer: not even its buffer is made."""
    monkeypatch.setattr(tcm.TensorCircuit, "_param_operands_force", None)
    pc, T, E = _hmm(1024)
    pc._param_operand_plan()
    layer = _marked(pc)[0]
    B = tcm._COPY_MIN_WORK // layer.num_edges // 2
    given = []
    real_fw, real_bw = SumLayer.forward, SumLayer.backward
    monkeypatch.setattr(SumLayer, "forward", lambda self, *a, op_params = None, **k: given.append(op_params is not None)
                        or real_fw(self, *a, op_params = op_params, **k))
    monkeypatch.setattr(SumLayer, "backward", lambda self, *a, op_params = None, **k: given.append(op_params is not None)
                        or real_bw(self, *a, op_params = op_params, **k))
    _step(pc, torch.randint(0, E, (B, T), device = DEV))
    assert given and not any(given) and "_param_operands_" not in pc.__dict__


@cuda_only
def test_a_recorded_graph_is_its_choices_own(monkeypatch):
    """A pass recorded into a CUDA graph with the copy and the same pass without it are graphs of their own (the choice
    is in the key): the override flipped between training steps, the replayed circuit computes each step's node flows
    as an eager one does, to the bit (every parameter given to it in place before the step)."""
    _prefer(monkeypatch, {"cuda_stream"}, set())
    pcs = [_hmm(1024)[0] for _ in range(2)]
    pcs[0].cuda_graphs = True
    B, T, E = 128, 8, 64
    x = torch.randint(0, E, (B, T), device = DEV)
    for step, force in enumerate((True, True, False, False, True, False)):
        monkeypatch.setattr(tcm.TensorCircuit, "_param_operands_force", force)
        with torch.no_grad():
            for p0, p1 in zip(pcs[0].parameters(), pcs[1].parameters()):
                if p0.is_floating_point():
                    p0.data.copy_(p1.data)
        for pc in pcs:
            pc.init_param_flows(flows_memory = 0.0)
            pc(x)
            pc.backward(x, flows_memory = 1.0)
        assert torch.equal(pcs[0].node_flows, pcs[1].node_flows), step
        for pc in pcs:
            pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    assert len(pcs[0]._recorded_cuda_graphs) == 4 and not pcs[1]._recorded_cuda_graphs   # (fwd, bwd x with, without)
