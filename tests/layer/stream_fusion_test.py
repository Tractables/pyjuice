"""
The product layer fused into the stream forward's pre-pass (`tlmm_forward_sum_stream_fused.cu`, `SumLayer.forward`'s
`stream_children`, `TensorCircuit._stream_fusion_groups`): one more option the tuners choose, per batch size, between a
product layer's forward followed by the sum layer's stream forward and the sum layer's fused stream forward alone. The
fused copy forms each child's value as the sum of its product's two child node values -- the product layer's own
arithmetic --, so the two compute the same bits; where the fused copy is taken, the circuit launches no forward for the
product layer below it.

And into the backward (`ele_backward_sum_stream_fused.cu`, `par_deferred_fp16_fused.cu`, `SumLayer.backward`'s
`stream_children`, `TensorCircuit._stream_fusion_bk_groups`): the sum layer's element flows and its deferred parameter
flows' capture form each element's value from its two children, and the merge writes the element's flow as its two
children's node flows -- what the product layer's recompute before and backward after the sum layer's give where each
child has one parent --, so neither product launch is made: the same bits.
"""

import subprocess

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
import pyjuice.layer.kernels.autotune as autotune
import pyjuice.layer.kernels.c as cuda_kernels
from pyjuice.layer.sum_layer import SumLayer
from pyjuice.layer.prod_layer import ProdLayer

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _hmm(L, T = 8, E = 64, dtype = torch.bfloat16, block_size = None):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = L, num_emits = E, homogeneous = True,
                              block_size = block_size or min(L // 4, 1024))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = dtype).to(DEV), T, E


def _unfused(pc):
    pc._stream_fusion_groups = lambda *a, **k: frozenset()
    pc._stream_fusion_bk_groups = lambda *a, **k: frozenset()
    return pc


def _fused_groups(pc, B, state = "_stream_fusion_"):
    for key, v in pc.__dict__.get(state, {}).items():
        if key[0] == B and isinstance(v, frozenset):
            return v
    return None


@pytest.fixture
def stream_steered(monkeypatch):
    """The plain forward's CUDA choice and the element backward's steered to the stream kernel where it is offered (the
    tuners' timing decides them otherwise -- each layer times its own element kernels, `_cached_bk_ele_choice`, so two
    circuits may take arms whose bits differ; the tests check what the choices compute), nothing kept from an earlier
    test."""
    real_choose, real_best_of = autotune.choose, autotune.best_of

    def choose(key, cands):
        if key[0] == "sum_fw_tlmm_cuda":
            stream = [(c, run) for c, run in cands if c[0] == "cuda_stream"]
            if stream:
                return real_choose(key, stream[:1])           # (cached as the tuner caches its choices)
        return real_choose(key, cands)
    monkeypatch.setattr(autotune, "_CACHE", {})
    monkeypatch.setattr(autotune, "choose", choose)
    monkeypatch.setattr(autotune, "best_of", lambda cands, *a, **k: next(
        (key for key, _ in cands if isinstance(key, tuple) and key[:1] == ("cuda_stream",)), None)
        or real_best_of(cands, *a, **k))


@pytest.fixture
def fused_preferred(stream_steered, monkeypatch):
    """`stream_steered`, and the fusion decisions -- forward and backward -- steered to the fused copies."""
    real_pick = autotune.pick
    monkeypatch.setattr(autotune, "pick", lambda key, cands, bench, *a, **k: "fused"
                        if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk") else real_pick(key, cands, bench, *a, **k))


def _needs_stream():
    if not (cuda_kernels.stream_is_available() and cuda_kernels.stream_fused_is_available()):
        pytest.skip("no stream forward on this toolchain")


@cuda_only
@pytest.mark.parametrize("L, B, dtype", [(4096, 128, torch.bfloat16), (2048, 256, torch.bfloat16),
                                         (4096, 128, torch.float32)])
def test_the_fused_forward_computes_the_unfused_bits(L, B, dtype, fused_preferred):
    """Two circuits, one never fusing: forward after forward, node values and log-likelihoods to the bit -- the first
    forward at a batch size unfused in both (its layers settle), the later ones with the products fused into the
    stream forward of the layers above them (fp32 parameters through their bf16 copy, `_param_operands`)."""
    _needs_stream()
    pcs = [_hmm(L, dtype = dtype)[0], _unfused(_hmm(L, dtype = dtype)[0])]
    T, E = 8, 64
    x = torch.randint(0, E, (B, T), device = DEV, generator = torch.Generator(device = DEV).manual_seed(1))
    with torch.no_grad():
        for it in range(3):
            lls = [pc(x).clone() for pc in pcs]
            assert torch.equal(lls[0], lls[1]) and torch.equal(pcs[0].node_mars, pcs[1].node_mars), it
            fused = _fused_groups(pcs[0], B)
            assert (fused is None) == (it == 0), it
    n_prod = sum(group.is_prod() for group in pcs[0].inner_layer_groups)
    assert len(fused) >= n_prod - 2, (len(fused), n_prod)        # (every product below a transition layer)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_training_steps_compute_the_unfused_bits(dtype, fused_preferred):
    """Training steps (forward, backward, EM) of a circuit fusing -- forward and backward -- and one never fusing,
    every parameter given to the fusing one in place before each step: each step's node values, node flows and
    parameter flows to the bit (bf16 parameters, and fp32 ones through their bf16 copy). (Not over several steps
    unsynced: the input layers' flows are summed with atomics, and two unfused circuits' EM updates already drift
    apart by ulps.)"""
    _needs_stream()
    pcs = [_hmm(4096, dtype = dtype)[0], _unfused(_hmm(4096, dtype = dtype)[0])]
    B, T, E = 128, 8, 64
    x = torch.randint(0, E, (B, T), device = DEV, generator = torch.Generator(device = DEV).manual_seed(2))
    for step in range(4):
        with torch.no_grad():
            for p0, p1 in zip(pcs[0].parameters(), pcs[1].parameters()):
                if p0.is_floating_point():
                    p0.data.copy_(p1.data)
        for pc in pcs:
            pc.init_param_flows(flows_memory = 0.0)
            pc(x)
            pc.backward(x, flows_memory = 1.0)
        assert torch.equal(pcs[0].node_mars, pcs[1].node_mars) and torch.equal(pcs[0].node_flows, pcs[1].node_flows), step
        assert torch.equal(pcs[0].param_flows, pcs[1].param_flows), step
        for pc in pcs:
            pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    assert _fused_groups(pcs[0], B) and _fused_groups(pcs[0], B, "_stream_fusion_bk_")


@cuda_only
def test_a_fused_pass_launches_no_product_forward(fused_preferred, monkeypatch):
    """Where the fused copy is taken, the forward launches no product layer forward below it, and every product layer it
    does launch is one whose sum group above was not fused."""
    _needs_stream()
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    with torch.no_grad():
        pc(x)
        pc(x)                                                    # (decided here)
        ran = []
        real = ProdLayer.forward
        monkeypatch.setattr(ProdLayer, "forward", lambda self, *a, **k: ran.append(self) or real(self, *a, **k))
        pc(x)
    fused = _fused_groups(pc, B)
    groups = pc.inner_layer_groups
    launched = {k for k, g in enumerate(groups) if g.is_prod() and any(layer in ran for layer in g)}
    assert fused and not launched & fused
    assert launched == {k for k, g in enumerate(groups) if g.is_prod()} - fused


@cuda_only
def test_the_fused_copy_is_one_more_option_of_the_tuners(stream_steered, monkeypatch):
    """The decision is the tuners' (`autotune.pick`, keyed by the pair's shape and the batch size: one trial for the 30
    equal pairs of an HMM): offered the fused copy and the unfused pair, the unfused first, and taken as it chooses --
    the unfused pair chosen, no product is left out."""
    _needs_stream()
    offered, real = [], autotune.pick

    def pick(key, cands, bench, *a, **k):
        if key[0] == "prod_stream_fusion":
            offered.append((key, list(cands)))
            return "unfused"
        return real(key, cands, bench, *a, **k)
    monkeypatch.setattr(autotune, "pick", pick)
    pc, T, E = _hmm(4096)
    for B in (64, 128):
        x = torch.randint(0, E, (B, T), device = DEV)
        with torch.no_grad():
            for _ in range(3):
                pc(x)
        assert _fused_groups(pc, B) == frozenset()
    assert offered and all(c == ["unfused", "fused"] for _, c in offered)
    assert {key[1] for key, _ in offered} == {64, 128}


@cuda_only
def test_no_fusion_where_the_stream_forward_does_not_run(fused_preferred, monkeypatch):
    """Layers whose settled forward is another kernel (the tlmm choice steered to the pre-pass kernel): nothing fused."""
    _needs_stream()
    real = autotune.choose

    def choose(key, cands):
        if key[0] == "sum_fw_tlmm_cuda":
            return real(key, [next((c, run) for c, run in cands if c[0] == "cuda_prepass")])
        return real(key, cands)
    monkeypatch.setattr(autotune, "choose", choose)
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    with torch.no_grad():
        for _ in range(3):
            pc(x)
    assert _fused_groups(pc, B) == frozenset()


@cuda_only
def test_no_fusion_for_a_pass_of_another_regime(fused_preferred):
    """A circuit whose plain forwards fuse: asked for a non-LL or tempered forward or one with keyword arguments, nothing
    fused; such a forward computes what a circuit never fusing computes."""
    _needs_stream()
    pcs = [_hmm(4096)[0], _unfused(_hmm(4096)[0])]
    B, T, E = 128, 8, 64
    x = torch.randint(0, E, (B, T), device = DEV)
    with torch.no_grad():
        for _ in range(2):
            for pc in pcs:
                pc(x)
    pc = pcs[0]
    assert _fused_groups(pc, B)
    plain = dict(precision = pc.precision, pflow_temperature = 1.0)
    og = pc._param_operand_groups(B)
    assert pc._stream_fusion_groups(B, plain, "LL", og) == _fused_groups(pc, B)
    for opts, alg in ((plain, "MPE"), (dict(plain, pflow_temperature = 0.5), "LL"), (dict(plain, foo = 1), "LL")):
        assert pc._stream_fusion_groups(B, opts, alg, og) == frozenset(), (opts, alg)
    with torch.no_grad():
        for kw in (dict(propagation_alg = "MPE"), dict(pflow_temperature = 0.5)):
            outs = [pc(x, **kw).clone() for pc in pcs]
            assert torch.equal(outs[0], outs[1]) and torch.equal(pcs[0].node_mars, pcs[1].node_mars), kw


@cuda_only
def test_external_parameter_layers_are_not_fused(fused_preferred):
    """A tied chain of BlockScale-gated layers (subclasses of `SumLayer`, their forward their parameterization's): nothing
    fused, the node values a circuit never fusing computes."""
    _needs_stream()
    from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
    B, T, L, C = 128, 5, 4096, 16

    def gated():
        torch.manual_seed(0)
        nodes = []
        with juice.set_block_size(1024):
            chain = inputs(0, num_node_blocks = L // 1024, dist = dists.Categorical(num_cats = C))
            first = None
            for t in range(1, T):
                emit = inputs(t, num_node_blocks = L // 1024, dist = dists.Categorical(num_cats = C))
                if first is None:
                    chain = first = summate(multiply(chain, emit), num_node_blocks = L // 1024,
                                            external_params = BlockScaleSumParams(ch_block_size = 32))
                else:
                    chain = first.duplicate(multiply(chain, emit), tie_params = True)
                nodes.append(chain)
            root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)
        root.init_parameters(perturbation = 2.0)
        return juice.compile(root, verbose = False, param_dtype = torch.bfloat16).to(DEV), nodes
    (pc, nodes), (pc2, nodes2) = gated(), gated()
    _unfused(pc2)
    g = torch.Generator(device = DEV).manual_seed(2)
    gates = [torch.randn(n.external_params.tensor_shapes(n, B)[0], device = DEV, generator = g) * 0.5 - 0.5 for n in nodes]
    x = torch.randint(0, C, (B, T), device = DEV)
    with torch.no_grad():
        for _ in range(3):
            outs = [c(x, external_params = {n: v for n, v in zip(ns, gates)}).clone()
                    for c, ns in ((pc, nodes), (pc2, nodes2))]
            assert torch.equal(outs[0], outs[1])
    assert not _fused_groups(pc, B)


@cuda_only
def test_the_fused_kernel_takes_a_dummy_chunk_as_minus_infinity(fused_preferred):
    """A chunk `cbase` marks dummy (-1, the padding element rows of a ragged block): the fused kernel's node values are
    the stream kernel's over those 32 element rows at -inf (every row's first edge tile's second half here; an HMM's
    layers have no dummy chunk of their own)."""
    _needs_stream()
    from pyjuice.layer.fused_children import element_children
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    with torch.no_grad():
        pc(x)
        pc(x)                                                    # (settled, decided)
        k = min(_fused_groups(pc, B))
        layer = list(pc.inner_layer_groups[k + 1])[0]
        plan = layer._stream_fusion((B, pc.params.dtype, pc.precision, False), element_children(pc)[k])
        cbase, (_, (_, c), nids, ebase, pbase, knt, groups) = plan[0]
        a, b = layer._layer_nid_range
        cb = cbase.clone()
        cb[:, 0, 1, :] = -1
        cuda_kernels.tlmm_forward_sum_stream_fused(pc.node_mars, cb, pc.params, nids, pbase, groups, B, layer.block_size,
                                                   knt, c)
        got = pc.node_mars[a:b].clone()
        pc.inner_layer_groups[k].forward(pc.node_mars, pc.element_mars)
        for r in ebase[:, 0].tolist():
            pc.element_mars[r + 32:r + 64] = -torch.inf
        cuda_kernels.tlmm_forward_sum_stream(pc.node_mars, pc.element_mars, pc.params, nids, ebase, pbase, groups, B,
                                             layer.block_size, knt, c)
        want = pc.node_mars[a:b].clone()
        cuda_kernels.tlmm_forward_sum_stream_fused(pc.node_mars, cbase, pc.params, nids, pbase, groups, B,
                                                   layer.block_size, knt, c)
        whole = pc.node_mars[a:b].clone()
    assert torch.equal(got, want)
    assert not torch.equal(got, whole)                           # (the chunk counted for something)


def test_the_plan_takes_dummy_chunks_and_runs_of_children_only():
    """`SumLayer._stream_fusion_cbase` on synthetic tables: a chunk of 32 dummy element rows (-1), a chunk of rows with
    two runs of consecutive children (their first rows); none where a chunk is part dummy, its children are not runs,
    or it runs past the table."""
    table = torch.zeros([128, 2], dtype = torch.int32)
    table[:, 0] = torch.arange(1000, 1128)
    table[:, 1] = torch.arange(5000, 5128)
    first = 64
    # two tiles of one row: a dummy tile (rows 0..63) and the table's first rows
    ebase = torch.tensor([[0, 64]])
    cb = SumLayer._stream_fusion_cbase(ebase, table, first)
    assert cb.shape == (1, 2, 2, 2)
    assert cb[0, 0].tolist() == [[-1, -1], [-1, -1]]
    assert cb[0, 1].tolist() == [[1000, 5000], [1032, 5032]]
    assert SumLayer._stream_fusion_cbase(torch.tensor([[48]]), table, first) is None          # (a chunk part dummy)
    assert SumLayer._stream_fusion_cbase(torch.tensor([[64 + 96]]), table, first) is None     # (past the table)
    broken = table.clone()
    broken[5, 1] = 7
    assert SumLayer._stream_fusion_cbase(torch.tensor([[64]]), broken, first) is None         # (not a run)


@cuda_only
def test_stream_children_takes_a_plain_call_with_a_plan(stream_steered):
    """`forward(stream_children = ...)` on a call that is not plain, before the layer's forward settled, or over products
    of other than two children: an error."""
    _needs_stream()
    pc, T, E = _hmm(4096)
    B = 128
    layer = [l for g in pc.inner_layer_groups if g.is_sum() for l in g][1]
    from pyjuice.layer.fused_children import element_children
    tables = element_children(pc)
    k = [i for i, g in enumerate(pc.inner_layer_groups) if layer in list(g)][0]
    x = torch.randint(0, E, (B, T), device = DEV)
    table, first = tables[k - 1]
    with torch.no_grad():
        pc(x)
        layer.forward(pc.node_mars, pc.element_mars, pc.params, stream_children = (table, first))   # (settled: a plan)
        with pytest.raises(ValueError):
            layer.forward(pc.node_mars, pc.element_mars, pc.params, propagation_alg = "MPE", stream_children = tables[k - 1])
        with pytest.raises(ValueError):                                    # (products of three children)
            layer.forward(pc.node_mars, pc.element_mars, pc.params,
                          stream_children = (torch.cat([table, table[:, :1]], 1), first))
        with pytest.raises(ValueError):
            layer.forward(pc.node_mars[:, :64].contiguous(), pc.element_mars[:, :64].contiguous(), pc.params,
                          stream_children = tables[k - 1])            # (batch 64: never settled)


@cuda_only
def test_a_fused_forward_replayed_from_a_graph_computes_the_eager_bits(fused_preferred):
    """A forward recorded into a CUDA graph with its products fused replays what it computes eagerly, a write to the
    parameters in between; the fusion is in the graphs' key: the unfused first forward's graph is not replayed for the
    fused ones."""
    _needs_stream()
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    with torch.no_grad():
        pc(x, record_cudagraph = True)
        n = len(pc._recorded_cuda_graphs)
        pc(x, record_cudagraph = True)
        assert _fused_groups(pc, B) and len(pc._recorded_cuda_graphs) == n + 1
        pc(x, record_cudagraph = True)
        assert len(pc._recorded_cuda_graphs) == n + 1
        pc.params.data[pc.params.numel() // 2:] *= 1.25           # (in place: the graph's buffers stay)
        replayed = pc(x).clone()
        eager = pc(x, apply_cudagraph = False).clone()
    assert torch.equal(replayed, eager)


def _decided(pc, x, steps = 2):
    for _ in range(steps):
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)


@cuda_only
def test_a_fused_backward_launches_no_product_kernels(fused_preferred, monkeypatch):
    """Where the backward fuses a product group, neither its forward (the recompute) nor its backward is launched in the
    backward; every other product group's are."""
    _needs_stream()
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    _decided(pc, x)
    pc(x)
    ran = {"forward": [], "backward": []}
    for name in ran:
        real = getattr(ProdLayer, name)
        monkeypatch.setattr(ProdLayer, name, lambda self, *a, real = real, name = name, **k: ran[name].append(self)
                            or real(self, *a, **k))
    pc.backward(x, flows_memory = 1.0)
    fused = _fused_groups(pc, B, "_stream_fusion_bk_")
    groups = pc.inner_layer_groups
    prods = {k for k, g in enumerate(groups) if g.is_prod()}
    assert fused and fused <= prods
    for name in ran:
        assert {k for k in prods if any(layer in ran[name] for layer in groups[k])} == prods - fused, name


@cuda_only
def test_the_fused_backward_is_one_more_option_of_the_tuners(stream_steered, monkeypatch):
    """The backward's decision is the tuners' (`autotune.pick`: the unfused first, margin 1.0, keyed by the pair's shape
    and the batch size), taken as chosen: the unfused pair chosen, no product left out."""
    _needs_stream()
    offered, real = [], autotune.pick

    def pick(key, cands, bench, *a, **k):
        if key[0] == "prod_stream_fusion_bk":
            offered.append((key, list(cands), k.get("margin")))
            return "unfused"
        return real(key, cands, bench, *a, **k)
    monkeypatch.setattr(autotune, "pick", pick)
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    _decided(pc, x, 3)
    assert offered and all(c == ["unfused", "fused"] and m == 1.0 for _, c, m in offered)
    assert {key[1] for key, _, _ in offered} == {B}
    assert _fused_groups(pc, B, "_stream_fusion_bk_") == frozenset()


@cuda_only
def test_no_backward_fusion_for_another_regime(fused_preferred, monkeypatch):
    """A circuit whose plain backwards fuse: asked for linear-space, modified, negated or tempered flows, another
    algorithm or keyword arguments, nothing fused; a backward with a layer callback launches every product layer."""
    _needs_stream()
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    _decided(pc, x)
    og = pc._param_operand_groups(B)
    plain = dict(param_flows = pc.param_flows, denom_param_flows = None, allow_modify_flows = False,
                 logspace_flows = True, negate_pflows = False, precision = pc.precision, pflow_temperature = 1.0,
                 temper_eflow = False, defer_param_flows = True, pf_store_first = False)
    fused = pc._stream_fusion_bk_groups(B, plain, "LL", og)
    assert fused
    for opts, alg in ((dict(plain, logspace_flows = False), "LL"), (plain, "MPE"), (dict(plain, negate_pflows = True), "LL"),
                      (dict(plain, pflow_temperature = 0.5), "LL"), (dict(plain, temper_eflow = True), "LL"),
                      (dict(plain, allow_modify_flows = True), "LL"), (dict(plain, foo = 1), "LL")):
        assert pc._stream_fusion_bk_groups(B, opts, alg, og) == frozenset(), (opts, alg)
    ran, real = [], ProdLayer.backward
    monkeypatch.setattr(ProdLayer, "backward", lambda self, *a, **k: ran.append(self) or real(self, *a, **k))
    pc(x)
    pc.backward(x, flows_memory = 1.0, sum_layer_post_backward_callback = lambda layer, **kwargs: None)
    prods = [k for k, g in enumerate(pc.inner_layer_groups) if g.is_prod()]
    assert all(any(layer in ran for layer in pc.inner_layer_groups[k]) for k in prods)


@cuda_only
def test_no_backward_fusion_where_a_child_has_another_parent(fused_preferred):
    """A product group one of whose children has another parent element is left unfused (the merge writes a child's
    flow; the product backward adds the parents' flows)."""
    _needs_stream()
    from pyjuice.layer.fused_children import element_children
    pcs = [_hmm(4096)[0], _hmm(4096)[0]]
    B, T, E = 128, 8, 64
    x = torch.randint(0, E, (B, T), device = DEV)
    _decided(pcs[0], x)
    every = _fused_groups(pcs[0], B, "_stream_fusion_bk_")
    pc = pcs[1]
    tables = element_children(pc)
    parents = torch.zeros(pc.num_nodes, dtype = torch.int32, device = DEV)
    for t in tables:
        if t is not None:
            parents.index_add_(0, t[0].reshape(-1).long(), torch.ones_like(t[0].reshape(-1)))
    k = min(every)
    parents[tables[k][0][5, 1].long()] = 2                     # (one child of group k given a second parent)
    pc.__dict__["_fused_child_parents"] = parents
    _decided(pc, x)
    assert _fused_groups(pc, B, "_stream_fusion_bk_") == every - {k}


@cuda_only
def test_no_backward_fusion_where_the_stream_element_kernel_does_not_run(fused_preferred, monkeypatch):
    """Element flows steered to the pre-pass kernel: nothing fused in the backward (`_stream_fusion_bk`)."""
    _needs_stream()
    real = autotune.best_of
    monkeypatch.setattr(autotune, "best_of", lambda cands, *a, **k: next(
        (key for key, _ in cands if isinstance(key, tuple) and key[:1] == ("cuda_prepass",)), None)
        or real(cands, *a, **k))
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    _decided(pc, x, 3)
    assert _fused_groups(pc, B, "_stream_fusion_bk_") == frozenset()


@cuda_only
def test_no_backward_fusion_of_parameter_flows_that_are_not_deferred(fused_preferred, monkeypatch):
    """Parameter flows not deferred (their kernels read the element values): no backward fusion with parameter flows
    -- one without them (`compute_param_flows = False`) fuses."""
    _needs_stream()
    import pyjuice.layer.deferred_par as deferred_par
    monkeypatch.setattr(deferred_par, "may_defer", lambda *a, **k: False)
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    _decided(pc, x, 3)
    assert _fused_groups(pc, B, "_stream_fusion_bk_") == frozenset()
    for _ in range(3):
        pc(x)
        pc.backward(x, compute_param_flows = False)
    assert pc.__dict__["_stream_fusion_bk_"][(B, pc.precision, pc.params.dtype, pc._param_operand_groups(B), False)]


@cuda_only
def test_a_fused_backward_whose_deferral_declines_computes_the_unfused_bits(fused_preferred, monkeypatch):
    """Parameter flows deferred when the backward fusion was decided, then not (the deferral's thresholds out of reach):
    the fused layers form the element values for the per-step kernels themselves -- the bits of a circuit never fusing,
    node flows and parameter flows."""
    _needs_stream()
    import pyjuice.layer.deferred_par as deferred_par
    pcs = [_hmm(4096)[0], _unfused(_hmm(4096)[0])]
    B, T, E = 128, 8, 64
    x = torch.randint(0, E, (B, T), device = DEV)
    for pc in pcs:
        _decided(pc, x)
    assert _fused_groups(pcs[0], B, "_stream_fusion_bk_")
    monkeypatch.setattr(deferred_par, "MIN_FLOWS", 1 << 62)
    monkeypatch.setattr(deferred_par, "MIN_FLOW_SAMPLES", 1 << 62)
    with torch.no_grad():
        for p0, p1 in zip(pcs[0].parameters(), pcs[1].parameters()):
            if p0.is_floating_point():
                p0.data.copy_(p1.data)
    for pc in pcs:
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)
    assert torch.equal(pcs[0].node_flows, pcs[1].node_flows) and torch.equal(pcs[0].param_flows, pcs[1].param_flows)


@cuda_only
def test_no_backward_fusion_on_kept_flow_buffers(fused_preferred):
    """A backward keeping the flow buffers it is given (`_disable_buffer_init`: a child's flows need not start at -inf,
    the product backward adds to them) is not fused: the bits of a circuit never fusing."""
    _needs_stream()
    pcs = [_hmm(4096)[0], _unfused(_hmm(4096)[0])]
    B, T, E = 128, 8, 64
    x = torch.randint(0, E, (B, T), device = DEV)
    for pc in pcs:
        _decided(pc, x)
    assert _fused_groups(pcs[0], B, "_stream_fusion_bk_")
    for pc in pcs:
        pc(x)
        pc.backward(x, flows_memory = 1.0)
        pc.backward(x, flows_memory = 1.0, _disable_buffer_init = True)
    assert torch.equal(pcs[0].node_flows, pcs[1].node_flows)


@cuda_only
def test_stream_children_backward_takes_a_settled_plain_call(stream_steered):
    """`backward(stream_children = ...)` in linear-space flows, or where the layer's element flows take no stream kernel
    (a batch of 48, not a multiple of its 64 columns): an error."""
    _needs_stream()
    from pyjuice.layer.fused_children import element_children
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    _decided(pc, x, 1)
    k = [i for i, g in enumerate(pc.inner_layer_groups) if g.is_sum()][2]
    layer = list(pc.inner_layer_groups[k])[0]
    table = element_children(pc)[k - 1]
    nf, ef = pc.node_flows.clone(), pc.element_flows.clone()
    assert layer._stream_fusion_bk(B, pc.params.dtype, False, False)
    with pytest.raises(ValueError):
        layer.backward(nf, ef, pc.node_mars, pc.element_mars, pc.params, logspace_flows = False, stream_children = table)
    with pytest.raises(ValueError):
        layer.backward(nf[:, :48].contiguous(), ef[:, :48].contiguous(), pc.node_mars[:, :48].contiguous(),
                       pc.element_mars[:, :48].contiguous(), pc.params, logspace_flows = True, stream_children = table)


@cuda_only
def test_a_fused_backward_replayed_from_a_graph_computes_the_eager_bits(fused_preferred):
    """A backward recorded into a CUDA graph with its products fused replays its eager node flows and parameter flows;
    the fusion is in the backward graphs' key (the unfused first backward's graph is not replayed for the fused)."""
    _needs_stream()
    pc, T, E = _hmm(4096)
    B = 128
    x = torch.randint(0, E, (B, T), device = DEV)
    pc(x)
    pc.init_param_flows(flows_memory = 0.0)
    pc.backward(x, flows_memory = 1.0, record_cudagraph = True)
    n = len(pc._recorded_cuda_graphs)
    pc(x)
    pc.init_param_flows(flows_memory = 0.0)
    pc.backward(x, flows_memory = 1.0, record_cudagraph = True)
    assert _fused_groups(pc, B, "_stream_fusion_bk_") and len(pc._recorded_cuda_graphs) == n + 1
    outs = []
    for apply in (True, False):
        pc(x, apply_cudagraph = False)
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(x, flows_memory = 1.0, apply_cudagraph = apply)
        outs.append((pc.node_flows.clone(), pc.param_flows.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def _two_gpus():
    """Two GPU ids for a child process, or None (the conftest pins each test process to one GPU; the pool it published
    lists them all, ids past the worker count preferred -- as `tests/model/device_move_test.py` reads it)."""
    import os
    pool = os.environ.get("PYJUICE_TEST_GPU_POOL")
    if pool:
        gpus = [g for g in pool.split(",") if g]
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output = True, text = True)
            gpus = [str(i) for i in range(len(out.stdout.strip().splitlines()))]
        except (FileNotFoundError, OSError):
            gpus = []
    if len(gpus) < 2:
        return None
    idle = gpus[int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")):]
    return idle[:2] if len(idle) >= 2 else gpus[-2:]


# Runs with two GPUs visible: the choices steered as `fused_preferred` steers them
_MOVED_CHILD = r"""
import torch
import pyjuice as juice
import pyjuice.layer.kernels.autotune as autotune
real_pick, real_choose = autotune.pick, autotune.choose

def choose(key, cands):
    if key[0] == "sum_fw_tlmm_cuda":
        stream = [(c, run) for c, run in cands if c[0] == "cuda_stream"]
        if stream:
            return real_choose(key, stream[:1])
    return real_choose(key, cands)
autotune.choose = choose
real_best_of = autotune.best_of
autotune.best_of = lambda cands, *a, **k: next((key for key, _ in cands if isinstance(key, tuple)
                                                and key[:1] == ("cuda_stream",)), None) or real_best_of(cands, *a, **k)
autotune.pick = lambda key, cands, bench, *a, **k: ("fused" if key[0] in ("prod_stream_fusion", "prod_stream_fusion_bk")
                                                    else real_pick(key, cands, bench, *a, **k))
d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
B, T, E = 128, 8, 64


def build(dev):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = 4096, num_emits = E, homogeneous = True, block_size = 1024)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, param_dtype = torch.bfloat16).to(dev)


def fused(pc):
    return [v for k, v in pc.__dict__.get("_stream_fusion_", {}).items() if k[0] == B and isinstance(v, frozenset)]


x = torch.randint(0, E, (B, T), generator = torch.Generator().manual_seed(1))
torch.cuda.set_device(d0)
pc = build(d0)
with torch.no_grad():
    for _ in range(3):
        pc(x.to(d0))
assert fused(pc) and fused(pc)[0], "fused before the move"
torch.cuda.set_device(d1)
twin = build(d1)
with torch.no_grad():
    for _ in range(3):
        want = twin(x.to(d1)).clone()
    pc.to(d1)
    for it in range(3):
        got = pc(x.to(d1)).clone()
        torch.cuda.synchronize()
        assert torch.equal(got, want), it
assert fused(pc) and fused(pc)[0] and fused(twin)[0], "fused after the move"
for it in range(3):                                      # (backwards: the fusion decided anew there too)
    for c in (twin, pc):
        c(x.to(d1))
        c.backward(x.to(d1))
    torch.cuda.synchronize()
    assert torch.equal(pc.node_flows, twin.node_flows), ("flows", it)
print("CHILD_OK")
"""


@pytest.mark.skipif(_two_gpus() is None, reason = "requires >= 2 GPUs")
def test_a_fused_circuit_moved_after_running_computes_as_one_compiled_there():
    """A circuit whose forwards fused on one GPU, `pc.to` another: its passes there decide anew (the layers' fused plans,
    the product groups' tables and the decisions made on the old device dropped), and compute the bits of a twin
    compiled there -- not the old device's pointers."""
    import os, sys
    gpus = _two_gpus()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env.pop("PYJUICE_TEST_GPU_POOL", None)
    proc = subprocess.run([sys.executable, "-c", _MOVED_CHILD], capture_output = True, text = True, timeout = 900, env = env)
    assert proc.returncode == 0 and "CHILD_OK" in proc.stdout, (
        f"a moved fused circuit failed on GPUs {gpus} (exit {proc.returncode})\n"
        f"--- child stdout ---\n{proc.stdout}\n--- child stderr ---\n{proc.stderr}")
