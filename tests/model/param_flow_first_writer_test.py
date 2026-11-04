"""
The pass's first writer of a parameter-flow slot stores it (`TensorCircuit._plan_param_flow_stores`,
`SumLayer.param_flow_slot_plan`, the read-add-store kernel's `PF_STORE` and the CUDA kernel's store mode): at
`flows_memory = 0` the circuit zeroes only the slots no storing partition covers, and the flows come out the same bits
as zeroing everything and accumulating -- on an untied HCLT (every partition the sole writer of its slots), a tied HMM
(the first layer in backward order stores, the rest accumulate), a deferring tied HMM (the flush accumulates: nothing
stores), small batches (the sparse kernels), a forced `mode`, external parameters (their hooks accumulate), a backward of
element flows only, launch-plan replays alternating memories, and a plan that mispredicts a partition (its slots reset
by the path itself). Skipped where the CUDA kernels are not available.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.layer.sum_layer import SumLayer
from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires a GPU")


def _hclt(latents, block_size, dt = None, V = 64, cats = 16):
    torch.manual_seed(0)
    x = torch.randint(0, cats, (512, V))
    ns = juice.structures.HCLT(x.float(), num_latents = latents, input_dist = dists.Categorical(num_cats = cats),
                               block_size = block_size)
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, **({} if dt is None else dict(param_dtype = dt))).to(DEV), V, cats


def _hmm(latents, T = 4, block_size = None, dt = None):
    torch.manual_seed(0)
    ns = juice.structures.HMM(seq_length = T, num_latents = latents, num_emits = 16, homogeneous = True,
                              **({} if block_size is None else dict(block_size = block_size)))
    ns.init_parameters(perturbation = 2.0)
    return juice.compile(ns, verbose = False, **({} if dt is None else dict(param_dtype = dt))).to(DEV), T, 16


def _sums(pc):
    return [l for g in pc.inner_layer_groups if g.is_sum() for l in g]


def _reset_plans(pc):
    pc.__dict__.pop("_pf_store_plans", None)
    pc.__dict__.pop("_pf_applied_key", None)
    for l in _sums(pc):
        l.__dict__.pop("_pf_store_flags", None)
        l.__dict__.pop("_pf_fallback_resets", None)


def _run(pc, x, **kw):
    pc(x, **{k: v for k, v in kw.items() if k == "sum_external_params"})
    pc.backward(x, flows_memory = 0.0, **{k: v for k, v in kw.items() if k != "sum_external_params"})
    torch.cuda.synchronize()
    return pc.param_flows.clone()


def _reference(pc, x, monkeypatch, **kw):
    """Every partition accumulating onto a full zero: the plan told nothing is capable."""
    real = SumLayer.param_flow_slot_plan
    with monkeypatch.context() as mp:
        mp.setattr(SumLayer, "param_flow_slot_plan", lambda self, *a, **k: [(r, False) for r, _ in real(self, *a, **k)])
        _reset_plans(pc)
        out = _run(pc, x, **kw)
    _reset_plans(pc)
    return out


def _plan(pc):
    plans = list(pc.__dict__.get("_pf_store_plans", {}).values())
    assert len(plans) == 1, "one plan for the one call signature"
    flags, keys, zero = plans[0]
    return [f for d in flags for f in d.values()], zero


def _stale(pc, value = 7.0):
    if getattr(pc, "param_flows", None) is not None:
        pc.param_flows.fill_(value)                        # (stale flows a store must overwrite and a zero must clear)


@cuda_only
@pytest.mark.parametrize("dt", [None, torch.bfloat16], ids = ["fp32", "bf16"])
def test_an_untied_hclt_stores_every_partition(dt, monkeypatch):
    pc, V, C = _hclt(64, 32, dt)
    x = torch.randint(0, C, (64, V), device = DEV)
    ref = _reference(pc, x, monkeypatch)
    _stale(pc)
    got = _run(pc, x)
    flags, zero = _plan(pc)
    assert sum(flags) >= len(flags) - 1 and (zero is None or sum(b - a for a, b in zero) <= pc.num_param_flows // 100), \
        (sum(flags), len(flags), zero)                     # (the root's one block may take another kernel)
    assert torch.equal(got, ref)
    assert sum(l.__dict__.get("_pf_fallback_resets", 0) for l in _sums(pc)) == 0, "the plan predicted every path"
    _stale(pc)
    assert torch.equal(_run(pc, x), ref), "a second pass over stale flows"


@cuda_only
def test_a_tied_hmm_stores_its_first_layer_only(monkeypatch):
    pc, T, C = _hmm(128)
    x = torch.randint(0, C, (64, T), device = DEV)
    ref = _reference(pc, x, monkeypatch)
    _stale(pc)
    got = _run(pc, x)
    flags, zero = _plan(pc)
    groups = pc.inner_layer_groups
    order = [l for lid in range(len(groups) - 1, -1, -1) if groups[lid].is_sum() for l in groups[lid]]
    storing = [l for l in order if any(l.__dict__["_pf_store_flags"].values())]
    tied = [l for l in order if l.param_flow_slot_ranges(0).tolist() == storing[0].param_flow_slot_ranges(0).tolist()]
    assert len(storing) == 1 and len(tied) >= 2 and tied[0] is storing[0], "the first tied layer of the pass stores"
    assert torch.equal(got, ref)


@cuda_only
def test_a_deferring_tied_hmm_stores_nothing(monkeypatch):
    pc, T, C = _hmm(1024, dt = torch.bfloat16)
    x = torch.randint(0, C, (512, T), device = DEV)
    ref = _reference(pc, x, monkeypatch)
    _stale(pc)
    got = _run(pc, x)
    flags, zero = _plan(pc)
    assert not any(flags) and zero is None                      # (nothing stores: one zero pass)
    fin = torch.isfinite(ref)
    assert torch.equal(fin, torch.isfinite(got)) and float((got[fin] - ref[fin]).abs().max()) < 1e-5 * float(ref[fin].abs().max())


@cuda_only
@pytest.mark.parametrize("B", [4, 8])
def test_small_batches_take_the_accumulating_path(B, monkeypatch):
    pc, V, C = _hclt(64, 32)
    x = torch.randint(0, C, (B, V), device = DEV)
    ref = _reference(pc, x, monkeypatch)
    _stale(pc)
    got = _run(pc, x)
    flags, zero = _plan(pc)
    assert not any(flags) and zero is None                      # (nothing stores: one zero pass)
    assert torch.equal(got, ref)


@cuda_only
@pytest.mark.parametrize("mode", ["sparse"])     # ("pytorch" via `pc.backward` rejects the circuit's options: pre-existing)
def test_a_forced_mode_takes_the_accumulating_path(mode, monkeypatch):
    pc, V, C = _hclt(64, 32)
    x = torch.randint(0, C, (64, V), device = DEV)
    ref = _reference(pc, x, monkeypatch, mode = mode)
    _stale(pc)
    got = _run(pc, x, mode = mode)
    flags, zero = _plan(pc)
    assert not any(flags) and zero is None                      # (nothing stores: one zero pass)
    assert torch.equal(got, ref)


@cuda_only
def test_a_mispredicted_partition_resets_its_slots_itself(monkeypatch):
    """The plan says a partition stores, the dispatch takes another kernel: the path resets the slots first."""
    pc, V, C = _hclt(64, 32)
    x = torch.randint(0, C, (64, V), device = DEV)
    ref = _reference(pc, x, monkeypatch)
    import pyjuice.layer.sum_layer as SL
    with monkeypatch.context() as mp:
        # the collision check says "collisions": the atomic kernel runs; the plan is computed before the patch
        _reset_plans(pc)
        pc(x)
        pc._plan_param_flow_stores(64, 0.0, 1.0, "LL", True)
        flags, zero = _plan(pc)
        assert any(flags)
        mp.setattr(SumLayer, "_par_flow_collision_free", lambda self, pfids: False)
        _stale(pc)
        pc.backward(x, flows_memory = 0.0)
        torch.cuda.synchronize()
        got = pc.param_flows.clone()
        assert sum(l.__dict__.get("_pf_fallback_resets", 0) for l in _sums(pc)) >= 1
    assert torch.equal(got, ref)


@cuda_only
def test_external_parameters_keep_the_accumulating_hooks(monkeypatch):
    torch.manual_seed(0)
    with juice.set_block_size(64):
        ni = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 8)) for v in range(2)]
        ns = summate(multiply(*ni), num_node_blocks = 2, external_params = BlockScaleSumParams(ch_block_size = 64))
        root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(DEV)
    x = torch.randint(0, 8, (64, 2), device = DEV)
    g = torch.Generator().manual_seed(0)
    phi = {ns: torch.randn(ns.external_params.tensor_shapes(ns, 64)[0], generator = g).to(DEV)}
    ref = _reference(pc, x, monkeypatch, sum_external_params = phi)
    _stale(pc)
    got = _run(pc, x, sum_external_params = phi)
    flags, zero = _plan(pc)
    gated = [l for l in _sums(pc) if getattr(l, "external_params", None) is not None]
    assert gated and all(not f for l in gated for f in l.__dict__["_pf_store_flags"].values())
    assert torch.equal(got, ref)


@cuda_only
def test_replays_alternating_memories(monkeypatch):
    """Launch plans: a pass at memory 0 (stores) and a pass at memory 1 (accumulates) replay as recorded, each under its
    own key."""
    pc, V, C = _hclt(64, 32)
    x = torch.randint(0, C, (64, V), device = DEV)
    ref0 = _reference(pc, x, monkeypatch)
    pc(x); pc.backward(x, flows_memory = 1.0); ref1 = pc.param_flows.clone()      # (= 2 x ref0 up to accumulation)
    for _ in range(3):
        _stale(pc)
        pc(x); pc.backward(x, flows_memory = 0.0); torch.cuda.synchronize()
        assert torch.equal(pc.param_flows, ref0)
        pc(x); pc.backward(x, flows_memory = 1.0); torch.cuda.synchronize()
        assert torch.equal(pc.param_flows, ref1)


@cuda_only
def test_memory_between_zero_and_one_scales_as_before():
    pc, V, C = _hclt(64, 32)
    x = torch.randint(0, C, (64, V), device = DEV)
    pc(x); pc.backward(x, flows_memory = 0.0); torch.cuda.synchronize()
    once = pc.param_flows.clone()
    pc(x); pc.backward(x, flows_memory = 0.5); torch.cuda.synchronize()
    assert torch.allclose(pc.param_flows, once * 1.5, rtol = 1e-6, atol = 1e-6)
    assert not any(f for l in _sums(pc) for f in l.__dict__.get("_pf_store_flags", {}).values())


@cuda_only
def test_a_backward_without_parameter_flows_leaves_them(monkeypatch):
    pc, V, C = _hclt(64, 32)
    x = torch.randint(0, C, (64, V), device = DEV)
    ref = _reference(pc, x, monkeypatch)
    pc(x); pc.backward(x, flows_memory = 0.0); torch.cuda.synchronize()
    pc(x); pc.backward(x, compute_param_flows = False); torch.cuda.synchronize()
    assert torch.equal(pc.param_flows, ref)
    assert not any(f for l in _sums(pc) for f in l.__dict__.get("_pf_store_flags", {}).values())


def _ragged():
    """A sum layer whose partition pads: node blocks with 8, 3 and 1 child blocks over the same products (padding edges
    address child row 0 and slot 0's block, which a real edge owns)."""
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ni = [inputs(v, num_node_blocks = 4, dist = dists.Categorical(num_cats = 8)) for v in range(4)]
        prods = multiply(ni[0], ni[1], edge_ids = torch.arange(4).repeat(2).view(2, 4).T.repeat(2, 1)[:8])   # 8 blocks
        n_ch = prods.num_node_blocks
        rows = [list(range(n_ch)), [0, 3, 5], [2]]
        edge_ids = torch.tensor([[b, c] for b, cs in enumerate(rows) for c in cs]).T
        ns = summate(prods, num_node_blocks = 3, edge_ids = edge_ids)
        up = multiply(ns, inputs(4, num_node_blocks = 3, dist = dists.Categorical(num_cats = 8)))
        root = summate(up, num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return juice.compile(root, verbose = False, max_num_partitions = 1).to(DEV), 5, 8   # (one padded partition)


@cuda_only
def test_a_padded_partition_is_not_planned(monkeypatch):
    """Padding edges address slot 0's block, which a real edge owns: the collision check counts them, the partition takes
    the atomic kernel and the plan leaves it accumulating (its slots zeroed); the flows are the reference's."""
    pc, V, C = _ragged()
    layer = [l for l in _sums(pc) if l.block_size == 32][0]
    cids, pfids = layer.partitioned_cids[0], layer.partitioned_pfids[0]
    assert bool(((cids == 0) & (pfids == 0)).any()), "the partition pads"
    assert bool((pfids[cids != 0] == 0).any()), "a real edge owns slot 0's block"
    x = torch.randint(0, C, (64, V), device = DEV)
    ref = _reference(pc, x, monkeypatch)
    for _ in range(3):
        _stale(pc)
        got = _run(pc, x)
        assert not any(layer.__dict__["_pf_store_flags"].values()), "the padded partition accumulates"
        assert layer.__dict__.get("_pf_fallback_resets", 0) == 0
        assert torch.equal(got, ref)


@cuda_only
def test_the_cuda_kernel_stores_in_its_store_mode(monkeypatch):
    """Blocks of 64 and 128 edges: the CUDA parameter-flow kernel's regime; its choice forced, it stores (mode 2)."""
    from pyjuice.layer.kernels import autotune, c as ck
    if not ck.par_is_available():
        pytest.skip("the CUDA parameter-flow kernel is not available here")
    pc, V, C = _hclt(128, 64)
    x = torch.randint(0, C, (64, V), device = DEV)
    real = autotune.best_of
    monkeypatch.setattr(autotune, "best_of", lambda cands, *a, **k: "cuda" if any(c == "cuda" for c, _ in cands)
                        else real(cands, *a, **k))
    ref = _reference(pc, x, monkeypatch)
    _stale(pc)
    got = _run(pc, x)
    chosen = [v for l in _sums(pc) for v in l._cached_bk_par_choice.values()]
    assert chosen and all(v == "cuda" for v in chosen), chosen
    assert sum(f for l in _sums(pc) for f in l.__dict__["_pf_store_flags"].values()) >= len(_sums(pc)) - 1
    assert torch.equal(got, ref)
    assert sum(l.__dict__.get("_pf_fallback_resets", 0) for l in _sums(pc)) == 0


@cuda_only
def test_colliding_slots_are_not_planned(monkeypatch):
    """Two tied node blocks in one layer share their slots (the atomic kernel): the plan knows, no fallback reset."""
    torch.manual_seed(0)
    with juice.set_block_size(32):
        ni = [inputs(v, num_node_blocks = 2, dist = dists.Categorical(num_cats = 8)) for v in range(4)]
        ns_a = summate(multiply(ni[0], ni[1]), num_node_blocks = 2)
        ns_b = ns_a.duplicate(multiply(ni[2], ni[3]), tie_params = True)
        root = summate(multiply(ns_a, ns_b), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc = juice.compile(root, verbose = False).to(DEV)
    x = torch.randint(0, 8, (64, 4), device = DEV)
    tied = [l for l in _sums(pc) if not l._par_flow_collision_free(l.partitioned_pfids[0])]
    assert tied, "a layer with colliding slots"
    ref = _reference(pc, x, monkeypatch)
    _stale(pc)
    got = _run(pc, x)
    assert all(not f for l in tied for f in l.__dict__["_pf_store_flags"].values())
    assert sum(l.__dict__.get("_pf_fallback_resets", 0) for l in _sums(pc)) == 0
    assert torch.equal(got, ref)
