"""
`param_dtype`: the sum layers' parameters stored in bf16 (`juice.compile(ns, param_dtype = torch.bfloat16)`) against the
same circuit at fp32 -- the kernels read the parameters as stored and compute in fp32 (or the bf16 / fp16 operand they
already used), so the two agree to the rounding of the parameters to bf16 (2^-9 relative, MEASURED on HCLT-1024:
log-likelihoods 5e-4 nats apart at most). Forward log-likelihoods, flows, EM steps (the parameters stay normalized, in
their dtype), the queries (marginal, conditional, sampling under one seed on the same values), `set_param_dtype` both
ways, the fp32 node parameters the nodes get back, a large dense block (off the cuBLAS path at bf16), `precision =
"fp32"` with bf16 parameters, and the refusal of fp16. Training through the fp32 master copy (`params_master`): every
optimizer follows its fp32 run step for step (EM at step sizes 1 to 0.001, SGD, Adam -- the parameters keep moving, which
bf16 storage alone stops: MEASURED 53-100% of the updates rounded away), `params` the master's rounding after every
step; a write to `params` between steps is kept; the copy's lifecycle. BlockScale-gated circuits at bf16. The default is
fp32, and at fp32 nothing changes: the rest of the suite is that check.
"""
import random

import pytest
import torch

import pyjuice as juice
import pyjuice.nodes.distributions as dists

DEV = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
BF16 = torch.bfloat16


def _pair(kind, **compile_kwargs):
    """The circuit at fp32 and at bf16, from the SAME nodes and initialization; (fp32 pc, bf16 pc, V, C)."""
    torch.manual_seed(0)
    if kind == "hclt":
        x = torch.randint(0, 16, (512, 32))
        ns = juice.structures.HCLT(x.float(), num_latents = 64, input_dist = dists.Categorical(num_cats = 16)); V, C = 32, 16
    elif kind == "hmm":
        ns = juice.structures.HMM(seq_length = 16, num_latents = 128, num_emits = 32); V, C = 16, 32
    elif kind == "hmm_dense":                                      # blocks of 1024: the dense (cuBLAS) path at fp32
        ns = juice.structures.HMM(seq_length = 6, num_latents = 1024, num_emits = 64); V, C = 6, 64
    else:
        ns = juice.structures.PD(data_shape = (4, 4), num_latents = 8, split_intervals = 2,
                                 input_dist = dists.Categorical(num_cats = 16)); V, C = 16, 16
    ns.init_parameters(perturbation = 2.0)
    pc32 = juice.compile(ns, verbose = False, **compile_kwargs).to(DEV)
    pc16 = juice.compile(ns, verbose = False, param_dtype = BF16, **compile_kwargs).to(DEV)
    assert pc32.params.dtype == torch.float32 and pc16.params.dtype == BF16 and pc16.param_dtype == BF16
    assert torch.equal(pc16.params.float(), pc32.params.to(BF16).float())    # the same values, rounded once
    return pc32, pc16, V, C


def _seeded(fn, seed):
    random.seed(seed); torch.manual_seed(seed)
    return fn()


@cuda_only
@pytest.mark.parametrize("kind", ["hclt", "hmm", "pd", "hmm_dense"])
@pytest.mark.parametrize("B", [8, 64])
def test_bf16_parameters_agree_with_fp32_to_their_rounding(kind, B):
    pc32, pc16, V, C = _pair(kind)
    x = torch.randint(0, C, (B, V), device = DEV)
    ll32, ll16 = pc32(x), pc16(x)
    assert torch.isfinite(ll16).all()
    assert torch.allclose(ll16, ll32, atol = 2e-2 * (1 + abs(ll32).mean().item() / 1000), rtol = 0.0), float((ll16 - ll32).abs().max())
    # the flows (log-space) of the same backward, where finite in both
    for pc in (pc32, pc16):
        pc.init_param_flows(flows_memory = 0.0)
        pc.backward(x, flows_memory = 1.0, logspace_flows = True)
    f32, f16 = pc32.node_flows, pc16.node_flows
    fin = torch.isfinite(f32) & torch.isfinite(f16)
    assert torch.equal(torch.isinf(f32), torch.isinf(f16)) or (~fin).float().mean() < 1e-3
    assert torch.allclose(f16[fin], f32[fin], atol = 5e-2, rtol = 1e-2), float((f16[fin] - f32[fin]).abs().max())
    # the parameter flows are fp32 in both and agree to the same rounding
    assert pc16.param_flows.dtype == torch.float32
    pf32, pf16 = pc32.param_flows, pc16.param_flows
    assert torch.allclose(pf16, pf32, atol = 1e-3 * pf32.abs().max().item() + 1e-6, rtol = 1e-2)


@cuda_only
@pytest.mark.parametrize("kind", ["hclt", "hmm"])
def test_em_at_bf16_stays_normalized_and_close_to_fp32(kind):
    pc32, pc16, V, C = _pair(kind)
    data = torch.randint(0, C, (1024, V), device = DEV)
    lls = {}
    for tag, pc in (("fp32", pc32), ("bf16", pc16)):
        for step in range(5):
            pc.init_param_flows(flows_memory = 0.0)
            tot = 0.0
            for i in range(0, 1024, 256):
                tot += pc(data[i:i + 256]).sum().item()
                pc.backward(data[i:i + 256], flows_memory = 1.0)
            pc.mini_batch_em(step_size = 1.0, pseudocount = 0.1)
        lls[tag] = tot / 1024
        assert pc.params.dtype == (BF16 if tag == "bf16" else torch.float32)       # the update kept the dtype
        # every sum node's parameters still sum to one (bf16: to its rounding over the block)
        n = pc.get_node_params(pc.root_ns)
        assert n.dtype == torch.float32
    assert abs(lls["bf16"] - lls["fp32"]) < 0.5, lls                               # per sample, out of hundreds of nats
    for ns in pc16.root_ns:
        if ns.is_sum() and not ns.is_tied():
            p = pc16.get_node_params(ns)                                           # [blocks, nodes, children]
            assert torch.allclose(p.sum(dim = 2), torch.ones_like(p.sum(dim = 2)), atol = 2e-2), ns


@cuda_only
def test_queries_at_bf16():
    pc32, pc16, V, C = _pair("hclt")
    B = 16
    x = torch.randint(0, C, (B, V), device = DEV)
    miss = torch.rand(B, V, device = DEV) < 0.4
    m32 = juice.queries.marginal(pc32, data = x.clone(), missing_mask = miss)
    m16 = juice.queries.marginal(pc16, data = x.clone(), missing_mask = miss)
    assert torch.allclose(m16, m32, atol = 2e-2, rtol = 0.0)
    c32 = juice.queries.conditional(pc32, data = x.clone(), missing_mask = miss, target_vars = [0, 3])
    c16 = juice.queries.conditional(pc16, data = x.clone(), missing_mask = miss, target_vars = [0, 3])
    assert torch.allclose(c16, c32, atol = 1e-2, rtol = 0.0)
    # sampling reads the parameters: under one seed, the bf16 circuit draws as the fp32 circuit does once the fp32
    # circuit holds the same (rounded) values
    pc32.set_param_dtype(BF16).set_param_dtype(torch.float32)
    assert pc32.params.dtype == torch.float32 and torch.equal(pc32.params, pc16.params.float())
    s32 = _seeded(lambda: juice.queries.sample(pc32, num_samples = 64), 7)
    s16 = _seeded(lambda: juice.queries.sample(pc16, num_samples = 64), 7)
    assert torch.equal(s32, s16)
    juice.queries.marginal(pc32, data = x.clone(), missing_mask = miss)
    juice.queries.marginal(pc16, data = x.clone(), missing_mask = miss)
    s32 = _seeded(lambda: juice.queries.sample(pc32, conditional = True), 8)
    s16 = _seeded(lambda: juice.queries.sample(pc16, conditional = True), 8)
    assert (s32 != s16).float().mean() < 0.01            # the walks read node values a rounding apart: a rare flip


@cuda_only
def test_set_param_dtype_and_the_refusals():
    pc32, pc16, V, C = _pair("pd")
    x = torch.randint(0, C, (8, V), device = DEV)
    ll32 = pc32(x).clone()
    pc32.set_param_dtype(BF16)
    assert pc32.params.dtype == BF16 and pc32.param_dtype == BF16 and torch.equal(pc32.params, pc16.params)
    assert torch.allclose(pc32(x), pc16(x))
    pc32.set_param_dtype(torch.float32)
    assert pc32.params.dtype == torch.float32 and torch.allclose(pc32(x), ll32, atol = 2e-2)
    pc32.update_parameters()                                           # the nodes get fp32 parameters back
    for ns in pc32.root_ns:
        if ns.is_sum() and not ns.is_tied():
            assert ns._params.dtype == torch.float32
            break
    with pytest.raises(ValueError, match = "param_dtype"):
        pc32.set_param_dtype(torch.float16)
    with pytest.raises(ValueError, match = "param_dtype"):
        juice.compile(pc32.root_ns, verbose = False, param_dtype = torch.float16)
    # precision = "fp32" (exact products) on bf16-stored parameters
    pc32e, pc16e, V, C = _pair("hclt", precision = "fp32")
    assert torch.allclose(pc16e(x[:, :1].repeat(1, V) % C), pc32e(x[:, :1].repeat(1, V) % C), atol = 2e-2)


# ---------------------------------------------------------------------------------------------------------------------
# Training through the fp32 master copy
# ---------------------------------------------------------------------------------------------------------------------

def _training_setup():
    """An HCLT over 64 variables and data sampled from another initialization of it (something to learn)."""
    torch.manual_seed(0)
    y = torch.randint(0, 32, (2048, 64))
    ns = juice.structures.HCLT(y.float(), num_latents = 64, input_dist = dists.Categorical(num_cats = 32))
    ns.init_parameters(perturbation = 2.0)
    data = juice.queries.sample(juice.compile(ns, verbose = False).to(DEV), num_samples = 2048).to(DEV)
    ns.init_parameters(perturbation = 4.0)
    return ns, data


OPTIMIZERS = {"em_1": lambda pc: juice.optim.MiniBatchEM(pc, step_size = 1.0, pseudocount = 0.1),
              "em_0.1": lambda pc: juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1),
              "em_0.01": lambda pc: juice.optim.MiniBatchEM(pc, step_size = 0.01, pseudocount = 0.1),
              "em_0.001": lambda pc: juice.optim.MiniBatchEM(pc, step_size = 0.001, pseudocount = 0.1),
              "em_keep_zero": lambda pc: juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1, keep_zero_params = True),
              "sgd": lambda pc: juice.optim.SGD(pc, lr = 0.1),
              "adam": lambda pc: juice.optim.Adam(pc, lr = 0.01)}


@cuda_only
@pytest.mark.parametrize("name", list(OPTIMIZERS))
def test_training_at_bf16_follows_fp32_training(name):
    """10 steps from the same state: the master copy moves as the fp32 parameters do (it is the fp32 update), stays
    within the drift the bf16 forward makes (MEASURED: median 1.3-3.1e-3 relative, p99 <= 1.3e-2), and `params` is its
    rounding after every step."""
    ns, data = _training_setup()
    runs = {}
    for dtype in (torch.float32, BF16):
        pc = juice.compile(ns, verbose = False, param_dtype = dtype).to(DEV)
        opt = OPTIMIZERS[name](pc)
        moved = []
        for s in range(10):
            xb = data[(s * 256) % 2048:(s * 256) % 2048 + 256]
            before, version = pc._params_in_fp32().clone(), pc.params._version
            pc(xb).mean().backward()
            opt.step()
            now = pc._params_in_fp32()
            if name in ("sgd", "adam"):
                # an update signals itself as it does at fp32 (`clamp_`), which caches keyed on the version read --
                # a store through `.data` alone would not
                assert pc.params._version != version, (name, dtype)
            nz = before > 0
            moved.append(float(((now != before) & nz).sum() / nz.sum()))
            if dtype is BF16:
                assert pc.params.dtype == BF16 and pc.params_master is not None and pc.params_master.dtype == torch.float32
                assert torch.equal(pc.params, pc.params_master.to(BF16)), s
            else:
                assert pc.params_master is None                                # fp32 storage never makes one
        runs[dtype] = (pc._params_in_fp32().clone(), sum(moved) / len(moved), pc(data[:256]).mean().item())
    (p32, moved32, ll32), (p16, moved16, ll16) = runs[torch.float32], runs[BF16]
    assert moved16 >= moved32 - 0.02, (moved16, moved32)
    nz = p32 > 1e-6
    rel = ((p16 - p32).abs() / p32)[nz]
    assert rel.median() < 6e-3 and rel.quantile(0.99) < 4e-2, (float(rel.median()), float(rel.quantile(0.99)))
    assert abs(ll16 - ll32) < 0.05, (ll16, ll32)


@cuda_only
@pytest.mark.parametrize("name", ["em_0.01", "sgd"])
def test_a_write_to_bf16_params_between_steps_is_kept(name):
    """`pc.params.data[...] = x` between two steps: the next update starts from x, not from the master copy's older
    value (the EM update checks in its own pass, the gradient path before its update) -- against an fp32 twin given the
    same write. The write is not uniform within a node's children (x1.5 / x0.5 alternating), so a renormalization cannot
    undo it, and it is far off the master copy (a missed write would leave the block ~40% from the twin's)."""
    ns, data = _training_setup()
    pc16 = juice.compile(ns, verbose = False, param_dtype = BF16).to(DEV)
    pc32 = juice.compile(ns, verbose = False, param_dtype = BF16).to(DEV).set_param_dtype(torch.float32)
    opts = [OPTIMIZERS[name](pc) for pc in (pc16, pc32)]
    for pc, opt in zip((pc16, pc32), opts):
        pc(data[:256]).mean().backward(); opt.step()                   # (the master copy exists)
    ps, pe = pc16.num_dummy_params, pc16.num_dummy_params + 4096
    factor = torch.tensor([1.5, 0.5], device = DEV).repeat(2048)
    written = (pc16.params.data[ps:pe].float() * factor).to(BF16)
    pc16.params.data[ps:pe] = written
    pc32.params.data[ps:pe] = written.float()
    for pc, opt in zip((pc16, pc32), opts):
        pc(data[256:512]).mean().backward(); opt.step()
    rel = (pc16.params_master[ps:pe] - pc32.params[ps:pe]).abs() / pc32.params[ps:pe].clamp_min(1e-12)
    assert rel.median() < 1e-2 and rel.max() < 0.1, (float(rel.median()), float(rel.max()))
    assert torch.equal(pc16.params, pc16.params_master.to(BF16))


@cuda_only
def test_anemone_at_bf16_accumulates_small_steps():
    """Anemone (`step_size_rescaling`) adds `(1 - s) * P_td * theta` to the flows and takes a step of size one, so at bf16
    that term must come from the master copy: built from the stored rounding, every step restarted the copy from it and
    an update below half a bf16 ulp never landed (MEASURED: near convergence, every parameter fp32 moved by more than 2
    ulps in 40 steps stayed where it was). Five steps of size 1e-3 on one batch -- the same direction each time -- from
    one state: the master copy travels as far as the fp32 parameters do, not one step's worth."""
    ns, data = _training_setup()
    moved = {}
    for dtype in (torch.float32, BF16):
        pc = juice.compile(ns, verbose = False, param_dtype = BF16).to(DEV).set_param_dtype(dtype)    # one start
        opt = juice.optim.Anemone(pc, step_size = 1e-3, momentum = 0.0)
        start = pc._params_in_fp32().clone()
        for _ in range(5):
            pc(data[:256]).mean().backward(); opt.step()
        moved[dtype] = pc._params_in_fp32() - start
        if dtype is BF16:
            assert torch.equal(pc.params, pc.params_master.to(BF16))
    d32, d16 = moved[torch.float32], moved[BF16]
    big = d32.abs() > 1e-6
    ratio = (d16[big] / d32[big]).median()
    assert big.sum() > 1000 and 0.8 < float(ratio) < 1.25, (int(big.sum()), float(ratio))


@cuda_only
def test_the_master_copy_lifecycle():
    ns, data = _training_setup()
    pc32 = juice.compile(ns, verbose = False).to(DEV)
    pc = juice.compile(ns, verbose = False, param_dtype = BF16).to(DEV)
    assert pc.params_master is None
    opt = juice.optim.MiniBatchEM(pc, step_size = 0.1, pseudocount = 0.1)
    pc(data[:256]).mean().backward(); opt.step()
    assert pc.params_master is not None and pc.params_master.device == pc.params.device
    assert "params_master" not in pc.state_dict()                      # not part of the state
    # the nodes get the master's fp32 values: recompiling them gives those, renormalized (compilation normalizes the
    # nodes' parameters, and an update of step size 0.1 from an upcast leaves them ~1e-3 off a sum of one) -- to fp32
    # rounding, far below the bf16 rounding they would carry from `params`
    from pyjuice.model.backend import normalize_parameters
    pc.update_parameters()
    re = juice.compile(ns, verbose = False).to(DEV)
    m = pc.params_master.clone()
    normalize_parameters(m, pc.par_update_kwargs, 0.0)                 # (in place)
    nz = m > 1e-6
    rel = ((re.params - m).abs() / m.clamp_min(1e-12))[nz]
    rel16 = ((re.params - pc.params.float()).abs() / m.clamp_min(1e-12))[nz]
    assert rel.max() < 1e-5, float(rel.max())
    assert rel16.max() > 1e-4                                          # (the check can tell the two apart)
    # a device move takes it along (a registered buffer)
    pc.cpu()
    assert pc.params_master.device.type == "cpu"
    pc.to(DEV)
    assert pc.params_master.device == pc.params.device
    # dropped by everything that rewrites `params`, and on request
    pc.drop_params_master()
    assert pc.params_master is None
    pc(data[:256]).mean().backward(); opt.step()
    assert pc.params_master is not None
    pc.load_state_dict(pc32.state_dict())
    assert pc.params_master is None and pc.params.dtype == BF16 and torch.equal(pc.params, pc32.params.to(BF16))
    pc(data[:256]).mean().backward(); opt.step()
    holder = torch.nn.ModuleDict({"pc": pc})                           # loaded as part of another module: dropped too
    holder.load_state_dict({"pc." + k: v for k, v in pc32.state_dict().items()})
    assert pc.params_master is None and torch.equal(pc.params, pc32.params.to(BF16))
    pc(data[:256]).mean().backward(); opt.step()
    master, requires_grad = pc.params_master.clone(), pc.params.requires_grad
    pc.set_param_dtype(torch.float32)                                   # the master copy's values, not their rounding
    assert pc.params_master is None and pc.params.dtype == torch.float32 and torch.equal(pc.params, master)
    assert pc.params.requires_grad == requires_grad


# ---------------------------------------------------------------------------------------------------------------------
# The CUDA kernels' parameter loads, the tuning cache
# ---------------------------------------------------------------------------------------------------------------------

def _misaligned_circuit():
    """A circuit whose 64-block top layer's parameters start at 4 mod 8: compilation packs the nodes' parameter ranges
    without padding, and the sum nodes below hold 192 + 192 + 12 + 16 (+ 4096 + 256 above them) parameters -- harmless at
    fp32 (the kernels' 16-byte loads take 4 floats), misaligned for one load of 8 bf16."""
    from pyjuice.nodes import inputs, multiply, summate
    torch.manual_seed(0)
    C = dists.Categorical(num_cats = 16)
    s0 = summate(inputs(0, num_node_blocks = 3, block_size = 1, dist = C), num_node_blocks = 16, block_size = 4)
    s1 = summate(inputs(1, num_node_blocks = 3, block_size = 1, dist = C), num_node_blocks = 16, block_size = 4)
    t2 = summate(inputs(2, num_node_blocks = 3, block_size = 1, dist = C), num_node_blocks = 1, block_size = 4)
    u3 = summate(inputs(3, num_node_blocks = 2, block_size = 2, dist = C), num_node_blocks = 1, block_size = 4)
    a = summate(multiply(s0, s1), num_node_blocks = 16, block_size = 4)
    b = summate(multiply(t2, u3), num_node_blocks = 16, block_size = 4)
    top = summate(multiply(a, b), num_node_blocks = 4, block_size = 64)
    root = summate(multiply(top), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    return root, top


@cuda_only
@pytest.mark.parametrize("B", [8, 128])
def test_misaligned_parameter_ranges_at_bf16(B):
    """The CUDA kernels' gates check the parameters' alignment for their dtype (`SumLayer._cuda_param_loads_aligned`):
    on bf16 parameters whose ranges sit at 4 mod 8 the layer takes the Triton kernels -- MEASURED before the check: CUDA
    "misaligned address" at batch 128 -- and agrees with the same values stored in fp32."""
    from pyjuice.layer import sum_layer as SL
    root, top = _misaligned_circuit()
    pc16 = juice.compile(root, verbose = False, param_dtype = BF16).to(DEV)
    pc32 = juice.compile(root, verbose = False, param_dtype = BF16).to(DEV).set_param_dtype(torch.float32)
    assert top._param_range[0] % 8 == 4                                # (the construction holds)
    x = torch.randint(0, 16, (B, 4), device = DEV)
    lls = [pc(x) for pc in (pc32, pc16)]
    assert torch.allclose(lls[1], lls[0], atol = 1e-4, rtol = 1e-5), float((lls[1] - lls[0]).abs().max())
    for pc in (pc32, pc16):
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)
    torch.cuda.synchronize()
    assert torch.allclose(pc16.param_flows, pc32.param_flows, atol = 1e-4 * pc32.param_flows.abs().max().item(), rtol = 1e-3)
    if B == 128 and SL.cuda_kernels.is_available():                    # the gate looked, and declined
        flags = [v for g in pc16.inner_layer_groups for l in g for v in getattr(l, "_cached_param_align8", {}).values()]
        assert False in flags, flags


@cuda_only
def test_cuda_kernels_refuse_other_parameter_dtypes(monkeypatch):
    """The CUDA kernels are instantiated for fp32 and bf16 parameters and check (as the `data_ptr<float>()` they had did):
    parameters left in another dtype -- fp16 here, written past `set_param_dtype`, which refuses it -- raise instead of
    being reinterpreted. Every timed gate is forced onto its last (CUDA) candidate, so the check is what is tested."""
    import pyjuice.layer.kernels.autotune as at
    from pyjuice.layer import sum_layer as SL
    if not SL.cuda_kernels.is_available():
        pytest.skip("no CUDA sum kernels here")
    pc, _, V, C = _pair("hmm")
    pc.params.data = pc.params.data.half()
    monkeypatch.setattr(at, "_CACHE", dict())
    monkeypatch.setattr(at, "best_of", lambda candidates, *args, **kwargs: candidates[-1][0])
    with pytest.raises(RuntimeError, match = "Float or BFloat16"):
        pc(torch.randint(0, C, (128, V), device = DEV))
        torch.cuda.synchronize()


@cuda_only
def test_tuned_choices_are_kept_per_parameter_dtype(monkeypatch):
    """The process-wide tuning cache keys on the parameters' dtype as well as the shape: an fp32 circuit next to a bf16
    one of the same shape tunes its own launches (the kernels read the parameters as stored; their speeds differ)."""
    import pyjuice.layer.kernels.autotune as at
    monkeypatch.setattr(at, "_CACHE", dict())                         # (whatever earlier tests tuned)
    monkeypatch.setattr(at, "ENABLED", True)
    pc32, pc16, V, C = _pair("hclt")
    x = torch.randint(0, C, (64, V), device = DEV)
    for pc in (pc16, pc32):
        pc.init_param_flows(flows_memory = 0.0)
        pc(x)
        pc.backward(x, flows_memory = 1.0)
    keys = [k[1] for k in at._CACHE if isinstance(k[1], tuple)]
    dtypes = {e for k in keys for e in k if isinstance(e, torch.dtype)}
    assert dtypes == {torch.float32, BF16}, dtypes


# ---------------------------------------------------------------------------------------------------------------------
# External parameters (BlockScale) at bf16
# ---------------------------------------------------------------------------------------------------------------------

def _blockscale_pair(B):
    from pyjuice.nodes import inputs, multiply, summate, BlockScaleSumParams
    torch.manual_seed(0)
    with juice.set_block_size(32):
        chain = inputs(0, num_node_blocks = 4, dist = dists.Categorical(num_cats = 8))
        transitions = []
        for t in range(1, 6):
            emit = inputs(t, num_node_blocks = 4, dist = dists.Categorical(num_cats = 8))
            chain = summate(multiply(chain, emit), num_node_blocks = 4,
                            external_params = BlockScaleSumParams(ch_block_size = 32))
            transitions.append(chain)
        root = summate(multiply(chain), num_node_blocks = 1, block_size = 1)
    root.init_parameters(perturbation = 2.0)
    pc32 = juice.compile(root, verbose = False).to(DEV)
    pc16 = juice.compile(root, verbose = False, param_dtype = BF16).to(DEV)
    pc32.set_param_dtype(BF16).set_param_dtype(torch.float32)           # the same (rounded) values
    g = torch.Generator(device = DEV).manual_seed(2)
    gates = {}
    for ns in transitions:
        shapes = ns.external_params.tensor_shapes(ns, B)
        tensors = tuple(torch.randn(s, device = DEV, generator = g) * 0.5 - 0.5 for s in shapes)
        gates[ns] = tensors if len(tensors) > 1 else tensors[0]
    return pc32, pc16, gates


@cuda_only
@pytest.mark.parametrize("B", [8, 64])
def test_blockscale_at_bf16(B):
    """The gated forward and backward (the BlockScale kernels, small-batch ones below 16) on bf16-stored parameters
    against the same values stored in fp32, and an EM step (its conditional M-step on the master copy)."""
    pc32, pc16, gates = _blockscale_pair(B)
    x = torch.randint(0, 8, (B, 6), device = DEV)
    lls = [pc(x, sum_external_params = gates) for pc in (pc32, pc16)]
    assert torch.allclose(lls[1], lls[0], atol = 1e-4, rtol = 1e-5), float((lls[1] - lls[0]).abs().max())
    for pc in (pc32, pc16):
        pc.init_param_flows(flows_memory = 0.0)
        pc(x, sum_external_params = gates)
        pc.backward(x, flows_memory = 1.0, sum_external_params = gates)
    assert torch.allclose(pc16.param_flows, pc32.param_flows, atol = 1e-4 * pc32.param_flows.abs().max().item(), rtol = 1e-3)
    for pc in (pc32, pc16):
        pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)
    assert torch.allclose(pc16.params_master, pc32.params, atol = 1e-6, rtol = 1e-4)
    assert torch.equal(pc16.params, pc16.params_master.to(BF16))


@cuda_only
@pytest.mark.parametrize("fold", [True, False], ids = ["fused_passes", "passes_of_their_own"])
def test_blockscale_em_at_bf16_keeps_a_write_between_steps(fold, monkeypatch):
    """`pc.params.data[...] = x` on a gated range between two EM steps of a bf16 circuit: its conditional M-step starts
    from x -- the master copy's check in the update's own passes (a range past the L2: forced here) or a pass of its
    own -- against an fp32 twin given the same write; and `params` is the master copy's rounding after each step. The
    write is not uniform within a node's children (x1.5 / x0.5 alternating), so a renormalization cannot undo it."""
    from pyjuice.nodes.external_params.external_params import ExternalSumParams
    monkeypatch.setattr(ExternalSumParams, "_fold_master", staticmethod(lambda params, n: fold))
    B = 64
    pc32, pc16, gates = _blockscale_pair(B)
    x = torch.randint(0, 8, (2 * B, 6), device = DEV)

    def step(pc, xb):
        pc.init_param_flows(flows_memory = 0.0)
        pc(xb, sum_external_params = gates)
        pc.backward(xb, flows_memory = 1.0, sum_external_params = gates)
        pc.mini_batch_em(step_size = 0.5, pseudocount = 0.1)

    for pc in (pc32, pc16):
        step(pc, x[:B])                                                # (the master copy exists)
    assert torch.equal(pc16.params, pc16.params_master.to(BF16))
    ns = next(iter(pc16._denom_correction_nss()))                       # a gated range the M-step corrects
    ps, pe = ns._param_range
    factor = torch.tensor([1.5, 0.5], device = DEV).repeat((pe - ps) // 2)
    written = (pc16.params.data[ps:pe].float() * factor).to(BF16)
    pc16.params.data[ps:pe] = written
    pc32.params.data[ps:pe] = written.float()
    for pc in (pc32, pc16):
        step(pc, x[B:])
    rel = (pc16.params_master[ps:pe] - pc32.params[ps:pe]).abs() / pc32.params[ps:pe].clamp_min(1e-12)
    assert rel.median() < 1e-3 and rel.max() < 1e-2, (float(rel.median()), float(rel.max()))
    assert torch.equal(pc16.params, pc16.params_master.to(BF16))


@cuda_only
@pytest.mark.parametrize("B", [8, 64])
def test_blockscale_gate_gradients_at_bf16(B):
    """`d LL / d log phi` on bf16-stored parameters against the same values in fp32. Its log-Z term sums theta per gate
    (`_sigma`) -- in fp32 whatever the storage (MEASURED: summed in bf16, the gradients came out 0.4-0.5% off). The bf16
    circuits: one compiled so, one that ran its gated passes at fp32 first and was then converted (`set_param_dtype`
    drops the plans and sums built for the old tensor, whose version a new one restarts)."""
    from pyjuice.layer.external_sum_layer import EXTERNAL_PARAMS_GRAD_KWARG
    pc32, pc16, gates = _blockscale_pair(B)
    pc_conv = juice.compile(pc32.root_ns, verbose = False, param_dtype = BF16).to(DEV).set_param_dtype(torch.float32)
    x = torch.randint(0, 8, (B, 6), device = DEV)

    def grads(pc):
        dsts = {ns: torch.zeros_like(g) for ns, g in gates.items()}
        pc.init_param_flows(flows_memory = 0.0)
        pc(x, sum_external_params = gates)
        pc.backward(x, flows_memory = 1.0, sum_external_params = gates, **{EXTERNAL_PARAMS_GRAD_KWARG: dsts})
        return torch.cat([d.flatten() for d in dsts.values()])

    g32 = grads(pc32)
    grads(pc_conv)                                                      # (its fp32 plans and sums built)
    pc_conv.set_param_dtype(BF16)
    # (two fp32 circuits, their forks timed apart, differ by up to 9e-5 of the largest gradient -- MEASURED)
    for g16 in (grads(pc16), grads(pc_conv)):
        assert torch.allclose(g16, g32, atol = 5e-4 * g32.abs().max().item(), rtol = 0.0), \
            float((g16 - g32).abs().max() / g32.abs().max())
