"""
The fused sums' CUDA kernel (:mod:`pyjuice.constraints.backends.dfa_exact.kernels.c`, ``fused_sum.cu``): the forward
launches of node blocks of 32 over shared children that the pre-pass serves, and the top-down launches of child blocks
of 32 children or a multiple under shared parents, run on it (taken for every such launch here, whatever its size) and
give the Triton kernels' marginals and conditionals -- HMMs of 4 and 5 node blocks (the latter's edge tables padded to
256: chunks of padding left out), plain, with external parameters per sample or one for every sample, soft evidence
inputs; dense storage (pruned columns) and compact (lanes); fp32 and TF32 products; a decoder's calls (held values) and
the circuit's own; batches whose columns fill no tile, tiles of 64 to 128 columns; launches split in shares of one
chunk (the merge kernel) or not; the weights' first entries read from a table (as when a block's weights are not one
run) or not; every launch's blocks given as an index list, last first, or not; copies of sums (the top-down's aliased
rows). At TF32 its products round to nearest: marginals stay near the float64 reference over 16 layers; inside a scope
that allows parameter copies they read the weights from copies in the kernel's fragment order, with the same results.
At bf16 (``precision = "bf16"``) the products are bf16 m16n8k16 ones, from a bf16 copy of the weights -- the scope's, or
a query's own -- and bf16 e.
"""
import inspect

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
from pyjuice.constraints.backends.dfa_exact.kernels import sum as lifted_sum
from pyjuice.constraints.backends.dfa_exact.kernels.c import get_module
from pyjuice.constraints.backends.dfa_exact.queries.backend import forward as forward_queries
from pyjuice.constraints.backends.dfa_exact.queries.backend import top_down as top_down_queries
from pyjuice.nodes import inputs as input_nodes, multiply, summate

V = 3
N = 5
DEV = torch.device("cuda:0")


def circuit(latents: int, external: bool, gate_sums, seed = 0):
    torch.manual_seed(seed)
    ns = juice.structures.GeneralizedHMM(seq_length = N, num_latents = latents, block_size = 32,
                                         input_dist = dists.SoftEvidenceCategorical(num_cats = V))
    if external:
        gate_sums(ns, child_gate = lambda s: 32, which = lambda s: s.num_nodes == latents)
    ns.init_parameters(perturbation = 4.0)
    return juice.compile(ns, verbose = False).to(DEV)


def aliased_circuit(external: bool, gate_sums):
    """Blocks of 32 over 3 variables: copies of two sums under a third (its children's rows the sums' own), and a
    product read by two of the sums."""
    torch.manual_seed(0)
    cat = lambda: dists.SoftEvidenceCategorical(num_cats = V)
    x = [input_nodes(v, num_node_blocks = 2, block_size = 32, dist = cat()) for v in range(2)]
    y = [input_nodes(v, num_node_blocks = 1, block_size = 32, dist = cat()) for v in range(2)]
    p = multiply(x[0], x[1])
    s1 = summate(p, num_node_blocks = 2, block_size = 32)
    s0 = summate(multiply(y[0], y[1]), num_node_blocks = 1, block_size = 32)
    s2 = summate(p, multiply(s1), multiply(s0), num_node_blocks = 2, block_size = 32)
    root = summate(multiply(s2, input_nodes(2, num_node_blocks = 2, block_size = 32, dist = cat())),
                   num_node_blocks = 1, block_size = 1)
    if external:
        gate_sums(root, child_gate = lambda s: 32, which = lambda s: s.block_size == 32)
    root.init_parameters(perturbation = 4.0)
    return juice.compile(root, verbose = False).to(DEV)


def inputs(pc, B: int, external: str, seed = 0, n = N):
    g = torch.Generator().manual_seed(seed)
    kw = dict(categorical_evidence_logp = torch.log_softmax(torch.randn(B, n, V, generator = g) * 2, dim = -1).to(DEV))
    if external != "none":
        rows = 1 if external == "shared" else B
        shape = lambda s: s.external_params.tensor_shapes(s, rows)[0]
        kw["sum_external_params"] = {s: (torch.randn(shape(s), generator = g) * 2).to(DEV)
                                     for s in pc.external_params_nodes}
    return kw


def calls(cc, B, kw, precision, storage, n = N, **options):
    """A decoder's marginals and conditionals left to right, then a fresh decoder's of other evidence (``options``:
    the decoders' others)."""
    g = torch.Generator().manual_seed(1)
    data = torch.randint(0, V, (B, n), generator = g).to(DEV)
    out = []
    with cc.decoder(precision = precision, storage = storage, **kw, **options) as decoder:
        for t in range(n):
            missing = torch.arange(n, device = DEV)[None, :].expand(B, n) >= t
            out.append(decoder.marginal(data, missing))
            out.append(decoder.conditional(data, missing, targets = [t]))
    missing = torch.rand(B, n, generator = g).to(DEV) < 0.5
    for query in ("marginal", "conditional"):                           # (a session's first call holds nothing)
        with cc.decoder(precision = precision, storage = storage, **kw, **options) as decoder:
            out.append(getattr(decoder, query)(data, missing))
    return out


def recorded_plans(monkeypatch):
    """The CUDA launches' plans (forward, top-down), as the launches make them."""
    plans, td_plans = [], []

    def recorded(real, into):
        def planned(*args, **kwargs):
            plan = real(*args, **kwargs)
            if plan is not None:
                into.append(plan)
            return plan
        return planned

    monkeypatch.setattr(lifted_sum, "_cuda_plan", recorded(lifted_sum._cuda_plan, plans))
    monkeypatch.setattr(lifted_sum, "_cuda_td_plan", recorded(lifted_sum._cuda_td_plan, td_plans))
    return plans, td_plans


def listed(monkeypatch):
    """Every fused launch given its blocks as an index list, last first (as a query plan's list)."""
    def with_list(real, key, count, size):
        def launch(*args, **kwargs):
            a = inspect.signature(real).bind(*args, **kwargs).arguments
            if a.get(key) is None:
                n = size(a)
                a[key] = torch.arange(n - 1, -1, -1, dtype = torch.int32, device = DEV)
                if count:
                    a["count"] = n
            return real(**a)
        return launch

    monkeypatch.setattr(forward_queries, "fused_sum",
                        with_list(lifted_sum.fused_sum, "sel", True, lambda a: a["cids"].size(0)))
    monkeypatch.setattr(top_down_queries, "fused_sum_top_down",
                        with_list(lifted_sum.fused_sum_top_down, "idx", False, lambda a: a["tables"]["cb_row"].numel()))


def operands(plan):
    """A CUDA plan's operands besides e (``_cuda_operands``): None, "tf32" or "bf16"."""
    return None if plan[4] is None else "bf16" if plan[4].dtype == torch.bfloat16 else "tf32"


def assert_same(got, want, precision):
    tol = 1e-5 if precision == "fp32" else 5e-3            # (bf16: against the Triton kernels' TF32, within ~1e-3)
    for i, (a, b) in enumerate(zip(got, want)):
        assert torch.equal(torch.isneginf(a), torch.isneginf(b)) and not torch.isnan(a).any(), i
        fin = torch.isfinite(b)
        assert ((a[fin] - b[fin]).abs() <= tol * (1 + b[fin].abs())).all(), (i, (a[fin] - b[fin]).abs().max())


@pytest.mark.parametrize("split, table, lists", [(False, False, False), (True, False, False), (False, True, False),
                                                 (True, True, False), (False, False, True), (True, False, True)])
@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("external", ["none", "per_sample", "shared"])
@pytest.mark.parametrize("latents, B", [(128, 3), (160, 37), (128, 70)])
def test_the_cuda_sums_are_the_triton_kernels(latents, B, external, storage, precision, split, table, lists, gate_sums,
                                             monkeypatch):
    mod = get_module()
    if mod is None:
        pytest.skip("the CUDA extension is unavailable")
    pc = circuit(latents, external != "none", gate_sums)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)), pc)
    kw = inputs(pc, B, external)
    plans, td_plans = recorded_plans(monkeypatch)
    if lists:
        listed(monkeypatch)
    if table:                                                           # (every block's weights are one run)
        tables, td_tables = lifted_sum._cuda_tables, lifted_sum._cuda_td_tables
        monkeypatch.setattr(lifted_sum, "_cuda_tables", lambda *a: (lambda t: t and t[:3] + (False,))(tables(*a)))
        monkeypatch.setattr(lifted_sum, "_cuda_td_tables",
                            lambda *a: (lambda t: t and t[:6] + (False,) + t[7:])(td_tables(*a)))
    monkeypatch.setattr(lifted_sum, "CUDA_MIN_TILES", 0)                # every launch it may take
    if split:
        monkeypatch.setattr(lifted_sum, "SPLIT_MIN_SHARE", 1)           # shares of a chunk
    got = calls(cc, B, kw, precision, storage)
    for ps in (plans, td_plans):                                        # (shares)
        assert ps and (any(p[3] > 1 for p in ps) if split else all(p[3] == 1 for p in ps)), [p[2:4] for p in ps]
    want_ops = {"fp32": None, "tf32": "tf32", "bf16": "bf16"}[precision]  # (fragment-ordered weights: a decoder's)
    assert all(operands(p) == want_ops for p in plans + td_plans)
    monkeypatch.setattr(lifted_sum, "SPLIT_MIN_SHARE", 8)
    monkeypatch.setattr(lifted_sum, "CUDA_MIN_TILES", 1 << 40)          # none
    want = calls(cc, B, kw, precision, storage)
    assert_same(got, want, precision)


@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
@pytest.mark.parametrize("storage", ["dense", "compact"])
@pytest.mark.parametrize("external", ["none", "per_sample"])
@pytest.mark.parametrize("split, lists", [(False, False), (True, False), (False, True)])
def test_the_cuda_top_down_writes_aliased_rows(external, storage, precision, split, lists, gate_sums, monkeypatch):
    """Copies of sums (:func:`aliased_circuit`): the CUDA top-down launches write into the copied sums' rows, as the
    Triton kernel does -- launches of two child blocks of two slices each, given as an index list or not."""
    if get_module() is None:
        pytest.skip("the CUDA extension is unavailable")
    n, B = 3, 5
    pc = aliased_circuit(external != "none", gate_sums)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    kw = inputs(pc, B, external, n = n)
    _, td_plans = recorded_plans(monkeypatch)
    if lists:
        listed(monkeypatch)
    monkeypatch.setattr(lifted_sum, "CUDA_MIN_TILES", 0)
    if split:
        monkeypatch.setattr(lifted_sum, "SPLIT_MIN_SHARE", 1)
    got = calls(cc, B, kw, precision, storage, n = n)
    assert any(bool((p[1][1] >= 0).any()) for p in td_plans), "no aliased rows"
    assert any(p[1][1].numel() > p[1][-1] and p[1][-1] > 1 for p in td_plans), "no launch of child blocks in slices"
    assert any(p[3] > 1 for p in td_plans) == split
    monkeypatch.setattr(lifted_sum, "SPLIT_MIN_SHARE", 8)
    monkeypatch.setattr(lifted_sum, "CUDA_MIN_TILES", 1 << 40)
    want = calls(cc, B, kw, precision, storage, n = n)
    assert_same(got, want, precision)


@pytest.mark.parametrize("precision", ["tf32", "bf16"])
@pytest.mark.parametrize("external", ["none", "per_sample"])
def test_the_weights_in_fragment_order(external, precision, gate_sums, set_split, monkeypatch):
    """The CUDA kernel reads the weights in its fragment order from copies (``_cuda_fragments``, one per direction):
    at TF32 in a decoder session that allows parameter copies (the default), at bf16 always -- outside such a session
    a query's own. A copy is built by the first launch that reads a part's tiles (parts reading the same tiles share
    it), dropped when the session (query) ends -- an in-place parameter update after it is read -- and the only
    weights the launches read (a copy of zeros: forward, every marginal -inf; top-down, the marginals as they were and
    other conditionals). Every result is bit-identical with and without a session's copies: at TF32 the products are
    those of the same operands in the same order as the kernel reading the weights themselves."""
    mod = get_module()
    if mod is None:
        pytest.skip("the CUDA extension is unavailable")
    B = 5
    pc = circuit(128, external != "none", gate_sums)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)), pc)
    kw = inputs(pc, B, external)
    monkeypatch.setattr(lifted_sum, "CUDA_MIN_TILES", 0)
    set_split("off")                                    # (a split follows the kernels' occupancy, which may differ)
    plans, td_plans = recorded_plans(monkeypatch)
    builds = []
    fragment_weights = mod.fragment_weights
    monkeypatch.setattr(mod, "fragment_weights",
                        lambda *a: builds.append((lifted_sum._tiles_key(a[2]), a[3])) or fragment_weights(*a))
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)

    want = calls(cc, B, kw, precision, "dense", allow_param_copy = False)
    no_copies = None if precision == "tf32" else "bf16"                 # (bf16: a query's own)
    assert plans and td_plans and all(operands(p) == no_copies for p in plans + td_plans)
    assert bool(builds) == (precision == "bf16")
    plans.clear(), td_plans.clear()
    assert same(calls(cc, B, kw, precision, "dense"), want)
    assert all(operands(p) == precision for p in plans + td_plans) and cc.backend.state._kept_stacks is None

    g = torch.Generator().manual_seed(2)
    data = torch.randint(0, V, (B, N), generator = g).to(DEV)
    missing = [torch.rand(B, N, generator = g).to(DEV) < 0.5 for _ in range(2)]
    builds.clear()
    with cc.decoder(precision = precision, **kw) as decoder:           # copied once a session, a tile set once
        decoder.conditional(data, missing[0])
        first = list(builds)
        decoder.conditional(data, missing[1])                           # (other evidence: every launch again)
    assert {td for _, td in first} == {False, True} and len(set(first)) == len(first) and builds == first

    with torch.no_grad():                                               # an in-place update between sessions
        pc.params.copy_(circuit(128, external != "none", gate_sums, seed = 1).params)
    updated = calls(cc, B, kw, precision, "dense", allow_param_copy = False)
    assert not same(updated, want)
    assert same(calls(cc, B, kw, precision, "dense"), updated)

    def queries():
        with cc.decoder(precision = precision, **kw) as decoder:
            marginal = decoder.marginal(data, missing[0])
        with cc.decoder(precision = precision, **kw) as decoder:
            return marginal, decoder.conditional(data, missing[0])

    marginal, conditional = queries()
    assert torch.isfinite(marginal).any()
    real = lifted_sum._cuda_fragments
    for zeroed in (False, True):                                        # forward, top-down
        monkeypatch.setattr(lifted_sum, "_cuda_fragments", lambda mod, frags, params, wbase, td, bf16 = False:
                            real(mod, frags, params, wbase, td, bf16) * (0.0 if td == zeroed else 1.0))
        m, c = queries()
        assert torch.isneginf(m).all() if not zeroed else torch.equal(m, marginal)
        assert not torch.equal(c, conditional)


def test_tf32_products_round_to_nearest(reference, monkeypatch):
    """At TF32 the CUDA kernel's products round both operands to the nearest TF32 value (the pre-pass rounds e, the
    kernel the weights): over an HMM of 16 steps its marginals keep within 2e-3 of the float64 reference on average,
    where operands truncated to TF32 drift below by ~6e-4 a layer."""
    if get_module() is None:
        pytest.skip("the CUDA extension is unavailable")
    n, B = 16, 8
    torch.manual_seed(0)
    ns = juice.structures.GeneralizedHMM(seq_length = n, num_latents = 128, block_size = 32,
                                         input_dist = dists.SoftEvidenceCategorical(num_cats = V))
    ns.init_parameters(perturbation = 4.0)
    pc = juice.compile(ns, verbose = False).to(DEV)
    cc = jc.compile(jc.DFA.contains([[1, 2]], V), pc)
    g = torch.Generator().manual_seed(0)
    kw = dict(categorical_evidence_logp = torch.log_softmax(torch.randn(B, n, V, generator = g) * 2, dim = -1).to(DEV))
    data = torch.randint(0, V, (B, n), generator = g).to(DEV)
    missing = torch.rand(B, n, generator = g).to(DEV) < 0.5
    want = reference.marginal(cc, data, missing, inputs = kw).view(-1).double()
    monkeypatch.setattr(lifted_sum, "CUDA_MIN_TILES", 0)
    with cc.decoder(precision = "tf32", **kw) as decoder:
        got = decoder.marginal(data, missing).view(-1).double()
    fin = torch.isfinite(want)
    assert torch.equal(torch.isfinite(got), fin)
    drift = float((got[fin] - want[fin]).mean())
    assert abs(drift) < 2e-3, drift
