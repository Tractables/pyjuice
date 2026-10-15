"""
The fused sum kernels' split reductions (:mod:`pyjuice.constraints.backends.dfa_exact.kernels.sum`): a launch of few,
long programs splits every program's reduction -- over a child block's parents (top-down), a node block's children
(forward) -- in shares, each a program of its own, which the last share in merges. Every kind of PC of conftest.py,
plain and gated, and wider circuits whose forward reductions split too (node and child gates finer than the blocks,
aliased children next to element ones, chunks the shares cannot divide evenly or would split mid-chunk), at batches
of one tile of columns and of several: marginals and conditionals split as far as their reductions allow are the
unsplit ones up to the merge's rounding, the same bit for bit call after call, and leave every counter at 0; and
the rule that picks the split.
"""
import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc
import pyjuice.nodes.distributions as dists
from pyjuice.constraints.backends.dfa_exact.kernels import sum as lifted_sum
from pyjuice.nodes import inputs, multiply, summate

V = 3
DEV = torch.device("cuda:0")
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "left_linear": 6, "pd": 6, "pd_prod_dominated": 6,
         "pd_blockified": 6, "hand": 5, "hand_permuted": 5, "hand_unit": 4, "hand_left": 5, "hand_shared": 4,
         "hand_fold": 4, "wide_hmm": 5, "wide_aliased": 3, "hmm48": 5}
WIDE = ("wide_hmm", "wide_aliased", "hmm48")


def wide_aliased():
    """A sum of 64 nodes in blocks of 16 over 48 element children -- once the aliased ones are apart, two chunks of
    edges that two shares split mid-chunk unless the shares are whole chunks -- and an aliased copy of a sum of 16."""
    cat = lambda: dists.Categorical(num_cats = V)
    x = [inputs(v, num_node_blocks = 3, block_size = 16, dist = cat()) for v in range(2)]
    y = [inputs(v, num_node_blocks = 1, block_size = 16, dist = cat()) for v in range(2)]
    s1 = summate(multiply(y[0], y[1]), num_node_blocks = 1, block_size = 16)
    s2 = summate(multiply(x[0], x[1]), multiply(s1), num_node_blocks = 4, block_size = 16)
    x2 = inputs(2, num_node_blocks = 4, block_size = 16, dist = cat())
    return summate(multiply(s2, x2), num_node_blocks = 1, block_size = 1)


def circuit(kind, gated, build_pc, gate_sums):
    n = KINDS[kind]
    child_gate = lambda ns: next(d for d in (8, 2, 1) if ns.ch_block_size % d == 0)
    node_gate = lambda ns: next(d for d in (8, 1) if ns.block_size % d == 0)
    if kind == "hmm48":                         # chunks of 16 children, one gate each: 3 per node block
        child_gate = node_gate = lambda ns: ns.ch_block_size
    if kind in WIDE:
        torch.manual_seed(0)
        latents = 48 if kind == "hmm48" else 64
        ns = wide_aliased() if kind == "wide_aliased" else \
            juice.structures.HMM(seq_length = n, num_latents = latents, num_emits = V, block_size = 16)
        if gated:
            gate_sums(ns, child_gate = child_gate, node_gate = node_gate, which = lambda s: s.num_nodes == latents)
        ns.init_parameters(perturbation = 4.0)
        return juice.compile(ns, verbose = False).to(DEV)
    return build_pc(kind, n, V, perturbation = 4.0, gates = dict(child_gate = child_gate) if gated else None)


def evidence(n, B, seed = 0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, V, (B, n), generator = g)
    missing = torch.rand(B, n, generator = g) < 0.5
    missing[0] = True
    return data.to(DEV), missing.to(DEV)


def gates_of(pc, B, seed = 0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for ns in pc.external_params_nodes:
        owner = ns.external_params.storage_owner(ns)
        if owner not in out:
            out[owner] = (torch.randn(ns.external_params.tensor_shapes(owner, B)[0], generator = g) * 2).to(DEV)
    return out


class Splits:
    """The SPLIT and grid of every fused sum launch, by kernel."""

    def __init__(self, monkeypatch):
        self.seen = {"forward": [], "top_down": []}
        for name, kind in (("_lifted_sum_kernel", "forward"), ("_lifted_sum_aux_kernel", "forward"),
                           ("_lifted_sum_top_down_kernel", "top_down"),
                           ("_lifted_sum_top_down_aux_kernel", "top_down")):
            monkeypatch.setattr(lifted_sum, name, self._spy(getattr(lifted_sum, name), kind))

    def _spy(self, kernel, kind):
        seen = self.seen[kind]

        class Spy:
            def __getitem__(self, grid):
                launch = kernel[grid]

                def run(*args, **kwargs):
                    seen.append((kwargs.get("SPLIT", 1), tuple(grid), kwargs.get("PRE", False)))
                    return launch(*args, **kwargs)
                return run
        return Spy()


def assert_close(got, want):
    got, want = got.reshape(-1), want.reshape(-1)
    assert torch.equal(torch.isneginf(got), torch.isneginf(want))
    fin = torch.isfinite(want)
    assert torch.isfinite(got[fin]).all()
    assert ((got[fin] - want[fin]).abs() <= 1e-5 + 1e-5 * want[fin].abs()).all(), (got[fin] - want[fin]).abs().max()


def queries(cc, data, missing, kw):
    out = []
    for storage in ("dense", "compact"):
        with cc.decoder(storage = storage, **kw) as decoder:
            out += [decoder.marginal(data, missing), decoder.conditional(data, missing)]
    return out


def compare(kind, gated, B, build_pc, gate_sums, set_split, monkeypatch) -> dict:
    """Split as far as every launch's reductions allow, then not at all: the same marginals and conditionals, dense
    and compact storage; the split launches (SPLIT, grid) by kernel."""
    pc = circuit(kind, gated, build_pc, gate_sums)
    data, missing = evidence(KINDS[kind], B)
    kw = dict(sum_external_params = gates_of(pc, B)) if gated else {}
    cc = jc.compile(jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)), pc)
    splits = Splits(monkeypatch)
    set_split("max")
    split = queries(cc, data, missing, kw)
    split_launches = {k: [(s, g) for s, g, _ in v if s > 1] for k, v in splits.seen.items()}
    set_split("off")
    count = {k: len(v) for k, v in splits.seen.items()}
    whole = queries(cc, data, missing, kw)
    assert all(s == 1 for k, v in splits.seen.items() for s, _, _ in v[count[k]:])
    for got, want in zip(split, whole):
        assert_close(got, want)
    return split_launches


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_split_reductions_are_the_unsplit_ones(kind, gated, build_pc, gate_sums, set_split, monkeypatch):
    """Every kind, plain or gated: the wider circuits split both kernels."""
    launches = compare(kind, gated, 3, build_pc, gate_sums, set_split, monkeypatch)
    if kind in WIDE:
        assert launches["forward"] and launches["top_down"], launches


@pytest.mark.parametrize("kind", ["wide_hmm", "hmm48"])
def test_split_launches_of_several_column_tiles(kind, build_pc, gate_sums, set_split, monkeypatch):
    """A batch wide enough for several tiles of columns (the HMMs' automaton columns): every tile its own shares and
    counter."""
    launches = compare(kind, True, 24, build_pc, gate_sums, set_split, monkeypatch)
    for k in ("forward", "top_down"):
        assert any(g[1] > 1 for _, g in launches[k]), (k, launches[k])


@pytest.mark.parametrize("kind", list(WIDE))
def test_split_calls_are_reproducible_and_reset_their_counters(kind, build_pc, gate_sums, set_split):
    """Split launches merge their shares in a fixed order: a call is the same bit for bit when repeated; and the
    last share in sets its tile's counter back to 0, so every counter is 0 between launches."""
    set_split("max")
    pc = circuit(kind, True, build_pc, gate_sums)
    data, missing = evidence(KINDS[kind], 4, seed = 1)
    kw = dict(sum_external_params = gates_of(pc, 4, seed = 1))
    cc = jc.compile(jc.DFA.contains([[0, 1]], V), pc)
    first, again = queries(cc, data, missing, kw), queries(cc, data, missing, kw)
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    torch.cuda.synchronize()
    for buf in lifted_sum._SPLIT_BUFFERS.values():
        assert int(buf[:lifted_sum.SPLIT_TILES].view(torch.int32).abs().sum()) == 0


def test_split_count():
    """No split once a launch has the programs it needs; else a power of two, at most ``SPLIT_MAX``, shares of at
    least ``SPLIT_MIN_SHARE`` chunks."""
    sms = lifted_sum._num_sms(DEV)
    need = lifted_sum.SPLIT_PROGRAMS_PER_SM * sms
    assert lifted_sum.split_count(need, 1 << 10, DEV) == 1
    assert lifted_sum.split_count(1, 1 << 10, DEV) == lifted_sum.SPLIT_MAX
    assert lifted_sum.split_count(1, 2 * lifted_sum.SPLIT_MIN_SHARE - 1, DEV) == 1
    assert lifted_sum.split_count(1, 4 * lifted_sum.SPLIT_MIN_SHARE, DEV) == 4
    for programs in (1, 3, 17, need // 3, need - 1):
        for chunks in (1, 5, 32, 1000):
            s = lifted_sum.split_count(programs, chunks, DEV)
            assert s & (s - 1) == 0 and s <= lifted_sum.SPLIT_MAX
            assert s == 1 or chunks // s >= lifted_sum.SPLIT_MIN_SHARE
            assert s == 1 or programs * s // 2 < need


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("kind", list(KINDS))
def test_pre_passes_are_the_plain_loops(kind, gated, build_pc, gate_sums, monkeypatch):
    """The fused kernels' pre-passes (every parent chunk's -- children chunk's -- shifted exponentials computed once)
    taken by every launch that may take them, then by none: the same marginals and conditionals; the wider circuits
    take them in the top-down, and those with node blocks over the same children (an HMM layer) in the forward."""
    pc = circuit(kind, gated, build_pc, gate_sums)
    data, missing = evidence(KINDS[kind], 3)
    kw = dict(sum_external_params = gates_of(pc, 3)) if gated else {}
    cc = jc.compile(jc.DFA.contains([[0, 1, 1], [2, 0]], V) & jc.Not(jc.DFA.contains([[2, 2]], V)), pc)
    splits = Splits(monkeypatch)
    monkeypatch.setattr(lifted_sum, "PRE_MIN_TILES", 1)
    pre = queries(cc, data, missing, kw)
    took = {k: any(p for _, _, p in v) for k, v in splits.seen.items()}
    if kind in WIDE:
        assert took["top_down"], splits.seen["top_down"]
    if kind == "hmm48" or (kind == "wide_hmm" and not gated):   # gates of 8 children are per edge: no forward pre
        assert took["forward"], splits.seen["forward"]
    monkeypatch.setattr(lifted_sum, "PRE_MIN_TILES", 1 << 30)
    count = {k: len(v) for k, v in splits.seen.items()}
    plain = queries(cc, data, missing, kw)
    assert not any(p for k, v in splits.seen.items() for _, _, p in v[count[k]:])
    for got, want in zip(pre, plain):
        assert_close(got, want)
