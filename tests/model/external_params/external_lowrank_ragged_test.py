"""
`LowRankSumParams` on RAGGED / BLOCK-SPARSE topologies.

Every other low-rank test builds with `num_node_blocks = 1`, i.e. exactly one edge block, so no
padded slot can arise -- which is why the following went unnoticed. Compilation pads each row of
`ext_slots` out to the widest row and marks the padding with a **`-1` sentinel**; the kernels used to
read that as an offset, giving a NEGATIVE address. Concretely, before the fix:

* the forward disagreed with its own `forward_torch` oracle by 0.12-0.44 nats on every ragged
  topology -- roughly ten times the size of the correction being computed, so not a rounding matter;
* `compute-sanitizer` reported 194 invalid reads, `lowrank_wa_partial_kernel` reading 160 bytes
  BEFORE the buffer.

Note what does NOT distinguish the broken case: block-sparsity on its own is fine. `sparse` below has
non-adjacent child blocks but is RECTANGULAR -- every row has the same number of edge blocks, so it
carries no padding -- and it was correct throughout. The dividing line is padding, so the topology
list keeps both kinds and the tests assert against the oracle rather than against each other.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyjuice as juice
import pyjuice.nodes.distributions as dists
from pyjuice.nodes import inputs, multiply, summate
from pyjuice.nodes import LowRankSumParams

import pyjuice.nodes.external_params.lowrank as _lowrank_mod

from external_blockscale_test import NUM_CATS, cuda_only
from external_blockscale_ragged_test import TOPOLOGIES, NONDENSE, _eids


RANK = 4
BATCH = 32

# The failure this file exists to catch is an ILLEGAL MEMORY ACCESS, and a CUDA error is sticky: once
# one kernel faults, every later CUDA call in the process raises too. So a regression here does not
# report the tests that actually detect it -- it reports every test that happens to run afterwards,
# whether or not that test has anything to say about padding. Measured on the unfixed kernels: 6 real
# failures as a whole file became 20, including tests that only read `ext_slots` and pass in isolation.
#
# The fixture below probes the context after each test and SKIPS the rest with the name of the test
# that broke it, so the first failure stays the only failure and the rest are visibly collateral.
_poisoned_by = {"test": None}


@pytest.fixture(autouse = True)
def _stop_at_the_first_cuda_fault(request):
    if _poisoned_by["test"] is not None:
        pytest.skip(f"CUDA context already faulted in {_poisoned_by['test']}; this result would be "
                    f"collateral, not an independent finding -- fix that test first")

    yield

    if not torch.cuda.is_available():
        return

    try:
        torch.zeros(1, device = "cuda:0").add_(1.0)
        torch.cuda.synchronize()
    except Exception:
        _poisoned_by["test"] = request.node.name

# Kept small deliberately: these run on every topology and the point is coverage of the addressing,
# not of scale.
ALL_TOPOLOGIES = ["dense"] + NONDENSE


def _needs_lowrank_cuda():
    from pyjuice.nodes.external_params.kernels.c import is_available

    return is_available()


needs_lowrank_cuda = pytest.mark.skipif(
    not _needs_lowrank_cuda(), reason = "the low-rank CUDA extension is unavailable")


def _build(name, rank = RANK, variant = None, seed = 0):
    edge_ids, bs, ch_bs, n_ch = TOPOLOGIES[name]
    n_nb = int(edge_ids[0].max()) + 1

    torch.manual_seed(seed)
    with juice.set_block_size(ch_bs):
        ni = [inputs(v, num_node_blocks = n_ch, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(2)]
        prod = multiply(*ni)

    kw = {} if variant is None else {"variant": variant}
    ns = summate(prod, num_node_blocks = n_nb, edge_ids = edge_ids, block_size = bs,
                 external_params = LowRankSumParams(rank = rank, **kw))
    root = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, ns


def _factors(ns, batch = BATCH, seed = 3):
    su, sv = ns.external_params.tensor_shapes(ns, batch)
    torch.manual_seed(seed)
    return torch.randn(su, device = "cuda:0") * 0.5, torch.randn(sv, device = "cuda:0") * 0.5


def _lls(name, force_reference, variant = None, batch = BATCH, factors = None):
    """
    One forward, either through the kernels or through `forward_torch`.

    The PC is COMPILED FRESH for each path on purpose. `forward_layer` caches its resolved plan on the
    layer (`_lr_fw_plan`) at the first forward, so flipping `_kernel_applicable` on an already-used PC
    leaves the first path in place and silently compares a run to itself -- which reads as a perfect
    0.00e+00 match on every topology.
    """
    original = _lowrank_mod.LowRankSumParams._kernel_applicable
    if force_reference:
        _lowrank_mod.LowRankSumParams._kernel_applicable = lambda *a, **k: False
    try:
        root, ns = _build(name, variant = variant)
        pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))

        torch.manual_seed(7)
        data = torch.randint(0, NUM_CATS, [batch, 2], device = "cuda:0")
        U, V = factors if factors is not None else _factors(ns, batch)

        return pc(data, sum_external_params = {ns: (U, V)}).detach().clone(), pc, ns
    finally:
        _lowrank_mod.LowRankSumParams._kernel_applicable = original


@cuda_only
@needs_lowrank_cuda
@pytest.mark.parametrize("name", ALL_TOPOLOGIES)
def test_forward_matches_the_torch_reference(name):
    """The whole bug, on every shape: a padded slot must contribute nothing rather than a bad address."""
    root, ns = _build(name)
    factors = _factors(ns)

    kernel, _, _ = _lls(name, force_reference = False, factors = factors)
    reference, _, _ = _lls(name, force_reference = True, factors = factors)

    # The correction is worth ~0.01-0.04 nats here, and the bug was worth 0.12-0.44 -- so a tolerance
    # anywhere below the correction's own size separates them. This one is float noise.
    assert torch.all((kernel - reference).abs() < 1e-4), \
        f"{name}: kernel and reference differ by {float((kernel - reference).abs().max()):.3e}"


@cuda_only
@needs_lowrank_cuda
@pytest.mark.parametrize("name", ["dense", "pad_tiles", "sparse_ragged"])
def test_the_triton_fallback_matches_the_torch_reference(name):
    """`variant="split2"` is what runs where `nvcc` is unavailable, and it had the same bug."""
    import pyjuice.nodes.external_params.kernels as kernels_pkg

    root, ns = _build(name, variant = "split2")
    factors = _factors(ns)

    calls = {"n": 0}
    original = kernels_pkg.fw_lowrank

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    # Patched on the PACKAGE, which is where `forward_layer`'s `from .kernels import fw_lowrank`
    # binds. Patching `kernels.lowrank_forward` instead leaves the package attribute untouched and
    # the counter reads zero while the Triton path is in fact running.
    kernels_pkg.fw_lowrank = counted
    try:
        kernel, _, _ = _lls(name, force_reference = False, variant = "split2", factors = factors)
    finally:
        kernels_pkg.fw_lowrank = original

    reference, _, _ = _lls(name, force_reference = True, variant = "split2", factors = factors)

    assert calls["n"] > 0, "the Triton fallback never ran, so this asserts nothing"
    assert torch.all((kernel - reference).abs() < 1e-4), \
        f"{name}: Triton fallback differs by {float((kernel - reference).abs().max()):.3e}"


@cuda_only
@needs_lowrank_cuda
@pytest.mark.parametrize("name", ["pad_tiles", "split", "sparse_ragged", "narrow_ragged", "pad_big"])
def test_padded_edge_blocks_carry_the_minus_one_sentinel(name):
    """
    The premise the kernels' guard rests on: padding is marked, and only on ragged shapes.

    If compilation ever stopped emitting `-1` -- say by padding with a duplicate of a real slot -- the
    guard would go quiet and the numeric tests above would be the only thing standing between a real
    edge and a double-counted one.
    """
    root, ns = _build(name)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]

    xu = layer.ext_slots[0][0]
    xv = layer.ext_slots[1][0]

    assert int((xu < 0).sum()) > 0, f"{name} is in the ragged list but no `xu` slot is padded"
    assert int((xv < 0).sum()) > 0, f"{name} is in the ragged list but no `xv` slot is padded"
    # -1 exactly, not "some negative number": the kernels test `< 0`, but a stray -2 would mean the
    # convention had changed under them.
    assert int(xu.min()) == -1 and int(xv.min()) == -1


@cuda_only
@needs_lowrank_cuda
def test_a_fully_connected_layer_has_no_padded_slot():
    """The control for the test above -- otherwise it could pass on a bug that pads everything."""
    root, ns = _build("dense")
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))
    layer = [l for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers
             if hasattr(l, "external_node_infos")][0]

    assert int((layer.ext_slots[0][0] < 0).sum()) == 0
    assert int((layer.ext_slots[1][0] < 0).sum()) == 0


def _two_lowrank_ns(edge_ids = None, bs = 64, ch_bs = 64, nb = 4, seed = 0):
    """Two gated `ns` at the same depth with the same signature -> ONE sum layer."""
    torch.manual_seed(seed)
    with juice.set_block_size(ch_bs):
        ni = [inputs(v, num_node_blocks = nb, dist = dists.Categorical(num_cats = NUM_CATS))
              for v in range(4)]
        p0, p1 = multiply(ni[0], ni[1]), multiply(ni[2], ni[3])

    ex = {} if edge_ids is None else {"edge_ids": edge_ids}
    s0 = summate(p0, num_node_blocks = nb, block_size = bs, **ex,
                 external_params = LowRankSumParams(rank = RANK))
    s1 = summate(p1, num_node_blocks = nb, block_size = bs, **ex,
                 external_params = LowRankSumParams(rank = RANK))
    root = summate(multiply(s0, s1), num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root.init_parameters(perturbation = 2.0)
    return root, s0, s1


@cuda_only
@needs_lowrank_cuda
@pytest.mark.parametrize("label,edge_ids", [
    ("dense", None),
    ("ragged", _eids([[0, 1, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]])),
])
def test_two_lowrank_ns_in_one_sum_layer(label, edge_ids):
    """
    SEVERAL gated `ns` in one layer, which is the case where the per-`ns` slabs have to compose.

    `ext_slots` is built per `ns` and concatenated, so a sentinel belonging to the first `ns` sits in
    the middle of the layer's table rather than at the end -- the padding is no longer a suffix.
    """
    def once(force_reference):
        original = _lowrank_mod.LowRankSumParams._kernel_applicable
        if force_reference:
            _lowrank_mod.LowRankSumParams._kernel_applicable = lambda *a, **k: False
        try:
            root, s0, s1 = _two_lowrank_ns(edge_ids = edge_ids)
            pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))

            torch.manual_seed(7)
            data = torch.randint(0, NUM_CATS, [BATCH, 4], device = "cuda:0")

            ext = {}
            for i, n in enumerate((s0, s1)):
                ext[n] = _factors(n, BATCH, seed = 3 + 8 * i)

            lls = pc(data, sum_external_params = ext).detach().clone()
            widest = max(len(getattr(l, "external_node_infos", []) or [])
                         for gg in pc.inner_layer_groups if gg.is_sum() for l in gg.layers)
            return lls, widest
        finally:
            _lowrank_mod.LowRankSumParams._kernel_applicable = original

    kernel, widest = once(False)
    reference, _ = once(True)

    assert widest == 2, f"the two `ns` did not share a layer (widest = {widest}), so this asserts nothing"
    assert torch.all((kernel - reference).abs() < 1e-4), \
        f"{label}: differ by {float((kernel - reference).abs().max()):.3e}"


@cuda_only
@needs_lowrank_cuda
@pytest.mark.parametrize("name", ["dense", "pad_tiles", "sparse_ragged"])
def test_the_backward_matches_finite_differences(name):
    """
    `dLL/dU` and `dLL/dV` on a ragged shape, checked against a DIRECTIONAL derivative.

    There is no torch oracle for this direction -- `pre_backward` / `post_backward` raise
    `NotImplementedError` -- so finite differences are the reference. Two things make this delicate
    and both are handled by taking a whole random direction rather than one entry:

    * a single entry moves the log-likelihood far below the fp32 noise floor, and comparing 0 to 0
      passes trivially;
    * the difference is formed from log-likelihood sums of order 100, so the relative error GROWS as
      the step shrinks. `eps` is therefore deliberately coarse; `dense` doubles as the control, since
      its path is unaffected by padding and shows the same residual.
    """
    root, ns = _build(name)
    pc = juice.compile(root, verbose = False).to(torch.device("cuda:0"))

    torch.manual_seed(7)
    data = torch.randint(0, NUM_CATS, [BATCH, 2], device = "cuda:0")
    U, V = _factors(ns)

    grad_U, grad_V = torch.zeros_like(U), torch.zeros_like(V)
    pc(data, sum_external_params = {ns: (U, V)})
    pc.backward(data, flows_memory = 0.0, logspace_flows = True,
                sum_external_params_grad = {ns: (grad_U, grad_V)})

    eps = 3e-2
    torch.manual_seed(11)
    for label, tensor, grad in (("U", U, grad_U), ("V", V, grad_V)):
        direction = torch.randn_like(tensor)

        def ll(scaled):
            pair = (scaled, V) if label == "U" else (U, scaled)
            return float(pc(data, sum_external_params = {ns: pair}).detach().sum())

        numeric = (ll(tensor + eps * direction) - ll(tensor - eps * direction)) / (2 * eps)
        analytic = float((grad * direction).sum())

        scale = max(abs(numeric), abs(analytic), 1e-9)
        assert abs(numeric - analytic) / scale < 5e-2, \
            f"{name} d/d{label}: finite diff {numeric:+.5f} vs analytic {analytic:+.5f}"
