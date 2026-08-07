import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, ExternalSumParams, LowRankSumParams
from pyjuice.layer import StagedExternalParams

import pytest


RANK = 3


class EchoParams(ExternalSumParams):
    """
    A minimal parameterization used to exercise the machinery end to end.

    It consumes two tensors shaped like `LowRankSumParams`', perturbs `node_mars` by a value derived
    from the first of them (so the forward provably *read* what was staged), and writes a known
    function of both into the gradient buffers (so the backward provably *wrote* where the caller
    reads). That makes every hop of the transfer -- caller -> staging buffer -> hook -> gradient
    buffer -> caller -> observable.
    """

    def __init__(self, rank: int = RANK):
        self.rank = rank
        self.calls = {"compile": 0, "forward": 0, "pre_backward": 0, "post_backward": 0}
        self.seen = dict()          # ns -> the tensors the hooks were handed

    def get_signature(self):
        return f"Echo_r{self.rank}"

    def tensor_shapes(self, ns, batch_size):
        num_edge_blocks = ns.edge_ids.size(1)
        return (
            (batch_size, num_edge_blocks, ns.ch_block_size, self.rank),
            (batch_size, num_edge_blocks, ns.block_size, self.rank),
        )

    def compile(self, layer):
        self.calls["compile"] += 1
        layer.register_external_buffers(
            "eblk_par", [ns_info.ns.edge_ids[0,:].clone() for ns_info in layer.external_node_infos]
        )

    def forward(self, layer, ns_info, tensors, node_mars, element_mars, params, **kwargs):
        self.calls["forward"] += 1
        self.seen[ns_info.ns] = tensors

        # Reads the staged values and leaves a deterministic, checkable mark on this ns's nodes
        node_mars[ns_info.nid_start:ns_info.nid_end,:] += tensors[0].sum(dim = (1, 2, 3))[None,:]

    def pre_backward(self, layer, ns_info, tensors, node_flows, element_flows, node_mars,
                     element_mars, params, **kwargs):
        self.calls["pre_backward"] += 1

        # Undo the forward's mark, so the standard backward sees the buffer it expects
        node_mars[ns_info.nid_start:ns_info.nid_end,:] -= tensors[0].sum(dim = (1, 2, 3))[None,:]

    def post_backward(self, layer, ns_info, tensors, grad_tensors, node_flows, element_flows,
                      node_mars, element_mars, params, param_flows = None, **kwargs):
        self.calls["post_backward"] += 1

        if grad_tensors is None:
            return None

        # ACCUMULATE, as a real parameterization must
        grad_tensors[0].add_(tensors[0] * 2.0)
        grad_tensors[1].add_(tensors[1] * 3.0)


def _build_hmm(external_params, seq_length = 4, num_latents = 8, num_emits = 5, seed = 0):
    """
    A homogeneous-HMM-shaped PC: one transition per timestep, all tied, each its own layer. Pass
    `external_params = None` for the plain equivalent.
    """

    torch.manual_seed(seed)

    ext = dict(external_params = external_params) if external_params is not None else dict()

    with juice.set_block_size(num_latents):

        ns_input = inputs(seq_length - 1, num_node_blocks = 1, dist = dists.Categorical(num_cats = num_emits))

        ns_sum, curr_zs = None, ns_input
        for var in range(seq_length - 2, -1, -1):
            curr_xs = ns_input.duplicate(var, tie_params = True)

            if ns_sum is None:
                ns_sum = summate(curr_zs, num_node_blocks = 1, **ext)
                ns = ns_sum
            else:
                ns = ns_sum.duplicate(curr_zs, tie_params = True)

            curr_zs = multiply(curr_xs, ns)

        root_ns = summate(curr_zs, num_node_blocks = 1, block_size = 1)

    torch.manual_seed(seed)
    root_ns.init_parameters(perturbation = 2.0)

    return root_ns, ns_sum


def _compiled(external_params, **kwargs):
    root_ns, ns_sum = _build_hmm(external_params, **kwargs)
    pc = juice.compile(root_ns)
    pc.to(torch.device("cuda:0"))
    return pc, root_ns, ns_sum


def _make_tensors(pc, batch_size, seed = 0, strided = False):
    """
    Per-copy DISTINCT tensors for every external node. With `strided`, they are batch-major slices of
    one tensor -- i.e. non-contiguous, the layout a router head naturally produces.
    """

    torch.manual_seed(seed)

    nss = list(pc.external_params_nodes)
    shapes = [pc.external_params_nodes[ns].external_params.tensor_shapes(ns, batch_size) for ns in nss]

    if not strided:
        return {ns: tuple(torch.randn(shape, device = pc.device) for shape in shape_pair)
                for ns, shape_pair in zip(nss, shapes)}

    # [B, num_copies, 2, ...] -> slicing copy `i` gives a strided view
    shape = shapes[0][0]
    big = torch.randn([shape[0], len(nss), 2, *shape[1:]], device = pc.device)

    return {ns: (big[:,i,0], big[:,i,1]) for i, ns in enumerate(nss)}


def test_external_params_layout():
    """
    Every node gets its own contiguous slot per declared tensor, laid out in a deterministic order and
    sized from the descriptor.
    """

    batch_size = 8

    pc, root_ns, trans_ns = _compiled(EchoParams())

    nss = list(pc.external_params_nodes)

    assert len(nss) == 3                                    # one transition per timestep
    assert trans_ns in pc.external_params_nodes
    assert all([ns in pc.external_params_nodes for ns in nss])

    total_numel, ns2slots = pc._external_params_layout(batch_size)

    # Slots tile the buffer exactly: contiguous, in order, no gaps and no overlap
    offset = 0
    for ns in nss:
        for slot_offset, shape in ns2slots[ns]:
            assert slot_offset == offset
            offset += int(torch.tensor(shape).prod())

    assert offset == total_numel

    # ... and they match what the descriptor declares
    expected = sum([int(torch.tensor(shape).prod())
                    for ns in nss for shape in ns.external_params.tensor_shapes(ns, batch_size)])
    assert total_numel == expected

    # Cached per batch size, and the layout scales with it
    assert pc._external_params_layout(batch_size) is ns2slots or \
           pc._external_params_layout(batch_size)[0] == total_numel
    assert pc._external_params_layout(2 * batch_size)[0] == 2 * total_numel


def test_external_params_buffers_init():
    """
    The staging buffers follow `node_mars`' lifecycle: absent until needed, sized from the batch,
    re-allocated when it changes, and stable otherwise (which is what CUDA-graph capture requires).
    """

    device = torch.device("cuda:0")

    pc, root_ns, trans_ns = _compiled(EchoParams())

    # Nothing allocated before a forward pass supplies external parameters
    assert pc.external_params is None and pc.external_params_grad is None

    batch_size = 8
    data = torch.randint(0, 5, [batch_size, 4]).to(device)

    pc(data, sum_external_params = _make_tensors(pc, batch_size))

    total_numel = pc._external_params_layout(batch_size)[0]

    assert pc.external_params is not None
    assert pc.external_params.numel() == total_numel
    assert pc.external_params.dtype == torch.float32
    assert pc.external_params.device == device
    assert pc.external_params.is_contiguous()

    # Stable across steps, so a captured graph keeps pointing at the right memory
    buffer_ptr = pc.external_params.data_ptr()
    pc(data, sum_external_params = _make_tensors(pc, batch_size, seed = 1))
    assert pc.external_params.data_ptr() == buffer_ptr

    # Re-allocated (and re-sized) when the batch changes
    data2 = torch.randint(0, 5, [2 * batch_size, 4]).to(device)
    pc(data2, sum_external_params = _make_tensors(pc, 2 * batch_size))
    assert pc.external_params.numel() == 2 * total_numel

    # A PC with no external nodes never allocates them at all
    pc_plain, _, _ = _compiled(None)
    pc_plain(data)
    assert pc_plain.external_params is None and pc_plain.external_params_grad is None


@pytest.mark.parametrize("strided", [False, True])
def test_external_params_staging_transfer(strided):
    """
    What the caller hands over is what the layers receive: same values, per node, in that node's own
    slot -- including when the caller's tensors are non-contiguous slices of their own head's output.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    data = torch.randint(0, 5, [batch_size, 4]).to(device)
    tensors = _make_tensors(pc, batch_size, strided = strided)

    if strided:
        assert not any([t.is_contiguous() for t in tensors[trans_ns]])

    pc(data, sum_external_params = tensors)

    staged = pc._staged_external_params

    assert isinstance(staged, StagedExternalParams)
    assert set(staged.keys()) == set(tensors.keys())

    for ns, ns_tensors in tensors.items():
        for staged_tensor, tensor in zip(staged[ns], ns_tensors):
            # Exact transfer, and the layers get contiguous views regardless of the source layout
            assert torch.equal(staged_tensor, tensor)
            assert staged_tensor.is_contiguous()
            assert staged_tensor.size() == tensor.size()

    # Distinct nodes occupy distinct slots -- no node can clobber another's values
    slot_ptrs = [t.data_ptr() for ns_tensors in staged.values() for t in ns_tensors]
    assert len(set(slot_ptrs)) == len(slot_ptrs)

    # The hooks are handed exactly the staged views
    for ns in tensors:
        assert all([a is b for a, b in zip(ext_params.seen[ns], staged[ns])])

    # A second pass overwrites rather than accumulating
    tensors2 = _make_tensors(pc, batch_size, seed = 7, strided = strided)
    pc(data, sum_external_params = tensors2)

    for ns, ns_tensors in tensors2.items():
        for staged_tensor, tensor in zip(pc._staged_external_params[ns], ns_tensors):
            assert torch.equal(staged_tensor, tensor)


def test_external_params_shared_tensors():
    """
    Passing the SAME tensors for several nodes is how one correction is shared across tied copies.
    Each node still gets its own slot, so the values are replicated rather than aliased.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    pc, root_ns, trans_ns = _compiled(EchoParams())

    shape_U, shape_V = trans_ns.external_params.tensor_shapes(trans_ns, batch_size)

    torch.manual_seed(11)
    U = torch.randn(shape_U, device = device)
    V = torch.randn(shape_V, device = device)

    data = torch.randint(0, 5, [batch_size, 4]).to(device)

    pc(data, sum_external_params = {ns: (U, V) for ns in pc.external_params_nodes})

    staged = pc._staged_external_params

    for ns in pc.external_params_nodes:
        assert torch.equal(staged[ns][0], U) and torch.equal(staged[ns][1], V)

    assert len(set([staged[ns][0].data_ptr() for ns in pc.external_params_nodes])) == len(staged)


def test_external_params_grad_buffer():
    """
    Gradients come back as views into a buffer that mirrors the value buffer slot for slot, zeroed
    once per backward so the hooks can accumulate into it.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    data = torch.randint(0, 5, [batch_size, 4]).to(device)
    tensors = _make_tensors(pc, batch_size)

    pc(data, sum_external_params = tensors)
    pc.backward(data)

    assert pc.external_params_grad is not None
    assert pc.external_params_grad.numel() == pc.external_params.numel()

    for ns in pc.external_params_nodes:
        grads = pc.get_external_params_grad(ns)

        assert len(grads) == 2
        for grad, tensor in zip(grads, tensors[ns]):
            assert grad.size() == tensor.size() and grad.is_contiguous()

        # `EchoParams` writes a known function of the staged values
        assert torch.allclose(grads[0], tensors[ns][0] * 2.0)
        assert torch.allclose(grads[1], tensors[ns][1] * 3.0)

        # A gradient sits at the same offset as the value it belongs to
        value_offset = pc._staged_external_params[ns][0].data_ptr() - pc.external_params.data_ptr()
        grad_offset = grads[0].data_ptr() - pc.external_params_grad.data_ptr()
        assert value_offset == grad_offset

    # Zeroed per backward: running it twice must not double the accumulated gradients
    first = pc.get_external_params_grad(trans_ns)[0].clone()
    pc.backward(data)
    assert torch.equal(pc.get_external_params_grad(trans_ns)[0], first)

    # Opting out skips the buffer entirely
    pc.backward(data, compute_external_grads = False)
    with pytest.raises(AssertionError):
        pc.get_external_params_grad(trans_ns)

    # The caller MAY supply their own destination buffers, keyed exactly like the forward. They are
    # filled from the PC's internal ones, which stay readable through `get_external_params_grad`.
    pc(data, sum_external_params = tensors)
    mine = tuple(torch.zeros_like(t) for t in tensors[trans_ns])
    pc.backward(data, sum_external_params_grad = {trans_ns: mine})
    for got, ref in zip(mine, pc.get_external_params_grad(trans_ns)):
        assert torch.equal(got, ref)


def test_external_params_backward_reuses_forward():
    """
    The forward leaves `node_mars` in a form only the matching external backward interprets correctly,
    so the backward takes the external parameters from what the forward staged rather than asking for
    them again -- and rejects a mapping that disagrees with it.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    data = torch.randint(0, 5, [batch_size, 4]).to(device)
    tensors = _make_tensors(pc, batch_size)

    pc(data, sum_external_params = tensors)

    calls_before = ext_params.calls["pre_backward"]
    pc.backward(data)                                    # no external kwarg needed
    assert ext_params.calls["pre_backward"] == calls_before + len(pc.external_params_nodes)

    # Naming the same nodes is fine (this is what the autograd hook does)
    pc(data, sum_external_params = tensors)
    pc.backward(data, sum_external_params = tensors)

    # Naming a different set is not
    partial_tensors = {trans_ns: tensors[trans_ns]}
    pc(data, sum_external_params = tensors)
    with pytest.raises(AssertionError):
        pc.backward(data, sum_external_params = partial_tensors)


def test_external_params_validation():
    """
    The caller's tensors are checked once, where they are staged. Strides are NOT a requirement --
    staging copies -- but shape, dtype, device and arity are.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    pc, root_ns, trans_ns = _compiled(EchoParams())

    data = torch.randint(0, 5, [batch_size, 4]).to(device)
    tensors = _make_tensors(pc, batch_size)

    U, V = tensors[trans_ns]

    def _with(bad):
        new_tensors = dict(tensors)
        new_tensors[trans_ns] = bad
        return new_tensors

    for bad in [
        (U[:,:,:,:RANK-1].contiguous(), V),          # wrong rank
        (U[:batch_size//2].contiguous(), V),         # wrong batch size
        (U.double(), V),                             # wrong dtype
        (U.cpu(), V.cpu()),                          # wrong device
        (U,),                                        # wrong number of tensors
    ]:
        with pytest.raises(AssertionError):
            pc(data, sum_external_params = _with(bad))

    # A node the PC did not compile with a parameterization would otherwise be a silent no-op
    with pytest.raises(AssertionError):
        pc(data, sum_external_params = {root_ns: (U, V)})

    with pytest.raises(AssertionError):
        pc(data, sum_external_params = (U, V))

    # Non-contiguous is accepted, since staging copies
    pc(data, sum_external_params = _make_tensors(pc, batch_size, strided = True))


def test_external_params_partial_supply():
    """
    Supplying external parameters for only some of the nodes leaves the rest as plain sum nodes.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    data = torch.randint(0, 5, [batch_size, 4]).to(device)
    tensors = _make_tensors(pc, batch_size)

    ext_params.calls["forward"] = 0
    pc(data, sum_external_params = {trans_ns: tensors[trans_ns]})

    assert ext_params.calls["forward"] == 1
    assert set(pc._staged_external_params.keys()) == {trans_ns}

    pc.backward(data)

    with pytest.raises(AssertionError):
        pc.get_external_params_grad([ns for ns in pc.external_params_nodes if ns is not trans_ns][0])


def test_external_params_usage():
    """
    The intended end-to-end usage, and what it has to guarantee: the effective parameters really are
    driven by the caller's tensors, the shared parameters keep training by EM, and the per-sample
    gradients come back matching what was supplied.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    # One entry per tied copy; the same tensors for all of them shares one correction across timesteps
    copies = list(pc.external_params_nodes)

    shape_U, shape_V = trans_ns.external_params.tensor_shapes(trans_ns, batch_size)

    torch.manual_seed(5)
    U = torch.randn(shape_U, device = device)
    V = torch.randn(shape_V, device = device)

    mapping = {ns: (U, V) for ns in copies}

    data = torch.randint(0, 5, [batch_size, 4]).to(device)

    lls = pc(data, sum_external_params = mapping)
    pc.backward(data)

    # The external tensors actually changed the answer
    lls_plain = pc(data)
    assert not torch.allclose(lls, lls_plain)

    # ... and changing them changes it again
    lls_other = pc(data, sum_external_params = {ns: (U * 0.5, V) for ns in copies})
    assert not torch.allclose(lls, lls_other)

    # The shared parameters trained by the ordinary EM path
    pc(data, sum_external_params = mapping)
    pc.backward(data)

    params_before = pc.params.clone()
    pc.mini_batch_em(step_size = 0.5, pseudocount = 0.01)

    assert not torch.allclose(params_before, pc.params)
    assert torch.all(torch.isfinite(pc.params))

    # The per-sample gradients come back, one set per copy, matching the tensors supplied
    pc(data, sum_external_params = mapping)
    pc.backward(data)

    for ns in copies:
        dU, dV = pc.get_external_params_grad(ns)
        assert dU.size() == U.size() and dV.size() == V.size()
        assert torch.allclose(dU, U * 2.0) and torch.allclose(dV, V * 3.0)

    # The autograd entry point routes the external parameters into the backward on its own
    ext_params.calls["post_backward"] = 0
    lls = pc(data, sum_external_params = mapping)
    lls.mean().backward()

    assert ext_params.calls["post_backward"] == len(copies)
    assert torch.allclose(pc.get_external_params_grad(trans_ns)[0], U * 2.0)


def test_lowrank_params_reaches_kernels():
    """
    The real parameterization is staged the same way, and its forward runs off the staged values.
    """

    device = torch.device("cuda:0")

    batch_size = 8

    pc, root_ns, trans_ns = _compiled(LowRankSumParams(rank = RANK))

    data = torch.randint(0, 5, [batch_size, 4]).to(device)
    tensors = _make_tensors(pc, batch_size)

    lls = pc(data, sum_external_params = tensors)

    assert torch.all(torch.isfinite(lls))

    # ... from the staged values, held in the kernels' storage order
    perm = trans_ns.external_params.storage_perm()
    for ns, ns_tensors in tensors.items():
        for staged_tensor, tensor in zip(pc._staged_external_params[ns], ns_tensors):
            assert torch.equal(staged_tensor, tensor.permute(perm))


if __name__ == "__main__":
    test_external_params_layout()
    test_external_params_buffers_init()
    for strided in [False, True]:
        test_external_params_staging_transfer(strided)
    test_external_params_shared_tensors()
    test_external_params_grad_buffer()
    test_external_params_backward_reuses_forward()
    test_external_params_validation()
    test_external_params_partial_supply()
    test_external_params_usage()
    test_lowrank_params_reaches_kernels()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_every_staging_transpose_tier_agrees():
    """REGRESSION. Staging picks a tiled-transpose backend at runtime -- the CUDA extension if it
    built, Triton otherwise, a generic strided `copy_` if neither. Only the first is reachable on a
    machine where the extension compiled, so the other two were never exercised, and the generic one
    carried a shape bug.

    `_staged_copy` is handed TWO destination shapes: staging a whole GROUP passes a FLAT slice of the
    buffer (`_group_fast_stage` returns a `narrow`), while the per-node path passes a storage-shaped
    view. The fallback did `src.permute(...)` and so matched only the second, which made a group stage
    raise a shape mismatch -- correct caller code refused because of a build detail. Both shapes are
    checked here, against all three tiers, which is what the end-to-end tests cannot do: they only ever
    reach whichever backend the machine happens to have.

    All three compute the same pure permutation, so they must agree BIT FOR BIT."""
    from pyjuice.model import tensorcircuit as tc
    from pyjuice.nodes.external_params.kernels.c import get_module
    from pyjuice.nodes.external_params.kernels.staging import staging_transpose_triton

    device = torch.device("cuda:0")
    batch, n_blk, gates = 32, 7, 4               # `src` is [B, 7, 4], as a 7-step group would give
    n = n_blk * gates

    torch.manual_seed(0)
    src = torch.randn([batch, n_blk, gates], device = device)
    expect = src.permute(1, 2, 0).reshape(-1)    # storage layout: batch innermost

    tiers = {"triton": staging_transpose_triton, "torch": None}
    if get_module() is not None:
        tiers["cuda"] = get_module().staging_transpose

    orig = tc._staging_transpose_fn
    try:
        for name, fn in tiers.items():
            tc._staging_transpose_fn = (lambda fn = fn: fn)

            flat = torch.zeros([batch * n], device = device)          # a whole-group destination
            tc._staged_copy([], [], [(flat, src)])
            assert torch.equal(flat, expect), f"'{name}' tier, flat destination"

            shaped = torch.zeros([n_blk, gates, batch], device = device)   # a per-node destination
            tc._staged_copy([], [], [(shaped, src)])
            assert torch.equal(shaped.reshape(-1), expect), f"'{name}' tier, storage-shaped destination"
    finally:
        tc._staging_transpose_fn = orig


def test_an_ungated_forward_drops_a_previously_staged_gate():
    """
    What is staged always describes the MOST RECENT forward.

    `pc.backward()` and a conditional `queries.sample()` both take the staged tensors
    unconditionally -- they must, since `node_mars` was computed with them. So a forward that
    supplies none has to drop what an earlier one staged, or those two consumers silently apply a
    gate the forward they follow never used. That went wrong in a decode loop that interleaved a
    1-row gated forward with a 10-row ungated one: at a matching batch it was silent, at a differing
    one it surfaced as a shape error from inside the sampler.
    """
    device = torch.device("cuda:0")

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    data = torch.randint(0, 5, [8, 4]).to(device)
    pc(data, sum_external_params = _make_tensors(pc, 8))
    assert pc._staged_external_params is not None

    pc(data)                                             # same batch, no external parameters
    assert pc._staged_external_params is None, \
        "an ungated forward must not leave the previous forward's tensors staged"


def test_an_ungated_forward_at_another_batch_size_drops_them_too():
    """The case that was reported: the ungated call is also at a different batch size."""
    device = torch.device("cuda:0")

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    pc(torch.randint(0, 5, [8, 4]).to(device), sum_external_params = _make_tensors(pc, 8))
    pc(torch.randint(0, 5, [3, 4]).to(device))           # ungated, different batch

    assert pc._staged_external_params is None


def test_a_backward_after_an_ungated_forward_runs_ungated():
    """It runs the SHARED-parameter backward -- consistent with the forward it follows -- rather than
    applying the stale gate or raising."""
    device = torch.device("cuda:0")

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    data = torch.randint(0, 5, [8, 4]).to(device)
    pc(data, sum_external_params = _make_tensors(pc, 8))
    pc.backward(data)
    gated_calls = ext_params.calls["pre_backward"]
    assert gated_calls > 0

    pc(data)                                             # ungated
    pc.backward(data)                                    # must not reach the parameterization
    assert ext_params.calls["pre_backward"] == gated_calls


def test_the_autograd_hook_still_sees_its_own_forwards_tensors():
    """
    `forward` mutates `kwargs` in place and registers the backward hook with `**kwargs`, so a staged
    dict comes back round to `_stage_external_params`. That branch must NOT be treated as "nothing
    supplied", or `lls.backward()` would silently lose the gate.
    """
    device = torch.device("cuda:0")

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    data = torch.randint(0, 5, [8, 4]).to(device)
    before = ext_params.calls["pre_backward"]

    lls = pc(data, sum_external_params = _make_tensors(pc, 8))
    lls.mean().backward()

    assert ext_params.calls["pre_backward"] > before, \
        "the autograd hook's backward lost the gate its own forward staged"
    assert pc._staged_external_params is not None

    # and handing a staged dict straight back to a forward keeps it staged
    staged = pc._staged_external_params
    assert isinstance(staged, StagedExternalParams)
    pc(data, sum_external_params = staged)
    assert pc._staged_external_params is staged


def test_interleaved_batch_sizes_are_fine_when_every_forward_is_gated():
    """Interleaving batch sizes is not itself a problem -- each gated forward re-stages."""
    device = torch.device("cuda:0")

    ext_params = EchoParams()
    pc, root_ns, trans_ns = _compiled(ext_params)

    for batch_size in (1, 10, 1, 4, 10):
        data = torch.randint(0, 5, [batch_size, 4]).to(device)
        pc(data, sum_external_params = _make_tensors(pc, batch_size))

        staged = next(iter(pc._staged_external_params.values()))[0]
        assert staged.size(0) == batch_size, \
            f"staged tensors describe batch {staged.size(0)}, but the forward ran at {batch_size}"
