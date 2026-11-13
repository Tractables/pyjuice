import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs
from pyjuice.parameterizations import LowRank, SumParameterization
from pyjuice.layer import SumLayer, ParameterizedSumLayer, ExternalNodeInfo

import pytest


def _sum_layers(pc):
    return [layer for layer_group in pc.inner_layer_groups if layer_group.is_sum()
            for layer in layer_group.layers]


def _external_layers(pc):
    return [layer for layer in _sum_layers(pc) if isinstance(layer, ParameterizedSumLayer)]


def test_parameterized_sum_layer_compilation():
    """
    Sum nodes are grouped into layers by (block size, parameterization signature): a layer compiles one set
    of kernels, so nodes whose effective parameters are formed differently must not share one.
    """

    device = torch.device("cuda:0")

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ni1 = inputs(1, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))

        ms = multiply(ni0, ni1)

        ns_ext = summate(ms, num_node_blocks = 2, parameterization = LowRank(rank = 3))
        ns_ext2 = summate(ms, num_node_blocks = 2, parameterization = LowRank(rank = 5))
        ns_plain = summate(ms, num_node_blocks = 2)

        n = summate(multiply(ns_ext), multiply(ns_ext2), multiply(ns_plain),
                    num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    # The three same-depth, same-block-size nodes land in three layers: one per parameterization signature,
    # plus the plain one. They coexist in a single layer group.
    layer_types = [type(layer).__name__ for layer_group in pc.inner_layer_groups
                   if layer_group.is_sum() and layer_group.num_layers == 3 for layer in layer_group.layers]

    assert sorted(layer_types) == ["ParameterizedSumLayer", "ParameterizedSumLayer", "SumLayer"]

    ext_layers = _external_layers(pc)

    assert len(ext_layers) == 2
    assert sorted([layer.parameterization_signature for layer in ext_layers]) == \
        ["LowRank_exp_sum_r3", "LowRank_exp_sum_r5"]

    for layer in ext_layers:
        assert len(layer.nodes) == 1
        assert isinstance(layer.parameterization, LowRank)

    # The PC records which nodes take external parameters, so a bad kwarg key can be rejected
    assert set(pc.external_params_nodes) == {ns_ext, ns_ext2}


def test_external_node_info():
    """
    `ExternalNodeInfo` translates the compiled layout into the caller's: external tensor axis `e`
    refers to `ns.edge_ids[:, e]`, and the buffers map that column to global node / element ids.
    """

    device = torch.device("cuda:0")

    for block_size in [1, 4]:
        for edge_ids in [None, torch.tensor([[0, 1, 1, 0], [2, 0, 1, 1]], dtype = torch.long)]:

            with juice.set_block_size(block_size):

                ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
                ms = multiply(ni0)

                ns = summate(ms, num_node_blocks = 2, edge_ids = edge_ids,
                             parameterization = LowRank(rank = 3))
                n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

            n.init_parameters(perturbation = 2.0)

            pc = juice.compile(n)
            pc.to(device)

            layer = _external_layers(pc)[0]

            assert len(layer.external_node_infos) == 1

            ns_info = layer.external_node_infos[0]

            assert isinstance(ns_info, ExternalNodeInfo)
            assert ns_info.ns is ns
            assert (ns_info.nid_start, ns_info.nid_end) == ns._output_ind_range
            assert ns_info.num_node_blocks == ns.num_node_blocks
            assert ns_info.num_edge_blocks == ns.edge_ids.size(1)
            assert ns_info.block_size == block_size and ns_info.ch_block_size == block_size

            # Every generic buffer followed the layer onto the GPU
            assert set(ns_info.buffer_names) == {"par_nids", "ch_eids", "eblk_ids", "par_ptr"}
            for name in ns_info.buffer_names:
                assert getattr(ns_info, name).device == device

            # `par_nids[e]` is the first node of edge block `e`'s parent block
            expected_par_nids = ns._output_ind_range[0] + ns.edge_ids[0,:] * block_size
            assert torch.all(ns_info.par_nids.cpu() == expected_par_nids)

            # `ch_eids[e]` is the first element of edge block `e`'s child block
            cs = ns.chs[0]
            expected_ch_eids = cs._output_ind_range[0] + ns.edge_ids[1,:] * block_size
            assert torch.all(ns_info.ch_eids.cpu() == expected_ch_eids)

            # `eblk_ids` / `par_ptr` group the edge blocks by parent block
            par_ptr = ns_info.par_ptr.cpu()
            eblk_ids = ns_info.eblk_ids.cpu()

            assert par_ptr.size(0) == ns.num_node_blocks + 1
            assert par_ptr[0] == 0 and par_ptr[-1] == ns.edge_ids.size(1)
            assert sorted(eblk_ids.tolist()) == list(range(ns.edge_ids.size(1)))

            max_n_eblks = 0
            for nblock_id in range(ns.num_node_blocks):
                curr_eblks = eblk_ids[par_ptr[nblock_id]:par_ptr[nblock_id + 1]]
                assert torch.all(ns.edge_ids[0, curr_eblks] == nblock_id)
                max_n_eblks = max(max_n_eblks, curr_eblks.size(0))

            assert ns_info.max_n_eblks == max_n_eblks


def test_register_parameterization_buffers():
    """
    A parameterization compiles its own indices through the layer, not onto itself: one descriptor
    instance is shared by every node built with it -- tied duplicates in other layers included -- so
    per-layer state on the descriptor would be overwritten by whichever layer compiles last.
    """

    device = torch.device("cuda:0")

    class _RecordingParams(SumParameterization):
        """A parameterization that only compiles a per-`ns` index and a layer-wide one."""

        def __init__(self):
            self.compiled_layers = []

        def get_signature(self):
            return "Recording"

        def tensor_shapes(self, ns, batch_size):
            return ((batch_size, ns.edge_ids.size(1)),)

        def compile(self, layer):
            self.compiled_layers.append(layer)

            layer.register_parameterization_buffers(
                "eblk_par", [ns_info.ns.edge_ids[0,:].clone() for ns_info in layer.external_node_infos]
            )
            layer.register_parameterization_buffer("num_ns", torch.tensor([len(layer.nodes)], dtype = torch.long))

    ext_params = _RecordingParams()

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))

        ns0 = summate(multiply(ni0), num_node_blocks = 2, parameterization = ext_params)
        # A tied duplicate shares the descriptor, and lands in its OWN layer (different depth)
        ns1 = ns0.duplicate(multiply(ns0), tie_params = True)

        n = summate(multiply(ns1), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    ext_layers = _external_layers(pc)

    # One descriptor instance, two layers -- and each layer holds its own compiled tensors
    assert len(ext_layers) == 2
    assert all([layer.parameterization is ext_params for layer in ext_layers])
    assert len(ext_params.compiled_layers) == 2

    for layer in ext_layers:
        ns_info = layer.external_node_infos[0]

        assert "eblk_par" in ns_info.buffer_names
        assert torch.all(ns_info.eblk_par.cpu() == ns_info.ns.edge_ids[0,:])
        # Registered buffers follow the layer across devices
        assert ns_info.eblk_par.device == device and layer.ext_num_ns.device == device

    layer = ext_layers[0]

    with pytest.raises(AssertionError):
        layer.register_parameterization_buffers("eblk_par", [torch.zeros([1], dtype = torch.long)])   # duplicate name

    with pytest.raises(AssertionError):
        layer.register_parameterization_buffers("too_many", [torch.zeros([1], dtype = torch.long)] * 3)


@pytest.mark.parametrize("corrected", [False, True])
def test_parameterized_sum_layer_without_external_tensors(corrected):
    """
    Given no external tensors the layer is a plain sum layer: the hooks are skipped, so a PC with an
    external node computes exactly what the same PC built with `summate` would -- bit for bit where the
    parameterization keeps no denominator flow. Where it does, the M-step is the conditional one, whose denominator
    of such a pass is `theta` times the node's flow: the standard step to rounding.
    """

    device = torch.device("cuda:0")

    def build(external):
        torch.manual_seed(430)
        ext = dict(parameterization = LowRank(rank = 3, apply_z_correction = corrected)) if external else dict()
        with juice.set_block_size(4):
            ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
            ns = summate(multiply(ni0), num_node_blocks = 2, **ext)
            n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
        torch.manual_seed(430)
        n.init_parameters(perturbation = 2.0)
        return juice.compile(n).to(device)

    pc_plain, pc_ext = build(False), build(True)

    assert isinstance(_external_layers(pc_ext)[0], ParameterizedSumLayer)
    assert len(_external_layers(pc_plain)) == 0

    # Identical parameter layout
    assert pc_plain.num_sum_params == pc_ext.num_sum_params
    assert pc_plain.num_param_flows == pc_ext.num_param_flows
    assert torch.equal(pc_plain.params, pc_ext.params)

    data = torch.randint(0, 5, [16, 1]).to(device)

    lls_plain, lls_ext = pc_plain(data), pc_ext(data)

    assert torch.equal(lls_plain, lls_ext)

    pc_plain.backward(data)
    pc_ext.backward(data)

    assert torch.equal(pc_plain.param_flows, pc_ext.param_flows)
    assert pc_plain._cum_flow == pc_ext._cum_flow

    pc_plain.mini_batch_em(step_size = 0.5, pseudocount = 0.01)
    pc_ext.mini_batch_em(step_size = 0.5, pseudocount = 0.01)

    if corrected:
        assert torch.allclose(pc_plain.params, pc_ext.params, rtol = 1e-5, atol = 1e-7)
    else:
        assert torch.equal(pc_plain.params, pc_ext.params)


def test_parameterized_sum_layer_tensor_validation():
    """
    The external tensors are checked at call time, against the layout the descriptor declares.
    """

    device = torch.device("cuda:0")

    rank = 3

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ns = summate(multiply(ni0), num_node_blocks = 2, parameterization = LowRank(rank = rank))
        n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    batch_size = 16
    num_edge_blocks = ns.edge_ids.size(1)

    data = torch.randint(0, 5, [batch_size, 1]).to(device)
    A = 0.3 * torch.randn([batch_size, ns.num_nodes, rank], device = device)
    B = 0.3 * torch.randn([batch_size, rank, ns.num_ch_nodes], device = device)

    bad_inputs = [
        (A[:,:,:rank-1].contiguous(), B),            # wrong rank
        (A[:batch_size//2].contiguous(), B),         # wrong batch size
        (A.double(), B),                             # wrong dtype
        (B.transpose(1, 2), A.transpose(1, 2)),      # the factors swapped (and non-contiguous)
        (A, B.transpose(1, 2)),                      # the second factor child-major
        (A.cpu(), B.cpu()),                          # wrong device
        (A,),                                        # wrong number of tensors
    ]
    for tensors in bad_inputs:
        with pytest.raises(AssertionError):
            pc(data, sum_external_params = {ns: tensors})

    # A key naming a node without an external parameterization would otherwise be a silent no-op
    with pytest.raises(AssertionError):
        pc(data, sum_external_params = {n: (A, B)})

    with pytest.raises(AssertionError):
        pc(data, sum_external_params = (A, B))

    # The layout the descriptor declares is the one that is accepted
    assert torch.all(torch.isfinite(pc(data, sum_external_params = {ns: (A, B)})))


def test_parameterized_sum_layer_unsupported_settings():
    """
    Settings an external parameterization cannot be combined with must fail loudly, and only when
    external tensors are actually supplied.
    """

    device = torch.device("cuda:0")

    rank = 3

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ns = summate(multiply(ni0), num_node_blocks = 2, parameterization = LowRank(rank = rank))
        n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    batch_size = 16
    num_edge_blocks = ns.edge_ids.size(1)

    data = torch.randint(0, 5, [batch_size, 1]).to(device)
    A = 0.3 * torch.randn([batch_size, ns.num_nodes, rank], device = device)
    B = 0.3 * torch.randn([batch_size, rank, ns.num_ch_nodes], device = device)

    # Without external tensors every setting is fair game -- the layer is a plain sum layer
    pc(data, propagation_alg = "MPE")
    pc(data)
    pc.backward(data, allow_modify_flows = True, logspace_flows = False)

    # (the tensors themselves are accepted: what is refused below is the setting)
    assert torch.all(torch.isfinite(pc(data, sum_external_params = {ns: (A, B)})))

    with pytest.raises(AssertionError, match = "propagation"):
        pc(data, sum_external_params = {ns: (A, B)}, propagation_alg = "MPE")

    # A backward follows the forward it reads: one that was given no external parameters refuses them
    pc(data)
    with pytest.raises(AssertionError, match = "forward pass did not receive"):
        pc.backward(data, sum_external_params = {ns: (A, B)})

    # `allow_modify_flows` consumes `node_flows` in place, so the flows the external gradients are
    # built from would be gone by the time they are needed
    pc(data, sum_external_params = {ns: (A, B)})
    with pytest.raises(AssertionError, match = "allow_modify_flows"):
        pc.backward(data, allow_modify_flows = True, logspace_flows = False)

    # `negate_pflows` belongs to the unnormalized partition pass of the gradient-based optimizers
    pc(data, sum_external_params = {ns: (A, B)})
    with pytest.raises(AssertionError, match = "negate_pflows"):
        pc.backward(data, allow_modify_flows = False, negate_pflows = True)


def test_parameterized_sum_layer_grad_buffers():
    """
    Gradient buffers are zeroed once per `pc.backward`, before any layer runs, so that several nodes
    can share one buffer and have their gradients accumulated into it.
    """

    device = torch.device("cuda:0")

    rank = 3

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ns = summate(multiply(ni0), num_node_blocks = 2, parameterization = LowRank(rank = rank))
        n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    batch_size = 16
    num_edge_blocks = ns.edge_ids.size(1)

    data = torch.randint(0, 5, [batch_size, 1]).to(device)
    A = 0.3 * torch.randn([batch_size, ns.num_nodes, rank], device = device)
    B = 0.3 * torch.randn([batch_size, rank, ns.num_ch_nodes], device = device)

    pc(data, sum_external_params = {ns: (A, B)})
    pc.backward(data, allow_modify_flows = False)

    # The gradients are PC-owned views laid out exactly like the supplied tensors, allocated and
    # zeroed once per backward so that the layers can accumulate into them.
    dA, dB = pc.get_external_params_grad(ns)

    assert dA.size() == A.size() and dB.size() == B.size()
    assert float(dA.abs().max()) > 0.0 and float(dB.abs().max()) > 0.0
    first = (dA.clone(), dB.clone())

    # Zeroed once per backward: a second one leaves the same gradients, not twice them
    pc.backward(data, allow_modify_flows = False)
    dA, dB = pc.get_external_params_grad(ns)
    assert torch.allclose(dA, first[0], rtol = 1e-5, atol = 1e-7) and torch.allclose(dB, first[1], rtol = 1e-5, atol = 1e-7)

    # The caller MAY supply destinations of their own -- `{ns: buffers}` or `{group: buffers}`, in
    # exactly the shape the forward takes the parameters -- and the gradients are copied into them
    # once the backward is done, over whatever they held.
    dst = (torch.full_like(A, 7.0), torch.full_like(B, 7.0))
    pc.backward(data, allow_modify_flows = False, sum_external_params_grad = {ns: dst})
    assert torch.allclose(dst[0], first[0], rtol = 1e-5, atol = 1e-7) and torch.allclose(dst[1], first[1], rtol = 1e-5, atol = 1e-7)

    # ...but the shape is checked before any of that, so a mis-shaped destination is caught outright
    # rather than being silently filled with the wrong thing.
    with pytest.raises(AssertionError):
        pc.backward(data, allow_modify_flows = False,
                    sum_external_params_grad = {ns: (dst[0].clone()[:, :-1], dst[1].clone())})


if __name__ == "__main__":
    test_parameterized_sum_layer_compilation()
    test_external_node_info()
    test_register_parameterization_buffers()
    for corrected in [False, True]:
        test_parameterized_sum_layer_without_external_tensors(corrected)
    test_parameterized_sum_layer_tensor_validation()
    test_parameterized_sum_layer_unsupported_settings()
    test_parameterized_sum_layer_grad_buffers()
