import pyjuice as juice
import torch

import pyjuice.nodes.distributions as dists
from pyjuice.nodes import multiply, summate, inputs, LowRankSumParams, ExternalSumParams
from pyjuice.layer import SumLayer, ExternalParamsSumLayer, ExternalNodeInfo

import pytest


def _sum_layers(pc):
    return [layer for layer_group in pc.inner_layer_groups if layer_group.is_sum()
            for layer in layer_group.layers]


def _external_layers(pc):
    return [layer for layer in _sum_layers(pc) if isinstance(layer, ExternalParamsSumLayer)]


def test_external_sum_layer_compilation():
    """
    Sum nodes are grouped into layers by (block size, external signature): a layer compiles one set
    of kernels, so nodes whose effective parameters are formed differently must not share one.
    """

    device = torch.device("cuda:0")

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ni1 = inputs(1, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))

        ms = multiply(ni0, ni1)

        ns_ext = summate(ms, num_node_blocks = 2, external_params = LowRankSumParams(rank = 3))
        ns_ext2 = summate(ms, num_node_blocks = 2, external_params = LowRankSumParams(rank = 5))
        ns_plain = summate(ms, num_node_blocks = 2)

        n = summate(multiply(ns_ext), multiply(ns_ext2), multiply(ns_plain),
                    num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    # The three same-depth, same-block-size nodes land in three layers: one per external signature,
    # plus the plain one. They coexist in a single layer group.
    layer_types = [type(layer).__name__ for layer_group in pc.inner_layer_groups
                   if layer_group.is_sum() and layer_group.num_layers == 3 for layer in layer_group.layers]

    assert sorted(layer_types) == ["ExternalParamsSumLayer", "ExternalParamsSumLayer", "SumLayer"]

    ext_layers = _external_layers(pc)

    assert len(ext_layers) == 2
    assert sorted([layer.external_signature for layer in ext_layers]) == ["LowRank_r3", "LowRank_r5"]

    for layer in ext_layers:
        assert len(layer.nodes) == 1
        assert isinstance(layer.external_params, LowRankSumParams)

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
                             external_params = LowRankSumParams(rank = 3))
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


def test_register_external_buffers():
    """
    A parameterization compiles its own indices through the layer, not onto itself: one descriptor
    instance is shared by every node built with it -- tied duplicates in other layers included -- so
    per-layer state on the descriptor would be overwritten by whichever layer compiles last.
    """

    device = torch.device("cuda:0")

    class _RecordingParams(ExternalSumParams):
        """A parameterization that only compiles a per-`ns` index and a layer-wide one."""

        def __init__(self):
            self.compiled_layers = []

        def get_signature(self):
            return "Recording"

        def tensor_shapes(self, ns, batch_size):
            return ((batch_size, ns.edge_ids.size(1)),)

        def compile(self, layer):
            self.compiled_layers.append(layer)

            layer.register_external_buffers(
                "eblk_par", [ns_info.ns.edge_ids[0,:].clone() for ns_info in layer.external_node_infos]
            )
            layer.register_external_buffer("num_ns", torch.tensor([len(layer.nodes)], dtype = torch.long))

    ext_params = _RecordingParams()

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 2, dist = dists.Categorical(num_cats = 5))

        ns0 = summate(multiply(ni0), num_node_blocks = 2, external_params = ext_params)
        # A tied duplicate shares the descriptor, and lands in its OWN layer (different depth)
        ns1 = ns0.duplicate(multiply(ns0), tie_params = True)

        n = summate(multiply(ns1), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    ext_layers = _external_layers(pc)

    # One descriptor instance, two layers -- and each layer holds its own compiled tensors
    assert len(ext_layers) == 2
    assert all([layer.external_params is ext_params for layer in ext_layers])
    assert len(ext_params.compiled_layers) == 2

    for layer in ext_layers:
        ns_info = layer.external_node_infos[0]

        assert "eblk_par" in ns_info.buffer_names
        assert torch.all(ns_info.eblk_par.cpu() == ns_info.ns.edge_ids[0,:])
        # Registered buffers follow the layer across devices
        assert ns_info.eblk_par.device == device and layer.ext_num_ns.device == device

    layer = ext_layers[0]

    with pytest.raises(AssertionError):
        layer.register_external_buffers("eblk_par", [torch.zeros([1], dtype = torch.long)])   # duplicate name

    with pytest.raises(AssertionError):
        layer.register_external_buffers("too_many", [torch.zeros([1], dtype = torch.long)] * 3)


def test_external_sum_layer_without_external_tensors():
    """
    Given no external tensors the layer is a plain sum layer: the hooks are skipped, so a PC with an
    external node computes exactly what the same PC built with `summate` would.
    """

    device = torch.device("cuda:0")

    def build(external):
        torch.manual_seed(430)
        ext = dict(external_params = LowRankSumParams(rank = 3)) if external else dict()
        with juice.set_block_size(4):
            ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
            ns = summate(multiply(ni0), num_node_blocks = 2, **ext)
            n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)
        torch.manual_seed(430)
        n.init_parameters(perturbation = 2.0)
        return juice.compile(n).to(device)

    pc_plain, pc_ext = build(False), build(True)

    assert isinstance(_external_layers(pc_ext)[0], ExternalParamsSumLayer)
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

    assert torch.equal(pc_plain.params, pc_ext.params)


def test_external_sum_layer_tensor_validation():
    """
    The external tensors are checked at call time, against the layout the descriptor declares.
    """

    device = torch.device("cuda:0")

    rank = 3

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ns = summate(multiply(ni0), num_node_blocks = 2, external_params = LowRankSumParams(rank = rank))
        n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    batch_size = 16
    num_edge_blocks = ns.edge_ids.size(1)

    data = torch.randint(0, 5, [batch_size, 1]).to(device)
    U = torch.full([batch_size, num_edge_blocks, 4, rank], -6.0, device = device)
    V = torch.full([batch_size, num_edge_blocks, 4, rank], -6.0, device = device)

    bad_inputs = [
        (U[:,:,:,:rank-1].contiguous(), V),          # wrong rank
        (U[:batch_size//2].contiguous(), V),         # wrong batch size
        (U.double(), V),                             # wrong dtype
        (U.transpose(1, 2), V),                      # non-contiguous
        (U.cpu(), V.cpu()),                          # wrong device
        (U,),                                        # wrong number of tensors
    ]
    for tensors in bad_inputs:
        with pytest.raises(AssertionError):
            pc(data, sum_external_params = {ns: tensors})

    # A key naming a node without an external parameterization would otherwise be a silent no-op
    with pytest.raises(AssertionError):
        pc(data, sum_external_params = {n: (U, V)})

    with pytest.raises(AssertionError):
        pc(data, sum_external_params = (U, V))

    # The layout the descriptor declares is the one that is accepted
    assert torch.all(torch.isfinite(pc(data, sum_external_params = {ns: (U, V)})))


def test_external_sum_layer_unsupported_settings():
    """
    Settings an external parameterization cannot be combined with must fail loudly, and only when
    external tensors are actually supplied.
    """

    device = torch.device("cuda:0")

    rank = 3

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ns = summate(multiply(ni0), num_node_blocks = 2, external_params = LowRankSumParams(rank = rank))
        n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    batch_size = 16
    num_edge_blocks = ns.edge_ids.size(1)

    data = torch.randint(0, 5, [batch_size, 1]).to(device)
    U = torch.full([batch_size, num_edge_blocks, 4, rank], -6.0, device = device)
    V = torch.full([batch_size, num_edge_blocks, 4, rank], -6.0, device = device)

    # Without external tensors every setting is fair game -- the layer is a plain sum layer
    pc(data, propagation_alg = "MPE")
    pc(data)
    pc.backward(data, allow_modify_flows = True, logspace_flows = False)

    with pytest.raises(AssertionError):
        pc(data, sum_external_params = {ns: (U, V)}, propagation_alg = "MPE")

    pc(data)

    # `allow_modify_flows` consumes `node_flows` in place, so the flows the external gradients are
    # built from would be gone by the time they are needed
    with pytest.raises(AssertionError):
        pc.backward(data, sum_external_params = {ns: (U, V)},
                    allow_modify_flows = True, logspace_flows = False)

    # `negate_pflows` belongs to the unnormalized partition pass of the gradient-based optimizers
    with pytest.raises(AssertionError):
        pc.backward(data, sum_external_params = {ns: (U, V)}, negate_pflows = True)


def test_external_sum_layer_grad_buffers():
    """
    Gradient buffers are zeroed once per `pc.backward`, before any layer runs, so that several nodes
    can share one buffer and have their gradients accumulated into it.
    """

    device = torch.device("cuda:0")

    rank = 3

    with juice.set_block_size(4):

        ni0 = inputs(0, num_node_blocks = 3, dist = dists.Categorical(num_cats = 5))
        ns = summate(multiply(ni0), num_node_blocks = 2, external_params = LowRankSumParams(rank = rank))
        n = summate(multiply(ns), num_node_blocks = 1, block_size = 1)

    n.init_parameters(perturbation = 2.0)

    pc = juice.compile(n)
    pc.to(device)

    batch_size = 16
    num_edge_blocks = ns.edge_ids.size(1)

    data = torch.randint(0, 5, [batch_size, 1]).to(device)
    U = torch.full([batch_size, num_edge_blocks, 4, rank], -6.0, device = device)
    V = torch.full([batch_size, num_edge_blocks, 4, rank], -6.0, device = device)

    pc(data, sum_external_params = {ns: (U, V)})

    # The backward kernels are not implemented yet; the gradient buffers are still set up first
    with pytest.raises(NotImplementedError):
        pc.backward(data)

    # The gradients are PC-owned views laid out exactly like the supplied tensors, allocated and
    # zeroed once per backward so that the layers can accumulate into them. They are views rather
    # than contiguous tensors, since the buffer holds them in the kernels' axis order.
    dU, dV = pc.get_external_params_grad(ns)

    assert dU.size() == U.size() and dV.size() == V.size()
    assert torch.all(dU == 0.0) and torch.all(dV == 0.0)

    # The caller MAY supply destinations of their own -- `{ns: buffers}` or `{group: buffers}`, in
    # exactly the shape the forward takes the parameters -- and the gradients are copied into them
    # once the backward is done. Accepted and validated here; this parameterization still has no
    # backward kernels, so the same `NotImplementedError` is what comes back.
    with pytest.raises(NotImplementedError):
        pc.backward(data, sum_external_params_grad = {ns: (dU.clone(), dV.clone())})

    # ...but the shape is checked before any of that, so a mis-shaped destination is caught outright
    # rather than being silently filled with the wrong thing.
    with pytest.raises(AssertionError):
        pc.backward(data, sum_external_params_grad = {ns: (dU.clone()[:, :-1], dV.clone())})


if __name__ == "__main__":
    test_external_sum_layer_compilation()
    test_external_node_info()
    test_register_external_buffers()
    test_external_sum_layer_without_external_tensors()
    test_external_sum_layer_tensor_validation()
    test_external_sum_layer_unsupported_settings()
    test_external_sum_layer_grad_buffers()
