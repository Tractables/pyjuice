"""
The lifted buffers: regions follow the PC's layers, every sum node group keeps exactly its own block per
sample, regions are aligned and disjoint, an input layer writes the input region exactly as it writes a
TensorCircuit's node_mars, and the storage is reused across batch sizes.
"""
import pytest
import torch

import pyjuice.constraints as jc
from pyjuice.constraints.backends.lifted.plan import ALIGN, buffer_layout

V = 3
KINDS = {"hmm": 6, "hmm_untied": 6, "hmm_block_sparse": 6, "pd": 8, "pd_prod_dominated": 8, "pd_blockified": 8,
         "hand": 5, "hand_permuted": 5, "hand_unit": 4}                     # see conftest.py


def compiled(build_pc, kind, constraint = None):
    n = KINDS[kind]
    return jc.compile(constraint or jc.DFA.contains([[1, 2]], V), build_pc(kind, n, V)), n


def slots_of(cc, n, ns):
    """Entry columns x exit columns of a node group's block, a sequence end counting as one."""
    (a, b), = cc.structure.node(ns).scope_runs
    w = cc.width_per_boundary.tolist()
    return (1 if a == 0 else w[a]) * (1 if b == n - 1 else w[b + 1])


@pytest.mark.parametrize("kind", list(KINDS))
def test_regions_follow_the_pcs_layers(kind, build_pc):
    cc, n = compiled(build_pc, kind)
    pc = cc.pc
    want_sums = sorted((ns._output_ind_range[0], ns._output_ind_range[1], slots_of(cc, n, ns))
                       for lg in pc.inner_layer_groups if lg.is_sum() for layer in lg.layers for ns in layer.nodes)
    assert list(cc.sum_regions) == want_sums
    covered = [r for first, end, _ in cc.sum_regions for r in range(first, end)]                 # all sum rows, once
    want_rows = [r for lg in pc.inner_layer_groups if lg.is_sum() for layer in lg.layers
                 for r in range(*layer._layer_nid_range)]
    assert sorted(covered) == sorted(want_rows) and len(set(covered)) == len(covered)
    prod_groups = [lg for lg in pc.inner_layer_groups if lg.is_prod()]
    assert len(cc.element_regions) == len(prod_groups)
    for (first, end, slots), lg in zip(cc.element_regions, prod_groups):
        assert first == min(l._layer_nid_range[0] for l in lg.layers) and end == max(l._layer_nid_range[1] for l in lg.layers)
        assert slots == max(slots_of(cc, n, ns) for l in lg.layers for ns in l.nodes)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_permuted"])
@pytest.mark.parametrize("B", [1, 3, 16, 33])
def test_layout_is_aligned_disjoint_and_tight(kind, B, build_pc):
    cc, _ = compiled(build_pc, kind)
    lay = buffer_layout(cc.input_range, cc.sum_regions, cc.element_regions, B)
    input_start, input_end = cc.input_range
    spans = [(0, input_end * B)]
    for (first, end, slots), offset, width in zip(cc.sum_regions, lay["sum_offsets"], lay["sum_widths"]):
        assert width == -(-B * slots // ALIGN) * ALIGN                        # the block, rounded up to ALIGN
        spans.append((offset, offset + (end - first) * width))
    assert all(start % ALIGN == 0 for start, _ in spans)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))     # in order, disjoint
    assert lay["node_size"] == spans[-1][1]
    assert lay["element_size"] == max((end - first) * w for (first, end, _), w in
                                      zip(cc.element_regions, lay["element_widths"]))
    assert cc.buffer_bytes(B) == 4 * (lay["node_size"] + lay["element_size"] + (input_end - input_start) * cc.num_classes)


@pytest.mark.parametrize("kind", ["hmm", "pd", "hand_unit"])
def test_an_input_layer_writes_the_input_region_as_in_a_tensor_circuit(kind, build_pc):
    cc, n = compiled(build_pc, kind)
    pc, B = cc.pc, 5
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, V, (B, n), generator = g).to(pc.device)
    missing = (torch.rand(B, n, generator = g) < 0.4).to(pc.device)
    bufs = cc._buffers(B)
    for layer in pc.input_layer_group:                                    # as TensorCircuit.forward calls them
        layer(x.permute(1, 0), bufs["input_mars"], missing_mask = missing)
    pc(x, missing_mask = missing)
    lo, hi = cc.input_range
    assert torch.equal(bufs["input_mars"][lo:hi], pc.node_mars[lo:hi])


@pytest.mark.parametrize("kind", ["hmm", "pd"])
def test_views_sit_in_their_regions(kind, build_pc):
    cc, n = compiled(build_pc, kind)
    B = 5
    bufs = cc._buffers(B)
    lay, node_mars = bufs["layout"], bufs["node_mars"]
    assert bufs["input_mars"].data_ptr() == node_mars.data_ptr()                              # offset 0
    # writing the input region (as an input layer does) leaves every other region untouched
    node_mars.fill_(7.0)
    x = torch.randint(0, V, (B, n), device = cc.pc.device)
    for layer in cc.pc.input_layer_group:
        layer(x.permute(1, 0), bufs["input_mars"])
    end_of_inputs = cc.input_range[1] * B
    assert (node_mars[end_of_inputs:] == 7.0).all()


def test_buffers_are_reused_across_batch_sizes(build_pc):
    cc, _ = compiled(build_pc, "pd")
    ptr = lambda B: cc._buffers(B)["node_mars"].data_ptr()
    p64 = ptr(64)
    assert ptr(32) == p64 and ptr(64) == p64                               # within 4x: re-cut, same addresses
    assert ptr(128) != p64                                                 # too small: grown
    p128 = ptr(128)
    assert ptr(8) != p128                                                  # more than 4x too large: shrunk
    fresh = cc._buffers(256)["node_mars"]                                  # a new allocation starts at -inf
    assert torch.isneginf(fresh).all()


def test_per_group_widths_are_smaller_than_per_layer_widths(build_pc):
    def per_layer(cc):
        total = 0
        for lg in cc.pc.inner_layer_groups:
            if lg.is_sum():
                for layer in lg.layers:
                    lo, hi = layer._layer_nid_range
                    total += (hi - lo) * max(s for first, _, s in cc.sum_regions if lo <= first < hi)
        return 4 * (total + max((e - f) * s for f, e, s in cc.element_regions) + cc.input_range[1])
    hmm, _ = compiled(build_pc, "hmm")
    assert hmm.bytes_per_sample == per_layer(hmm)                          # one node group per sum layer
    pd, _ = compiled(build_pc, "pd")
    assert pd.bytes_per_sample < per_layer(pd)


def test_to_reallocates_the_buffers_on_the_new_device(build_pc):
    cc, _ = compiled(build_pc, "hmm")
    assert cc._buffers(4)["node_mars"].device.type == "cuda"
    cc.to("cpu")
    bufs = cc._buffers(4)
    assert bufs["node_mars"].device.type == "cpu" and bufs["layout"]["sum_offsets_t"].device.type == "cpu"
