"""
`ExternalSumParams._sum_addends` -- a tie group's `F-` statistics summed for the dual-flow M-step: in one launch where
they are contiguous fp32 views of one buffer, the same bits as torch's in-order adds (views in any order, so negative
offsets; a ragged length past a block); the adds otherwise (views of separate buffers, non-contiguous views, fp64).
"""
import pytest
import torch

from pyjuice.nodes import BlockScaleSumParams

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _in_order(addends):
    out = addends[0].clone()
    for a in addends[1:]:
        out = out + a
    return out


@cuda_only
@pytest.mark.parametrize("starts,shape", [
    ((0, 1000, 2000), (250, 4)),
    ((7000, 0, 3000, 1500, 9000), (1000,)),                 # out of order: negative offsets from the first
    ((5, 2006, 4007), (1999,)),                             # unaligned starts, a length past a block of 1024
    ((0, 4096), (64, 64)),
])
def test_one_buffer_views_give_the_in_order_adds_bits(starts, shape):
    torch.manual_seed(0)
    buf = torch.randn(12000, device = "cuda") * 1e3
    n = 1
    for s in shape:
        n *= s
    views = [buf[s:s + n].view(shape) for s in starts]
    ps = BlockScaleSumParams(ch_block_size = 4)
    got = ps._sum_addends(views)
    assert got.shape == views[0].shape and got.dtype == torch.float32
    assert torch.equal(got, _in_order(views))
    assert torch.equal(ps._sum_addends(views), got)          # (the cached offset table)
    assert all(got.data_ptr() != v.data_ptr() for v in views)


@cuda_only
@pytest.mark.parametrize("case", ["separate buffers", "non-contiguous", "fp64"])
def test_other_addends_take_the_adds(case):
    torch.manual_seed(1)
    if case == "separate buffers":
        views = [torch.randn(300, device = "cuda") for _ in range(4)]
    elif case == "non-contiguous":
        buf = torch.randn(64, 64, device = "cuda")
        views = [buf[:, :8], buf[:, 8:16], buf[:, 16:24]]
    else:
        buf = torch.randn(3000, device = "cuda", dtype = torch.float64)
        views = [buf[0:1000], buf[1000:2000], buf[2000:3000]]
    got = BlockScaleSumParams(ch_block_size = 4)._sum_addends(views)
    assert torch.equal(got, _in_order(views))
