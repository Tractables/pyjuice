"""
`pyjuice.utils.transpose.transposed_rows` against torch's own transposed copy, bit for bit: consecutive and
gathered rows, into fp32 / bf16 / fp16 and into the columns of a wider table (nothing written around them), shapes
over several of the kernel's tiles and neither a multiple of them, launches split over the rows (the grid's limit
lowered), and torch's copy itself where the kernel does not apply (the CPU, an output whose columns are apart).
"""
import pytest
import torch

from pyjuice.utils import transpose as T

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
DEV = torch.device("cuda:0")


def table_of(R, V, dtype = torch.float32, dev = DEV):
    g = torch.Generator(device = dev).manual_seed(0)
    return torch.rand(R, V, device = dev, generator = g).to(dtype)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("pick", ["consecutive", "gathered"])
@pytest.mark.parametrize("shape", [(5, 37), (300, 1000), (1, 64), (128, 1)])
def test_the_rows_token_major_are_torchs_copy(shape, pick, dtype):
    R, V = shape
    table = table_of(R, V)
    if pick == "consecutive":
        first, K, rows = min(1, R - 1), max(1, R - 2), None
        want = table[first:first + K].t().to(dtype)
    else:
        first, rows = 0, torch.randperm(R, generator = torch.Generator().manual_seed(1))[:max(1, R // 3)].sort().values
        rows = rows.to(DEV)
        K = rows.numel()
        want = table[rows].t().to(dtype)
    out = torch.full((V, K + 7), float("nan"), dtype = dtype, device = DEV)
    assert T.transposed_rows(table, out[:, 3:3 + K], rows = rows, first = first).data_ptr() == out[:, 3:].data_ptr()
    assert torch.equal(out[:, 3:3 + K], want)
    assert out[:, :3].isnan().all() and out[:, 3 + K:].isnan().all()


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_whole_table_is_its_transpose(dtype):
    table = table_of(257, 333, dtype)
    out = torch.empty(333, 257, dtype = dtype, device = DEV)
    assert torch.equal(T.transposed_rows(table, out), table.t().contiguous())


@cuda_only
@pytest.mark.parametrize("gathered", [False, True])
def test_launches_split_over_the_rows_give_the_same_copy(gathered, monkeypatch):
    monkeypatch.setattr(T, "GRID_ROWS", 2)                       # 2 x 128 rows a launch: 3 launches over 600
    table = table_of(700, 90)
    rows = torch.arange(3, 603, device = DEV) if gathered else None
    out = torch.empty(90, 600, device = DEV)
    T.transposed_rows(table, out, rows = rows, first = 3)
    assert torch.equal(out, table[3:603].t())


def test_where_the_kernel_does_not_apply_it_is_torchs_copy():
    table = table_of(20, 30, dev = torch.device("cpu"))
    out = torch.empty(30, 20)
    assert torch.equal(T.transposed_rows(table, out), table.t())
    if torch.cuda.is_available():                                # columns apart: stride(1) != 1
        table = table.to(DEV)
        apart = torch.empty(20, 30, device = DEV).t()
        assert torch.equal(T.transposed_rows(table, apart), table.t())
