"""
The fused sum kernels' tiles: their shapes per product precision, the splits of a launch's output tiles and the pre-pass's buffers
under them (see :mod:`.fused_forward`).
"""

import functools
import torch
import triton.language as tl

from pyjuice.utils.util import host_cdiv


#: The fused kernel's tiles (TILE_M, TILE_E, TILE_B, num_warps, num_stages) per product precision, from a sweep on
#: a 4096-state HMM layer (4 x 1024 nodes, 4096 children) at 69 to 8832 columns.
_TILES = {"tf32": (128, 32, 128, 8, 3), "tf32x3": (64, 32, 64, 4, 3), "bf16x3": (64, 32, 64, 4, 3)}


def _tile_warps(warps: int, rows: int) -> int:
    """A fused launch's warps for output tiles of ``rows`` rows: the table's (:data:`_TILES`, tuned on tiles of 128
    rows), at most 4 for tiles of 32 rows or fewer (node blocks of 32: at 8 warps a launch took up to 2.4x as long as
    at 4, with the same results; tiles of 128 rows and few columns lost 7% at 4)."""
    return warps if rows > 32 else min(warps, 4)


#: A fused launch of fewer programs than this many per SM splits every program's reduction (over its parents or
#: children) in shares, each a program of its own (:func:`_split_merge`), as long as a share keeps
#: ``SPLIT_MIN_SHARE`` chunks of it (dots); a power of two, at most ``SPLIT_MAX``. A launch of few, long programs (an
#: HMM's layer of 1024 nodes in blocks of 32 at small batch: one program per tile of 32 nodes, 32 chunks of children
#: each) leaves most of the GPU idle. (2 and 2 before: shares of 2 chunks merged at a cost above their gain -- an LL1024
#: top-down launch took 3.6x as long; 4 and 8 ran HMM, PD and Codpula-shaped queries as fast or faster.)
SPLIT_PROGRAMS_PER_SM = 4
SPLIT_MIN_SHARE = 8
SPLIT_MAX = 16


def split_count(programs: int, chunks: int, dev) -> int:
    """Into how many shares a fused launch of ``programs`` programs, each reducing over ``chunks`` chunks (the most
    any has), splits every program's reduction: 1 (none) when the launch fills the GPU."""
    want = host_cdiv(SPLIT_PROGRAMS_PER_SM * _num_sms(dev), max(1, programs))
    split = 1
    while split * 2 <= min(want, SPLIT_MAX, chunks // SPLIT_MIN_SHARE):
        split *= 2
    return split


#: The most output tiles a split launch has (a launch of more does not split): their counters lead the split buffer.
SPLIT_TILES = 4096
_SPLIT_TILES = tl.constexpr(SPLIT_TILES)
_SPLIT_BUFFERS = {}
_NO_SPLIT = {}
#: A fused top-down launch whose parents are read by at least this many tiles of children gets a pre-pass
#: (:func:`_top_down_prepass_kernel`): every parent chunk's values less their log Z, their maximum per column and
#: their shifted exponentials, once instead of once per tile of children.
PRE_MIN_TILES = 4
_PRE_BUFFERS = {}


def pre_buffer(dev, size: int) -> torch.Tensor:
    """The pre-pass's buffer on ``dev``, of ``size`` floats at least, kept per device and grown as needed."""
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _PRE_BUFFERS.get(key)
    if buf is None or buf.numel() < size:
        buf = torch.empty(max(size, 1 << 20), dtype = torch.float32, device = dev)
        _PRE_BUFFERS[key] = buf
    return buf


def split_buffer(dev, tiles: int, per_tile: int) -> torch.Tensor:
    """The split launches' buffer on ``dev`` (float32, zeros when made): ``SPLIT_TILES`` counters (int32 bits, 0
    between launches: the last share in resets its tile's), then ``per_tile`` floats for each of ``tiles`` output
    tiles; kept per device, grown as needed."""
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _SPLIT_BUFFERS.get(key)
    if buf is None or buf.numel() < SPLIT_TILES + tiles * per_tile:
        buf = torch.zeros(SPLIT_TILES + max(tiles * per_tile, 1 << 20), dtype = torch.float32, device = dev)
        _SPLIT_BUFFERS[key] = buf
    return buf


@functools.lru_cache(maxsize = None)
def _num_sms_of(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _num_sms(dev) -> int:
    return _num_sms_of(dev.index if dev.index is not None else torch.cuda.current_device())
