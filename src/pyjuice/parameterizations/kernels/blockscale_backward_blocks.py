"""
Triton element-flow backward of `BlockScale`, a parent BLOCK at a time (`blockscale/backward.py`'s kind
("blocks", splits)).

`blockscale_backward._bs_triton_ele_kernel` walks the parents in the layer's k-tiles. Where the parent blocks are
narrower than a k-tile (`ptr_inc_step > 1`: every 32-wide gated layer, its k-tile 64 or 128), each k-tile is
`ptr_inc_step` dots over the whole tile with the other blocks' lanes zeroed -- `ptr_inc_step` times the products --,
because the gate differs per parent block. And its programs are only (batch tiles x element tiles): a 4096-node layer
of 32-blocks at batch 64 is 256 programs at 255 registers, 11.5% occupancy (ncu), 494 us a layer against a floor of
a few tens.

This kernel takes ONE parent block a step, from absolute tables the plan builds once (each step's first parent node,
first parameter and gate row: `full_ebase` / `full_pbase` / the gate table, per (k-tile, group) flattened), so a step
is one dot of the block's own width; and it splits the parent blocks into `SPLITS` ranges over a third grid axis, each
range's log-sum into a scratch row, merged after (`_bs_blocks_merge_kernel`). The gate gradient is emitted per parent
block as the original does at `ptr_inc_step > 1` -- a block belongs to one range, so its writers are the original's.
The steps' arithmetic is the original's (the TF32-rounded dot, the same running log-sum).
"""

import triton
import triton.language as tl

from pyjuice.utils.kernel_launcher import triton_jit
from .blockscale_backward import _tf32_rna


@triton_jit
def _bs_triton_ele_blocks_kernel(node_flows, out, node_mars, element_mars, mparams, ext, gate, chids, ebase_t, pbase_t,
                                 grad_ext, batch_size: tl.constexpr, num_steps: tl.constexpr,
                                 steps_per_split: tl.constexpr, BLOCK_B: tl.constexpr, TILE_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, TL_DOT: tl.constexpr,
                                 GATE_CBS: tl.constexpr, ext_base, SPLITS: tl.constexpr, out_rows,
                                 WRITE_GRAD: tl.constexpr = 0, GRAD_ATOMIC: tl.constexpr = 1, pid_m_offset = 0):
    """`out`: `element_flows` (SPLITS == 1), else the scratch [SPLITS, out_rows, batch_size], a row an element of the
    layer's element blocks in order (`eleblock_id * BLOCK_SIZE_M + m`)."""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1) + pid_m_offset
    pid_s = tl.program_id(2)

    eleblock_id = pid_m // (BLOCK_SIZE_M // TILE_SIZE_M)
    tile_id = pid_m % (BLOCK_SIZE_M // TILE_SIZE_M)

    offs_ele = tl.arange(0, TILE_SIZE_M) + tile_id * TILE_SIZE_M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_batch = tl.arange(0, BLOCK_B) + pid_b * BLOCK_B
    mask_batch = offs_batch < batch_size

    off_eleids = tl.load(chids + eleblock_id)
    emars = tl.load(element_mars + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:],
                    mask = mask_batch[None,:])                     # [TILE_SIZE_M, BLOCK_B]
    offs_gate = offs_ele // GATE_CBS

    # The steps' terms summed in linear space against a running maximum (`acc_m`), one log at the end: the log-space
    # merge a step (`log(pf + exp(acc - pfmax) + 1e-32) + pfmax` or its mirror, both branches evaluated) was 2 logs and 2
    # exps an element a step, ~a third of the kernel's stall samples (ncu, Codpula's layer at 29 samples). The 1e-32
    # stays, once, in the final log: a range whose terms are all exactly 0 keeps the finite value it had.
    acc_m = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32) - float("inf")
    acc_l = tl.zeros([TILE_SIZE_M, BLOCK_B], dtype = tl.float32)
    GROWS: tl.constexpr = TILE_SIZE_M // GATE_CBS if TILE_SIZE_M >= GATE_CBS else 1

    k0 = pid_s * steps_per_split
    for kk in range(0, steps_per_split):
        k = k0 + kk
        if k < num_steps:
            tab = eleblock_id * num_steps + k
            par_start = tl.load(pbase_t + tab)
            edge_start = tl.load(ebase_t + tab)
            gbase = tl.load(gate + tab)

            epars = tl.load(mparams + offs_ele[:,None] * BLOCK_SIZE_K + (par_start + offs_k)[None,:]).to(tl.float32)
            rows = (edge_start + offs_k)[:,None] * batch_size + offs_batch[None,:]
            nflows = tl.load(node_flows + rows, mask = mask_batch[None,:])
            nmars = tl.load(node_mars + rows, mask = mask_batch[None,:])
            log_n_fdm = tl.where(nmars == -float("inf"), -float("inf"), nflows - nmars)
            log_n_fdm_max = tl.max(log_n_fdm, axis = 0)[None,:]
            n_fdm_sub = tl.where(log_n_fdm_max != -float("inf"), tl.exp(log_n_fdm - log_n_fdm_max), 0.0)

            if TL_DOT == 1:
                partial_flows = tl.dot(_tf32_rna(epars), _tf32_rna(n_fdm_sub))
            else:
                # axis-0 form on purpose -- see `_BROADCAST_SUM_NOTE` in pyjuice/layer/kernels/__init__.py
                partial_flows = tl.sum(tl.trans(epars)[:,:,None] * n_fdm_sub[:,None,:], axis = 0)

            lphi = tl.where(
                gbase >= 0,
                tl.load(ext + (gbase + ext_base + offs_gate)[:,None] * batch_size + offs_batch[None,:],
                        mask = mask_batch[None,:] & (gbase >= 0), other = 0.0),
                -float("inf")
            )
            partial_flows_max = emars + log_n_fdm_max + lphi

            if WRITE_GRAD:
                contrib = partial_flows * tl.exp(partial_flows_max)
                if TILE_SIZE_M >= GATE_CBS:
                    gacc = tl.sum(tl.reshape(contrib, (TILE_SIZE_M // GATE_CBS, GATE_CBS, BLOCK_B)), axis = 1)
                    grow = gbase + ext_base + (tile_id * TILE_SIZE_M) // GATE_CBS + tl.arange(0, TILE_SIZE_M // GATE_CBS)
                else:
                    gacc = tl.sum(contrib, axis = 0)[None,:]
                    grow = gbase + ext_base + (tile_id * TILE_SIZE_M) // GATE_CBS + tl.arange(0, 1)
                if GRAD_ATOMIC:
                    tl.atomic_add(grad_ext + grow[:,None] * batch_size + offs_batch[None,:], gacc,
                                  mask = mask_batch[None,:] & (gbase >= 0))
                else:
                    tl.store(grad_ext + grow[:,None] * batch_size + offs_batch[None,:], gacc,
                             mask = mask_batch[None,:] & (gbase >= 0))

            new_m = tl.maximum(acc_m, partial_flows_max)
            acc_l = (acc_l * tl.where(acc_m == -float("inf"), 0.0, tl.exp(acc_m - new_m))
                     + partial_flows * tl.where(partial_flows_max == -float("inf"), 0.0, tl.exp(partial_flows_max - new_m)))
            acc_m = new_m

    acc = tl.where(acc_m == -float("inf"), -float("inf"), tl.log(acc_l + 1e-32) + acc_m)
    if SPLITS == 1:
        tl.store(out + (off_eleids + offs_ele[:,None]) * batch_size + offs_batch[None,:], acc, mask = mask_batch[None,:])
    else:
        r = pid_s * out_rows + eleblock_id * BLOCK_SIZE_M + offs_ele
        tl.store(out + r[:,None] * batch_size + offs_batch[None,:], acc, mask = mask_batch[None,:])


@triton_jit
def _bs_blocks_merge_kernel(scratch, element_flows, chids, batch_size: tl.constexpr, out_rows,
                            BLOCK_SIZE_M: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    """element_flows[chids[i // BLOCK_SIZE_M] + i % BLOCK_SIZE_M, b] = log sum_s exp scratch[s, i, b]."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = out_rows * batch_size
    mask = offs < n
    i = offs // batch_size
    b = offs - i * batch_size
    m = tl.zeros([BLOCK], dtype = tl.float32) - float("inf")
    for s in tl.static_range(SPLITS):
        m = tl.maximum(m, tl.load(scratch + s * n + offs, mask = mask, other = -float("inf")))
    tot = tl.zeros([BLOCK], dtype = tl.float32)
    for s in tl.static_range(SPLITS):
        v = tl.load(scratch + s * n + offs, mask = mask, other = -float("inf"))
        tot += tl.where(m == -float("inf"), 0.0, tl.exp(v - m))
    res = tl.where(m == -float("inf"), -float("inf"), tl.log(tot) + m)
    row = tl.load(chids + i // BLOCK_SIZE_M, mask = mask, other = 0) + i % BLOCK_SIZE_M
    tl.store(element_flows + row * batch_size + b, res, mask = mask)
