"""
The forward of :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams` (:class:`_ForwardMixin`, one
of its bases): a plan per layer and batch size -- every partition's kernel fork (CuTe/TMA, small-batch CUDA or
Triton) and launch arguments, resolved once and checked against each kernel's assumptions -- and the forward that runs
it, computing every gated node's value with its gates folded in (the standard sum forward is replaced, not
corrected).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch

from .common import _buffer_kwarg


def _stream_fused_module():
    from ..kernels.c import get_stream_fused_module
    return get_stream_fused_module()


def form_elements(element_mars, node_mars, children):
    """`pyjuice.layer.fused_children.form_elements` (imported at the call: `pyjuice.layer` imports `pyjuice.nodes`)."""
    from pyjuice.layer.fused_children import form_elements as form
    form(element_mars, node_mars, children)

#: How many forward plans to keep per layer, keyed by batch size. Each holds a `log Z` buffer per
#: partition sized by that batch, so this bounds memory; rebuilding an evicted one costs an AUTOTUNE,
#: ~66 ms a layer, which is why the bound is generous and the eviction is LRU rather than FIFO.
#:
#: 8 was far too small, and it failed discontinuously rather than gracefully. A caller whose batch is a
#: block count cycles through its distinct values, and FIFO against a cyclic pattern evicts precisely
#: the entry needed next: measured, 9 distinct shapes against 8 slots gave a 0%% hit rate, not 8/9 --
#: 750 ms a call against 4.3 ms at 8 shapes. A real workload (GSM8K micro-batches) has 20 distinct
#: values spanning 6-25, so this covers it outright with headroom.
#:
#: NOT keyed on a coarser bucket such as `next_power_of_2(batch)`, tempting as that is: CuTe
#: eligibility is gated on `batch %% 64 == 0`, so batches 96 and 128 share a bucket but not a set of
#: candidate forks, and they would silently share a plan built for the wrong one.
_FW_PLAN_CACHE = 32

#: The forks whose products round the parameters to bf16 -- the CuTe fork, its pre-pass copy, the tensor-core
#: small-batch fork from batch 16 (below it its high-precision form; no copy is made there) -- and the split forward,
#: which reads bf16 parameters only: these read the circuit's bf16 copy of tied fp32 parameters where a call has one
#: (`forward_layer`'s `op_params`; the same bits, half the bytes), the fp32 forks and the Triton fallback `params`.
_BF16_FORKS = frozenset(("blockscale_forward", "blockscale_forward_prepass", "blockscale_forward_stream",
                         "blockscale_sb_forward_mma"))


class _ForwardMixin:
    """The forward of :class:`~pyjuice.nodes.external_params.block_scale.BlockScaleSumParams`, one of its bases."""

    def _build_plan(self, layer, ns_tensors, node_mars, element_mars, params, external_params, op = None):
        """
        Resolve every per-layer launch argument once, and check each kernel's assumptions. `op`: the circuit's bf16
        copy of the fp32 `params` (`forward_layer`'s `op_params`) or None, what the forks of `_BF16_FORKS` read.

        FIVE kernels serve this type, and which apply is a property of the shape:

          * the CuTe/TMA fork, for batches it can tile (`batch % 64 == 0`) on sm_90+ with CUTLASS. It
            carries the normalizer as a gate-factored contraction against a precomputed `sigma`;
          * its pre-pass copy (kernels/c/blockscale_forward_prepass.cu), bit-identical, where an edge tile is one run of
            children (`ch_block_size >= 64`) and a gate spans at least 4 of them: the fork's work that
            depends on the edge tile alone done once per node block, the running maxima as prefix
            maxima, the tiles' MMAs spread over groups of warps and merged in order;
          * the split forward (kernels/c/blockscale_forward_stream.cu), bf16 parameters in the pre-pass copy's scope:
            the children cut into ranges each under one stabilizer, so each range is a plain bf16 GEMM, the ranges'
            partials merged after, and the normalizer from `sigma` on the SMs the GEMM leaves idle -- not the fork's
            bits (more accurate: fp32-level `log Z`). Its (tile, split) is measured once per shape (`_stream_code`);
          * a plain-CUDA small-batch kernel, one warp per 32 nodes, which accumulates the normalizer
            inline and needs neither `batch % 64` nor `num_edges % 64` nor CUTLASS;
          * the portable Triton fork, which needs no CUDA toolchain at all and applies everywhere,
            including the shapes the other two decline (a gate finer than the node block, in
            particular, is served by it alone).

        All three are collected here and the choice is MEASURED, so where they overlap the faster one
        wins rather than the one that happened to be checked first.
        """
        from ..kernels.c import get_cute_module, get_sb_module, get_sb_bt_module, get_sb_mma_module, get_prepass_module, get_stream_module
        from ..kernels import c as ck_ext
        import pyjuice.layer.kernels.c as ck

        # NEITHER CUDA extension is required any more. The CuTe forward needs nvcc, CUTLASS and
        # sm_90+, and the small-batch forward needs nvcc, so a machine without a CUDA toolchain used
        # to be refused outright here. The portable Triton fork below serves those machines -- slower
        # than either, but the alternative was not running at all.
        mod = get_cute_module()
        sb_mod = get_sb_module()
        pre_mod = get_prepass_module()

        if getattr(layer, "ext_slots", None) is None:
            raise NotImplementedError(
                "`BlockScaleSumParams` needs the compiled edge-block tables, which require a "
                "batch-innermost storage layout."
            )

        ns0 = ns_tensors[0][0].ns
        batch_size = node_mars.size(1)
        block_size = layer.block_size

        n_node_gates, n_child_gates = self.gate_counts(ns0)
        gate_bs, gate_cbs = self.gate_sizes(ns0)
        node_cbs = ns0.ch_block_size

        # A gate spanning fewer nodes than the block is served by the portable Triton forward alone: it
        # keeps its M tile inside one node gate, so `phi` is still constant over the tile's rows and
        # folds onto the child operand. The two CUDA forks apply one gate per edge block to a whole node
        # block and are declined above, so `calls` / `sb_calls` come back empty and the Triton path is
        # what remains -- no separate dispatch needed here.

        # The tiles that fit BOTH this layer's shape and this DEVICE. The kernel decides the second
        # part: its shared-memory need depends on the gate width, and the opt-in ceiling is a property
        # of the part (48 / 64 / 100 / 227 KB), not something to hardcode.
        cfgs = [tuple(c) for c in mod.configs()] if mod is not None else []
        valid = ([int(i) for i in mod.fitting_configs(block_size, batch_size, gate_cbs)]
                 if (mod is not None and block_size == gate_bs) else [])
        cfg = valid[0] if valid else 0    # provisional; measured once the launch args exist
        # The pre-pass copy's tiles for this shape: it takes the fork's arguments, in the fork's single-run case
        # (`node_cbs >= 64`), with a gate width >= 4, and like the fork one gate per node block.
        pre_valid = ([int(i) for i in pre_mod.fitting_configs(block_size, batch_size, node_cbs, gate_cbs)]
                     if (pre_mod is not None and block_size == gate_bs) else [])
        # What the bf16-product forks read (`_BF16_FORKS`): the copy where the call has one
        p_bf = params if op is None else op
        # The split forward's tiles: the same tuples and scope (bf16 parameters, node blocks of 64s).
        stream_mod = get_stream_module() if p_bf.dtype == torch.bfloat16 else None
        stream_valid = ([int(i) for i in stream_mod.fitting_configs(block_size, batch_size, node_cbs, gate_cbs)]
                        if (stream_mod is not None and block_size == gate_bs) else [])

        # The small-batch kernel needs only 32-node groups, but its children must be contiguous across
        # the WHOLE row rather than within each edge tile -- it walks the edge axis with a single base.
        # Neither CUDA fork carries a node-gate axis: each derives one gate per edge block and applies
        # it to a whole node block. A finer node gate falls to the portable Triton forward, which
        # indexes the gate per M tile.
        node_gates = block_size // gate_bs
        sb_ok = (sb_mod is not None) and (block_size % 32 == 0) and node_gates == 1

        self._require_full_supply(layer, ns_tensors)

        ext_base = self._ext_base(ns_tensors, external_params, batch_size)

        dev = node_mars.device
        calls, sb_calls, shift_args, log_zs = [], [], [], []
        for partition_id in range(layer.num_fw_partitions):
            nids = layer.partitioned_nids[partition_id]
            cids = layer.partitioned_cids[partition_id]
            pids = layer.partitioned_pids[partition_id]
            gate = layer.ext_slots[0][partition_id]

            num_edges = cids.size(1)
            if num_edges % node_cbs != 0:
                raise NotImplementedError(
                    f"the compiled edge count ({num_edges}) must be a whole number of child blocks "
                    f"({node_cbs})."
                )

            rows = nids.size(0)
            n_eblks = num_edges // node_cbs
            log_z = torch.empty([rows * block_size * batch_size], dtype = torch.float32, device = dev)

            # `log Z` is what the backward needs to turn the stored `log N - log Z` back into `log N`
            # (see `pre_backward_layer`). Collected here rather than dug out of `calls` afterwards,
            # because the two kernels order their arguments differently.
            shift_args.append((nids, log_z, rows))
            log_zs.append(log_z)

            c2 = cids.to(torch.int64)
            p2 = pids.to(torch.int64)
            ar_e = torch.arange(num_edges, device = cids.device, dtype = torch.int64)

            # PADDED SLOTS ARE EXCUSED from the contiguity requirements below, because the kernels
            # already compute exactly nothing for them.
            #
            # A row with fewer edge blocks than the partition's width is padded out with `cids == 0`
            # and `pids == 0` -- the dummy element and the dummy parameter -- and that is an EXACT
            # test, not a heuristic: a real child is an element `>= num_dummy_eles > 0` and a real
            # parameter is `>= num_dummy_params > 0`.
            #
            # Requiring contiguity over such a slot is what refused every ragged topology. It is not
            # something the kernels need. `ext_slots` carries `-1` for a padded EDGE BLOCK, the
            # forward reads that as `log phi = -inf`, and the fold happens BEFORE the max-stabilizer
            # is taken -- so a padded lane is `-inf` in the staged `element_mars` and contributes
            # exactly zero to `N`, while its normalizer operand `exp(log phi - mz)` is exactly zero
            # and so it contributes nothing to `Z` either. Neither depends on what the lane's DERIVED
            # addresses (`ebase + j`, `pbase + j * block_size`) happen to read, which is what lets the
            # base+stride addressing survive a padded -- even a part-padded -- tile. The gate table is
            # indexed per edge block, which is FINER than the 64-wide tile, so this covers a tile that
            # is part real and part padding, not merely a wholly padded one.
            #
            # What must still be refused is genuine BLOCK-SPARSITY: a row whose REAL children are not
            # a contiguous run cannot be addressed from one base at all. Hence the mask rather than
            # dropping the check.
            pad = (c2 == 0) & (p2 == 0)
            assert bool((((c2 == 0) == (p2 == 0))).all()), \
                "`cids == 0` and `pids == 0` disagree on which compiled slots are padding; the " \
                "padding convention this relaxation depends on has changed."

            # ---- the CuTe fork reads the COMPILED TABLES at step-run granularity ----
            #
            # `step = min(node_cbs, 64)` is the widest run of compiled edge slots that the compile
            # step GUARANTEES is contiguous in `element_mars` and `block_size`-strided in `params`:
            # `_compile_partition_edge_block_ids` asserts "each edge block's children occupy a
            # contiguous run of `cids`". So the kernel takes one `cids` / `pids` entry per step-run
            # and issues one bulk transfer per run, instead of assuming a whole 64-wide tile is one
            # run. That holds for ANY topology -- dense, ragged or block-sparse -- and it is why this
            # fork no longer needs a contiguity test, a padding mask, or a bounds clamp:
            #
            #   * a padded run has `cids == 0` / `pids == 0`, the dummy `-inf` element and the dummy
            #     zero parameter, exactly as the plain sum layer's padding works;
            #   * a block-sparse row's runs simply sit at unrelated offsets, which is now expressible.
            #
            # The within-run property is still VERIFIED rather than trusted -- it is an invariant of
            # another module, and a silent violation would be a wrong answer rather than a crash.
            if (valid or pre_valid or stream_valid) and num_edges % 64 == 0:
                step = min(node_cbs, 64)
                nrun = num_edges // step
                c3, p3 = c2.view(-1, nrun, step), p2.view(-1, nrun, step)
                ar = torch.arange(step, device = cids.device, dtype = torch.int64)

                # A run is all-or-nothing padded: padding is applied by whole edge blocks and a run
                # lies inside one, so it cannot be part real and part padding. Asserted, because the
                # verification below relies on it -- a padded run is all zeros rather than a
                # contiguous run starting at zero, so it has to be excused rather than checked.
                pad3 = pad.view(-1, nrun, step)
                assert bool((pad3.any(-1) == pad3.all(-1)).all()), \
                    "a step-run is part real and part padding, which the compiled layout should " \
                    "make impossible (padding is by whole edge blocks)."
                run_pad = pad3.all(-1).unsqueeze(-1)

                # Below fp32, the kernel's 8-parameter loads are one 16-byte load each: its runs' first parameters
                # must sit at multiples of 8 (`SumLayer._cuda_param_loads_aligned`; at fp32 the layout it always had)
                if (bool(((c3 == c3[:, :, :1] + ar.view(1, 1, -1)) | run_pad).all())
                        and bool(((p3 == p3[:, :, :1]
                                   + ar.view(1, 1, -1) * block_size) | run_pad).all())
                        and (p_bf.dtype == torch.float32
                             or (block_size % 8 == 0 and bool((p3[:, :, 0] % 8 == 0).all())))):
                    # The COMPILED TABLES go to the kernel unchanged -- no derived `ebase` / `pbase`
                    # slices to build here or to drift from what the layer compiled.
                    calls.append((nids, c3[:, :, 0].contiguous(), p3[:, :, 0].contiguous(),
                                  gate, log_z,
                                  block_size, num_edges, node_cbs, gate_cbs, n_node_gates,
                                  ext_base, cfg))

            # ---- the small-batch kernel reads the same tables, at EDGE-BLOCK granularity
            #
            # It loops per gate and already resolves the edge block for its gate lookup, so it takes
            # that block's `cids` / `pids` entry directly. A gate lies inside one edge block
            # (`gate_cbs` divides `node_cbs`), so its inner loop stays within the run the compile step
            # guarantees is contiguous -- the same invariant the CuTe fork relies on, at the same
            # granularity. It therefore needs no whole-row contiguity, no padding mask and no reach
            # bound; a padded block's entries are the dummies and contribute nothing.
            if sb_ok and num_edges % node_cbs == 0:
                nblk = num_edges // node_cbs
                cb, pbk = c2.view(-1, nblk, node_cbs), p2.view(-1, nblk, node_cbs)
                arb = torch.arange(node_cbs, device = cids.device, dtype = torch.int64)
                blk_pad = ((cb == 0) & (pbk == 0)).all(-1).unsqueeze(-1)
                if (bool(((cb == cb[:, :, :1] + arb.view(1, 1, -1)) | blk_pad).all())
                        and bool(((pbk == pbk[:, :, :1]
                                   + arb.view(1, 1, -1) * block_size) | blk_pad).all())):
                    # Whether the WHOLE row is one contiguous run -- the dense fully-connected case.
                    # The kernel specializes on it: reading a base per edge block costs two dependent
                    # global loads per gate, standing in front of the parameter address, which
                    # measured +9.5% where it is not needed. When it holds, the kernel keeps the
                    # single pair of row bases it always used.
                    row_contig = bool((c2 == c2[:, :1] + ar_e.view(1, -1)).all()
                                      and (p2 == p2[:, :1] + ar_e.view(1, -1) * block_size).all())
                    sb_calls.append((nids, cb[:, :, 0].contiguous(), pbk[:, :, 0].contiguous(),
                                     gate, log_z,
                                     block_size, num_edges, node_cbs, gate_cbs, ext_base,
                                     int(row_contig), 0))

        # Every partition must be served by the same kernel -- a plan that covers only some of them
        # would silently leave the rest of the layer unevaluated.
        if len(calls) != layer.num_fw_partitions:
            calls = []
        if len(sb_calls) != layer.num_fw_partitions:
            sb_calls = []

        # ---- the PORTABLE Triton fallback ----
        #
        # A FALLBACK, deliberately not an autotune candidate. Offered only where neither CUDA fork
        # applies, so a shape they do serve keeps exactly the candidate list, the measurement and the
        # pick it had before this existed -- and never pays to COMPILE a Triton kernel it would not
        # use, which is the mistake that once put 230s of compilation into the test suite.
        #
        # It exists for reach rather than speed: both CUDA forks need nvcc (and the CuTe one CUTLASS
        # and sm_90+), so without a CUDA toolchain this parameterization was refused outright rather
        # than merely being slow. It also has no layout requirements at all, reading `nids`/`cids`/
        # `pids` per lane, so it serves any topology the compiler produces.
        tri_calls = []
        if not calls:
            import triton
            for partition_id in range(layer.num_fw_partitions):
                nids = layer.partitioned_nids[partition_id]
                cids = layer.partitioned_cids[partition_id]
                pids = layer.partitioned_pids[partition_id]
                gate = layer.ext_slots[0][partition_id]
                ne = cids.size(1)

                tile_m = min(block_size, 16 if block_size >= 16 else block_size)
                # CAPPED at the gate: an M tile that spanned two node gates would need `phi` to vary
                # with the matmul's output row, which is the one thing that stops it factoring out of
                # the contraction. Inside one gate it stays a [K, B] fold and costs nothing extra.
                tile_m = min(tile_m, gate_bs)
                tile_k = min(ne, 64)
                blk_b = min(triton.next_power_of_2(batch_size), 64)
                tl_dot = 1 if (tile_m >= 16 and tile_k >= 16 and blk_b >= 16) else 0
                tri_calls.append(dict(
                    nids = nids, cids = cids.to(torch.int64), pids = pids.to(torch.int64),
                    gate = gate, log_z = log_zs[partition_id],
                    gate_nstride = self._gate_nstride(layer, nids),
                    num_edges = ne, BLOCK_B = blk_b, TILE_SIZE_K = tile_k,
                    K_NUM_TILES = triton.cdiv(ne, tile_k), TILE_SIZE_M = tile_m,
                    BLOCK_SIZE_M = block_size, TL_DOT = tl_dot,
                    GATE_BS = gate_bs, N_NODE_GATES = block_size // gate_bs,
                    grid = (triton.cdiv(batch_size, blk_b),
                            nids.size(0) * (block_size // tile_m)),
                ))

            if not sb_calls:
                layer._bs_bw_state = {
                    "block_size": block_size, "batch_size": batch_size, "ext_base": ext_base,
                    "gate_cbs": gate_cbs, "node_cbs": node_cbs, "shift_args": shift_args,
                }
                return "triton", "fw", tri_calls

        if False:
            raise NotImplementedError(
                f"no block-scale forward applies to this layer: block_size={block_size}, "
                f"batch={batch_size}, gate ch_block_size={gate_cbs}, CuTe tiles="
                f"{[(c[0], c[1]) for c in cfgs]}. The CuTe forward needs `block_size % BM == 0`, "
                f"`batch % BN == 0`, `num_edges % 64 == 0` and enough shared memory; the small-batch "
                f"forward needs `block_size % 32 == 0`. Both need each row's children contiguous and "
                f"its parameters strided by `block_size`."
            )

        # ---- PICK THE KERNEL AND ITS CONFIG BY MEASURING THEM ----
        #
        # No rule gets this right, for either choice. Between the two kernels: the CuTe fork wins
        # comfortably at large batch, but its cost is a serial walk over each row's edge tiles, so as
        # the batch shrinks towards its 64-wide tile it stops being obviously better than the
        # small-batch kernel, which tiles the node axis instead. Within the CuTe fork: a narrower tile
        # gives more blocks, which a latency-bound kernel wants, but the accumulator fragment is
        # `BM * BN / threads` registers and the fork carries three of them, so a tile with twice the
        # WARPS halves the fragment and doubles the warps resident per SM. Those pull opposite ways and
        # which wins flips with the batch size.
        #
        # Safe to run on the LIVE buffers because every write is an assignment: `node_mars` and
        # `log_z` are overwritten, never accumulated, so a repeated trial leaves exactly what one
        # run would. `forward_layer` then does the real launch.
        #
        # A read-accumulate-write kernel must NOT be tuned this way -- each trial would add its
        # contribution again. The backward is exactly that, so when it is autotuned it has to run into
        # a scratch clone, as `sum_layer`'s param-flow and element-flow tuners already do.
        def _with_cfg(argslist, i):
            return [tuple(a[:-1]) + (i,) for a in argslist]

        # Each trial launches the layer REPEATS times. The candidates differ by a few microseconds and
        # a single launch spends comparable time in Python and in the driver, so timing one launch
        # measures mostly that: with eight candidates the tuner mis-picked three of eight shapes,
        # costing 29% on the worst. Repeating inside the timed region amortizes the overhead away, and
        # is only safe because every launch here is idempotent (see the note above).
        REPEATS = 10

        def _runner(fn, argslist, i, p):
            tuned = _with_cfg(argslist, i)

            def run():
                for _ in range(REPEATS):
                    for args in tuned:
                        fn(node_mars, element_mars, p, external_params, *args)

            return run

        def _tri_runner(tcalls):
            from ..kernels.blockscale_forward import _bs_triton_fw_kernel

            def run():
                for _ in range(REPEATS):
                    for c in tcalls:
                        _bs_triton_fw_kernel[c["grid"]](
                            node_mars = node_mars, element_mars = element_mars, mparams = params,
                            ext = external_params, gate = c["gate"], log_z = c["log_z"],
                            nids = c["nids"], cids = c["cids"], pids = c["pids"],
                            batch_size = batch_size, num_edges = c["num_edges"],
                            BLOCK_B = c["BLOCK_B"], TILE_SIZE_K = c["TILE_SIZE_K"],
                            K_NUM_TILES = c["K_NUM_TILES"], TILE_SIZE_M = c["TILE_SIZE_M"],
                            BLOCK_SIZE_M = c["BLOCK_SIZE_M"], TL_DOT = c["TL_DOT"],
                            NODE_CBS = node_cbs, GATE_CBS = gate_cbs,
                            gate_nstride = c["gate_nstride"], GATE_BS = c["GATE_BS"],
                            N_NODE_GATES = c["N_NODE_GATES"],
                            gate_stride = c["gate"].size(1), ext_base = ext_base, num_stages = 1)
            return run

        def _stream_runner(code):
            tuned = _with_cfg(calls, code)

            def run():
                for _ in range(REPEATS):
                    for pid, args in enumerate(tuned):
                        stream_mod.blockscale_forward_stream(
                            node_mars, element_mars, p_bf, external_params, *args,
                            self._sigma(layer, p_bf, pid, block_size, gate_cbs)[0])
            return run

        cands = []
        if calls:
            for i in valid:
                cands.append((("cute", i), _runner(mod.blockscale_forward, calls, i, p_bf)))
            # The pre-pass copy takes the same launch tuples, and writes `node_mars` and `log Z` by assignment too.
            for i in pre_valid:
                cands.append((("pre", i), _runner(pre_mod.blockscale_forward_prepass, calls, i, p_bf)))
            # The split forward: the same tuples and each partition's `sigma`, by assignment too; one candidate, its
            # (tile, split) measured first.
            if stream_valid:
                code = self._stream_code(layer, calls, stream_mod, stream_valid, node_mars, element_mars, p_bf,
                                         external_params)
                cands.append((("stream", code), _stream_runner(code)))
        if sb_calls:
            for i in range(len(sb_mod.blockscale_sb_configs())):
                cands.append((("sb", i), _runner(sb_mod.blockscale_sb_forward, sb_calls, i, params)))
            # The batch-tiled copy of that kernel (the parameters loaded once per BT columns): the same launch tuples.
            # It wins from a few samples on (gated HMM-4096: batch 4 54 us a layer -> ~25, batch 16 Triton's 174 -> ~50).
            sbbt_mod = get_sb_bt_module()
            if sbbt_mod is not None:
                for i in range(len(sbbt_mod.blockscale_sb_bt_configs())):
                    cands.append((("sbbt", i), _runner(sbbt_mod.blockscale_sb_forward_bt, sb_calls, i, params)))
            # The tensor-core copy (blockscale_smallbatch_forward_mma.cu: N and Z both products of the parameter tile,
            # split-K ranges merged by a dependent launch), its (config, split) codes common to every partition, the
            # same launch tuples; below fp32 its 16-byte parameter loads need the edge blocks' first parameters at
            # multiples of 8 (as the CuTe fork's). Below batch 16 its high-precision form (~2^-16 a term).
            sbmma_mod = get_sb_mma_module()
            if sbmma_mod is not None and (p_bf.dtype == torch.float32
                                          or (block_size % 8 == 0 and all(bool((c[2] % 8 == 0).all()) for c in sb_calls))):
                codes = None
                for c in sb_calls:
                    arms = ck_ext.blockscale_sb_mma_arms(block_size, batch_size, c[6], c[0].numel(), node_cbs, bool(c[10]),
                                                         p_bf.dtype == torch.float32)
                    codes = list(arms) if codes is None else [a for a in codes if a in arms]
                for code in codes or []:
                    cands.append((("sbmma", code), _runner(sbmma_mod.blockscale_sb_forward_mma, sb_calls, code, p_bf)))

        # The portable fork is a CANDIDATE only where the CuTe fork is ineligible -- which is where
        # the small-batch one otherwise takes shapes it was never meant for. Its grid is one block per
        # (node group, SAMPLE), so the parameters are re-read once per sample: at `block_size = 32`,
        # where `block_size % BM == 0` rules the CuTe fork out for every BM >= 64, that made a
        # K=1024 / batch-512 forward 455us against Triton's 158us. Measured rather than ruled, since
        # below batch ~64 the small-batch fork is still the better of the two.
        #
        # Deliberately NOT offered where the CuTe fork applies: those shapes keep exactly the
        # candidates and the pick they had, and never pay to compile a kernel they will not use.
        if tri_calls and not calls:
            cands.append((("tri", 0), _tri_runner(tri_calls)))

        best = ck.autotune(cands, warmup = 2, reps = 9) or cands[0][0]
        kind, best_cfg = best

        if kind == "tri":
            layer._bs_bw_state = {
                "block_size": block_size, "batch_size": batch_size, "ext_base": ext_base,
                "gate_cbs": gate_cbs, "node_cbs": node_cbs, "shift_args": shift_args,
            }
            return "triton", "fw", tri_calls

        if kind == "cute":
            out_mod, fname, calls = mod, "blockscale_forward", _with_cfg(calls, best_cfg)
        elif kind == "pre":
            out_mod, fname, calls = pre_mod, "blockscale_forward_prepass", _with_cfg(calls, best_cfg)
        elif kind == "stream":
            out_mod, fname, calls = stream_mod, "blockscale_forward_stream", _with_cfg(calls, best_cfg)
        elif kind == "sbbt":
            out_mod, fname, calls = get_sb_bt_module(), "blockscale_sb_forward_bt", _with_cfg(sb_calls, best_cfg)
        elif kind == "sbmma":
            out_mod, fname, calls = get_sb_mma_module(), "blockscale_sb_forward_mma", _with_cfg(sb_calls, best_cfg)
        else:
            out_mod, fname, calls = sb_mod, "blockscale_sb_forward", _with_cfg(sb_calls, best_cfg)

        # Everything the backward needs that only the forward knows: where this layer's gates start in
        # the staging buffer, the gate geometry the kernels specialize on, and the per-partition `log Z`.
        layer._bs_bw_state = {
            "block_size": block_size,
            "batch_size": batch_size,
            "ext_base": ext_base,
            "gate_cbs": gate_cbs,
            "node_cbs": node_cbs,
            "shift_args": shift_args,
            # `log Z` summed from the copy's `sigma` (the split forward on it): the gate gradient's `sigma` from it too
            "z_from_copy": kind == "stream" and op is not None,
        }

        return out_mod, fname, calls

    def _stream_code(self, layer, calls, stream_mod, valid, node_mars, element_mars, params, external_params):
        """
        The split forward's `config id * 16 + split` for this layer's shape: the rule's -- the first fitting tile, the
        most splits that keep its grid within one block an SM -- or another (tile, split) where it measures faster
        (`autotune.pick`, keyed by shape: equal layers share it), timed on the live buffers (it writes `node_mars` and
        `log Z` by assignment).
        """
        from pyjuice.layer.kernels import autotune
        from .denominators import _sm_count

        batch = node_mars.size(1)
        bs_, node_cbs, gate_cbs = calls[0][5], calls[0][7], calls[0][8]
        cfgs = [tuple(int(v) for v in c) for c in stream_mod.configs()]
        sms = _sm_count(node_mars.device)
        cands = []
        for i in valid:
            bm, bn, kt = cfgs[i]
            tiles = sum((batch // bn) * c[0].numel() * (bs_ // bm) for c in calls)
            kmin = min(c[6] // kt for c in calls)
            splits = [s for s in (1, 2, 3, 4, 5, 6, 8) if s <= kmin and (s == 1 or s * tiles <= 8 * sms)]
            fit = [s for s in splits if s * tiles <= sms] or splits[:1]
            cands += [i * 16 + s for s in [fit[-1]] + [s for s in splits if s != fit[-1]]]
        if len(cands) == 1:
            return cands[0]
        sig = [self._sigma(layer, params, pid, bs_, gate_cbs)[0] for pid in range(len(calls))]

        def bench(code):
            tuned = [tuple(a[:-1]) + (code,) for a in calls]
            for _ in range(3):
                for args, s in zip(tuned, sig):
                    stream_mod.blockscale_forward_stream(node_mars, element_mars, params, external_params, *args, s)

        key = ("bs_fwd_stream", bs_, batch, node_cbs, gate_cbs, tuple((c[0].numel(), c[6]) for c in calls),
               tuple(cands))
        return autotune.pick(key, cands, bench)

    def stream_fusion_plan(self, layer, fast_key: tuple, children) -> Optional[tuple]:
        """The split forward's `cbase` per partition (`blockscale_forward_stream_fused.cu`) for a gated forward of
        ``fast_key``'s shape (`SumLayer.forward`'s: the batch, the parameters' dtype, the precision, the bf16 copy
        given) over the product group below (``children``), or None where its settled fork is not the split forward
        (or none settled yet): what `ExternalParamsSumLayer._stream_fusion` asks for the circuit's decision."""
        batch, _, _, op = fast_key
        entry = layer.__dict__.get("_bs_fw_plans", {}).get(batch if not op else (batch, "operands"))
        if entry is None:
            return None
        _, fname, calls = entry[0]
        return self._fused_cbases(layer, batch if not op else (batch, "operands"), fname, calls, children)

    def _fused_cbases(self, layer, pkey, fname, calls, children) -> Optional[tuple]:
        """Per partition, its edge tiles' chunks as runs of the product's children (`SumLayer._stream_fusion_cbase` of
        the split forward's tile starts, `cids`), for the fused split forward -- or None where the fork is another or
        a chunk is neither dummy rows nor runs. Kept with the plan's calls and the table they were made from."""
        if fname != "blockscale_forward_stream" or _stream_fused_module() is None:
            return None
        table, first_row = children
        if table.size(1) != 2:
            return None
        plans = layer.__dict__.setdefault("_stream_fusion_plans", {})
        hit = plans.get(("blockscale", pkey))
        if hit is not None and hit[0] is table and hit[1] is calls:
            return hit[2]
        from pyjuice.layer.sum_layer import SumLayer
        cbases = []
        for args in calls:
            cb = SumLayer._stream_fusion_cbase(args[1], table, first_row)      # (args[1]: `cids`, the tile starts)
            if cb is None:
                cbases = None
                break
            cbases.append(cb)
        plans[("blockscale", pkey)] = (table, calls, None if cbases is None else tuple(cbases))
        return plans[("blockscale", pkey)][2]

    def forward_layer(self, layer, ns_tensors, node_mars, element_mars, params, **kwargs) -> None:
        """
        Compute this layer's node values under the gate, REPLACING the standard sum forward.

        Unlike a corrective parameterization this owns the whole computation: it reweights the
        per-edge-block partial sums, and those do not survive the standard kernel.

        `op_params` (keyword): the circuit's bf16 copy of tied fp32 parameters (`reads_param_operands`), which the forks
        whose products are bf16 read in place of `params` (`_BF16_FORKS`: they round the parameters to bf16 anyway --
        the same bits) and with which the split forward (bf16 parameters only) joins them, its `sigma` summed from the
        copy: `log Z` from the rounded parameters' sums keeps the cancellation of their rounding between `N` and `Z`.
        A plan of its own (the candidates differ); the fp32 forks and the portable Triton fallback keep `params`.
        """
        op_params = kwargs.pop("op_params", None)
        # (the product layer group below, its forward not run: `SumLayer.forward`'s `stream_children` -- fused into
        # the split forward's pre-pass where that is the settled fork, else its values formed for the fork)
        stream_children = kwargs.pop("stream_children", None)
        if len(ns_tensors) == 0:
            return None

        external_params = kwargs.get(_buffer_kwarg(), None)
        if external_params is None:
            if stream_children is not None:
                form_elements(element_mars, node_mars, stream_children)
            return None

        # Keyed by SHAPE, and a dict. This was one entry keyed partly on data POINTERS, which was wrong
        # twice over: alternating between two batch sizes missed on every call, and any reallocation --
        # a batch resize, allocator churn between PC calls, `empty_cache` -- threw away a plan that was
        # still valid. Neither is cheap to get wrong, because `_build_plan` AUTOTUNES: ~66 ms a layer,
        # so a caller whose batch varies step to step paid ~2 s per step across 31 tied layers, and a
        # gated run came out 2.3x an ungated one whose steady-state cost is nil.
        #
        # Nothing in a plan is pointer-dependent. The four buffers are passed at launch, and `ext_base`
        # is an offset RELATIVE to the staging buffer, so it survives the whole thing moving. It is
        # re-derived and compared anyway -- a pointer subtraction against a rebuild that autotunes.
        batch = node_mars.size(1)
        pkey = batch if op_params is None else (batch, "operands")
        plans = layer.__dict__.setdefault("_bs_fw_plans", OrderedDict())
        entry = plans.get(pkey)
        if entry is not None:
            plans.move_to_end(pkey)           # LRU: a hit has to protect the entry, or it cannot help
        if entry is not None and entry[1]["ext_base"] != self._ext_base(ns_tensors, external_params,
                                                                       batch):
            entry = None
        if entry is None:
            if stream_children is not None:              # (the plan's trials read the element values)
                form_elements(element_mars, node_mars, stream_children)
                stream_children = None
            built = self._build_plan(layer, ns_tensors, node_mars, element_mars, params,
                                     external_params, op_params)
            # `_build_plan` also allocates a `log Z` buffer per partition, sized by batch, so an
            # unbounded cache would hold one set per batch size ever seen. Oldest out first.
            while len(plans) >= _FW_PLAN_CACHE:
                plans.popitem(last = False)
            entry = plans[pkey] = (built, layer._bs_bw_state)

        mod, fname, calls = entry[0]
        # The backward reads this, and on a cache HIT `_build_plan` did not run to set it.
        layer._bs_bw_state = entry[1]
        cbases = None
        if stream_children is not None:
            cbases = self._fused_cbases(layer, pkey, fname, calls, stream_children)
            if cbases is None:
                form_elements(element_mars, node_mars, stream_children)

        if mod == "triton":
            # The portable fallback: `calls` are kwarg dicts rather than positional tuples, since
            # this kernel takes its tiling as constexprs rather than a config id.
            from ..kernels.blockscale_forward import _bs_triton_fw_kernel
            for c in calls:
                _bs_triton_fw_kernel[c["grid"]](
                    node_mars = node_mars, element_mars = element_mars, mparams = params,
                    ext = external_params, gate = c["gate"], log_z = c["log_z"],
                    nids = c["nids"], cids = c["cids"], pids = c["pids"],
                    batch_size = node_mars.size(1), num_edges = c["num_edges"],
                    BLOCK_B = c["BLOCK_B"], TILE_SIZE_K = c["TILE_SIZE_K"],
                    K_NUM_TILES = c["K_NUM_TILES"], TILE_SIZE_M = c["TILE_SIZE_M"],
                    BLOCK_SIZE_M = c["BLOCK_SIZE_M"], TL_DOT = c["TL_DOT"],
                    NODE_CBS = layer._bs_bw_state["node_cbs"],
                    GATE_CBS = layer._bs_bw_state["gate_cbs"],
                    gate_nstride = c["gate_nstride"], GATE_BS = c["GATE_BS"],
                    N_NODE_GATES = c["N_NODE_GATES"],
                    gate_stride = c["gate"].size(1),
                    ext_base = layer._bs_bw_state["ext_base"], num_stages = 1)
            return None

        fn = getattr(mod, fname)
        p = op_params if op_params is not None and fname in _BF16_FORKS else params
        if fname == "blockscale_forward_stream":
            # and each partition's `sigma` (cached until the parameters change); the product layer below fused into
            # the pre-pass where the call was handed it (`cbases`)
            st = layer._bs_bw_state
            ffn = None if cbases is None else _stream_fused_module().blockscale_forward_stream_fused
            for pid, args in enumerate(calls):
                sigma = self._sigma(layer, p, pid, st["block_size"], st["gate_cbs"])[0]
                if ffn is None:
                    fn(node_mars, element_mars, p, external_params, *args, sigma)
                else:
                    ffn(node_mars, cbases[pid], p, external_params, *args, sigma)
            return None
        for args in calls:
            fn(node_mars, element_mars, p, external_params, *args)

        return None
