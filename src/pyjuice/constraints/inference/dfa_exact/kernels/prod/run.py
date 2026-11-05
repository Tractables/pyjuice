"""
The launches of a query's product steps: their tiles, the tuner's choices among them, what each kernel is told.
"""

from pyjuice.layer.kernels import autotune
from pyjuice.utils.util import host_cdiv, host_next_power_of_2
from ..common import grid_chunks
from ..common import pdl_launch
from ..transitions import enumerated_args
from .block_block import EXACT_BELOW, _block_block_compact_kernel, _block_block_kernel, _copy_kernel, compact_tiles
from .input_block import _block_input_kernel, _input_block_group_kernel, _input_block_kernel


#: :func:`_block_block_kernel`'s ``(output tile entry x exit columns, shared chunk, warps)`` candidates, the first the
#: default, the others timed by the autotuner per launch shape (:mod:`pyjuice.layer.kernels.autotune`); every tile a
#: multiple of :class:`SkipMasks`'s, every chunk equal to it. The tile sets the dots' accumulation order, so
#: candidates agree to fp32 rounding (~1e-7 relative on a 16-latent PD), as the autotuner's contract allows. PD16,
#: tuned against the default: x1.03-1.22 (16 x 16 / 2 warps mostly)
BLOCK_BLOCK_TILES = [(16, 32, 16, 4), (16, 16, 16, 2), (32, 32, 16, 4), (16, 32, 16, 2)]
#: the products' dot precision: "ieee" (exact fp32, CUDA cores) or "tf32x3" (three TF32 tensor-core passes)
PRODUCT_PRECISION = "ieee"
#: skip the (output tile, shared chunk) pairs the automaton cannot join; False computes every pair
SKIP_EMPTY_TILES = True
#: a ``block @ block`` launch with fewer (step, sample) pairs than this many per SM spreads each pair's output tiles
#: over several programs (a tile's result does not depend on which program computes it). 16 from a sweep: a
#: 600-state automaton x30 at batch 1, x8 at 4; Ctrl-G's on a 16-latent PD unchanged (more loses there: cheap tiles
#: lose their operands' L1 reuse)
BLOCK_BLOCK_PROGRAMS_PER_SM = 16


def _tile(cols: int) -> int:
    return min(256, max(16, host_next_power_of_2(cols)))


def _flat_tile(lanes: int) -> int:
    """Lanes per program of a step whose output is flattened (sample, entry column, exit column)."""
    return min(512, max(128, host_next_power_of_2(lanes) // 8))


#: An ``input @ block`` launch runs its steps in groups (see :func:`_input_block_group_kernel`) only when a step's
#: output has more lanes than this (batch x slots): a step that fits one program shares little, and its launch would
#: have G times fewer programs (HMM4096 with 67- and 101-state automata at batch 1: 1-3% slower grouped). A group
#: shares its steps' lanes' samples (a search over the batch's offsets a lane): Codpula-4096 x JSON hard_0 B64
#: diffusion (~77 lanes a step) 2129 -> 2002 ms grouped; B1 / B8 / B32 the same either way
GROUP_MIN_LANES = 64
#: (step, lane) pairs a program of :func:`_input_block_group_kernel` takes: G steps of GROUP_TILE // G lanes (or a step's
#: lanes, fewer), a pair a thread. Codpula-4096 x JSON hard_0 B64 diffusion (G 16 chosen: 64 lanes a program at 1024,
#: 168 registers a thread, 0.91 waves) 2036 -> 1949 ms at 128; 512 / 256 / 64 / 32: 1992 / 1982 / 2010 / 1994
GROUP_TILE = 128
#: The log-sum-exp steps the autotuner times for :func:`_input_block_group_kernel` (with every group size the
#: Program built): two exps (the default), and :func:`_lse_step_one_exp` -- bit-identical; x1.115 on the kernel with
#: most lanes live (one observed token), x0.91-0.97 under evidence pruning.
INPUT_BLOCK_ONE_EXP = (False, True)


def run_products(stages, bufs, prog, regions, element_mars, node_mars, masses, obs_class, live,
                 elem_first: int, elem_width: int, B: int, prior: dict, exact: bool, compact = None,
                 succ_lists = None, hold = None, sizes = ()):
    """
    The steps a query plan runs of one product layer, stage by stage (a stage reads only earlier stages' scratch
    rows). Products are fp32-level whatever the query's precision, which applies to the sum layers: ``block @ block``
    dots in :data:`PRODUCT_PRECISION` (exact fp32 by default), the other contractions exact fp32 log-sum-exps.

    :param sizes: the plan's sizes (:attr:`Cut.at`: of this automaton)
    :param stages: the layer's stages: lists of ``(launch, cut)`` -- a :class:`Launch` and the :class:`Cut` of its
        steps the plan runs. ``block @ block`` steps: ``(steps, triple index)``, the latter indexing ``Program.skip``;
        ``input @ block`` and ``block @ input`` steps: ``(steps, chunks, groups)``, chunks ``(first, end, None)`` step
        ranges (one, every step's transition masses being at hand: the Program keeps them all when the transitions
        are grouped; the field and the plans' chunk tables date from building them per chunk); ``groups`` None (one
        step per program), or ``{G: firsts}`` for :func:`_input_block_group_kernel`, used above
        :data:`GROUP_MIN_LANES`
    :param bufs: ``(temp, temp_width)``, the layer's scratch rows
    :param masses: the query's :class:`~..state.ClassMasses` (per sample, the ``SAMPLE`` kernels)
    :param live: None, or the query's liveness (:class:`~.live.Liveness`'s ``live``)
    :param prior: where the sum operands the plan keeps once are (see :func:`_block_ptr`): ``mars``, the regions'
        ``offset`` and ``width`` there, and ``reg_prior`` (device tensors)
    :param exact: whether ``block @ block`` tiles whose entries its shifts flush are computed again exactly (see
        :data:`EXACT_BELOW`): for the blocks kept once, every entry of which some sample may need, and under evidence
        pruning, where a sample's few live columns may carry very different masses; a batch observing nothing reads
        its blocks' entries weighted by its own masses, which the shifts respect (and pays nothing)
    :param compact: None, or the query's :class:`~.compact.CompactLayout`, by which the per-sample inside values are
        compact (``regions``, ``elem_width`` and the scratch rows' width are their rows'): a lane per pair of live
        columns (``block @ block`` steps: :func:`_block_block_compact_kernel`, ``exact`` always)
    :param succ_lists: None, or a function giving the query's :class:`~.live.SuccessorLists` (of ``Program.trans``,
        under evidence pruning), called by the first ``input @ block`` launch (a query running none builds none):
        their entry loops take them at the boundaries that do
    :param hold: None, or (compact) a function giving the query's :func:`~.compact.fresh_counts`: the ``input @ block``
        and ``block @ input`` steps writing a product's values skip the tiles of lanes of samples whose sums keep the
        values they hold (a fold's scratch rows: every lane)
    """
    temp, temp_width = bufs
    pdl = pdl_launch(element_mars.device)                # (the walks: dependents of the launch before them)
    common = dict(reg_offset = regions["offset"], reg_width = regions["width"], reg_first = regions["first"],
                  prior_mars = prior["mars"], prior_offset = prior["offset"], prior_width = prior["width"],
                  reg_prior = prior["reg_prior"], out_first = elem_first, out_width = elem_width,
                  temp_width = temp_width)
    n, W, C = prog.n, prog.W, prog.C
    ivs = prog.intervals
    prune, live = live is not None, live if live is not None else obs_class
    sample = masses.batch > 1
    inputs = dict(class_mars = masses.table, mass_row = masses.row,
                  mass_stride = masses.stride if sample else masses.row, SAMPLE = sample,
                  **enumerated_args(prog, masses))
    if compact is not None:
        lists = dict(coff = compact.off, ccount = compact.count, cstart = compact.start, ccolumns = compact.columns,
                     crank = compact.rank, csteps = compact.steps)
    else:
        lists = dict(coff = live, ccount = live, cstart = live, ccolumns = live, crank = live, csteps = 0)
    no_lists = dict(succ_start = live, succ_count = live, succ_use = live, succ_entries = live, E = 0, SUCC_LISTS = 0)

    def held(out_temp):
        """The steps' tile check (:func:`_fresh_tile`): those writing a product's values (not a fold's) under
        ``hold``."""
        if hold is None or compact is None or out_temp:
            return {}
        return dict(hold_cum = hold(), HOLD = True)

    def succ_of(launch):
        """The input @ block entry loops' successor lists: 2 every step's boundary takes them, 1 some (chosen per
        program), 0 none -- compiled in only where taken: a list loop next to the entry tables' slows that loop 10-20%
        even where no step takes it."""
        if succ_lists is None:
            return no_lists
        sl = succ_lists()
        taken = sl.boundaries[launch.input_boundaries]
        if not taken.any():
            return no_lists
        return dict(succ_start = sl.start, succ_count = sl.count, succ_use = sl.use, succ_entries = sl.entries,
                    E = sl.num_entries, SUCC_LISTS = 2 if taken.all() else 1)
    for launches in stages:
        for launch, cut in launches:
            form, kinds = launch.form, launch.kinds
            at = cut.at
            E_max, X_max, S_max, S = sizes[at], sizes[at + 1], sizes[at + 2], cut.count
            lanes = B * S_max if compact is None else compact.lanes(launch.out_ivs)   # an output's most
            if compact is not None and lanes == 0:                           # no sample has a live pair there
                continue
            if form in ("input_block", "block_input"):
                steps, _, groups = launch.steps
                margs, tt = prog.mass_args(), prog.trans
                succ = succ_of(launch) if form == "input_block" else no_lists
                for s0, s1, _, sel, cnt in cut.chunks:
                    part = steps[s0:s1]
                    idx = dict(sel = part if sel is None else sel.tensor, INDEXED = sel is not None)
                    if form == "input_block" and not tt.grouped and groups is not None and lanes > GROUP_MIN_LANES:
                        r_kind, out_temp, (out_full, *r_full) = kinds
                        assert (s0, s1) == (0, launch.rows.size(0))   # groups index the launch's one chunk

                        def input_block_group(cfg):
                            G, one_exp = cfg
                            firsts = groups[G]
                            gsel, gcnt = cut.groups[G]
                            G = host_next_power_of_2(G)            # lanes per group (a group may be shorter)
                            tile = min(_flat_tile(lanes), GROUP_TILE // G)
                            for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                                _input_block_group_kernel[(gcnt, size)](
                                    element_mars, node_mars, temp, obs_class, prog.next_col_arg, prog.delta, prog.state_id,
                                    prog.col_of_state, entry_ptr = tt.entry_ptr,
                                    entry_succ = tt.entry_succ, entry_x = tt.entry_x,
                                    width = prog.width_t, iv_of = ivs.of, iv_info = ivs.info, iv_table = ivs.table,
                                    live = live, steps = part, groups = firsts,
                                    sel = firsts if gsel is None else gsel.tensor, B = B, n = n, W = W, C = C,
                                    K = prog.K, input_start = prog.input_start, pid0 = first, HOMOGENEOUS = prog.homogeneous, R_KIND = r_kind,
                                    OUT_TEMP = out_temp, PRUNE = prune, OUT_FULL = out_full, R_FULL = all(r_full),
                                    G = G, ONE_EXP = one_exp, COMPACT = compact is not None,
                                    INDEXED = gsel is not None, ENTRIES = prog.entry_loop, TILE = tile,
                                    PDL = pdl, launch_pdl = pdl, num_warps = 4, **inputs, **lists, **common, **succ,
                                    **held(out_temp))

                        # the group sizes the Program built (its default first), each step form: timed per shape (the
                        # steps rounded up to a power of 2, so that queries running subsets of a launch share choices)
                        cands = [(G, one_exp) for one_exp in INPUT_BLOCK_ONE_EXP for G in groups]
                        key = ("lifted_input_block_group", kinds, prune, compact is not None, sample, C, prog.homogeneous,
                               host_next_power_of_2(S), host_next_power_of_2(lanes), tuple(groups),
                               succ["SUCC_LISTS"], bool(held(out_temp)))
                        input_block_group(autotune.pick(key, cands, input_block_group))   # outputs overwritten
                    elif form == "input_block":
                        r_kind, out_temp, (out_full, *r_full) = kinds
                        tile = _flat_tile(lanes)
                        for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                            _input_block_kernel[(cnt, size)](
                                element_mars, node_mars, temp, obs_class, prog.next_col_arg, prog.delta, prog.state_id,
                                    prog.col_of_state, **margs, entry_ptr = tt.entry_ptr, entry_succ = tt.entry_succ,
                                entry_x = prog.mass_x("entry_x"), width = prog.width_t, iv_of = ivs.of, iv_info = ivs.info,
                                iv_table = ivs.table, live = live, steps = part, B = B, n = n, W = W,
                                C = C, K = prog.K, input_start = prog.input_start, pid0 = first, HOMOGENEOUS = prog.homogeneous, R_KIND = r_kind,
                                OUT_TEMP = out_temp, GROUPED = tt.grouped, PRUNE = prune, OUT_FULL = out_full,
                                R_FULL = all(r_full), COMPACT = compact is not None, ENTRIES = prog.entry_loop,
                                TILE = tile, PDL = pdl, launch_pdl = pdl, num_warps = 4, **idx, **inputs, **lists,
                                **common, **succ,
                                **held(out_temp))
                    else:
                        l_kind, out_temp, (out_full, l_full) = kinds
                        tile = _flat_tile(lanes)
                        for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                            _block_input_kernel[(cnt, size)](
                                element_mars, node_mars, temp, obs_class, prog.next_col_arg, prog.delta, prog.state_id,
                                    prog.col_of_state, **margs, pred_ptr = tt.pred_ptr, pred_col = tt.pred_col,
                                pred_x = prog.mass_x("pred_x"), width = prog.width_t, iv_of = ivs.of, iv_info = ivs.info,
                                iv_table = ivs.table, live = live, steps = part, B = B, n = n, W = W,
                                C = C, K = prog.K, input_start = prog.input_start, pid0 = first, HOMOGENEOUS = prog.homogeneous, L_KIND = l_kind,
                                OUT_TEMP = out_temp, GROUPED = tt.grouped, PRUNE = prune, OUT_FULL = out_full,
                                L_FULL = l_full, COMPACT = compact is not None, TILE = tile, PDL = pdl,
                                launch_pdl = pdl, num_warps = 4, **idx,
                                **inputs, **lists, **common, **held(out_temp))
            elif form == "block_block" and compact is not None:     # a sample's rectangle at a time
                l_kind, r_kind, out_temp = kinds
                steps, sel = launch.steps[0], cut.chunks[0][3]
                ends = prog.iv_ends_np[launch.out_ivs]
                rmax, cmax = int(compact.max_count[ends[:, 0]].max()), int(compact.max_count[ends[:, 1]].max())
                TM, TN, TK, warps = compact_tiles(rmax, cmax)
                tiles = host_cdiv(rmax, TM) * host_cdiv(cmax, TN)
                split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, S * B))
                group = host_cdiv(tiles, max(1, split))
                split = host_cdiv(tiles, group)
                for first, size in grid_chunks(S * B * split, 0):
                    _block_block_compact_kernel[(size,)](
                        element_mars, node_mars, temp, prog.width_t, ivs.of, ivs.info, ivs.table, compact.off,
                        compact.count, compact.start, compact.columns, steps, steps if sel is None else sel.tensor,
                        B = B, n = n, SPLIT = split, GROUP = group, exact_below = EXACT_BELOW, pid0 = first,
                        L_KIND = l_kind, R_KIND = r_kind, OUT_TEMP = out_temp, TM = TM, TN = TN, TK = TK,
                        PRECISION = PRODUCT_PRECISION, INDEXED = sel is not None, num_warps = warps, num_stages = 1,
                        **common)
            elif form == "block_block":
                l_kind, r_kind, out_temp = kinds
                steps, triples = launch.steps
                sel = cut.chunks[0][3]
                skip = prog.skip

                def block_block(cfg):
                    TM, TN, TK, warps = cfg
                    assert TK == skip.tk and TM % skip.tm == 0 and TN % skip.tn == 0
                    tiles = host_cdiv(E_max, TM) * host_cdiv(X_max, TN)
                    split = min(tiles, host_cdiv(BLOCK_BLOCK_PROGRAMS_PER_SM * prog.num_sms, S * B))
                    group = host_cdiv(tiles, max(1, split))
                    split = host_cdiv(tiles, group)                     # no group past the widest step's tiles
                    for first, size in grid_chunks(S * B * split, 0):
                        _block_block_kernel[(size,)](
                            element_mars, node_mars, temp, prog.width_t, ivs.of, ivs.info, ivs.table, live, steps,
                            steps if sel is None else sel.tensor, triples, skip.bits, B = B, n = n, W = W,
                            TILES_I = skip.tiles_i, TILES_J = skip.tiles_j, WORDS = skip.words, SPLIT = split,
                            GROUP = group, exact_below = EXACT_BELOW, pid0 = first, L_KIND = l_kind, R_KIND = r_kind,
                            OUT_TEMP = out_temp, TM = TM, TN = TN, TK = TK, SUB_I = TM // skip.tm, SUB_J = TN // skip.tn,
                            SKIP = SKIP_EMPTY_TILES, PRUNE = prune, EXACT = exact, PRECISION = PRODUCT_PRECISION,
                            INDEXED = sel is not None, num_warps = warps, num_stages = 1, **common)

                key = ("lifted_block_block", kinds, prune, exact, SKIP_EMPTY_TILES, PRODUCT_PRECISION, B,
                       host_next_power_of_2(S), E_max, X_max)
                block_block(autotune.pick(key, BLOCK_BLOCK_TILES, block_block))   # outputs overwritten: timed in place
            else:                                                            # "copy"
                l_kind, out_temp = kinds
                sel = cut.chunks[0][3]
                tile = _tile(lanes)
                for first, size in grid_chunks(host_cdiv(lanes, tile), 1):
                    _copy_kernel[(S, size)](
                        element_mars, node_mars, temp, ivs.of, ivs.info, lists["coff"], launch.steps,
                        launch.steps if sel is None else sel.tensor, B = B, n = n, pid0 = first, L_KIND = l_kind,
                        OUT_TEMP = out_temp, COMPACT = compact is not None, INDEXED = sel is not None, TILE = tile,
                        num_warps = 4, **common)
