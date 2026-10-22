"""
What the forward pass reads besides the buffers and the parameters, built once per backend (:class:`Program`): every
product layer's launches -- each product folded into binary steps, the steps of one contraction launched together
(:mod:`.prod`) -- and every sum layer's dense runs and fused parts (:mod:`.sum`), with the interval,
transition and skip tables they read. A query runs the subsets of them its plan selects (:mod:`.plans`).
"""

from __future__ import annotations

import bisect
from collections import OrderedDict, defaultdict
from typing import Optional, Tuple

import numpy as np
import torch

from ..kernels.common import IDENTITY, SUM, TEMP
from ..kernels.prod import Launch, skip_masks
from ..kernels.sum import dense_groups
from ..kernels.live import live_successors
from ..kernels.transitions import (SampleMasses, enumerated_tokens, sample_classes, sample_masses,
                                   sample_transition_masses, transition_job, transition_masses, transition_stats,
                                   transition_tables)
from ..external_params import block_scale
from .layout import interval_tables


#: Read a one-child product over a sum straight from the sum's row instead of copying its block (see Program)
ALIAS_COPIES = True

#: A missing token's transitions are grouped by successor (see :class:`~.transitions.TransitionTables`) when the
#: boundaries' live (column, class) pairs are at least this many times their (column, successor) pairs -- live token
#: classes per successor, over every column of every boundary (:func:`grouped_worth`; and a query's masses must fit,
#: :func:`transition_budget`): the products then loop over successors, not classes, through transition masses built
#: per query. Below it, they loop over the live classes, and nothing more is built. 8 from a sweep on HMMs (grouped
#: against class by class, batch 1 / 16; every class live in every column, every column branching alike, so the
#: ratio over all columns is each column's): 512 classes, 16 successors x1.6-1.8 / x2.7-3.4; 128 and 16 x1.0-1.2 /
#: x1.4-1.5; at 4 classes per successor and below, nothing gained or losses (64 successors: 256 classes
#: x0.71-0.73 / x1.24, 64 classes x0.81-0.87). The ratio was compared against the MOST branching column before
#: (2026-10-21): JSON Schema automata, 30-47 live classes per column against 2-3 successors on average but one
#: column of 67-85, were never grouped; grouped, hard_0 x HMM 4096 decodes by diffusion 3.2x faster.
GROUP_MIN_RATIO = 8

#: ``input @ block`` steps at the same position and block interval run this many to a program on the class-by-class
#: path, when the block is not the identity (see :func:`~.prod._input_block_group_kernel`; launches wider than
#: :data:`~.prod.GROUP_MIN_LANES`): every class's successors and slots are read once for all of them, not once
#: per step. HMM4096 sweep over 1, 2, 4, 8, 16: 8 best or within 5% of it on every workload; against one step per
#: program, bit-identical: 13.8k- / 49k-state automata x1.09-1.15 with a prefix, x1.16-1.31 on the 49k one without;
#: batch 16 x1.02-1.18. 1: one step per program.
INPUT_BLOCK_GROUP = 8
#: the other group sizes the Program builds tables for: the autotuner times every one per launch shape (see
#: :func:`~.prod.run_products`); 16 measured best on a 168-state automaton at batch 16 (x1.05 over 8)
INPUT_BLOCK_GROUP_ALTERNATIVES = (16, 4)

#: Floats of transition masses (see :func:`~.transitions.transition_masses`) a program keeps at most: every input
#: The transition table over every boundary, ``next_col`` [n, W, C] int32 (the column every class leads every column to,
#: see :class:`~.layout.BoundaryLayout`), is kept on the device while it has at most this many entries (4 GiB) and read
#: directly; past that the kernels read a transition through the automaton's ``delta`` [K, C] and the boundaries'
#: column maps (:func:`~..kernels.common._next_col`), two dependent gathers instead of one load. The maps cost the walks
#: that loop over every class of a column and read the table per lane -- those of automata of FEW classes (Ctrl-G
#: ex108, 14 classes, B8 diffusion on croissant: the class weights' sample walk +11%, the input @ block group walk +17%,
#: decode 4479 -> 4721 ms), whose tables are small: 4 GiB admits 49152 columns of 16 classes over 1024 positions
#: (ex110). Automata of many classes walk the entry lists instead, and their tables are what grows past the bound
#: (JSON hard_0 at n = 1024: 3.1e9 entries): through the maps they lose nothing -- hard_0 left to right at n = 1024
#: 10.1 -> 9.5 ms a call (the maps stay in L2, the 12 GB table streamed from memory), hard_1 B8 diffusion the same --
#: and compile in 35 s on 4.7 GB instead of 59 s on 29.6 GB; the 10x schema's 61 GB of tables go.
TRANSITION_TABLE_MAX_ENTRIES = 1 << 30

#: step's, built once per computation of the class masses. Transitions whose masses would not fit are not grouped
#: (rebuilding them query by query would read every class's mass anyway, and write besides; the forward reads the
#: classes' masses directly instead): the floats they may take, at least -- or :data:`TRANSITION_MEMORY_SHARE` of the
#: device's free memory when the program is built, when that is more (JSON hard_0 x HMM 4096 at n = 128: 1.2G floats,
#: 4.7 GB, for a diffusion decode 3.3x faster). Under a session's per-sample inputs the masses are each sample's own,
#: over the entries its inputs leave possible alone (:class:`~..kernels.transitions.SampleMasses`: few, where the
#: inputs rule most classes out), and the query takes them when they fit (see :meth:`Program.transitions`)
TRANSITION_BUDGET = 1 << 26
TRANSITION_MEMORY_SHARE = 0.25


#: Ungrouped transitions: the forward kernels loop over a column's live entries (its live classes' successors and
#: codes, see :class:`~..kernels.transitions.TransitionTables`) when at most this fraction of the automaton's
#: (column, class) pairs is live, else over every class in class order (the dead ones -inf). An entry's two loads
#: ahead of its mass load cost about 1.7 of a class's: with 90% of the pairs live (Ctrl-G ex108 / ex110, Codpula
#: 4096, batch 16) ``_input_block_group_kernel`` over entries against classes 6.6 against 4.2 ms a decode (ex108
#: l2r), 8.3 against 4.7 (ex110 l2r), 29.9 against 21.5 (ex108 diffusion); with few live (JSON hard_1, 617 classes,
#: 36 successors at most) ``_input_block_kernel`` 103 against 129 ms. The crossover is estimated from these two ends,
#: not swept.
ENTRY_LOOP_MAX_LIVE = 0.5


def grouped_worth(stats: Tuple[int, int, int], pairs: int) -> bool:
    """Whether a missing token's transitions are worth grouping by successor: the boundaries' live (column, class)
    pairs (``stats``: the layout's :func:`~..kernels.transitions.transition_stats`), what the ungrouped kernels' loops
    read, are :data:`GROUP_MIN_RATIO` times their (column, successor) ``pairs`` or more, what the grouped kernels'
    read. (Whether the masses fit is the query's question, see :meth:`Program.transitions`.)"""
    live, _, _ = stats
    return live >= GROUP_MIN_RATIO * max(pairs, 1)


def transition_budget(dev) -> int:
    """The floats of transition masses a query may keep: :data:`TRANSITION_BUDGET`, or
    :data:`TRANSITION_MEMORY_SHARE` of the device's free memory (and the blocks its allocator holds unused) when
    that is more."""
    if dev.type != "cuda":
        return TRANSITION_BUDGET
    free = torch.cuda.mem_get_info(dev)[0] + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
    return max(TRANSITION_BUDGET, int(TRANSITION_MEMORY_SHARE * free) // 4)


def entry_loop(stats: Tuple[int, int, int], C: int) -> bool:
    """Whether the ungrouped forward kernels loop over a column's live entries rather than every class: at most
    :data:`ENTRY_LOOP_MAX_LIVE` of the (column, class) pairs live (``stats`` as :func:`grouped_worth`)."""
    live, columns, _ = stats
    return live < ENTRY_LOOP_MAX_LIVE * columns * C


#: (input row, position) columns of the steps whose input's transitions they read
TRANS_COLS = {"input_block": (1, 2), "block_input": (4, 5)}

#: the step fields of their outputs' entry and exit boundaries, and how far past that exit field the output ends (a
#: ``block @ input`` step's ends after its input; see :mod:`~..kernels.prod`)
OUT_ENDS = {"input_block": (2, 5, 0), "block_input": (3, 5, 1), "block_block": (5, 7, 0), "copy": (3, 4, 0)}


class Program:
    """Everything the lifted forward pass of one backend reads besides buffers and parameters."""

    def __init__(self, backend):
        pc, layout = backend.pc, backend.layout
        dev = backend.device
        n = backend.n
        self.n = n
        self.width = [int(w) for w in layout.width.tolist()]
        self.width_t = layout.width.to(dev, torch.int32)
        self.reach = layout.reachability                               # the slots of every interval's blocks
        self.dev = dev
        self.layout = layout
        # the automaton's transitions and the boundaries' column maps (int32: what the kernels read a boundary's
        # transitions through, see Transitions; the tables below are built from them, a boundary at a time)
        tr = layout.transitions(dev)
        self.delta, self.state_id, self.col_of_state = (x.to(torch.int32).contiguous()
                                                        for x in (tr.delta, tr.state_id, tr.col_of_state))
        self.K, self.W, self.C = tr.K, tr.W, tr.C
        self.next_col = None                                             # the table over every boundary, if kept
        if n * self.W * self.C <= TRANSITION_TABLE_MAX_ENTRIES:          # (built on the device, a boundary at a time)
            self.next_col = torch.full((n, self.W, self.C), -1, dtype = torch.int32, device = dev)
            for t in range(n):
                self.next_col[t, :self.width[t]] = tr.at(t)
        self.homogeneous = self.next_col is None
        self.next_col_arg = self.delta if self.homogeneous else self.next_col    # (a pointer the kernels never read)
        self.input_start, self.input_end = backend.input_range
        self.mass_row = backend.state.class_mass_rows()[2]               # input node -> its row of the class masses
        self.token_classes = backend.token_classes
        self.enum = enumerated_tokens(tr, backend.token_classes)         # the enumerated classes' codes and tokens
        # every column's distinct successors (evidence pruning's liveness sweeps; the forward's tables when its
        # transitions are grouped: decided below, once the steps are known)
        self.succ = transition_tables(tr, grouped = True, enum = self.enum)
        self.transition_stats = transition_stats(tr)

        firsts = [first for first, _, _ in backend.sum_regions]
        self.firsts = firsts
        self.slots = [slots for _, _, slots in backend.sum_regions]
        self.reg_rows = np.asarray([end - first for first, end, _ in backend.sum_regions], dtype = np.int64)
        self.reg_first = torch.tensor(firsts, dtype = torch.long, device = dev)
        self.reg_slots = torch.tensor([slots for _, _, slots in backend.sum_regions], dtype = torch.long, device = dev)
        slots_of_reg = [slots for _, _, slots in backend.sum_regions]
        region_of = lambda row: bisect.bisect_right(firsts, row) - 1

        # every sum region's interval (its node group's scope, as boundaries), for evidence pruning
        interval_of_row = {info.ns._output_ind_range[0]: (info.scope_runs[0][0], info.scope_runs[0][1] + 1)
                           for info in backend.structure.nodes if info.kind == "sum"}
        reg_ivs = [interval_of_row[first] for first in firsts]
        self.sum_ivs = sorted(set(reg_ivs))
        self.sum_iv_slots = [self.reach.slots(a, b) for a, b in self.sum_ivs]
        self.iv_number = {iv: k for k, iv in enumerate(self.sum_ivs)}
        # for query plans (host, numpy): every sum interval's bounds and slots, and which intervals lie inside which
        self.iv_bounds = np.array(self.sum_ivs, dtype = np.int64).reshape(-1, 2)
        self.iv_slots = np.array(self.sum_iv_slots, dtype = np.int64)
        a, b = self.iv_bounds[:, 0], self.iv_bounds[:, 1]
        self.iv_inside = (a[None, :] >= a[:, None]) & (b[None, :] <= b[:, None])   # [k, m]: m inside k
        self.reg_iv_host = [self.iv_number[iv] for iv in reg_ivs]
        self.reg_iv_np = np.asarray(self.reg_iv_host, dtype = np.int64)
        self.reg_iv = torch.tensor(self.reg_iv_host, dtype = torch.long, device = dev)
        self.operands = self._operand_table(backend.product_rows)
        # every sum interval's child intervals (its products' sum children's), and its depth: the longest chain of
        # product steps down to the inputs (0 for an interval of input children only) -- what the values kept once
        # are evicted and checkpointed by (:class:`~..state.PriorPool`)
        self.iv_children = [np.unique(self.reg_iv_np[self.operands[k]]) for k in range(len(self.sum_ivs))]
        self.iv_depth = np.zeros(len(self.sum_ivs), dtype = np.int64)
        for k in np.argsort(self.iv_bounds[:, 1] - self.iv_bounds[:, 0], kind = "stable"):     # children first
            if self.iv_children[k].size:
                self.iv_depth[k] = 1 + int(self.iv_depth[self.iv_children[k]].max())
        self.live_tables = OrderedDict()                               # (batch size, intervals) -> lane order tables
        self.no_evidence = torch.full((1, n), -1, dtype = torch.int32, device = dev)   # a sample with nothing observed
        self.plans = OrderedDict()                                     # query plans, see :mod:`.plans`
        self.layouts = OrderedDict()                                   # conditionals' class weights, see .conditional

        # every product layer's launches are first built on the host -- ``(form, kinds, rows, iv, sizes)``, see
        # :class:`~.prod.Launch` -- and passed over below, then made into launches (:meth:`_launch`). Gated sum layers
        # (:mod:`..external_params.block_scale`) run every node block fused, each edge with its gate.
        self.steps = []
        self.stack_uses = {}                                           # stack key -> dense runs reading it per query
        self.gate_cells = block_scale.cells(pc)
        prod_index = -1
        prod_layers = iter(backend.product_rows)
        for lg in pc.inner_layer_groups:
            if lg.is_prod():
                prod_index += 1
                layers = [self._product_stages(next(prod_layers)) for _ in lg.layers]
                self.steps.append(("prod", prod_index, layers))
            else:
                elem_first = backend.element_regions[prod_index][0]
                layers = []
                for layer in lg.layers:
                    parts = []
                    gated = any(gl is layer for gl in self.gate_cells.layers)
                    for nids, cids, pids in zip(layer.partitioned_nids, layer.partitioned_cids, layer.partitioned_pids):
                        regs = [region_of(int(r)) for r in nids.tolist()]
                        if not gated:
                            groups, rest = dense_groups(nids, cids, pids, layer.block_size, elem_first)
                        else:
                            groups, rest = {}, torch.arange(nids.numel(), device = nids.device)
                        dense = {}                                      # (region, first child row) -> node blocks
                        for c0, blocks in groups.items():
                            for k, nid, p0 in blocks:
                                dense.setdefault((regs[k], c0), []).append((k, nid, p0))
                        E = cids.size(1)
                        for rc, blocks in dense.items():                # and the key of their stacked weights
                            blocks.sort(key = lambda b: b[1])
                            key = _stack_key(blocks, E, layer.block_size)
                            dense[rc] = (blocks, key)
                            if key is not None:
                                self.stack_uses[key] = self.stack_uses.get(key, 0) + 1
                        fused = None
                        if rest.numel() > 0:
                            rest_regs = [regs[k] for k in rest.tolist()]
                            gate = None
                            nb_reg = torch.tensor(rest_regs, dtype = torch.long, device = dev)
                            if gated:
                                gate = block_scale.edge_tables(self.gate_cells, layer, nids[rest], cids[rest],
                                                               elem_first, dev)
                                gate["dense_run"] = block_scale.dense_run(nids[rest], cids[rest], nb_reg,
                                                                          layer.block_size, elem_first)
                            fused = (nids[rest].contiguous(), cids[rest].contiguous(), pids[rest].contiguous(), nb_reg,
                                     max(slots_of_reg[r] for r in rest_regs), None, gate)
                        parts.append((dense, cids.size(1), fused))
                    layers.append((layer.block_size, parts))
                self.steps.append(("sum", prod_index, layers))

        root_first, root_end = pc._root_node_range
        self.root_region = region_of(root_first)
        self.root_rows = (root_first, root_end)
        self.num_sms = torch.cuda.get_device_properties(dev).multi_processor_count if dev.type == "cuda" else 1

        # a one-child product over a sum is a copy of the sum's block: the sum layer that reads it reads the sum's
        # row instead, unless a dense node block reads it -- the dense path needs its children's rows contiguous --
        # and the copy is not made. ``alias`` records, per product layer group, every element row's (row within its
        # sum region, region), row -1 where the element is not aliased; the fused sum parts carry their aliased
        # children as a separate edge list
        self.alias = {}
        for k in range(len(self.steps) - 1) if ALIAS_COPIES else ():
            kind, prod_index, layers = self.steps[k]
            if kind != "prod" or self.steps[k + 1][:2] != ("sum", prod_index):
                continue
            first, end, _ = backend.element_regions[prod_index]
            dense_rows = {c0 + e for _, parts in self.steps[k + 1][2] for dense, E, _ in parts
                          for (_, c0) in dense for e in range(E)}
            alias_row = torch.full((end - first,), -1, dtype = torch.int32)
            alias_reg = torch.zeros(end - first, dtype = torch.int32)
            for stages, _, _ in layers:
                for launches in stages:
                    for li, (form, kinds, st, iv, sizes) in enumerate(launches):
                        if form != "copy" or kinds != (SUM, False):
                            continue
                        keep = torch.tensor([int(o) in dense_rows for o in st[:, 0].tolist()], dtype = torch.bool)
                        regs = st[~keep, 2]
                        alias_row[st[~keep, 0] - first] = (st[~keep, 1] - torch.tensor(firsts)[regs]).to(torch.int32)
                        alias_reg[st[~keep, 0] - first] = regs.to(torch.int32)
                        launches[li] = (form, kinds, st[keep], iv[keep], sizes[keep])
                    launches[:] = [l for l in launches if l[0] != "copy" or l[2].size(0) > 0]
            if (alias_row >= 0).any():
                self.alias[prod_index] = (alias_row, alias_reg)
                for _, parts in self.steps[k + 1][2]:
                    for pi, (dense, E, fused) in enumerate(parts):
                        if fused is not None:
                            parts[pi] = (dense, E, _split_aliased(fused, alias_row, alias_reg, first, dev))

        # where every interval a product reads or writes keeps its slots; the intervals themselves (host, and their
        # ends on the device), and per product layer group the intervals of its products, for compact storage (see
        # :mod:`..kernels.compact`)
        self.all_ivs = sorted(_intervals(self.steps) | set(self.sum_ivs))
        self.intervals = interval_tables(self.reach, self.all_ivs, dev)
        self.iv_index = {iv: k for k, iv in enumerate(self.all_ivs)}
        self.iv_ends_np = np.asarray(self.all_ivs, dtype = np.int64).reshape(-1, 2)
        self.iv_ends = torch.from_numpy(self.iv_ends_np).to(dev, torch.int32)
        self.reg_iv_all = np.asarray([self.iv_index[self.sum_ivs[k]] for k in self.reg_iv_host], dtype = np.int64)
        self.elem_ivs = {}
        for kind, prod_index, layers in self.steps:
            if kind == "prod":
                ivs = {int(k) for stages, _, _ in layers for launches in stages for *_, iv, _ in launches
                       for k in iv.unique().tolist()}
                self.elem_ivs[prod_index] = np.asarray(sorted(self.iv_index[self.sum_ivs[k]] for k in ivs),
                                                       dtype = np.int64)
        # ... as one table, a row per group padded with one past the last interval (one without lanes, see
        # :func:`~..kernels.compact._group_widths`), and every group's rows and slots: a query's widths of every
        # group's rows in a few array operations
        groups = backend.element_regions
        self.elem_iv_pad = np.full((len(groups), max([1, *map(len, self.elem_ivs.values())])), len(self.all_ivs),
                                   dtype = np.int64)
        for pi, ivs in self.elem_ivs.items():
            self.elem_iv_pad[pi, :len(ivs)] = ivs
        self.elem_rows = np.asarray([end - first for first, end, _ in groups], dtype = np.int64)
        self.elem_slots = np.asarray([slots for _, _, slots in groups], dtype = np.int64)

        # which (output tile, shared chunk) pairs every block @ block step can skip: one mask per boundary triple,
        # each step's launch carrying its triple's index
        launches_of = lambda forms: [(launches, li) for kind, _, layers in self.steps if kind == "prod"
                                     for stages, _, _ in layers for launches in stages
                                     for li, launch in enumerate(launches) if launch[0] in forms]
        triples = {}
        for launches, li in launches_of(("block_block",)):
            for tri in launches[li][2][:, 5:8].tolist():
                triples.setdefault(tuple(tri), len(triples))
        self.triple_of = triples
        self.skip_triples = list(triples)
        self.skip = skip_masks(self.reach, self.skip_triples, device = dev)

        # a missing token's transitions, GROUPED by successor -- every input @ block / block @ input step's transition
        # masses, a row per entry of the step's boundary, all kept and built once per computation of the class masses
        # -- when the boundaries' live classes are GROUP_MIN_RATIO times their successors or more (`grouped_worth`) and
        # the query's masses fit the budget (`grouped_for`, per query: `transitions`); else one entry per live class,
        # its mass read by the class's code (:class:`~..kernels.transitions.EnumeratedTokens`): the ungrouped tables,
        # built when first used. A step's masses are the r-th block of its boundary's after the position's first, r
        # its RANK among the position's steps (field 6, set here once): of the boundary's pairs when the masses are
        # one set for every sample (`pair_off`), of the sample's entries under per-sample ones (`SampleMasses`)
        ins = launches_of(tuple(TRANS_COLS))
        pairs = torch.tensor(self.succ.num_pairs, dtype = torch.long)
        self.group_worth = grouped_worth(self.transition_stats, int(pairs.sum()))
        self.trans_budget = transition_budget(dev)
        self.entry_loop = entry_loop(self.transition_stats, self.C)    # ungrouped kernels: entries, or every class
        self.trans_pairs = pairs
        self._tr, self._ungrouped = tr, None
        positions = [launches[li][2][:, TRANS_COLS[launches[li][0]][1]] for launches, li in ins]
        tcat = torch.cat(positions) if positions else torch.zeros(0, dtype = torch.long)
        steps_t = torch.bincount(tcat, minlength = self.n)               # input steps per position
        order = torch.argsort(tcat, stable = True)
        rank = torch.empty_like(tcat)
        rank[order] = torch.arange(tcat.numel()) - (torch.cumsum(steps_t, 0) - steps_t)[tcat[order]]
        rows, k = [], 0
        for (launches, li), pos in zip(ins, positions):
            form, kinds, st, iv, sz = launches[li]
            st = st.clone()
            st[:, 6] = rank[k:k + pos.numel()]
            k += pos.numel()
            rows.append(st[:, [*TRANS_COLS[form], 6]])
            launches[li] = (form, kinds, st, iv, sz)
        self.pos_steps = steps_t.to(dev)                                    # int64 [n]
        self.pairs_t = pairs.to(dev)                                        # int64 [n]: the boundaries' pairs
        self.pair_off = (torch.cumsum(steps_t * pairs, 0) - steps_t * pairs).to(dev)   # one set of masses: where a
        self.trans_size = int((steps_t * pairs).sum())                      # position's start
        self.trans_job = transition_job(torch.cat(rows), dev) if rows else None
        self._trans, self._trans_version, self._sample_layout = None, None, None
        self.trans_layout: Optional[SampleMasses] = None                    # the query's masses: one set, or a sample's
        self.trans = self.succ if self.grouped_for(self.trans_size) else self.ungrouped_tables()
        self.trans_persistent = self.trans.grouped                  # (grouped transitions are always kept whole)

        for launches, li in launches_of(("input_block", "block_input", "block_block", "copy")):
            launches[li] = self._launch(*launches[li])

    def _launch(self, form: str, kinds: tuple, rows: torch.Tensor, iv: torch.Tensor, sizes: torch.Tensor) -> Launch:
        """The :class:`~.prod.Launch` of a stage's steps built on the host (``rows`` [S, fields] int64, ``iv``
        [S], ``sizes`` [S, 3], see there); a query plan runs a subset of them through index lists
        (:class:`~.prod.Cut`)."""
        dev, S = self.dev, rows.size(0)
        groups_host = None
        if form == "block_block":
            idx = torch.tensor([self.triple_of[tuple(t)] for t in rows[:, 5:8].tolist()], dtype = torch.int32,
                               device = dev)
            steps = (rows.to(dev, torch.int32).contiguous(), idx)
        elif form in TRANS_COLS:
            groups = None                                               # input @ block steps, several to a program
            chunks = [(0, S, None)]                                     # (every step's masses at hand: one chunk)
            # (the step groups of the class-by-class kernels, built whatever the program's transitions: a query's
            # are its own, see `transitions`)
            if form == "input_block" and kinds[0] != IDENTITY and INPUT_BLOCK_GROUP > 1:
                sizes_g = dict.fromkeys((INPUT_BLOCK_GROUP, *INPUT_BLOCK_GROUP_ALTERNATIVES))   # default first
                order, firsts = _step_groups(rows, [G for G in sizes_g if G > 1])
                rows, iv, sizes = rows[order], iv[order], sizes[order]
                groups = {G: f.to(dev, torch.int32) for G, f in firsts.items()}
                groups_host = {G: f.numpy() for G, f in firsts.items()}
            steps = (rows.to(dev, torch.int32).contiguous(), chunks, groups)
        else:
            steps = rows.to(dev, torch.int32).contiguous()
        E_max, X_max, S_max = (int(sizes[:, k].max()) for k in range(3))
        a, e, past = OUT_ENDS[form]                                     # the outputs' intervals [a, e + past)
        ends = set(zip(rows[:, a].tolist(), (rows[:, e] + past).tolist()))
        out_ivs = np.unique(np.asarray([self.iv_index[iv] for iv in ends], dtype = np.int64))
        return Launch(form, kinds, steps, E_max, X_max, S_max, rows, iv, sizes, groups_host, out_ivs)

    def transition_buffer(self) -> torch.Tensor:
        """Where the query's transition masses go (``trans_size`` floats for one set of masses, a per-sample layout's
        ``size``, see :meth:`transitions`; allocated on first use, and again for more) -- under the ungrouped tables,
        whose kernels read no masses, a float."""
        if not self.trans.grouped:
            size = 1
        else:
            size = self.trans_size if self.trans_layout is None else self.trans_layout.size
        if self._trans is None or self._trans.numel() < size:
            self._trans = None
            self._trans = torch.empty(max(1, size), dtype = torch.float32, device = self.dev)
        return self._trans

    def grouped_for(self, size: int) -> bool:
        """Whether a query whose transition masses would take ``size`` floats takes the grouped transitions: worth it
        (:func:`grouped_worth`), and the masses within the budget (:func:`transition_budget`)."""
        return self.group_worth and size <= self.trans_budget

    def sample_layout(self, masses, succ) -> SampleMasses:
        """The layout of ``masses``' per-sample transition masses over the successor lists ``succ`` (their
        :class:`~..kernels.live.SampleSuccessors`), kept for the class masses' version."""
        kept = self._sample_layout
        if kept is None or kept.version != masses.version:
            kept = self._sample_layout = sample_masses(masses, succ, self)
        return kept

    def mass_args(self) -> dict:
        """The grouped kernels' arguments for the query's transition masses (see :meth:`transitions`): ``trans`` the
        buffer; a step's masses at ``mbase[b mb_stride + t] + rank * mcount[b mb_stride + t]`` for the lane's sample
        ``b``, its boundary ``t`` and its rank among the position's steps -- the position's first pair's and the
        boundary's pairs for one set of masses (``mb_stride`` 0), the sample's first and its entries' count under
        per-sample ones --, the entry's place there its pair's number ``x`` (``entry_x``), or ``SAMPLE_MASSES``: the
        sample's ``mrank[b mrank_stride + bbase[t] + x]`` (-1: no mass, see :class:`~..kernels.transitions.SampleMasses`)."""
        lay = self.trans_layout
        if lay is None:
            return dict(trans = self.transition_buffer(), mbase = self.pair_off, mcount = self.pairs_t,
                        mrank = self.pairs_t, bbase = self.succ.boundary_base, mb_stride = 0, mrank_stride = 0,
                        SAMPLE_MASSES = False)
        own = lay.Be > 1
        return dict(trans = self.transition_buffer(), mbase = lay.base, mcount = lay.count, mrank = lay.rank,
                    bbase = self.succ.boundary_base, mb_stride = self.n if own else 0,
                    mrank_stride = lay.rank.size(1) if own else 0, SAMPLE_MASSES = True)

    def ungrouped_tables(self):
        """The ungrouped transition tables (one entry per live class), built when first needed."""
        if self._ungrouped is None:
            self._ungrouped = transition_tables(self._tr, grouped = False, enum = self.enum)
        return self._ungrouped

    @property
    def trans_top_down(self):
        """The transitions as the top-down kernels read them: the forward's (the query's, see :meth:`transitions`)."""
        return self.trans

    def transitions(self, masses):
        """The transitions of a query under ``masses`` (its class masses, a :class:`~..state.ClassMasses`): grouped
        when :meth:`grouped_for` says so of the masses' size -- one set for every sample, ``trans_size`` floats, or
        under per-sample inputs that rule classes out (their :func:`~..kernels.live.live_successors`, built here when
        not yet) each sample's own over its entries (:meth:`sample_layout`) -- ``trans`` the grouped tables, the
        masses built from the class masses unless they already are -- else the ungrouped tables. A query's drivers
        call this before its liveness (whose successor lists follow ``trans``) and its passes."""
        succ = live_successors(self, masses)
        layout = None if succ is None else self.sample_layout(masses, succ)
        grouped = self.grouped_for(self.trans_size if layout is None else layout.size)
        if grouped != self.trans.grouped or (grouped and layout is not self.trans_layout):
            self.trans = self.succ if grouped else self.ungrouped_tables()
            self.trans_persistent = grouped
            self.trans_layout = layout if grouped else None
            self._trans, self._trans_version = None, None               # (the masses go with the grouped tables)
        if grouped and self.trans_job is not None and self._trans_version != masses.version:
            if layout is None:
                transition_masses(self.transition_buffer(), self.trans_job, masses, self)
            else:
                if layout.cls_ptr is None:
                    sample_classes(layout, succ, self)
                sample_transition_masses(self.transition_buffer(), self.trans_job, masses, self, layout)
            self._trans_version = masses.version

    def _entry(self, t: int) -> int:
        return self.width[t]                                         # boundary 0: the initial state alone

    def _exit(self, t: int) -> int:
        return 1 if t == self.n else self.width[t]                   # a sequence end: one column

    def _operand_table(self, product_rows) -> np.ndarray:
        """bool [sum intervals, sum regions]: whether a product over the interval (``sum_ivs``' numbers: a product's
        scope is its parent sum's) has a child in the region -- whose per-sample inside values computing the product, or
        its children's top-down values, reads (see :func:`~..queries.backend.plans.read_regions`)."""
        out = np.zeros((len(self.sum_ivs), len(self.firsts)), dtype = bool)
        firsts = np.asarray(self.firsts, dtype = np.int64)
        for _, child_rows, bounds in product_rows:
            child, bnd = child_rows.cpu().numpy(), bounds.cpu().numpy()
            k = (child >= 0).sum(axis = 1)
            ends, inverse = np.unique(bnd[:, 0] * (self.n + 1) + bnd[np.arange(k.size), k], return_inverse = True)
            iv = np.array([self.iv_number.get((int(e) // (self.n + 1), int(e) % (self.n + 1)), -1) for e in ends],
                          dtype = np.int64)[inverse.reshape(-1)]
            r, c = np.nonzero((child >= self.input_end) & (iv >= 0)[:, None])        # every sum child of a product
            out[iv[r], np.searchsorted(firsts, child[r, c], side = "right") - 1] = True
        return out

    def _product_stages(self, rows):
        """
        One product layer's table, compiled into launches. Rows that share their children's kinds and
        boundaries fold the same way, right to left: ``c_0 @ (c_1 @ (... @ c_{k-1}))``, every intermediate
        block in a scratch row, the last step writing the layer's ``element_mars`` row. Each step is one
        contraction of :mod:`.prod`, chosen by its operands; a stage holds the steps that read only
        earlier stages, grouped into one launch per contraction and operand kinds.

        :returns: ``(stages, num_temps, max_temp_slots)``: per stage, ``(form, kinds, rows, iv, sizes)`` launches on
            the host (see :class:`~.prod.Launch`); the scratch rows the layer needs and the most slots per
            sample any of them holds
        """
        out_rows, child_rows, bounds = (t.cpu().long() for t in rows)
        firsts = torch.tensor(self.firsts, dtype = torch.long)
        groups = OrderedDict()
        for r in range(out_rows.numel()):
            k = int((child_rows[r] >= 0).sum())
            kids = child_rows[r, :k].tolist()
            groups.setdefault((tuple(c < self.input_end for c in kids), tuple(bounds[r, :k + 1].tolist())), []).append(r)

        stages = defaultdict(lambda: defaultdict(list))              # stage -> (form, kinds) -> [(rows, iv, sizes)]
        temps = [0, 1]                                               # scratch rows so far, most slots per row
        zeros = None

        def block(rows_m):
            rows_m = rows_m.contiguous()
            return ("block", SUM, rows_m, torch.bucketize(rows_m, firsts, right = True) - 1)

        def scratch(R, entry, exit):
            first = temps[0]
            temps[0] += R
            temps[1] = max(temps[1], self.reach.slots(entry, exit))
            return ("block", TEMP, torch.arange(first, first + R), torch.zeros(R, dtype = torch.long))

        def emit(stage, form, kinds, fields, entry, exit, operands = ()):
            if form in ("input_block", "block_input"):    # which of the output's and operand's intervals keep every
                kinds += (tuple(self.reach.positions(a, b) is None for a, b in [(entry, exit), *operands]),)    # pair
            R = fields[0].numel()
            sizes = torch.tensor([[self._entry(entry), self._exit(exit), self.reach.slots(entry, exit)]]).expand(R, 3)
            stages[stage][(form, kinds)].append((torch.stack(fields, dim = 1), torch.full((R,), product), sizes))

        for (is_input, bd), members in groups.items():
            idx = torch.tensor(members)
            R, k = idx.numel(), len(is_input)
            product = self.iv_number[(bd[0], bd[k])]                 # the products' interval (their sums')
            kids, outs = child_rows[idx], out_rows[idx]
            zeros = torch.zeros(R, dtype = torch.long)
            full = lambda v: torch.full((R,), v, dtype = torch.long)
            if k == 1:
                if is_input[0]:                                      # the input's own block
                    emit(0, "input_block", (IDENTITY, False), [outs, kids[:, 0], full(bd[0]), zeros, zeros,
                                                               full(bd[1]), zeros], bd[0], bd[1])
                else:                                                # a product over one sum: its block as is
                    _, kind, l_rows, l_regs = block(kids[:, 0])
                    emit(0, "copy", (kind, False), [outs, l_rows, l_regs, full(bd[0]), full(bd[1])], bd[0], bd[1])
                continue
            acc = ("input", kids[:, k - 1], bd[k - 1]) if is_input[k - 1] else block(kids[:, k - 1])
            stage = 0
            for m in range(k - 2, -1, -1):
                dest = ("block", None, outs, zeros) if m == 0 else scratch(R, bd[m], bd[k])
                out_temp = m != 0
                if is_input[m]:
                    if acc[0] == "input":                            # an input next to an input: materialize it
                        tmp = scratch(R, acc[2], bd[k])
                        emit(stage, "input_block", (IDENTITY, True), [tmp[2], acc[1], full(acc[2]), zeros, zeros,
                                                                      full(bd[k]), zeros], acc[2], bd[k])
                        stage += 1
                        acc = tmp
                    emit(stage, "input_block", (acc[1], out_temp), [dest[2], kids[:, m], full(bd[m]), acc[2], acc[3],
                                                                    full(bd[k]), zeros], bd[m], bd[k],
                         [(bd[m] + 1, bd[k])])
                else:
                    _, l_kind, l_rows, l_regs = block(kids[:, m])
                    if acc[0] == "input":
                        emit(stage, "block_input", (l_kind, out_temp), [dest[2], l_rows, l_regs, full(bd[m]), acc[1],
                                                                        full(acc[2]), zeros], bd[m], bd[k],
                             [(bd[m], acc[2])])
                    else:
                        emit(stage, "block_block", (l_kind, acc[1], out_temp),
                             [dest[2], l_rows, l_regs, acc[2], acc[3], full(bd[m]), full(bd[m + 1]), full(bd[k])],
                             bd[m], bd[k])
                stage += 1
                acc = dest if m != 0 else acc

        out = [[(form, kinds, *(torch.cat([p[k] for p in parts]) for k in range(3)))
                for (form, kinds), parts in stages[stage].items()] for stage in sorted(stages)]
        return out, temps[0], temps[1]


def _intervals(steps) -> set:
    """Every interval ``(a, b)`` whose blocks the product steps read or write (the Program's host launches)."""
    out = set()
    for kind, _, layers in steps:
        if kind != "prod":
            continue
        for stages, _, _ in layers:
            for launches in stages:
                for form, kinds, st, *_ in launches:
                    if form == "input_block":                        # (input at t) @ [t + 1, t2) -> [t, t2)
                        t, t2 = st[:, 2], st[:, 5]
                        ivs = [(t, t2)] + ([(t + 1, t2)] if kinds[0] != IDENTITY else [])
                    elif form == "block_input":                      # [t0, t) @ (input at t) -> [t0, t + 1)
                        t0, t = st[:, 3], st[:, 5]
                        ivs = [(t0, t + 1), (t0, t)]
                    elif form == "block_block":                      # [t0, t1) @ [t1, t2) -> [t0, t2)
                        t0, t1, t2 = st[:, 5], st[:, 6], st[:, 7]
                        ivs = [(t0, t2), (t0, t1), (t1, t2)]
                    else:                                            # a copy of [t0, t2)
                        ivs = [(st[:, 3], st[:, 4])]
                    for a, b in ivs:
                        out.update(zip(a.tolist(), b.tolist()))
    return out


def _step_groups(st: torch.Tensor, sizes):
    """An ``input @ block`` launch's steps (host rows, see :data:`~.prod.INPUT_BLOCK_FIELDS`): ``order``, which
    sorts them by their position and block end, and per group size ``G`` of ``sizes``, ``firsts`` [groups + 1]: the
    first (sorted) step of every group of at most ``G`` consecutive steps that share both, then the end."""
    key = st[:, 2] * (int(st[:, 5].max()) + 1) + st[:, 5]
    order = torch.sort(key, stable = True).indices
    key = key[order]
    k = torch.arange(st.size(0))
    change = torch.ones(st.size(0), dtype = torch.bool)
    change[1:] = key[1:] != key[:-1]
    run_first = torch.cummax(torch.where(change, k, 0), 0).values           # the first step of each step's run
    end = torch.tensor([st.size(0)])
    return order, {G: torch.cat([torch.nonzero((k - run_first) % G == 0).flatten(), end]) for G in sizes}


def _stack_key(blocks, E: int, block_size: int):
    """The key of a dense run's :func:`~.sum.stacked_weights` -- ``(E, block size, first weights in row
    order)`` -- when two or more node blocks occupy consecutive rows (one product can write them all), else None."""
    nids = [nid for _, nid, _ in blocks]
    if len(blocks) < 2 or any(b - a != block_size for a, b in zip(nids, nids[1:])):
        return None
    return (E, block_size, tuple(p0 for _, _, p0 in blocks))


def _split_aliased(fused, alias_row, alias_reg, elem_first: int, dev):
    """A fused sum part's edges split in two: the children read in element_mars (left-packed, padded with element
    row 0, below every region) and the children that copy a sum, read in the sum's own row -- under a gate, each with
    its gate cell (``idx``, ``a_idx``) and child place (``lc``, ``a_lc``)."""
    nids, cids, pids, nb_reg, max_slots, _, gate = fused
    c, p = cids.cpu().long(), pids.cpu().long()
    inside = c >= elem_first
    row = torch.where(inside, alias_row[(c - elem_first).clamp(min = 0, max = alias_row.numel() - 1)].long(), -1)
    is_alias = inside & (row >= 0)
    if not is_alias.any():
        return fused

    def pack(keep, *cols, pad):
        order = torch.argsort((~keep).to(torch.int8), dim = 1, stable = True)
        k = max(1, int(keep.sum(dim = 1).max()))                       # never an empty table
        out = []
        for col, v in zip(cols, pad):
            g = torch.gather(col, 1, order)[:, :k]
            out.append(torch.where(torch.gather(keep, 1, order)[:, :k], g, torch.full_like(g, v)))
        return [t.to(dev, torch.int32).contiguous() for t in out]

    reg = torch.where(is_alias, alias_reg[(c - elem_first).clamp(min = 0, max = alias_row.numel() - 1)].long(), 0)
    if gate is None:
        ce, pe = pack(~is_alias, c, p, pad = (0, 0))
        ar, ag, ap = pack(is_alias, row, reg, p, pad = (-1, 0, 0))
        return nids, ce, pe, nb_reg, max_slots, (ar, ag, ap), None
    g, lc = gate["idx"].cpu().long(), torch.from_numpy(gate["lc"])
    ce, pe, ge, le = pack(~is_alias, c, p, g, lc, pad = (0, 0, 0, 0))
    ar, ag, ap, ga, la = pack(is_alias, row, reg, p, g, lc, pad = (-1, 0, 0, 0, 0))
    host = lambda t: t.cpu().long().numpy()
    gate = dict(gate, idx = ge, lc = host(le), a_idx = ga, a_lc = host(la), dense_run = None,
                uniform = block_scale.uniform_chunks(host(ge), host(ce) >= elem_first),
                a_uniform = block_scale.uniform_chunks(host(ga), host(ar) >= 0))
    return nids, ce, pe, nb_reg, max_slots, (ar, ag, ap), gate
