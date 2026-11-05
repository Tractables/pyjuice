"""
What the forward pass reads besides the buffers and the parameters, built once per backend (:class:`Program`): every
product layer's launches -- each product folded into binary steps, the steps of one contraction launched together
(:mod:`.prod`) -- and every sum layer's dense runs and fused parts (:mod:`.sum`), with the interval,
transition and skip tables they read. A query runs the subsets of them its plan selects (:mod:`.plans`).
"""

from __future__ import annotations

from collections import OrderedDict
from functools import cached_property
from typing import Optional, Tuple

import numpy as np
import torch

from ..kernels.prod import skip_masks_for
from ..kernels.live import live_successors
from ..kernels.sample_transitions import sample_transition_masses
from ..kernels.transitions import (SampleMasses, enumerated_tokens, next_col_table, sample_classes,
                                   sample_masses, transition_masses, transition_stats, transition_tables)
from ..kernels.common import available_memory, most_in_runs
from .circuit import TRANS_COLS, circuit_tables
from .reachability import interval_tables


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
#: path, when the block is not the identity (see :func:`~.kernels.prod.input_block._input_block_group_kernel`; launches wider than
#: :data:`~.kernels.prod.run.GROUP_MIN_LANES`): every class's successors and slots are read once for all of them, not once
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
    return max(TRANSITION_BUDGET, int(TRANSITION_MEMORY_SHARE * available_memory(dev)) // 4)


def entry_loop(stats: Tuple[int, int, int], C: int) -> bool:
    """Whether the ungrouped forward kernels loop over a column's live entries rather than every class: at most
    :data:`ENTRY_LOOP_MAX_LIVE` of the (column, class) pairs live (``stats`` as :func:`grouped_worth`)."""
    live, columns, _ = stats
    return live < ENTRY_LOOP_MAX_LIVE * columns * C


class Program:
    """Everything the lifted forward pass of one backend reads besides buffers and parameters: what the circuit alone
    says, shared with its other backends (:mod:`.circuit`), and what the constraint's layout says of it."""

    def __init__(self, backend):
        layout = backend.layout
        dev = backend.device
        circuit = circuit_tables(backend.pc, backend.structure, dev, alias_copies = ALIAS_COPIES,
                                 input_block_groups = tuple(dict.fromkeys((INPUT_BLOCK_GROUP, *INPUT_BLOCK_GROUP_ALTERNATIVES))))
        self.__dict__.update(circuit.attrs)
        n = self.n
        self.width = [int(w) for w in layout.width.tolist()]
        self.width_t = layout.width.to(dev, torch.int32)
        self.reach = layout.reachability                               # the slots of every interval's blocks
        self.layout = layout
        # the automaton's transitions and the boundaries' column maps (int32: what the kernels read a boundary's
        # transitions through, see Transitions; the tables below are built from them, a boundary at a time)
        tr = layout.transitions(dev)
        self.delta, self.state_id, self.col_of_state = (x.to(torch.int32).contiguous()
                                                        for x in (tr.delta, tr.state_id, tr.col_of_state))
        self.K, self.W, self.C = tr.K, tr.W, tr.C
        self.next_col = None                                             # the table over every boundary, if kept
        if n * self.W * self.C <= TRANSITION_TABLE_MAX_ENTRIES:
            self.next_col = next_col_table(tr)
        self.homogeneous = self.next_col is None
        self.next_col_arg = self.delta if self.homogeneous else self.next_col    # (a pointer the kernels never read)
        self.mass_row = backend.state.class_mass_rows()[2]               # input node -> its row of the class masses
        self.token_classes = backend.token_classes
        self.enum = enumerated_tokens(tr, backend.token_classes)         # the enumerated classes' codes and tokens
        # every column's distinct successors (evidence pruning's liveness sweeps; the forward's tables when its
        # transitions are grouped: decided below, once the steps are known)
        self.succ = transition_tables(tr, grouped = True, enum = self.enum)
        self.transition_stats = transition_stats(tr)

        regions = np.asarray(backend.sum_regions)
        self.slots = regions[:, 2].tolist()
        self.slots_host = regions[:, 2]
        self.reg_slots = torch.from_numpy(regions[:, 2].copy()).to(dev)
        # every sum interval's slots, for evidence pruning and query plans (host)
        self.iv_slots = self.reach.slots_of(self.iv_bounds)
        self.sum_iv_slots = self.iv_slots.tolist()
        # every boundary's columns and every interval's slots and whether it keeps every pair (host, by the interval's
        # number in `all_ivs`): what the launches' sizes read
        self.reach.prepare(self.all_ivs)
        ends = self.iv_ends_np
        self.entries = np.asarray(self.width, dtype = np.int64)
        self.exits = self.entries.copy()
        self.exits[n] = 1
        self.all_iv_slots = self.reach.slots_of(self.iv_ends_np)
        self.all_iv_full = self.all_iv_slots == self.entries[ends[:, 0]] * self.exits[ends[:, 1]]
        self.live_tables = OrderedDict()                               # (batch size, intervals) -> lane order tables
        self.plans = OrderedDict()                                     # query plans, see :mod:`.plans`
        self.cw_walks = {}                                             # (see :func:`~.kernels.score.class_weight_kernels._class_weight_cuda`)
        self.layouts = OrderedDict()                                   # conditionals' class weights, see .conditional

        # every product layer's launches, what the layout says of them; gated sum layers
        # (:mod:`..external_params.block_scale`) run every node block fused, each edge with its gate
        assembly = self.assembly = circuit.assembly(self.reach)
        self.score = assembly.score
        reach, all_slots = self.reach, self.all_iv_slots
        # what the automaton says of the launches: their parts' columns and slots, the slots of the scratch rows
        self.launch_parts = np.stack([self.entries[assembly.part_a], self.exits[assembly.part_b],
                                      all_slots[assembly.part_iv]], axis = 1)
        self.temp_slots = most_in_runs(all_slots[assembly.temp_iv], assembly.temp_ptr, 1).tolist()
        self.steps = assembly.steps
        self.intervals = interval_tables(reach, self.all_ivs, dev, of = circuit.of_t, ends = self.iv_ends_np)
        self.elem_slots = np.asarray([slots for _, _, slots in backend.element_regions], dtype = np.int64)

        # which (output tile, shared chunk) pairs every block @ block step can skip: one mask per boundary triple,
        # each step's launch carrying its triple's index
        self.triple_of = assembly.triple_of
        self.skip_triples = assembly.skip_triples
        self.skip = skip_masks_for(reach, assembly.skip_table, device = dev)

        # a missing token's transitions, GROUPED by successor -- every input @ block / block @ input step's transition
        # masses, a row per entry of the step's boundary, all kept and built once per computation of the class masses
        # -- when the boundaries' live classes are GROUP_MIN_RATIO times their successors or more (`grouped_worth`) and
        # the query's masses fit the budget (`grouped_for`, per query: `transitions`); else one entry per live class,
        # its mass read by the class's code (:class:`~..kernels.transitions.EnumeratedTokens`): the ungrouped tables,
        # built when first used. A step's masses are the r-th block of its boundary's after the position's first, r
        # its RANK among the position's steps (field 6, see :class:`~.circuit.Assembly`): of the boundary's pairs when
        # the masses are one set for every sample (`pair_off`), of the sample's entries under per-sample ones
        # (`SampleMasses`)
        pairs = torch.tensor(self.succ.num_pairs, dtype = torch.long)
        self.group_worth = grouped_worth(self.transition_stats, int(pairs.sum()))
        self.entry_loop = entry_loop(self.transition_stats, self.C)    # ungrouped kernels: entries, or every class
        self.trans_pairs = pairs
        self._tr, self._ungrouped = tr, None
        steps_t = assembly.steps_host                                       # input steps per position
        self.pos_steps = assembly.pos_steps                                 # int64 [n]
        self.pairs_t = pairs.to(dev)                                        # int64 [n]: the boundaries' pairs
        self.pair_off = (torch.cumsum(steps_t * pairs, 0) - steps_t * pairs).to(dev)   # one set of masses: where a
        self.trans_size = int((steps_t * pairs).sum())                      # position's start
        self.trans_job = assembly.trans_job
        self._trans, self._trans_version, self._sample_layout = None, None, None
        self.trans_layout: Optional[SampleMasses] = None                    # the query's masses: one set, or a sample's
        self.trans_sig = None                                               # ... one set by column signature, or not
        self._sig = None                                                    # (the signature layout and its row map)
        self.trans = self.succ if self.grouped_for(self.trans_size) else self.ungrouped_tables()
        self.trans_persistent = self.trans.grouped                  # (grouped transitions are always kept whole)

    def transition_buffer(self) -> torch.Tensor:
        """Where the query's transition masses go (``trans_size`` floats for one set of masses, a per-sample layout's
        ``size``, see :meth:`transitions`; allocated on first use, and again for more) -- under the ungrouped tables,
        whose kernels read no masses, a float."""
        if not self.trans.grouped:
            size = 1
        elif self.trans_layout is not None:
            size = self.trans_layout.size
        else:
            size = self.trans_size if self.trans_sig is None else self.trans_sig.size
        if self._trans is None or self._trans.numel() < size:
            self._trans = None
            self._trans = torch.empty(max(1, size), dtype = torch.float32, device = self.dev)
        return self._trans

    def grouped_for(self, size: int) -> bool:
        """Whether a query whose transition masses would take ``size`` floats takes the grouped transitions: worth it
        (:func:`grouped_worth`), and the masses within the budget (:func:`transition_budget`)."""
        if not self.group_worth:
            return False
        if "trans_budget" not in self.__dict__ and size <= TRANSITION_BUDGET:   # (it is at least that: no look at the device)
            return True
        return size <= self.trans_budget

    @cached_property
    def trans_budget(self) -> int:
        return transition_budget(self.dev)

    def most_of(self, launch) -> list:
        """The most entry columns, exit columns and slots of a :class:`~..kernels.prod.Launch`'s steps' outputs."""
        ptr = self.assembly.part_ptr
        return self.launch_parts[ptr[launch.number]:ptr[launch.number + 1]].max(axis = 0).tolist()

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
        lay, sig = self.trans_layout, self.trans_sig
        if lay is None:
            mbase, mcount = (self.pair_off, self.pairs_t) if sig is None else (sig.pair_off, sig.pairs)
            return dict(trans = self.transition_buffer(), mbase = mbase, mcount = mcount, mrank = mcount,
                        bbase = self.succ.boundary_base, mb_stride = 0, mrank_stride = 0, SAMPLE_MASSES = False)
        own = lay.Be > 1
        return dict(trans = self.transition_buffer(), mbase = lay.base, mcount = lay.count, mrank = lay.rank,
                    bbase = self.succ.boundary_base, mb_stride = self.n if own else 0,
                    mrank_stride = lay.rank.size(1) if own else 0, SAMPLE_MASSES = True)

    def mass_x(self, kind: str) -> torch.Tensor:
        """The query's tables' ``entry_x`` or ``pred_x`` (``kind``) as the grouped kernels index the masses with them:
        the pairs' signature slots under masses by column signature (:attr:`trans_sig`), else the tables' own."""
        if self.trans.grouped and self.trans_layout is None and self.trans_sig is not None:
            return getattr(self.trans_sig, kind)
        return getattr(self.trans, kind)

    def signature_layout(self, masses):
        """The :class:`~..kernels.transitions.signatures.SignatureMasses` of ``masses``' row map (one set), or None
        (no saving, or per-sample masses): built once a row map (the state's, which a query's one set reads)."""
        from ..kernels.transitions import signatures as TS
        if not TS.SIGNATURE_MASSES or masses.batch != 1 or masses.enum_logw is not None:
            return None
        if self._sig is None or self._sig[0] is not masses.row:
            self._sig = (masses.row, TS.signature_masses(self, masses, self.trans_size))
        return self._sig[1]

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
        sig = self.signature_layout(masses) if layout is None else None    # one set: by column signature, if it saves
        size = layout.size if layout is not None else (self.trans_size if sig is None else sig.size)
        grouped = self.grouped_for(size)
        if grouped != self.trans.grouped or (grouped and (layout is not self.trans_layout or sig is not self.trans_sig)):
            self.trans = self.succ if grouped else self.ungrouped_tables()
            self.trans_persistent = grouped
            self.trans_layout = layout if grouped else None
            self.trans_sig = sig if grouped else None
            self._trans, self._trans_version = None, None               # (the masses go with the grouped tables)
        if grouped and self.trans_job is not None and self._trans_version != masses.version:
            if layout is None and self.trans_sig is not None:
                from ..kernels.transitions.signatures import signature_transition_masses
                signature_transition_masses(self.transition_buffer(), self.trans_sig, masses, self)
            elif layout is None:
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
