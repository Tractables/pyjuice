"""
The lifted forward pass: ``log p(C, e)`` for a batch of evidence, in the buffers of a
:class:`~pyjuice.constraints.ConstrainedCircuit`.

Layer by layer, as :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of
every observed token, the class masses of missing ones are summed per token class (by the circuit, see
:meth:`ConstrainedCircuit._class_masses`), then every product layer group chains its children's blocks
(:mod:`.kernels.prod`) and every sum layer group sums its children column by column (:mod:`.kernels.sum`).
The root keeps a single value per sample. Like a :class:`TensorCircuit`, the pass is driven by tables: no
kernel knows the circuit's shape. A query runs the part of them its :mod:`.query_plan` says, and the values no
sample's evidence reaches once for all of them.

Everything the pass reads besides the buffers and the parameters is built once per constrained circuit
(:class:`Program`), from the plan's tables and the PC's own layer metadata.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict, defaultdict
from typing import Optional

import numpy as np
import torch

from .kernels.prod import IDENTITY, SUM, TEMP, Launch, max_successors, skip_masks, transition_job, transition_masses
from .kernels.prod import transition_tables
from .kernels.prod import run_products as products
from .kernels.live import prune
from .kernels.sum import dense_groups, dense_sum, fused_sum, stacked_weights
from .plan import _align, interval_tables
from .query_plan import prior_regions, query_plan

#: Matrix-product precision of the kernels: "fp32" (three TF32 products, fp32-level accuracy) or "tf32".
PRECISIONS = {"fp32": "tf32x3", "tf32": "tf32"}

#: Read a one-child product over a sum straight from the sum's row instead of copying its block (see Program)
ALIAS_COPIES = True

#: A missing token's transitions are grouped by successor (see :class:`~.kernels.prod.TransitionTables`) when the
#: automaton has at least this many token classes per successor of its most branching column: its products then
#: loop over successors, not classes, through transition masses built per query. Below it, they loop over the
#: classes, and nothing more is built. 8 from a sweep on HMMs (grouped against class by class, batch 1 / 16):
#: 512 classes, 16 successors x1.6-1.8 / x2.7-3.4; 128 and 16 x1.0-1.2 / x1.4-1.5; at 4 classes per successor
#: and below, nothing gained or losses (64 successors: 256 classes x0.71-0.73 / x1.24, 64 classes x0.81-0.87).
GROUP_MIN_RATIO = 8

#: The fewest columns (batch x slots) a dense sum region needs before evidence pruning pays with more than one live
#: column (see :func:`prune_dense`).
DENSE_PRUNE_MIN_COLUMNS = 48

#: Dense node blocks over the same children multiply their stacked weights in one product (see
#: :func:`~.kernels.sum.stacked_weights`) only up to this many columns per precision (the live ones under evidence
#: pruning); wider, one product per block. From an HMM4096 sweep (4 x 1024 rows) and lifted marginals on ex1-ex4 and
#: 13.8k- / 49k-state automata: stacked wins in TF32 at every width (x1.04-1.46) but loses 1% at 49k columns; in fp32
#: it wins up to ~2k (x1.1-1.3) and loses 1-3% wider (cuBLAS's fp32 kernel for the 4096-row shape, and the log pass
#: no longer finding a block's product in L2).
STACK_MAX_COLUMNS = {"fp32": 2048, "tf32": 16384}

#: ``input @ block`` steps at the same position and block interval run this many to a program on the class-by-class
#: path, when the block is not the identity (see :func:`~.kernels.prod._input_block_group_kernel`; launches wider than
#: :data:`~.kernels.prod.GROUP_MIN_LANES`): every class's successors and slots are read once for all of them, not once
#: per step. HMM4096 sweep over 1, 2, 4, 8, 16: 8 best or within 5% of it on every workload; against one step per
#: program, bit-identical: 13.8k- / 49k-state automata x1.09-1.15 with a prefix, x1.16-1.31 on the 49k one without;
#: batch 16 x1.02-1.18. 1: one step per program.
INPUT_BLOCK_GROUP = 8
#: the other group sizes the Program builds tables for: the autotuner times every one per launch shape (see
#: :func:`~.kernels.prod.run_products`); 16 measured best on a 168-state automaton at batch 16 (x1.05 over 8)
INPUT_BLOCK_GROUP_ALTERNATIVES = (16, 4)

#: Floats of transition masses (see :func:`~.kernels.prod.transition_masses`) held at once: when every input step's
#: fit, they are all kept, built once per computation of the class masses; else built launch by launch, in chunks
#: of steps that fit, into one scratch
TRANSITION_BUDGET = 1 << 26

#: (input row, position) columns of the steps whose input's transitions they read
TRANS_COLS = {"input_block": (1, 2), "block_input": (4, 5)}


class Program:
    """Everything the lifted forward pass of one constrained circuit reads besides buffers and parameters."""

    def __init__(self, cc):
        pc, layout = cc.pc, cc.layout
        dev = cc._device
        n = cc.n
        self.n = n
        self.width = [int(w) for w in layout.width.tolist()]
        self.width_t = layout.width.to(dev, torch.int32)
        self.reach = layout.reachability                               # the slots of every interval's blocks
        self.next_col = layout.next_col.to(dev, torch.int32).contiguous()
        self.input_start, self.input_end = cc.input_range
        self.mass_row = cc._class_mass_rows()[2]                       # input node -> its row of the class masses
        grouped = cc.num_classes >= GROUP_MIN_RATIO * max_successors(self.next_col, self.width, n)
        self.trans = transition_tables(self.next_col, self.width, n, grouped = grouped)
        # every column's distinct successors (evidence pruning's liveness sweeps)
        self.succ = self.trans if grouped else transition_tables(self.next_col, self.width, n, grouped = True)
        self.dev = dev

        firsts = [first for first, _, _ in cc.sum_regions]
        self.firsts = firsts
        self.slots = [slots for _, _, slots in cc.sum_regions]
        self.reg_first = torch.tensor(firsts, dtype = torch.long, device = dev)
        self.reg_slots = torch.tensor([slots for _, _, slots in cc.sum_regions], dtype = torch.long, device = dev)
        slots_of_reg = [slots for _, _, slots in cc.sum_regions]
        region_of = lambda row: bisect.bisect_right(firsts, row) - 1

        # every sum region's interval (its node group's scope, as boundaries), for evidence pruning
        interval_of_row = {info.ns._output_ind_range[0]: (info.scope_runs[0][0], info.scope_runs[0][1] + 1)
                           for info in cc.structure.nodes if info.kind == "sum"}
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
        self.live_tables = {}                                          # batch size -> evidence pruning's lists
        self.no_evidence = torch.full((1, n), -1, dtype = torch.int32, device = dev)   # a sample with nothing observed
        self.plans = OrderedDict()                                     # query plans, see :mod:`.query_plan`
        self.layouts = OrderedDict()                                   # conditionals' class weights, see .conditional

        # every product layer's launches are first built on the host -- ``(form, kinds, rows, iv, sizes)``, see
        # :class:`~.kernels.prod.Launch` -- and passed over below, then made into launches (:meth:`_launch`)
        self.steps = []
        self.stack_uses = {}                                           # stack key -> dense runs reading it per query
        prod_index = -1
        prod_layers = iter(cc.product_rows)
        for lg in pc.inner_layer_groups:
            if lg.is_prod():
                prod_index += 1
                layers = [self._product_stages(next(prod_layers)) for _ in lg.layers]
                self.steps.append(("prod", prod_index, layers))
            else:
                elem_first = cc.element_regions[prod_index][0]
                layers = []
                for layer in lg.layers:
                    parts = []
                    for nids, cids, pids in zip(layer.partitioned_nids, layer.partitioned_cids, layer.partitioned_pids):
                        regs = [region_of(int(r)) for r in nids.tolist()]
                        groups, rest = dense_groups(nids, cids, pids, layer.block_size, elem_first)
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
                            fused = (nids[rest].contiguous(), cids[rest].contiguous(), pids[rest].contiguous(),
                                     torch.tensor(rest_regs, dtype = torch.long, device = dev),
                                     max(slots_of_reg[r] for r in rest_regs), None)
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
            first, end, _ = cc.element_regions[prod_index]
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

        # where every interval a product reads or writes keeps its slots
        self.intervals = interval_tables(self.reach, sorted(_intervals(self.steps) | set(self.sum_ivs)), dev)

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

        # grouped transitions: every input @ block / block @ input step's transition masses, a row of the boundary's
        # pairs at t_off. All kept at once when they fit TRANSITION_BUDGET (offsets set here, once); else each launch
        # in chunks of steps that fit, sharing one scratch (a chunk's masses are built just before its steps run)
        ins = launches_of(tuple(TRANS_COLS))
        self.trans_pairs = torch.tensor(self.trans.num_pairs, dtype = torch.long) if self.trans.grouped else None
        sizes = [self.trans_pairs[launches[li][2][:, TRANS_COLS[launches[li][0]][1]]] if self.trans.grouped else
                 torch.zeros(launches[li][2].size(0), dtype = torch.long) for launches, li in ins]
        self.trans_persistent = sum(int(z.sum()) for z in sizes) <= TRANSITION_BUDGET
        self.trans_size, rows = 0, []
        if self.trans.grouped and self.trans_persistent:
            for z, (launches, li) in zip(sizes, ins):
                form, kinds, st, iv, sz = launches[li]
                st = st.clone()
                st[:, 6] = self.trans_size + torch.cumsum(z, 0) - z
                self.trans_size += int(z.sum())
                rows.append(st[:, [*TRANS_COLS[form], 6]])
                launches[li] = (form, kinds, st, iv, sz)
        self.trans_job = transition_job(torch.cat(rows), dev) if rows else None
        self._trans, self._trans_version = None, None

        for launches, li in launches_of(("input_block", "block_input", "block_block", "copy")):
            launches[li] = self._launch(*launches[li])

    def _launch(self, form: str, kinds: tuple, rows: torch.Tensor, iv: torch.Tensor, sizes: torch.Tensor) -> Launch:
        """The :class:`~.kernels.prod.Launch` of a stage's steps built on the host (``rows`` [S, fields] int64, ``iv``
        [S], ``sizes`` [S, 3], see there); a query plan runs a subset of them through index lists
        (:class:`~.kernels.prod.Cut`)."""
        dev, S = self.dev, rows.size(0)
        groups_host = None
        if form == "block_block":
            idx = torch.tensor([self.triple_of[tuple(t)] for t in rows[:, 5:8].tolist()], dtype = torch.int32,
                               device = dev)
            steps = (rows.to(dev, torch.int32).contiguous(), idx)
        elif form in TRANS_COLS:
            groups = None                                               # input @ block steps, several to a program
            if not self.trans.grouped:
                chunks = [(0, S, None)]
                if form == "input_block" and kinds[0] != IDENTITY and INPUT_BLOCK_GROUP > 1:
                    sizes_g = dict.fromkeys((INPUT_BLOCK_GROUP, *INPUT_BLOCK_GROUP_ALTERNATIVES))   # default first
                    order, firsts = _step_groups(rows, [G for G in sizes_g if G > 1])
                    rows, iv, sizes = rows[order], iv[order], sizes[order]
                    groups = {G: f.to(dev, torch.int32) for G, f in firsts.items()}
                    groups_host = {G: f.numpy() for G, f in firsts.items()}
            elif self.trans_persistent:
                chunks = [(0, S, None)]                                 # offsets set once, see __init__
            else:
                rows = rows.clone()
                bounds, start, used = [], 0, 0
                for k, size in enumerate(self.trans_pairs[rows[:, TRANS_COLS[form][1]]].tolist()):
                    if used + size > TRANSITION_BUDGET and k > start:
                        bounds.append((start, k))
                        start, used = k, 0
                    rows[k, 6] = used
                    used += size
                    self.trans_size = max(self.trans_size, used)
                bounds.append((start, S))
                chunks = [(s0, s1, transition_job(rows[s0:s1][:, [*TRANS_COLS[form], 6]], dev)) for s0, s1 in bounds]
            steps = (rows.to(dev, torch.int32).contiguous(), chunks, groups)
        else:
            steps = rows.to(dev, torch.int32).contiguous()
        E_max, X_max, S_max = (int(sizes[:, k].max()) for k in range(3))
        return Launch(form, kinds, steps, E_max, X_max, S_max, rows, iv, sizes, groups_host)

    def transition_buffer(self) -> torch.Tensor:
        """Where the transition masses go (all of them, or one chunk's; allocated on first use, and again when a
        query plan's chunks need more)."""
        if self._trans is None or self._trans.numel() < self.trans_size:
            self._trans = None
            self._trans = torch.empty(max(1, self.trans_size), dtype = torch.float32, device = self.dev)
        return self._trans

    def transitions(self, class_mars: torch.Tensor, version: int):
        """When transitions are grouped and all their masses kept: build them from the class masses of computation
        ``version``, unless they already are."""
        if self.trans.grouped and self.trans_persistent and self.trans_job is not None and \
                self._trans_version != version:
            transition_masses(self.transition_buffer(), self.trans_job, class_mars, self)
            self._trans_version = version

    def _entry(self, t: int) -> int:
        return self.width[t]                                         # boundary 0: the initial state alone

    def _exit(self, t: int) -> int:
        return 1 if t == self.n else self.width[t]                   # a sequence end: one column

    def _product_stages(self, rows):
        """
        One product layer's table, compiled into launches. Rows that share their children's kinds and
        boundaries fold the same way, right to left: ``c_0 @ (c_1 @ (... @ c_{k-1}))``, every intermediate
        block in a scratch row, the last step writing the layer's ``element_mars`` row. Each step is one
        contraction of :mod:`.kernels.prod`, chosen by its operands; a stage holds the steps that read only
        earlier stages, grouped into one launch per contraction and operand kinds.

        :returns: ``(stages, num_temps, max_temp_slots)``: per stage, ``(form, kinds, rows, iv, sizes)`` launches on
            the host (see :class:`~.kernels.prod.Launch`); the scratch rows the layer needs and the most slots per
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
    """An ``input @ block`` launch's steps (host rows, see :data:`~.kernels.prod.INPUT_BLOCK_FIELDS`): ``order``, which
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
    """The key of a dense run's :func:`~.kernels.sum.stacked_weights` -- ``(E, block size, first weights in row
    order)`` -- when two or more node blocks occupy consecutive rows (one product can write them all), else None."""
    nids = [nid for _, nid, _ in blocks]
    if len(blocks) < 2 or any(b - a != block_size for a, b in zip(nids, nids[1:])):
        return None
    return (E, block_size, tuple(p0 for _, _, p0 in blocks))


def _split_aliased(fused, alias_row, alias_reg, elem_first: int, dev):
    """A fused sum part's edges split in two: the children read in element_mars (left-packed, padded with element
    row 0, below every region) and the children that copy a sum, read in the sum's own row."""
    nids, cids, pids, nb_reg, max_slots, _ = fused
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
    ce, pe = pack(~is_alias, c, p, pad = (0, 0))
    ar, ag, ap = pack(is_alias, row, reg, p, pad = (-1, 0, 0))
    return nids, ce, pe, nb_reg, max_slots, (ar, ag, ap)


def prune_dense(ncols: int, num_live: int) -> bool:
    """
    Whether a dense sum region with ``ncols`` columns (batch x slots), ``num_live`` of them live under evidence
    pruning, multiplies only its live ones -- exact either way. From the library's dense products at an HMM4096 size
    (4096 x 4096 stacked weights, fp32 and tf32, 4 to 512 columns, forward and top-down): one live column (or none)
    always pays -- a matrix-vector product, x1.07-6.2; otherwise pruning pays once the product has left the
    weight-read floor (from ``DENSE_PRUNE_MIN_COLUMNS`` columns) and at least half the columns are dead (x1.2-2.4
    with a quarter live), while below, the gather and scatter cost more than they save (x0.74-0.98). Reading
    ``num_live`` is the query's one wait for the device (see :class:`~.kernels.live.Pruning`).
    """
    return num_live < ncols and (num_live <= 1 or (ncols >= DENSE_PRUNE_MIN_COLUMNS and 2 * num_live <= ncols))


def _missing(missing_mask: Optional[torch.Tensor], B: int, n: int, dev) -> torch.Tensor:
    if missing_mask is None:
        return torch.zeros(B, n, dtype = torch.bool, device = dev)
    missing = missing_mask.to(dev).bool()
    if missing.dim() == 1:
        missing = missing[None, :].expand(B, n)
    if missing.shape != (B, n):
        raise ValueError(f"`missing_mask` must be [{n}] or [{B}, {n}], got {list(missing_mask.shape)}.")
    return missing.contiguous()


def marginal(cc, data: torch.Tensor, missing_mask: Optional[torch.Tensor] = None, precision: str = "fp32"):
    """
    ``log p(C, e)`` for every sample: the probability that the PC generates a string that satisfies the
    constraint and agrees with the observed tokens.

    :param data: [B, n] token ids (ignored where missing)
    :param missing_mask: None (everything observed), [n] or [B, n]; True where a token is marginalized
    :param precision: "fp32" (default) or "tf32" for the sum layers' matrix products
    :returns: [B, number of root nodes], as :func:`pyjuice.queries.marginal`
    """
    if precision not in PRECISIONS:
        raise ValueError(f"`precision` must be one of {sorted(PRECISIONS)}, got {precision!r}.")
    pc, n, dev = cc.pc, cc.n, cc._device
    if dev.type != "cuda":
        raise NotImplementedError(f"The lifted backend runs on CUDA only, but the constrained circuit is on {dev}. "
                                  f"Move it with `cc.to('cuda')`.")
    if data.dim() != 2 or data.size(1) != n:
        raise ValueError(f"`data` must be [batch, {n}], got {list(data.shape)}.")
    B = data.size(0)
    root_first, root_end = pc._root_node_range
    if not cc.satisfiable:
        return torch.full((B, root_end - root_first), -float("inf"), device = dev)

    prog = cc._lifted_program()
    missing = _missing(missing_mask, B, n, dev)
    observed = np.ones(n, dtype = bool) if missing_mask is None else \
        (~missing).any(dim = 0).cpu().numpy()                           # the positions some sample observes
    x = torch.where(missing, 0, data.to(dev).long())                     # missing tokens: any valid id
    bufs = cc._buffers(B)

    with torch.no_grad():
        obs_class = torch.where(missing, -1, cc.token_classes.token_class[x]).contiguous()
        pruning = prune(prog, obs_class, B) if observed.any() else None     # nothing observed: nothing to prune
        valid, kept = cc._prior_validity()
        prior = prior_regions(prog, observed, B, kept, pruning)
        plan = query_plan(prog, observed, None, valid.numpy(), prior)   # every inside value
        run_forward(cc, bufs, plan, x, missing, obs_class, pruning, precision)
        valid |= plan.todo

        r = prog.root_region
        if plan.root_prior:                                             # nothing observed: every sample's value
            prior = cc._prior_buffer()
            base = prior["offset"][r] + (root_first - cc.sum_regions[r][0]) * prior["width"][r]
            rows = base + torch.arange(root_end - root_first, device = dev) * prior["width"][r]
            return prior["mars"][rows][None, :].expand(B, -1).contiguous()
        lay = bufs["layout"]
        base = lay["sum_offsets"][r] + (root_first - cc.sum_regions[r][0]) * lay["sum_widths"][r]
        width = lay["sum_widths"][r]
        rows = base + torch.arange(root_end - root_first, device = dev)[:, None] * width
        return bufs["node_mars"][rows + torch.arange(B, device = dev)[None, :]].t().contiguous()


def run_forward(cc, bufs: dict, plan, x: torch.Tensor, missing: torch.Tensor, obs_class: torch.Tensor, pruning,
                precision: str):
    """
    The inside values a query plan computes (see :mod:`.query_plan`): the per-sample ones into ``node_mars``, and
    those kept once that ``prior_mars`` does not hold yet, for a batch of one, into ``prior_mars``. Layer by layer, as
    :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of every observed token, then
    every product layer and every sum layer run their parts of the plan.

    :param bufs: the batch's :meth:`~pyjuice.constraints.ConstrainedCircuit._buffers`
    :param x: [B, n] token ids (any valid id where missing)
    :param missing: [B, n] bool, True where a token is missing
    :param obs_class: [B, n] int32, every observed token's class, -1 where missing
    :param pruning: None, or the query's :class:`~.kernels.live.Pruning`
    :param precision: "fp32" or "tf32" for the sum layers' matrix products
    :returns: the class masses the pass read (:meth:`~pyjuice.constraints.ConstrainedCircuit._class_masses`; computed
        for this query outside :func:`pyjuice.fast_inference`), for the query's later passes
    """
    pc, prog = cc.pc, cc._lifted_program()
    B = x.size(0)
    lay = bufs["layout"]
    node_mars, element_mars = bufs["node_mars"], bufs["element_mars"]
    regions = dict(offset = lay["sum_offsets_t"], width = lay["sum_widths_t"], first = prog.reg_first,
                   slots = prog.reg_slots)
    prior = cc._prior_buffer() if plan.prior else None
    if prior is not None:                                              # where the values kept once are
        once = dict(offset = prior["offset_t"], width = prior["width_t"], first = prog.reg_first, slots = prog.reg_slots)
        srcs = dict(mars = prior["mars"], offset = prior["offset_t"], width = prior["width_t"], reg_prior = plan.reg_prior)
    else:                                                              # none: never read
        srcs = dict(mars = node_mars, offset = regions["offset"], width = regions["width"], reg_prior = plan.reg_prior)
    dot = PRECISIONS[precision]
    live = pruning.live if pruning is not None else None

    for layer in pc.input_layer_group:                                 # the input region: pyjuice's own layout
        layer(x.permute(1, 0), bufs["input_mars"], missing_mask = missing)
    class_mars = cc._class_masses()
    prog.transitions(class_mars, cc._class_mars_version)
    stacks, kept = cc._weight_stacks()

    for kind, prod_index, layers in plan.steps:
        elem_first = cc.element_regions[prod_index][0]
        elem_width = lay["element_widths"][prod_index]
        if kind == "prod":
            for prior_stages, sample_stages, num_temps, temp_slots in layers:
                temp, temp_width = node_mars, 0                      # no scratch rows: never read
                if num_temps > 0:
                    temp_width = _align(B * temp_slots)
                    temp = cc._storage_view("product_scratch", num_temps * temp_width)
                if prior_stages:                                     # nothing observed in their scopes
                    products(prior_stages, (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             prog.no_evidence, None, elem_first, elem_width, 1, srcs, exact = True)
                if sample_stages:
                    products(sample_stages, (temp, temp_width), prog, regions, element_mars, node_mars, class_mars,
                             obs_class, live, elem_first, elem_width, B, srcs, exact = pruning is not None)
        else:
            for block_size, prior_parts, sample_parts in layers:
                if prior_parts:
                    _sums(cc, prior_parts, block_size, prior["mars"], element_mars, once, prior["offset"],
                          prior["width"], elem_first, elem_width, 1, None, stacks, kept, precision)
                if sample_parts:
                    _sums(cc, sample_parts, block_size, node_mars, element_mars, regions, lay["sum_offsets"],
                          lay["sum_widths"], elem_first, elem_width, B, pruning, stacks, kept, precision)
    return class_mars


def _sums(cc, parts, block_size: int, nodes: torch.Tensor, element_mars: torch.Tensor, regions: dict, offsets, widths,
          elem_first: int, elem_width: int, B: int, pruning, stacks: dict, kept: bool, precision: str):
    """A sum layer's parts a query plan runs (``(dense runs, num_edges, fused)``: the runs ``(region, c0, blocks,
    key)``, and None or ``(fused part, index list of its node blocks or None for all, count, most slots)``, see
    :func:`~.query_plan.query_plan`) into ``nodes`` -- ``node_mars``, or ``prior_mars`` with ``B`` 1 and no pruning --
    whose sum regions ``regions`` (device tensors) and ``offsets`` / ``widths`` (Python ints) describe."""
    prog, pc = cc._lifted_program(), cc.pc
    dot = PRECISIONS[precision]
    for runs, num_edges, fused in parts:
        for r, c0, blocks, key in runs:
            region = (offsets[r], widths[r], prog.firsts[r])
            ncols = B * prog.slots[r]
            cols = None                                                  # every column live
            if pruning is not None and prune_dense(ncols, pruning.num_live[prog.reg_iv_host[r]]):
                cols = pruning.segment(prog.reg_iv_host[r], ncols)
            stacked = None                                  # stacked if narrow enough and the copy is kept or reused
            width = cols[2] if cols is not None else ncols                                # the product's columns
            if key is not None and width <= STACK_MAX_COLUMNS[precision] and (kept or prog.stack_uses[key] > 1):
                if key not in stacks:
                    stacks[key] = stacked_weights(pc.params, key[2], num_edges, block_size)
                stacked = {c0: stacks[key]}
            dense_sum(nodes, element_mars, pc.params, {c0: blocks}, num_edges, block_size, region, elem_first,
                      elem_width, ncols, dot, live = cols, stacked = stacked)
        if fused is not None:
            (nids, cids, pids, nb_reg, _, aliased), sel, count, max_slots = fused
            fused_sum(nodes, element_mars, pc.params, nids, cids, pids, nb_reg, regions, elem_first, elem_width, B,
                      block_size, max_cols = B * max_slots, precision = dot, aliased = aliased, pruning = pruning,
                      sel = None if sel is None else sel.tensor, count = count)
