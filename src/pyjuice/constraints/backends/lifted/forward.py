"""
The lifted forward pass: ``log p(C, e)`` for a batch of evidence, in the buffers of a
:class:`~pyjuice.constraints.ConstrainedCircuit`.

Layer by layer, as :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of
every observed token, the class masses of missing ones are summed per token class (by the circuit, see
:meth:`ConstrainedCircuit._class_masses`), then every product layer group chains its children's blocks
(:mod:`.kernels.prod`) and every sum layer group sums its children column by column (:mod:`.kernels.sum`).
The root keeps a single value per sample. Like a :class:`TensorCircuit`, the pass is driven by tables: no
kernel knows the circuit's shape.

Everything the pass reads besides the buffers and the parameters is built once per constrained circuit
(:class:`Program`), from the plan's tables and the PC's own layer metadata.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict, defaultdict
from typing import Optional

import torch

from .kernels.prod import IDENTITY, SUM, TEMP, max_successors, skip_masks, transition_job, transition_masses
from .kernels.prod import transition_tables
from .kernels.prod import run_products as products
from .kernels.sum import dense_groups, dense_sum, fused_sum
from .plan import Reachability, _align

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

#: Floats of transition masses (see :func:`~.kernels.prod.transition_masses`) held at once: when every input step's
#: fit, they are all kept, built once per computation of the class masses; else built launch by launch, in chunks
#: of steps that fit, into one scratch
TRANSITION_BUDGET = 1 << 26


class Program:
    """Everything the lifted forward pass of one constrained circuit reads besides buffers and parameters."""

    def __init__(self, cc):
        pc, layout = cc.pc, cc.layout
        dev = cc._device
        n = cc.n
        self.n = n
        self.width = [int(w) for w in layout.width.tolist()]
        self.width_t = layout.width.to(dev, torch.int32)
        self.next_col = layout.next_col.to(dev, torch.int32).contiguous()
        self.input_start, self.input_end = cc.input_range
        self.mass_row = cc._class_mass_rows()[2]                       # input node -> its row of the class masses
        grouped = cc.num_classes >= GROUP_MIN_RATIO * max_successors(self.next_col, self.width, n)
        self.trans = transition_tables(self.next_col, self.width, n, grouped = grouped)
        self.dev = dev

        firsts = [first for first, _, _ in cc.sum_regions]
        self.firsts = firsts
        self.slots = [slots for _, _, slots in cc.sum_regions]
        self.reg_first = torch.tensor(firsts, dtype = torch.long, device = dev)
        self.reg_slots = torch.tensor([slots for _, _, slots in cc.sum_regions], dtype = torch.long, device = dev)
        slots_of_reg = [slots for _, _, slots in cc.sum_regions]
        region_of = lambda row: bisect.bisect_right(firsts, row) - 1

        self.steps = []
        prod_index = -1
        prod_layers = iter(cc.product_rows)
        for lg in pc.inner_layer_groups:
            if lg.is_prod():
                prod_index += 1
                layers = [self._product_stages(next(prod_layers), dev) for _ in lg.layers]
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
                    for li, (form, kinds, steps, E, X) in enumerate(launches):
                        if form != "copy" or kinds != (SUM, False):
                            continue
                        st = steps.cpu().long()
                        keep = torch.tensor([int(o) in dense_rows for o in st[:, 0].tolist()], dtype = torch.bool)
                        regs = st[~keep, 2]
                        alias_row[st[~keep, 0] - first] = (st[~keep, 1] - torch.tensor(firsts)[regs]).to(torch.int32)
                        alias_reg[st[~keep, 0] - first] = regs.to(torch.int32)
                        launches[li] = (form, kinds, st[keep].to(dev, torch.int32).contiguous(), E, X)
                    launches[:] = [l for l in launches if l[0] != "copy" or l[2].size(0) > 0]
            if (alias_row >= 0).any():
                self.alias[prod_index] = (alias_row, alias_reg)
                for _, parts in self.steps[k + 1][2]:
                    for pi, (dense, E, fused) in enumerate(parts):
                        if fused is not None:
                            parts[pi] = (dense, E, _split_aliased(fused, alias_row, alias_reg, first, dev))

        # which (output tile, shared chunk) pairs every block @ block step can skip: one mask per boundary triple,
        # each step's launch carrying its triple's index
        triples = {}
        bb = [(launches, li) for kind, _, layers in self.steps if kind == "prod" for stages, _, _ in layers
              for launches in stages for li, launch in enumerate(launches) if launch[0] == "block_block"]
        for launches, li in bb:
            for tri in launches[li][2][:, 5:8].tolist():
                triples.setdefault(tuple(tri), len(triples))
        self.skip_triples = list(triples)
        self.skip = skip_masks(Reachability(layout), self.skip_triples, device = dev)
        for launches, li in bb:
            form, kinds, steps, E, X = launches[li]
            idx = torch.tensor([triples[tuple(t)] for t in steps[:, 5:8].tolist()], dtype = torch.int32, device = dev)
            launches[li] = (form, kinds, (steps, idx), E, X)

        # grouped transitions: every input @ block / block @ input step's transition masses, a row of the boundary's
        # pairs at t_off. All kept at once when they fit TRANSITION_BUDGET; else each launch in chunks of steps that
        # fit, sharing one scratch (a chunk's masses are built just before its steps run)
        cols = {"input_block": (1, 2), "block_input": (4, 5)}               # (input row, position) columns
        ins = [(launches, li) for kind, _, layers in self.steps if kind == "prod" for stages, _, _ in layers
               for launches in stages for li, launch in enumerate(launches) if launch[0] in cols]
        tables = [launches[li][2].cpu().long() for launches, li in ins]
        if self.trans.grouped:
            pairs = torch.tensor(self.trans.num_pairs, dtype = torch.long)
            sizes = [pairs[st[:, cols[launches[li][0]][1]]] for st, (launches, li) in zip(tables, ins)]
        else:
            sizes = [torch.zeros(st.size(0), dtype = torch.long) for st in tables]
        self.trans_persistent = sum(int(z.sum()) for z in sizes) <= TRANSITION_BUDGET
        self.trans_size, rows = 0, []
        for st, z, (launches, li) in zip(tables, sizes, ins):
            form, kinds, _, E, X = launches[li]
            if not self.trans.grouped:
                chunks = [(0, st.size(0), None)]
            elif self.trans_persistent:
                st[:, 6] = self.trans_size + torch.cumsum(z, 0) - z
                self.trans_size += int(z.sum())
                chunks = [(0, st.size(0), None)]
                rows.append(st[:, [*cols[form], 6]])
            else:
                bounds, start, used = [], 0, 0
                for k, size in enumerate(z.tolist()):
                    if used + size > TRANSITION_BUDGET and k > start:
                        bounds.append((start, k))
                        start, used = k, 0
                    st[k, 6] = used
                    used += size
                    self.trans_size = max(self.trans_size, used)
                bounds.append((start, st.size(0)))
                chunks = [(s0, s1, transition_job(st[s0:s1][:, [*cols[form], 6]], dev)) for s0, s1 in bounds]
            launches[li] = (form, kinds, (st.to(dev, torch.int32).contiguous(), chunks), E, X)
        self.trans_job = transition_job(torch.cat(rows), dev) if self.trans.grouped and rows else None
        self._trans, self._trans_version = None, None

    def transition_buffer(self) -> torch.Tensor:
        """Where the transition masses go (all of them, or one chunk's; allocated on first use)."""
        if self._trans is None:
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

    def _product_stages(self, rows, dev):
        """
        One product layer's table, compiled into launches. Rows that share their children's kinds and
        boundaries fold the same way, right to left: ``c_0 @ (c_1 @ (... @ c_{k-1}))``, every intermediate
        block in a scratch row, the last step writing the layer's ``element_mars`` row. Each step is one
        contraction of :mod:`.kernels.prod`, chosen by its operands; a stage holds the steps that read only
        earlier stages, grouped into one launch per contraction and operand kinds.

        :returns: ``(stages, num_temps, max_temp_slots)``: per stage, ``(form, kinds, steps [S, F] int32, E_max,
            X_max)`` launches; the scratch rows the layer needs and the most slots per sample any of them holds
        """
        out_rows, child_rows, bounds = (t.cpu().long() for t in rows)
        firsts = torch.tensor(self.firsts, dtype = torch.long)
        groups = OrderedDict()
        for r in range(out_rows.numel()):
            k = int((child_rows[r] >= 0).sum())
            kids = child_rows[r, :k].tolist()
            groups.setdefault((tuple(c < self.input_end for c in kids), tuple(bounds[r, :k + 1].tolist())), []).append(r)

        stages = defaultdict(lambda: defaultdict(list))              # stage -> (form, kinds) -> [(fields, E, X)]
        temps = [0, 1]                                               # scratch rows so far, most slots per row
        zeros = None

        def block(rows_m):
            rows_m = rows_m.contiguous()
            return ("block", SUM, rows_m, torch.bucketize(rows_m, firsts, right = True) - 1)

        def scratch(R, entry, exit):
            first = temps[0]
            temps[0] += R
            temps[1] = max(temps[1], self._entry(entry) * self._exit(exit))
            return ("block", TEMP, torch.arange(first, first + R), torch.zeros(R, dtype = torch.long))

        def emit(stage, form, kinds, fields, entry, exit):
            stages[stage][(form, kinds)].append((torch.stack(fields, dim = 1), self._entry(entry), self._exit(exit)))

        for (is_input, bd), members in groups.items():
            idx = torch.tensor(members)
            R, k = idx.numel(), len(is_input)
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
                                                                    full(bd[k]), zeros], bd[m], bd[k])
                else:
                    _, l_kind, l_rows, l_regs = block(kids[:, m])
                    if acc[0] == "input":
                        emit(stage, "block_input", (l_kind, out_temp), [dest[2], l_rows, l_regs, full(bd[m]), acc[1],
                                                                        full(acc[2]), zeros], bd[m], bd[k])
                    else:
                        emit(stage, "block_block", (l_kind, acc[1], out_temp),
                             [dest[2], l_rows, l_regs, acc[2], acc[3], full(bd[m]), full(bd[m + 1]), full(bd[k])],
                             bd[m], bd[k])
                stage += 1
                acc = dest if m != 0 else acc

        out = []
        for stage in sorted(stages):
            launches = []
            for (form, kinds), parts in stages[stage].items():
                steps = torch.cat([f for f, _, _ in parts]).to(dev, torch.int32).contiguous()
                launches.append((form, kinds, steps, max(e for _, e, _ in parts), max(x for _, _, x in parts)))
            out.append(launches)
        return out, temps[0], temps[1]


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
    x = torch.where(missing, 0, data.to(dev).long())                     # missing tokens: any valid id
    bufs = cc._buffers(B)
    lay = bufs["layout"]
    node_mars, element_mars = bufs["node_mars"], bufs["element_mars"]
    input_mars = bufs["input_mars"]                                    # the input region: pyjuice's own layout
    regions = dict(offset = lay["sum_offsets_t"], width = lay["sum_widths_t"], first = prog.reg_first,
                   slots = prog.reg_slots)
    dot = PRECISIONS[precision]

    with torch.no_grad():
        for layer in pc.input_layer_group:
            layer(x.permute(1, 0), input_mars, missing_mask = missing)
        class_mars = cc._class_masses()
        prog.transitions(class_mars, cc._class_mars_version)
        obs_class = torch.where(missing, -1, cc.token_classes.token_class[x]).contiguous()

        for kind, prod_index, layers in prog.steps:
            elem_first = cc.element_regions[prod_index][0]
            elem_width = lay["element_widths"][prod_index]
            if kind == "prod":
                for stages, num_temps, temp_slots in layers:
                    temp, temp_width = node_mars, 0                  # no scratch rows: never read
                    if num_temps > 0:
                        temp_width = _align(B * temp_slots)
                        temp = cc._storage_view("product_scratch", num_temps * temp_width)
                    products(stages, (temp, temp_width), prog, regions, element_mars, node_mars,
                             class_mars, obs_class, elem_first, elem_width, B)
            else:
                for block_size, parts in layers:
                    for dense, num_edges, fused in parts:
                        for (r, c0), blocks in dense.items():
                            region = (lay["sum_offsets"][r], lay["sum_widths"][r], prog.firsts[r])
                            dense_sum(node_mars, element_mars, pc.params, {c0: blocks}, num_edges, block_size, region,
                                      elem_first, elem_width, B * prog.slots[r], dot)
                        if fused is not None:
                            nids, cids, pids, nb_reg, max_slots, aliased = fused
                            fused_sum(node_mars, element_mars, pc.params, nids, cids, pids, nb_reg, regions, elem_first,
                                      elem_width, B, block_size, max_cols = B * max_slots, precision = dot,
                                      aliased = aliased)

        r = prog.root_region
        base = lay["sum_offsets"][r] + (root_first - cc.sum_regions[r][0]) * lay["sum_widths"][r]
        width = lay["sum_widths"][r]
        rows = base + torch.arange(root_end - root_first, device = dev)[:, None] * width
        return node_mars[rows + torch.arange(B, device = dev)[None, :]].t().contiguous()
