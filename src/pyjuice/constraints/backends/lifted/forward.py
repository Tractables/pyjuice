"""
The lifted forward pass: ``log p(C, e)`` for a batch of evidence, in the buffers of a
:class:`~pyjuice.constraints.ConstrainedCircuit`.

Layer by layer, as :meth:`TensorCircuit.forward`: the PC's own input layers write the log-probability of
every observed token, the class masses of missing ones are summed per token class
(:mod:`.kernels.inputs`), then every product layer group chains its children's blocks
(:mod:`.kernels.prod`) and every sum layer group sums its children column by column (:mod:`.kernels.sum`).
The root keeps a single value per sample.

Everything the pass reads besides the buffers and the parameters is built once per constrained circuit
(:class:`Program`), from the plan's tables and the PC's own layer metadata.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from typing import Optional

import torch

from .kernels.inputs import class_masses, input_class_tables
from .kernels.prod import chain, input_suffix, transition_table
from .kernels.sum import dense_groups, dense_sum, fused_sum

#: Matrix-product precision of the kernels: "fp32" (three TF32 products, fp32-level accuracy) or "tf32".
PRECISIONS = {"fp32": "tf32x3", "tf32": "tf32"}


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
        self.token_class = layout.token_class.to(dev, torch.long)
        self.input_start, self.input_end = cc.input_range
        self.input_tables = input_class_tables(pc, int(layout.token_class.numel()))

        firsts = [first for first, _, _ in cc.sum_regions]
        self.firsts = firsts
        self.slots = [slots for _, _, slots in cc.sum_regions]
        self.reg_first = torch.tensor(firsts, dtype = torch.long, device = dev)
        self.reg_slots = torch.tensor([slots for _, _, slots in cc.sum_regions], dtype = torch.long, device = dev)
        slots_of_reg = [slots for _, _, slots in cc.sum_regions]
        region_of = lambda row: bisect.bisect_right(firsts, row) - 1

        self.transitions = {}
        self.steps = []
        prod_index = -1
        prod_layers = iter(cc.product_rows)
        for lg in pc.inner_layer_groups:
            if lg.is_prod():
                prod_index += 1
                layers = []
                for layer in lg.layers:
                    tables = next(prod_layers)
                    layers.append((self._input_suffix(tables.get("input_suffix"), region_of),
                                   self._chain(tables.get("chain"), region_of, dev)))
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
                                     max(slots_of_reg[r] for r in rest_regs))
                        parts.append((dense, cids.size(1), fused))
                    layers.append((layer.block_size, parts))
                self.steps.append(("sum", prod_index, layers))

        root_first, root_end = pc._root_node_range
        self.root_region = region_of(root_first)
        self.root_rows = (root_first, root_end)

    def _input_suffix(self, rows, region_of):
        if rows is None:
            return None
        out_rows, child_rows, boundaries = rows
        suf = child_rows[:, 1]
        regs = torch.tensor([region_of(int(r)) for r in suf.tolist()], dtype = torch.int32, device = suf.device)
        starts = boundaries[:, 0].contiguous()
        max_slots = max(self.width[a] for a in set(starts.tolist()))
        return (out_rows, child_rows[:, 0].contiguous(), suf.contiguous(), starts, regs), max_slots

    def _chain(self, rows, region_of, dev):
        if rows is None:
            return None
        out_rows, child_rows, boundaries = (t.cpu().long() for t in rows)
        groups = OrderedDict()
        for r in range(out_rows.numel()):
            k = int((child_rows[r] >= 0).sum())
            kids = child_rows[r, :k].tolist()
            bounds = tuple(boundaries[r, :k + 1].tolist())
            kinds = tuple("input" if row < self.input_end else "sum" for row in kids)
            groups.setdefault((kinds, bounds), []).append(r)
        out = []
        for (kinds, bounds), members in groups.items():
            idx = torch.tensor(members)
            children = []
            for m, kind in enumerate(kinds):
                rows_m = child_rows[idx, m]
                regs = None if kind == "input" else torch.tensor([region_of(int(r)) for r in rows_m.tolist()],
                                                                   dtype = torch.long, device = dev)
                if kind == "input" and bounds[m] not in self.transitions:
                    self.transitions[bounds[m]] = transition_table(self.next_col, bounds[m], self.n, self.width)
                children.append((kind, rows_m.to(dev), regs, bounds[m], bounds[m + 1]))
            out.append((out_rows[idx].to(dev), children))
        return out


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
    input_mars, class_mars = bufs["input_mars"], bufs["class_mars"]
    regions = dict(offset = lay["sum_offsets_t"], width = lay["sum_widths_t"], first = prog.reg_first,
                   slots = prog.reg_slots)
    dot = PRECISIONS[precision]

    with torch.no_grad():
        for layer in pc.input_layer_group:
            layer(x.permute(1, 0), input_mars, missing_mask = missing)
        class_masses(prog.input_tables, prog.token_class, cc.num_classes, class_mars, prog.input_start)
        obs_class = torch.where(missing, -1, prog.token_class[x]).to(torch.int32).contiguous()

        for kind, prod_index, layers in prog.steps:
            elem_first = cc.element_regions[prod_index][0]
            elem_width = lay["element_widths"][prod_index]
            if kind == "prod":
                for suffix, chains in layers:
                    if suffix is not None:
                        table, max_slots = suffix
                        input_suffix(element_mars, node_mars, input_mars, class_mars, obs_class, prog.next_col,
                                     prog.width_t, table, regions, elem_first, elem_width, B, n, prog.input_start,
                                     max_cols = B * max_slots)
                    if chains is not None:
                        chain(element_mars, node_mars, input_mars, class_mars, obs_class, chains, prog.transitions,
                              regions, prog.width, elem_first, elem_width, B, n, prog.input_start)
            else:
                for block_size, parts in layers:
                    for dense, num_edges, fused in parts:
                        for (r, c0), blocks in dense.items():
                            region = (lay["sum_offsets"][r], lay["sum_widths"][r], prog.firsts[r])
                            dense_sum(node_mars, element_mars, pc.params, {c0: blocks}, num_edges, block_size, region,
                                      elem_first, elem_width, B * prog.slots[r], dot)
                        if fused is not None:
                            nids, cids, pids, nb_reg, max_slots = fused
                            fused_sum(node_mars, element_mars, pc.params, nids, cids, pids, nb_reg, regions, elem_first,
                                      elem_width, B, block_size, max_cols = B * max_slots, precision = dot)

        r = prog.root_region
        base = lay["sum_offsets"][r] + (root_first - cc.sum_regions[r][0]) * lay["sum_widths"][r]
        width = lay["sum_widths"][r]
        rows = base + torch.arange(root_end - root_first, device = dev)[:, None] * width
        return node_mars[rows + torch.arange(B, device = dev)[None, :]].t().contiguous()
