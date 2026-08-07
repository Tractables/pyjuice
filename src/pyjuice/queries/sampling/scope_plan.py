"""
The sampler's frontier layout, derived from the circuit instead of from the draw.

The top-down pass keeps a frontier of selected nodes in a `[rows, num_samples]` buffer. Today a row
is whatever the previous layer's slot allocation happened to hand out, so finding a layer's entries
needs a `torch.where`, keeping the buffer dense needs a compaction, and handing out slots needs a
running per-column cursor -- per layer, every call. MEASURED on a `PD` circuit at batch 512, that
bookkeeping is 92% of the pass's GPU time and ~80% of its wall time; the sampling kernels themselves
are 0.064 ms of 4.15 ms.

None of it is necessary. A frontier entry always stands for a SCOPE, and which scopes a layer owns is
a property of the circuit. Giving every (layer, scope) a fixed row makes the whole layout structural:

    * a layer's rows are known at compile time  -> no `torch.where`
    * rows never move                           -> no compaction
    * a child's destination row is its scope's  -> no cursors, no prefix sums
    * every shape is fixed                      -> capturable in a CUDA graph, for ANY circuit

Liveness becomes a mask rather than a shape: a row holds `-1` when its scope is not on this sample's
path. That is what the existing kernels already do with an unmatched node id, so an inactive row
costs a masked-off lane and nothing else.

:note: the cost is those masked lanes. MEASURED, lanes processed against lanes live: structured
       decomposable circuits pay 1.00x (their live set IS the scope set), `PD` 3.49x and `RAT-SPN`
       3.80x. That multiplies the 8% and removes the 92%, which is why it is worth paying.

:note: the row counts this produces are exactly the buffer sizes the driver already allocates
       (`_num_nscopes` / `_num_escopes`) -- verified on HMM, HCLT and PD. The layout is a different
       discipline for using that space, not a new demand on it.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Sequence, Tuple


class ScopePlan():
    """
    Fixed frontier rows for one compiled circuit.

    Attributes:
        `num_node_rows`:  rows needed in `node_samples`
        `num_elem_rows`:  rows needed in `element_samples`
        `sum_rows`:       `id(layer) -> LongTensor[n]`, the `node_samples` rows that sum layer owns
        `sum_erows`:      `id(layer) -> LongTensor[n]`, where each of those rows' drawn child lands in
                          `element_samples`. A sum node's children share its scope, so this is that
                          scope's element row in the product group below.
        `prod_rows`:      `id(layer) -> LongTensor[n]`, the `element_samples` rows that product layer
                          owns
        `prod_crows`:     `id(layer) -> LongTensor[rows, num_edges]` per partition, the
                          `node_samples` row each child slot writes to -- the row of THAT CHILD's
                          scope. This is what lets a circuit whose decomposition varies with the draw
                          be addressed without a cursor, and `-1` marks a padded slot.
    """

    __slots__ = ("num_node_rows", "num_elem_rows", "sum_rows", "sum_erows", "prod_rows",
                 "prod_crows", "root_row")

    def __init__(self):
        self.num_node_rows = 0
        self.num_elem_rows = 0
        self.root_row = -1                  # where the pass seeds the root node
        self.sum_rows: Dict[int, torch.Tensor] = {}
        self.sum_erows: Dict[int, torch.Tensor] = {}
        self.prod_rows: Dict[int, torch.Tensor] = {}
        self.prod_crows: Dict[int, List[torch.Tensor]] = {}

    def to(self, device):
        """Move every table onto `device`; they are indexed by kernels, not by the host."""
        for table in (self.sum_rows, self.sum_erows, self.prod_rows):
            for key, value in table.items():
                table[key] = value.to(device)
        for key, value in self.prod_crows.items():
            self.prod_crows[key] = [v.to(device) for v in value]
        return self


def _scope_key(scope) -> Tuple[int, ...]:
    """
    A hashable, canonical scope.

    NOT the `BitSet` itself: its `__hash__` is taken over raw bytes while its `__eq__` ignores
    trailing zeros, so two equal scopes of different byte length hash apart and would be given
    separate rows.
    """
    return tuple(sorted(scope))


def build_scope_plan(pc) -> ScopePlan:
    """
    Assign every (layer, scope) a frontier row and tabulate where each product child writes.

    Node rows are global: a node frontier spans layers, because a sum layer's output is consumed by
    the product layer below it. Element rows are per layer GROUP, since `element_samples` is reset at
    the start of each group and groups therefore reuse the same space.
    """
    plan = ScopePlan()

    # ---- node rows, in the order the driver sizes `num_nscopes`: input group first, then each sum
    # group bottom-up. `scope key -> row`, per layer, so a child lookup can find its own scope's row.
    node_row_of_scope: Dict[Tuple[int, ...], int] = {}
    ns_ranges: List[Tuple[int, int, int]] = []          # (nid_start, nid_end, row)

    cursor = 0
    for layer_group in pc.layers(ret_layer_groups = True):
        if layer_group.is_prod():
            continue

        for layer in layer_group:
            rows = []
            for scope in layer.scopes:
                key = _scope_key(scope)
                # One row per (layer, scope). The same scope can appear in several layers -- a tied
                # HMM transition, say -- and those are different frontier entries at different depths.
                node_row_of_scope[key] = cursor
                rows.append(cursor)
                cursor += 1

            if layer_group.is_sum():
                plan.sum_rows[id(layer)] = torch.tensor(rows, dtype = torch.long)

            # The pass seeds the root at its own scope's row rather than at row 0
            for ns in getattr(layer, "nodes", []):
                if ns is pc.root_ns:
                    plan.root_row = node_row_of_scope[_scope_key(ns.scope)]

            # Where each of this layer's nodes lives, so a product child can be resolved to its row
            for ns in getattr(layer, "nodes", []):
                key = _scope_key(ns.scope)
                if key in node_row_of_scope:
                    lo, hi = ns._output_ind_range
                    ns_ranges.append((lo, hi, node_row_of_scope[key]))

    plan.num_node_rows = cursor

    # ---- element rows, per product layer group, plus the sum layers that feed them
    ns_ranges.sort()
    starts = torch.tensor([r[0] for r in ns_ranges], dtype = torch.long)
    ends = torch.tensor([r[1] for r in ns_ranges], dtype = torch.long)
    owner_rows = torch.tensor([r[2] for r in ns_ranges], dtype = torch.long)

    groups = list(pc.inner_layer_groups)
    for gid, layer_group in enumerate(groups):
        if not layer_group.is_prod():
            continue

        elem_row_of_scope: Dict[Tuple[int, ...], int] = {}
        cursor = 0
        for layer in layer_group:
            rows = []
            for scope in layer.scopes:
                key = _scope_key(scope)
                # ONE row per scope for the whole GROUP, not per (layer, scope). Several product
                # layers of a group can own the same scope -- an ordinary mixture of two sub-circuits
                # with different block sizes puts them in different layers -- and the sum layer above
                # has a single destination for that scope. Giving each layer its own row made
                # `sum_erows` point at one of them while the other layer's kernel scanned the other,
                # so that branch's elements were written and never read: samples silently came back
                # with variables missing (zero-filled), 2915 of 4096 on a 12-variable mixture.
                #
                # Sharing is safe because the layers read the row through their own `nids`: an element
                # belonging to the other layer matches nothing and is masked off, which is the same
                # mechanism that already lets one layer's partitions share a row.
                if key not in elem_row_of_scope:
                    elem_row_of_scope[key] = cursor
                    cursor += 1
                rows.append(elem_row_of_scope[key])
            plan.prod_rows[id(layer)] = torch.tensor(rows, dtype = torch.long)

            # Each child slot writes to the row of ITS OWN scope
            crows = []
            for partition_id in range(layer.num_fw_partitions):
                cids = layer.partitioned_cids[partition_id].cpu()
                pos = (torch.searchsorted(starts, cids, right = True) - 1).clamp(min = 0)
                row = owner_rows[pos]
                # A padded slot is the dummy child 0, and a child outside its candidate's range means
                # the search fell off the front; both are `-1` so the kernel can skip them.
                valid = (cids > 0) & (cids >= starts[pos]) & (cids < ends[pos])
                crows.append(torch.where(valid, row, torch.full_like(row, -1)))
            plan.prod_crows[id(layer)] = crows

        plan.num_elem_rows = max(plan.num_elem_rows, cursor)

        # The sum group ABOVE this one draws children that land in these element rows
        if gid + 1 < len(groups) and groups[gid + 1].is_sum():
            for layer in groups[gid + 1]:
                erows = [elem_row_of_scope.get(_scope_key(scope), -1) for scope in layer.scopes]
                plan.sum_erows[id(layer)] = torch.tensor(erows, dtype = torch.long)

    return plan
