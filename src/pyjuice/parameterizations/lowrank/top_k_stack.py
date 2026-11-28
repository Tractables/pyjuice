"""
STACKED passes of `LowRank(top_k)` sums (`top_k_fused`'s kernels): the work of a pass that no other layer's values
enter, made at once for every sum reading a leaf layer, and each sum's call reduced to the launches of the rest.

A sum node vector's forward is two parts. One reads only its own tensors and the shared parameters: its fp16 factors
(BP), its token rows (E16), theta's tables (LTh) -- the masked samples' slot weights and the work items are its leaf's,
which the leaf's forward made for every position. The other reads the children's values, which the layer before it in
the pass writes: their values over their largest (ev, cmax), then the node values (`tk_sk_kernel`) and, where a backward
may follow, its stash. Where such sums follow one another (a sequence model's positions), the second part alone is on
the chain of layers.

A STACK (`SumStack`) makes the first part for every reading sum of the leaf layer in one launch, into buffers allocated
once along with every buffer the sums' launches write, and keeps each reader's launch arguments in its plan
(`TopKStackPlan`, the sums' module): a reader's forward is one call into the module, `plan.forward(i)` -- two launches,
no tensor made. Where a backward may follow, the backward's own chain-independent operands (fp16 A, theta's backward
tables) are made with them, and a backward that asks for no gradients nor EM statistics (a query's) is one call too,
`plan.backward(i, ...)`; any other backward reads the stack's stash (`SumStack.kept`). A stack is made by the first
reading sum's call of a pass and kept in the leaf's pass cache (`top_k_leaf_fused.pass_cache`: made again by every
forward, so nothing in it outlives the tensors it was made from); it serves the calls of the pass it was made for
(`SumStack.serves`: the same buffers, staged tensors and options). Any other call, and any reader whose kernels do not
serve, takes the unstacked path (`top_k_fused.sum_forward`, `top_k_backward.terms`), the reference the stack's
operands are bit for bit (the same device code).

A layer whose node vectors a stack took gets a settled forward and backward (`settled_forward`, `settled_backward`:
`ParameterizedSumLayer`'s contract): its next calls go straight to the pass's stack -- a position's host time is what
the small batches' passes take.

Where the circuit hands a sum layer the product layer group below (`stream_children`: that group's passes not run --
LowRank's `stream_fusion_plan`, `stream_fusion_bk`), a reader's launches read each child's value as its product's (the
node values of its two children, summed: `tk_ev_kernel`) and store each child's flow as its product's children's node
flows (`tk_scatter_ef_kernel`) -- the children's rows composed once a node vector and table
(`top_k_fused.child_pairs`), so the reads are as deep as the element values'.

The readers of a leaf layer are the compile step's (`_resolve_sibling_leaves`: `_sum_readers`).
"""

from __future__ import annotations

from typing import Optional

import torch

from . import top_k_fused, top_k_leaf_fused

#: the stacked passes where they serve; off, every sum's forward takes the unstacked path (`top_k_fused.sum_forward`)
STACKED = True

_CACHE_KEY = "stack"               # (its entry in the leaf's pass cache: a SumStack, or False -- none serves this pass)
_SETTLED = "_lr_top_k_settled"     # (a layer's record for its settled calls: its node vectors and their leaves)

_MODS = {}


def _mods():
    """`(top_k, top_k_backward)` (imported when first asked for: they import this module's)"""
    got = _MODS.get("m")
    if got is None:
        from . import top_k, top_k_backward
        got = _MODS["m"] = (top_k, top_k_backward)
    return got


class SumStack:
    """A pass's stacked forward of the sums reading a leaf layer (the module's notes): its plan, each reader's index in
    it, and what a call must be to be one of its pass."""

    __slots__ = ("plan", "index", "keep", "node_mars", "element_mars", "staged", "emap", "precision", "grad", "taken",
                 "left")

    def __init__(self, plan, index: dict, keep: bool, node_mars, element_mars, kwargs: dict):
        top_k = _mods()[0]
        self.plan, self.index, self.keep = plan, index, keep
        self.node_mars, self.element_mars = node_mars, element_mars
        # (the call's options a stack's launches were made for: `serves`)
        self.staged, self.emap = top_k._staged(kwargs), top_k.external_rows(kwargs)
        self.precision, self.grad = kwargs.get("precision", "auto"), bool(kwargs.get("_grad_enabled", False))
        self.taken = set()          # (the readers whose kept operands a backward took: once)
        self.left = set(index.values())     # (the readers not launched yet: where no backward follows, the buffers
                                            #  are given back after the last -- the unstacked forward's footprint)

    def serves(self, node_mars, element_mars, kwargs: dict) -> bool:
        """Whether a call is one of the pass the stack was made for: its buffers, staged tensors, map of rows, options."""
        top_k = _mods()[0]
        return (node_mars is self.node_mars and element_mars is self.element_mars and
                top_k._staged(kwargs) is self.staged and top_k.external_rows(kwargs) is self.emap and
                kwargs.get("precision", "auto") == self.precision and
                bool(kwargs.get("_grad_enabled", False)) == self.grad)

    def takes(self, ns) -> bool:
        """Whether the stack takes the node vector `ns`'s forward: one of its readers, its buffers not given back (a
        reader's second call of a pass where no backward follows)."""
        return self.plan is not None and id(ns) in self.index

    def forward(self, ns, pairs = None) -> bool:
        """The node vector `ns`'s forward by its launches (True), or False: the stack does not take it (`takes`).
        `pairs`: the product layer group below handed over (`top_k_fused.child_pairs`: its forward not run), each
        child's value read as its product's."""
        i = self.index.get(id(ns))
        if i is None or self.plan is None:
            return False
        if pairs is None:
            self.plan.forward(i)
        else:
            self.plan.forward(i, pairs)
        self.left.discard(i)
        if not self.left and not self.keep:
            self.plan.release()
            self.plan = None
        return True

    def keeps(self, ns) -> bool:
        """Whether the stack takes the node vector `ns`'s backward (`backward`): one of its readers, its stash kept and
        not taken yet."""
        i = self.index.get(id(ns))
        return i is not None and self.keep and i not in self.taken and self.plan is not None

    def backward(self, ns, node_flows, element_flows, store: bool, pairs = None) -> bool:
        """The node vector `ns`'s backward by its launches where it asks for no gradients nor EM statistics (a query's:
        the caller's to check) -- the flows into its stash (its leaf's counts beside the slots' posterior), its element
        flows into `element_flows` (`store`: their one writer of the pass, not filled before) or, with `pairs` (the
        product layer group below handed over: `forward`'s; `store` then), stored as its products' children's node
        flows --, True; False where the stack does not take it (not a reader, nothing kept, or its stash taken already:
        a second backward)."""
        i = self.index.get(id(ns))
        if i is None or not self.keep or i in self.taken or self.plan is None:
            return False
        self.taken.add(i)
        if pairs is None:
            self.plan.backward(i, node_flows, element_flows, store)
        else:
            self.plan.backward(i, node_flows, element_flows, store, pairs)
        return True

    def kept(self, ns):
        """`(ops, stash)`: the forward's operands and stash of the reader `ns` as `top_k_fused.kept_of` hands them over
        (taken: a backward uses them once), or None (not one of the readers, nothing kept, or taken already)."""
        i = self.index.get(id(ns))
        if i is None or not self.keep or i in self.taken or self.plan is None:
            return None
        self.taken.add(i)
        got = self.plan.kept(i)
        return got[:11], got[11:]


def stack_of(sib: dict, B: int, kwargs: dict, node_mars, element_mars, params) -> Optional[SumStack]:
    """The pass's stack of the sums reading the leaf `sib`'s layer (`_sibling_leaves`) -- made by the first reader's
    call -- where it serves this call; else None (the unstacked path)."""
    if not STACKED:
        return None
    cache = top_k_leaf_fused.pass_cache(sib, B)
    if cache is None:
        return None
    got = cache.get(_CACHE_KEY)
    if got is None:
        got = cache[_CACHE_KEY] = _make(sib["layer"], B, kwargs, node_mars, element_mars, params, cache) or False
    return got if got and got.serves(node_mars, element_mars, kwargs) else None


def stack_made(sib: dict, B: int) -> Optional[SumStack]:
    """The pass's stack of the leaf `sib`'s layer if one was made (none made here)."""
    cache = top_k_leaf_fused.pass_cache(sib, B)
    got = None if cache is None else cache.get(_CACHE_KEY)
    return got or None


def _node_info(layer, ns):
    """`ns`'s `ExternalNodeInfo` in `layer` (kept on the layer)."""
    infos = layer.__dict__.get("_lr_top_k_infos")
    if infos is None:
        infos = layer.__dict__["_lr_top_k_infos"] = {id(info.ns): info for info in layer.external_node_infos}
    return infos.get(id(ns))


def _make(leaf_layer, B: int, kwargs: dict, node_mars, element_mars, params, cache: dict) -> Optional[SumStack]:
    """The stack of the leaf layer's readers (`_sum_readers`) whose kernels serve this call -- their tensors staged, the
    leaf's forward state their slots, weights and work items --, its prep launched (where a backward may follow, the
    backward's too); None where none serves."""
    from pyjuice.layer.parameterized_sum_layer import StagedExternalParams
    top_k, top_k_backward = _mods()
    st = top_k_leaf_fused._state(leaf_layer, B)
    readers = leaf_layer.__dict__.get("_sum_readers")
    staged = top_k._staged(kwargs)
    mod = top_k_fused.top_k_module()
    if (st is None or not readers or mod is None or not isinstance(staged, StagedExternalParams)
            or element_mars.stride(1) != 1 or "_inputs" not in kwargs):
        return None
    dev, K = node_mars.device, st["K"]
    emap = top_k.external_rows(kwargs)
    index_of = top_k_leaf_fused._plan(leaf_layer)["index"]
    picked = []
    for layer, ns in readers:
        desc, tensors, info = ns.parameterization, staged.get(ns), _node_info(layer, ns)
        if tensors is None or info is None or desc.top_k != K:
            continue
        geo = desc._geometry(layer, info, dev)
        if not top_k_fused.serves(desc, geo["M"], geo["N"], tensors[0].size(2), node_mars, kwargs) or \
                (emap is None and tensors[0].size(0) != B):
            continue
        sib = layer.__dict__["_sibling_leaves"][id(ns)]
        picked.append((layer, ns, desc, info, geo, tensors, sib, index_of[id(sib["ns"])]))
    if not picked:
        return None
    # (a backward may follow -- the call had grad on, or a query's -- and its kernels serve every reader: the stash kept)
    keep = bool(kwargs.get("_grad_enabled", False)) and all(
        top_k_backward.serves(desc, geo["M"], geo["N"], tensors[0].size(2), B, dev, kwargs, False)
        for _, _, desc, _, geo, tensors, _, _ in picked)
    # the parameters' sources (tied node vectors read one) -- their dense theta the pass's tables', shared with the
    # backward (`top_k_fused.theta_tables`)
    tabs, src = [], []
    for layer, ns, desc, info, geo, tensors, sib, t in picked:
        tab = top_k_fused.theta_tables(desc, sib, ns, geo, params, B, cache)
        s = next((k for k, u in enumerate(tabs) if u is tab), None)
        if s is None:
            s = len(tabs)
            tabs.append(tab)
        src.append(s)
    flat, _ = top_k_leaf_fused.work_items(st, None if emap is None else emap.long().contiguous(), cache)
    counts = top_k_leaf_fused._counts(st) if keep else torch.empty(0, device = dev)
    if keep:
        for *_, t in picked:                     # (their rows of the leaf's counts the sums' kernels' to write)
            if st["offs"][t + 1] > st["offs"][t]:
                st["written"].add(t)
    none_b = torch.empty(0, dtype = torch.bool, device = dev)
    plan = mod.TopKStackPlan(
        [p[5][0] for p in picked], [p[5][1] for p in picked], [p[5][2] for p in picked],
        [p[4]["eid"] for p in picked], [none_b if p[4]["full"] else p[4]["has"] for p in picked],
        [p[3].nid_start for p in picked], [p[7] for p in picked], src, [u["theta"] for u in tabs],
        *flat, st["logw"], st["offs"], counts, node_mars, element_mars, keep)
    for s, tab in enumerate(tabs):               # (the forward's table, for the readers the stack does not take)
        tab.setdefault("LTh", plan.lth(s))
    if keep:                                     # (the backward's operands too: made while the pass's start is
        plan.prepare_backward([p[4]["cols"] for p in picked], [p[4]["kids"] for p in picked], st["rows"])  # host-bound
    return SumStack(plan, {id(p[1]): i for i, p in enumerate(picked)}, keep, node_mars, element_mars, kwargs)


def settle(layer, taken) -> None:
    """Put the settled forward and backward on `layer` (`settled_forward`, `settled_backward`), every node vector of it
    a stack took: `taken` their `(ns, sib)`."""
    layer.__dict__[_SETTLED] = tuple(taken)
    layer.__dict__["_settled_forward"] = settled_forward
    layer.__dict__["_settled_backward"] = settled_backward


def settled_forward(layer, node_mars, element_mars, params, propagation_alg, op_params, kwargs) -> bool:
    """`ParameterizedSumLayer._settled_forward` of a `LowRank(top_k)` layer whose node vectors a stack took (`settle`):
    at a call whose every node vector a stack of its pass takes, what the full path does -- its checks (the algorithm,
    whole partitions, the tokens given; the tensors the staging's) and settings (`_lr_state`, `_lr_topk_batch`:
    `top_k.sum_forward`'s), the product layer group below handed over to the launches (`stream_children`: LowRank's
    forward takes it, `fuses_stream_children`) -- by their launches, True; else False, nothing done."""
    rec = layer.__dict__.get(_SETTLED)
    if rec is None or propagation_alg != "LL" or kwargs.get("partition_runs") is not None or "_inputs" not in kwargs:
        return False
    B = node_mars.size(1)
    stacks = []
    for ns, sib in rec:
        stack = stack_of(sib, B, kwargs, node_mars, element_mars, params)
        if stack is None or not stack.takes(ns):
            return False
        stacks.append(stack)
    layer.__dict__["_lr_state"] = dict()
    layer.__dict__["_lr_topk_batch"] = B
    sc = kwargs.get("stream_children")
    for (ns, _), stack in zip(rec, stacks):
        stack.forward(ns, top_k_fused.child_pairs(layer, ns, sc))
    return True


def settled_backward(layer, node_flows, element_flows, node_mars, element_mars, params, param_flows, denom_param_flows,
                     propagation_alg, op_params, kwargs) -> bool:
    """`ParameterizedSumLayer._settled_backward` of a `LowRank(top_k)` layer whose node vectors a stack took (`settle`):
    at a call asking for no gradients nor EM statistics (a query's) whose every node vector its pass's stack takes
    (`SumStack.backward`), what the full path does -- `ParameterizedSumLayer.backward`'s checks (the algorithm, whole
    partitions, flows in log space neither modified nor negated; the tensors the staging's) and settings (its gated
    call, the product layer group handed over -- LowRank's backward always takes it, `stream_fusion_bk` --, the copy of
    the parameters), the deferral's side stream joined, a launch-plan recording around it failed (none of this is
    recorded); `LowRank.pre_backward_layer`'s (the forward of this batch); `top_k.sum_backward`'s element flows (with
    the product layer group, its children's node flows: where each child's flow is one node vector's, `_kids_own`) --
    by their launches, True; else False, nothing done."""
    from pyjuice.layer import deferred_par
    from pyjuice.layer.parameterized_sum_layer import StagedExternalParams, _NO_BW_HOOKS
    from pyjuice.utils.launch_plans import LaunchPlans
    rec = layer.__dict__.get(_SETTLED)
    if rec is None or param_flows is not None or propagation_alg != "LL" or kwargs.get("partition_runs") is not None \
            or not kwargs.get("logspace_flows", False) or kwargs.get("allow_modify_flows", False) \
            or kwargs.get("negate_pflows", False):
        return False
    top_k = _mods()[0]
    B = node_mars.size(1)
    if not isinstance(top_k._staged(kwargs), StagedExternalParams) or layer.__dict__.get("_lr_topk_batch") != B or \
            element_flows.dtype != torch.float32 or element_flows.stride(1) != 1 or \
            node_flows.dtype != torch.float32 or node_flows.stride(1) != 1:
        return False
    gviews = top_k._grads_kw(kwargs)[0]
    stacks = []
    for ns, sib in rec:
        stack = stack_made(sib, B)
        if (gviews is not None and gviews.get(ns) is not None) or stack is None or not stack.keeps(ns):
            return False
        stacks.append(stack)
    # (`top_k.sum_backward`'s: the children's element flows each node vector's alone, else filled first and added into
    # -- with the product layer group below, the full path's: the element flows added up, then the children's)
    desc = layer.parameterization
    alone = desc._kids_own(layer, node_flows.device)
    sc = kwargs.get("stream_children")
    if sc is not None and not alone:
        return False
    # (the full path's settings for the call, its join and its recording's failure: `ParameterizedSumLayer.backward`)
    op = op_params if op_params is not None and desc.reads_param_operands and \
        layer.__dict__.get("_reads_param_operands", False) else None
    layer.__dict__.update(_bk_gated_call = True, _bk_stream_children = sc, _bk_op_params = op)
    deferred_par.join_side(node_mars.device)
    if LaunchPlans.nested():
        LaunchPlans.fail()
    if not alone:
        for ns, _ in rec:
            element_flows.index_fill_(0, desc._geometry(layer, _node_info(layer, ns), node_flows.device)["kids"],
                                      -float("inf"))
    for (ns, _), stack in zip(rec, stacks):
        stack.backward(ns, node_flows, element_flows, alone, top_k_fused.child_pairs(layer, ns, sc))
    layer.__dict__.update(_NO_BW_HOOKS)
    return True
