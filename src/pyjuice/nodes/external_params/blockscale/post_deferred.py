"""
BlockScale's post-backward -- the log-Z half of the gate gradient, the denominator flow `W` of `apply_z_correction` and
the `-log Z` shift that gives a gated layer's node values back (`BlockScaleSumParams._post_fused`) -- left by each
layer of a backward to the end of its inner layers (`pyjuice.utils.deferred`) and run there for all of them.

Per layer it is one launch of `_bs_post_backward_kernel`: a few microseconds of work in short programs, the gradient
summed over a node block's tiles through stored shares and tickets (MEASURED, a tied gated HMM-4096 at B 128: 25 us a
layer, 15 layers a step). At the end of the pass the layers' work is one launch of
`_bs_post_deferred_grad_kernel` (the gradient and the shift; each program walks a node block's tiles, no shares) and
one of `_bs_post_deferred_w_kernel` per `W` (each program walks the layers; one writer, no atomics) --
`kernels/blockscale_post_backward_deferred.py`, whose tiles compute what the per-layer kernel's compute: the
gradient and the shifted node values are the same bits; `W` the bits of one order of the adds the per-layer kernel
makes atomically where a layer spans several batch slices (its own bits where it spans one). Where two of the layers'
node blocks write one gate row -- copies of a node whose gates are tied (`tie_external`), which read the same rows --
the gradient's adds are atomic, in no fixed order (the per-layer launches add them one layer after another).

One more option of the tuners: at a group's first end of pass, `autotune.pick` (keyed by shape; margin 1.0: the same
results either way) times the layers' own launches -- now after the last layer -- against the deferred ones, into
scratch outputs; the faster is kept. Inside a CUDA graph capture an untimed group takes the layers' own launches, as
does one whose pointer tables (made with a host copy) are not made yet.

A layer defers only inside a scope -- a circuit's backward -- that allows it (`_defer_param_flows`: no layer callback,
which would see the node values and gradients between layers). What waits is safe to: nothing of the pass reads a
gated layer's node rows after its own backward (the layers below read their own rows, the products below them read
theirs and the inputs'), its gradient rows or `W` -- all read after the pass, which ends with the deferred work.
"""

from __future__ import annotations

import itertools

import torch
import triton

from pyjuice.utils.deferred import current_scope

# (device, the members' pointers and bases) -> their tables on the device: the addresses themselves, so a key is
# its own content (and holds no tensor); made outside a capture (a host copy), bounded
_TABLES = dict()
_MAX_TABLES = 256

# id(plan) -> (the plan, its shape key, the deferred kernels' constant arguments): made once per plan (a partition's,
# kept by its layer), so a backward builds no key; the plan held, so its id is not reused while the entry lives
_SHAPES = dict()



def _shape(member):
    plan = member["plan"]
    hit = _SHAPES.get(id(plan))
    if hit is None or hit[0] is not plan:
        grid, consts, _ = plan
        key = (tuple(sorted((k, v) for k, v in consts.items() if k != "ext_base")), tuple(grid), member["rows"],
               member["nids"].dtype, member["gb"].dtype)
        common = dict(batch_size = consts["batch_size"], n_gates = consts["n_gates"],
                      N_CHILD_GATES = consts["N_CHILD_GATES"], BLOCK_SIZE_M = consts["BLOCK_SIZE_M"],
                      gate_stride = consts["gate_stride"], TILE_M = consts["TILE_M"], TILE_G = consts["TILE_G"],
                      TILE_B = consts["TILE_B"], B_TILES_PER_PROG = consts["B_TILES_PER_PROG"],
                      USE_DOT = consts["USE_DOT"], NIDS_I64 = int(member["nids"].dtype == torch.int64),
                      GATE_I64 = int(member["gb"].dtype == torch.int64), num_warps = consts["num_warps"],
                      num_stages = consts["num_stages"])
        if len(_SHAPES) >= 4096:
            _SHAPES.clear()
        hit = _SHAPES[id(plan)] = (plan, key, common)
    return hit


def try_defer(owner, layer, member: dict) -> bool:
    """Leave this partition's post-backward (``member``: its tensors and its plan, as `_post_fused` launches them) to
    the end of the pass and return True; or return False, the caller launching it now."""
    if not layer.__dict__.get("_defer_param_flows", False):
        return False
    scope = current_scope()
    if scope is None:
        return False
    shape = _shape(member)[1]
    # (the tuners' choice for groups of this shape, once made, kept with the parameterization: "layers" -- each partition
    # then launches at its own `_post_fused`, deferring nothing: the same bits, no registration)
    if owner.__dict__.get("_bs_post_choice", {}).get(shape) == "layers":
        return False
    key = (id(owner), shape, member["node_flows"].data_ptr(), member["node_mars"].data_ptr(),
           member["ext"].data_ptr(), _ptr(member["grad_ext"]))
    groups = scope.__dict__.setdefault("_bs_post_groups", {})
    group = groups.get(key)
    if group is None:
        group = groups[key] = _PostGroup(owner, shape)
        scope.defer(group.flush)
    group.members.append(member)
    return True


def _ptr(t):
    return 0 if t is None else t.data_ptr()


class _PostGroup:
    """The partitions of one backward that deferred their post-backward with one plan's shape, in the pass's order."""

    def __init__(self, owner, shape):
        self.owner, self.shape, self.members = owner, shape, []

    def flush(self) -> None:
        members, self.members = self.members, []
        if not members:
            return
        from pyjuice.layer.kernels import autotune
        tabs = _tables(members)
        choice = "layers"
        if tabs is not None:
            m0 = members[0]
            key = ("bs_post_deferred", len(members), self.shape, tabs["w_sizes"])
            if autotune.cached(key) is None and autotune.should_tune(key):
                nm = torch.empty_like(m0["node_mars"])
                grad = None if m0["grad_ext"] is None else torch.zeros_like(m0["grad_ext"])
                ws = {p: torch.zeros_like(members[i]["W"]) for p, i in zip(tabs["w_keys"], tabs["w_first"])}

                def bench(c):
                    if c == "layers":
                        for m in members:
                            _launch_layer(self.owner, m, nm, grad, ws.get(_ptr(m["W"])))
                    else:
                        _launch_deferred(members, tabs, nm, grad, ws)

                choice = autotune.pick(key, ["layers", "deferred"], bench, margin = 1.0)
                del nm, grad, ws
            else:
                choice = autotune.cached(key) or "layers"
            if autotune.cached(key) is not None:
                self.owner.__dict__.setdefault("_bs_post_choice", {})[self.shape] = choice
        if choice == "deferred":
            _launch_deferred(members, tabs, members[0]["node_mars"], members[0]["grad_ext"],
                             {p: members[i]["W"] for p, i in zip(tabs["w_keys"], tabs["w_first"])})
        else:
            for m in members:
                _launch_layer(self.owner, m, m["node_mars"], m["grad_ext"], m["W"])


def _launch_layer(owner, m, node_mars, grad_ext, W) -> None:
    """The partition's own launch, as `_post_fused` makes it (its outputs given)."""
    from ..kernels.blockscale_post_backward import _bs_post_backward_kernel
    grid, consts, shares = m["plan"]
    log_z = m["log_z"]
    if shares is not None:
        partial, counter = owner._post_scratch(log_z.device, *shares)
    else:
        partial = counter = log_z
    _bs_post_backward_kernel[grid](
        node_flows = m["node_flows"], node_mars = node_mars, log_z = log_z, sigma = m["sigma"], ext = m["ext"],
        gate = m["gb"], nids = m["nids"], grad_ext = log_z if grad_ext is None else grad_ext,
        W = log_z if W is None else W, partial = partial, counter = counter, **consts)


def _launch_deferred(members, tabs, node_mars, grad_ext, ws) -> None:
    """The deferred kernels over every member (their outputs given: ``ws`` by each `W`'s address)."""
    from ..kernels.blockscale_post_backward_deferred import _bs_post_deferred_grad_kernel, _bs_post_deferred_w_kernel
    m0 = members[0]
    grid, consts, _ = m0["plan"]
    rows, bs_ = m0["rows"], consts["BLOCK_SIZE_M"]
    want_grad = bool(consts["WANT_GRAD"])
    common = _shape(m0)[2]
    log_z0 = m0["log_z"]
    t = tabs["all"]
    _bs_post_deferred_grad_kernel[(len(members) * rows, grid[1] if want_grad else 1, grid[2])](
        m0["node_flows"], node_mars, t["lz"], t["sg"] if want_grad else t["lz"], m0["ext"], t["gate"], t["nids"],
        t["ext_base"], log_z0 if grad_ext is None else grad_ext, ROWS = rows, WANT_GRAD = int(want_grad),
        GRAD_ATOMIC = tabs["grad_atomic"], **common)
    if consts["WANT_W"]:
        tw, w_ptrs = tabs["w"], tabs["w_ptrs"]
        if any(int(p) != _ptr(ws[int(p)]) for p in tabs["w_keys"]):        # (scratch `W`s: a table of their own)
            w_ptrs = torch.tensor([_ptr(ws[int(p)]) for p in tabs["w_keys"]], dtype = torch.int64, device = log_z0.device)
        _bs_post_deferred_w_kernel[(rows * (bs_ // consts["TILE_M"]), grid[1], len(tabs["w_keys"]))](
            m0["node_flows"], tw["lz"], m0["ext"], tw["gate"], tw["nids"], tw["ext_base"], w_ptrs, tw["off"],
            N_SLICES = grid[2], **common)


def _tables(members):
    """The members' pointer and base tables on their device -- ``all`` of them; and ``w``: the members grouped by their
    `W` (``w_keys`` the `W`s' addresses, in order of first use; ``off`` each group's first row; ``w_first`` a member of
    each), each group in the pass's order; ``grad_atomic``: whether two of their node blocks write one gate row
    (`post_rows_overlap`) --, or None inside a capture where they are not made yet."""
    m0 = members[0]
    dev = m0["log_z"].device
    rows_of = lambda m: (m["log_z"].data_ptr(), _ptr(m["sigma"]) if m["grad_ext"] is not None else 0,
                         m["gb"].data_ptr(), m["nids"].data_ptr(), int(m["plan"][1]["ext_base"]), _ptr(m["W"]))
    key = (dev, tuple(rows_of(m) for m in members))
    hit = _TABLES.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        return None

    def table(ix):
        rs = [rows_of(members[i]) for i in ix]
        mk = lambda c: torch.tensor([r[c] for r in rs], dtype = torch.int64, device = dev)
        return dict(lz = mk(0), sg = mk(1), gate = mk(2), nids = mk(3), ext_base = mk(4))

    w_groups = dict()
    if m0["W"] is not None:
        for i, m in enumerate(members):
            w_groups.setdefault(_ptr(m["W"]), []).append(i)
    order = [i for ix in w_groups.values() for i in ix]
    w = table(order)
    w["off"] = torch.tensor([0] + list(itertools.accumulate(len(ix) for ix in w_groups.values())), dtype = torch.int64,
                            device = dev)
    c = m0["plan"][1]
    if m0["grad_ext"] is None:
        grad_atomic = 0
    else:
        from .backward import post_rows_overlap
        grad_atomic = int(post_rows_overlap([(m["gb"], int(m["plan"][1]["ext_base"])) for m in members], c["n_gates"],
                                            c["N_CHILD_GATES"]))
    hit = dict(all = table(range(len(members))), w = w, w_keys = list(w_groups), w_first = [ix[0] for ix in w_groups.values()],
               w_sizes = tuple(len(ix) for ix in w_groups.values()),
               w_ptrs = torch.tensor(list(w_groups), dtype = torch.int64, device = dev), grad_atomic = grad_atomic)
    if len(_TABLES) >= _MAX_TABLES:
        _TABLES.clear()
    _TABLES[key] = hit
    return hit
