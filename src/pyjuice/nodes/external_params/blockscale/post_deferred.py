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
results either way) times the layers' own launches -- now after the last layer -- against the deferred ones, in their
own form and in `_forms`'s, into scratch outputs; the fastest is kept. Where it is a deferred one, the backwards that
follow run each form in turn, timed by their periods (`FORM_TRIALS`), and keep the fastest. Inside a CUDA graph capture
an untimed group takes the layers' own launches, as does one whose pointer tables (made with a host copy) are not made
yet.

A layer defers only inside a scope -- a circuit's backward -- that allows it (`_defer_param_flows`: no layer callback,
which would see the node values and gradients between layers). What waits is safe to: nothing of the pass reads a
gated layer's node rows after its own backward (the layers below read their own rows, the products below them read
theirs and the inputs'), its gradient rows or `W` -- all read after the pass, which ends with the deferred work.
"""

from __future__ import annotations

import contextlib
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

# device index -> the stream a form of `_forms` launches the `W` kernel on, beside the gradient's
_SIDE = dict()

#: Backwards a shape's deferred kernels run in each of their forms -- their own and `_forms`'s: the same bits -- in turn,
#: once the tuner chose one of them, each timed by its backward's period (CUDA events: its launch to the next backward's,
#: the whole step), before the fastest form is kept (the best of each form's) where it is at least `FORM_MARGIN` faster
#: than their own; after `FORM_WARM` backwards in the tuner's form -- those deferred_par's side-stream placement times
#: its own in (its `SIDE_WARM` and twice `SIDE_TRIALS`: they change a period by several percent). The tuner times the
#: forms back to back into scratch outputs, their inputs in L2; at the end of a pass they come mostly from DRAM, and the
#: order can change: MEASURED (tied gated HMMs, bf16; each form forced in one process) HMM-4096 in 4 blocks B 128 at the
#: plan's node tiles of 64, `_forms`'s -0.9% / -0.4% / +1.0% a training step, the tuner taking the second (in another
#: process +1.1% a step). Nor is the launch's own time enough where a step waits for its host: a form's side stream
#: costs ~25 us of host -- HMM-2048 in 4 blocks B 64, the launch 49 -> 35 us in the pass but +1.3% a step.
FORM_TRIALS, FORM_WARM, FORM_MARGIN = 3, 9, 0.005

# the tuner's key -> its shape's `_FormTrials`
_FORM_TRIALS = dict()


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
                forms, ran = _forms(m0), set()

                def bench(c):
                    if c == "layers":
                        for m in members:
                            _launch_layer(self.owner, m, nm, grad, ws.get(_ptr(m["W"])))
                    else:
                        _launch_deferred(members, tabs, nm, grad, ws, None if c == "deferred" else c)
                        ran.add(c)

                choice = autotune.pick(key, ["layers", "deferred"] + forms, bench, margin = 1.0)
                del nm, grad, ws
                # (then, a deferred form chosen, the forms that launched timed in the backwards that follow)
                forms = [f for f in forms if f in ran]
                if choice != "layers" and forms and FORM_TRIALS > 0 and key not in _FORM_TRIALS:
                    _FORM_TRIALS[key] = _FormTrials([None] + forms)
            else:
                choice = autotune.cached(key) or "layers"
            if autotune.cached(key) is not None:
                self.owner.__dict__.setdefault("_bs_post_choice", {})[self.shape] = choice
        if choice == "layers":
            for m in members:
                _launch_layer(self.owner, m, m["node_mars"], m["grad_ext"], m["W"])
            return
        form = None if choice == "deferred" else choice
        trials = _FORM_TRIALS.get(key)
        if trials is not None:
            form = trials.next(form)
        _launch_deferred(members, tabs, members[0]["node_mars"], members[0]["grad_ext"],
                         {p: members[i]["W"] for p, i in zip(tabs["w_keys"], tabs["w_first"])}, form)


class _FormTrials:
    """A shape's deferred kernels run in each of their forms (None: their own) in turn in its backwards, each timed by
    its backward's period, `FORM_TRIALS` times each."""

    def __init__(self, forms):
        self.forms, self.warm, self.chosen, self.marks = forms, FORM_WARM, None, []

    def next(self, pick):
        """The form of this backward's launch: `pick` (the tuner's) while warming up or in a capture, else the next one
        timed, or the one kept once each was."""
        if self.chosen is not None:
            return self.chosen[0]
        if torch.cuda.is_current_stream_capturing():
            return pick
        if self.warm > 0:
            self.warm -= 1
            return pick
        mark = torch.cuda.Event(enable_timing = True)
        mark.record()
        self.marks.append(mark)
        done = [[] for _ in self.forms]                         # (each period: a timed launch to the next one)
        for k, (a, b) in enumerate(zip(self.marks, self.marks[1:])):
            if b.query():
                done[k % len(self.forms)].append(a.elapsed_time(b))
        if min(len(d) for d in done) >= FORM_TRIALS:
            best = min(range(len(self.forms)), key = lambda i: min(done[i]))
            if min(done[best]) > (1 - FORM_MARGIN) * min(done[0]):
                best = 0
            self.chosen, self.marks = (self.forms[best],), None
            return self.chosen[0]
        return self.forms[(len(self.marks) - 1) % len(self.forms)]


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


def _forms(m0) -> list:
    """The deferred launch's other forms -- ``(gate tiles a gradient program, the gradient's batch tile, W's node
    tile)`` --, the bits of its own: `W` at a node tile of 64 -- the M dimension of its dots, free -- launched on a side
    stream beside the gradient's launch (forked before it, joined after both); and that with the gradient over up to 4
    gate tiles a program (`kernels/blockscale_post_backward_deferred_wide.py`) at a batch tile of 16 or of 32 -- the N
    dimension of its dots, free. MEASURED (tied gated HMMs, 15 layers, bf16; the gradient's launch then `W`'s, the
    plan's node tiles 32 / 64 -> the fastest form): HMM-4096 in 4 blocks, 128 gates a block, B 128 230 / 192 -> 167 /
    168 us, B 512 830 / 801 -> 656 / 770; 256 gates a block, B 128 427 -> 319; 64, 127 / 125 -> 97 / 102; HMM-2048 in
    2 blocks B 256 125 / 135 -> 97 / 102, in 4 blocks B 64 53 / 45 -> 43; HMM-1024 B 128 33 -> 31 -- `W` at node tiles
    of 64 the fastest at each but HMM-1024's (within 1 us), the wide gradient's batch tile 16 at B 128 and 32 at B 512
    / 256; over 8 gate tiles slower everywhere (and over shared memory at node tiles of 64). The dot form only
    (`USE_DOT`)."""
    consts = m0["plan"][1]
    if not consts["USE_DOT"]:
        return []
    w_tm = min(64, consts["BLOCK_SIZE_M"]) if consts["WANT_W"] else 0
    forms = [(0, consts["TILE_B"], w_tm)] if w_tm else []
    g_tiles = triton.cdiv(consts["n_gates"], consts["TILE_G"])
    if consts["WANT_GRAD"] and g_tiles > 1:
        ngt = min(4, triton.next_power_of_2(g_tiles))
        forms += [(ngt, 16, w_tm), (ngt, 32, w_tm)]
    return forms


def _launch_deferred(members, tabs, node_mars, grad_ext, ws, form = None) -> None:
    """The deferred kernels over every member (their outputs given: ``ws`` by each `W`'s address), in their own form
    (``form`` None) or one of `_forms`."""
    from ..kernels.blockscale_post_backward_deferred import _bs_post_deferred_grad_kernel, _bs_post_deferred_w_kernel
    m0 = members[0]
    grid, consts, _ = m0["plan"]
    rows, bs_ = m0["rows"], consts["BLOCK_SIZE_M"]
    want_grad = bool(consts["WANT_GRAD"])
    common = _shape(m0)[2]
    log_z0 = m0["log_z"]
    t = tabs["all"]
    ngt, tb, w_tm = form or (0, consts["TILE_B"], consts["TILE_M"])
    if consts["WANT_W"]:
        tw, w_ptrs = tabs["w"], tabs["w_ptrs"]
        if any(int(p) != _ptr(ws[int(p)]) for p in tabs["w_keys"]):        # (scratch `W`s: a table of their own)
            w_ptrs = torch.tensor([_ptr(ws[int(p)]) for p in tabs["w_keys"]], dtype = torch.int64, device = log_z0.device)
    main = side = None
    if form is not None and w_tm:
        main = torch.cuda.current_stream(log_z0.device)
        side = _SIDE.get(log_z0.device.index)
        if side is None:
            side = _SIDE[log_z0.device.index] = torch.cuda.Stream(device = log_z0.device)
        side.wait_stream(main)
    gargs = (m0["node_flows"], node_mars, t["lz"], t["sg"] if want_grad else t["lz"], m0["ext"], t["gate"], t["nids"],
             t["ext_base"], log_z0 if grad_ext is None else grad_ext)
    if ngt:
        from ..kernels.blockscale_post_backward_deferred_wide import _bs_post_deferred_grad_wide_kernel
        wide = {k: v for k, v in common.items() if k != "USE_DOT"}
        wide.update(TILE_B = tb, num_warps = 4)
        _bs_post_deferred_grad_wide_kernel[(len(members) * rows,
                                            triton.cdiv(triton.cdiv(consts["batch_size"], tb), consts["B_TILES_PER_PROG"]),
                                            triton.cdiv(grid[1], ngt))](
            *gargs, ROWS = rows, NGT = ngt, GRAD_ATOMIC = tabs["grad_atomic"], **wide)
    else:
        _bs_post_deferred_grad_kernel[(len(members) * rows, grid[1] if want_grad else 1, grid[2])](
            *gargs, ROWS = rows, WANT_GRAD = int(want_grad), GRAD_ATOMIC = tabs["grad_atomic"], **common)
    if consts["WANT_W"]:
        with contextlib.nullcontext() if side is None else torch.cuda.stream(side):
            _bs_post_deferred_w_kernel[(rows * (bs_ // w_tm), grid[1], len(tabs["w_keys"]))](
                m0["node_flows"], tw["lz"], m0["ext"], tw["gate"], tw["nids"], tw["ext_base"], w_ptrs, tw["off"],
                N_SLICES = grid[2], **(common if side is None else dict(common, TILE_M = w_tm, num_warps = 4)))
    if side is not None:
        main.wait_stream(side)


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
