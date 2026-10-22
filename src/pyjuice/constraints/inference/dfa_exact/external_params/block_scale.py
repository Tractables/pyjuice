"""
Sum nodes gated per sample (:class:`pyjuice.nodes.external_params.BlockScaleSumParams`) under a constraint.

A gated sum node ``n`` of sample ``b`` weighs child ``c`` by ``theta[n, c] phi_b[g(n, c)] / Z_b[n]``, ``g(n, c)`` the
gate of ``n``'s node block and ``c``'s child gate, ``Z_b[n] = sum_c theta[n, c] phi_b[g(n, c)]`` -- the library's
forward under ``pc(x, sum_external_params = ...)``. The gates are constant over the automaton's columns, so the lifted
sum adds ``log phi_b[g(n, c)]`` to child ``c``'s values and takes ``log Z_b[n]`` off node ``n``'s, column by column;
the top-down pass takes ``log Z`` off a parent's values and multiplies a parent block's sum by its gates (the kernel
helpers below). The normalizers depend on the sample and the parameters only: computed once per session
(:func:`log_normalizers`). Node gates of a node block or finer (then a node block's rows read several: the sum kernels
scale their products per node row, see :mod:`..kernels.sum`); child gates of a child block or finer.

A session's gates (:func:`session_inputs`) follow the library's rules: one tensor ``[batch, node gates, child gates]``
of log gates per node group, the copies of a tied node reading their own unless the gates are tied (``tie_external``:
the source's); a layer given gates for some of its node groups only is refused. They are kept in one layout fixed per
PC (:func:`cells`): every gate tensor's cells at an offset of its own, a row per sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl


def refusal(ns) -> Optional[str]:
    """Why a gated node group cannot be served under a constraint, or None (every one can)."""
    return None


@dataclass
class Cells:
    """
    Where every gated node group's gates are in a session's rows of them (:func:`cells`).

    :ivar base: storage owner (see ``BlockScaleSumParams.storage_owner``) -> its first cell
    :ivar shape: storage owner -> ``(node gates, child gates)``
    :ivar size: the cells of a row
    :ivar layers: the gated sum layers, by gated layer index (the PC's order)
    """

    base: dict
    shape: dict
    size: int
    layers: list


def cells(pc) -> Cells:
    """The cell layout of a PC's gated node groups (:class:`Cells`)."""
    base, shape, size, layers = {}, {}, 0, []
    for lg in pc.inner_layer_groups:
        for layer in lg.layers if lg.is_sum() else ():
            if getattr(layer, "external_params", None) is None:
                continue
            layers.append(layer)
            for ns in layer.nodes:
                owner = ns.external_params.storage_owner(ns)
                if owner not in base:
                    _, Nk, Ck = ns.external_params.tensor_shapes(owner, 1)[0]
                    base[owner], shape[owner] = size, (Nk, Ck)
                    size += Nk * Ck
    return Cells(base, shape, size, layers)


def edge_tables(layout: Cells, layer, nids: torch.Tensor, cids: torch.Tensor, elem_first: int, dev) -> dict:
    """
    The gate of every edge of a gated sum layer's node blocks (``nids`` [K], ``cids`` [K, E]: their first node rows and
    children): ``idx`` int32 [K, E] on the device, its cell (its node block's first row's); ``ck`` int32 [K] on the
    device, the grid's row length of every node block's node group -- the cells one node gate apart -- and ``node_gate``
    the layer's node gate size; for the top-down tables, every node block's grid row ``row`` (the cell of its first
    row's node gate and child gate 0, int64 [K]), its node group's child gate size ``cbs`` (int64 [K]) and every edge's
    child's place among the node group's children ``lc`` (int64 [K, E]; the children concatenated in ``ns.chs`` order),
    on the host; ``layer``: the layer's index among the gated ones. Padding edges (children below ``elem_first``) read
    cell 0.
    """
    nodes = sorted(layer.nodes, key = lambda ns: ns._output_ind_range[0])
    starts = np.asarray([ns._output_ind_range[0] for ns in nodes], dtype = np.int64)
    nid_h, cid_h = nids.cpu().long().numpy(), cids.cpu().long().numpy()
    K, E = cid_h.shape
    row, cbs, lc = np.zeros(K, dtype = np.int64), np.ones(K, dtype = np.int64), np.zeros((K, E), dtype = np.int64)
    ck = np.zeros(K, dtype = np.int64)
    node_gates = {ns.external_params.gate_sizes(ns)[0] for ns in nodes}
    assert len(node_gates) == 1, node_gates          # a layer's nodes share their block size and gate signature
    for k, at in enumerate(np.searchsorted(starts, nid_h, side = "right") - 1):
        ns = nodes[at]
        owner = ns.external_params.storage_owner(ns)
        gate_bs, gate_cbs = ns.external_params.gate_sizes(ns)
        ck[k] = layout.shape[owner][1]
        row[k] = layout.base[owner] + (nid_h[k] - starts[at]) // gate_bs * ck[k]
        cbs[k] = gate_cbs
        first = np.asarray([cs._output_ind_range[0] for cs in ns.chs], dtype = np.int64)
        if np.unique(first).size != first.size:
            raise NotImplementedError("A gated sum node group whose children list a node group twice is not "
                                      "supported under a constraint.")
        cum = np.cumsum([0] + [cs.num_nodes for cs in ns.chs])[:-1]
        order = np.argsort(first)
        c = cid_h[k]
        j = order[np.clip(np.searchsorted(first[order], c, side = "right") - 1, 0, None)]
        lc[k] = np.where(c >= elem_first, cum[j] + c - first[j], 0)
    idx = np.where(cid_h >= elem_first, row[:, None] + lc // cbs[:, None], 0)
    i32 = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev, torch.int32)
    return dict(idx = i32(idx), ck = i32(ck), ck_host = ck, node_gate = node_gates.pop(), row = row, cbs = cbs, lc = lc,
                layer = next(k for k, gl in enumerate(layout.layers) if gl is layer),
                uniform = uniform_chunks(idx, cid_h >= elem_first))


def dense_run(nids: torch.Tensor, cids: torch.Tensor, nb_reg: torch.Tensor, block_size: int,
              elem_first: int) -> Optional[dict]:
    """
    Whether a sum part under external parameters is one dense run -- every node block over the same ``E`` contiguous
    children (no padding edge), the blocks consecutive rows of one sum region -- so that a sample's weights
    (:func:`sample_weights`) make one matrix, and its sums one matrix product: ``dict(c0, E, nid0, rows, reg)`` (host
    ints; ``rows`` the run's node rows from ``nid0``), else None.
    """
    nid_h, cid_h, reg_h = nids.cpu().long().numpy(), cids.cpu().long().numpy(), nb_reg.cpu().long().numpy()
    K, E = cid_h.shape
    c0, nid0 = int(cid_h[0, 0]), int(nid_h[0])
    if c0 < elem_first or not (cid_h == c0 + np.arange(E)[None, :]).all() or (reg_h != reg_h[0]).any() or \
            not (nid_h == nid0 + block_size * np.arange(K)).all():
        return None
    return dict(c0 = c0, E = E, nid0 = nid0, rows = K * block_size, reg = int(reg_h[0]))


@triton.jit
def _sample_weights_kernel(out, weights, pids, nids, idx, ck, values, gstride, lz, lzstride, b, E, num_rtiles,
                           BS: tl.constexpr, NG: tl.constexpr, TILE_R: tl.constexpr, TILE_E: tl.constexpr):
    """One program per (node block, tile of its rows, tile of its edges): ``out[k * BS + i, e] = theta * exp(log
    gate - log Z)`` of node block ``k``'s row ``i`` and edge ``e`` for sample ``b`` -- its weight ``weights[pids[k, e] +
    i]``, its cell ``idx[k, e]`` (plus ``i // NG`` node gates of ``ck[k]`` cells each, node gates finer than the block),
    its row's ``log Z`` (a row of no mass: 0, its weights are 0) -- stored as ``out``'s dtype (bf16: to nearest)."""
    k = tl.program_id(0) // num_rtiles
    i = (tl.program_id(0) % num_rtiles) * TILE_R + tl.arange(0, TILE_R)
    e = tl.program_id(1) * TILE_E + tl.arange(0, TILE_E)
    im, em = i < BS, e < E
    mask = im[:, None] & em[None, :]
    at = k.to(tl.int64) * E + e
    pid = tl.load(pids + at, mask = em, other = 0).to(tl.int64)
    w = tl.load(weights + pid[None, :] + i[:, None], mask = mask, other = 0.0)
    cell = tl.load(idx + at, mask = em, other = 0)[None, :] + tl.zeros([TILE_R, 1], dtype = tl.int32)
    if NG < BS:
        cell += (i // NG)[:, None] * tl.load(ck + k)
    g = tl.load(values + b * gstride + cell, mask = mask, other = 0.0)
    z = tl.load(lz + b * lzstride + tl.load(nids + k).to(tl.int64) + i, mask = im, other = 0.0)
    z = tl.where(z == float("-inf"), 0.0, z)
    tl.store(out + (k * BS + i).to(tl.int64)[:, None] * E + e[None, :], w * tl.exp(g - z[:, None]), mask = mask)


def sample_weights(out: torch.Tensor, params: torch.Tensor, pids: torch.Tensor, nids: torch.Tensor, gate: dict,
                   gated: dict, b: int, block_size: int):
    """
    Sample ``b``'s weights of a dense run under external parameters (:func:`dense_run`) into ``out`` [K * block_size,
    E] (fp32, or bf16 rounded to nearest): ``theta[n, c] phi_b[g(n, c)] / Z_b[n]``, normalized per node as the sums
    take them (see the module docstring) -- from the part's weights ``pids`` [K, E] and node blocks ``nids`` [K], its
    edges' cells (``gate``: :func:`edge_tables`) and the session's gates and log normalizers (``gated``:
    :func:`~..queries.backend.forward.part_gates`; ``b`` 0 when they are every sample's).
    """
    K, E = pids.shape
    TILE_R, TILE_E = min(64, triton.next_power_of_2(block_size)), 64
    num_rtiles = triton.cdiv(block_size, TILE_R)
    _sample_weights_kernel[(K * num_rtiles, triton.cdiv(E, TILE_E))](
        out, params, pids, nids, gate["idx"], gate["ck"], gated["values"], gated["gstride"], gated["lz"],
        gated["lzstride"], b, E, num_rtiles, BS = block_size, NG = gate["node_gate"], TILE_R = TILE_R, TILE_E = TILE_E,
        num_warps = 4)


#: the chunk sizes the sum kernels may take, or split a chunk into, for which :func:`uniform_chunks` /
#: :func:`uniform_tiles` say whether a chunk's (a tile's) gate is one
GATE_CHUNKS = (1, 2, 4, 8, 16, 32, 64, 128)


def uniform_chunks(idx: np.ndarray, valid: np.ndarray) -> dict:
    """Chunk size -> whether every aligned chunk of that many edges of every node block (``idx`` [K, E], the edges'
    cells, ``valid`` the real ones) reads one gate cell: the forward then reads one gate per chunk and column."""
    out = {}
    K, E = idx.shape
    for T in GATE_CHUNKS:
        pad = (-E) % T
        i = np.pad(idx, ((0, 0), (0, pad))).reshape(K, -1, T)
        v = np.pad(valid, ((0, 0), (0, pad))).reshape(K, -1, T)
        out[T] = bool(np.all(~v | (i == i[:, :, :1])))
    return out


def uniform_tiles(lc: np.ndarray, cbs: np.ndarray, length: int) -> dict:
    """Tile size -> whether every aligned tile of that many children of a child block of ``length`` children falls in
    one child gate for every parent block (``lc`` its first child's place among its node group's children, ``cbs`` its
    child gate size): the top-down pass then reads one gate per parent block and column."""
    out = {}
    for T in GATE_CHUNKS:
        s = lc[:, None] + np.arange(0, length, T)[None, :]
        e = lc[:, None] + np.minimum(np.arange(0, length, T) + T, length)[None, :] - 1
        out[T] = bool(np.all(s // cbs[:, None] == e // cbs[:, None]))
    return out


@dataclass
class SessionGates:
    """
    A decoder session's gates (:func:`session_inputs`).

    :ivar values: [batch, cells] fp32 on the device: every sample's log gates in the PC's layout (:class:`Cells`; 0
        where no gate is given)
    :ivar batch: their batch size (1: every sample's)
    :ivar layers: bool [gated sum layers] (host): those given gates, run gated
    :ivar scopes: the intervals ``(a, b)`` (boundaries) of the gated node groups' scopes
    :ivar nodes: the gated node groups
    """

    values: torch.Tensor
    batch: int
    layers: np.ndarray
    scopes: list
    nodes: list = field(default_factory = list)
    _logz: Optional[torch.Tensor] = None
    _minor: Optional[torch.Tensor] = None


def sample_minor(gates: SessionGates) -> torch.Tensor:
    """
    The gates' log-scales sample-minor: fp32 [cells, batch rounded up to 4] on the device, cell ``k`` of sample ``b``
    at ``[k, b]`` (0 past the batch) -- a cell's samples side by side, as the CUDA fused sums read a tile's range of
    them, 16 bytes at a time. Built on first use, then kept with the gates (a batch of one: ``values`` itself).
    """
    if gates._minor is None:
        if gates.batch == 1:
            gates._minor = gates.values.view(-1, 1)
        else:
            minor = torch.zeros(gates.values.size(1), -(-gates.batch // 4) * 4, dtype = torch.float32,
                                device = gates.values.device)
            minor[:, :gates.batch] = gates.values.t()
            gates._minor = minor
    return gates._minor


def session_inputs(pc, layout: Cells, value) -> SessionGates:
    """``sum_external_params`` of a decoder session, checked against the PC (see the module docstring), in its rows."""
    from pyjuice.layer.external_sum_layer import validate_external_tensors
    if not isinstance(value, dict) or len(value) == 0:
        raise ValueError(f"`sum_external_params` must be a non-empty dict of node groups (or registered groups of them) "
                         f"to their gates, got {type(value).__name__}.")
    first = next(iter(value.values()))
    first = first[0] if isinstance(first, (tuple, list)) and len(first) > 0 else first
    if not torch.is_tensor(first) or first.dim() == 0:
        raise ValueError(f"`sum_external_params`' values must be tensors [batch, node gates, child gates], got "
                         f"{type(first).__name__}.")
    batch = int(first.size(0))
    try:
        for key in value:
            if not isinstance(key, str) and key not in pc.external_params_nodes:
                raise AssertionError(f"`sum_external_params` names a node group with no external parameters in this "
                                     f"PC: {key}.")
        given = pc._expand_external_params_groups(dict(value), batch)
        given = {ns: validate_external_tensors(ns, ns.external_params, t, batch, pc.params.device,
                                               require_contiguous = False)[0] for ns, t in given.items()}
    except AssertionError as e:
        raise ValueError(str(e)) from None
    owners = {ns.external_params.storage_owner(ns): t for ns, t in given.items()}
    layers = np.zeros(len(layout.layers), dtype = bool)
    gated = []
    for k, layer in enumerate(layout.layers):
        on = [ns.external_params.storage_owner(ns) in owners for ns in layer.nodes]
        if any(on) and not all(on):
            raise NotImplementedError(f"`sum_external_params` gates {sum(on)} of the {len(on)} gated node groups of a "
                                      f"sum layer: as in the library, a layer's nodes are gated all or none.")
        layers[k] = all(on)
        gated += list(layer.nodes) if layers[k] else []
    values = torch.zeros(batch, max(1, layout.size), dtype = torch.float32, device = pc.params.device)
    for owner, t in owners.items():
        b0 = layout.base[owner]
        values[:, b0:b0 + t[0].numel()] = t.reshape(batch, -1)
    scopes = sorted({(min(ns.scope.to_list()), max(ns.scope.to_list()) + 1) for ns in gated})
    return SessionGates(values = values, batch = batch, layers = layers, scopes = scopes, nodes = gated)


def log_normalizers(pc, layout: Cells, gates: SessionGates) -> torch.Tensor:
    """
    ``log Z_b[n]`` of every gated node row (see the module docstring) and sample: fp32 [batch, rows] on the device, by
    the PC's node row (0 for an ungated row) -- ``log sum_g phi_b[g] sigma[n, g]`` over node ``n``'s gates ``g``,
    ``sigma[n, g]`` the sum of its weights in ``g``. Computed on first use, then kept with the gates (the parameters do
    not change within a session).
    """
    if gates._logz is not None:
        return gates._logz
    dev = pc.params.device
    out = torch.zeros(gates.batch, pc.num_nodes, dtype = torch.float32, device = dev)
    groups = {}                         # the node groups of one parameter table and gate sizes: one sigma, one pass
    for ns in gates.nodes:
        groups.setdefault((ns.get_source_ns(), ns.external_params.gate_sizes(ns)), []).append(ns)
    for (_, (gate_bs, gate_cbs)), members in groups.items():
        lsig = _log_gate_sums(pc, members[0], gate_cbs)                         # [nodes, Ck] float64
        Nk, Ck = layout.shape[members[0].external_params.storage_owner(members[0])]
        N = members[0].num_nodes
        node = torch.arange(N, device = dev)
        # Z = sum_g phi[g] sigma[n, g], each node's sigma and each gate row's phi shifted by its maximum: a product of
        # [samples, Ck] by [Ck, nodes] per gate row, in float64
        top = lsig.max(dim = 1).values
        top = torch.where(torch.isfinite(top), top, 0.0)                         # (a node of no edge: Z 0)
        sig = torch.zeros(Nk * gate_bs, Ck, dtype = torch.float64, device = dev)
        sig[:N] = torch.exp(lsig - top[:, None])
        sig = sig.view(Nk, gate_bs, Ck).transpose(1, 2)                          # [Nk, Ck, gate_bs]
        step = max(1, LOGZ_CHUNK // (gates.batch * Nk * Ck))
        for k in range(0, len(members), step):
            chunk = members[k:k + step]
            cells = torch.tensor([layout.base[ns.external_params.storage_owner(ns)] for ns in chunk], device = dev)
            lphi = gates.values[:, cells[:, None] + torch.arange(Nk * Ck, device = dev)[None, :]].double()
            lphi = lphi.view(gates.batch * len(chunk), Nk, Ck)                    # (sample, member) major
            m = lphi.max(dim = 2, keepdim = True).values
            m = torch.where(torch.isfinite(m), m, 0.0)                           # (a row of no gate: Z 0)
            tot = torch.bmm(torch.exp(lphi - m).transpose(0, 1), sig)            # [Nk, samples x members, gate_bs]
            tot = tot.transpose(0, 1).reshape(lphi.size(0), Nk * gate_bs)[:, :N]
            lz = torch.log(tot) + m[:, node // gate_bs, 0] + top[None]
            low = tot == 0                                                       # underflowed (or no mass): exactly
            if bool(low.any()):
                at, n = low.nonzero(as_tuple = True)
                lz[at, n] = torch.logsumexp(lphi[at, n // gate_bs] + lsig[n], dim = 1)
            rows = torch.tensor([ns._output_ind_range[0] for ns in chunk], device = dev)[:, None] + node[None, :]
            out[:, rows.reshape(-1)] = lz.view(gates.batch, -1).float()
    gates._logz = out
    return out


#: gates :func:`log_normalizers` works on at once (a chunk of node groups' every sample's)
LOGZ_CHUNK = 1 << 24


def _log_gate_sums(pc, ns, gate_cbs: int) -> torch.Tensor:
    """float64 [nodes, child gates]: ``log sigma[n, g]``, the sum of node ``n``'s weights over the children of child gate
    ``g`` (-inf where it has no edge)."""
    src = ns.get_source_ns()
    bs, cbs = ns.block_size, ns.ch_block_size
    psid, peid = src._param_range
    local = ((src._param_ids - psid) // (bs * cbs)).to(pc.params.device)
    blocks = pc.params[psid:peid].double().reshape(-1, cbs, bs)[local].permute(0, 2, 1)   # [edge blocks, bs, cbs]
    per = cbs // gate_cbs
    sums = blocks.reshape(blocks.size(0), bs, per, gate_cbs).sum(dim = 3)                 # [edge blocks, bs, per]
    Ck = ns.num_ch_nodes // gate_cbs
    nb, cb = (ns.edge_ids[k].to(pc.params.device).long() for k in (0, 1))
    sigma = torch.zeros(ns.num_nodes * Ck, dtype = torch.float64, device = pc.params.device)
    at = (nb[:, None, None] * bs + torch.arange(bs, device = sigma.device)[None, :, None]) * Ck + \
        cb[:, None, None] * per + torch.arange(per, device = sigma.device)[None, None, :]
    sigma.index_add_(0, at.reshape(-1), sums.reshape(-1))                       # an edge block listed twice adds twice
    return torch.log(sigma.view(ns.num_nodes, Ck))


# -------------------------------------------------------------------------------------------------
# Kernel helpers (the lifted sum kernels' ``GATED`` parts): every column's sample ``samp`` (int32), the gates
# ``gstride`` floats apart per sample; ``SHARED``: the gates are every sample's (no ``samp``), one per row or scalar
# -------------------------------------------------------------------------------------------------

@triton.jit
def _edge_gates(gates, gidx, offs, emask, samp, gstride, cmask, SHARED: tl.constexpr):
    """[edges, columns] (``SHARED``: [edges, 1]): the log gates of the edges whose cells are at ``gidx + offs``
    (``emask`` the real ones) -- 0 where masked."""
    g = tl.load(gidx + offs, mask = emask, other = 0)
    if SHARED:
        return tl.load(gates + g, mask = emask, other = 0.0)[:, None]
    else:
        return tl.load(gates + samp[None, :] * gstride + g[:, None], mask = emask[:, None] & cmask[None, :],
                       other = 0.0)


@triton.jit
def _chunk_gates(gates, gidx, offs, emask, samp, gstride, cmask, SHARED: tl.constexpr):
    """[columns] (``SHARED``: a scalar): the log gate of a chunk of edges that reads one cell -- the edges' cells at
    ``gidx + offs`` (``emask`` the real ones) loaded as their children are, and the largest taken (a padding edge's is
    0) -- 0 where masked."""
    g = tl.max(tl.load(gidx + offs, mask = emask, other = 0), axis = 0)
    if SHARED:
        return tl.load(gates + g)
    else:
        return tl.load(gates + samp * gstride + g, mask = cmask, other = 0.0)


@triton.jit
def _segment_cell(gidx, offs, seg):
    """The cell a segment of a chunk of edges reads (its edges' cells at ``gidx + offs``, ``seg`` the segment's real
    ones: they are one, the largest taken; 0 for none) -- of its node block's first row."""
    return tl.max(tl.load(gidx + offs, mask = seg, other = 0), axis = 0)


@triton.jit
def _row_gates(gates, cell, roff, samp, gstride, rmask, cmask, SHARED: tl.constexpr):
    """[rows, columns] (``SHARED``: [rows, 1]): the log gates of a tile of node rows whose node gates differ -- row
    ``r`` reads cell ``cell + roff[r]`` (``roff``: its node gate's offset in the grid from the block's first row's) --
    of every column's sample; 0 where masked."""
    if SHARED:
        return tl.load(gates + cell + roff, mask = rmask, other = 0.0)[:, None]
    else:
        return tl.load(gates + samp[None, :] * gstride + cell + roff[:, None], mask = rmask[:, None] & cmask[None, :],
                       other = 0.0)


@triton.jit
def _tile_gates(gates, samp, gstride, grow, lc, gcbs, cmask, SHARED: tl.constexpr):
    """[columns] (``SHARED``: a scalar): the log gate of a tile of children in one child gate -- its first child's place
    ``lc`` over the child gate size ``gcbs``, in the parent block's grid row ``grow`` -- 0 where masked."""
    if SHARED:
        return tl.load(gates + grow + lc // gcbs)
    else:
        return tl.load(gates + samp * gstride + grow + lc // gcbs, mask = cmask, other = 0.0)


@triton.jit
def _node_log_z(lz, samp, lzstride, rows, rmask, cmask, SHARED: tl.constexpr):
    """[nodes, columns] (``SHARED``: [nodes, 1]): ``log Z`` of the node rows ``rows`` (``rmask`` the real ones) for every
    column's sample -- 0 where masked."""
    if SHARED:
        return tl.load(lz + rows, mask = rmask, other = 0.0)[:, None]
    else:
        return tl.load(lz + samp[None, :] * lzstride + rows[:, None], mask = rmask[:, None] & cmask[None, :],
                       other = 0.0)


@triton.jit
def _add_gated_parent(m2, acc2, m, acc, gates, samp, gstride, grow, lc, gcbs, offs_c, rmask, cmask,
                      SHARED: tl.constexpr):
    """The online log-sum-exp per child and column ``(m2, acc2)`` with one parent block's sum ``(m, acc)`` -- shifted
    per column, as the sum kernels' online log-sum-exp leaves it -- times its gates: child ``offs_c``'s (``rmask`` the
    real ones) gate ``(lc + offs_c) // gcbs`` in the block's grid row ``grow``, of every column's sample (they vary
    with both, so the shift is per child and column)."""
    j = grow + (lc + offs_c) // gcbs
    if SHARED:
        g = tl.load(gates + j, mask = rmask, other = 0.0)[:, None]
    else:
        g = tl.load(gates + samp[None, :] * gstride + j[:, None], mask = rmask[:, None] & cmask[None, :], other = 0.0)
    t = tl.log(acc) + tl.where(m == float("-inf"), 0.0, m)[None, :] + g
    m_new = tl.maximum(m2, t)
    s = tl.where(m_new == float("-inf"), 0.0, m_new)
    return m_new, acc2 * tl.exp(m2 - s) + tl.exp(t - s)
