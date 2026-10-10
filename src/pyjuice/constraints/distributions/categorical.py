"""
Categorical input nodes: a class's mass is the sum of the probabilities of its tokens (pyjuice keeps Categorical
parameters normalized, in probability space).

All of a layer's parameter rows at once, as one fp32 matrix product with a 0/1 class table: ``log(P @ I^T)`` for
the parameter table ``P`` [rows, num_cats] and ``I`` [classes, num_cats]. The table is read once, in its natural
order -- the same contraction :class:`~pyjuice.nodes.distributions.SoftEvidenceCategorical`'s dense forward runs,
measured there at the memory roofline. Tied nodes share a row, so nothing is computed twice. The cost is
``rows x num_cats x classes`` multiply-adds: on a 4096 x 50257 table it stays at a single read of the table up to a
few tens of classes (0.74 ms at 10), and grows from there (3.9 ms at 512, 28.5 ms at 4096).
"""

import contextlib

import torch

#: classes per matrix product: the class table is ``[classes, num_cats]`` floats
CLASS_CHUNK = 512


def num_values(dist) -> int:
    return dist.num_cats


def class_masses(layer, token_class: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``[number of nodes of layer, num_classes]``: ``log sum_{v: token_class[v] == c} p_n(v)`` for every node."""
    num_cats, rows = _table(layer)
    table = layer.params.view(-1, num_cats)                                 # [rows, num_cats]
    with _fp32_matmul():
        if num_classes <= CLASS_CHUNK:
            masses = table @ _indicator(token_class, 0, num_classes, table.dtype).t()
        else:
            masses = torch.empty(table.size(0), num_classes, dtype = table.dtype, device = table.device)
            for c0 in range(0, num_classes, CLASS_CHUNK):
                c1 = min(c0 + CLASS_CHUNK, num_classes)
                masses[:, c0:c1] = table @ _indicator(token_class, c0, c1, table.dtype).t()
    return masses.log_()[rows]


def _indicator(token_class: torch.Tensor, c0: int, c1: int, dtype) -> torch.Tensor:
    """``[c1 - c0, num_cats]``: 1 where the token is in the class."""
    return (token_class[None, :] == torch.arange(c0, c1, device = token_class.device)[:, None]).to(dtype)


def _table(layer):
    """``(num_cats, row of every node)``, after checking that the layer's parameters are one ``[rows, num_cats]``
    table and that every node starts a row of it. Kept on the layer with the ``s_pids`` it was computed from (a
    device move replaces them, and the rows are computed again)."""
    table = getattr(layer, "_constraint_table", None)
    if table is None or table[0] is not layer.s_pids:
        widths = {ns.dist.num_cats for ns in layer.nodes}
        width = widths.pop() if len(widths) == 1 else None
        if width is None or layer.params.numel() % width != 0 or not bool((layer.s_pids % width == 0).all()):
            raise NotImplementedError(
                f"Class masses need the input layer's Categorical parameters as one [rows, num_cats] table, but its "
                f"nodes have num_cats {sorted({ns.dist.num_cats for ns in layer.nodes})} over "
                f"{layer.params.numel()} parameters.")
        table = (layer.s_pids, width, layer.s_pids // width)
        layer._constraint_table = table
    return table[1:]


@contextlib.contextmanager
def _fp32_matmul():
    """Exact fp32 products (TF32's 10-bit mantissa would cost ~1e-3 of every mass)."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)
