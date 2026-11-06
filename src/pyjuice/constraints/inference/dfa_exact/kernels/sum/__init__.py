"""
The lifted sum layer. A sum node and its children (products over the same scope) keep blocks of the same
shape, so a sum acts on every column independently: it is pyjuice's sum over ``B * slots`` columns, where
``slots`` is the block size of the node group's scope. The edges are read from the sum layer's own partitions
(``partitioned_nids / cids / pids``): node ``i`` of the node block at ``nids[k]`` adds child element
``cids[k, e]`` with weight ``params[pids[k, e] + i]``.

Under evidence pruning (:mod:`.live`) only the live columns are computed, and the dead ones set to -inf. Two
paths:

* DENSE node blocks -- children a contiguous run of element rows, weights strided by the block size (an HMM's
  transitions) -- are a matrix product with weights read in place from the parameter buffer: a node block's
  weights are the ``[E, BS]`` matrix at ``params[pids[k, 0]]``. The children are shifted by their column
  maximum and exponentiated once (one pass), multiplied by cuBLAS, and the log taken and the maximum added back
  (one pass over the product, in place). In fp32 this is cuBLAS's exact fp32
  product (measured faster than pyjuice's own sum layer at every batch size on a 4096-state HMM, with ~100x
  smaller error). Node blocks that read the same children from consecutive rows can instead multiply their
  weights stacked into one ``[K * BS, E]`` matrix (:func:`stacked_weights`, a copy) in one product;
  :mod:`.forward` decides when the copy pays for itself;
* every other node block runs the fused Triton kernel: an online log-sum-exp over chunks of children, with
  the products on tensor cores. Padded edges point at pyjuice's dummy element rows, which lie below the
  product layer group's region, and are skipped. A gated sum layer's node blocks
  (:mod:`..external_params.block_scale`) run it with their gates (``GATED``, that module's kernel
  helpers), every one of them: the dense path has no per-sample weights.
"""

from . import tile_rules, steps, fused_cuda, fused_forward, dense_common, dense_live, dense, fused_top_down, dense_top_down
from .steps import gate_launch
from .fused_forward import fused_sum
from .dense_common import query_weights, sample_spans, stacked_weights
from .dense import dense_groups, dense_sum
from .fused_top_down import fused_sum_top_down
from .dense_top_down import dense_sum_top_down, fill_dead_rows, per_sample_dense_sum_top_down
