from __future__ import annotations

import torch
from typing import Union, Sequence

from pyjuice.nodes import CircuitNodes


class Layer():

    propagation_alg_mapping = {
        "LL": 0,
        "MPE": 1,
        "GeneralLL": 2
    }

    #: Whether this layer needs the PC's denominator param-flow buffer `pc.denom_param_flows` (a second
    #: accumulator shaped like `param_flows`). `False` for every ordinary layer, so a plain PC allocates
    #: nothing and pays nothing. Overridden by layers whose M-step is a conditional dual-flow update --
    #: see `ExternalSumParams.requests_denom_param_flows`.
    requests_denom_param_flows: bool = False

    #: Node-subset selections the layers' passes still accept. Always None now that partial evaluation is
    #: gone; they are removed together with the code that reads them. Declared here so that
    #: `provided()` -- asked on every pass -- finds None rather than going
    #: through `nn.Module.__getattr__`'s AttributeError: ~0.4 us per check, ~200 checks per step.
    fw_partition_local_ids = None
    bk_partition_local_ids = None

    def __init__(self, nodes: Sequence[CircuitNodes], disable_block_size_check: bool = False) -> None:

        # Nodes correspond to the current layer
        self.nodes = nodes

        # The set of unique scopes
        self.scopes = []
        for ns in self.nodes:
            if ns.scope not in self.scopes:
                self.scopes.append(ns.scope)

        if disable_block_size_check:
            self.block_size = None
        else:
            for i in range(1, len(nodes)):
                assert nodes[i].block_size == nodes[0].block_size, "`block_size` of nodes in the same layer must be identical."

            self.block_size = nodes[0].block_size

        self.device = torch.device("cpu")

    def provided(self, var_name):
        return hasattr(self, var_name) and getattr(self, var_name) is not None

    #: Whether the circuit may capture this layer's passes in a CUDA graph. A replay runs no Python, so a
    #: layer is graph-safe when its per-call Python leaves nothing behind but what `cuda_graph_state`
    #: hands over (a host sync is fine: the capture fails and the pass stays eager).
    cuda_graph_safe: bool = True

    def cuda_graph_state(self):
        """
        Host-side state a pass of this layer leaves for a LATER pass to read -- a forward's leftovers
        that its backward needs, say -- or None.

        A replayed CUDA graph runs no Python, so state a recorded pass sets would otherwise keep
        describing whichever call last ran eagerly. The circuit saves this after recording a graph and
        hands it back through :func:`restore_cuda_graph_state` after every replay, and a backward graph
        is only replayed against the state it was recorded with. A plain layer keeps nothing between
        passes.
        """
        return None

    def restore_cuda_graph_state(self, state) -> None:
        """Put back what :func:`cuda_graph_state` returned right after a graph of this pass was recorded."""
        pass

    def is_sum(self):
        return False

    def is_prod(self):
        return False

    def is_input(self):
        return False

    def _get_propagation_alg_kwargs(self, propagation_alg: str, **kwargs):
        if propagation_alg == "LL":
            return {"alpha": 0.0}
        elif propagation_alg == "MPE":
            return {"alpha": 0.0}
        elif propagation_alg == "GeneralLL":
            return {"alpha": kwargs["alpha"]}
        else:
            raise ValueError(f"Unknown propagation algorithm {propagation_alg}.")
