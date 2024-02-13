from __future__ import annotations

import torch
from typing import Union, Sequence

from pyjuice.nodes import CircuitNodes


class Layer():
    def __init__(self, nodes: Sequence[CircuitNodes], disable_block_size_check: bool = False) -> None:

        if disable_block_size_check:
            self.block_size = None
        else:
            for i in range(1, len(nodes)):
                assert nodes[i].block_size == nodes[0].block_size, "`block_size` of nodes in the same layer must be identical."

            self.block_size = nodes[0].block_size

        self.device = torch.device("cpu")

    def enable_partial_evaluation(self, fw_scopes: Optional[Sequence[BitSet]] = None, bk_scopes: Optional[Sequence[BitSet]] = None):
        if not self.provided("fw_scope2localids") or not self.provided("bk_scope2localids"):
            raise ValueError("Please initialize node cache by calling `pc._create_scope2nid_cache()` first.")

        # Filter forward nodes
        if fw_scopes is not None:
            fw_partition_local_ids = [[] for _ in range(self.num_fw_partitions)]
            for scope in fw_scopes:
                if scope not in self.fw_scope2localids:
                    continue

                for partition_id, ids in enumerate(self.fw_scope2localids[scope]):
                    fw_partition_local_ids[partition_id].append(self.fw_scope2localids[scope][partition_id])

            self.fw_partition_local_ids = [
                torch.cat(ids, dim = 0) if len(ids) > 0 else torch.zeros([0], dtype = torch.long) for ids in fw_partition_local_ids
            ]

        # Filter backward nodes
        if bk_scopes is not None:
            bk_partition_local_ids = [[] for _ in range(self.num_bk_partitions)]
            for scope in bk_scopes:
                if scope not in self.bk_scope2localids:
                    continue

                for partition_id, ids in enumerate(self.bk_scope2localids[scope]):
                    bk_partition_local_ids[partition_id].append(self.bk_scope2localids[scope][partition_id])

            self.bk_partition_local_ids = [
                torch.cat(ids, dim = 0) if len(ids) > 0 else torch.zeros([0], dtype = torch.long) for ids in bk_partition_local_ids
            ]

    def disable_partial_evaluation(self, forward: bool = True, backward: bool = True):
        if forward:
            self.fw_partition_local_ids = None

        if backward:
            self.bk_partition_local_ids = None

    def provided(self, var_name):
        return hasattr(self, var_name) and getattr(self, var_name) is not None
