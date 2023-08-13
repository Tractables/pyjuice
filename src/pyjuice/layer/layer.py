from __future__ import annotations

import torch
from typing import Union


class Layer():
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def init_layer(self, params: Union[torch.Tensor,None]):
        raise NotImplementedError()

    def enable_partial_evaluation(self, fw_scopes: Optional[Sequence[BitSet]] = None, bk_scopes: Optional[Sequence[BitSet]] = None):

        # Filter forward nodes
        if fw_scopes is not None:
            fw_group_local_ids = [[torch.zeros([0], dtype = torch.long)] for _ in range(self.num_fw_groups)]
            for scope in fw_scopes:
                if scope not in self.fw_scope2localids:
                    continue

                for group_id, ids in enumerate(self.fw_scope2localids[scope]):
                    fw_group_local_ids[group_id].append(self.fw_scope2localids[scope][group_id])

            self.fw_group_local_ids = [
                torch.cat(ids, dim = 0).to(self.device) for ids in fw_group_local_ids
            ]

        # Filter backward nodes
        if bk_scopes is not None:
            bk_group_local_ids = [[torch.zeros([0], dtype = torch.long)] for _ in range(self.num_bk_groups)]
            for scope in bk_scopes:
                if scope not in self.bk_scope2localids:
                    continue

                for group_id, ids in enumerate(self.bk_scope2localids[scope]):
                    bk_group_local_ids[group_id].append(self.bk_scope2localids[scope][group_id])

            self.bk_group_local_ids = [
                torch.cat(ids, dim = 0).to(self.device) for ids in bk_group_local_ids
            ]

    def disable_partial_evaluation(self, forward: bool = True, backward: bool = True):
        if forward:
            self.fw_group_local_ids = None

        if backward:
            self.bk_group_local_ids = None

    def provided(self, var_name):
        return hasattr(self, var_name) and getattr(self, var_name) is not None