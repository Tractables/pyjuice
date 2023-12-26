import torch
import torch.nn as nn


class FastParamList(nn.ParameterList):

    def __getitem__(self, idx):
        return getattr(self, str(idx))