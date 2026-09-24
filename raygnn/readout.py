"""Shared nonlinear per-square readout and global mean pool."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GlobalReadout(nn.Module):
    def __init__(self, width: int = 128, final_activation: bool = True):
        super().__init__()
        self.first = nn.Linear(128, width)
        self.second = nn.Linear(width, width)
        self.final_activation = final_activation

    def forward(self, h: Tensor, initial: Tensor) -> Tensor:
        square = F.silu(self.first(torch.cat((h, initial), -1)))
        square = self.second(square)
        if self.final_activation:
            square = F.silu(square)
        return square.mean(1)
