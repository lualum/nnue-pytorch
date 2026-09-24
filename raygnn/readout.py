"""Jointly trained piece-square, contextual, and spatial v3 readout."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GlobalReadout(nn.Module):
    def __init__(self, spatial_width: int = 8, pooling_alternative: bool = False):
        super().__init__()
        self.pooling_alternative = pooling_alternative
        self.piece_square = nn.Embedding(12 * 64, 1)
        nn.init.zeros_(self.piece_square.weight)
        self.local = nn.Sequential(nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 1))
        if pooling_alternative:
            self.spatial = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
            global_width = 128 + 16
        else:
            self.spatial = nn.Linear(192, spatial_width)
            global_width = 64 * spatial_width + 16
        self.head = nn.Sequential(nn.Linear(global_width, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.normal_(self.local[-1].weight, 0, 0.01)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.normal_(self.head[-1].weight, 0, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, h: Tensor, initial: Tensor, state: Tensor, piece: Tensor) -> Tensor:
        square = torch.arange(64, device=piece.device)[None]
        ids = (piece - 1).clamp_min(0) * 64 + square
        # Index zero-valued empty entries through a mask so they never learn.
        linear = (self.piece_square(ids).squeeze(-1) * (piece != 0)).sum(1, keepdim=True)
        combined = torch.cat((h, initial), -1)
        local = self.local(combined).sum(1) / 8
        spatial = self.spatial(combined)
        spatial = spatial.mean(1) if self.pooling_alternative else F.silu(spatial).flatten(1)
        global_value = self.head(torch.cat((spatial, state), -1))
        return linear + local + global_value
