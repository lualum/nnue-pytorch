"""Structured piece, spatial and king-centered summaries."""

import torch
from torch import Tensor, nn

from .geometry import BoardGeometry


class StructuredReadout(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.output_dim = 704 + 2 * d_model
        self.piece_project = nn.Linear(d_model + 1, 32)
        self.piece_type = nn.Embedding(12, 32)
        self.spatial_project = nn.Linear(d_model, 16)
        self.king_neighbor_project = nn.Linear(8 * d_model, 32)
        square = torch.arange(64)
        region = (square // 8 // 2) * 4 + (square % 8 // 2)
        self.register_buffer("region", region, persistent=False)

    def forward(self, h: Tensor, piece: Tensor, geometry: BoardGeometry) -> Tensor:
        b = h.shape[0]
        types = torch.arange(1, 13, device=h.device)
        masks = (piece[:, :, None] == types).float()
        grouped = torch.einsum("bsi,bsd->bid", masks, h)
        counts = masks.sum(1)
        # Sum retains multiplicity; count is also explicit, including empty groups.
        piece_vector = torch.cat((grouped, counts[..., None].to(grouped.dtype)), -1)
        piece_vector = self.piece_project(piece_vector)
        piece_vector = piece_vector + self.piece_type(types - 1)[None].to(piece_vector.dtype)
        piece_vector = piece_vector.flatten(1).to(h.dtype)

        projected = self.spatial_project(h)
        spatial = torch.zeros((b, 16, 16), device=h.device, dtype=projected.dtype)
        spatial.index_add_(1, self.region, projected)
        spatial = (spatial / 4).flatten(1).to(h.dtype)

        king_indices = torch.stack(((piece == 6).long().argmax(1), (piece == 12).long().argmax(1)), 1)
        king_h = h.gather(1, king_indices[..., None].expand(-1, -1, h.shape[-1]))
        neighbors = geometry.king_neighbors[king_indices]
        neighbor_h = h.gather(1, neighbors.clamp_min(0).flatten(1)[..., None].expand(-1, -1, h.shape[-1]))
        neighbor_h = neighbor_h.reshape(b, 2, 8, -1) * (neighbors >= 0)[..., None]
        king_summary = self.king_neighbor_project(neighbor_h.flatten(2)).to(h.dtype)
        king_vector = torch.cat((king_h, king_summary), -1).flatten(1)
        return torch.cat((piece_vector, spatial, king_vector), -1)
