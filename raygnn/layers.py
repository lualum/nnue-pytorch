"""Parallel ray, knight and pawn messages with gated residual updates."""

import torch
from torch import Tensor, nn

from .geometry import BoardGeometry, RayFeatures


class RayLayer(nn.Module):
    def __init__(self, d_model: int = 64, d_msg: int = 16):
        super().__init__()
        self.piece_embedding = nn.Embedding(13, 8)
        self.direction_embedding = nn.Embedding(8, 4)
        edge_width = 3 * 8 + 4 + 8
        ray_width = 2 * d_model + edge_width
        self.orthogonal = nn.Sequential(nn.Linear(ray_width, 64), nn.SiLU(), nn.Linear(64, d_msg))
        self.diagonal = nn.Sequential(nn.Linear(ray_width, 64), nn.SiLU(), nn.Linear(64, d_msg))
        self.ray_gate = nn.Linear(edge_width, d_msg)
        jump_width = 2 * d_model + 8 + 4
        self.knight_message = nn.Sequential(nn.Linear(jump_width, 64), nn.SiLU(), nn.Linear(64, d_msg))
        self.pawn_message = nn.Sequential(nn.Linear(jump_width, 64), nn.SiLU(), nn.Linear(64, d_msg))
        self.update = nn.Sequential(nn.Linear(d_model + 10 * d_msg, 128), nn.SiLU(), nn.Linear(128, d_model))
        self.update_gate = nn.Linear(d_model + 10 * d_msg, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _jump(self, h: Tensor, piece: Tensor, geometry: BoardGeometry, relation: str) -> Tensor:
        index, source_piece, mask = geometry.jump_sources(piece, relation)
        b, n, k = h.shape[0], 64, index.shape[-1]
        source_h = h[:, index].reshape(b, n, k, -1)
        dest_h = h[:, :, None, :].expand_as(source_h)
        direction = self.direction_embedding(torch.arange(k, device=h.device)).view(1, 1, k, 4).expand(b, n, -1, -1)
        edge = torch.cat((source_h, dest_h, self.piece_embedding(source_piece), direction), -1)
        net = self.knight_message if relation == "knight" else self.pawn_message
        message = net(edge) * mask[..., None]
        return (message.sum(2) / mask.sum(2).clamp(min=1).sqrt()[..., None]).to(h.dtype)

    def forward(self, h: Tensor, piece: Tensor, geometry: BoardGeometry, ray: RayFeatures,
                use_xray: bool = True, first_piece_only: bool = False) -> Tensor:
        b = h.shape[0]
        index = geometry.ray_index.clamp_min(0)
        source_h = h[:, index]
        dest_h = h[:, :, None, None, :].expand_as(source_h)
        direction = self.direction_embedding(torch.arange(8, device=h.device)).view(1, 1, 8, 1, 4).expand(b, 64, -1, 7, -1)
        distance = torch.arange(1, 8, device=h.device).view(1, 1, 1, 7).expand(b, 64, 8, -1).float() / 7
        numeric = torch.stack((
            distance, ray.direct.float(), ray.xray.float(), ray.attack.float(),
            ray.blocker_count.float() / 2, ray.blocker_distance.float() / 7,
            torch.where(ray.blocker_square >= 0, ray.blocker_square % 8, 0).float() / 7,
            torch.where(ray.blocker_square >= 0, ray.blocker_square // 8, 0).float() / 7,
        ), -1)
        edge = torch.cat((
            self.piece_embedding(ray.source_piece), self.piece_embedding(ray.destination_piece),
            self.piece_embedding(ray.blocker_piece), direction, numeric,
        ), -1)
        values = torch.cat((source_h, dest_h, edge), -1)
        orthogonal = (0, 2, 4, 6)
        diagonal = (1, 3, 5, 7)
        messages = torch.empty((*values.shape[:-1], self.ray_gate.out_features), device=h.device, dtype=h.dtype)
        messages[:, :, orthogonal] = self.orthogonal(values[:, :, orthogonal])
        messages[:, :, diagonal] = self.diagonal(values[:, :, diagonal])
        mask = ray.valid & (ray.source_piece != 0) & (ray.direct | (ray.xray & use_xray))
        if first_piece_only:
            mask = mask & ray.direct & (ray.destination_piece != 0)
        messages = messages * torch.sigmoid(self.ray_gate(edge)) * mask[..., None]
        direction_sum = messages.sum(3) / mask.sum(3).clamp(min=1).sqrt()[..., None]
        aggregate = torch.cat((
            direction_sum.flatten(2),
            self._jump(h, piece, geometry, "knight"),
            self._jump(h, piece, geometry, "pawn"),
        ), -1)
        combined = torch.cat((h, aggregate), -1)
        return self.norm(h + torch.sigmoid(self.update_gate(combined)) * self.update(combined))
