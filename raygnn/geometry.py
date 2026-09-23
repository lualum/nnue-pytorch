"""Static chess geometry and vectorized occupancy-dependent ray relations."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Direction is from destination toward source. Opposite directions share a class.
DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


@dataclass
class RayFeatures:
    source_piece: Tensor
    destination_piece: Tensor
    blocker_piece: Tensor
    blocker_distance: Tensor
    blocker_square: Tensor
    blocker_count: Tensor
    direct: Tensor
    xray: Tensor
    attack: Tensor
    valid: Tensor


class BoardGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        ray = torch.full((64, 8, 7), -1, dtype=torch.long)
        knight = torch.full((64, 8), -1, dtype=torch.long)
        pawn = torch.full((64, 4), -1, dtype=torch.long)
        king_neighbors = torch.full((64, 8), -1, dtype=torch.long)
        knight_steps = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
        # Pawn sources: two White sources below destination, then two Black sources above it.
        pawn_steps = ((-1, -1), (1, -1), (-1, 1), (1, 1))
        for square in range(64):
            file, rank = square % 8, square // 8
            for direction, (df, dr) in enumerate(DIRECTIONS):
                for distance in range(1, 8):
                    f, r = file + df * distance, rank + dr * distance
                    if 0 <= f < 8 and 0 <= r < 8:
                        ray[square, direction, distance - 1] = r * 8 + f
                f, r = file + df, rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    king_neighbors[square, direction] = r * 8 + f
            for index, (df, dr) in enumerate(knight_steps):
                f, r = file + df, rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    knight[square, index] = r * 8 + f
            for index, (df, dr) in enumerate(pawn_steps):
                f, r = file + df, rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    pawn[square, index] = r * 8 + f
        self.register_buffer("ray_index", ray, persistent=False)
        self.register_buffer("knight_index", knight, persistent=False)
        self.register_buffer("pawn_index", pawn, persistent=False)
        self.register_buffer("king_neighbors", king_neighbors, persistent=False)

    def rays(self, piece: Tensor) -> RayFeatures:
        b = piece.shape[0]
        ray = self.ray_index
        source = piece[:, ray.clamp_min(0)]
        valid = (ray >= 0)[None].expand(b, -1, -1, -1)
        occupied = (source != 0) & valid
        # Prefix excludes both endpoints. A ray slot at distance one has no intervening squares.
        count = torch.cat((torch.zeros_like(occupied[..., :1], dtype=torch.long), occupied.long().cumsum(-1)[..., :-1]), -1)
        positions = torch.arange(1, 8, device=piece.device).view(1, 1, 1, 7)
        last = torch.cummax(torch.where(occupied, positions, 0), -1).values
        last = torch.cat((torch.zeros_like(last[..., :1]), last[..., :-1]), -1)
        blocker = source.gather(-1, (last - 1).clamp_min(0))
        blocker = torch.where(last > 0, blocker, 0)
        blocker_square = ray[None].expand(b, -1, -1, -1).gather(-1, (last - 1).clamp_min(0))
        blocker_square = torch.where(last > 0, blocker_square, -1)
        destination = piece[:, :, None, None].expand_as(source)
        direct = valid & (count == 0)
        xray = valid & (count > 0)
        kind = ((source - 1) % 6) + 1
        orthogonal = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], device=piece.device, dtype=torch.bool)[None, None, :, None]
        adjacent = positions == 1
        slider = ((kind == 5) | (kind == 4) & orthogonal | (kind == 3) & ~orthogonal)
        king = (kind == 6) & adjacent
        # White pawns below a destination and Black pawns above it.
        white_pawn = (source == 1) & torch.tensor([0, 0, 0, 1, 0, 1, 0, 0], device=piece.device, dtype=torch.bool)[None, None, :, None]
        black_pawn = (source == 7) & torch.tensor([0, 1, 0, 0, 0, 0, 0, 1], device=piece.device, dtype=torch.bool)[None, None, :, None]
        pawn = (white_pawn | black_pawn) & adjacent
        attack = direct & (source != 0) & (slider | king | pawn)
        return RayFeatures(source, destination, blocker, (positions - last).where(last > 0, 0).expand_as(source), blocker_square, count.clamp(max=2), direct, xray, attack, valid)

    def jump_sources(self, piece: Tensor, relation: str) -> tuple[Tensor, Tensor, Tensor]:
        index = self.knight_index if relation == "knight" else self.pawn_index
        source = piece[:, index.clamp_min(0)]
        if relation == "knight":
            mask = (index >= 0)[None] & ((source == 2) | (source == 8))
        elif relation == "pawn":
            white = (source[..., :2] == 1)
            black = (source[..., 2:] == 7)
            mask = (index >= 0)[None] & torch.cat((white, black), -1)
        else:
            raise ValueError("relation must be knight or pawn")
        return index.clamp_min(0), source, mask
