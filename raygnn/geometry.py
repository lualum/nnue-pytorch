"""All 1,792 static candidate edges and occupancy-dependent edge features."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BoardGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        sources, destinations, paths, masks, deltas, knights = [], [], [], [], [], []
        for src in range(64):
            sf, sr = src % 8, src // 8
            for dst in range(64):
                if src == dst:
                    continue
                df, dr = dst % 8 - sf, dst // 8 - sr
                ray = df == 0 or dr == 0 or abs(df) == abs(dr)
                knight = sorted((abs(df), abs(dr))) == [1, 2]
                if not (ray or knight):
                    continue
                distance = max(abs(df), abs(dr))
                step_f = (df > 0) - (df < 0)
                step_r = (dr > 0) - (dr < 0)
                path = [8 * (sr + k * step_r) + sf + k * step_f for k in range(1, distance)] if ray else []
                sources.append(src)
                destinations.append(dst)
                paths.append(path + [0] * (6 - len(path)))
                masks.append([True] * len(path) + [False] * (6 - len(path)))
                deltas.append((df, dr))
                knights.append(knight)
        if len(sources) != 1792 or sum(knights) != 336:
            raise AssertionError("unexpected edge topology")
        self.register_buffer("src", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(destinations, dtype=torch.long))
        self.register_buffer("between", torch.tensor(paths, dtype=torch.long))
        self.register_buffer("between_mask", torch.tensor(masks, dtype=torch.bool))
        self.register_buffer("relative_delta", torch.tensor(deltas, dtype=torch.long))
        self.register_buffer("is_knight", torch.tensor(knights, dtype=torch.bool))

    def edge_features(self, piece: Tensor, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
        """Return [B,1792,31] features and exclusive relationship IDs; -1 is inactive."""
        b = piece.shape[0]
        interior = piece[:, self.between]
        occupied = (interior != 0) & self.between_mask[None]
        count = occupied.sum(-1)
        present = count > 0
        first_slot = occupied.to(torch.long).argmax(-1)
        first_piece = interior.gather(-1, first_slot[..., None]).squeeze(-1)
        first_piece = torch.where(present, first_piece, 0)
        first_square = self.between[None].expand(b, -1, -1).gather(-1, first_slot[..., None]).squeeze(-1)
        df, dr = self.relative_delta[:, 0], self.relative_delta[:, 1]
        orthogonal = (df == 0) | (dr == 0)
        diagonal = df.abs() == dr.abs()
        adjacent = (df.abs() <= 1) & (dr.abs() <= 1)
        source_piece = piece[:, self.src]
        destination_piece = piece[:, self.dst]
        kind = torch.where(source_piece > 6, source_piece - 6, source_piece)
        pawn_attack = (kind == 1) & (df.abs()[None] == 1) & (
            ((source_piece <= 6) & (dr[None] == 1)) |
            ((source_piece > 6) & (dr[None] == -1)))
        geometry = ((kind == 4) & orthogonal[None]) | ((kind == 3) & diagonal[None])
        geometry = geometry | ((kind == 5) & ~self.is_knight[None])
        geometry = geometry | ((kind == 2) & self.is_knight[None])
        geometry = geometry | ((kind == 6) & adjacent[None]) | pawn_attack
        geometry = geometry & (source_piece != 0)
        direct = geometry & ~present
        src_rank = self.src // 8
        pawn_forward = (kind == 1) & (df[None] == 0) & (
            ((source_piece <= 6) & ((dr[None] == 1) | ((src_rank[None] == 1) & (dr[None] == 2)))) |
            ((source_piece > 6) & ((dr[None] == -1) | ((src_rank[None] == 6) & (dr[None] == -2)))))
        first_delta = torch.stack((first_square % 8 - self.src[None] % 8,
                                   first_square // 8 - self.src[None] // 8), -1)
        first_delta = torch.where(present[..., None], first_delta, 0)
        relative = self.relative_delta.to(dtype)[None].expand(b, -1, -1) / 7
        features = torch.cat((
            relative, self.is_knight.to(dtype)[None, :, None].expand(b, -1, -1),
            F.one_hot(count, 7).to(dtype), F.one_hot(first_piece, 13).to(dtype),
            first_delta.to(dtype) / 7, present[..., None].to(dtype),
            geometry[..., None].to(dtype), direct[..., None].to(dtype),
            (source_piece != 0)[..., None].to(dtype), (destination_piece != 0)[..., None].to(dtype),
            pawn_forward[..., None].to(dtype),
        ), -1)
        slider = ((kind == 4) & orthogonal[None]) | ((kind == 3) & diagonal[None]) | ((kind == 5) & (orthogonal | diagonal)[None])
        xray = (source_piece != 0) & slider & (count == 1) & ~direct
        context = (~direct & ~xray) & (adjacent[None] | pawn_forward)
        classes = torch.where(direct, 0, torch.where(xray, 1, torch.where(context, 2, -1)))
        return features, classes
