"""Reference implementation of the white-positive RayGNN v0.1 evaluator.

This is deliberately a dense, batchable PyTorch reference.  It favours a
clear position contract and chess-correct geometry over incremental inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EMPTY = 0
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING = range(1, 7)
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING = range(7, 13)

_PIECE_FROM_FEN = {
    "P": WHITE_PAWN, "N": WHITE_KNIGHT, "B": WHITE_BISHOP, "R": WHITE_ROOK,
    "Q": WHITE_QUEEN, "K": WHITE_KING, "p": BLACK_PAWN, "n": BLACK_KNIGHT,
    "b": BLACK_BISHOP, "r": BLACK_ROOK, "q": BLACK_QUEEN, "k": BLACK_KING,
}
_RAY_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
_KNIGHT_OFFSETS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-1, 2), (-2, 1))
_MATERIAL = torch.tensor([0, 100, 320, 330, 500, 900, 0, -100, -320, -330, -500, -900, 0])


def _inside(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def _square(file: int, rank: int) -> int:
    return rank * 8 + file


@dataclass(frozen=True)
class RayGNNPosition:
    """Lossless v0.1 state input in fixed White-oriented coordinates.

    ``repetition_count`` is separate because ordinary FEN has no repetition
    history.  A threefold claim therefore must be supplied by the caller.
    """

    board: Tensor                 # [B, 64], 0=empty, then WP..WK, BP..BK
    side_to_move: Tensor          # [B], 1=White, 0=Black
    castling_rights: Tensor       # [B, 4], K, Q, k, q
    en_passant: Tensor            # [B], a1..h8 or 64 for none
    halfmove_clock: Tensor        # [B]
    repetition_count: Tensor      # [B]

    @classmethod
    def from_fens(
        cls, fens: str | Sequence[str], repetition_count: int | Sequence[int] = 0,
        device: torch.device | str | None = None,
    ) -> "RayGNNPosition":
        if isinstance(fens, str):
            fens = [fens]
        if isinstance(repetition_count, int):
            repetition_count = [repetition_count] * len(fens)
        if len(fens) != len(repetition_count):
            raise ValueError("repetition_count must have one entry per FEN.")
        boards, turns, castles, eps, halfmoves = [], [], [], [], []
        for fen in fens:
            fields = fen.split()
            if len(fields) != 6:
                raise ValueError("FEN must contain six fields.")
            placement, turn, castling, ep, halfmove, _fullmove = fields
            ranks = placement.split("/")
            if len(ranks) != 8 or turn not in ("w", "b"):
                raise ValueError(f"Invalid FEN: {fen}")
            board = [EMPTY] * 64
            for fen_rank, rank_text in enumerate(ranks):
                file = 0
                for symbol in rank_text:
                    if symbol.isdigit():
                        file += int(symbol)
                    elif symbol in _PIECE_FROM_FEN and file < 8:
                        board[_square(file, 7 - fen_rank)] = _PIECE_FROM_FEN[symbol]
                        file += 1
                    else:
                        raise ValueError(f"Invalid piece placement in FEN: {fen}")
                if file != 8:
                    raise ValueError(f"Each FEN rank must contain eight squares: {fen}")
            if board.count(WHITE_KING) != 1 or board.count(BLACK_KING) != 1:
                raise ValueError(f"FEN must contain exactly one king of each color: {fen}")
            boards.append(board)
            turns.append(turn == "w")
            if castling != "-" and (
                any(right not in "KQkq" for right in castling)
                or len(set(castling)) != len(castling)
            ):
                raise ValueError(f"Invalid castling field in FEN: {fen}")
            castles.append(["K" in castling, "Q" in castling, "k" in castling, "q" in castling])
            if ep == "-":
                eps.append(64)
            elif len(ep) == 2 and ep[0] in "abcdefgh" and ep[1] in "36":
                eps.append(_square(ord(ep[0]) - ord("a"), int(ep[1]) - 1))
            else:
                raise ValueError(f"Invalid en-passant square in FEN: {fen}")
            halfmoves.append(int(halfmove))
        return cls(
            torch.tensor(boards, dtype=torch.long, device=device),
            torch.tensor(turns, dtype=torch.float32, device=device),
            torch.tensor(castles, dtype=torch.float32, device=device),
            torch.tensor(eps, dtype=torch.long, device=device),
            torch.tensor(halfmoves, dtype=torch.float32, device=device),
            torch.tensor(repetition_count, dtype=torch.float32, device=device),
        )

    def to(self, device: torch.device | str) -> "RayGNNPosition":
        return RayGNNPosition(
            self.board.to(device), self.side_to_move.to(device),
            self.castling_rights.to(device), self.en_passant.to(device),
            self.halfmove_clock.to(device), self.repetition_count.to(device),
        )


class RayGNNLayer(nn.Module):
    def __init__(self, d_model: int, d_msg: int, edge_dim: int):
        super().__init__()
        self.direction_embedding = nn.Embedding(8, 8)
        self.orthogonal_message = nn.Sequential(nn.Linear(2 * d_model + edge_dim + 8, 96), nn.GELU(), nn.Linear(96, d_msg))
        self.diagonal_message = nn.Sequential(nn.Linear(2 * d_model + edge_dim + 8, 96), nn.GELU(), nn.Linear(96, d_msg))
        self.knight_message = nn.Sequential(nn.Linear(2 * d_model + edge_dim, 96), nn.GELU(), nn.Linear(96, d_msg))
        self.pawn_message = nn.Sequential(nn.Linear(2 * d_model + edge_dim, 96), nn.GELU(), nn.Linear(96, d_msg))
        self.update = nn.Sequential(nn.Linear(d_model + 10 * d_msg, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gate = nn.Linear(d_model + 10 * d_msg, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, ray_edge: Tensor, ray_valid: Tensor, knight: Tensor, pawn: Tensor) -> Tensor:
        # ray_edge is [B, destination, direction, distance, edge_dim].
        b, _, _, _, edge_dim = ray_edge.shape
        sources = self.ray_sources.expand(b, -1, -1, -1)
        source_h = h.gather(1, sources.reshape(b, -1).unsqueeze(-1).expand(-1, -1, h.size(-1))).view(b, 64, 8, 7, -1)
        dest_h = h[:, :, None, None, :].expand(-1, -1, 8, 7, -1)
        direction = self.direction_embedding(torch.arange(8, device=h.device))[None, None, :, None, :].expand(b, 64, -1, 7, -1)
        features = torch.cat((source_h, dest_h, ray_edge, direction), dim=-1)
        messages = h.new_zeros(b, 64, 8, 7, self.orthogonal_message[-1].out_features)
        messages[:, :, :4] = self.orthogonal_message(features[:, :, :4])
        messages[:, :, 4:] = self.diagonal_message(features[:, :, 4:])
        message_mask = ray_valid & ray_edge[..., 0].bool()
        messages = messages * message_mask.unsqueeze(-1)
        counts = message_mask.sum(dim=3, keepdim=True).clamp_min(1).to(h.dtype)
        directional = messages.sum(dim=3) / counts

        knight_agg = self._aggregate_jump(h, knight, self.knight_sources, self.knight_destinations, self.knight_message)
        pawn_agg = self._aggregate_jump(h, pawn, self.pawn_sources, self.pawn_destinations, self.pawn_message)
        aggregate = torch.cat((directional.flatten(2), knight_agg, pawn_agg), dim=-1)
        update_input = torch.cat((h, aggregate), dim=-1)
        candidate = self.update(update_input)
        return self.norm(h + torch.sigmoid(self.gate(update_input)) * candidate)

    @staticmethod
    def _aggregate_jump(h: Tensor, edge_features: Tensor, sources: Tensor, destinations: Tensor, message_net: nn.Module) -> Tensor:
        b, _, d = h.shape
        src = h.index_select(1, sources)
        dst = h.index_select(1, destinations)
        values = message_net(torch.cat((src, dst, edge_features), dim=-1))
        valid = edge_features[..., 0:1]  # source-occupied flag, supplied as first edge feature
        values = values * valid
        out = h.new_zeros(b, 64, values.size(-1))
        out.index_add_(1, destinations, values)
        counts = h.new_zeros(b, 64, 1)
        counts.index_add_(1, destinations, valid)
        return out / counts.clamp_min(1)


class RayGNNEvaluationNetwork(nn.Module):
    """Three-layer, d_model=64 RayGNN with a 928-wide value head."""

    d_model = 64
    d_msg = 16

    def __init__(self, layers: int = 3, with_wdl: bool = False):
        super().__init__()
        if layers < 1:
            raise ValueError("RayGNN requires at least one message-passing layer.")
        self.layers_count = layers
        self.with_wdl = with_wdl
        self.piece_embedding = nn.Embedding(13, 24)
        self.square_embedding = nn.Embedding(64, 16)
        self.coordinate_projection = nn.Linear(4, 16)
        self.king_offset_projection = nn.Linear(4, 16)
        self.initial_projection = nn.Linear(72, 64)
        self.edge_distance = nn.Embedding(8, 4)
        self.edge_piece = nn.Embedding(13, 6)
        self.edge_blocker_distance = nn.Embedding(8, 3)
        self.edge_blocker_count = nn.Embedding(4, 3)
        # occupied, direct, xray, actual_attack, source/destination/blocker embeddings, distances
        self.edge_dim = 1 + 3 + 6 + 6 + 6 + 4 + 3 + 3
        self.layers = nn.ModuleList(RayGNNLayer(64, 16, self.edge_dim) for _ in range(layers))
        self.piece_group_projection = nn.Linear(65, 32)
        self.spatial_projection = nn.Linear(64, 16)
        self.king_neighbor_projection = nn.Linear(64 + 8, 32)
        self.raw_board_projection = nn.Linear(64 * 13, 64)
        self.state_projection = nn.Linear(28, 32)
        self.value_head = nn.Sequential(nn.Linear(928, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.wdl_head = nn.Linear(64, 3) if with_wdl else None
        self._register_geometry()

    def _register_geometry(self) -> None:
        sources = torch.zeros(64, 8, 7, dtype=torch.long)
        valid = torch.zeros(64, 8, 7, dtype=torch.bool)
        for dest in range(64):
            df, dr = dest % 8, dest // 8
            for direction, (dx, dy) in enumerate(_RAY_DIRECTIONS):
                for distance in range(1, 8):
                    sf, sr = df - dx * distance, dr - dy * distance
                    if _inside(sf, sr):
                        sources[dest, direction, distance - 1] = _square(sf, sr)
                        valid[dest, direction, distance - 1] = True
        self.register_buffer("ray_sources", sources, persistent=False)
        self.register_buffer("ray_valid", valid, persistent=False)
        for layer in self.layers:
            layer.register_buffer("ray_sources", sources, persistent=False)
        knight_s, knight_d, pawn_s, pawn_d = [], [], [], []
        for source in range(64):
            f, r = source % 8, source // 8
            for dx, dy in _KNIGHT_OFFSETS:
                if _inside(f + dx, r + dy):
                    knight_s.append(source); knight_d.append(_square(f + dx, r + dy))
            for dy in (-1, 1):
                for dx in (-1, 1):
                    if _inside(f + dx, r + dy):
                        pawn_s.append(source); pawn_d.append(_square(f + dx, r + dy))
        for name, value in (("knight_sources", knight_s), ("knight_destinations", knight_d), ("pawn_sources", pawn_s), ("pawn_destinations", pawn_d)):
            tensor = torch.tensor(value, dtype=torch.long)
            self.register_buffer(name, tensor, persistent=False)
            for layer in self.layers:
                layer.register_buffer(name, tensor, persistent=False)

    @staticmethod
    def _color(board: Tensor) -> Tensor:
        return torch.where(board == EMPTY, 0, torch.where(board <= 6, 1, -1))

    def _initial_state(self, position: RayGNNPosition) -> Tensor:
        board = position.board
        if board.ndim != 2 or board.shape[1] != 64 or board.dtype not in (torch.int32, torch.int64):
            raise ValueError("board must be an integer tensor shaped [B, 64].")
        squares = torch.arange(64, device=board.device)
        files = (squares % 8).float() / 7.0
        ranks = (squares // 8).float() / 7.0
        coords = torch.stack((files, ranks, files * 2 - 1, ranks * 2 - 1), dim=-1)
        white_king = (board == WHITE_KING).float().argmax(dim=1)
        black_king = (board == BLACK_KING).float().argmax(dim=1)
        king_offsets = torch.stack((
            (squares[None] % 8 - white_king[:, None] % 8) / 7.0,
            (squares[None] // 8 - white_king[:, None] // 8) / 7.0,
            (squares[None] % 8 - black_king[:, None] % 8) / 7.0,
            (squares[None] // 8 - black_king[:, None] // 8) / 7.0,
        ), dim=-1).float()
        return self.initial_projection(torch.cat((
            self.piece_embedding(board),
            self.square_embedding(squares)[None].expand(board.size(0), -1, -1),
            self.coordinate_projection(coords)[None].expand(board.size(0), -1, -1),
            self.king_offset_projection(king_offsets),
        ), dim=-1))

    def _ray_features(self, board: Tensor) -> tuple[Tensor, Tensor]:
        b = board.size(0)
        sources = self.ray_sources[None].expand(b, -1, -1, -1)
        source_piece = board.gather(1, sources.reshape(b, -1)).view(b, 64, 8, 7)
        destination_piece = board[:, :, None, None].expand(-1, -1, 8, 7)
        valid = self.ray_valid[None].expand(b, -1, -1, -1)
        occupied_source = (source_piece != EMPTY) & valid
        # Squares between source and destination are the earlier source-side slots.
        blockers = (source_piece != EMPTY).cumsum(dim=3) - (source_piece != EMPTY)
        direct = blockers == 0
        xray = blockers > 0
        first_index = (source_piece != EMPTY).float().argmax(dim=3, keepdim=True).expand_as(source_piece)
        first_piece = source_piece.gather(3, first_index)
        first_piece = torch.where(xray, first_piece, torch.zeros_like(first_piece))
        distance = torch.arange(1, 8, device=board.device)[None, None, None, :].expand(b, 64, 8, -1)
        blocker_distance = torch.where(xray, first_index + 1, torch.zeros_like(first_index))
        count = blockers.clamp_max(3)
        source_type = torch.remainder(source_piece - 1, 6) + 1
        orthogonal = torch.arange(8, device=board.device)[None, None, :, None] < 4
        slider = ((source_type == WHITE_ROOK) | (source_type == WHITE_QUEEN)) & orthogonal | ((source_type == WHITE_BISHOP) | (source_type == WHITE_QUEEN)) & ~orthogonal
        king = (source_type == WHITE_KING) & (distance == 1)
        actual_attack = occupied_source & direct & (slider | king)
        edge = torch.cat((
            occupied_source.unsqueeze(-1).float(), direct.unsqueeze(-1).float(), xray.unsqueeze(-1).float(), actual_attack.unsqueeze(-1).float(),
            self.edge_piece(source_piece), self.edge_piece(destination_piece), self.edge_piece(first_piece),
            self.edge_distance(distance), self.edge_blocker_distance(blocker_distance), self.edge_blocker_count(count),
        ), dim=-1)
        return edge, occupied_source

    def _jump_features(self, h: Tensor, board: Tensor, sources: Tensor, destinations: Tensor, relation: str) -> Tensor:
        b = board.size(0)
        source_piece = board.index_select(1, sources)
        dest_piece = board.index_select(1, destinations)
        source_type = torch.remainder(source_piece - 1, 6) + 1
        occupied = source_piece != EMPTY
        if relation == "knight":
            attack = (source_type == WHITE_KNIGHT) & occupied
        else:
            ranks = sources // 8
            direction = (destinations // 8 > ranks)[None]
            attack = ((source_piece == WHITE_PAWN) & direction) | ((source_piece == BLACK_PAWN) & ~direction)
        # Keep exactly the ray edge width, with zeroed unused fields.
        zeros = h.new_zeros(b, sources.numel(), self.edge_dim - 1 - 6 - 6)
        return torch.cat((occupied.unsqueeze(-1).float(), self.edge_piece(source_piece), self.edge_piece(dest_piece), zeros), dim=-1) * attack.unsqueeze(-1)

    def _state_features(self, position: RayGNNPosition) -> Tensor:
        board = position.board
        counts = torch.stack([(board == code).sum(dim=1) for code in range(1, 13)], dim=1).float()
        # Use an 8-file plus none representation, preserving the only file-level
        # information that affects legal en-passant possibilities after board input.
        ep_file = torch.cat((F.one_hot((position.en_passant % 8).clamp(0, 7), 8).float() * (position.en_passant[:, None] < 64), (position.en_passant[:, None] == 64).float()), dim=1)
        phase = (counts[:, 1] + counts[:, 7] + counts[:, 2] + counts[:, 8] + 2 * (counts[:, 3] + counts[:, 9]) + 4 * (counts[:, 4] + counts[:, 10])) / 24.0
        raw = torch.cat((counts / 8.0, position.castling_rights, position.side_to_move[:, None], ep_file, (position.halfmove_clock / 100.0).clamp_max(2)[:, None], (position.repetition_count / 3.0).clamp_max(2)[:, None], phase[:, None]), dim=1)
        assert raw.shape[1] == 28
        return self.state_projection(raw)

    def _readout(self, h: Tensor, position: RayGNNPosition) -> tuple[Tensor, Tensor]:
        board = position.board
        groups = []
        for code in range(1, 13):
            mask = (board == code).float()
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1)
            groups.append(self.piece_group_projection(torch.cat((pooled, mask.sum(dim=1, keepdim=True)), dim=1)))
        piece_readout = torch.cat(groups, dim=1)
        spatial = self.spatial_projection(h).view(h.size(0), 8, 8, 16).view(h.size(0), 4, 2, 4, 2, 16).mean(dim=(2, 4)).reshape(h.size(0), -1)
        king_parts = []
        for king_code in (WHITE_KING, BLACK_KING):
            king_mask = board == king_code
            king_index = king_mask.float().argmax(dim=1)
            king_state = h[torch.arange(h.size(0), device=h.device), king_index]
            neighborhood = []
            for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
                files, ranks = king_index % 8 + dx, king_index // 8 + dy
                valid = (files >= 0) & (files < 8) & (ranks >= 0) & (ranks < 8)
                index = (ranks.clamp(0, 7) * 8 + files.clamp(0, 7)).long()
                neighbor = h[torch.arange(h.size(0), device=h.device), index] * valid[:, None]
                direction = F.one_hot(torch.full((h.size(0),), len(neighborhood), device=h.device), 8).float()
                neighborhood.append(self.king_neighbor_projection(torch.cat((neighbor, direction), dim=1)))
            king_parts.extend((king_state, torch.stack(neighborhood, dim=1).sum(dim=1)))
        king_readout = torch.cat(king_parts, dim=1)
        raw = self.raw_board_projection(F.one_hot(board, num_classes=13).float().flatten(1))
        return torch.cat((piece_readout, spatial, king_readout), dim=1), raw

    def forward(self, position: RayGNNPosition, return_wdl: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        h = self._initial_state(position)
        ray_edge, ray_valid = self._ray_features(position.board)
        knight = self._jump_features(h, position.board, self.knight_sources, self.knight_destinations, "knight")
        pawn = self._jump_features(h, position.board, self.pawn_sources, self.pawn_destinations, "pawn")
        for layer in self.layers:
            h = layer(h, ray_edge, ray_valid, knight, pawn)
        structured, raw = self._readout(h, position)
        state = self._state_features(position)
        head_input = torch.cat((structured, state, raw), dim=1)
        if head_input.shape[1] != 928:
            raise RuntimeError(f"Expected 928-wide RayGNN head input, got {head_input.shape[1]}.")
        correction_features = self.value_head[:-1](head_input)
        correction = self.value_head[-1](correction_features)
        material = _MATERIAL.to(position.board.device)[position.board].sum(dim=1, keepdim=True).float() / 100.0
        value = material + correction
        if return_wdl:
            if self.wdl_head is None:
                raise RuntimeError("This RayGNN was created without an auxiliary WDL head.")
            return value, self.wdl_head(correction_features)
        return value
