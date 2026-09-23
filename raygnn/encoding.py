"""Lossless piece and rule-state inputs, and initial square features."""

from dataclasses import dataclass

import chess
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class PositionBatch:
    # Piece IDs: 0 empty, 1..6 White pawn..king, 7..12 Black pawn..king.
    piece: Tensor
    white_to_move: Tensor
    # White K/Q, Black K/Q.
    castling: Tensor
    # Legal en-passant target in python-chess square order, or -1.
    en_passant: Tensor
    halfmove_clock: Tensor
    repetition: Tensor

    def to(self, device: torch.device | str) -> "PositionBatch":
        return PositionBatch(*(getattr(self, name).to(device) for name in self.__dataclass_fields__))

    def validate(self) -> None:
        batch = self.piece.shape[0]
        if self.piece.ndim != 2 or self.piece.shape[1] != 64:
            raise ValueError("piece must have shape [B, 64]")
        if self.piece.dtype != torch.long or torch.any((self.piece < 0) | (self.piece > 12)):
            raise ValueError("piece IDs must be integers in 0..12")
        if self.castling.shape != (batch, 4):
            raise ValueError("castling must have shape [B, 4]")
        for name in ("white_to_move", "en_passant", "halfmove_clock", "repetition"):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")
        if torch.any((self.en_passant < -1) | (self.en_passant > 63)):
            raise ValueError("en_passant must be -1 or a square in 0..63")
        if torch.any((self.piece == 6).sum(1) != 1) or torch.any((self.piece == 12).sum(1) != 1):
            raise ValueError("each position must contain exactly one king per color")


def boards_to_batch(boards: list[chess.Board], device: torch.device | str = "cpu") -> PositionBatch:
    """Convert legal chess.Board positions to the model's fixed White perspective."""
    if not boards:
        raise ValueError("boards cannot be empty")
    pieces = torch.zeros((len(boards), 64), dtype=torch.long)
    castling = torch.zeros((len(boards), 4), dtype=torch.float32)
    ep = torch.full((len(boards),), -1, dtype=torch.long)
    for row, board in enumerate(boards):
        for square, piece in board.piece_map().items():
            pieces[row, square] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        castling[row] = torch.tensor([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ], dtype=torch.float32)
        if board.ep_square is not None and board.has_legal_en_passant():
            ep[row] = board.ep_square
    batch = PositionBatch(
        pieces,
        torch.tensor([board.turn == chess.WHITE for board in boards], dtype=torch.bool),
        castling,
        ep,
        torch.tensor([board.halfmove_clock for board in boards], dtype=torch.float32),
        torch.tensor([board.is_repetition(2) for board in boards], dtype=torch.float32),
    )
    batch.validate()
    return batch.to(device)


class SquareEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.piece_embedding = nn.Embedding(13, 16)
        self.square_embedding = nn.Embedding(64, 16)
        # Piece + square + coordinates + offsets to both kings + side/castling/EP/clock/repetition.
        self.project = nn.Linear(16 + 16 + 2 + 4 + 1 + 4 + 2 + 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        squares = torch.arange(64)
        self.register_buffer("squares", squares, persistent=False)
        self.register_buffer("coords", torch.stack((squares % 8, squares // 8), -1).float(), persistent=False)

    def forward(self, batch: PositionBatch) -> Tensor:
        piece = batch.piece
        b = piece.shape[0]
        coords = self.coords.unsqueeze(0).expand(b, -1, -1)
        white_king = self.coords[(piece == 6).long().argmax(1)]
        black_king = self.coords[(piece == 12).long().argmax(1)]
        king_offsets = torch.cat((coords - white_king[:, None], coords - black_king[:, None]), -1) / 7
        ep_file = torch.where(batch.en_passant >= 0, batch.en_passant % 8, 0).float() / 7
        ep_rank = torch.where(batch.en_passant >= 0, batch.en_passant // 8, 0).float() / 7
        state = torch.cat((
            batch.white_to_move.float()[:, None], batch.castling.float(),
            ep_file[:, None], ep_rank[:, None],
            (batch.halfmove_clock / 100)[:, None],
            batch.repetition.float()[:, None],
        ), -1)[:, None].expand(-1, 64, -1)
        raw = torch.cat((
            self.piece_embedding(piece),
            self.square_embedding(self.squares)[None].expand(b, -1, -1),
            coords / 7, king_offsets, state,
        ), -1)
        return self.norm(self.project(raw))


def raw_board_one_hot(piece: Tensor) -> Tensor:
    return F.one_hot(piece, 13).flatten(1).float()


def swap_colors_rotate(batch: PositionBatch) -> PositionBatch:
    """Rotate 180 degrees and swap colors, including rule-state fields."""
    piece = batch.piece.flip(1)
    piece = torch.where(piece == 0, 0, torch.where(piece <= 6, piece + 6, piece - 6))
    ep = torch.where(batch.en_passant < 0, -1, 63 - batch.en_passant)
    return PositionBatch(
        piece, ~batch.white_to_move, batch.castling[:, [3, 2, 1, 0]], ep,
        batch.halfmove_clock, batch.repetition,
    )
