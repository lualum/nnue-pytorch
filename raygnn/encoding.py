"""Fixed White-oriented board and rule-state encoding for RayGNN."""

from dataclasses import dataclass

import chess
import torch
from torch import Tensor, nn
from torch.nn import functional as F

DRAW_STATE_FIELDS = ("halfmove_clock_div_100", "repetition_twofold")


@dataclass
class PositionBatch:
    piece: Tensor
    side_to_move: Tensor
    castling: Tensor
    en_passant: Tensor
    draw_state: Tensor | None = None

    def to(self, device: torch.device | str) -> "PositionBatch":
        return PositionBatch(*(value.to(device) if value is not None else None
                               for value in (self.piece, self.side_to_move, self.castling,
                                             self.en_passant, self.draw_state)))

    def validate(self, check_values: bool = True) -> None:
        if self.piece.ndim != 2 or self.piece.shape[1] != 64:
            raise ValueError("piece must have shape [B,64]")
        batch = self.piece.shape[0]
        if self.piece.dtype != torch.long or (check_values and bool(((self.piece < 0) | (self.piece > 12)).any())):
            raise ValueError("piece IDs must be long integers in 0..12")
        if self.side_to_move.shape != (batch, 1) or (check_values and bool((self.side_to_move.abs() != 1).any())):
            raise ValueError("side_to_move must have shape [B,1] with values +1 or -1")
        if self.castling.shape != (batch, 4) or (check_values and bool(((self.castling != 0) & (self.castling != 1)).any())):
            raise ValueError("castling must have shape [B,4] with binary values")
        if (self.en_passant.shape != (batch,) or self.en_passant.dtype != torch.long or
                (check_values and bool(((self.en_passant < 0) | (self.en_passant > 64)).any()))):
            raise ValueError("en_passant must contain square indices 0..63 or 64 for none")
        if self.draw_state is not None and (self.draw_state.ndim != 2 or self.draw_state.shape[0] != batch):
            raise ValueError("draw_state must have shape [B,D]")


def boards_to_batch(boards: list[chess.Board], device: torch.device | str = "cpu") -> PositionBatch:
    """Keep the FEN en-passant target even when no legal capture exists."""
    if not boards:
        raise ValueError("boards cannot be empty")
    pieces = torch.zeros((len(boards), 64), dtype=torch.long)
    castling = torch.zeros((len(boards), 4), dtype=torch.float32)
    ep = torch.full((len(boards),), 64, dtype=torch.long)
    for row, board in enumerate(boards):
        for square, piece in board.piece_map().items():
            pieces[row, square] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        castling[row] = torch.tensor([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ], dtype=torch.float32)
        if board.ep_square is not None:
            ep[row] = board.ep_square
    batch = PositionBatch(
        pieces,
        torch.tensor([[1 if board.turn == chess.WHITE else -1] for board in boards], dtype=torch.float32),
        castling,
        ep,
        torch.tensor([[board.halfmove_clock / 100, float(board.is_repetition(2))]
                      for board in boards], dtype=torch.float32),
    )
    batch.validate()
    return batch.to(device)


_FEN_PIECES = {symbol: piece + (0 if symbol.isupper() else 6)
               for piece, symbols in enumerate(("", "pP", "nN", "bB", "rR", "qQ", "kK"))
               for symbol in symbols}


def fens_to_batch(fens: list[str], device: torch.device | str = "cpu") -> PositionBatch:
    """Encode FEN fields directly for scored-position training without chess.Board objects."""
    if not fens:
        raise ValueError("fens cannot be empty")
    pieces, turns, castling, en_passant = [], [], [], []
    for fen in fens:
        fields = fen.split()
        if len(fields) < 4:
            raise ValueError(f"incomplete FEN: {fen!r}")
        ranks = fields[0].split("/")
        if len(ranks) != 8:
            raise ValueError(f"invalid FEN board: {fen!r}")
        row = [0] * 64
        for rank_index, rank in enumerate(ranks):
            file_index = 0
            for symbol in rank:
                if symbol in "12345678":
                    file_index += int(symbol)
                else:
                    if symbol not in _FEN_PIECES or file_index >= 8:
                        raise ValueError(f"invalid FEN piece placement: {fen!r}")
                    row[(7 - rank_index) * 8 + file_index] = _FEN_PIECES[symbol]
                    file_index += 1
            if file_index != 8:
                raise ValueError(f"invalid FEN rank width: {fen!r}")
        if fields[1] not in ("w", "b"):
            raise ValueError(f"invalid FEN side to move: {fen!r}")
        if fields[2] != "-" and (set(fields[2]) - set("KQkq")):
            raise ValueError(f"invalid FEN castling rights: {fen!r}")
        ep = fields[3]
        if ep == "-":
            ep_square = 64
        elif len(ep) == 2 and ep[0] in "abcdefgh" and ep[1] in "12345678":
            ep_square = (int(ep[1]) - 1) * 8 + ord(ep[0]) - ord("a")
        else:
            raise ValueError(f"invalid FEN en-passant square: {fen!r}")
        pieces.append(row)
        turns.append([1.0 if fields[1] == "w" else -1.0])
        castling.append([float(right in fields[2]) for right in "KQkq"])
        en_passant.append(ep_square)
    batch = PositionBatch(
        torch.tensor(pieces, dtype=torch.long),
        torch.tensor(turns, dtype=torch.float32),
        torch.tensor(castling, dtype=torch.float32),
        torch.tensor(en_passant, dtype=torch.long),
    )
    return batch.to(device)


class StateEncoder(nn.Module):
    def __init__(self, draw_state_fields: tuple[str, ...] = ()):
        super().__init__()
        if any(field not in DRAW_STATE_FIELDS for field in draw_state_fields) or len(set(draw_state_fields)) != len(draw_state_fields):
            raise ValueError(f"draw_state_fields must be distinct members of {DRAW_STATE_FIELDS}")
        self.draw_state_fields = draw_state_fields
        self.project = nn.Linear(70 + len(draw_state_fields), 16)

    def forward(self, side_to_move: Tensor, castling: Tensor, en_passant: Tensor,
                draw_state: Tensor | None = None) -> Tensor:
        ep = F.one_hot(en_passant, 65).to(dtype=self.project.weight.dtype)
        values = [side_to_move.to(ep.dtype), castling.to(ep.dtype), ep]
        if self.draw_state_fields:
            if draw_state is None or draw_state.shape != (side_to_move.shape[0], len(DRAW_STATE_FIELDS)):
                raise ValueError("draw_state must contain halfmove_clock_div_100 and repetition_twofold")
            values.append(draw_state[:, [DRAW_STATE_FIELDS.index(field) for field in self.draw_state_fields]].to(ep.dtype))
        return self.project(torch.cat(values, -1).to(self.project.weight.dtype))


class SquareEncoder(nn.Module):
    def __init__(self, king_relative: bool = False):
        super().__init__()
        self.king_relative = king_relative
        self.piece_embedding = nn.Embedding(13, 96)
        self.joint_embedding = nn.Embedding(13 * 64, 96)
        nn.init.normal_(self.piece_embedding.weight, 0, 0.05)
        nn.init.normal_(self.joint_embedding.weight, 0, 0.05)
        if king_relative:
            self.king_project = nn.Linear(4, 96, bias=False)
        self.register_buffer("squares", torch.arange(64))

    def forward(self, piece: Tensor) -> Tensor:
        result = self.piece_embedding(piece) + self.joint_embedding(piece * 64 + self.squares[None])
        if self.king_relative:
            if bool(((piece == 6).sum(1) != 1).any()) or bool(((piece == 12).sum(1) != 1).any()):
                raise ValueError("king-relative encoding requires exactly one king of each color")
            white = (piece == 6).long().argmax(1)
            black = (piece == 12).long().argmax(1)
            coords = torch.stack((self.squares % 8, self.squares // 8), -1).to(result.dtype)
            offsets = torch.cat((coords[None] - coords[white, None],
                                 coords[None] - coords[black, None]), -1) / 7
            result = result + self.king_project(offsets)
        return result


def raw_board_one_hot(piece: Tensor) -> Tensor:
    return F.one_hot(piece, 13).flatten(1).float()


def swap_colors_reflect_ranks(batch: PositionBatch) -> PositionBatch:
    """Reflect ranks, swap colors and rule state; applying twice is identity."""
    piece = batch.piece.reshape(-1, 8, 8).flip(1).reshape(-1, 64)
    piece = torch.where(piece == 0, 0, torch.where(piece <= 6, piece + 6, piece - 6))
    ep = torch.where(batch.en_passant == 64, 64, batch.en_passant ^ 56)
    return PositionBatch(piece, -batch.side_to_move, batch.castling[:, [2, 3, 0, 1]], ep,
                         batch.draw_state)
