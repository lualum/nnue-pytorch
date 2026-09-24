"""Second-design RayGNN: White-positive, unbounded pawn-unit value."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoding import PositionBatch, SquareEncoder, StateEncoder, raw_board_one_hot
from .geometry import BoardGeometry
from .layers import RayLayer
from .readout import GlobalReadout


@dataclass(frozen=True)
class RayGNNConfig:
    layers: int = 2
    message_width: int = 32
    readout_width: int = 128
    readout_final_activation: bool = True
    channelwise_gates: bool = False
    king_relative: bool = False
    relation_biases: bool = False
    raw_board_only: bool = False
    float32_reductions: bool = False
    edge_chunk_size: int | None = None
    sparse_board_projection: bool = False
    draw_state_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if self.layers < 0 or self.message_width < 1 or self.readout_width < 1:
            raise ValueError("layers and widths must be valid positive dimensions")
        if self.edge_chunk_size is not None and self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        if self.raw_board_only and self.layers:
            raise ValueError("raw-board-only variant requires layers=0")

    @classmethod
    def variant(cls, name: str) -> "RayGNNConfig":
        variants = {
            "reference": {},
            "original_compact": {"message_width": 16, "readout_width": 64,
                                 "readout_final_activation": False},
            "no_message": {"layers": 0},
            "raw_board_only": {"layers": 0, "raw_board_only": True},
            "channelwise_gates": {"channelwise_gates": True},
            "king_relative": {"king_relative": True},
            "smaller_messages": {"message_width": 16},
            "smaller_readout": {"readout_width": 64},
            "relation_biases": {"relation_biases": True},
            "one_layer": {"layers": 1},
            "three_layers": {"layers": 3},
        }
        if name not in variants:
            raise ValueError(f"unknown variant {name!r}; choose from {tuple(variants)}")
        return cls(**variants[name])


class RayGNN(nn.Module):
    def __init__(self, config: RayGNNConfig | None = None):
        super().__init__()
        self.config = config or RayGNNConfig()
        config = self.config
        self.state_encoder = StateEncoder(config.draw_state_fields)
        self.geometry = BoardGeometry() if config.layers else None
        self.encoder = None if config.raw_board_only else SquareEncoder(config.king_relative)
        self.edge_encoder = nn.Linear(31, config.message_width) if config.layers else None
        self.layers = nn.ModuleList(RayLayer(config.message_width, config.channelwise_gates,
                                             config.relation_biases, config.float32_reductions,
                                             config.edge_chunk_size) for _ in range(config.layers))
        self.readout = None if config.raw_board_only else GlobalReadout(
            config.readout_width, config.readout_final_activation)
        head_width = (0 if config.raw_board_only else config.readout_width) + 832 + 16
        self.head = nn.Sequential(nn.Linear(head_width, 64), nn.SiLU(), nn.Linear(64, 1))

    def _first_head(self, global_h: Tensor | None, piece: Tensor, state: Tensor) -> Tensor:
        parts = ([global_h] if global_h is not None else []) + [state]
        if not self.config.sparse_board_projection:
            head_parts = ([global_h] if global_h is not None else []) + [raw_board_one_hot(piece).to(state.dtype), state]
            return self.head[0](torch.cat(head_parts, -1))
        # Exactly the same 832 first-layer weights, selected once per square.
        first = self.head[0]
        global_width = 0 if global_h is None else global_h.shape[-1]
        nonboard_weight = torch.cat((first.weight[:, :global_width], first.weight[:, global_width + 832:]), -1)
        base = F.linear(torch.cat(parts, -1), nonboard_weight, first.bias)
        columns = 13 * torch.arange(64, device=piece.device)[None] + piece
        board_weights = first.weight[:, global_width:global_width + 832].transpose(0, 1)
        return base + board_weights[columns].sum(1)

    def forward(self, piece: Tensor | PositionBatch, side_to_move: Tensor | None = None,
                castling: Tensor | None = None, en_passant: Tensor | None = None,
                draw_state: Tensor | None = None) -> Tensor:
        if isinstance(piece, PositionBatch):
            batch = piece
        else:
            if side_to_move is None or castling is None or en_passant is None:
                raise ValueError("side_to_move, castling and en_passant are required")
            batch = PositionBatch(piece, side_to_move, castling, en_passant, draw_state)
        # Value checks on CUDA tensors synchronize the device on every forward.
        # Inputs constructed on CPU are checked before transfer.
        batch.validate(check_values=batch.piece.device.type == "cpu")
        piece = batch.piece
        state = self.state_encoder(batch.side_to_move, batch.castling,
                                   batch.en_passant, batch.draw_state)
        global_h = None
        if self.encoder is not None:
            initial = self.encoder(piece)
            h = initial
            if self.layers:
                features, classes = self.geometry.edge_features(piece, self.edge_encoder.weight.dtype)
                edges = self.edge_encoder(features)
                for layer in self.layers:
                    h = layer(h, state, edges, self.geometry, classes)
            global_h = self.readout(h, initial)
        return self.head[2](self.head[1](self._first_head(global_h, piece, state)))
