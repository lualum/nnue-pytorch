"""RayGNN v3: White-positive, unbounded pawn-unit value."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encoding import PositionBatch, SquareEncoder, StateEncoder
from .geometry import BoardGeometry
from .layers import RayLayer
from .readout import GlobalReadout


@dataclass(frozen=True)
class RayGNNConfig:
    layers: int = 2
    message_width: int = 32
    spatial_width: int = 8
    relation_separation: bool = True
    dense_context: bool = False
    king_relative: bool = False
    pooling_alternative: bool = False
    multiplicative_pair: bool = True
    dense_reference: bool = False
    edge_chunk_size: int | None = None
    draw_state_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if self.layers < 0 or self.message_width < 1 or self.spatial_width < 1:
            raise ValueError("layers and widths must be valid positive dimensions")
        if self.edge_chunk_size is not None and self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")

    @classmethod
    def variant(cls, name: str) -> "RayGNNConfig":
        variants = {
            "reference": {}, "no_message": {"layers": 0},
            "no_relation_separation": {"relation_separation": False},
            "dense_context": {"dense_context": True},
            "king_relative": {"king_relative": True},
            "wider_spatial": {"spatial_width": 16},
            "wider_messages": {"message_width": 48},
            "global_pooling": {"pooling_alternative": True},
            "three_layers": {"layers": 3},
            "simple_pair": {"multiplicative_pair": False},
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
        self.encoder = SquareEncoder(config.king_relative)
        self.edge_encoder = nn.Linear(31, config.message_width) if config.layers else None
        self.layers = nn.ModuleList(RayLayer(config.message_width, config.relation_separation,
                                             config.multiplicative_pair, config.edge_chunk_size,
                                             config.dense_reference) for _ in range(config.layers))
        self.readout = GlobalReadout(config.spatial_width, config.pooling_alternative)

    def forward(self, piece: Tensor | PositionBatch, side_to_move: Tensor | None = None,
                castling: Tensor | None = None, en_passant: Tensor | None = None,
                draw_state: Tensor | None = None) -> Tensor:
        if isinstance(piece, PositionBatch):
            batch = piece
        else:
            if side_to_move is None or castling is None or en_passant is None:
                raise ValueError("side_to_move, castling and en_passant are required")
            batch = PositionBatch(piece, side_to_move, castling, en_passant, draw_state)
        batch.validate(check_values=batch.piece.device.type == "cpu")
        piece = batch.piece
        state = self.state_encoder(batch.side_to_move, batch.castling,
                                   batch.en_passant, batch.draw_state)
        initial = self.encoder(piece)
        h = initial
        if self.layers:
            features, classes = self.geometry.edge_features(piece, self.edge_encoder.weight.dtype)
            if self.config.dense_context:
                classes = torch.where(classes < 0, 2, classes)
            active = classes >= 0
            if self.config.dense_reference:
                edges = self.edge_encoder(features)
            else:
                edges = torch.zeros((*classes.shape, self.config.message_width), device=piece.device, dtype=state.dtype)
                if bool(active.any()):
                    edges[active] = self.edge_encoder(features[active])
            for layer in self.layers:
                h = layer(h, state, edges, self.geometry, classes)
        return self.readout(h, initial, state, piece)
