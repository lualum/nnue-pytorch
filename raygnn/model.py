"""RayGNN reference architecture with a White-positive pawn-unit value."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encoding import (
    PositionBatch,
    SquareEncoder,
    raw_board_one_hot,
    swap_colors_rotate,
)
from .geometry import BoardGeometry
from .layers import RayLayer
from .readout import StructuredReadout


@dataclass(frozen=True)
class RayGNNConfig:
    d_model: int = 64
    d_msg: int = 16
    layers: int = 3
    use_xray: bool = True
    first_piece_only: bool = False
    structured_readout: bool = True
    raw_board_skip: bool = True
    wdl_head: bool = False
    enforce_color_symmetry: bool = True


@dataclass
class Evaluation:
    value: Tensor
    material: Tensor
    correction: Tensor
    wdl_logits: Tensor | None = None


class RayGNN(nn.Module):
    def __init__(self, config: RayGNNConfig | None = None):
        super().__init__()
        config = config or RayGNNConfig()
        self.config = config
        self.geometry = BoardGeometry()
        self.encoder = SquareEncoder(config.d_model)
        self.layers = nn.ModuleList(RayLayer(config.d_model, config.d_msg) for _ in range(config.layers))
        self.readout = StructuredReadout(config.d_model) if config.structured_readout else None
        self.state_project = nn.Sequential(nn.Linear(23, 32), nn.SiLU(), nn.Linear(32, 32))
        self.raw_project = nn.Linear(64 * 13, 64) if config.raw_board_skip else None
        readout_width = self.readout.output_dim if self.readout is not None else config.d_model
        head_width = readout_width + 32 + (64 if config.raw_board_skip else 0)
        self.head = nn.Sequential(nn.Linear(head_width, 256), nn.SiLU(), nn.Linear(256, 64), nn.SiLU())
        self.value_head = nn.Linear(64, 1)
        self.wdl_head = nn.Linear(64, 3) if config.wdl_head else None
        self.register_buffer("material_weights", torch.tensor([100, 320, 330, 500, 900, 0], dtype=torch.float32), persistent=False)

    def _forward_once(self, batch: PositionBatch) -> Evaluation:
        piece = batch.piece
        h = self.encoder(batch)
        if self.layers:
            ray = self.geometry.rays(piece)
            for layer in self.layers:
                h = layer(h, piece, self.geometry, ray, self.config.use_xray, self.config.first_piece_only)
        global_h = self.readout(h, piece, self.geometry) if self.readout is not None else h.mean(1)
        counts = torch.stack([(piece == index).sum(1) for index in range(1, 13)], -1).float()
        material = ((counts[:, :6] - counts[:, 6:]) * self.material_weights).sum(-1, keepdim=True) / 100
        phase = (counts[:, [1, 2, 3, 4, 7, 8, 9, 10]] * torch.tensor([1, 1, 2, 4, 1, 1, 2, 4], device=piece.device)).sum(-1, keepdim=True) / 24
        ep = batch.en_passant
        state = torch.cat((
            counts / 8, batch.castling.float(), batch.white_to_move.float()[:, None], phase,
            (ep >= 0).float()[:, None], torch.where(ep >= 0, ep % 8, 0).float()[:, None] / 7,
            torch.where(ep >= 0, ep // 8, 0).float()[:, None] / 7,
            (batch.halfmove_clock / 100)[:, None], batch.repetition.float()[:, None],
        ), -1)
        head_input = [global_h, self.state_project(state)]
        if self.raw_project is not None:
            head_input.append(self.raw_project(raw_board_one_hot(piece)))
        hidden = self.head(torch.cat(head_input, -1))
        correction = self.value_head(hidden)
        return Evaluation(material + correction, material, correction, self.wdl_head(hidden) if self.wdl_head is not None else None)

    def forward(self, batch: PositionBatch) -> Evaluation:
        batch.validate()
        original = self._forward_once(batch)
        if not self.config.enforce_color_symmetry:
            return original
        swapped = self._forward_once(swap_colors_rotate(batch))
        correction = (original.correction - swapped.correction) / 2
        wdl = None
        if original.wdl_logits is not None:
            wdl = (original.wdl_logits + swapped.wdl_logits[:, [2, 1, 0]]) / 2
        return Evaluation(original.material + correction, original.material, correction, wdl)
