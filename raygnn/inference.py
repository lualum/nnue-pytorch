"""Engine boundary for White-positive and side-to-move centipawn scores."""

from pathlib import Path

import chess
import torch

from .encoding import boards_to_batch
from .model import RayGNN, RayGNNConfig


class RayGNNEvaluator:
    def __init__(self, model: RayGNN, device: str | torch.device = "cpu"):
        self.device = device
        self.model = model.to(device).eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "RayGNNEvaluator":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = RayGNN(RayGNNConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        return cls(model, device)

    @torch.inference_mode()
    def evaluate_cp(self, boards: list[chess.Board], side_to_move: bool = False,
                    max_cp: int | None = None) -> torch.Tensor:
        """Rounded centipawns; caller handles terminal/draw rules and mate range."""
        batch = boards_to_batch(boards, self.device)
        value = self.model(batch)[:, 0]
        if side_to_move:
            value = value * batch.side_to_move[:, 0]
        score = torch.round(100 * value).to(torch.long)
        if max_cp is not None:
            if max_cp < 0:
                raise ValueError("max_cp must be nonnegative")
            score = score.clamp(-max_cp, max_cp)
        return score.cpu()
